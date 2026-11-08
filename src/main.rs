// Pixel and coordinate math lives in f32 (window and source units) and
// indices in usize and u32.
// The casts between them belong to that boundary.
// Every value here (window pixels, texture dimensions) stays far below
// the f32 exact-integer range.
// The pedantic cast lints are therefore noise.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    // win_w/win_h and friends are natural paired names in window math.
    clippy::similar_names
)]

//! versatile-viewer is an image viewer with JXL as a first-class format plus
//! PNG and JPEG.
//! It shows a directory thumbnail grid.
//! q quits.
//! ESC and Enter toggle between the grid and the image view.
//!
//! Usage: `vv [--mcp] [--wid <id>] <image-path-or-directory>`
//!
//! `VV_DEBUG`=1 traces input events.
//! `VV_PRESENT` picks the swapchain present mode.
//! `VV_TIMING`=1 prints the GPU stack and every slow frame.
//! `VV_SLOW_STREAM`=1 slows the JXL stream.
//! `VV_BLUR_BG`=1 draws a blurred copy of the viewed image as the image-view
//! background, scaled to cover the window and upscaled by the GPU.
//! `VV_BG_DIM`=0..1 sets its brightness (default 0.6).
//! `VV_BLUR_PX` sets the blur resolution, the long side of the tiny texture.
//! The default is 128, and fewer pixels give a blurrier result.
//!
//! `vv --mcp <path>` also serves the Model Context Protocol on stdio.
//! The viewer window stays interactive; the server owns stdin and stdout.
//!
//! `vv --wid <id> <path>` draws the image view inside a host window instead of
//! a top-level window of its own (see `Pane`).

use std::{
    env,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use versatile_viewer::{
    draw::Vec2,
    gpu::{FrameError, Gpu, GpuSettings, Present},
    input, mcp,
    pane::{PaneGeometry, Wid},
    viewer::{Event, Host, Launch, Viewer},
};
#[cfg(target_os = "linux")]
use winit::platform::x11::WindowAttributesExtX11 as _;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::{CursorGrabMode, Window, WindowId},
};

/// How long a frame sleeps when the swapchain has no image for the window.
///
/// A hidden or occluded window still receives `RedrawRequested`, because the
/// loop asks for one after every frame. Fifo paces a visible window, and a
/// skipped frame has nothing to pace, so without this pause the loop would spin
/// on acquire errors and burn a core for a window nobody sees.
/// Twenty wakeups a second keep the return of the window immediate.
const OCCLUDED_PAUSE: Duration = Duration::from_millis(50);

/// What kind of window this process draws into.
///
/// A host application that wants the picture inside a window of its own passes
/// that window with `--wid`, and the viewer then shows the image view inside it.
/// Every policy that differs between the two kinds reads this one value, so no
/// view code carries an `if embedded` test of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pane {
    /// A top-level window of our own: grid, title and `WM_CLASS`.
    Window,
    /// A child window inside a host: image view only, and the host owns the
    /// size and the keyboard focus.
    Embedded(Wid),
}

impl Pane {
    /// May the view fall back to the grid?
    ///
    /// A standalone window shows the grid, so `ESC` and `Enter` toggle between
    /// the two views and a failed open leaves that grid on screen.
    /// A host pane draws the image view only, so there is nothing to fall back
    /// to: `ESC` and `Enter` never leave the picture, and a failed open ends the
    /// process with the reason.
    const fn shows_grid(self) -> bool {
        matches!(self, Self::Window)
    }
}
/// Start the MCP stdio server once, in the background.
///
/// The server owns stdin and stdout, and every diagnostic of the viewer goes to
/// stderr, so no library writes a protocol frame by accident.
///
/// The whole program is single-threaded at this point.
fn start_mcp() -> Result<()> {
    mcp::spawn().context("failed to start the MCP server")
}

/// The `winit` window the binary draws into.
///
/// The viewer holds no window: it asks for four window actions through
/// [`Host`], and this is this process's answer to them. An embedding host
/// answers from its own toolkit.
struct WinitHost<'a> {
    window: &'a Window,
}

impl Host for WinitHost<'_> {
    fn set_title(&mut self, title: &str) {
        self.window.set_title(title);
    }

    fn grab_pointer(&mut self) {
        // `Locked` is the mode with no pointer motion at all, and X11 refuses
        // it, so `Confined` is the fallback. The raw device deltas arrive
        // either way, and the input layer makes them the pointer's motion.
        for mode in [CursorGrabMode::Locked, CursorGrabMode::Confined] {
            if self.window.set_cursor_grab(mode).is_ok() {
                break;
            }
        }
        self.window.set_cursor_visible(false);
    }

    fn release_pointer(&mut self) {
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
    }

    fn place_pointer(&mut self, at: Vec2) {
        let position = winit::dpi::PhysicalPosition::new(at.x.round(), at.y.round());
        let _ = self.window.set_cursor_position(position);
    }
}

/// The viewer process: the window, the swapchain and the pane it draws into.
///
/// Everything past the window and the device is a [`Viewer`], so the binary
/// owns the platform and the library owns the picture.
// The platform flags are independent by nature: the swapchain arrived, the
// window is occluded, and two traces are on. Each one drives a different
// decision, so one enum would only blur them.
#[allow(clippy::struct_excessive_bools)]
struct App {
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    viewer: Option<Viewer>,
    /// What kind of window this process draws into.
    pane: Pane,
    /// The host window an embedded pane fills, so the pane follows its size.
    ///
    /// None for a top-level window, and None for a pane whose display could not
    /// be reached, which then keeps the size it launched with.
    pane_geometry: Option<PaneGeometry>,
    /// The launch, held until the window and the device exist.
    launch: Option<Launch>,
    /// The window size of this frame, in physical pixels.
    size: (u32, u32),
    /// The title the window starts with, before the open picture names it.
    launch_title: String,
    /// The swapchain arrived and the viewer drew its first frame.
    started: bool,
    /// When the previous frame started, for the frame time.
    last_frame: Instant,
    /// A failure that stopped the loop, returned from `main`.
    fatal: Option<anyhow::Error>,
    /// The swapchain answered no image on the last frame, so the window is
    /// hidden or occluded and the loop paces itself.
    occluded: bool,
    /// `VV_DEBUG=1`: print the window scale factor at the start.
    debug: bool,
    /// `VV_TIMING=1`: trace the GPU stack at startup.
    timing: bool,
}

impl App {
    /// Hold the launch result until the window exists.
    fn new(launch: Launch, pane: Pane) -> Self {
        let (size, launch_title) = (launch.window_size(), launch.title().to_string());
        Self {
            window: None,
            gpu: None,
            viewer: None,
            pane,
            pane_geometry: None,
            launch: Some(launch),
            size,
            launch_title,
            started: false,
            last_frame: Instant::now(),
            fatal: None,
            occluded: false,
            debug: std::env::var_os("VV_DEBUG").is_some(),
            timing: versatile_viewer::timing_enabled(),
        }
    }

    /// Hand one input event to the viewer.
    fn send(&mut self, event: Event) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let Some(viewer) = self.viewer.as_mut() else {
            return;
        };
        let mut host = WinitHost { window: &window };
        viewer.handle_event(event, &mut host);
    }

    /// Run one frame: take the swapchain image, let the viewer paint it, and
    /// present it.
    ///
    /// The acquire and the present belong to the host that owns the surface,
    /// and the batch belongs to the viewer, which paints into any texture view.
    fn frame(&mut self) -> Result<()> {
        let Some(window) = self.window.clone() else {
            return Ok(());
        };
        // An embedded pane follows the size of the window it fills, before the
        // frame reads the window size.
        follow_pane(&mut self.pane_geometry, &window);
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        let size = self.size;
        let (Some(gpu), Some(viewer)) = (self.gpu.as_mut(), self.viewer.as_mut()) else {
            return Ok(());
        };
        // A dead device cannot draw again, and every later submit would fail
        // silently. Stop with the reason the driver gave instead of showing a
        // frozen picture.
        if let Some(reason) = gpu.lost_reason() {
            bail!("the GPU device was lost: {reason}");
        }
        let mut host = WinitHost { window: &window };
        match gpu.acquire() {
            Ok(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let result = viewer.render(&mut host, &view, size, dt);
                // wgpu 30 hands the present to the queue, so the frame value
                // travels here instead of presenting itself.
                gpu.queue().present(frame);
                self.occluded = false;
                result
            }
            Err(err) => {
                // A stale or lost swapchain asks for a new configuration and
                // skips the frame, so a resize never kills the window.
                if err.needs_reconfigure() {
                    gpu.reconfigure();
                } else if matches!(err, FrameError::Occluded | FrameError::Timeout) {
                    // No image came out of the swapchain, so the window is
                    // hidden: the loop paces itself instead of spinning.
                    self.occluded = true;
                } else {
                    eprintln!("vv: frame: {err}");
                }
                Ok(())
            }
        }
    }
}

/// Ask for the size of the host window an embedded pane fills.
///
/// The host resizes the container it created and never the child this process
/// draws into, so X sends the `ConfigureNotify` to the container alone. The pane
/// therefore reads the parent geometry itself and asks for the same size, the
/// way `mpv --wid` does. The swapchain reconfigures and the picture refits on
/// the `Resized` that comes back, so this frame still draws at the size it
/// already had.
fn follow_pane(slot: &mut Option<PaneGeometry>, window: &Window) {
    let Some(geometry) = slot.as_mut() else {
        return;
    };
    match geometry.poll() {
        Ok(Some((width, height))) => {
            let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(width, height));
        }
        Ok(None) => {}
        Err(err) => {
            // A host window that went away ends the poll instead of printing on
            // every frame, and the pane keeps its size.
            eprintln!("vv: {err}");
            *slot = None;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.started {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            return;
        }
        let Some(launch) = self.launch.take() else {
            return;
        };
        // A host window is the parent of the child this process draws into, so
        // `XCreateWindow` runs with that parent and the child never becomes a
        // top-level window: no window manager frames it, and the host decides
        // where it sits.
        // A child window carries no decoration to name and no `WM_CLASS` to
        // read, so neither is set.
        let (win0, win1) = launch.window_size();
        let mut attributes = Window::default_attributes()
            .with_inner_size(winit::dpi::PhysicalSize::new(win0, win1))
            .with_resizable(true);
        attributes = match self.pane {
            Pane::Window => attributes
                .with_title(self.launch_title.clone())
                // WM_CLASS on X11, which replaces `src/wmclass.rs`: the class
                // and the instance name are both `vv`.
                .with_name("vv", "vv"),
            Pane::Embedded(wid) => attributes.with_embed_parent_window(wid.raw()),
        };
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                let reason = match self.pane {
                    Pane::Window => format!("cannot create the window: {err}"),
                    Pane::Embedded(_) => {
                        format!("cannot create the embedded child window: {err}")
                    }
                };
                self.fatal = Some(anyhow::anyhow!(reason));
                event_loop.exit();
                return;
            }
        };
        let settings = GpuSettings {
            present: Present::from_env(),
            latency: versatile_viewer::gpu::DEFAULT_LATENCY,
            trace: self.debug || self.timing,
        };
        let gpu = match Gpu::new(&window, settings) {
            Ok(gpu) => gpu,
            Err(err) => {
                eprintln!("vv: {err}");
                self.fatal = Some(anyhow::anyhow!("{err}"));
                event_loop.exit();
                return;
            }
        };
        self.size = window.inner_size().into();
        let viewer = Viewer::new(
            gpu.device().clone(),
            gpu.queue().clone(),
            gpu.format(),
            gpu.max_texture_side(),
            launch,
            self.size,
        );
        self.window = Some(Arc::clone(&window));
        self.gpu = Some(gpu);
        self.viewer = Some(viewer);
        // A host pane follows the container it fills, because the host resizes
        // that container and never the child.
        // A display that cannot be reached leaves the pane at the size it
        // launched with instead of failing the run.
        self.pane_geometry = match self.pane {
            Pane::Window => None,
            Pane::Embedded(wid) => match PaneGeometry::new(wid) {
                Ok(geometry) => Some(geometry),
                Err(err) => {
                    eprintln!("vv: {err}");
                    None
                }
            },
        };
        if self.debug {
            // The content scale is the one display fact a drag can read, and a
            // feel report needs it next to the travel the release prints.
            eprintln!("vv: window scale factor {}", window.scale_factor());
        }
        self.started = true;
        self.last_frame = Instant::now();
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested if !self.started => {}
            WindowEvent::RedrawRequested => match self.frame() {
                Ok(()) => {
                    if self.viewer.as_ref().is_some_and(Viewer::quit) {
                        event_loop.exit();
                    } else if let Some(window) = &self.window {
                        // A hidden window skips every frame, so the loop sleeps
                        // instead of spinning on a picture nobody sees.
                        if self.occluded {
                            std::thread::sleep(OCCLUDED_PAUSE);
                        }
                        // A continuous loop, like the ramps and the eases need:
                        // a held key or a running ease draws the next frame.
                        window.request_redraw();
                    }
                }
                Err(err) => {
                    self.fatal = Some(err);
                    event_loop.exit();
                }
            },
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                }
                if let Some(window) = &self.window {
                    self.size = window.inner_size().into();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(window) = &self.window {
                    self.size = window.inner_size().into();
                    if let Some(gpu) = self.gpu.as_mut() {
                        gpu.resize(self.size.0, self.size.1);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event,
                is_synthetic,
                ..
            } => {
                let PhysicalKey::Code(key) = event.physical_key else {
                    return;
                };
                let text = event.text.as_ref().and_then(|text| text.chars().next());
                self.send(Event::Key {
                    key,
                    state: event.state,
                    repeat: event.repeat,
                    synthetic: is_synthetic,
                    text,
                });
            }
            WindowEvent::MouseWheel { delta, .. } => self.send(Event::Wheel(input::notches(delta))),
            WindowEvent::CursorMoved { position, .. } => self.send(Event::Pointer(Vec2::new(
                position.x as f32,
                position.y as f32,
            ))),
            WindowEvent::MouseInput { state, button, .. } => {
                self.send(Event::Button { button, state });
            }
            WindowEvent::Focused(focused) => self.send(Event::Focus(focused)),
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            // The raw travel of the device: no pointer acceleration and no
            // desktop scale, which is the source the view reads for a drag.
            self.send(Event::Travel(Vec2::new(delta.0 as f32, delta.1 as f32)));
        }
    }
}

/// The command line of one run.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Cli {
    /// `--mcp`: serve the Model Context Protocol on stdio.
    mcp: bool,
    /// `--wid`: the window a host handed over, so the picture draws inside it.
    wid: Option<Wid>,
    /// The image file or the directory to show.
    path: PathBuf,
}

/// The shape of the command line, printed beside a usage error.
const USAGE: &str = "usage: vv [--mcp] [--wid <id>] <image-path-or-directory>";

/// Read the command line.
///
/// `--mcp` and `--wid` may come before or after the path.
/// `--wid` takes its value apart (`--wid 74`) or joined (`--wid=74`), because a
/// wrapper script and a `.desktop` file usually write the joined form.
/// Every other argument is the path, and a second path is an error: the argv
/// loop this replaces kept the first one and silently dropped the rest.
fn parse_args<I: Iterator<Item = String>>(args: I) -> Result<Cli, String> {
    let mut mcp = false;
    let mut wid = None;
    let mut path: Option<PathBuf> = None;
    let mut args = args;
    while let Some(arg) = args.next() {
        if arg == "--mcp" {
            mcp = true;
        } else if arg == "--wid" {
            let Some(value) = args.next() else {
                return Err("--wid needs a window id".to_string());
            };
            wid = Some(parse_wid(&value)?);
        } else if let Some(value) = arg.strip_prefix("--wid=") {
            wid = Some(parse_wid(value)?);
        } else if arg.starts_with("--") {
            return Err(format!("unknown option {arg}"));
        } else if path.is_some() {
            return Err(format!("a second path: {arg}"));
        } else {
            path = Some(PathBuf::from(arg));
        }
    }
    match path {
        Some(path) => Ok(Cli { mcp, wid, path }),
        None => Err("no image path given".to_string()),
    }
}

/// Read the `--wid` value, naming the flag when it is not a window id.
fn parse_wid(value: &str) -> Result<Wid, String> {
    Wid::parse(value).ok_or_else(|| format!("--wid needs a window id, not {value:?}"))
}

fn main() -> Result<()> {
    let cli = match parse_args(env::args().skip(1)) {
        Ok(cli) => cli,
        Err(err) => bail!("{err}\n{USAGE}"),
    };
    if cli.mcp {
        start_mcp()?;
    }
    // A host that passes `--wid` shows the picture inside its own window, and
    // every policy that differs between the two kinds reads this one value.
    let pane = match cli.wid {
        Some(wid) => Pane::Embedded(wid),
        None => Pane::Window,
    };
    let launch = Launch::open(&cli.path, pane.shows_grid())?;
    let event_loop = EventLoop::new().context("cannot create the event loop")?;
    let mut app = App::new(launch, pane);
    event_loop
        .run_app(&mut app)
        .context("the event loop failed")?;
    match app.fatal.take() {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the asserted values are exact window arithmetic
mod tests {
    use super::*;

    /// The parsed command line of an argv list.
    fn parsed(args: &[&str]) -> Result<Cli, String> {
        parse_args(args.iter().map(|arg| (*arg).to_string()))
    }

    /// The path of a parsed command line, for a test that cares about it.
    fn parsed_path(args: &[&str]) -> PathBuf {
        parsed(args).expect("the argv parses").path
    }

    #[test]
    fn the_window_id_reads_before_or_after_the_path() {
        let before = parsed(&["--wid", "0x4a00007", "pic.jpg"]).expect("parses");
        assert_eq!(before.wid, Wid::parse("0x4a00007"));
        assert_eq!(before.path, PathBuf::from("pic.jpg"));
        assert!(!before.mcp);
        // The joined form, which a wrapper script writes.
        let joined = parsed(&["--wid=74", "pic.jpg"]).expect("parses");
        assert_eq!(joined.wid, Wid::parse("74"));
        // The path first, the flag after.
        let after = parsed(&["pic.jpg", "--wid", "74"]).expect("parses");
        assert_eq!(after.wid, Wid::parse("74"));
        // No window: a top-level launch keeps no id.
        assert_eq!(parsed_path(&["pic.jpg"]), PathBuf::from("pic.jpg"));
        assert_eq!(parsed(&["pic.jpg"]).expect("parses").wid, None);
    }

    #[test]
    fn the_mcp_flag_reads_before_or_after_the_path() {
        for args in [["--mcp", "pics"], ["pics", "--mcp"]] {
            let cli = parsed(&args).expect("parses");
            assert!(cli.mcp, "{args:?}");
            assert_eq!(cli.path, PathBuf::from("pics"));
        }
        // Both flags, so an agent path and a display path may run at once.
        let cli = parsed(&["--mcp", "--wid", "0x4a00007", "pics"]).expect("parses");
        assert!(cli.mcp);
        assert_eq!(cli.wid, Wid::parse("0x4a00007"));
        assert_eq!(cli.path, PathBuf::from("pics"));
    }

    #[test]
    fn a_bad_window_id_names_the_flag() {
        for args in [
            vec!["--wid"],
            vec!["--wid", "zz", "pic.jpg"],
            vec!["--wid", "0", "pic.jpg"],
            vec!["--wid=", "pic.jpg"],
        ] {
            let err = parsed(&args).expect_err("no window id");
            assert!(err.contains("--wid"), "{args:?}: {err}");
        }
    }

    #[test]
    fn a_second_path_or_an_unknown_option_is_an_error() {
        assert!(parsed(&["a.jpg", "b.jpg"]).is_err());
        assert!(parsed(&[]).is_err());
        let err = parsed(&["--nope", "pic.jpg"]).expect_err("unknown option");
        assert!(err.contains("--nope"), "{err}");
    }

    #[test]
    fn a_host_pane_falls_back_to_no_grid() {
        // A standalone window toggles between the grid and the image view, and
        // its failed open leaves the grid on screen.
        assert!(Pane::Window.shows_grid());
        // A child window inside a host draws the image view only.
        let wid = Wid::parse("0x4a00007").expect("a window id");
        assert!(!Pane::Embedded(wid).shows_grid());
    }
}
