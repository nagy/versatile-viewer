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
//! Usage: `vv [--mcp] <image-path-or-directory>`
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
    borrow::Cow,
    env, fs,
    path::{Path, PathBuf},
    sync::{Arc, mpsc::channel},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use versatile_viewer::{
    DecodedImage, TAP_SLACK,
    blurbg::{self, BlurBg},
    book::BookSurface,
    dim::{WinH, WinW},
    draw::{Color, Vec2},
    gpu::{FrameError, Gpu, GpuSettings, Present},
    grid::{DecodeFailure, Grid, GridAction, LoadPlan},
    input::Input,
    loader::{DecodeMsg, Loader},
    mcp,
    pane::{PaneGeometry, Wid},
    region_sel,
    render::Renderer,
    rotate,
    surface::{Size, Surface, View, open_surface},
    texture::Filter,
    view::{
        BACKGROUND, EASE_TIME, Mode, NavKeys, OpenAlign, PAN_RATE, Prefetch, ROTATE_DRAG_SPAN,
        ROTATE_DRAG_SPEED, Ramps, SPREAD_EASE_TIME, Ui, ViewState, ZOOM_ANCHOR_DRIFT,
        ZOOM_DRAG_SPAN, ZOOM_DRAG_SPEED, ZoomMode, aim_fill_pan, attach_blur_bg, draw_patch,
        draw_selection, draw_turned, keep_pivot, pan_drag_speed, patch_covers, put_back_view,
        read_nav_mouse, selection_rect, show_entry, view_filter,
    },
    zoomkeys::TypedKeys,
};
#[cfg(target_os = "linux")]
use winit::platform::x11::WindowAttributesExtX11 as _;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
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

/// The entry a launch opens before the user presses anything, if any.
///
/// A standalone window opens the only entry of a one-image directory and shows
/// the grid otherwise, so `ESC` has somewhere to go.
/// A host pane draws the image view only, so it opens the first entry of a
/// directory and never waits on a grid that nothing can show.
/// A document launch already carries its picture, so it opens nothing.
const fn open_initial(pane: Pane, single_plane: bool, entries: usize) -> Option<usize> {
    if single_plane || entries == 0 {
        return None;
    }
    if pane.shows_grid() && entries != 1 {
        return None;
    }
    Some(0)
}

/// Make the entry at index `idx` the open image.
fn tilde_path(path: &Path) -> String {
    if let Some(home) = std::env::var_os("HOME") {
        let home = Path::new(&home);
        // Skip the pathological HOME=/ case (everything collapses) and
        // an unset-but-empty value.
        if home.as_os_str().is_empty() || home == Path::new("/") {
            return path.display().to_string();
        }
        if let Ok(rest) = path.strip_prefix(home) {
            // strip_prefix yields a relative remainder ("pics"), so the code
            // adds the separator back.
            // A path equal to $HOME maps to plain "~".
            return if rest.as_os_str().is_empty() {
                "~".to_string()
            } else {
                format!("~/{rest}", rest = rest.display())
            };
        }
    }
    path.display().to_string()
}

/// The window title pieces shared by both launch modes.
const SEP: &str = " – ";
const APP: &str = "versatile-viewer";

/// The title for the image view of a single file: name, dir, app.
fn image_title(img_path: &Path) -> String {
    let abs = fs::canonicalize(img_path).unwrap_or_else(|_| img_path.to_path_buf());
    let file = abs.file_name().map_or_else(
        || abs.display().to_string(),
        |f| f.to_string_lossy().into_owned(),
    );
    let dir = abs.parent().unwrap_or(Path::new("/"));
    format!("{file}{SEP}{}{SEP}{APP}", tilde_path(dir))
}

/// The title for the grid over a directory with `n` images.
fn grid_title(dir_path: &Path, n: usize) -> String {
    format!(
        "({n} image{plural}){SEP}{}{SEP}{APP}",
        tilde_path(dir_path),
        plural = if n == 1 { "" } else { "s" },
    )
}

/// The initial window title for the launch mode: image view or grid view.
fn window_title(path: &Path, dir_grid: Option<&Grid>) -> String {
    // Make the path absolute (resolving `.` and symlinks), so the title is
    // meaningful regardless of the launch cwd.
    let abs: PathBuf = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    match dir_grid {
        Some(g) => grid_title(&abs, g.entries.len()),
        None => image_title(&abs),
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

// The event loop is one long state machine by design.
// Splitting it scatters the frame-order invariants across call sites.

/// What the launch decided before a window existed.
///
/// The CLI, the directory listing and, for a single file, the decode all run
/// here: the window title carries the image count, and a picture sizes its own
/// window. The GPU work waits for [`App::resumed`], because a surface needs a
/// window.
struct Launch {
    /// The window kind this launch draws into.
    pane: Pane,
    /// The grid of a directory launch, or the one-cell grid behind a document.
    grid: Option<Grid>,
    /// The decoded picture of a single-file launch.
    decoded: Option<DecodedImage>,
    /// The surface of a single-file launch, kept when it can repaint a region.
    source: Option<Arc<dyn Surface>>,
    /// The part of the plane a fresh open frames.
    open_box: Option<Size>,
    /// The tiny blurred background of a single-file launch.
    blur: Option<DecodedImage>,
    /// The launch starts in the image view.
    single_plane: bool,
    /// The window the launch asks for, in physical pixels.
    win0: (u32, u32),
    /// The title that both views start from.
    title: String,
}

impl Launch {
    /// Read the command line and open the path.
    fn open(path: &Path, pane: Pane) -> Result<Self> {
        // Directory launch: list the directory before opening the window (the
        // title shows the image count).
        // Single-file launch: open the surface now.
        // A document opens as a book (see `book.rs`) and keeps a one-cell grid,
        // so ESC still has a target and that cell shows the cover page.
        // A one-plane surface goes straight into the image view with no grid.
        let mut book = false;
        let (grid, single_surface) = if path.is_dir() {
            (Some(Grid::from_dir(path)?), None)
        } else if path.is_file() {
            let plain = open_surface(path)?;
            book = plain.plane_count() > 1;
            let surface: Arc<dyn Surface> = if book {
                Arc::new(BookSurface::new(plain.into())?)
            } else {
                plain.into()
            };
            let grid = book.then(|| Grid::from_paths(vec![path.to_path_buf()]));
            (grid, Some(surface))
        } else {
            bail!("no such file or directory: {}", path.display());
        };
        let single_plane = single_surface.is_some();

        // A single-plane launch renders before the window exists, so the window
        // can be sized to the picture.
        // A directory launch keeps the default window size.
        let decoded = if single_plane {
            Some(
                single_surface
                    .as_ref()
                    .expect("a single-file launch has a surface")
                    .render(View::whole(0, 1.0))?,
            )
        } else {
            None
        };
        // The box the launch frames: a book names its first spread, so the
        // cover page shows whole, and every other source frames its whole
        // plane.
        let open_box = single_surface
            .as_ref()
            .and_then(|surface| surface.open_box(0));
        // Only a source that can render a sharper picture of the same plane
        // keeps its handle: a vector file repaints the region its window shows,
        // while a raster file has nothing to add past its own pixels.
        let source = if single_plane
            && single_surface
                .as_ref()
                .is_some_and(|surface| surface.can_refine())
        {
            single_surface
        } else {
            None
        };
        // Blur-background source for a single-plane launch: computed while the
        // RGBA buffer still exists, so the window uploads it once.
        let blur = if single_plane {
            blurbg::Config::from_env()
                .map(|config| config.blur(decoded.as_ref().expect("single image decoded")))
        } else {
            None
        };
        // A book keeps the default size, because its canvas can stand taller
        // than any texture and wider than any screen.
        let win0 = decoded
            .as_ref()
            .filter(|_| !book)
            .map_or((1024, 768), |image| {
                (image.width().raw().max(1), image.height().raw().max(1))
            });
        let title = window_title(path, grid.as_ref().filter(|_| !single_plane));
        Ok(Self {
            pane,
            grid,
            decoded,
            source,
            open_box,
            blur,
            single_plane,
            win0,
            title,
        })
    }
}

/// The viewer: the window, the GPU stack, the two views and the latches of the
/// frame loop.
#[allow(clippy::struct_excessive_bools)]
struct App {
    // --- platform ---
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    renderer: Option<Renderer>,
    input: Input,
    /// What kind of window this process draws into.
    pane: Pane,
    /// The host window an embedded pane fills, so the pane follows its size.
    ///
    /// None for a top-level window, and None for a pane whose display could not
    /// be reached, which then keeps the size it launched with.
    pane_geometry: Option<PaneGeometry>,
    /// The swapchain arrived and the launch frame uploaded.
    started: bool,
    /// The window size of this frame, in physical pixels.
    size: (u32, u32),
    /// When the previous frame started, for the frame time.
    last_frame: Instant,
    /// When the first frame started, for `Ui::elapsed`.
    started_at: Instant,
    /// The frame time of the previous frame, in seconds.
    dt: f32,
    /// A failure that stopped the loop, returned from `main`.
    fatal: Option<anyhow::Error>,

    // --- launch data, consumed in `resumed` ---
    decoded: Option<DecodedImage>,
    pending_source: Option<Arc<dyn Surface>>,
    pending_box: Option<Size>,
    blur: Option<DecodedImage>,
    single_plane: bool,
    win0: (u32, u32),
    launch_title: String,

    // --- views ---
    st: ViewState,
    grid: Option<Grid>,

    // --- frame-loop latches ---
    /// The grid entry whose title the window shows now.
    title_open_id: Option<u64>,
    /// Set when the viewer exits entirely.
    quit: bool,
    /// `VV_DEBUG=1`: trace the input and the drag geometry to stderr.
    debug: bool,
    /// `VV_TIMING=1`: trace a slow frame to stderr.
    timing: bool,
    /// One spread step per press: a step disarms the next one, and the release
    /// of `j`/`k` arms it again.
    spread_armed: bool,
    /// The previous frame's window size, for the resize detection.
    last_win: Option<(f32, f32)>,
    /// The swapchain answered no image on the last frame, so the window is
    /// hidden or occluded and the loop paces itself.
    occluded: bool,
    /// True while an image-view drag holds the pointer.
    ///
    /// This is the view's latch; [`Input::capture`] carries the same fact to
    /// the input layer, so the raw travel moves the pointer. The two move
    /// together, and only through [`Ui::grab_pointer`], [`Ui::release_pointer`]
    /// or the lost-focus arm.
    pointer_captured: bool,
    /// How far one unit of raw device travel moves the picture in a drag.
    ///
    /// `VV_PAN_DRAG_SPEED` sets it once at launch (see [`pan_drag_speed`]).
    pan_drag_speed: f32,
    /// The window point where the current drag grabbed the pointer.
    grab_pos: Vec2,
    /// The pointer position while captured: the grab point plus the raw travel.
    drag_virtual: Vec2,
    /// A post-drag pointer restore that waits for the warp to land.
    pending_restore: Option<Vec2>,
    /// How many warps the restore tried.
    restore_tries: u32,
    /// `f` held captures the pointer (a pan, or a free zoom with Shift).
    f_capture: bool,
    /// The running `f` hold zooms instead of panning.
    f_zoom_capture: bool,
    /// The running `f` hold ever zoomed, so its release runs no tap action.
    f_zoom_tapped: bool,
    /// The free-zoom drag's on-screen anchor.
    f_zoom_anchor: Option<Vec2>,
    /// `r` held captures the pointer and turns the image.
    r_capture: bool,
    /// The `v` region drag in progress.
    region_drag: Option<region_sel::Drag>,
    /// The held state of the zoom keys on layouts where `+`/`-` sit elsewhere.
    zoom_keys: TypedKeys,
    /// The auto-repeat of the held side-button navigation.
    rep_nav_fwd: versatile_viewer::keyrepeat::RepeatState,
    rep_nav_back: versatile_viewer::keyrepeat::RepeatState,
}

impl App {
    /// Hold the launch result until the window exists.
    fn new(launch: Launch) -> Self {
        let (crop_tx, crop_rx) = channel();
        Self {
            window: None,
            gpu: None,
            renderer: None,
            input: Input::default(),
            pane: launch.pane,
            pane_geometry: None,
            started: false,
            size: launch.win0,
            last_frame: Instant::now(),
            started_at: Instant::now(),
            dt: 0.0,
            fatal: None,
            decoded: launch.decoded,
            pending_source: launch.source,
            pending_box: launch.open_box,
            blur: launch.blur,
            single_plane: launch.single_plane,
            win0: launch.win0,
            launch_title: launch.title,
            st: ViewState {
                // A single-file launch shows its picture (a document shows its
                // first spread) and keeps the grid, if any, behind it for ESC.
                mode: if launch.single_plane {
                    Mode::Image
                } else {
                    Mode::Grid
                },
                open_id: None,
                view_from_grid: None,
                view_tex: None,
                view_loading: false,
                img: Size { w: 0.0, h: 0.0 },
                zoom: ZoomMode::FitDown,
                pan: Vec2::ZERO,
                target_pan: Vec2::ZERO,
                view_scale: None,
                zoom_anchor: None,
                angle: 0.0,
                pivot: None,
                selection: None,
                ramps: Ramps::default(),
                pixelated: false,
                loader: None,
                blur_bg: BlurBg::from_env(),
                source: None,
                source_plane: 0,
                open_box: None,
                open_align: OpenAlign::Free,
                spread_glide: None,
                prefetch: Prefetch::Idle,
                prefetch_seq: 0,
                crop: None,
                crop_pending: false,
                crop_gen: 0,
                crop_seen: None,
                crop_quiet_at: Instant::now(),
                crop_tx,
                crop_rx,
            },
            grid: launch.grid,
            title_open_id: None,
            quit: false,
            debug: std::env::var_os("VV_DEBUG").is_some(),
            timing: versatile_viewer::timing_enabled(),
            spread_armed: true,
            last_win: None,
            occluded: false,
            pointer_captured: false,
            pan_drag_speed: pan_drag_speed(),
            grab_pos: Vec2::ZERO,
            drag_virtual: Vec2::ZERO,
            pending_restore: None,
            restore_tries: 0,
            f_capture: false,
            f_zoom_capture: false,
            f_zoom_tapped: false,
            f_zoom_anchor: None,
            r_capture: false,
            region_drag: None,
            zoom_keys: TypedKeys::default(),
            rep_nav_fwd: versatile_viewer::keyrepeat::RepeatState::default(),
            rep_nav_back: versatile_viewer::keyrepeat::RepeatState::default(),
        }
    }

    /// Upload the launch frame and start the background passes.
    ///
    /// The decode ran before the window existed, so this is the first moment a
    /// texture can exist.
    fn start_views(&mut self, window: &Arc<Window>) {
        self.size = window.inner_size().into();
        let (renderer, gpu) = (
            self.renderer.as_ref().expect("just built"),
            self.gpu.as_ref().expect("just built"),
        );
        if let Some(image) = self.decoded.take() {
            match renderer.upload(&image) {
                Ok(texture) => self.st.view_tex = Some(texture),
                Err(err) => eprintln!("vv: upload: {err}"),
            }
            self.st.adopt_source(self.pending_source.take());
            self.st.img = Size {
                w: image.width().as_f32(),
                h: image.height().as_f32(),
            };
            // A book frames its first spread, a picture frames the whole plane.
            self.st.open_box = self.pending_box.take();
            self.st.reset_view();
            if let Some(data) = self.blur.take() {
                attach_blur_bg(
                    &mut self.st.blur_bg,
                    &renderer.uploader(),
                    Some(&data),
                    None,
                );
            }
        }
        // The grid listed its directory before the window existed; the decodes
        // and the placeholder pass upload their own textures, so they start
        // here.
        if let Some(grid) = self.grid.as_mut() {
            grid.start(renderer.uploader());
        }
        // A one-image directory skips the grid: jump straight into that image
        // (ESC still returns to the grid view).
        // A host pane draws no grid at all, so it opens the first entry.
        let open = open_initial(
            self.pane,
            self.single_plane,
            self.grid.as_ref().map_or(0, |grid| grid.entries.len()),
        );
        if let Some(index) = open {
            let ui = Ui {
                input: &mut self.input,
                window,
                gpu,
                size: self.size,
                time: 0.0,
                dt: 0.0,
            };
            let win = (WinW::new(self.size.0 as f32), WinH::new(self.size.1 as f32));
            show_entry(
                &mut self.st,
                self.grid.as_mut().expect("a grid with entries"),
                index,
                &ui,
                renderer,
                win,
                true,
            );
        }
    }

    /// Ask for the size of the host window an embedded pane fills.
    ///
    /// The host resizes the container it created and never the child this
    /// process draws into, so X sends the `ConfigureNotify` to the container
    /// alone. The pane therefore reads the parent geometry itself and asks for
    /// the same size, the way `mpv --wid` does. The swapchain reconfigures and
    /// the picture refits on the `Resized` that comes back, so this frame still
    /// draws at the size it already had.
    fn follow_pane(&mut self, ui: &Ui) {
        let Some(geometry) = self.pane_geometry.as_mut() else {
            return;
        };
        match geometry.poll() {
            Ok(Some((width, height))) => {
                let _ = ui
                    .window
                    .request_inner_size(winit::dpi::PhysicalSize::new(width, height));
            }
            Ok(None) => {}
            Err(err) => {
                // A host window that went away ends the poll instead of
                // printing on every frame, and the pane keeps its size.
                eprintln!("vv: {err}");
                self.pane_geometry = None;
            }
        }
    }

    /// Run one frame: read the input, advance every ease, draw and present.
    //
    // One frame is one long state machine by design: the order of the steps
    // (drain, input, motion, draw) carries the invariants.
    #[allow(clippy::too_many_lines)]
    fn frame(&mut self) -> Result<()> {
        // A redraw can arrive before the GPU stack is up, or while a failed
        // start is on its way out: neither has a frame to draw.
        let (Some(window), Some(_)) = (self.window.clone(), self.renderer.as_ref()) else {
            return Ok(());
        };
        let now = Instant::now();
        let time = now.duration_since(self.started_at).as_secs_f64();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.dt = dt;
        // The frame body borrows the frame's input and GPU immutably while it
        // mutates the views, so the two move out of the app for the length of
        // the call and back in afterwards.
        let Some(gpu) = self.gpu.take() else {
            return Ok(());
        };
        // A dead device cannot draw again, and every later submit would fail
        // silently. Stop with the reason the driver gave instead of showing a
        // frozen picture.
        if let Some(reason) = gpu.lost_reason() {
            bail!("the GPU device was lost: {reason}");
        }
        let mut input = std::mem::take(&mut self.input);
        let mut ui = Ui {
            input: &mut input,
            window: &window,
            gpu: &gpu,
            size: self.size,
            time,
            dt,
        };
        let Some(mut r) = self.renderer.take() else {
            self.gpu = Some(gpu);
            return Ok(());
        };
        let result = self.body(&mut ui, &mut r);
        self.renderer = Some(r);
        self.gpu = Some(gpu);
        self.input = input;
        // The frame edges belong to one frame: the held state and the pointer
        // stay, the pressed, released, typed and travel sets start empty.
        self.input.begin_frame();
        result
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
        // A host window is the parent of the child this process draws into, so
        // `XCreateWindow` runs with that parent and the child never becomes a
        // top-level window: no window manager frames it, and the host decides
        // where it sits.
        // A child window carries no decoration to name and no `WM_CLASS` to
        // read, so neither is set.
        let mut attributes = Window::default_attributes()
            .with_inner_size(winit::dpi::PhysicalSize::new(self.win0.0, self.win0.1))
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
        let renderer = Renderer::new(
            gpu.device(),
            gpu.queue(),
            gpu.format(),
            gpu.max_texture_side(),
        );
        self.size = window.inner_size().into();
        self.window = Some(Arc::clone(&window));
        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
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
            eprintln!(
                "vv: window scale factor {}, pan drag speed {}",
                window.scale_factor(),
                self.pan_drag_speed
            );
        }
        self.started = true;
        self.started_at = Instant::now();
        self.last_frame = Instant::now();
        self.start_views(&window);
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested if !self.started => {}
            WindowEvent::RedrawRequested => match self.frame() {
                Ok(()) => {
                    if self.quit {
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
            WindowEvent::KeyboardInput { event, .. } => self.key(&event),
            WindowEvent::MouseWheel { delta, .. } => self.input.wheel(delta),
            WindowEvent::CursorMoved { position, .. } => {
                // A captured drag reads the raw device travel instead, which the
                // input layer applies on its own.
                self.input
                    .cursor(Vec2::new(position.x as f32, position.y as f32));
            }
            WindowEvent::MouseInput { state, button, .. } => self.input.button(button, state),
            WindowEvent::Focused(false) => {
                // A lost focus never delivers the releases, so a captured drag
                // must not outlive it.
                self.input.forget_held();
                self.input.capture(false);
                self.pointer_captured = false;
                self.f_capture = false;
                self.f_zoom_capture = false;
                self.r_capture = false;
                self.region_drag = None;
                if let Some(window) = &self.window {
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                    window.set_cursor_visible(true);
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            // The raw travel of the device: no pointer acceleration and no
            // desktop scale, which is the source raylib's `DisableCursor` fed
            // the view too.
            self.input.motion(Vec2::new(delta.0 as f32, delta.1 as f32));
        }
    }
}

impl App {
    /// One frame: read the input, advance every ease, draw and present.
    ///
    /// `ui` carries the window and the input of this frame, and `r` the batch
    /// the draw calls fill.
    #[allow(clippy::too_many_lines)]
    fn body(&mut self, ui: &mut Ui, r: &mut Renderer) -> Result<()> {
        // A host pane follows the size of the window it fills, at the top of the
        // frame so a resize lands before the fit reads the window.
        self.follow_pane(ui);
        // VV_TIMING: the frame started now.
        // The code reports it at the end of the frame when it ran long.
        let frame_started = self.timing.then(std::time::Instant::now);
        // VV_DEBUG: trace every key press this frame.
        if self.debug {
            for k in ui.pressed_keys() {
                eprintln!("vv: PRESSED key {k:?}");
            }
            // A frame longer than a key tap can hide the whole press. The input
            // layer records every event, so the trace above still shows it, and
            // this line names the frame that swallowed it.
            let ft = ui.frame_time();
            if ft > 0.1 {
                eprintln!("vv: slow frame {ft:.0} ms");
            }
        }
        let win_w = ui.width() as f32;
        let win_h = ui.height() as f32;
        // Post-drag restore: verify the warp landed where the code sent it.
        // If a competing re-warp (GLFW, WM, XWayland) moved it, snap again.
        // The code gives up after a few tries so a user moving the mouse is
        // never pinned.
        if let Some(target) = self.pending_restore {
            let cur = ui.pointer();
            let settled = (cur.x - target.x).abs() < 1.0 && (cur.y - target.y).abs() < 1.0;
            if self.debug && !settled {
                eprintln!(
                    "vv: restore try {}: cur ({:.0},{:.0}) -> ({:.0},{:.0})",
                    self.restore_tries, cur.x, cur.y, target.x, target.y
                );
            }
            if settled || self.restore_tries >= 3 {
                self.pending_restore = None;
            } else {
                ui.place_pointer(target);
                self.restore_tries += 1;
            }
        }
        // The code does the resize detection itself.
        // `is_window_resized()` can miss a WM-reflow resize that lands around
        // the time the window becomes visible (tiling WMs shrink the freshly
        // spawned window into its tile), so the fit ease glides "out of
        // nowhere".
        // Any size difference from the previous frame counts as a resize
        // frame.
        let resized = self.last_win != Some((win_w, win_h));
        self.last_win = Some((win_w, win_h));

        // Dynamic window title: follow the open image in image view.
        // Back to the launch title in the grid mode.
        // A host pane owns no title: its `WM_NAME` is not a decoration that a
        // window manager reads, and the host may clear it.
        if self.pane.shows_grid() && self.st.open_id != self.title_open_id {
            self.title_open_id = self.st.open_id;
            // A the grid title is the launch title already, so the borrow costs
            // no allocation.
            let title = self
                .st
                .open_id
                .and_then(|id| {
                    self.grid
                        .as_ref()?
                        .entries
                        .iter()
                        .find(|e| e.id == id)
                        .map(|e| Cow::Owned(image_title(e.source.path())))
                })
                .unwrap_or(Cow::Borrowed(self.launch_title.as_str()));
            ui.set_title(&title);
        }

        // Prefetch priority: in the grid mode the neighbors of the selected
        // entry (left, right, up, down).
        // In image mode the previous and next entries.
        // While the image view still waits for its own decode, only the
        // open entry decodes and the code pauses the rest of the queue, so
        // the visible image gets all the cores.
        // load_pending drains finished decodes (texture uploads happen here,
        // on the main thread) and dispatches new ones to the rayon pool.
        let plan = if self.st.mode == Mode::Image {
            match self
                .st
                .open_id
                .and_then(|id| self.grid.as_ref().and_then(|g| g.index_of(id)))
            {
                Some(i) => {
                    let n = self.grid.as_ref().unwrap().entries.len();
                    let mut priority = Vec::new();
                    if self.st.view_loading {
                        priority.push(i);
                    } else {
                        if i > 0 {
                            priority.push(i - 1);
                        }
                        if i + 1 < n {
                            priority.push(i + 1);
                        }
                    }
                    LoadPlan {
                        priority,
                        scan_start: i,
                        scan: !self.st.view_loading,
                        preload_full: true,
                    }
                }
                None => LoadPlan {
                    scan: true,
                    preload_full: true,
                    ..LoadPlan::default()
                },
            }
        } else if let Some(g) = self.grid.as_ref() {
            LoadPlan {
                priority: g.prefetch_neighbors(g.selected, WinW::new(win_w), WinH::new(win_h)),
                scan_start: g.selected,
                scan: true,
                preload_full: false,
            }
        } else {
            LoadPlan {
                scan: true,
                ..LoadPlan::default()
            }
        };
        if let Some(g) = self.grid.as_mut() {
            g.load_pending(&plan);
        }

        // Image mode: drain the streaming loader first.
        // Texture uploads need the main thread.
        if self.st.mode == Mode::Image {
            let mut open_failed: Option<String> = None;
            // Waiting on a the grid decode (no streaming loader for this open).
            // When its texture lands, the code takes it over as the view.
            if self.st.view_tex.is_none() && self.st.loader.is_none() && self.st.open_id.is_some() {
                let id = self.st.open_id.expect("checked above");
                match self
                    .grid
                    .as_mut()
                    .and_then(|g| g.entries.iter().position(|e| e.id == id).map(|i| (g, i)))
                {
                    Some((g, i)) => {
                        // A document keeps no full-res texture, so a finished
                        // decode re-enters here and starts the spread loader.
                        // A failed entry never re-enters, because its loader
                        // would fail the same way and spin.
                        let ready = !g.entries[i].cell.is_failed()
                            && (g.entries[i].full.is_some()
                                || (g.entries[i].no_full && !g.entries[i].queued));
                        if ready {
                            // Decode landed: the code kept its full-res
                            // texture for the open entry (keep set), and
                            // show_entry takes it over.
                            show_entry(
                                &mut self.st,
                                g,
                                i,
                                ui,
                                r,
                                (WinW::new(win_w), WinH::new(win_h)),
                                false,
                            );
                        }
                        // else: the decode is still in flight or awaits
                        // dispatch (thumb-only keep-set entry, and the keep set
                        // always holds the open entry, so the code dispatches
                        // it).
                        // The thumb placeholder covers the wait and the
                        // decode-landed takeover swaps it in.
                    }
                    None => open_failed = Some("the image is no longer in the grid".to_string()),
                }
            }
            // One message at a time: the borrow of the loader field ends
            // before the body touches the view, so a refit may borrow it.
            while let Some(msg) = self.st.loader.as_ref().and_then(Loader::try_recv) {
                match msg {
                    Ok(DecodeMsg::Header { width, height }) => {
                        // Dimensions known: fit down at once (the ease
                        // block only runs from the next frame on).
                        self.st.img = Size {
                            w: width.as_f32(),
                            h: height.as_f32(),
                        };
                        self.st.refit_open();
                    }
                    Ok(DecodeMsg::Preview { frame, blur }) => {
                        // Progressively better render of the same image.
                        self.st.view_tex = Some(frame.upload(&r.uploader())?);
                        attach_blur_bg(
                            &mut self.st.blur_bg,
                            &r.uploader(),
                            blur.as_ref(),
                            self.st.open_id,
                        );
                    }
                    Ok(DecodeMsg::Done { frame, blur }) => {
                        // Non-JXL formats only learn dimensions here.
                        if self.st.img.w == 0.0 {
                            let (width, height) = frame.dims();
                            self.st.img = Size {
                                w: width as f32,
                                h: height as f32,
                            };
                            self.st.refit_open();
                        }
                        self.st.view_tex = Some(frame.upload(&r.uploader())?);
                        attach_blur_bg(
                            &mut self.st.blur_bg,
                            &r.uploader(),
                            blur.as_ref(),
                            self.st.open_id,
                        );
                        self.st.view_loading = false;
                    }
                    Err(err) => {
                        eprintln!("vv: {err}");
                        open_failed = Some(err.to_string());
                    }
                }
            }
            if let Some(reason) = open_failed {
                // A host pane has no grid to fall back to, and the picture is
                // its whole contract, so the process stops with the reason
                // instead of leaving an empty window behind.
                if !self.pane.shows_grid() {
                    bail!("{reason}");
                }
                self.st.loader = None; // cancel the worker
                self.st.close_crop();
                // Keep the entry: mark it as failed, so its cell stays
                // visible (dimmed, error mark) instead of vanishing
                // silently.
                if let Some(id) = self.st.open_id.take() {
                    self.grid
                        .as_mut()
                        .unwrap()
                        .mark_failed(id, DecodeFailure::Decode);
                }
                self.st.view_from_grid = None;
                self.st.mode = Mode::Grid;
                self.st.view_tex = None;
                self.st.view_loading = false;
            }
        }

        // True while a Shift key is down.
        // Shift + h/j/k/l moves at double speed in both views: the the grid
        // selection step and the image pan and free zoom.
        // The flag lives outside the mode branches because both read it.
        let shift = ui.key_down(KeyCode::ShiftLeft) || ui.key_down(KeyCode::ShiftRight);
        // Shift + h/j/k/l moves at double speed in both views.
        // The the grid passes the step count to its input handler, and the image
        // view scales its pan and zoom ramps.
        let shift_mult = if shift {
            versatile_viewer::SHIFT_SPEED_MULTIPLIER
        } else {
            1
        };
        let shift_speed = shift_mult as f32;

        if self.st.mode == Mode::Grid {
            // Grid navigation: h/j/k/l and the arrows move the selection,
            // Enter opens the selected image, q quits (ESC is inert here,
            // because the the grid is the home view).
            match self.grid.as_mut().unwrap().handle_input(
                ui.input(),
                (WinW::new(win_w), WinH::new(win_h)),
                shift_mult,
            ) {
                Some(GridAction::Open(i)) => {
                    if self.debug {
                        eprintln!("vv: enter pressed -> open idx {i}");
                    }
                    show_entry(
                        &mut self.st,
                        self.grid.as_mut().unwrap(),
                        i,
                        ui,
                        r,
                        (WinW::new(win_w), WinH::new(win_h)),
                        true,
                    );
                }
                Some(GridAction::Quit) => self.quit = true,
                None => {}
            }
        } else {
            // Image mode.
            // Drain the raw key and char queues every frame (same rationale
            // as grid.rs handle_input).
            // key_pressed misses a press+release pair that lands inside
            // one frame, and this is easy here while frames stall on texture
            // uploads or decode.
            // Leftover queue entries otherwise leak into the grid mode and act
            // there (for example a missed Enter that reopens the
            // just-viewed image at once).
            // State queries (key_down panning, key_pressed W/E/=/-) are
            // unaffected: the queue is separate from the key snapshot.
            // Enter and ESC return to the the grid when one exists (nsxiv-like:
            // Enter toggles between the the grid and the open image).
            // q quits, and ESC never quits the program (inert in single-file
            // launches, where there is no the grid to return to).
            // Space and Backspace switch to the next or previous image
            // (nsxiv-style nav, with n or Space for next and p or
            // Backspace for previous, while the arrows and h/j/k/l stay
            // panning).
            // Shift+Space steps backwards, so one key serves both ways.
            // The mouse side buttons navigate too: forward is next and back
            // is previous, like n and p.
            // Nav keys auto-repeat while held, at the X server rate
            // (xset r rate, from keyrepeat::settings).
            // g/G jump to the first or last image.
            let mut enter = false;
            let mut escape = false;
            let mut quit_pressed = false;
            let mut pixelated_toggle = false;
            let mut f_key = false;
            let mut r_key = false;
            // The Shift state at the `r` press edge picks the pivot.
            let mut r_pivot_edge = false;
            // The `v` press edge, and the Shift state that turns its drag into
            // a square.
            let mut v_key = false;
            let mut v_square_edge = false;
            let mut nav_next_edge = false;
            let mut nav_prev_edge = false;
            let mut jump_first = false;
            let mut jump_last = false;
            // Shift+J/K and Page-down/Page-up step one spread in a book (see
            // [`ViewState::step_spread`]).
            // The press edge fires once per press, so the X server auto-repeat
            // never chains two steps from one hold.
            let mut spread_next = false;
            let mut spread_prev = false;
            // The key that went down this frame, if any.
            // A typed `+`/`-` pairs with it to learn which physical key holds
            // that character (see zoomkeys::TypedKeys).
            for k in ui.pressed_keys() {
                match k {
                    // ESC drops a region selection first and keeps the image
                    // view, so a second ESC returns to the the grid.
                    KeyCode::Escape => escape = true,
                    KeyCode::Enter | KeyCode::NumpadEnter => {
                        enter = true;
                    }
                    KeyCode::KeyQ => quit_pressed = true,
                    KeyCode::KeyA => pixelated_toggle = true,
                    KeyCode::KeyF => f_key = true,
                    KeyCode::KeyR => {
                        r_key = true;
                        r_pivot_edge = shift;
                    }
                    KeyCode::KeyV => {
                        v_key = true;
                        v_square_edge = shift;
                    }
                    KeyCode::Space => {
                        if shift {
                            nav_prev_edge = true;
                        } else {
                            nav_next_edge = true;
                        }
                    }
                    KeyCode::KeyN => nav_next_edge = true,
                    KeyCode::Backspace | KeyCode::KeyP => nav_prev_edge = true,
                    KeyCode::KeyG => {
                        if shift {
                            jump_last = true;
                        } else {
                            jump_first = true;
                        }
                    }
                    KeyCode::KeyJ => spread_next = shift,
                    KeyCode::KeyK => spread_prev = shift,
                    // Page-down and Page-up step a spread too, so a reader who
                    // never learned the Shift+J/K pair still turns the page.
                    KeyCode::PageDown => spread_next = true,
                    KeyCode::PageUp => spread_prev = true,
                    _ => {}
                }
            }
            // `f` pan-drag: the press edge captures the pointer exactly like
            // a left-drag (cursor hidden, unbounded virtual deltas).
            // The release ends it and restores the pointer to the grab point.
            // Travel marks it a hold, with no tap action.
            // Otherwise the release runs the `f` action at the grab point.
            // Shift at the press edge makes it a zoom hold instead, and a
            // Shift press or release mid-hold switches the mode.
            // Both holds capture the pointer, so the travel is unbounded: the
            // pointer keeps recentering and the motion never stops at a screen
            // edge.
            // The zoom anchor starts at the grab point, which the capture holds
            // still, so the image point under it drifts toward the center on
            // screen.
            // Auto-repeat re-fires the press edge while held, hence the
            // `!self.pointer_captured && !self.f_capture` guard.
            // The same guard covers the `r` rotation hold, which grabs the
            // pointer the same way.
            let mut f_click: Option<Vec2> = None;
            if (f_key || r_key) && !self.pointer_captured && !self.f_capture && !self.r_capture {
                self.grab_pos = ui.pointer();
                self.drag_virtual = self.grab_pos;
                ui.grab_pointer();
                self.pointer_captured = true;
                if r_key {
                    // `r` turns the image. The hold adds to the angle the image
                    // shows now, so the press never resets the turn.
                    // A plain `r` turns around the image center, and Shift+R
                    // turns around the grabbed point, which stays on its window
                    // spot while the pointer stays captured.
                    self.r_capture = true;
                    self.st.pivot = r_pivot_edge.then_some(self.grab_pos);
                    if self.debug {
                        eprintln!(
                            "vv: r hold grab at ({:.0},{:.0}), angle {}, pivot {:?}",
                            self.grab_pos.x, self.grab_pos.y, self.st.angle, self.st.pivot
                        );
                    }
                } else {
                    self.f_capture = true;
                    self.f_zoom_capture = shift;
                    self.f_zoom_anchor = self.f_zoom_capture.then_some(self.grab_pos);
                    if self.debug {
                        eprintln!(
                            "vv: f hold grab at ({:.0},{:.0}), zoom {}",
                            self.grab_pos.x, self.grab_pos.y, self.f_zoom_capture
                        );
                    }
                }
            }
            // A Shift change mid-hold switches the pan and the zoom.
            // The capture stays, so the travel stays unbounded.
            if self.f_capture && shift != self.f_zoom_capture {
                self.f_zoom_capture = shift;
                // Forget the distance the tap test and the pan measured.
                self.grab_pos = ui.pointer();
                self.drag_virtual = self.grab_pos;
                // A zoom from here on anchors on the view center: the image
                // point the window shows at its middle.
                // The pointer is hidden, so its last screen spot says nothing
                // about what the user looks at, and a pan had just pushed that
                // spot away from the middle.
                // The anchor is the middle of the window, so the ease needs no
                // drift toward it.
                self.f_zoom_anchor = self.f_zoom_capture.then_some(Vec2 {
                    x: win_w / 2.0,
                    y: win_h / 2.0,
                });
                // A zoom hold never counts as a tap.
                self.f_zoom_tapped = true;
                if self.debug {
                    eprintln!("vv: f hold switch, zoom {}", self.f_zoom_capture);
                }
            }
            // `!key_down` also covers a press+release landing inside one
            // frame (the pressed queue records the press, but there is no
            // release edge to observe).
            // Without this the capture sticks.
            // The `r` release keeps the angle, so the image stays turned.
            if (self.f_capture || self.r_capture)
                && !ui.key_down(KeyCode::KeyF)
                && !ui.key_down(KeyCode::KeyR)
            {
                // Fold in this frame's motion before the lock is released
                // (the drag block below no longer runs once capture ends).
                let delta = ui.travel();
                self.drag_virtual.x += delta.x;
                self.drag_virtual.y += delta.y;
                let f_dragged = (self.drag_virtual - self.grab_pos).length() > TAP_SLACK;
                if self.pointer_captured {
                    ui.release_pointer();
                    ui.place_pointer(self.grab_pos);
                    self.pending_restore = Some(self.grab_pos);
                    self.restore_tries = 0;
                    self.pointer_captured = false;
                }
                // A zoom hold has no tap action: it would otherwise run the
                // `f` fill aim on release.
                if self.f_capture && !f_dragged && !self.f_zoom_capture && !self.f_zoom_tapped {
                    f_click = Some(self.grab_pos);
                }
                if self.f_capture {
                    self.f_capture = false;
                    self.f_zoom_capture = false;
                    self.f_zoom_tapped = false;
                    self.f_zoom_anchor = None;
                }
                if self.r_capture {
                    // The release snaps a near quarter turn exactly, and it is
                    // the only place that snaps: the drag itself turns freely.
                    // A pivot turn keeps its point on screen through the snap.
                    let before = self.st.angle;
                    self.st.angle = rotate::snap_deg(self.st.angle);
                    keep_pivot(&mut self.st, before, win_w, win_h);
                    self.r_capture = false;
                }
                if self.debug {
                    // The accumulated travel is the number a feel report needs:
                    // a 100 px drag that reads 200 says the input source counts
                    // twice, whatever the factor is.
                    let travel = self.drag_virtual - self.grab_pos;
                    eprintln!(
                        "vv: hold release: grab ({:.0},{:.0}), virtual ({:.0},{:.0}), speed {}, \
                         dragged {f_dragged}, angle {}, travel ({:.0},{:.0})",
                        self.grab_pos.x,
                        self.grab_pos.y,
                        self.drag_virtual.x,
                        self.drag_virtual.y,
                        self.pan_drag_speed,
                        self.st.angle,
                        travel.x,
                        travel.y
                    );
                }
            }
            // Auto-repeat: the initial press fires at once (edge from the
            // queue above), and the hold fires at the X server repeat rate
            // after its delay.
            // The mouse side buttons ride the same nav path: forward is next
            // and back is previous, exactly like n and p.
            let now = ui.elapsed();
            let (delay, rate) = versatile_viewer::keyrepeat::settings();
            let mouse_nav = read_nav_mouse(ui.input());
            let next_edge = nav_next_edge || mouse_nav.next.edge;
            let prev_edge = nav_prev_edge || mouse_nav.prev.edge;
            // A held Shift moves the Space hold to the previous direction,
            // so a hold repeats the way its press edge stepped.
            let space_held = ui.key_down(KeyCode::Space);
            let next_held =
                ui.key_down(KeyCode::KeyN) || (space_held && !shift) || mouse_nav.next.held;
            let prev_held = ui.key_down(KeyCode::KeyP)
                || ui.key_down(KeyCode::Backspace)
                || (space_held && shift)
                || mouse_nav.prev.held;
            let mut nav = if self
                .rep_nav_fwd
                .tick(next_edge, next_held, now, delay, rate)
            {
                Some(1)
            } else if self
                .rep_nav_back
                .tick(prev_edge, prev_held, now, delay, rate)
            {
                Some(-1)
            } else {
                None
            };
            // g/G: jump to the first or last image (as a nav delta).
            if (jump_first || jump_last)
                && let Some(cur) = self
                    .st
                    .open_id
                    .and_then(|id| self.grid.as_ref().and_then(|g| g.index_of(id)))
            {
                let n = self.grid.as_ref().unwrap().entries.len();
                let t = if jump_first { 0 } else { n - 1 };
                nav = Some(t as i64 - cur as i64);
            }
            // Some input setups (IMEs, unusual X11 input methods) deliver
            // Enter as a character event ('\n'/ '\r').
            // The code accepts both.
            //
            // A typed '+'/'-' teaches its physical keycode (see
            // zoomkeys::TypedKeys).
            // That covers the layouts where the keys do not sit on their US
            // keycodes.
            // The code then polls that keycode's held state, and the motion
            // stays continuous however the auto-repeat behaves.
            for typed in ui.typed() {
                match typed.text {
                    '\n' | '\r' => enter = true,
                    '+' => self.zoom_keys.typed_zoom_in(Some(typed.key)),
                    '-' => self.zoom_keys.typed_zoom_out(Some(typed.key)),
                    _ => {}
                }
            }
            // ESC drops the region selection first: it clears a kept region
            // and cancels a drag in flight.
            // Only an ESC with nothing to drop returns to the the grid, so the
            // region never costs a surprise exit.
            if escape {
                let kept = self.st.selection.take();
                let dragging = self.region_drag.take();
                if kept.is_none() && dragging.is_none() {
                    enter = true;
                } else if self.debug {
                    eprintln!(
                        "vv: esc dropped region {:?}, drag {}",
                        kept.map(|r| (r.x, r.y, r.w, r.h)),
                        dragging.is_some()
                    );
                }
            }

            // One spread step per press of Shift+J/K or Page-down/Page-up: the
            // press queue reports a press once, and the latch covers an input
            // setup whose auto-repeat arrives as a second press.
            let spread_press = if spread_next {
                Some(1)
            } else if spread_prev {
                Some(-1)
            } else {
                None
            };
            let spread_step = spread_press.filter(|_| self.spread_armed);
            self.spread_armed = !(ui.key_down(KeyCode::KeyJ)
                || ui.key_down(KeyCode::KeyK)
                || ui.key_down(KeyCode::PageDown)
                || ui.key_down(KeyCode::PageUp));

            let mut return_to_grid = false;
            if quit_pressed {
                self.quit = true; // single-file launch: no self.grid to fall back to
            }
            // `a`: toggle smooth (bilinear, default) against pixelated
            // (nearest-neighbor) filtering for the shown texture.
            // The code re-applies the flag to every texture that arrives
            // later.
            if pixelated_toggle {
                // The filter is a draw-time choice of this frame on, so the
                // toggle only flips the flag: every texture draws with it.
                self.st.pixelated = !self.st.pixelated;
            }
            // `f` tap action: it needs known geometry, and the hit test uses
            // the drawn image rect (offset + pan at the on-screen scale).
            // A cursor over the background returns to the the grid (ESC-like).
            // A cursor over an image that covers the whole window zooms out
            // (recentred, like the zoom-out half of t).
            // Otherwise it zooms into fill view aimed at the cursor (like the
            // zoom-in half of t).
            // It fires on the key release, and only when the hold did not pan.
            let mut f_open_grid = false;
            let mut f_zoom = false;
            if let Some(m) = f_click
                && self.st.img.w > 0.0
                && let Some(scale) = self.st.view_scale
            {
                let center = Vec2::new(win_w / 2.0 + self.st.pan.x, win_h / 2.0 + self.st.pan.y);
                // A turned image must hit-test its own outline, so the point
                // goes back into the unrotated image space first.
                let m = rotate::unrotate_point(m, center, self.st.angle);
                let ox = (win_w - self.st.img.w * scale) / 2.0 + self.st.pan.x;
                let oy = (win_h - self.st.img.h * scale) / 2.0 + self.st.pan.y;
                let inside = m.x >= ox
                    && m.x < ox + self.st.img.w * scale
                    && m.y >= oy
                    && m.y < oy + self.st.img.h * scale;
                let filled = ox <= 0.0
                    && oy <= 0.0
                    && ox + self.st.img.w * scale >= win_w
                    && oy + self.st.img.h * scale >= win_h;
                if !inside {
                    f_open_grid = self.grid.is_some();
                } else if filled {
                    self.st.zoom = ZoomMode::FitAll;
                    self.st.target_pan = Vec2::ZERO;
                } else {
                    f_zoom = true;
                }
            }
            if f_open_grid {
                enter = true;
            }
            // A host pane draws no grid, so `Enter` and `ESC` stay in the image
            // view and never take the picture off screen.
            if self.pane.shows_grid() && self.grid.is_some() && enter {
                return_to_grid = true;
            }
            if return_to_grid {
                // Leaving image view mid-drag: give the pointer back before
                // the the grid click handling needs a visible cursor.
                if self.pointer_captured {
                    ui.release_pointer();
                    ui.place_pointer(self.grab_pos);
                    self.pending_restore = Some(self.grab_pos);
                    self.restore_tries = 0;
                    self.pointer_captured = false;
                }
                self.f_capture = false;
                self.r_capture = false;
                // The region belongs to the image view.
                self.region_drag = None;
                self.st.selection = None;
                self.st.mode = Mode::Grid;
                // Hand the shown texture back to its the grid entry and make
                // that entry the the grid selection (nsxiv-like).
                put_back_view(
                    &mut self.grid,
                    &mut self.st.view_from_grid,
                    &mut self.st.view_tex,
                );
                if let Some(id) = self.st.open_id.take()
                    && let Some(g) = self.grid.as_mut()
                    && let Some(i) = g.index_of(id)
                {
                    g.selected = i;
                    // Scrolled (zoomed-in) grids: bring it back on screen.
                    g.ensure_visible(WinW::new(win_w), WinH::new(win_h));
                }
                self.st.loader = None; // cancels a still-running stream
                self.st.view_loading = false;
                self.st.close_crop();
            }

            // Prev/next while viewing (only with a the grid to navigate).
            // A ready texture swaps in the same frame.
            // A decode in flight is waited on (auto-swap when it lands).
            // Otherwise the streaming loader takes over.
            // The priority list at the top of the loop prefetches the
            // neighbors of the new entry.
            // The spreads of a book all sit in one figure, so a step always
            // moves to another file.
            if !return_to_grid && let Some(delta) = nav {
                let cur = self
                    .st
                    .open_id
                    .and_then(|id| self.grid.as_ref().and_then(|g| g.index_of(id)));
                if let Some(cur) = cur {
                    let n = self.grid.as_ref().unwrap().entries.len();
                    let t = cur as i64 + delta;
                    if t >= 0 && (t as usize) < n {
                        let j = t as usize;
                        put_back_view(
                            &mut self.grid,
                            &mut self.st.view_from_grid,
                            &mut self.st.view_tex,
                        );
                        show_entry(
                            &mut self.st,
                            self.grid.as_mut().unwrap(),
                            j,
                            ui,
                            r,
                            (WinW::new(win_w), WinH::new(win_h)),
                            true,
                        );
                    }
                }
            }

            // Keyboard shortcuts. Capital W and capital E arrive as W/E plus
            // shift.
            if ui.key_pressed(KeyCode::KeyW) {
                self.st.zoom = if shift {
                    ZoomMode::FitAll
                } else {
                    ZoomMode::FitDown
                };
                self.st.target_pan = Vec2::ZERO;
            } else if ui.key_pressed(KeyCode::KeyE) {
                self.st.zoom = if shift {
                    ZoomMode::FitHeight
                } else {
                    ZoomMode::FitWidth
                };
                self.st.target_pan = Vec2::ZERO;
            } else if ui.key_pressed(KeyCode::KeyT) || f_zoom {
                // t: two-state toggle, whole image visible (fit-all) against
                // window completely covered (fill).
                // This stays distinct for any image and window combination,
                // unlike a fit-width/fit-height cycle.
                // f (f_zoom) always zooms in: it only reaches here when the
                // cursor is on a not-yet-covering image.
                if f_zoom {
                    self.st.zoom = ZoomMode::Fill;
                } else if self.st.zoom == ZoomMode::Fill {
                    self.st.zoom = ZoomMode::FitAll;
                } else {
                    self.st.zoom = ZoomMode::Fill;
                }
                // Zoom-in lands with the image point under the mouse at the
                // window center, clamped so the window never shows
                // background.
                // Zoom-out recentres.
                if self.st.zoom == ZoomMode::Fill
                    && self.st.img.w > 0.0
                    && self.st.view_scale.is_some()
                {
                    self.st.target_pan = aim_fill_pan(&self.st, ui.pointer(), win_w, win_h);
                } else {
                    self.st.target_pan = Vec2::ZERO;
                }
            }

            // Before the header arrives the image dimensions are unknown.
            // The code skips all scale and pan math (it divides by
            // img_w/img_h).
            if self.st.img.w > 0.0 {
                // Scale for the current mode (fit modes recompute every
                // frame, so resizing stays correct).
                // A fit measures the opening box when the surface names one:
                // a book frames its first spread, so the cover page shows
                // whole and whichever side binds decides.
                let (fit_w, fit_h) = self
                    .st
                    .open_box
                    .map_or((self.st.img.w, self.st.img.h), |boxed| (boxed.w, boxed.h));
                let fit_scale = match self.st.zoom {
                    ZoomMode::FitDown => (win_w / fit_w).min(win_h / fit_h).min(1.0),
                    ZoomMode::FitAll => (win_w / fit_w).min(win_h / fit_h),
                    ZoomMode::FitWidth => win_w / fit_w,
                    ZoomMode::FitHeight => win_h / fit_h,
                    // Cover: the larger ratio wins, so the image fills the
                    // window and the code crops the other axis.
                    // A free zoom keeps its scale as the fit base.
                    ZoomMode::Fill | ZoomMode::Free(_) => (win_w / fit_w).max(win_h / fit_h),
                };

                // Held keys feed the ramps.
                // The ramps make a hold ease in to cruise speed and a release
                // ease out, so the motion never steps on the auto-repeat
                // rate.
                // The code reads input before begin_drawing borrows rl
                // mutably.
                let dt = ui.frame_time();
                // Ctrl + j/k zoom like `+`/`-` (Ctrl + Shift + j/k keeps
                // the Shift speed).
                // `j` and `k` also pan, so a held Ctrl turns them into zoom
                // keys instead of pan keys.
                let ctrl = ui.key_down(KeyCode::ControlLeft) || ui.key_down(KeyCode::ControlRight);
                let ctrl_zoom_in = ctrl && ui.key_down(KeyCode::KeyJ);
                let ctrl_zoom_out = ctrl && ui.key_down(KeyCode::KeyK);
                // Shift+J/K step a spread when the open surface names a
                // spread column, so those two keys do not pan while the step
                // can run.
                // Every other source, and a book whose surface has not landed
                // yet, keeps the doubled pan.
                let spread_keys = shift
                    && self
                        .st
                        .source
                        .as_ref()
                        .is_some_and(|s| s.spread_count() > 0);
                let nav = NavKeys {
                    left: ui.key_down(KeyCode::KeyH) || ui.key_down(KeyCode::ArrowLeft),
                    right: ui.key_down(KeyCode::KeyL) || ui.key_down(KeyCode::ArrowRight),
                    up: (ui.key_down(KeyCode::KeyK) && !ctrl_zoom_out && !spread_keys)
                        || ui.key_down(KeyCode::ArrowUp),
                    down: (ui.key_down(KeyCode::KeyJ) && !ctrl_zoom_in && !spread_keys)
                        || ui.key_down(KeyCode::ArrowDown),
                    // US layouts put the zoom keys on the `=` and `-`
                    // keycodes (numpad included).
                    // Other layouts put `+`/`-` on another key, whose keycode
                    // the typed character taught earlier.
                    // Polling that keycode reads the real held state, so a
                    // second key going down does not stop the zoom.
                    // Ctrl + j/k join them as a hold, so the ramp eases the
                    // same way.
                    zoom_in: ui.key_down(KeyCode::Equal)
                        || ui.key_down(KeyCode::NumpadAdd)
                        || ctrl_zoom_in
                        || self.zoom_keys.zoom_in_key().is_some_and(|k| ui.key_down(k)),
                    zoom_out: ui.key_down(KeyCode::Minus)
                        || ui.key_down(KeyCode::NumpadSubtract)
                        || ctrl_zoom_out
                        || self
                            .zoom_keys
                            .zoom_out_key()
                            .is_some_and(|k| ui.key_down(k)),
                };
                // The pan cruise speed scales with the window, so the motion
                // covers the same fraction of the view on any screen size.
                // Shift multiplies both the pan cruise speed and the zoom
                // rate.
                let motion = self
                    .st
                    .ramps
                    .tick(nav, win_w.max(win_h) * PAN_RATE, dt, shift_speed);
                // Fold the ramped delta into the pan and its target together,
                // so the pan follows the ramp at once and the ease below sees
                // no leftover distance.
                self.st.pan += motion.pan;
                self.st.target_pan += motion.pan;
                // A held `f` with Shift turns the vertical mouse travel into
                // the free zoom, so the pointer replaces the wheel with one
                // continuous rate.
                // Up zooms in and down zooms out.
                // The capture above keeps the pointer recentering, so the travel
                // never stops at a screen edge, like the pan drag.
                // The anchor starts at the grab point and then drifts toward the
                // window center by the same rule as the wheel anchor below, so
                // the zoom focus recenters while it runs.
                // The step comes from this frame's travel, so a held pointer
                // holds the scale.
                // The wheel below still zooms with no Shift.
                let delta = ui.travel();
                let zoom_drag = self.f_capture && self.f_zoom_capture;
                if zoom_drag {
                    // The grab edge seeds the anchor, and the drift carries it
                    // from frame to frame.
                    self.st.zoom_anchor = Some(*self.f_zoom_anchor.get_or_insert(self.grab_pos));
                    if delta.y != 0.0 {
                        // One e-fold of scale per ZOOM_DRAG_SPAN of travel,
                        // where the span scales with the window.
                        let span = win_w.max(win_h) * ZOOM_DRAG_SPAN;
                        let r = (-delta.y * ZOOM_DRAG_SPEED / span).exp();
                        let base = self.st.zoom.free_or(fit_scale);
                        self.st.zoom = ZoomMode::Free((base * r).clamp(0.01, 100.0));
                    }
                }
                // A held `r` turns the vertical mouse travel into the image
                // angle: down turns it clockwise and up turns it
                // counter-clockwise.
                // The capture keeps the pointer recentering, so the travel
                // never stops at a screen edge, like the pan drag.
                // Each frame adds this frame's travel to the current angle, so
                // a new hold continues from the angle the image shows and a
                // release keeps it.
                if self.r_capture {
                    // One full turn per ROTATE_DRAG_SPAN of travel, where the
                    // span scales with the window.
                    let span = win_w.max(win_h) * ROTATE_DRAG_SPAN;
                    let from = self.st.angle;
                    self.st.angle =
                        rotate::apply_turn(self.st.angle, delta.y * ROTATE_DRAG_SPEED, span);
                    // A pivot turns the image in place: the pan takes the shift
                    // that the plain centered turn would move the pivot by, so
                    // the point under the pivot keeps its window spot.
                    keep_pivot(&mut self.st, from, win_w, win_h);
                }

                // Free zoom: the held +/- keys scale the free-zoom factor
                // itself, with a ramped speed, so a hold zooms at a steady
                // eased rate and a release coasts to a stop.
                // The fit modes keep their own scale.
                // The on-screen scale takes the ramped factor directly: an
                // exponential ease on top would lag a target that moves every
                // frame, and that lag reads as stepping.
                let zoom_step = motion.zoom;
                if let Some(factor) = zoom_step {
                    let base = self.st.zoom.free_or(fit_scale);
                    self.st.zoom = ZoomMode::Free((base * factor).clamp(0.01, 100.0));
                    // Keyboard zoom centers on the picture, not the cursor.
                    // None falls back to the window center anchor.
                    self.st.zoom_anchor = None;
                }
                // The Shift+F drag rides the pointer, so its target moves
                // every frame: the ease must stay out of its way, like the
                // wheel step.
                let target_scale = self.st.zoom.free_or(fit_scale);

                // The exponential ease only carries a discrete jump (a fit
                // mode change, the wheel, the `f` aim) the rest of the way.
                // A held key already set the target to the on-screen value, so
                // the ease stays out of its way.
                let prev_scale = self.st.view_scale;
                let alpha = 1.0 - (-dt / EASE_TIME).exp();
                self.st.view_scale = Some(match self.st.view_scale {
                    None => target_scale,
                    Some(_) if resized => target_scale,
                    Some(_) if zoom_step.is_some() => target_scale,
                    Some(s) => {
                        let s = s + (target_scale - s) * alpha;
                        if (target_scale - s).abs() < target_scale * 0.001 {
                            target_scale
                        } else {
                            s
                        }
                    }
                });

                // A discrete zoom step (the wheel) replaces the target that
                // the held keys build, and anchors the ease at the cursor:
                // while the scale eases, the image point under the mouse
                // stays put.
                let wheel = ui.wheel();
                if wheel != 0.0 {
                    let factor = 1.25f32.powf(wheel);
                    self.st.zoom = ZoomMode::Free((target_scale * factor).clamp(0.01, 100.0));
                    self.st.zoom_anchor = Some(ui.pointer());
                }
                // Left-drag pans: the image follows the cursor (grab-style).
                // No easing while the cursor drives the pan.
                // The drag delta is already per-frame, and piling the 50 ms
                // pan ease on top of the vsync display latency reads as lag
                // (same-frame input, no interpolation: chart-action 9a5394b
                // lesson).
                // The keyboard panning below keeps its glide.
                // It only makes sense once the dimensions are known.
                //
                // Infinite drag: while the button is held the pointer is
                // captured, so it neither disappears at the screen edge nor
                // blocks there.
                // Panning continues with virtual deltas no matter how far
                // the physical mouse travels.
                // Mouse-up shows the pointer again at the position where the
                // drag began.
                let drag_pressed = ui.button_pressed(MouseButton::Left);
                let drag_released = ui.button_released(MouseButton::Left);
                if drag_pressed && !self.pointer_captured && !self.r_capture {
                    self.grab_pos = ui.pointer();
                    self.drag_virtual = self.grab_pos;
                    ui.grab_pointer();
                    self.pointer_captured = true;
                    if self.debug {
                        eprintln!(
                            "vv: drag grab at ({:.0},{:.0})",
                            self.grab_pos.x, self.grab_pos.y
                        );
                    }
                } else if drag_released
                    && self.pointer_captured
                    && !self.f_capture
                    && !self.r_capture
                {
                    if self.debug {
                        let travel = self.drag_virtual - self.grab_pos;
                        eprintln!(
                            "vv: drag release: grab ({:.0},{:.0}), virtual ({:.0},{:.0}), speed \
                             {}, pointer ({:.0},{:.0}), travel ({:.0},{:.0})",
                            self.grab_pos.x,
                            self.grab_pos.y,
                            self.drag_virtual.x,
                            self.drag_virtual.y,
                            self.pan_drag_speed,
                            ui.pointer().x,
                            ui.pointer().y,
                            travel.x,
                            travel.y
                        );
                    }
                    ui.release_pointer();
                    ui.place_pointer(self.grab_pos);
                    self.pending_restore = Some(self.grab_pos);
                    self.restore_tries = 0;
                    self.pointer_captured = false;
                }
                // Any captured pointer (left-drag or a held `f`) pans or zooms.
                // The virtual cursor tracks the raw travel the same way.
                // A Shift+F capture zooms instead: it tracks the travel and
                // leaves the pan alone.
                // An `r` capture turns the image instead: the travel it needs
                // is vertical, so the pan stays put.
                let dragging = self.pointer_captured;
                if dragging {
                    let delta = ui.travel();
                    // The virtual cursor follows physical travel 1:1 (self.debug
                    // crosshair).
                    self.drag_virtual.x += delta.x;
                    self.drag_virtual.y += delta.y;
                    // The pan follows the travel at the run's own factor.
                    if !self.f_zoom_capture && !self.r_capture {
                        self.st.target_pan.x += delta.x * self.pan_drag_speed;
                        self.st.target_pan.y += delta.y * self.pan_drag_speed;
                    }
                }

                // Mouse-anchored zoom (free zoom only): while the on-screen
                // scale eases, shift the pan each frame so the image point
                // under the anchor (the cursor when the step started, else
                // the window center) stays fixed.
                // offset = center + pan, so keeping the anchor image point
                // put gives
                //   offset' = anchor - (anchor - offset) * (scale'/scale).
                //
                // A continuous zoom (the Shift+F hold) replaces its target
                // every frame, so there is no ease to follow: the pan is
                // solved exactly.
                // It drifts like the wheel: after each frame anchor step, the
                // anchor slides a little toward the window center, so the
                // image point being zoomed into gently recenters.
                // The pointer stays captured, so the anchor moves on screen and
                // the image follows it, instead of the cursor riding along.
                // The held `+`/`-` keys center the picture outright and skip the
                // drift.
                // A zoom hold that starts at the window center needs no drift
                // either: its anchor already sits there.
                let continuous_zoom = self.f_capture && self.f_zoom_capture;
                let scale = self.st.view_scale.unwrap();
                if matches!(self.st.zoom, ZoomMode::Free(_))
                    && let Some(s_old) = prev_scale
                    && (scale - s_old).abs() > f32::EPSILON
                    && s_old > 0.0
                {
                    let center = Vec2 {
                        x: win_w / 2.0,
                        y: win_h / 2.0,
                    };
                    let anchor = self.st.zoom_anchor;
                    let a = anchor.unwrap_or(center);
                    let (ax, ay) = (a.x, a.y);
                    let keyboard_zoom = anchor.is_none();
                    let r = scale / s_old;
                    let ox = ax - (ax - (win_w - self.st.img.w * s_old) / 2.0 - self.st.pan.x) * r;
                    let oy = ay - (ay - (win_h - self.st.img.h * s_old) / 2.0 - self.st.pan.y) * r;
                    let new_x = ox - (win_w - self.st.img.w * scale) / 2.0;
                    let new_y = oy - (win_h - self.st.img.h * scale) / 2.0;
                    // A held key zoom skips the drift: it centers anyway.
                    // So does an anchor that already sits at the window
                    // center: the drift can only move it away from there.
                    let centered = a == center;
                    let d = if zoom_step.is_some() || centered {
                        Vec2::ZERO
                    } else {
                        // The drift shares the exponential time constant of
                        // the ease: by the time the scale covers its remaining
                        // distance, the pair covers ZOOM_ANCHOR_DRIFT of its
                        // own.
                        (center - a) * (alpha * ZOOM_ANCHOR_DRIFT)
                    };
                    // Move the pan and its target by the same amount.
                    // The anchor then holds without discarding the offset a
                    // held pan key added this frame.
                    let dx = new_x + d.x - self.st.pan.x;
                    let dy = new_y + d.y - self.st.pan.y;
                    self.st.pan.x += dx;
                    self.st.pan.y += dy;
                    self.st.target_pan.x += dx;
                    self.st.target_pan.y += dy;
                    if !keyboard_zoom {
                        self.st.zoom_anchor = Some(a + d);
                        if continuous_zoom {
                            // The drag anchor carries the drift into the next
                            // frame.
                            self.f_zoom_anchor = self.st.zoom_anchor;
                        }
                    }
                    // Ride the pointer along with the drifted image point
                    // (never while a drag has it captured, and keyboard zoom
                    // leaves the cursor wherever it is).
                    // A continuous zoom keeps the pointer captured instead,
                    // so this never runs for it.
                    if d != Vec2::ZERO && !keyboard_zoom && !self.pointer_captured {
                        if self.debug {
                            eprintln!("vv: zoom drift anchor to ({ax:.0},{ay:.0})");
                        }
                        ui.place_pointer(self.st.zoom_anchor.unwrap());
                    }
                }

                // Vim-style panning (h/j/k/l and arrow keys).
                // The held-key ramps above already moved the pan and its
                // target by this frame's ramped delta.
                // The exponential ease here only carries a discrete jump (a
                // fit-mode change, the fill aim, a mouse drag) the rest of the
                // way.
                // It must not chase a moving target: easing after a target
                // that advances every frame makes the motion lag its own
                // input, and the lag grows and then snaps, which reads as a
                // key press rather than a ramp.
                // Speed continuity is the ramp's job.
                //
                // A fresh open shows the top of the plane: the fit frames
                // the opening box, and that box starts at the top.
                self.st.align_open_box(win_h);
                // Shift+J/K or Page-down/Page-up: one spread per press, eased
                // to the top of the window.
                // The step runs after the align, and the align then sees a
                // target that it did not write, so it stays out of the way and
                // the view keeps the spread the step landed on.
                if let Some(step) = spread_step
                    && self.st.step_spread(step, win_h)
                {
                    // The landing spread is known now, so its patch paints on a
                    // worker while the glide runs.
                    self.st
                        .prefetch_spread(Vec2::new(win_w, win_h), &r.uploader());
                }
                // On a resize frame, snap pan too, so it does not glide after
                // the new fit offset (matches the scale snap above).
                if resized {
                    self.st.pan = self.st.target_pan;
                }
                // While dragging, the image sticks to the cursor (only the
                // zoom-anchor shift above touches pan).
                if dragging {
                    self.st.pan = self.st.target_pan;
                }
                // A spread step glides at [`SPREAD_EASE_TIME`].
                // Any other motion of the target ends the glide, and the
                // ordinary ease takes over again.
                let gliding = self
                    .st
                    .spread_glide
                    .is_some_and(|goal| (goal - self.st.target_pan).length() <= 0.5);
                if !gliding {
                    self.st.spread_glide = None;
                }
                let ease_time = if gliding { SPREAD_EASE_TIME } else { EASE_TIME };
                let pan_alpha = 1.0 - (-dt / ease_time).exp();
                self.st.pan.x += (self.st.target_pan.x - self.st.pan.x) * pan_alpha;
                self.st.pan.y += (self.st.target_pan.y - self.st.pan.y) * pan_alpha;
                // Snap when the residual glide is sub-pixel.
                if (self.st.target_pan.x - self.st.pan.x).abs() < 0.25 {
                    self.st.pan.x = self.st.target_pan.x;
                }
                if (self.st.target_pan.y - self.st.pan.y).abs() < 0.25 {
                    self.st.pan.y = self.st.target_pan.y;
                }
                // The step stops once the pan lands on the spread.
                if self.st.pan == self.st.target_pan {
                    self.st.spread_glide = None;
                }
                // `v` region selection: the press anchors at the picture point
                // under the pointer, the pointer moves the opposite corner, and
                // the release keeps the rect.
                // A press inside the kept region carries that rect instead, and
                // keeps its size, so `v` both spans and moves.
                // A press that releases without travel commits nothing: outside
                // the region it drops the rect the way an empty `v` click drops
                // the visual selection in Vim, and inside it keeps the rect.
                // The cursor stays free, and the rect lives in source units, so
                // a pan, a zoom and a turn keep the region on the picture while
                // the drag runs.
                let img = Vec2::new(self.st.img.w, self.st.img.h);
                let frame = self.st.frame(Vec2::new(win_w, win_h));
                if v_key
                    && self.region_drag.is_none()
                    && !self.pointer_captured
                    && let Some(frame) = frame
                {
                    let mouse = ui.pointer();
                    let drag = region_sel::press(
                        frame.to_image(mouse),
                        mouse,
                        img,
                        self.st.selection,
                        v_square_edge,
                    );
                    self.region_drag = Some(drag);
                    if self.debug {
                        eprintln!(
                            "vv: region grab at ({:.0},{:.0}) -> image ({:.0},{:.0}), {:?}",
                            mouse.x, mouse.y, drag.anchor.x, drag.anchor.y, drag.kind
                        );
                    }
                }
                // `!key_down` also covers a press and release that land
                // inside one frame, like the `f` and `r` halves.
                if let Some(drag) = self.region_drag
                    && !ui.key_down(KeyCode::KeyV)
                {
                    // The release reads the same rect the draw shows, so the
                    // committed region is the one on screen, and a tap commits
                    // nothing: it drops a span and keeps a moved rect.
                    self.st.selection = selection_rect(
                        self.st.frame(Vec2::new(win_w, win_h)),
                        None,
                        Some(drag),
                        ui.pointer(),
                    );
                    self.region_drag = None;
                    if self.debug {
                        eprintln!(
                            "vv: region release {:?}",
                            self.st.selection.map(|r| (r.x, r.y, r.w, r.h))
                        );
                    }
                }
            } // self.st.img.w > 0.0: fit/pan math needs known dimensions
        }

        // The visible-region patch of a vector file: adopt a parsed source and
        // a finished paint, then repaint once the view holds still.
        // The upload needs the main thread, and the request reads the scale
        // and the pan of this frame.
        if self.st.mode == Mode::Image {
            self.st.drain_crops();
            self.st
                .tick_crop(Instant::now(), Vec2::new(win_w, win_h), &r.uploader());
        }

        // VV_BLUR_BG in the grid mode: the background follows the selected
        // entry with a slow crossfade (starts as soon as the the grid workers
        // decode the blurred copy of the entry).
        if self.st.mode == Mode::Grid
            && let (Some(bg), Some(g)) = (self.st.blur_bg.as_mut(), self.grid.as_ref())
            && let Some(e) = g.entries.get(g.selected)
            && let Some(data) = &e.blur
        {
            bg.transition(&r.uploader(), data, e.id);
        }

        // The region the image view marks now, read before the draw borrows
        // `rl`.
        // A drag in flight answers a fresh rect from the pointer position, so
        // the veil and the border follow the cursor every frame.
        let selection = if self.st.mode == Mode::Image {
            selection_rect(
                self.st.frame(Vec2::new(win_w, win_h)),
                self.st.selection,
                self.region_drag,
                ui.pointer(),
            )
        } else {
            None
        };

        ui.begin(r);
        Ui::clear(r, BACKGROUND);

        if self.st.mode == Mode::Grid {
            // VV_BLUR_BG gimmick: the blurred background follows the
            // selection (crossfading, and a no-op when the gimmick is off).
            // The code draws it before the the grid, so the fades run under the
            // thumbnails.
            if let Some(bg) = &mut self.st.blur_bg {
                bg.draw(r, WinW::new(win_w), WinH::new(win_h));
            }
            self.grid
                .as_ref()
                .unwrap()
                .draw(r, (WinW::new(win_w), WinH::new(win_h)));
        } else {
            // VV_BLUR_BG gimmick: blurred copy of the image, scaled to
            // cover the whole window (fit on the narrower side, with the other
            // axis overflowing and cropped) behind the sharp image.
            if let Some(bg) = &mut self.st.blur_bg {
                bg.draw(r, WinW::new(win_w), WinH::new(win_h));
            }
            // The region patch covers the window on its own with the sharp
            // pixels of the visible part, so the whole-image texture draws only
            // where the patch cannot reach: the frames of a resize or a pan
            // that outran the repaint.
            // Both over the same pixels would double the alpha of an
            // antialiased edge.
            if !patch_covers(&self.st, win_w, win_h) {
                if let Some(texture) = &self.st.view_tex {
                    // The image turns around its own center on screen, which
                    // is the pan offset on top of the window middle.
                    let center =
                        Vec2::new(win_w / 2.0 + self.st.pan.x, win_h / 2.0 + self.st.pan.y);
                    draw_turned(
                        r,
                        texture,
                        self.st.img,
                        self.st.view_scale.unwrap(),
                        center,
                        self.st.angle,
                        view_filter(self.st.pixelated),
                    );
                } else if self.st.img.w > 0.0
                    && let Some(g) = self.grid.as_ref()
                    && let Some(id) = self.st.open_id
                    && let Some(i) = g.index_of(id)
                    && let Some(tex) = g
                        .entries
                        .get(i)
                        // The decoded thumb first, then its few-byte
                        // placeholder.
                        .and_then(|e| e.cell.texture())
                {
                    // No full-res frame yet (decode in flight): show the entry
                    // thumb instead of flashing black, or its ThumbHash
                    // placeholder while the thumb itself still decodes.
                    // Both hold the whole image with the aspect preserved, so
                    // they map 1:1 onto the image rect and the later swap to
                    // the full-res texture is pixel-aligned.
                    let scale = self
                        .st
                        .view_scale
                        .unwrap_or((win_w / self.st.img.w).min(win_h / self.st.img.h));
                    let center =
                        Vec2::new(win_w / 2.0 + self.st.pan.x, win_h / 2.0 + self.st.pan.y);
                    draw_turned(
                        r,
                        tex,
                        self.st.img,
                        scale,
                        center,
                        self.st.angle,
                        Filter::Linear,
                    );
                }
            }
            // The region patches draw turned and scaled like the image, so
            // their pixels land where the whole-image texture would put them.
            // A spread step paints the spread it lands on before the glide
            // ends, so that patch draws here while it arrives and the whole
            // image loses its blur where the patch already reached.
            // A patch that still matches the view scale lands on whole
            // window pixels (see `draw_patch`), so its text stays sharp.
            if let Some(frame) = self.st.frame(Vec2::new(win_w, win_h)) {
                for patch in self.st.crop.iter().chain(self.st.prefetch.held()) {
                    draw_patch(r, patch, &frame, view_filter(self.st.pixelated));
                }
            }
            // A region selection dims the picture outside it and marks the
            // rect with a bright border.
            // The bands draw over the whole image and the patches, so the dim
            // holds whatever the view paints below it.
            if let Some(selection) = selection
                && let Some(frame) = self.st.frame(Vec2::new(win_w, win_h))
            {
                draw_selection(r, frame, selection);
            }
            // VV_DEBUG: while the pointer is captured, draw a crosshair at
            // the virtual cursor position (clamped to the window).
            // The real cursor is hidden, and this shows where vv thinks it
            // is.
            // The code also draws a ring at the grab point, so a wrong
            // restore is visible.
            if self.debug && self.pointer_captured {
                let vx = self.drag_virtual.x.clamp(0.0, win_w);
                let vy = self.drag_virtual.y.clamp(0.0, win_h);
                r.circle(
                    Vec2::new(self.grab_pos.x, self.grab_pos.y),
                    6.0,
                    Color::SKYBLUE,
                );
                r.line_v(
                    Vec2::new(vx - 12.0, vy),
                    Vec2::new(vx + 12.0, vy),
                    Color::YELLOW,
                );
                r.line_v(
                    Vec2::new(vx, vy - 12.0),
                    Vec2::new(vx, vy + 12.0),
                    Color::YELLOW,
                );
            }
        }
        // Draw the batch and present it. A stale or lost swapchain skips this
        // frame and asks for a new configuration, so a resize never kills the
        // window.
        self.occluded = false;
        if let Err(err) = ui.end(r) {
            if err.needs_reconfigure() {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.reconfigure();
                }
            } else if matches!(err, FrameError::Occluded | FrameError::Timeout) {
                // No image came out of the swapchain, so the window is hidden:
                // the loop paces itself instead of spinning.
                self.occluded = true;
            } else {
                eprintln!("vv: frame: {err}");
            }
        }
        if let Some(started) = frame_started {
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            if ms > 50.0 {
                eprintln!("vv: slow frame {ms:.1} ms");
            }
        }
        Ok(())
    }
}

impl App {
    /// Take one keyboard event.
    ///
    /// A repeat is not an edge: the input layer keeps it apart, and a one-shot
    /// key reads only the edge.
    fn key(&mut self, event: &KeyEvent) {
        let PhysicalKey::Code(code) = event.physical_key else {
            return;
        };
        let text = event.text.as_ref().and_then(|text| text.chars().next());
        self.input.key(code, event.state, event.repeat, text);
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
    let launch = Launch::open(&cli.path, pane)?;
    let event_loop = EventLoop::new().context("cannot create the event loop")?;
    let mut app = App::new(launch);
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

    #[test]
    fn a_host_pane_opens_the_first_entry_of_a_directory() {
        let wid = Wid::parse("0x4a00007").expect("a window id");
        // A host pane draws no grid, so a directory opens its first entry.
        assert_eq!(open_initial(Pane::Embedded(wid), false, 12), Some(0));
        assert_eq!(open_initial(Pane::Embedded(wid), false, 1), Some(0));
        // Nothing to open in an empty directory or in a document launch.
        assert_eq!(open_initial(Pane::Embedded(wid), false, 0), None);
        assert_eq!(open_initial(Pane::Embedded(wid), true, 1), None);
        // A standalone window shows the grid, except for a one-image
        // directory, which it opens at once.
        assert_eq!(open_initial(Pane::Window, false, 12), None);
        assert_eq!(open_initial(Pane::Window, false, 1), Some(0));
        assert_eq!(open_initial(Pane::Window, false, 0), None);
        assert_eq!(open_initial(Pane::Window, true, 1), None);
    }
}
