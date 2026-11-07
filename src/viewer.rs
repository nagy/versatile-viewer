// Pixel and coordinate math lives in f32 (window and source units) and indices
// in usize and u32.
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

//! One viewer value: what it shows, how it frames it and how it draws it.
//!
//! A host builds a [`Viewer`] on its own device and draws it into its own
//! texture. A host owns the window, the surface and the event loop; the viewer
//! owns the pipeline, the textures, the view state, the grid, the loaders and
//! the clock. The binary in `main.rs` is such a host, and so is
//! `examples/embed_host.rs`.
//!
//! [`Viewer::new`] takes a [`Launch`], because the launch happens before a
//! window exists: the command line names a file or a directory, a directory
//! lists itself, a single file decodes, and the result is what sizes a window.
//! A host therefore opens the launch first, creates its window from
//! [`Launch::window_size`], and then builds the viewer on its device.
//!
//! The input vocabulary is [`Event`], one level above `input::Input`, so a host
//! maps its own toolkit into it and [`Viewer::handle_event`] keeps the held
//! state. Three window actions stay with the host (name the window, grab the
//! pointer, place the pointer), and they arrive through the [`Host`] seam, so
//! the viewer holds no window and no surface.
//!
//! [`Viewer::render`] fills one frame's batch and paints it into the texture
//! view the host passes, through `Renderer::render_into`. The host presents it,
//! which is the swapchain for the binary and a texture the host composites for
//! an embedding host.

use std::{
    borrow::Cow,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, mpsc::channel},
    time::Instant,
};

use anyhow::{Result, bail};
use winit::{
    event::{ElementState, MouseButton},
    keyboard::KeyCode,
};

use crate::{
    DecodedImage, TAP_SLACK,
    blurbg::{self, BlurBg},
    book::BookSurface,
    dim::{WinH, WinW},
    draw::{Color, Vec2},
    grid::{DecodeFailure, Grid, GridAction, LoadPlan},
    input::Input,
    keyrepeat::{self, RepeatState},
    loader::{DecodeMsg, Loader},
    region_sel,
    render::Renderer,
    rotate,
    surface::{Size, Surface, View, open_surface},
    texture::Filter,
    view::{
        BACKGROUND, EASE_TIME, Mode, NavKeys, OpenAlign, PAN_RATE, Prefetch, ROTATE_DRAG_SPAN,
        ROTATE_DRAG_SPEED, Ramps, SPREAD_EASE_TIME, ViewState, ZOOM_ANCHOR_DRIFT, ZOOM_DRAG_SPAN,
        ZOOM_DRAG_SPEED, ZoomMode, aim_fill_pan, attach_blur_bg, draw_patch, draw_selection,
        draw_turned, keep_pivot, pan_drag_speed, patch_covers, put_back_view, read_nav_mouse,
        selection_rect, show_entry, view_filter,
    },
    zoomkeys::TypedKeys,
};

/// What the viewer asks of the window its host owns.
///
/// The viewer holds no window, and these four calls are the whole platform
/// wall between it and one: the binary answers them from `winit`, and an
/// embedding host answers them from its own toolkit.
pub trait Host {
    /// Name the window after the open picture.
    fn set_title(&mut self, title: &str);

    /// Grab the pointer and hide it, so a drag has unbounded travel.
    fn grab_pointer(&mut self);

    /// Give the pointer back and show it again.
    fn release_pointer(&mut self);

    /// Put the pointer at a window point.
    fn place_pointer(&mut self, at: Vec2);
}

/// Grab the pointer and hide it, so a drag has unbounded travel.
///
/// The grab is the host's call, and the capture is the input layer's state, so
/// the two move together here and a caller cannot set one without the other.
fn grab_pointer(input: &mut Input, host: &mut dyn Host) {
    host.grab_pointer();
    input.capture(true);
}

/// Give the pointer back and show it again.
fn release_pointer(input: &mut Input, host: &mut dyn Host) {
    host.release_pointer();
    input.capture(false);
}

/// Put the pointer at a window point.
///
/// The logical pointer follows at once, so the next frame reads the spot the
/// warp sent it to instead of the travel that ran before it.
fn place_pointer(input: &mut Input, host: &mut dyn Host, at: Vec2) {
    host.place_pointer(at);
    input.set_cursor(at);
}

/// One input event, free of any window handle.
///
/// A host maps its toolkit into these, and [`Viewer::handle_event`] keeps the
/// held state. A repeat is not an edge: the flag says so, and a one-shot action
/// reads only a press that is not one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// A key went down or came up, with the character it typed, if any.
    Key {
        /// The physical key.
        key: KeyCode,
        /// Down or up.
        state: ElementState,
        /// The platform repeated the press.
        repeat: bool,
        /// The character the press produced.
        text: Option<char>,
    },
    /// The pointer is at this window point, in physical pixels.
    Pointer(Vec2),
    /// The raw device travel since the last event, in device counts.
    Travel(Vec2),
    /// A mouse button went down or came up.
    Button {
        /// The button.
        button: MouseButton,
        /// Down or up.
        state: ElementState,
    },
    /// The wheel moved by this many notches, positive away from the user.
    Wheel(f32),
    /// The window gained or lost the keyboard focus.
    Focus(bool),
}

/// What the launch decided before a window existed.
///
/// The command line, the directory listing and, for a single file, the decode
/// all run here: the title carries the image count, and a picture sizes its own
/// window. The GPU work waits for [`Viewer::new`], because a texture needs a
/// device.
pub struct Launch {
    /// The pane draws no grid, so the launch opens the image view alone.
    shows_grid: bool,
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
    /// Open the path the command line names, or a path a host names.
    ///
    /// `shows_grid` is the pane policy: a window that draws the grid keeps it
    /// behind the image view for `ESC`, and a host pane draws the image view
    /// only.
    ///
    /// # Errors
    ///
    /// The listing error of a directory, the open error of a source, or an
    /// error naming a path that is neither.
    pub fn open(path: &Path, shows_grid: bool) -> Result<Self> {
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
        // A single-plane launch renders its one plane, and a directory launch
        // decodes nothing here.
        let decoded = match (single_plane, single_surface.as_ref()) {
            (true, Some(surface)) => Some(surface.render(View::whole(0, 1.0))?),
            _ => None,
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
        let blur = match (&decoded, blurbg::Config::from_env()) {
            (Some(image), Some(config)) => Some(config.blur(image)),
            _ => None,
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
            shows_grid,
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

    /// The window size the picture asks for, in physical pixels.
    ///
    /// A host creates its window at this size before it builds the viewer, so a
    /// picture opens the size it is, and a directory opens at 1024x768.
    #[must_use]
    pub const fn window_size(&self) -> (u32, u32) {
        self.win0
    }

    /// The title the window starts with.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }
}

/// The entry a launch opens before the user presses anything, if any.
///
/// A window that draws the grid opens the only entry of a one-image directory
/// and shows the grid otherwise, so `ESC` has somewhere to go.
/// A host pane draws the image view only, so it opens the first entry of a
/// directory and never waits on a grid that nothing can show.
/// A document launch already carries its picture, so it opens nothing.
const fn open_initial(shows_grid: bool, single_plane: bool, entries: usize) -> Option<usize> {
    if single_plane || entries == 0 {
        return None;
    }
    if shows_grid && entries != 1 {
        return None;
    }
    Some(0)
}

/// The window title pieces shared by both launch modes.
const SEP: &str = " \u{2013} ";
const APP: &str = "versatile-viewer";

/// A path with `$HOME` collapsed to `~`.
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

/// The viewer: the pipeline, the two views and the latches of the frame loop.
///
/// Everything here needs a device, and nothing here needs a window: the host
/// owns the window, the surface and the event loop, and it answers the four
/// [`Host`] calls.
#[allow(clippy::struct_excessive_bools)]
pub struct Viewer {
    // --- the host's device ---
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,

    // --- what the viewer shows ---
    st: ViewState,
    grid: Option<Grid>,
    /// The pane draws no grid, so `ESC` and `Enter` never leave the image view.
    shows_grid: bool,

    // --- input ---
    input: Input,

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
    /// True while an image-view drag holds the pointer.
    ///
    /// This is the view's latch; [`Input::capture`] carries the same fact to
    /// the input layer, so the raw travel moves the pointer. The two move
    /// together, and only through [`Viewer::grab_pointer`],
    /// [`Viewer::release_pointer`] or the lost-focus arm.
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
    rep_nav_fwd: RepeatState,
    rep_nav_back: RepeatState,

    /// The title the host starts its window with, which the image view falls
    /// back to.
    launch_title: String,

    // --- clock ---
    /// The seconds since the viewer started, summed from the frame times the
    /// host reports.
    elapsed: f64,
}

impl Viewer {
    /// Build the pipeline on the host's device and upload the launch frame.
    ///
    /// `size` is the pane the host draws into, in physical pixels: the launch
    /// frame uploads and a one-image directory opens at once, which both need
    /// the window size a host already knows.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
        max_texture_side: u32,
        launch: Launch,
        size: (u32, u32),
    ) -> Self {
        let renderer = Renderer::new(&device, &queue, format, max_texture_side);
        let (crop_tx, crop_rx) = channel();
        let launch_title = launch.title.clone();
        let mut viewer = Self {
            device,
            queue,
            renderer,
            st: ViewState {
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
            grid: None,
            shows_grid: launch.shows_grid,
            input: Input::default(),
            title_open_id: None,
            quit: false,
            debug: std::env::var_os("VV_DEBUG").is_some(),
            timing: crate::timing_enabled(),
            spread_armed: true,
            last_win: None,
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
            rep_nav_fwd: RepeatState::default(),
            rep_nav_back: RepeatState::default(),
            launch_title,
            elapsed: 0.0,
        };
        viewer.start_launch(launch, size);
        viewer
    }

    /// Upload the launch frame and start the background passes.
    ///
    /// The decode ran before a window existed, so this is the first moment a
    /// texture can exist.
    fn start_launch(&mut self, launch: Launch, size: (u32, u32)) {
        let Launch {
            grid,
            decoded,
            source,
            open_box,
            blur,
            single_plane,
            ..
        } = launch;
        if self.grid.is_none() {
            self.grid = grid;
        }
        let mut decoded = decoded;
        if let Some(image) = decoded.take() {
            match self.renderer.upload(&image) {
                Ok(texture) => self.st.view_tex = Some(texture),
                Err(err) => eprintln!("vv: upload: {err}"),
            }
            self.st.adopt_source(source);
            self.st.img = Size {
                w: image.width().as_f32(),
                h: image.height().as_f32(),
            };
            // A book frames its first spread, a picture frames the whole plane.
            self.st.open_box = open_box;
            self.st.reset_view();
            if let Some(data) = blur {
                attach_blur_bg(
                    &mut self.st.blur_bg,
                    &self.renderer.uploader(),
                    Some(&data),
                    None,
                );
            }
        }
        // The grid listed its directory before the window existed; the decodes
        // and the placeholder pass upload their own textures, so they start
        // here.
        if let Some(grid) = self.grid.as_mut() {
            grid.start(self.renderer.uploader());
        }
        // A one-image directory skips the grid: jump straight into that image
        // (ESC still returns to the grid view).
        // A host pane draws no grid at all, so it opens the first entry.
        let open = open_initial(
            self.shows_grid,
            single_plane,
            self.grid.as_ref().map_or(0, |grid| grid.entries.len()),
        );
        if let Some(index) = open {
            let win = (WinW::new(size.0 as f32), WinH::new(size.1 as f32));
            show_entry(
                &mut self.st,
                self.grid.as_mut().expect("a grid with entries"),
                index,
                &self.renderer,
                win,
                true,
            );
        }
    }

    /// Take one input event.
    ///
    /// The window actions a lost focus needs arrive through `host`, so a host
    /// that moves the keyboard away mid-drag cannot strand a capture.
    pub fn handle_event(&mut self, event: Event, host: &mut dyn Host) {
        match event {
            Event::Key {
                key,
                state,
                repeat,
                text,
            } => self.input.key(key, state, repeat, text),
            Event::Pointer(position) => self.input.cursor(position),
            Event::Travel(delta) => self.input.motion(delta),
            Event::Button { button, state } => self.input.button(button, state),
            Event::Wheel(notches) => self.input.wheel_lines(notches),
            Event::Focus(false) => self.forget_focus(host),
            Event::Focus(true) => {}
        }
    }

    /// Drop every held key, button and drag, and give the pointer back.
    ///
    /// A lost focus never delivers the releases, so a hold would otherwise stay
    /// on after the window came back, and a capture would outlive its drag.
    fn forget_focus(&mut self, host: &mut dyn Host) {
        self.input.forget_held();
        self.input.capture(false);
        self.pointer_captured = false;
        self.f_capture = false;
        self.f_zoom_capture = false;
        self.r_capture = false;
        self.region_drag = None;
        host.release_pointer();
    }

    /// Advance every ease and paint one frame into `target`.
    ///
    /// `size` is the pane the host draws into, in physical pixels, and `dt` the
    /// seconds since the previous frame. The viewer sums those into its own
    /// clock, so a host owns the clock and the viewer owns the eases.
    ///
    /// # Errors
    ///
    /// The open error of a source that failed mid-frame, which a host with a
    /// grid to fall back to handles by staying on the grid.
    pub fn render(
        &mut self,
        host: &mut dyn Host,
        target: &wgpu::TextureView,
        size: (u32, u32),
        dt: f32,
    ) -> Result<()> {
        self.elapsed += f64::from(dt);
        self.renderer.begin(&self.queue, size);
        self.renderer.set_clear(BACKGROUND);
        let result = self.frame(host, size, dt);
        self.renderer.render_into(&self.device, &self.queue, target);
        // The frame edges belong to one frame: the held state and the pointer
        // stay, the pressed, released, typed and travel sets start empty.
        self.input.begin_frame();
        result
    }

    /// Open another source and show it in the pane.
    ///
    /// A host path, and later an MCP tool, calls this. The launch policy is the
    /// pane's own: a directory opens its first entry, and a pane that draws no
    /// grid never shows one.
    ///
    /// # Errors
    ///
    /// The same errors [`Launch::open`] answers.
    pub fn open(&mut self, path: &Path, size: (u32, u32)) -> Result<()> {
        let launch = Launch::open(path, self.shows_grid)?;
        self.st.loader = None;
        self.st.close_crop();
        self.grid = None;
        self.st.mode = Mode::Grid;
        self.start_launch(launch, size);
        Ok(())
    }

    /// The size of the open picture, in physical pixels.
    ///
    /// A host that fits a window to the picture reads this.
    #[must_use]
    pub fn image_size(&self) -> (u32, u32) {
        (
            self.st.img.w.round().max(0.0) as u32,
            self.st.img.h.round().max(0.0) as u32,
        )
    }

    /// Did the viewer ask to exit?
    #[must_use]
    pub const fn quit(&self) -> bool {
        self.quit
    }

    /// The title of the open picture, for a host that names its window.
    ///
    /// None while the grid is the open view: a host then keeps the launch
    /// title.
    #[must_use]
    pub fn title(&self) -> Option<String> {
        let id = self.st.open_id?;
        let grid = self.grid.as_ref()?;
        let entry = grid.entries.iter().find(|entry| entry.id == id)?;
        Some(image_title(entry.source.path()))
    }

    /// Run the frame: read the input, advance every ease and fill the batch.
    #[allow(clippy::too_many_lines)]
    fn frame(&mut self, host: &mut dyn Host, size: (u32, u32), dt: f32) -> Result<()> {
        let r = &mut self.renderer;

        // VV_TIMING: the frame started now.
        // The code reports it at the end of the frame when it ran long.
        let frame_started = self.timing.then(std::time::Instant::now);
        // VV_DEBUG: trace every key press this frame.
        if self.debug {
            for k in self.input.pressed_keys() {
                eprintln!("vv: PRESSED key {k:?}");
            }
            // A frame longer than a key tap can hide the whole press. The input
            // layer records every event, so the trace above still shows it, and
            // this line names the frame that swallowed it.
            let ft = dt;
            if ft > 0.1 {
                eprintln!("vv: slow frame {ft:.0} ms");
            }
        }
        let win_w = size.0 as f32;
        let win_h = size.1 as f32;
        // Post-drag restore: verify the warp landed where the code sent it.
        // If a competing re-warp (GLFW, WM, XWayland) moved it, snap again.
        // The code gives up after a few tries so a user moving the mouse is
        // never pinned.
        if let Some(target) = self.pending_restore {
            let cur = self.input.cursor_position();
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
                place_pointer(&mut self.input, host, target);
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
        if self.shows_grid && self.st.open_id != self.title_open_id {
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
            host.set_title(&title);
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
                if !self.shows_grid {
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
        let shift = self.input.is_key_down(KeyCode::ShiftLeft)
            || self.input.is_key_down(KeyCode::ShiftRight);
        // Shift + h/j/k/l moves at double speed in both views.
        // The the grid passes the step count to its input handler, and the image
        // view scales its pan and zoom ramps.
        let shift_mult = if shift {
            crate::SHIFT_SPEED_MULTIPLIER
        } else {
            1
        };
        let shift_speed = shift_mult as f32;

        if self.st.mode == Mode::Grid {
            // Grid navigation: h/j/k/l and the arrows move the selection,
            // Enter opens the selected image, q quits (ESC is inert here,
            // because the the grid is the home view).
            match self.grid.as_mut().unwrap().handle_input(
                &self.input,
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
            for k in self.input.pressed_keys() {
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
                self.grab_pos = self.input.cursor_position();
                self.drag_virtual = self.grab_pos;
                grab_pointer(&mut self.input, host);
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
                self.grab_pos = self.input.cursor_position();
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
                && !self.input.is_key_down(KeyCode::KeyF)
                && !self.input.is_key_down(KeyCode::KeyR)
            {
                // Fold in this frame's motion before the lock is released
                // (the drag block below no longer runs once capture ends).
                let delta = self.input.delta();
                self.drag_virtual.x += delta.x;
                self.drag_virtual.y += delta.y;
                let f_dragged = (self.drag_virtual - self.grab_pos).length() > TAP_SLACK;
                if self.pointer_captured {
                    release_pointer(&mut self.input, host);
                    place_pointer(&mut self.input, host, self.grab_pos);
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
            let now = self.elapsed;
            let (delay, rate) = keyrepeat::settings();
            let mouse_nav = read_nav_mouse(&self.input);
            let next_edge = nav_next_edge || mouse_nav.next.edge;
            let prev_edge = nav_prev_edge || mouse_nav.prev.edge;
            // A held Shift moves the Space hold to the previous direction,
            // so a hold repeats the way its press edge stepped.
            let space_held = self.input.is_key_down(KeyCode::Space);
            let next_held = self.input.is_key_down(KeyCode::KeyN)
                || (space_held && !shift)
                || mouse_nav.next.held;
            let prev_held = self.input.is_key_down(KeyCode::KeyP)
                || self.input.is_key_down(KeyCode::Backspace)
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
            for typed in self.input.typed() {
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
            self.spread_armed = !(self.input.is_key_down(KeyCode::KeyJ)
                || self.input.is_key_down(KeyCode::KeyK)
                || self.input.is_key_down(KeyCode::PageDown)
                || self.input.is_key_down(KeyCode::PageUp));

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
            if self.shows_grid && self.grid.is_some() && enter {
                return_to_grid = true;
            }
            if return_to_grid {
                // Leaving image view mid-drag: give the pointer back before
                // the the grid click handling needs a visible cursor.
                if self.pointer_captured {
                    release_pointer(&mut self.input, host);
                    place_pointer(&mut self.input, host, self.grab_pos);
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
                            r,
                            (WinW::new(win_w), WinH::new(win_h)),
                            true,
                        );
                    }
                }
            }

            // Keyboard shortcuts. Capital W and capital E arrive as W/E plus
            // shift.
            if self.input.is_key_pressed(KeyCode::KeyW) {
                self.st.zoom = if shift {
                    ZoomMode::FitAll
                } else {
                    ZoomMode::FitDown
                };
                self.st.target_pan = Vec2::ZERO;
            } else if self.input.is_key_pressed(KeyCode::KeyE) {
                self.st.zoom = if shift {
                    ZoomMode::FitHeight
                } else {
                    ZoomMode::FitWidth
                };
                self.st.target_pan = Vec2::ZERO;
            } else if self.input.is_key_pressed(KeyCode::KeyT) || f_zoom {
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
                    self.st.target_pan =
                        aim_fill_pan(&self.st, self.input.cursor_position(), win_w, win_h);
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
                // Ctrl + j/k zoom like `+`/`-` (Ctrl + Shift + j/k keeps
                // the Shift speed).
                // `j` and `k` also pan, so a held Ctrl turns them into zoom
                // keys instead of pan keys.
                let ctrl = self.input.is_key_down(KeyCode::ControlLeft)
                    || self.input.is_key_down(KeyCode::ControlRight);
                let ctrl_zoom_in = ctrl && self.input.is_key_down(KeyCode::KeyJ);
                let ctrl_zoom_out = ctrl && self.input.is_key_down(KeyCode::KeyK);
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
                    left: self.input.is_key_down(KeyCode::KeyH)
                        || self.input.is_key_down(KeyCode::ArrowLeft),
                    right: self.input.is_key_down(KeyCode::KeyL)
                        || self.input.is_key_down(KeyCode::ArrowRight),
                    up: (self.input.is_key_down(KeyCode::KeyK) && !ctrl_zoom_out && !spread_keys)
                        || self.input.is_key_down(KeyCode::ArrowUp),
                    down: (self.input.is_key_down(KeyCode::KeyJ) && !ctrl_zoom_in && !spread_keys)
                        || self.input.is_key_down(KeyCode::ArrowDown),
                    // US layouts put the zoom keys on the `=` and `-`
                    // keycodes (numpad included).
                    // Other layouts put `+`/`-` on another key, whose keycode
                    // the typed character taught earlier.
                    // Polling that keycode reads the real held state, so a
                    // second key going down does not stop the zoom.
                    // Ctrl + j/k join them as a hold, so the ramp eases the
                    // same way.
                    zoom_in: self.input.is_key_down(KeyCode::Equal)
                        || self.input.is_key_down(KeyCode::NumpadAdd)
                        || ctrl_zoom_in
                        || self
                            .zoom_keys
                            .zoom_in_key()
                            .is_some_and(|k| self.input.is_key_down(k)),
                    zoom_out: self.input.is_key_down(KeyCode::Minus)
                        || self.input.is_key_down(KeyCode::NumpadSubtract)
                        || ctrl_zoom_out
                        || self
                            .zoom_keys
                            .zoom_out_key()
                            .is_some_and(|k| self.input.is_key_down(k)),
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
                let delta = self.input.delta();
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
                let wheel = self.input.wheel_move();
                if wheel != 0.0 {
                    let factor = 1.25f32.powf(wheel);
                    self.st.zoom = ZoomMode::Free((target_scale * factor).clamp(0.01, 100.0));
                    self.st.zoom_anchor = Some(self.input.cursor_position());
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
                let drag_pressed = self.input.is_button_pressed(MouseButton::Left);
                let drag_released = self.input.is_button_released(MouseButton::Left);
                if drag_pressed && !self.pointer_captured && !self.r_capture {
                    self.grab_pos = self.input.cursor_position();
                    self.drag_virtual = self.grab_pos;
                    grab_pointer(&mut self.input, host);
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
                            self.input.cursor_position().x,
                            self.input.cursor_position().y,
                            travel.x,
                            travel.y
                        );
                    }
                    release_pointer(&mut self.input, host);
                    place_pointer(&mut self.input, host, self.grab_pos);
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
                    let delta = self.input.delta();
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
                        place_pointer(&mut self.input, host, self.st.zoom_anchor.unwrap());
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
                    let mouse = self.input.cursor_position();
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
                    && !self.input.is_key_down(KeyCode::KeyV)
                {
                    // The release reads the same rect the draw shows, so the
                    // committed region is the one on screen, and a tap commits
                    // nothing: it drops a span and keeps a moved rect.
                    self.st.selection = selection_rect(
                        self.st.frame(Vec2::new(win_w, win_h)),
                        None,
                        Some(drag),
                        self.input.cursor_position(),
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
                self.input.cursor_position(),
            )
        } else {
            None
        };

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
        if let Some(started) = frame_started {
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            if ms > 50.0 {
                eprintln!("vv: slow frame {ms:.1} ms");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::open_initial;

    #[test]
    fn a_pane_that_draws_no_grid_opens_the_first_entry() {
        // A pane that draws no grid opens entry 0 of any directory.
        assert_eq!(open_initial(false, false, 12), Some(0));
        assert_eq!(open_initial(false, false, 1), Some(0));
        // Nothing to open in an empty directory or in a document launch.
        assert_eq!(open_initial(false, false, 0), None);
        assert_eq!(open_initial(false, true, 1), None);
        // A window that draws the grid opens only a one-image directory,
        // so `ESC` still has a grid to return to.
        assert_eq!(open_initial(true, false, 12), None);
        assert_eq!(open_initial(true, false, 1), Some(0));
        assert_eq!(open_initial(true, false, 0), None);
        assert_eq!(open_initial(true, true, 1), None);
    }
}
