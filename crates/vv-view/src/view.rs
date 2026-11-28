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

//! The image view and the grid, as the frame body drives them.
//!
//! `ViewState` is what the image view shows and how it frames it: the plane,
//! the scale, the pan, the turn, the region, the patch that refines a settled
//! zoom, and the loader of the open source.
//! The frame body in [`crate::viewer`] drives it: the view holds no window and
//! no device, so a host owns both.
//! The helpers here are the pure geometry and policy of that view: `Column`
//! for the spread math, `patch_dest` and `draw_patch` for the region patch,
//! `selection_rect` and `draw_selection` for the `v` region, and the
//! `frame`/`to_screen` mapping of [`region`].

use std::{
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender},
    },
    time::{Duration, Instant},
};

use winit::event::MouseButton;

use crate::{
    DIM_VEIL, DecodedImage,
    blurbg::BlurBg,
    dim::{self, PreviewPx, Rect, WinH, WinW},
    display::Request,
    draw::{Color, Quad, Turn, Vec2},
    grid::{DecodeFailure, EntrySource, Grid},
    input::Input,
    loader::Loader,
    ms_since,
    region::{self, Crop},
    region_sel,
    render::Renderer,
    rotate,
    scene::Primitive,
    surface::{Size, Surface, View, open_book_surface},
    texture::{Filter, Texture, Uploader},
    tween::Ramp,
};

pub const BACKGROUND: Color = Color::new(0x06, 0x23, 0x29, 0xff);

/// The border that marks a region selection.
/// It stays bright over the veil, so a region reads on a dark picture too.
pub const SELECTION_BORDER: Color = Color::new(0xff, 0xff, 0xff, 0xff);

/// The width of the region border, in window pixels.
pub const SELECTION_BORDER_PX: f32 = 2.0;

/// The fraction of the remaining distance to the window center that the
/// zoomed image point (and the cursor riding it) drifts over one zoom ease.
/// It shares the exponential time constant of the ease.
pub const ZOOM_ANCHOR_DRIFT: f32 = 0.25;

/// The time a held-key motion ramps in or out.
/// A hold reaches cruise speed after this time, and a release coasts to rest
/// over it.
pub const RAMP_TIME: Duration = Duration::from_millis(360);

/// How far one unit of raw device travel moves the picture during a drag.
///
/// A grab drag reads the raw device deltas of `DeviceEvent::MouseMotion`, which
/// are counts of the mouse, not window pixels, so this factor is a feel
/// constant and not a geometry one.
/// 1.0 moves the picture one window pixel per count, which is the default feel.
/// `VV_PAN_DRAG_SPEED` overrides it, so a feel comparison needs no rebuild, and
/// `RUST_LOG=vv=debug` prints the travel a drag accumulated (see the release
/// event).
pub const PAN_DRAG_SPEED: f32 = 1.0;

/// The accepted `VV_PAN_DRAG_SPEED` range.
///
/// Zero would freeze a drag and a huge value would throw the picture out of the
/// window on the first event, so the parser clamps instead of trusting the
/// value.
pub const PAN_DRAG_SPEED_RANGE: std::ops::RangeInclusive<f32> = 0.05..=16.0;

/// The drag factor for this run, from `VV_PAN_DRAG_SPEED` or the default.
///
/// The environment is read here, once, at the boundary, and the app carries the
/// value afterwards.
#[must_use]
pub fn pan_drag_speed() -> f32 {
    std::env::var("VV_PAN_DRAG_SPEED")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .map_or(PAN_DRAG_SPEED, |value| {
            value.clamp(*PAN_DRAG_SPEED_RANGE.start(), *PAN_DRAG_SPEED_RANGE.end())
        })
}

/// The pan cruise speed, in window long sides per second.
/// 0.9 suits a deliberate scan of the image.
/// The ramp scales it down near the press and the release.
pub const PAN_RATE: f32 = 0.9;

/// The free-zoom cruise rate, in natural-log e-folds per second.
/// A full-speed second covers about 5x.
/// The ramp scales the rate down near the press and the release.
pub const ZOOM_RATE: f32 = 1.65;

/// The vertical travel that a zoom hold needs for one natural-log e-fold, in
/// window long sides.
///
/// The hold is `Shift+F` or the right button, and both read this span.
/// Up (a negative y delta) zooms in, and down zooms out.
/// 1.5 suits fine aiming: a screen height of travel zooms one e-fold (2.7x),
/// so a small drag is a small change.
pub const ZOOM_DRAG_SPAN: f32 = 1.5;

/// A multiplier on the zoom hold's rate alone, `Shift+F` and the right button
/// alike.
/// 8.0 makes a hold eight times as fast as the raw travel suggests.
pub const ZOOM_DRAG_SPEED: f32 = 8.0;

/// The vertical travel that a held `r` needs for one full turn, in window long
/// sides.
/// Down turns clockwise and up turns counter-clockwise.
/// 1.0 gives a full turn over one screen of travel, so a small drag is a small
/// turn.
pub const ROTATE_DRAG_SPAN: f32 = 1.0;

/// A multiplier on the `r` rotation rate alone.
/// 1.0 maps the raw travel straight to the angle.
pub const ROTATE_DRAG_SPEED: f32 = 1.0;

/// The exponential ease time constant for the on-screen pan and scale, in
/// seconds.
/// About 95% of the remaining difference closes after three of them.
pub const EASE_TIME: f32 = 0.05;

/// The exponential ease time constant of a spread step, in seconds.
///
/// A Shift+J/K step moves the view by a whole spread, so it glides at its own
/// pace instead of jumping: five time constants land the spread after a
/// twentieth of a second, and the motion reads as a scroll.
pub const SPREAD_EASE_TIME: f32 = 0.05 * 1.0;

/// How long a spread step waits for its prefetched patch before the view
/// paints the region itself.
///
/// A worker that panics sends no message, so the wait cannot be unbounded, and
/// three seconds sits far past any paint the viewer has shown.
pub const PREFETCH_GIVE_UP: Duration = Duration::from_secs(3);

/// Target pan that centers the fill view on the image point under `mouse`.
///
/// The code clamps both axes to `[-|offset|, |offset|]`.
/// Outside that interval a window edge exposes background.
/// A cursor near the image edge therefore pans only as far as coverage
/// allows.
/// The answer is the user's pan and not the placement of the plane: a stated
/// viewport places itself (see [`ViewState::view_pan`]), so that part is taken
/// out here and added back by the frame.
/// A view that states such a box already fills the window on both axes, so the
/// caller never aims this one there: the box is its own fit.
#[must_use]
pub fn aim_fill_pan(st: &ViewState, mouse: Vec2, win_w: f32, win_h: f32) -> Vec2 {
    let target = region::Scale::uniform((win_w / st.img.w).max(win_h / st.img.h));
    let cur = st.view_scale.unwrap_or(target);
    // The offset is where the draw places the plane, which is the stated
    // viewport's placement plus the user's pan.
    let offset = st.offset(cur);
    // Mouse position in image coordinates at the current scale.
    // The offset is the window top-left of the unpanned image.
    let img_x = (mouse.x - (win_w - st.img.w * cur.x) / 2.0 - offset.x) / cur.x;
    let img_y = (mouse.y - (win_h - st.img.h * cur.y) / 2.0 - offset.y) / cur.y;
    let off_x = (win_w - st.img.w * target.x) / 2.0;
    let off_y = (win_h - st.img.h * target.y) / 2.0;
    let placed = st.view_pan(target);
    Vec2 {
        x: (win_w / 2.0 - off_x - img_x * target.x).clamp(-off_x.abs(), off_x.abs()) - placed.x,
        y: (win_h / 2.0 - off_y - img_y * target.y).clamp(-off_y.abs(), off_y.abs()) - placed.y,
    }
}

/// The furthest a free zoom may pull back, as a fraction of the fit scale.
///
/// The band is relative because a factor counts output pixels per surface
/// unit, and a surface unit is the plane's own: a document that measures its
/// plane in slot indices or prices fits at a factor a picture never shows, and
/// an absolute bound would turn the first wheel step into a jump.
pub const ZOOM_OUT: f32 = 0.01;

/// The furthest a free zoom may push in, as a multiple of the fit scale.
pub const ZOOM_IN: f32 = 100.0;

/// How the code scales the image to the window.
/// The code recomputes the scale every frame, so resizing always stays
/// correct.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZoomMode {
    /// Fit down to the window, centered.
    /// The code never upscales here (default).
    FitDown,
    /// Fit all sides.
    /// The code scales up or down until the image first touches a border
    /// (Shift+W).
    FitAll,
    /// Fit to the window width (e).
    FitWidth,
    /// Fit to the window height (Shift+E).
    FitHeight,
    /// Fill the window.
    /// The code scales until the image covers every side (t).
    /// The shorter relative dimension overflows and is cropped.
    Fill,
    /// The free zoom factor, set with +/- as multiples of the last fit scale.
    ///
    /// The factor is one number on both axes, because a zoom magnifies the view
    /// it is applied to: a stated viewport that fills the window keeps its own
    /// ratio between the axes and this number scales the pair.
    Free(region::Scale),
}

impl ZoomMode {
    /// The free scale, or the fit scale when the mode is not free.
    #[must_use]
    pub fn free_or(self, fit: region::Scale) -> region::Scale {
        match self {
            Self::Free(scale) => scale,
            Self::FitDown | Self::FitAll | Self::FitWidth | Self::FitHeight | Self::Fill => fit,
        }
    }
}

/// The viewer screen: thumbnail grid or single image.
#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Grid,
    Image,
}

/// The image-view state: what the code shows and how it frames it.
// The view holds several independent flags by nature: a load in flight, a
// paint in flight, the filter choice and a source still opening. Each one
// drives a different frame decision, so one enum would only blur them.
#[allow(clippy::struct_excessive_bools)]
pub struct ViewState {
    pub mode: Mode,
    /// The grid entry open in image mode, by stable id (None on a single-file
    /// launch).
    pub open_id: Option<u64>,
    /// The id of the grid entry whose full-res texture the view holds now.
    /// The texture goes back when the code leaves image mode or switches
    /// entries.
    pub view_from_grid: Option<u64>,
    /// The texture shown in image mode now.
    pub view_tex: Option<Texture>,
    pub view_loading: bool,
    /// The full-resolution plane extent, in surface units (0 x 0 until
    /// known).
    /// One `Size` names both axes, so no site can set or read them apart.
    pub img: Size,
    pub zoom: ZoomMode,
    /// The on-screen pan offset, eased toward `target_pan` each frame.
    ///
    /// The offset is the user's: a view that states the box its window shows
    /// places that box itself (see [`Self::view_pan`]), so this number moves
    /// the view off the stated placement and never carries it.
    pub pan: Vec2,
    pub target_pan: Vec2,
    /// The on-screen scale, eased toward the target scale each frame.
    pub view_scale: Option<region::Scale>,
    /// The plane box a producer stated its window shows, or `None` when the
    /// view frames the whole plane.
    ///
    /// A display list in the producer's own units is written for a viewport
    /// rather than for a window: the box names the part of the plane the window
    /// shows, and the view fills the window with it on each axis at its own
    /// ratio. The box is the fit then (see [`Self::viewport_scale`]), so it
    /// survives a resize, and the user's pan and zoom move off it.
    pub viewport: Option<region::Crop>,
    /// The held-key ramps that drive continuous motion.
    /// Each held key owns one ramp, so the speed eases in, holds, and eases
    /// out instead of stepping on the auto-repeat rate.
    pub ramps: Ramps,
    /// The window-space point the running zoom eases around.
    /// (+/- or wheel).
    /// None falls back to the window center, for example when an ease still
    /// runs from an earlier step.
    pub zoom_anchor: Option<Vec2>,
    /// The image rotation, in degrees clockwise.
    /// A held `r` drags it, and the code keeps it across a reload of the same
    /// image.
    /// The release snaps it with `rotate::snap_deg`, so a near quarter turn
    /// lands exactly.
    /// The drag itself never snaps.
    pub angle: f32,
    /// The window point that a Shift+R hold turns the image around.
    /// None turns around the image center, which is what plain `r` does.
    /// The hold moves the pan through `rotate::pivot`, so the point stays on
    /// its window spot at every angle.
    pub pivot: Option<Vec2>,
    /// The region the user selected with `v`, in source units (the units of
    /// [`region`]).
    /// A pan, a zoom and a turn keep it on the picture, because the draw maps
    /// its corners through the same transform the picture uses.
    /// A fresh open, a step to another image, a return to the grid and an
    /// empty `v` click all drop it.
    pub selection: Option<region::Crop>,
    /// `a` toggles nearest-neighbor filtering (pixelated) for pixel peeping.
    /// The default is smooth (bilinear).
    pub pixelated: bool,
    /// The streaming loader for the open image.
    /// A drop cancels the worker.
    pub loader: Option<Loader>,
    /// The `VV_BLUR_BG` background (None when the gimmick is off).
    /// The code declares it after `rl` (via `ViewState`) so it drops and
    /// unloads before the window.
    pub blur_bg: Option<BlurBg>,
    /// The source of the open image, so the view can repaint the region its
    /// window shows.
    /// None for a raster format, which has no more detail to give.
    pub source: Option<Arc<dyn Surface>>,
    /// The plane the running loader was started for, from its header, or `None`
    /// while no loader announces one.
    ///
    /// A loader reads the plane at the moment it starts, so a source that laid
    /// itself out again while the worker ran (a treemap) answers a plane the
    /// worker never saw, and its frame would carry the layout of the frame
    /// before it.
    pub loader_plane: Option<Size>,
    /// Whether the whole-plane texture on screen measures a plane the source
    /// reports no more.
    ///
    /// A treemap lays itself out again for a new window (see
    /// [`Surface::frame_window`]), so the texture of the old frame would draw
    /// its layout beside the layout of this plane wherever the region patch
    /// cannot reach the window.
    /// [`Self::refresh_plane`] paints the plane again once the window holds
    /// still, and clears this flag when it asks for the paint.
    pub plane_stale: bool,
    /// Whether the frame before this one drew the display list of the open
    /// source.
    ///
    /// A document that answers a list draws the whole picture from it, at any
    /// zoom and sharper than a patch (see `crate::viewer`), so the region
    /// patches have nothing to paint while it does.
    /// The flag is what tells them so; the fallback of a frame the list refuses
    /// lands after [`region::QUIET`] the moment it clears.
    pub list_drawn: bool,
    /// The plane of the open source that the view shows.
    /// It is 0 for every source the viewer opens: a book stacks its spreads
    /// into one plane (see [`vv_view::book`]).
    pub source_plane: usize,
    /// The part of the open plane that a fresh open frames, in surface units
    /// (see [`Surface::open_box`]).
    /// A book answers its first spread, so the cover page shows whole; every
    /// other source answers `None` and the fit frames the whole plane.
    pub open_box: Option<Size>,
    /// The slot page 0 fills inside the open plane, in surface units, or `None`
    /// for a source that shows one picture and no page geometry (see
    /// [`Surface::page_box`]).
    ///
    /// The grid cell of a document holds page 0 and never the whole book, so
    /// the view draws that picture here while the loader paints the spread,
    /// instead of stretching one page over every spread.
    pub cell_box: Option<region::Crop>,
    /// Whether the align still holds the opening box at the top of the
    /// window.
    ///
    /// The fit centers the plane, so the align scrolls to the box and keeps it
    /// there until the user moves the view.
    /// A window manager often resizes the window right after it maps, and a
    /// resize re-fits the scale, so the align repeats every frame instead of
    /// running once.
    pub open_align: OpenAlign,
    /// The pan a Shift+J/K spread step glides to.
    /// A step eases at [`SPREAD_EASE_TIME`], which now matches the ordinary
    /// pan ease, so the glide reads as a scroll of one spread.
    /// The field clears itself when the pan lands or the target moves away
    /// from it, so any other motion uses the ordinary ease again.
    pub spread_glide: Option<Vec2>,
    /// The prefetch of a spread step: the patch a worker paints, or the patch
    /// that landed and draws in place until the glide does.
    ///
    /// The step knows where it lands, so the patch of the target spread is on
    /// its way before the glide ends, and the view then holds a sharp spread
    /// the moment the pan arrives instead of a stretched one.
    /// A patch replaces [`ViewState::crop`] once it covers the window.
    pub prefetch: Prefetch,
    /// The newest prefetch token, so a result of an earlier aim drops.
    /// The counter stays beside the state, because it only ever grows.
    pub prefetch_seq: u64,
    /// The region patch on screen: the visible part of the document, painted
    /// at the scale the window shows it.
    /// None until the first paint lands, and then the whole-image texture in
    /// `view_tex` takes over only where the patch cannot reach.
    pub crop: Option<CropState>,
    /// Whether a worker paints a patch now.
    /// One paint in flight, because resvg cannot cancel one mid-paint.
    pub crop_pending: bool,
    /// Bumped on every open, every n/p step and every return to the grid, so
    /// the result of a worker for an earlier image drops instead of landing.
    pub crop_gen: u64,
    /// The region and scale the window needed when the debounce last saw it
    /// move.
    pub crop_seen: Option<region::Patch>,
    /// When the view last moved.
    pub crop_quiet_at: Instant,
    /// The region workers talk back over this pair.
    pub crop_tx: Sender<CropMsg>,
    pub crop_rx: Receiver<CropMsg>,
    /// The display list of the open plane, when the source answers one.
    ///
    /// A list draws at any zoom and under any turn, so a source that answers
    /// one needs no region patch: the picture *is* the list. The texture in
    /// `view_tex` and the patch in `crop` then stay empty, and the draw stops
    /// reading them.
    /// None for every source that answers a raster, which keeps the region
    /// patch and the whole-plane texture as the path those sources take.
    pub scene: Option<Arc<[Primitive]>>,
    /// The box of the plane that [`Self::scene`] answers, or `None` for a list of
    /// the whole plane.
    ///
    /// A source that lists one box at a time (a book, see
    /// [`Surface::display_box`]) answers the ink of that box and nothing else, so
    /// the view asks again once the window's own box leaves this one. A source
    /// that lists a whole plane answers the same list for every box, so this
    /// stays `None` and one build serves the plane.
    pub scene_box: Option<Crop>,
    /// Whether the next frame should ask for a list with the box it shows.
    ///
    /// A list of one box needs the window, which a caller that adopts a source
    /// does not have: the adopt asks for the list and the next frame, which shows
    /// the window, hands it the box (see [`Self::request_box_scene`]).
    pub scene_wanted: bool,
    /// Whether a worker builds the list of the plane the view holds now.
    pub scene_pending: bool,
}

/// The visible part of the open image, painted at the scale the window shows
/// it.
///
/// The patch carries its own texture, so the whole-image texture stays as the
/// fallback of a pan or a resize that outruns the repaint.
pub struct CropState {
    /// The pixels of the patch, already uploaded by the worker that painted
    /// them.
    pub texture: Texture,
    /// The source rect the patch covers and the scale it holds, so a paint
    /// rule reads one value instead of a rect beside a loose scale.
    pub patch: region::Patch,
}

/// Whether the image view still holds the opening box at the top of its
/// window.
///
/// One value carries the flag and the pan it last wrote, so a pan that the
/// user moved can never be confused with the one the align owns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpenAlign {
    /// The view is free: no align runs.
    Free,
    /// The align holds the opening box at the top, and wrote `pan` last.
    /// A pan or a target that moved away from it marks a user move, and the
    /// align turns [`OpenAlign::Free`].
    Holding {
        /// The pan the align wrote.
        pan: Vec2,
    },
}

/// The prefetch of a spread step, as one state instead of a patch, an aim and
/// a token that a caller has to read together.
pub enum Prefetch {
    /// Nothing prefetched.
    Idle,
    /// A worker still paints the aim.
    Painting(Painting),
    /// The patch of a step, drawn in place until the glide lands.
    Ready(CropState),
}

/// A spread-step patch whose worker still paints it.
#[derive(Debug, Clone, Copy)]
pub struct Painting {
    /// The rect and the scale the worker paints.
    pub patch: region::Patch,
    /// The token the result must carry, so an earlier aim drops.
    pub id: u64,
    /// When the paint started, so [`PREFETCH_GIVE_UP`] bounds the wait.
    pub started: Instant,
}

impl Prefetch {
    /// The patch this state shows now that its paint landed, if any.
    #[must_use]
    pub fn on_screen_patch(&self) -> Option<&CropState> {
        match self {
            Self::Ready(state) => Some(state),
            Self::Idle | Self::Painting(_) => None,
        }
    }

    /// The aim a worker paints now.
    #[must_use]
    pub fn painting(&self) -> Option<Painting> {
        match self {
            Self::Painting(painting) => Some(*painting),
            Self::Idle | Self::Ready(_) => None,
        }
    }
}

/// A message from a region worker to the image view.
///
/// The view carries one generation counter, so a paint of the previous image
/// never lands on the new one.
pub enum CropMsg {
    /// The source of the open image, or None when it needs no repaint.
    Source {
        generation: u64,
        source: Option<Arc<dyn Surface>>,
    },
    /// A finished patch of `patch.rect` at `patch.scale` pixels per source
    /// pixel.
    Crop {
        generation: u64,
        patch: region::Patch,
        texture: Texture,
    },
    /// A finished patch of a spread step, painted before the step landed.
    /// `id` names the aim, so the result of an older aim drops.
    Prefetch {
        generation: u64,
        id: u64,
        patch: region::Patch,
        texture: Texture,
    },
    /// The prefetch of aim `id` failed.
    PrefetchFailed { generation: u64, id: u64 },
    /// A paint that failed, so the view drops the pending flag and may ask
    /// again.
    Failed { generation: u64 },
    /// A whole-plane texture of the plane the view frames now, so the base
    /// picture follows a plane the source laid out again.
    Base { generation: u64, texture: Texture },
    /// The display list of the plane the view holds, built on a worker.
    ///
    /// `None` says the source answers a raster, which keeps the region patch
    /// and the whole-plane texture as the path the view takes.
    /// `part` is the box that list answers, and `None` a list of the whole
    /// plane: the view installs an answer for the box it asked for last and
    /// drops one for a box it left (see [`ViewState::scene_box`]).
    Scene {
        generation: u64,
        plane: Size,
        part: Option<Crop>,
        list: Option<Arc<[Primitive]>>,
    },
}

impl ViewState {
    /// Reset the fit and pan state for a freshly shown image.
    pub fn reset_view(&mut self) {
        // Freshly shown images open fit-all (Shift+W behavior): upscale or
        // downscale until the image first touches a window border.
        // The fit measures the opening box, so a book frames its first spread
        // and the cover page shows whole.
        self.zoom = ZoomMode::FitAll;
        self.align_open_box_again();
        self.pan = Vec2::ZERO;
        self.target_pan = Vec2::ZERO;
        self.zoom_anchor = None;
        self.spread_glide = None;
        self.view_scale = None;
        // A newly opened image starts upright: a turn never carries over to
        // the next image.
        self.angle = 0.0;
        self.pivot = None;
        // A region of the previous image points at nothing here.
        self.selection = None;
        self.ramps = Ramps::default();
    }

    /// The window mapping of the open plane, or `None` while the plane size or
    /// the view scale is still unknown.
    ///
    /// One frame answers a screen point for the draw, the `f` hit test and the
    /// `v` region drag, so no caller re-derives the transform on its own.
    pub fn frame(&self, win: Vec2) -> Option<region::Frame> {
        Some(self.frame_at(win, self.pan, self.view_scale?))
            .filter(region::Frame::has_drawable_plane)
    }

    /// The window mapping of the open plane at `pan` and `scale`.
    ///
    /// [`Self::frame`] answers the same mapping for the pan and the scale on
    /// screen; a caller that aims elsewhere (the target pan of a glide) asks
    /// here, so it never assembles the six numbers itself.
    /// `pan` is the user's offset: the placement of a stated viewport is added
    /// here (see [`Self::view_pan`]).
    #[must_use]
    pub fn frame_at(&self, win: Vec2, pan: Vec2, scale: region::Scale) -> region::Frame {
        let pan = pan + self.view_pan(scale);
        region::Frame {
            img: dim::SurfaceSize::new(self.img.w, self.img.h),
            win: dim::WindowSize::new(win.x, win.y),
            pan: dim::WindowVector::new(pan.x, pan.y),
            scale,
            angle: self.angle,
        }
    }

    /// The scale the view asks for in a window of `win`, before the ease.
    ///
    /// A box the document states is the fit for every zoom mode, so `W`, `E` and
    /// `T` keep the producer's framing; a view that states none answers the
    /// mode's own ratio, computed every frame so a resize stays correct.
    /// A fit measures the opening box when the surface names one: a book frames
    /// its first spread, so the cover page shows whole and whichever side binds
    /// decides.
    /// The caller asks only once the plane is known, because every ratio divides
    /// by it.
    #[must_use]
    pub fn fit_scale(&self, win: Vec2) -> region::Scale {
        if let Some(boxed) = self.viewport_scale(win) {
            return boxed;
        }
        let (fit_w, fit_h) = self
            .open_box
            .map_or((self.img.w, self.img.h), |boxed| (boxed.w, boxed.h));
        match self.zoom {
            ZoomMode::FitDown => {
                region::Scale::uniform((win.x / fit_w).min(win.y / fit_h).min(1.0))
            }
            ZoomMode::FitAll => region::Scale::uniform((win.x / fit_w).min(win.y / fit_h)),
            ZoomMode::FitWidth => region::Scale::uniform(win.x / fit_w),
            ZoomMode::FitHeight => region::Scale::uniform(win.y / fit_h),
            // Cover: the larger ratio wins, so the image fills the window and
            // the code crops the other axis.
            // A free zoom keeps its scale as the fit base.
            ZoomMode::Fill | ZoomMode::Free(_) => {
                region::Scale::uniform((win.x / fit_w).max(win.y / fit_h))
            }
        }
    }

    /// The scale the stated viewport asks for in a window of `win`.
    ///
    /// The box fills the window, so each axis takes the ratio of the window to
    /// the box: a stated view is its own fit, and no fit mode applies to it.
    #[must_use]
    pub fn viewport_scale(&self, win: Vec2) -> Option<region::Scale> {
        let view = self.viewport?;
        Some(region::Scale::axes(
            win.x / view.width(),
            win.y / view.height(),
        ))
    }

    /// The window offset the stated viewport places the plane at, before the
    /// user's pan.
    ///
    /// The box fills the window, so the box's centre sits at the window's middle
    /// and the plane's centre lands wherever that puts it.
    /// The offset is derived every frame rather than stored, because a resize
    /// changes both the fit and the plane's distance from the box.
    #[must_use]
    pub fn view_pan(&self, scale: region::Scale) -> Vec2 {
        let Some(view) = self.viewport else {
            return Vec2::ZERO;
        };
        Vec2::new(
            (self.img.w / 2.0 - view.center().x) * scale.x,
            (self.img.h / 2.0 - view.center().y) * scale.y,
        )
    }

    /// The window offset of the plane's centre: the stated viewport's placement
    /// plus the user's pan.
    #[must_use]
    pub fn offset(&self, scale: region::Scale) -> Vec2 {
        self.pan + self.view_pan(scale)
    }

    /// Frame the opening box again once the image dimensions land.
    ///
    /// A fresh open measures a box instead of the whole plane, and a scroll to
    /// the top shows it first.
    pub fn refit_open(&mut self) {
        self.zoom = ZoomMode::FitAll;
        self.pan = Vec2::ZERO;
        self.target_pan = Vec2::ZERO;
        self.align_open_box_again();
        self.spread_glide = None;
        // The frame computes the scale from the fit mode.
        self.view_scale = None;
        self.align_open_box_again();
    }

    /// Arm the align of the opening box again, at the pan the caller just
    /// wrote.
    ///
    /// A source without an opening box frames its whole plane, so the align
    /// stays out of the way.
    pub fn align_open_box_again(&mut self) {
        self.open_align = if self.open_box.is_some() {
            OpenAlign::Holding { pan: self.pan }
        } else {
            OpenAlign::Free
        };
    }

    /// Hold the opening box at the top of the window while the view stays
    /// untouched.
    ///
    /// The fit centers the whole plane, and a book starts at the top of its
    /// plane, so a fresh open scrolls there.
    /// The align repeats every frame, because a window manager resize re-fits
    /// the scale after the first frame and the pan would drift to the middle
    /// of the document.
    /// A pan or a target that moved away from the held pan marks a user move,
    /// and the align then stays out of the way.
    pub fn align_open_box(&mut self, win_h: f32) {
        let OpenAlign::Holding { pan: held } = self.open_align else {
            return;
        };
        let (Some(scale), Some(_)) = (self.view_scale, self.open_box) else {
            return;
        };
        if self.pan != held || self.target_pan != held {
            self.open_align = OpenAlign::Free;
            return;
        }
        let top = Vec2::new(0.0, Column::new(self.img, scale.y, win_h).top_pan());
        self.pan = top;
        self.target_pan = top;
        self.open_align = OpenAlign::Holding { pan: top };
    }

    /// Ease the view to the spread `step` spreads away from the one the window
    /// shows now.
    ///
    /// A document shows as a book: every spread spans the plane and stacks one
    /// page high, so a step moves the pan by one spread and leaves the scale
    /// alone.
    /// The target spread lands at the top of the window, which is where the
    /// open framing put the first one.
    /// The row comes from the pan the view aims at, so a step pressed at any
    /// point of the document counts from the spread the reader looks at.
    /// A step past either end does nothing, so the view stops on the first and
    /// the last spread instead of showing background.
    ///
    /// The function returns whether it aimed at a spread.
    pub fn step_spread(&mut self, step: isize, win_h: f32) -> bool {
        let (Some(scale), Some(source)) = (self.view_scale, self.source.as_ref()) else {
            return false;
        };
        if !scale.is_drawable() {
            return false;
        }
        let count = source.spread_count();
        let Some(first) = source.spread_box(0) else {
            return false;
        };
        if count < 2 || !(first.height().is_finite() && first.height() > 0.0) {
            return false;
        }
        let column = Column::new(self.img, scale.y, win_h);
        let row = column
            .spread_at_top(self.target_pan.y, first.height())
            .min(count - 1);
        let next = row as isize + step;
        if next < 0 || next as usize >= count {
            return false;
        }
        let Some(target) = source.spread_box(next as usize) else {
            return false;
        };
        self.target_pan = Vec2::new(0.0, column.pan_of(target.min_y()));
        self.spread_glide = Some(self.target_pan);
        true
    }

    /// Whether the view paints region patches, or the display list draws the
    /// picture itself.
    ///
    /// While the list draws, a spread step paints nothing on a worker and a
    /// settle repaints no patch, which keeps a glide off the raster path — the
    /// one moment the main thread is encoding the list for the band it left.
    /// The raster is the fallback of a frame the list refuses, and it resumes as
    /// soon as one clears the flag (see [`ViewState::list_drawn`]).
    #[must_use]
    fn wants_patch(&self) -> bool {
        !self.list_drawn
    }

    /// Paint the region a spread step lands on, so the sharp patch is ready
    /// when the glide arrives.
    ///
    /// The step knows its target pan at the press edge, so the paint needs no
    /// guess.
    /// The work runs on a decode worker (`rayon`), so the glide never waits on
    /// it, and the result lands in [`ViewState::prefetch`] instead of replacing
    /// the patch on screen.
    /// A step that misses a spread calls nothing, because it moves nothing, and
    /// a document whose display list drew the frame before paints no patch here
    /// at all (see [`Self::wants_patch`]).
    pub fn prefetch_spread(&mut self, win: Vec2, uploader: &Uploader) {
        let (Some(patch), Some(source)) = (self.spread_patch(win), self.source.clone()) else {
            return;
        };
        let (scale, plane) = (patch.scale, self.source_plane);
        let id = self.next_prefetch_id();
        self.prefetch = Prefetch::Painting(Painting {
            patch,
            id,
            started: Instant::now(),
        });
        let (generation, tx) = (self.crop_gen, self.crop_tx.clone());
        let uploader = uploader.clone();
        rayon::spawn(move || {
            let started = Instant::now();
            match source.render(View::region(plane, patch.rect, scale)) {
                Ok(image) => {
                    // The paint a spread step fires at the press edge is the one
                    // a pan stutter is about, so it carries the same fields as
                    // the settled patch.
                    tracing::debug!(
                        ms = ms_since(started),
                        cw = image.width().raw(),
                        ch = image.height().raw(),
                        scale,
                        "spread prefetch"
                    );
                    // The patch uploads on this worker, so the glide never
                    // waits on a pixel copy.
                    let Ok(texture) = uploader.image(&image) else {
                        let _ = tx.send(CropMsg::PrefetchFailed { generation, id });
                        return;
                    };
                    let _ = tx.send(CropMsg::Prefetch {
                        generation,
                        id,
                        patch,
                        texture,
                    });
                }
                Err(err) => {
                    // A step that cannot paint early keeps the ordinary path:
                    // the quiet window paints after the glide.
                    tracing::warn!(err = %err, "spread prefetch");
                    let _ = tx.send(CropMsg::PrefetchFailed { generation, id });
                }
            }
        });
    }

    /// The patch a spread step asks a worker to paint, or `None` when the step
    /// paints nothing.
    ///
    /// A view with no source, no plane or no scale paints nothing, and so does
    /// one whose display list drew the frame before (see [`Self::wants_patch`]).
    /// The rect is the one an ordinary paint would cover, so the landing frame
    /// adopts it without a second paint.
    fn spread_patch(&self, win: Vec2) -> Option<region::Patch> {
        if !self.wants_patch() || self.source.is_none() || self.img.w <= 0.0 {
            return None;
        }
        let scale = self.view_scale?;
        if !scale.is_drawable() {
            return None;
        }
        let aim = self.frame_at(win, self.target_pan, scale);
        // A patch is one raster at one scale, and only a raster source ever
        // paints one: a view that states its viewport draws a list and paints no
        // patch, so the coarser axis is the honest reading here.
        let patch = region::Patch::new(aim.crop(region::MARGIN), scale.coarsest());
        (!patch.rect.is_empty()).then_some(patch)
    }

    /// Drop the region state of the image that was open, and retire every
    /// worker result still in flight.
    ///
    /// What the view shows is what a content change replaces, so the stated
    /// viewport goes with it: a viewport belongs to the list it framed, and the
    /// next picture or document fits its own plane.
    pub fn close_crop(&mut self) {
        self.source = None;
        self.scene = None;
        self.scene_box = None;
        self.scene_wanted = false;
        self.scene_pending = false;
        self.list_drawn = false;
        self.viewport = None;
        self.drop_patch();
    }

    /// Drop the patch on screen and every paint still in flight for it.
    ///
    /// The source, when it lays itself out for the window (a treemap), answers
    /// a plane of another shape after a resize, and a patch of the old shape
    /// would land on a plane that no longer exists.
    /// The next quiet window then paints the plane the source reports now.
    pub fn drop_patch(&mut self) {
        self.crop = None;
        self.crop_pending = false;
        self.crop_seen = None;
        self.crop_quiet_at = Instant::now();
        self.crop_gen = self.crop_gen.wrapping_add(1);
        self.close_prefetch();
    }

    /// Hand the window's frame to the open source, and re-measure the plane it
    /// takes.
    ///
    /// A treemap lays itself out for the window it shows in (see
    /// [`Surface::frame_window`]), so a resize leaves a plane of the old shape
    /// and a fit that would letterbox it.
    /// The source answers whether its plane moved, so a picture that has one
    /// natural size costs a call and never a second header read.
    /// The picture on screen stretches until the settled repaint lands, because
    /// a layout per resize frame would parse and rasterize the whole tree per
    /// frame.
    pub fn frame_window(&mut self, win: Vec2) {
        // The clone buys the borrow: the fields below write the view this call
        // read the source from.
        let Some(source) = self.source.clone() else {
            return;
        };
        let Some(plane) = framed_plane(source.as_ref(), self.source_plane, win, self.img) else {
            return;
        };
        self.img = plane;
        self.plane_stale = true;
        self.drop_patch();
    }

    /// Paint the whole plane again when the texture on screen measures an older
    /// one.
    ///
    /// A treemap answers a plane of another shape for a new window (see
    /// [`Surface::frame_window`]), and the texture of the old frame draws the
    /// layout of that frame beside the layout of this plane.
    /// A resize frame only records the new plane (see [`Self::frame_window`]),
    /// so the caller runs this once the window stops moving: one worker per
    /// settled window, and no worker per frame of a drag.
    /// The paint is the whole plane at scale 1.0, which is the size the draw
    /// maps it from, and a worker uploads it like every other decode.
    /// A source that answered a list has no plane texture to repaint: its list
    /// is the picture, so the new frame asks for the list of the new plane and
    /// the one on screen draws until the answer lands.
    pub fn refresh_plane(&mut self, uploader: &Uploader) {
        if !self.plane_stale || self.crop_pending {
            return;
        }
        let Some(source) = self.source.clone() else {
            // A source that painted its plane and kept no handle cannot paint it
            // again, so the texture on screen stays the best picture there is.
            // No source of this viewer does that today: a plane that follows the
            // window always answers `can_refine`.
            self.plane_stale = false;
            return;
        };
        self.plane_stale = false;
        // A source that answers a list rebuilds it for the frame the window
        // names, and the list on screen draws until the answer lands.
        // The ask runs even when no list landed yet, because a list built
        // before the first frame describes the plane the launch guessed, and
        // that plane is not this one.
        self.request_scene();
        if self.scene.is_some() {
            return;
        }
        // One paint per plane: a window that moves again sets the flag again,
        // and its settled frame asks for the paint of the plane it lands on.
        let (generation, tx) = (self.crop_gen, self.crop_tx.clone());
        let plane = self.source_plane;
        let uploader = uploader.clone();
        rayon::spawn(move || {
            let started = Instant::now();
            match source.render(View::whole(plane, 1.0)) {
                Ok(image) => {
                    // A plane that follows the window paints again on every
                    // resize, so this is the second number a resize report
                    // needs, beside the display list's own.
                    tracing::debug!(
                        ms = ms_since(started),
                        w = image.width().raw(),
                        h = image.height().raw(),
                        "plane paint"
                    );
                    match uploader.image(&image) {
                        Ok(texture) => {
                            let _ = tx.send(CropMsg::Base {
                                generation,
                                texture,
                            });
                        }
                        Err(err) => tracing::warn!(err = %err, "plane paint"),
                    }
                }
                Err(err) => tracing::warn!(err = %err, "plane paint"),
            }
        });
    }

    /// Drop the prefetch of a spread step: the patch on screen, the aim a
    /// worker paints and every result still in flight.
    pub fn close_prefetch(&mut self) {
        self.prefetch = Prefetch::Idle;
        self.prefetch_seq = self.prefetch_seq.wrapping_add(1);
    }

    /// The token of a fresh prefetch, so a result of an earlier aim drops.
    pub fn next_prefetch_id(&mut self) -> u64 {
        self.prefetch_seq = self.prefetch_seq.wrapping_add(1);
        self.prefetch_seq
    }

    /// Adopt a source that the launch already opened, so the view can repaint
    /// the visible region at any zoom.
    pub fn adopt_source(&mut self, source: Option<Arc<dyn Surface>>) {
        self.close_crop();
        self.source = source;
        self.request_scene();
    }

    /// Ask a worker for the display list of the plane the view holds.
    ///
    /// A source that answers primitives draws at any zoom, so the list is the
    /// picture and the region patch has nothing to paint.
    /// The build runs off the main thread because a treemap rasterizes one
    /// label picture per label the first time it is asked, and a document
    /// parses its content stream.
    /// One build in flight: a plane that moves again asks once the first answer
    /// lands, and the list on screen stays until the new one does.
    ///
    /// A source that lists one box of its plane answers no list here: only a
    /// frame knows the window, so it asks with the box it shows (see
    /// [`Self::request_box_scene`]).
    pub fn request_scene(&mut self) {
        if self.scene_pending {
            return;
        }
        let Some(source) = self.source.clone() else {
            return;
        };
        if source.lists_a_box() {
            // The list of this source is the list of a box, and a box belongs to
            // a frame: the next frame hands one in (see
            // [`Self::request_box_scene`]).
            self.scene_wanted = true;
            return;
        }
        self.ask_scene(None);
    }

    /// Ask a worker for the list of `part` of the plane, or of the whole plane
    /// for `None`.
    ///
    /// One build in flight: the flag is what the next ask reads, and an answer
    /// for a box the view left clears it without landing (see [`CropMsg::Scene`]).
    fn ask_scene(&mut self, part: Option<Crop>) {
        if self.scene_pending {
            return;
        }
        let Some(source) = self.source.clone() else {
            return;
        };
        self.scene_pending = true;
        self.scene_wanted = false;
        self.scene_box = part;
        let (generation, tx) = (self.crop_gen, self.crop_tx.clone());
        let plane = self.source_plane;
        let size = self.img;
        rayon::spawn(move || {
            let started = Instant::now();
            let list = match part {
                Some(part) => source.display_box(plane, part).ok().flatten(),
                None => source.display(plane).ok().flatten(),
            };
            // The build is the one step of a list this viewer cannot see: a
            // document walks its tree and a map lays itself out, and a slow one
            // shows as a picture that arrives late.
            // A source that answers no list reports none, which is what its own
            // `debug` event beside this one names the reason for.
            tracing::debug!(
                ms = ms_since(started),
                listed = list.is_some(),
                prims = list.as_ref().map_or(0, |list| list.len()),
                "display list"
            );
            let _ = tx.send(CropMsg::Scene {
                generation,
                plane: size,
                part,
                list,
            });
        });
    }

    /// Ask a worker for the display list the window shows, when the list on
    /// screen answers another box.
    ///
    /// A book is one plane of every page, so a list of the plane is the whole
    /// document and a list of the box the window shows is a spread and the ink
    /// beside it (see [`Surface::display_box`]). The box is
    /// [`crate::display::band_for`]'s, which is the box the display pass culls
    /// its encode to, so the source and the renderer agree on what a frame can
    /// show and no page is listed for a window that cannot reach it.
    ///
    /// A source that lists a whole plane is asked once, and its list holds every
    /// box: this returns without a build for every frame after the first.
    pub fn request_box_scene(&mut self, win: Vec2) {
        if self.scene_pending {
            return;
        }
        let Some(source) = self.source.clone() else {
            return;
        };
        if !source.lists_a_box() {
            if self.scene_wanted {
                self.request_scene();
            }
            return;
        }
        let Some(frame) = self.frame(win) else {
            return;
        };
        let Some(band) = crate::display::band_for(&frame) else {
            return;
        };
        // The box the last answer was asked for, whether that answer was a list
        // or a source that lists nothing: a book whose page cannot list keeps the
        // raster path for that box instead of asking for it every frame.
        if self
            .scene_box
            .is_some_and(|asked| asked.contains_rect(&band))
            && !self.scene_wanted
        {
            return;
        }
        self.ask_scene(Some(band));
    }

    /// Ask a worker for the source of an entry, so the view can repaint the
    /// region its window shows.
    ///
    /// The grid keeps no source, so an image opened from the grid opens its
    /// own here, off the main thread.
    /// A raster file answers None at once.
    /// A document arrives as a book, so the patch covers whole spreads.
    pub fn request_source(&mut self, entry: &EntrySource) {
        self.close_crop();
        self.source_plane = 0;
        let path = entry.shared_path();
        let (generation, tx) = (self.crop_gen, self.crop_tx.clone());
        rayon::spawn(move || {
            // The open reads the head and, for a vector file, parses the tree
            // once. A source that cannot refine keeps no handle: a repaint
            // would only decode the same pixels again.
            let source: Option<Arc<dyn Surface>> = open_book_surface(path.as_path())
                .ok()
                .filter(|surface| surface.can_refine())
                .map(Arc::from);
            let _ = tx.send(CropMsg::Source { generation, source });
        });
    }

    /// Repaint the visible region once the view holds still, when the patch on
    /// screen no longer covers the window at the right scale.
    ///
    /// The paint area follows the window, not the document, so a deep zoom
    /// costs no more than a shallow one.
    /// The guard is one paint in flight: resvg cannot cancel mid-paint.
    /// A load in flight never holds the paint back: the patch covers what the
    /// window shows on its own, and a book's whole-plane render can take
    /// seconds, so waiting for it would only delay the sharp picture.
    /// The paint waits for its source instead, which [`Self::request_source`]
    /// opens on its own worker.
    /// The source is also the truth about the plane: a document's cell measures
    /// one page of it, so a patch of the pair the view holds would land in the
    /// wrong place until the plane the source answers agrees with it.
    /// A spread step paints its landing region before the glide ends, so a
    /// patch that already covers the window replaces the one on screen here and
    /// skips the debounce and the paint.
    pub fn tick_crop(&mut self, now: Instant, win: Vec2, uploader: &Uploader) {
        // A display list draws the whole picture, so no worker paints this one
        // (see [`Self::wants_patch`]).
        if !self.wants_patch() {
            return;
        }
        let Some(source) = self.source.as_ref() else {
            return;
        };
        if self.crop_pending || self.img.w <= 0.0 {
            return;
        }
        if source
            .bounds(self.source_plane)
            .is_ok_and(|plane| !same_plane(plane, self.img))
        {
            return;
        }
        let Some(scale) = self.view_scale else {
            return;
        };
        if !scale.is_drawable() {
            return;
        }
        let frame = self.frame_at(win, self.pan, scale);
        let needed = region::Patch::new(frame.crop(0.0), scale.coarsest());
        // A prefetched patch that holds the window is the patch on screen now.
        // The check runs before the debounce, so the landing frame shows the
        // sharp spread at once instead of waiting out [`region::QUIET`].
        if self
            .prefetch
            .on_screen_patch()
            .is_some_and(|state| region::patch_covers_at_scale(needed, state.patch))
        {
            let ready = std::mem::replace(&mut self.prefetch, Prefetch::Idle);
            if let Prefetch::Ready(state) = ready {
                self.crop = Some(state);
            }
            self.crop_seen = Some(needed);
            self.crop_quiet_at = now;
            return;
        }
        // A patch of a step that ended without holding the window (a resize, a
        // pan away) would only linger as a stretched texture over the whole
        // image, so it drops.
        // A patch still in flight lands and then drops the same way.
        if self.spread_glide.is_none() {
            self.prefetch = Prefetch::Idle;
        }
        // The debounce: any movement of the seen region restarts the timer, so
        // a drag, a pan or an ease never queues a paint per frame.
        let moved = self
            .crop_seen
            .is_none_or(|seen| !same_patch(&seen, &needed));
        if moved {
            self.crop_seen = Some(needed);
            self.crop_quiet_at = now;
            return;
        }
        if now.duration_since(self.crop_quiet_at) < region::QUIET {
            return;
        }
        // A prefetch of this very region is on its way, so the paint that would
        // duplicate it waits instead.
        // A worker that never answers stops the wait at [`PREFETCH_GIVE_UP`].
        if let Some(painting) = self.prefetch.painting()
            && region::patch_covers_at_scale(needed, painting.patch)
        {
            if now.duration_since(painting.started) < PREFETCH_GIVE_UP {
                return;
            }
            self.prefetch = Prefetch::Idle;
        }
        if !region::worth_painting(needed, self.crop.as_ref().map(|crop| crop.patch)) {
            return;
        }
        let Some(source) = self.source.clone() else {
            return;
        };
        // Paint a margin around what the window shows, so a small pan between
        // two paints stays covered, and a filter keeps its outer context.
        let paint = region::Patch::new(frame.crop(region::MARGIN), scale.coarsest());
        if paint.rect.is_empty() {
            return;
        }
        self.crop_pending = true;
        // The aim of a spread step is spent: this paint covers the same region,
        // and a patch of an older aim would only land late.
        self.close_prefetch();
        let (generation, tx) = (self.crop_gen, self.crop_tx.clone());
        let plane = self.source_plane;
        let uploader = uploader.clone();
        Self::spawn_patch(source, plane, paint, generation, tx, uploader);
    }

    /// Paint one patch on a worker, and upload the texture there too.
    ///
    /// The worker uploads its own texture (a queue is `Send + Sync`), so the
    /// frame that drains the answer moves handles and nothing else.
    fn spawn_patch(
        source: Arc<dyn Surface>,
        plane: usize,
        paint: region::Patch,
        generation: u64,
        tx: Sender<CropMsg>,
        uploader: Uploader,
    ) {
        rayon::spawn(
            // A patch is painted at one scale, the one its pixels hold.
            move || {
                let started = Instant::now();
                match source.render(View::region(plane, paint.rect, paint.scale)) {
                    Ok(image) => {
                        // A patch is the stutter a reader feels while a pan or a
                        // zoom moves, and its scale and box are the two fields a
                        // report needs to size the window against the plane.
                        tracing::debug!(
                            ms = ms_since(started),
                            cw = image.width().raw(),
                            ch = image.height().raw(),
                            scale = paint.scale,
                            "region paint"
                        );
                        let Ok(texture) = uploader.image(&image) else {
                            let _ = tx.send(CropMsg::Failed { generation });
                            return;
                        };
                        let _ = tx.send(CropMsg::Crop {
                            generation,
                            patch: paint,
                            texture,
                        });
                    }
                    Err(err) => {
                        tracing::warn!(err = %err, "region paint");
                        let _ = tx.send(CropMsg::Failed { generation });
                    }
                }
            },
        );
    }

    /// Apply finished region work.
    /// A source lands as is, and a patch replaces the one on screen.
    /// The paint worker uploaded its own texture, so this call only moves
    /// handles.
    /// A result of an earlier image drops.
    pub fn drain_crops(&mut self) {
        while let Ok(msg) = self.crop_rx.try_recv() {
            match msg {
                CropMsg::Source { generation, source } => {
                    if generation == self.crop_gen {
                        self.source = source;
                        // The list of this plane rides its own worker, so the
                        // frame that adopts the source draws the old answer
                        // until the new one lands.
                        self.request_scene();
                    }
                }
                CropMsg::Scene {
                    generation,
                    plane,
                    part,
                    list,
                } => {
                    // Any answer ends the build, stale or not: a flag left set
                    // by a dropped answer would refuse every later ask.
                    self.scene_pending = false;
                    // A list of a plane or a box the view left would draw the
                    // layout of the frame before it, so only the current one
                    // lands.
                    if generation == self.crop_gen
                        && same_plane(plane, self.img)
                        && part == self.scene_box
                    {
                        self.scene = list;
                    } else {
                        // The launch asks for a list before the first frame
                        // measures the plane, and a plane that follows the
                        // window (a treemap) moves on that frame, so the first
                        // answer is nearly always one plane behind. Ask again
                        // for the plane the view holds now.
                        self.request_scene();
                    }
                }
                CropMsg::Crop {
                    generation,
                    patch,
                    texture,
                } => {
                    if generation != self.crop_gen {
                        continue;
                    }
                    self.crop_pending = false;
                    // The old patch drops here, so one patch lives at a time.
                    self.crop = Some(CropState { texture, patch });
                }
                CropMsg::Base {
                    generation,
                    texture,
                } => {
                    if generation == self.crop_gen {
                        self.view_tex = Some(texture);
                    }
                }
                CropMsg::Failed { generation } => {
                    if generation == self.crop_gen {
                        // A failure waits for the next quiet window instead of
                        // retrying every frame.
                        self.crop_pending = false;
                        self.crop_quiet_at = Instant::now();
                    }
                }
                CropMsg::Prefetch {
                    generation,
                    id,
                    patch,
                    texture,
                } => {
                    // A result of an earlier image or of an aim the view left
                    // drops, so an old spread never lands on the new one.
                    let aimed = matches!(
                        self.prefetch,
                        Prefetch::Painting(Painting { id: want, .. }) if want == id
                    );
                    if generation != self.crop_gen || !aimed {
                        continue;
                    }
                    self.prefetch = Prefetch::Ready(CropState { texture, patch });
                }
                CropMsg::PrefetchFailed { generation, id } => {
                    let aimed = matches!(
                        self.prefetch,
                        Prefetch::Painting(Painting { id: want, .. }) if want == id
                    );
                    if generation == self.crop_gen && aimed {
                        // The ordinary path paints after the glide instead.
                        self.prefetch = Prefetch::Idle;
                    }
                }
            }
        }
    }
}

/// The held-key ramps behind the continuous image-view motions.
/// One ramp per direction, so a hold eases in, cruises, and eases out
/// instead of stepping on the auto-repeat rate.
/// The pan ramps feed the target pan, and the zoom ramps feed the free-zoom
/// target.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ramps {
    pub pan_left: Ramp,
    pub pan_right: Ramp,
    pub pan_up: Ramp,
    pub pan_down: Ramp,
    pub zoom_in: Ramp,
    pub zoom_out: Ramp,
}

impl Ramps {
    /// Advance every ramp by one frame.
    ///
    /// `down`: whether each key is held now.
    /// `base`: the pan cruise speed for one full frame.
    /// `dt`: the frame time, in seconds.
    /// `speed`: a multiplier on both the pan and the zoom rate, and 1.0 for
    /// normal speed and 2.0 while Shift is held.
    /// It scales the output speed only, not the ramp, so the ease stays the
    /// same length at any speed.
    ///
    /// The return value holds this frame's pan delta and zoom factors.
    /// The ease runs on the returned values, not on the call count, so a
    /// caller may call the function once per frame.
    pub fn tick(&mut self, down: NavKeys, base: f32, dt: f32, speed: f32) -> Motion {
        // Only the held state counts.
        // Every key press event, and the auto-repeat rate with it, has no
        // effect on these ramps.
        let pan_speed = base * dt * speed;
        let ramp = RAMP_TIME;
        let (l, r, u, d) = (
            self.pan_left.tick(down.left, dt, ramp),
            self.pan_right.tick(down.right, dt, ramp),
            self.pan_up.tick(down.up, dt, ramp),
            self.pan_down.tick(down.down, dt, ramp),
        );
        let (zi, zo) = (
            self.zoom_in.tick(down.zoom_in, dt, ramp),
            self.zoom_out.tick(down.zoom_out, dt, ramp),
        );
        // The zoom rate is an e-fold rate, so the scale factor is the
        // exponential of the ramped rate over this frame.
        // That keeps the zoom speed even in perceived terms at any scale.
        // A frame with no zoom key held answers `None`, so no caller reads a
        // sentinel factor of 1.0 and has to ask a second flag what it means.
        Motion {
            pan: Vec2 {
                x: pan_speed * (l - r),
                y: pan_speed * (u - d),
            },
            zoom: (zi > 0.0 || zo > 0.0).then(|| (ZOOM_RATE * speed * dt * (zi - zo)).exp()),
        }
    }
}

/// One frame of output from the held-key ramps.
#[derive(Debug, Clone, Copy)]
pub struct Motion {
    /// The pan delta for this frame, in window pixels.
    pub pan: Vec2,
    /// The free-zoom factor for this frame, or `None` when no zoom key ramps.
    pub zoom: Option<f32>,
}

/// The held keys that drive the continuous image-view motions.
/// The six directions are independent, so the struct stays a plain set of
/// flags.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NavKeys {
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub zoom_in: bool,
    pub zoom_out: bool,
}

/// The mouse side-button nav state for this frame.
/// X11 reports the thumb buttons as buttons 8 (back) and 9 (forward), and winit
/// maps them to `MouseButton::Back` and `MouseButton::Forward`.
/// The raw numbers stay accepted too, so a backend that hands them over
/// unmapped behaves the same.
#[derive(Debug, Clone, Copy, Default)]
pub struct NavMouse {
    /// The next-image button state.
    pub next: NavButton,
    /// The previous-image button state.
    pub prev: NavButton,
}

/// One side button, as an edge and a held state.
#[derive(Debug, Clone, Copy, Default)]
pub struct NavButton {
    /// The button went down this frame.
    pub edge: bool,
    /// The button is held now.
    pub held: bool,
}

/// Read the side-button nav state once per frame.
///
/// A held side button repeats through the auto-repeat path, exactly like `n` and
/// `p`.
/// X11 reports buttons 8 and 9, and winit's `Back` and `Forward` are the same
/// two buttons, so both spellings reach the same step.
#[must_use]
pub fn read_nav_mouse(input: &Input) -> NavMouse {
    const NEXT: [MouseButton; 2] = [MouseButton::Forward, MouseButton::Other(9)];
    const PREV: [MouseButton; 2] = [MouseButton::Back, MouseButton::Other(8)];
    NavMouse {
        next: NavButton {
            edge: input.any_button_pressed(&NEXT),
            held: input.any_button_down(&NEXT),
        },
        prev: NavButton {
            edge: input.any_button_pressed(&PREV),
            held: input.any_button_down(&PREV),
        },
    }
}

/// Draw a texture over the image rect, turned by `angle` degrees clockwise
/// around the image center.
///
/// `center` is the image center on screen, in window pixels.
/// The angle stays as it is: the code snaps it on the hold release, so a drag
/// turns freely and never jumps mid-drag.
/// The scale and the pan stay as they are, so a turned image shows its
/// corners past the window edge.
///
/// The quad turns around `dest.position`, and `origin` pulls the rect back by
/// half its size.
/// The dest position is therefore the image center, so the turn pivots on the
/// middle of the image at every angle.
/// A zero angle draws the plain centered rect, because the origin pulls it
/// back by the same half size.
pub fn draw_turned(
    r: &mut Renderer,
    texture: &Texture,
    img: Size,
    scale: f32,
    center: Vec2,
    angle: f32,
    filter: Filter,
) {
    let src = dim::whole(texture.width() as f32, texture.height() as f32);
    let (dw, dh) = (img.w * scale, img.h * scale);
    let dest = dim::rect(center.x, center.y, dw, dh);
    r.texture_pro(
        texture,
        src,
        dest,
        Turn::new(dim::center_origin(dest), rotate::normalize_deg(angle)),
        Color::WHITE,
        filter,
    );
}

/// The vertical half of one window mapping: how tall the plane stands on the
/// screen, and how tall the window is.
///
/// The spread math reads the y axis alone, so it takes these three numbers as
/// one named value instead of three loose floats that a caller can swap.
#[derive(Debug, Clone, Copy)]
pub struct Column {
    /// The plane height, in surface units.
    pub img_h: f32,
    /// Output pixels per surface unit.
    pub scale: f32,
    /// The window height, in window pixels.
    pub win_h: f32,
}

impl Column {
    /// The vertical mapping of `img` at `scale` inside a window `win_h` pixels
    /// tall.
    #[must_use]
    pub const fn new(img: Size, scale: f32, win_h: f32) -> Self {
        Self {
            img_h: img.h,
            scale,
            win_h,
        }
    }

    /// The pan that puts the top of the plane on the top edge of the window.
    ///
    /// The draw centers the plane and adds the pan, so the top edge sits at
    /// `(win_h - img_h * scale) / 2 + pan.y`.
    /// A positive pan of half the overflow zeroes that, and a plane shorter
    /// than the window stays centered.
    #[must_use]
    pub fn top_pan(self) -> f32 {
        (self.img_h * self.scale - self.win_h).max(0.0) / 2.0
    }

    /// The pan that puts the top of a spread on the top edge of the window.
    ///
    /// `origin_y` is the top of the spread in plane units, so spread 0 answers
    /// [`Self::top_pan`]: a step to the first spread frames it like a fresh
    /// open does.
    #[must_use]
    pub fn pan_of(self, origin_y: f32) -> f32 {
        self.top_pan() - origin_y * self.scale
    }

    /// The spread a pan shows at the top of the window, rounded to the
    /// nearest.
    ///
    /// The code reads the row from the pan the view aims at, so a step pressed
    /// while an earlier step still glides counts from the spread that step
    /// aims at, never from the pixels on the way.
    #[must_use]
    pub fn spread_at_top(self, pan_y: f32, spread_h: f32) -> usize {
        if !(spread_h > 0.0 && self.scale.is_finite() && self.scale > 0.0) {
            return 0;
        }
        let top = (self.top_pan() - pan_y) / self.scale;
        (top / spread_h).round().max(0.0) as usize
    }
}

/// The sampler filter of the image view.
///
/// The `a` key picks it, and it reaches the whole-image texture and the region
/// patches only: a thumbnail is always linear, because both users draw it far
/// larger than the texture and nearest aliases badly there.
#[must_use]
pub const fn view_filter(pixelated: bool) -> Filter {
    if pixelated {
        Filter::Nearest
    } else {
        Filter::Linear
    }
}

/// The snapped destination of a region patch, or `None` when the patch needs
/// the free draw.
///
/// A patch rasterizes at the view scale, so its pixels are already the size the
/// screen shows: the two sizes agree, and the patch holds every pixel the
/// window shows at its own scale.
/// The destination then snaps to the whole window pixels and takes the
/// texture's own pixel size, because a bilinear sample at a fractional
/// destination position blends two texels and softens every edge, which reads
/// as blurry text.
/// A stretched patch keeps its true size, so the sample phases already drift
/// across it, and only the position snaps.
/// A turned patch has no pixel grid to land on, so the caller keeps the free
/// draw.
#[must_use]
pub fn patch_dest(
    center: dim::WindowPoint,
    dw: f32,
    dh: f32,
    tw: f32,
    th: f32,
    angle: f32,
) -> Option<Rect> {
    if angle != 0.0 || !(dw > 0.0 && dh > 0.0) {
        return None;
    }
    // The paint rounds its pixel size, so a size within one pixel of the
    // painted one takes the painted one: the sample then sits on texel centers
    // for every pixel of the patch.
    let width = if (dw - tw).abs() <= 1.0 {
        tw
    } else {
        dw.round()
    };
    let height = if (dh - th).abs() <= 1.0 {
        th
    } else {
        dh.round()
    };
    Some(dim::rect(
        (center.x - dw / 2.0).round(),
        (center.y - dh / 2.0).round(),
        width,
        height,
    ))
}

/// Draw one region patch of the image view.
///
/// The patch covers a rect of the plane at `scale` output pixels per unit, so
/// the draw lands it 1:1 on the screen whenever the view scale still matches
/// the paint (see [`patch_dest`]).
/// A turned patch keeps the free draw of [`draw_turned`], so it holds the
/// transform of the whole image.
pub fn draw_patch(r: &mut Renderer, patch: &CropState, frame: &region::Frame, filter: Filter) {
    let rect = patch.patch.rect;
    let (scale, angle) = (frame.scale.coarsest(), frame.angle);
    let middle = dim::SurfacePoint::new(
        rect.min_x() + rect.width() / 2.0,
        rect.min_y() + rect.height() / 2.0,
    );
    let center = frame.to_screen(middle);
    let tw = patch.texture.width() as f32;
    let th = patch.texture.height() as f32;
    match patch_dest(
        center,
        rect.width() * scale,
        rect.height() * scale,
        tw,
        th,
        angle,
    ) {
        Some(dest) => r.texture_pro(
            &patch.texture,
            dim::whole(tw, th),
            dest,
            Turn::NONE,
            Color::WHITE,
            filter,
        ),
        None => draw_turned(
            r,
            &patch.texture,
            Size {
                w: rect.width(),
                h: rect.height(),
            },
            scale,
            dim::draw_window(center),
            angle,
            filter,
        ),
    }
}

/// The display-list pass, and the build of one a worker is running.
///
/// `vello` compiles its compute pipelines when the pass is built, which is a
/// moment of work no frame of this viewer can afford, so the first list a
/// source answers asks a worker for the pass: the frames before it lands draw
/// the plane texture and the region patch, and every frame after it draws the
/// list (see [`draw_list`]).
/// A run that shows only pictures never pays for the build, because no list
/// ever arrives and nothing here asks.
#[derive(Default)]
pub(crate) struct ListPass {
    /// The pass the last frame drew through, once a worker has built it.
    ///
    /// It lives for the session: the pass holds the pipelines and the
    /// pictures of one list, and a new list re-encodes it (see
    /// [`crate::display::SceneView::draw`]).
    built: Option<crate::display::SceneView>,
    /// The build a worker is running, or `None` while none is in flight.
    ///
    /// No pass and no build is what asks for the next one, and a build that
    /// fails latches [`Self::failed`] instead.
    building: Option<Receiver<Result<crate::display::SceneView, crate::display::SceneError>>>,
    /// Set when a build failed, which keeps every later frame on the raster
    /// path.
    ///
    /// Vello refuses a device for a feature or a limit it is missing, and no
    /// later frame adds one, so the viewer reports it once rather than asking a
    /// worker for a pass that cannot come.
    failed: bool,
}

impl ListPass {
    /// The pass to draw this frame with, asking a worker for one when there is
    /// none.
    ///
    /// `None` answers the frame that asks, and every frame of a build still in
    /// flight, which the caller draws from the raster path instead.
    fn ready(
        &mut self,
        device: &wgpu::Device,
        r: &Renderer,
    ) -> Option<&mut crate::display::SceneView> {
        if self.failed {
            return None;
        }
        if let Some(building) = self.building.as_ref() {
            match building.try_recv() {
                Ok(Ok(built)) => {
                    self.built = Some(built);
                    self.building = None;
                }
                // The build that failed reported itself (see [`build_pass`]),
                // and the viewer keeps the raster path it falls back to.
                Ok(Err(_)) => {
                    self.failed = true;
                    self.building = None;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                // The worker died before it answered, so the frame paints the
                // raster and names the loss instead of waiting for an answer
                // that cannot come.
                Err(mpsc::TryRecvError::Disconnected) => {
                    tracing::warn!("display list worker gone");
                    self.failed = true;
                    self.building = None;
                }
            }
        }
        if self.built.is_none() && self.building.is_none() {
            self.building = Some(ask_for_pass(device, r.uploader()));
        }
        self.built.as_mut()
    }
}

/// Ask a worker for a display-list pass, and answer the channel its build lands
/// on.
///
/// A device and an uploader are `Send + Sync` handles, so the worker builds the
/// pass against the same device the frames draw with.
fn ask_for_pass(
    device: &wgpu::Device,
    uploader: Uploader,
) -> Receiver<Result<crate::display::SceneView, crate::display::SceneError>> {
    let (tx, rx) = mpsc::channel();
    // The three clones buy the worker its own handles and its own end of the
    // channel, because the frame keeps the ones it draws with either way.
    let (worker_tx, worker_device, worker_uploader) =
        (tx.clone(), device.clone(), uploader.clone());
    let spawned = std::thread::Builder::new()
        .name("vv-scene".to_owned())
        .spawn(move || build_pass(&worker_device, worker_uploader, &worker_tx));
    if let Err(err) = spawned {
        // A host that cannot spare a thread still shows the list, and the
        // build then costs the frame what it cost while it sat in one.
        tracing::warn!(err = %err, "display list worker");
        build_pass(device, uploader, &tx);
    }
    rx
}

/// Build one pass, report what it cost, and hand it to the frame loop.
///
/// The report rides `debug` because the build is the one step of a listed
/// source that costs a moment, and the frame it no longer sits in is what a
/// slow-frame report names.
fn build_pass(
    device: &wgpu::Device,
    uploader: Uploader,
    tx: &Sender<Result<crate::display::SceneView, crate::display::SceneError>>,
) {
    let started = Instant::now();
    let built = crate::display::SceneView::new(device, uploader);
    match &built {
        Ok(_) => tracing::debug!(ms = ms_since(started), "display list build"),
        Err(err) => tracing::warn!(err = %err, "display list build"),
    }
    // A dropped receiver means the run ended while the build ran, and the pass
    // goes with this answer.
    let _ = tx.send(built);
}

/// Draw a display list over the window, and answer whether it drew.
///
/// The list is written in surface units and `frame` maps them onto the window,
/// so the primitives render at the zoom and the turn the window shows and the
/// quad pipeline composites the one picture that comes out.
/// The pass is built by a worker on the first list, because vello compiles its
/// compute pipelines when it is built and a run that shows only pictures never
/// pays for that: the frame that asks, and every frame of the build, answer
/// `false` and the caller paints the raster (see [`ListPass`]).
pub(crate) fn draw_list(
    pass: &mut ListPass,
    r: &mut Renderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    list: &Arc<[Primitive]>,
    request: Request<'_>,
) -> bool {
    let Some(view) = pass.ready(device, r) else {
        return false;
    };
    match view.draw(device, queue, list, request) {
        Ok(Some(painted)) => {
            // A frame the pass reused lands where the window it was painted for
            // puts the same surface units (see `SceneView::draw`), so the draw
            // takes the placement the pass answers instead of the whole
            // window.
            r.texture(
                painted.texture,
                painted.src,
                painted.quad,
                Color::WHITE,
                Filter::Linear,
            );
            true
        }
        // The frame asks vello for more than its fixed allocators hold, and a
        // frame it cannot hold paints nothing at all without saying so (see
        // `vv_gpu::display`), so the list answers no draw: the caller shows what
        // it keeps behind the list instead, which is the plane texture and the
        // region patch when a source is there to paint them.
        Ok(None) => false,
        Err(err) => {
            tracing::warn!(err = %err, "display list");
            false
        }
    }
}

/// The region the view marks now: the rect a `v` drag spans, or the kept
/// selection.
///
/// One function answers both the frame draw and the release, so the veil, the
/// border and the committed rect can never read the pointer in different
/// units.
/// The free pointer converts through [`region::Frame::to_image`], the
/// transform the draw inverts with [`region::Frame::to_screen`], and the drag
/// rect lives in source units from there on.
/// The drag wins while it runs, so a new region replaces the old one on the
/// first real move, a press inside the region carries it, and a press that
/// never leaves [`TAP_SLACK`] commits nothing.
pub fn selection_rect(
    frame: Option<region::Frame>,
    kept: Option<region::Crop>,
    drag: Option<region_sel::Drag>,
    mouse: dim::WindowPoint,
) -> Option<region::Crop> {
    let frame = frame.filter(region::Frame::has_drawable_plane)?;
    let Some(drag) = drag else {
        return kept;
    };
    // A press that has not left the tap slack answers nothing new: a span
    // commits no rect, and a move keeps the rect it grabbed.
    if drag.tapped(mouse) {
        return drag.tap_rect().or(kept);
    }
    Some(drag.rect(frame.to_image(mouse), frame.img))
        .filter(|rect| !rect.is_empty())
        .or(kept)
}

/// Darken the picture outside a region selection, and mark the region itself.
///
/// The four bands of [`region_sel::around`] take one veil, so the dim amount
/// matches the grid's dim tint (see [`vv_view::DIM_LEVEL`]).
/// The bands and the border map through [`region::Frame::to_screen`], so they
/// follow the pan, the zoom and the turn of the draw.
pub fn draw_selection(r: &mut Renderer, frame: region::Frame, selection: region::Crop) {
    let outer = frame.plane();
    for band in region_sel::around(outer, selection) {
        let p = band.map(|point| dim::draw_window(frame.to_screen(point)));
        // The pipeline culls nothing, so a band draws in either winding.
        r.fill(Quad::new(p), DIM_VEIL);
    }
    let corners =
        region_sel::corners(selection).map(|point| dim::draw_window(frame.to_screen(point)));
    for i in 0..4 {
        r.line(
            corners[i],
            corners[(i + 1) % 4],
            SELECTION_BORDER_PX,
            SELECTION_BORDER,
        );
    }
}

/// Whether two patches name the same rect, to within half a pixel, and the
/// same scale, to within a fifth of a percent.
///
/// The debounce reads this, so a pan, an ease or a zoom step that stays below
/// those bounds never restarts the quiet timer.
#[must_use]
pub fn same_patch(a: &region::Patch, b: &region::Patch) -> bool {
    (a.rect.min_x() - b.rect.min_x()).abs() <= 0.5
        && (a.rect.min_y() - b.rect.min_y()).abs() <= 0.5
        && (a.rect.width() - b.rect.width()).abs() <= 0.5
        && (a.rect.height() - b.rect.height()).abs() <= 0.5
        && (a.scale - b.scale).abs() <= b.scale * 0.002
}

/// Whether the region patch covers the whole window at the current transform.
///
/// The draw skips the whole-image texture then, because the patch already
/// holds every pixel the window shows.
/// A patch that a pan, a resize or a zoom out has outrun leaves the rest to
/// the whole-image texture until the repaint lands, so the document never
/// drops out of the window while the worker paints.
#[must_use]
pub fn patch_covers(st: &ViewState, win_w: f32, win_h: f32) -> bool {
    let Some(crop) = &st.crop else {
        return false;
    };
    let Some(scale) = st.view_scale else {
        return false;
    };
    if st.img.w <= 0.0 || !scale.is_drawable() {
        return false;
    }
    let Some(frame) = st.frame(Vec2::new(win_w, win_h)) else {
        return false;
    };
    let needed = region::Patch::new(frame.crop(0.0), scale.coarsest());
    region::patch_reaches_window(needed, crop.patch)
}

/// The plane a source takes for a window of `win`, or `None` when its plane does
/// not follow the window.
///
/// A treemap lays itself out for the window it shows in (see
/// [`Surface::frame_window`]), so a resize leaves the view holding a plane of
/// the old shape and a fit that would letterbox it.
/// A picture has one natural size, so it answers `None` and the view reads no
/// header twice.
fn framed_plane(source: &dyn Surface, plane: usize, win: Vec2, img: Size) -> Option<Size> {
    if !source.frame_window(Size { w: win.x, h: win.y }) {
        return None;
    }
    let next = source.bounds(plane).ok()?;
    (!same_plane(next, img)).then_some(next)
}

/// Whether two sizes name the same plane, to within a pixel.
///
/// The image view trusts the plane a grid cell reports while the decode
/// catches up, and it paints a region patch of the plane its source measures.
/// Both are honest only while the two pairs agree: a document's cell holds page
/// 0 and never the whole book, so its picture never covers the book plane and a
/// patch of the cell's guess lands in the wrong place.
#[must_use]
pub fn same_plane(a: Size, b: Size) -> bool {
    (a.w - b.w).abs() <= 1.0 && (a.h - b.h).abs() <= 1.0
}

/// Where the picture a grid cell holds lands on the open plane: the box it
/// covers, in surface units, and the point it turns around.
///
/// A cell keeps one plane: the whole plane of a single-plane source, or page 0
/// of a document (see [`Grid::from_dir`]).
/// A single-plane source maps its picture onto the whole plane, so the later
/// swap to the sharp texture is pixel-aligned.
/// A book stacks every spread into one plane, so one page never covers it: the
/// picture draws into page 0's slot ([`ViewState::cell_box`]), which is where
/// the loader's spread will paint it.
/// Neither place is honest for a book whose slot the view does not know yet: a
/// picture stretched over the whole book would paint one white page across the
/// window.
#[must_use]
pub fn cell_place(
    img: Size,
    cell_box: Option<region::Crop>,
    cell: Size,
) -> Option<(Size, dim::SurfacePoint)> {
    match cell_box {
        Some(slot) => Some((
            Size {
                w: slot.width(),
                h: slot.height(),
            },
            dim::SurfacePoint::new(
                slot.min_x() + slot.width() / 2.0,
                slot.min_y() + slot.height() / 2.0,
            ),
        )),
        None if same_plane(img, cell) => {
            Some((img, dim::SurfacePoint::new(img.w / 2.0, img.h / 2.0)))
        }
        None => None,
    }
}

/// Draw the picture a grid cell holds, while the open plane has no frame of its
/// own.
///
/// The picture lands where [`cell_place`] puts it, so a document's page shows in
/// its slot and never stretched over the whole book.
pub fn draw_cell(r: &mut Renderer, st: &ViewState, tex: &Texture, cell: Size, win: Vec2) {
    if st.img.w <= 0.0 {
        return;
    }
    let scale = st
        .view_scale
        .unwrap_or_else(|| region::Scale::uniform((win.x / st.img.w).min(win.y / st.img.h)));
    let Some((size, middle)) = cell_place(st.img, st.cell_box, cell) else {
        return;
    };
    let frame = st.frame_at(win, st.pan, scale);
    draw_turned(
        r,
        tex,
        size,
        scale.coarsest(),
        dim::draw_window(frame.to_screen(middle)),
        st.angle,
        Filter::Linear,
    );
}

/// Keep the pivot of a turned image on its window spot.
///
/// The angle already ran from `from` to the value `st.angle` shows, so the code
/// shifts the pan and its target by the distance that the plain centered turn
/// would move the pivot.
/// The pivot stays a draw transform: no pixel moves.
pub fn keep_pivot(st: &mut ViewState, from: f32, win_w: f32, win_h: f32) {
    let Some(pivot) = st.pivot else {
        return;
    };
    let center = Vec2::new(win_w / 2.0, win_h / 2.0);
    let pan = rotate::pivot(pivot, center, from, st.angle, st.pan);
    let shift = pan - st.pan;
    st.pan = pan;
    st.target_pan += shift;
}

/// Upload a tiny blurred copy as the image-view background (`VV_BLUR_BG`
/// gimmick).
/// `tag` identifies the source image (grid entry id, or None for a
/// single-file launch), so grid crossfades can skip redundant transitions.
/// The function does nothing when the gimmick is off or the copy is missing.
pub fn attach_blur_bg(
    blur_bg: &mut Option<BlurBg>,
    uploader: &Uploader,
    blur: Option<&DecodedImage>,
    tag: Option<u64>,
) {
    if let (Some(bg), Some(data)) = (blur_bg.as_mut(), blur) {
        bg.attach(uploader, data, tag);
    }
}

/// Put the viewed texture back into its grid entry, if it came from one, and
/// clear the view texture.
/// The code calls this when leaving image mode or switching to another grid
/// entry.
pub fn put_back_view(
    grid: &mut Option<Grid>,
    view_from_grid: &mut Option<u64>,
    view_tex: &mut Option<Texture>,
) {
    if let Some(id) = view_from_grid.take()
        && let Some(e) = grid
            .as_mut()
            .and_then(|g| g.entries.iter_mut().find(|e| e.id == id))
    {
        e.full = view_tex.take();
        e.viewing = false;
    }
    *view_tex = None;
}

/// Three paths share this code: grid-open, prev/next navigation and the
/// decode-landed takeover.
/// 1. The code kept the full-res texture for it (keep-set): show at once.
/// 2. A decode is in flight (`queued`): wait for the decode-landed takeover in
///    a later frame.
/// 3. Otherwise: start the streaming loader (JXL: blurry preview fast).
///
/// On paths 2 and 3 the draw loop shows the entry thumb (whole image,
/// aspect preserved) as a placeholder.
/// n/p navigation then never flashes black while a decode catches up.
///
/// With `set_scale_now`, the code sets the initial fit scale at once,
/// because the caller frame skips the per-frame scale math (it ran earlier).
/// `win` is the window size in pixels.
pub fn show_entry(
    st: &mut ViewState,
    grid: &mut Grid,
    idx: usize,
    r: &Renderer,
    win: (WinW, WinH),
    set_scale_now: bool,
) {
    let (win_w, win_h) = (win.0.raw(), win.1.raw());
    let (id, tex, w, h, source, queued, thumb, blur, no_full) = {
        let e = &mut grid.entries[idx];
        (
            e.id,
            e.full.take(),
            e.width,
            e.height,
            e.source.clone(),
            e.queued,
            e.cell.thumb().is_some(),
            // The copy is a tiny texture (`VV_BLUR_PX` long side, 128 by
            // default), and the entry keeps its own for a later open, because
            // a return to the grid and back reuses it.
            e.blur.clone(),
            e.no_full,
        )
    };
    st.open_id = Some(id);
    st.mode = Mode::Image;
    st.cell_box = None;
    st.reset_view();
    // A new image starts with no region patch: the source request names what
    // the window shows.
    st.request_source(&source);
    if let Some(tex) = tex {
        // A kept full-res texture is a whole plane, never a book.
        st.open_box = None;
        grid.entries[idx].viewing = true;
        st.view_from_grid = Some(id);
        st.view_tex = Some(tex);
        st.img = Size {
            w: w.as_f32(),
            h: h.as_f32(),
        };
        st.loader = None;
        st.view_loading = false;
        if set_scale_now {
            st.view_scale = Some(region::Scale::uniform(
                (win_w / st.img.w).min(win_h / st.img.h),
            ));
        }
        attach_blur_bg(&mut st.blur_bg, &r.uploader(), blur.as_ref(), st.open_id);
    } else {
        st.open_box = None;
        st.view_from_grid = None;
        st.view_tex = None;
        st.view_loading = true;
        // Whenever the entry thumb exists, the decode that produced it
        // also recorded the full dimensions.
        // The cache header of the placeholder pass stores the same pair, so a
        // warm entry holds them before any decode.
        // Set them now, so the draw loop can show the thumb (or its
        // placeholder) at the final transform (instead of a black frame) while
        // the full-res decode catches up.
        if w.raw() > 0 {
            st.img = Size {
                w: w.as_f32(),
                h: h.as_f32(),
            };
            if set_scale_now {
                st.view_scale = Some(region::Scale::uniform(
                    (win_w / st.img.w).min(win_h / st.img.h),
                ));
            }
        } else {
            // The dimensions arrive with the header or the texture.
            st.img = Size { w: 0.0, h: 0.0 };
        }
        // Whether this open waits for the cell's own decode: a picture waits,
        // because the texture that decode makes *is* the picture, and the thumb
        // holds the place until it lands.
        // A document waits for nothing, because its cell holds page 0 alone
        // while the view shows the whole book (see [`crate::book`]): a decode of
        // that cell can never answer this view, and waiting for it would show
        // that page in the middle of the window before a later frame moved it
        // into the right slot of its spread, which is the jump a grid open used
        // to show.
        // The header alone says whether the file can be a document, so a cell
        // that waits reads no more than it already has.
        let waits = (queued || (thumb && !no_full)) && !source.may_be_a_book();
        st.loader = if waits {
            None
        } else {
            // The placeholder takes the long side of the window, so a thumb
            // never upscales past the cell it fills.
            let long_side = win.0.raw().max(win.1.raw());
            let preview_px = PreviewPx::new(long_side.max(1.0) as u32);
            // The image view opens a document as a book, so plane 0 holds
            // every spread.
            // The opening box comes from the surface before the loader takes
            // it, so the fit frames the first spread.
            match source.open_book() {
                Ok((surface, plane)) => {
                    // The opened plane measures the view, never the cell: a book
                    // stacks every spread, and its page 0 fills one slot of the
                    // first one.
                    // Both numbers land here, so the first draw frames the
                    // spread and puts the cell picture in that slot, and the
                    // loader's header changes neither.
                    st.open_box = surface.open_box(plane);
                    if let Ok(plane) = surface.bounds(plane) {
                        st.img = plane;
                        // The fit measures the plane the surface reports, so the
                        // first frame frames the spread instead of easing from
                        // the cell's fit.
                        st.refit_open();
                    }
                    st.cell_box = surface.page_box(0);
                    Some(Loader::start_plane(
                        surface,
                        plane,
                        preview_px,
                        &Some(r.uploader()),
                    ))
                }
                Err(err) => {
                    tracing::warn!(source = source.log_label(), err = %err, "open failed");
                    grid.mark_failed(id, DecodeFailure::Decode);
                    None
                }
            }
        };
        attach_blur_bg(&mut st.blur_bg, &r.uploader(), blur.as_ref(), st.open_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{surface::SurfaceError, theme};

    /// The theme's paper is the window background, so a recolored page runs to
    /// the edge of the window without a letterbox of another shade.
    #[test]
    fn the_recolor_paper_is_the_window_background() {
        assert_eq!(theme::THEME.paper, BACKGROUND);
    }

    #[test]
    fn a_treemap_answers_the_plane_its_window_names() {
        use vv_source::treemap::{TreemapSurface, sample_tree};

        // A treemap divides the window itself, so its plane takes the window's
        // shape and a fit of it leaves no background bar.
        let source = TreemapSurface::from_tree(sample_tree());
        let held = source.bounds(0).expect("the plane");
        let win = Vec2::new(1920.0, 1080.0);
        assert_eq!(
            framed_plane(&source, 0, win, held),
            Some(Size {
                w: 1920.0,
                h: 1080.0,
            })
        );
        // The same window twice measures the same plane, so nothing repaints.
        let landed = Size {
            w: 1920.0,
            h: 1080.0,
        };
        assert_eq!(framed_plane(&source, 0, win, landed), None);
    }

    #[test]
    fn a_picture_keeps_its_plane_whatever_the_window() {
        // A raster has one natural size, so a resize answers no new plane and
        // the view reads no header twice.
        let image =
            DecodedImage::from_exact(dim::ImgW::new(4), dim::ImgH::new(4), vec![0_u8; 4 * 4 * 4]);
        let source = vv_source::surface::MemorySurface::new(image);
        let held = source.bounds(0).expect("the plane");
        assert_eq!(
            framed_plane(&source, 0, Vec2::new(1920.0, 1080.0), held),
            None
        );
    }

    #[test]
    fn a_books_cell_picture_never_covers_its_plane() {
        // A grid cell of a document holds page 0, and the image view opens the
        // whole book, so the cell picture measures neither the plane nor the
        // slot the page fills.
        let page = Size { w: 612.0, h: 792.0 };
        let slot = region::crop(613.0, 0.0, 612.0, 792.0);
        let book = Size {
            w: 1224.5,
            h: 8712.0,
        };
        // The slot the page fills, never the plane.
        assert_eq!(
            cell_place(book, Some(slot), page),
            Some((
                page,
                dim::SurfacePoint::new(slot.min_x() + 306.0, slot.min_y() + 396.0)
            ))
        );
        // A book whose slot the view does not know yet paints no stretched
        // page: one white page over the whole book reads as a white flash.
        assert_eq!(cell_place(book, None, page), None);
        // A single-plane source covers its own plane, so its picture maps onto
        // the middle of it and the sharp texture later lands on the same
        // pixels.
        assert_eq!(
            cell_place(page, None, page),
            Some((page, dim::SurfacePoint::new(306.0, 396.0)))
        );
    }

    #[test]
    fn a_patch_that_matches_the_view_lands_on_whole_pixels() {
        // A patch of 1532x1180 px whose screen center falls between pixels.
        let (cx, cy) = (512.32f32, 384.68f32);
        let dest = patch_dest(
            dim::WindowPoint::new(cx, cy),
            1532.0,
            1180.0,
            1532.0,
            1180.0,
            0.0,
        )
        .unwrap();
        // The corner snaps to the nearest whole pixel, and the patch keeps the
        // pixel size it was painted at.
        assert_eq!(
            (dest.min_x(), dest.min_y()),
            ((cx - 766.0).round(), (cy - 590.0).round())
        );
        assert_eq!((dest.width(), dest.height()), (1532.0, 1180.0));
        // Every edge sits on a whole window pixel, so the bilinear sample lands
        // on the texel centers of a 1:1 patch.
        assert_eq!(dest.min_x().fract(), 0.0);
        assert_eq!(dest.min_y().fract(), 0.0);
        assert_eq!((dest.max_x().fract(), dest.max_y().fract()), (0.0, 0.0));
    }

    #[test]
    fn a_patch_rounds_into_the_painted_pixel_size() {
        // The paint rounds its buffer, so a screen size within a pixel takes
        // the painted size instead of resampling the whole patch.
        let center = dim::WindowPoint::new(400.0, 300.0);
        let dest = patch_dest(center, 800.6, 600.4, 801.0, 600.0, 0.0).unwrap();
        assert_eq!((dest.width(), dest.height()), (801.0, 600.0));
        assert_eq!((dest.min_x(), dest.min_y()), (0.0, 0.0));
    }

    #[test]
    fn a_stretched_or_turned_patch_keeps_its_true_size() {
        // A zoom past the slack leaves the patch stretched, so its pixel size
        // is not the screen size: the size follows the view, and the position
        // still snaps so sampling stays stable.
        let center = dim::WindowPoint::new(400.4, 300.6);
        assert_eq!(
            patch_dest(center, 1200.0, 900.0, 800.0, 600.0, 0.0),
            Some(dim::rect(-200.0, -149.0, 1200.0, 900.0))
        );
        // A turned patch never lands on the pixel grid.
        assert_eq!(patch_dest(center, 800.0, 600.0, 800.0, 600.0, 3.0), None);
        // A degenerate rect has no destination.
        assert_eq!(patch_dest(center, 0.0, 600.0, 800.0, 600.0, 0.0), None);
    }

    /// A `Column` over a `h` unit plane in a `win_h` pixel window.
    fn column(h: f32, scale: f32, win_h: f32) -> Column {
        Column::new(Size { w: h, h }, scale, win_h)
    }

    #[test]
    fn the_top_pan_puts_the_plane_top_on_the_window_top() {
        // The draw centers the plane, so the pan offsets from the middle.
        let (h, scale, win_h) = (4752.0f32, 0.5f32, 768.0f32);
        let pan = column(h, scale, win_h).top_pan();
        let top = (win_h - h * scale) / 2.0 + pan;
        assert_eq!(top, 0.0);
        assert!(pan > 0.0, "a tall plane scrolls down, pan {pan}");
    }

    #[test]
    fn a_plane_that_fits_stays_centered() {
        assert_eq!(column(200.0, 1.0, 768.0).top_pan(), 0.0);
    }

    #[test]
    fn a_spread_step_frames_a_row_like_the_open_frames_the_first() {
        // A book of 40 letter pages, fit into the window.
        let (h, scale, win_h, row_h) = (16632.0f32, 0.9f32, 768.0f32, 792.0f32);
        let col = column(h, scale, win_h);
        assert_eq!(col.pan_of(0.0), col.top_pan());
        for row in [0, 1, 3, 20] {
            let pan = col.pan_of(row as f32 * row_h);
            // The top of that spread lands on the top edge of the window.
            let top = (win_h - h * scale) / 2.0 + pan + row as f32 * row_h * scale;
            assert!(top.abs() < 1e-3, "row {row} top {top}");
            assert_eq!(col.spread_at_top(pan, row_h), row);
        }
    }

    #[test]
    fn a_row_reads_from_the_aimed_pan_not_the_pixels() {
        // Half a spread of travel still names the spread the pan aims at, so a
        // step pressed while an earlier step glides never stalls.
        let (h, scale, win_h, row_h) = (4000.0f32, 0.5f32, 768.0f32, 800.0f32);
        let col = column(h, scale, win_h);
        let row = 2;
        let pan = col.pan_of(row as f32 * row_h);
        assert_eq!(col.spread_at_top(pan, row_h), row);
        // Above the plane top the row stays the first one.
        assert_eq!(col.spread_at_top(col.top_pan() + 500.0, row_h), 0);
        // A book that fits the window has no row to read.
        assert_eq!(column(200.0, 1.0, 768.0).spread_at_top(0.0, 100.0), 0);
    }

    #[test]
    fn a_region_drag_wins_over_the_kept_region() {
        // The window holds the plane one to one, so the pointer and the source
        // units agree here and the test reads the region alone.
        let frame = region::Frame {
            img: dim::SurfaceSize::new(1000.0, 500.0),
            win: dim::WindowSize::new(1000.0, 500.0),
            pan: dim::WindowVector::zero(),
            scale: region::Scale::uniform(1.0),
            angle: 0.0,
        };
        let kept = region::crop(10.0, 10.0, 100.0, 100.0);
        let drag = region_sel::Drag::new(
            dim::SurfacePoint::new(200.0, 100.0),
            dim::WindowPoint::zero(),
            false,
        );
        let still =
            region_sel::Drag::new(dim::SurfacePoint::zero(), dim::WindowPoint::zero(), false);
        // A press that has not left the tap slack spans no rect, so the old
        // region keeps its place on screen.
        assert_eq!(
            selection_rect(
                Some(frame),
                Some(kept),
                Some(still),
                dim::WindowPoint::zero()
            ),
            Some(kept)
        );
        // Any travel replaces it with the fresh rect.
        assert_eq!(
            selection_rect(
                Some(frame),
                Some(kept),
                Some(drag),
                dim::WindowPoint::new(500.0, 400.0)
            ),
            Some(region::crop(200.0, 100.0, 300.0, 300.0))
        );
        // A plane with no size and no drag marks nothing.
        assert_eq!(
            selection_rect(None, Some(kept), None, dim::WindowPoint::zero()),
            None
        );
    }

    #[test]
    fn a_region_drag_reads_the_pointer_in_source_units() {
        // Half scale and a pan of 100 window pixels to the right: the window
        // middle sits on the plane middle, one window pixel spans two source
        // pixels, and the pan moves the picture the other way.
        let frame = region::Frame {
            img: dim::SurfaceSize::new(1000.0, 500.0),
            win: dim::WindowSize::new(800.0, 600.0),
            pan: dim::WindowVector::new(100.0, 0.0),
            scale: region::Scale::uniform(0.5),
            angle: 0.0,
        };
        let drag = region_sel::Drag::new(
            dim::SurfacePoint::new(500.0, 250.0),
            dim::WindowPoint::zero(),
            false,
        );
        // The pointer at (600, 200) lands on the source point (700, 50), so the
        // rect runs from the anchor to there.
        // A raw window reading would answer (500, 200, 100, 50) instead.
        assert_eq!(
            selection_rect(
                Some(frame),
                None,
                Some(drag),
                dim::WindowPoint::new(600.0, 200.0)
            ),
            Some(region::crop(500.0, 50.0, 200.0, 200.0))
        );
        // A press that has not left the tap slack keeps the old region, so a
        // `v` click never flashes a sliver rect.
        let tap = region_sel::Drag::new(
            dim::SurfacePoint::new(500.0, 250.0),
            dim::WindowPoint::new(600.0, 200.0),
            false,
        );
        let kept = region::crop(10.0, 10.0, 100.0, 100.0);
        assert_eq!(
            selection_rect(
                Some(frame),
                Some(kept),
                Some(tap),
                dim::WindowPoint::new(601.0, 200.0)
            ),
            Some(kept)
        );
        // The release reads the same call, so a tap answers no region and
        // drops the kept one.
        assert_eq!(
            selection_rect(
                Some(frame),
                None,
                Some(tap),
                dim::WindowPoint::new(601.0, 200.0)
            ),
            None
        );
    }

    /// A source that lists one box of its plane, and remembers what it was
    /// asked for, so a test can watch the box follow the window.
    struct Boxed {
        /// Every box the source was asked for, so a test can count the builds.
        asked: Arc<std::sync::Mutex<Vec<Crop>>>,
        /// Whether the source answers a list at all.
        lists: bool,
    }

    impl Surface for Boxed {
        fn plane_count(&self) -> usize {
            1
        }

        fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
            if plane != 0 {
                return Err(SurfaceError::Plane(plane));
            }
            Ok(Size { w: 100.0, h: 800.0 })
        }

        fn render_whole(&self, _plane: usize, _scale: f32) -> Result<DecodedImage, SurfaceError> {
            Err(SurfaceError::Unsupported(
                "a list source barely rasterizes".into(),
            ))
        }

        fn progressive(
            &self,
        ) -> Result<Option<Box<dyn crate::surface::ChunkSource>>, SurfaceError> {
            Ok(None)
        }

        fn display(
            &self,
            _plane: usize,
        ) -> Result<Option<Arc<[crate::scene::Primitive]>>, SurfaceError> {
            Ok(None)
        }

        fn display_box(
            &self,
            plane: usize,
            part: Crop,
        ) -> Result<Option<Arc<[crate::scene::Primitive]>>, SurfaceError> {
            self.bounds(plane)?;
            self.asked.lock().expect("the asks").push(part);
            if !self.lists {
                return Ok(None);
            }
            Ok(Some(Arc::from(vec![crate::scene::Primitive::Rect {
                rect: crate::scene::rect_of(part),
                paint: crate::scene::Paint::Solid(Color::new(1, 2, 3, 255)),
            }])))
        }

        fn lists_a_box(&self) -> bool {
            true
        }

        fn can_refine(&self) -> bool {
            true
        }
    }

    /// A source that answers one rect, so a test can watch the list land.
    struct Listed;

    impl Surface for Listed {
        fn plane_count(&self) -> usize {
            1
        }

        fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
            if plane != 0 {
                return Err(SurfaceError::Plane(plane));
            }
            Ok(Size { w: 100.0, h: 50.0 })
        }

        fn render_whole(&self, _plane: usize, _scale: f32) -> Result<DecodedImage, SurfaceError> {
            Err(SurfaceError::Unsupported(
                "a list source barely rasterizes".into(),
            ))
        }

        fn progressive(
            &self,
        ) -> Result<Option<Box<dyn crate::surface::ChunkSource>>, SurfaceError> {
            Ok(None)
        }

        fn display(
            &self,
            _plane: usize,
        ) -> Result<Option<Arc<[crate::scene::Primitive]>>, SurfaceError> {
            Ok(Some(Arc::from(vec![crate::scene::Primitive::Rect {
                rect: crate::scene::rect_of(region::crop(0.0, 0.0, 100.0, 50.0)),
                paint: crate::scene::Paint::Solid(Color::new(1, 2, 3, 255)),
            }])))
        }

        fn can_refine(&self) -> bool {
            true
        }
    }

    /// A source whose first build waits for the test, so the plane can move
    /// while the answer is still in flight.
    struct Gated {
        gate: Arc<std::sync::Barrier>,
        blocked: std::sync::atomic::AtomicBool,
    }

    impl Surface for Gated {
        fn plane_count(&self) -> usize {
            1
        }

        fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
            if plane != 0 {
                return Err(SurfaceError::Plane(plane));
            }
            Ok(Size { w: 100.0, h: 50.0 })
        }

        fn render_whole(&self, _plane: usize, _scale: f32) -> Result<DecodedImage, SurfaceError> {
            Err(SurfaceError::Unsupported(
                "a list source barely rasterizes".into(),
            ))
        }

        fn progressive(
            &self,
        ) -> Result<Option<Box<dyn crate::surface::ChunkSource>>, SurfaceError> {
            Ok(None)
        }

        fn display(
            &self,
            _plane: usize,
        ) -> Result<Option<Arc<[crate::scene::Primitive]>>, SurfaceError> {
            if !self.blocked.swap(true, std::sync::atomic::Ordering::SeqCst) {
                self.gate.wait();
            }
            Ok(Some(Arc::from(vec![crate::scene::Primitive::Rect {
                rect: crate::scene::rect_of(region::crop(0.0, 0.0, 100.0, 50.0)),
                paint: crate::scene::Paint::Solid(Color::new(1, 2, 3, 255)),
            }])))
        }

        fn can_refine(&self) -> bool {
            true
        }
    }

    /// A fresh view with a channel pair and no open source.
    fn view_state() -> ViewState {
        let (crop_tx, crop_rx) = std::sync::mpsc::channel();
        ViewState {
            mode: Mode::Image,
            open_id: None,
            view_from_grid: None,
            view_tex: None,
            view_loading: false,
            img: Size { w: 0.0, h: 0.0 },
            zoom: ZoomMode::FitDown,
            pan: Vec2::ZERO,
            target_pan: Vec2::ZERO,
            view_scale: None,
            viewport: None,
            zoom_anchor: None,
            angle: 0.0,
            pivot: None,
            selection: None,
            ramps: Ramps::default(),
            pixelated: false,
            loader: None,
            blur_bg: None,
            source: None,
            source_plane: 0,
            loader_plane: None,
            plane_stale: false,
            list_drawn: false,
            open_box: None,
            cell_box: None,
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
            scene: None,
            scene_box: None,
            scene_wanted: false,
            scene_pending: false,
        }
    }

    /// Drain the worker channel until the list lands, or the wait gives up.
    fn wait_for_scene(st: &mut ViewState) -> bool {
        for _ in 0..200 {
            st.drain_crops();
            if st.scene.is_some() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        false
    }

    /// A stated viewport is the fit under every zoom mode, so `W`, `E` and `T`
    /// keep the producer's framing instead of replacing it; a view that states
    /// none answers the mode's own ratio.
    #[test]
    fn a_stated_viewport_is_the_fit_under_every_mode() {
        let mut st = view_state();
        st.img = Size { w: 200.0, h: 100.0 };
        st.viewport = Some(region::crop(50.0, 20.0, 200.0, 120.0));
        let win = Vec2::new(900.0, 600.0);
        let boxed = region::Scale::axes(4.5, 5.0);
        for zoom in [
            ZoomMode::FitDown,
            ZoomMode::FitAll,
            ZoomMode::FitWidth,
            ZoomMode::FitHeight,
            ZoomMode::Fill,
            ZoomMode::Free(region::Scale::uniform(2.0)),
        ] {
            st.zoom = zoom;
            assert_eq!(
                st.fit_scale(win),
                boxed,
                "the box is the fit under {zoom:?}"
            );
        }
        // Without a box the modes answer their own ratio again, and a fit never
        // upscales under `FitDown`.
        let mut plain = view_state();
        plain.img = Size { w: 200.0, h: 100.0 };
        let small = Vec2::new(100.0, 100.0);
        plain.zoom = ZoomMode::FitDown;
        assert_eq!(plain.fit_scale(small), region::Scale::uniform(0.5));
        plain.zoom = ZoomMode::FitWidth;
        assert_eq!(plain.fit_scale(small), region::Scale::uniform(0.5));
        plain.zoom = ZoomMode::Fill;
        assert_eq!(plain.fit_scale(small), region::Scale::uniform(1.0));
    }

    /// A stated viewport fills the window with its own box on each axis, so the
    /// box is the fit, a resize re-derives both ratios, and the pan moves off the
    /// placement rather than carrying it.
    #[test]
    fn a_stated_viewport_fills_the_window_with_its_box() {
        let mut st = view_state();
        st.img = Size { w: 200.0, h: 100.0 };
        // A box of 200 by 120: wider than tall in window terms, and not the
        // plane's own shape.
        st.viewport = Some(region::crop(50.0, 20.0, 200.0, 120.0));
        let win = Vec2::new(900.0, 600.0);
        let scale = st.viewport_scale(win).expect("a stated box is the fit");
        assert_eq!(scale, region::Scale::axes(4.5, 5.0), "one ratio per axis");
        let frame = st.frame_at(win, Vec2::ZERO, scale);
        // The box corners land on the window corners, so the box is what the
        // window shows.
        for (boxed, window) in [((50.0, 20.0), (0.0, 0.0)), ((250.0, 140.0), (900.0, 600.0))] {
            let at = frame.to_screen(dim::SurfacePoint::new(boxed.0, boxed.1));
            assert!(
                (at.x - window.0).abs() < 1e-3 && (at.y - window.1).abs() < 1e-3,
                "the box corner {boxed:?} lands on {window:?}: {at:?}"
            );
        }
        // A window of another shape re-derives both ratios and keeps the box.
        let narrow = Vec2::new(450.0, 600.0);
        let apart = st.viewport_scale(narrow).expect("the box is still the fit");
        assert_eq!(apart, region::Scale::axes(2.25, 5.0));
        let frame = st.frame_at(narrow, Vec2::ZERO, apart);
        let corner = frame.to_screen(dim::SurfacePoint::new(50.0, 20.0));
        assert!(corner.x.abs() < 1e-3 && corner.y.abs() < 1e-3);
        // The user's pan moves the picture off the stated placement, because the
        // placement is derived every frame rather than stored.
        let moved = st.frame_at(narrow, Vec2::new(30.0, -10.0), apart);
        let corner = moved.to_screen(dim::SurfacePoint::new(50.0, 20.0));
        assert!((corner.x - 30.0).abs() < 1e-3 && (corner.y + 10.0).abs() < 1e-3);
        // A view that states nothing keeps the uniform fit and places nothing.
        let plain = view_state();
        assert_eq!(plain.viewport_scale(win), None);
        assert_eq!(plain.view_pan(region::Scale::uniform(2.0)), Vec2::ZERO);
    }

    /// A document whose display list drew the frame before asks no worker for a
    /// region patch: the list is the picture, at any zoom and sharper than a
    /// patch, so the raster is only the fallback of a frame the list refuses.
    ///
    /// The gate is what keeps a spread step off the raster path: the step
    /// prefetches its landing patch at the press edge, which is the one moment
    /// the main thread is encoding the list for the band it just left.
    #[test]
    fn a_listed_document_paints_no_patch() {
        let mut st = view_state();
        st.img = Size { w: 100.0, h: 50.0 };
        st.view_scale = Some(region::Scale::uniform(1.0));
        st.source = Some(Arc::new(Listed));
        let win = Vec2::new(100.0, 50.0);
        assert!(
            st.spread_patch(win).is_some(),
            "a source that has drawn no list prefetches its landing patch"
        );
        st.list_drawn = true;
        assert!(
            !st.wants_patch() && st.spread_patch(win).is_none(),
            "a list that drew the frame asks no worker for a patch"
        );
        st.list_drawn = false;
        assert!(
            st.wants_patch() && st.spread_patch(win).is_some(),
            "a frame the list refused falls back to the raster again"
        );
        // A source closed and reopened starts with no list drawn, so its first
        // frame paints the patch it always painted.
        st.list_drawn = true;
        st.close_crop();
        assert!(st.wants_patch(), "an open source has drawn no list yet");
    }

    #[test]
    fn a_list_lands_once_the_plane_is_measured() {
        // The launch measures the plane and then adopts the source; a list asked
        // for before the measure answers no plane and would be dropped.
        let mut st = view_state();
        st.img = Size { w: 100.0, h: 50.0 };
        st.adopt_source(Some(Arc::new(Listed)));
        assert!(
            wait_for_scene(&mut st),
            "the list of the measured plane lands"
        );
    }

    /// A source that lists one box is asked for the box the window shows, and
    /// asked again only when the window leaves that box.
    ///
    /// A book is one plane of every page, so a list of the plane is the whole
    /// document: the view asks for the band of the frame (see
    /// `display::band_for`), which is the box the pass culls its own encode to,
    /// and it keeps that list while the pan stays inside it.
    #[test]
    fn a_boxed_list_follows_the_window() {
        let asked = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut st = view_state();
        st.img = Size { w: 100.0, h: 800.0 };
        st.view_scale = Some(region::Scale::uniform(1.0));
        let win = Vec2::new(200.0, 200.0);
        // A window 200 pixels tall over a plane 800 units tall at one to one: this
        // pan puts the top spread under it.
        st.pan = Vec2::new(0.0, 300.0);
        // The adopt has no window, so it asks for nothing yet.
        st.adopt_source(Some(Arc::new(Boxed {
            asked: Arc::clone(&asked),
            lists: true,
        })));
        assert!(st.scene_wanted, "the first frame carries the box");
        assert!(
            asked.lock().expect("the asks").is_empty(),
            "an adopt asks no worker without a window"
        );

        // The first frame names the box, and the list lands.
        st.request_box_scene(win);
        assert!(wait_for_scene(&mut st), "the list of that box lands");
        let first = asked.lock().expect("the asks").clone();
        assert_eq!(first.len(), 1, "one build for one box");
        let band = first[0];
        assert!(
            band.contains(dim::surface_point(Vec2::new(50.0, 200.0))),
            "the box holds the pixel the window's middle shows: {band:?}"
        );

        // A pan inside the box asks nothing: the box is padded by half the
        // window, which is what a glide over one spread costs.
        st.pan = Vec2::new(0.0, 200.0);
        st.request_box_scene(win);
        assert!(st.scene.is_some(), "the list on screen still holds");
        assert_eq!(
            asked.lock().expect("the asks").len(),
            1,
            "a pan inside the box asks for no build"
        );

        // A pan out of the box asks for the box the window shows now.
        st.pan = Vec2::new(0.0, 150.0);
        st.scene = None;
        st.request_box_scene(win);
        assert!(wait_for_scene(&mut st), "the list of the new box lands");
        let asks = asked.lock().expect("the asks").clone();
        assert_eq!(asks.len(), 2, "one build for the box the window left");
        assert!(
            asks[1].contains(dim::surface_point(Vec2::new(50.0, 250.0))),
            "the new box holds the pixel the window shows now: {:?}",
            asks[1]
        );
    }

    /// A source that answers no list for a box keeps the raster path for that box
    /// instead of asking a worker again on every frame.
    ///
    /// The frame asks whenever the box the window shows is not one it already
    /// asked for, so an answer of "none" must count as the answer for that box:
    /// a page a list cannot express would otherwise cost a build per frame.
    #[test]
    fn a_box_that_lists_nothing_is_asked_once() {
        let asked = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut st = view_state();
        st.img = Size { w: 100.0, h: 800.0 };
        st.view_scale = Some(region::Scale::uniform(1.0));
        let win = Vec2::new(200.0, 200.0);
        st.pan = Vec2::new(0.0, 300.0);
        st.adopt_source(Some(Arc::new(Boxed {
            asked: Arc::clone(&asked),
            lists: false,
        })));
        for _ in 0..3 {
            st.request_box_scene(win);
            // The answer lands before the frame after it, so a loop would show
            // up as a third ask.
            for _ in 0..200 {
                st.drain_crops();
                if !st.scene_pending {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
        assert!(st.scene.is_none(), "the box answers no list");
        assert_eq!(
            asked.lock().expect("the asks").len(),
            1,
            "a box that lists nothing is asked once"
        );
    }

    #[test]
    fn a_list_of_an_earlier_plane_is_asked_again() {
        // A plane that follows the window (a treemap) moves on the first frame,
        // and every plane change bumps the paint generation, so the list of the
        // plane before it drops. The answer must clear the pending flag and ask
        // for the plane the view holds now, or the source never lists again.
        let mut st = view_state();
        st.img = Size { w: 100.0, h: 50.0 };
        st.adopt_source(Some(Arc::new(Listed)));
        assert!(wait_for_scene(&mut st), "the first list lands");
        // The window moves, so the plane moves and the view drops its patch.
        st.drop_patch();
        st.img = Size { w: 100.0, h: 50.0 };
        st.scene = None;
        st.request_scene();
        assert!(
            wait_for_scene(&mut st),
            "the list of the plane the view holds now lands"
        );
    }

    #[test]
    fn a_stale_answer_asks_again_instead_of_wedging() {
        // The sequence every plane-shaped source takes: the launch asks for a
        // list, the first frame re-frames the plane and bumps the paint
        // generation, and only then does the launch's answer arrive.
        // An answer that left the pending flag set would refuse every later ask,
        // and the source would stay on the region patch for good.
        let gate = Arc::new(std::sync::Barrier::new(2));
        let surface = Arc::new(Gated {
            gate: Arc::clone(&gate),
            blocked: std::sync::atomic::AtomicBool::new(false),
        });
        let mut st = view_state();
        st.img = Size { w: 100.0, h: 50.0 };
        // The worker enters the build and holds it.
        st.adopt_source(Some(surface));
        assert!(st.scene_pending, "a build is in flight");
        // The first frame measures a plane of another shape.
        st.drop_patch();
        st.img = Size { w: 120.0, h: 60.0 };
        // Let the launch's build finish, now that it describes another plane.
        gate.wait();
        assert!(
            wait_for_scene(&mut st),
            "the stale answer asks again for the plane the view holds"
        );
        assert!(!st.scene_pending, "no build is left in flight");
    }
}
