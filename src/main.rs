// Pixel and coordinate math lives in f32 (raylib units) and indices in
// usize and u32.
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
//! `VV_TIMING`=1 traces main-thread texture uploads and slow frames.
//! `VV_SLOW_STREAM`=1 slows the JXL stream.
//! `VV_BLUR_BG`=1 draws a blurred copy of the viewed image as the image-view
//! background, scaled to cover the window and upscaled by the GPU.
//! `VV_BG_DIM`=0..1 sets its brightness (default 0.6).
//! `VV_BLUR_PX` sets the blur resolution, the long side of the tiny texture.
//! The default is 128, and fewer pixels give a blurrier result.
//!
//! `vv --mcp <path>` also serves the Model Context Protocol on stdio.
//! The viewer window stays interactive; the server owns stdin and stdout.

use std::{
    borrow::Cow,
    env, fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        mpsc::{Receiver, Sender, channel},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use versatile_viewer::{
    DIM_VEIL, DecodedImage, TAP_SLACK,
    blurbg::{self, BlurBg},
    book::BookSurface,
    dim::{PreviewPx, RampTime, Rect, WinH, WinW},
    draw::{Color, Quad, Turn, Vec2},
    gpu::{FrameError, Gpu, GpuSettings, Present},
    grid::{DecodeFailure, EntrySource, Grid, GridAction, LoadPlan},
    input::Input,
    loader::{DecodeMsg, Loader},
    mcp, region, region_sel,
    render::Renderer,
    rotate,
    surface::{Size, Surface, View, open_book_surface, open_surface},
    texture::{Filter, Texture, Uploader},
    tween::Ramp,
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

/// The window background color, `#062329`.
/// The grid and the image view both clear to it, so the letterbox bars and the
/// gaps between thumbnails match.
/// A source that paints no paper carries its own paper in its pixels (see
/// `src/mermaid.rs`), so the window color never has to follow a picture.
const BACKGROUND: Color = Color::new(0x06, 0x23, 0x29, 0xff);

/// The border that marks a region selection.
/// It stays bright over the veil, so a region reads on a dark picture too.
const SELECTION_BORDER: Color = Color::new(0xff, 0xff, 0xff, 0xff);

/// The width of the region border, in window pixels.
const SELECTION_BORDER_PX: f32 = 2.0;

/// The fraction of the remaining distance to the window center that the
/// zoomed image point (and the cursor riding it) drifts over one zoom ease.
/// It shares the exponential time constant of the ease.
const ZOOM_ANCHOR_DRIFT: f32 = 0.25;

/// The time a held-key motion ramps in or out, in seconds.
/// A hold reaches cruise speed after this time, and a release coasts to rest
/// over it.
const RAMP_TIME: RampTime = RampTime::new(0.36);

/// The pan cruise speed, in window long sides per second.
/// 0.9 suits a deliberate scan of the image.
/// The ramp scales it down near the press and the release.
const PAN_RATE: f32 = 0.9;

/// The free-zoom cruise rate, in natural-log e-folds per second.
/// A full-speed second covers about 5x.
/// The ramp scales the rate down near the press and the release.
const ZOOM_RATE: f32 = 1.65;

/// The vertical travel that a held Shift+F needs to zoom by one natural-log
/// e-fold, in window long sides.
/// Up (a negative y delta) zooms in, and down zooms out.
/// 1.5 suits fine aiming: a screen height of travel zooms one e-fold (2.7x),
/// so a small drag is a small change.
const ZOOM_DRAG_SPAN: f32 = 1.5;

/// A multiplier on the Shift+F zoom rate alone.
/// 8.0 makes a hold eight times as fast as the raw travel suggests.
const ZOOM_DRAG_SPEED: f32 = 8.0;

/// The vertical travel that a held `r` needs for one full turn, in window long
/// sides.
/// Down turns clockwise and up turns counter-clockwise.
/// 1.0 gives a full turn over one screen of travel, so a small drag is a small
/// turn.
const ROTATE_DRAG_SPAN: f32 = 1.0;

/// A multiplier on the `r` rotation rate alone.
/// 1.0 maps the raw travel straight to the angle.
const ROTATE_DRAG_SPEED: f32 = 1.0;

/// The exponential ease time constant for the on-screen pan and scale, in
/// seconds.
/// About 95% of the remaining difference closes after three of them.
const EASE_TIME: f32 = 0.05;

/// The exponential ease time constant of a spread step, in seconds.
///
/// A Shift+J/K step moves the view by a whole spread, so it glides at its own
/// pace instead of jumping: five time constants land the spread after a
/// twentieth of a second, and the motion reads as a scroll.
const SPREAD_EASE_TIME: f32 = 0.05 * 1.0;

/// How long a spread step waits for its prefetched patch before the view
/// paints the region itself.
///
/// A worker that panics sends no message, so the wait cannot be unbounded, and
/// three seconds sits far past any paint the viewer has shown.
const PREFETCH_GIVE_UP: Duration = Duration::from_secs(3);

/// Target pan that centers the fill view on the image point under `mouse`.
///
/// The code clamps both axes to `[-|offset|, |offset|]`.
/// Outside that interval a window edge exposes background.
/// A cursor near the image edge therefore pans only as far as coverage
/// allows.
fn aim_fill_pan(st: &ViewState, mouse: Vec2, win_w: f32, win_h: f32) -> Vec2 {
    let target = (win_w / st.img.w).max(win_h / st.img.h);
    let cur = st.view_scale.unwrap_or(target);
    // Mouse position in image coordinates at the current scale.
    // The offset is the window top-left of the unpanned image.
    let img_x = (mouse.x - (win_w - st.img.w * cur) / 2.0 - st.pan.x) / cur;
    let img_y = (mouse.y - (win_h - st.img.h * cur) / 2.0 - st.pan.y) / cur;
    let off_x = (win_w - st.img.w * target) / 2.0;
    let off_y = (win_h - st.img.h * target) / 2.0;
    Vec2 {
        x: (win_w / 2.0 - off_x - img_x * target).clamp(-off_x.abs(), off_x.abs()),
        y: (win_h / 2.0 - off_y - img_y * target).clamp(-off_y.abs(), off_y.abs()),
    }
}

/// How the code scales the image to the window.
/// The code recomputes the scale every frame, so resizing always stays
/// correct.
#[derive(Clone, Copy, PartialEq)]
enum ZoomMode {
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
    Free(f32),
}

impl ZoomMode {
    /// The free scale, or the fit scale when the mode is not free.
    #[must_use]
    fn free_or(self, fit: f32) -> f32 {
        match self {
            Self::Free(scale) => scale,
            _ => fit,
        }
    }
}

/// The viewer screen: thumbnail grid or single image.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Grid,
    Image,
}

/// The image-view state: what the code shows and how it frames it.
// The view holds several independent flags by nature: a load in flight, a
// paint in flight, the filter choice and a source still opening. Each one
// drives a different frame decision, so one enum would only blur them.
#[allow(clippy::struct_excessive_bools)]
struct ViewState {
    mode: Mode,
    /// The grid entry open in image mode, by stable id (None on a single-file
    /// launch).
    open_id: Option<u64>,
    /// The id of the grid entry whose full-res texture the view holds now.
    /// The texture goes back when the code leaves image mode or switches
    /// entries.
    view_from_grid: Option<u64>,
    /// The texture shown in image mode now.
    view_tex: Option<Texture>,
    view_loading: bool,
    /// The full-resolution plane extent, in surface units (0 x 0 until
    /// known).
    /// One `Size` names both axes, so no site can set or read them apart.
    img: Size,
    zoom: ZoomMode,
    /// The on-screen pan offset, eased toward `target_pan` each frame.
    pan: Vec2,
    target_pan: Vec2,
    /// The on-screen scale, eased toward the target scale each frame.
    view_scale: Option<f32>,
    /// The held-key ramps that drive continuous motion.
    /// Each held key owns one ramp, so the speed eases in, holds, and eases
    /// out instead of stepping on the auto-repeat rate.
    ramps: Ramps,
    /// The window-space point the running zoom eases around.
    /// (+/- or wheel).
    /// None falls back to the window center, for example when an ease still
    /// runs from an earlier step.
    zoom_anchor: Option<Vec2>,
    /// The image rotation, in degrees clockwise.
    /// A held `r` drags it, and the code keeps it across a reload of the same
    /// image.
    /// The release snaps it with `rotate::snap_deg`, so a near quarter turn
    /// lands exactly.
    /// The drag itself never snaps.
    angle: f32,
    /// The window point that a Shift+R hold turns the image around.
    /// None turns around the image center, which is what plain `r` does.
    /// The hold moves the pan through `rotate::pivot`, so the point stays on
    /// its window spot at every angle.
    pivot: Option<Vec2>,
    /// The region the user selected with `v`, in source units (the units of
    /// [`region`]).
    /// A pan, a zoom and a turn keep it on the picture, because the draw maps
    /// its corners through the same transform the picture uses.
    /// A fresh open, a step to another image, a return to the grid and an
    /// empty `v` click all drop it.
    selection: Option<region::Crop>,
    /// `a` toggles nearest-neighbor filtering (pixelated) for pixel peeping.
    /// The default is smooth (bilinear).
    pixelated: bool,
    /// The streaming loader for the open image.
    /// A drop cancels the worker.
    loader: Option<Loader>,
    /// The `VV_BLUR_BG` background (None when the gimmick is off).
    /// The code declares it after `rl` (via `ViewState`) so it drops and
    /// unloads before the window.
    blur_bg: Option<BlurBg>,
    /// The source of the open image, so the view can repaint the region its
    /// window shows.
    /// None for a raster format, which has no more detail to give.
    source: Option<Arc<dyn Surface>>,
    /// The plane of the open source that the view shows.
    /// It is 0 for every source the viewer opens: a book stacks its spreads
    /// into one plane (see [`versatile_viewer::book`]).
    source_plane: usize,
    /// The part of the open plane that a fresh open frames, in surface units
    /// (see [`Surface::open_box`]).
    /// A book answers its first spread, so the cover page shows whole; every
    /// other source answers `None` and the fit frames the whole plane.
    open_box: Option<Size>,
    /// Whether the align still holds the opening box at the top of the
    /// window.
    ///
    /// The fit centers the plane, so the align scrolls to the box and keeps it
    /// there until the user moves the view.
    /// A window manager often resizes the window right after it maps, and a
    /// resize re-fits the scale, so the align repeats every frame instead of
    /// running once.
    open_align: OpenAlign,
    /// The pan a Shift+J/K spread step glides to.
    /// A step eases at [`SPREAD_EASE_TIME`], which now matches the ordinary
    /// pan ease, so the glide reads as a scroll of one spread.
    /// The field clears itself when the pan lands or the target moves away
    /// from it, so any other motion uses the ordinary ease again.
    spread_glide: Option<Vec2>,
    /// The prefetch of a spread step: the patch a worker paints, or the patch
    /// that landed and draws in place until the glide does.
    ///
    /// The step knows where it lands, so the patch of the target spread is on
    /// its way before the glide ends, and the view then holds a sharp spread
    /// the moment the pan arrives instead of a stretched one.
    /// A patch replaces [`ViewState::crop`] once it covers the window.
    prefetch: Prefetch,
    /// The newest prefetch token, so a result of an earlier aim drops.
    /// The counter stays beside the state, because it only ever grows.
    prefetch_seq: u64,
    /// The region patch on screen: the visible part of the document, painted
    /// at the scale the window shows it.
    /// None until the first paint lands, and then the whole-image texture in
    /// `view_tex` takes over only where the patch cannot reach.
    crop: Option<CropState>,
    /// Whether a worker paints a patch now.
    /// One paint in flight, because resvg cannot cancel one mid-paint.
    crop_pending: bool,
    /// Bumped on every open, every n/p step and every return to the grid, so
    /// the result of a worker for an earlier image drops instead of landing.
    crop_gen: u64,
    /// The region and scale the window needed when the debounce last saw it
    /// move.
    crop_seen: Option<region::Patch>,
    /// When the view last moved.
    crop_quiet_at: Instant,
    /// The region workers talk back over this pair.
    crop_tx: Sender<CropMsg>,
    crop_rx: Receiver<CropMsg>,
}

/// The visible part of the open image, painted at the scale the window shows
/// it.
///
/// The patch carries its own texture, so the whole-image texture stays as the
/// fallback of a pan or a resize that outruns the repaint.
struct CropState {
    /// The pixels of the patch, already uploaded by the worker that painted
    /// them.
    texture: Texture,
    /// The source rect the patch covers and the scale it holds, so a paint
    /// rule reads one value instead of a rect beside a loose scale.
    patch: region::Patch,
}

/// Whether the image view still holds the opening box at the top of its
/// window.
///
/// One value carries the flag and the pan it last wrote, so a pan that the
/// user moved can never be confused with the one the align owns.
#[derive(Debug, Clone, Copy, PartialEq)]
enum OpenAlign {
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
enum Prefetch {
    /// Nothing prefetched.
    Idle,
    /// A worker paints `patch` under token `id`, from `at` on.
    Painting {
        /// The rect and the scale the worker paints.
        patch: region::Patch,
        /// The token the result must carry, so an earlier aim drops.
        id: u64,
        /// When the paint started, so [`PREFETCH_GIVE_UP`] bounds the wait.
        at: Instant,
    },
    /// The patch of a step, drawn in place until the glide lands.
    Ready(CropState),
}

impl Prefetch {
    /// The patch this state holds on screen, if any.
    fn held(&self) -> Option<&CropState> {
        match self {
            Self::Ready(state) => Some(state),
            Self::Idle | Self::Painting { .. } => None,
        }
    }

    /// The aim a worker paints now, with the token and the start time.
    fn painting(&self) -> Option<(region::Patch, u64, Instant)> {
        match self {
            Self::Painting { patch, id, at } => Some((*patch, *id, *at)),
            Self::Idle | Self::Ready(_) => None,
        }
    }
}

/// A message from a region worker to the image view.
///
/// The view carries one generation counter, so a paint of the previous image
/// never lands on the new one.
enum CropMsg {
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
}

impl ViewState {
    /// Reset the fit and pan state for a freshly shown image.
    fn reset_view(&mut self) {
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
    fn frame(&self, win: Vec2) -> Option<region::Frame> {
        Some(region::Frame {
            img: Vec2::new(self.img.w, self.img.h),
            win,
            pan: self.pan,
            scale: self.view_scale?,
            angle: self.angle,
        })
        .filter(region::Frame::drawable)
    }

    /// The window mapping of the open plane at `pan` and `scale`.
    ///
    /// [`Self::frame`] answers the same mapping for the pan and the scale on
    /// screen; a caller that aims elsewhere (the target pan of a glide) asks
    /// here, so it never assembles the five numbers itself.
    fn frame_at(&self, win: Vec2, pan: Vec2, scale: f32) -> region::Frame {
        region::Frame {
            img: Vec2::new(self.img.w, self.img.h),
            win,
            pan,
            scale,
            angle: self.angle,
        }
    }

    /// Frame the opening box again once the image dimensions land.
    ///
    /// A fresh open measures a box instead of the whole plane, and a scroll to
    /// the top shows it first.
    fn refit_open(&mut self) {
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
    fn align_open_box_again(&mut self) {
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
    fn align_open_box(&mut self, win_h: f32) {
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
        let top = Vec2::new(0.0, Column::new(self.img, scale, win_h).top_pan());
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
    fn step_spread(&mut self, step: isize, win_h: f32) -> bool {
        let (Some(scale), Some(source)) = (self.view_scale, self.source.as_ref()) else {
            return false;
        };
        if !(scale.is_finite() && scale > 0.0) {
            return false;
        }
        let count = source.spread_count();
        let Some(first) = source.spread_box(0) else {
            return false;
        };
        if count < 2 || !(first.h.is_finite() && first.h > 0.0) {
            return false;
        }
        let column = Column::new(self.img, scale, win_h);
        let row = column.row(self.target_pan.y, first.h).min(count - 1);
        let next = row as isize + step;
        if next < 0 || next as usize >= count {
            return false;
        }
        let Some(target) = source.spread_box(next as usize) else {
            return false;
        };
        self.target_pan = Vec2::new(0.0, column.pan_of(target.y));
        self.spread_glide = Some(self.target_pan);
        true
    }

    /// Paint the region a spread step lands on, so the sharp patch is ready
    /// when the glide arrives.
    ///
    /// The step knows its target pan at the press edge, so the paint needs no
    /// guess.
    /// The work runs on a decode worker (`rayon`), so the glide never waits on
    /// it, and the result lands in [`ViewState::prefetch`] instead of replacing
    /// the patch on screen.
    /// A step that misses a spread calls nothing, because it moves nothing.
    fn prefetch_spread(&mut self, win: Vec2, uploader: &Uploader) {
        if self.source.is_none() || self.img.w <= 0.0 || self.view_loading {
            return;
        }
        let Some(scale) = self.view_scale else {
            return;
        };
        if !(scale.is_finite() && scale > 0.0) {
            return;
        }
        let Some(source) = self.source.clone() else {
            return;
        };
        // The same rect an ordinary paint would cover, so the landing frame
        // adopts it without a second paint.
        let aim = self.frame_at(win, self.target_pan, scale);
        let patch = region::Patch::new(aim.crop(region::MARGIN), scale);
        if patch.rect.is_empty() {
            return;
        }
        let id = self.next_prefetch_id();
        self.prefetch = Prefetch::Painting {
            patch,
            id,
            at: Instant::now(),
        };
        let (generation, tx) = (self.crop_gen, self.crop_tx.clone());
        let plane = self.source_plane;
        let uploader = uploader.clone();
        rayon::spawn(move || {
            match source.render(View::region(plane, patch.rect, scale)) {
                Ok(image) => {
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
                    eprintln!("vv: {err}");
                    let _ = tx.send(CropMsg::PrefetchFailed { generation, id });
                }
            }
        });
    }

    /// Drop the region state of the image that was open, and retire every
    /// worker result still in flight.
    fn close_crop(&mut self) {
        self.source = None;
        self.crop = None;
        self.crop_pending = false;
        self.crop_seen = None;
        self.crop_quiet_at = Instant::now();
        self.crop_gen = self.crop_gen.wrapping_add(1);
        self.close_prefetch();
    }

    /// Drop the prefetch of a spread step: the patch on screen, the aim a
    /// worker paints and every result still in flight.
    fn close_prefetch(&mut self) {
        self.prefetch = Prefetch::Idle;
        self.prefetch_seq = self.prefetch_seq.wrapping_add(1);
    }

    /// The token of a fresh prefetch, so a result of an earlier aim drops.
    fn next_prefetch_id(&mut self) -> u64 {
        self.prefetch_seq = self.prefetch_seq.wrapping_add(1);
        self.prefetch_seq
    }

    /// Adopt a source that the launch already opened, so the view can repaint
    /// the visible region at any zoom.
    fn adopt_source(&mut self, source: Option<Arc<dyn Surface>>) {
        self.close_crop();
        self.source = source;
    }

    /// Ask a worker for the source of an entry, so the view can repaint the
    /// region its window shows.
    ///
    /// The grid keeps no source, so an image opened from the grid opens its
    /// own here, off the main thread.
    /// A raster file answers None at once.
    /// A document arrives as a book, so the patch covers whole spreads.
    fn request_source(&mut self, entry: &EntrySource) {
        self.close_crop();
        self.source_plane = 0;
        let path = entry.handle();
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
    /// A spread step paints its landing region before the glide ends, so a
    /// patch that already covers the window replaces the one on screen here and
    /// skips the debounce and the paint.
    fn tick_crop(&mut self, now: Instant, win: Vec2, uploader: &Uploader) {
        if self.source.is_none() || self.crop_pending || self.img.w <= 0.0 || self.view_loading {
            return;
        }
        let Some(scale) = self.view_scale else {
            return;
        };
        if !(scale.is_finite() && scale > 0.0) {
            return;
        }
        let frame = self.frame_at(win, self.pan, scale);
        let needed = region::Patch::new(frame.crop(0.0), scale);
        // A prefetched patch that holds the window is the patch on screen now.
        // The check runs before the debounce, so the landing frame shows the
        // sharp spread at once instead of waiting out [`region::QUIET`].
        if self
            .prefetch
            .held()
            .is_some_and(|state| region::covers(&needed, &state.patch))
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
        if let Some((aim, _, at)) = self.prefetch.painting()
            && region::covers(&needed, &aim)
        {
            if now.duration_since(at) < PREFETCH_GIVE_UP {
                return;
            }
            self.prefetch = Prefetch::Idle;
        }
        if !region::worth_painting(&needed, self.crop.as_ref().map(|crop| &crop.patch)) {
            return;
        }
        let Some(source) = self.source.clone() else {
            return;
        };
        // Paint a margin around what the window shows, so a small pan between
        // two paints stays covered, and a filter keeps its outer context.
        let paint = region::Patch::new(frame.crop(region::MARGIN), scale);
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
        rayon::spawn(
            move || match source.render(View::region(plane, paint.rect, scale)) {
                Ok(image) => {
                    // The paint is a window patch, so it uploads on this worker
                    // and the main thread only draws it.
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
                    eprintln!("vv: {err}");
                    let _ = tx.send(CropMsg::Failed { generation });
                }
            },
        );
    }

    /// Apply finished region work.
    /// A source lands as is, and a patch replaces the one on screen.
    /// The paint worker uploaded its own texture, so this call only moves
    /// handles.
    /// A result of an earlier image drops.
    fn drain_crops(&mut self) {
        while let Ok(msg) = self.crop_rx.try_recv() {
            match msg {
                CropMsg::Source { generation, source } => {
                    if generation == self.crop_gen {
                        self.source = source;
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
                        Prefetch::Painting { id: want, .. } if want == id
                    );
                    if generation != self.crop_gen || !aimed {
                        continue;
                    }
                    self.prefetch = Prefetch::Ready(CropState { texture, patch });
                }
                CropMsg::PrefetchFailed { generation, id } => {
                    let aimed = matches!(
                        self.prefetch,
                        Prefetch::Painting { id: want, .. } if want == id
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
struct Ramps {
    pan_left: Ramp,
    pan_right: Ramp,
    pan_up: Ramp,
    pan_down: Ramp,
    zoom_in: Ramp,
    zoom_out: Ramp,
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
    fn tick(&mut self, down: NavKeys, base: f32, dt: f32, speed: f32) -> Motion {
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
struct Motion {
    /// The pan delta for this frame, in window pixels.
    pan: Vec2,
    /// The free-zoom factor for this frame, or `None` when no zoom key ramps.
    zoom: Option<f32>,
}

/// The held keys that drive the continuous image-view motions.
/// The six directions are independent, so the struct stays a plain set of
/// flags.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, Default)]
struct NavKeys {
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    zoom_in: bool,
    zoom_out: bool,
}

/// The mouse side-button nav state for this frame.
/// The X11 thumb buttons arrive as raylib `MOUSE_BUTTON_SIDE` (X button 8, back)
/// and `MOUSE_BUTTON_EXTRA` (X button 9, forward).
/// The code accepts raylib's own BACK and FORWARD codes too, so a backend that
/// maps them directly behaves the same.
#[derive(Debug, Clone, Copy, Default)]
struct NavMouse {
    /// The next-image button state.
    next: NavButton,
    /// The previous-image button state.
    prev: NavButton,
}

/// One side button, as an edge and a held state.
#[derive(Debug, Clone, Copy, Default)]
struct NavButton {
    /// The button went down this frame.
    edge: bool,
    /// The button is held now.
    held: bool,
}

/// Read the side-button nav state once per frame.
///
/// A held side button repeats through the auto-repeat path, exactly like `n` and
/// `p`.
/// X11 reports buttons 8 and 9 as `Back` and `Forward`, which is what the raylib
/// backend mapped its side and extra codes onto.
fn read_nav_mouse(input: &Input) -> NavMouse {
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
fn draw_turned(
    r: &mut Renderer,
    texture: &Texture,
    img: Size,
    scale: f32,
    center: Vec2,
    angle: f32,
    filter: Filter,
) {
    let src = Rect::sized(texture.width() as f32, texture.height() as f32);
    let (dw, dh) = (img.w * scale, img.h * scale);
    let dest = Rect::new(center.x, center.y, dw, dh);
    r.texture_pro(
        texture,
        src,
        dest,
        Turn::new(dest.center_origin(), rotate::normalize_deg(angle)),
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
struct Column {
    /// The plane height, in surface units.
    img_h: f32,
    /// Output pixels per surface unit.
    scale: f32,
    /// The window height, in window pixels.
    win_h: f32,
}

impl Column {
    /// The vertical mapping of `img` at `scale` inside a window `win_h` pixels
    /// tall.
    const fn new(img: Size, scale: f32, win_h: f32) -> Self {
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
    fn top_pan(self) -> f32 {
        (self.img_h * self.scale - self.win_h).max(0.0) / 2.0
    }

    /// The pan that puts the top of a spread on the top edge of the window.
    ///
    /// `origin_y` is the top of the spread in plane units, so spread 0 answers
    /// [`Self::top_pan`]: a step to the first spread frames it like a fresh
    /// open does.
    fn pan_of(self, origin_y: f32) -> f32 {
        self.top_pan() - origin_y * self.scale
    }

    /// The spread a pan shows at the top of the window, rounded to the
    /// nearest.
    ///
    /// The code reads the row from the pan the view aims at, so a step pressed
    /// while an earlier step still glides counts from the spread that step
    /// aims at, never from the pixels on the way.
    fn row(self, pan_y: f32, spread_h: f32) -> usize {
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
const fn view_filter(pixelated: bool) -> Filter {
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
fn patch_dest(center: Vec2, dw: f32, dh: f32, tw: f32, th: f32, angle: f32) -> Option<Rect> {
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
    Some(Rect::new(
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
fn draw_patch(r: &mut Renderer, patch: &CropState, frame: &region::Frame, filter: Filter) {
    let rect = patch.patch.rect;
    let (scale, angle) = (frame.scale, frame.angle);
    let middle = Vec2::new(rect.x + rect.w / 2.0, rect.y + rect.h / 2.0);
    let center = frame.to_screen(middle);
    let tw = patch.texture.width() as f32;
    let th = patch.texture.height() as f32;
    match patch_dest(center, rect.w * scale, rect.h * scale, tw, th, angle) {
        Some(dest) => r.texture_pro(
            &patch.texture,
            Rect::sized(tw, th),
            dest,
            Turn::NONE,
            Color::WHITE,
            filter,
        ),
        None => draw_turned(
            r,
            &patch.texture,
            Size {
                w: rect.w,
                h: rect.h,
            },
            scale,
            center,
            angle,
            filter,
        ),
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
fn selection_rect(
    frame: Option<region::Frame>,
    kept: Option<region::Crop>,
    drag: Option<region_sel::Drag>,
    mouse: Vec2,
) -> Option<region::Crop> {
    let frame = frame.filter(region::Frame::drawable)?;
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
/// matches the grid's dim tint (see [`versatile_viewer::DIM_LEVEL`]).
/// The bands and the border map through [`region::Frame::to_screen`], so they
/// follow the pan, the zoom and the turn of the draw.
fn draw_selection(r: &mut Renderer, frame: region::Frame, selection: region::Crop) {
    let outer = region::Crop {
        x: 0.0,
        y: 0.0,
        w: frame.img.x,
        h: frame.img.y,
    };
    for band in region_sel::around(outer, selection) {
        let p = band.map(|point| frame.to_screen(point));
        // The pipeline culls nothing, so a band draws in either winding.
        r.fill(Quad::new(p), DIM_VEIL);
    }
    let corners = region_sel::corners(selection).map(|point| frame.to_screen(point));
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
fn same_patch(a: &region::Patch, b: &region::Patch) -> bool {
    (a.rect.x - b.rect.x).abs() <= 0.5
        && (a.rect.y - b.rect.y).abs() <= 0.5
        && (a.rect.w - b.rect.w).abs() <= 0.5
        && (a.rect.h - b.rect.h).abs() <= 0.5
        && (a.scale - b.scale).abs() <= b.scale * 0.002
}

/// Whether the region patch covers the whole window at the current transform.
///
/// The draw skips the whole-image texture then, because the patch already
/// holds every pixel the window shows.
/// A patch whose margin a pan or a resize consumed leaves the rest to the
/// whole-image texture until the repaint lands.
fn patch_covers(st: &ViewState, win_w: f32, win_h: f32) -> bool {
    let Some(crop) = &st.crop else {
        return false;
    };
    let Some(scale) = st.view_scale else {
        return false;
    };
    if st.img.w <= 0.0 || !(scale.is_finite() && scale > 0.0) {
        return false;
    }
    let Some(frame) = st.frame(Vec2::new(win_w, win_h)) else {
        return false;
    };
    frame.crop(0.0).contains(&crop.patch.rect)
}

/// Keep the pivot of a turned image on its window spot.
///
/// The angle already ran from `from` to the value `st.angle` shows, so the code
/// shifts the pan and its target by the distance that the plain centered turn
/// would move the pivot.
/// The pivot stays a draw transform: no pixel moves.
fn keep_pivot(st: &mut ViewState, from: f32, win_w: f32, win_h: f32) {
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
fn attach_blur_bg(
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
fn put_back_view(
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

/// Make the entry at index `idx` the open image.
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
fn show_entry(
    st: &mut ViewState,
    grid: &mut Grid,
    idx: usize,
    ui: &Ui,
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
            st.view_scale = Some((win_w / st.img.w).min(win_h / st.img.h));
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
                st.view_scale = Some((win_w / st.img.w).min(win_h / st.img.h));
            }
        } else {
            // The dimensions arrive with the header or the texture.
            st.img = Size { w: 0.0, h: 0.0 };
        }
        st.loader = if queued || (thumb && !no_full) {
            // A grid decode for this entry is in flight, or the entry has
            // a thumb and the keep-set refill dispatched (or will dispatch)
            // the missing full-res decode.
            // Wait for the decode-landed takeover instead of decoding the
            // file twice in a streaming worker.
            // A document keeps no full-res texture, so its cell waits for
            // nothing: the loader renders the book now.
            None
        } else {
            let preview_px = PreviewPx::new(
                ui.get_screen_width()
                    .max(ui.get_screen_height())
                    .try_into()
                    .unwrap_or(512),
            );
            // The image view opens a document as a book, so plane 0 holds
            // every spread.
            // The opening box comes from the surface before the loader takes
            // it, so the fit frames the first spread.
            match source.open_book() {
                Ok((surface, plane)) => {
                    st.open_box = surface.open_box(plane);
                    Some(Loader::start_plane(
                        surface,
                        plane,
                        preview_px,
                        &Some(r.uploader()),
                    ))
                }
                Err(err) => {
                    eprintln!("vv: {}: {err}", source.label());
                    grid.mark_failed(id, DecodeFailure::Decode);
                    None
                }
            }
        };
        attach_blur_bg(&mut st.blur_bg, &r.uploader(), blur.as_ref(), st.open_id);
    }
}

/// Display a path with `$HOME` collapsed to `~`.
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
/// One frame's window, input and clock, as the view code reads them.
///
/// The type bundles the platform facts that the raylib handle used to carry, so
/// the frame body reads one name: the pointer, the held keys, the frame time,
/// the window size and the three window calls (cursor, grab, title).
///
/// The drawing does not live here: the frame body holds the [`Renderer`] apart,
/// because the batch needs a mutable borrow for the whole frame while the input
/// is read at the same time.
struct Ui<'a> {
    input: &'a Input,
    window: &'a Window,
    gpu: &'a Gpu,
    size: (u32, u32),
    /// The seconds since the viewer started.
    time: f64,
    /// The seconds the previous frame took.
    dt: f32,
}

impl Ui<'_> {
    /// The window width, in physical pixels.
    fn get_screen_width(&self) -> i32 {
        i32::try_from(self.size.0).unwrap_or(i32::MAX)
    }

    /// The window height, in physical pixels.
    fn get_screen_height(&self) -> i32 {
        i32::try_from(self.size.1).unwrap_or(i32::MAX)
    }

    /// The seconds since the viewer started.
    fn get_time(&self) -> f64 {
        self.time
    }

    /// The seconds the previous frame took.
    fn get_frame_time(&self) -> f32 {
        self.dt
    }

    /// The pointer position, in window pixels.
    ///
    /// A captured drag moves this value by the raw device travel, so the view
    /// keeps reading one pointer while the real cursor stays grabbed.
    fn get_mouse_position(&self) -> Vec2 {
        self.input.cursor_position()
    }

    /// The raw device travel since the previous frame.
    fn get_mouse_delta(&self) -> Vec2 {
        self.input.delta()
    }

    /// The wheel travel since the previous frame, in notches.
    fn get_mouse_wheel_move(&self) -> f32 {
        self.input.wheel_move()
    }

    /// Is the key held now?
    fn is_key_down(&self, key: KeyCode) -> bool {
        self.input.is_key_down(key)
    }

    /// Did the key go down this frame, without a repeat?
    fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.input.is_key_pressed(key)
    }

    /// The keys that went down this frame, without a repeat.
    fn pressed_keys(&self) -> impl Iterator<Item = KeyCode> + '_ {
        self.input.pressed_keys()
    }

    /// The input state of this frame.
    fn input(&self) -> &Input {
        self.input
    }

    /// The typed characters of this frame, each with its physical key.
    fn typed(&self) -> &[versatile_viewer::input::Typed] {
        self.input.typed()
    }

    /// Did the mouse button go down this frame?
    fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.input.is_button_pressed(button)
    }

    /// Did the mouse button come up this frame?
    fn is_mouse_button_released(&self, button: MouseButton) -> bool {
        self.input.is_button_released(button)
    }

    /// Set the window title.
    fn set_window_title(&self, title: &str) {
        self.window.set_title(title);
    }

    /// Grab the pointer and hide it, so a drag has unbounded travel.
    ///
    /// `Locked` is the mode with no pointer motion at all, and X11 refuses it,
    /// so `Confined` is the fallback. The raw device deltas arrive either way.
    fn disable_cursor(&self) {
        for mode in [CursorGrabMode::Locked, CursorGrabMode::Confined] {
            if self.window.set_cursor_grab(mode).is_ok() {
                break;
            }
        }
        self.window.set_cursor_visible(false);
    }

    /// Give the pointer back and show it again.
    fn enable_cursor(&self) {
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
    }

    /// Put the pointer at a window point.
    fn set_mouse_position(&self, at: Vec2) {
        let position = winit::dpi::PhysicalPosition::new(at.x.round(), at.y.round());
        let _ = self.window.set_cursor_position(position);
    }

    /// Start a frame: set the viewport and empty the batch.
    fn begin(&self, r: &mut Renderer) {
        r.begin(self.gpu.queue(), self.size);
    }

    /// Clear the window to `color`.
    fn clear_background(r: &mut Renderer, color: Color) {
        r.set_clear(color);
    }

    /// Draw the batch and present it.
    fn end(&self, r: &mut Renderer) -> Result<(), FrameError> {
        r.flush(self.gpu)
    }
}

/// What the launch decided before a window existed.
///
/// The CLI, the directory listing and, for a single file, the decode all run
/// here: the window title carries the image count, and a picture sizes its own
/// window. The GPU work waits for [`App::resumed`], because a surface needs a
/// window.
struct Launch {
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
    fn open(path: &Path) -> Result<Self> {
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
    /// The swapchain arrived and the launch frame uploaded.
    started: bool,
    /// The window size of this frame, in physical pixels.
    size: (u32, u32),
    /// When the previous frame started, for the frame time.
    last_frame: Instant,
    /// When the first frame started, for `Ui::get_time`.
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
    /// True while an image-view drag holds the pointer.
    pointer_captured: bool,
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
            pointer_captured: false,
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
        if !self.single_plane && self.grid.as_ref().is_some_and(|g| g.entries.len() == 1) {
            let ui = Ui {
                input: &self.input,
                window,
                gpu,
                size: self.size,
                time: 0.0,
                dt: 0.0,
            };
            let win = (WinW::new(self.size.0 as f32), WinH::new(self.size.1 as f32));
            show_entry(
                &mut self.st,
                self.grid.as_mut().unwrap(),
                0,
                &ui,
                renderer,
                win,
                true,
            );
        }
    }

    /// Run one frame: read the input, advance every ease, draw and present.
    //
    // One frame is one long state machine by design: the order of the steps
    // (drain, input, motion, draw) carries the invariants.
    #[allow(clippy::too_many_lines)]
    fn frame(&mut self) -> Result<()> {
        let window = self.window.as_ref().expect("started").clone();
        let now = Instant::now();
        let time = now.duration_since(self.started_at).as_secs_f64();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.dt = dt;
        // The frame body borrows the frame's input and GPU immutably while it
        // mutates the views, so the two move out of the app for the length of
        // the call and back in afterwards.
        let gpu = self.gpu.take().expect("started");
        let input = std::mem::take(&mut self.input);
        let ui = Ui {
            input: &input,
            window: &window,
            gpu: &gpu,
            size: self.size,
            time,
            dt,
        };
        let mut r = self.renderer.take().expect("started");
        let result = self.body(&ui, &mut r);
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
        let attributes = Window::default_attributes()
            .with_title(self.launch_title.clone())
            // WM_CLASS on X11, which replaces `src/wmclass.rs`: the class and
            // the instance name are both `vv`.
            .with_name("vv", "vv")
            .with_inner_size(winit::dpi::PhysicalSize::new(self.win0.0, self.win0.1))
            .with_resizable(true);
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.fatal = Some(anyhow::anyhow!("cannot create the window: {err}"));
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
        self.started = true;
        self.started_at = Instant::now();
        self.last_frame = Instant::now();
        self.start_views(&window);
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => match self.frame() {
                Ok(()) => {
                    if self.quit {
                        event_loop.exit();
                    } else if let Some(window) = &self.window {
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
                // A captured drag reads the raw device travel, so the real
                // pointer does not steer it.
                if !self.pointer_captured {
                    self.input
                        .cursor(Vec2::new(position.x as f32, position.y as f32));
                }
            }
            WindowEvent::MouseInput { state, button, .. } => self.input.button(button, state),
            WindowEvent::Focused(false) => {
                // A lost focus never delivers the releases, so a captured drag
                // must not outlive it.
                self.input.forget_held();
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
    fn body(&mut self, ui: &Ui, r: &mut Renderer) -> Result<()> {
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
            let ft = ui.get_frame_time();
            if ft > 0.1 {
                eprintln!("vv: slow frame {ft:.0} ms");
            }
        }
        let win_w = ui.get_screen_width() as f32;
        let win_h = ui.get_screen_height() as f32;
        // Post-drag restore: verify the warp landed where the code sent it.
        // If a competing re-warp (GLFW, WM, XWayland) moved it, snap again.
        // The code gives up after a few tries so a user moving the mouse is
        // never pinned.
        if let Some(target) = self.pending_restore {
            let cur = ui.get_mouse_position();
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
                ui.set_mouse_position(target);
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
        if self.st.open_id != self.title_open_id {
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
            ui.set_window_title(&title);
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
            let mut open_failed = false;
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
                    None => open_failed = true, // entry vanished (decode failed)
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
                        open_failed = true;
                    }
                }
            }
            if open_failed {
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
        let shift = ui.is_key_down(KeyCode::ShiftLeft) || ui.is_key_down(KeyCode::ShiftRight);
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
            // is_key_pressed misses a press+release pair that lands inside
            // one frame, and this is easy here while frames stall on texture
            // uploads or decode.
            // Leftover queue entries otherwise leak into the grid mode and act
            // there (for example a missed Enter that reopens the
            // just-viewed image at once).
            // State queries (is_key_down panning, is_key_pressed W/E/=/-) are
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
                self.grab_pos = ui.get_mouse_position();
                self.drag_virtual = self.grab_pos;
                ui.disable_cursor();
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
                self.grab_pos = ui.get_mouse_position();
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
            // `!is_key_down` also covers a press+release landing inside one
            // frame (the pressed queue records the press, but there is no
            // release edge to observe).
            // Without this the capture sticks.
            // The `r` release keeps the angle, so the image stays turned.
            if (self.f_capture || self.r_capture)
                && !ui.is_key_down(KeyCode::KeyF)
                && !ui.is_key_down(KeyCode::KeyR)
            {
                // Fold in this frame's motion before the lock is released
                // (the drag block below no longer runs once capture ends).
                let delta = ui.get_mouse_delta();
                self.drag_virtual.x += delta.x;
                self.drag_virtual.y += delta.y;
                let f_dragged = (self.drag_virtual - self.grab_pos).length() > TAP_SLACK;
                if self.pointer_captured {
                    ui.enable_cursor();
                    ui.set_mouse_position(self.grab_pos);
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
                    eprintln!(
                        "vv: hold release: grab ({:.0},{:.0}), virtual ({:.0},{:.0}), dragged {}, \
                         angle {}",
                        self.grab_pos.x,
                        self.grab_pos.y,
                        self.drag_virtual.x,
                        self.drag_virtual.y,
                        f_dragged,
                        self.st.angle
                    );
                }
            }
            // Auto-repeat: the initial press fires at once (edge from the
            // queue above), and the hold fires at the X server repeat rate
            // after its delay.
            // The mouse side buttons ride the same nav path: forward is next
            // and back is previous, exactly like n and p.
            let now = ui.get_time();
            let (delay, rate) = versatile_viewer::keyrepeat::settings();
            let mouse_nav = read_nav_mouse(ui.input());
            let next_edge = nav_next_edge || mouse_nav.next.edge;
            let prev_edge = nav_prev_edge || mouse_nav.prev.edge;
            // A held Shift moves the Space hold to the previous direction,
            // so a hold repeats the way its press edge stepped.
            let space_held = ui.is_key_down(KeyCode::Space);
            let next_held =
                ui.is_key_down(KeyCode::KeyN) || (space_held && !shift) || mouse_nav.next.held;
            let prev_held = ui.is_key_down(KeyCode::KeyP)
                || ui.is_key_down(KeyCode::Backspace)
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
            self.spread_armed = !(ui.is_key_down(KeyCode::KeyJ)
                || ui.is_key_down(KeyCode::KeyK)
                || ui.is_key_down(KeyCode::PageDown)
                || ui.is_key_down(KeyCode::PageUp));

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
            if self.grid.is_some() && enter {
                return_to_grid = true;
            }
            if return_to_grid {
                // Leaving image view mid-drag: give the pointer back before
                // the the grid click handling needs a visible cursor.
                if self.pointer_captured {
                    ui.enable_cursor();
                    ui.set_mouse_position(self.grab_pos);
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
            if ui.is_key_pressed(KeyCode::KeyW) {
                self.st.zoom = if shift {
                    ZoomMode::FitAll
                } else {
                    ZoomMode::FitDown
                };
                self.st.target_pan = Vec2::ZERO;
            } else if ui.is_key_pressed(KeyCode::KeyE) {
                self.st.zoom = if shift {
                    ZoomMode::FitHeight
                } else {
                    ZoomMode::FitWidth
                };
                self.st.target_pan = Vec2::ZERO;
            } else if ui.is_key_pressed(KeyCode::KeyT) || f_zoom {
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
                        aim_fill_pan(&self.st, ui.get_mouse_position(), win_w, win_h);
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
                let dt = ui.get_frame_time();
                // Ctrl + j/k zoom like `+`/`-` (Ctrl + Shift + j/k keeps
                // the Shift speed).
                // `j` and `k` also pan, so a held Ctrl turns them into zoom
                // keys instead of pan keys.
                let ctrl =
                    ui.is_key_down(KeyCode::ControlLeft) || ui.is_key_down(KeyCode::ControlRight);
                let ctrl_zoom_in = ctrl && ui.is_key_down(KeyCode::KeyJ);
                let ctrl_zoom_out = ctrl && ui.is_key_down(KeyCode::KeyK);
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
                    left: ui.is_key_down(KeyCode::KeyH) || ui.is_key_down(KeyCode::ArrowLeft),
                    right: ui.is_key_down(KeyCode::KeyL) || ui.is_key_down(KeyCode::ArrowRight),
                    up: (ui.is_key_down(KeyCode::KeyK) && !ctrl_zoom_out && !spread_keys)
                        || ui.is_key_down(KeyCode::ArrowUp),
                    down: (ui.is_key_down(KeyCode::KeyJ) && !ctrl_zoom_in && !spread_keys)
                        || ui.is_key_down(KeyCode::ArrowDown),
                    // US layouts put the zoom keys on the `=` and `-`
                    // keycodes (numpad included).
                    // Other layouts put `+`/`-` on another key, whose keycode
                    // the typed character taught earlier.
                    // Polling that keycode reads the real held state, so a
                    // second key going down does not stop the zoom.
                    // Ctrl + j/k join them as a hold, so the ramp eases the
                    // same way.
                    zoom_in: ui.is_key_down(KeyCode::Equal)
                        || ui.is_key_down(KeyCode::NumpadAdd)
                        || ctrl_zoom_in
                        || self
                            .zoom_keys
                            .zoom_in_key()
                            .is_some_and(|k| ui.is_key_down(k)),
                    zoom_out: ui.is_key_down(KeyCode::Minus)
                        || ui.is_key_down(KeyCode::NumpadSubtract)
                        || ctrl_zoom_out
                        || self
                            .zoom_keys
                            .zoom_out_key()
                            .is_some_and(|k| ui.is_key_down(k)),
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
                let delta = ui.get_mouse_delta();
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
                let wheel = ui.get_mouse_wheel_move();
                if wheel != 0.0 {
                    let factor = 1.25f32.powf(wheel);
                    self.st.zoom = ZoomMode::Free((target_scale * factor).clamp(0.01, 100.0));
                    self.st.zoom_anchor = Some(ui.get_mouse_position());
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
                let drag_pressed = ui.is_mouse_button_pressed(MouseButton::Left);
                let drag_released = ui.is_mouse_button_released(MouseButton::Left);
                if drag_pressed && !self.pointer_captured && !self.r_capture {
                    self.grab_pos = ui.get_mouse_position();
                    self.drag_virtual = self.grab_pos;
                    ui.disable_cursor();
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
                        eprintln!(
                            "vv: drag release: grab ({:.0},{:.0}), virtual ({:.0},{:.0}), rl pos \
                             ({:.0},{:.0})",
                            self.grab_pos.x,
                            self.grab_pos.y,
                            self.drag_virtual.x,
                            self.drag_virtual.y,
                            ui.get_mouse_position().x,
                            ui.get_mouse_position().y
                        );
                    }
                    ui.enable_cursor();
                    ui.set_mouse_position(self.grab_pos);
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
                    let delta = ui.get_mouse_delta();
                    // The virtual cursor follows physical travel 1:1 (self.debug
                    // crosshair).
                    self.drag_virtual.x += delta.x;
                    self.drag_virtual.y += delta.y;
                    // The pan gets the 2x speedup.
                    if !self.f_zoom_capture && !self.r_capture {
                        self.st.target_pan.x += delta.x * 2.0;
                        self.st.target_pan.y += delta.y * 2.0;
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
                        ui.set_mouse_position(self.st.zoom_anchor.unwrap());
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
                    let mouse = ui.get_mouse_position();
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
                // `!is_key_down` also covers a press and release that land
                // inside one frame, like the `f` and `r` halves.
                if let Some(drag) = self.region_drag
                    && !ui.is_key_down(KeyCode::KeyV)
                {
                    // The release reads the same rect the draw shows, so the
                    // committed region is the one on screen, and a tap commits
                    // nothing: it drops a span and keeps a moved rect.
                    self.st.selection = selection_rect(
                        self.st.frame(Vec2::new(win_w, win_h)),
                        None,
                        Some(drag),
                        ui.get_mouse_position(),
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
                ui.get_mouse_position(),
            )
        } else {
            None
        };

        ui.begin(r);
        Ui::clear_background(r, BACKGROUND);

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
        if let Err(err) = ui.end(r) {
            if err.needs_reconfigure() {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.reconfigure();
                }
            } else if !matches!(err, FrameError::Occluded | FrameError::Timeout) {
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

fn main() -> Result<()> {
    // `--mcp` may come before or after the path, so the path is the first
    // argument that is not the flag.
    let mut mcp = false;
    let mut args = env::args().skip(1);
    let arg = loop {
        match args.next() {
            Some(flag) if flag == "--mcp" => mcp = true,
            Some(path) => break path,
            None => bail!("usage: vv [--mcp] <image-path-or-directory>"),
        }
    };
    if mcp {
        start_mcp()?;
    }
    let path = Path::new(&arg);
    let launch = Launch::open(path)?;
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

    #[test]
    fn a_patch_that_matches_the_view_lands_on_whole_pixels() {
        // A patch of 1532x1180 px whose screen center falls between pixels.
        let (cx, cy) = (512.32f32, 384.68f32);
        let dest = patch_dest(Vec2::new(cx, cy), 1532.0, 1180.0, 1532.0, 1180.0, 0.0).unwrap();
        // The corner snaps to the nearest whole pixel, and the patch keeps the
        // pixel size it was painted at.
        assert_eq!(
            (dest.x, dest.y),
            ((cx - 766.0).round(), (cy - 590.0).round())
        );
        assert_eq!((dest.width, dest.height), (1532.0, 1180.0));
        // Every edge sits on a whole window pixel, so the bilinear sample lands
        // on the texel centers of a 1:1 patch.
        assert_eq!(dest.x.fract(), 0.0);
        assert_eq!(dest.y.fract(), 0.0);
        assert_eq!((dest.right().fract(), dest.bottom().fract()), (0.0, 0.0));
    }

    #[test]
    fn a_patch_rounds_into_the_painted_pixel_size() {
        // The paint rounds its buffer, so a screen size within a pixel takes
        // the painted size instead of resampling the whole patch.
        let center = Vec2::new(400.0, 300.0);
        let dest = patch_dest(center, 800.6, 600.4, 801.0, 600.0, 0.0).unwrap();
        assert_eq!((dest.width, dest.height), (801.0, 600.0));
        assert_eq!((dest.x, dest.y), (0.0, 0.0));
    }

    #[test]
    fn a_stretched_or_turned_patch_keeps_its_true_size() {
        // A zoom past the slack leaves the patch stretched, so its pixel size
        // is not the screen size: the size follows the view, and the position
        // still snaps so sampling stays stable.
        let center = Vec2::new(400.4, 300.6);
        assert_eq!(
            patch_dest(center, 1200.0, 900.0, 800.0, 600.0, 0.0),
            Some(Rect::new(-200.0, -149.0, 1200.0, 900.0))
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
            assert_eq!(col.row(pan, row_h), row);
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
        assert_eq!(col.row(pan, row_h), row);
        // Above the plane top the row stays the first one.
        assert_eq!(col.row(col.top_pan() + 500.0, row_h), 0);
        // A book that fits the window has no row to read.
        assert_eq!(column(200.0, 1.0, 768.0).row(0.0, 100.0), 0);
    }

    #[test]
    fn a_region_drag_wins_over_the_kept_region() {
        // The window holds the plane one to one, so the pointer and the source
        // units agree here and the test reads the region alone.
        let frame = region::Frame {
            img: Vec2::new(1000.0, 500.0),
            win: Vec2::new(1000.0, 500.0),
            pan: Vec2::ZERO,
            scale: 1.0,
            angle: 0.0,
        };
        let kept = region::Crop {
            x: 10.0,
            y: 10.0,
            w: 100.0,
            h: 100.0,
        };
        let drag = region_sel::Drag::new(Vec2::new(200.0, 100.0), Vec2::ZERO, false);
        let still = region_sel::Drag::new(Vec2::ZERO, Vec2::ZERO, false);
        // A press that has not left the tap slack spans no rect, so the old
        // region keeps its place on screen.
        assert_eq!(
            selection_rect(Some(frame), Some(kept), Some(still), Vec2::ZERO),
            Some(kept)
        );
        // Any travel replaces it with the fresh rect.
        assert_eq!(
            selection_rect(Some(frame), Some(kept), Some(drag), Vec2::new(500.0, 400.0)),
            Some(region::Crop {
                x: 200.0,
                y: 100.0,
                w: 300.0,
                h: 300.0,
            })
        );
        // A plane with no size and no drag marks nothing.
        assert_eq!(selection_rect(None, Some(kept), None, Vec2::ZERO), None);
    }

    #[test]
    fn a_region_drag_reads_the_pointer_in_source_units() {
        // Half scale and a pan of 100 window pixels to the right: the window
        // middle sits on the plane middle, one window pixel spans two source
        // pixels, and the pan moves the picture the other way.
        let frame = region::Frame {
            img: Vec2::new(1000.0, 500.0),
            win: Vec2::new(800.0, 600.0),
            pan: Vec2::new(100.0, 0.0),
            scale: 0.5,
            angle: 0.0,
        };
        let drag = region_sel::Drag::new(Vec2::new(500.0, 250.0), Vec2::ZERO, false);
        // The pointer at (600, 200) lands on the source point (700, 50), so the
        // rect runs from the anchor to there.
        // A raw window reading would answer (500, 200, 100, 50) instead.
        assert_eq!(
            selection_rect(Some(frame), None, Some(drag), Vec2::new(600.0, 200.0)),
            Some(region::Crop {
                x: 500.0,
                y: 50.0,
                w: 200.0,
                h: 200.0,
            })
        );
        // A press that has not left the tap slack keeps the old region, so a
        // `v` click never flashes a sliver rect.
        let tap = region_sel::Drag::new(Vec2::new(500.0, 250.0), Vec2::new(600.0, 200.0), false);
        let kept = region::Crop {
            x: 10.0,
            y: 10.0,
            w: 100.0,
            h: 100.0,
        };
        assert_eq!(
            selection_rect(Some(frame), Some(kept), Some(tap), Vec2::new(601.0, 200.0)),
            Some(kept)
        );
        // The release reads the same call, so a tap answers no region and
        // drops the kept one.
        assert_eq!(
            selection_rect(Some(frame), None, Some(tap), Vec2::new(601.0, 200.0)),
            None
        );
    }
}
