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
        mpsc::{Receiver, Sender},
    },
    time::{Duration, Instant},
};

use winit::event::MouseButton;

use crate::{
    DIM_VEIL, DecodedImage,
    blurbg::BlurBg,
    dim::{PreviewPx, RampTime, Rect, WinH, WinW},
    draw::{Color, Quad, Turn, Vec2},
    grid::{DecodeFailure, EntrySource, Grid},
    input::Input,
    loader::Loader,
    region, region_sel,
    render::Renderer,
    rotate,
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

/// The time a held-key motion ramps in or out, in seconds.
/// A hold reaches cruise speed after this time, and a release coasts to rest
/// over it.
pub const RAMP_TIME: RampTime = RampTime::new(0.36);

/// How far one unit of raw device travel moves the picture during a drag.
///
/// A grab drag reads the raw device deltas of `DeviceEvent::MouseMotion`, which
/// are counts of the mouse, not window pixels, so this factor is a feel
/// constant and not a geometry one.
/// 1.0 moves the picture one window pixel per count: the spike at
/// `~/agent/versatile-viewer-wgpu-spike` picked it by feel and the viewer's own
/// author confirmed that feel, so it is the default here.
/// The raylib path multiplied the same raw travel by 2.0, and `git bug show
/// 94691ce` records that history.
/// `VV_PAN_DRAG_SPEED` overrides it, so a feel comparison needs no rebuild, and
/// `VV_DEBUG` prints the travel a drag accumulated (see the release line).
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

/// The vertical travel that a held Shift+F needs to zoom by one natural-log
/// e-fold, in window long sides.
/// Up (a negative y delta) zooms in, and down zooms out.
/// 1.5 suits fine aiming: a screen height of travel zooms one e-fold (2.7x),
/// so a small drag is a small change.
pub const ZOOM_DRAG_SPAN: f32 = 1.5;

/// A multiplier on the Shift+F zoom rate alone.
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
#[must_use]
pub fn aim_fill_pan(st: &ViewState, mouse: Vec2, win_w: f32, win_h: f32) -> Vec2 {
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
    Free(f32),
}

impl ZoomMode {
    /// The free scale, or the fit scale when the mode is not free.
    #[must_use]
    pub fn free_or(self, fit: f32) -> f32 {
        match self {
            Self::Free(scale) => scale,
            _ => fit,
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
    pub pan: Vec2,
    pub target_pan: Vec2,
    /// The on-screen scale, eased toward the target scale each frame.
    pub view_scale: Option<f32>,
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
    /// The plane of the open source that the view shows.
    /// It is 0 for every source the viewer opens: a book stacks its spreads
    /// into one plane (see [`versatile_viewer::book`]).
    pub source_plane: usize,
    /// The part of the open plane that a fresh open frames, in surface units
    /// (see [`Surface::open_box`]).
    /// A book answers its first spread, so the cover page shows whole; every
    /// other source answers `None` and the fit frames the whole plane.
    pub open_box: Option<Size>,
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
    /// The patch this state shows now that its paint landed, if any.
    #[must_use]
    pub fn on_screen_patch(&self) -> Option<&CropState> {
        match self {
            Self::Ready(state) => Some(state),
            Self::Idle | Self::Painting { .. } => None,
        }
    }

    /// The aim a worker paints now, with the token and the start time.
    #[must_use]
    pub fn painting(&self) -> Option<(region::Patch, u64, Instant)> {
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
        Some(region::Frame {
            img: Vec2::new(self.img.w, self.img.h),
            win,
            pan: self.pan,
            scale: self.view_scale?,
            angle: self.angle,
        })
        .filter(region::Frame::has_drawable_plane)
    }

    /// The window mapping of the open plane at `pan` and `scale`.
    ///
    /// [`Self::frame`] answers the same mapping for the pan and the scale on
    /// screen; a caller that aims elsewhere (the target pan of a glide) asks
    /// here, so it never assembles the five numbers itself.
    #[must_use]
    pub fn frame_at(&self, win: Vec2, pan: Vec2, scale: f32) -> region::Frame {
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
    pub fn step_spread(&mut self, step: isize, win_h: f32) -> bool {
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
        let row = column
            .spread_at_top(self.target_pan.y, first.h)
            .min(count - 1);
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
    pub fn prefetch_spread(&mut self, win: Vec2, uploader: &Uploader) {
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
    pub fn close_crop(&mut self) {
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
    /// A spread step paints its landing region before the glide ends, so a
    /// patch that already covers the window replaces the one on screen here and
    /// skips the debounce and the paint.
    pub fn tick_crop(&mut self, now: Instant, win: Vec2, uploader: &Uploader) {
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
            .on_screen_patch()
            .is_some_and(|state| region::patch_covers_at_scale(&needed, &state.patch))
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
            && region::patch_covers_at_scale(&needed, &aim)
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
    pub fn drain_crops(&mut self) {
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
pub fn patch_dest(center: Vec2, dw: f32, dh: f32, tw: f32, th: f32, angle: f32) -> Option<Rect> {
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
pub fn draw_patch(r: &mut Renderer, patch: &CropState, frame: &region::Frame, filter: Filter) {
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
pub fn selection_rect(
    frame: Option<region::Frame>,
    kept: Option<region::Crop>,
    drag: Option<region_sel::Drag>,
    mouse: Vec2,
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
/// matches the grid's dim tint (see [`versatile_viewer::DIM_LEVEL`]).
/// The bands and the border map through [`region::Frame::to_screen`], so they
/// follow the pan, the zoom and the turn of the draw.
pub fn draw_selection(r: &mut Renderer, frame: region::Frame, selection: region::Crop) {
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
#[must_use]
pub fn same_patch(a: &region::Patch, b: &region::Patch) -> bool {
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
    if st.img.w <= 0.0 || !(scale.is_finite() && scale > 0.0) {
        return false;
    }
    let Some(frame) = st.frame(Vec2::new(win_w, win_h)) else {
        return false;
    };
    let needed = region::Patch::new(frame.crop(0.0), scale);
    region::patch_reaches_window(&needed, &crop.patch)
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
                    st.open_box = surface.open_box(plane);
                    Some(Loader::start_plane(
                        surface,
                        plane,
                        preview_px,
                        &Some(r.uploader()),
                    ))
                }
                Err(err) => {
                    eprintln!("vv: {}: {err}", source.log_label());
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
