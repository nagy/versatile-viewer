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
    env, fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        mpsc::{Receiver, Sender, channel},
    },
    time::Instant,
};

use anyhow::{Context, Result, bail};
use raylib::{
    color::Color,
    consts::{KeyboardKey, MouseCursor, TextureFilter},
    prelude::*,
    texture::RaylibTexture2D,
};
#[cfg(target_os = "linux")]
use versatile_viewer::wmclass;
use versatile_viewer::{
    DecodedImage, UploadError,
    blurbg::{self, BlurBg},
    dim::{ImgH, ImgW, PreviewPx, RampTime, Rect, WinH, WinW},
    grid::{DecodeFailure, Grid, GridAction},
    keyrepeat,
    loader::{DecodeMsg, Loader},
    mcp, region, rotate,
    surface::{Surface, View, open_surface},
    svg,
    tween::Ramp,
    upload_rgba,
    zoomkeys::TypedKeys,
};

/// The window background color, `#062329`.
/// The grid and the image view both clear to it, so the letterbox bars and the
/// gaps between thumbnails match.
const BACKGROUND: Color = Color::new(0x06, 0x23, 0x29, 0xff);

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

/// Target pan that centers the fill view on the image point under `mouse`.
///
/// The code clamps both axes to `[-|offset|, |offset|]`.
/// Outside that interval a window edge exposes background.
/// A cursor near the image edge therefore pans only as far as coverage
/// allows.
fn aim_fill_pan(st: &ViewState, mouse: Vector2, win_w: f32, win_h: f32) -> Vector2 {
    let target = (win_w / st.img_w).max(win_h / st.img_h);
    let cur = st.view_scale.unwrap_or(target);
    // Mouse position in image coordinates at the current scale.
    // The offset is the window top-left of the unpanned image.
    let img_x = (mouse.x - (win_w - st.img_w * cur) / 2.0 - st.pan.x) / cur;
    let img_y = (mouse.y - (win_h - st.img_h * cur) / 2.0 - st.pan.y) / cur;
    let off_x = (win_w - st.img_w * target) / 2.0;
    let off_y = (win_h - st.img_h * target) / 2.0;
    Vector2 {
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
    view_tex: Option<Texture2D>,
    view_loading: bool,
    /// The full-resolution image dimensions (0 until known).
    img_w: f32,
    img_h: f32,
    zoom: ZoomMode,
    /// The on-screen pan offset, eased toward `target_pan` each frame.
    pan: Vector2,
    target_pan: Vector2,
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
    zoom_anchor: Option<Vector2>,
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
    pivot: Option<Vector2>,
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
    /// The region the window needed when the debounce last saw it move.
    crop_seen: Option<region::Crop>,
    /// The scale the debounce last saw.
    crop_seen_scale: f32,
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
    /// The pixels of the patch.
    texture: Texture2D,
    /// The source rect the patch covers, in base raster pixels.
    rect: region::Crop,
    /// The raster scale, in texture pixels per source pixel.
    scale: f32,
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
    /// A finished patch of `rect` at `scale` pixels per source pixel.
    Crop {
        generation: u64,
        rect: region::Crop,
        scale: f32,
        image: DecodedImage,
    },
    /// A paint that failed, so the view drops the pending flag and may ask
    /// again.
    Failed { generation: u64 },
}

impl ViewState {
    /// Reset the fit and pan state for a freshly shown image.
    fn reset_view(&mut self) {
        // Freshly shown images open fit-all (Shift+W behavior): upscale or
        // downscale until the image first touches a window border.
        self.zoom = ZoomMode::FitAll;
        self.pan = Vector2::ZERO;
        self.target_pan = Vector2::ZERO;
        self.zoom_anchor = None;
        self.view_scale = None;
        // A newly opened image starts upright: a turn never carries over to
        // the next image.
        self.angle = 0.0;
        self.pivot = None;
        self.ramps = Ramps::default();
    }

    /// Drop the region state of the image that was open, and retire every
    /// worker result still in flight.
    fn close_crop(&mut self) {
        self.source = None;
        self.crop = None;
        self.crop_pending = false;
        self.crop_seen = None;
        self.crop_seen_scale = 0.0;
        self.crop_quiet_at = Instant::now();
        self.crop_gen = self.crop_gen.wrapping_add(1);
    }

    /// Adopt a source that arrived with a decode, so the view can repaint the
    /// visible region at any zoom.
    fn adopt_source(&mut self, source: Option<Arc<dyn Surface>>) {
        self.close_crop();
        self.source = source;
    }

    /// Ask a worker for the source of `path`.
    ///
    /// The grid keeps no source, so an image opened from the grid opens one
    /// here, off the main thread.
    /// A raster file answers None at once.
    fn request_source(&mut self, path: PathBuf) {
        self.close_crop();
        let (generation, tx) = (self.crop_gen, self.crop_tx.clone());
        rayon::spawn(move || {
            // The sniff reads the head only, so a 200 MB photo never makes a
            // worker read the whole file.
            let source: Option<Arc<dyn Surface>> = if svg::sniff(&path) {
                open_surface(&path).ok().map(Arc::from)
            } else {
                None
            };
            let _ = tx.send(CropMsg::Source { generation, source });
        });
    }

    /// Repaint the visible region once the view holds still, when the patch on
    /// screen no longer covers the window at the right scale.
    ///
    /// The paint area follows the window, not the document, so a deep zoom
    /// costs no more than a shallow one.
    /// The guard is one paint in flight: resvg cannot cancel mid-paint.
    fn tick_crop(&mut self, now: Instant, win: Vector2) {
        if self.source.is_none() || self.crop_pending || self.img_w <= 0.0 || self.view_loading {
            return;
        }
        let Some(scale) = self.view_scale else {
            return;
        };
        if !(scale.is_finite() && scale > 0.0) {
            return;
        }
        let img = Vector2::new(self.img_w, self.img_h);
        let needed = region::crop(img, win, self.pan, scale, self.angle, 0.0);
        // The debounce: any movement of the seen region restarts the timer, so
        // a drag, a pan or an ease never queues a paint per frame.
        let moved = self.crop_seen.is_none_or(|seen| {
            (seen.x - needed.x).abs() > 0.5
                || (seen.y - needed.y).abs() > 0.5
                || (seen.w - needed.w).abs() > 0.5
                || (seen.h - needed.h).abs() > 0.5
                || (self.crop_seen_scale - scale).abs() > scale * 0.002
        });
        if moved {
            self.crop_seen = Some(needed);
            self.crop_seen_scale = scale;
            self.crop_quiet_at = now;
            return;
        }
        if now.duration_since(self.crop_quiet_at) < region::QUIET {
            return;
        }
        let painted = self.crop.as_ref().map(|c| (c.rect, c.scale));
        if !region::worth_painting(
            &needed,
            painted.as_ref().map(|(rect, _)| rect),
            scale,
            painted.map_or(0.0, |(_, painted_scale)| painted_scale),
        ) {
            return;
        }
        let Some(source) = self.source.clone() else {
            return;
        };
        // Paint a margin around what the window shows, so a small pan between
        // two paints stays covered, and a filter keeps its outer context.
        let paint = region::crop(img, win, self.pan, scale, self.angle, region::MARGIN);
        if paint.is_empty() {
            return;
        }
        self.crop_pending = true;
        let (generation, tx) = (self.crop_gen, self.crop_tx.clone());
        rayon::spawn(
            move || match source.render(View::region(0, paint.into(), scale)) {
                Ok(image) => {
                    let _ = tx.send(CropMsg::Crop {
                        generation,
                        rect: paint,
                        scale,
                        image,
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
    /// A source lands as is, and a patch uploads and replaces the one on
    /// screen.
    /// A result of an earlier image drops.
    fn drain_crops(
        &mut self,
        rl: &mut RaylibHandle,
        thread: &RaylibThread,
    ) -> Result<(), UploadError> {
        while let Ok(msg) = self.crop_rx.try_recv() {
            match msg {
                CropMsg::Source { generation, source } => {
                    if generation == self.crop_gen {
                        self.source = source;
                    }
                }
                CropMsg::Crop {
                    generation,
                    rect,
                    scale,
                    image,
                } => {
                    if generation != self.crop_gen {
                        continue;
                    }
                    self.crop_pending = false;
                    let texture = upload_rgba(rl, thread, &image.data, image.width, image.height)?;
                    apply_view_filter(thread, &texture, self.pixelated);
                    // The old patch drops here, so one patch lives at a time.
                    self.crop = Some(CropState {
                        texture,
                        rect,
                        scale,
                    });
                }
                CropMsg::Failed { generation } => {
                    if generation == self.crop_gen {
                        // A failure waits for the next quiet window instead of
                        // retrying every frame.
                        self.crop_pending = false;
                        self.crop_quiet_at = Instant::now();
                    }
                }
            }
        }
        Ok(())
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
        Motion {
            pan: Vector2 {
                x: pan_speed * (l - r),
                y: pan_speed * (u - d),
            },
            zoom_factor: (ZOOM_RATE * speed * dt * (zi - zo)).exp(),
            zoom_moving: zi > 0.0 || zo > 0.0,
        }
    }
}

/// One frame of output from the held-key ramps.
#[derive(Debug, Clone, Copy)]
struct Motion {
    /// The pan delta for this frame, in window pixels.
    pan: Vector2,
    /// The free-zoom factor for this frame.
    /// 1.0 when no zoom key moves.
    zoom_factor: f32,
    /// True while a +/- ramp gives speed.
    zoom_moving: bool,
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
/// A held side button auto-repeats through `keyrepeat`, exactly like `n` and
/// `p`.
fn read_nav_mouse(rl: &RaylibHandle) -> NavMouse {
    let buttons = |list: &[MouseButton]| NavButton {
        edge: list.iter().any(|&b| rl.is_mouse_button_pressed(b)),
        held: list.iter().any(|&b| rl.is_mouse_button_down(b)),
    };
    NavMouse {
        next: buttons(&[
            MouseButton::MOUSE_BUTTON_EXTRA,
            MouseButton::MOUSE_BUTTON_FORWARD,
        ]),
        prev: buttons(&[
            MouseButton::MOUSE_BUTTON_SIDE,
            MouseButton::MOUSE_BUTTON_BACK,
        ]),
    }
}

/// Show a new RGBA8 frame in image view.
/// When a texture with the same dimensions exists (progressive previews of
/// the same image), the function updates its pixels in place instead of
/// reallocating a GPU texture.
fn show_frame(
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    view_tex: &mut Option<Texture2D>,
    rgba: &[u8],
    width: ImgW,
    height: ImgH,
    pixelated: bool,
) -> Result<(), UploadError> {
    match view_tex {
        Some(tex) if tex.width() == width.get() as i32 && tex.height() == height.get() as i32 => {
            tex.update_texture(rgba)?;
        }
        _ => {
            let tex = upload_rgba(rl, thread, rgba, width, height)?;
            apply_view_filter(thread, &tex, pixelated);
            *view_tex = Some(tex);
        }
    }
    Ok(())
}

/// Set the texture filter for the image view.
/// The default is bilinear, which suits photos.
/// `a` toggles nearest-neighbor (pixelated) for 1:1 pixel peeping.
/// Grid thumbs always stay bilinear, because they are heavily downscaled and
/// nearest aliases badly.
fn apply_view_filter(thread: &RaylibThread, tex: &Texture2D, pixelated: bool) {
    tex.set_texture_filter(
        thread,
        if pixelated {
            TextureFilter::TEXTURE_FILTER_POINT
        } else {
            TextureFilter::TEXTURE_FILTER_BILINEAR
        },
    );
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
/// Raylib turns the quad around `dest.position`, and `origin` pulls the rect
/// back by half its size.
/// The dest position is therefore the image center, so the turn pivots on the
/// middle of the image at every angle.
/// A zero angle draws the plain centered rect, because the origin pulls it
/// back by the same half size.
fn draw_turned(
    d: &mut impl RaylibDraw,
    texture: &Texture2D,
    img_w: f32,
    img_h: f32,
    scale: f32,
    center: Vector2,
    angle: f32,
) {
    let src = Rect::sized(texture.width() as f32, texture.height() as f32);
    let dw = img_w * scale;
    let dh = img_h * scale;
    let dest = Rect::new(center.x, center.y, dw, dh);
    d.draw_texture_pro(
        texture,
        src,
        dest,
        dest.center_origin(),
        rotate::normalize_deg(angle),
        Color::WHITE,
    );
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
    if st.img_w <= 0.0 || !(scale.is_finite() && scale > 0.0) {
        return false;
    }
    let img = Vector2::new(st.img_w, st.img_h);
    let win = Vector2::new(win_w, win_h);
    let needed = region::crop(img, win, st.pan, scale, st.angle, 0.0);
    crop.rect.contains(&needed)
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
    let center = Vector2::new(win_w / 2.0, win_h / 2.0);
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
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    blur: Option<&DecodedImage>,
    tag: Option<u64>,
) {
    if let (Some(bg), Some(data)) = (blur_bg.as_mut(), blur) {
        bg.attach(rl, thread, data, tag);
    }
}

/// Put the viewed texture back into its grid entry, if it came from one, and
/// clear the view texture.
/// The code calls this when leaving image mode or switching to another grid
/// entry.
fn put_back_view(
    grid: &mut Option<Grid>,
    view_from_grid: &mut Option<u64>,
    view_tex: &mut Option<Texture2D>,
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
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    win: (WinW, WinH),
    set_scale_now: bool,
) {
    let (win_w, win_h) = (win.0.get(), win.1.get());
    let (id, tex, w, h, path, queued, thumb, blur) = {
        let e = &mut grid.entries[idx];
        (
            e.id,
            e.full.take(),
            e.width,
            e.height,
            e.path.clone(),
            e.queued,
            e.texture.is_some(),
            e.blur.clone(),
        )
    };
    st.open_id = Some(id);
    st.mode = Mode::Image;
    st.reset_view();
    // A new image starts with no region patch: the decode or the source
    // request names what the window shows.
    st.close_crop();
    if let Some(tex) = tex {
        grid.entries[idx].viewing = true;
        st.view_from_grid = Some(id);
        apply_view_filter(thread, &tex, st.pixelated);
        st.view_tex = Some(tex);
        st.img_w = w.as_f32();
        st.img_h = h.as_f32();
        st.loader = None;
        st.view_loading = false;
        // The grid keeps no parsed tree, so the view parses it on a worker
        // before a deeper zoom can repaint.
        st.request_source(path);
        if set_scale_now {
            st.view_scale = Some((win_w / st.img_w).min(win_h / st.img_h));
        }
        attach_blur_bg(&mut st.blur_bg, rl, thread, blur.as_ref(), st.open_id);
    } else {
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
        if w.get() > 0 {
            st.img_w = w.as_f32();
            st.img_h = h.as_f32();
            if set_scale_now {
                st.view_scale = Some((win_w / st.img_w).min(win_h / st.img_h));
            }
        } else {
            st.img_w = 0.0; // dimensions arrive with the header/texture
            st.img_h = 0.0;
        }
        st.loader = if queued || thumb {
            // A grid decode for this entry is in flight, or the entry has
            // a thumb and the keep-set refill dispatched (or will dispatch)
            // the missing full-res decode.
            // Wait for the decode-landed takeover instead of decoding the
            // file twice in a streaming worker.
            None
        } else {
            let preview_px =
                PreviewPx::new(rl.get_screen_width().max(rl.get_screen_height()) as u32);
            match open_surface(&path) {
                Ok(surface) => Some(Loader::start(surface, preview_px)),
                Err(err) => {
                    eprintln!("vv: {}: {err}", path.display());
                    grid.mark_failed(id, DecodeFailure::Decode);
                    None
                }
            }
        };
        attach_blur_bg(&mut st.blur_bg, rl, thread, blur.as_ref(), st.open_id);
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
/// The server owns stdin and stdout, so raylib must not write its trace log
/// there. The code redirects the trace log to stderr instead.
///
/// The whole program is single-threaded at this point.
fn start_mcp() -> Result<()> {
    raylib::core::callbacks::set_trace_log_callback(|_level, text| {
        eprintln!("vv: raylib: {text}");
    })
    .context("failed to redirect the raylib trace log")?;
    mcp::spawn().context("failed to start the MCP server")
}

// The event loop is one long state machine by design.
// Splitting it scatters the frame-order invariants across call sites.
#[allow(clippy::too_many_lines)]
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

    // Directory launch: list the directory before opening the window (the
    // title shows the image count).
    // No GL is involved yet.
    // Single-file launch: no grid.
    let dir_grid = if path.is_dir() {
        Some(Grid::from_dir(path)?)
    } else if path.is_file() {
        None
    } else {
        bail!("no such file or directory: {}", path.display());
    };

    // Single-image launch: render before opening the window, so the window
    // can be sized to the image.
    // Directory launch: fixed default window size.
    // A vector file keeps its surface, so the image view can repaint the
    // region its window shows.
    let single_decoded = if dir_grid.is_none() {
        let surface = open_surface(path)?;
        let image = surface.render(View::whole(0, 1.0))?;
        let source: Option<Arc<dyn Surface>> = if svg::sniff(path) {
            Some(Arc::from(surface))
        } else {
            None
        };
        Some((image, source))
    } else {
        None
    };
    // Blur-background source for a single-file launch: computed while the
    // RGBA buffer still exists (before the window and GL context exist).
    // The code uploads it once the window is open.
    let single_blur = if dir_grid.is_none() && blurbg::enabled() {
        let (image, _) = single_decoded.as_ref().expect("single file decoded");
        Some(blurbg::small_blur(image, blurbg::blur_px()))
    } else {
        None
    };
    let (win0_w, win0_h) = single_decoded.as_ref().map_or((1024, 768), |(image, _)| {
        (image.width.get() as i32, image.height.get() as i32)
    });

    let (mut rl, thread) = raylib::init()
        .size(win0_w, win0_h)
        .title(&window_title(path, dir_grid.as_ref()))
        .resizable()
        // Vsync on: the compositor or driver paces the frame loop to the
        // monitor refresh rate and removes tearing (most visible during the
        // zoom ease).
        // This replaces set_target_fps below.
        // A software 60 FPS cap fights a non-60 Hz monitor (judder) and
        // adds input latency.
        .vsync()
        .build();
    // WM_CLASS: raylib derives it from the creation title (the full path,
    // which is not useful for window rules).
    // The code stamps "vv" on X11 and does nothing on Wayland.
    #[cfg(target_os = "linux")]
    // SAFETY: the code only reads the handle.
    // The handle stays valid while the window is open.
    unsafe {
        wmclass::set_class(rl.get_window_handle());
    }
    // The code quits through the q key handling.
    // set_exit_key makes ESC close the window outright instead of
    // returning to the grid.
    rl.set_exit_key(None);
    // Explicit default-arrow cursor while the mouse hovers the window.
    rl.set_mouse_cursor(MouseCursor::MOUSE_CURSOR_ARROW);

    // VV_BLUR_BG gimmick: a blurred copy of the viewed image behind it.
    // The code declares it after `rl`, so it drops (and unloads its texture)
    // before the window.
    // Take the grid AFTER the window exists: it holds GPU textures, and
    // because the code declares it after `rl`, it drops (and unloads) BEFORE
    // the window closes, on normal exit and on panic unwinding.
    let mut grid = dir_grid;
    // Image-view state: what is shown and how it is framed.
    // Its Loader field drops (and cancels the worker) before the window
    // closes, like `grid`.
    let (crop_tx, crop_rx) = channel();
    let mut st = ViewState {
        mode: if grid.is_none() {
            Mode::Image
        } else {
            Mode::Grid
        },
        open_id: None,
        view_from_grid: None,
        view_tex: None,
        view_loading: false,
        img_w: 0.0,
        img_h: 0.0,
        zoom: ZoomMode::FitDown,
        pan: Vector2::ZERO,
        target_pan: Vector2::ZERO,
        view_scale: None,
        zoom_anchor: None,
        angle: 0.0,
        pivot: None,
        ramps: Ramps::default(),
        pixelated: false,
        loader: None,
        blur_bg: BlurBg::from_env(),
        source: None,
        crop: None,
        crop_pending: false,
        crop_gen: 0,
        crop_seen: None,
        crop_seen_scale: 0.0,
        crop_quiet_at: Instant::now(),
        crop_tx,
        crop_rx,
    };
    if let Some((image, source)) = single_decoded {
        st.view_tex = Some(upload_rgba(
            &mut rl,
            &thread,
            &image.data,
            image.width,
            image.height,
        )?);
        apply_view_filter(
            &thread,
            st.view_tex.as_ref().expect("just uploaded"),
            st.pixelated,
        );
        st.adopt_source(source);
        st.img_w = image.width.as_f32();
        st.img_h = image.height.as_f32();
        if let Some(data) = single_blur {
            attach_blur_bg(&mut st.blur_bg, &mut rl, &thread, Some(&data), None);
        }
    }
    // A one-image directory skips the grid: jump straight into that image
    // (ESC still returns to the grid view).
    if grid.as_ref().is_some_and(|g| g.entries.len() == 1) {
        let win = (
            WinW::new(rl.get_screen_width() as f32),
            WinH::new(rl.get_screen_height() as f32),
        );
        show_entry(
            &mut st,
            grid.as_mut().unwrap(),
            0,
            &mut rl,
            &thread,
            win,
            true,
        );
    }

    // Window title: grid mode and single-file launch keep the launch title.
    // Image view rewrites it per open entry (reset when open_id changes).
    let launch_title = window_title(path, grid.as_ref());
    let mut title_open_id: Option<u64> = None;

    // Set when the viewer exits entirely.
    let mut quit = false;
    // VV_DEBUG=1: trace grid open/return events to stderr.
    let debug = std::env::var_os("VV_DEBUG").is_some();
    // VV_TIMING=1: trace slow frames to stderr.
    let timing = versatile_viewer::timing_enabled();
    // Previous per-key down state for the VV_DEBUG event trace.
    let mut prev_down = [false; 349];

    // No set_target_fps: vsync paces the frame loop (see the builder above).
    // Auto-repeat state for image-mode prev/next (xset r rate values).
    let mut rep_nav_fwd = keyrepeat::RepeatState::default();
    let mut rep_nav_back = keyrepeat::RepeatState::default();
    // Previous frame window size: self-tracked resize detection (see the
    // `resized` computation in the loop).
    let mut last_win: Option<(f32, f32)> = None;
    // True while an image-view drag has the pointer captured (DisableCursor:
    // hidden and locked, unbounded virtual deltas, so the drag cannot hit a
    // screen edge).
    // The code releases it on mouse-up.
    // The pointer then reappears exactly where the drag started (GLFW
    // restores the pre-capture position).
    let mut pointer_captured = false;
    // Window position where the current (or last) drag grabbed the pointer.
    // The raylib EnableCursor does not reliably restore it, so the code warps
    // back explicitly on release.
    let mut grab_pos = Vector2::ZERO;
    // Virtual cursor position while captured (grab_pos + accumulated raw
    // deltas).
    // The raylib virtual position is unreliable to interpret.
    let mut drag_virtual = Vector2::ZERO;
    // Post-drag pointer restore, re-checked for a few frames: X11 warps are
    // async, and the GLFW EnableCursor re-warp can land after ours and
    // recenter the cursor.
    // The code warps again until the position sticks.
    let mut pending_restore: Option<Vector2> = None;
    let mut restore_tries = 0u32;
    // `f` held captures the pointer like a left-drag (hold = pan, cursor
    // hidden).
    // Shift+F held zooms instead: it captures the pointer the same way, and
    // the vertical mouse travel drives the free zoom (up zooms in, down zooms
    // out) on the wheel's anchor path.
    // A tap (no pointer travel) runs the `f` action on release instead, so
    // the code can tell a press and a press-and-hold apart.
    let mut f_capture = false;
    // True when the running `f` hold zooms instead of panning.
    // A Shift at the `f` press edge starts in the zoom, and Shift going down
    // or up mid-hold switches the mode.
    let mut f_zoom_capture = false;
    // Set when the running `f` hold ever zoomed, so its release runs no tap
    // action even though a zoom hold travels the pointer no distance.
    let mut f_zoom_tapped = false;
    // The free-zoom drag's on-screen anchor.
    // A zoom hold that starts with Shift at the grab point anchors there, and
    // the zoom drift then walks it toward the window center, exactly like the
    // wheel zoom's anchor.
    // The capture keeps the pointer still, so the image point under the anchor
    // is what slides toward the center on screen.
    // A switch from the pan hold anchors at the window center instead (see
    // the switch below).
    let mut f_zoom_anchor: Option<Vector2> = None;
    // `r` held captures the pointer like the `f` hold, and the vertical mouse
    // travel turns the image: down turns it clockwise and up turns it
    // counter-clockwise.
    // The turn adds to the angle the image shows now, so a second hold
    // continues from where the first one left it.
    // The release keeps the angle, so the image stays turned.
    let mut r_capture = false;
    // The held state of the zoom keys on layouts where `+`/`-` do not sit on
    // their US keycodes.
    // The raw character events repeat at the X auto-repeat rate, so this
    // turns them into a timed hold that the motion ramps can read.
    let mut zoom_keys = TypedKeys::default();

    while !rl.window_should_close() && !quit {
        // VV_TIMING: the frame started now.
        // The code reports it at the end of the frame when it ran long.
        let frame_started = timing.then(std::time::Instant::now);
        // VV_DEBUG: trace every key raylib sees (keycode per raylib/GLFW:
        // 257=Enter, 335=KpEnter, 256=Esc, 262/263=Right/Left,
        // 264/265=Up/Down, 72/74/75/76=h/j/k/l).
        if debug {
            for k in 32u32..=348 {
                let (down, pressed) = unsafe {
                    (
                        raylib::ffi::IsKeyDown(k as i32),
                        raylib::ffi::IsKeyPressed(k as i32),
                    )
                };
                if pressed {
                    eprintln!("vv: PRESSED key {k}");
                }
                if down != prev_down[k as usize] {
                    eprintln!("vv: key {k} {}", if down { "DOWN" } else { "UP" });
                    prev_down[k as usize] = down;
                }
            }
            // Frames longer than a key tap can swallow the IsKeyPressed edge
            // detection entirely (press+release between two polls).
            // The code logs them.
            let ft = rl.get_frame_time();
            if ft > 0.1 {
                eprintln!("vv: slow frame {ft:.0} ms");
            }
        }
        let win_w = rl.get_screen_width() as f32;
        let win_h = rl.get_screen_height() as f32;
        // Post-drag restore: verify the warp landed where the code sent it.
        // If a competing re-warp (GLFW, WM, XWayland) moved it, snap again.
        // The code gives up after a few tries so a user moving the mouse is
        // never pinned.
        if let Some(target) = pending_restore {
            let cur = rl.get_mouse_position();
            let settled = (cur.x - target.x).abs() < 1.0 && (cur.y - target.y).abs() < 1.0;
            if debug && !settled {
                eprintln!(
                    "vv: restore try {restore_tries}: cur ({:.0},{:.0}) -> ({:.0},{:.0})",
                    cur.x, cur.y, target.x, target.y
                );
            }
            if settled || restore_tries >= 3 {
                pending_restore = None;
            } else {
                rl.set_mouse_position(target);
                restore_tries += 1;
            }
        }
        // The code does the resize detection itself.
        // `is_window_resized()` can miss a WM-reflow resize that lands around
        // the time the window becomes visible (tiling WMs shrink the freshly
        // spawned window into its tile), so the fit ease glides "out of
        // nowhere".
        // Any size difference from the previous frame counts as a resize
        // frame.
        let resized = last_win != Some((win_w, win_h));
        last_win = Some((win_w, win_h));

        // Dynamic window title: follow the open image in image view.
        // Back to the launch title in grid mode.
        if st.open_id != title_open_id {
            title_open_id = st.open_id;
            let title = st
                .open_id
                .and_then(|id| {
                    grid.as_ref()?
                        .entries
                        .iter()
                        .find(|e| e.id == id)
                        .map(|e| image_title(&e.path))
                })
                .unwrap_or_else(|| launch_title.clone());
            rl.set_window_title(&thread, &title);
        }

        // Prefetch priority: in grid mode the neighbors of the selected
        // entry (left, right, up, down).
        // In image mode the previous and next entries.
        // While the image view still waits for its own decode, only the
        // open entry decodes and the code pauses the rest of the queue, so
        // the visible image gets all the cores.
        // load_pending drains finished decodes (texture uploads happen here,
        // on the main thread) and dispatches new ones to the rayon pool.
        let (priority, scan_start, scan, preload_full) = if st.mode == Mode::Image {
            match st
                .open_id
                .and_then(|id| grid.as_ref().and_then(|g| g.index_of(id)))
            {
                Some(i) => {
                    let n = grid.as_ref().unwrap().entries.len();
                    let mut v = Vec::new();
                    if st.view_loading {
                        v.push(i);
                    } else {
                        if i > 0 {
                            v.push(i - 1);
                        }
                        if i + 1 < n {
                            v.push(i + 1);
                        }
                    }
                    (v, i, !st.view_loading, true)
                }
                None => (Vec::new(), 0, true, true),
            }
        } else if let Some(g) = grid.as_ref() {
            (
                g.prefetch_neighbors(g.selected, WinW::new(win_w), WinH::new(win_h)),
                g.selected,
                true,
                false,
            )
        } else {
            (Vec::new(), 0, true, false)
        };
        if let Some(g) = grid.as_mut() {
            g.load_pending(&mut rl, &thread, &priority, scan_start, scan, preload_full);
        }

        // Image mode: drain the streaming loader first.
        // Texture uploads need the main thread.
        if st.mode == Mode::Image {
            let mut open_failed = false;
            // Waiting on a grid decode (no streaming loader for this open).
            // When its texture lands, the code takes it over as the view.
            if st.view_tex.is_none() && st.loader.is_none() && st.open_id.is_some() {
                let id = st.open_id.expect("checked above");
                match grid
                    .as_mut()
                    .and_then(|g| g.entries.iter().position(|e| e.id == id).map(|i| (g, i)))
                {
                    Some((g, i)) => {
                        if g.entries[i].full.is_some() {
                            // Decode landed: the code kept its full-res
                            // texture for the open entry (keep set), and
                            // show_entry takes it over.
                            show_entry(
                                &mut st,
                                g,
                                i,
                                &mut rl,
                                &thread,
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
            if let Some(loader) = &st.loader {
                while let Some(msg) = loader.try_recv() {
                    match msg {
                        Ok(DecodeMsg::Header { width, height }) => {
                            // Dimensions known: fit down at once (the ease
                            // block only runs from the next frame on).
                            st.img_w = width.as_f32();
                            st.img_h = height.as_f32();
                            st.zoom = ZoomMode::FitAll;
                            st.pan = Vector2::ZERO;
                            st.target_pan = Vector2::ZERO;
                            st.view_scale = Some((win_w / st.img_w).min(win_h / st.img_h));
                        }
                        Ok(DecodeMsg::Preview { image, blur }) => {
                            // Progressively better render of the same image.
                            show_frame(
                                &mut rl,
                                &thread,
                                &mut st.view_tex,
                                &image.data,
                                image.width,
                                image.height,
                                st.pixelated,
                            )?;
                            attach_blur_bg(
                                &mut st.blur_bg,
                                &mut rl,
                                &thread,
                                blur.as_ref(),
                                st.open_id,
                            );
                        }
                        Ok(DecodeMsg::Done { image, blur }) => {
                            // Non-JXL formats only learn dimensions here.
                            if st.img_w == 0.0 {
                                st.img_w = image.width.as_f32();
                                st.img_h = image.height.as_f32();
                                st.zoom = ZoomMode::FitAll;
                                st.pan = Vector2::ZERO;
                                st.target_pan = Vector2::ZERO;
                                st.view_scale = Some((win_w / st.img_w).min(win_h / st.img_h));
                            }
                            show_frame(
                                &mut rl,
                                &thread,
                                &mut st.view_tex,
                                &image.data,
                                image.width,
                                image.height,
                                st.pixelated,
                            )?;
                            attach_blur_bg(
                                &mut st.blur_bg,
                                &mut rl,
                                &thread,
                                blur.as_ref(),
                                st.open_id,
                            );
                            st.view_loading = false;
                        }
                        Err(err) => {
                            eprintln!("vv: {err}");
                            open_failed = true;
                        }
                    }
                }
            }
            if open_failed {
                st.loader = None; // cancel the worker
                st.close_crop();
                // Keep the entry: mark it as failed, so its cell stays
                // visible (dimmed, error glyph) instead of vanishing
                // silently.
                if let Some(id) = st.open_id.take() {
                    grid.as_mut()
                        .unwrap()
                        .mark_failed(id, DecodeFailure::Decode);
                }
                st.view_from_grid = None;
                st.mode = Mode::Grid;
                st.view_tex = None;
                st.view_loading = false;
            }
        }

        // True while a Shift key is down.
        // Shift + h/j/k/l moves at double speed in both views: the grid
        // selection step and the image pan and free zoom.
        // The flag lives outside the mode branches because both read it.
        let shift = rl.is_key_down(KeyboardKey::KEY_LEFT_SHIFT)
            || rl.is_key_down(KeyboardKey::KEY_RIGHT_SHIFT);
        // Shift + h/j/k/l moves at double speed in both views.
        // The grid passes the step count to its input handler, and the image
        // view scales its pan and zoom ramps.
        let shift_mult = if shift {
            versatile_viewer::SHIFT_SPEED_MULTIPLIER
        } else {
            1
        };
        let shift_speed = shift_mult as f32;

        if st.mode == Mode::Grid {
            // Grid navigation: h/j/k/l and the arrows move the selection,
            // Enter opens the selected image, q quits (ESC is inert here,
            // because the grid is the home view).
            match grid.as_mut().unwrap().handle_input(
                &mut rl,
                (WinW::new(win_w), WinH::new(win_h)),
                shift_mult,
            ) {
                Some(GridAction::Open(i)) => {
                    if debug {
                        eprintln!("vv: enter pressed -> open idx {i}");
                    }
                    show_entry(
                        &mut st,
                        grid.as_mut().unwrap(),
                        i,
                        &mut rl,
                        &thread,
                        (WinW::new(win_w), WinH::new(win_h)),
                        true,
                    );
                }
                Some(GridAction::Quit) => quit = true,
                None => {}
            }
        } else {
            // Image mode.
            // Drain the raw key and char queues every frame (same rationale
            // as grid.rs handle_input).
            // is_key_pressed misses a press+release pair that lands inside
            // one frame, and this is easy here while frames stall on texture
            // uploads or decode.
            // Leftover queue entries otherwise leak into grid mode and act
            // there (for example a missed Enter that reopens the
            // just-viewed image at once).
            // State queries (is_key_down panning, is_key_pressed W/E/=/-) are
            // unaffected: the queue is separate from the key snapshot.
            // Enter and ESC return to the grid when one exists (nsxiv-like:
            // Enter toggles between the grid and the open image).
            // q quits, and ESC never quits the program (inert in single-file
            // launches, where there is no grid to return to).
            // Space and Backspace switch to the next or previous image
            // (nsxiv-style nav, with n or Space for next and p or
            // Backspace for previous, while the arrows and h/j/k/l stay
            // panning).
            // The mouse side buttons navigate too: forward is next and back
            // is previous, like n and p.
            // Nav keys auto-repeat while held, at the X server rate
            // (xset r rate, from keyrepeat::settings).
            // g/G jump to the first or last image.
            let mut enter = false;
            let mut quit_pressed = false;
            let mut pixelated_toggle = false;
            let mut f_key = false;
            let mut r_key = false;
            // The Shift state at the `r` press edge picks the pivot.
            let mut r_pivot_edge = false;
            let mut nav_next_edge = false;
            let mut nav_prev_edge = false;
            let mut jump_first = false;
            let mut jump_last = false;
            // The key that went down this frame, if any.
            // A typed `+`/`-` pairs with it to learn which physical key holds
            // that character (see zoomkeys::TypedKeys).
            let mut pressed_key: Option<KeyboardKey> = None;
            while let Some(k) = rl.get_key_pressed() {
                pressed_key = Some(k);
                match k {
                    KeyboardKey::KEY_ESCAPE
                    | KeyboardKey::KEY_ENTER
                    | KeyboardKey::KEY_KP_ENTER => {
                        enter = true;
                    }
                    KeyboardKey::KEY_Q => quit_pressed = true,
                    KeyboardKey::KEY_A => pixelated_toggle = true,
                    KeyboardKey::KEY_F => f_key = true,
                    KeyboardKey::KEY_R => {
                        r_key = true;
                        r_pivot_edge = shift;
                    }
                    KeyboardKey::KEY_SPACE | KeyboardKey::KEY_N => nav_next_edge = true,
                    KeyboardKey::KEY_BACKSPACE | KeyboardKey::KEY_P => nav_prev_edge = true,
                    KeyboardKey::KEY_G => {
                        if shift {
                            jump_last = true;
                        } else {
                            jump_first = true;
                        }
                    }
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
            // `!pointer_captured && !f_capture` guard.
            // The same guard covers the `r` rotation hold, which grabs the
            // pointer the same way.
            let mut f_click: Option<Vector2> = None;
            if (f_key || r_key) && !pointer_captured && !f_capture && !r_capture {
                grab_pos = rl.get_mouse_position();
                drag_virtual = grab_pos;
                rl.disable_cursor();
                pointer_captured = true;
                if r_key {
                    // `r` turns the image. The hold adds to the angle the image
                    // shows now, so the press never resets the turn.
                    // A plain `r` turns around the image center, and Shift+R
                    // turns around the grabbed point, which stays on its window
                    // spot while the pointer stays captured.
                    r_capture = true;
                    st.pivot = r_pivot_edge.then_some(grab_pos);
                    if debug {
                        eprintln!(
                            "vv: r hold grab at ({:.0},{:.0}), angle {}, pivot {:?}",
                            grab_pos.x, grab_pos.y, st.angle, st.pivot
                        );
                    }
                } else {
                    f_capture = true;
                    f_zoom_capture = shift;
                    f_zoom_anchor = f_zoom_capture.then_some(grab_pos);
                    if debug {
                        eprintln!(
                            "vv: f hold grab at ({:.0},{:.0}), zoom {}",
                            grab_pos.x, grab_pos.y, f_zoom_capture
                        );
                    }
                }
            }
            // A Shift change mid-hold switches the pan and the zoom.
            // The capture stays, so the travel stays unbounded.
            if f_capture && shift != f_zoom_capture {
                f_zoom_capture = shift;
                // Forget the distance the tap test and the pan measured.
                grab_pos = rl.get_mouse_position();
                drag_virtual = grab_pos;
                // A zoom from here on anchors on the view center: the image
                // point the window shows at its middle.
                // The pointer is hidden, so its last screen spot says nothing
                // about what the user looks at, and a pan had just pushed that
                // spot away from the middle.
                // The anchor is the middle of the window, so the ease needs no
                // drift toward it.
                f_zoom_anchor = f_zoom_capture.then_some(Vector2 {
                    x: win_w / 2.0,
                    y: win_h / 2.0,
                });
                // A zoom hold never counts as a tap.
                f_zoom_tapped = true;
                if debug {
                    eprintln!("vv: f hold switch, zoom {f_zoom_capture}");
                }
            }
            // `!is_key_down` also covers a press+release landing inside one
            // frame (the pressed queue records the press, but there is no
            // release edge to observe).
            // Without this the capture sticks.
            // The `r` release keeps the angle, so the image stays turned.
            if (f_capture || r_capture)
                && !rl.is_key_down(KeyboardKey::KEY_F)
                && !rl.is_key_down(KeyboardKey::KEY_R)
            {
                // Fold in this frame's motion before the lock is released
                // (the drag block below no longer runs once capture ends).
                let delta = rl.get_mouse_delta();
                drag_virtual.x += delta.x;
                drag_virtual.y += delta.y;
                let f_dragged = (drag_virtual - grab_pos).length() > 4.0;
                if pointer_captured {
                    rl.enable_cursor();
                    rl.set_mouse_position(grab_pos);
                    pending_restore = Some(grab_pos);
                    restore_tries = 0;
                    pointer_captured = false;
                }
                // A zoom hold has no tap action: it would otherwise run the
                // `f` fill aim on release.
                if f_capture && !f_dragged && !f_zoom_capture && !f_zoom_tapped {
                    f_click = Some(grab_pos);
                }
                if f_capture {
                    f_capture = false;
                    f_zoom_capture = false;
                    f_zoom_tapped = false;
                    f_zoom_anchor = None;
                }
                if r_capture {
                    // The release snaps a near quarter turn exactly, and it is
                    // the only place that snaps: the drag itself turns freely.
                    // A pivot turn keeps its point on screen through the snap.
                    let before = st.angle;
                    st.angle = rotate::snap_deg(st.angle);
                    keep_pivot(&mut st, before, win_w, win_h);
                    r_capture = false;
                }
                if debug {
                    eprintln!(
                        "vv: hold release: grab ({:.0},{:.0}), virtual ({:.0},{:.0}), dragged {}, \
                         angle {}",
                        grab_pos.x, grab_pos.y, drag_virtual.x, drag_virtual.y, f_dragged, st.angle
                    );
                }
            }
            // Auto-repeat: the initial press fires at once (edge from the
            // queue above), and the hold fires at the X server repeat rate
            // after its delay.
            // The mouse side buttons ride the same nav path: forward is next
            // and back is previous, exactly like n and p.
            let now = rl.get_time();
            let (delay, rate) = keyrepeat::settings();
            let mouse_nav = read_nav_mouse(&rl);
            let next_edge = nav_next_edge || mouse_nav.next.edge;
            let prev_edge = nav_prev_edge || mouse_nav.prev.edge;
            let next_held = rl.is_key_down(KeyboardKey::KEY_N)
                || rl.is_key_down(KeyboardKey::KEY_SPACE)
                || mouse_nav.next.held;
            let prev_held = rl.is_key_down(KeyboardKey::KEY_P)
                || rl.is_key_down(KeyboardKey::KEY_BACKSPACE)
                || mouse_nav.prev.held;
            let mut nav = if rep_nav_fwd.tick(next_edge, next_held, now, delay, rate) {
                Some(1)
            } else if rep_nav_back.tick(prev_edge, prev_held, now, delay, rate) {
                Some(-1)
            } else {
                None
            };
            // g/G: jump to the first or last image (as a nav delta).
            if (jump_first || jump_last)
                && let Some(cur) = st
                    .open_id
                    .and_then(|id| grid.as_ref().and_then(|g| g.index_of(id)))
            {
                let n = grid.as_ref().unwrap().entries.len();
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
            while let Some(c) = rl.get_char_pressed() {
                match c {
                    '\n' | '\r' => enter = true,
                    '+' => zoom_keys.typed_zoom_in(pressed_key),
                    '-' => zoom_keys.typed_zoom_out(pressed_key),
                    _ => {}
                }
            }

            let mut return_to_grid = false;
            if quit_pressed {
                quit = true; // single-file launch: no grid to fall back to
            }
            // `a`: toggle smooth (bilinear, default) against pixelated
            // (nearest-neighbor) filtering for the shown texture.
            // The code re-applies the flag to every texture that arrives
            // later.
            if pixelated_toggle {
                st.pixelated = !st.pixelated;
                if let Some(tex) = &st.view_tex {
                    apply_view_filter(&thread, tex, st.pixelated);
                }
                if let Some(crop) = &st.crop {
                    apply_view_filter(&thread, &crop.texture, st.pixelated);
                }
            }
            // `f` tap action: it needs known geometry, and the hit test uses
            // the drawn image rect (offset + pan at the on-screen scale).
            // A cursor over the background returns to the grid (ESC-like).
            // A cursor over an image that covers the whole window zooms out
            // (recentred, like the zoom-out half of t).
            // Otherwise it zooms into fill view aimed at the cursor (like the
            // zoom-in half of t).
            // It fires on the key release, and only when the hold did not pan.
            let mut f_open_grid = false;
            let mut f_zoom = false;
            if let Some(m) = f_click
                && st.img_w > 0.0
                && let Some(scale) = st.view_scale
            {
                let center = Vector2::new(win_w / 2.0 + st.pan.x, win_h / 2.0 + st.pan.y);
                // A turned image must hit-test its own outline, so the point
                // goes back into the unrotated image space first.
                let m = rotate::unrotate_point(m, center, st.angle);
                let ox = (win_w - st.img_w * scale) / 2.0 + st.pan.x;
                let oy = (win_h - st.img_h * scale) / 2.0 + st.pan.y;
                let inside = m.x >= ox
                    && m.x < ox + st.img_w * scale
                    && m.y >= oy
                    && m.y < oy + st.img_h * scale;
                let filled = ox <= 0.0
                    && oy <= 0.0
                    && ox + st.img_w * scale >= win_w
                    && oy + st.img_h * scale >= win_h;
                if !inside {
                    f_open_grid = grid.is_some();
                } else if filled {
                    st.zoom = ZoomMode::FitAll;
                    st.target_pan = Vector2::ZERO;
                } else {
                    f_zoom = true;
                }
            }
            if f_open_grid {
                enter = true;
            }
            if grid.is_some() && enter {
                return_to_grid = true;
            }
            if return_to_grid {
                // Leaving image view mid-drag: give the pointer back before
                // the grid click handling needs a visible cursor.
                if pointer_captured {
                    rl.enable_cursor();
                    rl.set_mouse_position(grab_pos);
                    pending_restore = Some(grab_pos);
                    restore_tries = 0;
                    pointer_captured = false;
                }
                f_capture = false;
                r_capture = false;
                st.mode = Mode::Grid;
                // Hand the shown texture back to its grid entry and make
                // that entry the grid selection (nsxiv-like).
                put_back_view(&mut grid, &mut st.view_from_grid, &mut st.view_tex);
                if let Some(id) = st.open_id.take()
                    && let Some(g) = grid.as_mut()
                    && let Some(i) = g.index_of(id)
                {
                    g.selected = i;
                    // Scrolled (zoomed-in) grids: bring it back on screen.
                    g.ensure_visible(WinW::new(win_w), WinH::new(win_h));
                }
                st.loader = None; // cancels a still-running stream
                st.view_loading = false;
                st.close_crop();
            }

            // Prev/next while viewing (only with a grid to navigate).
            // A ready texture swaps in the same frame.
            // A decode in flight is waited on (auto-swap when it lands).
            // Otherwise the streaming loader takes over.
            // The priority list at the top of the loop prefetches the
            // neighbors of the new entry.
            if !return_to_grid && let Some(delta) = nav {
                let cur = st
                    .open_id
                    .and_then(|id| grid.as_ref().and_then(|g| g.index_of(id)));
                if let Some(cur) = cur {
                    let n = grid.as_ref().unwrap().entries.len();
                    let t = cur as i64 + delta;
                    if t >= 0 && (t as usize) < n {
                        let j = t as usize;
                        put_back_view(&mut grid, &mut st.view_from_grid, &mut st.view_tex);
                        show_entry(
                            &mut st,
                            grid.as_mut().unwrap(),
                            j,
                            &mut rl,
                            &thread,
                            (WinW::new(win_w), WinH::new(win_h)),
                            true,
                        );
                    }
                }
            }

            // Keyboard shortcuts. Capital W and capital E arrive as W/E plus
            // shift.
            if rl.is_key_pressed(KeyboardKey::KEY_W) {
                st.zoom = if shift {
                    ZoomMode::FitAll
                } else {
                    ZoomMode::FitDown
                };
                st.target_pan = Vector2::ZERO;
            } else if rl.is_key_pressed(KeyboardKey::KEY_E) {
                st.zoom = if shift {
                    ZoomMode::FitHeight
                } else {
                    ZoomMode::FitWidth
                };
                st.target_pan = Vector2::ZERO;
            } else if rl.is_key_pressed(KeyboardKey::KEY_T) || f_zoom {
                // t: two-state toggle, whole image visible (fit-all) against
                // window completely covered (fill).
                // This stays distinct for any image and window combination,
                // unlike a fit-width/fit-height cycle.
                // f (f_zoom) always zooms in: it only reaches here when the
                // cursor is on a not-yet-covering image.
                if f_zoom {
                    st.zoom = ZoomMode::Fill;
                } else if st.zoom == ZoomMode::Fill {
                    st.zoom = ZoomMode::FitAll;
                } else {
                    st.zoom = ZoomMode::Fill;
                }
                // Zoom-in lands with the image point under the mouse at the
                // window center, clamped so the window never shows
                // background.
                // Zoom-out recentres.
                if st.zoom == ZoomMode::Fill && st.img_w > 0.0 && st.view_scale.is_some() {
                    st.target_pan = aim_fill_pan(&st, rl.get_mouse_position(), win_w, win_h);
                } else {
                    st.target_pan = Vector2::ZERO;
                }
            }

            // Before the header arrives the image dimensions are unknown.
            // The code skips all scale and pan math (it divides by
            // img_w/img_h).
            if st.img_w > 0.0 {
                // Scale for the current mode (fit modes recompute every
                // frame, so resizing stays correct).
                let fit_scale = match st.zoom {
                    ZoomMode::FitDown => (win_w / st.img_w).min(win_h / st.img_h).min(1.0),
                    ZoomMode::FitAll => (win_w / st.img_w).min(win_h / st.img_h),
                    ZoomMode::FitWidth => win_w / st.img_w,
                    ZoomMode::FitHeight => win_h / st.img_h,
                    // Cover: the larger ratio wins, so the image fills the
                    // window and the code crops the other axis.
                    // A free zoom keeps its scale as the fit base.
                    ZoomMode::Fill | ZoomMode::Free(_) => (win_w / st.img_w).max(win_h / st.img_h),
                };

                // Held keys feed the ramps.
                // The ramps make a hold ease in to cruise speed and a release
                // ease out, so the motion never steps on the auto-repeat
                // rate.
                // The code reads input before begin_drawing borrows rl
                // mutably.
                let dt = rl.get_frame_time();
                // Ctrl + j/k zoom like `+`/`-` (Ctrl + Shift + j/k keeps
                // the Shift speed).
                // `j` and `k` also pan, so a held Ctrl turns them into zoom
                // keys instead of pan keys.
                let ctrl = rl.is_key_down(KeyboardKey::KEY_LEFT_CONTROL)
                    || rl.is_key_down(KeyboardKey::KEY_RIGHT_CONTROL);
                let ctrl_zoom_in = ctrl && rl.is_key_down(KeyboardKey::KEY_J);
                let ctrl_zoom_out = ctrl && rl.is_key_down(KeyboardKey::KEY_K);
                let nav = NavKeys {
                    left: rl.is_key_down(KeyboardKey::KEY_H)
                        || rl.is_key_down(KeyboardKey::KEY_LEFT),
                    right: rl.is_key_down(KeyboardKey::KEY_L)
                        || rl.is_key_down(KeyboardKey::KEY_RIGHT),
                    up: (rl.is_key_down(KeyboardKey::KEY_K) && !ctrl_zoom_out)
                        || rl.is_key_down(KeyboardKey::KEY_UP),
                    down: (rl.is_key_down(KeyboardKey::KEY_J) && !ctrl_zoom_in)
                        || rl.is_key_down(KeyboardKey::KEY_DOWN),
                    // US layouts put the zoom keys on the `=` and `-`
                    // keycodes (numpad included).
                    // Other layouts put `+`/`-` on another key, whose keycode
                    // the typed character taught earlier.
                    // Polling that keycode reads the real held state, so a
                    // second key going down does not stop the zoom.
                    // Ctrl + j/k join them as a hold, so the ramp eases the
                    // same way.
                    zoom_in: rl.is_key_down(KeyboardKey::KEY_EQUAL)
                        || rl.is_key_down(KeyboardKey::KEY_KP_ADD)
                        || ctrl_zoom_in
                        || zoom_keys.zoom_in_key().is_some_and(|k| rl.is_key_down(k)),
                    zoom_out: rl.is_key_down(KeyboardKey::KEY_MINUS)
                        || rl.is_key_down(KeyboardKey::KEY_KP_SUBTRACT)
                        || ctrl_zoom_out
                        || zoom_keys.zoom_out_key().is_some_and(|k| rl.is_key_down(k)),
                };
                // The pan cruise speed scales with the window, so the motion
                // covers the same fraction of the view on any screen size.
                // Shift multiplies both the pan cruise speed and the zoom
                // rate.
                let motion = st
                    .ramps
                    .tick(nav, win_w.max(win_h) * PAN_RATE, dt, shift_speed);
                // Fold the ramped delta into the pan and its target together,
                // so the pan follows the ramp at once and the ease below sees
                // no leftover distance.
                st.pan += motion.pan;
                st.target_pan += motion.pan;
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
                let delta = rl.get_mouse_delta();
                let zoom_drag = f_capture && f_zoom_capture;
                if zoom_drag {
                    // The grab edge seeds the anchor, and the drift carries it
                    // from frame to frame.
                    st.zoom_anchor = Some(*f_zoom_anchor.get_or_insert(grab_pos));
                    if delta.y != 0.0 {
                        // One e-fold of scale per ZOOM_DRAG_SPAN of travel,
                        // where the span scales with the window.
                        let span = win_w.max(win_h) * ZOOM_DRAG_SPAN;
                        let r = (-delta.y * ZOOM_DRAG_SPEED / span).exp();
                        let base = st.zoom.free_or(fit_scale);
                        st.zoom = ZoomMode::Free((base * r).clamp(0.01, 100.0));
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
                if r_capture {
                    // One full turn per ROTATE_DRAG_SPAN of travel, where the
                    // span scales with the window.
                    let span = win_w.max(win_h) * ROTATE_DRAG_SPAN;
                    let from = st.angle;
                    st.angle = rotate::apply_turn(st.angle, delta.y * ROTATE_DRAG_SPEED, span);
                    // A pivot turns the image in place: the pan takes the shift
                    // that the plain centered turn would move the pivot by, so
                    // the point under the pivot keeps its window spot.
                    keep_pivot(&mut st, from, win_w, win_h);
                }

                // Free zoom: the held +/- keys scale the free-zoom factor
                // itself, with a ramped speed, so a hold zooms at a steady
                // eased rate and a release coasts to a stop.
                // The fit modes keep their own scale.
                // The on-screen scale takes the ramped factor directly: an
                // exponential ease on top would lag a target that moves every
                // frame, and that lag reads as stepping.
                let held_zoom = motion.zoom_moving;
                if held_zoom {
                    let base = st.zoom.free_or(fit_scale);
                    st.zoom = ZoomMode::Free((base * motion.zoom_factor).clamp(0.01, 100.0));
                    // Keyboard zoom centers on the picture, not the cursor.
                    // None falls back to the window center anchor.
                    st.zoom_anchor = None;
                }
                // The Shift+F drag rides the pointer, so its target moves
                // every frame: the ease must stay out of its way, like the
                // wheel step.
                let target_scale = st.zoom.free_or(fit_scale);

                // The exponential ease only carries a discrete jump (a fit
                // mode change, the wheel, the `f` aim) the rest of the way.
                // A held key already set the target to the on-screen value, so
                // the ease stays out of its way.
                let prev_scale = st.view_scale;
                let alpha = 1.0 - (-dt / EASE_TIME).exp();
                st.view_scale = Some(match st.view_scale {
                    None => target_scale,
                    Some(_) if resized => target_scale,
                    Some(_) if held_zoom => target_scale,
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
                let wheel = rl.get_mouse_wheel_move();
                if wheel != 0.0 {
                    let factor = 1.25f32.powf(wheel);
                    st.zoom = ZoomMode::Free((target_scale * factor).clamp(0.01, 100.0));
                    st.zoom_anchor = Some(rl.get_mouse_position());
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
                let drag_pressed = rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT);
                let drag_released = rl.is_mouse_button_released(MouseButton::MOUSE_BUTTON_LEFT);
                if drag_pressed && !pointer_captured && !r_capture {
                    grab_pos = rl.get_mouse_position();
                    drag_virtual = grab_pos;
                    rl.disable_cursor();
                    pointer_captured = true;
                    if debug {
                        eprintln!("vv: drag grab at ({:.0},{:.0})", grab_pos.x, grab_pos.y);
                    }
                } else if drag_released && pointer_captured && !f_capture && !r_capture {
                    if debug {
                        eprintln!(
                            "vv: drag release: grab ({:.0},{:.0}), virtual ({:.0},{:.0}), rl pos \
                             ({:.0},{:.0})",
                            grab_pos.x,
                            grab_pos.y,
                            drag_virtual.x,
                            drag_virtual.y,
                            rl.get_mouse_position().x,
                            rl.get_mouse_position().y
                        );
                    }
                    rl.enable_cursor();
                    rl.set_mouse_position(grab_pos);
                    pending_restore = Some(grab_pos);
                    restore_tries = 0;
                    pointer_captured = false;
                }
                // Any captured pointer (left-drag or a held `f`) pans or zooms.
                // The virtual cursor tracks the raw travel the same way.
                // A Shift+F capture zooms instead: it tracks the travel and
                // leaves the pan alone.
                // An `r` capture turns the image instead: the travel it needs
                // is vertical, so the pan stays put.
                let dragging = pointer_captured;
                if dragging {
                    let delta = rl.get_mouse_delta();
                    // The virtual cursor follows physical travel 1:1 (debug
                    // crosshair).
                    drag_virtual.x += delta.x;
                    drag_virtual.y += delta.y;
                    // The pan gets the 2x speedup.
                    if !f_zoom_capture && !r_capture {
                        st.target_pan.x += delta.x * 2.0;
                        st.target_pan.y += delta.y * 2.0;
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
                let continuous_zoom = f_capture && f_zoom_capture;
                let scale = st.view_scale.unwrap();
                if matches!(st.zoom, ZoomMode::Free(_))
                    && let Some(s_old) = prev_scale
                    && (scale - s_old).abs() > f32::EPSILON
                    && s_old > 0.0
                {
                    let center = Vector2 {
                        x: win_w / 2.0,
                        y: win_h / 2.0,
                    };
                    let anchor = st.zoom_anchor;
                    let a = anchor.unwrap_or(center);
                    let (ax, ay) = (a.x, a.y);
                    let keyboard_zoom = anchor.is_none();
                    let r = scale / s_old;
                    let ox = ax - (ax - (win_w - st.img_w * s_old) / 2.0 - st.pan.x) * r;
                    let oy = ay - (ay - (win_h - st.img_h * s_old) / 2.0 - st.pan.y) * r;
                    let new_x = ox - (win_w - st.img_w * scale) / 2.0;
                    let new_y = oy - (win_h - st.img_h * scale) / 2.0;
                    // A held key zoom skips the drift: it centers anyway.
                    // So does an anchor that already sits at the window
                    // center: the drift can only move it away from there.
                    let centered = a == center;
                    let d = if held_zoom || centered {
                        Vector2::ZERO
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
                    let dx = new_x + d.x - st.pan.x;
                    let dy = new_y + d.y - st.pan.y;
                    st.pan.x += dx;
                    st.pan.y += dy;
                    st.target_pan.x += dx;
                    st.target_pan.y += dy;
                    if !keyboard_zoom {
                        st.zoom_anchor = Some(a + d);
                        if continuous_zoom {
                            // The drag anchor carries the drift into the next
                            // frame.
                            f_zoom_anchor = st.zoom_anchor;
                        }
                    }
                    // Ride the pointer along with the drifted image point
                    // (never while a drag has it captured, and keyboard zoom
                    // leaves the cursor wherever it is).
                    // A continuous zoom keeps the pointer captured instead,
                    // so this never runs for it.
                    if d != Vector2::ZERO && !keyboard_zoom && !pointer_captured {
                        if debug {
                            eprintln!("vv: zoom drift anchor to ({ax:.0},{ay:.0})");
                        }
                        rl.set_mouse_position(st.zoom_anchor.unwrap());
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
                // On a resize frame, snap pan too, so it does not glide after
                // the new fit offset (matches the scale snap above).
                if resized {
                    st.pan = st.target_pan;
                }
                // While dragging, the image sticks to the cursor (only the
                // zoom-anchor shift above touches pan).
                if dragging {
                    st.pan = st.target_pan;
                }
                let pan_alpha = 1.0 - (-dt / EASE_TIME).exp();
                st.pan.x += (st.target_pan.x - st.pan.x) * pan_alpha;
                st.pan.y += (st.target_pan.y - st.pan.y) * pan_alpha;
                // Snap when the residual glide is sub-pixel.
                if (st.target_pan.x - st.pan.x).abs() < 0.25 {
                    st.pan.x = st.target_pan.x;
                }
                if (st.target_pan.y - st.pan.y).abs() < 0.25 {
                    st.pan.y = st.target_pan.y;
                }
            } // st.img_w > 0.0: fit/pan math needs known dimensions
        }

        // The visible-region patch of a vector file: adopt a parsed source and
        // a finished paint, then repaint once the view holds still.
        // The upload needs the main thread, and the request reads the scale
        // and the pan of this frame.
        if st.mode == Mode::Image {
            st.drain_crops(&mut rl, &thread)?;
            st.tick_crop(Instant::now(), Vector2::new(win_w, win_h));
        }

        // VV_BLUR_BG in grid mode: the background follows the selected
        // entry with a slow crossfade (starts as soon as the grid workers
        // decode the blurred copy of the entry).
        if st.mode == Mode::Grid
            && let (Some(bg), Some(g)) = (st.blur_bg.as_mut(), grid.as_ref())
            && let Some(e) = g.entries.get(g.selected)
            && let Some(data) = &e.blur
        {
            bg.transition(&mut rl, &thread, data, e.id);
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(BACKGROUND);

        if st.mode == Mode::Grid {
            // VV_BLUR_BG gimmick: the blurred background follows the
            // selection (crossfading, and a no-op when the gimmick is off).
            // The code draws it before the grid, so the fades run under the
            // thumbnails.
            if let Some(bg) = &mut st.blur_bg {
                bg.draw(&mut d, WinW::new(win_w), WinH::new(win_h));
            }
            grid.as_ref()
                .unwrap()
                .draw(&mut d, (WinW::new(win_w), WinH::new(win_h)));
        } else {
            // VV_BLUR_BG gimmick: blurred copy of the image, scaled to
            // cover the whole window (fit on the narrower side, with the other
            // axis overflowing and cropped) behind the sharp image.
            if let Some(bg) = &mut st.blur_bg {
                bg.draw(&mut d, WinW::new(win_w), WinH::new(win_h));
            }
            // The region patch covers the window on its own with the sharp
            // pixels of the visible part, so the whole-image texture draws only
            // where the patch cannot reach: the frames of a resize or a pan
            // that outran the repaint.
            // Both over the same pixels would double the alpha of an
            // antialiased edge.
            if !patch_covers(&st, win_w, win_h) {
                if let Some(texture) = &st.view_tex {
                    // The image turns around its own center on screen, which
                    // is the pan offset on top of the window middle.
                    let center = Vector2::new(win_w / 2.0 + st.pan.x, win_h / 2.0 + st.pan.y);
                    draw_turned(
                        &mut d,
                        texture,
                        st.img_w,
                        st.img_h,
                        st.view_scale.unwrap(),
                        center,
                        st.angle,
                    );
                } else if st.img_w > 0.0
                    && let Some(g) = grid.as_ref()
                    && let Some(id) = st.open_id
                    && let Some(i) = g.index_of(id)
                    && let Some(tex) = g
                        .entries
                        .get(i)
                        // The decoded thumb first, then its few-byte
                        // placeholder.
                        .and_then(|e| e.texture.as_ref().or(e.placeholder.as_ref()))
                {
                    // No full-res frame yet (decode in flight): show the entry
                    // thumb instead of flashing black, or its ThumbHash
                    // placeholder while the thumb itself still decodes.
                    // Both hold the whole image with the aspect preserved, so
                    // they map 1:1 onto the image rect and the later swap to
                    // the full-res texture is pixel-aligned.
                    let scale = st
                        .view_scale
                        .unwrap_or((win_w / st.img_w).min(win_h / st.img_h));
                    let center = Vector2::new(win_w / 2.0 + st.pan.x, win_h / 2.0 + st.pan.y);
                    draw_turned(&mut d, tex, st.img_w, st.img_h, scale, center, st.angle);
                }
            }
            // The region patch draws turned and scaled like the image, so its
            // pixels land where the whole-image texture would put them.
            if let Some(crop) = &st.crop
                && st.img_w > 0.0
                && let Some(scale) = st.view_scale
            {
                let img = Vector2::new(st.img_w, st.img_h);
                let win = Vector2::new(win_w, win_h);
                let rect = crop.rect;
                let middle = Vector2::new(rect.x + rect.w / 2.0, rect.y + rect.h / 2.0);
                let center = region::to_screen(middle, img, win, st.pan, scale, st.angle);
                // The dest size follows the current view scale, so the patch
                // stretches by at most `region::SCALE_SLACK`.
                draw_turned(
                    &mut d,
                    &crop.texture,
                    rect.w,
                    rect.h,
                    scale,
                    center,
                    st.angle,
                );
            }
            // VV_DEBUG: while the pointer is captured, draw a crosshair at
            // the virtual cursor position (clamped to the window).
            // The real cursor is hidden, and this shows where vv thinks it
            // is.
            // The code also draws a ring at the grab point, so a wrong
            // restore is visible.
            if debug && pointer_captured {
                let vx = drag_virtual.x.clamp(0.0, win_w);
                let vy = drag_virtual.y.clamp(0.0, win_h);
                d.draw_circle_v(Vector2::new(grab_pos.x, grab_pos.y), 6.0, Color::SKYBLUE);
                d.draw_line_v(
                    Vector2::new(vx - 12.0, vy),
                    Vector2::new(vx + 12.0, vy),
                    Color::YELLOW,
                );
                d.draw_line_v(
                    Vector2::new(vx, vy - 12.0),
                    Vector2::new(vx, vy + 12.0),
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
    }
    Ok(())
}
