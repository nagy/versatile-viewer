// Pixel/coordinate math lives in f32 (raylib's units) and indices in
// usize/u32. The casts between them are inherent to that boundary, and
// every value here (window pixels, texture dimensions) is far below
// f32's exact-integer range, so the pedantic cast lints are noise.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    // win_w/win_h and friends are natural paired names in window math.
    clippy::similar_names
)]

//! versatile-viewer — image viewer (JXL first-class, plus PNG/JPEG) with a
//! directory thumbnail grid. q quits; ESC/Enter toggle grid ↔ image view.
//!
//! Usage: versatile-viewer <image-path-or-directory>
//!
//! Env gimmicks: `VV_DEBUG`=1 traces input events; `VV_SLOW_STREAM`=1 slows the
//! JXL stream; `VV_BLUR_BG`=1 draws a blurred copy of the viewed image as the
//! image-view background (scaled to cover the window, GPU-upscaled).
//! `VV_BG_DIM`=0..1 sets its brightness (default 0.6); `VV_BLUR_PX` sets the
//! blur resolution — the tiny texture's long side, default 128, fewer =
//! blurrier.

use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::mpsc::{Receiver, channel},
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
    DecodedImage,
    blurbg::{self, BlurBg},
    document::{PageInfo, has_page_grid, open_document},
    grid::{EntrySource, Grid, GridAction},
    keyrepeat,
    loader::{Loader, LoaderMsg},
    upload_rgba,
};

/// Fraction of its remaining distance to the window center that the
/// zoomed image point (and the cursor riding it) drifts over one zoom
/// ease; shares the ease's exponential time constant.
const ZOOM_ANCHOR_DRIFT: f32 = 0.25;

/// Target pan that centers the fill view on the image point under `mouse`.
///
/// Both axes are clamped to `[-|offset|, |offset|]`: outside that interval a
/// window edge would expose background, so a cursor near the image edge just
/// pans as far as coverage allows.
fn aim_fill_pan(st: &ViewState, mouse: Vector2, win_w: f32, win_h: f32) -> Vector2 {
    let target = (win_w / st.img_w).max(win_h / st.img_h);
    let cur = st.view_scale.unwrap_or(target);
    // Mouse position in image coordinates at the current scale (offset =
    // window top-left of the unpanned image).
    let img_x = (mouse.x - (win_w - st.img_w * cur) / 2.0 - st.pan.x) / cur;
    let img_y = (mouse.y - (win_h - st.img_h * cur) / 2.0 - st.pan.y) / cur;
    let off_x = (win_w - st.img_w * target) / 2.0;
    let off_y = (win_h - st.img_h * target) / 2.0;
    Vector2 {
        x: (win_w / 2.0 - off_x - img_x * target).clamp(-off_x.abs(), off_x.abs()),
        y: (win_h / 2.0 - off_y - img_y * target).clamp(-off_y.abs(), off_y.abs()),
    }
}

/// How the image is scaled to the window. Scale is recomputed every frame,
/// so resizing always stays correct.
#[derive(Clone, Copy, PartialEq)]
enum ZoomMode {
    /// Fit down to the window, centered; never upscaled (default).
    FitDown,
    /// Fit all sides: scale up or down until the image first touches a
    /// border (Shift+W).
    FitAll,
    /// Fit to the window width (e).
    FitWidth,
    /// Fit to the window height (Shift+E).
    FitHeight,
    /// Fill the window: scale until the image covers every side (t);
    /// the shorter relative dimension overflows and is cropped.
    Fill,
    /// Free zoom factor, set with +/- (multiples of the last fit scale).
    Free(f32),
}

/// Viewer screen: thumbnail grid or single image.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Grid,
    Image,
}

/// Result of one in-flight refine render, matched to the live view by
/// (entry id, page, scale).
type RefineResult = (u64, usize, f32, Result<DecodedImage, String>);

/// Image-view state: what is shown and how it is framed.
struct ViewState {
    mode: Mode,
    /// Grid entry open in image mode, by stable id (None on a single-file
    /// launch).
    open_id: Option<u64>,
    /// Id of the grid entry whose full-res texture the view currently
    /// holds; it goes back when leaving image mode or switching entries.
    view_from_grid: Option<u64>,
    /// Texture currently shown in image mode.
    view_tex: Option<Texture2D>,
    view_loading: bool,
    /// When the current load started (`rl.get_time`()); the "decoding..."
    /// indicator only appears once the load exceeds 1 s.
    view_loading_since: f64,
    /// Full-resolution image dimensions (0 until known).
    img_w: f32,
    img_h: f32,
    zoom: ZoomMode,
    /// On-screen pan offset, eased toward `target_pan` each frame.
    pan: Vector2,
    target_pan: Vector2,
    /// On-screen scale, eased toward the target scale each frame.
    view_scale: Option<f32>,
    /// Window-space point the running zoom eases around: captured from the
    /// mouse position when a zoom step starts (+/- or wheel). None falls back
    /// to the window center (e.g. easing still running from an earlier step).
    zoom_anchor: Option<Vector2>,
    /// `a` toggles nearest-neighbor filtering (pixelated) for pixel
    /// peeping; default smooth (bilinear).
    pixelated: bool,
    /// Streaming loader for the open image; drop cancels the worker.
    loader: Option<Loader>,
    /// Texture pixels per natural unit (points for PDF pages, pixels for
    /// images) of the currently shown texture. Images render once at native
    /// size (1.0); document pages re-render at the settled zoom (refine).
    view_render_scale: f32,
    /// In-flight refine render: (entry id, page, scale, result). The result
    /// is accepted only if it still matches the live view (same entry, page
    /// and zoom) — the user may have zoomed or navigated away meanwhile.
    refine: Option<Receiver<RefineResult>>,
    /// `VV_BLUR_BG` background (None when the gimmick is off). Declared after
    /// `rl` (via `ViewState`) so it drops and unloads before the window.
    blur_bg: Option<BlurBg>,
}

impl ViewState {
    /// Reset fit/pan state for a freshly shown image.
    fn reset_view(&mut self) {
        // Freshly shown images open fit-all (Shift+W behavior): upscale or
        // downscale until the image first touches a window border.
        self.zoom = ZoomMode::FitAll;
        self.pan = Vector2::ZERO;
        self.target_pan = Vector2::ZERO;
        self.zoom_anchor = None;
        self.view_scale = None;
    }
}

/// Show a new RGBA8 frame in image view. When a texture with the same
/// dimensions already exists (progressive previews of the same image), its
/// pixels are updated in place instead of reallocating a GPU texture.
fn show_frame(
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    view_tex: &mut Option<Texture2D>,
    rgba: &[u8],
    width: u32,
    height: u32,
    pixelated: bool,
) -> Result<()> {
    match view_tex {
        Some(tex) if tex.width() == width as i32 && tex.height() == height as i32 => {
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

/// Texture filter for the image view: bilinear by default (good for
/// photos), `a` toggles nearest-neighbor (pixelated) for 1:1 pixel
/// peeping. Grid thumbs always stay bilinear (heavily downscaled;
/// nearest would alias badly).
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

/// Upload a tiny blurred copy as the image-view background (`VV_BLUR_BG`
/// gimmick). `tag` identifies the source image (grid entry id; None for a
/// single-file launch) so grid crossfades can skip redundant transitions.
/// No-op when the gimmick is off or the copy is missing.
fn attach_blur_bg(
    blur_bg: &mut Option<BlurBg>,
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    blur: Option<&blurbg::BlurData>,
    tag: Option<u64>,
) {
    if let (Some(bg), Some(data)) = (blur_bg.as_mut(), blur) {
        bg.attach(rl, thread, data.clone(), tag);
    }
}

/// Put the viewed texture back into its grid entry (if it came from one)
/// and clear the view texture. Called when leaving image mode or switching
/// to another grid entry.
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

/// Fit-all scale for a page of natural size `info` inside a window,
/// clamped so renders neither vanish nor explode (see pdf.rs caps).
fn fit_scale(info: PageInfo, win_w: f32, win_h: f32) -> f32 {
    (win_w / info.width as f32)
        .min(win_h / info.height as f32)
        .clamp(0.01, 8.0)
}

/// Start the streaming loader for a grid entry source. Image files open as
/// one-page documents (render at native size); document pages render at the
/// fit scale (points: 1 pt = 1 px at scale 1.0, so the fit ratio doubles as
/// the render scale).
fn start_entry_loader(source: &EntrySource, preview_px: u32, win: (f32, f32)) -> Result<Loader> {
    match source {
        EntrySource::File(path) => Ok(Loader::start_doc(open_document(path)?, 0, 1.0, preview_px)),
        EntrySource::DocPage { doc, page } => {
            let fit = fit_scale(doc.page_info(*page)?, win.0, win.1);
            Ok(Loader::start_doc(doc.clone(), *page, fit, preview_px))
        }
    }
}

/// Make the entry at index `idx` the open image. Three paths, shared by
/// grid-open, prev/next navigation and the decode-landed takeover:
/// 1. its full-res texture was kept for it (keep-set): show instantly;
/// 2. a decode is in flight (`queued`): show "decoding" and wait for the
///    decode-landed takeover in a later frame;
/// 3. otherwise: start the streaming loader (JXL: blurry preview fast).
///
/// On paths 2 and 3 the draw loop shows the entry's thumb (whole image,
/// aspect preserved) as a placeholder, so n/p navigation never flashes
/// black while a decode catches up.
///
/// With `set_scale_now`, the initial fit scale is set immediately because
/// the caller's frame skips the per-frame scale math (it ran earlier).
fn show_entry(
    st: &mut ViewState,
    grid: &mut Grid,
    idx: usize,
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    win: (f32, f32),
    set_scale_now: bool,
) {
    let (id, tex, w, h, source, queued, thumb, blur) = {
        let e = &mut grid.entries[idx];
        (
            e.id,
            e.full.take(),
            e.width,
            e.height,
            e.source.clone(),
            e.queued,
            e.texture.is_some(),
            e.blur.clone(),
        )
    };
    st.open_id = Some(id);
    st.mode = Mode::Image;
    st.reset_view();
    st.view_render_scale = 1.0;
    st.refine = None;
    if let Some(tex) = tex {
        grid.entries[idx].viewing = true;
        st.view_from_grid = Some(id);
        apply_view_filter(thread, &tex, st.pixelated);
        st.view_tex = Some(tex);
        st.img_w = w as f32;
        st.img_h = h as f32;
        st.loader = None;
        st.view_loading = false;
        if set_scale_now {
            let (win_w, win_h) = win;
            st.view_scale = Some((win_w / st.img_w).min(win_h / st.img_h));
        }
        attach_blur_bg(&mut st.blur_bg, rl, thread, blur.as_ref(), st.open_id);
    } else {
        st.view_from_grid = None;
        st.view_tex = None;
        st.view_loading = true;
        st.view_loading_since = rl.get_time();
        // Whenever the entry's thumb exists, the decode that produced it
        // also recorded the full dimensions — set them now so the draw
        // loop can show the thumb as a placeholder at the final transform
        // (instead of a black frame) while the full-res decode catches up.
        if w > 0 {
            st.img_w = w as f32;
            st.img_h = h as f32;
            if set_scale_now {
                let (win_w, win_h) = win;
                st.view_scale = Some((win_w / st.img_w).min(win_h / st.img_h));
            }
        } else {
            st.img_w = 0.0; // dimensions arrive with the header/texture
            st.img_h = 0.0;
        }
        st.loader = if queued || thumb {
            // A grid decode for this entry is in flight, or the entry has
            // a thumb and the missing full-res decode was (or will be)
            // dispatched by the keep-set refill: wait for the
            // decode-landed takeover instead of decoding the file twice
            // in a streaming worker.
            None
        } else {
            let preview_px = rl.get_screen_width().max(rl.get_screen_height()) as u32;
            match start_entry_loader(&source, preview_px, win) {
                Ok(loader) => Some(loader),
                Err(err) => {
                    // Open failed (sniff/parse): keep the entry, dim it.
                    eprintln!("vv: {err:#}");
                    grid.mark_failed(id, format!("{err:#}"));
                    st.open_id = None;
                    st.mode = Mode::Grid;
                    st.view_loading = false;
                    None
                }
            }
        };
        attach_blur_bg(&mut st.blur_bg, rl, thread, blur.as_ref(), st.open_id);
    }
}

/// Display a path with `$HOME` collapsed to `~`.
fn tilde_path(path: &Path) -> String {
    if let Ok(home) = env::var("HOME") {
        // Skip the pathological HOME=/ case (everything would collapse).
        let home = home.trim_end_matches('/');
        if home.is_empty() || home == "/" {
            return path.display().to_string();
        }
        if let Ok(rest) = path.strip_prefix(home) {
            // strip_prefix yields a relative remainder ("pics"), so the
            // separator must be re-added; $HOME itself maps to plain "~".
            return if rest.as_os_str().is_empty() {
                "~".to_string()
            } else {
                format!("~/{rest}", rest = rest.display())
            };
        }
    }
    path.display().to_string()
}

/// Window title pieces shared by both launch modes.
const SEP: &str = " – ";
const APP: &str = "versatile-viewer";

/// Title for the image view of a single file: name, dir, app.
fn image_title(img_path: &Path) -> String {
    let abs = fs::canonicalize(img_path).unwrap_or_else(|_| img_path.to_path_buf());
    let file = abs.file_name().map_or_else(
        || abs.display().to_string(),
        |f| f.to_string_lossy().into_owned(),
    );
    let dir = abs.parent().unwrap_or(Path::new("/"));
    format!("{file}{SEP}{}{SEP}{APP}", tilde_path(dir))
}

/// Title for the grid over a directory with `n` images.
fn grid_title(dir_path: &Path, n: usize) -> String {
    format!(
        "({n} image{plural}){SEP}{}{SEP}{APP}",
        tilde_path(dir_path),
        plural = if n == 1 { "" } else { "s" },
    )
}

/// Initial window title for the launch mode: image view vs. grid view.
fn window_title(path: &Path, dir_grid: Option<&Grid>) -> String {
    // Make the path absolute (resolving `.` and symlinks) so the title is
    // meaningful regardless of the launch cwd.
    let abs: PathBuf = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    match dir_grid {
        Some(g) => grid_title(&abs, g.entries.len()),
        None => image_title(&abs),
    }
}

// The event loop is one long state machine by design; splitting it
// would scatter the frame-order invariants across call sites.
#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    let arg = env::args()
        .nth(1)
        .context("usage: versatile-viewer <image-path-or-directory>")?;
    let path = Path::new(&arg);

    // Directory launch: list the directory before opening the window (the
    // title shows the image count) — no GL involved yet. Single-file launch:
    // no grid.
    let dir_grid = if path.is_dir() {
        Some(Grid::from_dir(path)?)
    } else if path.is_file() {
        None
    } else {
        bail!("no such file or directory: {}", path.display());
    };

    // Single-file launch: open the document before the window so it can be
    // sized to page 0. Single-page documents decode up front (window =
    // image size); multi-page documents (PDF) size the window fitted to
    // page 0 and launch into the page grid. Directory launch: fixed size.
    let single_doc = if dir_grid.is_none() {
        Some(open_document(path)?)
    } else {
        None
    };
    let single_decoded = match &single_doc {
        Some(doc) if doc.page_count() == 1 => Some(doc.render(0, 1.0)?),
        _ => None,
    };
    let (win0_w, win0_h) = match (&single_decoded, &single_doc) {
        (Some(d), _) => (d.width as i32, d.height as i32),
        (None, Some(doc)) => {
            let info = doc.page_info(0)?;
            // Fit page 0 into ~1024x768 (never upscale past 1 pt = 2 px).
            let f = (1024.0 / info.width.max(1) as f32)
                .min(768.0 / info.height.max(1) as f32)
                .min(2.0);
            let w = (info.width as f32 * f).round() as i32;
            let h = (info.height as f32 * f).round() as i32;
            (w.max(320), h.max(240))
        }
        _ => (1024, 768),
    };
    // Blur-background source for a single-file launch: computed while the
    // RGBA buffer is still around (before the window/GL context exists);
    // uploaded once the window is open.
    let single_blur = if dir_grid.is_none() && blurbg::enabled() {
        let d = single_decoded.as_ref().unwrap();
        Some(blurbg::small_blur(
            &d.rgba,
            d.width,
            d.height,
            blurbg::blur_px(),
        ))
    } else {
        None
    };
    let (mut rl, thread) = raylib::init()
        .size(win0_w, win0_h)
        .title(&window_title(path, dir_grid.as_ref()))
        .resizable()
        // Vsync on: the compositor/driver paces us to the monitor's refresh
        // rate, eliminating tearing (most visible during the zoom ease).
        // This replaces set_target_fps below — a software 60 FPS cap would
        // fight a non-60 Hz monitor (judder) and add input latency.
        .vsync()
        .build();
    // WM_CLASS: raylib derives it from the creation title (full path —
    // unusable for window rules); stamp "vv" on X11, no-op on Wayland.
    #[cfg(target_os = "linux")]
    // SAFETY: handle only read; valid while the window is open.
    unsafe {
        wmclass::set_class(rl.get_window_handle())
    };
    // We quit via the q key handling ourselves (set_exit_key would make ESC
    // close the window outright instead of returning to the grid).
    rl.set_exit_key(None);
    // Explicit default-arrow cursor while the mouse hovers the window.
    rl.set_mouse_cursor(MouseCursor::MOUSE_CURSOR_ARROW);

    // VV_BLUR_BG gimmick: blurred copy of the viewed image behind it. Declared
    // after `rl` so it drops (and unloads its texture) before the window.
    // Take the grid AFTER the window exists: it holds GPU textures, and
    // being declared after `rl` it is dropped (and unloaded) BEFORE the
    // window closes — both on normal exit and on panic unwinding.
    let mut grid = dir_grid;
    // Active page grid stack: `home_grid` holds the directory grid while a
    // document's page-overview grid is open on top of it (one level deep).
    let mut home_grid: Option<Grid> = None;
    // Image-view state: what is shown and how it is framed. Its Loader field
    // drops (and cancels the worker) before the window closes, like `grid`.
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
        view_loading_since: 0.0,
        img_w: 0.0,
        img_h: 0.0,
        zoom: ZoomMode::FitDown,
        pan: Vector2::ZERO,
        target_pan: Vector2::ZERO,
        view_scale: None,
        zoom_anchor: None,
        pixelated: false,
        loader: None,
        view_render_scale: 1.0,
        refine: None,
        blur_bg: BlurBg::from_env(),
    };
    // A single-file multi-page document (PDF, .typ) launches into its page
    // grid; single-file images launch into the image view. `st.mode` was
    // decided above from the (then absent) grid — fix it here.
    if let Some(doc) = single_doc
        && doc.page_count() > 1
    {
        grid = Some(Grid::from_document(doc));
        st.mode = Mode::Grid;
    }
    if let Some(decoded) = single_decoded {
        st.view_tex = Some(upload_rgba(
            &mut rl,
            &thread,
            &decoded.rgba,
            decoded.width,
            decoded.height,
        )?);
        apply_view_filter(
            &thread,
            st.view_tex.as_ref().expect("just uploaded"),
            st.pixelated,
        );
        drop(decoded.rgba);
        st.img_w = win0_w as f32;
        st.img_h = win0_h as f32;
        if let Some(data) = single_blur {
            attach_blur_bg(&mut st.blur_bg, &mut rl, &thread, Some(&data), None);
        }
    }
    // A one-image directory skips the grid: jump straight into that image
    // (ESC still returns to the grid view).
    if grid.as_ref().is_some_and(|g| g.entries.len() == 1) {
        let win = (rl.get_screen_width() as f32, rl.get_screen_height() as f32);
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

    // Window title: grid mode / single-file launch keep the launch title;
    // image view rewrites it per open entry (reset when open_id changes).
    let launch_title = window_title(path, grid.as_ref());
    let mut title_open_id: Option<u64> = None;

    // Set when the viewer should exit entirely.
    let mut quit = false;
    // VV_DEBUG=1: trace grid open/return events to stderr.
    let debug = std::env::var_os("VV_DEBUG").is_some();
    // Previous per-key down state for the VV_DEBUG event trace.
    let mut prev_down = [false; 349];

    // No set_target_fps: vsync paces the frame loop (see the builder above).
    // Auto-repeat state for image-mode prev/next (xset r rate values).
    let mut rep_nav_fwd = keyrepeat::RepeatState::new();
    let mut rep_nav_back = keyrepeat::RepeatState::new();
    // Previous frame's window size: self-tracked resize detection (see the
    // `resized` computation in the loop).
    let mut last_win: Option<(f32, f32)> = None;
    // True while an image-view drag has the pointer captured (DisableCursor:
    // hidden + locked, unbounded virtual deltas — the drag cannot hit a
    // screen edge). Released on mouse-up; the pointer then reappears exactly
    // where the drag started (GLFW restores the pre-capture position).
    let mut pointer_captured = false;
    // Window position where the current (or last) drag grabbed the pointer;
    // raylib's EnableCursor does not reliably restore it, so we warp back
    // explicitly on release.
    let mut grab_pos = Vector2::ZERO;
    // Virtual cursor position while captured (grab_pos + accumulated raw
    // deltas; raylib's own virtual position is unreliable to interpret).
    let mut drag_virtual = Vector2::ZERO;
    // Post-drag pointer restore, re-checked for a few frames: X11 warps are
    // async, and GLFW's own EnableCursor re-warp can land after ours and
    // recenter the cursor. We warp again until the position sticks.
    let mut pending_restore: Option<Vector2> = None;
    let mut restore_tries = 0u32;
    // `f` held captures the pointer like a left-drag (hold = pan, cursor
    // hidden). A tap (no pointer travel) runs the `f` action on release
    // instead, so a press and a press-and-hold can be told apart.
    let mut f_capture = false;

    while !rl.window_should_close() && !quit {
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
            // Frames longer than a key tap can swallow IsKeyPressed edge
            // detection entirely (press+release between two polls); log them.
            let ft = rl.get_frame_time();
            if ft > 0.1 {
                eprintln!("vv: slow frame {ft:.0} ms");
            }
        }
        let win_w = rl.get_screen_width() as f32;
        let win_h = rl.get_screen_height() as f32;
        // Post-drag restore: verify the warp landed where we sent it; if a
        // competing re-warp (GLFW/WM/XWayland) moved it, snap again. Gives
        // up after a few tries so a user moving the mouse is never pinned.
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
        // Resize detection done ourselves: `is_window_resized()` can miss a
        // WM-reflow resize that lands around the time the window becomes
        // visible (tiling WMs shrink the freshly spawned window into its
        // tile), so the fit ease glides "out of nowhere". Any size difference
        // from the previous frame counts as a resize frame.
        let resized = last_win != Some((win_w, win_h));
        last_win = Some((win_w, win_h));

        // Dynamic window title: follow the open image in image view; back
        // to the launch title in grid mode.
        if st.open_id != title_open_id {
            title_open_id = st.open_id;
            let title = st
                .open_id
                .and_then(|id| {
                    grid.as_ref()?
                        .entries
                        .iter()
                        .find(|e| e.id == id)
                        .and_then(|e| match &e.source {
                            EntrySource::File(p) => Some(image_title(p)),
                            EntrySource::DocPage { .. } => None,
                        })
                })
                .unwrap_or_else(|| launch_title.clone());
            rl.set_window_title(&thread, &title);
        }

        // Prefetch priority: in grid mode the selected entry's neighbors
        // (left/right/up/down); in image mode the previous/next entries.
        // While the image view still waits for its own decode, only the
        // open entry may decode and the rest of the queue is paused, so
        // the visible image gets all the cores.
        // load_pending drains finished decodes (texture uploads happen here,
        // on the main thread) and dispatches new ones to the rayon pool.
        let (priority, scan_start, scan) = if st.mode == Mode::Image {
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
                    (v, i, !st.view_loading)
                }
                None => (Vec::new(), 0, true),
            }
        } else if let Some(g) = grid.as_ref() {
            (
                g.prefetch_neighbors(g.selected, win_w, win_h),
                g.selected,
                true,
            )
        } else {
            (Vec::new(), 0, true)
        };
        if let Some(g) = grid.as_mut() {
            g.load_pending(&mut rl, &thread, &priority, scan_start, scan);
        }

        // Image mode: drain the streaming loader first; texture uploads need
        // the main thread.
        if st.mode == Mode::Image {
            let mut open_failed = false;
            // Waiting on a grid decode (no streaming loader for this open):
            // when its texture lands, take it over as the view.
            if st.view_tex.is_none() && st.loader.is_none() && st.open_id.is_some() {
                let id = st.open_id.expect("checked above");
                match grid
                    .as_mut()
                    .and_then(|g| g.entries.iter().position(|e| e.id == id).map(|i| (g, i)))
                {
                    Some((g, i)) => {
                        if g.entries[i].full.is_some() {
                            // Decode landed: its full-res texture was kept
                            // for the open entry (keep set) — show_entry
                            // takes it over.
                            show_entry(&mut st, g, i, &mut rl, &thread, (win_w, win_h), false);
                        }
                        // else: the decode is still in flight or awaiting
                        // dispatch (thumb-only keep-set entry; the open
                        // entry is always in the keep set, so it will be
                        // dispatched) — the thumb placeholder covers the
                        // wait and the decode-landed takeover swaps it in.
                    }
                    None => open_failed = true, // entry vanished (decode failed)
                }
            }
            if let Some(loader) = &st.loader {
                while let Some(msg) = loader.try_recv() {
                    match msg {
                        LoaderMsg::Header { width, height } => {
                            // Dimensions known: fit-down immediately (the
                            // ease block only runs from the next frame on).
                            st.img_w = width as f32;
                            st.img_h = height as f32;
                            st.zoom = ZoomMode::FitAll;
                            st.pan = Vector2::ZERO;
                            st.target_pan = Vector2::ZERO;
                            st.view_scale = Some((win_w / st.img_w).min(win_h / st.img_h));
                        }
                        LoaderMsg::Preview {
                            rgba,
                            width,
                            height,
                            blur,
                        } => {
                            // Texture px per natural unit (points for PDFs).
                            if st.img_w > 0.0 && width > 0 {
                                st.view_render_scale = width as f32 / st.img_w;
                            }
                            // Progressively better render of the same image.
                            show_frame(
                                &mut rl,
                                &thread,
                                &mut st.view_tex,
                                &rgba,
                                width,
                                height,
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
                        LoaderMsg::Done {
                            rgba,
                            width,
                            height,
                            blur,
                        } => {
                            // Non-JXL formats only learn dimensions here.
                            if st.img_w == 0.0 {
                                st.img_w = width as f32;
                                st.img_h = height as f32;
                                st.zoom = ZoomMode::FitAll;
                                st.pan = Vector2::ZERO;
                                st.target_pan = Vector2::ZERO;
                                st.view_scale = Some((win_w / st.img_w).min(win_h / st.img_h));
                            }
                            show_frame(
                                &mut rl,
                                &thread,
                                &mut st.view_tex,
                                &rgba,
                                width,
                                height,
                                st.pixelated,
                            )?;
                            attach_blur_bg(
                                &mut st.blur_bg,
                                &mut rl,
                                &thread,
                                blur.as_ref(),
                                st.open_id,
                            );
                            if st.img_w > 0.0 && width > 0 {
                                st.view_render_scale = width as f32 / st.img_w;
                            }
                            st.view_loading = false;
                        }
                        LoaderMsg::Failed(err) => {
                            eprintln!("vv: {err}");
                            open_failed = true;
                        }
                    }
                }
            }
            if open_failed {
                st.loader = None; // cancel the worker
                // Keep the entry: mark it failed so its cell stays visible
                // (dimmed, error glyph) instead of silently vanishing.
                if let Some(id) = st.open_id.take() {
                    grid.as_mut()
                        .unwrap()
                        .mark_failed(id, "decoding failed".to_string());
                }
                st.view_from_grid = None;
                st.refine = None;
                st.view_render_scale = 1.0;
                st.mode = Mode::Grid;
                st.view_tex = None;
                st.view_loading = false;
            }
        }

        if st.mode == Mode::Grid {
            // Grid navigation: h/j/k/l + arrows move the selection,
            // Enter opens the selected image, q quits (ESC is inert here;
            // the grid is the home view).
            match grid.as_mut().unwrap().handle_input(&mut rl, win_w, win_h) {
                GridAction::Open(i) => {
                    // Multi-page documents (PDFs and .typ files) open a
                    // page-overview grid (one entry per page, decoded on the
                    // same rayon pool) instead of the image view; ESC pops
                    // back to the directory grid.
                    if let EntrySource::File(p) = &grid.as_ref().unwrap().entries[i].source
                        && has_page_grid(p)
                    {
                        match open_document(p) {
                            Ok(doc) if doc.page_count() > 1 => {
                                home_grid = grid.take();
                                grid = Some(Grid::from_document(doc));
                                continue;
                            }
                            Ok(_) => {} // single-page PDF: open like an image
                            Err(err) => {
                                eprintln!("vv: {err:#}");
                                let id = grid.as_ref().unwrap().entries[i].id;
                                grid.as_mut().unwrap().mark_failed(id, format!("{err:#}"));
                                continue;
                            }
                        }
                    }
                    if debug {
                        eprintln!("vv: enter pressed -> open idx {i}");
                    }
                    show_entry(
                        &mut st,
                        grid.as_mut().unwrap(),
                        i,
                        &mut rl,
                        &thread,
                        (win_w, win_h),
                        true,
                    );
                }
                GridAction::Back => {
                    // ESC on a page-overview grid: pop it (its textures drop
                    // with it) and restore the directory grid.
                    if let Some(home) = home_grid.take() {
                        grid = Some(home);
                    }
                }
                GridAction::Quit => quit = true,
                GridAction::None => {}
            }
        } else {
            // Image mode.
            // Drain the raw key/char queues every frame (same rationale as
            // grid.rs handle_input): is_key_pressed misses a press+release
            // pair that lands inside one frame — easy here while frames
            // stall on texture uploads or decode. Leftover queue entries
            // would otherwise leak into grid mode and act there (e.g. a
            // missed Enter immediately reopening the just-viewed image).
            // State queries (is_key_down panning, is_key_pressed W/E/=/-)
            // are unaffected: the queue is separate from the key snapshot.
            // Enter/ESC return to the grid when one exists (nsxiv-like:
            // Enter toggles between grid and the open image); q quits,
            // ESC never quits the program (inert in single-file launches,
            // where there is no grid to return to). Space/Backspace switch
            // to the next/previous image (nsxiv-style nav; Space/n next,
            // Backspace/p previous; arrows and h/j/k/l stay panning).
            // Nav keys auto-repeat while held, at the X server's rate
            // (xset r rate; keyrepeat::settings). g/G jump to the
            // first/last image.
            let mut enter = false;
            let mut quit_pressed = false;
            let mut pixelated_toggle = false;
            let mut f_key = false;
            let mut nav_next_edge = false;
            let mut nav_prev_edge = false;
            let mut jump_first = false;
            let mut jump_last = false;
            let shift = rl.is_key_down(KeyboardKey::KEY_LEFT_SHIFT)
                || rl.is_key_down(KeyboardKey::KEY_RIGHT_SHIFT);
            while let Some(k) = rl.get_key_pressed() {
                match k {
                    KeyboardKey::KEY_ESCAPE
                    | KeyboardKey::KEY_ENTER
                    | KeyboardKey::KEY_KP_ENTER => {
                        enter = true;
                    }
                    KeyboardKey::KEY_Q => quit_pressed = true,
                    KeyboardKey::KEY_A => pixelated_toggle = true,
                    KeyboardKey::KEY_F => f_key = true,
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
            // a left-drag (cursor hidden, unbounded virtual deltas); the
            // release ends it and restores the pointer to the grab point.
            // Travel marks it a hold (no tap action); otherwise the release
            // runs the `f` action at the grab point. Auto-repeat re-fires
            // the press edge while held, hence the `!pointer_captured` guard.
            let mut f_click: Option<Vector2> = None;
            if f_key && !pointer_captured {
                grab_pos = rl.get_mouse_position();
                drag_virtual = grab_pos;
                rl.disable_cursor();
                pointer_captured = true;
                f_capture = true;
                if debug {
                    eprintln!("vv: f hold grab at ({:.0},{:.0})", grab_pos.x, grab_pos.y);
                }
            }
            // `!is_key_down` also covers a press+release landing inside one
            // frame (the pressed queue records the press, but there is no
            // release edge to observe — without this the capture would stick).
            if f_capture && !rl.is_key_down(KeyboardKey::KEY_F) {
                // Fold in this frame's motion before the lock is released
                // (the drag block below no longer runs once capture ends).
                let delta = rl.get_mouse_delta();
                drag_virtual.x += delta.x;
                drag_virtual.y += delta.y;
                let f_dragged = (drag_virtual - grab_pos).length() > 4.0;
                rl.enable_cursor();
                rl.set_mouse_position(grab_pos);
                pending_restore = Some(grab_pos);
                restore_tries = 0;
                pointer_captured = false;
                f_capture = false;
                if !f_dragged {
                    f_click = Some(grab_pos);
                }
                if debug {
                    eprintln!(
                        "vv: f release: grab ({:.0},{:.0}), virtual ({:.0},{:.0}), dragged {}",
                        grab_pos.x, grab_pos.y, drag_virtual.x, drag_virtual.y, f_dragged
                    );
                }
            }
            // Auto-repeat: the initial press fires immediately (edge from
            // the queue above), holding fires at the X server's repeat rate
            // after its delay.
            let now = rl.get_time();
            let (delay, rate) = keyrepeat::settings();
            let mut nav = if rep_nav_fwd.tick(
                nav_next_edge,
                rl.is_key_down(KeyboardKey::KEY_N) || rl.is_key_down(KeyboardKey::KEY_SPACE),
                now,
                delay,
                rate,
            ) {
                Some(1)
            } else if rep_nav_back.tick(
                nav_prev_edge,
                rl.is_key_down(KeyboardKey::KEY_P) || rl.is_key_down(KeyboardKey::KEY_BACKSPACE),
                now,
                delay,
                rate,
            ) {
                Some(-1)
            } else {
                None
            };
            // g/G: jump to the first/last image (as a nav delta).
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
            // Enter as a character event ('\n'/'\r'); accept both. This
            // drain also covers the +/- zoom chars for every frame, so
            // nothing piles up while the header has not arrived yet.
            let mut zoom_in_char = false;
            let mut zoom_out_char = false;
            while let Some(c) = rl.get_char_pressed() {
                match c {
                    '\n' | '\r' => enter = true,
                    '+' => zoom_in_char = true,
                    '-' => zoom_out_char = true,
                    _ => {}
                }
            }

            let mut return_to_grid = false;
            if quit_pressed {
                quit = true; // single-file launch: no grid to fall back to
            }
            // `a`: toggle smooth (bilinear, default) vs pixelated
            // (nearest-neighbor) filtering for the shown texture; the flag
            // is re-applied to every texture that arrives later.
            if pixelated_toggle {
                st.pixelated = !st.pixelated;
                if let Some(tex) = &st.view_tex {
                    apply_view_filter(&thread, tex, st.pixelated);
                }
            }
            // `f` tap action (needs known geometry; the hit test uses the
            // drawn image rect — offset + pan at the on-screen scale):
            // cursor over the background returns to the grid (ESC-like);
            // over an image that covers the whole window it zooms out
            // (recentred, like t's zoom-out half); otherwise it zooms into
            // fill view aimed at the cursor (like t's zoom-in half). Fires
            // on the key release, and only when the hold did not pan.
            let mut f_open_grid = false;
            let mut f_zoom = false;
            if let Some(m) = f_click
                && st.img_w > 0.0
                && let Some(scale) = st.view_scale
            {
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
                // the grid's click handling needs a visible cursor.
                if pointer_captured {
                    rl.enable_cursor();
                    rl.set_mouse_position(grab_pos);
                    pending_restore = Some(grab_pos);
                    restore_tries = 0;
                    pointer_captured = false;
                }
                f_capture = false;
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
                    g.ensure_visible(win_w, win_h);
                }
                st.loader = None; // cancels a still-running stream
                st.view_loading = false;
                st.view_loading_since = 0.0;
                st.refine = None;
                st.view_render_scale = 1.0;
            }

            // Prev/next while viewing (only with a grid to navigate). A
            // ready texture swaps in the same frame; a decode in flight is
            // waited on (auto-swap when it lands); otherwise the streaming
            // loader takes over. The new entry's own neighbors are prefetched
            // via the priority list at the top of the loop.
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
                            (win_w, win_h),
                            true,
                        );
                    }
                }
            }

            // Keyboard shortcuts. Capital W / capital E arrive as W/E + shift.
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
                // t: two-state toggle — whole image visible (fit-all) vs
                // window completely covered (fill). Distinct for any
                // image/window combination, unlike a fit-width/fit-height
                // cycle. f (f_zoom) always zooms in: it only reaches here
                // when the cursor is on a not-yet-covering image.
                if f_zoom {
                    st.zoom = ZoomMode::Fill;
                } else if st.zoom == ZoomMode::Fill {
                    st.zoom = ZoomMode::FitAll;
                } else {
                    st.zoom = ZoomMode::Fill;
                }
                // Zoom-in lands with the image point under the mouse at the
                // window center, clamped so the window never shows
                // background; zoom-out recentres.
                if st.zoom == ZoomMode::Fill && st.img_w > 0.0 && st.view_scale.is_some() {
                    st.target_pan = aim_fill_pan(&st, rl.get_mouse_position(), win_w, win_h);
                } else {
                    st.target_pan = Vector2::ZERO;
                }
            }

            // Before the header arrives the image dimensions are unknown;
            // skip all scale/pan math (it divides by img_w/img_h).
            if st.img_w > 0.0 {
                // Scale for the current mode (fit modes recompute every frame, so
                // resizing stays correct).
                let target_scale = match st.zoom {
                    ZoomMode::FitDown => (win_w / st.img_w).min(win_h / st.img_h).min(1.0),
                    ZoomMode::FitAll => (win_w / st.img_w).min(win_h / st.img_h),
                    ZoomMode::FitWidth => win_w / st.img_w,
                    ZoomMode::FitHeight => win_h / st.img_h,
                    // Cover: the larger ratio wins, so the image fills the
                    // window and the other axis is cropped.
                    ZoomMode::Fill => (win_w / st.img_w).max(win_h / st.img_h),
                    ZoomMode::Free(scale) => scale,
                };

                // Ease the on-screen scale toward the target so zoom steps animate
                // smoothly (~95% of the way after 150 ms; snap when close enough).
                // On a window resize, snap instead: the new fit target should
                // track the window edge instantly, not glide after it.
                let prev_scale = st.view_scale;
                let alpha = 1.0 - (-rl.get_frame_time() / 0.05).exp();
                st.view_scale = Some(match st.view_scale {
                    None => target_scale,
                    Some(_) if resized => target_scale,
                    Some(s) => {
                        let s = s + (target_scale - s) * alpha;
                        if (target_scale - s).abs() < target_scale * 0.001 {
                            target_scale
                        } else {
                            s
                        }
                    }
                });

                // Free zoom: +/- steps the scale up/down by 25%, starting from the
                // scale currently on screen. Detected two ways: the keycode of the
                // US-layout =/- keys (incl. numpad) and the typed character, which
                // covers non-US layouts where '+' lives on another physical key.
                let zoom_in = rl.is_key_pressed(KeyboardKey::KEY_EQUAL)
                    || rl.is_key_pressed(KeyboardKey::KEY_KP_ADD)
                    || zoom_in_char;
                let zoom_out = rl.is_key_pressed(KeyboardKey::KEY_MINUS)
                    || rl.is_key_pressed(KeyboardKey::KEY_KP_SUBTRACT)
                    || zoom_out_char;
                if zoom_in || zoom_out {
                    let factor = if zoom_in { 1.25 } else { 1.0 / 1.25 };
                    st.zoom = ZoomMode::Free((target_scale * factor).clamp(0.01, 100.0));
                    // Keyboard zoom centers on the picture, not the cursor:
                    // None falls back to the window center anchor.
                    st.zoom_anchor = None;
                }
                // Mouse wheel zooms free-mode with the same 25% steps, anchored
                // at the cursor: while the scale eases, the image point under
                // the mouse stays put.
                let wheel = rl.get_mouse_wheel_move();
                if wheel != 0.0 {
                    let factor = 1.25f32.powf(wheel);
                    st.zoom = ZoomMode::Free((target_scale * factor).clamp(0.01, 100.0));
                    st.zoom_anchor = Some(rl.get_mouse_position());
                }
                // Left-drag pans: the image follows the cursor (grab-style).
                // No easing while the cursor drives the pan — the drag delta
                // is already per-frame, and piling the 50 ms pan ease on top
                // of vsync's display latency reads as lag (same-frame input,
                // no interpolation: chart-action 9a5394b lesson). Keyboard
                // panning below keeps its glide. Only meaningful once
                // dimensions are known.
                //
                // Infinite drag: while the button is held the pointer is
                // captured, so it neither disappears at the screen edge nor
                // blocks there — panning continues with virtual deltas no
                // matter how far the physical mouse travels. Mouse-up shows
                // it again at the position where the drag began.
                let drag_pressed = rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT);
                let drag_released = rl.is_mouse_button_released(MouseButton::MOUSE_BUTTON_LEFT);
                if drag_pressed && !pointer_captured {
                    grab_pos = rl.get_mouse_position();
                    drag_virtual = grab_pos;
                    rl.disable_cursor();
                    pointer_captured = true;
                    if debug {
                        eprintln!("vv: drag grab at ({:.0},{:.0})", grab_pos.x, grab_pos.y);
                    }
                } else if drag_released && pointer_captured && !f_capture {
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
                // Any captured pointer (left-drag or held `f`) pans: the
                // virtual deltas drive both the same way.
                let dragging = pointer_captured;
                if dragging {
                    let delta = rl.get_mouse_delta();
                    // Virtual cursor follows physical travel 1:1 (debug
                    // crosshair); the pan gets the 2× speedup.
                    drag_virtual.x += delta.x;
                    drag_virtual.y += delta.y;
                    st.target_pan.x += delta.x * 2.0;
                    st.target_pan.y += delta.y * 2.0;
                }

                // Mouse-anchored zoom (free zoom only): while the on-screen
                // scale eases, shift the pan each frame so the image point under
                // the anchor (the cursor when the step started, else the window
                // center) stays fixed. offset = center + pan, so keeping
                // the anchor's image point put gives
                //   offset' = anchor - (anchor - offset) * (scale'/scale).
                // Gimmick: after each frame's anchor step, the image and the
                // cursor slide TOGETHER a little toward the window center, so
                // zooming gently recenters while the cursor stays glued to the
                // same image point (image moves, pointer rides along).
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
                    let a = st.zoom_anchor.unwrap_or(center);
                    let (ax, ay) = (a.x, a.y);
                    let keyboard_zoom = st.zoom_anchor.is_none();
                    let r = scale / s_old;
                    let ox = ax - (ax - (win_w - st.img_w * s_old) / 2.0 - st.pan.x) * r;
                    let oy = ay - (ay - (win_h - st.img_h * s_old) / 2.0 - st.pan.y) * r;
                    st.pan.x = ox - (win_w - st.img_w * scale) / 2.0;
                    st.pan.y = oy - (win_h - st.img_h * scale) / 2.0;
                    // Drift shares the ease's exponential time constant: by
                    // the time the scale has covered its remaining distance,
                    // the pair has covered ZOOM_ANCHOR_DRIFT of its own.
                    let d = (center - a) * (alpha * ZOOM_ANCHOR_DRIFT);
                    st.pan.x += d.x;
                    st.pan.y += d.y;
                    if !keyboard_zoom {
                        st.zoom_anchor = Some(a + d);
                    }
                    // Pin the target too, so pan easing doesn't fight the anchor.
                    st.target_pan.x = st.pan.x;
                    st.target_pan.y = st.pan.y;
                    // Ride the pointer along with the drifted image point
                    // (never while a drag has it captured; keyboard zoom
                    // leaves the cursor wherever it is).
                    if !keyboard_zoom && !pointer_captured {
                        if debug {
                            eprintln!("vv: zoom drift anchor to ({ax:.0},{ay:.0})");
                        }
                        rl.set_mouse_position(st.zoom_anchor.unwrap());
                    }
                }

                // Vim-style panning (h/j/k/l + arrow keys); held keys move the
                // target offset, the on-screen pan eases after it (same exponential
                // easing as zoom), so taps glide and holds scroll smoothly.
                // Input read before begin_drawing borrows rl mutably.
                // Per-second speed: matches the old 3%-of-window-per-frame pace
                // (3% × 60 fps = 180% per second), now frame-time aware.
                let speed = win_w.max(win_h) * 1.8 * rl.get_frame_time();
                let pan_left =
                    rl.is_key_down(KeyboardKey::KEY_H) || rl.is_key_down(KeyboardKey::KEY_LEFT);
                let pan_right =
                    rl.is_key_down(KeyboardKey::KEY_L) || rl.is_key_down(KeyboardKey::KEY_RIGHT);
                let pan_up =
                    rl.is_key_down(KeyboardKey::KEY_K) || rl.is_key_down(KeyboardKey::KEY_UP);
                let pan_down =
                    rl.is_key_down(KeyboardKey::KEY_J) || rl.is_key_down(KeyboardKey::KEY_DOWN);
                if pan_left {
                    st.target_pan.x += speed;
                }
                if pan_right {
                    st.target_pan.x -= speed;
                }
                if pan_up {
                    st.target_pan.y += speed;
                }
                if pan_down {
                    st.target_pan.y -= speed;
                }
                // On a resize frame, snap pan too so it doesn't glide after the
                // new fit offset (matches the scale snap above).
                if resized {
                    st.pan = st.target_pan;
                }
                let pan_alpha = 1.0 - (-rl.get_frame_time() / 0.05).exp();
                st.pan = Vector2 {
                    x: st.pan.x + (st.target_pan.x - st.pan.x) * pan_alpha,
                    y: st.pan.y + (st.target_pan.y - st.pan.y) * pan_alpha,
                };
                // While dragging, skip the ease entirely: the image sticks to
                // the cursor (only the zoom-anchor shift above may touch pan).
                if dragging {
                    st.pan = st.target_pan;
                }
                // Snap when the residual glide is sub-pixel.
                if (st.target_pan.x - st.pan.x).abs() < 0.25 {
                    st.pan.x = st.target_pan.x;
                }
                if (st.target_pan.y - st.pan.y).abs() < 0.25 {
                    st.pan.y = st.target_pan.y;
                }

                // Document pages (PDF): once the zoom animation settles and
                // the settled scale differs enough from the scale the current
                // texture was rendered at, re-render the page at that scale.
                // Text stays sharp at every resting zoom level (fresh
                // rasterization at screen resolution — same trick zathura/
                // mupdf use). While the render runs, the old texture keeps
                // showing (slightly soft, never blank).
                let view_doc = st.open_id.and_then(|id| {
                    grid.as_ref().and_then(|g| {
                        g.index_of(id).and_then(|i| match &g.entries[i].source {
                            EntrySource::DocPage { doc, page } => Some((doc.clone(), *page)),
                            EntrySource::File(_) => None,
                        })
                    })
                });
                if let Some((doc, page)) = view_doc {
                    let settled = st
                        .view_scale
                        .is_some_and(|s| (s - target_scale).abs() <= target_scale * 0.001);
                    let scale = st.view_scale.unwrap_or(1.0);
                    let rel_diff =
                        (scale - st.view_render_scale).abs() / st.view_render_scale.max(1e-3);
                    if settled && st.refine.is_none() && rel_diff > 0.15 {
                        let (tx, rx) = channel();
                        let doc = doc.clone();
                        let id = st.open_id.unwrap_or(u64::MAX);
                        let rs = scale.clamp(0.05, 8.0);
                        rayon::spawn(move || {
                            let res = doc.render(page, rs).map_err(|e| format!("{e:#}"));
                            // Receiver gone (view left): result discarded.
                            let _ = tx.send((id, page, rs, res));
                        });
                        st.refine = Some(rx);
                    }
                }
                // Poll a finished refine render; accept it only if it still
                // matches what is on screen (same entry, page and zoom).
                if let Some(rx) = &st.refine
                    && let Ok((id, page, rs, res)) = rx.try_recv()
                {
                    st.refine = None;
                    let same_page = st.open_id == Some(id)
                        && grid.as_ref().and_then(|g| {
                            g.index_of(id).map(|i| match &g.entries[i].source {
                                EntrySource::DocPage { page: p, .. } => *p == page,
                                EntrySource::File(_) => false,
                            })
                        }) == Some(true);
                    if same_page
                        && st.view_scale.is_some_and(|s| (s - rs).abs() <= rs * 0.2)
                        && let Ok(d) = res
                    {
                        show_frame(
                            &mut rl,
                            &thread,
                            &mut st.view_tex,
                            &d.rgba,
                            d.width,
                            d.height,
                            st.pixelated,
                        )?;
                        st.view_render_scale = rs;
                    }
                }
            } // st.img_w > 0.0: fit/pan math needs known dimensions
        }

        // VV_BLUR_BG in grid mode: the background follows the selected
        // entry with a slow crossfade (starts as soon as the entry's
        // blurred copy has been decoded by the grid workers).
        if st.mode == Mode::Grid
            && let (Some(bg), Some(g)) = (st.blur_bg.as_mut(), grid.as_ref())
            && let Some(e) = g.entries.get(g.selected)
            && let Some(data) = &e.blur
        {
            bg.transition(&mut rl, &thread, data, e.id);
        }

        // Whether the "decoding..." indicator should show this frame: only
        // once the load has taken over a second; brief loads would otherwise
        // flash the text for a few frames. Computed before begin_drawing
        // (rl is mutably borrowed by the draw handle).
        let show_decoding = st.view_loading && rl.get_time() - st.view_loading_since > 1.0;

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        if st.mode == Mode::Grid {
            // VV_BLUR_BG gimmick: blurred background follows the selection
            // (crossfading; no-op when the gimmick is off). Drawn before
            // the grid so fades run under the thumbnails.
            if let Some(bg) = &mut st.blur_bg {
                bg.draw(&mut d, win_w, win_h);
            }
            grid.as_ref().unwrap().draw(&mut d, win_w, win_h);
        } else {
            // VV_BLUR_BG gimmick: blurred copy of the image, scaled to
            // cover the whole window (fit on the narrower side; the other
            // axis overflows and is cropped) behind the sharp image.
            if let Some(bg) = &mut st.blur_bg {
                bg.draw(&mut d, win_w, win_h);
            }
            if let Some(texture) = &st.view_tex {
                let dw = st.img_w * st.view_scale.unwrap();
                let dh = st.img_h * st.view_scale.unwrap();

                // Source rect in TEXTURE pixels, not natural units: PDF
                // pages re-render at the settled zoom scale, so a refined
                // texture holds points * scale pixels while img_w/img_h
                // stay in points (the fit math works in natural units).
                // Sampling img_w here would crop the top-left fraction of
                // the texture and stretch it over the full destination — a
                // crop+blowup that reads as "image zoom AND pdf zoom applied
                // at once". Plain images never notice: texture pixels equal
                // natural pixels there.
                // Also NOT the logical image size: uploads clamp oversized
                // textures to MAX_TEXTURE_SIDE (and streaming previews to
                // the screen size), so the texture can be smaller than
                // st.img_w/h. A src rect larger than the texture produces
                // UVs > 1, which raylib's repeat wrap renders as a tiled
                // mosaic.
                let src = Rectangle {
                    x: 0.0,
                    y: 0.0,
                    width: texture.width() as f32,
                    height: texture.height() as f32,
                };
                let dest = Rectangle {
                    x: (win_w - dw) / 2.0 + st.pan.x,
                    y: (win_h - dh) / 2.0 + st.pan.y,
                    width: dw,
                    height: dh,
                };
                d.draw_texture_pro(texture, src, dest, Vector2::ZERO, 0.0, Color::WHITE);
            } else if st.img_w > 0.0
                && let Some(g) = grid.as_ref()
                && let Some(id) = st.open_id
                && let Some(i) = g.index_of(id)
                && let Some(tex) = g.entries.get(i).and_then(|e| e.texture.as_ref())
            {
                // No full-res frame yet (decode in flight): show the
                // entry's thumb instead of flashing black. The thumb is
                // the whole image downscaled with the aspect preserved,
                // so it maps 1:1 onto the image rect and the later swap
                // to the full-res texture is pixel-aligned.
                let scale = st
                    .view_scale
                    .unwrap_or((win_w / st.img_w).min(win_h / st.img_h));
                let dw = st.img_w * scale;
                let dh = st.img_h * scale;
                let src = Rectangle {
                    x: 0.0,
                    y: 0.0,
                    width: tex.width() as f32,
                    height: tex.height() as f32,
                };
                let dest = Rectangle {
                    x: (win_w - dw) / 2.0 + st.pan.x,
                    y: (win_h - dh) / 2.0 + st.pan.y,
                    width: dw,
                    height: dh,
                };
                d.draw_texture_pro(tex, src, dest, Vector2::ZERO, 0.0, Color::WHITE);
            }
            // Threshold check happens above, before begin_drawing.
            if show_decoding {
                let msg = "decoding...";
                let tw = d.measure_text(msg, 20);
                d.draw_text(
                    msg,
                    (win_w as i32 - tw) / 2,
                    win_h as i32 / 2 - 10,
                    20,
                    Color::GRAY,
                );
            }
            // VV_DEBUG: while the pointer is captured, draw a crosshair at
            // the virtual cursor position (clamped to the window) — the real
            // cursor is hidden, and this shows where vv thinks it is. Plus a
            // ring at the grab point, so a wrong restore is visible.
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
    }
    Ok(())
}
