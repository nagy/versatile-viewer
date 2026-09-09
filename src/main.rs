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

use std::{env, path::Path};

use anyhow::{Context, Result, bail};
use raylib::{
    color::Color,
    consts::{KeyboardKey, PixelFormat},
    prelude::*,
};

mod blurbg;
mod document;
mod grid;
mod keyrepeat;
mod loader;
mod pdf;
use blurbg::BlurBg;
use document::{MAX_TEXTURE_SIDE, open_document};
// Decode helpers live in document.rs now; re-exported at the crate root
// until every module addresses them there (grid/loader rewiring follows).
pub(crate) use document::{decode_common, decode_image, downscale_rgba, fb_to_rgba, is_jxl};
use grid::{Grid, GridAction};
use loader::{Loader, LoaderMsg};

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
    /// Streaming loader for the open image; drop cancels the worker.
    loader: Option<Loader>,
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
        self.view_scale = None;
    }
}

/// Upload a raw RGBA8 buffer as a GPU texture.
///
/// Must be called on the main thread (GL context lives there). The buffer
/// is only borrowed for the upload; the `ffi::Image` wrapper is forgotten so
/// raylib never frees the caller's Vec.
fn upload_rgba(
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<Texture2D> {
    // Oversized images (huge panoramas) would fail the GL upload; clamp
    // them to the safe side limit here, in the single choke point.
    let owned;
    let (rgba, width, height) = if width.max(height) > MAX_TEXTURE_SIDE {
        let (buf, w, h) = downscale_rgba(rgba.to_vec(), width, height, MAX_TEXTURE_SIDE);
        owned = buf;
        (owned.as_slice(), w, h)
    } else {
        (rgba, width, height)
    };
    let ffi_image = raylib::ffi::Image {
        data: rgba.as_ptr() as *mut std::os::raw::c_void,
        width: width as i32,
        height: height as i32,
        mipmaps: 1,
        format: PixelFormat::PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 as i32,
    };
    let image = unsafe { Image::from_raw(ffi_image) };
    let texture = rl.load_texture_from_image(thread, &image)?;
    image.to_raw(); // forget: drop would MemFree our borrowed Vec
    Ok(texture)
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
) -> Result<()> {
    match view_tex {
        Some(tex) if tex.width() == width as i32 && tex.height() == height as i32 => {
            use raylib::texture::RaylibTexture2D;
            tex.update_texture(rgba)?;
        }
        _ => {
            *view_tex = Some(upload_rgba(rl, thread, rgba, width, height)?);
        }
    }
    Ok(())
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
    if let Some(tex) = tex {
        grid.entries[idx].viewing = true;
        st.view_from_grid = Some(id);
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
            Some(Loader::start(path, preview_px))
        };
        attach_blur_bg(&mut st.blur_bg, rl, thread, blur.as_ref(), st.open_id);
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
    // image size); multi-page documents (PDF) render page 0 at fit scale
    // for the window size (the page-grid launch replaces this).
    let single_doc = if dir_grid.is_none() {
        Some(open_document(path)?)
    } else {
        None
    };
    let single_decoded = match &single_doc {
        Some(doc) => {
            let info = doc.page_info(0)?;
            if doc.page_count() == 1 {
                Some(doc.render(0, 1.0)?)
            } else {
                let f = (1024.0 / info.width.max(1) as f32)
                    .min(768.0 / info.height.max(1) as f32)
                    .min(1.0);
                Some(doc.render(0, f.max(0.01))?)
            }
        }
        None => None,
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
    let (win0_w, win0_h) = single_decoded
        .as_ref()
        .map_or((1024, 768), |d| (d.width as i32, d.height as i32));

    let (mut rl, thread) = raylib::init()
        .size(win0_w, win0_h)
        .title(&format!(
            "versatile-viewer — {}",
            if let Some(g) = &dir_grid {
                format!("{} ({} images)", path.display(), g.entries.len())
            } else {
                path.display().to_string()
            }
        ))
        .resizable()
        .build();
    // We quit via the q key handling ourselves (set_exit_key would make ESC
    // close the window outright instead of returning to the grid).
    rl.set_exit_key(None);

    // VV_BLUR_BG gimmick: blurred copy of the viewed image behind it. Declared
    // after `rl` so it drops (and unloads its texture) before the window.
    // Take the grid AFTER the window exists: it holds GPU textures, and
    // being declared after `rl` it is dropped (and unloaded) BEFORE the
    // window closes — both on normal exit and on panic unwinding.
    let mut grid = dir_grid;
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
        loader: None,
        blur_bg: BlurBg::from_env(),
    };
    if let Some(decoded) = single_decoded {
        st.view_tex = Some(upload_rgba(
            &mut rl,
            &thread,
            &decoded.rgba,
            decoded.width,
            decoded.height,
        )?);
        drop(decoded.rgba);
        st.img_w = win0_w as f32;
        st.img_h = win0_h as f32;
        if let Some(data) = single_blur {
            attach_blur_bg(&mut st.blur_bg, &mut rl, &thread, Some(&data), None);
        }
    }

    // Set when the viewer should exit entirely.
    let mut quit = false;
    // VV_DEBUG=1: trace grid open/return events to stderr.
    let debug = std::env::var_os("VV_DEBUG").is_some();
    // Previous per-key down state for the VV_DEBUG event trace.
    let mut prev_down = [false; 349];

    rl.set_target_fps(60);
    // Auto-repeat state for image-mode prev/next (xset r rate values).
    let mut rep_nav_fwd = keyrepeat::RepeatState::new();
    let mut rep_nav_back = keyrepeat::RepeatState::new();

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

        // Prefetch priority: in grid mode the selected entry's neighbors
        // (left/right/up/down); in image mode the previous/next entries.
        // load_pending drains finished decodes (texture uploads happen here,
        // on the main thread) and dispatches new ones to the rayon pool.
        let (priority, scan_start) = if st.mode == Mode::Image {
            match st
                .open_id
                .and_then(|id| grid.as_ref().and_then(|g| g.index_of(id)))
            {
                Some(i) => {
                    let n = grid.as_ref().unwrap().entries.len();
                    let mut v = Vec::new();
                    if i > 0 {
                        v.push(i - 1);
                    }
                    if i + 1 < n {
                        v.push(i + 1);
                    }
                    (v, i)
                }
                None => (Vec::new(), 0),
            }
        } else if let Some(g) = grid.as_ref() {
            (g.prefetch_neighbors(g.selected, win_w, win_h), g.selected)
        } else {
            (Vec::new(), 0)
        };
        if let Some(g) = grid.as_mut() {
            g.load_pending(&mut rl, &thread, &priority, scan_start);
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
                            // Progressively better render of the same image.
                            show_frame(&mut rl, &thread, &mut st.view_tex, &rgba, width, height)?;
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
                            show_frame(&mut rl, &thread, &mut st.view_tex, &rgba, width, height)?;
                            attach_blur_bg(
                                &mut st.blur_bg,
                                &mut rl,
                                &thread,
                                blur.as_ref(),
                                st.open_id,
                            );
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
            if grid.is_some() && enter {
                return_to_grid = true;
            }
            if return_to_grid {
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
            } else if rl.is_key_pressed(KeyboardKey::KEY_T) {
                // Two-state toggle: whole image visible (fit-all) vs window
                // completely covered (fill). Distinct for any image/window
                // combination, unlike a fit-width/fit-height cycle.
                st.zoom = if st.zoom == ZoomMode::Fill {
                    ZoomMode::FitAll
                } else {
                    ZoomMode::Fill
                };
                st.target_pan = Vector2::ZERO;
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
                let prev_scale = st.view_scale;
                let alpha = 1.0 - (-rl.get_frame_time() / 0.05).exp();
                st.view_scale = Some(match st.view_scale {
                    None => target_scale,
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
                }
                // Mouse wheel zooms free-mode with the same 25% steps; the
                // center-anchored easing below keeps the zoom anchored.
                let wheel = rl.get_mouse_wheel_move();
                if wheel != 0.0 {
                    let factor = 1.25f32.powf(wheel);
                    st.zoom = ZoomMode::Free((target_scale * factor).clamp(0.01, 100.0));
                }
                // Left-drag pans: the image follows the cursor (grab-style).
                // Same easing path as h/j/k/l panning, so it glides and
                // settles; only meaningful once dimensions are known.
                if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT) {
                    let delta = rl.get_mouse_delta();
                    st.target_pan.x += delta.x;
                    st.target_pan.y += delta.y;
                }

                // Window-center-anchored zoom (free zoom only): while the on-screen
                // scale eases, shift the pan each frame so the image point under
                // the window center stays fixed. offset = center + pan, so keeping
                // the anchor's image point put gives
                //   offset' = anchor - (anchor - offset) * (scale'/scale).
                let scale = st.view_scale.unwrap();
                if matches!(st.zoom, ZoomMode::Free(_))
                    && let Some(s_old) = prev_scale
                    && (scale - s_old).abs() > f32::EPSILON
                    && s_old > 0.0
                {
                    let r = scale / s_old;
                    let ax = win_w / 2.0;
                    let ay = win_h / 2.0;
                    let ox = ax - (ax - (win_w - st.img_w * s_old) / 2.0 - st.pan.x) * r;
                    let oy = ay - (ay - (win_h - st.img_h * s_old) / 2.0 - st.pan.y) * r;
                    st.pan.x = ox - (win_w - st.img_w * scale) / 2.0;
                    st.pan.y = oy - (win_h - st.img_h * scale) / 2.0;
                    // Pin the target too, so pan easing doesn't fight the anchor.
                    st.target_pan.x = st.pan.x;
                    st.target_pan.y = st.pan.y;
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
                let pan_alpha = 1.0 - (-rl.get_frame_time() / 0.05).exp();
                st.pan = Vector2 {
                    x: st.pan.x + (st.target_pan.x - st.pan.x) * pan_alpha,
                    y: st.pan.y + (st.target_pan.y - st.pan.y) * pan_alpha,
                };
                // Snap when the residual glide is sub-pixel.
                if (st.target_pan.x - st.pan.x).abs() < 0.25 {
                    st.pan.x = st.target_pan.x;
                }
                if (st.target_pan.y - st.pan.y).abs() < 0.25 {
                    st.pan.y = st.target_pan.y;
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

                let src = Rectangle {
                    x: 0.0,
                    y: 0.0,
                    width: st.img_w,
                    height: st.img_h,
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
        }
    }
    Ok(())
}
