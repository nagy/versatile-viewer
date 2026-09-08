//! versatile-viewer — image viewer (JXL first-class, plus PNG/JPEG) with a
//! directory thumbnail grid. q quits; ESC/Enter toggle grid ↔ image view.
//!
//! Usage: versatile-viewer <image-path-or-directory>

use std::{env, path::Path};

use anyhow::{Context, Result, bail};
use raylib::{
    color::Color,
    consts::{KeyboardKey, PixelFormat},
    prelude::*,
};

mod grid;
mod loader;
use grid::{Grid, GridAction};
use loader::{Loader, LoaderMsg};

struct DecodedImage {
    width: u32,
    height: u32,
    /// RGBA8, row-major, 4 bytes per pixel.
    rgba: Vec<u8>,
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
    /// Free zoom factor, set with +/- (multiples of the last fit scale).
    Free(f32),
}

/// Viewer screen: thumbnail grid or single image.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Grid,
    Image,
}

/// Decode a JPEG XL file with jxl-oxide (pure Rust).
fn decode_jxl(path: &Path) -> Result<DecodedImage> {
    let image = jxl_oxide::JxlImage::builder()
        .open(path)
        .map_err(|e| anyhow::anyhow!("jxl-oxide: {e}"))
        .context("failed to open file")?;
    let render = image
        .render_frame(0)
        .map_err(|e| anyhow::anyhow!("jxl-oxide: {e}"))
        .context("failed to render frame")?;

    let (rgba, width, height) = fb_to_rgba(&render.image_all_channels())?;
    Ok(DecodedImage {
        width,
        height,
        rgba,
    })
}

/// Convert a jxl-oxide framebuffer (f32 samples, 3 or 4 interleaved channels)
/// to RGBA8, forcing alpha = 1.0 for opaque 3-channel data.
pub(crate) fn fb_to_rgba(fb: &jxl_oxide::FrameBuffer) -> Result<(Vec<u8>, u32, u32)> {
    let width = fb.width() as u32;
    let height = fb.height() as u32;
    let channels = fb.channels();
    let samples = fb.buf();
    if !matches!(channels, 3 | 4) {
        bail!("unexpected channel count from jxl-oxide: {channels}");
    }

    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    for (dst, src) in rgba.chunks_exact_mut(4).zip(samples.chunks_exact(channels)) {
        dst[0] = to_u8(src[0]);
        dst[1] = to_u8(src[1]);
        dst[2] = to_u8(src[2]);
        dst[3] = to_u8(src.get(3).copied().unwrap_or(1.0));
    }
    Ok((rgba, width, height))
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Decode common formats (PNG, JPEG, ...) with the `image` crate.
fn decode_common(path: &Path) -> Result<DecodedImage> {
    let rgba = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?
        .into_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(DecodedImage {
        width,
        height,
        rgba: rgba.into_raw(),
    })
}

fn is_jxl(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("jxl"))
        || {
            // JXL codestream sniff when extension is missing.
            std::fs::File::open(path)
                .and_then(|mut f| {
                    use std::io::Read;
                    let mut magic = [0u8; 2];
                    f.read_exact(&mut magic)?;
                    Ok(magic == [0xff, 0x0a])
                })
                .unwrap_or(false)
        }
}

fn decode_image(path: &Path) -> Result<DecodedImage> {
    if is_jxl(path) {
        decode_jxl(path)
    } else {
        decode_common(path)
    }
    .with_context(|| format!("failed to decode {path:?}"))
}

/// Decode an image and upload it as a GPU texture. The raw RGBA buffer is
/// wrapped in a raylib Image without re-encoding; the ffi::Image only
/// borrows it, so the wrapper is forgotten after upload and the Vec stays
/// Upload a raw RGBA8 buffer as a GPU texture. Must be called on the main
/// thread (GL context lives there). The buffer is only borrowed for the
/// upload; the ffi::Image wrapper is forgotten so raylib never frees the
/// caller's Vec.
fn upload_rgba(
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<Texture2D> {
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
        bail!("no such file or directory: {path:?}");
    };

    // Single-image launch: decode before opening the window so it can be
    // sized to the image. Directory launch: fixed default window size.
    let single_decoded = if dir_grid.is_none() {
        Some(decode_image(path)?)
    } else {
        None
    };
    let (win0_w, win0_h) = single_decoded
        .as_ref()
        .map(|d| (d.width as i32, d.height as i32))
        .unwrap_or((1024, 768));

    let (mut rl, thread) = raylib::init()
        .size(win0_w, win0_h)
        .title(&format!(
            "versatile-viewer — {}",
            if dir_grid.is_none() {
                path.display().to_string()
            } else {
                format!(
                    "{} ({} images)",
                    path.display(),
                    dir_grid.as_ref().unwrap().entries.len()
                )
            }
        ))
        .resizable()
        .build();
    // We quit via the q key handling ourselves (set_exit_key would make ESC
    // close the window outright instead of returning to the grid).
    rl.set_exit_key(None);

    // Take the grid AFTER the window exists: it holds GPU textures, and
    // being declared after `rl` it is dropped (and unloaded) BEFORE the
    // window closes — both on normal exit and on panic unwinding.
    let mut grid = dir_grid;
    // Background streamer for the open image (no GL objects inside, but like
    // `grid` declared after `rl` so it drops before the window closes). Drop
    // cancels the worker.
    let mut loader: Option<Loader> = None;

    // Texture currently shown in image mode (single-file launch sets it up
    // front; grid-opened images stream in via the loader).
    let mut view_tex: Option<Texture2D> = None;
    let mut view_loading = false;
    // When the current load started (rl.get_time()); the "decoding..."
    // indicator only appears once the load exceeds 1 s so fast loads never
    // flash text on screen.
    let mut view_loading_since = 0.0f64;
    let mut img_w = 0.0f32;
    let mut img_h = 0.0f32;
    if let Some(decoded) = single_decoded {
        let ffi_image = raylib::ffi::Image {
            data: decoded.rgba.as_ptr() as *mut std::os::raw::c_void,
            width: decoded.width as i32,
            height: decoded.height as i32,
            mipmaps: 1,
            format: PixelFormat::PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 as i32,
        };
        let image = unsafe { Image::from_raw(ffi_image) };
        view_tex = Some(rl.load_texture_from_image(&thread, &image)?);
        image.to_raw(); // forget: drop would MemFree our borrowed Vec
        drop(decoded.rgba);
        img_w = win0_w as f32;
        img_h = win0_h as f32;
    }

    let mut mode = if grid.is_none() {
        Mode::Image
    } else {
        Mode::Grid
    };
    // Which grid entry is open in image mode (None when launched with a file).
    let mut open_idx: Option<usize> = None;
    // Set when the viewer should exit entirely.
    let mut quit = false;
    // VV_DEBUG=1: trace grid open/return events to stderr.
    let debug = std::env::var_os("VV_DEBUG").is_some();
    // Previous per-key down state for the VV_DEBUG event trace.
    let mut prev_down = [false; 349];

    rl.set_target_fps(60);
    let mut zoom = ZoomMode::FitDown;
    // Target pan offset; on-screen pan eases toward it (same easing as zoom).
    let mut target_pan = Vector2::ZERO;
    let mut pan = Vector2::ZERO;
    // On-screen scale, eased toward the target scale each frame.
    let mut view_scale: Option<f32> = None;

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

        // Image mode: drain the streaming loader first; texture uploads need
        // the main thread.
        if mode == Mode::Image {
            let mut open_failed = false;
            if let Some(loader) = &loader {
                while let Some(msg) = loader.try_recv() {
                    match msg {
                        LoaderMsg::Header { width, height } => {
                            // Dimensions known: fit-down immediately (the
                            // ease block only runs from the next frame on).
                            img_w = width as f32;
                            img_h = height as f32;
                            zoom = ZoomMode::FitDown;
                            pan = Vector2::ZERO;
                            target_pan = Vector2::ZERO;
                            view_scale = Some((win_w / img_w).min(win_h / img_h).min(1.0));
                        }
                        LoaderMsg::Preview {
                            rgba,
                            width,
                            height,
                        } => {
                            // Progressively better render of the same image.
                            show_frame(&mut rl, &thread, &mut view_tex, &rgba, width, height)?;
                        }
                        LoaderMsg::Done {
                            rgba,
                            width,
                            height,
                        } => {
                            // Non-JXL formats only learn dimensions here.
                            if img_w == 0.0 {
                                img_w = width as f32;
                                img_h = height as f32;
                                zoom = ZoomMode::FitDown;
                                pan = Vector2::ZERO;
                                target_pan = Vector2::ZERO;
                                view_scale = Some((win_w / img_w).min(win_h / img_h).min(1.0));
                            }
                            show_frame(&mut rl, &thread, &mut view_tex, &rgba, width, height)?;
                            view_loading = false;
                        }
                        LoaderMsg::Failed(err) => {
                            eprintln!("vv: {err}");
                            open_failed = true;
                        }
                    }
                }
            }
            if open_failed {
                loader = None; // cancel the worker
                if let Some(i) = open_idx.take() {
                    grid.as_mut().unwrap().remove_entry(i);
                }
                mode = Mode::Grid;
                view_tex = None;
                view_loading = false;
            }
        }

        if mode == Mode::Grid {
            let g = grid.as_mut().unwrap();
            // Fill the thumbnail queue, a few decodes per frame so the grid
            // appears progressively instead of blocking on the whole
            // directory. Start near the selection so what you look at
            // appears first.
            g.load_pending(&mut rl, &thread);

            // Grid navigation: h/j/k/l + arrows move the selection,
            // Enter opens the selected image, q quits (ESC is inert here;
            // the grid is the home view).
            match g.handle_input(&mut rl, win_w, win_h) {
                GridAction::Open(i) => {
                    if debug {
                        eprintln!("vv: enter pressed -> open idx {i}");
                    }
                    // Open immediately; the image streams in on a worker
                    // thread (header -> blurry previews -> final render).
                    let path = g.entries[i].path.clone();
                    loader = Some(Loader::start(path));
                    open_idx = Some(i);
                    mode = Mode::Image;
                    view_loading = true;
                    view_loading_since = rl.get_time();
                    img_w = 0.0; // dimensions arrive with the header message
                    img_h = 0.0;
                    view_tex = None;
                    zoom = ZoomMode::FitDown;
                    pan = Vector2::ZERO;
                    target_pan = Vector2::ZERO;
                    view_scale = None;
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
            // where there is no grid to return to).
            let mut enter = false;
            let mut quit_pressed = false;
            while let Some(k) = rl.get_key_pressed() {
                match k {
                    KeyboardKey::KEY_ENTER | KeyboardKey::KEY_KP_ENTER => enter = true,
                    KeyboardKey::KEY_ESCAPE => enter = true,
                    KeyboardKey::KEY_Q => quit_pressed = true,
                    _ => {}
                }
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
                mode = Mode::Grid;
                open_idx = None;
                loader = None; // cancels a still-running stream
                view_tex = None;
                view_loading = false;
                view_loading_since = 0.0;
            }

            // Keyboard shortcuts. Capital W / capital E arrive as W/E + shift.
            let shift = rl.is_key_down(KeyboardKey::KEY_LEFT_SHIFT)
                || rl.is_key_down(KeyboardKey::KEY_RIGHT_SHIFT);
            if rl.is_key_pressed(KeyboardKey::KEY_W) {
                zoom = if shift {
                    ZoomMode::FitAll
                } else {
                    ZoomMode::FitDown
                };
                target_pan = Vector2::ZERO;
            } else if rl.is_key_pressed(KeyboardKey::KEY_E) {
                zoom = if shift {
                    ZoomMode::FitHeight
                } else {
                    ZoomMode::FitWidth
                };
                target_pan = Vector2::ZERO;
            }

            // Before the header arrives the image dimensions are unknown;
            // skip all scale/pan math (it divides by img_w/img_h).
            if img_w > 0.0 {
                // Scale for the current mode (fit modes recompute every frame, so
                // resizing stays correct).
                let target_scale = match zoom {
                    ZoomMode::FitDown => (win_w / img_w).min(win_h / img_h).min(1.0),
                    ZoomMode::FitAll => (win_w / img_w).min(win_h / img_h),
                    ZoomMode::FitWidth => win_w / img_w,
                    ZoomMode::FitHeight => win_h / img_h,
                    ZoomMode::Free(scale) => scale,
                };

                // Ease the on-screen scale toward the target so zoom steps animate
                // smoothly (~95% of the way after 150 ms; snap when close enough).
                let prev_scale = view_scale;
                let alpha = 1.0 - (-rl.get_frame_time() / 0.05).exp();
                view_scale = Some(match view_scale {
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
                let mut zoom_in = rl.is_key_pressed(KeyboardKey::KEY_EQUAL)
                    || rl.is_key_pressed(KeyboardKey::KEY_KP_ADD)
                    || zoom_in_char;
                let mut zoom_out = rl.is_key_pressed(KeyboardKey::KEY_MINUS)
                    || rl.is_key_pressed(KeyboardKey::KEY_KP_SUBTRACT)
                    || zoom_out_char;
                if zoom_in || zoom_out {
                    let factor = if zoom_in { 1.25 } else { 1.0 / 1.25 };
                    zoom = ZoomMode::Free((target_scale * factor).clamp(0.01, 100.0));
                }

                // Window-center-anchored zoom (free zoom only): while the on-screen
                // scale eases, shift the pan each frame so the image point under
                // the window center stays fixed. offset = center + pan, so keeping
                // the anchor's image point put gives
                //   offset' = anchor - (anchor - offset) * (scale'/scale).
                let scale = view_scale.unwrap();
                if matches!(zoom, ZoomMode::Free(_)) {
                    if let Some(s_old) = prev_scale {
                        if (scale - s_old).abs() > f32::EPSILON && s_old > 0.0 {
                            let r = scale / s_old;
                            let ax = win_w / 2.0;
                            let ay = win_h / 2.0;
                            let ox = ax - (ax - (win_w - img_w * s_old) / 2.0 - pan.x) * r;
                            let oy = ay - (ay - (win_h - img_h * s_old) / 2.0 - pan.y) * r;
                            pan.x = ox - (win_w - img_w * scale) / 2.0;
                            pan.y = oy - (win_h - img_h * scale) / 2.0;
                            // Pin the target too, so pan easing doesn't fight the anchor.
                            target_pan.x = pan.x;
                            target_pan.y = pan.y;
                        }
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
                    target_pan.x += speed;
                }
                if pan_right {
                    target_pan.x -= speed;
                }
                if pan_up {
                    target_pan.y += speed;
                }
                if pan_down {
                    target_pan.y -= speed;
                }
                let pan_alpha = 1.0 - (-rl.get_frame_time() / 0.05).exp();
                pan = Vector2 {
                    x: pan.x + (target_pan.x - pan.x) * pan_alpha,
                    y: pan.y + (target_pan.y - pan.y) * pan_alpha,
                };
                // Snap when the residual glide is sub-pixel.
                if (target_pan.x - pan.x).abs() < 0.25 {
                    pan.x = target_pan.x;
                }
                if (target_pan.y - pan.y).abs() < 0.25 {
                    pan.y = target_pan.y;
                }
            } // img_w > 0.0: fit/pan math needs known dimensions
        }

        // Whether the "decoding..." indicator should show this frame: only
        // once the load has taken over a second; brief loads would otherwise
        // flash the text for a few frames. Computed before begin_drawing
        // (rl is mutably borrowed by the draw handle).
        let show_decoding = view_loading && rl.get_time() - view_loading_since > 1.0;

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        if mode == Mode::Grid {
            grid.as_ref().unwrap().draw(&mut d, win_w, win_h);
        } else {
            if let Some(texture) = &view_tex {
                let dw = img_w * view_scale.unwrap();
                let dh = img_h * view_scale.unwrap();

                let src = Rectangle {
                    x: 0.0,
                    y: 0.0,
                    width: img_w,
                    height: img_h,
                };
                let dest = Rectangle {
                    x: (win_w - dw) / 2.0 + pan.x,
                    y: (win_h - dh) / 2.0 + pan.y,
                    width: dw,
                    height: dh,
                };
                d.draw_texture_pro(texture, src, dest, Vector2::ZERO, 0.0, Color::WHITE);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_u8_clamps_and_rounds() {
        assert_eq!(to_u8(0.0), 0);
        assert_eq!(to_u8(1.0), 255);
        assert_eq!(to_u8(-5.0), 0);
        assert_eq!(to_u8(42.0), 255);
        assert_eq!(to_u8(0.5), 128); // 0.5 * 255 + 0.5 = 128
        assert_eq!(to_u8(1.0 / 255.0), 1);
    }

    #[test]
    fn is_jxl_detects_codestream_magic() {
        let dir = std::env::temp_dir();
        let bare = dir.join(format!("vv-test-jxl-{}", std::process::id()));
        // Raw codestream starts with 0xFF 0x0A — JXL even without extension.
        std::fs::write(&bare, [0xffu8, 0x0a, 0x01, 0x02]).unwrap();
        assert!(is_jxl(&bare));
        // Not a codestream.
        std::fs::write(&bare, b"PNG").unwrap();
        assert!(!is_jxl(&bare));
        std::fs::remove_file(&bare).ok();
    }

    #[test]
    fn jxl_extension_is_honored_even_for_bad_content() {
        // Extension decides first; sniffing only runs without a .jxl suffix.
        let path = std::env::temp_dir().join(format!("vv-test-ext-{}.jxl", std::process::id()));
        std::fs::write(&path, b"not really jxl").unwrap();
        assert!(is_jxl(&path));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn decode_common_reads_png() {
        let dir = std::env::temp_dir().join(format!("vv-test-decode-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(3, 2).save(&path).unwrap();
        let decoded = decode_image(&path).unwrap();
        assert_eq!(decoded.width, 3);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.rgba.len(), 3 * 2 * 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decode_image_fails_cleanly_on_garbage() {
        let dir = std::env::temp_dir().join(format!("vv-test-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.png");
        std::fs::write(&path, b"garbage").unwrap();
        assert!(decode_image(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
