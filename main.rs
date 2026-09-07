//! versatile-viewer — step 1: display a single image (JXL first-class, plus
//! PNG/JPEG) in a raylib window. ESC or q quits.
//!
//! Usage: versatile-viewer <image-path>

use std::{env, path::Path};

use anyhow::{Context, Result, bail};
use raylib::{
    color::Color,
    consts::{KeyboardKey, PixelFormat},
    prelude::*,
};

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

    let fb = render.image_all_channels();
    let width = fb.width() as u32;
    let height = fb.height() as u32;
    let channels = fb.channels();
    let samples = fb.buf();
    if !matches!(channels, 3 | 4) {
        bail!("unexpected channel count from jxl-oxide: {channels}");
    }

    // f32 (0..1) -> RGBA8, forcing alpha = 1.0 for opaque 3-channel data.
    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    for (dst, src) in rgba.chunks_exact_mut(4).zip(samples.chunks_exact(channels)) {
        dst[0] = to_u8(src[0]);
        dst[1] = to_u8(src[1]);
        dst[2] = to_u8(src[2]);
        dst[3] = to_u8(src.get(3).copied().unwrap_or(1.0));
    }
    Ok(DecodedImage {
        width,
        height,
        rgba,
    })
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

fn main() -> Result<()> {
    let path = env::args()
        .nth(1)
        .context("usage: versatile-viewer <image-path>")?;
    let path = Path::new(&path);

    let decoded = if is_jxl(path) {
        decode_jxl(path)
    } else {
        decode_common(path)
    }
    .with_context(|| format!("failed to decode {path:?}"))?;

    let (mut rl, thread) = raylib::init()
        .size(decoded.width as i32, decoded.height as i32)
        .title(&format!("versatile-viewer — {}", path.display()))
        .resizable()
        .build();

    // Wrap the raw RGBA8 buffer in a raylib Image without re-encoding it.
    // The ffi::Image only borrows `decoded.rgba`; we must NOT let raylib's
    // UnloadImage free it, so the wrapper is forgotten after the texture
    // upload and the Rust Vec stays the owner.
    let ffi_image = raylib::ffi::Image {
        data: decoded.rgba.as_ptr() as *mut std::os::raw::c_void,
        width: decoded.width as i32,
        height: decoded.height as i32,
        mipmaps: 1,
        format: PixelFormat::PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 as i32,
    };
    let image = unsafe { Image::from_raw(ffi_image) };
    let texture = rl.load_texture_from_image(&thread, &image)?;
    image.to_raw(); // forget: drop would MemFree our borrowed Vec
    drop(decoded.rgba);

    rl.set_target_fps(60);
    let img_w = decoded.width as f32;
    let img_h = decoded.height as f32;
    let mut zoom = ZoomMode::FitDown;
    let mut pan = Vector2::ZERO;
    // On-screen scale, eased toward the target scale each frame.
    let mut view_scale: Option<f32> = None;
    while !rl.window_should_close() {
        let win_w = rl.get_screen_width() as f32;
        let win_h = rl.get_screen_height() as f32;

        // Keyboard shortcuts. Capital W / capital E arrive as W/E + shift.
        let shift = rl.is_key_down(KeyboardKey::KEY_LEFT_SHIFT)
            || rl.is_key_down(KeyboardKey::KEY_RIGHT_SHIFT);
        if rl.is_key_pressed(KeyboardKey::KEY_W) {
            zoom = if shift {
                ZoomMode::FitAll
            } else {
                ZoomMode::FitDown
            };
            pan = Vector2::ZERO;
        } else if rl.is_key_pressed(KeyboardKey::KEY_E) {
            zoom = if shift {
                ZoomMode::FitHeight
            } else {
                ZoomMode::FitWidth
            };
            pan = Vector2::ZERO;
        }

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
        let mut zoom_in =
            rl.is_key_pressed(KeyboardKey::KEY_EQUAL) || rl.is_key_pressed(KeyboardKey::KEY_KP_ADD);
        let mut zoom_out = rl.is_key_pressed(KeyboardKey::KEY_MINUS)
            || rl.is_key_pressed(KeyboardKey::KEY_KP_SUBTRACT);
        // Drain the character queue so repeats don't pile up.
        loop {
            match rl.get_char_pressed() {
                None => break,
                Some('+') => zoom_in = true,
                Some('-') => zoom_out = true,
                _ => {}
            }
        }
        if zoom_in || zoom_out {
            let factor = if zoom_in { 1.25 } else { 1.0 / 1.25 };
            zoom = ZoomMode::Free((target_scale * factor).clamp(0.01, 100.0));
        }
        let scale = view_scale.unwrap();

        // Vim-style panning (h/j/k/l + arrow keys); held keys scroll
        // continuously. Input read before begin_drawing borrows rl mutably.
        let win_w = rl.get_screen_width() as f32;
        let win_h = rl.get_screen_height() as f32;
        let step = win_w.max(win_h) * 0.03;
        let pan_left = rl.is_key_down(KeyboardKey::KEY_H) || rl.is_key_down(KeyboardKey::KEY_LEFT);
        let pan_right =
            rl.is_key_down(KeyboardKey::KEY_L) || rl.is_key_down(KeyboardKey::KEY_RIGHT);
        let pan_up = rl.is_key_down(KeyboardKey::KEY_K) || rl.is_key_down(KeyboardKey::KEY_UP);
        let pan_down = rl.is_key_down(KeyboardKey::KEY_J) || rl.is_key_down(KeyboardKey::KEY_DOWN);
        if pan_left {
            pan.x += step;
        }
        if pan_right {
            pan.x -= step;
        }
        if pan_up {
            pan.y += step;
        }
        if pan_down {
            pan.y -= step;
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        let dw = img_w * scale;
        let dh = img_h * scale;

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
        d.draw_texture_pro(&texture, src, dest, Vector2::ZERO, 0.0, Color::WHITE);
    }
    Ok(())
}
