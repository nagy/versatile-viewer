//! versatile-viewer — step 1: display a single image (JXL first-class, plus
//! PNG/JPEG) in a raylib window. ESC or q quits.
//!
//! Usage: versatile-viewer <image-path>

use std::{env, path::Path};

use anyhow::{Context, Result, bail};
use raylib::{color::Color, consts::PixelFormat, prelude::*};

struct DecodedImage {
    width: u32,
    height: u32,
    /// RGBA8, row-major, 4 bytes per pixel.
    rgba: Vec<u8>,
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
    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        // Fit to window, aspect preserved, centered; never larger than the
        // window in either dimension, and never upscaled.
        let win_w = d.get_screen_width() as f32;
        let win_h = d.get_screen_height() as f32;
        let scale = (win_w / img_w).min(win_h / img_h).min(1.0);
        let dw = img_w * scale;
        let dh = img_h * scale;
        let src = Rectangle {
            x: 0.0,
            y: 0.0,
            width: img_w,
            height: img_h,
        };
        let dest = Rectangle {
            x: (win_w - dw) / 2.0,
            y: (win_h - dh) / 2.0,
            width: dw,
            height: dh,
        };
        d.draw_texture_pro(&texture, src, dest, Vector2::ZERO, 0.0, Color::WHITE);
    }
    Ok(())
}
