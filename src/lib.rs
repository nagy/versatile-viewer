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

//! versatile-viewer's reusable core: image decoding, RGBA helpers and the
//! modules behind the grid, loader and gimmicks.
//!
//! Decoding dispatches by sniffed content, not file name: JPEG XL (pure-Rust
//! jxl-oxide, with progressive streaming support in [`loader`]) vs the common
//! formats via the `image` crate. [`downscale_rgba`] and [`upload_rgba`] are
//! the shared pixel/texture choke points. The binary in `main.rs` wires these
//! pieces into the window and event loop; unit tests and doctests here keep
//! the pure helpers honest.

pub mod blurbg;
pub mod grid;
pub mod keyrepeat;
pub mod loader;
pub mod map;
#[cfg(target_os = "linux")]
pub mod wmclass;

use std::path::Path;

use anyhow::{Context, Result, bail};
use raylib::{consts::PixelFormat, prelude::*};

/// A fully decoded image: RGBA8 plus its pixel dimensions.
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, 4 bytes per pixel.
    pub rgba: Vec<u8>,
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

/// Convert a jxl-oxide framebuffer (f32 samples, 1–4 interleaved channels)
/// to RGBA8.
///
/// Grayscale (1 channel) is replicated into R/G/B; gray+alpha (2 channels)
/// additionally takes alpha from channel 1. RGB (3 channels) gets alpha = 1,
/// RGBA (4 channels) is taken as-is.
pub(crate) fn fb_to_rgba(fb: &jxl_oxide::FrameBuffer) -> Result<(Vec<u8>, u32, u32)> {
    let width = fb.width() as u32;
    let height = fb.height() as u32;
    let channels = fb.channels();
    let samples = fb.buf();
    if !matches!(channels, 1..=4) {
        bail!("unexpected channel count from jxl-oxide: {channels}");
    }

    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    for (dst, src) in rgba.chunks_exact_mut(4).zip(samples.chunks_exact(channels)) {
        let g = to_u8(src[0]);
        dst[0] = g;
        dst[1] = if channels >= 3 { to_u8(src[1]) } else { g };
        dst[2] = if channels >= 3 { to_u8(src[2]) } else { g };
        dst[3] = if channels == 2 || channels == 4 {
            to_u8(src[channels - 1])
        } else {
            255
        };
    }
    Ok((rgba, width, height))
}

const fn to_u8(v: f32) -> u8 {
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

/// Decide the decoder by file content, not the file name.
///
/// The 2-byte magic is sniffed first (JXL codestream vs the common formats),
/// and the .jxl extension only acts as a tiebreaker for unknown magic (JXL
/// container files start with a box header, not the codestream magic).
fn is_jxl(path: &Path) -> bool {
    match file_magic(path) {
        Some([0xff, 0x0a]) => true, // raw JXL codestream
        Some(m) if is_common_magic(m) => false,
        _ => path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("jxl")),
    }
}

/// First two bytes of the file; None on a missing/short file.
fn file_magic(path: &Path) -> Option<[u8; 2]> {
    use std::io::Read;
    let mut magic = [0u8; 2];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .ok()
        .map(|()| magic)
}

/// Magics the image crate can decode (`with_guessed_format` sniffs the full
/// header; this only needs to steer files away from the JXL decoder).
// The magic table is kept flat and grouped by format on purpose; clippy's
// nested suggestion reorders it into a byte soup.
#[allow(clippy::unnested_or_patterns)]
const fn is_common_magic(m: [u8; 2]) -> bool {
    let [a, b] = m;
    matches!(
        (a, b),
        (0x89, b'P') // PNG
            | (0xff, 0xd8) // JPEG
            | (b'R', b'I') // RIFF (WebP)
            | (b'G', b'I') // GIF
            | (b'B', b'M') // BMP
            | (b'I', b'I')
            | (b'I', b'M') // TIFF, both byte orders
            | (b'M', b'I')
            | (b'M', b'M')
    )
}

/// Decode an image file, dispatching by content-sniffed format.
///
/// # Errors
///
/// Errors when the file cannot be opened, sniffed or decoded (missing
/// file, garbage bytes, unsupported content).
pub fn decode_image(path: &Path) -> Result<DecodedImage> {
    if is_jxl(path) {
        decode_jxl(path)
    } else {
        decode_common(path)
    }
    .with_context(|| format!("failed to decode {}", path.display()))
}

/// Longest texture side we upload. Desktop GL hardware ranges from 4096
/// to 16384; this is the safe middle (raylib does not expose the real
/// limit). Anything larger is downscaled here instead of failing the load.
const MAX_TEXTURE_SIDE: u32 = 8192;

/// Downscale an RGBA8 buffer so its long side is at most `long_side`.
///
/// Below the cap the buffer is returned unchanged — never upscaled. Zero
/// dimensions are clamped to 1, and the output aspect matches the input.
///
/// # Examples
///
/// ```
/// # use versatile_viewer::downscale_rgba;
/// // Below the cap: returned unchanged.
/// let (rgba, w, h) = downscale_rgba(vec![0; 4 * 2 * 2], 2, 2, 8);
/// assert_eq!((w, h), (2, 2));
/// assert_eq!(rgba.len(), 4 * 2 * 2);
///
/// // Over it: aspect preserved, long side exactly the cap.
/// let (rgba, w, h) = downscale_rgba(vec![0; 4 * 200 * 100], 200, 100, 100);
/// assert_eq!((w, h), (100, 50));
/// assert_eq!(rgba.len(), 4 * 100 * 50);
///
/// // Degenerate zero dimensions are clamped, the buffer untouched.
/// let (rgba, w, h) = downscale_rgba(Vec::new(), 0, 0, 8);
/// assert_eq!((w, h), (1, 1));
/// assert!(rgba.is_empty());
/// ```
///
/// # Panics
///
/// Panics if `rgba` does not hold exactly `width * height * 4` bytes
/// (and a downscale is actually needed).
#[must_use]
pub fn downscale_rgba(
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    long_side: u32,
) -> (Vec<u8>, u32, u32) {
    let (width, height) = (width.max(1), height.max(1));
    let m = width.max(height);
    if m <= long_side.max(1) {
        return (rgba, width, height);
    }
    let scale = long_side.max(1) as f32 / m as f32;
    let nw = ((width as f32 * scale).round() as u32).max(1);
    let nh = ((height as f32 * scale).round() as u32).max(1);
    let img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(width, height, rgba).expect("rgba matches dimensions");
    let small = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    (small.into_raw(), nw, nh)
}

/// Upload a raw RGBA8 buffer as a GPU texture.
///
/// Must be called on the main thread (GL context lives there). The buffer
/// is only borrowed for the upload; the `ffi::Image` wrapper is forgotten so
/// raylib never frees the caller's Vec.
///
/// # Errors
///
/// Errors when the GL texture upload fails (no current context, driver
/// refusal).
pub fn upload_rgba(
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

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    // Compile-time check that this stays const-evaluable.
    const _: u8 = to_u8(0.5);

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
        let tmp = tempfile::TempDir::new().unwrap();
        let bare = tmp.path().join("bare");
        // Raw codestream starts with 0xFF 0x0A — JXL even without extension.
        std::fs::write(&bare, [0xffu8, 0x0a, 0x01, 0x02]).unwrap();
        assert!(is_jxl(&bare));
        // PNG magic — not a codestream, even without an extension.
        std::fs::write(&bare, [0x89u8, b'P', 0x4e, 0x47]).unwrap();
        assert!(!is_jxl(&bare));
    }

    #[test]
    fn content_decides_over_extension() {
        // Content-first dispatch: a PNG renamed to .jxl decodes as PNG, a
        // JXL codestream renamed to .png is still JXL. Unknown magic with a
        // .jxl suffix (e.g. a container-format file) falls back to JXL.
        let dir = TempDir::new().unwrap();
        let dir = dir.path();

        let png = dir.join("real.png");
        image::DynamicImage::new_rgb8(2, 3).save(&png).unwrap();
        let renamed_jxl = dir.join("renamed.jxl");
        std::fs::copy(&png, &renamed_jxl).unwrap();
        assert!(!is_jxl(&renamed_jxl));
        let decoded = decode_image(&renamed_jxl).unwrap();
        assert_eq!((decoded.width, decoded.height), (2, 3));

        let renamed_png = dir.join("renamed.png");
        std::fs::write(&renamed_png, [0xffu8, 0x0a, 0x01, 0x02]).unwrap();
        assert!(is_jxl(&renamed_png));

        let container = dir.join("container.jxl");
        std::fs::write(&container, [0x00, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ']).unwrap();
        assert!(is_jxl(&container));
    }

    #[test]
    fn fb_to_rgba_replicates_grayscale() {
        // 1 channel (grayscale): gray sample goes to R, G and B; alpha = 255.
        let mut fb = jxl_oxide::FrameBuffer::new(2, 1, 1);
        fb.buf_mut()[..2].copy_from_slice(&[0.0, 0.5]);
        let (rgba, w, h) = fb_to_rgba(&fb).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(rgba, [0, 0, 0, 255, 128, 128, 128, 255]);
    }

    #[test]
    fn fb_to_rgba_gray_alpha_takes_alpha() {
        // 2 channels (gray + alpha): gray replicated, alpha from channel 1.
        let mut fb = jxl_oxide::FrameBuffer::new(1, 1, 2);
        fb.buf_mut()[..2].copy_from_slice(&[1.0, 0.5]);
        let (rgba, w, h) = fb_to_rgba(&fb).unwrap();
        assert_eq!((w, h), (1, 1));
        assert_eq!(rgba, [255, 255, 255, 128]);
    }

    #[test]
    fn decode_common_reads_png() {
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(3, 2).save(&path).unwrap();
        let decoded = decode_image(&path).unwrap();
        assert_eq!(decoded.width, 3);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.rgba.len(), 3 * 2 * 4);
    }

    #[test]
    fn decode_image_reads_webp() {
        // Lossless encode via image-webp, then decode back through
        // decode_image (format sniffed from bytes, not extension).
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let path = dir.join("img.webp");
        image::DynamicImage::new_rgb8(5, 4)
            .save_with_format(&path, image::ImageFormat::WebP)
            .unwrap();
        let decoded = decode_image(&path).unwrap();
        assert_eq!(decoded.width, 5);
        assert_eq!(decoded.height, 4);
        assert_eq!(decoded.rgba.len(), 5 * 4 * 4);
    }

    #[test]
    fn decode_image_reads_grid_filter_formats() {
        // Every format the grid filter accepts (grid.rs is_image_path) must
        // actually decode — the filter and the image-crate features must stay
        // in sync (see the Cargo.toml comment).
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        for (ext, format) in [
            ("bmp", image::ImageFormat::Bmp),
            ("gif", image::ImageFormat::Gif),
            ("tga", image::ImageFormat::Tga),
            ("tif", image::ImageFormat::Tiff),
        ] {
            let path = dir.join(format!("img.{ext}"));
            image::DynamicImage::new_rgb8(4, 2)
                .save_with_format(&path, format)
                .unwrap_or_else(|e| panic!("encode {ext}: {e}"));
            let decoded = decode_image(&path).unwrap_or_else(|e| panic!("decode {ext}: {e}"));
            assert_eq!((decoded.width, decoded.height), (4, 2), "{ext}");
            assert_eq!(decoded.rgba.len(), 4 * 2 * 4, "{ext}");
        }
    }

    #[test]
    fn decode_image_fails_cleanly_on_garbage() {
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let path = dir.join("x.png");
        std::fs::write(&path, b"garbage").unwrap();
        assert!(decode_image(&path).is_err());
    }
}
