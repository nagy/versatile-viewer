//! The content decode of a raster file.
//!
//! Dispatch reads the magic, never the file name: [`sign::detect`] names the
//! decoder, and this module runs it.
//! JPEG XL decodes with jxl-oxide, and every common format decodes with the
//! `image` crate.
//! [`measure_vector`] reads the declared size of a vector file without
//! painting it, so a cold grid cell knows the shape of a picture before its
//! decode lands.
//!
//! Every function here applies [`cap_texture_side`], so the buffer is
//! uploadable as it comes back, and it runs on a decode worker.

use std::path::Path;

use vv_core::dim::{ImgH, ImgW};
use vv_pixels::{DecodeError, DecodedImage, cap_texture_side, rgba_from_samples};

use crate::{sign, svg};

/// Decode a JPEG XL file with jxl-oxide (pure Rust).
///
/// # Errors
///
/// The function errors when the file cannot be opened or mapped, when
/// jxl-oxide rejects the codestream, and when the frame buffer holds an
/// unexpected channel count.
#[must_use = "a decode that nobody reads only costs time"]
pub fn decode_jxl(path: &Path) -> Result<DecodedImage, DecodeError> {
    let image = jxl_oxide::JxlImage::builder().open(path)?;
    let render = image.render_frame(0)?;

    fb_to_rgba(&render.image_all_channels())
}

/// Convert a jxl-oxide framebuffer to RGBA8.
///
/// The framebuffer holds f32 samples and 1 to 4 interleaved channels, and
/// [`rgba_from_samples`] owns the conversion: grayscale replicates into R, G
/// and B, gray plus alpha takes its alpha from channel 1, and RGB or RGBA pass
/// through.
/// The thumbnail cache decodes its own codestream through the same function,
/// so an entry and a file decode one way.
///
/// The function applies [`cap_texture_side`], so the result is uploadable as
/// is, like [`decode_common`].
/// It runs on a decode worker, never on the main thread.
///
/// # Errors
///
/// The function errors when the framebuffer holds an unexpected channel
/// count.
#[must_use = "a decode that nobody reads only costs time"]
pub fn fb_to_rgba(fb: &jxl_oxide::FrameBuffer) -> Result<DecodedImage, DecodeError> {
    rgba_from_samples(
        ImgW::new(fb.width() as u32),
        ImgH::new(fb.height() as u32),
        fb.channels(),
        fb.buf(),
    )
}

/// Decode common formats (PNG, JPEG, ...) with the `image` crate.
///
/// The function applies [`cap_texture_side`], so the result is uploadable as
/// is.
/// This runs on a decode worker, never on the main thread.
///
/// # Errors
///
/// The function errors when the file cannot be opened and when the `image`
/// decoder rejects the bytes.
pub fn decode_common(path: &Path) -> Result<DecodedImage, DecodeError> {
    let rgba = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?
        .into_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(cap_texture_side(DecodedImage::from_exact(
        ImgW::new(width),
        ImgH::new(height),
        rgba.into_raw(),
    )))
}

/// Decode a non-vector file with the decoder that [`sign::detect`] named.
///
/// The function applies [`cap_texture_side`], so the result is uploadable as
/// is.
/// It runs on a decode worker, never on the main thread.
///
/// # Errors
///
/// The function errors when the decoder of the named format rejects the file.
pub fn decode_raster(path: &Path, format: sign::Format) -> Result<DecodedImage, DecodeError> {
    match format {
        crate::sign::Format::Jxl => decode_jxl(path),
        crate::sign::Format::Common => decode_common(path),
    }
}

/// The source size of a vector file, measured without painting it.
///
/// The placeholder pass runs this on a cache miss, so a cold grid cell and the
/// image-view fit know the shape of the picture before its decode lands.
/// A raster format answers None, because its decode reports the size with the
/// pixels.
#[must_use]
pub fn measure_vector(path: &Path) -> Option<(ImgW, ImgH)> {
    if !svg::sniff(path) {
        return None;
    }
    svg::measure(path).ok()
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn fb_to_rgba_replicates_grayscale() {
        // 1 channel (grayscale): the gray sample goes to R, G and B.
        // The code sets alpha = 255.
        let mut fb = jxl_oxide::FrameBuffer::new(2, 1, 1);
        fb.buf_mut()[..2].copy_from_slice(&[0.0, 0.5]);
        let img = fb_to_rgba(&fb).unwrap();
        assert_eq!((img.width(), img.height()), (ImgW::new(2), ImgH::new(1)));
        assert_eq!(img.as_rgba(), [0, 0, 0, 255, 128, 128, 128, 255]);
    }

    #[test]
    fn fb_to_rgba_gray_alpha_takes_alpha() {
        // 2 channels (gray + alpha): the code replicates the gray value and
        // takes alpha from channel 1.
        let mut fb = jxl_oxide::FrameBuffer::new(1, 1, 2);
        fb.buf_mut()[..2].copy_from_slice(&[1.0, 0.5]);
        let img = fb_to_rgba(&fb).unwrap();
        assert_eq!((img.width(), img.height()), (ImgW::new(1), ImgH::new(1)));
        assert_eq!(img.as_rgba(), [255, 255, 255, 128]);
    }

    #[test]
    fn decode_raster_reads_a_png() {
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(3, 2).save(&path).unwrap();
        let decoded = decode_raster(&path, sign::Format::Common).unwrap();
        assert_eq!(decoded.width(), ImgW::new(3));
        assert_eq!(decoded.height(), ImgH::new(2));
        assert_eq!(decoded.as_rgba().len(), 3 * 2 * 4);
    }

    #[test]
    fn decode_raster_reads_webp() {
        // Lossless encode via image-webp, then decode back through
        // decode_raster with the format that the sniff named.
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let path = dir.join("img.webp");
        image::DynamicImage::new_rgb8(5, 4)
            .save_with_format(&path, image::ImageFormat::WebP)
            .unwrap();
        let format = sign::detect(&path).unwrap().raster().unwrap();
        let decoded = decode_raster(&path, format).unwrap();
        assert_eq!(decoded.width(), ImgW::new(5));
        assert_eq!(decoded.height(), ImgH::new(4));
        assert_eq!(decoded.as_rgba().len(), 5 * 4 * 4);
    }

    #[test]
    fn decode_raster_reads_grid_filter_formats() {
        // Every format the grid filter accepts (grid.rs is_image_path) must
        // decode.
        // The filter and the image-crate features must stay in sync (see
        // the Cargo.toml comment).
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
            let decoded = decode_raster(&path, sign::Format::Common)
                .unwrap_or_else(|e| panic!("decode {ext}: {e}"));
            assert_eq!(
                (decoded.width(), decoded.height()),
                (ImgW::new(4), ImgH::new(2)),
                "{ext}"
            );
            assert_eq!(decoded.as_rgba().len(), 4 * 2 * 4, "{ext}");
        }
    }

    #[test]
    fn decode_raster_fails_cleanly_on_garbage() {
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let path = dir.join("x.png");
        std::fs::write(&path, b"garbage").unwrap();
        assert!(decode_raster(&path, sign::Format::Common).is_err());
    }
}
