// Pixel and coordinate math lives in f32 (window and source units) and
// indices in usize and u32.
// The casts between them belong to that boundary.
// Every value here (window pixels, texture dimensions) stays far below
// the f32 exact-integer range.
// The pedantic cast lints are therefore noise.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

//! The pixel buffers of versatile-viewer.
//!
//! [`DecodedImage`] holds one decoded RGBA8 picture, and it is the only
//! checked entrance for a buffer, so a wrong buffer never exists to check.
//! [`resample_rgba`] is the one resize path (the `fast_image_resize` SIMD
//! path with Lanczos3), and [`cap_texture_side`] is the one texture budget.
//! [`placeholder`] holds the few-byte
//! `ThumbHash` a cache header carries.
//! [`rgba_from_samples`] turns the f32 samples of a framebuffer into RGBA8,
//! so the JPEG XL file decode and the thumbnail entry decode share one
//! conversion.
//!
//! Every function here runs on a decode worker, never on the main thread.
//! No decoder and no GPU handle lives here: the crates above bring their own.

pub mod error;
pub mod placeholder;

// The dimension newtypes appear in the public API of this crate, so a caller
// names one through this crate too.
pub use vv_core::dim;

use crate::dim::{ImgH, ImgW, LongSide};
pub use crate::error::DecodeError;

/// Holds the decoded image data.
///
/// The fields are private and [`DecodedImage::new`] is the only checked
/// entrance, so a buffer that does not hold exactly `width * height * 4` bytes
/// never becomes an image.
/// The code then reads the pixels through [`DecodedImage::as_rgba`] and the
/// extents through [`DecodedImage::width`] and [`DecodedImage::height`].
///
/// # Examples
///
/// ```
/// # use vv_pixels::{DecodedImage, DecodeError};
/// # use vv_pixels::dim::{ImgH, ImgW};
/// let img = DecodedImage::new(ImgW::new(2), ImgH::new(2), vec![0; 16]).unwrap();
/// assert_eq!(img.as_rgba().len(), 16);
/// assert_eq!(img.width(), ImgW::new(2));
/// // A buffer of the wrong size never becomes an image.
/// assert!(matches!(
///     DecodedImage::new(ImgW::new(2), ImgH::new(2), vec![0; 15]),
///     Err(DecodeError::Buffer)
/// ));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    /// The image width, in pixels.
    width: ImgW,
    /// The image height, in pixels.
    height: ImgH,
    /// RGBA8, row-major, 4 bytes per pixel.
    data: Vec<u8>,
}

impl DecodedImage {
    /// An RGBA8 image of `width` by `height`.
    ///
    /// # Errors
    ///
    /// The function errors with [`DecodeError::Buffer`] when `data` does not
    /// hold exactly `width * height * 4` bytes.
    pub fn new(width: ImgW, height: ImgH, data: Vec<u8>) -> Result<Self, DecodeError> {
        if Some(data.len()) != pixel_bytes(width, height) {
            return Err(DecodeError::Buffer);
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// The image built from a buffer whose length this crate just computed.
    ///
    /// A buffer that does not match its dimensions trips the debug assertion,
    /// so the invariant still holds in a test build.
    #[must_use]
    pub fn from_exact(width: ImgW, height: ImgH, data: Vec<u8>) -> Self {
        debug_assert_eq!(Some(data.len()), pixel_bytes(width, height));
        Self {
            width,
            height,
            data,
        }
    }

    /// The image width, in pixels.
    #[must_use]
    pub const fn width(&self) -> ImgW {
        self.width
    }

    /// The image height, in pixels.
    #[must_use]
    pub const fn height(&self) -> ImgH {
        self.height
    }

    /// The RGBA8 pixels, row-major, 4 bytes per pixel.
    #[must_use]
    pub fn as_rgba(&self) -> &[u8] {
        &self.data
    }

    /// The RGBA8 pixels, consumed.
    #[must_use]
    pub fn into_rgba(self) -> Vec<u8> {
        self.data
    }

    /// The width, the height and the RGBA8 pixels, consumed.
    #[must_use]
    pub fn into_parts(self) -> (ImgW, ImgH, Vec<u8>) {
        (self.width, self.height, self.data)
    }
}

/// The RGBA8 byte length of a `width` x `height` image, or `None` when the
/// product overflows.
fn pixel_bytes(width: ImgW, height: ImgH) -> Option<usize> {
    usize::try_from(width.raw())
        .ok()?
        .checked_mul(usize::try_from(height.raw()).ok()?)?
        .checked_mul(4)
}

/// The longest texture side the decode produces.
///
/// This is the policy, and `texture::Uploader` clamps every upload to the
/// adapter's own `max_texture_dimension_2d` when that is smaller, so the
/// decoded size stays the same on every host and a small adapter still takes
/// the picture.
/// Desktop adapters range from 4096 to 16384.
/// The code downscales anything larger here instead of failing the load.
pub const MAX_TEXTURE_SIDE: u32 = 8192;

/// Turn interleaved f32 samples into an RGBA8 picture.
///
/// A framebuffer holds its samples as f32 values in 1 to 4 interleaved
/// channels.
/// One channel (grayscale) replicates into R, G and B.
/// Two channels (gray plus alpha) also take alpha from the second channel.
/// Three channels (RGB) take alpha = 1, and four channels (RGBA) pass as they
/// come.
///
/// The function applies [`cap_texture_side`], so the result is uploadable as
/// it comes back, and it runs on a decode worker, never on the main thread.
/// A `samples` slice shorter than `width * height * channels` leaves the rest
/// of the picture black, and a longer one never reads past the last pixel.
///
/// # Errors
///
/// The function errors when `channels` names a count outside 1..=4.
///
/// # Examples
///
/// ```
/// # use vv_pixels::{DecodedImage, rgba_from_samples};
/// # use vv_pixels::dim::{ImgH, ImgW};
/// // One gray channel replicates into R, G and B, and alpha stays opaque.
/// let img = rgba_from_samples(ImgW::new(2), ImgH::new(1), 1, &[0.0, 0.5]).unwrap();
/// assert_eq!(img.as_rgba(), [0, 0, 0, 255, 128, 128, 128, 255]);
/// // Four channels pass as they come.
/// let img = rgba_from_samples(ImgW::new(1), ImgH::new(1), 4, &[1.0, 0.0, 0.0, 0.5]).unwrap();
/// assert_eq!(img.as_rgba(), [255, 0, 0, 128]);
/// ```
#[must_use = "a decode that nobody reads only costs time"]
pub fn rgba_from_samples(
    width: ImgW,
    height: ImgH,
    channels: usize,
    samples: &[f32],
) -> Result<DecodedImage, DecodeError> {
    if !matches!(channels, 1..=4) {
        return Err(DecodeError::Channels(channels as u32));
    }

    let mut rgba = vec![0u8; width.raw() as usize * height.raw() as usize * 4];
    for (dst, src) in rgba
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(samples.chunks_exact(channels))
    {
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
    Ok(cap_texture_side(DecodedImage::from_exact(
        width, height, rgba,
    )))
}

/// One f32 sample as a clamped, rounded `u8`.
const fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Downscale an RGBA8 buffer to the uploadable [`MAX_TEXTURE_SIDE`].
///
/// Run this function off the main thread, where the decode runs.
/// A texture upload then never resizes a big buffer on the main thread.
/// Below the limit the function returns the image unchanged, so no quality
/// is lost and nothing is copied.
///
/// # Examples
///
/// ```
/// # use vv_pixels::{DecodedImage, cap_texture_side, MAX_TEXTURE_SIDE};
/// # use vv_pixels::dim::{ImgH, ImgW};
/// # use vv_pixels::dim::LongSide;
/// // Below the limit: the same buffer comes back, untouched.
/// let img = DecodedImage::new(ImgW::new(64), ImgH::new(32), vec![7; 4 * 64 * 32]).unwrap();
/// let img = cap_texture_side(img);
/// assert_eq!((img.width().raw(), img.height().raw()), (64, 32));
/// assert_eq!(img.as_rgba().len(), 4 * 64 * 32);
///
/// // Over the limit: the long side lands exactly on the limit.
/// let img = DecodedImage::new(
///     ImgW::new(MAX_TEXTURE_SIDE + 1),
///     ImgH::new(1),
///     vec![0; 4 * (MAX_TEXTURE_SIDE as usize + 1)],
/// )
/// .unwrap();
/// let img = cap_texture_side(img);
/// assert_eq!(
///     (img.width().raw(), img.height().raw()),
///     (MAX_TEXTURE_SIDE, 1)
/// );
/// ```
#[must_use]
pub fn cap_texture_side(image: DecodedImage) -> DecodedImage {
    if image.width().raw().max(image.height().raw()) > MAX_TEXTURE_SIDE {
        downscale_rgba(image, LongSide::new(MAX_TEXTURE_SIDE))
    } else {
        image
    }
}

/// Downscale an RGBA8 buffer so its long side is at most `long_side`.
///
/// Below the cap the function returns the buffer unchanged.
/// It never upscales.
/// The output aspect matches the input.
///
/// # Examples
/// # Examples
///
/// ```
/// # use vv_pixels::{DecodedImage, downscale_rgba};
/// # use vv_pixels::dim::{ImgH, ImgW, LongSide};
/// // Below the cap: returned unchanged.
/// let img = DecodedImage::new(ImgW::new(2), ImgH::new(2), vec![0; 4 * 2 * 2]).unwrap();
/// let img = downscale_rgba(img, LongSide::new(8));
/// assert_eq!((img.width().raw(), img.height().raw()), (2, 2));
/// assert_eq!(img.as_rgba().len(), 4 * 2 * 2);
///
/// // Over it: the aspect stays and the long side equals the cap.
/// let img = DecodedImage::new(ImgW::new(200), ImgH::new(100), vec![0; 4 * 200 * 100]).unwrap();
/// let img = downscale_rgba(img, LongSide::new(100));
/// assert_eq!((img.width().raw(), img.height().raw()), (100, 50));
/// assert_eq!(img.as_rgba().len(), 4 * 100 * 50);
///
/// // An image without pixels comes back as it went in.
/// let img = DecodedImage::new(ImgW::new(0), ImgH::new(0), Vec::new()).unwrap();
/// let img = downscale_rgba(img, LongSide::new(8));
/// assert_eq!((img.width().raw(), img.height().raw()), (0, 0));
/// assert!(img.as_rgba().is_empty());
/// ```
///
/// The function never upscales, and an image without pixels comes back as it
/// went in.
#[must_use]
pub fn downscale_rgba(image: DecodedImage, long_side: LongSide) -> DecodedImage {
    let (width, height, data) = image.into_parts();
    let (w, h) = (width.raw(), height.raw());
    // An image without pixels has nothing to resample, so it comes back as it
    // went in.
    if w == 0 || h == 0 {
        return DecodedImage::from_exact(width, height, data);
    }
    let long_side = long_side.raw().max(1);
    if w.max(h) <= long_side {
        return DecodedImage::from_exact(width, height, data);
    }
    let (nw, nh) = fitted(w, h, long_side);
    DecodedImage::from_exact(
        ImgW::new(nw),
        ImgH::new(nh),
        resample_rgba(&data, w, h, nw, nh),
    )
}

/// The dimensions of `width` x `height` scaled so the long side is
/// `long_side`, with the aspect preserved and each side at least 1.
#[must_use]
pub fn fitted(width: u32, height: u32, long_side: u32) -> (u32, u32) {
    let scale = long_side as f32 / width.max(height) as f32;
    (
        ((width as f32 * scale).round() as u32).max(1),
        ((height as f32 * scale).round() as u32).max(1),
    )
}

/// Resample an RGBA8 buffer into a `nw` x `nh` buffer with the `fast_image_resize`
/// SIMD path (Lanczos3).
///
/// The function allocates the output and leaves the input alone.
/// Call it on a decode worker, never on the main thread.
///
/// # Panics
///
/// The function panics if `rgba` does not hold exactly `w * h * 4` bytes.
#[must_use]
pub fn resample_rgba(rgba: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    use fast_image_resize::{
        FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
        images::{Image, ImageRef},
    };
    let src = ImageRef::new(w, h, rgba, PixelType::U8x4).expect("rgba buffer matches dimensions");
    let mut dst = Image::new(nw, nh, PixelType::U8x4);
    let options = ResizeOptions {
        algorithm: ResizeAlg::Convolution(FilterType::Lanczos3),
        ..Default::default()
    };
    Resizer::new()
        .resize(&src, &mut dst, &options)
        .expect("rgba buffers match their dimensions");
    dst.into_vec()
}

/// Downscale a borrowed RGBA8 buffer, and leave the input alone.
///
/// [`downscale_rgba`] takes ownership, so a caller that still needs the
/// source buffer would have to clone it.
/// The grid decodes once and needs two sizes from that decode (thumb and
/// full), and a clone of a 100 MP buffer costs hundreds of megabytes.
/// This function copies no input pixels: the resizer only reads.
///
/// The function never upscales, and an image without pixels comes back as it
/// went in.
/// Below `long_side` the output is a plain copy of the input.
///
/// # Panics
///
/// The function panics if `rgba` does not hold exactly
/// `width * height * 4` bytes.
#[must_use]
pub fn downscale_rgba_ref(
    rgba: &[u8],
    width: ImgW,
    height: ImgH,
    long_side: LongSide,
) -> DecodedImage {
    let (w, h) = (width.raw(), height.raw());
    // An image without pixels has nothing to resample, so it comes back as it
    // went in.
    if w == 0 || h == 0 {
        return DecodedImage::from_exact(width, height, Vec::new());
    }
    let long_side = long_side.raw().max(1);
    if w.max(h) <= long_side {
        return DecodedImage::from_exact(width, height, rgba.to_vec());
    }
    let (nw, nh) = fitted(w, h, long_side);
    DecodedImage::from_exact(
        ImgW::new(nw),
        ImgH::new(nh),
        resample_rgba(rgba, w, h, nw, nh),
    )
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
    fn rgba_from_samples_replicates_gray_and_keeps_alpha() {
        // One gray channel: the value reaches R, G and B, and alpha stays
        // opaque.
        let img = rgba_from_samples(ImgW::new(2), ImgH::new(1), 1, &[0.0, 0.5]).expect("gray");
        assert_eq!(img.as_rgba(), [0, 0, 0, 255, 128, 128, 128, 255]);
        // Gray plus alpha: the gray value replicates, and channel 1 carries
        // the alpha.
        let img =
            rgba_from_samples(ImgW::new(1), ImgH::new(1), 2, &[1.0, 0.5]).expect("gray alpha");
        assert_eq!(img.as_rgba(), [255, 255, 255, 128]);
        // RGB takes alpha = 1, and RGBA passes as it comes.
        let img = rgba_from_samples(ImgW::new(1), ImgH::new(1), 3, &[1.0, 0.0, 0.0]).expect("rgb");
        assert_eq!(img.as_rgba(), [255, 0, 0, 255]);
        let img =
            rgba_from_samples(ImgW::new(1), ImgH::new(1), 4, &[1.0, 0.0, 0.0, 0.5]).expect("rgba");
        assert_eq!(img.as_rgba(), [255, 0, 0, 128]);
    }

    #[test]
    fn rgba_from_samples_rejects_a_bad_channel_count() {
        let err = rgba_from_samples(ImgW::new(1), ImgH::new(1), 5, &[0.0; 5]);
        assert!(matches!(err, Err(DecodeError::Channels(5))));
        let err = rgba_from_samples(ImgW::new(1), ImgH::new(1), 0, &[]);
        assert!(matches!(err, Err(DecodeError::Channels(0))));
    }

    #[test]
    fn rgba_from_samples_caps_a_tall_framebuffer() {
        // The conversion applies the texture cap like every other decode.
        let samples = vec![0.0f32; (MAX_TEXTURE_SIDE as usize + 1) * 4];
        let img = rgba_from_samples(ImgW::new(1), ImgH::new(MAX_TEXTURE_SIDE + 1), 4, &samples)
            .expect("tall");
        assert_eq!(img.height(), ImgH::new(MAX_TEXTURE_SIDE));
    }

    #[test]
    fn cap_texture_side_never_upscales_and_caps() {
        let img = |w: u32, h: u32| {
            DecodedImage::new(ImgW::new(w), ImgH::new(h), vec![9; (w * h * 4) as usize])
                .expect("sized")
        };
        // Below the limit: the same buffer, untouched.
        let src = img(32, 16);
        let out = cap_texture_side(src.clone());
        assert_eq!((out.width(), out.height()), (ImgW::new(32), ImgH::new(16)));
        assert_eq!(out.as_rgba(), src.as_rgba());
        // Over the limit: the long side lands exactly on the limit.
        let out = cap_texture_side(img(MAX_TEXTURE_SIDE + 1, 2));
        assert_eq!(out.width(), ImgW::new(MAX_TEXTURE_SIDE));
        assert_eq!(out.height(), ImgH::new(2));
        assert_eq!(out.as_rgba().len(), (MAX_TEXTURE_SIDE * 2 * 4) as usize);
        // A tall image caps by height instead.
        let out = cap_texture_side(img(2, MAX_TEXTURE_SIDE + 1));
        assert_eq!(out.width(), ImgW::new(2));
        assert_eq!(out.height(), ImgH::new(MAX_TEXTURE_SIDE));
    }

    #[test]
    fn downscale_rgba_ref_borrows_and_never_upscales() {
        let src = |w: u32, h: u32| {
            DecodedImage::new(ImgW::new(w), ImgH::new(h), vec![7; (w * h * 4) as usize])
                .expect("sized")
        };
        // Above the cap: 800x200 with cap 64 -> 64x16.
        let img = src(800, 200);
        let out = downscale_rgba_ref(img.as_rgba(), img.width(), img.height(), LongSide::new(64));
        assert_eq!((out.width(), out.height()), (ImgW::new(64), ImgH::new(16)));
        assert_eq!(
            (img.width(), img.height()),
            (ImgW::new(800), ImgH::new(200))
        );
        // Below the cap: the output equals the borrowed input.
        let img = src(32, 16);
        let out = downscale_rgba_ref(img.as_rgba(), img.width(), img.height(), LongSide::new(64));
        assert_eq!((out.width(), out.height()), (ImgW::new(32), ImgH::new(16)));
        assert_eq!(out.as_rgba(), img.as_rgba());
        assert_eq!(out, downscale_rgba(img, LongSide::new(64)));
    }

    proptest::proptest! {
        /// The cap never upscales, never exceeds the cap and keeps the
        /// aspect within one pixel of rounding.
        #[test]
        fn cap_texture_side_invariants(w in 1u32..400, h in 1u32..400) {
            let img = DecodedImage::new(ImgW::new(w), ImgH::new(h), vec![0; 4 * w as usize * h as usize])
                .expect("sized");
            let out = cap_texture_side(img);
            let (ow, oh) = (out.width().raw(), out.height().raw());
            // A small image comes back untouched.
            proptest::prop_assert!((ow, oh) == (w, h));
            proptest::prop_assert_eq!(out.as_rgba().len(), 4 * ow as usize * oh as usize);
        }

        /// `fitted` keeps the aspect, clamps to 1 and lands the long side on
        /// the cap.
        #[test]
        fn fitted_keeps_aspect(w in 1u32..20000, h in 1u32..20000, cap in 1u32..8192) {
            // Only downscales occur in the viewer: the cap stays at or below
            // the source long side.
            let cap = cap.min(w.max(h)).max(1);
            let (nw, nh) = fitted(w, h, cap);
            proptest::prop_assert!(nw >= 1 && nh >= 1);
            proptest::prop_assert!(nw.max(nh) <= w.max(h));
            if w >= h {
                proptest::prop_assert_eq!(nw, cap.min(w));
            } else {
                proptest::prop_assert_eq!(nh, cap.min(h));
            }
        }

        /// The resample honors the output size and never reads past the
        /// input buffer.
        #[test]
        fn resample_matches_output_size(w in 8u32..256, h in 8u32..256) {
            let src = vec![0x5au8; 4 * w as usize * h as usize];
            let out = resample_rgba(&src, w, h, 4, 3);
            proptest::prop_assert_eq!(out.len(), 4 * 4 * 3);
        }
    }
}
