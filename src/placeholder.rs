//! The few-byte placeholder of a grid thumbnail (`ThumbHash`).
//!
//! A `ThumbHash` holds the low-frequency half of a DCT of a thumbnail: a
//! luminance plane, a coarser colour plane and an optional alpha plane.
//! The payload is 5 to 32 bytes, so it fits in the fixed header of a cache
//! entry (see [`crate::thumbcache`]).
//! A warm launch therefore paints every cell from a header read of 81 bytes
//! and a DCT render of 32 px, instead of a JPEG XL decode of 512 px.
//!
//! The format comes from Evan Wallace: <https://evanw.github.io/thumbhash/>.
//! The crate is `thumbhash` (MIT, pure Rust, no dependencies).
//!
//! Encode and render both run on a decode worker, never on the main thread.
//! The main thread uploads the rendered texture.
//!
//! The placeholder is a cosmetic hint, so every failure answers `None` and
//! the grid keeps its plain grey cell.
//! A cold launch has no cache entry, so it has no placeholder either.

use crate::{
    DecodedImage,
    dim::{ImgH, ImgW, LongSide},
};

/// The largest `ThumbHash` the format defines.
/// The encoder produces at most 32 bytes, and a longer payload means a
/// corrupt entry.
pub const MAX_BYTES: usize = 32;

/// The long side of the rendered placeholder texture.
///
/// The DCT render lands at about 32 px, and the GPU scales it to a whole grid
/// cell.
/// A Lanczos upscale to this size keeps the bilinear filter from showing the
/// block grid of the raw render, and a 64 px RGBA texture still costs only
/// 16 KB.
pub const RENDER_LONG_SIDE: LongSide = LongSide::new(64);

/// The long side of the encoder input.
///
/// The `ThumbHash` encoder asserts an input of 100 px or less per side, so the
/// code downscales the thumb first.
/// Beyond 100 px the encoder adds no detail.
const ENCODE_LONG_SIDE: LongSide = LongSide::new(100);

/// The placeholder payload of one image.
///
/// The type stores a fixed array, so it stays `Copy` and fits in the cache
/// header without a heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placeholder {
    /// The number of meaningful bytes in `bytes`.
    len: u8,
    /// The `ThumbHash`, padded with zeros past `len`.
    bytes: [u8; MAX_BYTES],
}

impl Placeholder {
    /// A placeholder from a stored payload.
    ///
    /// An empty payload (no placeholder) and an over-long payload (a corrupt
    /// entry) both answer `None`.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let len = u8::try_from(bytes.len()).ok().filter(|n| {
            let n = usize::from(*n);
            n > 0 && n <= MAX_BYTES
        })?;
        let mut out = [0u8; MAX_BYTES];
        out.get_mut(..bytes.len())?.copy_from_slice(bytes);
        Some(Self { len, bytes: out })
    }

    /// The stored payload.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.get(..usize::from(self.len)).unwrap_or_default()
    }

    /// The payload length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        usize::from(self.len)
    }

    /// A placeholder always holds at least one byte, so this answers false.
    /// It exists beside [`Self::len`], as the standard library pairing asks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Encode the placeholder of `image`.
    ///
    /// The function downscales the input to [`ENCODE_LONG_SIDE`] first, which
    /// the encoder requires and which costs one SIMD resize.
    /// Run this on a decode worker.
    /// A payload that exceeds [`MAX_BYTES`] answers `None`: the cache entry
    /// then carries no placeholder, and the grid keeps its grey cell.
    #[must_use]
    pub fn encode(image: &DecodedImage) -> Option<Self> {
        let small =
            crate::downscale_rgba_ref(&image.data, image.width, image.height, ENCODE_LONG_SIDE);
        let hash = thumbhash::rgba_to_thumb_hash(
            small.width.get() as usize,
            small.height.get() as usize,
            &small.data,
        );
        Self::from_bytes(&hash)
    }

    /// Render the placeholder into a small RGBA8 texture source.
    ///
    /// The code renders the DCT at its natural size and upscales it to
    /// [`RENDER_LONG_SIDE`], so the GPU bilinear filter smooths a 64 px source
    /// and not the raw 32 px grid.
    /// Run this on a decode worker, then upload the result on the main thread.
    #[must_use]
    pub fn render(&self) -> Option<DecodedImage> {
        let (w, h, rgba) = thumbhash::thumb_hash_to_rgba(self.as_bytes()).ok()?;
        if w == 0 || h == 0 || rgba.len() != w * h * 4 {
            return None;
        }
        let long = RENDER_LONG_SIDE.get() as usize;
        let (w, h) = (w as u32, h as u32);
        let scale = long as f32 / w.max(h) as f32;
        let (nw, nh) = (
            ((w as f32 * scale).round() as u32).max(1),
            ((h as f32 * scale).round() as u32).max(1),
        );
        Some(DecodedImage {
            width: ImgW::new(nw),
            height: ImgH::new(nh),
            data: crate::resample_rgba(&rgba, w, h, nw, nh),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth colour ramp at `w` x `h`, with every alpha channel opaque.
    fn ramp(w: u32, h: u32) -> DecodedImage {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&[
                    (x * 255 / w.max(1)) as u8,
                    (y * 255 / h.max(1)) as u8,
                    128,
                    255,
                ]);
            }
        }
        DecodedImage {
            width: ImgW::new(w),
            height: ImgH::new(h),
            data,
        }
    }

    /// The average RGBA of a buffer, per channel.
    fn average(img: &DecodedImage) -> [f32; 4] {
        let mut sum = [0.0; 4];
        for px in img.data.as_chunks::<4>().0 {
            for (s, c) in sum.iter_mut().zip(px) {
                *s += f32::from(*c);
            }
        }
        let n = img.data.len() as f32 / 4.0;
        sum.map(|s| s / n)
    }

    #[test]
    fn from_bytes_rejects_empty_and_over_long_payloads() {
        assert!(Placeholder::from_bytes(&[]).is_none());
        assert!(Placeholder::from_bytes(&[0xAA; MAX_BYTES + 1]).is_none());
        let p = Placeholder::from_bytes(&[1, 2, 3]).expect("payload");
        assert_eq!(p.len(), 3);
        assert!(!p.is_empty());
        assert_eq!(p.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn a_round_trip_keeps_the_colour() {
        let src = ramp(120, 60);
        let p = Placeholder::encode(&src).expect("encode");
        assert!(p.len() <= MAX_BYTES);
        let out = p.render().expect("render");
        assert_eq!(out.width.get(), RENDER_LONG_SIDE.get());
        // The render derives its height from the aspect of the DCT grid, which
        // rounds, so only the ratio matches.
        let ratio = out.width.get() as f32 / out.height.get() as f32;
        assert!((ratio - 2.0).abs() < 0.3, "{ratio}");
        // The DCT drops the detail, so only the average colour survives.
        let (a, b) = (average(&src), average(&out));
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 24.0, "{a:?} against {b:?}");
        }
    }

    #[test]
    fn the_rendered_placeholder_keeps_the_aspect() {
        for (w, h) in [(100, 25), (25, 100), (50, 50)] {
            let p = Placeholder::encode(&ramp(w, h)).expect("encode");
            let out = p.render().expect("render");
            let want = w as f32 / h as f32;
            let got = out.width.get() as f32 / out.height.get() as f32;
            assert!((want - got).abs() < 0.25 * want, "{w}x{h}: {got}");
        }
    }

    #[test]
    fn a_transparent_image_round_trips() {
        // The encoder reads the average alpha to pick the alpha plane, so a
        // half transparent image must survive as a payload.
        let mut src = ramp(40, 40);
        for (i, px) in src.data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            px[3] = if i < 800 { 0 } else { 255 };
        }
        let p = Placeholder::encode(&src).expect("encode");
        assert!(p.render().is_some());
    }

    #[test]
    fn a_truncated_payload_renders_nothing() {
        // Two bytes cannot hold the fixed part of the format.
        assert!(
            Placeholder::from_bytes(&[1, 2])
                .expect("payload")
                .render()
                .is_none()
        );
    }
}
