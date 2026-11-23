//! The RGBA8 buffer every decoder answers with.
//!
//! [`DecodedImage`] is the one checked container of pixel data: the fields are
//! private, [`DecodedImage::new`] refuses a buffer that does not match its
//! dimensions, and [`DecodedImage::from_exact`] is the entrance for a length
//! the caller just computed.
//! The pixels leave through `as_rgba`, `into_rgba` and `into_parts`.
//!
//! The type lives this low because a [`crate::scene::Primitive`] carries one as
//! its image leaf, and the scene type sits below the renderer.
//! No decoder lives here: `vv-pixels` holds the resamplers and the decoders that
//! fill this container, and re-exports both names.

use derive_more::{Display, Error};

use crate::dim::{ImgH, ImgW};

/// A buffer that does not hold `width * height * 4` bytes.
///
/// [`DecodedImage::new`] answers this instead of building an image whose pixel
/// count and buffer disagree, so the mismatch never exists to check later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, Error)]
#[display("rgba buffer does not match its dimensions")]
pub struct BufferError;

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
/// # use vv_core::pixels::{BufferError, DecodedImage};
/// # use vv_core::dim::{ImgH, ImgW};
/// let img = DecodedImage::new(ImgW::new(2), ImgH::new(2), vec![0; 16]).unwrap();
/// assert_eq!(img.as_rgba().len(), 16);
/// assert_eq!(img.width(), ImgW::new(2));
/// // A buffer of the wrong size never becomes an image.
/// assert!(matches!(
///     DecodedImage::new(ImgW::new(2), ImgH::new(2), vec![0; 15]),
///     Err(BufferError)
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
    /// The function errors with [`BufferError`] when `data` does not hold
    /// exactly `width * height * 4` bytes.
    pub fn new(width: ImgW, height: ImgH, data: Vec<u8>) -> Result<Self, BufferError> {
        if Some(data.len()) != pixel_bytes(width, height) {
            return Err(BufferError);
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// The image built from a buffer whose length the caller just computed.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matching_buffer_becomes_an_image() {
        let img = DecodedImage::new(ImgW::new(3), ImgH::new(2), vec![7; 24]).expect("3x2 RGBA8");
        assert_eq!(img.width(), ImgW::new(3));
        assert_eq!(img.height(), ImgH::new(2));
        assert_eq!(img.as_rgba().len(), 24);
    }

    #[test]
    fn a_short_or_long_buffer_is_refused() {
        assert_eq!(
            DecodedImage::new(ImgW::new(3), ImgH::new(2), vec![0; 23]),
            Err(BufferError)
        );
        assert_eq!(
            DecodedImage::new(ImgW::new(3), ImgH::new(2), vec![0; 25]),
            Err(BufferError)
        );
    }

    #[test]
    fn an_empty_image_takes_an_empty_buffer() {
        let img = DecodedImage::new(ImgW::new(0), ImgH::new(0), Vec::new()).expect("empty");
        assert_eq!(img.as_rgba(), []);
    }

    #[test]
    fn the_parts_round_trip() {
        let img = DecodedImage::from_exact(ImgW::new(1), ImgH::new(1), vec![1, 2, 3, 4]);
        assert_eq!(img.into_parts(), (ImgW::new(1), ImgH::new(1), vec![1, 2, 3, 4]));
    }
}
