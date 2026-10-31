//! The stub of [`crate::svg`] for a build without the `svg` feature.
//!
//! [`sniff`] never claims a file, so every decode falls through to the raster
//! decoders and a `.svg` path fails there with the normal image error.
//! [`parse`] answers [`DecodeError::VectorOff`], so a caller that reaches it
//! by another route still gets a clear reason.
//! [`Vector`] exists only to name the type, so the code that holds a parsed
//! source compiles in both builds.

use std::path::Path;

use crate::{
    DecodeError, DecodedImage,
    dim::{ImgH, ImgW, LongSide},
};

/// No build without the `svg` feature parses a vector file.
///
/// The type stays empty, so no code can hold a tree that never exists.
#[derive(Debug)]
pub struct Vector;

impl Vector {
    /// A stub never paints, so it measures nothing.
    #[must_use]
    pub fn base_size(&self) -> (ImgW, ImgH) {
        (ImgW::new(0), ImgH::new(0))
    }

    /// A stub never paints.
    ///
    /// # Errors
    ///
    /// The function always errors with [`DecodeError::VectorOff`].
    pub fn render_base(&self) -> Result<DecodedImage, DecodeError> {
        Err(DecodeError::VectorOff)
    }

    /// A stub never paints.
    ///
    /// # Errors
    ///
    /// The function always errors with [`DecodeError::VectorOff`].
    pub fn rasterize(&self, _long: LongSide) -> Result<DecodedImage, DecodeError> {
        Err(DecodeError::VectorOff)
    }

    /// A stub never paints.
    ///
    /// # Errors
    ///
    /// The function always errors with [`DecodeError::VectorOff`].
    pub fn render_region(
        &self,
        _region: crate::region::Crop,
        _scale: f32,
        _background: Option<[u8; 4]>,
    ) -> Result<DecodedImage, DecodeError> {
        Err(DecodeError::VectorOff)
    }
}

/// No build without the `svg` feature reads a vector file.
#[must_use]
pub fn sniff(_path: &Path) -> bool {
    false
}

/// A build without the `svg` feature cannot parse a vector file.
///
/// # Errors
///
/// The function always errors with [`DecodeError::VectorOff`].
pub fn parse(_path: &Path) -> Result<Vector, DecodeError> {
    Err(DecodeError::VectorOff)
}

/// A build without the `svg` feature cannot measure a vector file.
///
/// # Errors
///
/// The function always errors with [`DecodeError::VectorOff`].
pub(crate) fn measure(_path: &Path) -> Result<(ImgW, ImgH), DecodeError> {
    Err(DecodeError::VectorOff)
}
