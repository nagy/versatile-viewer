//! The stub of [`crate::svg`] for a build without the `svg` feature.
//!
//! [`sniff`] never claims a file, so every decode falls through to the raster
//! decoders and a `.svg` path fails there with the normal image error.
//! [`decode_base`] and [`decode_thumb`] answer [`DecodeError::VectorOff`], so
//! a caller that reaches them by another route still gets a clear reason.

use std::path::Path;

use crate::{DecodeError, Decoded, dim::LongSide};

/// No build without the `svg` feature reads a vector file.
#[must_use]
pub(crate) fn sniff(_path: &Path) -> bool {
    false
}

/// A build without the `svg` feature cannot decode a vector file.
///
/// # Errors
///
/// The function always errors with [`DecodeError::VectorOff`].
pub(crate) fn decode_base(_path: &Path) -> Result<Decoded, DecodeError> {
    Err(DecodeError::VectorOff)
}

/// A build without the `svg` feature cannot decode a vector file.
///
/// # Errors
///
/// The function always errors with [`DecodeError::VectorOff`].
pub(crate) fn decode_thumb(_path: &Path, _long: LongSide) -> Result<Decoded, DecodeError> {
    Err(DecodeError::VectorOff)
}
