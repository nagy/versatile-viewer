//! The stub of [`crate::mermaid`] for a build without the `mermaid` feature.
//!
//! [`is_mermaid`] never claims a file, so a `.mmd` path falls through to the
//! raster decoders and fails there with the normal image error.
//! [`open`] answers [`SurfaceError::Unsupported`], so a caller that reaches it
//! by another route still gets a clear reason.

use std::path::Path;

use crate::surface::{Surface, SurfaceError};

/// No build without the `mermaid` feature claims a `.mmd` file.
#[must_use]
pub fn is_mermaid(_path: &Path) -> bool {
    false
}

/// A build without the `mermaid` feature cannot render a diagram.
///
/// # Errors
///
/// The function always errors with [`SurfaceError::Unsupported`].
pub fn open(_path: &Path) -> Result<Box<dyn Surface>, SurfaceError> {
    Err(SurfaceError::Unsupported(
        "Mermaid support not compiled in".into(),
    ))
}
