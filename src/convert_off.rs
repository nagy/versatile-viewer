//! The stub of [`crate::convert`] for a build without the `svg` feature.
//!
//! The converter path belongs to the SVG backend, because that backend paints
//! the converted document, so a build without SVG compiles no converter.
//! [`claims`] never answers true, so no path reaches the converter table and
//! every source routes by content alone.
//! [`open`] answers [`SurfaceError::Unsupported`], so a caller that reaches it
//! by another route still gets a clear reason.

use std::path::Path;

use crate::surface::{Surface, SurfaceError};

/// The rule table of a build with no converter path.
///
/// The grid directory filter takes a table of either build, and this one holds
/// no rule at all, so it never claims a file.
#[derive(Debug)]
pub struct Converters;

impl Converters {
    /// A build without the `svg` feature claims no path.
    pub(crate) fn claims(&self, _path: &Path) -> bool {
        false
    }
}

/// The rule table of this build, which claims no file.
pub(crate) fn rules() -> &'static Converters {
    static NO_RULES: Converters = Converters;
    &NO_RULES
}

/// No build without the `svg` feature claims a file.
#[must_use]
pub fn claims(_path: &Path) -> bool {
    false
}

/// A build without the `svg` feature runs no converter.
///
/// # Errors
///
/// The function always errors with [`SurfaceError::Unsupported`].
pub fn open(_path: &Path) -> Result<Box<dyn Surface>, SurfaceError> {
    Err(SurfaceError::Unsupported(
        "converter support not compiled in".into(),
    ))
}
