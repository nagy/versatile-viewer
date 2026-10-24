//! The thumbnail cache, compiled out by the `cache` feature.
//!
//! The grid compiles against the same API as `src/thumbcache.rs`, so no
//! decode worker carries a `cfg` of its own.
//! Every call answers [`CacheError::Off`], so every launch decodes every
//! file and the grid keeps no entry on disk.
//! The module name and the types match the real one, so `error.rs` and
//! `grid.rs` never branch on the feature.
//!
//! The real module holds the docstring for the cache itself.

use std::path::{Path, PathBuf};

use derive_more::{Display, Error};

use crate::{
    DecodedImage,
    dim::{ImgH, ImgW, LongSide},
};

/// Why a cache file did not load or store.
///
/// With the feature off the cache is always off, so this enum has one
/// variant. The real module holds the full set.
#[derive(Debug, Display, Error)]
pub enum CacheError {
    /// The cache is off: this build has no `cache` feature.
    #[display("cache is off")]
    Off,
}

/// A thumbnail from the cache, with the source dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedThumb {
    /// The cached RGBA8 thumbnail.
    pub thumb: DecodedImage,
    /// The source dimensions, as a decode capped at
    /// [`crate::MAX_TEXTURE_SIDE`] would report them.
    pub image: (ImgW, ImgH),
}

/// The thumbnail cache of a build without the `cache` feature.
///
/// It carries no directory and writes nothing, so the type is zero-sized.
#[derive(Debug, Default)]
pub struct Cache;

impl Cache {
    /// The cache is always off, so the function never fails.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// An off cache, for a viewer without a home directory and for tests.
    #[must_use]
    pub fn off() -> Self {
        Self
    }

    /// An off cache. The argument goes unused, so a caller that names a
    /// directory keeps one code path.
    #[must_use]
    pub fn in_dir(_dir: impl Into<PathBuf>) -> Self {
        Self
    }

    /// The cache directory, which never exists in this build.
    #[must_use]
    pub fn dir(&self) -> Option<&Path> {
        None
    }

    /// No entry exists, so the function answers `None`.
    #[must_use]
    pub fn file(&self, _path: &Path, _long_side: LongSide) -> Option<PathBuf> {
        None
    }

    /// The cache is off, so the function errors on every call.
    ///
    /// # Errors
    ///
    /// The function always errors with [`CacheError::Off`].
    pub fn load(&self, _path: &Path, _long_side: LongSide) -> Result<CachedThumb, CacheError> {
        Err(CacheError::Off)
    }

    /// The cache is off, so the function finds no placeholder.
    ///
    /// # Errors
    ///
    /// The function always errors with [`CacheError::Off`].
    pub fn load_placeholder(
        &self,
        _path: &Path,
        _long_side: LongSide,
    ) -> Result<(crate::placeholder::Placeholder, (ImgW, ImgH)), CacheError> {
        Err(CacheError::Off)
    }

    /// The cache is off, so the function stores nothing.
    ///
    /// # Errors
    ///
    /// The function always errors with [`CacheError::Off`].
    pub fn store(
        &self,
        _path: &Path,
        _long_side: LongSide,
        _thumb: &DecodedImage,
        _image: (ImgW, ImgH),
    ) -> Result<(), CacheError> {
        Err(CacheError::Off)
    }
}
