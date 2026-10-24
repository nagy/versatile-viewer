//! Errors for image decoding and texture upload.
//!
//! [`DecodeError`] covers opening, sniffing and decoding a file.
//! [`UploadError`] covers the GL texture upload.

use derive_more::{Display, Error, From};

/// A failed image decode.
#[derive(Debug, Display, Error, From)]
pub enum DecodeError {
    /// The file could not be opened.
    #[display("failed to open file")]
    Open(std::io::Error),
    /// The thumbnail cache could not serve or take the thumbnail.
    /// A decode that reached this point already produced the pixels, so the
    /// grid shows them and the cache only misses.
    #[display("thumbnail cache: {_0}")]
    Cache(#[from] crate::thumbcache::CacheError),
    /// The JPEG XL decoder rejected the bytes.
    #[display("jxl-oxide: {_0}")]
    Jxl(
        #[from]
        #[error(source)]
        Box<dyn std::error::Error + Send + Sync + 'static>,
    ),
    /// The `image` crate could not decode the bytes.
    #[display("image decode failed")]
    Image(image::ImageError),
    /// The decoded framebuffer had an unexpected channel count.
    #[display("unexpected channel count from jxl-oxide: {_0}")]
    Channels(#[error(ignore)] u32),
    /// The SVG parser or painter rejected the file.
    ///
    /// The variant carries a boxed error, like [`Self::Jxl`], so the code
    /// builds it by hand: a `From` impl would collide with the JXL one.
    #[display("svg: {_0}")]
    #[from(skip)]
    Svg(Box<dyn std::error::Error + Send + Sync + 'static>),
    /// The raster pixmap of a vector file could not allocate.
    #[display("svg raster allocation failed")]
    Raster,
    /// A vector file passed the input cap.
    #[display("svg input too large")]
    Input,
    /// The build has no SVG support.
    #[display("svg support not compiled in")]
    VectorOff,
}

/// A failed GL texture upload.
#[derive(Debug, Display, Error, From)]
pub enum UploadError {
    /// The GL texture upload failed (no current context, driver refusal).
    #[display("texture upload failed")]
    Upload(#[from] raylib::core::error::LoadTextureError),
    /// The in-place texture update failed.
    #[display("texture update failed")]
    Update(#[from] raylib::core::error::UpdateTextureError),
    /// The RGBA buffer did not match its dimensions.
    #[display("rgba buffer does not match its dimensions")]
    Buffer,
}
