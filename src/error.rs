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
