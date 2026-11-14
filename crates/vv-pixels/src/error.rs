//! Errors for image decoding.
//!
//! [`DecodeError`] covers opening, sniffing and decoding a file.

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
    /// The RGBA buffer did not match its dimensions.
    /// [`crate::DecodedImage::new`] answers this instead of building an image
    /// whose pixel count and buffer disagree.
    #[display("rgba buffer does not match its dimensions")]
    Buffer,
}
