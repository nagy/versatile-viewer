//! Content sniffing: decide the decoder by the bytes, not the file name.
//!
//! [`detect`] reads the magic once and answers a [`Kind`]: a raster file with
//! its [`Format`], or an SVG document.
//! The caller therefore routes on one decision and never re-sniffs the same
//! file.
//! [`dimensions`] reads the natural size of a raster file without decoding the
//! pixels.

use std::path::Path;

use derive_more::{Display, Error, From};

use crate::dim::{ImgH, ImgW};

/// Why a file is not a decodable image.
#[derive(Debug, Display, Error, From)]
pub enum SignError {
    /// The file could not be opened or read.
    #[display("failed to open file")]
    Open(#[from] std::io::Error),
    /// The content matches no known format.
    #[display("unknown image format")]
    Unknown,
    /// The `image` crate rejected the header.
    #[display("image header rejected")]
    Image(#[from] image::ImageError),
    /// The JPEG XL decoder rejected the bytes.
    #[display("jxl-oxide: {_0}")]
    Jxl(
        #[from]
        #[error(source)]
        Box<dyn std::error::Error + Send + Sync + 'static>,
    ),
}

/// What the content of a file calls for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A raster file, which the named [`Format`] decodes.
    Raster(Format),
    /// An SVG or SVGZ document, which the vector backend parses and paints.
    Vector,
}

impl Kind {
    /// The raster decoder, or `None` for a vector document.
    #[must_use]
    pub const fn raster(self) -> Option<Format> {
        match self {
            Self::Raster(format) => Some(format),
            Self::Vector => None,
        }
    }
}

/// The decoder that reads a raster file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// A JPEG XL codestream, which jxl-oxide decodes.
    Jxl,
    /// A format of the `image` crate (PNG, JPEG, WebP, ...).
    Common,
}

/// The two-byte magic of a raw JPEG XL codestream.
const JXL_MAGIC: [u8; 2] = [0xff, 0x0a];

/// The two-byte magic of `path`.
///
/// A file shorter than two bytes matches no format, so it answers
/// [`SignError::Unknown`] instead of a read error.
fn head_magic(path: &Path) -> Result<[u8; 2], SignError> {
    use std::io::Read;
    let mut magic = [0u8; 2];
    match std::fs::File::open(path)?.read_exact(&mut magic) {
        Ok(()) => Ok(magic),
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => Err(SignError::Unknown),
        Err(err) => Err(SignError::Open(err)),
    }
}

/// Sniff `path` once and name what its content calls for.
///
/// The function reads the 2-byte magic first and keeps it for both raster
/// rules, so no caller re-sniffs the same file to answer a second question.
/// A vector document goes to the SVG sniff before any raster rule, because it
/// paints from its own tree.
/// A JPEG XL codestream passes on its magic, and the `.jxl` extension acts as
/// the tiebreaker for an unknown magic (a container starts with a box header,
/// not the codestream magic).
/// Every other file passes only when the `image` crate parses its header, so
/// the conclusion comes from the bytes and the extension only picked the
/// decoder to try.
///
/// # Errors
///
/// The function errors when the file cannot be read, or when the content
/// matches no decoder.
pub fn detect(path: &Path) -> Result<Kind, SignError> {
    let magic = head_magic(path)?;
    if crate::svg::sniff(path) {
        return Ok(Kind::Vector);
    }
    if magic == JXL_MAGIC || (!is_common_magic(magic) && has_jxl_extension(path)) {
        return Ok(Kind::Raster(Format::Jxl));
    }
    image::ImageReader::open(path)?
        .with_guessed_format()?
        .into_dimensions()
        .map_err(|_| SignError::Unknown)?;
    Ok(Kind::Raster(Format::Common))
}

/// Whether the name of `path` ends in `.jxl`, in any letter case.
fn has_jxl_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("jxl"))
}

/// The first two bytes of the file.
/// The function returns None for a missing or short file.
pub(crate) fn file_magic(path: &Path) -> Option<[u8; 2]> {
    use std::io::Read;
    let mut magic = [0u8; 2];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .ok()
        .map(|()| magic)
}

/// The magics the image crate can decode.
/// `with_guessed_format` sniffs the full header.
/// This function only needs to steer files away from the JXL decoder.
// The magic table stays flat and grouped by format on purpose.
// The clippy nested suggestion reorders it into a byte soup.
#[allow(clippy::unnested_or_patterns)]
pub(crate) const fn is_common_magic(m: [u8; 2]) -> bool {
    let [a, b] = m;
    matches!(
        (a, b),
        (0x89, b'P') // PNG
            | (0xff, 0xd8) // JPEG
            | (b'R', b'I') // RIFF (WebP)
            | (b'G', b'I') // GIF
            | (b'B', b'M') // BMP
            | (b'I', b'I')
            | (b'I', b'M') // TIFF, both byte orders
            | (b'M', b'I')
            | (b'M', b'M')
    )
}

/// The natural dimensions of `path`, without decoding the pixels.
///
/// `format` selects the header reader, so the caller passes the decoder that
/// [`detect`] named for the same file.
///
/// # Errors
///
/// The function errors when the file cannot be read or the header is
/// invalid.
pub fn dimensions(path: &Path, format: Format) -> Result<(ImgW, ImgH), SignError> {
    match format {
        Format::Jxl => {
            let image = jxl_oxide::JxlImage::builder().open(path)?;
            Ok((ImgW::new(image.width()), ImgH::new(image.height())))
        }
        Format::Common => {
            let (width, height) = image::ImageReader::open(path)?
                .with_guessed_format()?
                .into_dimensions()?;
            Ok((ImgW::new(width), ImgH::new(height)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_names_the_jxl_codestream() {
        let tmp = tempfile::TempDir::new().unwrap();
        let bare = tmp.path().join("bare");
        // A raw codestream starts with 0xFF 0x0A, so it is JXL even without
        // an extension.
        std::fs::write(&bare, [0xffu8, 0x0a, 0x01, 0x02]).unwrap();
        assert_eq!(detect(&bare).unwrap(), Kind::Raster(Format::Jxl));
        // PNG magic reads as the `image` crate format, not a codestream.
        let png = tmp.path().join("bare-png");
        image::DynamicImage::new_rgb8(2, 2)
            .save_with_format(&png, image::ImageFormat::Png)
            .unwrap();
        assert!(png.extension().is_none());
        assert_eq!(detect(&png).unwrap(), Kind::Raster(Format::Common));
    }

    #[test]
    fn jxl_extension_wins_for_unknown_magic() {
        // Unknown magic with a .jxl suffix falls back to JXL (container files
        // start with a box header, not the codestream magic).
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("container.jxl");
        std::fs::write(&path, [0x00u8, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ']).unwrap();
        assert_eq!(detect(&path).unwrap(), Kind::Raster(Format::Jxl));
    }

    #[test]
    fn dimensions_reads_png_header() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("img.png");
        image::DynamicImage::new_rgb8(7, 5).save(&path).unwrap();
        assert_eq!(
            dimensions(&path, Format::Common).unwrap(),
            (ImgW::new(7), ImgH::new(5))
        );
    }

    #[test]
    fn detect_rejects_garbage() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("x.png");
        std::fs::write(&path, b"garbage").unwrap();
        assert!(detect(&path).is_err());
    }

    #[test]
    fn detect_rejects_a_file_shorter_than_the_magic() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("empty.png");
        std::fs::write(&path, []).unwrap();
        assert!(matches!(detect(&path), Err(SignError::Unknown)));
    }
}
