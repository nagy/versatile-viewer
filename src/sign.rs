//! Content sniffing: decide the decoder by the bytes, not the file name.
//!
//! [`is_jxl`] steers JPEG XL codestreams (and the `.jxl` extension) away from
//! the `image` crate.
//! [`format`] validates that the content is a decodable image.
//! [`dimensions`] reads the natural size without decoding the pixels.

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

/// Decide the decoder by file content, not by the file name.
///
/// The code sniffs the 2-byte magic first (JXL codestream against the
/// common formats).
/// The .jxl extension only acts as a tiebreaker for unknown magic.
/// JXL container files start with a box header, not the codestream magic.
pub(crate) fn is_jxl(path: &Path) -> bool {
    match file_magic(path) {
        Some([0xff, 0x0a]) => true, // raw JXL codestream
        Some(m) if is_common_magic(m) => false,
        _ => path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("jxl")),
    }
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

/// Sniff `path` and confirm a decoder can read its content.
///
/// A vector file passes on its head bytes, because the SVG parser reads the
/// content and not the `image` crate.
/// The function parses the raster header, so the conclusion comes from the
/// bytes.
/// `with_guessed_format` alone also honors the file extension, which would
/// accept a `.png` name over garbage content.
/// JPEG XL codestreams pass without an `image` crate check.
///
/// # Errors
///
/// The function errors when the file cannot be read, the JPEG XL header is
/// invalid, or the content matches no `image` crate format.
pub(crate) fn format(path: &Path) -> Result<(), SignError> {
    if crate::svg::sniff(path) {
        return Ok(());
    }
    if is_jxl(path) {
        return Ok(());
    }
    let reader = image::ImageReader::open(path)?.with_guessed_format()?;
    // `into_dimensions` parses the header, so the conclusion comes from the
    // bytes (the extension only picked the decoder to try).
    reader.into_dimensions().map_err(|_| SignError::Unknown)?;
    Ok(())
}

/// The natural dimensions of `path`, without decoding the pixels.
/// `jxl` selects the JPEG XL header reader.
///
/// # Errors
///
/// The function errors when the file cannot be read or the header is
/// invalid.
pub(crate) fn dimensions(path: &Path, jxl: bool) -> Result<(ImgW, ImgH), SignError> {
    if jxl {
        let image = jxl_oxide::JxlImage::builder().open(path)?;
        Ok((ImgW::new(image.width()), ImgH::new(image.height())))
    } else {
        let (width, height) = image::ImageReader::open(path)?
            .with_guessed_format()?
            .into_dimensions()?;
        Ok((ImgW::new(width), ImgH::new(height)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_jxl_detects_codestream_magic() {
        let tmp = tempfile::TempDir::new().unwrap();
        let bare = tmp.path().join("bare");
        // A raw codestream starts with 0xFF 0x0A, so it is JXL even without
        // an extension.
        std::fs::write(&bare, [0xffu8, 0x0a, 0x01, 0x02]).unwrap();
        assert!(is_jxl(&bare));
        // PNG magic is not a codestream, even without an extension.
        std::fs::write(&bare, [0x89u8, b'P', 0x4e, 0x47]).unwrap();
        assert!(!is_jxl(&bare));
    }

    #[test]
    fn jxl_extension_wins_for_unknown_magic() {
        // Unknown magic with a .jxl suffix falls back to JXL (container files
        // start with a box header, not the codestream magic).
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("container.jxl");
        std::fs::write(&path, [0x00u8, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ']).unwrap();
        assert!(is_jxl(&path));
    }

    #[test]
    fn dimensions_reads_png_header() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("img.png");
        image::DynamicImage::new_rgb8(7, 5).save(&path).unwrap();
        assert_eq!(
            dimensions(&path, false).unwrap(),
            (ImgW::new(7), ImgH::new(5))
        );
    }

    #[test]
    fn format_rejects_garbage() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("x.png");
        std::fs::write(&path, b"garbage").unwrap();
        assert!(format(&path).is_err());
    }
}
