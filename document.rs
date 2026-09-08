//! Abstract media sources: anything the viewer can open decodes through the
//! [`Document`] trait. An image file is a one-page document; a PDF has one
//! page per PDF page. Higher layers (grid, image view, loader) only ever see
//! documents and page indices — never file formats.
//!
//! Rendering happens off the main thread (rayon pool / loader worker); the
//! main thread only uploads the returned RGBA buffers as GPU textures.

use std::{path::Path, sync::Arc};

use anyhow::{Context, Result, bail};

/// A decoded page: RGBA8 pixels ready for GPU upload.
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, 4 bytes per pixel.
    pub rgba: Vec<u8>,
}

/// Static dimensions of one page, in its natural units:
/// pixels for image documents, PostScript points (1/72 inch) for PDFs.
/// [`Document::render`] scale 1.0 maps these 1:1 to pixels.
#[derive(Clone, Copy)]
pub struct PageInfo {
    pub width: u32,
    pub height: u32,
}

/// One openable document. Implementations must be usable from multiple
/// threads at once (`Send + Sync`): grid decodes and the streaming loader
/// call `render` concurrently on the shared pool.
pub trait Document: Send + Sync {
    /// Number of pages (1 for plain image files).
    fn page_count(&self) -> usize;

    /// Natural dimensions of `page` (0-based).
    fn page_info(&self, page: usize) -> Result<PageInfo>;

    /// Rasterize `page` at `scale` (1.0 = natural size, see [`PageInfo`]).
    fn render(&self, page: usize, scale: f32) -> Result<DecodedImage>;

    /// Byte-stream source for progressive decoding (JPEG XL only): the
    /// loader streams this file's bytes and renders blurry previews before
    /// the full decode. `None` = no streaming path (PDFs preview by
    /// rendering at a reduced scale instead).
    fn stream_path(&self) -> Option<&Path> {
        None
    }

    /// Whether the loader should send a fast low-resolution [`LoaderMsg`]
    /// preview before the full-quality render. True for PDFs (rasterizing
    /// takes a moment); false for images (already fast, and JXL has its own
    /// progressive path).
    fn previews(&self) -> bool {
        false
    }
}

/// Sniff the file format and open the matching document. The extension
/// decides for `.jxl`; everything else sniffs magic bytes (PDF), then falls
/// back to the `image` crate's format guessing (PNG, JPEG, WebP, ...).
pub fn open_document(path: &Path) -> Result<Arc<dyn Document>> {
    if is_jxl(path) {
        return Ok(Arc::new(ImageDocument::new(path)));
    }
    // PDF magic: "%PDF-" at the start of the file. Decided by content, not
    // extension, so misnamed files still open (and PDF content under a
    // random name still works).
    if std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            let mut magic = [0u8; 5];
            f.read_exact(&mut magic)?;
            Ok(magic == *b"%PDF-")
        })
        .unwrap_or(false)
    {
        return Ok(Arc::new(crate::pdf::PdfDocument::open(path)?));
    }
    // Not PDF: let the `image` crate sniff it. (Sniffing here doubles as a
    // validity check; render() would report the error otherwise.)
    image::ImageReader::open(path)
        .and_then(|r| r.with_guessed_format())
        .and_then(|r| {
            r.format().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "unknown image format")
            })
        })
        .with_context(|| format!("failed to open {path:?}"))?;
    Ok(Arc::new(ImageDocument::new(path)))
}

/// Does the file start with the PDF magic ("%PDF-")? Content decides, not
/// extension, so misnamed files still route correctly.
pub fn is_pdf(path: &Path) -> bool {
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            let mut magic = [0u8; 5];
            f.read_exact(&mut magic)?;
            Ok(magic == *b"%PDF-")
        })
        .unwrap_or(false)
}

/// A plain image file (JPEG XL, PNG, JPEG, WebP, ...) — always one page.
pub struct ImageDocument {
    path: std::path::PathBuf,
    jxl: bool,
}

impl ImageDocument {
    fn new(path: &Path) -> Self {
        ImageDocument {
            path: path.to_path_buf(),
            jxl: is_jxl(path),
        }
    }
}

impl Document for ImageDocument {
    fn page_count(&self) -> usize {
        1
    }

    fn page_info(&self, _page: usize) -> Result<PageInfo> {
        if self.jxl {
            let image = jxl_oxide::JxlImage::builder()
                .open(&self.path)
                .map_err(|e| anyhow::anyhow!("jxl-oxide: {e}"))
                .context("failed to open file")?;
            Ok(PageInfo {
                width: image.width(),
                height: image.height(),
            })
        } else {
            let (width, height) = image::ImageReader::open(&self.path)?
                .with_guessed_format()?
                .into_dimensions()?;
            Ok(PageInfo { width, height })
        }
    }

    fn render(&self, _page: usize, scale: f32) -> Result<DecodedImage> {
        // Images render at native size; `scale` other than 1.0 is a caller
        // bug (the view scales textures on the GPU).
        debug_assert!((scale - 1.0).abs() < 1e-3);
        let mut decoded = decode_image(&self.path)?;
        if (scale - 1.0).abs() > 1e-3 {
            // Be graceful anyway: nearest-neighbor scale, cheap and only a
            // safety net.
            let dyn_img = image::RgbaImage::from_raw(
                decoded.width,
                decoded.height,
                std::mem::take(&mut decoded.rgba),
            )
            .context("image buffer size mismatch")?;
            let w = ((decoded.width as f32 * scale) as u32).max(1);
            let h = ((decoded.height as f32 * scale) as u32).max(1);
            let scaled =
                image::imageops::resize(&dyn_img, w, h, image::imageops::FilterType::Nearest);
            decoded.width = w;
            decoded.height = h;
            decoded.rgba = scaled.into_raw();
        }
        Ok(decoded)
    }

    fn stream_path(&self) -> Option<&Path> {
        self.jxl.then_some(self.path.as_path())
    }
}

/// Decode a JPEG XL file with jxl-oxide (pure Rust).
pub fn decode_jxl(path: &Path) -> Result<DecodedImage> {
    let image = jxl_oxide::JxlImage::builder()
        .open(path)
        .map_err(|e| anyhow::anyhow!("jxl-oxide: {e}"))
        .context("failed to open file")?;
    let render = image
        .render_frame(0)
        .map_err(|e| anyhow::anyhow!("jxl-oxide: {e}"))
        .context("failed to render frame")?;

    let (rgba, width, height) = fb_to_rgba(&render.image_all_channels())?;
    Ok(DecodedImage {
        width,
        height,
        rgba,
    })
}

/// Convert a jxl-oxide framebuffer (f32 samples, 3 or 4 interleaved channels)
/// to RGBA8, forcing alpha = 1.0 for opaque 3-channel data.
pub fn fb_to_rgba(fb: &jxl_oxide::FrameBuffer) -> Result<(Vec<u8>, u32, u32)> {
    let width = fb.width() as u32;
    let height = fb.height() as u32;
    let channels = fb.channels();
    let samples = fb.buf();
    if !matches!(channels, 3 | 4) {
        bail!("unexpected channel count from jxl-oxide: {channels}");
    }

    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    for (dst, src) in rgba.chunks_exact_mut(4).zip(samples.chunks_exact(channels)) {
        dst[0] = to_u8(src[0]);
        dst[1] = to_u8(src[1]);
        dst[2] = to_u8(src[2]);
        dst[3] = to_u8(src.get(3).copied().unwrap_or(1.0));
    }
    Ok((rgba, width, height))
}

pub(crate) fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Decode common formats (PNG, JPEG, ...) with the `image` crate.
pub fn decode_common(path: &Path) -> Result<DecodedImage> {
    let rgba = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?
        .into_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(DecodedImage {
        width,
        height,
        rgba: rgba.into_raw(),
    })
}

pub(crate) fn is_jxl(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("jxl"))
        || {
            // JXL codestream sniff when extension is missing.
            std::fs::File::open(path)
                .and_then(|mut f| {
                    use std::io::Read;
                    let mut magic = [0u8; 2];
                    f.read_exact(&mut magic)?;
                    Ok(magic == [0xff, 0x0a])
                })
                .unwrap_or(false)
        }
}

fn decode_image(path: &Path) -> Result<DecodedImage> {
    if is_jxl(path) {
        decode_jxl(path)
    } else {
        decode_common(path)
    }
    .with_context(|| format!("failed to decode {path:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_jxl_detects_codestream_magic() {
        let dir = std::env::temp_dir();
        let bare = dir.join(format!("vv-test-jxl-{}", std::process::id()));
        // Raw codestream starts with 0xFF 0x0A — JXL even without extension.
        std::fs::write(&bare, [0xffu8, 0x0a, 0x01, 0x02]).unwrap();
        assert!(is_jxl(&bare));
        // Not a codestream.
        std::fs::write(&bare, b"PNG").unwrap();
        assert!(!is_jxl(&bare));
        std::fs::remove_file(&bare).ok();
    }

    #[test]
    fn jxl_extension_is_honored_even_for_bad_content() {
        // Extension decides first; sniffing only runs without a .jxl suffix.
        let path = std::env::temp_dir().join(format!("vv-test-ext-{}.jxl", std::process::id()));
        std::fs::write(&path, b"not really jxl").unwrap();
        assert!(is_jxl(&path));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn open_document_reads_png() {
        let dir = std::env::temp_dir().join(format!("vv-test-decode-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(3, 2).save(&path).unwrap();
        let doc = open_document(&path).unwrap();
        assert_eq!(doc.page_count(), 1);
        let decoded = doc.render(0, 1.0).unwrap();
        assert_eq!(decoded.width, 3);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.rgba.len(), 3 * 2 * 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_document_reads_webp() {
        // Lossless encode via image-webp, then decode back through the
        // document API (format sniffed from bytes, not extension).
        let dir = std::env::temp_dir().join(format!("vv-test-webp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("img.webp");
        image::DynamicImage::new_rgb8(5, 4)
            .save_with_format(&path, image::ImageFormat::WebP)
            .unwrap();
        let decoded = open_document(&path).unwrap().render(0, 1.0).unwrap();
        assert_eq!(decoded.width, 5);
        assert_eq!(decoded.height, 4);
        assert_eq!(decoded.rgba.len(), 5 * 4 * 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_document_fails_cleanly_on_garbage() {
        let dir = std::env::temp_dir().join(format!("vv-test-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.png");
        std::fs::write(&path, b"garbage").unwrap();
        let doc = open_document(&path).unwrap();
        assert!(doc.render(0, 1.0).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
