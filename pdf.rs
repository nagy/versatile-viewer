//! PDF backend: rasterizes PDF pages through [hayro](https://docs.rs/hayro),
//! a pure-Rust PDF renderer (no C dependencies, `#![forbid(unsafe_code)]`).
//!
//! This file owns the hayro dependency; the rest of the viewer only sees the
//! [`Document`] trait. Swapping renderers (e.g. pdfium) means rewriting this
//! file only.
//!
//! Rendering happens per call on the caller's thread (rayon pool / loader
//! worker): each call builds a fresh [`hayro::RenderCache`]. The cache
//! normally lives longer (one per document), but hayro's cache is not
//! `Send`, and our renders run on arbitrary pool threads — per-call caches
//! trade some font-parse time for thread safety. PDF rasterization at fit
//! scale takes well under a second for typical pages.
//!
//! Young-library risk: hayro may panic (not just error) on unusual PDFs.
//! Every render runs inside `catch_unwind` and degrades to a decode failure
//! message; the viewer never crashes on a bad file.

use std::path::Path;

use anyhow::{Context, Result, bail};
use hayro::{
    RenderCache, RenderSettings, hayro_interpret::InterpreterSettings, hayro_syntax::Pdf, render,
    vello_cpu::color::palette::css::WHITE,
};

use crate::document::{DecodedImage, Document, PageInfo};

/// Pixel budget for one rendered page: PDFs scale freely (points * zoom), so
/// an unclamped deep zoom could try to allocate gigabytes. ~67 MP ≈ 256 MB
/// RGBA8; beyond this the texture upscales on the GPU (soft, not sharp).
const MAX_PAGE_PIXELS: u64 = 4096 * 4096 + 4096 * 4096 / 2;

/// A PDF file, one [`Document`] page per PDF page.
pub struct PdfDocument {
    pdf: Pdf,
    /// Natural size (PostScript points, 1 pt = 1 px at render scale 1.0) of
    /// every page, captured once at open time — cheap and stable.
    dims: Vec<(f32, f32)>,
}

impl PdfDocument {
    /// Parse and validate the PDF file at `path`.
    pub fn open(path: &Path) -> Result<PdfDocument> {
        let data = std::fs::read(path).with_context(|| format!("failed to read {path:?}"))?;
        Self::from_data(data).with_context(|| format!("failed to parse PDF {path:?}"))
    }

    pub(crate) fn from_data(data: Vec<u8>) -> Result<PdfDocument> {
        let pdf = Pdf::new(data).map_err(|e| {
            anyhow::anyhow!(match e {
                hayro::hayro_syntax::LoadPdfError::Decryption(_) => "encrypted PDFs not supported",
                hayro::hayro_syntax::LoadPdfError::Invalid => "invalid PDF",
            })
        })?;
        let dims = pdf
            .pages()
            .iter()
            .map(|p| p.render_dimensions())
            .collect::<Vec<_>>();
        if dims.is_empty() {
            bail!("PDF has no pages");
        }
        Ok(PdfDocument { pdf, dims })
    }
}

impl Document for PdfDocument {
    fn page_count(&self) -> usize {
        self.dims.len()
    }

    fn page_info(&self, page: usize) -> Result<PageInfo> {
        let (w, h) = self
            .dims
            .get(page)
            .copied()
            .with_context(|| format!("page {page} out of range"))?;
        Ok(PageInfo {
            width: w.ceil() as u32,
            height: h.ceil() as u32,
        })
    }

    fn render(&self, page: usize, scale: f32) -> Result<DecodedImage> {
        if !scale.is_finite() || scale <= 0.0 {
            bail!("invalid render scale {scale}");
        }
        let (w, h) = self
            .dims
            .get(page)
            .copied()
            .with_context(|| format!("page {page} out of range"))?;
        // Clamp the pixel budget: rasterizing a page blown up 100x would
        // otherwise allocate absurd buffers. Everything past the clamp still
        // displays (GPU upscale), just without added sharpness. The u16 cap
        // comes from vello_cpu's pixmap width/height type.
        let mut scale = scale as f64;
        scale = scale.min((MAX_PAGE_PIXELS as f64 / ((w * h).max(1.0) as f64)).sqrt());
        scale = scale.min(65535.0 / w.max(h).max(1.0) as f64);
        let pw = (w as f64 * scale).floor().clamp(1.0, 65535.0) as u16;
        let ph = (h as f64 * scale).floor().clamp(1.0, 65535.0) as u16;

        let pages = self.pdf.pages();
        let page_ref = pages
            .get(page)
            .with_context(|| format!("page {page} out of range"))?;
        let cache = RenderCache::new();
        let interp = InterpreterSettings::default();
        let settings = RenderSettings {
            x_scale: scale as f32,
            y_scale: scale as f32,
            width: Some(pw),
            height: Some(ph),
            bg_color: WHITE,
        };

        // hayro is young: unexpected panics on malformed content must not
        // take the viewer down (the job runs on a rayon worker). catch_unwind
        // turns them into an ordinary decode failure.
        let pixmap = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            render(page_ref, &cache, &interp, &settings)
        }))
        .map_err(|_| anyhow::anyhow!("hayro panicked while rendering page {page}"))?;

        let width = pixmap.width() as u32;
        let height = pixmap.height() as u32;
        if width == 0 || height == 0 {
            bail!("hayro rendered page {page} at size 0x0");
        }
        // Pixmap data is premultiplied RGBA8; with an opaque white
        // background every pixel has alpha 255, so premultiplied == straight.
        let rgba = pixmap.data_as_u8_slice().to_vec();
        Ok(DecodedImage {
            width,
            height,
            rgba,
        })
    }

    fn previews(&self) -> bool {
        // Rasterizing takes a moment: send a fast low-res render first, then
        // the full-quality one (mirrors the JXL progressive experience).
        true
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::document::open_document;

    /// Minimal single-page PDF (8.5x11 in letter, one text object), raw bytes.
    pub(crate) const MINIMAL_PDF: &[u8] = b"%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 60>>stream
BT /F1 24 Tf 72 700 Td (Hello vv) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
trailer<</Root 1 0 R/Size 6>>";

    #[test]
    fn pdf_document_opens_and_reports_pages() {
        let doc = PdfDocument::from_data(MINIMAL_PDF.to_vec()).unwrap();
        assert_eq!(doc.page_count(), 1);
        let info = doc.page_info(0).unwrap();
        assert_eq!(info.width, 612);
        assert_eq!(info.height, 792);
        assert!(doc.page_info(1).is_err());
    }

    #[test]
    fn pdf_document_renders_page_at_scale() {
        let doc = PdfDocument::from_data(MINIMAL_PDF.to_vec()).unwrap();
        let img = doc.render(0, 2.0).unwrap();
        assert_eq!((img.width, img.height), (1224, 1584));
        assert_eq!(img.rgba.len(), img.width as usize * img.height as usize * 4);
        // Some glyph pixels must differ from the white background.
        assert!(
            img.rgba
                .chunks_exact(4)
                .any(|px| px != [255, 255, 255, 255])
        );
    }

    #[test]
    fn pdf_document_open_sniffs_magic_bytes() {
        let dir = std::env::temp_dir().join(format!("vv-test-pdf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // PDF content under a nonsense extension: magic decides.
        let path = dir.join("doc.weird");
        std::fs::write(&path, MINIMAL_PDF).unwrap();
        let doc = open_document(&path).unwrap();
        assert_eq!(doc.page_count(), 1);
        assert!(doc.previews());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn garbage_pdf_fails_cleanly() {
        let dir = std::env::temp_dir().join(format!("vv-test-badpdf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.pdf");
        std::fs::write(&path, b"%PDF-1.4 garbage not a pdf").unwrap();
        assert!(open_document(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
