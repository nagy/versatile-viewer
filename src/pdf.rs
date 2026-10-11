//! PDF backend: rasterizes pages through [hayro](https://docs.rs/hayro), a
//! pure-Rust PDF renderer.
//! Hayro has no C dependency and forbids unsafe code.
//!
//! This file owns the hayro dependency.
//! The rest of the viewer only sees the [`Surface`] trait.
//! A renderer swap (pdfium, say) means a rewrite of this file only.
//!
//! One PDF page is one plane.
//! The natural unit is the PostScript point (1/72 inch), so a render scale
//! of 1.0 maps one point to one pixel.
//! Hayro 0.8 renders through an arbitrary `kurbo::Affine`, so a region view
//! pays only for the pixels the screen shows.
//!
//! Rendering happens on the caller thread (rayon pool or loader worker).
//! Each call builds a fresh `RenderCache`, because hayro's cache is not
//! `Send`.
//! The cache normally lives for the life of a document.
//! A per-call cache trades some font-parse time for thread safety.
//!
//! Hayro is young and can panic on unusual PDFs.
//! Every render runs in `catch_unwind` and degrades to an ordinary render
//! error, so the viewer never crashes on a bad file.

use std::path::Path;

use derive_more::{Display, Error};
use hayro::{
    RenderCache, RenderSettings,
    hayro_interpret::{InterpreterSettings, util::TransformExt},
    hayro_syntax::{Pdf, page::Page},
    kurbo::Affine,
    render_into,
    vello_cpu::{
        Pixmap, RasterizerSettings, RenderContext, Resources, TargetInit,
        color::palette::css::WHITE,
    },
};

use crate::{
    DecodedImage,
    dim::{ImgH, ImgW},
    surface::{ChunkSource, Region, Size, Surface, SurfaceError, View},
};

/// Pixel budget for one render.
/// A huge page at a deep zoom could otherwise ask for gigabytes.
/// About 67 MP is 256 MB of RGBA8.
/// A clamped render is smaller than the request, and the returned scale
/// tells the truth, so the view upscales the short texture on the GPU
/// instead of stretching a wrong one.
const MAX_RENDER_PIXELS: f64 = 4096.0 * 4096.0 * 1.5;
/// The largest pixmap side the backend accepts (its pixel dimensions are
/// `u16`).
const MAX_RENDER_SIDE: f64 = 65535.0;

/// Why the PDF backend rejected a file or a page.
#[derive(Debug, Display, Error)]
pub enum PdfError {
    /// The file could not be read.
    #[display("failed to read file")]
    Io(#[error(source)] std::io::Error),
    /// The parser rejected the bytes.
    #[display("{_0}")]
    Parse(#[error(ignore)] &'static str),
    /// The document has no pages.
    #[display("PDF has no pages")]
    Empty,
    /// The plane index is out of range.
    #[display("page {_0} out of range")]
    Page(#[error(ignore)] usize),
    /// The render scale is not a positive, finite number.
    #[display("invalid render scale")]
    Scale(#[error(ignore)] f32),
    /// The rasterizer panicked on this page.
    #[display("hayro panicked while rendering page {_0}")]
    Panic(#[error(ignore)] usize),
}

impl From<std::io::Error> for PdfError {
    fn from(err: std::io::Error) -> Self {
        PdfError::Io(err)
    }
}

impl From<PdfError> for SurfaceError {
    fn from(err: PdfError) -> Self {
        SurfaceError::Render(err.to_string())
    }
}

/// A PDF file: one [`Surface`] plane per page.
///
/// The page sizes come from the parser at open time, so [`Surface::bounds`]
/// never re-parses.
// No `Debug`: hayro's `Pdf` does not implement it.
pub struct PdfSurface {
    pdf: Pdf,
    dims: Vec<(f32, f32)>,
}
impl PdfSurface {
    /// Parse the PDF file at `path` and capture every page size.
    ///
    /// # Errors
    ///
    /// The function errors when the file cannot be read, the parser rejects
    /// the bytes, or the document has no pages.
    pub fn open(path: &Path) -> Result<PdfSurface, PdfError> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// Parse a PDF held in memory.
    ///
    /// # Errors
    ///
    /// The function errors when the parser rejects the bytes or the document
    /// has no pages.
    pub fn from_bytes(data: Vec<u8>) -> Result<PdfSurface, PdfError> {
        let pdf = Pdf::new(data).map_err(|e| match e {
            hayro::hayro_syntax::LoadPdfError::Decryption(_) => {
                PdfError::Parse("encrypted PDFs not supported")
            }
            hayro::hayro_syntax::LoadPdfError::Invalid => PdfError::Parse("invalid PDF"),
        })?;
        let dims: Vec<(f32, f32)> = pdf.pages().iter().map(Page::render_dimensions).collect();
        if dims.is_empty() {
            return Err(PdfError::Empty);
        }
        Ok(PdfSurface { pdf, dims })
    }

    /// The natural size of `plane` in points, from the cached table.
    fn page_size(&self, plane: usize) -> Result<Size, SurfaceError> {
        let (w, h) = self
            .dims
            .get(plane)
            .copied()
            .ok_or(SurfaceError::Plane(plane))?;
        Ok(Size { w, h })
    }

    /// Rasterize one page.
    ///
    /// `region` is the source rectangle in points.
    /// `scale` is output pixels per point.
    /// The function returns the RGBA8 buffer plus the dimensions actually
    /// rendered (the scale clamp can make them smaller than the request).
    fn rasterize(
        &self,
        plane: usize,
        region: Region,
        scale: f32,
    ) -> Result<(DecodedImage, f32), SurfaceError> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(SurfaceError::Scale(scale));
        }
        self.page_size(plane)?;
        let (pw, ph, scale) = pixel_dimensions(region, scale);
        let pages = self.pdf.pages();
        let page_ref = pages.get(plane).ok_or(SurfaceError::Plane(plane))?;

        // A panic must not cross this boundary: the job runs on a rayon
        // worker or the loader thread, and the viewer must survive a
        // malformed page.
        let pixmap = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let cache = RenderCache::new();
            let interpreter = InterpreterSettings::default();
            let settings = RenderSettings::default();
            let mut ctx = RenderContext::new(0, 0);
            let mut pixmap = Pixmap::new(0, 0);
            let mut resources = Resources::default();

            ctx.reset_and_resize(pw, ph);
            pixmap.resize(ctx.width(), ctx.height());
            // Scale first, then move the requested region to the origin.
            // The page transform handles rotated and cropped pages.
            let transform = Affine::scale(f64::from(scale))
                * Affine::translate((-f64::from(region.x), -f64::from(region.y)))
                * page_ref.initial_transform(true).to_kurbo();
            render_into(
                page_ref,
                &cache,
                &interpreter,
                &settings,
                &mut ctx,
                transform,
            );
            ctx.flush();
            ctx.render_with(
                &mut pixmap,
                &mut resources,
                RasterizerSettings {
                    target_init: TargetInit::Clear(WHITE),
                    ..Default::default()
                },
            );
            pixmap
        }))
        .map_err(|_| SurfaceError::Render(PdfError::Panic(plane).to_string()))?;

        let (w, h) = (pixmap.width(), pixmap.height());
        if w == 0 || h == 0 {
            return Err(SurfaceError::Buffer);
        }
        // The pixmap holds premultiplied RGBA8.
        // An opaque white background makes every alpha 255, so premultiplied
        // equals straight and the buffer uploads as is.
        Ok((
            DecodedImage {
                width: ImgW::new(u32::from(w)),
                height: ImgH::new(u32::from(h)),
                data: pixmap.data_as_u8_slice().to_vec(),
            },
            scale,
        ))
    }
}

impl Surface for PdfSurface {
    fn plane_count(&self) -> usize {
        self.dims.len()
    }

    fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
        self.page_size(plane)
    }

    fn render_whole(&self, plane: usize, scale: f32) -> Result<DecodedImage, SurfaceError> {
        // The one method a backend must write: rasterize the whole page.
        // The code builds the full-page region from the cached bounds.
        let whole = self.page_size(plane)?.whole();
        self.rasterize(plane, whole, scale)
            .map(|(image, _used)| image)
    }

    fn render(&self, view: View) -> Result<DecodedImage, SurfaceError> {
        // The region anchor: hayro renders through an arbitrary
        // `kurbo::Affine`, so only the requested rectangle rasterizes.
        // A region view therefore pays for the window, not the whole page.
        let region = match view.region {
            Some(region) => region,
            None => self.page_size(view.plane)?.whole(),
        };
        self.rasterize(view.plane, region, view.scale)
            .map(|(image, _used)| image)
    }

    // A PDF has no progressive byte path: one page rasterizes in one step, so
    // the loader announces the page size and previews a coarse render instead.
    fn progressive(&self) -> Result<Option<Box<dyn ChunkSource>>, SurfaceError> {
        Ok(None)
    }

    // Rasterizing a page takes a moment, so the loader previews first.
    fn previews(&self) -> bool {
        true
    }

    // The scale is a request, not a promise: the view re-renders after each
    // settle, so a re-render is always possible and worthwhile.
    fn can_refine(&self) -> bool {
        true
    }
}

/// The pixmap dimensions for `region` at `scale`, and the scale after the
/// pixel-budget clamp.
/// The returned scale is at least as small as the requested one.
fn pixel_dimensions(region: Region, scale: f32) -> (u16, u16, f32) {
    let (rw, rh) = (f64::from(region.w.max(0.0)), f64::from(region.h.max(0.0)));
    let mut scale = f64::from(scale);
    let area = (rw * rh).max(1.0);
    scale = scale
        .min((MAX_RENDER_PIXELS / area).sqrt())
        .min(MAX_RENDER_SIDE / rw.max(rh).max(1.0))
        .max(1e-3);
    let pw = (rw * scale).round().clamp(1.0, MAX_RENDER_SIDE) as u16;
    let ph = (rh * scale).round().clamp(1.0, MAX_RENDER_SIDE) as u16;
    (pw, ph, scale as f32)
}

/// Does the file start with the PDF magic (`%PDF-`)?
/// The content decides, not the extension, so a misnamed file still routes
/// here.
#[must_use]
pub fn is_pdf(path: &Path) -> bool {
    file_magic_at_least(path, 5) == Some(b"%PDF-".to_vec())
}

/// The first `n` bytes of the file, or None on a missing or short file.
fn file_magic_at_least(path: &Path, n: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut buf = vec![0u8; n];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut buf))
        .ok()
        .map(|()| buf)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::surface::open_surface;

    /// A minimal single-page PDF (8.5x11 in letter, one text object).
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

    /// A two-page PDF with different page sizes, for the plane-grid tests.
    pub(crate) const TWO_PAGE_PDF: &[u8] = b"%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R 6 0 R]/Count 2>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 60>>stream
BT /F1 24 Tf 72 700 Td (Hello vv) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
6 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 100]/Contents 7 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
7 0 obj<</Length 40>>stream
BT /F1 12 Tf 10 50 Td (page two) Tj ET
endstream
endobj
trailer<</Root 1 0 R/Size 8>>";

    #[test]
    fn open_reports_one_page_and_its_size() {
        let surface = PdfSurface::from_bytes(MINIMAL_PDF.to_vec()).unwrap();
        assert_eq!(surface.plane_count(), 1);
        assert_eq!(surface.bounds(0).unwrap(), Size { w: 612.0, h: 792.0 });
        assert!(surface.bounds(1).is_err());
    }

    #[test]
    fn two_page_pdf_exposes_one_plane_per_page() {
        let surface = PdfSurface::from_bytes(TWO_PAGE_PDF.to_vec()).unwrap();
        assert_eq!(surface.plane_count(), 2);
        assert_eq!(surface.bounds(1).unwrap(), Size { w: 200.0, h: 100.0 });
        // Both planes render independently.
        for plane in 0..2 {
            let image = surface.render(View::whole(plane, 1.0)).unwrap();
            assert!(image.width.get() > 0 && image.height.get() > 0);
        }
    }

    #[test]
    fn whole_page_render_at_scale_has_the_right_size() {
        let surface = PdfSurface::from_bytes(MINIMAL_PDF.to_vec()).unwrap();
        let image = surface.render(View::whole(0, 2.0)).unwrap();
        assert_eq!(
            (image.width, image.height),
            (ImgW::new(1224), ImgH::new(1584))
        );
        assert_eq!(
            image.data.len(),
            1224 * 1584 * 4,
            "RGBA8 buffer matches the dimensions"
        );
        // The glyph pixels must differ from the white background.
        assert!(
            image
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .any(|px| px != &[255, 255, 255, 255])
        );
    }

    #[test]
    fn region_render_crops_and_scales() {
        let surface = PdfSurface::from_bytes(MINIMAL_PDF.to_vec()).unwrap();
        let region = Region {
            x: 70.0,
            y: 70.0,
            w: 100.0,
            h: 30.0,
        };
        let image = surface.render(View::region(0, region, 2.0)).unwrap();
        assert_eq!((image.width, image.height), (ImgW::new(200), ImgH::new(60)));
    }

    #[test]
    fn region_render_shows_the_same_ink_as_the_whole_page() {
        // Crop the page exactly around the text and confirm the ink lands
        // inside the crop: the region path must not shift the content.
        let surface = PdfSurface::from_bytes(MINIMAL_PDF.to_vec()).unwrap();
        let region = Region {
            x: 70.0,
            y: 70.0,
            w: 100.0,
            h: 30.0,
        };
        let image = surface.render(View::region(0, region, 1.0)).unwrap();
        assert!(
            image
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .any(|px| px != &[255, 255, 255, 255]),
            "the text sits inside the cropped region"
        );
    }

    #[test]
    fn bad_scale_and_plane_fail_cleanly() {
        let surface = PdfSurface::from_bytes(MINIMAL_PDF.to_vec()).unwrap();
        assert!(surface.render(View::whole(0, 0.0)).is_err());
        assert!(surface.render(View::whole(0, f32::NAN)).is_err());
        assert!(surface.render(View::whole(9, 1.0)).is_err());
    }

    #[test]
    fn garbage_fails_cleanly() {
        assert!(PdfSurface::from_bytes(b"%PDF-1.4 garbage not a pdf".to_vec()).is_err());
    }

    #[test]
    fn is_pdf_sniffs_magic_not_extension() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("document.weird");
        std::fs::write(&path, MINIMAL_PDF).unwrap();
        assert!(is_pdf(&path));
        let other = tmp.path().join("notes.pdf");
        std::fs::write(&other, b"not a pdf").unwrap();
        assert!(!is_pdf(&other));
    }

    #[test]
    fn open_surface_routes_pdf_content_to_the_backend() {
        // Content-first dispatch: PDF bytes under a nonsense extension still
        // open as a multi-plane PDF surface.
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("doc.weird");
        std::fs::write(&path, MINIMAL_PDF).unwrap();
        let surface = open_surface(&path).unwrap();
        assert_eq!(surface.plane_count(), 1);
        assert!(surface.can_refine());
        assert!(surface.previews());
    }
}
