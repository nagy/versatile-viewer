//! Renderable 2D sources: anything the viewer can show renders through the
//! [`Surface`] trait.
//! An image file is a one-plane surface; a multi-page document has one plane
//! per page.
//! Higher layers (grid, image view, loader) only see surfaces, plane indices
//! and regions, never file formats.
//!
//! A raster file renders its one raster, scaled to what the caller asked for.
//! A vector file keeps its parsed tree (see [`crate::svg`]), so a region
//! render pays only for the pixels the caller wants, at the scale it shows
//! them.
//! The base raster of a vector file is its source size, so the grid entry, the
//! cache header and the image fit all measure that pair.
//!
//! Rendering happens off the main thread (rayon pool and loader worker).
//! The main thread only uploads the returned RGBA buffers as GPU textures.

use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

use derive_more::{Display, Error, From};

use crate::{
    DecodedImage,
    dim::{ImgH, ImgW},
    error::DecodeError,
    sign::{self, SignError},
};

/// The natural extent of one plane, in surface units (pixels for an image,
/// PostScript points for a PDF).
/// A [`View`] with scale 1.0 maps these 1:1 to output pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size {
    /// The plane width, in surface units.
    pub w: f32,
    /// The plane height, in surface units.
    pub h: f32,
}

impl Size {
    /// The whole plane as a [`Region`].
    #[must_use]
    pub const fn whole(self) -> Region {
        Region {
            x: 0.0,
            y: 0.0,
            w: self.w,
            h: self.h,
        }
    }
}

/// A rectangle in surface units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    /// The left edge, in surface units.
    pub x: f32,
    /// The top edge, in surface units.
    pub y: f32,
    /// The width, in surface units.
    pub w: f32,
    /// The height, in surface units.
    pub h: f32,
}

impl From<crate::region::Crop> for Region {
    fn from(crop: crate::region::Crop) -> Self {
        Region {
            x: crop.x,
            y: crop.y,
            w: crop.w,
            h: crop.h,
        }
    }
}

impl From<Region> for crate::region::Crop {
    fn from(region: Region) -> Self {
        crate::region::Crop {
            x: region.x,
            y: region.y,
            w: region.w,
            h: region.h,
        }
    }
}

/// What to render: which plane, which region of it, and at what scale.
/// `None` region means the whole plane.
/// Further fields (a rotation, for example) land here without a change to
/// every call site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    /// The plane index (0 for a one-plane surface).
    pub plane: usize,
    /// The region to rasterize, in surface units ([`Size::whole`] when
    /// absent).
    pub region: Option<Region>,
    /// The output pixels per surface unit (1.0 = natural size).
    pub scale: f32,
}

impl View {
    /// A whole-plane view at `scale`.
    #[must_use]
    pub const fn whole(plane: usize, scale: f32) -> Self {
        View {
            plane,
            region: None,
            scale,
        }
    }

    /// A region view at `scale`.
    #[must_use]
    pub const fn region(plane: usize, region: Region, scale: f32) -> Self {
        View {
            plane,
            region: Some(region),
            scale,
        }
    }
}

/// A byte stream for progressive decoding.
///
/// A surface returns one of these when it can decode better and better while
/// bytes arrive (JPEG XL today).
/// It hides where the bytes come from, so an in-memory or network source can
/// stream without a file path.
/// The loader reads fixed-size chunks until [`ChunkSource::read`] returns 0.
pub trait ChunkSource: Send {
    /// Read up to `buf.len()` bytes into `buf`.
    ///
    /// A short read is normal; the caller reads again.
    ///
    /// # Errors
    ///
    /// The function errors when the underlying source fails (a read error, a
    /// dropped connection).
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize>;
}

/// A [`ChunkSource`] over a memory-mapped file.
///
/// The map keeps the SIGBUS contract of the streaming loader: a file rewritten
/// under the map faults instead of returning short data.
/// A chunk copy out of the map costs no read syscall.
struct FileChunks {
    map: memmap2::Mmap,
    pos: usize,
}

impl ChunkSource for FileChunks {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = buf.len().min(self.map.len() - self.pos);
        buf[..n].copy_from_slice(&self.map[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// One renderable surface.
/// Implementations must be usable from multiple threads at once
/// (`Send + Sync`): grid decodes and the streaming loader call
/// [`Self::render`] concurrently on the shared pool.
pub trait Surface: Send + Sync {
    /// Number of planes (1 for a plain image file).
    fn plane_count(&self) -> usize;

    /// The natural extent of `plane` (0-based).
    ///
    /// # Errors
    ///
    /// The function errors when the plane index is out of range or the file
    /// cannot be read.
    fn bounds(&self, plane: usize) -> Result<Size, SurfaceError>;

    /// Rasterize the whole of `plane` at `scale` output pixels per surface
    /// unit.
    ///
    /// This is the one method every backend must write.
    /// A backend that can also anchor a render at a sub-rectangle overrides
    /// [`Self::render`] for a cheaper region path.
    ///
    /// # Errors
    ///
    /// The function errors when the plane index is out of range, the scale is
    /// invalid or the decode fails.
    fn render_whole(&self, plane: usize, scale: f32) -> Result<DecodedImage, SurfaceError>;

    /// Rasterize `view`: a plane, an optional region of it, and a scale.
    ///
    /// The region arrives in surface units, and the scale is output pixels per
    /// surface unit, so a caller asks for exactly the pixels it shows.
    ///
    /// The default renders the whole plane and crops the region out of it, so
    /// a backend with no region anchor inherits region support.
    /// A backend with an anchor (a vector tree, a PDF rasterizer) overrides
    /// this method to pay only for the pixels the caller asked for.
    ///
    /// # Errors
    ///
    /// The function errors when the plane index is out of range, the scale is
    /// invalid or the decode fails.
    ///
    /// A backend that overrides this method and cannot serve a region must
    /// return [`SurfaceError::Unsupported`].
    /// It must never return the whole plane for a region view.
    fn render(&self, view: View) -> Result<DecodedImage, SurfaceError> {
        let full = self.render_whole(view.plane, view.scale)?;
        match view.region {
            None => Ok(full),
            Some(region) => {
                // `render_whole` applied the scale, so the crop runs at that
                // same scale.
                crop_and_resample(&full, region, view.scale, view.scale)
            }
        }
    }

    /// A byte stream for progressive decoding, or `None` when the source has
    /// no progressive path.
    ///
    /// A source whose bytes improve the picture as they arrive (JPEG XL)
    /// returns one; the loader reads it and sends previews.
    /// `None` means the load happens in one step.
    ///
    /// # Errors
    ///
    /// The function errors when the stream cannot be opened.
    fn progressive(&self) -> Result<Option<Box<dyn ChunkSource>>, SurfaceError>;

    /// Whether the loader sends a fast low-resolution preview before the
    /// full-quality render.
    /// True for slow rasterizers (rasterizing takes a moment).
    /// False for images (already fast, and JXL has its own progressive path).
    fn previews(&self) -> bool {
        false
    }

    /// Whether a settled zoom can render a sharper picture of the same plane.
    ///
    /// True for a resolution-independent source (a vector file, a document
    /// page): the image view re-renders the visible region at the settled
    /// scale.
    /// False for a raster image: a re-render only decodes the same pixels
    /// again, so the view scales the existing texture on the GPU instead.
    fn can_refine(&self) -> bool {
        false
    }

    /// The part of `plane` that a fresh open frames, in surface units, or
    /// `None` to frame the whole plane.
    ///
    /// A book answers its first spread: the whole document stands far taller
    /// than the window, and the reader starts on the cover page.
    /// The fit measures this box instead of the plane, and the view scrolls to
    /// the top of the plane so the box shows first.
    fn open_box(&self, _plane: usize) -> Option<Size> {
        None
    }

    /// How many spreads the plane stacks, or 0 for a source that shows no
    /// spread column.
    ///
    /// A book answers its row count (see [`crate::book`]), so the image view
    /// can step from one spread to the next.
    fn spread_count(&self) -> usize {
        0
    }

    /// The box of spread `spread` inside the plane, in surface units, or
    /// `None` when the source holds no such spread.
    ///
    /// A book answers the full width of the plane and one page tall, at the
    /// row the spread stacks in, so the image view can aim a step at it.
    fn spread_box(&self, _spread: usize) -> Option<Region> {
        None
    }
}

/// Why a surface could not open, describe or render a plane.
#[derive(Debug, Display, Error, From)]
pub enum SurfaceError {
    /// The file could not be opened.
    #[display("failed to open file")]
    Open(#[from] std::io::Error),
    /// The content is not a supported format.
    #[display("{_0}")]
    Sign(
        #[from]
        #[error(source)]
        SignError,
    ),
    /// The decoder rejected the bytes.
    #[display("decode failed")]
    Decode(#[from] DecodeError),
    /// The plane index is out of range.
    #[display("plane out of range")]
    Plane(#[error(ignore)] usize),
    /// The render scale is not a positive, finite number.
    #[display("invalid render scale")]
    Scale(#[error(ignore)] f32),
    /// The rendered buffer did not match its dimensions.
    #[display("rendered buffer does not match its dimensions")]
    Buffer,
    /// The rasterizer failed on this plane (a malformed page, a panicking
    /// renderer).
    /// The message names the cause.
    #[display("render failed: {_0}")]
    #[from(ignore)]
    Render(#[error(ignore)] String),
    /// The backend cannot render this view (a region anchor it lacks, a page
    /// too large for its buffers).
    /// The message names the cause.
    #[display("unsupported view: {_0}")]
    Unsupported(#[error(ignore)] String),
}

/// Sniff the file format and open the matching surface.
///
/// A PDF routes on the `%PDF-` magic to the multi-plane PDF backend, so a
/// misnamed file still opens.
/// A `.typ` source routes by extension to the Typst backend, which compiles it
/// to PDF bytes and hands them to that same backend.
/// A vector file keeps its parsed tree, so a region render of the image view
/// pays only for the pixels it shows.
/// Every other content goes to the raster backend, which dispatches JPEG XL
/// and the `image` crate formats by the sniffed bytes.
///
/// # Errors
///
/// The function errors when the file cannot be read, or when the content is
/// not a decodable image, PDF or vector file.
pub fn open_surface(path: &Path) -> Result<Box<dyn Surface>, SurfaceError> {
    // A PDF is a multi-plane surface (one plane per page).
    // The backend holds the hayro dependency in `pdf.rs`.
    if crate::pdf::is_pdf(path) {
        return Ok(Box::new(crate::pdf::PdfSurface::open(path)?));
    }
    // Typst: the extension decides (.typ is plain UTF-8 text, so there is no
    // magic to sniff). The backend compiles the file and renders the exported
    // pages through the PDF path.
    if crate::typst::is_typst(path) {
        return Ok(Box::new(crate::typst::TypstSurface::open(path)?));
    }
    // The sniff doubles as a validity check; render() reports the decode
    // error otherwise.
    sign::format(path)?;
    #[cfg(feature = "svg")]
    if crate::svg::sniff(path) {
        return Ok(Box::new(VectorSurface::open(path)?));
    }
    Ok(Box::new(ImageSurface::new(path)))
}

/// Open a file and, when its source holds more than one plane, show it as a
/// book (see [`crate::book`]).
///
/// The grid thumbnail keeps the plain surface from [`open_surface`], so a cell
/// shows page 0.
/// The image view takes this path, so a document becomes one tall figure with
/// every spread stacked.
///
/// # Errors
///
/// The function errors when the file cannot be read or the content is not a
/// decodable image, PDF or vector file.
pub fn open_book_surface(path: &Path) -> Result<Box<dyn Surface>, SurfaceError> {
    let surface = open_surface(path)?;
    if surface.plane_count() <= 1 {
        return Ok(surface);
    }
    let inner: Arc<dyn Surface> = surface.into();
    Ok(Box::new(crate::book::BookSurface::new(inner)?))
}

/// A raster image file (JPEG XL, PNG, JPEG, WebP, ...) which always holds one
/// plane.
#[derive(Debug)]
pub struct ImageSurface {
    path: PathBuf,
    jxl: bool,
}

impl ImageSurface {
    fn new(path: &Path) -> Self {
        ImageSurface {
            path: path.to_path_buf(),
            jxl: sign::is_jxl(path),
        }
    }

    /// The decoded raster, capped to the uploadable
    /// [`crate::MAX_TEXTURE_SIDE`].
    fn decode(&self) -> Result<DecodedImage, SurfaceError> {
        Ok(crate::decode_raster(&self.path)?)
    }
}

impl Surface for ImageSurface {
    fn plane_count(&self) -> usize {
        1
    }

    fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
        if plane != 0 {
            return Err(SurfaceError::Plane(plane));
        }
        let (width, height) = sign::dimensions(&self.path, self.jxl)?;
        // A raster that no GPU upload takes decodes capped, so the plane it
        // renders is the capped one.
        let (w, h) = capped(width, height);
        Ok(Size {
            w: w.get() as f32,
            h: h.get() as f32,
        })
    }

    fn render_whole(&self, plane: usize, scale: f32) -> Result<DecodedImage, SurfaceError> {
        if plane != 0 {
            return Err(SurfaceError::Plane(plane));
        }
        if !valid_scale(scale) {
            return Err(SurfaceError::Scale(scale));
        }
        let decoded = self.decode()?;
        // A native-size request returns the decode as is; any other scale
        // resamples the whole image here (the view scales textures on the
        // GPU instead).
        if is_native(scale) {
            return Ok(decoded);
        }
        let whole = Size {
            w: decoded.width.as_f32(),
            h: decoded.height.as_f32(),
        };
        resample(&decoded, whole.whole(), scale)
    }

    fn render(&self, view: View) -> Result<DecodedImage, SurfaceError> {
        // Decode once at native size, then crop (cheap) and resample only the
        // requested region. This beats the whole-plane default, which would
        // resample the whole image to reach a small region.
        if view.plane != 0 {
            return Err(SurfaceError::Plane(view.plane));
        }
        if !valid_scale(view.scale) {
            return Err(SurfaceError::Scale(view.scale));
        }
        let decoded = self.decode()?;
        let whole = Size {
            w: decoded.width.as_f32(),
            h: decoded.height.as_f32(),
        };
        match view.region {
            // A native-size request returns the decode as is.
            None if is_native(view.scale) => Ok(decoded),
            None => resample(&decoded, whole.whole(), view.scale),
            // The decoded plane is at native size (scale 1.0), so its pixel
            // space equals its surface-unit space.
            Some(region) => crop_and_resample(&decoded, region, 1.0, view.scale),
        }
    }

    fn progressive(&self) -> Result<Option<Box<dyn ChunkSource>>, SurfaceError> {
        // Only a JPEG XL source decodes progressively.
        if !self.jxl {
            return Ok(None);
        }
        let file = std::fs::File::open(&self.path)?;
        // SAFETY: the code only reads the map, and the file behind it stays
        // open for the length of the load.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Some(Box::new(FileChunks { map, pos: 0 })))
    }
}

/// A vector file (SVG, SVGZ) which always holds one plane.
///
/// The plane measures the base raster of the tree (see [`crate::svg`]): a
/// vector file has no pixel size of its own, so the code bakes one between
/// 1024 and [`crate::MAX_TEXTURE_SIDE`] on the long side.
/// The tree stays parsed, so every render paints from it at any scale.
#[cfg(feature = "svg")]
#[derive(Debug)]
pub struct VectorSurface {
    vector: std::sync::Arc<crate::svg::Vector>,
}

#[cfg(feature = "svg")]
impl VectorSurface {
    /// Parse `path` into a surface.
    ///
    /// # Errors
    ///
    /// The function errors when the file cannot be read or parsed.
    pub fn open(path: &Path) -> Result<Self, SurfaceError> {
        Ok(VectorSurface {
            vector: std::sync::Arc::new(crate::svg::parse(path)?),
        })
    }
}

#[cfg(feature = "svg")]
impl Surface for VectorSurface {
    fn plane_count(&self) -> usize {
        1
    }

    fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
        if plane != 0 {
            return Err(SurfaceError::Plane(plane));
        }
        let (width, height) = self.vector.base_size();
        Ok(Size {
            w: width.get() as f32,
            h: height.get() as f32,
        })
    }

    fn render_whole(&self, plane: usize, scale: f32) -> Result<DecodedImage, SurfaceError> {
        if plane != 0 {
            return Err(SurfaceError::Plane(plane));
        }
        if !valid_scale(scale) {
            return Err(SurfaceError::Scale(scale));
        }
        let region = self.bounds(0)?.whole();
        Ok(self
            .vector
            .render_region(crate::region::Crop::from(region), scale)?)
    }

    fn render(&self, view: View) -> Result<DecodedImage, SurfaceError> {
        if view.plane != 0 {
            return Err(SurfaceError::Plane(view.plane));
        }
        if !valid_scale(view.scale) {
            return Err(SurfaceError::Scale(view.scale));
        }
        // The tree has a region anchor, so it paints the requested rect and
        // never rasterizes the whole document for a window-sized patch.
        let region = view.region.unwrap_or(self.bounds(0)?.whole());
        Ok(self
            .vector
            .render_region(crate::region::Crop::from(region), view.scale)?)
    }

    fn progressive(&self) -> Result<Option<Box<dyn ChunkSource>>, SurfaceError> {
        // A vector file parses and paints in one step.
        Ok(None)
    }

    // The tree paints at any scale, so a settled zoom pays for a sharper patch
    // of the same picture.
    fn can_refine(&self) -> bool {
        true
    }
}

/// Whether the scale is a positive, finite number.
fn valid_scale(scale: f32) -> bool {
    scale.is_finite() && scale > 0.0
}

/// Whether `scale` asks for the natural size.
fn is_native(scale: f32) -> bool {
    (scale - 1.0).abs() <= 1e-3
}

/// A raster header size, capped to the uploadable
/// [`crate::MAX_TEXTURE_SIDE`].
///
/// The decode applies the same cap, so the plane a raster surface reports and
/// the pixels it renders measure the same picture.
fn capped(width: ImgW, height: ImgH) -> (ImgW, ImgH) {
    let (w, h) = (width.get(), height.get());
    if w.max(h) > crate::MAX_TEXTURE_SIDE {
        let (w, h) = crate::fitted(w, h, crate::MAX_TEXTURE_SIDE);
        (ImgW::new(w), ImgH::new(h))
    } else {
        (width, height)
    }
}

/// Crop `region` (surface units) out of `img`, then scale it to `scale`
/// output pixels per surface unit.
///
/// `img` holds the whole plane at `img_scale` pixels per surface unit.
/// The crop clamps to the image, so a view that runs off the edge returns the
/// part that overlaps instead of an error.
/// The resize is [`crate::resample_rgba`], the same SIMD path as every other
/// pixel resize of the viewer.
///
/// # Errors
///
/// The function errors when the buffer does not match its dimensions, or when
/// the crop is empty.
fn crop_and_resample(
    img: &DecodedImage,
    region: Region,
    img_scale: f32,
    scale: f32,
) -> Result<DecodedImage, SurfaceError> {
    let (iw, ih) = (img.width.get(), img.height.get());
    if img.data.len() != iw as usize * ih as usize * 4 {
        return Err(SurfaceError::Buffer);
    }
    // Region in `img` pixels: surface units times the buffer's own scale.
    let px = |v: f32| (v * img_scale).round();
    let x = px(region.x.max(0.0)).clamp(0.0, iw as f32) as u32;
    let y = px(region.y.max(0.0)).clamp(0.0, ih as f32) as u32;
    // Do not trust the caller's region size: a region past the edge crops.
    let cw = (px(region.w.max(0.0)) as u32).min(iw - x);
    let ch = (px(region.h.max(0.0)) as u32).min(ih - y);
    if cw == 0 || ch == 0 {
        return Err(SurfaceError::Buffer);
    }
    // Output size in target-ready pixels, from the (clamped) crop extent.
    let tw = (region.w * scale).round().max(1.0) as u32;
    let th = (region.h * scale).round().max(1.0) as u32;
    let out = if tw == cw && th == ch {
        crop_rows(&img.data, iw, x, y, cw, ch)
    } else {
        let cropped = crop_rows(&img.data, iw, x, y, cw, ch);
        crate::resample_rgba(&cropped, cw, ch, tw, th)
    };
    Ok(DecodedImage {
        width: ImgW::new(tw),
        height: ImgH::new(th),
        data: out,
    })
}

/// Copy the `cw` x `ch` rect at (`x`, `y`) out of a row-major RGBA8 buffer.
fn crop_rows(rgba: &[u8], width: u32, x: u32, y: u32, cw: u32, ch: u32) -> Vec<u8> {
    let stride = width as usize * 4;
    let row = cw as usize * 4;
    let mut out = Vec::with_capacity(row * ch as usize);
    for line in 0..ch as usize {
        let start = (y as usize + line) * stride + x as usize * 4;
        out.extend_from_slice(&rgba[start..start + row]);
    }
    out
}

/// Scale the whole of `img` to `scale` output pixels per surface unit.
fn resample(img: &DecodedImage, region: Region, scale: f32) -> Result<DecodedImage, SurfaceError> {
    crop_and_resample(img, region, 1.0, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::TempDir::new().unwrap()
    }

    fn write_png(dir: &Path, name: &str, w: u32, h: u32) -> PathBuf {
        let path = dir.join(name);
        image::DynamicImage::new_rgb8(w, h).save(&path).unwrap();
        path
    }

    /// A red square drawn as a 100 x 50 vector file.
    const SQUARE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">
        <rect width="50" height="50" fill="#ff0000"/>
        <rect x="50" width="50" height="50" fill="#0000ff"/>
    </svg>"##;

    fn write_svg(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn image_surface_reports_bounds_and_renders() {
        let dir = temp_dir();
        let path = write_png(dir.path(), "img.png", 3, 2);
        let surface = open_surface(&path).unwrap();
        assert_eq!(surface.plane_count(), 1);
        assert_eq!(
            surface.bounds(0).unwrap(),
            Size { w: 3.0, h: 2.0 },
            "natural pixel size"
        );
        let decoded = surface.render(View::whole(0, 1.0)).unwrap();
        assert_eq!(
            (decoded.width, decoded.height),
            (ImgW::new(3), ImgH::new(2))
        );
        assert_eq!(decoded.data.len(), 3 * 2 * 4);
    }

    #[test]
    fn image_surface_reads_webp() {
        // Lossless encode via image-webp, then decode back through the
        // surface API (the format comes from the bytes, not the extension).
        let dir = temp_dir();
        let path = dir.path().join("img.webp");
        image::DynamicImage::new_rgb8(5, 4)
            .save_with_format(&path, image::ImageFormat::WebP)
            .unwrap();
        let decoded = open_surface(&path)
            .unwrap()
            .render(View::whole(0, 1.0))
            .unwrap();
        assert_eq!(
            (decoded.width, decoded.height),
            (ImgW::new(5), ImgH::new(4))
        );
        assert_eq!(decoded.data.len(), 5 * 4 * 4);
    }

    #[test]
    fn image_surface_rejects_other_planes() {
        let dir = temp_dir();
        let path = write_png(dir.path(), "img.png", 2, 2);
        let surface = open_surface(&path).unwrap();
        assert!(surface.bounds(1).is_err());
        assert!(surface.render(View::whole(1, 1.0)).is_err());
    }

    #[test]
    fn image_surface_rejects_bad_scales() {
        let dir = temp_dir();
        let path = write_png(dir.path(), "img.png", 2, 2);
        let surface = open_surface(&path).unwrap();
        assert!(surface.render(View::whole(0, 0.0)).is_err());
        assert!(surface.render(View::whole(0, f32::NAN)).is_err());
        assert!(surface.render(View::whole(0, -1.0)).is_err());
    }

    #[test]
    fn image_surface_scales_off_native_size() {
        let dir = temp_dir();
        let path = write_png(dir.path(), "img.png", 8, 4);
        let decoded = open_surface(&path)
            .unwrap()
            .render(View::whole(0, 0.5))
            .unwrap();
        assert_eq!(
            (decoded.width, decoded.height),
            (ImgW::new(4), ImgH::new(2))
        );
        let decoded = open_surface(&path)
            .unwrap()
            .render(View::whole(0, 2.0))
            .unwrap();
        assert_eq!(
            (decoded.width, decoded.height),
            (ImgW::new(16), ImgH::new(8))
        );
    }

    #[test]
    fn image_surface_renders_a_region() {
        // A region view crops and scales: the left half of an 8x4 image at
        // scale 1.0 is 4x4 pixels.
        let dir = temp_dir();
        let path = write_png(dir.path(), "img.png", 8, 4);
        let region = Region {
            x: 0.0,
            y: 0.0,
            w: 4.0,
            h: 4.0,
        };
        let decoded = open_surface(&path)
            .unwrap()
            .render(View::region(0, region, 1.0))
            .unwrap();
        assert_eq!(
            (decoded.width, decoded.height),
            (ImgW::new(4), ImgH::new(4))
        );
        assert_eq!(decoded.data.len(), 4 * 4 * 4);
    }

    #[test]
    fn a_region_off_the_edge_crops_instead_of_failing() {
        let dir = temp_dir();
        let path = write_png(dir.path(), "img.png", 8, 4);
        let region = Region {
            x: 6.0,
            y: 2.0,
            w: 8.0,
            h: 8.0,
        };
        let decoded = open_surface(&path)
            .unwrap()
            .render(View::region(0, region, 1.0))
            .unwrap();
        assert_eq!(
            (decoded.width, decoded.height),
            (ImgW::new(8), ImgH::new(8)),
            "the output measures the requested region"
        );
        // The pixels past the edge repeat the crop, so the buffer stays
        // valid.
        assert_eq!(decoded.data.len(), 8 * 8 * 4);
    }

    #[test]
    fn image_surface_streams_only_jxl() {
        // The code honors a .jxl extension even for unknown content: JPEG XL
        // container files start with a box header, not the codestream magic.
        let dir = temp_dir();
        let container = dir.path().join("container.jxl");
        std::fs::write(&container, [0x00, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ']).unwrap();
        let surface = ImageSurface::new(&container);
        assert!(surface.progressive().unwrap().is_some());

        let png = write_png(dir.path(), "img.png", 1, 1);
        let surface = ImageSurface::new(&png);
        assert!(surface.progressive().unwrap().is_none());
    }

    #[test]
    fn a_file_chunk_source_streams_the_whole_file() {
        // The stream reads the file to its end and reports a clean end of
        // stream, so the loader needs no file path of its own.
        let dir = temp_dir();
        let container = dir.path().join("bytes.jxl");
        std::fs::write(&container, [0xffu8, 0x0a, 1, 2, 3, 4, 5]).unwrap();
        let mut source = ImageSurface::new(&container)
            .progressive()
            .unwrap()
            .expect("a JXL source streams");
        let mut got = Vec::new();
        let mut buf = [0u8; 3];
        loop {
            let n = source.read(&mut buf).unwrap();
            assert!(n <= buf.len());
            if n == 0 {
                break;
            }
            got.extend_from_slice(&buf[..n]);
        }
        assert_eq!(got, [0xff, 0x0a, 1, 2, 3, 4, 5]);
    }

    /// An in-memory [`ChunkSource`], to prove progressive decoding needs no
    /// file path.
    struct MemoryChunks {
        data: Vec<u8>,
        pos: usize,
    }

    impl ChunkSource for MemoryChunks {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = buf.len().min(self.data.len() - self.pos);
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    #[test]
    fn a_chunk_source_can_be_in_memory() {
        let mut src = MemoryChunks {
            data: b"hello".to_vec(),
            pos: 0,
        };
        let mut buf = [0u8; 2];
        assert_eq!(src.read(&mut buf).unwrap(), 2);
        assert_eq!(&buf, b"he");
        assert_eq!(src.read(&mut buf).unwrap(), 2);
        assert_eq!(&buf, b"ll");
        assert_eq!(src.read(&mut buf).unwrap(), 1);
        assert_eq!(src.read(&mut buf).unwrap(), 0, "end of stream");
    }

    #[test]
    fn open_surface_sniffs_content_over_extension() {
        // Content-first dispatch: a PNG renamed to .jxl decodes as PNG, a
        // JXL codestream renamed to .png is still JXL.
        let dir = temp_dir();
        let png = write_png(dir.path(), "real.png", 2, 3);

        let renamed_jxl = dir.path().join("renamed.jxl");
        std::fs::copy(&png, &renamed_jxl).unwrap();
        let decoded = open_surface(&renamed_jxl)
            .unwrap()
            .render(View::whole(0, 1.0))
            .unwrap();
        assert_eq!(
            (decoded.width, decoded.height),
            (ImgW::new(2), ImgH::new(3))
        );

        // A truncated JXL codestream renamed to .png still routes to the JXL
        // decoder (the caller then reports the decode failure).
        let renamed_png = dir.path().join("renamed.png");
        std::fs::write(&renamed_png, [0xffu8, 0x0a, 0x01, 0x02]).unwrap();
        assert!(open_surface(&renamed_png).is_ok());
        assert!(
            open_surface(&renamed_png)
                .unwrap()
                .render(View::whole(0, 1.0))
                .is_err()
        );
    }

    #[test]
    fn open_surface_fails_cleanly_on_garbage() {
        let dir = temp_dir();
        let path = dir.path().join("x.png");
        std::fs::write(&path, b"garbage").unwrap();
        assert!(open_surface(&path).is_err());
    }

    #[test]
    fn open_surface_fails_cleanly_on_a_missing_file() {
        let dir = temp_dir();
        assert!(open_surface(&dir.path().join("gone.png")).is_err());
    }

    #[cfg(feature = "svg")]
    #[test]
    fn a_vector_surface_measures_its_base_raster() {
        let dir = temp_dir();
        let path = write_svg(dir.path(), "drawing.svg", SQUARE);
        let surface = open_surface(&path).unwrap();
        assert_eq!(surface.plane_count(), 1);
        // Declared 100 x 50 bakes at 1024 x 512, and that pair is the source
        // size the viewer fits and zooms.
        assert_eq!(
            surface.bounds(0).unwrap(),
            Size {
                w: 1024.0,
                h: 512.0
            }
        );
        let base = surface.render(View::whole(0, 1.0)).unwrap();
        assert_eq!((base.width, base.height), (ImgW::new(1024), ImgH::new(512)));
        // The left half paints red.
        assert_eq!(&base.data[..4], &[255, 0, 0, 255]);
        assert!(surface.progressive().unwrap().is_none());
        assert!(!surface.previews());
        // The tree paints at any scale, so the view repaints a settled zoom.
        assert!(surface.can_refine());
    }

    #[cfg(feature = "svg")]
    #[test]
    fn a_vector_surface_renders_a_thumbnail_scale() {
        // The grid asks for the thumbnail size, and the tree paints it
        // directly instead of painting the base raster first.
        let dir = temp_dir();
        let path = write_svg(dir.path(), "drawing.svg", SQUARE);
        let thumb = open_surface(&path)
            .unwrap()
            .render(View::whole(0, 0.5))
            .unwrap();
        assert_eq!(
            (thumb.width, thumb.height),
            (ImgW::new(512), ImgH::new(256))
        );
        assert_eq!(thumb.data.len(), 512 * 256 * 4);
    }

    #[cfg(feature = "svg")]
    #[test]
    fn a_vector_surface_renders_the_region_the_window_shows() {
        let dir = temp_dir();
        let path = write_svg(dir.path(), "split.svg", SQUARE);
        let surface = open_surface(&path).unwrap();
        // The region arrives in base raster pixels, so the right half starts
        // at x = 512.
        let right = Region {
            x: 512.0,
            y: 0.0,
            w: 512.0,
            h: 512.0,
        };
        let patch = surface.render(View::region(0, right, 1.0)).unwrap();
        assert_eq!(
            (patch.width, patch.height),
            (ImgW::new(512), ImgH::new(512))
        );
        assert_eq!(&patch.data[..4], &[0, 0, 255, 255]);
        // A deeper zoom paints the same window rect at more pixels per source
        // pixel.
        let zoomed = surface.render(View::region(0, right, 2.0)).unwrap();
        assert_eq!(
            (zoomed.width, zoomed.height),
            (ImgW::new(1024), ImgH::new(1024))
        );
    }

    #[cfg(not(feature = "svg"))]
    #[test]
    fn a_vector_file_needs_the_svg_feature() {
        let dir = temp_dir();
        let path = write_svg(dir.path(), "drawing.svg", SQUARE);
        assert!(open_surface(&path).is_err());
    }

    #[test]
    fn a_region_converts_from_a_crop_and_back() {
        // The image view computes its patch as a `region::Crop`, and a surface
        // takes a `Region`, so the two must keep the same fields.
        let crop = crate::region::Crop {
            x: 1.0,
            y: 2.0,
            w: 3.0,
            h: 4.0,
        };
        let region = Region::from(crop);
        assert_eq!(
            region,
            Region {
                x: 1.0,
                y: 2.0,
                w: 3.0,
                h: 4.0
            }
        );
        assert_eq!(crate::region::Crop::from(region), crop);
    }

    /// A backend that writes only `render_whole`: it inherits region support
    /// from the trait default.
    struct WholeOnly;

    impl Surface for WholeOnly {
        fn plane_count(&self) -> usize {
            1
        }

        fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
            if plane != 0 {
                return Err(SurfaceError::Plane(plane));
            }
            Ok(Size { w: 8.0, h: 4.0 })
        }

        fn render_whole(&self, plane: usize, scale: f32) -> Result<DecodedImage, SurfaceError> {
            if plane != 0 {
                return Err(SurfaceError::Plane(plane));
            }
            let (w, h) = (
                ImgW::new((8.0 * scale).round().max(1.0) as u32),
                ImgH::new((4.0 * scale).round().max(1.0) as u32),
            );
            Ok(DecodedImage {
                width: w,
                height: h,
                data: vec![0u8; w.get() as usize * h.get() as usize * 4],
            })
        }

        fn progressive(&self) -> Result<Option<Box<dyn ChunkSource>>, SurfaceError> {
            Ok(None)
        }
    }

    #[test]
    fn default_render_crops_a_region_from_the_whole_plane() {
        // WholeOnly has no region anchor; the trait default must render the
        // whole plane and crop the requested region out of it.
        let region = Region {
            x: 2.0,
            y: 1.0,
            w: 4.0,
            h: 2.0,
        };
        let out = WholeOnly.render(View::region(0, region, 1.0)).unwrap();
        assert_eq!((out.width, out.height), (ImgW::new(4), ImgH::new(2)));
        assert_eq!(out.data.len(), 4 * 2 * 4);
    }

    #[test]
    fn default_render_scales_a_region() {
        let region = Region {
            x: 0.0,
            y: 0.0,
            w: 4.0,
            h: 2.0,
        };
        let out = WholeOnly.render(View::region(0, region, 2.0)).unwrap();
        assert_eq!((out.width, out.height), (ImgW::new(8), ImgH::new(4)));
    }

    #[test]
    fn default_render_rejects_a_bad_plane() {
        assert!(WholeOnly.render(View::whole(7, 1.0)).is_err());
    }

    /// A backend with no region anchor that overrides `render` and refuses a
    /// region. It must fail loudly, never return the whole plane.
    struct NoAnchor;

    impl Surface for NoAnchor {
        fn plane_count(&self) -> usize {
            1
        }

        fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
            if plane != 0 {
                return Err(SurfaceError::Plane(plane));
            }
            Ok(Size { w: 8.0, h: 4.0 })
        }

        fn render_whole(&self, _plane: usize, _scale: f32) -> Result<DecodedImage, SurfaceError> {
            Err(SurfaceError::Unsupported("test backend".to_owned()))
        }

        fn render(&self, view: View) -> Result<DecodedImage, SurfaceError> {
            if view.region.is_some() {
                return Err(SurfaceError::Unsupported("no region anchor".to_owned()));
            }
            self.render_whole(view.plane, view.scale)
        }

        fn progressive(&self) -> Result<Option<Box<dyn ChunkSource>>, SurfaceError> {
            Ok(None)
        }
    }

    #[test]
    fn a_backend_without_an_anchor_fails_loudly_on_a_region() {
        let region = Region {
            x: 0.0,
            y: 0.0,
            w: 2.0,
            h: 2.0,
        };
        let err = NoAnchor.render(View::region(0, region, 1.0));
        assert!(matches!(err, Err(SurfaceError::Unsupported(_))));
    }

    #[test]
    fn default_backend_does_not_refine() {
        // A backend that does not override `can_refine` is a raster source:
        // the image view never re-renders it on a zoom settle.
        assert!(!WholeOnly.can_refine());
        assert!(!NoAnchor.can_refine());
    }

    #[test]
    fn a_raster_image_does_not_refine() {
        let dir = temp_dir();
        let path = write_png(dir.path(), "img.png", 4, 2);
        let surface = open_surface(&path).unwrap();
        assert!(!surface.can_refine());
    }
}
