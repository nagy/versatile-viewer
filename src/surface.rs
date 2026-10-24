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

use std::path::{Path, PathBuf};

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

    /// Rasterize `view`: a plane, an optional region of it, and a scale.
    ///
    /// The region arrives in surface units, and the scale is output pixels per
    /// surface unit, so a caller asks for exactly the pixels it shows.
    /// A backend without a region anchor renders the whole plane and crops.
    ///
    /// # Errors
    ///
    /// The function errors when the plane index is out of range, the scale is
    /// invalid or the decode fails.
    fn render(&self, view: View) -> Result<DecodedImage, SurfaceError>;

    /// The file behind this surface, when a byte stream can drive a
    /// progressive decode.
    ///
    /// A source with no progressive path answers None, and the load happens in
    /// one step.
    fn stream_path(&self) -> Option<&Path>;

    /// Whether the loader sends a fast low-resolution preview before the
    /// full-quality render.
    /// True for slow rasterizers (rasterizing takes a moment).
    /// False for images (already fast, and JXL has its own progressive path).
    fn previews(&self) -> bool {
        false
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
    /// The backend cannot render this view (a region anchor it lacks, a page
    /// too large for its buffers).
    /// The message names the cause.
    #[display("unsupported view: {_0}")]
    Unsupported(#[error(ignore)] String),
}

/// Sniff the file format and open the matching surface.
///
/// A vector file keeps its parsed tree, so a region render of the image view
/// pays only for the pixels it shows.
/// Every other content goes to the raster backend, which dispatches JPEG XL
/// and the `image` crate formats by the sniffed bytes.
///
/// # Errors
///
/// The function errors when the file cannot be read, or when the content is
/// not a decodable image.
pub fn open_surface(path: &Path) -> Result<Box<dyn Surface>, SurfaceError> {
    // The sniff doubles as a validity check; render() reports the decode
    // error otherwise.
    sign::format(path)?;
    #[cfg(feature = "svg")]
    if crate::svg::sniff(path) {
        return Ok(Box::new(VectorSurface::open(path)?));
    }
    Ok(Box::new(ImageSurface::new(path)))
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

    fn render(&self, view: View) -> Result<DecodedImage, SurfaceError> {
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

    fn stream_path(&self) -> Option<&Path> {
        // Only a JPEG XL source decodes progressively.
        self.jxl.then_some(self.path.as_path())
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

    fn stream_path(&self) -> Option<&Path> {
        None
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
        assert!(surface.stream_path().is_some());

        let png = write_png(dir.path(), "img.png", 1, 1);
        let surface = ImageSurface::new(&png);
        assert!(surface.stream_path().is_none());
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
        assert!(surface.stream_path().is_none());
        assert!(!surface.previews());
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
}
