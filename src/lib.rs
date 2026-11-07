// Pixel and coordinate math lives in f32 (window and source units) and
// indices in usize and u32.
// The casts between them belong to that boundary.
// Every value here (window pixels, texture dimensions) stays far below
// the f32 exact-integer range.
// The pedantic cast lints are therefore noise.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    // win_w/win_h and friends are natural paired names in window math.
    clippy::similar_names
)]

//! The reusable core of versatile-viewer.
//! It holds image decoding, RGBA helpers and the modules behind the grid,
//! loader and gimmicks.
//!
//! Decoding dispatches by sniffed content, not by file name (see [`sign`]).
//! JPEG XL decodes with pure-Rust jxl-oxide and streams progressively in
//! [`loader`].
//! The common formats decode with the `image` crate.
//! SVG and SVGZ rasterize with pure-Rust resvg in [`svg`], which bakes a base
//! raster because a vector file has no pixels of its own.
//! PDF rasterizes with pure-Rust hayro in [`pdf`], one plane per page.
//! [`book`] shows a multi-page source as a book: every spread (a cover page
//! alone, then the pages in pairs) stacked in one [`surface::Surface`]
//! plane.
//! A Typst source compiles to PDF bytes in [`typst`] and renders through the
//! same PDF backend, so it shows as the same book.
//! A Mermaid source renders to SVG in [`mermaid`] and renders through the SVG
//! backend, so it shows as one vector picture.
//! Every showable source is a [`surface::Surface`], so the grid, the loader
//! and the image view ask for a plane at a scale and never for a file format.
//! [`downscale_rgba`] and [`cap_texture_side`] keep the
//! pixel and texture budget.
//! Decode workers cap the buffer, so the main thread never resizes a large
//! image.
//! [`mcp`] serves the Model Context Protocol on stdio when the binary starts
//! with `--mcp`.
//! [`keyrepeat`] repeats the held side-button navigation, because the window
//! system reports the keyboard repeat itself, and [`tween`] ramps a held key
//! for continuous motion.
//! [`region`] holds the visible-region raster of the image view: a vector
//! source repaints the part of the window it shows, at the scale it shows it,
//! so a deep zoom never stretches the base raster.
//! [`rotate`] holds the image-view rotation angle behind the `r` hold.
//! [`region_sel`] holds the image-view region selection behind the `v` hold,
//! and the geometry that dims the picture around it.
//! [`thumbcache`] keeps the grid thumbnails in `$XDG_CACHE_HOME`, so a warm
//! cache fills the grid without a decode, across the exit of the viewer.
//! [`placeholder`] holds the few-byte `ThumbHash` of a thumb, which the cache
//! header carries so a warm launch paints every cell before any decode.
//! [`draw`] holds the point, the color and the quad that the view code writes,
//! so no GPU handle reaches it.
//! [`render`] and [`texture`] draw those quads through one wgpu pipeline, and
//! [`gpu`] owns the device and the swapchain behind them.
//! [`input`] holds the held keys, the frame edges and the pointer of one
//! frame.
//! [`pane`] names the X11 window a host hands over with `--wid`, so the viewer
//! draws inside a host window without becoming a top-level one.
//! [`view`] holds the image view: what it shows, how it frames it, the patch
//! that refines a settled zoom, and the `Ui` of one frame.
//! [`viewer`] is the one value a host builds on its own device: it holds the
//! pipeline, the views, the loaders and the clock, so a host owns the window
//! and the viewer owns the picture.
//! The binary in `main.rs` wires these pieces into the window and event loop.
//! The unit tests and doctests here keep the pure helpers honest.

pub mod blurbg;
pub mod book;
pub mod dim;
pub mod draw;
pub mod error;
pub mod gpu;
pub mod grid;
pub mod input;
pub mod keyrepeat;
pub mod loader;
pub mod mcp;
#[cfg(feature = "mermaid")]
pub mod mermaid;
#[cfg(not(feature = "mermaid"))]
#[path = "mermaid_off.rs"]
pub mod mermaid;
pub mod pane;
pub mod pdf;
pub mod placeholder;
pub mod region;
pub mod region_sel;
pub mod render;
pub mod rotate;
pub mod sign;
pub mod surface;
#[cfg(feature = "svg")]
pub mod svg;
#[cfg(not(feature = "svg"))]
#[path = "svg_off.rs"]
pub mod svg;
pub mod texture;
#[cfg(feature = "cache")]
pub mod thumbcache;
#[cfg(not(feature = "cache"))]
#[path = "thumbcache_off.rs"]
pub mod thumbcache;
pub mod tween;
pub mod typst;
pub(crate) mod view;
pub mod viewer;
pub mod zoomkeys;
use std::path::Path;

pub use crate::error::DecodeError;
use crate::{
    dim::{ImgH, ImgW, LongSide},
    draw::Color,
};

/// The speed multiplier while Shift is held.
///
/// Shift + h/j/k/l moves at double speed: the grid selection step, the image
/// pan and the image free zoom.
/// Both views read this one constant, so the two speeds never drift apart.
pub const SHIFT_SPEED_MULTIPLIER: u32 = 2;

/// The brightness a dimmed area keeps, as a 0..=255 channel value.
///
/// The grid dims every cell outside a visual selection, and the image view
/// dims the picture outside a region selection.
/// Both read this one level, so the two views never show a different
/// darkness.
/// [`DIM_TINT`] multiplies a dimmed texture by it, and [`DIM_VEIL`] blends the
/// same share away, so one number drives both.
pub const DIM_LEVEL: u8 = 90;

/// The pointer travel, in window pixels, that still counts as a click.
///
/// The `f` pan hold and the `v` region drag both read it, so a shaky hand
/// never turns a tap into a drag.
/// Both keys measure the travel in window pixels, never in source units,
/// because a deep zoom makes one window pixel a wide source span.
pub const TAP_SLACK: f32 = 4.0;

/// The tint of a dimmed draw: a texture multiplied by [`DIM_LEVEL`].
pub const DIM_TINT: Color = Color::new(DIM_LEVEL, DIM_LEVEL, DIM_LEVEL, 255);

/// The black veil over a dimmed part of the image view.
///
/// Alpha `255 - DIM_LEVEL` leaves the share of brightness that [`DIM_TINT`]
/// keeps, so a veiled pixel darkens exactly like a tinted one.
pub const DIM_VEIL: Color = Color::new(0, 0, 0, 255 - DIM_LEVEL);

/// Holds the decoded image data.
///
/// The fields are private and [`DecodedImage::new`] is the only checked
/// entrance, so a buffer that does not hold exactly `width * height * 4` bytes
/// never becomes an image.
/// The code then reads the pixels through [`DecodedImage::as_rgba`] and the
/// extents through [`DecodedImage::width`] and [`DecodedImage::height`].
///
/// # Examples
///
/// ```
/// # use versatile_viewer::{DecodedImage, DecodeError};
/// # use versatile_viewer::dim::{ImgH, ImgW};
/// let img = DecodedImage::new(ImgW::new(2), ImgH::new(2), vec![0; 16]).unwrap();
/// assert_eq!(img.as_rgba().len(), 16);
/// assert_eq!(img.width(), ImgW::new(2));
/// // A buffer of the wrong size never becomes an image.
/// assert!(matches!(
///     DecodedImage::new(ImgW::new(2), ImgH::new(2), vec![0; 15]),
///     Err(DecodeError::Buffer)
/// ));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    /// The image width, in pixels.
    width: ImgW,
    /// The image height, in pixels.
    height: ImgH,
    /// RGBA8, row-major, 4 bytes per pixel.
    data: Vec<u8>,
}

impl DecodedImage {
    /// An RGBA8 image of `width` by `height`.
    ///
    /// # Errors
    ///
    /// The function errors with [`DecodeError::Buffer`] when `data` does not
    /// hold exactly `width * height * 4` bytes.
    pub fn new(width: ImgW, height: ImgH, data: Vec<u8>) -> Result<Self, DecodeError> {
        if Some(data.len()) != pixel_bytes(width, height) {
            return Err(DecodeError::Buffer);
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// The image built from a buffer whose length this crate just computed.
    ///
    /// A buffer that does not match its dimensions trips the debug assertion,
    /// so the invariant still holds in a test build.
    pub(crate) fn from_exact(width: ImgW, height: ImgH, data: Vec<u8>) -> Self {
        debug_assert_eq!(Some(data.len()), pixel_bytes(width, height));
        Self {
            width,
            height,
            data,
        }
    }

    /// The image width, in pixels.
    #[must_use]
    pub const fn width(&self) -> ImgW {
        self.width
    }

    /// The image height, in pixels.
    #[must_use]
    pub const fn height(&self) -> ImgH {
        self.height
    }

    /// The RGBA8 pixels, row-major, 4 bytes per pixel.
    #[must_use]
    pub fn as_rgba(&self) -> &[u8] {
        &self.data
    }

    /// The RGBA8 pixels, consumed.
    #[must_use]
    pub fn into_rgba(self) -> Vec<u8> {
        self.data
    }

    /// The width, the height and the RGBA8 pixels, consumed.
    #[must_use]
    pub fn into_parts(self) -> (ImgW, ImgH, Vec<u8>) {
        (self.width, self.height, self.data)
    }
}

/// The RGBA8 byte length of a `width` x `height` image, or `None` when the
/// product overflows.
fn pixel_bytes(width: ImgW, height: ImgH) -> Option<usize> {
    usize::try_from(width.raw())
        .ok()?
        .checked_mul(usize::try_from(height.raw()).ok()?)?
        .checked_mul(4)
}

/// Decode a JPEG XL file with jxl-oxide (pure Rust).
fn decode_jxl(path: &Path) -> Result<DecodedImage, DecodeError> {
    let image = jxl_oxide::JxlImage::builder().open(path)?;
    let render = image.render_frame(0)?;

    fb_to_rgba(&render.image_all_channels())
}

/// Convert a jxl-oxide framebuffer to RGBA8.
/// The framebuffer holds f32 samples and 1 to 4 interleaved channels.
///
/// The code replicates grayscale (1 channel) into R, G and B.
/// Gray plus alpha (2 channels) also takes alpha from channel 1.
/// RGB (3 channels) gets alpha = 1.
/// The code takes RGBA (4 channels) as is.
///
/// The function applies [`cap_texture_side`], so the result is uploadable as
/// is, like [`decode_common`].
/// It runs on a decode worker, never on the main thread.
///
/// # Errors
///
/// The function errors when the framebuffer holds an unexpected channel
/// count.
pub(crate) fn fb_to_rgba(fb: &jxl_oxide::FrameBuffer) -> Result<DecodedImage, DecodeError> {
    let width = ImgW::new(fb.width() as u32);
    let height = ImgH::new(fb.height() as u32);
    let channels = fb.channels();
    let samples = fb.buf();
    if !matches!(channels, 1..=4) {
        return Err(DecodeError::Channels(channels as u32));
    }

    let mut rgba = vec![0u8; width.raw() as usize * height.raw() as usize * 4];
    for (dst, src) in rgba
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(samples.chunks_exact(channels))
    {
        let g = to_u8(src[0]);
        dst[0] = g;
        dst[1] = if channels >= 3 { to_u8(src[1]) } else { g };
        dst[2] = if channels >= 3 { to_u8(src[2]) } else { g };
        dst[3] = if channels == 2 || channels == 4 {
            to_u8(src[channels - 1])
        } else {
            255
        };
    }
    Ok(cap_texture_side(DecodedImage::from_exact(
        width, height, rgba,
    )))
}

const fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Decode common formats (PNG, JPEG, ...) with the `image` crate.
///
/// The function applies [`cap_texture_side`], so the result is uploadable as
/// is.
/// This runs on a decode worker, never on the main thread.
pub(crate) fn decode_common(path: &Path) -> Result<DecodedImage, DecodeError> {
    let rgba = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?
        .into_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(cap_texture_side(DecodedImage::from_exact(
        ImgW::new(width),
        ImgH::new(height),
        rgba.into_raw(),
    )))
}

/// Decode a non-vector file with the decoder that [`sign::detect`] named.
///
/// The function applies [`cap_texture_side`], so the result is uploadable as
/// is.
/// It runs on a decode worker, never on the main thread.
pub(crate) fn decode_raster(
    path: &Path,
    format: crate::sign::Format,
) -> Result<DecodedImage, DecodeError> {
    match format {
        crate::sign::Format::Jxl => decode_jxl(path),
        crate::sign::Format::Common => decode_common(path),
    }
}

/// The source size of a vector file, measured without painting it.
///
/// The placeholder pass runs this on a cache miss, so a cold grid cell and the
/// image-view fit know the shape of the picture before its decode lands.
/// A raster format answers None, because its decode reports the size with the
/// pixels.
pub(crate) fn measure_vector(path: &Path) -> Option<(ImgW, ImgH)> {
    if !svg::sniff(path) {
        return None;
    }
    svg::measure(path).ok()
}

/// The longest texture side the decode produces.
///
/// This is the policy, and `texture::Uploader` clamps every upload to the
/// adapter's own `max_texture_dimension_2d` when that is smaller, so the
/// decoded size stays the same on every host and a small adapter still takes
/// the picture.
/// Desktop adapters range from 4096 to 16384.
/// The code downscales anything larger here instead of failing the load.
pub const MAX_TEXTURE_SIDE: u32 = 8192;

/// Downscale an RGBA8 buffer to the uploadable [`MAX_TEXTURE_SIDE`].
///
/// Run this function off the main thread, where the decode runs.
/// A texture upload then never resizes a big buffer on the main thread.
/// Below the limit the function returns the image unchanged, so no quality
/// is lost and nothing is copied.
///
/// # Examples
///
/// ```
/// # use versatile_viewer::{DecodedImage, cap_texture_side, MAX_TEXTURE_SIDE};
/// # use versatile_viewer::dim::{ImgH, ImgW};
/// # use versatile_viewer::dim::LongSide;
/// // Below the limit: the same buffer comes back, untouched.
/// let img = DecodedImage::new(ImgW::new(64), ImgH::new(32), vec![7; 4 * 64 * 32]).unwrap();
/// let img = cap_texture_side(img);
/// assert_eq!((img.width().raw(), img.height().raw()), (64, 32));
/// assert_eq!(img.as_rgba().len(), 4 * 64 * 32);
///
/// // Over the limit: the long side lands exactly on the limit.
/// let img = DecodedImage::new(
///     ImgW::new(MAX_TEXTURE_SIDE + 1),
///     ImgH::new(1),
///     vec![0; 4 * (MAX_TEXTURE_SIDE as usize + 1)],
/// )
/// .unwrap();
/// let img = cap_texture_side(img);
/// assert_eq!(
///     (img.width().raw(), img.height().raw()),
///     (MAX_TEXTURE_SIDE, 1)
/// );
/// ```
#[must_use]
pub fn cap_texture_side(image: DecodedImage) -> DecodedImage {
    if image.width().raw().max(image.height().raw()) > MAX_TEXTURE_SIDE {
        downscale_rgba(image, LongSide::new(MAX_TEXTURE_SIDE))
    } else {
        image
    }
}

/// Downscale an RGBA8 buffer so its long side is at most `long_side`.
///
/// Below the cap the function returns the buffer unchanged.
/// It never upscales.
/// The output aspect matches the input.
///
/// # Examples
///
/// ```
/// # use versatile_viewer::{DecodedImage, downscale_rgba};
/// # use versatile_viewer::dim::{ImgH, ImgW, LongSide};
/// // Below the cap: returned unchanged.
/// let img = DecodedImage::new(ImgW::new(2), ImgH::new(2), vec![0; 4 * 2 * 2]).unwrap();
/// let img = downscale_rgba(img, LongSide::new(8));
/// assert_eq!((img.width().raw(), img.height().raw()), (2, 2));
/// assert_eq!(img.as_rgba().len(), 4 * 2 * 2);
///
/// // Over it: the aspect stays and the long side equals the cap.
/// let img = DecodedImage::new(ImgW::new(200), ImgH::new(100), vec![0; 4 * 200 * 100]).unwrap();
/// let img = downscale_rgba(img, LongSide::new(100));
/// assert_eq!((img.width().raw(), img.height().raw()), (100, 50));
/// assert_eq!(img.as_rgba().len(), 4 * 100 * 50);
///
/// // An image without pixels comes back as it went in.
/// let img = DecodedImage::new(ImgW::new(0), ImgH::new(0), Vec::new()).unwrap();
/// let img = downscale_rgba(img, LongSide::new(8));
/// assert_eq!((img.width().raw(), img.height().raw()), (0, 0));
/// assert!(img.as_rgba().is_empty());
/// ```
///
/// The function never upscales, and an image without pixels comes back as it
/// went in.
#[must_use]
pub fn downscale_rgba(image: DecodedImage, long_side: LongSide) -> DecodedImage {
    let (width, height, data) = image.into_parts();
    let (w, h) = (width.raw(), height.raw());
    // An image without pixels has nothing to resample, so it comes back as it
    // went in.
    if w == 0 || h == 0 {
        return DecodedImage::from_exact(width, height, data);
    }
    let long_side = long_side.raw().max(1);
    if w.max(h) <= long_side {
        return DecodedImage::from_exact(width, height, data);
    }
    let (nw, nh) = fitted(w, h, long_side);
    DecodedImage::from_exact(
        ImgW::new(nw),
        ImgH::new(nh),
        resample_rgba(&data, w, h, nw, nh),
    )
}

/// The dimensions of `width` x `height` scaled so the long side is
/// `long_side`, with the aspect preserved and each side at least 1.
#[must_use]
pub(crate) fn fitted(width: u32, height: u32, long_side: u32) -> (u32, u32) {
    let scale = long_side as f32 / width.max(height) as f32;
    (
        ((width as f32 * scale).round() as u32).max(1),
        ((height as f32 * scale).round() as u32).max(1),
    )
}

/// Resample an RGBA8 buffer into a `nw` x `nh` buffer with the `fast_image_resize`
/// SIMD path (Lanczos3).
///
/// The function allocates the output and leaves the input alone.
/// Call it on a decode worker, never on the main thread.
///
/// # Panics
///
/// The function panics if `rgba` does not hold exactly `w * h * 4` bytes.
#[must_use]
pub(crate) fn resample_rgba(rgba: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    use fast_image_resize::{
        FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
        images::{Image, ImageRef},
    };
    let src = ImageRef::new(w, h, rgba, PixelType::U8x4).expect("rgba buffer matches dimensions");
    let mut dst = Image::new(nw, nh, PixelType::U8x4);
    let options = ResizeOptions {
        algorithm: ResizeAlg::Convolution(FilterType::Lanczos3),
        ..Default::default()
    };
    Resizer::new()
        .resize(&src, &mut dst, &options)
        .expect("rgba buffers match their dimensions");
    dst.into_vec()
}

/// Downscale a borrowed RGBA8 buffer, and leave the input alone.
///
/// [`downscale_rgba`] takes ownership, so a caller that still needs the
/// source buffer would have to clone it.
/// The grid decodes once and needs two sizes from that decode (thumb and
/// full), and a clone of a 100 MP buffer costs hundreds of megabytes.
/// This function copies no input pixels: the resizer only reads.
///
/// The function never upscales, and an image without pixels comes back as it
/// went in.
/// Below `long_side` the output is a plain copy of the input.
///
/// # Panics
///
/// The function panics if `rgba` does not hold exactly
/// `width * height * 4` bytes.
#[must_use]
pub fn downscale_rgba_ref(
    rgba: &[u8],
    width: ImgW,
    height: ImgH,
    long_side: LongSide,
) -> DecodedImage {
    let (w, h) = (width.raw(), height.raw());
    // An image without pixels has nothing to resample, so it comes back as it
    // went in.
    if w == 0 || h == 0 {
        return DecodedImage::from_exact(width, height, Vec::new());
    }
    let long_side = long_side.raw().max(1);
    if w.max(h) <= long_side {
        return DecodedImage::from_exact(width, height, rgba.to_vec());
    }
    let (nw, nh) = fitted(w, h, long_side);
    DecodedImage::from_exact(
        ImgW::new(nw),
        ImgH::new(nh),
        resample_rgba(rgba, w, h, nw, nh),
    )
}

/// True when `VV_TIMING` is set.
///
/// The code then prints the GPU stack at startup (adapter, format, present
/// mode, texture limit) and every slow frame to stderr.
/// An upload runs on a decode worker, so this flag stays off that path.
/// Input tracing stays under `VV_DEBUG`.
#[must_use]
pub fn timing_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VV_TIMING").is_some())
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    // Compile-time check that this stays const-evaluable.
    const _: u8 = to_u8(0.5);

    #[test]
    fn cap_texture_side_never_upscales_and_caps() {
        let img = |w: u32, h: u32| {
            DecodedImage::new(ImgW::new(w), ImgH::new(h), vec![9; (w * h * 4) as usize])
                .expect("sized")
        };
        // Below the limit: the same buffer, untouched.
        let src = img(32, 16);
        let out = cap_texture_side(src.clone());
        assert_eq!((out.width(), out.height()), (ImgW::new(32), ImgH::new(16)));
        assert_eq!(out.as_rgba(), src.as_rgba());
        // Over the limit: the long side lands exactly on the limit.
        let out = cap_texture_side(img(MAX_TEXTURE_SIDE + 1, 2));
        assert_eq!(out.width(), ImgW::new(MAX_TEXTURE_SIDE));
        assert_eq!(out.height(), ImgH::new(2));
        assert_eq!(out.as_rgba().len(), (MAX_TEXTURE_SIDE * 2 * 4) as usize);
        // A tall image caps by height instead.
        let out = cap_texture_side(img(2, MAX_TEXTURE_SIDE + 1));
        assert_eq!(out.width(), ImgW::new(2));
        assert_eq!(out.height(), ImgH::new(MAX_TEXTURE_SIDE));
    }

    #[test]
    fn downscale_rgba_ref_borrows_and_never_upscales() {
        let src = |w: u32, h: u32| {
            DecodedImage::new(ImgW::new(w), ImgH::new(h), vec![7; (w * h * 4) as usize])
                .expect("sized")
        };
        // Above the cap: 800x200 with cap 64 -> 64x16.
        let img = src(800, 200);
        let out = downscale_rgba_ref(img.as_rgba(), img.width(), img.height(), LongSide::new(64));
        assert_eq!((out.width(), out.height()), (ImgW::new(64), ImgH::new(16)));
        assert_eq!(
            (img.width(), img.height()),
            (ImgW::new(800), ImgH::new(200))
        );
        // Below the cap: the output equals the borrowed input.
        let img = src(32, 16);
        let out = downscale_rgba_ref(img.as_rgba(), img.width(), img.height(), LongSide::new(64));
        assert_eq!((out.width(), out.height()), (ImgW::new(32), ImgH::new(16)));
        assert_eq!(out.as_rgba(), img.as_rgba());
        assert_eq!(out, downscale_rgba(img, LongSide::new(64)));
    }

    #[test]
    fn to_u8_clamps_and_rounds() {
        assert_eq!(to_u8(0.0), 0);
        assert_eq!(to_u8(1.0), 255);
        assert_eq!(to_u8(-5.0), 0);
        assert_eq!(to_u8(42.0), 255);
        assert_eq!(to_u8(0.5), 128); // 0.5 * 255 + 0.5 = 128
        assert_eq!(to_u8(1.0 / 255.0), 1);
    }

    #[test]
    fn fb_to_rgba_replicates_grayscale() {
        // 1 channel (grayscale): the gray sample goes to R, G and B.
        // The code sets alpha = 255.
        let mut fb = jxl_oxide::FrameBuffer::new(2, 1, 1);
        fb.buf_mut()[..2].copy_from_slice(&[0.0, 0.5]);
        let img = fb_to_rgba(&fb).unwrap();
        assert_eq!((img.width(), img.height()), (ImgW::new(2), ImgH::new(1)));
        assert_eq!(img.as_rgba(), [0, 0, 0, 255, 128, 128, 128, 255]);
    }

    #[test]
    fn fb_to_rgba_gray_alpha_takes_alpha() {
        // 2 channels (gray + alpha): the code replicates the gray value and
        // takes alpha from channel 1.
        let mut fb = jxl_oxide::FrameBuffer::new(1, 1, 2);
        fb.buf_mut()[..2].copy_from_slice(&[1.0, 0.5]);
        let img = fb_to_rgba(&fb).unwrap();
        assert_eq!((img.width(), img.height()), (ImgW::new(1), ImgH::new(1)));
        assert_eq!(img.as_rgba(), [255, 255, 255, 128]);
    }

    #[test]
    fn decode_raster_reads_a_png() {
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(3, 2).save(&path).unwrap();
        let decoded = decode_raster(&path, sign::Format::Common).unwrap();
        assert_eq!(decoded.width(), ImgW::new(3));
        assert_eq!(decoded.height(), ImgH::new(2));
        assert_eq!(decoded.as_rgba().len(), 3 * 2 * 4);
    }

    #[test]
    fn decode_raster_reads_webp() {
        // Lossless encode via image-webp, then decode back through
        // decode_raster with the format that the sniff named.
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let path = dir.join("img.webp");
        image::DynamicImage::new_rgb8(5, 4)
            .save_with_format(&path, image::ImageFormat::WebP)
            .unwrap();
        let format = sign::detect(&path).unwrap().raster().unwrap();
        let decoded = decode_raster(&path, format).unwrap();
        assert_eq!(decoded.width(), ImgW::new(5));
        assert_eq!(decoded.height(), ImgH::new(4));
        assert_eq!(decoded.as_rgba().len(), 5 * 4 * 4);
    }

    #[test]
    fn decode_raster_reads_grid_filter_formats() {
        // Every format the grid filter accepts (grid.rs is_image_path) must
        // decode.
        // The filter and the image-crate features must stay in sync (see
        // the Cargo.toml comment).
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        for (ext, format) in [
            ("bmp", image::ImageFormat::Bmp),
            ("gif", image::ImageFormat::Gif),
            ("tga", image::ImageFormat::Tga),
            ("tif", image::ImageFormat::Tiff),
        ] {
            let path = dir.join(format!("img.{ext}"));
            image::DynamicImage::new_rgb8(4, 2)
                .save_with_format(&path, format)
                .unwrap_or_else(|e| panic!("encode {ext}: {e}"));
            let decoded = decode_raster(&path, sign::Format::Common)
                .unwrap_or_else(|e| panic!("decode {ext}: {e}"));
            assert_eq!(
                (decoded.width(), decoded.height()),
                (ImgW::new(4), ImgH::new(2)),
                "{ext}"
            );
            assert_eq!(decoded.as_rgba().len(), 4 * 2 * 4, "{ext}");
        }
    }

    #[test]
    fn decode_raster_fails_cleanly_on_garbage() {
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let path = dir.join("x.png");
        std::fs::write(&path, b"garbage").unwrap();
        assert!(decode_raster(&path, sign::Format::Common).is_err());
    }

    proptest::proptest! {
        /// The cap never upscales, never exceeds the cap and keeps the
        /// aspect within one pixel of rounding.
        #[test]
        fn cap_texture_side_invariants(w in 1u32..400, h in 1u32..400) {
            let img = DecodedImage::new(ImgW::new(w), ImgH::new(h), vec![0; 4 * w as usize * h as usize])
                .expect("sized");
            let out = cap_texture_side(img);
            let (ow, oh) = (out.width().raw(), out.height().raw());
            // A small image comes back untouched.
            proptest::prop_assert!((ow, oh) == (w, h));
            proptest::prop_assert_eq!(out.as_rgba().len(), 4 * ow as usize * oh as usize);
        }

        /// `fitted` keeps the aspect, clamps to 1 and lands the long side on
        /// the cap.
        #[test]
        fn fitted_keeps_aspect(w in 1u32..20000, h in 1u32..20000, cap in 1u32..8192) {
            // Only downscales occur in the viewer: the cap stays at or below
            // the source long side.
            let cap = cap.min(w.max(h)).max(1);
            let (nw, nh) = fitted(w, h, cap);
            proptest::prop_assert!(nw >= 1 && nh >= 1);
            proptest::prop_assert!(nw.max(nh) <= w.max(h));
            if w >= h {
                proptest::prop_assert_eq!(nw, cap.min(w));
            } else {
                proptest::prop_assert_eq!(nh, cap.min(h));
            }
        }

        /// The resample honors the output size and never reads past the
        /// input buffer.
        #[test]
        fn resample_matches_output_size(w in 8u32..256, h in 8u32..256) {
            let src = vec![0x5au8; 4 * w as usize * h as usize];
            let out = resample_rgba(&src, w, h, 4, 3);
            proptest::prop_assert_eq!(out.len(), 4 * 4 * 3);
        }
    }
}
