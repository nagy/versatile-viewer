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
    clippy::similar_names
)]

//! The showable sources of versatile-viewer.
//!
//! Every source is a [`surface::Surface`], so the grid, the loader and the
//! image view ask for a plane at a scale and never for a file format.
//! [`open_surface`](surface::open_surface) sniffs the content with [`sign`]
//! and picks the backend: a raster file, an SVG document, a PDF, or a path a
//! converter rule claims.
//! [`open_book_surface`](surface::open_book_surface) wraps a plane set of two
//! or more into [`book::BookSurface`], so a document shows as a book.
//!
//! No layer of this crate touches a GPU handle or a window, so the whole
//! surface API runs headless.
//! The decoder crates stay behind this seam: jxl-oxide, the `image` crate,
//! resvg for SVG, hayro for PDF, and the cache encoder jixel.

pub mod book;
pub mod convert;
mod decode;
pub mod pdf;
pub mod sign;
pub mod surface;
pub mod svg;
pub mod thumbcache;

// The layers below, named through this crate as well, so one `crate::dim` path
// serves a module of this crate and a caller of it.
pub use vv_core::{dim, draw, region};
pub use vv_pixels::{
    DecodeError, DecodedImage, MAX_TEXTURE_SIDE, cap_texture_side, downscale_rgba,
    downscale_rgba_ref, error, fitted, placeholder, resample_rgba,
};

pub use crate::decode::{decode_common, decode_jxl, decode_raster, fb_to_rgba, measure_vector};
