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

//! The two views of versatile-viewer, and the viewer that drives them.
//!
//! [`grid`] lists a directory as cells of thumbnails, the private `view` holds the image
//! view: what it shows, how it frames it, the patch that refines a settled
//! zoom, and the region selection.
//! [`viewer`] is the one value a host builds on its own device: it holds the
//! pipeline, the views, the loaders and the clock, so a host owns the window
//! and the viewer owns the picture.
//! [`loader`] streams one plane of a source off the main thread, [`input`]
//! holds the held keys, the frame edges and the pointer of one frame, and
//! [`pane`] names the X11 window a host hands over with `--wid`.
//!
//! The module `view` stays private to this crate: [`viewer`] drives it, and no
//! caller names a `ViewState` of its own.

pub mod blurbg;
/// The test-only fixtures that more than one module of this crate reads.
#[cfg(test)]
pub(crate) mod fixtures;
pub mod grid;
pub mod input;
pub mod loader;
pub mod pane;
pub(crate) mod view;
pub mod viewer;
pub mod zoomkeys;

// The layers below, named through this crate as well, so one `crate::dim` path
// serves a module of this crate and a caller of it.
pub use vv_core::{
    DIM_LEVEL, DIM_TINT, DIM_VEIL, SHIFT_SPEED_MULTIPLIER, TAP_SLACK, dim, draw, keyrepeat, region,
    region_sel, rotate, timing_enabled, tween,
};
pub use vv_gpu::{render, texture};
pub use vv_pixels::{
    DecodedImage, cap_texture_side, downscale_rgba, downscale_rgba_ref, error, placeholder,
    resample_rgba,
};
pub use vv_source::{
    book, convert, fb_to_rgba, measure_vector, pdf, sign, surface, svg, thumbcache,
};
