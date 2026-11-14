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

//! The GPU layer of versatile-viewer.
//!
//! [`gpu`] owns the instance, the adapter, the device, the queue and the
//! swapchain, [`render`] owns the one instanced-quad pipeline and its batch,
//! and [`texture`] owns one uploaded picture and its two filter groups.
//! Nothing here reads a window event and nothing here decodes a file: a host
//! hands this layer a texture view, and a decode worker hands it pixels
//! through [`texture::Uploader`].
//!
//! The backend is Vulkan only, and every texture is `Rgba8Unorm` beside a
//! non-sRGB swapchain, so the bytes a decoder writes reach the screen
//! unchanged.

pub mod gpu;
pub mod render;
pub mod texture;

// The layers below, named through this crate as well, so one `crate::dim` path
// serves a module of this crate and a caller of it.
pub use vv_core::{dim, draw};
pub use vv_pixels::{DecodedImage, MAX_TEXTURE_SIDE, downscale_rgba, error};
