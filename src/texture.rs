//! GPU textures: the pixels of a picture, ready to sample.
//!
//! The module replaces `raylib::Texture2D` and its `LoadTextureFromImage` and
//! `SetTextureFilter` calls.
//!
//! # Two bind groups, one texture
//!
//! raylib kept the filter on the texture and changed it at runtime with `a`.
//! wgpu makes the sampler a bind group member instead, so a texture carries one
//! group per filter and a draw picks the group it wants.
//! The two samplers are shared, so a group costs one texture view and nothing
//! else.
//!
//! # Uploads off the main thread
//!
//! `wgpu::Device` and `wgpu::Queue` are `Send + Sync`, so a decode worker
//! uploads its own pixels through an [`Uploader`] and the main thread only
//! draws what arrives.
//! [`Uploader`] holds cheap handles, so a job clones it for one file.
//!
//! # Format
//!
//! The texture format is a non-sRGB `Rgba8Unorm`, and the swapchain is a
//! non-sRGB format too, so the decoder's bytes reach the screen unchanged: that
//! is the pixel math the raylib path had.
//! Alpha stays straight, because the decoders write straight bytes and
//! [`crate::svg`] un-premultiplies before it hands a picture over.

use std::sync::Arc;

use derive_more::{Display, Error};
use wgpu::util::DeviceExt as _;

use crate::{DecodedImage, MAX_TEXTURE_SIDE, dim::LongSide, downscale_rgba};

/// The sampler filter a draw uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Filter {
    /// Smooth resampling, for a picture or a thumbnail drawn at any scale.
    #[default]
    Linear,
    /// One texel per pixel, for the `a` key at a 1:1 zoom.
    Nearest,
}

impl Filter {
    /// The filter the `a` key switches to.
    #[must_use]
    pub fn toggled(self) -> Self {
        match self {
            Self::Linear => Self::Nearest,
            Self::Nearest => Self::Linear,
        }
    }

    /// The wgpu filter mode of this choice.
    #[must_use]
    fn mode(self) -> wgpu::FilterMode {
        match self {
            Self::Linear => wgpu::FilterMode::Linear,
            Self::Nearest => wgpu::FilterMode::Nearest,
        }
    }
}

/// Why a texture could not be made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, Error)]
pub enum TextureError {
    /// The buffer does not hold `width * height * 4` bytes.
    #[display("the pixel buffer does not match its dimensions")]
    Buffer,
}

/// The sampler pair, the layout and the device that every texture shares.
struct Shared {
    layout: wgpu::BindGroupLayout,
    linear: wgpu::Sampler,
    nearest: wgpu::Sampler,
    /// The smallest of the decode policy and the adapter limit.
    max_side: u32,
}

/// The worker-side handle that turns pixels into a [`Texture`].
///
/// The clone is cheap: the device and the queue are shared handles, and the
/// sampler pair sits behind one [`Arc`].
#[derive(Clone)]
pub struct Uploader {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shared: Arc<Shared>,
}

#[allow(clippy::missing_fields_in_debug)] // a wgpu handle prints as an opaque id
impl std::fmt::Debug for Uploader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The device handle prints as an opaque id, so the limit is the fact a
        // log line wants.
        f.debug_struct("Uploader")
            .field("max_side", &self.shared.max_side)
            .finish()
    }
}

impl Uploader {
    /// Build the shared samplers, the bind group layout and the uploader.
    ///
    /// `adapter_side` is the adapter's largest 2D texture, so an upload clamps
    /// to what the device takes.
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter_side: u32,
        layout: wgpu::BindGroupLayout,
    ) -> Self {
        let sampler = |filter: Filter, label: &str| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                mag_filter: filter.mode(),
                min_filter: filter.mode(),
                // One mip level, so the mip filter never runs. A later mip
                // chain would remove the moire of a far-drawn thumbnail.
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            })
        };
        Self {
            device: device.clone(),
            queue: queue.clone(),
            shared: Arc::new(Shared {
                layout,
                linear: sampler(Filter::Linear, "vv linear sampler"),
                nearest: sampler(Filter::Nearest, "vv nearest sampler"),
                max_side: adapter_side.clamp(1, MAX_TEXTURE_SIDE),
            }),
        }
    }

    /// Upload a decoded picture.
    ///
    /// A buffer wider than the device takes is downscaled first, on the worker
    /// that called this, so the main thread never resizes pixels.
    ///
    /// # Errors
    ///
    /// Returns [`TextureError`] when the buffer does not match its size.
    pub fn image(&self, image: &DecodedImage) -> Result<Texture, TextureError> {
        self.rgba(
            image.as_rgba(),
            image.width().raw(),
            image.height().raw(),
            "vv picture",
        )
    }

    /// Upload raw straight-alpha RGBA8 pixels.
    ///
    /// # Errors
    ///
    /// Returns [`TextureError`] when the buffer does not hold exactly
    /// `width * height * 4` bytes.
    pub fn rgba(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        label: &str,
    ) -> Result<Texture, TextureError> {
        // The decoded type already checked this pair, so the test is a guard
        // for a hand-built buffer, not a hot path.
        if rgba.len() != width as usize * height as usize * 4 {
            return Err(TextureError::Buffer);
        }
        let side = width.max(height);
        let owned;
        let (rgba, width, height) = if side > self.shared.max_side {
            owned = downscale_rgba(
                DecodedImage::from_exact(
                    crate::dim::ImgW::new(width),
                    crate::dim::ImgH::new(height),
                    rgba.to_vec(),
                ),
                LongSide::new(self.shared.max_side),
            );
            (owned.as_rgba(), owned.width().raw(), owned.height().raw())
        } else {
            (rgba, width, height)
        };
        Ok(self.build(rgba, width, height, label))
    }

    /// The one-by-one white texture behind a solid fill.
    pub(crate) fn white(&self) -> Texture {
        self.build(&[0xff, 0xff, 0xff, 0xff], 1, 1, "vv white")
    }

    /// Build the texture and both filter groups.
    fn build(&self, rgba: &[u8], width: u32, height: u32, label: &str) -> Texture {
        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            rgba,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let group = |sampler: &wgpu::Sampler| {
            Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vv texture"),
                layout: &self.shared.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                ],
            }))
        };
        // The two bind groups hold the view, and the view holds the texture,
        // so the allocation lives as long as this value does and no handle has
        // to be kept beside them.
        Texture {
            dims: (width, height),
            linear: group(&self.shared.linear),
            nearest: group(&self.shared.nearest),
        }
    }
}

/// One uploaded picture and its two filter groups.
pub struct Texture {
    dims: (u32, u32),
    /// The linear group: the default of a picture and of a thumbnail.
    linear: Arc<wgpu::BindGroup>,
    /// The nearest group: what `a` selects for the whole image and the region
    /// patches, and what a solid fill uses.
    nearest: Arc<wgpu::BindGroup>,
}

#[allow(clippy::missing_fields_in_debug)] // the bind groups print as opaque handles
impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The wgpu handles print as opaque ids, so the size is the one fact a
        // reader wants from a texture in a log line.
        f.debug_struct("Texture").field("dims", &self.dims).finish()
    }
}

impl Texture {
    /// The pixel size of the texture.
    #[must_use]
    pub fn dims(&self) -> (u32, u32) {
        self.dims
    }

    /// The width, in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.dims.0
    }

    /// The height, in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.dims.1
    }

    /// The bind group for one filter.
    #[must_use]
    pub(crate) fn group(&self, filter: Filter) -> &Arc<wgpu::BindGroup> {
        match filter {
            Filter::Linear => &self.linear,
            Filter::Nearest => &self.nearest,
        }
    }

}

/// One rendered picture on its way to the window: pixels, or the texture they
/// became.
///
/// A decode worker uploads its own frame when it holds an [`Uploader`], so
/// production carries the texture and the pixels are freed on the worker.
/// A test builds the same pipeline with no device, so the frame stays pixels
/// and its dimensions can be asserted without a GPU.
#[derive(Debug)]
pub enum Ready {
    /// Decoded pixels, for a caller with no device.
    Pixels(DecodedImage),
    /// The uploaded texture.
    Texture(Texture),
}

impl Ready {
    /// The pixel size of the frame, whichever form it holds.
    #[must_use]
    pub fn dims(&self) -> (u32, u32) {
        match self {
            Self::Pixels(image) => (image.width().raw(), image.height().raw()),
            Self::Texture(texture) => texture.dims(),
        }
    }

    /// The pixels, when this frame stayed on the CPU.
    #[must_use]
    pub fn pixels(&self) -> Option<&DecodedImage> {
        match self {
            Self::Pixels(image) => Some(image),
            Self::Texture(_) => None,
        }
    }

    /// The texture, when this frame was uploaded.
    #[must_use]
    pub fn texture(&self) -> Option<&Texture> {
        match self {
            Self::Pixels(_) => None,
            Self::Texture(texture) => Some(texture),
        }
    }

    /// Take the texture, when this frame was uploaded.
    ///
    /// A caller with no use for the pixels (the grid cell) moves the handle
    /// and drops nothing else.
    #[must_use]
    pub fn into_texture(self) -> Option<Texture> {
        match self {
            Self::Texture(texture) => Some(texture),
            Self::Pixels(_) => None,
        }
    }

    /// Upload the frame when it is still pixels, and hand back its texture.
    ///
    /// # Errors
    ///
    /// Returns [`TextureError`] when the pixel buffer does not match its size.
    pub fn upload(self, uploader: &Uploader) -> Result<Texture, TextureError> {
        match self {
            Self::Texture(texture) => Ok(texture),
            Self::Pixels(image) => uploader.image(&image),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `a` key returns to the filter it left.
    #[test]
    fn a_toggle_returns_to_its_own_filter() {
        assert_eq!(Filter::Linear.toggled(), Filter::Nearest);
        assert_eq!(Filter::Linear.toggled().toggled(), Filter::Linear);
        assert_eq!(Filter::default(), Filter::Linear);
    }
}
