//! The display-list pass: `vello` paints a [`Primitive`] list into a
//! window-sized target, and the quad pipeline composites the result.
//!
//! # Why a second renderer
//!
//! The viewer draws everything through one instanced-quad pipeline, and a
//! primitive is not a quad: a path needs analytic anti-aliasing, a gradient
//! needs a ramp, a clip needs a stack and a glyph needs an outline. `vello`
//! answers all four on the device and queue this crate already holds, so no
//! second device and no second `wgpu` exists in the process.
//!
//! # The target
//!
//! `vello` renders with compute work into an `Rgba8Unorm` texture that carries
//! `STORAGE_BINDING`, and [`SceneView::draw`] answers the same texture in the
//! form a quad draw samples.
//! The target is the size of the window and the transform maps surface units
//! onto window pixels, so a zoom or a pan re-renders *cheap* primitives rather
//! than re-rasterizing a patch: the list itself never changes.
//!
//! # Image leaves
//!
//! A [`Primitive::Image`] carries pixels and not a texture, so this pass uploads
//! each leaf once and keeps it in a table keyed by the leaf's address. The list
//! lives in an `Arc<[Primitive]>` that the view holds for as long as the plane
//! does, so an address stays stable, and a different list clears the table.

use std::{collections::HashMap, sync::Arc};

use derive_more::{Display, Error, From};
use vello::{
    AaConfig, AaSupport, RenderParams, Renderer as VelloRenderer, RendererOptions, Scene as Vello,
    kurbo::{self, Affine},
    peniko::{
        self as peniko, Blob, Color as VelloColor, ColorStop, Extend, Fill, Gradient, GradientKind,
        ImageAlphaType, ImageBrush, ImageData, ImageFormat, LinearGradientPosition,
        RadialGradientPosition,
    },
};

use crate::{
    DecodedImage,
    draw::Color,
    region,
    scene::{FillRule, Paint, Primitive, Spread, Stop},
    texture::{Texture, Uploader},
};

/// The anti-aliasing a scene renders with.
///
/// Area sampling is the one mode that is always available, and the fine
/// variants cost pipelines at startup that this viewer would not switch to.
const AA: AaConfig = AaConfig::Area;

/// Why the display-list pass did not come up.
#[derive(Debug, Display, Error, From)]
pub enum SceneError {
    /// `vello` refused the device, which happens when a required feature or
    /// limit is missing.
    #[display("vello refused the device: {_0}")]
    Renderer(#[from] vello::Error),
}

/// One display-list pass: the `vello` renderer, its window-sized target and the
/// pictures of the list it draws.
pub struct SceneView {
    renderer: VelloRenderer,
    /// The primitives of the frame being drawn, reused so a frame allocates no
    /// scene.
    list: Vello,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    /// The composite side of `view`, which a quad draw samples.
    texture: Texture,
    /// The size of `target`, in window pixels.
    size: (u32, u32),
    uploader: Uploader,
    /// The picture of each [`Primitive::Image`] leaf, keyed by its address.
    images: HashMap<usize, ImageBrush>,
    /// The list `images` belongs to, as its first primitive's address.
    owner: Option<usize>,
}

impl SceneView {
    /// Build the pass on `device`.
    ///
    /// `vello` compiles its compute pipelines here, which costs a moment, so a
    /// caller builds this the first time a source answers a list and never for
    /// a picture.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError`] when vello refuses the device.
    pub fn new(device: &wgpu::Device, uploader: Uploader) -> Result<Self, SceneError> {
        let renderer = VelloRenderer::new(
            device,
            RendererOptions {
                use_cpu: false,
                antialiasing_support: AaSupport::area_only(),
                num_init_threads: None,
                pipeline_cache: None,
            },
        )?;
        let (target, view, texture) = target_of(device, &uploader, (1, 1));
        Ok(Self {
            renderer,
            list: Vello::new(),
            target,
            view,
            texture,
            size: (1, 1),
            uploader,
            images: HashMap::new(),
            owner: None,
        })
    }

    /// Draw `list` in window pixels and answer the texture to composite.
    ///
    /// `frame` maps surface units onto the window, so the same list draws
    /// sharply at every zoom and under any turn.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError`] when vello refuses the frame, which a missing
    /// device feature causes.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        list: &Arc<[Primitive]>,
        frame: &region::Frame,
        size: (u32, u32),
    ) -> Result<&Texture, SceneError> {
        self.resize(device, size);
        let owner = list.as_ptr() as usize;
        if self.owner != Some(owner) {
            // Another list is another set of leaves, so the pictures of the
            // last one cannot be trusted and the addresses they were keyed on
            // may now mean something else.
            self.images.clear();
            self.owner = Some(owner);
        }
        let to_window = to_window(frame);
        self.list.reset();
        for primitive in list.iter() {
            match primitive {
                Primitive::Rect { rect, paint } => {
                    // A rect carries no transform, so both its shape and its
                    // paint live in surface units.
                    self.paint(Fill::NonZero, to_window, paint, None, rect);
                }
                Primitive::Path {
                    path,
                    transform,
                    fill,
                    fill_rule,
                    stroke,
                } => {
                    let placed = to_window * *transform;
                    // A ramp is written in surface units and the shape in its
                    // own, so the brush takes the inverse the shape applied.
                    let paint_at = Some(transform.inverse());
                    if let Some(paint) = fill {
                        self.paint(vello_fill_rule(*fill_rule), placed, paint, paint_at, path);
                    }
                    if let Some(outline) = stroke {
                        self.outline(outline, placed, paint_at, path);
                    }
                }
                Primitive::PushClip {
                    path,
                    transform,
                    fill_rule,
                } => {
                    self.list.push_clip_layer(
                        vello_fill_rule(*fill_rule),
                        to_window * *transform,
                        path,
                    );
                }
                Primitive::PopClip | Primitive::PopLayer => self.list.pop_layer(),
                Primitive::PushLayer { alpha } => {
                    // vello opens a layer only around a clip shape, so a plain
                    // opacity layer clips to the whole target: every draw of
                    // this pass is written in window pixels.
                    let whole =
                        kurbo::Rect::new(0.0, 0.0, f64::from(self.size.0), f64::from(self.size.1));
                    self.list.push_layer(
                        Fill::NonZero,
                        peniko::Mix::Normal,
                        *alpha,
                        Affine::IDENTITY,
                        &whole,
                    );
                }
                Primitive::Image { pixels, transform } => {
                    let key = pixels.as_rgba().as_ptr() as usize;
                    let brush = self
                        .images
                        .entry(key)
                        .or_insert_with(|| image_brush(pixels));
                    self.list.draw_image(brush.as_ref(), to_window * *transform);
                }
            }
        }
        self.renderer.render_to_texture(
            device,
            queue,
            &self.list,
            &self.view,
            &RenderParams {
                // Transparent, so the letterbox bars and the window background
                // stay the window's own color and the scene only paints where a
                // primitive is.
                base_color: VelloColor::TRANSPARENT,
                width: self.size.0,
                height: self.size.1,
                antialiasing_method: AA,
            },
        )?;
        Ok(&self.texture)
    }

    /// Take a window size, rebuilding the target when it moved.
    fn resize(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        let size = (size.0.max(1), size.1.max(1));
        if self.size == size {
            return;
        }
        let (target, view, texture) = target_of(device, &self.uploader, size);
        self.target = target;
        self.view = view;
        self.texture = texture;
        self.size = size;
    }
}

/// The render target, its view and the composite form of that view.
fn target_of(
    device: &wgpu::Device,
    uploader: &Uploader,
    size: (u32, u32),
) -> (wgpu::Texture, wgpu::TextureView, Texture) {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vv scene"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // The one format vello takes, and the format of every other texture in
        // this tree, so the bytes reach the screen unchanged.
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            // vello paints with compute work, so the target is a storage
            // binding and not only a render attachment.
            | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let texture = uploader.wrap(&view, size);
    (target, view, texture)
}

/// The transform that maps surface units onto window pixels.
///
/// This is the draw transform of [`region::Frame`]: the plane scales, turns
/// clockwise by the frame's angle about its own center, and that center sits at
/// the window middle plus the pan.
#[must_use]
pub fn to_window(frame: &region::Frame) -> Affine {
    let (w, h) = (f64::from(frame.img.width), f64::from(frame.img.height));
    let scale = f64::from(frame.scale);
    let center = (
        f64::from(frame.win.width) / 2.0 + f64::from(frame.pan.x),
        f64::from(frame.win.height) / 2.0 + f64::from(frame.pan.y),
    );
    Affine::translate(center)
        * Affine::rotate(f64::from(frame.angle).to_radians())
        * Affine::scale(scale)
        * Affine::translate((-w / 2.0, -h / 2.0))
}

/// The one-color brush of a solid paint.
/// The brush of a paint, whether a flat color or a ramp this call owns.
enum Brush {
    /// One color.
    Solid(VelloColor),
    /// A ramp, whose geometry vello reads through a brush transform.
    Ramp(Gradient),
}

impl SceneView {
    /// Fill `shape` with `paint`.
    ///
    /// `paint_at` maps the paint's own space onto the shape's, which is the
    /// inverse of the shape transform because every ramp of this tree is
    /// written in surface units.
    fn paint(
        &mut self,
        style: Fill,
        placed: Affine,
        paint: &Paint,
        paint_at: Option<Affine>,
        shape: &impl kurbo::Shape,
    ) {
        match brush(paint) {
            Brush::Solid(color) => self.list.fill(style, placed, color, None, shape),
            Brush::Ramp(ramp) => self.list.fill(style, placed, &ramp, paint_at, shape),
        }
    }

    /// Stroke `path` with the outline `outline` names.
    fn outline(
        &mut self,
        outline: &crate::scene::Stroke,
        placed: Affine,
        paint_at: Option<Affine>,
        path: &kurbo::BezPath,
    ) {
        match brush(&outline.paint) {
            Brush::Solid(color) => self.list.stroke(&outline.style, placed, color, None, path),
            Brush::Ramp(ramp) => self
                .list
                .stroke(&outline.style, placed, &ramp, paint_at, path),
        }
    }
}

/// The brush of `paint`.
fn brush(paint: &Paint) -> Brush {
    match paint {
        Paint::Solid(color) => Brush::Solid(vello_color(*color)),
        Paint::Linear {
            from,
            to,
            stops,
            spread,
        } => Brush::Ramp(Gradient {
            kind: GradientKind::Linear(LinearGradientPosition {
                start: kurbo::Point::new(from.0, from.1),
                end: kurbo::Point::new(to.0, to.1),
            }),
            extend: extend(*spread),
            stops: ramp(stops),
            ..Gradient::default()
        }),
        Paint::Radial {
            center,
            radius,
            focus,
            focus_radius,
            stops,
            spread,
        } => Brush::Ramp(Gradient {
            kind: GradientKind::Radial(RadialGradientPosition {
                start_center: kurbo::Point::new(focus.0, focus.1),
                start_radius: *focus_radius as f32,
                end_center: kurbo::Point::new(center.0, center.1),
                end_radius: *radius as f32,
            }),
            extend: extend(*spread),
            stops: ramp(stops),
            ..Gradient::default()
        }),
    }
}

/// The stops of a ramp, as vello reads them.
fn ramp(stops: &[Stop]) -> peniko::ColorStops {
    let mut out = peniko::ColorStops::new();
    for stop in stops {
        out.push(ColorStop {
            offset: stop.offset,
            color: vello_color(stop.color).into(),
        });
    }
    out
}

/// What a ramp does past its last stop.
const fn extend(spread: Spread) -> Extend {
    match spread {
        Spread::Pad => Extend::Pad,
        Spread::Repeat => Extend::Repeat,
        // A mirror is a reflect: the ramp runs backwards past its last stop.
        Spread::RepeatMirror => Extend::Reflect,
    }
}

/// The fill rule of a shape.
const fn vello_fill_rule(rule: FillRule) -> Fill {
    match rule {
        FillRule::NonZero => Fill::NonZero,
        FillRule::EvenOdd => Fill::EvenOdd,
    }
}

/// One of this tree's colors as vello's.
fn vello_color(color: Color) -> VelloColor {
    VelloColor::from_rgba8(color.r, color.g, color.b, color.a)
}

/// The picture of an image leaf, as a brush a fill samples.
///
/// The bytes stay straight alpha, which is the convention of the whole tree:
/// every decoder writes straight bytes and the SVG backend un-premultiplies
/// before it hands a picture over.
fn image_brush(pixels: &DecodedImage) -> ImageBrush {
    let (width, height) = (pixels.width().raw(), pixels.height().raw());
    let data = ImageData {
        data: Blob::from(pixels.as_rgba().to_vec()),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width,
        height,
    };
    ImageBrush::new(data)
}
