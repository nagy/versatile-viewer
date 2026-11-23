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
//! # Encoding once
//!
//! A list carries no scale and no region, so the pass encodes it once in surface
//! units and places the whole scene per frame with one [`Vello::append`], which
//! multiplies the transform of every draw. A zoom or a pan therefore costs a copy
//! of the encoded stream (about a fiftieth of the encode) and never another walk
//! of the paths. `vello` reads the encoded curves and flattens them on the
//! device during the render, so the placed list is exactly the picture a list
//! encoded for that zoom would have drawn — not a magnified polyline.
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
    AaConfig, AaSupport, Glyph, RenderParams, Renderer as VelloRenderer, RendererOptions,
    Scene as Vello,
    kurbo::{self, Affine},
    peniko::{
        self as peniko, Blob, Color as VelloColor, ColorStop, Extend, Fill, FontData, Gradient,
        GradientKind, ImageAlphaType, ImageBrush, ImageData, ImageFormat, LinearGradientPosition,
        RadialGradientPosition,
    },
};

use crate::{
    DecodedImage,
    dim::{self, Rect},
    draw::{Color, Quad, Vec2},
    region,
    scene::{FillRule, Paint, Primitive, Spread, Stop},
    texture::{Texture, Uploader},
};

/// How many parsed faces the pass keeps.
///
/// A document names a handful of faces and a treemap names one, so the table
/// stays small; a session that walks a directory of PDFs would otherwise pin
/// the fonts of every document it ever showed.
const FACES: usize = 8;

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
    /// The list of the plane being drawn, encoded once in surface units.
    ///
    /// A list carries no scale and no region, so its geometry never changes: it
    /// encodes once and every frame places it with one [`Vello::append`], which
    /// multiplies the transform of every draw and leaves the paths, the ramps
    /// and the pictures alone. `vello` flattens a curve on the device during the
    /// render, so a placed list is as sharp at a 10x zoom as one encoded for it.
    base: Vello,
    /// What `base` holds: the list and the plane its layers are clipped to.
    encoded: Option<Base>,
    /// The scene of this frame: `base` placed by the frame's transform, reused
    /// so a frame allocates no scene.
    framed: Vello,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    /// The composite side of `view`, which a quad draw samples.
    texture: Texture,
    /// The size of `target`, in window pixels.
    size: (u32, u32),
    uploader: Uploader,
    /// The picture of each [`Primitive::Image`] leaf, keyed by its address.
    images: HashMap<usize, ImageBrush>,
    /// The parsed face of each [`Primitive::Glyphs`] run, keyed by its bytes.
    ///
    /// Each entry keeps the bytes it parsed, so the address it is keyed on
    /// cannot be handed to another face while the entry lives.
    /// The table also outlives the list that named it: a run carries its own
    /// bytes, and every label of a map names one shared face for the process,
    /// so a relaid map draws its labels without parsing the face again.
    fonts: HashMap<usize, (Arc<[u8]>, FontData)>,
    /// The frame `target` holds: its list, its transform and its size.
    ///
    /// A frame that asks for exactly this one draws the texture on screen
    /// again instead of painting the list, because a paint parses and encodes
    /// every primitive of it and the answer would be the same picture.
    drawn: Option<Drawn>,
    /// The window size the frame before this one asked for.
    ///
    /// A size that moved since then means the window itself is moving, and a
    /// drag of a window changes it on every frame: the paint of each one would
    /// re-encode the whole list, so the frame that finds a moved size reuses
    /// the texture it has (see [`Self::place_stale`]) and the paint lands once
    /// the size holds still.
    asked: Option<(u32, u32)>,
}

/// The list and the plane a [`SceneView::base`] holds.
#[derive(Clone)]
struct Base {
    /// The list it encoded.
    list: Arc<[Primitive]>,
    /// The plane, in surface units, which bounds an opacity layer.
    ///
    /// A layer is a buffer of the shape vello is handed, and the list is
    /// encoded without a window, so the plane is the one shape that bounds it
    /// for every frame the list draws in.
    plane: (f32, f32),
}

impl Base {
    /// Whether this base is the answer for `list` on `plane`.
    ///
    /// The handle and not the contents: a list is immutable and lives for as
    /// long as the plane does, so the same handle is the same geometry.
    fn holds(&self, list: &Arc<[Primitive]>, plane: (f32, f32)) -> bool {
        self.plane == plane && Arc::ptr_eq(&self.list, list)
    }
}

/// The frame the target holds.
#[derive(Clone)]
struct Drawn {
    /// The list it painted.
    ///
    /// The handle and not its address: a freed list hands its address to the
    /// next allocation, and a reuse that read it as the same list would draw a
    /// picture that is not there.
    list: Arc<[Primitive]>,
    /// The surface-to-window transform it painted with.
    to_window: Affine,
    /// The target size it painted at.
    size: (u32, u32),
}

/// What one display pass answers: the texture to composite, the part of it the
/// picture fills, and where that part lands.
///
/// A painted frame fills its texture and lands on the whole window. A frame
/// that reused the paint of an earlier one answers that paint, scaled, turned
/// and moved by the transform between the two windows, so a window drag shows a
/// picture that sits where the drag put it and only lags in sharpness.
#[derive(Clone, Copy)]
pub struct Painted<'a> {
    /// The rendered list.
    pub texture: &'a Texture,
    /// The part of `texture` the picture fills, in its own pixels.
    pub src: Rect,
    /// Where that part lands, in window pixels.
    pub quad: Quad,
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
            base: Vello::new(),
            encoded: None,
            framed: Vello::new(),
            target,
            view,
            texture,
            size: (1, 1),
            uploader,
            images: HashMap::new(),
            fonts: HashMap::new(),
            drawn: None,
            asked: None,
        })
    }

    /// Draw `list` in window pixels and answer the texture to composite.
    ///
    /// `frame` maps surface units onto the window, so the same list draws
    /// sharply at every zoom and under any turn.
    ///
    /// The list encodes once, in surface units, and each frame places the whole
    /// of it with one [`Vello::append`], so a zoom and a pan cost a copy and not
    /// an encode (see [`Self::base`]).
    ///
    /// A frame that asks for the paint of the frame before it reuses that paint,
    /// and a frame whose window moved since the last one reuses it too and
    /// places it where the new window puts the same surface units (see
    /// [`Self::place_stale`]). The paint is the one expensive step of this pass
    /// and none of those frames changes a pixel of it.
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
    ) -> Result<Painted<'_>, SceneError> {
        let to_window = to_window(frame);
        let moved = self.asked != Some(size);
        self.asked = Some(size);
        if let Some(drawn) = &self.drawn {
            // The same list under the same transform at the same size is the
            // picture the target already holds: the draw composites it again
            // and the pass does nothing.
            if !moved
                && drawn.size == size
                && drawn.to_window == to_window
                && Arc::ptr_eq(&drawn.list, list)
            {
                return Ok(self.painted_whole());
            }
            // A window that moved since the last frame is still moving, so this
            // frame places the paint it has and leaves the paint of the settled
            // size to the frame that finds the size at rest.
            // A list this pass never painted is painted: a new list is the one
            // thing a window drag cannot answer from the paint it has.
            if moved && drawn.size != size && Arc::ptr_eq(&drawn.list, list) {
                return Ok(self.place_stale(drawn, to_window));
            }
        }
        let plane = (frame.img.width, frame.img.height);
        if !self
            .encoded
            .as_ref()
            .is_some_and(|base| base.holds(list, plane))
        {
            if self
                .encoded
                .as_ref()
                .is_some_and(|base| !Arc::ptr_eq(&base.list, list))
            {
                // Another list is another set of leaves, so the pictures of the
                // last one cannot be trusted and the addresses they were keyed
                // on may now mean something else.
                // The faces stay: a run carries its bytes, so a face of this
                // tree outlives the list that drew it, and vello caches the
                // outlines of a face by its bytes. Dropping them would make the
                // first frame of every relaid map outline every label again.
                self.images.clear();
            }
            self.base.reset();
            self.encode(list, plane);
            self.encoded = Some(Base {
                list: Arc::clone(list),
                plane,
            });
        }
        self.resize(device, size);
        self.framed.reset();
        self.framed.append(&self.base, Some(to_window));
        self.renderer.render_to_texture(
            device,
            queue,
            &self.framed,
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
        self.drawn = Some(Drawn {
            list: Arc::clone(list),
            to_window,
            size: self.size,
        });
        Ok(self.painted_whole())
    }

    /// The paint on screen as the whole window it covers.
    fn painted_whole(&self) -> Painted<'_> {
        let whole = dim::whole(self.size.0 as f32, self.size.1 as f32);
        Painted {
            texture: &self.texture,
            src: whole,
            quad: Quad::rect(whole),
        }
    }

    /// The paint on screen placed where `to_window` puts the frame it painted.
    ///
    /// The paint of `drawn` covers the window of that frame, so the picture
    /// lands where the mapping `to_window ∘ drawn.to_window⁻¹` puts it: a
    /// window drag changes the fit scale, so that mapping is a scale, a turn and
    /// a move, and the texture fills the rect it names.
    fn place_stale(&self, drawn: &Drawn, to_window: Affine) -> Painted<'_> {
        let (src, quad) = stale_place(drawn.size, drawn.to_window, to_window);
        Painted {
            texture: &self.texture,
            src,
            quad,
        }
    }

    /// Encode every primitive of `list` in surface units.
    ///
    /// Nothing here is placed by a window: the frame that draws the list places
    /// all of it at once (see [`Self::draw`]), so the geometry of a zoom or a pan
    /// costs no encode.
    fn encode(&mut self, list: &[Primitive], plane: (f32, f32)) {
        for primitive in list {
            match primitive {
                Primitive::Rect { rect, paint } => {
                    // A rect carries no transform, so both its shape and its
                    // paint live in surface units.
                    self.paint(Fill::NonZero, Affine::IDENTITY, paint, None, rect);
                }
                Primitive::Path {
                    path,
                    transform,
                    fill,
                    fill_rule,
                    stroke,
                } => {
                    // The shape carries its own transform, which is what puts a
                    // leaf of a document in surface units.
                    // A ramp is written in surface units and the shape in its
                    // own, so the brush takes the inverse the shape applied.
                    let paint_at = Some(transform.inverse());
                    if let Some(paint) = fill {
                        self.paint(
                            vello_fill_rule(*fill_rule),
                            *transform,
                            paint,
                            paint_at,
                            path,
                        );
                    }
                    if let Some(outline) = stroke {
                        self.outline(outline, *transform, paint_at, path);
                    }
                }
                Primitive::Glyphs {
                    run,
                    transform,
                    paint,
                } => {
                    // The face is parsed once per run, keyed by its bytes, so a
                    // label never re-parses a font per frame.
                    // The table outlives the list: a face comes from
                    // `sans_serif_face`, one `Arc` for the process, so every
                    // label of every map names the same bytes and vello keeps
                    // the outlines it resolved for them.
                    let key = run.font.as_ptr() as usize;
                    if self.fonts.len() >= FACES && !self.fonts.contains_key(&key) {
                        self.fonts.clear();
                    }
                    let (_, font) = self.fonts.entry(key).or_insert_with(|| {
                        (
                            Arc::clone(&run.font),
                            FontData::new(Blob::from(run.font.to_vec()), run.face),
                        )
                    });
                    let placed = *transform;
                    let glyphs = run.glyphs.iter().map(|glyph| Glyph {
                        id: glyph.id,
                        x: glyph.x,
                        y: glyph.y,
                    });
                    match brush(paint) {
                        Brush::Solid(color) => {
                            self.base
                                .draw_glyphs(font)
                                .font_size(run.size)
                                .brush(color)
                                .transform(placed)
                                .draw(Fill::NonZero, glyphs);
                        }
                        Brush::Ramp { gradient, space } => {
                            // A glyph run's coordinates are in its own space, so
                            // the ramp reaches it the way it reaches a path.
                            let at = brush_at(Some(placed.inverse()), space);
                            self.base
                                .draw_glyphs(font)
                                .font_size(run.size)
                                .brush(&gradient)
                                .brush_transform(Some(at))
                                .transform(placed)
                                .draw(Fill::NonZero, glyphs);
                        }
                    }
                }
                Primitive::PushClip {
                    path,
                    transform,
                    fill_rule,
                } => {
                    self.base
                        .push_clip_layer(vello_fill_rule(*fill_rule), *transform, path);
                }
                Primitive::PopClip | Primitive::PopLayer => self.base.pop_layer(),
                Primitive::PushLayer { alpha } => {
                    // vello opens a layer only around a clip shape, and the list
                    // is encoded without a window, so the layer is bounded by
                    // the plane: no window shows more of it than that, and the
                    // `append` of the frame carries the clip into the window with
                    // the rest of the list.
                    let whole = kurbo::Rect::new(0.0, 0.0, f64::from(plane.0), f64::from(plane.1));
                    self.base.push_layer(
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
                    self.base.draw_image(brush.as_ref(), *transform);
                }
            }
        }
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
    /// A ramp, and the space its own geometry is written in.
    ///
    /// A linear ramp is written in surface units, so its space is the identity
    /// and its two points already land where they belong; a radial keeps its
    /// circles in its own space and lets that space squash them, so the space
    /// travels with the ramp and the draw composes it (see [`brush_at`]).
    Ramp { gradient: Gradient, space: Affine },
}

impl SceneView {
    /// Fill a shape of the list with `paint`, in surface units.
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
            Brush::Solid(color) => self.base.fill(style, placed, color, None, shape),
            Brush::Ramp { gradient, space } => {
                let at = Some(brush_at(paint_at, space));
                self.base.fill(style, placed, &gradient, at, shape);
            }
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
            Brush::Solid(color) => self.base.stroke(&outline.style, placed, color, None, path),
            Brush::Ramp { gradient, space } => {
                let at = Some(brush_at(paint_at, space));
                self.base
                    .stroke(&outline.style, placed, &gradient, at, path);
            }
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
        } => Brush::Ramp {
            gradient: Gradient {
                kind: GradientKind::Linear(LinearGradientPosition {
                    start: kurbo::Point::new(from.0, from.1),
                    end: kurbo::Point::new(to.0, to.1),
                }),
                extend: extend(*spread),
                stops: ramp(stops),
                ..Gradient::default()
            },
            // A linear ramp of this list is stated in surface units.
            space: Affine::IDENTITY,
        },
        Paint::Radial {
            center,
            radius,
            focus,
            focus_radius,
            transform,
            stops,
            spread,
        } => Brush::Ramp {
            gradient: Gradient {
                kind: GradientKind::Radial(RadialGradientPosition {
                    start_center: kurbo::Point::new(focus.0, focus.1),
                    start_radius: *focus_radius as f32,
                    end_center: kurbo::Point::new(center.0, center.1),
                    end_radius: *radius as f32,
                }),
                extend: extend(*spread),
                stops: ramp(stops),
                ..Gradient::default()
            },
            space: *transform,
        },
    }
}

/// The brush transform of a ramp whose own space lands on the plane by `space`.
///
/// `shape_at` is the inverse of the shape's transform, which is the brush
/// transform a ramp written in surface units takes: it hands the ramp to the
/// shape's own space, where the fill reads it.
/// A ramp with a space of its own goes through that space first, so `space`
/// multiplies in on the right and the ramp still lands in surface units before
/// the shape takes it back.
fn brush_at(shape_at: Option<Affine>, space: Affine) -> Affine {
    shape_at.map_or(space, |inverse| inverse * space)
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

/// Where the paint of a frame lands in a window that another frame names.
///
/// The paint covers the window of `drawn_to_window`, and the picture of it is
/// the one `to_window` puts in the same place, so the whole texture fills the
/// rect that `to_window ∘ drawn_to_window⁻¹` names: its middle is the image of
/// the old window's middle, and its scale and turn are the ones of that mapping.
/// Both transforms belong to a frame with a drawable plane, so the mapping is a
/// similarity and one rect beside one angle says it.
#[must_use]
fn stale_place(drawn_size: (u32, u32), drawn_to_window: Affine, to_window: Affine) -> (Rect, Quad) {
    let (width, height) = (f64::from(drawn_size.0), f64::from(drawn_size.1));
    let src = dim::whole(drawn_size.0 as f32, drawn_size.1 as f32);
    let delta = to_window * drawn_to_window.inverse();
    // The matrix of an affine is `[a c e; b d f]`, so this reads the picture's
    // own x and y coefficients by the axis they move.
    let [xx, yx, xy, yy, tx, ty] = delta.as_coeffs();
    let scale = xx.hypot(yx);
    let (scale, angle) = if scale.is_finite() && scale > 0.0 {
        (scale, yx.atan2(xx).to_degrees() as f32)
    } else {
        // A degenerate mapping cannot place a picture, and the window it names
        // shows nothing either: the paint lands where it was.
        (1.0, 0.0)
    };
    let center = Vec2::new(
        (xx * width / 2.0 + xy * height / 2.0 + tx) as f32,
        (yx * width / 2.0 + yy * height / 2.0 + ty) as f32,
    );
    let (dest_w, dest_h) = ((width * scale) as f32, (height * scale) as f32);
    let dest = dim::rect(
        center.x - dest_w / 2.0,
        center.y - dest_h / 2.0,
        dest_w,
        dest_h,
    );
    (src, Quad::turned(dest, center, angle))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame that fits a plane of `img` into a window of `win`, unturned and
    /// unpanned.
    fn fit(win: (f32, f32), img: (f32, f32)) -> Affine {
        to_window(&region::Frame {
            img: dim::SurfaceSize::new(img.0, img.1),
            win: dim::WindowSize::new(win.0, win.1),
            pan: dim::WindowVector::new(0.0, 0.0),
            scale: (win.0 / img.0).min(win.1 / img.1),
            angle: 0.0,
        })
    }

    /// The four corners of a window, in the order [`Quad::rect`] names them.
    fn window_corners(size: (u32, u32)) -> [kurbo::Point; 4] {
        let (w, h) = (f64::from(size.0), f64::from(size.1));
        [
            kurbo::Point::new(0.0, 0.0),
            kurbo::Point::new(w, 0.0),
            kurbo::Point::new(w, h),
            kurbo::Point::new(0.0, h),
        ]
    }

    /// A reused paint lands exactly where the mapping of the two windows puts
    /// it, corner for corner.
    ///
    /// The pass answers one rect and one angle, so this pins the reading of the
    /// similarity out of the affine: a sign, a corner order or a turn direction
    /// that disagreed would land a corner somewhere else.
    #[test]
    fn a_reused_paint_follows_the_mapping_of_the_two_windows() {
        let plane = (400.0, 300.0);
        let cases = [
            // The same window twice.
            ((800.0, 600.0), (800.0, 600.0), 0.0),
            // A window that grew on both axes.
            ((800.0, 600.0), (1200.0, 900.0), 0.0),
            // A turned one.
            ((800.0, 600.0), (800.0, 600.0), 90.0),
            // A grown and turned one.
            ((900.0, 700.0), (1400.0, 1000.0), 30.0),
            // A shrunk one.
            ((900.0, 700.0), (500.0, 900.0), -15.0),
        ];
        for (from, to, angle) in cases {
            let drawn = fit(from, plane);
            let now = to_window(&region::Frame {
                img: dim::SurfaceSize::new(plane.0, plane.1),
                win: dim::WindowSize::new(to.0, to.1),
                pan: dim::WindowVector::new(10.0, -20.0),
                scale: (to.0 / plane.0).min(to.1 / plane.1),
                angle,
            });
            let size = (from.0 as u32, from.1 as u32);
            let (src, quad) = stale_place(size, drawn, now);
            assert_eq!(
                src,
                dim::whole(from.0, from.1),
                "the paint fills its own texture"
            );
            let delta = now * drawn.inverse();
            for (got, corner) in quad.p.into_iter().zip(window_corners(size)) {
                let want = delta * corner;
                assert!(
                    got.abs_diff_eq(Vec2::new(want.x as f32, want.y as f32), 0.01),
                    "a corner of {from:?} -> {to:?} at {angle} deg landed at {got:?} instead of \
                     ({:.2}, {:.2})",
                    want.x,
                    want.y
                );
            }
        }
    }

    /// A window that doubles its size on the plane's fit covers the new window
    /// with the old paint, so the paint is not letterboxed by its own growth.
    #[test]
    fn a_doubled_window_stretches_the_fit_over_it() {
        let plane = (400.0, 300.0);
        let drawn = fit((800.0, 600.0), plane);
        let now = fit((1600.0, 1200.0), plane);
        let (_, quad) = stale_place((800, 600), drawn, now);
        let want = Quad::rect(dim::whole(1600.0, 1200.0));
        for (got, want) in quad.p.into_iter().zip(want.p) {
            assert!(
                got.abs_diff_eq(want, 0.01),
                "a corner landed at {got:?} instead of {want:?}"
            );
        }
    }
}
