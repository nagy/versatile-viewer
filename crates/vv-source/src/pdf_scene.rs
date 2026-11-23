//! The `hayro` device that answers one PDF page as a primitive list.
//!
//! [`crate::pdf`] rasterizes a page through `hayro`'s own CPU sink. This module
//! drives the same interpreter — `interpret_page` is generic over `Device` — and
//! collects the primitives instead of pixels, so a page draws at any zoom
//! instead of re-rasterizing at every one.
//!
//! # Units
//!
//! Every primitive lands in the page's render space: points with the y axis
//! down, the space [`crate::surface::Surface::bounds`] measures. `hayro` hands a
//! device the full transform of each draw (`DrawProps::transform`), so the walk
//! composes nothing and the list carries the page's own geometry.
//!
//! # What a list cannot express
//!
//! [`Unsupported`] names it, and the page then keeps the raster path whole —
//! the same trade the SVG walk makes. That covers a soft mask (an intermediate
//! buffer and a luminance pass), a blend mode other than `normal` (the backdrop),
//! a tiling pattern (a repeated tile) and the shadings that a ramp cannot state.
//!
//! # Shadings
//!
//! A gradient background is a *shading pattern*, and it answers a
//! [`Paint::Linear`] or a [`Paint::Radial`] — the same two ramps the SVG walk
//! builds. `hayro` encodes the shading and samples its function into stops, so
//! both paths read the same colours, and the transform the encoded shading
//! carries already maps its own space into the page, because the interpreter
//! pre-composes the pattern's matrix with the page — so the ramp's axis needs no
//! placement of its own.
//!
//! # Glyphs
//!
//! A PDF run arrives already shaped: `hayro` names each glyph and its placement,
//! and `OutlineGlyph::outline` answers its contour in a 1000-unit em. So the walk
//! bakes the placement into the outline and merges every glyph of a run into one
//! path — one primitive per run rather than one per glyph, and no face to parse
//! again. A Type 3 glyph is a drawing program instead of an outline, so it is
//! interpreted into this same device, which is the recursion `hayro` itself
//! makes.

use std::sync::Arc;

use hayro::hayro_interpret::{
    BlendMode, ClipPath, Device, DrawMode, DrawProps, FillRule, Image, ImageData, ImageDrawProps,
    Paint as PdfPaint, StrokeProps,
    encode::EncodedShadingType,
    font::{Glyph, GlyphRun},
    gradient::SvgGradientKind,
    pattern::{Pattern as PdfPattern, ShadingPattern},
};
use kurbo::{Affine, BezPath, Point, Rect, Shape};

use crate::{
    DecodedImage,
    dim::{ImgH, ImgW},
    draw::Color,
    scene::{self, Paint, Primitive, Stroke},
};

/// The paper a page draws on.
///
/// The raster path clears its pixmap to white (`TargetInit::Clear(WHITE)` in
/// [`crate::pdf`]), so a page of a PDF is white where it paints nothing. A list
/// paints no paper of its own, so the walk puts this one down first and every
/// other primitive of the page lands on it.
const PAPER: Color = Color::new(255, 255, 255, 255);

/// The largest colour error a ramp may carry against the shading it replaces.
///
/// A shading is a function sampled at every pixel, and a ramp is a list of stops
/// the renderer interpolates between. `hayro` approximates the same shading with
/// the same ramp at this tolerance, so both paths read the same colours out of
/// the same function.
const RAMP_TOLERANCE: f32 = 0.01;

/// How far from square and round a transform may sit and still count as a
/// similarity, relative to its own scale.
///
/// A page's transform is exact apart from rounding, so a squeezed shading misses
/// this by whole percent rather than by a hair.
const SIMILARITY: f64 = 1.0e-6;

/// What a page needs that a primitive list cannot express.
///
/// A caller that reads one of these answers no list, so the page keeps the
/// raster path and its picture stays exactly what it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unsupported {
    /// A soft mask, which needs an intermediate buffer and a luminance pass.
    Mask,
    /// A blend mode other than `normal`, which needs the backdrop.
    Blend,
    /// A shading that a ramp of this list cannot state.
    ///
    /// That covers a function-based or a mesh shading, which paints a function
    /// over an area rather than along an axis; a shading that does not cover the
    /// shape it paints, where a ramp would pad past its last stop; and a radial
    /// shading whose matrix squeezes its circles into ellipses, which two
    /// circular ramps cannot draw.
    Shading,
    /// A tiling pattern, which needs a tile drawn once and repeated.
    Pattern,
}

impl Unsupported {
    /// The name a diagnostic reports.
    #[must_use]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Mask => "soft mask",
            Self::Blend => "blend mode",
            Self::Shading => "shading",
            Self::Pattern => "pattern",
        }
    }
}

/// One page, walked into the primitives that draw it.
pub(crate) struct PdfScene {
    out: Vec<Primitive>,
    /// The page's own box, in render space.
    ///
    /// A ramp has to know the area it covers, and `hayro` clips that area to its
    /// target, which for this walk is the page.
    page: Rect,
    /// The feature that stopped the walk.
    ///
    /// The walk stops at the first one, because a page that needs it keeps the
    /// raster path whole: no later primitive of it could be drawn by the list.
    stop: Option<Unsupported>,
}

impl PdfScene {
    /// A collector that holds a page of `width` by `height`, paper included.
    pub(crate) fn new(width: f32, height: f32) -> Self {
        let page = Rect::new(0.0, 0.0, f64::from(width), f64::from(height));
        Self {
            out: vec![Primitive::Rect {
                rect: page,
                paint: Paint::Solid(PAPER),
            }],
            page,
            stop: None,
        }
    }

    /// The list of the page, or the feature that kept it a raster.
    pub(crate) fn finish(self) -> Result<Arc<[Primitive]>, Unsupported> {
        match self.stop {
            Some(why) => Err(why),
            None => Ok(Arc::from(self.out)),
        }
    }

    /// Stop the walk at `why`, keeping the first reason.
    fn stop(&mut self, why: Unsupported) {
        self.stop.get_or_insert(why);
    }

    /// Whether the walk already stopped, so this call has nothing to do.
    const fn stopped(&self) -> bool {
        self.stop.is_some()
    }

    /// The paint of a shape of `shape`'s box, and the clip its shading carries.
    ///
    /// `shape` is the box the paint has to cover, in surface units.
    /// `None` stops the walk: nothing of this page may enter the list after a
    /// paint it cannot state.
    fn paint_of(&mut self, paint: &PdfPaint<'_>, shape: Rect) -> Option<(Paint, Option<BezPath>)> {
        match paint {
            PdfPaint::Color(color) => {
                let [r, g, b, a] = color.to_rgba().to_rgba8();
                Some((Paint::Solid(Color::new(r, g, b, a)), None))
            }
            PdfPaint::Pattern(pattern) => match pattern.as_ref() {
                PdfPattern::Shading(shading) => self.ramp(shading, shape),
                PdfPattern::Tiling(_) => {
                    // A tile has to render once and repeat, which the list
                    // cannot say yet.
                    self.stop(Unsupported::Pattern);
                    None
                }
            },
        }
    }

    /// A shading as a ramp in surface units, and the box it clips itself to.
    ///
    /// `hayro` encodes the shading and approximates its colour function with
    /// stops, so this walk and the raster path read the same function at the same
    /// tolerance.
    /// A pattern's matrix is anchored in the default space of the page, not in
    /// the transform of the draw, and `hayro` pre-composes that space into the
    /// matrix, so the transform the encoded shading carries already lands in the
    /// render space: composing the draw's transform as well would apply it twice
    /// and move every ramp.
    fn ramp(&mut self, pattern: &ShadingPattern, shape: Rect) -> Option<(Paint, Option<BezPath>)> {
        let encoded = pattern.encode();
        let EncodedShadingType::RadialAxial(gradient) = &encoded.shading_type else {
            // A function-based shading paints a function over an area and a mesh
            // shading paints sampled corners: neither runs along an axis.
            self.stop(Unsupported::Shading);
            return None;
        };
        let Some(native) = gradient.as_svg_gradient(&encoded, shape, RAMP_TOLERANCE) else {
            // The shading does not cover the shape it paints — its extend is off,
            // or it lays its own background down — and a ramp pads past its last
            // stop, so it would paint where the page paints nothing.
            self.stop(Unsupported::Shading);
            return None;
        };
        let space = native.transform;
        let stops: Arc<[scene::Stop]> = native
            .stops
            .iter()
            .map(|stop| scene::Stop {
                offset: stop.offset,
                color: srgb(stop.color),
            })
            .collect();
        let paint = match native.kind {
            SvgGradientKind::Linear { start, end } => Paint::Linear {
                from: placed(space, start),
                to: placed(space, end),
                stops,
                spread: scene::Spread::Pad,
            },
            SvgGradientKind::Radial {
                start_center,
                start_radius,
                end_center,
                end_radius,
            } => {
                // A ramp of this list is a pair of circles. A shading matrix that
                // squeezes or shears turns those circles into ellipses, and
                // drawing them round would paint a different picture, so the page
                // keeps the raster path.
                let Some(scale) = similarity(space) else {
                    self.stop(Unsupported::Shading);
                    return None;
                };
                Paint::Radial {
                    center: placed(space, end_center),
                    radius: f64::from(end_radius) * scale,
                    focus: placed(space, start_center),
                    focus_radius: f64::from(start_radius) * scale,
                    stops,
                    spread: scene::Spread::Pad,
                }
            }
        };
        let clip = pattern.shading.clip_path.clone();
        Some((paint, clip))
    }

    /// The paint of a path drawn by `props`, its own box placed in the page.
    fn path_paint(
        &mut self,
        props: &DrawProps<'_>,
        path: &BezPath,
        draw_mode: &DrawMode,
    ) -> Option<(Paint, Option<BezPath>)> {
        let mut shape = props.transform.transform_rect_bbox(path.bounding_box());
        // `hayro` widens the box of a stroke by the width its outline takes, so a
        // ramp is measured against the ink and not the centre line.
        let width = match draw_mode {
            DrawMode::Stroke(stroke) | DrawMode::FillAndStroke(_, stroke) => stroke.line_width,
            DrawMode::Fill(_) | DrawMode::Invisible => 0.0,
        };
        if width > 0.0 {
            let factor = longest_advance(props.transform) * f64::from(width);
            shape = shape.inflate(factor, factor);
        }
        self.paint_of(&props.paint, shape.intersect(self.page))
    }

    /// Open the clip a shading carries, and answer whether it did.
    ///
    /// A shading names its own box, which `hayro` composes into device space, so
    /// the path lands in the render space as it is.
    fn open_clip(&mut self, clip: Option<&BezPath>) -> bool {
        let Some(path) = clip else {
            return false;
        };
        self.out.push(Primitive::PushClip {
            path: path.clone(),
            transform: Affine::IDENTITY,
            fill_rule: scene::FillRule::NonZero,
        });
        true
    }

    /// Close the clip [`Self::open_clip`] opened.
    fn close_clip(&mut self, opened: bool) {
        if opened {
            self.out.push(Primitive::PopClip);
        }
    }

    /// Whether a draw of `props` may enter this list.
    fn drawable(&mut self, soft_mask: bool, blend: BlendMode) -> bool {
        if soft_mask {
            self.stop(Unsupported::Mask);
            return false;
        }
        if blend != BlendMode::Normal {
            self.stop(Unsupported::Blend);
            return false;
        }
        true
    }

    /// Append the fill of one glyph run, merged into one path.
    ///
    /// The placement of every glyph is baked into its outline, so the run draws
    /// as one shape and one primitive, and a translucent paint lands once
    /// instead of once per glyph.
    fn fill_run(&mut self, glyph_run: &GlyphRun<'_, '_>, props: &DrawProps<'_>, rule: FillRule) {
        let mut merged = BezPath::new();
        for glyph in glyph_run.glyphs() {
            let Glyph::Outline(outline) = &**glyph else {
                continue;
            };
            let placed = props.transform * glyph.transform() * outline.outline();
            merged.extend(placed.elements().iter().copied());
        }
        if merged.elements().is_empty() {
            return;
        }
        // The outlines are already in the page, so the paint is measured against
        // them where they are.
        let shape = merged.bounding_box().intersect(self.page);
        let Some((paint, clip)) = self.paint_of(&props.paint, shape) else {
            return;
        };
        let opened = self.open_clip(clip.as_ref());
        self.out.push(Primitive::Path {
            path: merged,
            transform: Affine::IDENTITY,
            fill: Some(paint),
            fill_rule: fill_rule(rule),
            stroke: None,
        });
        self.close_clip(opened);
    }

    /// Append the stroke of one glyph run, one path per glyph.
    ///
    /// `hayro` strokes a glyph in its text space under the run's transform, so
    /// the line width is measured there and not in the page, and one outline per
    /// glyph says that exactly.
    fn stroke_run(
        &mut self,
        glyph_run: &GlyphRun<'_, '_>,
        props: &DrawProps<'_>,
        stroke: &StrokeProps,
    ) {
        let mut outlines = Vec::new();
        for glyph in glyph_run.glyphs() {
            let Glyph::Outline(outline) = &**glyph else {
                continue;
            };
            let placed = glyph.transform() * outline.outline();
            if !placed.elements().is_empty() {
                outlines.push(placed);
            }
        }
        if outlines.is_empty() {
            return;
        }
        // The box of the ink: every outline of the run, widened by the join the
        // widest corner of the stroke takes.
        let extent = f64::from(stroke.line_width) / 2.0
            * f64::from(stroke.miter_limit).max(std::f64::consts::SQRT_2);
        let union = outlines.iter().fold(Rect::ZERO, |area, outline| {
            area.union(outline.bounding_box())
        });
        let shape = props
            .transform
            .transform_rect_bbox(union.inflate(extent, extent))
            .intersect(self.page);
        let Some((paint, clip)) = self.paint_of(&props.paint, shape) else {
            return;
        };
        let opened = self.open_clip(clip.as_ref());
        let style = outline_style(stroke);
        for placed in outlines {
            self.out.push(Primitive::Path {
                path: placed,
                transform: props.transform,
                fill: None,
                fill_rule: scene::FillRule::NonZero,
                stroke: Some(Stroke {
                    style: style.clone(),
                    paint: paint.clone(),
                }),
            });
        }
        self.close_clip(opened);
    }

    /// Append one picture, in the box its transform names.
    ///
    /// `scale` is the picture's own scale factors — 1.0 where the decoder
    /// answered the size the page named, and more where it answered a lower
    /// resolution, so the picture covers the same box either way.
    /// An [`Primitive::Image`] maps the picture's **pixels** onto the plane, so
    /// the leaf already covers `width` by `height` of its own units and the
    /// transform carries pixels to points (see [`scene::image_of`], which divides
    /// a box by the picture's size for the same reason). `hayro` places the same
    /// picture as a rect of that size under this transform, so a walk that also
    /// composes the size into it squares the scale and throws the picture far
    /// outside the page.
    fn picture(&mut self, pixels: DecodedImage, scale: (f32, f32), transform: Affine) {
        let (scale_x, scale_y) = scale;
        self.out.push(Primitive::Image {
            pixels,
            transform: transform
                * Affine::scale_non_uniform(f64::from(scale_x), f64::from(scale_y)),
        });
    }
}

impl<'a> Device<'a> for PdfScene {
    fn draw_path(&mut self, path: &BezPath, props: DrawProps<'a>, draw_mode: &DrawMode) {
        if self.stopped() || !self.drawable(props.soft_mask.is_some(), props.blend_mode) {
            return;
        }
        if matches!(draw_mode, DrawMode::Invisible) {
            return;
        }
        let Some((paint, clip)) = self.path_paint(&props, path, draw_mode) else {
            return;
        };
        let (fill, stroke) = match draw_mode {
            DrawMode::Fill(rule) => (Some((paint, fill_rule(*rule))), None),
            DrawMode::Stroke(stroke) => (
                None,
                Some(Stroke {
                    style: outline_style(stroke),
                    paint,
                }),
            ),
            DrawMode::FillAndStroke(rule, stroke) => (
                Some((paint.clone(), fill_rule(*rule))),
                Some(Stroke {
                    style: outline_style(stroke),
                    paint,
                }),
            ),
            // An invisible draw lays down no paint, and returned above.
            DrawMode::Invisible => return,
        };
        let opened = self.open_clip(clip.as_ref());
        self.out.push(Primitive::Path {
            path: path.clone(),
            transform: props.transform,
            fill: fill.as_ref().map(|(paint, _)| paint.clone()),
            fill_rule: fill.map_or(scene::FillRule::NonZero, |(_, rule)| rule),
            stroke,
        });
        self.close_clip(opened);
    }

    fn push_clip_path(&mut self, clip_path: &ClipPath) {
        if self.stopped() {
            return;
        }
        // `hayro` composes the clip into device space before it names it, so the
        // path lands as it is.
        self.out.push(Primitive::PushClip {
            path: clip_path.path.clone(),
            transform: Affine::IDENTITY,
            fill_rule: fill_rule(clip_path.fill),
        });
    }

    fn pop_clip(&mut self) {
        if !self.stopped() {
            self.out.push(Primitive::PopClip);
        }
    }

    fn push_transparency_group(
        &mut self,
        opacity: f32,
        mask: Option<hayro::hayro_interpret::SoftMask<'a>>,
        blend_mode: BlendMode,
    ) {
        if self.stopped() || !self.drawable(mask.is_some(), blend_mode) {
            return;
        }
        self.out.push(Primitive::PushLayer { alpha: opacity });
    }

    fn pop_transparency_group(&mut self) {
        if !self.stopped() {
            self.out.push(Primitive::PopLayer);
        }
    }

    fn draw_glyph_run(
        &mut self,
        glyph_run: &GlyphRun<'_, 'a>,
        props: DrawProps<'a>,
        draw_mode: &DrawMode,
    ) {
        if self.stopped() || !self.drawable(props.soft_mask.is_some(), props.blend_mode) {
            return;
        }
        let outlines_only = glyph_run
            .glyphs()
            .iter()
            .all(|glyph| matches!(&**glyph, Glyph::Outline(_)));
        if !outlines_only {
            // A Type 3 glyph is a drawing program, so it is interpreted one at a
            // time into this same device, with the glyph's own placement — the
            // recursion `hayro` itself makes — and nothing of the run merges.
            for glyph in glyph_run.glyphs() {
                if let Glyph::Type3(type3) = &**glyph {
                    type3.interpret(self, props.transform, glyph.transform(), &props.paint);
                }
            }
            return;
        }
        match draw_mode {
            DrawMode::Fill(rule) => self.fill_run(glyph_run, &props, *rule),
            DrawMode::Stroke(stroke) => self.stroke_run(glyph_run, &props, stroke),
            DrawMode::FillAndStroke(rule, stroke) => {
                self.fill_run(glyph_run, &props, *rule);
                self.stroke_run(glyph_run, &props, stroke);
            }
            DrawMode::Invisible => {}
        }
    }

    fn draw_image(&mut self, image: Image<'a, '_>, props: ImageDrawProps<'a>) {
        if self.stopped() || !self.drawable(props.soft_mask.is_some(), props.blend_mode) {
            return;
        }
        match image {
            Image::Raster(raster) => raster.with_rgba(
                |data, alpha| {
                    let Some(pixels) = pixels(&data, alpha.as_ref()) else {
                        return;
                    };
                    self.picture(pixels, data.scale_factors(), props.transform);
                },
                None,
            ),
            Image::Stencil(stencil) => stencil.with_stencil(
                |mask, paint| {
                    let PdfPaint::Color(color) = paint else {
                        // A stencil painted by a pattern needs a mask under a
                        // ramp, which the list cannot say yet.
                        self.stop(Unsupported::Pattern);
                        return;
                    };
                    let [r, g, b, a] = color.to_rgba().to_rgba8();
                    let Some(pixels) = stencil_pixels(&mask, Color::new(r, g, b, a)) else {
                        return;
                    };
                    self.picture(pixels, mask.scale_factors, props.transform);
                },
                None,
            ),
        }
    }
}

/// The fill rule of a scene path.
const fn fill_rule(rule: FillRule) -> scene::FillRule {
    match rule {
        FillRule::NonZero => scene::FillRule::NonZero,
        FillRule::EvenOdd => scene::FillRule::EvenOdd,
    }
}

/// One stop colour of a ramp, out of the shading's sRGB components.
///
/// A shading function answers straight-alpha components in 0..1 and a stop wants
/// bytes, so the clamp keeps a function that overshoots from wrapping.
fn srgb(rgba: [f32; 4]) -> Color {
    let [r, g, b, a] = rgba.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8);
    Color::new(r, g, b, a)
}

/// Where a point of a shading's own space lands in surface units.
fn placed(space: Affine, point: Point) -> (f64, f64) {
    let point = space * point;
    (point.x, point.y)
}

/// The scale of `space` where it is a similarity, or `None` where it squeezes or
/// shears.
///
/// A ramp of this list is a pair of circles, so its geometry only survives a
/// transform that keeps circles round: two equal axes that stay square to each
/// other.
fn similarity(space: Affine) -> Option<f64> {
    let [xx, yx, xy, yy, _, _] = space.as_coeffs();
    let (x, y) = (xx.hypot(yx), xy.hypot(yy));
    let longest = x.max(y);
    let square = (x - y).abs() <= longest * SIMILARITY;
    let round = (xx * xy + yx * yy).abs() <= longest * longest * SIMILARITY;
    (square && round).then(|| f64::midpoint(x, y))
}

/// The longest of the two axis advances of `space`.
///
/// A stroke of a width named in a shape's own units lands on the page stretched
/// by the axes of its placement, so the wider axis measures its ink.
fn longest_advance(space: Affine) -> f64 {
    let [xx, yx, xy, yy, _, _] = space.as_coeffs();
    xx.hypot(yx).max(xy.hypot(yy))
}

/// The outline style of a PDF stroke.
///
/// The width stays the one the page names: `hayro` narrows it against the device
/// scale so a hairline stays visible on a small raster, which is a property of a
/// resolution and not of a page.
fn outline_style(stroke: &StrokeProps) -> kurbo::Stroke {
    kurbo::Stroke {
        width: f64::from(stroke.line_width),
        join: stroke.line_join,
        miter_limit: f64::from(stroke.miter_limit),
        start_cap: stroke.line_cap,
        end_cap: stroke.line_cap,
        dash_pattern: stroke
            .dash_array
            .iter()
            .map(|dash| f64::from(*dash))
            .collect(),
        dash_offset: f64::from(stroke.dash_offset),
    }
}

/// The straight-alpha RGBA8 of a decoded picture, its alpha mask applied.
///
/// `hayro` hands a device RGB or luma channels with an optional separate alpha
/// mask and leaves the premultiplication to the caller; this tree keeps every
/// pixel straight, so the mask scales the alpha channel alone.
fn pixels(
    data: &ImageData,
    alpha: Option<&hayro::hayro_interpret::LumaData>,
) -> Option<DecodedImage> {
    let (width, height) = (data.width(), data.height());
    if width == 0 || height == 0 {
        return None;
    }
    let count = usize::try_from(width).ok()? * usize::try_from(height).ok()?;
    let mut out = Vec::with_capacity(count * 4);
    match data {
        ImageData::Rgb(rgb) => {
            for (index, pixel) in rgb.data.as_chunks::<3>().0.iter().enumerate() {
                let a = alpha.map_or(255, |alpha| alpha.data.get(index).copied().unwrap_or(255));
                out.extend_from_slice(&[pixel[0], pixel[1], pixel[2], a]);
            }
        }
        ImageData::Luma(luma) => {
            for (index, gray) in luma.data.iter().enumerate() {
                let a = alpha.map_or(255, |alpha| alpha.data.get(index).copied().unwrap_or(255));
                out.extend_from_slice(&[*gray, *gray, *gray, a]);
            }
        }
    }
    DecodedImage::new(ImgW::new(width), ImgH::new(height), out).ok()
}

/// The straight-alpha RGBA8 of a stencil: a solid color under a luma mask.
fn stencil_pixels(mask: &hayro::hayro_interpret::LumaData, color: Color) -> Option<DecodedImage> {
    if mask.width == 0 || mask.height == 0 {
        return None;
    }
    let mut out = Vec::with_capacity(mask.data.len() * 4);
    for value in &mask.data {
        let alpha = (u32::from(*value) * u32::from(color.a) / 255) as u8;
        out.extend_from_slice(&[color.r, color.g, color.b, alpha]);
    }
    DecodedImage::new(ImgW::new(mask.width), ImgH::new(mask.height), out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pdf::{MINIMAL_PDF, PdfSurface},
        surface::Surface,
    };

    /// A one-page document with a line of text.
    fn text_page() -> PdfSurface {
        PdfSurface::from_bytes(MINIMAL_PDF.to_vec()).expect("the fixture parses")
    }

    /// A one-page document that paints a red square under a `Multiply` blend.
    ///
    /// A blend needs the backdrop, which a list on a transparent base does not
    /// have, so the page has to keep the raster path.
    fn blended_page() -> PdfSurface {
        const BLENDED: &[u8] = b"%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 100]/Contents 4 0 R/Resources<</ExtGState<</G1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
q /G1 gs 1 0 0 rg 10 10 50 50 re f Q
endstream
endobj
5 0 obj<</Type/ExtGState/BM/Multiply>>endobj
trailer<</Root 1 0 R/Size 6>>";
        PdfSurface::from_bytes(BLENDED.to_vec()).expect("the blend fixture parses")
    }

    /// A one-page document with a two by one picture drawn in a 50 unit box.
    ///
    /// The page names the box in its own units, `q 50 0 0 50 10 20 cm`, so the
    /// picture covers 10..60 by 20..70 points from the bottom left, whatever its
    /// pixel count is.
    fn picture_page() -> PdfSurface {
        const PICTURE: &[u8] = b"%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 100 100]/Resources<</XObject<</Im1 5 0 R>>>>/Contents 4 0 R>>endobj
4 0 obj<</Length 31>>stream
q 50 0 0 50 10 20 cm /Im1 Do Q
endstream
endobj
5 0 obj<</Type/XObject/Subtype/Image/Width 2/Height 1/ColorSpace/DeviceRGB/BitsPerComponent 8/Length 6>>stream
\xff\x00\x00\x00\xff\x00
endstream
endobj
trailer<</Root 1 0 R/Size 6>>";
        PdfSurface::from_bytes(PICTURE.to_vec()).expect("the picture fixture parses")
    }

    /// A one-page document whose whole page is painted by the shading `shading`
    /// names, under the pattern `matrix`.
    ///
    /// The page is 200 by 100 and its shading covers all of it, so a ramp built
    /// from it spans the page and is easy to read back.
    fn shaded_page(shading: &str, matrix: &str) -> PdfSurface {
        let content = "/Pattern cs /P1 scn 0 0 200 100 re f\n";
        let document = format!(
            "%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox [0 0 200 100]/Contents 4 0 R/Resources<</Pattern<</P1 5 \
             0 R>>>>>>endobj
4 0 obj<</Length {}>>stream
{content}endstream
endobj
5 0 obj<</Type/Pattern/PatternType 2/Shading 6 0 R/Matrix [{matrix}]>>endobj
6 0 obj<</ShadingType {shading}/Function 7 0 R/Extend [true true]>>endobj
7 0 obj<</FunctionType 2/Domain [0 1]/C0 [1 0 0]/C1 [0 0 1]/N 1>>endobj
trailer<</Root 1 0 R/Size 8>>",
            content.len()
        );
        PdfSurface::from_bytes(document.into_bytes()).expect("the shading fixture parses")
    }

    /// A page painted by an axial shading from its left edge to its right.
    ///
    /// The shading runs in the page's own space with the y axis up, so the ramp
    /// lands on the bottom edge of the render space, where the y axis points
    /// down.
    fn axial_page(matrix: &str) -> PdfSurface {
        shaded_page("2/ColorSpace/DeviceRGB/Coords [0 0 200 0]", matrix)
    }

    /// A page painted by a radial shading: one centre, two radii.
    fn radial_page(matrix: &str) -> PdfSurface {
        shaded_page("3/ColorSpace/DeviceRGB/Coords [0 0 0 0 0 1]", matrix)
    }

    /// The one path a shaded page draws, with its fill.
    fn shaded_fill(surface: &PdfSurface) -> Paint {
        let list = surface
            .display(0)
            .expect("a list")
            .expect("a shaded page lists");
        let fills = list
            .iter()
            .filter_map(|primitive| match primitive {
                Primitive::Path { fill, .. } => fill.clone(),
                // Every other leaf lays down no fill.
                Primitive::Rect { .. }
                | Primitive::Image { .. }
                | Primitive::Glyphs { .. }
                | Primitive::PushClip { .. }
                | Primitive::PopClip
                | Primitive::PushLayer { .. }
                | Primitive::PopLayer => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(fills.len(), 1, "one shaded shape: {list:?}");
        fills.into_iter().next().expect("one fill")
    }

    /// An axial shading answers a ramp from one edge of the page to the other.
    ///
    /// The stop colours come from the shading's function, so the first is the
    /// function's value at zero and the last its value at one — the same samples
    /// `hayro` reads when it rasterizes the page.
    #[test]
    fn an_axial_shading_answers_a_ramp_across_the_page() {
        // The pattern's matrix is the identity, so the shading runs in points.
        let fill = shaded_fill(&axial_page("1 0 0 1 0 0"));
        let Paint::Linear {
            from,
            to,
            stops,
            spread,
        } = fill
        else {
            panic!("an axial shading is a linear ramp: {fill:?}");
        };
        assert!(
            (from.0).abs() < 1e-6
                && (from.1 - 100.0).abs() < 1e-6
                && (to.0 - 200.0).abs() < 1e-6
                && (to.1 - 100.0).abs() < 1e-6,
            "the ramp runs along the bottom edge of the render space: {from:?} to {to:?}"
        );
        assert_eq!(spread, scene::Spread::Pad, "a ramp pads past its ends");
        assert_eq!(
            stops.first().map(|stop| stop.color),
            Some(Color::new(255, 0, 0, 255))
        );
        assert_eq!(
            stops.last().map(|stop| stop.color),
            Some(Color::new(0, 0, 255, 255))
        );
        assert_eq!(
            (stops.first(), stops.last()),
            (
                Some(&scene::Stop {
                    offset: 0.0,
                    color: Color::new(255, 0, 0, 255)
                }),
                Some(&scene::Stop {
                    offset: 1.0,
                    color: Color::new(0, 0, 255, 255)
                })
            ),
            "the function's ends are the ramp's ends"
        );
    }

    /// The pattern's matrix places the ramp, not only the colours.
    ///
    /// The matrix maps the shading's own space onto the page's default space, so
    /// squeezing it by twenty shortens the ramp and moving it moves the ramp.
    #[test]
    fn a_shading_matrix_places_its_ramp() {
        let fill = shaded_fill(&axial_page("0.05 0 0 1 10 0"));
        let Paint::Linear { from, to, .. } = fill else {
            panic!("an axial shading is a linear ramp: {fill:?}");
        };
        assert!(
            (from.0 - 10.0).abs() < 1e-6
                && (to.0 - 20.0).abs() < 1e-6
                && (from.1 - 100.0).abs() < 1e-6
                && (to.1 - 100.0).abs() < 1e-6,
            "the ramp covers the box the matrix names: {from:?} to {to:?}"
        );
    }

    /// A one-page document painted by a shading `extra` describes.
    ///
    /// The entries go into the shading dictionary, so a test states the one it
    /// cares about — a box of its own, an extend, or neither — and nothing else.
    fn shading_with(extra: &str) -> PdfSurface {
        let content = "/Pattern cs /P1 scn 0 0 200 100 re f\n";
        let document = format!(
            "%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox [0 0 200 100]/Contents 4 0 R/Resources<</Pattern<</P1 5 \
             0 R>>>>>>endobj
4 0 obj<</Length {}>>stream
{content}endstream
endobj
5 0 obj<</Type/Pattern/PatternType 2/Shading 6 0 R>>endobj
6 0 obj<</ShadingType 2/ColorSpace/DeviceRGB/Coords [0 0 20 0]/Function 7 0 R{extra}>>endobj
7 0 obj<</FunctionType 2/Domain [0 1]/C0 [1 0 0]/C1 [0 0 1]/N 1>>endobj
trailer<</Root 1 0 R/Size 8>>",
            content.len()
        );
        PdfSurface::from_bytes(document.into_bytes()).expect("the fixture parses")
    }

    /// A shading that names its own box clips the shape it paints to it.
    ///
    /// `hayro` hands the box to its rasterizer as a clip, so a list that ignored
    /// it would paint the ramp's padding outside the box the page drew.
    #[test]
    fn a_shading_clips_itself_to_its_box() {
        // The box covers the lower half of the page in the page's own space,
        // which is the lower half of the render space once the y axis flips.
        let list = shading_with("/Extend [true true]/BBox [0 0 200 50]")
            .display(0)
            .expect("a list")
            .expect("a shaded page lists");
        let kinds = list
            .iter()
            .map(|primitive| match primitive {
                Primitive::Rect { .. } => "rect",
                Primitive::Path { .. } => "path",
                Primitive::PushClip { .. } => "push",
                Primitive::PopClip => "pop",
                Primitive::Image { .. } => "image",
                Primitive::Glyphs { .. } => "glyphs",
                Primitive::PushLayer { .. } => "layer",
                Primitive::PopLayer => "unlayer",
            })
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            ["rect", "push", "path", "pop"],
            "the box wraps the shape it paints: {list:?}"
        );
        let Primitive::PushClip {
            path, transform, ..
        } = &list[1]
        else {
            panic!("a clip comes first: {list:?}");
        };
        assert_eq!(*transform, Affine::IDENTITY, "the box lands in the page");
        let bounds = kurbo::Shape::bounding_box(path);
        assert_eq!(
            (bounds.x0, bounds.y0, bounds.x1, bounds.y1),
            (0.0, 50.0, 200.0, 100.0),
            "the box is the one the shading names"
        );
    }

    /// A shading that stops inside the shape keeps the raster path.
    ///
    /// A ramp pads past its last stop, so carrying this one would paint the
    /// ramp's end colour over the whole page where the page paints nothing.
    #[test]
    fn a_shading_that_misses_its_shape_answers_no_list() {
        let surface = shading_with("/Extend [false false]");
        assert!(
            surface.display(0).expect("an answer").is_none(),
            "a ramp that pads past its ends is not the shading the page names"
        );
    }

    /// A radial shading answers a pair of circles in surface units.
    #[test]
    fn a_radial_shading_answers_a_circle() {
        // A circle of radius one, scaled by a hundred and moved to the page.
        let fill = shaded_fill(&radial_page("100 0 0 100 100 50"));
        let Paint::Radial {
            center,
            radius,
            focus,
            focus_radius,
            ..
        } = fill
        else {
            panic!("a radial shading is a radial ramp: {fill:?}");
        };
        assert!(
            (center.0 - 100.0).abs() < 1e-6 && (center.1 - 50.0).abs() < 1e-6,
            "the ramp is centred where the matrix puts it: {center:?}"
        );
        assert!(
            (radius - 100.0).abs() < 1e-6,
            "its radius is a hundred points"
        );
        assert!((focus_radius).abs() < 1e-6, "the inner circle is a point");
        assert!(
            (focus.0 - center.0).abs() < 1e-6 && (focus.1 - center.1).abs() < 1e-6,
            "a concentric shading has one centre"
        );
    }

    /// A radial shading squeezed into an ellipse keeps the raster path.
    ///
    /// A ramp of this list is a pair of circles, so drawing a squeezed one round
    /// would paint a picture nobody drew.
    #[test]
    fn a_squeezed_radial_shading_answers_no_list() {
        let surface = radial_page("200 0 0 50 0 0");
        assert!(
            surface.display(0).expect("an answer").is_none(),
            "an elliptical ramp is not a circle pair"
        );
    }

    /// A tiling pattern keeps the raster path, because a tile has to repeat.
    #[test]
    fn a_tiling_pattern_answers_no_list() {
        const TILED: &[u8] = b"%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox [0 0 100 100]/Contents 4 0 R/Resources<</Pattern<</P1 5 0 R>>>>>>endobj
4 0 obj<</Length 32>>stream
/Pattern cs /P1 scn 0 0 100 100 re f
endstream
endobj
5 0 obj<</Type/Pattern/PatternType 1/PaintType 1/TilingType 1/BBox [0 0 10 10]/XStep 10/YStep 10/Length 26>>stream
1 0 0 rg 0 0 10 10 re f
endstream
endobj
trailer<</Root 1 0 R/Size 6>>";
        let surface = PdfSurface::from_bytes(TILED.to_vec()).expect("the fixture parses");
        assert!(
            surface.display(0).expect("an answer").is_none(),
            "a repeated tile is not a ramp"
        );
    }

    /// A line of text is one run, and one run is one merged path.
    #[test]
    fn a_run_of_text_answers_one_path() {
        let list = text_page()
            .display(0)
            .expect("a list")
            .expect("a text page lists");
        assert_eq!(list.len(), 2, "the paper and one run: {list:?}");
        // The page is white where it paints nothing, which is the paper the
        // raster path clears to.
        let Primitive::Rect { rect, paint } = &list[0] else {
            panic!("the paper comes first: {list:?}");
        };
        assert_eq!(paint, &Paint::Solid(Color::new(255, 255, 255, 255)));
        assert_eq!(
            (rect.x0, rect.y0, rect.x1, rect.y1),
            (0.0, 0.0, 612.0, 792.0),
            "the paper is the whole page"
        );
        let Primitive::Path {
            path,
            transform,
            fill,
            fill_rule,
            stroke,
        } = &list[1]
        else {
            panic!("a run of text is a path: {list:?}");
        };
        assert_eq!(*fill, Some(Paint::Solid(Color::new(0, 0, 0, 255))));
        assert_eq!(*fill_rule, scene::FillRule::NonZero);
        assert!(stroke.is_none(), "text is filled here");
        assert_eq!(
            *transform,
            Affine::IDENTITY,
            "the placement of every glyph is baked into the outline"
        );
        // The fixture draws at 72 by 700 points on a page of 612 by 792, with
        // the y axis of the render space pointing down.
        let bounds = kurbo::Shape::bounding_box(path);
        assert!(
            bounds.x0 > 60.0 && bounds.max_x() < 200.0,
            "the run sits at the left margin: {bounds:?}"
        );
        assert!(
            bounds.y0 > 60.0 && bounds.max_y() < 120.0,
            "the baseline lands near the top: {bounds:?}"
        );
    }

    /// A picture lands in the box the page names, not in the square of its size.
    ///
    /// A [`Primitive::Image`] maps the picture's own pixels onto the plane, so a
    /// two by one picture under a transform that names a 50 unit box covers 50
    /// units. Composing the picture's size into that transform as well — the
    /// shape a unit-square reading of the leaf suggests — squares the scale and
    /// throws a 100 by 100 picture over the whole page and far past it.
    #[test]
    fn a_picture_lands_in_the_box_the_page_names() {
        let list = picture_page()
            .display(0)
            .expect("a list")
            .expect("a page with a picture lists");
        assert_eq!(list.len(), 4, "the paper and one wrapped picture: {list:?}");
        let Some(Primitive::Image { pixels, transform }) = list
            .iter()
            .find(|primitive| matches!(primitive, Primitive::Image { .. }))
        else {
            panic!("a picture is an image leaf: {list:?}");
        };
        assert_eq!(
            (pixels.width().raw(), pixels.height().raw()),
            (2, 1),
            "the leaf carries the picture's own pixels"
        );
        // The page's y axis points up and the render space's points down, so the
        // box of 20..70 points from the bottom is 30..80 from the top.
        let bounds = transform.transform_rect_bbox(kurbo::Rect::new(
            0.0,
            0.0,
            f64::from(pixels.width().raw()),
            f64::from(pixels.height().raw()),
        ));
        let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
        assert!(
            close(bounds.x0, 10.0)
                && close(bounds.y0, 30.0)
                && close(bounds.x1, 60.0)
                && close(bounds.y1, 80.0),
            "the picture covers the box the page names: {bounds:?}"
        );
    }

    /// The walk of a page happens once, whatever asks for it.
    #[test]
    fn the_list_of_a_page_is_walked_once() {
        let surface = text_page();
        let first = surface
            .display(0)
            .expect("a list")
            .expect("a text page lists");
        let again = surface
            .display(0)
            .expect("a list")
            .expect("a text page lists");
        assert!(
            std::sync::Arc::ptr_eq(&first, &again),
            "the walk of a page happens once"
        );
        assert!(
            surface.display(1).is_err(),
            "a page out of range is an error"
        );
    }

    /// A blend mode keeps the page a raster.
    #[test]
    fn a_blend_mode_answers_no_list() {
        let surface = blended_page();
        assert!(
            surface.display(0).expect("an answer").is_none(),
            "a blend needs the backdrop, which a list has not"
        );
    }
}
