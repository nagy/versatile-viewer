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
//! and a shading or a tiling pattern (a painted function or a repeated tile).
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
    font::{Glyph, GlyphRun},
};
use kurbo::{Affine, BezPath};

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
    /// A shading or a tiling pattern, which needs a painted function or a
    /// repeated tile.
    Pattern,
}

impl Unsupported {
    /// The name a diagnostic reports.
    #[must_use]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Mask => "soft mask",
            Self::Blend => "blend mode",
            Self::Pattern => "pattern",
        }
    }
}

/// One page, walked into the primitives that draw it.
pub(crate) struct PdfScene {
    out: Vec<Primitive>,
    /// The feature that stopped the walk.
    ///
    /// The walk stops at the first one, because a page that needs it keeps the
    /// raster path whole: no later primitive of it could be drawn by the list.
    stop: Option<Unsupported>,
}

impl PdfScene {
    /// A collector that holds a page of `width` by `height`, paper included.
    pub(crate) fn new(width: f32, height: f32) -> Self {
        Self {
            out: vec![Primitive::Rect {
                rect: scene::rect_of(crate::region::crop(0.0, 0.0, width, height)),
                paint: Paint::Solid(PAPER),
            }],
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

    /// The solid paint of `paint`, or the pattern that stops the walk.
    fn solid(&mut self, paint: &PdfPaint<'_>) -> Option<Paint> {
        match paint {
            PdfPaint::Color(color) => {
                let [r, g, b, a] = color.to_rgba().to_rgba8();
                Some(Paint::Solid(Color::new(r, g, b, a)))
            }
            PdfPaint::Pattern(_) => {
                self.stop(Unsupported::Pattern);
                None
            }
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
    fn fill_run(
        &mut self,
        glyph_run: &GlyphRun<'_, '_>,
        props: &DrawProps<'_>,
        rule: FillRule,
        paint: Paint,
    ) {
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
        self.out.push(Primitive::Path {
            path: merged,
            transform: Affine::IDENTITY,
            fill: Some(paint),
            fill_rule: fill_rule(rule),
            stroke: None,
        });
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
        paint: &Paint,
    ) {
        for glyph in glyph_run.glyphs() {
            let Glyph::Outline(outline) = &**glyph else {
                continue;
            };
            let placed = glyph.transform() * outline.outline();
            if placed.elements().is_empty() {
                continue;
            }
            self.out.push(Primitive::Path {
                path: placed,
                transform: props.transform,
                fill: None,
                fill_rule: scene::FillRule::NonZero,
                stroke: Some(Stroke {
                    style: outline_style(stroke),
                    paint: paint.clone(),
                }),
            });
        }
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
        let Some(paint) = self.solid(&props.paint) else {
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
            DrawMode::Invisible => return,
        };
        self.out.push(Primitive::Path {
            path: path.clone(),
            transform: props.transform,
            fill: fill.as_ref().map(|(paint, _)| paint.clone()),
            fill_rule: fill.map_or(scene::FillRule::NonZero, |(_, rule)| rule),
            stroke,
        });
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
        let Some(paint) = self.solid(&props.paint) else {
            return;
        };
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
            DrawMode::Fill(rule) => self.fill_run(glyph_run, &props, *rule, paint),
            DrawMode::Stroke(stroke) => self.stroke_run(glyph_run, &props, stroke, &paint),
            DrawMode::FillAndStroke(rule, stroke) => {
                self.fill_run(glyph_run, &props, *rule, paint.clone());
                self.stroke_run(glyph_run, &props, stroke, &paint);
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
                        // A stencil painted by a pattern needs the pattern work
                        // the walk does not do.
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
