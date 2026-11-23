//! The primitives of a parsed vector tree.
//!
//! [`crate::svg`] keeps a `usvg` tree, and this module turns it into the
//! [`Primitive`] list the renderer draws at any zoom.
//! The walk mirrors the one `resvg` makes: a group contributes its transform,
//! its clip and its opacity, and a path, an image or a run of text outlines is a
//! leaf.
//!
//! # What a list cannot express
//!
//! A list is one flat sequence with a clip stack and a layer stack, so a
//! document that needs something else answers [`Unsupported`] and the caller
//! keeps the raster path for it. That covers a filter, a mask, a blend mode and
//! a tiling pattern, which all need an intermediate buffer resvg knows how to
//! composite and this list does not describe.
//!
//! # Where a leaf lands
//!
//! `usvg` folds a node's own `transform` into its geometry or its size, and keeps
//! the *parent* chain in `abs_transform`, which is exactly the value `resvg`
//! threads down as it walks. A leaf therefore takes `abs_transform` and no
//! transform is passed through this walk.

use std::sync::Arc;

use usvg::{
    Node,
    tiny_skia_path::{Path as SkPath, PathSegment, Transform},
};

use crate::{
    DecodedImage,
    dim::{ImgH, ImgW},
    draw::Color,
    scene::{FillRule, Paint, Primitive, Spread, Stop, Stroke},
};

/// What a document needs that a primitive list cannot express.
///
/// A caller that reads one of these answers no list, so the document keeps the
/// raster path and its picture stays exactly what it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unsupported {
    /// A `filter` element, which needs an intermediate buffer.
    Filter,
    /// A `mask` element, which needs an intermediate buffer and a luminance
    /// pass.
    Mask,
    /// A blend mode other than `normal`, which needs the backdrop.
    Blend,
    /// A tiling `pattern`, which needs a rendered tile.
    Pattern,
    /// A clip path this walk cannot describe as one shape.
    Clip,
}

impl Unsupported {
    /// The name a diagnostic reports.
    #[must_use]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Filter => "filter",
            Self::Mask => "mask",
            Self::Blend => "blend mode",
            Self::Pattern => "pattern",
            Self::Clip => "clip path",
        }
    }
}

/// The primitives of `tree`, in paint order, placed through `space`.
///
/// `space` maps the document's declared units onto the plane the surface
/// reports, which is a base raster of at least [`super::svg`]'s shortest side:
/// a 16-unit icon draws in a 1024-pixel plane, so every leaf takes the factor
/// the same way [`crate::svg::Vector::render_region`] does.
///
/// # Errors
///
/// Returns [`Unsupported`] when the document holds a node this list cannot
/// express.
pub(crate) fn primitives(
    tree: &usvg::Tree,
    space: kurbo::Affine,
) -> Result<Vec<Primitive>, Unsupported> {
    let mut out = Vec::new();
    for node in tree.root().children() {
        node_of(node, space, &mut out)?;
    }
    Ok(out)
}

/// Append the primitives of one node.
fn node_of(node: &Node, space: kurbo::Affine, out: &mut Vec<Primitive>) -> Result<(), Unsupported> {
    match node {
        Node::Group(group) => group_of(group, space, out),
        // A text node is its own outline group, so a glyph run needs no font
        // here: `usvg` laid the outlines out while it parsed the document.
        Node::Text(text) => group_of(text.flattened(), space, out),
        Node::Image(image) => image_of(image, space, out),
        Node::Path(path) => path_of(path, space, out),
    }
}

/// Append the primitives of a group, its clip and its layer included.
fn group_of(
    group: &usvg::Group,
    space: kurbo::Affine,
    out: &mut Vec<Primitive>,
) -> Result<(), Unsupported> {
    if !group.filters().is_empty() {
        return Err(Unsupported::Filter);
    }
    if group.mask().is_some() {
        return Err(Unsupported::Mask);
    }
    if group.blend_mode() != usvg::BlendMode::Normal {
        return Err(Unsupported::Blend);
    }
    let mut clips = 0;
    let mut clip = group.clip_path();
    while let Some(path) = clip {
        clip = path.clip_path();
        for child in path.root().children() {
            match child {
                Node::Path(shape) if shape.is_visible() => {
                    out.push(Primitive::PushClip {
                        path: bez_path(shape.data()),
                        transform: space * affine(shape.abs_transform()),
                        fill_rule: vello_fill_rule(
                            shape
                                .fill()
                                .map_or(usvg::FillRule::NonZero, usvg::Fill::rule),
                        ),
                    });
                    clips += 1;
                }
                // A clip made of a group needs the group rendered on its own
                // buffer first, which is what `resvg` does and a list cannot.
                Node::Group(_) | Node::Path(_) | Node::Image(_) | Node::Text(_) => {
                    return Err(Unsupported::Clip);
                }
            }
        }
    }
    let alpha = group.opacity().get();
    let layered = alpha < 1.0;
    if layered {
        out.push(Primitive::PushLayer { alpha });
    }
    for child in group.children() {
        node_of(child, space, out)?;
    }
    if layered {
        out.push(Primitive::PopLayer);
    }
    for _ in 0..clips {
        out.push(Primitive::PopClip);
    }
    Ok(())
}

/// Append the primitive of one path, when it draws anything.
fn path_of(
    path: &usvg::Path,
    space: kurbo::Affine,
    out: &mut Vec<Primitive>,
) -> Result<(), Unsupported> {
    if !path.is_visible() {
        return Ok(());
    }
    let transform = space * affine(path.abs_transform());
    let fill = match path.fill() {
        Some(fill) => Some(paint(fill.paint(), fill.opacity(), transform)?),
        None => None,
    };
    let fill_rule = path
        .fill()
        .map_or(FillRule::NonZero, |fill| match fill.rule() {
            usvg::FillRule::NonZero => FillRule::NonZero,
            usvg::FillRule::EvenOdd => FillRule::EvenOdd,
        });
    let stroke = match path.stroke() {
        Some(stroke) => Some(stroke_of(stroke, transform)?),
        None => None,
    };
    if fill.is_none() && stroke.is_none() {
        return Ok(());
    }
    out.push(Primitive::Path {
        path: bez_path(path.data()),
        transform,
        fill,
        fill_rule,
        stroke,
    });
    Ok(())
}

/// Append the primitive of one embedded image, when its bytes decode.
fn image_of(
    image: &usvg::Image,
    space: kurbo::Affine,
    out: &mut Vec<Primitive>,
) -> Result<(), Unsupported> {
    if !image.is_visible() {
        return Ok(());
    }
    let bytes = match image.kind() {
        usvg::ImageKind::PNG(data)
        | usvg::ImageKind::JPEG(data)
        | usvg::ImageKind::GIF(data)
        | usvg::ImageKind::WEBP(data) => data,
        // A nested document is not a picture but a tree, so it is walked as one:
        // its own root paints inside the box this `<image>` names, and `usvg`
        // folded that box into `image_ts` (which maps the nested viewport onto
        // the box) before it handed the tree over.
        // A tree with a feature this list cannot express answers the same error
        // from the walk below, and then the whole document keeps its raster.
        usvg::ImageKind::SVG(tree) => {
            let space = space * affine(image.abs_transform());
            for node in tree.root().children() {
                node_of(node, space, out)?;
            }
            return Ok(());
        }
    };
    let Ok(decoded) = image::load_from_memory(bytes) else {
        // A picture that does not decode draws nothing, which is better than
        // falling back to a raster of the whole document for one broken byte.
        tracing::warn!(bytes = bytes.len(), "svg image");
        return Ok(());
    };
    let rgba = decoded.to_rgba8();
    let (px_w, px_h) = rgba.dimensions();
    if px_w == 0 || px_h == 0 {
        return Ok(());
    }
    let pixels = DecodedImage::from_exact(ImgW::new(px_w), ImgH::new(px_h), rgba.into_raw());
    // The image's own box is `size`, so the pixels scale onto it.
    let size = image.size();
    let scale = kurbo::Affine::scale_non_uniform(
        f64::from(size.width()) / f64::from(px_w),
        f64::from(size.height()) / f64::from(px_h),
    );
    out.push(Primitive::Image {
        pixels,
        transform: space * affine(image.abs_transform()) * scale,
    });
    Ok(())
}

/// The stroke of a path, its paint in surface units.
fn stroke_of(stroke: &usvg::Stroke, transform: kurbo::Affine) -> Result<Stroke, Unsupported> {
    let cap = |cap: usvg::LineCap| match cap {
        usvg::LineCap::Butt => kurbo::Cap::Butt,
        usvg::LineCap::Round => kurbo::Cap::Round,
        usvg::LineCap::Square => kurbo::Cap::Square,
    };
    let style = kurbo::Stroke {
        width: f64::from(stroke.width().get()),
        join: match stroke.linejoin() {
            usvg::LineJoin::Miter | usvg::LineJoin::MiterClip => kurbo::Join::Miter,
            usvg::LineJoin::Round => kurbo::Join::Round,
            usvg::LineJoin::Bevel => kurbo::Join::Bevel,
        },
        miter_limit: f64::from(stroke.miterlimit().get()),
        start_cap: cap(stroke.linecap()),
        end_cap: cap(stroke.linecap()),
        dash_pattern: stroke
            .dasharray()
            .map(|dashes| dashes.iter().map(|dash| f64::from(*dash)).collect())
            .unwrap_or_default(),
        dash_offset: f64::from(stroke.dashoffset()),
    };
    Ok(Stroke {
        style,
        paint: paint(stroke.paint(), stroke.opacity(), transform)?,
    })
}

/// The paint of a shape, in surface units.
///
/// `transform` maps the shape's own space onto the plane, so it places a ramp's
/// axis as well: the renderer draws a ramp whose geometry it reads in surface
/// units.
fn paint(
    paint: &usvg::Paint,
    opacity: usvg::Opacity,
    transform: kurbo::Affine,
) -> Result<Paint, Unsupported> {
    let alpha = opacity.get();
    match paint {
        usvg::Paint::Color(color) => Ok(Paint::Solid(dimmed(
            Color::new(color.red, color.green, color.blue, 255),
            alpha,
        ))),
        usvg::Paint::LinearGradient(gradient) => {
            let space = transform * affine(gradient.transform());
            Ok(Paint::Linear {
                from: apply(space, gradient.x1(), gradient.y1()),
                to: apply(space, gradient.x2(), gradient.y2()),
                stops: stops(gradient.stops(), alpha),
                spread: spread(gradient.spread_method()),
            })
        }
        usvg::Paint::RadialGradient(gradient) => {
            // The circles stay circles in the gradient's own space, which
            // `gradientTransform` maps onto the plane: a squash becomes an
            // ellipse the renderer draws, not two radii averaged into a round
            // ramp nobody drew.
            let space = transform * affine(gradient.transform());
            Ok(Paint::Radial {
                center: (f64::from(gradient.cx()), f64::from(gradient.cy())),
                radius: f64::from(gradient.r().get()),
                focus: (f64::from(gradient.fx()), f64::from(gradient.fy())),
                focus_radius: f64::from(gradient.fr().get()),
                transform: space,
                stops: stops(gradient.stops(), alpha),
                spread: spread(gradient.spread_method()),
            })
        }
        usvg::Paint::Pattern(_) => Err(Unsupported::Pattern),
    }
}

/// The stops of a ramp, each color at `alpha`.
fn stops(stops: &[usvg::Stop], alpha: f32) -> Arc<[Stop]> {
    stops
        .iter()
        .map(|stop| {
            let color = stop.color();
            Stop {
                offset: stop.offset().get(),
                color: dimmed(
                    Color::new(color.red, color.green, color.blue, 255),
                    alpha * stop.opacity().get(),
                ),
            }
        })
        .collect()
}

/// What a ramp does past its last stop.
const fn spread(spread: usvg::SpreadMethod) -> Spread {
    match spread {
        usvg::SpreadMethod::Pad => Spread::Pad,
        usvg::SpreadMethod::Reflect => Spread::RepeatMirror,
        usvg::SpreadMethod::Repeat => Spread::Repeat,
    }
}

/// `color` at `factor` of its alpha.
fn dimmed(color: Color, factor: f32) -> Color {
    let alpha = (f32::from(color.a) * factor).round().clamp(0.0, 255.0) as u8;
    Color::new(color.r, color.g, color.b, alpha)
}

/// A `usvg` transform as `kurbo`'s.
fn affine(transform: Transform) -> kurbo::Affine {
    kurbo::Affine::new([
        f64::from(transform.sx),
        f64::from(transform.ky),
        f64::from(transform.kx),
        f64::from(transform.sy),
        f64::from(transform.tx),
        f64::from(transform.ty),
    ])
}

/// `space` applied to one point.
fn apply(space: kurbo::Affine, x: f32, y: f32) -> (f64, f64) {
    let point = space * kurbo::Point::new(f64::from(x), f64::from(y));
    (point.x, point.y)
}

/// A `tiny-skia` path as `kurbo`'s.
fn bez_path(path: &SkPath) -> kurbo::BezPath {
    let mut out = kurbo::BezPath::new();
    for segment in path.segments() {
        match segment {
            PathSegment::MoveTo(p) => out.move_to(point(p)),
            PathSegment::LineTo(p) => out.line_to(point(p)),
            PathSegment::QuadTo(a, b) => out.quad_to(point(a), point(b)),
            PathSegment::CubicTo(a, b, c) => out.curve_to(point(a), point(b), point(c)),
            PathSegment::Close => out.close_path(),
        }
    }
    out
}

/// One `tiny-skia` point as `kurbo`'s.
fn point(point: usvg::tiny_skia_path::Point) -> kurbo::Point {
    kurbo::Point::new(f64::from(point.x), f64::from(point.y))
}

/// The fill rule a shape uses when the caller has one already.
const fn vello_fill_rule(rule: usvg::FillRule) -> FillRule {
    match rule {
        usvg::FillRule::NonZero => FillRule::NonZero,
        usvg::FillRule::EvenOdd => FillRule::EvenOdd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The primitives of `document`, or the feature that stops them.
    fn walk(document: &str) -> Result<Vec<Primitive>, Unsupported> {
        let vector = crate::svg::from_str(document).expect("the document parses");
        primitives(vector.tree(), kurbo::Affine::IDENTITY)
    }

    #[test]
    fn a_filled_rect_is_one_path() {
        let list = walk(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                 <rect width="10" height="10" fill="#ff0000"/>
               </svg>"##,
        )
        .expect("a plain document lists");
        assert_eq!(list.len(), 1);
        let Primitive::Path { fill, stroke, .. } = &list[0] else {
            panic!("a path: {list:?}");
        };
        assert_eq!(*fill, Some(Paint::Solid(Color::new(255, 0, 0, 255))));
        assert!(stroke.is_none());
    }

    #[test]
    fn a_stroke_carries_its_width_and_its_paint() {
        let list = walk(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                 <path d="M0 0 L10 10" fill="none" stroke="#00ff00" stroke-width="3"/>
               </svg>"##,
        )
        .expect("a stroke lists");
        let Primitive::Path { fill, stroke, .. } = &list[0] else {
            panic!("a path");
        };
        assert!(fill.is_none());
        let stroke = stroke.as_ref().expect("an outline");
        assert_eq!(stroke.style.width, 3.0);
        assert_eq!(stroke.paint, Paint::Solid(Color::new(0, 255, 0, 255)));
    }

    /// A squashed `gradientTransform` keeps its circles and takes the squash.
    ///
    /// A radial gradient states one circle and a transform, and the transform may
    /// squeeze it into an ellipse. A ramp of this list draws every ellipse it
    /// names, so the circle stays the gradient's own and the squeeze rides in the
    /// ramp's space instead of being averaged into a round ramp nobody drew.
    #[test]
    fn a_squashed_radial_gradient_keeps_its_circles() {
        let list = walk(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10">
                  <defs>
                    <radialGradient id="g" gradientTransform="scale(2 1)">
                      <stop offset="0" stop-color="#000"/>
                      <stop offset="1" stop-color="#fff"/>
                    </radialGradient>
                  </defs>
                  <rect width="20" height="10" fill="url(#g)"/>
                </svg>"##,
        )
        .expect("a ramp lists");
        let Primitive::Path { fill, .. } = &list[0] else {
            panic!("a path");
        };
        let Some(Paint::Radial {
            center,
            radius,
            transform,
            ..
        }) = fill
        else {
            panic!("a radial ramp: {fill:?}");
        };
        // `objectBoundingBox` units put a unit circle in the middle of the rect,
        // and `gradientTransform` works in the box's space after that: the circle
        // spans the rect's twenty units across and ten down, and the scale
        // doubles the wider axis.
        assert!(
            (center.0 - 0.5).abs() < 1e-3 && (center.1 - 0.5).abs() < 1e-3,
            "{center:?}"
        );
        assert!((radius - 0.5).abs() < 1e-3, "{radius}");
        let [xx, yx, xy, yy, _, _] = transform.as_coeffs();
        assert!(
            (xx.hypot(yx) - 40.0).abs() < 1e-3 && (xy.hypot(yy) - 10.0).abs() < 1e-3,
            "the space is the ellipse's own axes: {transform:?}"
        );
    }

    #[test]
    fn a_gradient_becomes_a_ramp_in_surface_units() {
        let list = walk(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                  <defs>
                    <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0" stop-color="#000"/>
                      <stop offset="1" stop-color="#fff"/>
                    </linearGradient>
                  </defs>
                  <rect width="10" height="10" fill="url(#g)"/>
                </svg>"##,
        )
        .expect("a ramp lists");
        let Primitive::Path { fill, .. } = &list[0] else {
            panic!("a path");
        };
        let Some(Paint::Linear {
            from,
            to,
            stops,
            spread,
        }) = fill
        else {
            panic!("a linear ramp: {fill:?}");
        };
        // `objectBoundingBox` units resolve against the rect, so the axis spans
        // the shape.
        assert!((from.0 - 0.0).abs() < 1e-3, "{from:?}");
        assert!((to.0 - 10.0).abs() < 1e-3, "{to:?}");
        assert_eq!(stops.len(), 2);
        assert_eq!(*spread, Spread::Pad);
    }

    #[test]
    fn a_group_opacity_opens_a_layer() {
        let list = walk(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                 <g opacity="0.5"><rect width="10" height="10" fill="#fff"/></g>
               </svg>"##,
        )
        .expect("a group lists");
        assert!(matches!(list[0], Primitive::PushLayer { alpha } if (alpha - 0.5).abs() < 1e-6));
        assert!(matches!(list[1], Primitive::Path { .. }));
        assert!(matches!(list[2], Primitive::PopLayer));
    }

    #[test]
    fn a_clip_path_opens_a_clip() {
        let list = walk(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                 <defs><clipPath id="c"><rect width="5" height="5"/></clipPath></defs>
                 <g clip-path="url(#c)"><rect width="10" height="10" fill="#fff"/></g>
               </svg>"##,
        )
        .expect("a clip lists");
        assert!(matches!(list[0], Primitive::PushClip { .. }));
        assert!(matches!(list[1], Primitive::Path { .. }));
        assert!(matches!(list[2], Primitive::PopClip));
    }

    /// A `data:` URI of `bytes`, percent-encoded.
    ///
    /// A picture reaches the parser through a data URI, and percent-encoding
    /// keeps the bytes readable in this file instead of a base64 blob.
    fn data_uri(mime: &str, bytes: &[u8]) -> String {
        use std::fmt::Write as _;
        let mut uri = format!("data:{mime},");
        for byte in bytes {
            write!(uri, "%{byte:02X}").expect("a string write");
        }
        uri
    }

    /// A red picture of two by two, encoded here so no test reads a fixture.
    fn picture(encode: fn(&mut Vec<u8>)) -> Vec<u8> {
        let mut bytes = Vec::new();
        encode(&mut bytes);
        bytes
    }

    fn gif(bytes: &mut Vec<u8>) {
        // The encoder trait is in scope for `encode`, which the GIF encoder
        // names directly.
        let mut encoder = image::codecs::gif::GifEncoder::new(&mut *bytes);
        encoder
            .encode(
                &[255, 0, 0, 255].repeat(4),
                2,
                2,
                image::ExtendedColorType::Rgba8,
            )
            .expect("a gif");
    }

    fn webp(bytes: &mut Vec<u8>) {
        use image::ImageEncoder as _;
        image::codecs::webp::WebPEncoder::new_lossless(&mut *bytes)
            .write_image(
                &[255, 0, 0, 255].repeat(4),
                2,
                2,
                image::ExtendedColorType::Rgba8,
            )
            .expect("a webp");
    }

    /// A document that holds one picture of `mime` in a box of 10 by 10.
    fn with_picture(mime: &str, bytes: &[u8]) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                 <image href="{}" width="10" height="10"/>
               </svg>"#,
            data_uri(mime, bytes)
        )
    }

    /// The picture of a one-leaf list, and the box it fills.
    fn only_picture(list: &[Primitive]) -> (&DecodedImage, kurbo::Affine) {
        assert_eq!(list.len(), 1, "one picture: {list:?}");
        let Primitive::Image { pixels, transform } = &list[0] else {
            panic!("a picture: {list:?}");
        };
        (pixels, *transform)
    }

    /// A GIF is a picture like a PNG, both to the decoder and to the list.
    #[test]
    fn an_embedded_gif_becomes_a_picture() {
        let list = walk(&with_picture("image/gif", &picture(gif))).expect("a picture lists");
        let (pixels, transform) = only_picture(&list);
        assert_eq!((pixels.width().raw(), pixels.height().raw()), (2, 2));
        assert_eq!(&pixels.as_rgba()[..4], [255, 0, 0, 255]);
        // The box is 10 by 10 declared units and the picture is 2 by 2 pixels,
        // so the pixels scale onto it.
        let corner = transform * kurbo::Point::new(2.0, 2.0);
        assert!(
            (corner.x - 10.0).abs() < 1e-3 && (corner.y - 10.0).abs() < 1e-3,
            "{corner:?}"
        );
    }

    /// And so is a WebP, in both of its encodings.
    #[test]
    fn an_embedded_webp_becomes_a_picture() {
        let list = walk(&with_picture("image/webp", &picture(webp))).expect("a picture lists");
        let (pixels, _) = only_picture(&list);
        assert_eq!(&pixels.as_rgba()[..4], [255, 0, 0, 255]);
    }

    /// A nested document is a tree, so it walks into the box its `<image>` names.
    #[test]
    fn a_nested_document_walks_into_its_box() {
        let inner = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                         <rect width="10" height="10" fill="#0000ff"/>
                       </svg>"##;
        let document = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10">
                 <image x="10" y="0" width="10" height="10" href="{}"/>
               </svg>"#,
            data_uri("image/svg+xml", inner.as_bytes())
        );
        let list = walk(&document).expect("a nested document lists");
        assert_eq!(list.len(), 1, "the nested rect and nothing else: {list:?}");
        let Primitive::Path {
            transform, fill, ..
        } = &list[0]
        else {
            panic!("the nested rect: {list:?}");
        };
        assert_eq!(*fill, Some(Paint::Solid(Color::new(0, 0, 255, 255))));
        // The nested viewport fills the box, so its own corners land on the
        // box's corners in the units of the document that holds it.
        let corner = |x: f64, y: f64| *transform * kurbo::Point::new(x, y);
        let (near, far) = (corner(0.0, 0.0), corner(10.0, 10.0));
        assert!(
            (near.x - 10.0).abs() < 1e-3 && near.y.abs() < 1e-3,
            "{near:?}"
        );
        assert!(
            (far.x - 20.0).abs() < 1e-3 && (far.y - 10.0).abs() < 1e-3,
            "{far:?}"
        );
    }

    /// A feature the list cannot express keeps the whole document a raster, even
    /// when it sits one document down.
    #[test]
    fn a_nested_document_with_a_filter_answers_unsupported() {
        let inner = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                         <defs><filter id="f"><feGaussianBlur stdDeviation="1"/></filter></defs>
                         <rect width="10" height="10" fill="#fff" filter="url(#f)"/>
                       </svg>"##;
        let document = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                 <image width="10" height="10" href="{}"/>
               </svg>"#,
            data_uri("image/svg+xml", inner.as_bytes())
        );
        assert_eq!(walk(&document), Err(Unsupported::Filter));
    }

    #[test]
    fn a_filter_answers_unsupported() {
        let answer = walk(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                 <defs><filter id="f"><feGaussianBlur stdDeviation="1"/></filter></defs>
                 <rect width="10" height="10" fill="#fff" filter="url(#f)"/>
               </svg>"##,
        );
        assert_eq!(answer, Err(Unsupported::Filter));
    }

    #[test]
    fn a_mask_answers_unsupported() {
        let answer = walk(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                 <defs><mask id="m"><rect width="5" height="5" fill="#fff"/></mask></defs>
                 <rect width="10" height="10" fill="#fff" mask="url(#m)"/>
               </svg>"##,
        );
        assert_eq!(answer, Err(Unsupported::Mask));
    }

    #[test]
    fn a_pattern_answers_unsupported() {
        let answer = walk(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                  <defs>
                    <pattern id="p" width="2" height="2" patternUnits="userSpaceOnUse">
                      <rect width="1" height="1" fill="#000"/>
                    </pattern>
                  </defs>
                  <rect width="10" height="10" fill="url(#p)"/>
                </svg>"##,
        );
        assert_eq!(answer, Err(Unsupported::Pattern));
    }
}
