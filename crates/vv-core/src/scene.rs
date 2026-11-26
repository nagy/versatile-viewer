//! The primitives a source can answer with instead of a raster.
//!
//! A source that is resolution independent builds one [`Primitive`] list per
//! plane and the renderer draws it. The list lives in surface units, so a zoom
//! or a pan changes the transform of a draw and never the list: a deep zoom
//! costs exactly nothing, where the raster path can only make its paint smaller.
//!
//! The geometry is `kurbo`'s and carries no space tag, because a primitive is
//! one shape in its own units with the [`kurbo::Affine`] that places it.
//! [`crate::dim::Rect`] and [`crate::region::Crop`] stay the tagged rects of the
//! window and of the plane, and [`rect_of`] is the one crossing.
//!
//! The pixels of an [`Primitive::Image`] leaf are a
//! [`DecodedImage`](crate::pixels::DecodedImage) and never a GPU handle: this
//! layer sits below the renderer, so the leaf carries raw bytes and the renderer
//! keeps the texture and its cache.
//!
//! # The list is in paint order
//!
//! [`Primitive::PushClip`] and [`Primitive::PushLayer`] open a scope that the
//! matching `Pop` closes, the way a display list reads in every painter: a
//! backend walks its tree once and appends, and the renderer replays. A clip and
//! a layer nest in the same sequence, so no primitive carries an index into a
//! side table.
//!
//! # The list crosses the wire
//!
//! Every type here derives `serde`, because a producer in another process
//! writes a list and a viewer reads it: the `scene_file` module of `vv-source`
//! holds the document that wraps one, and `docs/adr/0008` records the decision.
//! [`Primitive::Image`] is the one leaf that stays off the wire, because it
//! carries a [`DecodedImage`] and a picture is bytes rather than a shape; a
//! document that states one is refused by name, and a list that holds one does
//! not serialise.
//! Every struct and every struct variant denies an unknown field, so a producer
//! and a viewer agree on the whole shape of what they exchange or on nothing.
//!
//! # Examples
//!
//! ```
//! use vv_core::{draw::Color, region, scene};
//!
//! let rect = scene::rect_of(region::crop(0.0, 0.0, 10.0, 20.0));
//! assert_eq!((rect.x0, rect.y0, rect.x1, rect.y1), (0.0, 0.0, 10.0, 20.0));
//! let primitive = scene::Primitive::Rect {
//!     rect,
//!     paint: scene::Paint::Solid(Color::new(255, 0, 0, 255)),
//! };
//! assert_eq!(
//!     primitive,
//!     scene::Primitive::Rect {
//!         rect,
//!         paint: scene::Paint::Solid(Color::new(255, 0, 0, 255)),
//!     }
//! );
//! ```

use std::sync::Arc;

use kurbo::{Affine, BezPath, Rect};
use serde::{Deserialize, Serialize};

use crate::{draw::Color, pixels::DecodedImage, region};

/// Which side of a path a fill covers.
///
/// SVG carries this as `fill-rule`, and a glyph outline needs it, so the shape
/// and the rule travel together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillRule {
    /// A point inside an odd number of subpaths is inside the shape.
    NonZero,
    /// A point inside an odd number of subpaths is filled, and an even one is
    /// a hole.
    EvenOdd,
}

/// One stop of a gradient ramp.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stop {
    /// The position along the ramp, from 0 to 1.
    pub offset: f32,
    /// The color at that position, its alpha already applied.
    pub color: Color,
}

/// What a gradient does past its last stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Spread {
    /// The last color holds.
    Pad,
    /// The ramp starts again.
    Repeat,
    /// The ramp runs backwards.
    RepeatMirror,
}

/// How the pixels inside a shape are coloured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Paint {
    /// One color over the whole shape.
    Solid(Color),
    /// A ramp from `from` to `to`, both in surface units.
    Linear {
        /// The start of the ramp.
        from: (f64, f64),
        /// The end of the ramp.
        to: (f64, f64),
        /// The stops, in increasing order of offset.
        stops: Arc<[Stop]>,
        /// What the ramp does past its last stop.
        spread: Spread,
    },
    /// A ramp from an inner circle to an outer one.
    ///
    /// The circles are in the ramp's own space, which the `transform` maps onto
    /// the plane, so a radial that squashes its two circles differently keeps
    /// them circles here and puts the squash in the transform, where a renderer
    /// draws the ellipse it names.
    Radial {
        /// The center of the outer circle, in the ramp's own space.
        center: (f64, f64),
        /// The radius of the outer circle, in the ramp's own space.
        radius: f64,
        /// The center of the inner circle, which is the focus of the ramp.
        focus: (f64, f64),
        /// The radius of the inner circle, zero for a point focus.
        focus_radius: f64,
        /// Where the ramp's own space lands on the plane.
        transform: Affine,
        /// The stops, in increasing order of offset.
        stops: Arc<[Stop]>,
        /// What the ramp does past its last stop.
        spread: Spread,
    },
}

impl Paint {
    /// This paint placed by `at`.
    ///
    /// A solid colour has nowhere to go, but a ramp is geometry, so a list that
    /// moves whole — a book places one list per page — moves its ramps as well.
    /// A linear ramp's two points are in surface units, so `at` moves them; a
    /// radial keeps its circles in its own space, so `at` composes with that
    /// space and a squash the shading named stays a squash.
    #[must_use]
    pub fn placed(&self, at: Affine) -> Self {
        match self {
            Self::Solid(color) => Self::Solid(*color),
            Self::Linear {
                from,
                to,
                stops,
                spread,
            } => Self::Linear {
                from: placed_point(at, *from),
                to: placed_point(at, *to),
                stops: Arc::clone(stops),
                spread: *spread,
            },
            Self::Radial {
                center,
                radius,
                focus,
                focus_radius,
                transform,
                stops,
                spread,
            } => Self::Radial {
                center: *center,
                radius: *radius,
                focus: *focus,
                focus_radius: *focus_radius,
                transform: at * *transform,
                stops: Arc::clone(stops),
                spread: *spread,
            },
        }
    }
}

/// A surface point of a ramp, placed by `at`.
fn placed_point(at: Affine, point: (f64, f64)) -> (f64, f64) {
    let placed = at * kurbo::Point::new(point.0, point.1);
    (placed.x, placed.y)
}

/// An outline drawn around a shape.
///
/// The style is `kurbo`'s, because it already carries the width, the caps, the
/// joins, the miter limit and the dash array an SVG stroke names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stroke {
    /// The width, the caps, the joins and the dashes.
    pub style: kurbo::Stroke,
    /// The paint of the outline.
    pub paint: Paint,
}

/// One glyph of a run, and where its origin lands.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacedGlyph {
    /// The glyph id inside the font.
    pub id: u32,
    /// The glyph's origin, in surface units.
    pub x: f32,
    /// The baseline, in surface units.
    pub y: f32,
}

/// A font at one size, with the glyphs already placed.
///
/// The outlineless form of text: `usvg` outlines an SVG's own text, so a list
/// holds [`Primitive::Glyphs`] only for text this tree lays out itself (a
/// treemap label) and, later, for the glyph runs a PDF hands over.
/// The font travels as its own bytes, because this layer sits below the
/// renderer: the face index names which font of a collection to parse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlyphRun {
    /// The font file's bytes.
    pub font: Arc<[u8]>,
    /// The face inside the file, or 0 for a single font.
    pub face: u32,
    /// The size, in surface units per em.
    pub size: f32,
    /// The glyphs, in paint order.
    pub glyphs: Arc<[PlacedGlyph]>,
}

/// One drawable element of a display list, in surface units, in paint order.
///
/// Every variant but [`Self::Image`] crosses the wire, so a producer in another
/// process states a picture and a viewer paints it. A struct variant denies a
/// field it does not know, so a producer that invents one is refused rather
/// than having it dropped and painting a picture it never asked for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Primitive {
    /// A filled rectangle.
    ///
    /// The shape most documents spend their paper on, and the one a treemap
    /// needs. It carries no transform, so a producer states where it lands
    /// directly.
    Rect {
        /// The rectangle, in surface units.
        rect: Rect,
        /// The fill.
        paint: Paint,
    },
    /// A filled and stroked path.
    Path {
        /// The path, in its own units.
        path: BezPath,
        /// Where the path lands in surface units.
        transform: Affine,
        /// The fill, or `None` for an outline alone.
        fill: Option<Paint>,
        /// Which side of the path the fill covers.
        fill_rule: FillRule,
        /// The outline, or `None` for a fill alone.
        stroke: Option<Stroke>,
    },
    /// A picture placed in the plane.
    ///
    /// This is the leaf that everything else falls back to: an embedded image
    /// inside a document, one label of a treemap, and a subtree whose
    /// compositing the renderer cannot express.
    /// The leaf stays off the wire: a picture is bytes and the document that
    /// would carry them is a picture format, not a scene. A document that names
    /// this variant is refused, so a producer cannot hand over a list that
    /// half-paints.
    #[serde(skip)]
    Image {
        /// The pixels, and their extents.
        pixels: DecodedImage,
        /// Where the top-left corner lands and how the picture is scaled.
        transform: Affine,
    },
    /// A run of text outlines.
    ///
    /// The shapes come from the font rather than from a raster, so text stays
    /// sharp at any zoom and needs no picture per label.
    Glyphs {
        /// The font, the size and the placed glyphs.
        run: GlyphRun,
        /// Where the run's own origin lands in surface units.
        transform: Affine,
        /// The fill of every glyph.
        paint: Paint,
    },
    /// Open a clip for everything up to the matching [`Self::PopClip`].
    PushClip {
        /// The clip shape, in its own units.
        path: BezPath,
        /// Where the shape lands in surface units.
        transform: Affine,
        /// Which side of the shape is kept.
        fill_rule: FillRule,
    },
    /// Close the clip a [`Self::PushClip`] opened.
    PopClip,
    /// Open a transparency layer for everything up to the matching
    /// [`Self::PopLayer`].
    ///
    /// A layer is what makes a group opacity composite as one, so two
    /// overlapping shapes at half opacity never double their own alpha.
    PushLayer {
        /// The opacity of the whole layer, from 0 to 1.
        alpha: f32,
    },
    /// Close the layer a [`Self::PushLayer`] opened.
    PopLayer,
}

impl Primitive {
    /// This primitive placed by `at`, in the units of a plane of its own.
    ///
    /// A list is written for one plane, so a caller that stacks several of them
    /// — a book places one list per page — moves each one whole.
    /// A transform on the primitive takes `at` on the left, and a ramp inside a
    /// paint moves with it (see [`Paint::placed`]), because a ramp is written in
    /// the units of the plane its list was built for.
    /// A [`Self::Rect`] carries no transform of its own, so a placed rect is a
    /// path in the shape of that rect with `at` around it: the same leaf, in the
    /// one form that can be moved.
    #[must_use]
    pub fn placed(&self, at: Affine) -> Self {
        match self {
            Self::Rect { rect, paint } => Self::Path {
                path: kurbo::Shape::to_path(rect, 0.1),
                transform: at,
                fill: Some(paint.placed(at)),
                fill_rule: FillRule::NonZero,
                stroke: None,
            },
            Self::Path {
                path,
                transform,
                fill,
                fill_rule,
                stroke,
            } => Self::Path {
                path: path.clone(),
                transform: at * *transform,
                fill: fill.as_ref().map(|paint| paint.placed(at)),
                fill_rule: *fill_rule,
                stroke: stroke.as_ref().map(|outline| Stroke {
                    style: outline.style.clone(),
                    paint: outline.paint.placed(at),
                }),
            },
            Self::Image { pixels, transform } => Self::Image {
                pixels: pixels.clone(),
                transform: at * *transform,
            },
            Self::Glyphs {
                run,
                transform,
                paint,
            } => Self::Glyphs {
                run: run.clone(),
                transform: at * *transform,
                paint: paint.placed(at),
            },
            Self::PushClip {
                path,
                transform,
                fill_rule,
            } => Self::PushClip {
                path: path.clone(),
                transform: at * *transform,
                fill_rule: *fill_rule,
            },
            Self::PopClip => Self::PopClip,
            Self::PushLayer { alpha } => Self::PushLayer { alpha: *alpha },
            Self::PopLayer => Self::PopLayer,
        }
    }
}

/// `crop` as an untagged `kurbo` rect.
///
/// [`region::Crop`] carries the surface tag, and a primitive carries none, so
/// this is the one place a plane rect becomes geometry.
#[must_use]
pub fn rect_of(crop: region::Crop) -> Rect {
    Rect::new(
        f64::from(crop.min_x()),
        f64::from(crop.min_y()),
        f64::from(crop.max_x()),
        f64::from(crop.max_y()),
    )
}

/// An image leaf covering `crop`.
///
/// The transform maps the picture's pixels onto the box in surface units, so a
/// producer names the box it means and never an [`Affine`].
/// An empty box or an empty picture carries no scale, because neither has a
/// ratio to compute.
#[must_use]
pub fn image_of(crop: region::Crop, pixels: DecodedImage) -> Primitive {
    let (box_w, box_h) = (f64::from(crop.width()), f64::from(crop.height()));
    let (px_w, px_h) = (
        f64::from(pixels.width().raw()),
        f64::from(pixels.height().raw()),
    );
    let scale = if px_w > 0.0 && px_h > 0.0 {
        Affine::scale_non_uniform(box_w / px_w, box_h / px_h)
    } else {
        Affine::IDENTITY
    };
    Primitive::Image {
        pixels,
        transform: Affine::translate((f64::from(crop.min_x()), f64::from(crop.min_y()))) * scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::{ImgH, ImgW};

    /// A placed primitive carries the placement, one way or another.
    ///
    /// A list is written for one plane, and a caller that stacks several of them
    /// — a book stacks one page per slot — moves each one whole, so the shape
    /// has to land where `at` puts it and keep the paint it had.
    #[test]
    fn a_placed_primitive_moves_with_its_plane() {
        let at = Affine::scale(2.0) * Affine::translate((10.0, 20.0));
        let paint = Paint::Solid(Color::new(1, 2, 3, 255));
        let own = Affine::translate((5.0, 0.0));

        let path = Primitive::Path {
            path: BezPath::new(),
            transform: own,
            fill: Some(paint.clone()),
            fill_rule: FillRule::EvenOdd,
            stroke: None,
        };
        let Primitive::Path {
            transform,
            fill_rule,
            ..
        } = path.placed(at)
        else {
            panic!("a placed path stays a path");
        };
        assert_eq!(transform, at * own, "the placement goes on the left");
        assert_eq!(fill_rule, FillRule::EvenOdd, "and the rule rides along");

        // A rect carries no transform of its own, so a placed one is a path in
        // the shape of it, which is the leaf that can be moved.
        let rect = Primitive::Rect {
            rect: Rect::new(0.0, 0.0, 4.0, 2.0),
            paint: paint.clone(),
        };
        let Primitive::Path {
            path,
            transform,
            fill,
            ..
        } = rect.placed(at)
        else {
            panic!("a placed rect is a path");
        };
        assert_eq!(transform, at);
        assert_eq!(fill, Some(paint));
        // The path is the rect, and `at` puts it where it belongs: `at` scales
        // by two around the origin and only then moves, so the corner of the
        // rect lands at (20, 40).
        let bounds = at.transform_rect_bbox(kurbo::Shape::bounding_box(&path));
        assert_eq!(bounds, Rect::new(20.0, 40.0, 28.0, 44.0));

        // A clip keeps its shape and takes the placement.
        let clip = Primitive::PushClip {
            path: BezPath::new(),
            transform: own,
            fill_rule: FillRule::NonZero,
        };
        let Primitive::PushClip { transform, .. } = clip.placed(at) else {
            panic!("a placed clip stays a clip");
        };
        assert_eq!(transform, at * own);

        // A layer and the pops carry nothing that can move.
        assert_eq!(
            Primitive::PushLayer { alpha: 0.5 }.placed(at),
            Primitive::PushLayer { alpha: 0.5 }
        );
        assert_eq!(Primitive::PopClip.placed(at), Primitive::PopClip);
        assert_eq!(Primitive::PopLayer.placed(at), Primitive::PopLayer);
    }

    /// A ramp moves with the list that carries it.
    ///
    /// A ramp's axis and circles are geometry in the units of the plane its list
    /// was built for, so a list placed into another plane has to carry its ramps
    /// along: a book stacks pages in base units, and a ramp left in page points
    /// would paint the colour of one place over another.
    #[test]
    fn a_placed_ramp_moves_with_its_plane() {
        let at = Affine::scale(2.0) * Affine::translate((10.0, 20.0));
        let stops: Arc<[Stop]> = Arc::from(vec![Stop {
            offset: 0.0,
            color: Color::new(1, 2, 3, 255),
        }]);

        let linear = Paint::Linear {
            from: (0.0, 0.0),
            to: (10.0, 0.0),
            stops: Arc::clone(&stops),
            spread: Spread::Repeat,
        };
        let Paint::Linear {
            from,
            to,
            stops: kept,
            spread,
        } = linear.placed(at)
        else {
            panic!("a placed ramp stays a ramp");
        };
        assert_eq!(
            (from, to),
            ((20.0, 40.0), (40.0, 40.0)),
            "the axis moves as one"
        );
        assert_eq!(spread, Spread::Repeat, "and the spread rides along");
        assert_eq!(
            *kept, *stops,
            "the stops are colours, and colours move nowhere"
        );

        // A radial keeps its circles in its own space, so the placement goes
        // into that space and a squash stays a squash.
        let squash = Affine::scale_non_uniform(1.0, 0.5);
        let radial = Paint::Radial {
            center: (4.0, 0.0),
            radius: 2.0,
            focus: (0.0, 0.0),
            focus_radius: 0.0,
            transform: squash,
            stops,
            spread: Spread::Pad,
        };
        let Paint::Radial {
            center,
            radius,
            focus,
            transform,
            ..
        } = radial.placed(at)
        else {
            panic!("a placed ramp stays a ramp");
        };
        assert_eq!(
            (center, focus),
            ((4.0, 0.0), (0.0, 0.0)),
            "the circles stay where the ramp puts them"
        );
        assert_eq!(radius, 2.0);
        assert_eq!(transform, at * squash, "and the space takes the placement");
        // The two axes of that space are the ellipse the ramp draws: `at` scales
        // by two and the squash halves the y axis.
        let [xx, yx, xy, yy, _, _] = transform.as_coeffs();
        assert_eq!((xx.hypot(yx), xy.hypot(yy)), (2.0, 1.0));
    }

    #[test]
    fn a_crop_becomes_its_rect() {
        let rect = rect_of(region::crop(1.0, 2.0, 3.0, 4.0));
        assert_eq!(rect, Rect::new(1.0, 2.0, 4.0, 6.0));
    }

    #[test]
    fn a_solid_paint_compares_by_color() {
        let red = Paint::Solid(Color::new(255, 0, 0, 255));
        assert_eq!(red, red);
        assert_ne!(red, Paint::Solid(Color::new(0, 255, 0, 255)));
    }

    #[test]
    fn an_image_leaf_carries_its_own_pixels() {
        let pixels = DecodedImage::from_exact(ImgW::new(1), ImgH::new(1), vec![1, 2, 3, 4]);
        let leaf = Primitive::Image {
            pixels,
            transform: Affine::IDENTITY,
        };
        assert!(matches!(leaf, Primitive::Image { .. }));
    }

    #[test]
    fn an_image_leaf_maps_its_pixels_onto_its_box() {
        // Four pixels square into a box of twenty by ten: five across and two
        // and a half down, from the box corner.
        let pixels = DecodedImage::from_exact(ImgW::new(4), ImgH::new(4), vec![0; 64]);
        let leaf = image_of(region::crop(10.0, 20.0, 20.0, 10.0), pixels);
        let Primitive::Image { transform, .. } = leaf else {
            panic!("an image leaf");
        };
        let coeffs = transform.as_coeffs();
        assert_eq!(
            (coeffs[0], coeffs[1], coeffs[2], coeffs[3]),
            (5.0, 0.0, 0.0, 2.5)
        );
        assert_eq!((coeffs[4], coeffs[5]), (10.0, 20.0));
    }

    #[test]
    fn an_empty_image_leaf_keeps_its_corner() {
        let pixels = DecodedImage::from_exact(ImgW::new(0), ImgH::new(0), Vec::new());
        let leaf = image_of(region::crop(3.0, 4.0, 0.0, 0.0), pixels);
        let Primitive::Image { transform, .. } = leaf else {
            panic!("an image leaf");
        };
        assert_eq!(transform, Affine::translate((3.0, 4.0)));
    }

    #[test]
    fn a_glyph_run_carries_its_font_and_its_places() {
        let run = GlyphRun {
            font: Arc::from(vec![0_u8, 1, 2]),
            face: 1,
            size: 11.0,
            glyphs: Arc::from(vec![
                PlacedGlyph {
                    id: 3,
                    x: 10.0,
                    y: 20.0,
                },
                PlacedGlyph {
                    id: 4,
                    x: 17.0,
                    y: 20.0,
                },
            ]),
        };
        let primitive = Primitive::Glyphs {
            run,
            transform: Affine::IDENTITY,
            paint: Paint::Solid(Color::new(255, 255, 255, 255)),
        };
        let Primitive::Glyphs { run, .. } = &primitive else {
            panic!("a glyph run");
        };
        assert_eq!(run.face, 1);
        assert_eq!(run.glyphs.len(), 2);
        assert_eq!(run.glyphs[1].x, 17.0);
    }

    #[test]
    fn a_gradient_carries_its_stops_and_its_spread() {
        let stops: Arc<[Stop]> = Arc::from(vec![
            Stop {
                offset: 0.0,
                color: Color::new(0, 0, 0, 255),
            },
            Stop {
                offset: 1.0,
                color: Color::new(255, 255, 255, 255),
            },
        ]);
        let paint = Paint::Linear {
            from: (0.0, 0.0),
            to: (10.0, 0.0),
            stops: Arc::clone(&stops),
            spread: Spread::Repeat,
        };
        let Paint::Linear {
            stops: read,
            spread,
            ..
        } = &paint
        else {
            panic!("a linear ramp");
        };
        assert_eq!(read.len(), 2);
        assert_eq!(*spread, Spread::Repeat);
    }
}
