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

use crate::{draw::Color, pixels::DecodedImage, region};

/// Which side of a path a fill covers.
///
/// SVG carries this as `fill-rule`, and a glyph outline needs it, so the shape
/// and the rule travel together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillRule {
    /// A point inside an odd number of subpaths is inside the shape.
    NonZero,
    /// A point inside an odd number of subpaths is filled, and an even one is
    /// a hole.
    EvenOdd,
}

/// One stop of a gradient ramp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stop {
    /// The position along the ramp, from 0 to 1.
    pub offset: f32,
    /// The color at that position, its alpha already applied.
    pub color: Color,
}

/// What a gradient does past its last stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spread {
    /// The last color holds.
    Pad,
    /// The ramp starts again.
    Repeat,
    /// The ramp runs backwards.
    RepeatMirror,
}

/// How the pixels inside a shape are coloured.
#[derive(Debug, Clone, PartialEq)]
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
    /// A ramp from an inner circle to an outer one, both in surface units.
    Radial {
        /// The center of the outer circle.
        center: (f64, f64),
        /// The radius of the outer circle.
        radius: f64,
        /// The center of the inner circle, which is the focus of the ramp.
        focus: (f64, f64),
        /// The radius of the inner circle, zero for a point focus.
        focus_radius: f64,
        /// The stops, in increasing order of offset.
        stops: Arc<[Stop]>,
        /// What the ramp does past its last stop.
        spread: Spread,
    },
}

/// An outline drawn around a shape.
///
/// The style is `kurbo`'s, because it already carries the width, the caps, the
/// joins, the miter limit and the dash array an SVG stroke names.
#[derive(Debug, Clone, PartialEq)]
pub struct Stroke {
    /// The width, the caps, the joins and the dashes.
    pub style: kurbo::Stroke,
    /// The paint of the outline.
    pub paint: Paint,
}

/// One drawable element of a display list, in surface units, in paint order.
#[derive(Debug, Clone, PartialEq)]
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
    Image {
        /// The pixels, and their extents.
        pixels: DecodedImage,
        /// Where the top-left corner lands and how the picture is scaled.
        transform: Affine,
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
