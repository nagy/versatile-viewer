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
//! assert_eq!(primitive, scene::Primitive::Rect {
//!     rect,
//!     paint: scene::Paint::Solid(Color::new(255, 0, 0, 255)),
//! });
//! ```

use kurbo::{Affine, BezPath, Rect};

use crate::{draw::Color, pixels::DecodedImage, region};

/// How the pixels inside a primitive are coloured.
///
/// One variant today, because every source that answers a list so far paints a
/// solid. Gradients arrive with the PDF and SVG backends, and they land here as
/// further arms rather than as a color field on every variant, so a shape never
/// grows a second spelling of its paint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Paint {
    /// One color over the whole shape.
    Solid(Color),
}

/// One drawable element of a display list, in surface units.
#[derive(Debug, Clone, PartialEq)]
pub enum Primitive {
    /// A filled rectangle.
    ///
    /// The shape most documents spend their paper on, and the only one a
    /// treemap needs.
    Rect {
        /// The rectangle, in surface units.
        rect: Rect,
        /// The fill.
        paint: Paint,
    },
    /// A filled path.
    ///
    /// A glyph set arrives as one of these until the text path lands, and a
    /// vector file arrives as a sequence of them.
    Path {
        /// The path, in surface units.
        path: BezPath,
        /// The fill.
        paint: Paint,
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
}
