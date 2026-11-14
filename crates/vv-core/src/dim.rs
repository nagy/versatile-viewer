//! Newtypes that keep pixel and window dimensions from swapping.
//!
//! [`ImgW`] and [`ImgH`] name the two axes of an image.
//! [`WinW`] and [`WinH`] name the two axes of the window.
//! [`LongSide`] wraps a pixel limit.
//! [`PreviewPx`] wraps a progressive-preview pixel limit.
//! [`Rate`] wraps the key-repeat rate.
//! Each wrapper is a newtype over its primitive and costs nothing at runtime.
//!
//! [`Rect`] is `euclid`'s rectangle tagged with [`WindowSpace`], and
//! [`SurfaceSpace`] tags the rectangle of a plane, so the compiler refuses to
//! pass one where the other belongs.
//! [`rect`], [`whole`], [`center_square`], [`center_origin`] and
//! [`contains_point`] are the five readings the draw and the view make of a
//! `Rect`, and `euclid` answers the edges and the size.
//!
//! [`center_origin`] keeps a turned draw in place.

use euclid::{point2, size2};

use crate::draw::Vec2;

macro_rules! uint_newtype {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        pub struct $name(u32);

        impl $name {
            /// Wrap a raw value.
            #[must_use]
            pub const fn new(value: u32) -> Self {
                Self(value)
            }

            /// The wrapped raw value.
            #[must_use]
            pub const fn raw(self) -> u32 {
                self.0
            }

            /// The wrapped value as `f32`.
            #[must_use]
            pub fn as_f32(self) -> f32 {
                self.0 as f32
            }
        }
    };
}

macro_rules! float_newtype {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
        pub struct $name(f32);

        impl $name {
            /// Wrap a raw value.
            #[must_use]
            pub const fn new(value: f32) -> Self {
                Self(value)
            }

            /// The wrapped raw value.
            #[must_use]
            pub const fn raw(self) -> f32 {
                self.0
            }
        }
    };
}

/// The unit tag of a length or a position in window pixels.
///
/// `euclid` refuses to mix two space tags in one expression, so a window
/// rectangle never lands where a rectangle of a plane belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowSpace {}

/// The unit tag of a length or a position in surface units.
///
/// A surface unit is the plane's own: base raster pixels for an image or a
/// vector file, and PostScript points for a document page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceSpace {}

/// An axis-aligned rectangle in window pixels.
///
/// The origin gives the top-left corner, and `y` grows down.
/// `euclid` names the edges `min_x`/`max_x` and `min_y`/`max_y` and the extents
/// `width`/`height`, so a rect carries no second set of names here.
///
/// # Examples
///
/// ```
/// # use vv_core::dim::{self, Rect};
/// let r = dim::rect(10.0, 20.0, 30.0, 40.0);
/// assert_eq!((r.max_x(), r.max_y()), (40.0, 60.0));
/// // The square sits centered inside the rectangle.
/// assert_eq!(dim::center_square(r), dim::rect(10.0, 25.0, 30.0, 30.0));
/// assert!(r.contains(euclid::point2(25.0, 40.0)));
/// assert!(!r.contains(euclid::point2(41.0, 40.0)));
/// // The origin that keeps a turned rect on its own dest position.
/// assert_eq!(dim::center_origin(r), vv_core::draw::Vec2::new(15.0, 20.0));
/// ```
pub type Rect = euclid::Rect<f32, WindowSpace>;

/// A window rectangle from its top-left corner and its size.
#[must_use]
pub const fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
    Rect::new(point2(x, y), size2(width, height))
}

/// A window rectangle of `width` by `height` at the origin.
/// This is the source rect of a whole texture.
#[must_use]
pub const fn whole(width: f32, height: f32) -> Rect {
    rect(0.0, 0.0, width, height)
}

/// The origin that keeps a turned rect on its own `dest` position.
///
/// The draw turns the quad around the rect position, and this origin pulls
/// the rect back by half its size.
/// A position at the image center therefore turns the image in place, and
/// the same call draws the plain rect at zero degrees.
#[must_use]
pub fn center_origin(outer: Rect) -> Vec2 {
    Vec2::new(outer.width() / 2.0, outer.height() / 2.0)
}

/// Is the point inside `outer`, edges included?
///
/// `euclid`'s `contains` puts the far edge outside, while a press on the border
/// of a grid cell or of a region must land in it, so the viewer reads its own
/// test.
/// The function takes either space tag, so a window rect and a crop share the
/// one rule.
#[must_use]
pub fn contains_point<Unit>(outer: &euclid::Rect<f32, Unit>, x: f32, y: f32) -> bool {
    x >= outer.min_x() && y >= outer.min_y() && x <= outer.max_x() && y <= outer.max_y()
}

/// The largest square centered inside `outer`.
#[must_use]
pub fn center_square(outer: Rect) -> Rect {
    let side = if outer.width() < outer.height() {
        outer.width()
    } else {
        outer.height()
    };
    rect(
        outer.min_x() + (outer.width() - side) / 2.0,
        outer.min_y() + (outer.height() - side) / 2.0,
        side,
        side,
    )
}

uint_newtype!(ImgW, "An image width, in pixels.");
uint_newtype!(ImgH, "An image height, in pixels.");
uint_newtype!(LongSide, "The longest side of an image, in pixels.");
uint_newtype!(
    PreviewPx,
    "The longest side of a progressive preview, in pixels."
);
float_newtype!(WinW, "A window width, in pixels.");
float_newtype!(WinH, "A window height, in pixels.");
float_newtype!(Rate, "An auto-repeat rate, in repeats per second.");
