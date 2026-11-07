//! Newtypes that keep pixel and window dimensions from swapping.
//!
//! [`ImgW`] and [`ImgH`] name the two axes of an image.
//! [`WinW`] and [`WinH`] name the two axes of the window.
//! [`LongSide`] wraps a pixel limit.
//! [`PreviewPx`] wraps a progressive-preview pixel limit.
//! [`Delay`] and [`Rate`] wrap the key-repeat timing.
//! [`RampTime`] wraps the ramp time of a held-key motion.
//! [`Rect`] wraps an axis-aligned rectangle in window pixels.
//! Each wrapper is a newtype over its primitive and costs nothing at runtime.
//!
//! [`Rect::center_origin`] keeps a turned draw in place.

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

/// An axis-aligned rectangle in window pixels.
///
/// `x` and `y` give the top-left corner, and `y` grows down.
///
/// # Examples
///
/// ```
/// # use versatile_viewer::draw::Vec2;
/// # use versatile_viewer::dim::Rect;
/// let r = Rect::new(10.0, 20.0, 30.0, 40.0);
/// assert_eq!((r.right(), r.bottom()), (40.0, 60.0));
/// // The square sits centered inside the rectangle.
/// assert_eq!(r.center_square(), Rect::new(10.0, 25.0, 30.0, 30.0));
/// assert!(r.contains(25.0, 40.0));
/// assert!(!r.contains(41.0, 40.0));
/// // The origin that keeps a turned rect on its own dest position.
/// assert_eq!(r.center_origin(), Vec2::new(15.0, 20.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rect {
    /// The left edge.
    pub x: f32,
    /// The top edge.
    pub y: f32,
    /// The width, rightward from `x`.
    pub width: f32,
    /// The height, downward from `y`.
    pub height: f32,
}

impl Rect {
    /// A rectangle from its top-left corner and its size.
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// A rectangle of `width` x `height` at the origin.
    /// This is the source rect of a whole texture.
    #[must_use]
    pub const fn sized(width: f32, height: f32) -> Self {
        Self::new(0.0, 0.0, width, height)
    }

    /// The right edge, `x + width`.
    #[must_use]
    pub const fn right(self) -> f32 {
        self.x + self.width
    }

    /// The bottom edge, `y + height`.
    #[must_use]
    pub const fn bottom(self) -> f32 {
        self.y + self.height
    }

    /// Is the point inside the rectangle, edges included?
    #[must_use]
    pub const fn contains(self, px: f32, py: f32) -> bool {
        px >= self.x && py >= self.y && px <= self.right() && py <= self.bottom()
    }

    /// The origin that keeps a turned rect on its own `dest` position.
    ///
    /// The draw turns the quad around the rect position, and this origin pulls
    /// the rect back by half its size.
    /// A position at the image center therefore turns the image in place, and
    /// the same call draws the plain rect at zero degrees.
    #[must_use]
    pub fn center_origin(self) -> Vec2 {
        Vec2::new(self.width / 2.0, self.height / 2.0)
    }

    /// The largest square centered inside the rectangle.
    #[must_use]
    pub const fn center_square(self) -> Self {
        let side = if self.width < self.height {
            self.width
        } else {
            self.height
        };
        Self::new(
            self.x + (self.width - side) / 2.0,
            self.y + (self.height - side) / 2.0,
            side,
            side,
        )
    }
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
float_newtype!(Delay, "An auto-repeat initial delay, in seconds.");
float_newtype!(Rate, "An auto-repeat rate, in repeats per second.");
float_newtype!(
    RampTime,
    "The time an ease-in or ease-out of a held-key motion ramp takes, in seconds."
);
