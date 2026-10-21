//! Newtypes that keep pixel and window dimensions from swapping.
//!
//! [`ImgW`] and [`ImgH`] name the two axes of an image.
//! [`WinW`] and [`WinH`] name the two axes of the window.
//! [`LongSide`] wraps a pixel limit.
//! [`PreviewPx`] wraps a progressive-preview pixel limit.
//! [`Delay`] and [`Rate`] wrap the key-repeat timing.
//! [`RampTime`] wraps the ramp time of a held-key motion.
//! Each wrapper is a newtype over its primitive and costs nothing at runtime.

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
            pub const fn get(self) -> u32 {
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
            pub const fn get(self) -> f32 {
                self.0
            }
        }
    };
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
