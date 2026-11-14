// Pixel and coordinate math lives in f32 (window and source units) and
// indices in usize and u32.
// The casts between them belong to that boundary.
// Every value here (window pixels, texture dimensions) stays far below
// the f32 exact-integer range.
// The pedantic cast lints are therefore noise.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    // win_w/win_h and friends are natural paired names in window math.
    clippy::similar_names
)]

//! The geometry and motion core of versatile-viewer.
//!
//! This layer holds no decoder, no GPU handle and no window, so every crate
//! above it reads one vocabulary of dimensions, rectangles and easing.
//! [`dim`] wraps every raw pixel and window primitive in a newtype, so no call
//! site casts a dimension by hand.
//! [`draw`] holds the point, the color and the quad the view code writes.
//! [`region`] holds the visible-region geometry of the image view, and
//! [`region_sel`] the selection the `v` hold drags over it.
//! [`rotate`] holds the rotation angle behind the `r` hold.
//! [`keyrepeat`] repeats a held key at the X rate, and [`tween`] ramps one for
//! a continuous motion.
//!
//! The constants below are the shared policy of both views, so no view tunes
//! a speed or a dimness of its own.

pub mod dim;
pub mod draw;
pub mod keyrepeat;
pub mod region;
pub mod region_sel;
pub mod rotate;
pub mod tween;

use crate::draw::Color;

/// The speed multiplier while Shift is held.
///
/// Shift + h/j/k/l moves at double speed: the grid selection step, the image
/// pan and the image free zoom.
/// Both views read this one constant, so the two speeds never drift apart.
pub const SHIFT_SPEED_MULTIPLIER: u32 = 2;

/// The brightness a dimmed area keeps, as a 0..=255 channel value.
///
/// The grid dims every cell outside a visual selection, and the image view
/// dims the picture outside a region selection.
/// Both read this one level, so the two views never show a different
/// darkness.
/// [`DIM_TINT`] multiplies a dimmed texture by it, and [`DIM_VEIL`] blends the
/// same share away, so one number drives both.
pub const DIM_LEVEL: u8 = 90;

/// The pointer travel, in window pixels, that still counts as a click.
///
/// The `f` pan hold and the `v` region drag both read it, so a shaky hand
/// never turns a tap into a drag.
/// Both keys measure the travel in window pixels, never in source units,
/// because a deep zoom makes one window pixel a wide source span.
pub const TAP_SLACK: f32 = 4.0;

/// The tint of a dimmed draw: a texture multiplied by [`DIM_LEVEL`].
pub const DIM_TINT: Color = Color::new(DIM_LEVEL, DIM_LEVEL, DIM_LEVEL, 255);

/// The black veil over a dimmed part of the image view.
///
/// Alpha `255 - DIM_LEVEL` leaves the share of brightness that [`DIM_TINT`]
/// keeps, so a veiled pixel darkens exactly like a tinted one.
pub const DIM_VEIL: Color = Color::new(0, 0, 0, 255 - DIM_LEVEL);

/// True when `VV_TIMING` is set.
///
/// The code then prints the GPU stack at startup (adapter, format, present
/// mode, texture limit) and every slow frame to stderr.
/// An upload runs on a decode worker, so this flag stays off that path.
/// Input tracing stays under `VV_DEBUG`.
#[must_use]
pub fn timing_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VV_TIMING").is_some())
}
