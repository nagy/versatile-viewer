//! The pixel-tier ladder of the image view for a vector source.
//!
//! A vector file bakes one base raster (see [`crate::svg`]), and a deep zoom
//! would stretch those pixels.
//! The image view therefore paints a larger tier when the zoom settles.
//! [`rung`] snaps the needed side up to a step, [`worth_painting`] guards a
//! swap with a hysteresis, and [`QUIET`] debounces the request past the ease.
//!
//! The module holds only the pure decision math, so `proptest` holds its
//! invariants, and the paint itself stays in the view.
//!
//! # Examples
//!
//! ```
//! use versatile_viewer::{dim::LongSide, tier};
//!
//! // A 1024 px base at 2x zoom needs 2048 px.
//! let base = LongSide::new(1024);
//! let need = tier::needed(1024.0, 2.0);
//! assert_eq!(tier::rung(need, base, tier::TIER_CAP), LongSide::new(2048));
//! // A step up of 30% pays for a paint, a step of 5% does not.
//! assert!(tier::worth_painting(
//!     LongSide::new(2048),
//!     LongSide::new(1024)
//! ));
//! assert!(!tier::worth_painting(
//!     LongSide::new(1075),
//!     LongSide::new(1024)
//! ));
//! ```

use std::time::Duration;

use crate::dim::LongSide;

/// The granularity of a rung, in pixels.
///
/// Every base side is a multiple of this step (`BASE_MIN_SIDE` 1024), so the
/// ladder stays on round numbers and a paint never repeats for a few pixels.
pub const RUNG_STEP: u32 = 256;

/// The largest raster the image view paints for a vector source.
///
/// The photo path allows [`crate::MAX_TEXTURE_SIDE`] (8192), which costs
/// 268 MB of RGBA and a slow paint on a filter-heavy document.
/// A vector tier stops at half that: 64 MB, and a paint that stays inside a
/// frame budget for most documents.
pub const TIER_CAP: LongSide = LongSide::new(4096);

/// The ratio a rung must beat before a paint is worth its cost.
///
/// Below it the GPU upscale of the current texture stays invisible, so a
/// small wheel nudge never pays for a pixmap.
pub const HYSTERESIS: f32 = 1.25;

/// How long the scale must hold still before the view requests a rung.
///
/// The ease settles in about three of its time constants, so 150 ms sits just
/// past it, and a continuous drag never queues a paint per frame.
pub const QUIET: Duration = Duration::from_millis(150);

/// The raster long side that draws a `source_long` source at `scale` window
/// pixels per source pixel.
///
/// A zero, negative or non-finite result falls back to one pixel, so the
/// ladder math never sees a nonsense value.
#[must_use]
pub fn needed(source_long: f32, scale: f32) -> f32 {
    let need = source_long * scale;
    if need.is_finite() && need > 0.0 {
        need
    } else {
        1.0
    }
}

/// Snap a needed side up to the next rung, clamped into `base..=cap`.
///
/// The function never returns a side below the base raster, so a paint never
/// throws away resolution the entry already holds.
/// A `cap` below `base` collapses onto `base`.
#[must_use]
pub fn rung(need: f32, base: LongSide, cap: LongSide) -> LongSide {
    let base = base.get().max(1);
    let cap = cap.get().max(base);
    let need = if need.is_finite() && need > 0.0 {
        need.min(cap as f32)
    } else {
        base as f32
    };
    let step = RUNG_STEP as f32;
    let stepped = ((need / step).ceil() * step) as u32;
    LongSide::new(stepped.clamp(base, cap))
}

/// Whether `next` differs enough from `current` to pay for a paint.
///
/// The same ratio guards both directions, so zooming in and zooming back out
/// share one rule.
/// A `current` of zero always wants a paint.
#[must_use]
pub fn worth_painting(next: LongSide, current: LongSide) -> bool {
    let next = next.get() as f32;
    let current = current.get() as f32;
    next > current * HYSTERESIS || next * HYSTERESIS < current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rung_lands_on_a_step_and_inside_the_range() {
        let base = LongSide::new(1024);
        assert_eq!(rung(1024.0, base, TIER_CAP), base);
        assert_eq!(rung(1025.0, base, TIER_CAP), LongSide::new(1280));
        assert_eq!(rung(2048.0, base, TIER_CAP), LongSide::new(2048));
        assert_eq!(rung(9000.0, base, TIER_CAP), TIER_CAP);
        assert_eq!(rung(10.0, base, TIER_CAP), base);
        // A cap under the base collapses onto the base.
        let low = LongSide::new(512);
        assert_eq!(rung(4096.0, base, low), base);
        // A nonsense need falls back to the base.
        assert_eq!(rung(f32::NAN, base, TIER_CAP), base);
        assert_eq!(rung(-4.0, base, TIER_CAP), base);
    }

    #[test]
    fn a_zoom_sequence_walks_the_ladder() {
        // A 100x50 document bakes at 1024x512, and a 1920x1080 window fits it
        // at 1.875 window pixels per source pixel.
        let base = LongSide::new(1024);
        let walk = |scale: f32, current: LongSide| {
            let need = needed(1024.0, scale);
            let next = rung(need, base, TIER_CAP);
            (next, worth_painting(next, current))
        };
        let (fit, paint) = walk(1.875, base);
        assert_eq!(fit, LongSide::new(2048));
        assert!(paint);
        // A rung that beats the 25% step pays for a paint, and the cap clips
        // it.
        let (deep, paint) = walk(4.0, fit);
        assert_eq!(deep, TIER_CAP);
        assert!(paint);
        // At the cap a deeper zoom asks for nothing more.
        let (_, paint) = walk(32.0, deep);
        assert!(!paint);
        // Zooming back out paints the smaller rung again.
        let (back, paint) = walk(1.0, TIER_CAP);
        assert_eq!(back, base);
        assert!(paint);
    }

    #[test]
    fn the_hysteresis_guards_both_directions() {
        assert!(worth_painting(LongSide::new(2048), LongSide::new(1024)));
        assert!(!worth_painting(LongSide::new(1279), LongSide::new(1024)));
        assert!(worth_painting(LongSide::new(1024), LongSide::new(2048)));
        assert!(!worth_painting(LongSide::new(1024), LongSide::new(1200)));
        assert!(worth_painting(LongSide::new(512), LongSide::new(0)));
        assert!(!worth_painting(LongSide::new(1024), LongSide::new(1024)));
    }
}

#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// A rung never undershoots the need unless the cap clips it, and it
        /// always stays inside the base..=cap range.
        #[test]
        fn a_rung_stays_in_range_and_never_undershoots(
            need in 0.0f32..20000.0,
            base in 1u32..2048,
            cap in 1u32..8192,
        ) {
            let base = LongSide::new(base);
            let cap = LongSide::new(cap);
            let rung = rung(need, base, cap).get();
            let floor = base.get().min(cap.get());
            let ceiling = base.get().max(cap.get());
            prop_assert!(rung >= floor);
            prop_assert!(rung <= ceiling);
            prop_assert!(rung as f32 >= need.min(ceiling as f32));
        }

        /// A larger need never produces a smaller rung, so the ladder keeps
        /// its order.
        #[test]
        fn a_rung_is_monotone(a in 0.0f32..20000.0, b in 0.0f32..20000.0) {
            let base = LongSide::new(1024);
            let (first, second) = (a.min(b), a.max(b));
            prop_assert!(rung(first, base, TIER_CAP) <= rung(second, base, TIER_CAP));
        }

        /// A rung sits on a step when the base and the cap do, so two zooms
        /// inside one step request the same paint.
        #[test]
        fn a_rung_lands_on_a_step(need in 0.0f32..20000.0) {
            let base = LongSide::new(1024);
            let rung = rung(need, base, TIER_CAP).get();
            prop_assert_eq!(rung % RUNG_STEP, 0);
        }

        /// The hysteresis rejects a paint of the same size, and it never
        /// asks for one twice.
        #[test]
        fn an_identical_rung_never_repaints(side in 1u32..8192) {
            let side = LongSide::new(side);
            prop_assert!(!worth_painting(side, side));
        }
    }
}
