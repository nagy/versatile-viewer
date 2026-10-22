//! Hold-driven velocity ramps for continuous motion (pan and zoom).
//!
//! A key press is an edge.
//! A held key is a state.
//! This module reacts to the state only.
//! It ignores every press event, and the auto-repeat rate with it.
//!
//! Stepping a continuous motion on the auto-repeat rate makes that motion
//! pulse ([`crate::keyrepeat`] does this for the discrete moves).
//! This module turns a held key into a cruise velocity instead.
//!
//! The velocity ramps in when the key goes down, holds at cruise while the
//! key stays down, and ramps out when the key releases.
//! The ramp is linear: the returned weight is the linear progress itself.
//! Motion therefore starts and ends softly, and a long hold keeps a constant
//! speed.
//! A linear ramp gives an even speed change.
//! A curved ramp would hold the speed near zero too long and then jump.
//! That reads as a step, not as a ramp.
//!
//! The caller multiplies its cruise speed by the weight.
//! Opposite keys get one [`Ramp`] each, so reversing direction eases out of
//! the old motion and into the new one.

use crate::dim::RampTime;

/// One linear velocity ramp for one held key.
/// Feed it once per frame with [`Ramp::tick`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Ramp {
    /// Progress toward the cruise speed, 0.0 to 1.0.
    level: f32,
}

impl Ramp {
    /// Advance one frame and return this frame's speed weight, 0.0 to 1.0.
    ///
    /// `held`: the key is down now.
    /// `dt`: the frame time, in seconds.
    /// `ramp`: the time a full ease-in or ease-out takes.
    ///
    /// The weight is the progress itself, so the speed changes at an even
    /// rate in and out.
    /// The function starts from the level of the last frame, so a
    /// release halfway into the ramp-out continues from there.
    /// # Examples
    ///
    /// ```
    /// # use versatile_viewer::dim::RampTime;
    /// # use versatile_viewer::tween::Ramp;
    /// let mut r = Ramp::default();
    /// let ramp = RampTime::new(0.2);
    /// // The first frame of a fresh press barely moves.
    /// assert!(r.tick(true, 0.02, ramp) < 0.11);
    /// // Half the ramp time puts the weight at a half.
    /// assert!((r.tick(true, 0.08, ramp) - 0.5).abs() < 1e-6);
    /// // The hold reaches full speed and stays there.
    /// assert!((r.tick(true, 0.4, ramp) - 1.0).abs() < 1e-6);
    /// assert!((r.tick(true, 0.1, ramp) - 1.0).abs() < 1e-6);
    /// // The release eases out over the same ramp time.
    /// assert!((r.tick(false, 0.1, ramp) - 0.5).abs() < 1e-6);
    /// assert!(r.tick(false, 0.2, ramp) < 1e-6);
    #[must_use]
    pub fn tick(&mut self, held: bool, dt: f32, ramp: RampTime) -> f32 {
        let span = ramp.get();
        let step = if span > 0.0 { dt.max(0.0) / span } else { 1.0 };
        self.level = if held {
            (self.level + step).min(1.0)
        } else {
            (self.level - step).max(0.0)
        };
        self.level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The weight rises on a straight line, holds, then falls back.
    #[test]
    fn ramp_eases_in_and_out() {
        let mut r = Ramp::default();
        let ramp = RampTime::new(0.2);
        let w1 = r.tick(true, 0.05, ramp);
        let w2 = r.tick(true, 0.05, ramp);
        let w3 = r.tick(true, 0.05, ramp);
        assert!((w1 - 0.25).abs() < 1e-6);
        assert!((w2 - 0.5).abs() < 1e-6);
        assert!((w3 - 0.75).abs() < 1e-6);
        assert!((r.tick(true, 0.05, ramp) - 1.0).abs() < 1e-6);
        assert!((r.tick(true, 5.0, ramp) - 1.0).abs() < 1e-6); // clamped
        assert!((r.tick(false, 0.1, ramp) - 0.5).abs() < 1e-6);
        assert!(r.tick(false, 5.0, ramp) <= 0.0);
    }

    /// A release before the ease-in ends continues from the current level.
    #[test]
    fn ramp_reverses_from_the_current_level() {
        let mut r = Ramp::default();
        let ramp = RampTime::new(0.2);
        assert!((r.tick(true, 0.1, ramp) - 0.5).abs() < 1e-6);
        // Level 0.5 falls to 0.375 over 25 ms.
        assert!((r.tick(false, 0.025, ramp) - 0.375).abs() < 1e-6);
    }

    /// A zero ramp time switches the motion on and off at once.
    #[test]
    fn zero_ramp_is_a_switch() {
        let mut r = Ramp::default();
        let ramp = RampTime::new(0.0);
        assert!((r.tick(true, 0.016, ramp) - 1.0).abs() < 1e-6);
        assert!(r.tick(false, 0.016, ramp) <= 0.0);
    }

    /// An idle ramp stays at rest while the key is up.
    #[test]
    fn idle_ramp_stays_at_rest() {
        let mut r = Ramp::default();
        let ramp = RampTime::new(0.18);
        assert!(r.tick(false, 0.016, ramp) <= 0.0);
        assert!(r.tick(false, 0.016, ramp) <= 0.0);
    }

    /// The weight never depends on how many frames the hold spans, only on
    /// the summed time. A steady frame rate therefore gives a steady speed,
    /// and a frame hiccup does not restart the ramp.
    #[test]
    fn ramp_is_frame_rate_independent() {
        let ramp = RampTime::new(0.2);
        let mut slow = Ramp::default();
        for _ in 0..20 {
            let _ = slow.tick(true, 0.01, ramp);
        }
        let mut fast = Ramp::default();
        for _ in 0..40 {
            let _ = fast.tick(true, 0.005, ramp);
        }
        assert!((slow.tick(true, 0.0, ramp) - fast.tick(true, 0.0, ramp)).abs() < 1e-6);
    }

    proptest::proptest! {
        /// The weight stays in `0.0..=1.0` for any hold pattern and any
        /// frame time.
        #[test]
        fn ramp_stays_in_range(
            held in proptest::collection::vec(proptest::bool::ANY, 1..40),
            dt in 0.0f32..0.2,
            span in 0.0f32..1.0,
        ) {
            let ramp = RampTime::new(span);
            let mut r = Ramp::default();
            for &down in &held {
                let w = r.tick(down, dt, ramp);
                proptest::prop_assert!((0.0..=1.0).contains(&w), "weight {w}");
            }
        }

        /// A zero ramp time reaches the cruise speed in one frame, and drops
        /// to rest in one frame.
        #[test]
        fn ramp_with_zero_span_is_a_switch(dt in 0.0f32..0.5) {
            let ramp = RampTime::new(0.0);
            let mut r = Ramp::default();
            proptest::prop_assert!((r.tick(true, dt, ramp) - 1.0).abs() < 1e-6);
            proptest::prop_assert!(r.tick(false, dt, ramp) < 1e-6);
        }
    }
}
