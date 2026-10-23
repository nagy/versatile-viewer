//! Image rotation angles: the `r` hold, the quarter-turn snap and the
//! inverse turn of a mouse point.
//!
//! The angle is a free value in degrees.
//! A held `r` adds clockwise degrees as the mouse travels down, and
//! subtracts them as it travels up.
//! A release keeps the angle, so the image stays turned.
//!
//! The drawn angle stays free while the `r` hold runs, so a drag turns
//! smoothly.
//! A release snaps the angle to the nearest quarter turn when it lands
//! within [`SNAP_DEGREES`] of one, so a near right angle ends exactly.
//! The snap runs on the release only, never during the drag.
//! A positive angle turns clockwise on screen, which matches raylib
//! (`DrawTexturePro` grows the angle clockwise with y down).
//!
//! A plain `r` hold turns the image around its own center.
//! A Shift+R hold turns it around the mouse point that the press showed, and
//! [`pivot`] moves the pan so that point keeps its window spot.
//! The pivot stays a draw transform: no pixel moves.

use raylib::prelude::Vector2;

/// The largest distance, in degrees, that the stored angle may sit from the
/// nearest quarter turn before a release stops snapping it.
pub const SNAP_DEGREES: f32 = 2.0;

/// The degrees of turn for one frame of vertical mouse travel.
///
/// `delta_y` is the frame travel in window pixels, and down is positive.
/// `span` is the travel, in window pixels, for one full turn.
/// A span of zero turns nothing.
///
/// # Examples
///
/// ```
/// # use versatile_viewer::rotate::turn_deg;
/// // One span of downward travel is one full clockwise turn.
/// assert!((turn_deg(200.0, 200.0) - 360.0).abs() < 1e-4);
/// // Upward travel turns the other way.
/// assert!((turn_deg(-100.0, 200.0) + 180.0).abs() < 1e-4);
/// assert!(turn_deg(50.0, 0.0) == 0.0);
/// ```
#[must_use]
pub fn turn_deg(delta_y: f32, span: f32) -> f32 {
    if span <= 0.0 || !delta_y.is_finite() {
        0.0
    } else {
        delta_y / span * 360.0
    }
}

/// Wrap an angle into `[0, 360)`.
///
/// # Examples
///
/// ```
/// # use versatile_viewer::rotate::normalize_deg;
/// assert_eq!(normalize_deg(0.0), 0.0);
/// assert!((normalize_deg(-90.0) - 270.0).abs() < 1e-4);
/// assert!((normalize_deg(720.0)).abs() < 1e-4);
/// ```
#[must_use]
pub fn normalize_deg(deg: f32) -> f32 {
    if deg.is_finite() {
        deg.rem_euclid(360.0)
    } else {
        0.0
    }
}

/// Add one frame of mouse travel to the angle the image shows now.
///
/// The result is normalized into `[0, 360)`, so a hold adds to the current
/// angle instead of replacing it.
/// A second hold therefore continues from where the first one left off.
///
/// # Examples
///
/// ```
/// # use versatile_viewer::rotate::apply_turn;
/// // Half a span of downward travel is a half turn.
/// assert!((apply_turn(0.0, 100.0, 200.0) - 180.0).abs() < 1e-4);
/// // Upward travel turns the other way.
/// assert!((apply_turn(180.0, -100.0, 200.0) - 0.0).abs() < 1e-4);
/// // A turn adds to the current angle, so a second hold continues from it.
/// assert!((apply_turn(90.0, 50.0, 200.0) - 180.0).abs() < 1e-4);
/// // The result stays inside one turn, so a full turn comes back to zero.
/// assert!(apply_turn(300.0, 100.0, 200.0) < 360.0);
/// assert!(apply_turn(0.0, 200.0, 200.0) < 1e-4);
/// ```
#[must_use]
pub fn apply_turn(angle_deg: f32, delta_y: f32, span: f32) -> f32 {
    normalize_deg(normalize_deg(angle_deg) + turn_deg(delta_y, span))
}

/// Snap an angle to the nearest quarter turn when it sits within
/// [`SNAP_DEGREES`] of one.
///
/// The code calls this once, when the hold ends, so a drag turns freely and
/// the image lands on the quarter turn only on release.
/// The result is normalized into `[0, 360)`.
///
/// # Examples
///
/// ```
/// # use versatile_viewer::rotate::snap_deg;
/// assert_eq!(snap_deg(90.0), 90.0);
/// assert_eq!(snap_deg(89.0), 90.0);
/// assert_eq!(snap_deg(91.0), 90.0);
/// assert_eq!(snap_deg(87.0), 87.0);
/// // A full turn snaps back to the upright angle.
/// assert_eq!(snap_deg(359.0), 0.0);
/// ```
#[must_use]
pub fn snap_deg(deg: f32) -> f32 {
    let nearest = (deg / 90.0).round() * 90.0;
    if (deg - nearest).abs() <= SNAP_DEGREES {
        normalize_deg(nearest)
    } else {
        normalize_deg(deg)
    }
}

/// The pan that turns the image around `pivot` instead of its center.
///
/// `pivot` is the window point that must keep the image point it shows while
/// the angle runs from `from_deg` to `to_deg`.
/// `center` is the window middle with no pan, so `pan` shifts the draw.
///
/// The draw puts the image point that sits on the image middle at
/// `center + pan`, and a turn by `deg` runs every image point around that
/// origin.
/// A plain turn therefore sweeps the image point under the pivot off its
/// window spot, and the returned pan pulls the drawn image back by that
/// travel.
///
/// # Examples
///
/// ```
/// # use raylib::prelude::Vector2;
/// # use versatile_viewer::rotate::pivot;
/// let c = Vector2::new(50.0, 50.0);
/// let pan = Vector2::new(10.0, -10.0);
/// // A zero turn keeps the pan, whatever the pivot.
/// let same = pivot(Vector2::new(60.0, 50.0), c, 42.0, 42.0, pan);
/// assert!((same.x - pan.x).abs() < 1e-3 && (same.y - pan.y).abs() < 1e-3);
/// // A pivot left of the middle pulls the pan left as the turn grows.
/// let next = pivot(Vector2::new(30.0, 50.0), c, 0.0, 90.0, Vector2::ZERO);
/// assert!((next.x + 20.0).abs() < 1e-3 && (next.y - 20.0).abs() < 1e-3);
/// ```
#[must_use]
pub fn pivot(pivot: Vector2, center: Vector2, from_deg: f32, to_deg: f32, pan: Vector2) -> Vector2 {
    let (sin, cos) = (to_deg - from_deg).to_radians().sin_cos();
    // The offset from the drawn center to the pivot, in window pixels.
    let off = pivot - center - pan;
    // A plain turn sweeps that offset around, so the pan pulls it back.
    let swept = Vector2::new(off.x * cos - off.y * sin, off.x * sin + off.y * cos);
    pan + off - swept
}

/// Turn a window point around a draw origin, exactly like the draw does.
///
/// The code checks the pivot with this: the image point under the pivot keeps
/// its window spot at every angle.
///
/// # Examples
///
/// ```
/// # use raylib::prelude::Vector2;
/// # use versatile_viewer::rotate::turn_point;
/// let p = turn_point(Vector2::new(10.0, 0.0), Vector2::ZERO, 90.0);
/// assert!(p.x.abs() < 1e-3 && (p.y - 10.0).abs() < 1e-3);
/// ```
#[must_use]
pub fn turn_point(p: Vector2, origin: Vector2, deg: f32) -> Vector2 {
    let (sin, cos) = deg.to_radians().sin_cos();
    let (dx, dy) = (p.x - origin.x, p.y - origin.y);
    Vector2::new(
        origin.x + dx * cos - dy * sin,
        origin.y + dx * sin + dy * cos,
    )
}

/// Turn a window point back into the unrotated image space.
///
/// `center` is the window point that the image turns around, and `deg` is the
/// drawn angle.
/// The code uses this to hit-test a turned image against its plain
/// axis-aligned rectangle.
///
/// A pivot shifts the pan, so the draw and the hit test both take the center
/// as the window middle plus the pan, and the pivot changes neither.
///
/// # Examples
/// ```
/// # use raylib::prelude::Vector2;
/// # use versatile_viewer::rotate::unrotate_point;
/// let c = Vector2::new(50.0, 50.0);
/// // A quarter turn leaves a point on the vertical axis on the horizontal
/// // axis.
/// let p = unrotate_point(Vector2::new(60.0, 50.0), c, 90.0);
/// assert!((p.x - 50.0).abs() < 1e-3 && (p.y - 40.0).abs() < 1e-3);
/// ```
#[must_use]
pub fn unrotate_point(p: Vector2, center: Vector2, deg: f32) -> Vector2 {
    let (sin, cos) = deg.to_radians().sin_cos();
    let (dx, dy) = (p.x - center.x, p.y - center.y);
    // The inverse of the clockwise turn that the draw applies.
    Vector2::new(
        center.x + dx * cos + dy * sin,
        center.y - dx * sin + dy * cos,
    )
}

#[cfg(test)]
mod tests {
    use proptest::prop_assert;
    use raylib::prelude::Vector2;

    use super::{
        SNAP_DEGREES, apply_turn, normalize_deg, pivot, snap_deg, turn_deg, turn_point,
        unrotate_point,
    };

    #[test]
    fn apply_turn_continues_from_the_current_angle() {
        // A second hold adds to the angle the image shows, never resets it.
        let once = apply_turn(0.0, 25.0, 100.0);
        let twice = apply_turn(once, 25.0, 100.0);
        assert!((twice - 180.0).abs() < 1e-4, "got {twice}");
        // Releasing keeps the angle, and a hold that travels nowhere keeps it
        // too.
        assert!((apply_turn(twice, 0.0, 100.0) - twice).abs() < 1e-4);
    }

    #[test]
    fn snap_keeps_the_far_angles() {
        // Only the zone around each quarter turn snaps.
        assert!((snap_deg(45.0) - 45.0).abs() < 1e-4);
        assert!((snap_deg(90.0 + SNAP_DEGREES + 1.0) - 93.0).abs() < 1e-4);
        // The zone spans SNAP_DEGREES on both sides of a quarter turn.
        assert!((snap_deg(90.0 - SNAP_DEGREES) - 90.0).abs() < 1e-4);
        assert!((snap_deg(90.0 - SNAP_DEGREES - 0.5) - 87.5).abs() < 1e-4);
    }

    #[test]
    fn unrotate_turns_backwards() {
        // A clockwise drawn turn comes back as a counter-clockwise one.
        let c = Vector2::new(0.0, 0.0);
        let p = unrotate_point(Vector2::new(10.0, 0.0), c, 90.0);
        assert!(p.x.abs() < 1e-3 && (p.y + 10.0).abs() < 1e-3);
        // A round trip over four quarter turns comes back to the start.
        let mut q = Vector2::new(7.0, 3.0);
        for _ in 0..4 {
            q = unrotate_point(q, c, 90.0);
        }
        assert!((q.x - 7.0).abs() < 1e-3 && (q.y - 3.0).abs() < 1e-3);
    }

    #[test]
    fn pivot_holds_the_image_point() {
        // A pixel at the upright offset `u` from the image middle lands at
        // `center + pan + R(deg) u`. The pivot pan must keep the pixel under
        // the pivot at the pivot window point.
        let c = Vector2::new(100.0, 80.0);
        let pan = Vector2::new(12.0, -7.0);
        let p = Vector2::new(140.0, 55.0);
        for (from, to) in [(0.0, 90.0), (30.0, 210.0), (350.0, 10.0), (0.0, 180.0)] {
            let u = unrotate_point(p, c + pan, from) - (c + pan);
            let next = pivot(p, c, from, to, pan);
            let before = c + pan + turn_point(u, Vector2::ZERO, from);
            let after = c + next + turn_point(u, Vector2::ZERO, to);
            assert!(
                (before.x - after.x).abs() < 1e-2 && (before.y - after.y).abs() < 1e-2,
                "{from}->{to}: {before:?} against {after:?}"
            );
        }
        // The image middle as the pivot is the plain centered turn.
        let still = pivot(c + pan, c, 0.0, 137.0, pan);
        assert!((still.x - pan.x).abs() < 1e-3 && (still.y - pan.y).abs() < 1e-3);
    }

    proptest::proptest! {
        /// The normalized angle stays in range, and the snap never lands far
        /// from the input.
        #[test]
        fn snap_stays_inside_the_zone(deg in -10000.0f32..10000.0) {
            let n = normalize_deg(deg);
            prop_assert!((0.0..360.0).contains(&n));
            let s = snap_deg(deg);
            prop_assert!((0.0..360.0).contains(&s));
            // Compare around the ring, so 359 against 0 measures 1 degree.
            let diff = (n - s).abs();
            prop_assert!(diff <= SNAP_DEGREES + 1e-3 || (360.0 - diff) <= SNAP_DEGREES + 1e-3);
        }

        /// A pivot keeps the image point under it, whatever the angle.
        #[test]
        fn pivot_holds_any_image_point(
            px in -5000.0f32..5000.0,
            py in -5000.0f32..5000.0,
            from in -10000.0f32..10000.0,
            delta in -10000.0f32..10000.0,
            panx in -2000.0f32..2000.0,
            pany in -2000.0f32..2000.0,
        ) {
            let c = Vector2::new(640.0, 480.0);
            let p = Vector2::new(px, py);
            let pan = Vector2::new(panx, pany);
            let to = from + delta;
            let u = unrotate_point(p, c + pan, from) - (c + pan);
            let next = pivot(p, c, from, to, pan);
            let before = c + pan + turn_point(u, Vector2::ZERO, from);
            let after = c + next + turn_point(u, Vector2::ZERO, to);
            // The angles reach thousands of degrees and the offsets thousands
            // of pixels, so a f32 radian loses a fraction of a pixel.
            let slack = 0.001 * (1.0 + u.length());
            prop_assert!((before.x - p.x).abs() < slack);
            prop_assert!((before.y - p.y).abs() < slack);
            prop_assert!((before.x - after.x).abs() < slack);
            prop_assert!((before.y - after.y).abs() < slack);
        }

        /// One span of travel is one full turn, in either direction.
        #[test]
        fn one_span_is_one_turn(span in 1.0f32..4000.0) {
            prop_assert!((turn_deg(span, span) - 360.0).abs() < 1e-2);
            prop_assert!((turn_deg(-span, span) + 360.0).abs() < 1e-2);
        }
    }
}
