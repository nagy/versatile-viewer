//! The image-view region selection, the `v` hold.
//!
//! A press of `v` anchors a rect in the units of [`crate::region`], the
//! pointer drags the opposite corner, and the release keeps the rect.
//! The pointer stays free, so the drag reads the real cursor and the region
//! stops at the window edge.
//! A press and release without travel clears the kept rect, the way an empty
//! `v` click does in Vim, and ESC drops it too.
//!
//! The rect lives in source units, so a pan, a zoom and a turn all keep it on
//! the picture: a [`crate::region::Frame`] maps its corners to the window,
//! exactly like the draw maps the picture, and the same frame maps the free
//! cursor back, so the rect can never disagree with the pointer.
//!
//! [`around`] answers the four quads of the picture outside the rect, so the
//! view darkens the rest of the picture with one veil, and [`corners`] draws
//! the bright border of the rect itself.

use raylib::prelude::Vector2;

use crate::{TAP_SLACK, region::Crop};

/// A region drag in flight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drag {
    /// The corner the press landed on, in source units.
    pub anchor: Vector2,
    /// The same corner on screen, in window pixels, for the click test.
    pub grab: Vector2,
    /// True when a Shift at the press edge keeps both sides equal.
    pub square: bool,
}

impl Drag {
    /// Start a drag at `anchor`, which the caller already clamped into the
    /// plane.
    #[must_use]
    pub fn new(anchor: Vector2, grab: Vector2, square: bool) -> Self {
        Self {
            anchor,
            grab,
            square,
        }
    }

    /// The rect the drag spans now, clamped to the plane.
    ///
    /// `corner` is the free pointer in source units, from
    /// [`crate::region::Frame::to_image`], so it shares the units of `anchor`
    /// and of `img`.
    /// A square drag takes its side from the longer travel axis.
    /// The side then shrinks to the room the plane leaves past the anchor, so
    /// the region never leaves the picture and never stops being square.
    #[must_use]
    pub fn rect(&self, corner: Vector2, img: Vector2) -> Crop {
        let anchor = clamp(self.anchor, img);
        let corner = clamp(corner, img);
        if !self.square {
            return span(anchor, corner);
        }
        let dx = corner.x - anchor.x;
        let dy = corner.y - anchor.y;
        let sx = if dx < 0.0 { -1.0 } else { 1.0 };
        let sy = if dy < 0.0 { -1.0 } else { 1.0 };
        let room_x = if sx > 0.0 { img.x - anchor.x } else { anchor.x };
        let room_y = if sy > 0.0 { img.y - anchor.y } else { anchor.y };
        let side = dx.abs().max(dy.abs()).min(room_x).min(room_y).max(0.0);
        span(
            anchor,
            Vector2::new(anchor.x + side * sx, anchor.y + side * sy),
        )
    }

    /// Whether the pointer stayed close enough to the anchor to count as a
    /// click, which clears the kept region instead of committing a new one.
    #[must_use]
    pub fn tapped(&self, cursor: Vector2) -> bool {
        (cursor - self.grab).length() <= TAP_SLACK
    }
}

/// Clamp a source point into the plane.
#[must_use]
pub fn clamp(point: Vector2, img: Vector2) -> Vector2 {
    Vector2::new(
        point.x.clamp(0.0, img.x.max(0.0)),
        point.y.clamp(0.0, img.y.max(0.0)),
    )
}

/// The four corners of a rect, clockwise from the top-left.
#[must_use]
pub fn corners(rect: Crop) -> [Vector2; 4] {
    [
        Vector2::new(rect.x, rect.y),
        Vector2::new(rect.right(), rect.y),
        Vector2::new(rect.right(), rect.bottom()),
        Vector2::new(rect.x, rect.bottom()),
    ]
}

/// The four quads of `outer` that lie outside `inner`, clockwise.
///
/// Both rects keep the axes of the plane, so every quad is a trapezoid: the
/// band of `outer` past one side of `inner`.
/// Each quad comes out as `[top-left, top-right, bottom-right, bottom-left]`
/// of the band, so the draw has to reverse it for raylib's back-face culling
/// (see `draw_selection` in `main.rs`).
/// An empty `inner` leaves the bands with the area of the whole `outer`, and a
/// side that `inner` touches leaves its band with no area, so one function
/// covers every veil.
/// `inner` clamps into `outer`, so a drag that ran past the picture still
/// answers bands that stay on it.
#[must_use]
pub fn around(outer: Crop, inner: Crop) -> [[Vector2; 4]; 4] {
    let o = corners(outer);
    let i = corners(inside(inner, outer));
    [
        [o[0], o[1], i[1], i[0]],
        [o[1], o[2], i[2], i[1]],
        [o[2], o[3], i[3], i[2]],
        [o[3], o[0], i[0], i[3]],
    ]
}

/// Split a band quad into the two triangles the draw sends.
///
/// raylib culls a back face and reads its front face as counter-clockwise in
/// its own sense, which is a negative shoelace on a screen whose y grows
/// down.
/// A band arrives as `[top-left, top-right, bottom-right, bottom-left]`
/// (clockwise there), so it goes out reversed.
/// The other order draws nothing at all.
#[must_use]
pub fn veil_triangles(quad: [Vector2; 4]) -> [[Vector2; 3]; 2] {
    [[quad[0], quad[3], quad[2]], [quad[0], quad[2], quad[1]]]
}

/// The rect that two opposite corners span, in any drag direction.
fn span(a: Vector2, b: Vector2) -> Crop {
    Crop {
        x: a.x.min(b.x),
        y: a.y.min(b.y),
        w: (b.x - a.x).abs(),
        h: (b.y - a.y).abs(),
    }
}

/// Clip a rect into another one, so the veil never reaches past the picture.
fn inside(rect: Crop, outer: Crop) -> Crop {
    let x = rect.x.clamp(outer.x, outer.right());
    let y = rect.y.clamp(outer.y, outer.bottom());
    Crop {
        x,
        y,
        w: rect.w.min(outer.right() - x).max(0.0),
        h: rect.h.min(outer.bottom() - y).max(0.0),
    }
}

/// The area of a quad, by the shoelace formula.
#[cfg(test)]
fn quad_area(quad: &[Vector2; 4]) -> f32 {
    let mut sum = 0.0;
    for i in 0..4 {
        let a = quad[i];
        let b = quad[(i + 1) % 4];
        sum += a.x * b.y - b.x * a.y;
    }
    (sum / 2.0).abs()
}

/// The signed area of a triangle, by the shoelace formula.
///
/// The value reads positive for a counter-clockwise winding with y growing
/// up, so it reads positive for a clockwise winding on the screen, whose y
/// grows down.
#[cfg(test)]
fn signed_area(tri: &[Vector2; 3]) -> f32 {
    let mut sum = 0.0;
    for i in 0..3 {
        let a = tri[i];
        let b = tri[(i + 1) % 3];
        sum += a.x * b.y - b.x * a.y;
    }
    sum / 2.0
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the asserted values are exact plane arithmetic
mod tests {
    use super::*;

    fn plane() -> Vector2 {
        Vector2::new(1000.0, 500.0)
    }

    fn drag(anchor: Vector2) -> Drag {
        Drag::new(anchor, Vector2::ZERO, false)
    }

    #[test]
    fn a_drag_spans_its_corners_in_any_direction() {
        let a = Vector2::new(200.0, 100.0);
        let out = drag(a).rect(Vector2::new(500.0, 400.0), plane());
        assert_eq!((out.x, out.y), (200.0, 100.0));
        assert_eq!((out.w, out.h), (300.0, 300.0));
        // A drag that runs up and to the left reads the same way.
        let back = drag(a).rect(Vector2::new(50.0, 20.0), plane());
        assert_eq!((back.x, back.y), (50.0, 20.0));
        assert_eq!((back.w, back.h), (150.0, 80.0));
    }

    #[test]
    fn a_drag_clamps_to_the_plane() {
        let a = Vector2::new(200.0, 100.0);
        let r = drag(a).rect(Vector2::new(5000.0, -80.0), plane());
        assert_eq!((r.x, r.y), (200.0, 0.0));
        assert_eq!((r.w, r.h), (800.0, 100.0));
        // An anchor that a resize pushed past the plane clamps too.
        let past = Drag::new(Vector2::new(2000.0, 900.0), Vector2::ZERO, false);
        let r = past.rect(Vector2::new(2000.0, 900.0), plane());
        assert!(r.is_empty());
        assert_eq!((r.x, r.y), (1000.0, 500.0));
    }

    #[test]
    fn a_square_drag_takes_the_longer_side() {
        let a = Vector2::new(200.0, 100.0);
        let square = Drag::new(a, Vector2::ZERO, true);
        let r = square.rect(Vector2::new(500.0, 200.0), plane());
        assert_eq!((r.w, r.h), (300.0, 300.0));
        // The taller travel wins the same way.
        let r = square.rect(Vector2::new(260.0, 400.0), plane());
        assert_eq!((r.w, r.h), (300.0, 300.0));
        // A corner past the plane still lands inside it.
        let r = square.rect(Vector2::new(900.0, 400.0), plane());
        assert_eq!((r.x, r.y), (200.0, 100.0));
        assert_eq!((r.w, r.h), (400.0, 400.0));
    }

    #[test]
    fn a_square_drag_shrinks_to_the_room_past_the_anchor() {
        // The anchor sits 10 px from the right edge, so a square that grows
        // rightward cannot pass it.
        let square = Drag::new(Vector2::new(990.0, 10.0), Vector2::ZERO, true);
        let r = square.rect(Vector2::new(1500.0, 400.0), plane());
        assert_eq!((r.w, r.h), (10.0, 10.0));
        assert_eq!((r.x, r.y), (990.0, 10.0));
        // The same anchor growing leftward keeps the room the left side has.
        let r = square.rect(Vector2::new(0.0, 400.0), plane());
        assert_eq!((r.w, r.h), (490.0, 490.0));
        assert_eq!((r.x, r.y), (500.0, 10.0));
    }

    #[test]
    fn a_tap_counts_up_to_the_slack() {
        let d = Drag::new(
            Vector2::new(100.0, 100.0),
            Vector2::new(400.0, 300.0),
            false,
        );
        assert!(d.tapped(Vector2::new(400.0, 300.0)));
        assert!(d.tapped(Vector2::new(403.0, 300.0)));
        assert!(!d.tapped(Vector2::new(405.0, 300.0)));
    }

    #[test]
    fn the_bands_cover_the_plane_around_the_selection() {
        let outer = Crop {
            x: 0.0,
            y: 0.0,
            w: 1000.0,
            h: 500.0,
        };
        let inner = Crop {
            x: 100.0,
            y: 50.0,
            w: 200.0,
            h: 100.0,
        };
        let bands = around(outer, inner);
        // One band covers each side, with no band over the selection.
        assert_eq!(
            bands[0],
            [
                Vector2::new(0.0, 0.0),
                Vector2::new(1000.0, 0.0),
                Vector2::new(300.0, 50.0),
                Vector2::new(100.0, 50.0),
            ]
        );
        assert_eq!(
            bands[1],
            [
                Vector2::new(1000.0, 0.0),
                Vector2::new(1000.0, 500.0),
                Vector2::new(300.0, 150.0),
                Vector2::new(300.0, 50.0),
            ]
        );
    }

    #[test]
    fn an_empty_selection_leaves_the_whole_plane_dimmed() {
        let outer = Crop {
            x: 0.0,
            y: 0.0,
            w: 1000.0,
            h: 500.0,
        };
        let empty = Crop {
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 0.0,
        };
        let area: f32 = around(outer, empty).iter().map(quad_area).sum();
        assert_eq!(area, 1000.0 * 500.0);
    }

    #[test]
    fn a_selection_that_touches_a_side_leaves_that_band_empty() {
        let outer = Crop {
            x: 0.0,
            y: 0.0,
            w: 1000.0,
            h: 500.0,
        };
        let inner = Crop {
            x: 0.0,
            y: 0.0,
            w: 1000.0,
            h: 500.0,
        };
        let area: f32 = around(outer, inner).iter().map(quad_area).sum();
        assert_eq!(area, 0.0);
    }

    #[test]
    fn every_veil_triangle_faces_the_draw() {
        let outer = Crop {
            x: 0.0,
            y: 0.0,
            w: 1000.0,
            h: 500.0,
        };
        let inner = Crop {
            x: 100.0,
            y: 50.0,
            w: 200.0,
            h: 100.0,
        };
        for band in around(outer, inner) {
            // raylib's front face reads as a negative shoelace on a y-down
            // screen, so a triangle that faces the other way draws nothing.
            for tri in veil_triangles(band) {
                assert!(signed_area(&tri) < 0.0, "{tri:?} faces away");
            }
            // The two halves still hold the whole band.
            let split: f32 = veil_triangles(band)
                .iter()
                .map(|tri| signed_area(tri).abs())
                .sum();
            assert!((split - quad_area(&band)).abs() < 1e-3);
        }
    }
}

#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    use super::*;

    fn plane() -> Crop {
        Crop {
            x: 0.0,
            y: 0.0,
            w: 1000.0,
            h: 500.0,
        }
    }

    fn rect() -> impl Strategy<Value = Crop> {
        (0.0f32..1000.0, 0.0f32..500.0, 0.0f32..1000.0, 0.0f32..500.0)
            .prop_map(|(x, y, w, h)| Crop { x, y, w, h })
    }

    proptest! {
        /// The veil covers the plane outside the selection and nothing else,
        /// whatever the rect.
        #[test]
        fn the_bands_hold_the_area_outside_the_selection(raw in rect()) {
            let outer = plane();
            let keep = inside(raw, outer);
            let dimmed: f32 = around(outer, raw).iter().map(quad_area).sum();
            let want = outer.w * outer.h - keep.w * keep.h;
            // The shoelace sum runs in f32 over half a megapixel, so the
            // check holds the rounding of that sum and nothing else.
            prop_assert!((dimmed - want).abs() <= want * 1e-5 + 1.0);
        }

        /// Every band stays on the plane, so the veil never darkens the
        /// letterbox.
        #[test]
        fn the_bands_stay_on_the_plane(raw in rect()) {
            let outer = plane();
            for quad in around(outer, raw) {
                for p in quad {
                    prop_assert!((0.0..=outer.w).contains(&p.x));
                    prop_assert!((0.0..=outer.h).contains(&p.y));
                }
            }
        }

        /// A drag always answers a rect inside the plane, whatever the
        /// pointer does.
        #[test]
        fn a_drag_stays_on_the_plane(
            ax in -500.0f32..1500.0,
            ay in -500.0f32..1000.0,
            x in -500.0f32..1500.0,
            y in -500.0f32..1000.0,
            square in any::<bool>(),
        ) {
            let img = Vector2::new(1000.0, 500.0);
            let d = Drag::new(Vector2::new(ax, ay), Vector2::ZERO, square);
            let r = d.rect(Vector2::new(x, y), img);
            prop_assert!(r.x >= 0.0 && r.y >= 0.0);
            prop_assert!(r.right() <= img.x + 1e-3);
            prop_assert!(r.bottom() <= img.y + 1e-3);
            if square {
                prop_assert!((r.w - r.h).abs() < 1e-3);
            }
        }
    }
}
