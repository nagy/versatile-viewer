//! The visible-region raster of the image view.
//!
//! A vector file bakes one base raster (see [`crate::svg`]), and a zoom past
//! that size would stretch pixels.
//! The view therefore paints the region the window shows, at the scale the
//! window shows it, and blits that patch over the image.
//! The paint area then tracks window pixels, not document pixels, so a deep
//! zoom costs no more than a shallow one and the stretch stays exact.
//!
//! [`Frame::crop`] maps the window rect back into source units and expands it
//! by [`MARGIN`], [`worth_painting`] guards the repaint with [`SCALE_SLACK`]
//! plus a containment test, [`covers`] answers the same question for a patch
//! that already exists, and [`QUIET`] debounces the request past the ease.
//!
//! The mapping mirrors the draw: the image center sits at the window middle
//! plus the pan, and the image turns clockwise around that point.
//! [`Frame::to_screen`] and [`Frame::to_image`] are inverses, and `proptest`
//! holds that.
//! One [`Frame`] carries the five numbers that mapping needs, so no function
//! takes them loose and no call site swaps the scale for the angle.
//!
//! # Examples
//!
//! ```
//! use versatile_viewer::{draw::Vec2, region};
//!
//! let frame = |scale| region::Frame {
//!     img: Vec2::new(1024.0, 512.0),
//!     win: Vec2::new(800.0, 600.0),
//!     pan: Vec2::ZERO,
//!     scale,
//!     angle: 0.0,
//! };
//! // The whole document fits the window at this scale, so the crop covers it
//! // all.
//! let whole = frame(0.5).crop(0.0);
//! assert_eq!((whole.w, whole.h), (1024.0, 512.0));
//! // Zoomed in, the crop covers the window only, around the image center.
//! let zoomed = frame(4.0).crop(0.0);
//! assert!((zoomed.w - 200.0).abs() < 1e-3);
//! assert!((zoomed.h - 150.0).abs() < 1e-3);
//! ```

use std::time::Duration;

use crate::draw::Vec2;

/// The fraction of the window each side adds to the painted region.
///
/// The margin covers a small pan or nudge between two paints, so the patch
/// still fills the window while the debounce waits.
/// A quarter of a window per side costs 1.5x the linear size, and rotation
/// already grows the region on its own.
pub const MARGIN: f32 = 0.25;

/// The scale ratio that pays for a repaint.
///
/// A paint is exact at its own scale, and the view lands it one output pixel
/// per screen pixel only while the two agree.
/// A patch stretched by even a few percent resamples every pixel through the
/// GPU filter, which softens text, so the rule stays tight.
/// The tolerance covers the float noise of the two computations and the last
/// step of the zoom ease, and the [`QUIET`] debounce is what keeps a continuous
/// zoom from paying for a pixmap per frame.
pub const SCALE_SLACK: f32 = 1.002;

/// How long the view must hold still before the region repaints.
///
/// The ease settles in about three of its time constants, so 150 ms sits just
/// past it, and a continuous drag never queues a paint per frame.
pub const QUIET: Duration = Duration::from_millis(150);

/// An axis-aligned rectangle in surface units.
///
/// `x` and `y` give the top-left corner, and the axes grow right and down,
/// like the base raster the region belongs to.
/// The unit is the plane's own: base raster pixels for an image or a vector
/// file, and PostScript points for a document page.
/// [`crate::surface::View`] carries this same type, so the view and a backend
/// pass one value and never convert between two rect shapes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crop {
    /// The left edge, in surface units.
    pub x: f32,
    /// The top edge, in surface units.
    pub y: f32,
    /// The width, in surface units.
    pub w: f32,
    /// The height, in surface units.
    pub h: f32,
}

impl Crop {
    /// The right edge.
    #[must_use]
    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    /// The bottom edge.
    #[must_use]
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    /// Whether this rect covers all of `other`.
    #[must_use]
    pub fn contains_rect(&self, other: &Self) -> bool {
        other.x >= self.x
            && other.y >= self.y
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }

    /// Whether a source point lies inside this rect.
    ///
    /// The edges count as inside, so a press on the border grabs the rect.
    #[must_use]
    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.x
            && point.y >= self.y
            && point.x <= self.right()
            && point.y <= self.bottom()
    }

    /// Whether the rect covers no pixels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.w <= 0.0 || self.h <= 0.0
    }
}

/// One window mapping of a plane: the numbers the draw uses, bundled.
///
/// The picture centers at the window middle plus the pan, scales by `scale`
/// and turns clockwise by `angle` degrees.
/// [`Frame::to_screen`] and [`Frame::to_image`] are the one conversion pair,
/// so a caller that keeps a `Frame` never mixes window units with source
/// units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// The plane size, in source pixels.
    pub img: Vec2,
    /// The window size, in window pixels.
    pub win: Vec2,
    /// The offset of the plane center from the window middle, in window
    /// pixels.
    pub pan: Vec2,
    /// Output pixels per source pixel.
    pub scale: f32,
    /// The clockwise turn, in degrees.
    pub angle: f32,
}

impl Frame {
    /// The window position of a source point, matching the draw transform.
    ///
    /// The image center sits at the window middle plus the pan, the image
    /// scales around that point, and a positive `angle` turns it clockwise.
    #[must_use]
    pub fn to_screen(&self, point: Vec2) -> Vec2 {
        let (sin, cos) = self.angle.to_radians().sin_cos();
        let dx = (point.x - self.img.x / 2.0) * self.scale;
        let dy = (point.y - self.img.y / 2.0) * self.scale;
        Vec2::new(
            self.win.x / 2.0 + self.pan.x + dx * cos - dy * sin,
            self.win.y / 2.0 + self.pan.y + dx * sin + dy * cos,
        )
    }

    /// The source point under a window position, the inverse of
    /// [`Self::to_screen`].
    ///
    /// A zero or non-finite `scale` reads as one, so the mapping never divides
    /// by zero.
    #[must_use]
    pub fn to_image(&self, point: Vec2) -> Vec2 {
        let scale = if self.scale.is_finite() && self.scale > 0.0 {
            self.scale
        } else {
            1.0
        };
        let (sin, cos) = self.angle.to_radians().sin_cos();
        let dx = point.x - self.win.x / 2.0 - self.pan.x;
        let dy = point.y - self.win.y / 2.0 - self.pan.y;
        Vec2::new(
            self.img.x / 2.0 + (dx * cos + dy * sin) / scale,
            self.img.y / 2.0 + (-dx * sin + dy * cos) / scale,
        )
    }

    /// Whether the frame names a plane the view can draw.
    #[must_use]
    pub fn has_drawable_plane(&self) -> bool {
        self.img.x > 0.0 && self.img.y > 0.0
    }

    /// The source rect that the window shows, grown by `margin` and clipped to
    /// the plane.
    ///
    /// The function maps the four window corners back into source units and
    /// takes their bounding box, so a turned image grows the rect to cover its
    /// corners.
    /// A zero `margin` gives the rect the paint must cover, and [`MARGIN`]
    /// gives the rect worth painting.
    #[must_use]
    pub fn crop(&self, margin: f32) -> Crop {
        let margin = margin.max(0.0);
        if !(self.has_drawable_plane() && self.win.x > 0.0 && self.win.y > 0.0) {
            return Crop {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            };
        }
        let (x0, y0) = (-margin * self.win.x, -margin * self.win.y);
        let (x1, y1) = (self.win.x * (1.0 + margin), self.win.y * (1.0 + margin));
        let corners = [
            Vec2::new(x0, y0),
            Vec2::new(x1, y0),
            Vec2::new(x1, y1),
            Vec2::new(x0, y1),
        ];
        let (mut min, mut max) = (
            Vec2::new(f32::INFINITY, f32::INFINITY),
            Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        );
        for corner in corners {
            let point = self.to_image(corner);
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
        }
        let x = min.x.clamp(0.0, self.img.x);
        let y = min.y.clamp(0.0, self.img.y);
        let right = max.x.clamp(x, self.img.x);
        let bottom = max.y.clamp(y, self.img.y);
        Crop {
            x,
            y,
            w: right - x,
            h: bottom - y,
        }
    }
}

/// A rect at the scale its pixels were rasterized at.
///
/// [`worth_painting`] and [`covers`] compare two of these, so a rect never
/// travels beside a loose scale that a caller can swap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Patch {
    /// The source rect the pixels cover.
    pub rect: Crop,
    /// The pixels per source unit those pixels hold.
    pub scale: f32,
}

impl Patch {
    /// A patch of `rect` at `scale`.
    #[must_use]
    pub const fn new(rect: Crop, scale: f32) -> Self {
        Self { rect, scale }
    }
}

/// Whether the view must repaint its region.
///
/// A repaint pays when the scale moved past [`SCALE_SLACK`], or when the
/// painted rect no longer covers what the window shows.
/// The caller passes the rect without a margin, so a consumed margin alone
/// never pays for a paint.
#[must_use]
pub fn worth_painting(needed: &Patch, painted: Option<&Patch>) -> bool {
    if needed.rect.is_empty() {
        return false;
    }
    let Some(painted) = painted else {
        return true;
    };
    mismatched_scale(needed, painted) || !patch_reaches_window(needed, painted)
}

/// Whether a painted rect reaches every source pixel the window shows.
///
/// The test reads the rects alone, so a patch stretched by a zoom still holds
/// the window.
/// [`patch_covers_at_scale`] adds the scale to it, and the draw asks this one:
/// a patch that reaches the window alone may draw without the whole-image
/// texture under it.
#[must_use]
pub fn patch_reaches_window(needed: &Patch, painted: &Patch) -> bool {
    if needed.rect.is_empty() {
        return false;
    }
    painted.rect.contains_rect(&needed.rect)
}

/// Whether a painted rect reaches the window at the scale it shows.
///
/// This is the inverse of [`worth_painting`]: the view needs `needed` at its
/// own scale, and the patch on screen holds `painted` at another.
/// A spread step paints its landing patch before the glide ends, and this test
/// tells the view when that patch has become the one on screen.
#[must_use]
pub fn patch_covers_at_scale(needed: &Patch, painted: &Patch) -> bool {
    patch_reaches_window(needed, painted) && !mismatched_scale(needed, painted)
}

/// Whether two patches disagree about the scale beyond [`SCALE_SLACK`].
fn mismatched_scale(needed: &Patch, painted: &Patch) -> bool {
    needed.scale > painted.scale * SCALE_SLACK || needed.scale * SCALE_SLACK < painted.scale
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame over a 1024x512 plane inside an 800x600 window.
    fn frame(pan: Vec2, scale: f32, angle: f32) -> Frame {
        Frame {
            img: Vec2::new(1024.0, 512.0),
            win: Vec2::new(800.0, 600.0),
            pan,
            scale,
            angle,
        }
    }

    #[test]
    fn a_fit_scale_covers_the_whole_document() {
        let whole = frame(Vec2::ZERO, 600.0 / 1024.0, 0.0).crop(0.0);
        assert!((whole.w - 1024.0).abs() < 1.0);
        assert!((whole.h - 512.0).abs() < 1.0);
        assert!(whole.x.abs() < 1.0 && whole.y.abs() < 1.0);
    }

    #[test]
    fn a_zoomed_crop_covers_about_one_window() {
        let zoomed = frame(Vec2::ZERO, 4.0, 0.0).crop(0.0);
        // 800/4 by 600/4 source pixels, centered.
        assert!((zoomed.w - 200.0).abs() < 1e-3);
        assert!((zoomed.h - 150.0).abs() < 1e-3);
        assert!((zoomed.x - 412.0).abs() < 1e-3);
        assert!((zoomed.y - 181.0).abs() < 1e-3);
    }

    #[test]
    fn the_margin_grows_the_rect_and_the_clip_bounds_it() {
        let plain = frame(Vec2::ZERO, 4.0, 0.0).crop(0.0);
        let grown = frame(Vec2::ZERO, 4.0, 0.0).crop(MARGIN);
        assert!(grown.contains_rect(&plain));
        assert!(grown.w > plain.w && grown.h > plain.h);
        // A pan far outside the document still clips to it.
        let out = frame(Vec2::new(9000.0, 9000.0), 1.0, 0.0).crop(MARGIN);
        assert!(out.w <= 1024.0 && out.h <= 512.0);
        assert!(out.x >= 0.0 && out.y >= 0.0);
    }

    #[test]
    fn a_degenerate_frame_answers_an_empty_crop() {
        // A plane the view cannot draw, and a window with no pixels, both
        // answer an empty rect instead of a division by zero.
        let flat = Frame {
            img: Vec2::ZERO,
            ..frame(Vec2::ZERO, 1.0, 0.0)
        };
        assert!(flat.crop(0.0).is_empty());
        let blind = Frame {
            win: Vec2::ZERO,
            ..frame(Vec2::ZERO, 1.0, 0.0)
        };
        assert!(blind.crop(0.0).is_empty());
    }

    #[test]
    fn the_paint_rule_covers_scale_and_coverage() {
        let needed = Patch::new(frame(Vec2::ZERO, 4.0, 0.0).crop(0.0), 4.0);
        let painted = Patch::new(frame(Vec2::ZERO, 4.0, 0.0).crop(MARGIN), 4.0);
        // A fresh region and a covered one.
        assert!(worth_painting(&needed, None));
        assert!(!worth_painting(&needed, Some(&painted)));
        // A scale inside the float tolerance stays: the patch still lands one
        // pixel per pixel.
        assert!(!worth_painting(
            &needed,
            Some(&Patch {
                scale: 4.004,
                ..painted
            })
        ));
        // Any other scale stretch softens every pixel, so it pays.
        assert!(worth_painting(
            &needed,
            Some(&Patch {
                scale: 4.1,
                ..painted
            })
        ));
        assert!(worth_painting(
            &needed,
            Some(&Patch {
                scale: 3.9,
                ..painted
            })
        ));
        let far = Patch::new(frame(Vec2::new(300.0, 0.0), 4.0, 0.0).crop(0.0), 4.0);
        assert!(worth_painting(&far, Some(&painted)));
        // An empty need never pays.
        let none = Patch::new(
            Crop {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            },
            4.0,
        );
        assert!(!worth_painting(&none, None));
    }

    #[test]
    fn a_patch_covers_the_region_it_holds() {
        let needed = Patch::new(frame(Vec2::ZERO, 4.0, 0.0).crop(0.0), 4.0);
        let painted = Patch::new(frame(Vec2::ZERO, 4.0, 0.0).crop(MARGIN), 4.0);
        // The margin holds the window at the same scale.
        assert!(patch_covers_at_scale(&needed, &painted));
        // A scale past the slack, and a rect that misses the window, fail.
        assert!(!patch_covers_at_scale(
            &needed,
            &Patch {
                scale: 4.1,
                ..painted
            }
        ));
        let far = Patch::new(frame(Vec2::new(300.0, 0.0), 4.0, 0.0).crop(MARGIN), 4.0);
        assert!(!patch_covers_at_scale(&needed, &far));
        // The test and the paint rule never both fire.
        assert!(
            !(patch_covers_at_scale(&needed, &painted) && worth_painting(&needed, Some(&painted)))
        );
    }

    #[test]
    fn a_patch_holds_only_the_window_it_reaches() {
        let wide = frame(Vec2::ZERO, 4.0, 0.0);
        let needed = Patch::new(wide.crop(0.0), 4.0);
        let covering = Patch::new(wide.crop(MARGIN), 4.0);
        assert!(patch_reaches_window(&needed, &covering));
        // Zoomed out, the window shows more than the patch painted, so the
        // patch holds a part and the whole image must draw under it.
        let out = frame(Vec2::ZERO, 1.0, 0.0);
        let wide_need = Patch::new(out.crop(0.0), 1.0);
        assert!(!patch_reaches_window(&wide_need, &covering));
        // A patch at another scale still holds the rect it reaches, and the
        // scale is `patch_covers_at_scale`'s business alone.
        assert!(patch_reaches_window(
            &needed,
            &Patch {
                scale: 8.0,
                ..covering
            }
        ));
        assert!(!patch_covers_at_scale(
            &needed,
            &Patch {
                scale: 8.0,
                ..covering
            }
        ));
        // An empty need is never held.
        let none = Patch::new(
            Crop {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            },
            4.0,
        );
        assert!(!patch_reaches_window(&none, &covering));
    }
}

#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    use super::*;

    fn view() -> impl Strategy<Value = (f32, f32, f32, f32)> {
        (
            1.0f32..64.0,       // scale
            -360.0f32..360.0,   // angle
            -4000.0f32..4000.0, // pan x
            -4000.0f32..4000.0, // pan y
        )
    }

    /// A frame over a 1024x512 plane inside an 800x600 window.
    fn frame(pan: Vec2, scale: f32, angle: f32) -> Frame {
        Frame {
            img: Vec2::new(1024.0, 512.0),
            win: Vec2::new(800.0, 600.0),
            pan,
            scale,
            angle,
        }
    }

    proptest! {
        /// The two mappings undo each other, so the region lines up with the
        /// draw.
        #[test]
        fn the_mapping_round_trips(
            (scale, angle, px, py) in view(),
            x in 0.0f32..1024.0,
            y in 0.0f32..512.0,
        ) {
            let frame = frame(Vec2::new(px, py), scale, angle);
            let point = Vec2::new(x, y);
            let back = frame.to_image(frame.to_screen(point));
            prop_assert!((back.x - point.x).abs() < 1e-2);
            prop_assert!((back.y - point.y).abs() < 1e-2);
        }

        /// Every window corner lands inside the region the view paints, so the
        /// patch always covers the screen.
        #[test]
        fn the_crop_covers_the_window((scale, angle, px, py) in view()) {
            let frame = frame(Vec2::new(px, py), scale, angle);
            let painted = frame.crop(MARGIN);
            let needed = frame.crop(0.0);
            prop_assert!(painted.contains_rect(&needed));
            prop_assert!(painted.x >= 0.0 && painted.y >= 0.0);
            prop_assert!(painted.right() <= 1024.0 + 1e-3);
            prop_assert!(painted.bottom() <= 512.0 + 1e-3);
        }

        /// The painted area stays bounded by the window, so a deeper zoom
        /// costs no more pixels than a shallow one.
        #[test]
        fn the_paint_area_stays_bounded((scale, angle, px, py) in view()) {
            let frame = frame(Vec2::new(px, py), scale, angle);
            let painted = frame.crop(MARGIN);
            // A turned rect grows its bounding box by at most its own
            // diagonal, so that diagonal bounds both axes.
            let (w, h) = (frame.win.x, frame.win.y);
            let limit = (1.0 + 2.0 * MARGIN) * (w * w + h * h).sqrt();
            // A window turned toward its diagonal reaches the bound exactly,
            // and the rounding of the sine and cosine pushes the product past
            // it by a few floating-point steps. The slack covers that rounding
            // and nothing else.
            prop_assert!(painted.w * scale <= limit * (1.0 + 1e-3));
            prop_assert!(painted.h * scale <= limit * (1.0 + 1e-3));
        }
    }
}
