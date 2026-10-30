//! The visible-region raster of the image view.
//!
//! A vector file bakes one base raster (see [`crate::svg`]), and a zoom past
//! that size would stretch pixels.
//! The view therefore paints the region the window shows, at the scale the
//! window shows it, and blits that patch over the image.
//! The paint area then tracks window pixels, not document pixels, so a deep
//! zoom costs no more than a shallow one and the stretch stays exact.
//!
//! [`crop`] maps the window rect back into source pixels and expands it by
//! [`MARGIN`], [`worth_painting`] guards the repaint with [`SCALE_SLACK`] plus
//! a containment test, [`covers`] answers the same question for a patch that
//! already exists, and [`QUIET`] debounces the request past the ease.
//!
//! The mapping mirrors the draw: the image center sits at the window middle
//! plus the pan, and the image turns clockwise around that point.
//! [`to_screen`] and [`to_image`] are inverses, and `proptest` holds that.
//!
//! # Examples
//!
//! ```
//! use raylib::prelude::Vector2;
//! use versatile_viewer::region;
//!
//! let img = Vector2::new(1024.0, 512.0);
//! let win = Vector2::new(800.0, 600.0);
//! let pan = Vector2::ZERO;
//! // The whole document fits the window at this scale, so the crop covers it
//! // all.
//! let whole = region::crop(img, win, pan, 0.5, 0.0, 0.0);
//! assert_eq!((whole.w, whole.h), (1024.0, 512.0));
//! // Zoomed in, the crop covers the window only, around the image center.
//! let zoomed = region::crop(img, win, pan, 4.0, 0.0, 0.0);
//! assert!((zoomed.w - 200.0).abs() < 1e-3);
//! assert!((zoomed.h - 150.0).abs() < 1e-3);
//! ```

use std::time::Duration;

use raylib::prelude::Vector2;

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

/// An axis-aligned rectangle in source pixels.
///
/// `x` and `y` give the top-left corner, and the axes grow right and down,
/// like the base raster the region belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crop {
    /// The left edge, in source pixels.
    pub x: f32,
    /// The top edge, in source pixels.
    pub y: f32,
    /// The width, in source pixels.
    pub w: f32,
    /// The height, in source pixels.
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
    pub fn contains(&self, other: &Self) -> bool {
        other.x >= self.x
            && other.y >= self.y
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
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
    pub img: Vector2,
    /// The window size, in window pixels.
    pub win: Vector2,
    /// The offset of the plane center from the window middle, in window
    /// pixels.
    pub pan: Vector2,
    /// Output pixels per source pixel.
    pub scale: f32,
    /// The clockwise turn, in degrees.
    pub angle: f32,
}

impl Frame {
    /// The window position of a source point.
    #[must_use]
    pub fn to_screen(&self, point: Vector2) -> Vector2 {
        to_screen(point, self.img, self.win, self.pan, self.scale, self.angle)
    }

    /// The source point under a window position.
    #[must_use]
    pub fn to_image(&self, point: Vector2) -> Vector2 {
        to_image(point, self.img, self.win, self.pan, self.scale, self.angle)
    }

    /// Whether the frame names a plane the view can draw.
    #[must_use]
    pub fn drawable(&self) -> bool {
        self.img.x > 0.0 && self.img.y > 0.0
    }
}

/// The window position of a source point, matching the draw transform.
///
/// The image center sits at the window middle plus the pan, the image scales
/// around that point, and a positive `angle` turns it clockwise.
#[must_use]
pub fn to_screen(
    point: Vector2,
    img: Vector2,
    win: Vector2,
    pan: Vector2,
    scale: f32,
    angle: f32,
) -> Vector2 {
    let (sin, cos) = angle.to_radians().sin_cos();
    let dx = (point.x - img.x / 2.0) * scale;
    let dy = (point.y - img.y / 2.0) * scale;
    Vector2::new(
        win.x / 2.0 + pan.x + dx * cos - dy * sin,
        win.y / 2.0 + pan.y + dx * sin + dy * cos,
    )
}

/// The source point under a window position, the inverse of [`to_screen`].
///
/// A zero or non-finite `scale` reads as one, so the mapping never divides by
/// zero.
#[must_use]
pub fn to_image(
    point: Vector2,
    img: Vector2,
    win: Vector2,
    pan: Vector2,
    scale: f32,
    angle: f32,
) -> Vector2 {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let (sin, cos) = angle.to_radians().sin_cos();
    let dx = point.x - win.x / 2.0 - pan.x;
    let dy = point.y - win.y / 2.0 - pan.y;
    Vector2::new(
        img.x / 2.0 + (dx * cos + dy * sin) / scale,
        img.y / 2.0 + (-dx * sin + dy * cos) / scale,
    )
}

/// The source rect that the window shows, grown by `margin` and clipped to the
/// document.
///
/// The function maps the four window corners back into source pixels and takes
/// their bounding box, so a turned image grows the rect to cover its corners.
/// A zero `margin` gives the rect the paint must cover, and [`MARGIN`] gives
/// the rect worth painting.
#[must_use]
pub fn crop(img: Vector2, win: Vector2, pan: Vector2, scale: f32, angle: f32, margin: f32) -> Crop {
    let margin = margin.max(0.0);
    if !(img.x > 0.0 && img.y > 0.0 && win.x > 0.0 && win.y > 0.0) {
        return Crop {
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 0.0,
        };
    }
    let (x0, y0) = (-margin * win.x, -margin * win.y);
    let (x1, y1) = (win.x * (1.0 + margin), win.y * (1.0 + margin));
    let corners = [
        Vector2::new(x0, y0),
        Vector2::new(x1, y0),
        Vector2::new(x1, y1),
        Vector2::new(x0, y1),
    ];
    let mut min = Vector2::new(f32::INFINITY, f32::INFINITY);
    let mut max = Vector2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    for corner in corners {
        let point = to_image(corner, img, win, pan, scale, angle);
        min.x = min.x.min(point.x);
        min.y = min.y.min(point.y);
        max.x = max.x.max(point.x);
        max.y = max.y.max(point.y);
    }
    let x = min.x.clamp(0.0, img.x);
    let y = min.y.clamp(0.0, img.y);
    let right = max.x.clamp(x, img.x);
    let bottom = max.y.clamp(y, img.y);
    Crop {
        x,
        y,
        w: right - x,
        h: bottom - y,
    }
}

/// Whether the view must repaint its region.
///
/// A repaint pays when the scale moved past [`SCALE_SLACK`], or when the
/// painted rect no longer covers what the window shows.
/// The caller passes the rect without a margin, so a consumed margin alone
/// never pays for a paint.
#[must_use]
pub fn worth_painting(
    needed: &Crop,
    painted: Option<&Crop>,
    need_scale: f32,
    painted_scale: f32,
) -> bool {
    if needed.is_empty() {
        return false;
    }
    let Some(painted) = painted else {
        return true;
    };
    let scaled =
        need_scale > painted_scale * SCALE_SLACK || need_scale * SCALE_SLACK < painted_scale;
    scaled || !painted.contains(needed)
}

/// Whether a painted rect holds the region the window shows at its scale.
///
/// This is the inverse of [`worth_painting`]: the caller owns a patch of
/// `patch` at `patch_scale`, and the view needs `needed` at `need_scale`.
/// A spread step paints its landing patch before the glide ends, and this test
/// tells the view when that patch has become the one on screen.
#[must_use]
pub fn covers(needed: &Crop, patch: &Crop, need_scale: f32, patch_scale: f32) -> bool {
    if needed.is_empty() {
        return false;
    }
    let scaled = need_scale > patch_scale * SCALE_SLACK || need_scale * SCALE_SLACK < patch_scale;
    !scaled && patch.contains(needed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fit_scale_covers_the_whole_document() {
        let img = Vector2::new(1024.0, 512.0);
        let win = Vector2::new(800.0, 600.0);
        let pan = Vector2::ZERO;
        let whole = crop(img, win, pan, 600.0 / 1024.0, 0.0, 0.0);
        assert!((whole.w - 1024.0).abs() < 1.0);
        assert!((whole.h - 512.0).abs() < 1.0);
        assert!(whole.x.abs() < 1.0 && whole.y.abs() < 1.0);
    }

    #[test]
    fn a_zoomed_crop_covers_about_one_window() {
        let img = Vector2::new(1024.0, 512.0);
        let win = Vector2::new(800.0, 600.0);
        let zoomed = crop(img, win, Vector2::ZERO, 4.0, 0.0, 0.0);
        // 800/4 by 600/4 source pixels, centered.
        assert!((zoomed.w - 200.0).abs() < 1e-3);
        assert!((zoomed.h - 150.0).abs() < 1e-3);
        assert!((zoomed.x - 412.0).abs() < 1e-3);
        assert!((zoomed.y - 181.0).abs() < 1e-3);
    }

    #[test]
    fn the_margin_grows_the_rect_and_the_clip_bounds_it() {
        let img = Vector2::new(1024.0, 512.0);
        let win = Vector2::new(800.0, 600.0);
        let plain = crop(img, win, Vector2::ZERO, 4.0, 0.0, 0.0);
        let grown = crop(img, win, Vector2::ZERO, 4.0, 0.0, MARGIN);
        assert!(grown.contains(&plain));
        assert!(grown.w > plain.w && grown.h > plain.h);
        // A pan far outside the document still clips to it.
        let out = crop(img, win, Vector2::new(9000.0, 9000.0), 1.0, 0.0, MARGIN);
        assert!(out.w <= img.x && out.h <= img.y);
        assert!(out.x >= 0.0 && out.y >= 0.0);
    }

    #[test]
    fn the_paint_rule_covers_scale_and_coverage() {
        let img = Vector2::new(1024.0, 512.0);
        let win = Vector2::new(800.0, 600.0);
        let pan = Vector2::ZERO;
        let needed = crop(img, win, pan, 4.0, 0.0, 0.0);
        let painted = crop(img, win, pan, 4.0, 0.0, MARGIN);
        // A fresh region and a covered one.
        assert!(worth_painting(&needed, None, 4.0, 4.0));
        assert!(!worth_painting(&needed, Some(&painted), 4.0, 4.0));
        // A scale inside the float tolerance stays: the patch still lands one
        // pixel per pixel.
        assert!(!worth_painting(&needed, Some(&painted), 4.004, 4.0));
        // Any other scale stretch softens every pixel, so it pays.
        assert!(worth_painting(&needed, Some(&painted), 4.1, 4.0));
        assert!(worth_painting(&needed, Some(&painted), 3.9, 4.0));
        let far = crop(img, win, Vector2::new(300.0, 0.0), 4.0, 0.0, 0.0);
        assert!(worth_painting(&far, Some(&painted), 4.0, 4.0));
        // An empty need never pays.
        let none = Crop {
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 0.0,
        };
        assert!(!worth_painting(&none, None, 4.0, 4.0));
    }

    #[test]
    fn a_patch_covers_the_region_it_holds() {
        let img = Vector2::new(1024.0, 512.0);
        let win = Vector2::new(800.0, 600.0);
        let pan = Vector2::ZERO;
        let needed = crop(img, win, pan, 4.0, 0.0, 0.0);
        let painted = crop(img, win, pan, 4.0, 0.0, MARGIN);
        // The margin holds the window at the same scale.
        assert!(covers(&needed, &painted, 4.0, 4.0));
        // A scale past the slack, and a rect that misses the window, fail.
        assert!(!covers(&needed, &painted, 4.1, 4.0));
        let far = crop(img, win, Vector2::new(300.0, 0.0), 4.0, 0.0, MARGIN);
        assert!(!covers(&needed, &far, 4.0, 4.0));
        // The test and the paint rule never both fire.
        assert!(
            !(covers(&needed, &painted, 4.0, 4.0)
                && worth_painting(&needed, Some(&painted), 4.0, 4.0))
        );
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

    proptest! {
        /// The two mappings undo each other, so the region lines up with the
        /// draw.
        #[test]
        fn the_mapping_round_trips(
            (scale, angle, px, py) in view(),
            x in 0.0f32..1024.0,
            y in 0.0f32..512.0,
        ) {
            let img = Vector2::new(1024.0, 512.0);
            let win = Vector2::new(800.0, 600.0);
            let pan = Vector2::new(px, py);
            let point = Vector2::new(x, y);
            let screen = to_screen(point, img, win, pan, scale, angle);
            let back = to_image(screen, img, win, pan, scale, angle);
            prop_assert!((back.x - point.x).abs() < 1e-2);
            prop_assert!((back.y - point.y).abs() < 1e-2);
        }

        /// Every window corner lands inside the region the view paints, so the
        /// patch always covers the screen.
        #[test]
        fn the_crop_covers_the_window((scale, angle, px, py) in view()) {
            let img = Vector2::new(1024.0, 512.0);
            let win = Vector2::new(800.0, 600.0);
            let pan = Vector2::new(px, py);
            let painted = crop(img, win, pan, scale, angle, MARGIN);
            let needed = crop(img, win, pan, scale, angle, 0.0);
            prop_assert!(painted.contains(&needed));
            prop_assert!(painted.x >= 0.0 && painted.y >= 0.0);
            prop_assert!(painted.right() <= img.x + 1e-3);
            prop_assert!(painted.bottom() <= img.y + 1e-3);
        }

        /// The painted area stays bounded by the window, so a deeper zoom
        /// costs no more pixels than a shallow one.
        #[test]
        fn the_paint_area_stays_bounded((scale, angle, px, py) in view()) {
            let img = Vector2::new(1024.0, 512.0);
            let win = Vector2::new(800.0, 600.0);
            let pan = Vector2::new(px, py);
            let painted = crop(img, win, pan, scale, angle, MARGIN);
            // A turned rect grows its bounding box by at most its own
            // diagonal, so that diagonal bounds both axes.
            let limit = (1.0 + 2.0 * MARGIN) * (win.x * win.x + win.y * win.y).sqrt();
            // A window turned toward its diagonal reaches the bound exactly,
            // and the rounding of the sine and cosine pushes the product past
            // it by a few floating-point steps. The slack covers that rounding
            // and nothing else.
            prop_assert!(painted.w * scale <= limit * (1.0 + 1e-3));
            prop_assert!(painted.h * scale <= limit * (1.0 + 1e-3));
        }
    }
}
