//! The visible-region raster of the image view.
//!
//! A vector file bakes one base raster (see the vector backend), and a zoom past
//! that size would stretch pixels.
//! The view therefore paints the region the window shows, at the scale the
//! window shows it, and blits that patch over the image.
//! The paint area then tracks window pixels, not document pixels, so a deep
//! zoom costs no more than a shallow one and the stretch stays exact.
//!
//! [`Frame::crop`] maps the window rect back into source units and expands it
//! by [`MARGIN`], [`worth_painting`] guards the repaint with [`SCALE_SLACK`]
//! plus a containment test, [`patch_reaches_window`] answers the same question
//! for a patch
//! that already exists, and [`QUIET`] debounces the request past the ease.
//!
//! The mapping mirrors the draw: the image center sits at the window middle
//! plus the pan, the image scales by one factor per axis, and it turns
//! clockwise around that point.
//! [`Frame::to_screen`] and [`Frame::to_image`] are inverses, and `proptest`
//! holds that.
//! One [`Frame`] carries the six numbers that mapping needs, so no function
//! takes them loose and no call site swaps the scale for the angle.
//! [`Scale`] carries the two factors, because a view that shows a stated box
//! fills the window on both axes at once and a picture that fits keeps them
//! equal.
//!
//! # Examples
//!
//! ```
//! use vv_core::{
//!     dim::{SurfaceSize, WindowSize, WindowVector},
//!     region::{self, Scale},
//! };
//!
//! let frame = |scale| region::Frame {
//!     img: SurfaceSize::new(1024.0, 512.0),
//!     win: WindowSize::new(800.0, 600.0),
//!     pan: WindowVector::zero(),
//!     scale: Scale::uniform(scale),
//!     angle: 0.0,
//! };
//! // The whole document fits the window at this scale, so the crop covers it
//! // all.
//! let whole = frame(0.5).crop(0.0);
//! assert_eq!((whole.width(), whole.height()), (1024.0, 512.0));
//! // Zoomed in, the crop covers the window only, around the image center.
//! let zoomed = frame(4.0).crop(0.0);
//! assert!((zoomed.width() - 200.0).abs() < 1e-3);
//! assert!((zoomed.height() - 150.0).abs() < 1e-3);
//! // The two axes are independent, so a view that states one factor per axis
//! // shows a box of the plane that is not the plane's own shape.
//! let axes = |scale| region::Frame {
//!     scale,
//!     ..frame(0.5)
//! };
//! let squashed = axes(Scale::axes(1.0, 4.0)).crop(0.0);
//! assert!((squashed.width() - 800.0).abs() < 1e-3);
//! assert!((squashed.height() - 150.0).abs() < 1e-3);
//! ```

use std::time::Duration;

use crate::dim::{SurfacePoint, SurfaceSize, WindowPoint, WindowSize, WindowVector};

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
/// The origin gives the top-left corner, and the axes grow right and down,
/// like the base raster the region belongs to.
/// The unit is the plane's own: base raster pixels for an image or a vector
/// file, and PostScript points for a document page.
/// [`crate::dim::SurfaceSpace`] tags the type, so a window rect never passes
/// where a crop belongs, and the surface `View` carries this same type, so the
/// view and a backend pass one value and never convert between two rect shapes.
pub type Crop = euclid::Rect<f32, crate::dim::SurfaceSpace>;

/// A crop rect from its top-left corner and its size.
#[must_use]
pub const fn crop(x: f32, y: f32, w: f32, h: f32) -> Crop {
    Crop::new(euclid::point2(x, y), euclid::size2(w, h))
}

/// Whether a source point lies inside `rect`.
///
/// The edges count as inside, so a press on the border grabs the rect, while
/// `euclid` puts the far edge outside.
#[must_use]
pub fn contains_point(rect: Crop, point: SurfacePoint) -> bool {
    crate::dim::contains_point(&rect, point.x, point.y)
}

/// The output pixels one surface unit covers on each axis.
///
/// The two axes are independent because two different views want them so: a
/// fit scales a picture by one factor on both, and a view that states the box
/// its window shows fills the window on each axis at its own ratio.
/// [`Scale::uniform`] is the one-factor constructor, so a reader of a value
/// says which case it means, and [`Scale::coarsest`] is the one number a paint
/// of both axes can serve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scale {
    /// Output pixels per surface unit across.
    pub x: f32,
    /// Output pixels per surface unit down.
    pub y: f32,
}

impl Scale {
    /// One factor, both axes.
    #[must_use]
    pub const fn uniform(scale: f32) -> Self {
        Self { x: scale, y: scale }
    }

    /// One factor per axis.
    #[must_use]
    pub const fn axes(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Whether both factors are positive and finite.
    #[must_use]
    pub fn is_drawable(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.x > 0.0 && self.y > 0.0
    }

    /// The pair with a zero or non-finite factor read as one.
    ///
    /// The inverse mapping divides by the factors, so a frame the caller left
    /// half-built still answers a point instead of an infinity.
    #[must_use]
    pub fn drawable(self) -> Self {
        let one = |value: f32| {
            if value.is_finite() && value != 0.0 {
                value
            } else {
                1.0
            }
        };
        Self {
            x: one(self.x),
            y: one(self.y),
        }
    }

    /// The smaller factor: the one a paint of both axes is as sharp as.
    #[must_use]
    pub const fn coarsest(self) -> f32 {
        if self.x < self.y { self.x } else { self.y }
    }

    /// Both factors multiplied by `factor`.
    #[must_use]
    pub const fn scaled(self, factor: f32) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
        }
    }

    /// This scale kept between `low` and `high` times `fit`, factor by factor.
    ///
    /// The bound is relative to the fit because a factor counts output pixels
    /// per *surface unit*, and a surface unit is the plane's own: a document may
    /// measure its plane in slot indices, prices or points, so the factor a fit
    /// lands on can be a thousandth or a thousand. A bound in absolute pixels
    /// per unit would be a different zoom on every document, and on most of them
    /// a step that jumps.
    #[must_use]
    pub fn within(self, fit: Self, low: f32, high: f32) -> Self {
        Self {
            x: self.x.clamp(fit.x * low, fit.x * high),
            y: self.y.clamp(fit.y * low, fit.y * high),
        }
    }

    /// This scale eased `alpha` of the way toward `target`, factor by factor.
    ///
    /// `alpha` of one lands on the target, and a factor that arrived within a
    /// thousandth of its target snaps to it, so an ease settles instead of
    /// approaching for ever.
    #[must_use]
    pub fn eased(self, target: Self, alpha: f32) -> Self {
        let one = |from: f32, to: f32| {
            let moved = from + (to - from) * alpha;
            if (to - moved).abs() < to.abs() * 0.001 {
                to
            } else {
                moved
            }
        };
        Self {
            x: one(self.x, target.x),
            y: one(self.y, target.y),
        }
    }
}

/// One window mapping of a plane: the numbers the draw uses, bundled.
///
/// The picture centers at the window middle plus the pan, scales by `scale` on
/// each axis and turns clockwise by `angle` degrees.
/// [`Frame::to_screen`] and [`Frame::to_image`] are the one conversion pair,
/// and each side of the pair carries the tag of its space, so a source point
/// and a window point never pass for one another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// The plane size, in source pixels.
    pub img: SurfaceSize,
    /// The window size, in window pixels.
    pub win: WindowSize,
    /// The offset of the plane center from the window middle, in window
    /// pixels.
    pub pan: WindowVector,
    /// Output pixels per source pixel, on each axis.
    pub scale: Scale,
    /// The clockwise turn, in degrees.
    pub angle: f32,
}

impl Frame {
    /// The window position of a source point, matching the draw transform.
    ///
    /// The image center sits at the window middle plus the pan, the image
    /// scales around that point, and a positive `angle` turns it clockwise.
    #[must_use]
    pub fn to_screen(&self, point: SurfacePoint) -> WindowPoint {
        let (sin, cos) = self.angle.to_radians().sin_cos();
        let dx = (point.x - self.img.width / 2.0) * self.scale.x;
        let dy = (point.y - self.img.height / 2.0) * self.scale.y;
        WindowPoint::new(
            self.win.width / 2.0 + self.pan.x + dx * cos - dy * sin,
            self.win.height / 2.0 + self.pan.y + dx * sin + dy * cos,
        )
    }

    /// The source point under a window position, the inverse of
    /// [`Self::to_screen`].
    ///
    /// A zero or non-finite scale factor reads as one, so the mapping never
    /// divides by zero.
    #[must_use]
    pub fn to_image(&self, point: WindowPoint) -> SurfacePoint {
        let scale = self.scale.drawable();
        let (sin, cos) = self.angle.to_radians().sin_cos();
        let dx = point.x - self.win.width / 2.0 - self.pan.x;
        let dy = point.y - self.win.height / 2.0 - self.pan.y;
        SurfacePoint::new(
            self.img.width / 2.0 + (dx * cos + dy * sin) / scale.x,
            self.img.height / 2.0 + (-dx * sin + dy * cos) / scale.y,
        )
    }

    /// The pan that keeps the surface point under `anchor` where it is while
    /// the scale runs from this frame's to `to`.
    ///
    /// The anchor is a window point, and each axis solves its own ratio, so the
    /// point under the cursor stays under it even when the two factors differ:
    /// a wheel step on a view that states its viewport moves the picture without
    /// sliding the anchored point away.
    /// The solve reads the pan as the placement of the plane's centre and not
    /// the turn, which is what the zoom ease has always done: a turned view
    /// anchors the point the plain mapping names.
    #[must_use]
    pub fn anchored(&self, to: Scale, anchor: WindowPoint) -> WindowVector {
        let (from, to) = (self.scale.drawable(), to.drawable());
        WindowVector::new(
            hold_axis(
                anchor.x,
                from.x,
                to.x,
                self.pan.x,
                self.win.width,
                self.img.width,
            ),
            hold_axis(
                anchor.y,
                from.y,
                to.y,
                self.pan.y,
                self.win.height,
                self.img.height,
            ),
        )
    }

    /// The plane as a crop: the whole of `img` at the origin.
    #[must_use]
    pub const fn plane(&self) -> Crop {
        crop(0.0, 0.0, self.img.width, self.img.height)
    }

    /// Whether the frame names a plane the view can draw.
    #[must_use]
    pub fn has_drawable_plane(&self) -> bool {
        self.img.width > 0.0 && self.img.height > 0.0
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
        if !(self.has_drawable_plane() && self.win.width > 0.0 && self.win.height > 0.0) {
            return crop(0.0, 0.0, 0.0, 0.0);
        }
        let (x0, y0) = (-margin * self.win.width, -margin * self.win.height);
        let (x1, y1) = (
            self.win.width * (1.0 + margin),
            self.win.height * (1.0 + margin),
        );
        let corners = [
            WindowPoint::new(x0, y0),
            WindowPoint::new(x1, y0),
            WindowPoint::new(x1, y1),
            WindowPoint::new(x0, y1),
        ];
        let (mut min, mut max) = (
            WindowPoint::new(f32::INFINITY, f32::INFINITY),
            WindowPoint::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        );
        for corner in corners {
            let point = self.to_image(corner);
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
        }
        let x = min.x.clamp(0.0, self.img.width);
        let y = min.y.clamp(0.0, self.img.height);
        let right = max.x.clamp(x, self.img.width);
        let bottom = max.y.clamp(y, self.img.height);
        crop(x, y, right - x, bottom - y)
    }
}

/// The pan of one axis that holds the surface point under `anchor` where it is
/// while the factor runs from `from` to `to`.
///
/// The window offset of the plane's corner is `(win - img * factor) / 2 + pan`,
/// so the anchored point keeps its place while the ratio of the two factors
/// scales its distance from that corner, and the new corner is taken out again.
fn hold_axis(anchor: f32, from: f32, to: f32, pan: f32, win: f32, img: f32) -> f32 {
    let edge = |factor: f32| (win - img * factor) / 2.0;
    let held = anchor - (anchor - edge(from) - pan) * (to / from);
    held - edge(to)
}

/// A rect at the scale its pixels were rasterized at.
///
/// [`worth_painting`] and [`patch_reaches_window`] compare two of these, so a
/// rect never
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
pub fn worth_painting(needed: Patch, painted: Option<Patch>) -> bool {
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
pub fn patch_reaches_window(needed: Patch, painted: Patch) -> bool {
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
pub fn patch_covers_at_scale(needed: Patch, painted: Patch) -> bool {
    patch_reaches_window(needed, painted) && !mismatched_scale(needed, painted)
}

/// Whether two patches disagree about the scale beyond [`SCALE_SLACK`].
fn mismatched_scale(needed: Patch, painted: Patch) -> bool {
    needed.scale > painted.scale * SCALE_SLACK || needed.scale * SCALE_SLACK < painted.scale
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame over a 1024x512 plane inside an 800x600 window.
    fn frame(pan: WindowVector, scale: f32, angle: f32) -> Frame {
        Frame {
            img: SurfaceSize::new(1024.0, 512.0),
            win: WindowSize::new(800.0, 600.0),
            pan,
            scale: Scale::uniform(scale),
            angle,
        }
    }

    #[test]
    fn a_fit_scale_covers_the_whole_document() {
        let whole = frame(WindowVector::zero(), 600.0 / 1024.0, 0.0).crop(0.0);
        assert!((whole.width() - 1024.0).abs() < 1.0);
        assert!((whole.height() - 512.0).abs() < 1.0);
        assert!(whole.min_x().abs() < 1.0 && whole.min_y().abs() < 1.0);
    }

    #[test]
    fn a_zoomed_crop_covers_about_one_window() {
        let zoomed = frame(WindowVector::zero(), 4.0, 0.0).crop(0.0);
        // 800/4 by 600/4 source pixels, centered.
        assert!((zoomed.width() - 200.0).abs() < 1e-3);
        assert!((zoomed.height() - 150.0).abs() < 1e-3);
        assert!((zoomed.min_x() - 412.0).abs() < 1e-3);
        assert!((zoomed.min_y() - 181.0).abs() < 1e-3);
    }

    #[test]
    fn the_margin_grows_the_rect_and_the_clip_bounds_it() {
        let plain = frame(WindowVector::zero(), 4.0, 0.0).crop(0.0);
        let grown = frame(WindowVector::zero(), 4.0, 0.0).crop(MARGIN);
        assert!(grown.contains_rect(&plain));
        assert!(grown.width() > plain.width() && grown.height() > plain.height());
        // A pan far outside the document still clips to it.
        let out = frame(WindowVector::new(9000.0, 9000.0), 1.0, 0.0).crop(MARGIN);
        assert!(out.width() <= 1024.0 && out.height() <= 512.0);
        assert!(out.min_x() >= 0.0 && out.min_y() >= 0.0);
    }

    #[test]
    fn the_two_axes_carry_their_own_scale() {
        // A view that states the box its window shows fills the window on each
        // axis, so the two factors differ: 800 window pixels of 100 surface
        // units across, and 600 of 400 down.
        let squashed = Frame {
            scale: Scale::axes(8.0, 1.5),
            ..frame(WindowVector::zero(), 1.0, 0.0)
        };
        let shown = squashed.crop(0.0);
        assert!((shown.width() - 100.0).abs() < 1e-3);
        assert!((shown.height() - 400.0).abs() < 1e-3);
        // The plane center is the window middle, whichever factor each axis
        // carries.
        let middle = squashed.to_screen(SurfacePoint::new(512.0, 256.0));
        assert!((middle.x - 400.0).abs() < 1e-3);
        assert!((middle.y - 300.0).abs() < 1e-3);
        // A degenerate factor reads as one, so the inverse still answers a
        // point instead of an infinity.
        let flat = Frame {
            scale: Scale::axes(0.0, f32::NAN),
            ..frame(WindowVector::zero(), 1.0, 0.0)
        };
        let back = flat.to_image(WindowPoint::new(400.0, 300.0));
        assert_eq!((back.x, back.y), (512.0, 256.0));
    }

    #[test]
    fn a_zoom_band_is_relative_to_the_fit() {
        // A document may measure its plane in units that fit at a thousandth of
        // a pixel per unit, and a step of one wheel notch is still one factor:
        // the band bounds the fit and not an absolute factor.
        let fit = Scale::axes(5.12e-4, 7.68e-4);
        let step = fit.scaled(1.25);
        assert_eq!(step.within(fit, 0.01, 100.0), step);
        // Past the band the factor holds at the bound, axis by axis.
        let far = fit.scaled(1000.0).within(fit, 0.01, 100.0);
        assert!((far.x - fit.x * 100.0).abs() < 1e-9);
        assert!((far.y - fit.y * 100.0).abs() < 1e-9);
        let back = fit.scaled(0.001).within(fit, 0.01, 100.0);
        assert!((back.x - fit.x * 0.01).abs() < 1e-9);
        // Each axis takes its own bound, so the ratio the fit chose survives.
        assert!((far.x / far.y - fit.x / fit.y).abs() < 1e-6);
    }

    #[test]
    fn an_anchor_holds_its_point_under_either_scale() {
        // A frame whose two axes carry different factors, zoomed to another
        // pair that differs again: the surface point under the anchor keeps its
        // window spot on both axes, because each axis solves its own ratio.
        let frame = frame(WindowVector::new(12.0, -7.0), 3.0, 0.0);
        let frame = Frame {
            scale: Scale::axes(3.0, 5.0),
            ..frame
        };
        let point = SurfacePoint::new(300.0, 200.0);
        let anchor = frame.to_screen(point);
        for to in [
            Scale::axes(6.0, 2.5),
            Scale::axes(1.0, 9.0),
            Scale::uniform(4.0),
        ] {
            let pan = frame.anchored(to, anchor);
            let moved = Frame {
                pan,
                scale: to,
                ..frame
            };
            let back = moved.to_screen(point);
            assert!(
                (back.x - anchor.x).abs() < 1e-3 && (back.y - anchor.y).abs() < 1e-3,
                "the anchored point stays at {anchor:?} under {to:?}, and lands on {back:?}"
            );
        }
    }

    #[test]
    fn a_degenerate_frame_answers_an_empty_crop() {
        // A plane the view cannot draw, and a window with no pixels, both
        // answer an empty rect instead of a division by zero.
        let flat = Frame {
            img: SurfaceSize::zero(),
            ..frame(WindowVector::zero(), 1.0, 0.0)
        };
        assert!(flat.crop(0.0).is_empty());
        let blind = Frame {
            win: WindowSize::zero(),
            ..frame(WindowVector::zero(), 1.0, 0.0)
        };
        assert!(blind.crop(0.0).is_empty());
    }

    #[test]
    fn the_paint_rule_covers_scale_and_coverage() {
        let needed = Patch::new(frame(WindowVector::zero(), 4.0, 0.0).crop(0.0), 4.0);
        let painted = Patch::new(frame(WindowVector::zero(), 4.0, 0.0).crop(MARGIN), 4.0);
        // A fresh region and a covered one.
        assert!(worth_painting(needed, None));
        assert!(!worth_painting(needed, Some(painted)));
        // A scale inside the float tolerance stays: the patch still lands one
        // pixel per pixel.
        assert!(!worth_painting(
            needed,
            Some(Patch {
                scale: 4.004,
                ..painted
            })
        ));
        // Any other scale stretch softens every pixel, so it pays.
        assert!(worth_painting(
            needed,
            Some(Patch {
                scale: 4.1,
                ..painted
            })
        ));
        assert!(worth_painting(
            needed,
            Some(Patch {
                scale: 3.9,
                ..painted
            })
        ));
        let far = Patch::new(
            frame(WindowVector::new(300.0, 0.0), 4.0, 0.0).crop(0.0),
            4.0,
        );
        assert!(worth_painting(far, Some(painted)));
        // An empty need never pays.
        let none = Patch::new(crop(0.0, 0.0, 0.0, 0.0), 4.0);
        assert!(!worth_painting(none, None));
    }

    #[test]
    fn a_patch_covers_the_region_it_holds() {
        let needed = Patch::new(frame(WindowVector::zero(), 4.0, 0.0).crop(0.0), 4.0);
        let painted = Patch::new(frame(WindowVector::zero(), 4.0, 0.0).crop(MARGIN), 4.0);
        // The margin holds the window at the same scale.
        assert!(patch_covers_at_scale(needed, painted));
        // A scale past the slack, and a rect that misses the window, fail.
        assert!(!patch_covers_at_scale(
            needed,
            Patch {
                scale: 4.1,
                ..painted
            }
        ));
        let far = Patch::new(
            frame(WindowVector::new(300.0, 0.0), 4.0, 0.0).crop(MARGIN),
            4.0,
        );
        assert!(!patch_covers_at_scale(needed, far));
        // The test and the paint rule never both fire.
        assert!(!(patch_covers_at_scale(needed, painted) && worth_painting(needed, Some(painted))));
    }

    #[test]
    fn a_patch_holds_only_the_window_it_reaches() {
        let wide = frame(WindowVector::zero(), 4.0, 0.0);
        let needed = Patch::new(wide.crop(0.0), 4.0);
        let covering = Patch::new(wide.crop(MARGIN), 4.0);
        assert!(patch_reaches_window(needed, covering));
        // Zoomed out, the window shows more than the patch painted, so the
        // patch holds a part and the whole image must draw under it.
        let out = frame(WindowVector::zero(), 1.0, 0.0);
        let wide_need = Patch::new(out.crop(0.0), 1.0);
        assert!(!patch_reaches_window(wide_need, covering));
        // A patch at another scale still holds the rect it reaches, and the
        // scale is `patch_covers_at_scale`'s business alone.
        assert!(patch_reaches_window(
            needed,
            Patch {
                scale: 8.0,
                ..covering
            }
        ));
        assert!(!patch_covers_at_scale(
            needed,
            Patch {
                scale: 8.0,
                ..covering
            }
        ));
        // An empty need is never held.
        let none = Patch::new(crop(0.0, 0.0, 0.0, 0.0), 4.0);
        assert!(!patch_reaches_window(none, covering));
    }
}

#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    use super::*;

    fn view() -> impl Strategy<Value = (f32, f32, f32, f32, f32)> {
        (
            1.0f32..64.0,       // scale across
            1.0f32..64.0,       // scale down
            -360.0f32..360.0,   // angle
            -4000.0f32..4000.0, // pan x
            -4000.0f32..4000.0, // pan y
        )
    }

    /// A frame over a 1024x512 plane inside an 800x600 window.
    fn frame(pan: WindowVector, scale: Scale, angle: f32) -> Frame {
        Frame {
            img: SurfaceSize::new(1024.0, 512.0),
            win: WindowSize::new(800.0, 600.0),
            pan,
            scale,
            angle,
        }
    }

    /// One frame passes every geometric rule the properties below hold.
    fn assert_invariants(frame: Frame) {
        let painted = frame.crop(MARGIN);
        let needed = frame.crop(0.0);
        assert!(painted.contains_rect(&needed));
        assert!(painted.min_x() >= 0.0 && painted.min_y() >= 0.0);
        assert!(painted.max_x() <= 1024.0 + 1e-3);
        assert!(painted.max_y() <= 512.0 + 1e-3);
        let (w, h) = (frame.win.width, frame.win.height);
        let limit = (1.0 + 2.0 * MARGIN) * (w * w + h * h).sqrt();
        assert!(painted.width() * frame.scale.x <= limit * (1.0 + 1e-3));
        assert!(painted.height() * frame.scale.y <= limit * (1.0 + 1e-3));
    }

    /// A past failure of the paint bounds, at the pan that reached them.
    #[test]
    fn a_pan_down_keeps_the_paint_in_bounds() {
        assert_invariants(frame(
            WindowVector::new(0.0, -782.28406),
            Scale::uniform(1.0),
            0.0,
        ));
    }

    /// The failure that the diagonal bound of the paint area shrank to: the
    /// turned corner touches the window diagonal. The two values stay here,
    /// because the persistence of the seeds is off.
    #[test]
    fn a_deep_turned_zoom_keeps_the_paint_in_bounds() {
        assert_invariants(frame(
            WindowVector::new(0.0, 0.0),
            Scale::uniform(22.561_317),
            216.2562,
        ));
    }

    proptest! {
        // No seed file: a saved case becomes a plain `#[test]` above, so the
        // suite reads its regressions in one place.
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        /// The two mappings undo each other, so the region lines up with the
        /// draw, on each axis at its own factor.
        #[test]
        fn the_mapping_round_trips(
            (sx, sy, angle, px, py) in view(),
            x in 0.0f32..1024.0,
            y in 0.0f32..512.0,
        ) {
            let frame = frame(WindowVector::new(px, py), Scale::axes(sx, sy), angle);
            let point = SurfacePoint::new(x, y);
            let back = frame.to_image(frame.to_screen(point));
            prop_assert!((back.x - point.x).abs() < 1e-2);
            prop_assert!((back.y - point.y).abs() < 1e-2);
        }

        /// Every window corner lands inside the region the view paints, so the
        /// patch always covers the screen.
        #[test]
        fn the_crop_covers_the_window((sx, sy, angle, px, py) in view()) {
            let frame = frame(WindowVector::new(px, py), Scale::axes(sx, sy), angle);
            let painted = frame.crop(MARGIN);
            let needed = frame.crop(0.0);
            prop_assert!(painted.contains_rect(&needed));
            prop_assert!(painted.min_x() >= 0.0 && painted.min_y() >= 0.0);
            prop_assert!(painted.max_x() <= 1024.0 + 1e-3);
            prop_assert!(painted.max_y() <= 512.0 + 1e-3);
        }

        /// The painted area stays bounded by the window, so a deeper zoom
        /// costs no more pixels than a shallow one.
        #[test]
        fn the_paint_area_stays_bounded((sx, sy, angle, px, py) in view()) {
            let frame = frame(WindowVector::new(px, py), Scale::axes(sx, sy), angle);
            let painted = frame.crop(MARGIN);
            // A turned rect grows its bounding box by at most its own
            // diagonal, so that diagonal bounds both axes.
            let (w, h) = (frame.win.width, frame.win.height);
            let limit = (1.0 + 2.0 * MARGIN) * (w * w + h * h).sqrt();
            // A window turned toward its diagonal reaches the bound exactly,
            // and the rounding of the sine and cosine pushes the product past
            // it by a few floating-point steps. The slack covers that rounding
            // and nothing else.
            prop_assert!(painted.width() * sx <= limit * (1.0 + 1e-3));
            prop_assert!(painted.height() * sy <= limit * (1.0 + 1e-3));
        }
    }
}
