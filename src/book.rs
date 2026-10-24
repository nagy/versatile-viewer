//! A document as a book: every spread, one below the next, in one surface
//! plane.
//!
//! A [`BookSurface`] wraps a multi-plane source (a PDF) and shows it the way a
//! book reads: a cover page alone in the right slot of the first spread, then
//! the pages in pairs, each spread stacked directly under the one before.
//! The whole book is a single [`Surface`] plane, so the loader, the fit, the
//! pan, the turn and the visible-region patch treat it as one tall picture,
//! with no change.
//!
//! Every spread measures the widest and tallest page of the document, doubled
//! plus [`GUTTER`], so the pages of one spread line up in two columns down the
//! whole book.
//! A narrower page centers inside its slot, and a spread that holds one page
//! leaves the other slot transparent, so the window background shows through.
//!
//! The book keeps its own unit: **base units**, not points.
//! A document of a thousand pages stands far taller than one texture, and the
//! main thread never resizes a buffer, so the surface bakes a base scale
//! ([`BookSurface::base`]) that lands the whole book inside
//! [`MAX_TEXTURE_SIDE`] and [`MAX_RENDER_PIXELS`] at scale 1.0.
//! `bounds` answers in base units, so a scale-1.0 render matches it, and a
//! region patch at a settled zoom converts back to points and rasterizes only
//! the page parts the window shows.
//!
//! The grid never uses this wrapper: a cell shows page 0 and nothing else.
//!
//! [`BookSurface::spread_count`] and [`BookSurface::spread_box`] publish the
//! geometry the image view needs to step from one spread to the next, so the
//! step reads the book's own numbers instead of guessing them from the plane.
//!
//! The constants at the top of the file hold every book setting, so a
//! configuration system can lift them later without a code motion.

use std::sync::Arc;

use crate::{
    DecodedImage, MAX_TEXTURE_SIDE,
    dim::{ImgH, ImgW},
    surface::{Region, Size, Surface, SurfaceError, View},
};

/// The pages before the first pair.
///
/// Each one sits alone in its own spread, like a book cover.
/// Zero pairs every page, so page 0 shares the first spread with page 1.
pub const COVER_SLOTS: usize = 1;

/// Whether a cover page takes the right slot of its spread.
///
/// True for a left-to-right book, where the cover faces the window edge.
pub const COVER_ON_RIGHT: bool = true;

/// The gap between the two pages of one spread, in points.
///
/// The gap stays part of the book, so the pair reads as two pages and the
/// window background shows through.
pub const GUTTER: f32 = 8.0;

/// The gap between two spreads, in points.
///
/// Zero stacks the spreads edge to edge, so the pages touch and the book reads
/// as one continuous sheet.
pub const ROW_GAP: f32 = 0.0;

/// The slots of one spread, left and right.
pub const SLOTS: usize = 2;

/// The most output pixels one render makes.
///
/// A page backend clamps a huge render on its own, and two pages that clamp
/// apart would disagree in size, so the book clamps first and paints every
/// page at the one scale that fits.
pub const MAX_RENDER_PIXELS: f32 = 64.0 * 1024.0 * 1024.0;

/// A multi-plane source shown as a column of book spreads.
///
/// See the [module documentation](self).
pub struct BookSurface {
    inner: Arc<dyn Surface>,
    /// The widest and tallest page of the document, which sizes one spread.
    slot: Size,
    /// The whole book in points: a column of spreads.
    canvas: Size,
    /// How many spreads the book stacks.
    rows: usize,
    /// Base units per point, so the whole book at scale 1.0 fits one texture.
    base: f32,
}

impl std::fmt::Debug for BookSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BookSurface")
            .field("pages", &self.inner.plane_count())
            .field("slot", &self.slot)
            .field("canvas", &self.canvas)
            .field("rows", &self.rows)
            .field("base", &self.base)
            .finish()
    }
}

impl BookSurface {
    /// Wrap `inner`, measure every page, and bake the base scale.
    ///
    /// A page that cannot measure keeps the others usable.
    ///
    /// # Errors
    ///
    /// The function errors when the source has no planes, when no plane
    /// measures, or when the measured canvas stays empty.
    pub fn new(inner: Arc<dyn Surface>) -> Result<BookSurface, SurfaceError> {
        let planes = inner.plane_count();
        if planes == 0 {
            return Err(SurfaceError::Plane(0));
        }
        let mut slot = Size { w: 0.0, h: 0.0 };
        let mut failure = None;
        for plane in 0..planes {
            match inner.bounds(plane) {
                Ok(size) => {
                    slot.w = slot.w.max(size.w);
                    slot.h = slot.h.max(size.h);
                }
                Err(err) => {
                    if failure.is_none() {
                        failure = Some(err);
                    }
                }
            }
        }
        if let Some(err) = failure
            && (slot.w <= 0.0 || slot.h <= 0.0)
        {
            return Err(err);
        }
        if !(slot.w > 0.0 && slot.h > 0.0) {
            return Err(SurfaceError::Unsupported("empty document".to_string()));
        }
        let rows = Self::spread_count(planes);
        let canvas = Size {
            w: slot.w * SLOTS as f32 + GUTTER * (SLOTS - 1) as f32,
            h: rows as f32 * slot.h + ROW_GAP * (rows - 1) as f32,
        };
        // The base scale keeps the whole-book render uploadable as one
        // texture, and under the render cap.
        let long = canvas.w.max(canvas.h);
        let base = 1.0_f32
            .min(MAX_TEXTURE_SIDE as f32 / long)
            .min((MAX_RENDER_PIXELS / (canvas.w * canvas.h)).sqrt());
        Ok(BookSurface {
            inner,
            slot,
            canvas,
            rows,
            base,
        })
    }

    /// The wrapped source, so a caller can ask what the book holds.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn Surface> {
        &self.inner
    }

    /// How many spreads `planes` pages make, with [`COVER_SLOTS`] alone first.
    #[must_use]
    pub fn spread_count(planes: usize) -> usize {
        let cover = COVER_SLOTS.min(planes);
        cover + (planes - cover).div_ceil(SLOTS)
    }

    /// How many spreads the book stacks.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Base units per point.
    ///
    /// One point of the document measures this many of the plane's units, so
    /// a scale-1.0 render of the whole book fits one texture.
    #[must_use]
    pub fn base(&self) -> f32 {
        self.base
    }

    /// The canvas of the book in points, before the base scale.
    #[must_use]
    pub fn canvas(&self) -> Size {
        self.canvas
    }

    /// The plane size: the canvas in base units.
    #[must_use]
    fn size(&self) -> Size {
        Size {
            w: self.canvas.w * self.base,
            h: self.canvas.h * self.base,
        }
    }

    /// The first spread, in base units: the cover page alone, in the right
    /// slot.
    ///
    /// This is the box a fresh open frames, so the cover page shows whole.
    #[must_use]
    pub fn first_spread(&self) -> Size {
        Size {
            w: (self.slot.w * SLOTS as f32 + GUTTER * (SLOTS - 1) as f32) * self.base,
            h: self.slot.h * self.base,
        }
    }

    /// The pages of spread `spread`, by slot (left, right).
    ///
    /// A slot holds `None` when the document has no page for it.
    ///
    /// # Errors
    ///
    /// The function errors when the spread index is out of range.
    fn pages(&self, spread: usize) -> Result<[Option<usize>; SLOTS], SurfaceError> {
        if spread >= self.rows {
            return Err(SurfaceError::Plane(spread));
        }
        let planes = self.inner.plane_count();
        let cover = COVER_SLOTS.min(planes);
        let mut pages = [None; SLOTS];
        if spread < cover {
            pages[usize::from(COVER_ON_RIGHT)] = Some(spread);
        } else {
            let left = cover + (spread - cover) * SLOTS;
            pages[0] = Some(left);
            pages[1] = (left + 1 < planes).then_some(left + 1);
        }
        Ok(pages)
    }

    /// The top-left corner of a page of `size` inside slot `side` of `row`, in
    /// points.
    ///
    /// The page centers in its slot, so a narrower page leaves an even border
    /// on both sides.
    fn page_origin(&self, row: usize, side: usize, size: Size) -> (f32, f32) {
        let slot_x = side as f32 * (self.slot.w + GUTTER);
        let slot_y = row as f32 * (self.slot.h + ROW_GAP);
        (
            slot_x + (self.slot.w - size.w) / 2.0,
            slot_y + (self.slot.h - size.h) / 2.0,
        )
    }

    /// Paint `region` (in base units) at `scale` output pixels per base unit.
    fn paint(&self, region: Region, scale: f32) -> Result<DecodedImage, SurfaceError> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(SurfaceError::Scale(scale));
        }
        let rect = intersect(region, self.size().whole()).ok_or(SurfaceError::Buffer)?;
        // The request in points, and the output pixels per point, under the
        // render cap.
        let (x0, y0) = (rect.x / self.base, rect.y / self.base);
        let (w, h) = (rect.w / self.base, rect.h / self.base);
        let scale = (scale * self.base).min((MAX_RENDER_PIXELS / (w * h)).sqrt());
        let width = (w * scale).floor().max(1.0) as u32;
        let height = (h * scale).floor().max(1.0) as u32;
        // Transparent first: a slot with no page, and the gaps, show the window
        // background.
        let mut out = vec![0u8; width as usize * height as usize * 4];
        // Only the spreads the request touches cost a render.
        let row_h = self.slot.h + ROW_GAP;
        let first = (y0 / row_h).floor().max(0.0) as usize;
        let last = ((y0 + h) / row_h).ceil().max(0.0) as usize;
        for row in first..last.min(self.rows) {
            for (side, page) in self.pages(row)?.iter().enumerate() {
                let Some(page) = *page else {
                    continue;
                };
                let page_size = self.inner.bounds(page)?;
                let (origin_x, origin_y) = self.page_origin(row, side, page_size);
                let on_canvas = Region {
                    x: origin_x,
                    y: origin_y,
                    w: page_size.w,
                    h: page_size.h,
                };
                let Some(hit) = intersect(on_canvas, Region { x: x0, y: y0, w, h }) else {
                    continue;
                };
                // The request in page points, moved to the page origin.
                let part = Region {
                    x: hit.x - origin_x,
                    y: hit.y - origin_y,
                    w: hit.w,
                    h: hit.h,
                };
                let image = self.inner.render(View::region(page, part, scale))?;
                let dx = ((hit.x - x0) * scale).round().max(0.0) as u32;
                let dy = ((hit.y - y0) * scale).round().max(0.0) as u32;
                blit(&mut out, width, height, &image, dx, dy)?;
            }
        }
        Ok(DecodedImage {
            width: ImgW::new(width),
            height: ImgH::new(height),
            data: out,
        })
    }
}

impl Surface for BookSurface {
    fn plane_count(&self) -> usize {
        1
    }

    fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
        if plane != 0 {
            return Err(SurfaceError::Plane(plane));
        }
        Ok(self.size())
    }

    fn render_whole(&self, plane: usize, scale: f32) -> Result<DecodedImage, SurfaceError> {
        let region = self.bounds(plane)?.whole();
        self.paint(region, scale)
    }

    fn render(&self, view: View) -> Result<DecodedImage, SurfaceError> {
        // The region anchor: only the page parts the window asked for.
        if view.plane != 0 {
            return Err(SurfaceError::Plane(view.plane));
        }
        let region = view.region.unwrap_or_else(|| self.size().whole());
        self.paint(region, view.scale)
    }

    fn progressive(&self) -> Result<Option<Box<dyn crate::surface::ChunkSource>>, SurfaceError> {
        // The byte stream of the inner source holds pages, not the rendered
        // book, so a book has no progressive path.
        Ok(None)
    }

    fn previews(&self) -> bool {
        // Rasterizing every page takes a moment, so a cheap preview lands
        // before the full book.
        self.inner.previews()
    }

    fn can_refine(&self) -> bool {
        self.inner.can_refine()
    }

    fn open_box(&self, plane: usize) -> Option<Size> {
        // The reader starts on the cover spread, not on the whole book.
        (plane == 0).then(|| self.first_spread())
    }

    fn spread_count(&self) -> usize {
        self.rows
    }

    fn spread_box(&self, spread: usize) -> Option<Region> {
        if spread >= self.rows {
            return None;
        }
        // Every spread spans the whole canvas and stands one page tall, so a
        // step from one to the next moves the pan by one row.
        Some(Region {
            x: 0.0,
            y: spread as f32 * (self.slot.h + ROW_GAP) * self.base,
            w: self.canvas.w * self.base,
            h: self.slot.h * self.base,
        })
    }
}

/// The overlap of two regions, or `None` when they miss.
fn intersect(a: Region, b: Region) -> Option<Region> {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = (a.x + a.w).min(b.x + b.w);
    let bottom = (a.y + a.h).min(b.y + b.h);
    (right > x && bottom > y).then_some(Region {
        x,
        y,
        w: right - x,
        h: bottom - y,
    })
}

/// Copy `src` into the `dw` x `dh` buffer at (`dx`, `dy`), clipped to it.
///
/// The pages of a book never overlap, so the copy overwrites instead of
/// blending.
fn blit(
    dst: &mut [u8],
    dw: u32,
    dh: u32,
    src: &DecodedImage,
    dx: u32,
    dy: u32,
) -> Result<(), SurfaceError> {
    let (sw, sh) = (src.width.get(), src.height.get());
    if src.data.len() != sw as usize * sh as usize * 4 {
        return Err(SurfaceError::Buffer);
    }
    if dx >= dw || dy >= dh {
        return Ok(());
    }
    let w = sw.min(dw - dx);
    let h = sh.min(dh - dy);
    let row = w as usize * 4;
    for line in 0..h {
        let from = (line * sw) as usize * 4;
        let to = (((dy + line) * dw + dx) as usize) * 4;
        dst[to..to + row].copy_from_slice(&src.data[from..from + row]);
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the asserted values are exact book arithmetic
mod tests {
    use super::*;

    /// A document of solid pages, so a test can name a page by its pixels.
    struct Pages(Vec<(f32, f32)>);

    impl Surface for Pages {
        fn plane_count(&self) -> usize {
            self.0.len()
        }

        fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
            let (w, h) = *self.0.get(plane).ok_or(SurfaceError::Plane(plane))?;
            Ok(Size { w, h })
        }

        fn render_whole(&self, plane: usize, scale: f32) -> Result<DecodedImage, SurfaceError> {
            let size = self.bounds(plane)?;
            let width = (size.w * scale).round().max(1.0) as u32;
            let height = (size.h * scale).round().max(1.0) as u32;
            let red = ((10 * (plane + 1)) % 256) as u8;
            let mut data = vec![0u8; width as usize * height as usize * 4];
            for px in data.as_chunks_mut::<4>().0 {
                px.copy_from_slice(&[red, 0, 0, 255]);
            }
            Ok(DecodedImage {
                width: ImgW::new(width),
                height: ImgH::new(height),
                data,
            })
        }

        fn progressive(
            &self,
        ) -> Result<Option<Box<dyn crate::surface::ChunkSource>>, SurfaceError> {
            Ok(None)
        }
    }

    fn book(pages: Vec<(f32, f32)>) -> BookSurface {
        BookSurface::new(Arc::new(Pages(pages))).expect("a book")
    }

    /// The RGBA pixel at (`x`, `y`) of a render.
    fn pixel(img: &DecodedImage, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * img.width.get() + x) as usize) * 4;
        img.data[i..i + 4].try_into().expect("four channels")
    }

    #[test]
    fn a_cover_page_takes_a_spread_of_its_own() {
        // One cover page, then pairs: 0 | 1,2 | 3,4.
        assert_eq!(BookSurface::spread_count(0), 0);
        assert_eq!(BookSurface::spread_count(1), 1);
        assert_eq!(BookSurface::spread_count(2), 2);
        assert_eq!(BookSurface::spread_count(3), 2);
        assert_eq!(BookSurface::spread_count(4), 3);
        assert_eq!(BookSurface::spread_count(5), 3);
        assert_eq!(BookSurface::spread_count(6), 4);
    }

    #[test]
    fn a_small_book_keeps_point_size_and_stacks_every_spread() {
        let b = book(vec![(100.0, 200.0), (120.0, 180.0)]);
        assert_eq!(b.base(), 1.0);
        assert_eq!(b.rows(), 2);
        // Three rows of the tallest page, two slots of the widest page.
        assert_eq!(b.bounds(0).unwrap(), Size { w: 248.0, h: 400.0 });
        assert_eq!(b.bounds(1).unwrap_err().to_string(), "plane out of range");
        assert_eq!(b.plane_count(), 1);
    }

    #[test]
    fn the_cover_sits_in_the_right_slot_of_the_first_row() {
        let b = book(vec![(100.0, 200.0), (100.0, 200.0), (100.0, 200.0)]);
        let img = b.render_whole(0, 1.0).unwrap();
        assert_eq!((img.width.get(), img.height.get()), (208, 400));
        // Row 0: the left slot stays transparent, the cover paints.
        assert_eq!(pixel(&img, 10, 100), [0, 0, 0, 0]);
        assert_eq!(pixel(&img, 110, 100), [10, 0, 0, 255]);
        // The gutter stays transparent.
        assert_eq!(pixel(&img, 101, 100), [0, 0, 0, 0]);
        // Row 1: pages 1 and 2, one under the cover spread.
        assert_eq!(pixel(&img, 10, 300), [20, 0, 0, 255]);
        assert_eq!(pixel(&img, 110, 300), [30, 0, 0, 255]);
    }

    #[test]
    fn a_region_takes_pixels_from_both_pages_and_both_rows() {
        let b = book(vec![
            (100.0, 200.0),
            (100.0, 200.0),
            (100.0, 200.0),
            (100.0, 200.0),
        ]);
        // Row 1 holds pages 1 and 2: a window across the seam sees both.
        let seam = Region {
            x: 90.0,
            y: 200.0,
            w: 30.0,
            h: 200.0,
        };
        let img = b.render(View::region(0, seam, 1.0)).unwrap();
        assert_eq!((img.width.get(), img.height.get()), (30, 200));
        assert_eq!(pixel(&img, 0, 100), [20, 0, 0, 255]);
        assert_eq!(pixel(&img, 29, 100), [30, 0, 0, 255]);
        // A window down the right column sees the page of the last row.
        let down = Region {
            x: 150.0,
            y: 380.0,
            w: 50.0,
            h: 20.0,
        };
        let img = b.render(View::region(0, down, 1.0)).unwrap();
        assert_eq!(pixel(&img, 10, 10), [30, 0, 0, 255]);
    }

    #[test]
    fn a_region_past_the_canvas_clips_instead_of_failing() {
        let b = book(vec![(100.0, 200.0), (100.0, 200.0)]);
        let region = Region {
            x: -10.0,
            y: -10.0,
            w: 50.0,
            h: 50.0,
        };
        let img = b.render(View::region(0, region, 1.0)).unwrap();
        assert_eq!((img.width.get(), img.height.get()), (40, 40));
        let far = Region {
            x: 1000.0,
            y: 1000.0,
            w: 10.0,
            h: 10.0,
        };
        assert!(b.render(View::region(0, far, 1.0)).is_err());
    }

    #[test]
    fn a_long_book_bakes_a_base_scale_that_fits_one_texture() {
        // 40 pages of a letter page make a book taller than any texture.
        let b = book(vec![(612.0, 792.0); 40]);
        assert!(b.base() < 1.0, "base {}", b.base());
        assert_eq!(b.rows(), 21);
        assert_eq!(b.bounds(0).unwrap().h, 792.0 * 21.0 * b.base());
        let img = b.render_whole(0, 1.0).unwrap();
        assert_eq!(img.height.get(), MAX_TEXTURE_SIDE);
        assert!(img.width.get() < img.height.get());
        // The patch at a settled zoom converts back to points, so the page
        // backend rasterizes 100 x 100 points of text.
        let region = Region {
            x: 0.0,
            y: 0.0,
            w: 100.0 * b.base(),
            h: 100.0 * b.base(),
        };
        let patch = b.render(View::region(0, region, 4.0)).unwrap();
        assert_eq!(
            patch.width.get(),
            (100.0 * b.base() * 4.0).floor().max(1.0) as u32
        );
    }

    #[test]
    fn a_huge_scale_clamps_to_the_render_cap() {
        let b = book(vec![(100.0, 200.0), (100.0, 200.0)]);
        let img = b.render_whole(0, 1000.0).unwrap();
        let pixels = u64::from(img.width.get()) * u64::from(img.height.get());
        assert!(
            pixels <= MAX_RENDER_PIXELS as u64,
            "pixels {pixels} past the cap"
        );
    }

    #[test]
    fn the_opening_box_is_the_first_spread() {
        let b = book(vec![(612.0, 792.0), (612.0, 792.0)]);
        // The whole plane stacks two spreads, so it stands twice as tall.
        assert_eq!(
            b.bounds(0).unwrap(),
            Size {
                w: 1232.0,
                h: 1584.0
            }
        );
        assert_eq!(
            b.open_box(0).unwrap(),
            Size {
                w: 1232.0,
                h: 792.0
            }
        );
        assert!(b.open_box(1).is_none());
    }

    #[test]
    fn the_spreads_stack_evenly_and_name_their_boxes() {
        let b = book(vec![(100.0, 200.0), (120.0, 180.0), (100.0, 200.0)]);
        assert_eq!(b.spread_count(), 2);
        let first = b.spread_box(0).unwrap();
        // The widest page of the document sizes both slots, so the spread
        // spans two of them plus the gutter.
        assert_eq!(
            first,
            Region {
                x: 0.0,
                y: 0.0,
                w: 248.0,
                h: 200.0
            }
        );
        // The pair spread sits one page under the cover spread.
        assert_eq!(
            b.spread_box(1).unwrap(),
            Region {
                x: 0.0,
                y: 200.0,
                w: 248.0,
                h: 200.0
            }
        );
        assert!(b.spread_box(2).is_none());
        // The opening box agrees with the first spread.
        assert_eq!(
            b.open_box(0).unwrap(),
            Size {
                w: first.w,
                h: first.h
            }
        );
    }

    #[test]
    fn a_step_aims_the_window_at_the_pages_of_one_spread() {
        // The cover page alone, then pages 1 and 2, then pages 3 and 4.
        let b = book(vec![(100.0, 200.0); 5]);
        let fit = b.open_box(0).unwrap();
        // A window of exactly one spread, so the fit scale is 1.
        let (win_w, win_h) = (fit.w, fit.h);
        let scale = (win_w / fit.w).min(win_h / fit.h);
        let plane = b.bounds(0).unwrap();
        // The pan of a spread step: it puts the top of the spread on the top
        // of the window (see `spread_pan` in `main.rs`).
        let spread_pan = |row: usize| {
            let spread = b.spread_box(row).unwrap();
            (plane.h * scale - win_h) / 2.0 - spread.y * scale
        };
        // The window, expressed as a region of the plane.
        let window = |pan: f32| Region {
            x: -((win_w - plane.w * scale) / 2.0) / scale,
            y: -((win_h - plane.h * scale) / 2.0 + pan) / scale,
            w: win_w / scale,
            h: win_h / scale,
        };
        // Spread 0 frames the cover page in the right slot.
        let cover = b
            .render(View::region(0, window(spread_pan(0)), scale))
            .unwrap();
        assert_eq!((cover.width.get(), cover.height.get()), (208, 200));
        assert_eq!(pixel(&cover, 10, 100), [0, 0, 0, 0]);
        assert_eq!(pixel(&cover, 110, 100), [10, 0, 0, 255]);
        // One step down shows the pair of pages 1 and 2, and the next step the
        // pair of pages 3 and 4.
        let pair = b
            .render(View::region(0, window(spread_pan(1)), scale))
            .unwrap();
        assert_eq!(pixel(&pair, 10, 100), [20, 0, 0, 255]);
        assert_eq!(pixel(&pair, 110, 100), [30, 0, 0, 255]);
        let last = b
            .render(View::region(0, window(spread_pan(2)), scale))
            .unwrap();
        assert_eq!(pixel(&last, 10, 100), [40, 0, 0, 255]);
        assert_eq!(pixel(&last, 110, 100), [50, 0, 0, 255]);
    }

    #[test]
    fn an_empty_document_does_not_wrap() {
        let err = BookSurface::new(Arc::new(Pages(Vec::new()))).unwrap_err();
        assert_eq!(err.to_string(), "plane out of range");
    }
}
