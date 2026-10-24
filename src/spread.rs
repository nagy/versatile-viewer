//! A document spread: the pages of a document side by side, as one surface
//! plane.
//!
//! A [`SpreadSurface`] wraps a multi-plane source (a PDF) and shows it the way
//! a book reads: a cover page alone in its own spread, then the pages in
//! pairs.
//! Every spread the wrapper exposes is one [`Surface`] plane, so the loader,
//! the image-view fit, the pan, the turn and the visible-region patch all work
//! on a spread without a change.
//!
//! The canvas measures the widest and tallest page of the document, doubled
//! plus [`GUTTER`], so the fit, the zoom and the pan stay put while n/p steps
//! from one spread to the next.
//! A narrower page centers inside its slot, and a spread that holds one page
//! leaves the other slot transparent, so the window background shows through.
//!
//! The canvas units stay the inner surface's units (PostScript points for a
//! PDF).
//! A render asks each page for the part of itself the canvas request covers,
//! at one shared scale, so a page backend with a region anchor (the PDF
//! rasterizer) pays only for the pixels the window shows.
//!
//! The grid never uses this wrapper: a cell shows page 0 and nothing else.
//!
//! The constants at the top of the file hold every spread setting, so a
//! configuration system can lift them later without a code motion.

use std::sync::Arc;

use crate::{
    DecodedImage,
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

/// The canvas gap between the two slots, in surface units (points for a PDF).
///
/// The gap stays part of the canvas, so the two pages never touch and the
/// window background shows through.
pub const GUTTER: f32 = 8.0;

/// The slots of one spread, left and right.
pub const SLOTS: usize = 2;

/// The most output pixels one spread render makes.
///
/// A page backend clamps a huge render on its own, and two pages that clamp
/// apart would disagree in size, so the spread clamps first and paints both
/// pages at the one scale that fits.
pub const MAX_RENDER_PIXELS: f32 = 64.0 * 1024.0 * 1024.0;

/// A multi-plane source shown as book spreads.
///
/// See the [module documentation](self).
pub struct SpreadSurface {
    inner: Arc<dyn Surface>,
    /// The widest and tallest page of the document, which sizes every canvas.
    slot: Size,
}

impl std::fmt::Debug for SpreadSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpreadSurface")
            .field("planes", &self.inner.plane_count())
            .field("slot", &self.slot)
            .finish()
    }
}

impl SpreadSurface {
    /// Wrap `inner`, and measure every page for the canvas.
    ///
    /// A page that cannot measure keeps the others usable.
    ///
    /// # Errors
    ///
    /// The function errors when the source has no planes, when no plane
    /// measures, or when the measured canvas stays empty.
    pub fn new(inner: Arc<dyn Surface>) -> Result<SpreadSurface, SurfaceError> {
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
        Ok(SpreadSurface { inner, slot })
    }

    /// The wrapped source, so a caller can ask what a spread holds.
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

    /// The canvas of every spread, in surface units.
    #[must_use]
    pub fn canvas(&self) -> Size {
        Size {
            w: self.slot.w * SLOTS as f32 + GUTTER * (SLOTS - 1) as f32,
            h: self.slot.h,
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
        if spread >= self.plane_count() {
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

    /// The top-left corner of a page of `size` inside slot `side`, in canvas
    /// units.
    ///
    /// The page centers in its slot, so a narrower page leaves an even border
    /// on both sides.
    fn page_origin(&self, side: usize, size: Size) -> (f32, f32) {
        let slot_x = side as f32 * (self.slot.w + GUTTER);
        (
            slot_x + (self.slot.w - size.w) / 2.0,
            (self.slot.h - size.h) / 2.0,
        )
    }

    /// Paint `region` of spread `spread` at `scale` output pixels per surface
    /// unit.
    fn paint(
        &self,
        spread: usize,
        region: Region,
        scale: f32,
    ) -> Result<DecodedImage, SurfaceError> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(SurfaceError::Scale(scale));
        }
        let pages = self.pages(spread)?;
        let rect = intersect(region, self.canvas().whole()).ok_or(SurfaceError::Buffer)?;
        // One scale for the whole canvas, under the spread cap.
        // The floor keeps the rounded corner counts inside the cap.
        let scale = scale.min((MAX_RENDER_PIXELS / (rect.w * rect.h)).sqrt());
        let width = (rect.w * scale).floor().max(1.0) as u32;
        let height = (rect.h * scale).floor().max(1.0) as u32;
        // Transparent first: a slot with no page shows the window background.
        let mut out = vec![0u8; width as usize * height as usize * 4];
        for (side, page) in pages.iter().enumerate() {
            let Some(page) = *page else {
                continue;
            };
            let size = self.inner.bounds(page)?;
            let (origin_x, origin_y) = self.page_origin(side, size);
            let on_canvas = Region {
                x: origin_x,
                y: origin_y,
                w: size.w,
                h: size.h,
            };
            let Some(hit) = intersect(on_canvas, rect) else {
                continue;
            };
            // The request in page units: the canvas rect moved to the page
            // origin.
            let part = Region {
                x: hit.x - origin_x,
                y: hit.y - origin_y,
                w: hit.w,
                h: hit.h,
            };
            let image = self.inner.render(View::region(page, part, scale))?;
            let dx = ((hit.x - rect.x) * scale).round().max(0.0) as u32;
            let dy = ((hit.y - rect.y) * scale).round().max(0.0) as u32;
            blit(&mut out, width, height, &image, dx, dy)?;
        }
        Ok(DecodedImage {
            width: ImgW::new(width),
            height: ImgH::new(height),
            data: out,
        })
    }
}

impl Surface for SpreadSurface {
    fn plane_count(&self) -> usize {
        Self::spread_count(self.inner.plane_count())
    }

    fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
        if plane >= self.plane_count() {
            return Err(SurfaceError::Plane(plane));
        }
        Ok(self.canvas())
    }

    fn render_whole(&self, plane: usize, scale: f32) -> Result<DecodedImage, SurfaceError> {
        // The spread rewrites the plane index, so the canvas is the plane.
        let canvas = self.bounds(plane)?.whole();
        self.paint(plane, canvas, scale)
    }

    fn render(&self, view: View) -> Result<DecodedImage, SurfaceError> {
        // The region anchor: only the pixels the caller asked for, from each
        // page the request covers.
        let region = view.region.unwrap_or(self.canvas().whole());
        self.paint(view.plane, region, view.scale)
    }

    fn progressive(&self) -> Result<Option<Box<dyn crate::surface::ChunkSource>>, SurfaceError> {
        // A spread carries no progressive path: the byte stream of the inner
        // source holds one page, not the pair.
        if self.plane_count() > 1 {
            return Ok(None);
        }
        self.inner.progressive()
    }

    fn previews(&self) -> bool {
        // The pages rasterize, so a cheap preview lands before the full
        // spread.
        self.inner.previews()
    }

    fn can_refine(&self) -> bool {
        self.inner.can_refine()
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
/// The pages of a spread never overlap, so the copy overwrites instead of
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
#[allow(clippy::float_cmp)] // the asserted values are exact spread arithmetic
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
            let red = 10 * (plane as u8 + 1);
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

    fn spread(pages: Vec<(f32, f32)>) -> SpreadSurface {
        SpreadSurface::new(Arc::new(Pages(pages))).expect("a spread")
    }

    /// The RGBA pixel at (`x`, `y`) of a render.
    fn pixel(img: &DecodedImage, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * img.width.get() + x) as usize) * 4;
        img.data[i..i + 4].try_into().expect("four channels")
    }

    #[test]
    fn a_cover_page_takes_a_spread_of_its_own() {
        // One cover page, then pairs: 0 | 1,2 | 3,4.
        assert_eq!(SpreadSurface::spread_count(0), 0);
        assert_eq!(SpreadSurface::spread_count(1), 1);
        assert_eq!(SpreadSurface::spread_count(2), 2);
        assert_eq!(SpreadSurface::spread_count(3), 2);
        assert_eq!(SpreadSurface::spread_count(4), 3);
        assert_eq!(SpreadSurface::spread_count(5), 3);
        assert_eq!(SpreadSurface::spread_count(6), 4);
    }

    #[test]
    fn the_canvas_holds_two_widest_pages_and_the_gutter() {
        let s = spread(vec![(100.0, 200.0), (120.0, 180.0)]);
        let canvas = s.canvas();
        assert_eq!(canvas.w, 120.0 * 2.0 + GUTTER);
        assert_eq!(canvas.h, 200.0);
        for plane in 0..s.plane_count() {
            assert_eq!(s.bounds(plane).unwrap(), canvas);
        }
        assert_eq!(
            s.bounds(s.plane_count()).unwrap_err().to_string(),
            "plane out of range"
        );
    }

    #[test]
    fn the_cover_sits_in_the_right_slot_alone() {
        let s = spread(vec![(100.0, 200.0), (100.0, 200.0), (100.0, 200.0)]);
        let img = s.render_whole(0, 1.0).unwrap();
        assert_eq!((img.width.get(), img.height.get()), (208, 200));
        // The left slot stays transparent, and the cover paints white.
        assert_eq!(pixel(&img, 10, 100), [0, 0, 0, 0]);
        // The cover centers in the right slot, which starts after the gutter.
        assert_eq!(pixel(&img, 104, 100), [0, 0, 0, 0]);
        assert_eq!(pixel(&img, 110, 100), [10, 0, 0, 255]);
    }

    #[test]
    fn a_pair_puts_page_n_left_and_page_n_plus_one_right() {
        let s = spread(vec![
            (100.0, 200.0),
            (100.0, 200.0),
            (100.0, 200.0),
            (100.0, 200.0),
        ]);
        // Spread 1 holds pages 1 and 2.
        let img = s.render_whole(1, 1.0).unwrap();
        assert_eq!(pixel(&img, 10, 100), [20, 0, 0, 255]);
        assert_eq!(pixel(&img, 110, 100), [30, 0, 0, 255]);
        // The gutter stays transparent.
        assert_eq!(pixel(&img, 101, 100), [0, 0, 0, 0]);
        // Spread 2 holds page 3 alone, on the left, because a pair needs two
        // pages.
        let last = s.render_whole(2, 1.0).unwrap();
        assert_eq!(pixel(&last, 10, 100), [40, 0, 0, 255]);
        assert_eq!(pixel(&last, 200, 100), [0, 0, 0, 0]);
    }

    #[test]
    fn a_region_render_takes_pixels_from_both_pages() {
        let s = spread(vec![(100.0, 200.0), (100.0, 200.0), (100.0, 200.0)]);
        // A window that spans the seam of spread 1 gets both pages.
        let region = Region {
            x: 90.0,
            y: 0.0,
            w: 30.0,
            h: 200.0,
        };
        let img = s.render(View::region(1, region, 1.0)).unwrap();
        assert_eq!((img.width.get(), img.height.get()), (30, 200));
        assert_eq!(pixel(&img, 0, 100), [20, 0, 0, 255]);
        assert_eq!(pixel(&img, 29, 100), [30, 0, 0, 255]);
    }

    #[test]
    fn a_region_past_the_canvas_clips_instead_of_failing() {
        let s = spread(vec![(100.0, 200.0), (100.0, 200.0)]);
        let region = Region {
            x: -10.0,
            y: -10.0,
            w: 50.0,
            h: 50.0,
        };
        let img = s.render(View::region(0, region, 1.0)).unwrap();
        assert_eq!((img.width.get(), img.height.get()), (40, 40));
        let far = Region {
            x: 1000.0,
            y: 1000.0,
            w: 10.0,
            h: 10.0,
        };
        assert!(s.render(View::region(0, far, 1.0)).is_err());
    }

    #[test]
    fn a_huge_scale_clamps_to_the_spread_cap() {
        let s = spread(vec![(100.0, 200.0), (100.0, 200.0)]);
        let img = s.render_whole(0, 1000.0).unwrap();
        let pixels = u64::from(img.width.get()) * u64::from(img.height.get());
        assert!(
            pixels <= MAX_RENDER_PIXELS as u64,
            "pixels {pixels} past the cap"
        );
        assert!(img.width.get() > img.height.get());
    }

    #[test]
    fn a_single_page_document_keeps_one_plane() {
        let s = spread(vec![(612.0, 792.0)]);
        assert_eq!(s.plane_count(), 1);
        assert_eq!(s.canvas().w, 612.0 * 2.0 + GUTTER);
        assert!(s.progressive().unwrap().is_none());
    }

    #[test]
    fn an_empty_document_does_not_wrap() {
        let err = SpreadSurface::new(Arc::new(Pages(Vec::new()))).unwrap_err();
        assert_eq!(err.to_string(), "plane out of range");
    }
}
