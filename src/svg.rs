//! SVG and SVGZ rasterization with `resvg` and `tiny-skia`.
//!
//! The module sniffs the file head, parses the bytes into a `usvg` tree and
//! paints the tree into an RGBA8 buffer.
//! It runs on a decode worker only, like every decoder.
//!
//! Sniffing reads content, never the file name.
//! The code skips a UTF-8 BOM, whitespace, XML comments, a processing
//! instruction and a doctype, then requires an `<svg` root.
//! A gzip header (`1f 8b`, the `.svgz` form) decompresses first, with a
//! `MAX_INPUT` cap, so a gzip bomb fails instead of filling memory.
//!
//! A vector file has no pixel size, so the code picks one.
//! `base_side` clamps the declared long side into `BASE_MIN_SIDE` (1024 px)
//! .. [``crate::MAX_TEXTURE_SIDE``]:
//! a 24 px icon bakes at 1024 px, so it zooms without turning to mush, and a
//! huge viewBox lands on the texture limit instead of a 200 MB buffer.
//! The aspect stays exact, so the base raster and the 512 px thumbnail share
//! one shape.
//!
//! `Vector` keeps the parsed tree, so [`Vector::render_region`] paints the
//! region the window shows at the scale it shows it, and a deep zoom never
//! stretches the base raster (see [`crate::region`]).
//! The image view holds one tree behind an `Arc` for the length of an open
//! image.
//!
//! External references never resolve: `resources_dir` stays `None` and the
//! code passes no custom resolver, so an `<image href="file:///etc/passwd">`
//! draws nothing.
use std::{
    io::Read,
    path::Path,
    sync::{Arc, OnceLock},
};

use crate::{
    DecodeError, Decoded, DecodedImage, MAX_TEXTURE_SIDE, cap_texture_side,
    dim::{ImgH, ImgW, LongSide},
};

/// The smallest long side the code bakes for a vector file.
///
/// Many SVGs declared at icon size (16 to 48 px), and baking at 24 px would
/// fall apart at the first zoom step.
/// 1024 px costs 4 MB of RGBA once and covers a normal window at fit.
pub(crate) const BASE_MIN_SIDE: LongSide = LongSide::new(1024);

/// The most bytes the code reads or decompresses for one vector file.
///
/// An SVG is XML text, so a real file stays far below this.
/// The cap bounds a gzip bomb and a pathological file alike.
const MAX_INPUT: usize = 32 * 1024 * 1024;

/// The head the code sniffs, and the head it reads of a decompressed stream.
const HEAD: usize = 8 * 1024;

/// The size usvg assumes for an SVG without width, height or viewBox.
const DEFAULT_SIDE: f32 = 100.0;

/// A parsed SVG, ready to paint at any size.
///
/// The tree holds geometry with text already turned into paths, so it paints
/// again at any scale without the font database and without another XML parse.
/// [`crate::svg::parse`] returns one, and the image view keeps it behind an
/// `Arc` for the length of an open image.
#[derive(Debug)]
pub struct Vector {
    tree: usvg::Tree,
}

impl Vector {
    /// The declared size, guarded against zero, negative and non-finite.
    fn declared(&self) -> (f32, f32) {
        let size = self.tree.size();
        guard(size.width(), size.height())
    }

    /// The longest declared side, guarded like [`Self::declared`].
    fn declared_long(&self) -> f32 {
        let (w, h) = self.declared();
        w.max(h)
    }

    /// The declared units that one base raster pixel spans.
    ///
    /// The viewer measures the base raster, and the tree stays in declared
    /// units, so a region that arrives in base pixels converts with this
    /// factor.
    fn declared_per_base(&self) -> f32 {
        let (w, h) = self.declared();
        let (bw, bh) = self.base_size();
        let base_long = bw.get().max(bh.get());
        if base_long == 0 {
            return 1.0;
        }
        w.max(h) / base_long as f32
    }

    /// The pixel size of the base raster.
    ///
    /// The grid entry and the cache header store this pair, because a vector
    /// file has no pixel size of its own.
    /// The thumbnail and the base raster share the aspect, so a warm cell and
    /// a cold one measure the same picture.
    #[must_use]
    pub fn base_size(&self) -> (ImgW, ImgH) {
        scale_to(self.declared(), base_side(self.declared_long()))
    }

    /// Paint the tree at the base raster size.
    ///
    /// # Errors
    ///
    /// The function errors when the pixmap allocation fails.
    pub fn render_base(&self) -> Result<DecodedImage, DecodeError> {
        self.rasterize(base_side(self.declared_long()))
    }

    /// Paint the part of the tree under `region` at `scale` pixels per source
    /// pixel.
    ///
    /// The image view paints the region its window shows, so the paint area
    /// tracks window pixels and a deep zoom costs no more than a shallow one
    /// (see [`crate::region`]).
    /// A filter that samples past the region loses its outer context, so the
    /// caller paints a margin around what it shows.
    ///
    /// # Errors
    ///
    /// The function errors when the pixmap allocation fails.
    pub fn render_region(
        &self,
        region: crate::region::Crop,
        scale: f32,
    ) -> Result<DecodedImage, DecodeError> {
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        // A window wider than the texture limit at a diagonal turn would ask
        // for more pixels than an upload takes, so the patch drops resolution
        // instead of clipping, and the draw stretches it back to the region.
        let wanted = (region.w * scale).max(region.h * scale).max(1.0);
        let limit = MAX_TEXTURE_SIDE as f32;
        let fit = if wanted > limit { limit / wanted } else { 1.0 };
        let width = (region.w * scale * fit).round().max(1.0);
        let height = (region.h * scale * fit).round().max(1.0);
        let (rw, rh) = (width as u32, height as u32);
        let mut pixmap = tiny_skia::Pixmap::new(rw, rh).ok_or(DecodeError::Raster)?;
        // The region arrives in base raster pixels, and the tree paints in
        // declared units.
        let per_base = self.declared_per_base();
        let region_scale = scale * fit / per_base;
        let (rx, ry) = (region.x * per_base, region.y * per_base);
        // The region corner lands on pixel 0: scale around the origin, then
        // shift that corner to the top-left pixel.
        let transform = tiny_skia::Transform {
            sx: region_scale,
            kx: 0.0,
            ky: 0.0,
            sy: region_scale,
            tx: -rx * region_scale,
            ty: -ry * region_scale,
        };
        resvg::render(&self.tree, transform, &mut pixmap.as_mut());
        let mut data = pixmap.take();
        demultiply(&mut data);
        Ok(DecodedImage {
            width: ImgW::new(rw),
            height: ImgH::new(rh),
            data,
        })
    }

    /// Paint the tree with its long side on `long`, keeping the aspect.
    ///
    /// # Errors
    ///
    /// The function errors when the pixmap allocation fails.
    pub fn rasterize(&self, long: LongSide) -> Result<DecodedImage, DecodeError> {
        let (w, h) = scale_to(self.declared(), long);
        let (rw, rh) = (w.get(), h.get());
        let mut pixmap = tiny_skia::Pixmap::new(rw, rh).ok_or(DecodeError::Raster)?;
        let (dw, dh) = self.declared();
        let transform = tiny_skia::Transform::from_scale(rw as f32 / dw, rh as f32 / dh);
        resvg::render(&self.tree, transform, &mut pixmap.as_mut());
        let mut data = pixmap.take();
        demultiply(&mut data);
        Ok(cap_texture_side(DecodedImage {
            width: ImgW::new(rw),
            height: ImgH::new(rh),
            data,
        }))
    }
}

/// The base raster size for a declared long side.
///
/// The function clamps into [`BASE_MIN_SIDE`] .. [`MAX_TEXTURE_SIDE`], so the
/// paint cost and the texture stay bounded.
#[must_use]
pub(crate) fn base_side(declared_long: f32) -> LongSide {
    let long = if declared_long.is_finite() && declared_long > 0.0 {
        declared_long
    } else {
        DEFAULT_SIDE
    };
    let clamped = long
        .round()
        .clamp(BASE_MIN_SIDE.get() as f32, MAX_TEXTURE_SIDE as f32);
    LongSide::new(clamped as u32)
}

/// The pixel size of a raster whose long side lands on `long` and whose shape
/// matches `size`.
fn scale_to(size: (f32, f32), long: LongSide) -> (ImgW, ImgH) {
    let (w, h) = size;
    let scale = long.get().max(1) as f32 / w.max(h);
    let limit = MAX_TEXTURE_SIDE as f32;
    let rw = (w * scale).round().clamp(1.0, limit);
    let rh = (h * scale).round().clamp(1.0, limit);
    (ImgW::new(rw as u32), ImgH::new(rh as u32))
}

/// Replace a zero, negative or non-finite size with the SVG default.
fn guard(w: f32, h: f32) -> (f32, f32) {
    let sane = |v: f32| v.is_finite() && v > 0.0;
    if sane(w) && sane(h) {
        (w, h)
    } else {
        (DEFAULT_SIDE, DEFAULT_SIDE)
    }
}

/// Convert premultiplied RGBA to straight alpha, in place.
///
/// `tiny-skia` paints premultiplied, and raylib blends straight alpha, so the
/// upload needs this pass.
/// A fully transparent pixel becomes all zero.
fn demultiply(data: &mut [u8]) {
    for px in data.as_chunks_mut::<4>().0 {
        let a = u32::from(px[3]);
        if a == 0 {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
            continue;
        }
        for c in &mut px[..3] {
            *c = ((u32::from(*c) * 255 + a / 2) / a).min(255) as u8;
        }
    }
}

/// The host font database, loaded once.
///
/// A scan of the system fonts costs tens of milliseconds, so one load serves
/// every SVG parse and every cell of a directory.
fn fontdb() -> &'static Arc<fontdb::Database> {
    static DB: OnceLock<Arc<fontdb::Database>> = OnceLock::new();
    DB.get_or_init(|| {
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        Arc::new(db)
    })
}

/// The parser options.
fn options() -> usvg::Options<'static> {
    usvg::Options {
        fontdb: Arc::clone(fontdb()),
        // External files never resolve, so a file cannot make the viewer
        // read an unrelated path.
        resources_dir: None,
        ..usvg::Options::default()
    }
}

/// Read the whole file, decompressing a gzip payload.
///
/// # Errors
///
/// The function errors when the read, the decompression or the size cap
/// fails.
fn read(path: &Path) -> Result<Vec<u8>, DecodeError> {
    let raw = std::fs::read(path)?;
    if !is_gzip(&raw) {
        if raw.len() > MAX_INPUT {
            return Err(DecodeError::Input);
        }
        return Ok(raw);
    }
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(raw.as_slice())
        .take(MAX_INPUT as u64 + 1)
        .read_to_end(&mut out)?;
    if out.len() > MAX_INPUT {
        return Err(DecodeError::Input);
    }
    Ok(out)
}

/// Whether the bytes start with the gzip magic.
const fn is_gzip(bytes: &[u8]) -> bool {
    matches!(bytes, [0x1f, 0x8b, ..])
}

/// Parse a file into a [`Vector`].
///
/// # Errors
///
/// The function errors when the file cannot be read, or when usvg rejects
/// the XML, the namespaces or the document structure.
pub fn parse(path: &Path) -> Result<Vector, DecodeError> {
    let bytes = read(path)?;
    let tree =
        usvg::Tree::from_data(&bytes, &options()).map_err(|e| DecodeError::Svg(Box::new(e)))?;
    Ok(Vector { tree })
}

/// Measure a file without painting it.
///
/// The parse answers the root size, which is all a cell and the image-view fit
/// need before the first paint.
/// An empty font database skips the glyph work and the system font scan, so a
/// glyph never moves a root size and the measure stays cheap.
///
/// # Errors
///
/// The function errors when the file cannot be read, or when usvg rejects it.
pub(crate) fn measure(path: &Path) -> Result<(ImgW, ImgH), DecodeError> {
    let bytes = read(path)?;
    let options = usvg::Options {
        fontdb: Arc::new(fontdb::Database::new()),
        resources_dir: None,
        ..usvg::Options::default()
    };
    let tree =
        usvg::Tree::from_data(&bytes, &options).map_err(|e| DecodeError::Svg(Box::new(e)))?;
    Ok(Vector { tree }.base_size())
}

/// Whether the file holds SVG content, decided on the head bytes.
///
/// The code reads a 2-byte magic first: a known raster magic, a JPEG XL
/// codestream and a gzip header each skip the text test, so only a text
/// candidate costs a head read.
/// A gzip payload sniffs its decompressed head.
#[must_use]
pub fn sniff(path: &Path) -> bool {
    let Some(magic) = super::file_magic(path) else {
        return false;
    };
    if is_gzip(&magic) {
        return sniff_gzip(path);
    }
    if super::is_common_magic(magic) || magic == [0xff, 0x0a] {
        return false;
    }
    head(path).is_some_and(|bytes| looks_like_svg(&bytes))
}

/// Sniff the decompressed head of a gzip payload.
fn sniff_gzip(path: &Path) -> bool {
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let mut head = Vec::new();
    if flate2::read::GzDecoder::new(file)
        .take(HEAD as u64)
        .read_to_end(&mut head)
        .is_err()
    {
        return false;
    }
    looks_like_svg(&head)
}

/// The first [`HEAD`] bytes of the file.
fn head(path: &Path) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    let mut head = Vec::new();
    file.take(HEAD as u64).read_to_end(&mut head).ok()?;
    Some(head)
}

/// Whether a byte slice holds an SVG document.
///
/// The code strips the UTF-8 BOM, then the prolog: whitespace, XML comments,
/// processing instructions and a doctype.
/// What remains must start with `<svg`.
#[must_use]
fn looks_like_svg(bytes: &[u8]) -> bool {
    let text = std::str::from_utf8(bytes).unwrap_or("");
    let mut rest = text.trim_start_matches('\u{feff}');
    loop {
        rest = rest.trim_start();
        if let Some(tail) = rest.strip_prefix("<!--") {
            let Some(end) = tail.find("-->") else {
                return false; // a head cut inside a comment proves nothing
            };
            rest = &tail[end + 3..];
        } else if let Some(tail) = rest.strip_prefix("<?") {
            let Some(end) = tail.find("?>") else {
                return false;
            };
            rest = &tail[end + 2..];
        } else if rest.len() >= 9 && rest[..9].eq_ignore_ascii_case("<!doctype") {
            // The doctype of an SVG carries no internal subset, so the first
            // `>` ends it.
            let Some(end) = rest.find('>') else {
                return false;
            };
            rest = &rest[end + 1..];
        } else {
            break;
        }
    }
    rest.len() >= 5
        && rest[..4].eq_ignore_ascii_case("<svg")
        && matches!(
            rest.as_bytes()[4],
            b'>' | b'/' | b' ' | b'\t' | b'\r' | b'\n'
        )
}

/// Decode a vector file at the base raster size.
///
/// # Errors
///
/// The function errors when the parse or the paint fails.
pub(crate) fn decode_base(path: &Path) -> Result<Decoded, DecodeError> {
    let vector = parse(path)?;
    Ok(Decoded {
        source: vector.base_size(),
        image: vector.render_base()?,
    })
}

/// Decode a vector file with its long side on `long`.
///
/// The grid takes the thumbnail this way, so a cell never paints a full-size
/// pixmap only to shrink it.
///
/// # Errors
///
/// The function errors when the parse or the paint fails.
pub(crate) fn decode_thumb(path: &Path, long: LongSide) -> Result<Decoded, DecodeError> {
    let vector = parse(path)?;
    Ok(Decoded {
        source: vector.base_size(),
        image: vector.rasterize(long)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQUARE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">
        <rect width="100" height="50" fill="#ff0000"/>
    </svg>"##;

    fn gzip(bytes: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(bytes).expect("gzip write");
        enc.finish().expect("gzip finish")
    }

    #[test]
    fn sniffs_a_document_behind_a_prolog() {
        assert!(looks_like_svg(SQUARE.as_bytes()));
        assert!(looks_like_svg(b"<?xml version=\"1.0\"?><svg/>"));
        assert!(looks_like_svg(
            b"\xef\xbb\xbf  <!-- a note -->\n<!DOCTYPE svg PUBLIC \"x\">\n<SVG xmlns=\"\"/>"
        ));
    }

    #[test]
    fn rejects_other_content() {
        assert!(!looks_like_svg(b"<html><body/></html>"));
        assert!(!looks_like_svg(b""));
        assert!(!looks_like_svg(b"<svg")); // head cut before the root ends
        assert!(!looks_like_svg(b"<!-- open comment"));
        assert!(!looks_like_svg(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    fn a_gzip_payload_sniffs_through_the_decoder() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("drawing.svgz");
        std::fs::write(&path, gzip(SQUARE.as_bytes())).expect("write");
        assert!(sniff(&path));
        let decoded = decode_base(&path).expect("decode");
        assert_eq!(decoded.image.width, ImgW::new(1024));
        assert_eq!(decoded.image.height, ImgH::new(512));
    }

    #[test]
    fn a_raster_file_never_sniffs_as_svg() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("photo.png");
        let png = image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]));
        png.save(&path).expect("png");
        assert!(!sniff(&path));
    }

    #[test]
    fn the_base_side_clamps_and_guards() {
        assert_eq!(base_side(24.0), BASE_MIN_SIDE);
        assert_eq!(base_side(2000.0), LongSide::new(2000));
        assert_eq!(base_side(1.0e9), LongSide::new(MAX_TEXTURE_SIDE));
        assert_eq!(base_side(f32::NAN), BASE_MIN_SIDE);
        assert_eq!(base_side(0.0), BASE_MIN_SIDE);
        assert_eq!(base_side(-5.0), BASE_MIN_SIDE);
    }

    #[test]
    fn a_raster_keeps_the_aspect_and_lands_on_the_target() {
        assert_eq!(
            scale_to((200.0, 100.0), LongSide::new(512)),
            (ImgW::new(512), ImgH::new(256))
        );
        assert_eq!(
            scale_to((100.0, 200.0), LongSide::new(512)),
            (ImgW::new(256), ImgH::new(512))
        );
        assert_eq!(
            scale_to((1.0, 1000.0), LongSide::new(512)),
            (ImgW::new(1), ImgH::new(512))
        );
    }

    #[test]
    fn the_vector_measures_the_base_raster() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("drawing.svg");
        std::fs::write(&path, SQUARE).expect("write");
        let vector = parse(&path).expect("parse");
        assert_eq!(vector.base_size(), (ImgW::new(1024), ImgH::new(512)));
        let img = vector.render_base().expect("render");
        assert_eq!((img.width, img.height), (ImgW::new(1024), ImgH::new(512)));
        assert_eq!(img.data.len(), 1024 * 512 * 4);
        // The red rectangle covers the frame, and the painter writes opaque
        // red, so the demultiply pass changes nothing.
        assert_eq!(&img.data[..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn a_thumbnail_paints_smaller_than_the_base() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("drawing.svg");
        std::fs::write(&path, SQUARE).expect("write");
        let thumb = decode_thumb(&path, LongSide::new(512)).expect("thumb");
        assert_eq!(
            (thumb.image.width, thumb.image.height),
            (ImgW::new(512), ImgH::new(256))
        );
        // The source pair never reports the thumbnail size.
        assert_eq!(thumb.source, (ImgW::new(1024), ImgH::new(512)));
    }

    #[test]
    fn a_transparent_pixel_demultiplies_to_zero() {
        let mut data = vec![10, 20, 30, 0, 10, 20, 30, 255, 0, 0, 0, 128];
        demultiply(&mut data);
        assert_eq!(&data[..4], &[0, 0, 0, 0]);
        assert_eq!(&data[4..8], &[10, 20, 30, 255]);
        // A half-transparent black stays black.
        assert_eq!(&data[8..12], &[0, 0, 0, 128]);
    }

    #[cfg(feature = "cache")]
    #[test]
    fn a_vector_thumbnail_round_trips_through_the_cache() {
        use crate::thumbcache::Cache;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("drawing.svg");
        std::fs::write(&path, SQUARE).expect("write");
        let cache = Cache::in_dir(dir.path().join("cache"));
        let thumb = decode_thumb(&path, LongSide::new(512)).expect("thumb");
        cache
            .store(&path, LongSide::new(512), &thumb.image, thumb.source)
            .expect("store");
        let hit = cache.load(&path, LongSide::new(512)).expect("load");
        // The header keeps the base raster size, so a warm launch fits the
        // same picture a cold decode reports.
        assert_eq!(hit.image, (ImgW::new(1024), ImgH::new(512)));
        assert_eq!(
            (hit.thumb.width, hit.thumb.height),
            (ImgW::new(512), ImgH::new(256))
        );
    }

    #[test]
    fn a_document_with_text_paints_with_the_host_fonts() {
        // The test sandbox may hold no fonts at all, and resvg then skips the
        // glyphs, so the test asserts the paint, not the glyph pixels.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("label.svg");
        std::fs::write(
            &path,
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
                <rect width="200" height="100" fill="#000000"/>
                <text x="10" y="60" font-size="40" fill="#ffffff">vv</text>
            </svg>"##,
        )
        .expect("write");
        let d = decode_thumb(&path, LongSide::new(512)).expect("thumb");
        assert_eq!(
            (d.image.width, d.image.height),
            (ImgW::new(512), ImgH::new(256))
        );
    }

    #[test]
    fn a_measure_reads_the_base_size_without_painting() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("drawing.svg");
        std::fs::write(
            &path,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="300" height="100">
                <text x="10" y="60" font-size="40">vv</text>
            </svg>"#,
        )
        .expect("write");
        // A glyph never names the root size, so the cheap parse answers the
        // same pair the full decode reports.
        assert_eq!(
            measure(&path).expect("measure"),
            (ImgW::new(1024), ImgH::new(341))
        );
        assert_eq!(
            parse(&path).expect("parse").base_size(),
            measure(&path).expect("measure")
        );
        // A raster file never reaches the measure, and the caller filters it
        // out by sniff first.
        let png = dir.path().join("photo.png");
        image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]))
            .save(&png)
            .expect("png");
        assert!(!sniff(&png));
    }

    #[test]
    fn a_region_paints_the_visible_part_at_the_view_scale() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("split.svg");
        // The left half red, the right half blue, in declared units.
        std::fs::write(
            &path,
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">
                <rect width="50" height="50" fill="#ff0000"/>
                <rect x="50" width="50" height="50" fill="#0000ff"/>
            </svg>"##,
        )
        .expect("write");
        let vector = parse(&path).expect("parse");
        assert_eq!(vector.base_size(), (ImgW::new(1024), ImgH::new(512)));
        // The region arrives in base raster pixels, so the right half starts
        // at x = 512 and covers 512 pixels of the 1024 px source.
        let right = crate::region::Crop {
            x: 512.0,
            y: 0.0,
            w: 512.0,
            h: 512.0,
        };
        let patch = vector.render_region(right, 1.0).expect("paint");
        assert_eq!(
            (patch.width, patch.height),
            (ImgW::new(512), ImgH::new(512))
        );
        assert_eq!(&patch.data[..4], &[0, 0, 255, 255]);
        assert_eq!(&patch.data[patch.data.len() - 4..], &[0, 0, 255, 255]);
        // The left half paints red, so the region origin lands on pixel 0.
        let left = crate::region::Crop {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 50.0,
        };
        let patch = vector.render_region(left, 2.0).expect("paint");
        assert_eq!(
            (patch.width, patch.height),
            (ImgW::new(200), ImgH::new(100))
        );
        assert_eq!(&patch.data[..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn a_region_measures_in_base_pixels_when_the_base_rounds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("wide.svg");
        // Declared 300x100 bakes at 1024x341, so one base pixel spans
        // 0.2929 declared units and the halves meet at base x = 512.
        std::fs::write(
            &path,
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="300" height="100">
                <rect width="150" height="100" fill="#ff0000"/>
                <rect x="150" width="150" height="100" fill="#0000ff"/>
            </svg>"##,
        )
        .expect("write");
        let vector = parse(&path).expect("parse");
        assert_eq!(vector.base_size(), (ImgW::new(1024), ImgH::new(341)));
        let right = crate::region::Crop {
            x: 512.0,
            y: 0.0,
            w: 512.0,
            h: 341.0,
        };
        let patch = vector.render_region(right, 2.0).expect("paint");
        // The patch measures base pixels times the view scale.
        assert_eq!(
            (patch.width, patch.height),
            (ImgW::new(1024), ImgH::new(682))
        );
        assert_eq!(&patch.data[..4], &[0, 0, 255, 255]);
    }

    #[test]
    fn an_oversized_input_fails_instead_of_allocating() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("huge.svg");
        // A gzip bomb of zeros: the cap trips before the buffer grows.
        std::fs::write(&path, gzip(&vec![b' '; MAX_INPUT + 1])).expect("write");
        assert!(matches!(read(&path), Err(DecodeError::Input)));
    }
}
