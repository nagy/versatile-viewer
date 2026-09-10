//! Typst backend: compiles `.typ` files and rasterizes the result through
//! [typst-render](https://docs.rs/typst-render) — pure Rust, no C. This file
//! owns the typst/typst-kit/typst-render dependencies, mirroring pdf.rs's
//! ownership of hayro; the rest of the viewer only sees the [`Document`]
//! trait (document.rs).
//!
//! Pipeline at open time: read the file, build a filesystem-rooted
//! [`typst::World`] (the file's directory is the project root, so `#include`,
//! `#image` and `#read` resolve like the `typst` CLI), compile once, and keep
//! the laid-out [`PagedDocument`]. `render` then rasterizes a stored page at
//! any scale (the same preview-first / re-render-at-resting-zoom trick the
//! PDF path uses — text stays sharp at every resting zoom).
//!
//! Fonts are not embedded: they come from explicit search paths
//! (`VV_TYPST_FONT_PATHS` / `TYPST_FONT_PATHS`, plus the nix build's
//! compile-time `VV_NIX_FONT_PATHS` store paths) with a system font scan as
//! the fallback for non-nix machines. Font binaries are loaded lazily and
//! shared document-wide through a global store.

use std::{
    env,
    fmt::Write as _,
    path::{Path, PathBuf},
    sync::LazyLock,
};

use anyhow::{Context, Result, bail};
use typst::{
    Library, LibraryExt, World, WorldExt,
    diag::{FileError, FileResult, Warned},
    foundations::{Bytes, Datetime, Duration},
    syntax::{DiagSpan, DiagSpanKind, FileId, RootedPath, Source, VirtualPath, VirtualRoot},
    utils::LazyHash,
};
use typst_kit::fonts::{FontStore, scan, system};
use typst_layout::PagedDocument;
use typst_render::RenderOptions;

use crate::document::{DecodedImage, Document, PageInfo};

/// Pixel budget for one rendered page: zoom scales render resolution freely,
/// so an unclamped deep zoom could try to allocate gigabytes. Same cap as
/// pdf.rs (~67 MP ≈ 256 MB RGBA8); beyond this the texture upscales on the
/// GPU (soft, not sharp).
const MAX_PAGE_PIXELS: u64 = 4096 * 4096 + 4096 * 4096 / 2;

/// All fonts this process knows about, discovered once (a fontdb scan is far
/// too expensive to repeat per opened document). Font binaries load lazily
/// per slot on first use; the store is `Send + Sync`, so grid decodes, the
/// streaming loader and refine renders can all compile/render concurrently.
static FONTS: LazyLock<FontStore> = LazyLock::new(build_fonts);

/// Build the font store: explicit search paths first (they win font
/// matching), then the system scan. Explicit paths come from (in order):
/// `VV_TYPST_FONT_PATHS`, `TYPST_FONT_PATHS` (typst CLI convention) and the
/// nix build's baked-in `VV_NIX_FONT_PATHS`; each is a colon-separated list
/// of directories scanned recursively.
fn build_fonts() -> FontStore {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for var in [
        "VV_TYPST_FONT_PATHS",
        "TYPST_FONT_PATHS",
        "VV_NIX_FONT_PATHS",
    ] {
        if let Some(paths) = env::var_os(var) {
            dirs.extend(env::split_paths(&paths));
        }
    }
    // The nix build bakes its font store paths into the binary; `option_env!`
    // resolves at compile time, so plain `cargo build`s outside nix just see
    // nothing here and fall through to the system scan.
    if let Some(paths) = option_env!("VV_NIX_FONT_PATHS") {
        dirs.extend(env::split_paths(paths));
    }

    let mut store = FontStore::new();
    for dir in dirs {
        if dir.is_dir() {
            store.extend(scan(&dir));
        }
    }
    store.extend(system());
    store
}

/// A filesystem-rooted Typst world: `main.typ` is the opened `.typ` file and
/// the project root is its directory. Everything the compiler requests —
/// sources, images, data files — maps from a virtual path under the project
/// root to a real path on disk (realization guards path escapes).
struct FsWorld {
    /// Typst's standard library: built-ins, default styles (page size, font).
    library: LazyHash<Library>,
    /// Font metadata for matching `#set text(font: ...)` requests.
    book: LazyHash<typst::text::FontBook>,
    /// Entry point: virtual path of the opened file.
    main_id: FileId,
    /// Real directory the virtual paths realize against.
    root: PathBuf,
}

impl FsWorld {
    fn new(root: PathBuf, main_id: FileId) -> Self {
        Self {
            library: LazyHash::new(Library::default()),
            book: FONTS.book().clone(),
            main_id,
            root,
        }
    }

    /// Map a virtual file id to a real file's bytes. Typst packages would
    /// need network downloads and caching; unsupported (compile error).
    fn read(&self, id: FileId) -> FileResult<Bytes> {
        if id.root() != &VirtualRoot::Project {
            return Err(FileError::Other(Some(
                "Typst packages are not supported".into(),
            )));
        }
        let real = id
            .vpath()
            .realize(&self.root)
            .map_err(|_| FileError::Other(Some("path escapes the project root".into())))?;
        let data = std::fs::read(&real).map_err(|err| FileError::from_io(err, &real))?;
        Ok(Bytes::new(data))
    }
}

impl World for FsWorld {
    fn library(&self) -> &LazyHash<Library> {
        &self.library
    }

    fn book(&self) -> &LazyHash<typst::text::FontBook> {
        &self.book
    }

    fn main(&self) -> FileId {
        self.main_id
    }

    fn source(&self, id: FileId) -> FileResult<Source> {
        let bytes = self.read(id)?;
        let text = bytes.as_str().map_err(|_| FileError::InvalidUtf8)?;
        Ok(Source::new(id, text.to_owned()))
    }

    fn file(&self, id: FileId) -> FileResult<Bytes> {
        self.read(id)
    }

    fn font(&self, index: usize) -> Option<typst::text::Font> {
        FONTS.font(index)
    }

    /// Real time is a host concern; documents that read `datetime.today()`
    /// get a compile error instead (same as the in-memory experiment).
    fn today(&self, _offset: Option<Duration>) -> Option<Datetime> {
        None
    }
}

/// A compiled `.typ` file, one [`Document`] page per Typst page.
pub struct TypstDocument {
    /// Laid-out pages, kept for rendering at any scale (the World is only
    /// needed at compile time).
    document: PagedDocument,
    /// Natural size (PostScript points, 1 pt = 1 px at render scale 1.0) of
    /// every page, captured once at open time.
    dims: Vec<(f64, f64)>,
}

// Compile-time check: the laid-out document must be sharable across the
// rayon pool and the main thread without further wrapping.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PagedDocument>();
};

impl TypstDocument {
    /// Read, compile and lay out the `.typ` file at `path`. The file's
    /// directory becomes the project root. Compile errors (syntax, missing
    /// fonts, missing includes) surface as a readable multi-line error —
    /// the grid shows the file as a failed cell.
    pub fn open(path: &Path) -> Result<TypstDocument> {
        // Early read so a missing file or bad UTF-8 fails with a clear
        // message before the compiler spins up (it re-reads via the world).
        std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("file name is not valid UTF-8")?;
        let vpath = VirtualPath::new(file_name).context("invalid file name")?;
        let main_id = RootedPath::new(VirtualRoot::Project, vpath).intern();
        let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();

        let world = FsWorld::new(root, main_id);
        let Warned { output, warnings } = typst::compile::<PagedDocument>(&world);
        for warning in &warnings {
            eprintln!("vv: {}: warning: {}", path.display(), warning.message);
        }
        let document =
            output.map_err(|errors| anyhow::anyhow!(format_diagnostics(&world, &errors)))?;

        // Capture page sizes in points up front (cheap and stable), matching
        // what typst-render will rasterize (no bleed rendering).
        let dims = document
            .pages()
            .iter()
            .map(|page| {
                let size = page.frame.size() + page.bleed.sum_by_axis();
                (size.x.to_pt(), size.y.to_pt())
            })
            .collect::<Vec<_>>();
        if dims.is_empty() {
            bail!("document has no pages");
        }
        Ok(TypstDocument { document, dims })
    }
}

impl Document for TypstDocument {
    fn page_count(&self) -> usize {
        self.dims.len()
    }

    fn page_info(&self, page: usize) -> Result<PageInfo> {
        let (w, h) = self
            .dims
            .get(page)
            .copied()
            .with_context(|| format!("page {page} out of range"))?;
        Ok(PageInfo {
            width: w.ceil() as u32,
            height: h.ceil() as u32,
        })
    }

    fn render(&self, page: usize, scale: f32) -> Result<DecodedImage> {
        if !scale.is_finite() || scale <= 0.0 {
            bail!("invalid render scale {scale}");
        }
        let (w, h) = self
            .dims
            .get(page)
            .copied()
            .with_context(|| format!("page {page} out of range"))?;
        // Clamp the pixel budget like pdf.rs: everything past the clamp still
        // displays (GPU upscale), just without added sharpness.
        let scale = f64::from(scale)
            .min((MAX_PAGE_PIXELS as f64 / (w * h).max(1.0)).sqrt())
            .max(1.0 / 4096.0);

        let page_ref = self
            .document
            .pages()
            .get(page)
            .with_context(|| format!("page {page} out of range"))?;
        let opts = RenderOptions {
            pixel_per_pt: typst::utils::Scalar::new(scale),
            render_bleed: false,
        };
        // The compiler stack is mature, but a panic on pathological content
        // must degrade to a decode failure, not take down a rayon worker.
        let pixmap = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            typst_render::render(page_ref, &opts)
        }))
        .map_err(|_| anyhow::anyhow!("typst-render panicked while rendering page {page}"))?;

        let width = pixmap.width();
        let height = pixmap.height();
        if width == 0 || height == 0 {
            bail!("typst-render rendered page {page} at size 0x0");
        }
        let mut rgba = pixmap.data().to_vec();
        // typst-render paints an opaque white background (unless the page
        // fill is explicitly `none`), and tiny-skia's buffer is premultiplied
        // RGBA8 — with alpha 255 premultiplied equals straight. For the
        // transparent case, composite over white (the viewer has no alpha).
        if rgba.chunks_exact_mut(4).any(|px| px[3] != 255) {
            // Premultiplied over white: out = src + 255 * (1 - a/255). The
            // saturating add can only be reached by rounding, never overflow
            // (premultiplied channels are <= alpha).
            for px in rgba.chunks_exact_mut(4) {
                let a = u16::from(px[3]);
                for c in &mut px[..3] {
                    *c = (u16::from(*c) + 255 - a).min(255) as u8;
                }
                px[3] = 255;
            }
        }
        Ok(DecodedImage {
            width,
            height,
            rgba,
        })
    }

    fn previews(&self) -> bool {
        // Compiling and rasterizing take a moment: send a fast low-res render
        // first, then the full-quality one (mirrors the PDF experience).
        true
    }
}

/// Format compile diagnostics as `file:line:col: message` lines (with the
/// first hint each, when present). `file` is the project-root-relative
/// virtual path, matching how typst's own CLI reports them.
fn format_diagnostics(world: &FsWorld, errors: &[typst::diag::SourceDiagnostic]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for diag in errors.iter().take(5) {
        let mut line = match locate(world, &diag.span) {
            Some(loc) => format!("{loc}: error: {}", diag.message),
            None => format!("error: {}", diag.message),
        };
        if let Some(hint) = diag.hints.first() {
            let _ = writeln!(line, "\n  hint: {}", hint.v);
        }
        lines.push(line);
    }
    let rest = errors.len().saturating_sub(5);
    if rest > 0 {
        lines.push(format!("and {rest} more errors"));
    }
    lines.join("\n")
}

/// Human-readable `path:line:col` for a diagnostic span, resolved through the
/// world so spans in included files point at the right file.
fn locate(world: &FsWorld, span: &DiagSpan) -> Option<String> {
    let (id, start) = match span.get() {
        DiagSpanKind::Detached => return None,
        DiagSpanKind::Number { id, .. } => {
            let range = world.range(*span)?;
            (id, range.start)
        }
        DiagSpanKind::Range { id, range } => (id, range.start),
    };
    let source = world.source(id).ok()?;
    // typst-syntax reports zero-based line/column; diagnostics are 1-based.
    let (line, col) = source.lines().byte_to_line_column(start)?;
    Some(format!(
        "{}:{}:{}",
        id.vpath().get_with_slash(),
        line + 1,
        col + 1
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{is_typst, open_document};

    /// A two-page document with visible content, using only typst-assets-free
    /// defaults (Libertinus Serif comes from the configured font paths).
    const TWO_PAGES: &str = r#"
#set page(width: 400pt, height: 300pt, margin: 20pt)
= One
Hello from page one.

#pagebreak()
= Two
Hello from page two.
"#;

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn typst_document_opens_and_reports_pages() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "doc.typ", TWO_PAGES);
        let doc = TypstDocument::open(&path).unwrap();
        assert_eq!(doc.page_count(), 2);
        assert_eq!(
            doc.page_info(0).unwrap(),
            PageInfo {
                width: 400,
                height: 300
            }
        );
        assert_eq!(
            doc.page_info(1).unwrap(),
            PageInfo {
                width: 400,
                height: 300
            }
        );
        assert!(doc.page_info(2).is_err());
    }

    #[test]
    fn typst_document_renders_page_at_scale() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "doc.typ", TWO_PAGES);
        let doc = TypstDocument::open(&path).unwrap();
        let img = doc.render(0, 2.0).unwrap();
        assert_eq!((img.width, img.height), (800, 600));
        assert_eq!(img.rgba.len(), img.width as usize * img.height as usize * 4);
        // Some glyph pixels must differ from the white background.
        assert!(
            img.rgba
                .chunks_exact(4)
                .any(|px| px != [255, 255, 255, 255])
        );
        // Fully opaque: the viewer has no alpha channel.
        assert!(img.rgba.chunks_exact(4).all(|px| px[3] == 255));
    }

    #[test]
    fn open_document_routes_typ_files() {
        // Extension decides (.typ is plain UTF-8, no magic), and the routed
        // document actually renders.
        assert!(is_typst(Path::new("doc.typ")));
        assert!(is_typst(Path::new("doc.TYP")));
        assert!(!is_typst(Path::new("doc.pdf")));
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "doc.typ", TWO_PAGES);
        let doc = open_document(&path).unwrap();
        assert_eq!(doc.page_count(), 2);
        assert!(doc.previews());
        let decoded = doc.render(1, 1.0).unwrap();
        assert_eq!((decoded.width, decoded.height), (400, 300));
    }

    #[test]
    fn typst_resolves_includes_and_assets_relative_to_the_file() {
        // The opened file's directory is the project root: `#include` pulls
        // in a sibling source, `#image` a sibling picture.
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();
        write(dir, "chapter.typ", "= Chapter\nIncluded content.");
        // 4x2 red PNG via the image crate.
        let png = image::DynamicImage::from(image::RgbaImage::from_pixel(
            4,
            2,
            image::Rgba([255u8, 0, 0, 255]),
        ));
        png.save(dir.join("pic.png")).unwrap();
        let main = write(
            dir,
            "main.typ",
            r#"
#set page(width: 400pt, height: 300pt, margin: 20pt)
#include "chapter.typ"
#image("pic.png", width: 100pt)
"#,
        );
        let doc = TypstDocument::open(&main).unwrap();
        let img = doc.render(0, 1.0).unwrap();
        // The image's red pixels must be visible in the render.
        assert!(img.rgba.chunks_exact(4).any(|px| px[0] > 200 && px[1] < 60));
    }

    #[test]
    fn typst_compile_errors_are_readable() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "bad.typ", "#set page(width: 10cm)\n#foo()\n");
        let err = match TypstDocument::open(&path) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("bad document must not open"),
        };
        assert!(err.contains("error:"), "unexpected message: {err}");
        assert!(err.contains("bad.typ:2"), "no location: {err}");
    }

    #[test]
    fn typst_missing_file_fails_cleanly() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("nope.typ");
        assert!(open_document(&path).is_err());
    }
}
