//! Typst backend: compiles a `.typ` file and renders it through the PDF
//! backend, so a Typst source browses exactly like a PDF.
//!
//! This file owns the typst dependency.
//! The rest of the viewer only sees the [`Surface`] trait.
//!
//! Pipeline at open time: read the file, build a filesystem-rooted
//! [`typst::World`] (the file's directory is the project root, so `#include`,
//! `#image` and `#read` resolve like the `typst` CLI), compile once, keep the
//! laid-out [`PagedDocument`], and export it to PDF bytes in memory.
//! The bytes then go to [`PdfSurface`], which already rasterizes planes, crops
//! a region and refines a settled zoom.
//! One renderer draws every document, so the Typst path inherits the region
//! anchor of the PDF path with no second rasterizer.
//!
//! Fonts are embedded in the binary through `typst-assets` (its `fonts`
//! feature), so no font directory or environment variable is necessary.
//! The compiler works off the in-memory standard library.

use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    sync::LazyLock,
};

use typst::{
    Library, LibraryExt, World, WorldExt,
    diag::{FileError, FileResult, Warned},
    foundations::{Bytes, Datetime, Duration},
    syntax::{DiagSpan, DiagSpanKind, FileId, RootedPath, Source, VirtualPath, VirtualRoot},
    text::{Font, FontBook},
    utils::LazyHash,
};
use typst_layout::PagedDocument;
use typst_pdf::PdfOptions;

use crate::{
    pdf::PdfSurface,
    surface::{ChunkSource, Surface, SurfaceError},
};

/// Every font the compiler knows, parsed once per process.
/// `typst-assets` embeds the binaries with `include_bytes!`, so nothing reads
/// a font file at run time.
/// The list is `Send + Sync`, so grid decodes, the streaming loader and
/// refine renders can all compile at the same time.
static FONTS: LazyLock<Vec<Font>> = LazyLock::new(|| {
    typst_assets::fonts()
        .filter_map(|data| Font::new(Bytes::new(data), 0))
        .collect()
});

/// Font metadata for the embedded binaries, built once per process.
static BOOK: LazyLock<FontBook> = LazyLock::new(|| FontBook::from_fonts(FONTS.iter()));

/// Is this a Typst source file?
/// The test is the extension only: a `.typ` file is plain UTF-8 text, so
/// there is no magic to sniff.
#[must_use]
pub fn is_typst(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("typ"))
}

/// A filesystem-rooted Typst world.
/// The opened `.typ` file is `main`, and its directory is the project root.
/// Every file the compiler asks for maps from a virtual path under the root
/// to a real path on disk.
struct FsWorld {
    /// The Typst standard library: built-ins and the default styles.
    library: LazyHash<Library>,
    /// Font metadata for `#set text(font: ...)`.
    book: LazyHash<FontBook>,
    /// The entry point.
    main: FileId,
    /// The real directory that virtual paths resolve against.
    root: PathBuf,
}

impl FsWorld {
    fn new(root: PathBuf, main: FileId) -> Self {
        FsWorld {
            library: LazyHash::new(Library::default()),
            book: LazyHash::new(BOOK.clone()),
            main,
            root,
        }
    }

    /// Read the bytes of a file the compiler asked for.
    /// A Typst package needs a network download and a cache, which this
    /// viewer does not do, so the call fails with a clear message.
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

    fn book(&self) -> &LazyHash<FontBook> {
        &self.book
    }

    fn main(&self) -> FileId {
        self.main
    }

    fn source(&self, id: FileId) -> FileResult<Source> {
        let bytes = self.read(id)?;
        let text = bytes.as_str().map_err(|_| FileError::InvalidUtf8)?;
        Ok(Source::new(id, text.to_owned()))
    }

    fn file(&self, id: FileId) -> FileResult<Bytes> {
        self.read(id)
    }

    fn font(&self, index: usize) -> Option<Font> {
        FONTS.get(index).cloned()
    }

    /// The current date is a host concern.
    /// A document that reads `datetime.today()` gets an error, because the
    /// reader has no clock of its own.
    fn today(&self, _offset: Option<Duration>) -> Option<Datetime> {
        None
    }
}

/// A compiled `.typ` file, rendered as a PDF surface.
/// The type is a thin wrapper: it holds an open [`PdfSurface`] and hands
/// every call to it, so a Typst page and a PDF page behave the same way.
pub struct TypstSurface {
    pdf: PdfSurface,
}

impl TypstSurface {
    /// Read, compile and export the `.typ` file at `path`.
    /// The directory of the file becomes the project root.
    /// A compile error (a syntax error, a missing font, a missing include)
    /// becomes a readable multi-line message, so the grid shows the file as a
    /// failed cell.
    ///
    /// # Errors
    ///
    /// The function errors when the file cannot be read, the compiler rejects
    /// the source, or the export fails.
    pub fn open(path: &Path) -> Result<TypstSurface, SurfaceError> {
        // The world reads the file, so a missing file fails at compile time
        // with a clear message.
        // The directory of the file is the project root that the virtual
        // paths resolve against.
        let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| SurfaceError::Unsupported("file name is not valid UTF-8".into()))?;
        let vpath = VirtualPath::new(name)
            .map_err(|_| SurfaceError::Unsupported("invalid file name".into()))?;
        let main = RootedPath::new(VirtualRoot::Project, vpath).intern();

        let world = FsWorld::new(root, main);
        // The compiler is young, so a panic on odd content must not cross
        // this boundary: the open runs on a rayon worker or the main thread.
        let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let Warned { output, warnings } = typst::compile::<PagedDocument>(&world);
            for warning in &warnings {
                eprintln!("vv: {}: warning: {}", path.display(), warning.message);
            }
            output.map_err(|errors| format_diagnostics(&world, &errors))
        }))
        .map_err(|_| SurfaceError::Unsupported("the Typst compiler panicked".into()))?;
        let document = compiled.map_err(SurfaceError::Render)?;

        let options = PdfOptions::default();
        let bytes = typst_pdf::pdf(&document, &options)
            .map_err(|errors| SurfaceError::Render(format_diagnostics(&world, &errors)))?;
        Ok(TypstSurface {
            pdf: PdfSurface::from_bytes(bytes)?,
        })
    }
}

impl Surface for TypstSurface {
    fn plane_count(&self) -> usize {
        self.pdf.plane_count()
    }

    fn bounds(&self, plane: usize) -> Result<crate::surface::Size, SurfaceError> {
        self.pdf.bounds(plane)
    }

    fn render_whole(&self, plane: usize, scale: f32) -> Result<crate::DecodedImage, SurfaceError> {
        self.pdf.render_whole(plane, scale)
    }

    fn render(&self, view: crate::surface::View) -> Result<crate::DecodedImage, SurfaceError> {
        self.pdf.render(view)
    }

    // A compiled document has no byte stream to read: the export already
    // happened at open time, so the plane rasterizes in one step.
    fn progressive(&self) -> Result<Option<Box<dyn ChunkSource>>, SurfaceError> {
        Ok(None)
    }

    fn previews(&self) -> bool {
        true
    }

    fn can_refine(&self) -> bool {
        true
    }
}

/// Format compile diagnostics as `file:line:col: message` lines, with the
/// first hint of each when present.
/// The file name is the project-root-relative virtual path, the same way the
/// `typst` CLI reports it.
fn format_diagnostics(world: &FsWorld, errors: &[typst::diag::SourceDiagnostic]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for diag in errors.iter().take(5) {
        let mut line = match locate(world, &diag.span) {
            Some(loc) => format!("{loc}: error: {}", diag.message),
            None => format!("error: {}", diag.message),
        };
        if let Some(hint) = diag.hints.first() {
            let _ = write!(line, "\n  hint: {}", hint.v);
        }
        lines.push(line);
    }
    let rest = errors.len().saturating_sub(5);
    if rest > 0 {
        lines.push(format!("and {rest} more errors"));
    }
    lines.join("\n")
}

/// A readable `path:line:col` for a diagnostic span.
/// The lookup goes through the world, so a span in an included file names
/// that file.
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
    // The syntax layer counts lines and columns from zero; a diagnostic
    // counts them from one.
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
    use crate::surface::{View, open_surface};

    /// A two-page document with visible content.
    const TWO_PAGES: &str = r"
#set page(width: 400pt, height: 300pt, margin: 20pt)
= One
Hello from page one.

#pagebreak()
= Two
Hello from page two.
";

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn typst_opens_and_reports_pages() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "doc.typ", TWO_PAGES);
        let surface = TypstSurface::open(&path).unwrap();
        assert_eq!(surface.plane_count(), 2);
        assert_eq!(
            surface.bounds(0).unwrap(),
            crate::surface::Size { w: 400.0, h: 300.0 }
        );
        assert_eq!(
            surface.bounds(1).unwrap(),
            crate::surface::Size { w: 400.0, h: 300.0 }
        );
        assert!(surface.bounds(2).is_err());
    }

    #[test]
    fn typst_renders_a_page_at_scale() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "doc.typ", TWO_PAGES);
        let surface = TypstSurface::open(&path).unwrap();
        let image = surface.render(View::whole(0, 2.0)).unwrap();
        assert_eq!(
            (image.width, image.height),
            (crate::dim::ImgW::new(800), crate::dim::ImgH::new(600))
        );
        // Some pixels must differ from the white page.
        assert!(
            image
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .any(|px| px != &[255, 255, 255, 255])
        );
    }

    #[test]
    fn open_surface_routes_typ_files() {
        assert!(is_typst(Path::new("doc.typ")));
        assert!(is_typst(Path::new("doc.TYP")));
        assert!(!is_typst(Path::new("doc.pdf")));
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "doc.typ", TWO_PAGES);
        let surface = open_surface(&path).unwrap();
        assert_eq!(surface.plane_count(), 2);
        assert!(surface.previews());
        assert!(surface.can_refine());
    }

    #[test]
    #[allow(clippy::float_cmp)] // the asserted values are exact book arithmetic
    fn typst_document_opens_as_a_book() {
        // The image view wraps a multi-plane source in `BookSurface`, so a
        // Typst source shows its pages as book spreads, exactly like a PDF.
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "doc.typ", TWO_PAGES);
        let surface = crate::surface::open_book_surface(&path).unwrap();
        // One plane holds every spread.
        assert_eq!(surface.plane_count(), 1);
        // A fresh open frames the first spread, not the whole book.
        assert!(surface.open_box(0).is_some());
        // The book is wider than one page (two slots plus the gutter) and
        // taller than one page (a cover spread under a pair spread).
        let book = surface.bounds(0).unwrap();
        assert!(book.w > 400.0, "book too narrow: {book:?}");
        assert!(book.h > 300.0, "book too short: {book:?}");
        // The book names its spread column, so a Shift+J/K step knows where to
        // land: the cover spread, then the pair, and nothing past the last one.
        assert_eq!(surface.spread_count(), 2);
        let first = surface.spread_box(0).unwrap();
        assert_eq!(first.w, surface.open_box(0).unwrap().w);
        assert_eq!(first.h, surface.open_box(0).unwrap().h);
        assert_eq!(surface.spread_box(1).unwrap().y, first.y + first.h);
        assert!(surface.spread_box(2).is_none());
    }

    #[test]
    fn typst_resolves_includes_and_assets_relative_to_the_file() {
        // The directory of the opened file is the project root: `#include`
        // reads a sibling source and `#image` a sibling picture.
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();
        write(dir, "chapter.typ", "= Chapter\nIncluded content.");
        // A 4x2 red PNG, made with the image crate.
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
        let surface = TypstSurface::open(&main).unwrap();
        let image = surface.render(View::whole(0, 1.0)).unwrap();
        // The red pixels of the picture must show in the render.
        assert!(
            image
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .any(|px| px[0] > 200 && px[1] < 60)
        );
    }

    #[test]
    fn typst_page_renders_a_region() {
        // The region anchor comes from the PDF backend: a region view
        // rasterizes only the requested rectangle, which is the path a
        // settled zoom refine uses.
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "doc.typ", TWO_PAGES);
        let surface = TypstSurface::open(&path).unwrap();
        let region = crate::surface::Region {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 50.0,
        };
        let image = surface.render(View::region(0, region, 2.0)).unwrap();
        assert_eq!(
            (image.width, image.height),
            (crate::dim::ImgW::new(200), crate::dim::ImgH::new(100))
        );
    }

    #[test]
    fn typst_compile_errors_are_readable() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "bad.typ", "#set page(width: 10cm)\n#foo()\n");
        let err = match TypstSurface::open(&path) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("a bad document must not open"),
        };
        assert!(err.contains("error:"), "unexpected message: {err}");
        assert!(err.contains("bad.typ:2"), "no location: {err}");
    }

    #[test]
    fn typst_missing_file_fails_cleanly() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("nope.typ");
        assert!(open_surface(&path).is_err());
    }
}
