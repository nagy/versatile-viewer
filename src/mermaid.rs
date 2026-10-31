//! Mermaid backend: render a `.mmd` file to SVG, then hand it to the SVG
//! backend.
//!
//! This file owns the merman dependency.
//! The rest of the viewer only sees the [`Surface`] trait.
//!
//! Pipeline at open time: read the source, lay it out and emit an SVG string,
//! then parse that string with `usvg`.
//! The parsed tree becomes a [`VectorSurface`], so a Mermaid diagram inherits
//! the whole vector path: one base raster, a window patch at the view scale, a
//! settled-zoom refine, and the grid thumbnail.
//! No second rasterizer exists in this module.
//!
//! Layout runs through the `elk` cargo feature, a default feature, so a source
//! that asks for ELK (`%%{init: {"layout": "elk"}}%%`) keeps its own layout.
//! The engine comes from the merman fork that `[patch.crates-io]` names: there
//! `merman-layout-elk` drives `elk-layout`, which ships
//! `EPL-2.0 OR GPL-3.0-or-later`, so this crate takes the GPL arm and GPL-3.0
//! section 13 combines it with our AGPL-3.0-or-later.
//! Without the feature merman falls back to dagre without a word.
//! `architecture-beta` answers a capability error either way, because the
//! feature list leaves that family out.
//!
//! The view passes [`SvgPipeline::resvg_safe`], the preset merman uses for its
//! own PNG and PDF export.
//! That preset turns a `<foreignObject>` HTML label into an SVG `<text>`
//! element and sanitizes the stylesheet for a resvg consumer.
//! Without it every label of a flowchart, a class diagram or a mindmap
//! disappears, because `usvg` draws no HTML.
//!
//! Label boxes measure with a real font file, not with merman's fixed metric
//! table.
//! [`FONTS`] builds one font context per process from the faces of
//! [`crate::svg::fontdb`], and that is the same database the painter uses, so a
//! fallback label box fits the text it holds.
//! A host with no readable font file falls back to the deterministic metrics,
//! and the diagram still renders.
//!
//! Mermaid text has no magic bytes, so [`is_mermaid`] tests the extension, the
//! same way [`crate::typst::is_typst`] does.
//!
//! A panic inside the renderer becomes [`SurfaceError::Render`], because the
//! open runs on a decode worker.
//! A source that holds no diagram, or one that the parser rejects, becomes a
//! readable message, so the grid marks the cell as failed.
//!
//! A diagram paints no page: the emitted SVG draws the boxes and the labels,
//! and every other pixel stays transparent.
//! The surface therefore paints [`BACKGROUND`], the white paper a Mermaid
//! diagram assumes, under every render, so the lines never sit on the viewer
//! default.
//! That color lives here, and no other module knows the choice.

use std::{borrow::Cow, collections::BTreeSet, path::Path, sync::LazyLock};

use merman::{
    OperationControl, RenderOutput, RenderRequest, Renderer, SvgEnvironment, SvgRequest,
    svg::{
        SvgPipeline,
        export::{NativeFontContext, NativeFontLimits, NativeFontOptions},
    },
};
use raylib::color::Color;

use crate::surface::{Surface, SurfaceError, VectorSurface};

/// The color a diagram paints on.
///
/// Mermaid emits no page rectangle, so every pixel the diagram does not draw
/// stays transparent.
/// The SVG backend therefore paints this color under the tree
/// (`VectorSurface::with_background`), so the paper rides the pixels of the
/// grid cell, the thumbnail and the window patch, and the viewer window keeps
/// its own background.
const BACKGROUND: Color = Color::new(0xff, 0xff, 0xff, 0xff);

/// The directive that turns HTML labels off.
///
/// Mermaid reads an `%%{init}%%` directive before the diagram, and the last
/// directive wins, so a source with its own directive keeps the choice it
/// states later in the file.
const SVG_LABELS: &str = "%%{init: {\"htmlLabels\": false}}%%\n";

/// The most bytes this backend reads for one `.mmd` file.
///
/// A Mermaid source is text, so a real file stays far below this.
/// The cap bounds a pathological file, and the renderer has its own limits on
/// top of it.
const MAX_INPUT: usize = 8 * 1024 * 1024;

/// The font budget of one native measurement context.
///
/// A full system scan holds far more than merman's default 64 MiB (a typical
/// Linux host carries about 75 MiB of faces), so the footprint scale is raised.
/// The count limits stay high enough for a large font directory and low enough
/// that a runaway tree cannot allocate without bound.
const FONT_MAX_BYTES: usize = 512 * 1024 * 1024;
/// See [`FONT_MAX_BYTES`].
const FONT_MAX_FILES: usize = 4096;
/// See [`FONT_MAX_BYTES`].
const FONT_MAX_FACES: usize = 8192;

/// The Mermaid engine, built once per process.
///
/// A render builds its own layout state and shares no mutable session, so one
/// renderer serves the grid decodes, the image view and every worker at once.
static RENDERER: LazyLock<Renderer> = LazyLock::new(Renderer::new);

/// The font context for label measurement, built once per process.
///
/// The load reads and parses every face of the host database, which costs a
/// few hundred milliseconds, so one context serves every diagram.
/// The inner value is an `Arc`, so a clone is cheap.
/// A host without a readable face answers `None`, and the render falls back to
/// the deterministic metrics.
static FONTS: LazyLock<Option<NativeFontContext>> = LazyLock::new(load_fonts);

/// Is this a Mermaid source file?
/// The test is the extension only: a `.mmd` file is plain UTF-8 text, so there
/// is no magic to sniff.
#[must_use]
pub fn is_mermaid(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("mmd"))
}

/// Read the `.mmd` file at `path` and open it as a vector surface.
///
/// The render happens here, once, and the parsed tree serves every later paint.
/// The answer is boxed, so the stub of a build without this feature has the
/// same signature.
///
/// # Errors
///
/// The function errors when the file cannot be read, the size cap fails, the
/// source is not UTF-8, the renderer rejects the diagram, or `usvg` rejects the
/// emitted SVG.
pub fn open(path: &Path) -> Result<Box<dyn Surface>, SurfaceError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() > MAX_INPUT {
        return Err(SurfaceError::Unsupported(format!(
            "the Mermaid source is larger than {MAX_INPUT} bytes"
        )));
    }
    let source = String::from_utf8(bytes)
        .map_err(|_| SurfaceError::Unsupported("the Mermaid source is not UTF-8".into()))?;
    let svg = render(&svg_label_source(&source))?;
    Ok(Box::new(
        VectorSurface::from_vector(crate::svg::from_str(&svg)?).with_background(BACKGROUND),
    ))
}

/// Render one Mermaid source to an SVG string.
///
/// The call passes the resvg-safe pipeline and the process font context.
/// The renderer runs behind `catch_unwind`, because it runs on a decode worker
/// and a panic must not cross that boundary.
fn render(source: &str) -> Result<String, SurfaceError> {
    let request = SvgRequest {
        environment: environment(),
        pipeline: Some(SvgPipeline::resvg_safe()),
        ..SvgRequest::default()
    };
    let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        RENDERER.render(RenderRequest::svg(source, OperationControl::new(), request))
    }))
    .map_err(|_| SurfaceError::Render("the Mermaid renderer panicked".into()))?;
    match rendered {
        Ok(RenderOutput::Svg(Some(svg))) => Ok(svg.svg().to_owned()),
        Ok(_) => Err(SurfaceError::Render(
            "no Mermaid diagram found in the source".into(),
        )),
        Err(error) => Err(SurfaceError::Render(error.to_string())),
    }
}

/// Add the SVG-label directive to a source.
///
/// The directive goes before the diagram and after a leading YAML front matter
/// block, because Mermaid requires the front matter to come first.
/// A source with an unterminated front matter stays untouched: prepending the
/// directive would hide that open block and change the error of the renderer.
fn svg_label_source(source: &str) -> Cow<'_, str> {
    if let Some(end) = front_matter_end(source) {
        let mut out = String::with_capacity(source.len() + SVG_LABELS.len());
        out.push_str(&source[..end]);
        out.push_str(SVG_LABELS);
        out.push_str(&source[end..]);
        return Cow::Owned(out);
    }
    if source.trim_start().starts_with("---") {
        return Cow::Borrowed(source);
    }
    let mut out = String::with_capacity(source.len() + SVG_LABELS.len());
    out.push_str(SVG_LABELS);
    out.push_str(source);
    Cow::Owned(out)
}

/// The byte offset just after a leading YAML front matter block.
///
/// A block opens on a first line that holds only `---` and closes on the next
/// such line.
/// The function answers `None` when the block is missing or unterminated.
fn front_matter_end(source: &str) -> Option<usize> {
    let mut lines = source.split_inclusive('\n');
    let first = lines.next()?;
    if first.trim_end() != "---" {
        return None;
    }
    let mut offset = first.len();
    for line in lines {
        offset += line.len();
        if line.trim_end() == "---" {
            return Some(offset);
        }
    }
    None
}

/// The SVG environment of one render: the process font context when the host
/// has a readable face, and the deterministic default otherwise.
fn environment() -> SvgEnvironment {
    let Some(context) = FONTS.as_ref() else {
        return SvgEnvironment::deterministic();
    };
    SvgEnvironment::deterministic()
        .with_native_font_context(context.clone())
        .unwrap_or_else(|_| SvgEnvironment::deterministic())
}

/// Build the native font context from the faces of the SVG font database.
///
/// The walk collects the distinct face files, so a face that several entries
/// share (a variable font) counts once.
/// A database with no readable file answers `None`.
fn load_fonts() -> Option<NativeFontContext> {
    let files: BTreeSet<&std::path::Path> = crate::svg::fontdb()
        .faces()
        .filter_map(|face| match &face.source {
            fontdb::Source::File(path) | fontdb::Source::SharedFile(path, _) => {
                Some(path.as_path())
            }
            fontdb::Source::Binary(_) => None,
        })
        .collect();
    if files.is_empty() {
        return None;
    }
    let limits = NativeFontLimits {
        max_bytes: FONT_MAX_BYTES,
        max_files: FONT_MAX_FILES,
        max_faces: FONT_MAX_FACES,
        ..NativeFontLimits::default()
    };
    let mut options = NativeFontOptions::default().with_limits(limits);
    for file in files {
        options = options.with_font_file(file);
    }
    options.load(&OperationControl::new()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::{View, open_surface};

    /// A source with two nodes and an edge label.
    const FLOW: &str = "flowchart TD\n  A[Start here] --> B[Done]\n";

    fn write(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn svg_label_source_goes_before_the_diagram() {
        let source = svg_label_source("flowchart TD\n  A --> B\n");
        assert!(source.starts_with(SVG_LABELS));
        assert!(source.ends_with("flowchart TD\n  A --> B\n"));
    }

    #[test]
    fn svg_label_source_follows_a_front_matter() {
        // A YAML block must stay the first bytes of the file, so the
        // directive lands after its closing line.
        let source = svg_label_source("---\ntitle: Plan\n---\nflowchart TD\n  A --> B\n");
        assert_eq!(
            source,
            format!("---\ntitle: Plan\n---\n{SVG_LABELS}flowchart TD\n  A --> B\n")
        );
    }

    #[test]
    fn svg_label_source_keeps_an_unterminated_front_matter() {
        let source = svg_label_source("---\ntitle: Plan\nflowchart TD\n");
        assert_eq!(source, "---\ntitle: Plan\nflowchart TD\n");
    }

    #[test]
    fn is_mermaid_tests_the_extension() {
        assert!(is_mermaid(Path::new("diagram.mmd")));
        assert!(is_mermaid(Path::new("a/b/DIAGRAM.MMD")));
        assert!(!is_mermaid(Path::new("diagram.mermaid")));
        assert!(!is_mermaid(Path::new("diagram.txt")));
        assert!(!is_mermaid(Path::new("mmd")));
    }

    #[test]
    fn mermaid_opens_and_reports_a_vector_size() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "diagram.mmd", FLOW);
        let surface = crate::surface::open_surface(&path).unwrap();
        assert_eq!(surface.plane_count(), 1);
        // The base raster clamps its long side into the vector floor.
        let bounds = surface.bounds(0).unwrap();
        assert!(bounds.w >= 1024.0 || bounds.h >= 1024.0);
        assert!(surface.can_refine());
    }

    #[test]
    fn mermaid_paints_its_white_paper() {
        // The diagram draws no page, so the render carries the paper under its
        // lines: the grid cell, the thumbnail and the window patch all show
        // the diagram on white, and no window color arrives late.
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "diagram.mmd", FLOW);
        let surface = crate::surface::open_surface(&path).unwrap();
        let image = surface
            .render(crate::surface::View::whole(0, 0.25))
            .unwrap();
        // The top-left corner holds no box of a `flowchart LR`, so it shows
        // the paper.
        assert_eq!(&image.data[..4], &[0xff, 0xff, 0xff, 0xff]);
        assert_eq!(
            &image.data[image.data.len() - 4..],
            &[0xff, 0xff, 0xff, 0xff]
        );
    }

    #[test]
    fn mermaid_paints_text_not_only_boxes() {
        // A label box without its text is the failure mode of a raw merman
        // SVG with `usvg`, so the test counts the pixels the diagram draws
        // over its paper instead of trusting the size alone.
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "diagram.mmd", FLOW);
        let surface = crate::surface::open_surface(&path).unwrap();
        let image = surface.render_whole(0, 1.0).unwrap();
        let paper = [BACKGROUND.r, BACKGROUND.g, BACKGROUND.b, BACKGROUND.a];
        let painted = image
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| **px != paper)
            .count();
        assert!(painted > 0, "the diagram painted no pixel");
    }

    #[test]
    fn mermaid_renders_a_window_region() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "diagram.mmd", FLOW);
        let surface = open_surface(&path).unwrap();
        let bounds = surface.bounds(0).unwrap();
        let region = crate::surface::Region {
            x: 0.0,
            y: 0.0,
            w: (bounds.w / 4.0).max(1.0),
            h: (bounds.h / 4.0).max(1.0),
        };
        let image = surface
            .render(View {
                plane: 0,
                region: Some(region),
                scale: 2.0,
            })
            .unwrap();
        assert_eq!(image.width.get(), (region.w * 2.0).round() as u32);
        assert_eq!(image.height.get(), (region.h * 2.0).round() as u32);
    }

    #[test]
    fn mermaid_rejects_a_source_without_a_diagram() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write(tmp.path(), "diagram.mmd", "this is not mermaid\n");
        let error = crate::surface::open_surface(&path).err().expect("no error");
        assert!(
            matches!(error, SurfaceError::Render(_)),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn mermaid_rejects_non_utf8_content() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("bad.mmd");
        std::fs::write(&path, [0xffu8, 0xfe, 0x00]).unwrap();
        let error = crate::surface::open_surface(&path).err().expect("no error");
        assert!(
            matches!(error, SurfaceError::Unsupported(_)),
            "unexpected error: {error}"
        );
    }
}
