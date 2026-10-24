# Surface: one region-based render trait

Status: accepted

## Context

The viewer shows many kinds of content: raster images today, pages of a
document later, and possibly a viewport of a continuous plane. Each kind has
its own renderer and its own natural unit. The grid, the streaming loader and
the image view must not each learn every format.

## Decision

Every showable source is a `Surface` (`src/surface.rs`). A surface is a set of
planes. A plane has a natural extent in surface units. Rendering takes a
`View`: which plane, which region of it, and the output scale.

```rust
pub struct Size { pub w: f32, pub h: f32 }
pub struct Region { pub x: f32, pub y: f32, pub w: f32, pub h: f32 }

pub struct View {
    pub plane: usize,
    pub region: Option<Region>, // None = the whole plane
    pub scale: f32,             // output pixels per surface unit
}

pub trait Surface: Send + Sync {
    fn plane_count(&self) -> usize;
    fn bounds(&self, plane: usize) -> Result<Size, SurfaceError>;

    // The one method every backend writes.
    fn render_whole(&self, plane: usize, scale: f32) -> Result<DecodedImage, SurfaceError>;

    // Default: render_whole, then crop the region. A backend with an anchor
    // overrides this for the cheaper path.
    fn render(&self, view: View) -> Result<DecodedImage, SurfaceError> { .. }

    fn stream_path(&self) -> Option<&Path>;
    fn previews(&self) -> bool;
}
```

The callers see surfaces, plane indices and regions only, never file formats.
`open_surface` decides the concrete surface from sniffed content: a raster
file becomes an `ImageSurface`, and a vector file becomes a `VectorSurface`
that keeps its parsed `usvg` tree, so a region render paints only the pixels
the caller asked for.

## Why region-based

A paged source needs union of pages and a plane each. A continuous source
needs no pages, but its viewport is not the whole plane. `region: Option` is
the one shape that covers both: whole-plane for a page, a sub-rectangle for a
viewport. The trait then needs no second variant and no map-specific methods.

## Why plane_count and bounds stay separate from render

Layout needs the extent before any pixels exist. The grid sizes a cell, the
view fits a scale. `bounds` is cheap and must not decode.

## Considered options

- `page_count` / `page_info` / `render(page, scale)`: simpler, but a
  continuous source with no pages fits it badly.
- A `bounds()` plus `render(region, scale)` pair with no planes: makes a
  multi-page source into region selection, which the grid cannot navigate.

## Consequences

- Adding a source means one `impl Surface` plus one arm in `open_surface`.
- `scale` is output pixels per surface unit. 1.0 is natural size. A source
  whose natural unit is not a pixel (a typographic point, say) can still
  render a whole plane at 1.0 and get a pixel buffer.
- `render_whole` is the required method; `render` is a provided method whose
  default renders the whole plane and crops the region out of it. A backend
  with a region anchor (a PDF rasterizer) overrides `render` to pay only for
  the pixels the caller asked for. A backend with no anchor and no override
  cannot be wrong: it inherits a correct crop. A backend that overrides
  `render` and refuses a region returns `SurfaceError::Unsupported`; it must
  never return the whole plane for a region view.
- `stream_path` is the one file-shaped hole: a progressive byte-stream
  source. A source with no file returns `None`.
- A grid cell holds an `EntrySource`: either a file to open lazily, or the
  surface handle and plane of an already-open source. So an in-memory source
  browses like a directory, with no file behind it.
- The thumbnail cache keys on a file path, so a cell with no file skips the
  cache and the placeholder pass by construction.
