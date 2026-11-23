# A display list beside the raster render

Status: proposed. The treemap spike lands on this branch, so the record and its
first implementation travel together.

## Context

`Surface` (`crates/vv-source/src/surface.rs`) answers one `DecodedImage` per
`View`. A resolution-independent source is therefore only as sharp as its last
rasterization, and everything that decides *when* to rasterize again sits above
the trait rather than inside it:

- `crates/vv-core/src/region.rs` holds the policy: `MARGIN`, `SCALE_SLACK`,
  `QUIET`, `Patch`, `worth_painting`, `patch_covers_at_scale`.
- `crates/vv-view/src/view.rs` holds the bookkeeping: `ViewState`'s `crop`,
  `prefetch`, `crop_gen`, `crop_pending`, `crop_seen` and `crop_quiet_at`
  fields, plus `tick_crop`, `drain_crops`, `prefetch_spread`, `patch_dest`,
  `draw_patch`, `same_patch` and `patch_covers`.

That is roughly five hundred lines whose whole subject is when to rasterize and
where to blit the answer. It buys two things, and only one of them needs a
raster:

- **A paint costs window pixels, never the document.** The region anchor keeps a
  deep zoom as cheap as a shallow one. This is worth keeping whatever else
  changes.
- **Sharpness arrives 150 ms after the view stops moving, and only at the scale
  it was painted.** A picture stretched past `SCALE_SLACK` is re-rasterized, and
  until it lands the GPU filter stretches the old patch. This is the machinery
  the viewer actually feels.

Two of the three backends already build a display list and then throw it away.
`VectorSurface` keeps a `usvg::Tree` behind an `Arc` and hands it to
`resvg::render`, which walks it into a `tiny_skia::Pixmap`
(`crates/vv-source/src/svg.rs`). `PdfSurface` keeps a `Pdf` whose interpreter
emits drawing instructions to a `Device` sink, and hands them to hayro's own
sink, which walks them into a `vello_cpu` context for a `Pixmap`
(`crates/vv-source/src/pdf.rs`). The list exists in both cases; only the CPU
rasterizer has ever consumed it.

## Decision

Add a display list beside the raster, on a trait method that defaults to absence.

```rust
// crates/vv-core/src/scene.rs, re-exported as vv_core::scene
pub enum Primitive {
    Path  { path: kurbo::BezPath, transform: kurbo::Affine, paint: Paint, fill: FillRule, clip: ClipId },
    Rect  { rect: kurbo::Rect,    transform: kurbo::Affine, paint: Paint, clip: ClipId },
    Glyphs { run: GlyphRun,       transform: kurbo::Affine, paint: Paint, clip: ClipId },
    Image { pixels: DecodedImage, transform: kurbo::Affine, clip: ClipId },
}

// Surface, defaulted, so every backend compiles and no backend is coerced.
// No scale and no region: the list is in surface units.
fn display(&self, plane: usize) -> Result<Option<Arc<[Primitive]>>, SurfaceError> {
    Ok(None)
}
```

`None` keeps today's raster path, so `render`/`render_whole` stay and a backend
opts in one at a time. `vv-gpu` draws the list with `vello` on the device `Gpu`
already holds, into a window-sized target that the existing quad pipeline
composites as one more textured draw, which is why `Renderer::render_into` keeps
its signature and the offscreen GPU paints keep working.

**The list is built once per plane and drawn per frame.** It is written in
surface units, so a zoom or a pan changes the *transform*, not the list: the
view builds no list, waits for nothing and holds no debounce, which is the whole
point. Vello bins the primitives into tiles and clips to the viewport, so
primitives off screen cost nothing per frame, and a bounding-box cull over the
cached list stays available as an optimization if a pathological document ever
needs it. The cost of a document is then one interpretation pass per plane
instead of one per settled zoom, and a deep zoom costs exactly nothing — the
property the raster path cannot have, because it can only make the paint smaller.

This is why the method takes a plane and not a [`View`]: a `View` describes a
paint, and there is no paint any more. `bounds` still answers the plane's extent,
and the frame mapping of `region::Frame` still answers where the plane's box lands
in the window for the draw.

**The type lives in `vv-core`, not in `vv-source`.** `vv-core` already holds
`dim`, `draw`, `region`, `region_sel`, `rotate` and `tween`: value types with no
backend knowledge. A primitive is the same kind of thing, and both sides name it
through a crate they already depend on, so `vv-gpu` never learns that a source
exists and ADR 0004's edges stand unchanged. Putting it in `vv-source` would make
the renderer depend on the source layer for one type.

That placement moves `DecodedImage` down with it. The `Image` leaf carries
pixels, and `vv-core` sits below the crate that holds the pixel type, so
`vv_pixels::DecodedImage` becomes `vv_core::DecodedImage` and `vv-pixels`
re-exports it, which is the pattern every crate root already follows. Nothing at
a call site changes: every consumer reaches the type through a re-export
(`vv_source::DecodedImage`, `vv_gpu::DecodedImage`, `vv_pixels::DecodedImage`),
the size pair it is made of is already `dim::ImgW`/`dim::ImgH`, and the resample
policy (`resample_rgba`, `cap_texture_side`, `MAX_TEXTURE_SIDE`) stays in
`vv-pixels` beside the decoders that obey it. ADR 0004 carries the amendment.

**The fallback is per subtree, not per plane.** A `push_transparency_group` with
a soft mask, a tiling pattern, or a blend the GPU path cannot express rasterizes
that group into one `Image` leaf and keeps the rest vector. That is why the
`Primitive` type has an `Image` variant carrying a `DecodedImage` and not a GPU
handle: `vv-core` sits below `vv-gpu` and names no wgpu type, so the leaf is raw
pixels and the renderer keeps a texture cache keyed by leaf identity. The
convention is unchanged from [`vv_pixels::DecodedImage`], which crosses the same
edge the same way.

**`kurbo` becomes a direct dependency of `vv-core`**, with a why-comment. It is
already in the build through hayro, resvg and peniko (`peniko 0.6.1` wants
`kurbo ^0.13.1`, and the lock holds exactly `kurbo 0.13.1`), and the path type of
a display list belongs to neither a decoder nor the renderer. `vv-core` keeps its
`euclid` rects for the window and surface spaces: `dim::Rect` and
`region::Crop` stay the tagged rects of the view, and a `kurbo` path is the
untagged geometry of one primitive.

## Why not `Iter<Item = impl Surface>`

A `Surface` is a source: it has planes, a natural extent, and a decode that costs
what it costs. A rect has none of those, so `bounds` and `render` on a rect would
be a trait that lies about its implementors. A primitive also needs no trait at
all — the set is closed, the renderer has to handle every variant exhaustively
anyway, and `impl` in that position cannot be stored in the `Option` the
fallback needs. One enum plus one argument is the whole surface.

## Why no fork of hayro

A fork was on the table, because hayro's private compositing half looked like the
missing piece for the fallback rule above. It is not needed, and this is the
finding worth recording, because the obvious fork would buy almost nothing.

`hayro::Renderer` is a CPU sink. It holds `ctx: &'a mut vello_cpu::RenderContext`,
and every method of the private half — spread over `path.rs`, `glyph.rs`,
`glyph_cache.rs`, `image.rs`, `mask.rs`, `paint.rs`, `tiling.rs` and `clip.rs`,
about two thousand lines — ends in `ctx.set_paint`, `ctx.fill_path`,
`ctx.push_layer` or `ctx.set_mask`. Publishing the type publishes the rasterizer
we already have, and there is no semantics layer to lift out by changing a
visibility, because the PDF rules and the pixel calls share one body:
`path.rs::stroke_path` applies the minimum-line-width rule and then fills.

The interpreter is usable as it stands. `hayro_interpret::interpret_page` is
public and generic over `&mut impl Device<'a>`; `Context::new`,
`InterpreterCache::new` and the `hayro_syntax` `Pdf`/`Page` types are public; and
`DrawProps` carries `transform`, `paint`, `soft_mask` and `blend_mode`, so a
`Device` in `pdf.rs` that fills a `Vec<Primitive>` is a small amount of code.

Two facts make the fork unnecessary rather than merely inconvenient:

- **The re-entry points are already generic.** `SoftMask::interpret(&mut impl Device<'a>)`
  (`hayro-interpret/src/x_object/soft_mask.rs:136`) and
  `Pattern::interpret(..., &mut impl Device)` (`hayro-interpret/src/pattern.rs:218`)
  are public and take any device. Our own device calls them, draws the mask group
  or the tile into a GPU layer, and hayro's private wrappers
  (`mask.rs::draw_soft_mask`, `tiling.rs::render_tiling`) are never reached: they
  exist only to build a `vello_cpu` context for a `Pixmap` and hand back a
  `Mask`/`Image`. Per-subtree fallback therefore needs no fork at all.
- **Image XObjects are already decoded below the private layer.** `ImageData`
  (`hayro-interpret/src/types.rs:216`) carries `Vec<u8>` samples as `Rgb` or
  `Luma`, so JBIG2, JPEG 2000, CCITT and LZW are finished before a `Device` sees
  them. `image.rs`'s 828 lines are colour conversion, `pic_scale` resampling and
  mask application, which is work a GPU does better and which no GPU path would
  call.

What a fork would still offer is glyph outlines from `glyph.rs`, and `skrifa`
supplies those instead. The question reopens only if a page uses a `DrawMode` or a
blend mode that neither vello nor a layer can express, and even then the change
wanted is a `Device` parameter on the two wrapper functions above, not a `pub`.

No upstream change is requested either. This record uses `SoftMask::interpret`,
`Pattern::interpret` and `ImageData` exactly as published, and keeps hayro a
versioned dependency of one file rather than a patched source tree.

One ownership wrinkle follows from the seam and belongs here: `GlyphRun` and
`PositionedGlyph` are borrowed views (`glyphs: &'run [PositionedGlyph<'a>]`, no
constructor, `glyphs()` and `writing_mode()` as the accessors), so
`Primitive::Glyphs` cannot hold one. The primitive owns a run of its own — glyph
id, transform, and an `Arc` font handle — and the PDF path needs a handle to the
face hayro decoded, not a claim that it is a file on this machine.

## Alternatives

**Tune the CPU policy instead.** Raise the paint rate during the ease, or repaint
every frame at a capped resolution. It adds no dependency and keeps every
guarantee, but it cannot make a zoom sharp *while* it moves, and the patch
bookkeeping stays.

**Hand-write the pipelines on wgpu 30.** No new dependency and the existing
one-pipeline style, at the cost of analytic anti-aliasing (stencil-and-cover) and
a glyph atlas written here. Months rather than weeks.

**`vello 0.12`** is the dependency this record picks. It is pure Rust, its
`wgpu ^30.0.0` matches the `wgpu 30.0.2` this tree already locks, and it shares
`Gpu`'s device rather than opening a second. The bill is a second renderer beside
the quad pipeline, a window-sized target, raised device limits, and proving the
`checks.gpu` lavapipe run can execute its compute pipelines.

**Fall back per plane.** One unsupported primitive sends the whole plane to the
raster path. Trivially correct and keeps today's picture exactly, but one exotic
page loses the entire gain, and books are pages of a compiled document.

## Consequences

- The first backend to answer `Some` is `TreemapSurface`, and the spike converts
  no other. It has no parser, no document format and no external file: the layout
  is already a pure function of the tree and the frame, so the primitive list is
  the layout itself and a mistake in the seam cannot hide behind a decoder.
- An SVG display list retires the region patch for vector files: `tick_crop`,
  `drain_crops`, `CropMsg`, `Prefetch`, `prefetch_spread`, `patch_dest`,
  `draw_patch`, `same_patch`, `patch_covers`, `can_refine`, and the paint half of
  `region.rs`. `region::Frame` (`to_screen`, `to_image`, `crop`) stays, because
  the cull box and the quad's source rect still come from it.
- `BookSurface` stops rasterizing a page and blitting it: a book's plane becomes
  its pages' lists concatenated with a translated transform, which is the
  cheapest change in the tree and the slowest render today
  (`crates/vv-source/src/book.rs`). It is also the one place where "once per
  plane" needs care, because a book's plane is the whole document: a five
  hundred page book built as one list is hundreds of megabytes. The granularity
  of the cache therefore belongs to the backend — the book builds and keeps the
  list of a page row, which is the unit it already navigates in — and the trait
  asks only for a plane.
- `TreemapSurface` paints its rectangles into an SVG document it builds as text
  (`crates/vv-source/src/treemap.rs`), which `svg::from_str` then parses. As
  primitives it drops that round trip. It also proves the seam first, because it
  has no parser, no document format and no external file: the layout is already
  a pure function of the tree and the frame.
  Its labels are the one thing it does not simplify: `treemap::svg` emits
  `<text x=.. font-family="sans-serif" font-size=..>` per label (see
  `leaf_label` and `band_for`), so a treemap of primitives has text on its first
  day. The spike answers each label as an `Image` leaf, rasterized through the
  resvg path that already draws it, so no glyph machinery is needed and the
  fallback path gets exercised with real data on the first backend that runs.
  A follow-up replaces those leaves with `Glyphs`, which is where the font work
  starts.
- Glyph runs are the last and largest piece, and the one that stays raster the
  longest: vello fills outlines through `skrifa`, but the PDF side reaches glyphs
  through `hayro-font` while the SVG side reaches them through `fontdb`, so one
  glyph atlas has two producers and two keys.

## Testing

A converted backend asserts on geometry, not on pixels: the primitive list of a
`View` (counts, order, bounds, the `None` a plain raster source answers), and the
picture against the raster path with a perceptual tolerance. vello, resvg and
hayro each do their own anti-aliasing and hayro hints glyphs, so a tight
per-channel budget would be a test of two rasterizers agreeing rather than a test
of this seam. The comparison exists to catch a dropped primitive, a wrong cull
box and a wrong transform — the failures that arrive with the change.

## Open questions

- The work lands on this branch, `display-list`, so the record and its first
  implementation travel together.
- No stopwatch has been run. `PdfSurface::rasterize` at `window x 1.5` may already
  be a few milliseconds, in which case this work buys feel and deleted code
  rather than throughput, and the budget should say so. The claim this record
  needs to be true is narrower: that one interpretation pass per plane is cheaper
  than one rasterization per settled zoom, and that a zoom then costs nothing.
- `vello` wants compute pipelines and larger limits than the viewer asks for
  today, and `checks.gpu` runs on Mesa's lavapipe. Whether that gate stays green
  is unproven.
- A pixel-exactness bar is not set, and the tests deliberately avoid one.
  `resvg` and hayro do analytic anti-aliasing with their own glyph hinting;
  vello's output will differ on the same page, so the assertions are geometry
  plus a perceptual tolerance.
