# A scene document a producer writes

Status: accepted. Implemented on the tree that carries this record.

## Context

ADR 0007 put a display list beside the raster render, and every source that
answers one is a document this tree parses: a treemap lays itself out, an SVG
tree is walked, a PDF page is interpreted. The list has therefore always been
built *inside* the process that paints it, and the seam is
`Surface::display(plane)`.

A program that owns its own layout needs the opposite direction. A diagram
engine, a plot, a map, a drawing produced by a tool this tree knows nothing
about: each of them computes its own geometry — its axes, its scales, its
shapes, its hit targets — and today it has to paint pixels, so the picture
arrives as one raster at one size and a zoom can only stretch it. Handing the
list over instead puts the drawing into the path this tree already has for
shapes: encoded once per band and placed per frame, so a pan or a zoom costs a
copy of the encoded stream and no walk of the geometry.

Every fact such a producer needs is already in place. The vocabulary has a
variant for a rect, a path with a fill and a `kurbo::Stroke`, a glyph run, a
clip and an opacity layer. `SceneView::draw` encodes a list once per band and
places it per frame. What is missing is the wire: nothing here has ever read a
list from bytes it did not parse into a tree of its own.

## Decision

**`serde` on the vocabulary, in `vv-core`.** `FillRule`, `Stop`, `Spread`,
`Paint`, `Stroke`, `PlacedGlyph`, `GlyphRun` and `Primitive` derive `Serialize`
and `Deserialize`, so a variant this tree adds is a variant a document may state
and a renamed field breaks a document loudly. The geometry's serde is `kurbo`'s
(`features = ["serde"]` covers `Rect`, `BezPath`, `Stroke` and `Affine`), and
`serde`'s `rc` feature is what lets the two fields already held behind an `Arc`
— `Paint::Linear::stops` and `GlyphRun::font` — cross the wire without copying.
`draw::Color` gains the same two derives, because a paint names one and the four
channels are already the bytes a decoder wrote.

**The picture leaf stays off the wire.** `Primitive::Image` carries a
`DecodedImage`, and a picture is bytes rather than a shape: the document that
would carry one is an image format, and this tree already reads those. The
variant carries `#[serde(skip)]`, so a document that names it is refused as an
unknown variant rather than painting the rest of the list and leaving a hole,
and a list that holds one does not serialise either.

**The document lives in `vv-source`, not in `vv-core`.** `scene_file` holds
`SceneDoc { version, plane, primitives }`, `Extent`, `Scene` and `SceneError`,
and the `Scene` a reader answers is a `Size` and an `Arc<[Primitive]>`. The
vocabulary is a value type and lives in `vv-core`; a *file format* is a source
concern and lives in the layer that opens files. `vv-core` gained no
`serde_json`, and it stays a layer that reads nothing.

**One version field, read on its own.** `VERSION` is 1 when this route landed
and 2 once a document could state its viewport (`docs/adr/0009`); `OLDEST` names
the oldest version the reader still opens. The header is parsed
first, from the same bytes, so a document that states another number is refused
by version and not by whichever field this tree happens to reach first: a
producer that moved ahead reads why its document did not open, and no list is
allocated for a document about to be dropped. Every struct and every struct
variant of the vocabulary denies an unknown field, so the producer and this
reader agree on the whole shape or on nothing, down to the fields of one
primitive.

**The entrance is `vv --scene <file>`, and it is not a `Surface`.**
`--scene` conflicts with the positional path and takes its place, so one run
shows one thing and a reader can tell from the command line which reader ran.
The launch reads the document on the launch thread (`Launch::scene`), keeps the
plane and the list, and `start_launch` puts the list into `ViewState::scene`
with no source adopted, no loader started and no texture uploaded. A malformed
file therefore fails before a window opens.

**The launch frames the whole extent and sizes no window from it.** The extent is
in the units the list was written in, and one of those need not be a pixel, so
only the fit decides how many window pixels a unit covers. The window keeps its
default 1024 by 768 and the first frame fits the extent into it, exactly as a
book fits its first spread.

**With no source adopted, a frame that draws no list draws nothing.**
`wants_patch` reads `list_drawn`, and a source is what paints a fallback patch,
so a scene launch leaves `source` empty: `refresh_plane` on a resize records the
offer and stops, and a frame the pass refuses (`SceneView::draw` past
`LINE_BUDGET`) shows the window background instead of a plane texture that never
existed. That is the honest picture rather than a new policy: nothing has
rasterized this plane and nothing can.

## Why not content sniffing

`AGENTS.md` says to dispatch a decode on sniffed content and never on the file
name, and the two exceptions are a converter rule and a `.treemap` path. A scene
document cannot join them: JSON has no magic, `sign::detect` reads two bytes
before it decides, and the honest sniff would be "it starts with `{` and parses
as our document", which is a parser call inside a router. A third `sign::Kind`
would also make every existing caller of `detect` learn what to do with a source
that has no raster, which is a wider change than the feature.

## Why not a mirrored wire enum

A producer in another process will mirror this vocabulary in its own crate — its
own types, its own schema, meeting this tree through the document and nothing
else, so an addition here never breaks that build. Mirroring the vocabulary a
second time *inside* this tree would be one more table to keep in step, and the
one table no producer compiles against. The derives are the mirror: anything that
can write JSON in these shapes can write a document, and the `Image` skip is the
one difference a document states.

## Why not a `Surface`

A `Surface` is a source: it has planes, a natural extent, and a render. A scene
document has a plane and a list and no render at all, so the trait's one required
method would have to answer `Unsupported` forever, and three other methods would
exist only to say "not me". `Scene` is a plane and a list; the launch is what
connects one to the view.

The reconnection point is a push seam: installing a list built outside the
process, with its own plane and viewport, into a view that has no source either.
This record is that state, reached through a file instead of a socket.

## Consequences

- The vocabulary crosses the wire whole, and a glyph run carries the font's own
  bytes, so a document with a thousand labels repeats a font file a thousand
  times and deserialization makes a fresh `Arc` for each repetition (`rc` saves
  the copy on the writer's side only). A shared font table — an id, or one
  document-level font list — is the shape of the answer, and it is the work a
  producer that emits text will hit first. A run whose bytes or face index name
  no face is dropped by the pass with a `warn!` rather than taking the process
  down, because vello parses the same bytes with an `unwrap`.
- The fit is uniform for a document that states no viewport, because
  `region::Frame` answers one factor on both axes. A producer whose two axes
  scale differently states the box its window shows, and `docs/adr/0009` records
  that decision and the per-axis scale it rests on.
- `list_drawn` follows a pass refusal down to nothing rather than to a raster, so
  a heavy scene on a deep zoom shows the background for the refused frames. A
  policy for a list source with no raster of its own — keep the last encode and
  stretch it, or state that there is no fallback — is its own decision; this
  record only refuses to invent a picture.
- Text needs no new primitive here: `Glyphs` and `GlyphRun` cross the wire
  already, and `GlyphRun.font` carries the font's bytes. What is still missing
  is the measurement a producer needs before it lays a string out, which is
  separate work.
- Coordinates are `f64` and the vocabulary does not range-check them, so a
  primitive far outside what a frame can map paints nothing rather than
  answering an error. A producer that scales its units by a stray factor sees a
  blank window; the README says so rather than the reader guessing.
- A producer may write a large list, and nothing in this route bounds its size:
  the parse is on the launch thread and the list is held whole. A bound belongs
  with the push seam, where `MAX_PUSH_BYTES` already caps one payload; a file the
  user names is not a flood.
- `checks.gpu` carries the end-to-end proof: a two-rect document written as text,
  opened through `Launch::scene`, painted, and read back — the frame that waits
  for the pass shows the clear color, and the frame after it shows the rects at
  the fit the extent asks for.

## Testing

The reader asserts on the document, not on the tree's own output: the fixture is
hand-written JSON, so a renamed variant, a moved field, another version, an
unknown field, an image leaf and a zero extent each fail one test by name. The
paint asserts on pixels, because the one thing a file route has to prove is that
a list written by another program reaches the frame.
