# A treemap source that takes the shape of the window

Status: accepted. Implemented on the tree that carries this record.

## Context

The viewer shows pictures and documents, and every one of them has a size of its
own. The fit then frames that size inside the window and the letterbox bars take
the rest, which is right for a picture: a 4:3 photo in a 16:9 window should not
be cropped or stretched.

A treemap is the other kind of surface. Its whole point is that the rectangles
reach every edge of what it fills, so a bar of background around it is a bar of
wasted space. A treemap also carries no content of its own here: there is no
file format to parse yet, and the first cut draws a tree built in code.

Two things therefore had no place in the tree:

- a `Surface` whose plane shape is not a fact of a file but of the window, and
- a route for a path that a decoder never reads.

## Decision

Add `crates/vv-source/src/treemap.rs` and one method to the `Surface` trait.

The trait gains `frame_window(&self, win: Size) -> bool`: the view hands the
source its window size, and the source answers whether the plane it takes
changed. `ViewState::frame_window` (`crates/vv-view/src/view.rs`) calls it once a
frame in the image view, reads `bounds` again when the answer is `true`, takes
the new plane and drops the region patch of the old one. The default answers
`false` and touches nothing, so a picture keeps the one size it reports and the
view never reads a header twice for a resize.

`TreemapSurface` answers a plane whose shape is the window's, with the long side
clamped into the base raster range a vector file takes (1024..=8192 px, by
`svg::base_side`). The declared size, the base raster and the plane are then the
same numbers, so a region arrives in frame units and the fit of the plane covers
the window exactly.

The layout is a pure function of the tree and the frame
(`treemap::layout`), and the paint builds an SVG document from it
(`treemap::svg`) which `svg::from_str` parses. A label therefore rides the host
fonts, a region paint costs only the pixels the window shows, and a settled zoom
repaints sharply, all through the path a `.svg` file already takes.

A plane that changes shape needs one thing no other source needs: the base
picture of the whole plane has to follow it. The image view draws that texture
wherever the region patch cannot reach, which is exactly the frames of a zoom
that outran the paint, so a texture of the old frame would draw the old layout
beside the new one, and the picture would read as broken blocks. `ViewState`
therefore marks the base stale when the plane moves (`plane_stale`), paints it
again once the window holds still (`ViewState::refresh_plane`), and takes the
landed texture through `CropMsg::Base`. A frame a loader started before the
re-layout never replaces that texture, because a worker reads the plane when it
starts (`loader_plane`).

A `.treemap` path routes on its extension alone, before every sniff. The
extension is the one honest signal for a source with no content of its own, the
same reasoning that lets a converter rule claim a format with no magic.

## Consequences

A resize frame costs one call and one `Size` comparison: the layout is not
rebuilt and no SVG is parsed. The picture on screen stretches until the view
holds still, and the debounced region paint (`region::QUIET`) then pays for the
new layout once. A layout per resize frame would parse a few thousand rectangles
per frame, which reads as stutter.

The grid never calls the method, so a cell of a `.treemap` file shows the
default frame, which is a 1280x800 window. That is also the size the window
opens at for a single-file launch, so the first frame finds the plane it already
has.

The base picture costs one whole-plane render per settled window (tens of
milliseconds on a worker). A plan that repainted it per resize frame would spawn
a worker per frame of a drag, so the paint waits for the window to stop moving,
and the patch covers the window until it lands.

The layout keeps the frame as an argument, so a zoomable treemap lands later by
laying out the sub-frame under the viewport with the same call, and no change to
`TreemapSurface`.

The trait grows one method that every other backend ignores. It costs one
defaulted arm and no line in `ImageSurface`, `VectorSurface`, `PdfSurface`,
`MemorySurface` or `BookSurface`.

## Alternatives

**Fit the plane by covering (`ZoomMode::Fill`) instead.** That needs no trait
method at all, and it also shows no background. The crop then takes real
rectangles off the map at two edges, and panning reveals them one gesture later,
so a map would hide part of its own data at the size the user opens it.

**Letterbox a fixed plane.** Simplest, and it is what every other source does.
It is also exactly the space the map means to use.

**Plumb the window size into `open_surface`.** The open happens before the window
exists for a single-file launch (the launch renders its plane to size the
window), so the size would have to arrive through a second entry point that every
caller of `open_surface` then has to know about. A method on the trait keeps one
entrance and carries the fact at the moment it becomes true.
