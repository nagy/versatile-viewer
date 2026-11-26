# A producer states the box its window shows

Status: accepted. Implemented on the tree that carries this record; amends the
fit of ADR 0007 and the document of ADR 0008.

## Context

A display list is written in the producer's own units and carries no scale, so
the viewer decides where those units land in the window. Until this record, one
number decided it: `region::Frame` carried `scale: f32`, `to_window` built one
uniform `kurbo::Affine` from it, and `band_of`, `window_hull`, `place_stale` and
the linear part `too_heavy` reads all agreed with that one number.

That is right for a picture and for a document of a fixed page, and wrong for a
drawing whose axes are independent. A plot with a time axis and a price axis
computes its own scales: how many pixels one unit of time covers has nothing to
do with how many one unit of price covers, and a pan on one axis moves nothing
on the other. A producer that hands such a drawing over can state its extent,
and the fit shows the whole of it, but the picture that arrives is uniformly
scaled — whole and wrong — where the producer's own frame was right.

The geometry a frame needs is already two scales and two translates. The
question this record answers is where those numbers live, and the answer inside
`vv-core` is `Frame`: one mapping, so the list, the fit, the pan, the wheel and
the pointer never disagree about where a surface point sits on screen.

## Decision

**`region::Scale` carries the two factors, and `Frame` carries one `Scale`.**
`to_screen` and `to_image` apply each factor to its own axis, `crop` follows from
the inverse it already used, and `to_window` builds
`Affine::scale_non_uniform`. A uniform view is `Scale::uniform(k)`, so a reader
of a value says which case it means, and `Scale::coarsest` is the one number a
paint of both axes can honestly serve, which is what the raster patch, the
turned texture draw and the grid cell take.

Nothing in the paint path had to learn about the two factors. `SceneView::draw`
already built one `Affine` per frame and every consumer below it — the band, the
window hull, the placement of a reused paint — reads that affine and never the
frame. The one place that reads the linear part directly is `too_heavy`'s
measure, and it takes the larger of the affine's two column norms now, because a
turn mixes both axes into both columns.

**A stated viewport is the fit.** `ViewState::viewport` holds the box of the
plane the window shows, and `viewport_scale(win)` answers the two ratios of the
window to that box, one per axis. A view that states one takes that pair for
every zoom mode, so `W`, `E` and `T` keep the box rather than throwing away the
producer's framing; the free zoom multiplies the pair on both axes, so the
ratio the producer chose survives a wheel step, and the wheel keeps its
pointer anchor because the anchor math runs per axis.

**The placement is derived, the pan is the user's.** The box must fill the
window, so its centre sits at the window's middle, and the plane's centre lands
wherever that puts it: `view_pan(scale)` answers that offset from the plane and
the box. Deriving it every frame is what makes a resize keep the box, and it
keeps `pan` meaning what it meant — the user's offset from the framed view — so
a drag, a glide and a spread step compose with a stated viewport instead of
overwriting it.

**The document states the box, in the plane's units.** `SceneDoc` gains an
optional `viewport { x0, y0, x1, y1 }`, read into a `region::Crop` and installed
as `ViewState::viewport` by the launch. The box is not in pixels and carries no
window size, so a resize re-derives both ratios and no document is re-emitted
for a window that changed shape. It needs no relation to the plane's bounds: a
producer may show the empty space past the end of its data, and the viewer paints
the background there. A box of no width, of no height or of no number is refused
before a window opens, the way a plane of no extent is.

**`VERSION` is 2, and `OLDEST` is 1.** The viewport is an addition, so a producer
that states one writes version 2 and a viewer older than the field refuses the
document by version instead of by an unknown field. A version 1 document is a
version 2 document that states no viewport — every other field means what it
meant — so it still paints, and the reader accepts both.

## Why a box, and not four numbers in pixels

A producer's own viewport is `{sx, tx, sy, ty}` in the pixels of the window it
was emitted for, and that window is not this one: the launch keeps its default
size, and a resize happens after the document was written. Four numbers in
pixels would pin one window size, and every other size would either letterbox
the picture or need a refit rule the producer cannot predict.

The box is the same four numbers once the window is known — `sx` is the window's
width over the box's, `tx` follows from centring it — and it is the form that
survives a resize. It is also the form a viewer already speaks: `region::Crop`
is the region the window shows, and a book's opening box is fit the same way.

## Why the scale lives in `Frame`

The other candidate was leaving `Frame` alone and handing an `Affine` to the
scene pass. It touches less code, and it was rejected for one reason: the view
would then hold two mappings. The pointer, the drag, the `v` region, the `f` hit
test and the wheel anchor all invert `Frame`, and a picture placed by an affine
the view does not know is a picture whose pointer geometry is another picture's.
A viewer whose zoom eases a point that is not under the cursor is worse than one
that does not zoom.

A per-primitive affine was never live. `encode` bakes each primitive's own
`transform` into the surface-unit stream and never sees a window, and the encoded
base is keyed on the list and the band and reused across every pan and zoom; a
placement per primitive is a re-encode per frame, which is the cost the display
list exists to avoid. The transform a primitive carries stays emit-time
geometry.

## Consequences

- A raster is painted at one scale, and `region::Patch` still holds one `f32`.
  That is not a gap: only a list source can be anisotropic, because only a
  document states a viewport, and a document has no raster to patch. The patch
  and the turned texture draw take `Scale::coarsest`, whose docstring names the
  reading — the patch is as sharp as its coarser axis — so nothing silently
  picks an axis.
- `vv_get_status` reports `scale` as `{ "x": …, "y": … }`. One number would name
  neither axis of a view that states a viewport.
- The same answer carries `visible`, the box of the plane the window shows, in
  the plane's own units, named as a document names its viewport (`x0`, `y0`,
  `x1`, `y1`). The two factors say how large a unit is and the box says where
  the units land, so a producer that places its own content reads the framing
  back instead of recomputing a fit it does not own. The box is not clamped into
  the plane, because a window may show the space past an edge of it.
- The fit for a stated viewport is the box for every zoom mode, so `W`, `E` and
  `T` do nothing to such a view, and `f`'s fill aim is skipped for it. That is
  the honest reading of "the box is the fit"; a way to leave the box is a
  separate question and no key answers it yet.
- Adopting a source clears the viewport with the rest of the region state, so a
  picture or a book that replaces a scene frames its own plane. A pushed picture
  over a scene list therefore lands framed by its own fit.
- The two factors ease independently (`Scale::eased`), so a fit that changes
  shape animates both ratios rather than one.
- The free zoom's band (`ZOOM_OUT` to `ZOOM_IN`) is relative to the fit and not
  an absolute factor, because a factor counts output pixels per surface unit and
  a surface unit is now the producer's. A document that measures its plane in
  slot indices fits at a thousandth of a pixel per unit, and an absolute floor
  turned its first wheel step into a twentyfold jump and pinned every step after
  it, so the wheel could never come back to the stated view.

## Testing

`region`'s property tests run the round trip, the crop and the paint bound over
two independent factors, and plain tests pin an anisotropic frame's crop, its
degenerate-factor reading, the anchored pan (a point keeps its window spot under
a second, differently-scaled pair) and a zoom band that is a fraction of the fit.
`vv-gpu` pins the transform of an anisotropic frame and the band it asks for.
`vv-view` pins a stated viewport: the box is the fit under every zoom mode, it
lands corner to corner in the window, a resize re-derives both ratios, and a pan
moves off the placement. `scene_file` reads a box, refuses a flat one, refuses
one whose extents overflow and accepts a version 1 document without one.
`checks.gpu` paints a document whose box is wider than its plane's shape in a
window of another shape and reads the frame back: the box covers the window
corner to corner, where a uniform fit of the same plane would have left the
corners clear.
