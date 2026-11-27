# The display list's budget, measured

Status: measurement, 2026-11-27. The constants it sets live in
`crates/vv-gpu/src/display.rs`.

## The question

`Work` counts the lines a frame asks `vello` to flatten, and two constants rest
on that count: `LINE_BUDGET` decides whether a band is refused, and
`PRIMITIVE_BUDGET` decides whether one path is clipped to the band. `vello`
sizes its own buffers with Wang's formula, this pass counts with the same
formula, and a frame either of them starves paints *nothing* and reports
nothing, so the estimate has to be an upper bound. How far does it run past
what the flattener actually writes?

## The method

The flatten shader counts its own output into `BumpAllocators::lines`, and
`Renderer::render_to_texture_async` hands that number back when vello is built
with its `debug_layers` feature. A scratch crate beside this tree
(`~/agent/vello-gpu-bench`, `cargo run --release --bin overcount -- [shapes|deep]`
in this repo's devshell, where the driver is lavapipe) renders one path of
curves into an 800 by 600 target at a set of scales and prints the count beside
this pass's estimate for the same geometry, mirroring `Work` and `count_path`
line for line. Its output is beside it in `report-2026-11-27-overcount.txt`.

The crate stays out of the tree: it is a measurement of an engine, and it is
not a dependency of the build.

## What it found

**The floor was too low.** With `LINES_PER_CURVE` at the five that
`vello_encoding` floors its own count with, a wave of eight units at a quarter
of the fit scale answered 100,001 lines where the renderer wrote 119,003: 0.84
of the truth, an under-report, which a budget must never make. The real count
stops falling as a curve shrinks, because a curve is a few Euler-spiral pieces
and each piece draws at least one line, and six lines a curve is where this
sweep bottoms out. The floor is eight.

**The over-count grows with the zoom.** A ratio above one is an estimate that
is safe and loose, and the loosesness is not a constant:

| shape (20,000 cubics unless noted) | 0.125× | 0.25× | 1× | 8× | 64× | 512× | 1024× |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wave, 60 units | 1.16 | 1.63 | 2.16 | 3.02 | 3.63 | – | – |
| wave, 8 units | 3.90 | 1.34 | 1.64 | 1.74 | 1.97 | – | – |
| wave, 2 units | 7.44 | 3.86 | 2.60 | 1.49 | 1.62 | 1.87 | 1.84 |
| wave, 0.05 units | 7.44 | 7.44 | 7.27 | 3.77 | 2.88 | 2.99 | 3.12 |
| quarter circle, 4,000 | 1.29 | 1.59 | 1.70 | 1.65 | 1.72 | 1.75 | 1.77 |
| nearly straight, page long, 2,000 | 7.12 | 7.68 | 8.30 | 8.07 | 9.91 | 10.96 | 10.75 |
| wave, 60 units, deep zoom, 200 | – | – | – | – | 3.65 | 4.12 | 4.16 |

The deep-zoom row goes on to 4.53 at 32,768×. The smallest over-count anywhere
in the sweep is 1.16, at an eighth of the fit scale, and the largest is 11.4 for
a nearly straight curve whose floor dominates.

**Why.** `flatten.wgsl` fits each cubic to Euler spirals and then subdivides one
piece at most a hundred times (`clamp(ceil(…), 1.0, 100.0)`, "this may give
slightly incorrect rendering but avoids hangs"), and it approximates a curve by
smooth pieces rather than by a polyline, so its count sits nearer the optimal
integral than Wang's worst-case bound. Wang's keeps growing as the square root
of the scale where the flattener saturates, which is why the ratio climbs with
the zoom.

## What the numbers set

- `LINES_PER_CURVE` is 8, not `vello_encoding`'s 5, so the count never answers
  less than the renderer writes: the floor of a budget over-reports or it is not
  a floor.
- `LINE_BUDGET` is three quarters of the real `1 << 21`. With the smallest
  measured over-count of 1.16, an estimate that large is at most 65% of the
  buffer the renderer holds, so a third of it stays clear, and a document of
  straight segments, which the count measures exactly, sits at 75%.
- `PRIMITIVE_BUDGET` is `LINE_BUDGET` itself: a primitive past a whole band
  alone is one no band holding it can paint, so clipping it is the only way that
  frame paints at all, and a primitive under the budget costs no 20 to 45 ms walk
  of the frame thread. A page of twenty thousand curves at eight times its fit
  is 1.47 M estimated lines against the 1.57 M budget, and vello paints it whole
  at 490,000 real ones.

## Where the estimate stays loose

A deep zoom of a wavy path over-reports about 4.5 times, and a shallow one of
nearly straight curves about ten times, so the pass refuses and clips those
frames earlier than it must. Following the flattener there would need the
Euler-spiral fit error of each curve, which this walk does not compute; the
refusal, the clip and the fallback are what keep the picture on screen while the
count is loose, and the GPU tests pin the refusal against the frame the renderer
itself paints nothing of.

## After a vello upgrade

Re-run the sweep before trusting these numbers: check whether
`render_to_texture_async` still answers the allocator, whether
`vello_encoding::BufferSizes::new` still fixes `lines` at `1 << 21`, whether
`flatten.wgsl` still caps a piece at a hundred lines, and whether the flattener
still fits Euler spirals.
