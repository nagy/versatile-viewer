# versatile-viewer (`vv`)

Fast image viewer (video planned). nsxiv meets mpv, with JPEG XL as a
first-class citizen.

JPEG XL decodes through [jxl-oxide], pure Rust — no libjxl C dependency —
with progressive streaming: a blurry full-size preview appears long before
all bytes arrive. The stream reads from a memory map, so a large file costs
no extra copy. Common formats (PNG, JPEG, WebP, GIF, BMP, TIFF, TGA)
decode through the [image] crate. SVG and SVGZ rasterize through [resvg]
(pure Rust, no cairo): the file head is sniffed, the document parses with
usvg, and tiny-skia paints it. A vector file has no pixels of its own, so
the long side bakes at the declared size clamped into 1024..=8192 px — an
icon zooms without turning to mush, and a huge viewBox lands on the texture
limit. Text renders with the host fonts, embedded PNG images render, and
SVGZ decompresses under a size cap. The grid paints a vector cell straight
at 512 px, so a cell never paints a full-size pixmap only to shrink it. The
image view keeps the parsed tree and repaints the region its window shows,
at the scale it shows it, so a deep zoom never stretches the base raster:
the paint area follows the window, not the document. One paint in flight,
and a repaint once the view holds still. A cold vector cell measures its
shape from a size-only parse, so the grid and the image fit never wait for
the decode.
PDF rasterizes through [hayro], also pure Rust, with no C dependency. A
document shows as a book: the cover page alone in the first spread, then the
pages in pairs, every spread stacked in one tall figure so the whole document
is visible at once. The page counts PostScript points, a region render
rasterizes only the rect the window shows, and the image view repaints the
visible region over a rested zoom, so a deep zoom of a page never stretches a
raster. A coarse preview lands first, because rasterizing a page takes a
moment. The grid thumbnail stays the cover page: no cell ever holds a second
page.
A Typst source compiles through [typst] to PDF bytes and renders through that
same path, so it shows as that same book and zooms sharply for the same
reasons. The compile runs with the file's own directory as the project root,
so `#include`, `#image` and `#read` resolve like the Typst CLI, and the fonts
ship inside the binary.
A Mermaid source renders through [merman] to SVG text and parses through that
same vector path, so it shows as one vector picture and zooms sharply for the
same reasons. The render turns HTML labels into SVG text with merman's
resvg-safe pipeline and turns them off through an init directive, because the
HTML measure under-reports a wide label. Labels measure with the host font
files, so a label box fits its text.
Thumbnails render in a directory grid
that fills the window, decoded in parallel on a background pool, and sorted
in natural order (`img2` before `img10`). Every downscale, including the
grid thumbnail, the texture cap, the progressive preview and the blurred
background, runs through [fast_image_resize] (SIMD, Lanczos3). The grid
thumbnails cache in `$XDG_CACHE_HOME/vv/thumbnails/`, so a second launch of a
directory fills the grid from the cache without a decode. An entry holds a
lossy JPEG XL codestream (quality 95, exact alpha), so 512 px costs tens of
kilobytes instead of a megabyte of raw RGBA8. The entry header also carries a
[ThumbHash] of the thumb (at most 32 bytes), so a warm launch paints every
cell from a header read of about 200 bytes before any decode starts. The
cached thumbnail is 512 px and the grid thumb is the same size, so a warm
cell matches a fresh one. The cache keeps its entries after the viewer exits,
so the next launch starts warm. A stale entry re-decodes, and a deleted image
leaves its thumbnail to the user. The `cache` cargo feature (on by default)
owns the cache; `--no-default-features` decodes every file on every launch.
The `svg` cargo feature (on by default) owns the SVG path;
`--no-default-features --features cache` leaves it out.
The `mermaid` cargo feature (on by default) owns the Mermaid path and pulls the
`svg` feature in, because that backend paints the diagram.
`--no-default-features --features cache,svg` leaves it out.

## Architecture

Every showable source is a `Surface` (`src/surface.rs`): an image file is a
one-plane surface, and a raster file and a vector file differ only in the
backend behind the trait. A surface renders a `View` (a plane index, an
optional region and a scale) into an RGBA buffer, and it reports the natural
size of each plane. A grid cell holds an `EntrySource`: a file to open
lazily. A multi-page source wraps in `src/book.rs`, which stacks every spread
in one plane, so the image view shows a whole document with no change to the
loader or the grid. The grid, the streaming loader and the image view speak
surfaces, plane indices and regions, never file formats.
`src/sign.rs` decides the decoder from the file content, not the file name,
and a PDF routes on its `%PDF-` magic to the hayro backend in `src/pdf.rs`.
A Typst source routes by extension to `src/typst.rs`, which compiles it to
PDF bytes and hands them to that same backend.
A Mermaid source routes by extension to `src/mermaid.rs`, which renders it to
an SVG string and hands the parsed tree to the vector backend in
`src/surface.rs`.

The grid holds one cell per file, and a document cell shows its cover page.
The image view shows a document as a book, and a book opens on its first
spread: the fit frames that spread, so the cover page shows whole, and the
view scrolls to the top of the document. A Typst source compiles to the same
pages, so it opens as the same book. See `docs/adr/0001-*.md` and
`src/book.rs`.

[jxl-oxide]: https://crates.io/crates/jxl-oxide
[image]: https://crates.io/crates/image
[resvg]: https://crates.io/crates/resvg
[hayro]: https://crates.io/crates/hayro
[typst]: https://crates.io/crates/typst
[merman]: https://crates.io/crates/merman
[fast_image_resize]: https://crates.io/crates/fast_image_resize
[ThumbHash]: https://evanw.github.io/thumbhash/

## Run

```sh
nix run . -- <image-path-or-directory>   # single image or directory grid
nix run . -- --mcp <image-path-or-directory>   # the same, plus MCP on stdio
```

Requires the usual X11/GL runtime libraries; the Nix build bakes them in
via rpath, so the result is self-contained. Outside Nix:
`cargo build --release` against a system raylib works too.

## MCP

`vv --mcp <path>` also serves the [Model Context Protocol] on stdio, so an
agent can drive the viewer. The window stays interactive.

[Model Context Protocol]: https://modelcontextprotocol.io/specification/2026-07-28

```json
{ "command": "vv", "args": ["--mcp", "/path/to/images"] }
```

The server pins spec revision `2026-07-28`, which is stateless: there is no
`initialize` handshake and no session. Each request carries its protocol
version and client capabilities in the `io.modelcontextprotocol/*` `_meta`
fields, and `server/discover` replaces the handshake.

At present the server is a husk. It advertises the tools capability with an
empty tool list, and no resources or prompts. Every diagnostic goes to stderr,
because stdout carries protocol frames only.

## Keys

|key                  |grid view                   |image view                     |
|---------------------|----------------------------|-------------------------------|
|`q`                  |quit                        |quit                           |
|`Enter` / `Esc`      |open selection / clear visual selection|return to grid / (inert alone) |
|`h j k l` / arrows   |move selection (auto-repeat)|pan (hold: eased speed ramp)   |
|`Shift+h/l`          |move selection, double step |pan and free zoom, double speed|
|`Shift+j` / `Shift+k`|move selection, double step |pan, double speed; in a book: ease to the next / previous spread, one spread per press|
|`Page-down` / `Page-up`|—                          |in a book: ease to the next / previous spread (same step as `Shift+j` / `Shift+k`)|
|`Space` / `Shift+Space`|—                           |next / previous image          |
|`Backspace`          |—                           |previous image                 |
|`n` / `p`            |—                           |next / previous image          |
|`g` / `G`            |first / last entry          |first / last image             |
|`W` / `Shift+W`      |—                           |fit-down / fit-all             |
|`e` / `Shift+E`      |—                           |fit width / fit height         |
|`t`                  |—                           |toggle fit-all / fill          |
|`a`                  |—                           |toggle smooth / pixelated filter|
|`f`                  |open cell under cursor      |fill-zoom at cursor / zoom out / back to grid (hold: pan)|
|`Shift+f` (hold)     |—                           |free zoom by mouse travel: up zooms in, down zooms out (continuous)|
|`f` then `Shift`     |—                           |while `f` is held, Shift switches the pan drag over to the free zoom, anchored at the view center|
|`r` (hold)           |—                           |rotate by mouse travel: down turns clockwise, up turns counter-clockwise; the release snaps a near quarter turn|
|`Shift+r` (hold)     |—                           |rotate around the mouse point instead of the image center|
|`v` / `V` / `Ctrl+v` |visual selection: charwise / linewise / blockwise (dims the other cells)|region selection (hold `v` and drag; hold inside it to move it)|
|`Shift+v` (hold)     |—                           |region selection with equal sides (square)|
|`o`                  |swap visual anchor and cursor|—                             |
|`+` / `-`            |zoom thumbnails             |free zoom (hold: eased ramp)   |
|`Ctrl+j` / `Ctrl+k`  |move selection              |free zoom in / out (hold: eased ramp)|
|mouse wheel          |scroll (when zoomed in)     |free zoom (25% steps, anchored)|
|mouse side buttons   |—                           |next / previous image          |
|left-drag            |blockwise visual selection (`Ctrl+v` style)|pan                  |
|click / click        |select / open               |—                              |

`Esc` never quits the program.

`v` in the image view selects a region: hold `v`, drag the rectangle, and
release. The rest of the picture dims by the same amount as the grid cells
outside a visual selection, and a bright border marks the region. The region
stays while the view pans, zooms and turns, because it belongs to the picture
and not to the screen. `Shift+v` keeps both sides equal. Hold `v` inside a kept
region and drag to move it: the rect keeps its size and its edges stop at the
picture border. A `v` press and release without travel drops the region,
unless the press landed inside it, which commits nothing. `Esc` drops the
region too, and stays in the image view until the second press.

A PDF or a Typst source opens as a book: the cover page alone, then the
pages in pairs, every spread stacked in one figure. A zoom past a settled
ease repaints the region the window shows, at the scale it shows it, so a
page stays sharp at any zoom.
`Shift+J` and `Shift+K` step one spread down or up on a book: the view eases
the next spread to the top of the window and stops there. One press moves one
spread, and the step ends on the first and the last spread. `Page-down` and
`Page-up` do the same step.

## Environment variables

|variable          |effect                                                                 |
|------------------|-----------------------------------------------------------------------|
|`VV_DEBUG=1`      |trace grid open/return events and raw key events to stderr             |
|`VV_TIMING=1`     |trace every texture upload (size, duration) and slow frames to stderr  |
|`VV_SLOW_STREAM=1`|dribble 4 KB chunks with 1 s pauses until the first preview renders    |
|`VV_BLUR_BG=1`    |image-view background is a tiny blurred copy of the viewed image       |
|`VV_BG_DIM=0..1`  |brightness of the blurred background (default 0.6)                     |
|`VV_BLUR_PX=<n>`  |long side of the blur texture, 8..=1024 (default 128; fewer = blurrier)|
|`VV_CACHE=0`      |do not read or write the thumbnail cache under `$XDG_CACHE_HOME`      |

## Development

```sh
nix build                     # package
nix build .#checks.default    # tests
nix fmt                       # treefmt: rustfmt, taplo, nixfmt
nix develop                   # shell with LD_LIBRARY_PATH + rpath RUSTFLAGS
```

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE).
