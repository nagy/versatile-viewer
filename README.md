# versatile-viewer (`vv`)

Fast image viewer, after nsxiv and mpv. JPEG XL is a first-class format. Video
comes later.

JPEG XL decodes through [jxl-oxide], which is pure Rust and needs no libjxl C
library. The decoder streams: a blurry preview at full size appears long
before all bytes arrive. The stream reads from a memory map, so a large file
needs no extra copy. Common formats (PNG, JPEG, WebP, GIF, BMP, TIFF, TGA)
decode through the [image] crate.

SVG and SVGZ rasterize through [resvg], which is pure Rust and needs no cairo.
The viewer sniffs the file head, parses the document with usvg, and paints it
with tiny-skia. A vector file has no pixel size of its own, so the long side
bakes at the declared size clamped into 1024..=8192 px. An icon then zooms
without turning to mush, and a huge viewBox lands on the texture limit. Text
renders with the host fonts, an embedded PNG image renders, and SVGZ
decompresses under a size cap.

The grid paints a vector cell straight at 512 px, so a cell never paints a
full-size pixmap and then shrinks it. The image view keeps the parsed tree and
repaints the region its window shows, at the scale it shows it. A deep zoom
then never stretches the base raster, because the paint area follows the
window and not the document. One paint runs at a time, and a repaint runs once
the view holds still. A cold vector cell measures its shape from a parse of
the size alone, so the grid and the image fit never wait for the decode.

PDF rasterizes through [hayro], also pure Rust with no C dependency. A
document shows as a book. The cover page sits alone in the first spread, and
the pages then pair up. Every spread stacks in one tall figure, so the whole
document is visible at once.

A page counts PostScript points. A region render rasterizes only the rect the
window shows. The image view repaints the visible region after a zoom rests,
so a deep zoom of a page never stretches a raster. A coarse preview lands
first, because rasterizing a page takes a moment. The grid thumbnail stays the
cover page, so a cell never holds a second page.

A Typst source compiles through [typst] to PDF bytes. It then renders through
that same path, so it shows as the same book and zooms sharply for the same
reasons. The compile runs with the directory of the file as project root, so
`#include`, `#image` and `#read` resolve like the Typst CLI. The fonts ship
inside the binary.

A Mermaid source renders through [merman] to SVG text. The answer then parses
through the same vector path, so it shows as one vector picture and zooms
sharply for the same reasons. The render turns HTML labels into SVG text with
the resvg-safe pipeline of merman. It turns them off through an init
directive, because the HTML measure under-reports a wide label. Labels measure
with the host font files, so a label box fits its text.

Thumbnails render in a directory grid that fills the window. They decode in
parallel on a background pool and sort in natural order (`img2` before
`img10`). Every downscale runs through [fast_image_resize] (SIMD, Lanczos3).
That covers the grid thumbnail, the texture cap, the progressive preview and
the blurred background. The grid thumbnails cache in
`$XDG_CACHE_HOME/vv/thumbnails/`, so a second launch of a directory fills the
grid from the cache without a decode.

An entry holds a lossy JPEG XL codestream (quality 95, exact alpha), so 512 px
costs tens of kilobytes instead of a megabyte of raw RGBA8. The entry header
also carries a [ThumbHash] of the thumb (at most 32 bytes). A warm launch then
paints every cell from a header read of about 200 bytes, before any decode
starts. The cached thumbnail is 512 px and the grid thumb is the same size, so
a warm cell matches a fresh one. The cache keeps its entries after the viewer
exits, so the next launch starts warm. A stale entry re-decodes, and a deleted
image leaves its thumbnail to the user.

The `cache` cargo feature (on by default) owns the cache.
`--no-default-features` decodes every file on every launch. The `svg` cargo
feature (on by default) owns the SVG path.
`--no-default-features --features cache` leaves it out. The `mermaid` cargo
feature (on by default) owns the Mermaid path and pulls the `svg` feature in,
because that backend paints the diagram.
`--no-default-features --features cache,svg` leaves it out.

## Architecture

Every showable source is a `Surface` (`src/surface.rs`). An image file is a
one-plane surface, and a raster file and a vector file differ only in the
backend behind the trait. A surface renders a `View` (a plane index, an
optional region and a scale) into an RGBA buffer, and it reports the natural
size of each plane.

A grid cell holds an `EntrySource`, which is a file to open lazily. A
multi-page source wraps in `src/book.rs`, which stacks every spread in one
plane. The image view then shows a whole document with no change to the loader
or the grid. The grid, the streaming loader and the image view speak surfaces,
plane indices and regions, and never a file format.

`src/sign.rs` decides the decoder from the file content, and not from the file
name. A PDF routes on its `%PDF-` magic to the hayro backend in `src/pdf.rs`.
A Typst source routes by extension to `src/typst.rs`, which compiles it to PDF
bytes and hands them to that same backend.

A Mermaid source routes by extension to `src/mermaid.rs`, which renders it to
an SVG string and hands the parsed tree to the vector backend in
`src/surface.rs`.

The grid holds one cell per file, and a document cell shows its cover page.

The image view shows a document as a book, and a book opens on its first
spread. The fit frames that spread, so the cover page shows whole, and the
view scrolls to the top of the document.

A Typst source compiles to the same pages, so it opens as the same book. See
`docs/adr/0001-*.md` and `src/book.rs`.

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
nix run . -- --wid 0x3a00007 <image-path-or-directory>   # inside a host window
nix run . -- --version   # prints `vv 0.1.0`
nix run . -- --help      # every flag and the path
```

The viewer needs the X11 libraries and a Vulkan driver. The Nix build bakes
the loader and the X11 libraries in through rpath, and the driver comes from
the host (`/run/opengl-driver` on NixOS). Outside Nix, `cargo build --release`
and a `libvulkan.so.1` with the X11 libraries are enough, because every
decoder and the renderer are pure Rust. No C toolchain is necessary.

## MCP

`vv --mcp <path>` also serves the [Model Context Protocol] on stdio, so an
agent can drive the viewer. The window stays interactive.

[Model Context Protocol]: https://modelcontextprotocol.io/specification/2026-07-28

```json
{ "command": "vv", "args": ["--mcp", "/path/to/images"] }
```

The server pins spec revision `2026-07-28`, which is stateless. There is no
`initialize` handshake and no session. Each request carries its protocol
version and client capabilities in the `io.modelcontextprotocol/*` `_meta`
fields, and `server/discover` replaces the handshake.

At present the server is a husk. It advertises the tools capability with an
empty tool list, and offers no resources or prompts. Every diagnostic goes to
stderr, because stdout carries protocol frames only.

## Embedding

`vv --wid <id> <path>` shows the picture inside a window of a host
application, the way `mpv --wid=<XID>` shows a video. The id reads as decimal
or as `0x` hexadecimal, and it may come before or after the path. It composes
with `--mcp`.

```sh
vv --wid 0x3a00007 ~/pictures          # a directory: opens its first entry
vv --wid 0x3a00007 --mcp ~/pictures    # and serves MCP at the same time
```

The viewer creates its child window with the id as the parent, so it never
becomes a top-level window. No window manager frames it, and the host decides
where it sits. An embedded pane shows the image view only, with no grid and no
window title. `ESC` and `Enter` stay in the picture, a directory launch opens
its first entry, `n` /`p` still step through it, and `q` exits. A failed open
exits with the decode error, because there is no grid to fall back to.

The host owes the pane two things.

The host sizes the pane. Create a container window as a child of your own
window, size the container, and hand its id over. The pane reads the container
geometry about twenty times a second and resizes itself, so never resize the
child window yourself.

The host focuses the pane. Call `XSetInputFocus` on the child of the container
(the pane's own window) when the user clicks the pane or selects the buffer
that holds it. Keys follow focus, and the pane never focuses itself. Without
that call it shows a picture and ignores every key. A lost focus already drops
every held key, the pointer grab and every drag.

The pane claims no XEmbed support. It sets no `_XEMBED_INFO` and sends no
XEMBED message, so a host that speaks the protocol must focus the pane itself.

### As a library

A host that already owns a `wgpu` device and a render target builds a `Viewer`
instead of a process, which is also the Wayland-shaped path:

```rust
let launch = viewer::Launch::open(path, false)?;         // lists, decodes, sizes
let window = host.create_window(launch.window_size());   // the host's own window
let mut viewer = Viewer::new(device, queue, format, max_texture_side, launch, size);
viewer.handle_event(viewer::Event::Key { .. }, &mut host);
viewer.render(&mut host, &target_view, size, dt)?;       // paints the host's texture
```

The viewer owns the pipeline, the textures, the view state, the grid and the
loaders. The host owns the window, the surface, the event loop, the clock and
the present. Four window actions travel back through `viewer::Host`
(`set_title`, `grab_pointer`, `release_pointer`, `place_pointer`), so the
viewer never holds a window. `viewer::Event` is the input vocabulary. A host
maps its own toolkit into it, and the held keys, the capture rule and the
focus drop stay inside.

`cargo run --example viewer_host -- <path>` is a working host for the library
path. It owns the window, the surface and the swapchain, and it gives the
viewer a texture. It then copies that texture into the swapchain image each
frame, which is the shape a Wayland host takes.

`cargo run --example embed_host -- <path>` is a working host to try the
`--wid` path against. It creates a frame and a container under a bar and
prints both window ids. It resizes the container with the frame, and takes or
gives the keyboard on a click on the bar.
`cargo run --example embed_host -- --child-size 640x480 <path>` scripts a
resize. `xwininfo -tree -name "vv embed host"` shows the three levels (frame,
container, pane).

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

A motion starts only at a key press this window saw go down. A window manager
that binds `l` for a workspace switch still holds that key when `vv` arrives,
and X replays it as a synthetic press. `vv` ignores that replay and the
auto-repeat after it, so a pan never starts from a key another window owns.

`v` in the image view selects a region. Hold `v`, drag the rectangle, and
release. The rest of the picture dims by the same amount as the grid cells
outside a visual selection, and a bright border marks the region. The region
stays while the view pans, zooms and turns, because it belongs to the picture
and not to the screen.

`Shift+v` keeps both sides equal. Hold `v` inside a kept region and drag to
move it: the rect keeps its size and its edges stop at the picture border.

A `v` press and release without travel drops the region, unless the press
landed inside it, which commits nothing. `Esc` drops the region too, and stays
in the image view until the second press.

A PDF or a Typst source opens as a book. The cover page sits alone, the pages
then pair up, and every spread stacks in one figure. A zoom past a settled
ease repaints the region the window shows, at the scale it shows it, so a page
stays sharp at any zoom.

`Shift+J` and `Shift+K` step one spread down or up on a book. The view eases
the next spread to the top of the window and stops there. One press moves one
spread, and the step ends on the first and the last spread. `Page-down` and
`Page-up` do the same step.

## Environment variables

|variable          |effect                                                                 |
|------------------|-----------------------------------------------------------------------|
|`VV_DEBUG=1`      |trace grid open/return events, key events and the drag travel to stderr|
|`VV_TIMING=1`     |print the GPU stack at startup and every slow frame to stderr          |
|`VV_SLOW_STREAM=1`|dribble 4 KB chunks with 1 s pauses until the first preview renders    |
|`VV_BLUR_BG=1`    |image-view background is a tiny blurred copy of the viewed image       |
|`VV_BG_DIM=0..1`  |brightness of the blurred background (default 0.6)                     |
|`VV_BLUR_PX=<n>`  |long side of the blur texture, 8..=1024 (default 128; fewer = blurrier)|
|`VV_CACHE=0`      |do not read or write the thumbnail cache under `$XDG_CACHE_HOME`      |
|`VV_PRESENT=`     |swapchain present mode: `fifo` (default), `relaxed`, `mailbox`, `immediate`|
|`VV_PAN_DRAG_SPEED=`|picture pixels per unit of drag travel (default 1.0, range 0.05..=16)|

## Development

```sh
nix build                     # package
nix build .#checks.default    # tests
nix fmt                       # treefmt: rustfmt, taplo, nixfmt
nix develop                   # shell with LD_LIBRARY_PATH + rpath RUSTFLAGS
```

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
