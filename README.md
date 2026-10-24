# versatile-viewer (`vv`)

Fast image viewer (video planned). nsxiv meets mpv, with JPEG XL as a
first-class citizen.

JPEG XL decodes through [jxl-oxide], pure Rust — no libjxl C dependency —
with progressive streaming: a blurry full-size preview appears long before
all bytes arrive. The stream reads from a memory map, so a large file costs
no extra copy. Common formats (PNG, JPEG, WebP, GIF, BMP, TIFF, TGA)
decode through the [image] crate. Thumbnails render in a directory grid
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

[jxl-oxide]: https://crates.io/crates/jxl-oxide
[image]: https://crates.io/crates/image
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
|`Shift+h/j/k/l`      |move selection, double step |pan and free zoom, double speed|
|`Space` / `Backspace`|—                           |next / previous image          |
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
|`v` / `V` / `Ctrl+v` |visual selection: charwise / linewise / blockwise (dims the other cells)|—                       |
|`o`                  |swap visual anchor and cursor|—                             |
|`+` / `-`            |zoom thumbnails             |free zoom (hold: eased ramp)   |
|`Ctrl+j` / `Ctrl+k`  |move selection              |free zoom in / out (hold: eased ramp)|
|mouse wheel          |scroll (when zoomed in)     |free zoom (25% steps, anchored)|
|mouse side buttons   |—                           |next / previous image          |
|left-drag            |blockwise visual selection (`Ctrl+v` style)|pan                  |
|click / click        |select / open               |—                              |

`Esc` never quits the program.

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
