# versatile-viewer

Fast image viewer.
Video support comes later.
The design follows nsxiv and mpv.
JPEG XL is a first-class format.
The binary name is `vv`.

## Constraints

- License: AGPL-3.0-or-later.
- Use raylib as the renderer.
- Decode JPEG XL with `jxl-oxide` (pure Rust, no libjxl).
- Decode JPEG XL with `jxl-oxide` (pure Rust, no libjxl).
- Decode common formats with the `image` crate.
- Rasterize SVG and SVGZ with `resvg` (pure Rust, no cairo).
  The module is `src/svg.rs`, with `src/svg_off.rs` as the stub of a build without the `svg` feature.
  A vector file has no pixel size, so it bakes one: `svg::base_side` clamps the declared long side into `BASE_MIN_SIDE` (1024) ..= `MAX_TEXTURE_SIDE` (8192).
  The aspect stays exact, so the base raster and the thumbnail share one shape.
  The base size is the source size: the grid entry, the cache header and the image fit all measure that pair, never the thumbnail.
  `decode_thumb` paints a vector file at the thumbnail size directly, so a cell never paints a big pixmap only to shrink it.
  `decode_announce` sends the header of a vector file before the paint, so the window fits the picture while the raster arrives.
  Sniffing reads content: magic, then a BOM, comments, a prolog and a doctype, then an `<svg` root. A gzip head sniffs its decompressed head.
  `MAX_INPUT` (32 MiB) caps the read and the decompression, so a gzip bomb fails.
  External references never resolve (`resources_dir` stays `None`), so no file makes the viewer read another path.
  The host font database loads once per process (`fontdb`, `OnceLock`), so `<text>` renders with real fonts and a whole directory pays one scan.
  `resvg` covers the static subset only: no script, no animation, and `raster-images` stays off, so an embedded JPEG/WebP/GIF draws nothing while an embedded PNG does.
  A retained `usvg::Tree` (`svg::Vector`) is the hook for a later zoom-tier re-raster.
- Grid extension filter: `src/grid.rs` `is_image_path`.
  Keep the crate features in sync.
- Dispatch decode on sniffed content.
  Do not dispatch decode on the file name.
- Decode and every pixel resize stay off the main thread.
  Decode workers cap the buffer with `cap_texture_side` (`src/lib.rs`), so `upload_rgba` on the main thread uploads a buffer that already fits `MAX_TEXTURE_SIDE`.
  Never add a main-thread resize: a large image then freezes the window for seconds.
  Downscale a borrowed buffer with `downscale_rgba_ref`, not `downscale_rgba` on a clone.
  `BlurBg::attach` and `upload` take `&DecodedImage`, so a caller never clones the blur buffer.
  All pixel resize goes through `resample_rgba` (`src/lib.rs`), the `fast_image_resize` SIMD path with Lanczos3.
  Do not call `image::imageops::resize` for a downscale: it is several times slower.
  Keep the `rayon` feature of `fast_image_resize` off, so a resize never spawns nested workers inside a decode job.
  `VV_TIMING=1` prints upload cost and slow frames.
- Sort grid entries in natural order with `natord`, so `img2` comes before `img10`.
- The streaming JXL loader memory-maps the file (`memmap2`) and feeds slices of the map to jxl-oxide.
  A file rewritten under the map faults with SIGBUS instead of reading short data.
- Tests: `proptest` covers the pure helpers (layout, visual selection, selection step, the resize fit, the ramp).
  Property tests are dev-only and never gate a release build.
- Public error types derive `derive_more::{Display, Error, From}` (feature `from` on).
  Each module owns its error enum: `src/error.rs` (`DecodeError`, `UploadError`), `src/grid.rs` (`GridError`), `src/loader.rs` (`LoadError`), `src/mcp.rs` (`SpawnError`), `src/thumbcache.rs` (`CacheError`).
  `main.rs` keeps `anyhow`.
- Wrap raw pixel and window primitives in the `src/dim.rs` newtypes (`ImgW`, `ImgH`, `WinW`, `WinH`, `LongSide`, `PreviewPx`, `Delay`, `Rate`, `RampTime`, `Rect`).
  Public API boundaries take the newtype, not the bare primitive.
  `Rect` is a `#[repr(C)]` axis-aligned rectangle with raylib `Rectangle` field names.
  It converts with `From`, so a raylib call that takes `impl Into<Rectangle>` takes a `Rect` value directly.
  Do not build a bare `Rectangle` in the draw code.
- Two motion styles, and a key picks one:
  - Auto-repeat steps a discrete move.
    `src/keyrepeat.rs` reimplements the X rate for the grid selection and image next/previous.
  - A velocity ramp eases a continuous move.
    `src/tween.rs` ramps the speed in, holds it, and ramps it out for the image pan and the image free zoom.
    The ramp is linear, and it reads only the held state.
    Never step a continuous motion on the auto-repeat rate.
- Shift + h/j/k/l moves at double speed in both views.
  `SHIFT_SPEED_MULTIPLIER` (`src/lib.rs`) is the one constant.
  The grid takes it as the `step` argument of `handle_input` (`step_selection` clamps the move).
  The image view passes it as the `speed` argument of `Ramps::tick`, which scales the pan and the zoom rate but not the ramp, so the ease keeps its length at any speed.
- Read the held state of a key that no `KeyboardKey` names by learning its keycode.
  `src/zoomkeys.rs` pairs a typed `+`/`-` with the key that went down in the same frame, then the code polls that keycode.
  The X server repeats only the key that went down last, so a time window cannot stand in for the held state.
- The image view zooms with `+`/`-` and with `Ctrl+j`/`Ctrl+k` (in, out), all as held states.
  A held Ctrl turns the pan keys into zoom keys, so it guards the pan flags of `NavKeys`.
  The grid keeps `Ctrl+j`/`Ctrl+k` as plain selection moves.
- The grid caches its thumbnails on disk through `xdg` under `$XDG_CACHE_HOME/vv/thumbnails`.
  The feature `cache` (a default feature) owns `xdg`, `sha2` and `jixel`.
  Without it `src/thumbcache_off.rs` answers `CacheError::Off` from every call, so every launch decodes every file.
  One constant sizes the runtime thumb and the entry: `THUMB_LONG_SIDE` (512).
  A warm cell and a cold one therefore match, because a decode resizes once to the entry size and a cache write copies no pixels.
  An entry holds a fixed header and a JPEG XL codestream, so a 512 px thumb costs tens of kilobytes instead of one megabyte of RGBA8.
  The encode is `jixel`, pure Rust (BSD-3-Clause OR Apache-2.0), lossy at `QUALITY` (95) with `ALPHA_DISTANCE` 0, so only the alpha channel stays exact.
  The decode is the same `jxl-oxide` the viewer uses for a file.
  Both run on a decode worker.
  A `FORMAT` bump rejects every old entry, so a payload change needs no migration.
  The decode worker reads the cache first and decodes only on a miss, then stores the new thumbnail.
  A cache hit carries no full-resolution image, so a keep-set job (an entry that needs the full texture) skips the cache and decodes the file on its own.
  Only the image view keeps full textures: `load_pending` takes a `preload_full` flag, the grid view passes false and the image view passes true.
  A grid-only session therefore uploads thumbnails only, never a 200 MB source texture.
  The header carries the source dimensions, so `GridEntry::width`/`height` measure the image and not the thumb, on a hit and on a miss.
  The header also carries the placeholder of the thumb: a ThumbHash payload of at most `placeholder::MAX_BYTES` (32) bytes, from the `thumbhash` crate (MIT, no dependencies, `src/placeholder.rs`).
  A launch runs a placeholder pass beside the decodes: one cheap rayon job per entry reads the header only (`Cache::load_placeholder`, about 200 bytes, no codestream) and renders the DCT at 32 px on the worker.
  The main thread uploads at most `PLACEHOLDER_UPLOADS_PER_FRAME` (16) placeholder textures per frame, a cell draws its placeholder until the decoded thumb lands, and the thumb upload drops it.
  The image view falls back to the placeholder, and the same header lands the source dimensions, so it fits the picture before the first decode.
  `FORMAT` 3 adds the placeholder payload, so a version 2 entry stays unread and re-decodes.
  A cold launch has no entry, so it keeps the grey cell: the placeholder never invents pixels.
  A write beyond the `WRITE_BUDGET` stops the cache for the rest of the run.
  An entry stays past the exit of the viewer, so the next launch starts warm.
  No sweep runs on exit: a stale entry answers `CacheError::Stale` and re-decodes, and an entry whose source is gone stays unread.
  The only cleanup is `sweep_dead_tmp`, which a store runs to drop the half-written temporary files of killed workers.
  `VV_CACHE=0` turns the cache off.
  A test builds the cache with `Cache::in_dir` over a temporary directory. Never point a test at `Cache::new()`, because then it fills the cache of the user.
  A round-trip test bounds the color drift and holds the alpha channel exact.
- The grid has a Vim-style visual selection, `Visual` in `src/grid.rs`.
  It stores one mode (`Char`, `Line`, `Block`) and one anchor index.
  The selected set derives from the anchor, the cursor, and the current column count, so a resize or zoom reflows it.
  `v`, `V`, and `Ctrl+v` toggle the mode, `o` swaps the anchor and the cursor, ESC clears it.
  A mode switch keeps the anchor, so `v` on row 1 then `V` on row 3 selects rows 1..=3.
  Do not store a selected bit per entry.
  An active visual selection dims every cell outside it with `DIM_TINT`.
  The cursor without a visual selection dims nothing.
  A left-button drag over the grid starts a `Block` visual selection from the press cell, so a drag matches `Ctrl+v`.
  `Drag` holds the press anchor for the length of the hold.
  A hold that releases on its anchor cell stays a plain click.

## Build

- `nix build .#versatile-viewer-doc` makes the rustdoc HTML.
- `nix flake check` gates clippy (`--all-targets -- --deny warnings`), tests, doctests, and treefmt.
## Run

Run `nix run . -- <image-path-or-directory>`.
ESC never quits.

The window clears to `BACKGROUND` (`#062329`, `main.rs`).
Both views share it, so the letterbox bars and the grid gaps match.

- Keys: see the README table.
- `Shift+F` hold turns vertical mouse travel into the free zoom (`ZOOM_DRAG_SPAN` in `main.rs`).
  Up zooms in, down zooms out.
  The hold captures the pointer like the `f` pan drag, so the travel is unbounded and the motion never stops at a screen edge.
  The zoom anchor starts at the grab point and then drifts toward the window center by the wheel rule (`ZOOM_ANCHOR_DRIFT`), so the image point being zoomed into recenters.
  The capture holds the pointer still, so the anchor slides on screen and the image follows it.
  `ZOOM_DRAG_SPEED` scales the rate (now 8x).
  Shift going down or up mid-hold switches the `f` pan drag over to the zoom and back.
  The switch anchors at the window center, which is the image point the window shows at its middle.
  The pointer is hidden, and a pan just pushed its last screen spot away from the middle, so that spot cannot name what the user looks at.
  A center anchor skips the drift, because the drift can only move it away from the middle.
  Plain `f` hold keeps the pan drag.
- The `r` hold turns the image, and the release keeps the angle (`src/rotate.rs`).
  Down travel turns clockwise and up turns counter-clockwise, one full turn per `ROTATE_DRAG_SPAN` of window long side of travel.
  Each frame adds its travel to the current angle (`rotate::apply_turn`), so a new hold continues from the angle the image shows and never resets the turn.
  The hold captures the pointer like the `f` pan drag, and it never pans.
  A newly opened image starts upright, so a turn never carries over to the next image.
  The release snaps the angle with `snap_deg` when it lands within `SNAP_DEGREES` (2) of a quarter turn, and that release is the only place that snaps.
  A drag therefore turns freely and never jumps mid-drag.
  The scale and the pan stay as they are, so a turned image shows its corners past the window edge.
  The draw passes the image center as the `DrawTexturePro` dest position and `Rect::center_origin` as the origin, because raylib turns the quad around the dest position.
  A dest corner plus a half-size origin throws the image into the top-left corner.
  `f` hit-tests a turned image through `unrotate_point`.
  Do not rotate pixels on the main thread: the turn is a draw transform.
- Shift at the `r` press edge turns the image around the mouse point (`ViewState::pivot`) instead of the image center.
  The hold keeps the image point under that pivot on its window spot, so `rotate::pivot` shifts the pan and its target every frame, and again through the release snap (`keep_pivot` in `main.rs`).
  The pivot is a pan shift and not a second draw origin: the draw and the `f` hit test keep one center, the window middle plus the pan.
  A plain `r` press resets the pivot to the center, and a newly opened image resets it too.
  The hold captures and hides the pointer like a plain `r`, so the travel stays unbounded and the pivot sits under the hidden pointer at the grab point.
  A Shift+R hold therefore also doubles the pan and zoom speed, like every other Shift chord.
- Mouse side buttons step next / previous in image view, like `n` / `p` (`read_nav_mouse` in `main.rs`).
- Env vars `VV_DEBUG`, `VV_TIMING`, `VV_SLOW_STREAM`, `VV_BLUR_BG`, `VV_BG_DIM`, `VV_BLUR_PX`, `VV_CACHE`: see the README table.
- Window title uses en-dash separators: `<file> – <dir> – versatile-viewer`.
  The grid title is `(<n> image(s)) – <dir> – versatile-viewer`.
  Canonicalize the paths.
  Abbreviate `$HOME` to `~`.
  The image view title follows the open image.
  The grid keeps the launch title.
  A one-image directory skips the grid at launch.
  ESC returns to the grid.
- X11 WM_CLASS is `vv` (res_name + res_class) — `src/wmclass.rs`.
  Wayland is a no-op.

## MCP

- `vv --mcp <path>` also serves the Model Context Protocol on stdio, from `src/mcp.rs`.
  The `--mcp` flag accepts either argument order.
  The viewer window stays interactive.
- The server pins revision `2026-07-28` and serves that revision only.
  The revision is stateless: no `initialize` handshake, no session, and the client context rides in the per-request `io.modelcontextprotocol/*` `_meta` fields.
  An older client fails fast on `UnsupportedProtocolVersion`.
- The server is a husk: tools capability advertised, empty tool list, no resources, no prompts.
  Deprecated `logging` is not advertised.
- stdout carries protocol frames only.
  Every diagnostic goes to stderr, and `main.rs` redirects the raylib trace log to stderr before the server starts.
  The module never writes to stdout outside the transport.
- The server runs a tokio current-thread runtime on a background thread.
  A start or run failure prints to stderr and never stops the viewer.
- Client config: `{ "command": "vv", "args": ["--mcp", "/path/to/images"] }`.
- Adding a tool means one `TOOLS` entry plus one `call_tool` arm.
  The naming rule (`vv_<action>`), the four hint rules, and the planned tool set live in the `src/mcp.rs` module docstring.
- Tests: unit tests in `src/mcp.rs` drive the server over an in-memory duplex stream.
  `tests/mcp_stdio.rs` drives the real stdin and stdout of a child process.

## Next

- Add video playback (mpv inspiration).
  The backend is TBD.
- Add the first MCP tool (`vv_show_image`), per the tool rules in `src/mcp.rs`.

Module internals live in each `src/*.rs` `//!` docstring.
