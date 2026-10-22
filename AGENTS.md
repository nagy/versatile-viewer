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
- Decode common formats with the `image` crate.
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
  Each module owns its error enum: `src/error.rs` (`DecodeError`, `UploadError`), `src/grid.rs` (`GridError`), `src/loader.rs` (`LoadError`), `src/mcp.rs` (`SpawnError`).
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
- The grid has a Vim-style visual selection, `Visual` in `src/grid.rs`.
  It stores one mode (`Char`, `Line`, `Block`) and one anchor index.
  The selected set derives from the anchor, the cursor, and the current column count, so a resize or zoom reflows it.
  `v`, `V`, and `Ctrl+v` toggle the mode, `o` swaps the anchor and the cursor, ESC clears it.
  A mode switch keeps the anchor, so `v` on row 1 then `V` on row 3 selects rows 1..=3.
  Do not store a selected bit per entry.

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
- Mouse side buttons step next / previous in image view, like `n` / `p` (`read_nav_mouse` in `main.rs`).
- Env vars `VV_DEBUG`, `VV_TIMING`, `VV_SLOW_STREAM`, `VV_BLUR_BG`, `VV_BG_DIM`, `VV_BLUR_PX`: see the README table.
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
