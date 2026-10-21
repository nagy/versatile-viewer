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
  `VV_TIMING=1` prints upload cost and slow frames.
- Public error types derive `derive_more::{Display, Error, From}` (feature `from` on).
  Each module owns its error enum: `src/error.rs` (`DecodeError`, `UploadError`), `src/grid.rs` (`GridError`), `src/loader.rs` (`LoadError`), `src/mcp.rs` (`SpawnError`).
  `main.rs` keeps `anyhow`.
- Wrap raw pixel and window primitives in the `src/dim.rs` newtypes (`ImgW`, `ImgH`, `WinW`, `WinH`, `LongSide`, `PreviewPx`, `Delay`, `Rate`, `RampTime`).
  Public API boundaries take the newtype, not the bare primitive.
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

## Build

- `nix build .#versatile-viewer-doc` makes the rustdoc HTML.
- `nix flake check` gates clippy (`--all-targets -- --deny warnings`), tests, doctests, and treefmt.

## Run

Run `nix run . -- <image-path-or-directory>`.
ESC never quits.

- Keys: see the README table.
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
