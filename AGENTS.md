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

## Build

- `nix build .#versatile-viewer-doc` makes the rustdoc HTML.
- `nix flake check` gates clippy (`--all-targets -- --deny warnings`), tests, doctests, and treefmt.

## Run

Run `nix run . -- <image-path-or-directory>`.
ESC never quits.

- Keys: see the README table.
- Env vars `VV_DEBUG`, `VV_SLOW_STREAM`, `VV_BLUR_BG`, `VV_BG_DIM`, `VV_BLUR_PX`: see the README table.
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

## Next

- Add video playback (mpv inspiration).
  The backend is TBD.

Module internals live in each `src/*.rs` `//!` docstring.
