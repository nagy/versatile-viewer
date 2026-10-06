# versatile-viewer

Fast image viewer (video later). nsxiv meets mpv. JPEG XL first-class. Binary `vv`.

## Constraints

- License AGPL-3.0-or-later.
- Renderer raylib.
- Decode JPEG XL via `jxl-oxide` (pure Rust, no libjxl).
- Common formats via `image` crate.
- Grid extension filter: `src/grid.rs` `is_image_path`. Keep crate features in sync.
- Decode dispatch on sniffed content, not file name.

## Build

- `nix build .#versatile-viewer-doc` — rustdoc HTML.

## Run

`nix run . -- <image-path-or-directory>`. ESC never quits.

- Keys — README table.
- Env vars `VV_DEBUG`, `VV_SLOW_STREAM`, `VV_BLUR_BG`, `VV_BG_DIM`, `VV_BLUR_PX` — README table.
- Window title en-dash separated: `<file> – <dir> – versatile-viewer`, grid `(<n> image(s)) – <dir> – versatile-viewer`.
  Canonicalize paths, abbreviate `$HOME` to `~`. Image view title follows the open image; grid keeps the launch title.
  One-image directory skips the grid at launch; ESC returns to the grid.
- X11 WM_CLASS `vv` (res_name + res_class) — `src/wmclass.rs`. Wayland no-op.

## Next

- Video playback (mpv inspiration). Backend TBD.

Module internals live in each `src/*.rs` `//!` docstring.
