# versatile-viewer

Fast image (later: video) viewer. nsxiv meets mpv. JPEG XL first-class.

- License: AGPL-3.0-or-later
- Renderer: raylib (Rust bindings). wgpu probably not needed.
- JXL decode: `jxl-oxide`, pure Rust, no libjxl C dependency.
  jxl-oxide supports progressive decoding (partial bytes -> blurry preview,
  full bytes -> full image); planned for a later milestone.
- Common formats: `image` crate (PNG, JPEG).
- Build: nix flake (crane).
  - `nix build` — package
  - `nix build .#checks.default` — tests
  - `nix fmt` — format the tree (treefmt: rustfmt, taplo, nixfmt); rustfmt
    runs with extended options (import grouping, 100-col width) via
    `settings.formatter.rustfmt.options` with `lib.mkAfter`
  - `nix flake check` — run checks
  - `nix develop` — dev shell (LD_LIBRARY_PATH + rpath RUSTFLAGS set;
    treefmt wrapper included)
- Run: `nix run . -- <image-path>`; ESC/q quits. Binary name: `vv`.

## Milestones (step by step)

1. [x] Single image window: JXL + PNG + JPEG, ESC/q.
2. [x] Fit to window (aspect preserved, centered, no upscale, refits on resize).
3. [x] Zoom modes: `W` fit-down (default, no upscale), `Shift+W` fit all
   sides (upscales until first border touch), `e` fit width, `Shift+E`
   fit height.
4. [x] Panning: `h/j/k/l` + arrow keys, unrestricted.
5. [x] Free zoom: `+`/`-` (also numpad, layout-independent via typed
   character), 25% steps; zoom animates smoothly (exponential ease,
   ~150 ms) toward the target scale.
6. Zoom/pan polish (wheel zoom, cursor-anchored zoom) still open.
7. jxl-oxide progressive decoding (stream bytes, render preview early).
8. Directory browsing (nsxiv-style left/right).
9. Video playback (mpv inspiration; backend TBD).
