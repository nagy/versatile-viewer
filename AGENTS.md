# versatile-viewer

Fast image (later: video) viewer. nsxiv meets mpv. JPEG XL first-class.

- License: AGPL-3.0-or-later
- Renderer: raylib (Rust bindings). wgpu probably not needed.
- JXL decode: `jxl-oxide`, pure Rust, no libjxl C dependency.
  jxl-oxide supports progressive decoding (partial bytes -> blurry preview,
  full bytes -> full image); planned for a later milestone.
- Common formats: `image` crate (PNG, JPEG, WebP; webp via pure-Rust
  image-webp, lossy + lossless decode).
- Build: nix flake (crane).
  - `nix build` — package
  - `nix build .#checks.default` — tests
  - `nix fmt` — format the tree (treefmt: rustfmt, taplo, nixfmt); rustfmt
    runs with extended options (import grouping, 100-col width) via
    `settings.formatter.rustfmt.options` with `lib.mkAfter`
  - `nix flake check` — run checks
  - `nix develop` — dev shell (LD_LIBRARY_PATH + rpath RUSTFLAGS set;
    treefmt wrapper included)
- Run: `nix run . -- <image-path>`; q quits, ESC/Enter toggle grid ↔ image
  view (ESC never quits). Binary name: `vv`.

## Milestones (step by step)

1. [x] Single image window: JXL + PNG + JPEG, q quits.
2. [x] Fit to window (aspect preserved, centered, no upscale, refits on resize).
3. [x] Zoom modes: `W` fit-down (default, no upscale), `Shift+W` fit all
   sides (upscales until first border touch), `e` fit width, `Shift+E`
   fit height.
4. [x] Panning: `h/j/k/l` + arrow keys, unrestricted.
5. [x] Free zoom: `+`/`-` (also numpad, layout-independent via typed
   character), 25% steps; zoom animates smoothly (exponential ease,
   ~150 ms) toward the target scale.
6. [x] Zoom anchored at the window center (free zoom eases while keeping
   the image point under the window center fixed). Wheel zoom still open.
7. [x] jxl-oxide progressive decoding (stream bytes, render preview
   early; image view only — grid thumbs stay full-decode).
   `VV_SLOW_STREAM=1` dribbles 4 KB chunks with 1 s pauses until the
   first preview renders, for eyeballing the refinement.
8. [x] Directory browsing: thumbnail grid fills the window (square
   center-crop thumbs, dynamic layout, white selection border, background
   decodes on the rayon pool — several in parallel); Enter/ESC open,
   Enter/ESC return. Enter reuses the already-decoded full-res grid texture
   (instant, no re-decode); while a decode is in flight the viewer waits
   for it instead of decoding twice; otherwise the streaming loader takes
   over. Decodes are prefetched: grid selection's left/right/up/down
   neighbors, and prev/next while viewing. Opening an image fit-all
   (Shift+W behavior). Space/Backspace or n/p switch prev/next in image
   view (nsxiv-style; arrows and h/j/k/l stay panning); nav keys and grid
   h/j/k/l auto-repeat while held, at the X server's rate (xset r rate).
   g/G jump to the first/last image (grid selection and image view).
9. Video playback (mpv inspiration; backend TBD).
