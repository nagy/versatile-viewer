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
- Map: `nix run . -- --osm [z/lat/lon]` — OpenStreetMap slippy-map mode
  (see milestone 10).

## Milestones (step by step)

1. [x] Single image window: JXL + PNG + JPEG, q quits.
2. [x] Fit to window (aspect preserved, centered, no upscale, refits on resize).
3. [x] Zoom modes: `W` fit-down (default, no upscale), `Shift+W` fit all
   sides (upscales until first border touch), `e` fit width, `Shift+E`
   fit height.
   `t` toggles between fit-all (whole image visible) and fill (window fully
   covered, overflow cropped); distinct for any image/window shape.
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
   +/- zoom the grid thumbs (25% steps; default zoom fills the window
   exactly, zoom-out refits with smaller thumbs, zoom-in overflows and
   scrolls — mouse wheel + selection-follow).
9. Video playback (mpv inspiration; backend TBD).
10. [branchless, lives beside the file viewer until views merge] OSM
    slippy map, phase 1 (tiles + pan + integer zoom; phase 2 = disk
    cache):
    - `map.rs` is self-contained: hand-rolled Web-Mercator math
      (`lon_to_x`/`lat_to_y`/`x_to_lon`/`y_to_lat`, pure + unit-tested —
      no projection crate needed, ~40 lines; osm.el uses the same
      formulas), fractional global-pixel center state (osm.el's wx/wy
      approach: pan = add pixels, zoom = scale by 2^Δz).
    - NOT a `Document`: a map is an infinite canvas with no pages and no
      `page_info`; forcing it into `page_count()/render(page, scale)`
      hurts. It runs its own window + loop from `--osm` in main(); the
      two loops merge later, if ever, via a shared View abstraction.
    - Downloads: `ureq` (blocking, rustls — no tokio in this raylib app),
      3 worker threads over a shared priority queue (center tile first,
      Chebyshev distance; dedupe queued/in-flight; queue capped at 256;
      stale requests sink and get cut). PNG decode on the worker via the
      existing `image` crate; the main thread only drains an mpsc channel
      and uploads textures — never blocks (osm.el's stutter is exactly
      this: elisp decodes tiles on the UI thread).
    - Tile policy compliance: honest User-Agent (`vv/0.1 (personal
      single-user image/map viewer)`, browser UAs are frowned upon — we
      do NOT spoof Firefox), zoom clamped 2..19, modest concurrency,
      30 s retry cooldown on failed tiles. No attribution overlay for
      phase 1: ODbL binds on distribution, a private single-user viewer
      displays only — one `draw_text` away if that ever changes.
    - Zoom animates between integer levels: from-level tiles GPU-scale
      by 2^±f with the image-view exponential ease (~150 ms, anchor =
      window center), then snap to the new level; its tiles are requested
      during the anim so the settle is seamless.
    - In-memory LRU texture cache (512 tiles = 128 MiB GPU); Texture2D
      drop unloads (raylib-rs does it in Drop), eviction happens on the
      main thread only.
    - Keys: h/j/k/l + arrows pan (pixel-space speed, eased), +/- (also
      numpad, also typed chars for non-US layouts) and wheel and Space
      (Shift+Space = zoom out, osm.el-style) change zoom level, g = home,
      q quits, ESC inert. Status line bottom-left: z level + center
      lat/lon.
