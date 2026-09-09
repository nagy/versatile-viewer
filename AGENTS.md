# versatile-viewer

Fast image (later: video) viewer. nsxiv meets mpv. JPEG XL first-class.

- License: AGPL-3.0-or-later
- Renderer: raylib (Rust bindings). wgpu probably not needed.
- JXL decode: `jxl-oxide`, pure Rust, no libjxl C dependency.
  jxl-oxide supports progressive decoding (partial bytes -> blurry preview,
  full bytes -> full image); planned for a later milestone.
- Common formats: `image` crate (PNG, JPEG, WebP, GIF, BMP, TIFF, TGA;
  webp via pure-Rust image-webp, lossy + lossless decode). Features must
  stay in sync with the grid's extension filter (src/grid.rs `is_image_path`).
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

## Env vars

- `VV_DEBUG=1` — trace grid open/return events and raw key events to stderr.
  During an image-view drag it also draws the virtual cursor (yellow
  crosshair) and grab point (blue ring), and traces grab/release/warp
  events.
- `VV_SLOW_STREAM=1` — see milestone 7.
- `VV_BLUR_BG=1` — gimmick: image-view background is a tiny blurred copy of
  the viewed image (`blurbg` module), scaled to cover the window (fit on the
  narrower side, overflow cropped), GPU-upscaled with bilinear filtering.
  In grid view the background follows the selected entry, crossfading
  (0.4 s) between images; the very first background appears instantly.
  `VV_BG_DIM=0..1` sets its brightness (default 0.6); `VV_BLUR_PX` sets the
  tiny texture's long side (default 128, clamped 8..=1024; fewer =
  blurrier). Off = zero cost, black background as before.

## Milestones (step by step)

1. [x] Single image window: JXL + PNG + JPEG, q quits.
2. [x] Fit to window (aspect preserved, centered, no upscale, refits on resize).
3. [x] Zoom modes: `W` fit-down (default, no upscale), `Shift+W` fit all
   sides (upscales until first border touch), `e` fit width, `Shift+E`
   fit height.
   `t` toggles between fit-all (whole image visible) and fill (window fully
   covered, overflow cropped); distinct for any image/window shape.
   `a` toggles texture filtering in image view: smooth (bilinear,
   default) vs pixelated (nearest-neighbor, 1:1 pixel peeping).
4. [x] Panning: `h/j/k/l` + arrow keys, unrestricted. Left-drag follows the
   cursor directly (no ease; keyboard panning glides), at 2× travel speed.
   While dragging the pointer is captured (hidden, unbounded deltas — no
   screen-edge blocking); on release it is warped back to the grab point,
   re-verified for a few frames against competing re-warps.
   Frame loop paced by vsync (no software FPS cap) — tear-free.
5. [x] Free zoom: `+`/`-` (also numpad, layout-independent via typed
   character), 25% steps; zoom animates smoothly (exponential ease,
   ~150 ms) toward the target scale. Window resize snaps scale/pan
   instantly (no ease glide after the new fit); resizes are detected by
   comparing the window size across frames, so a tiling WM shrinking the
   freshly spawned window also snaps (raylib's resize flag can miss it).
6. [x] Zoom anchored at the cursor (free zoom eases while keeping the image
   point under the mouse fixed; anchor captured when the step starts,
   falls back to window center). While the ease runs, the image point and
   the cursor riding it drift together toward the window center
   (ZOOM_ANCHOR_DRIFT = 0.25 of their remaining distance; pointer warped
   along, never during a drag). Wheel zoom still open.
7. [x] jxl-oxide progressive decoding (stream bytes, render preview
   early; image view only — grid thumbs stay full-decode).
   `VV_SLOW_STREAM=1` dribbles 4 KB chunks with 1 s pauses until the
   first preview renders, for eyeballing the refinement.
8. [x] Directory browsing: thumbnail grid fills the window (square
   center-crop thumbs, dynamic layout, white selection border, background
   decodes on the rayon pool — several in parallel); Enter/ESC open,
   Enter/ESC return. Entries store a 1024px full-aspect thumb texture
   (grid cells center-crop it at draw time; the image view shows it
   full-frame as a placeholder while a decode catches up); the
   full-res texture is kept only for the selection/open entry plus its
   prefetched neighbors (the "keep set"), so Enter opens instantly there
   (no re-decode) and falls back to the streaming loader otherwise; while
   a decode is in flight the viewer waits for it instead of decoding
   twice; failed decodes stay visible as dimmed error cells. Decodes are
   prefetched: grid selection's left/right/up/down
   neighbors, and prev/next while viewing. Opening an image fit-all
   (Shift+W behavior). Space/Backspace or n/p switch prev/next in image
   view (nsxiv-style; arrows and h/j/k/l stay panning); nav keys and grid
   h/j/k/l auto-repeat while held, at the X server's rate (xset r rate);
   h/l wrap between rows (l on a row's last element moves to the next
   row's first element, h on the leftmost moves to the previous row's
   end).
   g/G jump to the first/last image (grid selection and image view).
   +/- zoom the grid thumbs (25% steps; default zoom fills the window
   exactly, zoom-out refits with smaller thumbs, zoom-in overflows and
   scrolls — mouse wheel + selection-follow). Mouse: click selects a grid
   cell, clicking the selected cell opens it; left-drag pans and wheel
   zooms (25% steps, center-anchored) in image view. Dotfiles are hidden;
   formats the grid lists must have matching image-crate features.
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
