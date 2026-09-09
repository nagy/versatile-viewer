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
  - `nix build .#versatile-viewer-doc` — static rustdoc HTML; doctests run
    as `checks.doctests` (`cargo test --doc` needs the lib target in
    `src/lib.rs`, which also hosts the decode/upload helpers)
  - `nix fmt` — format the tree (treefmt: rustfmt, taplo, nixfmt); rustfmt
    runs with extended options (import grouping, 100-col width) via
    `settings.formatter.rustfmt.options` with `lib.mkAfter`
  - `nix flake check` — run checks
  - `nix develop` — dev shell (LD_LIBRARY_PATH + rpath RUSTFLAGS set;
    treefmt wrapper included)
- Run: `nix run . -- <image-path>`; q quits, ESC/Enter toggle grid ↔ image
  view (ESC never quits). Binary name: `vv`.
  `f` action key: grid — open the cell under the cursor (double-click
  without the clicking); image view — cursor on the image: zoom into fill
  aimed at the cursor (like `t`), or zoom out if the image already covers
  the whole window; cursor over the background: back to the grid (ESC).
  In image view, holding `f` instead pans like a left-drag (pointer
  captured/hidden, image follows); the action fires on release only when
  the hold did not travel (tap vs. hold-and-pan).
- Window title (en-dash separated, paths canonicalized then $HOME
  abbreviated to `~`): single image — `<file> – <dir> – versatile-viewer`;
  grid — `(<n> image(s)) – <dir> – versatile-viewer`. Title updates
  dynamically in image view to follow the open image (grid mode keeps the
  launch title). A one-image directory skips the grid at launch and opens
  that image directly (ESC still returns to the grid).
- X11: WM_CLASS stamped to `vv` (both res_name and res_class) after window
  creation via `XSetClassHint` (`src/wmclass.rs`) — raylib only derives it
  from the creation title (path-derived, unusable for window rules).
  No-op on Wayland.

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
   covered, overflow cropped); distinct for any image/window shape; zooming
   in (`t` → fill) centers the fill view on the image point under the mouse,
   clamped so the window never shows background; zooming out recentres.
   `a` toggles texture filtering in image view: smooth (bilinear,
   default) vs pixelated (nearest-neighbor, 1:1 pixel peeping).
4. [x] Panning: `h/j/k/l` + arrow keys, unrestricted. Left-drag follows the
   cursor directly (no ease; keyboard panning glides), at 2× travel speed.
   Holding `f` in image view pans the same way (captured pointer, hidden
   cursor); `f`'s tap action runs on release only if the hold did not move.
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
   along, never during a drag). Keyboard zoom (+/-) anchors at the
   picture center instead and leaves the cursor alone; wheel zoom stays
   cursor-anchored.
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
9. [x] PDF viewing, read-only, no search/TOC/annotation UI:
   - `document.rs`: `Document` trait (page_count/page_info/render) is the
     source abstraction — image files open as one-page documents, PDFs as
     N-page documents; `open_document` sniffs JXL codestream / `%PDF-`
     magic / `image`-crate formats.
   - `pdf.rs`: hayro (pure Rust, no C) rasterizes pages; panics are caught,
     pixel budgets are clamped; the hayro dependency stays inside pdf.rs so
     a pdfium fallback swap touches one file.
   - PDFs in the directory grid show one tile (page-1 thumb); Enter opens a
     page-overview grid (aspect-fit thumbs, ESC pops back); single-file
     launch opens the page grid directly.
   - Page view: n/p/Space navigate pages, g/G first/last; zoom re-renders
     the page at the settled scale (zathura/mupdf trick — sharp text at
     every resting zoom, preview-first like JXL progressive).
10. Video playback (mpv inspiration; backend TBD).
