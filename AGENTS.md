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
  `svg::Vector` paints the base raster and any region of it, so one parsed tree serves the thumbnail, the base raster and the window patch.
  A [`surface::VectorSurface`] is that tree behind the `Surface` trait.
  Sniffing reads content: magic, then a BOM, comments, a prolog and a doctype, then an `<svg` root. A gzip head sniffs its decompressed head.
  `MAX_INPUT` (32 MiB) caps the read and the decompression, so a gzip bomb fails.
  External references never resolve (`resources_dir` stays `None`), so no file makes the viewer read another path.
  The host font database loads once per process (`fontdb`, `OnceLock`), so `<text>` renders with real fonts and a whole directory pays one scan.
  `resvg` covers the static subset only: no script, no animation, and `raster-images` stays off, so an embedded JPEG/WebP/GIF draws nothing while an embedded PNG does.
  A retained `usvg::Tree` (`svg::Vector`) paints the view again, so the seam for a deeper zoom stays.
- Rasterize PDF with `hayro` (pure Rust, no C dependency, vello_cpu).
  The module is `src/pdf.rs` (`PdfSurface`, `PdfError`), and it owns the hayro dependency, so a renderer swap touches one file.
  One PDF page is one plane, and the natural unit is the PostScript point (1 pt = 1 px at scale 1.0), so `bounds` answers from a page-size table captured once at open time.
  `open_surface` routes on the `%PDF-` magic, so a misnamed file still opens.
  hayro 0.8 renders through an arbitrary `kurbo::Affine` (`render_into`), so the backend overrides `render` as a region anchor: it translates the requested rect to the origin and rasterizes only those pixels.
  A single render caps at `MAX_RENDER_PIXELS` (about 67 MP) and `MAX_RENDER_SIDE` (65535), and the returned scale is the clamped one.
  `previews` and `can_refine` answer true, so a coarse preview lands first and the image view repaints the window patch over a rested zoom.
  Every render runs in `catch_unwind`, because a young rasterizer can panic on a malformed page and the job runs on a pool thread: a panic becomes `SurfaceError::Render`, never a crash.
  Each render builds a fresh `RenderCache`, because hayro's cache is not `Send`.
- A document shows as a book, `src/book.rs` (`BookSurface`, `COVER_SLOTS`, `COVER_ON_RIGHT`, `GUTTER`, `ROW_GAP`, `SLOTS`, `MAX_RENDER_PIXELS`).
  The whole book is one `Surface` plane, so the loader, the fit, the pan, the turn and the region patch treat it as one tall picture.
  The cover page sits alone in the right slot of row 0, then the pages pair up: `1|2`, `3|4`, one spread under the next.
  `ROW_GAP` is 0, so the spreads stack edge to edge; `GUTTER` (8 pt) separates the two pages of a spread, and both gaps stay transparent, so `BACKGROUND` shows through.
  Every spread measures the widest and tallest page of the document, so the pages line up in two columns down the whole book, and a page narrower than the widest one centers in its slot.
  The book keeps **base units**, not points: `BookSurface::base` bakes the scale that lands the whole book inside `MAX_TEXTURE_SIDE` and `MAX_RENDER_PIXELS` at scale 1.0, because `bounds` must match a scale-1.0 render and the main thread never resizes a buffer.
  `render` converts the request back to points, walks only the spreads the rect touches, and asks each page for its own part at one shared scale.
  `Surface::open_box` names the box a fresh open frames: a book answers its first spread, every other source answers `None`.
  The fit measures that box, so whichever side binds decides and the cover page shows whole.
  A fit centers the whole plane, so `ViewState::align_open_box` holds the pan on `main.rs` `top_pan` until the user moves the view: the align repeats every frame, because a window manager often resizes the window right after it maps and a resize re-fits the scale.
  `ViewState::open_pan` records the pan the align wrote, so a pan or a target that moved away from it marks a user move and the align stops.
  `ViewState::refit_open` arms the same framing again when the loader header or the first frame brings the real plane size.
  `open_book_surface` wraps when the source holds two planes or more; the grid thumbnail keeps the plain `open_surface`, so a cell shows page 0 and never a second page.
- Compile Typst with `typst`, `typst-layout`, `typst-pdf` and `typst-assets` (pure Rust).
  The module is `src/typst.rs` (`TypstSurface`), and it owns those dependencies.
  One compiled document is one multi-plane surface: `TypstSurface` forwards `plane_count`, `bounds`, `render_whole`, `render`, `previews` and `can_refine` to the inner `PdfSurface`, so a Typst page inherits the region anchor and the settled-zoom patch of the PDF path, with one rasterizer and one pixel budget.
  A Typst source holds one plane per page, so `open_book_surface` shows it as the same book as a PDF, and the grid cell shows the cover page.
  The compile runs through a `typst::World` rooted at the directory of the opened file, so `#include`, `#image` and `#read` resolve like the Typst CLI and a path escape fails.
  Fonts come from `typst-assets` (feature `fonts`), so no font directory and no environment variable is necessary.
  The reader has no clock, so a document that reads `datetime.today()` fails.
  A compile error becomes a `file:line:col: error: message` line with the first hint, and a panicking compiler or exporter becomes `SurfaceError::Render`, never a crash.
  `open_surface` routes a `.typ` file by extension, because plain UTF-8 text has no magic bytes to sniff.
- Render Mermaid with `merman` (pure Rust, MIT OR Apache-2.0), the module is `src/mermaid.rs`.
  It owns the dependency, and the `mermaid` cargo feature (a default feature) owns it too.
  The feature pulls `svg`, because the diagram paints through the vector backend: `open` renders the source to an SVG string, parses it with `svg::from_str`, and wraps the tree in `VectorSurface::from_vector`, so one plane shows with the region anchor, the settled-zoom refine and the 512 px grid thumb of a vector file.
  The feature list is `all-diagrams` + `svg`: ELK stays off (EPL-2.0) and so does `math`, so `architecture-beta` answers a capability error and every other family renders.
  Every render passes `SvgPipeline::resvg_safe()`, the preset merman uses for its own PNG and PDF export: it turns a `<foreignObject>` HTML label into an SVG `<text>` element and sanitizes the stylesheet, and without it `usvg` drops every label.
  `svg_label_source` inserts `%%{init: {"htmlLabels": false}}%%` after a leading YAML front matter and before the diagram, because the HTML measure under-reports a wide entity name and an ER header then loses its first and last glyph behind the label viewport; a later directive in the source still wins, so the source keeps its own choice.
  Label boxes measure with a real face: `FONTS` builds one `NativeFontContext` per process from the face files of `svg::fontdb`, so the measure and the paint share one database; the limits rise to 512 MiB and 4096 files because a host holds about 75 MiB of faces.
  A host without a readable face falls back to `SvgEnvironment::deterministic`, and the diagram still renders.
  `Renderer`, the font context and the panic guard (`catch_unwind`) live behind `LazyLock` and `OnceLock`, because the open runs on a decode worker.
  A rejected source, a missing diagram and a panicking renderer all become `SurfaceError::Render`, never a crash.
  `is_mermaid` reads the extension only (`.mmd`), because Mermaid text has no magic bytes.
  A cold `.mmd` cell stays grey until its decode lands, like a Typst cell, because no cheap header parse exists.
- Every showable source is a `src/surface.rs` `Surface`: `plane_count`, `bounds`, `render_whole`, `render`, `progressive` and `previews`.
  `open_surface` sniffs the content with `src/sign.rs` and picks the backend: a raster file (`ImageSurface`), a vector file (`VectorSurface`), a PDF (`PdfSurface`), a Typst source (`TypstSurface`, which compiles to PDF bytes) or a Mermaid source (`src/mermaid.rs`, which renders to SVG text).
  A backend writes `render_whole`; `render` is a provided method that renders the whole plane and crops the region out of it.
  A backend with a region anchor (the vector tree, the PDF rasterizer) overrides `render` to pay only for the pixels the caller asked for, and returns `SurfaceError::Unsupported` rather than the whole plane for a region it cannot serve.
  A `View` names a plane, an optional `Region` and a scale (output pixels per surface unit), so the grid asks for a thumbnail, the loader for the whole plane and the image view for the patch it shows.
  `src/sign.rs` owns the content sniffing (`is_jxl`, `file_magic`, `is_common_magic`, `format`, `dimensions`), and both the surface layer and `svg.rs` use it.
  `Surface::progressive` returns a `Box<dyn ChunkSource>` and not a path, so an in-memory or network source can stream, and the default `Ok(None)` means one step.
  `Surface::can_refine` says whether a settled zoom can yield a sharper picture of the same plane: a vector tree or a document page answers true, a raster image false.
  `Surface::spread_count` and `Surface::spread_box` name the spread column of a book (0 and `None` for every other source), so the image view steps a spread with the book's own geometry.
  A raster plane measures the capped size it renders, so `bounds` and the pixels of a scale-1.0 render agree.
  The trait decisions live in `docs/adr/0001-surface-one-region-based-render-trait.md`.
- The image view repaints the region its window shows, from the open surface, when `Surface::can_refine` says a sharper picture is possible.
  `src/region.rs` holds the pure geometry and policy: `crop`, `to_screen`, `to_image`, `Crop`, `worth_painting`, `MARGIN`, `SCALE_SLACK`, `QUIET`.
  A patch covers the window plus `MARGIN` (0.25 of each side), in source pixels, so the paint area follows the window and a deep zoom costs no more pixels than a shallow one.
  `Surface::render` paints that rect at the view scale, so the patch lands 1:1 on screen; a `VectorSurface` paints only those pixels, and a `PdfSurface` rasterizes only them.
  The region rect lives in surface units: base raster pixels for a vector file (`Vector::declared_per_base` converts to the tree's declared units), points for a PDF page.
  The view repaints when it held still for `QUIET` (150 ms) and `worth_painting` pays: a scale step past `SCALE_SLACK` (1.05) or a consumed margin.
  One paint in flight, because resvg cannot cancel mid-paint.
  `ViewState` keeps `Arc<dyn Surface>`, the `CropState` (texture, source rect, raster scale), the pending flag and one generation counter.
  Every open, every n/p step and every return to the grid bumps that counter, so the result of a worker for an earlier image drops.
  An image opened from the grid opens its own surface on a worker (`request_source`), because the grid keeps none, and keeps it only when `can_refine` is true.
  A single-file launch keeps the surface it already opened, under the same rule.
  `img_w`/`img_h` keep measuring the source, never the patch, so a bigger texture never disturbs fit, pan or zoom.
  The whole-image texture (`view_tex`) stays as the fallback and draws only where the patch cannot reach (`patch_covers`), so a resize or a pan that outruns the repaint never shows background.
  Drawing both over the same pixels would double the alpha of an antialiased edge, so the patch hides the whole image where it covers.
- Grid extension filter: `src/grid.rs` `is_image_path` (also `pdf`, `typ` and `mmd`).
  Keep the crate features in sync (neither `pdf`, `typ` nor `mmd` is an image crate feature).
- A grid cell holds an `EntrySource`, which is a path opened (and sniffed) lazily at decode time.
  One file is one cell, so a document cell shows page 0 and never a second page.
  `Grid::from_paths` builds a grid from a path list, and `Grid::from_dir` hands it the sorted directory listing.
  A single-file launch of a document hands it one path, so ESC still has a grid to return to.
  `EntrySource::open` opens the plain surface (the cell thumbnail); `EntrySource::open_book` opens the book surface (the image view), where plane 0 holds every spread.
  `run_job` answers `no_full` for a multi-plane source, because a spread is not the page its cell holds, so the entry never keeps a full-res texture and `dispatch` never re-decodes it for one.
  `Loader::start_plane` renders one plane, and `Loader::start` is the plane-0 shorthand.
- Dispatch decode on sniffed content.
  Do not dispatch decode on the file name.
- Decode and every pixel resize stay off the main thread.
  The raster backend caps its buffer with `cap_texture_side` (`src/lib.rs`), and `ImageSurface::bounds` reports that capped pair, so `upload_rgba` on the main thread uploads a buffer that already fits `MAX_TEXTURE_SIDE`.
  Never add a main-thread resize: a large image then freezes the window for seconds.
  Downscale a borrowed buffer with `downscale_rgba_ref`, not `downscale_rgba` on a clone.
  `BlurBg::attach` and `upload` take `&DecodedImage`, so a caller never clones the blur buffer.
  All pixel resize goes through `resample_rgba` (`src/lib.rs`), the `fast_image_resize` SIMD path with Lanczos3.
  Do not call `image::imageops::resize` for a downscale: it is several times slower.
  Keep the `rayon` feature of `fast_image_resize` off, so a resize never spawns nested workers inside a decode job.
  `VV_TIMING=1` prints upload cost and slow frames.
- The streaming JXL loader reads fixed-size chunks from the surface's `ChunkSource` (`src/surface.rs`) and feeds them to jxl-oxide.
  The file-backed stream memory-maps the file (`memmap2`), so a chunk copy costs no read syscall.
  A file rewritten under the map faults with SIGBUS instead of reading short data.
- Sort grid entries in natural order with `natord`, so `img2` comes before `img10`.
- Tests: `proptest` covers the pure helpers (layout, visual selection, selection step, the resize fit, the ramp).
  Property tests are dev-only and never gate a release build.
- Public error types derive `derive_more::{Display, Error, From}` (feature `from` on).
  Each module owns its error enum: `src/error.rs` (`DecodeError`, `UploadError`), `src/sign.rs` (`SignError`), `src/surface.rs` (`SurfaceError`), `src/pdf.rs` (`PdfError`), `src/grid.rs` (`GridError`), `src/loader.rs` (`LoadError`), `src/mcp.rs` (`SpawnError`), `src/thumbcache.rs` (`CacheError`).
  `src/typst.rs` has no error enum of its own: its compile diagnostics and export failures become `SurfaceError::Render`.
  `src/mermaid.rs` follows the same rule: a rejected source, a missing diagram and a panicking renderer become `SurfaceError::Render`, and a non-UTF-8 or oversized file becomes `SurfaceError::Unsupported`.
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
- A book view takes `Shift+J` / `Shift+K` as a spread step instead of a pan.
  `ViewState::step_spread` (`main.rs`) reads the row from `Surface::spread_count` and `Surface::spread_box`, eases one spread to the top of the window with `spread_pan`, and stops there: a press past the first or the last spread does nothing.
  The image view drops `Shift+J`/`Shift+K` from the pan while the open surface names a spread column, so those two keys step and never pan on a book.
  The press queue fires once per press, and the `spread_armed` latch in `main` needs a `j`/`k` release before the next step, so a hold never chains steps.
  The row comes from the aimed pan (`spread_row`), not the pixels on screen, so a step pressed while an earlier step still glides continues from the spread that step aimed at.
  The step glides at `SPREAD_EASE_TIME` (4x `EASE_TIME`) while `ViewState::spread_glide` holds the goal, and any other pan motion clears that field.
  A book answers both methods from `BookSurface`, and every other source keeps the defaults (0 and `None`), so its `Shift+J`/`Shift+K` still pans at double speed.
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
  A cold vector entry still measures its base size (`measure_vector`), because the placeholder job parses the root size with an empty font database when the header misses.
  The cell and the image-view fit therefore know the shape of an SVG before its decode, and the pass invents no pixels.
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

- `nix build .#versatile-viewer-doc` makes the rustdoc HTML.- `nix flake check` gates clippy (`--all-targets -- --deny warnings`), tests, doctests, and treefmt.
## Run

Run `nix run . -- <image-path-or-directory>`.
ESC never quits.

The window clears to `BACKGROUND` (`#062329`, `main.rs`).
Both views share it, so the letterbox bars and the grid gaps match.

- Keys: see the README table.
  `Space` steps forward and `Shift+Space` steps back, both as held states, so one key serves both ways.
  Shift at the press edge picks the direction, and the held state follows the same rule, so a hold repeats the way it stepped.
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
  A book holds every spread in one plane, so a step always moves to another file.
- Env vars `VV_DEBUG`, `VV_TIMING`, `VV_SLOW_STREAM`, `VV_BLUR_BG`, `VV_BG_DIM`, `VV_BLUR_PX`, `VV_CACHE`: see the README table.
- Window title uses en-dash separators: `<file> – <dir> – versatile-viewer`.
  The grid title is `(<n> image(s)) – <dir> – versatile-viewer`.
  Canonicalize the paths.
  Abbreviate `$HOME` to `~`.
  The image view title follows the open image.
  The grid keeps the launch title.
  A one-image directory skips the grid at launch.
  A Mermaid source opens as one vector picture, never as a book, because it holds one plane.
  A document launched directly (two planes or more) opens its whole book at once, over a one-cell grid that waits behind ESC, and its title stays the image title.
  A book keeps the default window size, because its canvas can stand wider than a screen and taller than a texture.
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
