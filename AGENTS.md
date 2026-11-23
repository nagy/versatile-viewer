# AGENTS.md — versatile-viewer

## Purpose

Fast image viewer, after nsxiv and mpv.
Video comes later.
JPEG XL is a first-class format, and every decoder is pure Rust, so the build compiles no C.
The tree is a Rust workspace of seven libraries and one binary named `vv`, on the Rust 1.99 toolchain the flake builds with.
License: AGPL-3.0-or-later.

## Layout

```
crates/vv-core/    dim draw pixels scene region region_sel rotate tween keyrepeat push, the shared constants
crates/vv-pixels/  resample_rgba, cap_texture_side, rgba_from_samples, error, placeholder
crates/vv-cache/   Cache, CacheError, CachedThumb, WRITE_BUDGET (thumbnails and converted answers)
crates/vv-gpu/     gpu render display texture
crates/vv-source/  sign surface svg vector_scene pdf pdf_scene book convert treemap decode
crates/vv-view/    input zoomkeys push pane blurbg loader grid view viewer
crates/vv-mcp/     mcp
crates/vv-cli/     main.rs (bin vv), tests/, examples/
```

The edges run one way: `vv-core` <- `vv-pixels` <- {`vv-gpu`, `vv-cache`, `vv-source`} <- `vv-view` <- `vv-cli`, and `vv-mcp` names `vv-core` alone.
A module never names a layer above it, so the compiled graph carries no cycle.
`vv-source` holds no wgpu and no winit, so the whole `Surface` API runs headless.
`crates/vv-cli` holds no library target, so the binary, its integration tests and its examples import each layer by its own crate name and no facade crate sits between them.
ADR 0004 records the layer split, and ADR 0005 the cache split.

A member declares its own dependencies beside their why-comments, and the workspace holds no `[workspace.dependencies]` table, so a comment never drifts from the crate that reads the dependency.
The root `Cargo.toml` carries the metadata and the lint table every member inherits.

## Build and test

### Build

- Run both gates when a change touches the build. `nix flake check` covers the tree, and `nix build .#default` proves the shipped binary, because the release link and the rpath fixup run there and nowhere else.
- Run `nix build .#checks.default` for a fast loop over the test gate alone.
- Run `nix fmt` to format the tree.
- A local run of the GPU paints names the driver itself, because a host without one skips them and the run still passes:
  ```sh
  export VK_DRIVER_FILES=$(nix build --no-link --print-out-paths nixpkgs#mesa)/share/vulkan/icd.d/lvp_icd.x86_64.json
  VV_GPU_REQUIRED=1 cargo test -p vv-cli --test gpu_offscreen -- --ignored
  ```
  `VV_GPU_REQUIRED` turns a missing adapter into a failure, so a run that was meant to prove the paints cannot pass by skipping them.
- Run `nix build .#versatile-viewer-doc` to build the rustdoc HTML of every member.
- The fast gates build in the `dev` profile and only the package builds in `release`.
  `devEnv` in `flake.nix` sets `CARGO_PROFILE = "dev"`, because crane answers `release` when that variable stays unset, and a test binary gains nothing from the cross-crate inlining that `[profile.release]` buys.
  Keep `CARGO_PROFILE_DEV_DEBUG` and `CARGO_PROFILE_TEST_DEBUG` at `1`, so a failing test names its line.
- Every cargo invocation of the flake passes `--workspace --locked`, so one derivation covers every member and a drifted lockfile fails the build instead of fetching.
  `RUSTFLAGS` names `mold` for the package derivation and the check derivations alike, so they share one flag set.
- `flake.nix` carries a why-comment for every knob it sets. Read that comment before you change a derivation.

### Run

- `nix run . -- <image-path-or-directory>` opens a file or a directory. `nix run . -- --mcp` opens a window that waits for a tool. `nix run . -- --wid <id> <path>` draws inside a host window.
- Keys, the region selection, the book step, the converter rules, the embed contract and the environment variables live in the README. Read the README table before you change a key or a variable.
- Both views clear to `BACKGROUND` (`crates/vv-view/src/view.rs`), so the letterbox bars and the grid gaps match.
  A source that paints no paper fills it into its own pixels, so no clear color ever follows a picture.

## Architecture

A path becomes a `Surface`, a `Surface` renders a `View` (plane, optional crop, scale) into an RGBA buffer, a decode worker uploads that buffer as a texture, and the frame draws the textures as quads.
A resolution-independent source may answer a `scene::Primitive` list instead, and then the GPU draws the primitives at the zoom the window shows (ADR 0007).

### Render

- The renderer is `wgpu`, the window is `winit`, and one instanced-quad pipeline draws everything. A picture, a veil band, a selection border and a solid fill are the same draw.
- The backend is Vulkan only (`Backends::VULKAN`, features `vulkan` and `wgsl`), so no DX12, Metal or GL instance exists and no library opens for one. Keep that set fixed: it is the one startup knob that matters.
- One color space and one alpha convention run from the decoder to the swapchain: a non-sRGB swapchain, every texture `Rgba8Unorm`, and straight alpha. Do not mix them.
- `Gpu::new` registers a device-lost callback and the frame loop stops with the driver's reason. A recovery path is not written.
- `MAX_TEXTURE_SIDE` is the decode policy, and `texture::Uploader` clamps every upload to the adapter, so a decoded size never depends on the host.
- Every decode worker uploads its own texture through `texture::Uploader`, because `wgpu::Queue` is `Send + Sync`. Never move a pixel copy back to the main thread.
- A source answers primitives or a raster, and `Surface::display` answers `None` by default so every backend keeps the raster path until it opts in.
  The list holds `scene::Primitive` values in surface units and carries no scale and no region, so a zoom or a pan is a change of transform: the view builds the list once per plane and draws it every frame.
- A source that answers `Some` from `display` is the list, so the view draws neither `view_tex` nor a region patch for it, and `refresh_plane` asks for the list of the new plane instead of painting the plane again.
  Drawing both would double the alpha of an antialiased edge.
- `TreemapSurface`, `VectorSurface`, `PdfSurface` and `BookSurface` answer lists. An SVG's own text arrives as **outlined paths** (`usvg` flattens a text node to its geometry), and a treemap's label is a `Primitive::Glyphs` run this tree lays out itself: the host's sans-serif face from `fontdb`, glyph ids and advance widths from `skrifa`, painted as outlines by vello. `svg::sans_serif_face` reads the face once for the process and hands out clones of one `Arc`, so a label copies a handle and never the face, which is hundreds of kilobytes and would otherwise be copied per label of a map.
- A [`scene::GlyphRun`] carries the font's **bytes** and its collection index, never a face handle, because a primitive crosses a crate edge below the renderer. `vv-gpu::display` parses one `FontData` per run, keyed by the run's bytes, and clears that table with the image table when a different list arrives; the table keeps `FACES` (8) entries, so a document with more faces than that re-parses them. This is the primitive a PDF glyph run lands in.
- A host with no fonts resolves no face, and then a treemap draws its rectangles and no labels — the same picture the raster path painted there, since resvg needed the same fonts.
- A vector document answers `None` when it holds a node the list cannot express, and then its picture stays exactly what the raster path painted: a `filter`, a `mask`, a blend mode other than `normal`, a tiling `pattern`, or a clip built from a group. `vector_scene::Unsupported` names each one and the view logs it at `debug`.
  An embedded picture is a leaf in every encoding `usvg` hands over — PNG, JPEG, GIF and WebP are bytes this crate already decodes — and a nested SVG is a tree, so it is walked with the box its `<image>` names composed into the space. A feature the list cannot express inside that nested tree answers the same `Unsupported` for the whole document.
- A vector tree paints in its declared units and a plane is a base raster, so the walk scales every leaf by `1 / declared_per_base` — the same factor `Vector::render_region` applies to the same tree. A list emitted in declared units draws a 16-unit icon into the corner of a 1024-pixel plane.
- A vector file paints no paper of its own, so its list carries none: where the tree draws nothing the window background shows, which is what the raster path does too.
- A PDF page **does** paint paper: the raster path clears its pixmap to white, so `pdf_scene` puts that white page down as the list's first primitive and every other primitive of the page lands on it. A list without it would show the window color where the page is blank.
- A PDF walks the same `hayro` interpreter the raster path drives (`interpret_page` is generic over `Device`), so the two paths agree by construction: `DrawProps::transform` is the page's own placement, and a glyph run is shaped already, so the walk bakes each glyph's transform into its outline and merges the run into one path.
- A page that needs a soft mask, a blend mode other than `normal`, a tiling pattern, or a shading that a ramp cannot state answers no list (`pdf_scene::Unsupported`), and then the page keeps the raster path whole. A book answers a list only up to `book::LIST_PAGES` pages: a book is one plane, so its list holds every page at once, and past that count it keeps the raster path.
- A shading pattern answers a `scene::Paint::Linear` or `Paint::Radial` ramp, the vocabulary the SVG walk already speaks: `hayro` encodes the shading, samples its colour function into stops at `RAMP_TOLERANCE`, and the encoded shading carries the pattern matrix, which the interpreter pre-composes with the page's own transform, so the ramp lands in surface units — composing the draw's transform as well applies it twice and moves every ramp. A shading that names a `/BBox` clips its shape with a `PushClip`/`PopClip` pair, and one whose extend does not cover the shape it paints keeps the raster path, because a ramp pads past its last stop where the page paints nothing.
- A `Paint::Radial` states its circles in a space of its own (`transform`) and a `Paint::Linear` states its two points in surface units, so a gradient or a shading whose matrix squeezes its circles into ellipses keeps them circles and puts the squash in that space. `vv-gpu::display` hands a ramp a brush transform of `shape⁻¹ * space`, which is what makes a vello fill draw the ellipse the producer named; the glyph path passes the same transform through `brush_transform`. `Paint::placed` composes a placement into that space and moves a linear ramp's points, so a book moves both whole.
- A page's list is in points and its slot already in base units, so `BookSurface::display` places each page by `Affine::translate(slot) * Affine::scale(units_per_point)`: the scale goes innermost, where it converts the page's own units, and the move is stated in the units the book measures. Scaling the slot as well shrinks every page's offset by `units_per_point` and stacks the pages near the first spread — which a short book hides, because it bakes `units_per_point` to exactly 1.0, so a test of the placement needs a book taller than `MAX_TEXTURE_SIDE`.
- `vv-gpu::display::SceneView` is the only vello code in the tree. It renders into a window-sized `Rgba8Unorm` target that the quad pipeline composites as one draw, and it uploads each image leaf once, keyed by the leaf's address, so the table clears when a different list arrives.
  The paint is the one expensive step of that pass, so a frame whose list, transform and size match the paint on screen composites it again and paints nothing, and a frame whose window size moved since the last one reuses the paint and places it by the transform between the two windows (`stale_place`), which keeps a window drag down to a composite per frame and leaves the paint of the settled size to the frame that finds the size at rest.
  A list is encoded **once** in surface units and each frame places the whole scene with one `Scene::append`, so a zoom or a pan costs a copy of the encoded stream and no walk of the paths; vello flattens the encoded curves on the device, so the placed list is as sharp as one encoded for that zoom. Nothing in `encode` may take the frame transform, and an opacity layer clips to the plane rather than to the window, because the base carries no window.
  A face is parsed once per `Arc` of bytes and the table outlives the list, because vello caches the outlines of a face by its bytes.
- A stale answer to a scene request still clears the pending flag: `ViewState::drain_crops` clears `scene_pending` before it checks the generation, and re-asks when the answer described another plane. The launch asks for a list before the first frame measures the plane, and the first frame re-frames a plane that follows the window, so the first answer is nearly always one plane behind.
- The scene type lives in `vv-core`, below both the sources and the renderer, so `Primitive::Image` carries `pixels::DecodedImage` and never a texture handle.
- A `Primitive::Image` transform maps the picture's **own pixels** onto the plane, so the leaf covers its `width` by `height` under it, and `scene::image_of` divides a box by that size to place a picture in one. A producer that also composes the picture's size into the transform squares the scale and throws the picture past the plane, which is the shape a unit-square reading of the leaf produces; `hayro` draws the same picture as a rect of its own size, so `pdf_scene::picture` takes the data's scale factors and nothing more.
- `Renderer::render_into` paints the batch into any texture view and `Renderer::flush` presents the swapchain. A host with no window passes a texture of its own, which is how the `#[ignore]`d offscreen paints read pixels back.
- A captured drag grabs the pointer and reads raw device deltas. A free pointer reads the platform position. `Input::capture` (`crates/vv-view/src/input.rs`) owns that swap, so a hover point never drifts.
- A lost focus forgets every held key and button and drops the grab, because the releases never arrive.
- X replays every held key as a synthetic press when the window gains focus, so `Input::key` marks a synthetic press stale and it never enters the held set. A `viewer::Event::Key` carries the flag.
- A frame is one method, `Viewer::frame`. `main.rs` keeps the window, the surface, the acquire, the present and the CLI, so no view code reads a raw `winit` event.
- The viewer carries no text layer. A failed grid cell shows an exclamation of two rectangles, and an empty directory draws nothing because the title carries the count.

### Diagnostics

- Every layer emits through `tracing`, and the binary owns the subscriber. `vv-source`, `vv-view` and `vv-gpu` name the facade only, so no library decides a level, a format or a destination.
- `RUST_LOG` is the whole filter, and a set `RUST_LOG` wins over the `VV_DEBUG` and `VV_TIMING` aliases.
- The level says who reads the event. `error!` marks a failure the viewer absorbed, `warn!` one the user sees, `info!` the startup fact a support report needs, `debug!` the input geometry and the slow frames, `trace!` the per-frame key press.
- Values ride the fields and the message stays a short constant event name, so one event reads the same in every filter.
- A hot path reads `tracing::enabled!` once when the work beside the event costs real time. No struct field holds a level.
- A span wraps the work that a slow report names. A synchronous worker enters it with a named guard (`let _e = span.enter()`), because `.instrument()` serves an async future.
- Neither a test nor an example inits a subscriber.

### Command line

- `clap` (derive) owns the command line. One `Cli` struct in `main.rs` carries `--mcp`, `--wid <ID>` and the positional path, and clap answers `-h` and `-V` before `main` runs.
- The path stays optional for a `--mcp` run alone (`required_unless_present`), because that run pushes every picture through a tool.
- `pane::Wid` is the one reader of the `--wid` value, so the accepted forms and the `0` rejection live in `crates/vv-view/src/pane.rs`.
- `Cli::try_parse_from` is the test seam, so no test reads `std::env::args`.

### Embed

- `vv --wid <id> <path>` shows the picture inside a host window, the way `mpv --wid` does, and it composes with `--mcp`.
- One `Pane` (`main.rs`) carries the policy: `Window` or `Embedded(Wid)`. Every site reads it once, and no view code carries an `if embedded` test. `Pane::shows_grid` answers the whole difference, so an embedded pane draws the image view only while the grid stays alive as data.
- The child window is created with `WindowAttributesExtX11::with_embed_parent_window`, so it never becomes a top-level window and no window manager frames it.
- The pane follows the host size itself, because X sends the `ConfigureNotify` of a container to the container alone.
- The host owns focus. vv sends no `XSetInputFocus`, so the pane receives keys only while the host focuses the pane's own window.
- vv claims no XEmbed support, so a host that expects the protocol focuses the pane itself.
- `examples/embed_host.rs` and `examples/viewer_host.rs` are two working hosts, one for the X11 pane and one for the library path. Read one before you change a `Host` action.
- `viewer::Viewer` is the library seam. A host builds it on its own device, maps its own toolkit into `viewer::Event`, passes `dt` and presents the texture itself, so the viewer holds no window, no surface and no clock.

### MCP

- `vv --mcp` serves the protocol on stdio from `crates/vv-mcp/src/mcp.rs`, on a tokio current-thread runtime on a background thread, and the viewer window stays interactive.
- A run with no path opens `Launch::blank`, which reads no file, so every picture it shows arrives through a tool.
- The server pins revision `2026-07-28` and serves that revision only. That revision is stateless, so no `initialize` handshake and no session exist.
- The server advertises four tools: `vv_show_image_bytes`, `vv_show_image`, `vv_clear` and `vv_get_status`. `tools()` plus one `call_tool` arm is the whole pair that lands one, and the naming and hint rules live in the module docstring.
- `mcp::Sink` is the whole wall between the protocol layer and the picture, so `vv-mcp` names no texture, no device and no window, and a test drives every tool with a sink of its own.
- `main.rs` implements that sink as `Bridge` over a channel that `App::accept_pushes` drains once per frame, so the frame loop stays the only owner of the view. The main thread never decodes and never uploads a pushed picture.
- stdout carries protocol frames only. Every diagnostic goes to stderr, and no library writes to stdout beside the transport.
- `MAX_PUSH_BYTES` (64 MiB) caps one payload, and the cap reads the encoded text before the decode allocates, so a flood stops first.

## Rules

- Every decoder stays pure Rust: jxl-oxide for JPEG XL, the `image` crate for the common formats, resvg for SVG and SVGZ, and hayro for PDF. The flake carries no CMake, no clang and no bindgen.
- Decode and every pixel resize stay off the main thread. Never add a main-thread resize, because a large image then freezes the window for seconds.
- All pixel resize goes through `resample_rgba` (`crates/vv-pixels/src/lib.rs`), the `fast_image_resize` SIMD path with Lanczos3. Do not call `image::imageops::resize` for a downscale, and keep the `rayon` feature of `fast_image_resize` off, so a resize never spawns nested workers inside a decode job.
- Dispatch decode on sniffed content, never on the file name. The two exceptions are a path a converter rule claims and a `.treemap` path: neither carries content this viewer reads, so the extension is the only signal there is.
- `DecodedImage` fields stay private. `new` and `from_exact` are the entrances, and `width`, `height`, `as_rgba` and `into_rgba` the ways out, so a wrong buffer never exists to check.
  The type lives in `vv-core::pixels` and `vv-pixels` re-exports it, because a `scene::Primitive` names it from below the pixel layer, and `BufferError` is the container's own error rather than a decoder's.
- A source is a `Surface`. A backend writes `render_whole`, and a backend with a region anchor (the vector tree, the PDF rasterizer) overrides `render` to pay only for the pixels the caller asked for. A region it cannot serve answers `SurfaceError::Unsupported` and never the whole plane.
- A `.treemap` path opens `treemap::TreemapSurface`, which reads no byte of the file: the extension asks for the module's sample tree, and the content decides nothing. Its plane takes the shape of the window, so `Surface::frame_window` answers a new plane on a resize, the view re-fits it, and `ViewState::refresh_plane` repaints the whole-plane texture the region patch falls back to, because that texture carries the layout of the frame before it; every other source keeps the one size it reports.
- A raster plane measures the capped size it renders, so `bounds` and the pixels of a scale-1.0 render agree.
- A document of two planes or more opens through `open_book_surface`, so the whole book is one plane and the loader, the fit, the pan, the turn and the region patch treat it as one tall picture. The grid keeps `open_surface`, so a cell shows page 0 and never a second page.
- A format this viewer does not parse reaches a backend through a converter rule only. The converter module always compiles, because the SVG backend paints its answer, so no format engine links into the binary and a new format needs no rebuild.
- Two motion styles, and a key picks one. Auto-repeat (`crates/vv-core/src/keyrepeat.rs`) steps a discrete move, and a velocity ramp (`crates/vv-core/src/tween.rs`) eases a continuous one. Never step a continuous motion on the auto-repeat rate.
- Read the held state of a key that no `KeyCode` constant names by learning its keycode (`crates/vv-view/src/zoomkeys.rs`). A time window cannot stand in for it, because the X server repeats only the key that went down last.
- Public error types derive `derive_more::{Display, Error, From}` (feature `from` on), each module owns its own error enum, and `main.rs` keeps `anyhow`.
- The thumbnail cache is not optional, because no feature guards the crate. A test builds its cache with `Cache::in_dir` over a temporary directory and never with `Cache::new()`, because that one fills the cache of the user.
- A payload change of a cache entry bumps `FORMAT`, which rejects every old entry and needs no migration.

## Conventions

- Wrap raw pixel and window primitives in the `crates/vv-core/src/dim.rs` newtypes (`ImgW`, `ImgH`, `WinW`, `WinH`, `LongSide`, `PreviewPx`, `Rate`), and name a span of time with `std::time::Duration` instead of a float newtype. `Rate` stays, because a repeat rate is a frequency and not a span.
  The accessor is `raw()` and never `get()`, and `as_f32()` is the crossing into the window mapping, so draw and upload code never casts a dimension by hand. A public boundary takes the newtype and not the bare primitive.
- A rectangle is `euclid`'s, not a local type. `dim::Rect` carries `WindowSpace` and `region::Crop` carries `SurfaceSpace`, so the two tags refuse a mix at compile time. `dim::rect`, `dim::whole` and `region::crop` are the constructors, `dim::center_square`, `dim::center_origin` and `dim::contains_point` the three readings the viewer adds, and `draw::Quad` the one thing that turns a rect into corners.
- A point, a size and an offset carry the same tags (`dim::WindowPoint`, `WindowSize`, `WindowVector`, `SurfacePoint`, `SurfaceSize`, `SurfaceVector`). `dim::window_point`, `dim::surface_point` and `dim::draw_window` are the crossings to `draw::Vec2`, so only the draw boundary reads a bare pair.
- Every `clone` names what it buys, and "the borrow checker complained" is not an answer. A borrow conflict is fixed in the structure the error describes, never in a clone that silences it.
  A value that outlives a borrow and has more than one holder is an `Arc`. A handle that moves into a `rayon` closure, an mpsc `Sender` clone and the font database handle are the same kind of thing.
- A function that only reads takes `&Path`, `&str` or `&[T]`, and takes ownership only when it stores the value. No code clones a pixel buffer, because `downscale_rgba_ref` borrows.
- A process-wide table is a `LazyLock` or an `OnceLock`, and each static docstring names why no owner can pass it down. The one `RefCell` (`Grid::layout_cache`) is a memo of a pure function of its key.
- Keep the value layer `const` where it costs nothing. `dim::as_f32`, `draw::Color::tint`, `region::Frame::plane` and the `region_sel` constructors are `const fn`, so a size table or a derived color lands at compile time.
  The workspace does not enable `clippy::missing_const_for_fn`, because it would force `const` on the accessors that read a runtime handle, and a `const` body refuses a later `debug!` or `Vec` call.
- A crate root re-exports the lower modules it names (`pub use vv_core::{dim, draw}` and friends), so a module writes `crate::dim::ImgW` and never spells the crate name. The same line carries what the public API names, so a caller reaches one path per type.
- An arm on an enum of this tree names every variant, because `wildcard_enum_match_arm` is on. An arm on a foreign enum keeps `_` under one `#[allow(clippy::wildcard_enum_match_arm, reason = "…")]`.
- Module internals live in the module `//!` docstring, and the layer a crate adds lives in its `crates/<crate>/src/lib.rs` docstring. Write a note that one file needs there, and write a rule that spans layers here.

## Testing

- `proptest` covers the pure helpers of `region`, the grid layout, the visual selection and the selection step, the resize fit and the ramp. Property tests are dev-only and never gate a release build.
  The `region` proptests keep `failure_persistence` off, so no `proptest-regressions/` entry exists, and a saved case becomes a plain `#[test]` beside the strategies with its shrunk values written out.
- The GPU paints are `#[ignore]`d, because a plain `cargo test` has no adapter. `checks.gpu` runs them on Mesa's lavapipe, which needs no GPU, no device node and no display.
  They cover the clear, a fill, a source rect, straight alpha, a turn, the texel grid of a 1:1 patch, the SVG, PDF, converted-source and book planes end to end, a pushed picture into a blank viewer, the treemap's own list drawn through vello over the plane its window names, an SVG list (a group transform, a ramp, a clip and text) against the plane `resvg` paints of the same document, a listed PDF page against the raster path of that same page (its square, both halves of its picture and its paper), a listed page of three shadings against the raster path (a ramp down one side, a pair of circles and the same circles squeezed into an ellipse), and a listed book that bakes a base scale below one.
- The MCP tests run in two layers: unit tests in `crates/vv-mcp/src/mcp.rs` over an in-memory duplex against a fake sink, and `crates/vv-cli/tests/mcp_stdio.rs` over the real stdin and stdout of a child process.
- `hayro` hints and the list does not: the raster path hints a glyph outline between 4 and 128 pixels per em and rounds the glyph's origin to a whole pixel, and it narrows a hairline against the device scale so the line stays visible on a small raster. A test that compares the two paths budgets a few grey pixels at the edge of every stem and a hairline a little wider on the raster side.
- A test that compares two painters asserts a tolerance and never a pixel budget: vello, resvg and hayro each anti-alias their own way. Where two painters must agree, compare against a raster of the same plane rather than a stored expectation, so a rounding change in either one does not break the test.
- The sandbox resolves no system font, so a test asserts glyph pixels only where a face resolves, or commits a font into itself.
- A behavior change ships a test beside the code it changes.

## Open work

- The display-list work this tree has not done yet lives in the git-bug tracker, one bug per feature (`git-bug bug` lists it): a tiling pattern, a blend mode, a mask, a filter, a group clip, the shadings a ramp cannot state, a book's list per spread, and the smaller threads beside them.
- Add video playback (mpv inspiration). The backend stays open.
