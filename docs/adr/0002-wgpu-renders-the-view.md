# wgpu renders the view

Status: accepted

## Context

The viewer drew through raylib, and raylib owns its window. That blocked three
things the project wants: a host application that embeds the picture (an
`mpv --wid` style child window), a texture the caller owns (the render-API
path), and a correction of the color pipeline, which raylib does in whatever
space the driver gives it.

Issue `94691ce` measured the raylib surface actually in use: 34 calls across 14
files, about 20 draw sites and nine primitives. It also measured a wgpu start on
the development host: 59.1 ms to the first presented frame, of which the Vulkan
loader takes 28.9 ms and shader compilation 0.3 ms.

Two facts decided the shape of the port. The decode layer, the thumbnail cache,
the surface trait, the book geometry, the region math and the MCP server hold no
renderer type at all. And `wgpu::Queue` is `Send + Sync`, so a decode worker can
upload its own texture instead of handing pixels to the main thread.

## Decision

The renderer is `wgpu`, the window and the input are `winit`, and one instanced
quad pipeline draws everything.

- `src/gpu.rs` owns the instance, adapter, device, queue, swapchain and the
  limits.
- `src/render.rs` owns the pipeline, the instance batch and the draw calls.
- `src/texture.rs` owns one uploaded picture, its two filter groups and the
  worker-side `Uploader`.
- `src/draw.rs` owns `Vec2`, `Color` and `Quad`, which carry no GPU handle.
- `src/input.rs` owns the held keys, the frame edges, the typed characters and
  the pointer.

A frame is one method (`App::body`). `Ui` bundles the window, the input and the
clock of that frame, and the `Renderer` takes the draw calls, so the view code
names neither a `Device` nor a `Queue` nor a `BindGroup`.

## The four choices worth naming

**Vulkan only.** `Backends::VULKAN`, crate features `vulkan` and `wgsl`, and
empty instance flags. A DX12, Metal or GL instance would open another library
for a path this build never takes. The cost is that a host without a Vulkan
driver shows no picture; the flake puts the loader on the rpath and the driver
stays the host's.

**Non-sRGB target, straight alpha.** The swapchain takes `Bgra8Unorm` when the
adapter offers it, every texture is `Rgba8Unorm`, and the blend is `SRC_ALPHA`
over `ONE_MINUS_SRC_ALPHA`. The decoder's bytes therefore reach the screen
unchanged, which is exactly what the raylib path showed, and `svg.rs` keeps its
un-premultiply pass. A later move to linear compositing is one decision in
`gpu::pick_format` plus the color constants, not a change spread over the tree.

**Fifo by default.** It never tears and paces the frame loop to the refresh
rate, which replaces raylib's `.vsync()`. `VV_PRESENT` picks another mode for a
latency experiment; the measured spike preferred `Mailbox` for a drag, and that
stays available.

**One instance is four corners.** The quad carries four window-space corners, a
source rect and a tint. Raylib's `DrawTexturePro` shape survives as
`Renderer::texture_pro`, which turns the rectangle about the dest position and
takes the pivot as an origin, so every call site keeps its geometry. A triangle
is a quad whose last corner repeats the first, so the veil bands need no second
primitive, and `cull_mode: None` retires the winding reversal that raylib
required.

## What the workers do now

A decode worker uploads. The grid job builds its thumb texture, the streaming
loader builds each preview and the final frame, and the region painter builds
its patch. Each sends a `texture::Ready`, which is a `Texture` in production and
stays `Pixels` for a caller with no device, which is what a test builds.

The main thread therefore lost the upload funnel, the
`PLACEHOLDER_UPLOADS_PER_FRAME` pacing and the loader texture handoff.

## Consequences

- No C toolchain: the flake carries no CMake, no clang and no bindgen.
- The texture filter is a draw-time choice, so `a` flips one flag.
- A captured drag reads `DeviceEvent::MouseMotion`, so the pointer warp loop is
  gone and the drag speed is a host property, as issue `94691ce` recorded.
- `src/wmclass.rs` is gone: the window name sets `WM_CLASS`.
- The grid keeps no text layer, and its failed cell shows two rectangles.
- An embedding API (a `Viewer` value with `new`, `handle_event` and `render`) is
  still ahead. This port keeps `main.rs`, because the winit loop already owns
  the frame; the split is a refactor of that method and not of the renderer.
- Wayland is not built yet. `winit` supports it, and the only X11-only piece
  left is the `Confined` grab fallback and the `xset` rate of the side buttons.
