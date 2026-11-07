# Embed a pane

Status: accepted. Stage 1 (the window id) is implemented, stage 2 (the
`Viewer` texture) follows on the same branch.

## Context

The viewer draws into a window it owns. A host application wants the picture
inside a window of its own, the way `mpv --wid=<XID>` shows a video, so the
first host is an Emacs package: Emacs creates a container window as a child of
its frame, sizes the container to the body of an Emacs window, and spawns an
engine that draws into that container. EXWM needs nothing from this: a vv
top-level window is an EXWM-managed window already. The container host is the
case that needs a window id, and it must serve more hosts than Emacs.

Two platform facts decided the design, and both come from the first host.

**A child window never learns that its parent shrank.** X sends
`ConfigureNotify` to the resized window itself, so an engine whose window sits
inside a host container keeps its old size and is clipped. `mpv` handles this
by reading the parent geometry with `XGetGeometry` and resizing its own window.
The alternative (the host resizes every child it creates) costs the host work
for every engine and breaks a host that treats the pane as opaque.

**Keyboard focus stays where the host put it.** A click reaches a child window,
but keys go to whatever window holds the input focus. Emacs holds focus on its
frame, so an engine that wants keys needs the host to focus it.

The port that preceded this decision replaced raylib with wgpu and left the
draw seam that stage 2 needs: `Renderer::render_into` paints the instance batch
into any texture view, and `tests/gpu_offscreen.rs` uses it with no window at
all. `Gpu` keeps the instance, the surface and the swapchain, and every texture
is uploaded on a worker through `texture::Uploader`.

Issue `317a7bf` carries the plan and the verification list. Issue `94691ce`
carries the option analysis, the measurement of the raylib surface and the
startup numbers.

## Decision

Two stages, one branch, one decision record.

| stage | host hands over | vv owns | platform |
|---|---|---|---|
| 1. window id | an X11 window id (`--wid 0x...`) | its own surface and swapchain on that child window | X11 (XWayland works) |
| 2. texture | a `wgpu::Device`, `Queue`, format and a target texture | its pipeline, textures, state and draws into the host's view | any wgpu backend, Wayland included |

Stage 1 is the user-visible result with no API split. Stage 2 is the design
that survives Wayland and gives zero-copy compositing. Stage 1 is written so
that stage 2 does not undo it.

### Stage 1: one pane, one policy

`--wid <id>` takes a decimal or a `0x` hexadecimal window id, before or after
the path, in the same shape as `--mcp`. One `Pane` value carries the decision:

```rust
enum Pane {
    Window,
    Embedded(Wid),
}
```

Every policy reads it once, at the site that owns the decision: the window
attributes, the one-image shortcut of the launch, the title update, the two
grid transitions and the open-failure fallback. The pane shows the image view
only, so an embedded launch always opens entry 0 of a directory, `ESC` and
`Enter` never leave the image view, the title stays untouched, and a failed
open exits the process with the decode error because there is no grid to fall
back to. The grid stays alive as data: `n`/`p` step it, the prefetch reads it
and a book's spread count comes from the surface.

The window is created with `with_embed_parent_window`, so `XCreateWindow` runs
with the host window as parent and the child never becomes a top-level. vv
creates its own wgpu surface on that child, and `Gpu` needs no change: its
window carries the X11 handle.

The pane follows the parent size itself, like `mpv`: `src/pane.rs` opens one
`x11rb` connection, reads the parent geometry about ten times a second, and
resizes the child through `Window::request_inner_size`. The host therefore sizes
the container only, which is what the Emacs presenter does. `x11rb` is pure
Rust (MIT OR Apache-2.0) and already in `Cargo.lock` through winit, so the
build still compiles no C.

The host owns focus. vv sends no `XSetInputFocus` and links no X11 call of its
own apart from the geometry read. A host focuses the pane when the user clicks
it or selects its buffer, and the X server then delivers keys to vv. A lost
focus already drops every held key, the pointer grab and every drag latch, so a
host that moves focus mid-drag cannot strand a capture.

`examples/embed_host.rs` is a minimal host: it creates a parent window, spawns
`vv --wid <parent> <path>`, resizes the parent, focuses the child on a click and
reaps the child on exit. It shows the two duties a host owes and gives the path
a demo that needs no Emacs.

### Stage 2: one `Viewer` value

The library gains `src/viewer.rs`, which owns a renderer pipeline on the host's
device, the pipeline's instance batch, the textures of the open source, the
view state, the grid, the loaders and the thumbnail cache. The host owns the
window, the surface, the present, the event loop and the clock.

```rust
Viewer::new(device, queue, format, max_texture_side, path)?;
viewer.handle_event(viewer::Event::Key { .. });
viewer.render(&target_view, size, dt);
```

`viewer::Event` is the viewer's own input vocabulary (key, pointer, travel,
button, wheel, focus), so a host maps from winit, from a toolkit or from a
socket, and `src/input.rs` keeps holding the state. `main.rs` keeps the CLI, the
pane, the winit handler, the window, `Gpu` and the event mapping, and its `App`
becomes a shell around a `Viewer`. `Ui` disappears: its accessors become the
arguments of `handle_event` and `render`.

The draw seam exists: `Renderer::render_into` already paints into an arbitrary
texture view, so stage 2 passes the host's view and does not invent a second
encoder.

## Consequences

- The `--wid` pane is the first X11-shaped piece of the viewer. It reads the
  parent geometry and nothing else, and it lives in one module, so a Wayland
  host goes through stage 2 and never through this module.
- vv claims no XEmbed support. winit sets an atom it names `_XEMBED` to
  `[0, 1]`, not the spec's `_XEMBED_INFO`, and it speaks no XEMBED message, so
  a host that expects the protocol gets no focus message from vv. Such a host
  focuses the pane itself, which is the documented contract.
- A host whose parent visual Vulkan refuses gets a clear error from `Gpu::new`.
  The escape hatch is `with_x11_visual` with the host's visual, which stays a
  host-integration matter: a 24-bit TrueColor parent, which every normal toolkit
  creates, works.
- A pane resize is a `Resized` event like any other, so the swapchain
  reconfigures and the picture refits through the path a standalone window
  already uses.
- An embedded open failure exits the process. A caller who wants a fallback
  drives the pane through stage 2, where `Viewer::open` answers a `Result` and
  the host decides what to show.
