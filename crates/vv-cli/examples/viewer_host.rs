// A host spells out the wgpu calls and the window sizes it passes, so the casts
// between them are the point of the file, exactly as they are in `main.rs`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

//! A host that composites a `Viewer` texture into its own window.
//!
//! ```text
//! host        owns the window, the surface, the swapchain, the event loop
//! ├── texture the viewer paints here
//! └── copy    one texture-to-texture copy into the swapchain image
//! ```
//!
//! Run it with `cargo run --example viewer_host -- <path>`.
//!
//! This is the stage-2 shape (`src/viewer.rs`), and the one a Wayland host
//! takes: the viewer is a value on the host's device, it paints a texture the
//! host owns, and the host decides when and where that texture reaches the
//! screen. `examples/embed_host.rs` is the other shape, where the viewer owns a
//! child X11 window and its own swapchain.
//!
//! The copy is the host's whole draw work here. A host that composites the
//! viewer into a scene of its own samples the same texture with its own
//! pipeline instead, and a host on another backend passes the texture it
//! already has. The texture takes the swapchain format, so a plain
//! `copy_texture_to_texture` carries the pixels and no second pipeline exists.

use std::{path::Path, sync::Arc, time::Instant};

use anyhow::{Context, Result, bail};
use vv_core::draw::Vec2;
use vv_gpu::gpu::{FrameError, Gpu, GpuSettings, Present};
use vv_view::viewer::{Event, Host, Launch, Viewer};
#[cfg(target_os = "linux")]
use winit::platform::x11::WindowAttributesExtX11 as _;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::{CursorGrabMode, Window, WindowId},
};

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        bail!("usage: viewer_host <image-path-or-directory>");
    };
    // The launch runs before the window exists: a picture sizes its own window.
    let launch = Launch::open(Path::new(&path), false)?;
    let event_loop = EventLoop::new().context("cannot create the event loop")?;
    let mut host = HostApp::new(launch);
    event_loop
        .run_app(&mut host)
        .context("the event loop failed")?;
    match host.fatal.take() {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// The four window actions the viewer asks of its host.
struct WinitHost<'a> {
    window: &'a Window,
}

impl Host for WinitHost<'_> {
    fn set_title(&mut self, title: &str) {
        self.window.set_title(title);
    }

    fn grab_pointer(&mut self) {
        for mode in [CursorGrabMode::Locked, CursorGrabMode::Confined] {
            if self.window.set_cursor_grab(mode).is_ok() {
                break;
            }
        }
        self.window.set_cursor_visible(false);
    }

    fn release_pointer(&mut self) {
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
    }

    fn place_pointer(&mut self, at: Vec2) {
        let position = winit::dpi::PhysicalPosition::new(at.x.round(), at.y.round());
        let _ = self.window.set_cursor_position(position);
    }
}

/// The host: window, surface, the texture the viewer paints, and the blit.
struct HostApp {
    launch: Option<Launch>,
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    viewer: Option<Viewer>,
    /// The texture the viewer paints.
    target: Option<wgpu::Texture>,
    /// The texture size, so a resize recreates it.
    size: (u32, u32),
    /// The title the viewer falls back to.
    launch_title: String,
    last_frame: Instant,
    fatal: Option<anyhow::Error>,
}

impl HostApp {
    fn new(launch: Launch) -> Self {
        Self {
            size: launch.window_size(),
            launch_title: launch.title().to_string(),
            launch: Some(launch),
            window: None,
            gpu: None,
            viewer: None,
            target: None,
            last_frame: Instant::now(),
            fatal: None,
        }
    }

    /// Hand one input event to the viewer.
    fn send(&mut self, event: Event) {
        let (Some(window), Some(viewer)) = (self.window.clone(), self.viewer.as_mut()) else {
            return;
        };
        let mut host = WinitHost { window: &window };
        viewer.handle_event(event, &mut host);
    }

    /// One frame: the viewer paints the texture, the host blits it and presents.
    fn frame(&mut self) -> Result<()> {
        let Some(window) = self.window.clone() else {
            return Ok(());
        };
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        let size = self.size;
        let (Some(gpu), Some(viewer), Some(target)) = (
            self.gpu.as_mut(),
            self.viewer.as_mut(),
            self.target.as_ref(),
        ) else {
            return Ok(());
        };
        if let Some(reason) = gpu.lost_reason() {
            bail!("the GPU device was lost: {reason}");
        }
        let mut host = WinitHost { window: &window };
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        // The viewer paints the host's texture: its own renderer, its own
        // batch, and no surface in the path.
        let result = viewer.render(&mut host, &target_view, size, dt);
        match gpu.acquire() {
            Ok(frame) => {
                // The host carries those pixels to the screen, which is a copy
                // here and a composite in a host with a scene of its own.
                let extent = wgpu::Extent3d {
                    width: size.0.max(1),
                    height: size.1.max(1),
                    depth_or_array_layers: 1,
                };
                let mut encoder =
                    gpu.device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("viewer host copy"),
                        });
                encoder.copy_texture_to_texture(
                    target.as_image_copy(),
                    frame.texture.as_image_copy(),
                    extent,
                );
                gpu.queue().submit(Some(encoder.finish()));
                gpu.queue().present(frame);
            }
            Err(FrameError::Occluded | FrameError::Timeout) => {}
            Err(err) => {
                if err.needs_reconfigure() {
                    gpu.reconfigure();
                }
                eprintln!("viewer host: frame: {err}");
            }
        }
        result
    }
}

impl ApplicationHandler for HostApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.viewer.is_some() {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            return;
        }
        let Some(launch) = self.launch.take() else {
            return;
        };
        let attributes = Window::default_attributes()
            .with_title(self.launch_title.clone())
            .with_name("vv-viewer-host", "vv-viewer-host")
            .with_inner_size(winit::dpi::PhysicalSize::new(self.size.0, self.size.1))
            .with_resizable(true);
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.fatal = Some(anyhow::anyhow!("cannot create the window: {err}"));
                event_loop.exit();
                return;
            }
        };
        let gpu = match Gpu::new(
            &window,
            GpuSettings {
                present: Present::from_env(),
                latency: vv_gpu::gpu::DEFAULT_LATENCY,
                trace: false,
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => {
                eprintln!("viewer host: {err}");
                self.fatal = Some(anyhow::anyhow!("{err}"));
                event_loop.exit();
                return;
            }
        };
        self.size = window.inner_size().into();
        // The target takes the swapchain format, so the host copies it rather
        // than converting it.
        self.target = Some(target_texture(gpu.device(), gpu.format(), self.size));
        let viewer = Viewer::new(
            gpu.device().clone(),
            gpu.queue().clone(),
            gpu.format(),
            gpu.max_texture_side(),
            launch,
            self.size,
        );
        self.window = Some(Arc::clone(&window));
        self.gpu = Some(gpu);
        self.viewer = Some(viewer);
        self.last_frame = Instant::now();
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => match self.frame() {
                Ok(()) => {
                    if self.viewer.as_ref().is_some_and(Viewer::quit) {
                        event_loop.exit();
                    } else if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                Err(err) => {
                    self.fatal = Some(err);
                    event_loop.exit();
                }
            },
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                }
                if let Some(window) = &self.window {
                    self.size = window.inner_size().into();
                    if let Some(gpu) = self.gpu.as_ref() {
                        // The texture the viewer paints follows the window, so
                        // the copy stays one to one.
                        self.target = Some(target_texture(gpu.device(), gpu.format(), self.size));
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event,
                is_synthetic,
                ..
            } => {
                let PhysicalKey::Code(key) = event.physical_key else {
                    return;
                };
                let text = event.text.as_ref().and_then(|text| text.chars().next());
                self.send(Event::Key {
                    key,
                    state: event.state,
                    repeat: event.repeat,
                    synthetic: is_synthetic,
                    text,
                });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.send(Event::Wheel(vv_view::input::notches(delta)));
            }
            WindowEvent::CursorMoved { position, .. } => self.send(Event::Pointer(Vec2::new(
                position.x as f32,
                position.y as f32,
            ))),
            WindowEvent::MouseInput { state, button, .. } => {
                self.send(Event::Button { button, state });
            }
            WindowEvent::Focused(focused) => self.send(Event::Focus(focused)),
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.send(Event::Travel(Vec2::new(delta.0 as f32, delta.1 as f32)));
        }
    }
}

/// A render target the viewer paints into and the host copies from.
fn target_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    size: (u32, u32),
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer host target"),
        size: wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        // `TEXTURE_BINDING` stays on, so a host that composites can sample the
        // same texture with its own pipeline.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}
