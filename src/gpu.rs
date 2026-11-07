//! The window's GPU stack: instance, adapter, device, queue and swapchain.
//!
//! The module owns every `wgpu` object that outlives one frame, so no view code
//! ever names an `Instance`, an `Adapter` or a `SurfaceConfiguration`.
//! It replaces `raylib::init()` and the GL context that came with it.
//!
//! # Startup
//!
//! The backend is Vulkan only (`Backends::VULKAN`), and the build carries only
//! the `vulkan` and `wgsl` features of `wgpu`.
//! That keeps a DX12, Metal or GL instance out of the process, so no library is
//! opened for one.
//! The measured run in `git bug show 94691ce` spends 28.9 ms of its 59.1 ms in
//! the Vulkan loader's ICD scan, so the backend selection is the one startup
//! knob that matters.
//!
//! # Color
//!
//! The swapchain takes a non-sRGB format when the adapter offers one, because
//! the raylib path did no color conversion either: the bytes a decoder wrote
//! reach the screen as they are, so a ported view draws the same pixels.
//! [`Gpu::format`] answers the format the pipeline must target.

use std::sync::Arc;

use derive_more::{Display, Error, From};
use winit::window::Window;

/// How the swapchain hands a frame to the compositor.
///
/// The default is `Fifo`, which never tears and paces the frame loop to the
/// monitor refresh rate: it replaces raylib's `.vsync()` build flag.
/// `VV_PRESENT` picks another mode for a latency experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Present {
    /// One image per refresh, from a queue. Classic vsync, never tears.
    #[default]
    Fifo,
    /// Adaptive vsync: a late frame does not force a repeat.
    FifoRelaxed,
    /// The newest image replaces a waiting one.
    Mailbox,
    /// No wait at all. Always tears.
    Immediate,
}

impl Present {
    /// Read `VV_PRESENT`, where an unset or unknown value answers the default.
    ///
    /// The environment is read here, at the boundary, and the rest of the code
    /// carries the chosen mode.
    #[must_use]
    pub fn from_env() -> Self {
        let Ok(value) = std::env::var("VV_PRESENT") else {
            return Self::default();
        };
        match value.as_str() {
            "fifo" | "vsync" => Self::Fifo,
            "relaxed" => Self::FifoRelaxed,
            "mailbox" => Self::Mailbox,
            "immediate" => Self::Immediate,
            _ => Self::default(),
        }
    }

    /// The wgpu mode of this choice.
    #[must_use]
    fn mode(self) -> wgpu::PresentMode {
        match self {
            Self::Fifo => wgpu::PresentMode::Fifo,
            Self::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
            Self::Mailbox => wgpu::PresentMode::Mailbox,
            Self::Immediate => wgpu::PresentMode::Immediate,
        }
    }
}

/// The frames the swapchain may queue, on top of the one on screen.
///
/// One keeps the classic double buffer: the picture trails the pointer by about
/// one frame, and no frame waits behind two.
pub const DEFAULT_LATENCY: u32 = 1;

/// What the boundary decides about the GPU stack.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuSettings {
    /// The present mode.
    pub present: Present,
    /// The swapchain image count minus one.
    pub latency: u32,
    /// Print the adapter and the swapchain to stderr once.
    pub trace: bool,
}

/// Why the GPU stack did not come up.
#[derive(Debug, Display, Error, From)]
pub enum GpuError {
    /// The window cannot carry a swapchain.
    #[display("cannot create a surface: {_0}")]
    Surface(#[from] wgpu::CreateSurfaceError),
    /// No device answered the request.
    #[display("no adapter answered: {_0}")]
    Adapter(#[from] wgpu::RequestAdapterError),
    /// The adapter refused to open a device.
    #[display("no device: {_0}")]
    Device(#[from] wgpu::RequestDeviceError),
    /// The adapter offers no swapchain for this surface.
    #[display("the adapter cannot draw to this window")]
    Swapchain,
}

/// Why no frame came out of the swapchain.
///
/// The caller skips the frame, and a [`FrameError::Outdated`] or
/// [`FrameError::Lost`] asks it to configure the swapchain again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, Error)]
pub enum FrameError {
    /// The acquire waited too long.
    #[display("the surface timed out")]
    Timeout,
    /// The window is minimized or hidden.
    #[display("the window is occluded")]
    Occluded,
    /// The swapchain no longer matches the window.
    #[display("the swapchain is outdated")]
    Outdated,
    /// The surface died and needs a new one.
    #[display("the surface was lost")]
    Lost,
    /// A validation error interrupted the acquire.
    #[display("a validation error interrupted the acquire")]
    Validation,
}

impl FrameError {
    /// Does the swapchain need a new configuration before the next acquire?
    #[must_use]
    pub fn needs_reconfigure(self) -> bool {
        matches!(self, Self::Outdated | Self::Lost)
    }
}

/// The device, the queue and the configured swapchain.
pub struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    max_texture_side: u32,
    present: wgpu::PresentMode,
}

impl Gpu {
    /// Bring the whole GPU stack up from scratch.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] when the window carries no surface, no adapter
    /// answers, the adapter opens no device, or the adapter offers no
    /// swapchain for this surface.
    pub fn new(window: &Arc<Window>, settings: GpuSettings) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let surface = instance.create_surface(Arc::clone(window))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            ..Default::default()
        }))?;
        let info = adapter.get_info();
        let max_texture_side = adapter.limits().max_texture_dimension_2d;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("vv"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            }))?;

        let caps = surface.get_capabilities(&adapter);
        let format = pick_format(&caps.formats);
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(GpuError::Swapchain)?;
        config.format = format;
        config.present_mode = settings.present.mode();
        config.desired_maximum_frame_latency = settings.latency.max(1);
        surface.configure(&device, &config);
        let present = config.present_mode;

        if settings.trace {
            eprintln!(
                "vv: gpu backend={:?} adapter=\"{}\" type={:?} driver=\"{}\" {} format={format:?} \
                 srgb={} present={:?} latency={} max_texture={max_texture_side}",
                info.backend,
                info.name,
                info.device_type,
                info.driver_info,
                info.driver,
                format.is_srgb(),
                config.present_mode,
                config.desired_maximum_frame_latency,
            );
        }

        Ok(Self {
            surface,
            device,
            queue,
            config,
            max_texture_side,
            present,
        })
    }

    /// The logical device, for creating pipelines, buffers and textures.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue, which carries uploads and submits.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The swapchain size, in physical pixels.
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// The swapchain format the pipeline must target.
    #[must_use]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// The present mode in use.
    #[must_use]
    pub fn present(&self) -> wgpu::PresentMode {
        self.present
    }

    /// The largest texture side this adapter takes.
    ///
    /// [`crate::MAX_TEXTURE_SIDE`] stays the decode policy, so a picture
    /// decodes to the same size on every host; the upload clamps to this
    /// smaller adapter limit when one exists.
    #[must_use]
    pub fn max_texture_side(&self) -> u32 {
        self.max_texture_side
    }

    /// Resize the swapchain to the window's new pixel pair.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if (width, height) == (self.config.width, self.config.height) {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Configure the swapchain again after a surface error.
    pub fn reconfigure(&mut self) {
        self.surface.configure(&self.device, &self.config);
    }

    /// Take the next frame out of the swapchain.
    ///
    /// # Errors
    ///
    /// Returns [`FrameError`] when the surface is occluded, outdated, lost or
    /// still busy, which the caller answers by skipping the frame.
    pub fn acquire(&self) -> Result<wgpu::SurfaceTexture, FrameError> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => Ok(frame),
            wgpu::CurrentSurfaceTexture::Timeout => Err(FrameError::Timeout),
            wgpu::CurrentSurfaceTexture::Occluded => Err(FrameError::Occluded),
            wgpu::CurrentSurfaceTexture::Outdated => Err(FrameError::Outdated),
            wgpu::CurrentSurfaceTexture::Lost => Err(FrameError::Lost),
            wgpu::CurrentSurfaceTexture::Validation => Err(FrameError::Validation),
        }
    }
}

/// Pick a non-sRGB swapchain format that the surface offers.
///
/// `Bgra8Unorm` is the byte order the raylib path received, and `Rgba8Unorm`
/// is its portable fallback.
/// An sRGB-only surface keeps its first format, and the draw then goes through
/// an sRGB target, which the pixel comparison must know.
fn pick_format(formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
    let wanted = [
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Rgba8Unorm,
    ];
    for format in wanted {
        if formats.contains(&format) {
            return format;
        }
    }
    formats
        .first()
        .copied()
        .unwrap_or(wgpu::TextureFormat::Bgra8Unorm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A surface that offers the raylib byte order gets it, not its sRGB twin.
    #[test]
    fn a_linear_format_wins() {
        let formats = [
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ];
        assert_eq!(pick_format(&formats), wgpu::TextureFormat::Bgra8Unorm);
    }

    /// A surface with no linear format keeps a format it offers.
    #[test]
    fn an_srgb_only_surface_keeps_its_format() {
        let formats = [wgpu::TextureFormat::Bgra8UnormSrgb];
        assert_eq!(pick_format(&formats), wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    /// The default present mode is the one that never tears.
    #[test]
    fn the_default_present_mode_is_fifo() {
        assert_eq!(Present::default(), Present::Fifo);
        assert_eq!(Present::default().mode(), wgpu::PresentMode::Fifo);
    }

    /// Only a stale or lost surface asks for a new configuration.
    #[test]
    fn only_a_stale_surface_needs_reconfiguring() {
        assert!(FrameError::Outdated.needs_reconfigure());
        assert!(FrameError::Lost.needs_reconfigure());
        assert!(!FrameError::Timeout.needs_reconfigure());
        assert!(!FrameError::Occluded.needs_reconfigure());
        assert!(!FrameError::Validation.needs_reconfigure());
    }

    /// The latency never reaches zero, which Vulkan rejects.
    #[test]
    fn the_latency_floor_is_one() {
        assert_eq!(DEFAULT_LATENCY, 1);
    }
}
