//! Headless paints: the renderer draws into a texture and the test reads the
//! pixels back.
//!
//! The default gate has no adapter, so every test here is `#[ignore]`, and the
//! `checks.gpu` gate runs them on Mesa's lavapipe, a software Vulkan driver
//! that needs no GPU and no display. The devshell exports that driver, so one
//! command runs the same paints there:
//!
//! ```sh
//! cargo test --test gpu_offscreen -- --ignored
//! ```
//!
//! A test that finds no adapter prints one line and passes, so a run outside
//! the shell stays safe on a host without Vulkan, and `VV_GPU_REQUIRED`, which
//! the shell and the gate set, turns that skip into a failure.
//!
//! # What the tests pin
//!
//! - the clear color, and that an empty frame still clears;
//! - a solid fill, on exactly the pixels of its rect;
//! - the source rect and the texel grid of a texture draw, which is the
//!   sharpness rule of `patch_dest` in the viewer;
//! - straight alpha, and a tint that scales it;
//! - a turned quad, on the pixels its corners name;
//! - the decoder-to-draw chain of an SVG, a PDF page and a converted source;
//! - the transparent gutter of a book plane;
//! - a treemap laid out for the window, which leaves no letterbox and no
//!   background pixel, and its fallback picture after a re-layout;
//! - the display-list pass a worker builds, whose first frames paint the plane.

// The tests read and name window pixels as whole numbers, so a float rect
// becomes a pixel index with one cast, and a dimension becomes an `f32`.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use std::fmt::Write as _;

use vv_core::{
    dim::{self, Rect},
    draw::{Color, Quad, Turn, Vec2},
    theme::THEME,
};
use vv_gpu::{render::Renderer, texture::Filter};
use vv_source::surface::{View, open_surface};

/// The window background, which the viewer clears to.
const CLEAR: Color = Color::new(0x06, 0x23, 0x29, 0xff);

/// One offscreen render target and the renderer that paints it.
struct Canvas {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    target: wgpu::Texture,
    size: (u32, u32),
}

impl Canvas {
    /// Bring up a device with no surface and a target of `width` by `height`.
    ///
    /// `None` means no Vulkan adapter answered, which a test reports and skips.
    fn new(width: u32, height: u32) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vv offscreen"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        let renderer = Renderer::new(
            &device,
            &queue,
            FORMAT,
            adapter.limits().max_texture_dimension_2d,
        );
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vv offscreen target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Some(Self {
            device,
            queue,
            renderer,
            target,
            size: (width, height),
        })
    }

    /// Begin a frame that clears to `clear`.
    fn begin(&mut self, clear: Color) {
        self.renderer.begin(&self.queue, self.size);
        self.renderer.set_clear(clear);
    }

    /// Paint the batch, then read every pixel back as `[r, g, b, a]`.
    fn print(&mut self) -> Pixels {
        let view = self
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render_into(&self.device, &self.queue, &view);
        self.read()
    }

    /// Read every pixel of the target back as `[r, g, b, a]`.
    ///
    /// A caller that painted the target itself (the viewer owns its own
    /// renderer) reads the result here, so one readback serves both.
    fn read(&self) -> Pixels {
        let (width, height) = self.size;
        let unpadded = width * 4;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vv offscreen readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vv offscreen copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("the device polls");
        receiver
            .recv()
            .expect("the map callback runs")
            .expect("mapped");
        let data = buffer
            .slice(..)
            .get_mapped_range()
            .expect("the range maps")
            .to_vec();
        Pixels {
            data,
            width: width as usize,
            height: height as usize,
            padded: padded as usize,
        }
    }
}

/// The format the offscreen target takes, and the one the pipeline targets.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A read-back frame of straight-alpha RGBA8 pixels.
struct Pixels {
    data: Vec<u8>,
    width: usize,
    height: usize,
    padded: usize,
}

impl Pixels {
    /// The pixel at `(x, y)`.
    fn at(&self, x: usize, y: usize) -> [u8; 4] {
        assert!(
            x < self.width && y < self.height,
            "({x},{y}) is off the frame"
        );
        let start = y * self.padded + x * 4;
        [
            self.data[start],
            self.data[start + 1],
            self.data[start + 2],
            self.data[start + 3],
        ]
    }

    /// Does every pixel of this rect answer `color`?
    fn rect_is(&self, rect: Rect, color: [u8; 4]) -> bool {
        let (x0, y0) = (rect.min_x() as usize, rect.min_y() as usize);
        let (x1, y1) = (rect.max_x() as usize, rect.max_y() as usize);
        (y0..y1).all(|y| (x0..x1).all(|x| self.at(x, y) == color))
    }
}

/// Report a host that answers no Vulkan device.
///
/// A plain run skips the test, so the same command is safe on a machine with
/// no driver. A run that sets `VV_GPU_REQUIRED` fails instead, so a gate that
/// provides a driver of its own proves that driver answered.
fn skip_without_adapter() {
    assert!(
        std::env::var_os("VV_GPU_REQUIRED").is_none(),
        "no Vulkan adapter, and VV_GPU_REQUIRED asks for one"
    );
    eprintln!("no Vulkan adapter: skipped");
}

/// Skip a test when the host answers no Vulkan device.
macro_rules! canvas {
    ($width:expr, $height:expr) => {
        match Canvas::new($width, $height) {
            Some(canvas) => canvas,
            None => {
                skip_without_adapter();
                return;
            }
        }
    };
}

/// A host that answers no window action and presses keys for a test.
struct NullHost;

impl vv_view::viewer::Host for NullHost {
    fn set_title(&mut self, _title: &str) {}

    fn grab_pointer(&mut self) {}

    fn release_pointer(&mut self) {}

    fn place_pointer(&mut self, _at: Vec2) {}
}

impl NullHost {
    /// Press and release `key`, so the viewer sees one press edge.
    fn press(&mut self, viewer: &mut vv_view::viewer::Viewer, key: winit::keyboard::KeyCode) {
        for state in [
            winit::event::ElementState::Pressed,
            winit::event::ElementState::Released,
        ] {
            viewer.handle_event(
                vv_view::viewer::Event::Key {
                    key,
                    state,
                    repeat: false,
                    synthetic: false,
                    text: None,
                },
                self,
            );
        }
    }
}

/// The clear color reaches the whole frame, even with nothing to draw.
#[test]
#[ignore = "needs a Vulkan device"]
fn an_empty_frame_clears_to_the_background() {
    let mut canvas = canvas!(8, 4);
    canvas.begin(CLEAR);
    let pixels = canvas.print();
    assert!(
        pixels.rect_is(dim::rect(0.0, 0.0, 8.0, 4.0), [0x06, 0x23, 0x29, 0xff]),
        "the clear color fills the frame"
    );
}

/// A solid fill lands on exactly the pixels of its rect.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_solid_rect_lands_on_its_pixels() {
    let mut canvas = canvas!(16, 16);
    canvas.begin(CLEAR);
    let rect = dim::rect(4.0, 6.0, 8.0, 4.0);
    canvas.renderer.fill_rect(rect, Color::WHITE);
    let pixels = canvas.print();
    assert!(
        pixels.rect_is(rect, [255, 255, 255, 255]),
        "the rect is white"
    );
    // One pixel past each edge stays background.
    assert_eq!(pixels.at(12, 7), [0x06, 0x23, 0x29, 0xff]);
    assert_eq!(pixels.at(5, 10), [0x06, 0x23, 0x29, 0xff]);
}

/// A straight-alpha tint keeps its share of the background.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_half_alpha_fill_blends_with_the_clear() {
    let mut canvas = canvas!(4, 4);
    // White at half alpha over black: every channel reads about half.
    canvas.begin(Color::BLACK);
    canvas.renderer.fill_rect(
        dim::rect(0.0, 0.0, 4.0, 4.0),
        Color::new(255, 255, 255, 128),
    );
    let pixels = canvas.print();
    let [r, g, b, a] = pixels.at(2, 2);
    // The color channels keep their share, and the target stays opaque
    // (`ONE` / `ONE_MINUS_SRC_ALPHA` on the alpha channel).
    for (name, channel) in [("red", r), ("green", g), ("blue", b)] {
        assert!(
            channel.abs_diff(128) <= 1,
            "a half-alpha white reads about 128 in {name}, got {channel}"
        );
    }
    assert_eq!(a, 255, "the window target stays opaque");
}

/// A texture draw maps its source rect and its texel grid.
///
/// This is also the reason `patch_dest` snaps a patch onto whole window pixels:
/// a 1:1 draw reproduces the texels, and a half-pixel offset blends them.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_texture_keeps_its_texels_at_one_to_one() {
    let mut canvas = canvas!(8, 8);
    // A two-by-two texture: red, green on the top row, blue, white below.
    let rgba = [
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ];
    let texture = canvas
        .renderer
        .upload_rgba(&rgba, 2, 2)
        .expect("the buffer matches its size");
    canvas.begin(Color::BLACK);
    // Four window pixels per texel, on whole pixels: nearest keeps the blocks.
    canvas.renderer.texture_pro(
        &texture,
        dim::rect(0.0, 0.0, 2.0, 2.0),
        dim::rect(0.0, 0.0, 8.0, 8.0),
        Turn::NONE,
        Color::WHITE,
        Filter::Nearest,
    );
    let pixels = canvas.print();
    assert_eq!(pixels.at(1, 1), [255, 0, 0, 255], "the red texel");
    assert_eq!(pixels.at(6, 1), [0, 255, 0, 255], "the green texel");
    assert_eq!(pixels.at(1, 6), [0, 0, 255, 255], "the blue texel");
    assert_eq!(pixels.at(6, 6), [255, 255, 255, 255], "the white texel");
}

/// A channel within `slack` of `want`, which a float round trip in the shader
/// needs at the last bit.
fn close(got: [u8; 4], want: [u8; 4], slack: u8) -> bool {
    got.iter()
        .zip(want.iter())
        .all(|(got, want)| got.abs_diff(*want) <= slack)
}

/// The recolor theme maps a black pixel onto the ink and a white one onto the
/// paper.
///
/// The map runs in the fragment stage, on the pixels the quad samples, so this
/// is the whole feature as a picture sees it: no decoder and no cache entry
/// carries a recolored byte, and `Shift+T` is the same draw with the uniform
/// written.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_recolored_picture_maps_black_onto_the_ink() {
    let mut canvas = canvas!(4, 2);
    // A two-by-one texture: black on the left, white on the right.
    let rgba = [0, 0, 0, 255, 255, 255, 255, 255];
    let texture = canvas.renderer.upload_rgba(&rgba, 2, 1).expect("upload");
    canvas.begin(Color::BLACK);
    canvas.renderer.set_theme(&canvas.queue, Some(THEME));
    canvas.renderer.texture_pro(
        &texture,
        dim::rect(0.0, 0.0, 2.0, 1.0),
        dim::rect(0.0, 0.0, 4.0, 2.0),
        Turn::NONE,
        Color::WHITE,
        Filter::Nearest,
    );
    let pixels = canvas.print();
    assert!(
        close(pixels.at(0, 0), THEME.ink.to_bytes(), 1),
        "black is the ink: {:?}",
        pixels.at(0, 0)
    );
    assert!(
        close(pixels.at(3, 0), THEME.paper.to_bytes(), 1),
        "white is the paper: {:?}",
        pixels.at(3, 0)
    );
}

/// A colored pixel keeps its own hue, so a chart or a figure still reads.
///
/// Without that the red line of a plot and the blue one beside it would both
/// turn one shade of the paper, and the legend would say nothing.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_recolored_red_pixel_stays_red() {
    let mut canvas = canvas!(2, 1);
    let rgba = [255, 0, 0, 255, 0, 0, 255, 255];
    let texture = canvas.renderer.upload_rgba(&rgba, 2, 1).expect("upload");
    canvas.begin(Color::BLACK);
    canvas.renderer.set_theme(&canvas.queue, Some(THEME));
    canvas.renderer.texture_pro(
        &texture,
        dim::rect(0.0, 0.0, 2.0, 1.0),
        dim::rect(0.0, 0.0, 2.0, 1.0),
        Turn::NONE,
        Color::WHITE,
        Filter::Nearest,
    );
    let pixels = canvas.print();
    let [red, green, blue, _] = pixels.at(0, 0);
    assert!(red > green && red > blue, "red stays reddest: {red:?}");
    let [red, green, blue, _] = pixels.at(1, 0);
    assert!(blue > red && blue > green, "blue stays bluest: {blue:?}");
}

/// A fill is a coat of color and not a picture, so the theme leaves it alone.
///
/// A veil band, a selection border and a grid gap are fills: recoloring them
/// would turn the dimming veil into the paper and hide the region selection.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_veil_keeps_its_color_while_the_theme_is_on() {
    let mut canvas = canvas!(4, 4);
    canvas.begin(Color::BLACK);
    canvas.renderer.set_theme(&canvas.queue, Some(THEME));
    canvas
        .renderer
        .fill_rect(dim::rect(0.0, 0.0, 4.0, 4.0), Color::WHITE);
    let pixels = canvas.print();
    assert_eq!(pixels.at(1, 1), [255, 255, 255, 255]);
}

/// The theme is on the frame and not on the texture, so a frame that turns it
/// back off paints the same picture plain.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_second_frame_can_turn_the_theme_off_again() {
    let mut canvas = canvas!(2, 1);
    let rgba = [255, 255, 255, 255, 0, 0, 0, 255];
    let texture = canvas.renderer.upload_rgba(&rgba, 2, 1).expect("upload");
    canvas.begin(Color::BLACK);
    canvas.renderer.set_theme(&canvas.queue, Some(THEME));
    canvas.renderer.texture_pro(
        &texture,
        dim::rect(0.0, 0.0, 2.0, 1.0),
        dim::rect(0.0, 0.0, 2.0, 1.0),
        Turn::NONE,
        Color::WHITE,
        Filter::Nearest,
    );
    let pixels = canvas.print();
    assert!(close(pixels.at(0, 0), THEME.paper.to_bytes(), 1));
    canvas.begin(Color::BLACK);
    canvas.renderer.set_theme(&canvas.queue, None);
    canvas.renderer.texture_pro(
        &texture,
        dim::rect(0.0, 0.0, 2.0, 1.0),
        dim::rect(0.0, 0.0, 2.0, 1.0),
        Turn::NONE,
        Color::WHITE,
        Filter::Nearest,
    );
    let pixels = canvas.print();
    assert_eq!(pixels.at(0, 0), [255, 255, 255, 255], "the plain white");
    assert_eq!(pixels.at(1, 0), [0, 0, 0, 255], "the plain black");
}

/// A dimmed cell stays dimmed, because the map runs before the tint.
///
/// The grid multiplies a cell outside the selection with the dim tint, and a
/// map applied after that product would pull the cell back to full paper.
#[test]
#[ignore = "needs a Vulkan device"]
fn the_tint_lands_after_the_map() {
    let mut canvas = canvas!(2, 1);
    let rgba = [255, 255, 255, 255, 255, 255, 255, 255];
    let texture = canvas.renderer.upload_rgba(&rgba, 2, 1).expect("upload");
    canvas.begin(Color::BLACK);
    canvas.renderer.set_theme(&canvas.queue, Some(THEME));
    let dim = Color::new(
        vv_core::DIM_LEVEL,
        vv_core::DIM_LEVEL,
        vv_core::DIM_LEVEL,
        255,
    );
    canvas.renderer.texture_pro(
        &texture,
        dim::rect(0.0, 0.0, 2.0, 1.0),
        dim::rect(0.0, 0.0, 2.0, 1.0),
        Turn::NONE,
        dim,
        Filter::Nearest,
    );
    let pixels = canvas.print();
    let [red, green, blue, _] = pixels.at(0, 0);
    assert!(
        red < THEME.paper.r && green < THEME.paper.g && blue < THEME.paper.b,
        "the dimmed cell is darker than the paper: {:?}",
        pixels.at(0, 0)
    );
}

/// The lightness adjustment is the other curve, and it lands on the shader.
///
/// It pulls the middle of the range toward the ink and keeps the ends, which
/// makes a grey rule or a grey glyph bolder on the page and flattens the middle
/// of a photograph. The default theme leaves it off, so this is the branch a
/// page never takes unless someone asks.
#[test]
#[ignore = "needs a Vulkan device"]
fn the_lightness_adjustment_bends_the_middle_of_the_range() {
    let mut canvas = canvas!(4, 4);
    // One mid-grey texel, which is where the two curves disagree most.
    let texture = canvas
        .renderer
        .upload_rgba(&[128, 128, 128, 255], 1, 1)
        .expect("upload");
    let mut paint = |theme: Option<vv_core::theme::Theme>| {
        canvas.begin(Color::BLACK);
        canvas.renderer.set_theme(&canvas.queue, theme);
        canvas.renderer.texture_pro(
            &texture,
            dim::rect(0.0, 0.0, 1.0, 1.0),
            dim::rect(0.0, 0.0, 4.0, 4.0),
            Turn::NONE,
            Color::WHITE,
            Filter::Nearest,
        );
        let [r, g, b, _] = canvas.print().at(2, 2);
        u32::from(r) + u32::from(g) + u32::from(b)
    };
    let plain = paint(None);
    let straight = paint(Some(THEME));
    let bent = paint(Some(THEME.with_adjust_lightness(true)));
    assert_eq!(
        plain,
        3 * 128,
        "a theme that is off keeps the decoded bytes"
    );
    assert!(
        bent > straight,
        "the bent curve travels toward the ink: {bent} vs {straight}"
    );
}

/// A source rect selects the part of the texture the draw asks for.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_source_rect_crops_the_texture() {
    let mut canvas = canvas!(4, 4);
    let rgba = [
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ];
    let texture = canvas.renderer.upload_rgba(&rgba, 2, 2).expect("upload");
    canvas.begin(Color::BLACK);
    // The lower-right texel, stretched over the whole frame.
    canvas.renderer.texture_pro(
        &texture,
        dim::rect(1.0, 1.0, 1.0, 1.0),
        dim::rect(0.0, 0.0, 4.0, 4.0),
        Turn::NONE,
        Color::WHITE,
        Filter::Nearest,
    );
    let pixels = canvas.print();
    assert!(
        pixels.rect_is(dim::rect(0.0, 0.0, 4.0, 4.0), [255, 255, 255, 255]),
        "the cropped texel fills the dest rect"
    );
}

/// A quarter turn moves the corners the quad names and no others.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_turned_quad_reaches_its_own_corners() {
    let mut canvas = canvas!(40, 40);
    canvas.begin(Color::BLACK);
    // A 20 by 10 rect turned a quarter around its center becomes 10 by 20.
    let dest = dim::rect(20.0, 20.0, 20.0, 10.0);
    let quad = Quad::pro(dest, Turn::new(dim::center_origin(dest), 90.0));
    canvas.renderer.fill(quad, Color::WHITE);
    let pixels = canvas.print();
    // The 20 by 10 rect becomes 10 by 20 around its center, so the turn
    // covers x in 15..25 and y in 10..30.
    assert_eq!(
        pixels.at(20, 12),
        [255, 255, 255, 255],
        "the top of the turn"
    );
    assert_eq!(pixels.at(20, 28), [255, 255, 255, 255], "the bottom");
    assert_eq!(pixels.at(16, 20), [255, 255, 255, 255], "the left");
    assert_eq!(pixels.at(24, 20), [255, 255, 255, 255], "the right");
    // The long sides of the unturned rect stay clear, because the turn pulled
    // them in.
    assert_eq!(pixels.at(12, 20), [0, 0, 0, 255], "outside the turn");
    assert_eq!(pixels.at(28, 20), [0, 0, 0, 255], "outside the turn");
    assert_eq!(pixels.at(20, 6), [0, 0, 0, 255], "outside the turn");
}

/// Paint a whole surface at one scale, and answer the rect it covered.
///
/// The rect is the fit destination, so a test samples the picture and never the
/// letterbox.
fn paint_surface(canvas: &mut Canvas, path: &std::path::Path) -> Rect {
    let surface = open_surface(path).expect("the source opens");
    paint_plane(canvas, surface.as_ref())
}

/// Paint one open plane into the frame, fitted.
fn paint_plane(canvas: &mut Canvas, surface: &dyn vv_source::surface::Surface) -> Rect {
    let image = surface
        .render(View::whole(0, 1.0))
        .expect("the plane renders");
    let texture = canvas
        .renderer
        .upload_rgba(image.as_rgba(), image.width().raw(), image.height().raw())
        .expect("the frame uploads");
    let width = i32::try_from(image.width().raw()).expect("a sane width");
    let height = i32::try_from(image.height().raw()).expect("a sane height");
    let (win_w, win_h) = (canvas.size.0 as f32, canvas.size.1 as f32);
    // Fit the plane into the frame.
    let scale = (win_w / width as f32).min(win_h / height as f32);
    let (dw, dh) = (width as f32 * scale, height as f32 * scale);
    let dest = dim::rect((win_w - dw) / 2.0, (win_h - dh) / 2.0, dw, dh);
    canvas.begin(CLEAR);
    canvas.renderer.texture_pro(
        &texture,
        dim::whole(width as f32, height as f32),
        dest,
        Turn::NONE,
        Color::WHITE,
        Filter::Linear,
    );
    dest
}

/// A listed PDF page paints the box its page names.
///
/// A PDF draws in points with the y axis up, so the page's own transform flips
/// it into the render space `bounds` measures. A square in the page's upper left
/// therefore lands in the window's upper left, and a paper that the list forgot
/// would leave the window color where the page is blank.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_listed_pdf_page_lands_on_its_plane() {
    use vv_core::{
        dim::{SurfaceSize, WindowSize, WindowVector},
        region,
    };

    // A 64 by 64 page that fills its upper half with red, in PDF coordinates
    // (y up), which is the top of the render space (y down).
    let document = pdf_with(1, "1 0 0 rg\n0 32 32 32 re f\n", "");
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("page.pdf");
    std::fs::write(&path, document).expect("write the pdf");
    let surface = vv_source::surface::open_surface(&path).expect("the page opens");
    let list = surface.display(0).expect("a list").expect("a page lists");

    let mut canvas = canvas!(200, 200);
    canvas.begin(CLEAR);
    let plane = surface.bounds(0).expect("the page measures");
    let frame = region::Frame {
        img: SurfaceSize::new(plane.w, plane.h),
        win: WindowSize::new(200.0, 200.0),
        pan: WindowVector::zero(),
        // A page of 64 points fills the window at this scale.
        scale: 200.0 / plane.w,
        angle: 0.0,
    };
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    let painted = pass
        .draw(&canvas.device, &canvas.queue, &list, &frame, canvas.size)
        .expect("the list draws")
        .expect("the list fits the frame");
    canvas.renderer.texture(
        painted.texture,
        painted.src,
        painted.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let pixels = canvas.print();
    assert_eq!(
        pixels.at(50, 50),
        [255, 0, 0, 255],
        "the red half of the page, in the top left"
    );
    assert_eq!(
        pixels.at(150, 150),
        [255, 255, 255, 255],
        "the paper of the page, not the window color"
    );
}

/// A nested document inside an `<image>` walks into the box that names it.
///
/// A nested SVG is a tree and not a picture (see `vector_scene::image_of`), so
/// the walk descends into it with the box composed into the space. A nested
/// document that landed in declared units instead would draw the red square in
/// the corner of the plane.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_nested_document_lands_in_its_box() {
    use vv_core::{
        dim::{SurfaceSize, WindowSize, WindowVector},
        region,
    };

    // A `data:` URI of `bytes`, percent-encoded: a nested document reaches the
    // parser that way, and percent-encoding keeps it readable here.
    fn data_uri(mime: &str, bytes: &[u8]) -> String {
        use std::fmt::Write as _;
        let mut uri = format!("data:{mime},");
        for byte in bytes {
            write!(uri, "%{byte:02X}").expect("a string write");
        }
        uri
    }

    let inner = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
                     <rect width="10" height="10" fill="#ff0000"/>
                   </svg>"##;
    let document = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
             <rect width="200" height="200" fill="#0000ff"/>
             <image x="100" y="0" width="100" height="200" href="{}"/>
           </svg>"##,
        data_uri("image/svg+xml", inner.as_bytes())
    );
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("nested.svg");
    std::fs::write(&path, document).expect("write the svg");
    let surface = vv_source::surface::open_surface(&path).expect("the document opens");
    let list = surface
        .display(0)
        .expect("a list")
        .expect("a nested document lists");

    let mut canvas = canvas!(200, 200);
    canvas.begin(CLEAR);
    let plane = surface.bounds(0).expect("the plane measures").w;
    let frame = region::Frame {
        img: SurfaceSize::new(plane, plane),
        win: WindowSize::new(200.0, 200.0),
        pan: WindowVector::zero(),
        scale: 200.0 / plane,
        angle: 0.0,
    };
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    let painted = pass
        .draw(&canvas.device, &canvas.queue, &list, &frame, canvas.size)
        .expect("the list draws")
        .expect("the list fits the frame");
    canvas.renderer.texture(
        painted.texture,
        painted.src,
        painted.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let pixels = canvas.print();
    assert_eq!(pixels.at(50, 100), [0, 0, 255, 255], "the outer document");
    assert_eq!(
        pixels.at(150, 100),
        [255, 0, 0, 255],
        "the nested document, inside its box"
    );
}

/// A display list paints its rects and its pictures at any zoom.
///
/// This is the seam itself: the list is written in surface units, the frame maps
/// them onto the window, and `vello` renders the primitives into the target the
/// quad pipeline composites. Half the window is a red rect, and a blue picture
/// sits inside the other half, so a wrong transform, a lost primitive and an
/// uncomposited pass all land on a different pixel.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_display_list_paints_its_rects_and_its_pictures() {
    use std::sync::Arc;

    use vv_core::{
        dim::{ImgH, ImgW, SurfaceSize, WindowSize, WindowVector},
        pixels::DecodedImage,
        region,
        scene::{self, Paint, Primitive},
    };

    let mut canvas = canvas!(200, 100);
    canvas.begin(CLEAR);
    let (win_w, win_h) = (canvas.size.0 as f32, canvas.size.1 as f32);
    let frame = region::Frame {
        img: SurfaceSize::new(win_w, win_h),
        win: WindowSize::new(win_w, win_h),
        pan: WindowVector::zero(),
        scale: 1.0,
        angle: 0.0,
    };
    let blue = [0, 0, 255, 255];
    let pixels = DecodedImage::from_exact(ImgW::new(2), ImgH::new(2), blue.repeat(4));
    let list: Arc<[Primitive]> = Arc::from(vec![
        Primitive::Rect {
            rect: scene::rect_of(region::crop(0.0, 0.0, 100.0, 100.0)),
            paint: Paint::Solid(Color::new(255, 0, 0, 255)),
        },
        scene::image_of(region::crop(120.0, 20.0, 40.0, 40.0), pixels),
    ]);
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    let painted = pass
        .draw(&canvas.device, &canvas.queue, &list, &frame, canvas.size)
        .expect("the list draws")
        .expect("the list fits the frame");
    canvas.renderer.texture(
        painted.texture,
        painted.src,
        painted.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let out = canvas.print();
    assert_eq!(out.at(50, 50), [255, 0, 0, 255], "the rect fills its half");
    assert_eq!(out.at(140, 40), blue, "the picture lands in its box");
    assert_eq!(
        out.at(190, 90),
        [CLEAR.r, CLEAR.g, CLEAR.b, CLEAR.a],
        "the transparent base leaves the clear color alone"
    );
}

/// A list draws at every scale it is placed at.
///
/// The pass encodes the list once in surface units and places the whole of it
/// with one append per frame (see `SceneView::draw`), so a zoom has to move the
/// geometry exactly as the frame says and has to keep the curves sharp. The rect
/// here names surface units, and the two scales put its edges on different
/// window pixels.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_zoomed_list_follows_its_frame() {
    use std::sync::Arc;

    use vv_core::{
        dim::{SurfaceSize, WindowSize, WindowVector},
        region,
        scene::{self, Paint, Primitive},
    };

    // The rect covers surface units 25..75 of a plane of 200, and the plane's
    // middle sits on the window's middle at every scale.
    let frame = |scale: f32| region::Frame {
        img: SurfaceSize::new(200.0, 200.0),
        win: WindowSize::new(200.0, 200.0),
        pan: WindowVector::zero(),
        scale,
        angle: 0.0,
    };
    let list: Arc<[Primitive]> = Arc::from(vec![Primitive::Rect {
        rect: scene::rect_of(region::crop(25.0, 25.0, 50.0, 50.0)),
        paint: Paint::Solid(Color::new(255, 0, 0, 255)),
    }]);

    let mut canvas = canvas!(200, 200);
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    let at_one = pass
        .draw(
            &canvas.device,
            &canvas.queue,
            &list,
            &frame(1.0),
            canvas.size,
        )
        .expect("the list draws")
        .expect("the list fits the frame");
    canvas.begin(CLEAR);
    canvas.renderer.texture(
        at_one.texture,
        at_one.src,
        at_one.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let one = canvas.print();
    assert_eq!(one.at(60, 60), [255, 0, 0, 255], "the rect at scale 1");

    // At scale 2 the rect spans window -50..50, so the pixel inside it at scale
    // 1 is background now.
    let at_two = pass
        .draw(
            &canvas.device,
            &canvas.queue,
            &list,
            &frame(2.0),
            canvas.size,
        )
        .expect("the list draws")
        .expect("the list fits the frame");
    canvas.begin(CLEAR);
    canvas.renderer.texture(
        at_two.texture,
        at_two.src,
        at_two.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let two = canvas.print();
    assert_eq!(two.at(20, 20), [255, 0, 0, 255], "the rect at scale 2");
    assert_eq!(
        two.at(60, 60),
        [CLEAR.r, CLEAR.g, CLEAR.b, CLEAR.a],
        "the scale moved the rect's edge past this pixel"
    );
}

/// An opacity layer of a list blends its content at the alpha it names.
///
/// A layer is a buffer, and the list is encoded without a window, so the pass
/// bounds it by the plane and the frame's append carries that clip into the
/// window. A layer that lost its clip would draw nothing, and one that kept the
/// plane's corners in surface units would draw them in the wrong place.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_listed_opacity_layer_blends_its_content() {
    use std::sync::Arc;

    use vv_core::{
        dim::{SurfaceSize, WindowSize, WindowVector},
        region,
        scene::{self, Paint, Primitive},
    };

    let frame = |scale: f32| region::Frame {
        img: SurfaceSize::new(200.0, 200.0),
        win: WindowSize::new(200.0, 200.0),
        pan: WindowVector::zero(),
        scale,
        angle: 0.0,
    };
    let list: Arc<[Primitive]> = Arc::from(vec![
        Primitive::PushLayer { alpha: 0.5 },
        Primitive::Rect {
            rect: scene::rect_of(region::crop(0.0, 0.0, 200.0, 200.0)),
            paint: Paint::Solid(Color::new(255, 0, 0, 255)),
        },
        Primitive::PopLayer,
    ]);

    let mut canvas = canvas!(200, 200);
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    // A zoom, so the layer is a plane the frame magnifies.
    let painted = pass
        .draw(
            &canvas.device,
            &canvas.queue,
            &list,
            &frame(2.0),
            canvas.size,
        )
        .expect("the list draws")
        .expect("the list fits the frame");
    canvas.begin(Color::BLACK);
    canvas.renderer.texture(
        painted.texture,
        painted.src,
        painted.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let out = canvas.print();
    let [r, g, b, a] = out.at(100, 100);
    // A red cover at half alpha over black reads about 128 of red and no green
    // or blue, which is what a layer that lost its content would not read.
    assert!(
        r.abs_diff(128) <= 2,
        "a half-alpha layer reads about 128 of red, got {r}"
    );
    assert_eq!((g, b), (0, 0), "the cover carries no green or blue");
    assert_eq!(a, 255, "the window target stays opaque");
}

/// A window that resizes between two frames shows the picture where the new
/// window puts it, from the paint the pass already has.
///
/// The pass reuses its last paint while the window is still moving (see
/// `SceneView::draw`), so the frame of a window drag composites that paint
/// scaled and moved, and the pixels have to land where a paint of the new window
/// would have put them. A red half and a blue half of the plane tell the answers
/// apart: a stretch that ignored the fit would still show both halves, and a
/// picture that landed at another scale would not.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_resized_window_places_the_paint_it_has() {
    use std::sync::Arc;

    use vv_core::{
        dim::{SurfaceSize, WindowSize, WindowVector},
        region,
        scene::{self, Paint, Primitive},
    };

    // The frame a plane of 200 x 100 fills at the scale of `win`.
    let frame = |win: (f32, f32)| region::Frame {
        img: SurfaceSize::new(200.0, 100.0),
        win: WindowSize::new(win.0, win.1),
        pan: WindowVector::zero(),
        scale: (win.0 / 200.0).min(win.1 / 100.0),
        angle: 0.0,
    };

    let mut canvas = canvas!(400, 200);
    let list: Arc<[Primitive]> = Arc::from(vec![
        Primitive::Rect {
            rect: scene::rect_of(region::crop(0.0, 0.0, 100.0, 100.0)),
            paint: Paint::Solid(Color::new(255, 0, 0, 255)),
        },
        Primitive::Rect {
            rect: scene::rect_of(region::crop(100.0, 0.0, 100.0, 100.0)),
            paint: Paint::Solid(Color::new(0, 0, 255, 255)),
        },
    ]);
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    let _ = pass
        .draw(
            &canvas.device,
            &canvas.queue,
            &list,
            &frame((200.0, 100.0)),
            (200, 100),
        )
        .expect("the list draws")
        .expect("the list fits the frame");

    // The window doubles, so the paint of the first frame covers the whole of
    // it and the halves stay halves.
    let painted = pass
        .draw(
            &canvas.device,
            &canvas.queue,
            &list,
            &frame((400.0, 200.0)),
            canvas.size,
        )
        .expect("the list draws")
        .expect("the list fits the frame");

    canvas.begin(CLEAR);
    canvas.renderer.texture(
        painted.texture,
        painted.src,
        painted.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let out = canvas.print();
    assert_eq!(
        out.at(100, 100),
        [255, 0, 0, 255],
        "the red half, at the left"
    );
    assert_eq!(
        out.at(300, 100),
        [0, 0, 255, 255],
        "the blue half, at the right"
    );
}

/// A treemap paints the window it was framed for, edge to edge.
///
/// The plane takes the shape of the window (see `Surface::frame_window`), so a
/// fit maps it 1:1 onto the frame and no window color survives between two
/// rectangles.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_treemap_leaves_no_background_pixel() {
    use vv_source::{
        surface::{Size, Surface},
        treemap::{TreemapSurface, sample_tree},
    };

    let mut canvas = canvas!(240, 150);
    let surface = TreemapSurface::from_tree(sample_tree());
    let window = Size {
        w: canvas.size.0 as f32,
        h: canvas.size.1 as f32,
    };
    surface.frame_window(window);
    let dest = paint_plane(&mut canvas, &surface);
    // The frame is the plane shape, so the fit fills it exactly.
    assert_eq!(
        (dest.min_x(), dest.min_y(), dest.width(), dest.height()),
        (0.0, 0.0, window.w, window.h),
        "the plane covers the frame"
    );
    let pixels = canvas.print();
    let clear = [CLEAR.r, CLEAR.g, CLEAR.b, CLEAR.a];
    let strays = (0..canvas.size.1 as usize)
        .flat_map(|y| (0..canvas.size.0 as usize).map(move |x| (x, y)))
        .filter(|(x, y)| pixels.at(*x, *y) == clear)
        .count();
    assert_eq!(strays, 0, "the window color shows nowhere in the frame");
}

/// A treemap that lays itself out again draws the layout of the window it names.
///
/// The launch frames the default window, and a window of another shape makes the
/// source lay the tree out again (see `Surface::frame_window`).
/// A treemap answers a display list, so a frame draws that list over the plane
/// the source reports now. Every pixel outside the zoomed patch is compared
/// against a raster of the same plane, so the geometry of the list and the
/// geometry of the SVG document that used to paint it have to agree.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_resized_treemap_repaints_its_fallback_picture() {
    let mut canvas = canvas!(1200, 800);
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("map.treemap");
    // The bytes of a `.treemap` file decide nothing, so any file asks for the map.
    std::fs::write(&path, "the tree lives in the code, not here\n").expect("write the file");
    let launch = vv_view::viewer::Launch::open(&path, true).expect("the launch opens");
    assert_eq!(
        launch.window_size(),
        (1280, 800),
        "the launch frames the default window"
    );
    let mut viewer = vv_view::viewer::Viewer::new(
        canvas.device.clone(),
        canvas.queue.clone(),
        FORMAT,
        canvas.device.limits().max_texture_dimension_2d,
        launch,
        canvas.size,
    );
    let mut host = NullHost;
    treemap_fallback_matches_the_plane(&mut canvas, &mut viewer, &mut host);
}

/// The same, for a map a grid cell opens.
///
/// A directory launch lists before it decodes, so a cell opens the map and the
/// grid's own decode lands beside it.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_treemap_opened_from_the_grid_repaints_its_fallback_picture() {
    let mut canvas = canvas!(1200, 800);
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(dir.path().join("a.svg"), SMALL_SVG).expect("write the svg");
    std::fs::write(dir.path().join("b.treemap"), "no bytes matter\n").expect("write the map");
    let launch = vv_view::viewer::Launch::open(dir.path(), true).expect("the launch opens");
    let mut viewer = vv_view::viewer::Viewer::new(
        canvas.device.clone(),
        canvas.queue.clone(),
        FORMAT,
        canvas.device.limits().max_texture_dimension_2d,
        launch,
        canvas.size,
    );
    let mut host = NullHost;
    // Let the grid decode both cells, then open the map in the second one.
    for _ in 0..60 {
        shot_frame(&canvas, &mut viewer, &mut host, 0.016, true);
    }
    host.press(&mut viewer, winit::keyboard::KeyCode::ArrowRight);
    host.press(&mut viewer, winit::keyboard::KeyCode::Enter);
    treemap_fallback_matches_the_plane(&mut canvas, &mut viewer, &mut host);
}

/// A blue 16 px square, which is a grid cell beside a map.
const SMALL_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect width="16" height="16" fill="#0000ff"/></svg>"##;

/// Run one frame of `viewer`, and optionally let the wall clock pass.
///
/// The region patch waits for the view to hold still, so a settle needs the wall
/// clock and not only the frame time.
fn shot_frame(
    canvas: &Canvas,
    viewer: &mut vv_view::viewer::Viewer,
    host: &mut NullHost,
    dt: f32,
    pause: bool,
) {
    let view = canvas
        .target
        .create_view(&wgpu::TextureViewDescriptor::default());
    viewer
        .render(host, &view, canvas.size, dt)
        .expect("the frame renders");
    if pause {
        std::thread::sleep(std::time::Duration::from_millis(12));
    }
}

/// Settle the window, zoom in, return to the whole plane and read the pixels the
/// region patch cannot reach.
///
/// The patch covers the middle of the window after a zoom in, so the pixels
/// outside it come from the fallback picture alone.
fn treemap_fallback_matches_the_plane(
    canvas: &mut Canvas,
    viewer: &mut vv_view::viewer::Viewer,
    host: &mut NullHost,
) {
    use vv_source::{
        surface::{Size, Surface},
        treemap::{TreemapSurface, sample_tree},
    };

    // The window center is the zoom anchor, so the pan stays where `w` puts it.
    let (win_w, win_h) = (canvas.size.0 as f32, canvas.size.1 as f32);
    viewer.handle_event(
        vv_view::viewer::Event::Pointer(Vec2::new(win_w / 2.0, win_h / 2.0)),
        host,
    );
    // Settle the plane this window names: the patch covers the window and the
    // fallback picture is painted for it.
    for _ in 0..60 {
        shot_frame(canvas, viewer, host, 0.016, true);
    }
    // Zoom in, so the patch on screen covers the middle of the plane only.
    for _ in 0..6 {
        viewer.handle_event(vv_view::viewer::Event::Wheel(1.0), host);
        for _ in 0..8 {
            shot_frame(canvas, viewer, host, 0.016, true);
        }
    }
    for _ in 0..40 {
        shot_frame(canvas, viewer, host, 0.016, true);
    }
    // Back to the whole plane: the patch no longer reaches the window, so the
    // fallback picture paints the frame, and the quiet window has not elapsed.
    host.press(viewer, winit::keyboard::KeyCode::KeyW);
    for _ in 0..6 {
        shot_frame(canvas, viewer, host, 0.1, false);
    }
    let pixels = canvas.read();

    // What the plane of this window paints, which is what the fallback picture
    // must hold.
    let surface = TreemapSurface::from_tree(sample_tree());
    surface.frame_window(Size { w: win_w, h: win_h });
    let want = surface
        .render(vv_source::surface::View::whole(0, 1.0))
        .expect("the plane renders");
    assert_eq!(
        (want.width().raw(), want.height().raw()),
        canvas.size,
        "the plane measures the window"
    );
    // The plane of the frame the launch painted, drawn at this window's size: the
    // layout a frame that kept the old picture would show.
    let stale = TreemapSurface::from_tree(sample_tree());
    let stale_scale = win_w / stale.bounds(0).expect("a frame").w;
    let before = stale
        .render(vv_source::surface::View::whole(0, stale_scale))
        .expect("the old plane renders");
    let (inner_x, inner_y) = (300..900, 200..600);
    // Two rasterizers paint this frame now: vello draws the list and resvg draws
    // the reference, and they disagree on every antialiased edge and on every
    // glyph, so the test asks which layout the frame shows and not whether two
    // painters agree.
    let mut wrong = 0;
    let mut stale_wrong = 0;
    for y in 0..canvas.size.1 as usize {
        for x in 0..canvas.size.0 as usize {
            if inner_x.contains(&x) && inner_y.contains(&y) {
                continue;
            }
            let got = pixels.at(x, y);
            let at = (y * canvas.size.0 as usize + x) * 4;
            let wanted = &want.as_rgba()[at..at + 4];
            if got.iter().zip(wanted).any(|(a, b)| a.abs_diff(*b) > 1) {
                wrong += 1;
            }
            // The stale raster covers another aspect, so the window point is
            // sampled through its own extents.
            let (bw, bh) = (
                before.width().raw() as usize,
                before.height().raw() as usize,
            );
            let bx = x * bw / canvas.size.0 as usize;
            let by = y * bh / canvas.size.1 as usize;
            let was = &before.as_rgba()[(by * bw + bx) * 4..][..4];
            if got.iter().zip(was).any(|(a, b)| a.abs_diff(*b) > 1) {
                stale_wrong += 1;
            }
        }
    }
    assert!(
        wrong * 2 < stale_wrong,
        "the frame matches the layout of this window ({wrong}) no better than the layout of the \
         frame before it ({stale_wrong})"
    );
}

/// A vector file draws from its list, and lands where its raster lands.
///
/// The document exercises the whole walk: a group transform, a ramp, a clip and
/// a run of text, which `usvg` hands over as outlined glyphs. Three points pin
/// the geometry, and the glyph band pins that the outlines landed somewhere
/// rather than at the origin.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_listed_svg_lands_where_its_plane_does() {
    use vv_core::{
        dim::{SurfaceSize, WindowSize, WindowVector},
        region,
        scene::Primitive,
    };

    let document = r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
        <defs>
          <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stop-color="#0000ff"/>
            <stop offset="1" stop-color="#0000ff"/>
          </linearGradient>
          <clipPath id="c"><rect x="0" y="0" width="50" height="200"/></clipPath>
        </defs>
        <g transform="translate(20,20)"><rect width="80" height="80" fill="#ff0000"/></g>
        <rect x="120" y="20" width="60" height="60" fill="url(#g)"/>
        <g clip-path="url(#c)"><rect x="0" y="120" width="200" height="60" fill="#00ff00"/></g>
        <text x="100" y="196" font-size="10" fill="#ffffff" text-anchor="middle">vv</text>
      </svg>"##;
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("scene.svg");
    std::fs::write(&path, document).expect("write the svg");
    let surface = vv_source::surface::open_surface(&path).expect("the document opens");
    let list = surface
        .display(0)
        .expect("a list")
        .expect("a plain document lists");
    assert!(
        list.iter()
            .any(|primitive| matches!(primitive, Primitive::PushClip { .. })),
        "the clip rides the list"
    );

    let mut canvas = canvas!(200, 200);
    canvas.begin(CLEAR);
    // A vector file's plane is a base raster rather than the size the document
    // declares, so the frame fits that plane onto the window and the declared
    // coordinates land on the window pixels they name.
    let plane = surface.bounds(0).expect("the plane measures").w;
    let frame = region::Frame {
        img: SurfaceSize::new(plane, plane),
        win: WindowSize::new(200.0, 200.0),
        pan: WindowVector::zero(),
        scale: 200.0 / plane,
        angle: 0.0,
    };
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    let painted = pass
        .draw(&canvas.device, &canvas.queue, &list, &frame, canvas.size)
        .expect("the list draws")
        .expect("the list fits the frame");
    canvas.renderer.texture(
        painted.texture,
        painted.src,
        painted.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let pixels = canvas.print();

    // The group transform moved the square to 20..100.
    assert_eq!(pixels.at(60, 60), [255, 0, 0, 255], "the square, moved");
    assert_eq!(
        pixels.at(10, 10),
        [CLEAR.r, CLEAR.g, CLEAR.b, CLEAR.a],
        "nothing paints outside it"
    );
    // The ramp is one color here, so the fill is checked, not the ramp.
    assert_eq!(pixels.at(150, 50), [0, 0, 255, 255], "the ramp's rect");
    // The clip keeps the left half of the green band and drops the right.
    assert_eq!(pixels.at(30, 150), [0, 255, 0, 255], "inside the clip");
    assert_eq!(
        pixels.at(150, 150),
        [CLEAR.r, CLEAR.g, CLEAR.b, CLEAR.a],
        "outside the clip"
    );
    // The whole frame against the plane `resvg` paints of the same document: two
    // painters disagree on every antialiased edge, so the budget is generous, and
    // a list that landed somewhere else misses it by far.
    // The comparison is font-neutral: a host with no fonts flattens the text to
    // nothing in both painters.
    let want = surface
        .render(vv_source::surface::View::whole(0, 200.0 / plane))
        .expect("the plane renders");
    assert_eq!(
        (want.width().raw(), want.height().raw()),
        canvas.size,
        "the plane paints at the window size"
    );
    let off = misses_the_pixels(&pixels, &want, canvas.size);
    assert!(
        off * 50 < 200 * 200,
        "{off} of 40000 pixels miss the plane, so the list did not land on it"
    );
}

/// A listed document paints its plane until the pass that draws its list lands.
///
/// `vello` compiles the pass's compute pipelines when it is built, which is a
/// moment of work no frame can afford, so a worker builds it: the frame that
/// asks for the pass paints the plane texture, and the frame that finds the
/// pass installed draws the list. The two regimes show the same picture, which
/// is what the two comparisons below pin.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_listed_document_paints_its_plane_until_the_pass_lands() {
    let document = r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
        <g transform="translate(20,20)"><rect width="80" height="80" fill="#ff0000"/></g>
        <path d="M 120 20 L 180 20 L 150 80 Z" fill="#00ff00"/>
      </svg>"##;
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("scene.svg");
    std::fs::write(&path, document).expect("write the svg");
    let launch = vv_view::viewer::Launch::open(&path, false).expect("the launch opens");
    // The launch frames the plane a vector file renders at scale 1.0, so the
    // window is that plane's own size and neither regime resamples it.
    let (win_w, win_h) = launch.window_size();
    let canvas = canvas!(win_w, win_h);
    let mut viewer = vv_view::viewer::Viewer::new(
        canvas.device.clone(),
        canvas.queue.clone(),
        FORMAT,
        canvas.device.limits().max_texture_dimension_2d,
        launch,
        canvas.size,
    );
    assert!(!viewer.listed(), "a launch builds no display-list pass");
    let mut host = NullHost;
    // The launch asks a worker for the list, and a tiny document walks in a
    // moment, so this wait is what puts the list in the first frame: the frame
    // that asks for the pass must still paint the raster, and a pass built in
    // the frame would draw the list here and leave `raster` empty below.
    std::thread::sleep(std::time::Duration::from_millis(200));

    // The build starts in the frame that first holds a list and answers that
    // frame with `false`, so at least one frame paints the plane texture: the
    // one this loop keeps as the raster regime.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let mut raster = None;
    let mut frames = 0;
    let listed = loop {
        assert!(
            std::time::Instant::now() < deadline,
            "the worker's display-list pass lands"
        );
        let pixels = shot_frame_read(&canvas, &mut viewer, &mut host);
        frames += 1;
        if viewer.listed() {
            break pixels;
        }
        raster = Some(pixels);
        std::thread::sleep(std::time::Duration::from_millis(4));
    };
    assert!(frames > 1, "the frame that asks for the pass draws no list");
    let raster = raster.expect("a frame of the build drew the raster");

    // The picture the plane paints of this document, which both regimes show.
    let surface = open_surface(&path).expect("the document opens");
    let want = surface
        .render(View::whole(0, 1.0))
        .expect("the plane renders");
    assert_eq!(
        (want.width().raw(), want.height().raw()),
        canvas.size,
        "the plane paints at the window size"
    );
    let total = canvas.size.0 as usize * canvas.size.1 as usize;
    for (regime, pixels) in [("the raster", raster), ("the list", listed)] {
        let off = misses_the_pixels(&pixels, &want, canvas.size);
        assert!(
            off * 50 < total,
            "{off} of {total} pixels miss the plane in {regime}"
        );
    }
}

/// A scene document paints the list it states, and its own extent frames it.
///
/// `vv --scene <file>` is the route a producer outside this process takes: a
/// plane, a display list and no raster at all, so the frame that waits for the
/// display-list pass shows the window background and nothing else, and every
/// frame after it shows the shapes.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_scene_file_paints_its_list() {
    let document = r#"{
      "version": 1,
      "plane": { "width": 200.0, "height": 100.0 },
      "primitives": [
        { "Rect": { "rect": { "x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0 },
                    "paint": { "Solid": { "r": 255, "g": 0, "b": 0, "a": 255 } } } },
        { "Rect": { "rect": { "x0": 100.0, "y0": 0.0, "x1": 200.0, "y1": 100.0 },
                    "paint": { "Solid": { "r": 0, "g": 255, "b": 0, "a": 255 } } } }
      ]
    }"#;
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("scene.json");
    std::fs::write(&path, document).expect("write the scene");
    let launch = vv_view::viewer::Launch::scene(&path, false).expect("the scene opens");
    // The launch keeps the default window, because the extent is in the units the
    // list is written in: the fit decides how many window pixels a unit covers.
    let (win_w, win_h) = launch.window_size();
    assert_eq!((win_w, win_h), (1024, 768));
    let canvas = canvas!(win_w, win_h);
    let mut viewer = vv_view::viewer::Viewer::new(
        canvas.device.clone(),
        canvas.queue.clone(),
        FORMAT,
        canvas.device.limits().max_texture_dimension_2d,
        launch,
        canvas.size,
    );
    let mut host = NullHost;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let mut waiting = None;
    let listed = loop {
        assert!(
            std::time::Instant::now() < deadline,
            "the display-list pass lands"
        );
        let pixels = shot_frame_read(&canvas, &mut viewer, &mut host);
        if viewer.listed() {
            break pixels;
        }
        waiting.get_or_insert(pixels);
        std::thread::sleep(std::time::Duration::from_millis(4));
    };
    let clear = [CLEAR.r, CLEAR.g, CLEAR.b, CLEAR.a];
    // A scene has no raster behind it, so a frame that draws no list falls back
    // to nothing: the background is the whole picture.
    let waiting = waiting.expect("a frame of the build drew no list");
    assert_eq!(
        waiting.at(256, 384),
        clear,
        "there is no plane to fall back to"
    );
    // The plane is 200 by 100 in a 1024 by 768 window, so the fit is 5.12 and
    // the plane fills the middle 1024 by 512 of the window.
    assert_eq!(
        listed.at(256, 384),
        [255, 0, 0, 255],
        "the left half is the red rect"
    );
    assert_eq!(
        listed.at(768, 384),
        [0, 255, 0, 255],
        "the right half is the green rect"
    );
    assert_eq!(
        listed.at(256, 8),
        clear,
        "the letterbox keeps the window background"
    );
}

/// A glyph run whose bytes name no face is dropped, and the frame still paints.
///
/// A run carries the producer's own bytes, so the face is the one field this
/// tree cannot trust: vello parses it with an `unwrap`, so the pass checks it
/// first. A refused run costs its labels and never the process.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_glyph_run_that_names_no_face_paints_nothing() {
    use std::sync::Arc;

    use vv_core::{
        dim::{SurfaceSize, WindowSize, WindowVector},
        region,
        scene::{self, GlyphRun, Paint, PlacedGlyph, Primitive},
    };

    let mut canvas = canvas!(200, 100);
    canvas.begin(CLEAR);
    let (win_w, win_h) = (canvas.size.0 as f32, canvas.size.1 as f32);
    let frame = region::Frame {
        img: SurfaceSize::new(win_w, win_h),
        win: WindowSize::new(win_w, win_h),
        pan: WindowVector::zero(),
        scale: 1.0,
        angle: 0.0,
    };
    // Two runs a document can state: bytes that are not a font at all, and a
    // face index past the end of the one empty file.
    let run = |bytes: &[u8], face: u32| GlyphRun {
        font: Arc::from(bytes.to_vec()),
        face,
        size: 24.0,
        glyphs: Arc::from(vec![PlacedGlyph {
            id: 1,
            x: 120.0,
            y: 60.0,
        }]),
    };
    let list: Arc<[Primitive]> = Arc::from(vec![
        Primitive::Rect {
            rect: scene::rect_of(region::crop(0.0, 0.0, 100.0, 100.0)),
            paint: Paint::Solid(Color::new(255, 0, 0, 255)),
        },
        Primitive::Glyphs {
            run: run(b"not a font at all", 0),
            transform: kurbo::Affine::IDENTITY,
            paint: Paint::Solid(Color::WHITE),
        },
        Primitive::Glyphs {
            run: run(b"", 9),
            transform: kurbo::Affine::IDENTITY,
            paint: Paint::Solid(Color::WHITE),
        },
    ]);
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    let painted = pass
        .draw(&canvas.device, &canvas.queue, &list, &frame, canvas.size)
        .expect("the frame draws")
        .expect("the frame fits");
    canvas.renderer.texture(
        painted.texture,
        painted.src,
        painted.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let out = canvas.print();
    assert_eq!(out.at(50, 50), [255, 0, 0, 255], "the rect still paints");
    assert_eq!(
        out.at(150, 50),
        [CLEAR.r, CLEAR.g, CLEAR.b, CLEAR.a],
        "a run that names no face draws nothing"
    );
}

/// A pushed picture replaces a scene list.
///
/// A run may carry `--scene` and `--mcp` at once, and a push is where the two
/// meet: the push adopts its own source, which drops the list, so a tool call
/// takes the window over and nothing stale draws beside it.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_pushed_picture_replaces_a_scene_list() {
    let document = r#"{ "version": 1, "plane": { "width": 100.0, "height": 100.0 },
      "primitives": [ { "Rect": { "rect": { "x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0 },
                    "paint": { "Solid": { "r": 0, "g": 0, "b": 255, "a": 255 } } } } ] }"#;
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("scene.json");
    std::fs::write(&path, document).expect("write the scene");
    let launch = vv_view::viewer::Launch::scene(&path, true).expect("the scene opens");
    let (win_w, win_h) = launch.window_size();
    let canvas = canvas!(win_w, win_h);
    let mut viewer = vv_view::viewer::Viewer::new(
        canvas.device.clone(),
        canvas.queue.clone(),
        FORMAT,
        canvas.device.limits().max_texture_dimension_2d,
        launch,
        canvas.size,
    );
    let mut host = NullHost;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let mut blue = false;
    while !blue {
        assert!(
            std::time::Instant::now() < deadline,
            "the scene's list lands"
        );
        let pixels = shot_frame_read(&canvas, &mut viewer, &mut host);
        blue = viewer.listed() && pixels.at(512, 384) == [0, 0, 255, 255];
        std::thread::sleep(std::time::Duration::from_millis(4));
    }

    // A push of an SVG document, over the scene the launch opened.
    let square = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="#ff0000"/></svg>"##;
    let (reply, answers) = std::sync::mpsc::channel();
    viewer.accept(vv_view::push::Push::Bytes {
        data: square.as_bytes().to_vec(),
        reply,
    });
    answers
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the worker answers")
        .expect("the bytes decode");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let mut red = false;
    while !red {
        assert!(
            std::time::Instant::now() < deadline,
            "the pushed picture lands"
        );
        red = shot_frame_read(&canvas, &mut viewer, &mut host).at(512, 384) == [255, 0, 0, 255];
        std::thread::sleep(std::time::Duration::from_millis(4));
    }
    assert!(
        !viewer.listed(),
        "the list is gone, so no scene draws beside the picture"
    );
    assert_eq!(viewer.status().source, "mcp");
}

/// How many pixels of `frame` disagree with the raster `want`.
///
/// A vector file paints no paper of its own, so the raster is transparent where
/// the tree draws nothing and the frame shows the window color there.
/// Partial alpha is an antialiased edge and answers to neither painter.
fn misses_the_pixels(frame: &Pixels, want: &vv_source::DecodedImage, size: (u32, u32)) -> usize {
    let clear = [CLEAR.r, CLEAR.g, CLEAR.b, CLEAR.a];
    let mut off = 0;
    for y in 0..size.1 as usize {
        for x in 0..size.0 as usize {
            let at = (y * size.0 as usize + x) * 4;
            let painted = &want.as_rgba()[at..at + 4];
            let expected = match painted[3] {
                0 => clear,
                255 => [painted[0], painted[1], painted[2], painted[3]],
                _ => continue,
            };
            if frame
                .at(x, y)
                .iter()
                .zip(&expected)
                .any(|(a, b)| a.abs_diff(*b) > 24)
            {
                off += 1;
            }
        }
    }
    off
}

/// An SVG fills the pixel its rect names.
#[test]
#[ignore = "needs a Vulkan device"]
fn an_svg_paints_its_rect() {
    let mut canvas = canvas!(32, 32);
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("red.svg");
    std::fs::write(
        &path,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect width="16" height="16" fill="#ff0000"/></svg>"##,
    )
    .expect("write the svg");
    let dest = paint_surface(&mut canvas, &path);
    let pixels = canvas.print();
    assert_eq!(
        pixels.at(16, 16),
        [255, 0, 0, 255],
        "the SVG fills the frame"
    );
    assert_eq!(
        pixels.at(dest.min_x() as usize, dest.min_y() as usize),
        [0xff, 0, 0, 255],
        "the edge counts too"
    );
}

/// A converted source paints the paper its rule names, under its own paints.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_converted_source_paints_on_its_paper() {
    use std::os::unix::fs::PermissionsExt as _;

    let mut canvas = canvas!(64, 64);
    let dir = tempfile::tempdir().expect("a temp dir");
    // A stand-in converter: it swallows the source and answers a small red
    // square on a transparent page.
    let program = dir.path().join("convert.sh");
    std::fs::write(
        &program,
        "#!/bin/sh\ncat > /dev/null\nprintf '%s' '<svg xmlns=\"http://www.w3.org/2000/svg\" \
         width=\"100\" height=\"50\"><rect width=\"10\" height=\"10\" fill=\"#ff0000\"/></svg>'\n",
    )
    .expect("write the converter");
    let mut perms = std::fs::metadata(&program).expect("stat").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&program, perms).expect("chmod the converter");
    let path = dir.path().join("source.mmd");
    std::fs::write(&path, "a source the viewer does not parse").expect("write the source");
    // The rule names white paper, so every pixel the page leaves clear shows
    // white, and the square keeps its own red.
    let config = format!(
        "[[convert]]\next = [\"mmd\"]\ncommand = [\"{}\"]\nbackground = \"#ffffff\"\n",
        program.display()
    );
    let table = vv_source::convert::Converters::parse(&config)
        .expect("the rule table")
        .with_cache_dir(dir.path());
    let surface = table.open(&path).expect("the converted surface");
    let dest = paint_plane(&mut canvas, surface.as_ref());
    let pixels = canvas.print();
    let (x0, y0) = (dest.min_x() as usize, dest.min_y() as usize);
    let (x1, y1) = (dest.max_x() as usize, dest.max_y() as usize);
    assert_eq!(
        pixels.at(x1 - 2, y1 - 2),
        [255, 255, 255, 255],
        "the rule paints white paper"
    );
    let inked = (y0..y1).any(|y| (x0..x1).any(|x| pixels.at(x, y) == [255, 0, 0, 255]));
    assert!(inked, "the square paints over the paper");
}

/// A book plane leaves its gutter transparent, so the window shows through.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_book_plane_keeps_its_gutter_transparent() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("book.pdf");
    // Three white pages, so the book holds a cover row and one spread.
    std::fs::write(&path, page_pdf(3, [255, 255, 255])).expect("write the pdf");
    let surface = vv_source::surface::open_book_surface(&path).expect("a book");
    assert_eq!(surface.plane_count(), 1, "a book is one plane");
    let image = surface
        .render(View::whole(0, 1.0))
        .expect("the book plane renders");
    let (width, height) = (image.width().raw() as usize, image.height().raw() as usize);
    // The gutter is a transparent column between the two page columns.
    let middle = width / 2;
    let column_clear = |x: usize| {
        (0..height).all(|y| {
            let start = (y * width + x) * 4;
            image.as_rgba()[start + 3] == 0
        })
    };
    let gutter = (middle.saturating_sub(4)..middle + 4).find(|&x| column_clear(x));
    assert!(
        gutter.is_some(),
        "a transparent gutter near the middle of a {width}x{height} book"
    );
}

/// A listed book paints every page in its own slot.
///
/// A book is one plane of spreads, and each page of it is a list of its own, so
/// the book's list is those lists placed in the slots the book measured. A page
/// that landed in points instead of base units would draw at the top left of the
/// plane, and the slots of the two spreads would show the same page.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_listed_book_places_its_pages() {
    use vv_core::{
        dim::{SurfacePoint, SurfaceSize, WindowSize, WindowVector},
        region,
        scene::Primitive,
    };

    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("book.pdf");
    // Two pages: the cover is red and the page after it is blue.
    let red = "1 0 0 rg\n0 0 64 64 re f\n";
    let blue = "0 0 1 rg\n0 0 64 64 re f\n";
    std::fs::write(&path, pdf_with(2, red, blue)).expect("write the pdf");
    let surface = vv_source::surface::open_book_surface(&path).expect("a book");
    let list = surface
        .display(0)
        .expect("a list")
        .expect("a book of two pages lists");
    assert!(
        list.iter().any(|p| matches!(p, Primitive::Path { .. })),
        "the pages ride the list"
    );

    let mut canvas = canvas!(256, 256);
    canvas.begin(CLEAR);
    let plane = surface.bounds(0).expect("the book measures");
    let frame = region::Frame {
        img: SurfaceSize::new(plane.w, plane.h),
        win: WindowSize::new(256.0, 256.0),
        pan: WindowVector::zero(),
        scale: (256.0 / plane.w).min(256.0 / plane.h),
        angle: 0.0,
    };
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    let painted = pass
        .draw(&canvas.device, &canvas.queue, &list, &frame, canvas.size)
        .expect("the list draws")
        .expect("the list fits the frame");
    canvas.renderer.texture(
        painted.texture,
        painted.src,
        painted.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let pixels = canvas.print();

    for (page, want) in [(0usize, [255, 0, 0, 255]), (1usize, [0, 0, 255, 255])] {
        let slot = surface.page_box(page).expect("the page has a slot");
        let center = frame.to_screen(SurfacePoint::new(
            slot.min_x() + slot.width() / 2.0,
            slot.min_y() + slot.height() / 2.0,
        ));
        let (x, y) = (center.x.round() as usize, center.y.round() as usize);
        assert_eq!(pixels.at(x, y), want, "page {page} in its own slot");
    }
}

/// A book that bakes a base scale below one still places every page.
///
/// The book's plane is in base units and each page's list is in points, so the
/// placement converts the page's own units and only then moves it to its slot.
/// A placement that scales the slot as well shrinks every page's offset by
/// `units_per_point`, and the pages then pile up near the first spread. A short
/// book bakes that scale to exactly 1.0, where both readings agree, so this one
/// stands taller than [`vv_source::MAX_TEXTURE_SIDE`] and every page carries its
/// own color: a page whose color is missing from its own slot is a page that
/// landed somewhere else.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_listed_book_places_its_pages_below_one_base_unit() {
    use std::sync::Arc;

    use vv_core::{
        dim::{SurfacePoint, SurfaceSize, WindowSize, WindowVector},
        region,
    };
    use vv_source::{book::BookSurface, surface::Surface};

    // 64 tall pages: the book is far taller than one texture, so the base scale
    // lands below one.
    let page_color = |page: usize| 2 * (page + 1);
    let document = pdf_with_box(64, [0, 0, 64, 400], |page| {
        format!(
            "{} 0 0 rg\n0 0 64 400 re f\n",
            f64::from(page_color(page) as u8) / 255.0
        )
    });
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("tall.pdf");
    std::fs::write(&path, document).expect("write the pdf");
    let inner = vv_source::pdf::PdfSurface::open(&path).expect("the document opens");
    let surface = BookSurface::new(Arc::new(inner)).expect("a book");
    assert!(
        surface.units_per_point() < 0.9,
        "the fixture bakes a base scale below one: {}",
        surface.units_per_point()
    );
    let list = surface.display(0).expect("a list").expect("the book lists");
    assert!(!list.is_empty(), "the pages ride the list");

    let (win_w, win_h) = (512.0f32, 512.0f32);
    let mut canvas = canvas!(512, 512);
    canvas.begin(CLEAR);
    let plane = surface.bounds(0).expect("the book measures");
    let frame = region::Frame {
        img: SurfaceSize::new(plane.w, plane.h),
        win: WindowSize::new(win_w, win_h),
        pan: WindowVector::zero(),
        scale: (win_w / plane.w).min(win_h / plane.h),
        angle: 0.0,
    };
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    let painted = pass
        .draw(&canvas.device, &canvas.queue, &list, &frame, canvas.size)
        .expect("the list draws")
        .expect("the list fits the frame");
    canvas.renderer.texture(
        painted.texture,
        painted.src,
        painted.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let pixels = canvas.print();

    for page in 0..surface.inner().plane_count() {
        let slot = surface.page_box(page).expect("the page has a slot");
        let center = frame.to_screen(SurfacePoint::new(
            slot.min_x() + slot.width() / 2.0,
            slot.min_y() + slot.height() / 2.0,
        ));
        let (x, y) = (center.x.round() as usize, center.y.round() as usize);
        let [r, _, _, a] = pixels.at(x, y);
        let want = page_color(page);
        assert!(
            a == 255 && r.abs_diff(want as u8) <= 4,
            "page {page} is {want} in its own slot, and the frame holds {:?} at ({x}, {y})",
            pixels.at(x, y)
        );
    }
}

/// A 200 by 100 page with a red square and a two by one picture in a box.
///
/// The picture is the leaf a list has to place in the box the page names, so a
/// transform that also carries the picture's own pixel count throws it far past
/// the page, where it smears over everything else the page draws.
fn pictured_page() -> Vec<u8> {
    let content = "q 1 0 0 rg 20 10 60 30 re f Q\nq 80 0 0 60 100 20 cm /Im1 Do Q\n";
    let mut picture = b"5 0 obj << /Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace \
        /DeviceRGB /BitsPerComponent 8 /Length 6 >> stream\n"
        .to_vec();
    // Two pixels: red then green, so each half of the box is flat and one
    // misplaced transform cannot accidentally sample the other half.
    picture.extend_from_slice(&[0xff, 0x00, 0x00, 0x00, 0xff, 0x00]);
    picture.extend_from_slice(b"\nendstream\nendobj\n");
    let objects = vec![
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n".to_vec(),
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n".to_vec(),
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R \
          /Resources << /XObject << /Im1 5 0 R >> >> >> endobj\n"
            .to_vec(),
        format!(
            "4 0 obj << /Length {} >> stream\n{content}endstream\nendobj\n",
            content.len()
        )
        .into_bytes(),
        picture,
    ];
    pdf_bytes(&objects)
}

/// A listed page paints what the raster path paints, picture included.
///
/// `hayro` drives both sides of this comparison — one into its own rasterizer
/// and one into the display list — so the two have to agree wherever either one
/// paints. A tolerance and not a hash: the two rasterizers anti-alias their own
/// way, and a stored sum would break on a rounding change in either of them,
/// while a leaf placed in the wrong box breaks the comparison everywhere. The
/// probes pin the interesting places: the square, both halves of the picture,
/// and the paper between them.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_listed_page_paints_what_the_raster_path_paints() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("pictured.pdf");
    std::fs::write(&path, pictured_page()).expect("write the pdf");
    let surface = open_surface(&path).expect("the page opens");
    let plane = surface.bounds(0).expect("the page measures");
    assert_eq!((plane.w, plane.h), (200.0, 100.0));
    let raster = surface.render_whole(0, 1.0).expect("the raster renders");
    assert_eq!(
        (raster.width().raw(), raster.height().raw()),
        (200, 100),
        "one point is one pixel at this scale"
    );
    let list = surface.display(0).expect("a list").expect("the page lists");

    let mut canvas = canvas!(200, 100);
    let pixels = paint_list_flat(&mut canvas, &list);

    // The page draws its square at 20..80 by 60..90 of the render space (y
    // down), and its picture at 100..180 by 20..80, red then green.
    let probes = [
        (50, 75, "the red square"),
        (120, 50, "the red half of the picture"),
        (160, 50, "the green half of the picture"),
        (10, 5, "the paper"),
    ];
    assert_probes(&pixels, &raster, &probes);
    // And no corner of the page disagrees by more than an edge does.
    let mean = mean_difference(&pixels, &raster, (200, 100));
    assert!(
        mean <= 8,
        "the list and the raster paint the same page: mean difference {mean}"
    );
}

/// Paint a list into `canvas` at one output pixel per unit, and read it back.
///
/// A plane under a frame of scale 1.0 and no pan puts one unit on one pixel,
/// which is the scale `render_whole` draws the raster path at, so a plane whose
/// size is the canvas size lands on it whole.
fn paint_list_flat(
    canvas: &mut Canvas,
    list: &std::sync::Arc<[vv_source::scene::Primitive]>,
) -> Pixels {
    use vv_core::{
        dim::{SurfaceSize, WindowSize, WindowVector},
        region,
    };

    let frame = region::Frame {
        img: SurfaceSize::new(canvas.size.0 as f32, canvas.size.1 as f32),
        win: WindowSize::new(canvas.size.0 as f32, canvas.size.1 as f32),
        pan: WindowVector::zero(),
        scale: 1.0,
        angle: 0.0,
    };
    paint_list_at(canvas, list, frame)
}

/// Paint a list into `canvas` under `frame`, and read it back.
fn paint_list_at(
    canvas: &mut Canvas,
    list: &std::sync::Arc<[vv_source::scene::Primitive]>,
    frame: vv_core::region::Frame,
) -> Pixels {
    canvas.begin(CLEAR);
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    let painted = pass
        .draw(&canvas.device, &canvas.queue, list, &frame, canvas.size)
        .expect("the list draws")
        .expect("the list fits the frame");
    canvas.renderer.texture(
        painted.texture,
        painted.src,
        painted.quad,
        Color::WHITE,
        Filter::Linear,
    );
    canvas.print()
}

/// Assert that a painted frame agrees with a rendered picture at `probes`.
///
/// A test of two painters asserts a tolerance and never an equality: vello and
/// `hayro` anti-alias their own way, so an edge is a few shades apart even where
/// both agree about the geometry.
fn assert_probes(
    pixels: &Pixels,
    raster: &vv_source::DecodedImage,
    probes: &[(usize, usize, &str)],
) {
    for &(x, y, what) in probes {
        let got = pixels.at(x, y);
        let want = raster_pixel(raster, x, y);
        for channel in 0..4 {
            assert!(
                got[channel].abs_diff(want[channel]) <= 24,
                "{what} at ({x}, {y}): the list paints {got:?}, the raster {want:?}"
            );
        }
    }
}

/// The mean difference per colour channel between a painted frame and a
/// rendered picture of the same size.
fn mean_difference(pixels: &Pixels, raster: &vv_source::DecodedImage, size: (usize, usize)) -> u64 {
    let mut sum = 0u64;
    for y in 0..size.1 {
        for x in 0..size.0 {
            let (got, want) = (pixels.at(x, y), raster_pixel(raster, x, y));
            sum += (0..3)
                .map(|channel| u64::from(got[channel].abs_diff(want[channel])))
                .sum::<u64>();
        }
    }
    sum / (size.0 as u64 * size.1 as u64 * 3)
}

/// The pixel of a rendered picture, as RGBA8.
fn raster_pixel(image: &vv_source::DecodedImage, x: usize, y: usize) -> [u8; 4] {
    let width = image.width().raw() as usize;
    let start = (y * width + x) * 4;
    let rgba = image.as_rgba();
    [
        rgba[start],
        rgba[start + 1],
        rgba[start + 2],
        rgba[start + 3],
    ]
}

/// A 200 by 160 page painted by three shadings: a ramp down the left side, a
/// pair of circles in the right half, and the same circles squeezed into an
/// ellipse below them.
///
/// A shading paints a function rather than a colour, so it is the case a list
/// can only carry as a ramp — and a ramp placed wrong smears the whole page,
/// while a radial drawn round where the page names an ellipse paints one.
fn shaded_page() -> Vec<u8> {
    let content = "/Pattern cs /P1 scn 0 0 100 160 re f\n/Pattern cs /P2 scn 100 80 100 80 re \
                   f\n/Pattern cs /P3 scn 100 0 100 80 re f\n";
    let objects = vec![
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n".to_vec(),
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n".to_vec(),
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 160] /Contents 4 0 R \
          /Resources << /Pattern << /P1 5 0 R /P2 8 0 R /P3 11 0 R >> >> >> endobj\n"
            .to_vec(),
        format!(
            "4 0 obj << /Length {} >> stream\n{content}endstream\nendobj\n",
            content.len()
        )
        .into_bytes(),
        // The left column: red at the left edge, blue at the middle of the page.
        b"5 0 obj << /Type /Pattern /PatternType 2 /Shading 6 0 R >> endobj\n".to_vec(),
        b"6 0 obj << /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 100 0] \
          /Function 7 0 R /Extend [true true] >> endobj\n"
            .to_vec(),
        b"7 0 obj << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> endobj\n"
            .to_vec(),
        // The circles: a green centre and a white rim, in the unit circle scaled
        // by forty and moved into the top right quarter.
        b"8 0 obj << /Type /Pattern /PatternType 2 /Shading 9 0 R \
          /Matrix [40 0 0 40 150 120] >> endobj\n"
            .to_vec(),
        b"9 0 obj << /ShadingType 3 /ColorSpace /DeviceRGB /Coords [0 0 0 0 0 1] \
          /Function 10 0 R /Extend [true true] >> endobj\n"
            .to_vec(),
        b"10 0 obj << /FunctionType 2 /Domain [0 1] /C0 [0 0.6 0] /C1 [1 1 1] /N 1 >> endobj\n"
            .to_vec(),
        // The same circles under a matrix that squeezes the y axis by half: an
        // ellipse forty points across and twenty down, below the first one.
        b"11 0 obj << /Type /Pattern /PatternType 2 /Shading 9 0 R \
          /Matrix [40 0 0 20 150 40] >> endobj\n"
            .to_vec(),
    ];
    pdf_bytes(&objects)
}

/// A listed page paints the ramps its shadings name, where the raster does.
///
/// A shading is a function sampled at every pixel, and the list carries it as a
/// ramp the renderer interpolates between stops, so the two cannot agree to the
/// byte — but they land in the same place, with the same colours at the same
/// points. A ramp anchored in the wrong space smears the page and fails this.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_listed_page_paints_its_shadings() {
    use vv_core::scene::{Paint, Primitive};

    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("shaded.pdf");
    std::fs::write(&path, shaded_page()).expect("write the pdf");
    let surface = open_surface(&path).expect("the page opens");
    let plane = surface.bounds(0).expect("the page measures");
    assert_eq!((plane.w, plane.h), (200.0, 160.0));
    let raster = surface.render_whole(0, 1.0).expect("the raster renders");
    let list = surface.display(0).expect("a list").expect("the page lists");
    let fills = list
        .iter()
        .filter_map(|primitive| match primitive {
            Primitive::Path {
                fill: Some(Paint::Linear { .. } | Paint::Radial { .. }),
                ..
            } => Some(()),
            // Every other leaf carries a solid paint or no fill at all.
            Primitive::Path { .. }
            | Primitive::Rect { .. }
            | Primitive::Image { .. }
            | Primitive::Glyphs { .. }
            | Primitive::PushClip { .. }
            | Primitive::PopClip
            | Primitive::PushLayer { .. }
            | Primitive::PopLayer => None,
        })
        .count();
    assert_eq!(fills, 3, "every shading rides the list as a ramp: {list:?}");

    let mut canvas = canvas!(200, 160);
    let pixels = paint_list_flat(&mut canvas, &list);

    // The ramp runs red at the left edge to blue at the middle of the page, the
    // circles go from green at their centre to white at their rim, and the
    // ellipse is the same ramp in a shape of its own. The page's y axis points
    // up, so the circles of the page land in the upper half of the frame.
    let probes = [
        (5, 80, "the red end of the ramp"),
        (99, 80, "the blue end of the ramp"),
        (50, 80, "the middle of the ramp"),
        (150, 40, "the green centre of the circles"),
        (185, 40, "the ramp between the circle and its rim"),
        (150, 120, "the green centre of the ellipse"),
        // Forty points to the right of the ellipse's centre, which is inside an
        // ellipse forty across and outside a circle twenty across: a ramp drawn
        // round where the page draws an ellipse disagrees here.
        (190, 120, "the wide end of the ellipse"),
        (150, 132, "the narrow top of the ellipse"),
    ];
    assert_probes(&pixels, &raster, &probes);
    // And the ramps paint what the shading names, so a pair of blank shapes
    // cannot agree with the raster and pass.
    let [red, _, blue, _] = pixels.at(5, 80);
    assert!(
        red > 200 && blue < 60,
        "the ramp starts red at the left edge: {:?}",
        pixels.at(5, 80)
    );
    let [red, _, blue, _] = pixels.at(99, 80);
    assert!(
        blue > 200 && red < 60,
        "and ends blue at the middle: {:?}",
        pixels.at(99, 80)
    );
    for (x, y, what) in [(150, 40, "circles"), (150, 120, "ellipse")] {
        let [red, green, blue, _] = pixels.at(x, y);
        assert!(
            green > 120 && red < 60 && blue < 60,
            "the {what} start green at their centre: {:?}",
            pixels.at(x, y)
        );
    }
    // The ellipse reaches forty points across its centre and only twenty above
    // it, so two probes tell it from a circle: twenty points out sideways is
    // still inside the ramp, and twenty-five points up is past its rim.
    let wide = pixels.at(170, 120);
    assert!(
        wide[0] > 60 && wide[0] < 220 && wide[1] > wide[0],
        "the ellipse is still painted twenty points out: {wide:?}"
    );
    let narrow = pixels.at(150, 145);
    assert_eq!(
        narrow,
        [255, 255, 255, 255],
        "and has ended above that: {narrow:?}"
    );
    let mean = mean_difference(&pixels, &raster, (200, 160));
    assert!(
        mean <= 8,
        "the list and the raster paint the same shading: mean difference {mean}"
    );
}

/// Build a PDF of `pages` pages, each filled with `color`.
///
/// The xref offsets are computed here, so the file is valid and needs no
/// fixture on disk.
fn page_pdf(pages: usize, color: [u8; 3]) -> Vec<u8> {
    let fill = format!(
        "{} {} {} rg\n0 0 64 64 re f\n",
        f64::from(color[0]) / 255.0,
        f64::from(color[1]) / 255.0,
        f64::from(color[2]) / 255.0
    );
    pdf_with(pages, &fill, &fill)
}

/// Build a PDF of `pages` 64x64 pages: the first one draws `first`, and every
/// other one draws `rest`.
fn pdf_with(pages: usize, first: &str, rest: &str) -> Vec<u8> {
    pdf_with_box(pages, [0, 0, 64, 64], |page| {
        if page == 0 { first } else { rest }.to_owned()
    })
}

/// Build a PDF of `pages` pages of `media_box`, each drawing its own content.
///
/// The xref offsets are computed here, so the file is valid and needs no
/// fixture on disk.
fn pdf_with_box(pages: usize, media_box: [u32; 4], content: impl Fn(usize) -> String) -> Vec<u8> {
    let [x0, y0, x1, y1] = media_box;
    let kids = (0..pages)
        .map(|page| format!("{} 0 R", 3 + 2 * page))
        .collect::<Vec<_>>()
        .join(" ");
    let mut objects = vec![
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n".to_vec(),
        format!("2 0 obj << /Type /Pages /Kids [{kids}] /Count {pages} >> endobj\n").into_bytes(),
    ];
    for page in 0..pages {
        let (body, stream) = (3 + 2 * page, 4 + 2 * page);
        objects.push(
            format!(
                "{body} 0 obj << /Type /Page /Parent 2 0 R /MediaBox [{x0} {y0} {x1} {y1}] \
                 /Contents {stream} 0 R /Resources << >> >> endobj\n"
            )
            .into_bytes(),
        );
        let content = content(page);
        objects.push(
            format!(
                "{stream} 0 obj << /Length {} >> stream\n{content}endstream\nendobj\n",
                content.len()
            )
            .into_bytes(),
        );
    }
    pdf_bytes(&objects)
}

/// Assemble `objects`, numbered from one, into a PDF.
///
/// The xref offsets are computed here, so the file is valid and needs no
/// fixture on disk. A picture's stream holds bytes rather than text, so an
/// object is a byte string and a caller that formats one pushes the encoder's
/// answer through [`String::into_bytes`].
fn pdf_bytes(objects: &[Vec<u8>]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for object in objects {
        offsets.push(pdf.len());
        pdf.extend_from_slice(object);
    }
    let xref = pdf.len();
    let mut tail = format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
    for offset in offsets {
        let _ = writeln!(tail, "{offset:010} 00000 n ");
    }
    let _ = write!(
        tail,
        "trailer << /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
        objects.len() + 1
    );
    pdf.extend_from_slice(tail.as_bytes());
    pdf
}

/// A book of twenty pages: a red cover, and nineteen pages of wavy filled
/// paths.
///
/// The curves are the work a flatten pass pays for, and each spans the whole
/// page, so one encode of the whole book passes `vello`'s fixed allocators once
/// the frame zooms in: the curves of the pages the window does not show are
/// flattened all the same.
fn dense_book() -> Vec<u8> {
    pdf_with_box(20, [0, 0, 200, 200], |page| {
        if page == 0 {
            return "1 0 0 rg\n0 0 200 200 re f\n".to_owned();
        }
        let mut out = String::from("0 0 1 rg\n");
        for wave in 0..50 {
            let y = 2.0 + wave as f32 * 4.0;
            let _ = write!(out, "0 {y} m");
            for step in 0..40 {
                let x = step as f32 * 5.0;
                let _ = write!(
                    out,
                    " {a} {b} {c} {d} {e} {f} c",
                    a = x + 2.0,
                    b = y + 60.0,
                    c = x + 3.0,
                    d = y - 60.0,
                    e = x + 5.0,
                    f = y,
                );
            }
            let _ = writeln!(out, " h f");
        }
        out
    })
}

/// A page whose whole drawing is one path of twenty thousand curves.
///
/// A band keeps a primitive whole, and `vello` flattens it whole, so a path like
/// this one starves the renderer's fixed allocators once the frame zooms in —
/// and a frame it starves paints nothing at all, without an error.
fn heavy_page() -> Vec<u8> {
    pdf_with_box(1, [0, 0, 200, 200], |_| {
        let mut out = String::from("1 0 0 rg\n0 0 200 200 re f\n0 0 1 rg\n0 100 m");
        for step in 0..20_000 {
            let x = (step % 40) as f32 * 5.0;
            let _ = write!(
                out,
                " {a} {b} {c} {d} {e} {f} c",
                a = x + 2.0,
                b = 160.0,
                c = x + 3.0,
                d = 40.0,
                e = x + 5.0,
                f = 100.0,
            );
        }
        let _ = writeln!(out, " h f");
        out
    })
}

/// A frame past `vello`'s flattening limit is refused instead of painted
/// blank.
///
/// `vello` abandons a frame whose intermediate buffers run out, writes nothing
/// to the target, and reports nothing (`Renderer::render_to_texture` is its
/// non-robust path), so the pass has to know before it paints: it estimates the
/// lines the frame would flatten (`BumpEstimator`) and answers `None`, which
/// makes the view draw the plane texture and the region patch — the picture the
/// source rasterizes at the window's own scale, with no such limit.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_frame_past_the_flatten_limit_is_refused() {
    use vv_core::{
        dim::{SurfaceSize, WindowSize, WindowVector},
        region,
    };

    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("heavy.pdf");
    std::fs::write(&path, heavy_page()).expect("write the pdf");
    let surface = open_surface(&path).expect("the page opens");
    let plane = surface.bounds(0).expect("the page measures");
    let list = surface.display(0).expect("a list").expect("the page lists");
    let frame = |scale: f32| region::Frame {
        img: SurfaceSize::new(plane.w, plane.h),
        win: WindowSize::new(256.0, 256.0),
        pan: WindowVector::zero(),
        scale,
        angle: 0.0,
    };

    let mut canvas = canvas!(256, 256);
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");
    assert!(
        pass.draw(
            &canvas.device,
            &canvas.queue,
            &list,
            &frame(400.0),
            canvas.size,
        )
        .expect("the pass answers")
        .is_none(),
        "a frame that cannot fit the renderer is refused, not painted blank"
    );

    // The same page at one to one is a frame the pass paints.
    let near = pass
        .draw(
            &canvas.device,
            &canvas.queue,
            &list,
            &frame(1.0),
            canvas.size,
        )
        .expect("the pass answers")
        .expect("the near frame fits the renderer");
    canvas.begin(CLEAR);
    canvas.renderer.texture(
        near.texture,
        near.src,
        near.quad,
        Color::WHITE,
        Filter::Linear,
    );
    let pixels = canvas.print();
    assert_eq!(
        pixels.at(128, 128),
        [255, 0, 0, 255],
        "the paper of the page, from the list"
    );
}

/// A book zoomed in past the point where its whole list fits `vello`'s
/// allocators still paints the page the window shows.
///
/// `vello` flattens every encoded curve and its bump allocators hold a fixed
/// number of lines, picked for a bench scene, so a book — one plane holding
/// every page, so one list — paints *nothing at all* past a few zoom steps, and
/// reports no error while it does. The pass keeps only the band the window
/// reaches (see `vv_gpu::display`), so the cover paints here however many pages
/// the book hides beside it.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_zoomed_book_paints_the_page_it_shows() {
    use vv_core::{
        dim::{SurfaceSize, WindowSize, WindowVector},
        region,
        scene::Primitive,
    };

    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("dense-book.pdf");
    std::fs::write(&path, dense_book()).expect("write the pdf");
    let surface = vv_source::surface::open_book_surface(&path).expect("a book");
    let list = surface.display(0).expect("a list").expect("the book lists");
    let plane = surface.bounds(0).expect("the book measures");
    // A page's paper comes first in its own list, and a book places every page,
    // so the first primitive of the book is the cover's paper: a rect of the page
    // carried as a path, because a rect states no transform of its own.
    let Primitive::Path { transform, .. } = &list[0] else {
        panic!("the book's first primitive is the cover's paper: {list:?}");
    };
    let [xx, yx, xy, yy, tx, ty] = transform.as_coeffs();
    // The page is 200 by 200, so this is where its middle lands in the plane.
    let (center_x, center_y) = (xx * 100.0 + xy * 100.0 + tx, yx * 100.0 + yy * 100.0 + ty);

    let scale = 64.0;
    let frame = region::Frame {
        img: SurfaceSize::new(plane.w, plane.h),
        win: WindowSize::new(256.0, 256.0),
        // The cover sits in its own slot of the book, so the pan is what puts it
        // under the window: the window middle lands on a surface point when the
        // pan carries that point away from the plane's own middle.
        pan: WindowVector::new(
            ((f64::from(plane.w) / 2.0 - center_x) * f64::from(scale)) as f32,
            ((f64::from(plane.h) / 2.0 - center_y) * f64::from(scale)) as f32,
        ),
        scale,
        angle: 0.0,
    };
    let mut canvas = canvas!(256, 256);
    let pixels = paint_list_at(&mut canvas, &list, frame);
    assert_eq!(
        pixels.at(128, 128),
        [255, 0, 0, 255],
        "the cover paints at a zoom the whole book does not fit"
    );
}

/// A PDF page paints its own paper under the drawing.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_pdf_page_paints_its_content() {
    let mut canvas = canvas!(32, 32);
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("page.pdf");
    std::fs::write(&path, page_pdf(1, [0, 128, 0])).expect("write the pdf");
    let dest = paint_surface(&mut canvas, &path);
    let pixels = canvas.print();
    let middle = (
        (dest.min_x() + dest.width() / 2.0) as usize,
        (dest.min_y() + dest.height() / 2.0) as usize,
    );
    let [r, g, b, a] = pixels.at(middle.0, middle.1);
    assert!(g > 100 && r < 80 && b < 80, "the page draws its green fill");
    assert_eq!(a, 255, "the page covers the window");
}

/// A recolored PDF page maps its black onto the ink and its paper onto the
/// theme.
///
/// The raster path clears the page to white paper and draws the content on it,
/// so the two ends of a page are exactly the two ends of the map, and this is
/// the paint the `Shift+T` toggle shows.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_recolored_pdf_page_maps_its_black_and_its_paper() {
    let mut canvas = canvas!(32, 32);
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("half.pdf");
    // A 64-wide page whose left half is a black rect, on white paper.
    std::fs::write(&path, pdf_with(1, "0 0 0 rg\n0 0 32 64 re f\n", "")).expect("write the pdf");
    canvas.renderer.set_theme(&canvas.queue, Some(THEME));
    let dest = paint_surface(&mut canvas, &path);
    let pixels = canvas.print();
    let y = (dest.min_y() + dest.height() / 2.0) as usize;
    let black = ((dest.min_x() + dest.width() * 0.25) as usize, y);
    let paper = ((dest.min_x() + dest.width() * 0.75) as usize, y);
    assert!(
        close(pixels.at(black.0, black.1), THEME.ink.to_bytes(), 2),
        "the black half is the ink: {:?}",
        pixels.at(black.0, black.1)
    );
    assert!(
        close(pixels.at(paper.0, paper.1), THEME.paper.to_bytes(), 2),
        "the white half is the paper: {:?}",
        pixels.at(paper.0, paper.1)
    );
}

/// A patch lands on its texels, and a fractional position blends them.
///
/// The viewer's `patch_dest` snaps a patch onto whole window pixels for exactly
/// this reason: a bilinear sample halfway between two texels softens the text.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_fractional_destination_blends_the_texels() {
    let mut canvas = canvas!(4, 4);
    // Left column black, right column white.
    let rgba = [
        0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255,
    ];
    let texture = canvas.renderer.upload_rgba(&rgba, 2, 2).expect("upload");
    // Two pixels wide on whole pixels: the two texels stay apart.
    canvas.begin(Color::BLACK);
    canvas.renderer.texture_pro(
        &texture,
        dim::whole(2.0, 2.0),
        dim::rect(0.0, 0.0, 2.0, 2.0),
        Turn::NONE,
        Color::WHITE,
        Filter::Linear,
    );
    let exact = canvas.print();
    assert_eq!(exact.at(0, 0), [0, 0, 0, 255], "the black texel");
    assert_eq!(exact.at(1, 0), [255, 255, 255, 255], "the white texel");
    // The same patch half a pixel to the right: every sample sits between the
    // two texels.
    canvas.begin(Color::BLACK);
    canvas.renderer.texture_pro(
        &texture,
        dim::whole(2.0, 2.0),
        dim::rect(0.5, 0.0, 2.0, 2.0),
        Turn::NONE,
        Color::WHITE,
        Filter::Linear,
    );
    let offset = canvas.print();
    // The first pixel samples the black texel, because the quad starts half a
    // pixel in. The second sits between the two texels.
    assert_eq!(offset.at(0, 0), [0, 0, 0, 255], "still the black texel");
    let blended = offset.at(1, 0)[0];
    assert!(
        (1..255).contains(&blended),
        "a half-pixel offset blends the texels, got {blended}"
    );
}

/// A decode worker uploads its own frame, and the main thread only draws it.
///
/// This is the funnel the port deleted: the streaming loader holds an
/// [`vv_gpu::texture::Uploader`], so `DecodeMsg::Done` carries a
/// texture and no pixel copy runs on the frame thread.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_loader_worker_uploads_its_own_frame() {
    use vv_core::dim::PreviewPx;
    use vv_gpu::texture::Ready;
    use vv_view::loader::Loader;

    let mut canvas = canvas!(32, 32);
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("page.pdf");
    std::fs::write(&path, page_pdf(1, [200, 200, 0])).expect("write the pdf");
    let surface = open_surface(&path).expect("the page opens");
    let uploader = Some(canvas.renderer.uploader());
    let loader = Loader::start(surface, PreviewPx::new(32), &uploader);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let frame = loop {
        assert!(
            std::time::Instant::now() < deadline,
            "the loader answers in time"
        );
        match loader.try_recv() {
            Some(Ok(vv_view::loader::DecodeMsg::Done { frame, .. })) => break frame,
            Some(Ok(_)) => {}
            Some(Err(err)) => panic!("the decode failed: {err}"),
            None => std::thread::sleep(std::time::Duration::from_millis(10)),
        }
    };
    assert!(
        matches!(frame, Ready::Texture(_)),
        "a worker with a device uploads its frame"
    );
    assert_eq!(frame.dims(), (64, 64), "the page renders at its own size");
    // The texture the worker made draws like any other.
    let texture = frame.into_texture().expect("an uploaded frame");
    canvas.begin(Color::BLACK);
    canvas.renderer.texture_pro(
        &texture,
        dim::whole(64.0, 64.0),
        dim::rect(0.0, 0.0, 32.0, 32.0),
        Turn::NONE,
        Color::WHITE,
        Filter::Linear,
    );
    let pixels = canvas.print();
    let [r, g, b, _] = pixels.at(16, 16);
    assert!(
        r > 150 && g > 150 && b < 80,
        "the worker's pixels reach the frame"
    );
}

/// The region veil dims the picture outside the selection and nothing else.
///
/// The four bands come from `region_sel::around` and map through
/// `region::Frame::to_screen`, which is the same transform the picture draws
/// with, so the veil stays on the picture through a pan, a zoom and a turn.
#[test]
#[ignore = "needs a Vulkan device"]
fn the_region_veil_dims_only_outside_the_selection() {
    use vv_core::{DIM_LEVEL, DIM_VEIL, draw::Vec2, region, region_sel};

    let mut canvas = canvas!(64, 64);
    let selection = region::crop(16.0, 16.0, 32.0, 32.0);
    for angle in [0.0_f32, 30.0] {
        let frame = region::Frame {
            img: dim::SurfaceSize::new(64.0, 64.0),
            win: dim::WindowSize::new(64.0, 64.0),
            pan: dim::WindowVector::zero(),
            scale: 1.0,
            angle,
        };
        canvas.begin(Color::BLACK);
        // The picture: a white plane, turned like the draw turns it.
        let dest = dim::rect(32.0, 32.0, 64.0, 64.0);
        canvas.renderer.fill(
            Quad::pro(dest, Turn::new(dim::center_origin(dest), angle)),
            Color::WHITE,
        );
        let outer = frame.plane();
        for band in region_sel::around(outer, selection) {
            let corners = band.map(|point| dim::draw_window(frame.to_screen(point)));
            canvas.renderer.fill(Quad::new(corners), DIM_VEIL);
        }
        let pixels = canvas.print();
        let at = |point: Vec2| {
            let screen = frame.to_screen(dim::surface_point(point));
            pixels.at(screen.x as usize, screen.y as usize)
        };
        // The picture center sits inside the selection at any turn, so it keeps
        // every bit of its brightness.
        assert_eq!(
            at(Vec2::new(32.0, 32.0)),
            [255, 255, 255, 255],
            "angle {angle}: the selection stays bright"
        );
        // A picture point outside the selection takes the veil: a black veil at
        // `255 - DIM_LEVEL` alpha over white answers exactly `DIM_LEVEL`.
        let dimmed = at(Vec2::new(4.0, 4.0))[0];
        assert!(
            dimmed.abs_diff(DIM_LEVEL) <= 2,
            "angle {angle}: outside the selection reads {dimmed}, not the dim level"
        );
    }
}

/// A rectangle outline covers the outer band of its rect, nothing outside.
///
/// This is raylib's `DrawRectangleLinesEx` placement, which the grid cursor ring
/// and its visual-selection ring keep: a centered ring would spill into the gap
/// between two cells.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_stroke_rect_stays_inside_its_rect() {
    let mut canvas = canvas!(32, 32);
    canvas.begin(Color::BLACK);
    let rect = dim::rect(8.0, 8.0, 16.0, 16.0);
    canvas.renderer.stroke_rect(rect, 4.0, Color::WHITE);
    let pixels = canvas.print();
    // The band: the four outer pixels of the rect on every side.
    assert_eq!(pixels.at(8, 8), [255, 255, 255, 255], "the top-left corner");
    assert_eq!(pixels.at(11, 8), [255, 255, 255, 255], "the top band");
    assert_eq!(pixels.at(8, 20), [255, 255, 255, 255], "the bottom band");
    assert_eq!(pixels.at(20, 11), [255, 255, 255, 255], "the right band");
    // Nothing outside, and the middle stays hollow.
    assert_eq!(pixels.at(7, 8), [0, 0, 0, 255], "outside the rect");
    assert_eq!(pixels.at(8, 24), [0, 0, 0, 255], "below the rect");
    assert_eq!(pixels.at(16, 16), [0, 0, 0, 255], "hollow middle");
}

/// A document launch paints its first spread without a whole-plane render.
///
/// A book holds every page in one plane, so a scale-1.0 render of it costs one
/// page render per page before the window opens.
/// The launch therefore renders nothing, and the loader paints the spread the
/// fit frames (see [`vv_view::loader`]), which is the sharp picture this test
/// reads out of the host texture.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_document_launch_paints_its_first_spread() {
    struct NullHost;

    impl vv_view::viewer::Host for NullHost {
        fn set_title(&mut self, _title: &str) {}

        fn grab_pointer(&mut self) {}

        fn release_pointer(&mut self) {}

        fn place_pointer(&mut self, _at: Vec2) {}
    }

    let canvas = canvas!(64, 64);
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("book.pdf");
    // Three pages, so the surface holds a book: a cover row and one spread.
    std::fs::write(&path, page_pdf(3, [0, 128, 0])).expect("write the pdf");
    let launch = vv_view::viewer::Launch::open(&path, true).expect("the launch opens");
    // A book keeps the default window size: its canvas can stand taller than
    // any texture.
    assert_eq!(launch.window_size(), (1024, 768));
    let mut viewer = vv_view::viewer::Viewer::new(
        canvas.device.clone(),
        canvas.queue.clone(),
        FORMAT,
        canvas.device.limits().max_texture_dimension_2d,
        launch,
        canvas.size,
    );

    let mut host = NullHost;
    let mut green = false;
    for _ in 0..240 {
        let view = canvas
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        viewer
            .render(&mut host, &view, canvas.size, 0.016)
            .expect("the frame renders");
        let pixels = canvas.read();
        if (0..64).any(|y| {
            (0..64).any(|x| {
                let [r, g, b, _] = pixels.at(x, y);
                g > 100 && r < 80 && b < 80
            })
        }) {
            green = true;
            break;
        }
        // The spread paints on a worker, so the early frames show the clear
        // color and the picture lands a moment later.
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    assert!(green, "the first spread lands in the host texture");
    // The view measures the whole book, never the spread it paints: three
    // 64x64 pages stack into a cover row and a spread row.
    let (width, height) = viewer.image_size();
    assert!(
        width > 64 && height > 64,
        "a book, not one page: {width}x{height}"
    );
}

/// Opening a document from the grid paints no stretched page.
///
/// A grid cell holds page 0 and never the whole book, so the thumb of a white
/// page measures one page of a plane that stacks every spread.
/// The open used to stretch that picture over the whole plane, which painted the
/// window white until the loader's spread landed.
/// The cell picture now draws into page 0's slot, so the empty slot beside the
/// cover keeps the background and the page lands in the slot it fills.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_document_opened_from_the_grid_keeps_its_page_in_its_slot() {
    // A window wider than the book, so both slots of the cover row show beside
    // the plane and the empty one stays background.
    let canvas = canvas!(160, 64);
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(
        dir.path().join("a.svg"),
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect width="16" height="16" fill="#0000ff"/></svg>"##,
    )
    .expect("write the svg");
    // Three white pages, so the book holds a cover row and one spread.
    // The first page carries many white paths, so its render costs a while and
    // the cell picture shows for a few frames before the sharp spread lands.
    let mut ink = String::from("1 1 1 rg\n");
    for i in 0..4000 {
        let _ = writeln!(ink, "{} {} 1 1 re f", i % 64, (i / 64) % 64);
    }
    let paper = "1 1 1 rg\n0 0 64 64 re f\n";
    std::fs::write(dir.path().join("b.pdf"), pdf_with(3, &ink, paper)).expect("write the pdf");
    // A directory launch lists before any decode, so it keeps the default
    // window size and shows the grid: the document opens from a cell, which is
    // the path that holds a page-sized picture.
    let launch = vv_view::viewer::Launch::open(dir.path(), true).expect("the launch opens");
    assert_eq!(launch.window_size(), (1024, 768));
    let mut viewer = vv_view::viewer::Viewer::new(
        canvas.device.clone(),
        canvas.queue.clone(),
        FORMAT,
        canvas.device.limits().max_texture_dimension_2d,
        launch,
        canvas.size,
    );

    let mut host = NullHost;
    // Let the grid decode both cells, then open the document in the second
    // one.
    for _ in 0..60 {
        let view = canvas
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        viewer
            .render(&mut host, &view, canvas.size, 0.016)
            .expect("the grid frame renders");
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    host.press(&mut viewer, winit::keyboard::KeyCode::ArrowRight);
    host.press(&mut viewer, winit::keyboard::KeyCode::Enter);

    // The cover row spans two slots of one 64 px page at a fit scale of 1, so
    // the plane starts at x 16: the empty left slot covers x 16..80 and the
    // cover covers x 80..144.
    // The sample sits in the empty slot and left of every page a correct draw
    // makes, so only a picture stretched over the whole plane reaches it.
    let (empty, cover) = ((30, 32), (112, 32));
    let background = [0x06, 0x23, 0x29, 0xff];
    let mut landed = false;
    for _ in 0..120 {
        let view = canvas
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        viewer
            .render(&mut host, &view, canvas.size, 0.016)
            .expect("the image frame renders");
        let pixels = canvas.read();
        assert_eq!(
            pixels.at(empty.0, empty.1),
            background,
            "the slot beside the cover keeps the background"
        );
        landed = landed || pixels.at(cover.0, cover.1) == [255, 255, 255, 255];
        // The spread paints on a worker, so the early frames hold the cell
        // picture and the sharp spread lands a moment later.
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    assert!(landed, "the page paints in the slot it fills");
}

/// A `Viewer` paints into a texture its host owns.
///
/// This is the stage-2 seam end to end: a device with no surface, a launch, a
/// frame loop that reports time, and a host that answers the four window calls.
/// The viewer's own renderer paints the host's texture, so no swapchain is in
/// the path.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_viewer_paints_into_a_host_texture() {
    struct NullHost {
        titles: Vec<String>,
    }

    impl vv_view::viewer::Host for NullHost {
        fn set_title(&mut self, title: &str) {
            self.titles.push(title.to_string());
        }

        fn grab_pointer(&mut self) {}

        fn release_pointer(&mut self) {}

        fn place_pointer(&mut self, _at: Vec2) {}
    }

    let Some(canvas) = Canvas::new(32, 32) else {
        skip_without_adapter();
        return;
    };
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("red.svg");
    std::fs::write(
        &path,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect width="16" height="16" fill="#ff0000"/></svg>"##,
    )
    .expect("write the svg");
    let launch = vv_view::viewer::Launch::open(dir.path(), true).expect("the launch opens");
    // A directory launch lists before any decode, so it keeps the default
    // window size; a single-file launch sizes the window to its picture.
    assert_eq!(launch.window_size(), (1024, 768));
    let mut viewer = vv_view::viewer::Viewer::new(
        canvas.device.clone(),
        canvas.queue.clone(),
        FORMAT,
        canvas.device.limits().max_texture_dimension_2d,
        launch,
        canvas.size,
    );

    let mut host = NullHost { titles: Vec::new() };
    let mut red = false;
    for _ in 0..240 {
        let view = canvas
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        viewer
            .render(&mut host, &view, canvas.size, 0.016)
            .expect("the frame renders");
        let pixels = canvas.read();
        if pixels.at(16, 16) == [255, 0, 0, 255] {
            red = true;
            break;
        }
        // The decode runs on a worker, so the first frames show the clear
        // color and the picture lands a moment later.
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    assert!(red, "the picture lands in the host texture");
    // The base raster of a vector file is the source size, clamped into
    // 1024..=8192, so a 16x16 SVG measures 1024 square and a host that fits a
    // window to the picture reads that.
    assert_eq!(viewer.image_size(), (1024, 1024));
    assert!(
        host.titles.iter().any(|title| title.contains("red.svg")),
        "the host names its window after the open picture: {:?}",
        host.titles
    );
    assert!(!viewer.quit(), "nothing asked the viewer to exit");
}

/// How many pixels of the frame are exactly `color`.
fn count(pixels: &Pixels, color: Color) -> usize {
    let want = color.to_bytes();
    (0..pixels.height)
        .flat_map(|y| (0..pixels.width).map(move |x| (x, y)))
        .filter(|(x, y)| pixels.at(*x, *y) == want)
        .count()
}

/// A themed grid plates its cells, because a recolored page ends where the
/// window background begins.
///
/// A grid cell draws no backdrop: the gaps between cells are the clear color.
/// A theme's paper is that same color, so a wall of white pages would recolor
/// into one continuous dark field with no cell in it. The plate is the cell's
/// own shade, and it appears only while the theme is on.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_themed_grid_plates_its_cells() {
    /// The square the grid fills while a cell is still decoding.
    const PENDING: Color = Color::new(28, 28, 28, 255);
    let Some(canvas) = Canvas::new(256, 256) else {
        skip_without_adapter();
        return;
    };
    let dir = tempfile::tempdir().expect("a temp dir");
    for name in ["a.svg", "b.svg"] {
        std::fs::write(
            dir.path().join(name),
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect width="16" height="16" fill="#ffffff"/></svg>"##,
        )
        .expect("write the svg");
    }
    let launch = vv_view::viewer::Launch::open(dir.path(), true).expect("the launch opens");
    let mut viewer = vv_view::viewer::Viewer::new(
        canvas.device.clone(),
        canvas.queue.clone(),
        FORMAT,
        canvas.device.limits().max_texture_dimension_2d,
        launch,
        canvas.size,
    );
    let mut host = QuietHost;
    let plate = vv_core::theme::PLATE;

    // The plain grid, once every thumbnail has landed: white pages, and no
    // plate anywhere, because white paper already ends a cell.
    let mut plain = None;
    for _ in 0..240 {
        let view = canvas
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        viewer
            .render(&mut host, &view, canvas.size, 0.016)
            .expect("the frame renders");
        let pixels = canvas.read();
        if count(&pixels, PENDING) == 0 && count(&pixels, Color::WHITE) > 0 {
            plain = Some(pixels);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    let plain = plain.expect("the thumbnails land");
    let plain_white = count(&plain, Color::WHITE);
    assert_eq!(count(&plain, plate), 0, "a plain grid draws no plate");

    // The same grid under the theme: every cell has a band of its own, the
    // pages are no longer white, and the background is still the background.
    press(&mut viewer, winit::keyboard::KeyCode::ShiftLeft);
    press(&mut viewer, winit::keyboard::KeyCode::KeyT);
    let mut themed = None;
    for _ in 0..240 {
        let view = canvas
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        viewer
            .render(&mut host, &view, canvas.size, 0.016)
            .expect("the frame renders");
        let pixels = canvas.read();
        if count(&pixels, plate) > 0 {
            themed = Some(pixels);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    let themed = themed.expect("the themed grid draws its plates");
    assert!(
        count(&themed, plate) > 1000,
        "each cell carries a band, not a speck: {}",
        count(&themed, plate)
    );
    assert!(
        count(&themed, Color::WHITE) < plain_white / 2,
        "the pages are no longer white: {} of {plain_white}",
        count(&themed, Color::WHITE)
    );
    assert!(
        count(&themed, CLEAR) > 0,
        "the paper and the gaps between cells are still the background"
    );
}

/// A host that answers the four window calls and records nothing but the
/// titles, for a test that drives the viewer itself.
struct QuietHost;

impl vv_view::viewer::Host for QuietHost {
    fn set_title(&mut self, _title: &str) {}
    fn grab_pointer(&mut self) {}
    fn release_pointer(&mut self) {}
    fn place_pointer(&mut self, _at: Vec2) {}
}

/// Press one key through the viewer's own input path.
fn press(viewer: &mut vv_view::viewer::Viewer, key: winit::keyboard::KeyCode) {
    viewer.handle_event(
        vv_view::viewer::Event::Key {
            key,
            state: winit::event::ElementState::Pressed,
            repeat: false,
            synthetic: false,
            text: None,
        },
        &mut QuietHost,
    );
}

/// `Shift+T` recolors the picture the image view shows, and turns it back.
///
/// The theme is a uniform the fragment stage reads, so neither frame decodes
/// anything again: the second paint shows the same texture through the other
/// uniform, which is the whole point of the toggle living in the shader.
#[test]
#[ignore = "needs a Vulkan device"]
fn shift_t_recolors_the_picture_and_turns_it_back() {
    let Some(canvas) = Canvas::new(32, 32) else {
        skip_without_adapter();
        return;
    };
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("white.svg");
    std::fs::write(
        &path,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect width="16" height="16" fill="#ffffff"/></svg>"##,
    )
    .expect("write the svg");
    let launch = vv_view::viewer::Launch::open(&path, true).expect("the launch opens");
    let mut viewer = vv_view::viewer::Viewer::new(
        canvas.device.clone(),
        canvas.queue.clone(),
        FORMAT,
        canvas.device.limits().max_texture_dimension_2d,
        launch,
        canvas.size,
    );
    let mut host = QuietHost;

    // Wait out the decode, then read the plain picture.
    let mut painted = None;
    for _ in 0..240 {
        let view = canvas
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        viewer
            .render(&mut host, &view, canvas.size, 0.016)
            .expect("the frame renders");
        let pixels = canvas.read();
        if close(pixels.at(16, 16), [255, 255, 255, 255], 2) {
            painted = Some(pixels);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    let plain = painted.expect("the white picture lands");

    // Shift+T, then one frame for the key and one for the uniform it wrote.
    press(&mut viewer, winit::keyboard::KeyCode::ShiftLeft);
    press(&mut viewer, winit::keyboard::KeyCode::KeyT);
    for _ in 0..2 {
        let view = canvas
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        viewer
            .render(&mut host, &view, canvas.size, 0.016)
            .expect("the frame renders");
    }
    let recolored = canvas.read();
    assert!(
        close(recolored.at(16, 16), THEME.paper.to_bytes(), 2),
        "white paper is the theme paper: {:?} (was {:?})",
        recolored.at(16, 16),
        plain.at(16, 16)
    );

    // The keys go up, then `Shift+T` again: the same texture, plain.
    for key in [
        winit::keyboard::KeyCode::ShiftLeft,
        winit::keyboard::KeyCode::KeyT,
    ] {
        viewer.handle_event(
            vv_view::viewer::Event::Key {
                key,
                state: winit::event::ElementState::Released,
                repeat: false,
                synthetic: false,
                text: None,
            },
            &mut host,
        );
    }
    let view = canvas
        .target
        .create_view(&wgpu::TextureViewDescriptor::default());
    viewer
        .render(&mut host, &view, canvas.size, 0.016)
        .expect("the frame renders");
    press(&mut viewer, winit::keyboard::KeyCode::ShiftLeft);
    press(&mut viewer, winit::keyboard::KeyCode::KeyT);
    for _ in 0..2 {
        let view = canvas
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        viewer
            .render(&mut host, &view, canvas.size, 0.016)
            .expect("the frame renders");
    }
    let back = canvas.read();
    assert!(
        close(back.at(16, 16), [255, 255, 255, 255], 2),
        "the theme is off again: {:?}",
        back.at(16, 16)
    );
}

/// A picture that arrives from outside the process paints in a blank viewer.
///
/// The launch names no path, so nothing this test shows ever reaches the disk:
/// the bytes travel through `Viewer::accept`, decode on the viewer's worker and
/// land in the host texture.
#[test]
#[ignore = "needs a Vulkan adapter"]
fn a_pushed_picture_lands_in_a_blank_viewer() {
    struct NullHost;

    impl vv_view::viewer::Host for NullHost {
        fn set_title(&mut self, _title: &str) {}

        fn grab_pointer(&mut self) {}

        fn release_pointer(&mut self) {}

        fn place_pointer(&mut self, _at: Vec2) {}
    }

    let Some(canvas) = Canvas::new(32, 32) else {
        skip_without_adapter();
        return;
    };
    let launch = vv_view::viewer::Launch::blank(true);
    // A blank launch keeps the default window size and names the protocol.
    assert_eq!(launch.window_size(), (1024, 768));
    assert!(launch.title().contains("mcp"), "{}", launch.title());
    let mut viewer = vv_view::viewer::Viewer::new(
        canvas.device.clone(),
        canvas.queue.clone(),
        FORMAT,
        canvas.device.limits().max_texture_dimension_2d,
        launch,
        canvas.size,
    );

    let mut host = NullHost;
    let mut frame = |viewer: &mut vv_view::viewer::Viewer| {
        let view = canvas
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        viewer
            .render(&mut host, &view, canvas.size, 0.016)
            .expect("the frame renders");
        canvas.read()
    };

    // A blank launch clears to the background and shows nothing.
    assert_eq!(frame(&mut viewer).at(16, 16), [6, 0x23, 0x29, 0xff]);
    let status = viewer.status();
    assert_eq!(
        (status.grid, status.planes, status.source.as_str()),
        (true, 0, "mcp")
    );

    // One push of an SVG document, named by its bytes alone: the sniff decides
    // the backend, so the call carries no extension.
    let square = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="#ff0000"/></svg>"##;
    let (reply, answers) = std::sync::mpsc::channel();
    viewer.accept(vv_view::push::Push::Bytes {
        data: square.as_bytes().to_vec(),
        reply,
    });
    let shown = answers
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the worker answers")
        .expect("the bytes decode");
    assert_eq!(shown.planes, 1);
    // The base raster of a vector file clamps its long side into 1024..=8192.
    assert_eq!(shown.width, 1024.0);

    let mut red = false;
    for _ in 0..240 {
        if frame(&mut viewer).at(16, 16) == [255, 0, 0, 255] {
            red = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    assert!(red, "the pushed picture lands in the host texture");
    let status = viewer.status();
    assert_eq!((status.grid, status.planes), (false, 1));
    assert_eq!(status.source, "mcp");

    // A clear returns to the blank view, which clears to the background again.
    let (reply, answers) = std::sync::mpsc::channel();
    viewer.accept(vv_view::push::Push::Clear { reply });
    let status = answers
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the clear answers");
    assert_eq!((status.grid, status.planes), (true, 0));
    assert_eq!(frame(&mut viewer).at(16, 16), [6, 0x23, 0x29, 0xff]);
}

/// The height of one page of [`heavy_plane`], in surface units.
const PAGE_HEIGHT: f64 = 200.0;

/// A plane of `pages` pages stacked, each a wave of `curves` segments inside
/// its own slot.
///
/// A page's band is that page padded by half on every side, so it reaches half
/// way into the slot below: a frame of one page counts the page and the one
/// under it. The counts of this test are read that way.
fn heavy_plane(pages: usize, curves: usize) -> std::sync::Arc<[vv_source::scene::Primitive]> {
    use vv_core::{
        draw::Color,
        scene::{FillRule, Paint, Primitive},
    };

    let list: Vec<Primitive> = (0..pages)
        .map(|page| {
            let mid = page as f64 * PAGE_HEIGHT + PAGE_HEIGHT / 2.0;
            let mut path = kurbo::BezPath::new();
            path.move_to((0.0, mid));
            for step in 0..curves {
                let x = (step % 40) as f64 * 5.0;
                // One amplitude for every curve, so a curve costs the same at
                // every step and the work of a band follows the pages it
                // reaches.
                let up = if (step / 40) % 2 == 0 { 30.0 } else { -30.0 };
                path.curve_to((x + 1.7, mid + up), (x + 3.3, mid - up), (x + 5.0, mid));
            }
            Primitive::Path {
                path,
                transform: kurbo::Affine::IDENTITY,
                fill: Some(Paint::Solid(Color::new(255, 0, 0, 255))),
                fill_rule: FillRule::NonZero,
                stroke: None,
            }
        })
        .collect();
    list.into()
}

/// A band the window outgrew is given up, so a zoom back in paints the list
/// again.
///
/// `Work` counts a primitive whole however little of it the window shows, and a
/// band still covers every frame the window later zooms into, so the band of the
/// whole plane answers a frame of one page with the work of all six pages: at the
/// zoom of a page that bill is past `LINE_BUDGET`, and it stays past it for every
/// frame after, because the band covers them and no frame encodes again. So the
/// whole plane keeps the raster for good once the window has shown it in one
/// frame, which is the zoom out and back in that stuck. A page band is a frame
/// the renderer holds at that same zoom, so the band is the whole of the
/// difference.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_zoom_back_in_answers_the_list_again() {
    use vv_core::{
        dim::{SurfaceSize, WindowSize, WindowVector},
        region,
    };

    const PAGES: usize = 6;
    let list = heavy_plane(PAGES, 15_000);
    let plane = SurfaceSize::new(PAGE_HEIGHT as f32, (PAGES as f64 * PAGE_HEIGHT) as f32);
    let page = |page: usize, scale: f32| region::Frame {
        img: plane,
        win: WindowSize::new(256.0, 256.0),
        pan: WindowVector::new(
            0.0,
            -((page as f64 * PAGE_HEIGHT + PAGE_HEIGHT / 2.0) as f32 - plane.height / 2.0) * scale,
        ),
        scale,
        angle: 0.0,
    };
    let whole = page(0, 256.0 / plane.height);
    let one = page(0, 256.0 / PAGE_HEIGHT as f32);

    let canvas = canvas!(256, 256);
    let mut pass = canvas
        .renderer
        .display_pass(&canvas.device)
        .expect("a display-list pass");

    // Every page of the plane at once fits the renderer as the window leaves it,
    // so this frame paints the list and keeps a band of the whole plane.
    assert!(
        pass.draw(&canvas.device, &canvas.queue, &list, &whole, canvas.size)
            .expect("the pass answers")
            .is_some(),
        "the whole plane at the fit scale paints the list"
    );

    // A page of it is a frame the renderer holds on its own, and it is the frame
    // a zoom back in draws: the band of the whole plane must not be what decides
    // it.
    assert!(
        pass.draw(&canvas.device, &canvas.queue, &list, &one, canvas.size)
            .expect("the pass answers")
            .is_some(),
        "a page of the plane the window zoomed into paints the list"
    );

    // And the plane at the fit scale is still a frame the pass paints, so the
    // band a frame gives up is the band of the frame before it and never one the
    // picture needs.
    assert!(
        pass.draw(&canvas.device, &canvas.queue, &list, &whole, canvas.size)
            .expect("the pass answers")
            .is_some(),
        "a zoom back out paints the list it painted before"
    );
}

/// A book opened from the grid never shows one page in the middle of the window.
///
/// A cell of a document holds page 0 and never the spread the image view shows,
/// so the decode a cell waits for can never answer that view. Waiting for it
/// anyway framed that page on its own — the cell's fit, in the middle of the
/// window — and a frame later moved it into its slot of the spread, which is the
/// jump a grid open used to show.
/// This window is 1200 by 800 and the cover sits in the right slot, so the two
/// framings are told apart by which side of the window holds the cover's red:
/// centered, the left probe is red; in its slot, only the right one is.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_book_opened_from_the_grid_is_framed_before_its_page_lands() {
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(
        dir.path().join("a.pdf"),
        pdf_with_box(3, [0, 0, 200, 280], |page| {
            let ink = if page == 0 { "1 0 0" } else { "0 0 1" };
            format!("{ink} rg\n0 0 200 280 re f\n")
        }),
    )
    .expect("write the book");
    std::fs::write(dir.path().join("b.svg"), SMALL_SVG).expect("write the svg");
    let canvas = canvas!(1200, 800);
    let open = || {
        let launch = vv_view::viewer::Launch::open(dir.path(), true).expect("the launch opens");
        vv_view::viewer::Viewer::new(
            canvas.device.clone(),
            canvas.queue.clone(),
            FORMAT,
            canvas.device.limits().max_texture_dimension_2d,
            launch,
            canvas.size,
        )
    };
    let mut host = NullHost;

    // A first launch that settles, so the cells hold the headers a warm launch
    // has: the case a grid open waits on is a cell whose picture is on its way.
    let mut viewer = open();
    for _ in 0..60 {
        shot_frame(&canvas, &mut viewer, &mut host, 0.016, true);
    }
    drop(viewer);

    // The warm launch, entered while that cell is still queued.
    let mut viewer = open();
    for _ in 0..2 {
        shot_frame(&canvas, &mut viewer, &mut host, 0.016, true);
    }
    host.press(&mut viewer, winit::keyboard::KeyCode::Enter);

    let mut seen = false;
    for _ in 0..90 {
        let pixels = shot_frame_read(&canvas, &mut viewer, &mut host);
        let (left, right) = (pixels.at(520, 400), pixels.at(900, 400));
        let ink = |pixel: [u8; 4]| pixel[0] > 128 && pixel[2] < 64;
        assert!(
            !ink(left),
            "the cover is never framed on its own: left of the spread is {left:?}"
        );
        seen |= ink(right);
    }
    assert!(
        seen,
        "the cover lands in the slot of its spread, right of the fold"
    );
}

/// Run one frame of `viewer`, and read the target back.
fn shot_frame_read(
    canvas: &Canvas,
    viewer: &mut vv_view::viewer::Viewer,
    host: &mut NullHost,
) -> Pixels {
    let view = canvas
        .target
        .create_view(&wgpu::TextureViewDescriptor::default());
    viewer
        .render(host, &view, canvas.size, 0.016)
        .expect("the frame renders");
    canvas.read()
}
