//! Headless paints: the renderer draws into a texture and the test reads the
//! pixels back.
//!
//! The default gate has no adapter, so every test here is `#[ignore]`, and the
//! `checks.gpu` gate runs them on Mesa's lavapipe, a software Vulkan driver
//! that needs no GPU and no display.
//! A host runs the same tests by hand:
//!
//! ```sh
//! export VK_DRIVER_FILES=$(nix build --no-link --print-out-paths nixpkgs#mesa)/share/vulkan/icd.d/lvp_icd.x86_64.json
//! cargo test --test gpu_offscreen -- --ignored
//! ```
//!
//! A test that finds no adapter prints one line and passes, so the same command
//! is safe on a host without Vulkan, and `VV_GPU_REQUIRED` turns that skip into
//! a failure for a gate that provides a driver of its own.
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
//!   background pixel, and its fallback picture after a re-layout.

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
        .expect("the list draws");
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
        .expect("the list draws");

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
        .expect("the list draws");

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
        .expect("the list draws");
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
    let kids = (0..pages)
        .map(|page| format!("{} 0 R", 3 + 2 * page))
        .collect::<Vec<_>>()
        .join(" ");
    let mut objects = vec![
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n".to_string(),
        format!("2 0 obj << /Type /Pages /Kids [{kids}] /Count {pages} >> endobj\n"),
    ];
    for page in 0..pages {
        let (body, stream) = (3 + 2 * page, 4 + 2 * page);
        objects.push(format!(
            "{body} 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 64 64] /Contents {stream} 0 \
             R /Resources << >> >> endobj\n"
        ));
        let content = if page == 0 { first } else { rest };
        objects.push(format!(
            "{stream} 0 obj << /Length {} >> stream\n{content}endstream\nendobj\n",
            content.len()
        ));
    }
    let mut pdf = String::from("%PDF-1.4\n");
    let mut offsets = Vec::new();
    for object in &objects {
        offsets.push(pdf.len());
        pdf.push_str(object);
    }
    let xref = pdf.len();
    let _ = write!(pdf, "xref\n0 {}\n", objects.len() + 1);
    pdf.push_str("0000000000 65535 f \n");
    for offset in offsets {
        let _ = writeln!(pdf, "{offset:010} 00000 n ");
    }
    let _ = write!(
        pdf,
        "trailer << /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
        objects.len() + 1
    );
    pdf.into_bytes()
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
