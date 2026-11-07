//! Headless paints: the renderer draws into a texture and the test reads the
//! pixels back.
//!
//! The gate has no display, and a `Surface` needs a window, so every test here
//! is `#[ignore]`: a host with a Vulkan device runs them by hand, and the gate
//! only compiles them.
//! That compiles the whole draw seam of [`versatile_viewer::render`] and keeps
//! it from rotting, and one command proves the pipeline draws what the view
//! asks for:
//!
//! ```sh
//! export VK_DRIVER_FILES=$(nix build --no-link --print-out-paths nixpkgs#mesa)/share/vulkan/icd.d/lvp_icd.x86_64.json
//! cargo test --test gpu_offscreen -- --ignored
//! ```
//!
//! A test that finds no adapter prints one line and passes, so the same command
//! is safe on a host without Vulkan.
//!
//! # What the tests pin
//!
//! - the clear color, and that an empty frame still clears;
//! - a solid fill, on exactly the pixels of its rect;
//! - the source rect and the texel grid of a texture draw, which is the
//!   sharpness rule of `patch_dest` in the viewer;
//! - straight alpha, and a tint that scales it;
//! - a turned quad, on the pixels its corners name;
//! - the decoder-to-draw chain of an SVG, a Typst page and a Mermaid diagram;
//! - the transparent gutter of a book plane.

// The tests read and name window pixels as whole numbers, so a float rect
// becomes a pixel index with one cast, and a dimension becomes an `f32`.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use std::fmt::Write as _;

use versatile_viewer::{
    dim::Rect,
    draw::{Color, Quad, Turn, Vec2},
    render::Renderer,
    surface::{View, open_surface},
    texture::Filter,
};

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
        let (x0, y0) = (rect.x as usize, rect.y as usize);
        let (x1, y1) = (rect.right() as usize, rect.bottom() as usize);
        (y0..y1).all(|y| (x0..x1).all(|x| self.at(x, y) == color))
    }
}

/// Skip a test when the host answers no Vulkan device.
macro_rules! canvas {
    ($width:expr, $height:expr) => {
        match Canvas::new($width, $height) {
            Some(canvas) => canvas,
            None => {
                eprintln!("no Vulkan adapter: skipped");
                return;
            }
        }
    };
}

/// The clear color reaches the whole frame, even with nothing to draw.
#[test]
#[ignore = "needs a Vulkan device"]
fn an_empty_frame_clears_to_the_background() {
    let mut canvas = canvas!(8, 4);
    canvas.begin(CLEAR);
    let pixels = canvas.print();
    assert!(
        pixels.rect_is(Rect::new(0.0, 0.0, 8.0, 4.0), [0x06, 0x23, 0x29, 0xff]),
        "the clear color fills the frame"
    );
}

/// A solid fill lands on exactly the pixels of its rect.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_solid_rect_lands_on_its_pixels() {
    let mut canvas = canvas!(16, 16);
    canvas.begin(CLEAR);
    let rect = Rect::new(4.0, 6.0, 8.0, 4.0);
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
        Rect::new(0.0, 0.0, 4.0, 4.0),
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
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Rect::new(0.0, 0.0, 8.0, 8.0),
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
        Rect::new(1.0, 1.0, 1.0, 1.0),
        Rect::new(0.0, 0.0, 4.0, 4.0),
        Turn::NONE,
        Color::WHITE,
        Filter::Nearest,
    );
    let pixels = canvas.print();
    assert!(
        pixels.rect_is(Rect::new(0.0, 0.0, 4.0, 4.0), [255, 255, 255, 255]),
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
    let dest = Rect::new(20.0, 20.0, 20.0, 10.0);
    let quad = Quad::pro(dest, Turn::new(dest.center_origin(), 90.0));
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
    let dest = Rect::new((win_w - dw) / 2.0, (win_h - dh) / 2.0, dw, dh);
    canvas.begin(CLEAR);
    canvas.renderer.texture_pro(
        &texture,
        Rect::sized(width as f32, height as f32),
        dest,
        Turn::NONE,
        Color::WHITE,
        Filter::Linear,
    );
    dest
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
        pixels.at(dest.x as usize, dest.y as usize),
        [0xff, 0, 0, 255],
        "the edge counts too"
    );
}

/// A Typst page paints its own paper.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_typst_page_paints_a_red_box() {
    let mut canvas = canvas!(32, 32);
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("page.typ");
    std::fs::write(
        &path,
        "#set page(width: 32pt, height: 32pt, margin: 0pt, fill: rgb(\"#0000ff\"))\n#rect(width: \
         100%, height: 100%, fill: rgb(\"#ff0000\"))\n",
    )
    .expect("write the typst source");
    let dest = paint_surface(&mut canvas, &path);
    let pixels = canvas.print();
    let middle = (
        (dest.x + dest.width / 2.0) as usize,
        (dest.y + dest.height / 2.0) as usize,
    );
    assert_eq!(pixels.at(middle.0, middle.1), [255, 0, 0, 255], "the box");
}

/// A Mermaid diagram paints white paper under its boxes.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_mermaid_diagram_paints_on_white_paper() {
    let mut canvas = canvas!(64, 64);
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("flow.mmd");
    std::fs::write(&path, "flowchart LR\n  A[one] --> B[two]\n").expect("write the source");
    let dest = paint_surface(&mut canvas, &path);
    let pixels = canvas.print();
    // The diagram paints its own paper (`mermaid::BACKGROUND`, white), and its
    // boxes and labels leave dark ink on it.
    let (x0, y0) = (dest.x as usize, dest.y as usize);
    let (x1, y1) = (dest.right() as usize, dest.bottom() as usize);
    assert_eq!(
        pixels.at(x0 + 1, y0 + 1),
        [255, 255, 255, 255],
        "white paper"
    );
    let inked = (y0..y1).any(|y| (x0..x1).any(|x| pixels.at(x, y)[0] < 200));
    assert!(inked, "the boxes and labels leave ink on the paper");
}

/// A book plane leaves its gutter transparent, so the window shows through.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_book_plane_keeps_its_gutter_transparent() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("book.typ");
    // Three pages, so the book holds a cover row and one spread.
    std::fs::write(
        &path,
        "#set page(width: 64pt, height: 64pt, margin: 0pt, fill: \
         rgb(\"#ffffff\"))\none\n#pagebreak()\ntwo\n#pagebreak()\nthree\n",
    )
    .expect("write the typst source");
    let surface = versatile_viewer::surface::open_book_surface(&path).expect("a book");
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

/// Build a one-page PDF whose page is filled with `color`.
///
/// The xref offsets are computed here, so the file is valid and needs no
/// fixture on disk.
fn one_page_pdf(color: [u8; 3]) -> Vec<u8> {
    let content = format!(
        "{} {} {} rg\n0 0 64 64 re f\n",
        f64::from(color[0]) / 255.0,
        f64::from(color[1]) / 255.0,
        f64::from(color[2]) / 255.0
    );
    let objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n".to_string(),
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n".to_string(),
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 64 64] /Contents 4 0 R /Resources << \
         >> >> endobj\n"
            .to_string(),
        format!(
            "4 0 obj << /Length {} >> stream\n{}endstream\nendobj\n",
            content.len(),
            content
        ),
    ];
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
    std::fs::write(&path, one_page_pdf([0, 128, 0])).expect("write the pdf");
    let dest = paint_surface(&mut canvas, &path);
    let pixels = canvas.print();
    let middle = (
        (dest.x + dest.width / 2.0) as usize,
        (dest.y + dest.height / 2.0) as usize,
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
        Rect::sized(2.0, 2.0),
        Rect::new(0.0, 0.0, 2.0, 2.0),
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
        Rect::sized(2.0, 2.0),
        Rect::new(0.5, 0.0, 2.0, 2.0),
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
/// [`versatile_viewer::texture::Uploader`], so `DecodeMsg::Done` carries a
/// texture and no pixel copy runs on the frame thread.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_loader_worker_uploads_its_own_frame() {
    use versatile_viewer::{dim::PreviewPx, loader::Loader, texture::Ready};

    let mut canvas = canvas!(32, 32);
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("page.pdf");
    std::fs::write(&path, one_page_pdf([200, 200, 0])).expect("write the pdf");
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
            Some(Ok(versatile_viewer::loader::DecodeMsg::Done { frame, .. })) => break frame,
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
        Rect::sized(64.0, 64.0),
        Rect::new(0.0, 0.0, 32.0, 32.0),
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
    use versatile_viewer::{DIM_LEVEL, DIM_VEIL, draw::Vec2, region, region_sel};

    let mut canvas = canvas!(64, 64);
    let selection = region::Crop {
        x: 16.0,
        y: 16.0,
        w: 32.0,
        h: 32.0,
    };
    for angle in [0.0_f32, 30.0] {
        let frame = region::Frame {
            img: Vec2::new(64.0, 64.0),
            win: Vec2::new(64.0, 64.0),
            pan: Vec2::ZERO,
            scale: 1.0,
            angle,
        };
        canvas.begin(Color::BLACK);
        // The picture: a white plane, turned like the draw turns it.
        let dest = Rect::new(32.0, 32.0, 64.0, 64.0);
        canvas.renderer.fill(
            Quad::pro(dest, Turn::new(dest.center_origin(), angle)),
            Color::WHITE,
        );
        let outer = region::Crop {
            x: 0.0,
            y: 0.0,
            w: frame.img.x,
            h: frame.img.y,
        };
        for band in region_sel::around(outer, selection) {
            let corners = band.map(|point| frame.to_screen(point));
            canvas.renderer.fill(Quad::new(corners), DIM_VEIL);
        }
        let pixels = canvas.print();
        let at = |point: Vec2| {
            let screen = frame.to_screen(point);
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
    let rect = Rect::new(8.0, 8.0, 16.0, 16.0);
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

    impl versatile_viewer::viewer::Host for NullHost {
        fn set_title(&mut self, title: &str) {
            self.titles.push(title.to_string());
        }

        fn grab_pointer(&mut self) {}

        fn release_pointer(&mut self) {}

        fn place_pointer(&mut self, _at: Vec2) {}
    }

    let Some(canvas) = Canvas::new(32, 32) else {
        eprintln!("no Vulkan adapter, skipping");
        return;
    };
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("red.svg");
    std::fs::write(
        &path,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect width="16" height="16" fill="#ff0000"/></svg>"##,
    )
    .expect("write the svg");
    let launch =
        versatile_viewer::viewer::Launch::open(dir.path(), true).expect("the launch opens");
    // A directory launch lists before any decode, so it keeps the default
    // window size; a single-file launch sizes the window to its picture.
    assert_eq!(launch.window_size(), (1024, 768));
    let mut viewer = versatile_viewer::viewer::Viewer::new(
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
