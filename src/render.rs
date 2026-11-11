//! One instanced-quad pipeline: the whole draw surface of the viewer.
//!
//! raylib drew through nine primitives and a per-call texture bind. This module
//! replaces all of them with one pipeline, one vertex shader and one instance
//! per quad, so the grid, the region veil and the selection border are the same
//! draw.
//!
//! # One instance is four corners
//!
//! An [`Inst`] names four window-space corners, a source rect in normalized
//! texture coordinates and a tint.
//! The corners arrive already turned, so the shader needs no trigonometry: the
//! CPU turns a picture through [`crate::draw::Quad::turned`], which is one
//! `sin_cos` per quad.
//! A triangle is a quad whose last corner repeats the first, so the second
//! triangle has no area and the call needs no second primitive.
//!
//! The corner list is the general form: a rect, a turned rect, a veil band and
//! a selection border all name four points around their perimeter, and none of
//! them needs a different pipeline.
//!
//! # The three raylib facts that become explicit
//!
//! - straight alpha, `SRC_ALPHA` over `ONE_MINUS_SRC_ALPHA`, like raylib;
//! - `cull_mode: None`, so a quad never depends on its winding;
//! - a non-sRGB target, so the pixel values stay what the decoder wrote.
//!
//! # Batching
//!
//! A frame collects instances into one list and one call per texture, so a grid
//! of a thousand cells costs one draw call per visible cell and one buffer
//! upload, not one bind per quad.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::{
    dim::Rect,
    draw::{Color, Quad, Turn, Vec2},
    gpu::{FrameError, Gpu},
    texture::{Filter, Texture, TextureError, Uploader},
};

/// How many instances one frame begins with.
///
/// The buffer grows on demand, so this is the first allocation and not a cap.
const START_INSTANCES: u32 = 1024;

/// The one shader of the viewer.
///
/// A float `vec2f` per corner keeps the instance at 64 bytes and every
/// attribute inside its alignment.
const SHADER: &str = r"
struct Globals {
    viewport: vec4f,
};

@group(0) @binding(0) var<uniform> globals: Globals;

struct Inst {
    @location(0) p0: vec2f,
    @location(1) p1: vec2f,
    @location(2) p2: vec2f,
    @location(3) p3: vec2f,
    @location(4) src: vec4f,
    @location(5) tint: vec4f,
};

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
    @location(1) tint: vec4f,
};

@vertex
fn vmain(@builtin(vertex_index) index: u32, inst: Inst) -> VOut {
    var points = array<vec2f, 4>(inst.p0, inst.p1, inst.p2, inst.p3);
    var corners = array<vec2f, 4>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
    );
    var order = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
    let corner = order[index];
    let point = points[corner];

    var out: VOut;
    out.clip = vec4f(
        point.x / globals.viewport.x * 2.0 - 1.0,
        1.0 - point.y / globals.viewport.y * 2.0,
        0.0,
        1.0,
    );
    out.uv = mix(inst.src.xy, inst.src.zw, corners[corner]);
    out.tint = inst.tint;
    return out;
}

@group(1) @binding(0) var tex_sampler: sampler;
@group(1) @binding(1) var tex: texture_2d<f32>;

@fragment
fn fmain(in: VOut) -> @location(0) vec4f {
    // Straight alpha in, straight alpha out: the blend state does the rest.
    return textureSample(tex, tex_sampler, in.uv) * in.tint;
}
";

/// One quad: four window-space corners, a source rect and a tint.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Inst {
    /// The corners, around the perimeter, in window pixels.
    pub p: [[f32; 2]; 4],
    /// `u0, v0, u1, v1` of the source rect, normalized to the texture.
    pub src: [f32; 4],
    /// The straight-alpha multiplier, `[r, g, b, a]`.
    pub tint: [f32; 4],
}

impl Inst {
    /// A solid quad over the one-by-one white texture.
    #[must_use]
    pub fn solid(quad: Quad, color: Color) -> Self {
        Self {
            p: quad.p.map(|point| [point.x, point.y]),
            src: [0.0, 0.0, 1.0, 1.0],
            tint: color.to_f32(),
        }
    }

    /// A textured quad over `src` pixels of a texture of `dims`.
    #[must_use]
    pub fn textured(quad: Quad, src: Rect, dims: (u32, u32), tint: Color) -> Self {
        let (width, height) = (f64::from(dims.0.max(1)), f64::from(dims.1.max(1)));
        Self {
            p: quad.p.map(|point| [point.x, point.y]),
            src: [
                (f64::from(src.x) / width) as f32,
                (f64::from(src.y) / height) as f32,
                (f64::from(src.right()) / width) as f32,
                (f64::from(src.bottom()) / height) as f32,
            ],
            tint: tint.to_f32(),
        }
    }

    /// The six attributes, with their offsets written out.
    ///
    /// The offsets are explicit because the shader reads them by location and
    /// the CPU writes them by field. The test below pins them to the real
    /// struct offsets, so a field reorder cannot silently shift a value.
    const ATTRIBS: [wgpu::VertexAttribute; 6] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 8,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 16,
            shader_location: 2,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 24,
            shader_location: 3,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 32,
            shader_location: 4,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 48,
            shader_location: 5,
        },
    ];

    /// The instance buffer layout.
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// One run of instances that share a bind group.
struct DrawCall {
    group: Arc<wgpu::BindGroup>,
    start: u32,
    len: u32,
}

/// The pipeline, its viewport uniform, the shared textures and one frame's
/// batch.
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    globals_group: wgpu::BindGroup,
    uploader: Uploader,
    /// The one-by-one white texture behind a solid fill.
    white: Texture,
    instances: wgpu::Buffer,
    capacity: u32,
    insts: Vec<Inst>,
    calls: Vec<DrawCall>,
    size: (u32, u32),
    /// The color the frame clears to, which the view sets every frame.
    clear: Color,
}

impl Renderer {
    /// Build the pipeline layout, the pipeline, the shared textures and the
    /// buffers.
    ///
    /// `adapter_side` is the adapter's largest 2D texture, which caps every
    /// upload.
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        adapter_side: u32,
    ) -> Self {
        let texture_layout = texture_layout(device);
        let uploader = Uploader::new(device, queue, adapter_side, texture_layout.clone());
        let white = uploader.white();
        let (globals, globals_layout, globals_group) = globals(device);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vv shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = pipeline_layout(device, &globals_layout, &texture_layout);
        Self {
            pipeline: pipeline(device, format, &shader, &layout),
            globals,
            globals_group,
            uploader,
            white,
            instances: instance_buffer(device, START_INSTANCES),
            capacity: START_INSTANCES,
            insts: Vec::new(),
            calls: Vec::new(),
            size: (1, 1),
            clear: Color::BLACK,
        }
    }

    /// The handle a decode worker uses to upload its own pixels.
    #[must_use]
    pub fn uploader(&self) -> Uploader {
        self.uploader.clone()
    }

    /// Upload a decoded picture on the calling thread.
    ///
    /// # Errors
    ///
    /// Returns [`TextureError`] when the buffer does not match its size.
    pub fn upload(&self, image: &crate::DecodedImage) -> Result<Texture, TextureError> {
        self.uploader.image(image)
    }

    /// Upload raw pixels on the calling thread.
    ///
    /// # Errors
    ///
    /// Returns [`TextureError`] when the buffer does not hold exactly
    /// `width * height * 4` bytes.
    pub fn upload_rgba(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Texture, TextureError> {
        self.uploader.rgba(rgba, width, height, "vv picture")
    }

    /// Start a frame: set the viewport and empty the batch.
    pub fn begin(&mut self, queue: &wgpu::Queue, size: (u32, u32)) {
        self.insts.clear();
        self.calls.clear();
        if self.size != size {
            self.size = size;
            queue.write_buffer(
                &self.globals,
                0,
                bytemuck::cast_slice(&[size.0 as f32, size.1 as f32, 0.0, 0.0]),
            );
        }
    }

    /// Draw a solid quad.
    ///
    /// The tint multiplies a one-by-one white texture, so a rectangle, a veil
    /// band and a selection border are the same draw as a picture.
    pub fn fill(&mut self, quad: Quad, color: Color) {
        self.push(
            self.white.group(Filter::Nearest).clone(),
            Inst::solid(quad, color),
        );
    }

    /// The color the next frame clears to.
    ///
    /// The view sets it once per frame, so the window background and the grid
    /// gaps never disagree.
    pub fn set_clear(&mut self, color: Color) {
        self.clear = color;
    }

    /// Draw a solid axis-aligned rectangle.
    pub fn fill_rect(&mut self, rect: Rect, color: Color) {
        self.fill(Quad::rect(rect), color);
    }

    /// Draw a line of `thickness` between two window points.
    pub fn line(&mut self, a: Vec2, b: Vec2, thickness: f32, color: Color) {
        self.fill(Quad::line(a, b, thickness), color);
    }

    /// Draw a hairline between two window points.
    pub fn line_v(&mut self, a: Vec2, b: Vec2, color: Color) {
        self.line(a, b, 1.0, color);
    }

    /// Draw the outline of a rectangle, `thickness` pixels wide, inside it.
    ///
    /// The ring covers the outer band of the rect, which is where raylib drew
    /// its rectangle outline, so a cell ring never spills into the gap between
    /// two cells.
    /// The four edges meet at the corners, so the ring closes with no notch.
    pub fn stroke_rect(&mut self, rect: Rect, thickness: f32, color: Color) {
        // A line of `thickness` sits centered on its segment. The top and the
        // bottom run the whole width so they close both corners, and the two
        // sides span only the gap between them, exactly like raylib's own
        // outline.
        let half = thickness / 2.0;
        let (left, right) = (rect.x + half, rect.right() - half);
        let (top, bottom) = (rect.y + half, rect.bottom() - half);
        // The horizontal edges take the band's center line, from edge to edge.
        self.line(
            Vec2::new(rect.x, top),
            Vec2::new(rect.right(), top),
            thickness,
            color,
        );
        self.line(
            Vec2::new(rect.x, bottom),
            Vec2::new(rect.right(), bottom),
            thickness,
            color,
        );
        self.line(
            Vec2::new(left, top),
            Vec2::new(left, bottom),
            thickness,
            color,
        );
        self.line(
            Vec2::new(right, top),
            Vec2::new(right, bottom),
            thickness,
            color,
        );
    }

    /// Draw a filled triangle.
    ///
    /// The triangle rides the quad pipeline as a quad whose last corner repeats
    /// the first, so the second triangle has no area and the call needs no
    /// second primitive.
    pub fn triangle(&mut self, a: Vec2, b: Vec2, c: Vec2, color: Color) {
        self.fill(Quad::triangle(a, b, c), color);
    }

    /// Draw a filled circle of `radius` around `center`, as a fan of sixteen
    /// triangles.
    ///
    /// The debug crosshair is the only caller, so sixteen sides read as round
    /// at a six-pixel radius.
    pub fn circle(&mut self, center: Vec2, radius: f32, color: Color) {
        const SIDES: usize = 16;
        for i in 0..SIDES {
            let a = std::f32::consts::TAU * i as f32 / SIDES as f32;
            let b = std::f32::consts::TAU * (i + 1) as f32 / SIDES as f32;
            self.triangle(
                center,
                center + Vec2::new(a.cos(), a.sin()) * radius,
                center + Vec2::new(b.cos(), b.sin()) * radius,
                color,
            );
        }
    }

    /// Draw a texture over a rect, turned by `angle` degrees around the point
    /// `origin` inside it.
    ///
    /// This is raylib's `DrawTexturePro` mapping, so a call site keeps the
    /// geometry it had: `dest` names the rect, `origin` names the pivot inside
    /// it, and a zero angle draws the plain rect.
    pub fn texture_pro(
        &mut self,
        texture: &Texture,
        src: Rect,
        dest: Rect,
        turn: Turn,
        tint: Color,
        filter: Filter,
    ) {
        self.texture(texture, src, Quad::pro(dest, turn), tint, filter);
    }

    /// Draw a texture over a quad with one filter.
    pub fn texture(
        &mut self,
        texture: &Texture,
        src: Rect,
        quad: Quad,
        tint: Color,
        filter: Filter,
    ) {
        let inst = Inst::textured(quad, src, texture.dims(), tint);
        self.push(texture.group(filter).clone(), inst);
    }

    /// Append one instance, merging it into the previous call when the bind
    /// group is the same.
    fn push(&mut self, group: Arc<wgpu::BindGroup>, inst: Inst) {
        if let Some(call) = self.calls.last_mut()
            && Arc::ptr_eq(&call.group, &group)
        {
            call.len += 1;
            self.insts.push(inst);
            return;
        }
        let start = self.insts.len() as u32;
        self.insts.push(inst);
        self.calls.push(DrawCall {
            group,
            start,
            len: 1,
        });
    }

    /// Draw the batch into the next swapchain image and present it.
    ///
    /// # Errors
    ///
    /// Returns [`FrameError`] when no image came out of the swapchain, which the
    /// caller answers by skipping the frame.
    pub fn flush(&mut self, gpu: &Gpu) -> Result<(), FrameError> {
        let frame = gpu.acquire();
        match frame {
            Ok(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                self.render_into(gpu.device(), gpu.queue(), &view);
                // wgpu 30 hands the present to the queue, so the frame value
                // travels here instead of presenting itself.
                gpu.queue().present(frame);
                Ok(())
            }
            Err(err) => {
                // A skipped frame must not leave its batch behind, or the next
                // frame would draw those quads a second time.
                self.insts.clear();
                self.calls.clear();
                Err(err)
            }
        }
    }

    /// Paint the batch into any texture view and submit it.
    ///
    /// This is the one place that encodes the frame: [`Self::flush`] passes the
    /// swapchain image of the window, and a host with no window passes a texture
    /// of its own, so a test can read the pixels back without a display.
    /// An empty batch still clears the target, so a blank frame and an occluded
    /// one look the same.
    pub fn render_into(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        self.upload_instances(device, queue);
        let clear = self.clear.to_unit();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vv encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vv frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear[0],
                            g: clear[1],
                            b: clear[2],
                            a: clear[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !self.calls.is_empty() {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.globals_group, &[]);
                pass.set_vertex_buffer(0, self.instances.slice(..));
                for call in &self.calls {
                    pass.set_bind_group(1, call.group.as_ref(), &[]);
                    pass.draw(0..6, call.start..call.start + call.len);
                }
            }
        }
        queue.submit(Some(encoder.finish()));
    }

    /// Write this frame's instances, growing the buffer first when it is
    /// short.
    fn upload_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let count = u32::try_from(self.insts.len()).unwrap_or(u32::MAX);
        if count > self.capacity {
            let capacity = count.next_power_of_two();
            self.instances = instance_buffer(device, capacity);
            self.capacity = capacity;
        }
        if !self.insts.is_empty() {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.insts));
        }
    }
}

/// The viewport uniform, its layout and its bind group.
fn globals(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
    use wgpu::util::DeviceExt as _;
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vv viewport"),
        contents: bytemuck::cast_slice(&[0.0_f32, 0.0, 0.0, 0.0]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vv globals"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vv globals"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, layout, group)
}

/// The texture layout: one sampler, one texture view.
fn texture_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vv texture"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

/// The pipeline layout: the viewport uniform, then the texture.
fn pipeline_layout(
    device: &wgpu::Device,
    globals_layout: &wgpu::BindGroupLayout,
    texture_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vv layout"),
        bind_group_layouts: &[Some(globals_layout), Some(texture_layout)],
        immediate_size: 0,
    })
}

/// Straight alpha, like raylib.
///
/// The decoder writes non-premultiplied bytes, so the color source factor stays
/// `SRC_ALPHA` and the alpha channel keeps one source of one, which is what a
/// non-premultiplied target expects.
fn blend() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// Compile the one pipeline.
fn pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("vv pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vmain"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(Inst::layout())],
        },
        primitive: wgpu::PrimitiveState {
            // No culling, so a quad draws in either winding and a veil band
            // never has to reverse its corners.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fmain"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend()),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// The instance buffer, sized for `capacity` quads.
fn instance_buffer(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vv instances"),
        size: u64::from(capacity) * std::mem::size_of::<Inst>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the asserted values are exact window arithmetic
mod tests {
    use super::*;

    /// The instance stride keeps every attribute inside its alignment.
    #[test]
    fn the_instance_stride_is_sixty_four_bytes() {
        assert_eq!(std::mem::size_of::<Inst>(), 64);
    }

    /// Every attribute offset names the field it claims to.
    ///
    /// This test exists because a reorder of the fields once moved the tint
    /// while the attribute table still named the old offset, and the picture
    /// then drew fully transparent with no error anywhere.
    #[test]
    fn the_attribute_offsets_match_the_fields() {
        let offsets = Inst::ATTRIBS.map(|attribute| attribute.offset);
        let fields = [
            std::mem::offset_of!(Inst, p) as u64,
            std::mem::offset_of!(Inst, p) as u64 + 8,
            std::mem::offset_of!(Inst, p) as u64 + 16,
            std::mem::offset_of!(Inst, p) as u64 + 24,
            std::mem::offset_of!(Inst, src) as u64,
            std::mem::offset_of!(Inst, tint) as u64,
        ];
        assert_eq!(offsets, fields);
    }

    /// A textured instance normalizes its source rect to the texture.
    #[test]
    fn a_textured_instance_normalizes_its_source() {
        let quad = Quad::rect(Rect::new(0.0, 0.0, 10.0, 20.0));
        let inst = Inst::textured(
            quad,
            Rect::new(5.0, 10.0, 20.0, 40.0),
            (40, 80),
            Color::WHITE,
        );
        assert_eq!(inst.src, [0.125, 0.125, 0.625, 0.625]);
        assert_eq!(inst.p[0], [0.0, 0.0]);
        assert_eq!(inst.p[2], [10.0, 20.0]);
        assert_eq!(inst.tint, [1.0, 1.0, 1.0, 1.0]);
    }

    /// A solid instance covers the quad and keeps the tint as straight alpha.
    #[test]
    fn a_solid_instance_carries_the_tint() {
        let inst = Inst::solid(
            Quad::rect(Rect::new(1.0, 2.0, 3.0, 4.0)),
            Color::new(0, 51, 255, 128),
        );
        assert_eq!(inst.src, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(inst.p[1], [4.0, 2.0]);
        assert!((inst.tint[2] - 1.0).abs() < 1e-6);
        assert!((inst.tint[3] - 128.0 / 255.0).abs() < 1e-6);
    }
}
