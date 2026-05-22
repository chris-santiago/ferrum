use wgpu::util::DeviceExt;

use crate::error::WasmRenderError;
use crate::gpu::GpuContext;
use crate::pipelines::RenderPipelines;
use crate::scene_load::{DrawCommand, DrawKind, SceneData};

struct ImageGpu {
    vertex_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub struct GpuBuffers {
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    /// Identity-transform uniform buffer used for non-mark elements (axes,
    /// gridlines, legend, title) so they stay fixed during zoom/pan.
    /// Prefixed with underscore because the field is only read indirectly
    /// through `identity_uniform_bind_group`, but must stay alive to keep
    /// the GPU buffer backing the bind group valid.
    _identity_uniform_buffer: wgpu::Buffer,
    identity_uniform_bind_group: wgpu::BindGroup,
    quad_vertex_buffer: wgpu::Buffer,
    circle_instance_buffer: Option<wgpu::Buffer>,
    rect_instance_buffer: Option<wgpu::Buffer>,
    /// Mark mesh (lines, areas, paths from mark batches) — zoom transform.
    mesh_vertex_buffer: Option<wgpu::Buffer>,
    mesh_index_buffer: Option<wgpu::Buffer>,
    mesh_index_count: u32,
    /// Static mesh (grid lines, axis ticks, annotations, legend, title,
    /// decorations) — identity transform (stays fixed during zoom/pan).
    static_mesh_vertex_buffer: Option<wgpu::Buffer>,
    static_mesh_index_buffer: Option<wgpu::Buffer>,
    static_mesh_index_count: u32,
    image_draws: Vec<ImageGpu>,
    /// Ordered draw commands for per-batch pipeline selection.
    draw_commands: Vec<DrawCommand>,
}

/// Uniform data uploaded to the GPU once per panel draw.
///
/// Layout (48 bytes / 3 × vec4<f32>):
///   canvas:    {canvas_w, canvas_h, 0, 0}
///   transform: {sx, sy, tx, ty}         identity = {1,1,0,0}
///   clip:      {clip_x, clip_y, clip_w, clip_h}
///
/// Must use vec4 packing — WGSL uniform address space enforces a 16-byte
/// stride for `array<f32, N>`, which would cause a size mismatch with the
/// Rust struct if individual padding arrays are used instead.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    // vec4: canvas_w, canvas_h, unused, unused
    pub canvas_w: f32,
    pub canvas_h: f32,
    pub _canvas_pad: [f32; 2],
    // vec4: sx, sy, tx, ty
    pub sx: f32,
    pub sy: f32,
    pub tx: f32,
    pub ty: f32,
    // vec4: clip_x, clip_y, clip_w, clip_h
    pub clip_x: f32,
    pub clip_y: f32,
    pub clip_w: f32,
    pub clip_h: f32,
}

impl Uniforms {
    pub fn identity(canvas_w: f32, canvas_h: f32) -> Self {
        Self {
            canvas_w,
            canvas_h,
            _canvas_pad: [0.0; 2],
            sx: 1.0,
            sy: 1.0,
            tx: 0.0,
            ty: 0.0,
            clip_x: 0.0,
            clip_y: 0.0,
            clip_w: canvas_w,
            clip_h: canvas_h,
        }
    }
}

const QUAD_VERTICES: [[f32; 2]; 4] = [
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0,  1.0],
];

impl GpuBuffers {
    pub fn from_scene(
        gpu: &GpuContext,
        pipelines: &RenderPipelines,
        scene: &SceneData,
    ) -> Self {
        let uniforms = Uniforms::identity(scene.width, scene.height);
        let uniform_buffer =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let uniform_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniforms_bg"),
            layout: &pipelines.uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Identity-transform buffer: always sx=1,sy=1,tx=0,ty=0.
        // Used for non-mark elements (axes, gridlines, legend, title) so
        // they stay fixed during zoom/pan.
        let identity_uniforms = Uniforms::identity(scene.width, scene.height);
        let identity_uniform_buffer =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("identity_uniforms"),
                contents: bytemuck::bytes_of(&identity_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let identity_uniform_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("identity_uniforms_bg"),
            layout: &pipelines.uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: identity_uniform_buffer.as_entire_binding(),
            }],
        });

        let quad_vertex_buffer =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quad"),
                contents: bytemuck::cast_slice(&QUAD_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let circle_instance_buffer = if scene.circle_instances.is_empty() {
            None
        } else {
            Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("circles"),
                contents: bytemuck::cast_slice(&scene.circle_instances),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        };

        let rect_instance_buffer = if scene.rect_instances.is_empty() {
            None
        } else {
            Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rects"),
                contents: bytemuck::cast_slice(&scene.rect_instances),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        };

        let (mesh_vertex_buffer, mesh_index_buffer) =
            if scene.mesh_buffers.vertices.is_empty() {
                (None, None)
            } else {
                (
                    Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mesh_verts"),
                        contents: bytemuck::cast_slice(&scene.mesh_buffers.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    })),
                    Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mesh_idx"),
                        contents: bytemuck::cast_slice(&scene.mesh_buffers.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    })),
                )
            };

        let (static_mesh_vertex_buffer, static_mesh_index_buffer) =
            if scene.static_mesh_buffers.vertices.is_empty() {
                (None, None)
            } else {
                (
                    Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("static_mesh_verts"),
                        contents: bytemuck::cast_slice(&scene.static_mesh_buffers.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    })),
                    Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("static_mesh_idx"),
                        contents: bytemuck::cast_slice(&scene.static_mesh_buffers.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    })),
                )
            };

        let image_draws = scene
            .image_quads
            .iter()
            .filter_map(|img| upload_image_quad(gpu, pipelines, img))
            .collect();

        Self {
            uniform_buffer,
            uniform_bind_group,
            _identity_uniform_buffer: identity_uniform_buffer,
            identity_uniform_bind_group,
            quad_vertex_buffer,
            circle_instance_buffer,
            rect_instance_buffer,
            mesh_vertex_buffer,
            mesh_index_buffer,
            mesh_index_count: scene.mesh_buffers.indices.len() as u32,
            static_mesh_vertex_buffer,
            static_mesh_index_buffer,
            static_mesh_index_count: scene.static_mesh_buffers.indices.len() as u32,
            image_draws,
            draw_commands: scene.draw_commands.clone(),
        }
    }
}

impl GpuBuffers {
    /// Upload a new uniform block and re-render the frame.
    pub fn upload_uniforms(&self, gpu: &GpuContext, uniforms: &Uniforms) {
        gpu.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }
}

fn upload_image_quad(
    gpu: &GpuContext,
    pipelines: &RenderPipelines,
    img: &crate::scene_load::ImageQuad,
) -> Option<ImageGpu> {
    if img.img_width == 0 || img.img_height == 0 {
        return None;
    }

    let size = wgpu::Extent3d {
        width: img.img_width,
        height: img.img_height,
        depth_or_array_layers: 1,
    };
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("image"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        texture.as_image_copy(),
        &img.data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * img.img_width),
            rows_per_image: Some(img.img_height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("img_bg"),
        layout: &pipelines.texture_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });

    // Quad vertices: position(2) + tex_coord(2), TriangleStrip
    let (x0, y0) = (img.x, img.y);
    let (x1, y1) = (img.x + img.w, img.y + img.h);
    #[rustfmt::skip]
    let vertices: [[f32; 4]; 4] = [
        [x0, y0, 0.0, 0.0],
        [x1, y0, 1.0, 0.0],
        [x0, y1, 0.0, 1.0],
        [x1, y1, 1.0, 1.0],
    ];
    let vertex_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("img_quad"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    Some(ImageGpu { vertex_buffer, bind_group })
}

pub fn render_frame(
    gpu: &GpuContext,
    pipelines: &RenderPipelines,
    buffers: &GpuBuffers,
    clear_color: Option<[f32; 4]>,
) -> Result<(), WasmRenderError> {
    let surface_tex = match gpu.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t)
        | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        other => {
            return Err(WasmRenderError::GpuInit(format!(
                "get_current_texture: {other:?}"
            )));
        }
    };
    let view = surface_tex
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let bg = clear_color.unwrap_or([1.0, 1.0, 1.0, 1.0]);

    let mut encoder =
        gpu.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: bg[0] as f64,
                        g: bg[1] as f64,
                        b: bg[2] as f64,
                        a: bg[3] as f64,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Draw order:
        //   1. Static mesh (grid, axes, annotations, legend, title) — identity transform
        //   2. Mark mesh (lines, areas, paths) — zoom/pan transform
        //   3. Images — zoom/pan transform
        //   4. Per-batch circle/rect commands — mixed (is_mark selects transform)
        //
        // Static mesh is drawn first so grid lines appear behind data marks.
        // Mark mesh is drawn second so data lines/areas appear on top of the grid.

        // 1. Static mesh — identity transform (stays fixed during zoom/pan)
        if let (Some(vb), Some(ib)) =
            (&buffers.static_mesh_vertex_buffer, &buffers.static_mesh_index_buffer)
        {
            pass.set_pipeline(&pipelines.mesh);
            pass.set_bind_group(0, &buffers.identity_uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..buffers.static_mesh_index_count, 0, 0..1);
        }

        // 2. Mark mesh — zoom/pan transform
        if let (Some(vb), Some(ib)) =
            (&buffers.mesh_vertex_buffer, &buffers.mesh_index_buffer)
        {
            pass.set_pipeline(&pipelines.mesh);
            pass.set_bind_group(0, &buffers.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..buffers.mesh_index_count, 0, 0..1);
        }

        for img in &buffers.image_draws {
            pass.set_pipeline(&pipelines.textured);
            pass.set_bind_group(0, &buffers.uniform_bind_group, &[]);
            pass.set_bind_group(1, &img.bind_group, &[]);
            pass.set_vertex_buffer(0, img.vertex_buffer.slice(..));
            pass.draw(0..4, 0..1);
        }

        // Per-batch circle/rect draw commands. Mark commands use the
        // zoom-transformed uniforms; non-mark commands (axes, gridlines,
        // legend, title, etc.) use the identity-transform uniforms so
        // they stay fixed during zoom/pan.
        for cmd in &buffers.draw_commands {
            if cmd.instance_count == 0 {
                continue;
            }
            let bind_group = if cmd.is_mark {
                &buffers.uniform_bind_group
            } else {
                &buffers.identity_uniform_bind_group
            };
            let start = cmd.instance_start;
            let end = start + cmd.instance_count;
            match cmd.kind {
                DrawKind::Rect => {
                    if let Some(ib) = &buffers.rect_instance_buffer {
                        let pipeline = if cmd.additive {
                            &pipelines.instanced_rect_additive
                        } else {
                            &pipelines.instanced_rect
                        };
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.set_vertex_buffer(0, buffers.quad_vertex_buffer.slice(..));
                        pass.set_vertex_buffer(1, ib.slice(..));
                        pass.draw(0..4, start..end);
                    }
                }
                DrawKind::Circle => {
                    if let Some(ib) = &buffers.circle_instance_buffer {
                        let pipeline = if cmd.additive {
                            &pipelines.instanced_circle_additive
                        } else {
                            &pipelines.instanced_circle
                        };
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.set_vertex_buffer(0, buffers.quad_vertex_buffer.slice(..));
                        pass.set_vertex_buffer(1, ib.slice(..));
                        pass.draw(0..4, start..end);
                    }
                }
            }
        }
    }

    gpu.queue.submit(std::iter::once(encoder.finish()));
    surface_tex.present();
    Ok(())
}
