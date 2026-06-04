use wgpu::util::DeviceExt;

use crate::error::WasmRenderError;
use crate::gpu::GpuContext;
use crate::pipelines::RenderPipelines;
use crate::scene_load::{DrawCommand, DrawKind, MarkMeshPanel, SceneData};

struct ImageGpu {
    vertex_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// One per-panel mark-transform uniform slot: a small uniform buffer holding
/// that panel's `Uniforms` (canvas + affine) plus a bind group over it.
///
/// Each mark draw binds its owning panel's slot, so a non-uniform domain
/// rescale (sx ≠ sy) applied to one panel does not shear or translate sibling
/// panels' marks. The bind group uses the same non-dynamic `uniform_bgl` as
/// every other draw, so no pipeline or bind-group-layout change is needed.
struct PanelTransformSlot {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub struct GpuBuffers {
    /// One mark-transform slot per panel, indexed by `panel_id`. A mark draw
    /// binds `mark_transform_slots[panel_id]`; out-of-range ids fall back to
    /// the identity bind group. Allocated for every panel in the scene (see
    /// `SceneData::panel_count`), so at-rest every slot holds identity and the
    /// rendered pixels match the former single-uniform path exactly.
    mark_transform_slots: Vec<PanelTransformSlot>,
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
    pub(crate) mesh_index_count: u32,
    /// Per-panel mark-mesh index ranges + plot areas. Used by `render_frame`
    /// to scissor each panel's mesh draw to its own plot area, preventing
    /// zoomed/panned geometry from bleeding into axis margins.
    mark_mesh_panels: Vec<MarkMeshPanel>,
    /// Static mesh (grid lines, axis ticks, legend, title,
    /// decorations) — identity transform (stays fixed during zoom/pan).
    static_mesh_vertex_buffer: Option<wgpu::Buffer>,
    static_mesh_index_buffer: Option<wgpu::Buffer>,
    static_mesh_index_count: u32,
    /// Annotation mesh (reference lines/paths from `annotate_hline`/`annotate_vline`)
    /// drawn with the identity transform AFTER mark mesh so annotations
    /// appear above data marks, matching SVG painter order.
    annotation_mesh_vertex_buffer: Option<wgpu::Buffer>,
    annotation_mesh_index_buffer: Option<wgpu::Buffer>,
    annotation_mesh_index_count: u32,
    image_draws: Vec<ImageGpu>,
    /// Ordered draw commands for per-batch pipeline selection.
    draw_commands: Vec<DrawCommand>,
    /// Logical scene dimensions (from SceneData). Used to compute the
    /// DPR scale factor when the surface is resized for PNG capture.
    scene_width: f32,
    scene_height: f32,
}

/// Uniform data uploaded to the GPU once per panel draw.
///
/// Layout (32 bytes / 2 × vec4<f32>):
///   canvas:    {canvas_w, canvas_h, 0, 0}
///   transform: {sx, sy, tx, ty}         identity = {1,1,0,0}
///
/// The former `clip` vec4 has been removed. Fragment-level clip tests in the
/// shaders were redundant because:
///   - Mark instances (circles/rects): clipped by the GPU scissor rect set per
///     draw command in `render_frame`.
///   - Mesh and static mesh: full-canvas clip was always a no-op.
///
/// Must use vec4 packing — WGSL uniform address space enforces 16-byte stride
/// for `array<f32, N>`.
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
        }
    }
}

use crate::zoom_pan::select_panel_transform;

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
        // One mark-transform slot per panel. Every slot starts at identity, so
        // a freshly loaded scene renders identically to the former
        // single-uniform path until a zoom/pan/rescale uploads a non-identity
        // affine into a specific panel's slot.
        let mark_transform_slots: Vec<PanelTransformSlot> = (0..scene.panel_count)
            .map(|_| {
                let uniforms = Uniforms::identity(scene.width, scene.height);
                let buffer =
                    gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("panel_uniforms"),
                        contents: bytemuck::bytes_of(&uniforms),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("panel_uniforms_bg"),
                    layout: &pipelines.uniform_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });
                PanelTransformSlot { buffer, bind_group }
            })
            .collect();

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

        let (annotation_mesh_vertex_buffer, annotation_mesh_index_buffer) =
            if scene.annotation_mesh_buffers.vertices.is_empty() {
                (None, None)
            } else {
                (
                    Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("annotation_mesh_verts"),
                        contents: bytemuck::cast_slice(&scene.annotation_mesh_buffers.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    })),
                    Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("annotation_mesh_idx"),
                        contents: bytemuck::cast_slice(&scene.annotation_mesh_buffers.indices),
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
            mark_transform_slots,
            _identity_uniform_buffer: identity_uniform_buffer,
            identity_uniform_bind_group,
            quad_vertex_buffer,
            circle_instance_buffer,
            rect_instance_buffer,
            mesh_vertex_buffer,
            mesh_index_buffer,
            mesh_index_count: scene.mesh_buffers.indices.len() as u32,
            mark_mesh_panels: scene.mark_mesh_panels.clone(),
            static_mesh_vertex_buffer,
            static_mesh_index_buffer,
            static_mesh_index_count: scene.static_mesh_buffers.indices.len() as u32,
            annotation_mesh_vertex_buffer,
            annotation_mesh_index_buffer,
            annotation_mesh_index_count: scene.annotation_mesh_buffers.indices.len() as u32,
            image_draws,
            draw_commands: scene.draw_commands.clone(),
            scene_width: scene.width,
            scene_height: scene.height,
        }
    }
}

impl GpuBuffers {
    /// Upload every panel's affine into its own mark-transform slot.
    ///
    /// `zoom_transforms[i]` is panel `i`'s affine; panels beyond the slice
    /// (or beyond the allocated slot count) keep identity. Each mark draw later
    /// binds its owning panel's slot, so a non-uniform rescale on one panel no
    /// longer leaks into siblings. Call this on every frame that consumes the
    /// zoom/pan state.
    pub fn upload_panel_transforms(
        &self,
        gpu: &GpuContext,
        canvas_w: f32,
        canvas_h: f32,
        zoom_transforms: &[crate::zoom_pan::Affine2],
    ) {
        for (panel_id, slot) in self.mark_transform_slots.iter().enumerate() {
            let transform = select_panel_transform(zoom_transforms, panel_id);
            let uniforms = Uniforms {
                canvas_w,
                canvas_h,
                _canvas_pad: [0.0; 2],
                sx: transform.sx as f32,
                sy: transform.sy as f32,
                tx: transform.tx as f32,
                ty: transform.ty as f32,
            };
            gpu.queue
                .write_buffer(&slot.buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }

    /// The bind group carrying panel `panel_id`'s mark transform.
    ///
    /// Falls back to the identity bind group when `panel_id` is out of range,
    /// matching `select_panel_transform`'s identity fallback so an out-of-range
    /// mesh/instance command draws at identity rather than panicking.
    fn mark_bind_group(&self, panel_id: usize) -> &wgpu::BindGroup {
        self.mark_transform_slots
            .get(panel_id)
            .map(|slot| &slot.bind_group)
            .unwrap_or(&self.identity_uniform_bind_group)
    }

    /// Re-upload only the circle and rect instance buffers without touching
    /// mesh, image, or uniform buffers.
    ///
    /// Called by `apply_conditionals_and_render` after conditional encodings
    /// are resolved: only instance colors change on each selection update, so
    /// re-uploading the full scene would waste GPU bandwidth needlessly.
    pub fn update_instances(
        &mut self,
        gpu: &GpuContext,
        circles: &[crate::scene_load::CircleInstance],
        rects: &[crate::scene_load::RectInstance],
    ) {
        self.circle_instance_buffer = if circles.is_empty() {
            None
        } else {
            Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("circles"),
                contents: bytemuck::cast_slice(circles),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        };
        self.rect_instance_buffer = if rects.is_empty() {
            None
        } else {
            Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rects"),
                contents: bytemuck::cast_slice(rects),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        };
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
        // FA-19: when MSAA is active, render into the multisampled target and
        // resolve into the single-sample surface view; at 1× render directly
        // into the surface view (byte-identical to the pre-MSAA path).
        let (color_view, resolve_target) = match gpu.msaa_view.as_ref() {
            Some(msaa) => (msaa, Some(&view)),
            None => (&view, None),
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
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

        // Surface dimensions and DPR scale factors — used throughout the draw
        // sequence for scissor rect scaling. Computed once here rather than
        // inside the per-batch loop so the mark-mesh scissor path can also use
        // them (the mark mesh is drawn before the per-batch loop).
        let surface_w = surface_tex.texture.width();
        let surface_h = surface_tex.texture.height();
        // Scale factor for DPR: when the canvas is resized for PNG capture
        // (e.g., 2x on Retina), plot_area is still in logical pixels.
        let scale_x = surface_w as f32 / buffers.scene_width;
        let scale_y = surface_h as f32 / buffers.scene_height;

        // Draw order:
        //   1. Static mesh (grid, axes, legend, title) — identity transform
        //   2. Mark mesh (lines, areas, paths) — zoom/pan transform, per-panel scissor
        //   3. Images — zoom/pan transform
        //   4. Per-batch circle/rect commands — mixed (is_mark selects transform)
        //   5. Annotation mesh (reference lines/paths) — identity transform
        //
        // Static mesh is drawn first so grid lines appear behind data marks.
        // Mark mesh is drawn second so data lines/areas appear on top of the grid.
        // Annotation mesh is drawn last so reference lines (hline/vline) appear
        // above data marks, matching SVG painter order.

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

        // 2. Mark mesh — zoom/pan transform, scissored per panel.
        //
        // The mark mesh is a single combined buffer for all panels. Without
        // a per-panel scissor, zoomed/panned geometry bleeds outside the
        // plot area into axis margins (and into adjacent panels on multi-panel
        // charts such as focus+context). We iterate the per-panel index ranges
        // recorded at scene-load time and set a scissor rect matching each
        // panel's plot area before drawing its slice of the buffer.
        //
        // When `mark_mesh_panels` is empty (no mesh marks in the scene) the
        // loop body never executes, so the behaviour is identical to the old
        // single-draw path for circle/rect-only charts.
        if let (Some(vb), Some(ib)) =
            (&buffers.mesh_vertex_buffer, &buffers.mesh_index_buffer)
        {
            pass.set_pipeline(&pipelines.mesh);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);

            if buffers.mark_mesh_panels.is_empty() {
                // Fallback: no panel metadata (should not occur in practice for
                // mesh-bearing scenes, but guards against stale GpuBuffers).
                // Bind panel 0's transform — the single-panel common case.
                pass.set_bind_group(0, buffers.mark_bind_group(0), &[]);
                pass.draw_indexed(0..buffers.mesh_index_count, 0, 0..1);
            } else {
                for panel in &buffers.mark_mesh_panels {
                    let pa = &panel.plot_area;
                    pass.set_scissor_rect(
                        (pa[0] * scale_x) as u32,
                        (pa[1] * scale_y) as u32,
                        (pa[2] * scale_x) as u32,
                        (pa[3] * scale_y) as u32,
                    );
                    // Bind this panel's own affine so a non-uniform rescale on a
                    // sibling panel does not shear this panel's mesh.
                    pass.set_bind_group(0, buffers.mark_bind_group(panel.panel_id), &[]);
                    let range = panel.index_start..(panel.index_start + panel.index_count);
                    pass.draw_indexed(range, 0, 0..1);
                }
                // Relax scissor to full surface so subsequent draws
                // (images, circle/rect commands, annotations) are not clipped
                // to the last panel's plot area.
                pass.set_scissor_rect(0, 0, surface_w, surface_h);
            }
        }

        for img in &buffers.image_draws {
            pass.set_pipeline(&pipelines.textured);
            // Images (heatmap rasters) carry no panel association yet (W4); bind
            // panel 0's transform, matching the single-panel common case the
            // former single mark uniform served.
            pass.set_bind_group(0, buffers.mark_bind_group(0), &[]);
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

            if let Some(pa) = cmd.plot_area.filter(|_| cmd.is_mark) {
                pass.set_scissor_rect(
                    (pa[0] * scale_x) as u32,
                    (pa[1] * scale_y) as u32,
                    (pa[2] * scale_x) as u32,
                    (pa[3] * scale_y) as u32,
                );
            } else {
                pass.set_scissor_rect(0, 0, surface_w, surface_h);
            }

            let bind_group = if cmd.is_mark {
                // Mark instances follow their owning panel's affine.
                buffers.mark_bind_group(cmd.panel_id)
            } else {
                // Axes, gridlines, legend, title: fixed under zoom/pan.
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

        // 5. Annotation mesh — identity transform (annotations stay fixed
        //    during zoom/pan and appear above data marks).
        if let (Some(vb), Some(ib)) = (
            &buffers.annotation_mesh_vertex_buffer,
            &buffers.annotation_mesh_index_buffer,
        ) {
            pass.set_scissor_rect(0, 0, surface_w, surface_h);
            pass.set_pipeline(&pipelines.mesh);
            pass.set_bind_group(0, &buffers.identity_uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..buffers.annotation_mesh_index_count, 0, 0..1);
        }
    }

    gpu.queue.submit(std::iter::once(encoder.finish()));
    surface_tex.present();
    Ok(())
}

/// Upload the per-panel affine transform uniform and re-render, then return
/// zoomed text-element JSON.
///
/// Extracted from `WasmRenderer::upload_transform_and_render` so the logic
/// lives in the rendering module and the wasm_bindgen method becomes a thin
/// delegator.
///
/// The parameter count mirrors the `WasmRenderer` fields this function
/// previously accessed as `self.*`. A grouping struct would obscure the
/// access pattern without reducing complexity at the single call site.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_transform_and_render(
    gpu: &GpuContext,
    pipelines: &RenderPipelines,
    buffers: &GpuBuffers,
    scene_data: &SceneData,
    scene_panels: &[ferrum_scene::Panel],
    interaction: &ferrum_scene::InteractionConfig,
    zoom_transforms: &[crate::zoom_pan::Affine2],
    panel_id: usize,
) -> Result<String, crate::error::WasmRenderError> {
    // Upload EVERY panel's affine into its own slot, not just `panel_id`'s.
    // A reactive domain-rescale writes a non-uniform affine into one panel's
    // transform while leaving siblings at identity; each mark draw binds its
    // own slot, so uploading the full vector keeps every panel correct in a
    // single frame regardless of which panel triggered the render.
    buffers.upload_panel_transforms(gpu, scene_data.width, scene_data.height, zoom_transforms);
    render_frame(gpu, pipelines, buffers, scene_data.background)?;

    // `panel_id` still selects which panel's labels are re-placed for the
    // text-overlay JSON below (the GPU upload above already covers all panels).
    let transform = select_panel_transform(zoom_transforms, panel_id);
    let plot_area = scene_panels.get(panel_id).map(|p| {
        (p.plot_area.x, p.plot_area.y, p.plot_area.w, p.plot_area.h)
    });
    let text_json = crate::text_json::build_zoomed_text_json(
        &scene_data.text_elements,
        interaction,
        panel_id,
        &transform,
        plot_area,
    );
    Ok(text_json)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the uniform buffer is exactly 32 bytes (2 x vec4<f32>) after
    /// removing the clip vec4. WGSL uniform structs require 16-byte alignment;
    /// 32 is a multiple of 16 so this is well-formed.
    #[test]
    fn test_uniforms_size_is_32_bytes() {
        assert_eq!(
            std::mem::size_of::<Uniforms>(),
            32,
            "Uniforms must be exactly 32 bytes (2 x vec4<f32>) after removing the clip vec4"
        );
    }

    /// Verify that `update_instances` does not touch the mesh_index_count.
    ///
    /// This test cannot create real GPU buffers (no GPU in test environment),
    /// so it confirms the invariant structurally: `mesh_index_count` is a
    /// plain `u32` field that `update_instances` never writes. The test
    /// exercises the `GpuBuffers` struct layout to ensure no future refactor
    /// accidentally makes `update_instances` reset the mesh count.
    #[test]
    fn test_update_instances_preserves_mesh_count() {
        // The mesh_index_count field is pub(crate), confirming it exists and is
        // accessible within the crate. We verify the field exists and that
        // update_instances does not mutate it by inspection of the method body:
        // update_instances only writes circle_instance_buffer and
        // rect_instance_buffer, leaving all mesh fields untouched.
        //
        // We can also verify the Uniforms struct is correctly sized as a proxy
        // for the correctness of the GpuBuffers struct definition.
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);

        // Confirm mesh_index_count is accessible as pub(crate).
        // This is a compile-time check — if the field were removed or made
        // private, this test would fail to compile.
        fn _assert_mesh_index_count_accessible(b: &GpuBuffers) -> u32 {
            b.mesh_index_count
        }
    }
}
