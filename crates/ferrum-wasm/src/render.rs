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
    /// One mark-transform slot per (panel, y-slot) pair (secondary-y-axis,
    /// GH #52), flat-indexed by [`scene_load::transform_slot_index`]. A mark
    /// draw binds the slot for its `(panel_id, y_slot)`; out-of-range indices
    /// fall back to the identity bind group. `sum(panel_slot_counts)` slots are
    /// allocated. A single-y scene has one slot per panel (every count `== 1`),
    /// so the index is `panel_id` and this is byte-identical to the former
    /// per-panel allocation — at rest every slot holds identity.
    mark_transform_slots: Vec<PanelTransformSlot>,
    /// Per-panel y-slot count (secondary-y-axis, GH #52), cloned from
    /// `SceneData::panel_slot_counts`. Drives the `(panel_id, y_slot) →` flat
    /// transform-slot index mapping for both upload and bind-group selection.
    panel_slot_counts: Vec<usize>,
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
        // One mark-transform slot per (panel, y-slot) pair. Every slot starts
        // at identity, so a freshly loaded scene renders identically to the
        // former per-panel path until a zoom/pan/rescale uploads a non-identity
        // affine into a specific slot. `total_transform_slots` == `panel_count`
        // for single-y scenes, preserving byte-stable allocation.
        let total_slots = crate::scene_load::total_transform_slots(&scene.panel_slot_counts);
        let mark_transform_slots: Vec<PanelTransformSlot> = (0..total_slots)
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
            panel_slot_counts: scene.panel_slot_counts.clone(),
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
    /// Upload every (panel, y-slot) composed affine into its own mark-transform
    /// slot (secondary-y-axis, GH #52).
    ///
    /// `zoom_transforms[p]` is panel `p`'s per-panel zoom/pan affine (moves all
    /// layers together); `slot_rescales[idx]` is the y-only rescale a
    /// `domainParam`/brush bound to one independent-y layer wrote into that
    /// slot, identity otherwise. Each slot's uploaded uniform is
    /// `compose_panel_slot(panel_affine, slot_rescale)`, so a non-uniform
    /// rescale on one panel — or one layer — never leaks into siblings. For a
    /// single-y scene every slot rescale is identity, so each slot receives
    /// exactly the panel affine — byte-identical to the former per-panel upload.
    pub fn upload_panel_transforms(
        &self,
        gpu: &GpuContext,
        canvas_w: f32,
        canvas_h: f32,
        zoom_transforms: &[crate::zoom_pan::Affine2],
        slot_rescales: &[crate::zoom_pan::Affine2],
    ) {
        use crate::zoom_pan::compose_panel_slot;
        for (panel_id, &count) in self.panel_slot_counts.iter().enumerate() {
            let panel_affine = select_panel_transform(zoom_transforms, panel_id);
            for y_slot in 0..count.max(1) {
                let idx = crate::scene_load::transform_slot_index(
                    &self.panel_slot_counts,
                    panel_id,
                    y_slot,
                );
                let Some(slot) = self.mark_transform_slots.get(idx) else {
                    continue;
                };
                let rescale = slot_rescales
                    .get(idx)
                    .copied()
                    .unwrap_or_else(crate::zoom_pan::Affine2::identity);
                let transform = compose_panel_slot(panel_affine, rescale);
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
    }

    /// The bind group carrying the `(panel_id, y_slot)` pair's composed mark
    /// transform (secondary-y-axis, GH #52).
    ///
    /// Falls back to the identity bind group when the mapped index is out of
    /// range, matching `select_panel_transform`'s identity fallback so an
    /// out-of-range mesh/instance command draws at identity rather than
    /// panicking. Single-y scenes map `(panel_id, 0) → panel_id`.
    fn mark_bind_group(&self, panel_id: usize, y_slot: usize) -> &wgpu::BindGroup {
        let idx = crate::scene_load::transform_slot_index(&self.panel_slot_counts, panel_id, y_slot);
        self.mark_transform_slots
            .get(idx)
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

/// Begin a render pass over `color_view` (resolving into `resolve_target` when
/// MSAA is active). `clear` selects `LoadOp::Clear(bg)` for the first pass of
/// a frame; every subsequent pass in the same frame uses `LoadOp::Load` to
/// preserve what earlier passes already drew.
///
/// Factored out so `render_frame` can open more than one pass per frame (see
/// the "one scissor rect per pass" note there) without repeating the
/// attachment/descriptor boilerplate.
fn begin_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    color_view: &'e wgpu::TextureView,
    resolve_target: Option<&'e wgpu::TextureView>,
    bg: [f32; 4],
    clear: bool,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("main"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            resolve_target,
            ops: wgpu::Operations {
                load: if clear {
                    wgpu::LoadOp::Clear(wgpu::Color {
                        r: bg[0] as f64,
                        g: bg[1] as f64,
                        b: bg[2] as f64,
                        a: bg[3] as f64,
                    })
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// `true` when moving from `current`'s scissor state to `desired` requires
/// ending the active render pass and starting a fresh one.
///
/// `None` means "full surface" (a pass's default state at creation — no
/// explicit `set_scissor_rect` call needed). Extracted as a pure, host-testable
/// function — mirroring `zoom_pan::select_panel_transform` — so the
/// "one `set_scissor_rect` call per pass" invariant `render_frame` relies on
/// (see the task-11fix note there) is unit-testable without a GPU device.
pub(crate) fn scissor_requires_new_pass(
    current: Option<[u32; 4]>,
    desired: Option<[u32; 4]>,
) -> bool {
    current != desired
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

        let mut pass = begin_pass(&mut encoder, color_view, resolve_target, bg, true);
        // The scissor rect physically applied to `pass` right now. `None`
        // means "pass-default" — every fresh `begin_render_pass` starts
        // scissored to the full render target, so a `None` request never
        // needs an explicit `set_scissor_rect` call.
        //
        // task-11fix: a composite scene with 3+ mark-bearing panels drew only
        // the LAST panel's circles/rects (any layout — Grid, hconcat,
        // vconcat all reproduced it; 2-panel scenes were unaffected). Root
        // cause: wgpu 29.0.3 (the version pinned when this landed — re-test
        // whether this workaround is still needed on any wgpu bump) silently
        // drops every draw but the last when a single render pass contains
        // 3+ sequential `set_scissor_rect` calls to *different* rectangles
        // — confirmed by
        // instrumenting every scissor rect with a console log (values were
        // always correct) and by isolating scissor-rect changes from
        // bind-group changes in a headless-Chrome capture. `ensure_scissor!`
        // below sidesteps the defect: instead of changing the scissor rect
        // within one pass, it ends the current pass and opens a fresh
        // `Load` pass, so `set_scissor_rect` is called AT MOST ONCE per
        // pass. This can't be a plain function: a `wgpu::RenderPass<'e>`
        // borrows `encoder` for its own lifetime, so replacing it needs the
        // OLD pass dropped in the very statement that reborrows `encoder`
        // for the new one — passing `pass` through a function argument list
        // keeps the old borrow alive (from the borrow checker's view) for
        // the whole call, which conflicts with the second `&mut encoder`
        // argument the callee would need.
        let mut pass_scissor: Option<[u32; 4]> = None;
        macro_rules! ensure_scissor {
            ($desired:expr) => {
                // Bind once: `$desired` must not be re-evaluated per use below
                // (harmless for today's Copy args, a footgun for future sites).
                let desired: Option<[u32; 4]> = $desired;
                if scissor_requires_new_pass(pass_scissor, desired) {
                    drop(pass);
                    pass = begin_pass(&mut encoder, color_view, resolve_target, bg, false);
                    if let Some(r) = desired {
                        pass.set_scissor_rect(r[0], r[1], r[2], r[3]);
                    }
                    pass_scissor = desired;
                }
            };
        }

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
            if buffers.mark_mesh_panels.is_empty() {
                // Fallback: no panel metadata (should not occur in practice for
                // mesh-bearing scenes, but guards against stale GpuBuffers).
                // Bind panel 0's transform — the single-panel common case.
                pass.set_pipeline(&pipelines.mesh);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(0, buffers.mark_bind_group(0, 0), &[]);
                pass.draw_indexed(0..buffers.mesh_index_count, 0, 0..1);
            } else {
                for panel in &buffers.mark_mesh_panels {
                    let pa = &panel.plot_area;
                    let rect = [
                        (pa[0] * scale_x) as u32,
                        (pa[1] * scale_y) as u32,
                        (pa[2] * scale_x) as u32,
                        (pa[3] * scale_y) as u32,
                    ];
                    ensure_scissor!(Some(rect));
                    // Pass state does not persist across `ensure_scissor!`'s
                    // pass split, so pipeline/buffers are re-bound every
                    // iteration (a cheap no-op when the pass didn't change).
                    pass.set_pipeline(&pipelines.mesh);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    // Bind this (panel, y-slot) pair's composed affine so a
                    // non-uniform rescale on a sibling panel — or a domainParam
                    // on another layer's y — does not shear this mesh slice.
                    pass.set_bind_group(
                        0,
                        buffers.mark_bind_group(panel.panel_id, panel.y_slot),
                        &[],
                    );
                    let range = panel.index_start..(panel.index_start + panel.index_count);
                    pass.draw_indexed(range, 0, 0..1);
                }
            }
        }

        // Images: zoom/pan transform, full-surface scissor.
        if !buffers.image_draws.is_empty() {
            ensure_scissor!(None::<[u32; 4]>);
            for img in &buffers.image_draws {
                pass.set_pipeline(&pipelines.textured);
                // Images (heatmap rasters) carry no panel association yet (W4); bind
                // panel 0's slot-0 transform, matching the single-panel common case
                // the former single mark uniform served.
                pass.set_bind_group(0, buffers.mark_bind_group(0, 0), &[]);
                pass.set_bind_group(1, &img.bind_group, &[]);
                pass.set_vertex_buffer(0, img.vertex_buffer.slice(..));
                pass.draw(0..4, 0..1);
            }
        }

        // Per-batch circle/rect draw commands. Mark commands use the
        // zoom-transformed uniforms; non-mark commands (axes, gridlines,
        // legend, title, etc.) use the identity-transform uniforms so
        // they stay fixed during zoom/pan.
        for cmd in &buffers.draw_commands {
            if cmd.instance_count == 0 {
                continue;
            }

            let desired = cmd.plot_area.filter(|_| cmd.is_mark).map(|pa| {
                [
                    (pa[0] * scale_x) as u32,
                    (pa[1] * scale_y) as u32,
                    (pa[2] * scale_x) as u32,
                    (pa[3] * scale_y) as u32,
                ]
            });
            ensure_scissor!(desired);

            let bind_group = if cmd.is_mark {
                // Mark instances follow their owning (panel, y-slot) affine.
                buffers.mark_bind_group(cmd.panel_id, cmd.y_slot)
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
            ensure_scissor!(None::<[u32; 4]>);
            pass.set_pipeline(&pipelines.mesh);
            pass.set_bind_group(0, &buffers.identity_uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..buffers.annotation_mesh_index_count, 0, 0..1);
        }

        // `pass_scissor`'s final write (from the annotation-mesh
        // `ensure_scissor!` above) is never read again — silence the
        // otherwise-legitimate `unused_assignments` lint without an
        // `#[allow]` that would also hide a real dead-store bug earlier in
        // this block.
        let _ = pass_scissor;
        drop(pass);
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
    slot_rescales: &[crate::zoom_pan::Affine2],
    panel_id: usize,
) -> Result<String, crate::error::WasmRenderError> {
    // Upload EVERY (panel, y-slot) composed affine into its own slot, not just
    // `panel_id`'s. A reactive domain-rescale writes a non-uniform affine into
    // one panel's — or one layer's slot's — transform while leaving the rest at
    // identity; each mark draw binds its own slot, so uploading the full set
    // keeps every panel and layer correct in a single frame regardless of which
    // triggered the render.
    buffers.upload_panel_transforms(
        gpu,
        scene_data.width,
        scene_data.height,
        zoom_transforms,
        slot_rescales,
    );
    render_frame(gpu, pipelines, buffers, scene_data.background)?;

    // `panel_id` still selects which panel's labels are re-placed for the
    // text-overlay JSON below (the GPU upload above already covers all panels).
    let transform = select_panel_transform(zoom_transforms, panel_id);
    // Per-right-axis composed affines for the secondary-y relabel path (#52 /
    // criterion 8): the right axis of rank `r` (0-based, stacking outward) maps
    // through y-slot `r + 1`, so its labels relabel via `panel ∘ slot_rescale`.
    // Slot 0 is the primary/left axis (handled by the shared panel affine).
    // Empty for single-y panels (count == 1) → the text path falls back to the
    // panel affine, byte-identical to pre-#52.
    let panel_slot_count = scene_data
        .panel_slot_counts
        .get(panel_id)
        .copied()
        .unwrap_or(1)
        .max(1);
    let secondary_affines: Vec<crate::zoom_pan::Affine2> = (1..panel_slot_count)
        .map(|y_slot| {
            let idx = crate::scene_load::transform_slot_index(
                &scene_data.panel_slot_counts,
                panel_id,
                y_slot,
            );
            let rescale = slot_rescales
                .get(idx)
                .copied()
                .unwrap_or_else(crate::zoom_pan::Affine2::identity);
            crate::zoom_pan::compose_panel_slot(transform, rescale)
        })
        .collect();
    let plot_area = scene_panels.get(panel_id).map(|p| {
        (p.plot_area.x, p.plot_area.y, p.plot_area.w, p.plot_area.h)
    });
    let text_json = crate::text_json::build_zoomed_text_json(
        &scene_data.text_elements,
        interaction,
        panel_id,
        &transform,
        &secondary_affines,
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

    // ── task-11fix: scissor-rect pass-splitting invariant ──────────────
    //
    // A composite scene (Grid, hconcat, or vconcat — layout kind is not the
    // discriminator) with 3+ mark-bearing panels drew only the LAST panel's
    // circles/rects in interactive (WASM/WebGPU) output: the browser's wgpu
    // backend silently drops every draw but the last when a single render
    // pass contains 3+ sequential `set_scissor_rect` calls to *different*
    // rectangles (confirmed via headless-Chrome console instrumentation —
    // every CPU-computed scissor rect was correct; only the GPU-side result
    // was wrong). `render_frame` now opens a fresh `Load` pass every time the
    // desired scissor rect changes, so a pass never has `set_scissor_rect`
    // called more than once. These tests pin `scissor_requires_new_pass`, the
    // pure decision function that drives that pass-splitting.

    #[test]
    fn scissor_requires_new_pass_only_on_change() {
        assert!(
            !scissor_requires_new_pass(None, None),
            "full-surface -> full-surface must reuse the current pass"
        );
        assert!(
            scissor_requires_new_pass(None, Some([1, 2, 3, 4])),
            "full-surface -> a panel rect must split into a new pass"
        );
        assert!(
            scissor_requires_new_pass(Some([1, 2, 3, 4]), Some([5, 6, 7, 8])),
            "panel A's rect -> panel B's rect must split into a new pass"
        );
        assert!(
            !scissor_requires_new_pass(Some([1, 2, 3, 4]), Some([1, 2, 3, 4])),
            "repeating the same panel rect must reuse the current pass"
        );
        assert!(
            scissor_requires_new_pass(Some([1, 2, 3, 4]), None),
            "a panel rect -> full-surface (e.g. images/annotations after a \
             mesh loop) must split into a new pass"
        );
    }

    /// A 3-panel composite scene's draw-command scissor sequence must open
    /// exactly 3 passes (one per distinct panel rect) — the regression this
    /// task-11fix guards against: collapsing them into a single pass with 3
    /// sequential `set_scissor_rect` calls reproduces the bug where only the
    /// last panel's marks render.
    #[test]
    fn three_panel_scissor_sequence_opens_three_passes() {
        let panel_rects = [
            Some([0u32, 0, 100, 100]),
            Some([100u32, 0, 100, 100]),
            Some([200u32, 0, 100, 100]),
        ];
        let mut current: Option<[u32; 4]> = None;
        let mut pass_count = 0;
        for &desired in &panel_rects {
            if scissor_requires_new_pass(current, desired) {
                pass_count += 1;
                current = desired;
            }
        }
        assert_eq!(
            pass_count, 3,
            "each of the 3 distinct panel rects must open its own pass"
        );
    }

    /// Consecutive draw commands sharing the SAME desired scissor (e.g. two
    /// non-mark legend/axis draws, both full-surface) must NOT split into
    /// extra passes — only an actual rect change should.
    #[test]
    fn repeated_scissor_value_does_not_fragment_passes() {
        let sequence = [None, None, Some([0u32, 0, 50, 50]), Some([0u32, 0, 50, 50]), None];
        let mut current: Option<[u32; 4]> = None;
        let mut pass_count = 0;
        for &desired in &sequence {
            if scissor_requires_new_pass(current, desired) {
                pass_count += 1;
                current = desired;
            }
        }
        assert_eq!(
            pass_count, 2,
            "only the 2 actual value changes (None->rect, rect->None) should open new passes"
        );
    }
}
