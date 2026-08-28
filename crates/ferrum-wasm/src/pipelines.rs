use wgpu::{Device, RenderPipeline, TextureFormat};

pub struct RenderPipelines {
    pub instanced_circle: RenderPipeline,
    pub instanced_rect: RenderPipeline,
    pub mesh: RenderPipeline,
    pub textured: RenderPipeline,
    /// Second pipeline with additive blend state (src + dst, no alpha attenuation).
    /// Selected per-batch when `MarkBatch.blend == BlendMode::Additive`.
    pub instanced_circle_additive: RenderPipeline,
    pub instanced_rect_additive: RenderPipeline,
    pub uniform_bgl: wgpu::BindGroupLayout,
    pub texture_bgl: wgpu::BindGroupLayout,
}

impl RenderPipelines {
    pub fn new(device: &Device, format: TextureFormat, sample_count: u32) -> Self {
        let circle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("circle.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/circle.wgsl").into()),
        });
        let rect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rect.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rect.wgsl").into()),
        });
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh.wgsl").into()),
        });
        let textured_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("textured.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/textured.wgsl").into()),
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let alpha_blend = Some(wgpu::BlendState::ALPHA_BLENDING);
        let additive_blend = Some(additive_blend_state());

        let instanced_circle = build_instanced_pipeline(
            device,
            &circle_shader,
            format,
            &uniform_bgl,
            alpha_blend,
            &circle_instance_layout(),
            sample_count,
        );
        let instanced_rect = build_instanced_pipeline(
            device,
            &rect_shader,
            format,
            &uniform_bgl,
            alpha_blend,
            &rect_instance_layout(),
            sample_count,
        );
        let mesh = build_mesh_pipeline(
            device,
            &mesh_shader,
            format,
            &uniform_bgl,
            alpha_blend,
            sample_count,
        );
        let textured = build_textured_pipeline(
            device,
            &textured_shader,
            format,
            &uniform_bgl,
            &texture_bgl,
            alpha_blend,
            sample_count,
        );

        let instanced_circle_additive = build_instanced_pipeline(
            device,
            &circle_shader,
            format,
            &uniform_bgl,
            additive_blend,
            &circle_instance_layout(),
            sample_count,
        );
        let instanced_rect_additive = build_instanced_pipeline(
            device,
            &rect_shader,
            format,
            &uniform_bgl,
            additive_blend,
            &rect_instance_layout(),
            sample_count,
        );

        Self {
            instanced_circle,
            instanced_rect,
            mesh,
            textured,
            instanced_circle_additive,
            instanced_rect_additive,
            uniform_bgl,
            texture_bgl,
        }
    }
}

const QUAD_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 8,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x2,
    }],
};

fn circle_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    // center(2) + radius(1) + fill(4) + stroke(4) + stroke_w(1) + opacity(1)
    // + stroke_opacity(1) + stroke_dash(1) + angle(1) = 16 floats = 64 bytes
    wgpu::VertexBufferLayout {
        array_stride: 16 * 4,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            }, // center
            wgpu::VertexAttribute {
                offset: 8,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32,
            }, // radius
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            }, // fill_color
            wgpu::VertexAttribute {
                offset: 28,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32x4,
            }, // stroke_color
            wgpu::VertexAttribute {
                offset: 44,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32,
            }, // stroke_width
            wgpu::VertexAttribute {
                offset: 48,
                shader_location: 6,
                format: wgpu::VertexFormat::Float32,
            }, // opacity
            wgpu::VertexAttribute {
                offset: 52,
                shader_location: 7,
                format: wgpu::VertexFormat::Float32,
            }, // stroke_opacity
            wgpu::VertexAttribute {
                offset: 56,
                shader_location: 8,
                format: wgpu::VertexFormat::Float32,
            }, // stroke_dash
            wgpu::VertexAttribute {
                offset: 60,
                shader_location: 9,
                format: wgpu::VertexFormat::Float32,
            }, // angle
        ],
    }
}

fn rect_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    // pos(2) + size(2) + corner_r(1) + fill(4) + stroke(4) + stroke_w(1) + opacity(1)
    // + stroke_opacity(1) + stroke_dash(1) + angle(1) = 18 floats = 72 bytes
    wgpu::VertexBufferLayout {
        array_stride: 18 * 4,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            }, // position
            wgpu::VertexAttribute {
                offset: 8,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x2,
            }, // size
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32,
            }, // corner_radius
            wgpu::VertexAttribute {
                offset: 20,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32x4,
            }, // fill_color
            wgpu::VertexAttribute {
                offset: 36,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32x4,
            }, // stroke_color
            wgpu::VertexAttribute {
                offset: 52,
                shader_location: 6,
                format: wgpu::VertexFormat::Float32,
            }, // stroke_width
            wgpu::VertexAttribute {
                offset: 56,
                shader_location: 7,
                format: wgpu::VertexFormat::Float32,
            }, // opacity
            wgpu::VertexAttribute {
                offset: 60,
                shader_location: 8,
                format: wgpu::VertexFormat::Float32,
            }, // stroke_opacity
            wgpu::VertexAttribute {
                offset: 64,
                shader_location: 9,
                format: wgpu::VertexFormat::Float32,
            }, // stroke_dash
            wgpu::VertexAttribute {
                offset: 68,
                shader_location: 10,
                format: wgpu::VertexFormat::Float32,
            }, // angle
        ],
    }
}

/// Additive blend state: src + dst with no alpha attenuation.
///
/// Used for `mark_raster(blend="additive")` GPU compositing. Per-batch
/// pipeline selection: raster batches with `BlendMode::Additive` use this
/// pipeline; all others use `wgpu::BlendState::ALPHA_BLENDING`.
///
/// Spec §4 / §5 / §8: implemented as a second `wgpu::RenderPipeline` —
/// NOT a post-process pass, NOT a fragment shader hack.
pub fn additive_blend_state() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

fn build_instanced_pipeline(
    device: &Device,
    shader: &wgpu::ShaderModule,
    format: TextureFormat,
    uniform_bgl: &wgpu::BindGroupLayout,
    blend: Option<wgpu::BlendState>,
    instance_layout: &wgpu::VertexBufferLayout<'_>,
    sample_count: u32,
) -> RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(uniform_bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[QUAD_LAYOUT, instance_layout.clone()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    })
}

fn build_mesh_pipeline(
    device: &Device,
    shader: &wgpu::ShaderModule,
    format: TextureFormat,
    uniform_bgl: &wgpu::BindGroupLayout,
    blend: Option<wgpu::BlendState>,
    sample_count: u32,
) -> RenderPipeline {
    // FA-16: affine-invariant stroke width.
    // Layout: position(2) + normal(2) + half_width(1) + color(4) = 9 floats = 36 bytes.
    // Attribute locations must match the @location annotations in mesh.wgsl and the
    // field order of MeshVertex in tessellate.rs.
    let mesh_vertex_layout = wgpu::VertexBufferLayout {
        array_stride: 9 * 4, // 36 bytes
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }, // position
            wgpu::VertexAttribute {
                offset: 8,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            }, // normal
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32,
            }, // half_width
            wgpu::VertexAttribute {
                offset: 20,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            }, // color
        ],
    };
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(uniform_bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[mesh_vertex_layout],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    })
}

fn build_textured_pipeline(
    device: &Device,
    shader: &wgpu::ShaderModule,
    format: TextureFormat,
    uniform_bgl: &wgpu::BindGroupLayout,
    texture_bgl: &wgpu::BindGroupLayout,
    blend: Option<wgpu::BlendState>,
    sample_count: u32,
) -> RenderPipeline {
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: 4 * 4, // position(2) + tex_coord(2) = 4 floats
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: 8,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            },
        ],
    };
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(uniform_bgl), Some(texture_bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    })
}
