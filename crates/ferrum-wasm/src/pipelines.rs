use wgpu::{Device, RenderPipeline, TextureFormat};

pub struct RenderPipelines {
    pub instanced_circle: RenderPipeline,
    pub instanced_rect: RenderPipeline,
    pub mesh: RenderPipeline,
    pub textured: RenderPipeline,
    pub uniform_bgl: wgpu::BindGroupLayout,
    pub texture_bgl: wgpu::BindGroupLayout,
}

impl RenderPipelines {
    pub fn new(device: &Device, format: TextureFormat) -> Self {
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

        let uniform_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform_bgl"),
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

        let texture_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let blend = Some(wgpu::BlendState::ALPHA_BLENDING);

        let instanced_circle =
            build_instanced_pipeline(device, &circle_shader, format, &uniform_bgl, blend, &circle_instance_layout());
        let instanced_rect =
            build_instanced_pipeline(device, &rect_shader, format, &uniform_bgl, blend, &rect_instance_layout());
        let mesh = build_mesh_pipeline(device, &mesh_shader, format, &uniform_bgl, blend);
        let textured =
            build_textured_pipeline(device, &textured_shader, format, &uniform_bgl, &texture_bgl, blend);

        Self {
            instanced_circle,
            instanced_rect,
            mesh,
            textured,
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
    // center(2) + radius(1) + fill(4) + stroke(4) + stroke_w(1) + opacity(1) = 13 floats
    wgpu::VertexBufferLayout {
        array_stride: 13 * 4,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute { offset: 0,  shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
            wgpu::VertexAttribute { offset: 8,  shader_location: 2, format: wgpu::VertexFormat::Float32 },
            wgpu::VertexAttribute { offset: 12, shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 28, shader_location: 4, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 44, shader_location: 5, format: wgpu::VertexFormat::Float32 },
            wgpu::VertexAttribute { offset: 48, shader_location: 6, format: wgpu::VertexFormat::Float32 },
        ],
    }
}

fn rect_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    // pos(2) + size(2) + corner_r(1) + fill(4) + stroke(4) + stroke_w(1) + opacity(1) = 15 floats
    wgpu::VertexBufferLayout {
        array_stride: 15 * 4,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute { offset: 0,  shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
            wgpu::VertexAttribute { offset: 8,  shader_location: 2, format: wgpu::VertexFormat::Float32x2 },
            wgpu::VertexAttribute { offset: 16, shader_location: 3, format: wgpu::VertexFormat::Float32 },
            wgpu::VertexAttribute { offset: 20, shader_location: 4, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 36, shader_location: 5, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 52, shader_location: 6, format: wgpu::VertexFormat::Float32 },
            wgpu::VertexAttribute { offset: 56, shader_location: 7, format: wgpu::VertexFormat::Float32 },
        ],
    }
}

fn build_instanced_pipeline(
    device: &Device,
    shader: &wgpu::ShaderModule,
    format: TextureFormat,
    uniform_bgl: &wgpu::BindGroupLayout,
    blend: Option<wgpu::BlendState>,
    instance_layout: &wgpu::VertexBufferLayout<'_>,
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
        multisample: wgpu::MultisampleState::default(),
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
) -> RenderPipeline {
    let mesh_vertex_layout = wgpu::VertexBufferLayout {
        array_stride: 6 * 4, // position(2) + color(4) = 6 floats
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
            wgpu::VertexAttribute { offset: 8,  shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
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
        multisample: wgpu::MultisampleState::default(),
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
) -> RenderPipeline {
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: 4 * 4, // position(2) + tex_coord(2) = 4 floats
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
            wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
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
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}
