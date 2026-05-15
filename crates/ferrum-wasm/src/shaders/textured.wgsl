struct Uniforms {
    canvas_w: f32,
    canvas_h: f32,
    sx: f32,
    sy: f32,
    tx: f32,
    ty: f32,
    clip_x: f32,
    clip_y: f32,
    clip_w: f32,
    clip_h: f32,
    _pad: array<f32, 6>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Apply per-panel affine transform.
    let px = vec2<f32>(in.position.x * u.sx + u.tx, in.position.y * u.sy + u.ty);
    let ndc = vec2<f32>(
        px.x / u.canvas_w * 2.0 - 1.0,
        1.0 - px.y / u.canvas_h * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.tex_coord = in.tex_coord;
    return out;
}

@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coord);
}
