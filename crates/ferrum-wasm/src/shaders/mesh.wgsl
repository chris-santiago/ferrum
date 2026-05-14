struct Uniforms {
    viewport: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let ndc = vec2<f32>(
        in.position.x / u.viewport.x * 2.0 - 1.0,
        1.0 - in.position.y / u.viewport.y * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.color.a < 0.001 { discard; }
    return in.color;
}
