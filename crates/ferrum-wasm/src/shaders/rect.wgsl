struct Uniforms {
    viewport: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
    @location(1) rect_pos: vec2<f32>,
    @location(2) rect_size: vec2<f32>,
    @location(3) corner_radius: f32,
    @location(4) fill_color: vec4<f32>,
    @location(5) stroke_color: vec4<f32>,
    @location(6) stroke_width: f32,
    @location(7) opacity: f32,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) half_size: vec2<f32>,
    @location(2) corner_radius: f32,
    @location(3) fill_color: vec4<f32>,
    @location(4) stroke_color: vec4<f32>,
    @location(5) stroke_width: f32,
    @location(6) opacity: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pad = in.stroke_width + 1.0;
    let center = in.rect_pos + in.rect_size * 0.5;
    let half = in.rect_size * 0.5 + pad;
    let px = center + in.quad_pos * half;
    let ndc = vec2<f32>(
        px.x / u.viewport.x * 2.0 - 1.0,
        1.0 - px.y / u.viewport.y * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.local_pos = in.quad_pos * half;
    out.half_size = in.rect_size * 0.5;
    out.corner_radius = in.corner_radius;
    out.fill_color = in.fill_color;
    out.stroke_color = in.stroke_color;
    out.stroke_width = in.stroke_width;
    out.opacity = in.opacity;
    return out;
}

fn sdf_rounded_rect(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half_size + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sdf = sdf_rounded_rect(in.local_pos, in.half_size, in.corner_radius);
    let fill_alpha = 1.0 - smoothstep(-0.5, 0.5, sdf);
    var color = in.fill_color * fill_alpha;
    if in.stroke_width > 0.0 {
        let stroke_sdf = abs(sdf + in.stroke_width * 0.5) - in.stroke_width * 0.5;
        let stroke_alpha = 1.0 - smoothstep(-0.5, 0.5, stroke_sdf);
        color = mix(color, in.stroke_color, stroke_alpha);
    }
    color.a *= in.opacity;
    if color.a < 0.001 { discard; }
    return color;
}
