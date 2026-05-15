// Uniforms layout (3 × vec4 = 48 bytes):
//   canvas.xy    = canvas width, height
//   transform    = {sx, sy, tx, ty}  (identity = 1,1,0,0)
//   clip         = {clip_x, clip_y, clip_w, clip_h}
struct Uniforms {
    canvas: vec4<f32>,
    transform: vec4<f32>,
    clip: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
    @location(1) center: vec2<f32>,
    @location(2) radius: f32,
    @location(3) fill_color: vec4<f32>,
    @location(4) stroke_color: vec4<f32>,
    @location(5) stroke_width: f32,
    @location(6) opacity: f32,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) fill_color: vec4<f32>,
    @location(2) stroke_color: vec4<f32>,
    @location(3) stroke_width: f32,
    @location(4) radius: f32,
    @location(5) opacity: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let extent = in.radius + in.stroke_width + 1.0;
    let px_local = in.center + in.quad_pos * extent;
    // Apply per-panel affine transform: transform.xy = (sx, sy), transform.zw = (tx, ty).
    let sx = u.transform.x; let sy = u.transform.y;
    let tx = u.transform.z; let ty = u.transform.w;
    let px = vec2<f32>(px_local.x * sx + tx, px_local.y * sy + ty);
    let ndc = vec2<f32>(
        px.x / u.canvas.x * 2.0 - 1.0,
        1.0 - px.y / u.canvas.y * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.local_pos = in.quad_pos * extent;
    out.fill_color = in.fill_color;
    out.stroke_color = in.stroke_color;
    out.stroke_width = in.stroke_width;
    out.radius = in.radius;
    out.opacity = in.opacity;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.local_pos);
    let sdf = dist - in.radius;
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
