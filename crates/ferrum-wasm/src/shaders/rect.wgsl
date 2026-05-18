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
    // Instance attributes:
    @location(1) rect_pos: vec2<f32>,
    @location(2) rect_size: vec2<f32>,
    @location(3) corner_radius: f32,
    @location(4) fill_color: vec4<f32>,
    @location(5) stroke_color: vec4<f32>,
    @location(6) stroke_width: f32,
    @location(7) opacity: f32,
    @location(8) stroke_opacity: f32,
    @location(9) stroke_dash: f32,   // palette index as float
    @location(10) angle: f32,          // screen-space rotation degrees around rect center
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
    @location(7) stroke_opacity: f32,
    @location(8) stroke_dash: f32,
    @location(9) scene_pos: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pad = in.stroke_width + 1.0;
    let center = in.rect_pos + in.rect_size * 0.5;
    let half = in.rect_size * 0.5 + pad;

    // Apply per-instance rotation around the rect center.
    let rad = in.angle * (3.14159265358979 / 180.0);
    let cos_a = cos(rad);
    let sin_a = sin(rad);
    let rotated_quad = vec2<f32>(
        in.quad_pos.x * cos_a - in.quad_pos.y * sin_a,
        in.quad_pos.x * sin_a + in.quad_pos.y * cos_a,
    );

    let px_local = center + rotated_quad * half;
    // Apply per-panel affine transform.
    let sx = u.transform.x; let sy = u.transform.y;
    let tx = u.transform.z; let ty = u.transform.w;
    let px = vec2<f32>(px_local.x * sx + tx, px_local.y * sy + ty);
    let ndc = vec2<f32>(
        px.x / u.canvas.x * 2.0 - 1.0,
        1.0 - px.y / u.canvas.y * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    // local_pos uses the un-rotated quad_pos for SDF alignment.
    out.local_pos = in.quad_pos * half;
    out.half_size = in.rect_size * 0.5;
    out.corner_radius = in.corner_radius;
    out.fill_color = in.fill_color;
    out.stroke_color = in.stroke_color;
    out.stroke_width = in.stroke_width;
    out.opacity = in.opacity;
    out.stroke_opacity = in.stroke_opacity;
    out.stroke_dash = in.stroke_dash;
    // Pass pre-transform position for fragment-stage panel clipping.
    out.scene_pos = px_local;
    return out;
}

fn sdf_rounded_rect(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half_size + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Clip to panel boundaries (scene-space, pre-transform).
    if (in.scene_pos.x < u.clip.x || in.scene_pos.x > u.clip.x + u.clip.z ||
        in.scene_pos.y < u.clip.y || in.scene_pos.y > u.clip.y + u.clip.w) {
        discard;
    }
    let sdf = sdf_rounded_rect(in.local_pos, in.half_size, in.corner_radius);
    let fill_alpha = 1.0 - smoothstep(-0.5, 0.5, sdf);
    var color = in.fill_color * fill_alpha;
    if in.stroke_width > 0.0 {
        let stroke_sdf = abs(sdf + in.stroke_width * 0.5) - in.stroke_width * 0.5;
        let stroke_alpha = 1.0 - smoothstep(-0.5, 0.5, stroke_sdf);

        // Apply stroke_opacity to the stroke color alpha independently.
        var sc = in.stroke_color;
        sc.a *= in.stroke_opacity;

        // Stroke dash pattern: approximate in screen-space using the SDF distance.
        let dash_idx = i32(clamp(floor(in.stroke_dash + 0.5), 0.0, 3.0));
        let dist = abs(sdf);
        var dash_visible = 1.0;
        if dash_idx == 1 {
            let period = 9.0;
            let on_frac = 6.0 / period;
            let phase = fract(dist / period);
            dash_visible = select(0.0, 1.0, phase < on_frac);
        } else if dash_idx == 2 {
            let period = 5.0;
            let on_frac = 2.0 / period;
            let phase = fract(dist / period);
            dash_visible = select(0.0, 1.0, phase < on_frac);
        } else if dash_idx == 3 {
            let period = 14.0;
            let phase = fract(dist / period) * period;
            dash_visible = select(0.0, 1.0,
                phase < 6.0 || (phase >= 9.0 && phase < 11.0));
        }
        if dash_idx == 0 { dash_visible = 1.0; }

        color = mix(color, sc * dash_visible, stroke_alpha * dash_visible);
    }
    color.a *= in.opacity;
    if color.a < 0.001 { discard; }
    return color;
}
