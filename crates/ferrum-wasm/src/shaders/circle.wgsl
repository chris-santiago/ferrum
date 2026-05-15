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
    @location(1) center: vec2<f32>,
    @location(2) radius: f32,
    @location(3) fill_color: vec4<f32>,
    @location(4) stroke_color: vec4<f32>,
    @location(5) stroke_width: f32,
    @location(6) opacity: f32,
    @location(7) stroke_opacity: f32,
    @location(8) stroke_dash: f32,  // palette index as float: 0=solid,1=dashed,2=dotted,3=dash-dot
    @location(9) angle: f32,         // screen-space rotation degrees around circle center
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) fill_color: vec4<f32>,
    @location(2) stroke_color: vec4<f32>,
    @location(3) stroke_width: f32,
    @location(4) radius: f32,
    @location(5) opacity: f32,
    @location(6) stroke_opacity: f32,
    @location(7) stroke_dash: f32,
    // angle is consumed entirely in the vertex stage (no per-fragment use)
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let extent = in.radius + in.stroke_width + 1.0;

    // Apply per-instance rotation around the circle center.
    // angle is in degrees; convert to radians.
    let rad = in.angle * (3.14159265358979 / 180.0);
    let cos_a = cos(rad);
    let sin_a = sin(rad);
    let rotated_quad = vec2<f32>(
        in.quad_pos.x * cos_a - in.quad_pos.y * sin_a,
        in.quad_pos.x * sin_a + in.quad_pos.y * cos_a,
    );

    let px_local = in.center + rotated_quad * extent;
    // Apply per-panel affine transform: transform.xy = (sx, sy), transform.zw = (tx, ty).
    let sx = u.transform.x; let sy = u.transform.y;
    let tx = u.transform.z; let ty = u.transform.w;
    let px = vec2<f32>(px_local.x * sx + tx, px_local.y * sy + ty);
    let ndc = vec2<f32>(
        px.x / u.canvas.x * 2.0 - 1.0,
        1.0 - px.y / u.canvas.y * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    // local_pos uses the un-rotated quad_pos so the SDF stays axis-aligned.
    out.local_pos = in.quad_pos * extent;
    out.fill_color = in.fill_color;
    out.stroke_color = in.stroke_color;
    out.stroke_width = in.stroke_width;
    out.radius = in.radius;
    out.opacity = in.opacity;
    out.stroke_opacity = in.stroke_opacity;
    out.stroke_dash = in.stroke_dash;
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
        // Apply stroke_opacity to the stroke color alpha channel independently.
        var sc = in.stroke_color;
        sc.a *= in.stroke_opacity;

        // Stroke dash pattern: select based on palette index.
        // stroke_dash encodes a palette index (0=solid, 1=dashed 6/3, 2=dotted 2/3, 3=dash-dot 6/3/2/3).
        // We approximate dashing in screen-space using the distance along the circle perimeter.
        // For GPU efficiency, we use a simple angular modulation on the SDF distance.
        let dash_idx = i32(clamp(floor(in.stroke_dash + 0.5), 0.0, 3.0));
        var dash_visible = 1.0;
        if dash_idx == 1 {
            // Dashed: 6 on, 3 off — period = 9
            let period = 9.0;
            let on_frac = 6.0 / period;
            let phase = fract(dist / period);
            dash_visible = select(0.0, 1.0, phase < on_frac);
        } else if dash_idx == 2 {
            // Dotted: 2 on, 3 off — period = 5
            let period = 5.0;
            let on_frac = 2.0 / period;
            let phase = fract(dist / period);
            dash_visible = select(0.0, 1.0, phase < on_frac);
        } else if dash_idx == 3 {
            // Dash-dot: 6 on, 3 off, 2 on, 3 off — period = 14
            let period = 14.0;
            let phase = fract(dist / period) * period;
            dash_visible = select(0.0, 1.0,
                phase < 6.0 || (phase >= 9.0 && phase < 11.0));
        }
        // dash_visible = 0 for solid (always visible)
        if dash_idx == 0 { dash_visible = 1.0; }

        color = mix(color, sc * dash_visible, stroke_alpha * dash_visible);
    }
    color.a *= in.opacity;
    if color.a < 0.001 { discard; }
    return color;
}
