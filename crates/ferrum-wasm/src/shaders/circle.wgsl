// Uniforms layout (2 x vec4 = 32 bytes):
//   canvas.xy    = canvas width, height
//   transform    = {sx, sy, tx, ty}  (identity = 1,1,0,0)
//
// The former `clip` vec4 has been removed. Fragment-level clip tests were
// redundant: mark instances are clipped by the GPU scissor rect set per draw
// command in render_frame; full-canvas clip on mesh was always a no-op.
struct Uniforms {
    canvas: vec4<f32>,
    transform: vec4<f32>,
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
    @location(8) stroke_dash: f32,  // palette index as float: 0=solid, 1..7 index DASH_PATTERNS below
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
};

// Stroke-dash pattern table: on/off segment lengths per non-solid palette
// index, zero-padded to 4 components. Mirrors ferrum_scene::DASH_PALETTE (a
// Rust test parses this declaration and asserts equality against it, so the
// two cannot silently drift). Index 0 (solid) has no entry.
const DASH_PATTERNS: array<vec4<f32>, 7> = array<vec4<f32>, 7>(
    vec4<f32>(6.0, 3.0, 0.0, 0.0),
    vec4<f32>(2.0, 3.0, 0.0, 0.0),
    vec4<f32>(6.0, 3.0, 2.0, 3.0),
    vec4<f32>(4.0, 2.0, 0.0, 0.0),
    vec4<f32>(1.0, 3.0, 0.0, 0.0),
    vec4<f32>(4.0, 2.0, 1.0, 2.0),
    vec4<f32>(8.0, 4.0, 0.0, 0.0),
);

// Dash visibility at `dist_along` (distance measured along the stroke's
// perimeter, never radially) for palette index `dash_idx`. Index 0 is
// always visible (solid).
fn dash_visible_along(dist_along: f32, dash_idx: i32) -> f32 {
    if dash_idx == 0 {
        return 1.0;
    }
    let pat = DASH_PATTERNS[clamp(dash_idx, 1, 7) - 1];
    let period = pat.x + pat.y + pat.z + pat.w;
    let phase = fract(dist_along / period) * period;
    let c1 = pat.x;
    let c2 = c1 + pat.y;
    let c3 = c2 + pat.z;
    var visible = phase < c1;
    if pat.z > 0.0 {
        visible = visible || (phase >= c2 && phase < c3);
    }
    return select(0.0, 1.0, visible);
}

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

    // Apply per-panel affine transform to the center only, so zoom
    // repositions marks without scaling their visual size (radius, stroke).
    let sx = u.transform.x; let sy = u.transform.y;
    let tx = u.transform.z; let ty = u.transform.w;
    let center_tx = vec2<f32>(in.center.x * sx + tx, in.center.y * sy + ty);
    let px = center_tx + rotated_quad * extent;
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

        // Stroke dash pattern: select based on palette index. stroke_dash
        // encodes a palette index (0=solid, 1..7 index DASH_PATTERNS above).
        // Phase is measured along the stroke's perimeter (arc length =
        // angle x radius), never radially -- a dash must alternate around
        // the ring, not concentrically inward from it.
        let dash_idx = i32(clamp(floor(in.stroke_dash + 0.5), 0.0, 7.0));
        let ang = atan2(in.local_pos.y, in.local_pos.x);
        let ang01 = select(ang, ang + 6.283185307179586, ang < 0.0);
        let dist_along = ang01 * in.radius;
        let dash_visible = dash_visible_along(dist_along, dash_idx);

        color = mix(color, sc * dash_visible, stroke_alpha * dash_visible);
    }
    color.a *= in.opacity;
    if color.a < 0.001 { discard; }
    return color;
}
