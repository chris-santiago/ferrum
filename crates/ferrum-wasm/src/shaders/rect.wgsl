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
    @location(1) rect_pos: vec2<f32>,
    @location(2) rect_size: vec2<f32>,
    @location(3) corner_radius: f32,
    @location(4) fill_color: vec4<f32>,
    @location(5) stroke_color: vec4<f32>,
    @location(6) stroke_width: f32,
    @location(7) opacity: f32,
    @location(8) stroke_opacity: f32,
    @location(9) stroke_dash: f32,   // palette index as float: 0=solid, 1..7 index DASH_PATTERNS below
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

    // Apply per-panel affine transform to the center only, so zoom
    // repositions rects without scaling their visual size (width, height, stroke).
    let sx = u.transform.x; let sy = u.transform.y;
    let tx = u.transform.z; let ty = u.transform.w;
    let center_tx = vec2<f32>(center.x * sx + tx, center.y * sy + ty);
    let px = center_tx + rotated_quad * half;
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
    return out;
}

fn sdf_rounded_rect(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half_size + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

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

// Position along the rect's (unrounded) outline perimeter, walking
// clockwise from the top-left corner: top left-to-right, right
// top-to-bottom, bottom right-to-left, left bottom-to-top -- so a dash
// alternates around the perimeter instead of concentrically inward from
// it. `p` is in the same axis-aligned local space `sdf_rounded_rect` reads
// (origin at the rect center); `corner_radius` is intentionally not
// consulted here, so a rounded rect's dash phase treats its corners as
// sharp -- a visually minor approximation, not a perimeter-length error on
// the straight edges the ring samples land on.
fn rect_perimeter_pos(p: vec2<f32>, half_size: vec2<f32>) -> f32 {
    let hw = half_size.x;
    let hh = half_size.y;
    let d_top = abs(p.y + hh);
    let d_bottom = abs(p.y - hh);
    let d_left = abs(p.x + hw);
    let d_right = abs(p.x - hw);
    let peri_top_bottom = 2.0 * hw;
    let peri_left_right = 2.0 * hh;
    if min(d_top, d_bottom) <= min(d_left, d_right) {
        if d_top <= d_bottom {
            return clamp(p.x + hw, 0.0, peri_top_bottom);
        }
        return peri_top_bottom + peri_left_right + clamp(hw - p.x, 0.0, peri_top_bottom);
    }
    if d_right <= d_left {
        return peri_top_bottom + clamp(p.y + hh, 0.0, peri_left_right);
    }
    return peri_top_bottom + peri_left_right + peri_top_bottom
        + clamp(hh - p.y, 0.0, peri_left_right);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sdf = sdf_rounded_rect(in.local_pos, in.half_size, in.corner_radius);
    let fill_alpha = 1.0 - smoothstep(-0.5, 0.5, sdf);
    var color = in.fill_color * fill_alpha;
    if in.stroke_width > 0.0 {
        let stroke_sdf = abs(sdf + in.stroke_width * 0.5) - in.stroke_width * 0.5;
        let stroke_alpha = 1.0 - smoothstep(-0.5, 0.5, stroke_sdf);

        // Apply stroke_opacity to the stroke color alpha independently.
        var sc = in.stroke_color;
        sc.a *= in.stroke_opacity;

        // Stroke dash pattern: select based on palette index. stroke_dash
        // encodes a palette index (0=solid, 1..7 index DASH_PATTERNS above).
        // Phase is measured along the rect's outline perimeter, never
        // radially (never by distance from the edge) -- a dash must
        // alternate around the perimeter, not concentrically inward from it.
        let dash_idx = i32(clamp(floor(in.stroke_dash + 0.5), 0.0, 7.0));
        let dist_along = rect_perimeter_pos(in.local_pos, in.half_size);
        let dash_visible = dash_visible_along(dist_along, dash_idx);

        color = mix(color, sc * dash_visible, stroke_alpha * dash_visible);
    }
    color.a *= in.opacity;
    if color.a < 0.001 { discard; }
    return color;
}
