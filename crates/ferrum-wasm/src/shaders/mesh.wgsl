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

// FA-16: affine-invariant stroke width, correct under non-uniform affine.
//
// Strokes store the centerline in `position` and the pre-affine offset components
// (`normal`, `half_width`) separately. Lyon emits the normal in scene space; under
// a non-uniform affine (sx != sy) the on-screen line direction rotates, so the raw
// scene normal is no longer perpendicular to the screen edge. We must transform the
// normal by the affine's inverse-transpose before offsetting.
//
// For an affine with linear part diag(sx, sy):
//   inverse-transpose direction = normalize( vec2(nx/sx, ny/sy) )
//                               = normalize( vec2(sy*nx, sx*ny) )   (multiply form)
//
// At uniform scale (sx == sy == k):
//   normalize(vec2(k*nx, k*ny)) == normalize(vec2(nx, ny))
//   so screen_dir * (half_width * miter) == normal * half_width  -- identical to old code.
//
// At identity (sx=sy=1): same reduction -- byte-identical behavior.
//
// Under non-uniform (sx != sy): direction is corrected to screen-perpendicular;
// magnitude stays half_width * miter, so width is constant in screen pixels.
//
// miter = length(in.normal): lyon extends |normal| > 1 at miter joins to preserve
// the correct corner geometry; we preserve that factor in the magnitude.
//
// Zero-normal guard: the `half_width > 1e-4` gate already excludes fills (half_width=0).
// For strokes, lyon guarantees non-zero normals on valid paths. If a degenerate
// zero-normal did reach here, WGSL normalize(vec2(0,0)) returns vec2(0,0) per spec,
// so the offset is zero rather than NaN -- safe.
//
// Fills set normal=[0,0] and half_width=0; the `select` produces no offset and
// fills render exactly as before.
//
// Attribute layout (must match pipelines.rs and MeshVertex in tessellate.rs):
//   @location(0) position   — Float32x2, offset 0
//   @location(1) normal     — Float32x2, offset 8
//   @location(2) half_width — Float32x1, offset 16
//   @location(3) color      — Float32x4, offset 20
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) half_width: f32,
    @location(3) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Apply per-panel affine transform to the centerline (or fill position).
    let sx = u.transform.x; let sy = u.transform.y;
    let tx = u.transform.z; let ty = u.transform.w;
    let px = vec2<f32>(in.position.x * sx + tx, in.position.y * sy + ty);
    // Apply the stroke width offset in screen space (after the affine), corrected
    // for non-uniform scale via the inverse-transpose normal-transform rule.
    // Fills have half_width == 0 and the select produces no offset.
    let nx = in.normal.x;
    let ny = in.normal.y;
    // Inverse-transpose direction: normalize(vec2(nx/sx, ny/sy)) == normalize(vec2(sy*nx, sx*ny)).
    let screen_dir = normalize(vec2<f32>(sy * nx, sx * ny));
    // Preserve lyon's miter-join factor (|normal| > 1 at sharp corners).
    let miter = length(in.normal);
    let offset = screen_dir * (in.half_width * miter);
    let final_px = px + select(vec2<f32>(0.0, 0.0), offset, in.half_width > 1e-4);
    let ndc = vec2<f32>(
        final_px.x / u.canvas.x * 2.0 - 1.0,
        1.0 - final_px.y / u.canvas.y * 2.0,
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
