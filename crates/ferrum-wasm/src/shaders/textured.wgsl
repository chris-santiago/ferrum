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
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) scene_pos: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Apply per-panel affine transform.
    let sx = u.transform.x; let sy = u.transform.y;
    let tx = u.transform.z; let ty = u.transform.w;
    let px = vec2<f32>(in.position.x * sx + tx, in.position.y * sy + ty);
    let ndc = vec2<f32>(
        px.x / u.canvas.x * 2.0 - 1.0,
        1.0 - px.y / u.canvas.y * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.tex_coord = in.tex_coord;
    // Pass pre-transform position for fragment-stage panel clipping.
    out.scene_pos = in.position;
    return out;
}

@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Clip to panel boundaries (scene-space, pre-transform).
    if (in.scene_pos.x < u.clip.x || in.scene_pos.x > u.clip.x + u.clip.z ||
        in.scene_pos.y < u.clip.y || in.scene_pos.y > u.clip.y + u.clip.w) {
        discard;
    }
    return textureSample(t_diffuse, s_diffuse, in.tex_coord);
}
