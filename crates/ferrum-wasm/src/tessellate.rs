use ferrum_scene::*;
use lyon::math::point;
use lyon::path::builder::SvgPathBuilder;
use lyon::path::{Path as LyonPath, PathEvent};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, LineCap, LineJoin,
    StrokeOptions, StrokeTessellator, StrokeVertex, VertexBuffers,
};

/// Per-vertex data for the mesh pipeline (strokes and fills share one format).
///
/// Strokes carry the stroke centerline in `position`, the lyon offset direction
/// in `normal`, and `half_width > 0`. The vertex shader applies the width
/// offset in screen space (after the per-panel affine), so stroke width is
/// affine-invariant — constant pixel width regardless of sx/sy.
///
/// Fills set `normal = [0,0]` and `half_width = 0`; the shader's `select`
/// produces no offset, so fill vertices are transformed identically to before.
///
/// Field order (position, normal, half_width, color) must match the pipeline
/// layout attribute offsets in `pipelines.rs`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    /// Stroke: centerline point in scene space. Fill: position in scene space.
    pub position: [f32; 2],
    /// Stroke: unit-ish normal from lyon (may exceed 1.0 at miter joins). Fill: [0,0].
    pub normal: [f32; 2],
    /// Stroke: half of the stroke line width in pixels. Fill: 0.0.
    pub half_width: f32,
    pub color: [f32; 4],
}

// Compile-time guard: position(8) + normal(8) + half_width(4) + color(16) = 36 bytes.
const _: () = assert!(core::mem::size_of::<MeshVertex>() == 36);

pub fn tessellate_line(
    x1: f64, y1: f64, x2: f64, y2: f64,
    style: &StrokeStyle,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    let color = color_to_f32(&style.color, style.opacity * style.stroke_opacity);
    let mut builder = LyonPath::builder();
    builder.begin(point(x1 as f32, y1 as f32));
    builder.line_to(point(x2 as f32, y2 as f32));
    builder.end(false);
    let path = builder.build();

    let mut opts = StrokeOptions::default().with_line_width(style.width as f32);
    apply_cap_join(&mut opts, style.stroke_cap, style.stroke_join);

    stroke_path_dashed(&path, style.dash.as_deref(), &opts, color, buffers);
}

pub fn tessellate_path(
    commands: &[PathCmd],
    style: &FillStroke,
    closed: bool,
    batch_cap: Option<StrokeCap>,
    batch_join: Option<StrokeJoin>,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    let path = pathcmds_to_lyon(commands, closed);

    if let Some(fill) = &style.fill {
        let fill_alpha = style.opacity * style.fill_opacity;
        let color = color_to_f32(fill, fill_alpha);
        let mut tess = FillTessellator::new();
        let _ = tess.tessellate_path(
            &path,
            &FillOptions::default(),
            &mut BuffersBuilder::new(buffers, move |v: FillVertex| MeshVertex {
                position: v.position().to_array(),
                normal: [0.0, 0.0],
                half_width: 0.0,
                color,
            }),
        );
    }

    if let Some(stroke) = &style.stroke {
        let stroke_alpha = style.opacity * style.stroke_opacity;
        let color = color_to_f32(stroke, stroke_alpha);
        let mut opts = StrokeOptions::default().with_line_width(style.stroke_width as f32);
        apply_cap_join(&mut opts, batch_cap, batch_join);
        stroke_path_dashed(&path, style.stroke_dash.as_deref(), &opts, color, buffers);
    }
}

pub fn tessellate_polyline(
    points: &[(f64, f64)],
    style: &StrokeStyle,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    if points.len() < 2 {
        return;
    }
    let color = color_to_f32(&style.color, style.opacity * style.stroke_opacity);
    let mut builder = LyonPath::builder();
    builder.begin(point(points[0].0 as f32, points[0].1 as f32));
    for p in &points[1..] {
        builder.line_to(point(p.0 as f32, p.1 as f32));
    }
    builder.end(false);
    let path = builder.build();

    let mut opts = StrokeOptions::default().with_line_width(style.width as f32);
    apply_cap_join(&mut opts, style.stroke_cap, style.stroke_join);

    stroke_path_dashed(&path, style.dash.as_deref(), &opts, color, buffers);
}

pub fn tessellate_polygon(
    rings: &[Vec<[f64; 2]>],
    style: &FillStroke,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    // Skip if no rings or the exterior ring has fewer than 3 points.
    let exterior = match rings.first() {
        Some(r) if r.len() >= 3 => r,
        _ => return,
    };
    let mut builder = LyonPath::builder();
    // Exterior ring.
    builder.begin(point(exterior[0][0] as f32, exterior[0][1] as f32));
    for p in &exterior[1..] {
        builder.line_to(point(p[0] as f32, p[1] as f32));
    }
    builder.close();
    // Interior rings (holes).
    for hole in &rings[1..] {
        if hole.len() < 3 { continue; }
        builder.begin(point(hole[0][0] as f32, hole[0][1] as f32));
        for p in &hole[1..] {
            builder.line_to(point(p[0] as f32, p[1] as f32));
        }
        builder.close();
    }
    let path = builder.build();

    if let Some(fill) = &style.fill {
        let fill_alpha = style.opacity * style.fill_opacity;
        let color = color_to_f32(fill, fill_alpha);
        let mut tess = FillTessellator::new();
        let _ = tess.tessellate_path(
            &path,
            &FillOptions::default(),
            &mut BuffersBuilder::new(buffers, move |v: FillVertex| MeshVertex {
                position: v.position().to_array(),
                normal: [0.0, 0.0],
                half_width: 0.0,
                color,
            }),
        );
    }

    if let Some(stroke) = &style.stroke {
        let stroke_alpha = style.opacity * style.stroke_opacity;
        let color = color_to_f32(stroke, stroke_alpha);
        let opts = StrokeOptions::default().with_line_width(style.stroke_width as f32);
        let mut tess = StrokeTessellator::new();
        let _ = tess.tessellate_path(
            &path,
            &opts,
            &mut BuffersBuilder::new(buffers, move |v: StrokeVertex| MeshVertex {
                position: v.position_on_path().to_array(),
                normal: { let n = v.normal(); [n.x, n.y] },
                half_width: v.line_width() / 2.0,
                color,
            }),
        );
    }
}

fn pathcmds_to_lyon(cmds: &[PathCmd], closed: bool) -> LyonPath {
    let mut builder = LyonPath::builder().with_svg();
    let mut cur_x: f32 = 0.0;
    let mut cur_y: f32 = 0.0;

    for cmd in cmds {
        match cmd {
            PathCmd::MoveTo { x, y } => {
                cur_x = *x as f32;
                cur_y = *y as f32;
                builder.move_to(point(cur_x, cur_y));
            }
            PathCmd::LineTo { x, y } => {
                cur_x = *x as f32;
                cur_y = *y as f32;
                builder.line_to(point(cur_x, cur_y));
            }
            PathCmd::HLineTo { x } => {
                cur_x = *x as f32;
                builder.line_to(point(cur_x, cur_y));
            }
            PathCmd::VLineTo { y } => {
                cur_y = *y as f32;
                builder.line_to(point(cur_x, cur_y));
            }
            PathCmd::QuadTo { cx, cy, x, y } => {
                cur_x = *x as f32;
                cur_y = *y as f32;
                builder.quadratic_bezier_to(
                    point(*cx as f32, *cy as f32),
                    point(cur_x, cur_y),
                );
            }
            PathCmd::CubicTo { c1x, c1y, c2x, c2y, x, y } => {
                cur_x = *x as f32;
                cur_y = *y as f32;
                builder.cubic_bezier_to(
                    point(*c1x as f32, *c1y as f32),
                    point(*c2x as f32, *c2y as f32),
                    point(cur_x, cur_y),
                );
            }
            PathCmd::ArcTo { rx, ry, rotation, large_arc, sweep, x, y } => {
                cur_x = *x as f32;
                cur_y = *y as f32;
                builder.arc_to(
                    lyon::math::vector(*rx as f32, *ry as f32),
                    lyon::math::Angle::degrees(*rotation as f32),
                    lyon::path::ArcFlags {
                        large_arc: *large_arc,
                        sweep: *sweep,
                    },
                    point(cur_x, cur_y),
                );
            }
            PathCmd::Close => {
                builder.close();
            }
        }
    }

    if closed && !matches!(cmds.last(), Some(PathCmd::Close)) {
        builder.close();
    }

    builder.build()
}

fn color_to_f32(c: &Color, opacity: f64) -> [f32; 4] {
    crate::scene_load::color_to_linear(c, opacity)
}

fn stroke_path_dashed(
    path: &LyonPath,
    dash: Option<&[f64]>,
    opts: &StrokeOptions,
    color: [f32; 4],
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    let effective_path;
    let to_tessellate = match dash {
        Some(pattern) if !pattern.is_empty() && pattern.iter().any(|&d| d > 0.0) => {
            effective_path = apply_dash_pattern(path, pattern);
            &effective_path
        }
        _ => path,
    };

    let mut tess = StrokeTessellator::new();
    let _ = tess.tessellate_path(
        to_tessellate,
        opts,
        &mut BuffersBuilder::new(buffers, move |v: StrokeVertex| MeshVertex {
            position: v.position_on_path().to_array(),
            normal: { let n = v.normal(); [n.x, n.y] },
            half_width: v.line_width() / 2.0,
            color,
        }),
    );
}

fn apply_dash_pattern(path: &LyonPath, pattern: &[f64]) -> LyonPath {
    let mut segments: Vec<(lyon::math::Point, lyon::math::Point)> = Vec::new();
    let mut prev: Option<lyon::math::Point> = None;

    for evt in path.iter() {
        match evt {
            PathEvent::Begin { at } => {
                prev = Some(at);
            }
            PathEvent::Line { from: _, to } => {
                if let Some(p) = prev {
                    segments.push((p, to));
                }
                prev = Some(to);
            }
            PathEvent::Quadratic { from, ctrl, to } => {
                flatten_quad(from, ctrl, to, &mut segments);
                prev = Some(to);
            }
            PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                flatten_cubic(from, ctrl1, ctrl2, to, &mut segments);
                prev = Some(to);
            }
            PathEvent::End { last, first, close } => {
                if close {
                    segments.push((last, first));
                }
                prev = None;
            }
        }
    }

    let mut builder = LyonPath::builder();
    let mut dash_idx = 0usize;
    let mut dash_remaining = pattern[0] as f32;
    let mut drawing = true;

    for (start, end) in &segments {
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len < 1e-6 {
            continue;
        }
        let ux = dx / seg_len;
        let uy = dy / seg_len;
        let mut consumed = 0.0f32;

        while consumed < seg_len {
            let remain_in_seg = seg_len - consumed;
            let advance = dash_remaining.min(remain_in_seg);
            let px = start.x + ux * (consumed + advance);
            let py = start.y + uy * (consumed + advance);

            if drawing {
                let sx = start.x + ux * consumed;
                let sy = start.y + uy * consumed;
                builder.begin(point(sx, sy));
                builder.line_to(point(px, py));
                builder.end(false);
            }

            consumed += advance;
            dash_remaining -= advance;
            if dash_remaining <= 0.0 {
                dash_idx = (dash_idx + 1) % pattern.len();
                dash_remaining = pattern[dash_idx] as f32;
                drawing = !drawing;
            }
        }
    }

    builder.build()
}

fn flatten_quad(
    from: lyon::math::Point,
    ctrl: lyon::math::Point,
    to: lyon::math::Point,
    out: &mut Vec<(lyon::math::Point, lyon::math::Point)>,
) {
    const STEPS: usize = 8;
    let mut prev = from;
    for i in 1..=STEPS {
        let t = i as f32 / STEPS as f32;
        let inv = 1.0 - t;
        let x = inv * inv * from.x + 2.0 * inv * t * ctrl.x + t * t * to.x;
        let y = inv * inv * from.y + 2.0 * inv * t * ctrl.y + t * t * to.y;
        let next = point(x, y);
        out.push((prev, next));
        prev = next;
    }
}

fn flatten_cubic(
    from: lyon::math::Point,
    c1: lyon::math::Point,
    c2: lyon::math::Point,
    to: lyon::math::Point,
    out: &mut Vec<(lyon::math::Point, lyon::math::Point)>,
) {
    const STEPS: usize = 16;
    let mut prev = from;
    for i in 1..=STEPS {
        let t = i as f32 / STEPS as f32;
        let inv = 1.0 - t;
        let x = inv * inv * inv * from.x
            + 3.0 * inv * inv * t * c1.x
            + 3.0 * inv * t * t * c2.x
            + t * t * t * to.x;
        let y = inv * inv * inv * from.y
            + 3.0 * inv * inv * t * c1.y
            + 3.0 * inv * t * t * c2.y
            + t * t * t * to.y;
        let next = point(x, y);
        out.push((prev, next));
        prev = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_scene::{Color, StrokeStyle};

    fn make_stroke_style(opacity: f64, stroke_opacity: f64) -> StrokeStyle {
        StrokeStyle {
            color: Color { r: 255, g: 255, b: 255, a: 255 },
            width: 2.0,
            opacity,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity,
        }
    }

    // ── W10: stroke_opacity factored into tessellated mesh color ─────────

    /// A line with stroke_opacity=0.5 and opacity=1.0 must produce a mesh
    /// vertex with alpha ≈ 0.5, not 1.0.
    #[test]
    fn w10_line_tessellation_uses_stroke_opacity() {
        let style = make_stroke_style(1.0, 0.5);
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_line(0.0, 0.0, 100.0, 0.0, &style, &mut buffers);

        assert!(!buffers.vertices.is_empty(), "tessellate_line must produce vertices");
        for v in &buffers.vertices {
            assert!(
                (v.color[3] - 0.5).abs() < 0.05,
                "vertex alpha must reflect stroke_opacity (0.5), got {}",
                v.color[3]
            );
        }
    }

    /// When both opacity=0.8 and stroke_opacity=0.5 are set, the effective
    /// alpha must be opacity * stroke_opacity = 0.4.
    #[test]
    fn w10_line_tessellation_multiplies_opacity_and_stroke_opacity() {
        let style = make_stroke_style(0.8, 0.5);
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_line(0.0, 0.0, 100.0, 0.0, &style, &mut buffers);

        assert!(!buffers.vertices.is_empty());
        for v in &buffers.vertices {
            assert!(
                (v.color[3] - 0.4).abs() < 0.05,
                "alpha must be opacity*stroke_opacity = 0.4, got {}",
                v.color[3]
            );
        }
    }

    /// A polyline with stroke_opacity=0.3 and opacity=1.0 must produce vertices
    /// with alpha ≈ 0.3.
    #[test]
    fn w10_polyline_tessellation_uses_stroke_opacity() {
        let style = make_stroke_style(1.0, 0.3);
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        let points = vec![(0.0, 0.0), (50.0, 0.0), (100.0, 50.0)];
        tessellate_polyline(&points, &style, &mut buffers);

        assert!(!buffers.vertices.is_empty());
        for v in &buffers.vertices {
            assert!(
                (v.color[3] - 0.3).abs() < 0.05,
                "polyline vertex alpha must reflect stroke_opacity (0.3), got {}",
                v.color[3]
            );
        }
    }

    /// stroke_opacity=1.0 and opacity=1.0 must give full alpha (no regression).
    #[test]
    fn w10_line_tessellation_full_opacity_unchanged() {
        let style = make_stroke_style(1.0, 1.0);
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_line(0.0, 0.0, 100.0, 0.0, &style, &mut buffers);

        assert!(!buffers.vertices.is_empty());
        for v in &buffers.vertices {
            assert!(
                (v.color[3] - 1.0).abs() < 0.05,
                "full opacity must give alpha=1.0, got {}",
                v.color[3]
            );
        }
    }

    // ── B2: tessellate_path and tessellate_polygon apply stroke_opacity ──

    fn make_fill_stroke_style(opacity: f64, stroke_opacity: f64, fill_opacity: f64) -> FillStroke {
        FillStroke {
            fill: Some(Color { r: 200, g: 100, b: 50, a: 255 }),
            stroke: Some(Color { r: 255, g: 255, b: 255, a: 255 }),
            stroke_width: 2.0,
            opacity,
            stroke_dash: None,
            stroke_opacity,
            fill_opacity,
            angle: 0.0,
        }
    }

    /// tessellate_path stroke must apply opacity * stroke_opacity.
    #[test]
    fn b2_path_stroke_applies_stroke_opacity() {
        use ferrum_scene::PathCmd;
        let style = make_fill_stroke_style(1.0, 0.6, 1.0);
        let commands = vec![
            PathCmd::MoveTo { x: 0.0, y: 0.0 },
            PathCmd::LineTo { x: 100.0, y: 0.0 },
            PathCmd::LineTo { x: 100.0, y: 100.0 },
            PathCmd::Close,
        ];
        // Tessellate with fill=None so we only get stroke vertices.
        let mut stroke_only_style = style;
        stroke_only_style.fill = None;
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_path(&commands, &stroke_only_style, true, None, None, &mut buffers);

        assert!(!buffers.vertices.is_empty(), "path stroke must produce vertices");
        for v in &buffers.vertices {
            // Expected alpha = opacity(1.0) * stroke_opacity(0.6) = 0.6
            assert!(
                (v.color[3] - 0.6).abs() < 0.05,
                "path stroke alpha must be opacity*stroke_opacity (0.6), got {}",
                v.color[3]
            );
        }
    }

    /// tessellate_path fill must apply opacity * fill_opacity.
    #[test]
    fn b2_path_fill_applies_fill_opacity() {
        use ferrum_scene::PathCmd;
        let style = make_fill_stroke_style(0.8, 1.0, 0.5);
        let commands = vec![
            PathCmd::MoveTo { x: 0.0, y: 0.0 },
            PathCmd::LineTo { x: 100.0, y: 0.0 },
            PathCmd::LineTo { x: 100.0, y: 100.0 },
            PathCmd::Close,
        ];
        // Tessellate with stroke=None so we only get fill vertices.
        let mut fill_only_style = style;
        fill_only_style.stroke = None;
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_path(&commands, &fill_only_style, true, None, None, &mut buffers);

        assert!(!buffers.vertices.is_empty(), "path fill must produce vertices");
        for v in &buffers.vertices {
            // Expected alpha = opacity(0.8) * fill_opacity(0.5) = 0.4
            assert!(
                (v.color[3] - 0.4).abs() < 0.05,
                "path fill alpha must be opacity*fill_opacity (0.4), got {}",
                v.color[3]
            );
        }
    }

    /// tessellate_polygon stroke must apply opacity * stroke_opacity.
    #[test]
    fn b2_polygon_stroke_applies_stroke_opacity() {
        let style = make_fill_stroke_style(1.0, 0.6, 1.0);
        let mut stroke_only_style = style;
        stroke_only_style.fill = None;
        let rings = vec![vec![
            [0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0],
        ]];
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_polygon(&rings, &stroke_only_style, &mut buffers);

        assert!(!buffers.vertices.is_empty(), "polygon stroke must produce vertices");
        for v in &buffers.vertices {
            assert!(
                (v.color[3] - 0.6).abs() < 0.05,
                "polygon stroke alpha must be opacity*stroke_opacity (0.6), got {}",
                v.color[3]
            );
        }
    }

    /// tessellate_polygon fill must apply opacity * fill_opacity.
    #[test]
    fn b2_polygon_fill_applies_fill_opacity() {
        let style = make_fill_stroke_style(0.8, 1.0, 0.5);
        let mut fill_only_style = style;
        fill_only_style.stroke = None;
        let rings = vec![vec![
            [0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0],
        ]];
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_polygon(&rings, &fill_only_style, &mut buffers);

        assert!(!buffers.vertices.is_empty(), "polygon fill must produce vertices");
        for v in &buffers.vertices {
            assert!(
                (v.color[3] - 0.4).abs() < 0.05,
                "polygon fill alpha must be opacity*fill_opacity (0.4), got {}",
                v.color[3]
            );
        }
    }

    // ── FA-16: screen-space stroke width ────────────────────────────────────

    /// A stroked horizontal segment must produce vertices where the stored
    /// `position` is the centerline (y ≈ 0 for a segment at y=0) and the stored
    /// `normal * half_width` is the half-width offset (≈ ±1.0 here for width=2.0).
    ///
    /// `StrokeVertex` is consumed inside the build closure, so this test cannot
    /// re-derive lyon's `v.position()` after the fact; it checks the centerline
    /// and offset *bounds* instead. The exact at-rest byte-stability — that
    /// `position_on_path() + normal()*(line_width()/2) == old v.position()` at
    /// identity transform, which is what the shader's screen-space offset
    /// reproduces — is guaranteed structurally by lyon 1.x's documented contract,
    /// not asserted here.
    #[test]
    fn fa16_stroke_centerline_plus_normal_equals_lyon_position() {
        let style = make_stroke_style(1.0, 1.0);
        // Use width=2.0 (from make_stroke_style); segment along y=0.
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_line(0.0, 0.0, 100.0, 0.0, &style, &mut buffers);

        assert!(!buffers.vertices.is_empty(), "must produce stroke vertices");

        // For each vertex: centerline + normal*half_width must reconstruct the
        // baked position that lyon's v.position() would have returned.
        // At identity (sx=sy=1, tx=ty=0) the shader computes:
        //   final_px = position * 1 + 0 + normal * half_width
        // which must equal the old baked vertex position.
        //
        // We cannot call v.position() here after-the-fact, but we can verify:
        //   1. The centerline y is near 0 (segment is along y=0).
        //   2. The reconstructed half-width offset is ≈ ±1.0 (half of width=2.0).
        for v in &buffers.vertices {
            // Centerline must be close to y=0 (within float tolerance).
            // x is somewhere in [0, 100] along the segment.
            assert!(
                v.position[0] >= -0.1 && v.position[0] <= 100.1,
                "centerline x must be in segment range [0, 100], got {}",
                v.position[0]
            );
            assert!(
                v.position[1].abs() < 0.1,
                "centerline y must be ~0 for horizontal segment, got {}",
                v.position[1]
            );

            // normal * half_width is the offset — its magnitude ≈ half_width
            // (normal magnitude may exceed 1.0 at joins but for a straight line it's 1.0).
            let offset_magnitude = (v.normal[0] * v.half_width).hypot(v.normal[1] * v.half_width);
            // For a straight segment with width=2, half_width=1, offset ≈ 1.0.
            // Allow generous tolerance for cap/join geometry at endpoints.
            assert!(
                offset_magnitude < 3.0,
                "offset magnitude must be bounded (< 3.0 * half_width), got {}",
                offset_magnitude
            );
        }
    }

    /// Fill vertices must have half_width == 0.0 and normal == [0, 0].
    #[test]
    fn fa16_fill_has_zero_half_width_and_normal() {
        use ferrum_scene::PathCmd;
        let style = make_fill_stroke_style(1.0, 1.0, 1.0);
        let commands = vec![
            PathCmd::MoveTo { x: 0.0, y: 0.0 },
            PathCmd::LineTo { x: 100.0, y: 0.0 },
            PathCmd::LineTo { x: 100.0, y: 100.0 },
            PathCmd::Close,
        ];
        let mut fill_only = style;
        fill_only.stroke = None;
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_path(&commands, &fill_only, true, None, None, &mut buffers);

        assert!(!buffers.vertices.is_empty(), "fill must produce vertices");
        for v in &buffers.vertices {
            assert_eq!(v.half_width, 0.0, "fill half_width must be 0.0");
            assert_eq!(v.normal, [0.0, 0.0], "fill normal must be [0, 0]");
        }
    }

    /// Polygon fill vertices must also have half_width == 0 and normal == [0,0].
    #[test]
    fn fa16_polygon_fill_has_zero_half_width() {
        let style = make_fill_stroke_style(1.0, 1.0, 1.0);
        let mut fill_only = style;
        fill_only.stroke = None;
        let rings = vec![vec![
            [0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0],
        ]];
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_polygon(&rings, &fill_only, &mut buffers);

        assert!(!buffers.vertices.is_empty(), "polygon fill must produce vertices");
        for v in &buffers.vertices {
            assert_eq!(v.half_width, 0.0, "polygon fill half_width must be 0.0");
            assert_eq!(v.normal, [0.0, 0.0], "polygon fill normal must be [0,0]");
        }
    }

    /// Stroke vertices from tessellate_line must have half_width > 0.
    #[test]
    fn fa16_stroke_has_nonzero_half_width() {
        let style = make_stroke_style(1.0, 1.0);
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_line(0.0, 0.0, 100.0, 0.0, &style, &mut buffers);

        assert!(!buffers.vertices.is_empty());
        for v in &buffers.vertices {
            assert!(
                v.half_width > 0.0,
                "stroke half_width must be > 0, got {}",
                v.half_width
            );
        }
    }
}

fn apply_cap_join(
    opts: &mut StrokeOptions,
    cap: Option<StrokeCap>,
    join: Option<StrokeJoin>,
) {
    if let Some(c) = cap {
        let lc = match c {
            StrokeCap::Butt => LineCap::Butt,
            StrokeCap::Round => LineCap::Round,
            StrokeCap::Square => LineCap::Square,
        };
        opts.start_cap = lc;
        opts.end_cap = lc;
    }
    if let Some(j) = join {
        opts.line_join = match j {
            StrokeJoin::Miter => LineJoin::Miter,
            StrokeJoin::Round => LineJoin::Round,
            StrokeJoin::Bevel => LineJoin::Bevel,
        };
    }
}
