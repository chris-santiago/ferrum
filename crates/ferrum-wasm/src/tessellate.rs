use ferrum_scene::*;
use lyon::math::point;
use lyon::path::builder::SvgPathBuilder;
use lyon::path::{Path as LyonPath, PathEvent};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, LineCap, LineJoin, StrokeOptions,
    StrokeTessellator, StrokeVertex, VertexBuffers,
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
    /// Stroke: unit normal from lyon. With bevel joins |normal| ≈ 1.0 at all
    /// vertices, which is required for the screen-space-width shader to be
    /// correct under non-uniform affines (sx ≠ sy). Miter joins would bake
    /// elongated normals (|normal| > 1) in scene space, producing the
    /// "ribbon" artifact on rescale. Fill: [0,0].
    pub normal: [f32; 2],
    /// Stroke: half of the stroke line width in pixels. Fill: 0.0.
    pub half_width: f32,
    pub color: [f32; 4],
}

// Compile-time guard: position(8) + normal(8) + half_width(4) + color(16) = 36 bytes.
const _: () = assert!(core::mem::size_of::<MeshVertex>() == 36);

pub fn tessellate_line(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    style: &StrokeStyle,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    // A cleared paint paints nothing: no geometry at all, mirroring
    // `tessellate_path`'s `if let Some(stroke)` branch and the SVG walker's
    // omitted `stroke` attribute (`#110`).
    let Some(c) = &style.color else { return };
    let color = color_to_f32(c, style.opacity * style.stroke_opacity);
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
    let Some(c) = &style.color else { return };
    let color = color_to_f32(c, style.opacity * style.stroke_opacity);
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

/// Tessellate a [`SceneNode::Polyline`] that carries `widths`: one width per
/// vertex, laid out and dashed by the shared taper helper so a dash cut takes
/// the width interpolated at the cut. Refuses misaligned or non-positive
/// `widths` before drawing anything.
pub fn tessellate_tapered_polyline(
    points: &[(f64, f64)],
    widths: &[f64],
    style: &StrokeStyle,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) -> Result<(), TaperError> {
    let pieces = dash_tapered(points, widths, style.dash.as_deref().unwrap_or(&[]))?;
    let Some(c) = &style.color else { return Ok(()) };
    let color = color_to_f32(c, style.opacity * style.stroke_opacity);
    let mut opts = StrokeOptions::default();
    apply_cap_join(&mut opts, style.stroke_cap, style.stroke_join);
    stroke_tapered(&pieces, &opts, color, buffers);
    Ok(())
}

/// Tessellate a [`SceneNode::Path`] that carries `widths`: one width per
/// command endpoint, flattened and dashed by the shared taper helper. A
/// tapered path is an open stroke, so its fill is not drawn. Refuses a closed
/// path and misaligned or non-positive `widths` before drawing anything.
pub fn tessellate_tapered_path(
    commands: &[PathCmd],
    widths: &[f64],
    style: &FillStroke,
    closure: PathClosure,
    batch_cap: Option<StrokeCap>,
    batch_join: Option<StrokeJoin>,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) -> Result<(), TaperError> {
    let pieces = dash_tapered_path(commands, closure, widths, style.stroke_dash.as_deref().unwrap_or(&[]))?;
    let Some(stroke) = &style.stroke else { return Ok(()) };
    let color = color_to_f32(stroke, style.opacity * style.stroke_opacity);
    let mut opts = StrokeOptions::default();
    apply_cap_join(&mut opts, batch_cap, batch_join);
    stroke_tapered(&pieces, &opts, color, buffers);
    Ok(())
}

/// Stroke tapered on-pieces, each one subpath whose vertices carry their
/// width as a path attribute.
///
/// Lyon strokes at the widest width (so its fold, join and round-subdivision
/// decisions hold for every vertex) with its fixed-width normals, which are
/// perpendicular to the centerline. Each vertex then takes half its own
/// interpolated width as `half_width`, so the shader offsets it by that width
/// along the screen-space perpendicular: the edge between two vertices is the
/// straight line joining their offsets, and the thickness at every vertex is
/// its width in screen pixels under any affine, uniform or not. Lyon's
/// variable-width mode is not used because it tilts the normal off the
/// perpendicular by the taper's slope, which the shader's inverse-transpose
/// turns into a thickness error under `sx != sy`. Every turn is bracketed by
/// [`bracket_turns`] so its one miter-length normal stays inside the join.
fn stroke_tapered(
    pieces: &[TaperPiece],
    opts: &StrokeOptions,
    color: [f32; 4],
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    let widest = pieces
        .iter()
        .flat_map(|p| match p {
            TaperPiece::Run { widths, .. } => widths.as_slice(),
            TaperPiece::Dot { width, .. } => std::slice::from_ref(width),
        })
        .fold(0.0_f64, |a, &w| a.max(w));
    let mut builder = LyonPath::builder_with_attributes(1);
    // Square dots, each a butt-capped run one width long along its heading,
    // which is the square a square cap draws turned to the line; lyon's own
    // square cap on a zero-length subpath stays axis-aligned.
    let mut squares = LyonPath::builder_with_attributes(1);
    let mut any_square = false;
    for piece in pieces {
        let (points, widths) = match piece {
            TaperPiece::Dot { point: (x, y), width: w, heading: (hx, hy) } => {
                // A round cap's disk is lyon's cap on a zero-length subpath;
                // a butt cap draws nothing.
                let w32 = [*w as f32];
                match opts.start_cap {
                    LineCap::Round => {
                        builder.begin(point(*x as f32, *y as f32), &w32);
                        builder.line_to(point(*x as f32, *y as f32), &w32);
                        builder.end(false);
                    }
                    LineCap::Square => {
                        let r = w / 2.0;
                        squares.begin(point((x - hx * r) as f32, (y - hy * r) as f32), &w32);
                        squares.line_to(point((x + hx * r) as f32, (y + hy * r) as f32), &w32);
                        squares.end(false);
                        any_square = true;
                    }
                    LineCap::Butt => {}
                }
                continue;
            }
            TaperPiece::Run { points, widths } => bracket_turns(points, widths),
        };
        let mut vertices = points.iter().zip(&widths);
        let Some((&(x, y), &w)) = vertices.next() else { continue };
        builder.begin(point(x as f32, y as f32), &[w as f32]);
        for (&(x, y), &w) in vertices {
            builder.line_to(point(x as f32, y as f32), &[w as f32]);
        }
        builder.end(false);
    }
    let path = builder.build();
    let opts = opts.with_line_width(widest as f32);
    let half_width = |v: &mut StrokeVertex| v.interpolated_attributes().first().copied().unwrap_or_default() / 2.0;
    stroke_to_mesh(&path, &opts, color, buffers, half_width);
    if any_square {
        let butt = opts.with_line_cap(LineCap::Butt);
        stroke_to_mesh(&squares.build(), &butt, color, buffers, half_width);
    }
}

/// Turns flatter than this (cosine of the angle between two segments) are
/// left unbracketed: their miter normal is within 0.01% of unit length.
const COS_STRAIGHT: f64 = 0.9998;

/// The run `points`/`widths` with a vertex added on each segment either side of every turn,
/// where that segment's inner offset line meets the other's.
///
/// Lyon joins a turn's inner side at one vertex whose normal is the miter
/// vector, longer than unit. The shader maps a normal to screen space by the
/// affine's inverse-transpose, which is exact for a unit perpendicular but
/// not for a miter under `sx != sy`: unbracketed, the misplaced inner vertex
/// tilts both adjacent edges along their whole length and thickens the
/// stroke. A bracketing vertex is collinear with its segment, so the outline
/// is unchanged, and it carries a unit perpendicular normal, so every edge
/// away from the join is exact under any affine and the miter only shapes
/// the join itself. A bracket sits `w/2 * tan(turn/2)` from the turn (the
/// inner lines' meeting point, `w` the widest of the three vertices) or
/// halfway along a shorter segment; its width is the taper's at that point.
fn bracket_turns(points: &[(f64, f64)], widths: &[f64]) -> (Vec<(f64, f64)>, Vec<f64>) {
    let (Some(&first), Some(&first_width)) = (points.first(), widths.first()) else {
        return (points.to_vec(), widths.to_vec());
    };
    if points.len() != widths.len() {
        return (points.to_vec(), widths.to_vec());
    }
    let mut out = Centerline { points: vec![first], widths: vec![first_width] };
    for i in 1..points.len() {
        let p = points[i];
        let Some(&next) = points.get(i + 1) else {
            out.points.push(p);
            out.widths.push(widths[i]);
            break;
        };
        let prev = points[i - 1];
        let (l1, l2) = ((p.0 - prev.0).hypot(p.1 - prev.1), (next.0 - p.0).hypot(next.1 - p.1));
        let (t1, t2) = (((p.0 - prev.0) / l1, (p.1 - prev.1) / l1), ((next.0 - p.0) / l2, (next.1 - p.1) / l2));
        let cos = (t1.0 * t2.0 + t1.1 * t2.1).clamp(-1.0, 1.0);
        if !(l1 > 0.0 && l2 > 0.0 && cos < COS_STRAIGHT) {
            out.points.push(p);
            out.widths.push(widths[i]);
            continue;
        }
        let half = widths[i - 1].max(widths[i]).max(widths[i + 1]) / 2.0;
        let reach = half * ((1.0 - cos) / (1.0 + cos)).sqrt();
        let (d1, d2) = (reach.min(l1 / 2.0), reach.min(l2 / 2.0));
        let before = (p.0 - t1.0 * d1, p.1 - t1.1 * d1);
        let after = (p.0 + t2.0 * d2, p.1 + t2.1 * d2);
        out.points.extend([before, p, after]);
        out.widths.extend([
            width_at(widths, i - 1, 1.0 - d1 / l1).unwrap_or(widths[i]),
            widths[i],
            width_at(widths, i, d2 / l2).unwrap_or(widths[i]),
        ]);
    }
    (out.points, out.widths)
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
        if hole.len() < 3 {
            continue;
        }
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
        // Use Bevel joins so every stroke vertex carries a unit normal (|normal| ≈ 1).
        // This is required for the screen-space-width shader to stay correct under
        // non-uniform affines (sx ≠ sy). Miter joins bake elongated bisector normals
        // in scene space and produce the "ribbon" artefact on reactive rescale.
        let opts = StrokeOptions::default()
            .with_line_width(style.stroke_width as f32)
            .with_line_join(LineJoin::Bevel);
        let mut tess = StrokeTessellator::new();
        let _ = tess.tessellate_path(
            &path,
            &opts,
            &mut BuffersBuilder::new(buffers, move |v: StrokeVertex| MeshVertex {
                position: v.position_on_path().to_array(),
                normal: {
                    let n = v.normal();
                    [n.x, n.y]
                },
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
                builder.quadratic_bezier_to(point(*cx as f32, *cy as f32), point(cur_x, cur_y));
            }
            PathCmd::CubicTo {
                c1x,
                c1y,
                c2x,
                c2y,
                x,
                y,
            } => {
                cur_x = *x as f32;
                cur_y = *y as f32;
                builder.cubic_bezier_to(
                    point(*c1x as f32, *c1y as f32),
                    point(*c2x as f32, *c2y as f32),
                    point(cur_x, cur_y),
                );
            }
            PathCmd::ArcTo {
                rx,
                ry,
                rotation,
                large_arc,
                sweep,
                x,
                y,
            } => {
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
    stroke_to_mesh(to_tessellate, opts, color, buffers, |v| v.line_width() / 2.0);
}

/// The one stroke sink: tessellate `path` into `buffers`, each vertex taking
/// its position and unit-scale normal from lyon and its `half_width` from
/// `half_width`, the only thing that differs between a constant-width stroke
/// (lyon's line width) and a tapered one (the vertex's width attribute).
fn stroke_to_mesh(
    path: &LyonPath,
    opts: &StrokeOptions,
    color: [f32; 4],
    buffers: &mut VertexBuffers<MeshVertex, u32>,
    half_width: impl Fn(&mut StrokeVertex) -> f32,
) {
    let mut tess = StrokeTessellator::new();
    let _ = tess.tessellate_path(
        path,
        opts,
        &mut BuffersBuilder::new(buffers, move |mut v: StrokeVertex| MeshVertex {
            half_width: half_width(&mut v),
            position: v.position_on_path().to_array(),
            normal: {
                let n = v.normal();
                [n.x, n.y]
            },
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
            PathEvent::Cubic {
                from,
                ctrl1,
                ctrl2,
                to,
            } => {
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
            color: Some(Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            }),
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

        assert!(
            !buffers.vertices.is_empty(),
            "tessellate_line must produce vertices"
        );
        for v in &buffers.vertices {
            assert!(
                (v.color[3] - 0.5).abs() < 0.05,
                "vertex alpha must reflect stroke_opacity (0.5), got {}",
                v.color[3]
            );
        }
    }

    /// `#110`: a cleared stroke paint (`color: None`) tessellates to NO geometry,
    /// the interactive-render counterpart of the SVG walker omitting the `stroke`
    /// attribute. Both stroke-only tessellators are covered; the paired
    /// non-cleared assertions above and below are what keep this from passing by
    /// simply never tessellating anything.
    #[test]
    fn cleared_stroke_paint_tessellates_no_geometry() {
        let cleared = StrokeStyle { color: None, ..make_stroke_style(1.0, 1.0) };

        let mut line: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_line(0.0, 0.0, 100.0, 0.0, &cleared, &mut line);
        assert!(
            line.vertices.is_empty() && line.indices.is_empty(),
            "a cleared stroke must tessellate no line geometry, got {} vertices",
            line.vertices.len()
        );

        let mut poly: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_polyline(&[(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)], &cleared, &mut poly);
        assert!(
            poly.vertices.is_empty() && poly.indices.is_empty(),
            "a cleared stroke must tessellate no polyline geometry, got {} vertices",
            poly.vertices.len()
        );

        // Control: the same geometry with a paint DOES tessellate.
        let painted = make_stroke_style(1.0, 1.0);
        let mut ctrl: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_polyline(&[(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)], &painted, &mut ctrl);
        assert!(!ctrl.vertices.is_empty(), "a painted polyline must tessellate geometry");
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
            fill: Some(Color {
                r: 200,
                g: 100,
                b: 50,
                a: 255,
            }),
            stroke: Some(Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            }),
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
        tessellate_path(
            &commands,
            &stroke_only_style,
            true,
            None,
            None,
            &mut buffers,
        );

        assert!(
            !buffers.vertices.is_empty(),
            "path stroke must produce vertices"
        );
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

        assert!(
            !buffers.vertices.is_empty(),
            "path fill must produce vertices"
        );
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
        let rings = vec![vec![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]];
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_polygon(&rings, &stroke_only_style, &mut buffers);

        assert!(
            !buffers.vertices.is_empty(),
            "polygon stroke must produce vertices"
        );
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
        let rings = vec![vec![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]];
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_polygon(&rings, &fill_only_style, &mut buffers);

        assert!(
            !buffers.vertices.is_empty(),
            "polygon fill must produce vertices"
        );
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

            // normal * half_width is the offset — its magnitude ≈ half_width.
            // With bevel joins, |normal| ≈ 1.0 at every vertex (no elongation);
            // for a straight segment half_width=1, so offset magnitude ≈ 1.0.
            // Allow generous tolerance for cap geometry at endpoints.
            let offset_magnitude = (v.normal[0] * v.half_width).hypot(v.normal[1] * v.half_width);
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
        let rings = vec![vec![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]];
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_polygon(&rings, &fill_only, &mut buffers);

        assert!(
            !buffers.vertices.is_empty(),
            "polygon fill must produce vertices"
        );
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

    // ── Batch E: curve interpolation reaches the GPU path ────────────────

    /// The exact commands `ferrum-core`'s `render::marks::curve::subpath`
    /// emits for `interpolate="monotone"` over the points
    /// `[(0,0), (10,30), (20,10), (30,40)]`.
    ///
    /// `ferrum-wasm` builds for `wasm32-unknown-unknown` and `ferrum-core`
    /// links pyo3, so this crate cannot call the producer directly; the list
    /// is pinned instead, and
    /// `ferrum_core::render::marks::curve::tests::wasm_parity_fixture_matches_real_monotone_output`
    /// asserts the producer still emits exactly these commands. KEEP THE TWO
    /// IN SYNC.
    fn monotone_curve_fixture() -> Vec<PathCmd> {
        vec![
            PathCmd::MoveTo { x: 0.0, y: 0.0 },
            PathCmd::CubicTo {
                c1x: 3.3333333333333335,
                c1y: 10.0,
                c2x: 6.666666666666666,
                c2y: 30.0,
                x: 10.0,
                y: 30.0,
            },
            PathCmd::CubicTo {
                c1x: 13.333333333333334,
                c1y: 30.0,
                c2x: 16.666666666666668,
                c2y: 10.0,
                x: 20.0,
                y: 10.0,
            },
            PathCmd::CubicTo {
                c1x: 23.333333333333332,
                c1y: 10.0,
                c2x: 26.666666666666668,
                c2y: 30.0,
                x: 30.0,
                y: 40.0,
            },
        ]
    }

    /// Vertex count for [`monotone_curve_fixture`] stroked through
    /// [`tessellate_path`]. A literal, because its *value* is not the point —
    /// its *stability* is: it changes if and only if the flattened curve
    /// changes shape, which is the drift the core/WASM fixture pairing exists
    /// to catch.
    ///
    /// Deliberately NOT the only shape pin: this count moves with lyon's
    /// flattening density as well as with the curve, so on a lyon upgrade it is
    /// the assertion most likely to fail for a reason that is not drift. The
    /// per-segment [`CHORD_SPANS`] band is the density-invariant half.
    const EXPECTED_MONOTONE_VERTEX_COUNT: usize = 82;

    /// One segment of [`monotone_curve_fixture`], paired with how far the
    /// flattened curve departs from the straight chord between that segment's
    /// own endpoints.
    struct ChordSpan {
        /// Open x interval of the segment.
        x: (f32, f32),
        /// The chord the segment's endpoints imply: `y = slope * x + intercept`.
        slope: f32,
        intercept: f32,
        /// Max `|curve - chord|` over the interval, to [`DEVIATION_TOL`].
        deviation: f32,
    }

    /// All three fixture segments. A one-sided check over segment 1 alone —
    /// what this test carried before — cannot see a control point moved in
    /// segment 0 or 2, and the disclosed RED for the vertex count
    /// (`c2y: 30.0 -> 24.0`) is exactly such a perturbation: it leaves both the
    /// bbox and segment 1's deviation untouched.
    const CHORD_SPANS: [ChordSpan; 3] = [
        ChordSpan { x: (0.0, 10.0), slope: 3.0, intercept: 0.0, deviation: 4.424_215_3 },
        ChordSpan { x: (10.0, 20.0), slope: -2.0, intercept: 50.0, deviation: 1.911_701_2 },
        ChordSpan { x: (20.0, 30.0), slope: 3.0, intercept: -50.0, deviation: 4.424_22 },
    ];

    /// Band half-width for [`CHORD_SPANS`]. Wide enough to absorb a change in
    /// lyon's flattening density (a coarser polyline samples the extremum less
    /// exactly), an order of magnitude tighter than the ~1.66 px shift the
    /// disclosed RED produces.
    const DEVIATION_TOL: f32 = 0.15;

    /// Vertices sitting exactly on a segment boundary belong to both chords and
    /// deviate from neither; skipped so a shared endpoint cannot drag a
    /// segment's measured maximum toward zero.
    const EDGE_MARGIN: f32 = 0.001;

    /// WASM parity (spec §7): a curve-interpolated line must render through
    /// the GPU path, not just the SVG walker. `CubicTo` reaches lyon's
    /// `cubic_bezier_to`, so the stroked mesh follows the flattened curve —
    /// which means its vertices must leave the straight chord between the
    /// data points that a `linear` trace of the same points would hug, span
    /// the fixture's own extent, and flatten to a stable vertex count.
    #[test]
    fn curve_cubics_from_ferrum_core_tessellate() {
        let commands = monotone_curve_fixture();
        let mut style = make_fill_stroke_style(1.0, 1.0, 1.0);
        style.fill = None;
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_path(&commands, &style, false, None, None, &mut buffers);

        assert!(
            !buffers.vertices.is_empty(),
            "a CubicTo path must tessellate into vertices"
        );
        assert!(
            buffers.vertices.iter().all(|v| v.position[0].is_finite() && v.position[1].is_finite()),
            "every tessellated vertex must be finite"
        );

        // Two-sided per-segment deviation band. Each fixture segment departs
        // from the straight chord between its own endpoints by a fixed amount
        // that is a property of the CURVE, not of how finely lyon subdivides
        // it — so this discriminates a moved control point while staying
        // robust to a flattening-density change, which the vertex count below
        // cannot. A perturbation in any one segment moves that segment's own
        // band; a one-sided check over a single span (what this was) missed
        // two of the three segments entirely.
        for (i, span) in CHORD_SPANS.iter().enumerate() {
            let observed = buffers
                .vertices
                .iter()
                .filter(|v| v.position[0] > span.x.0 + EDGE_MARGIN
                    && v.position[0] < span.x.1 - EDGE_MARGIN)
                .map(|v| (v.position[1] - (span.slope * v.position[0] + span.intercept)).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                (observed - span.deviation).abs() < DEVIATION_TOL,
                "segment {i}: the flattened curve must depart from its chord by \
                 {} px (±{DEVIATION_TOL}), got {observed} — a straight-segment path \
                 would be ~0 and a moved control point shifts this",
                span.deviation
            );
        }

        // Extent pin. Without the shape pins in this test the assertions pass
        // for ANY three cubics, so the pairing with `ferrum-core`'s fixture
        // would be inert: if the curve math changed, the core test would fail
        // and be updated, this literal would go stale, and this test would stay
        // green on a shape production no longer emits.
        let (min_x, max_x, min_y, max_y) = buffers.vertices.iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(lo_x, hi_x, lo_y, hi_y), v| {
                (
                    lo_x.min(v.position[0]),
                    hi_x.max(v.position[0]),
                    lo_y.min(v.position[1]),
                    hi_y.max(v.position[1]),
                )
            },
        );
        let bbox = (min_x, max_x, min_y, max_y);
        let expected = (0.0_f32, 30.0_f32, 0.0_f32, 40.0_f32);
        // Flattening tolerance, NOT a stroke allowance: `MeshVertex::position`
        // is the CENTERLINE — stroke width rides `normal`/`half_width` and is
        // applied in the shader — so `style.stroke_width` does not widen this
        // bbox at all. The slack is lyon's chord error where the flattened
        // polyline meets the fixture's extreme points.
        let tol = 0.05_f32;
        assert!(
            (bbox.0 - expected.0).abs() <= tol
                && (bbox.1 - expected.1).abs() <= tol
                && (bbox.2 - expected.2).abs() <= tol
                && (bbox.3 - expected.3).abs() <= tol,
            "the tessellated mesh must span the fixture's own extent \
             (x {}..{}, y {}..{}); got x {}..{}, y {}..{} — the core fixture drifted",
            expected.0, expected.1, expected.2, expected.3,
            bbox.0, bbox.1, bbox.2, bbox.3,
        );
        assert_eq!(
            buffers.vertices.len(),
            EXPECTED_MONOTONE_VERTEX_COUNT,
            "lyon's flattening of the pinned cubics produced a different vertex count; \
             either the fixture drifted from `ferrum-core`'s monotone output or the \
             tessellation tolerance changed"
        );
    }

    /// Guard for the claim above: the SAME data points traced linearly stay on
    /// the chord, so the deviation the curve test measures comes from the
    /// cubics and not from stroke width or join geometry.
    #[test]
    fn linear_path_over_the_same_points_stays_on_the_chord() {
        let commands = vec![
            PathCmd::MoveTo { x: 0.0, y: 0.0 },
            PathCmd::LineTo { x: 10.0, y: 30.0 },
            PathCmd::LineTo { x: 20.0, y: 10.0 },
            PathCmd::LineTo { x: 30.0, y: 40.0 },
        ];
        let mut style = make_fill_stroke_style(1.0, 1.0, 1.0);
        style.fill = None;
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_path(&commands, &style, false, None, None, &mut buffers);

        assert!(!buffers.vertices.is_empty());
        let max_deviation = buffers
            .vertices
            .iter()
            .filter(|v| v.position[0] > 10.0 && v.position[0] < 20.0)
            .map(|v| (v.position[1] - (50.0 - 2.0 * v.position[0])).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_deviation < 0.1,
            "a linear trace of the same points must hug the chord; got {max_deviation}"
        );
    }

    // ── Tapered strokes (`widths`) ───────────────────────────────────────

    /// The on-screen offset the mesh shader gives `v` under the affine
    /// `diag(sx, sy)`: the inverse-transpose direction of the normal, at
    /// `half_width * |normal|` (mirrors `mesh.wgsl`'s `vs_main`).
    fn shader_offset(v: &MeshVertex, sx: f32, sy: f32) -> (f32, f32) {
        let (nx, ny) = (v.normal[0], v.normal[1]);
        let (dx, dy) = (sy * nx, sx * ny);
        let len = dx.hypot(dy);
        let mag = v.half_width * nx.hypot(ny);
        (dx / len * mag, dy / len * mag)
    }

    fn tapered(points: &[(f64, f64)], widths: &[f64], dash: Option<Vec<f64>>) -> Vec<MeshVertex> {
        let style = StrokeStyle { dash, ..make_stroke_style(1.0, 1.0) };
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_tapered_polyline(points, widths, &style, &mut buffers).expect("well-formed widths");
        assert!(!buffers.indices.is_empty(), "a tapered stroke must tessellate geometry");
        buffers.vertices
    }

    /// Every vertex of a tapered stroke carries half the taper's width at the
    /// centerline point it sits on (a turn's bracketing vertices included), so
    /// `half_width` varies along the stroke and equals `widths[i] / 2` at each
    /// centerline vertex.
    #[test]
    fn tapered_stroke_half_width_is_each_vertex_width_over_two() {
        let points = [(0.0, 0.0), (50.0, 20.0), (100.0, 0.0)];
        let widths = [2.0, 6.0, 10.0];
        let vertices = tapered(&points, &widths, None);
        let mut seen = [false; 3];
        for v in &vertices {
            let (x, y) = (v.position[0] as f64, v.position[1] as f64);
            if let Some(i) = points.iter().position(|p| (p.0 - x).hypot(p.1 - y) < 1e-3) {
                seen[i] = true;
            }
            let (segment, t) = points
                .windows(2)
                .enumerate()
                .find_map(|(i, pair)| {
                    let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
                    let len = (x1 - x0).hypot(y1 - y0);
                    let t = ((x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)) / (len * len);
                    let off = ((x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)).abs() / len;
                    ((-1e-6..=1.0 + 1e-6).contains(&t) && off < 1e-3).then_some((i, t))
                })
                .unwrap_or_else(|| panic!("vertex off the centerline: {:?}", v.position));
            let want = width_at(&widths, segment, t).expect("segment in range") / 2.0;
            assert!(
                (v.half_width as f64 - want).abs() < 1e-4,
                "vertex at {:?}: half_width {} != {want}",
                v.position,
                v.half_width
            );
        }
        assert_eq!(seen, [true; 3], "every centerline vertex must emit stroke vertices");
    }

    /// The screen-space contract: under the identity, a uniform zoom and a
    /// non-uniform rescale alike, every vertex of a straight tapered stroke
    /// is offset perpendicular to the on-screen centerline by exactly half
    /// its width, so the on-screen thickness at each vertex is its width.
    #[test]
    fn tapered_stroke_thickness_is_its_width_on_screen_under_any_affine() {
        let points = [(0.0, 0.0), (40.0, 30.0), (100.0, 75.0)];
        let widths = [2.0, 8.0, 14.0];
        let vertices = tapered(&points, &widths, None);
        for (sx, sy) in [(1.0_f32, 1.0_f32), (2.5, 2.5), (3.0, 1.0), (1.0, 4.0)] {
            let (tx, ty) = (100.0 * sx, 75.0 * sy);
            let len = tx.hypot(ty);
            let (tx, ty) = (tx / len, ty / len);
            for v in &vertices {
                let (ox, oy) = shader_offset(v, sx, sy);
                let along = ox * tx + oy * ty;
                let across = (ox * -ty + oy * tx).abs();
                assert!(
                    along.abs() < 1e-3,
                    "sx={sx} sy={sy}: offset at {:?} leans {along} px along the centerline",
                    v.position
                );
                assert!(
                    (across - v.half_width).abs() < 1e-3,
                    "sx={sx} sy={sy}: offset at {:?} is {across} px across, want {}",
                    v.position,
                    v.half_width
                );
            }
        }
    }

    /// How far the tessellated stroke covers the screen along `dir` (a unit
    /// screen vector) either side of the scene point `at`, under the affine
    /// `diag(sx, sy)`: every triangle is placed where the shader puts its
    /// vertices and the line is marched in 0.01 px steps.
    fn screen_extent(
        buffers: &VertexBuffers<MeshVertex, u32>,
        (sx, sy): (f32, f32),
        at: (f32, f32),
        dir: (f32, f32),
    ) -> f32 {
        let screen: Vec<(f32, f32)> = buffers
            .vertices
            .iter()
            .map(|v| {
                let (ox, oy) = shader_offset(v, sx, sy);
                (v.position[0] * sx + ox, v.position[1] * sy + oy)
            })
            .collect();
        let inside = |q: (f32, f32)| {
            buffers.indices.chunks_exact(3).any(|t| {
                let [a, b, c] = [screen[t[0] as usize], screen[t[1] as usize], screen[t[2] as usize]];
                let side = |p: (f32, f32), r: (f32, f32)| (r.0 - p.0) * (q.1 - p.1) - (r.1 - p.1) * (q.0 - p.0);
                let (d1, d2, d3) = (side(a, b), side(b, c), side(c, a));
                !((d1 < 0.0 || d2 < 0.0 || d3 < 0.0) && (d1 > 0.0 || d2 > 0.0 || d3 > 0.0))
            })
        };
        let centre = (at.0 * sx, at.1 * sy);
        let reach = |sign: f32| {
            let mut t = 0.0_f32;
            while inside((centre.0 + sign * t * dir.0, centre.1 + sign * t * dir.1)) {
                t += 0.01;
            }
            t
        };
        reach(1.0) + reach(-1.0)
    }

    /// The screen-space contract across turns: mid-segment on a zigzag, the
    /// stroke's on-screen thickness perpendicular to the on-screen centerline
    /// is the width there at rest and under a non-uniform rescale either way.
    #[test]
    fn tapered_zigzag_keeps_its_width_on_screen_under_non_uniform_rescale() {
        let points = [(0.0, 0.0), (40.0, 30.0), (80.0, 0.0), (120.0, 30.0)];
        let widths = [4.0, 8.0, 12.0, 16.0];
        let style = make_stroke_style(1.0, 1.0);
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_tapered_polyline(&points, &widths, &style, &mut buffers).expect("well-formed widths");
        for (sx, sy) in [(1.0_f32, 1.0_f32), (3.0, 1.0), (1.0, 3.0)] {
            for (seg, pair) in points.windows(2).enumerate() {
                let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
                let mid = (((x0 + x1) / 2.0) as f32, ((y0 + y1) / 2.0) as f32);
                let (tx, ty) = ((x1 - x0) as f32 * sx, (y1 - y0) as f32 * sy);
                let len = tx.hypot(ty);
                let normal = (-ty / len, tx / len);
                let want = ((widths[seg] + widths[seg + 1]) / 2.0) as f32;
                let got = screen_extent(&buffers, (sx, sy), mid, normal);
                assert!(
                    (got - want).abs() < 0.05,
                    "sx={sx} sy={sy} segment {seg}: {got} px thick on screen, want {want}"
                );
            }
        }
    }

    /// A dashed tapered stroke draws only its on-dashes, and each dash's cut
    /// ends carry the width interpolated at the cut.
    #[test]
    fn dashed_tapered_stroke_keeps_interpolated_widths_at_cuts() {
        let vertices = tapered(&[(0.0, 0.0), (100.0, 0.0)], &[2.0, 12.0], Some(vec![10.0, 10.0]));
        let mut xs: Vec<f32> = Vec::new();
        for v in &vertices {
            let x = v.position[0];
            assert!(
                (v.half_width - (2.0 + 0.1 * x) / 2.0).abs() < 1e-3,
                "x={x}: half_width {} is not the interpolated width",
                v.half_width
            );
            assert!(
                (x % 20.0) <= 10.0 + 1e-3 || (x % 20.0) >= 20.0 - 1e-3,
                "x={x} lies inside an off-dash"
            );
            if !xs.iter().any(|seen| (seen - x).abs() < 1e-3) {
                xs.push(x);
            }
        }
        xs.sort_by(f32::total_cmp);
        let want = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        assert_eq!(xs.len(), want.len(), "dash ends {xs:?}");
        for (x, w) in xs.iter().zip(want) {
            assert!((x - w).abs() < 1e-3, "dash ends {xs:?}");
        }
    }

    /// A zero-length on-dash on a tapered line draws a dot at each
    /// on-position, matching the static outline and a constant-width SVG
    /// stroke: a disk of radius `w/2` under a round cap, a `w`-wide square
    /// turned to the line under a square cap, and nothing under a butt cap.
    /// The line runs diagonally, 45 px long, so `[0, 10]` puts a dot at 0,
    /// 10, 20, 30 and 40 px along it.
    #[test]
    fn a_zero_length_dash_draws_a_dot_at_each_on_position_by_its_cap() {
        let (h, n, r) = ((0.6f32, 0.8f32), (-0.8f32, 0.6f32), 2.0f32);
        let dots: Vec<(f32, f32)> = (0..5).map(|k| (h.0 * 10.0 * k as f32, h.1 * 10.0 * k as f32)).collect();
        let mesh = |cap: StrokeCap| {
            let style = StrokeStyle { dash: Some(vec![0.0, 10.0]), stroke_cap: Some(cap), ..make_stroke_style(1.0, 1.0) };
            let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
            tessellate_tapered_polyline(&[(0.0, 0.0), (27.0, 36.0)], &[4.0, 4.0], &style, &mut buffers).unwrap();
            buffers
        };
        // Where each vertex lands on screen at rest: its centerline point
        // pushed out along its normal by its half width.
        let ends = |buffers: &VertexBuffers<MeshVertex, u32>| -> Vec<(f32, f32)> {
            buffers
                .vertices
                .iter()
                .map(|v| (v.position[0] + v.normal[0] * v.half_width, v.position[1] + v.normal[1] * v.half_width))
                .collect()
        };
        let dist = |a: (f32, f32), b: (f32, f32)| (a.0 - b.0).hypot(a.1 - b.1);
        let nearest = |v: (f32, f32)| dots.iter().map(|d| dist(v, *d)).fold(f32::INFINITY, f32::min);

        let round = ends(&mesh(StrokeCap::Round));
        assert!(round.iter().all(|v| (nearest(*v) - r).abs() < 1e-3), "round: only disks of radius w/2");
        for dot in &dots {
            let on_disk = round.iter().filter(|v| (dist(**v, *dot) - r).abs() < 1e-3).count();
            assert!(on_disk >= 8, "round: a disk about {dot:?}, {on_disk} vertices");
        }

        let square = ends(&mesh(StrokeCap::Square));
        for dot in &dots {
            for (a, b) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                let corner = (dot.0 + r * (a * h.0 + b * n.0), dot.1 + r * (a * h.1 + b * n.1));
                assert!(square.iter().any(|v| dist(*v, corner) < 1e-3), "square: corner {corner:?} of {dot:?}");
            }
        }
        assert!(square.iter().all(|v| nearest(*v) < r * 2f32.sqrt() + 1e-3), "square: only the dots' squares");

        assert!(mesh(StrokeCap::Butt).indices.is_empty(), "butt: a zero-length dash draws nothing");
    }

    /// A tapered curve tessellates from the shared flattener: its endpoints
    /// take their command widths and every vertex between lies within them.
    #[test]
    fn tapered_curve_takes_its_endpoint_widths() {
        let commands = [
            PathCmd::MoveTo { x: 0.0, y: 0.0 },
            PathCmd::CubicTo { c1x: 30.0, c1y: 60.0, c2x: 70.0, c2y: -60.0, x: 100.0, y: 0.0 },
        ];
        let style = FillStroke {
            stroke: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
            ..make_fill_stroke_style(1.0, 1.0, 1.0)
        };
        let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
        tessellate_tapered_path(&commands, &[2.0, 10.0], &style, PathClosure::Open, None, None, &mut buffers)
            .expect("well-formed widths");
        let at = |x: f32| {
            buffers
                .vertices
                .iter()
                .filter(|v| (v.position[0] - x).abs() < 1e-3 && v.position[1].abs() < 1e-3)
                .map(|v| v.half_width)
                .collect::<Vec<_>>()
        };
        assert!(!at(0.0).is_empty() && at(0.0).iter().all(|h| (h - 1.0).abs() < 1e-4), "{:?}", at(0.0));
        assert!(!at(100.0).is_empty() && at(100.0).iter().all(|h| (h - 5.0).abs() < 1e-4), "{:?}", at(100.0));
        let distinct = buffers.vertices.iter().filter(|v| v.half_width > 1.01 && v.half_width < 4.99).count();
        assert!(distinct > 0, "the width must vary between the curve's endpoints");
        assert!(buffers.vertices.iter().all(|v| v.half_width >= 1.0 - 1e-4 && v.half_width <= 5.0 + 1e-4));
    }

    /// A malformed tapered stroke is refused through the shared validation
    /// and draws nothing.
    #[test]
    fn malformed_tapered_stroke_is_refused_and_draws_nothing() {
        let style = make_stroke_style(1.0, 1.0);
        let points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)];
        for (widths, want) in [
            (vec![1.0, 2.0], TaperError::Misaligned { points: 3, widths: 2 }),
            (vec![1.0, 0.0, 2.0], TaperError::InvalidWidth { index: 1, value: 0.0 }),
        ] {
            let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
            assert_eq!(tessellate_tapered_polyline(&points, &widths, &style, &mut buffers), Err(want));
            assert!(buffers.vertices.is_empty(), "a refused stroke must draw nothing");
        }

        let fill_stroke = FillStroke {
            stroke: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
            ..make_fill_stroke_style(1.0, 1.0, 1.0)
        };
        let open = [PathCmd::MoveTo { x: 0.0, y: 0.0 }, PathCmd::LineTo { x: 10.0, y: 0.0 }];
        let with_close = [open[0].clone(), open[1].clone(), PathCmd::Close];
        for (commands, closure, widths) in [
            (&open[..], PathClosure::Closed, vec![1.0, 2.0]),
            (&with_close[..], PathClosure::Open, vec![1.0, 2.0, 2.0]),
        ] {
            let mut buffers: VertexBuffers<MeshVertex, u32> = VertexBuffers::new();
            assert_eq!(
                tessellate_tapered_path(commands, &widths, &fill_stroke, closure, None, None, &mut buffers),
                Err(TaperError::Closed)
            );
            assert!(buffers.vertices.is_empty(), "a refused path must draw nothing");
        }
    }

}

fn apply_cap_join(opts: &mut StrokeOptions, cap: Option<StrokeCap>, join: Option<StrokeJoin>) {
    // Default to Bevel joins unconditionally. Bevel emits unit-normal vertices
    // (|normal| ≈ 1) at every join, which is required for the screen-space-width
    // shader to remain correct under non-uniform affines (sx ≠ sy). Miter joins
    // bake elongated bisector normals in scene space, causing the "ribbon" artefact
    // during reactive rescale. The caller may override this default by passing an
    // explicit `StrokeJoin` (including `StrokeJoin::Miter` if the caller accepts
    // the limitation), but for all interactive WASM mesh tessellation the default
    // must be Bevel.
    opts.line_join = LineJoin::Bevel;

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
