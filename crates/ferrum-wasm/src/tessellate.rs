use ferrum_scene::*;
use lyon::math::point;
use lyon::path::builder::SvgPathBuilder;
use lyon::path::{Path as LyonPath, PathEvent};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, LineCap, LineJoin,
    StrokeOptions, StrokeTessellator, StrokeVertex, VertexBuffers,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

pub fn tessellate_line(
    x1: f64, y1: f64, x2: f64, y2: f64,
    style: &StrokeStyle,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    let color = color_to_f32(&style.color, style.opacity);
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
        let color = color_to_f32(fill, style.opacity);
        let mut tess = FillTessellator::new();
        let _ = tess.tessellate_path(
            &path,
            &FillOptions::default(),
            &mut BuffersBuilder::new(buffers, move |v: FillVertex| MeshVertex {
                position: v.position().to_array(),
                color,
            }),
        );
    }

    if let Some(stroke) = &style.stroke {
        let color = color_to_f32(stroke, style.opacity);
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
    let color = color_to_f32(&style.color, style.opacity);
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
        let color = color_to_f32(fill, style.opacity);
        let mut tess = FillTessellator::new();
        let _ = tess.tessellate_path(
            &path,
            &FillOptions::default(),
            &mut BuffersBuilder::new(buffers, move |v: FillVertex| MeshVertex {
                position: v.position().to_array(),
                color,
            }),
        );
    }

    if let Some(stroke) = &style.stroke {
        let color = color_to_f32(stroke, style.opacity);
        let opts = StrokeOptions::default().with_line_width(style.stroke_width as f32);
        let mut tess = StrokeTessellator::new();
        let _ = tess.tessellate_path(
            &path,
            &opts,
            &mut BuffersBuilder::new(buffers, move |v: StrokeVertex| MeshVertex {
                position: v.position().to_array(),
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
    use crate::scene_load::srgb_to_linear;
    [
        srgb_to_linear(c.r as f32 / 255.0),
        srgb_to_linear(c.g as f32 / 255.0),
        srgb_to_linear(c.b as f32 / 255.0),
        (c.a as f32 / 255.0) * opacity as f32,
    ]
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
            position: v.position().to_array(),
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
