use ferrum_scene::{MarkBatchKind, PathCmd, SceneNode};

use crate::render::arrow_cast::{col_as_f64, col_as_str};
use crate::render::color::with_opacity;
use crate::render::draw::{to_scene_fill_stroke, DrawCtx, MarkBuildResult};
use crate::render::scale_resolve::ColorScale;
use crate::spec::coord::{CoordKind as SpecCoord, PolarThetaChannel};

/// Build arc (wedge) nodes for pie/donut charts.
///
/// Requires `CoordPolar` — returns empty for any other coord.  Each row
/// becomes one wedge whose angular sweep is proportional to its value in
/// the theta-mapped encoding field.
pub fn build(ctx: &DrawCtx<'_>) -> MarkBuildResult {
    let (theta_ch, start_angle, inner_radius, outer_radius_opt) = match &ctx.spec.coord {
        Some(SpecCoord::Polar { theta, start_angle, inner_radius, outer_radius, .. }) => {
            (*theta, *start_angle, *inner_radius, *outer_radius)
        }
        _ => return MarkBuildResult::empty(MarkBatchKind::Arc),
    };

    let theta_field = match theta_ch {
        PolarThetaChannel::X => ctx.spec.encoding.x.as_ref().map(|e| e.field.as_str()),
        PolarThetaChannel::Y => ctx.spec.encoding.y.as_ref().map(|e| e.field.as_str()),
    };
    let Some(field) = theta_field else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };

    let Ok(values) = col_as_f64(ctx.batch, field) else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };

    let total: f64 = values.iter()
        .filter_map(|v| *v)
        .filter(|v| v.is_finite() && *v > 0.0)
        .sum();
    if total <= 0.0 {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    }

    let cx = ctx.panel.plot_area.x + ctx.panel.plot_area.w / 2.0;
    let cy = ctx.panel.plot_area.y + ctx.panel.plot_area.h / 2.0;
    let half_min = ctx.panel.plot_area.w.min(ctx.panel.plot_area.h) / 2.0;
    let outer_radius = outer_radius_opt.unwrap_or(half_min);

    // Per-slice color: read color encoding field and look up in the color scale.
    let cfield = ctx.spec.encoding.color.as_ref().map(|e| e.field.as_str());
    let color_str: Option<Vec<Option<String>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Categorical { .. }), Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let color_f64: Option<Vec<Option<f64>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Continuous { .. }), Some(f)) => col_as_f64(ctx.batch, f).ok(),
        _ => None,
    };

    let mut nodes: Vec<SceneNode> = Vec::with_capacity(values.len());
    let mut data_indices: Vec<usize> = Vec::with_capacity(values.len());
    let mut cum_angle = start_angle;
    let tau = std::f64::consts::TAU;

    for (i, v_opt) in values.iter().enumerate() {
        let v = match v_opt {
            Some(v) if v.is_finite() && *v > 0.0 => *v,
            _ => continue,
        };
        let sweep = (v / total) * tau;
        let angle_start = cum_angle;
        let angle_end = cum_angle + sweep;
        cum_angle = angle_end;

        // Resolve per-slice fill from color scale, fall back to mark_style.fill.
        let fill_base = match (&ctx.scales.color, &color_f64, &color_str) {
            (Some(scale @ ColorScale::Continuous { .. }), Some(vals), _) => {
                vals.get(i).and_then(|v| *v)
                    .and_then(|v| if v.is_finite() { scale.lookup_f64(v) } else { None })
                    .unwrap_or(ctx.mark_style.fill)
            }
            (Some(scale @ ColorScale::Categorical { .. }), _, Some(vals)) => {
                vals.get(i).and_then(|v| v.as_deref())
                    .and_then(|v| scale.lookup(v))
                    .unwrap_or(ctx.mark_style.fill)
            }
            _ => ctx.mark_style.fill,
        };
        let fill_color = with_opacity(fill_base, ctx.mark_style.opacity);

        let commands = wedge_path(cx, cy, inner_radius, outer_radius, angle_start, angle_end);
        nodes.push(SceneNode::Path {
            commands,
            style: to_scene_fill_stroke(
                Some(fill_color),
                ctx.mark_style.stroke,
                ctx.mark_style.stroke_width,
                ctx.mark_style.opacity,
                ctx.mark_style.stroke_dash.as_deref(),
            ),
            closed: true,
        });
        data_indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Arc,
        nodes,
        data_indices: Some(data_indices),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    }
}

/// SVG path commands for an arc wedge from `angle_start` to `angle_end`.
///
/// Angles are measured clockwise from 12 o'clock (north):
/// `x = cx + r·sin(θ)`, `y = cy − r·cos(θ)`.
fn wedge_path(
    cx: f64,
    cy: f64,
    inner_r: f64,
    outer_r: f64,
    angle_start: f64,
    angle_end: f64,
) -> Vec<PathCmd> {
    let mut cmds = Vec::new();
    let sweep = angle_end - angle_start;
    let full_circle = sweep.abs() >= std::f64::consts::TAU - 1e-9;
    let large_arc = sweep.abs() > std::f64::consts::PI;

    let ox0 = cx + outer_r * angle_start.sin();
    let oy0 = cy - outer_r * angle_start.cos();
    let ox1 = cx + outer_r * angle_end.sin();
    let oy1 = cy - outer_r * angle_end.cos();

    cmds.push(PathCmd::MoveTo { x: ox0, y: oy0 });

    if full_circle {
        let mid = angle_start + std::f64::consts::PI;
        let oxm = cx + outer_r * mid.sin();
        let oym = cy - outer_r * mid.cos();
        cmds.push(PathCmd::ArcTo { rx: outer_r, ry: outer_r, rotation: 0.0, large_arc: false, sweep: true, x: oxm, y: oym });
        cmds.push(PathCmd::ArcTo { rx: outer_r, ry: outer_r, rotation: 0.0, large_arc: false, sweep: true, x: ox0, y: oy0 });
    } else {
        cmds.push(PathCmd::ArcTo { rx: outer_r, ry: outer_r, rotation: 0.0, large_arc, sweep: true, x: ox1, y: oy1 });
    }

    if inner_r > 0.0 {
        let ix1 = cx + inner_r * angle_end.sin();
        let iy1 = cy - inner_r * angle_end.cos();
        let ix0 = cx + inner_r * angle_start.sin();
        let iy0 = cy - inner_r * angle_start.cos();
        cmds.push(PathCmd::LineTo { x: ix1, y: iy1 });
        if full_circle {
            let mid = angle_start + std::f64::consts::PI;
            let ixm = cx + inner_r * mid.sin();
            let iym = cy - inner_r * mid.cos();
            cmds.push(PathCmd::ArcTo { rx: inner_r, ry: inner_r, rotation: 0.0, large_arc: false, sweep: false, x: ixm, y: iym });
            cmds.push(PathCmd::ArcTo { rx: inner_r, ry: inner_r, rotation: 0.0, large_arc: false, sweep: false, x: ix0, y: iy0 });
        } else {
            cmds.push(PathCmd::ArcTo { rx: inner_r, ry: inner_r, rotation: 0.0, large_arc, sweep: false, x: ix0, y: iy0 });
        }
    } else {
        cmds.push(PathCmd::LineTo { x: cx, y: cy });
    }

    cmds.push(PathCmd::Close);
    cmds
}
