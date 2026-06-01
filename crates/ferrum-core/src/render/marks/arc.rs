use ferrum_scene::{
    MarkBatchKind, PathCmd, SceneNode, TooltipContent as FsTooltipContent,
    TooltipField as FsTooltipField,
};

use crate::render::arrow_cast::{col_as_f64, col_as_str};
use crate::render::color::with_opacity;
use crate::render::draw::{to_scene_fill_stroke, DrawCtx, MarkBuildResult, MetadataColumns};
use crate::render::scale_resolve::ColorScale;
use crate::spec::coord::{CoordKind as SpecCoord, PolarThetaChannel};

/// Build arc (wedge) nodes for pie/donut charts.
///
/// Requires `CoordPolar` — returns empty for any other coord.  Each row
/// becomes one wedge whose angular sweep is proportional to its value in
/// the theta-mapped encoding field.
pub fn build(ctx: &DrawCtx<'_>) -> MarkBuildResult {
    let (theta_ch, start_angle, inner_radius, outer_radius_opt, pad_angle) = match &ctx.spec.coord {
        Some(SpecCoord::Polar { theta, start_angle, inner_radius, outer_radius, pad_angle, .. }) => {
            (*theta, *start_angle, *inner_radius, *outer_radius, *pad_angle)
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

    // Per-row opacity encoding.
    let opacity_values: Option<Vec<Option<f64>>> = ctx.spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    // Collect tooltip column data up front so we can index by row later.
    let meta = MetadataColumns::from_ctx(ctx);

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
        let angle_start = cum_angle + pad_angle / 2.0;
        let angle_end = cum_angle + sweep - pad_angle / 2.0;
        cum_angle += sweep;
        // Skip degenerate slices that collapse to zero or negative sweep after padding.
        if angle_end <= angle_start { continue; }

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
        // Resolve per-row opacity through scale if present; fall back to mark_style.opacity.
        let row_opacity = if let (Some(values), Some(scale)) = (&opacity_values, &ctx.scales.opacity) {
            match values.get(i).copied().flatten().and_then(|v| scale.inner.to_pixel_f64(v)) {
                Some(op) => op,
                None => ctx.mark_style.opacity,
            }
        } else {
            ctx.mark_style.opacity
        };
        let fill_color = with_opacity(fill_base, row_opacity);

        let commands = wedge_path(cx, cy, inner_radius, outer_radius, angle_start, angle_end);
        nodes.push(SceneNode::Path {
            commands,
            style: to_scene_fill_stroke(
                Some(fill_color),
                ctx.mark_style.stroke,
                ctx.mark_style.stroke_width,
                row_opacity,
                ctx.mark_style.stroke_dash.as_deref(),
            ),
            closed: true,
        });
        data_indices.push(i);
    }

    // Build one tooltip per rendered node, indexed via data_indices.
    let tooltips: Option<Vec<FsTooltipContent>> = if meta.tooltip_cols.is_empty() {
        None
    } else {
        Some(
            data_indices
                .iter()
                .map(|&row_idx| FsTooltipContent {
                    fields: meta
                        .tooltip_cols
                        .iter()
                        .map(|(name, col)| FsTooltipField {
                            name: name.clone(),
                            value: col
                                .get(row_idx)
                                .and_then(|v| v.clone())
                                .unwrap_or_default(),
                        })
                        .collect(),
                })
                .collect(),
        )
    };

    MarkBuildResult {
        kind: MarkBatchKind::Arc,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::{resolve_mark_style, DrawCtx};
    use crate::render::scale_resolve::{OpacityScale, ResolvedScales, ScaleKind};
    use crate::scale::linear::LinearScale;
    use crate::spec::chart::ChartSpec;
    use crate::spec::coord::{CoordKind as SpecCoord, PolarThetaChannel};
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use ferrum_scene::{PolarDirection, SceneNode};
    use std::sync::Arc;

    fn polar_spec(with_opacity: bool) -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Arc,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                opacity: if with_opacity {
                    Some(EncodingSpec { field: "op".into(), type_: Some(SDT::Quantitative), ..Default::default() })
                } else {
                    None
                },
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: None,
                pad_angle: 0.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
        }
    }

    fn make_batch(with_opacity: bool) -> arrow::record_batch::RecordBatch {
        let mut fields = vec![
            Field::new("val", DataType::Float64, false),
        ];
        let mut arrays: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(Float64Array::from(vec![10.0, 30.0, 60.0])),
        ];
        if with_opacity {
            fields.push(Field::new("op", DataType::Float64, false));
            arrays.push(Arc::new(Float64Array::from(vec![0.2, 0.5, 0.9])));
        }
        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays).unwrap()
    }

    fn make_scales(with_opacity: bool) -> ResolvedScales {
        ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 100.0], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 100.0], vec![100.0, 0.0], false, false)),
            color: None,
            size: None,
            shape: None,
            opacity: if with_opacity {
                Some(OpacityScale {
                    inner: ScaleKind::Linear(LinearScale::new_internal(
                        vec![0.2, 0.9], vec![0.2, 0.9], false, false,
                    )),
                })
            } else {
                None
            },
            x2: None,
            y2: None,
        }
    }

    fn make_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        }
    }

    /// Basic smoke test: 3 slices → 3 Path nodes.
    #[test]
    fn arc_emits_one_path_per_slice() {
        let spec = polar_spec(false);
        let batch = make_batch(false);
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = make_scales(false);
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx);
        let paths = result.nodes.iter().filter(|n| matches!(n, SceneNode::Path { .. })).count();
        assert_eq!(paths, 3, "expected 3 arc Path nodes, got {paths}");
    }

    /// W18: When an opacity encoding is present, arc slices must have different
    /// alpha values in their fill color (per-row opacity applied through scale).
    #[test]
    fn w18_arc_opacity_encoding_applied_per_slice() {
        let spec = polar_spec(true);
        let batch = make_batch(true);
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = make_scales(true);
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx);

        let paths: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Path { style, .. } = n { Some(style.clone()) } else { None }
        }).collect();
        assert_eq!(paths.len(), 3, "expected 3 Path nodes");

        // With opacity values [0.2, 0.5, 0.9] the alphas must all differ.
        // ferrum_scene::Color uses .a for alpha.
        let alphas: Vec<u8> = paths.iter()
            .map(|s| s.fill.as_ref().map(|c| c.a).unwrap_or(255))
            .collect();
        let all_same = alphas.iter().all(|&a| a == alphas[0]);
        assert!(
            !all_same,
            "per-row opacity encoding must produce different alphas on arc slices; all were {:?}",
            alphas
        );
    }
}
