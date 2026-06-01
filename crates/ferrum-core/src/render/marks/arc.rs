use ferrum_scene::{
    MarkBatchKind, PathCmd, SceneNode, TooltipContent as FsTooltipContent,
    TooltipField as FsTooltipField,
};

use crate::render::arrow_cast::{col_as_f64, col_as_str};
use crate::render::color::with_opacity;
use crate::render::draw::{to_scene_fill_stroke, DrawCtx, MarkBuildResult, MetadataColumns};
use crate::render::scale_resolve::ColorScale;
use crate::spec::coord::{CoordKind as SpecCoord, PolarThetaChannel};

/// Resolved polar geometry shared by the pie/donut and annular-wedge paths.
pub(crate) struct PolarGeom {
    pub cx: f64,
    pub cy: f64,
    pub inner_radius: f64,
    pub outer_radius: f64,
    pub start_angle: f64,
    pub pad_angle: f64,
}

/// Extract polar coord parameters and resolve the panel-relative center +
/// outer radius. Returns `None` for any non-polar coord.
pub(crate) fn polar_geom(ctx: &DrawCtx<'_>) -> Option<PolarGeom> {
    let (start_angle, inner_radius, outer_radius_opt, pad_angle) = match &ctx.spec.coord {
        Some(SpecCoord::Polar { start_angle, inner_radius, outer_radius, pad_angle, .. }) => {
            (*start_angle, *inner_radius, *outer_radius, *pad_angle)
        }
        _ => return None,
    };
    let cx = ctx.panel.plot_area.x + ctx.panel.plot_area.w / 2.0;
    let cy = ctx.panel.plot_area.y + ctx.panel.plot_area.h / 2.0;
    let half_min = ctx.panel.plot_area.w.min(ctx.panel.plot_area.h) / 2.0;
    let outer_radius = outer_radius_opt.unwrap_or(half_min);
    Some(PolarGeom { cx, cy, inner_radius, outer_radius, start_angle, pad_angle })
}

/// Map a radial *data* value to a pixel radius, linearly interpolating the
/// radial domain onto the `[inner_radius, outer_radius]` pixel band. The
/// radial channel is `y` (when `theta="x"`) or `x` (when `theta="y"`);
/// `radius_scale` is the corresponding `ScaleKind`.
///
/// The radial domain is anchored at `0` (a radius is a magnitude from the
/// center, like a bar measured from its baseline): a data value of `0` maps to
/// the inner radius and the domain maximum maps to the outer radius. Using the
/// scale's raw `(lo, hi)` domain would instead pin `lo → inner_radius`, which
/// is wrong for radii (e.g. `r_inner=20` on a `[20, 80]` domain would collapse
/// to the center). A degenerate (non-positive) maximum maps every value to
/// `outer_radius` so a single-radius dataset still draws a visible ring.
pub(crate) fn radius_to_pixel(
    radius_scale: &crate::render::scale_resolve::ScaleKind,
    inner_radius: f64,
    outer_radius: f64,
    value: f64,
) -> f64 {
    let (dlo, dhi) = radius_scale.data_domain().unwrap_or((0.0, 1.0));
    let dmax = dhi.max(dlo).max(0.0);
    if dmax <= 0.0 {
        return outer_radius;
    }
    let t = (value / dmax).clamp(0.0, 1.0);
    inner_radius + t * (outer_radius - inner_radius)
}

/// Build arc (wedge) nodes for pie/donut/annular charts.
///
/// Requires `CoordPolar` — returns empty for any other coord.
///
/// Two render paths, gated by which channels are bound:
///
/// * **Annular** — when `theta2` (angular end) *or* `radius2` (outer radius)
///   is bound, each row becomes an annular wedge spanning `[theta, theta2]`
///   angularly and `[radius, radius2]` radially. Angles are taken directly
///   from the data columns (in radians, offset by `start_angle`); radii map
///   through the radial scale via [`radius_to_pixel`].
/// * **Pie/donut (legacy)** — otherwise each row's angular sweep is
///   proportional to its value in the theta-mapped field, swept from a fixed
///   inner radius to a fixed outer radius. This path is byte-stable with the
///   pre-D7 implementation.
pub fn build(ctx: &DrawCtx<'_>) -> MarkBuildResult {
    let theta_ch = match &ctx.spec.coord {
        Some(SpecCoord::Polar { theta, .. }) => *theta,
        _ => return MarkBuildResult::empty(MarkBatchKind::Arc),
    };
    let Some(geom) = polar_geom(ctx) else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };

    // Channel assignment under the Python polar remapping:
    //   theta="x" → theta=x, theta2=x2, radius=y, radius2=y2
    //   theta="y" → theta=y, theta2=y2, radius=x, radius2=x2
    let (theta_field, theta2_field, radius_field, radius2_field, radius_scale) = match theta_ch {
        PolarThetaChannel::X => (
            ctx.spec.encoding.x.as_ref().map(|e| e.field.as_str()),
            ctx.spec.encoding.x2.as_ref().map(|e| e.field.as_str()),
            ctx.spec.encoding.y.as_ref().map(|e| e.field.as_str()),
            ctx.spec.encoding.y2.as_ref().map(|e| e.field.as_str()),
            &ctx.scales.y,
        ),
        PolarThetaChannel::Y => (
            ctx.spec.encoding.y.as_ref().map(|e| e.field.as_str()),
            ctx.spec.encoding.y2.as_ref().map(|e| e.field.as_str()),
            ctx.spec.encoding.x.as_ref().map(|e| e.field.as_str()),
            ctx.spec.encoding.x2.as_ref().map(|e| e.field.as_str()),
            &ctx.scales.x,
        ),
    };

    // Annular mode: a second angular OR radial extent is bound. Note the
    // legacy pie path always populates a dummy radius field equal to theta
    // (Python adds `enc["y"] = enc["x"]`), so the gate is on *2 channels only.
    if theta2_field.is_some() || radius2_field.is_some() {
        return build_annular(
            ctx, &geom, theta_field, theta2_field, radius_field, radius2_field, radius_scale,
        );
    }

    let Some(field) = theta_field else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };

    let Ok(values) = col_as_f64(ctx.batch, field) else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };

    let (cx, cy, inner_radius, outer_radius, start_angle, pad_angle) = (
        geom.cx, geom.cy, geom.inner_radius, geom.outer_radius, geom.start_angle, geom.pad_angle,
    );

    let total: f64 = values.iter()
        .filter_map(|v| *v)
        .filter(|v| v.is_finite() && *v > 0.0)
        .sum();
    if total <= 0.0 {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    }

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

/// Build annular-wedge nodes from per-row angular and radial extents.
///
/// Each row spans `[theta, theta2]` angularly (radians from the data columns,
/// offset by `start_angle`) and `[radius, radius2]` radially (data values
/// mapped through `radius_scale`). When `theta2` is unbound the legacy gate
/// would not have routed here; this path requires `theta2` to be present for
/// a meaningful angular span (a row with no theta2 collapses to zero sweep
/// and is skipped). When `radius2` is unbound the outer radius falls back to
/// the coord's outer radius (solid-to-edge wedge).
#[allow(clippy::too_many_arguments)]
fn build_annular(
    ctx: &DrawCtx<'_>,
    geom: &PolarGeom,
    theta_field: Option<&str>,
    theta2_field: Option<&str>,
    radius_field: Option<&str>,
    radius2_field: Option<&str>,
    radius_scale: &crate::render::scale_resolve::ScaleKind,
) -> MarkBuildResult {
    let Some(tf) = theta_field else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };
    let Ok(theta) = col_as_f64(ctx.batch, tf) else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };
    let theta2: Option<Vec<Option<f64>>> =
        theta2_field.and_then(|f| col_as_f64(ctx.batch, f).ok());
    let radius: Option<Vec<Option<f64>>> =
        radius_field.and_then(|f| col_as_f64(ctx.batch, f).ok());
    let radius2: Option<Vec<Option<f64>>> =
        radius2_field.and_then(|f| col_as_f64(ctx.batch, f).ok());

    // Per-slice color (mirrors the legacy path).
    let cfield = ctx.spec.encoding.color.as_ref().map(|e| e.field.as_str());
    let color_str: Option<Vec<Option<String>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Categorical { .. }), Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let color_f64: Option<Vec<Option<f64>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Continuous { .. }), Some(f)) => col_as_f64(ctx.batch, f).ok(),
        _ => None,
    };
    let opacity_values: Option<Vec<Option<f64>>> = ctx.spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let meta = MetadataColumns::from_ctx(ctx);

    let mut nodes: Vec<SceneNode> = Vec::with_capacity(theta.len());
    let mut data_indices: Vec<usize> = Vec::with_capacity(theta.len());

    for (i, t0_opt) in theta.iter().enumerate() {
        let Some(t0) = t0_opt.filter(|v| v.is_finite()) else { continue };
        // Angular end: explicit theta2 column. Without it there's no span.
        let t1 = match theta2.as_ref().and_then(|v| v.get(i).copied().flatten()) {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        let angle_start = geom.start_angle + t0 + geom.pad_angle / 2.0;
        let angle_end = geom.start_angle + t1 - geom.pad_angle / 2.0;
        if angle_end <= angle_start { continue; }

        // Radial extents: map data values through the radial scale. Missing
        // inner radius defaults to the coord inner radius; missing outer
        // radius defaults to the coord outer radius.
        let inner_r = match radius.as_ref().and_then(|v| v.get(i).copied().flatten()) {
            Some(rv) if rv.is_finite() => {
                radius_to_pixel(radius_scale, geom.inner_radius, geom.outer_radius, rv)
            }
            _ => geom.inner_radius,
        };
        let outer_r = match radius2.as_ref().and_then(|v| v.get(i).copied().flatten()) {
            Some(rv) if rv.is_finite() => {
                radius_to_pixel(radius_scale, geom.inner_radius, geom.outer_radius, rv)
            }
            _ => geom.outer_radius,
        };

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
        let row_opacity = if let (Some(values), Some(scale)) = (&opacity_values, &ctx.scales.opacity) {
            match values.get(i).copied().flatten().and_then(|v| scale.inner.to_pixel_f64(v)) {
                Some(op) => op,
                None => ctx.mark_style.opacity,
            }
        } else {
            ctx.mark_style.opacity
        };
        let fill_color = with_opacity(fill_base, row_opacity);

        let commands = wedge_path(geom.cx, geom.cy, inner_r, outer_r, angle_start, angle_end);
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
                            value: col.get(row_idx).and_then(|v| v.clone()).unwrap_or_default(),
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
pub(crate) fn wedge_path(
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
            params: Vec::new(),
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

    /// D7: radius_to_pixel anchors the radial domain at 0 — a data value of 0
    /// maps to the inner radius and the domain max maps to the outer radius,
    /// regardless of the scale's lower bound.
    #[test]
    fn radius_to_pixel_anchors_domain_at_zero() {
        // Domain [20, 80] but radii are magnitudes from center: 0 → inner.
        let scale = ScaleKind::Linear(LinearScale::new_internal(
            vec![20.0, 80.0], vec![100.0, 0.0], false, false,
        ));
        // dmax = 80. r=0 → inner (0). r=80 → outer (200). r=40 → 100.
        assert!((radius_to_pixel(&scale, 0.0, 200.0, 0.0) - 0.0).abs() < 1e-9);
        assert!((radius_to_pixel(&scale, 0.0, 200.0, 80.0) - 200.0).abs() < 1e-9);
        assert!((radius_to_pixel(&scale, 0.0, 200.0, 40.0) - 100.0).abs() < 1e-9);
        // r=20 must NOT collapse to the center (the old (lo,hi) bug).
        assert!(radius_to_pixel(&scale, 0.0, 200.0, 20.0) > 1.0);
        // Non-zero inner radius offsets the band.
        assert!((radius_to_pixel(&scale, 50.0, 200.0, 0.0) - 50.0).abs() < 1e-9);
    }

    /// D7: a degenerate (non-positive max) radial domain maps every value to
    /// the outer radius so a single-radius dataset still draws a visible ring.
    #[test]
    fn radius_to_pixel_degenerate_domain_uses_outer() {
        let scale = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 0.0], vec![0.0, 0.0], false, false,
        ));
        assert!((radius_to_pixel(&scale, 0.0, 150.0, 5.0) - 150.0).abs() < 1e-9);
    }

    fn annular_spec() -> ChartSpec {
        // theta=x (t0), theta2=x2 (t1), radius=y (r0), radius2=y2 (r1).
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Arc,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "t0".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "t1".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "r0".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "r1".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
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
            params: Vec::new(),
        }
    }

    /// Count the number of Arc (`A`) commands in a Path node's command list.
    fn arc_count(node: &SceneNode) -> usize {
        if let SceneNode::Path { commands, .. } = node {
            commands.iter().filter(|c| matches!(c, PathCmd::ArcTo { .. })).count()
        } else {
            0
        }
    }

    /// D7-C: a two-ring sunburst with theta/theta2/radius/radius2 produces one
    /// wedge per row, and the outer ring (radius=40) has a non-zero inner
    /// radius while the inner ring (radius=0) starts at the center.
    #[test]
    fn annular_wedges_have_per_row_radii_and_partial_sweeps() {
        let tau = std::f64::consts::TAU;
        let spec = annular_spec();
        // 4 outer-ring rows (r0=40, r1=80) + 2 inner-ring rows (r0=0, r1=40).
        let schema = Arc::new(Schema::new(vec![
            Field::new("t0", DataType::Float64, false),
            Field::new("t1", DataType::Float64, false),
            Field::new("r0", DataType::Float64, false),
            Field::new("r1", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, tau/4.0, tau/2.0, 3.0*tau/4.0, 0.0, tau/2.0])),
            Arc::new(Float64Array::from(vec![tau/4.0, tau/2.0, 3.0*tau/4.0, tau, tau/2.0, tau])),
            Arc::new(Float64Array::from(vec![40.0, 40.0, 40.0, 40.0, 0.0, 0.0])),
            Arc::new(Float64Array::from(vec![80.0, 80.0, 80.0, 80.0, 40.0, 40.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        // Radial scale (y): domain [0, 80], range irrelevant for radius mapping.
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, tau], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 80.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None,
            x2: Some("t1".into()), y2: Some("r1".into()),
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx);

        let paths: Vec<&SceneNode> = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Path { .. }))
            .collect();
        assert_eq!(paths.len(), 6, "expected one wedge per row");

        // No wedge spans the full circle: each is a quarter or half, so the
        // full-circle 2-arc split never triggers (≤ 2 arcs per path).
        for n in &paths {
            assert!(arc_count(n) <= 2, "partial wedge should have ≤ 2 arcs");
        }

        // Outer-ring rows (first 4) have a non-zero inner arc (annular: 2 arcs).
        // Inner-ring rows (last 2, r0=0) collapse to center (1 outer arc).
        let outer_arcs: Vec<usize> = paths[..4].iter().map(|n| arc_count(n)).collect();
        let inner_arcs: Vec<usize> = paths[4..].iter().map(|n| arc_count(n)).collect();
        assert!(outer_arcs.iter().all(|&c| c == 2),
            "outer-ring wedges should be annular (inner + outer arc): {outer_arcs:?}");
        assert!(inner_arcs.iter().all(|&c| c == 1),
            "inner-ring wedges (r0=0) should be solid (single outer arc): {inner_arcs:?}");
    }
}
