//! mark_area: filled region between y(x) and the x-axis baseline. Single area
//! over all rows when no color encoding; one area per category otherwise.

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::ColorScale;

/// Build `Vec<PathCmd>` for the top-edge line using the given interpolation
/// method. Mirrors `build_top_line_path` but emits structured commands.
fn build_top_line_cmds(top: &[(f64, f64)], interpolate: Option<&str>) -> Vec<ferrum_scene::PathCmd> {
    use ferrum_scene::PathCmd;
    if top.is_empty() { return Vec::new(); }
    let method = interpolate.unwrap_or("linear");
    let mut cmds = Vec::with_capacity(top.len() * 2);
    cmds.push(PathCmd::MoveTo { x: top[0].0, y: top[0].1 });
    for i in 1..top.len() {
        let (px, py) = top[i - 1];
        let (cx, cy) = top[i];
        match method {
            "step" => {
                let mid_x = (px + cx) / 2.0;
                cmds.push(PathCmd::LineTo { x: mid_x, y: py });
                cmds.push(PathCmd::LineTo { x: mid_x, y: cy });
                cmds.push(PathCmd::LineTo { x: cx, y: cy });
            }
            "step-before" => {
                cmds.push(PathCmd::LineTo { x: px, y: cy });
                cmds.push(PathCmd::LineTo { x: cx, y: cy });
            }
            "step-after" => {
                cmds.push(PathCmd::LineTo { x: cx, y: py });
                cmds.push(PathCmd::LineTo { x: cx, y: cy });
            }
            _ => {
                cmds.push(PathCmd::LineTo { x: cx, y: cy });
            }
        }
    }
    cmds
}

/// Build closed area path commands: top edge (with interpolation) + baseline closure.
fn build_area_cmds(top: &[(f64, f64)], baseline: f64, interpolate: Option<&str>) -> Vec<ferrum_scene::PathCmd> {
    use ferrum_scene::PathCmd;
    let mut cmds = build_top_line_cmds(top, interpolate);
    let last_x = top[top.len() - 1].0;
    let x0 = top[0].0;
    cmds.push(PathCmd::LineTo { x: last_x, y: baseline });
    cmds.push(PathCmd::LineTo { x: x0, y: baseline });
    cmds.push(PathCmd::Close);
    cmds
}

/// Build closed stacked area path commands: top edge forward, bottom edge reversed.
fn build_stacked_area_cmds(top: &[(f64, f64)], bottom: &[(f64, f64)], interpolate: Option<&str>) -> Vec<ferrum_scene::PathCmd> {
    use ferrum_scene::PathCmd;
    let mut cmds = build_top_line_cmds(top, interpolate);
    for &(x, y) in bottom.iter().rev() {
        cmds.push(PathCmd::LineTo { x, y });
    }
    cmds.push(PathCmd::Close);
    cmds
}

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{
        to_scene_fill_stroke, MarkBuildResult, MetadataColumns,
    };
    use ferrum_scene::MarkBatchKind;

    let empty = || MarkBuildResult {
        kind: MarkBatchKind::Area,
        nodes: vec![],
        data_indices: Some(vec![]),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    };

    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return empty(),
    };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty() };

    // y2 column: when bound, the area fills the band between y and y2.
    let y2f_opt = spec.encoding.y2.as_ref().map(|e| e.field.as_str());
    let y2s_opt: Option<Vec<Option<f64>>> = y2f_opt
        .and_then(|f| col_as_f64(ctx.batch, f).ok());
    let has_y2 = y2s_opt.is_some();

    let baseline_y = ctx.panel.plot_area.y + ctx.panel.plot_area.h;

    let cf = color_field(ctx, spec);
    let color_values = cf.and_then(|f| col_as_str(ctx.batch, f).ok());

    let groups: Vec<(Option<String>, Vec<usize>)> = match (color_values.as_ref(), &ctx.scales.color) {
        (Some(values), Some(_)) => {
            let mut g: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in values.iter().enumerate() {
                let key = v.clone();
                match g.iter().position(|(k, _)| k == &key) {
                    Some(p) => g[p].1.push(i),
                    None => g.push((key, vec![i])),
                }
            }
            g
        }
        _ => vec![(None, (0..xs.len()).collect())],
    };

    // Phase 9c — per-row position-adjustment pixel offsets (Stack).
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    // Stack writes __stack_y_base__ with per-segment baselines.
    let stack_bases: Option<Vec<Option<f64>>> = ctx.batch.schema()
        .index_of("__stack_y_base__")
        .ok()
        .and_then(|i| {
            ctx.batch.column(i).as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .map(|a| a.iter().collect())
        });
    let is_stacked = stack_bases.is_some();

    // Stacked areas use opaque fills so each band is visually distinct.
    let base_opacity = if is_stacked { 1.0 } else { ctx.mark_style.opacity };

    // Per-row opacity/fill_opacity channels — sampled at the first row of each group.
    let opacity_vals: Option<Vec<Option<f64>>> = spec.encoding.opacity.as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let fill_opacity_vals: Option<Vec<Option<f64>>> = spec.encoding.fill_opacity.as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let interpolate = ctx.mark_style.interpolate.as_deref();

    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut data_indices = Vec::new();

    for (key, rows) in groups {
        let mut top: Vec<(f64, f64)> = Vec::new();
        let mut bottom: Vec<(f64, f64)> = Vec::new();
        let mut row_indices: Vec<usize> = Vec::new();
        for i in rows {
            let (xv, yv) = match (xs[i], ys[i]) {
                (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a, b),
                _ => continue,
            };
            let cx = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            let cy = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
            let xo = x_offsets.get(i).copied().unwrap_or(0.0);
            let yo = y_offsets.get(i).copied().unwrap_or(0.0);
            let cx = cx + xo;
            let cy_top = cy + yo;
            top.push((cx, cy_top));
            row_indices.push(i);
            if let Some(ref y2s) = y2s_opt {
                // y2 band: bottom edge comes from the y2 column.
                let by = y2s.get(i)
                    .and_then(|v| *v)
                    .filter(|v| v.is_finite())
                    .and_then(|v| ctx.scales.y.to_pixel_f64(v))
                    .unwrap_or(baseline_y);
                bottom.push((cx, by + yo));
            } else if let Some(ref bases) = stack_bases {
                let base_y = bases.get(i)
                    .and_then(|v| *v)
                    .and_then(|b| ctx.scales.y.to_pixel_f64(b))
                    .unwrap_or(baseline_y);
                bottom.push((cx, base_y));
            }
        }
        if top.len() < 2 { continue; }

        // Sample opacity and fill_opacity from the first row of the group.
        let first = row_indices.first().copied().unwrap_or(0);
        let effective_opacity = opacity_vals.as_ref()
            .and_then(|v| v.get(first).copied().flatten())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(base_opacity);
        let group_fill_opacity = fill_opacity_vals.as_ref()
            .and_then(|v| v.get(first).copied().flatten())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(1.0);

        let cmds = if is_stacked || has_y2 {
            build_stacked_area_cmds(&top, &bottom, interpolate)
        } else {
            build_area_cmds(&top, baseline_y, interpolate)
        };
        let fill = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale)) => {
                let base = scale.lookup(v).unwrap_or(ctx.mark_style.fill);
                with_opacity(base, effective_opacity)
            }
            _ => with_opacity(ctx.mark_style.fill, effective_opacity),
        };
        let stroke_color = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale)) => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
            _ => ctx.mark_style.fill,
        };
        let mut style = to_scene_fill_stroke(
            Some(fill),
            ctx.mark_style.stroke,
            ctx.mark_style.stroke_width,
            1.0,
            None,
        );
        style.fill_opacity = group_fill_opacity;
        nodes.push(ferrum_scene::SceneNode::Path {
            commands: cmds,
            style,
            closed: true,
        });

        // S9: line border on top of the area fill.
        if ctx.mark_style.line_border == Some(true) {
            let line_cmds = build_top_line_cmds(&top, interpolate);
            let border_style = to_scene_fill_stroke(
                None,
                Some(stroke_color),
                ctx.mark_style.stroke_width.max(1.0),
                1.0,
                None,
            );
            nodes.push(ferrum_scene::SceneNode::Path {
                commands: line_cmds,
                style: border_style,
                closed: false,
            });
        }

        // S10: border lines on both top and bottom edges.
        if ctx.mark_style.borders == Some(true) {
            let top_cmds = build_top_line_cmds(&top, interpolate);
            let sw = ctx.mark_style.stroke_width.max(1.0);
            let border_style = to_scene_fill_stroke(
                None,
                Some(stroke_color),
                sw,
                1.0,
                None,
            );
            nodes.push(ferrum_scene::SceneNode::Path {
                commands: top_cmds,
                style: border_style.clone(),
                closed: false,
            });

            // Bottom edge: trace the y2 edge when bound, else flat baseline.
            let bottom_cmds = if has_y2 && !bottom.is_empty() {
                build_top_line_cmds(&bottom, interpolate)
            } else {
                let x0 = top[0].0;
                let x_last = top[top.len() - 1].0;
                vec![
                    ferrum_scene::PathCmd::MoveTo { x: x0, y: baseline_y },
                    ferrum_scene::PathCmd::LineTo { x: x_last, y: baseline_y },
                ]
            };
            nodes.push(ferrum_scene::SceneNode::Path {
                commands: bottom_cmds,
                style: border_style,
                closed: false,
            });
        }

        data_indices.extend(row_indices);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Area,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use ferrum_scene::SceneNode;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn area_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(), mark: Mark::Area,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        }
    }

    #[test]
    fn area_emits_one_path_with_z_close() {
        let spec = area_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0,1.0,2.0,3.0,4.0])),
            Arc::new(Float64Array::from(vec![0.0,1.0,2.0,3.0,4.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Area);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Path { .. })).count(), 1);
        // Area path must be closed.
        let is_closed = result.nodes.iter().any(|n| matches!(n, SceneNode::Path { closed: true, .. }));
        assert!(is_closed, "area path must be closed");
    }

    #[test]
    fn area_uses_translucent_fill() {
        let spec = area_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Area);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        // At least one path must have a translucent fill (alpha < 255).
        let has_translucent = result.nodes.iter().any(|n| {
            if let SceneNode::Path { style, .. } = n {
                style.fill.map_or(false, |c| c.a < 255)
            } else {
                false
            }
        });
        assert!(has_translucent, "expected area path with translucent fill");
    }

    /// D3: mark_area with y + y2 must produce a band polygon, not a baseline fill.
    #[test]
    fn area_y2_forms_band_not_baseline_fill() {
        use crate::spec::encoding::{DataType as SDT, EncodingSpec};

        let spec_with_y2 = ChartSpec {
            data: DataRef::default(), mark: Mark::Area,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "lo".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None,
            position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let spec_no_y2 = ChartSpec {
            encoding: Encoding {
                y2: None,
                ..spec_with_y2.encoding.clone()
            },
            ..spec_with_y2.clone()
        };

        // lo=[2,3,4,3], hi=[5,6,7,6]: band area.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",  DataType::Float64, false),
            Field::new("lo", DataType::Float64, false),
            Field::new("hi", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![2.0, 3.0, 4.0, 3.0])),
            Arc::new(Float64Array::from(vec![5.0, 6.0, 7.0, 6.0])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };

        let (scales_y2, _) = resolve_scales(&spec_with_y2, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Area);

        let ctx_y2 = DrawCtx {
            spec: &spec_with_y2, panel: &panel, theme: &theme,
            scales: &scales_y2, batch: &batch, mark_style: &mark_style,
        };
        let result_y2 = super::build(&ctx_y2);

        let (scales_no_y2, _) = resolve_scales(&spec_no_y2, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let ctx_no_y2 = DrawCtx {
            spec: &spec_no_y2, panel: &panel, theme: &theme,
            scales: &scales_no_y2, batch: &batch, mark_style: &mark_style,
        };
        let result_no_y2 = super::build(&ctx_no_y2);

        // Both renders must produce at least one closed path.
        let path_y2 = result_y2.nodes.iter().find_map(|n| {
            if let SceneNode::Path { commands, closed: true, .. } = n { Some(commands.clone()) } else { None }
        }).expect("y2 area must emit a closed path");
        let path_no_y2 = result_no_y2.nodes.iter().find_map(|n| {
            if let SceneNode::Path { commands, closed: true, .. } = n { Some(commands.clone()) } else { None }
        }).expect("no-y2 area must emit a closed path");

        // The two paths must differ: y2 changes the shape structurally.
        assert_ne!(path_y2, path_no_y2,
            "mark_area with y2 must produce a different path than without y2");

        // The band area path should NOT contain the axis baseline (y=100.0) as a
        // coordinate — it closes along the hi edge, not the baseline.
        let baseline_pixel = 100.0_f64;
        let has_baseline = path_y2.iter().any(|cmd| {
            let y = match cmd {
                ferrum_scene::PathCmd::MoveTo { y, .. } => *y,
                ferrum_scene::PathCmd::LineTo { y, .. } => *y,
                _ => return false,
            };
            (y - baseline_pixel).abs() <= 2.0
        });
        assert!(!has_baseline,
            "area band path must not include the axis baseline pixel {baseline_pixel}; path={path_y2:?}");
    }
}
