//! mark_rule: reference lines. Four modes:
//!   y only → horizontal span; x only → vertical span;
//!   ordinal x + y + y2 → ranged vertical segment (boxplot whisker, Phase 10c-pre).
//!   ordinal y + x + x2 → ranged horizontal segment (Phase 10d-pre,
//!     feature-importance error bars).

use crate::render::draw::{col_as_f64, col_as_str, x_field, y_field, DrawCtx};

/// Build a per-row stroke style for rule segments, applying encoding column values.
fn rule_stroke_style(
    ctx: &DrawCtx,
    i: usize,
    so_vals: &Option<Vec<Option<f64>>>,
    sw_vals: &Option<Vec<Option<f64>>>,
    sd_vals: &Option<Vec<Option<f64>>>,
) -> ferrum_scene::StrokeStyle {
    use crate::render::draw::to_scene_stroke;

    let stroke_opacity = so_vals.as_ref()
        .and_then(|v| v.get(i).copied().flatten())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(1.0);
    let stroke_width = sw_vals.as_ref()
        .and_then(|v| v.get(i).copied().flatten())
        .filter(|v| *v >= 0.0 && v.is_finite())
        .unwrap_or(ctx.mark_style.stroke_width);
    let dash_vec: Option<Vec<f64>> = sd_vals.as_ref()
        .and_then(|v| v.get(i).copied().flatten())
        .filter(|v| v.is_finite())
        .and_then(|idx| {
            let idx = (idx.round() as i64).clamp(0, 3);
            match idx {
                1 => Some(vec![6.0, 3.0]),
                2 => Some(vec![2.0, 3.0]),
                3 => Some(vec![6.0, 3.0, 2.0, 3.0]),
                _ => None,
            }
        });
    let effective_dash = dash_vec.as_deref().or(ctx.mark_style.stroke_dash.as_deref());
    let stroke_color = ctx.mark_style.stroke.unwrap_or(ctx.mark_style.fill);
    let mut style = to_scene_stroke(stroke_color, stroke_width, 1.0, effective_dash, None, None);
    style.stroke_opacity = stroke_opacity;
    style
}

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let panel = ctx.panel.plot_area;

    // Per-row stroke channel vectors.
    let so_vals: Option<Vec<Option<f64>>> = spec.encoding.stroke_opacity.as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let sw_vals: Option<Vec<Option<f64>>> = spec.encoding.stroke_width.as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let sd_vals: Option<Vec<Option<f64>>> = spec.encoding.stroke_dash.as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let empty = || MarkBuildResult {
        kind: MarkBatchKind::Rule,
        nodes: vec![],
        data_indices: Some(vec![]),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    };

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    let xf_opt = x_field(ctx, spec);
    let yf_opt = y_field(ctx, spec);
    let y2f_opt = spec.encoding.y2.as_ref().map(|e| e.field.as_str());
    let x2f_opt = spec.encoding.x2.as_ref().map(|e| e.field.as_str());

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

    // Ranged rule: ordinal x + quantitative y + y2 → vertical segment per row.
    if let (Some(xf), Some(yf), Some(y2f)) = (xf_opt, yf_opt, y2f_opt) {
        if let Ok(xs) = col_as_str(ctx.batch, xf) {
            let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty() };
            let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return empty() };
            for i in 0..xs.len() {
                let xv = match &xs[i] { Some(s) => s.as_str(), None => continue };
                let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
                let y2v = match y2s[i] { Some(v) if v.is_finite() => v, _ => continue };
                let px = match ctx.scales.x.to_pixel_str(xv) { Some(p) => p, None => continue };
                let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
                let py2 = match ctx.scales.y.to_pixel_f64(y2v) { Some(p) => p, None => continue };
                let px = px + x_offsets[i];
                nodes.push(SceneNode::Line {
                    x1: px,
                    y1: py + y_offsets[i],
                    x2: px,
                    y2: py2 + y_offsets[i],
                    style: rule_stroke_style(ctx, i, &so_vals, &sw_vals, &sd_vals),
                });
                indices.push(i);
            }
            return MarkBuildResult {
                kind: MarkBatchKind::Rule,
                nodes,
                data_indices: Some(indices),
                tooltips,
                hrefs,
                descriptions,            };
        }
    }

    // Ranged rule: ordinal y + quantitative x + x2 → horizontal segment per row.
    if let (Some(yf), Some(xf), Some(x2f)) = (yf_opt, xf_opt, x2f_opt) {
        if let Ok(ys) = col_as_str(ctx.batch, yf) {
            let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
            let x2s = match col_as_f64(ctx.batch, x2f) { Ok(v) => v, Err(_) => return empty() };
            for i in 0..ys.len() {
                let yv = match &ys[i] { Some(s) => s.as_str(), None => continue };
                let xv = match xs[i] { Some(v) if v.is_finite() => v, _ => continue };
                let x2v = match x2s[i] { Some(v) if v.is_finite() => v, _ => continue };
                let py = match ctx.scales.y.to_pixel_str(yv) { Some(p) => p, None => continue };
                let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
                let px2 = match ctx.scales.x.to_pixel_f64(x2v) { Some(p) => p, None => continue };
                let py = py + y_offsets[i];
                nodes.push(SceneNode::Line {
                    x1: px + x_offsets[i],
                    y1: py,
                    x2: px2 + x_offsets[i],
                    y2: py,
                    style: rule_stroke_style(ctx, i, &so_vals, &sw_vals, &sd_vals),
                });
                indices.push(i);
            }
            return MarkBuildResult {
                kind: MarkBatchKind::Rule,
                nodes,
                data_indices: Some(indices),
                tooltips,
                hrefs,
                descriptions,            };
        }
    }

    // Horizontal span: y only (no x), or y + x inherited from chart-level encoding.
    if let Some(yf) = yf_opt {
        if let Ok(ys) = col_as_f64(ctx.batch, yf) {
            if y2f_opt.is_none() {
                for (i, yopt) in ys.iter().enumerate() {
                    let yv = match yopt {
                        Some(v) if v.is_finite() => *v,
                        _ => continue,
                    };
                    let py = match ctx.scales.y.to_pixel_f64(yv) {
                        Some(p) => p, None => continue,
                    };
                    let py = py + y_offsets[i];
                    nodes.push(SceneNode::Line {
                        x1: panel.x,
                        y1: py,
                        x2: panel.x + panel.w,
                        y2: py,
                        style: rule_stroke_style(ctx, i, &so_vals, &sw_vals, &sd_vals),
                    });
                    indices.push(i);
                }
                return MarkBuildResult {
                    kind: MarkBatchKind::Rule,
                    nodes,
                    data_indices: Some(indices),
                    tooltips,
                    hrefs,
                    descriptions,                };
            }
        }
    }

    // Vertical span: x only (no y).
    if let (Some(xf), None) = (xf_opt, yf_opt) {
        let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
        for (i, xopt) in xs.iter().enumerate() {
            let xv = match xopt { Some(v) if v.is_finite() => *v, _ => continue };
            let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            let px = px + x_offsets[i];
            nodes.push(SceneNode::Line {
                x1: px,
                y1: panel.y,
                x2: px,
                y2: panel.y + panel.h,
                style: rule_stroke_style(ctx, i, &so_vals, &sw_vals, &sd_vals),
            });
            indices.push(i);
        }
    }

    MarkBuildResult {
        kind: MarkBatchKind::Rule,
        nodes,
        data_indices: Some(indices),
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
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn ranged_rule_emits_vertical_segments_for_ordinal_x() {
        // Phase 10c-pre: ordinal x + y + y2 → vertical segment per row (boxplot whisker).
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", arrow::datatypes::DataType::Utf8, false),
            Field::new("lo",  arrow::datatypes::DataType::Float64, false),
            Field::new("hi",  arrow::datatypes::DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![5.0, 8.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count(), 2, "expected 2 ranged rule lines");
    }

    #[test]
    fn ranged_rule_emits_horizontal_segments_for_ordinal_y() {
        // Phase 10d-pre: ordinal y + x + x2 → horizontal segment per row
        // (feature-importance error bars on horizontal-bar charts).
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                y: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                x: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", arrow::datatypes::DataType::Utf8, false),
            Field::new("lo",  arrow::datatypes::DataType::Float64, false),
            Field::new("hi",  arrow::datatypes::DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![5.0, 8.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count(), 2, "expected 2 horizontal-ranged rule lines");
    }

    #[test]
    fn y_only_rule_emits_horizontal_lines() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: None,
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let mut spec_for_scales = spec.clone();
        spec_for_scales.encoding.x = Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() });
        let (scales, _) = resolve_scales(&spec_for_scales, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count(), 2);
    }
}
