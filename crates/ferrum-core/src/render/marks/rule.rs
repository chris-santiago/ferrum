//! mark_rule: reference lines. Four modes:
//!   y only → horizontal span; x only → vertical span;
//!   ordinal x + y + y2 → ranged vertical segment (boxplot whisker, Phase 10c-pre).
//!   ordinal y + x + x2 → ranged horizontal segment (Phase 10d-pre,
//!     feature-importance error bars).

use crate::render::color::Color;
use crate::render::draw::{
    col_as_f64, col_as_str, color_field, resolve_stroke_color, resolve_stroke_dash, x_field,
    y_field, DrawCtx,
};

/// Resolve a per-row stroke color from the color encoding + color scale, if both
/// are present. Each row's category value is mapped through `ctx.scales.color`
/// (the same path `line.rs`/`point.rs` use). Returns `None` when there is no
/// color encoding, so callers fall back to the constant mark-style stroke.
fn rule_color_values(ctx: &DrawCtx) -> Option<Vec<Option<Color>>> {
    let field = color_field(ctx, ctx.spec)?;
    let scale = ctx.scales.color.as_ref()?;
    let cats = col_as_str(ctx.batch, field).ok()?;
    Some(
        cats.iter()
            .map(|c| c.as_deref().and_then(|v| scale.lookup(v)))
            .collect(),
    )
}

/// Build a per-row stroke style for rule segments, applying encoding column values.
fn rule_stroke_style(
    ctx: &DrawCtx,
    i: usize,
    so_vals: &Option<Vec<Option<f64>>>,
    sw_vals: &Option<Vec<Option<f64>>>,
    sd_vals: &Option<Vec<Option<f64>>>,
    opacity_vals: &Option<Vec<Option<f64>>>,
    color_vals: &Option<Vec<Option<Color>>>,
) -> ferrum_scene::StrokeStyle {
    use crate::render::color::with_opacity;
    use crate::render::draw::to_scene_stroke;

    let stroke_opacity = so_vals.as_ref()
        .and_then(|v| v.get(i).copied().flatten())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(1.0);
    let opacity = opacity_vals.as_ref()
        .and_then(|v| v.get(i).copied().flatten())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(ctx.mark_style.opacity);
    let stroke_width = sw_vals.as_ref()
        .and_then(|v| v.get(i).copied().flatten())
        .filter(|v| *v >= 0.0 && v.is_finite())
        .unwrap_or(ctx.mark_style.stroke_width);
    let dash_vec: Option<Vec<f64>> = sd_vals.as_ref()
        .and_then(|v| v.get(i).copied().flatten())
        .filter(|v| v.is_finite())
        .and_then(resolve_stroke_dash);
    let effective_dash = dash_vec.as_deref().or(ctx.mark_style.stroke_dash.as_deref());
    // Precedence (explicit constant stroke > per-row color > theme > fill) lives
    // in `resolve_stroke_color`. An explicit `stroke=` in mark_kwargs must not be
    // overridden by a color encoding inherited from a parent chart (e.g. boxplot
    // whiskers keep their neutral gray even when the chart encodes `hue`).
    let row_color = color_vals
        .as_ref()
        .and_then(|v| v.get(i).copied().flatten());
    let base_color = resolve_stroke_color(ctx.mark_style, row_color);
    let stroke_color = with_opacity(base_color, opacity);
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
    let opacity_vals: Option<Vec<Option<f64>>> = spec.encoding.opacity.as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let color_vals = rule_color_values(ctx);

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
                    style: rule_stroke_style(ctx, i, &so_vals, &sw_vals, &sd_vals, &opacity_vals, &color_vals),
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
                    style: rule_stroke_style(ctx, i, &so_vals, &sw_vals, &sd_vals, &opacity_vals, &color_vals),
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
                        style: rule_stroke_style(ctx, i, &so_vals, &sw_vals, &sd_vals, &opacity_vals, &color_vals),
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
                style: rule_stroke_style(ctx, i, &so_vals, &sw_vals, &sd_vals, &opacity_vals, &color_vals),
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

    #[test]
    fn ranged_rule_resolves_per_row_color_encoding() {
        // mark_rule(...).encode(color="dir:N") on a ranged vertical rule must
        // map each row's category through the color scale, yielding distinct
        // stroke colors per category (candlestick wicks colored up/down).
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "dir".into(), type_: Some(crate::spec::encoding::DataType::Nominal), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("lo",  DataType::Float64, false),
            Field::new("hi",  DataType::Float64, false),
            Field::new("dir", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![5.0, 8.0])),
            Arc::new(StringArray::from(vec!["up", "down"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let strokes: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { style, .. } => Some((style.color.r, style.color.g, style.color.b)),
            _ => None,
        }).collect();
        assert_eq!(strokes.len(), 2, "expected 2 ranged rule lines");
        assert_ne!(strokes[0], strokes[1], "color encoding must yield distinct per-row stroke colors");
    }

    #[test]
    fn ranged_rule_explicit_stroke_wins_over_color_encoding() {
        // Regression: an explicit constant stroke= in mark_kwargs must NOT be
        // overridden by a per-row color encoding inherited from a parent chart
        // (e.g. boxplot whiskers set stroke="theme:label" but catplot encodes
        // hue via color — the whisker must stay gray, not turn per-row accent).
        use crate::spec::mark_style::MarkKwargsSpec;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
                // Color encoding present (simulates chart-level hue=species).
                color: Some(EncodingSpec { field: "dir".into(), type_: Some(crate::spec::encoding::DataType::Nominal), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("lo",  DataType::Float64, false),
            Field::new("hi",  DataType::Float64, false),
            Field::new("dir", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![5.0, 8.0])),
            Arc::new(StringArray::from(vec!["up", "down"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        // Explicit constant stroke override — simulates what boxplot whisker layers do.
        let overrides = MarkKwargsSpec { stroke: Some("#6b7280".into()), stroke_dash: Some(vec![]), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Rule);
        assert!(mark_style.stroke_is_user_set, "stroke_is_user_set must be true after explicit override");
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let strokes: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { style, .. } => Some((style.color.r, style.color.g, style.color.b)),
            _ => None,
        }).collect();
        assert_eq!(strokes.len(), 2, "expected 2 ranged rule lines");
        // Both rows must use the explicit constant stroke, NOT per-row accent colors.
        assert_eq!(strokes[0], strokes[1], "explicit constant stroke must win over per-row color encoding");
        // And the color must match the explicit override (#6b7280).
        assert_eq!(strokes[0], (0x6b, 0x72, 0x80), "stroke must be the explicitly set color, not a per-row accent");
    }

    #[test]
    fn ranged_rule_without_color_uses_constant_stroke() {
        // No color encoding → both rules share the constant mark-style stroke
        // (no regression for the candlestick-without-color case).
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
            Field::new("cat", DataType::Utf8, false),
            Field::new("lo",  DataType::Float64, false),
            Field::new("hi",  DataType::Float64, false),
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
        let strokes: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { style, .. } => Some((style.color.r, style.color.g, style.color.b)),
            _ => None,
        }).collect();
        assert_eq!(strokes.len(), 2);
        assert_eq!(strokes[0], strokes[1], "no color encoding must use a single constant stroke");
    }
}
