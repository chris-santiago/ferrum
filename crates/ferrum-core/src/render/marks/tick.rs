//! mark_tick: four modes —
//!   quantitative x only → x-rug: vertical ticks at panel baseline;
//!   quantitative y only → y-rug: horizontal ticks at left axis edge;
//!   ordinal x + quantitative y → horizontal tick at data y position (boxplot median);
//!   ordinal y + quantitative x → vertical tick at data x position (strip plot).

use crate::render::draw::{col_as_f64, col_as_str, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::ScaleKind;

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_stroke, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let panel = ctx.panel.plot_area;

    // Common setup shared by all four tick modes.
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    let stroke_color = ctx.mark_style.stroke.unwrap_or(ctx.mark_style.fill);
    let default_opacity = ctx.mark_style.opacity;
    let default_stroke_width = ctx.mark_style.stroke_width.max(1.0);

    // Per-row opacity and stroke_width encoding columns.
    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let stroke_width_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_width
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let row_stroke = |i: usize| -> ferrum_scene::StrokeStyle {
        let opacity = opacity_values.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .unwrap_or(default_opacity);
        let width = stroke_width_values.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .unwrap_or(default_stroke_width);
        to_scene_stroke(stroke_color, width, opacity, None, None, None)
    };

    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);
    let mut nodes = Vec::new();
    let mut indices = Vec::new();

    let xf_opt = x_field(ctx, spec);

    // Quantitative y only → y-rug: horizontal ticks at left axis edge.
    if xf_opt.is_none() {
        if let Some(yf) = y_field(ctx, spec) {
            let tick_len = ctx.theme.tick_size * 2.0;
            let ys = match col_as_f64(ctx.batch, yf) {
                Ok(v) => v,
                Err(_) => return MarkBuildResult {
                    kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
                    tooltips: None, hrefs: None, descriptions: None,
                },
            };
            let baseline_x = panel.x;
            for i in 0..ys.len() {
                let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
                let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
                let py = py + y_offsets[i];
                let bx = baseline_x + x_offsets[i];
                nodes.push(SceneNode::Line {
                    x1: bx, y1: py,
                    x2: bx + tick_len, y2: py,
                    style: row_stroke(i),
                });
                indices.push(i);
            }
        }
        return MarkBuildResult {
            kind: MarkBatchKind::Tick, nodes, data_indices: Some(indices),
            tooltips, hrefs, descriptions,
        };
    }

    let xf = xf_opt.expect("invariant: xf_opt is Some — None case returned above");

    // Ordinal x + quantitative y → horizontal tick at data y position.
    if matches!(&ctx.scales.x, ScaleKind::Ordinal(_)) {
        if let Some(yf) = y_field(ctx, spec) {
            let xs = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return MarkBuildResult {
                kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
                tooltips: None, hrefs: None, descriptions: None,
            }};
            let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return MarkBuildResult {
                kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
                tooltips: None, hrefs: None, descriptions: None,
            }};
            let n_cats = {
                let mut set = std::collections::HashSet::<&str>::new();
                for v in xs.iter().flatten() { set.insert(v.as_str()); }
                set.len().max(1)
            };
            let tick_half = (panel.w / n_cats as f64) * ctx.mark_style.band_size.unwrap_or(0.3);
            for i in 0..xs.len() {
                let xv = match &xs[i] { Some(s) => s.as_str(), None => continue };
                let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
                let cx = match ctx.scales.x.to_pixel_str(xv) { Some(p) => p, None => continue };
                let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
                let cx = cx + x_offsets[i];
                let py = py + y_offsets[i];
                nodes.push(SceneNode::Line {
                    x1: cx - tick_half,
                    y1: py,
                    x2: cx + tick_half,
                    y2: py,
                    style: row_stroke(i),
                });
                indices.push(i);
            }
            return MarkBuildResult {
                kind: MarkBatchKind::Tick,
                nodes,
                data_indices: Some(indices),
                tooltips,
                hrefs,
                descriptions,            };
        }
    }

    // Ordinal y + quantitative x → vertical tick at data x position (strip plot).
    if matches!(&ctx.scales.y, ScaleKind::Ordinal(_)) {
        if let Some(yf) = y_field(ctx, spec) {
            let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return MarkBuildResult {
                kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
                tooltips: None, hrefs: None, descriptions: None,
            }};
            let ys = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return MarkBuildResult {
                kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
                tooltips: None, hrefs: None, descriptions: None,
            }};
            let n_cats = {
                let mut set = std::collections::HashSet::<&str>::new();
                for v in ys.iter().flatten() { set.insert(v.as_str()); }
                set.len().max(1)
            };
            let tick_half = (panel.h / n_cats as f64) * ctx.mark_style.band_size.unwrap_or(0.3);
            for i in 0..xs.len() {
                let xv = match xs[i] { Some(v) if v.is_finite() => v, _ => continue };
                let yv = match &ys[i] { Some(s) => s.as_str(), None => continue };
                let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
                let cy = match ctx.scales.y.to_pixel_str(yv) { Some(p) => p, None => continue };
                let px = px + x_offsets[i];
                let cy = cy + y_offsets[i];
                nodes.push(SceneNode::Line {
                    x1: px,
                    y1: cy - tick_half,
                    x2: px,
                    y2: cy + tick_half,
                    style: row_stroke(i),
                });
                indices.push(i);
            }
            return MarkBuildResult {
                kind: MarkBatchKind::Tick,
                nodes,
                data_indices: Some(indices),
                tooltips,
                hrefs,
                descriptions,            };
        }
    }

    // Quantitative x → rug-style vertical tick at panel baseline.
    let tick_len = ctx.theme.tick_size * 2.0;
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return MarkBuildResult {
        kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
        tooltips: None, hrefs: None, descriptions: None,
    }};
    let baseline_y = panel.y + panel.h;
    for (i, xopt) in xs.iter().enumerate() {
        let xv = match xopt { Some(v) if v.is_finite() => *v, _ => continue };
        let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
        let px = px + x_offsets[i];
        let by = baseline_y + y_offsets[i];
        nodes.push(SceneNode::Line {
            x1: px,
            y1: by,
            x2: px,
            y2: by - tick_len,
            style: row_stroke(i),
        });
        indices.push(i);
    }
    let _ = y_field(ctx, spec);

    MarkBuildResult {
        kind: MarkBatchKind::Tick,
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
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn tick_ordinal_x_emits_horizontal_tick_at_y() {
        // Phase 10c-pre: ordinal x + quantitative y → horizontal tick (boxplot median).
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "median".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("cat",    arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("median", arrow::datatypes::DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![3.0, 5.0, 7.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        // 3 horizontal ticks — one per (cat, median) row.
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Line { .. })).count(), 3, "expected 3 horizontal tick lines");
        // Lines are horizontal: x1 != x2 for at least one line.
        let has_horizontal = result.nodes.iter().any(|n| {
            if let ferrum_scene::SceneNode::Line { x1, x2, .. } = n { (x1 - x2).abs() > f64::EPSILON } else { false }
        });
        assert!(has_horizontal, "ticks must have different x1 and x2 endpoints");
    }

    #[test]
    fn tick_emits_one_line_per_row() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
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
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Line { .. })).count(), 3);
    }

    #[test]
    fn tick_uses_stroke_color_when_mark_style_stroke_is_set() {
        // Composite structural ticks (boxplot caps, median, errorbar caps) pass
        // stroke: "theme:label" via mark_kwargs. After resolve_mark_style,
        // mark_style.stroke = Some(label_color). tick.rs must use that stroke
        // color, not fall back to mark_style.fill (which is mark_color = blue).
        use arrow::array::StringArray;
        use crate::spec::mark_style::MarkKwargsSpec;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("cat", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("val", arrow::datatypes::DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(Float64Array::from(vec![5.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();

        // Stroke = #aabbcc (distinctive; different from mark_color and fill)
        let overrides = MarkKwargsSpec { stroke: Some("#aabbcc".into()), ..Default::default() };
        let mark_style = crate::render::draw::resolve_mark_style(Some(&overrides), &theme, &Mark::Tick);
        // Confirm fill is NOT #aabbcc (it's still mark_color blue)
        assert_ne!(mark_style.fill.red, 0xAA, "fill should not be the stroke color");
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let line = result.nodes.iter().find_map(|n| {
            if let ferrum_scene::SceneNode::Line { style, .. } = n { Some(style.clone()) } else { None }
        }).expect("expected at least one Line node");
        assert_eq!(line.color.r, 0xAA, "tick stroke must use mark_style.stroke, not mark_style.fill");
        assert_eq!(line.color.g, 0xBB, "tick stroke must use mark_style.stroke, not mark_style.fill");
        assert_eq!(line.color.b, 0xCC, "tick stroke must use mark_style.stroke, not mark_style.fill");
    }
}
