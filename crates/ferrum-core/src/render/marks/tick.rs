//! mark_tick: six modes —
//!   quantitative x only → x-rug: vertical ticks at panel baseline;
//!   quantitative y only → y-rug: horizontal ticks at left axis edge;
//!   ordinal x + quantitative y → horizontal tick at data y position (boxplot median);
//!   ordinal y + quantitative x → vertical tick at data x position (strip plot);
//!   ordinal y only → horizontal crossbars centered on each category band;
//!   ordinal x only → vertical crossbars centered on each category band.

use crate::render::draw::{col_as_f64, col_as_positional_category_str, x_field, y_field, DrawCtx};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::opacity::{OpacityFallback, OpacityResolver};
use crate::render::scale_resolve::ScaleKind;

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_stroke, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let panel = ctx.panel.plot_area;

    // Common setup shared by all four tick modes.
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    let stroke_color = ctx.mark_style.stroke.unwrap_or(ctx.mark_style.fill);
    let default_stroke_width = ctx.mark_style.stroke_width.max(1.0);

    // Per-row opacity via the shared OpacityResolver (C7); stroke_width stays
    // local. Tick is a stroke-only mark (no fill / stroke_opacity columns), so
    // only the resolver's `opacity` slot is read.
    let opacity_res =
        OpacityResolver::load(ctx, OpacityFallback::Standard, (ctx.mark_style.opacity, 1.0, 1.0));
    let stroke_width_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_width
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let row_stroke = |i: usize| -> ferrum_scene::StrokeStyle {
        let (opacity, _, _) = opacity_res.at_row(i);
        let width = stroke_width_values.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .unwrap_or(default_stroke_width);
        to_scene_stroke(stroke_color, width, opacity, None, None, None)
    };

    let meta = MetadataColumns::from_ctx(ctx);

    // All modes use the accumulator pattern: push each emitted node with its
    // source row index, finalize, then build metadata for kept indices only
    // (#6 defect class fix — metadata aligned to nodes, not all rows).
    let xf_opt = x_field(ctx, spec);

    // No x field: either ordinal-y-only or quantitative-y-rug.
    if xf_opt.is_none() {
        if let Some(yf) = y_field(ctx, spec) {
            // Ordinal y only → horizontal crossbars centered on each category band.
            if matches!(&ctx.scales.y, ScaleKind::Ordinal(_)) {
                let ys = match col_as_positional_category_str(ctx.batch, yf) {
                    Ok(v) => v,
                    Err(_) => return MarkBuildResult {
                        kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
                        tooltips: None, hrefs: None, descriptions: None,
                    },
                };
                let n_cats = {
                    let mut set = std::collections::HashSet::<&str>::new();
                    for v in ys.iter().flatten() { set.insert(v.as_str()); }
                    set.len().max(1)
                };
                let tick_half = (panel.w / n_cats as f64) * ctx.mark_style.band_size.unwrap_or(0.3);
                let baseline_x = panel.x;
                let mut acc = MarkNodes::with_capacity(ys.len());
                for i in 0..ys.len() {
                    let yv = match &ys[i] { Some(s) => s.as_str(), None => continue };
                    let cy = match ctx.scales.y.to_pixel_str(yv) { Some(p) => p, None => continue };
                    let cy = cy + y_offsets[i];
                    let bx = baseline_x + x_offsets[i];
                    acc.push(SceneNode::Line {
                        x1: bx,
                        y1: cy,
                        x2: bx + 2.0 * tick_half,
                        y2: cy,
                        style: row_stroke(i),
                    }, i);
                }
                let (nodes, data_indices) = acc.finalize();
                let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
                return MarkBuildResult {
                    kind: MarkBatchKind::Tick, nodes, data_indices: Some(data_indices),
                    tooltips, hrefs, descriptions,
                };
            }

            // Quantitative y only → y-rug: horizontal ticks at left axis edge.
            let tick_len = ctx.theme.sizes.tick_size * 2.0;
            let ys = match col_as_f64(ctx.batch, yf) {
                Ok(v) => v,
                Err(_) => return MarkBuildResult {
                    kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
                    tooltips: None, hrefs: None, descriptions: None,
                },
            };
            let baseline_x = panel.x;
            let mut acc = MarkNodes::with_capacity(ys.len());
            for i in 0..ys.len() {
                let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
                let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
                let py = py + y_offsets[i];
                let bx = baseline_x + x_offsets[i];
                acc.push(SceneNode::Line {
                    x1: bx, y1: py,
                    x2: bx + tick_len, y2: py,
                    style: row_stroke(i),
                }, i);
            }
            let (nodes, data_indices) = acc.finalize();
            let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
            return MarkBuildResult {
                kind: MarkBatchKind::Tick, nodes, data_indices: Some(data_indices),
                tooltips, hrefs, descriptions,
            };
        }
        // No x and no y: return empty.
        return MarkBuildResult {
            kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
            tooltips: None, hrefs: None, descriptions: None,
        };
    }

    let xf = xf_opt.expect("invariant: xf_opt is Some — None case returned above");

    // Ordinal x + quantitative y → horizontal tick at data y position.
    if matches!(&ctx.scales.x, ScaleKind::Ordinal(_)) {
        if let Some(yf) = y_field(ctx, spec) {
            let xs = match col_as_positional_category_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return MarkBuildResult {
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
            let mut acc = MarkNodes::with_capacity(xs.len());
            for i in 0..xs.len() {
                let xv = match &xs[i] { Some(s) => s.as_str(), None => continue };
                let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
                let cx = match ctx.scales.x.to_pixel_str(xv) { Some(p) => p, None => continue };
                let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
                let cx = cx + x_offsets[i];
                let py = py + y_offsets[i];
                acc.push(SceneNode::Line {
                    x1: cx - tick_half,
                    y1: py,
                    x2: cx + tick_half,
                    y2: py,
                    style: row_stroke(i),
                }, i);
            }
            let (nodes, data_indices) = acc.finalize();
            let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
            return MarkBuildResult {
                kind: MarkBatchKind::Tick,
                nodes,
                data_indices: Some(data_indices),
                tooltips,
                hrefs,
                descriptions,
            };
        }
    }

    // Ordinal y + quantitative x → vertical tick at data x position (strip plot).
    if matches!(&ctx.scales.y, ScaleKind::Ordinal(_)) {
        if let Some(yf) = y_field(ctx, spec) {
            let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return MarkBuildResult {
                kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
                tooltips: None, hrefs: None, descriptions: None,
            }};
            let ys = match col_as_positional_category_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return MarkBuildResult {
                kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
                tooltips: None, hrefs: None, descriptions: None,
            }};
            let n_cats = {
                let mut set = std::collections::HashSet::<&str>::new();
                for v in ys.iter().flatten() { set.insert(v.as_str()); }
                set.len().max(1)
            };
            let tick_half = (panel.h / n_cats as f64) * ctx.mark_style.band_size.unwrap_or(0.3);
            let mut acc = MarkNodes::with_capacity(xs.len());
            for i in 0..xs.len() {
                let xv = match xs[i] { Some(v) if v.is_finite() => v, _ => continue };
                let yv = match &ys[i] { Some(s) => s.as_str(), None => continue };
                let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
                let cy = match ctx.scales.y.to_pixel_str(yv) { Some(p) => p, None => continue };
                let px = px + x_offsets[i];
                let cy = cy + y_offsets[i];
                acc.push(SceneNode::Line {
                    x1: px,
                    y1: cy - tick_half,
                    x2: px,
                    y2: cy + tick_half,
                    style: row_stroke(i),
                }, i);
            }
            let (nodes, data_indices) = acc.finalize();
            let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
            return MarkBuildResult {
                kind: MarkBatchKind::Tick,
                nodes,
                data_indices: Some(data_indices),
                tooltips,
                hrefs,
                descriptions,
            };
        }
    }

    // Ordinal x only (no y field) → vertical crossbars at each category, anchored at panel bottom.
    if matches!(&ctx.scales.x, ScaleKind::Ordinal(_)) && y_field(ctx, spec).is_none() {
        let xs = match col_as_positional_category_str(ctx.batch, xf) {
            Ok(v) => v,
            Err(_) => return MarkBuildResult {
                kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
                tooltips: None, hrefs: None, descriptions: None,
            },
        };
        let n_cats = {
            let mut set = std::collections::HashSet::<&str>::new();
            for v in xs.iter().flatten() { set.insert(v.as_str()); }
            set.len().max(1)
        };
        let tick_half = (panel.h / n_cats as f64) * ctx.mark_style.band_size.unwrap_or(0.3);
        let baseline_y = panel.y + panel.h;
        let mut acc = MarkNodes::with_capacity(xs.len());
        for i in 0..xs.len() {
            let xv = match &xs[i] { Some(s) => s.as_str(), None => continue };
            let cx = match ctx.scales.x.to_pixel_str(xv) { Some(p) => p, None => continue };
            let cx = cx + x_offsets[i];
            let by = baseline_y + y_offsets[i];
            acc.push(SceneNode::Line {
                x1: cx,
                y1: by,
                x2: cx,
                y2: by - 2.0 * tick_half,
                style: row_stroke(i),
            }, i);
        }
        let (nodes, data_indices) = acc.finalize();
        let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
        return MarkBuildResult {
            kind: MarkBatchKind::Tick, nodes, data_indices: Some(data_indices),
            tooltips, hrefs, descriptions,
        };
    }

    // Quantitative x → rug-style vertical tick at panel baseline.
    let tick_len = ctx.theme.sizes.tick_size * 2.0;
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return MarkBuildResult {
        kind: MarkBatchKind::Tick, nodes: vec![], data_indices: Some(vec![]),
        tooltips: None, hrefs: None, descriptions: None,
    }};
    let baseline_y = panel.y + panel.h;
    let mut acc = MarkNodes::with_capacity(xs.len());
    for (i, xopt) in xs.iter().enumerate() {
        let xv = match xopt { Some(v) if v.is_finite() => *v, _ => continue };
        let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
        let px = px + x_offsets[i];
        let by = baseline_y + y_offsets[i];
        acc.push(SceneNode::Line {
            x1: px,
            y1: by,
            x2: px,
            y2: by - tick_len,
            style: row_stroke(i),
        }, i);
    }
    let _ = y_field(ctx, spec);

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Tick,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
    }
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
        params: Vec::new(),
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
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
        params: Vec::new(),
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
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
        params: Vec::new(),
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
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

    #[test]
    fn tick_ordinal_y_only_emits_horizontal_lines() {
        // Regression: ordinal-y-only (no x) previously returned empty nodes because
        // col_as_f64 failed on the string column. Now it should emit one horizontal
        // crossbar per row centered on the ordinal band.
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: None,
                y: Some(EncodingSpec {
                    field: "cat".into(),
                    type_: Some(crate::spec::encoding::DataType::Ordinal),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("cat", arrow::datatypes::DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let line_count = result.nodes.iter()
            .filter(|n| matches!(n, ferrum_scene::SceneNode::Line { .. }))
            .count();
        assert_eq!(line_count, 3, "expected 3 horizontal tick lines for 3 ordinal-y categories, got {line_count}");

        // All lines must be horizontal: y1 == y2.
        for node in &result.nodes {
            if let ferrum_scene::SceneNode::Line { y1, y2, .. } = node {
                assert!(
                    (y1 - y2).abs() < f64::EPSILON,
                    "ordinal-y-only ticks must be horizontal (y1={y1}, y2={y2})"
                );
            }
        }

        // Lines must have non-zero width (x1 != x2).
        let has_width = result.nodes.iter().any(|n| {
            if let ferrum_scene::SceneNode::Line { x1, x2, .. } = n {
                (x2 - x1).abs() > f64::EPSILON
            } else {
                false
            }
        });
        assert!(has_width, "ordinal-y-only tick lines must have non-zero horizontal extent");
    }

    #[test]
    fn tick_ordinal_x_only_emits_vertical_lines() {
        // Ordinal-x-only (no y field) should emit one vertical crossbar per row
        // anchored at the panel bottom, centered on the ordinal band.
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "cat".into(),
                    type_: Some(crate::spec::encoding::DataType::Ordinal),
                    ..Default::default()
                }),
                y: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("cat", arrow::datatypes::DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["x", "y", "z"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let line_count = result.nodes.iter()
            .filter(|n| matches!(n, ferrum_scene::SceneNode::Line { .. }))
            .count();
        assert_eq!(line_count, 3, "expected 3 vertical tick lines for 3 ordinal-x categories, got {line_count}");

        // All lines must be vertical: x1 == x2.
        for node in &result.nodes {
            if let ferrum_scene::SceneNode::Line { x1, x2, .. } = node {
                assert!(
                    (x1 - x2).abs() < f64::EPSILON,
                    "ordinal-x-only ticks must be vertical (x1={x1}, x2={x2})"
                );
            }
        }

        // Lines must have non-zero height (y1 != y2).
        let has_height = result.nodes.iter().any(|n| {
            if let ferrum_scene::SceneNode::Line { y1, y2, .. } = n {
                (y2 - y1).abs() > f64::EPSILON
            } else {
                false
            }
        });
        assert!(has_height, "ordinal-x-only tick lines must have non-zero vertical extent");
    }

    // ── Metadata-alignment regression tests (#6 defect class) ────────────────
    //
    // Tick has six modes; all now use MarkNodes. Tests cover:
    //   - quantitative-x rug (primary skip path tested below)
    //   - ordinal-x + quantitative-y (second mode, href channel)
    //
    // Fail-before: `build_metadata(ctx)` produced full per-row vectors before any
    // loop. When row 1 was skipped, node 1 received row 1's metadata (the bug).
    //
    // Pass-after: `MarkNodes` + `build_metadata_for_indices` aligns to kept rows.

    fn make_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        }
    }

    /// Regression: quantitative-x rug tick with a non-finite x skips that row.
    /// The tooltip on each surviving node must point to its true source row.
    ///
    /// Batch: 3 rows, x=[10.0, NaN, 90.0], tooltip=["tip_a","tip_b","tip_c"].
    /// Row 1 (NaN x) is skipped → 2 nodes. Node 1 must have "tip_c", not "tip_b".
    #[test]
    fn tick_quant_x_skipped_nan_tooltip_aligned() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",   DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        // Row 1 has NaN x → skipped by `v.is_finite()` guard.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, f64::NAN, 90.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2,
            "expected 2 tick nodes after NaN-x skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be 'tip_a' (row 0); got '{t0}'");

        // Node 1 → row 2 → "tip_c". Old code (full-row): "tip_b" (the bug).
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be 'tip_c' (row 2), not 'tip_b' (row 1); got '{t1}'. \
             This fails on pre-migration code using build_metadata(ctx).");
    }

    /// Href-channel alignment on the ordinal-x + quantitative-y mode.
    /// Row 1 has a null y-value → skipped. Href on node 1 must be "url_c" (row 2),
    /// not "url_b" (row 1, old bug).
    #[test]
    fn tick_ordinal_x_quant_y_skipped_null_y_href_aligned() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                href: Some(EncodingSpec { field: "url".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8,    false),
            Field::new("val", DataType::Float64, true),   // nullable — row 1 null → skip
            Field::new("url", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![Some(10.0_f64), None, Some(80.0)])),
            Arc::new(StringArray::from(vec!["url_a", "url_b", "url_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2,
            "expected 2 tick nodes after null-y skip; got {}", result.nodes.len());

        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), 2, "href count must equal node count");
        assert_eq!(hrefs[0].as_deref(), Some("url_a"), "node 0 href must be 'url_a'");
        assert_eq!(hrefs[1].as_deref(), Some("url_c"),
            "node 1 href must be 'url_c' (row 2), not 'url_b' (row 1); \
             old build_metadata would give 'url_b'");
    }

    /// No-skip backward-compat (quantitative-x rug mode): all rows are finite →
    /// all nodes produced, tooltips in original row order.
    #[test]
    fn tick_no_skip_tooltips_unchanged() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",   DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 3, "all 3 rows must produce tick nodes");
        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 3, "tooltip count must equal node count");
        let values: Vec<&str> = tooltips.iter().map(|t| t.fields[0].value.as_str()).collect();
        assert_eq!(values, vec!["tip_a", "tip_b", "tip_c"],
            "no-skip: tooltips must be in original row order");
    }

    /// C7 regression guard: after migrating tick's per-row opacity to the shared
    /// `OpacityResolver`, each tick line must still carry its own row's opacity
    /// value (per-row sampling, not a single constant). Fails if a dedup ever
    /// collapses the per-row sample to one value or drops the encoding column.
    #[test]
    fn tick_per_row_opacity_is_sampled_per_row() {
        use crate::spec::encoding::DataType as SDT;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                opacity: Some(EncodingSpec { field: "op".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",  DataType::Float64, false),
            Field::new("op", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(Float64Array::from(vec![0.2_f64, 0.5, 0.9])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let opacities: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Line { style, .. } = n { Some(style.opacity) } else { None }
        }).collect();
        assert_eq!(opacities, vec![0.2, 0.5, 0.9],
            "each tick must carry its own row's opacity (per-row OpacityResolver sample)");
    }

    /// C7 family-consistency guard: tick now clamps out-of-range opacity to
    /// `[0, 1]` and falls non-finite values back to the default, matching every
    /// other mark via the shared `OpacityResolver`. (Before C7 tick passed these
    /// raw, which could emit SVG with an invalid `opacity` attribute.) No real
    /// chart feeds out-of-range opacity, so goldens are unaffected.
    #[test]
    fn tick_opacity_is_clamped_and_finite_checked() {
        use crate::spec::encoding::DataType as SDT;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                opacity: Some(EncodingSpec { field: "op".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",  DataType::Float64, false),
            Field::new("op", DataType::Float64, false),
        ]));
        // Row 0: > 1 → clamps to 1.0. Row 1: NaN → falls back to default opacity.
        // Row 2: in-range → passes through.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(Float64Array::from(vec![1.5_f64, f64::NAN, 0.3])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let default_opacity = mark_style.opacity;
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let opacities: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Line { style, .. } = n { Some(style.opacity) } else { None }
        }).collect();
        assert_eq!(opacities, vec![1.0, default_opacity, 0.3],
            "tick opacity must clamp >1 to 1.0, fall NaN to default, pass in-range through");
    }
}
