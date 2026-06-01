//! Segment mark — diagonal line from (x, y) to (x2, y2).
//! Distinct from rule (axis-aligned only): segments may go in any direction.

use crate::render::draw::{
    col_as_f64, col_as_str, color_field, resolve_stroke_color, x_field, y_field, DrawCtx,
};

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{to_scene_stroke, MarkBuildResult, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult {
        kind: MarkBatchKind::Segment,
        nodes: vec![],
        data_indices: Some(vec![]),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    };

    let spec = ctx.spec;
    let (Some(xf), Some(yf)) = (x_field(ctx, spec), y_field(ctx, spec)) else { return empty(); };
    let Some(x2f) = spec.encoding.x2.as_ref().map(|e| e.field.as_str()) else { return empty(); };
    let Some(y2f) = spec.encoding.y2.as_ref().map(|e| e.field.as_str()) else { return empty(); };

    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty() };
    let x2s = match col_as_f64(ctx.batch, x2f) { Ok(v) => v, Err(_) => return empty() };
    let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return empty() };

    // Per-row opacity and stroke_width from encoding columns (if mapped).
    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let stroke_width_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_width
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    // Per-row stroke color from the color encoding + color scale (same path as
    // line.rs/point.rs). Only wins when no explicit user stroke override is set.
    let color_values: Option<Vec<Option<crate::render::color::Color>>> =
        match (color_field(ctx, spec), ctx.scales.color.as_ref()) {
            (Some(field), Some(scale)) => col_as_str(ctx.batch, field).ok().map(|cats| {
                cats.iter()
                    .map(|c| c.as_deref().and_then(|v| scale.lookup(v)))
                    .collect()
            }),
            _ => None,
        };

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

    let n = xs.len().min(ys.len()).min(x2s.len()).min(y2s.len());
    for i in 0..n {
        let (xv, yv, x2v, y2v) = match (xs[i], ys[i], x2s[i], y2s[i]) {
            (Some(a), Some(b), Some(c), Some(d))
                if a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() =>
                (a, b, c, d),
            _ => continue,
        };
        let p1x = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
        let p1y = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
        let p2x = match ctx.scales.x.to_pixel_f64(x2v) { Some(p) => p, None => continue };
        let p2y = match ctx.scales.y.to_pixel_f64(y2v) { Some(p) => p, None => continue };
        let xo = x_offsets.get(i).copied().unwrap_or(0.0);
        let yo = y_offsets.get(i).copied().unwrap_or(0.0);

        let row_opacity = opacity_values.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .unwrap_or(ctx.mark_style.opacity);
        let row_stroke_width = stroke_width_values.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .unwrap_or(ctx.mark_style.stroke_width);

        // Precedence (explicit constant stroke > per-row color > theme > fill)
        // lives in `resolve_stroke_color`.
        let row_color_opt = color_values
            .as_ref()
            .and_then(|v| v.get(i).copied().flatten());
        let row_color = resolve_stroke_color(ctx.mark_style, row_color_opt);
        let row_style = to_scene_stroke(
            row_color,
            row_stroke_width,
            row_opacity,
            ctx.mark_style.stroke_dash.as_deref(),
            None,
            None,
        );

        nodes.push(SceneNode::Line {
            x1: p1x + xo,
            y1: p1y + yo,
            x2: p2x + xo,
            y2: p2y + yo,
            style: row_style,
        });
        indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Segment,
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
    fn segment_renders_diagonal_lines() {
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Segment,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment);
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Line { .. })).count(), 2);
    }

    #[test]
    fn segment_uses_explicit_stroke_color() {
        use crate::spec::mark_style::MarkKwargsSpec;

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Segment,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![1.0])),
            Arc::new(Float64Array::from(vec![1.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();

        // Create mark style with explicit stroke color
        let overrides = MarkKwargsSpec { stroke: Some("#e4572e".into()), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Segment);

        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = super::build(&ctx);

        // Check that the segment line has the correct stroke color
        assert_eq!(result.nodes.len(), 1);
        if let ferrum_scene::SceneNode::Line { style, .. } = &result.nodes[0] {
            assert_eq!(style.color.r, 0xe4, "stroke red component should match explicit color");
            assert_eq!(style.color.g, 0x57, "stroke green component should match explicit color");
            assert_eq!(style.color.b, 0x2e, "stroke blue component should match explicit color");
        } else {
            panic!("Expected Line node");
        }
    }

    #[test]
    fn segment_resolves_per_row_color_encoding() {
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Segment,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "dir".into(), type_: Some(crate::spec::encoding::DataType::Nominal), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None,
            mark_style: None, position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("dir", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(StringArray::from(vec!["up", "down"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let colors: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            ferrum_scene::SceneNode::Line { style, .. } => Some(style.color),
            _ => None,
        }).collect();
        assert_eq!(colors.len(), 2);
        assert_ne!(
            (colors[0].r, colors[0].g, colors[0].b),
            (colors[1].r, colors[1].g, colors[1].b),
            "segment color encoding must yield distinct per-row stroke colors"
        );
    }
}
