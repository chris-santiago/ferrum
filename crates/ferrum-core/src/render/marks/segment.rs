//! Segment mark — diagonal line from (x, y) to (x2, y2).
//! Distinct from rule (axis-aligned only): segments may go in any direction.

use crate::render::draw::{
    col_as_f64, col_as_str, color_field, resolve_stroke_color, x_field, y_field, DrawCtx,
};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::opacity::{OpacityFallback, OpacityResolver};

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{to_scene_stroke, MarkBuildResult, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult::empty(MarkBatchKind::Segment);

    let spec = ctx.spec;
    let (Some(xf), Some(yf)) = (x_field(ctx, spec), y_field(ctx, spec)) else { return empty(); };
    let Some(x2f) = spec.encoding.x2.as_ref().map(|e| e.field.as_str()) else { return empty(); };
    let Some(y2f) = spec.encoding.y2.as_ref().map(|e| e.field.as_str()) else { return empty(); };

    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty() };
    let x2s = match col_as_f64(ctx.batch, x2f) { Ok(v) => v, Err(_) => return empty() };
    let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return empty() };

    // Per-row opacity via the shared OpacityResolver (C7); stroke_width stays
    // local. Segment is a stroke-only mark (no fill / stroke_opacity columns),
    // so only the resolver's `opacity` slot is read.
    let opacity_res =
        OpacityResolver::load(ctx, OpacityFallback::Standard, (ctx.mark_style.opacity, 1.0, 1.0));

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

    // Accumulate nodes and source-row indices in lockstep so metadata is
    // aligned to kept nodes only (#6 defect class fix).
    let n = xs.len().min(ys.len()).min(x2s.len()).min(y2s.len());
    let mut acc = MarkNodes::with_capacity(n);

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

        let (row_opacity, _, _) = opacity_res.at_row(i);
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

        acc.push(SceneNode::Line {
            x1: p1x + xo,
            y1: p1y + yo,
            x2: p2x + xo,
            y2: p2y + yo,
            style: row_style,
        }, i);
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Segment,
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
            params: Vec::new(),
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
            params: Vec::new(),
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
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None, params: Vec::new(),
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

    // ── Metadata-alignment regression tests (#6 defect class) ────────────────
    //
    // Each test creates a batch where a middle row is skipped (null / non-finite)
    // and asserts that tooltip or href metadata on each emitted node points to its
    // TRUE source row, not the node-position row.
    //
    // Fail-before: `meta.build_metadata(ctx)` produced full per-row vectors before
    // the loop. When row 1 was skipped, node 1 received row 1's metadata (the
    // bug). These tests would have failed: node 1's tooltip would be "tip_b"
    // (row 1, skipped) instead of "tip_c" (row 2, the true source row of node 1).
    //
    // Pass-after: `MarkNodes` + `build_metadata_for_indices` aligns metadata to
    // kept nodes only, so node 1 correctly receives row 2's metadata.

    fn make_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        }
    }

    /// Regression: `build_segment` with a null y-value skips that row. The
    /// tooltip on each surviving node must point to its true source row.
    ///
    /// Batch: 3 rows, y=[10.0, null, 30.0], tooltip=["tip_a","tip_b","tip_c"].
    /// Row 1 (null y) is skipped → 2 nodes. Node 0 → row 0 → "tip_a"; node 1
    /// → row 2 → "tip_c". Old code: node 1 → "tip_b".
    #[test]
    fn segment_skipped_null_y_tooltip_aligned() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Segment,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "x".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                y:  Some(EncodingSpec { field: "y".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
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
            Field::new("y",   DataType::Float64, true),   // nullable — row 1 null → skip
            Field::new("x2",  DataType::Float64, false),
            Field::new("y2",  DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        // Row 1 has null y → skipped by the (xv,yv,x2v,y2v) destructure guard.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0_f64,  1.0, 2.0])),
            Arc::new(Float64Array::from(vec![Some(0.0_f64), None, Some(20.0)])),
            Arc::new(Float64Array::from(vec![10.0_f64, 11.0, 12.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 11.0, 20.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // 2 nodes survive (row 1 with null y is skipped).
        assert_eq!(result.nodes.len(), 2,
            "expected 2 segments after null-y skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        // Node 0 → source row 0 → "tip_a".
        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a",
            "node 0 tooltip must be 'tip_a' (row 0); got '{t0}'");

        // Node 1 → source row 2 → "tip_c". Old code gave "tip_b" (row 1, the bug).
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be 'tip_c' (row 2), not 'tip_b' (row 1); got '{t1}'. \
             This fails on pre-migration code using build_metadata(ctx).");
    }

    /// Href-channel alignment: same setup but uses href encoding instead of
    /// tooltip. The href channel must also be aligned to kept rows only.
    ///
    /// Batch: 3 rows, y=[10.0, null, 30.0], href=["url_a","url_b","url_c"].
    /// Row 1 skipped → 2 nodes. Node 1 must get "url_c", not "url_b".
    #[test]
    fn segment_skipped_null_y_href_aligned() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Segment,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "x".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                y:  Some(EncodingSpec { field: "y".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
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
            Field::new("x",   DataType::Float64, false),
            Field::new("y",   DataType::Float64, true),
            Field::new("x2",  DataType::Float64, false),
            Field::new("y2",  DataType::Float64, false),
            Field::new("url", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0_f64,  1.0, 2.0])),
            Arc::new(Float64Array::from(vec![Some(0.0_f64), None, Some(20.0)])),
            Arc::new(Float64Array::from(vec![10.0_f64, 11.0, 12.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 11.0, 20.0])),
            Arc::new(StringArray::from(vec!["url_a", "url_b", "url_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2, "expected 2 segments");
        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), 2, "href count must equal node count");

        assert_eq!(hrefs[0].as_deref(), Some("url_a"), "node 0 href must be 'url_a'");
        assert_eq!(hrefs[1].as_deref(), Some("url_c"),
            "node 1 href must be 'url_c' (row 2), not 'url_b' (row 1); \
             old code using build_metadata would give 'url_b'");
    }

    /// No-skip backward-compat: when no rows are skipped the output is unchanged
    /// from the full-row path (build_metadata_for_indices with a full range ≡
    /// build_metadata for all-kept batches).
    #[test]
    fn segment_no_skip_tooltips_unchanged() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Segment,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "x".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                y:  Some(EncodingSpec { field: "y".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
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
            Field::new("y",   DataType::Float64, false),
            Field::new("x2",  DataType::Float64, false),
            Field::new("y2",  DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        // All rows finite → no skipping.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0_f64, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0_f64, 5.0, 10.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 11.0, 12.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 15.0, 20.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 3, "all 3 rows must produce nodes when none are skipped");
        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 3, "tooltip count must equal node count");
        let values: Vec<&str> = tooltips.iter().map(|t| t.fields[0].value.as_str()).collect();
        assert_eq!(values, vec!["tip_a", "tip_b", "tip_c"],
            "no-skip: tooltips must be in original row order");
    }

    /// C7 regression guard: after migrating segment's per-row opacity to the
    /// shared `OpacityResolver`, each segment must still carry its own row's
    /// opacity. In-range values are byte-stable (no clamp effect); a regression
    /// that collapses or drops the per-row sample fails here.
    #[test]
    fn segment_per_row_opacity_is_sampled_per_row() {
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Segment,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
                opacity: Some(EncodingSpec { field: "op".into(), type_: None, ..Default::default() }),
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
            Field::new("y",  DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("op", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0_f64, 5.0])),
            Arc::new(Float64Array::from(vec![0.0_f64, 5.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 15.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 15.0])),
            Arc::new(Float64Array::from(vec![0.25_f64, 0.75])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let opacities: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Line { style, .. } = n { Some(style.opacity) } else { None }
        }).collect();
        assert_eq!(opacities, vec![0.25, 0.75],
            "each segment must carry its own row's opacity (per-row OpacityResolver sample)");
    }
}
