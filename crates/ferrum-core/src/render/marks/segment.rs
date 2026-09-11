//! Segment mark — diagonal line from (x, y) to (x2, y2).
//! Distinct from rule (axis-aligned only): segments may go in any direction.

use crate::render::draw::{
    arrowhead_path_with_opacity, arrowhead_triangle, col_as_f64, resolve_stroke_color, x_field, y_field,
    DrawCtx,
};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::channels::row_colors_from_scale;
use crate::render::marks::opacity::{OpacityFallback, OpacityResolver};
use crate::render::RenderError;

/// Build one `Line` scene node per row from `(x, y)` to `(x2, y2)`.
///
/// Fallible for one reason: the per-row color read propagates the typed
/// `RenderError` its reader constructs instead of discarding it, matching
/// `rule.rs` — the sibling that shares this mark's diagonal geometry and the
/// same shared color reader (`channels::row_colors_from_scale`). Every other
/// refusal in this builder is a presence/dtype guard that predates that rule
/// and stays an `Ok(empty())`; #118's silent-empty policy sweep owns those,
/// not this change.
pub fn build(ctx: &DrawCtx) -> Result<crate::render::draw::MarkBuildResult, RenderError> {
    use crate::render::draw::{to_scene_stroke, MarkBuildResult, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult::empty(MarkBatchKind::Segment);

    let spec = ctx.spec;
    let (Some(xf), Some(yf)) = (x_field(ctx, spec), y_field(ctx, spec)) else { return Ok(empty()); };
    let Some(x2f) = spec.encoding.x2.as_ref().map(|e| e.field.as_str()) else { return Ok(empty()); };
    let Some(y2f) = spec.encoding.y2.as_ref().map(|e| e.field.as_str()) else { return Ok(empty()); };

    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return Ok(empty()) };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return Ok(empty()) };
    let x2s = match col_as_f64(ctx.batch, x2f) { Ok(v) => v, Err(_) => return Ok(empty()) };
    let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return Ok(empty()) };

    // Per-row `opacity`, `fill_opacity`, and `stroke_opacity` via the shared
    // OpacityResolver (C7); stroke_width stays local. The shaft `Line` only
    // ever consumes `opacity`/`stroke_opacity` (`StrokeStyle` has no fill
    // component `to_scene_stroke` could apply `fill_opacity` to) — that part
    // mirrors `tick.rs`'s `row_stroke`, whose module doc carries the full
    // enumeration against `rule.rs`. `fill_opacity` WAS genuinely unused here
    // (discarded via `_`) before batch H Task 1: `arrow=True`'s arrowhead is
    // segment's first FILLED node, so it reads this same per-row resolver's
    // `fill_opacity` slot too, rather than a second independently-resolved
    // value.
    let opacity_res =
        OpacityResolver::load(ctx, OpacityFallback::Standard, (ctx.mark_style.paint.opacity, 1.0, 1.0));

    let stroke_width_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_width
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    // Per-row stroke color through the shared NF-A3 reader (same path as
    // rule.rs). Only wins when no explicit user stroke override is set.
    let color_values = row_colors_from_scale(ctx)?;

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let meta = MetadataColumns::from_ctx(ctx);

    // Batch H Task 1: `mark_segment(arrow=True)` draws a filled triangle
    // arrowhead at each row's (x2, y2) end, colored by that row's resolved
    // stroke. `arrow_size` is a constant mark kwarg (not per-row), mirroring
    // `annotate_arrow`'s `head_size`. `None`/unset `arrow` is `false` —
    // byte-identical to every segment rendered before this field existed.
    let arrow_enabled = ctx.mark_style.segment.arrow.unwrap_or(false);
    let arrow_size = ctx.mark_style.segment.arrow_size.unwrap_or(8.0);

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

        let (row_opacity, row_fill_opacity, row_stroke_opacity) = opacity_res.at_row(i);
        let row_stroke_width = stroke_width_values.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .unwrap_or(ctx.mark_style.paint.stroke_width);

        // Precedence (explicit constant stroke > per-row color > theme > fill)
        // lives in `resolve_stroke_color`.
        let row_color_opt = color_values
            .as_ref()
            .and_then(|v| v.get(i).copied().flatten());
        let row_color = resolve_stroke_color(ctx.mark_style, row_color_opt);
        let mut row_style = to_scene_stroke(
            row_color,
            row_stroke_width,
            row_opacity,
            ctx.mark_style.paint.stroke_dash.as_deref(),
            None,
            None,
        );
        // Multiplied in (not overwritten), the same mechanism `tick.rs`'s
        // `row_stroke` uses and for the same reason: with no `stroke_opacity`
        // encoding bound, `OpacityResolver` defaults this slot to `1.0`, and
        // `x * 1.0 == x` for every finite `x` — so every segment this crate
        // already renders stays bit-identical, and only a chart that newly
        // binds a DISTINCT `stroke_opacity=` observes any change.
        row_style.stroke_opacity *= row_stroke_opacity;

        let (line_x1, line_y1, line_x2, line_y2) = (p1x + xo, p1y + yo, p2x + xo, p2y + yo);
        let line_color = row_style.color;
        // Captured before `row_style` moves into the `Line` node below, so
        // the arrowhead can inherit the SAME resolved opacity/stroke_opacity
        // the shaft carries — not the annotation path's fixed 1.0 (batch H
        // Task 1 quality-review fix). `line_opacity` is `row_style.opacity`
        // itself (set verbatim by `to_scene_stroke`); `line_stroke_opacity`
        // is the post-multiply value (`row_style.stroke_opacity`), not the
        // raw `row_stroke_opacity`, so this reflects exactly what the shaft
        // renders.
        let line_opacity = row_style.opacity;
        let line_stroke_opacity = row_style.stroke_opacity;
        let line_node = SceneNode::Line { x1: line_x1, y1: line_y1, x2: line_x2, y2: line_y2, style: row_style };

        if arrow_enabled {
            // `arrowhead_triangle` returns `None` for a zero-length segment
            // (no direction to point in) — that row still renders its shaft,
            // just without a head, rather than a NaN-coordinate triangle.
            match arrowhead_triangle(line_x1, line_y1, line_x2, line_y2, arrow_size) {
                // ORDER IS LOAD-BEARING: the shaft `Line` must precede its
                // arrowhead `Path` in this array. `break_axis.rs`'s
                // `remap_segment_nodes` rebuilds an in-gap arrowhead from
                // "the most recently placed `Line`" when it walks a broken
                // `Segment` batch's nodes in order — it has no other way to
                // learn which shaft a head belongs to, since a batch holds
                // many rows' worth of `[Line, Path]` pairs back to back.
                // Reversing this pair (or interleaving a `Line` between an
                // unrelated row's shaft and head) makes that walk silently
                // pair an arrowhead with the WRONG row's shaft direction
                // under a break — wrong, not absent, so no gate catches it
                // without the test-side assertion below pinning this order.
                Some(points) => acc.push_many(
                    [
                        line_node,
                        arrowhead_path_with_opacity(
                            points, line_color, line_opacity, line_stroke_opacity, row_fill_opacity,
                        ),
                    ],
                    i,
                ),
                None => acc.push(line_node, i),
            }
        } else {
            acc.push(line_node, i);
        }
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    Ok(MarkBuildResult {
        polar_spans: None,
        kind: MarkBatchKind::Segment,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
    })
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment).unwrap();
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = super::build(&ctx).unwrap();
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
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Segment).unwrap();

        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = super::build(&ctx).unwrap();

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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        let opacities: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Line { style, .. } = n { Some(style.opacity) } else { None }
        }).collect();
        assert_eq!(opacities, vec![0.25, 0.75],
            "each segment must carry its own row's opacity (per-row OpacityResolver sample)");
    }

    // ── Per-row stroke_opacity (batch-E close remediation) ───────────────────

    /// Four rows carrying `x`/`y`/`x2`/`y2` plus one extra opacity-family
    /// column, so the two tests below differ only in WHICH channel they bind.
    fn opacity_channel_batch(field: &str, values: Vec<f64>) -> arrow::record_batch::RecordBatch {
        let n = values.len();
        let coord = |base: f64| -> Float64Array {
            (0..n).map(|i| Some(base + i as f64 * 5.0)).collect()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new(field, DataType::Float64, false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(coord(0.0)),
            Arc::new(coord(0.0)),
            Arc::new(coord(10.0)),
            Arc::new(coord(10.0)),
            Arc::new(Float64Array::from(values)),
        ]).unwrap()
    }

    fn segment_spec_with(channel: &str, field: &str) -> ChartSpec {
        let enc = |f: &str| Some(EncodingSpec { field: f.into(), type_: None, ..Default::default() });
        let mut encoding = Encoding {
            x: enc("x"), y: enc("y"), x2: enc("x2"), y2: enc("y2"),
            ..Default::default()
        };
        match channel {
            "opacity" => encoding.opacity = enc(field),
            "stroke_opacity" => encoding.stroke_opacity = enc(field),
            other => panic!("unhandled channel {other}"),
        }
        ChartSpec {
            data: DataRef::default(), mark: Mark::Segment, encoding,
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        }
    }

    fn segment_stroke_opacities(
        spec: &ChartSpec,
        batch: &arrow::record_batch::RecordBatch,
    ) -> (Vec<f64>, OpacityResolver) {
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(spec, batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment).unwrap();
        let ctx = DrawCtx { spec, panel: &panel, theme: &theme, scales: &scales, batch, mark_style: &mark_style };
        let resolver = OpacityResolver::load(
            &ctx, OpacityFallback::Standard, (ctx.mark_style.paint.opacity, 1.0, 1.0),
        );
        let rendered = super::build(&ctx).unwrap().nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Line { style, .. } = n { Some(style.stroke_opacity) } else { None }
        }).collect();
        (rendered, resolver)
    }

    /// RED-proven: a `stroke_opacity`-encoded segment must render a distinct
    /// `StrokeStyle.stroke_opacity` per row. Before this fix `segment.rs`
    /// discarded the resolver's third tuple slot (`let (row_opacity, _, _) =
    /// …`) behind a copied-and-false comment claiming segment has no
    /// `stroke_opacity` column — but the channel is generic and
    /// `RENDERER_HONORED` (`ferrum.encoding._channel_policy`), resolving its
    /// own `OpacityScale` regardless of mark, so a bound `stroke_opacity=`
    /// silently had no effect. `tick.rs` closed the same drop one cycle
    /// earlier and documented segment's as still open; this closes it.
    #[test]
    fn segment_stroke_opacity_encoding_is_applied_per_row() {
        let spec = segment_spec_with("stroke_opacity", "so");
        let batch = opacity_channel_batch("so", vec![0.2, 0.5, 0.9]);
        let (rendered, resolver) = segment_stroke_opacities(&spec, &batch);

        // Expected values come from the SAME shared resolver `build` consults,
        // since `stroke_opacity` maps through a theme-band `OpacityScale`
        // rather than passing the raw column through.
        let expected: Vec<f64> = (0..3).map(|i| resolver.at_row(i).2).collect();
        assert!(
            expected.windows(2).all(|w| w[0] < w[1]),
            "test sanity: resolved stroke_opacity must be strictly increasing across rows, got {expected:?}"
        );
        assert_eq!(
            rendered, expected,
            "each segment must carry its own row's resolved stroke_opacity, not a constant"
        );
    }

    /// Byte-identity guard: with no `stroke_opacity` encoding bound, the
    /// resolver defaults that slot to `1.0` and `x * 1.0 == x` bit-for-bit, so
    /// `StrokeStyle.stroke_opacity` stays exactly what `to_scene_stroke` set it
    /// to (it seeds the field from the `opacity` argument) — every segment this
    /// crate already renders is unchanged, `opacity=`-encoded or not.
    #[test]
    fn segment_no_stroke_opacity_encoding_is_byte_identical() {
        let spec = segment_spec_with("opacity", "op");
        let batch = opacity_channel_batch("op", vec![0.2, 0.5, 0.9]);
        let (rendered, _) = segment_stroke_opacities(&spec, &batch);

        assert_eq!(
            rendered,
            vec![0.2, 0.5, 0.9],
            "with no stroke_opacity encoding, StrokeStyle.stroke_opacity must equal the \
             resolved opacity value unchanged (the pre-fix behavior, bit-for-bit)"
        );
    }

    // ── Batch H Task 1: mark_segment(arrow=) ─────────────────────────────────

    /// Single-row x/y/x2/y2 segment spec + batch + resolved `DrawCtx`, with the
    /// given `arrow`/`arrow_size` mark-kwargs override. Domain/range are both
    /// `(0.0, 100.0)`, so `resolve_scales` need not be re-derived by hand in
    /// each test — callers instead ask `ctx.scales` for the same pixel
    /// endpoints the production path used, keeping the geometry-equality test
    /// robust to whatever scale-padding `resolve_scales` applies.
    fn arrow_test_build(
        x2: f64,
        y2: f64,
        arrow: Option<bool>,
        arrow_size: Option<f64>,
    ) -> (crate::render::draw::MarkBuildResult, crate::render::scale_resolve::ResolvedScales) {
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
            transforms: Vec::new(), facet: None, layers: None, coord: None,
            mark_style: None, position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0_f64])),
            Arc::new(Float64Array::from(vec![0.0_f64])),
            Arc::new(Float64Array::from(vec![x2])),
            Arc::new(Float64Array::from(vec![y2])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let overrides = MarkKwargsSpec { arrow, arrow_size, ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Segment).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        (result, scales)
    }

    /// `arrow=true` on a non-degenerate segment emits exactly one arrowhead
    /// `Path` alongside the shaft `Line`. Fails to compile pre-change:
    /// `MarkKwargsSpec::arrow` did not exist before this task.
    #[test]
    fn segment_arrow_true_emits_arrowhead_path() {
        let (result, _) = arrow_test_build(10.0, 5.0, Some(true), Some(6.0));
        let lines = result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Line { .. })).count();
        let paths = result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Path { .. })).count();
        assert_eq!(lines, 1, "shaft line must still be emitted");
        assert_eq!(paths, 1, "arrow=true must emit exactly one arrowhead path");
    }

    /// The shaft/head node order the arrowhead comment above `push_many`
    /// calls out as load-bearing: for every arrowed row, `build` must emit
    /// `[Line, Path]` ADJACENT, in that order, not just "one Line and one
    /// Path somewhere in the batch." `break_axis.rs`'s `remap_segment_nodes`
    /// pairs a head with "the most recently placed `Line`" while walking the
    /// batch in order; if a future change reordered or interleaved these
    /// pairs, that walk would silently attach an arrowhead to the WRONG
    /// row's shaft under a break, and neither this test's sibling assertions
    /// (which only count node kinds) nor the render would visibly complain
    /// on a chart with no break. Three rows, arrow enabled, none degenerate.
    #[test]
    fn segment_arrow_true_emits_line_then_path_adjacent_per_row() {
        use crate::spec::mark_style::MarkKwargsSpec;
        let enc = |f: &str| Some(EncodingSpec { field: f.into(), type_: None, ..Default::default() });
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Segment,
            encoding: Encoding { x: enc("x"), y: enc("y"), x2: enc("x2"), y2: enc("y2"), ..Default::default() },
            transforms: Vec::new(), facet: None, layers: None, coord: None,
            mark_style: None, position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0_f64, 5.0, 10.0])),
            Arc::new(Float64Array::from(vec![0.0_f64, 5.0, 10.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 25.0, 40.0])),
            Arc::new(Float64Array::from(vec![5.0_f64, 30.0, 45.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let overrides = MarkKwargsSpec { arrow: Some(true), arrow_size: Some(6.0), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Segment).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let result = super::build(&ctx).unwrap();
        assert_eq!(result.nodes.len(), 6, "3 rows x [Line, Path]");
        for row in 0..3 {
            assert!(
                matches!(result.nodes[row * 2], ferrum_scene::SceneNode::Line { .. }),
                "row {row}: expected the shaft Line at index {}",
                row * 2
            );
            assert!(
                matches!(result.nodes[row * 2 + 1], ferrum_scene::SceneNode::Path { .. }),
                "row {row}: expected the arrowhead Path immediately after its shaft at index {}",
                row * 2 + 1
            );
        }
    }

    /// `arrow` unset (`None`, the default) emits only the shaft — no
    /// arrowhead `Path` — the byte-stable no-op every segment rendered
    /// before this field existed.
    #[test]
    fn segment_arrow_unset_is_byte_identical_no_arrowhead() {
        let (result, _) = arrow_test_build(10.0, 5.0, None, None);
        assert_eq!(result.nodes.len(), 1, "no arrow override must emit only the shaft Line");
        assert!(matches!(result.nodes[0], ferrum_scene::SceneNode::Line { .. }));
    }

    /// `arrow=false` explicitly is identical to unset: only the shaft.
    #[test]
    fn segment_arrow_false_emits_no_arrowhead() {
        let (result, _) = arrow_test_build(10.0, 5.0, Some(false), None);
        assert_eq!(result.nodes.len(), 1);
        assert!(matches!(result.nodes[0], ferrum_scene::SceneNode::Line { .. }));
    }

    /// A zero-length segment (`x2 == x`, `y2 == y`) has no direction to point
    /// an arrowhead in. `arrow=true` on that row must still emit its shaft
    /// (a zero-length line, matching every other zero-length segment ferrum
    /// already renders) with no arrowhead — never a NaN-coordinate triangle
    /// or a panic.
    #[test]
    fn segment_arrow_true_zero_length_segment_emits_only_line() {
        let (result, _) = arrow_test_build(0.0, 0.0, Some(true), Some(8.0));
        assert_eq!(
            result.nodes.len(), 1,
            "zero-length segment with arrow=true must emit only its (zero-length) shaft"
        );
        assert!(matches!(result.nodes[0], ferrum_scene::SceneNode::Line { .. }));
        if let ferrum_scene::SceneNode::Line { x1, y1, x2, y2, .. } = result.nodes[0] {
            assert!(x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite(),
                "no NaN coordinates from the degenerate direction");
        }
    }

    /// `arrow=true` with no `arrow_size` override resolves to `8.0` — the
    /// same default `annotate_arrow`'s unlabelled overlay uses.
    #[test]
    fn segment_arrow_true_default_size_is_eight() {
        let (result, scales) = arrow_test_build(50.0, 0.0, Some(true), None);
        let path = result.nodes.iter().find_map(|n| match n {
            ferrum_scene::SceneNode::Path { commands, .. } => Some(commands.clone()),
            _ => None,
        }).expect("arrow=true must emit an arrowhead path");

        let p1x = scales.x.to_pixel_f64(0.0).unwrap();
        let p1y = scales.y.to_pixel_f64(0.0).unwrap();
        let p2x = scales.x.to_pixel_f64(50.0).unwrap();
        let p2y = scales.y.to_pixel_f64(0.0).unwrap();
        let expected = arrowhead_triangle(p1x, p1y, p2x, p2y, 8.0)
            .expect("non-degenerate segment must yield a triangle");
        let got: Vec<(f64, f64)> = path.iter().filter_map(|c| match c {
            ferrum_scene::PathCmd::MoveTo { x, y } | ferrum_scene::PathCmd::LineTo { x, y } => Some((*x, *y)),
            _ => None,
        }).collect();
        assert_eq!(got, vec![expected[0], expected[1], expected[2]]);
    }

    /// Geometry-equality proof (task requirement): `segment.rs`'s emitted
    /// arrowhead triangle for a row must equal EXACTLY what
    /// `draw::arrowhead_triangle` computes from that row's own resolved
    /// pixel endpoints — the same shared function `annotation::emit_arrow`
    /// calls (pinned by `emit_arrow_triangle_matches_shared_arrowhead_triangle_helper`
    /// in `annotation.rs`), so `mark_segment(arrow=True)` and
    /// `annotate_arrow`'s unlabelled overlay draw byte-identical arrowheads
    /// for the same endpoints/size, by construction rather than by
    /// coincidence.
    #[test]
    fn segment_arrow_triangle_matches_shared_arrowhead_triangle_helper() {
        let (result, scales) = arrow_test_build(30.0, 70.0, Some(true), Some(5.0));
        let path = result.nodes.iter().find_map(|n| match n {
            ferrum_scene::SceneNode::Path { commands, .. } => Some(commands.clone()),
            _ => None,
        }).expect("arrow=true must emit an arrowhead path");

        let p1x = scales.x.to_pixel_f64(0.0).unwrap();
        let p1y = scales.y.to_pixel_f64(0.0).unwrap();
        let p2x = scales.x.to_pixel_f64(30.0).unwrap();
        let p2y = scales.y.to_pixel_f64(70.0).unwrap();
        let expected = arrowhead_triangle(p1x, p1y, p2x, p2y, 5.0)
            .expect("non-degenerate segment must yield a triangle");
        let got: Vec<(f64, f64)> = path.iter().filter_map(|c| match c {
            ferrum_scene::PathCmd::MoveTo { x, y } | ferrum_scene::PathCmd::LineTo { x, y } => Some((*x, *y)),
            _ => None,
        }).collect();
        assert_eq!(
            got,
            vec![expected[0], expected[1], expected[2]],
            "segment's arrowhead vertices must equal arrowhead_triangle's output exactly"
        );
    }

    /// Quality-review fix: the arrowhead must inherit the SAME resolved
    /// `opacity`/`stroke_opacity` the shaft `Line` carries — not the
    /// annotation path's fixed `1.0`. Before this fix, `arrow=True` with a
    /// non-default `opacity` rendered a translucent shaft with a fully
    /// opaque triangle glued to its tip.
    #[test]
    fn segment_arrow_true_arrowhead_opacity_matches_shaft_constant_opacity() {
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
            transforms: Vec::new(), facet: None, layers: None, coord: None,
            mark_style: None, position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0_f64])),
            Arc::new(Float64Array::from(vec![0.0_f64])),
            Arc::new(Float64Array::from(vec![10.0_f64])),
            Arc::new(Float64Array::from(vec![5.0_f64])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let overrides = MarkKwargsSpec {
            arrow: Some(true),
            arrow_size: Some(6.0),
            opacity: Some(0.4),
            ..Default::default()
        };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Segment).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        let shaft = result.nodes.iter().find_map(|n| match n {
            ferrum_scene::SceneNode::Line { style, .. } => Some((style.opacity, style.stroke_opacity)),
            _ => None,
        }).expect("shaft Line must be present");
        let head_style = result.nodes.iter().find_map(|n| match n {
            ferrum_scene::SceneNode::Path { style, .. } => Some(style.clone()),
            _ => None,
        }).expect("arrowhead Path must be present");

        assert_eq!(shaft.0, 0.4, "test sanity: shaft opacity must be the overridden 0.4, not the 1.0 default");
        assert_eq!(
            head_style.opacity, shaft.0,
            "arrowhead opacity must match the shaft's resolved opacity ({}), not a fixed 1.0; got {}",
            shaft.0, head_style.opacity,
        );
        assert_eq!(
            head_style.stroke_opacity, shaft.1,
            "arrowhead stroke_opacity must match the shaft's resolved stroke_opacity"
        );
    }

    /// Discriminating fixture (close-review mutation gate): the test above
    /// sets only `opacity=`, which leaves the shaft's `opacity` and
    /// `stroke_opacity` EQUAL and `fill_opacity` at its `1.0` default — so
    /// swapping the arrowhead's opacity arguments, or hardcoding its
    /// `fill_opacity` to `1.0`, passes it. Here all three opacity-family
    /// channels are bound to distinct columns, so each slot must land in its
    /// own field: `opacity` verbatim, `stroke_opacity` as the shaft's
    /// post-multiply product, `fill_opacity` as the head's own fill.
    #[test]
    fn segment_arrowhead_places_each_opacity_family_slot_in_its_own_field() {
        use crate::spec::mark_style::MarkKwargsSpec;

        let enc = |f: &str| Some(EncodingSpec { field: f.into(), type_: None, ..Default::default() });
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Segment,
            encoding: Encoding {
                x: enc("x"),
                y: enc("y"),
                x2: enc("x2"),
                y2: enc("y2"),
                opacity: enc("op"),
                stroke_opacity: enc("so"),
                fill_opacity: enc("fo"),
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
            Field::new("op", DataType::Float64, false),
            Field::new("so", DataType::Float64, false),
            Field::new("fo", DataType::Float64, false),
        ]));
        // Three rows, asserted on the MIDDLE one: `stroke_opacity` and
        // `fill_opacity` resolve through the theme's opacity band, so an
        // endpoint row would saturate to the band's own ends (1.0 among
        // them) and could not tell the three slots apart.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0_f64, 5.0, 10.0])),
            Arc::new(Float64Array::from(vec![0.0_f64, 5.0, 10.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 25.0, 40.0])),
            Arc::new(Float64Array::from(vec![5.0_f64, 30.0, 45.0])),
            Arc::new(Float64Array::from(vec![0.2_f64, 0.5, 0.8])),
            Arc::new(Float64Array::from(vec![0.3_f64, 0.6, 0.9])),
            Arc::new(Float64Array::from(vec![0.1_f64, 0.4, 0.7])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let overrides = MarkKwargsSpec { arrow: Some(true), arrow_size: Some(6.0), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Segment).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        // The same resolver `build` consults, so the expected values are the
        // resolved per-row ones rather than the raw column values.
        let resolver = OpacityResolver::load(
            &ctx, OpacityFallback::Standard, (ctx.mark_style.paint.opacity, 1.0, 1.0),
        );
        const ROW: usize = 1;
        let (row_opacity, row_fill_opacity, row_stroke_opacity) = resolver.at_row(ROW);
        let shaft_stroke_opacity = row_opacity * row_stroke_opacity;
        assert!(
            (row_opacity - shaft_stroke_opacity).abs() > 1e-6
                && (row_opacity - row_fill_opacity).abs() > 1e-6
                && (shaft_stroke_opacity - row_fill_opacity).abs() > 1e-6
                && (row_fill_opacity - 1.0).abs() > 1e-6,
            "test sanity: the three slots must be pairwise distinct and none of them 1.0 \
             (opacity={row_opacity}, stroke={shaft_stroke_opacity}, fill={row_fill_opacity})"
        );

        let result = super::build(&ctx).unwrap();
        let heads: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            ferrum_scene::SceneNode::Path { style, .. } => Some(style.clone()),
            _ => None,
        }).collect();
        assert_eq!(heads.len(), 3, "one arrowhead per row");
        let head = &heads[ROW];

        assert_eq!(head.opacity, row_opacity, "opacity must land in the head's opacity");
        assert_eq!(
            head.stroke_opacity, shaft_stroke_opacity,
            "stroke_opacity must land in the head's stroke_opacity — the shaft's own \
             post-multiply value, not the raw channel and not the opacity slot"
        );
        assert_eq!(
            head.fill_opacity, row_fill_opacity,
            "the head is segment's only FILLED node; it must carry the row's resolved \
             fill_opacity, not a hardcoded 1.0"
        );
    }
}
