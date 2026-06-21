//! mark_text: renders a text label at (scale_x(x), scale_y(y)).
//!   - Supports ordinal x / y axes (categorical positioning, mirroring mark_rect).
//!   - A `text` encoding channel supplies explicit label content (Utf8 column).
//!
//! When the text channel is absent and y is numeric, the label falls back to
//! `format_numeric(y)`.

use crate::layout::TextAnchor;
use crate::render::draw::{col_as_f64, col_as_str, x_field, y_field, DrawCtx};
use crate::render::format::{format_numeric, format_time, format_with_spec};
use crate::render::mark_nodes::MarkNodes;
use crate::render::scale_resolve::ScaleKind;

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{to_scene_text_style, MarkBuildResult, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult::empty(MarkBatchKind::Text);

    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return empty(),
    };

    let x_ordinal = matches!(ctx.scales.x, ScaleKind::Ordinal(_));
    let y_ordinal = matches!(ctx.scales.y, ScaleKind::Ordinal(_));

    let xs_f: Option<Vec<Option<f64>>> =
        if !x_ordinal { col_as_f64(ctx.batch, xf).ok() } else { None };
    let xs_s: Option<Vec<Option<String>>> =
        if x_ordinal { col_as_str(ctx.batch, xf).ok() } else { None };
    let ys_f: Option<Vec<Option<f64>>> =
        if !y_ordinal { col_as_f64(ctx.batch, yf).ok() } else { None };
    let ys_s: Option<Vec<Option<String>>> =
        if y_ordinal { col_as_str(ctx.batch, yf).ok() } else { None };

    let n_x = match (&xs_f, &xs_s) {
        (Some(v), _) => v.len(),
        (_, Some(v)) => v.len(),
        _ => return empty(),
    };
    let n_y = match (&ys_f, &ys_s) {
        (Some(v), _) => v.len(),
        (_, Some(v)) => v.len(),
        _ => return empty(),
    };
    if n_x != n_y { return empty(); }

    // Explicit text channel (same resolution as draw()).
    let text_enc = spec.encoding.text.as_ref();
    let text_field = text_enc.map(|e| e.field.as_str());
    let text_format = text_enc.and_then(|e| e.format.as_deref());
    let text_format_type = text_enc.and_then(|e| e.format_type.as_deref());
    let texts: Option<Vec<Option<String>>> = match text_field {
        None => None,
        Some(f) => col_as_str(ctx.batch, f).ok().or_else(|| {
            col_as_f64(ctx.batch, f).ok().map(|nums| {
                nums.into_iter()
                    .map(|opt_v| {
                        opt_v.and_then(|v| {
                            if !v.is_finite() {
                                return None;
                            }
                            if text_format_type == Some("time") {
                                Some(format_time(v as i64, 86_400_000))
                            } else {
                                Some(format_with_spec(v, text_format))
                            }
                        })
                    })
                    .collect()
            })
        }),
    };

    let anchor = match ctx.mark_style.align.as_deref() {
        Some("left") => TextAnchor::Start,
        Some("right") => TextAnchor::End,
        _ => TextAnchor::Middle,
    };
    let dx = ctx.mark_style.dx.unwrap_or(0.0);
    let dy = ctx.mark_style.dy.unwrap_or(0.0);
    let base_font_size = ctx.mark_style.font_size.unwrap_or(ctx.theme.typography.label_font_size);
    let base_angle = ctx.mark_style.angle.unwrap_or(0.0);
    let base_opacity = ctx.mark_style.opacity;

    // Per-row encoding channels (same pattern as point.rs).
    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let size_values: Option<Vec<Option<f64>>> = spec.encoding.size
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let angle_values: Option<Vec<Option<f64>>> = spec.encoding.angle
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep so metadata is
    // aligned to kept nodes only (#6 defect class fix).
    let mut acc = MarkNodes::with_capacity(n_x);

    for i in 0..n_x {
        let px = if let Some(xs) = &xs_f {
            match xs[i] {
                Some(v) if v.is_finite() => match ctx.scales.x.to_pixel_f64(v) {
                    Some(p) => p, None => continue,
                },
                _ => continue,
            }
        } else if let Some(xs) = &xs_s {
            match xs[i].as_deref() {
                Some(s) => match ctx.scales.x.to_pixel_str(s) {
                    Some(p) => p, None => continue,
                },
                None => continue,
            }
        } else { continue };

        let py = if let Some(ys) = &ys_f {
            match ys[i] {
                Some(v) if v.is_finite() => match ctx.scales.y.to_pixel_f64(v) {
                    Some(p) => p, None => continue,
                },
                _ => continue,
            }
        } else if let Some(ys) = &ys_s {
            match ys[i].as_deref() {
                Some(s) => match ctx.scales.y.to_pixel_str(s) {
                    Some(p) => p, None => continue,
                },
                None => continue,
            }
        } else { continue };

        let raw_label: String = if let Some(t) = &texts {
            match &t[i] {
                Some(s) => s.clone(),
                None => continue,
            }
        } else {
            match &ys_f {
                Some(ys) => match ys[i] {
                    Some(v) if v.is_finite() => format_numeric(v),
                    _ => continue,
                },
                None => continue,
            }
        };

        // S7: truncate label to `limit` characters (including the ellipsis).
        let label = if let Some(limit) = ctx.mark_style.limit {
            if limit > 0 && raw_label.chars().count() > limit {
                let truncated: String = raw_label.chars().take(limit.saturating_sub(1)).collect();
                format!("{truncated}\u{2026}") // …
            } else {
                raw_label
            }
        } else {
            raw_label
        };

        // Resolve per-row opacity (fill-opacity on text).
        let row_opacity = opacity_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(base_opacity);

        // Resolve per-row font-size from size encoding.
        let row_font_size = size_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| *v > 0.0 && v.is_finite())
            .unwrap_or(base_font_size);

        // Resolve per-row angle (rotation) from angle encoding.
        let row_angle = angle_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| v.is_finite())
            .unwrap_or(base_angle);

        acc.push(SceneNode::Text {
            x: px + dx,
            y: py + dy,
            content: label,
            style: to_scene_text_style(
                ctx.theme.colors.font_color,
                row_font_size,
                anchor,
                row_angle,
                &ctx.theme.typography.font_family,
                ctx.mark_style.font_weight.as_deref(),
                ctx.mark_style.baseline.as_deref(),
                row_opacity,
            ),
        }, i);
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Text,
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
    fn text_emits_one_text_element_per_row() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Text,
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
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Text { .. })).count(), 2);
    }

    #[test]
    fn text_channel_renders_explicit_labels_on_ordinal_axes() {
        // Phase 10c-pre: confusion-matrix-style labels. Ordinal x/y with an
        // explicit `text` channel reading a Utf8 column.
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "px".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "ax".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                text: Some(EncodingSpec { field: "label".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("px", DataType::Utf8, false),
            Field::new("ax", DataType::Utf8, false),
            Field::new("label", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(StringArray::from(vec!["x", "y"])),
            Arc::new(StringArray::from(vec!["42", "hello"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &crate::layout::ThemeInputs::default(),
        ).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text);
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Text { .. })).count(), 2);
        // Check that the explicit label content is present in the Text nodes.
        let contents: Vec<&str> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Text { content, .. } = n { Some(content.as_str()) } else { None }
        }).collect();
        assert!(contents.contains(&"42"), "expected literal '42' label, got: {contents:?}");
        assert!(contents.contains(&"hello"), "expected literal 'hello' label, got: {contents:?}");
    }

    // ── Task 2: mark_text must still produce MarkBatchKind::Text (not Label) ──

    /// Plain mark_text build produces MarkBuildResult with kind == MarkBatchKind::Text.
    /// This guards against accidentally rerouting mark_text to Label.
    #[test]
    fn text_build_still_produces_text_kind() {
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
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
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 80.0])),
            Arc::new(Float64Array::from(vec![20.0, 70.0])),
        ])
        .unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text);
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = super::build(&ctx);
        assert_eq!(
            result.kind,
            ferrum_scene::MarkBatchKind::Text,
            "mark_text must still produce MarkBatchKind::Text, got {:?}",
            result.kind
        );
    }

    // ── Metadata-alignment regression tests (#6 defect class) ────────────────
    //
    // Text emits 1 node per kept row. A null text value causes `continue`, so
    // node indices diverge from row indices as soon as any row is skipped.
    //
    // Fail-before: `build_metadata(ctx)` produced full per-row vectors before the
    // loop; node j got row j's metadata regardless of skips. These tests would
    // have failed: node 1's tooltip would be "tip_b" (skipped row) not "tip_c".
    //
    // Pass-after: `MarkNodes` + `build_metadata_for_indices` aligns metadata to
    // kept nodes only.

    fn make_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        }
    }

    /// Regression: `build_text` with a null text-channel value skips that row.
    /// The tooltip on each surviving node must point to its true source row.
    ///
    /// Batch: 3 rows, text=[Some("a"), None, Some("c")],
    /// tooltip=["tip_a","tip_b","tip_c"]. Row 1 (null text) is skipped → 2
    /// nodes. Node 1 must have "tip_c" (row 2), not "tip_b" (row 1, old bug).
    #[test]
    fn text_skipped_null_text_tooltip_aligned() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                text: Some(EncodingSpec { field: "lbl".into(), ..Default::default() }),
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
            Field::new("lbl", DataType::Utf8,    true),   // nullable — row 1 null → skip
            Field::new("tip", DataType::Utf8,    false),
        ]));
        // Row 1 has null label → skipped (the `None => continue` on the text value).
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // 2 nodes survive (row 1 with null label is skipped).
        assert_eq!(result.nodes.len(), 2,
            "expected 2 text nodes after null-label skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be 'tip_a' (row 0); got '{t0}'");

        // Node 1 → row 2 → "tip_c". Old code: "tip_b" (row 1, the alignment bug).
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be 'tip_c' (row 2), not 'tip_b' (row 1); got '{t1}'. \
             This fails on pre-migration code using build_metadata(ctx).");
    }

    /// Href-channel alignment: a null text value skips row 1; href on node 1
    /// must point to row 2's url ("url_c"), not row 1's ("url_b").
    #[test]
    fn text_skipped_null_text_href_aligned() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                text: Some(EncodingSpec { field: "lbl".into(), ..Default::default() }),
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
            Field::new("y",   DataType::Float64, false),
            Field::new("lbl", DataType::Utf8,    true),
            Field::new("url", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            Arc::new(StringArray::from(vec!["url_a", "url_b", "url_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2, "expected 2 text nodes");
        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), 2, "href count must equal node count");
        assert_eq!(hrefs[0].as_deref(), Some("url_a"), "node 0 href must be 'url_a'");
        assert_eq!(hrefs[1].as_deref(), Some("url_c"),
            "node 1 href must be 'url_c' (row 2), not 'url_b' (row 1); \
             old build_metadata would give 'url_b'");
    }

    /// No-skip backward-compat: when no rows are skipped all tooltips appear in
    /// original row order — same result as the old `build_metadata(ctx)` path.
    #[test]
    fn text_no_skip_tooltips_unchanged() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                text: Some(EncodingSpec { field: "lbl".into(), ..Default::default() }),
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
            Field::new("lbl", DataType::Utf8,    false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        // All rows have non-null text → no skipping.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 3, "all 3 rows must produce nodes");
        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 3, "tooltip count must equal node count");
        let values: Vec<&str> = tooltips.iter().map(|t| t.fields[0].value.as_str()).collect();
        assert_eq!(values, vec!["tip_a", "tip_b", "tip_c"],
            "no-skip: tooltips must be in original row order");
    }
}
