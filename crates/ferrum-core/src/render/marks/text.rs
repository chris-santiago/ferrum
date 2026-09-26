//! mark_text: renders a text label at (scale_x(x), scale_y(y)).
//!   - Supports ordinal x / y axes (categorical positioning, mirroring mark_rect).
//!   - A `text` encoding channel supplies explicit label content (Utf8 column).
//!
//! When the text channel is absent and y is numeric, the label falls back to
//! `format_numeric(y)`.
//!
//! # The `text` channel read propagates its typed error (#166)
//!
//! A bound `text` channel naming a column this batch does not carry — most
//! commonly a misspelled field — used to resolve to `texts = None` (both
//! `col_as_str` and the `col_as_f64` fallback return `Err`, and `.ok()`
//! discarded whichever one ran last). Every row then fell through to the
//! y-numeric-format branch, and on an ordinal y axis that branch has no
//! numeric column to format, so every row is skipped: a chart with an intact
//! panel and zero label nodes, no diagnostic. `build` now propagates that
//! `RenderError` (`UnknownColumn` for a missing column, `UnsupportedDtype`
//! for a present column neither reader can format) with `?`, matching
//! `rule.rs`'s reader-propagation idiom (this module's `build` is now
//! fallible for exactly this reason — every other reader here is unaffected).
//! A batch that does carry the column, or a chart with no `text` channel at
//! all, is byte-identical.

use crate::layout::TextAnchor;
use crate::render::draw::{
    col_as_f64, col_as_positional_category_str, col_as_str, resolve_fill_color, x_field, y_field,
    ClearablePaint, DrawCtx,
};
use crate::render::format::{format_numeric, format_value_with_spec, Temporality};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::channels::color_column_loader;
use crate::render::scale_resolve::ScaleKind;
use crate::render::RenderError;

pub fn build(ctx: &DrawCtx) -> Result<crate::render::draw::MarkBuildResult, RenderError> {
    use crate::render::draw::{to_scene_text_style, MarkBuildResult, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let empty = || Ok(MarkBuildResult::empty(MarkBatchKind::Text));

    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return empty(),
    };

    // GH #42: text is the value-label layer for dodged bars (e.g.
    // importance_chart's `show_values=True`). Every other positional mark
    // renderer adds the per-row `__pos_x_offset__`/`__pos_y_offset__` to its
    // resolved pixel position (see tick.rs); text must do the same so dodged
    // labels track their bar's sub-band. Zero-effect when absent — the
    // accessor returns all-zero offsets.
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let x_ordinal = matches!(ctx.scales.x, ScaleKind::Ordinal(_));
    let y_ordinal = matches!(ctx.scales.y, ScaleKind::Ordinal(_));

    // The ordinal reads use `col_as_positional_category_str`, the same reader
    // `point`/`bar`/`tick`/`rule` use for an ordinal positional channel (NF-A3,
    // spec §4.4 — swept 2026-09-02). `col_as_str` errors on every non-`Utf8`
    // dtype and the `.ok()` swallowed it, so an `Int*`/`Float*`/`Bool` category
    // column on an ordinal axis left BOTH halves `None` and `build` took the
    // `return empty()` below — every label silently dropped from an otherwise
    // intact panel. Identical on `Utf8` apart from a null row, which now lands
    // in the FA-9 null band like every other positional mark's does.
    let xs_f: Option<Vec<Option<f64>>> =
        if !x_ordinal { col_as_f64(ctx.batch, xf).ok() } else { None };
    let xs_s: Option<Vec<Option<String>>> =
        if x_ordinal { col_as_positional_category_str(ctx.batch, xf).ok() } else { None };
    let ys_f: Option<Vec<Option<f64>>> =
        if !y_ordinal { col_as_f64(ctx.batch, yf).ok() } else { None };
    let ys_s: Option<Vec<Option<String>>> =
        if y_ordinal { col_as_positional_category_str(ctx.batch, yf).ok() } else { None };

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
    // `#147` close (wave 6A T21): a time-typed format on a non-temporal text
    // field refuses with the axes' own error, instead of rendering every
    // label as a 1970 epoch date.
    // The field's temporality decides whether an undeclared `%`-bearing spec
    // is a date pattern, as the axis's scale does (`#147` T23), and is the
    // one answer the refusal below checks against.
    let text_temporality = text_enc
        .and_then(|enc| crate::render::prepare::field_temporality(enc, ctx.batch))
        .unwrap_or(Temporality::NonTemporal);
    if let Some(enc) = text_enc {
        crate::render::prepare::refuse_format_temporality_mismatch_field(
            enc,
            ctx.batch,
            text_format,
            text_format_type,
            "text",
            crate::render::FormatSurface::Channel,
        )?;
    }
    let texts: Option<Vec<Option<String>>> = match text_field {
        None => None,
        Some(f) => Some(col_as_str(ctx.batch, f).or_else(|_| {
            col_as_f64(ctx.batch, f).map(|nums| {
                nums.into_iter()
                    .map(|opt_v| {
                        opt_v.and_then(|v| {
                            if !v.is_finite() {
                                return None;
                            }
                            // ONE resolution (`#147`): the same
                            // field-temporality classification the colorbar
                            // uses for the identical pair, so a
                            // `text=fm.Text(field, format="%Y/%m/%d")` channel
                            // over a datetime field APPLIES its pattern.
                            Some(format_value_with_spec(
                                v,
                                text_format,
                                text_format_type,
                                text_temporality,
                            ))
                        })
                    })
                    .collect()
            })
        })?),
    };

    let anchor = match ctx.mark_style.text.align.as_deref() {
        Some("left") => TextAnchor::Start,
        Some("right") => TextAnchor::End,
        _ => TextAnchor::Middle,
    };
    let dx = ctx.mark_style.text.dx.unwrap_or(0.0);
    let dy = ctx.mark_style.text.dy.unwrap_or(0.0);
    let base_font_size = ctx.mark_style.text.font_size.unwrap_or(ctx.theme.typography.label_font_size);
    let base_angle = ctx.mark_style.text.angle.unwrap_or(0.0);
    let base_opacity = ctx.mark_style.paint.opacity;

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

    // Per-row fill color (NF-A2 / spec §4.4, 2026-08-28 T4 amendment): color
    // channel via `ctx.scales.color` when bound on the text layer's OWN
    // encoding (never an inherited chart-level channel — see
    // `scene_build.rs::build_panel_mark_batches`'s per-layer `DrawCtx`
    // construction, which clears `ctx.spec.encoding.color` for a Text layer
    // whose color was purely inherited, on a copy local to this draw call
    // only; `LayerPrepared.encoding.color` itself — read by the legend and
    // dodge/stack position grouping — stays fully inherited), else
    // `mark_style.paint.fill` when the user set it explicitly,
    // else the theme's font color — mirroring `label.rs`'s constant-fill
    // precedence but defaulting to `font_color` (not `mark_color`) since
    // text's baseline theme style carries no mark-aware fill override (see
    // `resolve_mark_style`'s `Mark::Text` arm).
    //
    // The "fill set by user" gate reads `ctx.mark_style.paint.fill_source`
    // — the *layer-resolved* provenance — rather than `ctx.spec.mark_style` (the raw
    // `MarkKwargsSpec`). `ctx.spec` is `scene_build.rs`'s synthetic per-layer
    // `ChartSpec`, whose `mark_style` field is NOT overridden per layer (it
    // stays the chart-level kwargs via `..spec.clone()`), so a
    // `ctx.spec.mark_style` read would report the WRONG layer's `fill=` inside
    // a `LayerChart` (e.g. `mark_bar(fill=...) + mark_text()` would wrongly
    // read the bar's fill as "the text layer's fill was set"). `ctx.mark_style`
    // is always built via `resolve_mark_style(layer.mark_style.as_ref(), ...)`
    // for THIS layer, so its `fill_source` is correct in both flat and
    // layered charts.
    //
    // A user-set fill goes in through `fill_paint()` so it keeps its
    // provenance: `fill="none"` is `Cleared`, which `resolve_fill_color` keeps
    // cleared over a color-channel hit and which reaches the scene as
    // `TextStyle.color: None` — the same `#110` rule every filled mark follows.
    let base_text_paint = if ctx.mark_style.paint.fill_source.is_user_set() {
        ctx.mark_style.paint.fill_paint()
    } else {
        ClearablePaint::Paint(ctx.theme.colors.font_color)
    };
    let (color_values_str, color_values_f64) = color_column_loader(ctx);

    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep so metadata is
    // aligned to kept nodes only (#6 defect class fix).
    let mut acc = MarkNodes::with_capacity(n_x);
    // #230-3: one exact `(px, py)` data anchor per emitted node — the
    // scale-mapped position BEFORE `dx`/`dy` is baked into the node's own
    // `x`/`y` below. A break's glyph remap needs this anchor, not the node's
    // own (already-offset) coordinate, so `dx`/`dy` translates rigidly with
    // the glyph instead of being read back as if it were the data position.
    let mut anchors: Vec<(f64, f64)> = Vec::with_capacity(n_x);

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
        let px = px + x_offsets[i];

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
        let py = py + y_offsets[i];

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
        let label = if let Some(limit) = ctx.mark_style.text.limit {
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

        // Resolve per-row fill paint: color channel wins when bound, else the
        // resolved constant (`base_text_paint`, see above) — unless that
        // constant is a clear, which outranks the channel.
        let row_paint = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_values_str.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_values_f64.as_ref().and_then(|v| v.get(i).copied().flatten()),
            base_text_paint,
        );

        acc.push(SceneNode::Text {
            x: px + dx,
            y: py + dy,
            content: label,
            slot: None,
            style: to_scene_text_style(
                row_paint,
                row_font_size,
                anchor,
                row_angle,
                &ctx.theme.typography.font_family,
                ctx.mark_style.text.font_weight.as_deref(),
                ctx.mark_style.text.baseline.as_deref(),
                row_opacity,
            ),
        }, i);
        anchors.push((px, py));
    }

    let (nodes, data_indices) = acc.finalize();

    Ok(MarkBuildResult::populated(MarkBatchKind::Text, nodes, data_indices, &meta)
        .with_glyph_anchors(anchors))
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx).unwrap();
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Text { .. })).count(), 2);
        // Check that the explicit label content is present in the Text nodes.
        let contents: Vec<&str> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Text { content, .. } = n { Some(content.as_str()) } else { None }
        }).collect();
        assert!(contents.contains(&"42"), "expected literal '42' label, got: {contents:?}");
        assert!(contents.contains(&"hello"), "expected literal 'hello' label, got: {contents:?}");
    }

    /// `#147` close (wave 6A T23, user ruling): the text channel reads an
    /// untyped spec by its field's temporality, as the axis reads it by the
    /// scale's. Over a datetime column `"%Y"` renders years (it used to
    /// refuse as a d3 spec) and `".2f"` refuses (it used to format the
    /// epoch-ms count, `978307200000.00`).
    #[test]
    fn text_reads_an_untyped_spec_by_the_field_temporality() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::{StringArray, TimestampMillisecondArray};
        let spec_with = |format: &str| ChartSpec {
            data: DataRef::default(),
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "px".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "ax".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                text: Some(EncodingSpec { field: "d".into(), format: Some(format.into()), ..Default::default() }),
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
            Field::new("d", DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None), false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(StringArray::from(vec!["x", "y"])),
            Arc::new(TimestampMillisecondArray::from(vec![978_307_200_000_i64, 1_009_843_200_000])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let build = |spec: &ChartSpec| {
            let (scales, _) = resolve_scales(spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
            let ctx = DrawCtx {
                spec, panel: &panel, theme: &theme,
                scales: &scales, batch: &batch, mark_style: &mark_style,
            };
            super::build(&ctx).map(|r| {
                r.nodes.iter().filter_map(|n| match n {
                    ferrum_scene::SceneNode::Text { content, .. } => Some(content.clone()),
                    _ => None,
                }).collect::<Vec<_>>()
            })
        };
        assert_eq!(build(&spec_with("%Y")).unwrap(), ["2001", "2002"]);
        assert_eq!(
            build(&spec_with(".2f")).unwrap_err(),
            crate::render::RenderError::NumberFormatOnTemporalScale {
                channel: "text",
                surface: crate::render::FormatSurface::Channel,
                format: ".2f".into(),
                coord_flipped: false,
            }
        );
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = super::build(&ctx).unwrap();
        assert_eq!(
            result.kind,
            ferrum_scene::MarkBatchKind::Text,
            "mark_text must still produce MarkBatchKind::Text, got {:?}",
            result.kind
        );
    }

    // ── GH #42: dodge position-offset consumption ────────────────────────────
    //
    // Text is the value-label layer for dodged bars (e.g. importance_chart's
    // `show_values=True`). Every other positional mark renderer (tick, rect,
    // point, ...) reads `__pos_x_offset__`/`__pos_y_offset__` via
    // `read_position_offsets` and adds the per-row offset to its resolved
    // pixel position; text alone ignored them, so dodged labels sat at the
    // undodged band center. These offset columns are absent from user data —
    // no field of that name is ever encoded — so this is additive: a batch
    // without them renders byte-identically (`text_no_skip_tooltips_unchanged`
    // and friends above already pin that path).

    /// With `__pos_x_offset__` present, each glyph's x shifts by its row's
    /// offset relative to the undodged position.
    #[test]
    fn text_applies_pos_x_offset_when_present() {
        use arrow::array::Float64Array as F64Arr;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("__pos_x_offset__", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(F64Arr::from(vec![7.5, -3.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let result = super::build(&ctx).unwrap();
        let xs: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Text { x, .. } = n { Some(*x) } else { None }
        }).collect();
        assert_eq!(xs.len(), 2);

        // Same spec/batch minus the offset column → undodged baseline x.
        let baseline_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let baseline_batch = arrow::record_batch::RecordBatch::try_new(baseline_schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
        ]).unwrap();
        let (baseline_scales, _) = resolve_scales(&spec, &baseline_batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let baseline_ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &baseline_scales, batch: &baseline_batch, mark_style: &mark_style };
        let baseline_result = super::build(&baseline_ctx).unwrap();
        let baseline_xs: Vec<f64> = baseline_result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Text { x, .. } = n { Some(*x) } else { None }
        }).collect();

        assert!((xs[0] - (baseline_xs[0] + 7.5)).abs() < 1e-9,
            "row 0 x must shift by its __pos_x_offset__ (7.5); got {} vs baseline {}", xs[0], baseline_xs[0]);
        assert!((xs[1] - (baseline_xs[1] + -3.0)).abs() < 1e-9,
            "row 1 x must shift by its __pos_x_offset__ (-3.0); got {} vs baseline {}", xs[1], baseline_xs[1]);
    }

    /// Without offset columns, text output is unaffected — the accessor
    /// returns all-zero offsets and glyph positions are unchanged.
    #[test]
    fn text_no_offset_columns_zero_effect() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        let xs: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Text { x, .. } = n { Some(*x) } else { None }
        }).collect();
        let expected = vec![
            scales.x.to_pixel_f64(10.0).unwrap(),
            scales.x.to_pixel_f64(50.0).unwrap(),
        ];
        assert_eq!(xs, expected, "no offset columns: text x must equal raw scaled position");
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        assert_eq!(result.nodes.len(), 3, "all 3 rows must produce nodes");
        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 3, "tooltip count must equal node count");
        let values: Vec<&str> = tooltips.iter().map(|t| t.fields[0].value.as_str()).collect();
        assert_eq!(values, vec!["tip_a", "tip_b", "tip_c"],
            "no-skip: tooltips must be in original row order");
    }

    // ── Ported from bug_hunt_marks_rendering_r2.rs (R1) ─────────────────────
    // `mark_style.text.limit` truncation boundary conditions, exercised via
    // the real `build` pipeline with an explicit `text` channel.

    fn limit_test_spec() -> ChartSpec {
        use crate::spec::encoding::DataType as SDT;
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                text: Some(EncodingSpec { field: "label".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        }
    }

    fn build_with_limit(limit: Option<usize>) -> String {
        use crate::spec::mark_style::MarkKwargsSpec;
        use arrow::array::StringArray;
        let spec = limit_test_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("label", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(StringArray::from(vec!["Hello"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec { limit, ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        result.nodes.iter().find_map(|n| {
            if let ferrum_scene::SceneNode::Text { content, .. } = n { Some(content.clone()) } else { None }
        }).expect("expected exactly one Text node")
    }

    #[test]
    fn text_limit_equal_to_label_length_does_not_truncate() {
        // "Hello".chars().count() == 5 == limit -> the `>` comparison is false.
        assert_eq!(build_with_limit(Some(5)), "Hello");
    }

    #[test]
    fn text_limit_one_truncates_to_bare_ellipsis() {
        // limit=1: saturating_sub(1) keeps 0 chars, so only the ellipsis remains.
        assert_eq!(build_with_limit(Some(1)), "\u{2026}");
    }

    #[test]
    fn text_limit_zero_is_treated_as_unset() {
        // `limit > 0` is false for limit=0, so no truncation occurs at all.
        assert_eq!(build_with_limit(Some(0)), "Hello");
    }

    // ── Task 4: fill=/encode(color=) precedence (spec §4.4) ─────────────────
    //
    // Precedence: color channel via `ctx.scales.color` when bound, else
    // `mark_style.paint.fill` when the user set it explicitly, else the theme's
    // font color (mirrors `label.rs`'s constant-fill resolution, but text's
    // fallback is `font_color` rather than `mark_color` — see the comment on
    // `base_text_color` in `build`).

    use crate::render::draw::to_scene_color;
    use crate::spec::mark_style::MarkKwargsSpec;

    /// Two-row spec/batch with quantitative x/y and an optional Utf8 `c`
    /// (color) column. `fill_override` maps to `mark_style.fill=`;
    /// `with_color_channel` binds `encoding.color` to the `c` field.
    fn color_precedence_ctx_colors(
        fill_override: Option<&str>,
        with_color_channel: bool,
    ) -> Vec<ferrum_scene::Color> {
        use arrow::array::StringArray;

        let overrides = fill_override.map(|hex| MarkKwargsSpec { fill: Some(hex.into()), ..Default::default() });
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: if with_color_channel {
                    Some(EncodingSpec { field: "c".into(), type_: None, ..Default::default() })
                } else {
                    None
                },
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: overrides.clone(), position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("c", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(overrides.as_ref(), &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Text { style, .. } = n { style.color } else { None }
        }).collect()
    }

    /// NF-A2: no `fill=` override and no `color` channel — every row's text
    /// color must stay exactly the theme's font color (byte-identical default).
    #[test]
    fn text_default_color_is_theme_font_color() {
        let colors = color_precedence_ctx_colors(None, false);
        let expected = to_scene_color(ThemeInputs::default().colors.font_color);
        assert_eq!(colors.len(), 2);
        assert!(colors.iter().all(|c| *c == expected),
            "default mark_text color must equal theme.colors.font_color; got {colors:?}, expected {expected:?}");
    }

    /// `fill=` honored: with no color channel bound, an explicit `fill=`
    /// override wins over the theme font color for every row.
    #[test]
    fn text_fill_kwarg_sets_constant_color() {
        let colors = color_precedence_ctx_colors(Some("#ff0000"), false);
        let red = ferrum_scene::Color { r: 0xff, g: 0x00, b: 0x00, a: 255 };
        assert_eq!(colors.len(), 2);
        assert!(colors.iter().all(|c| *c == red),
            "fill='#ff0000' must set every row's text color to red; got {colors:?}");
        let font_color = to_scene_color(ThemeInputs::default().colors.font_color);
        assert_ne!(red, font_color, "test fixture sanity: red must differ from the theme font color");
    }

    /// `encode(color=)` honored: with a bound color channel and no `fill=`
    /// override, distinct category values resolve to distinct colors (not the
    /// theme font color).
    #[test]
    fn text_color_channel_sets_per_row_color() {
        let colors = color_precedence_ctx_colors(None, true);
        assert_eq!(colors.len(), 2);
        assert_ne!(colors[0], colors[1],
            "rows with distinct color-channel categories must resolve to distinct colors; got {colors:?}");
        let font_color = to_scene_color(ThemeInputs::default().colors.font_color);
        assert!(colors.iter().all(|c| *c != font_color),
            "color-channel rows must not fall back to the theme font color; got {colors:?}");
    }

    /// Precedence: a bound color channel wins over an explicit `fill=`
    /// override (channel is checked first).
    #[test]
    fn text_color_channel_overrides_fill_kwarg() {
        let colors = color_precedence_ctx_colors(Some("#ff0000"), true);
        let red = ferrum_scene::Color { r: 0xff, g: 0x00, b: 0x00, a: 255 };
        assert_eq!(colors.len(), 2);
        assert!(colors.iter().all(|c| *c != red),
            "a bound color channel must override fill='#ff0000'; got {colors:?}");
        assert_ne!(colors[0], colors[1],
            "channel-resolved colors must still vary per category; got {colors:?}");
    }

    // ── NF-A3: ordinal positional reads key off the scale, not the dtype ─────

    /// An `Int64` nominal `x` on an ordinal scale must place its labels, not
    /// drop them.
    ///
    /// RED (verified in place): with `col_as_str` here, the read returns `Err`
    /// for `Int64`, `.ok()` swallows it, BOTH `xs_f` (skipped — the scale is
    /// ordinal) and `xs_s` end up `None`, and `build` takes the `n_x` match's
    /// `return empty()`. Every data label vanishes from an otherwise intact
    /// panel — the silent-empty class `rule.rs` eliminated, still live in a
    /// reader choice. The Utf8 twin is asserted alongside so the test cannot
    /// pass by producing zero labels for both.
    #[test]
    fn text_places_labels_on_an_int64_ordinal_x() {
        use arrow::array::{Int64Array, StringArray};
        use crate::spec::encoding::DataType as SpecType;

        let build_for = |x_field: Field, x_col: arrow::array::ArrayRef| -> Vec<String> {
            let spec = ChartSpec {
                data: DataRef::default(),
                mark: Mark::Text,
                encoding: Encoding {
                    x: Some(EncodingSpec { field: "x".into(), type_: Some(SpecType::Nominal), ..Default::default() }),
                    y: Some(EncodingSpec { field: "y".into(), type_: Some(SpecType::Quantitative), ..Default::default() }),
                    text: Some(EncodingSpec { field: "lbl".into(), ..Default::default() }),
                    ..Default::default()
                },
                transforms: Vec::new(), facet: None, layers: None, coord: None,
                mark_style: None, position: None, title: None, axis_x: None, axis_y: None,
                selections: Vec::new(), conditionals: Vec::new(), chart_description: None,
                params: Vec::new(),
            };
            let schema = Arc::new(Schema::new(vec![
                x_field,
                Field::new("y", DataType::Float64, false),
                Field::new("lbl", DataType::Utf8, false),
            ]));
            let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
                x_col,
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(StringArray::from(vec!["one", "two", "three"])),
            ]).unwrap();
            let theme = ThemeInputs::default();
            let panel = PanelLayout {
                plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
                facet_key: None, row: 0, col: 0,
                strip_title: None, row_strip_title: None, row_facet_key: None,
            };
            let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
            assert!(
                matches!(scales.x, crate::render::scale_resolve::ScaleKind::Ordinal(_)),
                "fixture must resolve an ordinal x scale, else it exercises the numeric branch"
            );
            let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
            let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
            super::build(&ctx).unwrap().nodes.iter().filter_map(|n| match n {
                ferrum_scene::SceneNode::Text { content, .. } => Some(content.clone()),
                _ => None,
            }).collect()
        };

        let utf8 = build_for(
            Field::new("x", DataType::Utf8, false),
            Arc::new(StringArray::from(vec!["10", "20", "30"])),
        );
        assert_eq!(utf8, vec!["one", "two", "three"],
            "Utf8 reference: all three labels must render");

        let int64 = build_for(
            Field::new("x", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![10_i64, 20, 30])),
        );
        assert_eq!(int64, utf8,
            "an Int64 nominal x must place the same three labels as its Utf8 twin; \
             got {int64:?}. An empty vec here is the col_as_str drop.");
    }

    // ── #166: `text` channel reader propagation ──────────────────────────

    /// `encode(text="nope")` naming a column absent from the batch used to
    /// resolve `texts = None` silently (both `col_as_str` and its
    /// `col_as_f64` fallback error, and `.ok()` discarded whichever ran
    /// last). On this fixture's ordinal y axis the per-row fallback branch
    /// (`match &ys_f { ... None => continue }`) has no numeric column to
    /// format, so every row was skipped — an intact panel with zero label
    /// nodes and no diagnostic. `build` now propagates the reader's typed
    /// error with `?` instead (rule.rs's reader-propagation idiom).
    #[test]
    fn text_encode_text_unknown_column_refuses_instead_of_rendering_blank() {
        use crate::spec::encoding::DataType as SpecType;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SpecType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SpecType::Ordinal), ..Default::default() }),
                text: Some(EncodingSpec { field: "nope".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None,
            mark_style: None, position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Utf8, false),
            Field::new("y", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(StringArray::from(vec!["p", "q"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let err = match super::build(&ctx) {
            Err(e) => e,
            Ok(r) => panic!(
                "a text= column absent from the batch must refuse, not render {} label nodes silently",
                r.nodes.len()
            ),
        };
        assert!(
            matches!(&err, crate::render::RenderError::UnknownColumn { name } if name == "nope"),
            "expected UnknownColumn naming 'nope'; got {err:?}"
        );
    }

    /// The text-reader twin of `rule.rs`'s
    /// `rule_treats_a_color_column_absent_from_its_layer_batch_as_no_color`
    /// (spec §7's structural `Ok(None)` carve-out, `channels.rs:157-192`): a
    /// `color` field this layer's own batch does not carry (the ordinary
    /// layered-chart shape `Encoding::inherit_from` produces, e.g.
    /// `mark_qq(line=True)`'s reference layer) is not an error, even though
    /// `build` is now fallible for the `text` channel above.
    /// `color_column_loader` stays infallible and untouched by this task, so
    /// this pins that making `build`'s return type fallible did not also
    /// start refusing the case the carve-out exists for.
    #[test]
    fn text_treats_a_color_column_absent_from_its_layer_batch_as_no_color() {
        use crate::spec::encoding::DataType as SpecType;

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SpecType::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SpecType::Quantitative), ..Default::default() }),
                color: Some(EncodingSpec { field: "cat".into(), type_: Some(SpecType::Nominal), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None,
            mark_style: None, position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None,
            params: Vec::new(),
        };
        // The layer's own batch — no `cat` column, mirroring `qq_line`.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        // Scale resolution must not itself refuse the color column absent
        // from this batch, so resolve against an x/y-only spec — `build`
        // runs on the real spec, which is what this test exercises.
        let scale_spec = ChartSpec {
            encoding: Encoding { color: None, ..spec.encoding.clone() },
            ..spec.clone()
        };
        let (scales, _) = resolve_scales(&scale_spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let result = super::build(&ctx).expect(
            "a color column absent from this layer's own batch must render with the \
             fallback color, not refuse",
        );
        assert_eq!(
            result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Text { .. })).count(),
            2
        );
    }
}
