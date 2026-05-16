//! mark_text: renders a text label at (scale_x(x), scale_y(y)).
//!   - Supports ordinal x / y axes (categorical positioning, mirroring mark_rect).
//!   - A `text` encoding channel supplies explicit label content (Utf8 column).
//!
//! When the text channel is absent and y is numeric, the label falls back to
//! `format_numeric(y)`.

use crate::layout::TextAnchor;
use crate::render::draw::{col_as_f64, col_as_str, x_field, y_field, DrawCtx, MetadataColumns};
use crate::render::format::{format_numeric, format_time};
use crate::render::scale_resolve::ScaleKind;

/// Format a numeric value per a tiny subset of d3-format specs. The full grammar
/// is deliberately out of scope; we honor only ".Nf" (fixed N decimals) and
/// ".Ne" (scientific N digits) — the patterns Phase 9 `heatmap(annot=True)`
/// uses (".2f") and a couple of common variants. Falls back to format_numeric.
fn format_with_spec(v: f64, spec: Option<&str>) -> String {
    let Some(s) = spec else { return format_numeric(v) };
    let trimmed = s.strip_prefix('.').unwrap_or(s);
    // Match the trailing format character.
    let (digits_part, fmt_char) = match trimmed.chars().last() {
        Some(c @ ('f' | 'e' | 'g')) => (&trimmed[..trimmed.len() - 1], c),
        _ => return format_numeric(v),
    };
    let n: usize = digits_part.parse().unwrap_or(2);
    match fmt_char {
        'f' => format!("{v:.*}", n),
        'e' => format!("{v:.*e}", n),
        'g' => format_numeric(v),
        _ => format_numeric(v),
    }
}

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{to_scene_text_style, MarkBuildResult, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult {
        kind: MarkBatchKind::Text,
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
    let base_font_size = ctx.mark_style.font_size.unwrap_or(ctx.theme.label_font_size);
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
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

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

        nodes.push(SceneNode::Text {
            x: px + dx,
            y: py + dy,
            content: label,
            style: to_scene_text_style(
                ctx.theme.font_color,
                row_font_size,
                anchor,
                row_angle,
                &ctx.theme.font_family,
                ctx.mark_style.font_weight.as_deref(),
                ctx.mark_style.baseline.as_deref(),
                row_opacity,
            ),
        });
        indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Text,
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
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
            facet_key: None, row: 0, col: 0, strip_title: None,
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
}
