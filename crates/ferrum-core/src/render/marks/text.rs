//! mark_text: renders a text label at (scale_x(x), scale_y(y)). Phase 10c-pre
//! extends the Phase 7 stub with:
//!   - support for ordinal x / y axes (categorical positioning, mirroring mark_rect)
//!   - a `text` encoding channel for explicit label content (Utf8 column)
//!
//! Backward-compat: when the text channel is absent and y is numeric, the
//! label is `format_numeric(y)` (Phase 7 behavior).

use crate::layout::TextAnchor;
use crate::render::draw::{col_as_f64, col_as_str, x_field, y_field, DrawCtx};
use crate::render::format::format_numeric;
use crate::render::scale_resolve::ScaleKind;
use crate::render::svg::{SvgBuffer, TextStyle};

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

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return,
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
        _ => return,
    };
    let n_y = match (&ys_f, &ys_s) {
        (Some(v), _) => v.len(),
        (_, Some(v)) => v.len(),
        _ => return,
    };
    if n_x != n_y { return; }

    // Explicit text channel: Utf8 column of labels, or numeric column whose
    // values are formatted via format_numeric (heatmap-annot path). When the
    // EncodingSpec carries a `format` string (e.g. ".2f"), it is honored for
    // numeric columns. Absent text channel → format_numeric(y) (legacy).
    let text_enc = spec.encoding.text.as_ref();
    let text_field = text_enc.map(|e| e.field.as_str());
    let text_format = text_enc.and_then(|e| e.format.as_deref());
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
                            Some(format_with_spec(v, text_format))
                        })
                    })
                    .collect()
            })
        }),
    };

    let style = TextStyle {
        fill: ctx.theme.font_color,
        font_size: ctx.theme.label_font_size,
        anchor: TextAnchor::Middle,
        angle: 0.0,
    };

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

        let label: String = if let Some(t) = &texts {
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
        out.text(px, py, &label, &style);
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
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<text ").count(), 2);
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
            coord: None, mark_style: None, position: None,
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
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<text ").count(), 2);
        assert!(s.contains(">42<"), "expected literal '42' label, got: {s}");
        assert!(s.contains(">hello<"), "expected literal 'hello' label, got: {s}");
    }
}
