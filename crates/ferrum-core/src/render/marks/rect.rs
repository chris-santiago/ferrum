//! mark_rect: two paths —
//!   ordinal x × ordinal y → heatmap cell (original Phase 7 path);
//!   ordinal x + quantitative y + y2 → vertical band rect (boxplot box body,
//!   Phase 10c-pre). Gate: y2 encoding present → range path; else heatmap path.

use crate::layout::Rect;
use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::{ColorScale, ScaleKind};
use crate::render::svg::{FillStroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    if ctx.spec.encoding.y2.is_some() {
        draw_ordinal_range(ctx, out);
    } else {
        draw_heatmap(ctx, out);
    }
}

/// Ordinal x + quantitative y + y2 → vertical band rect per row (boxplot box body).
fn draw_ordinal_range(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return };
    let y2f = match spec.encoding.y2.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f, None => return,
    };

    let n_categories = match &ctx.scales.x {
        ScaleKind::Ordinal(_) => {
            let xs_probe = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
            count_distinct(&xs_probe).max(1)
        }
        _ => return,
    };
    let xs = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return };
    if xs.len() != ys.len() || y2s.len() != ys.len() { return; }

    let panel = ctx.panel.plot_area;
    // 60% of band width matches the Python-side mark_kwargs width: 0.6 default.
    let box_w = (panel.w / n_categories as f64) * 0.6;

    let cfield = color_field(ctx, spec);
    let color_strings: Option<Vec<Option<String>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Categorical { .. }), Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    for i in 0..xs.len() {
        let xv = match &xs[i] { Some(s) => s.as_str(), None => continue };
        let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
        let y2v = match y2s[i] { Some(v) if v.is_finite() => v, _ => continue };
        let cx = match ctx.scales.x.to_pixel_str(xv) { Some(p) => p, None => continue };
        let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
        let py2 = match ctx.scales.y.to_pixel_f64(y2v) { Some(p) => p, None => continue };
        let cx = cx + x_offsets[i];
        let rect_top = py.min(py2) + y_offsets[i];
        let rect_h = (py - py2).abs().max(1.0);
        let r = Rect { x: cx - box_w / 2.0, y: rect_top, w: box_w, h: rect_h };

        let fill = match (&ctx.scales.color, &color_strings) {
            (Some(scale @ ColorScale::Categorical { .. }), Some(values)) => {
                match values[i].as_deref() {
                    Some(v) => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                    None => ctx.mark_style.fill,
                }
            }
            _ => ctx.mark_style.fill,
        };
        let fill = with_opacity(fill, ctx.mark_style.opacity);
        out.rect(r, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        }, Some(ctx.mark_style.corner_radius));
    }
}

/// Ordinal x × ordinal y → heatmap cell (original path).
fn draw_heatmap(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return,
    };
    let xs = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    if xs.len() != ys.len() { return; }

    let panel = ctx.panel.plot_area;
    let n_x = match &ctx.scales.x { ScaleKind::Ordinal(_) => count_distinct(&xs).max(1), _ => return };
    let n_y = match &ctx.scales.y { ScaleKind::Ordinal(_) => count_distinct(&ys).max(1), _ => return };
    let cell_w = panel.w / n_x as f64;
    let cell_h = panel.h / n_y as f64;

    // Phase 10c-pre: read color values as f64 when scale is Continuous, as
    // string otherwise.
    let cfield = color_field(ctx, spec);
    let color_numeric: Option<Vec<Option<f64>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Continuous { .. }), Some(f)) => col_as_f64(ctx.batch, f).ok(),
        _ => None,
    };
    let color_strings: Option<Vec<Option<String>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Categorical { .. }), Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    for i in 0..xs.len() {
        let xs_v = match &xs[i] { Some(s) => s.as_str(), None => continue };
        let ys_v = match &ys[i] { Some(s) => s.as_str(), None => continue };
        let cx = match ctx.scales.x.to_pixel_str(xs_v) { Some(p) => p, None => continue };
        let cy = match ctx.scales.y.to_pixel_str(ys_v) { Some(p) => p, None => continue };
        let cx = cx + x_offsets[i];
        let cy = cy + y_offsets[i];

        let r = Rect { x: cx - cell_w / 2.0, y: cy - cell_h / 2.0, w: cell_w, h: cell_h };
        let fill = match (&ctx.scales.color, &color_numeric, &color_strings) {
            (Some(scale @ ColorScale::Continuous { .. }), Some(values), _) => {
                match values[i] {
                    Some(v) if v.is_finite() => {
                        scale.lookup_f64(v).unwrap_or(ctx.mark_style.fill)
                    }
                    _ => ctx.mark_style.fill,
                }
            }
            (Some(scale @ ColorScale::Categorical { .. }), _, Some(values)) => {
                match values[i].as_deref() {
                    Some(v) => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                    None => ctx.mark_style.fill,
                }
            }
            _ => ctx.mark_style.fill,
        };
        let fill = with_opacity(fill, ctx.mark_style.opacity);
        out.rect(r, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        }, Some(ctx.mark_style.corner_radius));
    }
}

fn count_distinct(values: &[Option<String>]) -> usize {
    let mut seen = std::collections::HashSet::<&str>::new();
    for v in values.iter().flatten() { seen.insert(v); }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn rect_ordinal_range_draws_band_rect_per_row() {
        // Phase 10c-pre: ordinal x + quantitative y + y2 → boxplot box body.
        use arrow::array::Float64Array;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "q1".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "q3".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None,
        };
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("cat", DataType::Utf8, false),
            arrow::datatypes::Field::new("q1",  DataType::Float64, false),
            arrow::datatypes::Field::new("q3",  DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![2.0, 4.0])),
            Arc::new(Float64Array::from(vec![6.0, 8.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert_eq!(out.finish().matches("<rect ").count(), 2, "expected 2 band rects");
    }

    #[test]
    fn rect_emits_four_cells_for_2x2_ordinal_grid() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "row".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "col".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("row", DataType::Utf8, false),
            Field::new("col", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a","a","b","b"])),
            Arc::new(StringArray::from(vec!["x","y","x","y"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<rect ").count(), 4);
    }

    #[test]
    fn rect_continuous_color_paints_distinct_fills_per_cell() {
        // Phase 10c-pre: heatmap-style — Float64 color column → continuous scale.
        // Prior bug: col_as_str failed on Float64, so all cells fell back to default fill.
        use arrow::array::Float64Array;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "row".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "col".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                color: Some(EncodingSpec {
                    field: "v".into(),
                    type_: Some(SDT::Quantitative),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("row", DataType::Utf8, false),
            Field::new("col", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a","a","b","b"])),
            Arc::new(StringArray::from(vec!["x","y","x","y"])),
            Arc::new(Float64Array::from(vec![0.0, 5.0, 2.0, 10.0])),
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<rect ").count(), 4);
        // Distinct values 0/2/5/10 must produce distinct fill colors.
        // Pull all `fill="..."` attribute values and count unique ones.
        let mut fills: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut cursor = 0usize;
        while let Some(start) = s[cursor..].find(r#"fill=""#) {
            let from = cursor + start + r#"fill=""#.len();
            if let Some(end) = s[from..].find('"') {
                fills.insert(s[from..from + end].to_string());
                cursor = from + end + 1;
            } else {
                break;
            }
        }
        // At least 3 distinct fill values among the rects (colormap may collapse extremes).
        assert!(
            fills.len() >= 3,
            "expected >=3 distinct fills, got {}: {:?}",
            fills.len(), fills
        );
    }

    #[test]
    fn rect_skips_non_ordinal_axes() {
        use arrow::array::Float64Array;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert!(!out.finish().contains("<rect "));
    }
}
