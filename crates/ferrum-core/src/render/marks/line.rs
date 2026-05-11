//! mark_line: render rows as a polyline.
//!
//! Two axis modes:
//! - Quantitative x: read `x` as `f64`, project via `scales.x.to_pixel_f64`.
//! - Ordinal x: read `x` as `str`, project via `scales.x.to_pixel_str`. Used
//!   by parallel coordinates where the x axis is a sequence of feature
//!   names. Ordinal y is symmetric (read as `str` via `to_pixel_str`).
//!
//! Grouping rules:
//! - Color encoding only: one polyline per color category (rows of the
//!   same color value linked in batch order). Color determines stroke.
//! - `mark_style.detail` only: one polyline per detail value, theme-default
//!   stroke. Polylines are not legendable.
//! - Both: one polyline per (color, detail) pair — color determines stroke,
//!   detail subdivides each color group. Used by parallel_coordinates so
//!   multiple samples within a class each get their own polyline.
//! - Otherwise: one polyline over all rows in batch order.

use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::ScaleKind;
use crate::render::svg::{Stroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b),
        _ => return,
    };

    // Per-row pixel projections — populated from whichever axis kind matches
    // the column's data type. Ordinal scales use `to_pixel_str`; quantitative
    // scales use `to_pixel_f64`. Either axis may be ordinal independently.
    let n_rows = ctx.batch.num_rows();
    let xs_pix: Vec<Option<f64>> = match &ctx.scales.x {
        ScaleKind::Ordinal(_) => {
            let xs_str = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
            xs_str.iter()
                .map(|opt| opt.as_deref().and_then(|s| ctx.scales.x.to_pixel_str(s)))
                .collect()
        }
        _ => {
            let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
            xs.iter()
                .map(|opt| opt
                    .filter(|v| v.is_finite())
                    .and_then(|v| ctx.scales.x.to_pixel_f64(v)))
                .collect()
        }
    };
    let ys_pix: Vec<Option<f64>> = match &ctx.scales.y {
        ScaleKind::Ordinal(_) => {
            let ys_str = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
            ys_str.iter()
                .map(|opt| opt.as_deref().and_then(|s| ctx.scales.y.to_pixel_str(s)))
                .collect()
        }
        _ => {
            let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
            ys.iter()
                .map(|opt| opt
                    .filter(|v| v.is_finite())
                    .and_then(|v| ctx.scales.y.to_pixel_f64(v)))
                .collect()
        }
    };
    if xs_pix.len() != n_rows || ys_pix.len() != n_rows {
        return;
    }

    let cf = color_field(ctx, spec);
    let color_values = cf.and_then(|f| col_as_str(ctx.batch, f).ok());
    let detail_values = ctx.mark_style.detail.as_deref()
        .and_then(|f| col_as_str(ctx.batch, f).ok());

    // Build groups: each tuple is (color_key_for_lookup, row_indices). When
    // `detail` is set, each color group is partitioned into per-detail
    // sub-groups; all sub-groups within a color class share the same lookup
    // key so they stroke identically.
    let groups: Vec<(Option<String>, Vec<usize>)> = match (
        color_values.as_ref(),
        detail_values.as_ref(),
        &ctx.scales.color,
    ) {
        (Some(cv), None, Some(_)) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in cv.iter().enumerate() {
                let key = v.clone();
                match groups.iter().position(|(k, _)| k == &key) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            groups
        }
        (None, Some(dv), _) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in dv.iter().enumerate() {
                let key = v.clone();
                match groups.iter().position(|(k, _)| k == &key) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            groups.into_iter().map(|(_, rows)| (None, rows)).collect()
        }
        (Some(cv), Some(dv), _) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for i in 0..n_rows {
                let composite = (cv[i].clone(), dv[i].clone());
                match groups.iter().position(|(_, rows)| {
                    rows.first().map(|&r| (cv[r].clone(), dv[r].clone()) == composite)
                        .unwrap_or(false)
                }) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((cv[i].clone(), vec![i])),
                }
            }
            groups
        }
        _ => vec![(None, (0..n_rows).collect())],
    };

    for (key, rows) in groups {
        let mut points: Vec<(f64, f64)> = Vec::new();
        for i in rows {
            let (cx, cy) = match (xs_pix[i], ys_pix[i]) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            points.push((cx, cy));
        }
        if points.len() < 2 { continue; }

        let stroke_color = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale)) =>
                scale.lookup(v).unwrap_or(ctx.mark_style.fill),
            _ => ctx.mark_style.fill,
        };
        out.polyline(&points, &Stroke {
            stroke: stroke_color,
            stroke_width: ctx.mark_style.stroke_width,
            stroke_dash: ctx.mark_style.stroke_dash.clone(),
        });
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
    use crate::spec::mark_style::MarkKwargsSpec;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn line_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(), mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None,
            position: None,
        }
    }

    #[test]
    fn line_emits_one_polyline_for_5_rows() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0,1.0,2.0,3.0,4.0])),
            Arc::new(Float64Array::from(vec![0.0,1.0,2.0,3.0,4.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<polyline ").count(), 1);
    }

    #[test]
    fn line_skips_when_fewer_than_two_points() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert!(!out.finish().contains("<polyline"));
    }

    #[test]
    fn line_detail_only_emits_one_polyline_per_sample() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("sample_id", DataType::Utf8, false),
        ]));
        let xs: Vec<f64> = (0..3).flat_map(|_| vec![0.0, 1.0, 2.0, 3.0]).collect();
        let ys: Vec<f64> = (0..3).flat_map(|s| vec![s as f64; 4]).collect();
        let sids: Vec<&str> = (0..12).map(|i| match i / 4 {
            0 => "s0", 1 => "s1", _ => "s2",
        }).collect();
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(sids)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec {
            detail: Some("sample_id".into()),
            ..Default::default()
        };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert_eq!(out.finish().matches("<polyline ").count(), 3);
    }

    #[test]
    fn line_color_plus_detail_emits_one_polyline_per_pair() {
        let mut spec = line_spec();
        spec.encoding.color = Some(EncodingSpec {
            field: "class".into(),
            type_: None,
            ..Default::default()
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("class", DataType::Utf8, false),
            Field::new("sample_id", DataType::Utf8, false),
        ]));
        let n = 24;
        let xs: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i / 4) as f64).collect();
        let classes: Vec<&str> = (0..n).map(|i| if (i / 4) < 3 { "A" } else { "B" }).collect();
        let sids: Vec<String> = (0..n).map(|i| format!("s{}", i / 4)).collect();
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(classes)),
            Arc::new(StringArray::from(sids)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec {
            detail: Some("sample_id".into()),
            ..Default::default()
        };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert_eq!(out.finish().matches("<polyline ").count(), 6);
    }

    #[test]
    fn line_ordinal_x_emits_polyline_via_band_centers() {
        // x: Utf8 categorical, y: Float64 quantitative. Used by
        // parallel_coordinates where features form the x axis.
        let mut spec = line_spec();
        use crate::spec::encoding::DataType as EncDataType;
        spec.encoding.x = Some(EncodingSpec {
            field: "feature".into(),
            type_: Some(EncDataType::Nominal),
            ..Default::default()
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("feature", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["f0", "f1", "f2", "f3"])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<polyline ").count(), 1);
    }
}
