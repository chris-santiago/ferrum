//! mark_area: filled region between y(x) and the x-axis baseline. Single area
//! over all rows when no color encoding; one area per category otherwise.

use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::ColorScale;
use crate::render::svg::{FillStroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return,
    };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };

    let baseline_y = ctx.panel.plot_area.y + ctx.panel.plot_area.h;

    let cf = color_field(ctx, spec);
    let color_values = cf.and_then(|f| col_as_str(ctx.batch, f).ok());

    let groups: Vec<(Option<String>, Vec<usize>)> = match (color_values.as_ref(), &ctx.scales.color) {
        (Some(values), Some(_)) => {
            let mut g: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in values.iter().enumerate() {
                let key = v.clone();
                match g.iter().position(|(k, _)| k == &key) {
                    Some(p) => g[p].1.push(i),
                    None => g.push((key, vec![i])),
                }
            }
            g
        }
        _ => vec![(None, (0..xs.len()).collect())],
    };

    // Phase 9c — per-row position-adjustment pixel offsets (Stack).
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    for (key, rows) in groups {
        let mut top: Vec<(f64, f64)> = Vec::new();
        for i in rows {
            let (xv, yv) = match (xs[i], ys[i]) {
                (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a, b),
                _ => continue,
            };
            let cx = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            let cy = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
            let cx = cx + x_offsets.get(i).copied().unwrap_or(0.0);
            let cy = cy + y_offsets.get(i).copied().unwrap_or(0.0);
            top.push((cx, cy));
        }
        if top.len() < 2 { continue; }
        let path = build_area_path(&top, baseline_y);
        let fill = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale @ ColorScale::Categorical { .. })) => {
                let base = scale.lookup(v).unwrap_or(ctx.mark_style.fill);
                crate::render::color::with_opacity(base, ctx.theme.area_opacity)
            }
            _ => ctx.mark_style.fill,
        };
        out.path(&path, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        });
    }
}

fn build_area_path(top: &[(f64, f64)], baseline: f64) -> String {
    use crate::render::svg::fmt_f;
    let mut d = String::new();
    let (x0, y0) = top[0];
    d.push_str(&format!("M{} {}", fmt_f(x0), fmt_f(y0)));
    for &(x, y) in &top[1..] {
        d.push_str(&format!(" L{} {}", fmt_f(x), fmt_f(y)));
    }
    let last_x = top[top.len() - 1].0;
    d.push_str(&format!(" L{} {}", fmt_f(last_x), fmt_f(baseline)));
    d.push_str(&format!(" L{} {}", fmt_f(x0), fmt_f(baseline)));
    d.push_str(" Z");
    d
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

    fn area_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(), mark: Mark::Area,
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
    fn area_emits_one_path_with_z_close() {
        let spec = area_spec();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Area);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<path ").count(), 1);
        assert!(s.contains(" Z\""), "area path must close with Z");
    }

    #[test]
    fn area_uses_translucent_fill() {
        let spec = area_spec();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Area);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert!(out.finish().contains("rgba("));
    }
}
