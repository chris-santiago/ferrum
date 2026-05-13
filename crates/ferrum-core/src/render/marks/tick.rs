//! mark_tick: three modes —
//!   quantitative x → rug-style vertical segment at panel bottom (original);
//!   ordinal x + quantitative y → horizontal tick at y position (boxplot median,
//!   Phase 10c-pre);
//!   ordinal y + quantitative x → vertical tick at x position (strip plot).

use crate::render::draw::{col_as_f64, col_as_str, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::ScaleKind;
use crate::render::svg::{Stroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let panel = ctx.panel.plot_area;
    let style = Stroke {
        stroke: ctx.mark_style.fill,
        stroke_width: ctx.mark_style.stroke_width.max(1.0),
        stroke_dash: None,
    };
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return };
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    // Ordinal x + quantitative y → horizontal tick at data y position.
    if matches!(&ctx.scales.x, ScaleKind::Ordinal(_)) {
        if let Some(yf) = y_field(ctx, spec) {
            let xs = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
            let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
            let n_cats = {
                let mut set = std::collections::HashSet::<&str>::new();
                for v in xs.iter().flatten() { set.insert(v.as_str()); }
                set.len().max(1)
            };
            // S8: band_size overrides the 0.3 default (fraction of band width).
            let tick_half = (panel.w / n_cats as f64) * ctx.mark_style.band_size.unwrap_or(0.3);
            for i in 0..xs.len() {
                let xv = match &xs[i] { Some(s) => s.as_str(), None => continue };
                let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
                let cx = match ctx.scales.x.to_pixel_str(xv) { Some(p) => p, None => continue };
                let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
                let cx = cx + x_offsets[i];
                let py = py + y_offsets[i];
                out.line(cx - tick_half, py, cx + tick_half, py, &style);
            }
            return;
        }
    }

    // Ordinal y + quantitative x → vertical tick at data x position (strip plot).
    if matches!(&ctx.scales.y, ScaleKind::Ordinal(_)) {
        if let Some(yf) = y_field(ctx, spec) {
            let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
            let ys = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
            let n_cats = {
                let mut set = std::collections::HashSet::<&str>::new();
                for v in ys.iter().flatten() { set.insert(v.as_str()); }
                set.len().max(1)
            };
            let tick_half = (panel.h / n_cats as f64) * ctx.mark_style.band_size.unwrap_or(0.3);
            for i in 0..xs.len() {
                let xv = match xs[i] { Some(v) if v.is_finite() => v, _ => continue };
                let yv = match &ys[i] { Some(s) => s.as_str(), None => continue };
                let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
                let cy = match ctx.scales.y.to_pixel_str(yv) { Some(p) => p, None => continue };
                let px = px + x_offsets[i];
                let cy = cy + y_offsets[i];
                out.line(px, cy - tick_half, px, cy + tick_half, &style);
            }
            return;
        }
    }

    // Quantitative x → rug-style vertical tick at panel baseline.
    let tick_len = ctx.theme.tick_size * 2.0;
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let baseline_y = panel.y + panel.h;
    for (i, xopt) in xs.iter().enumerate() {
        let xv = match xopt { Some(v) if v.is_finite() => *v, _ => continue };
        let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
        let px = px + x_offsets[i];
        let by = baseline_y + y_offsets[i];
        out.line(px, by, px, by - tick_len, &style);
    }
    let _ = y_field(ctx, spec);
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
    fn tick_ordinal_x_emits_horizontal_tick_at_y() {
        // Phase 10c-pre: ordinal x + quantitative y → horizontal tick (boxplot median).
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "median".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        };
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("cat",    arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("median", arrow::datatypes::DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![3.0, 5.0, 7.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        // 3 horizontal ticks — one per (cat, median) row.
        assert_eq!(s.matches("<line ").count(), 3, "expected 3 horizontal tick lines");
        // Lines are horizontal: they must contain both x1 and x2 attributes with different values.
        assert!(s.contains("x1=") && s.contains("x2="), "ticks must have x1 and x2 endpoints");
    }

    #[test]
    fn tick_emits_one_line_per_row() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
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
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<line ").count(), 3);
    }
}
