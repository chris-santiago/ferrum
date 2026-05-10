//! mark_rule: horizontal or vertical reference line per row. If only y is
//! encoded → horizontal across panel; if only x encoded → vertical.

use crate::render::draw::{col_as_f64, x_field, y_field, DrawCtx};
use crate::render::svg::{Stroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let panel = ctx.panel.plot_area;
    let style = Stroke {
        stroke: ctx.mark_style.fill,
        stroke_width: ctx.mark_style.stroke_width,
        stroke_dash: ctx.mark_style.stroke_dash.clone(),
    };

    if let (Some(yf), None) = (y_field(ctx, spec), x_field(ctx, spec)) {
        let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
        for yv in ys.into_iter().flatten() {
            if !yv.is_finite() { continue; }
            let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
            out.line(panel.x, py, panel.x + panel.w, py, &style);
        }
        return;
    }
    if let (Some(xf), None) = (x_field(ctx, spec), y_field(ctx, spec)) {
        let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
        for xv in xs.into_iter().flatten() {
            if !xv.is_finite() { continue; }
            let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            out.line(px, panel.y, px, panel.y + panel.h, &style);
        }
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
    fn y_only_rule_emits_horizontal_lines() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: None,
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let mut spec_for_scales = spec.clone();
        spec_for_scales.encoding.x = Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() });
        let (scales, _) = resolve_scales(&spec_for_scales, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Rule);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<line ").count(), 2);
    }
}
