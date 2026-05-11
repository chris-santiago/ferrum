//! mark_text: minimal Phase 7 stub — renders format_numeric(y) at (scale_x(x), scale_y(y)).

use crate::layout::TextAnchor;
use crate::render::draw::{col_as_f64, x_field, y_field, DrawCtx};
use crate::render::format::format_numeric;
use crate::render::svg::{SvgBuffer, TextStyle};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return,
    };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    if xs.len() != ys.len() { return; }

    let style = TextStyle {
        fill: ctx.theme.font_color,
        font_size: ctx.theme.label_font_size,
        anchor: TextAnchor::Middle,
        angle: 0.0,
    };

    for i in 0..xs.len() {
        let (xv, yv) = match (xs[i], ys[i]) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a, b), _ => continue,
        };
        let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
        let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
        let label = format_numeric(yv);
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
}
