//! mark_rect: heatmap-style. Requires both x and y to be ordinal/temporal-binned
//! with a known band width. Phase 7 supports the simplest case: ordinal x,
//! ordinal y → one rect per (x, y) row spanning that band-cell.

use crate::layout::Rect;
use crate::render::color::with_opacity;
use crate::render::draw::{col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::{ColorScale, ScaleKind};
use crate::render::svg::{FillStroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
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

    let color_values = color_field(ctx, spec).and_then(|f| col_as_str(ctx.batch, f).ok());
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    for i in 0..xs.len() {
        let xs_v = match &xs[i] { Some(s) => s.as_str(), None => continue };
        let ys_v = match &ys[i] { Some(s) => s.as_str(), None => continue };
        let cx = match ctx.scales.x.to_pixel_str(xs_v) { Some(p) => p, None => continue };
        let cy = match ctx.scales.y.to_pixel_str(ys_v) { Some(p) => p, None => continue };
        let cx = cx + x_offsets[i];
        let cy = cy + y_offsets[i];

        let r = Rect { x: cx - cell_w / 2.0, y: cy - cell_h / 2.0, w: cell_w, h: cell_h };
        let fill = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
            match values[i].as_deref() {
                Some(v) => match scale {
                    ColorScale::Categorical { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                },
                None => ctx.mark_style.fill,
            }
        } else {
            ctx.mark_style.fill
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
