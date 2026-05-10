//! mark_bar: ordinal x → quantitative y. One <rect> per row, anchored at
//! the ordinal x-band, extending from baseline (y=0 mapped) to scale_y(row.y).

use crate::layout::Rect;
use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::{ColorScale, ScaleKind};
use crate::render::svg::{FillStroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return };
    let x_strs = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    if x_strs.len() != ys.len() { return; }

    let panel = ctx.panel.plot_area;
    let baseline_y = panel.y + panel.h;

    let n_categories = match &ctx.scales.x {
        ScaleKind::Ordinal(_) => x_strs.iter().flatten().collect::<std::collections::HashSet<_>>().len().max(1),
        _ => return,
    };
    // Phase 9c — if a position adjustment (Dodge) injected `__pos_x_offset__`
    // / `__pos_y_offset__` columns, narrow each bar to fit a per-group sub-band.
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    let has_pos_offsets = ctx.batch.schema().index_of("__pos_x_offset__").is_ok();
    let n_groups = if has_pos_offsets {
        // Number of distinct non-zero offsets approximates n_groups; clamp to ≥1.
        let mut set: std::collections::HashSet<u64> =
            x_offsets.iter().map(|v| v.to_bits()).collect();
        set.remove(&0.0_f64.to_bits()); // ignore exact zero (single-group fallback)
        if set.is_empty() { 1 } else { set.len() + if x_offsets.iter().any(|v| *v == 0.0) { 1 } else { 0 } }
    } else {
        1
    };
    let bar_width = if has_pos_offsets {
        // Per-category band width is panel.w / n_categories; per-group sub-band
        // is bandwidth / n_groups. We keep the 0.8 fill ratio.
        ((panel.w / n_categories as f64) / n_groups.max(1) as f64) * 0.8
    } else {
        (panel.w / n_categories as f64) * 0.8
    };

    let color_values = color_field(ctx, spec).and_then(|f| col_as_str(ctx.batch, f).ok());

    for i in 0..x_strs.len() {
        let xs = match &x_strs[i] { Some(s) => s.as_str(), None => continue };
        let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
        let cx = match ctx.scales.x.to_pixel_str(xs) { Some(p) => p, None => continue };
        let top_y = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
        let height = (baseline_y - top_y).max(0.0);
        let cx = cx + x_offsets[i];
        let top_y = top_y + y_offsets[i];
        let r = Rect { x: cx - bar_width / 2.0, y: top_y, w: bar_width, h: height };

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
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn bar_emits_four_rects_for_four_categories() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "v".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a","b","c","d"])),
            Arc::new(Float64Array::from(vec![1.0,2.0,3.0,4.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<rect ").count(), 4);
    }

    #[test]
    fn bar_corner_radius_emitted_when_theme_sets_it() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "v".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
        ]).unwrap();
        let mut theme = ThemeInputs::default();
        theme.bar_corner_radius = 3.0;
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert!(out.finish().contains("rx=\"3\""));
    }
}
