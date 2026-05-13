//! mark_bar: three paths —
//!   ordinal x → quantitative y: one <rect> per row anchored at x-band center.
//!   quantitative x + x2 → quantitative y: bin rect from x_pixel to x2_pixel
//!   (histogram path added Phase 10c-pre).
//!   quantitative x → ordinal y: horizontal bar per row from panel-left to
//!   x_pixel (Phase 10d-pre, feature-importance chart).

use crate::layout::Rect;
use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx, MetadataColumns};
use crate::render::scale_resolve::{ColorScale, ScaleKind};
use crate::render::svg::{FillStroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    // Encoding-presence check picks between the four quantitative paths:
    // - x + x2 + y       → vertical histogram (draw_quantitative)
    // - y + y2 + x       → horizontal histogram (draw_quantitative_horizontal),
    //                       used by JointChart's right marginal with
    //                       mark_histogram(orientation="horizontal").
    let has_x2 = ctx.spec.encoding.x2.is_some();
    let has_y2 = ctx.spec.encoding.y2.is_some();
    match (&ctx.scales.x, &ctx.scales.y) {
        (ScaleKind::Ordinal(_), _) => draw_ordinal(ctx, out),
        (_, ScaleKind::Ordinal(_)) => draw_ordinal_y(ctx, out),
        (_, _) if has_y2 && !has_x2 => draw_quantitative_horizontal(ctx, out),
        (ScaleKind::Linear(_) | ScaleKind::Log(_) | ScaleKind::Symlog(_), _) => {
            draw_quantitative(ctx, out)
        }
        _ => {}
    }
}

/// Ordinal-x bar path: categorical bar chart.
fn draw_ordinal(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return };
    let x_strs = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    if x_strs.len() != ys.len() { return; }

    // Stacked-bar segments carry their lower bound in `__stack_y_base__`
    // (injected by `position::apply_stack`). When absent, every segment is
    // anchored at the plot baseline.
    let y_bases: Option<Vec<Option<f64>>> =
        col_as_f64(ctx.batch, "__stack_y_base__").ok();

    let panel = ctx.panel.plot_area;
    let baseline_y = panel.y + panel.h;

    let n_categories = x_strs.iter().flatten().collect::<std::collections::HashSet<_>>().len().max(1);

    // Phase 9c — if a position adjustment (Dodge) injected `__pos_x_offset__`
    // / `__pos_y_offset__` columns, narrow each bar to fit a per-group sub-band.
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    let has_pos_offsets = ctx.batch.schema().index_of("__pos_x_offset__").is_ok();
    let n_groups = if has_pos_offsets {
        let mut set: std::collections::HashSet<u64> =
            x_offsets.iter().map(|v| v.to_bits()).collect();
        set.remove(&0.0_f64.to_bits());
        if set.is_empty() { 1 } else { set.len() + if x_offsets.iter().any(|v| *v == 0.0) { 1 } else { 0 } }
    } else {
        1
    };
    let bar_width = if has_pos_offsets {
        ((panel.w / n_categories as f64) / n_groups.max(1) as f64) * 0.8
    } else {
        (panel.w / n_categories as f64) * 0.8
    };

    let color_values = color_field(ctx, spec).and_then(|f| col_as_str(ctx.batch, f).ok());
    let meta = MetadataColumns::from_ctx(ctx);

    for i in 0..x_strs.len() {
        let xs = match &x_strs[i] { Some(s) => s.as_str(), None => continue };
        let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
        let cx = match ctx.scales.x.to_pixel_str(xs) { Some(p) => p, None => continue };
        let top_y = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
        // Segment bottom comes from __stack_y_base__ if present; otherwise
        // the bar grows from the plot baseline (single-bar / unstacked path).
        let bottom_y = match y_bases.as_ref().and_then(|v| v[i]) {
            Some(b) if b.is_finite() => {
                ctx.scales.y.to_pixel_f64(b).unwrap_or(baseline_y)
            }
            _ => baseline_y,
        };
        let height = (bottom_y - top_y).max(0.0);
        let cx = cx + x_offsets[i];
        let top_y = top_y + y_offsets[i];
        let r = Rect { x: cx - bar_width / 2.0, y: top_y, w: bar_width, h: height };

        let fill = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
            match values[i].as_deref() {
                Some(v) => match scale {
                    ColorScale::Categorical { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                    ColorScale::Continuous { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                },
                None => ctx.mark_style.fill,
            }
        } else {
            ctx.mark_style.fill
        };
        let fill = with_opacity(fill, ctx.mark_style.opacity);

        let wrapped = meta.open(i, out);
        out.rect(r, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        }, Some(ctx.mark_style.corner_radius));
        if wrapped { meta.close(i, out); }
    }
}

/// Ordinal-y bar path: horizontal categorical bar chart. Two modes —
///   `x` only: bars grow rightward from the left panel edge to
///     `to_pixel_f64(value)`. Used by `mark_importance(orient="horizontal")`.
///   `x` + `x2`: ranged horizontal bar from `to_pixel_f64(x)` to
///     `to_pixel_f64(x2)`. Used by Phase 10d `mark_shap_waterfall` to draw
///     each per-feature contribution as a segment from the cumulative
///     baseline to the new cumulative value.
/// Stacking (`__stack_x_base__`) is not mirrored from the vertical path —
/// no horizontal-stacked-bar consumer yet.
fn draw_ordinal_y(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return };
    let y_strs = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    if y_strs.len() != xs.len() { return; }

    let x2f_opt = spec.encoding.x2.as_ref().map(|e| e.field.as_str());
    let x2s_opt: Option<Vec<Option<f64>>> = x2f_opt
        .and_then(|f| col_as_f64(ctx.batch, f).ok());

    let panel = ctx.panel.plot_area;
    let baseline_x = panel.x;

    let n_categories = y_strs
        .iter()
        .flatten()
        .collect::<std::collections::HashSet<_>>()
        .len()
        .max(1);

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    let bar_height = (panel.h / n_categories as f64) * 0.8;

    let color_values = color_field(ctx, spec).and_then(|f| col_as_str(ctx.batch, f).ok());
    let meta = MetadataColumns::from_ctx(ctx);

    for i in 0..y_strs.len() {
        let ys = match &y_strs[i] { Some(s) => s.as_str(), None => continue };
        let xv = match xs[i] { Some(v) if v.is_finite() => v, _ => continue };
        let cy = match ctx.scales.y.to_pixel_str(ys) { Some(p) => p, None => continue };
        let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };

        let cy = cy + y_offsets[i];
        let px = px + x_offsets[i];

        let (left_x, width) = if let Some(x2s) = &x2s_opt {
            let x2v = match x2s[i] { Some(v) if v.is_finite() => v, _ => continue };
            let px2 = match ctx.scales.x.to_pixel_f64(x2v) { Some(p) => p, None => continue };
            let px2 = px2 + x_offsets[i];
            (px.min(px2), (px - px2).abs())
        } else {
            (baseline_x, (px - baseline_x).max(0.0))
        };

        let r = Rect {
            x: left_x,
            y: cy - bar_height / 2.0,
            w: width,
            h: bar_height,
        };

        let fill = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
            match values[i].as_deref() {
                Some(v) => match scale {
                    ColorScale::Categorical { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                    ColorScale::Continuous { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                },
                None => ctx.mark_style.fill,
            }
        } else {
            ctx.mark_style.fill
        };
        let fill = with_opacity(fill, ctx.mark_style.opacity);

        let wrapped = meta.open(i, out);
        out.rect(r, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        }, Some(ctx.mark_style.corner_radius));
        if wrapped { meta.close(i, out); }
    }
}

/// Quantitative-x bar path: histogram / bin chart.
/// Requires x2 encoding (bin end). x and x2 are both f64; y is f64 count/value.
fn draw_quantitative(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return };
    let x2f = match spec.encoding.x2.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f, None => return,
    };

    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let x2s = match col_as_f64(ctx.batch, x2f) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    if xs.len() != ys.len() || x2s.len() != ys.len() { return; }

    let panel = ctx.panel.plot_area;
    let baseline_y = panel.y + panel.h;

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let color_values = color_field(ctx, spec).and_then(|f| col_as_str(ctx.batch, f).ok());
    let meta = MetadataColumns::from_ctx(ctx);

    for i in 0..xs.len() {
        let xv = match xs[i] { Some(v) if v.is_finite() => v, _ => continue };
        let x2v = match x2s[i] { Some(v) if v.is_finite() => v, _ => continue };
        let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
        let px_left = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
        let px_right = match ctx.scales.x.to_pixel_f64(x2v) { Some(p) => p, None => continue };
        let top_y = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };

        let px_left = px_left + x_offsets[i];
        let top_y = top_y + y_offsets[i];
        let width = (px_right - px_left).abs().max(1.0);
        let height = (baseline_y - top_y).max(0.0);
        let r = Rect { x: px_left.min(px_right), y: top_y, w: width, h: height };

        let fill = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
            match values[i].as_deref() {
                Some(v) => match scale {
                    ColorScale::Categorical { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                    ColorScale::Continuous { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                },
                None => ctx.mark_style.fill,
            }
        } else {
            ctx.mark_style.fill
        };
        let fill = with_opacity(fill, ctx.mark_style.opacity);

        let wrapped = meta.open(i, out);
        out.rect(r, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        }, Some(ctx.mark_style.corner_radius));
        if wrapped { meta.close(i, out); }
    }
}

/// Quantitative-y horizontal histogram path: y is `bin_start`, y2 is `bin_end`,
/// x is the count/density value. Mirror of `draw_quantitative` with axes
/// swapped — bars grow rightward from the left panel edge, one stacked
/// vertically per bin. Used by JointChart's right marginal so the binned
/// data dimension stays on the marginal's y-axis (shared with the centre
/// panel's y-scale).
fn draw_quantitative_horizontal(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return };
    let y2f = match spec.encoding.y2.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f, None => return,
    };

    let xs  = match col_as_f64(ctx.batch, xf)  { Ok(v) => v, Err(_) => return };
    let ys  = match col_as_f64(ctx.batch, yf)  { Ok(v) => v, Err(_) => return };
    let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return };
    if xs.len() != ys.len() || y2s.len() != ys.len() { return; }

    let panel = ctx.panel.plot_area;
    let baseline_x = panel.x;

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let color_values = color_field(ctx, spec).and_then(|f| col_as_str(ctx.batch, f).ok());
    let meta = MetadataColumns::from_ctx(ctx);

    for i in 0..xs.len() {
        let xv  = match xs[i]  { Some(v) if v.is_finite() => v, _ => continue };
        let yv  = match ys[i]  { Some(v) if v.is_finite() => v, _ => continue };
        let y2v = match y2s[i] { Some(v) if v.is_finite() => v, _ => continue };
        let px_right  = match ctx.scales.x.to_pixel_f64(xv)  { Some(p) => p, None => continue };
        let py_top    = match ctx.scales.y.to_pixel_f64(yv)  { Some(p) => p, None => continue };
        let py_bottom = match ctx.scales.y.to_pixel_f64(y2v) { Some(p) => p, None => continue };

        let px_right = px_right + x_offsets[i];
        let py_top   = py_top   + y_offsets[i];
        let py_bottom = py_bottom + y_offsets[i];
        let width  = (px_right - baseline_x).max(0.0);
        let height = (py_top - py_bottom).abs().max(1.0);
        let r = Rect {
            x: baseline_x,
            y: py_top.min(py_bottom),
            w: width,
            h: height,
        };

        let fill = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
            match values[i].as_deref() {
                Some(v) => match scale {
                    ColorScale::Categorical { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                    ColorScale::Continuous { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                },
                None => ctx.mark_style.fill,
            }
        } else {
            ctx.mark_style.fill
        };
        let fill = with_opacity(fill, ctx.mark_style.opacity);

        let wrapped = meta.open(i, out);
        out.rect(r, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        }, Some(ctx.mark_style.corner_radius));
        if wrapped { meta.close(i, out); }
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
    fn bar_quantitative_histogram_emits_bin_rects() {
        // Phase 10c-pre: quantitative x + x2 + y → histogram bar per bin.
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "bin_start".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "count".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "bin_end".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("bin_start", DataType::Float64, false),
            Field::new("bin_end",   DataType::Float64, false),
            Field::new("count",     DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![5.0, 10.0, 3.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<rect ").count(), 3, "expected 3 histogram bars, got: {s}");
    }

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
        title: None,
        axis_x: None, axis_y: None,
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
    fn bar_ordinal_y_emits_horizontal_rects() {
        // Phase 10d-pre: quantitative x + ordinal y → horizontal bars.
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "v".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<rect ").count(), 3, "expected 3 horizontal bars, got: {s}");
    }

    #[test]
    fn bar_ordinal_y_with_x2_emits_ranged_horizontal_rects() {
        // Phase 10d (Task 22-pre): quantitative x + x2 + ordinal y →
        // ranged horizontal bars (each row spans from x to x2 horizontally).
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x0".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "x1".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x0", DataType::Float64, false),
            Field::new("x1", DataType::Float64, false),
            Field::new("g",  DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<rect ").count(), 3, "expected 3 ranged-horizontal bars, got: {s}");
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
        title: None,
        axis_x: None, axis_y: None,
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
