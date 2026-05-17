//! mark_rect: three paths —
//!   ordinal x × ordinal y → heatmap cell (original Phase 7 path);
//!   ordinal x + quantitative y + y2 → vertical band rect (boxplot box body,
//!   Phase 10c-pre);
//!   quantitative x + x2 + y + y2 → free-floating rect with explicit
//!   pixel bounds (Phase 10f, silhouette + decision-boundary). Dispatch:
//!   both x2 & y2 present and both axes quantitative → quantitative-range
//!   path; only y2 present → ordinal-range; else heatmap.

use crate::layout::Rect;
use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx, MetadataColumns};
use crate::render::scale_resolve::{ColorScale, ScaleKind};

fn count_distinct(values: &[Option<String>]) -> usize {
    let mut seen = std::collections::HashSet::<&str>::new();
    for v in values.iter().flatten() { seen.insert(v); }
    seen.len()
}

// ── Scene-graph build path (11a) ────────────────────────────────────

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    let both_ranges = ctx.spec.encoding.x2.is_some() && ctx.spec.encoding.y2.is_some();
    let both_quant = matches!(
        (&ctx.scales.x, &ctx.scales.y),
        (
            ScaleKind::Linear(_) | ScaleKind::Log(_) | ScaleKind::Symlog(_) | ScaleKind::Pow(_),
            ScaleKind::Linear(_) | ScaleKind::Log(_) | ScaleKind::Symlog(_) | ScaleKind::Pow(_),
        )
    );
    if both_ranges && both_quant {
        build_quantitative_range(ctx)
    } else if ctx.spec.encoding.y2.is_some() || ctx.spec.encoding.x2.is_some() {
        // y2.is_some(): normal orientation (ordinal x + quantitative y/y2).
        // x2.is_some() without y2: CoordFlip case (ordinal y + quantitative x/x2).
        build_ordinal_range(ctx)
    } else {
        build_heatmap(ctx)
    }
}

fn empty_result() -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::MarkBuildResult;
    use ferrum_scene::MarkBatchKind;
    MarkBuildResult {
        kind: MarkBatchKind::Rect,
        nodes: vec![],
        data_indices: Some(vec![]),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    }
}

fn build_quantitative_range(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_fill_stroke};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let x2f = match spec.encoding.x2.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f, None => return empty_result(),
    };
    let y2f = match spec.encoding.y2.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f, None => return empty_result(),
    };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
    let x2s = match col_as_f64(ctx.batch, x2f) { Ok(v) => v, Err(_) => return empty_result() };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
    let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return empty_result() };
    let n = xs.len();
    if x2s.len() != n || ys.len() != n || y2s.len() != n { return empty_result(); }

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
    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

    for i in 0..n {
        let x_lo = match xs[i] { Some(v) if v.is_finite() => v, _ => continue };
        let x_hi = match x2s[i] { Some(v) if v.is_finite() => v, _ => continue };
        let y_lo = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
        let y_hi = match y2s[i] { Some(v) if v.is_finite() => v, _ => continue };
        let px_lo = match ctx.scales.x.to_pixel_f64(x_lo) { Some(p) => p, None => continue };
        let px_hi = match ctx.scales.x.to_pixel_f64(x_hi) { Some(p) => p, None => continue };
        let py_lo = match ctx.scales.y.to_pixel_f64(y_lo) { Some(p) => p, None => continue };
        let py_hi = match ctx.scales.y.to_pixel_f64(y_hi) { Some(p) => p, None => continue };
        let px_left = px_lo.min(px_hi) + x_offsets[i];
        let py_top = py_lo.min(py_hi) + y_offsets[i];
        let w = (px_hi - px_lo).abs().max(0.5);
        let h = (py_hi - py_lo).abs().max(0.5);

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

        nodes.push(SceneNode::Rect {
            x: px_left,
            y: py_top,
            w,
            h,
            style: to_scene_fill_stroke(
                Some(fill),
                ctx.mark_style.stroke,
                ctx.mark_style.stroke_width,
                ctx.mark_style.opacity,
                ctx.mark_style.stroke_dash.as_deref(),
            ),
            corner_radius: ctx.mark_style.corner_radius,
        });
        indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Rect,
        nodes,
        data_indices: Some(indices),
        tooltips,
        hrefs,
        descriptions,    }
}

fn build_ordinal_range(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_fill_stroke};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return empty_result() };

    // Detect orientation: normal (x-ordinal + y2) vs. CoordFlip (y-ordinal + x2).
    let x_is_ordinal = matches!(ctx.scales.x, ScaleKind::Ordinal(_));
    let y_is_ordinal = matches!(ctx.scales.y, ScaleKind::Ordinal(_));

    let panel = ctx.panel.plot_area;

    let cfield = color_field(ctx, spec);
    let color_strings: Option<Vec<Option<String>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Categorical { .. }), Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

    if x_is_ordinal {
        // Normal orientation: x is categorical band, y and y2 are quantitative extents.
        let y2f = match spec.encoding.y2.as_ref().map(|e| e.field.as_str()) {
            Some(f) => f, None => return empty_result(),
        };
        let n_categories = {
            let xs_probe = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
            count_distinct(&xs_probe).max(1)
        };
        let xs = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
        let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
        let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return empty_result() };
        if xs.len() != ys.len() || y2s.len() != ys.len() { return empty_result(); }
        let box_w = (panel.w / n_categories as f64) * ctx.mark_style.band_size.unwrap_or(0.6);

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

            nodes.push(SceneNode::Rect {
                x: cx - box_w / 2.0,
                y: rect_top,
                w: box_w,
                h: rect_h,
                style: to_scene_fill_stroke(
                    Some(fill),
                    ctx.mark_style.stroke,
                    ctx.mark_style.stroke_width,
                    ctx.mark_style.opacity,
                    ctx.mark_style.stroke_dash.as_deref(),
                ),
                corner_radius: ctx.mark_style.corner_radius,
            });
            indices.push(i);
        }
    } else if y_is_ordinal {
        // CoordFlip orientation: y is categorical band, x and x2 are quantitative extents.
        let x2f = match spec.encoding.x2.as_ref().map(|e| e.field.as_str()) {
            Some(f) => f, None => return empty_result(),
        };
        let n_categories = {
            let ys_probe = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
            count_distinct(&ys_probe).max(1)
        };
        let ys = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
        let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
        let x2s = match col_as_f64(ctx.batch, x2f) { Ok(v) => v, Err(_) => return empty_result() };
        if ys.len() != xs.len() || x2s.len() != xs.len() { return empty_result(); }
        let box_h = (panel.h / n_categories as f64) * ctx.mark_style.band_size.unwrap_or(0.6);

        for i in 0..ys.len() {
            let yv = match &ys[i] { Some(s) => s.as_str(), None => continue };
            let xv = match xs[i] { Some(v) if v.is_finite() => v, _ => continue };
            let x2v = match x2s[i] { Some(v) if v.is_finite() => v, _ => continue };
            let cy = match ctx.scales.y.to_pixel_str(yv) { Some(p) => p, None => continue };
            let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            let px2 = match ctx.scales.x.to_pixel_f64(x2v) { Some(p) => p, None => continue };
            let cy = cy + y_offsets[i];
            let rect_left = px.min(px2) + x_offsets[i];
            let rect_w = (px - px2).abs().max(1.0);

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

            nodes.push(SceneNode::Rect {
                x: rect_left,
                y: cy - box_h / 2.0,
                w: rect_w,
                h: box_h,
                style: to_scene_fill_stroke(
                    Some(fill),
                    ctx.mark_style.stroke,
                    ctx.mark_style.stroke_width,
                    ctx.mark_style.opacity,
                    ctx.mark_style.stroke_dash.as_deref(),
                ),
                corner_radius: ctx.mark_style.corner_radius,
            });
            indices.push(i);
        }
    } else {
        return empty_result();
    }

    MarkBuildResult {
        kind: MarkBatchKind::Rect,
        nodes,
        data_indices: Some(indices),
        tooltips,
        hrefs,
        descriptions,    }
}

fn build_heatmap(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_fill_stroke, to_scene_text_style};
    use crate::render::format::format_numeric;
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return empty_result(),
    };
    let xs = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
    let ys = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
    if xs.len() != ys.len() { return empty_result(); }

    let panel = ctx.panel.plot_area;
    let n_x = match &ctx.scales.x { ScaleKind::Ordinal(_) => count_distinct(&xs).max(1), _ => return empty_result() };
    let n_y = match &ctx.scales.y { ScaleKind::Ordinal(_) => count_distinct(&ys).max(1), _ => return empty_result() };
    let cell_w = panel.w / n_x as f64;
    let cell_h = panel.h / n_y as f64;

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
    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    // Per-row encoding channels: opacity, fill_opacity, stroke_width.
    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let fill_opacity_values: Option<Vec<Option<f64>>> = spec.encoding.fill_opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let stroke_width_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_width
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    // Optional text annotation channel for heatmap cells.
    let text_enc = spec.encoding.text.as_ref();
    let text_field = text_enc.map(|e| e.field.as_str());
    let text_values: Option<Vec<Option<String>>> = text_field.and_then(|f| {
        col_as_str(ctx.batch, f).ok().or_else(|| {
            col_as_f64(ctx.batch, f).ok().map(|nums| {
                nums.into_iter()
                    .map(|v| v.map(|n| format_numeric(n)))
                    .collect()
            })
        })
    });

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

    for i in 0..xs.len() {
        let xs_v = match &xs[i] { Some(s) => s.as_str(), None => continue };
        let ys_v = match &ys[i] { Some(s) => s.as_str(), None => continue };
        let cx = match ctx.scales.x.to_pixel_str(xs_v) { Some(p) => p, None => continue };
        let cy = match ctx.scales.y.to_pixel_str(ys_v) { Some(p) => p, None => continue };
        let cx = cx + x_offsets[i];
        let cy = cy + y_offsets[i];

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
        // Resolve per-row opacity (through scale if present).
        let row_opacity = if let (Some(values), Some(scale)) = (&opacity_values, &ctx.scales.opacity) {
            match values[i].and_then(|v| scale.inner.to_pixel_f64(v)) {
                Some(op) => op,
                None => ctx.mark_style.opacity,
            }
        } else {
            ctx.mark_style.opacity
        };

        let fill = with_opacity(fill, row_opacity);

        let row_fill_opacity = fill_opacity_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(1.0);

        let row_stroke_width = stroke_width_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| *v >= 0.0 && v.is_finite())
            .unwrap_or(ctx.mark_style.stroke_width);

        // When stroke_width encoding produces a positive value but no explicit
        // stroke color exists, use the fill color as the stroke so the width is
        // visible in SVG (stroke-width is only emitted when stroke is Some).
        let effective_stroke = if row_stroke_width > 0.0 && ctx.mark_style.stroke.is_none() && stroke_width_values.is_some() {
            Some(fill)
        } else {
            ctx.mark_style.stroke
        };

        let mut style = to_scene_fill_stroke(
            Some(fill),
            effective_stroke,
            row_stroke_width,
            row_opacity,
            ctx.mark_style.stroke_dash.as_deref(),
        );
        style.fill_opacity = row_fill_opacity;

        nodes.push(SceneNode::Rect {
            x: cx - cell_w / 2.0,
            y: cy - cell_h / 2.0,
            w: cell_w,
            h: cell_h,
            style,
            corner_radius: ctx.mark_style.corner_radius,
        });
        indices.push(i);

        // Emit text annotation at cell center when text encoding is present.
        if let Some(ref texts) = text_values {
            if let Some(Some(content)) = texts.get(i) {
                if !content.is_empty() {
                    let text_color = ctx.theme.font_color;
                    let font_size = ctx.mark_style.font_size.unwrap_or(11.0);
                    nodes.push(SceneNode::Text {
                        x: cx,
                        y: cy,
                        content: content.clone(),
                        style: to_scene_text_style(
                            text_color,
                            font_size,
                            crate::layout::TextAnchor::Middle,
                            0.0,
                            &ctx.theme.font_family,
                            None,
                            Some("central"),
                            1.0,
                        ),
                    });
                }
            }
        }
    }

    MarkBuildResult {
        kind: MarkBatchKind::Rect,
        nodes,
        data_indices: Some(indices),
        tooltips,
        hrefs,
        descriptions,    }
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
    use ferrum_scene::SceneNode;
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
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
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
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count(), 2, "expected 2 band rects");
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
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
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
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count(), 4);
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
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
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
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count(), 4);
        // Distinct values 0/2/5/10 must produce distinct fill colors.
        let mut fills: std::collections::HashSet<String> = std::collections::HashSet::new();
        for node in &result.nodes {
            if let SceneNode::Rect { style, .. } = node {
                if let Some(c) = &style.fill {
                    fills.insert(format!("{},{},{},{}", c.r, c.g, c.b, c.a));
                }
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
    fn rect_quant_range_draws_per_row_with_explicit_bounds() {
        // Phase 10f: quantitative x + x2 + y + y2 → free-floating rect per row.
        // Verifies the silhouette / decision-boundary path renders.
        use arrow::array::Float64Array;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",  DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y",  DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![3.0, 4.0, 5.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(
            result.nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count(),
            3,
            "expected 3 quant-range rects (one per row)",
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
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
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
        let result = super::build(&ctx);
        assert!(result.nodes.iter().all(|n| !matches!(n, SceneNode::Rect { .. })));
    }
}
