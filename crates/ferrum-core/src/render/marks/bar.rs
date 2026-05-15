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

/// Per-row stroke encoding column vectors loaded from a batch.
struct StrokeChannels {
    opacity: Option<Vec<Option<f64>>>,
    width: Option<Vec<Option<f64>>>,
    dash: Option<Vec<Option<f64>>>,
    angle: Option<Vec<Option<f64>>>,
}

impl StrokeChannels {
    fn load(ctx: &DrawCtx) -> Self {
        Self {
            opacity: ctx.spec.encoding.stroke_opacity.as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
            width: ctx.spec.encoding.stroke_width.as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
            dash: ctx.spec.encoding.stroke_dash.as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
            angle: ctx.spec.encoding.angle.as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
        }
    }

    /// Build a `FillStroke` for row `i`, overriding `base_*` defaults with any
    /// per-row column values.  `corner_radius` is passed through unchanged.
    fn row_fill_stroke(
        &self,
        fill: Option<ferrum_scene::Color>,
        stroke: Option<ferrum_scene::Color>,
        base_sw: f64,
        opacity: f64,
        base_dash: Option<&[f64]>,
        corner_radius: f64,
        i: usize,
    ) -> (ferrum_scene::FillStroke, f64) {
        let stroke_opacity = self.opacity.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(1.0);

        let stroke_width = self.width.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .filter(|v| *v >= 0.0 && v.is_finite())
            .unwrap_or(base_sw);

        let dash_vec: Option<Vec<f64>> = self.dash.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .filter(|v| v.is_finite())
            .and_then(|idx| {
                let idx = (idx.round() as i64).clamp(0, 3);
                match idx {
                    1 => Some(vec![6.0, 3.0]),
                    2 => Some(vec![2.0, 3.0]),
                    3 => Some(vec![6.0, 3.0, 2.0, 3.0]),
                    _ => None,
                }
            });
        let effective_dash = dash_vec.as_deref().or(base_dash).map(|d| d.to_vec());

        let angle = self.angle.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);

        let fs = ferrum_scene::FillStroke {
            fill,
            stroke,
            stroke_width,
            opacity,
            stroke_dash: effective_dash,
            stroke_opacity,
            angle,
        };
        (fs, corner_radius)
    }
}

// ── Scene-graph build path (11a) ────────────────────────────────────

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    let has_x2 = ctx.spec.encoding.x2.is_some();
    let has_y2 = ctx.spec.encoding.y2.is_some();
    match (&ctx.scales.x, &ctx.scales.y) {
        (ScaleKind::Ordinal(_), _) => build_ordinal(ctx),
        (_, ScaleKind::Ordinal(_)) => build_ordinal_y(ctx),
        (_, _) if has_y2 && !has_x2 => build_quantitative_horizontal(ctx),
        (ScaleKind::Linear(_) | ScaleKind::Log(_) | ScaleKind::Symlog(_), _) => {
            build_quantitative(ctx)
        }
        _ => empty_result(),
    }
}

fn empty_result() -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::MarkBuildResult;
    use ferrum_scene::MarkBatchKind;
    MarkBuildResult {
        kind: MarkBatchKind::Bar,
        nodes: vec![],
        data_indices: Some(vec![]),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    }
}

fn build_ordinal(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_color};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let x_strs = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
    if x_strs.len() != ys.len() { return empty_result(); }

    let y_bases: Option<Vec<Option<f64>>> =
        col_as_f64(ctx.batch, "__stack_y_base__").ok();

    let panel = ctx.panel.plot_area;
    let baseline_y = panel.y + panel.h;

    let n_categories = x_strs.iter().flatten().collect::<std::collections::HashSet<_>>().len().max(1);

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
    let sc = StrokeChannels::load(ctx);
    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

    for i in 0..x_strs.len() {
        let xs = match &x_strs[i] { Some(s) => s.as_str(), None => continue };
        let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
        let cx = match ctx.scales.x.to_pixel_str(xs) { Some(p) => p, None => continue };
        let top_y = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
        let bottom_y = match y_bases.as_ref().and_then(|v| v[i]) {
            Some(b) if b.is_finite() => {
                ctx.scales.y.to_pixel_f64(b).unwrap_or(baseline_y)
            }
            _ => baseline_y,
        };
        let height = (bottom_y - top_y).max(0.0);
        let cx = cx + x_offsets[i];
        let top_y = top_y + y_offsets[i];

        let fill_color = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
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
        let fill = with_opacity(fill_color, ctx.mark_style.opacity);

        let stroke_sc = ctx.mark_style.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let (style, cr) = sc.row_fill_stroke(
            Some(fill_sc), stroke_sc,
            ctx.mark_style.stroke_width, ctx.mark_style.opacity,
            ctx.mark_style.stroke_dash.as_deref(), ctx.mark_style.corner_radius, i,
        );

        nodes.push(SceneNode::Rect {
            x: cx - bar_width / 2.0,
            y: top_y,
            w: bar_width,
            h: height,
            style,
            corner_radius: cr,
        });
        indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Bar,
        nodes,
        data_indices: Some(indices),
        tooltips,
        hrefs,
        descriptions,    }
}

fn build_ordinal_y(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_color};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let y_strs = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
    if y_strs.len() != xs.len() { return empty_result(); }

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
    let sc = StrokeChannels::load(ctx);
    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

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

        let fill_color = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
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
        let fill = with_opacity(fill_color, ctx.mark_style.opacity);

        let stroke_sc = ctx.mark_style.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let (style, cr) = sc.row_fill_stroke(
            Some(fill_sc), stroke_sc,
            ctx.mark_style.stroke_width, ctx.mark_style.opacity,
            ctx.mark_style.stroke_dash.as_deref(), ctx.mark_style.corner_radius, i,
        );

        nodes.push(SceneNode::Rect {
            x: left_x,
            y: cy - bar_height / 2.0,
            w: width,
            h: bar_height,
            style,
            corner_radius: cr,
        });
        indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Bar,
        nodes,
        data_indices: Some(indices),
        tooltips,
        hrefs,
        descriptions,    }
}

fn build_quantitative(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_color};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let x2f = match spec.encoding.x2.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f, None => return empty_result(),
    };

    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
    let x2s = match col_as_f64(ctx.batch, x2f) { Ok(v) => v, Err(_) => return empty_result() };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
    if xs.len() != ys.len() || x2s.len() != ys.len() { return empty_result(); }

    let panel = ctx.panel.plot_area;
    let baseline_y = panel.y + panel.h;

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let color_values = color_field(ctx, spec).and_then(|f| col_as_str(ctx.batch, f).ok());
    let sc = StrokeChannels::load(ctx);
    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

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

        let fill_color = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
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
        let fill = with_opacity(fill_color, ctx.mark_style.opacity);

        let stroke_sc = ctx.mark_style.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let (style, cr) = sc.row_fill_stroke(
            Some(fill_sc), stroke_sc,
            ctx.mark_style.stroke_width, ctx.mark_style.opacity,
            ctx.mark_style.stroke_dash.as_deref(), ctx.mark_style.corner_radius, i,
        );

        nodes.push(SceneNode::Rect {
            x: px_left.min(px_right),
            y: top_y,
            w: width,
            h: height,
            style,
            corner_radius: cr,
        });
        indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Bar,
        nodes,
        data_indices: Some(indices),
        tooltips,
        hrefs,
        descriptions,    }
}

fn build_quantitative_horizontal(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_color};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let y2f = match spec.encoding.y2.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f, None => return empty_result(),
    };

    let xs  = match col_as_f64(ctx.batch, xf)  { Ok(v) => v, Err(_) => return empty_result() };
    let ys  = match col_as_f64(ctx.batch, yf)  { Ok(v) => v, Err(_) => return empty_result() };
    let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return empty_result() };
    if xs.len() != ys.len() || y2s.len() != ys.len() { return empty_result(); }

    let panel = ctx.panel.plot_area;
    let baseline_x = panel.x;

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let color_values = color_field(ctx, spec).and_then(|f| col_as_str(ctx.batch, f).ok());
    let sc = StrokeChannels::load(ctx);
    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

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

        let fill_color = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
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
        let fill = with_opacity(fill_color, ctx.mark_style.opacity);

        let stroke_sc = ctx.mark_style.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let (style, cr) = sc.row_fill_stroke(
            Some(fill_sc), stroke_sc,
            ctx.mark_style.stroke_width, ctx.mark_style.opacity,
            ctx.mark_style.stroke_dash.as_deref(), ctx.mark_style.corner_radius, i,
        );

        nodes.push(SceneNode::Rect {
            x: baseline_x,
            y: py_top.min(py_bottom),
            w: width,
            h: height,
            style,
            corner_radius: cr,
        });
        indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Bar,
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
        selections: Vec::new(), conditionals: Vec::new(),
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
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count(), 3, "expected 3 histogram bars");
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
        selections: Vec::new(), conditionals: Vec::new(),
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
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count(), 4);
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
        selections: Vec::new(), conditionals: Vec::new(),
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
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count(), 3, "expected 3 horizontal bars");
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
        selections: Vec::new(), conditionals: Vec::new(),
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
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count(), 3, "expected 3 ranged-horizontal bars");
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
        selections: Vec::new(), conditionals: Vec::new(),
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
        let result = super::build(&ctx);
        let has_corner_radius = result.nodes.iter().any(|n| {
            if let SceneNode::Rect { corner_radius, .. } = n {
                (*corner_radius - 3.0).abs() < f64::EPSILON
            } else {
                false
            }
        });
        assert!(has_corner_radius, "expected at least one rect with corner_radius == 3.0");
    }
}
