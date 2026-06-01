//! mark_bar: three paths —
//!   ordinal x → quantitative y: one <rect> per row anchored at x-band center.
//!   quantitative x + x2 → quantitative y: bin rect from x_pixel to x2_pixel
//!   (histogram path added Phase 10c-pre).
//!   quantitative x → ordinal y: horizontal bar per row from panel-left to
//!   x_pixel (Phase 10d-pre, feature-importance chart).

#[cfg(test)]
use crate::layout::Rect;
use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_ordinal_category_str, col_as_str, color_field, resolve_fill_color, resolve_stroke_dash, x_field, y_field, DrawCtx, MetadataColumns};
use crate::render::scale_resolve::ScaleKind;

/// Load the per-row color-encoding columns for fill resolution, mirroring the
/// point renderer: the categorical string column is read for `Categorical`
/// (and scale-less) charts, the numeric column for `Continuous` charts. The
/// continuous branch reads `col_as_f64` so `resolve_fill_color` can sample via
/// `lookup_f64` without an `f64 → String → f64` round-trip.
type ColorColumns = (Option<Vec<Option<String>>>, Option<Vec<Option<f64>>>);

fn load_color_columns(ctx: &DrawCtx) -> ColorColumns {
    use crate::render::scale_resolve::ColorScale;
    let field = color_field(ctx, ctx.spec);
    let cat = match (&ctx.scales.color, field) {
        (Some(ColorScale::Categorical { .. }), Some(f)) => col_as_str(ctx.batch, f).ok(),
        (None, Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let num = match (&ctx.scales.color, field) {
        (Some(ColorScale::Continuous { .. }), Some(f)) => col_as_f64(ctx.batch, f).ok(),
        _ => None,
    };
    (cat, num)
}

#[inline]
fn row_cat(col: &Option<Vec<Option<String>>>, i: usize) -> Option<&str> {
    col.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref())
}

#[inline]
fn row_num(col: &Option<Vec<Option<f64>>>, i: usize) -> Option<f64> {
    col.as_ref().and_then(|v| v.get(i).copied().flatten())
}

struct BarBaseStyle<'a> {
    stroke_width: f64,
    opacity: f64,
    stroke_dash: Option<&'a [f64]>,
    corner_radius: f64,
}

/// Per-row stroke and fill encoding column vectors loaded from a batch.
struct StrokeChannels {
    general_opacity: Option<Vec<Option<f64>>>,
    opacity: Option<Vec<Option<f64>>>,
    width: Option<Vec<Option<f64>>>,
    dash: Option<Vec<Option<f64>>>,
    angle: Option<Vec<Option<f64>>>,
    fill_opacity: Option<Vec<Option<f64>>>,
}

impl StrokeChannels {
    fn load(ctx: &DrawCtx) -> Self {
        Self {
            general_opacity: ctx.spec.encoding.opacity.as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
            opacity: ctx.spec.encoding.stroke_opacity.as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
            width: ctx.spec.encoding.stroke_width.as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
            dash: ctx.spec.encoding.stroke_dash.as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
            angle: ctx.spec.encoding.angle.as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
            fill_opacity: ctx.spec.encoding.fill_opacity.as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
        }
    }

    /// Build a `FillStroke` for row `i`, overriding `base_*` defaults with any
    /// per-row column values.  `corner_radius` is passed through unchanged.
    fn row_fill_stroke(
        &self,
        fill: Option<ferrum_scene::Color>,
        stroke: Option<ferrum_scene::Color>,
        base: &BarBaseStyle<'_>,
        i: usize,
    ) -> (ferrum_scene::FillStroke, f64) {
        let (base_sw, opacity, base_dash, corner_radius) =
            (base.stroke_width, base.opacity, base.stroke_dash, base.corner_radius);
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
            .and_then(resolve_stroke_dash);
        let effective_dash = dash_vec.as_deref().or(base_dash).map(|d| d.to_vec());

        let angle = self.angle.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);

        let general_opacity = self.general_opacity.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0));

        let fill_opacity = self.fill_opacity.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .or(general_opacity)
            .unwrap_or(1.0);

        // When stroke_width encoding produces a positive value but no explicit
        // stroke color exists, use the fill color as the stroke so the width is
        // visible in SVG (stroke-width is only emitted when stroke is Some).
        let effective_stroke = if stroke_width > 0.0 && stroke.is_none() && self.width.is_some() {
            fill
        } else {
            stroke
        };

        let fs = ferrum_scene::FillStroke {
            fill,
            stroke: effective_stroke,
            stroke_width,
            opacity,
            stroke_dash: effective_dash,
            stroke_opacity,
            fill_opacity,
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
        (ScaleKind::Linear(_) | ScaleKind::Log(_) | ScaleKind::Symlog(_) | ScaleKind::Pow(_) | ScaleKind::Time(_), _) => {
            build_quantitative(ctx)
        }
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
    // Use col_as_ordinal_category_str so integer-typed ordinal x columns (e.g.
    // Int64 year values) stringify consistently with the ordinal domain.
    let x_strs = match col_as_ordinal_category_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
    if x_strs.len() != ys.len() { return empty_result(); }

    // y2 column: when bound, the rect spans [y, y2] rather than [y, baseline].
    let y2f_opt = spec.encoding.y2.as_ref().map(|e| e.field.as_str());
    let y2s_opt: Option<Vec<Option<f64>>> = y2f_opt
        .and_then(|f| col_as_f64(ctx.batch, f).ok());

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
        if set.is_empty() { 1 } else { set.len() + if x_offsets.contains(&0.0) { 1 } else { 0 } }
    } else {
        1
    };
    let bar_width = if has_pos_offsets {
        ((panel.w / n_categories as f64) / n_groups.max(1) as f64) * 0.8
    } else {
        (panel.w / n_categories as f64) * 0.8
    };

    let (color_values, color_values_f64) = load_color_columns(ctx);
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
        // Priority: explicit y2 column > stacking baseline > axis baseline.
        let bottom_y = if let Some(ref y2s) = y2s_opt {
            // y2 present: use the y2 data value mapped through the y-scale.
            // Handle mixed-sign (y2 may be above or below y in data space):
            // the rect always spans [min_pixel, max_pixel] in screen coords.
            match y2s.get(i).and_then(|v| *v).filter(|v| v.is_finite()) {
                Some(y2v) => ctx.scales.y.to_pixel_f64(y2v).unwrap_or(baseline_y),
                None => baseline_y,
            }
        } else {
            match y_bases.as_ref().and_then(|v| v[i]) {
                Some(b) if b.is_finite() => {
                    ctx.scales.y.to_pixel_f64(b).unwrap_or(baseline_y)
                }
                _ => baseline_y,
            }
        };
        // Use abs so the rect height is positive regardless of y/y2 ordering.
        let height = (bottom_y - top_y).abs().max(0.0);
        let rect_top_y = top_y.min(bottom_y);
        let cx = cx + x_offsets[i];
        let top_y = rect_top_y + y_offsets[i];

        let fill_color = resolve_fill_color(
            ctx.scales.color.as_ref(),
            row_cat(&color_values, i),
            row_num(&color_values_f64, i),
            ctx.mark_style.fill,
        );
        let fill = with_opacity(fill_color, ctx.mark_style.opacity);

        let stroke_sc = ctx.mark_style.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let base = BarBaseStyle {
            stroke_width: ctx.mark_style.stroke_width,
            opacity: ctx.mark_style.opacity,
            stroke_dash: ctx.mark_style.stroke_dash.as_deref(),
            corner_radius: ctx.mark_style.corner_radius,
        };
        let (style, cr) = sc.row_fill_stroke(Some(fill_sc), stroke_sc, &base, i);

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
    // Use col_as_ordinal_category_str so integer-typed ordinal y columns
    // stringify consistently with the ordinal domain.
    let y_strs = match col_as_ordinal_category_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
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

    let (color_values, color_values_f64) = load_color_columns(ctx);
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

        let fill_color = resolve_fill_color(
            ctx.scales.color.as_ref(),
            row_cat(&color_values, i),
            row_num(&color_values_f64, i),
            ctx.mark_style.fill,
        );
        let fill = with_opacity(fill_color, ctx.mark_style.opacity);

        let stroke_sc = ctx.mark_style.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let base = BarBaseStyle {
            stroke_width: ctx.mark_style.stroke_width,
            opacity: ctx.mark_style.opacity,
            stroke_dash: ctx.mark_style.stroke_dash.as_deref(),
            corner_radius: ctx.mark_style.corner_radius,
        };
        let (style, cr) = sc.row_fill_stroke(Some(fill_sc), stroke_sc, &base, i);

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

    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
    if xs.len() != ys.len() { return empty_result(); }

    // Load x2 column if the encoding is present.
    let x2s_opt: Option<Vec<Option<f64>>> = spec.encoding.x2.as_ref()
        .map(|e| e.field.as_str())
        .and_then(|f| col_as_f64(ctx.batch, f).ok());

    if let Some(ref x2s) = x2s_opt {
        if x2s.len() != ys.len() { return empty_result(); }
    }

    // When x2 is absent, auto-compute bar width from minimum spacing between
    // adjacent x values (like ggplot2's continuous-x bar behavior).
    let auto_bar_width: Option<f64> = if x2s_opt.is_none() {
        let mut sorted_xs: Vec<f64> = xs.iter()
            .filter_map(|v| v.filter(|x| x.is_finite()))
            .collect();
        sorted_xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_xs.dedup();

        if sorted_xs.len() >= 2 {
            let min_step = sorted_xs.windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .filter(|s| *s > 0.0)
                .fold(f64::INFINITY, f64::min);

            if min_step.is_finite() {
                // Convert data-space step to pixel width via the scale.
                let p0 = ctx.scales.x.to_pixel_f64(sorted_xs[0]).unwrap_or(0.0);
                let p1 = ctx.scales.x.to_pixel_f64(sorted_xs[0] + min_step).unwrap_or(0.0);
                let px_width = (p1 - p0).abs() * 0.8; // 0.8 gap factor
                Some(px_width.max(1.0))
            } else {
                Some(ctx.panel.plot_area.w * 0.2)
            }
        } else {
            // Single data point: use 20% of plot width.
            Some(ctx.panel.plot_area.w * 0.2)
        }
    } else {
        None
    };

    let panel = ctx.panel.plot_area;
    let baseline_y = panel.y + panel.h;

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let (color_values, color_values_f64) = load_color_columns(ctx);
    let sc = StrokeChannels::load(ctx);
    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

    for i in 0..xs.len() {
        let xv = match xs[i] { Some(v) if v.is_finite() => v, _ => continue };
        let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
        let top_y = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
        let top_y = top_y + y_offsets[i];
        let height = (baseline_y - top_y).max(0.0);

        let (rect_x, width) = if let Some(ref x2s) = x2s_opt {
            // x2 present: bin-style rect from x to x2 (histogram path).
            let x2v = match x2s[i] { Some(v) if v.is_finite() => v, _ => continue };
            let px_left = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            let px_right = match ctx.scales.x.to_pixel_f64(x2v) { Some(p) => p, None => continue };
            let px_left = px_left + x_offsets[i];
            let w = (px_right - px_left).abs().max(1.0);
            (px_left.min(px_right), w)
        } else {
            // No x2: center bar at x pixel with auto-computed width.
            let cx = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            let cx = cx + x_offsets[i];
            let bw = auto_bar_width.unwrap_or(20.0);
            (cx - bw / 2.0, bw)
        };

        let fill_color = resolve_fill_color(
            ctx.scales.color.as_ref(),
            row_cat(&color_values, i),
            row_num(&color_values_f64, i),
            ctx.mark_style.fill,
        );
        let fill = with_opacity(fill_color, ctx.mark_style.opacity);

        let stroke_sc = ctx.mark_style.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let base = BarBaseStyle {
            stroke_width: ctx.mark_style.stroke_width,
            opacity: ctx.mark_style.opacity,
            stroke_dash: ctx.mark_style.stroke_dash.as_deref(),
            corner_radius: ctx.mark_style.corner_radius,
        };
        let (style, cr) = sc.row_fill_stroke(Some(fill_sc), stroke_sc, &base, i);

        nodes.push(SceneNode::Rect {
            x: rect_x,
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
        descriptions,
    }
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

    let (color_values, color_values_f64) = load_color_columns(ctx);
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

        let fill_color = resolve_fill_color(
            ctx.scales.color.as_ref(),
            row_cat(&color_values, i),
            row_num(&color_values_f64, i),
            ctx.mark_style.fill,
        );
        let fill = with_opacity(fill_color, ctx.mark_style.opacity);

        let stroke_sc = ctx.mark_style.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let base = BarBaseStyle {
            stroke_width: ctx.mark_style.stroke_width,
            opacity: ctx.mark_style.opacity,
            stroke_dash: ctx.mark_style.stroke_dash.as_deref(),
            corner_radius: ctx.mark_style.corner_radius,
        };
        let (style, cr) = sc.row_fill_stroke(Some(fill_sc), stroke_sc, &base, i);

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
        chart_description: None,
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
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
        chart_description: None,
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
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
        chart_description: None,
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
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
        chart_description: None,
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count(), 3, "expected 3 ranged-horizontal bars");
    }

    #[test]
    fn bar_quantitative_x_no_x2_auto_width() {
        // Continuous x without x2: bars should auto-compute width from min spacing.
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 15.0, 25.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 200.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 200.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        // Should produce 4 bars (one per row)
        let rects: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { x, w, .. } = n { Some((*x, *w)) } else { None }
        }).collect();
        assert_eq!(rects.len(), 4, "expected 4 bars for 4 data points");
        // All bars should have the same width (auto-computed from uniform spacing)
        let first_w = rects[0].1;
        for (_, w) in &rects {
            assert!((w - first_w).abs() < 1e-6, "all bars should have equal width, got {} vs {}", w, first_w);
        }
        // Width should be positive and less than 1/4 of plot width (they need to fit)
        assert!(first_w > 0.0 && first_w < 300.0 / 3.0, "bar width {} should be positive and reasonable", first_w);
    }

    #[test]
    fn bar_quantitative_x_no_x2_single_point() {
        // Single data point: should use fallback width (20% of plot width).
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![5.0])),
            Arc::new(Float64Array::from(vec![10.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 200.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 200.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let rects: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { w, .. } = n { Some(*w) } else { None }
        }).collect();
        assert_eq!(rects.len(), 1, "expected 1 bar for 1 data point");
        // Fallback width = 20% of 200.0 = 40.0
        assert!((rects[0] - 40.0).abs() < 1e-6, "single-point bar width should be 40.0, got {}", rects[0]);
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
        chart_description: None,
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
        theme.sizes.bar_corner_radius = 3.0;
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
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

    #[test]
    fn bar_integer_ordinal_x_emits_rects() {
        // D9-B regression: ordinal x with Int64 column must emit bars, not return
        // empty. Previously col_as_str failed on Int64 and build_ordinal returned
        // empty_result().
        use arrow::array::Int64Array;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "year".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "v".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("year", DataType::Int64, false),
            Field::new("v",    DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(vec![2000i64, 2001, 2002, 2003])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let rect_count = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Rect { .. }))
            .count();
        assert_eq!(rect_count, 4,
            "Int64 ordinal x must emit one rect per row; got {rect_count}");
    }

    /// D3: ordinal x + y + y2 — bar spans [y, y2], not [y, baseline].
    #[test]
    fn bar_ordinal_x_y2_spans_y_to_y2_not_baseline() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "lo".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
        };
        // lo=[5,7], hi=[10,12]: bars should float entirely above the baseline.
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8,    false),
            Field::new("lo",  DataType::Float64, false),
            Field::new("hi",  DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![5.0, 7.0])),
            Arc::new(Float64Array::from(vec![10.0, 12.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let rects: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { y, h, .. } = n { Some((*y, *h)) } else { None }
        }).collect();
        assert_eq!(rects.len(), 2, "expected 2 bars");

        // The baseline (y=0 in data) maps to y_pixel = panel.h = 100.0.
        // With lo=5..10 and hi=7..12 strictly above 0, no bar should bottom out at 100.0.
        let baseline_pixel = 100.0_f64;
        for (y, h) in &rects {
            let bar_bottom = y + h;
            assert!(
                (bar_bottom - baseline_pixel).abs() > 2.0,
                "bar bottom {bar_bottom:.3} is at the baseline {baseline_pixel:.3}: y2 is being ignored"
            );
        }
    }

    /// D3: ordinal x + y + y2 with mixed-sign values — bar crosses zero.
    #[test]
    fn bar_ordinal_x_y2_mixed_sign_crosses_baseline() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "lo".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
        };
        // lo=-3, hi=2: bar must span across zero (both sides of baseline).
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8,    false),
            Field::new("lo",  DataType::Float64, false),
            Field::new("hi",  DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(Float64Array::from(vec![-3.0])),
            Arc::new(Float64Array::from(vec![2.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let rects: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { y, h, .. } = n { Some((*y, *h)) } else { None }
        }).collect();
        assert_eq!(rects.len(), 1, "expected 1 bar for diverging range");

        // With range [-3, 2], scale spans 5 data units across 100px.
        // Baseline (0) is at 3/5 * 100 = 60px from top.
        // hi=2 maps to 3/5 * 100 - 2/5 * 100 = 40px from top (above baseline).
        // lo=-3 maps to 3/5 * 100 + 3/5 * 100 = ... let the scale decide.
        // Just assert the rect top is above the baseline and bottom is below.
        let (rect_y, rect_h) = rects[0];
        let rect_bottom = rect_y + rect_h;

        // The scale anchors at the data range min/max; baseline pixel is where 0 maps.
        let baseline_pixel = scales.y.to_pixel_f64(0.0).expect("0 must map through y scale");
        assert!(rect_y < baseline_pixel - 1.0,
            "rect top {rect_y:.3} should be above baseline {baseline_pixel:.3} (hi=2 side)");
        assert!(rect_bottom > baseline_pixel + 1.0,
            "rect bottom {rect_bottom:.3} should be below baseline {baseline_pixel:.3} (lo=-3 side)");
    }
}
