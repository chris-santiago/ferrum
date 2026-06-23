//! mark_bar: three paths —
//!   ordinal x → quantitative y: one <rect> per row anchored at x-band center.
//!   quantitative x + x2 → quantitative y: bin rect from x_pixel to x2_pixel
//!   (histogram path added Phase 10c-pre).
//!   quantitative x → ordinal y: horizontal bar per row from panel-left to
//!   x_pixel (Phase 10d-pre, feature-importance chart).

#[cfg(test)]
use crate::layout::Rect;
use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_ordinal_category_str, col_as_positional_category_str, resolve_fill_color, resolve_stroke_dash, x_field, y_field, DrawCtx, MetadataColumns};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::opacity::{OpacityFallback, OpacityResolver};
use crate::render::scale_resolve::ScaleKind;

/// Load the per-row color-encoding columns for fill resolution via the shared
/// [`color_column_loader`](crate::render::marks::channels::color_column_loader)
/// (C9): the categorical string column for `Categorical` (and scale-less) charts,
/// the numeric column for `Continuous` charts. Byte-identical to the prior local
/// helper and to `point`'s inline split.
fn load_color_columns(ctx: &DrawCtx) -> crate::render::marks::channels::ColorColumns {
    crate::render::marks::channels::color_column_loader(ctx)
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

/// Per-row stroke encoding column vectors loaded from a batch.
///
/// The `opacity` / `fill_opacity` / `stroke_opacity` channels are resolved by
/// the shared [`OpacityResolver`] (FA-11) and passed into
/// [`row_fill_stroke`](StrokeChannels::row_fill_stroke); this struct owns only
/// the stroke-geometry channels (`width`, `dash`, `angle`).
struct StrokeChannels {
    width: Option<Vec<Option<f64>>>,
    dash: Option<Vec<Option<f64>>>,
    angle: Option<Vec<Option<f64>>>,
}

impl StrokeChannels {
    fn load(ctx: &DrawCtx) -> Self {
        Self {
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
    ///
    /// `fill_opacity` / `stroke_opacity` are pre-resolved (finite-checked,
    /// clamped, defaulted) by the shared [`OpacityResolver`]; bar's
    /// `fill_opacity ← opacity` fallback lives in that resolver
    /// (`OpacityFallback::BarLike`).
    fn row_fill_stroke(
        &self,
        fill: Option<ferrum_scene::Color>,
        stroke: Option<ferrum_scene::Color>,
        base: &BarBaseStyle<'_>,
        fill_opacity: f64,
        stroke_opacity: f64,
        i: usize,
    ) -> (ferrum_scene::FillStroke, f64) {
        let (base_sw, opacity, base_dash, corner_radius) =
            (base.stroke_width, base.opacity, base.stroke_dash, base.corner_radius);

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
    // CoordPolar: bars become arc wedges (wind-rose / coxcomb). The angular
    // position/width comes from the angle channel; the radial span is the
    // stacked [base, top] mapped through the radial scale.
    if matches!(ctx.spec.coord, Some(crate::spec::coord::CoordKind::Polar { .. })) {
        return build_polar(ctx);
    }
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

/// CoordPolar bar → arc-wedge renderer (wind-rose / coxcomb).
///
/// Under the Python polar remapping `theta="x"` puts the angular channel in
/// `encoding.x` and the radial (value) channel in `encoding.y`; `theta="y"`
/// mirrors. Each distinct angular category occupies an equal slice of the
/// circle (`tau / n`). A bar's radial span is `[base, top]` where `top` is the
/// y value and `base` is the stacking base (`__stack_y_base__`, 0 when
/// unstacked) — both mapped through the radial scale. Stacked segments thus
/// accumulate outward: segment B's inner radius equals segment A's outer
/// radius, with no overlap at r=0.
fn build_polar(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::MarkBuildResult;
    use crate::render::marks::arc::{polar_geom, radius_to_pixel, wedge_path};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let theta_ch = match &ctx.spec.coord {
        Some(crate::spec::coord::CoordKind::Polar { theta, .. }) => *theta,
        _ => return empty_result(),
    };
    let Some(geom) = polar_geom(ctx) else { return empty_result() };

    // Angular channel = theta-mapped axis; radial (value) channel = the other.
    // The shared resolver (C9) encodes the canonical theta→radius convention also
    // used by arc.rs. Byte-identical to the prior inline `match theta_ch`.
    let pc = crate::render::marks::channels::polar_channel_resolver(
        theta_ch, &ctx.spec.encoding, ctx.scales,
    );
    let radius_scale = pc.radius_scale;
    let (Some(af), Some(vf)) = (pc.theta_field, pc.radius_field) else { return empty_result() };

    // Angular categories: stringify so ordinal and integer-coded angle columns
    // group consistently. Each distinct value (first-appearance order) gets an
    // equal angular band.
    let angle_strs = match col_as_ordinal_category_str(ctx.batch, af) {
        Ok(v) => v,
        Err(_) => return empty_result(),
    };
    let tops = match col_as_f64(ctx.batch, vf) { Ok(v) => v, Err(_) => return empty_result() };
    if angle_strs.len() != tops.len() { return empty_result(); }

    // Stacking base (segment bottoms). Absent when unstacked → base = 0.
    let bases: Option<Vec<Option<f64>>> = col_as_f64(ctx.batch, "__stack_y_base__").ok();

    // Distinct angular categories in first-appearance order.
    let mut cat_index: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut cat_order: Vec<&str> = Vec::new();
    for s in angle_strs.iter().flatten() {
        if !cat_index.contains_key(s.as_str()) {
            cat_index.insert(s.as_str(), cat_order.len());
            cat_order.push(s.as_str());
        }
    }
    let n_cats = cat_order.len().max(1);
    let tau = std::f64::consts::TAU;
    let band = tau / n_cats as f64;
    // A single angular category would span the full circle. SVG cannot draw a
    // 360° arc in one command (start==end is degenerate), so `wedge_path`
    // splits it into two semicircle arcs. Leave a hairline gap when no explicit
    // pad is set so each ring renders as a single arc (coxcomb convention) and
    // concentric stacked rings stay individually identifiable.
    let pad_angle = if geom.pad_angle > 0.0 {
        geom.pad_angle
    } else if n_cats == 1 {
        1e-3
    } else {
        0.0
    };

    let (color_values, color_values_f64) = load_color_columns(ctx);
    let sc = StrokeChannels::load(ctx);
    // opacity / fill_opacity / stroke_opacity via the shared resolver (FA-11),
    // sampled per-row. `OpacityFallback::BarLike` preserves bar's unique
    // `fill_opacity ← opacity` fallback. The resolved opacity output is unused:
    // bar bakes `mark_style.paint.opacity` into the fill color and the FillStroke.
    let opacity_res = OpacityResolver::load(ctx, OpacityFallback::BarLike, (ctx.mark_style.paint.opacity, 1.0, 1.0));
    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep so that metadata is
    // aligned to the KEPT nodes only (not all rows). Rows are skipped below
    // for null categories and non-finite radial values; `build_metadata(ctx)`
    // would return full per-row vectors indexed 0..n_rows, misaligning node j
    // with row j whenever any row is skipped (#6 defect class).
    let mut acc = MarkNodes::with_capacity(angle_strs.len());

    for i in 0..angle_strs.len() {
        let cat = match &angle_strs[i] { Some(s) => s.as_str(), None => continue };
        let top = match tops[i] { Some(v) if v.is_finite() => v, _ => continue };
        let base = bases.as_ref().and_then(|v| v[i]).filter(|v| v.is_finite()).unwrap_or(0.0);

        let k = *cat_index.get(cat).unwrap_or(&0);
        let angle_start = geom.start_angle + k as f64 * band + pad_angle / 2.0;
        let angle_end = geom.start_angle + (k as f64 + 1.0) * band - pad_angle / 2.0;
        if angle_end <= angle_start { continue; }

        let inner_r = radius_to_pixel(radius_scale, geom.inner_radius, geom.outer_radius, base);
        let outer_r = radius_to_pixel(radius_scale, geom.inner_radius, geom.outer_radius, top);

        let fill_color = resolve_fill_color(
            ctx.scales.color.as_ref(),
            row_cat(&color_values, i),
            row_num(&color_values_f64, i),
            ctx.mark_style.paint.fill,
        );
        let fill_c = crate::render::draw::to_scene_color(with_opacity(fill_color, ctx.mark_style.paint.opacity));
        let stroke_sc = ctx.mark_style.paint.stroke.map(crate::render::draw::to_scene_color);
        let base_style = BarBaseStyle {
            stroke_width: ctx.mark_style.paint.stroke_width,
            opacity: ctx.mark_style.paint.opacity,
            stroke_dash: ctx.mark_style.paint.stroke_dash.as_deref(),
            corner_radius: 0.0,
        };
        let (_, fill_opacity, stroke_opacity) = opacity_res.at_row(i);
        let (style, _) = sc.row_fill_stroke(Some(fill_c), stroke_sc, &base_style, fill_opacity, stroke_opacity, i);

        let commands = wedge_path(geom.cx, geom.cy, inner_r, outer_r, angle_start, angle_end);
        acc.push(SceneNode::Path { commands, style, closed: true }, i);
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Arc,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
    }
}

fn empty_result() -> crate::render::draw::MarkBuildResult {
    crate::render::draw::MarkBuildResult::empty(ferrum_scene::MarkBatchKind::Bar)
}

fn build_ordinal(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_color};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    // Use col_as_positional_category_str so integer-typed ordinal x columns (e.g.
    // Int64 year values) stringify consistently with the ordinal domain, and a
    // null x category lands in its own band (FA-9), matching the positional domain.
    let x_strs = match col_as_positional_category_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
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
    // opacity / fill_opacity / stroke_opacity via the shared resolver (FA-11),
    // sampled per-row. `OpacityFallback::BarLike` preserves bar's unique
    // `fill_opacity ← opacity` fallback. The resolved opacity output is unused:
    // bar bakes `mark_style.paint.opacity` into the fill color and the FillStroke.
    let opacity_res = OpacityResolver::load(ctx, OpacityFallback::BarLike, (ctx.mark_style.paint.opacity, 1.0, 1.0));
    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep so metadata is
    // aligned to kept nodes only. Rows are skipped for null categories, non-finite
    // y values, and out-of-range pixels (#6 defect class fix).
    let mut acc = MarkNodes::with_capacity(x_strs.len());

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
            ctx.mark_style.paint.fill,
        );
        let fill = with_opacity(fill_color, ctx.mark_style.paint.opacity);

        let stroke_sc = ctx.mark_style.paint.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let base = BarBaseStyle {
            stroke_width: ctx.mark_style.paint.stroke_width,
            opacity: ctx.mark_style.paint.opacity,
            stroke_dash: ctx.mark_style.paint.stroke_dash.as_deref(),
            corner_radius: ctx.mark_style.paint.corner_radius,
        };
        let (_, fill_opacity, stroke_opacity) = opacity_res.at_row(i);
        let (style, cr) = sc.row_fill_stroke(Some(fill_sc), stroke_sc, &base, fill_opacity, stroke_opacity, i);

        acc.push(SceneNode::Rect {
            x: cx - bar_width / 2.0,
            y: top_y,
            w: bar_width,
            h: height,
            style,
            corner_radius: cr,
        }, i);
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Bar,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
    }
}

fn build_ordinal_y(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_color};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return empty_result() };
    // Use col_as_positional_category_str so integer-typed ordinal y columns
    // stringify consistently with the ordinal domain, and a null y category gets
    // its own band (FA-9).
    let y_strs = match col_as_positional_category_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
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
    // opacity / fill_opacity / stroke_opacity via the shared resolver (FA-11),
    // sampled per-row. `OpacityFallback::BarLike` preserves bar's unique
    // `fill_opacity ← opacity` fallback. The resolved opacity output is unused:
    // bar bakes `mark_style.paint.opacity` into the fill color and the FillStroke.
    let opacity_res = OpacityResolver::load(ctx, OpacityFallback::BarLike, (ctx.mark_style.paint.opacity, 1.0, 1.0));
    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep so metadata is
    // aligned to kept nodes only (#6 defect class fix).
    let mut acc = MarkNodes::with_capacity(y_strs.len());

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
            ctx.mark_style.paint.fill,
        );
        let fill = with_opacity(fill_color, ctx.mark_style.paint.opacity);

        let stroke_sc = ctx.mark_style.paint.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let base = BarBaseStyle {
            stroke_width: ctx.mark_style.paint.stroke_width,
            opacity: ctx.mark_style.paint.opacity,
            stroke_dash: ctx.mark_style.paint.stroke_dash.as_deref(),
            corner_radius: ctx.mark_style.paint.corner_radius,
        };
        let (_, fill_opacity, stroke_opacity) = opacity_res.at_row(i);
        let (style, cr) = sc.row_fill_stroke(Some(fill_sc), stroke_sc, &base, fill_opacity, stroke_opacity, i);

        acc.push(SceneNode::Rect {
            x: left_x,
            y: cy - bar_height / 2.0,
            w: width,
            h: bar_height,
            style,
            corner_radius: cr,
        }, i);
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Bar,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
    }
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
    // opacity / fill_opacity / stroke_opacity via the shared resolver (FA-11),
    // sampled per-row. `OpacityFallback::BarLike` preserves bar's unique
    // `fill_opacity ← opacity` fallback. The resolved opacity output is unused:
    // bar bakes `mark_style.paint.opacity` into the fill color and the FillStroke.
    let opacity_res = OpacityResolver::load(ctx, OpacityFallback::BarLike, (ctx.mark_style.paint.opacity, 1.0, 1.0));
    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep so metadata is
    // aligned to kept nodes only (#6 defect class fix).
    let mut acc = MarkNodes::with_capacity(xs.len());

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
            ctx.mark_style.paint.fill,
        );
        let fill = with_opacity(fill_color, ctx.mark_style.paint.opacity);

        let stroke_sc = ctx.mark_style.paint.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let base = BarBaseStyle {
            stroke_width: ctx.mark_style.paint.stroke_width,
            opacity: ctx.mark_style.paint.opacity,
            stroke_dash: ctx.mark_style.paint.stroke_dash.as_deref(),
            corner_radius: ctx.mark_style.paint.corner_radius,
        };
        let (_, fill_opacity, stroke_opacity) = opacity_res.at_row(i);
        let (style, cr) = sc.row_fill_stroke(Some(fill_sc), stroke_sc, &base, fill_opacity, stroke_opacity, i);

        acc.push(SceneNode::Rect {
            x: rect_x,
            y: top_y,
            w: width,
            h: height,
            style,
            corner_radius: cr,
        }, i);
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Bar,
        nodes,
        data_indices: Some(data_indices),
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
    // opacity / fill_opacity / stroke_opacity via the shared resolver (FA-11),
    // sampled per-row. `OpacityFallback::BarLike` preserves bar's unique
    // `fill_opacity ← opacity` fallback. The resolved opacity output is unused:
    // bar bakes `mark_style.paint.opacity` into the fill color and the FillStroke.
    let opacity_res = OpacityResolver::load(ctx, OpacityFallback::BarLike, (ctx.mark_style.paint.opacity, 1.0, 1.0));
    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep so metadata is
    // aligned to kept nodes only (#6 defect class fix).
    let mut acc = MarkNodes::with_capacity(xs.len());

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
            ctx.mark_style.paint.fill,
        );
        let fill = with_opacity(fill_color, ctx.mark_style.paint.opacity);

        let stroke_sc = ctx.mark_style.paint.stroke.map(to_scene_color);
        let fill_sc = to_scene_color(fill);
        let base = BarBaseStyle {
            stroke_width: ctx.mark_style.paint.stroke_width,
            opacity: ctx.mark_style.paint.opacity,
            stroke_dash: ctx.mark_style.paint.stroke_dash.as_deref(),
            corner_radius: ctx.mark_style.paint.corner_radius,
        };
        let (_, fill_opacity, stroke_opacity) = opacity_res.at_row(i);
        let (style, cr) = sc.row_fill_stroke(Some(fill_sc), stroke_sc, &base, fill_opacity, stroke_opacity, i);

        acc.push(SceneNode::Rect {
            x: baseline_x,
            y: py_top.min(py_bottom),
            w: width,
            h: height,
            style,
            corner_radius: cr,
        }, i);
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Bar,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
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
        params: Vec::new(),
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
        params: Vec::new(),
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
        params: Vec::new(),
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
        params: Vec::new(),
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
            params: Vec::new(),
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
            params: Vec::new(),
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
        params: Vec::new(),
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
            params: Vec::new(),
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
            params: Vec::new(),
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
            params: Vec::new(),
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

    /// D7: a stacked bar under CoordPolar renders arc wedges (not rects), and
    /// stacked segments accumulate outward — each segment's inner radius equals
    /// the previous segment's outer radius, with no overlap at r=0.
    #[test]
    fn polar_stacked_bar_emits_contiguous_wedges() {
        use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
        use crate::scale::linear::LinearScale;
        use crate::spec::coord::{CoordKind as SpecCoord, PolarThetaChannel};
        use ferrum_scene::{PathCmd, PolarDirection};

        // Single direction, three stacked categories A/B/C with values
        // 10/20/30 → segment tops 10/30/60, bases 0/10/30 (as apply_stack
        // would produce). The __stack_y_base__ column is supplied directly so
        // this test exercises the wedge geometry in isolation.
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "dir".into(), type_: Some(SDT::Nominal), ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                color: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Nominal), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: Some(120.0),
                pad_angle: 0.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("dir", DataType::Utf8, false),
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
            Field::new("__stack_y_base__", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["N", "N", "N"])),
            Arc::new(StringArray::from(vec!["A", "B", "C"])),
            Arc::new(Float64Array::from(vec![10.0, 30.0, 60.0])), // segment tops
            Arc::new(Float64Array::from(vec![0.0, 10.0, 30.0])),  // segment bases
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        // Radial scale (y): domain [0, 60] anchored at 0.
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 1.0], vec![0.0, 300.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 60.0], vec![300.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None, x2: None, y2: None,
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // Polar bars render as Path wedges, never Rect.
        assert!(result.nodes.iter().all(|n| !matches!(n, SceneNode::Rect { .. })),
            "polar bars must not emit Rect nodes");
        let paths: Vec<&SceneNode> = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Path { .. })).collect();
        assert_eq!(paths.len(), 3, "expected one wedge per stacked segment");

        // Extract (inner_r, outer_r) per wedge from the first/last arc radii.
        // Solid wedge (inner_r=0): one arc. Annular: outer arc then inner arc.
        let radii: Vec<(f64, f64)> = paths.iter().map(|n| {
            let cmds = if let SceneNode::Path { commands, .. } = n { commands } else { unreachable!() };
            let arcs: Vec<f64> = cmds.iter().filter_map(|c| {
                if let PathCmd::ArcTo { rx, .. } = c { Some(*rx) } else { None }
            }).collect();
            if arcs.len() == 1 { (0.0, arcs[0]) } else { (arcs[arcs.len()-1], arcs[0]) }
        }).collect();

        // Expected pixel radii: top/60 * 120. A: 0..20, B: 20..60, C: 60..120.
        let mut sorted = radii.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert!((sorted[0].0 - 0.0).abs() < 1e-6, "first segment inner_r should be 0");
        // Contiguity: each segment's outer == next segment's inner.
        for i in 0..sorted.len() - 1 {
            assert!((sorted[i].1 - sorted[i + 1].0).abs() < 1e-6,
                "segment {i} outer_r={} != segment {} inner_r={}", sorted[i].1, i + 1, sorted[i + 1].0);
        }
        // At least one segment has a non-zero inner radius (no r=0 overlap).
        assert!(sorted.iter().any(|(inner, _)| *inner > 1.0),
            "stacked segments must accumulate outward (non-zero inner radii)");
    }

    /// FA-2: 2-category polar bar must produce 2 wedges whose angular sweeps
    /// sum to ~2π (full circle) and each wedge spans exactly π (180°).
    ///
    /// The double-transform bug caused arc paths to be mispositioned (their
    /// start/end points were NOT on the circle of the stated A-radius) and
    /// their sweeps were ~60° instead of 180°. This test verifies the geometry
    /// directly from the Path arc commands.
    #[test]
    fn polar_bar_two_cats_equal_angular_bands() {
        use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
        use crate::scale::linear::LinearScale;
        use crate::spec::coord::{CoordKind as SpecCoord, PolarThetaChannel};
        use ferrum_scene::{PathCmd, PolarDirection};
        use std::f64::consts::PI;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Nominal), ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: Some(100.0),
                pad_angle: 0.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["A", "B"])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 1.0], vec![0.0, 300.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 20.0], vec![300.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None, x2: None, y2: None,
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let paths: Vec<&SceneNode> = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Path { .. })).collect();
        assert_eq!(paths.len(), 2, "2 categories must produce 2 wedge paths");

        // For each wedge: extract (M start, A outer_r, A end) and verify
        // the start and end points ARE on the outer circle (i.e. distance from
        // cx,cy equals outer_r), proving no double-transform occurred.
        let cx = 150.0_f64; // panel center x = 300/2
        let cy = 150.0_f64; // panel center y = 300/2

        let mut sweeps = Vec::new();
        for node in &paths {
            let cmds = if let SceneNode::Path { commands, .. } = node { commands } else { unreachable!() };

            // Start point from MoveTo.
            let (mx, my) = cmds.iter().find_map(|c| {
                if let PathCmd::MoveTo { x, y } = c { Some((*x, *y)) } else { None }
            }).expect("no MoveTo in wedge path");

            // First ArcTo: outer_r and end point.
            let (outer_r, ex, ey) = cmds.iter().find_map(|c| {
                if let PathCmd::ArcTo { rx, x, y, .. } = c { Some((*rx, *x, *y)) } else { None }
            }).expect("no ArcTo in wedge path");

            // Both start and end must be on the outer circle.
            let dist_start = ((mx - cx).powi(2) + (my - cy).powi(2)).sqrt();
            let dist_end   = ((ex - cx).powi(2) + (ey - cy).powi(2)).sqrt();
            assert!((dist_start - outer_r).abs() < 0.5,
                "Wedge start ({mx:.3},{my:.3}) is {dist_start:.3}px from center, expected outer_r={outer_r:.3}. \
                 Double-transform bug would make start NOT on the outer circle.");
            assert!((dist_end - outer_r).abs() < 0.5,
                "Wedge end ({ex:.3},{ey:.3}) is {dist_end:.3}px from center, expected outer_r={outer_r:.3}.");

            // Chord and sweep angle.
            let chord = ((ex - mx).powi(2) + (ey - my).powi(2)).sqrt();
            let ratio = (chord / (2.0 * outer_r)).min(1.0);
            let sweep = 2.0 * ratio.asin();
            sweeps.push(sweep);
        }

        // Each of the 2 categories must span π (180°).
        for (i, &sweep) in sweeps.iter().enumerate() {
            assert!((sweep - PI).abs() < 0.05,
                "Wedge {i} sweep = {:.1}°, expected 180°. n=2 → band=π.", sweep.to_degrees());
        }
        // Total must be ≈ 2π.
        let total: f64 = sweeps.iter().sum();
        assert!((total - std::f64::consts::TAU).abs() < 0.1,
            "Total sweep {:.1}°, expected 360°.", total.to_degrees());
    }

    /// FA-2: 4-category polar bar — each wedge spans π/2 (90°).
    #[test]
    fn polar_bar_four_cats_equal_angular_bands() {
        use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
        use crate::scale::linear::LinearScale;
        use crate::spec::coord::{CoordKind as SpecCoord, PolarThetaChannel};
        use ferrum_scene::{PathCmd, PolarDirection};
        use std::f64::consts::PI;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Nominal), ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: Some(100.0),
                pad_angle: 0.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["N", "E", "S", "W"])),
            Arc::new(Float64Array::from(vec![10.0, 15.0, 20.0, 8.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 1.0], vec![0.0, 300.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 20.0], vec![300.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None, x2: None, y2: None,
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let paths: Vec<&SceneNode> = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Path { .. })).collect();
        assert_eq!(paths.len(), 4, "4 categories must produce 4 wedge paths");

        let cx = 150.0_f64;
        let cy = 150.0_f64;
        let mut sweeps = Vec::new();

        for node in &paths {
            let cmds = if let SceneNode::Path { commands, .. } = node { commands } else { unreachable!() };
            let (mx, my) = cmds.iter().find_map(|c| {
                if let PathCmd::MoveTo { x, y } = c { Some((*x, *y)) } else { None }
            }).expect("no MoveTo");
            let (outer_r, ex, ey) = cmds.iter().find_map(|c| {
                if let PathCmd::ArcTo { rx, x, y, .. } = c { Some((*rx, *x, *y)) } else { None }
            }).expect("no ArcTo");

            let dist_start = ((mx - cx).powi(2) + (my - cy).powi(2)).sqrt();
            assert!((dist_start - outer_r).abs() < 0.5,
                "Start not on outer circle: dist={dist_start:.3}, r={outer_r:.3}. Double-transform bug?");

            let chord = ((ex - mx).powi(2) + (ey - my).powi(2)).sqrt();
            let ratio = (chord / (2.0 * outer_r)).min(1.0);
            sweeps.push(2.0 * ratio.asin());
        }

        let half_pi = PI / 2.0;
        for (i, &sweep) in sweeps.iter().enumerate() {
            assert!((sweep - half_pi).abs() < 0.05,
                "Wedge {i} sweep = {:.1}°, expected 90°.", sweep.to_degrees());
        }
        let total: f64 = sweeps.iter().sum();
        assert!((total - std::f64::consts::TAU).abs() < 0.1,
            "Total sweep {:.1}°, expected 360°.", total.to_degrees());
    }

    // ── Metadata-alignment regression tests (#6 defect class) ────────────────
    //
    // Each test creates a batch where some rows are skipped (null / non-finite)
    // and asserts that tooltip metadata on each emitted node points to its TRUE
    // source row, not the node-position row.
    //
    // Fail-before: prior to migrating to MarkNodes + build_metadata_for_indices,
    // all 5 builders called `meta.build_metadata(ctx)` (full per-row vectors)
    // before the loop. When any row was skipped, node j received row j's metadata
    // instead of its true source row — the #6 defect class. These tests would
    // have failed on that old code because node 1's tooltip would be "tip_b"
    // (the skipped row) instead of "tip_c" (the true source row of node 1).
    //
    // Pass-after: migrated builders finalize with build_metadata_for_indices
    // using the kept data_indices, so node j always receives its true source row.

    /// Regression: `build_ordinal` (ordinal-x bar) with a null y-value skips
    /// that row. The tooltip on each surviving node must point to its true source
    /// row, not the node-position row.
    ///
    /// Batch: 3 rows, y=[10.0, null, 30.0], tooltip=["tip_a", "tip_b", "tip_c"].
    /// Row 1 (null y) is skipped → 2 nodes. Node 0 → row 0 → "tip_a"; node 1
    /// → row 2 → "tip_c". The old code would give node 1 → "tip_b" (row 1's
    /// tooltip via full-row indexing).
    #[test]
    fn ordinal_skipped_null_y_tooltip_aligned() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8,    false),
            Field::new("val", DataType::Float64, true),  // nullable
            Field::new("tip", DataType::Utf8,    false),
        ]));
        // Row 1 has null val → skipped.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![Some(10.0_f64), None, Some(30.0)])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = crate::render::draw::DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx);

        // 2 nodes survive (row 1 with null val is skipped).
        assert_eq!(result.nodes.len(), 2,
            "expected 2 bars after null-val skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        // Node 0 → source row 0 → "tip_a".
        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a",
            "node 0 tooltip must be row 0's ('tip_a'); got '{t0}'. \
             Old code would give 'tip_a' here (row 0 passes both old and new).");
        // Node 1 → source row 2 → "tip_c".
        // Old code (full-row indexing) would give row 1's "tip_b" — the bug.
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be row 2's ('tip_c'), not row 1's ('tip_b'); got '{t1}'. \
             This fails on pre-migration code that uses build_metadata(ctx).");
    }

    /// Regression: `build_quantitative` (quantitative-x bar) with a null y-value
    /// skips that row. The tooltip on each surviving node must point to its true
    /// source row.
    ///
    /// Batch: 3 rows, x=[1.0, 2.0, 3.0], y=[10.0, null, 30.0],
    /// tooltip=["tip_a", "tip_b", "tip_c"]. Row 1 (null y) is skipped → 2 nodes.
    /// Node 0 → row 0 → "tip_a"; node 1 → row 2 → "tip_c". Old code: node 1 →
    /// "tip_b".
    #[test]
    fn quantitative_skipped_null_y_tooltip_aligned() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",   DataType::Float64, false),
            Field::new("y",   DataType::Float64, true),  // nullable
            Field::new("tip", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![Some(10.0_f64), None, Some(30.0)])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = crate::render::draw::DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2,
            "expected 2 bars after null-y skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be 'tip_a'; got '{t0}'");

        // The alignment failure: old code gives "tip_b" here (row 1's tooltip);
        // new code gives "tip_c" (row 2's, the true source row of node 1).
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be row 2's ('tip_c'), not row 1's ('tip_b'); got '{t1}'");
    }

    /// Regression: `build_polar` (polar bar / wind-rose) with a null angular
    /// category skips that row. The tooltip on each surviving node must point to
    /// its true source row.
    ///
    /// Batch: 3 rows, cat=["A", null, "B"], val=[10.0, 20.0, 30.0],
    /// tooltip=["tip_a", "tip_null", "tip_b"]. Row 1 (null cat) is skipped →
    /// 2 nodes. Node 0 → row 0 → "tip_a"; node 1 → row 2 → "tip_b". Old code:
    /// node 1 → "tip_null".
    #[test]
    fn polar_skipped_null_category_tooltip_aligned() {
        use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
        use crate::scale::linear::LinearScale;
        use crate::spec::coord::{CoordKind as SpecCoord, PolarThetaChannel};
        use ferrum_scene::PolarDirection;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Nominal), ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: Some(100.0),
                pad_angle: 0.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8,    true),  // nullable
            Field::new("val", DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        // Row 1 has null category → skipped.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec![Some("A"), None, Some("B")])),
            Arc::new(Float64Array::from(vec![10.0_f64, 20.0, 30.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_null", "tip_b"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1.0], vec![0.0, 300.0], false, false,
            )),
            y: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 30.0], vec![300.0, 0.0], false, false,
            )),
            color: None, size: None, shape: None, opacity: None, x2: None, y2: None,
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = crate::render::draw::DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2,
            "expected 2 wedges after null-category skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be 'tip_a'; got '{t0}'");

        // Alignment failure: old code gives "tip_null" (row 1's tooltip) here;
        // new code gives "tip_b" (row 2's, the true source row of node 1).
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_b",
            "node 1 tooltip must be row 2's ('tip_b'), not row 1's ('tip_null'); got '{t1}'");
    }

    /// Stability: `build_ordinal` with NO skipped rows produces tooltips in original
    /// row order — `build_metadata_for_indices` with a full-range index must equal
    /// `build_metadata` for a complete dataset. Backward-compat guard.
    #[test]
    fn ordinal_no_skipped_rows_tooltips_unchanged() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8,    false),
            Field::new("val", DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![10.0_f64, 20.0, 30.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = crate::render::draw::DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 3, "all 3 rows must produce nodes");
        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 3);
        assert_eq!(&tooltips[0].fields[0].value, "tip_a");
        assert_eq!(&tooltips[1].fields[0].value, "tip_b");
        assert_eq!(&tooltips[2].fields[0].value, "tip_c");
    }

    /// Regression: `build_ordinal_y` (horizontal-ordinal bar, y = ordinal category)
    /// with a null x-value skips that row. The tooltip on each surviving node must
    /// point to its true source row, not the node-position row.
    ///
    /// Batch: 3 rows, x=[10.0, null, 30.0], y=["A","B","C"],
    /// tooltip=["tip_a","tip_b","tip_c"]. Row 1 (null x) is skipped → 2 nodes.
    /// Node 0 → row 0 → "tip_a"; node 1 → row 2 → "tip_c".
    /// Old code (pre-migration build_metadata(ctx)) would give node 1 → "tip_b".
    #[test]
    fn ordinal_y_skipped_null_x_tooltip_aligned() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "xval".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "cat".into(),  type_: Some(SDT::Ordinal),      ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("xval", DataType::Float64, true),   // nullable — row 1 will be null
            Field::new("cat",  DataType::Utf8,    false),
            Field::new("tip",  DataType::Utf8,    false),
        ]));
        // Row 1 has null xval → skipped by build_ordinal_y.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![Some(10.0_f64), None, Some(30.0)])),
            Arc::new(StringArray::from(vec!["A", "B", "C"])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = crate::render::draw::DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2,
            "expected 2 bars after null-x skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be row 0's ('tip_a'); got '{t0}'");

        // Alignment failure: old code (build_metadata(ctx)) gives "tip_b" here
        // (row 1's tooltip, indexed by node position); new code gives "tip_c"
        // (row 2's, the true source row of node 1).
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be row 2's ('tip_c'), not row 1's ('tip_b'); got '{t1}'");
    }

    /// Regression: `build_quantitative_horizontal` (x + y + y2 all numeric, horizontal
    /// span bar) with a null x-value skips that row. The tooltip on each surviving
    /// node must point to its true source row.
    ///
    /// Batch: 3 rows, x=[10.0, null, 30.0], y=[0.2,0.5,0.8], y2=[0.4,0.7,1.0],
    /// tooltip=["tip_a","tip_b","tip_c"]. Row 1 (null x) is skipped → 2 nodes.
    /// Node 0 → row 0 → "tip_a"; node 1 → row 2 → "tip_c".
    /// Old code would give node 1 → "tip_b".
    #[test]
    fn quantitative_horizontal_skipped_null_x_tooltip_aligned() {
        use crate::spec::encoding::EncodingSpec as ES;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x:  Some(ES { field: "xval".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y:  Some(ES { field: "y1".into(),   type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(ES { field: "y2".into(),   type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(ES { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("xval", DataType::Float64, true),   // nullable — row 1 will be null
            Field::new("y1",   DataType::Float64, false),
            Field::new("y2",   DataType::Float64, false),
            Field::new("tip",  DataType::Utf8,    false),
        ]));
        // Row 1 has null xval → skipped by build_quantitative_horizontal.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![Some(10.0_f64), None, Some(30.0)])),
            Arc::new(Float64Array::from(vec![0.2_f64, 0.5, 0.8])),
            Arc::new(Float64Array::from(vec![0.4_f64, 0.7, 1.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = crate::render::draw::DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2,
            "expected 2 bars after null-x skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be row 0's ('tip_a'); got '{t0}'");

        // Alignment failure: old code (build_metadata(ctx)) gives "tip_b" here;
        // new code gives "tip_c" (true source row of node 1).
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be row 2's ('tip_c'), not row 1's ('tip_b'); got '{t1}'");
    }

    /// Regression: href metadata alignment with a skipped row. Uses `build_ordinal`
    /// (ordinal-x bar) with href encoding and a null y-value that skips row 1. The
    /// href on each surviving node must point to its true source row's href, not the
    /// node-position row's.
    ///
    /// Batch: 3 rows, y=[10.0, null, 30.0], href=["url_a","url_b","url_c"].
    /// Row 1 (null y) is skipped → 2 nodes. Node 0 → row 0 → "url_a"; node 1 →
    /// row 2 → "url_c". Old code (build_metadata(ctx)) would give node 1 →
    /// "url_b" because hrefs were built from the full column before the loop.
    #[test]
    fn ordinal_skipped_null_y_href_aligned() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x:    Some(EncodingSpec { field: "cat".into(),  type_: Some(SDT::Ordinal),      ..Default::default() }),
                y:    Some(EncodingSpec { field: "val".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                href: Some(EncodingSpec { field: "link".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat",  DataType::Utf8,    false),
            Field::new("val",  DataType::Float64, true),   // nullable — row 1 will be null
            Field::new("link", DataType::Utf8,    false),
        ]));
        // Row 1 has null val → skipped. href column has distinct values per row so
        // a position-indexed result ("url_b") is detectably different from the
        // correct source-row result ("url_c").
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![Some(10.0_f64), None, Some(30.0)])),
            Arc::new(StringArray::from(vec!["url_a", "url_b", "url_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = crate::render::draw::DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2,
            "expected 2 bars after null-val skip; got {}", result.nodes.len());

        // hrefs are independently populated from tooltips via build_metadata_for_indices;
        // this guards that the href path aligns correctly too.
        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), 2, "href count must equal node count");

        let h0 = hrefs[0].as_deref().expect("node 0 href must be Some");
        assert_eq!(h0, "url_a", "node 0 href must be row 0's ('url_a'); got '{h0}'");

        // Alignment failure: old code (build_metadata(ctx)) gives "url_b" here
        // (row 1's href, indexed by node position); new code gives "url_c"
        // (row 2's href, the true source row of node 1).
        let h1 = hrefs[1].as_deref().expect("node 1 href must be Some");
        assert_eq!(h1, "url_c",
            "node 1 href must be row 2's ('url_c'), not row 1's ('url_b'); got '{h1}'");
    }

    // ── FA-11 (#5): OpacityResolver migration preserves bar's per-row opacity ──

    /// FA-11 guard: a bar with an explicit `fill_opacity` encoding applies it
    /// per-row (finite-checked, clamped) — unchanged by the resolver dedup.
    #[test]
    fn bar_per_row_fill_opacity_unchanged_after_resolver() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "v".into(), type_: None, ..Default::default() }),
                fill_opacity: Some(EncodingSpec { field: "fo".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None, selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("g",  DataType::Utf8,    false),
            Field::new("v",  DataType::Float64, false),
            Field::new("fo", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.2, 0.5, 0.9])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let fos: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { style, .. } = n { Some(style.fill_opacity) } else { None }
        }).collect();
        assert_eq!(fos.len(), 3, "expected 3 bar rects");
        let expected = [0.2, 0.5, 0.9];
        for (i, (got, exp)) in fos.iter().zip(expected).enumerate() {
            assert!((got - exp).abs() < 1e-9,
                "bar row {i} fill_opacity expected {exp}, got {got}");
        }
    }

    /// FA-11 guard: bar's UNIQUE `fill_opacity ← opacity` fallback
    /// (`OpacityFallback::BarLike`) is preserved by the resolver. With no
    /// `fill_opacity` encoding but an `opacity` encoding present, each bar's
    /// `fill_opacity` must equal its (clamped) per-row `opacity` value. No other
    /// mark does this; the resolver flag isolates the quirk.
    #[test]
    fn bar_fill_opacity_falls_back_to_opacity_column() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "v".into(), type_: None, ..Default::default() }),
                // opacity present, fill_opacity ABSENT → bar falls fill_opacity
                // back to the opacity column.
                opacity: Some(EncodingSpec { field: "op".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None, selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("g",  DataType::Utf8,    false),
            Field::new("v",  DataType::Float64, false),
            Field::new("op", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.3, 0.7])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let fos: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { style, .. } = n { Some(style.fill_opacity) } else { None }
        }).collect();
        assert_eq!(fos.len(), 2, "expected 2 bar rects");
        assert!((fos[0] - 0.3).abs() < 1e-9,
            "bar row 0 fill_opacity must fall back to opacity 0.3; got {}", fos[0]);
        assert!((fos[1] - 0.7).abs() < 1e-9,
            "bar row 1 fill_opacity must fall back to opacity 0.7; got {}", fos[1]);
    }
}
