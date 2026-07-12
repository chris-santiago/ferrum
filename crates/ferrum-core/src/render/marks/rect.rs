//! mark_rect: three paths —
//!   ordinal x × ordinal y → heatmap cell (original Phase 7 path);
//!   ordinal x + quantitative y + y2 → vertical band rect (boxplot box body,
//!   Phase 10c-pre);
//!   quantitative x + x2 + y + y2 → free-floating rect with explicit
//!   pixel bounds (Phase 10f, silhouette + decision-boundary). Dispatch:
//!   both x2 & y2 present and both axes quantitative → quantitative-range
//!   path; only y2 present → ordinal-range; else heatmap.

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_positional_category_str, col_as_str, color_field, resolve_effective_stroke, resolve_fill_color, x_field, y_field, DrawCtx, MetadataColumns};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::opacity::{resolve_scaled_opacity, OpacityFallback, OpacityResolver};
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
    crate::render::draw::MarkBuildResult::empty(ferrum_scene::MarkBatchKind::Rect)
}

fn build_quantitative_range(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_fill_stroke_full};
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
    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    // fill_opacity via the shared resolver (FA-11), sampled per-row. opacity is
    // scale-mapped at the call site below; stroke_opacity is not read by rect.
    let opacity_res = OpacityResolver::load(ctx, OpacityFallback::Standard, (ctx.mark_style.paint.opacity, 1.0, 1.0));
    let stroke_width_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_width
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep so metadata is
    // aligned to kept nodes only. Rows are skipped for null/non-finite values
    // and scale-resolution failures (#6 defect class fix).
    let mut acc = MarkNodes::with_capacity(n);

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

        let fill = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_strings.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_numeric.as_ref().and_then(|v| v.get(i).copied().flatten()),
            ctx.mark_style.paint.fill,
        );
        // Resolve per-row opacity through scale if present; fall back to mark_style.paint.opacity.
        let row_opacity =
            resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);
        let fill = with_opacity(fill, row_opacity);

        let (_, row_fill_opacity, _) = opacity_res.at_row(i);

        let row_stroke_width = stroke_width_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| *v >= 0.0 && v.is_finite())
            .unwrap_or(ctx.mark_style.paint.stroke_width);

        // When stroke_width encoding produces a positive value but no explicit
        // stroke color exists, use the fill color as the stroke so the width is
        // visible in SVG (stroke-width is only emitted when stroke is Some).
        let effective_stroke = resolve_effective_stroke(
            row_stroke_width,
            ctx.mark_style.paint.stroke,
            fill,
            stroke_width_values.is_some(),
        );

        let style = to_scene_fill_stroke_full(
            Some(fill),
            effective_stroke,
            row_stroke_width,
            row_opacity,
            ctx.mark_style.paint.stroke_dash.as_deref(),
            row_fill_opacity,
            1.0,
            0.0,
        );

        acc.push(SceneNode::Rect {
            x: px_left,
            y: py_top,
            w,
            h,
            style,
            corner_radius: ctx.mark_style.paint.corner_radius,
        }, i);
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Rect,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
    }
}

fn build_ordinal_range(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_fill_stroke_full};
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
    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    // fill_opacity via the shared resolver (FA-11), sampled per-row. opacity is
    // scale-mapped at the call sites below; stroke_opacity is not read by rect.
    let opacity_res = OpacityResolver::load(ctx, OpacityFallback::Standard, (ctx.mark_style.paint.opacity, 1.0, 1.0));
    let stroke_width_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_width
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep so metadata is
    // aligned to kept nodes only. Rows are skipped for null categories, non-finite
    // values, and scale-resolution failures (#6 defect class fix).
    let mut acc = MarkNodes::with_capacity(32);

    if x_is_ordinal {
        // Normal orientation: x is categorical band, y and y2 are quantitative extents.
        let y2f = match spec.encoding.y2.as_ref().map(|e| e.field.as_str()) {
            Some(f) => f, None => return empty_result(),
        };
        let n_categories = {
            let xs_probe = match col_as_positional_category_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
            count_distinct(&xs_probe).max(1)
        };
        let xs = match col_as_positional_category_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
        let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
        let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return empty_result() };
        if xs.len() != ys.len() || y2s.len() != ys.len() { return empty_result(); }
        // Under an ordinal-band Dodge, shrink each box body to its sub-band so
        // adjacent dodge groups don't overlap. No Dodge → n_groups == 1 →
        // byte-identical to the non-dodged box width.
        let n_groups = crate::render::position::n_dodge_groups(ctx.batch);
        // Band-geometry unification (#39 phase 2): honor an explicit x-band
        // pixel range when the resolver recorded one; otherwise `panel.w`.
        let band_extent = crate::render::marks::channels::band_extent_or(&ctx.scales.x, panel.w);
        let box_w = (band_extent / n_categories as f64 / n_groups as f64)
            * ctx.mark_style.misc.band_size.unwrap_or(0.6);

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

            // Ordinal rects bind only a categorical color column (no continuous
            // path is loaded here), so `num_value` is always None.
            let fill = resolve_fill_color(
                ctx.scales.color.as_ref(),
                color_strings.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
                None,
                ctx.mark_style.paint.fill,
            );
            let row_opacity =
                resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);
            let fill = with_opacity(fill, row_opacity);

            let (_, row_fill_opacity, _) = opacity_res.at_row(i);

            let row_stroke_width = stroke_width_values
                .as_ref()
                .and_then(|v| v[i])
                .filter(|v| *v >= 0.0 && v.is_finite())
                .unwrap_or(ctx.mark_style.paint.stroke_width);

            let effective_stroke = resolve_effective_stroke(
                row_stroke_width,
                ctx.mark_style.paint.stroke,
                fill,
                stroke_width_values.is_some(),
            );

            let style = to_scene_fill_stroke_full(
                Some(fill),
                effective_stroke,
                row_stroke_width,
                row_opacity,
                ctx.mark_style.paint.stroke_dash.as_deref(),
                row_fill_opacity,
                1.0,
                0.0,
            );

            acc.push(SceneNode::Rect {
                x: cx - box_w / 2.0,
                y: rect_top,
                w: box_w,
                h: rect_h,
                style,
                corner_radius: ctx.mark_style.paint.corner_radius,
            }, i);
        }
    } else if y_is_ordinal {
        // CoordFlip orientation: y is categorical band, x and x2 are quantitative extents.
        let x2f = match spec.encoding.x2.as_ref().map(|e| e.field.as_str()) {
            Some(f) => f, None => return empty_result(),
        };
        let n_categories = {
            let ys_probe = match col_as_positional_category_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
            count_distinct(&ys_probe).max(1)
        };
        let ys = match col_as_positional_category_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
        let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
        let x2s = match col_as_f64(ctx.batch, x2f) { Ok(v) => v, Err(_) => return empty_result() };
        if ys.len() != xs.len() || x2s.len() != xs.len() { return empty_result(); }
        // Under an ordinal-band Dodge (CoordFlip orientation), shrink each box
        // body to its sub-band so adjacent dodge groups don't overlap. No Dodge
        // → n_groups == 1 → byte-identical to the non-dodged box height.
        let n_groups = crate::render::position::n_dodge_groups(ctx.batch);
        // Band-geometry unification (#39 phase 2): honor an explicit y-band
        // pixel range when the resolver recorded one; otherwise `panel.h`.
        let band_extent = crate::render::marks::channels::band_extent_or(&ctx.scales.y, panel.h);
        let box_h = (band_extent / n_categories as f64 / n_groups as f64)
            * ctx.mark_style.misc.band_size.unwrap_or(0.6);

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

            // Ordinal rects bind only a categorical color column (no continuous
            // path is loaded here), so `num_value` is always None.
            let fill = resolve_fill_color(
                ctx.scales.color.as_ref(),
                color_strings.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
                None,
                ctx.mark_style.paint.fill,
            );
            let row_opacity =
                resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);
            let fill = with_opacity(fill, row_opacity);

            let (_, row_fill_opacity, _) = opacity_res.at_row(i);

            let row_stroke_width = stroke_width_values
                .as_ref()
                .and_then(|v| v[i])
                .filter(|v| *v >= 0.0 && v.is_finite())
                .unwrap_or(ctx.mark_style.paint.stroke_width);

            let effective_stroke = resolve_effective_stroke(
                row_stroke_width,
                ctx.mark_style.paint.stroke,
                fill,
                stroke_width_values.is_some(),
            );

            let style = to_scene_fill_stroke_full(
                Some(fill),
                effective_stroke,
                row_stroke_width,
                row_opacity,
                ctx.mark_style.paint.stroke_dash.as_deref(),
                row_fill_opacity,
                1.0,
                0.0,
            );

            acc.push(SceneNode::Rect {
                x: rect_left,
                y: cy - box_h / 2.0,
                w: rect_w,
                h: box_h,
                style,
                corner_radius: ctx.mark_style.paint.corner_radius,
            }, i);
        }
    } else {
        return empty_result();
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Rect,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
    }
}

fn build_heatmap(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_fill_stroke_full, to_scene_text_style};
    use crate::render::format::format_numeric;
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return empty_result(),
    };
    let xs = match col_as_positional_category_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty_result() };
    let ys = match col_as_positional_category_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty_result() };
    if xs.len() != ys.len() { return empty_result(); }

    let panel = ctx.panel.plot_area;
    let n_x = match &ctx.scales.x { ScaleKind::Ordinal(_) => count_distinct(&xs).max(1), _ => return empty_result() };
    let n_y = match &ctx.scales.y { ScaleKind::Ordinal(_) => count_distinct(&ys).max(1), _ => return empty_result() };
    // Band-geometry unification (#39 phase 2): heatmap cells honor an explicit
    // band pixel range per axis when the resolver recorded one; otherwise
    // identical to `panel.w` / `panel.h`. Denominators stay per-axis category
    // counts (`n_x` / `n_y`), not `n_categories`/`n_groups` — only the extent
    // term changes.
    let cell_w = crate::render::marks::channels::band_extent_or(&ctx.scales.x, panel.w) / n_x as f64;
    let cell_h = crate::render::marks::channels::band_extent_or(&ctx.scales.y, panel.h) / n_y as f64;

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

    // Per-row encoding channels: opacity (scale-mapped at call site),
    // fill_opacity (shared resolver, FA-11), stroke_width.
    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let opacity_res = OpacityResolver::load(ctx, OpacityFallback::Standard, (ctx.mark_style.paint.opacity, 1.0, 1.0));
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
                    .map(|v| v.map(format_numeric))
                    .collect()
            })
        })
    });

    // Accumulate nodes and source-row indices in lockstep so metadata is
    // aligned to kept nodes only. Rows are skipped for null categories and
    // scale-resolution failures (#6 defect class fix).
    // Text annotation nodes share the same source row as their rect node, so
    // both the rect and its text label are pushed with the same `i`.
    let mut acc = MarkNodes::with_capacity(xs.len() * 2);

    for i in 0..xs.len() {
        let xs_v = match &xs[i] { Some(s) => s.as_str(), None => continue };
        let ys_v = match &ys[i] { Some(s) => s.as_str(), None => continue };
        let cx = match ctx.scales.x.to_pixel_str(xs_v) { Some(p) => p, None => continue };
        let cy = match ctx.scales.y.to_pixel_str(ys_v) { Some(p) => p, None => continue };
        let cx = cx + x_offsets[i];
        let cy = cy + y_offsets[i];

        let fill = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_strings.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_numeric.as_ref().and_then(|v| v.get(i).copied().flatten()),
            ctx.mark_style.paint.fill,
        );
        // Resolve per-row opacity (through scale if present).
        let row_opacity =
            resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);

        let fill = with_opacity(fill, row_opacity);

        let (_, row_fill_opacity, _) = opacity_res.at_row(i);

        let row_stroke_width = stroke_width_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| *v >= 0.0 && v.is_finite())
            .unwrap_or(ctx.mark_style.paint.stroke_width);

        // When stroke_width encoding produces a positive value but no explicit
        // stroke color exists, use the fill color as the stroke so the width is
        // visible in SVG (stroke-width is only emitted when stroke is Some).
        let effective_stroke = resolve_effective_stroke(
            row_stroke_width,
            ctx.mark_style.paint.stroke,
            fill,
            stroke_width_values.is_some(),
        );

        let style = to_scene_fill_stroke_full(
            Some(fill),
            effective_stroke,
            row_stroke_width,
            row_opacity,
            ctx.mark_style.paint.stroke_dash.as_deref(),
            row_fill_opacity,
            1.0,
            0.0,
        );

        acc.push(SceneNode::Rect {
            x: cx - cell_w / 2.0,
            y: cy - cell_h / 2.0,
            w: cell_w,
            h: cell_h,
            style,
            corner_radius: ctx.mark_style.paint.corner_radius,
        }, i);

        // Emit text annotation at cell center when text encoding is present.
        // The text node shares the same source row `i` so node+metadata stay
        // aligned even when rows are skipped (#6 defect class fix).
        if let Some(ref texts) = text_values {
            if let Some(Some(content)) = texts.get(i) {
                if !content.is_empty() {
                    let text_color = ctx.theme.colors.font_color;
                    let font_size = ctx.mark_style.text.font_size.unwrap_or(11.0);
                    acc.push(SceneNode::Text {
                        x: cx,
                        y: cy,
                        content: content.clone(),
                        slot: None,
                        style: to_scene_text_style(
                            text_color,
                            font_size,
                            crate::layout::TextAnchor::Middle,
                            0.0,
                            &ctx.theme.typography.font_family,
                            None,
                            Some("central"),
                            1.0,
                        ),
                    }, i);
                }
            }
        }
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Rect,
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
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::{resolve_mark_style, DrawCtx};
    use crate::render::scale_resolve::{resolve_scales, OpacityScale, ResolvedScales, ScaleKind};
    use crate::scale::linear::LinearScale;
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
        params: Vec::new(),
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
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
        params: Vec::new(),
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
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
        params: Vec::new(),
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
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
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
        params: Vec::new(),
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
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

    // --- W18: opacity encoding must be applied per-row ---

    /// W18: build_quantitative_range must read the opacity encoding and apply it
    /// per-row. Two rows with different opacity values must produce different
    /// alpha values in their FillStroke fill color.
    #[test]
    fn w18_rect_quantitative_range_applies_per_row_opacity() {
        use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
        use crate::scale::linear::LinearScale;
        use arrow::array::Float64Array;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                opacity: Some(EncodingSpec { field: "op".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None, mark_style: None,
            position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None, params: Vec::new(),
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("x",  DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y",  DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("op", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
            Arc::new(Float64Array::from(vec![1.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.2, 0.8])),  // distinct opacity values
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };

        // Build scales manually with an opacity scale so it maps [0,1] → [0,1].
        use crate::render::scale_resolve::OpacityScale;
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 2.0], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 1.0], vec![100.0, 0.0], false, false)),
            color: None,
            size: None,
            shape: None,
            opacity: Some(OpacityScale {
                inner: ScaleKind::Linear(LinearScale::new_internal(vec![0.2, 0.8], vec![0.2, 0.8], false, false)),
            }),
            x2: None,
            y2: None,
            y_slots: Default::default(),
        };

        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let rects: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { style, .. } = n { Some(style.clone()) } else { None }
        }).collect();
        assert_eq!(rects.len(), 2, "expected 2 rects");

        // The two rects must have different fill alpha (opacity applied per-row).
        // ferrum_scene::Color uses .a for alpha.
        let alpha0 = rects[0].fill.as_ref().map(|c| c.a).unwrap_or(0);
        let alpha1 = rects[1].fill.as_ref().map(|c| c.a).unwrap_or(0);
        assert_ne!(
            alpha0, alpha1,
            "per-row opacity encoding must produce different alphas; both were {alpha0}"
        );
    }

    /// W18: build_heatmap also already reads opacity encoding (Phase 10 added it).
    /// This test verifies that ordinal-range path also reads opacity encoding.
    #[test]
    fn w18_rect_ordinal_range_applies_per_row_opacity() {
        use crate::render::scale_resolve::{OpacityScale, ResolvedScales, ScaleKind};
        use crate::scale::linear::LinearScale;
        use arrow::array::Float64Array;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "q1".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "q3".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                opacity: Some(EncodingSpec { field: "op".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None, mark_style: None,
            position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None, params: Vec::new(),
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("q1",  DataType::Float64, false),
            Field::new("q3",  DataType::Float64, false),
            Field::new("op",  DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 3.0])),
            Arc::new(Float64Array::from(vec![2.0, 4.0])),
            Arc::new(Float64Array::from(vec![0.2, 0.9])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let scales_default = ResolvedScales {
            x: ScaleKind::Ordinal(crate::scale::ordinal::OrdinalScale::new_internal(vec!["a".into(), "b".into()], vec![25.0, 75.0], 0.0)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![1.0, 4.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None,
            opacity: Some(OpacityScale {
                inner: ScaleKind::Linear(LinearScale::new_internal(vec![0.2, 0.9], vec![0.2, 0.9], false, false)),
            }),
            x2: None, y2: None,
            y_slots: Default::default(),
        };

        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales_default, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let rects: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { style, .. } = n { Some(style.clone()) } else { None }
        }).collect();
        assert_eq!(rects.len(), 2, "expected 2 ordinal-range rects");

        // ferrum_scene::Color uses .a for alpha.
        let alpha0 = rects[0].fill.as_ref().map(|c| c.a).unwrap_or(0);
        let alpha1 = rects[1].fill.as_ref().map(|c| c.a).unwrap_or(0);
        assert_ne!(
            alpha0, alpha1,
            "ordinal-range opacity encoding must produce different alphas; both were {alpha0}"
        );
    }

    /// D4 regression: Int64-keyed heatmap must render the same number of cells
    /// as a string-keyed heatmap with identical structure. Previously `col_as_str`
    /// rejected Int64 columns and returned an empty result.
    #[test]
    fn rect_int64_keyed_heatmap_renders_cells() {
        use arrow::array::Int64Array;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "row".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "col".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
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
            Field::new("row", DataType::Int64, false),
            Field::new("col", DataType::Int64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(vec![2000i64, 2000i64, 2001i64, 2001i64])),
            Arc::new(Int64Array::from(vec![1i64, 2i64, 1i64, 2i64])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &crate::layout::ThemeInputs::default(),
        ).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(
            result.nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count(),
            4,
            "Int64-keyed heatmap must render 4 cells (was returning 0 before D4 fix)",
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
        params: Vec::new(),
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert!(result.nodes.iter().all(|n| !matches!(n, SceneNode::Rect { .. })));
    }

    // ── Metadata-alignment regression tests (#6 defect class) ────────────────
    //
    // Each test creates a batch where a middle row is skipped (null / degenerate)
    // and asserts that tooltip metadata on each emitted node points to its TRUE
    // source row, not its node-position row.
    //
    // Fail-before: prior to this migration the three rect builders called
    // `meta.build_metadata(ctx)` (full per-row vectors) before the loop.
    // When row 1 (of 3) was skipped, node 1 received row 1's tooltip instead of
    // row 2's — the #6 defect class. These tests fail on that old code: the second
    // node's tooltip would be "tip_b" (row 1, skipped) instead of "tip_c" (row 2,
    // the true source row of node 1).
    //
    // Pass-after: migrated builders use MarkNodes + build_metadata_for_indices so
    // node j always receives its true source row's metadata.

    /// Alignment test for `build_quantitative_range`.
    ///
    /// Batch: 3 rows, x=[0,1,2], x2=[1,2,3], y=[0,0,0], y2=[1,null,3].
    /// Row 1 has a null y2 → skipped. 2 nodes survive: node 0 → row 0 → "tip_a",
    /// node 1 → row 2 → "tip_c". Old code (build_metadata): node 1 → "tip_b".
    #[test]
    fn quantitative_range_skipped_null_y2_tooltip_aligned() {
        use arrow::array::Float64Array;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "x".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y:  Some(EncodingSpec { field: "y".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
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
            Field::new("x2",  DataType::Float64, false),
            Field::new("y",   DataType::Float64, false),
            Field::new("y2",  DataType::Float64, true),   // nullable — row 1 null → skip
            Field::new("tip", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0_f64,  1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0_f64,  2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0_f64,  0.0, 0.0])),
            Arc::new(Float64Array::from(vec![Some(1.0_f64), None, Some(3.0)])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 300.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // 2 nodes survive (row 1 with null y2 is skipped).
        assert_eq!(result.nodes.len(), 2,
            "expected 2 rects after null-y2 skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2,
            "tooltip count ({}) must equal node count (2)", tooltips.len());

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be row 0's ('tip_a'); got '{t0}'");

        // Node 1 → true source row 2 → "tip_c".
        // Old code (full-row indexing): node 1 → row 1 → "tip_b" (the alignment bug).
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be row 2's ('tip_c'), not row 1's ('tip_b'); \
             got '{t1}'. Fails on pre-migration code using build_metadata(ctx).");
    }

    /// Alignment test for `build_ordinal_range` (x-ordinal orientation).
    ///
    /// Batch: 3 rows, cat=["a","b","c"], q1=[1,2,3], q3=[4,null,6].
    /// Row 1 has a null q3 → skipped. 2 nodes survive: node 0 → row 0 → "tip_a",
    /// node 1 → row 2 → "tip_c". Old code: node 1 → "tip_b".
    #[test]
    fn ordinal_range_skipped_null_q3_tooltip_aligned() {
        use arrow::array::Float64Array;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal),      ..Default::default() }),
                y:  Some(EncodingSpec { field: "q1".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "q3".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
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
            Field::new("q1",  DataType::Float64, false),
            Field::new("q3",  DataType::Float64, true),   // nullable — row 1 null → skip
            Field::new("tip", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![Some(4.0_f64), None, Some(6.0)])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 300.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // 2 nodes survive (row 1 with null q3 is skipped).
        assert_eq!(result.nodes.len(), 2,
            "expected 2 ordinal-range rects after null-q3 skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2,
            "tooltip count ({}) must equal node count (2)", tooltips.len());

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be row 0's ('tip_a'); got '{t0}'");

        // Node 1 → true source row 2 → "tip_c".
        // Old code (full-row indexing via build_metadata): node 1 → row 1 → "tip_b".
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be row 2's ('tip_c'), not row 1's ('tip_b'); \
             got '{t1}'. Fails on pre-migration code using build_metadata(ctx).");
    }

    /// Alignment test for `build_heatmap`.
    ///
    /// The heatmap builder skips rows where the x or y category can't be resolved
    /// by the ordinal scale (i.e. the category isn't in the scale's domain). This
    /// is triggered by supplying a manually-constructed scale whose x domain
    /// includes only ["A", "C"] — so row 1 ("B") fails the `to_pixel_str` lookup
    /// and is skipped.
    ///
    /// 2 rect nodes survive: node 0 → row 0 → "tip_a",
    /// node 1 → row 2 → "tip_c". Old code (build_metadata): node 1 → "tip_b".
    #[test]
    fn heatmap_skipped_unlookup_x_tooltip_aligned() {
        use crate::render::scale_resolve::ResolvedScales;
        use crate::scale::ordinal::OrdinalScale;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "xcat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "ycat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
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
            Field::new("xcat", DataType::Utf8, false),
            Field::new("ycat", DataType::Utf8, false),
            Field::new("tip",  DataType::Utf8, false),
        ]));
        // Row 1 has x="B" which is absent from the x-scale domain → skipped.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["A", "B", "C"])),
            Arc::new(StringArray::from(vec!["p", "p", "p"])),  // same y for simplicity
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };

        // Manually build scales so x-domain contains only ["A", "C"], not "B".
        // Row 1 ("B") will fail to_pixel_str → continue (skipped).
        let scales = ResolvedScales {
            x: ScaleKind::Ordinal(OrdinalScale::new_internal(
                vec!["A".into(), "C".into()],
                vec![50.0, 250.0],
                0.0,
            )),
            y: ScaleKind::Ordinal(OrdinalScale::new_internal(
                vec!["p".into()],
                vec![150.0],
                0.0,
            )),
            color: None, size: None, shape: None, opacity: None, x2: None, y2: None, y_slots: Default::default(),
        };

        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // 2 rect nodes survive (row 1 with x="B" not in domain is skipped).
        let rect_count = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Rect { .. }))
            .count();
        assert_eq!(rect_count, 2,
            "expected 2 heatmap cells after unlookup-x skip; got {rect_count}");

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), result.nodes.len(),
            "tooltip count ({}) must equal total node count ({})",
            tooltips.len(), result.nodes.len());

        // Node 0 is the first rect → source row 0 → "tip_a".
        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be row 0's ('tip_a'); got '{t0}'");

        // Node 1 is the second rect → source row 2 → "tip_c".
        // Old code (full-row build_metadata): node 1 → row 1 → "tip_b".
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be row 2's ('tip_c'), not row 1's ('tip_b'); \
             got '{t1}'. Fails on pre-migration code using build_metadata(ctx).");
    }

    /// Href channel alignment test for `build_quantitative_range`.
    ///
    /// The href channel is populated independently of tooltips; a row-skip that
    /// misaligns tooltips misaligns hrefs too. This test exercises the href path
    /// explicitly to confirm both channels are aligned by the accumulator.
    ///
    /// Batch: 3 rows, row 1 has null y2 → skipped. 2 nodes survive.
    /// Node 0 → row 0 → href "url_a"; node 1 → row 2 → href "url_c".
    #[test]
    fn quantitative_range_skipped_row_href_aligned() {
        use arrow::array::Float64Array;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "x".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y:  Some(EncodingSpec { field: "y".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                href: Some(EncodingSpec { field: "href".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",    DataType::Float64, false),
            Field::new("x2",   DataType::Float64, false),
            Field::new("y",    DataType::Float64, false),
            Field::new("y2",   DataType::Float64, true),   // nullable — row 1 null → skip
            Field::new("href", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0_f64,  1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0_f64,  2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0_f64,  0.0, 0.0])),
            Arc::new(Float64Array::from(vec![Some(1.0_f64), None, Some(3.0)])),
            Arc::new(StringArray::from(vec!["url_a", "url_b", "url_c"])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 300.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2,
            "expected 2 rects after null-y2 skip; got {}", result.nodes.len());

        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), 2,
            "href count ({}) must equal node count (2)", hrefs.len());

        let h0 = hrefs[0].as_deref().unwrap_or("");
        assert_eq!(h0, "url_a", "node 0 href must be row 0's ('url_a'); got '{h0}'");

        // Node 1 → true source row 2 → "url_c".
        // Old code would give "url_b" (row 1, the skipped row).
        let h1 = hrefs[1].as_deref().unwrap_or("");
        assert_eq!(h1, "url_c",
            "node 1 href must be row 2's ('url_c'), not row 1's ('url_b'); \
             got '{h1}'. Fails on pre-migration code using build_metadata(ctx).");
    }

    /// No-skip backward-compat test for all three rect builders.
    ///
    /// When no rows are skipped, `build_metadata_for_indices` on a full
    /// `[0, 1, 2, ...]` index list produces the same metadata as `build_metadata`
    /// did — so existing correct charts render byte-identically.
    #[test]
    fn no_skip_backward_compat_all_three_builders() {
        use arrow::array::Float64Array;

        // — build_quantitative_range: 3 rows, all kept —
        let spec_qr = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "x".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y:  Some(EncodingSpec { field: "y".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema_qr = Arc::new(Schema::new(vec![
            Field::new("x",   DataType::Float64, false),
            Field::new("x2",  DataType::Float64, false),
            Field::new("y",   DataType::Float64, false),
            Field::new("y2",  DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        let batch_qr = arrow::record_batch::RecordBatch::try_new(schema_qr, vec![
            Arc::new(Float64Array::from(vec![0.0_f64, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0_f64, 0.0, 0.0])),
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0, 3.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales_qr, _) = resolve_scales(&spec_qr, &batch_qr, (0.0, 300.0), (0.0, 300.0), &theme).unwrap();
        let mark_style_qr = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx_qr = DrawCtx { spec: &spec_qr, panel: &panel, theme: &theme, scales: &scales_qr, batch: &batch_qr, mark_style: &mark_style_qr };
        let result_qr = super::build(&ctx_qr);
        assert_eq!(result_qr.nodes.len(), 3, "qr: all 3 rows kept → 3 nodes");
        let tt_qr = result_qr.tooltips.expect("tooltips present");
        assert_eq!(tt_qr[0].fields[0].value, "tip_a");
        assert_eq!(tt_qr[1].fields[0].value, "tip_b");
        assert_eq!(tt_qr[2].fields[0].value, "tip_c");

        // — build_ordinal_range: 3 rows, all kept —
        let spec_or = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal),      ..Default::default() }),
                y:  Some(EncodingSpec { field: "q1".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "q3".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema_or = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8,    false),
            Field::new("q1",  DataType::Float64, false),
            Field::new("q3",  DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        let batch_or = arrow::record_batch::RecordBatch::try_new(schema_or, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![4.0_f64, 5.0, 6.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let (scales_or, _) = resolve_scales(&spec_or, &batch_or, (0.0, 300.0), (0.0, 300.0), &theme).unwrap();
        let mark_style_or = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx_or = DrawCtx { spec: &spec_or, panel: &panel, theme: &theme, scales: &scales_or, batch: &batch_or, mark_style: &mark_style_or };
        let result_or = super::build(&ctx_or);
        assert_eq!(result_or.nodes.len(), 3, "or: all 3 rows kept → 3 nodes");
        let tt_or = result_or.tooltips.expect("tooltips present");
        assert_eq!(tt_or[0].fields[0].value, "tip_a");
        assert_eq!(tt_or[1].fields[0].value, "tip_b");
        assert_eq!(tt_or[2].fields[0].value, "tip_c");

        // — build_heatmap: 3 rows, all kept, no text labels —
        let spec_hm = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "xcat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "ycat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema_hm = Arc::new(Schema::new(vec![
            Field::new("xcat", DataType::Utf8, false),
            Field::new("ycat", DataType::Utf8, false),
            Field::new("tip",  DataType::Utf8, false),
        ]));
        let batch_hm = arrow::record_batch::RecordBatch::try_new(schema_hm, vec![
            Arc::new(StringArray::from(vec!["A", "B", "C"])),
            Arc::new(StringArray::from(vec!["p", "q", "r"])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let (scales_hm, _) = resolve_scales(&spec_hm, &batch_hm, (0.0, 300.0), (0.0, 300.0), &theme).unwrap();
        let mark_style_hm = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx_hm = DrawCtx { spec: &spec_hm, panel: &panel, theme: &theme, scales: &scales_hm, batch: &batch_hm, mark_style: &mark_style_hm };
        let result_hm = super::build(&ctx_hm);
        assert_eq!(result_hm.nodes.len(), 3, "hm: all 3 rows kept → 3 rect nodes");
        let tt_hm = result_hm.tooltips.expect("tooltips present");
        assert_eq!(tt_hm.len(), 3, "hm: tooltip count must equal node count");
        assert_eq!(tt_hm[0].fields[0].value, "tip_a");
        assert_eq!(tt_hm[1].fields[0].value, "tip_b");
        assert_eq!(tt_hm[2].fields[0].value, "tip_c");
    }

    /// Alignment test for `build_heatmap` with a text (label) encoding AND a
    /// tooltip encoding, with one row skipped via unlookup-x.
    ///
    /// The heatmap builder emits 2 nodes per labeled cell (a Rect + a Text).
    /// This test pins the secondary #6 fix where both nodes are pushed with the
    /// same source-row index `i`, so metadata for the Text node aligns to the
    /// same true source row as its Rect sibling — not to the node-position index.
    ///
    /// Batch: 3 rows, x=["A","B","C"], y=["p","p","p"], text=["cell_a","cell_b","cell_c"],
    /// tip=["tip_a","tip_b","tip_c"].
    /// x-scale domain = ["A","C"] only → row 1 ("B") fails to_pixel_str → skipped.
    ///
    /// Surviving 4 nodes: rect(row 0), text(row 0), rect(row 2), text(row 2).
    /// Assertions:
    ///   (a) tooltips.len() == nodes.len() == 4
    ///   (b) each node's tooltip is its true source row's string, not its
    ///       node-position index's string.
    ///
    /// Old code would have placed tooltip[2] = "tip_b" (node-position 2 → row 1,
    /// the skipped row) instead of "tip_c" (true source row 2).
    #[test]
    fn heatmap_text_label_and_tooltip_skipped_row_aligned() {
        use crate::render::scale_resolve::ResolvedScales;
        use crate::scale::ordinal::OrdinalScale;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x:       Some(EncodingSpec { field: "xcat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y:       Some(EncodingSpec { field: "ycat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                text:    Some(EncodingSpec { field: "label".into(), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(),   ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("xcat",  DataType::Utf8, false),
            Field::new("ycat",  DataType::Utf8, false),
            Field::new("label", DataType::Utf8, false),
            Field::new("tip",   DataType::Utf8, false),
        ]));
        // Row 1 has x="B" which is absent from the x-scale domain → skipped.
        // Tooltip strings are distinct so node-position vs. source-row indexing is
        // detectably wrong: old code would emit "tip_b" instead of "tip_c".
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["A", "B", "C"])),
            Arc::new(StringArray::from(vec!["p", "p", "p"])),
            Arc::new(StringArray::from(vec!["cell_a", "cell_b", "cell_c"])),
            Arc::new(StringArray::from(vec!["tip_a",  "tip_b",  "tip_c"])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };

        // x-domain excludes "B" → row 1 fails to_pixel_str → skipped.
        let scales = ResolvedScales {
            x: ScaleKind::Ordinal(OrdinalScale::new_internal(
                vec!["A".into(), "C".into()],
                vec![50.0, 250.0],
                0.0,
            )),
            y: ScaleKind::Ordinal(OrdinalScale::new_internal(
                vec!["p".into()],
                vec![150.0],
                0.0,
            )),
            color: None, size: None, shape: None, opacity: None, x2: None, y2: None, y_slots: Default::default(),
        };

        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // 2 rows kept (rows 0 and 2), each emitting rect + text → 4 nodes total.
        assert_eq!(result.nodes.len(), 4,
            "expected 4 nodes (2 rects + 2 text labels); got {}", result.nodes.len());

        let rect_count = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Rect { .. }))
            .count();
        assert_eq!(rect_count, 2, "expected 2 rect nodes among the 4");

        // (a) tooltips.len() must equal node count (2 per kept cell = 4).
        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 4,
            "tooltip count ({}) must equal total node count (4); \
             old code omitted Text-node metadata entries", tooltips.len());

        // (b) Node layout: [rect(row0), text(row0), rect(row2), text(row2)].
        // Every node must carry its true source row's tooltip.
        let t0 = &tooltips[0].fields[0].value;
        let t1 = &tooltips[1].fields[0].value;
        let t2 = &tooltips[2].fields[0].value;
        let t3 = &tooltips[3].fields[0].value;

        assert_eq!(t0, "tip_a",
            "node 0 (rect, row 0) tooltip must be 'tip_a'; got '{t0}'");
        assert_eq!(t1, "tip_a",
            "node 1 (text, row 0) tooltip must be 'tip_a'; got '{t1}'");
        // Old code (node-position indexing): node 2 → position 2 → row 1 → "tip_b".
        assert_eq!(t2, "tip_c",
            "node 2 (rect, row 2) tooltip must be 'tip_c', not 'tip_b' (skipped row 1); \
             got '{t2}'");
        assert_eq!(t3, "tip_c",
            "node 3 (text, row 2) tooltip must be 'tip_c', not 'tip_b' (skipped row 1); \
             got '{t3}'");
    }

    /// Alignment test for `build_ordinal_range` — CoordFlip (y-ordinal) branch with
    /// a skipped middle row.
    ///
    /// The existing `ordinal_range_skipped_null_q3_tooltip_aligned` test only covers
    /// the x-ordinal branch (null y2 → skip). This test covers the mirror branch:
    /// y-ordinal + x/x2 quantitative (CoordFlip / horizontal bar), with row 1 having
    /// a null x2 → skip.
    ///
    /// Batch: 3 rows, cat=["a","b","c"], x1=[0,0,0], x2=[4,null,6].
    /// Row 1 has null x2 → skipped. 2 nodes survive:
    ///   node 0 → row 0 → "tip_a"
    ///   node 1 → row 2 → "tip_c"
    ///
    /// Old code (build_metadata full-row indexing): node 1 → row 1 → "tip_b".
    #[test]
    fn ordinal_range_coordflip_y_ordinal_skipped_null_x2_tooltip_aligned() {
        use arrow::array::Float64Array;

        // y-ordinal + x + x2 → CoordFlip branch of build_ordinal_range.
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                y:  Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal),      ..Default::default() }),
                x:  Some(EncodingSpec { field: "x1".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
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
            Field::new("x1",  DataType::Float64, false),
            Field::new("x2",  DataType::Float64, true),   // nullable — row 1 null → skip
            Field::new("tip", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![0.0_f64, 0.0, 0.0])),
            Arc::new(Float64Array::from(vec![Some(4.0_f64), None, Some(6.0)])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 300.0), (0.0, 300.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // 2 nodes survive (row 1 with null x2 is skipped).
        assert_eq!(result.nodes.len(), 2,
            "expected 2 CoordFlip ordinal-range rects after null-x2 skip; got {}",
            result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2,
            "tooltip count ({}) must equal node count (2)", tooltips.len());

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be row 0's ('tip_a'); got '{t0}'");

        // Node 1 → true source row 2 → "tip_c".
        // Old code (full-row build_metadata): node 1 → row 1 → "tip_b".
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be row 2's ('tip_c'), not row 1's ('tip_b'); \
             got '{t1}'. Fails on pre-migration code using build_metadata(ctx).");
    }

    // ── Dodge box-body narrowing (Task 3c) ───────────────────────────────────

    /// Ordinal-x box-body spec (`mark_boxplot(position=Dodge(...))`) plus a batch
    /// that already carries the synthetic Dodge offset columns and (optionally)
    /// the `__dodge_n_groups__` schema-metadata key `n_dodge_groups` reads.
    /// `x_offsets` supplies `__pos_x_offset__`; `__pos_y_offset__` is all-zero.
    fn ordinal_box_dodge_ctx_batch(
        x_offsets: Option<Vec<f64>>,
        n_groups_metadata: Option<usize>,
    ) -> (ChartSpec, arrow::record_batch::RecordBatch) {
        use arrow::array::Float64Array;
        use std::collections::HashMap;
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
            chart_description: None, params: Vec::new(),
        };
        // Two categories (a, b), two dodge groups per category (4 rows).
        let mut fields = vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("q1", DataType::Float64, false),
            Field::new("q3", DataType::Float64, false),
        ];
        let mut cols: Vec<arrow::array::ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
            Arc::new(Float64Array::from(vec![2.0, 3.0, 2.5, 3.5])),
            Arc::new(Float64Array::from(vec![6.0, 7.0, 6.5, 7.5])),
        ];
        if let Some(xo) = x_offsets {
            fields.push(Field::new("__pos_x_offset__", DataType::Float64, false));
            fields.push(Field::new("__pos_y_offset__", DataType::Float64, false));
            cols.push(Arc::new(Float64Array::from(xo.clone())));
            cols.push(Arc::new(Float64Array::from(vec![0.0; xo.len()])));
        }
        let mut schema = Schema::new(fields);
        if let Some(n) = n_groups_metadata {
            let mut metadata = HashMap::new();
            metadata.insert(crate::render::position::DODGE_N_GROUPS_KEY.to_string(), n.to_string());
            schema = schema.with_metadata(metadata);
        }
        let batch = arrow::record_batch::RecordBatch::try_new(Arc::new(schema), cols).unwrap();
        (spec, batch)
    }

    fn box_widths(spec: &ChartSpec, batch: &arrow::record_batch::RecordBatch) -> Vec<(f64, f64)> {
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(spec, batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec, panel: &panel, theme: &theme, scales: &scales, batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { x, w, .. } = n { Some((*x, *w)) } else { None }
        }).collect()
    }

    #[test]
    fn rect_ordinal_box_dodge_narrows_width_by_group_count() {
        // Task 3c: a dodged box body must shrink its band-dimension extent
        // (width) by the dodge group count. panel.w=100, 2 cats → bandwidth 50,
        // 2 groups → sub_band 25 → offsets -12.5 / +12.5; box_w = (100 / 2 / 2)
        // * band_size(0.6) = 15.0. Narrowing is driven by the explicit
        // __dodge_n_groups__ metadata (Some(2)), not distinct offset values.
        let (spec, batch) = ordinal_box_dodge_ctx_batch(Some(vec![-12.5, 12.5, -12.5, 12.5]), Some(2));
        let rects = box_widths(&spec, &batch);
        assert_eq!(rects.len(), 4, "expected 4 dodged box bodies");
        for (_, w) in &rects {
            assert!((w - 15.0).abs() < 1e-9, "dodged box width must be 15.0 (narrowed by 2 groups); got {w}");
        }
        // Adjacent dodge groups must not overlap in x.
        let mut ivals: Vec<(f64, f64)> = rects.iter().map(|(x, w)| (*x, *x + *w)).collect();
        ivals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for pair in ivals.windows(2) {
            assert!(pair[0].1 <= pair[1].0 + 1e-9,
                "dodged box intervals must not overlap: {:?} vs {:?}", pair[0], pair[1]);
        }
    }

    #[test]
    fn rect_ordinal_box_no_dodge_width_unchanged() {
        // Regression: no offset columns → n_groups == 1 → box_w is the full band
        // fraction (100 / 2) * 0.6 = 30.0, byte-identical to pre-Task-3c.
        let (spec, batch) = ordinal_box_dodge_ctx_batch(None, None);
        let rects = box_widths(&spec, &batch);
        assert_eq!(rects.len(), 4);
        for (_, w) in &rects {
            assert!((w - 30.0).abs() < 1e-9, "non-dodged box width must be 30.0; got {w}");
        }
    }

    // ── Band-geometry unification (#39 phase 2) ──────────────────────────────

    /// An explicit `BandScale` x-axis (extent 220px, distinct from panel.w =
    /// 300px) must drive the boxplot box body's `box_w` from the scale's
    /// extent, not `panel.w`. Fails on pre-Task-3 code, which always divides
    /// by `panel.w`.
    #[test]
    fn rect_ordinal_range_x_explicit_range_scales_box_width_by_extent() {
        use crate::render::scale_resolve::ResolvedScales;
        use crate::scale::linear::LinearScale;
        use crate::scale::ordinal::OrdinalScale;
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
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("q1", DataType::Float64, false),
            Field::new("q3", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![2.0, 4.0])),
            Arc::new(Float64Array::from(vec![6.0, 8.0])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };

        let x_scale = OrdinalScale::new_internal(vec!["a".into(), "b".into()], vec![40.0, 260.0], 0.0)
            .with_explicit_range(true);
        let scales = ResolvedScales {
            x: ScaleKind::Ordinal(x_scale),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![2.0, 8.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None,
            x2: None, y2: None, y_slots: Default::default(),
        };

        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let widths: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { w, .. } = n { Some(*w) } else { None }
        }).collect();
        assert_eq!(widths.len(), 2);
        // |260 - 40| = 220 extent; 2 categories, 1 group, band_size 0.6 (default)
        // → box_w = 220 / 2 * 0.6 = 66.0.
        for w in widths {
            assert!((w - 66.0).abs() < 1e-9, "expected box_w 66.0 from the 220px explicit extent, got {w}");
        }
    }

    /// CoordFlip orientation (y-ordinal + x/x2 quantitative): an explicit
    /// `BandScale` y-axis drives `box_h` from the scale's extent, not
    /// `panel.h`.
    #[test]
    fn rect_ordinal_range_y_explicit_range_scales_box_height_by_extent() {
        use crate::render::scale_resolve::ResolvedScales;
        use crate::scale::linear::LinearScale;
        use crate::scale::ordinal::OrdinalScale;
        use arrow::array::Float64Array;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "q1".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "q3".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("q1", DataType::Float64, false),
            Field::new("q3", DataType::Float64, false),
            Field::new("cat", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![2.0, 4.0])),
            Arc::new(Float64Array::from(vec![6.0, 8.0])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 300.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };

        let y_scale = OrdinalScale::new_internal(vec!["a".into(), "b".into()], vec![40.0, 260.0], 0.0)
            .with_explicit_range(true);
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![2.0, 8.0], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Ordinal(y_scale),
            color: None, size: None, shape: None, opacity: None,
            x2: None, y2: None, y_slots: Default::default(),
        };

        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let heights: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { h, .. } = n { Some(*h) } else { None }
        }).collect();
        assert_eq!(heights.len(), 2);
        // |260 - 40| = 220 extent; 2 categories, 1 group, band_size 0.6 (default)
        // → box_h = 220 / 2 * 0.6 = 66.0.
        for h in heights {
            assert!((h - 66.0).abs() < 1e-9, "expected box_h 66.0 from the 220px explicit extent, got {h}");
        }
    }

    /// Heatmap cell extent: an explicit `BandScale` range on x scales `cell_w`
    /// from the scale's extent, while the unranged y axis keeps deriving
    /// `cell_h` from `panel.h` (only the ranged axis's term changes).
    #[test]
    fn rect_heatmap_explicit_x_range_scales_cell_width_only() {
        use crate::render::scale_resolve::ResolvedScales;
        use crate::scale::ordinal::OrdinalScale;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "row".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "col".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("row", DataType::Utf8, false),
            Field::new("col", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
            Arc::new(StringArray::from(vec!["x", "y", "x", "y"])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 400.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };

        let x_scale = OrdinalScale::new_internal(vec!["a".into(), "b".into()], vec![40.0, 260.0], 0.0)
            .with_explicit_range(true);
        // y (col) keeps its ordinary panel-derived fallback range — no explicit flag.
        let y_scale = OrdinalScale::new_internal(vec!["x".into(), "y".into()], vec![0.0, 400.0], 0.0);
        let scales = ResolvedScales {
            x: ScaleKind::Ordinal(x_scale),
            y: ScaleKind::Ordinal(y_scale),
            color: None, size: None, shape: None, opacity: None,
            x2: None, y2: None, y_slots: Default::default(),
        };

        let mark_style = resolve_mark_style(None, &theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let dims: Vec<(f64, f64)> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { w, h, .. } = n { Some((*w, *h)) } else { None }
        }).collect();
        assert_eq!(dims.len(), 4, "expected 4 heatmap cells");
        for (w, h) in dims {
            // |260 - 40| = 220 extent / 2 row categories → cell_w = 110.0.
            assert!((w - 110.0).abs() < 1e-9, "expected cell_w 110.0 from the 220px explicit x extent, got {w}");
            // Unranged y: panel.h (400) / 2 col categories → cell_h = 200.0.
            assert!((h - 200.0).abs() < 1e-9, "expected unranged cell_h to stay at panel.h/n_y = 200.0, got {h}");
        }
    }
}
