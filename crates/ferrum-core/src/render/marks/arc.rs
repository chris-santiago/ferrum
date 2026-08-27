use ferrum_scene::{MarkBatchKind, PathCmd, SceneNode};

use crate::render::arrow_cast::{col_as_f64, col_as_ordinal_category_str, col_as_str};
use crate::render::color::with_opacity;
use crate::render::draw::{
    color_field, resolve_fill_color, to_scene_fill_stroke, DrawCtx, MarkBuildResult,
    MetadataColumns,
};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::opacity::resolve_scaled_opacity;
use crate::render::scale_resolve::ColorScale;
use crate::spec::coord::{CoordKind as SpecCoord, PolarThetaChannel};
use crate::spec::encoding::DataType as SpecDataType;

/// Resolved polar geometry shared by the pie/donut and annular-wedge paths.
pub(crate) struct PolarGeom {
    pub cx: f64,
    pub cy: f64,
    pub inner_radius: f64,
    pub outer_radius: f64,
    pub start_angle: f64,
    pub pad_angle: f64,
}

/// Extract polar coord parameters and resolve the panel-relative center +
/// outer radius. Returns `None` for any non-polar coord.
pub(crate) fn polar_geom(ctx: &DrawCtx<'_>) -> Option<PolarGeom> {
    let (start_angle, inner_radius, outer_radius_opt, pad_angle) = match &ctx.spec.coord {
        Some(SpecCoord::Polar { start_angle, inner_radius, outer_radius, pad_angle, .. }) => {
            (*start_angle, *inner_radius, *outer_radius, *pad_angle)
        }
        _ => return None,
    };
    let cx = ctx.panel.plot_area.x + ctx.panel.plot_area.w / 2.0;
    let cy = ctx.panel.plot_area.y + ctx.panel.plot_area.h / 2.0;
    let half_min = ctx.panel.plot_area.w.min(ctx.panel.plot_area.h) / 2.0;
    let outer_radius = outer_radius_opt.unwrap_or(half_min);
    Some(PolarGeom { cx, cy, inner_radius, outer_radius, start_angle, pad_angle })
}

// The theta→radius-scale convention (`theta="x"` → radial scale = y, `theta="y"`
// → radial scale = x) now lives in the shared
// `marks::channels::polar_channel_resolver` (C9), which both `arc` and
// `bar::build_polar` consume; the former standalone `polar_radius_scale` helper
// was subsumed by it.

/// Map a radial *data* value to a pixel radius, linearly interpolating the
/// radial domain onto the `[inner_radius, outer_radius]` pixel band. The
/// radial channel is `y` (when `theta="x"`) or `x` (when `theta="y"`);
/// `radius_scale` is the corresponding `ScaleKind`.
///
/// The radial domain is anchored at `0` (a radius is a magnitude from the
/// center, like a bar measured from its baseline): a data value of `0` maps to
/// the inner radius and the domain maximum maps to the outer radius. Using the
/// scale's raw `(lo, hi)` domain would instead pin `lo → inner_radius`, which
/// is wrong for radii (e.g. `r_inner=20` on a `[20, 80]` domain would collapse
/// to the center). A degenerate (non-positive) maximum maps every value to
/// `outer_radius` so a single-radius dataset still draws a visible ring.
pub(crate) fn radius_to_pixel(
    radius_scale: &crate::render::scale_resolve::ScaleKind,
    inner_radius: f64,
    outer_radius: f64,
    value: f64,
) -> f64 {
    let (dlo, dhi) = radius_scale.data_domain().unwrap_or((0.0, 1.0));
    let dmax = dhi.max(dlo).max(0.0);
    if dmax <= 0.0 {
        return outer_radius;
    }
    let t = (value / dmax).clamp(0.0, 1.0);
    inner_radius + t * (outer_radius - inner_radius)
}

/// Build arc (wedge) nodes for pie/donut/annular charts.
///
/// Requires `CoordPolar` — returns empty for any other coord.
///
/// Two render paths, gated by which channels are bound:
///
/// * **Annular** — when `theta2` (angular end) *or* `radius2` (outer radius)
///   is bound, each row becomes an annular wedge spanning `[theta, theta2]`
///   angularly and `[radius, radius2]` radially. Angles are taken directly
///   from the data columns (in radians, offset by `start_angle`); radii map
///   through the radial scale via [`radius_to_pixel`].
/// * **Pie/donut (legacy)** — otherwise each row's angular sweep is
///   proportional to its value in the theta-mapped field, swept from a fixed
///   inner radius to a fixed outer radius. This path is byte-stable with the
///   pre-D7 implementation.
pub fn build(ctx: &DrawCtx<'_>) -> MarkBuildResult {
    let theta_ch = match &ctx.spec.coord {
        Some(SpecCoord::Polar { theta, .. }) => *theta,
        _ => return MarkBuildResult::empty(MarkBatchKind::Arc),
    };
    let Some(geom) = polar_geom(ctx) else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };

    // Channel assignment under the Python polar remapping (shared resolver, C9):
    //   theta="x" → theta=x, theta2=x2, radius=y, radius2=y2
    //   theta="y" → theta=y, theta2=y2, radius=x, radius2=x2
    let pc = crate::render::marks::channels::polar_channel_resolver(
        theta_ch, &ctx.spec.encoding, ctx.scales,
    );
    let (theta_field, theta2_field, radius_field, radius2_field, radius_scale) = (
        pc.theta_field, pc.theta2_field, pc.radius_field, pc.radius2_field, pc.radius_scale,
    );

    // Annular mode: a second angular OR radial extent is bound. `radius_field`
    // is `None` for a theta-only arc (Python no longer synthesizes a dummy
    // radius channel; the single-axis exemption arm in
    // `render::scale_resolve::resolve_scales_with_leaf_context` (~1367)
    // supplies a dummy *unit scale* for the absent axis instead). The legacy
    // pie sweep below only ever reads `field` (the theta column) — never
    // `radius_field` or `radius_scale` — so this dummy scale is inert *for
    // wedge geometry*. It is NOT inert overall: `render::prepare::build_axes`
    // / `layout::compute_layout` size the panel's axis margin from whatever
    // `ScaleKind` the y channel resolves to, even under `CoordPolar` where
    // that axis is never drawn, so swapping a real domain for the dummy
    // unit scale does shift the panel's margin (see the P8 finding in
    // `tests/test_finding_p8.py`). So the annular-mode gate below is on *2
    // channels only.
    if theta2_field.is_some() || radius2_field.is_some() {
        return build_annular(
            ctx, &geom, theta_field, theta2_field, radius_field, radius2_field, radius_scale,
        );
    }

    let Some(field) = theta_field else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };

    // Detect nominal/ordinal theta: use equal-band layout where each category
    // gets an equal angular slice and the radius channel sets the outer radius.
    // This handles `mark_arc(theta:N, radius:Q)` — the Nightingale coxcomb via
    // the arc mark rather than mark_bar + CoordPolar.
    let theta_enc = match theta_ch {
        PolarThetaChannel::X => ctx.spec.encoding.x.as_ref(),
        PolarThetaChannel::Y => ctx.spec.encoding.y.as_ref(),
    };
    let theta_is_categorical = theta_enc.is_some_and(|e| {
        matches!(e.type_, Some(SpecDataType::Nominal) | Some(SpecDataType::Ordinal))
    });
    if theta_is_categorical {
        return build_nominal_theta(ctx, &geom, field, radius_field, radius_scale);
    }

    let Ok(values) = col_as_f64(ctx.batch, field) else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };

    let (cx, cy, inner_radius, outer_radius, start_angle, pad_angle) = (
        geom.cx, geom.cy, geom.inner_radius, geom.outer_radius, geom.start_angle, geom.pad_angle,
    );

    let total: f64 = values.iter()
        .filter_map(|v| *v)
        .filter(|v| v.is_finite() && *v > 0.0)
        .sum();
    if total <= 0.0 {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    }

    // Per-slice color: read color encoding field and look up in the color scale.
    let cfield = ctx.spec.encoding.color.as_ref().map(|e| e.field.as_str());
    let color_str: Option<Vec<Option<String>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Categorical { .. }), Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let color_f64: Option<Vec<Option<f64>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Continuous { .. }), Some(f)) => col_as_f64(ctx.batch, f).ok(),
        _ => None,
    };

    // Per-row opacity encoding.
    let opacity_values: Option<Vec<Option<f64>>> = ctx.spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    // Collect tooltip column data up front so we can index by row later.
    let meta = MetadataColumns::from_ctx(ctx);

    let mut acc = MarkNodes::with_capacity(values.len());
    let mut cum_angle = start_angle;
    let tau = std::f64::consts::TAU;

    for (i, v_opt) in values.iter().enumerate() {
        let v = match v_opt {
            Some(v) if v.is_finite() && *v > 0.0 => *v,
            _ => continue,
        };
        let sweep = (v / total) * tau;
        let angle_start = cum_angle + pad_angle / 2.0;
        let angle_end = cum_angle + sweep - pad_angle / 2.0;
        cum_angle += sweep;
        // Skip degenerate slices that collapse to zero or negative sweep after padding.
        if angle_end <= angle_start { continue; }

        // Resolve per-slice fill from color scale, fall back to mark_style.paint.fill.
        let fill_base = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_str.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_f64.as_ref().and_then(|v| v.get(i).copied().flatten()),
            ctx.mark_style.paint.fill,
        );
        // Resolve per-row opacity through scale if present; fall back to mark_style.paint.opacity.
        let row_opacity =
            resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);
        let fill_color = with_opacity(fill_base, row_opacity);

        let commands = wedge_path(cx, cy, inner_radius, outer_radius, angle_start, angle_end);
        acc.push(SceneNode::Path {
            commands,
            style: to_scene_fill_stroke(
                Some(fill_color),
                ctx.mark_style.paint.stroke,
                ctx.mark_style.paint.stroke_width,
                row_opacity,
                ctx.mark_style.paint.stroke_dash.as_deref(),
            ),
            closed: true,
        }, i);
    }

    // Build tooltip/href/description aligned to KEPT nodes only. Some rows are
    // skipped above (null or non-positive value), so the kept set is a strict
    // subset of the original batch. `build_metadata_for_indices` gathers
    // exactly the kept rows in node order so node j receives its true source
    // row's metadata.
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

/// Build arc wedges for a nominal/ordinal theta channel.
///
/// Each distinct category in `theta_field` gets an equal angular band spanning
/// `tau / n_cats` radians. The `radius_field` (if present) sets the outer
/// radius for each row via `radius_to_pixel`; absent radius falls back to the
/// coord's full outer radius (every wedge reaches the edge).
///
/// This enables `mark_arc(theta:N, radius:Q)` (Nightingale coxcomb via the arc
/// mark), producing the same equal-band layout as `build_polar` in `bar.rs`
/// but for the `Arc` mark.
fn build_nominal_theta(
    ctx: &DrawCtx<'_>,
    geom: &PolarGeom,
    theta_field: &str,
    radius_field: Option<&str>,
    radius_scale: &crate::render::scale_resolve::ScaleKind,
) -> MarkBuildResult {
    let angle_strs = match col_as_ordinal_category_str(ctx.batch, theta_field) {
        Ok(v) => v,
        Err(_) => return MarkBuildResult::empty(MarkBatchKind::Arc),
    };

    let radii: Option<Vec<Option<f64>>> = radius_field
        .and_then(|f| col_as_f64(ctx.batch, f).ok());

    // Build category index in first-appearance order.
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

    let pad_angle = if geom.pad_angle > 0.0 {
        geom.pad_angle
    } else if n_cats == 1 {
        1e-3
    } else {
        0.0
    };

    // Per-row color columns.
    let cfield = color_field(ctx, ctx.spec);
    let color_str: Option<Vec<Option<String>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Categorical { .. }), Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let color_f64: Option<Vec<Option<f64>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Continuous { .. }), Some(f)) => col_as_f64(ctx.batch, f).ok(),
        _ => None,
    };
    let opacity_values: Option<Vec<Option<f64>>> = ctx.spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let meta = MetadataColumns::from_ctx(ctx);

    let mut acc = MarkNodes::with_capacity(angle_strs.len());

    for (i, cat_opt) in angle_strs.iter().enumerate() {
        let cat = match cat_opt { Some(s) => s.as_str(), None => continue };
        let k = *cat_index.get(cat).unwrap_or(&0);

        let angle_start = geom.start_angle + k as f64 * band + pad_angle / 2.0;
        let angle_end = geom.start_angle + (k as f64 + 1.0) * band - pad_angle / 2.0;
        if angle_end <= angle_start {
            continue;
        }

        // Outer radius: from the radius channel (data value) if present, else
        // full coord outer radius (wedge reaches the edge).
        let outer_r = match radii.as_ref().and_then(|v| v.get(i).copied().flatten()) {
            Some(rv) if rv.is_finite() => {
                radius_to_pixel(radius_scale, geom.inner_radius, geom.outer_radius, rv)
            }
            _ => geom.outer_radius,
        };

        let fill_base = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_str.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_f64.as_ref().and_then(|v| v.get(i).copied().flatten()),
            ctx.mark_style.paint.fill,
        );
        let row_opacity =
            resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);
        let fill_color = with_opacity(fill_base, row_opacity);

        let commands = wedge_path(
            geom.cx, geom.cy, geom.inner_radius, outer_r, angle_start, angle_end,
        );
        acc.push(SceneNode::Path {
            commands,
            style: to_scene_fill_stroke(
                Some(fill_color),
                ctx.mark_style.paint.stroke,
                ctx.mark_style.paint.stroke_width,
                row_opacity,
                ctx.mark_style.paint.stroke_dash.as_deref(),
            ),
            closed: true,
        }, i);
    }

    // Metadata must be aligned to the KEPT nodes, not all rows. Some rows are
    // skipped above (null category, degenerate zero-span wedge), so the kept
    // set is a subset of the original batch. `build_metadata_for_indices`
    // gathers exactly the kept rows in node order so node j receives its true
    // source row's tooltip/href/description (not row j's).
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

/// Build annular-wedge nodes from per-row angular and radial extents.
///
/// Each row spans `[theta, theta2]` angularly (radians from the data columns,
/// offset by `start_angle`) and `[radius, radius2]` radially (data values
/// mapped through `radius_scale`). When `theta2` is unbound the legacy gate
/// would not have routed here; this path requires `theta2` to be present for
/// a meaningful angular span (a row with no theta2 collapses to zero sweep
/// and is skipped). When `radius2` is unbound the outer radius falls back to
/// the coord's outer radius (solid-to-edge wedge).
#[allow(clippy::too_many_arguments)]
fn build_annular(
    ctx: &DrawCtx<'_>,
    geom: &PolarGeom,
    theta_field: Option<&str>,
    theta2_field: Option<&str>,
    radius_field: Option<&str>,
    radius2_field: Option<&str>,
    radius_scale: &crate::render::scale_resolve::ScaleKind,
) -> MarkBuildResult {
    let Some(tf) = theta_field else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };
    let Ok(theta) = col_as_f64(ctx.batch, tf) else {
        return MarkBuildResult::empty(MarkBatchKind::Arc);
    };
    let theta2: Option<Vec<Option<f64>>> =
        theta2_field.and_then(|f| col_as_f64(ctx.batch, f).ok());
    let radius: Option<Vec<Option<f64>>> =
        radius_field.and_then(|f| col_as_f64(ctx.batch, f).ok());
    let radius2: Option<Vec<Option<f64>>> =
        radius2_field.and_then(|f| col_as_f64(ctx.batch, f).ok());

    // Per-slice color (mirrors the legacy path).
    let cfield = ctx.spec.encoding.color.as_ref().map(|e| e.field.as_str());
    let color_str: Option<Vec<Option<String>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Categorical { .. }), Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let color_f64: Option<Vec<Option<f64>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Continuous { .. }), Some(f)) => col_as_f64(ctx.batch, f).ok(),
        _ => None,
    };
    let opacity_values: Option<Vec<Option<f64>>> = ctx.spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let meta = MetadataColumns::from_ctx(ctx);

    let mut acc = MarkNodes::with_capacity(theta.len());

    for (i, t0_opt) in theta.iter().enumerate() {
        let Some(t0) = t0_opt.filter(|v| v.is_finite()) else { continue };
        // Angular end: explicit theta2 column. Without it there's no span.
        let t1 = match theta2.as_ref().and_then(|v| v.get(i).copied().flatten()) {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        let angle_start = geom.start_angle + t0 + geom.pad_angle / 2.0;
        let angle_end = geom.start_angle + t1 - geom.pad_angle / 2.0;
        if angle_end <= angle_start { continue; }

        // Radial extents: map data values through the radial scale. Missing
        // inner radius defaults to the coord inner radius; missing outer
        // radius defaults to the coord outer radius.
        let inner_r = match radius.as_ref().and_then(|v| v.get(i).copied().flatten()) {
            Some(rv) if rv.is_finite() => {
                radius_to_pixel(radius_scale, geom.inner_radius, geom.outer_radius, rv)
            }
            _ => geom.inner_radius,
        };
        let outer_r = match radius2.as_ref().and_then(|v| v.get(i).copied().flatten()) {
            Some(rv) if rv.is_finite() => {
                radius_to_pixel(radius_scale, geom.inner_radius, geom.outer_radius, rv)
            }
            _ => geom.outer_radius,
        };

        let fill_base = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_str.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_f64.as_ref().and_then(|v| v.get(i).copied().flatten()),
            ctx.mark_style.paint.fill,
        );
        let row_opacity =
            resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);
        let fill_color = with_opacity(fill_base, row_opacity);

        let commands = wedge_path(geom.cx, geom.cy, inner_r, outer_r, angle_start, angle_end);
        acc.push(SceneNode::Path {
            commands,
            style: to_scene_fill_stroke(
                Some(fill_color),
                ctx.mark_style.paint.stroke,
                ctx.mark_style.paint.stroke_width,
                row_opacity,
                ctx.mark_style.paint.stroke_dash.as_deref(),
            ),
            closed: true,
        }, i);
    }

    // Metadata must be aligned to the KEPT nodes, not all rows. Some rows are
    // skipped above (null theta, non-finite theta2, degenerate zero-span wedge),
    // so the kept set is a subset of the original batch. `build_metadata_for_indices`
    // gathers exactly the kept rows in node order so node j receives its true
    // source row's tooltip/href/description (not row j's).
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

/// SVG path commands for an arc wedge from `angle_start` to `angle_end`.
///
/// Angles are measured clockwise from 12 o'clock (north):
/// `x = cx + r·sin(θ)`, `y = cy − r·cos(θ)`.
pub(crate) fn wedge_path(
    cx: f64,
    cy: f64,
    inner_r: f64,
    outer_r: f64,
    angle_start: f64,
    angle_end: f64,
) -> Vec<PathCmd> {
    let mut cmds = Vec::new();
    let sweep = angle_end - angle_start;
    let full_circle = sweep.abs() >= std::f64::consts::TAU - 1e-9;
    let large_arc = sweep.abs() > std::f64::consts::PI;

    let ox0 = cx + outer_r * angle_start.sin();
    let oy0 = cy - outer_r * angle_start.cos();
    let ox1 = cx + outer_r * angle_end.sin();
    let oy1 = cy - outer_r * angle_end.cos();

    cmds.push(PathCmd::MoveTo { x: ox0, y: oy0 });

    if full_circle {
        let mid = angle_start + std::f64::consts::PI;
        let oxm = cx + outer_r * mid.sin();
        let oym = cy - outer_r * mid.cos();
        cmds.push(PathCmd::ArcTo { rx: outer_r, ry: outer_r, rotation: 0.0, large_arc: false, sweep: true, x: oxm, y: oym });
        cmds.push(PathCmd::ArcTo { rx: outer_r, ry: outer_r, rotation: 0.0, large_arc: false, sweep: true, x: ox0, y: oy0 });
    } else {
        cmds.push(PathCmd::ArcTo { rx: outer_r, ry: outer_r, rotation: 0.0, large_arc, sweep: true, x: ox1, y: oy1 });
    }

    if inner_r > 0.0 {
        let ix1 = cx + inner_r * angle_end.sin();
        let iy1 = cy - inner_r * angle_end.cos();
        let ix0 = cx + inner_r * angle_start.sin();
        let iy0 = cy - inner_r * angle_start.cos();
        cmds.push(PathCmd::LineTo { x: ix1, y: iy1 });
        if full_circle {
            let mid = angle_start + std::f64::consts::PI;
            let ixm = cx + inner_r * mid.sin();
            let iym = cy - inner_r * mid.cos();
            cmds.push(PathCmd::ArcTo { rx: inner_r, ry: inner_r, rotation: 0.0, large_arc: false, sweep: false, x: ixm, y: iym });
            cmds.push(PathCmd::ArcTo { rx: inner_r, ry: inner_r, rotation: 0.0, large_arc: false, sweep: false, x: ix0, y: iy0 });
        } else {
            cmds.push(PathCmd::ArcTo { rx: inner_r, ry: inner_r, rotation: 0.0, large_arc, sweep: false, x: ix0, y: iy0 });
        }
    } else {
        cmds.push(PathCmd::LineTo { x: cx, y: cy });
    }

    cmds.push(PathCmd::Close);
    cmds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::{resolve_mark_style, DrawCtx};
    use crate::render::scale_resolve::{OpacityScale, ResolvedScales, ScaleKind};
    use crate::scale::linear::LinearScale;
    use crate::spec::chart::ChartSpec;
    use crate::spec::coord::{CoordKind as SpecCoord, PolarThetaChannel};
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use ferrum_scene::{PolarDirection, SceneNode};
    use std::sync::Arc;

    fn polar_spec(with_opacity: bool) -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Arc,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                opacity: if with_opacity {
                    Some(EncodingSpec { field: "op".into(), type_: Some(SDT::Quantitative), ..Default::default() })
                } else {
                    None
                },
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: None,
                pad_angle: 0.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        }
    }

    fn make_batch(with_opacity: bool) -> arrow::record_batch::RecordBatch {
        let mut fields = vec![
            Field::new("val", DataType::Float64, false),
        ];
        let mut arrays: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(Float64Array::from(vec![10.0, 30.0, 60.0])),
        ];
        if with_opacity {
            fields.push(Field::new("op", DataType::Float64, false));
            arrays.push(Arc::new(Float64Array::from(vec![0.2, 0.5, 0.9])));
        }
        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays).unwrap()
    }

    fn make_scales(with_opacity: bool) -> ResolvedScales {
        ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 100.0], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 100.0], vec![100.0, 0.0], false, false)),
            color: None,
            size: None,
            shape: None,
            opacity: if with_opacity {
                Some(OpacityScale {
                    inner: ScaleKind::Linear(LinearScale::new_internal(
                        vec![0.2, 0.9], vec![0.2, 0.9], false, false,
                    )),
                })
            } else {
                None
            },
            x2: None,
            y2: None,
            y_slots: Default::default(),
        }
    }

    fn make_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        }
    }

    /// Basic smoke test: 3 slices → 3 Path nodes.
    #[test]
    fn arc_emits_one_path_per_slice() {
        let spec = polar_spec(false);
        let batch = make_batch(false);
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = make_scales(false);
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx);
        let paths = result.nodes.iter().filter(|n| matches!(n, SceneNode::Path { .. })).count();
        assert_eq!(paths, 3, "expected 3 arc Path nodes, got {paths}");
    }

    /// W18: When an opacity encoding is present, arc slices must have different
    /// alpha values in their fill color (per-row opacity applied through scale).
    #[test]
    fn w18_arc_opacity_encoding_applied_per_slice() {
        let spec = polar_spec(true);
        let batch = make_batch(true);
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = make_scales(true);
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx);

        let paths: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Path { style, .. } = n { Some(style.clone()) } else { None }
        }).collect();
        assert_eq!(paths.len(), 3, "expected 3 Path nodes");

        // With opacity values [0.2, 0.5, 0.9] the alphas must all differ.
        // ferrum_scene::Color uses .a for alpha.
        let alphas: Vec<u8> = paths.iter()
            .map(|s| s.fill.as_ref().map(|c| c.a).unwrap_or(255))
            .collect();
        let all_same = alphas.iter().all(|&a| a == alphas[0]);
        assert!(
            !all_same,
            "per-row opacity encoding must produce different alphas on arc slices; all were {:?}",
            alphas
        );
    }

    /// D7: radius_to_pixel anchors the radial domain at 0 — a data value of 0
    /// maps to the inner radius and the domain max maps to the outer radius,
    /// regardless of the scale's lower bound.
    #[test]
    fn radius_to_pixel_anchors_domain_at_zero() {
        // Domain [20, 80] but radii are magnitudes from center: 0 → inner.
        let scale = ScaleKind::Linear(LinearScale::new_internal(
            vec![20.0, 80.0], vec![100.0, 0.0], false, false,
        ));
        // dmax = 80. r=0 → inner (0). r=80 → outer (200). r=40 → 100.
        assert!((radius_to_pixel(&scale, 0.0, 200.0, 0.0) - 0.0).abs() < 1e-9);
        assert!((radius_to_pixel(&scale, 0.0, 200.0, 80.0) - 200.0).abs() < 1e-9);
        assert!((radius_to_pixel(&scale, 0.0, 200.0, 40.0) - 100.0).abs() < 1e-9);
        // r=20 must NOT collapse to the center (the old (lo,hi) bug).
        assert!(radius_to_pixel(&scale, 0.0, 200.0, 20.0) > 1.0);
        // Non-zero inner radius offsets the band.
        assert!((radius_to_pixel(&scale, 50.0, 200.0, 0.0) - 50.0).abs() < 1e-9);
    }

    /// D7: a degenerate (non-positive max) radial domain maps every value to
    /// the outer radius so a single-radius dataset still draws a visible ring.
    #[test]
    fn radius_to_pixel_degenerate_domain_uses_outer() {
        let scale = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 0.0], vec![0.0, 0.0], false, false,
        ));
        assert!((radius_to_pixel(&scale, 0.0, 150.0, 5.0) - 150.0).abs() < 1e-9);
    }

    fn annular_spec() -> ChartSpec {
        // theta=x (t0), theta2=x2 (t1), radius=y (r0), radius2=y2 (r1).
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Arc,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "t0".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "t1".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "r0".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "r1".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: None,
                pad_angle: 0.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        }
    }

    /// Count the number of Arc (`A`) commands in a Path node's command list.
    fn arc_count(node: &SceneNode) -> usize {
        if let SceneNode::Path { commands, .. } = node {
            commands.iter().filter(|c| matches!(c, PathCmd::ArcTo { .. })).count()
        } else {
            0
        }
    }

    /// D7-C: a two-ring sunburst with theta/theta2/radius/radius2 produces one
    /// wedge per row, and the outer ring (radius=40) has a non-zero inner
    /// radius while the inner ring (radius=0) starts at the center.
    #[test]
    fn annular_wedges_have_per_row_radii_and_partial_sweeps() {
        let tau = std::f64::consts::TAU;
        let spec = annular_spec();
        // 4 outer-ring rows (r0=40, r1=80) + 2 inner-ring rows (r0=0, r1=40).
        let schema = Arc::new(Schema::new(vec![
            Field::new("t0", DataType::Float64, false),
            Field::new("t1", DataType::Float64, false),
            Field::new("r0", DataType::Float64, false),
            Field::new("r1", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, tau/4.0, tau/2.0, 3.0*tau/4.0, 0.0, tau/2.0])),
            Arc::new(Float64Array::from(vec![tau/4.0, tau/2.0, 3.0*tau/4.0, tau, tau/2.0, tau])),
            Arc::new(Float64Array::from(vec![40.0, 40.0, 40.0, 40.0, 0.0, 0.0])),
            Arc::new(Float64Array::from(vec![80.0, 80.0, 80.0, 80.0, 40.0, 40.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        // Radial scale (y): domain [0, 80], range irrelevant for radius mapping.
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, tau], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 80.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None,
            x2: Some("t1".into()), y2: Some("r1".into()),
            y_slots: Default::default(),
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx);

        let paths: Vec<&SceneNode> = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Path { .. }))
            .collect();
        assert_eq!(paths.len(), 6, "expected one wedge per row");

        // No wedge spans the full circle: each is a quarter or half, so the
        // full-circle 2-arc split never triggers (≤ 2 arcs per path).
        for n in &paths {
            assert!(arc_count(n) <= 2, "partial wedge should have ≤ 2 arcs");
        }

        // Outer-ring rows (first 4) have a non-zero inner arc (annular: 2 arcs).
        // Inner-ring rows (last 2, r0=0) collapse to center (1 outer arc).
        let outer_arcs: Vec<usize> = paths[..4].iter().map(|n| arc_count(n)).collect();
        let inner_arcs: Vec<usize> = paths[4..].iter().map(|n| arc_count(n)).collect();
        assert!(outer_arcs.iter().all(|&c| c == 2),
            "outer-ring wedges should be annular (inner + outer arc): {outer_arcs:?}");
        assert!(inner_arcs.iter().all(|&c| c == 1),
            "inner-ring wedges (r0=0) should be solid (single outer arc): {inner_arcs:?}");
    }

    // ── Metadata-alignment regression tests ─────────────────────────────────
    //
    // These tests guard against the bug where `build_nominal_theta` and
    // `build_annular` called `meta.build_metadata(ctx)` (which returns full
    // per-row vectors indexed 0..n_rows) instead of
    // `meta.build_metadata_for_indices(&data_indices)` (which gathers only the
    // kept rows in node order). When any wedge is skipped the SVG walker's
    // node-enumeration index j diverges from the source row index, so node j
    // would receive row j's metadata rather than its true source row's.

    /// Regression: pie with a null value (skipped wedge) + href encoding must
    /// keep href aligned to kept nodes. The main `build` path already used
    /// `data_indices` for tooltips; this test confirms it also handles hrefs
    /// correctly and that the pie-with-null path is byte-stable.
    #[test]
    fn pie_skipped_null_href_stays_aligned() {
        // 3 rows: val=[10.0, null, 60.0]. The null is skipped → 2 nodes.
        // href values are ["http://a", "http://b", "http://c"]; after the skip
        // the surviving nodes (rows 0 and 2) must map to "http://a" and "http://c".
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, true),
            Field::new("link", DataType::Utf8, true),
        ]));
        // Build a nullable Float64Array with a null at index 1.
        let val_array = Float64Array::from(vec![Some(10.0_f64), None, Some(60.0_f64)]);
        let href_array = StringArray::from(vec![
            Some("http://a"),
            Some("http://b"),
            Some("http://c"),
        ]);
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(val_array),
            Arc::new(href_array),
        ]).unwrap();

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Arc,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                href: Some(EncodingSpec { field: "link".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: None,
                pad_angle: 0.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = make_scales(false);
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx);

        // Exactly 2 wedges survive (null value is skipped).
        assert_eq!(result.nodes.len(), 2, "expected 2 nodes after null skip; got {}", result.nodes.len());

        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), 2, "hrefs length must equal node count");
        // Node 0 = source row 0 → "http://a". Node 1 = source row 2 → "http://c".
        assert_eq!(hrefs[0].as_deref(), Some("http://a"),
            "node 0 href must be row 0's href; got {:?}", hrefs[0]);
        assert_eq!(hrefs[1].as_deref(), Some("http://c"),
            "node 1 href must be row 2's href (not row 1's 'http://b'); got {:?}", hrefs[1]);
    }

    /// Regression: nominal-theta arc with a null category (skipped wedge) + tooltip
    /// and href encoding must keep metadata aligned to kept nodes.
    ///
    /// Layout: 3 rows with categories ["A", null, "B"]. The null is skipped.
    /// Surviving nodes are (in iteration order): row 0 (cat="A") and row 2 (cat="B").
    /// href values are ["http://a", "http://b_skipped", "http://c"]; the surviving
    /// nodes must receive "http://a" and "http://c" respectively, not the shifted
    /// values "http://a" and "http://b_skipped" that the old full-row indexing
    /// would have produced.
    #[test]
    fn nominal_theta_skipped_null_metadata_aligned() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, true),
            Field::new("r",   DataType::Float64, false),
            Field::new("link", DataType::Utf8, false),
            Field::new("tip", DataType::Utf8, false),
        ]));
        // Row 1 has a null category → skipped.
        let cat_array = StringArray::from(vec![Some("A"), None, Some("B")]);
        let r_array   = Float64Array::from(vec![80.0_f64, 80.0, 80.0]);
        let link_array = StringArray::from(vec!["http://a", "http://b_skipped", "http://c"]);
        let tip_array  = StringArray::from(vec!["tip_a", "tip_b_skipped", "tip_c"]);
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(cat_array),
            Arc::new(r_array),
            Arc::new(link_array),
            Arc::new(tip_array),
        ]).unwrap();

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Arc,
            encoding: Encoding {
                // theta is nominal x; radius is quantitative y.
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Nominal), ..Default::default() }),
                y: Some(EncodingSpec { field: "r".into(),   type_: Some(SDT::Quantitative), ..Default::default() }),
                href: Some(EncodingSpec { field: "link".into(), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: None,
                pad_angle: 0.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 100.0], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 80.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None, x2: None, y2: None, y_slots: Default::default(),
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx);

        // 2 nodes survive (null category row is skipped).
        assert_eq!(result.nodes.len(), 2,
            "expected 2 nodes after null-category skip; got {}", result.nodes.len());

        // hrefs: node 0 → row 0 ("http://a"); node 1 → row 2 ("http://c").
        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), 2, "hrefs length must equal node count");
        assert_eq!(hrefs[0].as_deref(), Some("http://a"),
            "node 0 href must be row 0's href; got {:?}", hrefs[0]);
        assert_eq!(hrefs[1].as_deref(), Some("http://c"),
            "node 1 href must be row 2's href (not row 1's 'http://b_skipped'); got {:?}", hrefs[1]);

        // tooltips: node 0 → "tip_a"; node 1 → "tip_c".
        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2, "tooltips length must equal node count");
        let tip0 = &tooltips[0].fields[0].value;
        let tip1 = &tooltips[1].fields[0].value;
        assert_eq!(tip0, "tip_a",
            "node 0 tooltip must be row 0's; got '{tip0}'");
        assert_eq!(tip1, "tip_c",
            "node 1 tooltip must be row 2's (not row 1's 'tip_b_skipped'); got '{tip1}'");
    }

    /// Regression: annular arc with a skipped row (null theta2 → no span) + href
    /// encoding must keep metadata aligned to kept nodes.
    ///
    /// 3 rows. Row 1 has null theta2 → skipped. Surviving nodes are rows 0 and 2.
    /// href values are ["http://a", "http://b_skipped", "http://c"]; surviving
    /// nodes must receive "http://a" and "http://c".
    #[test]
    fn annular_skipped_row_metadata_aligned() {
        let tau = std::f64::consts::TAU;
        let schema = Arc::new(Schema::new(vec![
            Field::new("t0",   DataType::Float64, false),
            Field::new("t1",   DataType::Float64, true),  // nullable → null at row 1
            Field::new("r0",   DataType::Float64, false),
            Field::new("r1",   DataType::Float64, false),
            Field::new("link", DataType::Utf8,    false),
        ]));
        // Row 1 has null t1 → no angular span → skipped.
        let t0_arr  = Float64Array::from(vec![0.0_f64,    tau / 4.0, tau / 2.0]);
        let t1_arr  = Float64Array::from(vec![Some(tau / 4.0), None, Some(tau)]);
        let r0_arr  = Float64Array::from(vec![0.0_f64, 0.0, 0.0]);
        let r1_arr  = Float64Array::from(vec![80.0_f64, 80.0, 80.0]);
        let link_arr = StringArray::from(vec!["http://a", "http://b_skipped", "http://c"]);
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(t0_arr),
            Arc::new(t1_arr),
            Arc::new(r0_arr),
            Arc::new(r1_arr),
            Arc::new(link_arr),
        ]).unwrap();

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Arc,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "t0".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "t1".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y:  Some(EncodingSpec { field: "r0".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "r1".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                href: Some(EncodingSpec { field: "link".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: None,
                pad_angle: 0.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, tau], vec![0.0, 200.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 80.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None,
            x2: Some("t1".into()), y2: Some("r1".into()),
            y_slots: Default::default(),
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx);

        // 2 nodes survive (row 1 has null t1 → skipped).
        assert_eq!(result.nodes.len(), 2,
            "expected 2 nodes after null-theta2 skip; got {}", result.nodes.len());

        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), 2, "hrefs length must equal node count");
        assert_eq!(hrefs[0].as_deref(), Some("http://a"),
            "node 0 href must be row 0's href; got {:?}", hrefs[0]);
        assert_eq!(hrefs[1].as_deref(), Some("http://c"),
            "node 1 href must be row 2's href (not row 1's 'http://b_skipped'); got {:?}", hrefs[1]);
    }

    /// Stability: arcs with NO skipped wedges (all rows valid) must produce
    /// hrefs in the original row order — `build_metadata_for_indices` with
    /// a full-range index must equal `build_metadata` for a complete dataset.
    #[test]
    fn pie_no_skipped_wedges_hrefs_unchanged() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("val",  DataType::Float64, false),
            Field::new("link", DataType::Utf8, false),
        ]));
        let val_array  = Float64Array::from(vec![10.0_f64, 30.0, 60.0]);
        let href_array = StringArray::from(vec!["http://a", "http://b", "http://c"]);
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(val_array),
            Arc::new(href_array),
        ]).unwrap();

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Arc,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                href: Some(EncodingSpec { field: "link".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: None,
                pad_angle: 0.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = make_scales(false);
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx);

        // All 3 rows are valid → 3 nodes, hrefs in original order.
        assert_eq!(result.nodes.len(), 3, "expected 3 nodes; got {}", result.nodes.len());
        let hrefs = result.hrefs.expect("hrefs must be Some");
        assert_eq!(hrefs.len(), 3);
        assert_eq!(hrefs[0].as_deref(), Some("http://a"));
        assert_eq!(hrefs[1].as_deref(), Some("http://b"));
        assert_eq!(hrefs[2].as_deref(), Some("http://c"));
    }

    // ── Ported from bug_hunt_marks_rendering(_r2).rs (R1) ──────────────────
    // These call `wedge_path` and `build` directly instead of reimplementing
    // the trig/degenerate-sweep logic inline, so a future regression in the
    // real function fails the test instead of the test's own copy.

    /// A near-full-circle sweep (angle_end - angle_start ~= TAU) takes the
    /// `full_circle` branch, which emits two half-arc `ArcTo` pairs instead of
    /// one full-sweep `ArcTo`. All emitted coordinates must stay finite.
    #[test]
    fn wedge_path_full_circle_sweep_produces_finite_coords() {
        let cmds = wedge_path(100.0, 100.0, 0.0, 50.0, 0.0, std::f64::consts::TAU - 1e-10);
        assert!(matches!(cmds.first(), Some(PathCmd::MoveTo { .. })));
        assert!(matches!(cmds.last(), Some(PathCmd::Close)));
        let arc_count = cmds.iter().filter(|c| matches!(c, PathCmd::ArcTo { .. })).count();
        assert_eq!(arc_count, 2, "full_circle branch must emit two half-arc ArcTo commands");
        for cmd in &cmds {
            match cmd {
                PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => {
                    assert!(x.is_finite() && y.is_finite(), "MoveTo/LineTo coords must be finite");
                }
                PathCmd::ArcTo { rx, ry, x, y, .. } => {
                    assert!(rx.is_finite() && ry.is_finite() && x.is_finite() && y.is_finite(),
                        "ArcTo coords/radii must be finite");
                }
                PathCmd::Close => {}
                other => panic!("unexpected PathCmd in wedge_path output: {other:?}"),
            }
        }
    }

    /// When `inner_r == outer_r` the wedge collapses to a zero-width ring:
    /// the inner and outer arc endpoints must exactly coincide (not merely
    /// be finite). For the non-full-circle, `inner_r > 0.0` branch, `cmds`
    /// is exactly `[MoveTo(outer_start), ArcTo(outer_end), LineTo(inner_end),
    /// ArcTo(inner_start), Close]` (`arc.rs:510,519,527,535,541`) -- so the
    /// zero-width collapse means `LineTo`'s endpoint must equal the outer
    /// `ArcTo`'s endpoint (both at `angle_end`), and the inner `ArcTo`'s
    /// endpoint must equal the initial `MoveTo` (both at `angle_start`).
    #[test]
    fn wedge_path_inner_radius_equals_outer_radius_collapses_to_zero_width() {
        let r = 50.0;
        let cmds = wedge_path(100.0, 100.0, r, r, 0.0, std::f64::consts::FRAC_PI_2);
        assert_eq!(cmds.len(), 5, "MoveTo + outer ArcTo + inner LineTo + inner ArcTo + Close");

        let (ox0, oy0) = match cmds[0] { PathCmd::MoveTo { x, y } => (x, y), ref c => panic!("expected MoveTo, got {c:?}") };
        let (ox1, oy1) = match cmds[1] { PathCmd::ArcTo { x, y, .. } => (x, y), ref c => panic!("expected outer ArcTo, got {c:?}") };
        let (ix1, iy1) = match cmds[2] { PathCmd::LineTo { x, y } => (x, y), ref c => panic!("expected inner LineTo, got {c:?}") };
        let (ix0, iy0) = match cmds[3] { PathCmd::ArcTo { x, y, .. } => (x, y), ref c => panic!("expected inner ArcTo, got {c:?}") };
        assert!(matches!(cmds[4], PathCmd::Close));

        // Outer endpoint at angle_end must coincide with the inner LineTo's
        // endpoint at angle_end -- zero radial width, not merely "finite".
        assert!((ox1 - ix1).abs() < 1e-9 && (oy1 - iy1).abs() < 1e-9,
            "outer end ({ox1}, {oy1}) must coincide with inner end ({ix1}, {iy1}) when inner_r == outer_r");
        // Inner endpoint at angle_start must coincide with the initial outer
        // MoveTo at angle_start -- the ring closes back on itself exactly.
        assert!((ox0 - ix0).abs() < 1e-9 && (oy0 - iy0).abs() < 1e-9,
            "outer start ({ox0}, {oy0}) must coincide with inner start ({ix0}, {iy0}) when inner_r == outer_r");
    }

    /// Very large start angles (many full rotations away from zero) must
    /// place the wedge at the *same* geometry sin/cos periodicity implies,
    /// not merely produce finite output. Pinned against hand-computed
    /// coordinates (verified independently in Python's `math.sin`/`cos`,
    /// not read off a Rust run) rather than a same-function self-comparison:
    /// a self-comparison against `wedge_path(..., 0.0, sweep)` is blind to a
    /// sin<->cos swap at `arc.rs:505-506`, because the swap would apply
    /// identically to *both* calls and periodicity holds for cos as much as
    /// sin -- confirmed empirically while writing this test (that
    /// self-comparison design stayed green under the swap mutation).
    /// `cx=cy=100, outer_r=50, inner_r=0, start=100*TAU, sweep=PI/4`:
    /// `MoveTo` (`angle_start`) = `(100 + 50*sin(100*TAU), 100 - 50*cos(100*TAU))`
    ///   ~= `(100.0, 50.0)` (sin/cos of a TAU multiple are 0/1, up to ~2e-13
    ///   argument-reduction residual measured in Python for this exact value);
    /// `ArcTo` endpoint (`angle_end = start + PI/4`) =
    ///   `(100 + 50*sin(PI/4), 100 - 50*cos(PI/4))` ~= `(135.35533906, 64.64466094)`
    ///   exactly (adding a whole number of periods to the argument before
    ///   the `+ PI/4` does not change this value); `LineTo` = `(cx, cy)` =
    ///   `(100.0, 100.0)` (the `inner_r <= 0.0` branch, `arc.rs:538`).
    #[test]
    fn wedge_path_very_large_start_angle_matches_hand_computed_geometry() {
        let sweep = std::f64::consts::FRAC_PI_4;
        let start = 100.0 * std::f64::consts::TAU;
        let cmds = wedge_path(100.0, 100.0, 0.0, 50.0, start, start + sweep);
        assert_eq!(cmds.len(), 4, "MoveTo + outer ArcTo + LineTo(cx,cy) + Close (inner_r <= 0.0)");

        let tol = 1e-6;
        match cmds[0] {
            PathCmd::MoveTo { x, y } => {
                assert!((x - 100.0).abs() < tol && (y - 50.0).abs() < tol,
                    "MoveTo at a 100*TAU start must land at (100.0, 50.0); got ({x}, {y})");
            }
            ref c => panic!("expected MoveTo, got {c:?}"),
        }
        match cmds[1] {
            PathCmd::ArcTo { x, y, .. } => {
                assert!((x - 135.35533905932738).abs() < tol && (y - 64.64466094067262).abs() < tol,
                    "outer ArcTo endpoint at start+PI/4 must land at (135.35533906, 64.64466094); got ({x}, {y})");
            }
            ref c => panic!("expected outer ArcTo, got {c:?}"),
        }
        match cmds[2] {
            PathCmd::LineTo { x, y } => {
                assert!((x - 100.0).abs() < tol && (y - 100.0).abs() < tol,
                    "inner_r <= 0.0 LineTo must land at center (100.0, 100.0); got ({x}, {y})");
            }
            ref c => panic!("expected LineTo, got {c:?}"),
        }
        assert!(matches!(cmds[3], PathCmd::Close));
    }

    /// A slice whose `pad_angle` exceeds its own sweep (`angle_end <= angle_start`)
    /// is skipped entirely by `build`'s per-row guard — it must not appear in
    /// the output, and the remaining rows must render normally.
    #[test]
    fn build_skips_row_whose_pad_angle_exceeds_its_sweep() {
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Arc,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 0.0,
                outer_radius: None,
                // ~57.3deg pad; the tiny 0.5-value slice's natural sweep
                // (~0.0347 rad) is far smaller, so angle_end <= angle_start.
                pad_angle: 1.0,
                direction: PolarDirection::Clockwise,
            }),
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Float64, false)]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![0.5, 30.0, 60.0]))],
        ).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = make_scales(false);
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx);

        let paths = result.nodes.iter().filter(|n| matches!(n, SceneNode::Path { .. })).count();
        assert_eq!(paths, 2, "the pad-exceeds-sweep row must be skipped, leaving 2 wedges");
    }
}
