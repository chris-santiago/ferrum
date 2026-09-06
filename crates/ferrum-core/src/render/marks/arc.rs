use ferrum_scene::{MarkBatchKind, PathCmd, SceneNode};

use crate::render::arrow_cast::{col_as_f64, col_as_ordinal_category_str};
use crate::render::color::with_opacity;
use crate::render::draw::{
    resolve_fill_color, to_scene_fill_stroke, DrawCtx, MarkBuildResult, MetadataColumns,
};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::channels::color_column_loader;
use crate::render::marks::opacity::resolve_scaled_opacity;
use crate::render::RenderError;
use crate::spec::coord::{CoordKind as SpecCoord, PolarThetaChannel};
use crate::spec::encoding::DataType as SpecDataType;

/// The typed refusal for `mark_arc` under a non-`Polar` coord (spec §6.4,
/// batch E). Two producers share this one definition so the wording cannot
/// drift between them:
///
/// - `scale_resolve::resolve_scales_with_leaf_context` raises it FIRST, at
///   the single scale-resolution chokepoint every render path shares, so it
///   preempts the generic `EncodingTypeMismatch` fallthrough that used to
///   surface the misleading "encoding 'y' expected EncodingSpec, got None".
/// - This module's own [`build`] raises it too, from the same non-Polar arm
///   that used to return a silent empty result — the path-independent
///   backstop for a caller that reaches this mark builder WITHOUT going
///   through the standalone/composite-leaf scale resolution above (a
///   merged-flat layer's per-mark dispatch — e.g.
///   `LayerChart(bar, arc).interactive()`'s single-panel path, or
///   `resolve={"y": "independent"}` — resolves scales against the OTHER
///   layer's mark, so `Mark::Arc`'s own guard never runs there).
pub(crate) fn requires_polar_coord() -> RenderError {
    RenderError::MarkRequiresCoord {
        mark: "mark_arc",
        required: "Polar",
        hint: "add .coord(fm.CoordPolar(...))",
    }
}

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
pub fn build(ctx: &DrawCtx<'_>) -> Result<MarkBuildResult, RenderError> {
    let theta_ch = match &ctx.spec.coord {
        Some(SpecCoord::Polar { theta, .. }) => *theta,
        _ => return Err(requires_polar_coord()),
    };
    let Some(geom) = polar_geom(ctx) else {
        return Ok(MarkBuildResult::empty(MarkBatchKind::Arc));
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
        return Ok(build_annular(
            ctx, &geom, theta_field, theta2_field, radius_field, radius2_field, radius_scale,
        ));
    }

    let Some(field) = theta_field else {
        return Ok(MarkBuildResult::empty(MarkBatchKind::Arc));
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
        return Ok(build_nominal_theta(ctx, &geom, field, radius_field, radius_scale));
    }

    let Ok(values) = col_as_f64(ctx.batch, field) else {
        return Ok(MarkBuildResult::empty(MarkBatchKind::Arc));
    };

    let (cx, cy, inner_radius, outer_radius, start_angle, pad_angle) = (
        geom.cx, geom.cy, geom.inner_radius, geom.outer_radius, geom.start_angle, geom.pad_angle,
    );

    let total: f64 = values.iter()
        .filter_map(|v| *v)
        .filter(|v| v.is_finite() && *v > 0.0)
        .sum();
    if total <= 0.0 {
        return Ok(MarkBuildResult::empty(MarkBatchKind::Arc));
    }

    // Per-slice color: read color encoding field and look up in the color scale.
    let (color_str, color_f64) = color_column_loader(ctx);

    // Per-row opacity encoding.
    let opacity_values: Option<Vec<Option<f64>>> = ctx.spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    // Collect tooltip column data up front so we can index by row later.
    let meta = MetadataColumns::from_ctx(ctx);

    let mut acc = MarkNodes::with_capacity(values.len());
    // Ground truth for hit-testing (cycle 5): `(angle_start, angle_end)` per
    // KEPT node, pushed in the same lockstep as `acc.push` below so this
    // vec is already "kept nodes only, in node order" with no separate
    // gather step needed — see `MarkBatch::polar_spans`'s doc comment for
    // why a wedge's own emitted path geometry cannot always be inverted
    // back into this value (a rounded or zero-thickness wedge's endpoints
    // don't determine it uniquely).
    let mut spans: Vec<(f64, f64)> = Vec::with_capacity(values.len());
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
        // Arc slices are not yet threaded for provenance-gated paint clearing
        // (out of this fix round's scope — see the FillStroke clearing
        // follow-up in the task report); a cleared constant fill/stroke here
        // still serializes as `rgba(0,0,0,0.000)`, matching pre-c2 behavior.
        let (fill_base, _) = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_str.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_f64.as_ref().and_then(|v| v.get(i).copied().flatten()),
            ctx.mark_style.paint.fill,
            false,
        );
        // Resolve per-row opacity through scale if present; fall back to mark_style.paint.opacity.
        let row_opacity =
            resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);
        let fill_color = with_opacity(fill_base, row_opacity);

        let commands = wedge_path(
            cx, cy, inner_radius, outer_radius, angle_start, angle_end,
            ctx.mark_style.paint.corner_radius,
        );
        acc.push(SceneNode::Path {
            commands,
            style: to_scene_fill_stroke(
                Some(fill_color),
                false,
                ctx.mark_style.paint.stroke,
                false,
                ctx.mark_style.paint.stroke_width,
                row_opacity,
                ctx.mark_style.paint.stroke_dash.as_deref(),
            ),
            closed: true,
        }, i);
        spans.push((angle_start, angle_end));
    }

    // Build tooltip/href/description aligned to KEPT nodes only. Some rows are
    // skipped above (null or non-positive value), so the kept set is a strict
    // subset of the original batch. `build_metadata_for_indices` gathers
    // exactly the kept rows in node order so node j receives its true source
    // row's metadata.
    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    Ok(MarkBuildResult {
        kind: MarkBatchKind::Arc,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
        polar_spans: Some(spans),
    })
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
    // Polar angular band: `tau / n`, counted off the BATCH's distinct angular
    // categories — the same model `bar::build_polar` uses, and untouched by the
    // F-L04-03 band-geometry work for the same reason (spec §3 non-goals):
    // there is no pixel range, no `padding_inner`/`padding_outer`/`align`, and
    // no resolved ordinal scale behind `theta` (it is remapped in Python and
    // never resolves a `ScaleKind::Ordinal`), so there is no `bandwidth()` to
    // ask. Padding here is `pad_angle`, a separate radial-chart concept.
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
    let (color_str, color_f64) = color_column_loader(ctx);
    let opacity_values: Option<Vec<Option<f64>>> = ctx.spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let meta = MetadataColumns::from_ctx(ctx);

    let mut acc = MarkNodes::with_capacity(angle_strs.len());
    // Ground truth for hit-testing (cycle 5) — see `build`'s identical comment.
    // This is the path a `y == 0` category takes: its `outer_r` resolves to
    // exactly `geom.inner_radius` (a zero-thickness wedge), geometrically
    // indistinguishable from a full circle by any predicate over its emitted
    // `PathCmd`s alone — the reason this value must be carried, not inferred.
    let mut spans: Vec<(f64, f64)> = Vec::with_capacity(angle_strs.len());

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

        let (fill_base, _) = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_str.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_f64.as_ref().and_then(|v| v.get(i).copied().flatten()),
            ctx.mark_style.paint.fill,
            false,
        );
        let row_opacity =
            resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);
        let fill_color = with_opacity(fill_base, row_opacity);

        let commands = wedge_path(
            geom.cx, geom.cy, geom.inner_radius, outer_r, angle_start, angle_end,
            ctx.mark_style.paint.corner_radius,
        );
        acc.push(SceneNode::Path {
            commands,
            style: to_scene_fill_stroke(
                Some(fill_color),
                false,
                ctx.mark_style.paint.stroke,
                false,
                ctx.mark_style.paint.stroke_width,
                row_opacity,
                ctx.mark_style.paint.stroke_dash.as_deref(),
            ),
            closed: true,
        }, i);
        spans.push((angle_start, angle_end));
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
        polar_spans: Some(spans),
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
    let (color_str, color_f64) = color_column_loader(ctx);
    let opacity_values: Option<Vec<Option<f64>>> = ctx.spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let meta = MetadataColumns::from_ctx(ctx);

    let mut acc = MarkNodes::with_capacity(theta.len());
    // Ground truth for hit-testing (cycle 5) — see `build`'s identical comment.
    let mut spans: Vec<(f64, f64)> = Vec::with_capacity(theta.len());

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

        let (fill_base, _) = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_str.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_f64.as_ref().and_then(|v| v.get(i).copied().flatten()),
            ctx.mark_style.paint.fill,
            false,
        );
        let row_opacity =
            resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);
        let fill_color = with_opacity(fill_base, row_opacity);

        let commands = wedge_path(
            geom.cx, geom.cy, inner_r, outer_r, angle_start, angle_end,
            ctx.mark_style.paint.corner_radius,
        );
        acc.push(SceneNode::Path {
            commands,
            style: to_scene_fill_stroke(
                Some(fill_color),
                false,
                ctx.mark_style.paint.stroke,
                false,
                ctx.mark_style.paint.stroke_width,
                row_opacity,
                ctx.mark_style.paint.stroke_dash.as_deref(),
            ),
            closed: true,
        }, i);
        spans.push((angle_start, angle_end));
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
        polar_spans: Some(spans),
    }
}

/// SVG path commands for an arc wedge from `angle_start` to `angle_end`.
///
/// Angles are measured clockwise from 12 o'clock (north):
/// `x = cx + r·sin(θ)`, `y = cy − r·cos(θ)`.
///
/// `corner_radius` rounds the wedge's corners (spec §6.9) via the exact
/// tangent-circle construction d3-shape's `arc().cornerRadius()` uses
/// (`cornerTangents` in its source) — NOT the rounded-rectangle
/// approximation that treats the arc edge as locally straight near the
/// corner (that approximation shipped in cycle 1 and was found, in review,
/// to bulge the filleted boundary outside `outer_r` at ordinary radii; see
/// `rounded_wedge_path`'s doc comment for the closed-form fix). `0`/unset
/// and a full circle (no discrete corners to round) both fall through to
/// the unrounded geometry unchanged — byte-identical to pre-fillet output.
/// A zero inner radius rounds only the two outer corners; the center apex
/// (formed by two radial edges of possibly-unequal angle) stays sharp,
/// mirroring d3's own convention for `innerRadius = 0`.
pub(crate) fn wedge_path(
    cx: f64,
    cy: f64,
    inner_r: f64,
    outer_r: f64,
    angle_start: f64,
    angle_end: f64,
    corner_radius: f64,
) -> Vec<PathCmd> {
    let sweep = angle_end - angle_start;
    let full_circle = sweep.abs() >= std::f64::consts::TAU - 1e-9;

    if corner_radius > 0.0 && !full_circle {
        if let Some(cmds) =
            rounded_wedge_path(cx, cy, inner_r, outer_r, angle_start, angle_end, corner_radius)
        {
            return cmds;
        }
    }

    let mut cmds = Vec::new();
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

/// Build a corner-filleted wedge path, or `None` when the wedge is too thin
/// for any positive clamped radius on either ring (the caller then falls
/// back to the unrounded geometry).
///
/// ## Geometry: exact tangent circles, not a rounded-rectangle approximation
///
/// Cycle 1 shipped a rounded-rectangle-style approximation — trim the arc
/// edge by `cr/r` radians and the radial edge to radius `r ∓ cr`, then join
/// with an `ArcTo` of radius `cr` — reasoning that since a circle's tangent
/// is perpendicular to its radius, the two edges meet at a right angle just
/// like a rectangle's corner. That reasoning is only valid in the limit
/// `cr/r → 0`: the arc edge is genuinely curved, not straight, so treating
/// it as locally straight makes the trimmed points **not actually tangent**
/// to either edge, and at the clamp's own boundary the resulting fillet
/// visibly bulges outside `outer_r` (measured in review: a fillet radius
/// clamped to `radial_thickness/2` on a thin ring pushed the boundary ~11.5
/// px past `outer_r` on a `100`-radius wedge). This function instead solves
/// the two tangency conditions exactly, for a fillet of radius `cr` tangent
/// to BOTH a radial line through the wedge's own center AND the ring's own
/// circle (internally, for the outer ring; externally, for the inner one):
///
/// - `ρ = outer_r − cr` (outer) or `ρ = inner_r + cr` (inner) — the required
///   distance from the wedge's center to the fillet's own center, `O_f`,
///   for `O_f`'s circle of radius `cr` to be tangent to the ring.
/// - `Δ = asin(cr / ρ)` — `O_f` sits at angular offset `Δ` from the radial
///   edge (into the wedge's interior), which is also the correct trim on
///   the arc-edge side: the arc's own tangent point is colinear with the
///   wedge center and `O_f`, at radius `outer_r`/`inner_r`, i.e. at
///   `angle_edge ± Δ` — not `± cr/r`.
/// - `τ = sqrt(ρ² − cr²)` (equivalently `sqrt(r·(r∓2·cr))` — expand
///   `(r∓cr)² − cr²`) — the trim point's radius along the radial edge,
///   the foot of the perpendicular from `O_f` onto that line — not `r ∓ cr`.
///
/// With this construction the fillet circle is exactly internally tangent
/// to the outer ring (farthest point from the wedge's center is exactly
/// `ρ + cr = outer_r`, never more) and exactly externally tangent to the
/// inner ring (nearest point is exactly `ρ − cr = inner_r`, never less) —
/// the boundary-overshoot defect is structurally impossible, not merely
/// clamped smaller. Both `Δ`/`τ` reduce to the cycle-1 approximation's
/// `cr/r`/`r∓cr` as `cr/ρ → 0`, so this is a strict correction, not a
/// behavior change at realistic (small relative to the ring) radii.
///
/// The corner-turn direction (incoming edge → outgoing edge) has the same
/// rotational handedness at all four corners (`outer-start`, `outer-end`,
/// `inner-end`, `inner-start`), so every fillet `ArcTo` uses the same
/// `large_arc: false, sweep: true` regardless of which corner it is —
/// verified algebraically, not per corner (see the cycle-1 doc history in
/// git blame for the derivation of the shared cross-product sign).
///
/// ## Clamp
///
/// Per spec §6.9 each corner's radius is independently clamped to
/// `min(requested, radial_leg, angular_leg)`, where both legs are the exact
/// closed-form inverse of this construction's excursion functions (not the
/// spec's literal `radial_thickness/2` / `r·sweep/2`, which assumed the
/// rejected approximation and would under-clamp here — see each leg's own
/// comment below for the derivation). Both legs converge to the spec's
/// literal formulas as `cr/r → 0`. A corner whose own clamp resolves to `0`
/// keeps its original sharp point while its neighbors still round; per the
/// guards enforced by every `wedge_path` call site (a positive sweep, since
/// `angle_end` exceeds `angle_start`) and by this function's own positive-
/// radial-thickness bailout, both `outer_r` and (whenever `has_inner`)
/// `inner_r` are always strictly positive once we reach the clamp math
/// below, and both `cr_outer` and (whenever `has_inner`) `cr_inner` are
/// therefore provably strictly positive too — asserted via `debug_assert!`,
/// not branched on, since the sharp-corner fallback the branch would guard
/// is unreachable given those invariants.
fn rounded_wedge_path(
    cx: f64,
    cy: f64,
    inner_r: f64,
    outer_r: f64,
    angle_start: f64,
    angle_end: f64,
    corner_radius: f64,
) -> Option<Vec<PathCmd>> {
    let sweep = angle_end - angle_start;
    let has_inner = inner_r > 0.0;
    let radial_thickness = outer_r - inner_r.max(0.0);
    if radial_thickness <= 0.0 {
        return None;
    }
    let half_thickness = radial_thickness / 2.0;
    let half_sweep = sweep / 2.0;
    let sin_half_sweep = half_sweep.sin();

    // Radial leg: solve `outer_r − sqrt(outer_r·(outer_r − 2·cr)) ==
    // half_thickness` for `cr` (the outer fillet's excursion past
    // `outer_r − half_thickness` — the ring's midline — must not exceed
    // `half_thickness`, else it crosses into the inner fillet's own half).
    // A difference-of-squares simplification of that equation gives this
    // closed form directly (no iteration).
    let cr_limit_radial_outer =
        half_thickness * (2.0 * outer_r - half_thickness) / (2.0 * outer_r);
    // Angular leg: solve `asin(cr / ρ) == sweep / 2` for `cr`, where `ρ =
    // outer_r − cr` (outer) or `ρ = inner_r + cr` (inner) — each corner's
    // angular trim `Δ = asin(cr / ρ)` must not exceed half the ring's own
    // sweep, else the two corners on that ring overlap.
    //
    // Validity domain (cycle 3, recurrence of the cycle-1 failure class:
    // cycle 1 applied a small-angle approximation outside the small-angle
    // regime; cycle 2's `asin`-inverting closed form below is likewise only
    // the correct inverse on `asin`'s own range, `Δ ∈ [0, π/2]`, i.e. for
    // `sweep / 2 <= π/2` (`sweep <= π`). Since `Δ <= π/2` *always* holds by
    // construction (it's an `asin` output), the constraint `Δ <= sweep/2` is
    // satisfied **unconditionally, for every `cr`**, the moment `sweep/2 >=
    // π/2` — the leg is non-binding (`INFINITY`), not the `sin`-inverted
    // formula, which past that threshold folds and *shrinks* as the wedge
    // widens instead of relaxing. This is stated once, at `half_sweep`, and
    // enforced identically for both legs below — not patched into one leg's
    // `sweep == π` edge case, which is only the boundary of a domain that
    // starts at `sweep == π`, not the whole of it.
    let angular_leg_binds = half_sweep < std::f64::consts::FRAC_PI_2; // sweep < π
    let cr_limit_angular_outer = if angular_leg_binds {
        outer_r * sin_half_sweep / (1.0 + sin_half_sweep)
    } else {
        f64::INFINITY
    };
    let cr_outer = corner_radius.min(cr_limit_radial_outer).min(cr_limit_angular_outer);
    debug_assert!(cr_outer > 0.0, "unreachable given radial_thickness > 0, outer_r > 0, sweep > 0, corner_radius > 0");

    let cr_inner = if has_inner {
        // Mirror image of the outer legs (external, not internal, tangency):
        // `sqrt(inner_r·(inner_r + 2·cr)) − inner_r == half_thickness`, and
        // `asin(cr / (inner_r + cr)) == sweep / 2`, same validity domain
        // (`sweep < π`) as the outer leg above — `1.0 - sin_half_sweep` is
        // therefore always `> 0` on this branch (it only vanishes at
        // `sin_half_sweep == 1`, i.e. `sweep == π` exactly, which the same
        // `angular_leg_binds` guard already routes to `INFINITY`).
        let cr_limit_radial_inner =
            half_thickness * (2.0 * inner_r + half_thickness) / (2.0 * inner_r);
        let cr_limit_angular_inner = if angular_leg_binds {
            inner_r * sin_half_sweep / (1.0 - sin_half_sweep)
        } else {
            f64::INFINITY
        };
        let cr = corner_radius.min(cr_limit_radial_inner).min(cr_limit_angular_inner);
        debug_assert!(cr > 0.0, "unreachable given radial_thickness > 0, inner_r > 0, sweep > 0, corner_radius > 0");
        cr
    } else {
        0.0
    };

    // Point on the wedge's own circle of radius `r` at angle `theta`.
    let pt = |r: f64, theta: f64| (cx + r * theta.sin(), cy - r * theta.cos());

    // Outer corner geometry (internal tangency): ρ = outer_r − cr_outer.
    let outer_rho = outer_r - cr_outer;
    let outer_delta = (cr_outer / outer_rho).asin();
    let outer_tau = (outer_rho * outer_rho - cr_outer * cr_outer).max(0.0).sqrt();

    let large_arc_outer = (sweep - 2.0 * outer_delta).abs() > std::f64::consts::PI;

    let mut cmds = Vec::new();

    // Outer arc, trimmed at both ends by the outer corners' fillets.
    let (ax, ay) = pt(outer_r, angle_start + outer_delta);
    let (bx, by) = pt(outer_r, angle_end - outer_delta);
    cmds.push(PathCmd::MoveTo { x: ax, y: ay });
    cmds.push(PathCmd::ArcTo {
        rx: outer_r, ry: outer_r, rotation: 0.0, large_arc: large_arc_outer, sweep: true, x: bx, y: by,
    });

    // Outer-end corner: fillet from the outer arc onto the θ1 radial edge.
    let (bpx, bpy) = pt(outer_tau, angle_end);
    cmds.push(PathCmd::ArcTo {
        rx: cr_outer, ry: cr_outer, rotation: 0.0, large_arc: false, sweep: true, x: bpx, y: bpy,
    });

    if has_inner {
        // Inner corner geometry (external tangency): ρ = inner_r + cr_inner.
        let inner_rho = inner_r + cr_inner;
        let inner_delta = (cr_inner / inner_rho).asin();
        let inner_tau = (inner_rho * inner_rho - cr_inner * cr_inner).max(0.0).sqrt();
        let large_arc_inner = (sweep - 2.0 * inner_delta).abs() > std::f64::consts::PI;

        // θ1 radial edge, trimmed at the inner end by the inner-end fillet.
        let (cpx, cpy) = pt(inner_tau, angle_end);
        cmds.push(PathCmd::LineTo { x: cpx, y: cpy });

        // Inner-end corner: fillet from the radial edge onto the inner arc.
        let (dx, dy) = pt(inner_r, angle_end - inner_delta);
        cmds.push(PathCmd::ArcTo {
            rx: cr_inner, ry: cr_inner, rotation: 0.0, large_arc: false, sweep: true, x: dx, y: dy,
        });

        // Inner arc, traveling from θ1 back to θ0 (decreasing angle).
        let (ex, ey) = pt(inner_r, angle_start + inner_delta);
        cmds.push(PathCmd::ArcTo {
            rx: inner_r, ry: inner_r, rotation: 0.0, large_arc: large_arc_inner, sweep: false, x: ex, y: ey,
        });

        // Inner-start corner: fillet from the inner arc onto the θ0 radial edge.
        let (fpx, fpy) = pt(inner_tau, angle_start);
        cmds.push(PathCmd::ArcTo {
            rx: cr_inner, ry: cr_inner, rotation: 0.0, large_arc: false, sweep: true, x: fpx, y: fpy,
        });

        // θ0 radial edge, back out to the outer-start fillet's trim point.
        let (apx, apy) = pt(outer_tau, angle_start);
        cmds.push(PathCmd::LineTo { x: apx, y: apy });
    } else {
        // No inner ring: the θ1 edge runs straight to the center apex (left
        // sharp, per the doc comment above) and back out along θ0.
        cmds.push(PathCmd::LineTo { x: cx, y: cy });
        let (apx, apy) = pt(outer_tau, angle_start);
        cmds.push(PathCmd::LineTo { x: apx, y: apy });
    }

    // Outer-start corner: fillet from the θ0 radial edge back onto the outer
    // arc, closing exactly at the path's start point (`Close` becomes a
    // zero-length no-op).
    cmds.push(PathCmd::ArcTo {
        rx: cr_outer, ry: cr_outer, rotation: 0.0, large_arc: false, sweep: true, x: ax, y: ay,
    });

    cmds.push(PathCmd::Close);
    Some(cmds)
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
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx).unwrap();
        let paths = result.nodes.iter().filter(|n| matches!(n, SceneNode::Path { .. })).count();
        assert_eq!(paths, 3, "expected 3 arc Path nodes, got {paths}");
    }

    /// `build`'s OWN non-Polar guard (the merged-flat backstop, quality
    /// review round 1 / Required Fix 1) — called directly, bypassing scale
    /// resolution entirely, so this pins THIS function's arm specifically
    /// rather than the `scale_resolve` chokepoint pinned in
    /// `scale_resolve::tests`. Both `x` AND `y` are bound (mirroring
    /// `arc_both_x_and_y_bound_under_non_polar_coord_refuses`): pre-fix this
    /// exact shape reached this function's body with nothing missing to
    /// short-circuit on, and fell through to a silent empty result — the
    /// branch a merged-flat `LayerChart` layer takes, since it never goes
    /// through `resolve_scales_with_leaf_context` for the arc layer's own
    /// mark.
    #[test]
    fn build_own_guard_refuses_non_polar_coord_with_both_x_and_y() {
        let mut spec = polar_spec(false);
        spec.coord = Some(SpecCoord::Cartesian {
            x_domain: None,
            y_domain: None,
            expand: true,
            clip: true,
        });
        spec.encoding.y = Some(EncodingSpec {
            field: "val".into(),
            type_: Some(SDT::Quantitative),
            ..Default::default()
        });
        let batch = make_batch(false);
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = make_scales(false);
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        // `MarkBuildResult` (the `Ok` type) carries no `Debug` impl, so
        // `expect_err` (which requires `T: Debug` for its panic message)
        // can't be used here — match explicitly instead.
        let err = match build(&ctx) {
            Err(e) => e,
            Ok(_) => panic!(
                "mark_arc's own build() must refuse a non-Polar coord even with both x and y bound"
            ),
        };
        match &err {
            RenderError::MarkRequiresCoord { mark, required, .. } => {
                assert_eq!(*mark, "mark_arc");
                assert_eq!(*required, "Polar");
            }
            other => panic!("expected MarkRequiresCoord, got {other:?}"),
        }
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx).unwrap();

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
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
            x2: Some("t1".into()), y2: Some("r1".into()),
            y_slots: Default::default(),
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx).unwrap();

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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx).unwrap();

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

    /// Producer-side alignment for `polar_spans` (batch E Task 2, cycle 6 /
    /// quality-review finding 8), mirroring `pie_skipped_null_href_stays_aligned`
    /// immediately above: `polar_spans` is the sixth per-node channel
    /// (`mark_nodes.rs`'s alignment guard now checks it too, finding 7), and
    /// until this test nothing in the tree asserted that `build()` actually
    /// emits it aligned to its kept nodes — only that the guard would panic
    /// if it didn't.
    ///
    /// 5 rows, values `[10.0, null, 20.0, 0.0, 30.0]`. Both the null (row 1)
    /// and the zero value (row 3) are skipped by the same `Some(v) if
    /// v.is_finite() && *v > 0.0` guard — `total` (used for each kept row's
    /// sweep) only ever sums the three kept values, `10 + 20 + 30 = 60`.
    /// Kept rows are 0, 2, 4; their cumulative sweeps are exactly `tau/6`,
    /// `tau/3`, `tau/2` (`10/60`, `20/60`, `30/60` of a full turn), so the
    /// expected spans are `(0, tau/6)`, `(tau/6, tau/2)`, `(tau/2, tau)` —
    /// hand-computed from the `total`/`sweep` formula `build()` itself uses,
    /// not read off a Rust run.
    #[test]
    fn build_polar_spans_aligned_through_null_and_zero_skips() {
        let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Float64, true)]));
        let val_array = Float64Array::from(vec![Some(10.0_f64), None, Some(20.0), Some(0.0), Some(30.0)]);
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![Arc::new(val_array)]).unwrap();

        let spec = polar_spec(false);
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = make_scales(false);
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx).unwrap();

        assert_eq!(result.nodes.len(), 3, "expected 3 kept nodes (null row 1 and zero row 3 skipped)");
        let spans = result.polar_spans.expect("build() must always emit polar_spans for an Arc batch");
        assert_eq!(spans.len(), 3, "polar_spans must have exactly one entry per kept node");
        assert_eq!(
            result.data_indices.as_deref(),
            Some(&[0usize, 2, 4][..]),
            "kept rows must be 0 (val=10), 2 (val=20), 4 (val=30)"
        );

        let tau = std::f64::consts::TAU;
        let tol = 1e-9;
        let expected = [(0.0, tau / 6.0), (tau / 6.0, tau / 2.0), (tau / 2.0, tau)];
        for (i, (exp_start, exp_end)) in expected.iter().enumerate() {
            let (start, end) = spans[i];
            assert!((start - exp_start).abs() < tol && (end - exp_end).abs() < tol,
                "node {i}'s span must be ({exp_start}, {exp_end}); got ({start}, {end})");
        }
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
            color: None, size: None, shape: None, opacity: None, fill_opacity: None, stroke_opacity: None, stroke_dash: None, x2: None, y2: None, y_slots: Default::default(),
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx).unwrap();

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

    /// Producer-side `polar_spans` alignment for `build_nominal_theta`
    /// (cycle 6 / quality-review finding 8) — the exact code path the
    /// reviewer's decisive zero-thickness finding travels
    /// (`fr.Chart(df).mark_arc().encode(x="c:N", y="v:Q").coord(fr.CoordPolar(
    /// inner_radius=60.0))`). 5 categories `a, b, c, d, e`; `b`'s category is
    /// null (the explicit `None => continue` skip branch) and `d`'s radius
    /// value is `0.0` (a zero-thickness wedge — kept, NOT skipped, since a
    /// zero *radius* is a different condition from a null *category*; this
    /// is the shape that made every geometry-based hit-test classifier in
    /// cycles 2–4 fail). 4 distinct categories survive in `cat_index`
    /// (`a, c, d, e` — `b` is null and never enters it), so `band = tau/4 =
    /// π/2` and the kept rows' spans are exactly `(0, π/2)`, `(π/2, π)`,
    /// `(π, 3π/2)`, `(3π/2, tau)` regardless of `d`'s degenerate radius.
    #[test]
    fn build_nominal_theta_polar_spans_aligned_through_null_skip_and_zero_radius() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, true),
            Field::new("r", DataType::Float64, true),
        ]));
        let cat_array = StringArray::from(vec![Some("a"), None, Some("c"), Some("d"), Some("e")]);
        let r_array = Float64Array::from(vec![Some(10.0_f64), Some(999.0), Some(25.0), Some(0.0), Some(35.0)]);
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(cat_array),
            Arc::new(r_array),
        ]).unwrap();

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Arc,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Nominal), ..Default::default() }),
                y: Some(EncodingSpec { field: "r".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: Some(SpecCoord::Polar {
                theta: PolarThetaChannel::X,
                start_angle: 0.0,
                inner_radius: 60.0,
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
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 35.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None, fill_opacity: None, stroke_opacity: None, stroke_dash: None, x2: None, y2: None, y_slots: Default::default(),
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx).unwrap();

        assert_eq!(result.nodes.len(), 4, "expected 4 kept nodes (only the null category, row 1, is skipped)");
        let spans = result.polar_spans.expect("build_nominal_theta must always emit polar_spans for an Arc batch");
        assert_eq!(spans.len(), 4, "polar_spans must have exactly one entry per kept node");
        assert_eq!(
            result.data_indices.as_deref(),
            Some(&[0usize, 2, 3, 4][..]),
            "kept rows must be 0 (a), 2 (c), 3 (d, zero radius), 4 (e) — row 1 (null category b) skipped"
        );

        let frac_pi_2 = std::f64::consts::FRAC_PI_2;
        let pi = std::f64::consts::PI;
        let tau = std::f64::consts::TAU;
        let expected = [(0.0, frac_pi_2), (frac_pi_2, pi), (pi, pi + frac_pi_2), (pi + frac_pi_2, tau)];
        let tol = 1e-9;
        for (i, (exp_start, exp_end)) in expected.iter().enumerate() {
            let (start, end) = spans[i];
            assert!((start - exp_start).abs() < tol && (end - exp_end).abs() < tol,
                "node {i}'s span must be ({exp_start}, {exp_end}); got ({start}, {end})");
        }
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
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
            x2: Some("t1".into()), y2: Some("r1".into()),
            y_slots: Default::default(),
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx).unwrap();

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

    /// Producer-side `polar_spans` alignment for `build_annular` (cycle 6 /
    /// quality-review finding 8). 4 rows: row 1 has a null `t1` (the explicit
    /// skip branch — no angular span); row 2 has `r0 == r1` (a
    /// zero-radial-thickness wedge — kept, not skipped, the same degenerate
    /// shape that decided cycle 4). `build_annular`'s
    /// `angle_start`/`angle_end` are exactly `start_angle + t0 ± pad/2`, so
    /// with `pad_angle = 0` the expected spans are literally each kept row's
    /// own `(t0, t1)` pair.
    #[test]
    fn build_annular_polar_spans_aligned_through_null_skip_and_zero_thickness() {
        let tau = std::f64::consts::TAU;
        let schema = Arc::new(Schema::new(vec![
            Field::new("t0", DataType::Float64, false),
            Field::new("t1", DataType::Float64, true),
            Field::new("r0", DataType::Float64, false),
            Field::new("r1", DataType::Float64, false),
        ]));
        let t0_arr = Float64Array::from(vec![0.0_f64, tau / 4.0, tau / 2.0, 3.0 * tau / 4.0]);
        let t1_arr = Float64Array::from(vec![Some(tau / 4.0), None, Some(3.0 * tau / 4.0), Some(tau)]);
        let r0_arr = Float64Array::from(vec![0.0_f64, 0.0, 40.0, 0.0]);
        let r1_arr = Float64Array::from(vec![80.0_f64, 80.0, 40.0, 80.0]); // row 2: r0 == r1 (zero thickness)
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(t0_arr),
            Arc::new(t1_arr),
            Arc::new(r0_arr),
            Arc::new(r1_arr),
        ]).unwrap();

        let spec = ChartSpec {
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
        };
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, tau], vec![0.0, 200.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 80.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None,
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
            x2: Some("t1".into()), y2: Some("r1".into()),
            y_slots: Default::default(),
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx).unwrap();

        assert_eq!(result.nodes.len(), 3, "expected 3 kept nodes (row 1's null t1 skipped)");
        let spans = result.polar_spans.expect("build_annular must always emit polar_spans for an Arc batch");
        assert_eq!(spans.len(), 3, "polar_spans must have exactly one entry per kept node");
        assert_eq!(
            result.data_indices.as_deref(),
            Some(&[0usize, 2, 3][..]),
            "kept rows must be 0, 2 (zero-thickness, r0==r1==40), 3 — row 1 (null t1) skipped"
        );

        let tol = 1e-9;
        let expected = [(0.0, tau / 4.0), (tau / 2.0, 3.0 * tau / 4.0), (3.0 * tau / 4.0, tau)];
        for (i, (exp_start, exp_end)) in expected.iter().enumerate() {
            let (start, end) = spans[i];
            assert!((start - exp_start).abs() < tol && (end - exp_end).abs() < tol,
                "node {i}'s span must be ({exp_start}, {exp_end}); got ({start}, {end})");
        }
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx).unwrap();

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
        let cmds = wedge_path(100.0, 100.0, 0.0, 50.0, 0.0, std::f64::consts::TAU - 1e-10, 0.0);
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
        let cmds = wedge_path(100.0, 100.0, r, r, 0.0, std::f64::consts::FRAC_PI_2, 0.0);
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
        let cmds = wedge_path(100.0, 100.0, 0.0, 50.0, start, start + sweep, 0.0);
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = build(&ctx).unwrap();

        let paths = result.nodes.iter().filter(|n| matches!(n, SceneNode::Path { .. })).count();
        assert_eq!(paths, 2, "the pad-exceeds-sweep row must be skipped, leaving 2 wedges");
    }

    // ── NF-A3: the pie path's per-slice color read (spec §4.4) ───────────────

    /// An `Int64` nominal color column must tint the slices, not leave all
    /// three in the first palette color.
    ///
    /// RED (verified in place): the three inline `(color_input, cfield)`
    /// matches this file carried read `col_as_str`, which returns `Err` for
    /// `Int64`; `.ok()` swallowed it, `color_str` was `None`, and every slice
    /// took `resolve_fill_color`'s constant fallback while the legend
    /// enumerated all three categories. The `Utf8` twin is built from the same
    /// categories in the same row order and asserted equal, so the test cannot
    /// pass by collapsing both.
    #[test]
    fn arc_tints_slices_from_an_int64_nominal_color() {
        use arrow::array::Int64Array;

        let slice_fills = |color_field: Field, color_col: Arc<dyn arrow::array::Array>| -> Vec<String> {
            let mut spec = polar_spec(false);
            spec.encoding.color =
                Some(EncodingSpec { field: "grp".into(), type_: Some(SDT::Nominal), ..Default::default() });
            let schema = Arc::new(Schema::new(vec![
                Field::new("val", DataType::Float64, false),
                color_field,
            ]));
            let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
                Arc::new(Float64Array::from(vec![10.0, 30.0, 60.0])),
                color_col,
            ]).unwrap();
            let theme = ThemeInputs::default();
            let panel = make_panel();
            let mut scales = make_scales(false);
            // Resolve the color scale through the production resolver rather
            // than hand-building one: a hand-built `Categorical` would pin the
            // domain strings this test is trying to prove the reader agrees
            // with. Arc's polar spec binds no `y`, and `resolve_scales`
            // requires both positional channels, so the color scale is taken
            // from an equivalent cartesian spec over the same batch.
            let mut scale_spec = spec.clone();
            scale_spec.mark = Mark::Point;
            scale_spec.coord = None;
            scale_spec.encoding.y =
                Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() });
            let (resolved, _) = crate::render::scale_resolve::resolve_scales(
                &scale_spec, &batch, (0.0, 200.0), (0.0, 200.0), &theme,
            ).unwrap();
            scales.color = resolved.color;
            assert!(scales.color.is_some(), "fixture must resolve a color scale");
            let mark_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
            let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
            build(&ctx).unwrap().nodes.iter().filter_map(|n| match n {
                SceneNode::Path { style, .. } => style.fill.map(|c| format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b)),
                _ => None,
            }).collect()
        };

        let utf8 = slice_fills(
            Field::new("grp", DataType::Utf8, false),
            Arc::new(StringArray::from(vec!["1", "2", "3"])),
        );
        assert_eq!(utf8.len(), 3, "Utf8 reference must produce 3 slices");
        assert_eq!(
            utf8.iter().collect::<std::collections::BTreeSet<_>>().len(),
            3,
            "Utf8 reference must give three distinct slice fills; got {utf8:?}"
        );

        let int64 = slice_fills(
            Field::new("grp", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
        );
        assert_eq!(
            int64, utf8,
            "an Int64 nominal color column must tint the slices exactly as its \
             Utf8 twin does; got {int64:?}. Three identical fills here is the \
             col_as_str drop."
        );
    }

    // ── Task 2 (batch E, F-L02-06): corner_radius fillets ───────────────────
    //
    // `wedge_path` gained a 7th `corner_radius` parameter (spec §6.9). These
    // tests prove: (1) `0.0` reproduces the pre-fix sharp geometry exactly
    // (byte-stability contract), (2) a positive radius actually rounds (adds
    // small-radius fillet `ArcTo` commands distinct from the main-radius
    // arc), (3) the per-corner clamp binds on whichever of
    // `radial_thickness/2` or the angular arc half-length is smaller, on
    // both the outer and inner ring, and (4) a full-circle sweep and a
    // zero-thickness ring both refuse to round rather than emitting
    // self-intersecting geometry.

    /// `corner_radius = 0.0` must produce the exact same 4-command sharp
    /// geometry the pre-fix `wedge_path` always emitted for a solid
    /// (`inner_r == 0`) quarter-wedge — pinned to hand-computed coordinates,
    /// not a self-comparison, so a regression that quietly rounds at 0.0
    /// cannot hide behind an unchanged command count.
    #[test]
    fn wedge_path_zero_corner_radius_matches_sharp_geometry() {
        let cmds = wedge_path(0.0, 0.0, 0.0, 100.0, 0.0, std::f64::consts::FRAC_PI_2, 0.0);
        assert_eq!(cmds.len(), 4, "MoveTo + outer ArcTo + LineTo(center) + Close");
        match cmds[0] {
            PathCmd::MoveTo { x, y } => assert!((x - 0.0).abs() < 1e-9 && (y - -100.0).abs() < 1e-9,
                "MoveTo must land at (0, -100); got ({x}, {y})"),
            ref c => panic!("expected MoveTo, got {c:?}"),
        }
        match cmds[1] {
            PathCmd::ArcTo { rx, ry, large_arc, sweep, x, y, .. } => {
                assert_eq!(rx, 100.0);
                assert_eq!(ry, 100.0);
                assert!(!large_arc, "a PI/2 sweep is not a large arc");
                assert!(sweep, "increasing-angle sweep must set sweep=true");
                assert!((x - 100.0).abs() < 1e-9 && (y - 0.0).abs() < 1e-9,
                    "outer ArcTo endpoint must land at (100, 0); got ({x}, {y})");
            }
            ref c => panic!("expected outer ArcTo, got {c:?}"),
        }
        assert!(matches!(cmds[2], PathCmd::LineTo { x, y } if (x - 0.0).abs() < 1e-9 && (y - 0.0).abs() < 1e-9));
        assert!(matches!(cmds[3], PathCmd::Close));
    }

    /// A positive `corner_radius` on the same solid quarter-wedge must round:
    /// the sharp corner-count (4 commands, 1 arc) grows to 7 commands with 3
    /// arcs — the original outer sweep plus two small-radius fillets at the
    /// resolved (unclamped, since 10.0 is well under both clamp limits here)
    /// corner radius.
    #[test]
    fn wedge_path_positive_corner_radius_rounds_and_differs_from_sharp() {
        let sharp = wedge_path(0.0, 0.0, 0.0, 100.0, 0.0, std::f64::consts::FRAC_PI_2, 0.0);
        let rounded = wedge_path(0.0, 0.0, 0.0, 100.0, 0.0, std::f64::consts::FRAC_PI_2, 10.0);
        assert_ne!(sharp.len(), rounded.len(), "rounding must change the command shape");
        assert_eq!(rounded.len(), 7, "MoveTo + outer ArcTo + fillet ArcTo + LineTo(center) + LineTo + fillet ArcTo + Close");

        let arc_radii: Vec<f64> = rounded.iter().filter_map(|c| {
            if let PathCmd::ArcTo { rx, .. } = c { Some(*rx) } else { None }
        }).collect();
        assert_eq!(arc_radii.len(), 3, "expected 1 main arc + 2 corner fillets; got {arc_radii:?}");
        let fillets = arc_radii.iter().filter(|r| (**r - 10.0).abs() < 1e-9).count();
        let mains = arc_radii.iter().filter(|r| (**r - 100.0).abs() < 1e-9).count();
        assert_eq!(fillets, 2, "both outer corners must fillet at the unclamped 10.0 radius; got {arc_radii:?}");
        assert_eq!(mains, 1, "the outer sweep itself must keep radius 100.0; got {arc_radii:?}");

        for cmd in &rounded {
            if let PathCmd::ArcTo { rx, ry, x, y, .. } = cmd {
                assert!(rx.is_finite() && ry.is_finite() && x.is_finite() && y.is_finite());
            }
        }
        // The path must still close back to its own MoveTo point exactly.
        let (mx, my) = match rounded[0] { PathCmd::MoveTo { x, y } => (x, y), _ => unreachable!() };
        let (lx, ly) = match rounded[5] { PathCmd::ArcTo { x, y, .. } => (x, y), ref c => panic!("expected closing fillet ArcTo, got {c:?}") };
        assert!((mx - lx).abs() < 1e-9 && (my - ly).abs() < 1e-9,
            "the final fillet must land exactly back on the MoveTo point; ({mx},{my}) vs ({lx},{ly})");
    }

    /// An annular wedge (`inner_r > 0`) rounds all four corners: the sharp
    /// 5-command geometry (`MoveTo, ArcTo, LineTo, ArcTo, Close`) grows to 10
    /// commands with 6 arcs (2 main rings + 4 fillets), and — the discriminating
    /// assertion — both inner corners fillet too, not just the outer ones.
    #[test]
    fn wedge_path_annular_rounds_all_four_corners() {
        let rounded = wedge_path(0.0, 0.0, 40.0, 60.0, 0.0, std::f64::consts::FRAC_PI_2, 5.0);
        assert_eq!(rounded.len(), 10,
            "MoveTo + outer ArcTo + fillet + LineTo + fillet + inner ArcTo + fillet + LineTo + fillet + Close");
        let arc_radii: Vec<f64> = rounded.iter().filter_map(|c| {
            if let PathCmd::ArcTo { rx, .. } = c { Some(*rx) } else { None }
        }).collect();
        assert_eq!(arc_radii.len(), 6, "expected 2 main arcs + 4 corner fillets; got {arc_radii:?}");
        let fillets = arc_radii.iter().filter(|r| (**r - 5.0).abs() < 1e-9).count();
        assert_eq!(fillets, 4, "all four corners (outer x2, inner x2) must fillet; got {arc_radii:?}");
        assert!(arc_radii.iter().any(|r| (*r - 60.0).abs() < 1e-9), "outer main arc radius must survive; got {arc_radii:?}");
        assert!(arc_radii.iter().any(|r| (*r - 40.0).abs() < 1e-9), "inner main arc radius must survive; got {arc_radii:?}");
    }

    /// Cycle 2 (Required Fix 1, spec-reviewer + rust-quality-reviewer):
    /// the fillet must be the exact tangent-circle construction (matching
    /// d3-shape's `cornerTangents`), not the rounded-rectangle approximation
    /// that treats the arc edge as locally straight. For an outer corner
    /// with fillet radius `cr` at radius `outer_r`: `ρ = outer_r − cr`,
    /// `Δ = asin(cr/ρ)` (the angular trim on the arc side), `τ = sqrt(ρ² −
    /// cr²)` (the radius of the trim point on the radial edge — NOT `outer_r
    /// − cr`, which is the rejected approximation). This test pins both
    /// trim points of the outer-end corner and the outer-start corner
    /// against values computed independently (a standalone Python script,
    /// not read off a Rust run) for `outer_r=100, inner_r=0, angle_start=0,
    /// angle_end=π/2, cr=20` (well under both clamp limits, so `cr_outer ==
    /// 20` exactly): `ρ=80`, `Δ=asin(0.25)≈0.25268025514207865`,
    /// `τ=sqrt(6000)≈77.45966692414834`.
    #[test]
    fn wedge_path_outer_fillet_matches_exact_tangent_construction() {
        let cmds = wedge_path(0.0, 0.0, 0.0, 100.0, 0.0, std::f64::consts::FRAC_PI_2, 20.0);
        // [MoveTo(A), ArcTo(B), ArcTo(B'), LineTo(center), LineTo(A'), ArcTo(A), Close]
        assert_eq!(cmds.len(), 7);
        let tol = 1e-9;

        // A = outer arc tangent point at angle_start+Δ = pt(100, 0.25268025514207865).
        match cmds[0] {
            PathCmd::MoveTo { x, y } => assert!((x - 25.0).abs() < tol && (y - -96.82458365518542).abs() < tol,
                "outer-start arc tangent point A must be (25.0, -96.82458365518542) — the EXACT tangent point at angle_start+asin(0.25), not the rounded-rectangle approximation's pt(100, cr/outer_r); got ({x}, {y})"),
            ref c => panic!("expected MoveTo, got {c:?}"),
        }
        // B' = outer-end radial-edge tangent point at radius τ, angle_end (π/2).
        match cmds[2] {
            PathCmd::ArcTo { x, y, .. } => assert!((x - 77.45966692414834).abs() < tol && y.abs() < 1e-6,
                "outer-end radial tangent point B' must be (77.45966692414834, ~0.0) at radius τ=sqrt(6000), not outer_r-cr=80.0 (the rejected approximation); got ({x}, {y})"),
            ref c => panic!("expected outer-end fillet ArcTo, got {c:?}"),
        }
        // A' = outer-start radial-edge tangent point at radius τ, angle_start (0).
        match cmds[4] {
            PathCmd::LineTo { x, y } => assert!(x.abs() < 1e-6 && (y - -77.45966692414834).abs() < tol,
                "outer-start radial tangent point A' must be (~0.0, -77.45966692414834) at radius τ, not outer_r-cr=80.0; got ({x}, {y})"),
            ref c => panic!("expected LineTo(A'), got {c:?}"),
        }
    }

    /// Same as above but for an inner corner (external tangency to the inner
    /// ring, rather than internal tangency to the outer ring): `inner_r=40,
    /// outer_r=100, angle_start=0, angle_end=π/2, cr=15` (unclamped: radial
    /// limit ≈41.25, angular limit ≈96.57, both ≫15). `ρ = inner_r+cr = 55`,
    /// `Δ = asin(15/55) ≈ 0.27622663076359155`, `τ = sqrt(55²−15²) =
    /// sqrt(2800) ≈ 52.91502622129181`. Values hand-computed independently.
    #[test]
    fn wedge_path_inner_fillet_matches_exact_tangent_construction() {
        let cmds = wedge_path(0.0, 0.0, 40.0, 100.0, 0.0, std::f64::consts::FRAC_PI_2, 15.0);
        // [MoveTo(A), ArcTo(B), ArcTo(B'), LineTo(C'), ArcTo(D), ArcTo(E), ArcTo(F'), LineTo(A'), ArcTo(A), Close]
        assert_eq!(cmds.len(), 10);
        let tol = 1e-9;

        // D = inner-end arc tangent point at angle_end-Δ = pt(40, π/2 - 0.27622663076359155).
        match cmds[4] {
            PathCmd::ArcTo { x, y, .. } => assert!((x - 38.48365543366677).abs() < tol && (y - -10.909090909090907).abs() < tol,
                "inner-end arc tangent point D must be (38.48365543366677, -10.909090909090907) — the EXACT tangent point, not the rounded-rectangle approximation's pt(40, angle_end - cr/inner_r); got ({x}, {y})"),
            ref c => panic!("expected inner-end fillet ArcTo, got {c:?}"),
        }
        // C' = inner-end radial-edge tangent point at radius τ, angle_end (π/2).
        match cmds[3] {
            PathCmd::LineTo { x, y } => assert!((x - 52.91502622129181).abs() < tol && y.abs() < 1e-6,
                "inner-end radial tangent point C' must be (52.91502622129181, ~0.0) at radius τ=sqrt(2800), not inner_r+cr=55.0 (the rejected approximation); got ({x}, {y})"),
            ref c => panic!("expected LineTo(C'), got {c:?}"),
        }
        // E = inner-start arc tangent point at angle_start+Δ = pt(40, 0.27622663076359155).
        match cmds[5] {
            PathCmd::ArcTo { x, y, .. } => assert!((x - 10.909090909090908).abs() < tol && (y - -38.48365543366677).abs() < tol,
                "inner-start arc tangent point E must be (10.909090909090908, -38.48365543366677); got ({x}, {y})"),
            ref c => panic!("expected inner main ArcTo, got {c:?}"),
        }
        // F' = inner-start radial-edge tangent point at radius τ, angle_start (0).
        match cmds[6] {
            PathCmd::ArcTo { x, y, .. } => assert!(x.abs() < 1e-6 && (y - -52.91502622129181).abs() < tol,
                "inner-start radial tangent point F' must be (~0.0, -52.91502622129181) at radius τ, not inner_r+cr=55.0; got ({x}, {y})"),
            ref c => panic!("expected inner-start fillet ArcTo, got {c:?}"),
        }
    }

    /// Reconstruct an SVG circular arc's own center from its two endpoints,
    /// radius, and `sweep` flag (this codebase never emits `rotation != 0`
    /// or `rx != ry`, so the general elliptical-arc endpoint parameterization
    /// collapses to the standard "circle through two points with a given
    /// radius" construction). Used by the containment test below to check
    /// the *whole fillet circle* against the ring bounds, not just its two
    /// listed endpoints — a curve can bulge outside `[inner_r, outer_r]`
    /// between its endpoints even when both endpoints are in bounds, which
    /// is exactly how the rejected rounded-rectangle approximation escaped
    /// the outer ring (cycle 2, Required Fix 1).
    /// Cycle 3 correction: a prior version of this helper picked between the
    /// two candidate centers using only `sweep` (a cross-product-sign
    /// convention derived from this file's `large_arc: false` fillet arcs),
    /// which is exactly the SVG endpoint-parameterization's `large_arc: true`
    /// case wrong — the standard requires BOTH flags together (`sign = -1`
    /// iff `large_arc == sweep`), and the wide-sweep tests below (cycle 3,
    /// `sweep > π`) exercise a `large_arc: true` main arc for the first time,
    /// which caught it (a spuriously "bulging" 596 px reconstruction of an
    /// otherwise-correct 202 px arc). Implements the standard SVG
    /// elliptical-arc endpoint→center formula directly (rx = ry = r,
    /// rotation = 0 collapses it to this), rather than a derived heuristic.
    fn arc_center(p0: (f64, f64), p1: (f64, f64), r: f64, large_arc: bool, sweep: bool) -> (f64, f64) {
        let x1p = (p0.0 - p1.0) / 2.0;
        let y1p = (p0.1 - p1.1) / 2.0;
        let d2 = x1p * x1p + y1p * y1p;
        if d2 < 1e-24 {
            return p0;
        }
        let sign = if large_arc == sweep { -1.0 } else { 1.0 };
        let ratio = ((r * r - d2) / d2).max(0.0);
        let co = sign * ratio.sqrt();
        let cxp = co * y1p;
        let cyp = -co * x1p;
        (cxp + (p0.0 + p1.0) / 2.0, cyp + (p0.1 + p1.1) / 2.0)
    }

    /// Containment: no point on ANY emitted arc's curve (not just its listed
    /// endpoints) may lie outside `[inner_r, outer_r]` — the exact defect
    /// the rust-quality-reviewer measured (`mark_arc(corner_radius=150)` on
    /// a 3-slice pie visibly crossing the polar frame: the rounded-rectangle
    /// approximation's fillet endpoints were both in bounds, but the curve
    /// between them bulged past `outer_r`). For every `ArcTo`, reconstructs
    /// the arc's own center via [`arc_center`] and asserts the *whole
    /// circle* — `|center| ± rx` — stays within `[inner_r, outer_r]`; this
    /// is what the exact tangent construction guarantees by design (an
    /// outer fillet's circle is internally tangent to the outer ring, so its
    /// farthest point from the wedge's own center is exactly `outer_r`,
    /// never more) and what the approximation violated. Swept across solid
    /// and annular wedges and a range of requested radii including ones far
    /// past what any clamp leg alone would suggest is safe.
    #[test]
    fn wedge_path_fillet_never_exceeds_ring_bounds() {
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            // (inner_r, outer_r, angle_start, angle_end, requested corner_radius)
            (0.0, 100.0, 0.0, std::f64::consts::FRAC_PI_2, 20.0),
            (0.0, 100.0, 0.0, std::f64::consts::FRAC_PI_2, 49.0), // near the solid clamp ceiling
            (0.0, 202.0, 0.0, 2.0 * std::f64::consts::PI / 3.0, 150.0), // the reviewer's mark_arc(corner_radius=150) case, scaled to a 120° slice
            (40.0, 100.0, 0.0, std::f64::consts::FRAC_PI_2, 15.0),
            (40.0, 41.0, 0.0, std::f64::consts::FRAC_PI_2, 1000.0), // thin-ring clamp case
            // Cycle 3 (spec-reviewer Issue 1 / quality-reviewer F5): wide
            // sweeps (> π), where the angular leg is non-binding — re-verify
            // containment still holds now that those legs return INFINITY
            // instead of a folded, incorrectly-small limit.
            (0.0, 202.48, 0.0, 0.97 * std::f64::consts::TAU, 25.0), // the reviewer's [97,3] pie's dominant slice
            (0.0, 100.0, 0.0, 6.0, 1000.0),
            (40.0, 100.0, 0.0, 6.0, 1000.0),
            (40.0, 100.0, 0.0, std::f64::consts::TAU - 0.01, 1000.0), // near-full-circle, not quite
        ];
        for &(inner_r, outer_r, angle_start, angle_end, cr) in cases {
            let cmds = wedge_path(0.0, 0.0, inner_r, outer_r, angle_start, angle_end, cr);
            let has_inner = inner_r > 0.0;
            // `rounded_wedge_path`'s emission order (verified directly by
            // the coordinate-pin tests above) puts exactly 3 outer-side arcs
            // first/second/last and, when annular, exactly 3 inner-side arcs
            // in between: [outer_main, outer_end_fillet, (inner_end_fillet,
            // inner_main, inner_start_fillet), outer_start_fillet]. Only the
            // *meaningful* bound is checked per side: an outer arc's full
            // circle can legitimately dip toward the far side of its own
            // circle well below `inner_r` without the actually-drawn ~90°
            // sub-arc ever going there (and symmetrically for an inner arc's
            // far side vs `outer_r`) — checking the "wrong" bound on the
            // whole reconstructed circle is over-conservative and produces
            // false failures on thin rings, not a real defect.
            let total_arcs = cmds.iter().filter(|c| matches!(c, PathCmd::ArcTo { .. })).count();
            let mut arc_idx = 0usize;
            let mut cur = (0.0, 0.0);
            for cmd in &cmds {
                match cmd {
                    PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => {
                        let r = (x * x + y * y).sqrt();
                        assert!(r <= outer_r + 1e-6 && r >= inner_r - 1e-6,
                            "vertex ({x}, {y}) at radius {r} outside [{inner_r}, {outer_r}] for case {inner_r},{outer_r},{angle_start},{angle_end},{cr}");
                        cur = (*x, *y);
                    }
                    PathCmd::ArcTo { rx, x, y, large_arc, sweep, .. } => {
                        let is_outer_side = !has_inner || arc_idx == 0 || arc_idx == 1 || arc_idx == total_arcs - 1;
                        let center = arc_center(cur, (*x, *y), *rx, *large_arc, *sweep);
                        let dist_from_origin = (center.0 * center.0 + center.1 * center.1).sqrt();
                        if is_outer_side {
                            let farthest = dist_from_origin + rx;
                            assert!(farthest <= outer_r + 1e-6,
                                "arc to ({x}, {y}) with rx={rx} bulges to radius {farthest} > outer_r={outer_r} for case {inner_r},{outer_r},{angle_start},{angle_end},{cr}");
                        } else {
                            let nearest = (dist_from_origin - rx).abs();
                            assert!(nearest >= inner_r - 1e-6,
                                "arc to ({x}, {y}) with rx={rx} dips to radius {nearest} < inner_r={inner_r} for case {inner_r},{outer_r},{angle_start},{angle_end},{cr}");
                        }
                        cur = (*x, *y);
                        arc_idx += 1;
                    }
                    PathCmd::Close => {}
                    other => panic!("unexpected PathCmd in wedge_path output: {other:?}"),
                }
            }
        }
    }

    /// Tangency, verified geometrically rather than by re-deriving
    /// `rounded_wedge_path`'s own formula: for the outer-end corner of an
    /// unclamped fillet, the fillet's center `O_f` is independently
    /// computable from just the arc-side trim point, `cr`, and `outer_r`
    /// (`O_f` lies on the ray from the wedge center through the arc trim
    /// point, at distance `outer_r − cr`). A true tangent fillet requires
    /// BOTH `|O_f − line_pt| == cr` AND `(line_pt − O_f)` perpendicular to
    /// the radial edge's own direction — the second condition is what a
    /// merely-equal-distance-by-coincidence construction would fail.
    #[test]
    fn wedge_path_outer_fillet_is_tangent_to_both_edges() {
        let outer_r = 100.0_f64;
        let angle_end = std::f64::consts::FRAC_PI_2;
        let cr = 20.0_f64;
        let cmds = wedge_path(0.0, 0.0, 0.0, outer_r, 0.0, angle_end, cr);
        let (arc_x, arc_y) = match cmds[1] { PathCmd::ArcTo { x, y, .. } => (x, y), ref c => panic!("expected main outer ArcTo, got {c:?}") };
        let (line_x, line_y) = match cmds[2] { PathCmd::ArcTo { x, y, .. } => (x, y), ref c => panic!("expected outer-end fillet ArcTo, got {c:?}") };

        // O_f: on the ray from (0,0) through (arc_x, arc_y), at distance outer_r - cr.
        let rho = outer_r - cr;
        let ofx = arc_x * rho / outer_r;
        let ofy = arc_y * rho / outer_r;

        let dist_to_arc_trim = ((arc_x - ofx).powi(2) + (arc_y - ofy).powi(2)).sqrt();
        assert!((dist_to_arc_trim - cr).abs() < 1e-9,
            "O_f must be exactly cr={cr} from the arc-side trim point; got {dist_to_arc_trim}");

        let dist_to_line_trim = ((line_x - ofx).powi(2) + (line_y - ofy).powi(2)).sqrt();
        assert!((dist_to_line_trim - cr).abs() < 1e-9,
            "the fillet must be exactly cr={cr} from the radial-edge trim point (tangent radius), not merely close; got {dist_to_line_trim}");

        // Perpendicularity: the radial edge at angle_end has direction u(angle_end)
        // = (sin(angle_end), -cos(angle_end)); (line_pt - O_f) must be parallel to
        // this direction (a chord tangent to the line at line_pt lies ALONG the
        // line, and O_f - line_pt must be perpendicular to it — equivalently,
        // line_pt itself must lie on the radial line, i.e. atan2 must recover
        // angle_end exactly).
        let mut line_theta = line_x.atan2(-line_y);
        if line_theta < 0.0 { line_theta += std::f64::consts::TAU; }
        assert!((line_theta - angle_end).abs() < 1e-9,
            "the radial-edge trim point must lie exactly on the angle_end radial line, not merely cr from O_f in some other direction; got theta={line_theta}, expected {angle_end}");
    }

    /// A thin annulus (`radial_thickness = 1.0`) with a very large requested
    /// `corner_radius` must clamp near `radial_thickness / 2`, not the
    /// requested value — otherwise the fillet geometry would eat past the
    /// opposite ring and self-intersect.
    ///
    /// The exact clamp is the closed-form inverse of the *exact* tangent
    /// construction's excursion function (cycle 2, Required Fix 1), not the
    /// literal `radial_thickness / 2`: excursion(cr) = `r − sqrt(r·(r−2·cr))`
    /// for an outer corner, `sqrt(r·(r+2·cr)) − r` for an inner one. Solving
    /// each for `excursion = h` (`h = radial_thickness / 2`) gives
    /// `cr_limit_radial_outer = h·(2·outer_r−h) / (2·outer_r)` and
    /// `cr_limit_radial_inner = h·(2·inner_r+h) / (2·inner_r)` — both
    /// converge to `h` as `h → 0` (matching the naive formula to first
    /// order) but are strictly smaller for any positive `h`, which is what
    /// actually prevents the two fillets from crossing under exact tangency.
    /// Values below hand-computed independently (not read off a Rust run).
    #[test]
    fn wedge_path_clamps_to_half_radial_thickness_on_thin_ring() {
        let cmds = wedge_path(0.0, 0.0, 40.0, 41.0, 0.0, std::f64::consts::FRAC_PI_2, 1000.0);
        let arc_radii: Vec<f64> = cmds.iter().filter_map(|c| {
            if let PathCmd::ArcTo { rx, .. } = c { Some(*rx) } else { None }
        }).collect();
        let fillets: Vec<f64> = arc_radii.iter().copied().filter(|r| *r < 5.0).collect();
        assert_eq!(fillets.len(), 4, "expected 4 corner fillets on an annular wedge; got {arc_radii:?}");
        let outer_fillets = fillets.iter().filter(|r| (**r - 0.4969512195121951).abs() < 1e-9).count();
        let inner_fillets = fillets.iter().filter(|r| (**r - 0.503125).abs() < 1e-9).count();
        assert_eq!(outer_fillets, 2,
            "outer corners must clamp to the exact radial limit 0.4969512195121951 (not the naive 0.5), not the requested 1000.0; got {fillets:?}");
        assert_eq!(inner_fillets, 2,
            "inner corners must clamp to the exact radial limit 0.503125 (not the naive 0.5), not the requested 1000.0; got {fillets:?}");
        for cmd in &cmds {
            if let PathCmd::ArcTo { x, y, .. } | PathCmd::LineTo { x, y } | PathCmd::MoveTo { x, y } = cmd {
                assert!(x.is_finite() && y.is_finite(), "clamped geometry must stay finite");
            }
        }
    }

    /// A very narrow angular sweep with a moderate `corner_radius` request
    /// must clamp to the *angular* limit even though the radial thickness
    /// alone would permit a much larger radius — the two clamp legs bind
    /// independently.
    ///
    /// The exact angular limit (cycle 2, Required Fix 1) solves
    /// `asin(cr / (outer_r − cr)) = sweep / 2` for `cr`, giving
    /// `cr_limit_angular_outer = outer_r·sin(sweep/2) / (1 + sin(sweep/2))`
    /// — not the naive `outer_r · sweep / 2`, though the two agree to first
    /// order for small `sweep`. Value hand-computed independently.
    #[test]
    fn wedge_path_clamps_to_angular_half_length_on_narrow_sweep() {
        let sweep = 0.01; // ~0.57 degrees
        let cmds = wedge_path(0.0, 0.0, 0.0, 100.0, 0.0, sweep, 50.0);
        let arc_radii: Vec<f64> = cmds.iter().filter_map(|c| {
            if let PathCmd::ArcTo { rx, .. } = c { Some(*rx) } else { None }
        }).collect();
        // radial_thickness/2 = 50.0 (no bind); angular limit binds instead.
        let fillets: Vec<f64> = arc_radii.iter().copied().filter(|r| (*r - 100.0).abs() > 1e-6).collect();
        assert_eq!(fillets.len(), 2, "expected 2 outer fillets on a solid narrow wedge; got {arc_radii:?}");
        for r in fillets {
            assert!((r - 0.49751037515826596).abs() < 1e-9,
                "corner radius must clamp to the exact angular limit 0.49751037515826596 (not the naive 0.5), not the requested 50.0; got {r}");
        }
    }

    /// Cycle 3 (spec-reviewer Issue 1 + rust-quality-reviewer F5): a wide
    /// sweep (`> π`) must let the radial leg govern alone — the angular
    /// leg's `asin`-inverting closed form is only the correct inverse of
    /// `Δ = asin(cr/ρ) <= sweep/2` on `asin`'s own range, `Δ ∈ [0, π/2]`,
    /// i.e. for `sweep <= π`. Since `Δ` can never exceed `π/2` by
    /// construction, `Δ <= sweep/2` holds unconditionally for every `cr`
    /// once `sweep >= π`; applying the closed form past that point folds it
    /// (`sin` decreasing back toward `0` as `sweep/2` moves past `π/2`) and
    /// silently *shrinks* the limit as the wedge gets wider, the opposite of
    /// the truth. This mirrors the real end-to-end defect measured against
    /// this exact code: a 342°-sweep pie slice at `corner_radius=30` was
    /// clamped to `~8` px by the old formula.
    ///
    /// `outer_r=100, inner_r=0` (solid, so the radial leg's own exact limit
    /// is `h·(2·outer_r−h)/(2·outer_r) = 37.5` for `h = 50`), `sweep=6.0`
    /// rad (`≈343.8°`, `> π`), request `30.0` (`< 37.5`, so the radial leg
    /// alone should pass it through unclamped). The pre-fix angular formula
    /// would give `100·sin(3.0)/(1+sin(3.0)) ≈ 12.37`, well below `30.0`.
    #[test]
    fn wedge_path_wide_sweep_angular_leg_is_non_binding() {
        let cmds = wedge_path(0.0, 0.0, 0.0, 100.0, 0.0, 6.0, 30.0);
        let arc_radii: Vec<f64> = cmds.iter().filter_map(|c| {
            if let PathCmd::ArcTo { rx, .. } = c { Some(*rx) } else { None }
        }).collect();
        let fillets: Vec<f64> = arc_radii.iter().copied().filter(|r| (*r - 100.0).abs() > 1e-6).collect();
        assert_eq!(fillets.len(), 2, "expected 2 outer fillets on a wide-sweep solid wedge; got {arc_radii:?}");
        for r in fillets {
            assert!((r - 30.0).abs() < 1e-9,
                "corner radius must pass through unclamped at 30.0 (the angular leg is non-binding for sweep > π \
                 and the radial leg's own exact limit is 37.5) — a folded angular leg would incorrectly clamp to ~12.37; got {r}");
        }
    }

    /// The same wide-sweep non-binding behavior on the INNER angular leg —
    /// an annular wedge whose inner corners would, under the folded formula,
    /// clamp far below what the radial leg alone permits.
    ///
    /// `inner_r=40, outer_r=100` (radial leg's exact limits: outer `25.5`,
    /// inner `41.25`, for `h = 30`), `sweep=6.0` rad, request `20.0` (under
    /// both radial limits, so BOTH the outer and inner corners pass through
    /// unclamped — this test's request is deliberately chosen so neither
    /// radial leg binds, isolating the angular-leg fix on both corners at
    /// once). The pre-fix inner formula would give `40·sin(3.0)/(1−
    /// sin(3.0)) ≈ 6.57`; the pre-fix outer formula would give `≈12.37`
    /// (same as the solid case above) — both well below `20.0`.
    #[test]
    fn wedge_path_wide_sweep_inner_angular_leg_is_non_binding() {
        let cmds = wedge_path(0.0, 0.0, 40.0, 100.0, 0.0, 6.0, 20.0);
        let arc_radii: Vec<f64> = cmds.iter().filter_map(|c| {
            if let PathCmd::ArcTo { rx, .. } = c { Some(*rx) } else { None }
        }).collect();
        let unclamped = arc_radii.iter().filter(|r| (**r - 20.0).abs() < 1e-9).count();
        assert_eq!(unclamped, 4, "both outer and inner corners must pass through unclamped at 20.0 (radial legs \
             permit 25.5 outer / 41.25 inner, angular legs are non-binding for sweep > π); got {arc_radii:?}");
    }

    /// Boundary: at `sweep == π` exactly, the angular leg must already be
    /// non-binding (`Δ` maxes out at exactly `π/2` there too) — this is the
    /// single point cycle-2's fix folded into the inner leg alone (`1.0 -
    /// sin_half_sweep` guarded near zero); cycle 3 generalizes the guard to
    /// the whole `sweep >= π` domain on both legs, and this test pins the
    /// boundary itself stays covered.
    #[test]
    fn wedge_path_exactly_semicircle_sweep_angular_leg_non_binding() {
        let cmds = wedge_path(0.0, 0.0, 40.0, 100.0, 0.0, std::f64::consts::PI, 25.0);
        let arc_radii: Vec<f64> = cmds.iter().filter_map(|c| {
            if let PathCmd::ArcTo { rx, .. } = c { Some(*rx) } else { None }
        }).collect();
        // radial_thickness/2 = 30.0; request 25.0 < 30.0, so every corner
        // (outer and inner) must pass through unclamped at exactly π sweep.
        let unclamped = arc_radii.iter().filter(|r| (**r - 25.0).abs() < 1e-9).count();
        assert_eq!(unclamped, 4, "all four corners must clamp at 25.0 (unclamped) at exactly sweep=π; got {arc_radii:?}");
    }

    /// A full-circle sweep has no discrete start/end corners to round — the
    /// fillet path must never engage, regardless of `corner_radius`, and the
    /// output must be byte-identical to the `corner_radius = 0.0` case.
    #[test]
    fn wedge_path_full_circle_ignores_corner_radius() {
        let sweep_full = std::f64::consts::TAU - 1e-10;
        let unrounded = wedge_path(100.0, 100.0, 0.0, 50.0, 0.0, sweep_full, 0.0);
        let requested_round = wedge_path(100.0, 100.0, 0.0, 50.0, 0.0, sweep_full, 25.0);
        assert_eq!(unrounded.len(), requested_round.len());
        for (a, b) in unrounded.iter().zip(requested_round.iter()) {
            assert_eq!(format!("{a:?}"), format!("{b:?}"),
                "a full-circle wedge must ignore corner_radius entirely");
        }
        let arc_count = requested_round.iter().filter(|c| matches!(c, PathCmd::ArcTo { .. })).count();
        assert_eq!(arc_count, 2, "full-circle geometry (2 half-arcs) must be unaffected by corner_radius");
    }

    /// A zero-thickness ring (`inner_r == outer_r`) has no radial edge to
    /// trim (`radial_thickness / 2 == 0`) — rounding must refuse (fall back
    /// to the unrounded, already-degenerate geometry) rather than emitting
    /// negative-radius or self-intersecting fillets.
    #[test]
    fn wedge_path_zero_thickness_ring_does_not_round() {
        let r = 50.0;
        let unrounded = wedge_path(100.0, 100.0, r, r, 0.0, std::f64::consts::FRAC_PI_2, 0.0);
        let requested_round = wedge_path(100.0, 100.0, r, r, 0.0, std::f64::consts::FRAC_PI_2, 25.0);
        assert_eq!(unrounded.len(), requested_round.len());
        for (a, b) in unrounded.iter().zip(requested_round.iter()) {
            assert_eq!(format!("{a:?}"), format!("{b:?}"),
                "a zero-thickness ring has no radial edge to round and must fall back unchanged");
        }
    }

    /// End-to-end wiring: `mark_arc(corner_radius=)` reaches `build()` via
    /// `ctx.mark_style.paint.corner_radius` (populated by the generic
    /// override applier, `draw.rs:367`) and actually rounds every emitted
    /// slice — not just the standalone `wedge_path` unit. A 3-slice pie
    /// (weights 10/30/60, all solid `inner_r == 0` sectors) must show 3 arcs
    /// per path (1 main + 2 fillets) once `corner_radius` is set, vs. 1 arc
    /// per path at the theme baseline (unset ⇒ 0.0).
    #[test]
    fn build_arc_corner_radius_override_rounds_every_slice() {
        let spec = polar_spec(false);
        let batch = make_batch(false);
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let scales = make_scales(false);

        let baseline_style = resolve_mark_style(None, &theme, &Mark::Arc).unwrap();
        assert_eq!(baseline_style.paint.corner_radius, 0.0, "fixture sanity: Arc's baseline corner radius is 0.0");
        let baseline_ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &baseline_style };
        let baseline = build(&baseline_ctx).unwrap();

        let overrides = crate::spec::mark_style::MarkKwargsSpec { corner_radius: Some(8.0), ..Default::default() };
        let rounded_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Arc).unwrap();
        assert_eq!(rounded_style.paint.corner_radius, 8.0);
        let rounded_ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &rounded_style };
        let rounded = build(&rounded_ctx).unwrap();

        assert_eq!(baseline.nodes.len(), 3);
        assert_eq!(rounded.nodes.len(), 3);
        for (base_n, round_n) in baseline.nodes.iter().zip(rounded.nodes.iter()) {
            let base_arcs = arc_count(base_n);
            let round_arcs = arc_count(round_n);
            assert_eq!(base_arcs, 1, "baseline (corner_radius unset) slice must be a single sharp arc");
            assert_eq!(round_arcs, 3, "rounded slice must be 1 main arc + 2 corner fillets; got {round_arcs}");
        }
    }
}
