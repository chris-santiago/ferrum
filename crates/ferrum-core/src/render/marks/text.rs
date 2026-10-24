//! mark_text: renders a text label at (scale_x(x), scale_y(y)).
//!   - Supports ordinal x / y axes (categorical positioning, mirroring mark_rect).
//!   - A `text` encoding channel supplies explicit label content (Utf8 column).
//!
//! When the text channel is absent and y is numeric, the label falls back to
//! `format_numeric(y)`.
//!
//! # The `text` channel read propagates its typed error (#166)
//!
//! A bound `text` channel naming a column this batch does not carry — most
//! commonly a misspelled field — used to resolve to `texts = None` (both
//! `col_as_str` and the `col_as_f64` fallback return `Err`, and `.ok()`
//! discarded whichever one ran last). Every row then fell through to the
//! y-numeric-format branch, and on an ordinal y axis that branch has no
//! numeric column to format, so every row is skipped: a chart with an intact
//! panel and zero label nodes, no diagnostic. `build` now propagates that
//! `RenderError` (`UnknownColumn` for a missing column, `UnsupportedDtype`
//! for a present column neither reader can format) with `?`, matching
//! `rule.rs`'s reader-propagation idiom (this module's `build` is now
//! fallible for exactly this reason — every other reader here is unaffected).
//! A batch that does carry the column, or a chart with no `text` channel at
//! all, is byte-identical.
//!
//! # Polar labels
//!
//! Under `CoordPolar`, a text mark binding theta and no radius sits on its own
//! theta value, as a point there would, on a ring: its `radius=` style value,
//! else mid-ring ([`polar_label_placement`]). With `Theta(stack=...)` it stacks
//! theta the way the arc does instead and sits at its wedge's angular
//! midpoint, from the arc's own pie sweep ([`arc::pie_wedges`]) or nominal
//! bands ([`arc::band_wedges`]), so a pie's labels land inside their slices
//! by construction; its radius binding, else the ring, places it there. An
//! unstacked text with a radius binding is placed as a point is, by the
//! panel's polar projection of its Cartesian position.

use crate::layout::TextAnchor;
use crate::render::arrow_cast::col_as_ordinal_category_str;
use crate::render::draw::{
    col_as_f64, col_as_positional_category_str, col_as_str, resolve_fill_color, x_field, y_field,
    ClearablePaint, DrawCtx,
};
use crate::render::format::{format_numeric, format_value_with_spec, Temporality};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::arc;
use crate::render::marks::channels::color_column_loader;
use crate::render::scale_resolve::ScaleKind;
use crate::render::{scene_build, RenderError};
use crate::spec::coord::PolarThetaChannel;

/// How a text mark under `CoordPolar` places its labels when the panel's
/// polar projection of its Cartesian position does not: the fields it reads
/// and how its theta becomes an angle.
pub(crate) struct PolarLabels<'a> {
    theta_channel: PolarThetaChannel,
    theta_field: &'a str,
    angle: LabelAngle,
    pub(crate) radius_field: Option<&'a str>,
    radius_scale: &'a ScaleKind,
}

/// How a polar label's theta becomes its angle.
enum LabelAngle {
    /// The label's own theta value on the theta axis, the angle a point with
    /// that value draws at.
    Value,
    /// The mid-angle of the wedge `Theta(stack=...)` stacks the label into:
    /// the arc's equal band for a categorical theta, else its pie sweep.
    Wedge { categorical: bool },
}

/// Whether `encoding` stacks the coord's theta channel (`Theta(stack=...)`).
pub(crate) fn theta_stacked(
    encoding: &crate::spec::encoding::Encoding,
    theta_channel: PolarThetaChannel,
) -> bool {
    let theta = match theta_channel {
        PolarThetaChannel::X => encoding.x.as_ref(),
        PolarThetaChannel::Y => encoding.y.as_ref(),
    };
    theta.and_then(|e| e.stack_offset()).is_some()
}

/// The text mark's own polar placement, or `None` when the panel's polar
/// projection of its Cartesian position places it, as it places a point:
/// outside `CoordPolar`, with no theta binding, or with an unstacked theta
/// and a radius binding.
///
/// The rule the single-axis exemption relies on: a polar text binding theta
/// and no radius always places itself, so it never needs the radius axis
/// that exemption leaves unresolved.
pub(crate) fn polar_label_placement<'a>(ctx: &'a DrawCtx<'_>) -> Option<PolarLabels<'a>> {
    let Some(crate::spec::coord::CoordKind::Polar { theta, .. }) = &ctx.spec.coord else {
        return None;
    };
    let channels = crate::render::marks::channels::polar_channel_resolver(*theta, &ctx.spec.encoding, ctx.scales);
    let theta_field = channels.theta_field?;
    let angle = if theta_stacked(&ctx.spec.encoding, *theta) {
        LabelAngle::Wedge { categorical: arc::theta_is_categorical(&ctx.spec.encoding, ctx.scales, *theta) }
    } else if channels.radius_field.is_none() {
        LabelAngle::Value
    } else {
        return None;
    };
    Some(PolarLabels {
        theta_channel: *theta,
        theta_field,
        angle,
        radius_field: channels.radius_field,
        radius_scale: channels.radius_scale,
    })
}

/// The theta field a polar text draws at its own values when that field is a
/// quantity: a `CoordPolar` text binding theta without `Theta(stack=...)`.
fn unstacked_quantity_theta<'a>(
    encoding: &'a crate::spec::encoding::Encoding,
    coord: Option<&crate::spec::coord::CoordKind>,
    scales: &crate::render::scale_resolve::ResolvedScales,
) -> Option<&'a str> {
    let Some(crate::spec::coord::CoordKind::Polar { theta, .. }) = coord else {
        return None;
    };
    let field = match theta {
        PolarThetaChannel::X => encoding.x.as_ref(),
        PolarThetaChannel::Y => encoding.y.as_ref(),
    }?
    .field
    .as_str();
    (!theta_stacked(encoding, *theta) && !arc::theta_is_categorical(encoding, scales, *theta)).then_some(field)
}

/// One drawn layer of a panel, as [`labels_off_a_pie`] reads it: its mark,
/// its own coord, the encoding it draws and the scales it draws through.
pub(crate) struct PanelLayer<'a> {
    pub(crate) mark: crate::spec::mark::Mark,
    pub(crate) coord: Option<&'a crate::spec::coord::CoordKind>,
    pub(crate) encoding: &'a crate::spec::encoding::Encoding,
    pub(crate) scales: &'a crate::render::scale_resolve::ResolvedScales,
}

/// Every pairing, among one panel's `layers`, of a polar text drawing its
/// labels at their own theta values and an arc stacking that theta field
/// into a pie ([`arc::pie_theta_field`]): the labels then sit off the
/// wedges they name, which `Theta(stack=True)` on the text fixes. Each
/// entry is `(text layer, arc layer)`, as indices into `layers`.
pub(crate) fn labels_off_a_pie(layers: &[PanelLayer<'_>]) -> Vec<(usize, usize)> {
    use crate::spec::mark::Mark;
    let pies: Vec<(usize, &str)> = layers
        .iter()
        .enumerate()
        .filter(|(_, l)| l.mark == Mark::Arc)
        .filter_map(|(i, l)| Some((i, arc::pie_theta_field(l.encoding, l.coord)?)))
        .collect();
    let mut pairs = Vec::new();
    for (ti, text) in layers.iter().enumerate().filter(|(_, l)| l.mark == Mark::Text) {
        let Some(field) = unstacked_quantity_theta(text.encoding, text.coord, text.scales) else { continue };
        pairs.extend(pies.iter().filter(|(_, pie)| *pie == field).map(|&(ai, _)| (ti, ai)));
    }
    pairs
}

/// Each row's label position, `None` for a row that draws no label, and the
/// label a row shows when no `text` channel is bound: its value on the
/// channel that places it (y on Cartesian axes, theta on a polar wedge).
struct LabelAnchors {
    anchors: Vec<Option<(f64, f64)>>,
    fallback_labels: Vec<Option<String>>,
}

/// [`format_numeric`] of each finite value, `None` elsewhere.
fn numeric_labels(values: &[Option<f64>]) -> Vec<Option<String>> {
    values.iter().map(|v| v.filter(|v| v.is_finite()).map(format_numeric)).collect()
}

/// Each row's polar label angle (`None` for a row with none) and the label it
/// shows without a `text` channel, its theta value.
struct RowAngles {
    angles: Vec<Option<f64>>,
    fallback_labels: Vec<Option<String>>,
}

/// Each row's angle and its theta-derived fallback label, for a label that
/// sits on its own theta value: the angle a point with that value draws at,
/// read through the theta axis's scale ([`scene_build::polar_projection_turn`]).
fn value_angles(ctx: &DrawCtx<'_>, placement: &PolarLabels<'_>) -> Option<RowAngles> {
    let channel = placement.theta_channel;
    let area = ctx.panel.plot_area;
    let scale = match channel {
        PolarThetaChannel::X => &ctx.scales.x,
        PolarThetaChannel::Y => &ctx.scales.y,
    };
    let turn = |pixel: f64| scene_build::polar_projection_turn(channel, pixel, &area) * std::f64::consts::TAU;
    if matches!(scale, ScaleKind::Ordinal(_)) {
        let categories = col_as_positional_category_str(ctx.batch, placement.theta_field).ok()?;
        let angles = categories.iter().map(|c| scale.to_pixel_str(c.as_deref()?).map(&turn)).collect();
        Some(RowAngles { angles, fallback_labels: categories })
    } else {
        let values = col_as_f64(ctx.batch, placement.theta_field).ok()?;
        let angles = values
            .iter()
            .map(|v| scale.to_pixel_f64(v.filter(|v| scale.admits_for_pixel(*v))?).map(&turn))
            .collect();
        Some(RowAngles { angles, fallback_labels: numeric_labels(&values) })
    }
}

/// Each row's wedge mid-angle and theta-derived fallback label, for a label
/// `Theta(stack=...)` stacks the way the arc does: the arc's own bands or pie
/// sweep, so the label's angle is its wedge's. A row whose wedge the arc
/// would not draw has none.
fn wedge_angles(
    ctx: &DrawCtx<'_>,
    placement: &PolarLabels<'_>,
    geom: &arc::PolarGeom,
    categorical: bool,
) -> Option<RowAngles> {
    let mid = |wedge: &Option<(f64, f64)>| wedge.map(|(start, end)| (start + end) / 2.0);
    if categorical {
        // The arc's own category reader, so both marks key the same bands.
        let categories = col_as_ordinal_category_str(ctx.batch, placement.theta_field).ok()?;
        let wedges = arc::band_wedges(&categories, geom.start_angle, geom.pad_angle);
        Some(RowAngles { angles: wedges.iter().map(mid).collect(), fallback_labels: categories })
    } else {
        let values = col_as_f64(ctx.batch, placement.theta_field).ok()?;
        let wedges = arc::pie_wedges(&values, geom.start_angle, geom.pad_angle)
            .unwrap_or_else(|| vec![None; values.len()]);
        Some(RowAngles { angles: wedges.iter().map(mid).collect(), fallback_labels: numeric_labels(&values) })
    }
}

/// Label positions for a text mark that places itself under `CoordPolar`
/// (see the module doc's "Polar labels"), in the panel's polar pixel space:
/// each row's angle at its radius binding, else the `radius=` style value,
/// else mid-ring. A row with no angle or no usable radius value draws no
/// label.
fn polar_anchors(ctx: &DrawCtx<'_>, placement: &PolarLabels<'_>) -> Option<LabelAnchors> {
    let geom = arc::polar_geom(ctx)?;
    let RowAngles { angles, fallback_labels } = match placement.angle {
        LabelAngle::Value => value_angles(ctx, placement)?,
        LabelAngle::Wedge { categorical } => wedge_angles(ctx, placement, &geom, categorical)?,
    };
    let radii = match placement.radius_field {
        Some(field) => Some(col_as_f64(ctx.batch, field).ok()?),
        None => None,
    };
    let ring = ctx.mark_style.text.radius.unwrap_or((geom.inner_radius + geom.outer_radius) / 2.0);
    let anchors = angles
        .iter()
        .enumerate()
        .map(|(i, angle)| {
            let angle = (*angle)?;
            let r = match &radii {
                Some(radii) => {
                    let v = radii[i].filter(|v| v.is_finite())?;
                    arc::radius_to_pixel(placement.radius_scale, geom.inner_radius, geom.outer_radius, v)
                }
                None => ring,
            };
            Some((geom.cx + r * angle.sin(), geom.cy - r * angle.cos()))
        })
        .collect();
    Some(LabelAnchors { anchors, fallback_labels })
}

/// Label positions on the panel's x/y scales (ordinal or continuous on each
/// axis), plus the per-row position offsets a dodge writes. `None` when
/// either axis is unbound or unreadable.
fn cartesian_anchors(ctx: &DrawCtx<'_>) -> Option<LabelAnchors> {
    let spec = ctx.spec;
    let (xf, yf) = (x_field(ctx, spec)?, y_field(ctx, spec)?);

    // GH #42: text is the value-label layer for dodged bars (e.g.
    // importance_chart's `show_values=True`). Every other positional mark
    // renderer adds the per-row `__pos_x_offset__`/`__pos_y_offset__` to its
    // resolved pixel position (see tick.rs); text must do the same so dodged
    // labels track their bar's sub-band. Zero-effect when absent — the
    // accessor returns all-zero offsets.
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let x_ordinal = matches!(ctx.scales.x, ScaleKind::Ordinal(_));
    let y_ordinal = matches!(ctx.scales.y, ScaleKind::Ordinal(_));

    // The ordinal reads use `col_as_positional_category_str`, the same reader
    // `point`/`bar`/`tick`/`rule` use for an ordinal positional channel (NF-A3,
    // spec §4.4 — swept 2026-09-02). `col_as_str` errors on every non-`Utf8`
    // dtype and the `.ok()` swallowed it, so an `Int*`/`Float*`/`Bool` category
    // column on an ordinal axis left BOTH halves `None` and `build` took the
    // `return empty()` below — every label silently dropped from an otherwise
    // intact panel. Identical on `Utf8` apart from a null row, which now lands
    // in the FA-9 null band like every other positional mark's does.
    let xs_f: Option<Vec<Option<f64>>> =
        if !x_ordinal { col_as_f64(ctx.batch, xf).ok() } else { None };
    let xs_s: Option<Vec<Option<String>>> =
        if x_ordinal { col_as_positional_category_str(ctx.batch, xf).ok() } else { None };
    let ys_f: Option<Vec<Option<f64>>> =
        if !y_ordinal { col_as_f64(ctx.batch, yf).ok() } else { None };
    let ys_s: Option<Vec<Option<String>>> =
        if y_ordinal { col_as_positional_category_str(ctx.batch, yf).ok() } else { None };

    let n_x = match (&xs_f, &xs_s) {
        (Some(v), _) => v.len(),
        (_, Some(v)) => v.len(),
        _ => return None,
    };
    let n_y = match (&ys_f, &ys_s) {
        (Some(v), _) => v.len(),
        (_, Some(v)) => v.len(),
        _ => return None,
    };
    if n_x != n_y { return None; }

    let anchors = (0..n_x)
        .map(|i| {
            let px = if let Some(xs) = &xs_f {
                ctx.scales.x.to_pixel_f64(xs[i].filter(|v| ctx.scales.x.admits_for_pixel(*v))?)?
            } else {
                ctx.scales.x.to_pixel_str(xs_s.as_ref()?[i].as_deref()?)?
            };
            let py = if let Some(ys) = &ys_f {
                ctx.scales.y.to_pixel_f64(ys[i].filter(|v| ctx.scales.y.admits_for_pixel(*v))?)?
            } else {
                ctx.scales.y.to_pixel_str(ys_s.as_ref()?[i].as_deref()?)?
            };
            Some((px + x_offsets[i], py + y_offsets[i]))
        })
        .collect();
    let fallback_labels = match &ys_f {
        Some(ys) => numeric_labels(ys),
        None => vec![None; n_x],
    };
    Some(LabelAnchors { anchors, fallback_labels })
}

pub fn build(ctx: &DrawCtx) -> Result<crate::render::draw::MarkBuildResult, RenderError> {
    use crate::render::draw::{to_scene_text_style, MarkBuildResult, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let empty = || Ok(MarkBuildResult::empty(MarkBatchKind::Text));

    let spec = ctx.spec;
    let placement = polar_label_placement(ctx);
    let positions = match &placement {
        Some(placement) => polar_anchors(ctx, placement),
        None => cartesian_anchors(ctx),
    };
    let Some(LabelAnchors { anchors: row_anchors, fallback_labels }) = positions else {
        return empty();
    };
    let n_rows = row_anchors.len();

    // Explicit text channel (same resolution as draw()).
    let text_enc = spec.encoding.text.as_ref();
    let text_field = text_enc.map(|e| e.field.as_str());
    let text_format = text_enc.and_then(|e| e.format.as_deref());
    let text_format_type = text_enc.and_then(|e| e.format_type.as_deref());
    // `#147` close (wave 6A T21): a time-typed format on a non-temporal text
    // field refuses with the axes' own error, instead of rendering every
    // label as a 1970 epoch date.
    // The field's temporality decides whether an undeclared `%`-bearing spec
    // is a date pattern, as the axis's scale does (`#147` T23), and is the
    // one answer the refusal below checks against.
    let text_temporality = text_enc
        .and_then(|enc| crate::render::prepare::field_temporality(enc, ctx.batch))
        .unwrap_or(Temporality::NonTemporal);
    if let Some(enc) = text_enc {
        crate::render::prepare::refuse_format_temporality_mismatch_field(
            enc,
            ctx.batch,
            text_format,
            text_format_type,
            "text",
            crate::render::FormatSurface::Channel,
        )?;
    }
    let texts: Option<Vec<Option<String>>> = match text_field {
        None => None,
        Some(f) => Some(col_as_str(ctx.batch, f).or_else(|_| {
            col_as_f64(ctx.batch, f).map(|nums| {
                nums.into_iter()
                    .map(|opt_v| {
                        opt_v.and_then(|v| {
                            if !v.is_finite() {
                                return None;
                            }
                            // ONE resolution (`#147`): the same
                            // field-temporality classification the colorbar
                            // uses for the identical pair, so a
                            // `text=fm.Text(field, format="%Y/%m/%d")` channel
                            // over a datetime field APPLIES its pattern.
                            Some(format_value_with_spec(
                                v,
                                text_format,
                                text_format_type,
                                text_temporality,
                            ))
                        })
                    })
                    .collect()
            })
        })?),
    };

    let anchor = match ctx.mark_style.text.align.as_deref() {
        Some("left") => TextAnchor::Start,
        Some("right") => TextAnchor::End,
        _ => TextAnchor::Middle,
    };
    let dx = ctx.mark_style.text.dx.unwrap_or(0.0);
    let dy = ctx.mark_style.text.dy.unwrap_or(0.0);
    let base_font_size = ctx.mark_style.text.font_size.unwrap_or(ctx.theme.typography.label_font_size);
    let base_angle = ctx.mark_style.text.angle.unwrap_or(0.0);
    let base_opacity = ctx.mark_style.paint.opacity;

    // Per-row encoding channels (same pattern as point.rs).
    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let size_values: Option<Vec<Option<f64>>> = spec.encoding.size
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let angle_values: Option<Vec<Option<f64>>> = spec.encoding.angle
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    // Per-row fill color. A text layer does not inherit color: it draws a
    // color only when it binds one itself, so labels stay legible on the
    // marks they annotate. Its completed encoding carries no other color
    // (`Layer::completed_encoding`, by `Mark::draws_inherited`); an inherited
    // color groups its dodge and stack through its `group` key instead. The
    // color channel maps through `ctx.scales.color` when bound,
    // else `mark_style.paint.fill` when the user set it explicitly, else the
    // theme's font color — `label.rs`'s constant-fill precedence, defaulting
    // to `font_color` (not `mark_color`) since text's baseline theme style
    // carries no mark-aware fill override (see `resolve_mark_style`'s
    // `Mark::Text` arm).
    //
    // The "fill set by user" gate reads `ctx.mark_style.paint.fill_source`
    // — the *layer-resolved* provenance — rather than `ctx.spec.mark_style` (the raw
    // `MarkKwargsSpec`). `ctx.spec` is `scene_build.rs`'s synthetic per-layer
    // `ChartSpec`, whose `mark_style` field is NOT overridden per layer (it
    // stays the chart-level kwargs via `..spec.clone()`), so a
    // `ctx.spec.mark_style` read would report the WRONG layer's `fill=` inside
    // a `LayerChart` (e.g. `mark_bar(fill=...) + mark_text()` would wrongly
    // read the bar's fill as "the text layer's fill was set"). `ctx.mark_style`
    // is always built via `resolve_mark_style(layer.mark_style.as_ref(), ...)`
    // for THIS layer, so its `fill_source` is correct in both flat and
    // layered charts.
    //
    // A user-set fill goes in through `fill_paint()` so it keeps its
    // provenance: `fill="none"` is `Cleared`, which `resolve_fill_color` keeps
    // cleared over a color-channel hit and which reaches the scene as
    // `TextStyle.color: None` — the same `#110` rule every filled mark follows.
    let base_text_paint = if ctx.mark_style.paint.fill_source.is_user_set() {
        ctx.mark_style.paint.fill_paint()
    } else {
        ClearablePaint::Paint(ctx.theme.colors.font_color)
    };
    let (color_values_str, color_values_f64) = color_column_loader(ctx);

    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep so metadata is
    // aligned to kept nodes only (#6 defect class fix).
    let mut acc = MarkNodes::with_capacity(n_rows);
    // #230-3: one exact `(px, py)` data anchor per emitted node — the
    // scale-mapped position BEFORE `dx`/`dy` is baked into the node's own
    // `x`/`y` below. A break's glyph remap needs this anchor, not the node's
    // own (already-offset) coordinate, so `dx`/`dy` translates rigidly with
    // the glyph instead of being read back as if it were the data position.
    let mut anchors: Vec<(f64, f64)> = Vec::with_capacity(n_rows);

    for (i, position) in row_anchors.into_iter().enumerate() {
        let Some((px, py)) = position else { continue };

        let raw_label: String = if let Some(t) = &texts {
            match &t[i] {
                Some(s) => s.clone(),
                None => continue,
            }
        } else {
            match &fallback_labels[i] {
                Some(s) => s.clone(),
                None => continue,
            }
        };

        // S7: truncate label to `limit` characters (including the ellipsis).
        let label = if let Some(limit) = ctx.mark_style.text.limit {
            if limit > 0 && raw_label.chars().count() > limit {
                let truncated: String = raw_label.chars().take(limit.saturating_sub(1)).collect();
                format!("{truncated}\u{2026}") // …
            } else {
                raw_label
            }
        } else {
            raw_label
        };

        // Resolve per-row opacity (fill-opacity on text), through the
        // opacity scale when the channel resolved one.
        let row_opacity = crate::render::marks::opacity::resolve_scaled_or_raw(
            &opacity_values,
            &ctx.scales.opacity,
            i,
            base_opacity,
        );

        // Resolve per-row font-size from size encoding.
        let row_font_size = size_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| *v > 0.0 && v.is_finite())
            .unwrap_or(base_font_size);

        // Resolve per-row angle (rotation) from angle encoding.
        let row_angle = angle_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| v.is_finite())
            .unwrap_or(base_angle);

        // Resolve per-row fill paint: color channel wins when bound, else the
        // resolved constant (`base_text_paint`, see above) — unless that
        // constant is a clear, which outranks the channel.
        let row_paint = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_values_str.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_values_f64.as_ref().and_then(|v| v.get(i).copied().flatten()),
            base_text_paint,
        );

        acc.push(SceneNode::Text {
            x: px + dx,
            y: py + dy,
            content: label,
            slot: None,
            style: to_scene_text_style(
                row_paint,
                row_font_size,
                anchor,
                row_angle,
                &ctx.theme.typography.font_family,
                ctx.mark_style.text.font_weight.as_deref(),
                ctx.mark_style.text.baseline.as_deref(),
                row_opacity,
            ),
        }, i);
        anchors.push((px, py));
    }

    let (nodes, data_indices) = acc.finalize();

    let result = MarkBuildResult::populated(MarkBatchKind::Text, nodes, data_indices, &meta)
        .with_glyph_anchors(anchors);
    Ok(if placement.is_some() { result.placed_in_polar() } else { result })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn text_emits_one_text_element_per_row() {
        let spec = ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Text { .. })).count(), 2);
    }

    #[test]
    fn text_channel_renders_explicit_labels_on_ordinal_axes() {
        // Phase 10c-pre: confusion-matrix-style labels. Ordinal x/y with an
        // explicit `text` channel reading a Utf8 column.
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;
        let spec = ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "px".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "ax".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                text: Some(EncodingSpec { field: "label".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("px", DataType::Utf8, false),
            Field::new("ax", DataType::Utf8, false),
            Field::new("label", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(StringArray::from(vec!["x", "y"])),
            Arc::new(StringArray::from(vec!["42", "hello"])),
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx).unwrap();
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Text { .. })).count(), 2);
        // Check that the explicit label content is present in the Text nodes.
        let contents: Vec<&str> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Text { content, .. } = n { Some(content.as_str()) } else { None }
        }).collect();
        assert!(contents.contains(&"42"), "expected literal '42' label, got: {contents:?}");
        assert!(contents.contains(&"hello"), "expected literal 'hello' label, got: {contents:?}");
    }

    /// `#147` close (wave 6A T23, user ruling): the text channel reads an
    /// untyped spec by its field's temporality, as the axis reads it by the
    /// scale's. Over a datetime column `"%Y"` renders years (it used to
    /// refuse as a d3 spec) and `".2f"` refuses (it used to format the
    /// epoch-ms count, `978307200000.00`).
    #[test]
    fn text_reads_an_untyped_spec_by_the_field_temporality() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::{StringArray, TimestampMillisecondArray};
        let spec_with = |format: &str| ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "px".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "ax".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                text: Some(EncodingSpec { field: "d".into(), format: Some(format.into()), ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("px", DataType::Utf8, false),
            Field::new("ax", DataType::Utf8, false),
            Field::new("d", DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None), false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(StringArray::from(vec!["x", "y"])),
            Arc::new(TimestampMillisecondArray::from(vec![978_307_200_000_i64, 1_009_843_200_000])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let build = |spec: &ChartSpec| {
            let (scales, _) = resolve_scales(spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
            let ctx = DrawCtx {
                spec, panel: &panel, theme: &theme,
                scales: &scales, batch: &batch, mark_style: &mark_style,
            };
            super::build(&ctx).map(|r| {
                r.nodes.iter().filter_map(|n| match n {
                    ferrum_scene::SceneNode::Text { content, .. } => Some(content.clone()),
                    _ => None,
                }).collect::<Vec<_>>()
            })
        };
        assert_eq!(build(&spec_with("%Y")).unwrap(), ["2001", "2002"]);
        assert_eq!(
            build(&spec_with(".2f")).unwrap_err(),
            crate::render::RenderError::NumberFormatOnTemporalScale {
                channel: "text",
                surface: crate::render::FormatSurface::Channel,
                format: ".2f".into(),
                coord_flipped: false,
            }
        );
    }

    // ── Task 2: mark_text must still produce MarkBatchKind::Text (not Label) ──

    /// Plain mark_text build produces MarkBuildResult with kind == MarkBatchKind::Text.
    /// This guards against accidentally rerouting mark_text to Label.
    #[test]
    fn text_build_still_produces_text_kind() {
        let spec = ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 80.0])),
            Arc::new(Float64Array::from(vec![20.0, 70.0])),
        ])
        .unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = super::build(&ctx).unwrap();
        assert_eq!(
            result.kind,
            ferrum_scene::MarkBatchKind::Text,
            "mark_text must still produce MarkBatchKind::Text, got {:?}",
            result.kind
        );
    }

    // ── GH #42: dodge position-offset consumption ────────────────────────────
    //
    // Text is the value-label layer for dodged bars (e.g. importance_chart's
    // `show_values=True`). Every other positional mark renderer (tick, rect,
    // point, ...) reads `__pos_x_offset__`/`__pos_y_offset__` via
    // `read_position_offsets` and adds the per-row offset to its resolved
    // pixel position; text alone ignored them, so dodged labels sat at the
    // undodged band center. These offset columns are absent from user data —
    // no field of that name is ever encoded — so this is additive: a batch
    // without them renders byte-identically (`text_no_skip_tooltips_unchanged`
    // and friends above already pin that path).

    /// With `__pos_x_offset__` present, each glyph's x shifts by its row's
    /// offset relative to the undodged position.
    #[test]
    fn text_applies_pos_x_offset_when_present() {
        use arrow::array::Float64Array as F64Arr;

        let spec = ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("__pos_x_offset__", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(F64Arr::from(vec![7.5, -3.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let result = super::build(&ctx).unwrap();
        let xs: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Text { x, .. } = n { Some(*x) } else { None }
        }).collect();
        assert_eq!(xs.len(), 2);

        // Same spec/batch minus the offset column → undodged baseline x.
        let baseline_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let baseline_batch = arrow::record_batch::RecordBatch::try_new(baseline_schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
        ]).unwrap();
        let (baseline_scales, _) = resolve_scales(&spec, &baseline_batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let baseline_ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &baseline_scales, batch: &baseline_batch, mark_style: &mark_style };
        let baseline_result = super::build(&baseline_ctx).unwrap();
        let baseline_xs: Vec<f64> = baseline_result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Text { x, .. } = n { Some(*x) } else { None }
        }).collect();

        assert!((xs[0] - (baseline_xs[0] + 7.5)).abs() < 1e-9,
            "row 0 x must shift by its __pos_x_offset__ (7.5); got {} vs baseline {}", xs[0], baseline_xs[0]);
        assert!((xs[1] - (baseline_xs[1] + -3.0)).abs() < 1e-9,
            "row 1 x must shift by its __pos_x_offset__ (-3.0); got {} vs baseline {}", xs[1], baseline_xs[1]);
    }

    /// Without offset columns, text output is unaffected — the accessor
    /// returns all-zero offsets and glyph positions are unchanged.
    #[test]
    fn text_no_offset_columns_zero_effect() {
        let spec = ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        let xs: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Text { x, .. } = n { Some(*x) } else { None }
        }).collect();
        let expected = vec![
            scales.x.to_pixel_f64(10.0).unwrap(),
            scales.x.to_pixel_f64(50.0).unwrap(),
        ];
        assert_eq!(xs, expected, "no offset columns: text x must equal raw scaled position");
    }

    // ── Metadata-alignment regression tests (#6 defect class) ────────────────
    //
    // Text emits 1 node per kept row. A null text value causes `continue`, so
    // node indices diverge from row indices as soon as any row is skipped.
    //
    // Fail-before: `build_metadata(ctx)` produced full per-row vectors before the
    // loop; node j got row j's metadata regardless of skips. These tests would
    // have failed: node 1's tooltip would be "tip_b" (skipped row) not "tip_c".
    //
    // Pass-after: `MarkNodes` + `build_metadata_for_indices` aligns metadata to
    // kept nodes only.

    fn make_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        }
    }

    /// Regression: `build_text` with a null text-channel value skips that row.
    /// The tooltip on each surviving node must point to its true source row.
    ///
    /// Batch: 3 rows, text=[Some("a"), None, Some("c")],
    /// tooltip=["tip_a","tip_b","tip_c"]. Row 1 (null text) is skipped → 2
    /// nodes. Node 1 must have "tip_c" (row 2), not "tip_b" (row 1, old bug).
    #[test]
    fn text_skipped_null_text_tooltip_aligned() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                text: Some(EncodingSpec { field: "lbl".into(), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",   DataType::Float64, false),
            Field::new("y",   DataType::Float64, false),
            Field::new("lbl", DataType::Utf8,    true),   // nullable — row 1 null → skip
            Field::new("tip", DataType::Utf8,    false),
        ]));
        // Row 1 has null label → skipped (the `None => continue` on the text value).
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        // 2 nodes survive (row 1 with null label is skipped).
        assert_eq!(result.nodes.len(), 2,
            "expected 2 text nodes after null-label skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be 'tip_a' (row 0); got '{t0}'");

        // Node 1 → row 2 → "tip_c". Old code: "tip_b" (row 1, the alignment bug).
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be 'tip_c' (row 2), not 'tip_b' (row 1); got '{t1}'. \
             This fails on pre-migration code using build_metadata(ctx).");
    }

    /// Href-channel alignment: a null text value skips row 1; href on node 1
    /// must point to row 2's url ("url_c"), not row 1's ("url_b").
    #[test]
    fn text_skipped_null_text_href_aligned() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                text: Some(EncodingSpec { field: "lbl".into(), ..Default::default() }),
                href: Some(EncodingSpec { field: "url".into(), ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",   DataType::Float64, false),
            Field::new("y",   DataType::Float64, false),
            Field::new("lbl", DataType::Utf8,    true),
            Field::new("url", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            Arc::new(StringArray::from(vec!["url_a", "url_b", "url_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        assert_eq!(result.nodes.len(), 2, "expected 2 text nodes");
        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), 2, "href count must equal node count");
        assert_eq!(hrefs[0].as_deref(), Some("url_a"), "node 0 href must be 'url_a'");
        assert_eq!(hrefs[1].as_deref(), Some("url_c"),
            "node 1 href must be 'url_c' (row 2), not 'url_b' (row 1); \
             old build_metadata would give 'url_b'");
    }

    /// No-skip backward-compat: when no rows are skipped all tooltips appear in
    /// original row order — same result as the old `build_metadata(ctx)` path.
    #[test]
    fn text_no_skip_tooltips_unchanged() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                text: Some(EncodingSpec { field: "lbl".into(), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",   DataType::Float64, false),
            Field::new("y",   DataType::Float64, false),
            Field::new("lbl", DataType::Utf8,    false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        // All rows have non-null text → no skipping.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        assert_eq!(result.nodes.len(), 3, "all 3 rows must produce nodes");
        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 3, "tooltip count must equal node count");
        let values: Vec<&str> = tooltips.iter().map(|t| t.fields[0].value.as_str()).collect();
        assert_eq!(values, vec!["tip_a", "tip_b", "tip_c"],
            "no-skip: tooltips must be in original row order");
    }

    // ── Ported from bug_hunt_marks_rendering_r2.rs (R1) ─────────────────────
    // `mark_style.text.limit` truncation boundary conditions, exercised via
    // the real `build` pipeline with an explicit `text` channel.

    fn limit_test_spec() -> ChartSpec {
        use crate::spec::encoding::DataType as SDT;
        ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                text: Some(EncodingSpec { field: "label".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn build_with_limit(limit: Option<usize>) -> String {
        use crate::spec::mark_style::MarkKwargsSpec;
        use arrow::array::StringArray;
        let spec = limit_test_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("label", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(StringArray::from(vec!["Hello"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec { limit, ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        result.nodes.iter().find_map(|n| {
            if let ferrum_scene::SceneNode::Text { content, .. } = n { Some(content.clone()) } else { None }
        }).expect("expected exactly one Text node")
    }

    #[test]
    fn text_limit_equal_to_label_length_does_not_truncate() {
        // "Hello".chars().count() == 5 == limit -> the `>` comparison is false.
        assert_eq!(build_with_limit(Some(5)), "Hello");
    }

    #[test]
    fn text_limit_one_truncates_to_bare_ellipsis() {
        // limit=1: saturating_sub(1) keeps 0 chars, so only the ellipsis remains.
        assert_eq!(build_with_limit(Some(1)), "\u{2026}");
    }

    #[test]
    fn text_limit_zero_is_treated_as_unset() {
        // `limit > 0` is false for limit=0, so no truncation occurs at all.
        assert_eq!(build_with_limit(Some(0)), "Hello");
    }

    // ── Task 4: fill=/encode(color=) precedence (spec §4.4) ─────────────────
    //
    // Precedence: color channel via `ctx.scales.color` when bound, else
    // `mark_style.paint.fill` when the user set it explicitly, else the theme's
    // font color (mirrors `label.rs`'s constant-fill resolution, but text's
    // fallback is `font_color` rather than `mark_color` — see the comment on
    // `base_text_color` in `build`).

    use crate::render::draw::to_scene_color;
    use crate::spec::mark_style::MarkKwargsSpec;

    /// Two-row spec/batch with quantitative x/y and an optional Utf8 `c`
    /// (color) column. `fill_override` maps to `mark_style.fill=`;
    /// `with_color_channel` binds `encoding.color` to the `c` field.
    fn color_precedence_ctx_colors(
        fill_override: Option<&str>,
        with_color_channel: bool,
    ) -> Vec<ferrum_scene::Color> {
        use arrow::array::StringArray;

        let overrides = fill_override.map(|hex| MarkKwargsSpec { fill: Some(hex.into()), ..Default::default() });
        let spec = ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: if with_color_channel {
                    Some(EncodingSpec { field: "c".into(), type_: None, ..Default::default() })
                } else {
                    None
                },
                ..Default::default()
            }, mark_style: overrides.clone(),
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("c", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(overrides.as_ref(), &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Text { style, .. } = n { style.color } else { None }
        }).collect()
    }

    /// NF-A2: no `fill=` override and no `color` channel — every row's text
    /// color must stay exactly the theme's font color (byte-identical default).
    #[test]
    fn text_default_color_is_theme_font_color() {
        let colors = color_precedence_ctx_colors(None, false);
        let expected = to_scene_color(ThemeInputs::default().colors.font_color);
        assert_eq!(colors.len(), 2);
        assert!(colors.iter().all(|c| *c == expected),
            "default mark_text color must equal theme.colors.font_color; got {colors:?}, expected {expected:?}");
    }

    /// `fill=` honored: with no color channel bound, an explicit `fill=`
    /// override wins over the theme font color for every row.
    #[test]
    fn text_fill_kwarg_sets_constant_color() {
        let colors = color_precedence_ctx_colors(Some("#ff0000"), false);
        let red = ferrum_scene::Color { r: 0xff, g: 0x00, b: 0x00, a: 255 };
        assert_eq!(colors.len(), 2);
        assert!(colors.iter().all(|c| *c == red),
            "fill='#ff0000' must set every row's text color to red; got {colors:?}");
        let font_color = to_scene_color(ThemeInputs::default().colors.font_color);
        assert_ne!(red, font_color, "test fixture sanity: red must differ from the theme font color");
    }

    /// `encode(color=)` honored: with a bound color channel and no `fill=`
    /// override, distinct category values resolve to distinct colors (not the
    /// theme font color).
    #[test]
    fn text_color_channel_sets_per_row_color() {
        let colors = color_precedence_ctx_colors(None, true);
        assert_eq!(colors.len(), 2);
        assert_ne!(colors[0], colors[1],
            "rows with distinct color-channel categories must resolve to distinct colors; got {colors:?}");
        let font_color = to_scene_color(ThemeInputs::default().colors.font_color);
        assert!(colors.iter().all(|c| *c != font_color),
            "color-channel rows must not fall back to the theme font color; got {colors:?}");
    }

    /// Precedence: a bound color channel wins over an explicit `fill=`
    /// override (channel is checked first).
    #[test]
    fn text_color_channel_overrides_fill_kwarg() {
        let colors = color_precedence_ctx_colors(Some("#ff0000"), true);
        let red = ferrum_scene::Color { r: 0xff, g: 0x00, b: 0x00, a: 255 };
        assert_eq!(colors.len(), 2);
        assert!(colors.iter().all(|c| *c != red),
            "a bound color channel must override fill='#ff0000'; got {colors:?}");
        assert_ne!(colors[0], colors[1],
            "channel-resolved colors must still vary per category; got {colors:?}");
    }

    // ── NF-A3: ordinal positional reads key off the scale, not the dtype ─────

    /// An `Int64` nominal `x` on an ordinal scale must place its labels, not
    /// drop them.
    ///
    /// RED (verified in place): with `col_as_str` here, the read returns `Err`
    /// for `Int64`, `.ok()` swallows it, BOTH `xs_f` (skipped — the scale is
    /// ordinal) and `xs_s` end up `None`, and `build` takes the `n_x` match's
    /// `return empty()`. Every data label vanishes from an otherwise intact
    /// panel — the silent-empty class `rule.rs` eliminated, still live in a
    /// reader choice. The Utf8 twin is asserted alongside so the test cannot
    /// pass by producing zero labels for both.
    #[test]
    fn text_places_labels_on_an_int64_ordinal_x() {
        use arrow::array::{Int64Array, StringArray};
        use crate::spec::encoding::DataType as SpecType;

        let build_for = |x_field: Field, x_col: arrow::array::ArrayRef| -> Vec<String> {
            let spec = ChartSpec {
                mark: Mark::Text,
                encoding: Encoding {
                    x: Some(EncodingSpec { field: "x".into(), type_: Some(SpecType::Nominal), ..Default::default() }),
                    y: Some(EncodingSpec { field: "y".into(), type_: Some(SpecType::Quantitative), ..Default::default() }),
                    text: Some(EncodingSpec { field: "lbl".into(), ..Default::default() }),
                    ..Default::default()
                },
                ..Default::default()
            };
            let schema = Arc::new(Schema::new(vec![
                x_field,
                Field::new("y", DataType::Float64, false),
                Field::new("lbl", DataType::Utf8, false),
            ]));
            let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
                x_col,
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(StringArray::from(vec!["one", "two", "three"])),
            ]).unwrap();
            let theme = ThemeInputs::default();
            let panel = PanelLayout {
                plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
                facet_key: None, row: 0, col: 0,
                strip_title: None, row_strip_title: None, row_facet_key: None,
            };
            let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
            assert!(
                matches!(scales.x, crate::render::scale_resolve::ScaleKind::Ordinal(_)),
                "fixture must resolve an ordinal x scale, else it exercises the numeric branch"
            );
            let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
            let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
            super::build(&ctx).unwrap().nodes.iter().filter_map(|n| match n {
                ferrum_scene::SceneNode::Text { content, .. } => Some(content.clone()),
                _ => None,
            }).collect()
        };

        let utf8 = build_for(
            Field::new("x", DataType::Utf8, false),
            Arc::new(StringArray::from(vec!["10", "20", "30"])),
        );
        assert_eq!(utf8, vec!["one", "two", "three"],
            "Utf8 reference: all three labels must render");

        let int64 = build_for(
            Field::new("x", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![10_i64, 20, 30])),
        );
        assert_eq!(int64, utf8,
            "an Int64 nominal x must place the same three labels as its Utf8 twin; \
             got {int64:?}. An empty vec here is the col_as_str drop.");
    }

    // ── #166: `text` channel reader propagation ──────────────────────────

    /// `encode(text="nope")` naming a column absent from the batch used to
    /// resolve `texts = None` silently (both `col_as_str` and its
    /// `col_as_f64` fallback error, and `.ok()` discarded whichever ran
    /// last). On this fixture's ordinal y axis the per-row fallback branch
    /// (`match &ys_f { ... None => continue }`) has no numeric column to
    /// format, so every row was skipped — an intact panel with zero label
    /// nodes and no diagnostic. `build` now propagates the reader's typed
    /// error with `?` instead (rule.rs's reader-propagation idiom).
    #[test]
    fn text_encode_text_unknown_column_refuses_instead_of_rendering_blank() {
        use crate::spec::encoding::DataType as SpecType;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SpecType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SpecType::Ordinal), ..Default::default() }),
                text: Some(EncodingSpec { field: "nope".into(), ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Utf8, false),
            Field::new("y", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(StringArray::from(vec!["p", "q"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let err = match super::build(&ctx) {
            Err(e) => e,
            Ok(r) => panic!(
                "a text= column absent from the batch must refuse, not render {} label nodes silently",
                r.nodes.len()
            ),
        };
        assert!(
            matches!(&err, crate::render::RenderError::UnknownColumn { name } if name == "nope"),
            "expected UnknownColumn naming 'nope'; got {err:?}"
        );
    }

    /// The text-reader twin of `rule.rs`'s
    /// `rule_treats_a_color_column_absent_from_its_layer_batch_as_no_color`
    /// (spec §7's structural `Ok(None)` carve-out, `channels.rs:157-192`): a
    /// `color` field this layer's own batch does not carry (the ordinary
    /// layered-chart shape `Encoding::inherit_from` produces, e.g.
    /// `mark_qq(line=True)`'s reference layer) is not an error, even though
    /// `build` is now fallible for the `text` channel above.
    /// `color_column_loader` stays infallible and untouched by this task, so
    /// this pins that making `build`'s return type fallible did not also
    /// start refusing the case the carve-out exists for.
    #[test]
    fn text_treats_a_color_column_absent_from_its_layer_batch_as_no_color() {
        use crate::spec::encoding::DataType as SpecType;

        let spec = ChartSpec {
            mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SpecType::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SpecType::Quantitative), ..Default::default() }),
                color: Some(EncodingSpec { field: "cat".into(), type_: Some(SpecType::Nominal), ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        // The layer's own batch — no `cat` column, mirroring `qq_line`.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        // Scale resolution must not itself refuse the color column absent
        // from this batch, so resolve against an x/y-only spec — `build`
        // runs on the real spec, which is what this test exercises.
        let scale_spec = ChartSpec {
            encoding: Encoding { color: None, ..spec.encoding.clone() },
            ..spec.clone()
        };
        let (scales, _) = resolve_scales(&scale_spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let result = super::build(&ctx).expect(
            "a color column absent from this layer's own batch must render with the \
             fallback color, not refuse",
        );
        assert_eq!(
            result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Text { .. })).count(),
            2
        );
    }

    /// Text under `CoordPolar`: a label sits on its own theta value as a point
    /// there does, and with `Theta(stack=...)` at the angular midpoint of the
    /// wedge an arc over the same rows draws; on the ring's mid radius unless
    /// a radius binding or a `radius=` style places it. Every case runs the
    /// whole pipeline on the spec shape Python's lowering emits for it, with
    /// the text's theta carrying the wire `stack` Python forwards for
    /// `Theta(stack=True)`.
    pub(crate) mod polar_placement {
        use crate::render::RenderWarning;
        use crate::render::RenderError;
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use ferrum_scene::{SceneGraph, SceneNode};
        use serde_json::{json, Value};
        use std::f64::consts::TAU;
        use std::sync::Arc;

        const G: [&str; 5] = ["a", "b", "a", "b", "c"];
        const Z: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        const R: [f64; 5] = [2.0, 4.0, 6.0, 8.0, 10.0];

        fn batch() -> RecordBatch {
            let schema = Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("z", DataType::Float64, false),
                Field::new("r", DataType::Float64, false),
            ]);
            RecordBatch::try_new(
                Arc::new(schema),
                vec![
                    Arc::new(StringArray::from(G.to_vec())),
                    Arc::new(Float64Array::from(Z.to_vec())),
                    Arc::new(Float64Array::from(R.to_vec())),
                ],
            )
            .unwrap()
        }

        fn polar(inner_radius: f64) -> Value {
            json!({"kind": "polar", "theta": "x", "start_angle": 0.0, "direction": "clockwise",
                   "inner_radius": inner_radius, "pad_angle": 0.0})
        }

        fn spec(mut json: Value, inner_radius: f64) -> crate::spec::chart::ChartSpec {
            json["data"] = json!({"kind": "named", "name": "default"});
            json["coord"] = polar(inner_radius);
            serde_json::from_value(json).unwrap()
        }

        /// A theta binding of `field`, stacked as `Theta(field, stack=True)`
        /// puts it on the wire when `stacked`.
        fn theta(field: &str, stacked: bool) -> Value {
            if stacked { json!({"field": field, "stack": "zero"}) } else { json!({"field": field}) }
        }

        /// `arc(theta="z", color="g") + text(theta=Theta("z", stack=True), text="g")`.
        fn plus_wire_with(stacked: bool) -> Value {
            json!({"mark": "arc", "encoding": {"x": {"field": "z"}}, "layers": [
                {"mark": "arc", "encoding": {"x": {"field": "z"}, "color": {"field": "g"}}},
                {"mark": "text", "encoding": {"x": theta("z", stacked), "text": {"field": "g"}}}]})
        }

        fn plus_wire() -> Value {
            plus_wire_with(true)
        }

        /// `arc(theta="z", color="g").layer(Layer(mark="text",
        /// encoding={"theta": Theta("z", stack=True), "text": "g"}))`.
        fn layer_wire() -> Value {
            json!({"mark": "arc", "encoding": {"x": {"field": "z"}, "color": {"field": "g"}}, "layers": [
                {"mark": "arc", "encoding": {"x": {"field": "z"}, "color": {"field": "g"}}},
                {"mark": "text", "encoding": {"x": theta("z", true), "text": {"field": "g"}}}]})
        }

        /// `arc(theta="sum(z)", color="g") + text(theta=Theta("sum(z)",
        /// stack=True), text="g")`: each layer reads its own aggregate.
        fn summed_wire_with(stacked: bool) -> Value {
            let agg = |name: &str| json!({"type": "aggregate", "ops": [{"field": "z", "fn": "sum", "as": "sum_z"}],
                                          "groupby": ["g"], "name": name});
            json!({"mark": "arc", "encoding": {"x": {"field": "z"}},
                   "transforms": [agg("_layer_agg_0"), agg("_layer_agg_1")], "layers": [
                {"mark": "arc", "encoding": {"x": {"field": "sum_z"}, "color": {"field": "g"}}, "data_source": "_layer_agg_0"},
                {"mark": "text", "encoding": {"x": theta("sum_z", stacked), "text": {"field": "g"}}, "data_source": "_layer_agg_1"}]})
        }

        fn summed_wire() -> Value {
            summed_wire_with(true)
        }

        fn scene(spec: &crate::spec::chart::ChartSpec) -> Result<SceneGraph, RenderError> {
            let (json, _, _) = crate::render::render_scene_json(
                spec,
                &batch(),
                &crate::layout::ThemeInputs::default(),
                crate::layout::Viewport { width: 400.0, height: 300.0 },
                &crate::render::config::RenderConfig::default(),
                &crate::render::chart_config::ChartConfig::default(),
            )?;
            Ok(serde_json::from_str(&json).unwrap())
        }

        fn svg(spec: &crate::spec::chart::ChartSpec) -> Result<(String, Vec<RenderWarning>), RenderError> {
            crate::render::render_svg(
                spec,
                &batch(),
                &crate::layout::ThemeInputs::default(),
                crate::layout::Viewport { width: 400.0, height: 300.0 },
                &crate::render::config::RenderConfig::default(),
                &crate::render::chart_config::ChartConfig::default(),
            )
            .map(|out| (out.bytes, out.warnings))
        }

        /// The panel's polar center and outer radius.
        fn frame(scene: &SceneGraph) -> (f64, f64, f64) {
            let area = &scene.panels[0].plot_area;
            (area.x + area.w / 2.0, area.y + area.h / 2.0, area.w.min(area.h) / 2.0)
        }

        /// Each text node of batch `b`: its source row, content, angle
        /// (clockwise from north, in `[0, TAU)`) and distance from the center.
        fn labels(scene: &SceneGraph, b: usize) -> Vec<(usize, String, f64, f64)> {
            let (cx, cy, _) = frame(scene);
            let batch = &scene.panels[0].marks[b];
            let rows = batch.data_indices.clone().unwrap();
            batch
                .nodes
                .iter()
                .zip(rows)
                .map(|(node, row)| match node {
                    SceneNode::Text { x, y, content, .. } => {
                        let (dx, dy) = (x - cx, cy - y);
                        (row, content.clone(), dx.atan2(dy).rem_euclid(TAU), dx.hypot(dy))
                    }
                    other => panic!("batch {b} draws a non-text node: {other:?}"),
                })
                .collect()
        }

        /// Each wedge of arc batch `b`: its source row and mid-angle.
        fn wedge_mids(scene: &SceneGraph, b: usize) -> Vec<(usize, f64)> {
            let batch = &scene.panels[0].marks[b];
            let spans = batch.polar_spans.clone().expect("an arc batch carries its spans");
            batch
                .data_indices
                .clone()
                .unwrap()
                .into_iter()
                .zip(spans)
                .map(|(row, (a0, a1))| (row, ((a0 + a1) / 2.0).rem_euclid(TAU)))
                .collect()
        }

        /// The cumulative pie mid-angle of each row of `values`, computed
        /// independently of the renderer.
        fn pie_mids(values: &[f64]) -> Vec<f64> {
            let total: f64 = values.iter().sum();
            let mut start = 0.0;
            values
                .iter()
                .map(|v| {
                    let sweep = v / total * TAU;
                    let mid = start + sweep / 2.0;
                    start += sweep;
                    mid
                })
                .collect()
        }

        fn close(a: f64, b: f64) -> bool {
            (a - b).abs() < 1e-9
        }

        #[test]
        fn a_labelled_pie_puts_each_label_inside_its_own_wedge_on_every_wire() {
            for (wire, json) in [("+", plus_wire()), ("layer", layer_wire()), ("sum", summed_wire())] {
                let scene = scene(&spec(json, 0.0)).unwrap_or_else(|e| panic!("{wire}: {e}"));
                let (_, _, outer) = frame(&scene);
                let wedges = wedge_mids(&scene, 0);
                let labels = labels(&scene, 1);
                assert!(!wedges.is_empty(), "{wire}: the arc layer draws wedges");
                assert_eq!(labels.len(), wedges.len(), "{wire}: one label per wedge");
                for ((row, content, angle, radius), (wedge_row, mid)) in labels.iter().zip(&wedges) {
                    assert_eq!(row, wedge_row, "{wire}: label and wedge read the same row");
                    assert!(close(*angle, *mid), "{wire}: row {row} ({content}) at {angle}, its wedge's mid-angle is {mid}");
                    assert!(close(*radius, outer / 2.0), "{wire}: row {row} at radius {radius}, half the outer radius is {}", outer / 2.0);
                }
            }
        }

        #[test]
        fn a_summed_labelled_pie_draws_one_label_per_group() {
            let scene = scene(&spec(summed_wire(), 0.0)).unwrap();
            let contents: Vec<String> = labels(&scene, 1).into_iter().map(|l| l.1).collect();
            assert_eq!(contents, ["a", "b", "c"]);
            let angles: Vec<f64> = labels(&scene, 1).into_iter().map(|l| l.2).collect();
            let expected = pie_mids(&[4.0, 6.0, 5.0]);
            assert!(angles.iter().zip(&expected).all(|(a, e)| close(*a, *e)), "{angles:?} vs {expected:?}");
        }

        #[test]
        fn a_labelled_pie_renders_static() {
            // Each label once per row it draws, plus once in the color legend.
            for (wire, json, drawn) in [
                ("+", plus_wire(), [2, 2, 1]),
                ("layer", layer_wire(), [2, 2, 1]),
                ("sum", summed_wire(), [1, 1, 1]),
            ] {
                let (svg, _) = svg(&spec(json, 0.0)).unwrap_or_else(|e| panic!("{wire}: {e}"));
                for (label, rows) in ["a", "b", "c"].into_iter().zip(drawn) {
                    let count = svg.matches(&format!(">{label}</text>")).count();
                    assert_eq!(count, rows + 1, "{wire}: label {label}");
                }
            }
        }

        #[test]
        fn a_stacked_text_only_polar_chart_stacks_its_theta_like_a_pie() {
            let json = json!({"mark": "text", "encoding": {"x": theta("z", true), "text": {"field": "g"}}});
            let scene = scene(&spec(json, 0.0)).unwrap();
            let (_, _, outer) = frame(&scene);
            let labels = labels(&scene, 0);
            assert_eq!(labels.len(), Z.len());
            for ((row, content, angle, radius), mid) in labels.iter().zip(pie_mids(&Z)) {
                assert_eq!(content, G[*row]);
                assert!(close(*angle, mid), "row {row} at {angle}, expected {mid}");
                assert!(close(*radius, outer / 2.0), "row {row} at radius {radius}");
            }
        }

        #[test]
        fn a_stacked_polar_label_sits_at_its_radius_binding() {
            let json = json!({"mark": "text", "encoding": {
                "x": theta("z", true), "y": {"field": "r"}, "text": {"field": "g"}}});
            let scene = scene(&spec(json, 0.0)).unwrap();
            let (_, _, outer) = frame(&scene);
            let labels = labels(&scene, 0);
            assert_eq!(labels.len(), Z.len());
            for ((row, _, angle, radius), mid) in labels.iter().zip(pie_mids(&Z)) {
                assert!(close(*angle, mid), "row {row} at {angle}, expected {mid}");
                let expected = R[*row] / 10.0 * outer;
                assert!(close(*radius, expected), "row {row} at radius {radius}, expected {expected}");
            }
        }

        #[test]
        fn a_polar_label_sits_at_its_radius_style_value() {
            let json = json!({"mark": "text", "encoding": {"x": {"field": "z"}, "text": {"field": "g"}},
                              "mark_style": {"radius": 30.0}});
            let scene = scene(&spec(json, 0.0)).unwrap();
            for (row, _, _, radius) in labels(&scene, 0) {
                assert!(close(radius, 30.0), "row {row} at radius {radius}");
            }
        }

        #[test]
        fn a_donut_label_sits_mid_ring() {
            let scene = scene(&spec(plus_wire(), 40.0)).unwrap();
            let (_, _, outer) = frame(&scene);
            for (row, _, _, radius) in labels(&scene, 1) {
                assert!(close(radius, (40.0 + outer) / 2.0), "row {row} at radius {radius}");
            }
        }

        #[test]
        fn a_nominal_theta_label_sits_at_its_band_mid_angle() {
            // Stacked, the label takes the arc's own band; unstacked, its own
            // value on the band axis, which is that band's centre.
            for stacked in [true, false] {
                let mut text_theta = theta("g", stacked);
                text_theta["type"] = json!("nominal");
                let json = json!({"mark": "arc", "encoding": {"x": {"field": "g", "type": "nominal"}}, "layers": [
                    {"mark": "arc", "encoding": {"x": {"field": "g", "type": "nominal"}, "color": {"field": "g"}}},
                    {"mark": "text", "encoding": {"x": text_theta, "text": {"field": "g"}}}]});
                let scene = scene(&spec(json, 0.0)).unwrap();
                let wedges = wedge_mids(&scene, 0);
                let labels = labels(&scene, 1);
                assert_eq!(labels.len(), wedges.len(), "stacked={stacked}");
                for ((row, _, angle, _), (wedge_row, mid)) in labels.iter().zip(&wedges) {
                    assert_eq!(row, wedge_row);
                    assert!(close(*angle, *mid), "stacked={stacked}: row {row} at {angle}, its band's mid-angle is {mid}");
                }
            }
        }

        /// An untyped string theta resolves an ordinal scale and is a
        /// category on both marks: the arc draws its bands and the label sits
        /// on each, rather than both reading the strings as a pie's numbers
        /// and drawing nothing.
        #[test]
        fn an_untyped_string_theta_labels_its_bands() {
            let json = json!({"mark": "arc", "encoding": {"x": {"field": "g"}}, "layers": [
                {"mark": "arc", "encoding": {"x": {"field": "g"}, "color": {"field": "g"}}},
                {"mark": "text", "encoding": {"x": {"field": "g"}, "text": {"field": "g"}}}]});
            let scene = scene(&spec(json, 0.0)).unwrap();
            let wedges = wedge_mids(&scene, 0);
            let labels = labels(&scene, 1);
            assert_eq!(wedges.len(), G.len(), "the arc draws one band wedge per row");
            assert_eq!(labels.len(), wedges.len());
            for ((row, _, angle, _), (wedge_row, mid)) in labels.iter().zip(&wedges) {
                assert_eq!(row, wedge_row);
                assert!(close(*angle, *mid), "row {row} at {angle}, its band's mid-angle is {mid}");
            }
        }

        /// A theta-only label beside a polar bar sits on the bar's band:
        /// `bar(theta="g", radius="z") + text(theta="g", text="g")`.
        #[test]
        fn a_theta_only_label_sits_on_its_polar_bar() {
            let json = json!({"mark": "bar", "encoding": {"x": {"field": "g"}, "y": {"field": "z"}}, "layers": [
                {"mark": "bar", "encoding": {"x": {"field": "g"}, "y": {"field": "z"}}},
                {"mark": "text", "encoding": {"x": {"field": "g"}, "text": {"field": "g"}}}]});
            let scene = scene(&spec(json, 0.0)).unwrap();
            let bands = wedge_mids(&scene, 0);
            let labels = labels(&scene, 1);
            assert_eq!(labels.len(), G.len());
            for (row, _, angle, _) in &labels {
                let (_, mid) = bands.iter().find(|(r, _)| r == row).expect("the bar draws every row");
                assert!(close(*angle, *mid), "row {row} at {angle}, its bar's mid-angle is {mid}");
            }
        }

        /// An unstacked theta-only label sits on its own theta value: at the
        /// angle a point with that value draws at, on the mid ring.
        #[test]
        fn an_unstacked_polar_label_sits_on_its_own_theta_value() {
            let json = json!({"mark": "point", "encoding": {"x": {"field": "z"}}, "layers": [
                {"mark": "point", "encoding": {"x": {"field": "z"}, "y": {"field": "r"}}},
                {"mark": "text", "encoding": {"x": {"field": "z"}, "text": {"field": "g"}}}]});
            let scene = scene(&spec(json, 0.0)).unwrap();
            let (cx, cy, outer) = frame(&scene);
            let point_angles: Vec<f64> = scene.panels[0].marks[0]
                .nodes
                .iter()
                .map(|n| match n {
                    SceneNode::Circle { cx: x, cy: y, .. } => (x - cx).atan2(cy - y).rem_euclid(TAU),
                    other => panic!("a point draws circles: {other:?}"),
                })
                .collect();
            let labels = labels(&scene, 1);
            assert_eq!(labels.len(), Z.len());
            for (row, _, angle, radius) in labels {
                let expected = point_angles[row];
                assert!(close(angle, expected), "row {row} at {angle}, its point's angle is {expected}");
                assert!(close(radius, outer / 2.0), "row {row} at radius {radius}");
            }
        }

        /// Polar scatter labels: `point(theta="z", radius="r") +
        /// text(theta="z", radius="r", text="g")` puts each label exactly on
        /// its point.
        #[test]
        fn a_polar_scatter_label_sits_on_its_point() {
            let json = json!({"mark": "point", "encoding": {"x": {"field": "z"}, "y": {"field": "r"}}, "layers": [
                {"mark": "point", "encoding": {"x": {"field": "z"}, "y": {"field": "r"}}},
                {"mark": "text", "encoding": {"x": {"field": "z"}, "y": {"field": "r"}, "text": {"field": "g"}}}]});
            let scene = scene(&spec(json, 0.0)).unwrap();
            let points: Vec<(f64, f64)> = scene.panels[0].marks[0]
                .nodes
                .iter()
                .map(|n| match n {
                    SceneNode::Circle { cx, cy, .. } => (*cx, *cy),
                    other => panic!("a point draws circles: {other:?}"),
                })
                .collect();
            let texts: Vec<(f64, f64)> = scene.panels[0].marks[1]
                .nodes
                .iter()
                .map(|n| match n {
                    SceneNode::Text { x, y, .. } => (*x, *y),
                    other => panic!("a text draws text: {other:?}"),
                })
                .collect();
            assert_eq!(points.len(), Z.len());
            assert_eq!(texts, points, "each label sits on its point");
        }

        /// The text a text layer at index 1 placed by `mark_text` warns with
        /// when its labels sit off an arc's pie.
        pub(crate) const UNSTACKED_WARNING: &str = "layer 1: mark_text: these labels sit at their own theta values, \
            but an arc in the same panel stacks that theta into a pie, so they sit off its wedges; write \
            stack=True on the text's theta (Theta(..., stack=True), on the text or on the theta it inherits) \
            to place each label in its wedge";

        /// Unstacked labels beside a pie of the same theta warn, naming
        /// `Theta(stack=True)`, once; stacked ones, or labels beside no
        /// arc, do not.
        #[test]
        fn unstacked_labels_beside_a_pie_of_their_theta_warn() {
            let unstacked = |warnings: &[RenderWarning]| -> Vec<String> {
                warnings.iter().map(|w| w.to_string()).filter(|w| w.contains("stack=True")).collect()
            };
            let (_, warnings) = svg(&spec(plus_wire_with(false), 0.0)).unwrap();
            assert_eq!(unstacked(&warnings), [UNSTACKED_WARNING], "{warnings:?}");

            // An aggregate's internal output column is no remedy the user can
            // write, so the warning names no column.
            let (_, warnings) = svg(&spec(summed_wire_with(false), 0.0)).unwrap();
            let warned = unstacked(&warnings);
            assert_eq!(warned, [UNSTACKED_WARNING], "{warnings:?}");
            assert!(!warned[0].contains("sum_z"), "{}", warned[0]);

            for (case, json) in [
                ("stacked", plus_wire()),
                ("no arc", json!({"mark": "text", "encoding": {"x": {"field": "z"}, "text": {"field": "g"}}})),
            ] {
                let (_, warnings) = svg(&spec(json, 0.0)).unwrap();
                assert!(unstacked(&warnings).is_empty(), "{case}: {warnings:?}");
            }
        }

        /// `radius=` keeps the pixel contract: below 0 refuses, 0 puts each
        /// label on the center.
        #[test]
        fn a_negative_radius_style_refuses_and_zero_sits_on_the_center() {
            let with = |radius: f64| {
                spec(json!({"mark": "text", "encoding": {"x": {"field": "z"}, "text": {"field": "g"}},
                            "mark_style": {"radius": radius}}), 0.0)
            };
            let err = svg(&with(-10.0)).unwrap_err().to_string();
            assert!(err.contains("mark_text(radius=...) must be non-negative; received -10"), "{err}");
            let scene = scene(&with(0.0)).unwrap();
            for (row, _, _, radius) in labels(&scene, 0) {
                assert!(close(radius, 0.0), "row {row} at radius {radius}");
            }
        }

        /// A polar text layer that no theta reaches refuses with the
        /// positional remedy instead of drawing nothing, though a sibling arc
        /// binds theta: here the text reads its own statistic's rows, which
        /// inherit no positional channel.
        #[test]
        fn a_polar_text_layer_no_theta_reaches_refuses() {
            let mut json = summed_wire();
            json["layers"][1]["encoding"] = json!({"text": {"field": "g"}});
            let err = svg(&spec(json, 0.0)).unwrap_err().to_string();
            assert!(err.contains("layer 1: mark_text needs both positional channels"), "{err}");
        }

        /// With theta on y, a stacked label's `stack` sits on the channel
        /// the value-axis stack shorthand reads; it still places the label in
        /// its wedge, and stacks no value axis.
        #[test]
        fn a_stacked_label_with_theta_on_y_sits_in_its_wedge() {
            let mut sp = spec(json!({"mark": "arc", "encoding": {"y": {"field": "z"}}, "layers": [
                {"mark": "arc", "encoding": {"y": {"field": "z"}, "color": {"field": "g"}}},
                {"mark": "text", "encoding": {"y": theta("z", true), "text": {"field": "g"}, "color": {"field": "g"}}}]}), 0.0);
            if let Some(crate::spec::coord::CoordKind::Polar { theta, .. }) = sp.coord.as_mut() {
                *theta = crate::spec::coord::PolarThetaChannel::Y;
            }
            let scene = scene(&sp).unwrap();
            let wedges = wedge_mids(&scene, 0);
            let labels = labels(&scene, 1);
            assert_eq!(labels.len(), wedges.len());
            for ((row, _, angle, _), (wedge_row, mid)) in labels.iter().zip(&wedges) {
                assert_eq!(row, wedge_row);
                assert!(close(*angle, *mid), "row {row} at {angle}, its wedge's mid-angle is {mid}");
            }
        }

        /// An arc stacks its theta into its pie whatever its `stack` says, so
        /// a stack on its theta under `CoordPolar(theta="y")` (the channel
        /// the value-axis stack shorthand reads) draws the same pie as none.
        #[test]
        fn an_arcs_theta_stack_on_y_draws_the_same_pie() {
            let with = |stacked: bool| {
                let mut sp = spec(json!({"mark": "arc", "encoding": {"y": theta("z", stacked), "color": {"field": "g"}}}), 0.0);
                if let Some(crate::spec::coord::CoordKind::Polar { theta, .. }) = sp.coord.as_mut() {
                    *theta = crate::spec::coord::PolarThetaChannel::Y;
                }
                svg(&sp).map(|(svg, _)| svg)
            };
            assert_eq!(with(true).unwrap(), with(false).unwrap());
        }

        /// With theta on y, an unstacked theta-only label's angle is its
        /// value's position on the y scale, bottom to top as one turn.
        #[test]
        fn an_unstacked_label_with_theta_on_y_sits_at_its_value_on_the_y_scale() {
            let mut sp = spec(json!({"mark": "text", "encoding": {"y": {"field": "z"}, "text": {"field": "g"}}}), 0.0);
            if let Some(crate::spec::coord::CoordKind::Polar { theta, .. }) = sp.coord.as_mut() {
                *theta = crate::spec::coord::PolarThetaChannel::Y;
            }
            let scene = scene(&sp).unwrap();
            let area = &scene.panels[0].plot_area;
            let (scales, _) = crate::render::scale_resolve::resolve_scales(
                &sp,
                &batch(),
                (area.x, area.x + area.w),
                (area.y, area.y + area.h),
                &crate::layout::ThemeInputs::default(),
            )
            .unwrap();
            let labels = labels(&scene, 0);
            assert_eq!(labels.len(), Z.len());
            for (row, _, angle, _) in labels {
                let py = scales.y.to_pixel_f64(Z[row]).unwrap();
                let expected = ((area.y + area.h - py) / area.h * TAU).rem_euclid(TAU);
                assert!(close(angle, expected), "row {row} at {angle}, its y value's angle is {expected}");
            }
        }

        #[test]
        fn a_radius_style_without_a_ring_placement_warns_inert() {
            let inert = |warnings: &[RenderWarning]| {
                warnings.iter().any(|w| matches!(w, RenderWarning::MarkStyleInert { field, .. } if field == "radius"))
            };
            let bound = json!({"mark": "text", "encoding": {
                "x": {"field": "z"}, "y": {"field": "r"}, "text": {"field": "g"}}, "mark_style": {"radius": 30.0}});
            let (_, warnings) = svg(&spec(bound, 0.0)).unwrap();
            assert!(inert(&warnings), "a radius binding places the label: {warnings:?}");

            let mut cartesian = spec(json!({"mark": "text", "encoding": {
                "x": {"field": "z"}, "y": {"field": "r"}, "text": {"field": "g"}}, "mark_style": {"radius": 30.0}}), 0.0);
            cartesian.coord = None;
            let (_, warnings) = svg(&cartesian).unwrap();
            assert!(inert(&warnings), "a Cartesian label has no wedge: {warnings:?}");

            let placed = json!({"mark": "text", "encoding": {"x": {"field": "z"}, "text": {"field": "g"}},
                                "mark_style": {"radius": 30.0}});
            let (_, warnings) = svg(&spec(placed, 0.0)).unwrap();
            assert!(!inert(&warnings), "control: a theta-only polar label draws its radius: {warnings:?}");
        }
    }
}
