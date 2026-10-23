//! Legend + colorbar construction for `prepare_render_inputs`.
//!
//! Builds the color legend (categorical entries / continuous colorbar /
//! conditional-color fallback), the per-channel legend style overrides, the
//! size/shape auxiliary legend blocks, and the legend title — including the
//! same-field color+size merge that suppresses the redundant colorbar. The
//! orchestrator calls [`build_color_legend`] and unpacks the returned bundle
//! straight into `PreparedInputs`.

use crate::layout::{
    AuxLegendInput, ColorbarInput, LegendDirection, LegendEntry, LegendOrient, OpacityLegendEntry,
    ShapeLegendEntry, SizeLegendEntry, StrokeDashLegendEntry, StrokeWidthLegendEntry, SymbolKind,
};
use crate::layout::axis::resolve_axis_title;
use crate::render::chart_config::LegendStyleSpec;
use crate::render::format::Temporality;
use crate::spec::chart::{ChartConditional, ChartSpec};
use crate::spec::encoding::{Encoding, EncodingSpec};
use arrow::record_batch::RecordBatch;
use ferrum_scene::ChannelName;
use pyo3::prelude::*;

use super::super::scale_resolve::{LeafScaleContext, ResolvedScales};
use super::LegendPreparedOverrides;

/// What the legends read off a chart. Each legend channel is bound as its
/// scale was built, from the rendering encoding
/// (`PreparedInputs::rendering_encoding()`): the binding of the layer that
/// establishes the channel, layer 0 when it draws it and otherwise the first
/// layer that does, so the legend's title and `Legend(...)` style come from
/// that binding and never from a later layer's. `x`/`y` stay as
/// authored: they feed only the color legend's `Legend(...)` cascade, which
/// must not swap under `CoordFlip`.
pub(crate) struct LegendBindings<'a> {
    color: Option<&'a EncodingSpec>,
    size: Option<&'a EncodingSpec>,
    shape: Option<&'a EncodingSpec>,
    stroke_dash: Option<&'a EncodingSpec>,
    opacity: Option<&'a EncodingSpec>,
    fill_opacity: Option<&'a EncodingSpec>,
    stroke_opacity: Option<&'a EncodingSpec>,
    stroke_width: Option<&'a EncodingSpec>,
    x: Option<&'a EncodingSpec>,
    y: Option<&'a EncodingSpec>,
    conditionals: &'a [ChartConditional],
}

impl<'a> LegendBindings<'a> {
    /// The legend channels of `rendering`, the authored `x`/`y` and the
    /// conditionals of `spec`.
    pub(crate) fn new(spec: &'a ChartSpec, rendering: &'a Encoding) -> Self {
        Self {
            color: rendering.color.as_ref(),
            size: rendering.size.as_ref(),
            shape: rendering.shape.as_ref(),
            stroke_dash: rendering.stroke_dash.as_ref(),
            opacity: rendering.opacity.as_ref(),
            fill_opacity: rendering.fill_opacity.as_ref(),
            stroke_opacity: rendering.stroke_opacity.as_ref(),
            stroke_width: rendering.stroke_width.as_ref(),
            x: spec.encoding.x.as_ref(),
            y: spec.encoding.y.as_ref(),
            conditionals: &spec.conditionals,
        }
    }
}

/// The full color-legend output produced from the resolved scales + bindings.
/// Unpacked straight into `PreparedInputs` by the orchestrator.
pub(crate) struct ColorLegendBundle {
    pub legend_entries: Vec<LegendEntry>,
    pub colorbar: Option<ColorbarInput>,
    pub legend_title: Option<String>,
    pub legend_overrides: LegendPreparedOverrides,
    pub aux_legends: Vec<AuxLegendInput>,
}

impl ColorLegendBundle {
    /// No legend at all: a chart whose legends another chart draws.
    pub(crate) fn none() -> Self {
        Self {
            legend_entries: Vec::new(),
            colorbar: None,
            legend_title: None,
            legend_overrides: LegendPreparedOverrides::default(),
            aux_legends: Vec::new(),
        }
    }
}

/// The per-channel `legend={...}` specs that address the chart's ONE legend
/// surface, in precedence order.
///
/// The color channel owns that surface, so its own `legend=` wins outright.
/// `X(legend=...)` / `Y(legend=...)` — documented as honored on the positional
/// channels (`encoding/positional.py`'s Notes, `_honored.py`'s
/// `PRIMARY_POSITIONAL`) but until now routed nowhere (NF-B13, adjudicated
/// 2026-09-02: implement) — fill the fields color left unset. A positional
/// channel has no legend block of its own; the surface its `legend=` can
/// address is the chart's, which is what this cascade gives it.
///
/// Precedence within the per-channel level is color > x > y; the whole level
/// still beats chart-level `configure_legend` and the theme, per the batch's
/// one cascade discipline.
fn per_channel_legend_specs<'a>(bindings: &LegendBindings<'a>) -> Vec<&'a LegendStyleSpec> {
    [
        bindings.color,
        bindings.x,
        bindings.y,
    ]
    .into_iter()
    .flatten()
    .filter_map(|e| e.legend.as_deref())
    .collect()
}

/// First `Some` answer for `field` across `specs`, in their precedence order —
/// the one-line fill-only cascade every per-channel legend override reads
/// through, so no field can be wired to a different precedence than its
/// siblings by hand.
fn pick<'a, T>(
    specs: &[&'a LegendStyleSpec],
    field: impl Fn(&'a LegendStyleSpec) -> Option<T>,
) -> Option<T> {
    specs.iter().find_map(|s| field(s))
}

/// Walk `bindings.conditionals` for the given `channel` and collect the field names
/// referenced by either branch (`if_selected` / `if_not`) as
/// `EncodingValue::Field { name }`. Dedup preserving first-appearance order.
/// Returns an empty `Vec` when no field-driven conditional exists for `channel`.
///
/// Generalizes the former `resolve_conditional_color_domain`: that function is
/// now this for `ChannelName::Color` followed by `distinct_values_in_order` over
/// each field's values (see the `None` arm of [`build_color_legend`]).
pub(crate) fn resolve_conditional_field_names(
    bindings: &LegendBindings<'_>,
    channel: ChannelName,
) -> Vec<String> {
    use ferrum_scene::EncodingValue;
    let mut seen = std::collections::HashSet::<String>::new();
    let mut out = Vec::<String>::new();

    for cond in bindings.conditionals.iter().map(|c| &c.encoding) {
        if cond.channel != channel {
            continue;
        }
        // Collect field names from both branches; each branch may independently
        // be a Field reference (vs a literal value).
        let candidates: [&EncodingValue; 2] = [&cond.if_selected, &cond.if_not];
        for ev in candidates {
            if let EncodingValue::Field { name } = ev {
                if seen.insert(name.clone()) {
                    out.push(name.clone());
                }
            }
        }
    }
    out
}

/// Build the categorical-color domain from `when(Color(field))` conditionals when
/// no base color encoding exists. For each field referenced by a Color
/// conditional, extend the output with `distinct_values_in_order` over
/// `transformed`. Dedup preserving first-appearance order.
///
/// Used by the legend-build arm when `provisional_scales.color` is `None` (i.e.
/// no base color encoding) so that `when(Color(field))` conditionals still
/// produce a categorical legend whose entries the WASM `bind="legend"` toggle
/// can use.
fn resolve_conditional_color_domain(bindings: &LegendBindings<'_>, transformed: &RecordBatch) -> Vec<String> {
    let mut seen = std::collections::HashSet::<String>::new();
    let mut out = Vec::<String>::new();
    for name in resolve_conditional_field_names(bindings, ChannelName::Color) {
        // Ignore errors (field not present in batch) — just skip.
        if let Ok(values) = super::super::arrow_cast::distinct_values_in_order(transformed, &name) {
            for v in values {
                if seen.insert(v.clone()) {
                    out.push(v);
                }
            }
        }
    }
    out
}

/// Build the full color legend / colorbar bundle from the resolved scales + bindings.
///
/// Mirrors the orchestrator's former inline legend block exactly: categorical
/// color → discrete entries; continuous color → colorbar gradient + ticks;
/// no base color but a conditional Color field → conditional-domain entries.
/// Then derives the legend title, extracts the per-channel legend style
/// overrides, builds the size/shape aux legends, and applies the same-field
/// color+size merge (which suppresses the now-redundant colorbar).
///
/// `colorbar_color` is the color scale with the chart-level
/// `configure_color(domain=, range=)` applied
/// ([`crate::render::config_apply::configured_color_scale`]), the scale the
/// marks are drawn from. Only the colorbar arms (continuous gradient and
/// discretizing swatches) read it; whether a bar exists, the categorical
/// entries and everything else read `provisional_scales`.
///
/// `layers` is this panel's prepared layer list (from `build_layers`, already
/// available at the caller before this is invoked) — the single coherent
/// point where the color-legend decision can see the mark set consuming the
/// resolved color scale, which `provisional_scales` alone cannot (spec §4.0,
/// 2026-08-28: line/ribbon's inert-continuous-color suppression, see
/// [`inert_numeric_color_on_line_or_ribbon_only`]). Under the composite path
/// `layers` only ever sees THIS leaf's own marks; `leaf_scales` carries the
/// composite seam's verdicts on what sibling leaves ELSEWHERE in the same
/// Overlay group draw:
/// [`LeafScaleContext::color_scale_has_non_line_ribbon_sibling`] (a sibling
/// renders the shared color scale,
/// `render::composite_render::plan_line_ribbon_color_group_exemptions`) and
/// [`LeafScaleContext::size_scale_has_non_line_sibling`] (a sibling draws the
/// size field as dots, so the size legend keeps circle swatches). `None` for
/// every standalone (flat/facet) render.
/// `warnings` is the panel's live warnings sink (mirrors `build_axes`'s
/// `&mut scale_warnings` at the same call site).
pub(crate) fn build_color_legend(
    bindings: &LegendBindings<'_>,
    transformed: &RecordBatch,
    provisional_scales: &ResolvedScales,
    colorbar_color: Option<&crate::render::scale_resolve::ColorScale>,
    layers: &[super::LayerPrepared],
    leaf_scales: Option<&LeafScaleContext>,
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> ColorLegendBundle {
    use super::super::scale_resolve::ColorScale;

    // Schwabish SB3 (2026-05-11): respect ``legend={"disabled": true}`` on the
    // color encoding by emitting no legend entries AND no colorbar. The
    // Python ``Color`` class translates ``legend=None`` / ``legend=False``
    // from ``encode(color=Color(field, legend=None))`` into this JSON shape
    // so direct-label diagnostic charts can opt out of redundant legends.
    //
    // COUPLING (GH #74): the chart-level ``configure_legend(orient="none")``
    // mirror in ``render::config_apply::suppress_legend_if_chart_level_disabled``
    // post-hoc clears the SAME
    // legend-content fields this branch leaves empty (entries, colorbar, plus
    // the chart-wide extras aux_legends/title). If a new legend-content field
    // is added to ``PreparedInputs`` and wired here, wire it into that clear
    // block too, or a chart-level disable will silently leave it populated.
    //
    // NF-B13 / spec §4.4: the suppression question — like every other
    // per-channel legend override below — is answered by the color > x > y
    // cascade, so `X(legend=None)` reaches the same consumer `Color(legend=
    // None)` does. `orient="none"` is the second spelling of the same intent,
    // and both resolve field-by-field with the same first-`Some` rule the
    // `pick`s below use (see `LegendStyleSpec::suppressed_by`).
    let legend_specs = per_channel_legend_specs(bindings);
    let legend_disabled = LegendStyleSpec::suppressed_by(&legend_specs);
    // Colorbar label overrides, shared by the continuous (gradient) and the
    // discretizing (k-swatch) arms: an explicit `tickLabels` list replaces the
    // computed labels outright, and `format=` applies a Python-style format spec
    // to each computed value.
    let custom_tick_labels: Option<Vec<String>> = pick(&legend_specs, |l| l.tick_labels.clone());
    let format_spec: Option<&str> = pick(&legend_specs, |l| l.format.as_deref());
    // D8 (legend half of format_type threading, Task 4): a colorbar whose
    // domain is temporal (`format_type == "time"`) formats each sampled tick
    // value as an epoch-ms timestamp instead of a d3 numeric spec — mirrors
    // the axis half (`AxisStyleSpec.label_format_type`).
    let format_type: Option<&str> = pick(&legend_specs, |l| l.format_type.as_deref());
    // The color field's temporality decides whether an undeclared `%`-bearing
    // spec is a date pattern, as the axis's scale does (`#147` T23): the one
    // answer `refuse_format_temporality_mismatch_legend_fields` checked.
    let color_temporality = bindings.color
        .and_then(|enc| super::field_temporality(enc, transformed))
        .unwrap_or(Temporality::NonTemporal);
    // The two gates below (`format_spec.is_some() || format_type ==
    // Some("time")`) are deliberately narrower than "either field set"
    // (quality-review S2, 2026-09-03): `fm.Legend(format_type=...)` alone,
    // with NO `format=`, is a real public per-channel parameter
    // (`ferrum/legend.py`'s `format_type`), and any non-`"time"` value
    // (e.g. `format_type="number"`, which `resolve_format_field` can emit)
    // must fall through to the EXISTING range-aware `format_colorbar_tick`
    // default, not divert into `format_value_with_spec(v, None, ...)` =
    // `format_numeric` — a byte-identity break for that combination the
    // original `format_type.is_some()` gate introduced undisclosed. Only
    // `"time"` genuinely needs the new path (a colorbar with no explicit
    // spec still needs SOME way to know its domain is temporal).

    let (mut legend_entries, colorbar): (Vec<LegendEntry>, Option<ColorbarInput>) = if legend_disabled {
        (Vec::new(), None)
    } else {
        match &provisional_scales.color {
            Some(ColorScale::Categorical { domain, .. }) => {
                let entries = domain
                    .iter()
                    .map(|v| LegendEntry::new(v.clone(), SymbolKind::Circle))
                    .collect();
                (entries, None)
            }
            Some(ColorScale::Continuous { domain, scheme, midpoint, transform }) => {
                // The bar is drawn from `colorbar_color`, the scale with
                // `configure_color` applied. Contract: it has the resolved
                // scale's variant, because `apply_color_config_to_color_scale`
                // only mutates fields of the variant it matched, and
                // `prepare_render_inputs`, the one production caller, always
                // passes that configured clone. The fallback arm (no clone, or
                // another variant) is reached only by a caller passing `None`,
                // and then the resolved scale is the one the marks read.
                let (domain, scheme, midpoint, transform) = match colorbar_color {
                    Some(ColorScale::Continuous { domain, scheme, midpoint, transform }) => {
                        (domain, scheme, midpoint, transform)
                    }
                    _ => (domain, scheme, midpoint, transform),
                };
                let stops = continuous_colorbar_stops(*domain, scheme, *midpoint, *transform);
                // Tick labels: check for explicit tickLabels override from the
                // typed legend spec (e.g. ["Low", "High"] for SHAP beeswarm),
                // else compute 5 ticks across the domain at 0, 0.25, 0.5, 0.75, 1.0,
                // each reading the data value at its position through the
                // scale's transform (raw counts on a symlog raster bar).
                // When `format=` is set, apply a Python-style format spec to each tick value.
                let custom_tick_labels = custom_tick_labels.clone();
                // `cb_domain` is the numeric span the labels cover — carried for
                // `tick_min_step` thinning in `compute_layout`. `None` for explicit
                // non-numeric label overrides (their step is undefined).
                let (tick_labels, cb_domain) = if let Some(labels) = custom_tick_labels {
                    (labels, None)
                } else {
                    let (lo, hi) = *domain;
                    let labels = (0..5)
                        .map(|i| {
                            let t = i as f64 / 4.0;
                            let v = transform.value_at((lo, hi), t);
                            if format_spec.is_some() || format_type == Some("time") {
                                crate::render::format::format_value_with_spec(
                                    v,
                                    format_spec,
                                    format_type,
                                    color_temporality,
                                )
                            } else {
                                format_colorbar_tick(v, lo, hi)
                            }
                        })
                        .collect();
                    (labels, Some((lo, hi)))
                };
                (
                    Vec::new(),
                    Some(ColorbarInput {
                        stops,
                        tick_labels,
                        domain: cb_domain,
                    }),
                )
            }
            Some(ColorScale::Discretizing(buckets)) => {
                // Swatches from the configured scale, under the same variant
                // contract as the continuous arm above.
                let buckets = match colorbar_color {
                    Some(ColorScale::Discretizing(configured)) => configured,
                    _ => buckets,
                };
                // Discrete colorbar: emit each swatch as a pair of gradient
                // stops at its own edges, so the `linearGradient` paints k flat
                // bands instead of interpolating between them. Reusing the
                // gradient carrier (rather than a second colorbar shape) keeps
                // the layout, SVG, and WASM colorbar paths single-sourced.
                let swatches = buckets.colors();
                let k = swatches.len() as f64;
                let stops: Vec<(f64, String)> = swatches
                    .iter()
                    .enumerate()
                    .flat_map(|(i, c)| {
                        let hex = super::super::color::fmt_svg(*c);
                        [(i as f64 / k, hex.clone()), ((i + 1) as f64 / k, hex)]
                    })
                    .collect();
                // One label per boundary (k + 1 of them) — the layout
                // distributes labels linearly over the bar, which lands each on
                // its own band edge.
                let bounds = buckets.bounds();
                let (lo, hi) = (bounds[0], bounds[bounds.len() - 1]);
                let tick_labels = custom_tick_labels.unwrap_or_else(|| {
                    bounds
                        .iter()
                        .map(|&v| {
                            if format_spec.is_some() || format_type == Some("time") {
                                crate::render::format::format_value_with_spec(
                                    v,
                                    format_spec,
                                    format_type,
                                    color_temporality,
                                )
                            } else {
                                format_colorbar_tick(v, lo, hi)
                            }
                        })
                        .collect()
                });
                // `domain: None`: `tick_min_step` thinning assumes labels
                // sampled evenly across a span, but bucket boundaries are
                // spaced by the scale's thresholds, not by the bar.
                (Vec::new(), Some(ColorbarInput { stops, tick_labels, domain: None }))
            }
            None => {
                // #9 [FA-15]: when no base color encoding is set, check whether
                // a conditional color encoding (`when(Color(field))`) provides a
                // categorical domain.  If so, build legend entries from the
                // field's distinct values so `bind="legend"` has categories to
                // toggle.  Gated on `provisional_scales.color == None` so base-
                // color charts remain byte-identical.
                let cond_domain = resolve_conditional_color_domain(bindings, transformed);
                if cond_domain.is_empty() {
                    (Vec::new(), None)
                } else {
                    let entries = cond_domain
                        .iter()
                        .map(|v| LegendEntry::new(v.clone(), SymbolKind::Circle))
                        .collect();
                    (entries, None)
                }
            }
        }
    };

    // D13 (B5-typed): extract legend style overrides from the typed per-channel
    // `legend` specs (color > x > y, see `per_channel_legend_specs`). Per-channel
    // precedence: these win over chart-level `configure_legend` (which fills only
    // what is still `None`).
    let legend_overrides = LegendPreparedOverrides {
        // `"none"` is absent here on purpose: it is a suppression, not a
        // placement, and was already consumed by `legend_disabled` above.
        orient: pick(&legend_specs, |l| l.orient.as_deref()).and_then(LegendOrient::parse),
        title: pick(&legend_specs, |l| l.title.clone()),
        title_font_size: pick(&legend_specs, |l| l.title_font_size),
        columns: pick(&legend_specs, |l| l.columns),
        tick_count: pick(&legend_specs, |l| l.tick_count).map(|n| n as usize),
        label_font_size: pick(&legend_specs, |l| l.label_font_size),
        gradient_length: pick(&legend_specs, |l| l.gradient_length),
        gradient_thickness: pick(&legend_specs, |l| l.gradient_thickness),
        direction: pick(&legend_specs, |l| l.direction.as_deref())
            .and_then(LegendDirection::parse),
        // `values`: explicit tick/entry labels for the legend. Accepts an array of
        // strings or numbers. Numbers are formatted to a short decimal string.
        values: pick(&legend_specs, |l| l.values.as_ref()).map(|arr| {
            arr.iter()
                .map(|item| {
                    if let Some(s) = item.as_str() {
                        s.to_string()
                    } else if let Some(n) = item.as_f64() {
                        if n.fract() == 0.0 && n.abs() < 1e15 {
                            format!("{}", n as i64)
                        } else {
                            format!("{:.4}", n)
                                .trim_end_matches('0')
                                .trim_end_matches('.')
                                .to_string()
                        }
                    } else {
                        item.to_string()
                    }
                })
                .collect()
        }),
        // `type`: "gradient" forces colorbar path; "symbol" forces categorical entries.
        legend_type: pick(&legend_specs, |l| l.legend_type.clone()),
        symbol_type: pick(&legend_specs, |l| l.symbol_type.clone()),
        // B5 unit 3: orphan legend styling/positioning fields. Per-channel here;
        // chart-level `configure_legend(...)` fills any that stay `None`.
        symbol_stroke_width: pick(&legend_specs, |l| l.symbol_stroke_width),
        row_padding: pick(&legend_specs, |l| l.row_padding),
        column_padding: pick(&legend_specs, |l| l.column_padding),
        label_limit: pick(&legend_specs, |l| l.label_limit),
        clip_height: pick(&legend_specs, |l| l.clip_height),
        tick_min_step: pick(&legend_specs, |l| l.tick_min_step),
        zindex: pick(&legend_specs, |l| l.zindex),
        // B5 unit 6a orphans. Per-channel here; chart-level `configure_legend`
        // fills any that stay `None`.
        symbol_size: pick(&legend_specs, |l| l.symbol_size),
        label_color: pick(&legend_specs, |l| l.label_color.clone()),
        offset: pick(&legend_specs, |l| l.offset),
        padding: pick(&legend_specs, |l| l.padding),
        title_padding: pick(&legend_specs, |l| l.title_padding),
    };

    // spec §4.0 (2026-08-28) / spec-review 2026-08-28 (cycle-3 finding):
    // computed HERE, before `build_aux_legends`, so the same-field color+size
    // merge below can also see it — a merge whose color scale is inert on
    // line/ribbon must not sample it into the size legend's swatches either
    // (the size legend renders in the neutral/default swatch color, exactly
    // as if no color merge existed — size's own semantics on line stay a
    // separate follow-up, out of this wave). Reused again below for the
    // warning itself, so it is computed exactly once.
    let inert_consumer_marks = inert_numeric_color_on_line_or_ribbon_only(
        provisional_scales,
        layers,
        leaf_scales.is_some_and(|c| c.color_scale_has_non_line_ribbon_sibling),
    );

    // Multivariate B1: build size/shape/dash auxiliary legends from the
    // resolved scales. A size/shape/dash channel that shares its field with a
    // color legend that ALREADY enumerates that field is folded into that
    // legend rather than emitted as a separate block (#111, spec §4.6) — and
    // folding means the color entries take the folded channel's identity, not
    // that the identity is dropped.
    let AuxLegends { blocks: aux_legends, folded } = build_aux_legends(
        bindings,
        transformed,
        provisional_scales,
        ColorLegendState {
            inert_on_line_or_ribbon: inert_consumer_marks.is_some(),
            enumerates_field: !legend_entries.is_empty(),
        },
        size_legend_symbol(
            bindings,
            layers,
            leaf_scales.is_some_and(|c| c.size_scale_has_non_line_sibling),
        ),
        DrawnAuxChannels::of(layers),
    );
    fold_identity_into_color_entries(&mut legend_entries, &folded, warnings);

    // Legend title (Themes-T2.5b, #90): three-state per spec §6 "Legend title
    // resolution" — `encoding.color.title` resolves through the same
    // `resolve_axis_title` helper the axis path uses (present+blank →
    // suppress with no fallback, present+non-blank → verbatim, absent →
    // field name). When entries were built from a conditional color field
    // (no base encoding), derive the title from the first conditional Color
    // Field branch instead — the conditional branch carries no `title` of
    // its own, so that leg is unaffected by #90.
    //
    // #111 / spec §4.6: a folded channel's own EXPLICIT title addresses this
    // same (now shared) surface, so it fills in where color stated none
    // rather than being silently discarded, and a genuine disagreement is
    // reported. Resolved after `build_aux_legends` because only that call
    // knows which channels were actually folded.
    let legend_title = if !legend_entries.is_empty() || colorbar.is_some() {
        let color_title = match bindings.color {
            Some(c) => resolve_axis_title(c.title.as_deref(), Some(c.field.clone())),
            None => {
                // Conditional-color case: find the first Color conditional with a
                // Field branch and use its field name as the title.
                use ferrum_scene::EncodingValue;
                bindings.conditionals.iter().map(|c| &c.encoding).find_map(|cond| {
                    if cond.channel != ChannelName::Color {
                        return None;
                    }
                    for ev in [&cond.if_selected, &cond.if_not] {
                        if let EncodingValue::Field { name } = ev {
                            return Some(name.clone());
                        }
                    }
                    None
                })
            }
        };
        let color_said = bindings.color
            .map_or(ColorTitle::Unstated, color_title_state);
        resolve_shared_legend_title(color_said, color_title, &folded.titles, warnings)
    } else {
        None
    };

    // Same-field merge: when color (continuous) and size share a field, the
    // combined block is the size legend whose symbols also carry color
    // (`color_hex`). Suppress the now-redundant colorbar so a single combined
    // legend renders rather than a colorbar plus a size legend.
    //
    // Detected as the AND of two conditions (spec-review 2026-08-28, cycle-4
    // finding, correcting cycle 3): `same_field_numeric_size_color_merge`
    // alone only proves the FIELD/SCALE-KIND condition holds — it says
    // nothing about whether `build_aux_legends` actually emitted a merged
    // Size block. Size legend emission has its own independent gates
    // (`legend_channel_disabled`, `scales.size.is_some()`,
    // `size_scale.inner.data_domain()` returning `Some`, non-empty entries),
    // any of which can make `aux_legends` carry NO `Size` entry even though
    // the field/scale-kind condition holds — e.g. `mark_point().encode(
    // color='v:Q', size=Size('v', legend=None))`. Nulling the colorbar in
    // that case would drop the chart's ONLY color legend for a merge that
    // never actually happened. Checking for an emitted `AuxLegendInput::Size`
    // block (rather than reintroducing the `color_hex.is_some()` coupling
    // cycle 3 correctly broke — that signal is unreliable now that inert
    // line/ribbon withholds `color_hex` even when Size WAS emitted) captures
    // exactly "a merged block a user will actually see" without re-deriving
    // any of `build_aux_legends`'s own emission gates here.
    //
    // Keeping the merge (not the inert-check below) as the nuller when both
    // conditions hold also keeps the warning's `suppressed` field `false`
    // for the inert-line/ribbon case, per the cycle-2 adjudicated ruling:
    // the merge, not the inert check, is the one redirecting the legend.
    let colorbar = {
        let merged_color_size = same_field_numeric_size_color_merge(bindings, provisional_scales)
            && aux_legends.iter().any(|a| matches!(a, AuxLegendInput::Size { .. }));
        if merged_color_size {
            None
        } else {
            colorbar
        }
    };

    // spec §4.0 (2026-08-28), decoupled per the 2026-08-28 spec-review
    // ruling: loudness is a property of the CHANNEL, not of whether a
    // colorbar happens to render. A Continuous/Discretizing color scale is
    // inert on line/ribbon (no per-segment color) whenever EVERY consumer of
    // it draws with one of those marks — that is true (and the warning must
    // fire) regardless of `colorbar`'s current state: `Color(v, legend=None)`
    // warns even though no colorbar was ever built, and the same-field
    // color+size merge above warns even though ITS colorbar was already
    // nulled (that case's size legend still carries the inert scale's
    // colored swatches — observed behavior, no change here). Colorbar
    // suppression is the ADDITIONAL consequence, applied unconditionally
    // alongside the warning (a no-op when `colorbar` was already `None`).
    let colorbar = if let Some(consumer_marks) = inert_consumer_marks {
        use super::super::scale_resolve::ColorScale;
        // Explicit arms, no wildcard (spec-review 2026-08-28, cycle-4
        // finding): `inert_numeric_color_on_line_or_ribbon_only` already
        // proved `provisional_scales.color` is `Some(Continuous | Discretizing)`
        // via `ColorScale::input() == ColorInput::Numeric`, so the remaining
        // arms are unreachable today — but naming them explicitly rather
        // than `_ =>` means a FUTURE numeric-keyed `ColorScale` variant fails
        // to compile here (forcing an explicit decision) instead of silently
        // falling through the wildcard into a runtime panic on a path
        // reachable from the `render_svg`/`render_composite_svg` PyO3 entries.
        let scale_kind = match &provisional_scales.color {
            Some(ColorScale::Continuous { .. }) => "continuous",
            Some(ColorScale::Discretizing(_)) => "discretizing",
            Some(ColorScale::Categorical { .. }) | None => {
                unreachable!("inert_numeric_color_on_line_or_ribbon_only already confirmed a Numeric-keyed scale")
            }
        };
        let mut marks: Vec<String> = Vec::new();
        for m in &consumer_marks {
            let name = m.as_str().to_string();
            if !marks.contains(&name) {
                marks.push(name);
            }
        }
        // 2026-08-28 spec-review ruling: the message must only claim a
        // legend was suppressed when one actually existed to suppress —
        // captured BEFORE nulling `colorbar` below, so `legend=None` and the
        // same-field color+size merge (whose colorbar was already `None`
        // when this arm runs) get the accurate "no per-mark effect" wording
        // instead of a false suppression claim.
        let suppressed = colorbar.is_some();
        warnings.push(crate::render::RenderWarning::UnsupportedColorScaleOnMark {
            marks,
            scale_kind: scale_kind.to_string(),
            suppressed,
        });
        None
    } else {
        colorbar
    };

    let legend_title = if colorbar.is_none() && legend_entries.is_empty() {
        None
    } else {
        legend_title
    };

    ColorLegendBundle {
        legend_entries,
        colorbar,
        legend_title,
        legend_overrides,
        aux_legends,
    }
}

/// `#147` close (wave 6A T21/T22): a time-typed legend format on a
/// non-temporal field, or the ordinal preset on a temporal one, is refused
/// unless the caller disabled that legend.
///
/// This validates the spec the caller wrote, not what happens to be drawn:
/// the color legend (the colorbar, or a categorical legend) and every aux
/// legend channel [`build_aux_legends`] builds (size, shape, stroke-dash,
/// read from the one [`aux_legend_channel_pairs`] list both sides walk) each
/// check their own field through the rule the axes use
/// ([`super::refuse_format_temporality_mismatch_field`]), so the refusal
/// fires whatever the scale kind, and also where the renderer itself later
/// suppresses the colorbar with a warning (a line or ribbon layer that cannot
/// draw continuous color), folds it into the size legend, or folds a shape or
/// dash block into the color legend. Shape and stroke-dash legends never
/// format their labels, so before this they dropped the pair silently where
/// the color legend refused it. The only
/// exemption is a legend the caller disabled, and both spellings of that
/// count alike: per channel (`Color(legend=None)`, `Legend(orient="none")`,
/// read over the same color > x > y cascade the color legend's format is
/// read from) and chart-level (`configure_legend(orient="none")`). Before,
/// these surfaces rendered every label as a `1970-01-01` epoch date.
///
/// Axes have no such exemption (`X(axis=None, format=..., format_type=
/// "time")` refuses) on purpose: an `X`/`Y` `Legend(...)` format cascades
/// onto the color field, so validating a legend nobody draws would judge a
/// field the caller never formatted, while an axis format is written on the
/// channel it formats.
///
/// The refusal names the `'color'` legend even when the format reached it
/// through the cascade from an `X`/`Y` `Legend(...)`: the value checked is
/// the color field, and naming `'x'` would pin the "not temporal" verdict on
/// a field that was never checked.
pub(crate) fn refuse_format_temporality_mismatch_legend_fields(
    bindings: &LegendBindings<'_>,
    transformed: &RecordBatch,
    chart_config: &crate::render::chart_config::ChartConfig,
) -> Result<(), crate::render::RenderError> {
    use crate::render::FormatSurface;
    if crate::render::config_apply::chart_config_legend_disabled(chart_config) {
        return Ok(());
    }
    let legend_specs = per_channel_legend_specs(bindings);
    if let Some(color_enc) = bindings.color {
        if !LegendStyleSpec::suppressed_by(&legend_specs) {
            super::refuse_format_temporality_mismatch_field(
                color_enc,
                transformed,
                pick(&legend_specs, |l| l.format.as_deref()),
                pick(&legend_specs, |l| l.format_type.as_deref()),
                "color",
                FormatSurface::Legend,
            )?;
        }
    }
    for (channel, enc) in aux_legend_channel_pairs(bindings) {
        let Some(enc) = enc else { continue };
        if legend_channel_disabled(Some(enc)) {
            continue;
        }
        let legend = enc.legend.as_deref();
        super::refuse_format_temporality_mismatch_field(
            enc,
            transformed,
            legend.and_then(|l| l.format.as_deref()),
            legend.and_then(|l| l.format_type.as_deref()),
            channel,
            FormatSurface::Legend,
        )?;
    }
    Ok(())
}

/// Canonical aux-legend channel names, in block order: the one list both
/// [`aux_legend_channel_pairs`] (the Rust-side per-spec builder/validator,
/// directly below) and the Python-visible [`aux_legend_channels`] pyfunction
/// read (#123), so Python reads the set instead of mirroring it.
const AUX_LEGEND_CHANNEL_NAMES: [&str; 7] =
    ["size", "shape", "stroke_dash", "opacity", "fill_opacity", "stroke_opacity", "stroke_width"];

/// The aux legend channels paired with their binding, in block order:
/// the one list [`build_aux_legends`] destructures and
/// [`refuse_format_temporality_mismatch_legend_fields`] walks. The guarantee
/// runs one way: a new entry here fails to compile at the destructure until
/// the builder handles it, but an aux block the builder reads straight from
/// the [`LegendBindings`] (bypassing this list) would compile without meeting the
/// `#147` rule, so a new aux legend channel belongs in this list. Channel
/// names come from [`AUX_LEGEND_CHANNEL_NAMES`], the same array the
/// Python-visible [`aux_legend_channels`] pyfunction reads, so the two
/// cannot drift apart on names or order.
fn aux_legend_channel_pairs<'a>(bindings: &LegendBindings<'a>) -> [(&'static str, Option<&'a EncodingSpec>); 7] {
    [
        (AUX_LEGEND_CHANNEL_NAMES[0], bindings.size),
        (AUX_LEGEND_CHANNEL_NAMES[1], bindings.shape),
        (AUX_LEGEND_CHANNEL_NAMES[2], bindings.stroke_dash),
        (AUX_LEGEND_CHANNEL_NAMES[3], bindings.opacity),
        (AUX_LEGEND_CHANNEL_NAMES[4], bindings.fill_opacity),
        (AUX_LEGEND_CHANNEL_NAMES[5], bindings.stroke_opacity),
        (AUX_LEGEND_CHANNEL_NAMES[6], bindings.stroke_width),
    ]
}

/// `ferrum._core.aux_legend_channels()`: [`AUX_LEGEND_CHANNEL_NAMES`], the
/// array [`aux_legend_channel_pairs`] reads, so Python's list and the Rust
/// validator are one source (#123). Pinned by
/// `aux_legend_channels_pyfunction_matches_aux_legend_channel_pairs_names`.
/// Registered in `lib.rs`, re-exported through
/// `crate::render::prepare::aux_legend_channels` (this module, `legend`, is
/// private), typed in `src/ferrum/_core.pyi`.
#[pyfunction]
pub(crate) fn aux_legend_channels() -> Vec<String> {
    AUX_LEGEND_CHANNEL_NAMES.iter().map(|s| s.to_string()).collect()
}

/// Read whether a channel's typed `legend` spec suppresses that channel's aux
/// legend block — `legend=None` / `legend=False` from the Python `Size`/`Shape`
/// classes, or `orient="none"`. An aux block answers for ONE channel, so this
/// is the one-element case of the same [`LegendStyleSpec::suppressed_by`] the
/// color legend reads its whole cascade through: the two spellings cannot come
/// to mean different things on different channels.
fn legend_channel_disabled(enc: Option<&crate::spec::encoding::EncodingSpec>) -> bool {
    enc.and_then(|e| e.legend.as_deref())
        .is_some_and(|l| LegendStyleSpec::suppressed_by(&[l]))
}

/// Aux legend title, three-state per spec §6 (#90):
/// `legend.title` > channel `title` > field name, each rung resolved through
/// [`resolve_axis_title`] (present+blank → suppress with no further
/// fallback, present+non-blank → verbatim, absent → fall through). Mirrors
/// the color legend's own precedence above and `build_axis_input`'s
/// spec-title/field-title cascade in `prepare/mod.rs`.
fn aux_legend_title(enc: &crate::spec::encoding::EncodingSpec) -> Option<String> {
    let channel_fallback = resolve_axis_title(enc.title.as_deref(), Some(enc.field.clone()));
    let legend_title = enc.legend.as_ref().and_then(|l| l.title.as_deref());
    resolve_axis_title(legend_title, channel_fallback)
}

/// The label a numeric aux legend (size, the opacity family, stroke width)
/// gives one of its values: the channel's own `Legend(format=,
/// format_type=)`, with an undeclared `%` pattern read as a date when the
/// channel's field is temporal.
fn aux_value_labeler<'e>(enc: &'e EncodingSpec, transformed: &RecordBatch) -> impl Fn(f64) -> String + 'e {
    let legend = enc.legend.as_deref();
    let format_spec = legend.and_then(|l| l.format.as_deref());
    let format_type = legend.and_then(|l| l.format_type.as_deref());
    let temporality = super::field_temporality(enc, transformed).unwrap_or(Temporality::NonTemporal);
    move |v| crate::render::format::format_value_with_spec(v, format_spec, format_type, temporality)
}

/// The title a channel STATED, as opposed to the one it falls back to.
///
/// [`aux_legend_title`] answers "what does this channel's own block say",
/// which for a silent channel is its field name. This answers the narrower
/// question a folded channel poses (#111, spec §4.6): did the user say
/// anything about this legend's title at all? The field name is a default, and
/// a default must never displace or contradict another channel's statement, so
/// the field-name rung is deliberately absent here.
fn explicit_channel_title(enc: &crate::spec::encoding::EncodingSpec) -> Option<String> {
    resolve_axis_title(
        enc.legend.as_ref().and_then(|l| l.title.as_deref()),
        resolve_axis_title(enc.title.as_deref(), None),
    )
}

/// What a same-field channel hands to the color legend when it is folded in
/// rather than given its own block (#111, spec §4.6).
///
/// Keyed by LABEL, never by position: the color domain and the shape/dash
/// domain are both built by `sorted_categorical_domain` over the same field
/// but with that channel's OWN `sort=` bindings, so `stroke_dash=StrokeDash('g',
/// sort='descending')` beside `color='g'` yields two correct domains in
/// opposite orders. Pairing them by index would transpose every pattern
/// silently.
#[derive(Debug, Default)]
struct FoldedChannels {
    /// Category label → the point-glyph name its `shape` channel assigned.
    shapes: std::collections::HashMap<String, String>,
    /// Category label → the dasharray its `stroke_dash` channel assigned. The
    /// solid slot is present with an EMPTY pattern: the entry still has to
    /// become a line swatch, it just draws undashed.
    dashes: std::collections::HashMap<String, Vec<f64>>,
    /// Explicit titles the folded channels stated, in channel order.
    titles: Vec<String>,
}

impl FoldedChannels {
    /// Nothing to give the color ENTRIES. [`titles`](Self::titles) is
    /// deliberately not consulted: a folded channel whose domain came back
    /// empty still stated a title, and the title cascade reads that field
    /// directly rather than through this gate.
    fn has_no_entry_identity(&self) -> bool {
        self.shapes.is_empty() && self.dashes.is_empty()
    }
}

/// What the color legend already shows — the two facts that decide how a
/// same-field aux channel is treated.
#[derive(Debug, Clone, Copy)]
struct ColorLegendState {
    /// The color scale was judged inert on line/ribbon (spec §4.0), so a
    /// merged size legend must not sample swatch colors from it.
    inert_on_line_or_ribbon: bool,
    /// The color legend renders categorical ENTRIES enumerating its field. A
    /// same-field shape/dash block would duplicate that enumeration, so it
    /// folds into it instead. When this is `false` there is no enumeration to
    /// duplicate — `Color(…, legend=None)`, or a colorbar — and the aux block
    /// renders on its own (spec §4.6).
    enumerates_field: bool,
}

/// The aux blocks to render, plus whatever was folded into the color legend.
struct AuxLegends {
    blocks: Vec<AuxLegendInput>,
    folded: FoldedChannels,
}

/// Give each color entry the identity of the same-field channels folded into
/// it, matched by label (#111).
///
/// A dash-carrying entry also becomes a LINE swatch: `dash_pattern` is only
/// read on the line arm of the renderer's swatch match, so a circle carrying a
/// dash renders byte-identically to one without — the whole reason the issue's
/// own prescribed "thread the pattern through" fix measured as a no-op. Set on
/// the entry (not forced in the renderer) so a user `symbol_type=` override
/// still wins, exactly as it does over any other entry symbol.
///
/// # When BOTH `shape` and `stroke_dash` fold onto one field
///
/// One swatch cannot be a glyph and a dash pattern at once, and the two
/// channels here describe different marks anyway (the glyph is what a point
/// draws, the dash what a line draws). The GLYPH wins, for the reason every
/// other choice in this task was made: it is what a standalone `shape` aux
/// block already draws, so a folded legend agrees with an unfolded one rather
/// than drifting from it. A dash threaded onto a glyph would not help — the
/// legend's glyphs are painted `fill`-only (`stroke: None`, `stroke_width:
/// 0.0`), so a dasharray on them paints nothing at all on `circle`/`square`
/// and merely frays a 5px cross arm on the stroke-drawn glyphs.
///
/// The losing channel's pattern is therefore NOT written onto the entry. It is
/// not that the renderer happens to ignore it: an entry that carried a dash it
/// cannot draw would serialize `dash_pattern` into the scene for a swatch
/// nothing paints, leaving the SVG and WASM walkers free to disagree about it.
/// The entry carries exactly what it draws, and the precedence lives in the
/// data model rather than in the renderer's branch order.
///
/// Silence is the one thing this must not do. An unshowable *title* already
/// reports through [`RenderWarning::LegendTitleConflict`](crate::render::RenderWarning::LegendTitleConflict);
/// an unshowable *identity* reports the same way, or the two halves of one
/// merge disagree about whether losing information is worth mentioning.
fn fold_identity_into_color_entries(
    entries: &mut [LegendEntry],
    folded: &FoldedChannels,
    warnings: &mut Vec<crate::render::RenderWarning>,
) {
    if folded.has_no_entry_identity() {
        return;
    }
    // Both maps are built by `sorted_categorical_domain` over the SAME field —
    // that is the precondition for folding at all — so their label sets are
    // equal and a collision on one entry is a collision on every entry. That
    // makes this a property of the CHART, reported once, rather than one
    // warning per category (the rollup discipline `LegendValuesUnknown` and
    // `AnnotationSuppressedByBreak` already follow).
    let glyph_beats_dash = !folded.shapes.is_empty() && !folded.dashes.is_empty();
    if glyph_beats_dash {
        warnings.push(crate::render::RenderWarning::LegendFoldNotShown {
            kept: "shape".to_string(),
            dropped: "stroke_dash".to_string(),
        });
    }
    for entry in entries.iter_mut() {
        if let Some(shape_name) = folded.shapes.get(&entry.label) {
            entry.shape_name = Some(shape_name.clone());
        }
        if glyph_beats_dash {
            continue;
        }
        if let Some(dash) = folded.dashes.get(&entry.label) {
            entry.symbol = SymbolKind::Line;
            entry.dash_pattern = (!dash.is_empty()).then(|| dash.clone());
        }
    }
}

/// What the color channel SAID about the title of the legend surface it owns
/// — as distinct from what that title resolves to (#111, spec §4.6).
enum ColorTitle {
    /// A non-blank title at either rung (`Color(title=)` or
    /// `Color(legend=Legend(title=))`). It wins outright.
    Stated(String),
    /// A blank title, which is this codebase's suppression spelling. It wins
    /// too: "no title" is a statement, and a folded channel does not overturn
    /// it.
    Suppressed,
    /// Nothing. The title is the field name, a default — which is the one
    /// case a folded channel's own title fills in.
    Unstated,
}

/// Read [`ColorTitle`] off the color encoding.
fn color_title_state(enc: &crate::spec::encoding::EncodingSpec) -> ColorTitle {
    let stated = enc
        .legend
        .as_ref()
        .and_then(|l| l.title.as_deref())
        .or(enc.title.as_deref());
    match stated {
        None => ColorTitle::Unstated,
        Some(t) if t.trim().is_empty() => ColorTitle::Suppressed,
        Some(t) => ColorTitle::Stated(t.to_string()),
    }
}

/// Resolve the title of a legend that now answers for more than one channel
/// (#111, spec §4.6).
///
/// `color_default` is the title the color channel resolves to on its own — the
/// value this function returns whenever nothing was folded in, or whenever the
/// color channel stated a title of its own, so every chart outside the merge
/// is untouched. Only where color said NOTHING does a folded channel's
/// explicit title fill in, because a field-name default must not outrank a
/// user's statement; that silent discard is the shipped behavior this closes.
/// There is one title slot, so any folded title that disagrees with the winner
/// cannot be shown, and it is named in a warning rather than vanishing.
fn resolve_shared_legend_title(
    color: ColorTitle,
    color_default: Option<String>,
    folded_titles: &[String],
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> Option<String> {
    if folded_titles.is_empty() {
        return color_default;
    }
    let kept = match &color {
        ColorTitle::Stated(t) => Some(t.clone()),
        ColorTitle::Suppressed => None,
        ColorTitle::Unstated => folded_titles.first().cloned(),
    };
    let discarded: Vec<String> = folded_titles
        .iter()
        .filter(|t| Some(*t) != kept.as_ref())
        .cloned()
        .collect();
    if !discarded.is_empty() {
        warnings.push(crate::render::RenderWarning::LegendTitleConflict {
            kept: kept.clone(),
            discarded,
        });
    }
    match color {
        ColorTitle::Unstated => kept.or(color_default),
        // A stated title resolves through `color_default` exactly as it did
        // before this merge existed: the `Color(title=)` rung IS that value,
        // and the `Legend(title=)` rung outranks it downstream in
        // `config_apply::effective_legend_title`.
        ColorTitle::Stated(_) | ColorTitle::Suppressed => color_default,
    }
}

/// Whether the size channel shares its field with a Numeric-keyed
/// (`Continuous`/`Discretizing`) color encoding — the same-field merge
/// FIELD/SCALE-KIND condition, computed once so [`build_aux_legends`]
/// (whether to sample `color_hex` into the swatches) and its caller (whether
/// the merge is even a *candidate* to null the colorbar) read one answer
/// rather than recomputing it independently and risking drift.
///
/// This is necessary but not SUFFICIENT for "a merged legend was actually
/// emitted" — the caller must additionally confirm `build_aux_legends`
/// actually pushed a `Size` block (size could still be disabled, unscaled,
/// or empty-domained; see the cycle-4 finding on the colorbar-nulling call
/// site). Independent of whether the color scale later turns out inert on
/// line/ribbon — that only gates the SWATCH color, not whether a merge
/// condition holds at all (spec-review 2026-08-28, cycle-3 finding).
fn same_field_numeric_size_color_merge(bindings: &LegendBindings<'_>, scales: &ResolvedScales) -> bool {
    use super::super::scale_resolve::{ColorInput, ColorScale};
    let color_field = bindings.color.map(|c| c.field.as_str());
    let color_is_numeric =
        scales.color.as_ref().map(ColorScale::input) == Some(ColorInput::Numeric);
    let same_field_as_color = bindings.size
        .is_some_and(|size_enc| color_field == Some(size_enc.field.as_str()));
    same_field_as_color && color_is_numeric
}

/// Build the auxiliary legend blocks: size, shape, stroke-dash, then one per
/// bound opacity-family channel, then stroke-width.
///
/// Size: graduated symbols at ~5 nice round values spanning the size domain
/// (`nice_ticks`), each scaled to the size scale's pixel radius and labeled
/// with the value. Shape: one glyph per category in the shape scale.
/// Stroke-dash (T12): one dashed-line swatch per category in the
/// [`StrokeDashScale`](crate::render::scale_resolve::StrokeDashScale), beside
/// shape's block, entry-style like shape rather than graduated like size.
///
/// Same-field merge (Vega-Lite behavior): when the size channel shares its
/// field with a *continuous* color encoding, the size legend's symbols also
/// carry the color the shared field maps to (`color_hex`), and the colorbar is
/// expected to be suppressed by the caller — a single combined block.
///
/// A size, shape, or stroke-dash channel that shares its field with a color
/// legend ALREADY ENUMERATING that field emits no block of its own, because
/// the color legend labels every one of its categories. Shape and stroke-dash
/// then hand their glyph/pattern back through [`AuxLegends::folded`] so those
/// color swatches show what the marks actually draw (#111, spec §4.6); size's
/// graduated radii have no per-entry slot on a categorical color legend and
/// are genuinely dropped, unchanged. Where the color legend enumerates
/// NOTHING — `Color(…, legend=None)`, or a colorbar — there is no duplication
/// to avoid, so the aux block renders on its own.
///
/// `state.inert_on_line_or_ribbon` (spec-review 2026-08-28, cycle-3
/// finding): when `true`, the same-field merge still emits the size legend
/// (its own semantics on line/ribbon are a separate follow-up, out of this
/// wave — size is NOT suppressed) but never samples `color_hex` from the
/// color scale the warning already judged inert: painting graduated swatches
/// from a scale line/ribbon doesn't honor would repeat the exact misleading
/// promise spec §4.0 suppresses the colorbar for. Swatches fall back to the
/// neutral/default color, i.e. a plain size legend, byte-identical to the
/// no-color-merge case.
///
/// `size_symbol` is the swatch the size block draws ([`size_legend_symbol`]),
/// `None` when no layer draws size and so no size block is emitted.
///
/// Opacity family (`opacity`, `fill_opacity`, `stroke_opacity`): one block
/// per bound channel some layer draws, as Vega-Lite draws an opacity legend:
/// a circle swatch at each nice value of the scale's domain, drawn at the
/// alpha that value maps to (a `stroke_opacity` swatch fades its stroke). A
/// constant column's scale has a one-value domain, so its block has one entry.
///
/// Stroke width: line swatches (the stroke-dash block's sample shape) drawn
/// at the width each nice domain value maps to, as ggplot2 draws a linewidth
/// legend.
///
/// `drawn` says which of these channels some layer's mark draws
/// ([`DrawnAuxChannels`]); a binding no mark draws gets no block.
fn build_aux_legends(
    bindings: &LegendBindings<'_>,
    transformed: &RecordBatch,
    scales: &ResolvedScales,
    state: ColorLegendState,
    size_symbol: Option<SymbolKind>,
    drawn: DrawnAuxChannels,
) -> AuxLegends {
    use crate::scale::ticks::nice_ticks;

    use super::super::scale_resolve::{ColorInput, ColorScale};

    let color_field = bindings.color.map(|c| c.field.as_str());
    // Numeric color (continuous or discretizing): the size legend can sample it
    // per value via `lookup_f64`, so a shared field merges into one block.
    // Categorical color enumerates the field instead, so the aux legend is
    // suppressed instead.
    let color_is_numeric =
        scales.color.as_ref().map(ColorScale::input) == Some(ColorInput::Numeric);
    // The full field+scale-kind merge condition, read from the shared
    // predicate (S1 dedup, spec-review cycle-4) rather than recomputed here —
    // `same_field_as_color` still needs to stand alone below for the
    // categorical-suppression branch, which asks a different question
    // (same field but NOT numeric).
    let numeric_merge = same_field_numeric_size_color_merge(bindings, scales);

    let mut out = Vec::new();
    let mut folded = FoldedChannels::default();
    let [(_, size_enc), (_, shape_enc), (_, dash_enc), (_, opacity_enc), (_, fill_opacity_enc), (_, stroke_opacity_enc), (_, width_enc)] =
        aux_legend_channel_pairs(bindings);

    // ── Size legend ──────────────────────────────────────────────────────
    if let (Some(size_scale), Some(size_enc), Some(size_symbol)) = (&scales.size, size_enc, size_symbol) {
        let disabled = legend_channel_disabled(Some(size_enc));
        let same_field_as_color = color_field == Some(size_enc.field.as_str());
        // Merge into a categorical color legend → suppress (color labels it).
        //
        // Size keeps the SCALE-KIND condition rather than shape/dash's
        // "does the color legend enumerate this field" one (#111 scoped
        // itself to those two channels): a graduated radius has no per-entry
        // slot on a categorical color legend to fold into, so there is
        // nothing for the generalized gate to preserve here. The residual —
        // `Color(…, legend=None)` plus a same-field categorical size, which
        // still erases both — is reachable only for a size scale over a
        // string field, which `size=` refuses outright, and is logged rather
        // than changed here.
        let suppressed = disabled || (same_field_as_color && !color_is_numeric);
        if !suppressed {
            let label = aux_value_labeler(size_enc, transformed);
            if let Some((lo, hi)) = size_scale.inner.data_domain() {
                let values = nice_ticks(lo, hi, 5);
                let entries: Vec<SizeLegendEntry> = values
                    .iter()
                    .filter_map(|&v| {
                        // The size scale maps a value to a mark *area* (square
                        // pixels); the point mark draws radius = sqrt(area/π).
                        // Match that exactly so legend symbols equal the marks.
                        let area = size_scale.inner.to_pixel_f64(v)?;
                        let radius = (area / std::f64::consts::PI).sqrt();
                        if !(radius.is_finite() && radius > 0.0) {
                            return None;
                        }
                        // Merge color+size on the same continuous field: sample
                        // the color scale at the value so the symbol varies in
                        // both radius and color — UNLESS that color scale was
                        // judged inert on line/ribbon (spec-review cycle-3),
                        // in which case swatches stay the neutral default.
                        let color_hex = if numeric_merge && !state.inert_on_line_or_ribbon {
                            scales
                                .color
                                .as_ref()
                                .and_then(|c| c.lookup_f64(v))
                                .map(crate::render::color::fmt_svg)
                        } else {
                            None
                        };
                        Some(SizeLegendEntry {
                            label: label(v),
                            radius,
                            color_hex,
                        })
                    })
                    .collect();
                if !entries.is_empty() {
                    out.push(AuxLegendInput::Size {
                        title: aux_legend_title(size_enc),
                        entries,
                        symbol: size_symbol,
                    });
                }
            }
        }
    }

    // ── Shape legend ─────────────────────────────────────────────────────
    // Shape always maps a categorical field; where color enumerates the same
    // field, the shape glyphs fold into that enumeration instead of repeating
    // it (#111).
    if let (Some(shape_scale), Some(shape_enc)) = (&scales.shape, shape_enc) {
        match categorical_aux_fate(shape_enc, color_field, state) {
            AuxBlockFate::Drop => {}
            AuxBlockFate::FoldIntoColor => {
                folded.shapes.extend(
                    shape_scale
                        .domain
                        .iter()
                        .zip(shape_scale.shapes.iter())
                        .map(|(label, kind)| (label.clone(), kind.name().to_string())),
                );
                folded.titles.extend(explicit_channel_title(shape_enc));
            }
            AuxBlockFate::Emit => {
                let entries: Vec<ShapeLegendEntry> = shape_scale
                    .domain
                    .iter()
                    .zip(shape_scale.shapes.iter())
                    .map(|(label, kind)| ShapeLegendEntry {
                        label: label.clone(),
                        shape_name: kind.name().to_string(),
                    })
                    .collect();
                if !entries.is_empty() {
                    out.push(AuxLegendInput::Shape {
                        title: aux_legend_title(shape_enc),
                        entries,
                    });
                }
            }
        }
    }

    // ── Stroke-dash legend (T12) ─────────────────────────────────────────
    // Stroke-dash always maps a categorical field (a quantitative field
    // resolves no scale and stays index-based, see StrokeDashScale's own doc),
    // so it takes shape's fate rule unchanged — the two are deliberate
    // mirrors.
    if let (Some(dash_scale), Some(dash_enc)) = (&scales.stroke_dash, dash_enc) {
        match categorical_aux_fate(dash_enc, color_field, state) {
            AuxBlockFate::Drop => {}
            AuxBlockFate::FoldIntoColor => {
                folded.dashes.extend(
                    dash_scale
                        .domain
                        .iter()
                        .zip(dash_scale.patterns.iter())
                        .map(|(label, pattern)| (label.clone(), pattern.clone())),
                );
                folded.titles.extend(explicit_channel_title(dash_enc));
            }
            AuxBlockFate::Emit => {
                let entries: Vec<StrokeDashLegendEntry> = dash_scale
                    .domain
                    .iter()
                    .zip(dash_scale.patterns.iter())
                    .map(|(label, pattern)| StrokeDashLegendEntry {
                        label: label.clone(),
                        dash: pattern.clone(),
                    })
                    .collect();
                if !entries.is_empty() {
                    out.push(AuxLegendInput::StrokeDash {
                        title: aux_legend_title(dash_enc),
                        entries,
                    });
                }
            }
        }
    }

    // ── Opacity-family legends ───────────────────────────────────────────
    use crate::layout::OpacityPaint;
    let opacity_channels = [
        (opacity_enc, &scales.opacity, drawn.opacity, OpacityPaint::Fill),
        (fill_opacity_enc, &scales.fill_opacity, drawn.fill_opacity, OpacityPaint::Fill),
        (stroke_opacity_enc, &scales.stroke_opacity, drawn.stroke_opacity, OpacityPaint::Stroke),
    ];
    for (enc, scale, channel_drawn, paint) in opacity_channels {
        let (Some(enc), Some(scale), true) = (enc, scale, channel_drawn) else { continue };
        if legend_channel_disabled(Some(enc)) {
            continue;
        }
        let Some((lo, hi)) = scale.inner.data_domain() else { continue };
        let label = aux_value_labeler(enc, transformed);
        let entries: Vec<OpacityLegendEntry> = nice_ticks(lo, hi, 5)
            .into_iter()
            .filter_map(|v| Some(OpacityLegendEntry { label: label(v), opacity: scale.inner.to_pixel_f64(v)? }))
            .collect();
        if !entries.is_empty() {
            out.push(AuxLegendInput::Opacity { title: aux_legend_title(enc), entries, paint });
        }
    }

    // ── Stroke-width legend ──────────────────────────────────────────────
    let width_block = match (&scales.stroke_width, width_enc) {
        (Some(scale), Some(enc)) if drawn.stroke_width && !legend_channel_disabled(Some(enc)) => {
            scale.inner.data_domain().map(|domain| (scale, enc, domain))
        }
        _ => None,
    };
    if let Some((scale, enc, (lo, hi))) = width_block {
        let label = aux_value_labeler(enc, transformed);
        let entries: Vec<StrokeWidthLegendEntry> = nice_ticks(lo, hi, 5)
            .into_iter()
            .filter_map(|v| Some(StrokeWidthLegendEntry { label: label(v), width: scale.width(v)? }))
            .collect();
        if !entries.is_empty() {
            out.push(AuxLegendInput::StrokeWidth { title: aux_legend_title(enc), entries });
        }
    }

    AuxLegends { blocks: out, folded }
}

/// Which of the continuous auxiliary channels without a symbol of their own
/// some layer's mark draws: the gate on their legend blocks, so a legend
/// never states a mapping no mark paints. Each mirrors the size legend's
/// [`super::LayerPrepared::drawn_size_field`] question.
#[derive(Debug, Clone, Copy, Default)]
struct DrawnAuxChannels {
    opacity: bool,
    fill_opacity: bool,
    stroke_opacity: bool,
    stroke_width: bool,
}

impl DrawnAuxChannels {
    /// The answer over `layers` (every member's, for an overlay group).
    fn of(layers: &[super::LayerPrepared]) -> Self {
        use crate::render::scale_resolve::OpacityChannel;
        let opacity = |channel| layers.iter().any(|l| l.drawn_opacity_field(channel).is_some());
        Self {
            opacity: opacity(OpacityChannel::Opacity),
            fill_opacity: opacity(OpacityChannel::Fill),
            stroke_opacity: opacity(OpacityChannel::Stroke),
            stroke_width: layers.iter().any(|l| l.drawn_stroke_width_field().is_some()),
        }
    }
}

/// What becomes of a categorical aux channel's block — the one gate the shape
/// and stroke-dash arms of [`build_aux_legends`] share, so their deliberate
/// mirroring cannot drift on the condition that matters most (#111).
enum AuxBlockFate {
    /// Emit the channel's own block.
    Emit,
    /// Hand the channel's per-category identity to the color legend, which
    /// already enumerates this field, and emit no block.
    FoldIntoColor,
    /// The user turned this channel's legend off (`legend=None` /
    /// `orient="none"`). Nothing is emitted and nothing is folded: opting out
    /// of a block is not a request to repaint another one.
    Drop,
}

fn categorical_aux_fate(
    enc: &crate::spec::encoding::EncodingSpec,
    color_field: Option<&str>,
    state: ColorLegendState,
) -> AuxBlockFate {
    if legend_channel_disabled(Some(enc)) {
        AuxBlockFate::Drop
    } else if color_field == Some(enc.field.as_str()) && state.enumerates_field {
        AuxBlockFate::FoldIntoColor
    } else {
        AuxBlockFate::Emit
    }
}

/// The mark set of every layer that consumes the resolved color scale, when
/// that scale is `Continuous`/`Discretizing` (`ColorInput::Numeric`) AND
/// every one of those layers draws with `Mark::Line` or `Mark::Ribbon` — the
/// two stroke-continuous marks that cannot render a per-value color today
/// (spec §4.0, 2026-08-28: line/ribbon fail loudly on this combination
/// rather than silently rendering a colorbar nothing honors). Returns `None`
/// (no suppression, no warning) in every other case:
/// - the scale is `Categorical`, or absent — line/ribbon's categorical color
///   grouping (NF-A3) is unaffected by this check.
/// - no layer's (post-inheritance) `encoding.color` is bound — nothing
///   consumes the scale from this panel's mark set.
/// - at least one color-consuming layer's mark is something other than
///   line/ribbon (e.g. `point`) — that layer genuinely renders the mapping,
///   so the colorbar stays truthful and the caller must not warn.
/// - a line or ribbon consumer draws under its binned color scale (a
///   `Color(bin=...)` binding under the scale its bin rule cut): it draws one
///   series per bin, so it renders the mapping too ([`draws_numeric_color`]).
///   Under a continuous, quantize, quantile or threshold scale there is no
///   such series split, whatever the binding says, and the line stays inert.
///
/// A "consumer" is any layer whose `encoding.color` resolved to `Some` AND
/// binds the SAME field the resolved `scales.color` scale was built from —
/// field-keyed (spec-review 2026-08-28, cycle-2 finding 2 corrected the
/// composite path this way; cycle-4 finding 2 extends it here so the flat
/// path — which also serves `LayerChart.interactive()` — agrees). The
/// resolved field is the color field of the first layer that draws color:
/// `prepare_render_inputs` feeds the rendering encoding
/// (`PreparedInputs::rendering_encoding()`), whose color that layer
/// establishes, into scale resolution, so that is definitionally the field
/// `scales.color` was resolved from.
/// `LayerPrepared::from_chart_and_layer` clones each layer's own encoding,
/// so per-layer color fields genuinely differ in a merged multi-layer spec
/// (`fm.layer(line(color='v:Q'), point(color='g:N'))`) — without field-keying,
/// the point layer's unrelated `g` binding would count as "consuming" the
/// `v` scale it never reads, wrongly exempting line's inert `v` channel.
///
/// Beyond the field match, a consumer's `encoding.color` may be its own
/// declaration or inherited from the chart level — the same signal every
/// other color consumer in this pipeline reads (`LayerPrepared::color_is_own`'s
/// doc comment: "`encoding.color` itself is ALWAYS fully inherited... so
/// every consumer that reads it directly... sees exactly what main sees").
/// `color_is_own` itself is NOT the right signal here: it answers "did this
/// layer ask for color", not "does this layer's resolved encoding include
/// it" — a layer inheriting color purely from the chart level still paints
/// by it (as long as it's the same field).
///
/// `composite_exempt` short-circuits to `None` regardless of the local mark
/// set: under the composite path, `layers` only ever contains THIS leaf's
/// own marks, so a sibling leaf elsewhere in the same Overlay group (e.g.
/// `fm.layer(line(color=v), point(color=v))`) is invisible to the check
/// above — the composite seam (`render::composite_render::
/// plan_line_ribbon_color_group_exemptions`, threaded down via
/// `LeafScaleContext::color_scale_has_non_line_ribbon_sibling`) already
/// determined that sibling renders the mapping, so this leaf must not warn
/// either (spec-review 2026-08-28 finding). `false` for every standalone
/// (flat/facet) render.
fn inert_numeric_color_on_line_or_ribbon_only(
    scales: &ResolvedScales,
    layers: &[super::LayerPrepared],
    composite_exempt: bool,
) -> Option<Vec<crate::spec::mark::Mark>> {
    use super::super::scale_resolve::{ColorInput, ColorScale};
    use crate::spec::mark::Mark;

    if composite_exempt {
        return None;
    }
    let is_numeric = scales.color.as_ref().map(ColorScale::input) == Some(ColorInput::Numeric);
    if !is_numeric {
        return None;
    }
    // The field the resolved color scale was actually built from — see the
    // doc comment above: the first layer that draws color establishes the
    // scale. With no layer drawing color, `scales.color` could not have
    // resolved (the `is_numeric` check above would have already returned
    // `None`), so this is unreachable as `None` in practice but handled
    // rather than unwrapped.
    let resolved_field = layers.iter().find_map(|l| l.encoding.color.as_ref()).map(|e| e.field.as_str())?;
    let consumers: Vec<(Mark, &EncodingSpec)> = layers
        .iter()
        .filter_map(|l| l.encoding.color.as_ref().filter(|e| e.field == resolved_field).map(|e| (l.mark, e)))
        .collect();
    let scale = scales.color.as_ref();
    if consumers.is_empty() || consumers.iter().any(|&(mark, binding)| draws_numeric_color(mark, binding, scale)) {
        None
    } else {
        Some(consumers.into_iter().map(|(mark, _)| mark).collect())
    }
}

/// Whether a `mark` drawing the numeric (continuous or discretizing) color
/// `binding` through `scale` paints each mark by its value. Every mark does
/// except a line or a ribbon, which paint one color per series: under its
/// binned color scale (a binned `binding` under the scale its bin rule cut,
/// [`crate::render::draw::binned_color_scale`]) their series split by bin,
/// each in its bin's color
/// ([`crate::render::marks::channels::BinnedColorSeries`]), so they draw it
/// too; under any other numeric scale they draw one series and the color has
/// no per-mark effect. Read by the flat inert-color check and the composite
/// overlay group's exemption, so both judge a line as the series split does.
pub(in crate::render) fn draws_numeric_color(
    mark: crate::spec::mark::Mark,
    binding: &EncodingSpec,
    scale: Option<&super::super::scale_resolve::ColorScale>,
) -> bool {
    use crate::spec::mark::Mark;
    !matches!(mark, Mark::Line | Mark::Ribbon) || crate::render::draw::binned_color_scale(Some(binding), scale).is_some()
}

/// The marks that draw the size channel bound to `field`: every layer whose
/// drawn size field ([`super::LayerPrepared::drawn_size_field`]) is `field`,
/// in layer order. A layer that only inherits the field onto a mark that
/// never draws size (a rule beside a line), or a line that yields it to a
/// width of its own (`line(stroke_width=2)` beside a sized rule), is no
/// consumer.
pub(in crate::render) fn size_drawing_marks(
    layers: &[super::LayerPrepared],
    field: &str,
) -> Vec<crate::spec::mark::Mark> {
    layers
        .iter()
        .filter(|l| l.drawn_size_field() == Some(field))
        .map(|l| l.mark)
        .collect()
}

/// The swatch a size legend draws, `None` when no layer draws the size field
/// and so there is no size legend. A line draws its size as a dot-diameter
/// width, so when every layer drawing the size field is a line, the legend
/// draws line swatches at that width; any other consumer draws the size as
/// a dot or a font size, and the legend keeps circles. Consumers are
/// [`size_drawing_marks`]. `sibling_draws_dots` is the
/// compositor's verdict that a sibling leaf of the same overlay group draws
/// the field as dots, which this leaf's own `layers` cannot see.
fn size_legend_symbol(
    bindings: &LegendBindings<'_>,
    layers: &[super::LayerPrepared],
    sibling_draws_dots: bool,
) -> Option<SymbolKind> {
    use crate::spec::mark::Mark;
    let size_enc = bindings.size?;
    let consumers = size_drawing_marks(layers, &size_enc.field);
    if consumers.is_empty() {
        None
    } else if !sibling_draws_dots && consumers.iter().all(|m| *m == Mark::Line) {
        Some(SymbolKind::Line)
    } else {
        Some(SymbolKind::Circle)
    }
}

/// The `linearGradient` stops of a continuous colorbar, drawn from the same
/// mapping the marks use so the bar shows a value's cell color at that
/// value's position in the scale's `transform` space (linear, or symlog,
/// where a position's label reads `transform.value_at`).
///
/// Eleven evenly spaced positions keep the gradient smooth without bloating
/// the SVG. With no midpoint each position `t` samples the scheme at `t`
/// directly, which is what the marks' linear normalization gives and keeps a
/// degenerate domain sampling the whole ramp. With a midpoint each position
/// `p` is colored at its value `lo + p·(hi − lo)` through the marks' piecewise
/// normalization, and a midpoint strictly inside the domain adds a stop at its
/// own position so the scheme's mid color sits at the midpoint; a midpoint
/// outside the domain yields the cells' one-sided ramp with no extra stop.
///
/// Two positions are the same stop when they render to the same offset
/// ([`colorbar_offset_text`]). A midpoint that renders at the center offset is
/// the plain bar: the no-midpoint stops, byte for byte, however many ulps its
/// position sits from 0.5. A midpoint that renders at another even offset adds
/// no stop; that stop is colored through the midpoint normalization.
///
/// [`colorbar_offset_text`]: crate::render::marks::legend::colorbar_offset_text
fn continuous_colorbar_stops(
    domain: (f64, f64),
    scheme: &crate::render::color::ContinuousScheme,
    midpoint: Option<f64>,
    transform: crate::render::scale_resolve::ColorTransform,
) -> Vec<(f64, String)> {
    use crate::render::color::fmt_svg;
    use crate::render::marks::legend::colorbar_offset_text;
    const N_STOPS: usize = 11;
    let mut positions: Vec<f64> = (0..N_STOPS).map(|i| i as f64 / (N_STOPS - 1) as f64).collect();
    let plain = |positions: Vec<f64>| -> Vec<(f64, String)> {
        positions.into_iter().map(|t| (t, fmt_svg(scheme.sample(t)))).collect()
    };
    let Some(mid) = midpoint else {
        return plain(positions);
    };
    let (lo, hi) = domain;
    if lo < mid && mid < hi {
        let f = |x: f64| transform.forward(x);
        let p_mid = (f(mid) - f(lo)) / (f(hi) - f(lo));
        let p_mid_text = colorbar_offset_text(p_mid);
        if p_mid_text == colorbar_offset_text(0.5) {
            return plain(positions);
        }
        if !positions.iter().any(|&p| colorbar_offset_text(p) == p_mid_text) {
            let at = positions.partition_point(|&p| p < p_mid);
            positions.insert(at, p_mid);
        }
    }
    positions
        .into_iter()
        .map(|p| {
            let t = transform.normalize(domain, midpoint, transform.value_at((lo, hi), p));
            (p, fmt_svg(scheme.sample(t)))
        })
        .collect()
}

/// Format a single colorbar tick value into a short human-readable label.
/// Picks decimal precision from the domain span so that small ranges still
/// show enough digits and large ranges don't waste pixels on noise.
fn format_colorbar_tick(value: f64, lo: f64, hi: f64) -> String {
    let span = (hi - lo).abs();
    let precision: usize = if span == 0.0 || !span.is_finite() {
        2
    } else if span >= 100.0 {
        0
    } else if span >= 10.0 {
        1
    } else if span >= 1.0 {
        2
    } else {
        3
    };
    let s = format!("{:.*}", precision, value);
    // Strip trailing zeros / decimal point when the integer form is exact.
    if s.contains('.') {
        let trimmed = s.trim_end_matches('0').trim_end_matches('.').to_string();
        if trimmed.is_empty() {
            "0".into()
        } else {
            trimmed
        }
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::prepare::LayerPrepared;
    use crate::render::scale_resolve::ColorScale;
    use crate::spec::mark::Mark;
    use ferrum_scene::{ConditionalEncoding, EncodingValue};

    /// Minimal point spec with no encodings, ready to receive conditionals.
    /// The bindings of a chart whose rendering encoding is its own.
    fn bindings(spec: &ChartSpec) -> LegendBindings<'_> {
        LegendBindings::new(spec, &spec.encoding)
    }

    fn empty_spec() -> ChartSpec {
        ChartSpec {
            mark: Mark::Point,
            encoding: Default::default(),
            ..Default::default()
        }
    }

    fn color_field_cond(name: &str) -> ConditionalEncoding {
        ConditionalEncoding {
            selection_name: "sel".into(),
            channel: ChannelName::Color,
            if_selected: EncodingValue::Field { name: name.into() },
            if_not: EncodingValue::Color {
                value: ferrum_scene::Color {
                    r: 200,
                    g: 200,
                    b: 200,
                    a: 255,
                },
            },
        }
    }

    #[test]
    fn resolve_conditional_field_names_collects_color_field() {
        let mut spec = empty_spec();
        spec.conditionals = vec![color_field_cond("cat").into()];
        assert_eq!(
            resolve_conditional_field_names(&bindings(&spec), ChannelName::Color),
            vec!["cat".to_string()]
        );
    }

    #[test]
    fn resolve_conditional_field_names_dedups_first_appearance() {
        let mut spec = empty_spec();
        // Two branches referencing the same field across two conditionals.
        let mut c2 = color_field_cond("cat");
        c2.if_not = EncodingValue::Field { name: "cat".into() };
        spec.conditionals = vec![color_field_cond("cat").into(), c2.into(), color_field_cond("other").into()];
        assert_eq!(
            resolve_conditional_field_names(&bindings(&spec), ChannelName::Color),
            vec!["cat".to_string(), "other".to_string()]
        );
    }

    /// A `ResolvedScales` carrying only the given color scale.
    fn scales_with_color(color: ColorScale) -> ResolvedScales {
        use crate::render::scale_resolve::ScaleKind;
        use crate::scale::linear::LinearScale;
        ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1.0],
                vec![0.0, 1.0],
                false,
                false,
            )),
            y: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1.0],
                vec![0.0, 1.0],
                false,
                false,
            )),
            color: Some(color),
            size: None,
            shape: None,
            opacity: None,
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
            stroke_width: None,
            x2: None,
            y2: None,
            y_slots: Default::default(),
            raster_cells: None,
        }
    }

    /// A spec whose color channel names `field`, so the legend has a title
    /// source and the colorbar arm is reachable.
    fn spec_with_color_field(field: &str) -> ChartSpec {
        let mut spec = empty_spec();
        spec.encoding.color = Some(crate::spec::encoding::EncodingSpec {
            field: field.into(),
            ..Default::default()
        });
        spec
    }

    fn empty_batch() -> RecordBatch {
        RecordBatch::new_empty(std::sync::Arc::new(arrow::datatypes::Schema::empty()))
    }

    // ── NF-B13: the per-channel legend cascade (color > x > y) ────────────

    fn enc_with_legend(field: &str, legend: LegendStyleSpec) -> crate::spec::encoding::EncodingSpec {
        crate::spec::encoding::EncodingSpec {
            field: field.into(),
            legend: Some(Box::new(legend)),
            ..Default::default()
        }
    }

    /// `X(legend=...)` / `Y(legend=...)` reach the same override path
    /// `Color(legend=...)` uses, and color wins where both name a field.
    #[test]
    fn per_channel_legend_specs_cascade_color_then_x_then_y() {
        let mut spec = empty_spec();
        spec.encoding.color = Some(enc_with_legend(
            "c",
            LegendStyleSpec { orient: Some("left".into()), ..Default::default() },
        ));
        spec.encoding.x = Some(enc_with_legend(
            "x",
            LegendStyleSpec {
                orient: Some("top".into()),
                columns: Some(3),
                ..Default::default()
            },
        ));
        spec.encoding.y = Some(enc_with_legend(
            "y",
            LegendStyleSpec {
                columns: Some(9),
                title: Some("from y".into()),
                ..Default::default()
            },
        ));
        let specs = per_channel_legend_specs(&bindings(&spec));
        assert_eq!(specs.len(), 3);
        // Color's orient wins over x's; x's columns wins over y's; y still
        // supplies the field neither of the others named.
        assert_eq!(pick(&specs, |l| l.orient.as_deref()), Some("left"));
        assert_eq!(pick(&specs, |l| l.columns), Some(3));
        assert_eq!(pick(&specs, |l| l.title.as_deref()), Some("from y"));
    }

    /// A channel with no `legend=` contributes nothing (it is skipped, not
    /// treated as an all-`None` spec that could shadow a later channel).
    #[test]
    fn per_channel_legend_specs_skips_channels_without_a_legend() {
        let mut spec = empty_spec();
        spec.encoding.color = Some(crate::spec::encoding::EncodingSpec {
            field: "c".into(),
            ..Default::default()
        });
        spec.encoding.y = Some(enc_with_legend(
            "y",
            LegendStyleSpec { orient: Some("bottom".into()), ..Default::default() },
        ));
        let specs = per_channel_legend_specs(&bindings(&spec));
        assert_eq!(specs.len(), 1);
        assert_eq!(pick(&specs, |l| l.orient.as_deref()), Some("bottom"));
    }

    /// `orient="none"` on a positional channel suppresses the chart's legend
    /// the same way `Color(legend=None)` does — one suppression predicate,
    /// reached through the same cascade.
    #[test]
    fn per_channel_orient_none_on_x_suppresses() {
        let mut spec = empty_spec();
        spec.encoding.x = Some(enc_with_legend(
            "x",
            LegendStyleSpec { orient: Some("none".into()), ..Default::default() },
        ));
        assert!(LegendStyleSpec::suppressed_by(&per_channel_legend_specs(&bindings(&spec))));
    }

    /// The cycle-1 S3: suppression is read at the SAME precedence as every
    /// other field, so a lower-precedence channel cannot blank a legend a
    /// higher-precedence channel explicitly placed. RED before the fix — the
    /// old `any(|l| l.suppresses())` let y's `"none"` win over color's
    /// `"right"`, reading one field at two precedences inside one function.
    #[test]
    fn higher_precedence_placement_beats_lower_precedence_orient_none() {
        let mut spec = empty_spec();
        spec.encoding.color = Some(enc_with_legend(
            "c",
            LegendStyleSpec { orient: Some("right".into()), ..Default::default() },
        ));
        spec.encoding.y = Some(enc_with_legend(
            "y",
            LegendStyleSpec { orient: Some("none".into()), ..Default::default() },
        ));
        let specs = per_channel_legend_specs(&bindings(&spec));
        assert!(
            !LegendStyleSpec::suppressed_by(&specs),
            "color's explicit placement outranks y's suppression",
        );
        // The mirror: with color expressing no opinion on `orient`, y's does win.
        spec.encoding.color = Some(enc_with_legend(
            "c",
            LegendStyleSpec { title: Some("t".into()), ..Default::default() },
        ));
        assert!(LegendStyleSpec::suppressed_by(&per_channel_legend_specs(&bindings(&spec))));
    }

    /// `disabled` and `orient` cascade INDEPENDENTLY, field by field — the
    /// same discipline the `pick`s use. A channel that named only `orient`
    /// expressed no opinion on `disabled`, so a lower-precedence
    /// `legend=None` still suppresses.
    #[test]
    fn disabled_and_orient_cascade_as_independent_fields() {
        let mut spec = empty_spec();
        spec.encoding.color = Some(enc_with_legend(
            "c",
            LegendStyleSpec { orient: Some("right".into()), ..Default::default() },
        ));
        spec.encoding.y = Some(enc_with_legend(
            "y",
            LegendStyleSpec { disabled: Some(true), ..Default::default() },
        ));
        assert!(LegendStyleSpec::suppressed_by(&per_channel_legend_specs(&bindings(&spec))));
    }

    /// The single-channel askers (aux blocks, chart-level) are the
    /// one-element case of the same function, not a second rule.
    #[test]
    fn suppressed_by_on_one_spec_covers_both_spellings() {
        let disabled = LegendStyleSpec { disabled: Some(true), ..Default::default() };
        let orient_none = LegendStyleSpec { orient: Some("none".into()), ..Default::default() };
        let placed = LegendStyleSpec { orient: Some("bottom".into()), ..Default::default() };
        assert!(LegendStyleSpec::suppressed_by(&[&disabled]));
        assert!(LegendStyleSpec::suppressed_by(&[&orient_none]));
        assert!(!LegendStyleSpec::suppressed_by(&[&placed]));
        assert!(!LegendStyleSpec::suppressed_by(&[&LegendStyleSpec::default()]));
        assert!(!LegendStyleSpec::suppressed_by(&[]));
    }

    /// A `LayerPrepared` fixture with the given mark and (optionally bound)
    /// color field — for the T5b line/ribbon inert-color-suppression tests,
    /// which need to construct specific mark sets directly rather than going
    /// through `build_layers`/Python lowering.
    fn layer_with_mark_and_color(mark: Mark, color_field: Option<&str>) -> LayerPrepared {
        LayerPrepared {
            mark,
            encoding: crate::spec::encoding::Encoding {
                color: color_field.map(|f| crate::spec::encoding::EncodingSpec {
                    field: f.into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            mark_style: None,
            data_source: None,
            position: None,
            independent_y: false, coord: None,
            reads_chart_data: true,
            color_is_own: color_field.is_some(),
            size_is_own: false,
            stroke_width_is_own: false,
            dropped_reads: Vec::new(),
            raster_value_field: None,
            grouping_series: crate::render::prepare::SeriesGrouping::default(),
        }
    }

    /// A discretizing color scale renders k flat swatches — two gradient stops
    /// per bucket at its own edges — plus one label per bucket boundary, so the
    /// labels the layout distributes linearly land on the band edges.
    #[test]
    fn discretizing_color_builds_a_k_swatch_colorbar() {
        use crate::render::color::from_rgba;
        use crate::render::scale_resolve::DiscretizedColors;
        let buckets = DiscretizedColors::new(
            vec![0.0, 10.0, 20.0],
            vec![from_rgba(255, 0, 0, 255), from_rgba(0, 0, 255, 255)],
        )
        .unwrap();
        let scales = scales_with_color(ColorScale::Discretizing(buckets));
        let spec = spec_with_color_field("v");
        let layers = vec![LayerPrepared::from_chart_only(&spec)];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);

        assert!(bundle.legend_entries.is_empty(), "buckets render as a colorbar");
        let cb = bundle.colorbar.expect("discretizing color must build a colorbar");
        assert_eq!(
            cb.stops,
            vec![
                (0.0, "#ff0000".to_string()),
                (0.5, "#ff0000".to_string()),
                (0.5, "#0000ff".to_string()),
                (1.0, "#0000ff".to_string()),
            ],
            "each swatch spans its own band with hard stops at both edges"
        );
        assert_eq!(cb.tick_labels, vec!["0", "10", "20"]);
        assert_eq!(cb.domain, None, "bucket boundaries are not an evenly-sampled span");
        assert!(warnings.is_empty(), "a Point-mark consumer must not trigger the line/ribbon suppression");
    }

    /// The continuous colorbar is untouched: 11 interpolated stops and 5 evenly
    /// spaced tick labels across the domain (§7 byte-identity invariant).
    #[test]
    fn continuous_color_still_builds_an_11_stop_gradient() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let scales = scales_with_color(ColorScale::Continuous {
            domain: (0.0, 100.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        let spec = spec_with_color_field("v");
        let layers = vec![LayerPrepared::from_chart_only(&spec)];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        let cb = bundle.colorbar.expect("continuous color must build a colorbar");
        assert_eq!(cb.stops.len(), 11);
        assert_eq!(cb.tick_labels.len(), 5);
        assert_eq!(cb.domain, Some((0.0, 100.0)));
        assert!(warnings.is_empty(), "a Point-mark consumer must not trigger the line/ribbon suppression");
    }

    /// Quality-review S2 fix (2026-09-03): `fm.Legend(format_type=...)` alone
    /// (no explicit `format=`) with a NON-`"time"` value must stay
    /// byte-identical to the untouched default — the over-broad
    /// `format_spec.is_some() || format_type.is_some()` gate this finding
    /// caught diverted a real public parameter
    /// (`ferrum/legend.py`'s `format_type`) away from the range-aware
    /// `format_colorbar_tick` default and into plain `format_numeric`, an
    /// undisclosed behavior change nothing pinned.
    #[test]
    fn continuous_color_format_type_number_alone_stays_byte_identical_to_default() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let scale = || ColorScale::Continuous {
            domain: (0.001, 0.0031),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        };
        let mut spec = spec_with_color_field("v");
        let layers = vec![LayerPrepared::from_chart_only(&spec)];
        let mut warnings = Vec::new();
        let scales_default = scales_with_color(scale());
        let default_bundle =
            build_color_legend(&bindings(&spec), &empty_batch(), &scales_default, scales_default.color.as_ref(), &layers, None, &mut warnings);
        let default_labels = default_bundle.colorbar.expect("must build a colorbar").tick_labels;

        spec.encoding.color.as_mut().unwrap().legend = Some(Box::new(
            crate::render::chart_config::LegendStyleSpec {
                format_type: Some("number".to_string()),
                ..Default::default()
            },
        ));
        let scales_typed = scales_with_color(scale());
        let typed_bundle =
            build_color_legend(&bindings(&spec), &empty_batch(), &scales_typed, scales_typed.color.as_ref(), &layers, None, &mut warnings);
        let typed_labels = typed_bundle.colorbar.expect("must build a colorbar").tick_labels;

        assert_eq!(
            default_labels, typed_labels,
            "format_type=\"number\" alone (no format=) must not change colorbar tick labels"
        );
    }

    /// Control for the fix above: `format_type == "time"` alone DOES need
    /// the new path — it must not fall back to `format_colorbar_tick`'s
    /// range-aware DECIMAL formatting of a raw epoch-ms domain value.
    ///
    /// The field is a real epoch-ms timestamp column, so the fixture crosses
    /// the `#147` legend refusal and is accepted by it (a temporal field)
    /// rather than slipping past it on a column the batch does not carry;
    /// the same spec over a Float64 column is the refused shape.
    #[test]
    fn continuous_color_format_type_time_alone_uses_epoch_formatting() {
        use crate::render::chart_config::ChartConfig;
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        use arrow::array::{Float64Array, TimestampMillisecondArray};
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use std::sync::Arc;
        let (lo, hi) = (1_577_836_800_000_i64, 1_580_515_200_000_i64); // 2020-01-01 .. 2020-02-01
        let scales = scales_with_color(ColorScale::Continuous {
            domain: (lo as f64, hi as f64),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        let mut spec = spec_with_color_field("v");
        spec.encoding.color.as_mut().unwrap().legend = Some(Box::new(
            crate::render::chart_config::LegendStyleSpec {
                format_type: Some("time".to_string()),
                ..Default::default()
            },
        ));
        let batch_of = |data_type: DataType, column: arrow::array::ArrayRef| {
            RecordBatch::try_new(Arc::new(Schema::new(vec![Field::new("v", data_type, false)])), vec![column])
                .unwrap()
        };
        let temporal = batch_of(
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![lo, hi])),
        );
        let numeric = batch_of(DataType::Float64, Arc::new(Float64Array::from(vec![lo as f64, hi as f64])));
        let config = ChartConfig::default();
        assert_eq!(refuse_format_temporality_mismatch_legend_fields(&bindings(&spec), &temporal, &config), Ok(()));
        assert!(
            refuse_format_temporality_mismatch_legend_fields(&bindings(&spec), &numeric, &config).is_err(),
            "negative control: the same spec over a numeric column is refused"
        );
        let layers = vec![LayerPrepared::from_chart_only(&spec)];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &temporal, &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        let labels = bundle.colorbar.expect("must build a colorbar").tick_labels;
        assert!(
            labels.iter().any(|l| l.starts_with("2020-01") || l.starts_with("2020-02")),
            "expected date-shaped labels from the epoch-ms domain, got {labels:?}"
        );
    }

    /// `#147` close (wave 6A T23, user ruling): the colorbar reads an untyped
    /// spec by the color field's temporality, the answer the refusal pass
    /// checks. Over a datetime field `"%Y"` renders years (it used to refuse
    /// as d3) and `".2f"` refuses (it used to format the epoch-ms count);
    /// over a numeric field `".2f"` formats numbers.
    #[test]
    fn colorbar_reads_an_untyped_spec_by_the_color_field_temporality() {
        use crate::render::chart_config::ChartConfig;
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        use arrow::array::{Float64Array, TimestampMillisecondArray};
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use std::sync::Arc;
        let (lo, hi) = (978_307_200_000_i64, 1_104_537_600_000_i64); // 2001-01-01 .. 2005-01-01
        let scales = scales_with_color(ColorScale::Continuous {
            domain: (lo as f64, hi as f64),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        let spec_with = |format: &str| {
            let mut spec = spec_with_color_field("v");
            spec.encoding.color.as_mut().unwrap().legend =
                Some(Box::new(crate::render::chart_config::LegendStyleSpec {
                    format: Some(format.to_string()),
                    ..Default::default()
                }));
            spec
        };
        let batch_of = |data_type: DataType, column: arrow::array::ArrayRef| {
            RecordBatch::try_new(Arc::new(Schema::new(vec![Field::new("v", data_type, false)])), vec![column])
                .unwrap()
        };
        let temporal = batch_of(
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![lo, hi])),
        );
        let numeric = batch_of(DataType::Float64, Arc::new(Float64Array::from(vec![1.0, 2.0])));
        let config = ChartConfig::default();
        let labels = |spec: &ChartSpec, batch: &RecordBatch| {
            let layers = vec![LayerPrepared::from_chart_only(spec)];
            build_color_legend(&bindings(spec), batch, &scales, scales.color.as_ref(), &layers, None, &mut Vec::new())
                .colorbar
                .expect("must build a colorbar")
                .tick_labels
        };

        let year = spec_with("%Y");
        assert_eq!(refuse_format_temporality_mismatch_legend_fields(&bindings(&year), &temporal, &config), Ok(()));
        assert_eq!(labels(&year, &temporal), ["2001", "2002", "2003", "2004", "2005"]);

        let number = spec_with(".2f");
        assert_eq!(
            refuse_format_temporality_mismatch_legend_fields(&bindings(&number), &temporal, &config),
            Err(crate::render::RenderError::NumberFormatOnTemporalScale {
                channel: "color",
                surface: crate::render::FormatSurface::Legend,
                format: ".2f".into(),
                coord_flipped: false,
            })
        );
        assert_eq!(refuse_format_temporality_mismatch_legend_fields(&bindings(&number), &numeric, &config), Ok(()));
        // The fixture's scale domain is the epoch-ms span; over a numeric
        // field those are just numbers, formatted as the d3 spec says.
        assert_eq!(labels(&number, &numeric)[0], "978307200000.00");
    }

    /// `#147` close (wave 6A T22): the shape and stroke-dash legends meet the
    /// same refusal the color and size legends do, because the validator
    /// walks the [`aux_legend_channel_pairs`] list `build_aux_legends` builds
    /// from. A `.2f`/`"time"` pair and a bare `"time"` over a string column
    /// refuse on each; the same spec with that legend disabled (either
    /// spelling) or with the chart legend disabled does not.
    #[test]
    fn shape_and_stroke_dash_legends_refuse_a_time_format_on_a_string_field() {
        use crate::render::chart_config::{ChartConfig, LegendConfigSpec};
        use arrow::array::StringArray;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("g", DataType::Utf8, false)])),
            vec![Arc::new(StringArray::from(vec!["a", "b"]))],
        )
        .unwrap();
        let time_pairs = [(Some(".2f"), "time"), (None, "time")];
        let spec_with = |channel: &str, legend: LegendStyleSpec| {
            let mut spec = empty_spec();
            let enc = Some(enc_with_legend("g", legend));
            match channel {
                "shape" => spec.encoding.shape = enc,
                "stroke_dash" => spec.encoding.stroke_dash = enc,
                other => unreachable!("{other}"),
            }
            spec
        };
        let config = ChartConfig::default();
        for channel in ["shape", "stroke_dash"] {
            for (format, format_type) in time_pairs {
                let pair = LegendStyleSpec {
                    format: format.map(str::to_string),
                    format_type: Some(format_type.to_string()),
                    ..Default::default()
                };
                let err = refuse_format_temporality_mismatch_legend_fields(
                    &bindings(&spec_with(channel, pair.clone())),
                    &batch,
                    &config,
                )
                .expect_err("a time format over a string field must refuse");
                assert!(err.to_string().contains(channel), "{channel}: {err}");

                for disabled in [
                    LegendStyleSpec { disabled: Some(true), ..pair.clone() },
                    LegendStyleSpec { orient: Some("none".to_string()), ..pair.clone() },
                ] {
                    assert_eq!(
                        refuse_format_temporality_mismatch_legend_fields(
                            &bindings(&spec_with(channel, disabled)),
                            &batch,
                            &config,
                        ),
                        Ok(()),
                        "{channel}: a disabled legend is exempt"
                    );
                }
                let chart_off = ChartConfig {
                    legend: Some(LegendConfigSpec {
                        style: LegendStyleSpec { orient: Some("none".to_string()), ..Default::default() },
                    }),
                    ..Default::default()
                };
                assert_eq!(
                    refuse_format_temporality_mismatch_legend_fields(&bindings(&spec_with(channel, pair)), &batch, &chart_off),
                    Ok(()),
                    "{channel}: a chart-level disabled legend is exempt"
                );
            }
        }
    }

    // ── T5b: line/ribbon inert-continuous-color suppression (spec §4.0, 2026-08-28) ──

    /// A flat `mark_line` chart with a Continuous color scale: the colorbar
    /// promises a per-value mapping line cannot render, so it must be
    /// suppressed, and a `RenderWarning` must name the mark + scale kind.
    #[test]
    fn line_only_with_continuous_color_suppresses_colorbar_and_warns() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let scales = scales_with_color(ColorScale::Continuous {
            domain: (0.0, 100.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        let spec = spec_with_color_field("v");
        let layers = vec![layer_with_mark_and_color(Mark::Line, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        assert!(bundle.colorbar.is_none(), "line cannot render continuous color; colorbar must be suppressed");
        assert!(bundle.legend_entries.is_empty());
        assert_eq!(warnings.len(), 1, "exactly one warning must fire");
        match &warnings[0] {
            crate::render::RenderWarning::UnsupportedColorScaleOnMark { marks, scale_kind, suppressed } => {
                assert_eq!(marks, &vec!["line".to_string()]);
                assert_eq!(scale_kind, "continuous");
                assert!(*suppressed, "a colorbar existed here and was suppressed");
            }
            other => panic!("expected UnsupportedColorScaleOnMark, got {other:?}"),
        }
    }

    /// Same hazard, `mark_ribbon`: a Continuous color scale is inert on a
    /// closed band exactly as it is on a stroked line.
    #[test]
    fn ribbon_only_with_continuous_color_suppresses_colorbar_and_warns() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let scales = scales_with_color(ColorScale::Continuous {
            domain: (0.0, 100.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        let spec = spec_with_color_field("v");
        let layers = vec![layer_with_mark_and_color(Mark::Ribbon, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        assert!(bundle.colorbar.is_none(), "ribbon cannot render continuous color; colorbar must be suppressed");
        assert_eq!(warnings.len(), 1);
        match &warnings[0] {
            crate::render::RenderWarning::UnsupportedColorScaleOnMark { marks, scale_kind, suppressed } => {
                assert_eq!(marks, &vec!["ribbon".to_string()]);
                assert_eq!(scale_kind, "continuous");
                assert!(*suppressed, "a colorbar existed here and was suppressed");
            }
            other => panic!("expected UnsupportedColorScaleOnMark, got {other:?}"),
        }
    }

    /// A line or ribbon draws a numeric color only when it is binned (one
    /// series per bin); every other mark draws any numeric color. The flat
    /// check and the composite overlay exemption both read this.
    #[test]
    fn only_a_binned_color_is_drawn_by_a_line_or_ribbon() {
        use crate::render::color::from_rgba;
        use crate::render::scale_resolve::DiscretizedColors;
        let plain = crate::spec::encoding::EncodingSpec { field: "z".into(), ..Default::default() };
        let binned: crate::spec::encoding::EncodingSpec = serde_json::from_value(serde_json::json!(
            {"field": "z", "scale": {"type": "bin-ordinal", "bin": {"bin_count": 2, "nice": false}}}
        ))
        .unwrap();
        let buckets = || {
            DiscretizedColors::new(vec![0.0, 10.0, 20.0], vec![from_rgba(255, 0, 0, 255), from_rgba(0, 0, 255, 255)]).unwrap()
        };
        let bin_cut = ColorScale::Discretizing(buckets().cut_by_bin(true));
        let quantize = ColorScale::Discretizing(buckets());
        let continuous = ColorScale::Continuous {
            domain: (0.0, 20.0),
            scheme: crate::render::color::ContinuousScheme::Named(crate::render::color::NamedContinuous::Viridis),
            midpoint: None,
            transform: crate::render::scale_resolve::ColorTransform::Linear,
        };
        for mark in [Mark::Line, Mark::Ribbon] {
            assert!(draws_numeric_color(mark, &binned, Some(&bin_cut)), "{mark:?} draws a binned color under its bin-cut scale");
            assert!(!draws_numeric_color(mark, &binned, Some(&quantize)), "{mark:?}: a binned binding under a quantize scale");
            assert!(!draws_numeric_color(mark, &binned, Some(&continuous)), "{mark:?}: a binned binding under a continuous scale");
            assert!(!draws_numeric_color(mark, &plain, Some(&bin_cut)), "{mark:?}: an unbinned binding under a bin-cut scale");
        }
        for scale in [&bin_cut, &quantize, &continuous] {
            assert!(draws_numeric_color(Mark::Point, &plain, Some(scale)));
            assert!(draws_numeric_color(Mark::Point, &binned, Some(scale)));
        }
    }

    /// The Discretizing (Quantize/Quantile/Threshold/BinOrdinal) variant is
    /// numeric-keyed exactly like Continuous (`ColorInput::Numeric`) and must
    /// be caught the same way — the spec explicitly covers both.
    #[test]
    fn line_only_with_discretizing_color_suppresses_colorbar_and_warns() {
        use crate::render::color::from_rgba;
        use crate::render::scale_resolve::DiscretizedColors;
        let buckets = DiscretizedColors::new(
            vec![0.0, 10.0, 20.0],
            vec![from_rgba(255, 0, 0, 255), from_rgba(0, 0, 255, 255)],
        )
        .unwrap();
        let scales = scales_with_color(ColorScale::Discretizing(buckets));
        let spec = spec_with_color_field("v");
        let layers = vec![layer_with_mark_and_color(Mark::Line, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        assert!(bundle.colorbar.is_none(), "line cannot render a discretizing color scale either");
        assert_eq!(warnings.len(), 1);
        match &warnings[0] {
            crate::render::RenderWarning::UnsupportedColorScaleOnMark { marks, scale_kind, suppressed } => {
                assert_eq!(marks, &vec!["line".to_string()]);
                assert_eq!(scale_kind, "discretizing");
                assert!(*suppressed, "a colorbar existed here and was suppressed");
            }
            other => panic!("expected UnsupportedColorScaleOnMark, got {other:?}"),
        }
    }

    /// The mixed case: a `line` layer and a `point` layer share the same
    /// Continuous color field. The point layer genuinely renders the
    /// mapping, so the colorbar must stay AND no warning may fire — warning
    /// here would be spurious since the legend is not, in fact, a false
    /// promise for this chart.
    #[test]
    fn mixed_line_and_point_layers_sharing_continuous_color_keeps_colorbar_no_warning() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let scales = scales_with_color(ColorScale::Continuous {
            domain: (0.0, 100.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        let spec = spec_with_color_field("v");
        let layers = vec![
            layer_with_mark_and_color(Mark::Line, Some("v")),
            layer_with_mark_and_color(Mark::Point, Some("v")),
        ];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        let cb = bundle.colorbar.expect("a point layer sharing the scale genuinely renders it; colorbar must stay");
        assert_eq!(cb.stops.len(), 11);
        assert!(warnings.is_empty(), "a mixed line+point chart sharing the scale must not warn spuriously");
    }

    /// Categorical color on line is the NF-A3 grouping path (a separate
    /// fix) and must be completely unaffected by this suppression — it
    /// never resolves `ColorInput::Numeric`, so the gate never engages.
    #[test]
    fn line_only_with_categorical_color_is_unaffected() {
        let scales = scales_with_color(ColorScale::Categorical {
            domain: vec!["a".into(), "b".into(), "c".into()],
            palette: std::borrow::Cow::Owned(vec![]),
        });
        let spec = spec_with_color_field("v");
        let layers = vec![layer_with_mark_and_color(Mark::Line, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        assert_eq!(bundle.legend_entries.len(), 3, "categorical color on line stays byte-identical (NF-A3)");
        assert!(bundle.colorbar.is_none());
        assert!(warnings.is_empty(), "categorical color must never trigger the inert-scale suppression");
    }

    /// Spec-review ruling (2026-08-28): loudness is a property of the
    /// CHANNEL, not of whether a colorbar would render. `Color(v,
    /// legend=None)` on a line never builds a colorbar (`legend_disabled`
    /// short-circuits to `(Vec::new(), None)` before the inert-scale check
    /// even runs), but the channel is still just as inert, so the warning
    /// must still fire.
    #[test]
    fn line_only_with_continuous_color_and_legend_none_still_warns() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let scales = scales_with_color(ColorScale::Continuous {
            domain: (0.0, 100.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        let mut spec = spec_with_color_field("v");
        spec.encoding.color.as_mut().unwrap().legend = Some(Box::new(
            crate::render::chart_config::LegendStyleSpec {
                disabled: Some(true),
                ..Default::default()
            },
        ));
        let layers = vec![layer_with_mark_and_color(Mark::Line, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        assert!(bundle.colorbar.is_none(), "legend=None never built a colorbar in the first place");
        assert!(bundle.legend_entries.is_empty());
        assert_eq!(warnings.len(), 1, "the inert channel must still warn even with no legend to suppress");
        match &warnings[0] {
            crate::render::RenderWarning::UnsupportedColorScaleOnMark { marks, scale_kind, suppressed } => {
                assert_eq!(marks, &vec!["line".to_string()]);
                assert_eq!(scale_kind, "continuous");
                assert!(!*suppressed, "no colorbar ever existed here; the message must not claim one was suppressed");
                let text = format!("{}", warnings[0]);
                assert!(!text.contains("suppressed"), "{text}");
            }
            other => panic!("expected UnsupportedColorScaleOnMark, got {other:?}"),
        }
    }

    /// Spec-review ruling (2026-08-28): the same-field color+size merge
    /// nulls `colorbar` for a legitimately different reason (its content
    /// moves into the size legend's colored swatches — unchanged this
    /// round), but the color CHANNEL on the line mark is still inert, so
    /// the warning must still fire alongside that merge.
    #[test]
    fn line_only_with_continuous_color_and_size_merge_still_warns() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        use crate::scale::linear::LinearScale;
        use crate::render::scale_resolve::{ScaleKind, SizeScale};
        let mut scales = scales_with_color(ColorScale::Continuous {
            domain: (0.0, 100.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        // Size shares the SAME field ("v") as color, on a genuine numeric
        // scale — the same-field merge condition `build_aux_legends` checks.
        scales.size = Some(SizeScale {
            inner: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 100.0],
                vec![10.0, 400.0],
                false,
                false,
            )),
        });
        let mut spec = spec_with_color_field("v");
        spec.encoding.size = Some(crate::spec::encoding::EncodingSpec {
            field: "v".into(),
            ..Default::default()
        });
        let mut line = layer_with_mark_and_color(Mark::Line, Some("v"));
        line.encoding.size = spec.encoding.size.clone();
        let layers = vec![line];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        // Sanity: the merge premise actually held (colorbar nulled by the
        // merge, not the inert-scale check testing nothing).
        assert!(bundle.colorbar.is_none());
        assert_eq!(warnings.len(), 1, "the inert channel must still warn even though the merge already nulled the colorbar");
        match &warnings[0] {
            crate::render::RenderWarning::UnsupportedColorScaleOnMark { marks, scale_kind, suppressed } => {
                assert_eq!(marks, &vec!["line".to_string()]);
                assert_eq!(scale_kind, "continuous");
                assert!(!*suppressed, "the merge, not this check, already nulled the colorbar; the message must not double-claim suppression");
                let text = format!("{}", warnings[0]);
                assert!(!text.contains("suppressed"), "{text}");
            }
            other => panic!("expected UnsupportedColorScaleOnMark, got {other:?}"),
        }
        // Spec-review 2026-08-28 (cycle-3 finding): the size legend must
        // still render (size is NOT suppressed — its own line/ribbon
        // semantics are a separate follow-up), but every swatch must fall
        // back to the neutral default — none may carry a graduated color
        // sampled from the scale the warning just judged inert.
        let (size_entries, symbol) = bundle.aux_legends.iter().find_map(|a| match a {
            AuxLegendInput::Size { entries, symbol, .. } => Some((entries, *symbol)),
            _ => None,
        }).expect("the size legend must still render, unaffected by the color channel's own suppression");
        assert!(!size_entries.is_empty(), "the size legend must have swatches to check");
        assert!(
            size_entries.iter().all(|e| e.color_hex.is_none()),
            "every swatch must be the neutral default, not a graduated color from the inert scale: {size_entries:?}"
        );
        assert_eq!(symbol, SymbolKind::Line, "the line is the size legend's only consumer, so it draws line swatches");
    }

    // ── Size legend swatch: lines iff every size consumer is a line ──────

    /// A spec and scales sizing by `v`, with no color.
    fn size_only_fixture() -> (ResolvedScales, ChartSpec) {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        use crate::render::scale_resolve::{ScaleKind, SizeScale};
        use crate::scale::linear::LinearScale;
        let mut scales = scales_with_color(ColorScale::Continuous {
            domain: (0.0, 1.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        scales.color = None;
        scales.size = Some(SizeScale {
            inner: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 100.0], vec![10.0, 400.0], false, false)),
        });
        let mut spec = empty_spec();
        spec.encoding.size = Some(crate::spec::encoding::EncodingSpec { field: "v".into(), ..Default::default() });
        (scales, spec)
    }

    /// A layer carrying the fixture's size field as an inherited binding.
    fn inheriting_size(mark: Mark) -> LayerPrepared {
        let mut l = layer_with_mark_and_color(mark, None);
        l.encoding.size = size_only_fixture().1.encoding.size;
        l
    }

    /// A line carrying the fixture's size field as an inherited binding
    /// beside its own constant `stroke_width=2`, which outranks it.
    fn line_yielding_size() -> LayerPrepared {
        let mut l = inheriting_size(Mark::Line);
        l.mark_style = Some(crate::spec::mark_style::MarkKwargsSpec { stroke_width: Some(2.0), ..Default::default() });
        l
    }

    /// The size block `layers` build, given whether a composite sibling draws
    /// the field as dots: its swatch and entry radii, `None` without a block.
    fn size_block_of(layers: &[LayerPrepared], sibling_draws_dots: bool) -> Option<(SymbolKind, Vec<f64>)> {
        let (scales, spec) = size_only_fixture();
        let ctx = LeafScaleContext { size_scale_has_non_line_sibling: sibling_draws_dots, ..Default::default() };
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), layers, Some(&ctx), &mut Vec::new());
        let mut blocks = bundle.aux_legends.iter().filter_map(|a| match a {
            AuxLegendInput::Size { entries, symbol, .. } => Some((*symbol, entries.iter().map(|e| e.radius).collect())),
            _ => None,
        });
        let block = blocks.next();
        assert!(blocks.next().is_none(), "at most one size block");
        block
    }

    /// A size-field layer the matrix places: a mark that draws an inherited
    /// size, or a line whose own `stroke_width` outranks the size it inherits.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum SizeLayer {
        Draws(Mark),
        LineYielding,
    }

    impl SizeLayer {
        fn prepared(self) -> LayerPrepared {
            match self {
                SizeLayer::Draws(mark) => inheriting_size(mark),
                SizeLayer::LineYielding => line_yielding_size(),
            }
        }
    }

    /// Every set of size layers drawn from {line, point, a line whose own
    /// `stroke_width` outranks the size}, the empty set included, alone or
    /// beside each mark that never draws `size` (every mark
    /// `draws_size_channel` rejects), alone or beside a composite sibling
    /// drawing dots. The drawers are the line and point layers only: with no
    /// drawer there is no size block, else the block draws line swatches
    /// exactly when every drawer is a line and no sibling draws dots. So a
    /// lone rule, area, bar, tick or rect layer with an encoded `size`, or a
    /// `stroke_width=2` line beside one, has no size legend. The entries'
    /// radii never depend on the swatch.
    #[test]
    fn size_legend_symbol_matrix_over_consumers_and_composite_siblings() {
        let non_drawing: Vec<Mark> =
            crate::spec::mark::all_marks().filter(|m| !m.draws_size_channel()).collect();
        assert!(non_drawing.contains(&Mark::Rule), "test sanity: the population is derived, not empty");
        let kinds = [SizeLayer::Draws(Mark::Line), SizeLayer::Draws(Mark::Point), SizeLayer::LineYielding];
        let layer_sets: Vec<Vec<SizeLayer>> = (0..1u32 << kinds.len())
            .map(|bits| kinds.iter().enumerate().filter(|(i, _)| bits & (1 << i) != 0).map(|(_, k)| *k).collect())
            .collect();
        let (_, reference_radii) = size_block_of(&[inheriting_size(Mark::Point)], false).expect("a point draws size");
        let mut outcomes: Vec<Option<SymbolKind>> = Vec::new();
        for set in &layer_sets {
            for filler in std::iter::once(None).chain(non_drawing.iter().copied().map(Some)) {
                let layers: Vec<LayerPrepared> =
                    set.iter().map(|k| k.prepared()).chain(filler.map(inheriting_size)).collect();
                if layers.is_empty() {
                    continue;
                }
                let drawers: Vec<Mark> = set
                    .iter()
                    .filter_map(|k| match k {
                        SizeLayer::Draws(m) => Some(*m),
                        SizeLayer::LineYielding => None,
                    })
                    .collect();
                for sibling_draws_dots in [false, true] {
                    let block = size_block_of(&layers, sibling_draws_dots);
                    let want = if drawers.is_empty() {
                        None
                    } else if drawers.iter().all(|m| *m == Mark::Line) && !sibling_draws_dots {
                        Some(SymbolKind::Line)
                    } else {
                        Some(SymbolKind::Circle)
                    };
                    let cell = format!("layers {set:?} + {filler:?}, sibling draws dots: {sibling_draws_dots}");
                    assert_eq!(block.as_ref().map(|b| b.0), want, "{cell}");
                    if let Some((_, radii)) = &block {
                        assert_eq!(radii, &reference_radii, "the swatch never changes an entry's radius: {cell}");
                    }
                    if !outcomes.contains(&want) {
                        outcomes.push(want);
                    }
                }
            }
        }
        // Vacuity control: every outcome occurs.
        assert_eq!(outcomes.len(), 3, "{outcomes:?}");
    }

    /// A layer that does not draw the size field is not a consumer: a point
    /// beside the sized line without the field, or a rule inheriting it, draws
    /// no size, so the line decides alone. A text drawing it as a font size
    /// is a consumer and keeps the circles.
    #[test]
    fn a_layer_that_does_not_draw_the_size_field_does_not_consume_it() {
        let (scales, spec) = size_only_fixture();
        let sized = |mark: Mark| {
            let mut l = layer_with_mark_and_color(mark, None);
            l.encoding.size = spec.encoding.size.clone();
            l
        };
        let symbol_beside_line = |other: LayerPrepared| {
            let layers = [sized(Mark::Line), other];
            let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut Vec::new());
            match &bundle.aux_legends[0] {
                AuxLegendInput::Size { symbol, .. } => *symbol,
                other => panic!("expected a size block, got {other:?}"),
            }
        };
        assert_eq!(symbol_beside_line(layer_with_mark_and_color(Mark::Point, None)), SymbolKind::Line);
        assert_eq!(symbol_beside_line(sized(Mark::Rule)), SymbolKind::Line);
        assert_eq!(symbol_beside_line(sized(Mark::Text)), SymbolKind::Circle);
    }

    /// The same same-field color+size merge on `mark_point` (a mark that
    /// genuinely renders per-value color) must be completely unaffected —
    /// no warning, and the size legend's swatches stay graduated exactly as
    /// before this fix. Byte-identity control for the cycle-3 finding.
    #[test]
    fn point_only_with_continuous_color_and_size_merge_keeps_graduated_swatches() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        use crate::scale::linear::LinearScale;
        use crate::render::scale_resolve::{ScaleKind, SizeScale};
        let mut scales = scales_with_color(ColorScale::Continuous {
            domain: (0.0, 100.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        scales.size = Some(SizeScale {
            inner: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 100.0],
                vec![10.0, 400.0],
                false,
                false,
            )),
        });
        let mut spec = spec_with_color_field("v");
        spec.encoding.size = Some(crate::spec::encoding::EncodingSpec {
            field: "v".into(),
            ..Default::default()
        });
        let mut point = layer_with_mark_and_color(Mark::Point, Some("v"));
        point.encoding.size = spec.encoding.size.clone();
        let layers = vec![point];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        assert!(bundle.colorbar.is_none(), "the merge still folds the colorbar into the size legend");
        assert!(warnings.is_empty(), "point genuinely renders continuous color; nothing is inert here");
        let size_entries = bundle.aux_legends.iter().find_map(|a| match a {
            AuxLegendInput::Size { entries, .. } => Some(entries),
            _ => None,
        }).expect("the size legend must render");
        assert!(!size_entries.is_empty(), "the size legend must have swatches to check");
        assert!(
            size_entries.iter().all(|e| e.color_hex.is_some()),
            "point's swatches must stay graduated — unaffected by the line/ribbon-only fix: {size_entries:?}"
        );
    }

    // ── spec-review 2026-08-28, cycle-4: colorbar-nulling must track ────────
    // ── the merged Size legend ACTUALLY being emitted ────────────────────────

    fn point_color_size_merge_fixture() -> (ResolvedScales, ChartSpec) {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        use crate::scale::linear::LinearScale;
        use crate::render::scale_resolve::{ScaleKind, SizeScale};
        let mut scales = scales_with_color(ColorScale::Continuous {
            domain: (0.0, 100.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        scales.size = Some(SizeScale {
            inner: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 100.0],
                vec![10.0, 400.0],
                false,
                false,
            )),
        });
        let mut spec = spec_with_color_field("v");
        spec.encoding.size = Some(crate::spec::encoding::EncodingSpec {
            field: "v".into(),
            ..Default::default()
        });
        (scales, spec)
    }

    /// The reviewer's exact repro (S3 finding 1): `mark_point().encode(
    /// color='v:Q', size=Size('v', legend=None))` — the size legend is
    /// explicitly disabled, so `build_aux_legends` never emits a `Size`
    /// block despite the field/scale-kind merge condition holding. Nulling
    /// the colorbar anyway would leave the chart with NO color legend at
    /// all. The colorbar must be KEPT.
    #[test]
    fn point_color_size_merge_with_size_legend_disabled_keeps_colorbar() {
        let (scales, mut spec) = point_color_size_merge_fixture();
        spec.encoding.size.as_mut().unwrap().legend = Some(Box::new(
            crate::render::chart_config::LegendStyleSpec {
                disabled: Some(true),
                ..Default::default()
            },
        ));
        let layers = vec![layer_with_mark_and_color(Mark::Point, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        assert!(bundle.colorbar.is_some(), "no Size legend was emitted; the colorbar is the chart's only color legend and must survive");
        assert!(
            !bundle.aux_legends.iter().any(|a| matches!(a, AuxLegendInput::Size { .. })),
            "size=None must not emit a Size block"
        );
        assert!(warnings.is_empty());
    }

    /// Test-quality gap (S2 finding): the size CHANNEL is bound in the spec
    /// but its scale never resolved (`scales.size` stays `None` — e.g. a
    /// resolution failure upstream). `build_aux_legends` cannot emit a Size
    /// block without a resolved scale, so the colorbar must stay too.
    #[test]
    fn point_color_size_merge_with_unresolved_size_scale_keeps_colorbar() {
        let (mut scales, spec) = point_color_size_merge_fixture();
        scales.size = None; // the channel is bound in `spec` but never resolved
        let layers = vec![layer_with_mark_and_color(Mark::Point, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        assert!(bundle.colorbar.is_some(), "no resolved size scale means no Size block; the colorbar must survive");
        assert!(!bundle.aux_legends.iter().any(|a| matches!(a, AuxLegendInput::Size { .. })));
    }

    /// Test-quality gap (S2 finding): the size scale resolves but produces a
    /// zero-area pixel range, so every candidate swatch fails the
    /// `radius > 0.0` filter and `entries` ends up empty — `build_aux_legends`
    /// pushes no `Size` block in that case either (`if !entries.is_empty()`).
    /// The colorbar must stay for the same reason as the other two cases.
    #[test]
    fn point_color_size_merge_with_empty_size_entries_keeps_colorbar() {
        use crate::scale::linear::LinearScale;
        use crate::render::scale_resolve::{ScaleKind, SizeScale};
        let (mut scales, spec) = point_color_size_merge_fixture();
        // Zero-width pixel range: to_pixel_f64 always yields area 0.0, so
        // every candidate radius is 0.0 and gets filtered out.
        scales.size = Some(SizeScale {
            inner: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 100.0],
                vec![0.0, 0.0],
                false,
                false,
            )),
        });
        let layers = vec![layer_with_mark_and_color(Mark::Point, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        assert!(bundle.colorbar.is_some(), "an empty-entries Size legend is not emitted; the colorbar must survive");
        assert!(!bundle.aux_legends.iter().any(|a| matches!(a, AuxLegendInput::Size { .. })));
    }

    // ── spec-review 2026-08-28, cycle-4: the flat-path consumer check ───────
    // ── must be field-keyed, matching the composite path ─────────────────────

    /// The reviewer's exact repro (S3 finding 2), on the FLAT (non-composite)
    /// path — the same path `LayerChart.interactive()` shares:
    /// `line(color='v:Q') + point(color='g:N')` as one chart's `layers`.
    /// The point layer never reads `v`; it must not exempt line's inert `v`
    /// channel just because SOME other layer binds SOME color field.
    #[test]
    fn flat_line_and_point_on_different_fields_still_warns_and_suppresses() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let scales = scales_with_color(ColorScale::Continuous {
            domain: (0.0, 100.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        let spec = spec_with_color_field("v");
        let layers = vec![
            layer_with_mark_and_color(Mark::Line, Some("v")),
            layer_with_mark_and_color(Mark::Point, Some("g")),
        ];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        assert!(bundle.colorbar.is_none(), "line's inert v channel must still lose its colorbar");
        assert_eq!(warnings.len(), 1, "point's unrelated field g must not exempt line's v: {warnings:?}");
        match &warnings[0] {
            crate::render::RenderWarning::UnsupportedColorScaleOnMark { marks, .. } => {
                assert_eq!(marks, &vec!["line".to_string()]);
            }
            other => panic!("expected UnsupportedColorScaleOnMark, got {other:?}"),
        }
    }

    /// Same-field control (paired with the test above): when BOTH layers
    /// genuinely bind the SAME field, the exemption must still fire —
    /// field-keying must not turn into "always warn." Flat-path counterpart
    /// to `mixed_line_and_point_layers_sharing_continuous_color_keeps_colorbar_no_warning`
    /// (which already covers this shape); restated here explicitly as the
    /// paired control for this cycle's field-keying fix.
    #[test]
    fn flat_line_and_point_on_same_field_stays_exempt() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let scales = scales_with_color(ColorScale::Continuous {
            domain: (0.0, 100.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
transform: crate::render::scale_resolve::ColorTransform::Linear,
        });
        let spec = spec_with_color_field("v");
        let layers = vec![
            layer_with_mark_and_color(Mark::Line, Some("v")),
            layer_with_mark_and_color(Mark::Point, Some("v")),
        ];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut warnings);
        assert!(bundle.colorbar.is_some(), "point genuinely renders the shared v mapping; colorbar must stay");
        assert!(warnings.is_empty(), "same-field point sibling must still exempt line");
    }

    #[test]
    fn resolve_conditional_field_names_filters_by_channel() {
        let mut spec = empty_spec();
        let mut size_cond = color_field_cond("sz");
        size_cond.channel = ChannelName::Size;
        spec.conditionals = vec![color_field_cond("cat").into(), size_cond.into()];
        // Color channel sees only "cat"; Size channel sees only "sz".
        assert_eq!(
            resolve_conditional_field_names(&bindings(&spec), ChannelName::Color),
            vec!["cat".to_string()]
        );
        assert_eq!(
            resolve_conditional_field_names(&bindings(&spec), ChannelName::Size),
            vec!["sz".to_string()]
        );
    }

    // ── #90: channel-title three-state resolution + precedence ────────────
    //
    // The Python end-to-end sweep (`tests/test_legend_channel_title.py`)
    // already exercises this through the real render pipeline; these are
    // the in-crate unit-level counterpart the quality review asked for, so
    // the contract fails where it lives rather than only two layers up.

    /// The color legend's title, direct: absent -> field name,
    /// present+non-blank -> verbatim, present+blank -> suppressed with no
    /// fallback. Exercised through `build_color_legend` (the real producer),
    /// with a `Categorical` color scale so `legend_entries` is non-empty and
    /// the title branch is reachable.
    #[test]
    fn legend_title_three_state_resolution() {
        let layers = |spec: &ChartSpec| vec![LayerPrepared::from_chart_only(spec)];
        let scale = || ColorScale::Categorical {
            domain: vec!["a".into(), "b".into()],
            palette: std::borrow::Cow::Owned(vec![]),
        };

        // absent -> field name.
        let spec = spec_with_color_field("g");
        let mut warnings = Vec::new();
        let bundle = build_color_legend(
            &bindings(&spec),
            &empty_batch(),
            &scales_with_color(scale()),
            None,
            &layers(&spec),
            None,
            &mut warnings,
        );
        assert_eq!(bundle.legend_title.as_deref(), Some("g"));

        // present + non-blank -> verbatim.
        let mut spec = spec_with_color_field("g");
        spec.encoding.color.as_mut().unwrap().title = Some("Group".into());
        let mut warnings = Vec::new();
        let bundle = build_color_legend(
            &bindings(&spec),
            &empty_batch(),
            &scales_with_color(scale()),
            None,
            &layers(&spec),
            None,
            &mut warnings,
        );
        assert_eq!(bundle.legend_title.as_deref(), Some("Group"));

        // present + blank (the `title=None` sentinel) -> suppressed, no
        // fallback to the field name.
        let mut spec = spec_with_color_field("g");
        spec.encoding.color.as_mut().unwrap().title = Some(String::new());
        let mut warnings = Vec::new();
        let bundle = build_color_legend(
            &bindings(&spec),
            &empty_batch(),
            &scales_with_color(scale()),
            None,
            &layers(&spec),
            None,
            &mut warnings,
        );
        assert_eq!(bundle.legend_title, None);
    }

    /// `Legend(title=...)` still outranks the channel `title=` on the color
    /// path — `legend_title` (this module's local field-vs-title resolve)
    /// only decides channel-title-vs-field; the outer override is applied
    /// downstream in `config_apply::effective_legend_title`, which this test
    /// does not re-test — it pins that `legend_title` itself does NOT
    /// already fold in `legend.title`, so the two stay layered rather than
    /// one silently absorbing the other.
    #[test]
    fn legend_title_does_not_fold_in_the_legend_style_override() {
        let mut spec = spec_with_color_field("g");
        let color_enc = spec.encoding.color.as_mut().unwrap();
        color_enc.title = Some("ChannelTitle".into());
        color_enc.legend = Some(Box::new(crate::render::chart_config::LegendStyleSpec {
            title: Some("LegendTitle".into()),
            ..Default::default()
        }));
        let layers = vec![LayerPrepared::from_chart_only(&spec)];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(
            &bindings(&spec),
            &empty_batch(),
            &scales_with_color(ColorScale::Categorical {
                domain: vec!["a".into(), "b".into()],
                palette: std::borrow::Cow::Owned(vec![]),
            }),
            None,
            &layers,
            None,
            &mut warnings,
        );
        // `legend_title` resolves the channel title, not the `Legend(title=)`
        // override — the override is layered on later by
        // `config_apply::effective_legend_title`.
        assert_eq!(bundle.legend_title.as_deref(), Some("ChannelTitle"));
    }

    /// `aux_legend_title`'s three-state + `legend.title > channel title >
    /// field` precedence, unit-level and direct (no scale/render plumbing
    /// needed — the function takes only an `EncodingSpec`).
    #[test]
    fn aux_legend_title_three_state_and_precedence() {
        // absent -> field name.
        let enc = crate::spec::encoding::EncodingSpec { field: "sz".into(), ..Default::default() };
        assert_eq!(aux_legend_title(&enc), Some("sz".to_string()));

        // present + non-blank -> verbatim.
        let enc = crate::spec::encoding::EncodingSpec {
            field: "sz".into(),
            title: Some("SizeTitle".into()),
            ..Default::default()
        };
        assert_eq!(aux_legend_title(&enc), Some("SizeTitle".to_string()));

        // present + blank -> suppressed, no fallback to the field name.
        let enc = crate::spec::encoding::EncodingSpec {
            field: "sz".into(),
            title: Some(String::new()),
            ..Default::default()
        };
        assert_eq!(aux_legend_title(&enc), None);

        // `legend.title` outranks the channel `title`.
        let enc = crate::spec::encoding::EncodingSpec {
            field: "sz".into(),
            title: Some("ChannelTitle".into()),
            legend: Some(Box::new(LegendStyleSpec { title: Some("LegendTitle".into()), ..Default::default() })),
            ..Default::default()
        };
        assert_eq!(aux_legend_title(&enc), Some("LegendTitle".to_string()));

        // `legend.title` blank still suppresses, even though the channel
        // title beneath it is non-blank — no fallback past an explicit
        // suppress at the higher rung.
        let enc = crate::spec::encoding::EncodingSpec {
            field: "sz".into(),
            title: Some("ChannelTitle".into()),
            legend: Some(Box::new(LegendStyleSpec { title: Some(String::new()), ..Default::default() })),
            ..Default::default()
        };
        assert_eq!(aux_legend_title(&enc), None);
    }

    /// The merged-legend title cascade (#111, spec §4.6), rung by rung.
    ///
    /// The end-to-end pins live in `tests/test_legend_same_field_swatches.py`;
    /// this covers the two rungs a rendered chart cannot show apart — a color
    /// channel that SUPPRESSED its title versus one that never stated one,
    /// which both render no field-name title but must treat a folded
    /// channel's title differently.
    #[test]
    fn merged_legend_title_prefers_a_statement_over_a_default() {
        let mut warnings = Vec::new();
        let folded = ["Style".to_string()];

        // Color said nothing: its title is the field-name default, which a
        // folded channel's statement outranks. No conflict — one statement.
        assert_eq!(
            resolve_shared_legend_title(
                ColorTitle::Unstated,
                Some("g".into()),
                &folded,
                &mut warnings
            ),
            Some("Style".to_string())
        );
        assert!(warnings.is_empty(), "{warnings:?}");

        // Color stated a title: it owns the surface and wins, and the folded
        // statement that cannot be shown is named.
        assert_eq!(
            resolve_shared_legend_title(
                ColorTitle::Stated("Species".into()),
                Some("Species".into()),
                &folded,
                &mut warnings
            ),
            Some("Species".to_string())
        );
        assert_eq!(
            warnings,
            vec![crate::render::RenderWarning::LegendTitleConflict {
                kept: Some("Species".into()),
                discarded: vec!["Style".into()],
            }]
        );

        // Color asked for NO title. "No title" is a statement too, so the
        // folded one does not resurrect one — and is reported, not dropped.
        warnings.clear();
        assert_eq!(
            resolve_shared_legend_title(ColorTitle::Suppressed, None, &folded, &mut warnings),
            None
        );
        assert_eq!(
            warnings,
            vec![crate::render::RenderWarning::LegendTitleConflict {
                kept: None,
                discarded: vec!["Style".into()],
            }]
        );

        // Nothing folded: the color default stands untouched, whatever it is.
        warnings.clear();
        assert_eq!(
            resolve_shared_legend_title(ColorTitle::Unstated, Some("g".into()), &[], &mut warnings),
            Some("g".to_string())
        );
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// A folded channel's identity lands on the entry with the MATCHING label,
    /// never on the entry at the matching position (#111).
    ///
    /// The two domains here are deliberately in opposite orders, which is what
    /// `sort="descending"` on the folded channel produces. An index-paired
    /// implementation gives `a` the pattern `c` draws.
    #[test]
    fn folded_identity_matches_by_label_not_by_position() {
        let mut entries = vec![
            LegendEntry::new("a", SymbolKind::Circle),
            LegendEntry::new("b", SymbolKind::Circle),
            LegendEntry::new("c", SymbolKind::Circle),
        ];
        let mut folded = FoldedChannels::default();
        // Descending dash domain: c takes the palette's solid slot.
        folded.dashes.insert("c".into(), vec![]);
        folded.dashes.insert("b".into(), vec![6.0, 3.0]);
        folded.dashes.insert("a".into(), vec![2.0, 3.0]);
        let mut warnings = Vec::new();
        fold_identity_into_color_entries(&mut entries, &folded, &mut warnings);

        assert_eq!(entries[0].dash_pattern, Some(vec![2.0, 3.0]));
        assert_eq!(entries[1].dash_pattern, Some(vec![6.0, 3.0]));
        // The solid slot carries no pattern but still becomes a line swatch,
        // so the block reads as one family rather than two circles and a line.
        assert_eq!(entries[2].dash_pattern, None);
        assert!(entries.iter().all(|e| e.symbol == SymbolKind::Line));
        // One channel folded, nothing lost, nothing to report.
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// `color` + `shape` + `stroke_dash` all on one field: the glyph is drawn,
    /// the dash is NOT written onto the entry, and the loss is reported.
    ///
    /// Measured before this pin (`mark_line` with all three on one field): the
    /// marks drew `solid` / `6,3` / `2,3` while the legend's three swatches
    /// carried the glyphs and ZERO dasharrays, with no warning — the renderer's
    /// `shape_name` branch precedes its `symbol_kind` match and its `ShapeStyle`
    /// hardcodes `stroke_dash: None`, so the folded pattern reached the scene
    /// and was discarded there.
    ///
    /// Three things are pinned, and each is a separate decision:
    ///
    /// 1. **The glyph wins.** Not arbitrary: it is what a standalone `shape`
    ///    aux block already draws, so a folded legend agrees with an unfolded
    ///    one instead of drifting from it.
    /// 2. **The loser is not written.** An entry carrying a dash it cannot draw
    ///    would serialize `dash_pattern` into the scene for a swatch nothing
    ///    paints, which is how the SVG and WASM walkers get room to disagree.
    ///    The entry carries exactly what it draws, so the precedence is a
    ///    property of the data model rather than of the renderer's branch order
    ///    — and the `symbol` stays untouched too, rather than being flipped to
    ///    `Line` for a swatch that draws a glyph.
    /// 3. **It is reported.** `#111`'s whole subject is an identity that
    ///    vanishes silently. Fixing that for the swatch while opening it for
    ///    the title's sibling case would leave the two halves of one merge
    ///    disagreeing about whether losing information is worth mentioning.
    #[test]
    fn a_glyph_and_a_dash_folded_onto_one_entry_keep_the_glyph_and_report_the_dash() {
        let mut entries = vec![
            LegendEntry::new("a", SymbolKind::Circle),
            LegendEntry::new("b", SymbolKind::Circle),
        ];
        let mut folded = FoldedChannels::default();
        folded.shapes.insert("a".into(), "circle".into());
        folded.shapes.insert("b".into(), "square".into());
        folded.dashes.insert("a".into(), vec![2.0, 3.0]);
        folded.dashes.insert("b".into(), vec![6.0, 3.0]);
        let mut warnings = Vec::new();
        fold_identity_into_color_entries(&mut entries, &folded, &mut warnings);

        assert_eq!(entries[0].shape_name.as_deref(), Some("circle"));
        assert_eq!(entries[1].shape_name.as_deref(), Some("square"));
        assert_eq!(
            (&entries[0].dash_pattern, &entries[1].dash_pattern),
            (&None, &None),
            "a pattern the swatch cannot draw must not reach the scene"
        );
        assert!(
            entries.iter().all(|e| e.symbol == SymbolKind::Circle),
            "the swatch draws a glyph, so nothing flips it to a line: {entries:?}"
        );
        assert_eq!(
            warnings,
            vec![crate::render::RenderWarning::LegendFoldNotShown {
                kept: "shape".into(),
                dropped: "stroke_dash".into(),
            }],
            "rolled up per chart, not one per category"
        );
        let text = warnings[0].to_string();
        assert!(text.contains("stroke_dash") && text.contains("could not be shown"), "{text}");
        assert!(!text.contains("LegendFoldNotShown"), "Display leaked Debug: {text}");
    }

    // ── Aux legend channel list: Python-visible pyfunction vs the Rust-side
    //    builder (#123) ────

    /// `aux_legend_channels()` (the `#[pyfunction]` published to Python) and
    /// `aux_legend_channel_pairs()` (the Rust-side per-spec builder/validator)
    /// must name the same channels, in the same order — mirroring `scale_accepted_keys`'s own
    /// `scale_accepted_keys_matches_accepted_keys_for_scale_type_for_every_tag`
    /// precedent (`spec/encoding/gate.rs`). Both read `AUX_LEGEND_CHANNEL_NAMES`
    /// directly, so this test is a belt-and-suspenders proof, not the only
    /// thing standing between them.
    #[test]
    fn aux_legend_channels_pyfunction_matches_aux_legend_channel_pairs_names() {
        let spec = empty_spec();
        let pair_names: Vec<String> =
            aux_legend_channel_pairs(&bindings(&spec)).iter().map(|(name, _)| name.to_string()).collect();
        assert_eq!(aux_legend_channels(), pair_names);
        assert_eq!(
            aux_legend_channels(),
            vec!["size", "shape", "stroke_dash", "opacity", "fill_opacity", "stroke_opacity", "stroke_width"]
        );
    }

    // ── Continuous colorbar stops drawn from the marks' color mapping ──────

    fn colorbar_of(scale: ColorScale) -> ColorbarInput {
        let scales = scales_with_color(scale);
        let spec = spec_with_color_field("v");
        let layers = vec![LayerPrepared::from_chart_only(&spec)];
        build_color_legend(&bindings(&spec), &empty_batch(), &scales, scales.color.as_ref(), &layers, None, &mut Vec::new())
            .colorbar
            .expect("a continuous color scale builds a colorbar")
    }

    fn diverging(domain: (f64, f64), midpoint: Option<f64>) -> ColorScale {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        ColorScale::Continuous {
            domain,
            scheme: ContinuousScheme::Named(NamedContinuous::BlueToRed),
            midpoint,
            transform: crate::render::scale_resolve::ColorTransform::Linear,
        }
    }

    /// The mid color of the scheme [`diverging`] builds.
    fn scheme_mid_hex() -> String {
        use crate::render::color::{fmt_svg, ContinuousScheme, NamedContinuous};
        fmt_svg(ContinuousScheme::Named(NamedContinuous::BlueToRed).sample(0.5))
    }

    fn hex_rgb(hex: &str) -> [f64; 3] {
        let c = crate::render::color::parse_color(hex).expect("stop colors are hex");
        [c.red as f64, c.green as f64, c.blue as f64]
    }

    /// No two stops render to the same offset text.
    fn assert_unique_offsets(cb: &ColorbarInput) {
        let texts: Vec<String> = cb.stops.iter().map(|(o, _)| crate::render::marks::legend::colorbar_offset_text(*o)).collect();
        let mut deduped = texts.clone();
        deduped.dedup();
        assert_eq!(deduped, texts, "duplicate offsets: {:?}", cb.stops);
    }

    fn cell_color(scale: &ColorScale, v: f64) -> [f64; 3] {
        let c = scale.lookup_f64(v).expect("continuous scales map numbers");
        [c.red as f64, c.green as f64, c.blue as f64]
    }

    /// A centered heatmap's domain is asymmetric about its center: every stop
    /// of the bar carries exactly the cell color of the value at its linear
    /// position, and the scheme's mid color sits at the midpoint's own stop.
    /// Sampling the scheme at even offsets (ignoring the midpoint) puts the
    /// mid color at the geometric center instead.
    #[test]
    fn diverging_colorbar_matches_its_cells() {
        let (lo, hi) = (-0.42, 1.0);
        let scale = diverging((lo, hi), Some(0.0));
        let cb = colorbar_of(scale.clone());
        let p_mid = (0.0 - lo) / (hi - lo);
        assert_eq!(cb.stops.len(), 12, "11 even stops plus the midpoint's: {:?}", cb.stops);
        assert!(cb.stops.windows(2).all(|w| w[0].0 < w[1].0), "offsets ascend: {:?}", cb.stops);
        assert_unique_offsets(&cb);
        let mid_stop = cb.stops.iter().find(|(o, _)| *o == p_mid).expect("a stop at the midpoint");
        let scheme_mid = scheme_mid_hex();
        assert_eq!(mid_stop.1, scheme_mid);
        for (o, hex) in &cb.stops {
            let v = lo + o * (hi - lo);
            assert_eq!(hex_rgb(hex), cell_color(&scale, v), "stop at offset {o} (value {v})");
        }
    }

    /// The tick labels stay linear in value with a midpoint: the same labels
    /// as the bar without one.
    #[test]
    fn diverging_colorbar_ticks_stay_linear_in_value() {
        let with_mid = colorbar_of(diverging((-0.42, 1.0), Some(0.0)));
        let without = colorbar_of(diverging((-0.42, 1.0), None));
        assert_eq!(with_mid.tick_labels, without.tick_labels);
        assert_eq!(with_mid.domain, Some((-0.42, 1.0)));
    }

    /// No midpoint: 11 stops at `k/10` sampling the scheme at `k/10`,
    /// including a degenerate domain (which still shows the whole ramp, not
    /// the mid color the cells' normalization gives it).
    #[test]
    fn colorbar_without_midpoint_samples_the_scheme_at_even_offsets() {
        use crate::render::color::{fmt_svg, ContinuousScheme, NamedContinuous};
        let scheme = ContinuousScheme::Named(NamedContinuous::BlueToRed);
        let expected: Vec<(f64, String)> =
            (0..11).map(|i| i as f64 / 10.0).map(|t| (t, fmt_svg(scheme.sample(t)))).collect();
        for domain in [(-0.42, 1.0), (5.0, 5.0), (3.0, 1.0)] {
            let cb = colorbar_of(diverging(domain, None));
            assert_unique_offsets(&cb);
            assert_eq!(cb.stops, expected, "domain {domain:?}");
        }
    }

    /// A midpoint at the geometric center is the plain bar, byte for byte,
    /// including centers a few ulps off 0.5 (`(a + b) / 2` is rarely exact).
    #[test]
    fn midpoint_at_geometric_center_matches_the_plain_bar() {
        let exact = [(-2.0, 2.0), (-1.0, 1.0), (0.0, 10.0), (-0.3, 0.7), (2.5, 7.5)];
        let inexact = [(-0.1, 0.5), (0.2, 0.8), (0.1, 0.7), (-0.3, 0.9), (-0.1, 0.7)];
        for (lo, hi) in exact.into_iter().chain(inexact) {
            let plain = colorbar_of(diverging((lo, hi), None));
            let centered = colorbar_of(diverging((lo, hi), Some((lo + hi) / 2.0)));
            assert_unique_offsets(&centered);
            assert_eq!(centered.stops, plain.stops, "domain ({lo}, {hi})");
        }
    }

    /// A midpoint already on an even offset adds no stop; its stop carries
    /// the scheme's mid color.
    #[test]
    fn midpoint_on_an_even_offset_adds_no_stop() {
        let cb = colorbar_of(diverging((-1.0, 4.0), Some(0.0)));
        assert_eq!(cb.stops.len(), 11);
        assert_unique_offsets(&cb);
        let scheme_mid = scheme_mid_hex();
        assert_eq!(cb.stops[2], (0.2, scheme_mid));
    }

    /// A midpoint whose position renders at an even offset, though a few ulps
    /// off it ((-1, 2) mid 0.2 sits at 0.39999999999999997), adds no stop.
    #[test]
    fn midpoint_ulps_off_an_even_offset_adds_no_stop() {
        let cb = colorbar_of(diverging((-1.0, 2.0), Some(0.2)));
        assert_eq!(cb.stops.len(), 11, "{:?}", cb.stops);
        assert_unique_offsets(&cb);
        let at_04 = cb.stops.iter().find(|(o, _)| crate::render::marks::legend::colorbar_offset_text(*o) == "0.4000").expect("a 0.4000 stop");
        assert_eq!(at_04.1, scheme_mid_hex());
    }

    /// A midpoint within the offset precision of an even offset, though far
    /// more than an ulp from it, adds no stop either.
    #[test]
    fn midpoint_within_the_offset_precision_adds_no_stop() {
        let cb = colorbar_of(diverging((0.0, 1.0), Some(0.40003)));
        assert_eq!(cb.stops.len(), 11, "{:?}", cb.stops);
        assert_unique_offsets(&cb);
    }

    /// A genuinely off-center midpoint keeps its own stop.
    #[test]
    fn off_center_midpoint_keeps_its_stop() {
        let cb = colorbar_of(diverging((-1.0, 10.0), Some(3.0)));
        assert_eq!(cb.stops.len(), 12, "{:?}", cb.stops);
        assert_unique_offsets(&cb);
        let mid = cb.stops.iter().find(|(o, _)| crate::render::marks::legend::colorbar_offset_text(*o) == "0.3636").expect("a 0.3636 stop");
        assert_eq!(mid.1, scheme_mid_hex());
    }

    /// A midpoint outside the domain colors the 11 even stops through the
    /// cells' one-sided normalization, with no extra stop.
    #[test]
    fn midpoint_outside_the_domain_shows_the_cells_one_sided_ramp() {
        for (domain, mid) in [((0.1, 0.9), -1.0), ((0.1, 0.9), 2.0), ((0.1, 0.9), 0.1)] {
            let scale = diverging(domain, Some(mid));
            let cb = colorbar_of(scale.clone());
            assert_eq!(cb.stops.len(), 11, "mid {mid}");
            assert_unique_offsets(&cb);
            for (k, (o, hex)) in cb.stops.iter().enumerate() {
                assert_eq!(*o, k as f64 / 10.0);
                let v = domain.0 + o * (domain.1 - domain.0);
                assert_eq!(*hex, crate::render::color::fmt_svg(scale.lookup_f64(v).unwrap()), "mid {mid}, offset {o}");
            }
        }
    }
}
