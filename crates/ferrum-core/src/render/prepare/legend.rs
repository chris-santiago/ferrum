//! Legend + colorbar construction for `prepare_render_inputs`.
//!
//! Builds the color legend (categorical entries / continuous colorbar /
//! conditional-color fallback), the per-channel legend style overrides, the
//! size/shape auxiliary legend blocks, and the legend title — including the
//! same-field color+size merge that suppresses the redundant colorbar. The
//! orchestrator calls [`build_color_legend`] and unpacks the returned bundle
//! straight into `PreparedInputs`.

use crate::layout::{
    AuxLegendInput, ColorbarInput, LegendDirection, LegendEntry, LegendOrient, ShapeLegendEntry,
    SizeLegendEntry, StrokeDashLegendEntry, SymbolKind,
};
use crate::render::chart_config::LegendStyleSpec;
use crate::spec::chart::ChartSpec;
use arrow::record_batch::RecordBatch;
use ferrum_scene::ChannelName;

use super::super::scale_resolve::ResolvedScales;
use super::LegendPreparedOverrides;

/// The full color-legend output produced from the resolved scales + spec.
/// Unpacked straight into `PreparedInputs` by the orchestrator.
pub(crate) struct ColorLegendBundle {
    pub legend_entries: Vec<LegendEntry>,
    pub colorbar: Option<ColorbarInput>,
    pub legend_title: Option<String>,
    pub legend_overrides: LegendPreparedOverrides,
    pub aux_legends: Vec<AuxLegendInput>,
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
fn per_channel_legend_specs(spec: &ChartSpec) -> Vec<&LegendStyleSpec> {
    [
        spec.encoding.color.as_ref(),
        spec.encoding.x.as_ref(),
        spec.encoding.y.as_ref(),
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

/// Walk `spec.conditionals` for the given `channel` and collect the field names
/// referenced by either branch (`if_selected` / `if_not`) as
/// `EncodingValue::Field { name }`. Dedup preserving first-appearance order.
/// Returns an empty `Vec` when no field-driven conditional exists for `channel`.
///
/// Generalizes the former `resolve_conditional_color_domain`: that function is
/// now this for `ChannelName::Color` followed by `distinct_values_in_order` over
/// each field's values (see the `None` arm of [`build_color_legend`]).
pub(crate) fn resolve_conditional_field_names(
    spec: &ChartSpec,
    channel: ChannelName,
) -> Vec<String> {
    use ferrum_scene::EncodingValue;
    let mut seen = std::collections::HashSet::<String>::new();
    let mut out = Vec::<String>::new();

    for cond in &spec.conditionals {
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
fn resolve_conditional_color_domain(spec: &ChartSpec, transformed: &RecordBatch) -> Vec<String> {
    let mut seen = std::collections::HashSet::<String>::new();
    let mut out = Vec::<String>::new();
    for name in resolve_conditional_field_names(spec, ChannelName::Color) {
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

/// Build the full color legend / colorbar bundle from the resolved scales + spec.
///
/// Mirrors the orchestrator's former inline legend block exactly: categorical
/// color → discrete entries; continuous color → colorbar gradient + ticks;
/// no base color but a conditional Color field → conditional-domain entries.
/// Then derives the legend title, extracts the per-channel legend style
/// overrides, builds the size/shape aux legends, and applies the same-field
/// color+size merge (which suppresses the now-redundant colorbar).
///
/// `layers` is this panel's prepared layer list (from `build_layers`, already
/// available at the caller before this is invoked) — the single coherent
/// point where the color-legend decision can see the mark set consuming the
/// resolved color scale, which `provisional_scales` alone cannot (spec §4.0,
/// 2026-08-28: line/ribbon's inert-continuous-color suppression, see
/// [`inert_numeric_color_on_line_or_ribbon_only`]). Under the composite path
/// `layers` only ever sees THIS leaf's own marks; `composite_color_has_non_line_ribbon_sibling`
/// is the composite seam's verdict on whether a sibling leaf ELSEWHERE in the
/// same Overlay group renders the shared scale (spec-review 2026-08-28
/// finding — `render::composite_render::plan_line_ribbon_color_group_exemptions`).
/// `warnings` is the panel's live warnings sink (mirrors `build_axes`'s
/// `&mut scale_warnings` at the same call site).
pub(crate) fn build_color_legend(
    spec: &ChartSpec,
    transformed: &RecordBatch,
    provisional_scales: &ResolvedScales,
    layers: &[super::LayerPrepared],
    composite_color_has_non_line_ribbon_sibling: bool,
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
    // mirror in ``render::mod::prepare_and_layout`` post-hoc clears the SAME
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
    let legend_specs = per_channel_legend_specs(spec);
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

    let (legend_entries, colorbar): (Vec<LegendEntry>, Option<ColorbarInput>) = if legend_disabled {
        (Vec::new(), None)
    } else {
        match &provisional_scales.color {
            Some(ColorScale::Categorical { domain, .. }) => {
                let entries = domain
                    .iter()
                    .map(|v| LegendEntry {
                        label: v.clone(),
                        symbol: SymbolKind::Circle,
                    })
                    .collect();
                (entries, None)
            }
            Some(ColorScale::Continuous { domain, scheme, .. }) => {
                // Sample the scheme at 11 evenly-spaced positions so the
                // gradient looks smooth without bloating the SVG. The
                // renderer emits these as `linearGradient` stops.
                let n_stops = 11;
                let stops: Vec<(f64, String)> = (0..n_stops)
                    .map(|i| {
                        let t = i as f64 / (n_stops - 1) as f64;
                        let color = scheme.sample(t);
                        (t, super::super::color::fmt_svg(color))
                    })
                    .collect();
                // Tick labels: check for explicit tickLabels override from the
                // typed legend spec (e.g. ["Low", "High"] for SHAP beeswarm),
                // else compute 5 ticks across the domain at 0, 0.25, 0.5, 0.75, 1.0.
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
                            let v = lo + t * (hi - lo);
                            if format_spec.is_some() || format_type == Some("time") {
                                crate::render::format::format_value_with_spec(v, format_spec, format_type)
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
                                crate::render::format::format_value_with_spec(v, format_spec, format_type)
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
                let cond_domain = resolve_conditional_color_domain(spec, transformed);
                if cond_domain.is_empty() {
                    (Vec::new(), None)
                } else {
                    let entries = cond_domain
                        .iter()
                        .map(|v| LegendEntry {
                            label: v.clone(),
                            symbol: SymbolKind::Circle,
                        })
                        .collect();
                    (entries, None)
                }
            }
        }
    };

    // Legend title (Themes-T2.5b): default to the color encoding's field name.
    // When entries were built from a conditional color field (no base encoding),
    // derive the title from the first conditional Color Field branch instead.
    let legend_title = if !legend_entries.is_empty() || colorbar.is_some() {
        spec.encoding.color.as_ref().map(|c| c.field.clone()).or_else(|| {
            // Conditional-color case: find the first Color conditional with a
            // Field branch and use its field name as the title.
            use ferrum_scene::EncodingValue;
            spec.conditionals.iter().find_map(|cond| {
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
        })
    } else {
        None
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
        composite_color_has_non_line_ribbon_sibling,
    );

    // Multivariate B1: build size/shape auxiliary legends from the resolved
    // scales. A size/shape channel that shares its field with the color channel
    // is merged into the color legend rather than emitted as a separate block.
    let aux_legends = build_aux_legends(spec, provisional_scales, inert_consumer_marks.is_some());

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
        let merged_color_size = same_field_numeric_size_color_merge(spec, provisional_scales)
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

/// Optional explicit legend title from a channel's typed `legend.title`,
/// falling back to the channel's field name.
fn aux_legend_title(enc: &crate::spec::encoding::EncodingSpec) -> Option<String> {
    let explicit = enc.legend.as_ref().and_then(|l| l.title.clone());
    explicit.or_else(|| Some(enc.field.clone()))
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
fn same_field_numeric_size_color_merge(spec: &ChartSpec, scales: &ResolvedScales) -> bool {
    use super::super::scale_resolve::{ColorInput, ColorScale};
    let color_field = spec.encoding.color.as_ref().map(|c| c.field.as_str());
    let color_is_numeric =
        scales.color.as_ref().map(ColorScale::input) == Some(ColorInput::Numeric);
    let same_field_as_color = spec
        .encoding
        .size
        .as_ref()
        .is_some_and(|size_enc| color_field == Some(size_enc.field.as_str()));
    same_field_as_color && color_is_numeric
}

/// Build the size/shape/stroke-dash auxiliary legend blocks.
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
/// expected to be suppressed by the caller — a single combined block. A size,
/// shape, or stroke-dash channel that shares its field with a categorical
/// color encoding is suppressed entirely (the color legend already labels
/// that field) — stroke-dash mirrors shape's suppression condition exactly,
/// it has no same-field color+dash merge of its own.
///
/// `color_is_inert_on_line_or_ribbon` (spec-review 2026-08-28, cycle-3
/// finding): when `true`, the same-field merge still emits the size legend
/// (its own semantics on line/ribbon are a separate follow-up, out of this
/// wave — size is NOT suppressed) but never samples `color_hex` from the
/// color scale the warning already judged inert: painting graduated swatches
/// from a scale line/ribbon doesn't honor would repeat the exact misleading
/// promise spec §4.0 suppresses the colorbar for. Swatches fall back to the
/// neutral/default color, i.e. a plain size legend, byte-identical to the
/// no-color-merge case.
fn build_aux_legends(
    spec: &ChartSpec,
    scales: &ResolvedScales,
    color_is_inert_on_line_or_ribbon: bool,
) -> Vec<AuxLegendInput> {
    use crate::render::format::format_value_with_spec;
    use crate::scale::ticks::nice_ticks;

    use super::super::scale_resolve::{ColorInput, ColorScale};

    let color_field = spec.encoding.color.as_ref().map(|c| c.field.as_str());
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
    let numeric_merge = same_field_numeric_size_color_merge(spec, scales);

    let mut out = Vec::new();

    // ── Size legend ──────────────────────────────────────────────────────
    if let (Some(size_scale), Some(size_enc)) = (&scales.size, &spec.encoding.size) {
        let disabled = legend_channel_disabled(spec.encoding.size.as_ref());
        let same_field_as_color = color_field == Some(size_enc.field.as_str());
        // Merge into a categorical color legend → suppress (color labels it).
        let suppressed = disabled || (same_field_as_color && !color_is_numeric);
        if !suppressed {
            let format_spec = size_enc.legend.as_ref().and_then(|l| l.format.as_deref());
            // D8 (legend half of format_type threading, Task 4): mirrors the
            // color legend's colorbar handling just above.
            let format_type = size_enc.legend.as_ref().and_then(|l| l.format_type.as_deref());
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
                        let color_hex = if numeric_merge && !color_is_inert_on_line_or_ribbon {
                            scales
                                .color
                                .as_ref()
                                .and_then(|c| c.lookup_f64(v))
                                .map(crate::render::color::fmt_svg)
                        } else {
                            None
                        };
                        Some(SizeLegendEntry {
                            label: format_value_with_spec(v, format_spec, format_type),
                            radius,
                            color_hex,
                        })
                    })
                    .collect();
                if !entries.is_empty() {
                    out.push(AuxLegendInput::Size {
                        title: aux_legend_title(size_enc),
                        entries,
                    });
                }
            }
        }
    }

    // ── Shape legend ─────────────────────────────────────────────────────
    if let (Some(shape_scale), Some(shape_enc)) = (&scales.shape, &spec.encoding.shape) {
        let disabled = legend_channel_disabled(spec.encoding.shape.as_ref());
        let same_field_as_color = color_field == Some(shape_enc.field.as_str());
        // Shape always maps a categorical field; if color encodes the same
        // categorical field, the color legend already enumerates it — suppress.
        let suppressed = disabled || same_field_as_color;
        if !suppressed {
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

    // ── Stroke-dash legend (T12) ─────────────────────────────────────────
    if let (Some(dash_scale), Some(dash_enc)) = (&scales.stroke_dash, &spec.encoding.stroke_dash) {
        let disabled = legend_channel_disabled(spec.encoding.stroke_dash.as_ref());
        let same_field_as_color = color_field == Some(dash_enc.field.as_str());
        // Stroke-dash always maps a categorical field (a quantitative field
        // resolves no scale and stays index-based, see StrokeDashScale's own
        // doc); if color encodes the same categorical field, the color legend
        // already enumerates it — suppress, mirroring shape's condition above.
        let suppressed = disabled || same_field_as_color;
        if !suppressed {
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

    out
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
///
/// A "consumer" is any layer whose `encoding.color` resolved to `Some` AND
/// binds the SAME field the resolved `scales.color` scale was built from —
/// field-keyed (spec-review 2026-08-28, cycle-2 finding 2 corrected the
/// composite path this way; cycle-4 finding 2 extends it here so the flat
/// path — which also serves `LayerChart.interactive()` — agrees). The
/// resolved field is `layers[0]`'s own color field: `prepare_render_inputs`
/// feeds `layers[0].encoding` into scale resolution
/// (`rendering_encoding = layers[0].encoding.clone()`), so that is
/// definitionally the field `scales.color` was resolved from.
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
    // doc comment above. `layers[0]` always exists (`build_layers` never
    // returns an empty `Vec`); if it has no color bound, `scales.color`
    // could not have resolved (the `is_numeric` check above would have
    // already returned `None`), so this is unreachable as `None` in
    // practice but handled defensively rather than indexed/unwrapped.
    let resolved_field = layers.first().and_then(|l| l.encoding.color.as_ref()).map(|e| e.field.as_str())?;
    let consumers: Vec<Mark> = layers
        .iter()
        .filter(|l| l.encoding.color.as_ref().map(|e| e.field.as_str()) == Some(resolved_field))
        .map(|l| l.mark)
        .collect();
    if !consumers.is_empty() && consumers.iter().all(|m| matches!(m, Mark::Line | Mark::Ribbon)) {
        Some(consumers)
    } else {
        None
    }
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
    fn empty_spec() -> ChartSpec {
        ChartSpec {
            data: Default::default(),
            mark: Mark::Point,
            encoding: Default::default(),
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            params: Vec::new(),
            chart_description: None,
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
        spec.conditionals = vec![color_field_cond("cat")];
        assert_eq!(
            resolve_conditional_field_names(&spec, ChannelName::Color),
            vec!["cat".to_string()]
        );
    }

    #[test]
    fn resolve_conditional_field_names_dedups_first_appearance() {
        let mut spec = empty_spec();
        // Two branches referencing the same field across two conditionals.
        let mut c2 = color_field_cond("cat");
        c2.if_not = EncodingValue::Field { name: "cat".into() };
        spec.conditionals = vec![color_field_cond("cat"), c2, color_field_cond("other")];
        assert_eq!(
            resolve_conditional_field_names(&spec, ChannelName::Color),
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
            x2: None,
            y2: None,
            y_slots: Default::default(),
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
        let specs = per_channel_legend_specs(&spec);
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
        let specs = per_channel_legend_specs(&spec);
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
        assert!(LegendStyleSpec::suppressed_by(&per_channel_legend_specs(&spec)));
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
        let specs = per_channel_legend_specs(&spec);
        assert!(
            !LegendStyleSpec::suppressed_by(&specs),
            "color's explicit placement outranks y's suppression",
        );
        // The mirror: with color expressing no opinion on `orient`, y's does win.
        spec.encoding.color = Some(enc_with_legend(
            "c",
            LegendStyleSpec { title: Some("t".into()), ..Default::default() },
        ));
        assert!(LegendStyleSpec::suppressed_by(&per_channel_legend_specs(&spec)));
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
        assert!(LegendStyleSpec::suppressed_by(&per_channel_legend_specs(&spec)));
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
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            independent_y: false,
            color_is_own: color_field.is_some(),
            x_is_own: false,
            y_is_own: false,
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
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);

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
        });
        let spec = spec_with_color_field("v");
        let layers = vec![LayerPrepared::from_chart_only(&spec)];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        };
        let mut spec = spec_with_color_field("v");
        let layers = vec![LayerPrepared::from_chart_only(&spec)];
        let mut warnings = Vec::new();
        let scales_default = scales_with_color(scale());
        let default_bundle =
            build_color_legend(&spec, &empty_batch(), &scales_default, &layers, false, &mut warnings);
        let default_labels = default_bundle.colorbar.expect("must build a colorbar").tick_labels;

        spec.encoding.color.as_mut().unwrap().legend = Some(Box::new(
            crate::render::chart_config::LegendStyleSpec {
                format_type: Some("number".to_string()),
                ..Default::default()
            },
        ));
        let scales_typed = scales_with_color(scale());
        let typed_bundle =
            build_color_legend(&spec, &empty_batch(), &scales_typed, &layers, false, &mut warnings);
        let typed_labels = typed_bundle.colorbar.expect("must build a colorbar").tick_labels;

        assert_eq!(
            default_labels, typed_labels,
            "format_type=\"number\" alone (no format=) must not change colorbar tick labels"
        );
    }

    /// Control for the fix above: `format_type == "time"` alone DOES need
    /// the new path — it must not fall back to `format_colorbar_tick`'s
    /// range-aware DECIMAL formatting of a raw epoch-ms domain value.
    #[test]
    fn continuous_color_format_type_time_alone_uses_epoch_formatting() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let scales = scales_with_color(ColorScale::Continuous {
            domain: (1_577_836_800_000.0, 1_580_515_200_000.0), // 2020-01-01 .. 2020-02-01
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
        });
        let mut spec = spec_with_color_field("v");
        spec.encoding.color.as_mut().unwrap().legend = Some(Box::new(
            crate::render::chart_config::LegendStyleSpec {
                format_type: Some("time".to_string()),
                ..Default::default()
            },
        ));
        let layers = vec![LayerPrepared::from_chart_only(&spec)];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
        let labels = bundle.colorbar.expect("must build a colorbar").tick_labels;
        assert!(
            labels.iter().any(|l| l.starts_with("2020-01") || l.starts_with("2020-02")),
            "expected date-shaped labels from the epoch-ms domain, got {labels:?}"
        );
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
        });
        let spec = spec_with_color_field("v");
        let layers = vec![layer_with_mark_and_color(Mark::Line, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        });
        let spec = spec_with_color_field("v");
        let layers = vec![layer_with_mark_and_color(Mark::Ribbon, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        });
        let spec = spec_with_color_field("v");
        let layers = vec![
            layer_with_mark_and_color(Mark::Line, Some("v")),
            layer_with_mark_and_color(Mark::Point, Some("v")),
        ];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        let layers = vec![layer_with_mark_and_color(Mark::Line, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        let size_entries = bundle.aux_legends.iter().find_map(|a| match a {
            AuxLegendInput::Size { entries, .. } => Some(entries),
            _ => None,
        }).expect("the size legend must still render, unaffected by the color channel's own suppression");
        assert!(!size_entries.is_empty(), "the size legend must have swatches to check");
        assert!(
            size_entries.iter().all(|e| e.color_hex.is_none()),
            "every swatch must be the neutral default, not a graduated color from the inert scale: {size_entries:?}"
        );
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
        let layers = vec![layer_with_mark_and_color(Mark::Point, Some("v"))];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        });
        let spec = spec_with_color_field("v");
        let layers = vec![
            layer_with_mark_and_color(Mark::Line, Some("v")),
            layer_with_mark_and_color(Mark::Point, Some("g")),
        ];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
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
        });
        let spec = spec_with_color_field("v");
        let layers = vec![
            layer_with_mark_and_color(Mark::Line, Some("v")),
            layer_with_mark_and_color(Mark::Point, Some("v")),
        ];
        let mut warnings = Vec::new();
        let bundle = build_color_legend(&spec, &empty_batch(), &scales, &layers, false, &mut warnings);
        assert!(bundle.colorbar.is_some(), "point genuinely renders the shared v mapping; colorbar must stay");
        assert!(warnings.is_empty(), "same-field point sibling must still exempt line");
    }

    #[test]
    fn resolve_conditional_field_names_filters_by_channel() {
        let mut spec = empty_spec();
        let mut size_cond = color_field_cond("sz");
        size_cond.channel = ChannelName::Size;
        spec.conditionals = vec![color_field_cond("cat"), size_cond];
        // Color channel sees only "cat"; Size channel sees only "sz".
        assert_eq!(
            resolve_conditional_field_names(&spec, ChannelName::Color),
            vec!["cat".to_string()]
        );
        assert_eq!(
            resolve_conditional_field_names(&spec, ChannelName::Size),
            vec!["sz".to_string()]
        );
    }
}
