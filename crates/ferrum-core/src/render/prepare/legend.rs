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
    SizeLegendEntry, SymbolKind,
};
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
pub(crate) fn build_color_legend(
    spec: &ChartSpec,
    transformed: &RecordBatch,
    provisional_scales: &ResolvedScales,
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
    let legend_disabled = spec
        .encoding
        .color
        .as_ref()
        .and_then(|c| c.legend.as_ref())
        .and_then(|l| l.disabled)
        .unwrap_or(false);
    // Colorbar label overrides, shared by the continuous (gradient) and the
    // discretizing (k-swatch) arms: an explicit `tickLabels` list replaces the
    // computed labels outright, and `format=` applies a Python-style format spec
    // to each computed value.
    let color_legend = spec.encoding.color.as_ref().and_then(|c| c.legend.as_ref());
    let custom_tick_labels: Option<Vec<String>> = color_legend.and_then(|l| l.tick_labels.clone());
    let format_spec: Option<&str> = color_legend.and_then(|l| l.format.as_deref());

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
                            if let Some(spec_str) = format_spec {
                                crate::render::format::format_with_spec(v, Some(spec_str))
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
                        .map(|&v| match format_spec {
                            Some(s) => crate::render::format::format_with_spec(v, Some(s)),
                            None => format_colorbar_tick(v, lo, hi),
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

    // D13 (B5-typed): extract legend style overrides from the typed
    // `encoding.color.legend` spec. Per-channel precedence: these win over
    // chart-level `configure_legend` (which fills only what is still `None`).
    let color_legend = spec.encoding.color.as_ref().and_then(|c| c.legend.as_ref());
    let legend_overrides = LegendPreparedOverrides {
        orient: color_legend
            .and_then(|l| l.orient.as_deref())
            .and_then(|s| match s {
                "right" => Some(LegendOrient::Right),
                "left" => Some(LegendOrient::Left),
                "top" => Some(LegendOrient::Top),
                "bottom" => Some(LegendOrient::Bottom),
                _ => None,
            }),
        title: color_legend.and_then(|l| l.title.clone()),
        title_font_size: color_legend.and_then(|l| l.title_font_size),
        columns: color_legend.and_then(|l| l.columns),
        tick_count: color_legend.and_then(|l| l.tick_count).map(|n| n as usize),
        label_font_size: color_legend.and_then(|l| l.label_font_size),
        gradient_length: color_legend.and_then(|l| l.gradient_length),
        gradient_thickness: color_legend.and_then(|l| l.gradient_thickness),
        direction: color_legend
            .and_then(|l| l.direction.as_deref())
            .and_then(|s| match s {
                "horizontal" => Some(LegendDirection::Horizontal),
                "vertical" => Some(LegendDirection::Vertical),
                _ => None,
            }),
        // `values`: explicit tick/entry labels for the legend. Accepts an array of
        // strings or numbers. Numbers are formatted to a short decimal string.
        values: color_legend.and_then(|l| l.values.as_ref()).map(|arr| {
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
        legend_type: color_legend.and_then(|l| l.legend_type.clone()),
        symbol_type: color_legend.and_then(|l| l.symbol_type.clone()),
        // B5 unit 3: orphan legend styling/positioning fields. Per-channel here;
        // chart-level `configure_legend(...)` fills any that stay `None`.
        symbol_stroke_width: color_legend.and_then(|l| l.symbol_stroke_width),
        row_padding: color_legend.and_then(|l| l.row_padding),
        column_padding: color_legend.and_then(|l| l.column_padding),
        label_limit: color_legend.and_then(|l| l.label_limit),
        clip_height: color_legend.and_then(|l| l.clip_height),
        tick_min_step: color_legend.and_then(|l| l.tick_min_step),
        zindex: color_legend.and_then(|l| l.zindex),
        // B5 unit 6a orphans. Per-channel here; chart-level `configure_legend`
        // fills any that stay `None`.
        symbol_size: color_legend.and_then(|l| l.symbol_size),
        label_color: color_legend.and_then(|l| l.label_color.clone()),
        offset: color_legend.and_then(|l| l.offset),
        padding: color_legend.and_then(|l| l.padding),
        title_padding: color_legend.and_then(|l| l.title_padding),
    };

    // Multivariate B1: build size/shape auxiliary legends from the resolved
    // scales. A size/shape channel that shares its field with the color channel
    // is merged into the color legend rather than emitted as a separate block.
    let aux_legends = build_aux_legends(spec, provisional_scales);

    // Same-field merge: when color (continuous) and size share a field, the
    // combined block is the size legend whose symbols also carry color
    // (`color_hex`). Suppress the now-redundant colorbar so a single combined
    // legend renders rather than a colorbar plus a size legend.
    let colorbar = {
        let merged_color_size = aux_legends.iter().any(|a| {
            matches!(
                a,
                AuxLegendInput::Size { entries, .. }
                    if entries.iter().any(|e| e.color_hex.is_some())
            )
        });
        if merged_color_size {
            None
        } else {
            colorbar
        }
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

/// Read whether a channel's typed `legend` spec carries `disabled: true`
/// (`legend=None` / `legend=False` from the Python `Size`/`Shape` classes).
fn legend_channel_disabled(enc: Option<&crate::spec::encoding::EncodingSpec>) -> bool {
    enc.and_then(|e| e.legend.as_ref())
        .and_then(|l| l.disabled)
        .unwrap_or(false)
}

/// Optional explicit legend title from a channel's typed `legend.title`,
/// falling back to the channel's field name.
fn aux_legend_title(enc: &crate::spec::encoding::EncodingSpec) -> Option<String> {
    let explicit = enc.legend.as_ref().and_then(|l| l.title.clone());
    explicit.or_else(|| Some(enc.field.clone()))
}

/// Build the size/shape auxiliary legend blocks.
///
/// Size: graduated symbols at ~5 nice round values spanning the size domain
/// (`nice_ticks`), each scaled to the size scale's pixel radius and labeled
/// with the value. Shape: one glyph per category in the shape scale.
///
/// Same-field merge (Vega-Lite behavior): when the size channel shares its
/// field with a *continuous* color encoding, the size legend's symbols also
/// carry the color the shared field maps to (`color_hex`), and the colorbar is
/// expected to be suppressed by the caller — a single combined block. A size or
/// shape channel that shares its field with a categorical color encoding is
/// suppressed entirely (the color legend already labels that field).
fn build_aux_legends(spec: &ChartSpec, scales: &ResolvedScales) -> Vec<AuxLegendInput> {
    use crate::render::format::format_with_spec;
    use crate::scale::ticks::nice_ticks;

    use super::super::scale_resolve::{ColorInput, ColorScale};

    let color_field = spec.encoding.color.as_ref().map(|c| c.field.as_str());
    // Numeric color (continuous or discretizing): the size legend can sample it
    // per value via `lookup_f64`, so a shared field merges into one block.
    // Categorical color enumerates the field itself, so the aux legend is
    // suppressed instead.
    let color_is_numeric =
        scales.color.as_ref().map(ColorScale::input) == Some(ColorInput::Numeric);

    let mut out = Vec::new();

    // ── Size legend ──────────────────────────────────────────────────────
    if let (Some(size_scale), Some(size_enc)) = (&scales.size, &spec.encoding.size) {
        let disabled = legend_channel_disabled(spec.encoding.size.as_ref());
        let same_field_as_color = color_field == Some(size_enc.field.as_str());
        // Merge into a categorical color legend → suppress (color labels it).
        let suppressed = disabled || (same_field_as_color && !color_is_numeric);
        if !suppressed {
            let format_spec = size_enc.legend.as_ref().and_then(|l| l.format.as_deref());
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
                        // both radius and color.
                        let color_hex = if same_field_as_color && color_is_numeric {
                            scales
                                .color
                                .as_ref()
                                .and_then(|c| c.lookup_f64(v))
                                .map(crate::render::color::fmt_svg)
                        } else {
                            None
                        };
                        Some(SizeLegendEntry {
                            label: format_with_spec(v, format_spec),
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

    out
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
        let bundle =
            build_color_legend(&spec_with_color_field("v"), &empty_batch(), &scales);

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
        let bundle =
            build_color_legend(&spec_with_color_field("v"), &empty_batch(), &scales);
        let cb = bundle.colorbar.expect("continuous color must build a colorbar");
        assert_eq!(cb.stops.len(), 11);
        assert_eq!(cb.tick_labels.len(), 5);
        assert_eq!(cb.domain, Some((0.0, 100.0)));
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
