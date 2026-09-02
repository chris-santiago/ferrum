//! Color scale resolution: categorical and continuous color encoding.

use std::borrow::Cow;
use std::collections::HashMap;

use arrow::record_batch::RecordBatch;

use crate::layout::ThemeInputs;
use crate::spec::encoding::DataType as SpecDataType;

use crate::render::color::Color;
use crate::render::color::palette;
use crate::render::RenderError;

use super::domain::{apply_sort_to_domain, locate_field, SortContext};
use super::{distinct_values_in_order, infer_spec_type, numeric_extent, shared_categorical_batch, union_panel_with_global_extent, ColorScale, DiscretizedColors, SharedDomain};

/// Resolve the color encoding into a `ColorScale`.
///
/// Routes to `Continuous` when the field's Arrow dtype is `Float64` or
/// `UInt64`, else `Categorical`. (F16 will widen the numeric detection
/// to consult `EncodingSpec.type_` first and treat all numeric dtypes
/// as continuous; today's narrow check is preserved.)
///
/// Color is the one secondary channel that may live outside the primary
/// batch — composite-mark color fields can resolve via `transform_outputs`
/// — so this function accepts the named-outputs map. Size/shape/opacity
/// builders below are primary-batch-only by design.
///
/// Returns `(scale, warnings)` so the caller can fold warnings into its
/// accumulator alongside build_shape_scale warnings.
///
/// # D1 — explicit string range on categorical scale
///
/// When the color encoding carries `scale={"type": "ordinal", "domain": [...],
/// "range": ["#ccc", "#e4572e"]}`, the resolver builds the palette directly
/// from the string range (parsed via `parse_color`) rather than looking up a
/// named categorical palette.  The categorical domain is taken from the declared
/// `scale.domain` (if present) rather than data first-appearance order, so the
/// range colors always zip against the declared domain positions.  `enc.sort` is
/// applied on top of the declared domain (or data-derived domain when no explicit
/// domain is given), mirroring the positional-scale behavior.
///
/// The parse is all-or-nothing: if any color string fails to parse, the entire
/// explicit range is discarded and the function emits a `ColorRangeParseFailure`
/// warning naming the first offending entry, then falls through to the default
/// theme palette.
///
/// # D4 — scheme inside `scale` dict for continuous color
///
/// When the color encoding carries `scale={"type": "linear", "scheme": "blues"}`,
/// the `scheme` field now lives on `ContinuousScaleCommon` and is honored by
/// the continuous path.  The precedence is:
///   1. `encoding.scheme` (top-level field on `EncodingSpec`)
///   2. `encoding.scale.common.scheme` (inside a continuous scale spec)
///   3. Theme sequential/diverging scheme
///   4. Hard fallback: Viridis
///
/// # FA-5 — `force_categorical` for area marks
///
/// When `force_categorical = true`, the Quantitative/Temporal path is skipped
/// and the color field is always resolved as a `Categorical` scale regardless
/// of its Arrow dtype.  Set this flag for `mark_area`, which always groups rows
/// into discrete per-color bands (via `col_as_ordinal_category_str`) and therefore
/// must use the same categorical palette for both fills and legend swatches.
///
/// Without this flag a Float64/Int64 color column would produce a `Continuous`
/// scale (gradient colorbar legend) while the area fills sampled discrete points
/// on that ramp — legend ≠ fill (the FA-5 bug).
///
/// # T3 — `facet_shared` for continuous color in faceted charts
///
/// When `facet_shared = true` (chart is faceted; color has no independent option),
/// the auto-inferred continuous color domain is unioned with the global
/// `FINAL_OUTPUT_KEY` batch's extent, so per-panel marks normalize through the same
/// domain as the global colorbar. Explicit `scale_explicit_domain` overrides still
/// win. Non-faceted callers pass `false`; the per-panel-only path is byte-identical.
///
/// # 10-pre-b — `composite_domain` for shared color across a composition
///
/// When a composite node shares `color`, the resolve pass
/// ([`crate::render::composite`]) unions one domain across the leaves and threads
/// it here as `composite_domain`. It seeds the auto path, mirroring `facet_shared`
/// but with an explicit (rather than global-batch-derived) domain:
/// - continuous color: a [`SharedDomain::Numeric`] extent replaces the auto data
///   extent (an explicit user `scale.domain` still wins — but such a leaf is
///   excluded from sharing upstream, so the two never collide);
/// - categorical color: a [`SharedDomain::Ordinal`] vector replaces the
///   data-derived first-appearance domain, so every leaf's swatches/legend agree.
///
/// `None` (every standalone and faceted caller) reproduces the pre-10-pre-b path
/// byte-for-byte.
#[allow(clippy::too_many_arguments)]
pub fn build_color_scale(
    encoding: &crate::spec::encoding::Encoding,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    theme: &ThemeInputs,
    force_categorical: bool,
    facet_shared: bool,
    composite_domain: Option<&SharedDomain>,
) -> Result<(Option<ColorScale>, Vec<crate::render::RenderWarning>), RenderError> {
    let Some(c_enc) = &encoding.color else {
        return Ok((None, Vec::new()));
    };
    let located = locate_field(&c_enc.field, primary_batch, transform_outputs)
        .ok_or_else(|| RenderError::UnknownColumn { name: c_enc.field.clone() })?;

    let inferred = infer_spec_type(c_enc, located.col.data_type());
    // FA-5: When force_categorical is set (mark_area), treat any quantitative/temporal
    // color field as categorical.  Area always groups by distinct color values
    // (col_as_ordinal_category_str), so both the fills and the legend must agree on
    // a categorical palette rather than diverging: fills from a continuous ramp vs.
    // legend showing a gradient colorbar.
    let is_continuous_color = !force_categorical && matches!(
        inferred,
        SpecDataType::Quantitative | SpecDataType::Temporal,
    );
    if is_continuous_color {
        if !crate::render::arrow_cast::is_numeric(located.col.data_type()) {
            return Err(RenderError::EncodingTypeMismatch {
                channel: "color",
                expected: "numeric column for quantitative/temporal type",
                got: format!("{:?}", located.col.data_type()),
                // "color" is never a positional channel, so `CoordFlip` never
                // touches it — `user_facing_channel` is identity regardless
                // (R3). `build_color_scale` has no `ChartSpec`/coord in scope
                // here, so this is hardcoded rather than threaded for a value
                // that can never change the rendered message.
                coord_flipped: false,
            });
        }
        // T3: When faceted (Shared), union the per-panel extent with the global
        // FINAL_OUTPUT_KEY batch so marks normalize through the same domain as the
        // global colorbar legend.
        let panel_extent = numeric_extent(located.col);
        let data_extent = if facet_shared {
            union_panel_with_global_extent(panel_extent, &c_enc.field, transform_outputs)
        } else {
            panel_extent
        };
        // Quantize/Quantile/Threshold/BinOrdinal bucket the value instead of
        // interpolating it. Resolved before the continuous path because they
        // reuse the same numeric column and scheme precedence; `None` means the
        // spec is not (or declares no usable) discretization and the continuous
        // path below applies unchanged.
        //
        // T3: a `Quantile` spec with no declared sample derives its cut-points
        // from the data, so it takes the global FINAL_OUTPUT_KEY batch under
        // facet sharing for the same reason the extent above is unioned — per-
        // panel quantiles would paint the same value differently in each panel
        // while the colorbar stayed global.
        let sample_batch =
            shared_categorical_batch(located.batch, &c_enc.field, transform_outputs, facet_shared);
        if let Some((buckets, warnings)) =
            build_discretizing_color_scale(c_enc, sample_batch, data_extent, theme)?
        {
            return Ok((Some(ColorScale::Discretizing(buckets)), warnings));
        }

        // D1: honor explicit domain from the scale spec. When the spec carries
        // an extent (`positional_extent`), use those bounds instead of
        // auto-inferring from the data column. When the spec domain is absent, a
        // 10-pre-b composite shared extent wins over the per-leaf data extent;
        // otherwise fall back to the data extent.
        let (lo, hi) = scale_explicit_domain(c_enc)
            .or_else(|| composite_numeric_extent(composite_domain))
            .unwrap_or(data_extent);

        let (scheme, scheme_warnings) = resolve_continuous_scheme(c_enc, theme, (lo, hi));
        // Gap 2: resolve the diverging midpoint.  Priority:
        //   1. explicit `domain_mid`/`domainMid` field on DivergingScale spec
        //   2. middle element of a 3-tuple domain=[lo, mid, hi]
        //   3. None — geometric center falls out from pure-linear normalization
        // Sequential scales always get None (pure-linear).
        let midpoint = scale_diverging_midpoint(c_enc);
        Ok((Some(ColorScale::Continuous { domain: (lo, hi), scheme, midpoint }), scheme_warnings))
    } else {
        // Every categorical color scale is keyed per row by
        // `col_as_ordinal_category_str`, so the column must be category-readable
        // whatever the domain's source. Two of the three sources never read it —
        // an explicit `scale.domain` (D1, below) and a composite shared domain
        // (10-pre-b) both supersede `distinct_values_in_order` — so without this
        // gate a `Timestamp` color column declared `type="nominal"` with an
        // explicit `domain`+`range` resolved a perfectly well-formed
        // `Categorical` scale over a column nothing could key: `rule`/`segment`
        // refused the chart at mark build while `point`/`bar` painted every
        // element the theme fill under a legend enumerating the declared range.
        // Refusing on the dtype here makes that uniform, loud, and at scale
        // resolution, in the same words `distinct_values_in_order` uses — the
        // default path below is byte-identical, since the gate admits exactly
        // the dtypes that builder does (pinned by
        // `arrow_cast::category_readers_accept_exactly_the_dtypes_ensure_category_keyable_does`).
        crate::render::arrow_cast::ensure_category_keyable(
            &c_enc.field,
            located.col.data_type(),
        )?;

        // T3/categorical: when the chart is faceted (Shared), resolve the domain
        // and sort-context batch from the global FINAL_OUTPUT_KEY batch so that
        // every panel assigns the same palette color to the same category string
        // — matching the global legend.  Falls back to `primary_batch` when the
        // key is absent or the field is missing from the global batch.
        let domain_batch = shared_categorical_batch(primary_batch, &c_enc.field, transform_outputs, facet_shared);

        // Data-aware sort (channel shorthand `"-y"`, sort-field objects) reorders
        // the legend domain by an aggregate, mirroring the positional-axis path.
        // The category column is the color field; candidate value columns live in
        // the primary batch alongside it.
        //
        // When faceted/Shared, use the same global batch for sorting so the
        // sort order matches the global legend, not the per-panel aggregate.
        let sort_ctx = SortContext {
            category_field: &c_enc.field,
            batch: domain_batch,
            x_field: encoding.x.as_ref().map(|e| e.field.as_str()),
            y_field: encoding.y.as_ref().map(|e| e.field.as_str()),
        };
        // D1: if the color encoding carries an explicit ordinal string range, build
        // the palette from those colors (parsed via `parse_color`).
        //
        // Domain resolution order (mirrors positional.rs OrdinalScale behavior):
        //   1. `scale.domain` (declared explicit domain)
        //   2. Data first-appearance order (`distinct_values_in_order`)
        // `enc.sort` is applied on top of the resolved domain.
        //
        // The parse is all-or-nothing: if any entry fails to parse, the entire
        // explicit range is discarded, a `ColorRangeParseFailure` warning is emitted
        // naming the first offending entry, and the resolver falls through to the
        // default theme palette below.
        if let Some(color_strings) = explicit_string_range(c_enc) {
            let mut warnings: Vec<crate::render::RenderWarning> = Vec::new();
            let parse_result: Result<Vec<Color>, String> = color_strings
                .iter()
                .map(|s| {
                    crate::render::color::parse_color(s)
                        .map_err(|_| s.clone())
                })
                .collect::<Result<Vec<_>, _>>();
            match parse_result {
                Ok(parsed) => {
                    // Build the domain from the declared scale.domain (if present),
                    // falling back to data first-appearance order from the
                    // domain_batch (global when facet_shared).
                    let mut domain = match explicit_ordinal_domain(c_enc) {
                        Some(declared) => declared,
                        None => distinct_values_in_order(domain_batch, &c_enc.field)?,
                    };
                    apply_sort_to_domain(&mut domain, c_enc.sort.as_ref(), &sort_ctx, &mut warnings);
                    let palette: Cow<'static, [Color]> = Cow::Owned(parsed);
                    // No overflow warning when the user supplied an explicit range;
                    // they own the mapping and repeated colors are intentional.
                    return Ok((Some(ColorScale::Categorical { domain, palette }), warnings));
                }
                Err(bad_entry) => {
                    warnings.push(crate::render::RenderWarning::ColorRangeParseFailure {
                        entry: bad_entry,
                    });
                    // Fall through to default palette, carrying the warning.
                    let mut domain = distinct_values_in_order(domain_batch, &c_enc.field)?;
                    apply_sort_to_domain(&mut domain, c_enc.sort.as_ref(), &sort_ctx, &mut warnings);
                    let scale = build_default_categorical_scale(domain, c_enc, theme, &mut warnings);
                    return Ok((Some(scale), warnings));
                }
            }
        }

        // Default path: look up a named categorical palette.
        //
        // 10-pre-b: a composite shared categorical domain (unioned across the
        // composition's leaves, first-appearance order) replaces the per-leaf
        // data-derived domain so every leaf's swatch/legend order agrees. `sort`
        // still applies on top, mirroring the facet-shared categorical path. The
        // explicit-string-range branches above are unreachable when a composite
        // domain is present (color is excluded from sharing on an explicit
        // `enc.scale`), so seeding only this default path is sufficient.
        let mut warnings: Vec<crate::render::RenderWarning> = Vec::new();
        let mut domain = match composite_categorical_domain(composite_domain) {
            Some(cats) => cats,
            None => distinct_values_in_order(domain_batch, &c_enc.field)?,
        };
        apply_sort_to_domain(&mut domain, c_enc.sort.as_ref(), &sort_ctx, &mut warnings);
        let scale = build_default_categorical_scale(domain, c_enc, theme, &mut warnings);
        Ok((Some(scale), warnings))
    }
}

/// Resolve the continuous color scheme for an encoding:
///   1. a `Sequential` spec's `stops` ([`gradient_scheme_from_stops`]) — a
///      `Gradient`-backed scheme, F-L04-02 second revision
///   2. `encoding.scheme` (top-level field on `EncodingSpec`, D1/D4)
///   3. the scale spec's own scheme ([`scale_common_scheme`])
///   4. the theme's diverging scheme when `(lo, hi)` straddles zero, else its
///      sequential scheme
///   5. hard fallback: Viridis
///
/// A `Sequential { reverse: true }` spec wraps the result in
/// [`ContinuousScheme::Reverse`] — including a stops-resolved `Gradient`, for
/// robustness against a hand-written spec that sets both fields (every
/// `_to_scale_spec_dict`-emitted spec already composes reverse into the stop
/// order itself and always carries `reverse: false` alongside `stops`, so
/// this is a no-op on that path). Shared by the continuous and the
/// discretizing paths so both honor the same precedence.
///
/// Returns any `RenderWarning`s the resolution produced (currently only a
/// stops all-or-nothing parse failure); the caller folds them into its
/// accumulator.
fn resolve_continuous_scheme(
    enc: &crate::spec::encoding::EncodingSpec,
    theme: &ThemeInputs,
    (lo, hi): (f64, f64),
) -> (crate::render::color::ContinuousScheme, Vec<crate::render::RenderWarning>) {
    use crate::render::color::{ContinuousScheme, NamedContinuous};
    let mut warnings = Vec::new();
    let mut scheme = gradient_scheme_from_stops(enc, &mut warnings).unwrap_or_else(|| {
        enc.scheme
            .as_deref()
            .or(scale_common_scheme(enc))
            .and_then(NamedContinuous::from_name)
            .map(ContinuousScheme::Named)
            .unwrap_or_else(|| {
                let theme_scheme = if lo < 0.0 && hi > 0.0 {
                    &theme.palette.diverging_scheme
                } else {
                    &theme.palette.sequential_scheme
                };
                NamedContinuous::from_name(theme_scheme)
                    .map(ContinuousScheme::Named)
                    .unwrap_or(ContinuousScheme::Named(NamedContinuous::Viridis))
            })
    });
    if scale_spec_is_reversed(enc) {
        scheme = ContinuousScheme::Reverse(Box::new(scheme));
    }
    (scheme, warnings)
}

/// The `stops` list of a `Sequential` scale spec, when present and non-empty.
fn sequential_stops(enc: &crate::spec::encoding::EncodingSpec) -> Option<&[(f64, String)]> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
        ScaleSpec::Sequential { stops: Some(s), .. } if !s.is_empty() => Some(s),
        _ => None,
    }
}

/// Build a `ContinuousScheme::Gradient` from a `Sequential` spec's `stops`
/// (F-L04-02 second revision, spec §4.2 amended 2026-08-28: `Color(scale=
/// fm.Gradient([...]))` renders via this path instead of refusing). `None`
/// when the spec carries no usable stops, leaving the caller to fall through
/// to scheme-name resolution.
///
/// Stops carry the real `t` position each pair declared (spec reviewer
/// cycle 3, finding 1) — parsed via the full-CSS `parse_color` (named
/// colors, not just hex, work in a gradient's stops), positions passed
/// through unchanged, not re-spaced to `i / (n - 1)`.
///
/// The color parse is all-or-nothing, mirroring `build_color_scale`'s
/// `explicit_string_range` / `ColorRangeParseFailure` convention (T2's
/// committed all-or-nothing convention, `render/scale_resolve/color.rs`
/// lines 211-246): one unparseable stop discards the whole list and pushes a
/// `ColorRangeParseFailure` warning naming the first offending entry,
/// falling through to scheme-name resolution rather than rendering a partial
/// or silently-wrong gradient.
fn gradient_scheme_from_stops(
    enc: &crate::spec::encoding::EncodingSpec,
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> Option<crate::render::color::ContinuousScheme> {
    let stops = sequential_stops(enc)?;
    let parsed: Result<Vec<(f64, Color)>, String> = stops
        .iter()
        .map(|(t, s)| {
            crate::render::color::parse_color(s)
                .map(|c| (*t, c))
                .map_err(|_| s.clone())
        })
        .collect();
    match parsed {
        Ok(gradient_stops) if gradient_stops.len() >= 2 => {
            Some(crate::render::color::ContinuousScheme::Gradient(gradient_stops))
        }
        // Fewer than 2 stops can't form a gradient. The `Gradient(...)`
        // pyfunction now rejects this at construction (spec reviewer cycle
        // 3, finding 2 — `_to_scale_spec_dict` only ever emits what a valid
        // `ContinuousScheme::Gradient` carries), so this arm is reachable
        // only from a hand-written-JSON spec that bypasses that constructor;
        // falls through with no warning, matching `bucket_colors`' treatment
        // of a range whose length doesn't fit the partition it can't
        // describe.
        Ok(_) => None,
        Err(entry) => {
            warnings.push(crate::render::RenderWarning::ColorRangeParseFailure { entry });
            None
        }
    }
}

/// The bucket partition a discretizing scale spec declares.
struct BucketPartition {
    /// Ascending interior boundaries; the scale has `thresholds.len() + 1`
    /// buckets.
    thresholds: Vec<f64>,
    /// Ascending `(lo, hi)` labeling extent for the colorbar's end labels.
    extent: (f64, f64),
    /// The spec's declared domain runs high → low, so the resolved colors are
    /// reversed to keep the first `range` entry on the declared `lo` end.
    ///
    /// Spec §4.2 (amended 2026-08-28): a descending `Quantize` domain
    /// `[hi, lo]` normalizes to `[lo, hi]` with the swatch order reversed —
    /// deterministic, and the reading that matches reversed-colormap intent.
    /// This is knowingly asymmetric with the continuous path, whose
    /// `normalize_continuous` collapses `hi <= lo` to a flat 0.5; the asymmetry
    /// is a logged campaign follow-up, not resolved in this batch.
    descending: bool,
}

/// Resolve a `Quantize`/`Quantile`/`Threshold`/`BinOrdinal` color encoding into
/// bucket boundaries plus one flat color per bucket (spec §4.2).
///
/// Returns `Ok(None)` — leaving the caller on the continuous path,
/// byte-identically to the pre-discretizing behavior — for every
/// non-discretizing scale spec and for a discretizing spec that declares no
/// bucket count at all (a `Quantize` or `Quantile` with no `range`, a
/// `Threshold` with no `domain`, a `BinOrdinal` with no `bins`). The Python
/// constructors make all four mandatory, so those are hand-written-JSON shapes.
///
/// Returns `Err` for a `Threshold`/`BinOrdinal` boundary list that is neither
/// ascending nor descending — see [`declared_boundaries`].
///
/// `sample_batch` is the batch a `Quantile` spec reads its cut-point sample from
/// when it declares no explicit one — the caller picks it via
/// [`shared_categorical_batch`], so faceted panels agree.
fn build_discretizing_color_scale(
    enc: &crate::spec::encoding::EncodingSpec,
    sample_batch: &RecordBatch,
    data_extent: (f64, f64),
    theme: &ThemeInputs,
) -> Result<Option<(DiscretizedColors, Vec<crate::render::RenderWarning>)>, RenderError> {
    let Some(partition) = bucket_partition(enc, sample_batch, data_extent)? else {
        return Ok(None);
    };
    let n_buckets = partition.thresholds.len() + 1;
    let mut warnings = Vec::new();
    let mut colors = bucket_colors(enc, theme, n_buckets, partition.extent, &mut warnings);
    if partition.descending {
        colors.reverse();
    }
    let (lo, hi) = partition.extent;
    let mut bounds = Vec::with_capacity(n_buckets + 1);
    bounds.push(lo);
    bounds.extend_from_slice(&partition.thresholds);
    bounds.push(hi);
    Ok(DiscretizedColors::new(bounds, colors).map(|scale| (scale, warnings)))
}

/// Extract the bucket partition from a discretizing scale spec. See
/// [`build_discretizing_color_scale`] for the `Ok(None)` and `Err` contracts.
fn bucket_partition(
    enc: &crate::spec::encoding::EncodingSpec,
    sample_batch: &RecordBatch,
    data_extent: (f64, f64),
) -> Result<Option<BucketPartition>, RenderError> {
    use crate::spec::encoding::ScaleSpec;
    let Some(scale) = enc.scale.as_ref() else { return Ok(None) };
    match scale {
        // Uniform buckets over the declared extent (or the data extent when the
        // spec omits one); the color `range` fixes the bucket count.
        ScaleSpec::Quantize { domain, range } => {
            let Some(colors) = non_empty(range.as_deref()) else { return Ok(None) };
            let n = colors.len();
            let ((lo, hi), descending) = declared_extent(domain.as_deref(), data_extent)?;
            // Endpoints already normalized ascending, so the shared formula
            // yields ascending boundaries.
            let thresholds = crate::scale::core::uniform_bin_thresholds(lo, hi, n);
            Ok(Some(BucketPartition { thresholds, extent: (lo, hi), descending }))
        }
        // Equal-frequency buckets at the sample's quantile cut-points; the
        // `range` fixes the bucket count. The sample is the declared `domain`
        // when present, else the encoded column itself.
        ScaleSpec::Quantile { domain, range } => {
            let Some(outputs) = non_empty(range.as_deref()) else { return Ok(None) };
            let n = outputs.len();
            let sample = match non_empty(domain.as_deref()) {
                Some(d) => d.to_vec(),
                None => column_sample(sample_batch, &enc.field),
            };
            let mut sorted: Vec<f64> = sample.into_iter().filter(|v| v.is_finite()).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let extent = match (sorted.first(), sorted.last()) {
                (Some(lo), Some(hi)) => (*lo, *hi),
                _ => data_extent,
            };
            let thresholds = crate::scale::core::compute_quantile_cuts(&sorted, n);
            Ok(Some(BucketPartition { thresholds, extent, descending: false }))
        }
        // Explicit boundaries: k thresholds → k + 1 buckets, both end buckets
        // open. The labeling extent comes from the data, widened to cover the
        // outermost threshold so `bounds` stays ascending.
        ScaleSpec::Threshold { domain, .. } => {
            let Some(declared) = non_empty(domain.as_deref()) else { return Ok(None) };
            declared_boundaries(declared, "threshold", "domain", data_extent).map(Some)
        }
        ScaleSpec::BinOrdinal { bins, .. } => {
            let Some(declared) = non_empty(bins.as_deref()) else { return Ok(None) };
            declared_boundaries(declared, "bin-ordinal", "bins", data_extent).map(Some)
        }
        _ => Ok(None),
    }
}

/// Turn a `Quantize` spec's declared `domain` into an ascending `(lo, hi)`
/// extent plus the swatch-reversal flag (spec §4.2, amended 2026-08-28).
///
/// The `Quantize` sibling of [`declared_boundaries`], meeting the same
/// constructor-bypass problem at the same boundary: `QuantizeScale::new` rejects
/// `lo == hi`, but `Color(scale={...})` accepts a raw dict, so a zero-width
/// declared domain would otherwise collapse every boundary onto one value
/// (`[5, 5, 5, 5]` for three buckets) — middle swatches unreachable, no
/// diagnostic. Refused with the constructor's own sentence
/// ([`DEGENERATE_DOMAIN_MESSAGE`]). A descending domain normalizes and reverses
/// the swatches, exactly as a descending boundary list does.
///
/// Only a **declared** domain is validated. With no declared domain the extent
/// comes from the data, where a constant column legitimately yields `lo == hi`;
/// that is pre-existing data-derived behavior (and mirrors the continuous path's
/// degenerate handling), not user error, so it is left alone.
fn declared_extent(
    declared: Option<&[f64]>,
    data_extent: (f64, f64),
) -> Result<((f64, f64), bool), RenderError> {
    use crate::scale::core::DEGENERATE_DOMAIN_MESSAGE;

    let Some(d) = declared.filter(|d| d.len() >= 2) else {
        return Ok((data_extent, false));
    };
    let (lo, hi) = (d[0], d[d.len() - 1]);
    if lo == hi {
        return Err(RenderError::ScaleResolutionFailed(format!(
            "color scale of type 'quantize': {DEGENERATE_DOMAIN_MESSAGE}"
        )));
    }
    let descending = hi < lo;
    Ok((if descending { (hi, lo) } else { (lo, hi) }, descending))
}

/// Turn a `Threshold`/`BinOrdinal` spec's declared boundary list into an
/// ascending partition (spec §4.2, amended 2026-08-28).
///
/// The `ThresholdScale`/`BinOrdinalScale` constructors reject anything but a
/// strictly ascending list, but `Color(scale={...})` accepts a raw dict and
/// bypasses them, so this is the boundary where an unvalidated list is met:
///
/// - **strictly ascending** — used as declared (every list a constructor would
///   have accepted takes this path, so those renders are byte-identical);
/// - **strictly descending** — normalized to ascending with the swatches
///   reversed, exactly as [`ScaleSpec::Quantize`]'s descending domain is
///   handled, so the same user intent reads the same way on all three variants;
/// - **anything else** (non-monotonic, or repeated boundaries) — a typed
///   refusal quoting the constructors' own rejection sentence
///   ([`not_strictly_ascending_message`]), never a silent mis-bucketing. An
///   unordered list has no defensible bucket assignment: `lookup`'s
///   `partition_point` would be well-defined but arbitrary, painting
///   middle-bucket values the top bucket and leaving swatches unreachable.
///
/// `scale_type` is the wire tag (`"threshold"`, `"bin-ordinal"`) and `field`
/// the list's spelling in that spec (`"domain"`, `"bins"`) — the same word the
/// matching constructor names in its own error.
fn declared_boundaries(
    declared: &[f64],
    scale_type: &str,
    field: &str,
    data_extent: (f64, f64),
) -> Result<BucketPartition, RenderError> {
    use crate::scale::core::{is_strictly_ascending, not_strictly_ascending_message};

    // An empty or single-element list is trivially ascending, so it never reads
    // as "descending" and never reverses its swatches.
    let ascending = is_strictly_ascending(declared);
    let descending = !ascending && declared.windows(2).all(|w| w[0] > w[1]);
    if !ascending && !descending {
        return Err(RenderError::ScaleResolutionFailed(format!(
            "color scale of type '{scale_type}': {}",
            not_strictly_ascending_message(field)
        )));
    }
    let thresholds: Vec<f64> = if descending {
        declared.iter().rev().copied().collect()
    } else {
        declared.to_vec()
    };
    let extent = widen_to_cover(data_extent, &thresholds);
    Ok(BucketPartition { thresholds, extent, descending })
}

/// `Some(slice)` when the option holds a non-empty slice, else `None`.
fn non_empty<T>(values: Option<&[T]>) -> Option<&[T]> {
    values.filter(|v| !v.is_empty())
}

/// The encoded column's values in `batch`, used as the quantile sample when a
/// `Quantile` spec declares no explicit sample domain.
///
/// `batch` is the caller's [`shared_categorical_batch`] selection: the global
/// `FINAL_OUTPUT_KEY` rows under facet sharing (so every panel cuts at the same
/// quantiles as the global colorbar), the panel's own rows otherwise.
///
/// Nulls are dropped; an unsupported dtype yields an empty sample (the caller
/// then falls back to the data extent with no cut-points), which the
/// numeric-dtype guard in `build_color_scale` already makes unreachable.
fn column_sample(batch: &RecordBatch, field: &str) -> Vec<f64> {
    crate::render::arrow_cast::col_as_f64(batch, field)
        .map(|values| values.into_iter().flatten().collect())
        .unwrap_or_default()
}

/// Widen `(lo, hi)` so it covers every threshold — the outer labeling bounds
/// must bracket the interior boundaries for `bounds` to stay ascending.
fn widen_to_cover((lo, hi): (f64, f64), thresholds: &[f64]) -> (f64, f64) {
    let mut lo = lo;
    let mut hi = hi;
    for t in thresholds {
        lo = lo.min(*t);
        hi = hi.max(*t);
    }
    (lo, hi)
}

/// The `n_buckets` swatch colors for a discretizing color scale (spec §4.2
/// precedence): an explicit string `range` wins, then the resolved scheme, then
/// the theme default scheme.
///
/// The scheme name is whatever [`scale_common_scheme`] resolves for *this*
/// encoding — `enc.scheme` or the spec's own `scheme` field — so the categorical
/// branch below is reachable from every discretizing variant, not just
/// `BinOrdinal` (which is merely the only one with a `scheme` field of its own;
/// the rest reach it through the top-level `encoding.scheme`).
///
/// Per spec §4.2 (amended 2026-08-28): a **categorical** scheme name (e.g.
/// `"tableau10"`) contributes its palette entries in declaration order rather
/// than being sampled, and when `n_buckets` exceeds the palette length the
/// entries cycle *and* a `ColorPaletteOverflowed` warning is emitted — mirroring
/// [`build_default_categorical_scale`]'s recycling contract, never a silent
/// degrade. Every other scheme is sampled at `n_buckets` evenly spaced points.
///
/// The explicit range parse is all-or-nothing, mirroring the categorical path:
/// one unparseable entry discards the whole range, pushes a
/// `ColorRangeParseFailure` warning naming it, and falls through to the scheme.
fn bucket_colors(
    enc: &crate::spec::encoding::EncodingSpec,
    theme: &ThemeInputs,
    n_buckets: usize,
    extent: (f64, f64),
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> Vec<Color> {
    if let Some(strings) = explicit_string_range(enc) {
        let parsed: Result<Vec<Color>, String> = strings
            .iter()
            .map(|s| crate::render::color::parse_color(s).map_err(|_| s.clone()))
            .collect();
        match parsed {
            Ok(colors) if colors.len() == n_buckets => return colors,
            // A range of the wrong length cannot describe this partition; the
            // bucket count is derived from the range for every spec that can
            // carry one, so this is unreachable for a well-formed spec.
            Ok(_) => {}
            Err(entry) => warnings
                .push(crate::render::RenderWarning::ColorRangeParseFailure { entry }),
        }
    }
    let scheme_name = enc.scheme.as_deref().or_else(|| scale_common_scheme(enc));
    if let Some(name) = scheme_name.filter(|n| palette::is_categorical_scheme(n)) {
        let entries = palette::categorical_palette(name);
        if n_buckets > entries.len() {
            warnings.push(crate::render::RenderWarning::ColorPaletteOverflowed {
                categories: n_buckets as u32,
            });
        }
        return (0..n_buckets).map(|i| entries[i % entries.len()]).collect();
    }
    let (scheme, scheme_warnings) = resolve_continuous_scheme(enc, theme, extent);
    warnings.extend(scheme_warnings);
    (0..n_buckets)
        .map(|i| {
            let t = if n_buckets <= 1 {
                0.5
            } else {
                i as f64 / (n_buckets - 1) as f64
            };
            scheme.sample(t)
        })
        .collect()
}

/// The `(lo, hi)` extent of a 10-pre-b composite shared color domain, when it is
/// the continuous ([`SharedDomain::Numeric`]) variant. `None` otherwise.
fn composite_numeric_extent(domain: Option<&SharedDomain>) -> Option<(f64, f64)> {
    match domain? {
        SharedDomain::Numeric { lo, hi } => Some((*lo, *hi)),
        SharedDomain::Ordinal(_) => None,
    }
}

/// The category vector of a 10-pre-b composite shared color domain, when it is
/// the categorical ([`SharedDomain::Ordinal`]) variant. `None` otherwise.
fn composite_categorical_domain(domain: Option<&SharedDomain>) -> Option<Vec<String>> {
    match domain? {
        SharedDomain::Ordinal(cats) => Some(cats.clone()),
        SharedDomain::Numeric { .. } => None,
    }
}

/// Build a `ColorScale::Categorical` from the default theme palette.
///
/// Resolves the palette name from (1) `enc.scheme`, falling back to
/// `theme.palette.color_scheme`. When the resolved name is a sequential/diverging
/// scheme (not a categorical one), "tableau10" is used as the fallback to avoid
/// single-hue categorical color. Appends a `ColorPaletteOverflowed` warning to
/// `warnings` when the domain has more categories than palette entries.
fn build_default_categorical_scale(
    domain: Vec<String>,
    enc: &crate::spec::encoding::EncodingSpec,
    theme: &ThemeInputs,
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> ColorScale {
    let resolved_name: &str = enc.scheme.as_deref().unwrap_or(&theme.palette.color_scheme);
    let static_palette: &'static [Color] = if palette::is_sequential_scheme(resolved_name) {
        palette::categorical_palette("tableau10")
    } else {
        palette::categorical_palette(resolved_name)
    };
    let palette: Cow<'static, [Color]> = Cow::Borrowed(static_palette);
    if domain.len() > palette.len() {
        warnings.push(crate::render::RenderWarning::ColorPaletteOverflowed {
            categories: domain.len() as u32,
        });
    }
    ColorScale::Categorical { domain, palette }
}

/// Extract the `scheme` string from a `ScaleSpec`.
///
/// Covers the `Sequential`/`Diverging`/`BinOrdinal` variants (which carry their
/// own `scheme` field) and the `ContinuousScaleCommon`-bearing variants (Linear,
/// Log, Time, Symlog, Pow, Sqrt, Utc). Returns `None` for ordinal/categorical
/// variants and when no scheme is set.
fn scale_common_scheme(enc: &crate::spec::encoding::EncodingSpec) -> Option<&str> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
        // D1: Sequential and Diverging carry their own scheme field;
        // BinOrdinal names the scheme its bin swatches are drawn from.
        ScaleSpec::Sequential { scheme, .. }
        | ScaleSpec::Diverging { scheme, .. }
        | ScaleSpec::BinOrdinal { scheme, .. } => scheme.as_deref(),
        // D4 (pre-existing): Linear and friends embed scheme in ContinuousScaleCommon.
        ScaleSpec::Linear   { common, .. }
        | ScaleSpec::Log    { common, .. }
        | ScaleSpec::Time   { common, .. }
        | ScaleSpec::Symlog { common, .. }
        | ScaleSpec::Pow    { common, .. }
        | ScaleSpec::Sqrt   { common }
        | ScaleSpec::Utc    { common, .. } => common.scheme.as_deref(),
        _ => None,
    }
}

/// Extract the explicit `(lo, hi)` color extent a scale spec declares.
///
/// Delegates the extent-vs-binning classification to
/// [`ScaleSpec::positional_extent`], the single place a scale variant declares
/// whether its `domain` is an extent (`Sequential`/`Diverging` outer bounds, a
/// `Linear` and friends `common.domain`) or a binning artifact (`Quantile`
/// sample lists, `Threshold` boundaries, `BinOrdinal` edges — resolved by
/// [`bucket_partition`] instead). Honoring the continuous variants' `common`
/// domain here is what makes `heatmap(vmin=, vmax=)` take effect.
///
/// A `Diverging` midpoint is carried separately by `scale_diverging_midpoint`
/// and threaded into `ColorScale::Continuous { midpoint }` for piecewise-linear
/// normalization.
///
/// Returns `None` when the spec declares no extent, allowing the caller to fall
/// back to the auto-inferred data extent.
fn scale_explicit_domain(enc: &crate::spec::encoding::EncodingSpec) -> Option<(f64, f64)> {
    let extent = enc.scale.as_ref()?.positional_extent()?;
    (extent.len() >= 2).then(|| (extent[0], extent[extent.len() - 1]))
}

/// Extract the diverging midpoint from a `Diverging` scale spec.
///
/// Midpoint resolution priority:
///   1. `domain_mid` field (`"domainMid"` in JSON) when explicitly set.
///   2. Middle element of a 3-tuple `domain = [lo, mid, hi]`.
///   3. `None` — the caller uses the geometric center implicitly via
///      pure-linear normalization (unchanged behavior for symmetric domains).
///
/// Always returns `None` for non-`Diverging` scale variants and when
/// neither `domain_mid` nor a 3-element domain is present.
fn scale_diverging_midpoint(enc: &crate::spec::encoding::EncodingSpec) -> Option<f64> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
        ScaleSpec::Diverging { domain_mid: Some(mid), .. } => Some(*mid),
        ScaleSpec::Diverging { domain: Some(d), .. } if d.len() == 3 => Some(d[1]),
        _ => None,
    }
}

/// Returns `true` when the scale spec is `Sequential { reverse: true }`.
///
/// Used by `build_color_scale` to wrap the resolved scheme in
/// `ContinuousScheme::Reverse` without cluttering the main resolution logic.
/// `Diverging` does not currently expose a `reverse` field in the spec.
fn scale_spec_is_reversed(enc: &crate::spec::encoding::EncodingSpec) -> bool {
    use crate::spec::encoding::ScaleSpec;
    matches!(enc.scale.as_ref(), Some(ScaleSpec::Sequential { reverse: true, .. }))
}

/// Extract an explicit color-string range from a color `EncodingSpec`.
///
/// Returns `Some(Vec<String>)` for `ScaleSpec::Ordinal` with a string-array
/// `range` (D1 path) and for `ScaleSpec::Quantize`, whose `range` is typed as
/// color strings — the one discretizing variant that can carry explicit
/// swatches (`Quantile`/`Threshold` ranges are numeric outputs and `BinOrdinal`
/// names a scheme instead). Returns `None` for all other scale types and when
/// the range is absent or numeric.
fn explicit_string_range(enc: &crate::spec::encoding::EncodingSpec) -> Option<Vec<String>> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
        ScaleSpec::Ordinal { range, .. } => {
            crate::scale::ordinal::OrdinalRangeValue::all_strings(range.as_deref()?)
        }
        ScaleSpec::Quantize { range, .. } => range.clone(),
        _ => None,
    }
}

/// Extract the declared `domain` from a `ScaleSpec::Ordinal` encoding.
///
/// Returns `Some(Vec<String>)` when the encoding has `scale = ScaleSpec::Ordinal`
/// with an explicit non-empty `domain`.  Returns `None` for all other scale types
/// and when no domain is declared (the caller falls back to data first-appearance
/// order).
fn explicit_ordinal_domain(enc: &crate::spec::encoding::EncodingSpec) -> Option<Vec<String>> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
        ScaleSpec::Ordinal { domain: Some(d), .. } if !d.is_empty() => Some(d.clone()),
        _ => None,
    }
}
