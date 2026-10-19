//! Color scale resolution: categorical and continuous color encoding.

use std::borrow::Cow;
use std::collections::HashMap;

use arrow::record_batch::RecordBatch;

use crate::layout::ThemeInputs;
use crate::spec::encoding::DataType as SpecDataType;

use crate::render::color::Color;
use crate::render::color::palette;
use crate::render::RenderError;

use super::domain::{admitted_aux_reads, admitted_binned_reads, apply_sort_to_domain, aux_category_union, aux_numeric_union, aux_numeric_values, locate_field, AuxRead, SortContext};
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
/// The parse is all-or-nothing: if any color string fails to parse, the whole
/// resolution refuses with a [`RenderError::ColorRangeParseFailure`] naming
/// the first offending entry (#126, spec §4.2 amended 2026-09-11) — the
/// entire explicit range used to be silently discarded in favor of the
/// default theme palette, with only a warning naming it.
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
/// # FA-5 — `grouping_area` for area marks
///
/// When `grouping_area` is `Some`, the Quantitative/Temporal path is skipped
/// and the color field is always resolved as a `Categorical` scale regardless
/// of its Arrow dtype.  Set it for `mark_area`, which always groups rows
/// into discrete per-color bands (via `col_as_ordinal_category_str`) and therefore
/// must use the same categorical palette for both fills and legend swatches.
///
/// Without this flag a Float64/Int64 color column would produce a `Continuous`
/// scale (gradient colorbar legend) while the area fills sampled discrete points
/// on that ramp — legend ≠ fill (the FA-5 bug).
///
/// A binned color (`Color(bin=...)`) is the exception: the area groups its
/// rows by the bin the binned scale assigns, one band per bin, so fills and
/// legend already agree on the bins and `grouping_area` is ignored.
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
///
/// # `layer_reads` — the domain of a layered spec's color
///
/// A layered spec's reads of the color channel ([`AuxRead`]): every layer that
/// draws color, with the field it draws read from the batch it draws. The
/// auto domain is then their union: the numeric extent
/// ([`aux_numeric_union`]) or the categories in layer order
/// ([`aux_category_union`]). Empty for an unlayered spec, which reads the
/// located column (continuous) or the primary batch (categorical). An
/// explicit domain and a composite shared domain still win as above.
///
/// The scale's kind, continuous or categorical, is decided by the rendering
/// encoding's field, the establishing layer's (layer 0 when it draws color,
/// else the first layer that does), read from the batch that layer draws; a
/// later layer whose own color field reads the other way stays out of the
/// union ([`admitted_aux_reads`], one
/// `LayerScaleExcluded` each, returned first among the warnings).
/// `grouping_area` is the layer of an area that draws color, which groups it
/// by category (FA-5, [`super::domain::color_grouping_area`]): under it
/// every column reads as categories, so every layer is admitted, and a
/// raster image's cells, which have none, are refused naming the layer that
/// establishes the categorical scale.
#[allow(clippy::too_many_arguments)]
pub fn build_color_scale(
    encoding: &crate::spec::encoding::Encoding,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    theme: &ThemeInputs,
    grouping_area: Option<usize>,
    facet_shared: bool,
    composite_domain: Option<&SharedDomain>,
    // `spec.coord` is `CoordFlip`: `encoding` above is the RENDERED encoding,
    // so an authored `sort="-y"` shorthand on the color channel needs this to
    // find the channel the user wrote (#95). `false` for every unflipped
    // chart → byte-identical.
    coord_flipped: bool,
    layer_reads: &[AuxRead<'_>],
) -> Result<(Option<ColorScale>, Vec<crate::render::RenderWarning>), RenderError> {
    let Some(c_enc) = &encoding.color else {
        return Ok((None, Vec::new()));
    };
    // A binned color bins numbers whatever type its binding declares, and an
    // area that draws it groups by its bins rather than by category, so the
    // grouping area's categorical rule does not apply to it.
    let binned = c_enc.color_bin().is_some();
    let grouping_area = grouping_area.filter(|_| !binned);
    let force_categorical = grouping_area.is_some();
    // A raster image's cells establish the scale: they are numbers, so a
    // nominal or ordinal binding (or a mark that groups color) has no color
    // for them, and the image would draw nothing.
    if let Some(lead) = layer_reads.first().filter(|lead| lead.is_raster_cells()) {
        let declared = match c_enc.type_.filter(|_| !binned) {
            Some(SpecDataType::Nominal) => Some("nominal"),
            Some(SpecDataType::Ordinal) => Some("ordinal"),
            _ => None,
        };
        let reason = match declared {
            Some(kind) => Some(crate::render::RasterColorReason::Declared(kind)),
            None => grouping_area.map(|layer| area_establishes(layer, layer_reads, c_enc)),
        };
        if let Some(refusal) = reason.and_then(|reason| lead.raster_color_refusal(reason)) {
            return Err(refusal);
        }
        let mut warnings = Vec::new();
        let reads = if binned {
            admitted_binned_reads(layer_reads, primary_batch, transform_outputs, &mut warnings)?
        } else {
            admitted_aux_reads(layer_reads, "color", false, primary_batch, transform_outputs, &mut warnings)?
        };
        refuse_unbinnable_reads(c_enc, &reads, primary_batch, transform_outputs)?;
        let scale = raster_cell_scale(Some(c_enc), &reads, composite_domain, theme, primary_batch, transform_outputs, &mut warnings)?;
        return Ok((scale, warnings));
    }
    // The column that decides the kind: the establishing layer's, read from
    // the batch it draws, on a layered spec; the one batch of an unlayered
    // spec otherwise.
    let located = match layer_reads.first() {
        Some(lead) => lead.locate(primary_batch, transform_outputs),
        None => locate_field(&c_enc.field, primary_batch),
    }
    .ok_or_else(|| RenderError::UnknownColumn { name: c_enc.field.clone() })?;

    let inferred = infer_spec_type(c_enc, located.col.data_type());
    // A binned color's establishing column must be one the `Bin` transform
    // reads.
    if binned {
        let field = layer_reads.first().map_or(c_enc.field.as_str(), AuxRead::field);
        require_binnable(field, located.col)?;
    }
    // FA-5: When force_categorical is set (mark_area), treat any quantitative/temporal
    // color field as categorical.  Area always groups by distinct color values
    // (col_as_ordinal_category_str), so both the fills and the legend must agree on
    // a categorical palette rather than diverging: fills from a continuous ramp vs.
    // legend showing a gradient colorbar.
    let is_continuous_color = !force_categorical
        && (binned || matches!(inferred, SpecDataType::Quantitative | SpecDataType::Temporal));
    let mut exclusions: Vec<crate::render::RenderWarning> = Vec::new();
    let layer_reads = if force_categorical {
        // A mark that groups color reads every column as categories; a
        // raster's cells have none.
        if let (Some(lead), Some(raster), Some(layer)) =
            (layer_reads.first(), layer_reads.iter().find(|read| read.is_raster_cells()), grouping_area)
        {
            // A lead whose own column reads as categories establishes the
            // scale; otherwise the area forces it.
            let reason = match lead.reads_categorical(primary_batch, transform_outputs) {
                true => crate::render::RasterColorReason::CategoricalScale { layer: lead.layer(), field: lead.field().to_string() },
                false => area_establishes(layer, layer_reads, c_enc),
            };
            if let Some(refusal) = raster.raster_color_refusal(reason) {
                return Err(refusal);
            }
        }
        layer_reads.to_vec()
    } else {
        if binned {
            admitted_binned_reads(layer_reads, primary_batch, transform_outputs, &mut exclusions)?
        } else {
            admitted_aux_reads(layer_reads, "color", !is_continuous_color, primary_batch, transform_outputs, &mut exclusions)?
        }
    };
    let layer_reads = layer_reads.as_slice();
    if is_continuous_color {
        refuse_unbinnable_reads(c_enc, layer_reads, primary_batch, transform_outputs)?;
        let continuous_extent = |col: &dyn arrow::array::Array, _: &str| {
            if !crate::render::arrow_cast::is_numeric(col.data_type()) {
                return Err(RenderError::EncodingTypeMismatch {
                    channel: "color",
                    expected: "numeric column for quantitative/temporal type",
                    got: format!("{:?}", col.data_type()),
                    // "color" is never a positional channel, so `CoordFlip` never
                    // touches it, and `user_facing_channel` is identity
                    // regardless. `build_color_scale` has no `ChartSpec`/coord in scope
                    // here, so this is hardcoded rather than threaded for a value
                    // that can never change the rendered message.
                    coord_flipped: false,
                });
            }
            Ok(numeric_extent(col))
        };
        let panel_extent = continuous_extent(located.col, &c_enc.field)?;
        // A layered spec's color spans every layer's own column. Otherwise, T3:
        // when faceted (Shared), union the per-panel extent with the global
        // FINAL_OUTPUT_KEY batch so marks normalize through the same domain as
        // the global colorbar legend.
        let data_extent = match aux_numeric_union(layer_reads, primary_batch, transform_outputs, facet_shared, continuous_extent)? {
            Some(union) => union,
            None if facet_shared => {
                union_panel_with_global_extent(panel_extent, &c_enc.field, transform_outputs)
            }
            None => panel_extent,
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
        //
        // A binned color's edges and a quantile color's cut-points are taken
        // from the values every admitted layer draws, as the extent above is.
        let sample = || match (composite_samples(composite_domain), layer_reads) {
            (Some(shared), _) => shared,
            (None, []) => column_sample(
                shared_categorical_batch(located.batch, &c_enc.field, transform_outputs, facet_shared),
                &c_enc.field,
            ),
            (None, reads) => aux_numeric_values(reads, primary_batch, transform_outputs, facet_shared),
        };
        if let Some((buckets, warnings)) =
            build_discretizing_color_scale(c_enc, sample, data_extent, theme)?
        {
            exclusions.extend(warnings);
            return Ok((Some(ColorScale::Discretizing(buckets)), exclusions));
        }

        // D1: honor explicit domain from the scale spec. When the spec carries
        // an extent (`positional_extent`), use those bounds instead of
        // auto-inferring from the data column. When the spec domain is absent, a
        // 10-pre-b composite shared extent wins over the per-leaf data extent;
        // otherwise fall back to the data extent.
        let (lo, hi) = scale_explicit_domain(c_enc)
            .or_else(|| composite_numeric_extent(composite_domain))
            .unwrap_or(data_extent);

        let (scheme, scheme_warnings) = resolve_continuous_scheme(c_enc, theme, (lo, hi), None)?;
        // Gap 2: resolve the diverging midpoint.  Priority:
        //   1. explicit `domain_mid`/`domainMid` field on DivergingScale spec
        //   2. middle element of a 3-tuple domain=[lo, mid, hi]
        //   3. None — geometric center falls out from pure-linear normalization
        // Sequential scales always get None (pure-linear).
        let midpoint = scale_diverging_midpoint(c_enc);
        exclusions.extend(scheme_warnings);
        // A `SymlogScale` maps through symlog; every other scale linearly.
        let transform = super::ColorTransform::of_scale(c_enc.scale.as_ref()).unwrap_or(super::ColorTransform::Linear);
        Ok((Some(ColorScale::Continuous { domain: (lo, hi), scheme, midpoint, transform }), exclusions))
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
        //
        // A layered spec's categories are every layer's own, in layer order
        // (`layer_reads`); a data-aware sort then aggregates over the first
        // contributing batch.
        let (layer_categories, domain_batch) =
            match aux_category_union(layer_reads, primary_batch, transform_outputs, facet_shared)? {
                Some((categories, batch)) => (Some(categories), batch),
                None => (
                    None,
                    shared_categorical_batch(primary_batch, &c_enc.field, transform_outputs, facet_shared),
                ),
            };
        let data_categories = |batch| -> Result<Vec<String>, RenderError> {
            match &layer_categories {
                Some(categories) => Ok(categories.clone()),
                None => distinct_values_in_order(batch, &c_enc.field),
            }
        };

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
            // Rendered channels (`encoding` is post-`CoordFlip`); the flag is
            // what lets an authored `sort="-y"` still name the user's own y
            // channel (#95, same rule the positional axes resolve under).
            x_field: encoding.x.as_ref().map(|e| e.field.as_str()),
            y_field: encoding.y.as_ref().map(|e| e.field.as_str()),
            coord_flipped,
        };
        // D1: if the color encoding carries an explicit ordinal string range, build
        // the palette from those colors (parsed via `parse_color`).
        //
        // Domain resolution order (mirrors positional.rs OrdinalScale behavior):
        //   1. `scale.domain` (declared explicit domain)
        //   2. Data first-appearance order (`distinct_values_in_order`)
        // `enc.sort` is applied on top of the resolved domain.
        //
        // The parse is all-or-nothing: if any entry fails to parse, resolution
        // refuses with a `RenderError::ColorRangeParseFailure` naming the
        // first offending entry (#126) rather than falling through to the
        // default theme palette below.
        if let Some(color_strings) = explicit_string_range(c_enc) {
            let parse_result: Result<Vec<Color>, String> = color_strings
                .iter()
                .map(|s| {
                    crate::render::color::parse_color(s)
                        .map_err(|_| s.clone())
                })
                .collect::<Result<Vec<_>, _>>();
            match parse_result {
                Ok(parsed) => {
                    let mut warnings: Vec<crate::render::RenderWarning> = Vec::new();
                    // Build the domain from the declared scale.domain (if present),
                    // falling back to data first-appearance order from the
                    // domain_batch (global when facet_shared).
                    let mut domain = match super::domain::explicit_ordinal_domain(c_enc) {
                        Some(declared) => declared,
                        None => data_categories(domain_batch)?,
                    };
                    apply_sort_to_domain(&mut domain, c_enc.sort.as_ref(), &sort_ctx, &mut warnings)?;
                    let palette: Cow<'static, [Color]> = Cow::Owned(parsed);
                    // No overflow warning when the user supplied an explicit range;
                    // they own the mapping and repeated colors are intentional.
                    exclusions.extend(warnings);
                    return Ok((Some(ColorScale::Categorical { domain, palette }), exclusions));
                }
                Err(bad_entry) => {
                    // #126: all-or-nothing parse refuses rather than falling
                    // through to the default palette — one bad entry used to
                    // silently swap the user's requested mapping for the
                    // theme's, with only a warning naming it.
                    return Err(RenderError::ColorRangeParseFailure { entry: bad_entry });
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
            None => data_categories(domain_batch)?,
        };
        apply_sort_to_domain(&mut domain, c_enc.sort.as_ref(), &sort_ctx, &mut warnings)?;
        let scale = build_default_categorical_scale(domain, c_enc, theme, &mut warnings);
        exclusions.extend(warnings);
        Ok((Some(scale), exclusions))
    }
}

/// Resolve the continuous color scheme for an encoding:
///   1. a `Sequential` spec's `stops` ([`gradient_scheme_from_stops`]) — a
///      `Gradient`-backed scheme, F-L04-02 second revision
///   2. `encoding.scheme` (top-level field on `EncodingSpec`, D1/D4)
///   3. the scale spec's own scheme ([`scale_common_scheme`])
///   4. the theme's diverging scheme for a `Diverging` scale spec or when
///      `(lo, hi)` straddles zero, else its sequential scheme
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
/// #240-4b (routed rider): when the user named a scheme at step 2 or 3 (via
/// `encoding.scheme` or the scale spec's own `scheme` field) and that name is
/// not a recognized [`NamedContinuous`], the requested name is dropped in
/// favor of the theme default — previously silent. That drop is now reported
/// via [`crate::render::RenderWarning::UnknownColorScheme`], naming the
/// unrecognized name. A chart that names NO scheme at all (both step 2 and
/// step 3 absent) still resolves the theme default silently — nothing was
/// requested, so nothing was dropped.
///
/// Errors when the stops parse fails (see [`gradient_scheme_from_stops`]).
/// Returns any `RenderWarning`s the resolution produced; the caller folds
/// them into its accumulator.
fn resolve_continuous_scheme(
    enc: &crate::spec::encoding::EncodingSpec,
    theme: &ThemeInputs,
    (lo, hi): (f64, f64),
    // A mark's own colormap (`mark_raster(cmap=)`), after the encoding's and
    // its scale's scheme and before the theme default.
    mark_cmap: Option<&str>,
) -> Result<(crate::render::color::ContinuousScheme, Vec<crate::render::RenderWarning>), RenderError> {
    use crate::render::color::{ContinuousScheme, NamedContinuous};
    let mut warnings = Vec::new();
    let mut scheme = match gradient_scheme_from_stops(enc, &mut warnings)? {
        Some(s) => s,
        None => {
            let requested = enc.scheme.as_deref().or(scale_common_scheme(enc));
            match requested.and_then(NamedContinuous::from_name) {
                Some(named) => ContinuousScheme::Named(named),
                None => {
                    // #240-4b: the user asked for a name and it did not
                    // resolve — warn before falling back. A `None` `requested`
                    // means nothing was asked for at all, so no warning.
                    if let Some(name) = requested {
                        warnings.push(crate::render::RenderWarning::UnknownColorScheme {
                            name: name.to_string(),
                        });
                    }
                    // A `Diverging` scale asked for a diverging ramp whatever
                    // the data sign; any other scale guesses from the sign.
                    let diverging_spec = matches!(
                        enc.scale.as_ref(),
                        Some(crate::spec::encoding::ScaleSpec::Diverging { .. })
                    );
                    let theme_scheme = if diverging_spec || (lo < 0.0 && hi > 0.0) {
                        &theme.palette.diverging_scheme
                    } else {
                        &theme.palette.sequential_scheme
                    };
                    mark_cmap
                        .and_then(NamedContinuous::from_name)
                        .or_else(|| NamedContinuous::from_name(theme_scheme))
                        .map(ContinuousScheme::Named)
                        .unwrap_or(ContinuousScheme::Named(NamedContinuous::Viridis))
                }
            }
        }
    };
    if scale_spec_is_reversed(enc) {
        scheme = ContinuousScheme::Reverse(Box::new(scheme));
    }
    Ok((scheme, warnings))
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
/// `explicit_string_range` convention: one unparseable stop refuses with a
/// [`RenderError::ColorRangeParseFailure`] naming the first offending entry
/// (#126) rather than rendering a partial or silently-wrong gradient.
///
/// Pushes [`crate::render::RenderWarning::GradientTooFewStops`] onto
/// `warnings` (wave-1 close remediation, #240-4b's last silent arm) when the
/// stops parse but fewer than 2 survive — see that variant's doc for why
/// this arm is reachable at all despite the typed `fm.Gradient(...)`
/// constructor refusing the same shape at construction.
fn gradient_scheme_from_stops(
    enc: &crate::spec::encoding::EncodingSpec,
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> Result<Option<crate::render::color::ContinuousScheme>, RenderError> {
    let Some(stops) = sequential_stops(enc) else { return Ok(None) };
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
            Ok(Some(crate::render::color::ContinuousScheme::Gradient(gradient_stops)))
        }
        // Fewer than 2 stops can't form a gradient. The `Gradient(...)`
        // pyfunction now rejects this at construction (spec reviewer cycle
        // 3, finding 2 — `_to_scale_spec_dict` only ever emits what a valid
        // `ContinuousScheme::Gradient` carries), so this arm is reachable
        // only from a hand-written-JSON spec that bypasses that constructor.
        // Falls back to the theme default scheme (the caller's `None` arm)
        // rather than refusing — this is a WARN-DEGRADED shape, not a
        // malformed one, matching `UnknownColorScheme`'s own treatment of
        // "asked for something that didn't resolve".
        Ok(gradient_stops) => {
            warnings.push(crate::render::RenderWarning::GradientTooFewStops {
                count: gradient_stops.len(),
            });
            Ok(None)
        }
        Err(entry) => Err(RenderError::ColorRangeParseFailure { entry }),
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

/// The area at `layer` establishes a categorical color scale by the field it
/// draws: its read's among `reads`, else the rendering binding's (an
/// unlayered area chart, which has no reads).
fn area_establishes(
    layer: usize,
    reads: &[AuxRead<'_>],
    binding: &crate::spec::encoding::EncodingSpec,
) -> crate::render::RasterColorReason {
    let field = reads.iter().find(|read| read.layer() == layer).map_or(binding.field.as_str(), AuxRead::field);
    crate::render::RasterColorReason::CategoricalScale { layer, field: field.to_string() }
}

/// Resolve a `Quantize`/`Quantile`/`Threshold`/`BinOrdinal` color encoding into
/// bucket boundaries plus one flat color per bucket (spec §4.2).
///
/// Returns `Ok(None)` — leaving the caller on the continuous path,
/// byte-identically to the pre-discretizing behavior — for every
/// non-discretizing scale spec and for a discretizing spec that declares no
/// bucket count at all (a `Quantize` or `Quantile` with no `range`, a
/// `Threshold` with no `domain`, a `BinOrdinal` with neither `bins` nor
/// `bin`). The Python constructors make all four mandatory, so those are
/// hand-written-JSON shapes. A binned color (`BinOrdinal` with `bin` and no
/// `bins`) whose `sample` holds no finite value also returns `Ok(None)`.
///
/// Returns `Err` for a `Threshold`/`BinOrdinal` boundary list that is neither
/// ascending nor descending — see [`declared_boundaries`].
///
/// `sample` yields the values a `Quantile` spec takes its cut-points from when
/// it declares no explicit sample, and a binned color its edges from: the
/// encoded column (its caller picks the batch via
/// [`shared_categorical_batch`], so faceted panels agree), or, on a layered
/// spec, the values of every admitted layer, raster cells included
/// ([`aux_numeric_values`]).
fn build_discretizing_color_scale(
    enc: &crate::spec::encoding::EncodingSpec,
    sample: impl FnOnce() -> Vec<f64>,
    data_extent: (f64, f64),
    theme: &ThemeInputs,
) -> Result<Option<(DiscretizedColors, Vec<crate::render::RenderWarning>)>, RenderError> {
    let Some(partition) = bucket_partition(enc, sample, data_extent)? else {
        return Ok(None);
    };
    let n_buckets = partition.thresholds.len() + 1;
    let mut warnings = Vec::new();
    let mut colors = bucket_colors(enc, theme, n_buckets, partition.extent, &mut warnings)?;
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
    sample: impl FnOnce() -> Vec<f64>,
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
        // when present, else the encoded values (`sample`): the column, or a
        // raster's cells.
        ScaleSpec::Quantile { domain, range } => {
            let Some(outputs) = non_empty(range.as_deref()) else { return Ok(None) };
            let n = outputs.len();
            let sample = match non_empty(domain.as_deref()) {
                Some(d) => d.to_vec(),
                None => sample(),
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
        // Explicit edges win; otherwise a binned color's `bin` parameters cut
        // the encoded values into the `Bin` transform's edges.
        ScaleSpec::BinOrdinal { bins, bin, .. } => match (non_empty(bins.as_deref()), bin) {
            (Some(declared), _) => declared_boundaries(declared, "bin-ordinal", "bins", data_extent).map(Some),
            (None, Some(bin)) => {
                let finite: Vec<f64> = sample().into_iter().filter(|v| v.is_finite()).collect();
                derived_bin_partition(bin, &finite)
            }
            (None, None) => Ok(None),
        },
        _ => Ok(None),
    }
}

/// The partition a binned color's `bin` parameters make of the finite
/// `values`: one bucket per bin of the edges the `Bin` transform computes
/// over them ([`crate::transform::bin::scale_bin_edges`]), its interior
/// edges the thresholds and its outer edges the extent. `Ok(None)` when
/// there is no value, leaving the caller on the continuous path as an empty
/// domain leaves any continuous color. A `bin` the edge rule refuses (a zero
/// `bin_count`, a non-positive `bin_width`, an extent with `lo >= hi`) is
/// refused with its text.
fn derived_bin_partition(
    bin: &crate::spec::encoding::ScaleBin,
    values: &[f64],
) -> Result<Option<BucketPartition>, RenderError> {
    let edges = crate::transform::bin::scale_bin_edges(bin, values)
        .map_err(|e| RenderError::ScaleResolutionFailed(format!("color scale of type 'bin-ordinal': {e}")))?;
    Ok(edges.and_then(|edges| match edges.as_slice() {
        [lo, interior @ .., hi] => {
            Some(BucketPartition { thresholds: interior.to_vec(), extent: (*lo, *hi), descending: false })
        }
        _ => None,
    }))
}

/// A binned color bins its column with the `Bin` transform's edge rule,
/// which reads only numbers: any other column `field` holds is refused,
/// naming it.
fn require_binnable(field: &str, col: &dyn arrow::array::Array) -> Result<(), RenderError> {
    if crate::transform::numeric_util::is_numeric_dtype(col.data_type()) {
        return Ok(());
    }
    Err(RenderError::BinnedColorNotNumeric {
        channel: "color",
        field: field.to_string(),
        dtype: format!("{:?}", col.data_type()),
    })
}

/// [`require_binnable`] over every column read in `reads`, for a binned
/// color: each admitted layer's column joins the edges' values.
fn refuse_unbinnable_reads(
    enc: &crate::spec::encoding::EncodingSpec,
    reads: &[AuxRead<'_>],
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
) -> Result<(), RenderError> {
    if enc.color_bin().is_none() {
        return Ok(());
    }
    reads
        .iter()
        .filter_map(|read| read.locate(primary_batch, transform_outputs).map(|located| (read.field(), located)))
        .try_for_each(|(field, located)| require_binnable(field, located.col))
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
/// The explicit range parse is all-or-nothing, mirroring the categorical
/// path: one unparseable entry refuses with a
/// [`RenderError::ColorRangeParseFailure`] naming it (#126), rather than
/// falling through to the scheme.
fn bucket_colors(
    enc: &crate::spec::encoding::EncodingSpec,
    theme: &ThemeInputs,
    n_buckets: usize,
    extent: (f64, f64),
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> Result<Vec<Color>, RenderError> {
    if let Some(strings) = explicit_string_range(enc) {
        let parsed: Result<Vec<Color>, String> = strings
            .iter()
            .map(|s| crate::render::color::parse_color(s).map_err(|_| s.clone()))
            .collect();
        match parsed {
            Ok(colors) if colors.len() == n_buckets => return Ok(colors),
            // A range of the wrong length cannot describe this partition; the
            // bucket count is derived from the range for every spec that can
            // carry one, so this is unreachable for a well-formed spec.
            Ok(_) => {}
            Err(entry) => return Err(RenderError::ColorRangeParseFailure { entry }),
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
        return Ok((0..n_buckets).map(|i| entries[i % entries.len()]).collect());
    }
    let (scheme, scheme_warnings) = resolve_continuous_scheme(enc, theme, extent, None)?;
    warnings.extend(scheme_warnings);
    Ok((0..n_buckets)
        .map(|i| {
            let t = if n_buckets <= 1 {
                0.5
            } else {
                i as f64 / (n_buckets - 1) as f64
            };
            scheme.sample(t)
        })
        .collect())
}

/// The `(lo, hi)` extent of a 10-pre-b composite shared color domain, when it is
/// the continuous ([`SharedDomain::Numeric`]) variant. `None` otherwise.
fn composite_numeric_extent(domain: Option<&SharedDomain>) -> Option<(f64, f64)> {
    match domain? {
        SharedDomain::Numeric { lo, hi } => Some((*lo, *hi)),
        SharedDomain::Ordinal(_) | SharedDomain::Samples { .. } | SharedDomain::FixedBins { .. } => None,
    }
}

/// The pooled values of a composite-shared binned color
/// ([`SharedDomain::Samples`]), its edges' sample in place of the leaf's own.
/// `None` otherwise.
fn composite_samples(domain: Option<&SharedDomain>) -> Option<Vec<f64>> {
    match domain? {
        SharedDomain::Samples { values, .. } => Some(values.clone()),
        SharedDomain::FixedBins { .. } => None,
        SharedDomain::Numeric { .. } | SharedDomain::Ordinal(_) => None,
    }
}

/// The category vector of a 10-pre-b composite shared color domain, when it is
/// the categorical ([`SharedDomain::Ordinal`]) variant. `None` otherwise.
fn composite_categorical_domain(domain: Option<&SharedDomain>) -> Option<Vec<String>> {
    match domain? {
        SharedDomain::Ordinal(cats) => Some(cats.clone()),
        SharedDomain::Numeric { .. } | SharedDomain::Samples { .. } | SharedDomain::FixedBins { .. } => None,
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
/// and threaded into `ColorScale::Continuous { midpoint, transform: crate::render::scale_resolve::ColorTransform::Linear }` for piecewise-linear
/// normalization.
///
/// Returns `None` when the spec declares no extent, allowing the caller to fall
/// back to the auto-inferred data extent.
fn scale_explicit_domain(enc: &crate::spec::encoding::EncodingSpec) -> Option<(f64, f64)> {
    super::domain::encoding_explicit_extent(enc)
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
/// Returns `Some(Vec<String>)` for `ScaleSpec::Ordinal` with a non-empty,
/// all-string `range` (D1 path, on [`super::domain::explicit_ordinal_range`])
/// and for `ScaleSpec::Quantize`, whose `range` is typed as color strings —
/// the one discretizing variant that can carry explicit swatches
/// (`Quantile`/`Threshold` ranges are numeric outputs and `BinOrdinal` names
/// a scheme instead). Returns `None` for all other scale types and when the
/// range is absent, empty or numeric.
fn explicit_string_range(enc: &crate::spec::encoding::EncodingSpec) -> Option<Vec<String>> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
        ScaleSpec::Ordinal { .. } => crate::scale::ordinal::OrdinalRangeValue::all_strings(
            super::domain::explicit_ordinal_range(enc)?,
        ),
        ScaleSpec::Quantize { range, .. } => range.clone(),
        _ => None,
    }
}

/// The color scale a raster image's cells map through, the one helper for
/// both the chart color scale a raster establishes (`binding`, its `color`)
/// and the scale every raster drawing no color shares (`binding` `None`,
/// `scale_resolve`'s `raster_cells` scale), so the two share their domain
/// rule, transform and scheme.
///
/// - Domain: the union of `reads` (each raster's cells as the image draws
///   them, [`crate::render::marks::image::raster_cell_domain`]: from zero
///   when every cell is non-negative; a column read's extent), unless the
///   binding states one, or a composite shares one.
/// - Transform: the binding's `LinearScale`/`SymlogScale` when it states
///   one, else the scale the first raster batch names (`value_scale`:
///   symlog under `log_scale`), else linear.
/// - Scheme: the binding's scheme or scale scheme, else the first raster's
///   colormap (`mark_raster(cmap=)`), else the theme default.
///
/// A discretizing binding (`Quantize`, `Threshold`, `BinOrdinal`) instead
/// gives the discretizing scale it gives every other mark, over the same
/// domain: the cells step through its buckets and its legend is its stepped
/// bar.
///
/// `None` when no read holds a finite value (an empty raster).
pub(in crate::render) fn raster_cell_scale(
    binding: Option<&crate::spec::encoding::EncodingSpec>,
    reads: &[AuxRead<'_>],
    composite_domain: Option<&SharedDomain>,
    theme: &ThemeInputs,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> Result<Option<ColorScale>, RenderError> {
    let numeric = |col: &dyn arrow::array::Array, _: &str| Ok(numeric_extent(col));
    let Some(data_domain) = aux_numeric_union(reads, primary_batch, transform_outputs, false, numeric)? else {
        return Ok(None);
    };
    if !(data_domain.0.is_finite() && data_domain.1.is_finite()) {
        return Ok(None);
    }
    // A discretizing binding (`Quantize`, `Threshold`, a binned color's
    // `BinOrdinal`) steps the cells through the one discretizing scale it
    // draws on any mark, over the cells' domain; its legend is that scale's
    // stepped bar.
    if let Some(enc) = binding {
        let sample = || {
            composite_samples(composite_domain)
                .unwrap_or_else(|| aux_numeric_values(reads, primary_batch, transform_outputs, false))
        };
        if let Some((buckets, bucket_warnings)) = build_discretizing_color_scale(enc, sample, data_domain, theme)? {
            warnings.extend(bucket_warnings);
            return Ok(Some(ColorScale::Discretizing(buckets)));
        }
    }
    let default_binding = crate::spec::encoding::EncodingSpec::default();
    let enc = binding.unwrap_or(&default_binding);
    let domain = binding
        .and_then(scale_explicit_domain)
        .or_else(|| composite_numeric_extent(composite_domain))
        .unwrap_or(data_domain);
    let raster = reads.iter().find(|read| read.is_raster_cells());
    let transform = super::ColorTransform::of_scale(enc.scale.as_ref())
        .or_else(|| raster.and_then(|read| read.raster_transform(primary_batch, transform_outputs)))
        .unwrap_or(super::ColorTransform::Linear);
    let (scheme, scheme_warnings) = resolve_continuous_scheme(enc, theme, domain, raster.and_then(AuxRead::cmap))?;
    warnings.extend(scheme_warnings);
    let midpoint = binding.and_then(scale_diverging_midpoint);
    Ok(Some(ColorScale::Continuous { domain, scheme, midpoint, transform }))
}

