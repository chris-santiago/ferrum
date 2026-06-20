//! Color scale resolution: categorical and continuous color encoding.

use std::borrow::Cow;
use std::collections::HashMap;

use arrow::record_batch::RecordBatch;

use crate::layout::ThemeInputs;
use crate::spec::encoding::DataType as SpecDataType;

use crate::render::color::Color;
use crate::render::palette;
use crate::render::RenderError;

use super::domain::{apply_sort_to_domain, locate_field, SortContext};
use super::{distinct_values_in_order, infer_spec_type, numeric_extent, shared_categorical_batch, union_panel_with_global_extent, ColorScale};

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
pub fn build_color_scale(
    encoding: &crate::spec::encoding::Encoding,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    theme: &ThemeInputs,
    force_categorical: bool,
    facet_shared: bool,
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
        // D1: honor explicit domain from Sequential/Diverging scale specs.
        // When the spec carries domain=[lo, hi] or domain=[lo, mid, hi], use
        // those bounds instead of auto-inferring from the data column. When the
        // spec domain is absent, fall back to the data extent.
        let (lo, hi) = scale_explicit_domain(c_enc).unwrap_or(data_extent);
        use crate::render::color::{ContinuousScheme, NamedContinuous};

        // D1/D4: resolve scheme from (1) encoding.scheme, (2) encoding.scale
        // Sequential/Diverging scheme field OR common.scheme on other variants,
        // (3) theme, (4) Viridis fallback.
        let scale_scheme: Option<&str> = scale_common_scheme(c_enc);
        let mut scheme = c_enc
            .scheme
            .as_deref()
            .or(scale_scheme)
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
            });
        // D1: honor the `reverse` flag on Sequential scale specs.
        if scale_spec_is_reversed(c_enc) {
            scheme = ContinuousScheme::Reverse(Box::new(scheme));
        }
        // Gap 2: resolve the diverging midpoint.  Priority:
        //   1. explicit `domain_mid`/`domainMid` field on DivergingScale spec
        //   2. middle element of a 3-tuple domain=[lo, mid, hi]
        //   3. None — geometric center falls out from pure-linear normalization
        // Sequential scales always get None (pure-linear).
        let midpoint = scale_diverging_midpoint(c_enc);
        Ok((Some(ColorScale::Continuous { domain: (lo, hi), scheme, midpoint }), Vec::new()))
    } else {
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
        let mut warnings: Vec<crate::render::RenderWarning> = Vec::new();
        let mut domain = distinct_values_in_order(domain_batch, &c_enc.field)?;
        apply_sort_to_domain(&mut domain, c_enc.sort.as_ref(), &sort_ctx, &mut warnings);
        let scale = build_default_categorical_scale(domain, c_enc, theme, &mut warnings);
        Ok((Some(scale), warnings))
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
/// Covers both the `Sequential`/`Diverging` variants (which carry their own
/// `scheme` field) and the `ContinuousScaleCommon`-bearing variants (Linear,
/// Log, Time, Symlog, Pow, Sqrt, Utc). Returns `None` for ordinal/categorical
/// variants and when no scheme is set.
fn scale_common_scheme(enc: &crate::spec::encoding::EncodingSpec) -> Option<&str> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
        // D1: Sequential and Diverging carry their own scheme field.
        ScaleSpec::Sequential { scheme, .. }
        | ScaleSpec::Diverging { scheme, .. } => scheme.as_deref(),
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

/// Extract the explicit numeric domain from a `Sequential` or `Diverging`
/// scale spec.
///
/// For `Sequential`: returns `Some((domain[0], domain[1]))` when a 2-element
/// (or longer) domain is declared.
///
/// For `Diverging`: accepts both 2-element `[lo, hi]` and 3-element
/// `[lo, mid, hi]` domains; always returns the outer bounds `(lo, hi)`.
/// The midpoint is carried separately by `scale_diverging_midpoint` and
/// threaded into `ColorScale::Continuous { midpoint }` for piecewise-linear
/// normalization.
///
/// Returns `None` for all other scale types and when no domain is declared,
/// allowing the caller to fall back to the auto-inferred data extent.
fn scale_explicit_domain(enc: &crate::spec::encoding::EncodingSpec) -> Option<(f64, f64)> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
        ScaleSpec::Sequential { domain: Some(d), .. } if d.len() >= 2 => {
            Some((d[0], d[d.len() - 1]))
        }
        ScaleSpec::Diverging { domain: Some(d), .. } if d.len() >= 2 => {
            // 3-element [lo, mid, hi]: take the outer bounds.
            // 2-element [lo, hi]: use directly.
            Some((d[0], d[d.len() - 1]))
        }
        _ => None,
    }
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
/// Returns `Some(Vec<String>)` when the encoding has `scale = ScaleSpec::Ordinal`
/// with a string-array `range` (D1 path).  Returns `None` for all other scale
/// types and when the range is absent or numeric.
fn explicit_string_range(enc: &crate::spec::encoding::EncodingSpec) -> Option<Vec<String>> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
        ScaleSpec::Ordinal { range, .. } => {
            crate::scale::ordinal::OrdinalRangeValue::all_strings(range.as_deref()?)
        }
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
