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
use super::{distinct_values_in_order, infer_spec_type, numeric_extent, ColorScale};

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
pub fn build_color_scale(
    encoding: &crate::spec::encoding::Encoding,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    theme: &ThemeInputs,
) -> Result<(Option<ColorScale>, Vec<crate::render::RenderWarning>), RenderError> {
    let Some(c_enc) = &encoding.color else {
        return Ok((None, Vec::new()));
    };
    let located = locate_field(&c_enc.field, primary_batch, transform_outputs)
        .ok_or_else(|| RenderError::UnknownColumn { name: c_enc.field.clone() })?;

    let inferred = infer_spec_type(c_enc, located.col.data_type());
    let is_continuous_color = matches!(
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
        let (lo, hi) = numeric_extent(located.col);
        use crate::render::color::{ContinuousScheme, NamedContinuous};

        // D4: resolve scheme from (1) encoding.scheme, (2) encoding.scale.common.scheme,
        // (3) theme, (4) Viridis fallback.
        let scale_scheme: Option<&str> = scale_common_scheme(c_enc);
        let scheme = c_enc
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
        Ok((Some(ColorScale::Continuous { domain: (lo, hi), scheme }), Vec::new()))
    } else {
        // Data-aware sort (channel shorthand `"-y"`, sort-field objects) reorders
        // the legend domain by an aggregate, mirroring the positional-axis path.
        // The category column is the color field; candidate value columns live in
        // the primary batch alongside it.
        let sort_ctx = SortContext {
            category_field: &c_enc.field,
            batch: primary_batch,
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
                    // falling back to data first-appearance order.
                    let mut domain = match explicit_ordinal_domain(c_enc) {
                        Some(declared) => declared,
                        None => distinct_values_in_order(primary_batch, &c_enc.field)?,
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
                    let mut domain = distinct_values_in_order(primary_batch, &c_enc.field)?;
                    apply_sort_to_domain(&mut domain, c_enc.sort.as_ref(), &sort_ctx, &mut warnings);
                    let scale = build_default_categorical_scale(domain, c_enc, theme, &mut warnings);
                    return Ok((Some(scale), warnings));
                }
            }
        }

        // Default path: look up a named categorical palette.
        let mut warnings: Vec<crate::render::RenderWarning> = Vec::new();
        let mut domain = distinct_values_in_order(primary_batch, &c_enc.field)?;
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

/// Extract the `scheme` string from a continuous `ScaleSpec`'s `common` field.
///
/// Returns the first `Some(scheme)` found across the continuous scale variants
/// that embed `ContinuousScaleCommon`, or `None` for ordinal/categorical variants
/// and when no scheme is set.
fn scale_common_scheme(enc: &crate::spec::encoding::EncodingSpec) -> Option<&str> {
    use crate::spec::encoding::ScaleSpec;
    match enc.scale.as_ref()? {
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
