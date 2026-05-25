//! Color scale resolution: categorical and continuous color encoding.

use std::borrow::Cow;
use std::collections::HashMap;

use arrow::record_batch::RecordBatch;

use crate::layout::ThemeInputs;
use crate::spec::encoding::DataType as SpecDataType;

use crate::render::color::Color;
use crate::render::palette;
use crate::render::RenderError;

use super::domain::locate_field;
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
/// Returns `(scale, optional palette-overflow warning)` so the caller
/// can fold the warning into its accumulator alongside the
/// build_shape_scale warning.
pub fn build_color_scale(
    encoding: &crate::spec::encoding::Encoding,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    theme: &ThemeInputs,
) -> Result<(Option<ColorScale>, Option<crate::render::RenderWarning>), RenderError> {
    let Some(c_enc) = &encoding.color else {
        return Ok((None, None));
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
        let scheme = c_enc
            .scheme
            .as_deref()
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
        Ok((Some(ColorScale::Continuous { domain: (lo, hi), scheme }), None))
    } else {
        let domain = distinct_values_in_order(primary_batch, &c_enc.field)?;
        let resolved_name: &str = c_enc.scheme.as_deref().unwrap_or(&theme.palette.color_scheme);
        let static_palette: &'static [Color] = if palette::is_sequential_scheme(resolved_name) {
            palette::categorical_palette("tableau10")
        } else {
            palette::categorical_palette(resolved_name)
        };
        let palette: Cow<'static, [Color]> = Cow::Borrowed(static_palette);
        let warn = (domain.len() > palette.len()).then(|| {
            crate::render::RenderWarning::ColorPaletteOverflowed { categories: domain.len() as u32 }
        });
        Ok((Some(ColorScale::Categorical { domain, palette }), warn))
    }
}
