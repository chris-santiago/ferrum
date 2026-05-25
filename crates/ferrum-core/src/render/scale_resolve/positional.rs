//! Positional scale resolution: build x/y axis scales from encoding specs.

use std::collections::HashMap;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;

use crate::scale::linear::LinearScale;
use crate::scale::log::LogScale;
use crate::scale::ordinal::OrdinalScale;
use crate::scale::pow::PowScale;
use crate::scale::symlog::SymlogScale;
use crate::scale::time::TimeScale;
use crate::spec::chart::ChartSpec;
use crate::spec::encoding::DataType as SpecDataType;

use crate::render::RenderError;

use super::domain::{apply_sort_to_domain, locate_field, numeric_domain_union};
use super::{distinct_values_in_order, infer_spec_type, ScaleKind};

/// Themes-T4 default inward padding fraction for quantitative / temporal
/// scales when neither `scale.padding` nor `scale.domain` is set.
const DEFAULT_SCALE_PADDING_FRAC: f64 = 0.05;

/// Themes-T4 maximum inward padding in pixels. The padding band is the
/// smaller of `fraction * span` and this cap so small panels don't lose
/// too much plot area.
const SCALE_PADDING_MAX_PX: f64 = 8.0;

/// Resolve the effective padding fraction.
///
/// Precedence:
///  - `Some(p)` → use `p` (including 0.0 to disable).
///  - `None && has_explicit_domain` → 0.0 (user-specified domain wins).
///  - `None && !has_explicit_domain` → `DEFAULT_SCALE_PADDING_FRAC`.
pub(in crate::render) fn resolve_padding_fraction(scale_padding: Option<f64>, has_explicit_domain: bool) -> f64 {
    if let Some(p) = scale_padding {
        return p;
    }
    if has_explicit_domain {
        return 0.0;
    }
    DEFAULT_SCALE_PADDING_FRAC
}

/// Inset a pixel range by the resolved padding band (capped at
/// `SCALE_PADDING_MAX_PX`). Handles inverted ranges (y-axis is
/// `(high_px, low_px)` for quantitative scales).
pub(in crate::render) fn inset_pixel_range(pr: (f64, f64), padding_frac: f64) -> (f64, f64) {
    if padding_frac == 0.0 {
        return pr;
    }
    let span = (pr.1 - pr.0).abs();
    let pad = (span * padding_frac).min(SCALE_PADDING_MAX_PX);
    let sign = if pr.1 >= pr.0 { 1.0 } else { -1.0 };
    (pr.0 + sign * pad, pr.1 - sign * pad)
}

pub(in crate::render) fn build_axis_scale(
    channel: &'static str,
    enc: &crate::spec::encoding::EncodingSpec,
    paired_enc: Option<&crate::spec::encoding::EncodingSpec>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    pixel_range: (f64, f64),
    spec: &ChartSpec,
) -> Result<ScaleKind, RenderError> {
    let located = locate_field(&enc.field, primary_batch, transform_outputs)
        .ok_or_else(|| RenderError::UnknownColumn { name: enc.field.clone() })?;
    let dtype = infer_spec_type(enc, located.col.data_type());
    let pr = axis_pixel_range(channel, &dtype, pixel_range);

    if let Some(scale_spec) = &enc.scale {
        return build_from_scale_spec(scale_spec, enc, located.batch, pr);
    }

    let inset = inset_pixel_range(pr, resolve_padding_fraction(None, false));

    match dtype {
        SpecDataType::Quantitative => {
            let (min, max) = numeric_domain_union(
                channel, &enc.field, paired_enc.map(|p| p.field.as_str()),
                primary_batch, transform_outputs, spec,
            )?;
            Ok(ScaleKind::Linear(LinearScale::new_internal(
                vec![min, max], vec![inset.0, inset.1], false, false,
            )))
        }
        SpecDataType::Ordinal | SpecDataType::Nominal => {
            let mut domain = distinct_values_in_order(located.batch, &enc.field)?;
            apply_sort_to_domain(&mut domain, enc.sort.as_ref());
            Ok(ScaleKind::Ordinal(OrdinalScale::new_internal(
                domain, vec![pr.0, pr.1], 0.0,
            )))
        }
        SpecDataType::Temporal => {
            let (min, max) = numeric_domain_union(
                channel, &enc.field, paired_enc.map(|p| p.field.as_str()),
                primary_batch, transform_outputs, spec,
            )?;
            Ok(ScaleKind::Time(TimeScale::new_internal(
                vec![min, max], vec![inset.0, inset.1], false, false,
            )))
        }
    }
}

/// Y-axis pixel range is inverted ONLY for quantitative/temporal scales.
pub(in crate::render) fn axis_pixel_range(channel: &str, dtype: &SpecDataType, pixel_range: (f64, f64)) -> (f64, f64) {
    if channel == "y" && !matches!(dtype, SpecDataType::Ordinal | SpecDataType::Nominal) {
        (pixel_range.1, pixel_range.0)
    } else {
        pixel_range
    }
}

/// Build a ScaleKind from an explicit ScaleSpec, using the given pixel range.
pub(in crate::render) fn build_from_scale_spec(
    scale_spec: &crate::spec::encoding::ScaleSpec,
    enc: &crate::spec::encoding::EncodingSpec,
    batch: &RecordBatch,
    pr: (f64, f64),
) -> Result<ScaleKind, RenderError> {
    use crate::spec::encoding::ScaleSpec;

    let col = batch
        .column_by_name(&enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: enc.field.clone() })?;

    Ok(match scale_spec {
        ScaleSpec::Linear { common, nice, zero } => {
            let (mut d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            if *zero && d.len() == 2 {
                if d[0] > 0.0 { d[0] = 0.0; }
                if d[1] < 0.0 { d[1] = 0.0; }
            }
            ScaleKind::Linear(LinearScale::new_internal(d, r, common.clamp, *nice))
        }
        ScaleSpec::Log { base, common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Log(LogScale::new_internal(d, r, *base, common.clamp, *nice))
        }
        ScaleSpec::Time { common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Time(TimeScale::new_internal(d, r, common.clamp, *nice))
        }
        ScaleSpec::Symlog { constant, common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Symlog(SymlogScale::new_internal(d, r, *constant, common.clamp, *nice))
        }
        ScaleSpec::Ordinal { domain, range, padding } => {
            let mut d = match domain {
                Some(d) => d.clone(),
                None => distinct_values_in_order(batch, &enc.field)?,
            };
            apply_sort_to_domain(&mut d, enc.sort.as_ref());
            ScaleKind::Ordinal(OrdinalScale::new_internal(
                d,
                range.clone().unwrap_or_else(|| vec![pr.0, pr.1]),
                *padding,
            ))
        }
        ScaleSpec::Pow { exponent, common } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Pow(PowScale::new_internal(d, r, *exponent, common.clamp))
        }
        ScaleSpec::Sqrt { common } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Pow(PowScale::new_internal(d, r, 0.5, common.clamp))
        }
        ScaleSpec::Utc { common, nice } => {
            let (d, r) = resolve_continuous_domain_and_range(&common.domain, &common.range, common.padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Time(TimeScale::new_internal(d, r, common.clamp, *nice))
        }
        ScaleSpec::Band { domain, padding, padding_inner, .. } => {
            let mut d = match domain {
                Some(d) => d.clone(),
                None => distinct_values_in_order(batch, &enc.field)?,
            };
            apply_sort_to_domain(&mut d, enc.sort.as_ref());
            let effective_padding = padding_inner.unwrap_or(*padding);
            ScaleKind::Ordinal(OrdinalScale::new_internal(
                d,
                vec![pr.0, pr.1],
                effective_padding,
            ))
        }
        ScaleSpec::Point { domain, padding, .. } => {
            let mut d = match domain {
                Some(d) => d.clone(),
                None => distinct_values_in_order(batch, &enc.field)?,
            };
            apply_sort_to_domain(&mut d, enc.sort.as_ref());
            ScaleKind::Ordinal(OrdinalScale::new_internal(
                d,
                vec![pr.0, pr.1],
                *padding,
            ))
        }
        ScaleSpec::Sequential { domain, .. }
        | ScaleSpec::Diverging { domain, .. }
        | ScaleSpec::Quantize { domain, .. } => {
            let (d, r) = resolve_continuous_domain_and_range(domain, &None, None, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Linear(LinearScale::new_internal(d, r, false, false))
        }
        ScaleSpec::BinOrdinal { .. } => {
            let (d, r) = resolve_continuous_domain_and_range(&None, &None, None, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Linear(LinearScale::new_internal(d, r, false, false))
        }
    })
}

/// Resolve `(domain, range)` for a continuous ScaleSpec variant.
fn resolve_continuous_domain_and_range(
    domain: &Option<Vec<f64>>,
    range: &Option<Vec<f64>>,
    padding: Option<f64>,
    col: &dyn Array,
    field: &str,
    pr: (f64, f64),
) -> Result<(Vec<f64>, Vec<f64>), RenderError> {
    let d = match domain {
        Some(d) => d.clone(),
        None => {
            let (mn, mx) = column_min_max_f64(col).map_err(|_| RenderError::UnsupportedDtype {
                field: field.to_string(),
                dtype: format!("{:?}", col.data_type()),
                context: Some("scale"),
            })?;
            vec![mn, mx]
        }
    };
    let r = if let Some(r) = range {
        r.clone()
    } else {
        let frac = resolve_padding_fraction(padding, domain.is_some());
        let inset = inset_pixel_range(pr, frac);
        vec![inset.0, inset.1]
    };
    Ok((d, r))
}

/// Apply coord-level domain and padding overrides to the x/y scales.
pub(in crate::render) fn apply_coord_domain_overrides(
    spec: &ChartSpec,
    x: &mut ScaleKind,
    y: &mut ScaleKind,
    x_pixel_range: (f64, f64),
    y_pixel_range: (f64, f64),
) {
    use crate::spec::coord::CoordKind;

    let (x_domain_override, y_domain_override, expand) = match &spec.coord {
        Some(CoordKind::Cartesian { x_domain, y_domain, expand, .. }) => {
            (*x_domain, *y_domain, *expand)
        }
        Some(CoordKind::Fixed { x_domain, y_domain, expand, .. }) => {
            (*x_domain, *y_domain, *expand)
        }
        _ => return,
    };

    if let Some((lo, hi)) = x_domain_override {
        let pr = if expand {
            inset_pixel_range(x_pixel_range, resolve_padding_fraction(None, false))
        } else {
            x_pixel_range
        };
        *x = ScaleKind::Linear(LinearScale::new_internal(
            vec![lo, hi], vec![pr.0, pr.1], false, false,
        ));
    } else if !expand {
        if let ScaleKind::Linear(ref inner) = x {
            let d = inner.domain_pair().to_vec();
            *x = ScaleKind::Linear(LinearScale::new_internal(
                d, vec![x_pixel_range.0, x_pixel_range.1], false, false,
            ));
        }
    }

    if let Some((lo, hi)) = y_domain_override {
        let pr = if expand {
            inset_pixel_range(y_pixel_range, resolve_padding_fraction(None, false))
        } else {
            y_pixel_range
        };
        *y = ScaleKind::Linear(LinearScale::new_internal(
            vec![lo, hi], vec![pr.1, pr.0], false, false,
        ));
    } else if !expand {
        if let ScaleKind::Linear(ref inner) = y {
            let d = inner.domain_pair().to_vec();
            *y = ScaleKind::Linear(LinearScale::new_internal(
                d, vec![y_pixel_range.1, y_pixel_range.0], false, false,
            ));
        }
    }
}

fn column_min_max_f64(col: &dyn Array) -> Result<(f64, f64), String> {
    crate::render::arrow_cast::min_max_f64(col)
}
