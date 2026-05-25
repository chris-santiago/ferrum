//! Auxiliary scale resolution: size, shape, and opacity encodings.

use arrow::record_batch::RecordBatch;

use crate::layout::ThemeInputs;
use crate::scale::linear::LinearScale;

use crate::render::RenderError;

use super::{column_min_max_f64, distinct_values_in_order, OpacityScale, ScaleKind, ShapeKind, ShapeScale, SizeScale, SHAPE_PALETTE};

/// Build a SizeScale if `encoding.size` is present.
///
/// Honors a user-supplied `scale.range` (Phase 10f); when absent, falls back
/// to `[theme.sizes.point_size_min, theme.sizes.point_size_max]`.
pub fn build_size_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    theme: &ThemeInputs,
) -> Result<Option<SizeScale>, RenderError> {
    let Some(size_enc) = &encoding.size else {
        return Ok(None);
    };
    let col = batch
        .column_by_name(&size_enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: size_enc.field.clone() })?;
    let (min, max) = column_min_max_f64(col).map_err(|_| RenderError::UnsupportedDtype {
        field: size_enc.field.clone(),
        dtype: format!("{:?}", col.data_type()),
        context: Some("size"),
    })?;
    let (lo, hi) = if let Some(crate::spec::encoding::ScaleSpec::Linear { common, .. })
        = &size_enc.scale
    {
        if let Some(r) = &common.range {
            if r.len() == 2 {
                (r[0], r[1])
            } else {
                (theme.sizes.point_size_min, theme.sizes.point_size_max)
            }
        } else {
            (theme.sizes.point_size_min, theme.sizes.point_size_max)
        }
    } else {
        (theme.sizes.point_size_min, theme.sizes.point_size_max)
    };
    let inner = ScaleKind::Linear(LinearScale::new_internal(
        vec![min, max],
        vec![lo, hi],
        false,
        true,
    ));
    let _ = (lo, hi); // bounds now read from inner.pixel_range() via accessors
    Ok(Some(SizeScale { inner }))
}

/// Build a ShapeScale if `encoding.shape` is present.
/// Returns the scale (if built) and an optional overflow warning.
pub fn build_shape_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
) -> Result<(Option<ShapeScale>, Option<crate::render::RenderWarning>), RenderError> {
    let Some(shape_enc) = &encoding.shape else {
        return Ok((None, None));
    };
    let distinct = distinct_values_in_order(batch, &shape_enc.field)?;
    let warn = if distinct.len() > SHAPE_PALETTE.len() {
        Some(crate::render::RenderWarning::ShapePaletteOverflowed {
            categories: distinct.len() as u32,
        })
    } else {
        None
    };
    let shapes: Vec<ShapeKind> = (0..distinct.len())
        .map(|i| SHAPE_PALETTE[i % SHAPE_PALETTE.len()])
        .collect();
    Ok((Some(ShapeScale { domain: distinct, shapes }), warn))
}

/// Build an OpacityScale if `encoding.opacity` is present.
pub fn build_opacity_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    theme: &ThemeInputs,
) -> Result<Option<OpacityScale>, RenderError> {
    let Some(op_enc) = &encoding.opacity else {
        return Ok(None);
    };
    let col = batch
        .column_by_name(&op_enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: op_enc.field.clone() })?;
    let (min, max) = column_min_max_f64(col).map_err(|_| RenderError::UnsupportedDtype {
        field: op_enc.field.clone(),
        dtype: format!("{:?}", col.data_type()),
        context: Some("opacity"),
    })?;
    let inner = ScaleKind::Linear(LinearScale::new_internal(
        vec![min, max],
        vec![theme.sizes.opacity_min, theme.sizes.opacity_max],
        true,
        false,
    ));
    Ok(Some(OpacityScale { inner }))
}
