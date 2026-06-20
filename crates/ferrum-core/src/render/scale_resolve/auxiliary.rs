//! Auxiliary scale resolution: size, shape, and opacity encodings.

use std::collections::HashMap;

use arrow::record_batch::RecordBatch;

use crate::layout::ThemeInputs;
use crate::scale::linear::LinearScale;

use crate::render::RenderError;

use super::{column_min_max_f64, distinct_values_in_order, shared_categorical_batch, union_panel_with_global_extent, OpacityScale, ScaleKind, ShapeKind, ShapeScale, SizeScale, SHAPE_PALETTE};

/// Build a SizeScale if `encoding.size` is present.
///
/// Honors a user-supplied `scale.range` (Phase 10f); when absent, falls back
/// to `[theme.sizes.point_size_min, theme.sizes.point_size_max]`.
///
/// `facet_shared`: when `true` (chart is faceted with no independent option for
/// size), unions `batch`'s extent with the global `FINAL_OUTPUT_KEY` batch so
/// that per-panel marks normalize against the same domain as the global legend.
/// Non-faceted callers pass `false`; the per-panel-only path is byte-identical.
pub fn build_size_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
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
    // T3: When faceted (Shared), union the per-panel extent with the global
    // FINAL_OUTPUT_KEY batch so marks scale through the same domain as the legend.
    let (min, max) = if facet_shared {
        union_panel_with_global_extent((min, max), &size_enc.field, transform_outputs)
    } else {
        (min, max)
    };
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
///
/// `facet_shared`: when `true` (chart is faceted), resolves the categorical
/// domain from the global `FINAL_OUTPUT_KEY` batch so that every panel assigns
/// the same glyph to the same category string — matching the global shape legend.
/// Falls back to `batch` when the global batch or field is absent.
/// Non-faceted callers pass `false`; the per-panel-only path is byte-identical.
pub fn build_shape_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
) -> Result<(Option<ShapeScale>, Option<crate::render::RenderWarning>), RenderError> {
    let Some(shape_enc) = &encoding.shape else {
        return Ok((None, None));
    };
    // T3-shape: when faceted (Shared), resolve the domain from the global
    // FINAL_OUTPUT_KEY batch so every panel's glyph assignment matches the
    // global shape legend. Falls back to `batch` when the global batch or
    // field is absent (non-faceted path is byte-identical: facet_shared=false).
    let domain_batch = shared_categorical_batch(batch, &shape_enc.field, transform_outputs, facet_shared);
    let distinct = distinct_values_in_order(domain_batch, &shape_enc.field)?;
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
///
/// `facet_shared`: when `true` (chart is faceted with no independent option for
/// opacity), unions `batch`'s extent with the global `FINAL_OUTPUT_KEY` batch so
/// that per-panel marks normalize against the same domain as the global legend.
/// Non-faceted callers pass `false`; the per-panel-only path is byte-identical.
pub fn build_opacity_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    facet_shared: bool,
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
    // T3: When faceted (Shared), union the per-panel extent with the global
    // FINAL_OUTPUT_KEY batch so marks scale through the same domain as the legend.
    let (min, max) = if facet_shared {
        union_panel_with_global_extent((min, max), &op_enc.field, transform_outputs)
    } else {
        (min, max)
    };
    let inner = ScaleKind::Linear(LinearScale::new_internal(
        vec![min, max],
        vec![theme.sizes.opacity_min, theme.sizes.opacity_max],
        true,
        false,
    ));
    Ok(Some(OpacityScale { inner }))
}

