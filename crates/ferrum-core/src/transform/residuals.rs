//! Shared residual-metrics & annotation-column emission for fitted-line
//! transforms (Smooth, Robust, and any future GLM/Logistic overlay).
//! Owns the canonical R²/RMSE/MAE formatting and the
//! `_metrics_text` / `_metrics_y` / `_ref_zero` schema columns so a
//! downstream `mark_text` / `mark_rule` overlay sees byte-identical
//! anchor semantics no matter which transform produced the batch.
//!
//! Lifted from `transform::smooth` during the SB-followup C4 commit
//! (2026-05-12) to collapse three duplicated emission blocks (smooth
//! Residuals / smooth Fitted / robust Residuals) into one home.

use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field};
use std::sync::Arc;

/// R² / RMSE / MAE over `(ys_obs, resids)`. Filters non-finite residuals
/// before aggregating. `R²` is `0.0` when `ys_obs` has zero variance.
///
/// Returns `(r2, rmse, mae)`. Used by both the OLS/LOESS path in
/// `transform::smooth` and the Huber-IRLS path in `transform::robust`
/// so the annotation text formats byte-identically across siblings.
pub(crate) fn compute(ys_obs: &[f64], resids: &[f64]) -> (f64, f64, f64) {
    let n_total = ys_obs.len() as f64;
    if n_total == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let mean_y = ys_obs.iter().sum::<f64>() / n_total;
    let ss_tot: f64 = ys_obs.iter().map(|y| (y - mean_y).powi(2)).sum();
    let mut ss_res = 0.0;
    let mut sum_abs = 0.0;
    let mut n_valid = 0.0;
    for r in resids {
        if r.is_finite() {
            ss_res += r * r;
            sum_abs += r.abs();
            n_valid += 1.0;
        }
    }
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let denom = if n_valid > 0.0 { n_valid } else { 1.0 };
    let rmse = (ss_res / denom).sqrt();
    let mae = sum_abs / denom;
    (r2, rmse, mae)
}

/// Append nullable `_metrics_text` (Utf8) + `_metrics_y` (Float64)
/// columns to `fields` / `cols` in place. The single non-null row sits
/// at the row with the max finite `xs` value; the anchor y is the max
/// finite of `y_anchor` (residuals for the Residuals path, fitted-y
/// for the Fitted path). When no finite x exists the columns are
/// emitted with all-null entries; when no finite y_anchor exists the
/// anchor y falls back to `0.0`.
///
/// Schema invariant: the text format is exactly
/// `"R² {r2:.3}\nRMSE {rmse:.3}\nMAE {mae:.3}"` — every downstream
/// `mark_text` overlay relies on that byte sequence (it shows up in
/// SVG goldens), so changes here must be coordinated with goldens.
///
/// `xs` is taken by slice so the caller can keep ownership of the
/// underlying `Vec`; the lengths of `xs` and `y_anchor` must match (n
/// rows). `cols` rows already pushed before this call should also be
/// length `xs.len()` — Arrow `RecordBatch::try_new` will reject
/// mismatches.
pub(crate) fn append_metrics_columns(
    fields: &mut Vec<Field>,
    cols: &mut Vec<ArrayRef>,
    xs: &[f64],
    y_anchor: &[f64],
    metrics: (f64, f64, f64),
) {
    let n = xs.len();
    let (r2, rmse, mae) = metrics;

    let anchor_idx = xs
        .iter()
        .enumerate()
        .filter(|(_, x)| x.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);

    let anchor_y_raw = y_anchor
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let anchor_y = if anchor_y_raw.is_finite() { anchor_y_raw } else { 0.0 };

    let text = format!("R² {r2:.3}\nRMSE {rmse:.3}\nMAE {mae:.3}");
    let mut text_col: Vec<Option<String>> = vec![None; n];
    let mut y_col: Vec<Option<f64>> = vec![None; n];
    if let Some(i) = anchor_idx {
        text_col[i] = Some(text);
        y_col[i] = Some(anchor_y);
    }
    fields.push(Field::new("_metrics_text", DataType::Utf8, true));
    cols.push(Arc::new(StringArray::from(text_col)));
    fields.push(Field::new("_metrics_y", DataType::Float64, true));
    cols.push(Arc::new(Float64Array::from(y_col)));
}

/// Append nullable `_ref_zero` (Float64) column of length `n` with
/// `Some(0.0)` on the first row and `None` on the rest. A downstream
/// `mark_rule(y="_ref_zero")` renders exactly one horizontal reference
/// line at y=0 without any extra Python-side injection.
pub(crate) fn append_zero_ref_column(
    fields: &mut Vec<Field>,
    cols: &mut Vec<ArrayRef>,
    n: usize,
) {
    let mut col: Vec<Option<f64>> = vec![None; n];
    if n > 0 {
        col[0] = Some(0.0);
    }
    fields.push(Field::new("_ref_zero", DataType::Float64, true));
    cols.push(Arc::new(Float64Array::from(col)));
}
