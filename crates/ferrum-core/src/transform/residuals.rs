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

use arrow::array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The names a residual-mode step's own computed columns are published
/// under. Internal: never a user kwarg on `Smooth`/`Robust` (set only
/// through the private `_with_residual_columns` pymethod both pyclasses
/// carry), so the figure layer can give a computed column a fresh name
/// when the caller's data already uses the default one, without ever
/// renaming the caller's own frame — the caller's column keeps its own
/// name and values in every door that reads the caller's rows. The
/// fitted path never reads this field; only `output="residuals"` does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ResidualColumns {
    #[serde(skip_serializing_if = "is_default_residual_x")]
    pub(crate) x: String,
    #[serde(skip_serializing_if = "is_default_residual_residual")]
    pub(crate) residual: String,
    #[serde(skip_serializing_if = "is_default_residual_ref_zero")]
    pub(crate) ref_zero: String,
    #[serde(skip_serializing_if = "is_default_residual_metrics_text")]
    pub(crate) metrics_text: String,
    #[serde(skip_serializing_if = "is_default_residual_metrics_y")]
    pub(crate) metrics_y: String,
}

// The one owner of every default residual-column name: `Default`, the
// five `is_default_residual_*` wire-omission checks below, and the Fitted
// path's metrics columns (`append_null_metrics_columns` here,
// `smooth::fits::build_smooth_batch` and `robust::build_fitted_batch`,
// neither of which reads `residual_columns`) all read these constants
// instead of re-spelling the literals — so `Default` and an
// `is_default_residual_*` check can never disagree about what "default"
// means.
pub(crate) const DEFAULT_RESIDUAL_X: &str = "x";
pub(crate) const DEFAULT_RESIDUAL_RESIDUAL: &str = "residual";
pub(crate) const DEFAULT_RESIDUAL_REF_ZERO: &str = "_ref_zero";
pub(crate) const DEFAULT_RESIDUAL_METRICS_TEXT: &str = "_metrics_text";
pub(crate) const DEFAULT_RESIDUAL_METRICS_Y: &str = "_metrics_y";

// One `skip_serializing_if` function per field (rather than a single
// struct-level skip) so a spec with ONLY ONE name set — the realistic
// shape, since `_with_residual_columns` only ever changes the names a
// collision actually forces — serializes just that one field, not the
// whole struct; `#[serde(default)]` on the container still fills in every
// OTHER field from `ResidualColumns::default()` on the way back in.
fn is_default_residual_x(v: &str) -> bool { v == DEFAULT_RESIDUAL_X }
fn is_default_residual_residual(v: &str) -> bool { v == DEFAULT_RESIDUAL_RESIDUAL }
fn is_default_residual_ref_zero(v: &str) -> bool { v == DEFAULT_RESIDUAL_REF_ZERO }
fn is_default_residual_metrics_text(v: &str) -> bool { v == DEFAULT_RESIDUAL_METRICS_TEXT }
fn is_default_residual_metrics_y(v: &str) -> bool { v == DEFAULT_RESIDUAL_METRICS_Y }

impl Default for ResidualColumns {
    fn default() -> Self {
        Self {
            x: DEFAULT_RESIDUAL_X.to_string(),
            residual: DEFAULT_RESIDUAL_RESIDUAL.to_string(),
            ref_zero: DEFAULT_RESIDUAL_REF_ZERO.to_string(),
            metrics_text: DEFAULT_RESIDUAL_METRICS_TEXT.to_string(),
            metrics_y: DEFAULT_RESIDUAL_METRICS_Y.to_string(),
        }
    }
}

impl ResidualColumns {
    /// `true` for the stock names: the wire omits the field at this value,
    /// so every spec written before this setting existed round-trips
    /// byte-identically.
    pub(crate) fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Refuses an empty name, or the same name given to two of these
    /// columns. `ctx` is the caller's own voice (`"stat_smooth"`/`"Robust"`).
    pub(crate) fn check(&self, ctx: &str) -> PyResult<()> {
        let names = [&self.x, &self.residual, &self.ref_zero, &self.metrics_text, &self.metrics_y];
        for name in names {
            if name.is_empty() {
                return Err(PyValueError::new_err(format!("{ctx}: a residual column name cannot be empty")));
            }
        }
        let mut seen: Vec<&str> = Vec::with_capacity(names.len());
        for name in names {
            if seen.contains(&name.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "{ctx}: residual column name '{name}' is given to more than one column"
                )));
            }
            seen.push(name.as_str());
        }
        Ok(())
    }
}

/// The caller's rows a residual batch carries, at the surviving row
/// positions, and what it must not carry twice: `batch`'s `group_cols`
/// (the grouped wrapper's stacking template re-appends them once, after
/// every group's own batch is built) and the predictor's own input
/// column, but only when it is literally named like the computed `x`
/// column this call emits. `predictor_field` is always the caller's OWN
/// column name, never a demanded log axis's log-mapped private one — see
/// `smooth::apply_in_data_space`'s doc for why that distinction matters.
pub(crate) struct ResidualCarry<'a> {
    pub(crate) batch: &'a RecordBatch,
    pub(crate) row_idx: &'a [usize],
    pub(crate) predictor_field: &'a str,
    pub(crate) group_cols: &'a [String],
}

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
    metrics_text_name: &str,
    metrics_y_name: &str,
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
    fields.push(Field::new(metrics_text_name, DataType::Utf8, true));
    cols.push(Arc::new(StringArray::from(text_col)));
    fields.push(Field::new(metrics_y_name, DataType::Float64, true));
    cols.push(Arc::new(Float64Array::from(y_col)));
}

/// Append nullable `_metrics_text` (Utf8) / `_metrics_y` (Float64) columns
/// of length `n`, entirely `null` — batch-D T9 round 4. Used when
/// `inject_metrics` was requested but this particular fit degenerated
/// (rank-deficient design, zero-variance x, non-convergent IRLS, or fewer
/// than 2 usable points), so there is no anchor row to compute. Unlike
/// [`append_metrics_columns`], which places one non-null row at the
/// max-finite-x anchor, this keeps the two columns present but every row
/// null — so a degenerate fit's output schema matches its well-supported
/// siblings' (`x`/`y`/`ci_lower`/`ci_upper` plus these two), rather than
/// varying with the data. The schema no longer diverging at the source is
/// what let `render::prepare::concat_batches_by_name` and
/// `transform::smooth::apply_grouped`'s by-name reconciliation become a
/// belt-and-braces safety net instead of the primary defense.
pub(crate) fn append_null_metrics_columns(fields: &mut Vec<Field>, cols: &mut Vec<ArrayRef>, n: usize) {
    fields.push(Field::new(DEFAULT_RESIDUAL_METRICS_TEXT, DataType::Utf8, true));
    cols.push(Arc::new(StringArray::from(vec![None::<&str>; n])));
    fields.push(Field::new(DEFAULT_RESIDUAL_METRICS_Y, DataType::Float64, true));
    cols.push(Arc::new(Float64Array::from(vec![None::<f64>; n])));
}

/// Append nullable `_ref_zero` (Float64) column of length `n` with
/// `Some(0.0)` on the first row and `None` on the rest. A downstream
/// `mark_rule(y="_ref_zero")` renders exactly one horizontal reference
/// line at y=0 without any extra Python-side injection.
pub(crate) fn append_zero_ref_column(
    fields: &mut Vec<Field>,
    cols: &mut Vec<ArrayRef>,
    n: usize,
    name: &str,
) {
    let mut col: Vec<Option<f64>> = vec![None; n];
    if n > 0 {
        col[0] = Some(0.0);
    }
    fields.push(Field::new(name, DataType::Float64, true));
    cols.push(Arc::new(Float64Array::from(col)));
}

/// Append every column of `carry.batch` the caller's input row held, at
/// exactly the surviving positions `carry.row_idx` names, in order, to
/// `fields`/`cols` — the `Bin(per_row=True)` row-preserving contract
/// (`crate::transform::bin::apply_per_row`) applied to a residuals-mode
/// batch: nothing here is computed, each value is the caller's own row,
/// untouched (and, under a demanded log axis, read from `batch` before any
/// forward mapping — only a fresh private column carries the mapped
/// values, so a caller's own column is never touched by it; that private
/// column IS always carried here, since it is present in the mapped batch
/// and matches neither the excluded predictor name nor an emitted one —
/// `log_space::run_in_log_space`'s own post-body filter is what strips it
/// afterward, since it is always one of the names it tracks as private).
///
/// The emitted set — the output columns this call actually emits — and
/// the excluded predictor name are both derived from `fields` as the
/// caller already built it (the computed `x`/`residual`, and whichever of
/// `_ref_zero`/`_metrics_text`/`_metrics_y` the call requested), rather
/// than hand-listed here: a caller cannot forget to declare one of its own
/// names, because there is nothing left to declare. `fields`'s first entry
/// is always the computed `x` column, under whatever name the step's own
/// `ResidualColumns` setting gives it (refused, rather than indexed
/// unchecked, when a caller ever passes an empty `fields`); a carried
/// column is excluded, not refused, when its name matches that `x` column
/// AND it is the actual predictor (`carry.predictor_field` equals that
/// name) — the one column whose carried content would be byte-identical
/// to the computed one. `carry.group_cols` are excluded the same way
/// (`apply_grouped`'s stacking template re-appends them once, after every
/// group's own batch is built). Any other input column named like one of
/// the emitted names is refused rather than shadowed, mirroring
/// `apply_per_row`'s own collision text.
pub(crate) fn append_carried_columns(
    ctx: &str,
    fields: &mut Vec<Field>,
    cols: &mut Vec<ArrayRef>,
    carry: &ResidualCarry,
) -> PyResult<()> {
    let emitted: Vec<String> = fields.iter().map(|f| f.name().clone()).collect();
    let Some(computed_x) = fields.first().map(|f| f.name().clone()) else {
        return Err(PyValueError::new_err(format!(
            "{ctx}: internal error — append_carried_columns called before any computed column \
             was built; this is a bug in ferrum, not a user error"
        )));
    };
    let take_indices = UInt64Array::from(carry.row_idx.iter().map(|&i| i as u64).collect::<Vec<u64>>());
    for (i, field) in carry.batch.schema().fields().iter().enumerate() {
        let name = field.name().as_str();
        if carry.group_cols.iter().any(|g| g == name) {
            continue;
        }
        if name == computed_x && carry.predictor_field == computed_x {
            continue;
        }
        if emitted.iter().any(|e| e == name) {
            return Err(PyValueError::new_err(format!(
                "{ctx}: residual output column '{name}' collides with an input column of the \
                 same name; rename the input column"
            )));
        }
        let column = arrow::compute::take(carry.batch.column(i), &take_indices, None)
            .map_err(|e| PyValueError::new_err(format!("{ctx}: {e}")))?;
        fields.push(field.as_ref().clone());
        cols.push(column);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Schema;

    /// `append_carried_columns`' `pub(crate)` signature doesn't rule out a
    /// caller passing an empty `fields` — this must refuse with a typed
    /// error, not index `fields[0]` unchecked and panic.
    #[test]
    fn append_carried_columns_with_no_computed_fields_yet_is_a_typed_error_not_a_panic() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
        ]).unwrap();
        let mut fields: Vec<Field> = Vec::new();
        let mut cols: Vec<ArrayRef> = Vec::new();
        let carry = ResidualCarry { batch: &batch, row_idx: &[0, 1], predictor_field: "v", group_cols: &[] };
        let err = append_carried_columns("Test", &mut fields, &mut cols, &carry)
            .expect_err("an empty fields vec must be refused, not indexed unchecked");
        assert!(err.to_string().contains("internal error"), "{err}");
    }
}
