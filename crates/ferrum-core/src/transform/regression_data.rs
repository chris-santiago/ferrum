//! Data transform: RegressionData — regression as a data transform.
//!
//! Fits a regression model (linear, polynomial, or exponential/log/power)
//! to the data and outputs the fitted line as a two-column batch.

use arrow::array::{ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::group_key::{
    group_partition_multi, groupby_field_nullable, materialize_groupby_col, KeyValue,
};
use crate::transform::numeric_util::coerce_xy_pairs_at;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RegressionDataSpec {
    pub x: String,
    pub y: String,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default = "default_order")]
    pub order: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    #[serde(default = "default_regression_as")]
    pub as_: (String, String),
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_method() -> String {
    "linear".into()
}
fn default_order() -> usize {
    1
}
fn default_regression_as() -> (String, String) {
    ("x".into(), "y".into())
}

/// Grid size shared by [`apply_one_group`] and [`all_nan_output`] — kept as
/// one constant so the two can't drift apart.
const N_OUT: usize = 100;

pub(crate) fn apply(spec: &RegressionDataSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    match &spec.groupby {
        Some(cols) if !cols.is_empty() => apply_grouped(spec, batch, cols),
        _ => apply_one_group(spec, batch, None),
    }
}

/// Fit one regression over `batch`, restricted to `only_indices` when given.
/// Ingests via [`coerce_xy_pairs_at`] so a grouped partition and the
/// ungrouped path (`only_indices = None`) share one cleaning walk (T2/M1).
fn apply_one_group(
    spec: &RegressionDataSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
) -> PyResult<RecordBatch> {
    // Extract clean paired data (any numeric dtype; non-numeric raises).
    let (xs, ys) = coerce_xy_pairs_at(batch, &spec.x, &spec.y, "data_regression", only_indices)?;

    if xs.len() < 2 {
        return all_nan_output(spec);
    }

    // Fit the model.
    let coeffs = match spec.method.as_str() {
        "linear" | "poly" => fit_polynomial(&xs, &ys, spec.order)?,
        "exp" => fit_exponential(&xs, &ys)?,
        "log" => fit_log(&xs, &ys)?,
        "pow" => fit_power(&xs, &ys)?,
        other => {
            return Err(PyValueError::new_err(format!(
                "data_regression: unknown method '{other}'"
            )));
        }
    };

    // Generate output points along the x range.
    let x_min = xs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let step = (x_max - x_min) / (N_OUT - 1).max(1) as f64;

    let mut out_x: Vec<f64> = Vec::with_capacity(N_OUT);
    let mut out_y: Vec<f64> = Vec::with_capacity(N_OUT);

    for i in 0..N_OUT {
        let xv = x_min + i as f64 * step;
        let yv = predict(xv, &coeffs, &spec.method);
        out_x.push(xv);
        out_y.push(yv);
    }

    let out_schema = Arc::new(Schema::new(vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(out_x)),
        Arc::new(Float64Array::from(out_y)),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("data_regression: {e}")))
}

/// Partition `batch` by `group_cols`, fit each partition independently via
/// [`apply_one_group`], then stack the per-group grids into one batch with
/// the group columns appended at their original dtype (XFORM-03) as
/// additional fields — the `smooth.rs:186-275` multi-column grouped template
/// (spec §4.3 / decision-doc M3a).
///
/// Each group's output grid spans that group's OWN x range (`x_min`/`x_max`
/// derived from its own rows inside `apply_one_group`), not a range shared
/// across groups — this mirrors `stat_smooth`'s grouped behavior (see
/// `smooth.rs::apply_one_group`, which folds `data_x_min`/`data_x_max` from
/// the group-restricted `xs`), so a per-group fit line only ever draws over
/// the x-values that group actually has.
///
/// A group with fewer than 2 usable (non-null, non-NaN) rows cannot support a
/// fit; `apply_one_group` returns [`all_nan_output`] for it rather than an
/// empty batch, so that group still contributes [`N_OUT`] rows and keeps its
/// group key in the concatenated output — the same choice `smooth.rs` makes
/// for its below-threshold groups. A silent 0-row drop would remove the group
/// from the color-scale domain and legend entirely; a NaN-filled row set
/// draws no visible curve (the line mark filters non-finite points) but keeps
/// the group's identity intact everywhere else.
fn apply_grouped(
    spec: &RegressionDataSpec,
    batch: &RecordBatch,
    group_cols: &[String],
) -> PyResult<RecordBatch> {
    let (group_order, group_idx_map, gtypes) =
        group_partition_multi(batch, group_cols, "data_regression")?;

    let mut all_x: Vec<f64> = Vec::new();
    let mut all_y: Vec<f64> = Vec::new();
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::new();

    for g in &group_order {
        let ixs = group_idx_map.get(g).ok_or_else(|| {
            PyValueError::new_err(format!("data_regression: missing group key {g:?} in index map"))
        })?;
        let out = apply_one_group(spec, batch, Some(ixs))?;
        let out_x = out.column(0).as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
            PyValueError::new_err("data_regression: expected Float64Array in grouped output")
        })?;
        let out_y = out.column(1).as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
            PyValueError::new_err("data_regression: expected Float64Array in grouped output")
        })?;
        for i in 0..out.num_rows() {
            all_x.push(out_x.value(i));
            all_y.push(out_y.value(i));
            group_keys_out.push(g.clone());
        }
    }

    let mut out_fields = vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
    ];
    let mut out_cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(all_x)),
        Arc::new(Float64Array::from(all_y)),
    ];

    for (gi, (col_name, dtype)) in group_cols.iter().zip(gtypes.iter()).enumerate() {
        out_fields.push(Field::new(col_name, dtype.clone(), groupby_field_nullable()));
        let group_col_arr = materialize_groupby_col(&group_keys_out, gi, dtype)
            .map_err(PyValueError::new_err)?;
        out_cols.push(group_col_arr);
    }

    let out_schema = Arc::new(Schema::new(out_fields));
    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("data_regression: {e}")))
}

/// Below-fit-threshold (`xs.len() < 2`) output: mirrors
/// `smooth.rs::all_nan_output` by emitting [`N_OUT`] NaN-valued rows instead
/// of an empty batch.
///
/// **Serves both grouped and ungrouped paths:** Under `apply_grouped`, a
/// below-threshold group contributes its full [`N_OUT`]-row count and keeps
/// its group key when concatenated — it stays in the output schema, the
/// color-scale domain, and the legend, rather than silently vanishing (see
/// the `apply_grouped` doc comment for the full rationale). Under the
/// ungrouped path (`apply_one_group(..., None)`), an entire below-threshold
/// input frame likewise emits [`N_OUT`] NaN rows rather than 0 rows, which
/// means such a chart renders with a fabricated fallback 0–1 domain for the
/// axes (since there is no real x-range to draw from empty input) — no error,
/// no warning, but also no visible curve (the line mark filters non-finite
/// points before building polylines).
fn all_nan_output(spec: &RegressionDataSpec) -> PyResult<RecordBatch> {
    let out_schema = Arc::new(Schema::new(vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![f64::NAN; N_OUT])),
        Arc::new(Float64Array::from(vec![f64::NAN; N_OUT])),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("data_regression: {e}")))
}

/// Fit polynomial of given order using least squares (normal equations).
fn fit_polynomial(xs: &[f64], ys: &[f64], order: usize) -> PyResult<Vec<f64>> {
    let n = xs.len();
    let p = order + 1;

    // Build Vandermonde matrix X and solve X'X b = X'y.
    // For small orders this is fine via direct solve.
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];

    for i in 0..n {
        let mut xi_pow = 1.0;
        for j in 0..p {
            xty[j] += xi_pow * ys[i];
            let mut xi_pow2 = 1.0;
            for k in 0..p {
                xtx[j * p + k] += xi_pow * xi_pow2;
                xi_pow2 *= xs[i];
            }
            xi_pow *= xs[i];
        }
    }

    // Solve via Gaussian elimination.
    solve_linear_system(&mut xtx, &mut xty, p)
        .map_err(|_| PyValueError::new_err("data_regression: singular matrix in polynomial fit"))
}

fn fit_exponential(xs: &[f64], ys: &[f64]) -> PyResult<Vec<f64>> {
    // y = a * exp(b * x) → ln(y) = ln(a) + b*x (linear in log space).
    let log_ys: Vec<f64> = ys.iter().map(|&y| if y > 0.0 { y.ln() } else { f64::NAN }).collect();
    let clean: Vec<(f64, f64)> = xs
        .iter()
        .zip(log_ys.iter())
        .filter(|(_, ly)| !ly.is_nan())
        .map(|(&x, &ly)| (x, ly))
        .collect();
    if clean.len() < 2 {
        return Ok(vec![1.0, 0.0]); // Fallback.
    }
    let cxs: Vec<f64> = clean.iter().map(|(x, _)| *x).collect();
    let cys: Vec<f64> = clean.iter().map(|(_, y)| *y).collect();
    let coeffs = fit_polynomial(&cxs, &cys, 1)?;
    // coeffs[0] = ln(a), coeffs[1] = b
    Ok(vec![coeffs[0].exp(), coeffs[1]])
}

fn fit_log(xs: &[f64], ys: &[f64]) -> PyResult<Vec<f64>> {
    // y = a + b * ln(x)
    let log_xs: Vec<f64> = xs.iter().map(|&x| if x > 0.0 { x.ln() } else { f64::NAN }).collect();
    let clean: Vec<(f64, f64)> = log_xs
        .iter()
        .zip(ys.iter())
        .filter(|(lx, _)| !lx.is_nan())
        .map(|(&lx, &y)| (lx, y))
        .collect();
    if clean.len() < 2 {
        return Ok(vec![0.0, 0.0]);
    }
    let cxs: Vec<f64> = clean.iter().map(|(x, _)| *x).collect();
    let cys: Vec<f64> = clean.iter().map(|(_, y)| *y).collect();
    fit_polynomial(&cxs, &cys, 1)
}

fn fit_power(xs: &[f64], ys: &[f64]) -> PyResult<Vec<f64>> {
    // y = a * x^b → ln(y) = ln(a) + b * ln(x)
    let clean: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys.iter())
        .filter(|(&x, &y)| x > 0.0 && y > 0.0)
        .map(|(&x, &y)| (x.ln(), y.ln()))
        .collect();
    if clean.len() < 2 {
        return Ok(vec![1.0, 0.0]);
    }
    let cxs: Vec<f64> = clean.iter().map(|(x, _)| *x).collect();
    let cys: Vec<f64> = clean.iter().map(|(_, y)| *y).collect();
    let coeffs = fit_polynomial(&cxs, &cys, 1)?;
    Ok(vec![coeffs[0].exp(), coeffs[1]])
}

fn predict(x: f64, coeffs: &[f64], method: &str) -> f64 {
    match method {
        "linear" | "poly" => {
            let mut y = 0.0;
            let mut x_pow = 1.0;
            for &c in coeffs {
                y += c * x_pow;
                x_pow *= x;
            }
            y
        }
        "exp" => {
            // coeffs = [a, b]; y = a * exp(b * x)
            if coeffs.len() >= 2 {
                coeffs[0] * (coeffs[1] * x).exp()
            } else {
                f64::NAN
            }
        }
        "log" => {
            // coeffs = [a, b]; y = a + b * ln(x)
            if coeffs.len() >= 2 && x > 0.0 {
                coeffs[0] + coeffs[1] * x.ln()
            } else {
                f64::NAN
            }
        }
        "pow" => {
            // coeffs = [a, b]; y = a * x^b
            if coeffs.len() >= 2 && x > 0.0 {
                coeffs[0] * x.powf(coeffs[1])
            } else {
                f64::NAN
            }
        }
        _ => f64::NAN,
    }
}

/// Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &mut [f64], b: &mut [f64], n: usize) -> Result<Vec<f64>, ()> {
    // Forward elimination.
    for col in 0..n {
        // Find pivot.
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(());
        }
        // Swap rows.
        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }
        // Eliminate below.
        for row in (col + 1)..n {
            let factor = a[row * n + col] / a[col * n + col];
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
    // Back substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum / a[i * n + i];
    }
    Ok(x)
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "RegressionData")]
#[derive(Debug, Clone)]
pub(crate) struct PyRegressionData(pub(crate) TransformSpec);

#[pymethods]
impl PyRegressionData {
    #[new]
    #[pyo3(signature = (x, y, *, method = "linear", order = 1, name = None))]
    fn new(x: String, y: String, method: &str, order: usize, name: Option<String>) -> Self {
        PyRegressionData(TransformSpec::RegressionData(RegressionDataSpec {
            x,
            y,
            method: method.into(),
            order,
            groupby: None,
            as_: default_regression_as(),
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::RegressionData(s) => format!("RegressionData(x='{}', y='{}')", s.x, s.y),
            _ => "RegressionData(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn regression_linear_basic() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        // Perfect line: y = 2x + 1
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 4.0])),
                Arc::new(Float64Array::from(vec![1.0, 3.0, 5.0, 7.0, 9.0])),
            ],
        )
        .unwrap();

        let spec = RegressionDataSpec {
            x: "x".into(),
            y: "y".into(),
            method: "linear".into(),
            order: 1,
            groupby: None,
            as_: ("x".into(), "y".into()),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 100);

        // Check first and last points.
        let out_x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let out_y = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // At x=0, y should be ~1.0
        assert!((out_y.value(0) - 1.0).abs() < 1e-6);
        // At x=4, y should be ~9.0
        assert!((out_y.value(99) - 9.0).abs() < 1e-6);
        assert!((out_x.value(0) - 0.0).abs() < 1e-10);
        assert!((out_x.value(99) - 4.0).abs() < 1e-10);
    }

    // ── M3a groupby coverage (F-L05-03) ──────────────────────────────────────

    fn spec_with_groupby(groupby: Option<Vec<String>>) -> RegressionDataSpec {
        RegressionDataSpec {
            x: "x".into(),
            y: "y".into(),
            method: "linear".into(),
            order: 1,
            groupby,
            as_: ("rx".into(), "ry".into()),
            name: None,
        }
    }

    #[test]
    fn regression_grouped_fits_each_group_independently() {
        use arrow::array::StringArray;
        // Group A: y = 2x (exact). Group B: y = 10 - 3x (exact).
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A", "A", "A", "A", "B", "B", "B", "B"])),
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![0.0, 2.0, 4.0, 6.0, 10.0, 7.0, 4.0, 1.0])),
            ],
        )
        .unwrap();

        let spec = spec_with_groupby(Some(vec!["g".into()]));
        let out = apply(&spec, &batch).unwrap();

        // Schema: rx, ry, g (group column appended, XFORM-03 dtype-preserved).
        assert_eq!(out.num_columns(), 3);
        assert_eq!(out.schema().field(2).name(), "g");
        assert_eq!(out.schema().field(2).data_type(), &DataType::Utf8);
        // 100 grid points per group, first-appearance order (A, B).
        assert_eq!(out.num_rows(), 200);

        let rx = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let ry = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let g = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();

        // Group A occupies rows [0, 100): y = 2x exactly.
        assert_eq!(g.value(0), "A");
        assert!((rx.value(0) - 0.0).abs() < 1e-9);
        assert!((ry.value(0) - 0.0).abs() < 1e-6, "A at x=0 should be y=0, got {}", ry.value(0));
        assert!((rx.value(99) - 3.0).abs() < 1e-9);
        assert!((ry.value(99) - 6.0).abs() < 1e-6, "A at x=3 should be y=6, got {}", ry.value(99));

        // Group B occupies rows [100, 200): y = 10 - 3x exactly.
        assert_eq!(g.value(100), "B");
        assert!((rx.value(100) - 0.0).abs() < 1e-9);
        assert!((ry.value(100) - 10.0).abs() < 1e-6, "B at x=0 should be y=10, got {}", ry.value(100));
        assert!((rx.value(199) - 3.0).abs() < 1e-9);
        assert!((ry.value(199) - 1.0).abs() < 1e-6, "B at x=3 should be y=1, got {}", ry.value(199));

        // Neither per-group fit matches the blended global line (y = 5 - 0.5x),
        // proving these are independent per-group fits, not one shared curve.
        let global_at_0 = 5.0;
        assert!((ry.value(0) - global_at_0).abs() > 1.0);
        assert!((ry.value(100) - global_at_0).abs() > 1.0);
    }

    #[test]
    fn regression_grouped_int_group_dtype_preserved() {
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![0_i64, 0, 1, 1])),
                Arc::new(Float64Array::from(vec![0.0, 1.0, 0.0, 1.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0, 0.0, 2.0])),
            ],
        )
        .unwrap();

        let spec = spec_with_groupby(Some(vec!["g".into()]));
        let out = apply(&spec, &batch).unwrap();

        assert_eq!(out.schema().field(2).data_type(), &DataType::Int64);
        let g = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        // First-appearance order: group 0 then group 1.
        assert_eq!(g.value(0), 0);
        assert_eq!(g.value(100), 1);
    }

    #[test]
    fn regression_grouped_multi_column_partitions_distinct_combos() {
        use arrow::array::{Int64Array, StringArray};
        let schema = Arc::new(Schema::new(vec![
            Field::new("g1", DataType::Utf8, false),
            Field::new("g2", DataType::Int64, false),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        // Two distinct (g1, g2) combos, each with 4 rows of exact linear data.
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A", "A", "A", "A", "B", "B", "B", "B"])),
                Arc::new(Int64Array::from(vec![1_i64, 1, 1, 1, 2, 2, 2, 2])),
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0])),
            ],
        )
        .unwrap();

        let spec = spec_with_groupby(Some(vec!["g1".into(), "g2".into()]));
        let out = apply(&spec, &batch).unwrap();

        // Schema: rx, ry, g1, g2 — both groupby columns appended, in order.
        assert_eq!(out.num_columns(), 4);
        assert_eq!(out.schema().field(2).name(), "g1");
        assert_eq!(out.schema().field(3).name(), "g2");
        assert_eq!(out.schema().field(3).data_type(), &DataType::Int64);
        // 2 distinct combos × 100 grid points each.
        assert_eq!(out.num_rows(), 200);
    }

    #[test]
    fn regression_grouped_below_threshold_group_emits_nan_rows_with_group_key() {
        use arrow::array::StringArray;
        // Group A: 4 healthy rows (exact linear fit, y = 2x). Group B: 1 row —
        // below the <2-point fit threshold. Per the `apply_grouped` doc
        // comment (mirroring `smooth.rs::all_nan_output`), B must still
        // contribute N_OUT rows and keep its group key, not vanish.
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A", "A", "A", "A", "B"])),
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 5.0])),
                Arc::new(Float64Array::from(vec![0.0, 2.0, 4.0, 6.0, 9.0])),
            ],
        )
        .unwrap();

        let spec = spec_with_groupby(Some(vec!["g".into()]));
        let out = apply(&spec, &batch).unwrap();

        // Group A contributes N_OUT real rows; group B contributes N_OUT NaN
        // rows — the group is not dropped from the output.
        assert_eq!(out.num_rows(), 2 * N_OUT);
        assert_eq!(out.num_columns(), 3);

        let ry = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let g = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(g.value(0), "A");
        assert!(!ry.value(0).is_nan(), "healthy group A must produce real fit values");

        assert_eq!(g.value(N_OUT), "B");
        for i in N_OUT..2 * N_OUT {
            assert!(ry.value(i).is_nan(), "below-threshold group B row {i} should be NaN, got {}", ry.value(i));
            assert_eq!(g.value(i), "B", "below-threshold group B must keep its group key at row {i}");
        }
    }

    #[test]
    fn regression_groupby_none_and_empty_vec_are_both_ungrouped() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![0.0, 2.0, 4.0, 6.0])),
            ],
        )
        .unwrap();

        let out_none = apply(&spec_with_groupby(None), &batch).unwrap();
        let out_empty = apply(&spec_with_groupby(Some(vec![])), &batch).unwrap();

        assert_eq!(out_none.num_columns(), 2, "ungrouped output has no group column");
        assert_eq!(out_empty.num_columns(), 2, "empty groupby list is also ungrouped");
        assert_eq!(out_none.num_rows(), out_empty.num_rows());
    }

    #[test]
    fn regression_ungrouped_below_threshold_emits_nan_rows() {
        // Ungrouped input with 1 row — below the <2-point fit threshold.
        // Per the `all_nan_output` doc comment, the ungrouped path emits [`N_OUT`]
        // NaN rows rather than 0 rows, mirroring the grouped-degenerate behavior
        // and `smooth.rs` semantics. No error, no warning, but no visible curve
        // (line mark filters non-finite points).
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![5.0])),
                Arc::new(Float64Array::from(vec![9.0])),
            ],
        )
        .unwrap();

        let spec = spec_with_groupby(None);
        let out = apply(&spec, &batch).unwrap();

        // Below-threshold ungrouped input should emit N_OUT NaN rows, not 0 rows.
        assert_eq!(out.num_rows(), N_OUT, "ungrouped below-threshold must emit N_OUT rows");
        assert_eq!(out.num_columns(), 2, "ungrouped output has no group column");

        let out_y = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..out.num_rows() {
            assert!(
                out_y.value(i).is_nan(),
                "ungrouped below-threshold output row {i} should be NaN, got {}",
                out_y.value(i)
            );
        }
    }
}
