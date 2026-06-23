//! Data transform: RegressionData — regression as a data transform.
//!
//! Fits a regression model (linear, polynomial, or exponential/log/power)
//! to the data and outputs the fitted line as a two-column batch.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

pub(crate) fn apply(spec: &RegressionDataSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let x_idx = schema.index_of(&spec.x).map_err(|_| {
        PyValueError::new_err(format!("data_regression: column '{}' not found", spec.x))
    })?;
    let y_idx = schema.index_of(&spec.y).map_err(|_| {
        PyValueError::new_err(format!("data_regression: column '{}' not found", spec.y))
    })?;

    let x_col = batch
        .column(x_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!("data_regression: column '{}' must be Float64", spec.x))
        })?;
    let y_col = batch
        .column(y_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!("data_regression: column '{}' must be Float64", spec.y))
        })?;

    // Extract clean paired data.
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let n_rows = batch.num_rows();
    for i in 0..n_rows {
        if x_col.is_null(i) || y_col.is_null(i) {
            continue;
        }
        let xv = x_col.value(i);
        let yv = y_col.value(i);
        if xv.is_nan() || yv.is_nan() {
            continue;
        }
        xs.push(xv);
        ys.push(yv);
    }

    if xs.len() < 2 {
        return empty_result(spec);
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
    let n_out = 100;
    let step = (x_max - x_min) / (n_out - 1).max(1) as f64;

    let mut out_x: Vec<f64> = Vec::with_capacity(n_out);
    let mut out_y: Vec<f64> = Vec::with_capacity(n_out);

    for i in 0..n_out {
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

fn empty_result(spec: &RegressionDataSpec) -> PyResult<RecordBatch> {
    let out_schema = Arc::new(Schema::new(vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(Vec::<f64>::new())),
        Arc::new(Float64Array::from(Vec::<f64>::new())),
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
    use arrow::array::Float64Array;
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
}
