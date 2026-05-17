//! Data transform: LoessData — LOESS smoothing as a data transform.
//!
//! Fits a locally weighted polynomial regression (LOESS/LOWESS) to produce
//! a smooth curve output.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct LoessDataSpec {
    pub x: String,
    pub y: String,
    #[serde(default = "default_bandwidth")]
    pub bandwidth: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    #[serde(default = "default_loess_as")]
    pub as_: (String, String),
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_bandwidth() -> f64 {
    0.3
}
fn default_loess_as() -> (String, String) {
    ("x".into(), "y".into())
}

pub(crate) fn apply(spec: &LoessDataSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let x_idx = schema.index_of(&spec.x).map_err(|_| {
        PyValueError::new_err(format!("data_loess: column '{}' not found", spec.x))
    })?;
    let y_idx = schema.index_of(&spec.y).map_err(|_| {
        PyValueError::new_err(format!("data_loess: column '{}' not found", spec.y))
    })?;

    let x_col = batch
        .column(x_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!("data_loess: column '{}' must be Float64", spec.x))
        })?;
    let y_col = batch
        .column(y_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!("data_loess: column '{}' must be Float64", spec.y))
        })?;

    // Extract clean paired data.
    let n_rows = batch.num_rows();
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
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

    if xs.len() < 3 {
        let out_schema = Arc::new(Schema::new(vec![
            Field::new(&spec.as_.0, DataType::Float64, false),
            Field::new(&spec.as_.1, DataType::Float64, false),
        ]));
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
        ];
        return RecordBatch::try_new(out_schema, cols)
            .map_err(|e| PyValueError::new_err(format!("data_loess: {e}")));
    }

    // Sort by x.
    let mut pairs: Vec<(f64, f64)> = xs.iter().zip(ys.iter()).map(|(&x, &y)| (x, y)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_x: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
    let sorted_y: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();

    // Generate output grid.
    let n_out = 200;
    let x_min = sorted_x[0];
    let x_max = *sorted_x.last().unwrap();
    let step = (x_max - x_min) / (n_out - 1).max(1) as f64;
    let x_grid: Vec<f64> = (0..n_out).map(|i| x_min + i as f64 * step).collect();

    // LOESS: for each grid point, compute locally weighted linear regression.
    let n = sorted_x.len();
    let span = (spec.bandwidth * n as f64).ceil().max(3.0) as usize;
    let span = span.min(n);

    let mut y_grid: Vec<f64> = Vec::with_capacity(n_out);
    for &xg in &x_grid {
        let y_hat = loess_predict(xg, &sorted_x, &sorted_y, span);
        y_grid.push(y_hat);
    }

    let out_schema = Arc::new(Schema::new(vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(x_grid)),
        Arc::new(Float64Array::from(y_grid)),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("data_loess: {e}")))
}

/// Predict y at point x using LOESS with tri-cube kernel.
fn loess_predict(x: f64, xs: &[f64], ys: &[f64], span: usize) -> f64 {
    let n = xs.len();

    // Find the `span` nearest neighbors.
    let mut dists: Vec<(usize, f64)> = xs
        .iter()
        .enumerate()
        .map(|(i, &xi)| (i, (xi - x).abs()))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let neighbors = &dists[..span.min(n)];

    let max_dist = neighbors.last().map(|(_, d)| *d).unwrap_or(1.0);
    let h = if max_dist > 0.0 { max_dist } else { 1.0 };

    // Weighted linear regression: y = a + b * x with tri-cube weights.
    let mut sw = 0.0;
    let mut swx = 0.0;
    let mut swxx = 0.0;
    let mut swy = 0.0;
    let mut swxy = 0.0;

    for &(i, d) in neighbors {
        let u = d / h;
        let w = if u < 1.0 {
            let t = 1.0 - u * u * u;
            t * t * t
        } else {
            0.0
        };
        let xi = xs[i];
        let yi = ys[i];
        sw += w;
        swx += w * xi;
        swxx += w * xi * xi;
        swy += w * yi;
        swxy += w * xi * yi;
    }

    if sw < 1e-15 {
        return f64::NAN;
    }

    let det = sw * swxx - swx * swx;
    if det.abs() < 1e-15 {
        return swy / sw; // Constant fit.
    }

    let a = (swxx * swy - swx * swxy) / det;
    let b = (sw * swxy - swx * swy) / det;
    a + b * x
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "LoessData")]
#[derive(Debug, Clone)]
pub(crate) struct PyLoessData(pub(crate) TransformSpec);

#[pymethods]
impl PyLoessData {
    #[new]
    #[pyo3(signature = (x, y, *, bandwidth = 0.3, name = None))]
    fn new(x: String, y: String, bandwidth: f64, name: Option<String>) -> Self {
        PyLoessData(TransformSpec::LoessData(LoessDataSpec {
            x,
            y,
            bandwidth,
            groupby: None,
            as_: default_loess_as(),
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::LoessData(s) => format!("LoessData(x='{}', y='{}')", s.x, s.y),
            _ => "LoessData(?)".to_string(),
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
    fn loess_smooth_linear() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        // Perfect line: y = 2x
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
                Arc::new(Float64Array::from(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])),
            ],
        )
        .unwrap();

        let spec = LoessDataSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: 0.5,
            groupby: None,
            as_: ("x".into(), "y".into()),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 200);

        // LOESS on a perfect line should recover approximately y = 2x.
        let out_x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let out_y = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // Check midpoint: x~4.5, y should be ~9.0
        let mid = 100;
        let expected_y = out_x.value(mid) * 2.0;
        assert!(
            (out_y.value(mid) - expected_y).abs() < 0.5,
            "LOESS at x={} gave y={}, expected ~{}",
            out_x.value(mid),
            out_y.value(mid),
            expected_y
        );
    }
}
