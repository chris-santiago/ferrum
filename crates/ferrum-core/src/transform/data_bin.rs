//! Data transform: DataBin — binning as a data transform.
//!
//! Adds a bin column to the input batch (unlike stat_bin which collapses
//! to bin counts). Output is the original batch plus a "{field}_bin" column
//! containing the bin start value for each row.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct DataBinSpec {
    pub field: String,
    /// Output column name. Defaults to "{field}_bin".
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub as_: Option<String>,
    /// Maximum number of bins.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub maxbins: Option<usize>,
    /// Explicit bin width (overrides maxbins).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub step: Option<f64>,
    /// Whether to "nice" the bin boundaries.
    #[serde(default = "default_nice")]
    pub nice: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_nice() -> bool {
    true
}

pub(crate) fn apply(spec: &DataBinSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let col_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_bin: column '{}' not found", spec.field))
    })?;
    let col = batch
        .column(col_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!("data_bin: column '{}' must be Float64", spec.field))
        })?;

    let n_rows = batch.num_rows();

    // Compute data extent.
    let mut data_min = f64::INFINITY;
    let mut data_max = f64::NEG_INFINITY;
    for i in 0..n_rows {
        if col.is_null(i) {
            continue;
        }
        let v = col.value(i);
        if v.is_nan() {
            continue;
        }
        if v < data_min {
            data_min = v;
        }
        if v > data_max {
            data_max = v;
        }
    }

    if data_min > data_max {
        // All null/NaN — produce NaN bin column.
        let bin_values = vec![f64::NAN; n_rows];
        return build_output(spec, batch, bin_values);
    }

    // Determine bin width.
    let bin_width = if let Some(step) = spec.step {
        step
    } else {
        let maxbins = spec.maxbins.unwrap_or(10);
        let raw_width = (data_max - data_min) / maxbins as f64;
        if spec.nice {
            nice_step(raw_width)
        } else {
            raw_width
        }
    };

    if bin_width <= 0.0 || !bin_width.is_finite() {
        let bin_values = vec![data_min; n_rows];
        return build_output(spec, batch, bin_values);
    }

    // Compute bin start for the extent.
    let bin_start = if spec.nice {
        (data_min / bin_width).floor() * bin_width
    } else {
        data_min
    };

    // Assign each row to a bin.
    let mut bin_values: Vec<f64> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        if col.is_null(i) {
            bin_values.push(f64::NAN);
        } else {
            let v = col.value(i);
            if v.is_nan() {
                bin_values.push(f64::NAN);
            } else {
                let bin_idx = ((v - bin_start) / bin_width).floor();
                bin_values.push(bin_start + bin_idx * bin_width);
            }
        }
    }

    build_output(spec, batch, bin_values)
}

fn build_output(spec: &DataBinSpec, batch: &RecordBatch, bin_values: Vec<f64>) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let out_name = spec
        .as_
        .clone()
        .unwrap_or_else(|| format!("{}_bin", spec.field));

    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new(&out_name, DataType::Float64, true));
    let out_schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns.push(Arc::new(Float64Array::from(bin_values)));

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_bin: {e}")))
}

/// Round a raw step size to a "nice" number (1, 2, 5 × 10^k).
fn nice_step(raw: f64) -> f64 {
    if raw <= 0.0 || !raw.is_finite() {
        return 1.0;
    }
    let exp = raw.log10().floor();
    let frac = raw / 10.0_f64.powf(exp);
    let nice_frac = if frac <= 1.5 {
        1.0
    } else if frac <= 3.0 {
        2.0
    } else if frac <= 7.0 {
        5.0
    } else {
        10.0
    };
    nice_frac * 10.0_f64.powf(exp)
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "DataBin")]
#[derive(Debug, Clone)]
pub(crate) struct PyDataBin(pub(crate) TransformSpec);

#[pymethods]
impl PyDataBin {
    #[new]
    #[pyo3(signature = (field, *, as_ = None, maxbins = None, step = None, nice = true, name = None))]
    fn new(
        field: String,
        as_: Option<String>,
        maxbins: Option<usize>,
        step: Option<f64>,
        nice: bool,
        name: Option<String>,
    ) -> Self {
        PyDataBin(TransformSpec::DataBin(DataBinSpec {
            field,
            as_,
            maxbins,
            step,
            nice,
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::DataBin(s) => format!("DataBin(field='{}')", s.field),
            _ => "DataBin(?)".to_string(),
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
    fn data_bin_assigns_bins() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
            ]))],
        )
        .unwrap();

        let spec = DataBinSpec {
            field: "x".into(),
            as_: Some("x_bin".into()),
            maxbins: Some(5),
            step: None,
            nice: true,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 2);
        assert_eq!(out.schema().field(1).name(), "x_bin");
        assert_eq!(out.num_rows(), 10);

        // All bin values should be finite.
        let bins = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..bins.len() {
            assert!(bins.value(i).is_finite());
        }
    }

    #[test]
    fn data_bin_explicit_step() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]))],
        )
        .unwrap();

        let spec = DataBinSpec {
            field: "x".into(),
            as_: None,
            maxbins: None,
            step: Some(2.0),
            nice: false,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let bins = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // With step=2 starting at 1.0: bins at 1.0, 1.0, 3.0, 3.0, 5.0
        assert_eq!(bins.value(0), 1.0);
        assert_eq!(bins.value(1), 1.0);
        assert_eq!(bins.value(2), 3.0);
        assert_eq!(bins.value(3), 3.0);
        assert_eq!(bins.value(4), 5.0);
    }
}
