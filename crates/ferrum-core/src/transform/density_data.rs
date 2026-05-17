//! Data transform: DensityData — KDE as a data transform.
//!
//! Produces a two-column output (x-grid, density) similar to the stat KDE
//! transform but exposed as a data transform with different parameters.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct DensityDataSpec {
    pub field: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bandwidth: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub cumulative: bool,
    #[serde(default = "default_density_as")]
    pub as_: (String, String),
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_density_as() -> (String, String) {
    ("value".into(), "density".into())
}

pub(crate) fn apply(spec: &DensityDataSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_density: column '{}' not found", spec.field))
    })?;

    if let Some(groupby) = &spec.groupby {
        return apply_grouped(spec, batch, idx, groupby);
    }

    apply_one_group(spec, batch, idx, None, None)
}

fn apply_grouped(
    spec: &DensityDataSpec,
    batch: &RecordBatch,
    field_idx: usize,
    groupby: &[String],
) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    // Only support single groupby column for simplicity.
    if groupby.is_empty() {
        return apply_one_group(spec, batch, field_idx, None, None);
    }

    let g_col_name = &groupby[0];
    let g_idx = schema.index_of(g_col_name).map_err(|_| {
        PyValueError::new_err(format!("data_density: groupby column '{g_col_name}' not found"))
    })?;
    let g_col = batch.column(g_idx);
    let g_arr = g_col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        PyValueError::new_err("data_density: groupby column must be Utf8")
    })?;

    // Partition rows by group.
    let mut groups: std::collections::BTreeMap<String, Vec<usize>> = std::collections::BTreeMap::new();
    for row in 0..n_rows {
        let key = if g_arr.is_null(row) {
            "__null__".to_string()
        } else {
            g_arr.value(row).to_string()
        };
        groups.entry(key).or_default().push(row);
    }

    // Run KDE per group and concatenate.
    let mut all_x: Vec<f64> = Vec::new();
    let mut all_y: Vec<f64> = Vec::new();
    let mut all_g: Vec<String> = Vec::new();

    for (group_key, rows) in &groups {
        let result = apply_one_group(spec, batch, field_idx, Some(rows), None)?;
        let x_col = result.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let y_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..result.num_rows() {
            all_x.push(x_col.value(i));
            all_y.push(y_col.value(i));
            all_g.push(group_key.clone());
        }
    }

    let out_schema = Arc::new(Schema::new(vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
        Field::new(g_col_name, DataType::Utf8, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(all_x)),
        Arc::new(Float64Array::from(all_y)),
        Arc::new(StringArray::from(all_g)),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("data_density: {e}")))
}

fn apply_one_group(
    spec: &DensityDataSpec,
    batch: &RecordBatch,
    field_idx: usize,
    only_rows: Option<&[usize]>,
    _shared_extent: Option<(f64, f64)>,
) -> PyResult<RecordBatch> {
    let col = batch
        .column(field_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "data_density: column '{}' must be Float64",
                spec.field
            ))
        })?;

    // Extract clean values.
    let mut data: Vec<f64> = Vec::new();
    match only_rows {
        Some(rows) => {
            for &r in rows {
                if !col.is_null(r) {
                    let v = col.value(r);
                    if !v.is_nan() {
                        data.push(v);
                    }
                }
            }
        }
        None => {
            for i in 0..col.len() {
                if !col.is_null(i) {
                    let v = col.value(i);
                    if !v.is_nan() {
                        data.push(v);
                    }
                }
            }
        }
    }

    if data.is_empty() {
        let out_schema = Arc::new(Schema::new(vec![
            Field::new(&spec.as_.0, DataType::Float64, false),
            Field::new(&spec.as_.1, DataType::Float64, false),
        ]));
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
        ];
        return RecordBatch::try_new(out_schema, cols)
            .map_err(|e| PyValueError::new_err(format!("data_density: {e}")));
    }

    let n = data.len() as f64;
    let steps = spec.steps.unwrap_or(200);

    // Bandwidth: Silverman's rule if not specified.
    let bw = spec.bandwidth.unwrap_or_else(|| {
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = var.sqrt();
        1.06 * std_dev * n.powf(-0.2)
    });

    // Extent.
    let data_min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let data_max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let (lo, hi) = spec.extent.unwrap_or((data_min - 3.0 * bw, data_max + 3.0 * bw));

    // Generate x grid.
    let step_size = (hi - lo) / (steps - 1).max(1) as f64;
    let x_grid: Vec<f64> = (0..steps).map(|i| lo + i as f64 * step_size).collect();

    // Compute KDE.
    let mut density: Vec<f64> = vec![0.0; steps];
    let norm = 1.0 / (n * bw * (2.0 * std::f64::consts::PI).sqrt());

    for &xi in &data {
        for (j, &xj) in x_grid.iter().enumerate() {
            let z = (xj - xi) / bw;
            density[j] += norm * (-0.5 * z * z).exp();
        }
    }

    // Cumulative if requested.
    if spec.cumulative {
        let dx = step_size;
        let mut cumsum = 0.0;
        for d in density.iter_mut() {
            cumsum += *d * dx;
            *d = cumsum;
        }
    }

    let out_schema = Arc::new(Schema::new(vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(x_grid)),
        Arc::new(Float64Array::from(density)),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("data_density: {e}")))
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "DensityData")]
#[derive(Debug, Clone)]
pub(crate) struct PyDensityData(pub(crate) TransformSpec);

#[pymethods]
impl PyDensityData {
    #[new]
    #[pyo3(signature = (field, *, name = None))]
    fn new(field: String, name: Option<String>) -> Self {
        PyDensityData(TransformSpec::DensityData(DensityDataSpec {
            field,
            bandwidth: None,
            groupby: None,
            extent: None,
            steps: None,
            cumulative: false,
            as_: default_density_as(),
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::DensityData(s) => format!("DensityData(field='{}')", s.field),
            _ => "DensityData(?)".to_string(),
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
    fn density_basic() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                1.0, 2.0, 3.0, 4.0, 5.0,
            ]))],
        )
        .unwrap();

        let spec = DensityDataSpec {
            field: "x".into(),
            bandwidth: Some(1.0),
            groupby: None,
            extent: Some((0.0, 6.0)),
            steps: Some(50),
            cumulative: false,
            as_: ("value".into(), "density".into()),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 50);
        assert_eq!(out.num_columns(), 2);
        assert_eq!(out.schema().field(0).name(), "value");
        assert_eq!(out.schema().field(1).name(), "density");

        // Density should be positive.
        let d = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..d.len() {
            assert!(d.value(i) >= 0.0);
        }
    }
}
