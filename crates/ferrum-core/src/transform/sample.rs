//! Data transform: Sample — deterministic random row sampling.
//!
//! Uses seeded ChaCha8Rng for byte-deterministic output (per CLAUDE.md
//! randomness contract).

use arrow::array::{RecordBatch, UInt32Array};
use arrow::compute::take;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct SampleSpec {
    /// Number of rows to sample.
    pub n: usize,
    /// RNG seed for deterministic sampling.
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_seed() -> u64 {
    42
}

pub(crate) fn apply(spec: &SampleSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let n_rows = batch.num_rows();

    if spec.n == 0 {
        return Err(PyValueError::new_err("data_sample: n must be > 0"));
    }

    // If n >= num_rows, return the full batch.
    if spec.n >= n_rows {
        return Ok(batch.clone());
    }

    // Seeded shuffle + take first n.
    let mut rng = ChaCha8Rng::seed_from_u64(spec.seed);
    let mut indices: Vec<u32> = (0..n_rows as u32).collect();
    indices.shuffle(&mut rng);
    indices.truncate(spec.n);
    // Sort indices to preserve original row order.
    indices.sort_unstable();

    let idx_array = UInt32Array::from(indices);
    let schema = batch.schema();
    let columns: Vec<_> = (0..batch.num_columns())
        .map(|i| take(batch.column(i).as_ref(), &idx_array, None))
        .collect::<Result<_, _>>()
        .map_err(|e| PyValueError::new_err(format!("data_sample: take failed: {e}")))?;

    RecordBatch::try_new(schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_sample: {e}")))
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "Sample")]
#[derive(Debug, Clone)]
pub(crate) struct PySample(pub(crate) TransformSpec);

#[pymethods]
impl PySample {
    #[new]
    #[pyo3(signature = (n, *, seed = 42, name = None))]
    fn new(n: usize, seed: u64, name: Option<String>) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("Sample: n must be > 0"));
        }
        Ok(PySample(TransformSpec::Sample(SampleSpec { n, seed, name })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Sample(s) => format!("Sample(n={})", s.n),
            _ => "Sample(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn sample_produces_correct_count() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            ]))],
        )
        .unwrap();

        let spec = SampleSpec {
            n: 3,
            seed: 42,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 3);
    }

    #[test]
    fn sample_is_deterministic() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            ]))],
        )
        .unwrap();

        let spec = SampleSpec {
            n: 5,
            seed: 123,
            name: None,
        };
        let out1 = apply(&spec, &batch).unwrap();
        let out2 = apply(&spec, &batch).unwrap();
        let x1 = out1.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let x2 = out2.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..5 {
            assert_eq!(x1.value(i), x2.value(i));
        }
    }

    #[test]
    fn sample_n_exceeds_rows_returns_all() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]))],
        )
        .unwrap();

        let spec = SampleSpec {
            n: 100,
            seed: 42,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 3);
    }
}
