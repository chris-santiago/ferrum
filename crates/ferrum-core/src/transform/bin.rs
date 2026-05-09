use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::scale::ticks::sturges_floor;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct BinSpec {
    pub field: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bin_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bin_width: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64)>,
    #[serde(default = "default_true")]
    pub nice: bool,
}

fn default_true() -> bool { true }

pub(crate) fn apply(spec: &BinSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!(
            "stat_bin: column '{}' not found; available: {:?}",
            spec.field,
            schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        ))
    })?;
    let field = schema.field(idx);
    if field.data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_bin: column '{}' must be Float64; got {:?}",
            spec.field, field.data_type()
        )));
    }
    let arr = batch
        .column(idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("dtype check above guarantees Float64Array");

    // Drop nulls and NaN
    let mut clean: Vec<f64> = Vec::with_capacity(arr.len());
    for i in 0..arr.len() {
        if !arr.is_null(i) {
            let v = arr.value(i);
            if !v.is_nan() {
                clean.push(v);
            }
        }
    }

    // Empty input → empty output (per spec §6: stat_bin is the exception that allows empty)
    if clean.is_empty() {
        return empty_bin_output();
    }

    let (lo, hi) = match spec.extent {
        Some((a, b)) if a < b => (a, b),
        Some((a, b)) => return Err(PyValueError::new_err(format!(
            "stat_bin: extent must satisfy lo < hi; got ({a}, {b})"
        ))),
        None => {
            let (lo, hi) = clean.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
            if lo == hi {
                // Spec §4.1 numeric edge: all-equal → single unit bin
                return single_unit_bin(lo, clean.len() as u64);
            }
            (lo, hi)
        }
    };

    let n_bins: usize = match (spec.bin_count, spec.bin_width) {
        (Some(c), _) if c > 0 => c,
        (None, Some(w)) if w > 0.0 => ((hi - lo) / w).ceil().max(1.0) as usize,
        _ => sturges_floor(clean.len()),
    };

    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| lo + (hi - lo) * (i as f64) / (n_bins as f64))
        .collect();

    let mut counts = vec![0u64; n_bins];
    for v in &clean {
        if *v < lo || *v > hi { continue; }
        // Last edge is inclusive; otherwise [lo, hi) per bin.
        let pos = if *v == hi {
            n_bins - 1
        } else {
            let raw = ((*v - lo) / (hi - lo) * (n_bins as f64)).floor() as usize;
            raw.min(n_bins - 1)
        };
        counts[pos] += 1;
    }

    let total = clean.len() as f64;
    let bin_starts: Vec<f64> = (0..n_bins).map(|i| edges[i]).collect();
    let bin_ends:   Vec<f64> = (0..n_bins).map(|i| edges[i + 1]).collect();
    let densities:  Vec<f64> = counts
        .iter()
        .zip(bin_starts.iter().zip(bin_ends.iter()))
        .map(|(c, (s, e))| (*c as f64) / (total * (e - s)))
        .collect();

    build_bin_batch(bin_starts, bin_ends, counts, densities)
}

fn build_bin_batch(
    starts: Vec<f64>,
    ends: Vec<f64>,
    counts: Vec<u64>,
    densities: Vec<f64>,
) -> PyResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bin_start", DataType::Float64, false),
        Field::new("bin_end",   DataType::Float64, false),
        Field::new("count",     DataType::UInt64,  false),
        Field::new("density",   DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(starts)),
        Arc::new(Float64Array::from(ends)),
        Arc::new(UInt64Array::from(counts)),
        Arc::new(Float64Array::from(densities)),
    ];
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_bin: {e}")))
}

fn empty_bin_output() -> PyResult<RecordBatch> {
    build_bin_batch(Vec::new(), Vec::new(), Vec::new(), Vec::new())
}

fn single_unit_bin(v: f64, count: u64) -> PyResult<RecordBatch> {
    let start = v - 0.5;
    let end   = v + 0.5;
    let density = (count as f64) / ((count as f64) * (end - start));
    build_bin_batch(vec![start], vec![end], vec![count], vec![density])
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, UInt64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch_with(values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    fn col_f64<'a>(b: &'a RecordBatch, name: &str) -> &'a Float64Array {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
    }

    fn col_u64<'a>(b: &'a RecordBatch, name: &str) -> &'a UInt64Array {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
    }

    #[test]
    fn test_bin_basic_counts_match_numpy_histogram() {
        // numpy.histogram([1,2,3,4,5,6,7,8,9,10], bins=5, range=(1,10))
        // edges: [1.0, 2.8, 4.6, 6.4, 8.2, 10.0]
        // counts: [2, 2, 2, 2, 2]   (10 inclusive captured by upper-edge convention)
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(5),
            bin_width: None,
            extent: Some((1.0, 10.0)),
            nice: false,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 5);
        let counts = col_u64(&out, "count");
        for i in 0..5 {
            assert_eq!(counts.value(i), 2, "bin {i} count: got {}", counts.value(i));
        }
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        for i in 0..5 {
            let expected_start = 1.0 + 1.8 * i as f64;
            let expected_end = expected_start + 1.8;
            assert!((starts.value(i) - expected_start).abs() < 1e-9);
            assert!((ends.value(i) - expected_end).abs() < 1e-9);
        }
    }

    #[test]
    fn test_bin_density_normalizes_to_one() {
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(5),
            bin_width: None,
            extent: Some((1.0, 10.0)),
            nice: false,
        };
        let out = apply(&spec, &batch).unwrap();
        let densities = col_f64(&out, "density");
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        let mut total: f64 = 0.0;
        for i in 0..5 {
            total += densities.value(i) * (ends.value(i) - starts.value(i));
        }
        assert!((total - 1.0).abs() < 1e-12, "density integrates to {total}");
    }

    #[test]
    fn test_bin_default_count_uses_sturges_floor() {
        // sturges_floor(8) = 4 per scale::ticks::sturges_floor
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: None,
            bin_width: None,
            extent: None,
            nice: false,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 4);
    }

    #[test]
    fn test_bin_all_equal_data_emits_single_unit_bin() {
        let batch = batch_with(vec![3.0, 3.0, 3.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: None,
            bin_width: None,
            extent: None,
            nice: false,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        let counts = col_u64(&out, "count");
        assert!((starts.value(0) - 2.5).abs() < 1e-12);
        assert!((ends.value(0)   - 3.5).abs() < 1e-12);
        assert_eq!(counts.value(0), 3);
    }

    #[test]
    fn test_bin_drops_nulls_and_nans() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        let arr = Float64Array::from(vec![Some(1.0), None, Some(2.0), Some(f64::NAN), Some(3.0)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(2),
            bin_width: None,
            extent: Some((1.0, 3.0)),
            nice: false,
        };
        let out = apply(&spec, &batch).unwrap();
        let counts = col_u64(&out, "count");
        let total: u64 = (0..out.num_rows()).map(|i| counts.value(i)).sum();
        assert_eq!(total, 3, "expected 3 non-null/non-nan values");
    }

    #[test]
    fn test_bin_missing_field_errors() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let spec = BinSpec {
            field: "ghost".into(),
            bin_count: Some(5),
            bin_width: None,
            extent: None,
            nice: false,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("ghost"), "err: {err}");
    }

    #[test]
    fn test_bin_wrong_dtype_errors() {
        pyo3::Python::initialize();
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
        ).unwrap();
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(2),
            bin_width: None,
            extent: Some((1.0, 3.0)),
            nice: false,
        };
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Float64") || msg.contains("dtype"), "err: {msg}");
    }
}
