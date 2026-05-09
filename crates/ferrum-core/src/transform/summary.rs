use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ErrorFn {
    Ci,
    Stderr,
    Stdev,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct SummarySpec {
    pub field: String,
    #[serde(default)]
    pub groupby: Vec<String>,
    pub error_fn: ErrorFn,
    pub ci: f64,
    pub n_boot: usize,
    #[serde(default)]
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum KeyValue {
    Str(String),
    Float(u64),
}

pub(crate) fn apply(spec: &SummarySpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let v_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("stat_summary: column '{}' not found", spec.field))
    })?;
    if schema.field(v_idx).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_summary: column '{}' must be Float64",
            spec.field
        )));
    }

    let mut group_dtypes: Vec<DataType> = Vec::with_capacity(spec.groupby.len());
    for g in &spec.groupby {
        let i = schema.index_of(g).map_err(|_| {
            PyValueError::new_err(format!(
                "stat_summary: groupby column '{}' not found",
                g
            ))
        })?;
        let dt = schema.field(i).data_type().clone();
        if dt != DataType::Float64 && !matches!(dt, DataType::Utf8) {
            return Err(PyValueError::new_err(format!(
                "stat_summary: groupby column '{}' must be Float64 or Utf8",
                g
            )));
        }
        group_dtypes.push(dt);
    }

    let n_rows = batch.num_rows();
    if n_rows == 0 {
        return Err(PyValueError::new_err("stat_summary: empty input batch"));
    }

    let v_arr = batch
        .column(v_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    // Collect rows per group key.
    let mut groups: BTreeMap<Vec<KeyValue>, Vec<usize>> = BTreeMap::new();
    let group_arrays: Vec<&dyn arrow::array::Array> = spec
        .groupby
        .iter()
        .map(|g| batch.column(schema.index_of(g).unwrap()).as_ref())
        .collect();

    for row in 0..n_rows {
        let mut key = Vec::with_capacity(spec.groupby.len());
        for (gi, arr) in group_arrays.iter().enumerate() {
            match group_dtypes[gi] {
                DataType::Float64 => {
                    let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
                    if a.is_null(row) {
                        key.push(KeyValue::Float(f64::NAN.to_bits()));
                    } else {
                        key.push(KeyValue::Float(a.value(row).to_bits()));
                    }
                }
                DataType::Utf8 => {
                    let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
                    if a.is_null(row) {
                        key.push(KeyValue::Str(String::new()));
                    } else {
                        key.push(KeyValue::Str(a.value(row).to_string()));
                    }
                }
                _ => unreachable!(),
            }
        }
        groups.entry(key).or_default().push(row);
    }
    if spec.groupby.is_empty() {
        let all: Vec<usize> = (0..n_rows).collect();
        groups.clear();
        groups.insert(Vec::new(), all);
    }

    // Compute mean + (lower, upper) per group.
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::with_capacity(groups.len());
    let mut means: Vec<f64> = Vec::with_capacity(groups.len());
    let mut lowers: Vec<f64> = Vec::with_capacity(groups.len());
    let mut uppers: Vec<f64> = Vec::with_capacity(groups.len());

    for (key, rows) in &groups {
        group_keys_out.push(key.clone());
        let vals: Vec<f64> = rows
            .iter()
            .filter_map(|&r| {
                if v_arr.is_null(r) {
                    return None;
                }
                let v = v_arr.value(r);
                if v.is_nan() {
                    return None;
                }
                Some(v)
            })
            .collect();
        let (m, lo, hi) = summarize(&vals, spec);
        means.push(m);
        lowers.push(lo);
        uppers.push(hi);
    }

    // Build output.
    let mut fields: Vec<Field> = Vec::with_capacity(spec.groupby.len() + 3);
    for (gi, g) in spec.groupby.iter().enumerate() {
        fields.push(Field::new(g, group_dtypes[gi].clone(), false));
    }
    fields.push(Field::new("mean", DataType::Float64, true));
    fields.push(Field::new("lower", DataType::Float64, true));
    fields.push(Field::new("upper", DataType::Float64, true));
    let out_schema = Arc::new(Schema::new(fields));

    let mut cols: Vec<ArrayRef> = Vec::with_capacity(spec.groupby.len() + 3);
    for gi in 0..spec.groupby.len() {
        match group_dtypes[gi] {
            DataType::Float64 => {
                let v: Vec<f64> = group_keys_out
                    .iter()
                    .map(|k| match &k[gi] {
                        KeyValue::Float(bits) => f64::from_bits(*bits),
                        KeyValue::Str(_) => unreachable!(),
                    })
                    .collect();
                cols.push(Arc::new(Float64Array::from(v)));
            }
            DataType::Utf8 => {
                let v: Vec<String> = group_keys_out
                    .iter()
                    .map(|k| match &k[gi] {
                        KeyValue::Str(s) => s.clone(),
                        KeyValue::Float(_) => unreachable!(),
                    })
                    .collect();
                cols.push(Arc::new(StringArray::from(v)));
            }
            _ => unreachable!(),
        }
    }
    cols.push(Arc::new(Float64Array::from(means)));
    cols.push(Arc::new(Float64Array::from(lowers)));
    cols.push(Arc::new(Float64Array::from(uppers)));

    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_summary: {e}")))
}

fn summarize(vals: &[f64], spec: &SummarySpec) -> (f64, f64, f64) {
    if vals.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let n = vals.len();
    let mean = vals.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, f64::NAN, f64::NAN);
    }
    match spec.error_fn {
        ErrorFn::Stdev => {
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
            let sd = var.sqrt();
            (mean, mean - sd, mean + sd)
        }
        ErrorFn::Stderr => {
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
            let se = (var / n as f64).sqrt();
            (mean, mean - se, mean + se)
        }
        ErrorFn::Ci => {
            // Bootstrap CI lands in Task 20.
            (mean, f64::NAN, f64::NAN)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch_value_group(values: Vec<Option<f64>>, groups: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("group", DataType::Utf8, true),
        ]));
        let v = Float64Array::from(values);
        let g = StringArray::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v), Arc::new(g)]).unwrap()
    }

    fn col_f64(b: &RecordBatch, name: &str) -> Vec<f64> {
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        (0..arr.len())
            .map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) })
            .collect()
    }

    fn col_str(b: &RecordBatch, name: &str) -> Vec<String> {
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    }

    #[test]
    fn test_summary_stdev_per_group() {
        // Group a: [1, 2, 3] → mean=2, var=1.0, sd=1.0
        // Group b: [10, 20] → mean=15, var=50, sd~7.07
        let batch = batch_value_group(
            vec![Some(1.0), Some(2.0), Some(3.0), Some(10.0), Some(20.0)],
            vec!["a", "a", "a", "b", "b"],
        );
        let spec = SummarySpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            error_fn: ErrorFn::Stdev,
            ci: 0.95,
            n_boot: 0,
            seed: 0,
        };
        let out = apply(&spec, &batch).unwrap();
        let groups = col_str(&out, "group");
        let mean = col_f64(&out, "mean");
        let lower = col_f64(&out, "lower");
        let upper = col_f64(&out, "upper");
        let a = groups.iter().position(|g| g == "a").unwrap();
        let b = groups.iter().position(|g| g == "b").unwrap();
        assert!((mean[a] - 2.0).abs() < 1e-12);
        assert!((upper[a] - 3.0).abs() < 1e-12, "mean+sd should be 3.0");
        assert!((lower[a] - 1.0).abs() < 1e-12);
        assert!((mean[b] - 15.0).abs() < 1e-12);
        assert!((upper[b] - lower[b] - 2.0 * 50.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_summary_stderr_uses_var_div_n() {
        let batch = batch_value_group(
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
            vec!["a", "a", "a", "a"],
        );
        let spec = SummarySpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            error_fn: ErrorFn::Stderr,
            ci: 0.95,
            n_boot: 0,
            seed: 0,
        };
        let out = apply(&spec, &batch).unwrap();
        let mean = col_f64(&out, "mean");
        let lower = col_f64(&out, "lower");
        let upper = col_f64(&out, "upper");
        let var = ((1.0_f64 - 2.5_f64).powi(2)
            + (2.0_f64 - 2.5_f64).powi(2)
            + (3.0_f64 - 2.5_f64).powi(2)
            + (4.0_f64 - 2.5_f64).powi(2))
            / 3.0_f64;
        let se = (var / 4.0).sqrt();
        assert!((mean[0] - 2.5).abs() < 1e-12);
        assert!((upper[0] - (2.5 + se)).abs() < 1e-12);
        assert!((lower[0] - (2.5 - se)).abs() < 1e-12);
    }

    #[test]
    fn test_summary_n_lt_2_emits_nan_bounds() {
        let batch = batch_value_group(
            vec![Some(7.0), Some(1.0), Some(2.0)],
            vec!["a", "b", "b"],
        );
        let spec = SummarySpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            error_fn: ErrorFn::Stdev,
            ci: 0.95,
            n_boot: 0,
            seed: 0,
        };
        let out = apply(&spec, &batch).unwrap();
        let groups = col_str(&out, "group");
        let mean = col_f64(&out, "mean");
        let lower = col_f64(&out, "lower");
        let upper = col_f64(&out, "upper");
        let a = groups.iter().position(|g| g == "a").unwrap();
        assert!((mean[a] - 7.0).abs() < 1e-12);
        assert!(lower[a].is_nan());
        assert!(upper[a].is_nan());
    }

    #[test]
    fn test_summary_no_groupby_global() {
        let batch = batch_value_group(
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec!["a", "b", "c"],
        );
        let spec = SummarySpec {
            field: "v".into(),
            groupby: vec![],
            error_fn: ErrorFn::Stderr,
            ci: 0.95,
            n_boot: 0,
            seed: 0,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
    }

    #[test]
    fn test_summary_round_trip_json() {
        let original = SummarySpec {
            field: "v".into(),
            groupby: vec!["g".into()],
            error_fn: ErrorFn::Ci,
            ci: 0.95,
            n_boot: 1000,
            seed: 42,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: SummarySpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }
}
