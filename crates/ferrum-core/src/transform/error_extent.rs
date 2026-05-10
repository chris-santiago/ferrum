use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ErrorMethod {
    Ci,
    Stderr,
    Stdev,
    Iqr,
}

fn default_seed() -> u64 { 0 }
fn default_n_boot() -> usize { 1000 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ErrorExtentSpec {
    pub field: String,
    #[serde(default)]
    pub groupby: Vec<String>,
    pub method: ErrorMethod,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_n_boot")]
    pub n_boot: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum KeyValue {
    Str(String),
    Float(u64),
}

pub(crate) fn apply(spec: &ErrorExtentSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let v_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("stat_error_extent: column '{}' not found", spec.field))
    })?;
    if schema.field(v_idx).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_error_extent: column '{}' must be Float64",
            spec.field
        )));
    }

    let mut group_dtypes: Vec<DataType> = Vec::with_capacity(spec.groupby.len());
    for g in &spec.groupby {
        let i = schema.index_of(g).map_err(|_| {
            PyValueError::new_err(format!(
                "stat_error_extent: groupby column '{}' not found",
                g
            ))
        })?;
        let dt = schema.field(i).data_type().clone();
        if dt != DataType::Float64 && !matches!(dt, DataType::Utf8) {
            return Err(PyValueError::new_err(format!(
                "stat_error_extent: groupby column '{}' must be Float64 or Utf8",
                g
            )));
        }
        group_dtypes.push(dt);
    }

    let n_rows = batch.num_rows();
    if n_rows == 0 {
        return Err(PyValueError::new_err("stat_error_extent: empty input batch"));
    }

    let v_arr = batch
        .column(v_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    // Bucket rows by group key.
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

    // Compute extents per group.
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
        let (m, lo, hi) = compute_extent(&vals, spec);
        means.push(m);
        lowers.push(lo);
        uppers.push(hi);
    }

    // Build output schema: groupby cols + mean + lower + upper.
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
        .map_err(|e| PyValueError::new_err(format!("stat_error_extent: {e}")))
}

fn compute_extent(vals: &[f64], spec: &ErrorExtentSpec) -> (f64, f64, f64) {
    if vals.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let n = vals.len();
    let mean = vals.iter().sum::<f64>() / n as f64;

    match spec.method {
        ErrorMethod::Iqr => {
            // IQR mode: mean column = median, lower = Q1, upper = Q3 (Type-7).
            let median = quartile(vals, 0.5);
            if n < 2 {
                return (median, f64::NAN, f64::NAN);
            }
            let q1 = quartile(vals, 0.25);
            let q3 = quartile(vals, 0.75);
            (median, q1, q3)
        }
        _ => {
            if n < 2 {
                return (mean, f64::NAN, f64::NAN);
            }
            match spec.method {
                ErrorMethod::Stdev => {
                    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / (n as f64 - 1.0);
                    let sd = var.sqrt();
                    (mean, mean - sd, mean + sd)
                }
                ErrorMethod::Stderr => {
                    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / (n as f64 - 1.0);
                    let se = (var / n as f64).sqrt();
                    (mean, mean - se, mean + se)
                }
                ErrorMethod::Ci => bootstrap_ci(vals, 0.95, spec.n_boot, spec.seed),
                ErrorMethod::Iqr => unreachable!(),
            }
        }
    }
}

fn bootstrap_ci(vals: &[f64], level: f64, n_boot: usize, seed: u64) -> (f64, f64, f64) {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let n = vals.len();
    let mean = vals.iter().sum::<f64>() / n as f64;
    if n < 2 || n_boot == 0 || level <= 0.0 || level >= 1.0 {
        return (mean, f64::NAN, f64::NAN);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut boot_means: Vec<f64> = Vec::with_capacity(n_boot);
    let mut sample = vec![0.0; n];
    for _ in 0..n_boot {
        for i in 0..n {
            let j = rng.gen_range(0..n);
            sample[i] = vals[j];
        }
        let m = sample.iter().sum::<f64>() / n as f64;
        boot_means.push(m);
    }
    boot_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let alpha = 1.0 - level;
    let lo_q = alpha / 2.0;
    let hi_q = 1.0 - alpha / 2.0;
    let lo = percentile_sorted(&boot_means, lo_q);
    let hi = percentile_sorted(&boot_means, hi_q);
    (mean, lo, hi)
}

fn percentile_sorted(s: &[f64], p: f64) -> f64 {
    let n = s.len();
    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = (h.ceil() as usize).min(n - 1);
    let frac = h - h.floor();
    s[lo] * (1.0 - frac) + s[hi] * frac
}

/// Type-7 quantile (linear interpolation) — matches numpy/scipy default.
fn quartile(values: &[f64], p: f64) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = (n - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

use pyo3::prelude::*;

#[pyclass(eq, module = "ferrum._core", name = "ErrorExtent")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyErrorExtent(pub(crate) crate::transform::core::TransformSpec);

#[pymethods]
impl PyErrorExtent {
    #[new]
    #[pyo3(signature = (field, *, method = "ci", groupby = vec![], seed = 0, n_boot = 1000, name = None))]
    fn new(
        field: &str,
        method: &str,
        groupby: Vec<String>,
        seed: u64,
        n_boot: usize,
        name: Option<String>,
    ) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("ErrorExtent: field must be non-empty"));
        }
        let parsed = match method {
            "ci" => ErrorMethod::Ci,
            "stderr" => ErrorMethod::Stderr,
            "stdev" => ErrorMethod::Stdev,
            "iqr" => ErrorMethod::Iqr,
            other => {
                return Err(PyValueError::new_err(format!(
                    "ErrorExtent: unknown method '{other}'; expected ci|stderr|stdev|iqr"
                )))
            }
        };
        if matches!(parsed, ErrorMethod::Ci) && n_boot == 0 {
            return Err(PyValueError::new_err(
                "ErrorExtent: n_boot must be > 0 when method='ci'",
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for g in &groupby {
            if !seen.insert(g.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "ErrorExtent: duplicate groupby field '{g}'"
                )));
            }
        }
        Ok(Self(crate::transform::core::TransformSpec::ErrorExtent(
            ErrorExtentSpec {
                field: field.to_string(),
                groupby,
                method: parsed,
                seed,
                n_boot,
                name,
            },
        )))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            crate::transform::core::TransformSpec::ErrorExtent(s) => format!(
                "ErrorExtent(field='{}', groupby={:?}, method={:?}, seed={}, n_boot={})",
                s.field, s.groupby, s.method, s.seed, s.n_boot
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch(field: &str, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(field, DataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    fn batch_value_group(values: Vec<f64>, groups: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, false),
            Field::new("group", DataType::Utf8, false),
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
    fn error_extent_stdev_method() {
        pyo3::Python::initialize();
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = batch("v", vals.clone());
        let spec = ErrorExtentSpec {
            field: "v".into(),
            groupby: vec![],
            method: ErrorMethod::Stdev,
            seed: 0,
            n_boot: 1000,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let sd = var.sqrt();
        let m = col_f64(&out, "mean");
        let lo = col_f64(&out, "lower");
        let hi = col_f64(&out, "upper");
        assert!((m[0] - mean).abs() < 1e-12);
        assert!((lo[0] - (mean - sd)).abs() < 1e-12);
        assert!((hi[0] - (mean + sd)).abs() < 1e-12);
    }

    #[test]
    fn error_extent_stderr_method() {
        pyo3::Python::initialize();
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = batch("v", vals.clone());
        let spec = ErrorExtentSpec {
            field: "v".into(),
            groupby: vec![],
            method: ErrorMethod::Stderr,
            seed: 0,
            n_boot: 1000,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let sd = var.sqrt();
        let lo = col_f64(&out, "lower");
        let hi = col_f64(&out, "upper");
        // stderr CI should be narrower than stdev CI
        assert!(hi[0] - lo[0] < 2.0 * sd, "stderr extent {} >= 2*stdev {}", hi[0] - lo[0], 2.0 * sd);
    }

    #[test]
    fn error_extent_iqr_method() {
        pyo3::Python::initialize();
        // 1..=10 → median=5.5; Q1 (Type-7 at 0.25, h=2.25) = 3.25; Q3 (h=6.75) = 7.75.
        let vals: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let b = batch("v", vals.clone());
        let spec = ErrorExtentSpec {
            field: "v".into(),
            groupby: vec![],
            method: ErrorMethod::Iqr,
            seed: 0,
            n_boot: 1000,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let m = col_f64(&out, "mean");
        let lo = col_f64(&out, "lower");
        let hi = col_f64(&out, "upper");
        assert!((m[0] - 5.5).abs() < 1e-12, "median should be 5.5; got {}", m[0]);
        assert!((lo[0] - 3.25).abs() < 1e-12, "Q1 should be 3.25; got {}", lo[0]);
        assert!((hi[0] - 7.75).abs() < 1e-12, "Q3 should be 7.75; got {}", hi[0]);
    }

    #[test]
    fn error_extent_per_group_distinct_extents() {
        pyo3::Python::initialize();
        // Group a: [1,2,3] mean=2, sd=1; Group b: [10,20,30] mean=20, sd=10.
        let b = batch_value_group(
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            vec!["a", "a", "a", "b", "b", "b"],
        );
        let spec = ErrorExtentSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            method: ErrorMethod::Stdev,
            seed: 0,
            n_boot: 1000,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 2);
        let groups = col_str(&out, "group");
        let m = col_f64(&out, "mean");
        let lo = col_f64(&out, "lower");
        let hi = col_f64(&out, "upper");
        let ai = groups.iter().position(|g| g == "a").unwrap();
        let bi = groups.iter().position(|g| g == "b").unwrap();
        assert!((m[ai] - 2.0).abs() < 1e-12);
        assert!((m[bi] - 20.0).abs() < 1e-12);
        assert!((m[ai] - m[bi]).abs() > 1e-6);
        assert!((lo[ai] - lo[bi]).abs() > 1e-6);
        assert!((hi[ai] - hi[bi]).abs() > 1e-6);
    }

    #[test]
    fn bootstrap_ci_byte_deterministic_with_fixed_seed() {
        pyo3::Python::initialize();
        let b = batch("v", (0..100).map(|i| i as f64).collect());
        let spec = ErrorExtentSpec {
            field: "v".into(),
            groupby: vec![],
            method: ErrorMethod::Ci,
            seed: 42,
            n_boot: 1000,
            name: None,
        };
        let out_a = apply(&spec, &b).unwrap();
        let out_b = apply(&spec, &b).unwrap();
        let lo_a = col_f64(&out_a, "lower")[0];
        let lo_b = col_f64(&out_b, "lower")[0];
        let hi_a = col_f64(&out_a, "upper")[0];
        let hi_b = col_f64(&out_b, "upper")[0];
        assert_eq!(lo_a.to_bits(), lo_b.to_bits(), "lower not deterministic");
        assert_eq!(hi_a.to_bits(), hi_b.to_bits(), "upper not deterministic");
    }
}
