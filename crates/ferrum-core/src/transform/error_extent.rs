use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::context::TransformContext;
use crate::transform::group_key::{
    group_partition_multi, groupby_field_nullable, materialize_groupby_col,
    refuse_groupby_output_collision, KeyValue, NullKeyPolicy,
};
use crate::transform::numeric_util::coerce_to_float64;

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
#[serde(deny_unknown_fields)]
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

pub(crate) fn apply(spec: &ErrorExtentSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// This transform's own output column names, in schema order — the list a
/// `groupby` request is refused against (GH #179). Used both for the guard and
/// for the emitted schema, so the two cannot drift.
fn output_field_names() -> [String; 3] {
    ["mean".to_string(), "lower".to_string(), "upper".to_string()]
}

/// Context-aware entry point. `ctx` is used only as the GH #162
/// degrade-warning sink, for the one degrade this transform can report: an
/// all-null `groupby` key column (GH #197), raised inside the shared
/// partitioner.
pub(crate) fn apply_with_context(
    spec: &ErrorExtentSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let v_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("stat_error_extent: column '{}' not found", spec.field))
    })?;

    // One partitioning idiom for every grouped transform (GH #179): the shared
    // partitioner owns column validation, the repeated-name refusal,
    // first-appearance group order, and the null-key policy. Run BEFORE the
    // zero-row refusal below so a malformed `groupby` is still reported as
    // such on an empty batch, exactly as it was when the validation loop stood
    // here — the same ordering `violin` keeps.
    refuse_groupby_output_collision("stat_error_extent", &spec.groupby, &output_field_names())?;
    let partition = group_partition_multi(
        batch,
        &spec.groupby,
        "stat_error_extent",
        NullKeyPolicy::Skip,
        ctx,
    )?;
    let group_dtypes = &partition.dtypes;

    let n_rows = batch.num_rows();
    if n_rows == 0 {
        return Err(PyValueError::new_err("stat_error_extent: empty input batch"));
    }

    let v_arr = coerce_to_float64(batch.column(v_idx), "stat_error_extent", &spec.field)?;

    // Compute extents per group.
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::with_capacity(partition.groups.len());
    let mut means: Vec<f64> = Vec::with_capacity(partition.groups.len());
    let mut lowers: Vec<f64> = Vec::with_capacity(partition.groups.len());
    let mut uppers: Vec<f64> = Vec::with_capacity(partition.groups.len());

    for (key, rows) in &partition.groups {
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
        // FA-9: nullable so a null-keyed group materializes as an Arrow null.
        fields.push(Field::new(g, group_dtypes[gi].clone(), groupby_field_nullable()));
    }
    for name in output_field_names() {
        fields.push(Field::new(name, DataType::Float64, true));
    }
    let out_schema = Arc::new(Schema::new(fields));

    let mut cols: Vec<ArrayRef> = Vec::with_capacity(spec.groupby.len() + 3);
    for (gi, dt) in group_dtypes.iter().enumerate() {
        cols.push(materialize_groupby_col(&group_keys_out, gi, dt).map_err(PyValueError::new_err)?);
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
    boot_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

/// Per-group error-bar extent transform.
///
/// Computes the center (mean) and a symmetric error half-width for each
/// group using one of four methods. The result is used to draw error bars
/// in ``mark_errorbar`` and ``mark_errorband`` composite marks.
///
/// Output columns: groupby columns (if any), then ``mean`` (Float64),
/// ``lower`` (Float64), ``upper`` (Float64).
///
/// Parameters
/// ----------
/// field : str
///     Numeric column to summarize (any integer, unsigned, or float dtype;
///     widened to Float64). A non-numeric column raises, naming the
///     transform, column, and dtype.
/// method : {"ci", "stderr", "stdev", "iqr"}, default "ci"
///     Error-bar extent method:
///     ``"ci"`` — bootstrap confidence interval (uses ``seed`` and
///     ``n_boot``); ``"stderr"`` — standard error of the mean;
///     ``"stdev"`` — one standard deviation; ``"iqr"`` — inter-quartile
///     range.
/// groupby : list of str, default []
///     Columns to group by before computing statistics.
/// seed : int, default 0
///     RNG seed for the bootstrap (``method="ci"``). Seeds ``ChaCha8Rng``
///     for byte-deterministic output across platforms.
/// n_boot : int, default 1000
///     Bootstrap resamples when ``method="ci"``. Ignored for other methods.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_errorbar().encode(x="cut", y="price")
#[pyclass(eq, module = "ferrum._core", name = "ErrorExtent")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyErrorExtent(pub(crate) crate::transform::core::TransformSpec);

/// The single source of truth for `ErrorExtent`'s free-string `method`
/// vocabulary — read by [`PyErrorExtent::new`]'s error message and by
/// `registry::entry_for_kind("error_extent")`'s `methods` list, mirroring the
/// `smooth::METHODS` / `kde::KERNELS` / `qq::DISTRIBUTIONS` precedent (R1
/// design finding 1: this entry was a hand-copied literal in the registry
/// with no guard). The match arms below are the enforcement site; a value
/// here without an arm is caught by
/// `error_extent_methods_accepted_and_bogus_method_rejected`
/// (`registry/tests.rs`), which calls through the real PyO3 constructor.
pub(crate) const METHODS: &[&str] = &["ci", "stderr", "stdev", "iqr"];

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
                    "ErrorExtent: unknown method '{other}'; expected {}",
                    METHODS.join("|"),
                )))
            }
        };
        if matches!(parsed, ErrorMethod::Ci) && n_boot == 0 {
            return Err(PyValueError::new_err(
                "ErrorExtent: n_boot must be > 0 when method='ci'",
            ));
        }
        // The duplicate-groupby refusal is the shared rule every grouped
        // transform enforces, in the shared words — still refused here, at the
        // earliest boundary that has the facts, and again inside the
        // partitioner for the wire path that has no constructor (GH #179).
        crate::transform::group_key::refuse_repeated_groupby_names("ErrorExtent", &groupby)?;
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
        let se = (var / n).sqrt();
        let m = col_f64(&out, "mean");
        let lo = col_f64(&out, "lower");
        let hi = col_f64(&out, "upper");
        assert!((m[0] - mean).abs() < 1e-12);
        assert!((lo[0] - (mean - se)).abs() < 1e-12, "lower should be mean - se; got {}", lo[0]);
        assert!((hi[0] - (mean + se)).abs() < 1e-12, "upper should be mean + se; got {}", hi[0]);
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

    // ── FA-7: shared group-keying — integer groupby support ──────────────────

    fn batch_value_int64_group(values: Vec<f64>, groups: Vec<i64>) -> RecordBatch {
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, false),
            Field::new("grp", DataType::Int64, false),
        ]));
        let v = Float64Array::from(values);
        let g = Int64Array::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v), Arc::new(g)]).unwrap()
    }

    fn col_i64(b: &RecordBatch, name: &str) -> Vec<i64> {
        use arrow::array::Int64Array;
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    /// FA-7: error_extent grouped on an Int64 column groups correctly and
    /// preserves the Int64 output dtype. Before FA-7 the private `KeyValue`
    /// enum rejected non-Float64/Utf8 groupby columns with an error.
    #[test]
    fn error_extent_int64_groupby_succeeds_and_preserves_dtype() {
        pyo3::Python::initialize();
        // Group 1: [1,2,3] mean=2, sd=1; Group 2: [10,20,30] mean=20, sd=10.
        let b = batch_value_int64_group(
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            vec![1, 1, 1, 2, 2, 2],
        );
        let spec = ErrorExtentSpec {
            field: "v".into(),
            groupby: vec!["grp".into()],
            method: ErrorMethod::Stdev,
            seed: 0,
            n_boot: 1000,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 2);
        assert_eq!(
            out.schema().field(out.schema().index_of("grp").unwrap()).data_type(),
            &DataType::Int64
        );
        let grps = col_i64(&out, "grp");
        let m = col_f64(&out, "mean");
        let g1 = grps.iter().position(|&g| g == 1).unwrap();
        let g2 = grps.iter().position(|&g| g == 2).unwrap();
        assert!((m[g1] - 2.0).abs() < 1e-12, "group 1 mean: {}", m[g1]);
        assert!((m[g2] - 20.0).abs() < 1e-12, "group 2 mean: {}", m[g2]);
    }

    // ── FA-9: null groupby key regression ─────────────────────────────────────

    fn batch_value_int64_group_nullable(
        values: Vec<Option<f64>>,
        groups: Vec<Option<i64>>,
    ) -> RecordBatch {
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::Int64,   true),
        ]));
        let v_arr = Float64Array::from(values);
        let g_arr = Int64Array::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap()
    }

    /// FA-9 regression: a null value in an Int64 groupby column must not cause
    /// RecordBatch::try_new to panic or error. The null-keyed group must
    /// materialize as an Arrow null in the output (distinct from a real 0 key).
    /// GH #197: a null groupby key is the ABSENCE of a key, so the rows
    /// carrying it belong to no group and contribute to no output row. This
    /// test was written for the pre-#197 hand-rolled partitioner, where those
    /// rows pooled into a `KeyValue::Null` group emitted as an Arrow null; the
    /// adjudicated family-wide policy replaced that with the skip every
    /// partitioner-routed transform already used.
    #[test]
    fn error_extent_null_int64_groupby_rows_drop_out_of_every_group() {
        pyo3::Python::initialize();
        let batch = batch_value_int64_group_nullable(
            vec![Some(1.0), Some(2.0), Some(3.0), Some(10.0), Some(20.0), Some(30.0)],
            vec![None,      None,      None,      Some(7),    Some(7),    Some(7)],
        );
        let spec = ErrorExtentSpec {
            field: "v".into(),
            groupby: vec!["grp".into()],
            method: ErrorMethod::Stdev,
            seed: 0,
            n_boot: 0,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1, "only the key=7 group survives");

        use arrow::array::Int64Array;
        let grp_arr = out.column(out.schema().index_of("grp").unwrap())
            .as_any().downcast_ref::<Int64Array>().unwrap();
        assert!(!grp_arr.is_null(0), "no null-keyed group is emitted at all");
        assert_eq!(grp_arr.value(0), 7);
        let m = col_f64(&out, "mean");
        assert!(
            (m[0] - 20.0).abs() < 1e-12,
            "the surviving group's mean comes from ITS OWN rows; got {}",
            m[0]
        );
    }

    /// FA-7 byte-stability guard: a Utf8 groupby still materializes a StringArray
    /// with identical per-group extents after migrating to the shared module.
    #[test]
    fn error_extent_utf8_groupby_byte_stable() {
        pyo3::Python::initialize();
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
        assert_eq!(
            out.schema().field(out.schema().index_of("group").unwrap()).data_type(),
            &DataType::Utf8
        );
        let groups = col_str(&out, "group");
        let m = col_f64(&out, "mean");
        let a = groups.iter().position(|g| g == "a").unwrap();
        assert!((m[a] - 2.0).abs() < 1e-12);
    }

    // ── GH #179: shared partitioner rules ───────────────────────────────────

    /// DISCLOSED BEHAVIOR CHANGE (#179): first-appearance group order.
    #[test]
    fn error_extent_group_rows_are_in_first_appearance_order() {
        pyo3::Python::initialize();
        let batch = batch_value_group(
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            vec!["b", "b", "b", "a", "a", "a"],
        );
        let spec = ErrorExtentSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            method: ErrorMethod::Stdev,
            seed: 0,
            n_boot: 0,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        use arrow::array::StringArray;
        let g = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!((g.value(0), g.value(1)), ("b", "a"), "first-appearance, not sorted");

        // The flip is a PERMUTATION: b's values are [1, 2, 3] (mean 2) and a's
        // are [10, 20, 30] (mean 20), each travelling with its own key.
        let m = col_f64(&out, "mean");
        assert_eq!(
            (m[0], m[1]),
            (2.0, 20.0),
            "each group's mean stays with its own key after the reorder"
        );
    }

    /// Grouping by a column named like one of this transform's fixed outputs
    /// refuses (GH #179 residual 1: this append site had no guard).
    #[test]
    fn error_extent_groupby_colliding_with_an_output_column_refuses() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("upper", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(arrow::array::StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let spec = ErrorExtentSpec {
            field: "v".into(),
            groupby: vec!["upper".into()],
            method: ErrorMethod::Stdev,
            seed: 0,
            n_boot: 0,
            name: None,
        };
        let err = apply(&spec, &batch)
            .expect_err("grouping by a column named 'upper' must refuse");
        assert!(err.to_string().contains("collides"), "{err}");
    }

    #[test]
    fn error_extent_repeated_groupby_name_refuses_from_the_wire_path() {
        pyo3::Python::initialize();
        let batch = batch_value_group(vec![1.0], vec!["a"]);
        let spec = ErrorExtentSpec {
            field: "v".into(),
            groupby: vec!["group".into(), "group".into()],
            method: ErrorMethod::Stdev,
            seed: 0,
            n_boot: 0,
            name: None,
        };
        let err = apply(&spec, &batch).expect_err("a repeated groupby name must refuse");
        assert!(err.to_string().contains("repeated"), "{err}");
    }

}
