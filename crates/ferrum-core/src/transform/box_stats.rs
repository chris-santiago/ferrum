//! BoxStats: per-group Type-7 quartiles + Tukey/min-max whiskers.
//! Output schema = groupby cols + q1, median, q3, lower_whisker, upper_whisker (all f64, nullable).
//! One row per group.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transform::group_key::{
    groupby_key_at, is_groupby_supported_dtype, materialize_groupby_col, KeyValue,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum WhiskerExtent {
    MinMax(String),     // "min-max"
    IqrMultiplier(f64), // 1.5 default
}

impl Default for WhiskerExtent {
    fn default() -> Self {
        Self::IqrMultiplier(1.5)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct BoxStatsSpec {
    pub field: String,
    #[serde(default)]
    pub groupby: Vec<String>,
    #[serde(default)]
    pub whisker_extent: WhiskerExtent,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &BoxStatsSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let v_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!(
            "stat_box_stats: column '{}' not found",
            spec.field
        ))
    })?;
    if schema.field(v_idx).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_box_stats: column '{}' must be Float64",
            spec.field
        )));
    }

    let mut group_dtypes: Vec<DataType> = Vec::with_capacity(spec.groupby.len());
    for g in &spec.groupby {
        let i = schema.index_of(g).map_err(|_| {
            PyValueError::new_err(format!(
                "stat_box_stats: groupby column '{}' not found",
                g
            ))
        })?;
        let dt = schema.field(i).data_type().clone();
        if !is_groupby_supported_dtype(&dt) {
            return Err(PyValueError::new_err(format!(
                "stat_box_stats: groupby column '{}' has unsupported dtype {:?}; \
                 supported: Utf8/LargeUtf8, Float64/Float32, \
                 Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean",
                g, dt
            )));
        }
        group_dtypes.push(dt);
    }

    let n_rows = batch.num_rows();
    if n_rows == 0 {
        return Err(PyValueError::new_err("stat_box_stats: empty input batch"));
    }

    let v_arr = batch
        .column(v_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("stat_box_stats: expected Float64Array for value column"))?;

    // Bucket rows by group key (BTreeMap → deterministic ordering).
    let mut groups: BTreeMap<Vec<KeyValue>, Vec<usize>> = BTreeMap::new();
    let group_arrays: Vec<&dyn arrow::array::Array> = spec
        .groupby
        .iter()
        .map(|g| batch.column(schema.index_of(g).expect("invariant: groupby columns validated above")).as_ref())
        .collect();

    for row in 0..n_rows {
        let mut key = Vec::with_capacity(spec.groupby.len());
        for (gi, arr) in group_arrays.iter().enumerate() {
            let kv = groupby_key_at(*arr, &group_dtypes[gi], row).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "stat_box_stats: internal error extracting groupby key at row {row}"
                ))
            })?;
            key.push(kv);
        }
        groups.entry(key).or_default().push(row);
    }
    if spec.groupby.is_empty() {
        let all: Vec<usize> = (0..n_rows).collect();
        groups.clear();
        groups.insert(Vec::new(), all);
    }

    // Compute stats per group.
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::with_capacity(groups.len());
    let mut q1s: Vec<f64> = Vec::with_capacity(groups.len());
    let mut medians: Vec<f64> = Vec::with_capacity(groups.len());
    let mut q3s: Vec<f64> = Vec::with_capacity(groups.len());
    let mut lows: Vec<f64> = Vec::with_capacity(groups.len());
    let mut highs: Vec<f64> = Vec::with_capacity(groups.len());

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
        let (q1, med, q3, lo, hi) = compute_stats(&vals, &spec.whisker_extent);
        q1s.push(q1);
        medians.push(med);
        q3s.push(q3);
        lows.push(lo);
        highs.push(hi);
    }

    // Build output schema: groupby cols + q1 + median + q3 + lower_whisker + upper_whisker.
    let mut fields: Vec<Field> = Vec::with_capacity(spec.groupby.len() + 5);
    for (gi, g) in spec.groupby.iter().enumerate() {
        fields.push(Field::new(g, group_dtypes[gi].clone(), false));
    }
    fields.push(Field::new("q1", DataType::Float64, true));
    fields.push(Field::new("median", DataType::Float64, true));
    fields.push(Field::new("q3", DataType::Float64, true));
    fields.push(Field::new("lower_whisker", DataType::Float64, true));
    fields.push(Field::new("upper_whisker", DataType::Float64, true));
    let out_schema = Arc::new(Schema::new(fields));

    let mut cols: Vec<ArrayRef> = Vec::with_capacity(spec.groupby.len() + 5);
    for (gi, dt) in group_dtypes.iter().enumerate() {
        cols.push(materialize_groupby_col(&group_keys_out, gi, dt).map_err(PyValueError::new_err)?);
    }
    cols.push(Arc::new(Float64Array::from(q1s)));
    cols.push(Arc::new(Float64Array::from(medians)));
    cols.push(Arc::new(Float64Array::from(q3s)));
    cols.push(Arc::new(Float64Array::from(lows)));
    cols.push(Arc::new(Float64Array::from(highs)));

    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_box_stats: {e}")))
}

/// Compute (q1, median, q3, lower_whisker, upper_whisker) for a group's values.
fn compute_stats(
    vals: &[f64],
    whisker_extent: &WhiskerExtent,
) -> (f64, f64, f64, f64, f64) {
    if vals.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let n = vals.len();
    if n == 1 {
        let v = vals[0];
        return (v, v, v, v, v);
    }
    let q1 = quartile(vals, 0.25);
    let median = quartile(vals, 0.5);
    let q3 = quartile(vals, 0.75);
    let mut sorted: Vec<f64> = vals.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let data_min = sorted[0];
    let data_max = sorted[sorted.len() - 1];

    let (lo, hi) = match whisker_extent {
        WhiskerExtent::MinMax(_) => (data_min, data_max),
        WhiskerExtent::IqrMultiplier(k) => {
            let iqr = q3 - q1;
            let fence_lo = q1 - k * iqr;
            let fence_hi = q3 + k * iqr;
            // Whiskers extend to the most extreme datum within the fence —
            // in other words, clamped to actual data range, never past it.
            let lo = sorted
                .iter()
                .copied()
                .find(|&v| v >= fence_lo)
                .unwrap_or(data_min);
            let hi = sorted
                .iter()
                .rev()
                .copied()
                .find(|&v| v <= fence_hi)
                .unwrap_or(data_max);
            (lo, hi)
        }
    };
    (q1, median, q3, lo, hi)
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

#[derive(FromPyObject)]
enum WhiskerExtentArg {
    #[pyo3(transparent)]
    String(String),
    #[pyo3(transparent)]
    Float(f64),
}

impl Default for WhiskerExtentArg {
    fn default() -> Self {
        Self::Float(1.5)
    }
}

impl WhiskerExtentArg {
    fn into_spec(self) -> PyResult<WhiskerExtent> {
        match self {
            Self::String(s) if s == "min-max" => Ok(WhiskerExtent::MinMax("min-max".into())),
            Self::String(s) => Err(PyValueError::new_err(format!(
                "BoxStats: whisker_extent string must be 'min-max'; got '{s}'"
            ))),
            Self::Float(f) if f.is_finite() && f >= 0.0 => Ok(WhiskerExtent::IqrMultiplier(f)),
            Self::Float(f) => Err(PyValueError::new_err(format!(
                "BoxStats: whisker_extent multiplier must be a non-negative finite number; got {f}"
            ))),
        }
    }
}

/// Box-plot five-number summary with Tukey or min-max whiskers.
///
/// Computes Q1, median, Q3 and whisker endpoints per group using Type-7
/// quantile interpolation (same as NumPy/SciPy default). Output has one row
/// per unique group-key combination.
///
/// Output columns: groupby columns (if any), then ``q1``, ``median``,
/// ``q3``, ``lower_whisker``, ``upper_whisker`` (all Float64, nullable).
///
/// Parameters
/// ----------
/// field : str
///     Numeric column to summarize (must be Float64).
/// groupby : list of str, default []
///     Columns to group by before computing statistics.
/// whisker_extent : float or "min-max", default 1.5
///     When a float, whiskers extend to the most extreme data point within
///     ``whisker_extent * IQR`` of Q1/Q3. Use ``"min-max"`` to extend
///     whiskers to the data minimum and maximum.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_boxplot().encode(x="cut", y="price")
#[pyclass(eq, module = "ferrum._core", name = "BoxStats")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyBoxStats(pub(crate) crate::transform::core::TransformSpec);

#[pymethods]
impl PyBoxStats {
    #[new]
    #[pyo3(signature = (field, *, groupby = vec![], whisker_extent = WhiskerExtentArg::default(), name = None))]
    fn new(
        field: &str,
        groupby: Vec<String>,
        whisker_extent: WhiskerExtentArg,
        name: Option<String>,
    ) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("BoxStats: field must be non-empty"));
        }
        let mut seen = std::collections::HashSet::new();
        for g in &groupby {
            if !seen.insert(g.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "BoxStats: duplicate groupby field '{g}'"
                )));
            }
        }
        let we = whisker_extent.into_spec()?;
        Ok(Self(crate::transform::core::TransformSpec::BoxStats(
            BoxStatsSpec {
                field: field.to_string(),
                groupby,
                whisker_extent: we,
                name,
            },
        )))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            crate::transform::core::TransformSpec::BoxStats(s) => format!(
                "BoxStats(field='{}', groupby={:?}, whisker_extent={:?}, name={:?})",
                s.field, s.groupby, s.whisker_extent, s.name
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
    fn box_stats_quartiles_for_uniform_data() {
        pyo3::Python::initialize();
        let vals: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let b = batch("v", vals);
        let spec = BoxStatsSpec {
            field: "v".into(),
            groupby: vec![],
            whisker_extent: WhiskerExtent::IqrMultiplier(1.5),
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let q1 = col_f64(&out, "q1");
        let median = col_f64(&out, "median");
        let q3 = col_f64(&out, "q3");
        assert!((q1[0] - 3.25).abs() < 1e-12, "q1 should be 3.25; got {}", q1[0]);
        assert!((median[0] - 5.5).abs() < 1e-12, "median should be 5.5; got {}", median[0]);
        assert!((q3[0] - 7.75).abs() < 1e-12, "q3 should be 7.75; got {}", q3[0]);
    }

    #[test]
    fn box_stats_min_max_whiskers() {
        pyo3::Python::initialize();
        let vals: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let b = batch("v", vals);
        let spec = BoxStatsSpec {
            field: "v".into(),
            groupby: vec![],
            whisker_extent: WhiskerExtent::MinMax("min-max".into()),
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let lo = col_f64(&out, "lower_whisker");
        let hi = col_f64(&out, "upper_whisker");
        assert!((lo[0] - 1.0).abs() < 1e-12, "lower_whisker should be 1.0; got {}", lo[0]);
        assert!((hi[0] - 10.0).abs() < 1e-12, "upper_whisker should be 10.0; got {}", hi[0]);
    }

    #[test]
    fn box_stats_iqr_multiplier_clamps_to_data_range() {
        pyo3::Python::initialize();
        // 1..=10: Q1=3.25, Q3=7.75, IQR=4.5; theoretical fence = -3.5, 14.5; clamp to data [1, 10].
        let vals: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let b = batch("v", vals);
        let spec = BoxStatsSpec {
            field: "v".into(),
            groupby: vec![],
            whisker_extent: WhiskerExtent::IqrMultiplier(1.5),
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let lo = col_f64(&out, "lower_whisker");
        let hi = col_f64(&out, "upper_whisker");
        assert_eq!(lo[0], 1.0, "lower_whisker should clamp to data min 1.0; got {}", lo[0]);
        assert_eq!(hi[0], 10.0, "upper_whisker should clamp to data max 10.0; got {}", hi[0]);
    }

    #[test]
    fn box_stats_per_group_distinct_quartiles() {
        pyo3::Python::initialize();
        // Group a: [1,2,3,4,5] median=3, q1=2, q3=4
        // Group b: [10,20,30,40,50] median=30, q1=20, q3=40
        let b = batch_value_group(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            vec!["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        );
        let spec = BoxStatsSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            whisker_extent: WhiskerExtent::IqrMultiplier(1.5),
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 2);
        let groups = col_str(&out, "group");
        let q1 = col_f64(&out, "q1");
        let median = col_f64(&out, "median");
        let q3 = col_f64(&out, "q3");
        let ai = groups.iter().position(|g| g == "a").unwrap();
        let bi = groups.iter().position(|g| g == "b").unwrap();
        assert!((q1[ai] - 2.0).abs() < 1e-12, "group a q1 should be 2.0; got {}", q1[ai]);
        assert!((median[ai] - 3.0).abs() < 1e-12, "group a median should be 3.0; got {}", median[ai]);
        assert!((q3[ai] - 4.0).abs() < 1e-12, "group a q3 should be 4.0; got {}", q3[ai]);
        assert!((q1[bi] - 20.0).abs() < 1e-12, "group b q1 should be 20.0; got {}", q1[bi]);
        assert!((median[bi] - 30.0).abs() < 1e-12, "group b median should be 30.0; got {}", median[bi]);
        assert!((q3[bi] - 40.0).abs() < 1e-12, "group b q3 should be 40.0; got {}", q3[bi]);
        assert!((q1[ai] - q1[bi]).abs() > 1e-6);
        assert!((median[ai] - median[bi]).abs() > 1e-6);
        assert!((q3[ai] - q3[bi]).abs() > 1e-6);
    }

    #[test]
    fn box_stats_round_trip() {
        // MinMax variant: serializes as JSON string "min-max".
        let s_minmax = BoxStatsSpec {
            field: "v".into(),
            groupby: vec!["g".into()],
            whisker_extent: WhiskerExtent::MinMax("min-max".into()),
            name: Some("bs".into()),
        };
        let json_minmax = serde_json::to_string(&s_minmax).unwrap();
        let parsed_minmax: BoxStatsSpec = serde_json::from_str(&json_minmax).unwrap();
        assert_eq!(parsed_minmax, s_minmax);

        // IqrMultiplier variant: serializes as JSON number.
        let s_iqr = BoxStatsSpec {
            field: "v".into(),
            groupby: vec![],
            whisker_extent: WhiskerExtent::IqrMultiplier(2.0),
            name: None,
        };
        let json_iqr = serde_json::to_string(&s_iqr).unwrap();
        let parsed_iqr: BoxStatsSpec = serde_json::from_str(&json_iqr).unwrap();
        assert_eq!(parsed_iqr, s_iqr);
    }

    #[test]
    fn box_stats_single_row_all_equal() {
        // Single-row input: q1=median=q3=value, whiskers=value, IQR=0.
        pyo3::Python::initialize();
        let b = batch("v", vec![9.0]);
        let spec = BoxStatsSpec {
            field: "v".into(),
            groupby: vec![],
            whisker_extent: WhiskerExtent::IqrMultiplier(1.5),
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 1);
        let q1 = col_f64(&out, "q1");
        let median = col_f64(&out, "median");
        let q3 = col_f64(&out, "q3");
        let lo = col_f64(&out, "lower_whisker");
        let hi = col_f64(&out, "upper_whisker");
        assert!((q1[0] - 9.0).abs() < 1e-12, "q1 should be 9.0; got {}", q1[0]);
        assert!((median[0] - 9.0).abs() < 1e-12, "median should be 9.0; got {}", median[0]);
        assert!((q3[0] - 9.0).abs() < 1e-12, "q3 should be 9.0; got {}", q3[0]);
        assert!((lo[0] - 9.0).abs() < 1e-12, "lower whisker should be 9.0; got {}", lo[0]);
        assert!((hi[0] - 9.0).abs() < 1e-12, "upper whisker should be 9.0; got {}", hi[0]);
    }

    #[test]
    fn box_stats_constant_values_iqr_zero() {
        // 10 identical values → q1=q3=7.0, IQR=0, whiskers=7.0.
        pyo3::Python::initialize();
        let b = batch("v", vec![7.0; 10]);
        let spec = BoxStatsSpec {
            field: "v".into(),
            groupby: vec![],
            whisker_extent: WhiskerExtent::IqrMultiplier(1.5),
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 1);
        let q1 = col_f64(&out, "q1");
        let median = col_f64(&out, "median");
        let q3 = col_f64(&out, "q3");
        let lo = col_f64(&out, "lower_whisker");
        let hi = col_f64(&out, "upper_whisker");
        assert!((q1[0] - 7.0).abs() < 1e-12, "q1 should be 7.0; got {}", q1[0]);
        assert!((median[0] - 7.0).abs() < 1e-12, "median should be 7.0; got {}", median[0]);
        assert!((q3[0] - 7.0).abs() < 1e-12, "q3 should be 7.0; got {}", q3[0]);
        assert!((lo[0] - 7.0).abs() < 1e-12, "lower whisker should be 7.0; got {}", lo[0]);
        assert!((hi[0] - 7.0).abs() < 1e-12, "upper whisker should be 7.0; got {}", hi[0]);
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

    /// FA-7: box_stats grouped on an Int64 column groups correctly and preserves
    /// the Int64 output dtype. Before FA-7 the private `KeyValue` enum rejected
    /// non-Float64/Utf8 groupby columns with an error.
    #[test]
    fn box_stats_int64_groupby_succeeds_and_preserves_dtype() {
        pyo3::Python::initialize();
        // Group 1: [1,2,3,4,5] q1=2 median=3 q3=4
        // Group 2: [10,20,30,40,50] q1=20 median=30 q3=40
        let b = batch_value_int64_group(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        );
        let spec = BoxStatsSpec {
            field: "v".into(),
            groupby: vec!["grp".into()],
            whisker_extent: WhiskerExtent::IqrMultiplier(1.5),
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 2);
        assert_eq!(
            out.schema().field(out.schema().index_of("grp").unwrap()).data_type(),
            &DataType::Int64
        );
        let grps = col_i64(&out, "grp");
        let median = col_f64(&out, "median");
        let g1 = grps.iter().position(|&g| g == 1).unwrap();
        let g2 = grps.iter().position(|&g| g == 2).unwrap();
        assert!((median[g1] - 3.0).abs() < 1e-12, "group 1 median: {}", median[g1]);
        assert!((median[g2] - 30.0).abs() < 1e-12, "group 2 median: {}", median[g2]);
    }

    /// FA-7 byte-stability guard: a Utf8 groupby still materializes a StringArray
    /// with identical per-group quartiles after migrating to the shared module.
    #[test]
    fn box_stats_utf8_groupby_byte_stable() {
        pyo3::Python::initialize();
        let b = batch_value_group(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            vec!["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        );
        let spec = BoxStatsSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            whisker_extent: WhiskerExtent::IqrMultiplier(1.5),
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(
            out.schema().field(out.schema().index_of("group").unwrap()).data_type(),
            &DataType::Utf8
        );
        let groups = col_str(&out, "group");
        let median = col_f64(&out, "median");
        let a = groups.iter().position(|g| g == "a").unwrap();
        assert!((median[a] - 3.0).abs() < 1e-12);
    }
}
