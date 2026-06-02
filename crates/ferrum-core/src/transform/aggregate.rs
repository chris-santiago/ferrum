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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum AggFn {
    Mean,
    Sum,
    Count,
    Min,
    Max,
    Median,
    Variance,
    Stdev,
    Q1,
    Q3,
    Distinct,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct AggregateOp {
    pub field: String,
    #[serde(rename = "fn")]
    pub fn_: AggFn,
    #[serde(rename = "as")]
    pub as_: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct AggregateSpec {
    pub ops: Vec<AggregateOp>,
    #[serde(default)]
    pub groupby: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &AggregateSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    if spec.ops.is_empty() {
        return Err(PyValueError::new_err("stat_aggregate: ops must be non-empty"));
    }
    let schema = batch.schema();

    // Validate fields/dtypes for ops.
    // Count with an empty field (i.e., `count():Q` shorthand) uses the batch row count
    // directly — no column lookup is needed, so we skip validation in that case.
    for op in &spec.ops {
        let is_count_star = op.fn_ == AggFn::Count && (op.field.is_empty() || op.field == "*");
        if is_count_star {
            continue;
        }
        let idx = schema.index_of(&op.field).map_err(|_| {
            PyValueError::new_err(format!(
                "stat_aggregate: column '{}' not found", op.field
            ))
        })?;
        if op.fn_ != AggFn::Count && schema.field(idx).data_type() != &DataType::Float64 {
            return Err(PyValueError::new_err(format!(
                "stat_aggregate: op field '{}' must be Float64", op.field
            )));
        }
    }

    // Validate groupby fields and capture their dtypes for output schema preservation.
    let mut group_dtypes: Vec<DataType> = Vec::with_capacity(spec.groupby.len());
    for g in &spec.groupby {
        let idx = schema.index_of(g).map_err(|_| {
            PyValueError::new_err(format!(
                "stat_aggregate: groupby column '{}' not found", g
            ))
        })?;
        let dt = schema.field(idx).data_type().clone();
        if !is_groupby_supported_dtype(&dt) {
            return Err(PyValueError::new_err(format!(
                "stat_aggregate: groupby column '{}' has unsupported dtype {:?}; \
                 supported: Utf8/LargeUtf8, Float64/Float32, \
                 Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean",
                g, dt
            )));
        }
        group_dtypes.push(dt);
    }

    // Build a per-row group key, then collect row indices per key.
    let n_rows = batch.num_rows();
    let group_arrays: Vec<&dyn arrow::array::Array> =
        spec.groupby.iter().map(|g| batch.column(
            schema.index_of(g).expect("invariant: groupby columns validated above")
        ).as_ref()).collect();

    let mut groups: BTreeMap<Vec<KeyValue>, Vec<usize>> = BTreeMap::new();
    for row in 0..n_rows {
        let mut key = Vec::with_capacity(spec.groupby.len());
        for (gi, arr) in group_arrays.iter().enumerate() {
            let kv = groupby_key_at(*arr, &group_dtypes[gi], row)
                .ok_or_else(|| PyValueError::new_err(format!(
                    "stat_aggregate: internal error extracting groupby key at row {row}"
                )))?;
            key.push(kv);
        }
        groups.entry(key).or_default().push(row);
    }

    // No groupby → single global group containing all rows.
    if spec.groupby.is_empty() {
        let all_rows: Vec<usize> = (0..n_rows).collect();
        groups.clear();
        groups.insert(Vec::new(), all_rows);
    }

    // Materialize op columns: per (group, op) compute aggregate.
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::with_capacity(groups.len());
    let mut op_values_out: Vec<Vec<f64>> =
        (0..spec.ops.len()).map(|_| Vec::with_capacity(groups.len())).collect();

    for (key, rows) in &groups {
        group_keys_out.push(key.clone());
        for (op_i, op) in spec.ops.iter().enumerate() {
            // count():Q shorthand — empty field or "*" means count all rows in the group.
            let is_count_star = op.fn_ == AggFn::Count && (op.field.is_empty() || op.field == "*");
            if is_count_star {
                op_values_out[op_i].push(rows.len() as f64);
                continue;
            }

            let col = batch
                .column(schema.index_of(&op.field).expect("invariant: op fields validated above"));

            if op.fn_ == AggFn::Count && col.data_type() != &DataType::Float64 {
                // Count on non-numeric columns: count non-null rows.
                let non_null = rows.iter().filter(|&&r| !col.is_null(r)).count();
                op_values_out[op_i].push(non_null as f64);
            } else {
                let arr = col
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| PyValueError::new_err(format!(
                        "stat_aggregate: expected Float64Array for op field '{}'", op.field
                    )))?;
                // Filter to non-null, non-NaN values within this group.
                let vals: Vec<f64> = rows.iter().filter_map(|&r| {
                    if arr.is_null(r) { return None; }
                    let v = arr.value(r);
                    if v.is_nan() { return None; }
                    Some(v)
                }).collect();
                let result = aggregate(&vals, op.fn_, rows.len());
                op_values_out[op_i].push(result);
            }
        }
    }

    // Build output schema and arrays.
    let mut fields: Vec<Field> = Vec::with_capacity(spec.groupby.len() + spec.ops.len());
    for (gi, g) in spec.groupby.iter().enumerate() {
        fields.push(Field::new(g, group_dtypes[gi].clone(), false));
    }
    for op in &spec.ops {
        fields.push(Field::new(&op.as_, DataType::Float64, true));
    }
    let out_schema = Arc::new(Schema::new(fields));

    let mut cols: Vec<ArrayRef> = Vec::with_capacity(spec.groupby.len() + spec.ops.len());
    for (gi, dt) in group_dtypes.iter().enumerate() {
        let col = materialize_groupby_col(&group_keys_out, gi, dt)
            .map_err(PyValueError::new_err)?;
        cols.push(col);
    }
    for op_vec in op_values_out.into_iter() {
        cols.push(Arc::new(Float64Array::from(op_vec)));
    }

    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_aggregate: {e}")))
}

pub(crate) fn aggregate(vals: &[f64], fn_: AggFn, group_size_including_nulls: usize) -> f64 {
    if vals.is_empty() {
        // Per spec §4.4: all-null group → NaN. Count is the exception.
        return if matches!(fn_, AggFn::Count | AggFn::Distinct) {
            0.0
        } else {
            f64::NAN
        };
    }
    let _ = group_size_including_nulls; // reserved for a future "count_all" variant
    match fn_ {
        AggFn::Mean => vals.iter().sum::<f64>() / vals.len() as f64,
        AggFn::Sum  => vals.iter().sum(),
        AggFn::Count => vals.len() as f64,
        AggFn::Min => vals.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        AggFn::Max => vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        AggFn::Median => {
            let mut sorted = vals.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            if n % 2 == 1 { sorted[n / 2] }
            else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) }
        }
        AggFn::Variance => {
            if vals.len() < 2 { return f64::NAN; }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let sum_sq: f64 = vals.iter().map(|&x| (x - mean).powi(2)).sum();
            // Sample variance: Bessel's correction (n-1 denominator)
            sum_sq / (vals.len() - 1) as f64
        }
        AggFn::Stdev => {
            if vals.len() < 2 { return f64::NAN; }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let sum_sq: f64 = vals.iter().map(|&x| (x - mean).powi(2)).sum();
            (sum_sq / (vals.len() - 1) as f64).sqrt()
        }
        AggFn::Q1 => quantile(vals, 0.25),
        AggFn::Q3 => quantile(vals, 0.75),
        AggFn::Distinct => {
            use std::collections::HashSet;
            // Represent each f64 by its bits for hashing; NaN was already filtered.
            let unique: HashSet<u64> = vals.iter().map(|&v| v.to_bits()).collect();
            unique.len() as f64
        }
    }
}

/// Linear-interpolation quantile matching numpy's `np.quantile(method='linear')`.
///
/// Sorts `vals`, computes the virtual index `h = p * (n - 1)`, then linearly
/// interpolates between `sorted[floor(h)]` and `sorted[ceil(h)]`.
pub(crate) fn quantile(vals: &[f64], p: f64) -> f64 {
    debug_assert!(!vals.is_empty(), "caller must guard against empty slice");
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 { return sorted[0]; }
    let h = p * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    let frac = h - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::transform::core::TransformSpec;

/// A single aggregation operation descriptor for use with ``Aggregate``.
///
/// Specifies one field→output-name aggregation with a named function.
/// Pass a list of these to ``Aggregate(ops=[...])``.
///
/// Parameters
/// ----------
/// field : str
///     Column to aggregate (must exist in the input batch).
/// fn_ : {"mean", "sum", "count", "min", "max", "median"}
///     Aggregation function to apply.
/// as_ : str
///     Name of the output column produced by this operation.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> op = fm.AggregateOp("price", "mean", "mean_price")
/// >>> agg = fm.Aggregate([op], groupby=["cut"])
#[pyclass(eq, module = "ferrum._core", name = "AggregateOp")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyAggregateOp(pub(crate) AggregateOp);

#[pymethods]
impl PyAggregateOp {
    #[new]
    #[pyo3(signature = (field, fn_, as_))]
    fn new(field: &str, fn_: &str, as_: &str) -> PyResult<Self> {
        if as_.is_empty() {
            return Err(PyValueError::new_err("AggregateOp: as_ must be non-empty"));
        }
        let parsed = match fn_ {
            "mean"     => AggFn::Mean,
            "sum"      => AggFn::Sum,
            "count"    => AggFn::Count,
            "min"      => AggFn::Min,
            "max"      => AggFn::Max,
            "median"   => AggFn::Median,
            "variance" | "var"   => AggFn::Variance,
            "stdev"    => AggFn::Stdev,
            "q1"       => AggFn::Q1,
            "q3"       => AggFn::Q3,
            "distinct" => AggFn::Distinct,
            other => return Err(PyValueError::new_err(format!(
                "AggregateOp: unknown fn '{other}'; expected \
                 mean|sum|count|min|max|median|variance|stdev|q1|q3|distinct"
            ))),
        };
        // count() with empty field is the `count():Q` shorthand — allowed.
        if field.is_empty() && parsed != AggFn::Count {
            return Err(PyValueError::new_err(
                "AggregateOp: field must be non-empty (use \"*\" for count of all rows)"
            ));
        }
        Ok(PyAggregateOp(AggregateOp {
            field: field.to_string(), fn_: parsed, as_: as_.to_string(),
        }))
    }

    fn __repr__(&self) -> String {
        format!("AggregateOp(field='{}', fn='{:?}', as_='{}')",
            self.0.field, self.0.fn_, self.0.as_)
    }
}

/// Group-by aggregation transform.
///
/// Computes one or more scalar summaries (mean, sum, count, min, max, median)
/// per group, reducing the batch to one row per unique group key combination.
/// When ``groupby`` is empty the entire batch is collapsed to a single row.
///
/// Parameters
/// ----------
/// ops : list of AggregateOp
///     Aggregation operations to perform, each specifying ``field``,
///     ``fn_``, and output ``as_``. Must be non-empty.
/// groupby : list of str, optional
///     Column names to group by. Default is ``[]`` (whole-batch aggregate).
/// name : str, optional
///     Named output label used by ``Reorder(from_=...)`` to look up this
///     transform's output. Ignored when no sibling references it.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> op = fm.AggregateOp("price", "mean", "mean_price")
/// >>> agg = fm.Aggregate([op], groupby=["cut"])
/// >>> fm.Chart(df).mark_bar().encode(x="cut", y="mean_price",
/// ...     transform=agg)
#[pyclass(eq, module = "ferrum._core", name = "Aggregate")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyAggregate(pub(crate) TransformSpec);

#[pymethods]
impl PyAggregate {
    #[new]
    #[pyo3(signature = (ops, *, groupby = None, name = None))]
    fn new(
        ops: &Bound<'_, PyAny>,
        groupby: Option<Vec<String>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let ops_list: &Bound<'_, PyList> = ops.downcast::<PyList>()
            .map_err(|_| PyValueError::new_err("Aggregate: ops must be a list of AggregateOp"))?;
        if ops_list.is_empty() {
            return Err(PyValueError::new_err("Aggregate: ops must be non-empty"));
        }
        let mut parsed_ops = Vec::with_capacity(ops_list.len());
        for (i, item) in ops_list.iter().enumerate() {
            let op = item.extract::<PyAggregateOp>().map_err(|_| {
                PyValueError::new_err(format!("Aggregate: ops[{i}] must be an AggregateOp"))
            })?;
            parsed_ops.push(op.0);
        }
        let gb = groupby.unwrap_or_default();
        // Reject duplicate field names within groupby per spec §6.
        let mut seen = std::collections::HashSet::new();
        for g in &gb {
            if !seen.insert(g.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "Aggregate: duplicate groupby field '{g}'"
                )));
            }
        }
        Ok(PyAggregate(TransformSpec::Aggregate(AggregateSpec {
            ops: parsed_ops,
            groupby: gb,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Aggregate(s) => format!(
                "Aggregate(ops=[{} ops], groupby={:?})",
                s.ops.len(), s.groupby,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch_value_group(values: Vec<Option<f64>>, groups: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",     DataType::Float64, true),
            Field::new("group", DataType::Utf8,    true),
        ]));
        let v_arr  = Float64Array::from(values);
        let g_arr  = StringArray::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap()
    }

    fn col_f64(b: &RecordBatch, name: &str) -> Vec<f64> {
        let arr = b.column(b.schema().index_of(name).unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        (0..arr.len()).map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) }).collect()
    }

    fn col_str(b: &RecordBatch, name: &str) -> Vec<String> {
        let arr = b.column(b.schema().index_of(name).unwrap())
            .as_any().downcast_ref::<StringArray>().unwrap();
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    }

    #[test]
    fn test_aggregate_mean_sum_count_min_max_per_group() {
        let batch = batch_value_group(
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0), Some(6.0)],
            vec!["a", "a", "a", "b", "b", "b"],
        );
        let spec = AggregateSpec {
            ops: vec![
                AggregateOp { field: "v".into(), fn_: AggFn::Mean,  as_: "m".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Sum,   as_: "s".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Count, as_: "c".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Min,   as_: "lo".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Max,   as_: "hi".into() },
            ],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let groups = col_str(&out, "group");
        let m = col_f64(&out, "m");
        let s = col_f64(&out, "s");
        let c = col_f64(&out, "c");
        let lo = col_f64(&out, "lo");
        let hi = col_f64(&out, "hi");

        let a_idx = groups.iter().position(|g| g == "a").unwrap();
        let b_idx = groups.iter().position(|g| g == "b").unwrap();
        assert!((m[a_idx] - 2.0).abs() < 1e-12);
        assert!((m[b_idx] - 5.0).abs() < 1e-12);
        assert!((s[a_idx] - 6.0).abs() < 1e-12);
        assert!((s[b_idx] - 15.0).abs() < 1e-12);
        assert_eq!(c[a_idx] as u64, 3);
        assert_eq!(c[b_idx] as u64, 3);
        assert!((lo[a_idx] - 1.0).abs() < 1e-12);
        assert!((hi[b_idx] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_median() {
        let batch = batch_value_group(
            vec![Some(1.0), Some(2.0), Some(100.0), Some(3.0), Some(4.0)],
            vec!["a", "a", "a", "b", "b"],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Median, as_: "med".into() }],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let groups = col_str(&out, "group");
        let med = col_f64(&out, "med");
        let a = groups.iter().position(|g| g == "a").unwrap();
        let b = groups.iter().position(|g| g == "b").unwrap();
        assert!((med[a] - 2.0).abs() < 1e-12, "median(1,2,100) = 2");
        assert!((med[b] - 3.5).abs() < 1e-12, "median(3,4) = 3.5");
    }

    #[test]
    fn test_aggregate_no_groupby_emits_single_global_row() {
        let batch = batch_value_group(
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec!["a", "b", "c"],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Mean, as_: "m".into() }],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let m = col_f64(&out, "m");
        assert!((m[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_all_null_group_field_emits_nan() {
        let batch = batch_value_group(
            vec![None, None, Some(5.0)],
            vec!["a", "a", "b"],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Mean, as_: "m".into() }],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let groups = col_str(&out, "group");
        let m = col_f64(&out, "m");
        let a = groups.iter().position(|g| g == "a").unwrap();
        let b = groups.iter().position(|g| g == "b").unwrap();
        assert!(m[a].is_nan());
        assert!((m[b] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_missing_field_errors() {
        pyo3::Python::initialize();
        let batch = batch_value_group(vec![Some(1.0)], vec!["a"]);
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "ghost".into(), fn_: AggFn::Mean, as_: "m".into() }],
            groupby: vec!["group".into()],
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("ghost"));
    }

    #[test]
    fn test_aggregate_round_trip_json() {
        let original = AggregateSpec {
            ops: vec![
                AggregateOp { field: "x".into(), fn_: AggFn::Sum, as_: "tot".into() },
                AggregateOp { field: "y".into(), fn_: AggFn::Mean, as_: "avg".into() },
            ],
            groupby: vec!["k".into()],
            name: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: AggregateSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_aggregate_count_on_utf8_column() {
        // Regression: Count on a non-Float64 (Utf8) column should count non-null rows.
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("group", DataType::Utf8, false),
        ]));
        let names = StringArray::from(vec![
            Some("alice"),
            Some("bob"),
            None,
            Some("carol"),
            Some("dave"),
            None,
        ]);
        let groups = StringArray::from(vec!["a", "a", "a", "b", "b", "b"]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(names), Arc::new(groups)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![AggregateOp {
                field: "name".into(),
                fn_: AggFn::Count,
                as_: "n".into(),
            }],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let groups_out = col_str(&out, "group");
        let counts = col_f64(&out, "n");
        let a_idx = groups_out.iter().position(|g| g == "a").unwrap();
        let b_idx = groups_out.iter().position(|g| g == "b").unwrap();
        // Group a has 2 non-null names (alice, bob); group b has 2 non-null (carol, dave).
        assert_eq!(counts[a_idx] as u64, 2);
        assert_eq!(counts[b_idx] as u64, 2);
    }

    #[test]
    fn test_aggregate_all_null_float64_mean_returns_nan() {
        // Spec §4.4: all-null column with Mean op → NaN.
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
        ]));
        let arr = Float64Array::from(vec![None, None, None, None, None]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![AggregateOp {
                field: "v".into(),
                fn_: AggFn::Mean,
                as_: "m".into(),
            }],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let m = col_f64(&out, "m");
        assert!(m[0].is_nan(), "all-null Mean should be NaN; got {}", m[0]);
    }

    // ── count() with empty field ──────────────────────────────────────────────

    #[test]
    fn test_aggregate_count_empty_field_uses_row_count() {
        pyo3::Python::initialize();
        // When field is "" (or "*") and fn_ is Count, use num_rows of the batch/group.
        let batch = batch_value_group(
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
            vec!["a", "a", "b", "b"],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp {
                field: "".into(),
                fn_: AggFn::Count,
                as_: "n".into(),
            }],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let groups = col_str(&out, "group");
        let counts = col_f64(&out, "n");
        let a = groups.iter().position(|g| g == "a").unwrap();
        let b = groups.iter().position(|g| g == "b").unwrap();
        assert_eq!(counts[a] as u64, 2);
        assert_eq!(counts[b] as u64, 2);
    }

    // ── new aggregate functions ───────────────────────────────────────────────

    #[test]
    fn test_aggregate_variance_known_values() {
        // var([2,4,4,4,5,5,7,9]) = 4.0 (sample variance, Bessel's correction)
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![
            Some(2.0), Some(4.0), Some(4.0), Some(4.0),
            Some(5.0), Some(5.0), Some(7.0), Some(9.0),
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Variance, as_: "var".into() }],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let var = col_f64(&out, "var");
        assert!((var[0] - 4.571428571428571).abs() < 1e-9, "got {}", var[0]);
    }

    #[test]
    fn test_aggregate_stdev_known_values() {
        // stdev([2,4,4,4,5,5,7,9]) = sqrt(sample_var)
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![
            Some(2.0), Some(4.0), Some(4.0), Some(4.0),
            Some(5.0), Some(5.0), Some(7.0), Some(9.0),
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Stdev, as_: "sd".into() }],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let sd = col_f64(&out, "sd");
        assert!((sd[0] - 2.138089935325867).abs() < 1e-9, "got {}", sd[0]);
    }

    #[test]
    fn test_aggregate_q1_q3_known_values() {
        // numpy default (linear interpolation): Q1([1,2,3,4]) = 1.75, Q3 = 3.25
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![
            Some(1.0), Some(2.0), Some(3.0), Some(4.0),
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![
                AggregateOp { field: "v".into(), fn_: AggFn::Q1, as_: "q1".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Q3, as_: "q3".into() },
            ],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let q1 = col_f64(&out, "q1");
        let q3 = col_f64(&out, "q3");
        assert!((q1[0] - 1.75).abs() < 1e-9, "Q1 got {}", q1[0]);
        assert!((q3[0] - 3.25).abs() < 1e-9, "Q3 got {}", q3[0]);
    }

    #[test]
    fn test_aggregate_q1_q3_odd_count() {
        // Q1([1,2,3,4,5]) = 2.0, Q3 = 4.0 (numpy linear)
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![
            Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0),
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![
                AggregateOp { field: "v".into(), fn_: AggFn::Q1, as_: "q1".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Q3, as_: "q3".into() },
            ],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let q1 = col_f64(&out, "q1");
        let q3 = col_f64(&out, "q3");
        assert!((q1[0] - 2.0).abs() < 1e-9, "Q1 got {}", q1[0]);
        assert!((q3[0] - 4.0).abs() < 1e-9, "Q3 got {}", q3[0]);
    }

    #[test]
    fn test_aggregate_distinct_count() {
        let batch = batch_value_group(
            vec![Some(1.0), Some(1.0), Some(2.0), Some(3.0), Some(3.0), Some(4.0)],
            vec!["a", "a", "a", "b", "b", "b"],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp {
                field: "v".into(),
                fn_: AggFn::Distinct,
                as_: "d".into(),
            }],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let groups = col_str(&out, "group");
        let d = col_f64(&out, "d");
        let a = groups.iter().position(|g| g == "a").unwrap();
        let b = groups.iter().position(|g| g == "b").unwrap();
        // group a: {1.0, 2.0} → 2 distinct; group b: {3.0, 4.0} → 2 distinct
        assert_eq!(d[a] as u64, 2);
        assert_eq!(d[b] as u64, 2);
    }

    #[test]
    fn test_aggregate_fn_serde_roundtrip_new_variants() {
        // Verify that new AggFn variants serialize and deserialize correctly via
        // their canonical serde names (lowercase, from rename_all = "lowercase").
        // Python-layer aliases ("var", "stdevp") are tested via PyAggregateOp::new.
        let cases = vec![
            (AggFn::Variance, "variance"),
            (AggFn::Stdev,    "stdev"),
            (AggFn::Q1,       "q1"),
            (AggFn::Q3,       "q3"),
            (AggFn::Distinct, "distinct"),
        ];
        for (fn_, canonical_name) in cases {
            // Serialize the enum variant.
            let json = serde_json::to_string(&fn_).unwrap();
            assert_eq!(json, format!("\"{}\"", canonical_name), "unexpected serde name for {:?}", fn_);

            // Deserialize back from the canonical name.
            let json_op = format!(r#"{{"field":"x","fn":"{}","as":"out"}}"#, canonical_name);
            let parsed: AggregateOp = serde_json::from_str(&json_op).unwrap_or_else(|e| {
                panic!("Failed to deserialize fn '{}': {}", canonical_name, e)
            });
            assert_eq!(parsed.fn_, fn_, "fn '{}' should round-trip to {:?}", canonical_name, fn_);
        }
    }

    // ── FA-3: integer groupby columns ─────────────────────────────────────────

    fn batch_value_int64_group(values: Vec<Option<f64>>, groups: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::Int64,   false),
        ]));
        let v_arr = Float64Array::from(values);
        let g_arr = Int64Array::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap()
    }

    fn col_i64(b: &RecordBatch, name: &str) -> Vec<i64> {
        let arr = b.column(b.schema().index_of(name).unwrap())
            .as_any().downcast_ref::<Int64Array>().unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    #[test]
    fn test_aggregate_int64_groupby_sum() {
        // FA-3: sum grouped on an Int64 column produces correct per-group sums.
        let batch = batch_value_int64_group(
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
            vec![10, 10, 20, 20, 30],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 3);

        // Output groupby column must retain Int64 dtype.
        assert_eq!(out.schema().field(0).data_type(), &DataType::Int64);

        let grps = col_i64(&out, "grp");
        let sums = col_f64(&out, "s");
        let g10 = grps.iter().position(|&g| g == 10).unwrap();
        let g20 = grps.iter().position(|&g| g == 20).unwrap();
        let g30 = grps.iter().position(|&g| g == 30).unwrap();

        assert!((sums[g10] - 3.0).abs() < 1e-12, "group 10 sum: {}", sums[g10]);
        assert!((sums[g20] - 7.0).abs() < 1e-12, "group 20 sum: {}", sums[g20]);
        assert!((sums[g30] - 5.0).abs() < 1e-12, "group 30 sum: {}", sums[g30]);
    }

    #[test]
    fn test_aggregate_int64_groupby_mean() {
        // FA-3: mean grouped on Int64 groupby.
        let batch = batch_value_int64_group(
            vec![Some(1.0), Some(3.0), Some(2.0), Some(4.0)],
            vec![2020, 2020, 2021, 2021],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Mean, as_: "m".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);

        let grps = col_i64(&out, "grp");
        let means = col_f64(&out, "m");
        let y2020 = grps.iter().position(|&g| g == 2020).unwrap();
        let y2021 = grps.iter().position(|&g| g == 2021).unwrap();
        assert!((means[y2020] - 2.0).abs() < 1e-12, "2020 mean: {}", means[y2020]);
        assert!((means[y2021] - 3.0).abs() < 1e-12, "2021 mean: {}", means[y2021]);
    }

    #[test]
    fn test_aggregate_int64_groupby_count() {
        // FA-3: count grouped on Int64 groupby.
        let batch = batch_value_int64_group(
            vec![Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0)],
            vec![1, 1, 1, 2, 2],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Count, as_: "c".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);

        let grps = col_i64(&out, "grp");
        let counts = col_f64(&out, "c");
        let g1 = grps.iter().position(|&g| g == 1).unwrap();
        let g2 = grps.iter().position(|&g| g == 2).unwrap();
        assert_eq!(counts[g1] as u64, 3, "group 1 count: {}", counts[g1]);
        assert_eq!(counts[g2] as u64, 2, "group 2 count: {}", counts[g2]);
    }

    #[test]
    fn test_aggregate_int32_groupby_sum() {
        // FA-3: Int32 groupby also accepted.
        use arrow::array::Int32Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::Int32,   false),
        ]));
        let v_arr = Float64Array::from(vec![Some(10.0), Some(20.0), Some(30.0)]);
        let g_arr = Int32Array::from(vec![1i32, 1, 2]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap();

        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        // Output dtype preserved as Int32.
        assert_eq!(out.schema().field(0).data_type(), &DataType::Int32);
    }

    #[test]
    fn test_aggregate_uint32_groupby_sum() {
        // FA-3: UInt32 groupby also accepted.
        use arrow::array::UInt32Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::UInt32,  false),
        ]));
        let v_arr = Float64Array::from(vec![Some(5.0), Some(15.0), Some(25.0)]);
        let g_arr = UInt32Array::from(vec![100u32, 100, 200]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap();

        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        assert_eq!(out.schema().field(0).data_type(), &DataType::UInt32);
    }

    // ── FA-3 null-key grouping regression ────────────────────────────────────

    /// Build a batch with a nullable Float64 groupby column.
    fn batch_value_float64_group_nullable(
        values: Vec<Option<f64>>,
        groups: Vec<Option<f64>>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::Float64, true),
        ]));
        let v_arr = Float64Array::from(values);
        let g_arr = Float64Array::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap()
    }

    /// Build a batch with a nullable Int64 groupby column.
    fn batch_value_int64_group_nullable(
        values: Vec<Option<f64>>,
        groups: Vec<Option<i64>>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::Int64,   true),
        ]));
        let v_arr = Float64Array::from(values);
        let g_arr = Int64Array::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap()
    }

    #[test]
    fn test_aggregate_null_float64_groupby_key_collapses_to_single_group() {
        // FA-3 regression: null keys in a Float64 groupby column must all land in
        // the SAME group (KeyValue::Null), separate from any real-valued groups.
        //
        // Materialization contract (Float64 null key → output NaN):
        //   row 0: grp=None (null)  → KeyValue::Null  → output NaN
        //   row 1: grp=None (null)  → KeyValue::Null  → same group as row 0
        //   row 2: grp=Some(1.0)    → KeyValue::Float → separate group
        //
        // Pre-FA-3 bug: null was encoded as Float(NAN.to_bits()), which would
        // collide with a genuine NaN *value* in the groupby column and could
        // produce multiple spurious groups if NaN bits differ.
        let batch = batch_value_float64_group_nullable(
            vec![Some(10.0), Some(20.0), Some(5.0)],
            vec![None,        None,        Some(1.0)],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        // Exactly two groups: the null group and the real-valued group.
        assert_eq!(out.num_rows(), 2, "expected 2 groups (null + 1.0), got {}", out.num_rows());

        let grp_arr = out.column(out.schema().index_of("grp").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        let sums = col_f64(&out, "s");

        // Find the group whose key is NaN (the null-key proxy) and the one whose key is 1.0.
        let null_idx = (0..grp_arr.len()).find(|&i| grp_arr.value(i).is_nan())
            .expect("null-key group must materialize with NaN output key");
        let real_idx = (0..grp_arr.len()).find(|&i| (grp_arr.value(i) - 1.0).abs() < 1e-12)
            .expect("real-valued group (key=1.0) must be present");

        // Null group: rows 0 and 1 → sum = 10 + 20 = 30.
        assert!(
            (sums[null_idx] - 30.0).abs() < 1e-12,
            "null-key group sum should be 30.0, got {}",
            sums[null_idx]
        );
        // Real group: row 2 → sum = 5.
        assert!(
            (sums[real_idx] - 5.0).abs() < 1e-12,
            "real-key group (1.0) sum should be 5.0, got {}",
            sums[real_idx]
        );
    }

    #[test]
    fn test_aggregate_null_int64_groupby_key_collapses_to_single_group() {
        // FA-3 regression: null keys in an Int64 groupby column must all land in
        // the SAME group (KeyValue::Null), separate from non-null groups.
        //
        // Materialization contract (Int64 null key → output 0i64):
        //   row 0: grp=None  → KeyValue::Null → output 0
        //   row 1: grp=None  → KeyValue::Null → same group as row 0
        //   row 2: grp=Some(7) → KeyValue::Int(7) → separate group
        //
        // NOTE: the null-proxy output value is 0i64 (same as a real key of 0).
        // This is the documented behavior: null keys are grouped correctly during
        // aggregation, but the materialized output key is a proxy value (0 for
        // integers), not a typed null. Callers that need to distinguish a true
        // null group from a zero group must not use Int64 columns with both nulls
        // and a real zero key in the same groupby.
        let batch = batch_value_int64_group_nullable(
            vec![Some(3.0), Some(7.0), Some(100.0)],
            vec![None,       None,       Some(7)],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        // Exactly two groups: the null group and the real-valued group.
        assert_eq!(out.num_rows(), 2, "expected 2 groups (null + 7), got {}", out.num_rows());

        let grps = col_i64(&out, "grp");
        let sums = col_f64(&out, "s");

        // Null-key group materializes as key=0i64.
        let null_idx = grps.iter().position(|&g| g == 0)
            .expect("null-key group must materialize with output key 0i64");
        // Real-valued group: key=7.
        let real_idx = grps.iter().position(|&g| g == 7)
            .expect("real-valued group (key=7) must be present");

        // Null group: rows 0 and 1 → sum = 3 + 7 = 10.
        assert!(
            (sums[null_idx] - 10.0).abs() < 1e-12,
            "null-key group sum should be 10.0, got {}",
            sums[null_idx]
        );
        // Real group: row 2 → sum = 100.
        assert!(
            (sums[real_idx] - 100.0).abs() < 1e-12,
            "real-key group (7) sum should be 100.0, got {}",
            sums[real_idx]
        );
    }

    #[test]
    fn test_aggregate_null_float64_groupby_separate_from_genuine_nan_key() {
        // FA-3 regression: a genuine NaN *value* in a Float64 groupby column must
        // form its own group (KeyValue::Float(NAN.to_bits())), distinct from a null
        // key (KeyValue::Null).  Pre-FA-3 this was a collision: both mapped to
        // Float(NAN.to_bits()), merging what should be two separate groups.
        //
        // row 0: grp = None    → KeyValue::Null              → null group
        // row 1: grp = Some(NaN) → KeyValue::Float(NAN bits) → NaN group
        // row 2: grp = Some(2.0) → KeyValue::Float(2.0 bits) → real group
        let nan = f64::NAN;
        let batch = batch_value_float64_group_nullable(
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec![None,       Some(nan), Some(2.0)],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        // Three distinct groups: null, NaN, and 2.0.
        assert_eq!(
            out.num_rows(), 3,
            "expected 3 groups (null, NaN, 2.0), got {}; FA-3 null/NaN collision?",
            out.num_rows()
        );

        let grp_arr = out.column(out.schema().index_of("grp").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        let sums = col_f64(&out, "s");

        // Count how many output keys are NaN.
        let nan_indices: Vec<usize> = (0..grp_arr.len())
            .filter(|&i| grp_arr.value(i).is_nan())
            .collect();
        // Both the null-proxy (NaN) and the genuine-NaN group materialize with NaN
        // as their output key — they are distinguished only during grouping, not in
        // the materialized output.  So we expect exactly 2 NaN-key rows.
        assert_eq!(
            nan_indices.len(), 2,
            "expected 2 NaN-keyed output rows (one for null group, one for NaN group), got {}",
            nan_indices.len()
        );

        // The real-valued group (key=2.0) must be present with sum=3.0.
        let real_idx = (0..grp_arr.len()).find(|&i| (grp_arr.value(i) - 2.0).abs() < 1e-12)
            .expect("real-valued group (key=2.0) must be present");
        assert!(
            (sums[real_idx] - 3.0).abs() < 1e-12,
            "real-key group (2.0) sum should be 3.0, got {}",
            sums[real_idx]
        );
        // The two NaN-keyed rows together account for sums 1.0 (null group) and 2.0 (NaN group).
        let nan_sums: Vec<f64> = nan_indices.iter().map(|&i| sums[i]).collect();
        assert!(
            nan_sums.contains(&1.0) || nan_sums.iter().any(|&s| (s - 1.0).abs() < 1e-12),
            "null group (row 0) sum should be 1.0; nan_sums = {:?}",
            nan_sums
        );
        assert!(
            nan_sums.iter().any(|&s| (s - 2.0).abs() < 1e-12),
            "NaN-key group (row 1) sum should be 2.0; nan_sums = {:?}",
            nan_sums
        );
    }

    #[test]
    fn test_aggregate_unsupported_groupby_dtype_errors_clearly() {
        // FA-3 regression: genuinely unsupported dtypes (e.g., List) must raise a
        // clear error, not panic or silently miscount.
        use arrow::array::ListArray;
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::Field as ArrowField;
        pyo3::Python::initialize();

        // Construct a minimal ListArray column.
        let values = arrow::array::Int32Array::from(vec![1, 2, 3]);
        let offsets = OffsetBuffer::new(arrow::buffer::ScalarBuffer::from(vec![0i32, 2, 3]));
        let field = ArrowField::new("item", DataType::Int32, false);
        let list_arr = ListArray::new(Arc::new(field), offsets, Arc::new(values), None);

        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, false))), false),
        ]));
        let v_arr = Float64Array::from(vec![Some(1.0), Some(2.0)]);
        let batch = RecordBatch::try_new(
            schema, vec![Arc::new(v_arr), Arc::new(list_arr)]
        ).unwrap();

        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unsupported dtype"), "expected 'unsupported dtype' in: {msg}");
        assert!(msg.contains("grp"), "expected column name 'grp' in error: {msg}");
    }
}
