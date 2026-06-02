//! Data transform: TopK — keep top-k rows by an aggregate value.
//!
//! Groups by a field, computes an aggregate, then keeps only the top-k groups.

use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::compute::take;
use arrow::datatypes::DataType;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct TopKSpec {
    /// Number of top groups to keep.
    pub n: usize,
    /// Field to aggregate for ranking.
    pub field: String,
    /// Aggregation operation: sum, mean, count, min, max.
    #[serde(default = "default_op")]
    pub op: String,
    /// Sort direction: "descending" (default) or "ascending".
    #[serde(default = "default_sort")]
    pub sort: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_op() -> String {
    "sum".into()
}
fn default_sort() -> String {
    "descending".into()
}

/// Extract a numeric column as a `Vec<Option<f64>>` for every supported
/// primitive integer and float Arrow dtype.
///
/// Null slots are preserved as `None`. Returns `None` (not an error) for
/// non-numeric dtypes so the caller can fall back to a count aggregate instead.
/// This mirrors `crate::render::arrow_cast::col_as_f64` semantics but avoids a
/// cross-layer dependency (transform → render) by inlining the same dispatch
/// locally with a narrower scope (no temporal types needed here).
fn numeric_col_as_f64(col: &dyn Array) -> Option<Vec<Option<f64>>> {
    macro_rules! collect_as {
        ($t:ty) => {{
            let a = col.as_any().downcast_ref::<$t>()?;
            return Some(a.iter().map(|v| v.map(|x| x as f64)).collect());
        }};
    }
    match col.data_type() {
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>()?;
            Some(a.iter().collect())
        }
        DataType::Float32 => collect_as!(Float32Array),
        DataType::Int64   => collect_as!(Int64Array),
        DataType::Int32   => collect_as!(Int32Array),
        DataType::Int16   => collect_as!(Int16Array),
        DataType::Int8    => collect_as!(Int8Array),
        DataType::UInt64  => collect_as!(UInt64Array),
        DataType::UInt32  => collect_as!(UInt32Array),
        DataType::UInt16  => collect_as!(UInt16Array),
        DataType::UInt8   => collect_as!(UInt8Array),
        _ => None,
    }
}

/// Format a single numeric value from `col` at `row` as a stable group key
/// string.  Uses the same formatting as the aggregate coercion so that group
/// keys and aggregate lookups are consistent across all numeric dtypes.
fn numeric_key_at(col: &dyn Array, row: usize) -> Option<String> {
    macro_rules! int_key {
        ($t:ty) => {{
            if let Some(a) = col.as_any().downcast_ref::<$t>() {
                return Some(a.value(row).to_string());
            }
        }};
    }
    // Float64: integer-format when whole, fractional otherwise.
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        let v = a.value(row);
        return Some(format!("{v}"));
    }
    if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        let v = a.value(row);
        return Some(format!("{v}"));
    }
    int_key!(Int64Array);
    int_key!(Int32Array);
    int_key!(Int16Array);
    int_key!(Int8Array);
    int_key!(UInt64Array);
    int_key!(UInt32Array);
    int_key!(UInt16Array);
    int_key!(UInt8Array);
    None
}

pub(crate) fn apply(spec: &TopKSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    let col_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_top_k: column '{}' not found", spec.field))
    })?;

    // Use the field column as both the groupby and the aggregate target.
    // TopK semantics: group by field, aggregate, keep top-n groups.
    let col = batch.column(col_idx);

    // Coerce to f64 once; None means non-numeric (falls back to count aggregate).
    let numeric_vals: Option<Vec<Option<f64>>> = numeric_col_as_f64(col.as_ref());

    // Build groups — keys for numeric dtypes use the coerced numeric value
    // so that equal integers collapse to the same group.
    let mut groups: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for row in 0..n_rows {
        let key = if col.is_null(row) {
            "__null__".to_string()
        } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
            arr.value(row).to_string()
        } else if let Some(k) = numeric_key_at(col.as_ref(), row) {
            k
        } else {
            format!("row_{row}")
        };
        groups.entry(key).or_default().push(row);
    }

    let mut group_aggs: Vec<(String, f64)> = groups
        .iter()
        .map(|(key, rows)| {
            let agg_val = match &numeric_vals {
                Some(vals) => {
                    let col_vals: Vec<f64> = rows
                        .iter()
                        .filter_map(|&r| {
                            let v = vals[r]?;
                            if v.is_nan() { None } else { Some(v) }
                        })
                        .collect();
                    compute_agg(&col_vals, &spec.op)
                }
                None => {
                    // Non-numeric column: fall back to count.
                    rows.len() as f64
                }
            };
            (key.clone(), agg_val)
        })
        .collect();

    // Sort by aggregate.
    if spec.sort == "ascending" {
        group_aggs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        group_aggs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Keep top-n groups.
    let top_groups: std::collections::HashSet<&str> = group_aggs
        .iter()
        .take(spec.n)
        .map(|(k, _)| k.as_str())
        .collect();

    // Collect row indices that belong to top groups.
    let mut keep_indices: Vec<u32> = Vec::new();
    for (key, rows) in &groups {
        if top_groups.contains(key.as_str()) {
            for &r in rows {
                keep_indices.push(r as u32);
            }
        }
    }
    keep_indices.sort_unstable();

    let idx_array = UInt32Array::from(keep_indices);
    let columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| take(batch.column(i).as_ref(), &idx_array, None))
        .collect::<Result<_, _>>()
        .map_err(|e| PyValueError::new_err(format!("data_top_k: take: {e}")))?;

    RecordBatch::try_new(schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_top_k: {e}")))
}

fn compute_agg(vals: &[f64], op: &str) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    match op {
        "sum" => vals.iter().sum(),
        "mean" => vals.iter().sum::<f64>() / vals.len() as f64,
        "count" => vals.len() as f64,
        "min" => vals.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        "max" => vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        _ => vals.iter().sum(),
    }
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "TopK")]
#[derive(Debug, Clone)]
pub(crate) struct PyTopK(pub(crate) TransformSpec);

#[pymethods]
impl PyTopK {
    #[new]
    #[pyo3(signature = (n, field, *, op = "sum", sort = "descending", name = None))]
    fn new(n: usize, field: String, op: &str, sort: &str, name: Option<String>) -> Self {
        PyTopK(TransformSpec::TopK(TopKSpec {
            n,
            field,
            op: op.into(),
            sort: sort.into(),
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::TopK(s) => format!("TopK(n={}, field='{}')", s.n, s.field),
            _ => "TopK(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_batch_string_float(cats: Vec<&str>, vals: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(cats)),
                Arc::new(Float64Array::from(vals)),
            ],
        )
        .unwrap()
    }

    /// Original regression test — Float64 group-by still works.
    #[test]
    fn top_k_keeps_top_groups() {
        let batch = make_batch_string_float(
            vec!["a", "a", "b", "b", "c", "c"],
            vec![1.0, 2.0, 10.0, 20.0, 5.0, 5.0],
        );

        // Use count op on a string column — all groups have count 2.
        let spec = TopKSpec {
            n: 2,
            field: "cat".into(),
            op: "count".into(),
            sort: "descending".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // All groups have count 2, so top 2 of 3 groups → 4 rows.
        assert_eq!(out.num_rows(), 4);
    }

    /// BUG FIX: Int64 field with op="sum".
    ///
    /// Data layout (group key = the Int64 value itself):
    ///   value=10  appears 3 times → sum=30
    ///   value=100 appears 1 time  → sum=100
    ///   value=5   appears 2 times → sum=10
    ///
    /// Sum-ranking (desc): 100 → 30 → 10  → top 2 = {100, 10}
    /// Count-ranking (desc): 10 → 5 → 100  → top 2 = {10, 5}  (WRONG old behaviour)
    ///
    /// The test is constructed so that the count-ranking top 2 and the sum-ranking
    /// top 2 are DIFFERENT sets, ensuring the test fails on the old buggy code.
    #[test]
    fn top_k_int64_sum_uses_sum_not_count() {
        // field column is Int64
        let schema = Arc::new(Schema::new(vec![
            Field::new("score", DataType::Int64, false),
        ]));
        // score: [10, 10, 10, 100, 5, 5]
        //   group 10  → 3 rows, sum=30
        //   group 100 → 1 row,  sum=100
        //   group 5   → 2 rows, sum=10
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![10i64, 10, 10, 100, 5, 5]))],
        )
        .unwrap();

        let spec = TopKSpec {
            n: 2,
            field: "score".into(),
            op: "sum".into(),
            sort: "descending".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        // Top 2 by sum → groups {100, 10} → 1+3 = 4 rows.
        assert_eq!(out.num_rows(), 4, "expected 4 rows (groups 100 and 10)");

        // Verify that value=5 is NOT in the output (it has the lowest sum).
        let score_col = out
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..out.num_rows() {
            assert_ne!(
                score_col.value(i),
                5,
                "value=5 (lowest sum group) should have been filtered out"
            );
        }
    }

    /// BUG FIX: Int32 field with op="max".
    ///
    /// Data layout (group key = the Int32 value itself):
    ///   value=1  appears 3 times → max=1
    ///   value=50 appears 1 time  → max=50
    ///   value=20 appears 2 times → max=20
    ///
    /// Max-ranking (desc): 50 → 20 → 1   → top 2 = {50, 20}
    /// Count-ranking (desc): 1 → 20 → 50  → top 2 = {1, 20}   (WRONG old behaviour)
    #[test]
    fn top_k_int32_max_uses_max_not_count() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Int32, false),
        ]));
        // val: [1, 1, 1, 50, 20, 20]
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1i32, 1, 1, 50, 20, 20]))],
        )
        .unwrap();

        let spec = TopKSpec {
            n: 2,
            field: "val".into(),
            op: "max".into(),
            sort: "descending".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        // Top 2 by max → groups {50, 20} → 1+2 = 3 rows.
        assert_eq!(out.num_rows(), 3, "expected 3 rows (groups 50 and 20)");

        let val_col = out
            .column_by_name("val")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..out.num_rows() {
            assert_ne!(
                val_col.value(i),
                1,
                "value=1 (lowest max group) should have been filtered out"
            );
        }
    }

    /// BUG FIX: Equal integer values group together (not as separate rows).
    ///
    /// Three rows with Int64 value=7 must form one group of 3, not three
    /// groups of 1.  With n=1 and op="sum", the single group must survive.
    #[test]
    fn top_k_int_groups_equal_values_together() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, false),
        ]));
        // All rows have value 7 → one group, sum=21.
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![7i64, 7, 7]))],
        )
        .unwrap();

        let spec = TopKSpec {
            n: 1,
            field: "x".into(),
            op: "sum".into(),
            sort: "descending".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // If grouped correctly → 1 group of 3 rows is kept.
        assert_eq!(out.num_rows(), 3, "all 3 rows share the same group key");
    }

    /// Regression: Float64 field with op="sum" still works correctly.
    #[test]
    fn top_k_float64_sum_unchanged() {
        let batch = make_batch_string_float(
            vec!["a", "a", "b", "b", "c"],
            vec![10.0, 20.0, 1.0, 2.0, 100.0],
        );

        // Top 2 by sum of val using val as the field (each val unique → 1-row groups).
        // Groups: 10→10, 20→20, 1→1, 2→2, 100→100  — top 2 = {100, 20}
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, false),
        ]));
        let batch2 = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![10.0, 20.0, 1.0, 2.0, 100.0]))],
        )
        .unwrap();

        let spec = TopKSpec {
            n: 2,
            field: "val".into(),
            op: "sum".into(),
            sort: "descending".into(),
            name: None,
        };
        let out = apply(&spec, &batch2).unwrap();
        // Top 2 unique-val groups → 2 rows.
        assert_eq!(out.num_rows(), 2);

        let val_col = out
            .column_by_name("val")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let kept: Vec<f64> = (0..out.num_rows()).map(|i| val_col.value(i)).collect();
        assert!(kept.contains(&100.0), "100.0 must be in top-2 by sum");
        assert!(kept.contains(&20.0), "20.0 must be in top-2 by sum");
    }
}
