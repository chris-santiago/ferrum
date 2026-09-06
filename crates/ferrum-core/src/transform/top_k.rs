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

use crate::transform::group_key::{group_partition_multi, parse_groupby_pyany, KeyValue};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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
    /// When set, groups are the distinct tuples of these columns (ranked by
    /// `op` aggregated over `field`) instead of the legacy "group by the
    /// `field` value itself" behavior. `None`/absent preserves the original
    /// wire and byte-identical output (F-L05-01).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

/// The single source of truth for `TopK`'s free-string `op` vocabulary.
/// Read by [`compute_agg`] (the enforcement site), its error message, and
/// `registry::entry_for_kind("top_k")`'s `methods` list — R1 design finding
/// 1, mirroring the `smooth::METHODS` precedent.
pub(crate) const OPS: &[&str] = &["sum", "mean", "count", "min", "max"];

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
    match &spec.groupby {
        Some(cols) if !cols.is_empty() => apply_grouped(spec, batch, cols),
        _ => apply_ungrouped(spec, batch),
    }
}

/// Legacy (no `groupby`) behavior, preserved byte-identically: the `field`
/// column doubles as its own groupby key — every distinct `field` value
/// forms one group, ranked by `op` aggregated over that same column. This
/// is documented as a historical quirk (F-L05-01), not the intended
/// top-k-per-category use case; pass `groupby=` for that.
fn apply_ungrouped(spec: &TopKSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
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

    // The field column doubles as the group key here, so every non-null row
    // must be able to produce one: either a string value or a numeric key.
    // Any other dtype (e.g. Boolean, Date) used to fall back to a synthesized
    // `row_{n}` key per row (silently defeating grouping entirely — every row
    // became its own singleton group). That is now a typed refusal instead.
    if numeric_vals.is_none() && col.as_any().downcast_ref::<StringArray>().is_none() {
        return Err(PyValueError::new_err(format!(
            "data_top_k: field column '{}' has unsupported dtype {:?} for ungrouped top_k \
             (the field doubles as the group key); expected Utf8 or a numeric dtype \
             (Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Float32/Float64) — \
             pass groupby= to rank by a separate category column instead",
            spec.field,
            col.data_type()
        )));
    }

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
            unreachable!(
                "validated above: field column dtype supports key extraction (Utf8 or numeric)"
            )
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
                    compute_agg(&col_vals, &spec.op)?
                }
                None => {
                    // Non-numeric column: fall back to count.
                    rows.len() as f64
                }
            };
            Ok((key.clone(), agg_val))
        })
        .collect::<PyResult<_>>()?;

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

/// With `groupby=` set, groups are the distinct `groupby`-column tuples
/// (via the shared [`group_partition_multi`] partitioner), ranked by `op`
/// aggregated over `field`; every original row belonging to one of the
/// top-`n` groups is kept.
///
/// Unlike the `smooth.rs`/`regression_data.rs` partition-fit-concat
/// template (spec §4.3's default grouped-transform shape), top_k
/// synthesizes no new rows or columns per group — it selects a subset of
/// the batch's existing rows, so this is closer to `impute.rs`'s
/// row-preserving idea, simplified further: there is no per-row fill
/// value to compute, just a `take` over the surviving row indices. The
/// groupby columns need no separate materialization (unlike smooth/
/// regression_data/impute) because they already exist, verbatim, on every
/// kept row — `group_key`'s partitioning is used purely to find those rows.
///
/// **Null-groupby-key rows are dropped, unconditionally.** `group_partition_multi`
/// excludes any row whose groupby value is null from every group (its own
/// documented null-key policy), so such a row can never appear in a kept
/// group's index list and is never selected — regardless of `n`, even when
/// `n` is large enough to keep every real group. This mirrors `outliers.rs`'s
/// row-filter over the same partitioner: "an outlier point with no
/// corresponding box would be orphaned otherwise" applies here too, since a
/// selected row needs a group to have been ranked by. It is a deliberate
/// asymmetry with [`apply_ungrouped`], which collects null `field` values
/// into their own `"__null__"` group that CAN survive top-n: there, `field`
/// doubles as the group key so a null key is still a key; here, the groupby
/// column is a distinct category axis and a null category has nothing to
/// rank.
fn apply_grouped(spec: &TopKSpec, batch: &RecordBatch, group_cols: &[String]) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let field_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_top_k: column '{}' not found", spec.field))
    })?;
    let field_col = batch.column(field_idx);

    // Coerce to f64 once; None means non-numeric (falls back to count
    // aggregate) — same policy as the ungrouped path, but here `field` is
    // purely the aggregate target, never a group key, so no dtype refusal
    // is needed: any dtype is a valid thing to count.
    let numeric_vals: Option<Vec<Option<f64>>> = numeric_col_as_f64(field_col.as_ref());

    let (group_order, group_idx_map, _group_dtypes) =
        group_partition_multi(batch, group_cols, "data_top_k")?;

    let mut group_aggs: Vec<(Vec<KeyValue>, f64)> = group_order
        .iter()
        .map(|g| {
            let rows = &group_idx_map[g];
            let agg_val = match &numeric_vals {
                Some(vals) => {
                    let col_vals: Vec<f64> = rows
                        .iter()
                        .filter_map(|&r| {
                            let v = vals[r]?;
                            if v.is_nan() { None } else { Some(v) }
                        })
                        .collect();
                    compute_agg(&col_vals, &spec.op)?
                }
                None => {
                    // Non-numeric field: fall back to count.
                    rows.len() as f64
                }
            };
            Ok((g.clone(), agg_val))
        })
        .collect::<PyResult<_>>()?;

    // Sort by aggregate.
    if spec.sort == "ascending" {
        group_aggs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        group_aggs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Collect row indices belonging to the top-n groups, in ascending row
    // order (via the final sort), matching the ungrouped path's `take` shape.
    // `group_aggs`/`group_idx_map` only ever contain the non-null-key groups
    // `group_partition_multi` returned, so a null-groupby row is already
    // absent here and can never be kept — see the null-groupby-key policy
    // documented on this function above.
    let mut keep_indices: Vec<u32> = Vec::new();
    for (key, _) in group_aggs.iter().take(spec.n) {
        if let Some(rows) = group_idx_map.get(key) {
            keep_indices.extend(rows.iter().map(|&r| r as u32));
        }
    }
    keep_indices.sort_unstable();

    let idx_array = UInt32Array::from(keep_indices);
    let columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| take(batch.column(i).as_ref(), &idx_array, None))
        .collect::<Result<_, _>>()
        .map_err(|e| PyValueError::new_err(format!("data_top_k: take: {e}")))?;

    RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| PyValueError::new_err(format!("data_top_k: {e}")))
}

/// Rank one group under `op`.
///
/// R1 design finding 1: the unknown-`op` arm used to fall back to `sum`
/// silently — a typo'd `op` returned a different top-k with no diagnostic,
/// and the silent fallback left this declared vocabulary with no refusal
/// site to guard. It now refuses, which is what
/// `top_k_methods_accepted_and_bogus_op_rejected` (`registry/tests.rs`) holds
/// the registry entry to [`OPS`] with.
fn compute_agg(vals: &[f64], op: &str) -> PyResult<f64> {
    if vals.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(match op {
        "sum" => vals.iter().sum(),
        "mean" => vals.iter().sum::<f64>() / vals.len() as f64,
        "count" => vals.len() as f64,
        "min" => vals.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        "max" => vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        other => {
            return Err(PyValueError::new_err(format!(
                "data_top_k: unknown op '{other}'; expected {}",
                OPS.join("|"),
            )))
        }
    })
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
    #[pyo3(signature = (n, field, *, op = "sum", sort = "descending", groupby = None, name = None))]
    fn new(
        n: usize,
        field: String,
        op: &str,
        sort: &str,
        groupby: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        // Normalized through the same boundary helper as PyBin/PyKde/PyKde2d/
        // PySmooth so `str | list[str] | None` is accepted uniformly across
        // the transform family (quality-review finding, cycle 2).
        let groupby = parse_groupby_pyany(groupby)?;
        let groupby = (!groupby.is_empty()).then_some(groupby);
        Ok(PyTopK(TransformSpec::TopK(TopKSpec {
            n,
            field,
            op: op.into(),
            sort: sort.into(),
            groupby,
            name,
        })))
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
            groupby: None,
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
            groupby: None,
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
            groupby: None,
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
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // If grouped correctly → 1 group of 3 rows is kept.
        assert_eq!(out.num_rows(), 3, "all 3 rows share the same group key");
    }

    // ---- R1-relocated coverage (tests/bug_hunt_release_transforms.rs, 2026-08-27) ----
    //
    // `top_k_sum_ranking_differs_from_count_ranking` was dropped as an exact
    // duplicate of `top_k_int64_sum_uses_sum_not_count` above (identical
    // group data and "sum ranking != count ranking" contract).

    /// An empty group (all values NaN-filtered out before `compute_agg` is
    /// called) ranks last under descending top-k sort via the NEG_INFINITY
    /// sentinel — and, as a documented asymmetry, ranks FIRST under an
    /// ascending sort. The ranking claim is verified through the real
    /// `apply()` path (quality-review finding, cycle 2), not a local
    /// `Vec::sort_by` — the field doubles as its own groupby key, so two NaN
    /// rows form an all-filtered "NaN" group (empty -> NEG_INFINITY) while one
    /// `5.0` row forms a real group (agg = 5.0); ascending top-1 must keep the
    /// NaN group's 2 rows, not the real group's 1 row.
    #[test]
    fn compute_agg_empty_group_uses_neg_infinity_sentinel() {
        let empty: [f64; 0] = [];
        assert_eq!(compute_agg(&empty, "sum").unwrap(), f64::NEG_INFINITY);
        assert_eq!(compute_agg(&empty, "max").unwrap(), f64::NEG_INFINITY);

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![5.0, f64::NAN, f64::NAN]))],
        )
        .unwrap();
        let spec = TopKSpec {
            n: 1,
            field: "x".into(),
            op: "sum".into(),
            sort: "ascending".into(),
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(
            out.num_rows(), 2,
            "ascending top-1 must surface the empty (NaN) group's 2 rows, not the real 5.0 group's 1 row"
        );
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
            groupby: None,
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

    // ── M3 groupby coverage (F-L05-01) ────────────────────────────────────
    //
    // Before this fix, `TopKSpec` had no `groupby` field at all: top_k
    // always grouped by the `field` column's own distinct values, making
    // top-k-per-category impossible. These tests use the guide's own
    // product/revenue repro and are constructed so that the naive
    // "group by field value" behavior yields a DIFFERENT top-n set than
    // correct per-groupby-key aggregation — so a fix that silently ignores
    // `groupby` and falls through to the legacy path fails loudly here.

    fn make_batch_product_revenue() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("product", DataType::Utf8, false),
            Field::new("revenue", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A", "A", "B", "C", "D", "E", "E"])),
                Arc::new(Float64Array::from(vec![50.0, 55.0, 80.0, 170.0, 82.0, 40.0, 45.0])),
            ],
        )
        .unwrap()
    }

    /// F-L05-01 repro: per-product revenue sums are A=105, B=80, C=170,
    /// D=82, E=85 — top 3 by sum = {C, A, E} (spec's own {A, C, E}).
    ///
    /// The bug this pins: grouping by `revenue`'s own (mostly unique)
    /// values instead of `product` would treat each row as close to its
    /// own singleton group and rank by raw revenue value, keeping
    /// {C(170), D(82), B(80)} instead — a verifiably different set of
    /// rows and row count.
    #[test]
    fn top_k_groupby_ranks_by_group_not_by_field_value() {
        let batch = make_batch_product_revenue();
        let spec = TopKSpec {
            n: 3,
            field: "revenue".into(),
            op: "sum".into(),
            sort: "descending".into(),
            groupby: Some(vec!["product".into()]),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        let products = out
            .column_by_name("product")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let kept: std::collections::BTreeSet<&str> =
            (0..out.num_rows()).map(|i| products.value(i)).collect();
        let expected: std::collections::BTreeSet<&str> = ["A", "C", "E"].into_iter().collect();
        assert_eq!(
            kept, expected,
            "top-3 by per-product revenue sum must be {{A, C, E}}, got {kept:?}"
        );
        // A (2 rows) + C (1 row) + E (2 rows) = 5 rows; B and D dropped entirely.
        assert_eq!(out.num_rows(), 5);
    }

    /// Per-group sums verified exactly against the finding's own numbers
    /// (n=5 keeps every group, since there are exactly 5 distinct products,
    /// so this also confirms the grouped path is row-preserving: all 7
    /// input rows survive when no group is filtered out).
    #[test]
    fn top_k_groupby_per_group_sums_match_hand_computed() {
        let batch = make_batch_product_revenue();
        let spec = TopKSpec {
            n: 5,
            field: "revenue".into(),
            op: "sum".into(),
            sort: "descending".into(),
            groupby: Some(vec!["product".into()]),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 7, "n=5 keeps all 5 groups (7 rows total)");

        let products = out
            .column_by_name("product")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let revenues = out
            .column_by_name("revenue")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let mut sums: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
        for i in 0..out.num_rows() {
            *sums.entry(products.value(i).to_string()).or_insert(0.0) += revenues.value(i);
        }
        assert_eq!(sums["A"], 105.0);
        assert_eq!(sums["B"], 80.0);
        assert_eq!(sums["C"], 170.0);
        assert_eq!(sums["D"], 82.0);
        assert_eq!(sums["E"], 85.0);
    }

    /// Wire-compat: a pre-groupby-era serialized `TopKSpec` (no `"groupby"`
    /// key at all) must still deserialize, defaulting `groupby` to `None`
    /// (the ungrouped, byte-identical legacy path).
    #[test]
    fn top_k_wire_compat_old_json_without_groupby_parses() {
        let json = r#"{"n": 3, "field": "revenue", "op": "sum", "sort": "descending"}"#;
        let spec: TopKSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.groupby, None);
        assert_eq!(spec.n, 3);
        assert_eq!(spec.field, "revenue");
    }

    /// F-L05-01: the ungrouped path's `row_{n}` synthesized-key fallback
    /// (for a `field` column that is neither Utf8 nor numeric — e.g.
    /// Boolean) is now a typed refusal, not a silent per-row singleton
    /// group. Before this fix, this call succeeded (each row became its
    /// own group under a `"row_{n}"` key), silently defeating grouping.
    #[test]
    fn top_k_ungrouped_unsupported_field_dtype_errors_instead_of_row_n_fallback() {
        pyo3::Python::initialize();
        use arrow::array::BooleanArray;
        let schema = Arc::new(Schema::new(vec![Field::new("flag", DataType::Boolean, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BooleanArray::from(vec![true, false, true]))],
        )
        .unwrap();
        let spec = TopKSpec {
            n: 1,
            field: "flag".into(),
            op: "count".into(),
            sort: "descending".into(),
            groupby: None,
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("data_top_k"), "err: {msg}");
        assert!(msg.contains("flag"), "err must name the column: {msg}");
        assert!(msg.contains("Boolean"), "err must name the dtype: {msg}");
    }

    // ── round-2 quality-review coverage (F-L05-01, cycle 2) ────────────────

    /// S3-1: a null groupby-key row is dropped even when `n` is large enough
    /// to keep every real group. Fixture: 5 rows, 3 real products (A, B, C)
    /// plus one row with a null `product`; `n=10` keeps all 3 real groups, so
    /// the only way the row count can be less than 5 is the null-key drop
    /// documented on `apply_grouped`. This is the reviewer's own repro
    /// (`transform_top_k(10, field='revenue', op='sum', groupby=['product'])`
    /// -> 4 rows out of 5), re-derived through the Rust `apply()` entry point.
    #[test]
    fn top_k_groupby_null_key_row_is_dropped_even_when_n_keeps_every_group() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("product", DataType::Utf8, true),
            Field::new("revenue", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![
                    Some("A"),
                    Some("A"),
                    Some("B"),
                    None,
                    Some("C"),
                ])),
                Arc::new(Float64Array::from(vec![50.0, 55.0, 80.0, 999.0, 10.0])),
            ],
        )
        .unwrap();
        let spec = TopKSpec {
            n: 10,
            field: "revenue".into(),
            op: "sum".into(),
            sort: "descending".into(),
            groupby: Some(vec!["product".into()]),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(
            out.num_rows(), 4,
            "n=10 keeps all 3 real groups (4 rows: A x2, B x1, C x1); the \
             null-product row is dropped regardless of n"
        );
    }

    /// S3-2 sibling coverage: a multi-column `groupby` on the grouped path
    /// ranks by the composite (region, product) key, not either column
    /// alone. Region/product sums: (east,A)=50+55=105, (west,B)=80,
    /// (east,C)=10 -> top 2 by sum = {(east,A), (west,B)}.
    #[test]
    fn top_k_groupby_multi_column_ranks_by_composite_key() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("product", DataType::Utf8, false),
            Field::new("revenue", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["east", "east", "west", "east"])),
                Arc::new(StringArray::from(vec!["A", "A", "B", "C"])),
                Arc::new(Float64Array::from(vec![50.0, 55.0, 80.0, 10.0])),
            ],
        )
        .unwrap();
        let spec = TopKSpec {
            n: 2,
            field: "revenue".into(),
            op: "sum".into(),
            sort: "descending".into(),
            groupby: Some(vec!["region".into(), "product".into()]),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // (east,A) keeps 2 rows, (west,B) keeps 1 row; (east,C) dropped.
        assert_eq!(out.num_rows(), 3, "top-2 composite groups keep 3 of 4 rows");

        let products = out
            .column_by_name("product")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let kept: std::collections::BTreeSet<&str> =
            (0..out.num_rows()).map(|i| products.value(i)).collect();
        assert!(!kept.contains("C"), "(east,C), the lowest composite sum, must be dropped");
    }

    /// S3-2 sibling coverage: `sort="ascending"` on the grouped path keeps
    /// the LOWEST-ranked groups by `op`, mirroring the ungrouped path's
    /// ascending branch. Per-product sums: A=105, B=80, C=170, D=82, E=85;
    /// ascending top-2 = {B(80), D(82)}, the two smallest.
    #[test]
    fn top_k_groupby_ascending_sort_keeps_lowest_groups() {
        let batch = make_batch_product_revenue();
        let spec = TopKSpec {
            n: 2,
            field: "revenue".into(),
            op: "sum".into(),
            sort: "ascending".into(),
            groupby: Some(vec!["product".into()]),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        let products = out
            .column_by_name("product")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let kept: std::collections::BTreeSet<&str> =
            (0..out.num_rows()).map(|i| products.value(i)).collect();
        let expected: std::collections::BTreeSet<&str> = ["B", "D"].into_iter().collect();
        assert_eq!(
            kept, expected,
            "ascending top-2 by per-product sum must be the two LOWEST groups {{B, D}}, got {kept:?}"
        );
        assert_eq!(out.num_rows(), 2, "B and D are each single-row groups");
    }

    /// Sibling precedent: `regression_data.rs`'s
    /// `regression_groupby_none_and_empty_vec_are_both_ungrouped`.
    /// `groupby: Some(vec![])` must dispatch to the ungrouped path exactly
    /// like `groupby: None` — `apply`'s `Some(cols) if !cols.is_empty()`
    /// guard is what makes this true; this test pins that guard against a
    /// future edit that drops the emptiness check and routes an empty vec
    /// into `apply_grouped` (which would panic-free but wrongly report zero
    /// groupby columns as a group-by-nothing single group, not the
    /// legacy field-doubles-as-key behavior).
    #[test]
    fn top_k_groupby_some_empty_vec_dispatches_to_ungrouped_path() {
        let batch = make_batch_string_float(
            vec!["a", "a", "b", "b", "c"],
            vec![10.0, 20.0, 1.0, 2.0, 100.0],
        );
        let spec_none = TopKSpec {
            n: 2,
            field: "val".into(),
            op: "sum".into(),
            sort: "descending".into(),
            groupby: None,
            name: None,
        };
        let spec_empty = TopKSpec {
            groupby: Some(vec![]),
            ..spec_none.clone()
        };
        let out_none = apply(&spec_none, &batch).unwrap();
        let out_empty = apply(&spec_empty, &batch).unwrap();
        assert_eq!(
            out_none.num_rows(), out_empty.num_rows(),
            "groupby: Some(vec![]) must behave identically to groupby: None"
        );
        assert_eq!(out_empty.num_rows(), 2, "ungrouped path: top-2 unique-val groups");
    }

    /// A missing groupby column on the grouped path produces the
    /// `data_top_k` ctx-prefixed error from `group_partition_multi`, naming
    /// the offending column.
    #[test]
    fn top_k_groupby_missing_column_errors_with_data_top_k_ctx() {
        pyo3::Python::initialize();
        let batch = make_batch_product_revenue();
        let spec = TopKSpec {
            n: 2,
            field: "revenue".into(),
            op: "sum".into(),
            sort: "descending".into(),
            groupby: Some(vec!["nonexistent".into()]),
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("data_top_k"), "err: {msg}");
        assert!(msg.contains("nonexistent"), "err must name the missing column: {msg}");
    }
}
