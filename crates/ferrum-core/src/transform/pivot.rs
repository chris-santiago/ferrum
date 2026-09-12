//! Data transform: Pivot — long to wide.
//!
//! Groups by `groupby` columns, then for each unique value of `field`,
//! creates a new column containing the aggregated `value` column.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transform::context::TransformContext;
use crate::transform::group_key::{
    group_partition_multi, groupby_field_nullable, materialize_groupby_col,
    refuse_groupby_output_collision, KeyValue, NullKeyPolicy,
};
use crate::transform::numeric_util::coerce_to_float64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PivotSpec {
    /// Column whose unique values become new column headers.
    pub field: String,
    /// Column whose values fill the pivoted cells.
    pub value: String,
    /// Columns to group by. If None, each row is its own group.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    /// Maximum number of pivot columns to create.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub limit: Option<usize>,
    /// Aggregation operation when multiple values map to same cell.
    #[serde(default = "default_op")]
    pub op: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

/// The single source of truth for `Pivot`'s free-string `op` vocabulary.
/// Read by [`aggregate_values`] (the enforcement site), its error message,
/// and `registry::entry_for_kind("pivot")`'s `methods` list — R1 design
/// finding 1, mirroring the `smooth::METHODS` precedent.
pub(crate) const OPS: &[&str] = &["sum", "mean", "count", "min", "max", "first", "last"];

fn default_op() -> String {
    "sum".into()
}

pub(crate) fn apply(spec: &PivotSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Context-aware entry point. `ctx` is used only as the GH #162
/// degrade-warning sink, for the one degrade this transform can report: an
/// all-null `groupby` key column (GH #197), raised inside the shared
/// partitioner.
pub(crate) fn apply_with_context(
    spec: &PivotSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    // Validate field column (must be Utf8).
    let field_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_pivot: column '{}' not found", spec.field))
    })?;
    let field_col = batch
        .column(field_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "data_pivot: column '{}' must be Utf8",
                spec.field
            ))
        })?;

    // Validate value column (any numeric dtype, coerced to Float64).
    let value_idx = schema.index_of(&spec.value).map_err(|_| {
        PyValueError::new_err(format!("data_pivot: column '{}' not found", spec.value))
    })?;
    let value_col = coerce_to_float64(batch.column(value_idx), "data_pivot", &spec.value)?;

    // Collect unique pivot values (ordered by first appearance).
    let mut pivot_values: Vec<String> = Vec::new();
    let mut pivot_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for i in 0..n_rows {
        if field_col.is_null(i) {
            continue;
        }
        let v = field_col.value(i).to_string();
        if pivot_set.insert(v.clone()) {
            pivot_values.push(v);
        }
    }

    // Apply limit. This truncates the output COLUMNS only; which groups earn
    // an output row is decided independently below, from whether they hold any
    // row with a non-null `field`.
    if let Some(limit) = spec.limit {
        pivot_values.truncate(limit);
    }
    let kept_pivot_values: std::collections::HashSet<&str> =
        pivot_values.iter().map(|s| s.as_str()).collect();

    // One partitioning idiom for every grouped transform (GH #179): the shared
    // partitioner owns column validation, the repeated-name refusal,
    // first-appearance group order, and the null-key policy. Before this
    // migration a repeated name (reachable from Python via
    // `fm.transform_pivot(groupby=["g", "g"])`) silently produced two
    // identically-named output fields.
    let groupby = spec.groupby.clone().unwrap_or_default();
    refuse_groupby_output_collision("data_pivot", &groupby, &pivot_values)?;
    let partition =
        group_partition_multi(batch, &groupby, "data_pivot", NullKeyPolicy::Skip, ctx)?;
    let group_dtypes = &partition.dtypes;

    // Per group, bucket each surviving row's value under its pivot column. A
    // group contributing no cell at all (every one of its rows has a null or
    // truncated-away `field`) emits no output row, as it did before this
    // migration.
    let n_groups = partition.groups.len();
    let mut group_key_vecs: Vec<Vec<KeyValue>> = Vec::with_capacity(n_groups);
    let mut pivot_data: Vec<Vec<f64>> = vec![Vec::with_capacity(n_groups); pivot_values.len()];

    for (key, rows) in &partition.groups {
        // A group earns an output row by having at least one row with a
        // non-null `field`, INDEPENDENTLY of whether that row's pivot value
        // survived `limit` — so a group whose every value was truncated away
        // still emits its row, with NaN in every kept column. That is what
        // this transform did before the migration and it is not one of the
        // disclosed behavior changes, so it is preserved deliberately.
        let mut has_field_row = false;
        let mut cells: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for &row in rows {
            if field_col.is_null(row) {
                continue;
            }
            has_field_row = true;
            let Some(&pv) = kept_pivot_values.get(field_col.value(row)) else {
                continue; // Pruned by `limit`; the group still counts above.
            };
            let val = if value_col.is_null(row) {
                f64::NAN
            } else {
                value_col.value(row)
            };
            cells.entry(pv).or_default().push(val);
        }
        if !has_field_row {
            continue;
        }
        group_key_vecs.push(key.clone());
        for (pi, pv) in pivot_values.iter().enumerate() {
            let agg_val = match cells.get(pv.as_str()) {
                Some(vals) => aggregate_values(vals, &spec.op)?,
                None => f64::NAN,
            };
            pivot_data[pi].push(agg_val);
        }
    }

    // Build output: groupby columns (original dtype, nullable per FA-9) + one
    // Float64 column per pivot value.
    let mut out_fields: Vec<Field> = groupby
        .iter()
        .enumerate()
        .map(|(i, name)| Field::new(name, group_dtypes[i].clone(), groupby_field_nullable()))
        .collect();
    for pv in &pivot_values {
        out_fields.push(Field::new(pv, DataType::Float64, true));
    }
    let out_schema = Arc::new(Schema::new(out_fields));

    // Build Arrow arrays. The groupby output dtype must match the declared schema
    // field; materialize_groupby_col guarantees this (the previous code declared
    // the original dtype but emitted a StringArray, crashing RecordBatch::try_new
    // for non-Utf8 groupby columns).
    let mut out_cols: Vec<ArrayRef> = Vec::with_capacity(groupby.len() + pivot_values.len());
    for (gi, dt) in group_dtypes.iter().enumerate() {
        let col = materialize_groupby_col(&group_key_vecs, gi, dt)
            .map_err(PyValueError::new_err)?;
        out_cols.push(col);
    }
    for pv_data in pivot_data {
        out_cols.push(Arc::new(Float64Array::from(pv_data)));
    }

    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("data_pivot: {e}")))
}

/// Aggregate one cell's values under `op`.
///
/// R1 design finding 1: the unknown-`op` arm used to fall back to `sum`
/// silently, which made this the one declared vocabulary in the registry with
/// no refusal site to guard it — and, worse, meant a typo'd `op` rendered a
/// plausible-looking wrong number with no diagnostic. It now refuses, which
/// is both the honest answer and what lets
/// `pivot_methods_accepted_and_bogus_op_rejected` (`registry/tests.rs`) hold
/// the registry entry to [`OPS`].
fn aggregate_values(vals: &[f64], op: &str) -> PyResult<f64> {
    let clean: Vec<f64> = vals.iter().copied().filter(|v| !v.is_nan()).collect();
    if clean.is_empty() {
        return Ok(f64::NAN);
    }
    Ok(match op {
        "sum" => clean.iter().sum(),
        "mean" => clean.iter().sum::<f64>() / clean.len() as f64,
        "count" => clean.len() as f64,
        "min" => clean.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        "max" => clean.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        "first" => clean[0],
        "last" => *clean.last().unwrap(),
        other => {
            return Err(PyValueError::new_err(format!(
                "data_pivot: unknown op '{other}'; expected {}",
                OPS.join("|"),
            )))
        }
    })
}

// No PyO3 wrapper: `DataPivot` is constructed only via the dict-emitting
// `transform_pivot` Python function and carried through the `transforms_json`
// serde path (SEAM-02). The removed `#[new]` performed no validation beyond
// serde's required fields. `PivotSpec` above is the serde target.

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn pivot_basic() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("group", DataType::Utf8, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(StringArray::from(vec!["x", "y", "x", "y"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            ],
        )
        .unwrap();

        let spec = PivotSpec {
            field: "key".into(),
            value: "val".into(),
            groupby: Some(vec!["group".into()]),
            limit: None,
            op: "sum".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        // Columns: group, x, y
        assert_eq!(out.num_columns(), 3);
    }

    // ── GH #179: the wire-only three reach the partitioner's rules ──────────
    //
    // `fm.transform_pivot(field, value, groupby=[...])` is a typed public
    // function that emits this spec as a dict, so every rule below was
    // publicly reachable and silently ignored before the migration.

    /// Three columns: a `g` group key, a `key` column whose values become the
    /// pivot columns, and a numeric `val`.
    fn pivot_batch(groups: Vec<&str>, keys: Vec<&str>, vals: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(groups)),
                Arc::new(StringArray::from(keys)),
                Arc::new(Float64Array::from(vals)),
            ],
        )
        .unwrap()
    }

    fn pivot_spec(groupby: Option<Vec<String>>) -> PivotSpec {
        PivotSpec {
            field: "key".into(),
            value: "val".into(),
            groupby,
            limit: None,
            op: "sum".into(),
            name: None,
        }
    }

    /// A repeated groupby name refuses. Before the migration it produced an
    /// output schema with TWO fields both named `g`, and `Schema::index_of`
    /// resolves such a name to the first match — so every downstream reader
    /// silently read one column while believing it read the other.
    #[test]
    fn pivot_repeated_groupby_name_refuses_instead_of_corrupting_the_schema() {
        pyo3::Python::initialize();
        let batch = pivot_batch(vec!["a", "b"], vec!["x", "x"], vec![1.0, 2.0]);
        let err = apply(&pivot_spec(Some(vec!["g".into(), "g".into()])), &batch)
            .expect_err("groupby=['g', 'g'] must refuse");
        let msg = err.to_string();
        assert!(msg.contains("repeated"), "{msg}");
        assert!(msg.contains("data_pivot"), "error stays in this transform's voice: {msg}");
    }

    /// A groupby name that collides with one of the pivot columns refuses for
    /// the same reason: the output would carry two fields of that name.
    #[test]
    fn pivot_groupby_colliding_with_a_pivot_column_refuses() {
        pyo3::Python::initialize();
        // The `key` values become the output column names, so a pivot column
        // literally named `g` is produced here.
        let batch = pivot_batch(vec!["a", "b"], vec!["g", "g"], vec![1.0, 2.0]);
        let err = apply(&pivot_spec(Some(vec!["g".into()])), &batch)
            .expect_err("a groupby column named like a pivot column must refuse");
        let msg = err.to_string();
        assert!(msg.contains("collides"), "{msg}");
        assert!(msg.contains("data_pivot"), "{msg}");
    }

    /// DISCLOSED BEHAVIOR CHANGE (#179): group rows come out in
    /// first-appearance order, matching every partitioner-routed transform.
    /// This module previously emitted BTreeMap-sorted order, so `b` before `a`
    /// in the data still rendered `a` first.
    #[test]
    fn pivot_group_rows_are_in_first_appearance_order() {
        pyo3::Python::initialize();
        let batch = pivot_batch(vec!["b", "a"], vec!["x", "x"], vec![1.0, 2.0]);
        let out = apply(&pivot_spec(Some(vec!["g".into()])), &batch).unwrap();
        let g = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            (g.value(0), g.value(1)),
            ("b", "a"),
            "first-appearance order, not sorted"
        );

        // The flip is a PERMUTATION: each key keeps its own pivoted cell.
        let x = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((x.value(0), x.value(1)), (1.0, 2.0), "b's cell stays with b");
    }

    /// GH #197: a groupby column that is entirely null yields no groups at all,
    /// and says so once through the degrade-warning channel.
    #[test]
    fn pivot_all_null_group_keys_yield_no_rows_and_warn() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("key", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![None, None] as Vec<Option<&str>>)),
                Arc::new(StringArray::from(vec!["x", "x"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
            ],
        )
        .unwrap();
        let ctx = TransformContext::default();
        let out = apply_with_context(&pivot_spec(Some(vec!["g".into()])), &batch, &ctx).unwrap();
        assert_eq!(out.num_rows(), 0);
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "one warning for the whole transform");
        assert!(warnings[0].to_string().contains("data_pivot"), "{:?}", warnings[0]);
    }


    /// `limit` truncates the pivot COLUMNS, not the group set: a group whose
    /// every row carries a truncated-away pivot value still emits its row,
    /// with NaN in each kept column. Pinned because the migration rewrote the
    /// bucketing loop and this is the one place that shape is observable.
    #[test]
    fn pivot_limit_truncates_columns_without_dropping_groups() {
        pyo3::Python::initialize();
        // Group "a" only ever carries key "x"; group "b" only "y". With
        // limit=1 the kept column is "x" (first appearance), so "b" has no
        // cell in it — but "b" is still a group.
        let batch = pivot_batch(
            vec!["a", "b"],
            vec!["x", "y"],
            vec![1.0, 2.0],
        );
        let mut spec = pivot_spec(Some(vec!["g".into()]));
        spec.limit = Some(1);
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 2, "groupby column + the one kept pivot column");
        assert_eq!(out.schema().field(1).name(), "x");
        assert_eq!(out.num_rows(), 2, "both groups keep their row");
        let g = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let x = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((g.value(0), g.value(1)), ("a", "b"));
        assert_eq!(x.value(0), 1.0);
        assert!(x.value(1).is_nan(), "group 'b' has no cell in the kept column");
    }

}
