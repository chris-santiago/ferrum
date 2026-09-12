//! Data transform: DataAggregate — group-by aggregation (data transform variant).
//!
//! Same semantics as stat_aggregate but exposed as a data transform.
//! Delegates to the existing aggregate module's logic.

use arrow::array::{ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::aggregate::AggColumn;
use crate::transform::context::TransformContext;
use crate::transform::group_key::{
    ensure_ungrouped_whole_batch_group, group_partition_multi, groupby_field_nullable,
    materialize_groupby_col, refuse_groupby_output_collision, KeyValue, NullKeyPolicy,
};
use crate::transform::join_aggregate::AggSpec;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct DataAggregateSpec {
    pub aggregates: Vec<AggSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &DataAggregateSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Context-aware entry point. `ctx` is used only as the GH #162
/// degrade-warning sink, for the one degrade this transform can report: an
/// all-null `groupby` key column (GH #197), raised inside the shared
/// partitioner.
pub(crate) fn apply_with_context(
    spec: &DataAggregateSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let groupby = spec.groupby.clone().unwrap_or_default();

    if spec.aggregates.is_empty() {
        return Err(PyValueError::new_err(
            "data_aggregate: aggregates must be non-empty",
        ));
    }

    // Validate aggregate fields.
    for agg in &spec.aggregates {
        schema.index_of(&agg.field).map_err(|_| {
            PyValueError::new_err(format!(
                "data_aggregate: column '{}' not found",
                agg.field
            ))
        })?;
    }

    // One partitioning idiom for every grouped transform (GH #179): the shared
    // partitioner owns column validation, the repeated-name refusal,
    // first-appearance group order, and the null-key policy. Before this
    // migration a repeated name (reachable from Python via
    // `fm.transform_aggregate(groupby=["g", "g"])`) silently produced two
    // identically-named output fields.
    let output_fields: Vec<String> = spec.aggregates.iter().map(|a| a.as_.clone()).collect();
    refuse_groupby_output_collision("data_aggregate", &groupby, &output_fields)?;
    let mut partition =
        group_partition_multi(batch, &groupby, "data_aggregate", NullKeyPolicy::Skip, ctx)?;
    ensure_ungrouped_whole_batch_group(&mut partition, &groupby);
    let group_dtypes = &partition.dtypes;

    let n_out_rows = partition.groups.len();

    // Compute aggregates per group.
    let mut group_key_vecs: Vec<Vec<KeyValue>> = Vec::with_capacity(n_out_rows);
    let mut agg_results: Vec<Vec<f64>> = vec![Vec::with_capacity(n_out_rows); spec.aggregates.len()];

    // Resolve each aggregate's source column once, per the shared dtype policy.
    let agg_columns: Vec<AggColumn> = spec
        .aggregates
        .iter()
        .map(|agg| {
            let col_idx = schema.index_of(&agg.field).map_err(|_| {
                PyValueError::new_err(format!(
                    "data_aggregate: column '{}' not found",
                    agg.field
                ))
            })?;
            AggColumn::resolve(batch.column(col_idx), &agg.field, agg.fn_, "data_aggregate")
        })
        .collect::<PyResult<_>>()?;

    for (key, rows) in &partition.groups {
        group_key_vecs.push(key.clone());
        for ai in 0..spec.aggregates.len() {
            agg_results[ai].push(agg_columns[ai].aggregate_rows(rows));
        }
    }

    // Build output schema: groupby columns (original dtype, nullable for FA-9
    // null keys) + aggregate columns (Float64).
    let mut fields: Vec<Field> = Vec::new();
    for (gi, g) in groupby.iter().enumerate() {
        fields.push(Field::new(g, group_dtypes[gi].clone(), groupby_field_nullable()));
    }
    for agg in &spec.aggregates {
        fields.push(Field::new(&agg.as_, DataType::Float64, true));
    }
    let out_schema = Arc::new(Schema::new(fields));

    // Build columns. The groupby output array dtype must match the declared
    // schema field; materialize_groupby_col guarantees this (the previous code
    // declared the original dtype but emitted a StringArray).
    let mut columns: Vec<ArrayRef> = Vec::new();
    for (gi, dt) in group_dtypes.iter().enumerate() {
        let col = materialize_groupby_col(&group_key_vecs, gi, dt)
            .map_err(PyValueError::new_err)?;
        columns.push(col);
    }

    for agg_vals in agg_results {
        columns.push(Arc::new(Float64Array::from(agg_vals)));
    }

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_aggregate: {e}")))
}

// No PyO3 wrapper: `DataAggregate` is constructed only via the dict-emitting
// `transform_aggregate` Python function and carried through the
// `transforms_json` serde path (SEAM-02). `DataAggregateSpec` above is the
// serde target.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::aggregate::AggFn;
    use arrow::array::{Array, Float64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn data_aggregate_grouped_mean() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("group", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 3.0, 10.0, 20.0])),
            ],
        )
        .unwrap();

        let spec = DataAggregateSpec {
            aggregates: vec![AggSpec {
                field: "val".into(),
                fn_: AggFn::Mean,
                as_: "mean_val".into(),
            }],
            groupby: Some(vec!["group".into()]),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        assert_eq!(out.num_columns(), 2); // group + mean_val

        let groups = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let means = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

        let a_idx = (0..groups.len()).find(|&i| groups.value(i) == "a").unwrap();
        let b_idx = (0..groups.len()).find(|&i| groups.value(i) == "b").unwrap();
        assert!((means.value(a_idx) - 2.0).abs() < 1e-12);
        assert!((means.value(b_idx) - 15.0).abs() < 1e-12);
    }

    #[test]
    fn data_aggregate_no_groupby() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]))],
        )
        .unwrap();

        let spec = DataAggregateSpec {
            aggregates: vec![AggSpec {
                field: "val".into(),
                fn_: AggFn::Sum,
                as_: "total".into(),
            }],
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let total = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert!((total.value(0) - 15.0).abs() < 1e-12);
    }

    // ── GH #179/#197: the shared partitioner's rules reach this wire transform ──
    //
    // `fm.transform_aggregate(*aggregates, groupby=[...])` emits this spec as a
    // dict, so a repeated name was publicly reachable and silently corrupted
    // the output schema before the migration.

    fn gv_batch(groups: Vec<&str>, vals: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(groups)),
                Arc::new(Float64Array::from(vals)),
            ],
        )
        .unwrap()
    }

    fn mean_spec(groupby: Option<Vec<String>>, as_: &str) -> DataAggregateSpec {
        DataAggregateSpec {
            aggregates: vec![AggSpec {
                field: "val".into(),
                fn_: AggFn::Mean,
                as_: as_.into(),
            }],
            groupby,
            name: None,
        }
    }

    #[test]
    fn data_aggregate_repeated_groupby_name_refuses() {
        pyo3::Python::initialize();
        let batch = gv_batch(vec!["a", "b"], vec![1.0, 2.0]);
        let err = apply(&mean_spec(Some(vec!["g".into(), "g".into()]), "m"), &batch)
            .expect_err("groupby=['g', 'g'] must refuse");
        assert!(err.to_string().contains("repeated"), "{err}");
        assert!(err.to_string().contains("data_aggregate"), "{err}");
    }

    #[test]
    fn data_aggregate_groupby_colliding_with_an_output_name_refuses() {
        pyo3::Python::initialize();
        let batch = gv_batch(vec!["a", "b"], vec![1.0, 2.0]);
        let err = apply(&mean_spec(Some(vec!["g".into()]), "g"), &batch)
            .expect_err("an aggregate output named like the groupby column must refuse");
        assert!(err.to_string().contains("collides"), "{err}");
    }

    /// DISCLOSED BEHAVIOR CHANGE (#179): first-appearance group order.
    #[test]
    fn data_aggregate_group_rows_are_in_first_appearance_order() {
        pyo3::Python::initialize();
        let batch = gv_batch(vec!["b", "a"], vec![1.0, 2.0]);
        let out = apply(&mean_spec(Some(vec!["g".into()]), "m"), &batch).unwrap();
        let g = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!((g.value(0), g.value(1)), ("b", "a"), "first-appearance, not sorted");

        // The flip is a PERMUTATION: each key keeps its own mean.
        let m = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((m.value(0), m.value(1)), (1.0, 2.0), "b's mean stays with b");
    }

    /// An ungrouped aggregate over a ZERO-ROW batch still collapses to one
    /// summary row — the boundary the shared partitioner cannot discover from
    /// rows, so this transform asks for it explicitly.
    #[test]
    fn data_aggregate_ungrouped_zero_row_input_still_yields_one_row() {
        pyo3::Python::initialize();
        let batch = gv_batch(Vec::new(), Vec::new());
        let out = apply(&mean_spec(None, "m"), &batch).unwrap();
        assert_eq!(out.num_rows(), 1, "the whole-batch group survives an empty batch");
    }

}
