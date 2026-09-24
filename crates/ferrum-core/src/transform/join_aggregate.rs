//! Data transform: JoinAggregate — add aggregate columns without collapsing rows.
//!
//! Computes grouped aggregates and joins them back to the original rows,
//! so the output has the same number of rows as the input plus new columns.

use arrow::array::{ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::aggregate::{AggColumn, AggFn};
use crate::transform::context::TransformContext;
use crate::transform::group_key::{
    group_partition_multi, refuse_groupby_output_collision, NullKeyPolicy,
};

/// A single aggregation specification for JoinAggregate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AggSpec {
    pub field: String,
    #[serde(rename = "fn")]
    pub fn_: AggFn,
    #[serde(rename = "as")]
    pub as_: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct JoinAggregateSpec {
    pub aggregates: Vec<AggSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &JoinAggregateSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Context-aware entry point. `ctx` is used only as the GH #162
/// degrade-warning sink, for the one degrade this transform can report: an
/// all-null `groupby` key column (GH #197), raised inside the shared
/// partitioner.
pub(crate) fn apply_with_context(
    spec: &JoinAggregateSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();
    let groupby = spec.groupby.clone().unwrap_or_default();

    if spec.aggregates.is_empty() {
        return Err(PyValueError::new_err(
            "data_join_aggregate: aggregates must be non-empty",
        ));
    }

    // One partitioning idiom for every grouped transform (GH #179): the shared
    // partitioner owns column validation, the repeated-name refusal,
    // first-appearance group order, and the null-key policy. `Pool` is #197's
    // stated exemption for this transform: it joins one aggregate value back
    // onto every INPUT row, so a null-keyed row must land in some group —
    // dropping it would leave it holding NaN where a real number belongs.
    let output_fields: Vec<String> = spec.aggregates.iter().map(|a| a.as_.clone()).collect();
    refuse_groupby_output_collision("data_join_aggregate", &groupby, &output_fields)?;
    let partition = group_partition_multi(
        batch,
        &groupby,
        "data_join_aggregate",
        NullKeyPolicy::Pool,
        ctx,
    )?;

    // For each aggregate, compute per-group value, then broadcast back.
    let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(spec.aggregates.len());

    for agg in &spec.aggregates {
        let col_idx = schema.index_of(&agg.field).map_err(|_| {
            PyValueError::new_err(format!(
                "data_join_aggregate: column '{}' not found",
                agg.field
            ))
        })?;

        // Resolve the source column once, then compute per-group aggregates.
        let agg_column = AggColumn::resolve(
            batch.column(col_idx),
            &agg.field,
            agg.fn_,
            "data_join_aggregate",
        )?;
        let mut row_values: Vec<f64> = vec![f64::NAN; n_rows];

        for (_, rows) in &partition.groups {
            let agg_val = agg_column.aggregate_rows(rows);
            for &r in rows {
                row_values[r] = agg_val;
            }
        }

        new_cols.push(Arc::new(Float64Array::from(row_values)));
    }

    // Build output schema: original columns + new aggregate columns.
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    for agg in &spec.aggregates {
        fields.push(Field::new(&agg.as_, DataType::Float64, true));
    }
    let out_schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns.extend(new_cols);

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_join_aggregate: {e}")))
}


// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "JoinAggregate")]
#[derive(Debug, Clone)]
pub(crate) struct PyJoinAggregate(pub(crate) TransformSpec);

#[pymethods]
impl PyJoinAggregate {
    #[new]
    #[pyo3(signature = (*, name = None))]
    fn new(name: Option<String>) -> Self {
        PyJoinAggregate(TransformSpec::JoinAggregate(JoinAggregateSpec {
            aggregates: Vec::new(),
            groupby: None,
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::JoinAggregate(s) => format!("JoinAggregate(name={:?})", s.name),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn join_aggregate_preserves_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("group", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            ],
        )
        .unwrap();

        let spec = JoinAggregateSpec {
            aggregates: vec![AggSpec {
                field: "val".into(),
                fn_: AggFn::Mean,
                as_: "group_mean".into(),
            }],
            groupby: Some(vec!["group".into()]),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // Same row count as input.
        assert_eq!(out.num_rows(), 5);
        // Original columns + new aggregate column.
        assert_eq!(out.num_columns(), 3);

        let means = out
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        // Group "a": mean(1,2) = 1.5
        assert!((means.value(0) - 1.5).abs() < 1e-12);
        assert!((means.value(1) - 1.5).abs() < 1e-12);
        // Group "b": mean(3,4,5) = 4.0
        assert!((means.value(2) - 4.0).abs() < 1e-12);
        assert!((means.value(3) - 4.0).abs() < 1e-12);
        assert!((means.value(4) - 4.0).abs() < 1e-12);
    }

    // ── GH #179/#197: shared rules, row-preserving exemption ────────────────

    fn gv_batch(groups: Vec<Option<&str>>, vals: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
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

    fn mean_spec(groupby: Vec<&str>, as_: &str) -> JoinAggregateSpec {
        JoinAggregateSpec {
            aggregates: vec![AggSpec {
                field: "val".into(),
                fn_: AggFn::Mean,
                as_: as_.into(),
            }],
            groupby: Some(groupby.into_iter().map(String::from).collect()),
            name: None,
        }
    }

    #[test]
    fn join_aggregate_repeated_groupby_name_refuses() {
        pyo3::Python::initialize();
        let batch = gv_batch(vec![Some("a"), Some("b")], vec![1.0, 2.0]);
        let err = apply(&mean_spec(vec!["g", "g"], "m"), &batch)
            .expect_err("groupby=['g', 'g'] must refuse");
        assert!(err.to_string().contains("repeated"), "{err}");
        assert!(err.to_string().contains("data_join_aggregate"), "{err}");
    }

    #[test]
    fn join_aggregate_output_name_colliding_with_the_groupby_column_refuses() {
        pyo3::Python::initialize();
        let batch = gv_batch(vec![Some("a")], vec![1.0]);
        let err = apply(&mean_spec(vec!["g"], "g"), &batch)
            .expect_err("an aggregate output named like the groupby column must refuse");
        assert!(err.to_string().contains("collides"), "{err}");
    }

    /// GH #197 EXEMPTION: this transform joins an aggregate back onto every
    /// INPUT row, so null-keyed rows are pooled into one group rather than
    /// dropped — dropping them would leave real rows holding NaN.
    #[test]
    fn join_aggregate_pools_null_keyed_rows_instead_of_dropping_them() {
        pyo3::Python::initialize();
        let batch = gv_batch(
            vec![None, Some("a"), None],
            vec![1.0, 5.0, 3.0],
        );
        let out = apply(&mean_spec(vec!["g"], "m"), &batch).unwrap();
        let m = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(m.value(0), 2.0, "the pooled null group's mean is (1 + 3) / 2");
        assert_eq!(m.value(2), 2.0, "both null-keyed rows read the same pooled mean");
        assert_eq!(m.value(1), 5.0, "the real group is independent");
    }

    /// GH #197: an all-null key column still joins a value onto every row (the
    /// exemption), but says once that the grouping could not be honored.
    #[test]
    fn join_aggregate_all_null_group_keys_pool_and_warn() {
        pyo3::Python::initialize();
        let batch = gv_batch(vec![None, None], vec![1.0, 3.0]);
        let ctx = TransformContext::default();
        let out = apply_with_context(&mean_spec(vec!["g"], "m"), &batch, &ctx).unwrap();
        let m = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(m.value(0), 2.0);
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].to_string().contains("all rows were treated as one unlabeled group"),
            "{:?}",
            warnings[0]
        );
    }

}
