//! Data transform: DataStack — compute stacked (cumulative) positions.
//!
//! Computes y0 and y1 columns for stacked bar/area charts. Groups by
//! `groupby`, sorts within each group, and accumulates field values.
//! Supports offset modes: "zero" (default), "normalize", "center".

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::context::TransformContext;
use crate::transform::group_key::{
    group_partition_multi, refuse_groupby_output_collision, NullKeyPolicy,
};
use crate::transform::numeric_util::coerce_to_float64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct DataStackSpec {
    /// Field to stack.
    pub field: String,
    /// Columns to group by (define each stack).
    #[serde(default)]
    pub groupby: Vec<String>,
    /// Sort order within each stack.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sort: Option<Vec<String>>,
    /// Output column names: (y0, y1).
    #[serde(default = "default_stack_as")]
    pub as_: (String, String),
    /// Offset mode: "zero", "normalize", "center".
    #[serde(default = "default_offset")]
    pub offset: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_stack_as() -> (String, String) {
    ("y0".into(), "y1".into())
}
/// The single source of truth for `DataStack`'s free-string `offset`
/// vocabulary. Read by [`apply`]'s enforcement match, its error message, and
/// `registry::entry_for_kind("data_stack")`'s `methods` list — R1 design
/// finding 1, mirroring the `smooth::METHODS` precedent.
pub(crate) const OFFSETS: &[&str] = &["zero", "normalize", "center"];

fn default_offset() -> String {
    "zero".into()
}

pub(crate) fn apply(spec: &DataStackSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Context-aware entry point. `ctx` is used only as the GH #162
/// degrade-warning sink, for the one degrade this transform can report: an
/// all-null `groupby` key column (GH #197), raised inside the shared
/// partitioner.
pub(crate) fn apply_with_context(
    spec: &DataStackSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    let field_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_stack: column '{}' not found", spec.field))
    })?;
    let field_col = coerce_to_float64(batch.column(field_idx), "data_stack", &spec.field)?;

    // One partitioning idiom for every grouped transform (GH #179): the shared
    // partitioner owns column validation, the repeated-name refusal,
    // first-appearance group order, and the null-key policy. `Pool` is #197's
    // stated exemption for this transform: it emits one stacked position per
    // INPUT row, so a null-keyed row must land in some stack — dropping it
    // would leave it holding a default 0/0 position that renders as a
    // zero-height bar indistinguishable from real data.
    refuse_groupby_output_collision(
        "data_stack",
        &spec.groupby,
        &[spec.as_.0.clone(), spec.as_.1.clone()],
    )?;
    let partition =
        group_partition_multi(batch, &spec.groupby, "data_stack", NullKeyPolicy::Pool, ctx)?;

    // Compute y0, y1 for each group. The output is written per input row, so
    // the group ORDER never reaches the output — only group membership does.
    let mut y0 = vec![0.0f64; n_rows];
    let mut y1 = vec![0.0f64; n_rows];

    for (_, rows) in &partition.groups {
        // Accumulate.
        let mut cumsum = 0.0;
        for &row in rows {
            let val = if field_col.is_null(row) {
                0.0
            } else {
                let v = field_col.value(row);
                if v.is_nan() { 0.0 } else { v }
            };
            y0[row] = cumsum;
            y1[row] = cumsum + val;
            cumsum += val;
        }

        // Apply offset.
        match spec.offset.as_str() {
            "zero" => {} // Already done.
            "normalize" => {
                let total = cumsum;
                if total.abs() > 1e-15 {
                    for &row in rows {
                        y0[row] /= total;
                        y1[row] /= total;
                    }
                }
            }
            "center" => {
                let half = cumsum / 2.0;
                for &row in rows {
                    y0[row] -= half;
                    y1[row] -= half;
                }
            }
            // Unknown offset: all Python entry points (transform_stack, PyDataStack::new)
            // validate offset before the spec reaches here, so this arm is dead in
            // normal operation. Raise an explicit error so any direct Rust construction
            // with a bad offset string fails loudly rather than silently using zero.
            _ => {
                return Err(PyValueError::new_err(format!(
                    "data_stack: unknown offset {:?}; expected one of {}",
                    spec.offset,
                    OFFSETS.iter().map(|o| format!("{o:?}")).collect::<Vec<_>>().join(", "),
                )));
            }
        }
    }

    // Build output: original columns + y0, y1.
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new(&spec.as_.0, DataType::Float64, false));
    fields.push(Field::new(&spec.as_.1, DataType::Float64, false));
    let out_schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns.push(Arc::new(Float64Array::from(y0)));
    columns.push(Arc::new(Float64Array::from(y1)));

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_stack: {e}")))
}

// No PyO3 wrapper: `DataStack` is constructed only via the dict-emitting
// `transform_stack` Python function and carried through the `transforms_json`
// serde path (SEAM-02). The removed `#[new]` validated `offset`; the dict path
// validates it earlier in `transform_stack` (`_validate_stack_offset`), so the
// parity is preserved. `DataStackSpec` above is the serde target.

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    // ---- R1-relocated coverage (tests/bug_hunt_release_transforms.rs, 2026-08-27) ----

    /// FA-7 contract: an Int64 groupby column defines one stack per distinct
    /// value (previously int columns fell through to an "unsupported dtype"
    /// error, which would have merged every row into a single stack). Group 1
    /// (values [1.0, 2.0]) and group 2 (values [3.0, 4.0]) must accumulate
    /// independently — group 2's y0 must reset to 0.0 rather than continuing
    /// group 1's running total.
    #[test]
    fn int_groupby_column_separates_stacks() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            ],
        )
        .unwrap();

        let spec = DataStackSpec {
            field: "val".into(),
            groupby: vec!["g".into()],
            sort: None,
            as_: ("y0".into(), "y1".into()),
            offset: "zero".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let y0_col = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let y1_col = out.column(3).as_any().downcast_ref::<Float64Array>().unwrap();

        // Group 1: cumsum [1.0, 2.0] → y0=[0,1], y1=[1,3].
        assert_eq!(y0_col.value(0), 0.0);
        assert_eq!(y1_col.value(0), 1.0);
        assert_eq!(y0_col.value(1), 1.0);
        assert_eq!(y1_col.value(1), 3.0);
        // Group 2 must restart its own cumsum, NOT continue group 1's total of 3.0.
        assert_eq!(y0_col.value(2), 0.0, "group 2 must reset y0, not merge with group 1");
        assert_eq!(y1_col.value(2), 3.0);
        assert_eq!(y0_col.value(3), 3.0);
        assert_eq!(y1_col.value(3), 7.0);
    }

    #[test]
    fn stack_zero_offset() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "a"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap();

        let spec = DataStackSpec {
            field: "val".into(),
            groupby: vec!["cat".into()],
            sort: None,
            as_: ("y0".into(), "y1".into()),
            offset: "zero".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let y0_col = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let y1_col = out.column(3).as_any().downcast_ref::<Float64Array>().unwrap();

        assert_eq!(y0_col.value(0), 0.0);
        assert_eq!(y1_col.value(0), 1.0);
        assert_eq!(y0_col.value(1), 1.0);
        assert_eq!(y1_col.value(1), 3.0);
        assert_eq!(y0_col.value(2), 3.0);
        assert_eq!(y1_col.value(2), 6.0);
    }

    #[test]
    fn stack_normalize_offset() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![2.0, 3.0, 5.0]))],
        )
        .unwrap();

        let spec = DataStackSpec {
            field: "val".into(),
            groupby: vec![],
            sort: None,
            as_: ("y0".into(), "y1".into()),
            offset: "normalize".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let y1_col = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        // Last y1 should be 1.0 (normalized total).
        assert!((y1_col.value(2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stack_center_offset() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![2.0, 2.0, 2.0]))],
        )
        .unwrap();

        let spec = DataStackSpec {
            field: "val".into(),
            groupby: vec![],
            sort: None,
            as_: ("y0".into(), "y1".into()),
            offset: "center".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let y0_col = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let y1_col = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        // Total = 6, half = 3. First y0 = 0-3 = -3, first y1 = 2-3 = -1.
        assert!((y0_col.value(0) - (-3.0)).abs() < 1e-12);
        assert!((y1_col.value(0) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn unknown_offset_returns_error() {
        // An unknown offset string must produce an explicit Err, not silently
        // fall through to zero-offset behavior.
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
        )
        .unwrap();

        let spec = DataStackSpec {
            field: "val".into(),
            groupby: vec![],
            sort: None,
            as_: ("y0".into(), "y1".into()),
            offset: "streamgraph".into(), // not a valid offset
            name: None,
        };
        let result = apply(&spec, &batch);
        assert!(result.is_err(), "unknown offset must return Err, not silently use zero");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("streamgraph") || err_msg.contains("offset"),
            "error message must reference the bad offset: {err_msg}"
        );
    }

    // ── GH #179/#197: shared rules, row-preserving exemption ────────────────

    fn gv_batch(groups: Vec<Option<&str>>, vals: Vec<f64>) -> RecordBatch {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("v", DataType::Float64, false),
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

    fn stack_spec(groupby: Vec<&str>) -> DataStackSpec {
        DataStackSpec {
            field: "v".into(),
            groupby: groupby.into_iter().map(String::from).collect(),
            sort: None,
            as_: default_stack_as(),
            offset: "zero".into(),
            name: None,
        }
    }

    #[test]
    fn data_stack_repeated_groupby_name_refuses() {
        pyo3::Python::initialize();
        let batch = gv_batch(vec![Some("a"), Some("b")], vec![1.0, 2.0]);
        let err = apply(&stack_spec(vec!["g", "g"]), &batch)
            .expect_err("groupby=['g', 'g'] must refuse");
        assert!(err.to_string().contains("repeated"), "{err}");
        assert!(err.to_string().contains("data_stack"), "{err}");
    }

    /// An output name colliding with the groupby column refuses — the output
    /// schema is the input's plus `as_`, so a collision means two same-named
    /// fields here too.
    #[test]
    fn data_stack_output_name_colliding_with_the_groupby_column_refuses() {
        pyo3::Python::initialize();
        let batch = gv_batch(vec![Some("a")], vec![1.0]);
        let mut spec = stack_spec(vec!["g"]);
        spec.as_ = ("g".into(), "y1".into());
        let err = apply(&spec, &batch).expect_err("as_=('g', ...) must refuse");
        assert!(err.to_string().contains("collides"), "{err}");
    }

    /// GH #197 EXEMPTION: this transform emits one stacked position per INPUT
    /// row, so null-keyed rows are pooled into one stack rather than dropped —
    /// dropping them would leave a real bar holding a default 0/0 position.
    #[test]
    fn data_stack_pools_null_keyed_rows_instead_of_dropping_them() {
        pyo3::Python::initialize();
        let batch = gv_batch(
            vec![None, Some("a"), None],
            vec![1.0, 5.0, 2.0],
        );
        let out = apply(&stack_spec(vec!["g"]), &batch).unwrap();
        let y1 = out.column(out.schema().index_of("y1").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        // The two null-keyed rows stack on each other: 0→1 then 1→3.
        assert_eq!(y1.value(0), 1.0);
        assert_eq!(y1.value(2), 3.0, "the second null-keyed row stacks on the first");
        // The real group is independent.
        assert_eq!(y1.value(1), 5.0);
    }

    /// GH #197: an all-null key column still stacks every row (the exemption),
    /// but says once that the grouping request could not be honored.
    #[test]
    fn data_stack_all_null_group_keys_pool_and_warn() {
        pyo3::Python::initialize();
        let batch = gv_batch(vec![None, None], vec![1.0, 2.0]);
        let ctx = TransformContext::default();
        let out = apply_with_context(&stack_spec(vec!["g"]), &batch, &ctx).unwrap();
        assert_eq!(out.num_rows(), 2, "every input row still gets a position");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1);
        let text = warnings[0].to_string();
        assert!(text.contains("data_stack"), "{text}");
        assert!(
            text.contains("all rows were treated as one unlabeled group"),
            "the warning states this transform's OWN disposition: {text}"
        );
    }

}
