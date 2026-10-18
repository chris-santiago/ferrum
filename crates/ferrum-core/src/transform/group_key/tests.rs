//! Tests for the grouping toolkit (`group_key/mod.rs`).
//!
//! Promoted out of an inline `mod tests` by the GH #179/#197 partitioner
//! migration, per the crate's Rust test-module convention: the migration added
//! the partition-policy, output-collision, and all-null-warning coverage that
//! pushed the test body past the source module it tests (899 lines against
//! 864), which is the documented trigger for a directory module. What lives
//! here is everything that exercises `mod.rs`: key extraction and
//! materialization round-trips (FA-3/FA-9), the partitioner's ordering, null,
//! repeat and dtype policies across all three `NullKeyPolicy` variants, the
//! two grouped-execution templates, and the wire/PyO3 `groupby` parsers.

use super::*;

use arrow::array::Int64Array;

/// Every test below partitions under the adjudicated default policy
/// ([`NullKeyPolicy::Skip`]) with a throwaway warning sink; the tests that
/// are ABOUT the policy or the sink call `group_partition_multi` directly.
fn partition_skip(
    batch: &RecordBatch,
    group_cols: &[String],
    label: &str,
) -> PyResult<GroupPartition> {
    group_partition_multi(
        batch,
        group_cols,
        label,
        NullKeyPolicy::Skip,
        &TransformContext::default(),
    )
}

#[test]
fn null_and_nan_float_keys_are_distinct() {
    // FA-3 contract: a null float key (KeyValue::Null) and a genuine NaN
    // value key (KeyValue::Float(NAN.to_bits())) must not collide.
    let arr = Float64Array::from(vec![None, Some(f64::NAN), Some(1.0)]);
    let k_null = groupby_key_at(&arr, &DataType::Float64, 0).unwrap();
    let k_nan = groupby_key_at(&arr, &DataType::Float64, 1).unwrap();
    let k_real = groupby_key_at(&arr, &DataType::Float64, 2).unwrap();
    assert_eq!(k_null, KeyValue::Null);
    assert!(matches!(k_nan, KeyValue::Float(_)));
    assert_ne!(k_null, k_nan, "null and NaN must be distinct keys");
    assert_ne!(k_nan, k_real);
}

#[test]
fn int64_key_extract_and_materialize_roundtrip() {
    let arr = Int64Array::from(vec![Some(10), None, Some(20)]);
    let keys: Vec<Vec<KeyValue>> = (0..3)
        .map(|r| vec![groupby_key_at(&arr, &DataType::Int64, r).unwrap()])
        .collect();
    assert_eq!(keys[0][0], KeyValue::Int(10));
    assert_eq!(keys[1][0], KeyValue::Null);
    assert_eq!(keys[2][0], KeyValue::Int(20));

    let col = materialize_groupby_col(&keys, 0, &DataType::Int64).unwrap();
    let out = col.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(out.value(0), 10);
    // FA-9: the null int key materializes as an actual Arrow null, NOT a 0
    // proxy, so a null group stays distinct from a genuine 0 key downstream.
    assert!(out.is_null(1), "null int key materializes as Arrow null");
    assert_eq!(out.value(2), 20);
}

#[test]
fn fa9_null_int_key_distinct_from_real_zero_key() {
    // FA-9 regression: {Null, Int(0), Int(5)} must produce three
    // distinguishable output values — null as an Arrow null, 0 and 5 as
    // genuine values — so a downstream category scale shows three groups.
    let keys: Vec<Vec<KeyValue>> = vec![
        vec![KeyValue::Null],
        vec![KeyValue::Int(0)],
        vec![KeyValue::Int(5)],
    ];
    let col = materialize_groupby_col(&keys, 0, &DataType::Int64).unwrap();
    let out = col.as_any().downcast_ref::<Int64Array>().unwrap();
    assert!(out.is_null(0), "null key must be a distinct Arrow null");
    assert!(!out.is_null(1));
    assert_eq!(out.value(1), 0, "real 0 key stays 0");
    assert!(!out.is_null(2));
    assert_eq!(out.value(2), 5);
}

#[test]
fn unsupported_dtype_rejected() {
    assert!(!is_groupby_supported_dtype(&DataType::Date32));
    assert!(is_groupby_supported_dtype(&DataType::Int32));
    assert!(is_groupby_supported_dtype(&DataType::Boolean));
    assert!(is_groupby_supported_dtype(&DataType::Utf8));
}

// ── group_partition_multi single-column coverage (T0.3 / T3.8a) ──────────
//
// These tests previously called the now-deleted `group_partition` helper.
// They are rewritten to call `group_partition_multi` with a 1-element slice,
// which is the canonical entry point after T3.8a. The single-key case
// exercises identical code paths inside `group_partition_multi`; only the
// return shape differs (Vec<Vec<KeyValue>> keys, Vec<DataType> dtypes).

use arrow::array::BooleanArray;
use arrow::datatypes::{Field, Schema};
use std::sync::Arc;

fn batch_with_group(col: ArrayRef) -> RecordBatch {
    let nullable = col.null_count() > 0;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "g",
        col.data_type().clone(),
        nullable,
    )]));
    RecordBatch::try_new(schema, vec![col]).unwrap()
}

#[test]
fn group_partition_int64_first_appearance_order() {
    // Values 20, 10, 20, 10 → first-appearance order is [20, 10].
    pyo3::Python::initialize();
    let col: ArrayRef = Arc::new(Int64Array::from(vec![20_i64, 10, 20, 10]));
    let batch = batch_with_group(col);
    let p = partition_skip(&batch, &["g".to_string()], "test").unwrap();
    assert_eq!(p.dtypes[0], DataType::Int64);
    assert_eq!(
        p.groups,
        vec![
            (vec![KeyValue::Int(20)], vec![0_usize, 2]),
            (vec![KeyValue::Int(10)], vec![1_usize, 3]),
        ]
    );
}

#[test]
fn group_partition_boolean_groups() {
    pyo3::Python::initialize();
    let col: ArrayRef = Arc::new(BooleanArray::from(vec![true, false, true]));
    let batch = batch_with_group(col);
    let p = partition_skip(&batch, &["g".to_string()], "test").unwrap();
    assert_eq!(p.dtypes[0], DataType::Boolean);
    assert_eq!(
        p.groups,
        vec![
            (vec![KeyValue::Bool(1)], vec![0_usize, 2]),
            (vec![KeyValue::Bool(0)], vec![1_usize]),
        ]
    );
}

#[test]
fn group_partition_skips_null_keys() {
    // Centralized null-key policy: null rows are excluded from the partition.
    pyo3::Python::initialize();
    let col: ArrayRef = Arc::new(Int64Array::from(vec![Some(1_i64), None, Some(1), None]));
    let batch = batch_with_group(col);
    let p = partition_skip(&batch, &["g".to_string()], "test").unwrap();
    assert_eq!(p.groups, vec![(vec![KeyValue::Int(1)], vec![0_usize, 2])]);
    assert!(
        !p.groups.iter().any(|(k, _)| k.contains(&KeyValue::Null)),
        "null key must be skipped"
    );
}

#[test]
fn group_partition_unsupported_dtype_errors() {
    pyo3::Python::initialize();
    use arrow::array::Date32Array;
    let col: ArrayRef = Arc::new(Date32Array::from(vec![0_i32, 1]));
    let batch = batch_with_group(col);
    let err = partition_skip(&batch, &["g".to_string()], "stat_kde").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("unsupported dtype"), "err: {msg}");
    assert!(msg.contains("stat_kde"), "err must carry ctx: {msg}");
}

#[test]
fn group_partition_missing_column_errors() {
    pyo3::Python::initialize();
    let col: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 2]));
    let batch = batch_with_group(col);
    let err = partition_skip(&batch, &["ghost".to_string()], "stat_smooth").unwrap_err();
    assert!(err.to_string().contains("ghost"), "err: {err}");
}

// ── T3.8a regression: deserialize_groupby_vec wire formats ──────────────

/// `deserialize_groupby_vec` accepts the three legacy wire forms:
///   - JSON `null` → `[]`
///   - JSON bare string `"col"` → `["col"]`
///   - JSON array `["a","b"]` → `["a","b"]`
#[test]
fn deserialize_groupby_vec_accepts_null_string_and_array() {
    use serde_json;

    // Wrap in a helper struct so we can drive the serde machinery.
    #[derive(serde::Deserialize)]
    struct W {
        #[serde(deserialize_with = "deserialize_groupby_vec")]
        groupby: Vec<String>,
    }

    let null_form: W = serde_json::from_str(r#"{"groupby": null}"#).unwrap();
    assert!(null_form.groupby.is_empty(), "null must deserialize to []");

    let str_form: W = serde_json::from_str(r#"{"groupby": "col"}"#).unwrap();
    assert_eq!(str_form.groupby, vec!["col"], "bare string must deserialize to [\"col\"]");

    let arr_form: W = serde_json::from_str(r#"{"groupby": ["a","b"]}"#).unwrap();
    assert_eq!(arr_form.groupby, vec!["a", "b"], "array must deserialize as-is");
}

/// When `groupby` is absent from the JSON, `default` yields `[]`.
#[test]
fn deserialize_groupby_vec_missing_field_yields_empty() {
    #[derive(serde::Deserialize)]
    struct W {
        #[serde(default, deserialize_with = "deserialize_groupby_vec")]
        groupby: Vec<String>,
    }

    let absent: W = serde_json::from_str(r#"{}"#).unwrap();
    assert!(absent.groupby.is_empty(), "missing field must default to []");
}

// ── T3.8a regression: group_partition_multi ──────────────────────────────

/// `group_partition_multi` on two string columns produces composite keys
/// (one `Vec<KeyValue>` per group) and preserves first-appearance order.
#[test]
fn group_partition_multi_two_string_columns() {
    pyo3::Python::initialize();
    use arrow::array::StringArray;
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("g1", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("g2", arrow::datatypes::DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["A", "A", "B", "B"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["x", "y", "x", "y"])) as ArrayRef,
        ],
    )
    .unwrap();

    let p = partition_skip(&batch, &["g1".to_string(), "g2".to_string()], "test").unwrap();

    assert_eq!(p.dtypes.len(), 2);
    assert_eq!(p.dtypes[0], arrow::datatypes::DataType::Utf8);
    assert_eq!(p.dtypes[1], arrow::datatypes::DataType::Utf8);

    // Four distinct groups in first-appearance order, each with its rows.
    assert_eq!(
        p.groups,
        vec![
            (vec![KeyValue::Str("A".into()), KeyValue::Str("x".into())], vec![0]),
            (vec![KeyValue::Str("A".into()), KeyValue::Str("y".into())], vec![1]),
            (vec![KeyValue::Str("B".into()), KeyValue::Str("x".into())], vec![2]),
            (vec![KeyValue::Str("B".into()), KeyValue::Str("y".into())], vec![3]),
        ]
    );
}

/// Rows where any dimension is null are skipped by `group_partition_multi`.
#[test]
fn group_partition_multi_skips_null_rows() {
    pyo3::Python::initialize();
    use arrow::array::StringArray;
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("g1", arrow::datatypes::DataType::Utf8, true),
        arrow::datatypes::Field::new("g2", arrow::datatypes::DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![Some("A"), None, Some("B")])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("x"), Some("x"), Some("y")])) as ArrayRef,
        ],
    )
    .unwrap();

    let p = partition_skip(&batch, &["g1".to_string(), "g2".to_string()], "test").unwrap();

    // Row 1 has g1=null → excluded; only rows 0 and 2 appear.
    assert_eq!(p.groups.len(), 2);
    assert_eq!(p.groups[0].1, vec![0]);
    assert_eq!(p.groups[1].1, vec![2]);
}

// ── BUG-HUNT (step4): partition order, multi-col, deserializer adversarial ──

/// Groups come back in FIRST-APPEARANCE order, even though the internal
/// key index is an `Ord`-keyed `BTreeMap`. When first-appearance order
/// DIFFERS from sorted order the returned order must follow first
/// appearance, not the map's sorted iteration — otherwise downstream group
/// columns would be emitted in the wrong order. Insert "B" before "A" so
/// the two orders differ.
#[test]
fn bughunt_group_order_is_first_appearance_not_sorted() {
    pyo3::Python::initialize();
    use arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![Field::new("g", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(vec!["B", "A", "B", "A", "C"])) as ArrayRef],
    )
    .unwrap();
    let p = partition_skip(&batch, &["g".to_string()], "test").unwrap();
    assert_eq!(
        p.groups.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
        vec![
            vec![KeyValue::Str("B".into())],
            vec![KeyValue::Str("A".into())],
            vec![KeyValue::Str("C".into())],
        ],
        "group order must be first-appearance (B,A,C), not sorted (A,B,C)"
    );
}

/// A 3-column groupby with interleaved combos: the composite-key partition must
/// produce exactly the distinct tuples, in first-appearance order, with each
/// row routed to the correct tuple. Tests joint partitioning across 3 dims.
#[test]
fn bughunt_group_partition_multi_three_columns_joint() {
    pyo3::Python::initialize();
    use arrow::array::{Int64Array, StringArray};
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, false),
        Field::new("b", DataType::Int64, false),
        Field::new("c", DataType::Utf8, false),
    ]));
    // 5 rows; distinct (a,b,c): (A,1,x)[rows 0,3], (A,2,x)[1], (B,1,x)[2], (A,1,y)[4]
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["A", "A", "B", "A", "A"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 1, 1, 1])) as ArrayRef,
            Arc::new(StringArray::from(vec!["x", "x", "x", "x", "y"])) as ArrayRef,
        ],
    )
    .unwrap();
    let p = partition_skip(
        &batch,
        &["a".to_string(), "b".to_string(), "c".to_string()],
        "test",
    )
    .unwrap();
    assert_eq!(p.dtypes, vec![DataType::Utf8, DataType::Int64, DataType::Utf8]);
    assert_eq!(p.groups.len(), 4, "expected 4 distinct triples");
    // First-appearance: (A,1,x), (A,2,x), (B,1,x), (A,1,y)
    let k_a1x = vec![KeyValue::Str("A".into()), KeyValue::Int(1), KeyValue::Str("x".into())];
    assert_eq!(p.groups[0].0, k_a1x);
    assert_eq!(p.groups[0].1, vec![0, 3], "(A,1,x) must collect rows 0 and 3");
    let k_a1y = vec![KeyValue::Str("A".into()), KeyValue::Int(1), KeyValue::Str("y".into())];
    let a1y = p.groups.iter().find(|(k, _)| *k == k_a1y).expect("(A,1,y) group");
    assert_eq!(a1y.1, vec![4], "(A,1,y) must be distinct from (A,1,x)");
}

/// Null-skip is ANY-dimension: a row with a null in the SECOND column (first
/// non-null) must still be skipped entirely. Pins that the early `break` on the
/// first null dimension doesn't accidentally retain rows whose later dims are null.
#[test]
fn bughunt_group_partition_multi_null_in_later_dim_skips_row() {
    pyo3::Python::initialize();
    use arrow::array::{Int64Array, StringArray};
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, false),
        Field::new("b", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["A", "A", "A"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(1_i64), None, Some(1)])) as ArrayRef,
        ],
    )
    .unwrap();
    let p = partition_skip(&batch, &["a".to_string(), "b".to_string()], "test").unwrap();
    assert_eq!(p.groups.len(), 1, "only (A,1) survives; (A,null) row is skipped");
    assert_eq!(p.groups[0].1, vec![0, 2], "rows 0 and 2 only; row 1 (b=null) excluded");
}

/// Float group keys: a genuine NaN value and a null must NOT collide (FA-3),
/// and both must be distinct from a finite key, when used as a 2-D composite.
#[test]
fn bughunt_group_partition_multi_float_nan_vs_null_distinct() {
    pyo3::Python::initialize();
    let schema = Arc::new(Schema::new(vec![Field::new("g", DataType::Float64, true)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Float64Array::from(vec![Some(1.0), Some(f64::NAN), None, Some(1.0)])) as ArrayRef],
    )
    .unwrap();
    let p = partition_skip(&batch, &["g".to_string()], "test").unwrap();
    // null row skipped; 1.0 and NaN form two distinct groups.
    assert_eq!(p.groups.len(), 2, "1.0 and NaN are two groups; null is skipped");
    let k_one = vec![KeyValue::Float(1.0_f64.to_bits())];
    let one = p.groups.iter().find(|(k, _)| *k == k_one).expect("1.0 group");
    assert_eq!(one.1, vec![0, 3], "the two 1.0 rows group together");
    assert!(
        p.groups
            .iter()
            .any(|(k, _)| matches!(k[0], KeyValue::Float(b) if f64::from_bits(b).is_nan())),
        "a NaN-keyed group must exist, distinct from null"
    );
}

/// `deserialize_groupby_vec` must REJECT a non-string array element (e.g. an
/// integer), not silently coerce or drop it. Groupby column names are strings.
#[test]
fn bughunt_deserialize_groupby_vec_rejects_non_string_element() {
    #[derive(serde::Deserialize)]
    struct W {
        #[serde(deserialize_with = "deserialize_groupby_vec")]
        #[allow(dead_code)]
        groupby: Vec<String>,
    }
    let r: Result<W, _> = serde_json::from_str(r#"{"groupby": ["a", 7]}"#);
    assert!(r.is_err(), "an integer array element must be rejected, not coerced");
}

/// `deserialize_groupby_vec` on an empty array yields `[]` (ungrouped), exactly
/// like null/absent — three spellings of "no grouping" must all converge.
#[test]
fn bughunt_deserialize_groupby_vec_empty_array_is_ungrouped() {
    #[derive(serde::Deserialize)]
    struct W {
        #[serde(deserialize_with = "deserialize_groupby_vec")]
        groupby: Vec<String>,
    }
    let w: W = serde_json::from_str(r#"{"groupby": []}"#).unwrap();
    assert!(w.groupby.is_empty(), "empty array must mean ungrouped");
}

/// `materialize_groupby_col` indexes `k[gi]`; an out-of-range `gi` would panic.
/// Pins that materializing dim 0 of a single-dim key vec is valid and that the
/// null/value mix produces the right Arrow null mask for a string column.
#[test]
fn bughunt_materialize_string_col_null_mask_correct() {
    let keys: Vec<Vec<KeyValue>> = vec![
        vec![KeyValue::Str("A".into())],
        vec![KeyValue::Null],
        vec![KeyValue::Str("B".into())],
    ];
    let col = materialize_groupby_col(&keys, 0, &DataType::Utf8).unwrap();
    let s = col.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(s.value(0), "A");
    assert!(s.is_null(1), "Null key must materialize as Arrow null, not empty string");
    assert_eq!(s.value(2), "B");
}

// ── stack_grouped_batches (batch-F T2h hoist) ─────────────────────────

fn xy_group_batch(x: Vec<f64>, y: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(x)) as ArrayRef,
            Arc::new(Float64Array::from(y)) as ArrayRef,
        ],
    )
    .unwrap()
}

/// The ordinary case: two groups' per-group output batches stack into
/// one batch with the group column appended at its original dtype,
/// rows in the same order the per-group batches were supplied.
#[test]
fn stack_grouped_batches_happy_path_appends_group_column() {
    let per_group_batches: Vec<(Vec<KeyValue>, RecordBatch)> = vec![
        (vec![KeyValue::Str("a".into())], xy_group_batch(vec![1.0, 2.0], vec![10.0, 20.0])),
        (vec![KeyValue::Str("b".into())], xy_group_batch(vec![3.0], vec![30.0])),
    ];
    let out = stack_grouped_batches(
        "test",
        &per_group_batches,
        &["g".to_string()],
        &[DataType::Utf8],
    )
    .unwrap();

    assert_eq!(out.num_rows(), 3);
    assert_eq!(out.schema().fields().len(), 3);
    assert_eq!(out.schema().field(2).name(), "g");
    let g_col = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(g_col.value(0), "a");
    assert_eq!(g_col.value(1), "a");
    assert_eq!(g_col.value(2), "b");
}

/// Batch-F T2h rider 1: a groupby column whose name collides with one of
/// the per-group output's own field names must refuse with a typed
/// error naming the colliding column, rather than appending a second
/// same-named field that `Schema::index_of` would silently resolve to
/// the wrong one.
#[test]
fn stack_grouped_batches_rejects_name_collision_with_output_column() {
    pyo3::Python::initialize();
    let per_group_batches: Vec<(Vec<KeyValue>, RecordBatch)> = vec![
        (vec![KeyValue::Str("a".into())], xy_group_batch(vec![1.0], vec![10.0])),
    ];
    let err = stack_grouped_batches(
        "test",
        &per_group_batches,
        &["y".to_string()],
        &[DataType::Utf8],
    )
    .expect_err("groupby column 'y' collides with the output's own 'y' field");
    let msg = err.to_string();
    assert!(msg.contains('y'), "error must name the colliding column: {msg}");
    assert!(msg.contains("collides"), "error must explain the collision: {msg}");
    assert!(
        msg.contains("x") && msg.contains("y"),
        "error must name the transform's actual output columns so the message is \
         actionable from Python: {msg}"
    );
}

/// Task 2h review finding 3: the moved multi-column append loop
/// (`group_cols.iter().zip(gtypes.iter()).enumerate()`) is where a
/// transposed `gi`/dtype pairing would hide, and nothing exercised
/// `gi > 0` through this helper. Two group columns of different dtypes
/// (Utf8 then Int64) pins that each output column gets ITS OWN dtype and
/// values, not the first dimension's for both.
#[test]
fn stack_grouped_batches_two_column_groupby_preserves_each_dtype() {
    use arrow::array::Int64Array;
    pyo3::Python::initialize();
    let per_group_batches: Vec<(Vec<KeyValue>, RecordBatch)> = vec![
        (
            vec![KeyValue::Str("a".into()), KeyValue::Int(1)],
            xy_group_batch(vec![1.0, 2.0], vec![10.0, 20.0]),
        ),
        (
            vec![KeyValue::Str("b".into()), KeyValue::Int(2)],
            xy_group_batch(vec![3.0], vec![30.0]),
        ),
    ];
    let out = stack_grouped_batches(
        "test",
        &per_group_batches,
        &["g1".to_string(), "g2".to_string()],
        &[DataType::Utf8, DataType::Int64],
    )
    .unwrap();

    assert_eq!(out.num_rows(), 3);
    assert_eq!(out.schema().fields().len(), 4, "x, y, g1, g2");
    assert_eq!(out.schema().field(2).name(), "g1");
    assert_eq!(out.schema().field(2).data_type(), &DataType::Utf8);
    assert_eq!(out.schema().field(3).name(), "g2");
    assert_eq!(out.schema().field(3).data_type(), &DataType::Int64);

    let g1 = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(g1.value(0), "a");
    assert_eq!(g1.value(1), "a");
    assert_eq!(g1.value(2), "b");

    let g2 = out.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(g2.value(0), 1);
    assert_eq!(g2.value(1), 1);
    assert_eq!(g2.value(2), 2);
}

/// Task 2h review finding 2, re-homed by the batch-F close review (F1):
/// `group_cols` repeating a name (e.g. a caller passing
/// `groupby=["g", "g"]`) must refuse — the guard exists to establish "no
/// duplicate field names in the output", not merely "no group column
/// collides with a fit column". It lives in `group_partition_multi` (the
/// shared partitioner) rather than in one stacking template, so
/// `kde`/`bin`/`kde_2d`/`top_k`/`impute`/`outliers` are covered by the
/// same eight lines that cover `smooth`/`robust`.
#[test]
fn group_partition_multi_rejects_duplicate_name_within_group_cols() {
    pyo3::Python::initialize();
    let batch = xy_group_batch(vec![1.0, 2.0], vec![10.0, 20.0]);
    // `x` exists in this batch, so the refusal is about the REPEAT, not
    // about a missing column.
    let err = partition_skip(&batch, &["x".to_string(), "x".to_string()], "test")
        .expect_err("groupby=['x', 'x'] must refuse rather than key on one column twice");
    let msg = err.to_string();
    assert!(msg.contains('x'), "error must name the repeated column: {msg}");
    assert!(msg.contains("repeated"), "error must explain the duplication: {msg}");
    assert!(msg.contains("test"), "error must stay in the caller's voice: {msg}");
}

// ── GH #179 / #197: the one partitioning idiom ──────────────────────────

/// The extracted output-collision guard refuses a groupby name that is also
/// one of the transform's own output columns, and its message names both
/// the offender and the full output list so the caller can act on it.
#[test]
fn refuse_groupby_output_collision_names_offender_and_output_columns() {
    pyo3::Python::initialize();
    let err = refuse_groupby_output_collision(
        "stat_summary",
        &["mean".to_string()],
        &["mean".to_string(), "lower".to_string(), "upper".to_string()],
    )
    .expect_err("a groupby column named like an output column must refuse");
    let msg = err.to_string();
    assert!(msg.contains("stat_summary"), "error stays in the caller's voice: {msg}");
    assert!(msg.contains("collides"), "error explains the collision: {msg}");
    assert!(
        msg.contains("mean, lower, upper"),
        "error names the transform's own output columns: {msg}"
    );
}

/// The guard passes a groupby that shares no name with the outputs — the
/// ordinary case must not be refused.
#[test]
fn refuse_groupby_output_collision_accepts_disjoint_names() {
    pyo3::Python::initialize();
    assert!(refuse_groupby_output_collision(
        "stat_summary",
        &["g".to_string()],
        &["mean".to_string(), "lower".to_string(), "upper".to_string()],
    )
    .is_ok());
}

/// `NullKeyPolicy::Pool` keeps null-keyed rows as one pooled group instead
/// of dropping them — the row-preserving exemption (`data_stack`,
/// `data_join_aggregate`), where a dropped row would silently hold a
/// default value rather than a computed one.
#[test]
fn null_key_policy_pool_keeps_null_keyed_rows_as_one_group() {
    pyo3::Python::initialize();
    let col: ArrayRef = Arc::new(Int64Array::from(vec![Some(1_i64), None, Some(1), None]));
    let batch = batch_with_group(col);
    let p = group_partition_multi(
        &batch,
        &["g".to_string()],
        "data_stack",
        NullKeyPolicy::Pool,
        &TransformContext::default(),
    )
    .unwrap();
    assert_eq!(p.groups.len(), 2, "the real key and the pooled null key");
    assert_eq!(p.groups[0], (vec![KeyValue::Int(1)], vec![0, 2]));
    assert_eq!(
        p.groups[1],
        (vec![KeyValue::Null], vec![1, 3]),
        "every null-keyed row lands in ONE pooled group, in first-appearance order"
    );
}

/// An all-null key column warns once through the #162 channel, naming the
/// transform, the groupby columns, and the policy's own disposition.
#[test]
fn all_null_group_keys_warn_once_naming_columns_and_disposition() {
    pyo3::Python::initialize();
    let col: ArrayRef = Arc::new(Int64Array::from(vec![None, None, None] as Vec<Option<i64>>));
    let batch = batch_with_group(col);
    let ctx = TransformContext::default();
    let p = group_partition_multi(
        &batch,
        &["g".to_string()],
        "stat_summary",
        NullKeyPolicy::Skip,
        &ctx,
    )
    .unwrap();
    assert!(p.groups.is_empty(), "the adjudicated arm is the empty group set");

    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1, "exactly one warning, not one per row");
    let text = warnings[0].to_string();
    // "Summary", not the wire tag "stat_summary" — wave-1 close remediation
    // (`display_name_for_label`).
    assert!(text.contains("Summary"), "names the transform: {text}");
    assert!(text.contains('g'), "names the groupby column: {text}");
    assert!(
        text.contains("the grouped output is empty"),
        "carries the Skip policy's own disposition: {text}"
    );
}

/// Each policy reports its OWN disposition, so a transform that falls back
/// to a whole-batch computation or pools its rows never claims an empty
/// output it did not produce.
#[test]
fn all_null_warning_disposition_follows_the_policy() {
    pyo3::Python::initialize();
    let mk = |policy: NullKeyPolicy| {
        let col: ArrayRef =
            Arc::new(Int64Array::from(vec![None, None] as Vec<Option<i64>>));
        let batch = batch_with_group(col);
        let ctx = TransformContext::default();
        group_partition_multi(&batch, &["g".to_string()], "T", policy, &ctx).unwrap();
        let text = ctx.warnings.borrow()[0].to_string();
        text
    };
    assert!(mk(NullKeyPolicy::Skip).contains("the grouped output is empty"));
    assert!(mk(NullKeyPolicy::SkipWithWholeBatchFallback)
        .contains("the whole batch was used as one ungrouped computation instead"));
    assert!(mk(NullKeyPolicy::Pool).contains("all rows were treated as one unlabeled group"));
}

/// Wave-1 close remediation (S2 finding): `GroupKeyAllNull`'s `transform`
/// field must use the same Python-facing vocabulary
/// `DegenerateFit`/`ParameterClamped`/`FallbackApplied` already use (the
/// `TransformSpec` variant's own name), not the wire/type tag every
/// `group_partition_multi` call site passes as `label` for its OWN error
/// text. One render used to say `"data_impute: ..."` immediately followed by
/// `"Impute: ..."` for the SAME transform; this test pins every current
/// call site's mapping so a sixteenth silent mismatch (`robust.rs`'s
/// pre-fix "Robust" vs. its siblings' wire tags) cannot recur unnoticed.
#[test]
fn group_key_all_null_warning_uses_the_python_facing_name_not_the_wire_tag() {
    pyo3::Python::initialize();
    let mk_transform = |label: &str| {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![None, None] as Vec<Option<i64>>));
        let batch = batch_with_group(col);
        let ctx = TransformContext::default();
        group_partition_multi(&batch, &["g".to_string()], label, NullKeyPolicy::Skip, &ctx)
            .unwrap();
        let warnings = ctx.warnings.borrow();
        match &warnings[0] {
            crate::transform::context::TransformWarning::GroupKeyAllNull { transform, .. } => {
                transform.clone()
            }
            other => panic!("expected GroupKeyAllNull, got {other:?}"),
        }
    };
    let cases = [
        ("stat_bin", "Bin"),
        ("data_top_k", "TopK"),
        ("data_aggregate", "DataAggregate"),
        ("data_join_aggregate", "JoinAggregate"),
        ("stat_aggregate", "Aggregate"),
        ("data_density", "DensityData"),
        ("data_pivot", "Pivot"),
        ("stat_error_extent", "ErrorExtent"),
        ("stat_kde_2d", "Kde2D"),
        ("data_stack", "DataStack"),
        ("stat_violin", "Violin"),
        ("stat_box_stats", "BoxStats"),
        ("stat_outliers", "Outliers"),
        ("stat_summary", "Summary"),
        ("stat_smooth", "Smooth"),
        ("stat_kde", "Kde"),
        ("stat_qq", "QQ"),
        ("stat_letter_value", "LetterValue"),
        ("data_impute", "Impute"),
        ("data_window", "DataWindow"),
        // `robust.rs` already passes the display name as `label` (the
        // odd one out among its siblings) — identity, not a gap.
        ("Robust", "Robust"),
    ];
    for (label, expected) in cases {
        assert_eq!(
            mk_transform(label), expected,
            "label {label:?} must map to display name {expected:?}"
        );
    }
}

/// The warning is for the ALL-null case only: a column with even one usable
/// key produced groups, so nothing degraded and nothing is said.
#[test]
fn partially_null_group_keys_do_not_warn() {
    pyo3::Python::initialize();
    let col: ArrayRef = Arc::new(Int64Array::from(vec![None, Some(1_i64)]));
    let batch = batch_with_group(col);
    let ctx = TransformContext::default();
    group_partition_multi(&batch, &["g".to_string()], "T", NullKeyPolicy::Skip, &ctx)
        .unwrap();
    assert!(ctx.warnings.borrow().is_empty(), "a partially-null key column is not a degrade");
}

/// MULTI-COLUMN, the shape that separates the policies (review finding
/// G1): `groupby=["a","b"]` with an all-null `a` and a varying `b`.
///
/// Under the skipping policies every row is excluded, so the group set IS
/// empty and the warning is true. Under `Pool` the rows still key on `b`,
/// so the batch DID split into groups — claiming "all rows were treated as
/// one unlabeled group" there would be false in both of its clauses, and
/// the single-column tests above cannot see the difference.
#[test]
fn multi_column_all_null_dimension_warns_only_when_grouping_really_failed() {
    pyo3::Python::initialize();
    use arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, true),
        Field::new("b", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![None, None, None] as Vec<Option<&str>>)) as ArrayRef,
            Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef,
        ],
    )
    .unwrap();
    let cols = vec!["a".to_string(), "b".to_string()];

    for policy in [NullKeyPolicy::Skip, NullKeyPolicy::SkipWithWholeBatchFallback] {
        let ctx = TransformContext::default();
        let p = group_partition_multi(&batch, &cols, "T", policy, &ctx).unwrap();
        assert!(p.groups.is_empty(), "{policy:?}: a null in any dimension excludes the row");
        assert_eq!(ctx.warnings.borrow().len(), 1, "{policy:?} must warn");
    }

    let ctx = TransformContext::default();
    let p = group_partition_multi(&batch, &cols, "T", NullKeyPolicy::Pool, &ctx).unwrap();
    assert_eq!(
        p.groups.len(),
        2,
        "Pool still splits on the non-null dimension: one group per distinct b"
    );
    assert!(
        ctx.warnings.borrow().is_empty(),
        "grouping succeeded, so nothing degraded; got {:?}",
        ctx.warnings.borrow()
    );
}

/// The `Pool` warning fires when the key is null in EVERY dimension — then
/// there really is one unlabeled group holding everything.
#[test]
fn multi_column_fully_null_key_warns_under_pool() {
    pyo3::Python::initialize();
    use arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, true),
        Field::new("b", DataType::Utf8, true),
    ]));
    let nulls = || Arc::new(StringArray::from(vec![None, None] as Vec<Option<&str>>)) as ArrayRef;
    let batch = RecordBatch::try_new(schema, vec![nulls(), nulls()]).unwrap();
    let ctx = TransformContext::default();
    let p = group_partition_multi(
        &batch,
        &["a".to_string(), "b".to_string()],
        "data_stack",
        NullKeyPolicy::Pool,
        &ctx,
    )
    .unwrap();
    assert_eq!(p.groups.len(), 1, "one pooled group holding every row");
    assert_eq!(p.groups[0].1, vec![0, 1]);
    let warnings = ctx.warnings.borrow();
    assert_eq!(warnings.len(), 1);
    assert!(
        warnings[0].to_string().contains("all rows were treated as one unlabeled group"),
        "{}",
        warnings[0]
    );
}

/// An UNGROUPED request never warns, however empty the batch: there was no
/// grouping to fail.
#[test]
fn ungrouped_request_never_warns_about_null_keys() {
    pyo3::Python::initialize();
    let col: ArrayRef = Arc::new(Int64Array::from(vec![None, None] as Vec<Option<i64>>));
    let batch = batch_with_group(col);
    let ctx = TransformContext::default();
    group_partition_multi(&batch, &[], "T", NullKeyPolicy::Skip, &ctx).unwrap();
    assert!(ctx.warnings.borrow().is_empty());
}

/// [`ensure_ungrouped_whole_batch_group`] only fires for the ungrouped
/// zero-row shape, and never invents a group for a real `groupby`.
#[test]
fn ensure_ungrouped_whole_batch_group_only_fires_when_ungrouped_and_empty() {
    pyo3::Python::initialize();
    let empty: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
    let batch = batch_with_group(empty);

    let mut ungrouped =
        partition_skip(&batch, &[], "stat_aggregate").unwrap();
    ensure_ungrouped_whole_batch_group(&mut ungrouped, &[]);
    assert_eq!(
        ungrouped.groups,
        vec![(Vec::new(), Vec::new())],
        "a zero-row ungrouped aggregate still collapses to ONE output row"
    );

    let mut grouped =
        partition_skip(&batch, &["g".to_string()], "stat_aggregate").unwrap();
    ensure_ungrouped_whole_batch_group(&mut grouped, &["g".to_string()]);
    assert!(grouped.groups.is_empty(), "a real groupby on no rows stays empty");
}

/// [`refuse_repeated_groupby_names`] is the one rule the five PyO3
/// constructors and the partitioner now share, so the same request refuses
/// with the same sentence whichever boundary catches it first.
#[test]
fn refuse_repeated_groupby_names_is_one_rule_in_the_callers_voice() {
    pyo3::Python::initialize();
    assert!(refuse_repeated_groupby_names("Aggregate", &["a".to_string(), "b".to_string()])
        .is_ok());
    let err =
        refuse_repeated_groupby_names("Aggregate", &["g".to_string(), "g".to_string()])
            .expect_err("a repeated name must refuse");
    let msg = err.to_string();
    assert!(msg.contains("Aggregate"), "{msg}");
    assert!(msg.contains("repeated"), "{msg}");
    assert!(msg.contains("distinct name"), "{msg}");
}

/// `validate_groupby_columns` — the rules `data_window` shares without
/// sharing the partition algorithm — enforces all three refusals and
/// returns each column's index and dtype in request order.
#[test]
fn validate_groupby_columns_enforces_the_three_request_rules() {
    pyo3::Python::initialize();
    let batch = xy_with_group_col();
    let ok = validate_groupby_columns(&batch, &["g".to_string()], "data_window").unwrap();
    assert_eq!(ok, vec![(2_usize, DataType::Utf8)]);

    let missing = validate_groupby_columns(&batch, &["ghost".to_string()], "data_window")
        .expect_err("a missing column must refuse");
    assert!(missing.to_string().contains("not found"), "{missing}");

    let repeated =
        validate_groupby_columns(&batch, &["g".to_string(), "g".to_string()], "data_window")
            .expect_err("a repeated name must refuse");
    assert!(repeated.to_string().contains("repeated"), "{repeated}");
    assert!(repeated.to_string().contains("data_window"), "{repeated}");

    let bad_dtype = validate_groupby_columns(&batch, &["x".to_string()], "data_window");
    assert!(bad_dtype.is_ok(), "Float64 is a supported key dtype");
}

/// `x`/`y` Float64 plus a Utf8 `g` — the shape the validation tests need.
fn xy_with_group_col() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("g", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![10.0, 20.0])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
        ],
    )
    .unwrap()
}

/// A missing column that is ALSO repeated still reports "not found",
/// not "repeated" — i.e. the missing-column error is the one that fires
/// in the observable output when both conditions are present.
///
/// This does not pin a genuine ordering/precedence property: a name can
/// only be detected as repeated on its second occurrence, and by then
/// its first occurrence has already passed `schema.index_of` — a
/// missing name fails `index_of` on its very first occurrence, before a
/// repeat could ever be observed. So the two checks cannot actually
/// race on any input; this test documents which error text a caller
/// sees for this shape, not which check "wins".
#[test]
fn group_partition_multi_missing_column_reports_not_found_not_repeated() {
    pyo3::Python::initialize();
    let batch = xy_group_batch(vec![1.0], vec![10.0]);
    let err =
        partition_skip(&batch, &["ghost".to_string(), "ghost".to_string()], "test")
            .expect_err("a missing groupby column must still refuse");
    let msg = err.to_string();
    assert!(msg.contains("not found"), "missing column must report 'not found': {msg}");
}
