//! `complete_series` and the transform door over it, end to end: every
//! option, every value and key kind, the order an imputed row is inserted in,
//! and the refusals and warnings each case raises. It is a sibling `tests.rs`
//! beside `impute.rs` (named `complete_tests` because `impute.rs` keeps its
//! earlier inline `tests` module) because this body of cases would outgrow
//! the source it exercises; each helper module's own unit tests stay inline.

use super::*;
use arrow::array::{
    BooleanArray, Decimal128Array, Float32Array, Int32Array, Int64Array, LargeStringArray,
    StringArray, TimestampMillisecondArray, TimestampNanosecondArray, UInt8Array,
};
use arrow::datatypes::TimeUnit;
use std::collections::HashMap;
use serde_json::{json, Value};

fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> =
        cols.iter().map(|(name, arr)| Field::new(*name, arr.data_type().clone(), true)).collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols.into_iter().map(|(_, a)| a).collect())
        .unwrap()
}

fn f64s(values: &[Option<f64>]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec()))
}

fn ints(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}

fn strs(values: &[&str]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec()))
}

fn base(field: &str) -> ImputeSpec {
    ImputeSpec {
        field: field.into(),
        method: "value".into(),
        value: None,
        groupby: None,
        key: None,
        keyvals: None,
        frame: None,
        name: None,
    }
}

fn with_value(mut spec: ImputeSpec, value: Value) -> ImputeSpec {
    spec.value = Some(value);
    spec
}

fn keyed(mut spec: ImputeSpec, key: &str, groupby: &[&str]) -> ImputeSpec {
    spec.key = Some(key.into());
    spec.groupby = Some(groupby.iter().map(|g| g.to_string()).collect());
    spec
}

fn with_keyvals(mut spec: ImputeSpec, keyvals: Keyvals) -> ImputeSpec {
    spec.keyvals = Some(keyvals);
    spec
}

fn with_frame(mut spec: ImputeSpec, method: &str, frame: (Option<i64>, Option<i64>)) -> ImputeSpec {
    spec.method = method.into();
    spec.frame = Some(frame);
    spec
}

fn with_method(mut spec: ImputeSpec, method: &str) -> ImputeSpec {
    spec.method = method.into();
    spec
}

fn col<'a, T: 'static>(out: &'a RecordBatch, name: &str) -> &'a T {
    out.column_by_name(name).unwrap().as_any().downcast_ref::<T>().unwrap()
}

fn values(out: &RecordBatch, name: &str) -> Vec<f64> {
    col::<Float64Array>(out, name).values().to_vec()
}

fn labels(out: &RecordBatch, name: &str) -> Vec<String> {
    let arr = col::<StringArray>(out, name);
    (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
}

fn every(field: &str) -> Option<ImputeKey<'_>> {
    Some(ImputeKey { field, observed: ObservedKeys::Every, companions: &[] })
}

fn run(spec: &ImputeSpec, b: &RecordBatch) -> (PyResult<RecordBatch>, Vec<String>) {
    let ctx = TransformContext::default();
    let out = apply_with_context(spec, b, &ctx);
    let texts = ctx.warnings.borrow().iter().map(ToString::to_string).collect();
    (out, texts)
}

fn ok(spec: &ImputeSpec, b: &RecordBatch) -> RecordBatch {
    apply(spec, b).unwrap()
}

fn refusal(spec: &ImputeSpec, b: &RecordBatch) -> String {
    pyo3::Python::initialize();
    apply(spec, b).unwrap_err().to_string()
}

// ─── methods, with and without groups ──────────────────────────────────────

#[test]
fn every_method_fills_null_and_nan_cells_from_the_usable_values_and_keeps_an_infinity() {
    let b = batch(vec![(
        "v",
        f64s(&[Some(1.0), None, Some(f64::NAN), Some(3.0), Some(f64::INFINITY), Some(5.0)]),
    )]);
    for (method, fill) in [("mean", 3.0), ("median", 3.0), ("min", 1.0), ("max", 5.0)] {
        let out = ok(&with_method(base("v"), method), &b);
        assert_eq!(
            values(&out, "v"),
            vec![1.0, fill, fill, 3.0, f64::INFINITY, 5.0],
            "{method}: usable values 1, 3, 5; the infinite cell stays"
        );
    }
}

#[test]
fn every_method_fills_each_group_from_its_own_values() {
    let b = batch(vec![
        ("g", strs(&["a", "a", "a", "b", "b", "b"])),
        ("v", f64s(&[Some(1.0), None, Some(5.0), Some(10.0), Some(f64::NAN), Some(40.0)])),
    ]);
    for (method, a, bb) in [("mean", 3.0, 25.0), ("median", 3.0, 25.0), ("min", 1.0, 10.0), ("max", 5.0, 40.0)] {
        let mut spec = with_method(base("v"), method);
        spec.groupby = Some(vec!["g".into()]);
        let out = ok(&spec, &b);
        assert_eq!(values(&out, "v")[1], a, "{method}, group a");
        assert_eq!(values(&out, "v")[4], bb, "{method}, group b");
    }
}

#[test]
fn the_value_method_defaults_to_zero_and_fills_null_and_nan_cells() {
    let b = batch(vec![("v", f64s(&[Some(1.0), None, Some(f64::NAN), Some(f64::NEG_INFINITY)]))]);
    let out = ok(&base("v"), &b);
    assert_eq!(values(&out, "v"), vec![1.0, 0.0, 0.0, f64::NEG_INFINITY]);
    let out = ok(&with_value(base("v"), json!(2.5)), &b);
    assert_eq!(values(&out, "v"), vec![1.0, 2.5, 2.5, f64::NEG_INFINITY]);
    assert!(!out.schema().field(0).is_nullable());
    assert_eq!(out.schema().field(0).data_type(), &DataType::Float64);
}

#[test]
fn a_numeric_field_of_any_width_is_read_and_written_as_float64() {
    let int32: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]));
    let uint8: ArrayRef = Arc::new(UInt8Array::from(vec![Some(2), None, Some(4)]));
    let float32: ArrayRef = Arc::new(Float32Array::from(vec![Some(1.5), None, Some(2.5)]));
    for (arr, fill) in [(int32, 2.0), (uint8, 3.0), (float32, 2.0)] {
        let out = ok(&with_method(base("v"), "mean"), &batch(vec![("v", arr.clone())]));
        assert_eq!(out.schema().field(0).data_type(), &DataType::Float64, "{:?}", arr.data_type());
        assert_eq!(values(&out, "v")[1], fill, "{:?}", arr.data_type());
    }
}

#[test]
fn a_decimal_field_has_no_numeric_reading_and_is_refused_on_every_method() {
    let decimal: ArrayRef =
        Arc::new(Decimal128Array::from(vec![Some(100_i128), None]).with_precision_and_scale(10, 2).unwrap());
    let b = batch(vec![("v", decimal)]);
    for method in ["value", "mean"] {
        assert_eq!(
            refusal(&with_method(base("v"), method), &b),
            "ValueError: data_impute: column 'v' must be numeric, got Decimal128(10, 2)"
        );
    }
}

#[test]
fn a_value_beside_a_statistic_is_refused_for_each_statistic() {
    let b = batch(vec![("v", f64s(&[Some(1.0)]))]);
    for method in ["mean", "median", "min", "max"] {
        let spec = with_value(with_method(base("v"), method), json!(0));
        assert_eq!(
            refusal(&spec, &b),
            format!(
                "ValueError: data_impute: value= applies to method='value', not '{method}'; \
                 drop value= or set method='value'"
            )
        );
    }
}

#[test]
fn a_frame_beside_the_value_method_is_refused_even_when_open_on_both_sides() {
    let b = batch(vec![("v", f64s(&[Some(1.0)]))]);
    for frame in [(Some(-1), Some(1)), (None, None)] {
        let mut spec = base("v");
        spec.frame = Some(frame);
        assert_eq!(
            refusal(&spec, &b),
            "ValueError: data_impute: frame= applies to mean, median, min and max; method 'value' fills a constant"
        );
    }
}

#[test]
fn keyvals_without_a_key_is_refused_and_a_missing_field_or_method_is_named() {
    let b = batch(vec![("v", f64s(&[Some(1.0)]))]);
    let spec = with_keyvals(base("v"), Keyvals::List(vec![json!(1)]));
    assert_eq!(
        refusal(&spec, &b),
        "ValueError: data_impute: keyvals requires key=; the key universe is meaningless without a key column"
    );
    assert_eq!(refusal(&base("nope"), &b), "ValueError: data_impute: column 'nope' not found");
    assert_eq!(
        refusal(&with_method(base("v"), "bogus"), &b),
        "ValueError: data_impute: unknown method 'bogus'; expected value|mean|median|min|max"
    );
}

// ─── value kinds ───────────────────────────────────────────────────────────

#[test]
fn a_text_column_is_filled_with_a_string_and_keeps_its_dtype() {
    for large in [false, true] {
        let cells: Vec<Option<&str>> = vec![Some("a"), None, Some("c")];
        let arr: ArrayRef = if large {
            Arc::new(LargeStringArray::from(cells))
        } else {
            Arc::new(StringArray::from(cells))
        };
        let dtype = arr.data_type().clone();
        let out = ok(&with_value(base("grade"), json!("none")), &batch(vec![("grade", arr)]));
        assert_eq!(out.schema().field(0).data_type(), &dtype);
        assert!(!out.schema().field(0).is_nullable());
        let text = arrow::compute::cast(out.column(0), &DataType::Utf8).unwrap();
        let text = text.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!((0..3).map(|i| text.value(i)).collect::<Vec<_>>(), vec!["a", "none", "c"]);
    }
}

#[test]
fn a_boolean_column_is_filled_with_a_bool() {
    let b = batch(vec![("ok", Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])) as ArrayRef)]);
    let out = ok(&with_value(base("ok"), json!(false)), &b);
    let flags = col::<BooleanArray>(&out, "ok");
    assert_eq!((flags.value(0), flags.value(1), flags.value(2)), (true, false, false));
    let out = ok(&with_value(base("ok"), json!(true)), &b);
    assert!(col::<BooleanArray>(&out, "ok").value(1));
}

#[test]
fn a_timestamp_column_is_filled_with_an_instant_at_its_unit_and_zone() {
    let tagged = json!({"epoch_ms": 1_704_153_600_000_i64});
    let ms_zoned = DataType::Timestamp(TimeUnit::Millisecond, Some("Europe/Paris".into()));
    let ms: ArrayRef = Arc::new(
        TimestampMillisecondArray::from(vec![Some(1), None]).with_timezone("Europe/Paris"),
    );
    let ns: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![Some(1), None]));
    for value in [tagged.clone(), json!(1_704_153_600_000_i64), json!(1_704_153_600_000.0)] {
        let out = ok(&with_value(base("when"), value), &batch(vec![("when", ms.clone())]));
        assert_eq!(out.schema().field(0).data_type(), &ms_zoned, "unit and zone are the column's");
        assert_eq!(col::<TimestampMillisecondArray>(&out, "when").value(1), 1_704_153_600_000);
    }
    let out = ok(&with_value(base("when"), tagged), &batch(vec![("when", ns)]));
    assert_eq!(col::<TimestampNanosecondArray>(&out, "when").value(1), 1_704_153_600_000_000_000);
}

#[test]
fn a_value_of_another_kind_than_its_column_is_refused_through_the_transform() {
    let cases: Vec<(ArrayRef, Value, &str)> = vec![
        (f64s(&[None]), json!("none"), "holds numbers, so value= must be a number, but 'none' is text"),
        (
            f64s(&[None]),
            json!({"epoch_ms": 1_704_067_200_000_i64}),
            "holds numbers, so value= must be a number, but 2024-01-01T00:00:00 is a date or datetime",
        ),
        (strs(&["a"]), json!(5), "holds text, so value= must be a string, but 5 is a number"),
        (
            Arc::new(BooleanArray::from(vec![Some(true)])),
            json!("none"),
            "holds booleans, so value= must be True or False, but 'none' is text",
        ),
        (
            Arc::new(TimestampMillisecondArray::from(vec![Some(1)])),
            json!(true),
            "holds dates and times, so value= must be a datetime.date, a datetime.datetime or an \
             epoch-millisecond number, but True is a boolean",
        ),
    ];
    for (arr, value, tail) in cases {
        let b = batch(vec![("c", arr)]);
        assert_eq!(
            refusal(&with_value(base("c"), value), &b),
            format!("ValueError: data_impute: column 'c' {tail}")
        );
    }
}

#[test]
fn a_column_that_is_not_numeric_has_no_default_value_and_no_statistic() {
    let text = batch(vec![("grade", strs(&["a"]))]);
    assert_eq!(
        refusal(&base("grade"), &text),
        "ValueError: data_impute: column 'grade' holds text, so value= must be a string, and none was given; pass value="
    );
    let boolean = batch(vec![("ok", Arc::new(BooleanArray::from(vec![true])) as ArrayRef)]);
    let when = batch(vec![("when", Arc::new(TimestampMillisecondArray::from(vec![1])) as ArrayRef)]);
    for (b, field, got) in [
        (&text, "grade", "Utf8"),
        (&boolean, "ok", "Boolean"),
        (&when, "when", "Timestamp(Millisecond, None)"),
    ] {
        for method in ["mean", "median", "min", "max"] {
            assert_eq!(
                refusal(&with_method(base(field), method), b),
                format!("ValueError: data_impute: column '{field}' must be numeric, got {got}")
            );
        }
    }
}

#[test]
fn a_value_that_is_no_literal_is_refused_by_the_engine() {
    let b = batch(vec![("v", f64s(&[None]))]);
    for value in [json!(null), json!([1]), json!({"a": 1})] {
        assert_eq!(
            refusal(&with_value(base("v"), value.clone()), &b),
            format!(
                "ValueError: data_impute: value= must be a number, a string, True or False, a date \
                 or a datetime; got {value}"
            )
        );
    }
}

// ─── the order an imputed row is inserted in ───────────────────────────────

/// Group x holds keys 0 and 2 of a three-key universe, group y holds all three:
/// x lacks the middle one. `v` is the row's own number; the imputed row is -1.
fn interior(key: ArrayRef) -> RecordBatch {
    batch(vec![
        ("g", strs(&["x", "x", "y", "y", "y"])),
        ("t", key),
        ("v", f64s(&[Some(10.0), Some(30.0), Some(1.0), Some(2.0), Some(3.0)])),
    ])
}

fn complete(b: &RecordBatch) -> RecordBatch {
    ok(&keyed(with_value(base("v"), json!(-1)), "t", &["g"]), b)
}

#[test]
fn an_interior_key_is_inserted_between_its_neighbours_for_every_key_dtype() {
    let order = [0_usize, 2, 0, 1, 2];
    let keys: Vec<(&str, ArrayRef)> = vec![
        ("Int64", ints(&order.map(|i| [-5, 0, 7][i]))),
        ("UInt8", Arc::new(UInt8Array::from(order.map(|i| [1_u8, 2, 3][i]).to_vec()))),
        ("Float64", f64s(&order.map(|i| Some([-1.5, 0.0, 2.5][i])))),
        ("Float32", Arc::new(Float32Array::from(order.map(|i| [-1.5_f32, 0.0, 2.5][i]).to_vec()))),
        ("Int32", Arc::new(Int32Array::from(order.map(|i| [-9, 0, 9][i]).to_vec()))),
        (
            "Timestamp ms zoned",
            Arc::new(
                TimestampMillisecondArray::from(order.map(|i| [-1000_i64, 0, 5000][i]).to_vec())
                    .with_timezone("Europe/Paris"),
            ),
        ),
        (
            "Timestamp ns",
            Arc::new(TimestampNanosecondArray::from(order.map(|i| [-1_i64, 0, 5][i]).to_vec())),
        ),
    ];
    for (name, key) in keys {
        let out = complete(&interior(key));
        assert_eq!(values(&out, "v"), vec![10.0, -1.0, 30.0, 1.0, 2.0, 3.0], "{name}");
        assert_eq!(labels(&out, "g"), vec!["x", "x", "x", "y", "y", "y"], "{name}");
    }
}

#[test]
fn a_text_key_follows_the_order_it_first_appears_in_not_alphabetical_order() {
    // the order is c, a, b; x holds a and b and lacks c, which comes first
    let b = batch(vec![
        ("g", strs(&["y", "y", "y", "x", "x"])),
        ("t", strs(&["c", "a", "b", "a", "b"])),
        ("v", f64s(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)])),
    ]);
    assert_eq!(values(&complete(&b), "v"), vec![1.0, 2.0, 3.0, -1.0, 4.0, 5.0]);
}

#[test]
fn a_boolean_key_places_false_before_true() {
    let b = batch(vec![
        ("g", strs(&["x", "y", "y"])),
        ("t", Arc::new(BooleanArray::from(vec![true, false, true])) as ArrayRef),
        ("v", f64s(&[Some(10.0), Some(1.0), Some(2.0)])),
    ]);
    let out = complete(&b);
    // x holds `true` and lacks `false`, which precedes it
    assert_eq!(values(&out, "v"), vec![-1.0, 10.0, 1.0, 2.0]);
    let flags = col::<BooleanArray>(&out, "t");
    assert_eq!((0..4).map(|i| flags.value(i)).collect::<Vec<_>>(), vec![false, true, false, true]);
}

#[test]
fn a_text_key_is_placed_in_first_appearance_order_and_keyvals_lead() {
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "y", "y"])),
        ("t", strs(&["c", "a", "b", "a", "b"])),
        ("v", f64s(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)])),
    ]);
    // the order is c, a, b: y lacks c, which precedes every row of y
    assert_eq!(values(&complete(&b), "v"), vec![1.0, 2.0, 3.0, -1.0, 4.0, 5.0]);
    // keyvals z leads the order and is added to both groups before their first rows
    let spec = with_keyvals(keyed(with_value(base("v"), json!(-1)), "t", &["g"]), Keyvals::List(vec![json!("z")]));
    assert_eq!(values(&ok(&spec, &b), "v"), vec![-1.0, 1.0, 2.0, 3.0, -1.0, -1.0, 4.0, 5.0]);
}

#[test]
fn a_callers_row_never_moves_and_a_key_with_no_later_row_follows_the_last_row() {
    // x is written 2, 0 and lacks 1: no row of x holds a key after 1 that comes last
    let b = batch(vec![
        ("g", strs(&["x", "x", "y", "y", "y"])),
        ("t", ints(&[2, 0, 0, 1, 2])),
        ("v", f64s(&[Some(30.0), Some(10.0), Some(1.0), Some(2.0), Some(3.0)])),
    ]);
    assert_eq!(values(&complete(&b), "v"), vec![30.0, 10.0, -1.0, 1.0, 2.0, 3.0]);
}

#[test]
fn a_key_below_every_row_of_a_series_goes_before_its_first_row_and_one_above_after_its_last() {
    let b = batch(vec![
        ("g", strs(&["x", "x", "y", "y", "y"])),
        ("t", ints(&[1, 2, 0, 1, 2])),
        ("v", f64s(&[Some(10.0), Some(20.0), Some(1.0), Some(2.0), Some(3.0)])),
    ]);
    // x lacks 0 (before its first row); with keyvals 3 both groups also lack the top key
    let spec = with_keyvals(keyed(with_value(base("v"), json!(-1)), "t", &["g"]), Keyvals::List(vec![json!(3)]));
    let out = ok(&spec, &b);
    assert_eq!(values(&out, "v"), vec![-1.0, 10.0, 20.0, -1.0, 1.0, 2.0, 3.0, -1.0]);
    assert_eq!(labels(&out, "g"), vec!["x", "x", "x", "x", "y", "y", "y", "y"]);
}

#[test]
fn rows_interleaved_with_another_groups_are_completed_at_their_own_places() {
    let b = batch(vec![
        ("g", strs(&["x", "y", "x", "y", "y"])),
        ("t", ints(&[1, 1, 3, 2, 3])),
        ("v", f64s(&[Some(10.0), Some(1.0), Some(30.0), Some(2.0), Some(3.0)])),
    ]);
    // x lacks 2: it follows x's row at key 1 (batch row 0), before y's row
    assert_eq!(values(&complete(&b), "v"), vec![10.0, -1.0, 1.0, 30.0, 2.0, 3.0]);
}

#[test]
fn keys_that_land_together_come_out_in_key_order_whatever_order_keyvals_were_written() {
    let b = batch(vec![("t", ints(&[5])), ("v", f64s(&[Some(50.0)]))]);
    for keyvals in [vec![json!(1), json!(2), json!(8)], vec![json!(8), json!(2), json!(1)]] {
        let spec = with_keyvals(keyed(with_value(base("v"), json!(-1)), "t", &[]), Keyvals::List(keyvals));
        let out = ok(&spec, &b);
        assert_eq!(values(&out, "v"), vec![-1.0, -1.0, 50.0, -1.0]);
        assert_eq!(col::<Int64Array>(&out, "t").values().to_vec(), vec![1, 2, 5, 8]);
    }
}

#[test]
fn imputed_rows_that_are_already_last_in_the_last_group_equal_the_old_appended_rows() {
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "y", "y"])),
        ("t", ints(&[1, 2, 3, 1, 2])),
        ("v", f64s(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)])),
    ]);
    let out = complete(&b);
    assert_eq!(values(&out, "v"), vec![1.0, 2.0, 3.0, 4.0, 5.0, -1.0]);
    assert_eq!(col::<Int64Array>(&out, "t").values().to_vec(), vec![1, 2, 3, 1, 2, 3]);
    assert_eq!(labels(&out, "g"), vec!["x", "x", "x", "y", "y", "y"]);
}

#[test]
fn a_key_holding_nan_is_a_key_of_its_own_completed_into_every_group_and_sorts_last() {
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "y", "y"])),
        ("t", f64s(&[Some(1.0), Some(2.0), Some(f64::NAN), Some(1.0), Some(2.0)])),
        ("v", f64s(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)])),
    ]);
    let out = complete(&b);
    assert_eq!(values(&out, "v"), vec![1.0, 2.0, 3.0, 4.0, 5.0, -1.0]);
    assert!(values(&out, "t")[5].is_nan());
}

#[test]
fn an_infinite_key_is_a_key_of_its_own_and_sorts_by_value() {
    for (infinity, imputed_at) in [(f64::INFINITY, 2), (f64::NEG_INFINITY, 0)] {
        let b = batch(vec![
            ("g", strs(&["x", "x", "y", "y", "y"])),
            ("t", f64s(&[Some(1.0), Some(2.0), Some(1.0), Some(2.0), Some(infinity)])),
            ("v", f64s(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)])),
        ]);
        let out = complete(&b);
        let v = values(&out, "v");
        assert_eq!(v.len(), 6);
        assert_eq!(v.iter().position(|&x| x == -1.0), Some(imputed_at), "{infinity}: {v:?}");
        assert_eq!(values(&out, "t")[imputed_at], infinity);
    }
}

#[test]
fn a_null_key_is_no_key_and_is_completed_into_no_group() {
    let b = batch(vec![
        ("g", strs(&["y", "y", "x", "x"])),
        ("t", f64s(&[Some(1.0), Some(2.0), None, Some(1.0)])),
        ("v", f64s(&[Some(3.0), Some(4.0), Some(2.0), Some(1.0)])),
    ]);
    let out = complete(&b);
    // x keeps its null-key row and gains key 2 after its last row; y gains nothing
    assert_eq!(values(&out, "v"), vec![3.0, 4.0, 2.0, 1.0, -1.0]);
    assert_eq!(col::<Float64Array>(&out, "t").null_count(), 1);
}

#[test]
fn a_group_of_one_row_and_a_batch_with_no_row_are_completed_around_what_they_hold() {
    let b = batch(vec![
        ("g", strs(&["x", "y", "y", "y"])),
        ("t", ints(&[2, 1, 2, 3])),
        ("v", f64s(&[Some(20.0), Some(1.0), Some(2.0), Some(3.0)])),
    ]);
    assert_eq!(values(&complete(&b), "v"), vec![-1.0, 20.0, -1.0, 1.0, 2.0, 3.0]);
    let empty = batch(vec![("g", strs(&[])), ("t", ints(&[])), ("v", f64s(&[]))]);
    let out = complete(&empty);
    assert_eq!(out.num_rows(), 0);
    assert_eq!(out.schema().field(2).data_type(), &DataType::Float64);
}

#[test]
fn a_key_with_no_groupby_is_one_group_in_key_order() {
    let b = batch(vec![("t", ints(&[1, 3, 4])), ("v", f64s(&[Some(10.0), Some(30.0), Some(40.0)]))]);
    let spec = with_keyvals(keyed(with_value(base("v"), json!(-1)), "t", &[]), Keyvals::List(vec![json!(2)]));
    assert_eq!(values(&ok(&spec, &b), "v"), vec![10.0, -1.0, 30.0, 40.0]);
    let mut spec = with_value(base("v"), json!(-1));
    spec.key = Some("t".into());
    assert_eq!(ok(&spec, &b).num_rows(), 3, "key with no groupby: the observed keys are all present");
}

#[test]
fn an_imputed_row_copies_its_series_columns_and_leaves_every_other_column_null() {
    let zoned = DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()));
    let day: ArrayRef =
        Arc::new(TimestampMillisecondArray::from(vec![5, 5, 5]).with_timezone("UTC"));
    let b = batch(vec![
        ("h", strs(&["p", "p", "q"])),
        ("day", day),
        ("t", ints(&[1, 3, 1])),
        ("note", strs(&["a", "b", "c"])),
        ("v", f64s(&[Some(1.0), Some(3.0), Some(9.0)])),
    ]);
    let mut spec = with_value(base("v"), json!(-1));
    spec.key = Some("t".into());
    spec.groupby = Some(vec!["h".into(), "day".into()]);
    let out = ok(&spec, &b);
    // p lacks nothing (keys 1, 3 and the universe {1, 3}); q lacks 3
    assert_eq!(out.num_rows(), 4);
    assert_eq!(labels(&out, "h"), vec!["p", "p", "q", "q"]);
    assert_eq!(out.schema().field(1).data_type(), &zoned, "a series column keeps its unit and zone");
    assert_eq!(col::<TimestampMillisecondArray>(&out, "day").values().to_vec(), vec![5, 5, 5, 5]);
    let note = col::<StringArray>(&out, "note");
    assert_eq!((note.is_null(0), note.is_null(1), note.is_null(2), note.is_null(3)), (false, false, false, true));
    let schema = out.schema();
    let nullable: Vec<bool> = schema.fields().iter().map(|f| f.is_nullable()).collect();
    // every column was nullable going in; the value column is filled, so it is not
    assert_eq!(nullable, vec![true, true, true, true, false]);
}

#[test]
fn the_nullability_of_a_series_and_key_column_survives_and_another_column_widens() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Utf8, false),
        Field::new("t", DataType::Int64, false),
        Field::new("other", DataType::Utf8, false),
        Field::new("v", DataType::Float64, true),
    ]));
    let b = RecordBatch::try_new(
        schema,
        vec![
            strs(&["a", "a", "b"]),
            ints(&[1, 2, 1]),
            strs(&["p", "q", "r"]),
            f64s(&[Some(1.0), Some(2.0), Some(3.0)]),
        ],
    )
    .unwrap();
    let out = complete(&b);
    let out_schema = out.schema();
    let flags: Vec<bool> = out_schema.fields().iter().map(|f| f.is_nullable()).collect();
    assert_eq!(flags, vec![false, false, true, false]);
}

// ─── keyvals ───────────────────────────────────────────────────────────────

#[test]
fn keyvals_add_to_the_observed_keys_instead_of_replacing_them() {
    // keyvals names 1 and 2 only; key 3 is observed in x and still completes y
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "y", "y"])),
        ("t", ints(&[1, 2, 3, 1, 2])),
        ("v", f64s(&[Some(10.0), Some(20.0), Some(30.0), Some(1.0), Some(2.0)])),
    ]);
    let spec = with_keyvals(keyed(with_value(base("v"), json!(-1)), "t", &["g"]), Keyvals::List(vec![json!(1), json!(2)]));
    assert_eq!(values(&ok(&spec, &b), "v"), vec![10.0, 20.0, 30.0, 1.0, 2.0, -1.0]);
}

#[test]
fn a_keyvals_sequence_adds_its_keys_and_a_text_or_boolean_key_refuses_one() {
    let b = batch(vec![("g", strs(&["x", "x"])), ("t", ints(&[1, 3])), ("v", f64s(&[Some(10.0), Some(30.0)]))]);
    let sequence = |start, stop, step| {
        Keyvals::Sequence(KeySequence { start, stop, step })
    };
    for (seq, expected) in [
        (sequence(Some(0.0), 5.0, None), vec![-1.0, 10.0, -1.0, 30.0, -1.0]),
        (sequence(Some(5.0), 0.0, None), vec![10.0, -1.0, 30.0, -1.0, -1.0]),
        (sequence(None, 3.0, None), vec![-1.0, 10.0, -1.0, 30.0]),
        (sequence(Some(5.0), 0.0, Some(-2.0)), vec![10.0, 30.0, -1.0]),
    ] {
        let spec = with_keyvals(keyed(with_value(base("v"), json!(-1)), "t", &["g"]), seq);
        assert_eq!(values(&ok(&spec, &b), "v"), expected);
    }
    for (key, holds) in [(strs(&["a", "b"]), "text"), (Arc::new(BooleanArray::from(vec![true, false])) as ArrayRef, "booleans")] {
        let b = batch(vec![("g", strs(&["x", "x"])), ("k", key), ("v", f64s(&[Some(1.0), Some(2.0)]))]);
        let spec = with_keyvals(keyed(base("v"), "k", &["g"]), sequence(None, 3.0, None));
        assert_eq!(
            refusal(&spec, &b),
            format!(
                "ValueError: data_impute: keyvals={{'start': ...}} makes numbers, but the key 'k' \
                 holds {holds}; list the keys instead"
            )
        );
    }
    let zero = with_keyvals(keyed(base("v"), "t", &["g"]), sequence(Some(0.0), 5.0, Some(0.0)));
    assert_eq!(
        refusal(&zero, &b),
        "ValueError: data_impute: keyvals={'start': ..., 'step': 0} never reaches stop; use a non-zero step"
    );
    let long = with_keyvals(keyed(base("v"), "t", &["g"]), sequence(None, 1e9, None));
    assert!(refusal(&long, &b).contains("would make 1000000000 keys"));
}

#[test]
fn a_tagged_keyval_is_an_instant_on_a_temporal_key_at_every_unit_and_the_dtype_is_kept() {
    let day = 86_400_000_i64;
    for (unit, scale) in [
        (TimeUnit::Millisecond, 1_i64),
        (TimeUnit::Microsecond, 1_000),
        (TimeUnit::Nanosecond, 1_000_000),
    ] {
        let dtype = DataType::Timestamp(unit, Some("Europe/Paris".into()));
        let raw = |ms: i64| ms * scale;
        let key: ArrayRef = match unit {
            TimeUnit::Millisecond => Arc::new(
                TimestampMillisecondArray::from(vec![raw(0), raw(2 * day)]).with_timezone("Europe/Paris"),
            ),
            TimeUnit::Nanosecond => Arc::new(
                TimestampNanosecondArray::from(vec![raw(0), raw(2 * day)]).with_timezone("Europe/Paris"),
            ),
            _ => Arc::new(
                arrow::array::TimestampMicrosecondArray::from(vec![raw(0), raw(2 * day)])
                    .with_timezone("Europe/Paris"),
            ),
        };
        let b = batch(vec![("g", strs(&["x", "x"])), ("t", key), ("v", f64s(&[Some(10.0), Some(30.0)]))]);
        for entry in [json!({"epoch_ms": day}), json!(day)] {
            let spec = with_keyvals(keyed(with_value(base("v"), json!(-1)), "t", &["g"]), Keyvals::List(vec![entry]));
            let out = ok(&spec, &b);
            assert_eq!(values(&out, "v"), vec![10.0, -1.0, 30.0], "{unit:?}");
            assert_eq!(out.schema().field(1).data_type(), &dtype);
        }
    }
}

#[test]
fn a_tagged_keyval_on_a_key_that_is_not_temporal_is_refused_as_a_date() {
    let b = batch(vec![("g", strs(&["x"])), ("t", ints(&[1])), ("v", f64s(&[Some(1.0)]))]);
    let spec = with_keyvals(
        keyed(base("v"), "t", &["g"]),
        Keyvals::List(vec![json!({"epoch_ms": 1_704_067_200_000_i64})]),
    );
    assert_eq!(
        refusal(&spec, &b),
        "ValueError: data_impute: keyvals value 2024-01-01T00:00:00 (a date or datetime) is not \
         compatible with key column 't' dtype Int64"
    );
}

// ─── frame ─────────────────────────────────────────────────────────────────

/// v = 10, null, 40, 100, null at keys 1 to 5 in one group.
fn framed(key_order: &[usize]) -> RecordBatch {
    let t = [1_i64, 2, 3, 4, 5];
    let v = [Some(10.0), None, Some(40.0), Some(100.0), None];
    batch(vec![
        ("g", strs(&vec!["x"; 5])),
        ("t", ints(&key_order.iter().map(|&i| t[i]).collect::<Vec<_>>())),
        ("v", f64s(&key_order.iter().map(|&i| v[i]).collect::<Vec<_>>())),
    ])
}

fn windowed(method: &str, frame: (Option<i64>, Option<i64>)) -> ImputeSpec {
    keyed(with_frame(base("v"), method, frame), "t", &["g"])
}

#[test]
fn each_statistic_reads_the_window_two_rows_either_side_in_key_order() {
    let in_order = [0, 1, 2, 3, 4];
    for (method, expected) in [
        ("mean", vec![10.0, 50.0, 40.0, 100.0, 70.0]),
        ("median", vec![10.0, 40.0, 40.0, 100.0, 70.0]),
        ("min", vec![10.0, 10.0, 40.0, 100.0, 40.0]),
        ("max", vec![10.0, 100.0, 40.0, 100.0, 100.0]),
    ] {
        let out = ok(&windowed(method, (Some(-2), Some(2))), &framed(&in_order));
        assert_eq!(values(&out, "v"), expected, "{method}");
    }
}

#[test]
fn a_frame_open_at_one_end_reads_to_the_groups_edge() {
    let b = framed(&[0, 1, 2, 3, 4]);
    for (frame, expected) in [
        ((None, Some(0)), vec![10.0, 10.0, 40.0, 100.0, 50.0]),
        ((Some(-1), Some(1)), vec![10.0, 25.0, 40.0, 100.0, 100.0]),
        ((Some(0), None), vec![10.0, 70.0, 40.0, 100.0, 50.0]),
    ] {
        let out = ok(&windowed("mean", frame), &b);
        assert_eq!(values(&out, "v"), expected, "{frame:?}");
    }
}

#[test]
fn a_frame_open_on_both_sides_is_the_groups_statistic_and_warns_nothing() {
    let b = framed(&[0, 1, 2, 3, 4]);
    let (out, warnings) = run(&windowed("mean", (None, None)), &b);
    assert_eq!(values(&out.unwrap(), "v"), vec![10.0, 50.0, 40.0, 100.0, 50.0]);
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn a_frame_open_on_both_sides_sums_in_row_order_exactly_as_no_frame_does() {
    // 1e16 + -1e16 + 1.0 is 1.0 summed in row order and 0.0 summed in key order (keys 2, 3, 1)
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "x"])),
        ("t", ints(&[2, 3, 1, 4])),
        ("v", f64s(&[Some(1e16), Some(-1e16), Some(1.0), None])),
    ]);
    let none = ok(&keyed(with_method(base("v"), "mean"), "t", &["g"]), &b);
    let open = ok(&windowed("mean", (None, None)), &b);
    assert_eq!(values(&none, "v")[3], 1.0 / 3.0);
    assert_eq!(values(&open, "v"), values(&none, "v"));
}

#[test]
fn a_frame_counts_in_key_order_not_row_order() {
    // rows written in the order keys 4, 1, 5, 3, 2
    let b = framed(&[3, 0, 4, 2, 1]);
    let out = ok(&windowed("mean", (Some(-2), Some(2))), &b);
    assert_eq!(col::<Int64Array>(&out, "t").values().to_vec(), vec![4, 1, 5, 3, 2], "no row moved");
    let by_key: Vec<(i64, f64)> = values(&out, "v")
        .into_iter()
        .zip(col::<Int64Array>(&out, "t").values().to_vec())
        .map(|(v, t)| (t, v))
        .collect();
    let mut sorted = by_key;
    sorted.sort_by_key(|(t, _)| *t);
    assert_eq!(sorted.iter().map(|(_, v)| *v).collect::<Vec<_>>(), vec![10.0, 50.0, 40.0, 100.0, 70.0]);
}

#[test]
fn an_imputed_row_counts_in_a_frame_as_a_row_holding_no_value() {
    // key 3 is added by keyvals and key 4 holds null: key 4 sees keys 3..5, where 3 holds nothing
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "x"])),
        ("t", ints(&[1, 2, 4, 5])),
        ("v", f64s(&[Some(10.0), Some(20.0), None, Some(50.0)])),
    ]);
    let spec = with_keyvals(windowed("mean", (Some(-1), Some(1))), Keyvals::List(vec![json!(3)]));
    let out = ok(&spec, &b);
    // key 3 sees keys 2, 3, 4: 20 only; key 4 sees keys 3, 4, 5: 50 only
    assert_eq!(values(&out, "v"), vec![10.0, 20.0, 20.0, 50.0, 50.0]);
    assert_eq!(col::<Int64Array>(&out, "t").values().to_vec(), vec![1, 2, 3, 4, 5]);
}

#[test]
fn a_frame_counts_each_group_apart() {
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "y", "y", "y"])),
        ("t", ints(&[1, 2, 3, 1, 2, 3])),
        ("v", f64s(&[Some(10.0), None, Some(30.0), Some(100.0), None, Some(300.0)])),
    ]);
    let out = ok(&windowed("mean", (Some(-1), Some(1))), &b);
    assert_eq!(values(&out, "v"), vec![10.0, 20.0, 30.0, 100.0, 200.0, 300.0]);
}

#[test]
fn a_frame_with_no_key_counts_in_the_groups_row_order() {
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "x"])),
        ("v", f64s(&[Some(10.0), None, Some(30.0), None])),
    ]);
    let mut spec = with_frame(base("v"), "mean", (Some(-1), Some(0)));
    spec.groupby = Some(vec!["g".into()]);
    assert_eq!(values(&ok(&spec, &b), "v"), vec![10.0, 10.0, 30.0, 30.0]);
}

#[test]
fn a_window_that_holds_no_usable_value_takes_the_groups_statistic_with_a_notice() {
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "x"])),
        ("t", ints(&[1, 2, 3, 4])),
        ("v", f64s(&[Some(10.0), None, None, Some(40.0)])),
    ]);
    let (out, warnings) = run(&windowed("mean", (Some(-1), Some(0))), &b);
    assert_eq!(values(&out.unwrap(), "v"), vec![10.0, 10.0, 25.0, 40.0]);
    assert_eq!(
        warnings,
        vec![
            "Impute: group g=x has 1 row(s) with no usable (non-null, finite) value within the \
             frame; the group's whole mean was used instead"
        ]
    );
}

#[test]
fn a_window_notice_counts_every_fill_that_fell_back_and_names_the_statistic() {
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "x", "x"])),
        ("t", ints(&[1, 2, 3, 4, 5])),
        ("v", f64s(&[Some(2.0), None, None, None, Some(8.0)])),
    ]);
    let (out, warnings) = run(&windowed("max", (Some(0), Some(0))), &b);
    // each window is the cell itself, which holds nothing: all three fall back to the group's max
    assert_eq!(values(&out.unwrap(), "v"), vec![2.0, 8.0, 8.0, 8.0, 8.0]);
    assert_eq!(
        warnings,
        vec![
            "Impute: group g=x has 3 row(s) with no usable (non-null, finite) value within the \
             frame; the group's whole max was used instead"
        ]
    );
}

#[test]
fn an_ungrouped_window_that_holds_no_usable_value_names_the_whole_column() {
    let b = batch(vec![("t", ints(&[1, 2, 3])), ("v", f64s(&[Some(2.0), None, Some(8.0)]))]);
    let mut spec = with_frame(base("v"), "mean", (Some(0), Some(0)));
    spec.key = Some("t".into());
    let (out, warnings) = run(&spec, &b);
    assert_eq!(values(&out.unwrap(), "v"), vec![2.0, 5.0, 8.0]);
    assert_eq!(
        warnings,
        vec![
            "Impute: 1 row(s) have no usable (non-null, finite) value within the frame; the whole \
             column's mean was used instead"
        ]
    );
}

#[test]
fn a_group_with_no_usable_value_takes_the_whole_columns_statistic_with_its_own_notice_only() {
    let b = batch(vec![
        ("g", strs(&["x", "x", "y", "y"])),
        ("t", ints(&[1, 2, 1, 2])),
        ("v", f64s(&[None, None, Some(10.0), Some(30.0)])),
    ]);
    let (out, warnings) = run(&windowed("mean", (Some(-1), Some(1))), &b);
    assert_eq!(values(&out.unwrap(), "v"), vec![20.0, 20.0, 10.0, 30.0]);
    assert_eq!(
        warnings,
        vec!["Impute: group g=x has no usable (non-null, finite) values; the whole column's statistic was used instead"]
    );
}

#[test]
fn a_column_with_no_usable_value_takes_the_degenerate_default_with_both_existing_notices() {
    let b = batch(vec![("g", strs(&["x", "x"])), ("t", ints(&[1, 2])), ("v", f64s(&[None, None]))]);
    let (out, warnings) = run(&windowed("mean", (Some(-1), Some(1))), &b);
    assert_eq!(values(&out.unwrap(), "v"), vec![0.0, 0.0]);
    assert_eq!(
        warnings,
        vec![
            "Impute: the column has no usable (non-null, finite) values at all; used the mean method's degenerate default instead",
            "Impute: group g=x has no usable (non-null, finite) values; the whole column's statistic was used instead",
        ]
    );
}

#[test]
fn a_window_skips_an_infinite_cell_and_leaves_it_in_place() {
    let b = batch(vec![
        ("g", strs(&["x", "x", "x"])),
        ("t", ints(&[1, 2, 3])),
        ("v", f64s(&[Some(f64::INFINITY), None, Some(6.0)])),
    ]);
    let out = ok(&windowed("mean", (Some(-1), Some(1))), &b);
    assert_eq!(values(&out, "v"), vec![f64::INFINITY, 6.0, 6.0]);
}

#[test]
fn a_window_over_nothing_in_front_of_the_row_falls_back_not_wraps() {
    // preceding and following both before the row: the window is rows 1 back and no further
    let b = batch(vec![("t", ints(&[1, 2])), ("v", f64s(&[None, Some(4.0)]))]);
    let mut spec = with_frame(base("v"), "mean", (Some(-2), Some(-1)));
    spec.key = Some("t".into());
    let (out, warnings) = run(&spec, &b);
    assert_eq!(values(&out.unwrap(), "v"), vec![4.0, 4.0]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
}

// ─── rows in no series, warnings, doors ────────────────────────────────────

#[test]
fn a_row_in_no_series_takes_the_whole_column_statistic_and_is_never_completed() {
    let b = batch(vec![
        ("g", Arc::new(StringArray::from(vec![Some("a"), Some("a"), None])) as ArrayRef),
        ("t", ints(&[1, 2, 3])),
        ("v", f64s(&[Some(10.0), Some(20.0), None])),
    ]);
    let (out, warnings) = run(&keyed(with_method(base("v"), "mean"), "t", &["g"]), &b);
    let out = out.unwrap();
    // group a lacks key 3 and gains it; the null-keyed row is no group and gains nothing
    assert_eq!(out.num_rows(), 4);
    assert_eq!(values(&out, "v"), vec![10.0, 20.0, 15.0, 15.0]);
    assert_eq!(
        warnings,
        vec!["Impute: 1 row(s) have a null groupby key; the whole column's statistic was used instead"]
    );
}

#[test]
fn complete_series_speaks_in_the_voice_of_the_door_that_called_it() {
    pyo3::Python::initialize();
    let b = batch(vec![
        ("h", strs(&["p", "p"])),
        ("t", ints(&[1, 2])),
        ("v", f64s(&[Some(1.0), None])),
    ]);
    let series = Series { of_row: vec![Some(0), Some(0)], labels: vec![Some("h=p".into())], leave_unseriesed: false };
    let columns = ImputeColumns { value: "v", key: every("t"), series: &["h".to_string()] };
    let ctx = TransformContext::default();
    let call = |options: ImputeOptions<'_>, columns: &ImputeColumns<'_>| {
        complete_series(&b, columns, &series, &options, "impute", &ctx).unwrap_err().to_string()
    };
    let plain = ImputeOptions { method: "value", value: None, keyvals: None, frame: None };
    let one = json!(0);
    assert_eq!(
        call(ImputeOptions { method: "mean", value: Some(&one), ..plain }, &columns),
        "impute: value= applies to method='value', not 'mean'; drop value= or set method='value'"
    );
    assert_eq!(
        call(ImputeOptions { method: "bogus", ..plain }, &columns),
        "impute: unknown method 'bogus'; expected value|mean|median|min|max"
    );
    assert_eq!(
        call(ImputeOptions { frame: Some((None, Some(1))), ..plain }, &columns),
        "impute: frame= applies to mean, median, min and max; method 'value' fills a constant"
    );
    let keyvals = Keyvals::List(vec![json!(1)]);
    assert_eq!(
        call(ImputeOptions { keyvals: Some(&keyvals), ..plain }, &ImputeColumns { key: None, ..columns }),
        "impute: keyvals requires key=; the key universe is meaningless without a key column"
    );
    assert_eq!(
        call(plain, &ImputeColumns { key: every("nope"), ..columns }),
        "impute: key column 'nope' not found"
    );
    assert_eq!(
        call(plain, &ImputeColumns { key: every("v"), ..columns }),
        "impute: key column 'v' cannot be the same as the imputed field 'v'"
    );
    assert_eq!(
        call(plain, &ImputeColumns { value: "nope", ..columns }),
        "impute: column 'nope' not found"
    );
    let text = json!("none");
    assert_eq!(
        call(ImputeOptions { value: Some(&text), ..plain }, &columns),
        "impute: column 'v' holds numbers, so value= must be a number, but 'none' is text"
    );
    let unsupported = batch(vec![
        ("t", Arc::new(arrow::array::Date32Array::from(vec![1, 2])) as ArrayRef),
        ("v", f64s(&[Some(1.0), None])),
    ]);
    let err = complete_series(
        &unsupported,
        &ImputeColumns { value: "v", key: every("t"), series: &[] },
        &Series { of_row: vec![Some(0), Some(0)], labels: vec![None], leave_unseriesed: false },
        &plain,
        "impute",
        &ctx,
    )
    .unwrap_err()
    .to_string();
    assert_eq!(
        err,
        format!("impute: key column 't' has unsupported dtype Date32; supported: {GROUPBY_SUPPORTED_DTYPES}")
    );
}

// ─── the spec on the wire, and the Python class ────────────────────────────

fn parse(json: Value) -> Result<ImputeSpec, serde_json::Error> {
    serde_json::from_value(json)
}

#[test]
fn a_saved_spec_with_a_numeric_value_still_reads_and_writes_the_same_json() {
    let saved = json!({"field": "v", "method": "value", "value": 3.0, "groupby": ["g"], "key": "t", "keyvals": [1, 2]});
    let spec = parse(saved.clone()).unwrap();
    assert_eq!(spec.value, Some(json!(3.0)));
    assert_eq!(spec.keyvals, Some(Keyvals::List(vec![json!(1), json!(2)])));
    assert_eq!(spec.frame, None);
    assert_eq!(serde_json::to_value(&spec).unwrap(), saved);
}

#[test]
fn every_value_kind_and_both_keyvals_forms_and_a_frame_read_from_the_wire() {
    for value in [json!(1), json!("none"), json!(true), json!({"epoch_ms": 1_704_067_200_000_i64})] {
        let spec = parse(json!({"field": "v", "value": value})).unwrap();
        assert_eq!(spec.value, Some(value));
    }
    let spec = parse(json!({"field": "v", "key": "t", "keyvals": {"start": 1, "stop": 5, "step": 2}})).unwrap();
    assert_eq!(
        spec.keyvals,
        Some(Keyvals::Sequence(KeySequence { start: Some(1.0), stop: 5.0, step: Some(2.0) }))
    );
    let spec = parse(json!({"field": "v", "key": "t", "keyvals": {"stop": 5}})).unwrap();
    assert_eq!(spec.keyvals, Some(Keyvals::Sequence(KeySequence { start: None, stop: 5.0, step: None })));
    assert_eq!(
        serde_json::to_value(&spec).unwrap(),
        json!({"field": "v", "method": "value", "key": "t", "keyvals": {"stop": 5.0}})
    );
    for (frame, expected) in [
        (json!([-1, 1]), (Some(-1), Some(1))),
        (json!([null, 0]), (None, Some(0))),
        (json!([0, null]), (Some(0), None)),
    ] {
        let spec = parse(json!({"field": "v", "method": "mean", "frame": frame.clone()})).unwrap();
        assert_eq!(spec.frame, Some(expected));
        assert_eq!(serde_json::to_value(&spec).unwrap()["frame"], frame);
    }
}

#[test]
fn a_spec_with_an_option_the_engine_does_not_read_is_refused() {
    assert!(parse(json!({"field": "v", "bogus": 1})).is_err());
    assert!(parse(json!({"field": "v", "key": "t", "keyvals": {"stop": 5, "by": 2}})).is_err());
    assert!(parse(json!({"field": "v", "key": "t", "keyvals": {"start": 5}})).is_err());
    assert!(parse(json!({"field": "v", "method": "mean", "frame": [1]})).is_err());
}

#[test]
fn the_python_class_keeps_its_signature_and_repr_and_refuses_a_value_a_float_cannot_hold() {
    pyo3::Python::initialize();
    let made = PyImpute::new("v".into(), "mean", Some(1.5), Some("out".into())).unwrap();
    assert_eq!(made.__repr__(), "Impute(field='v', method='mean', value=Some(1.5), name=Some(\"out\"))");
    let none = PyImpute::new("v".into(), "value", None, None).unwrap();
    assert_eq!(none.__repr__(), "Impute(field='v', method='value', value=None, name=None)");
    let err = PyImpute::new("v".into(), "value", Some(f64::NAN), None).unwrap_err().to_string();
    assert_eq!(err, "ValueError: Impute: value must be a finite number");
}

#[test]
fn a_series_column_the_batch_lacks_and_a_series_map_of_the_wrong_length_are_refused() {
    pyo3::Python::initialize();
    let b = batch(vec![("t", ints(&[1, 3])), ("v", f64s(&[Some(1.0), Some(3.0)]))]);
    let ctx = TransformContext::default();
    let plain = ImputeOptions { method: "value", value: None, keyvals: None, frame: None };
    let one_series = Series { of_row: vec![Some(0), Some(0)], labels: vec![None], leave_unseriesed: false };
    let missing_column = ImputeColumns { value: "v", key: every("t"), series: &["nope".to_string()] };
    // key 2 is held by nobody, so the batch must be completed with it
    let keyvals = Keyvals::List(vec![json!(2)]);
    let options = ImputeOptions { keyvals: Some(&keyvals), ..plain };
    let err = complete_series(&b, &missing_column, &one_series, &options, "impute", &ctx)
        .unwrap_err()
        .to_string();
    assert_eq!(err, "impute: groupby column 'nope' not found");

    let columns = ImputeColumns { value: "v", key: every("t"), series: &[] };
    let short = Series { of_row: vec![Some(0)], labels: vec![None], leave_unseriesed: false };
    let err = complete_series(&b, &columns, &short, &plain, "impute", &ctx).unwrap_err().to_string();
    assert_eq!(err, "impute: internal error: 1 series assignments for 2 rows");
}

// ─── the interface the encoding door calls ─────────────────────────────────

fn series_of(ids: &[Option<usize>], n_series: usize) -> Series {
    Series { of_row: ids.to_vec(), labels: vec![None; n_series], leave_unseriesed: false }
}

fn plain() -> ImputeOptions<'static> {
    ImputeOptions { method: "value", value: None, keyvals: None, frame: None }
}

/// A completion with the value `-1`, through `complete_series` directly under the
/// door `impute`.
fn direct(
    b: &RecordBatch,
    key: Option<ImputeKey<'_>>,
    series_columns: &[String],
    series: &Series,
    options: ImputeOptions<'_>,
) -> Result<RecordBatch, ImputeRefusal> {
    let minus_one = json!(-1);
    let options = ImputeOptions { value: Some(&minus_one), ..options };
    let columns = ImputeColumns { value: "v", key, series: series_columns };
    complete_series(b, &columns, series, &options, "impute", &TransformContext::default())
}

fn placed<'a>(field: &'a str, companions: &'a [String]) -> Option<ImputeKey<'a>> {
    Some(ImputeKey { field, observed: ObservedKeys::Placed, companions })
}

#[test]
fn a_placed_universe_leaves_a_nan_or_infinite_key_out_and_fills_its_row() {
    let group = ["g".to_string()];
    let float32 = |values: [f32; 4]| -> ArrayRef { Arc::new(Float32Array::from(values.to_vec())) };
    for (name, key) in [
        ("NaN", f64s(&[Some(1.0), Some(2.0), Some(1.0), Some(f64::NAN)])),
        ("+inf", f64s(&[Some(1.0), Some(2.0), Some(1.0), Some(f64::INFINITY)])),
        ("-inf", f64s(&[Some(1.0), Some(2.0), Some(1.0), Some(f64::NEG_INFINITY)])),
        ("f32 NaN", float32([1.0, 2.0, 1.0, f32::NAN])),
        ("f32 +inf", float32([1.0, 2.0, 1.0, f32::INFINITY])),
    ] {
        let b = batch(vec![
            ("g", strs(&["x", "x", "y", "y"])),
            ("t", key),
            ("v", f64s(&[Some(10.0), Some(20.0), Some(1.0), None])),
        ]);
        let out = direct(&b, placed("t", &[]), &group, &series_of(&[Some(0), Some(0), Some(1), Some(1)], 2), plain())
            .unwrap();
        // x gains nothing (the bad key adds none); y gains 2 right after its key-1 row, and its
        // row with the bad key is kept, its null value filled, and never a slot anchor
        assert_eq!(values(&out, "v"), vec![10.0, 20.0, 1.0, -1.0, -1.0], "{name}");
        assert_eq!(labels(&out, "g"), vec!["x", "x", "y", "y", "y"], "{name}");
    }
    // the same keys are keys of their own when every observed key joins
    let b = batch(vec![
        ("g", strs(&["x", "x", "y", "y"])),
        ("t", f64s(&[Some(1.0), Some(2.0), Some(1.0), Some(f64::NAN)])),
        ("v", f64s(&[Some(10.0), Some(20.0), Some(1.0), None])),
    ]);
    let every = ImputeKey { field: "t", observed: ObservedKeys::Every, companions: &[] };
    let out = direct(&b, Some(every), &group, &series_of(&[Some(0), Some(0), Some(1), Some(1)], 2), plain()).unwrap();
    assert_eq!(out.num_rows(), 6, "x gains the NaN key and y gains key 2");
}

#[test]
fn a_placed_key_with_a_nan_row_in_a_frame_follows_the_keyed_positions() {
    let b = batch(vec![
        ("t", f64s(&[Some(f64::NAN), Some(1.0), Some(2.0)])),
        ("v", f64s(&[None, Some(10.0), None])),
    ]);
    let frame = ImputeOptions { method: "mean", value: None, keyvals: None, frame: Some((Some(-1), Some(0))) };
    let out = direct_no_value(&b, placed("t", &[]), &series_of(&[Some(0); 3], 1), frame).unwrap();
    // key order: 1 (row 1), 2 (row 2), then the NaN row (row 0) last: row 2's window is rows
    // 1 and 2, the NaN row's window is rows 2 and itself, which hold no usable value
    assert_eq!(values(&out, "v"), vec![10.0, 10.0, 10.0]);
}

fn direct_no_value(
    b: &RecordBatch,
    key: Option<ImputeKey<'_>>,
    series: &Series,
    options: ImputeOptions<'_>,
) -> Result<RecordBatch, ImputeRefusal> {
    let columns = ImputeColumns { value: "v", key, series: &[] };
    complete_series(b, &columns, series, &options, "impute", &TransformContext::default())
}

fn with_companion(frame_t: &[i64], x2: ArrayRef) -> RecordBatch {
    let n = frame_t.len();
    let g: Vec<&str> = (0..n).map(|_| "g").collect();
    batch(vec![
        ("g", strs(&g)),
        ("t", ints(frame_t)),
        ("x2", x2),
        ("v", f64s(&vec![Some(1.0); n])),
    ])
}

#[test]
fn a_companion_column_is_copied_from_the_first_row_holding_the_key() {
    let companions = ["x2".to_string(), "lab".to_string()];
    let b = batch(vec![
        ("g", strs(&["x", "x", "y", "y", "y"])),
        ("t", ints(&[1, 3, 1, 2, 3])),
        ("x2", f64s(&[Some(10.0), Some(30.0), Some(10.0), Some(20.0), Some(30.0)])),
        ("lab", strs(&["a", "c", "a", "b", "c"])),
        ("v", f64s(&[Some(1.0), Some(3.0), Some(1.0), Some(2.0), Some(3.0)])),
    ]);
    let group = ["g".to_string()];
    let series = series_of(&[Some(0), Some(0), Some(1), Some(1), Some(1)], 2);
    let key = Some(ImputeKey { field: "t", observed: ObservedKeys::Placed, companions: &companions });
    let out = direct(&b, key, &group, &series, plain()).unwrap();
    assert_eq!(values(&out, "x2"), vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
    assert_eq!(labels(&out, "lab"), vec!["a", "b", "c", "a", "b", "c"]);
    assert_eq!(col::<Int64Array>(&out, "t").values().to_vec(), vec![1, 2, 3, 1, 2, 3]);
    // a companion is a copy, so it is not widened to nullable
    assert!(out.schema().fields().iter().all(|f| f.is_nullable() || f.name() == "v"));
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Utf8, false),
        Field::new("t", DataType::Int64, false),
        Field::new("x2", DataType::Float64, false),
        Field::new("v", DataType::Float64, true),
    ]));
    let strict = RecordBatch::try_new(
        schema,
        vec![strs(&["x", "x", "y"]), ints(&[1, 2, 1]), f64s(&[Some(1.0), Some(2.0), Some(1.0)]), f64s(&[Some(1.0), Some(2.0), Some(3.0)])],
    )
    .unwrap();
    let key = Some(ImputeKey { field: "t", observed: ObservedKeys::Placed, companions: &["x2".to_string()] });
    let out = direct(&strict, key, &group, &series_of(&[Some(0), Some(0), Some(1)], 2), plain()).unwrap();
    assert!(!out.schema().field(2).is_nullable(), "a copied companion keeps its nullability");
    assert_eq!(values(&out, "x2"), vec![1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn rows_that_share_a_key_and_disagree_on_a_companion_are_refused() {
    let companions = ["x2".to_string()];
    let b = batch(vec![
        ("g", strs(&["x", "y", "z"])),
        ("t", ints(&[2, 2, 5])),
        ("x2", f64s(&[Some(22.0), Some(99.0), Some(5.0)])),
        ("v", f64s(&[Some(1.0), Some(1.0), Some(1.0)])),
    ]);
    let key = Some(ImputeKey { field: "t", observed: ObservedKeys::Placed, companions: &companions });
    let err = direct(&b, key, &["g".to_string()], &series_of(&[Some(0), Some(1), Some(2)], 3), plain()).unwrap_err();
    assert_eq!(
        err,
        ImputeRefusal::CompanionDiffers { door: "impute".into(), key: "2".into(), companion: "x2".into() }
    );
    assert_eq!(
        err.to_string(),
        "impute: rows with key 2 hold different 'x2' values, so an imputed row has no one 'x2' to copy"
    );
    // the first disagreeing key in key order is named: series z holds neither 2 nor 5, so both
    // keys are copied into it
    let b = batch(vec![
        ("g", strs(&["x", "y", "x", "y", "z"])),
        ("t", ints(&[5, 5, 2, 2, 9])),
        ("x2", f64s(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(9.0)])),
        ("v", f64s(&[Some(1.0); 5])),
    ]);
    let err = direct(&b, key, &["g".to_string()], &series_of(&[Some(0), Some(1), Some(0), Some(1), Some(2)], 3), plain()).unwrap_err();
    assert!(matches!(&err, ImputeRefusal::CompanionDiffers { key, .. } if key == "2"), "{err:?}");
    // an instant key is spelled as every warning spells it
    let day: ArrayRef = Arc::new(TimestampMillisecondArray::from(vec![86_400_000, 86_400_000, 172_800_000]));
    let b = batch(vec![
        ("t", day),
        ("x2", f64s(&[Some(1.0), Some(2.0), Some(3.0)])),
        ("v", f64s(&[Some(1.0), Some(1.0), Some(1.0)])),
    ]);
    let err = direct_no_value(&b, key_on("t", &companions), &series_of(&[Some(0), Some(0), Some(1)], 2), plain()).unwrap_err();
    assert!(matches!(&err, ImputeRefusal::CompanionDiffers { key, .. } if key == "1970-01-02T00:00:00"), "{err:?}");
}

fn key_on<'a>(field: &'a str, companions: &'a [String]) -> Option<ImputeKey<'a>> {
    Some(ImputeKey { field, observed: ObservedKeys::Placed, companions })
}

#[test]
fn rows_that_share_a_key_agree_on_a_null_or_nan_companion() {
    let companions = ["x2".to_string()];
    for shared in [None, Some(f64::NAN)] {
        let b = batch(vec![
            ("g", strs(&["x", "y", "y"])),
            ("t", ints(&[2, 3, 3])),
            ("x2", f64s(&[Some(7.0), shared, shared])),
            ("v", f64s(&[Some(1.0), Some(1.0), Some(1.0)])),
        ]);
        let series = series_of(&[Some(0), Some(1), Some(1)], 2);
        let out = direct(&b, key_on("t", &companions), &["g".to_string()], &series, plain()).unwrap();
        // x gains key 3, copying the shared null or NaN; y gains key 2, copying 7
        assert_eq!(out.num_rows(), 5);
        let x2 = col::<Float64Array>(&out, "x2");
        assert_eq!(x2.is_null(1) || x2.value(1).is_nan(), true, "{shared:?}");
        assert_eq!(x2.value(2), 7.0, "{shared:?}");
        assert_eq!(x2.is_null(1), shared.is_none(), "{shared:?}");
    }
}

#[test]
fn a_keyvals_key_no_row_holds_is_refused_when_a_companion_rides_with_the_key() {
    let companions = ["x2".to_string()];
    let b = with_companion(&[1, 2], f64s(&[Some(1.0), Some(2.0)]));
    let keyvals = Keyvals::List(vec![json!(4)]);
    let options = ImputeOptions { keyvals: Some(&keyvals), ..plain() };
    let series = series_of(&[Some(0), Some(0)], 1);
    let err = direct(&b, key_on("t", &companions), &["g".to_string()], &series, options).unwrap_err();
    assert_eq!(
        err,
        ImputeRefusal::CompanionUnheld { door: "impute".into(), key: "4".into(), companion: "x2".into() }
    );
    assert_eq!(
        err.to_string(),
        "impute: keyvals key 4 is held by no row, so an imputed row has no 'x2' to copy"
    );
    // with no companion the same keyvals key completes
    let out = direct(&b, key_on("t", &[]), &["g".to_string()], &series, options).unwrap();
    assert_eq!(out.num_rows(), 3);
    // a keyvals key some row holds has a companion to copy
    let held = Keyvals::List(vec![json!(2)]);
    let held = ImputeOptions { keyvals: Some(&held), ..plain() };
    let one = with_companion(&[1, 3, 2], f64s(&[Some(1.0), Some(3.0), Some(2.0)]));
    let out = direct(&one, key_on("t", &companions), &["g".to_string()], &series_of(&[Some(0), Some(0), None], 1), held).unwrap();
    assert_eq!(out.num_rows(), 4);
    assert_eq!(values(&out, "x2"), vec![1.0, 2.0, 3.0, 2.0]);
}

#[test]
fn a_series_id_beyond_the_labels_is_an_internal_error_not_a_panic() {
    let b = batch(vec![("t", ints(&[1, 2])), ("v", f64s(&[Some(1.0), Some(2.0)]))]);
    let err = direct_no_value(&b, None, &series_of(&[Some(0), Some(2)], 2), plain()).unwrap_err();
    assert_eq!(
        err,
        ImputeRefusal::Refused("impute: internal error: row 1 is in series 2, but there are 2 series".into())
    );
}

#[test]
fn a_series_that_holds_no_row_is_an_internal_error() {
    let b = batch(vec![("t", ints(&[1, 2])), ("v", f64s(&[Some(1.0), None]))]);
    let err = direct_no_value(&b, None, &series_of(&[Some(0), Some(0)], 2), plain()).unwrap_err();
    assert_eq!(err.to_string(), "impute: internal error: series 1 holds no row");
    // a series named by no row is no false notice either
    let ctx = TransformContext::default();
    let columns = ImputeColumns { value: "v", key: None, series: &[] };
    let mean = ImputeOptions { method: "mean", ..plain() };
    let series = Series { of_row: vec![Some(0), Some(0)], labels: vec![Some("g=a".into()), Some("g=b".into())], leave_unseriesed: false };
    assert!(complete_series(&b, &columns, &series, &mean, "impute", &ctx).is_err());
    assert!(ctx.warnings.borrow().is_empty(), "{:?}", ctx.warnings.borrow());
}

#[test]
fn an_internal_overlap_of_a_companion_with_the_value_key_or_a_series_column_is_refused() {
    let b = with_companion(&[1, 2], f64s(&[Some(1.0), Some(2.0)]));
    let series = series_of(&[Some(0), Some(0)], 1);
    for (companion, role) in [("v", "the value"), ("t", "the key"), ("g", "a series column")] {
        let companions = [companion.to_string()];
        let err = direct(&b, key_on("t", &companions), &["g".to_string()], &series, plain()).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("impute: internal error: column '{companion}' cannot be both the key's companion and {role}")
        );
    }
    let missing = ["nope".to_string()];
    let err = direct(&b, key_on("t", &missing), &["g".to_string()], &series, plain()).unwrap_err();
    assert_eq!(err.to_string(), "impute: column 'nope' not found");
}

#[test]
fn imputed_rows_at_one_place_follow_batch_order_whatever_the_series_ids() {
    // x holds key 1 and y holds keys 2 and 3: x gains 2 and 3 after its row, y gains 1 before its first row
    let b = batch(vec![
        ("g", strs(&["x", "y", "y"])),
        ("t", ints(&[1, 2, 3])),
        ("v", f64s(&[Some(1.0), Some(2.0), Some(3.0)])),
    ]);
    let group = ["g".to_string()];
    for ids in [[Some(0), Some(1), Some(1)], [Some(1), Some(0), Some(0)]] {
        let out = direct(&b, Some(ImputeKey { field: "t", observed: ObservedKeys::Every, companions: &[] }), &group, &series_of(&ids, 2), plain()).unwrap();
        assert_eq!(labels(&out, "g"), vec!["x", "x", "x", "y", "y", "y"], "{ids:?}");
        assert_eq!(col::<Int64Array>(&out, "t").values().to_vec(), vec![1, 2, 3, 1, 2, 3], "{ids:?}");
        assert_eq!(values(&out, "v"), vec![1.0, -1.0, -1.0, -1.0, 2.0, 3.0], "{ids:?}");
    }
}

#[test]
fn imputed_rows_of_two_groups_at_one_place_close_the_earlier_group_first() {
    let b = batch(vec![
        ("g", strs(&["x", "y", "y"])),
        ("t", ints(&[1, 2, 3])),
        ("v", f64s(&[Some(1.0), Some(2.0), Some(3.0)])),
    ]);
    let out = complete(&b);
    assert_eq!(labels(&out, "g"), vec!["x", "x", "x", "y", "y", "y"]);
    assert_eq!(col::<Int64Array>(&out, "t").values().to_vec(), vec![1, 2, 3, 1, 2, 3]);
    assert_eq!(values(&out, "v"), vec![1.0, -1.0, -1.0, -1.0, 2.0, 3.0]);
}

#[test]
fn schema_and_field_metadata_survive_a_completion() {
    let with_meta = |name: &str, dt: DataType, key: &str| {
        Field::new(name, dt, true).with_metadata(HashMap::from([("role".to_string(), key.to_string())]))
    };
    let schema = Arc::new(
        Schema::new(vec![
            with_meta("g", DataType::Utf8, "group"),
            with_meta("t", DataType::Int64, "key"),
            with_meta("note", DataType::Utf8, "other"),
            with_meta("v", DataType::Float64, "value"),
        ])
        .with_metadata(HashMap::from([("source".to_string(), "frame".to_string())])),
    );
    let columns: Vec<ArrayRef> = vec![
        strs(&["x", "x", "y"]),
        ints(&[1, 2, 1]),
        strs(&["a", "b", "c"]),
        f64s(&[Some(1.0), Some(2.0), None]),
    ];
    let b = RecordBatch::try_new(schema, columns).unwrap();
    for imputes in [true, false] {
        let batch = if imputes { b.clone() } else { b.slice(0, 2) };
        let spec = keyed(with_value(base("v"), json!(-1)), "t", &["g"]);
        let out = ok(&spec, &batch);
        assert_eq!(out.num_rows(), if imputes { 4 } else { 2 });
        let out_schema = out.schema();
        assert_eq!(out_schema.metadata().get("source").map(String::as_str), Some("frame"));
        for (field, role) in out_schema.fields().iter().zip(["group", "key", "other", "value"]) {
            assert_eq!(field.metadata().get("role").map(String::as_str), Some(role), "{}", field.name());
        }
    }
}

#[test]
fn a_zero_row_batch_with_no_groupby_is_left_empty_with_and_without_a_key() {
    let empty = batch(vec![("t", ints(&[])), ("v", f64s(&[]))]);
    assert_eq!(ok(&base("v"), &empty).num_rows(), 0);
    let mut spec = with_value(base("v"), json!(-1));
    spec.key = Some("t".into());
    assert_eq!(ok(&spec, &empty).num_rows(), 0);
    assert_eq!(ok(&with_method(base("v"), "mean"), &empty).num_rows(), 0);
}

#[test]
fn a_sign_bit_nan_key_sorts_last_as_a_positive_one_does() {
    let negative_nan = f64::from_bits(0xFFF8_0000_0000_0000);
    let b = batch(vec![
        ("g", strs(&["x", "x", "y", "y"])),
        ("t", f64s(&[Some(1.0), Some(2.0), Some(1.0), Some(negative_nan)])),
        ("v", f64s(&[Some(10.0), Some(20.0), Some(1.0), Some(4.0)])),
    ]);
    let out = complete(&b);
    // x lacks the NaN key (after its last row); y lacks 2 (after its key-1 row)
    assert_eq!(values(&out, "v"), vec![10.0, 20.0, -1.0, 1.0, -1.0, 4.0]);
}

#[test]
fn a_window_with_no_usable_value_falls_back_to_its_own_groups_statistic_not_the_columns() {
    let b = batch(vec![
        ("g", strs(&["a", "a", "a", "a", "b", "b", "b"])),
        ("t", ints(&[1, 2, 3, 4, 1, 2, 3])),
        ("v", f64s(&[Some(10.0), None, None, None, Some(100.0), Some(200.0), Some(300.0)])),
    ]);
    let (out, warnings) = run(&windowed("mean", (Some(-1), Some(1))), &b);
    // rows t=3 and t=4 of group a see no usable value: group a's mean, 10, not the column's 152.5
    // (t=2 sees t=1's 10)
    assert_eq!(values(&out.unwrap(), "v")[..4], [10.0, 10.0, 10.0, 10.0]);
    assert_eq!(
        warnings,
        vec![
            "Impute: group g=a has 2 row(s) with no usable (non-null, finite) value within the \
             frame; the group's whole mean was used instead"
        ]
    );
}

#[test]
fn a_series_written_out_of_order_gets_a_key_above_both_its_rows_after_the_last() {
    let b = batch(vec![("t", ints(&[3, 1])), ("v", f64s(&[Some(30.0), Some(10.0)]))]);
    let spec = with_keyvals(keyed(with_value(base("v"), json!(-1)), "t", &[]), Keyvals::List(vec![json!(4)]));
    let out = ok(&spec, &b);
    assert_eq!(values(&out, "v"), vec![30.0, 10.0, -1.0]);
    assert_eq!(col::<Int64Array>(&out, "t").values().to_vec(), vec![3, 1, 4]);
}

#[test]
fn a_frame_offset_past_the_series_reads_to_its_edge_without_falling_back() {
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "x"])),
        ("t", ints(&[1, 2, 3, 4])),
        ("v", f64s(&[Some(1.0), None, Some(3.0), Some(4.0)])),
    ]);
    for (frame, fill) in [
        ((Some(0), Some(i64::MAX)), 3.5),
        ((None, Some(i64::MAX)), 2.666_666_666_666_667),
        ((Some(i64::MIN), Some(0)), 1.0),
        ((Some(i64::MIN), Some(i64::MAX)), 2.666_666_666_666_667),
    ] {
        let (out, warnings) = run(&windowed("mean", frame), &b);
        let got = values(&out.unwrap(), "v")[1];
        assert!((got - fill).abs() < 1e-12, "{frame:?}: {got}");
        assert!(warnings.is_empty(), "{frame:?}: {warnings:?}");
    }
}

#[test]
fn a_null_groupby_row_under_a_constant_fill_is_named_only_when_a_key_completes_the_groups() {
    let b = batch(vec![
        ("g", Arc::new(StringArray::from(vec![Some("a"), Some("a"), None])) as ArrayRef),
        ("t", ints(&[1, 2, 3])),
        ("v", f64s(&[Some(1.0), Some(2.0), None])),
    ]);
    let keyed_spec = keyed(with_value(base("v"), json!(-1)), "t", &["g"]);
    let (out, warnings) = run(&keyed_spec, &b);
    assert_eq!(values(&out.unwrap(), "v"), vec![1.0, 2.0, -1.0, -1.0]);
    assert_eq!(
        warnings,
        vec!["Impute: 1 row(s) have a null groupby key, so they belong to no group and no key is completed for them"]
    );
    let mut no_key = with_value(base("v"), json!(-1));
    no_key.groupby = Some(vec!["g".into()]);
    let (out, warnings) = run(&no_key, &b);
    assert_eq!(values(&out.unwrap(), "v"), vec![1.0, 2.0, -1.0]);
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn a_float32_keyval_beyond_the_finite_range_is_refused_not_completed_at_infinity() {
    let b = batch(vec![
        ("t", Arc::new(Float32Array::from(vec![1.0_f32, 2.0])) as ArrayRef),
        ("v", f64s(&[Some(1.0), Some(2.0)])),
    ]);
    let too_big = Keyvals::List(vec![json!(1e39)]);
    for observed in [ObservedKeys::Placed, ObservedKeys::Every] {
        let options = ImputeOptions { keyvals: Some(&too_big), ..plain() };
        let key = Some(ImputeKey { field: "t", observed, companions: &[] });
        let err = direct_no_value_with(&b, key, options).unwrap_err();
        assert_eq!(
            err.to_string(),
            "impute: keyvals value 1e+39 is not compatible with key column 't' dtype Float32"
        );
    }
}

fn direct_no_value_with(
    b: &RecordBatch,
    key: Option<ImputeKey<'_>>,
    options: ImputeOptions<'_>,
) -> Result<RecordBatch, ImputeRefusal> {
    direct(b, key, &[], &series_of(&[Some(0), Some(0)], 1), options)
}

#[test]
fn a_companion_that_is_no_function_of_the_key_is_refused_only_for_a_key_an_imputed_row_copies() {
    let companions = ["x2".to_string()];
    // both rows hold key 1 with different x2, and the one series holds every key: no row is
    // imputed, so no end is copied and the rows are free to differ
    let b = batch(vec![
        ("g", strs(&["x", "x"])),
        ("t", ints(&[1, 1])),
        ("x2", f64s(&[Some(1.0), Some(2.0)])),
        ("v", f64s(&[Some(1.0), Some(1.0)])),
    ]);
    let out = direct(&b, key_on("t", &companions), &["g".to_string()], &series_of(&[Some(0), Some(0)], 1), plain())
        .unwrap();
    assert_eq!(out.num_rows(), 2);
    assert_eq!(values(&out, "x2"), vec![1.0, 2.0]);
    // a second series that lacks key 1 makes an imputed row for it, so key 1's end must be one value
    let b = batch(vec![
        ("g", strs(&["x", "x", "y"])),
        ("t", ints(&[1, 1, 2])),
        ("x2", f64s(&[Some(1.0), Some(2.0), Some(3.0)])),
        ("v", f64s(&[Some(1.0), Some(1.0), Some(1.0)])),
    ]);
    let err = direct(&b, key_on("t", &companions), &["g".to_string()], &series_of(&[Some(0), Some(0), Some(1)], 2), plain())
        .unwrap_err();
    assert!(matches!(&err, ImputeRefusal::CompanionDiffers { key, .. } if key == "1"), "{err:?}");
    // key 1 has two ends but only key 2 is imputed (series y lacks it), and key 2 has one end
    let b = batch(vec![
        ("g", strs(&["x", "x", "x", "y", "y"])),
        ("t", ints(&[1, 2, 3, 1, 3])),
        ("x2", f64s(&[Some(1.0), Some(2.0), Some(3.0), Some(9.0), Some(3.0)])),
        ("v", f64s(&[Some(1.0); 5])),
    ]);
    let out = direct(&b, key_on("t", &companions), &["g".to_string()], &series_of(&[Some(0), Some(0), Some(0), Some(1), Some(1)], 2), plain())
        .unwrap();
    assert_eq!(values(&out, "x2"), vec![1.0, 2.0, 3.0, 9.0, 2.0, 3.0], "the imputed row copies key 2's one end");
}

#[test]
fn companions_that_differ_only_by_the_sign_of_zero_agree_and_nans_agree() {
    let companions = ["x2".to_string()];
    // series z lacks key 1, so key 1's two rows (series x and y) must agree for the copy
    let run_with = |a: f64, b: f64| {
        let frame = batch(vec![
            ("g", strs(&["x", "y", "z"])),
            ("t", ints(&[1, 1, 2])),
            ("x2", f64s(&[Some(a), Some(b), Some(5.0)])),
            ("v", f64s(&[Some(1.0), Some(1.0), Some(1.0)])),
        ]);
        direct(&frame, key_on("t", &companions), &["g".to_string()], &series_of(&[Some(0), Some(1), Some(2)], 3), plain())
    };
    assert!(run_with(f64::NAN, f64::NAN).is_ok());
    assert!(run_with(0.0, 0.0).is_ok());
    assert!(run_with(0.0, -0.0).is_ok(), "0.0 and -0.0 are one position");
    assert!(run_with(0.0, 1.0).is_err());
    assert!(run_with(f64::NAN, 1.0).is_err());
}
