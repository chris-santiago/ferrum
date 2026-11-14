//! The key universe an impute completes every series over, and its order.
//!
//! The universe is the keys the caller names in `keyvals` (a list, or a
//! `{start, stop, step}` sequence) followed by the keys observed across the
//! whole batch that it does not already hold (Vega's rule: `keyvals` are "used
//! in addition to the key values observed"). Its order is the order a key axis
//! places the values: ascending for a numeric or temporal key (`NaN` last),
//! `false` before `true`, and for a text key the universe's own order,
//! `keyvals` first and then the first appearance of each observed key.

use arrow::array::Array;
use arrow::datatypes::DataType;
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use super::literal::{as_instant, unit_words, AsInstant, Literal};
use crate::transform::context::format_instant;
use super::{KeySequence, Keyvals, ObservedKeys};
use crate::transform::group_key::{groupby_key_at, KeyValue};

/// The most keys a `{start, stop, step}` sequence may make. A series is
/// completed with one row per missing key in every group, so a sequence such
/// as `{'stop': 1e12}` would allocate without limit; it is refused instead,
/// naming the two remedies (list the keys, or a larger step).
pub(super) const MAX_SEQUENCE_KEYS: usize = 1_000_000;

/// An integral JSON float (e.g. `1.0`, produced by `json.dumps` for a Python
/// `float` that happens to be whole-valued) narrowed to `i64`, or `None` if
/// it has a fractional part or doesn't fit in range. Used to let `keyvals`
/// coerce a whole-valued float onto an Int key column without silently
/// truncating a genuinely fractional or out-of-range value.
/// The upper bound is strict (`f < -(i64::MIN as f64)`) to prevent saturation
/// before the range check; `i64::MAX as f64` equals 2^63, one above the actual max.
fn integral_f64_as_i64(f: f64) -> Option<i64> {
    (f.is_finite() && f.fract() == 0.0 && f >= i64::MIN as f64 && f < -(i64::MIN as f64))
        .then_some(f as i64)
}

/// Same as [`integral_f64_as_i64`], narrowed to `u64` (non-negative only).
/// The upper bound is strict (`f < 18446744073709551616.0`, i.e. 2^64) to prevent
/// saturation before the range check; `u64::MAX as f64` equals 2^64, one above the actual max.
fn integral_f64_as_u64(f: f64) -> Option<u64> {
    (f.is_finite() && f.fract() == 0.0 && (0.0..18446744073709551616.0).contains(&f))
        .then_some(f as u64)
}

/// Convert one `keyvals` JSON entry to a [`KeyValue`] typed per the key
/// column's own dtype. A JSON value that doesn't fit that dtype's family
/// (e.g. a string against a numeric key column, a tagged date against anything
/// but a temporal one, or a non-integral float against an Int/UInt key column,
/// the genuinely lossy case) is a typed refusal.
pub(super) fn keyvalue_from_json(
    v: &Value,
    dt: &DataType,
    key_name: &str,
    door: &str,
) -> Result<KeyValue, String> {
    let incompatible = || {
        let shown = match Literal::of(v) {
            Some(Literal::Instant(ms)) => format!(
                "{} (a date or datetime)",
                format_instant(ms, arrow::datatypes::TimeUnit::Millisecond)
            ),
            _ => v.to_string(),
        };
        format!(
            "{door}: keyvals value {shown} is not compatible with key column '{key_name}' dtype {dt:?}"
        )
    };
    let signed = |min: i64, max: i64| {
        let val = v.as_i64().or_else(|| v.as_f64().and_then(integral_f64_as_i64))?;
        (min..=max).contains(&val).then_some(KeyValue::Int(val))
    };
    let unsigned = |max: u64| {
        let val = v.as_u64().or_else(|| v.as_f64().and_then(integral_f64_as_u64))?;
        (val <= max).then_some(KeyValue::UInt(val))
    };
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 => {
            v.as_str().map(|s| KeyValue::Str(s.to_string())).ok_or_else(incompatible)
        }
        DataType::Float64 => {
            v.as_f64().map(|f| KeyValue::Float(f.to_bits())).ok_or_else(incompatible)
        }
        // Float32 key values are widened to f64 through an `f32` round-trip
        // in `group_key::groupby_key_at`: matching that exactly here (not
        // just `as_f64()`) is required, since a JSON value that isn't
        // exactly representable in f32 would otherwise never equal any
        // value actually present in the group.
        // An entry beyond `f32`'s finite range, or a nonzero one that underflows
        // to zero, is refused: narrowing it would give a key (an infinity, a
        // zero) that is not what the caller wrote and would fabricate a row at
        // it. An in-range entry that loses precision becomes the `f32` a column
        // of that dtype holds, so it equals the observed key it names.
        DataType::Float32 => v
            .as_f64()
            .and_then(|f| {
                let narrowed = f as f32;
                (narrowed.is_finite() && (narrowed != 0.0 || f == 0.0))
                    .then(|| KeyValue::Float((narrowed as f64).to_bits()))
            })
            .ok_or_else(incompatible),
        // A whole-valued number coerces onto an integer key (an integral JSON
        // float such as `1.0`, from `df["month"].unique().to_list()` off a
        // float column) and a fractional one is refused, since coercing it
        // would be lossy. Each value is range-checked against the target
        // dtype's width, so it never wraps when narrowed.
        DataType::Int8 => signed(i8::MIN.into(), i8::MAX.into()).ok_or_else(incompatible),
        DataType::Int16 => signed(i16::MIN.into(), i16::MAX.into()).ok_or_else(incompatible),
        DataType::Int32 => signed(i32::MIN.into(), i32::MAX.into()).ok_or_else(incompatible),
        DataType::Int64 => signed(i64::MIN, i64::MAX).ok_or_else(incompatible),
        DataType::UInt8 => unsigned(u8::MAX.into()).ok_or_else(incompatible),
        DataType::UInt16 => unsigned(u16::MAX.into()).ok_or_else(incompatible),
        DataType::UInt32 => unsigned(u32::MAX.into()).ok_or_else(incompatible),
        DataType::UInt64 => unsigned(u64::MAX).ok_or_else(incompatible),
        DataType::Boolean => v.as_bool().map(|b| KeyValue::Bool(b as u8)).ok_or_else(incompatible),
        // A tagged instant is that instant; a bare number is epoch milliseconds.
        DataType::Timestamp(unit, _) => {
            let literal = Literal::of(v).ok_or_else(incompatible)?;
            match as_instant(&literal, *unit) {
                AsInstant::Raw(raw) => Ok(KeyValue::Instant(raw, *unit)),
                AsInstant::Fractional => Err(format!(
                    "{door}: keyvals value {} is not a whole number of {} in key column \
                     '{key_name}'",
                    literal.spelled_instant(),
                    unit_words(*unit)
                )),
                AsInstant::OutOfRange => Err(format!(
                    "{door}: keyvals value {} is beyond the instants key column '{key_name}' \
                     ({}) holds",
                    literal.spelled_instant(),
                    unit_words(*unit)
                )),
                AsInstant::NotAnInstant => Err(incompatible()),
            }
        }
        other => Err(format!("{door}: key column '{key_name}' has unsupported dtype {other:?}")),
    }
}

/// What a key column holds, as the refusal of a number sequence names it, or
/// `None` for a column a number sequence can key.
fn non_numeric_holds(dt: &DataType) -> Option<&'static str> {
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 => Some("text"),
        DataType::Boolean => Some("booleans"),
        _ => None,
    }
}

/// The keys a `{start, stop, step}` sequence makes, as JSON numbers: `start`
/// (default 0), then each `step` (default 1, or -1 when `stop < start`) up to
/// but not including `stop`.
fn sequence_entries(
    seq: &KeySequence,
    dt: &DataType,
    key_name: &str,
    door: &str,
) -> Result<Vec<Value>, String> {
    if let Some(holds) = non_numeric_holds(dt) {
        return Err(format!(
            "{door}: keyvals={{'start': ...}} makes numbers, but the key '{key_name}' holds \
             {holds}; list the keys instead"
        ));
    }
    let start = seq.start.unwrap_or(0.0);
    let step = seq.step.unwrap_or(if seq.stop < start { -1.0 } else { 1.0 });
    if step == 0.0 {
        return Err(format!(
            "{door}: keyvals={{'start': ..., 'step': 0}} never reaches stop; use a non-zero step"
        ));
    }
    let span = (seq.stop - start) / step;
    if span.is_nan() || span <= 0.0 {
        return Ok(Vec::new());
    }
    let count = span.ceil();
    if count > MAX_SEQUENCE_KEYS as f64 {
        return Err(format!(
            "{door}: keyvals={{'start': ...}} would make {count} keys, more than the \
             {MAX_SEQUENCE_KEYS} a series can be completed over; list the keys, or use a larger step"
        ));
    }
    (0..count as usize)
        .map(|i| {
            serde_json::Number::from_f64(start + i as f64 * step).map(Value::Number).ok_or_else(|| {
                format!("{door}: keyvals={{'start': ...}} makes a key that is not a finite number")
            })
        })
        .collect()
}

/// The order a key axis places two keys of one column: numbers and instants by
/// value, `NaN` after every other number, `false` before `true`.
fn key_order(a: &KeyValue, b: &KeyValue) -> Ordering {
    match (a, b) {
        (KeyValue::Float(x), KeyValue::Float(y)) => {
            let (x, y) = (f64::from_bits(*x), f64::from_bits(*y));
            match (x.is_nan(), y.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => x.total_cmp(&y),
            }
        }
        _ => a.cmp(b),
    }
}

/// Whether `observed` lets `key` join the universe and hold a place in the key
/// order: every key does under [`ObservedKeys::Every`], and under
/// [`ObservedKeys::Placed`] a float key that is `NaN` or infinite, which no key
/// axis places, does not.
fn is_placed(observed: ObservedKeys, key: &KeyValue) -> bool {
    match (observed, key) {
        (ObservedKeys::Placed, KeyValue::Float(bits)) => f64::from_bits(*bits).is_finite(),
        _ => true,
    }
}

/// Every key a series is completed over, in key order, each key's place in
/// that order, and the first batch row that holds it.
pub(super) struct KeyUniverse {
    order: Vec<KeyValue>,
    rank: BTreeMap<KeyValue, usize>,
    first_row: Vec<Option<usize>>,
    observed: ObservedKeys,
}

impl KeyUniverse {
    /// The universe over `key_arr`: `keyvals` (typed against `dtype`), then the
    /// observed keys `observed` lets in, in key order.
    pub(super) fn build(
        key_arr: &dyn Array,
        dtype: &DataType,
        key_name: &str,
        keyvals: Option<&Keyvals>,
        observed: ObservedKeys,
        door: &str,
    ) -> Result<Self, String> {
        let named: Vec<Value> = match keyvals {
            None => Vec::new(),
            Some(Keyvals::List(values)) => values.clone(),
            Some(Keyvals::Sequence(seq)) => sequence_entries(seq, dtype, key_name, door)?,
        };
        let mut order: Vec<KeyValue> = Vec::new();
        let mut seen: BTreeSet<KeyValue> = BTreeSet::new();
        for v in &named {
            let key = keyvalue_from_json(v, dtype, key_name, door)?;
            if seen.insert(key.clone()) {
                order.push(key);
            }
        }
        let mut first_row_of: BTreeMap<KeyValue, usize> = BTreeMap::new();
        for row in 0..key_arr.len() {
            if let Some(key) = groupby_key_at(key_arr, dtype, row) {
                if matches!(key, KeyValue::Null) || !is_placed(observed, &key) {
                    continue;
                }
                first_row_of.entry(key.clone()).or_insert(row);
                if seen.insert(key.clone()) {
                    order.push(key);
                }
            }
        }
        if !matches!(dtype, DataType::Utf8 | DataType::LargeUtf8) {
            order.sort_by(key_order);
        }
        let first_row = order.iter().map(|key| first_row_of.get(key).copied()).collect();
        let rank = order.iter().cloned().enumerate().map(|(i, key)| (key, i)).collect();
        Ok(Self { order, rank, first_row, observed })
    }

    pub(super) fn len(&self) -> usize {
        self.order.len()
    }

    /// The key at place `rank` in the order.
    pub(super) fn key(&self, rank: usize) -> Option<&KeyValue> {
        self.order.get(rank)
    }

    /// The first batch row holding the key at place `rank`, `None` for a key
    /// only `keyvals` names.
    pub(super) fn first_row(&self, rank: usize) -> Option<usize> {
        self.first_row.get(rank).copied().flatten()
    }

    /// Each row's place in the key order, `None` for a null key and for a key
    /// the universe leaves out.
    pub(super) fn row_ranks(&self, key_arr: &dyn Array, dtype: &DataType) -> Vec<Option<usize>> {
        (0..key_arr.len())
            .map(|row| {
                groupby_key_at(key_arr, dtype, row)
                    .filter(|key| is_placed(self.observed, key))
                    .and_then(|key| self.rank.get(&key).copied())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BooleanArray, Float32Array, Float64Array, Int64Array, StringArray, TimestampSecondArray,
    };
    use arrow::datatypes::TimeUnit;
    use serde_json::json;

    const DOOR: &str = "data_impute";

    fn seq(start: Option<f64>, stop: f64, step: Option<f64>) -> KeySequence {
        KeySequence { start, stop, step }
    }

    fn numbers(values: &[Value]) -> Vec<f64> {
        values.iter().map(|v| v.as_f64().unwrap()).collect()
    }

    #[test]
    fn a_sequence_counts_up_from_zero_with_the_stop_excluded() {
        let got = sequence_entries(&seq(None, 3.0, None), &DataType::Int64, "t", DOOR).unwrap();
        assert_eq!(numbers(&got), vec![0.0, 1.0, 2.0]);
        let got = sequence_entries(&seq(Some(1.0), 6.0, Some(2.0)), &DataType::Int64, "t", DOOR).unwrap();
        assert_eq!(numbers(&got), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn a_sequence_with_its_stop_below_its_start_counts_down_one_at_a_time() {
        let got = sequence_entries(&seq(Some(5.0), 0.0, None), &DataType::Int64, "t", DOOR).unwrap();
        assert_eq!(numbers(&got), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let got = sequence_entries(&seq(Some(5.0), 0.0, Some(-2.0)), &DataType::Int64, "t", DOOR).unwrap();
        assert_eq!(numbers(&got), vec![5.0, 3.0, 1.0]);
    }

    #[test]
    fn a_sequence_that_never_reaches_its_stop_in_its_direction_is_empty() {
        // a positive step toward a stop below the start makes no key
        let got = sequence_entries(&seq(Some(5.0), 0.0, Some(1.0)), &DataType::Int64, "t", DOOR).unwrap();
        assert!(got.is_empty());
        let got = sequence_entries(&seq(Some(2.0), 2.0, None), &DataType::Int64, "t", DOOR).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn a_fractional_sequence_counts_from_the_start_without_accumulating_error() {
        let got = sequence_entries(&seq(Some(0.0), 1.0, Some(0.1)), &DataType::Float64, "t", DOOR).unwrap();
        assert_eq!(got.len(), 10);
        assert_eq!(numbers(&got)[3], 3.0 * 0.1);
    }

    #[test]
    fn a_zero_step_is_refused() {
        pyo3::Python::initialize();
        let err = sequence_entries(&seq(Some(0.0), 5.0, Some(0.0)), &DataType::Int64, "t", DOOR).unwrap_err();
        assert_eq!(
            err.to_string(),
            "data_impute: keyvals={'start': ..., 'step': 0} never reaches stop; use a non-zero step"
        );
    }

    #[test]
    fn a_sequence_on_a_text_or_boolean_key_is_refused_naming_what_the_key_holds() {
        pyo3::Python::initialize();
        for (dt, holds) in [(DataType::Utf8, "text"), (DataType::LargeUtf8, "text"), (DataType::Boolean, "booleans")] {
            let err = sequence_entries(&seq(None, 3.0, None), &dt, "k", DOOR).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "data_impute: keyvals={{'start': ...}} makes numbers, but the key \
                     'k' holds {holds}; list the keys instead"
                )
            );
        }
    }

    #[test]
    fn a_sequence_longer_than_the_cap_is_refused_and_one_at_the_cap_is_made() {
        pyo3::Python::initialize();
        let at_cap = sequence_entries(&seq(None, MAX_SEQUENCE_KEYS as f64, None), &DataType::Int64, "t", DOOR).unwrap();
        assert_eq!(at_cap.len(), MAX_SEQUENCE_KEYS);
        let err = sequence_entries(&seq(None, MAX_SEQUENCE_KEYS as f64 + 1.0, None), &DataType::Int64, "t", DOOR)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "data_impute: keyvals={'start': ...} would make 1000001 keys, more than the \
             1000000 a series can be completed over; list the keys, or use a larger step"
        );
        assert!(sequence_entries(&seq(None, 1e300, None), &DataType::Int64, "t", DOOR).is_err());
    }

    #[test]
    fn a_sequence_on_a_date_key_is_epoch_milliseconds() {
        let dt = DataType::Timestamp(TimeUnit::Second, None);
        let entries = sequence_entries(&seq(Some(0.0), 3000.0, Some(1000.0)), &dt, "t", DOOR).unwrap();
        let keys: Vec<_> =
            entries.iter().map(|v| keyvalue_from_json(v, &dt, "t", DOOR).unwrap()).collect();
        assert_eq!(
            keys,
            vec![
                KeyValue::Instant(0, TimeUnit::Second),
                KeyValue::Instant(1, TimeUnit::Second),
                KeyValue::Instant(2, TimeUnit::Second)
            ]
        );
    }

    #[test]
    fn a_keyval_is_typed_against_each_key_dtype() {
        assert_eq!(keyvalue_from_json(&json!("a"), &DataType::Utf8, "k", DOOR).unwrap(), KeyValue::Str("a".into()));
        assert_eq!(
            keyvalue_from_json(&json!(1.5), &DataType::Float64, "k", DOOR).unwrap(),
            KeyValue::Float(1.5_f64.to_bits())
        );
        assert_eq!(keyvalue_from_json(&json!(2.0), &DataType::Int64, "k", DOOR).unwrap(), KeyValue::Int(2));
        assert_eq!(keyvalue_from_json(&json!(7), &DataType::UInt8, "k", DOOR).unwrap(), KeyValue::UInt(7));
        assert_eq!(keyvalue_from_json(&json!(true), &DataType::Boolean, "k", DOOR).unwrap(), KeyValue::Bool(1));
    }

    #[test]
    fn a_float32_keyval_outside_the_finite_range_or_underflowing_to_zero_is_refused() {
        let dt = DataType::Float32;
        for v in [json!(1e39), json!(-1e39), json!(1e300), json!(1e-50), json!(-1e-50)] {
            let msg = keyvalue_from_json(&v, &dt, "k", DOOR).unwrap_err();
            assert_eq!(
                msg,
                format!("{DOOR}: keyvals value {v} is not compatible with key column 'k' dtype Float32")
            );
        }
        // the finite extremes, and zero of either sign, are keys
        for v in [json!(f32::MAX as f64), json!(f32::MIN as f64), json!(0.0), json!(-0.0)] {
            assert!(keyvalue_from_json(&v, &dt, "k", DOOR).is_ok(), "{v}");
        }
    }

    #[test]
    fn a_float32_keyval_that_loses_precision_equals_the_observed_key_it_names() {
        let arr = Float32Array::from(vec![0.1_f32, 0.2]);
        let named = Keyvals::List(vec![json!(0.1), json!(0.100_000_000_01)]);
        let u = KeyUniverse::build(&arr, &DataType::Float32, "k", Some(&named), ObservedKeys::Placed, DOOR).unwrap();
        // both entries narrow to the f32 0.1 the column holds: no fabricated key
        assert_eq!(u.len(), 2);
    }

    #[test]
    fn a_keyval_beyond_a_narrow_dtype_is_refused_not_wrapped() {
        pyo3::Python::initialize();
        for (v, dt) in [
            (json!(128), DataType::Int8),
            (json!(-129), DataType::Int8),
            (json!(32_768), DataType::Int16),
            (json!(3_000_000_000_i64), DataType::Int32),
            (json!(256), DataType::UInt8),
            (json!(-1), DataType::UInt8),
            (json!(65_536), DataType::UInt16),
            (json!(4_294_967_296_u64), DataType::UInt32),
            (json!(1.5), DataType::Int64),
        ] {
            let msg = keyvalue_from_json(&v, &dt, "k", DOOR).unwrap_err().to_string();
            assert_eq!(
                msg,
                format!("data_impute: keyvals value {v} is not compatible with key column 'k' dtype {dt:?}")
            );
        }
    }

    #[test]
    fn a_tagged_instant_is_that_instant_on_a_temporal_key_at_every_unit() {
        let tagged = json!({"epoch_ms": 1_704_067_200_000_i64});
        for (unit, raw) in [
            (TimeUnit::Second, 1_704_067_200),
            (TimeUnit::Millisecond, 1_704_067_200_000),
            (TimeUnit::Microsecond, 1_704_067_200_000_000),
            (TimeUnit::Nanosecond, 1_704_067_200_000_000_000),
        ] {
            let dt = DataType::Timestamp(unit, Some("Europe/Paris".into()));
            assert_eq!(keyvalue_from_json(&tagged, &dt, "k", DOOR).unwrap(), KeyValue::Instant(raw, unit));
            // a bare number on a temporal key is the same epoch milliseconds
            assert_eq!(
                keyvalue_from_json(&json!(1_704_067_200_000_i64), &dt, "k", DOOR).unwrap(),
                KeyValue::Instant(raw, unit)
            );
        }
    }

    #[test]
    fn a_tagged_instant_on_a_key_that_is_not_temporal_is_refused_as_a_date() {
        pyo3::Python::initialize();
        let tagged = json!({"epoch_ms": 1_704_067_200_000_i64});
        for dt in [DataType::Int64, DataType::Float64, DataType::Utf8, DataType::Boolean, DataType::UInt8] {
            assert_eq!(
                keyvalue_from_json(&tagged, &dt, "k", DOOR).unwrap_err().to_string(),
                format!(
                    "data_impute: keyvals value 2024-01-01T00:00:00 (a date or datetime) \
                     is not compatible with key column 'k' dtype {dt:?}"
                )
            );
        }
    }

    #[test]
    fn a_keyval_of_another_kind_than_a_temporal_key_is_incompatible() {
        pyo3::Python::initialize();
        let dt = DataType::Timestamp(TimeUnit::Millisecond, None);
        for v in [json!("2024-01-01"), json!(true), json!(null), json!([1])] {
            let msg = keyvalue_from_json(&v, &dt, "k", DOOR).unwrap_err().to_string();
            assert_eq!(
                msg,
                format!(
                    "data_impute: keyvals value {v} is not compatible with key column \
                     'k' dtype Timestamp(Millisecond, None)"
                )
            );
        }
    }

    #[test]
    fn a_keyval_that_is_no_whole_number_of_the_unit_or_beyond_it_is_named_for_its_cause() {
        let seconds = DataType::Timestamp(TimeUnit::Second, None);
        let nanos = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let cases: Vec<(Value, &DataType, &str)> = vec![
            (
                json!({"epoch_ms": 1_500}),
                &seconds,
                "keyvals value 1970-01-01T00:00:01.500 is not a whole number of seconds in key column 'k'",
            ),
            (
                json!(1_500),
                &seconds,
                "keyvals value 1500 epoch milliseconds is not a whole number of seconds in key column 'k'",
            ),
            (
                json!(9_000_000_000_000_000_i64),
                &nanos,
                "keyvals value 9000000000000000 epoch milliseconds is beyond the instants key column 'k' (nanoseconds) holds",
            ),
        ];
        for (v, dt, tail) in cases {
            let msg = keyvalue_from_json(&v, dt, "k", DOOR).unwrap_err();
            assert_eq!(msg, format!("{DOOR}: {tail}"), "{v}");
        }
    }

    #[test]
    fn a_key_dtype_with_no_reader_is_refused() {
        pyo3::Python::initialize();
        let msg = keyvalue_from_json(&json!(1), &DataType::Date32, "k", DOOR).unwrap_err().to_string();
        assert_eq!(msg, "data_impute: key column 'k' has unsupported dtype Date32");
    }

    fn universe_of(values: &[f64], keyvals: Option<&Keyvals>) -> KeyUniverse {
        let arr = Float64Array::from(values.to_vec());
        KeyUniverse::build(&arr, &DataType::Float64, "t", keyvals, ObservedKeys::Every, DOOR).unwrap()
    }

    fn keys_of(u: &KeyUniverse) -> Vec<f64> {
        (0..u.len())
            .map(|i| match u.key(i) {
                Some(KeyValue::Float(bits)) => f64::from_bits(*bits),
                other => panic!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn a_numeric_universe_is_ascending_whatever_order_the_keys_were_seen_in() {
        let u = universe_of(&[3.0, -1.0, 2.0, 3.0, -5.0], None);
        assert_eq!(keys_of(&u), vec![-5.0, -1.0, 2.0, 3.0]);
    }

    #[test]
    fn keyvals_are_added_to_the_observed_keys_and_never_replace_them() {
        let named = Keyvals::List(vec![json!(1), json!(4)]);
        let u = universe_of(&[1.0, 2.0, 3.0], Some(&named));
        assert_eq!(keys_of(&u), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn a_universe_holds_a_key_once_however_often_it_is_named_or_seen() {
        let named = Keyvals::List(vec![json!(2), json!(2), json!(9)]);
        let u = universe_of(&[2.0, 2.0, 9.0], Some(&named));
        assert_eq!(keys_of(&u), vec![2.0, 9.0]);
    }

    #[test]
    fn nan_sorts_after_every_number_and_infinities_at_the_ends() {
        // a NaN whose sign bit is set (what 0.0 / 0.0 gives on x86) sorts last too,
        // and NaNs keep their first-appearance order among themselves
        let negative_nan = f64::from_bits(0xFFF8_0000_0000_0000);
        let u = universe_of(
            &[negative_nan, 1.0, f64::NEG_INFINITY, f64::NAN, f64::INFINITY],
            None,
        );
        let keys = keys_of(&u);
        assert_eq!(&keys[..3], &[f64::NEG_INFINITY, 1.0, f64::INFINITY]);
        assert!(keys[3].is_nan() && keys[4].is_nan());
        assert_eq!(keys[3].to_bits(), negative_nan.to_bits());
        assert_eq!(keys[4].to_bits(), f64::NAN.to_bits());
    }

    #[test]
    fn a_placed_universe_leaves_nan_and_infinite_float_keys_out_and_gives_them_no_rank() {
        for values in [[f64::NAN, 1.0, 2.0], [f64::INFINITY, 1.0, 2.0], [f64::NEG_INFINITY, 1.0, 2.0]] {
            let arr = Float64Array::from(values.to_vec());
            let u = KeyUniverse::build(&arr, &DataType::Float64, "t", None, ObservedKeys::Placed, DOOR).unwrap();
            assert_eq!(keys_of(&u), vec![1.0, 2.0]);
            assert_eq!(u.row_ranks(&arr, &DataType::Float64), vec![None, Some(0), Some(1)]);
        }
        let single = Float32Array::from(vec![f32::NAN, f32::INFINITY, 1.5]);
        let u = KeyUniverse::build(&single, &DataType::Float32, "t", None, ObservedKeys::Placed, DOOR).unwrap();
        assert_eq!(u.len(), 1);
        assert_eq!(u.row_ranks(&single, &DataType::Float32), vec![None, None, Some(0)]);
        // under Every the same keys are keys of their own
        let every = KeyUniverse::build(&single, &DataType::Float32, "t", None, ObservedKeys::Every, DOOR).unwrap();
        assert_eq!(every.len(), 3);
    }

    #[test]
    fn a_placed_universe_keeps_every_other_key_kind() {
        let ints = Int64Array::from(vec![i64::MAX, i64::MIN, 0]);
        let u = KeyUniverse::build(&ints, &DataType::Int64, "t", None, ObservedKeys::Placed, DOOR).unwrap();
        assert_eq!(u.len(), 3);
        let text = StringArray::from(vec!["nan", "inf"]);
        let u = KeyUniverse::build(&text, &DataType::Utf8, "t", None, ObservedKeys::Placed, DOOR).unwrap();
        assert_eq!(u.len(), 2);
    }

    #[test]
    fn a_key_remembers_the_first_row_that_holds_it_and_a_keyvals_only_key_has_none() {
        let arr = Float64Array::from(vec![3.0, 1.0, 3.0, 1.0]);
        let named = Keyvals::List(vec![json!(2)]);
        let u = KeyUniverse::build(&arr, &DataType::Float64, "t", Some(&named), ObservedKeys::Every, DOOR).unwrap();
        // order 1, 2, 3
        assert_eq!((u.first_row(0), u.first_row(1), u.first_row(2)), (Some(1), None, Some(0)));
        assert_eq!(u.first_row(9), None);
    }

    #[test]
    fn a_null_key_is_no_key_and_has_no_rank() {
        let arr = Float64Array::from(vec![Some(2.0), None, Some(1.0)]);
        let u = KeyUniverse::build(&arr, &DataType::Float64, "t", None, ObservedKeys::Every, DOOR).unwrap();
        assert_eq!(u.len(), 2);
        assert_eq!(u.row_ranks(&arr, &DataType::Float64), vec![Some(1), None, Some(0)]);
    }

    #[test]
    fn a_text_universe_is_keyvals_first_then_first_appearance() {
        let arr = StringArray::from(vec!["c", "a", "b", "a"]);
        let plain = KeyUniverse::build(&arr, &DataType::Utf8, "t", None, ObservedKeys::Every, DOOR).unwrap();
        let text = |u: &KeyUniverse| -> Vec<String> {
            (0..u.len()).map(|i| match u.key(i) {
                Some(KeyValue::Str(s)) => s.clone(),
                other => panic!("{other:?}"),
            }).collect()
        };
        assert_eq!(text(&plain), vec!["c", "a", "b"]);
        let named = Keyvals::List(vec![json!("z"), json!("b")]);
        let with = KeyUniverse::build(&arr, &DataType::Utf8, "t", Some(&named), ObservedKeys::Every, DOOR).unwrap();
        assert_eq!(text(&with), vec!["z", "b", "c", "a"]);
    }

    #[test]
    fn a_boolean_universe_puts_false_before_true() {
        let arr = BooleanArray::from(vec![true, false]);
        let u = KeyUniverse::build(&arr, &DataType::Boolean, "t", None, ObservedKeys::Every, DOOR).unwrap();
        assert_eq!(u.key(0), Some(&KeyValue::Bool(0)));
        assert_eq!(u.key(1), Some(&KeyValue::Bool(1)));
    }

    #[test]
    fn an_instant_universe_orders_by_time_across_negative_values() {
        let arr = TimestampSecondArray::from(vec![5, -3, 0]);
        let dt = DataType::Timestamp(TimeUnit::Second, None);
        let u = KeyUniverse::build(&arr, &dt, "t", None, ObservedKeys::Every, DOOR).unwrap();
        assert_eq!(
            (0..3).filter_map(|i| u.key(i).cloned()).collect::<Vec<_>>(),
            vec![
                KeyValue::Instant(-3, TimeUnit::Second),
                KeyValue::Instant(0, TimeUnit::Second),
                KeyValue::Instant(5, TimeUnit::Second)
            ]
        );
    }

    #[test]
    fn an_integer_universe_orders_negative_values_before_positive_ones() {
        let arr = Int64Array::from(vec![2, -7, 0]);
        let u = KeyUniverse::build(&arr, &DataType::Int64, "t", None, ObservedKeys::Every, DOOR).unwrap();
        assert_eq!(
            (0..3).filter_map(|i| u.key(i).cloned()).collect::<Vec<_>>(),
            vec![KeyValue::Int(-7), KeyValue::Int(0), KeyValue::Int(2)]
        );
    }

    #[test]
    fn a_sequence_universe_is_added_to_the_observed_keys() {
        let named = Keyvals::Sequence(seq(Some(0.0), 4.0, None));
        let u = universe_of(&[2.0, 9.0], Some(&named));
        assert_eq!(keys_of(&u), vec![0.0, 1.0, 2.0, 3.0, 9.0]);
    }
}
