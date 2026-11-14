//! What an impute writes into a cell: the method, the constant a `value=`
//! names for each kind of column, and the statistics a `method=` and a
//! `frame=` compute from the usable values around it.

use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, LargeStringArray, PrimitiveArray, StringArray,
};
use arrow::datatypes::{
    ArrowTimestampType, DataType, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType,
};
use serde_json::Value;
use std::cmp::Ordering;
use std::sync::Arc;

use super::literal::{as_instant, unit_words, AsInstant, Literal};
use super::METHODS;
use crate::transform::numeric_util::usable_at;
use crate::transform::vocabulary::refuse_undeclared;

/// A fill statistic over the usable values it is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Statistic {
    Mean,
    Median,
    Min,
    Max,
}

/// How a missing value is filled: with the one constant `value=` names, or
/// with a statistic of the usable values around it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Method {
    Value,
    Statistic(Statistic),
}

impl Method {
    /// The method `method` spells, refused with the vocabulary's own text when
    /// it is not one of [`METHODS`].
    pub(super) fn parse(door: &str, method: &str) -> Result<Self, String> {
        refuse_undeclared(METHODS, door, "method", method)?;
        Ok(match method {
            "value" => Self::Value,
            "mean" => Self::Statistic(Statistic::Mean),
            "median" => Self::Statistic(Statistic::Median),
            "min" => Self::Statistic(Statistic::Min),
            _ => Self::Statistic(Statistic::Max),
        })
    }

    #[cfg(test)]
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Value => "value",
            Self::Statistic(Statistic::Mean) => "mean",
            Self::Statistic(Statistic::Median) => "median",
            Self::Statistic(Statistic::Min) => "min",
            Self::Statistic(Statistic::Max) => "max",
        }
    }
}

impl Statistic {
    /// The statistic's name, as a warning says which one was used.
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Median => "median",
            Self::Min => "min",
            Self::Max => "max",
        }
    }

    /// The fill for a set with no usable value at all, which a caller falls
    /// back from to a wider set first: `0.0` for a mean or median, an infinity
    /// for a minimum or maximum.
    pub(super) fn degenerate_default(self) -> f64 {
        match self {
            Self::Mean | Self::Median => 0.0,
            Self::Min => f64::INFINITY,
            Self::Max => f64::NEG_INFINITY,
        }
    }

    /// The statistic of `values` (already only the usable ones), or `None`
    /// when there is none. Sums in the order given, so a group's arithmetic is
    /// the same whatever else is computed.
    pub(super) fn of(self, mut values: Vec<f64>) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        let n = values.len();
        Some(match self {
            Self::Mean => values.iter().sum::<f64>() / n as f64,
            Self::Median => {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                if n % 2 == 1 {
                    values[n / 2]
                } else {
                    0.5 * (values[n / 2 - 1] + values[n / 2])
                }
            }
            Self::Min => values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            Self::Max => values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        })
    }
}

/// One position in a series, in key order, that a `frame=` window counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Entry {
    /// A row of the batch.
    Caller(usize),
    /// A row the impute adds, which holds no value: the series' `n`-th missing key.
    Imputed(usize),
}

/// The statistic of the usable values the entries in the half-open `window`
/// (as `data_window::frame_range` gives it) hold, or `None` when it holds none.
pub(super) fn window_statistic(
    stat: Statistic,
    col: &Float64Array,
    entries: &[Entry],
    (lo, hi): (usize, usize),
) -> Option<f64> {
    let values = entries[lo..hi]
        .iter()
        .filter_map(|entry| match entry {
            Entry::Caller(row) => usable_at(col, *row),
            Entry::Imputed(_) => None,
        })
        .collect();
    stat.of(values)
}

/// What a column holds, as a `value=` must match it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ValueKind {
    Numbers,
    Instants,
    Text,
    Booleans,
}

impl ValueKind {
    /// A dtype with no kind of its own reads as numbers, which the numeric
    /// reader then refuses with the dtype's name.
    pub(super) fn of(dtype: &DataType) -> Self {
        match dtype {
            DataType::Timestamp(..) => Self::Instants,
            DataType::Utf8 | DataType::LargeUtf8 => Self::Text,
            DataType::Boolean => Self::Booleans,
            _ => Self::Numbers,
        }
    }

    fn holds(self) -> &'static str {
        match self {
            Self::Numbers => "numbers",
            Self::Instants => "dates and times",
            Self::Text => "text",
            Self::Booleans => "booleans",
        }
    }

    fn must(self) -> &'static str {
        match self {
            Self::Numbers => "must be a number",
            Self::Instants => {
                "must be a datetime.date, a datetime.datetime or an epoch-millisecond number"
            }
            Self::Text => "must be a string",
            Self::Booleans => "must be True or False",
        }
    }
}

/// The literal a constant fill is: the caller's `value`, else 0 for a numeric
/// column. Any other kind of column has no default constant, so a missing
/// `value` is refused rather than read as a number that kind cannot hold.
pub(super) fn resolve_literal(
    value: Option<&Value>,
    kind: ValueKind,
    door: &str,
    field: &str,
) -> Result<Literal, String> {
    match value {
        Some(v) => Literal::of(v).ok_or_else(|| {
            format!(
                "{door}: value= must be a number, a string, True or False, a date or a datetime; \
                 got {v}"
            )
        }),
        None if kind == ValueKind::Numbers => Ok(Literal::Number(0.into())),
        None => Err(format!(
            "{door}: column '{field}' holds {}, so value= {}, and none was given; pass value=",
            kind.holds(),
            kind.must()
        )),
    }
}

fn instants<T: ArrowTimestampType>(raw: i64, n: usize, tz: &Option<Arc<str>>) -> ArrayRef {
    Arc::new(
        PrimitiveArray::<T>::from_iter_values(std::iter::repeat_n(raw, n))
            .with_timezone_opt(tz.clone()),
    )
}

/// `n` copies of the constant `lit` at the column's own dtype: a `Float64`
/// number, a timestamp at its unit and zone, text, or a boolean. A literal of
/// another kind than the column holds is refused naming both.
pub(super) fn fill_array(
    lit: &Literal,
    dtype: &DataType,
    n: usize,
    door: &str,
    field: &str,
) -> Result<ArrayRef, String> {
    let kind = ValueKind::of(dtype);
    let mismatch = || {
        format!(
            "{door}: column '{field}' holds {}, so value= {}, but {} is {}",
            kind.holds(),
            kind.must(),
            lit.spelled(),
            lit.kind()
        )
    };
    match (dtype, lit) {
        (DataType::Float64, Literal::Number(num)) => {
            let v = num.as_f64().ok_or_else(mismatch)?;
            Ok(Arc::new(Float64Array::from(vec![v; n])))
        }
        (DataType::Utf8, Literal::Text(s)) => Ok(Arc::new(StringArray::from(vec![s.as_str(); n]))),
        (DataType::LargeUtf8, Literal::Text(s)) => {
            Ok(Arc::new(LargeStringArray::from(vec![s.as_str(); n])))
        }
        (DataType::Boolean, Literal::Bool(b)) => Ok(Arc::new(BooleanArray::from(vec![*b; n]))),
        (DataType::Timestamp(unit, tz), _) => match as_instant(lit, *unit) {
            AsInstant::Raw(raw) => Ok(match unit {
                TimeUnit::Second => instants::<TimestampSecondType>(raw, n, tz),
                TimeUnit::Millisecond => instants::<TimestampMillisecondType>(raw, n, tz),
                TimeUnit::Microsecond => instants::<TimestampMicrosecondType>(raw, n, tz),
                TimeUnit::Nanosecond => instants::<TimestampNanosecondType>(raw, n, tz),
            }),
            AsInstant::Fractional => Err(format!(
                "{door}: column '{field}' holds {} timestamps, so value= {} must be a whole \
                 number of {}",
                unit_words(*unit),
                lit.spelled_instant(),
                unit_words(*unit)
            )),
            AsInstant::OutOfRange => Err(format!(
                "{door}: column '{field}' holds {} timestamps, and value= {} is beyond the \
                 instants they hold",
                unit_words(*unit),
                lit.spelled_instant()
            )),
            AsInstant::NotAnInstant => Err(mismatch()),
        },
        _ => Err(mismatch()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Array, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray,
    };
    use serde_json::json;

    const DOOR: &str = "data_impute";

    #[test]
    fn every_declared_method_parses_to_itself() {
        for method in METHODS {
            assert_eq!(Method::parse(DOOR, method).unwrap().name(), *method);
        }
    }

    #[test]
    fn an_undeclared_method_is_refused_with_the_vocabulary() {
        pyo3::Python::initialize();
        let msg = Method::parse(DOOR, "bogus").unwrap_err().to_string();
        assert_eq!(msg, "data_impute: unknown method 'bogus'; expected value|mean|median|min|max");
        let msg = Method::parse("impute", "Mean").unwrap_err().to_string();
        assert_eq!(msg, "impute: unknown method 'Mean'; expected value|mean|median|min|max");
    }

    #[test]
    fn each_statistic_reads_the_values_it_is_given() {
        let vals = || vec![4.0, 1.0, 10.0, 3.0];
        assert_eq!(Statistic::Mean.of(vals()), Some(4.5));
        assert_eq!(Statistic::Median.of(vals()), Some(3.5));
        assert_eq!(Statistic::Median.of(vec![5.0, 1.0, 3.0]), Some(3.0));
        assert_eq!(Statistic::Min.of(vals()), Some(1.0));
        assert_eq!(Statistic::Max.of(vals()), Some(10.0));
        for stat in [Statistic::Mean, Statistic::Median, Statistic::Min, Statistic::Max] {
            assert_eq!(stat.of(Vec::new()), None, "{stat:?}");
        }
    }

    #[test]
    fn a_set_with_no_usable_value_has_the_degenerate_default_of_its_statistic() {
        assert_eq!(Statistic::Mean.degenerate_default(), 0.0);
        assert_eq!(Statistic::Median.degenerate_default(), 0.0);
        assert_eq!(Statistic::Min.degenerate_default(), f64::INFINITY);
        assert_eq!(Statistic::Max.degenerate_default(), f64::NEG_INFINITY);
    }

    #[test]
    fn a_window_counts_an_imputed_entry_as_a_row_with_no_value() {
        let col = Float64Array::from(vec![Some(10.0), Some(20.0), None, Some(f64::INFINITY), Some(f64::NAN)]);
        let entries = [Entry::Caller(0), Entry::Imputed(0), Entry::Caller(1), Entry::Caller(2), Entry::Caller(3), Entry::Caller(4)];
        // 10 and 20 are usable; the imputed, null, infinite and NaN entries hold nothing
        assert_eq!(window_statistic(Statistic::Mean, &col, &entries, (0, 6)), Some(15.0));
        assert_eq!(window_statistic(Statistic::Mean, &col, &entries, (1, 2)), None);
        assert_eq!(window_statistic(Statistic::Mean, &col, &entries, (3, 6)), None);
        assert_eq!(window_statistic(Statistic::Max, &col, &entries, (0, 3)), Some(20.0));
    }

    #[test]
    fn a_number_value_fills_a_float_column_and_defaults_to_zero() {
        let lit = resolve_literal(Some(&json!(2.5)), ValueKind::Numbers, DOOR, "v").unwrap();
        let arr = fill_array(&lit, &DataType::Float64, 3, DOOR, "v").unwrap();
        assert_eq!(arr.as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec(), vec![2.5; 3]);
        let lit = resolve_literal(None, ValueKind::Numbers, DOOR, "v").unwrap();
        let arr = fill_array(&lit, &DataType::Float64, 2, DOOR, "v").unwrap();
        assert_eq!(arr.as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec(), vec![0.0; 2]);
    }

    #[test]
    fn a_string_fills_text_and_large_text_and_a_bool_fills_a_boolean() {
        let text = fill_array(&Literal::Text("none".into()), &DataType::Utf8, 2, DOOR, "g").unwrap();
        let text = text.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!((text.value(0), text.value(1)), ("none", "none"));
        let large = fill_array(&Literal::Text("x".into()), &DataType::LargeUtf8, 1, DOOR, "g").unwrap();
        assert_eq!(large.as_any().downcast_ref::<LargeStringArray>().unwrap().value(0), "x");
        let flag = fill_array(&Literal::Bool(false), &DataType::Boolean, 2, DOOR, "ok").unwrap();
        let flag = flag.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!flag.value(0) && !flag.value(1) && flag.null_count() == 0);
    }

    fn first_raw(arr: &ArrayRef, unit: TimeUnit) -> i64 {
        match unit {
            TimeUnit::Second => arr.as_any().downcast_ref::<TimestampSecondArray>().unwrap().value(0),
            TimeUnit::Millisecond => {
                arr.as_any().downcast_ref::<TimestampMillisecondArray>().unwrap().value(0)
            }
            TimeUnit::Microsecond => {
                arr.as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap().value(0)
            }
            TimeUnit::Nanosecond => {
                arr.as_any().downcast_ref::<TimestampNanosecondArray>().unwrap().value(0)
            }
        }
    }

    #[test]
    fn an_instant_fills_a_timestamp_at_its_unit_and_zone() {
        let tz: Option<Arc<str>> = Some(Arc::from("Europe/Paris"));
        for (unit, raw) in [
            (TimeUnit::Second, 1_704_153_600_i64),
            (TimeUnit::Millisecond, 1_704_153_600_000),
            (TimeUnit::Microsecond, 1_704_153_600_000_000),
            (TimeUnit::Nanosecond, 1_704_153_600_000_000_000),
        ] {
            let dt = DataType::Timestamp(unit, tz.clone());
            for lit in [Literal::Instant(1_704_153_600_000), Literal::Number(1_704_153_600_000_i64.into())] {
                let arr = fill_array(&lit, &dt, 2, DOOR, "when").unwrap();
                assert_eq!(arr.data_type(), &dt, "the unit and zone are the column's");
                assert_eq!(arr.len(), 2);
                assert_eq!(first_raw(&arr, unit), raw, "{unit:?}");
            }
        }
    }

    #[test]
    fn a_value_that_is_no_whole_number_of_the_unit_or_beyond_it_is_named_for_its_cause() {
        let seconds = DataType::Timestamp(TimeUnit::Second, None);
        let nanos = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let cases: Vec<(Literal, &DataType, &str)> = vec![
            (
                Literal::Number(1500.into()),
                &seconds,
                "column 'when' holds seconds timestamps, so value= 1500 epoch milliseconds must be a whole number of seconds",
            ),
            (
                Literal::Instant(1_500),
                &seconds,
                "column 'when' holds seconds timestamps, so value= 1970-01-01T00:00:01.500 must be a whole number of seconds",
            ),
            (
                Literal::Instant(i64::MAX / 10),
                &nanos,
                "column 'when' holds nanoseconds timestamps, and value= 922337203685477580 ms since the Unix epoch is beyond the instants they hold",
            ),
            (
                Literal::Number((i64::MAX / 10).into()),
                &nanos,
                "column 'when' holds nanoseconds timestamps, and value= 922337203685477580 epoch milliseconds is beyond the instants they hold",
            ),
        ];
        for (lit, dt, tail) in cases {
            let msg = fill_array(&lit, dt, 1, DOOR, "when").unwrap_err();
            assert_eq!(msg, format!("{DOOR}: {tail}"), "{lit:?}");
        }
    }

    #[test]
    fn a_literal_of_another_kind_than_its_column_is_refused_naming_both() {
        pyo3::Python::initialize();
        let ts = DataType::Timestamp(TimeUnit::Millisecond, None);
        let cases: Vec<(Literal, DataType, &str)> = vec![
            (Literal::Text("none".into()), DataType::Float64, "holds numbers, so value= must be a number, but 'none' is text"),
            (Literal::Bool(true), DataType::Float64, "holds numbers, so value= must be a number, but True is a boolean"),
            (Literal::Instant(1_704_067_200_000), DataType::Float64, "holds numbers, so value= must be a number, but 2024-01-01T00:00:00 is a date or datetime"),
            (Literal::Text("none".into()), ts.clone(), "holds dates and times, so value= must be a datetime.date, a datetime.datetime or an epoch-millisecond number, but 'none' is text"),
            (Literal::Bool(false), ts, "holds dates and times, so value= must be a datetime.date, a datetime.datetime or an epoch-millisecond number, but False is a boolean"),
            (Literal::Number(5.into()), DataType::Utf8, "holds text, so value= must be a string, but 5 is a number"),
            (Literal::Bool(true), DataType::LargeUtf8, "holds text, so value= must be a string, but True is a boolean"),
            (Literal::Instant(1_704_067_200_000), DataType::Utf8, "holds text, so value= must be a string, but 2024-01-01T00:00:00 is a date or datetime"),
            (Literal::Text("none".into()), DataType::Boolean, "holds booleans, so value= must be True or False, but 'none' is text"),
            (Literal::Number(0.into()), DataType::Boolean, "holds booleans, so value= must be True or False, but 0 is a number"),
            (Literal::Instant(1_704_067_200_000), DataType::Boolean, "holds booleans, so value= must be True or False, but 2024-01-01T00:00:00 is a date or datetime"),
        ];
        for (lit, dt, tail) in cases {
            let msg = fill_array(&lit, &dt, 1, DOOR, "c").unwrap_err().to_string();
            assert_eq!(msg, format!("data_impute: column 'c' {tail}"), "{lit:?} on {dt:?}");
        }
    }

    #[test]
    fn a_value_with_no_literal_kind_is_refused() {
        pyo3::Python::initialize();
        for v in [json!(null), json!([1]), json!({"a": 1})] {
            let msg = resolve_literal(Some(&v), ValueKind::Numbers, DOOR, "v").unwrap_err().to_string();
            assert_eq!(
                msg,
                format!("data_impute: value= must be a number, a string, True or False, a date or a datetime; got {v}")
            );
        }
    }

    #[test]
    fn a_column_that_is_not_numeric_has_no_default_value() {
        pyo3::Python::initialize();
        for (kind, tail) in [
            (ValueKind::Instants, "holds dates and times, so value= must be a datetime.date, a datetime.datetime or an epoch-millisecond number"),
            (ValueKind::Text, "holds text, so value= must be a string"),
            (ValueKind::Booleans, "holds booleans, so value= must be True or False"),
        ] {
            let msg = resolve_literal(None, kind, "impute", "c").unwrap_err().to_string();
            assert_eq!(msg, format!("impute: column 'c' {tail}, and none was given; pass value="));
        }
    }

    #[test]
    fn a_dtype_is_read_as_the_kind_of_value_it_holds() {
        assert_eq!(ValueKind::of(&DataType::Int8), ValueKind::Numbers);
        assert_eq!(ValueKind::of(&DataType::Float32), ValueKind::Numbers);
        assert_eq!(ValueKind::of(&DataType::Decimal128(10, 2)), ValueKind::Numbers);
        assert_eq!(ValueKind::of(&DataType::Timestamp(TimeUnit::Second, None)), ValueKind::Instants);
        assert_eq!(ValueKind::of(&DataType::Utf8), ValueKind::Text);
        assert_eq!(ValueKind::of(&DataType::LargeUtf8), ValueKind::Text);
        assert_eq!(ValueKind::of(&DataType::Boolean), ValueKind::Booleans);
    }
}
