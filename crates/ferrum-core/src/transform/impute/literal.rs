//! The literals an impute option carries: a `value=` and each `keyvals` entry,
//! as the JSON the caller's Python wrote them.
//!
//! A number, a string and a bool cross the wire as themselves. A date, a
//! datetime or a `numpy.datetime64` crosses it tagged, `{"epoch_ms": <integer>}`,
//! so a column that is not temporal can tell it from a number and refuse it
//! instead of reading its millisecond count. A temporal column or key reads
//! the tagged form as that instant, and a bare number as epoch milliseconds,
//! each scaled to the column's own unit.

use arrow::datatypes::TimeUnit;
use serde_json::{Number, Value};

use crate::transform::context::format_instant;

/// The one key of a tagged instant.
const INSTANT_TAG: &str = "epoch_ms";

/// What a `value=` or a `keyvals` entry is, read from its JSON.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Literal {
    Number(Number),
    Text(String),
    Bool(bool),
    /// A tagged instant, in epoch milliseconds.
    Instant(i64),
}

impl Literal {
    /// The literal `v` spells, or `None` for a JSON null, an array, or an
    /// object that is not a tagged instant.
    pub(super) fn of(v: &Value) -> Option<Self> {
        match v {
            Value::Number(n) => Some(Self::Number(n.clone())),
            Value::String(s) => Some(Self::Text(s.clone())),
            Value::Bool(b) => Some(Self::Bool(*b)),
            Value::Object(map) if map.len() == 1 => {
                map.get(INSTANT_TAG)?.as_i64().map(Self::Instant)
            }
            _ => None,
        }
    }

    /// The literal as the Python caller wrote it: a string quoted, a bool as
    /// `True` or `False`, an instant as an ISO-8601 date-time.
    pub(super) fn spelled(&self) -> String {
        match self {
            Self::Number(n) => n.to_string(),
            Self::Text(s) => format!("'{s}'"),
            Self::Bool(true) => "True".to_string(),
            Self::Bool(false) => "False".to_string(),
            Self::Instant(ms) => format_instant(*ms, TimeUnit::Millisecond),
        }
    }

    /// The literal as an instant is spelled in a refusal about a timestamp
    /// column or key: a tagged instant as ISO-8601, a bare number as the
    /// epoch milliseconds it is read as.
    pub(super) fn spelled_instant(&self) -> String {
        match self {
            Self::Number(n) => format!("{n} epoch milliseconds"),
            other => other.spelled(),
        }
    }

    /// What kind of thing the literal is, as a refusal says it after "is".
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Number(_) => "a number",
            Self::Text(_) => "text",
            Self::Bool(_) => "a boolean",
            Self::Instant(_) => "a date or datetime",
        }
    }
}

/// A literal read as an instant at a column's unit.
pub(super) enum AsInstant {
    /// The raw value at the column's unit.
    Raw(i64),
    /// An instant, but not a whole number of the column's unit (a second
    /// column given a fraction of a second).
    Fractional,
    /// An instant beyond what `i64` holds at the column's unit.
    OutOfRange,
    /// A literal that is not a number or an instant.
    NotAnInstant,
}

/// `lit` as an instant at `unit`: a tagged instant or a bare number is epoch
/// milliseconds, scaled to the unit exactly.
pub(super) fn as_instant(lit: &Literal, unit: TimeUnit) -> AsInstant {
    match lit {
        Literal::Instant(ms) => scale_whole(*ms, unit),
        Literal::Number(n) => match n.as_i64() {
            Some(ms) => scale_whole(ms, unit),
            None => n.as_f64().map_or(AsInstant::OutOfRange, |ms| scale_fractional(ms, unit)),
        },
        Literal::Text(_) | Literal::Bool(_) => AsInstant::NotAnInstant,
    }
}

fn scale_whole(ms: i64, unit: TimeUnit) -> AsInstant {
    match unit {
        TimeUnit::Second if ms % 1_000 != 0 => AsInstant::Fractional,
        TimeUnit::Second => AsInstant::Raw(ms / 1_000),
        TimeUnit::Millisecond => AsInstant::Raw(ms),
        TimeUnit::Microsecond => ms.checked_mul(1_000).map_or(AsInstant::OutOfRange, AsInstant::Raw),
        TimeUnit::Nanosecond => {
            ms.checked_mul(1_000_000).map_or(AsInstant::OutOfRange, AsInstant::Raw)
        }
    }
}

fn scale_fractional(ms: f64, unit: TimeUnit) -> AsInstant {
    let scaled = match unit {
        TimeUnit::Second => ms / 1e3,
        TimeUnit::Millisecond => ms,
        TimeUnit::Microsecond => ms * 1e3,
        TimeUnit::Nanosecond => ms * 1e6,
    };
    // The upper bound is strict: `i64::MAX as f64` is 2^63, one above the maximum.
    let in_range = scaled >= i64::MIN as f64 && scaled < -(i64::MIN as f64);
    if !scaled.is_finite() || !in_range {
        AsInstant::OutOfRange
    } else if scaled.fract() != 0.0 {
        AsInstant::Fractional
    } else {
        AsInstant::Raw(scaled as i64)
    }
}

/// The unit as a refusal names it.
pub(super) fn unit_words(unit: TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Second => "seconds",
        TimeUnit::Millisecond => "milliseconds",
        TimeUnit::Microsecond => "microseconds",
        TimeUnit::Nanosecond => "nanoseconds",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_literal_is_read_from_each_json_kind_and_a_tagged_instant() {
        assert_eq!(Literal::of(&json!(5)), Some(Literal::Number(5.into())));
        assert_eq!(Literal::of(&json!("a")), Some(Literal::Text("a".into())));
        assert_eq!(Literal::of(&json!(true)), Some(Literal::Bool(true)));
        assert_eq!(Literal::of(&json!({"epoch_ms": 86_400_000})), Some(Literal::Instant(86_400_000)));
    }

    #[test]
    fn a_json_shape_that_is_no_literal_is_none() {
        for v in [
            json!(null),
            json!([1]),
            json!({}),
            json!({"epoch_ms": "x"}),
            json!({"epoch_ms": 1.5}),
            json!({"epoch_ms": 1, "other": 2}),
            json!({"other": 1}),
        ] {
            assert_eq!(Literal::of(&v), None, "{v}");
        }
    }

    #[test]
    fn a_literal_is_spelled_as_the_python_caller_wrote_it() {
        assert_eq!(Literal::Text("none".into()).spelled(), "'none'");
        assert_eq!(Literal::Bool(true).spelled(), "True");
        assert_eq!(Literal::Bool(false).spelled(), "False");
        assert_eq!(Literal::of(&json!(5)).map(|l| l.spelled()), Some("5".to_string()));
        assert_eq!(Literal::of(&json!(2.5)).map(|l| l.spelled()), Some("2.5".to_string()));
        assert_eq!(Literal::Instant(1_704_067_200_000).spelled(), "2024-01-01T00:00:00");
    }

    #[test]
    fn a_literal_names_its_kind() {
        assert_eq!(Literal::Number(1.into()).kind(), "a number");
        assert_eq!(Literal::Text(String::new()).kind(), "text");
        assert_eq!(Literal::Bool(false).kind(), "a boolean");
        assert_eq!(Literal::Instant(0).kind(), "a date or datetime");
    }

    fn raw(lit: &Literal, unit: TimeUnit) -> Option<i64> {
        match as_instant(lit, unit) {
            AsInstant::Raw(r) => Some(r),
            AsInstant::Fractional | AsInstant::OutOfRange | AsInstant::NotAnInstant => None,
        }
    }

    #[test]
    fn an_instant_and_a_bare_number_scale_to_every_unit_alike() {
        for lit in [Literal::Instant(1_704_067_200_000), Literal::Number(1_704_067_200_000_i64.into())] {
            assert_eq!(raw(&lit, TimeUnit::Second), Some(1_704_067_200));
            assert_eq!(raw(&lit, TimeUnit::Millisecond), Some(1_704_067_200_000));
            assert_eq!(raw(&lit, TimeUnit::Microsecond), Some(1_704_067_200_000_000));
            assert_eq!(raw(&lit, TimeUnit::Nanosecond), Some(1_704_067_200_000_000_000));
        }
    }

    #[test]
    fn a_whole_valued_float_scales_like_the_integer() {
        let lit = Literal::of(&json!(1_704_067_200_000.0)).unwrap();
        assert_eq!(raw(&lit, TimeUnit::Nanosecond), Some(1_704_067_200_000_000_000));
        assert_eq!(raw(&lit, TimeUnit::Second), Some(1_704_067_200));
    }

    #[test]
    fn an_instant_that_is_no_whole_number_of_the_unit_is_fractional() {
        // 1 ms is not a whole second
        assert!(matches!(as_instant(&Literal::Instant(1_001), TimeUnit::Second), AsInstant::Fractional));
        // a half millisecond is not a whole millisecond, but is a whole microsecond
        let half = Literal::of(&json!(1.5)).unwrap();
        assert!(matches!(as_instant(&half, TimeUnit::Millisecond), AsInstant::Fractional));
        assert_eq!(raw(&half, TimeUnit::Microsecond), Some(1_500));
        let tiny = Literal::of(&json!(1500.5)).unwrap();
        assert!(matches!(as_instant(&tiny, TimeUnit::Second), AsInstant::Fractional));
    }

    #[test]
    fn an_instant_beyond_i64_at_the_unit_is_out_of_range() {
        assert!(matches!(
            as_instant(&Literal::Instant(i64::MAX / 10), TimeUnit::Nanosecond),
            AsInstant::OutOfRange
        ));
        assert!(matches!(
            as_instant(&Literal::Instant(i64::MAX), TimeUnit::Microsecond),
            AsInstant::OutOfRange
        ));
        let huge = Literal::of(&json!(1e300)).unwrap();
        assert!(matches!(as_instant(&huge, TimeUnit::Millisecond), AsInstant::OutOfRange));
        assert!(matches!(as_instant(&huge, TimeUnit::Second), AsInstant::OutOfRange));
    }

    #[test]
    fn an_instant_is_spelled_as_iso_and_a_number_as_the_epoch_milliseconds_it_is_read_as() {
        assert_eq!(Literal::Instant(1_704_067_200_000).spelled_instant(), "2024-01-01T00:00:00");
        assert_eq!(Literal::of(&json!(1500)).unwrap().spelled_instant(), "1500 epoch milliseconds");
        assert_eq!(Literal::of(&json!(1.5)).unwrap().spelled_instant(), "1.5 epoch milliseconds");
    }

    #[test]
    fn text_and_a_bool_are_no_instant() {
        assert!(matches!(as_instant(&Literal::Text("a".into()), TimeUnit::Millisecond), AsInstant::NotAnInstant));
        assert!(matches!(as_instant(&Literal::Bool(true), TimeUnit::Millisecond), AsInstant::NotAnInstant));
    }
}
