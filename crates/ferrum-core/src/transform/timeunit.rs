//! Data transform: TimeUnit — extract temporal unit from a datetime field.
//!
//! Extracts year, month, day, hour, minute, second, day_of_week, week,
//! quarter from a Timestamp column, or from any numeric column treated as
//! epoch milliseconds, and outputs a new Float64 column.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, TimestampMillisecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::numeric_util::{coerce_to_float64, is_numeric_dtype};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TimeUnitSpec {
    pub field: String,
    /// Unit to extract: year, month, day, hour, minute, second,
    /// day_of_week, week, quarter.
    pub unit: String,
    #[serde(default)]
    pub utc: bool,
    /// Output column name. Defaults to "{unit}_{field}".
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub as_: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &TimeUnitSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let col_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_timeunit: column '{}' not found", spec.field))
    })?;
    let col = batch.column(col_idx);

    // Validate the unit once, against extract_unit's own dispatch, so an
    // unrecognised name refuses by name instead of yielding an all-NaN column.
    if extract_unit(0, &spec.unit).is_none() {
        return Err(PyValueError::new_err(format!(
            "data_timeunit: unknown unit '{}'; expected {}",
            spec.unit,
            SUPPORTED_UNITS.join(" | ")
        )));
    }

    // Handle Timestamp columns, or any numeric column treated as epoch millis.
    let millis: Vec<Option<i64>> =
        extract_millis(col, schema.field(col_idx).data_type(), &spec.field)?;

    // Extract the unit. A null input timestamp yields a null output cell.
    let values: Vec<f64> = millis
        .iter()
        .map(|opt_ms| match opt_ms {
            Some(ms) => extract_unit(*ms, &spec.unit).expect("invariant: unit validated above"),
            None => f64::NAN,
        })
        .collect();

    let out_name = spec
        .as_
        .clone()
        .unwrap_or_else(|| format!("{}_{}", spec.unit, spec.field));

    // Build output: original columns + new unit column.
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new(&out_name, DataType::Float64, true));
    let out_schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns.push(Arc::new(Float64Array::from(values)));

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_timeunit: {e}")))
}

fn extract_millis(col: &ArrayRef, dtype: &DataType, field: &str) -> PyResult<Vec<Option<i64>>> {
    let n = col.len();
    // Timestamp types.
    match dtype {
        DataType::Timestamp(ArrowTimeUnit::Millisecond, _) => {
            let arr = col
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .ok_or_else(|| {
                    PyValueError::new_err("data_timeunit: expected TimestampMillisecondArray")
                })?;
            Ok((0..n)
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i))
                    }
                })
                .collect())
        }
        DataType::Int64 => {
            let arr = col
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .ok_or_else(|| {
                    PyValueError::new_err("data_timeunit: expected Int64Array")
                })?;
            Ok((0..n)
                .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
                .collect())
        }
        other if is_numeric_dtype(other) => {
            // Batch-D M1 policy: every remaining numeric dtype (Float32,
            // UInt64, Int32, ...) is treated as epoch millis via the shared
            // coercion helper. Int64 keeps its own arm above because arrow
            // already stores it at the right width with no cast needed (avoiding
            // precision loss above 2^53).
            let arr = coerce_to_float64(col, "data_timeunit", field)?;
            Ok((0..n)
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        let v = arr.value(i);
                        if v.is_nan() {
                            None
                        } else {
                            Some(v as i64)
                        }
                    }
                })
                .collect())
        }
        other => Err(PyValueError::new_err(format!(
            "data_timeunit: unsupported dtype {:?} for field; expected Timestamp or any numeric (int/uint/float)",
            other
        ))),
    }
}

/// Unit names [`extract_unit`] recognises, for error messages.
///
/// `every_supported_unit_is_recognised` pins one direction — every name listed
/// here is recognised by `extract_unit` — so the error message can never claim
/// a unit that doesn't actually work. It does not pin the converse: an arm
/// added to `extract_unit`'s dispatch without a matching entry here would
/// drift silently (the unit would work but the error message wouldn't offer
/// it). `extract_unit`'s own match is still the single source of truth for
/// *behavior*; this list only has to stay a subset of it.
const SUPPORTED_UNITS: &[&str] = &[
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "day_of_week",
    "hour",
    "minute",
    "second",
];

/// Extract a temporal unit from millisecond epoch.
///
/// Returns `None` for an unrecognised unit name. `apply` validates the spec's
/// unit against this once per batch, so an unknown unit refuses by name rather
/// than silently producing an all-NaN column.
fn extract_unit(ms: i64, unit: &str) -> Option<f64> {
    // Simple calendar decomposition from epoch millis.
    // This uses a simplified algorithm without timezone handling.
    let secs = ms / 1000;
    let days_since_epoch = secs.div_euclid(86400);
    let time_of_day = secs.rem_euclid(86400);
    let ymd = || days_to_ymd(days_since_epoch);

    Some(match unit {
        "hour" => (time_of_day / 3600) as f64,
        "minute" => ((time_of_day % 3600) / 60) as f64,
        "second" => (time_of_day % 60) as f64,
        "day_of_week" => {
            // 1970-01-01 was Thursday (4). 0=Sunday convention.
            ((days_since_epoch + 4) % 7) as f64
        }
        "year" => ymd().0 as f64,
        "month" => ymd().1 as f64,
        "day" => ymd().2 as f64,
        "quarter" => ((ymd().1 - 1) / 3 + 1) as f64,
        "week" => {
            // ISO week approximation: day of year / 7 + 1.
            let (y, m, d) = ymd();
            let doy = day_of_year(y, m, d);
            ((doy - 1) / 7 + 1) as f64
        }
        _ => return None,
    })
}

/// Convert days since epoch to (year, month, day).
fn days_to_ymd(days: i64) -> (i32, i32, i32) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64 + era * 400) as i32;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as i32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as i32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn day_of_year(y: i32, m: i32, d: i32) -> i32 {
    let is_leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    let days_in_months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut doy: i32 = d;
    for i in 1..m {
        doy += days_in_months[i as usize];
    }
    if is_leap && m > 2 {
        doy += 1;
    }
    doy
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "TimeUnit")]
#[derive(Debug, Clone)]
pub(crate) struct PyTimeUnit(pub(crate) TransformSpec);

#[pymethods]
impl PyTimeUnit {
    #[new]
    #[pyo3(signature = (field, unit, *, utc = false, as_ = None, name = None))]
    fn new(field: String, unit: String, utc: bool, as_: Option<String>, name: Option<String>) -> Self {
        PyTimeUnit(TransformSpec::TimeUnit(TimeUnitSpec {
            field,
            unit,
            utc,
            as_,
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::TimeUnit(s) => format!("TimeUnit(field='{}', unit='{}')", s.field, s.unit),
            _ => "TimeUnit(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn timeunit_extract_year_month() {
        // 2023-06-15 12:30:00 UTC in epoch millis.
        let ms: f64 = 1_686_831_000_000.0;
        let schema = Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![ms]))],
        )
        .unwrap();

        let spec = TimeUnitSpec {
            field: "ts".into(),
            unit: "year".into(),
            utc: true,
            as_: Some("yr".into()),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let yr = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(yr.value(0), 2023.0);

        let spec_month = TimeUnitSpec {
            field: "ts".into(),
            unit: "month".into(),
            utc: true,
            as_: Some("mo".into()),
            name: None,
        };
        let out2 = apply(&spec_month, &batch).unwrap();
        let mo = out2.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(mo.value(0), 6.0);
    }

    #[test]
    fn timeunit_extract_hour() {
        // 2023-01-01 14:30:00 UTC
        let ms: f64 = 1_672_583_400_000.0;
        let schema = Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![ms]))],
        )
        .unwrap();

        let spec = TimeUnitSpec {
            field: "ts".into(),
            unit: "hour".into(),
            utc: true,
            as_: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let h = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(h.value(0), 14.0);
    }

    // ---- Batch-D dtype policy (spec §4.1) ----
    //
    // extract_millis previously accepted only Timestamp, Float64, and Int64,
    // so an epoch-millis column of any other numeric dtype raised
    // "unsupported dtype" even though the M1 policy elsewhere in the sweep
    // widens every int/uint/float dtype. These pin the widened `other` arm.

    #[test]
    fn timeunit_extracts_year_from_float32_epoch_millis() {
        // 2023-06-15 12:30:00 UTC, narrowed to Float32 (loses some precision
        // in the millisecond digits but not enough to move the year/month).
        use arrow::array::Float32Array;
        let ms: f32 = 1_686_831_000_000.0;
        let schema = Arc::new(Schema::new(vec![Field::new("ts", DataType::Float32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Float32Array::from(vec![ms]))]).unwrap();
        let spec = TimeUnitSpec {
            field: "ts".into(),
            unit: "year".into(),
            utc: true,
            as_: Some("yr".into()),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let yr = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(yr.value(0), 2023.0);
    }

    #[test]
    fn timeunit_extracts_hour_from_uint64_epoch_millis() {
        // 2023-01-01 14:30:00 UTC as UInt64 epoch millis.
        use arrow::array::UInt64Array;
        let ms: u64 = 1_672_583_400_000;
        let schema = Arc::new(Schema::new(vec![Field::new("ts", DataType::UInt64, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(vec![ms]))]).unwrap();
        let spec = TimeUnitSpec {
            field: "ts".into(),
            unit: "hour".into(),
            utc: true,
            as_: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let h = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(h.value(0), 14.0);
    }

    #[test]
    fn timeunit_unknown_unit_raises_naming_it_and_the_accepted_set() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("ts", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1_686_831_000_000.0]))],
        )
        .unwrap();
        let spec = TimeUnitSpec {
            field: "ts".into(),
            unit: "fortnight".into(),
            utc: true,
            as_: None,
            name: None,
        };

        let msg = apply(&spec, &batch).unwrap_err().to_string();
        assert!(msg.contains("data_timeunit"), "err: {msg}");
        assert!(msg.contains("'fortnight'"), "err must name the unit: {msg}");
        assert!(msg.contains("day_of_week"), "err must list the accepted set: {msg}");
    }

    #[test]
    fn every_supported_unit_is_recognised() {
        // One-directional pin: every name the error message lists must
        // actually be handled by extract_unit. Does not catch the converse
        // (a unit extract_unit handles but SUPPORTED_UNITS omits) — see the
        // doc comment on SUPPORTED_UNITS.
        for unit in SUPPORTED_UNITS {
            assert!(
                extract_unit(1_686_831_000_000, unit).is_some(),
                "SUPPORTED_UNITS lists '{unit}' but extract_unit does not handle it"
            );
        }
    }
}
