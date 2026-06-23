//! Data transform: TimeUnit — extract temporal unit from a datetime field.
//!
//! Extracts year, month, day, hour, minute, second, day_of_week, week,
//! quarter from a timestamp column and outputs a new Float64 column.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, TimestampMillisecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

    // Handle Float64 columns (epoch millis) or Timestamp columns.
    let millis: Vec<Option<i64>> = extract_millis(col.as_ref(), schema.field(col_idx).data_type())?;

    // Extract the unit.
    let values: Vec<f64> = millis
        .iter()
        .map(|opt_ms| match opt_ms {
            Some(ms) => extract_unit(*ms, &spec.unit),
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

fn extract_millis(col: &dyn Array, dtype: &DataType) -> PyResult<Vec<Option<i64>>> {
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
        DataType::Float64 => {
            // Treat as epoch millis.
            let arr = col.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                PyValueError::new_err("data_timeunit: expected Float64Array")
            })?;
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
        other => Err(PyValueError::new_err(format!(
            "data_timeunit: unsupported dtype {:?} for field; expected Timestamp, Float64, or Int64",
            other
        ))),
    }
}

/// Extract a temporal unit from millisecond epoch.
fn extract_unit(ms: i64, unit: &str) -> f64 {
    // Simple calendar decomposition from epoch millis.
    // This uses a simplified algorithm without timezone handling.
    let secs = ms / 1000;
    let days_since_epoch = secs.div_euclid(86400);
    let time_of_day = secs.rem_euclid(86400);

    match unit {
        "hour" => (time_of_day / 3600) as f64,
        "minute" => ((time_of_day % 3600) / 60) as f64,
        "second" => (time_of_day % 60) as f64,
        "day_of_week" => {
            // 1970-01-01 was Thursday (4). 0=Sunday convention.
            ((days_since_epoch + 4) % 7) as f64
        }
        "year" | "month" | "day" | "week" | "quarter" => {
            let (y, m, d) = days_to_ymd(days_since_epoch);
            match unit {
                "year" => y as f64,
                "month" => m as f64,
                "day" => d as f64,
                "quarter" => ((m - 1) / 3 + 1) as f64,
                "week" => {
                    // ISO week approximation: day of year / 7 + 1.
                    let doy = day_of_year(y, m, d);
                    ((doy - 1) / 7 + 1) as f64
                }
                _ => f64::NAN,
            }
        }
        _ => f64::NAN,
    }
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
}
