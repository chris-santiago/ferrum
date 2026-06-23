//! Outliers: per-group IQR row filter.
//! Output schema = input schema, filtered to outlier rows.

use arrow::array::{Array, BooleanArray, Float64Array, RecordBatch, StringArray};
use arrow::compute::filter_record_batch;
use arrow::datatypes::DataType;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutliersSpec {
    pub field: String,
    #[serde(default)]
    pub groupby: Vec<String>,
    pub extent: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &OutliersSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let n = batch.num_rows();
    if n == 0 {
        return Ok(batch.clone());
    }

    // Validate value field exists and is Float64.
    let schema = batch.schema();
    let val_idx = schema
        .index_of(&spec.field)
        .map_err(|_| PyValueError::new_err(format!(
            "stat_outliers: column '{}' not found", spec.field
        )))?;
    if schema.field(val_idx).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_outliers: column '{}' must be Float64", spec.field
        )));
    }
    let values_arr = batch
        .column(val_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err(format!(
            "stat_outliers: expected Float64Array for column '{}'", spec.field)))?;

    // Validate groupby columns.
    for g in &spec.groupby {
        let i = schema.index_of(g).map_err(|_| {
            PyValueError::new_err(format!("stat_outliers: groupby column '{}' not found", g))
        })?;
        let dt = schema.field(i).data_type();
        if dt != &DataType::Float64 && !matches!(dt, DataType::Utf8) {
            return Err(PyValueError::new_err(format!(
                "stat_outliers: groupby column '{}' must be Float64 or Utf8; got {:?}", g, dt
            )));
        }
    }

    // Build group buckets keyed by stringified group values.
    let groups: std::collections::HashMap<Vec<String>, Vec<usize>> = if spec.groupby.is_empty() {
        let mut m = std::collections::HashMap::new();
        m.insert(Vec::new(), (0..n).collect());
        m
    } else {
        let mut m: std::collections::HashMap<Vec<String>, Vec<usize>> =
            std::collections::HashMap::new();
        let arrs: Vec<(&dyn Array, DataType)> = spec
            .groupby
            .iter()
            .map(|g| {
                let i = schema.index_of(g).expect("invariant: groupby columns validated above");
                let dt = schema.field(i).data_type().clone();
                (batch.column(i).as_ref(), dt)
            })
            .collect();
        for row in 0..n {
            let mut key = Vec::with_capacity(arrs.len());
            for (arr, dt) in &arrs {
                match dt {
                    DataType::Float64 => {
                        let a = arr.as_any().downcast_ref::<Float64Array>()
                            .expect("invariant: dtype validated as Float64 above");
                        key.push(if a.is_null(row) {
                            "__null__".to_string()
                        } else {
                            a.value(row).to_bits().to_string()
                        });
                    }
                    DataType::Utf8 => {
                        let a = arr.as_any().downcast_ref::<StringArray>()
                            .expect("invariant: dtype validated as Utf8 above");
                        key.push(if a.is_null(row) {
                            "__null__".to_string()
                        } else {
                            a.value(row).to_string()
                        });
                    }
                    _ => unreachable!(),
                }
            }
            m.entry(key).or_default().push(row);
        }
        m
    };

    // Compute mask: true = outlier (keep).
    let mut mask = vec![false; n];
    for indices in groups.values() {
        // Collect non-NaN, non-null values for quartile computation.
        let group_values: Vec<f64> = indices
            .iter()
            .filter_map(|&i| {
                if values_arr.is_null(i) {
                    return None;
                }
                let v = values_arr.value(i);
                if v.is_nan() {
                    return None;
                }
                Some(v)
            })
            .collect();
        if group_values.len() < 2 {
            continue; // no IQR with <2 values
        }
        let (q1, q3) = quartiles_q1_q3(&group_values);
        let iqr = q3 - q1;
        let lo = q1 - spec.extent * iqr;
        let hi = q3 + spec.extent * iqr;
        for &i in indices {
            if values_arr.is_null(i) {
                continue;
            }
            let v = values_arr.value(i);
            if v.is_nan() {
                continue;
            }
            if v < lo || v > hi {
                mask[i] = true;
            }
        }
    }

    let bool_arr = BooleanArray::from(mask);
    filter_record_batch(batch, &bool_arr)
        .map_err(|e| PyValueError::new_err(format!("stat_outliers: filter failed: {e}")))
}

/// Type-7 quartile (linear interpolation) — matches numpy/scipy default.
fn quartiles_q1_q3(values: &[f64]) -> (f64, f64) {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let q = |p: f64| -> f64 {
        let h = (n - 1) as f64 * p;
        let lo = h.floor() as usize;
        let hi = h.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
        }
    };
    (q(0.25), q(0.75))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    fn batch(field: &str, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(field, DataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    #[test]
    fn no_outliers_in_uniform_distribution() {
        pyo3::Python::initialize();
        let b = batch("v", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let spec = OutliersSpec {
            field: "v".into(),
            groupby: vec![],
            extent: 1.5,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 0);
    }

    #[test]
    fn extreme_value_flagged_as_outlier() {
        pyo3::Python::initialize();
        let b = batch("v", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0]);
        let spec = OutliersSpec {
            field: "v".into(),
            groupby: vec![],
            extent: 1.5,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 1);
        let arr = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(arr.value(0), 1000.0);
    }

    #[test]
    fn outliers_spec_round_trip() {
        let s = OutliersSpec {
            field: "v".into(),
            groupby: vec!["g".into()],
            extent: 2.0,
            name: Some("o".into()),
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: OutliersSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
    }

    #[test]
    fn schema_preserved() {
        pyo3::Python::initialize();
        let b = batch("v", vec![1.0, 2.0, 100.0]);
        let spec = OutliersSpec {
            field: "v".into(),
            groupby: vec![],
            extent: 1.5,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.schema(), b.schema());
    }

    fn batch_g_v(groups: Vec<&str>, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(values)),
                Arc::new(StringArray::from(groups)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn outliers_per_group_iqr_isolates_outlier() {
        pyo3::Python::initialize();
        // Group A: uniform 1..=10, no outliers within group.
        // Group B: 100..=108 then 1000.0, last value is an extreme outlier within its group.
        let groups = vec![
            "A", "B", "A", "B", "A", "B", "A", "B", "A", "B",
            "A", "B", "A", "B", "A", "B", "A", "B", "A", "B",
        ];
        let values = vec![
            1.0, 100.0, 2.0, 101.0, 3.0, 102.0, 4.0, 103.0, 5.0, 104.0,
            6.0, 105.0, 7.0, 106.0, 8.0, 107.0, 9.0, 108.0, 10.0, 1000.0,
        ];
        let b = batch_g_v(groups, values);
        let spec = OutliersSpec {
            field: "v".into(),
            groupby: vec!["g".into()],
            extent: 1.5,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 1);
        let v_arr = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let g_arr = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(v_arr.value(0), 1000.0);
        assert_eq!(g_arr.value(0), "B");
    }
}

// PyO3 wrapper appended below
use pyo3::prelude::*;

/// IQR-based outlier detection transform.
///
/// Identifies observations that fall outside the Tukey fence
/// ``[Q1 - extent*IQR, Q3 + extent*IQR]`` and retains only those rows.
/// Used alongside ``BoxStats`` to draw the outlier scatter layer of a
/// box plot.
///
/// Output: the subset of input rows where ``field`` is outside the fence,
/// with all original columns preserved.
///
/// Parameters
/// ----------
/// field : str
///     Numeric column to test (must be Float64).
/// groupby : list of str, default []
///     Columns to group by; fences computed independently per group.
/// extent : float, default 1.5
///     Tukey fence multiplier (``IQR * extent``). Must be ≥ 0. Use 0 to
///     flag points outside the IQR itself.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_boxplot().encode(x="cut", y="price")
#[pyclass(eq, module = "ferrum._core", name = "Outliers")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyOutliers(pub(crate) crate::transform::core::TransformSpec);

#[pymethods]
impl PyOutliers {
    #[new]
    #[pyo3(signature = (field, *, groupby = vec![], extent = 1.5, name = None))]
    fn new(
        field: String,
        groupby: Vec<String>,
        extent: f64,
        name: Option<String>,
    ) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("Outliers: field must be non-empty"));
        }
        if !extent.is_finite() || extent < 0.0 {
            return Err(PyValueError::new_err(
                "Outliers: extent must be a non-negative finite number",
            ));
        }
        Ok(Self(crate::transform::core::TransformSpec::Outliers(
            OutliersSpec {
                field,
                groupby,
                extent,
                name,
            },
        )))
    }

    fn __repr__(&self) -> String {
        if let crate::transform::core::TransformSpec::Outliers(s) = &self.0 {
            format!(
                "Outliers(field='{}', groupby={:?}, extent={}, name={:?})",
                s.field, s.groupby, s.extent, s.name
            )
        } else {
            unreachable!()
        }
    }
}
