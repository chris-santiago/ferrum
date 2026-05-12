//! ReferenceLine transform — emits exactly two rows representing a line's
//! endpoints. Used by `calibration_chart` (and any chart that needs a static
//! reference line) so the diagonal layer can read from a named transform
//! output via `data_source` rather than iterating every row of the primary
//! batch and accidentally drawing overlapping lines per group.
//!
//! Output schema: `[x_field (Float64), y_field (Float64)]`.

use arrow::array::{ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReferenceLineSpec {
    /// Output column name for the x axis.
    pub x_field: String,
    /// Output column name for the y axis.
    pub y_field: String,
    /// Endpoint x values: `(start, end)`.
    pub x: (f64, f64),
    /// Endpoint y values: `(start, end)`.
    pub y: (f64, f64),
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &ReferenceLineSpec, _batch: &RecordBatch) -> PyResult<RecordBatch> {
    let out_schema = Arc::new(Schema::new(vec![
        Field::new(&spec.x_field, DataType::Float64, false),
        Field::new(&spec.y_field, DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![spec.x.0, spec.x.1])),
        Arc::new(Float64Array::from(vec![spec.y.0, spec.y.1])),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_reference_line: {e}")))
}

use crate::transform::core::TransformSpec;

/// Static-line transform — emits two endpoint rows for a reference line.
///
/// The output schema names are configurable so the layer that consumes this
/// transform (via ``data_source``) can encode the same column names as the
/// primary mark. The two rows represent the start and end of the line; the
/// downstream ``mark_line`` connects them.
///
/// Used internally by ``calibration_chart`` to draw the y=x reliability
/// diagonal as a proper Phase 8a layer (named transform → ``data_source``)
/// instead of relying on the diagonal layer iterating every row of the
/// primary calibration data.
///
/// Parameters
/// ----------
/// x_field : str
///     Output column name for the x-axis values.
/// y_field : str
///     Output column name for the y-axis values.
/// x : (float, float)
///     ``(start, end)`` x-coordinate of the reference line.
/// y : (float, float)
///     ``(start, end)`` y-coordinate of the reference line.
/// name : str, optional
///     Named output label so a sibling layer can target this transform's
///     output via ``Layer.data_source``.
#[pyclass(eq, module = "ferrum._core", name = "ReferenceLine")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyReferenceLine(pub(crate) TransformSpec);

#[pymethods]
impl PyReferenceLine {
    #[new]
    #[pyo3(signature = (x_field, y_field, *, x = (0.0, 1.0), y = (0.0, 1.0), name = None))]
    fn new(
        x_field: &str,
        y_field: &str,
        x: (f64, f64),
        y: (f64, f64),
        name: Option<String>,
    ) -> PyResult<Self> {
        if x_field.is_empty() || y_field.is_empty() {
            return Err(PyValueError::new_err(
                "ReferenceLine: x_field and y_field must be non-empty",
            ));
        }
        if !x.0.is_finite() || !x.1.is_finite() || !y.0.is_finite() || !y.1.is_finite() {
            return Err(PyValueError::new_err(
                "ReferenceLine: endpoint coordinates must be finite",
            ));
        }
        Ok(PyReferenceLine(TransformSpec::ReferenceLine(ReferenceLineSpec {
            x_field: x_field.to_string(),
            y_field: y_field.to_string(),
            x,
            y,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::ReferenceLine(s) => format!(
                "ReferenceLine(x_field='{}', y_field='{}', x={:?}, y={:?}, name={:?})",
                s.x_field, s.y_field, s.x, s.y, s.name
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    fn empty_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("_", DataType::Float64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(Vec::<f64>::new()))]).unwrap()
    }

    #[test]
    fn reference_line_emits_two_rows_with_named_columns() {
        let spec = ReferenceLineSpec {
            x_field: "fpr".into(),
            y_field: "tpr".into(),
            x: (0.0, 1.0),
            y: (0.0, 1.0),
            name: Some("roc_ref".into()),
        };
        let out = apply(&spec, &empty_batch()).unwrap();
        assert_eq!(out.num_rows(), 2);
        assert_eq!(out.num_columns(), 2);
        assert_eq!(out.schema().field(0).name(), "fpr");
        assert_eq!(out.schema().field(1).name(), "tpr");
        let xs = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let ys = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(xs.value(0), 0.0);
        assert_eq!(xs.value(1), 1.0);
        assert_eq!(ys.value(0), 0.0);
        assert_eq!(ys.value(1), 1.0);
    }

    #[test]
    fn reference_line_asymmetric_endpoints() {
        let spec = ReferenceLineSpec {
            x_field: "x".into(),
            y_field: "y".into(),
            x: (-2.0, 2.0),
            y: (5.0, -3.0),
            name: None,
        };
        let out = apply(&spec, &empty_batch()).unwrap();
        let ys = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ys.value(0), 5.0);
        assert_eq!(ys.value(1), -3.0);
    }

    #[test]
    fn reference_line_round_trip_json() {
        let original = ReferenceLineSpec {
            x_field: "x".into(),
            y_field: "y".into(),
            x: (0.0, 1.0),
            y: (0.0, 1.0),
            name: Some("ref".into()),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: ReferenceLineSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }
}
