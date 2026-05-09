use serde::{Deserialize, Serialize};

use crate::spec::data_ref::DataRef;
use crate::spec::encoding::Encoding;
use crate::spec::mark::Mark;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::str::FromStr;

use crate::spec::encoding::EncodingSpec;

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartSpec {
    #[serde(default)]
    pub data: DataRef,
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
}

#[pymethods]
impl ChartSpec {
    #[new]
    #[pyo3(signature = (*, mark, x = None, y = None, data = None))]
    fn new(
        mark: &str,
        x: Option<&Bound<'_, PyAny>>,
        y: Option<&Bound<'_, PyAny>>,
        data: Option<&str>,
    ) -> PyResult<Self> {
        let mark = Mark::from_str(mark)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let x = x.map(coerce_encoding).transpose()?;
        let y = y.map(coerce_encoding).transpose()?;

        let data = match data {
            None => DataRef::default(),
            Some(name) if name.is_empty() => {
                return Err(PyValueError::new_err("data name must be non-empty"))
            }
            Some(name) => DataRef::Named { name: name.to_string() },
        };

        Ok(ChartSpec {
            data,
            mark,
            encoding: Encoding { x, y },
        })
    }

    #[getter]
    fn mark(&self) -> &'static str {
        self.mark.as_str()
    }

    #[getter]
    fn x(&self) -> Option<EncodingSpec> {
        self.encoding.x.clone()
    }

    #[getter]
    fn y(&self) -> Option<EncodingSpec> {
        self.encoding.y.clone()
    }

    #[getter]
    fn data(&self) -> &str {
        match &self.data {
            DataRef::Named { name } => name,
        }
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[classmethod]
    fn from_json<'py>(_cls: &Bound<'py, PyType>, s: &str) -> PyResult<Self> {
        serde_json::from_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        let mark = self.mark.as_str();
        let data = match &self.data {
            DataRef::Named { name } => name.as_str(),
        };
        let x = match &self.encoding.x {
            None => "None".to_string(),
            Some(e) => format!("EncodingSpec(field='{}')", e.field),
        };
        let y = match &self.encoding.y {
            None => "None".to_string(),
            Some(e) => format!("EncodingSpec(field='{}')", e.field),
        };
        format!("ChartSpec(mark='{mark}', x={x}, y={y}, data='{data}')")
    }
}

fn coerce_encoding(obj: &Bound<'_, PyAny>) -> PyResult<EncodingSpec> {
    if let Ok(s) = obj.extract::<String>() {
        if s.is_empty() {
            return Err(PyValueError::new_err("encoding field name must be non-empty"));
        }
        return Ok(EncodingSpec { field: s, type_: None });
    }
    if let Ok(spec) = obj.extract::<EncodingSpec>() {
        return Ok(spec);
    }
    Err(PyTypeError::new_err(
        "expected str or EncodingSpec for encoding channel",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::encoding::{DataType, EncodingSpec};

    fn minimal_scatter() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "price".into(), type_: None }),
                y: Some(EncodingSpec {
                    field: "weight".into(),
                    type_: Some(DataType::Quantitative),
                }),
            },
        }
    }

    #[test]
    fn test_chart_spec_round_trip_minimal() {
        let original = minimal_scatter();
        let json = serde_json::to_string(&original).unwrap();
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_chart_spec_round_trip_idempotent_json() {
        let original = minimal_scatter();
        let json1 = serde_json::to_string(&original).unwrap();
        let parsed: ChartSpec = serde_json::from_str(&json1).unwrap();
        let json2 = serde_json::to_string(&parsed).unwrap();
        assert_eq!(json1, json2, "two-pass JSON differed");
    }

    #[test]
    fn test_chart_spec_round_trip_each_mark_variant() {
        for m in [
            Mark::Point, Mark::Line, Mark::Bar, Mark::Area,
            Mark::Rule, Mark::Text, Mark::Tick, Mark::Rect,
        ] {
            let mut spec = minimal_scatter();
            spec.mark = m;
            let json = serde_json::to_string(&spec).unwrap();
            let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, spec, "round-trip failed for {m:?}");
        }
    }

    #[test]
    fn test_data_ref_defaults_when_omitted() {
        let json = r#"{"mark":"point","encoding":{}}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.data, DataRef::Named { name: "default".into() });
    }

    #[test]
    fn test_unknown_mark_in_json_errors() {
        let json = r#"{"data":{"kind":"named","name":"d"},"mark":"spaghetti","encoding":{}}"#;
        let err = serde_json::from_str::<ChartSpec>(json).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("spaghetti") || msg.contains("variant"), "msg: {msg}");
    }

    #[test]
    fn test_missing_required_field_errors() {
        let json = r#"{"encoding":{}}"#;
        let err = serde_json::from_str::<ChartSpec>(json).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("mark"), "expected 'mark' in error, got: {msg}");
    }

    #[test]
    fn test_unknown_field_silently_dropped() {
        let json = r#"{"mark":"point","encoding":{},"future_field":42}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.mark, Mark::Point);
    }

    #[test]
    fn test_canonical_json_shape() {
        let spec = ChartSpec {
            data: DataRef::Named { name: "default".into() },
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "price".into(), type_: None }),
                y: Some(EncodingSpec {
                    field: "weight".into(),
                    type_: Some(DataType::Quantitative),
                }),
            },
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(
            json,
            r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"price"},"y":{"field":"weight","type":"quantitative"}}}"#,
        );
    }
}
