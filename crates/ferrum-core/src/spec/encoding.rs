use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Quantitative,
    Nominal,
    Ordinal,
    Temporal,
}

impl DataType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataType::Quantitative => "quantitative",
            DataType::Nominal => "nominal",
            DataType::Ordinal => "ordinal",
            DataType::Temporal => "temporal",
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub struct ParseDataTypeError(pub String);

impl fmt::Display for ParseDataTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown data type '{}'; expected one of [Q, N, O, T, quantitative, nominal, ordinal, temporal]",
            self.0
        )
    }
}

impl std::error::Error for ParseDataTypeError {}

impl FromStr for DataType {
    type Err = ParseDataTypeError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Q" | "quantitative" => Ok(DataType::Quantitative),
            "N" | "nominal" => Ok(DataType::Nominal),
            "O" | "ordinal" => Ok(DataType::Ordinal),
            "T" | "temporal" => Ok(DataType::Temporal),
            other => Err(ParseDataTypeError(other.to_string())),
        }
    }
}

/// Scale override on an encoding channel. Honored by scale_resolve.rs in Phase 8a.
/// Mirrors the Python ScaleLog/ScalePow/etc. classes via tagged enum.
///
/// Uses `tag = "type"` (NOT the spec-module convention `tag = "kind"`) for Vega-Lite wire-format
/// alignment — see design spec §11 row 16 ("Vega-Lite interop stays open without translation").
/// This is the only tagged enum in this module that uses `"type"`; the choice is intentional.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ScaleSpec {
    Linear {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<f64>>,
        #[serde(default)]
        nice: bool,
        #[serde(default)]
        zero: bool,
        #[serde(default)]
        clamp: bool,
        /// Fractional inward pixel padding (0.0 = no padding). Themes-T4
        /// quantitative default is 0.05, applied at the renderer when
        /// `padding.is_none()` and `domain.is_none()`. User-specified
        /// `domain` suppresses the default to 0.0 unless `padding` is
        /// also set.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        padding: Option<f64>,
    },
    Log {
        #[serde(default = "default_log_base")]
        base: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<f64>>,
        #[serde(default)]
        nice: bool,
        #[serde(default)]
        clamp: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        padding: Option<f64>,
    },
    Time {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<f64>>,
        #[serde(default)]
        nice: bool,
        #[serde(default)]
        clamp: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        padding: Option<f64>,
    },
    Symlog {
        #[serde(default = "default_symlog_constant")]
        constant: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<f64>>,
        #[serde(default)]
        nice: bool,
        #[serde(default)]
        clamp: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        padding: Option<f64>,
    },
    Ordinal {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<f64>>,
        #[serde(default)]
        padding: f64,
    },
}

fn default_log_base() -> f64 {
    10.0
}
fn default_symlog_constant() -> f64 {
    1.0
}

/// Opaque-but-typed axis spec. Round-trips JSON; renderer ignores in 8a.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AxisSpec {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LegendSpec {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Encoding channel specification — maps a data field to a visual variable.
///
/// Created implicitly by Python's encoding channel classes (``X``, ``Y``,
/// ``Color``, ...). Carries the field name, optional inferred data type,
/// and optional scale/title overrides.
///
/// Parameters
/// ----------
/// field : str
///     Column name in the input DataFrame.
/// type_ : {"Q", "N", "O", "T", "quantitative", "nominal", "ordinal", \
///          "temporal"}, optional
///     Data type. Inferred from the column dtype when omitted.
/// scale : dict, optional
///     Scale override (e.g. ``{"type": "log"}``). Honored by the renderer.
/// title : str, optional
///     Axis or legend title. Overrides the auto-generated field name.
/// axis : dict, optional
///     Axis style overrides. **Reserved** — round-trips JSON but not yet
///     honored by the renderer in Phase 8a.
/// legend : dict, optional
///     Legend style overrides. **Reserved** — round-trips JSON but not yet
///     honored by the renderer in Phase 8a.
/// sort : dict or str, optional
///     Sort order. **Reserved** — round-trips JSON but not yet honored by
///     the renderer in Phase 8a.
/// stack : str, optional
///     Stack method. **Reserved** — round-trips JSON but not yet honored by
///     the renderer in Phase 8a.
/// impute : dict, optional
///     Imputation strategy. **Reserved** — round-trips JSON but not yet
///     honored by the renderer in Phase 8a.
/// scheme : str, optional
///     Color scheme name for quantitative color encodings (e.g. ``"viridis"``).
///     Honored by the renderer via ``scale_resolve``.
/// format : str, optional
///     Tick/label format string. **Reserved** — round-trips JSON but not yet
///     honored by the renderer.
/// format_type : str, optional
///     Format type (e.g. ``"time"``). **Reserved** — round-trips JSON but
///     not yet honored by the renderer.
///
/// Notes
/// -----
/// Users typically work with the higher-level encoding channel classes
/// from ``ferrum.encoding`` (``X``, ``Y``, ``Color``, ...);
/// ``EncodingSpec`` is the internal IR that ``Chart.encode(...)`` builds.
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EncodingSpec {
    pub field: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none", default)]
    pub type_: Option<DataType>,

    // NEW honored fields (Phase 8a):
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<ScaleSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    // NEW deferred fields (Phase 8a — round-trip + warn-once at Python layer):
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<AxisSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legend: Option<LegendSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impute: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(rename = "formatType", default, skip_serializing_if = "Option::is_none")]
    pub format_type: Option<String>,
}

impl EncodingSpec {
    pub(crate) fn repr_string(&self) -> String {
        match &self.type_ {
            None => format!("EncodingSpec(field='{}')", self.field),
            Some(t) => format!("EncodingSpec(field='{}', type_='{}')", self.field, t.as_str()),
        }
    }
}

#[pymethods]
impl EncodingSpec {
    #[new]
    #[pyo3(signature = (
        field, type_ = None, *,
        scale = None, title = None,
        axis = None, legend = None, sort = None, stack = None,
        impute = None, scheme = None, format = None, format_type = None,
    ))]
    fn new(
        py: Python,
        field: &str,
        type_: Option<&str>,
        scale: Option<&Bound<'_, PyAny>>,
        title: Option<String>,
        axis: Option<&Bound<'_, PyAny>>,
        legend: Option<&Bound<'_, PyAny>>,
        sort: Option<&Bound<'_, PyAny>>,
        stack: Option<String>,
        impute: Option<&Bound<'_, PyAny>>,
        scheme: Option<String>,
        format: Option<String>,
        format_type: Option<String>,
    ) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("field must be non-empty"));
        }
        let type_ = match type_ {
            Some(s) => Some(
                s.parse::<DataType>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            None => None,
        };

        fn json_round<T: for<'de> serde::Deserialize<'de>>(
            py: Python,
            obj: Option<&Bound<'_, PyAny>>,
            name: &str,
        ) -> PyResult<Option<T>> {
            let Some(o) = obj else { return Ok(None) };
            let json_module = py.import("json")?;
            let s: String = json_module.call_method1("dumps", (o,))?.extract()?;
            Ok(Some(
                serde_json::from_str(&s)
                    .map_err(|e| PyValueError::new_err(format!("{name}: {e}")))?,
            ))
        }

        Ok(EncodingSpec {
            field: field.to_string(),
            type_,
            scale: json_round(py, scale, "scale")?,
            title,
            axis: json_round(py, axis, "axis")?,
            legend: json_round(py, legend, "legend")?,
            sort: json_round(py, sort, "sort")?,
            stack,
            impute: json_round(py, impute, "impute")?,
            scheme,
            format,
            format_type,
        })
    }

    /// Column name in the input DataFrame.
    #[getter]
    fn field(&self) -> &str {
        &self.field
    }

    /// Data type string (``"quantitative"``, ``"nominal"``, ``"ordinal"``,
    /// ``"temporal"``), or ``None`` when inferred.
    #[getter]
    fn type_(&self) -> Option<&'static str> {
        self.type_.as_ref().map(|t| t.as_str())
    }

    /// Scale override dict, or ``None``.
    #[getter]
    fn scale(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        match &self.scale {
            None => Ok(None),
            Some(s) => {
                let json = serde_json::to_string(s)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let json_module = py.import("json")?;
                Ok(Some(json_module.call_method1("loads", (json,))?.unbind()))
            }
        }
    }

    /// Axis or legend title override, or ``None``.
    #[getter]
    fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Axis style overrides (reserved — not yet honored by renderer).
    #[getter]
    fn axis(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        match &self.axis {
            None => Ok(None),
            Some(s) => {
                let json = serde_json::to_string(s)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let json_module = py.import("json")?;
                Ok(Some(json_module.call_method1("loads", (json,))?.unbind()))
            }
        }
    }

    /// Legend style overrides (reserved — not yet honored by renderer).
    #[getter]
    fn legend(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        match &self.legend {
            None => Ok(None),
            Some(s) => {
                let json = serde_json::to_string(s)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let json_module = py.import("json")?;
                Ok(Some(json_module.call_method1("loads", (json,))?.unbind()))
            }
        }
    }

    /// Sort order (reserved — not yet honored by renderer).
    #[getter]
    fn sort(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        match &self.sort {
            None => Ok(None),
            Some(s) => {
                let json = serde_json::to_string(s)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let json_module = py.import("json")?;
                Ok(Some(json_module.call_method1("loads", (json,))?.unbind()))
            }
        }
    }

    /// Stack method (reserved — not yet honored by renderer).
    #[getter]
    fn stack(&self) -> Option<&str> {
        self.stack.as_deref()
    }

    /// Imputation strategy (reserved — not yet honored by renderer).
    #[getter]
    fn impute(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        match &self.impute {
            None => Ok(None),
            Some(s) => {
                let json = serde_json::to_string(s)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let json_module = py.import("json")?;
                Ok(Some(json_module.call_method1("loads", (json,))?.unbind()))
            }
        }
    }

    /// Color scheme name for quantitative encodings (e.g. ``"viridis"``).
    #[getter]
    fn scheme(&self) -> Option<&str> {
        self.scheme.as_deref()
    }

    /// Tick/label format string (reserved — not yet honored by renderer).
    #[getter]
    fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    /// Format type string (reserved — not yet honored by renderer).
    #[getter]
    fn format_type(&self) -> Option<&str> {
        self.format_type.as_deref()
    }

    /// Return a string representation of this encoding spec.
    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Encoding {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub y: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub color: Option<EncodingSpec>,
    // NEW Phase 8a:
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub size: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub shape: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub opacity: Option<EncodingSpec>,
    // NEW Phase 8b Task 22 (ribbon mark): paired-channel endpoints. x2 reserved for
    // future scale_resolve work in Task 36; ribbon drawer reads y2 directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x2: Option<EncodingSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y2: Option<EncodingSpec>,
    // Phase 10c: text channel for mark_text label content. When set, mark_text
    // reads this column for the rendered label; otherwise it falls back to
    // formatting the y value (legacy Phase 7 behavior).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<EncodingSpec>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_short_and_long_forms() {
        assert_eq!(DataType::from_str("Q").unwrap(), DataType::Quantitative);
        assert_eq!(DataType::from_str("quantitative").unwrap(), DataType::Quantitative);
        assert_eq!(DataType::from_str("N").unwrap(), DataType::Nominal);
        assert_eq!(DataType::from_str("nominal").unwrap(), DataType::Nominal);
    }

    #[test]
    fn test_data_type_unknown() {
        let err = DataType::from_str("Z").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("'Z'"), "msg: {msg}");
        assert!(msg.contains("quantitative"), "msg: {msg}");
    }

    #[test]
    fn test_data_type_serde_long_form() {
        assert_eq!(serde_json::to_string(&DataType::Quantitative).unwrap(), "\"quantitative\"");
    }

    #[test]
    fn test_encoding_spec_round_trip_no_type() {
        let original = EncodingSpec { field: "price".into(), type_: None, ..Default::default() };
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(json, r#"{"field":"price"}"#);
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_encoding_spec_round_trip_with_type() {
        let original = EncodingSpec {
            field: "weight".into(),
            type_: Some(DataType::Quantitative),
            ..Default::default()
        };
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(json, r#"{"field":"weight","type":"quantitative"}"#);
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_encoding_round_trip_both_axes() {
        let e = Encoding {
            x: Some(EncodingSpec { field: "price".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec {
                field: "weight".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            color: None,
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(
            json,
            r#"{"x":{"field":"price"},"y":{"field":"weight","type":"quantitative"}}"#,
        );
        let parsed: Encoding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn test_encoding_omits_none_fields() {
        let e = Encoding::default();
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_encoding_round_trip_with_color() {
        let e = Encoding {
            x: Some(EncodingSpec { field: "price".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "weight".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec {
                field: "species".into(),
                type_: Some(DataType::Nominal),
                ..Default::default()
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(
            json,
            r#"{"x":{"field":"price"},"y":{"field":"weight"},"color":{"field":"species","type":"nominal"}}"#,
        );
        let parsed: Encoding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn test_encoding_omits_color_when_none() {
        let e = Encoding {
            x: Some(EncodingSpec { field: "a".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "b".into(), type_: None, ..Default::default() }),
            color: None,
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, r#"{"x":{"field":"a"},"y":{"field":"b"}}"#);
    }

    // --- Phase 8a new tests ---

    #[test]
    fn encoding_spec_round_trips_with_scale() {
        let e = EncodingSpec {
            field: "price".into(),
            type_: Some(DataType::Quantitative),
            scale: Some(ScaleSpec::Log {
                base: 10.0,
                domain: None,
                range: None,
                nice: true,
                clamp: false,
                padding: None,
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""scale":{"type":"log""#));
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn encoding_spec_round_trips_with_title() {
        let e = EncodingSpec {
            field: "x".into(),
            type_: None,
            title: Some("My X Axis".into()),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""title":"My X Axis""#));
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn encoding_spec_round_trips_with_axis_opaque() {
        use serde_json::json;
        let mut axis_extra = serde_json::Map::new();
        axis_extra.insert("grid".into(), json!(false));
        axis_extra.insert("orient".into(), json!("bottom"));
        let e = EncodingSpec {
            field: "x".into(),
            type_: None,
            axis: Some(AxisSpec { extra: axis_extra }),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn encoding_spec_phase_7_canonical_json_byte_identical_when_no_new_fields() {
        let e = EncodingSpec { field: "x".into(), type_: None, ..Default::default() };
        assert_eq!(serde_json::to_string(&e).unwrap(), r#"{"field":"x"}"#);

        let e2 = EncodingSpec {
            field: "y".into(),
            type_: Some(DataType::Quantitative),
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_string(&e2).unwrap(),
            r#"{"field":"y","type":"quantitative"}"#,
        );
    }

    #[test]
    fn encoding_spec_round_trips_pre_phase_8_json() {
        // Existing JSON without any new fields must deserialize.
        let json = r#"{"field":"price","type":"quantitative"}"#;
        let parsed: EncodingSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.field, "price");
        assert_eq!(parsed.type_, Some(DataType::Quantitative));
        assert_eq!(parsed.scale, None);
        assert_eq!(parsed.title, None);
    }

    // --- Phase 8b Task 22: x2 / y2 channels (ribbon support) ---

    #[test]
    fn encoding_round_trips_with_y2() {
        let e = Encoding {
            x: Some(EncodingSpec { field: "t".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
            y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""y2":{"field":"hi"}"#), "json: {json}");
        let parsed: Encoding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn encoding_omits_x2_y2_when_none() {
        // Existing 8a JSON without x2/y2 must remain byte-identical.
        let e = Encoding {
            x: Some(EncodingSpec { field: "a".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "b".into(), type_: None, ..Default::default() }),
            ..Default::default()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, r#"{"x":{"field":"a"},"y":{"field":"b"}}"#);
        assert!(!json.contains("x2"));
        assert!(!json.contains("y2"));
    }

    #[test]
    fn scale_spec_log_default_base_is_10() {
        let json = r#"{"type":"log"}"#;
        let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
        match parsed {
            ScaleSpec::Log { base, .. } => assert_eq!(base, 10.0),
            _ => panic!("expected Log variant"),
        }
    }
}
