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
///     Axis style overrides. Supported keys: ``grid``, ``title``,
///     ``labels``, ``ticks``, ``domain``, ``label_angle`` /
///     ``labelAngle``. Honored by the renderer.
/// legend : dict, optional
///     Legend style overrides. Supported keys: ``orient``, ``title``,
///     ``titleFontSize`` / ``title_font_size``, ``disabled``. Honored
///     by the renderer.
/// sort : dict or str, optional
///     Sort order for ordinal/nominal scales. Accepts ``"ascending"``,
///     ``"descending"``, or an explicit array of domain values. Honored
///     by the renderer.
/// stack : str, optional
///     Stack method for bar/area marks. Accepts ``"zero"``,
///     ``"normalize"``, or ``"center"``. Honored by the renderer.
/// impute : dict, optional
///     Imputation strategy. Accepts ``{"value": N}`` to fill missing
///     group×x combinations with constant *N*. Honored by the renderer.
/// scheme : str, optional
///     Color scheme name for quantitative color encodings (e.g. ``"viridis"``).
///     Honored by the renderer via ``scale_resolve``.
/// format : str, optional
///     Tick/label format string. Applied to axis tick labels for x/y
///     encodings and to text mark labels. Honored by the renderer.
/// format_type : str, optional
///     Format type (e.g. ``"time"``). When set to ``"time"``, the
///     ``format`` string is interpreted as a date/time pattern. Honored
///     by the renderer for text mark labels.
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

    // Honored renderer fields (D7–D13 — consumed by prepare.rs / position.rs / scale_resolve.rs):
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<AxisSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legend: Option<LegendSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<serde_json::Value>,
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
        condition = None, impute = None, scheme = None, format = None, format_type = None,
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
        condition: Option<&Bound<'_, PyAny>>,
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
            condition: json_round(py, condition, "condition")?,
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

    /// Axis style overrides (honored: grid, title, labels, ticks, domain, label_angle).
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

    /// Legend style overrides (honored: orient, title, titleFontSize, disabled).
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

    /// Sort order for ordinal/nominal scales ("ascending", "descending", or explicit array).
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

    /// Stack method for bar/area marks ("zero", "normalize", "center").
    #[getter]
    fn stack(&self) -> Option<&str> {
        self.stack.as_deref()
    }

    /// Imputation strategy. {"value": N} fills missing group×x combinations with N.
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

    /// Tick/label format string. Applied to axis tick labels (x/y) and text mark labels.
    #[getter]
    fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    /// Format type string. When "time", format is applied as a date/time pattern.
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
    // Phase 10 gallery-defaults: tooltip field emitted as SVG <title> on each mark.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tooltip: Option<EncodingSpec>,
    // Multi-field tooltip support: when set, takes precedence over `tooltip`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tooltip_fields: Option<Vec<EncodingSpec>>,
    // Phase 10 gallery-defaults: href field wraps marks in SVG <a xlink:href=...>.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub href: Option<EncodingSpec>,
    // Phase 10 gallery-defaults: description field emits SVG <desc> for accessibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<EncodingSpec>,
    // Phase 11c: key channel for animated transitions — identifies marks
    // across data updates so the WASM renderer can lerp between old/new.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<EncodingSpec>,
}

impl Encoding {
    /// Inherit unset encoding channels from `parent`.
    ///
    /// For each of the 9 channels (x, y, color, size, shape, opacity, x2, y2, text):
    ///   - If this layer's channel is unset (`None`), adopt the parent's value.
    ///   - If this layer's channel is set with the same `field` as the parent
    ///     and has no `scale` of its own, inherit the parent's scale spec.
    ///     This lets a chart-level explicit scale (domain/range/padding)
    ///     apply to every layer that references the same field, while
    ///     leaving layer-supplied scales untouched.
    ///
    /// Phase 10f: pre-F7 only x/y/color/size received the scale merge;
    /// shape/opacity/x2/y2/text fell through with no merge. F7 applies the
    /// merge uniformly so the policy is symmetric and predictable — the
    /// per-channel asymmetry was an undocumented accident.
    pub fn inherit_from(&mut self, parent: &Encoding) {
        fn inherit(
            child: &mut Option<EncodingSpec>,
            parent: &Option<EncodingSpec>,
        ) {
            match (child.as_mut(), parent.as_ref()) {
                (None, Some(_)) => { *child = parent.clone(); }
                (Some(c), Some(p)) if c.field == p.field => {
                    if c.scale.is_none() && p.scale.is_some() {
                        c.scale = p.scale.clone();
                    }
                    if c.title.is_none() && p.title.is_some() {
                        c.title = p.title.clone();
                    }
                    if c.scheme.is_none() && p.scheme.is_some() {
                        c.scheme = p.scheme.clone();
                    }
                    if c.type_.is_none() && p.type_.is_some() {
                        c.type_ = p.type_;
                    }
                    if c.axis.is_none() && p.axis.is_some() {
                        c.axis = p.axis.clone();
                    }
                    if c.legend.is_none() && p.legend.is_some() {
                        c.legend = p.legend.clone();
                    }
                    if c.format.is_none() && p.format.is_some() {
                        c.format = p.format.clone();
                    }
                    if c.format_type.is_none() && p.format_type.is_some() {
                        c.format_type = p.format_type.clone();
                    }
                }
                _ => {}
            }
        }
        inherit(&mut self.x, &parent.x);
        inherit(&mut self.y, &parent.y);
        inherit(&mut self.color, &parent.color);
        inherit(&mut self.size, &parent.size);
        inherit(&mut self.shape, &parent.shape);
        inherit(&mut self.opacity, &parent.opacity);
        inherit(&mut self.x2, &parent.x2);
        inherit(&mut self.y2, &parent.y2);
        inherit(&mut self.text, &parent.text);
        inherit(&mut self.tooltip, &parent.tooltip);
        if self.tooltip_fields.is_none() && parent.tooltip_fields.is_some() {
            self.tooltip_fields = parent.tooltip_fields.clone();
        }
        inherit(&mut self.href, &parent.href);
        inherit(&mut self.description, &parent.description);
        inherit(&mut self.key, &parent.key);
    }

    /// Overlay channels from `overlay` onto `self`.
    ///
    /// For each of the 12 channels: if `overlay.{channel}.is_some()`,
    /// replace `self.{channel}` with `overlay`'s value. Channels where
    /// `overlay` is `None` are left untouched.
    ///
    /// This is the semantic inverse of [`inherit_from`](Self::inherit_from):
    /// `inherit_from` fills gaps (child inherits absent channels from
    /// parent), while `overlay_from` replaces present channels (overlay
    /// wins when `Some`).
    pub fn overlay_from(&mut self, overlay: &Encoding) {
        macro_rules! ov {
            ($($ch:ident),*) => {
                $( if overlay.$ch.is_some() { self.$ch = overlay.$ch.clone(); } )*
            };
        }
        ov!(x, y, color, size, shape, opacity, x2, y2, text, tooltip, tooltip_fields, href, description, key);
    }
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

    #[test]
    fn inherit_from_propagates_title_on_same_field() {
        let parent = Encoding {
            x: Some(EncodingSpec {
                field: "fpr".into(),
                title: Some("False Positive Rate".into()),
                ..Default::default()
            }),
            y: Some(EncodingSpec {
                field: "tpr".into(),
                title: Some("True Positive Rate".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut child = Encoding {
            x: Some(EncodingSpec { field: "fpr".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "tpr".into(), ..Default::default() }),
            ..Default::default()
        };
        child.inherit_from(&parent);
        assert_eq!(child.x.as_ref().unwrap().title.as_deref(), Some("False Positive Rate"));
        assert_eq!(child.y.as_ref().unwrap().title.as_deref(), Some("True Positive Rate"));
    }

    #[test]
    fn inherit_from_propagates_scheme_on_same_field() {
        let parent = Encoding {
            color: Some(EncodingSpec {
                field: "species".into(),
                scheme: Some("paper_ink".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut child = Encoding {
            color: Some(EncodingSpec { field: "species".into(), ..Default::default() }),
            ..Default::default()
        };
        child.inherit_from(&parent);
        assert_eq!(child.color.as_ref().unwrap().scheme.as_deref(), Some("paper_ink"));
    }

    #[test]
    fn inherit_from_does_not_overwrite_child_title() {
        let parent = Encoding {
            x: Some(EncodingSpec {
                field: "fpr".into(),
                title: Some("Parent Title".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut child = Encoding {
            x: Some(EncodingSpec {
                field: "fpr".into(),
                title: Some("Child Title".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        child.inherit_from(&parent);
        assert_eq!(child.x.as_ref().unwrap().title.as_deref(), Some("Child Title"));
    }

    #[test]
    fn inherit_from_does_not_cross_pollinate_different_fields() {
        let parent = Encoding {
            x: Some(EncodingSpec {
                field: "fpr".into(),
                title: Some("False Positive Rate".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut child = Encoding {
            x: Some(EncodingSpec { field: "other_field".into(), ..Default::default() }),
            ..Default::default()
        };
        child.inherit_from(&parent);
        assert_eq!(child.x.as_ref().unwrap().title, None);
    }

    #[test]
    fn overlay_from_replaces_present_channels() {
        let mut base = Encoding::default();
        base.x = Some(EncodingSpec { field: "base_x".into(), ..Default::default() });
        base.y = Some(EncodingSpec { field: "base_y".into(), ..Default::default() });

        let mut overlay = Encoding::default();
        overlay.x = Some(EncodingSpec { field: "overlay_x".into(), ..Default::default() });
        // overlay.y is None — should NOT replace base.y

        base.overlay_from(&overlay);
        assert_eq!(base.x.as_ref().unwrap().field, "overlay_x");
        assert_eq!(base.y.as_ref().unwrap().field, "base_y");
    }

    #[test]
    fn overlay_from_covers_all_12_channels() {
        let mut base = Encoding {
            x: Some(EncodingSpec { field: "bx".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "by".into(), ..Default::default() }),
            color: Some(EncodingSpec { field: "bc".into(), ..Default::default() }),
            size: Some(EncodingSpec { field: "bs".into(), ..Default::default() }),
            shape: Some(EncodingSpec { field: "bsh".into(), ..Default::default() }),
            opacity: Some(EncodingSpec { field: "bo".into(), ..Default::default() }),
            x2: Some(EncodingSpec { field: "bx2".into(), ..Default::default() }),
            y2: Some(EncodingSpec { field: "by2".into(), ..Default::default() }),
            text: Some(EncodingSpec { field: "bt".into(), ..Default::default() }),
            tooltip: Some(EncodingSpec { field: "btt".into(), ..Default::default() }),
            tooltip_fields: None,
            href: Some(EncodingSpec { field: "bh".into(), ..Default::default() }),
            description: Some(EncodingSpec { field: "bd".into(), ..Default::default() }),
            key: Some(EncodingSpec { field: "bk".into(), ..Default::default() }),
        };
        // Overlay only tooltip, href, description (the three that were missed
        // by the old inline merge).
        let overlay = Encoding {
            tooltip: Some(EncodingSpec { field: "ott".into(), ..Default::default() }),
            href: Some(EncodingSpec { field: "oh".into(), ..Default::default() }),
            description: Some(EncodingSpec { field: "od".into(), ..Default::default() }),
            ..Default::default()
        };
        base.overlay_from(&overlay);
        // Replaced channels:
        assert_eq!(base.tooltip.as_ref().unwrap().field, "ott");
        assert_eq!(base.href.as_ref().unwrap().field, "oh");
        assert_eq!(base.description.as_ref().unwrap().field, "od");
        // Untouched channels:
        assert_eq!(base.x.as_ref().unwrap().field, "bx");
        assert_eq!(base.y.as_ref().unwrap().field, "by");
        assert_eq!(base.color.as_ref().unwrap().field, "bc");
    }
}
