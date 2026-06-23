use serde::{Deserialize, Serialize};

use crate::parameter::{ParamBinding, ParameterSpec};
use crate::types::Color;

// ── 3.7 SelectionSpec ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SelectionSpec {
    Point {
        name: String,
        fields: Option<Vec<String>>,
        encodings: Option<Vec<ChannelName>>,
        nearest: bool,
        toggle: EventExpr,
        on: EventExpr,
        clear: EventExpr,
        resolve: SelectionResolve,
    },
    Interval {
        name: String,
        fields: Option<Vec<String>>,
        encodings: Option<Vec<ChannelName>>,
        translate: bool,
        zoom: bool,
        mark: Option<SelectionMarkStyle>,
        resolve: SelectionResolve,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SelectionResolve {
    Global,
    Union,
    Intersect,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChannelName {
    X,
    Y,
    Color,
    Size,
    Shape,
    Opacity,
    StrokeWidth,
    StrokeOpacity,
    StrokeDash,
    FillOpacity,
    Angle,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectionMarkStyle {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub fill_opacity: f64,
    pub stroke_opacity: f64,
    pub stroke_width: f64,
    pub stroke_dash: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EventExpr {
    Click,
    Mouseout,
    Mouseover,
    ShiftKey,
    Dblclick,
    Custom(String),
}

// ── 3.8 ConditionalEncoding ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConditionalEncoding {
    pub selection_name: String,
    pub channel: ChannelName,
    pub if_selected: EncodingValue,
    pub if_not: EncodingValue,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EncodingValue {
    Color { value: Color },
    Opacity { value: f64 },
    Size { value: f64 },
    StrokeWidth { value: f64 },
    StrokeDash { value: Vec<f64> },
    StrokeOpacity { value: f64 },
    FillOpacity { value: f64 },
    Angle { value: f64 },
    Field { name: String },
}

impl EncodingValue {
    /// The encoding channel this value targets.
    ///
    /// `EncodingValue` and [`ChannelName`] are parallel axes: each value variant
    /// can only affect one channel (a `Color` value targets `Color`, an
    /// `Opacity` value targets `Opacity`, …). This method makes the value the
    /// single source of truth for that mapping so consumers dispatch on the
    /// value alone instead of cross-checking a separately-stored channel.
    ///
    /// `Field` carries no concrete visual channel (it names a data field rather
    /// than a literal value); it maps to [`ChannelName::Color`] as the
    /// conventional default, matching the Python serializer's fallback.
    pub fn channel(&self) -> ChannelName {
        match self {
            EncodingValue::Color { .. } => ChannelName::Color,
            EncodingValue::Opacity { .. } => ChannelName::Opacity,
            EncodingValue::Size { .. } => ChannelName::Size,
            EncodingValue::StrokeWidth { .. } => ChannelName::StrokeWidth,
            EncodingValue::StrokeDash { .. } => ChannelName::StrokeDash,
            EncodingValue::StrokeOpacity { .. } => ChannelName::StrokeOpacity,
            EncodingValue::FillOpacity { .. } => ChannelName::FillOpacity,
            EncodingValue::Angle { .. } => ChannelName::Angle,
            EncodingValue::Field { .. } => ChannelName::Color,
        }
    }
}

// ── 3.8b FieldValue ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FieldValue {
    String { value: String },
    Number { value: f64 },
    Bool { value: bool },
    Null,
}

// ── 3.9 InteractionConfig ───────────────────────────────────────────

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionConfig {
    pub zoom_enabled: bool,
    pub pan_enabled: bool,
    pub conditionals: Vec<ConditionalEncoding>,
    pub linked_panels: Vec<Vec<usize>>,
    pub tick_levels: Vec<PanelTickLevels>,
    #[serde(default = "default_true")]
    pub toolbar: bool,
    /// Declared reactive parameters (D6), carried through to the WASM runtime.
    /// Omitted from JSON when empty to keep param-free interaction configs
    /// byte-identical to their pre-D6 form.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<ParameterSpec>,
    /// Param→scene bindings (D6): which panel/scale each declared parameter
    /// drives, so the WASM runtime can route live updates. Omitted from JSON
    /// when empty to keep param-free interaction configs byte-identical.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub param_bindings: Vec<ParamBinding>,
}

impl Default for InteractionConfig {
    fn default() -> Self {
        Self {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: Vec::new(),
            linked_panels: Vec::new(),
            tick_levels: Vec::new(),
            toolbar: true,
            params: Vec::new(),
            param_bindings: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PanelTickLevels {
    pub panel_id: usize,
    pub x_levels: Vec<TickLevel>,
    pub y_levels: Vec<TickLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TickLevel {
    #[serde(
        serialize_with = "zoom_serde::serialize",
        deserialize_with = "zoom_serde::deserialize"
    )]
    pub min_zoom: f64,
    #[serde(
        serialize_with = "zoom_serde::serialize",
        deserialize_with = "zoom_serde::deserialize"
    )]
    pub max_zoom: f64,
    pub ticks: Vec<Tick>,
}

/// Custom serde for zoom fields that may contain non-finite f64 values.
/// JSON has no representation for Infinity/NaN, so we serialize them as strings.
/// For backward compatibility, `null` deserializes as `f64::INFINITY`.
mod zoom_serde {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if value.is_infinite() && value.is_sign_positive() {
            serializer.serialize_str("Infinity")
        } else if value.is_infinite() && value.is_sign_negative() {
            serializer.serialize_str("-Infinity")
        } else if value.is_nan() {
            serializer.serialize_str("NaN")
        } else {
            serializer.serialize_f64(*value)
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum FloatOrString {
            Float(f64),
            Str(String),
            Null,
        }

        match FloatOrString::deserialize(deserializer)? {
            FloatOrString::Float(v) => Ok(v),
            FloatOrString::Str(s) => match s.as_str() {
                "Infinity" => Ok(f64::INFINITY),
                "-Infinity" => Ok(f64::NEG_INFINITY),
                "NaN" => Ok(f64::NAN),
                other => Err(serde::de::Error::custom(format!(
                    "unexpected string value for zoom field: {other:?}"
                ))),
            },
            // Backward compat: previously serde_json serialized INFINITY as null
            FloatOrString::Null => Ok(f64::INFINITY),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tick {
    pub value: f64,
    pub label: String,
    pub pixel: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter::{BindingRole, ParamBinding, ParamKind};

    #[test]
    fn interaction_config_round_trips_with_params() {
        let config = InteractionConfig {
            params: vec![ParameterSpec {
                name: "d".into(),
                kind: ParamKind::Variable,
                value: Some(serde_json::json!([0, 100])),
                bind: None,
                select: None,
            }],
            ..InteractionConfig::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"params\""), "non-empty params must serialize");
        let parsed: InteractionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, config);
        assert_eq!(parsed.params.len(), 1);
        assert_eq!(parsed.params[0].name, "d");
    }

    /// Byte-stability gate: a param-free config must omit the `params` key
    /// entirely so existing param-free interaction JSON stays byte-identical.
    #[test]
    fn interaction_config_omits_empty_params() {
        let json = serde_json::to_string(&InteractionConfig::default()).unwrap();
        assert!(!json.contains("params"), "empty params must be skipped: {json}");
    }

    /// Old JSON with no `params` key deserializes to an empty vec via the
    /// serde default.
    #[test]
    fn interaction_config_defaults_missing_params() {
        let json = r#"{"zoom_enabled":false,"pan_enabled":false,"conditionals":[],"linked_panels":[],"tick_levels":[],"toolbar":true}"#;
        let parsed: InteractionConfig = serde_json::from_str(json).unwrap();
        assert!(parsed.params.is_empty());
        assert!(parsed.param_bindings.is_empty());
    }

    #[test]
    fn interaction_config_round_trips_with_param_bindings() {
        let config = InteractionConfig {
            param_bindings: vec![ParamBinding {
                param: "d".into(),
                role: BindingRole::Domain,
                panel: Some(0),
                channel: Some("x".into()),
            }],
            ..InteractionConfig::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(
            json.contains("\"param_bindings\""),
            "non-empty param_bindings must serialize"
        );
        let parsed: InteractionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, config);
    }

    /// `EncodingValue::channel()` must map every value variant to its single
    /// target channel — the invariant that lets WASM dispatch on the value
    /// alone instead of trusting a separately-stored `channel` field.
    #[test]
    fn encoding_value_channel_maps_each_variant() {
        use crate::types::Color;
        assert_eq!(
            EncodingValue::Color { value: Color { r: 0, g: 0, b: 0, a: 255 } }.channel(),
            ChannelName::Color
        );
        assert_eq!(EncodingValue::Opacity { value: 0.5 }.channel(), ChannelName::Opacity);
        assert_eq!(EncodingValue::Size { value: 10.0 }.channel(), ChannelName::Size);
        assert_eq!(
            EncodingValue::StrokeWidth { value: 2.0 }.channel(),
            ChannelName::StrokeWidth
        );
        assert_eq!(
            EncodingValue::StrokeDash { value: vec![6.0, 3.0] }.channel(),
            ChannelName::StrokeDash
        );
        assert_eq!(
            EncodingValue::StrokeOpacity { value: 0.3 }.channel(),
            ChannelName::StrokeOpacity
        );
        assert_eq!(
            EncodingValue::FillOpacity { value: 0.7 }.channel(),
            ChannelName::FillOpacity
        );
        assert_eq!(EncodingValue::Angle { value: 45.0 }.channel(), ChannelName::Angle);
        // Field carries no literal channel; it defaults to Color.
        assert_eq!(
            EncodingValue::Field { name: "g".into() }.channel(),
            ChannelName::Color
        );
    }

    /// Byte-stability gate: a param-free config must omit `param_bindings`.
    #[test]
    fn interaction_config_omits_empty_param_bindings() {
        let json = serde_json::to_string(&InteractionConfig::default()).unwrap();
        assert!(
            !json.contains("param_bindings"),
            "empty param_bindings must be skipped: {json}"
        );
    }
}
