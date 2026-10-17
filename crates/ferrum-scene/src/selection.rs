use serde::{Deserialize, Serialize};

use crate::parameter::{ParamBinding, ParameterSpec};
use crate::types::Color;

// ── 3.7 SelectionSpec ───────────────────────────────────────────────

/// A selection as a chart declares it: the `ChartSpec.selections` wire.
/// The scene carries each one as a [`SceneSelection`], which adds the owner
/// panels the scene build stamps. An unknown key, `panels` among them, is
/// refused rather than dropped: no producer may set what the scene build
/// owns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
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

impl SelectionSpec {
    pub fn name(&self) -> &str {
        match self {
            SelectionSpec::Point { name, .. } | SelectionSpec::Interval { name, .. } => name,
        }
    }

    /// The data fields this selection matches marks by, when it names any:
    /// the runtime reads each from a mark's tooltip.
    pub fn fields(&self) -> Option<&[String]> {
        match self {
            SelectionSpec::Point { fields, .. } | SelectionSpec::Interval { fields, .. } => fields.as_deref(),
        }
    }
}

/// A selection as the scene carries it: the declared [`SelectionSpec`] and
/// its owner panels, every panel of the chart that declares it, every facet
/// panel and every layer of a layered chart included. Owners are
/// [`Panel::id`](crate::Panel)s, never array positions in
/// `SceneGraph.panels`; input handlers translate the position an event
/// arrives with before asking [`SceneSelection::takes_input_from`]. The
/// scene build stamps `panels`; `None` means every panel of the scene, the
/// form a one-panel scene carries, and is omitted on the wire, so such a
/// scene's selections serialize exactly as their declarations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneSelection {
    #[serde(flatten)]
    pub selection: SelectionSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panels: Option<Vec<usize>>,
}

impl SceneSelection {
    pub fn name(&self) -> &str {
        self.selection.name()
    }

    /// Whether an input event on `event_panels` (the `Panel.id`s of the
    /// panels a click or brush landed on) may update this selection: only input on one of its owner
    /// panels does. A selection with no owner list takes every input, even
    /// one outside every panel.
    pub fn takes_input_from(&self, event_panels: &[usize]) -> bool {
        self.panels
            .as_ref()
            .is_none_or(|owners| event_panels.iter().any(|p| owners.contains(p)))
    }
}

impl From<SelectionSpec> for SceneSelection {
    /// A selection owned by every panel of its scene.
    fn from(selection: SelectionSpec) -> Self {
        Self { selection, panels: None }
    }
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

impl ChannelName {
    /// The channel's name as Python's `encode(...)` writes it, the same
    /// string its wire form serializes to.
    pub fn as_str(self) -> &'static str {
        match self {
            ChannelName::X => "x",
            ChannelName::Y => "y",
            ChannelName::Color => "color",
            ChannelName::Size => "size",
            ChannelName::Shape => "shape",
            ChannelName::Opacity => "opacity",
            ChannelName::StrokeWidth => "stroke_width",
            ChannelName::StrokeOpacity => "stroke_opacity",
            ChannelName::StrokeDash => "stroke_dash",
            ChannelName::FillOpacity => "fill_opacity",
            ChannelName::Angle => "angle",
        }
    }
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

impl ConditionalEncoding {
    /// Whether either branch is a size: the conditionals a sized line's
    /// width answers to.
    pub fn sets_size(&self) -> bool {
        [&self.if_selected, &self.if_not].into_iter().any(|b| matches!(b, EncodingValue::Size { .. }))
    }
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
    /// Primary (slot 0 / left-axis) y tick levels.
    pub y_levels: Vec<TickLevel>,
    /// Per-secondary-slot y tick levels (secondary-y-axis, GH #52), one entry
    /// per independent-y layer's right axis in slot order (`[0]` = slot 1, the
    /// first right axis; `[1]` = slot 2; …). Each entry mirrors `y_levels`'
    /// zoom-breakpoint structure but is generated from that slot's own y-scale,
    /// so the runtime can recognize and reposition right-axis tick labels under
    /// zoom exactly as it does the left axis. Empty for single-axis charts —
    /// omitted from JSON so every pre-#52 tick-levels blob is byte-identical.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub y_slot_levels: Vec<Vec<TickLevel>>,
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
        assert!(
            json.contains("\"params\""),
            "non-empty params must serialize"
        );
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
        assert!(
            !json.contains("params"),
            "empty params must be skipped: {json}"
        );
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
                y_slot: 0,
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
            EncodingValue::Color {
                value: Color {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 255
                }
            }
            .channel(),
            ChannelName::Color
        );
        assert_eq!(
            EncodingValue::Opacity { value: 0.5 }.channel(),
            ChannelName::Opacity
        );
        assert_eq!(
            EncodingValue::Size { value: 10.0 }.channel(),
            ChannelName::Size
        );
        assert_eq!(
            EncodingValue::StrokeWidth { value: 2.0 }.channel(),
            ChannelName::StrokeWidth
        );
        assert_eq!(
            EncodingValue::StrokeDash {
                value: vec![6.0, 3.0]
            }
            .channel(),
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
        assert_eq!(
            EncodingValue::Angle { value: 45.0 }.channel(),
            ChannelName::Angle
        );
        // Field carries no literal channel; it defaults to Color.
        assert_eq!(
            EncodingValue::Field { name: "g".into() }.channel(),
            ChannelName::Color
        );
    }

    /// `as_str` names every channel as its wire form serializes it.
    #[test]
    fn channel_name_as_str_matches_its_wire_name() {
        use ChannelName::*;
        for channel in [X, Y, Color, Size, Shape, Opacity, StrokeWidth, StrokeOpacity, StrokeDash, FillOpacity, Angle] {
            let wire = serde_json::to_value(channel).unwrap();
            assert_eq!(wire.as_str(), Some(channel.as_str()), "{channel:?}");
        }
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

    /// Owner panels ride the scene wire only when stamped: an unstamped
    /// selection serializes exactly as its declaration and reads back
    /// unstamped, and a stamped one round-trips its list.
    #[test]
    fn selection_owner_panels_are_omitted_unless_stamped() {
        let json = r#"{"type":"interval","name":"b","fields":null,"encodings":null,"translate":true,"zoom":true,"mark":null,"resolve":"global"}"#;
        let mut scene: SceneSelection = serde_json::from_str(json).unwrap();
        assert_eq!(scene.panels, None);
        assert_eq!(serde_json::to_string(&scene).unwrap(), json);
        assert_eq!(serde_json::to_string(&scene.selection).unwrap(), json);

        scene.panels = Some(vec![1, 2]);
        let stamped = serde_json::to_string(&scene).unwrap();
        assert!(stamped.ends_with(r#","panels":[1,2]}"#), "{stamped}");
        assert_eq!(serde_json::from_str::<SceneSelection>(&stamped).unwrap(), scene);
    }

    /// Owner panels are the scene build's to stamp: a declared selection
    /// carrying `panels` is refused, as is any other key the declaration does
    /// not have, while the same key on the scene wire is read.
    #[test]
    fn a_declared_selection_refuses_owner_panels_and_unknown_keys() {
        let declared = r#"{"type":"point","name":"p","fields":null,"encodings":null,"nearest":false,"toggle":"shift_key","on":"click","clear":"dblclick","resolve":"global""#;
        assert!(serde_json::from_str::<SelectionSpec>(&format!("{declared}}}")).is_ok());
        for extra in [r#""panels":[0]"#, r#""bogus":1"#] {
            let json = format!("{declared},{extra}}}");
            let err = serde_json::from_str::<SelectionSpec>(&json).unwrap_err().to_string();
            let key = extra.split('"').nth(1).unwrap();
            assert!(err.contains(&format!("unknown field `{key}`")), "{json}: {err}");
        }
        let scene: SceneSelection = serde_json::from_str(&format!(r#"{declared},"panels":[0]}}"#)).unwrap();
        assert_eq!(scene.panels, Some(vec![0]));
    }
}
