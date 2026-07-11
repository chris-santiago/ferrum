use serde::{Deserialize, Serialize};

use crate::spec::encoding::Encoding;
use crate::spec::mark::Mark;
use crate::transform::core::TransformSpec;

/// A single layer within a multi-layer ChartSpec. Inherits chart-level
/// encoding for any field set to None at the layer level.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Layer {
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<TransformSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mark_style: Option<crate::spec::mark_style::MarkKwargsSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub data_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub position: Option<crate::spec::position::PositionAdjust>,
    /// Pixel-level blending mode for this layer. `None` = Normal (default).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub blend: Option<ferrum_scene::BlendMode>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// Whether this layer resolves its own y-scale slot instead of sharing
    /// the primary (layer 0) y-scale. `false` (absent on the wire) = shared,
    /// today's behavior. Layer 0 is always the primary/left axis regardless
    /// of this flag (spec §6, GH #52).
    #[serde(default)]
    pub independent_y: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::encoding::EncodingSpec;

    #[test]
    fn layer_round_trips_minimal() {
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
        position: None, blend: None, name: None, independent_y: false,
        };
        let json = serde_json::to_string(&layer).unwrap();
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_round_trips_with_encoding() {
        let layer = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false,
        };
        let json = serde_json::to_string(&layer).unwrap();
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_round_trips_with_mark_style() {
        use crate::spec::mark_style::MarkKwargsSpec;
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            mark_style: Some(MarkKwargsSpec {
                size: Some(50.0),
                ..Default::default()
            }),
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false,
        };
        let json = serde_json::to_string(&layer).unwrap();
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_data_source_round_trip_some() {
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            mark_style: None,
            data_source: Some("box".into()),
            position: None, blend: None, name: None, independent_y: false,
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(
            json.contains(r#""data_source":"box""#),
            "expected data_source in JSON: {json}"
        );
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_position_round_trips() {
        use crate::spec::position::PositionAdjust;
        let layer = Layer {
            mark: Mark::Bar,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: Some(PositionAdjust::Dodge { by: Some("g".into()), padding: 0.05 }),
            blend: None,
            name: None,
            independent_y: false,
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(json.contains(r#""position""#));
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_position_none_omits_from_json() {
        let layer = Layer {
            mark: Mark::Bar,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None, blend: None, name: None, independent_y: false,
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(!json.contains("position"), "position=None must be omitted: {json}");
    }

    #[test]
    fn layer_data_source_none_omits_from_json() {
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
        position: None, blend: None, name: None, independent_y: false,
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(
            !json.contains("data_source"),
            "data_source=None must be omitted: {json}"
        );
    }

    #[test]
    fn layer_independent_y_defaults_false_when_absent_on_wire() {
        // Deserializing a pre-#52 layer dict (no `independent_y` key at all)
        // must default to false — absolute wire back-compat.
        let json = r#"{"mark":"point"}"#;
        let layer: Layer = serde_json::from_str(json).unwrap();
        assert!(!layer.independent_y);
    }

    #[test]
    fn layer_independent_y_round_trips_true() {
        let layer = Layer {
            mark: Mark::Line,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: true,
        };
        let json = serde_json::to_string(&layer).unwrap();
        assert!(
            json.contains(r#""independent_y":true"#),
            "expected independent_y:true in JSON: {json}"
        );
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }

    #[test]
    fn layer_independent_y_round_trips_false() {
        let layer = Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false,
        };
        let json = serde_json::to_string(&layer).unwrap();
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
        assert!(!parsed.independent_y);
    }
}
