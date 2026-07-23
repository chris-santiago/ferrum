//! Phase 9c — position adjustments (Identity, Dodge, Jitter, Stack).
//!
//! Position adjustments rewrite per-row data values for an eligible mark layer
//! after scale resolution but before mark rendering. They live on Layer.position
//! (and ChartSpec.position for non-layered charts).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JitterAxis { X, Y, Both }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StackOffset { Zero, Normalize, Center }

/// Schwabish SB-followup (audit C6, 2026-05-12): controls whether a
/// stacked layer's y output lands at the segment ``Top`` (rect-style
/// marks: bar, area, ribbon, rect — draw from base→top) or the segment
/// ``Mid`` (annotation-style marks: text, point, rule, tick — anchor at
/// the visual centre of each segment so per-segment labels read cleanly).
/// Set by the composite-mark desugar; the renderer is mark-agnostic.
/// Default ``Top`` keeps pre-Schwabish stack semantics byte-identical.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StackAnchor { Top, Mid }

/// GH #77: which positional channel carries the stacked *value* column.
///
/// `apply_stack` (`render/position.rs`) historically picked the value vs.
/// category axis purely from `coord_flipped` (set only by real `CoordFlip`).
/// Composite-mark desugars with a horizontal orientation (e.g.
/// `desugar_histogram`/`desugar_density` with `orientation="horizontal"`)
/// remap the value channel onto `encoding.x` directly, without setting
/// `CoordFlip` — so `coord_flipped` stays `false` while the value lives on
/// x. `StackValueAxis` lets the desugar declare the value axis explicitly on
/// the wire; `None` preserves the pre-#77 `coord_flipped`-only convention
/// byte-identically.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum StackValueAxis { X, Y }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PositionAdjust {
    Identity,
    Dodge {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        by: Option<String>,
        #[serde(default = "default_padding")]
        padding: f64,
    },
    Jitter {
        axis: JitterAxis,
        #[serde(default = "default_jitter_width")]
        width: f64,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        seed: Option<u64>,
    },
    Stack {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        by: Option<String>,
        #[serde(default = "default_stack_offset")]
        offset: StackOffset,
        #[serde(default = "default_stack_anchor")]
        anchor: StackAnchor,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        value_axis: Option<StackValueAxis>,
    },
}

fn default_padding() -> f64 { 0.05 }
fn default_jitter_width() -> f64 { 0.4 }
fn default_stack_offset() -> StackOffset { StackOffset::Zero }
fn default_stack_anchor() -> StackAnchor { StackAnchor::Top }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_round_trip() {
        let p = PositionAdjust::Identity;
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, r#"{"type":"identity"}"#);
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn dodge_round_trip() {
        let p = PositionAdjust::Dodge { by: Some("species".into()), padding: 0.1 };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains(r#""type":"dodge""#));
        assert!(json.contains(r#""by":"species""#));
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn jitter_round_trip_with_seed() {
        let p = PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.3, seed: Some(42) };
        let json = serde_json::to_string(&p).unwrap();
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn stack_normalize_round_trip() {
        let p = PositionAdjust::Stack {
            by: Some("hue".into()),
            offset: StackOffset::Normalize,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains(r#""offset":"normalize""#));
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn stack_anchor_mid_round_trip() {
        let p = PositionAdjust::Stack {
            by: Some("actual".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Mid,
            value_axis: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains(r#""anchor":"mid""#));
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn stack_anchor_default_when_absent() {
        // Pre-Schwabish-C6 JSON (no anchor field) must deserialize to Top.
        let json = r#"{"type":"stack","offset":"zero"}"#;
        let parsed: PositionAdjust = serde_json::from_str(json).unwrap();
        match parsed {
            PositionAdjust::Stack { anchor, .. } => {
                assert_eq!(anchor, StackAnchor::Top);
            }
            _ => panic!("expected Stack"),
        }
    }

    #[test]
    fn dodge_default_padding_round_trips() {
        let json = r#"{"type":"dodge"}"#;
        let parsed: PositionAdjust = serde_json::from_str(json).unwrap();
        match parsed {
            PositionAdjust::Dodge { padding, by } => {
                assert!((padding - 0.05).abs() < 1e-12);
                assert!(by.is_none());
            }
            _ => panic!("expected Dodge"),
        }
    }

    #[test]
    fn jitter_default_width_round_trips() {
        let json = r#"{"type":"jitter","axis":"both"}"#;
        let parsed: PositionAdjust = serde_json::from_str(json).unwrap();
        match parsed {
            PositionAdjust::Jitter { axis, width, seed } => {
                assert_eq!(axis, JitterAxis::Both);
                assert!((width - 0.4).abs() < 1e-12);
                assert!(seed.is_none());
            }
            _ => panic!("expected Jitter"),
        }
    }

    #[test]
    fn stack_default_offset_round_trips() {
        let json = r#"{"type":"stack"}"#;
        let parsed: PositionAdjust = serde_json::from_str(json).unwrap();
        match parsed {
            PositionAdjust::Stack { offset, by, anchor, value_axis } => {
                assert_eq!(offset, StackOffset::Zero);
                assert!(by.is_none());
                assert_eq!(anchor, StackAnchor::Top);
                assert!(value_axis.is_none());
            }
            _ => panic!("expected Stack"),
        }
    }

    #[test]
    fn stack_value_axis_x_round_trips_and_serializes() {
        // GH #77: value_axis Some(X) must serialize as a present "x" key.
        let p = PositionAdjust::Stack {
            by: Some("hue".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: Some(StackValueAxis::X),
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains(r#""value_axis":"x""#), "got {json}");
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn stack_value_axis_absent_when_none() {
        // GH #77: value_axis None must be absent from the wire JSON entirely
        // (skip_serializing_if) so the spec-JSON byte-identity fixtures used
        // by `tests/test_scale_spec_parity.py` (scale_wire_baseline.json)
        // cannot change for pre-#77 specs.
        let p = PositionAdjust::Stack {
            by: Some("hue".into()),
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.contains("value_axis"), "got {json}");
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn stack_value_axis_defaults_to_none_when_absent_from_json() {
        // Pre-#77 JSON (no value_axis key) must deserialize to None.
        let json = r#"{"type":"stack"}"#;
        let parsed: PositionAdjust = serde_json::from_str(json).unwrap();
        match parsed {
            PositionAdjust::Stack { value_axis, .. } => assert!(value_axis.is_none()),
            _ => panic!("expected Stack"),
        }
    }
}
