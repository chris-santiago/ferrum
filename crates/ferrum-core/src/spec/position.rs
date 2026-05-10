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
    },
}

fn default_padding() -> f64 { 0.05 }
fn default_jitter_width() -> f64 { 0.4 }
fn default_stack_offset() -> StackOffset { StackOffset::Zero }

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
        let p = PositionAdjust::Stack { by: Some("hue".into()), offset: StackOffset::Normalize };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains(r#""offset":"normalize""#));
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
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
            PositionAdjust::Stack { offset, by } => {
                assert_eq!(offset, StackOffset::Zero);
                assert!(by.is_none());
            }
            _ => panic!("expected Stack"),
        }
    }
}
