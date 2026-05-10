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
    // mark_style: Option<MarkKwargsSpec> added in Task 5
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
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(),
        };
        let json = serde_json::to_string(&layer).unwrap();
        let parsed: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, layer);
    }
}
