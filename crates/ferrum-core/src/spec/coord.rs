use serde::{Deserialize, Serialize};

/// Coordinate system. Phase 8a honors Cartesian (default no-op) and Flip (swap x/y).
/// Other variants (Polar, Geo, Fixed) are added in Phase 9+.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum CoordKind {
    Cartesian,
    Flip,
}

impl Default for CoordKind {
    fn default() -> Self { CoordKind::Cartesian }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_kind_round_trip_cartesian() {
        let c = CoordKind::Cartesian;
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, r#"{"kind":"cartesian"}"#);
        let parsed: CoordKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
    }

    #[test]
    fn coord_kind_round_trip_flip() {
        let c = CoordKind::Flip;
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, r#"{"kind":"flip"}"#);
        let parsed: CoordKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
    }
}
