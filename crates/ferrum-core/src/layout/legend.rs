//! Legend input (caller-supplied entries) and output (engine-computed rects).

use serde::{Deserialize, Serialize};

use super::geometry::Rect;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LegendOrient {
    Right,
    Left,
    Top,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LegendDirection {
    Vertical,
    Horizontal,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SymbolKind {
    Square,
    Circle,
    Line,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LegendEntry {
    pub label: String,
    pub symbol: SymbolKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegendLayout {
    pub rect: Rect,
    pub orient: LegendOrient,
    pub direction: LegendDirection,
    pub entries: Vec<LegendEntryLayout>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegendEntryLayout {
    pub label: String,
    pub label_anchor_x: f64,
    pub label_anchor_y: f64,
    pub symbol_anchor_x: f64,
    pub symbol_anchor_y: f64,
    pub symbol_kind: SymbolKind,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legend_layout_round_trip() {
        let l = LegendLayout {
            rect: Rect { x: 500.0, y: 50.0, w: 100.0, h: 300.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![LegendEntryLayout {
                label: "A".into(),
                label_anchor_x: 520.0,
                label_anchor_y: 70.0,
                symbol_anchor_x: 510.0,
                symbol_anchor_y: 70.0,
                symbol_kind: SymbolKind::Circle,
            }],
        };
        let json = serde_json::to_string(&l).unwrap();
        let parsed: LegendLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, l);
        assert!(json.contains(r#""orient":"right""#));
        assert!(json.contains(r#""symbol_kind":"circle""#));
    }
}
