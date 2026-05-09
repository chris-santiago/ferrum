//! Per-panel layout output. A non-faceted chart yields one PanelLayout with
//! `facet_key = None` and `(row, col) = (0, 0)`.

use serde::{Deserialize, Serialize};

use super::geometry::Rect;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PanelLayout {
    pub plot_area: Rect,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub facet_key: Option<FacetKey>,
    pub row: u32,
    pub col: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FacetKey {
    pub field: String,
    pub value: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_layout_round_trip_no_facet() {
        let p = PanelLayout {
            plot_area: Rect { x: 10.0, y: 20.0, w: 300.0, h: 200.0 },
            facet_key: None,
            row: 0,
            col: 0,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.contains("facet_key"), "facet_key None must be skipped: {json}");
        let parsed: PanelLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn panel_layout_round_trip_with_facet() {
        let p = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 200.0, h: 100.0 },
            facet_key: Some(FacetKey {
                field: "species".into(),
                value: "setosa".into(),
            }),
            row: 1,
            col: 2,
        };
        let json = serde_json::to_string(&p).unwrap();
        let parsed: PanelLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }
}
