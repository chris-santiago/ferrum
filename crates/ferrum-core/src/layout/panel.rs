//! Per-panel layout (single chart or one cell of a facet grid).

use serde::{Deserialize, Serialize};

use super::geometry::Rect;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PanelLayout {
    pub plot_area: Rect,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facet_key: Option<FacetKey>,
    pub row: u32,
    pub col: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strip_title: Option<StripTitleLayout>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FacetKey {
    pub field: String,
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextAnchor {
    Start,
    Middle,
    End,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StripTitleLayout {
    pub text: String,
    pub anchor: (f64, f64),
    pub align: TextAnchor,
    pub font_size: f64,
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
            strip_title: None,
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
            strip_title: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        let parsed: PanelLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }
}

#[cfg(test)]
mod strip_title_tests {
    use super::*;

    #[test]
    fn strip_title_layout_round_trip() {
        let s = StripTitleLayout {
            text: "setosa".into(),
            anchor: (10.0, 20.0),
            align: TextAnchor::Middle,
            font_size: 13.0,
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: StripTitleLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
    }

    #[test]
    fn panel_layout_strip_title_omitted_when_none() {
        let p = PanelLayout {
            plot_area: crate::layout::Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.contains("strip_title"), "expected omission, got: {json}");
    }

    #[test]
    fn panel_layout_with_strip_title_round_trip() {
        let p = PanelLayout {
            plot_area: crate::layout::Rect { x: 0.0, y: 0.0, w: 100.0, h: 80.0 },
            facet_key: Some(FacetKey { field: "species".into(), value: "setosa".into() }),
            row: 0,
            col: 1,
            strip_title: Some(StripTitleLayout {
                text: "setosa".into(),
                anchor: (50.0, 5.0),
                align: TextAnchor::Middle,
                font_size: 13.0,
            }),
        };
        let json = serde_json::to_string(&p).unwrap();
        let parsed: PanelLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }
}
