//! Facet input/output types and grid arithmetic. Phase 6 supports two modes:
//! Wrap (ncols set, nrows derived from n_panels) and Grid (both explicit;
//! panels beyond nrows*ncols are dropped with a warning).

use serde::{Deserialize, Serialize};

use super::geometry::Rect;
use super::panel::FacetKey;

/// Spec-side facet declaration. Carried by `ChartSpec.facet`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FacetSpec {
    pub field: String,
    pub mode: FacetMode,
    /// If set, overrides `theme.column_padding` / `theme.row_padding` symmetrically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spacing: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum FacetMode {
    Wrap { ncols: u32 },
    Grid { nrows: u32, ncols: u32 },
}

/// Caller-supplied per-panel input. `n_rows` is informational only — Phase 6
/// does not use it for layout decisions but Phase 7+ may.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetGroup {
    pub key: FacetKey,
    pub n_rows: u64,
}

/// Computed grid sizing. `cell_rect(row, col, origin)` returns the panel rect.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetGrid {
    pub mode: FacetMode,
    pub n_panels: u32,
    pub cell_w: f64,
    pub cell_h: f64,
    pub gutter_x: f64,
    pub gutter_y: f64,
    pub origin: Rect,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facet_spec_round_trip_wrap() {
        let s = FacetSpec {
            field: "species".into(),
            mode: FacetMode::Wrap { ncols: 3 },
            spacing: None,
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: FacetSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
        assert!(json.contains(r#""kind":"wrap""#));
        assert!(json.contains(r#""ncols":3"#));
    }

    #[test]
    fn facet_spec_round_trip_grid() {
        let s = FacetSpec {
            field: "year".into(),
            mode: FacetMode::Grid { nrows: 2, ncols: 3 },
            spacing: Some(12.0),
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: FacetSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
        assert!(json.contains(r#""kind":"grid""#));
    }

    #[test]
    fn facet_spec_omits_spacing_when_none() {
        let s = FacetSpec {
            field: "f".into(),
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("spacing"));
    }
}
