use serde::{Deserialize, Serialize};

/// Per-mark constant style overrides. Phase 8a fields cover all kwargs accepted
/// by the 8 primitive mark_*() Python methods. All None defaults; renderer falls
/// back to theme defaults when None.
///
/// Resolution priority in prepare.rs: layer.mark_style > chart.mark_style > theme.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MarkKwargsSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corner_radius: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_width: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_dash: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_weight: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dx: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle: Option<f64>,
    /// Polygon mark grouping column (e.g. `hex_id`, `violin_id`, `level_id`).
    /// When None, all rows form a single polygon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Named continuous colormap (e.g. `viridis`, `plasma`). Used by polygon mark
    /// when `color` encoding maps to a quantitative column. Defaults to `viridis`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cmap: Option<String>,
    // ── S1: interpolate (line/area) ──────────────────────────────────────────
    /// Path interpolation method. Supported: "linear" (default), "step",
    /// "step-before", "step-after". Others fall back to linear.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpolate: Option<String>,
    // ── S2: stroke_cap (line) ────────────────────────────────────────────────
    /// SVG stroke-linecap. Values: "butt" (default), "round", "square".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_cap: Option<String>,
    // ── S3: stroke_join (line/area) ──────────────────────────────────────────
    /// SVG stroke-linejoin. Values: "miter" (default), "round", "bevel".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_join: Option<String>,
    // ── S5: filled (point) ───────────────────────────────────────────────────
    /// When false, points are hollow: fill="none", color applied to stroke.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filled: Option<bool>,
    // ── S6: shape (point, constant) ──────────────────────────────────────────
    /// Constant point shape when shape encoding is absent.
    /// Values: "circle", "square", "diamond", "triangle-up", "cross", "triangle-down".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
    // ── S7: limit (text) ─────────────────────────────────────────────────────
    /// Max character length for text labels; truncates with "…" if exceeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    // ── S8: band_size (tick/rect) ────────────────────────────────────────────
    /// Width of tick marks as a fraction of the band width (e.g. 0.3).
    /// Also controls boxplot IQR rect width when set on rect marks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_size: Option<f64>,
    // ── S9: line (area) ──────────────────────────────────────────────────────
    /// When true, draw an additional line border on top of the area fill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<bool>,
    // ── S10: borders (area/errorband) ────────────────────────────────────────
    /// When true, draw border lines on both top and bottom edges of an area.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub borders: Option<bool>,
    // ── mark_image URL-tile sizing ───────────────────────────────────────────
    /// Constant tile width in pixels for mark_image URL tiles. Overridden
    /// per-row by a `width` column in the data. Defaults to 32.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
    /// Constant tile height in pixels for mark_image URL tiles. Overridden
    /// per-row by a `height` column in the data. Defaults to 32.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_kwargs_default_omits_all_fields() {
        let m = MarkKwargsSpec::default();
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn mark_kwargs_round_trip_with_size_and_stroke() {
        let m = MarkKwargsSpec {
            size: Some(100.0),
            stroke: Some("#ff0000".into()),
            opacity: Some(0.5),
            ..Default::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        let parsed: MarkKwargsSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn mark_kwargs_round_trip_with_stroke_dash() {
        let m = MarkKwargsSpec {
            stroke_dash: Some(vec![5.0, 3.0]),
            ..Default::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains(r#""stroke_dash":[5.0,3.0]"#));
        let parsed: MarkKwargsSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, m);
    }
}
