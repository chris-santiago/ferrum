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
