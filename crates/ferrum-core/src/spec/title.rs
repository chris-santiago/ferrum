//! Two-line title spec — see `ferrum-spec.md` §3.19, Schwabish SB1.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TitleSpec {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    /// Per-chart anchor override. `None` = fall back to theme's `title_anchor`.
    /// Python's `Title.to_spec_dict()` only emits `anchor` when explicitly set
    /// (i.e. not the default "start"), so `None` here means "use theme".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_weight: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle_font_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle_color: Option<String>,
}

impl Default for TitleSpec {
    fn default() -> Self {
        Self {
            text: String::new(),
            subtitle: None,
            anchor: None,
            offset: None,
            font_size: None,
            font_weight: None,
            color: None,
            subtitle_font_size: None,
            subtitle_color: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_roundtrip() {
        let t = TitleSpec {
            text: "foo".into(),
            ..TitleSpec::default()
        };
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, r#"{"text":"foo"}"#);
        let parsed: TitleSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
    }

    #[test]
    fn with_subtitle_roundtrip() {
        let json = r#"{"text":"foo","subtitle":"bar"}"#;
        let t: TitleSpec = serde_json::from_str(json).unwrap();
        assert_eq!(t.text, "foo");
        assert_eq!(t.subtitle.as_deref(), Some("bar"));
        let reserialized = serde_json::to_string(&t).unwrap();
        assert_eq!(reserialized, json);
    }

    #[test]
    fn unknown_key_rejected() {
        let json = r#"{"text":"foo","typo":"x"}"#;
        let result: Result<TitleSpec, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn anchor_middle_round_trips() {
        let t = TitleSpec {
            text: "x".into(),
            anchor: Some("middle".into()),
            ..TitleSpec::default()
        };
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, r#"{"text":"x","anchor":"middle"}"#);
    }
}
