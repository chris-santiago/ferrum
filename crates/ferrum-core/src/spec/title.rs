//! Two-line title spec — see `ferrum-spec.md` §3.19, Schwabish SB1.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TitleSpec {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    #[serde(default = "default_anchor", skip_serializing_if = "is_default_anchor")]
    pub anchor: String,
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

fn default_anchor() -> String {
    "start".into()
}

fn is_default_anchor(s: &String) -> bool {
    s == "start"
}

impl Default for TitleSpec {
    fn default() -> Self {
        Self {
            text: String::new(),
            subtitle: None,
            anchor: "start".into(),
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
            anchor: "middle".into(),
            ..TitleSpec::default()
        };
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, r#"{"text":"x","anchor":"middle"}"#);
    }
}
