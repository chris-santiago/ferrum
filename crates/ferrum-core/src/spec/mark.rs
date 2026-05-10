use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Mark {
    Point,
    Line,
    Bar,
    Area,
    Rule,
    Text,
    Tick,
    Rect,
    Polygon,
    Image,
}

impl Mark {
    pub fn as_str(&self) -> &'static str {
        match self {
            Mark::Point => "point",
            Mark::Line => "line",
            Mark::Bar => "bar",
            Mark::Area => "area",
            Mark::Rule => "rule",
            Mark::Text => "text",
            Mark::Tick => "tick",
            Mark::Rect => "rect",
            Mark::Polygon => "polygon",
            Mark::Image => "image",
        }
    }
}

impl fmt::Display for Mark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub struct ParseMarkError(pub String);

impl fmt::Display for ParseMarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown mark '{}'; expected one of [point, line, bar, area, rule, text, tick, rect, polygon, image]",
            self.0
        )
    }
}

impl std::error::Error for ParseMarkError {}

impl FromStr for Mark {
    type Err = ParseMarkError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "point" => Ok(Mark::Point),
            "line" => Ok(Mark::Line),
            "bar" => Ok(Mark::Bar),
            "area" => Ok(Mark::Area),
            "rule" => Ok(Mark::Rule),
            "text" => Ok(Mark::Text),
            "tick" => Ok(Mark::Tick),
            "rect" => Ok(Mark::Rect),
            "polygon" => Ok(Mark::Polygon),
            "image" => Ok(Mark::Image),
            other => Err(ParseMarkError(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mark_round_trip_each_variant() {
        for m in [
            Mark::Point, Mark::Line, Mark::Bar, Mark::Area,
            Mark::Rule, Mark::Text, Mark::Tick, Mark::Rect, Mark::Polygon, Mark::Image,
        ] {
            let json = serde_json::to_string(&m).unwrap();
            let parsed: Mark = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, m, "round-trip failed for {m:?}");
        }
    }

    #[test]
    fn test_mark_serde_form_is_lowercase() {
        assert_eq!(serde_json::to_string(&Mark::Point).unwrap(), "\"point\"");
        assert_eq!(serde_json::to_string(&Mark::Bar).unwrap(),   "\"bar\"");
    }

    #[test]
    fn test_mark_from_str_known() {
        assert_eq!(Mark::from_str("point").unwrap(), Mark::Point);
        assert_eq!(Mark::from_str("rect").unwrap(),  Mark::Rect);
    }

    #[test]
    fn test_mark_from_str_unknown_lists_variants() {
        let err = Mark::from_str("pont").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("'pont'"), "msg was: {msg}");
        assert!(msg.contains("point"), "msg was: {msg}");
        assert!(msg.contains("rect"),  "msg was: {msg}");
    }
}
