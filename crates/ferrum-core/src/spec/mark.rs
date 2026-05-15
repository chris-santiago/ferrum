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
    Ribbon,
    Segment,
    Arc,
    Label,
    Geoshape,
}

/// Single source of truth for the 12 primitive `Mark` variants. Mirrors the
/// `for_each_transform!` pattern in `transform/core.rs` — every site that
/// enumerates marks (as_str, from_str, render::draw::dispatch_mark) drives
/// off this table. The lowercase form is encoded as the second column so
/// it doubles as both the serde tag and the `render::marks` submodule name.
#[macro_export]
macro_rules! for_each_mark {
    ($mac:ident) => {
        $mac! {
            Point    => point,
            Line     => line,
            Bar      => bar,
            Area     => area,
            Rule     => rule,
            Text     => text,
            Tick     => tick,
            Rect     => rect,
            Polygon  => polygon,
            Image    => image,
            Ribbon   => ribbon,
            Segment  => segment,
            Arc      => arc,
            Label    => label,
            Geoshape => geoshape,
        }
    };
}
pub use for_each_mark;

impl Mark {
    pub fn as_str(&self) -> &'static str {
        macro_rules! arm {
            ($($V:ident => $name:ident,)*) => {
                match self { $( Mark::$V => stringify!($name), )* }
            };
        }
        for_each_mark!(arm)
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
            "unknown mark '{}'; expected one of [point, line, bar, area, rule, text, tick, rect, polygon, image, ribbon, segment, arc, label, geoshape]",
            self.0
        )
    }
}

impl std::error::Error for ParseMarkError {}

impl FromStr for Mark {
    type Err = ParseMarkError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        macro_rules! arm {
            ($($V:ident => $name:ident,)*) => {
                match s {
                    $( s if s == stringify!($name) => Ok(Mark::$V), )*
                    other => Err(ParseMarkError(other.to_string())),
                }
            };
        }
        for_each_mark!(arm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mark_round_trip_each_variant() {
        for m in [
            Mark::Point, Mark::Line, Mark::Bar, Mark::Area,
            Mark::Rule, Mark::Text, Mark::Tick, Mark::Rect, Mark::Polygon, Mark::Image, Mark::Ribbon, Mark::Segment,
            Mark::Arc, Mark::Label, Mark::Geoshape,
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
    fn test_mark_segment_round_trip() {
        let m = Mark::Segment;
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, "\"segment\"");
        let parsed: Mark = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, Mark::Segment);
        assert_eq!(Mark::from_str("segment").unwrap(), Mark::Segment);
        assert_eq!(Mark::Segment.as_str(), "segment");
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
