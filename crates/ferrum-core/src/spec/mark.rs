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

/// Single source of truth for the primitive `Mark` variants. Mirrors the
/// `for_each_transform!` pattern in `transform/core.rs` — the name mappings
/// (`as_str`, `from_str`) drive off this table. The lowercase form is encoded
/// as the second column so it doubles as both the serde tag and the
/// `render::marks` submodule name. `render::draw`'s build dispatch spells its
/// arms out explicitly instead; see the rationale at that match.
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

impl Mark {
    pub fn as_str(&self) -> &'static str {
        macro_rules! arm {
            ($($V:ident => $name:ident,)*) => {
                match self { $( Mark::$V => stringify!($name), )* }
            };
        }
        for_each_mark!(arm)
    }

    /// How this mark draws `size`: the one table every question about which
    /// marks draw `size` reads, at render and (through
    /// [`marks_drawing_constant_size`]) at construction.
    pub(crate) fn size_drawing(self) -> SizeDrawing {
        match self {
            Mark::Point | Mark::Line => SizeDrawing::ConstantOrEncoded,
            Mark::Text => SizeDrawing::EncodedOnly,
            Mark::Bar | Mark::Area | Mark::Rule | Mark::Tick | Mark::Rect | Mark::Polygon | Mark::Image
            | Mark::Ribbon | Mark::Segment | Mark::Arc | Mark::Label | Mark::Geoshape => SizeDrawing::Never,
        }
    }

    /// Whether this mark draws an encoded `size` channel: a point's area, a
    /// line's dot-diameter width, a text's font size. Every other mark
    /// carries an inherited `size` binding without drawing it.
    pub(crate) fn draws_size_channel(self) -> bool {
        self.size_drawing() != SizeDrawing::Never
    }

    /// Whether this mark draws a constant `size=`.
    pub(crate) fn draws_constant_size(self) -> bool {
        self.size_drawing() == SizeDrawing::ConstantOrEncoded
    }

    /// The mark's own kwarg that sets how big or thick it draws, named when a
    /// constant `size=` it never draws is refused.
    pub(crate) fn size_knob(self) -> Option<&'static str> {
        match self {
            Mark::Rule | Mark::Segment => Some("stroke_width"),
            Mark::Text | Mark::Label => Some("font_size"),
            Mark::Tick | Mark::Rect => Some("band_size"),
            _ => None,
        }
    }
}

/// What a mark draws for `size` ([`Mark::size_drawing`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SizeDrawing {
    /// A constant `size=` or an encoded `size` channel: a point's area, a
    /// line's dot-diameter width.
    ConstantOrEncoded,
    /// An encoded `size` channel only: a text's font size. A constant
    /// `size=` is never read; `font_size=` sizes the mark.
    EncodedOnly,
    /// Nothing: the mark carries an inherited `size` binding without drawing it.
    Never,
}

/// The primitive marks that draw a constant `size=`, in [`for_each_mark!`]
/// order: Python's mark constructors read this rather than keeping a list of
/// their own.
#[pyo3::pyfunction]
pub fn marks_drawing_constant_size() -> Vec<&'static str> {
    macro_rules! names {
        ($($V:ident => $name:ident,)*) => {
            [$( Mark::$V, )*]
        };
    }
    for_each_mark!(names).into_iter().filter(|m| m.draws_constant_size()).map(|m| m.as_str()).collect()
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
