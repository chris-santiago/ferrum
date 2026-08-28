//! The universal color primitive: `Color = palette::Srgba<u8>` (RGBA backed by
//! the `palette` crate). Provides SVG-formatted output, hex/CSS-name parsing,
//! and opacity application. Categorical palettes live in the sibling
//! [`super::palette`] module; this file is the primitive they are built from.

use palette::Srgba;

pub type Color = Srgba<u8>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorParseError(pub String);

/// Canonical accepted-forms phrasing, owned here and referenced verbatim by
/// every downstream color-parse error surface (Rust `RenderError::InvalidColor`
/// and the Python `to_hex`/`MarkBase` construction-time `ValueError`s). Do not
/// reword independently at call sites — quote this text so the vocabulary
/// description never drifts between Rust and Python.
pub const ACCEPTED_COLOR_FORMS: &str =
    "expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or rgb()/rgba()";

impl std::fmt::Display for ColorParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid color '{}': {ACCEPTED_COLOR_FORMS}", self.0)
    }
}

impl std::error::Error for ColorParseError {}

pub fn from_rgb(r: u8, g: u8, b: u8) -> Color {
    Srgba::new(r, g, b, 0xFF)
}

pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
    Srgba::new(r, g, b, a)
}

/// Parse a CSS color string: a hex literal (`#rgb`, `#rgba`, `#rrggbb`, or
/// `#rrggbbaa`), an `rgb()`/`rgba()` functional form, or any of the 148
/// standard CSS named colors. Three- and four-digit hex shorthand is expanded
/// by doubling each digit (`#abc` → `#aabbcc`). The input is trimmed of
/// surrounding whitespace and matched case-insensitively.
pub fn parse_color(s: &str) -> Result<Color, ColorParseError> {
    let trimmed = s.trim();
    if trimmed.starts_with('#') {
        return from_hex_str(trimmed);
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("rgb(") || lower.starts_with("rgba(") {
        return parse_rgb_function(trimmed, &lower);
    }
    let (r, g, b) = match lower.as_str() {
        "aliceblue"            => (240, 248, 255),
        "antiquewhite"         => (250, 235, 215),
        "aqua"                 => (  0, 255, 255),
        "aquamarine"           => (127, 255, 212),
        "azure"                => (240, 255, 255),
        "beige"                => (245, 245, 220),
        "bisque"               => (255, 228, 196),
        "black"                => (  0,   0,   0),
        "blanchedalmond"       => (255, 235, 205),
        "blue"                 => (  0,   0, 255),
        "blueviolet"           => (138,  43, 226),
        "brown"                => (165,  42,  42),
        "burlywood"            => (222, 184, 135),
        "cadetblue"            => ( 95, 158, 160),
        "chartreuse"           => (127, 255,   0),
        "chocolate"            => (210, 105,  30),
        "coral"                => (255, 127,  80),
        "cornflowerblue"       => (100, 149, 237),
        "cornsilk"             => (255, 248, 220),
        "crimson"              => (220,  20,  60),
        "cyan"                 => (  0, 255, 255),
        "darkblue"             => (  0,   0, 139),
        "darkcyan"             => (  0, 139, 139),
        "darkgoldenrod"        => (184, 134,  11),
        "darkgray" | "darkgrey"  => (169, 169, 169),
        "darkgreen"            => (  0, 100,   0),
        "darkkhaki"            => (189, 183, 107),
        "darkmagenta"          => (139,   0, 139),
        "darkolivegreen"       => ( 85, 107,  47),
        "darkorange"           => (255, 140,   0),
        "darkorchid"           => (153,  50, 204),
        "darkred"              => (139,   0,   0),
        "darksalmon"           => (233, 150, 122),
        "darkseagreen"         => (143, 188, 143),
        "darkslateblue"        => ( 72,  61, 139),
        "darkslategray" | "darkslategrey" => ( 47,  79,  79),
        "darkturquoise"        => (  0, 206, 209),
        "darkviolet"           => (148,   0, 211),
        "deeppink"             => (255,  20, 147),
        "deepskyblue"          => (  0, 191, 255),
        "dimgray" | "dimgrey"  => (105, 105, 105),
        "dodgerblue"           => ( 30, 144, 255),
        "firebrick"            => (178,  34,  34),
        "floralwhite"          => (255, 250, 240),
        "forestgreen"          => ( 34, 139,  34),
        "fuchsia"              => (255,   0, 255),
        "gainsboro"            => (220, 220, 220),
        "ghostwhite"           => (248, 248, 255),
        "gold"                 => (255, 215,   0),
        "goldenrod"            => (218, 165,  32),
        "gray" | "grey"        => (128, 128, 128),
        "green"                => (  0, 128,   0),
        "greenyellow"          => (173, 255,  47),
        "honeydew"             => (240, 255, 240),
        "hotpink"              => (255, 105, 180),
        "indianred"            => (205,  92,  92),
        "indigo"               => ( 75,   0, 130),
        "ivory"                => (255, 255, 240),
        "khaki"                => (240, 230, 140),
        "lavender"             => (230, 230, 250),
        "lavenderblush"        => (255, 240, 245),
        "lawngreen"            => (124, 252,   0),
        "lemonchiffon"         => (255, 250, 205),
        "lightblue"            => (173, 216, 230),
        "lightcoral"           => (240, 128, 128),
        "lightcyan"            => (224, 255, 255),
        "lightgoldenrodyellow" => (250, 250, 210),
        "lightgray" | "lightgrey" => (211, 211, 211),
        "lightgreen"           => (144, 238, 144),
        "lightpink"            => (255, 182, 193),
        "lightsalmon"          => (255, 160, 122),
        "lightseagreen"        => ( 32, 178, 170),
        "lightskyblue"         => (135, 206, 250),
        "lightslategray" | "lightslategrey" => (119, 136, 153),
        "lightsteelblue"       => (176, 196, 222),
        "lightyellow"          => (255, 255, 224),
        "lime"                 => (  0, 255,   0),
        "limegreen"            => ( 50, 205,  50),
        "linen"                => (250, 240, 230),
        "magenta"              => (255,   0, 255),
        "maroon"               => (128,   0,   0),
        "mediumaquamarine"     => (102, 205, 170),
        "mediumblue"           => (  0,   0, 205),
        "mediumorchid"         => (186,  85, 211),
        "mediumpurple"         => (147, 112, 219),
        "mediumseagreen"       => ( 60, 179, 113),
        "mediumslateblue"      => (123, 104, 238),
        "mediumspringgreen"    => (  0, 250, 154),
        "mediumturquoise"      => ( 72, 209, 204),
        "mediumvioletred"      => (199,  21, 133),
        "midnightblue"         => ( 25,  25, 112),
        "mintcream"            => (245, 255, 250),
        "mistyrose"            => (255, 228, 225),
        "moccasin"             => (255, 228, 181),
        "navajowhite"          => (255, 222, 173),
        "navy"                 => (  0,   0, 128),
        "oldlace"              => (253, 245, 230),
        "olive"                => (128, 128,   0),
        "olivedrab"            => (107, 142,  35),
        "orange"               => (255, 165,   0),
        "orangered"            => (255,  69,   0),
        "orchid"               => (218, 112, 214),
        "palegoldenrod"        => (238, 232, 170),
        "palegreen"            => (152, 251, 152),
        "paleturquoise"        => (175, 238, 238),
        "palevioletred"        => (219, 112, 147),
        "papayawhip"           => (255, 239, 213),
        "peachpuff"            => (255, 218, 185),
        "peru"                 => (205, 133,  63),
        "pink"                 => (255, 192, 203),
        "plum"                 => (221, 160, 221),
        "powderblue"           => (176, 224, 230),
        "purple"               => (128,   0, 128),
        "rebeccapurple"        => (102,  51, 153),
        "red"                  => (255,   0,   0),
        "rosybrown"            => (188, 143, 143),
        "royalblue"            => ( 65, 105, 225),
        "saddlebrown"          => (139,  69,  19),
        "salmon"               => (250, 128, 114),
        "sandybrown"           => (244, 164,  96),
        "seagreen"             => ( 46, 139,  87),
        "seashell"             => (255, 245, 238),
        "sienna"               => (160,  82,  45),
        "silver"               => (192, 192, 192),
        "skyblue"              => (135, 206, 235),
        "slateblue"            => (106,  90, 205),
        "slategray" | "slategrey" => (112, 128, 144),
        "snow"                 => (255, 250, 250),
        "springgreen"          => (  0, 255, 127),
        "steelblue"            => ( 70, 130, 180),
        "tan"                  => (210, 180, 140),
        "teal"                 => (  0, 128, 128),
        "thistle"              => (216, 191, 216),
        "tomato"               => (255,  99,  71),
        "turquoise"            => ( 64, 224, 208),
        "violet"               => (238, 130, 238),
        "wheat"                => (245, 222, 179),
        "white"                => (255, 255, 255),
        "whitesmoke"           => (245, 245, 245),
        "yellow"               => (255, 255,   0),
        "yellowgreen"          => (154, 205,  50),
        _ => return Err(ColorParseError(trimmed.to_string())),
    };
    Ok(from_rgb(r, g, b))
}

/// Parse the CSS `rgb()`/`rgba()` functional forms. `lower` is the
/// already-lowercased, already-trimmed input (used for matching); `original`
/// is the pre-lowercase trimmed input (used only for the error message, so
/// the error echoes the caller's exact spelling).
fn parse_rgb_function(original: &str, lower: &str) -> Result<Color, ColorParseError> {
    let err = || ColorParseError(original.to_string());
    let (has_alpha, inner) = if let Some(inner) = lower.strip_prefix("rgba(") {
        (true, inner)
    } else {
        (false, lower.strip_prefix("rgb(").ok_or_else(err)?)
    };
    let inner = inner.strip_suffix(')').ok_or_else(err)?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    let expected_parts = if has_alpha { 4 } else { 3 };
    if parts.len() != expected_parts || parts.iter().any(|p| p.is_empty()) {
        return Err(err());
    }
    let channel = |p: &str| p.parse::<u8>().map_err(|_| err());
    let r = channel(parts[0])?;
    let g = channel(parts[1])?;
    let b = channel(parts[2])?;
    if !has_alpha {
        return Ok(from_rgb(r, g, b));
    }
    // Alpha is a float in 0..=1 only. The CSS percentage-free integer 0..255
    // form is not accepted — an out-of-range value like "255" falls through
    // the range check below rather than needing separate syntax detection.
    let a: f64 = parts[3].parse().map_err(|_| err())?;
    if !(0.0..=1.0).contains(&a) {
        return Err(err());
    }
    Ok(from_rgba(r, g, b, (a * 255.0).round() as u8))
}

pub fn from_hex_str(s: &str) -> Result<Color, ColorParseError> {
    let s = s.trim();
    if !s.starts_with('#') {
        return Err(ColorParseError(s.to_string()));
    }
    // Expand CSS shorthand: #rgb -> #rrggbb and #rgba -> #rrggbbaa
    // (each digit doubled, e.g. #abc -> #aabbcc).
    let short = &s[1..];
    // Reject non-ASCII up front. Every branch below indexes `short`/`hex` by
    // byte offset (hex-digit-pair boundaries), which assumes 1 byte == 1
    // char; a multi-byte UTF-8 character (e.g. "#a€") would otherwise let a
    // byte-length hex-digit count line up with a length arm below while its
    // char count does not, and the later `&hex[i..i+2]` slice can land mid
    // character and panic instead of returning a parse error.
    if !short.is_ascii() {
        return Err(ColorParseError(s.to_string()));
    }
    let expanded: String;
    let hex: &str = match short.len() {
        3 | 4 => {
            expanded = short.chars().flat_map(|c| [c, c]).collect();
            &expanded
        }
        _ => short,
    };
    let parse = |i: usize| -> Result<u8, ColorParseError> {
        u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| ColorParseError(s.to_string()))
    };
    match hex.len() {
        6 => Ok(Srgba::new(parse(0)?, parse(2)?, parse(4)?, 0xFF)),
        8 => Ok(Srgba::new(parse(0)?, parse(2)?, parse(4)?, parse(6)?)),
        _ => Err(ColorParseError(s.to_string())),
    }
}

pub fn with_opacity(c: Color, opacity_0_1: f64) -> Color {
    // NaN.clamp(0.0, 1.0) returns NaN in Rust; guard explicitly so NaN opacity
    // doesn't silently produce a=0 (fully transparent element disappears).
    let a = if opacity_0_1.is_nan() {
        c.alpha
    } else {
        (c.alpha as f64 * opacity_0_1.clamp(0.0, 1.0)).round() as u8
    };
    Srgba::new(c.red, c.green, c.blue, a)
}

pub fn fmt_svg(c: Color) -> String {
    if c.alpha == 0xFF {
        to_hex_string(c)
    } else {
        let a = (c.alpha as f64) / 255.0;
        format!("rgba({},{},{},{:.3})", c.red, c.green, c.blue, a)
    }
}

/// Format a color as normalized hex: `#rrggbb` when fully opaque,
/// `#rrggbbaa` when it carries alpha. Unlike [`fmt_svg`] (which falls back to
/// the `rgba()` CSS functional form for translucent colors), this is the wire
/// format `ferrum.color.to_hex` promises Python callers — hex in, hex out.
/// `fmt_svg`'s opaque branch delegates here so the two formatters never drift
/// on the `#rrggbb` case.
pub fn to_hex_string(c: Color) -> String {
    if c.alpha == 0xFF {
        format!("#{:02x}{:02x}{:02x}", c.red, c.green, c.blue)
    } else {
        format!("#{:02x}{:02x}{:02x}{:02x}", c.red, c.green, c.blue, c.alpha)
    }
}

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Parse any color string in ferrum's accepted vocabulary (CSS name, hex, or
/// `rgb()`/`rgba()`) and return its normalized hex form. This is the single
/// Python-visible color-parsing entry point (`ferrum.color.to_hex`'s string
/// path) — Python owns no color vocabulary of its own, it defers to this
/// parser so there is exactly one accepted-forms definition in the codebase.
#[pyfunction]
pub fn parse_color_to_hex(s: &str) -> PyResult<String> {
    parse_color(s)
        .map(to_hex_string)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_color tests (written before implementation — TDD) ---

    /// CSS Color 4 reference table: all 148 named colors, asserted against
    /// their canonical RGB values (generated from the CSS3 keyword registry,
    /// plus `rebeccapurple` from CSS Color 4). Catches transcription errors
    /// like the `mediumpurple` bug this table pins.
    const CSS_COLOR_4_REFERENCE: &[(&str, u8, u8, u8)] = &[
        ("aliceblue", 240, 248, 255),
        ("antiquewhite", 250, 235, 215),
        ("aqua", 0, 255, 255),
        ("aquamarine", 127, 255, 212),
        ("azure", 240, 255, 255),
        ("beige", 245, 245, 220),
        ("bisque", 255, 228, 196),
        ("black", 0, 0, 0),
        ("blanchedalmond", 255, 235, 205),
        ("blue", 0, 0, 255),
        ("blueviolet", 138, 43, 226),
        ("brown", 165, 42, 42),
        ("burlywood", 222, 184, 135),
        ("cadetblue", 95, 158, 160),
        ("chartreuse", 127, 255, 0),
        ("chocolate", 210, 105, 30),
        ("coral", 255, 127, 80),
        ("cornflowerblue", 100, 149, 237),
        ("cornsilk", 255, 248, 220),
        ("crimson", 220, 20, 60),
        ("cyan", 0, 255, 255),
        ("darkblue", 0, 0, 139),
        ("darkcyan", 0, 139, 139),
        ("darkgoldenrod", 184, 134, 11),
        ("darkgray", 169, 169, 169),
        ("darkgreen", 0, 100, 0),
        ("darkgrey", 169, 169, 169),
        ("darkkhaki", 189, 183, 107),
        ("darkmagenta", 139, 0, 139),
        ("darkolivegreen", 85, 107, 47),
        ("darkorange", 255, 140, 0),
        ("darkorchid", 153, 50, 204),
        ("darkred", 139, 0, 0),
        ("darksalmon", 233, 150, 122),
        ("darkseagreen", 143, 188, 143),
        ("darkslateblue", 72, 61, 139),
        ("darkslategray", 47, 79, 79),
        ("darkslategrey", 47, 79, 79),
        ("darkturquoise", 0, 206, 209),
        ("darkviolet", 148, 0, 211),
        ("deeppink", 255, 20, 147),
        ("deepskyblue", 0, 191, 255),
        ("dimgray", 105, 105, 105),
        ("dimgrey", 105, 105, 105),
        ("dodgerblue", 30, 144, 255),
        ("firebrick", 178, 34, 34),
        ("floralwhite", 255, 250, 240),
        ("forestgreen", 34, 139, 34),
        ("fuchsia", 255, 0, 255),
        ("gainsboro", 220, 220, 220),
        ("ghostwhite", 248, 248, 255),
        ("gold", 255, 215, 0),
        ("goldenrod", 218, 165, 32),
        ("gray", 128, 128, 128),
        ("green", 0, 128, 0),
        ("greenyellow", 173, 255, 47),
        ("grey", 128, 128, 128),
        ("honeydew", 240, 255, 240),
        ("hotpink", 255, 105, 180),
        ("indianred", 205, 92, 92),
        ("indigo", 75, 0, 130),
        ("ivory", 255, 255, 240),
        ("khaki", 240, 230, 140),
        ("lavender", 230, 230, 250),
        ("lavenderblush", 255, 240, 245),
        ("lawngreen", 124, 252, 0),
        ("lemonchiffon", 255, 250, 205),
        ("lightblue", 173, 216, 230),
        ("lightcoral", 240, 128, 128),
        ("lightcyan", 224, 255, 255),
        ("lightgoldenrodyellow", 250, 250, 210),
        ("lightgray", 211, 211, 211),
        ("lightgreen", 144, 238, 144),
        ("lightgrey", 211, 211, 211),
        ("lightpink", 255, 182, 193),
        ("lightsalmon", 255, 160, 122),
        ("lightseagreen", 32, 178, 170),
        ("lightskyblue", 135, 206, 250),
        ("lightslategray", 119, 136, 153),
        ("lightslategrey", 119, 136, 153),
        ("lightsteelblue", 176, 196, 222),
        ("lightyellow", 255, 255, 224),
        ("lime", 0, 255, 0),
        ("limegreen", 50, 205, 50),
        ("linen", 250, 240, 230),
        ("magenta", 255, 0, 255),
        ("maroon", 128, 0, 0),
        ("mediumaquamarine", 102, 205, 170),
        ("mediumblue", 0, 0, 205),
        ("mediumorchid", 186, 85, 211),
        ("mediumpurple", 147, 112, 219),
        ("mediumseagreen", 60, 179, 113),
        ("mediumslateblue", 123, 104, 238),
        ("mediumspringgreen", 0, 250, 154),
        ("mediumturquoise", 72, 209, 204),
        ("mediumvioletred", 199, 21, 133),
        ("midnightblue", 25, 25, 112),
        ("mintcream", 245, 255, 250),
        ("mistyrose", 255, 228, 225),
        ("moccasin", 255, 228, 181),
        ("navajowhite", 255, 222, 173),
        ("navy", 0, 0, 128),
        ("oldlace", 253, 245, 230),
        ("olive", 128, 128, 0),
        ("olivedrab", 107, 142, 35),
        ("orange", 255, 165, 0),
        ("orangered", 255, 69, 0),
        ("orchid", 218, 112, 214),
        ("palegoldenrod", 238, 232, 170),
        ("palegreen", 152, 251, 152),
        ("paleturquoise", 175, 238, 238),
        ("palevioletred", 219, 112, 147),
        ("papayawhip", 255, 239, 213),
        ("peachpuff", 255, 218, 185),
        ("peru", 205, 133, 63),
        ("pink", 255, 192, 203),
        ("plum", 221, 160, 221),
        ("powderblue", 176, 224, 230),
        ("purple", 128, 0, 128),
        ("red", 255, 0, 0),
        ("rosybrown", 188, 143, 143),
        ("royalblue", 65, 105, 225),
        ("saddlebrown", 139, 69, 19),
        ("salmon", 250, 128, 114),
        ("sandybrown", 244, 164, 96),
        ("seagreen", 46, 139, 87),
        ("seashell", 255, 245, 238),
        ("sienna", 160, 82, 45),
        ("silver", 192, 192, 192),
        ("skyblue", 135, 206, 235),
        ("slateblue", 106, 90, 205),
        ("slategray", 112, 128, 144),
        ("slategrey", 112, 128, 144),
        ("snow", 255, 250, 250),
        ("springgreen", 0, 255, 127),
        ("steelblue", 70, 130, 180),
        ("tan", 210, 180, 140),
        ("teal", 0, 128, 128),
        ("thistle", 216, 191, 216),
        ("tomato", 255, 99, 71),
        ("turquoise", 64, 224, 208),
        ("violet", 238, 130, 238),
        ("wheat", 245, 222, 179),
        ("white", 255, 255, 255),
        ("whitesmoke", 245, 245, 245),
        ("yellow", 255, 255, 0),
        ("yellowgreen", 154, 205, 50),
        ("rebeccapurple", 102, 51, 153),
    ];

    #[test]
    fn test_parse_color_full_148_name_table_matches_css_color_4() {
        assert_eq!(
            CSS_COLOR_4_REFERENCE.len(),
            148,
            "reference table itself must have 148 entries"
        );
        for &(name, r, g, b) in CSS_COLOR_4_REFERENCE {
            let c = parse_color(name).unwrap_or_else(|e| panic!("{name} failed to parse: {e}"));
            assert_eq!(
                (c.red, c.green, c.blue),
                (r, g, b),
                "{name} mismatched CSS Color 4 reference"
            );
            assert_eq!(c.alpha, 0xFF, "{name} should be fully opaque");
        }
    }

    /// #99-shaped regression: `mediumpurple` was transcribed as
    /// `(147, 111, 219)`; the correct CSS Color 4 value is `(147, 112, 219)`.
    /// Pinned individually (in addition to the full-table sweep above) since
    /// this is the exact bug the task names.
    #[test]
    fn test_parse_color_mediumpurple_is_147_112_219() {
        let c = parse_color("mediumpurple").unwrap();
        assert_eq!((c.red, c.green, c.blue), (147, 112, 219));
    }

    #[test]
    fn test_parse_color_rgb_function() {
        let c = parse_color("rgb(255, 99, 71)").unwrap();
        assert_eq!((c.red, c.green, c.blue, c.alpha), (255, 99, 71, 0xFF));
    }

    #[test]
    fn test_parse_color_rgb_function_no_spaces() {
        let c = parse_color("rgb(255,99,71)").unwrap();
        assert_eq!((c.red, c.green, c.blue, c.alpha), (255, 99, 71, 0xFF));
    }

    #[test]
    fn test_parse_color_rgb_function_case_insensitive() {
        let c = parse_color("RGB(1, 2, 3)").unwrap();
        assert_eq!((c.red, c.green, c.blue), (1, 2, 3));
    }

    #[test]
    fn test_parse_color_rgba_function_float_alpha() {
        let c = parse_color("rgba(255, 99, 71, 0.5)").unwrap();
        assert_eq!((c.red, c.green, c.blue), (255, 99, 71));
        assert_eq!(c.alpha, 128); // 0.5 * 255 = 127.5, rounds half-away-from-zero to 128
    }

    #[test]
    fn test_parse_color_rgba_function_alpha_endpoints() {
        let zero = parse_color("rgba(10, 20, 30, 0)").unwrap();
        assert_eq!(zero.alpha, 0);
        let one = parse_color("rgba(10, 20, 30, 1)").unwrap();
        assert_eq!(one.alpha, 0xFF);
    }

    #[test]
    fn test_parse_color_rgba_alpha_decimal_without_leading_zero() {
        // CSS allows ".5" as a float literal.
        let c = parse_color("rgba(10, 20, 30, .5)").unwrap();
        assert_eq!(c.alpha, 128);
    }

    #[test]
    fn test_parse_color_rgba_percentage_free_integer_alpha_rejected() {
        // Spec: alpha must be a float in 0..=1; an out-of-range integer like
        // 255 (the percentage-free 0-255 alpha form) is NOT accepted.
        assert!(parse_color("rgba(10, 20, 30, 255)").is_err());
        assert!(parse_color("rgba(10, 20, 30, 2)").is_err());
        assert!(parse_color("rgba(10, 20, 30, -1)").is_err());
    }

    #[test]
    fn test_parse_color_rgb_channel_out_of_range_errors() {
        assert!(parse_color("rgb(256, 0, 0)").is_err());
        assert!(parse_color("rgb(-1, 0, 0)").is_err());
        assert!(parse_color("rgb(1.5, 0, 0)").is_err());
    }

    #[test]
    fn test_parse_color_rgb_wrong_arity_errors() {
        assert!(parse_color("rgb(1, 2)").is_err());
        assert!(parse_color("rgb(1, 2, 3, 4)").is_err());
        assert!(parse_color("rgba(1, 2, 3)").is_err());
        assert!(parse_color("rgb()").is_err());
        assert!(parse_color("rgb(1,,3)").is_err());
    }

    #[test]
    fn test_parse_color_rgb_missing_paren_errors() {
        assert!(parse_color("rgb(1, 2, 3").is_err());
        assert!(parse_color("rgb 1, 2, 3)").is_err());
    }

    #[test]
    fn test_parse_color_error_message_names_accepted_forms() {
        let err = parse_color("not-a-color").unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid color 'not-a-color': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or rgb()/rgba()"
        );
        let rgb_err = parse_color("rgb(1,2)").unwrap_err();
        assert!(rgb_err.to_string().contains("expected a CSS color name"));
    }

    /// S4 regression (rust-quality-reviewer, Task 1): `from_hex_str` sliced
    /// `hex[i..i+2]` by byte offset, so a hex-shaped string containing a
    /// multi-byte UTF-8 character panicked with "byte index is not a char
    /// boundary" instead of returning `Err`. Before the fix, each of these
    /// three inputs panicked: `"#a€"` (`€` is 3 bytes, so the 2-char/4-byte
    /// short form hit the 3|4-digit doubling branch and the doubled string
    /// landed on the 8-byte arm, slicing mid-character at index 6); `"#d50中"`
    /// (`中` is 3 bytes, so the 6-byte short form hit the 6-byte arm directly,
    /// slicing mid-character at index 4); `"#中"` (1-byte short-circuit: 3
    /// bytes but not length 3/4/6/8 as *bytes*, still non-ASCII). All three
    /// must return `Err` naming the accepted forms, never panic.
    #[test]
    fn test_parse_color_non_ascii_hex_shaped_input_errors_not_panics() {
        for input in ["#a€", "#d50中", "#中"] {
            let err = parse_color(input).unwrap_err();
            assert!(
                err.to_string().contains("expected a CSS color name"),
                "{input:?} should error naming the accepted forms, got: {err}"
            );
        }
    }

    #[test]
    fn test_to_hex_string_opaque_and_translucent() {
        assert_eq!(to_hex_string(from_rgb(0x1f, 0x77, 0xb4)), "#1f77b4");
        assert_eq!(to_hex_string(from_rgba(0x1f, 0x77, 0xb4, 0x80)), "#1f77b480");
    }

    #[test]
    fn test_to_hex_string_roundtrips_named_and_rgb_forms() {
        let named = parse_color("steelblue").unwrap();
        assert_eq!(to_hex_string(named), "#4682b4");
        let functional = parse_color("rgb(70, 130, 180)").unwrap();
        assert_eq!(to_hex_string(functional), "#4682b4");
    }

    #[test]
    fn test_parse_color_hex_passthrough() {
        // steelblue is #4682b4 = (70, 130, 180)
        let c = parse_color("#4682b4").unwrap();
        assert_eq!(c.red, 0x46);
        assert_eq!(c.green, 0x82);
        assert_eq!(c.blue, 0xb4);
        assert_eq!(c.alpha, 0xFF);
    }

    #[test]
    fn test_parse_color_named_steelblue() {
        let c = parse_color("steelblue").unwrap();
        assert_eq!(c.red, 70);
        assert_eq!(c.green, 130);
        assert_eq!(c.blue, 180);
        assert_eq!(c.alpha, 0xFF);
    }

    #[test]
    fn test_parse_color_named_case_insensitive() {
        let c = parse_color("SteelBlue").unwrap();
        assert_eq!(c.red, 70);
        assert_eq!(c.green, 130);
        assert_eq!(c.blue, 180);
    }

    #[test]
    fn test_parse_color_named_with_whitespace() {
        let c = parse_color("  steelblue  ").unwrap();
        assert_eq!(c.red, 70);
        assert_eq!(c.green, 130);
        assert_eq!(c.blue, 180);
    }

    #[test]
    fn test_parse_color_unknown_name_errors() {
        let err = parse_color("notacolor").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("notacolor"), "error message should contain the input: {msg}");
        assert!(
            msg.contains("CSS color name") || msg.contains("#rrggbb"),
            "error message should mention CSS color name or #rrggbb: {msg}"
        );
    }

    #[test]
    fn test_parse_color_grey_spelling() {
        // Both British and American spellings must work for gray/grey variants
        let gray = parse_color("gray").unwrap();
        let grey = parse_color("grey").unwrap();
        assert_eq!(gray.red, 128);
        assert_eq!(gray.green, 128);
        assert_eq!(gray.blue, 128);
        assert_eq!(gray.red, grey.red);
        assert_eq!(gray.green, grey.green);
        assert_eq!(gray.blue, grey.blue);

        let dk_gray = parse_color("darkgray").unwrap();
        let dk_grey = parse_color("darkgrey").unwrap();
        assert_eq!(dk_gray.red, 169);
        assert_eq!(dk_gray.red, dk_grey.red);
        assert_eq!(dk_gray.green, dk_grey.green);
        assert_eq!(dk_gray.blue, dk_grey.blue);
    }

    // --- existing tests ---

    #[test]
    fn parse_six_digit_hex() {
        let c = from_hex_str("#1f77b4").unwrap();
        assert_eq!(c.red, 0x1f);
        assert_eq!(c.green, 0x77);
        assert_eq!(c.blue, 0xb4);
        assert_eq!(c.alpha, 0xFF);
    }

    #[test]
    fn parse_eight_digit_hex() {
        let c = from_hex_str("#1f77b4cc").unwrap();
        assert_eq!(c.alpha, 0xCC);
    }

    #[test]
    fn parse_named_color_fails() {
        assert!(from_hex_str("red").is_err());
    }

    #[test]
    fn parse_three_digit_hex_shorthand() {
        // #rgb expands to #rrggbb (each digit doubled).
        assert_eq!(parse_color("#ccc").unwrap(), parse_color("#cccccc").unwrap());
        assert_eq!(parse_color("#abc").unwrap(), parse_color("#aabbcc").unwrap());
        let c = parse_color("#abc").unwrap();
        assert_eq!(c.red, 0xaa);
        assert_eq!(c.green, 0xbb);
        assert_eq!(c.blue, 0xcc);
        assert_eq!(c.alpha, 0xFF);
    }

    #[test]
    fn parse_four_digit_hex_shorthand() {
        // #rgba expands to #rrggbbaa (alpha included).
        assert_eq!(parse_color("#abcd").unwrap(), parse_color("#aabbccdd").unwrap());
        let c = parse_color("#abcd").unwrap();
        assert_eq!(c.red, 0xaa);
        assert_eq!(c.green, 0xbb);
        assert_eq!(c.blue, 0xcc);
        assert_eq!(c.alpha, 0xdd);
    }

    #[test]
    fn parse_six_and_eight_digit_unchanged_by_shorthand() {
        let c6 = parse_color("#1f77b4").unwrap();
        assert_eq!((c6.red, c6.green, c6.blue, c6.alpha), (0x1f, 0x77, 0xb4, 0xFF));
        let c8 = parse_color("#1f77b4cc").unwrap();
        assert_eq!((c8.red, c8.green, c8.blue, c8.alpha), (0x1f, 0x77, 0xb4, 0xcc));
    }

    #[test]
    fn parse_invalid_short_hex_errors() {
        // #xy is not valid hex and not a valid length.
        assert!(parse_color("#xy").is_err());
        // 3 chars but non-hex digits must error.
        assert!(parse_color("#xyz").is_err());
    }

    #[test]
    fn opacity_multiplies() {
        let c = with_opacity(from_rgb(0xFF, 0x00, 0x00), 0.5);
        assert_eq!(c.alpha, 128);
    }

    /// #99 residue: `with_opacity`'s clamp-boundary and endpoint contract,
    /// pinned against the real function (spec §4.6). Endpoints: opacity 0
    /// zeroes the alpha; opacity 1 leaves it unchanged. Clamp arms: negative
    /// opacity clamps to the 0 endpoint, opacity > 1 clamps to the 1
    /// endpoint (never scales past the color's own alpha).
    #[test]
    fn with_opacity_endpoint_zero_zeroes_alpha() {
        let c = with_opacity(from_rgba(0x10, 0x20, 0x30, 200), 0.0);
        assert_eq!(c.alpha, 0);
    }

    #[test]
    fn with_opacity_endpoint_one_preserves_alpha() {
        let c = with_opacity(from_rgba(0x10, 0x20, 0x30, 200), 1.0);
        assert_eq!(c.alpha, 200);
    }

    #[test]
    fn with_opacity_negative_clamps_to_zero_endpoint() {
        let c = with_opacity(from_rgba(0x10, 0x20, 0x30, 200), -0.5);
        assert_eq!(c.alpha, 0, "negative opacity must clamp to the same result as opacity=0");
    }

    #[test]
    fn with_opacity_above_one_clamps_to_original_alpha() {
        // alpha=100, opacity=2.0: unclamped this would multiply to 200 —
        // clamping opacity to 1.0 first must instead leave alpha at 100
        // (its own, unscaled value), never overshoot past it.
        let c = with_opacity(from_rgba(0x10, 0x20, 0x30, 100), 2.0);
        assert_eq!(c.alpha, 100, "opacity > 1 must clamp to the 1.0 endpoint, not scale past the original alpha");
    }

    /// R1 port (bug_hunt_draw.rs): `NaN.clamp(0.0, 1.0)` is NaN in Rust, which
    /// would otherwise silently zero out an element's alpha (invisible) instead
    /// of preserving it. `with_opacity` guards this explicitly.
    #[test]
    fn opacity_nan_preserves_original_alpha() {
        let c = with_opacity(from_rgba(0x10, 0x20, 0x30, 0xFF), f64::NAN);
        assert_eq!(c.alpha, 0xFF, "NaN opacity must preserve alpha, not zero it");
        let c2 = with_opacity(from_rgba(0x10, 0x20, 0x30, 0x80), f64::NAN);
        assert_eq!(c2.alpha, 0x80, "NaN opacity must preserve a partial alpha too");
    }

    /// R1 port: hex parsing must be case-insensitive (upper/lower/mixed all agree).
    #[test]
    fn from_hex_str_is_case_insensitive() {
        let lower = from_hex_str("#aabbcc").unwrap();
        let upper = from_hex_str("#AABBCC").unwrap();
        let mixed = from_hex_str("#aAbBcC").unwrap();
        assert_eq!(lower, upper);
        assert_eq!(lower, mixed);
        assert_eq!((lower.red, lower.green, lower.blue), (0xAA, 0xBB, 0xCC));
    }

    #[test]
    fn fmt_svg_opaque_uses_hex() {
        assert_eq!(fmt_svg(from_rgb(0x1f, 0x77, 0xb4)), "#1f77b4");
    }

    #[test]
    fn fmt_svg_translucent_uses_rgba() {
        let c = from_rgba(0x1f, 0x77, 0xb4, 0x80);
        assert_eq!(fmt_svg(c), "rgba(31,119,180,0.502)");
    }
}
