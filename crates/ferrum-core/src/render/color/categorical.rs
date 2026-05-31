//! Color = palette::Srgba<u8>. SVG-formatted output, hex parsing, opacity.

use palette::Srgba;

pub type Color = Srgba<u8>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorParseError(pub String);

impl std::fmt::Display for ColorParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid color string: '{}' (expected a CSS color name or #rgb / #rgba / #rrggbb / #rrggbbaa)",
            self.0
        )
    }
}

impl std::error::Error for ColorParseError {}

pub fn from_rgb(r: u8, g: u8, b: u8) -> Color {
    Srgba::new(r, g, b, 0xFF)
}

pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
    Srgba::new(r, g, b, a)
}

/// Parse a CSS color string: either a hex literal (`#rgb`, `#rgba`, `#rrggbb`,
/// or `#rrggbbaa`) or any of the 148 standard CSS named colors. Three- and
/// four-digit shorthand is expanded by doubling each digit (`#abc` → `#aabbcc`).
/// The input is trimmed of surrounding whitespace and matched case-insensitively.
pub fn parse_color(s: &str) -> Result<Color, ColorParseError> {
    let trimmed = s.trim();
    if trimmed.starts_with('#') {
        return from_hex_str(trimmed);
    }
    let lower = trimmed.to_ascii_lowercase();
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
        "mediumpurple"         => (147, 111, 219),
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

pub fn from_hex_str(s: &str) -> Result<Color, ColorParseError> {
    let s = s.trim();
    if !s.starts_with('#') {
        return Err(ColorParseError(s.to_string()));
    }
    // Expand CSS shorthand: #rgb -> #rrggbb and #rgba -> #rrggbbaa
    // (each digit doubled, e.g. #abc -> #aabbcc).
    let short = &s[1..];
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
        format!("#{:02x}{:02x}{:02x}", c.red, c.green, c.blue)
    } else {
        let a = (c.alpha as f64) / 255.0;
        format!("rgba({},{},{},{:.3})", c.red, c.green, c.blue, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_color tests (written before implementation — TDD) ---

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
