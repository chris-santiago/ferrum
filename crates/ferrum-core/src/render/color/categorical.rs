//! Color = palette::Srgba<u8>. SVG-formatted output, hex parsing, opacity.

use palette::Srgba;

pub type Color = Srgba<u8>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorParseError(pub String);

impl std::fmt::Display for ColorParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid color string: '{}' (expected #rrggbb or #rrggbbaa)", self.0)
    }
}

impl std::error::Error for ColorParseError {}

pub fn from_rgb(r: u8, g: u8, b: u8) -> Color {
    Srgba::new(r, g, b, 0xFF)
}

pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
    Srgba::new(r, g, b, a)
}

pub fn from_hex_str(s: &str) -> Result<Color, ColorParseError> {
    let s = s.trim();
    if !s.starts_with('#') {
        return Err(ColorParseError(s.to_string()));
    }
    let hex = &s[1..];
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
    let a = (c.alpha as f64 * opacity_0_1.clamp(0.0, 1.0)).round() as u8;
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
