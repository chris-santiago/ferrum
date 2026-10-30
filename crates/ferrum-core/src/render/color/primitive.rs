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

/// The fully transparent color (`rgba(0, 0, 0, 0)`), CSS's `transparent`
/// keyword.
///
/// This is how an explicit `fill="none"` / `stroke="none"` paint clear travels
/// through the render pipeline (`draw::resolve_mark_style`): a real color that
/// paints nothing. Carrying the clear as a color — rather than as a `None`
/// paired with an "is cleared" flag — means no downstream fallback can
/// resurrect a paint the user explicitly cleared: `resolve_stroke_color`'s
/// `unwrap_or(fill)` and `resolve_effective_stroke`'s stroke-width rescue both
/// only fire on an *absent* stroke, and a cleared stroke is present.
pub const TRANSPARENT: Color = Srgba::new(0, 0, 0, 0);

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

/// Hex-literal arm of [`parse_color`]. Private to this module by design: every
/// color string the crate reads — theme overrides, legend/axis/title colors,
/// mark kwargs, the PyO3 background color — goes through [`parse_color`] so no
/// call site can accept a narrower vocabulary than the one
/// [`ACCEPTED_COLOR_FORMS`] advertises.
fn from_hex_str(s: &str) -> Result<Color, ColorParseError> {
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

/// The WCAG 2.x "AA" contrast threshold (4.5:1) [`contrast_ink`] reads a
/// row's own color against before falling back to black/white.
const CONTRAST_AA: f64 = 4.5;

/// Alpha-composite `fg` over `bg` ("source-over"), read as if `bg` were
/// fully opaque, so only `fg`'s alpha ever shapes the blend. The result is
/// always fully opaque: WCAG contrast is defined for a color actually
/// painted on a background, not for the paint and the background read
/// apart. [`contrast_ink`] is this crate's one caller, and calls it only
/// with an opaque `bg` (the composed cell, or an already-opaque theme
/// background) — it does not call this function at all when a theme
/// background is itself translucent, since there is then no known opaque
/// base to blend against.
fn composite_over(fg: Color, bg: Color) -> Color {
    let fa = fg.alpha as f64 / 255.0;
    let blend = |f: u8, b: u8| -> u8 {
        ((f as f64 * fa) + (b as f64 * (1.0 - fa))).round().clamp(0.0, 255.0) as u8
    };
    Srgba::new(blend(fg.red, bg.red), blend(fg.green, bg.green), blend(fg.blue, bg.blue), 0xFF)
}

/// WCAG 2.x relative luminance of `c` (<https://www.w3.org/TR/WCAG21/#dfn-relative-luminance>),
/// in `[0, 1]`. Alpha is ignored: every caller reads the luminance of an
/// already-opaque color ([`composite_over`]'s result, or a theme's own solid
/// paint).
fn relative_luminance(c: Color) -> f64 {
    fn channel(v: u8) -> f64 {
        let s = v as f64 / 255.0;
        if s <= 0.03928 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    }
    0.2126 * channel(c.red) + 0.7152 * channel(c.green) + 0.0722 * channel(c.blue)
}

/// WCAG 2.x contrast ratio between `a` and `b`, in `[1, 21]`: the lighter
/// relative luminance plus `0.05` over the darker plus `0.05`, order-free in
/// its two arguments.
fn contrast_ratio(a: Color, b: Color) -> f64 {
    let (la, lb) = (relative_luminance(a), relative_luminance(b));
    let (hi, lo) = if la >= lb { (la, lb) } else { (lb, la) };
    (hi + 0.05) / (lo + 0.05)
}

/// The spec's contrast-ink rule for a text label sitting on a colored cell:
/// paint whichever of `font_color`/`background` reads better against the
/// cell as it actually shows at [`CONTRAST_AA`], or whichever of black/white
/// does when neither clears it.
///
/// Two colors in this rule can themselves be translucent — a theme's
/// `background` (a chart commonly sits on a page of unknown color, so a
/// translucent or fully transparent background is ordinary, not malformed)
/// and a theme's `font_color` — and judging either as if it were opaque
/// would let an ink win a comparison it does not actually paint: a
/// transparent background composited as if solid can read as the better
/// choice and then paint nothing (alpha 0), an invisible label. So every
/// comparison here is of the color a candidate actually LEAVES on the cell,
/// never of a candidate's own raw channel values:
///
/// - **The cell as shown.** When `background` is opaque, the cell is
///   [`composite_over`]'s blend of `cell` over `background`, as always. When
///   `background` is translucent, what lies beneath it is outside the
///   chart and unknowable, so the cell is read at its own RGB and full
///   alpha instead of guessing an underlay — the same "judge the cell
///   alone" answer seaborn and plotly give an annotation.
/// - **Each candidate's paint.** `font_color` and `background` are each
///   judged by [`contrast_ratio`] between the composed cell and the color
///   that candidate actually leaves on it ([`composite_over`] of the
///   candidate over the composed cell), not the candidate's raw value. A
///   fully transparent `background` then always leaves the cell unchanged
///   (contrast 1:1 against itself) and can never win; a translucent
///   `font_color` competes as the blend it paints. The WINNING candidate is
///   still returned as its own (possibly translucent) color, so it paints
///   on the scene exactly the blend it was judged by.
/// - **The black/white tier** is unchanged: both are already opaque, so
///   judging them by their raw value and by what they paint agree.
pub fn contrast_ink(cell: Color, font_color: Color, background: Color) -> Color {
    let composed = if background.alpha == 0xFF {
        composite_over(cell, background)
    } else {
        from_rgb(cell.red, cell.green, cell.blue)
    };
    let font_contrast = contrast_ratio(composite_over(font_color, composed), composed);
    let background_contrast = contrast_ratio(composite_over(background, composed), composed);
    let (best, best_ratio) = if font_contrast >= background_contrast {
        (font_color, font_contrast)
    } else {
        (background, background_contrast)
    };
    if best_ratio >= CONTRAST_AA {
        return best;
    }
    let black = from_rgb(0, 0, 0);
    let white = from_rgb(255, 255, 255);
    if contrast_ratio(black, composed) >= contrast_ratio(white, composed) { black } else { white }
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

/// Parse any color string in ferrum's accepted vocabulary and return its
/// `(r, g, b, a)` byte tuple. The selection wire helper's Python-visible
/// entry point: rather than round-tripping through [`parse_color_to_hex`]
/// and slicing the hex string back into bytes, a caller that needs the raw
/// channels gets them directly from the same parse. Shares [`parse_color`]
/// with `parse_color_to_hex`, so the two never drift on what they accept or
/// on the bytes they produce for a given input; an opaque color's alpha is
/// always 255.
#[pyfunction]
pub fn parse_color_to_rgba(s: &str) -> PyResult<(u8, u8, u8, u8)> {
    parse_color(s)
        .map(|c| (c.red, c.green, c.blue, c.alpha))
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

    // --- parse_color_to_rgba tests: every accepted form must yield the exact
    // byte tuple parse_color_to_hex would have encoded for the same input. ---

    #[test]
    fn test_parse_color_to_rgba_pinned_tuples_for_every_accepted_form() {
        let cases: &[(&str, (u8, u8, u8, u8))] = &[
            ("#abc", (170, 187, 204, 255)),     // 3-digit hex shorthand
            ("#abcd", (170, 187, 204, 221)),    // 4-digit hex shorthand
            ("#4682b4", (70, 130, 180, 255)),   // 6-digit hex
            ("#4682b480", (70, 130, 180, 128)), // 8-digit hex
            ("steelblue", (70, 130, 180, 255)), // named
            ("rgb(70, 130, 180)", (70, 130, 180, 255)),
            ("rgba(70, 130, 180, 0.5)", (70, 130, 180, 128)), // 0.5 * 255 = 127.5, rounds to 128
        ];
        for &(input, expected) in cases {
            let actual = parse_color_to_rgba(input).unwrap();
            assert_eq!(actual, expected, "{input:?} tuple mismatch");
        }
    }

    #[test]
    fn test_parse_color_to_rgba_opaque_forms_alpha_255() {
        for input in ["#abc", "#4682b4", "steelblue", "rgb(70, 130, 180)"] {
            let (_, _, _, a) = parse_color_to_rgba(input).unwrap();
            assert_eq!(a, 255, "{input:?} should be fully opaque");
        }
    }

    /// `PyErr::value`/`to_string` need an initialized interpreter; a bare
    /// `err.to_string()` panics with "The Python interpreter is not
    /// initialized" when this test runs standalone.
    #[test]
    fn test_parse_color_to_rgba_and_to_hex_refuse_with_pinned_identical_text() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let cases = [
                (
                    "not-a-color",
                    "invalid color 'not-a-color': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or rgb()/rgba()",
                ),
                (
                    "rgb(1,2)",
                    "invalid color 'rgb(1,2)': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or rgb()/rgba()",
                ),
                (
                    "#xy",
                    "invalid color '#xy': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or rgb()/rgba()",
                ),
            ];
            for (input, expected) in cases {
                let hex_msg = parse_color_to_hex(input).unwrap_err().value(py).to_string();
                let rgba_msg = parse_color_to_rgba(input).unwrap_err().value(py).to_string();
                assert_eq!(hex_msg, expected, "{input:?} parse_color_to_hex refusal text");
                assert_eq!(rgba_msg, expected, "{input:?} parse_color_to_rgba refusal text");
            }
        });
    }

    // ── WCAG contrast primitives (spec §6.5) ─────────────────────────────

    #[test]
    fn relative_luminance_of_black_and_white_are_the_scale_endpoints() {
        assert_eq!(relative_luminance(from_rgb(0, 0, 0)), 0.0);
        assert!((relative_luminance(from_rgb(255, 255, 255)) - 1.0).abs() < 1e-9);
    }

    /// Black-on-white is WCAG's maximum, 21:1, independent of argument order.
    #[test]
    fn contrast_ratio_black_white_is_21_and_order_free() {
        let black = from_rgb(0, 0, 0);
        let white = from_rgb(255, 255, 255);
        assert!((contrast_ratio(black, white) - 21.0).abs() < 1e-6);
        assert_eq!(contrast_ratio(black, white), contrast_ratio(white, black));
    }

    #[test]
    fn contrast_ratio_of_a_color_with_itself_is_one() {
        let gray = from_rgb(0x80, 0x80, 0x80);
        assert!((contrast_ratio(gray, gray) - 1.0).abs() < 1e-9);
    }

    /// A fully opaque `fg` ignores `bg` entirely.
    #[test]
    fn composite_over_opaque_fg_ignores_background() {
        let fg = from_rgb(0x11, 0x22, 0x33);
        let bg = from_rgb(0xff, 0xff, 0xff);
        assert_eq!(composite_over(fg, bg), fg);
    }

    /// A half-alpha white over a black background lands at mid-gray — proof
    /// the blend actually reads `fg`'s alpha rather than discarding it.
    #[test]
    fn composite_over_blends_by_alpha() {
        let half_white = from_rgba(0xff, 0xff, 0xff, 0x80);
        let black = from_rgb(0, 0, 0);
        let composed = composite_over(half_white, black);
        assert_eq!(composed.alpha, 0xff, "a composited color is always opaque");
        // 0x80 / 255 ≈ 0.502 of the way from black (0) to white (255).
        assert!((composed.red as i32 - 128).abs() <= 1, "got {composed:?}");
        assert_eq!(composed.red, composed.green);
        assert_eq!(composed.green, composed.blue);
    }

    /// A light cell (white) reads better against the dark font color than
    /// against the (also light) background, and that contrast clears AA, so
    /// the ink is the font color — the ordinary "dark label on a light cell"
    /// heatmap-annotation shape.
    #[test]
    fn contrast_ink_picks_font_color_over_a_light_cell() {
        let font_color = from_rgb(0x1f, 0x29, 0x37);
        let background = from_rgb(0xff, 0xff, 0xff);
        let cell = from_rgb(0xff, 0xff, 0xff);
        assert_eq!(contrast_ink(cell, font_color, background), font_color);
    }

    /// A dark cell (black) reads better against the light background than
    /// against the (also dark) font color, so the ink is the background —
    /// "light label on a dark cell".
    #[test]
    fn contrast_ink_picks_background_over_a_dark_cell() {
        let font_color = from_rgb(0x1f, 0x29, 0x37);
        let background = from_rgb(0xff, 0xff, 0xff);
        let cell = from_rgb(0x00, 0x00, 0x00);
        assert_eq!(contrast_ink(cell, font_color, background), background);
    }

    /// Neither `font_color` (#A0A0A0) nor `background` (#606060) clears AA
    /// against the mid-gray cell (#808080) — both sit around a 1.5:1 ratio —
    /// so the rule falls back to whichever of black/white contrasts better:
    /// black at ≈5.3:1 beats white's ≈4.0:1.
    #[test]
    fn contrast_ink_falls_back_to_black_or_white_below_the_aa_bar() {
        let font_color = from_rgb(0xa0, 0xa0, 0xa0);
        let background = from_rgb(0x60, 0x60, 0x60);
        let cell = from_rgb(0x80, 0x80, 0x80);
        assert!(contrast_ratio(font_color, cell) < CONTRAST_AA);
        assert!(contrast_ratio(background, cell) < CONTRAST_AA);
        assert_eq!(contrast_ink(cell, font_color, background), from_rgb(0, 0, 0));
    }

    /// A composited cell with alpha feeds into the same bar check as an
    /// opaque one: a half-transparent cell is read as the color it actually
    /// paints (blended with the background), never its raw, unblended
    /// channel values. Half-alpha black over a white background composites
    /// to mid-gray (≈0.21 luminance): against that, `background` (white)
    /// only reaches ≈4.0:1 — short of AA — so the rule falls to the
    /// black/white tie-break, which picks black (≈5.2:1 beats white's
    /// ≈4.0:1 against the composited gray). Read the RAW cell instead (true
    /// black, luminance 0) and `background` clears AA outright at 21:1,
    /// painting white — the wrong answer this test discriminates against.
    #[test]
    fn contrast_ink_reads_the_composited_color_not_the_raw_cell() {
        let font_color = from_rgb(0x1f, 0x29, 0x37);
        let background = from_rgb(0xff, 0xff, 0xff);
        let half_black = from_rgba(0x00, 0x00, 0x00, 0x80);
        assert_eq!(contrast_ink(half_black, font_color, background), from_rgb(0, 0, 0));
    }

    /// A fully transparent background, under either common spelling
    /// (`"#ffffff00"` / `"rgba(0,0,0,0)"`), can never win: composited over
    /// itself it always reads 1:1 (`composite_over` of an alpha-0 color
    /// returns the base unchanged), so neither spelling's own RGB leaks
    /// through, and the ink is always one of font_color/black/white —
    /// never the transparent background, which would paint an invisible
    /// label. A black cell has nothing else to contrast with and falls to
    /// white; a white cell reads well against the font color directly.
    #[test]
    fn contrast_ink_with_a_fully_transparent_background_never_returns_a_transparent_ink() {
        let font_color = from_rgb(0x1f, 0x29, 0x37);
        let black_cell = from_rgb(0, 0, 0);
        let white_cell = from_rgb(255, 255, 255);
        for background in [from_rgba(0xff, 0xff, 0xff, 0), from_rgba(0, 0, 0, 0)] {
            let ink = contrast_ink(black_cell, font_color, background);
            assert_eq!(ink.alpha, 0xff, "never a transparent ink: {ink:?}");
            assert_eq!(ink, from_rgb(0xff, 0xff, 0xff), "background {background:?}: a black cell with nothing else to contrast falls to white");

            let ink = contrast_ink(white_cell, font_color, background);
            assert_eq!(ink.alpha, 0xff, "never a transparent ink: {ink:?}");
            assert_eq!(ink, font_color, "background {background:?}: a white cell reads well against the font color with no background to compete");
        }
    }

    /// A translucent font color (`"#1f293780"`) is judged by the color it
    /// actually paints (its blend over the cell), not by its raw channel
    /// values read as if opaque. Against a white cell and the real default
    /// background `#faf7f2`: the blend reads only ≈3.05:1 against the cell
    /// (below AA), where the raw, wrongly-opaque value would read ≈14.68:1
    /// (clears AA) — a different branch entirely. The background manages
    /// only ≈1.07:1 against the same white cell, so the correctly-judged
    /// font is still the better of the two, but neither clears AA, so the
    /// answer falls to the black/white tier and lands on black (21:1 beats
    /// white's 1:1 against a white cell). The rejected, raw-value reading
    /// would instead return the translucent font color directly.
    #[test]
    fn contrast_ink_judges_a_translucent_font_color_by_its_blend() {
        let font_color = from_rgba(0x1f, 0x29, 0x37, 0x80);
        let background = from_rgb(0xfa, 0xf7, 0xf2);
        let cell = from_rgb(0xff, 0xff, 0xff);
        assert_eq!(contrast_ink(cell, font_color, background), from_rgb(0, 0, 0));
    }

    /// The real default theme (font `#1f2937`, background `#faf7f2`) on a
    /// black cell returns the background: `#faf7f2` reads ≈19.65:1 against
    /// black, decisively clearing AA, where the font itself reads only
    /// ≈1.43:1.
    #[test]
    fn contrast_ink_default_theme_on_a_black_cell_gives_the_background() {
        let font_color = from_rgb(0x1f, 0x29, 0x37);
        let background = from_rgb(0xfa, 0xf7, 0xf2);
        let cell = from_rgb(0, 0, 0);
        assert_eq!(contrast_ink(cell, font_color, background), background);
    }

    /// A translucent background (alpha `0x80`, neither opaque nor fully
    /// transparent) is read at the cell's own RGB, never blended against
    /// it as if it were a solid backing: what lies beneath a translucent
    /// background is outside the chart and unknowable. A translucent cell
    /// (also alpha `0x80`) is where this actually changes the winning ink
    /// — reading the cell at its own white, full-alpha RGB clears AA
    /// against the dark font color outright, while blending that same
    /// cell over the background's raw channel (as if the background were
    /// opaque) composites it down to a mid-gray neither candidate clears,
    /// falling to the black/white tier and painting black instead.
    #[test]
    fn contrast_ink_with_a_translucent_background_reads_the_cell_at_its_own_rgb() {
        let font_color = from_rgb(0x1f, 0x29, 0x37);
        let background = from_rgba(0, 0, 0, 0x80);
        let cell = from_rgba(255, 255, 255, 0x80);
        assert_eq!(contrast_ink(cell, font_color, background), font_color);
    }
}
