//! SVG compositor for horizontal (`|`) and vertical (`&`) chart composition.
//!
//! Parses our deterministic-output SVGs (per Phase 7 §4.4 the `<svg>` root has a fixed
//! attribute order) and stitches them inside a wrapping `<svg>` with
//! `<g transform="translate(...)">` per child.
//!
//! Because our `SvgBuffer` always emits
//! `<svg xmlns="..." width="W" height="H" viewBox="0 0 W H">`
//! in that exact order, we can extract W/H by string search without a real XML parser.

use super::svg::fmt_f;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum CompositorError {
    NoSvgRoot,
    MalformedRoot,
    NoClosingTag,
    MissingAttr(String),
    MalformedAttr(String),
    AttrNotNumeric(String),
    EmptyInput,
}

impl std::fmt::Display for CompositorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompositorError::NoSvgRoot => write!(f, "input does not contain <svg ...>"),
            CompositorError::MalformedRoot => write!(f, "<svg> open tag is malformed"),
            CompositorError::NoClosingTag => write!(f, "input does not contain </svg>"),
            CompositorError::MissingAttr(n) => write!(f, "<svg> missing required attr '{n}'"),
            CompositorError::MalformedAttr(n) => write!(f, "<svg> attr '{n}' is malformed"),
            CompositorError::AttrNotNumeric(n) => write!(f, "<svg> attr '{n}' is not numeric"),
            CompositorError::EmptyInput => write!(f, "no svgs to compose"),
        }
    }
}

impl std::error::Error for CompositorError {}

// ---------------------------------------------------------------------------
// Alignment enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerticalAlign {
    Top,
    Center,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HorizontalAlign {
    Left,
    Center,
    Right,
}

// ---------------------------------------------------------------------------
// Internal parsing
// ---------------------------------------------------------------------------

pub(crate) struct ParsedSvg<'a> {
    pub(crate) width: f64,
    pub(crate) height: f64,
    pub(crate) body: &'a str,
}

/// Parse the width and height attributes from an SVG root element.
/// Returns a `ParsedSvg` where `body` is the content between the opening
/// `<svg ...>` tag and the closing `</svg>`.
pub(crate) fn parse_svg_root(svg: &str) -> Result<ParsedSvg<'_>, CompositorError> {
    let svg_open_start = svg.find("<svg").ok_or(CompositorError::NoSvgRoot)?;
    let svg_open_end = svg[svg_open_start..]
        .find('>')
        .ok_or(CompositorError::MalformedRoot)?
        + svg_open_start
        + 1;
    let svg_close = svg.rfind("</svg>").ok_or(CompositorError::NoClosingTag)?;

    let attrs = &svg[svg_open_start..svg_open_end];
    let width = extract_attr_f64(attrs, "width")?;
    let height = extract_attr_f64(attrs, "height")?;

    Ok(ParsedSvg {
        width,
        height,
        body: &svg[svg_open_end..svg_close],
    })
}

fn extract_attr_f64(attrs: &str, name: &str) -> Result<f64, CompositorError> {
    let needle = format!(r#"{}=""#, name);
    let start = attrs
        .find(&needle)
        .ok_or_else(|| CompositorError::MissingAttr(name.into()))?;
    let val_start = start + needle.len();
    let val_end = attrs[val_start..]
        .find('"')
        .ok_or_else(|| CompositorError::MalformedAttr(name.into()))?
        + val_start;
    attrs[val_start..val_end]
        .parse::<f64>()
        .map_err(|_| CompositorError::AttrNotNumeric(name.into()))
}

// ---------------------------------------------------------------------------
// Font-defs deduplication
// ---------------------------------------------------------------------------

const FONT_DEFS_MARKER: &str = "<defs><style>@font-face";

/// Strip the `<defs><style>@font-face ...</style></defs>` block from `body`.
/// Called only for non-first children to avoid duplicating embedded fonts.
/// Allocates a new `String`; returns `body.to_string()` unchanged if the marker
/// is not found.
pub(crate) fn strip_font_defs(body: &str) -> String {
    let Some(start) = body.find(FONT_DEFS_MARKER) else {
        return body.to_string();
    };
    let defs_start = body[..start].rfind("<defs").unwrap_or(start);
    let defs_end_marker = "</defs>";
    let Some(end_rel) = body[defs_start..].find(defs_end_marker) else {
        return body.to_string();
    };
    let defs_end = defs_start + end_rel + defs_end_marker.len();
    let mut out = String::with_capacity(body.len() - (defs_end - defs_start));
    out.push_str(&body[..defs_start]);
    out.push_str(&body[defs_end..]);
    out
}

// ---------------------------------------------------------------------------
// clipPath id uniquification (cross-cell collision fix)
// ---------------------------------------------------------------------------

/// Rewrite every `id="ferrum-clip-N"` and `url(#ferrum-clip-N)` reference in
/// a single child SVG body so each composited cell uses globally-unique IDs.
///
/// Each ferrum chart numbers its clipPaths per-panel starting at 0
/// (`ferrum-clip-0`, `ferrum-clip-1`, ...). When multiple single-panel charts
/// are composited (JointChart, ClusterMapChart, HConcat/VConcat) they all
/// define `id="ferrum-clip-0"`, and the SVG renderer silently picks one
/// definition — clipping every cell by the wrong rect. Prefixing each cell's
/// IDs with `cellNN-` makes them disjoint while preserving each body's
/// internal `id` ↔ `url(#id)` references.
pub(crate) fn uniquify_clip_ids(body: &str, cell_idx: usize) -> String {
    let clip_prefix = format!("cell{cell_idx}-ferrum-clip-");
    let colorbar_prefix = format!("cell{cell_idx}-ferrum-colorbar-");
    body
        .replace("id=\"ferrum-clip-", &format!("id=\"{clip_prefix}"))
        .replace("url(#ferrum-clip-", &format!("url(#{clip_prefix}"))
        .replace("id=\"ferrum-colorbar-", &format!("id=\"{colorbar_prefix}"))
        .replace("url(#ferrum-colorbar-", &format!("url(#{colorbar_prefix}"))
}

// ---------------------------------------------------------------------------
// Public compose functions
// ---------------------------------------------------------------------------

/// Compose SVGs side-by-side (horizontal concat, `|` operator).
///
/// The total width is the sum of child widths plus `spacing` between each pair.
/// The canvas height is the tallest child; shorter children are aligned per `align`.
/// Font-face definitions are included only from the first child.
pub fn compose_svg_horizontal(
    svgs: &[String],
    spacing: f64,
    align: VerticalAlign,
) -> Result<String, CompositorError> {
    if svgs.is_empty() {
        return Err(CompositorError::EmptyInput);
    }
    let parsed: Vec<ParsedSvg<'_>> = svgs
        .iter()
        .map(|s| parse_svg_root(s))
        .collect::<Result<_, _>>()?;

    let total_width: f64 =
        parsed.iter().map(|p| p.width).sum::<f64>() + spacing * (svgs.len() - 1) as f64;
    let max_height: f64 = parsed.iter().map(|p| p.height).fold(0.0_f64, f64::max);

    let mut out =
        String::with_capacity(svgs.iter().map(|s| s.len()).sum::<usize>() + 256);
    out.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        fmt_f(total_width),
        fmt_f(max_height),
        fmt_f(total_width),
        fmt_f(max_height),
    ));

    let mut x_offset = 0.0_f64;
    for (i, p) in parsed.iter().enumerate() {
        let y_offset = match align {
            VerticalAlign::Top => 0.0,
            VerticalAlign::Center => (max_height - p.height) / 2.0,
            VerticalAlign::Bottom => max_height - p.height,
        };
        out.push_str(&format!(
            r#"<g transform="translate({},{})">"#,
            fmt_f(x_offset),
            fmt_f(y_offset),
        ));
        let intermediate = if i == 0 { p.body.to_string() } else { strip_font_defs(p.body) };
        out.push_str(&uniquify_clip_ids(&intermediate, i));
        out.push_str("</g>");
        x_offset += p.width + spacing;
    }
    out.push_str("</svg>");
    Ok(out)
}

/// Compose SVGs stacked vertically (vertical concat, `&` operator).
///
/// The canvas width is the widest child; shorter children are aligned per `align`.
/// The total height is the sum of child heights plus `spacing` between each pair.
/// Font-face definitions are included only from the first child.
pub fn compose_svg_vertical(
    svgs: &[String],
    spacing: f64,
    align: HorizontalAlign,
) -> Result<String, CompositorError> {
    if svgs.is_empty() {
        return Err(CompositorError::EmptyInput);
    }
    let parsed: Vec<ParsedSvg<'_>> = svgs
        .iter()
        .map(|s| parse_svg_root(s))
        .collect::<Result<_, _>>()?;

    let max_width: f64 = parsed.iter().map(|p| p.width).fold(0.0_f64, f64::max);
    let total_height: f64 =
        parsed.iter().map(|p| p.height).sum::<f64>() + spacing * (svgs.len() - 1) as f64;

    let mut out =
        String::with_capacity(svgs.iter().map(|s| s.len()).sum::<usize>() + 256);
    out.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        fmt_f(max_width),
        fmt_f(total_height),
        fmt_f(max_width),
        fmt_f(total_height),
    ));

    let mut y_offset = 0.0_f64;
    for (i, p) in parsed.iter().enumerate() {
        let x_offset = match align {
            HorizontalAlign::Left => 0.0,
            HorizontalAlign::Center => (max_width - p.width) / 2.0,
            HorizontalAlign::Right => max_width - p.width,
        };
        out.push_str(&format!(
            r#"<g transform="translate({},{})">"#,
            fmt_f(x_offset),
            fmt_f(y_offset),
        ));
        let intermediate = if i == 0 { p.body.to_string() } else { strip_font_defs(p.body) };
        out.push_str(&uniquify_clip_ids(&intermediate, i));
        out.push_str("</g>");
        y_offset += p.height + spacing;
    }
    out.push_str("</svg>");
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal but structurally valid ferrum-style SVG.
    fn make_svg(w: f64, h: f64, body: &str) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">{}</svg>"#,
            w, h, w, h, body,
        )
    }

    #[test]
    fn parse_svg_root_extracts_dimensions() {
        let s = make_svg(100.0, 50.0, "<rect/>");
        let parsed = parse_svg_root(&s).unwrap();
        assert_eq!(parsed.width, 100.0);
        assert_eq!(parsed.height, 50.0);
        assert_eq!(parsed.body, "<rect/>");
    }

    #[test]
    fn hconcat_two_svgs_sums_widths_plus_spacing() {
        let a = make_svg(100.0, 50.0, "<circle/>");
        let b = make_svg(80.0, 60.0, "<rect/>");
        let out = compose_svg_horizontal(&[a, b], 10.0, VerticalAlign::Top).unwrap();
        // Total width = 100 + 10 + 80 = 190; max height = 60
        assert!(out.contains(r#"width="190""#), "out: {out}");
        assert!(out.contains(r#"height="60""#), "out: {out}");
    }

    #[test]
    fn hconcat_wraps_each_child_in_translate_g() {
        let a = make_svg(100.0, 50.0, "<circle/>");
        let b = make_svg(80.0, 60.0, "<rect/>");
        let out = compose_svg_horizontal(&[a, b], 10.0, VerticalAlign::Top).unwrap();
        assert!(out.contains(r#"<g transform="translate(0,0)">"#), "out: {out}");
        assert!(out.contains(r#"<g transform="translate(110,0)">"#), "out: {out}");
    }

    #[test]
    fn vconcat_two_svgs_sums_heights() {
        let a = make_svg(100.0, 50.0, "<circle/>");
        let b = make_svg(80.0, 60.0, "<rect/>");
        let out = compose_svg_vertical(&[a, b], 10.0, HorizontalAlign::Left).unwrap();
        // Total height = 50 + 10 + 60 = 120; max width = 100
        assert!(out.contains(r#"width="100""#), "out: {out}");
        assert!(out.contains(r#"height="120""#), "out: {out}");
    }

    #[test]
    fn font_defs_stripped_from_second_child_only() {
        let a = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50" viewBox="0 0 100 50"><defs><style>@font-face{{src:url("data:fake")}}</style></defs><circle/></svg>"#,
        );
        let b = a.clone();
        let out = compose_svg_horizontal(&[a, b], 0.0, VerticalAlign::Top).unwrap();
        // Exactly one occurrence of @font-face in the composed output
        assert_eq!(
            out.matches("@font-face").count(),
            1,
            "expected 1 @font-face, got: {}",
            out.matches("@font-face").count()
        );
    }

    #[test]
    fn compose_three_svgs_hconcat() {
        let a = make_svg(50.0, 50.0, "<circle/>");
        let b = make_svg(50.0, 50.0, "<rect/>");
        let c = make_svg(50.0, 50.0, "<line/>");
        let out = compose_svg_horizontal(&[a, b, c], 5.0, VerticalAlign::Top).unwrap();
        // Total = 50 + 5 + 50 + 5 + 50 = 160
        assert!(out.contains(r#"width="160""#), "out: {out}");
    }

    #[test]
    fn compose_empty_returns_error() {
        assert_eq!(
            compose_svg_horizontal(&[], 0.0, VerticalAlign::Top),
            Err(CompositorError::EmptyInput),
        );
    }

    #[test]
    fn compose_single_svg_roundtrips_body() {
        let a = make_svg(200.0, 100.0, "<path d=\"M0,0\"/>");
        let out = compose_svg_horizontal(&[a], 10.0, VerticalAlign::Top).unwrap();
        assert!(out.contains(r#"width="200""#));
        assert!(out.contains(r#"height="100""#));
        assert!(out.contains(r#"<path d="M0,0"/>"#));
    }
}
