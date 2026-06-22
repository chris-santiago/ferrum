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
    /// No SVGs to compose: 1D composers receive an empty list, or a grid
    /// has every cell set to None.
    EmptyInput,
    /// An input list arity doesn't match the declared shape — `cells` length
    /// against `rows * cols`, or `row_ratios` / `col_ratios` against their
    /// respective dimensions. Carries the parameter name plus the expected
    /// and actual lengths so callers can pin the bug to a specific argument.
    LengthMismatch { what: &'static str, expected: usize, got: usize },
    /// `row_ratios` or `col_ratios` sums to ≤ 0 (every entry zero or
    /// negative). The grid would have zero physical extent on at least
    /// one axis.
    InvalidRatios,
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
            CompositorError::LengthMismatch { what, expected, got } =>
                write!(f, "{what} length mismatch: expected {expected}, got {got}"),
            CompositorError::InvalidRatios => write!(f, "row/col ratios must sum to a positive value"),
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

impl TryFrom<&str> for VerticalAlign {
    type Error = String;

    /// Parse a vertical-alignment string.
    ///
    /// Accepted: `"top"`, `"center"`, `"bottom"`.
    /// Returns `Err` for any other value.
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "top" => Ok(VerticalAlign::Top),
            "center" => Ok(VerticalAlign::Center),
            "bottom" => Ok(VerticalAlign::Bottom),
            other => Err(format!(
                "align must be one of 'top'|'center'|'bottom', got '{other}'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HorizontalAlign {
    Left,
    Center,
    Right,
}

impl TryFrom<&str> for HorizontalAlign {
    type Error = String;

    /// Parse a horizontal-alignment string.
    ///
    /// Accepted: `"left"`, `"center"`, `"right"`.
    /// Returns `Err` for any other value.
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "left" => Ok(HorizontalAlign::Left),
            "center" => Ok(HorizontalAlign::Center),
            "right" => Ok(HorizontalAlign::Right),
            other => Err(format!(
                "align must be one of 'left'|'center'|'right', got '{other}'"
            )),
        }
    }
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

/// Rewrite every clipPath `id="…"` and `url(#…)` reference in a single child
/// SVG body so each composited cell uses globally-unique IDs. Covers the three
/// id families ferrum emits: `ferrum-clip-`, `ferrum-colorbar-`, and
/// `ferrum-legend-clip-`.
///
/// Each ferrum chart numbers its clipPaths per-panel starting at 0
/// (`ferrum-clip-0`, `ferrum-clip-1`, ...). When multiple single-panel charts
/// are composited (JointChart, ClusterMapChart, HConcat/VConcat) they all
/// define `id="ferrum-clip-0"`, and the SVG renderer silently picks one
/// definition — clipping every cell by the wrong rect. The same collision hits
/// `ferrum-colorbar-0` colorbar gradients and `ferrum-legend-clip-0` legend
/// `clip_height` clips. Prefixing each cell's IDs with `cellNN-` makes them
/// disjoint while preserving each body's internal `id` ↔ `url(#id)` references.
pub(crate) fn uniquify_clip_ids(body: &str, cell_idx: usize) -> String {
    let clip_prefix = format!("cell{cell_idx}-ferrum-clip-");
    let colorbar_prefix = format!("cell{cell_idx}-ferrum-colorbar-");
    let legend_clip_prefix = format!("cell{cell_idx}-ferrum-legend-clip-");
    body
        .replace("id=\"ferrum-clip-", &format!("id=\"{clip_prefix}"))
        .replace("url(#ferrum-clip-", &format!("url(#{clip_prefix}"))
        .replace("id=\"ferrum-colorbar-", &format!("id=\"{colorbar_prefix}"))
        .replace("url(#ferrum-colorbar-", &format!("url(#{colorbar_prefix}"))
        .replace("id=\"ferrum-legend-clip-", &format!("id=\"{legend_clip_prefix}"))
        .replace("url(#ferrum-legend-clip-", &format!("url(#{legend_clip_prefix}"))
}

// ---------------------------------------------------------------------------
// Shared emission helpers
// ---------------------------------------------------------------------------

/// Emit the outer `<svg ...>` opening tag with width/height/viewBox set
/// to the same `(w, h)` dimensions. The doc-comment in [`parse_svg_root`]
/// notes that attribute order matters for the round-trip; this helper
/// is the single source of truth for that order.
pub(crate) fn write_svg_open(out: &mut String, w: f64, h: f64) {
    out.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        fmt_f(w), fmt_f(h), fmt_f(w), fmt_f(h),
    ));
}

/// Emit a per-cell `<g transform="translate(x,y)">…</g>` wrapper, with
/// `body` first passed through [`strip_font_defs`] when `is_first` is
/// false (font-face defs should appear only in the first emitted cell)
/// and then through [`uniquify_clip_ids`] for clipPath collision safety.
///
/// `idx` is the cell's global index (used to namespace the clip IDs).
/// Composers must pass `is_first = true` exactly once per output —
/// either for the first cell when no holes precede it (1D composers)
/// or for the first non-`None` cell (grid composer).
///
/// Use this when the cell's native dimensions match its allocated slot.
/// For grid cells in slots that differ from native (ratio-weighted
/// allocations smaller or larger than the cell's intrinsic size), use
/// [`write_scaled_cell`] which wraps in a nested `<svg viewBox>` that
/// stretches the body to fit.
pub(crate) fn write_cell(out: &mut String, x: f64, y: f64, idx: usize, body: &str, is_first: bool) {
    out.push_str(&format!(
        r#"<g transform="translate({},{})">"#,
        fmt_f(x), fmt_f(y),
    ));
    let intermediate: std::borrow::Cow<'_, str> = if is_first {
        std::borrow::Cow::Borrowed(body)
    } else {
        std::borrow::Cow::Owned(strip_font_defs(body))
    };
    out.push_str(&uniquify_clip_ids(&intermediate, idx));
    out.push_str("</g>");
}

/// Emit a per-cell wrapper that scales `body` from its native
/// `(native_w, native_h)` into the allocated slot `(slot_w, slot_h)`.
///
/// Wraps the body in a nested `<svg x=... y=... width=... height=...
/// viewBox="0 0 native_w native_h" preserveAspectRatio="none">…</svg>`.
/// `preserveAspectRatio="none"` lets the cell stretch independently in
/// X and Y so a JointChart marginal at native 600×400 fits a 600×80 slot
/// by squishing the Y axis while leaving X aligned with the centre cell.
pub(crate) fn write_scaled_cell(
    out: &mut String,
    x: f64,
    y: f64,
    slot_w: f64,
    slot_h: f64,
    native_w: f64,
    native_h: f64,
    idx: usize,
    body: &str,
    is_first: bool,
) {
    out.push_str(&format!(
        r#"<svg x="{}" y="{}" width="{}" height="{}" viewBox="0 0 {} {}" preserveAspectRatio="none">"#,
        fmt_f(x), fmt_f(y),
        fmt_f(slot_w), fmt_f(slot_h),
        fmt_f(native_w), fmt_f(native_h),
    ));
    let intermediate: std::borrow::Cow<'_, str> = if is_first {
        std::borrow::Cow::Borrowed(body)
    } else {
        std::borrow::Cow::Owned(strip_font_defs(body))
    };
    out.push_str(&uniquify_clip_ids(&intermediate, idx));
    out.push_str("</svg>");
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
    write_svg_open(&mut out, total_width, max_height);

    let mut x_offset = 0.0_f64;
    for (i, p) in parsed.iter().enumerate() {
        let y_offset = match align {
            VerticalAlign::Top => 0.0,
            VerticalAlign::Center => (max_height - p.height) / 2.0,
            VerticalAlign::Bottom => max_height - p.height,
        };
        write_cell(&mut out, x_offset, y_offset, i, p.body, i == 0);
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
    write_svg_open(&mut out, max_width, total_height);

    let mut y_offset = 0.0_f64;
    for (i, p) in parsed.iter().enumerate() {
        let x_offset = match align {
            HorizontalAlign::Left => 0.0,
            HorizontalAlign::Center => (max_width - p.width) / 2.0,
            HorizontalAlign::Right => max_width - p.width,
        };
        write_cell(&mut out, x_offset, y_offset, i, p.body, i == 0);
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

    /// Two composited children each emit a `clip_height` legend, so both bodies
    /// carry `id="ferrum-legend-clip-0"` + `clip-path="url(#ferrum-legend-clip-0)"`.
    /// The uniquifier must namespace them per cell so the merged SVG holds two
    /// distinct legend-clip ids and each `url(#…)` reference resolves to the def
    /// in its own cell. Mirrors the JointChart/HConcat collision the uniquifier
    /// exists to prevent (`ferrum-clip-` / `ferrum-colorbar-`).
    #[test]
    fn hconcat_legend_clip_ids_uniquified_across_cells() {
        let legend_body = r#"<defs><clipPath id="ferrum-legend-clip-0"><rect x="0" y="0" width="40" height="30"/></clipPath></defs><g clip-path="url(#ferrum-legend-clip-0)"><rect/></g>"#;
        let a = make_svg(100.0, 80.0, legend_body);
        let b = make_svg(100.0, 80.0, legend_body);
        let out = compose_svg_horizontal(&[a, b], 0.0, VerticalAlign::Top).unwrap();

        // Original colliding id must be fully rewritten away — neither the def
        // nor the reference may keep the bare (cell-less) `ferrum-legend-clip-`
        // prefix that both children share.
        assert!(
            !out.contains(r#"id="ferrum-legend-clip-"#),
            "unprefixed legend-clip def leaked: {out}",
        );
        assert!(
            !out.contains("url(#ferrum-legend-clip-"),
            "unprefixed legend-clip reference leaked: {out}",
        );

        // Each cell gets its own namespaced id, def + reference in lockstep.
        for (cell, id) in [(0_usize, "cell0-ferrum-legend-clip-0"), (1, "cell1-ferrum-legend-clip-0")] {
            assert!(
                out.contains(&format!(r#"id="{id}""#)),
                "missing legend-clip def for cell {cell}: {out}",
            );
            assert!(
                out.contains(&format!(r#"clip-path="url(#{id})""#)),
                "missing legend-clip reference for cell {cell}: {out}",
            );
        }

        // Exactly two distinct legend-clip defs (one per cell, no shared id).
        assert_eq!(
            out.matches(r#"<clipPath id="cell"#).count(),
            2,
            "expected 2 distinct legend-clip defs: {out}",
        );
    }

    // --- RSUP-09: VerticalAlign / HorizontalAlign string→enum contracts ---

    #[test]
    fn vertical_align_try_from_maps_known_strings() {
        assert_eq!(VerticalAlign::try_from("top"),    Ok(VerticalAlign::Top));
        assert_eq!(VerticalAlign::try_from("center"), Ok(VerticalAlign::Center));
        assert_eq!(VerticalAlign::try_from("bottom"), Ok(VerticalAlign::Bottom));
    }

    #[test]
    fn vertical_align_try_from_rejects_unknown_string() {
        let err = VerticalAlign::try_from("middle").unwrap_err();
        assert!(
            err.contains("align must be one of 'top'|'center'|'bottom'"),
            "unexpected error message: {err}",
        );
        assert!(err.contains("middle"), "error should echo the bad input: {err}");
    }

    #[test]
    fn horizontal_align_try_from_maps_known_strings() {
        assert_eq!(HorizontalAlign::try_from("left"),   Ok(HorizontalAlign::Left));
        assert_eq!(HorizontalAlign::try_from("center"), Ok(HorizontalAlign::Center));
        assert_eq!(HorizontalAlign::try_from("right"),  Ok(HorizontalAlign::Right));
    }

    #[test]
    fn horizontal_align_try_from_rejects_unknown_string() {
        let err = HorizontalAlign::try_from("start").unwrap_err();
        assert!(
            err.contains("align must be one of 'left'|'center'|'right'"),
            "unexpected error message: {err}",
        );
        assert!(err.contains("start"), "error should echo the bad input: {err}");
    }
}
