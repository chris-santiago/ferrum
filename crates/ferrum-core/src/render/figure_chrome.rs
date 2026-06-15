//! Figure-level chrome bands for composite charts.
//!
//! Wraps a composed SVG (from `compose_svg_horizontal`, `compose_svg_vertical`,
//! or `compose_svg_grid`) with an optional title + subtitle band above the panels
//! and an optional caption band below the panels.
//!
//! **Byte-stability guarantee:** when all three of `title`, `subtitle`, and `caption`
//! are `None`, `wrap_with_chrome` returns the input SVG unmodified. Callers can rely
//! on this to preserve golden-test stability.
//!
//! Typography is consistent with single-chart title rendering in `scene_build.rs`:
//! - title: 16 px, weight 600, color #1F2937 (ferrum dark text), font-family "Inter"
//! - subtitle: 13 px (≈ 0.81 × title size), weight normal, color #6B7280 (ferrum label gray)
//! - caption: 11 px, weight normal, color #6B7280 (ferrum label gray)
//!
//! These constants match `layout::ThemeTypography` defaults so the chrome looks
//! consistent with per-chart titles even when no theme dict is passed to the PyO3
//! compositor binding (which operates at the SVG string level, post-render).

use super::svg::{escape_text, fmt_f};
use super::compositor::{parse_svg_root, write_svg_open, CompositorError};

// ---------------------------------------------------------------------------
// Constants (consistent with ThemeTypography defaults + single-chart titles)
// ---------------------------------------------------------------------------

const FIGURE_TITLE_FONT_SIZE: f64 = 16.0;
const FIGURE_SUBTITLE_FONT_SIZE: f64 = 13.0;
const FIGURE_CAPTION_FONT_SIZE: f64 = 11.0;

/// Horizontal inset (in px) from the panel edges to the chrome text when the
/// chrome anchor is `Start`/`End`. Mirrors `ThemePadding::default().padding`
/// (the value the single-chart title inset uses — see `layout/mod.rs:172,513`),
/// so figure-level chrome aligns with per-chart titles.
pub const DEFAULT_CHROME_INSET: f64 = 16.0;

/// Vertical gap between the top of the SVG and the title baseline.
const TITLE_TOP_PAD: f64 = 6.0;
/// Vertical gap between the title baseline and the subtitle baseline.
const TITLE_SUBTITLE_GAP: f64 = 4.0;
/// Vertical gap between the last title-band line and the composed panels.
const HEADER_BOTTOM_PAD: f64 = 8.0;
/// Vertical gap between the panels and the caption baseline.
const CAPTION_TOP_PAD: f64 = 6.0;
/// Vertical gap between the caption baseline and the bottom of the SVG.
const CAPTION_BOTTOM_PAD: f64 = 4.0;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Horizontal alignment for figure-level chrome text (title, subtitle, caption).
///
/// Governs all three chrome lines uniformly. The binding parses the user-facing
/// anchor string once into this typed enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChromeAnchor {
    /// Flush-left: `x = left_inset`, `text-anchor="start"`.
    #[default]
    Start,
    /// Centered: `x = panel_w / 2`, `text-anchor="middle"`.
    Middle,
    /// Flush-right: `x = panel_w - right_inset`, `text-anchor="end"`.
    End,
}

/// Chrome parameters for a figure-level band.
///
/// `title`/`subtitle`/`caption` are `Option<&str>`. When all three are `None`,
/// `wrap_with_chrome` returns the input unchanged (byte-identical round-trip).
///
/// `left_inset`/`right_inset`/`anchor` resolve horizontal placement. The
/// [`Default`] impl is hand-written so the insets default to
/// [`DEFAULT_CHROME_INSET`] (16.0) rather than `f64::default()` (0.0).
#[derive(Debug, Clone, Copy)]
pub struct FigureChrome<'a> {
    /// Large text above the composed panels (bold, 16 px).
    pub title: Option<&'a str>,
    /// Smaller text below the title, above the panels (normal weight, 13 px).
    pub subtitle: Option<&'a str>,
    /// Small muted text below the composed panels (11 px, label gray).
    pub caption: Option<&'a str>,
    /// Left inset (px) used when `anchor == Start`.
    pub left_inset: f64,
    /// Right inset (px) used when `anchor == End`.
    pub right_inset: f64,
    /// Horizontal alignment of all three chrome lines.
    pub anchor: ChromeAnchor,
}

impl Default for FigureChrome<'_> {
    fn default() -> Self {
        Self {
            title: None,
            subtitle: None,
            caption: None,
            left_inset: DEFAULT_CHROME_INSET,
            right_inset: DEFAULT_CHROME_INSET,
            anchor: ChromeAnchor::Start,
        }
    }
}

impl FigureChrome<'_> {
    /// Returns `true` when no chrome needs to be emitted.
    pub fn is_empty(&self) -> bool {
        self.title.is_none() && self.subtitle.is_none() && self.caption.is_none()
    }

    /// Resolve the `x` coordinate and `text-anchor` value for chrome text,
    /// given the composed panel width. Shared by header and footer emitters so
    /// all three lines align identically.
    fn resolve_anchor(&self, panel_w: f64) -> (f64, &'static str) {
        match self.anchor {
            ChromeAnchor::Start => (self.left_inset, "start"),
            ChromeAnchor::Middle => (panel_w / 2.0, "middle"),
            ChromeAnchor::End => (panel_w - self.right_inset, "end"),
        }
    }
}

/// Wrap a composed SVG with figure-level title/subtitle/caption bands.
///
/// When `chrome.is_empty()` this function returns `svg` unmodified —
/// callers MUST rely on this guarantee for golden-test byte stability.
///
/// # Layout
///
/// ```text
/// ┌────────────────────────────────────────────┐
/// │  [title]                                   │  ← TITLE_TOP_PAD + font_size
/// │  [subtitle]                                │  ← TITLE_SUBTITLE_GAP + font_size
/// │                        HEADER_BOTTOM_PAD   │
/// │  ┌──────────────────────────────────────┐  │
/// │  │  (composed panels — inner SVG body)  │  │
/// │  └──────────────────────────────────────┘  │
/// │                        CAPTION_TOP_PAD     │
/// │  [caption]                                 │  ← font_size
/// │                        CAPTION_BOTTOM_PAD  │
/// └────────────────────────────────────────────┘
/// ```
pub fn wrap_with_chrome(svg: &str, chrome: FigureChrome<'_>) -> Result<String, CompositorError> {
    if chrome.is_empty() {
        return Ok(svg.to_string());
    }

    let parsed = parse_svg_root(svg)?;
    let panel_w = parsed.width;
    let panel_h = parsed.height;

    // --- Header height ---
    let header_h = compute_header_height(&chrome);

    // --- Footer height ---
    let footer_h = compute_footer_height(&chrome);

    // --- Total canvas ---
    let total_w = panel_w;
    let total_h = panel_h + header_h + footer_h;

    let mut out = String::with_capacity(svg.len() + 512);
    write_svg_open(&mut out, total_w, total_h);

    // Emit header band text nodes
    emit_header(&mut out, &chrome, panel_w, header_h);

    // Wrap inner panels with a vertical offset by header_h.
    // We re-wrap the full inner SVG (including its <svg> open tag) in a
    // nested <svg> positioned at y=header_h.  This is the cleanest approach:
    // it preserves all internal coordinate systems (clip paths, gradients, etc.)
    // without re-parsing the inner body's structure.
    out.push_str(&format!(
        r#"<svg x="0" y="{}" width="{}" height="{}" viewBox="0 0 {} {}" preserveAspectRatio="none">"#,
        fmt_f(header_h),
        fmt_f(panel_w),
        fmt_f(panel_h),
        fmt_f(panel_w),
        fmt_f(panel_h),
    ));
    out.push_str(parsed.body);
    out.push_str("</svg>");

    // Emit footer band text nodes
    emit_footer(&mut out, &chrome, panel_w, panel_h + header_h);

    out.push_str("</svg>");
    Ok(out)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn compute_header_height(chrome: &FigureChrome<'_>) -> f64 {
    if chrome.title.is_none() && chrome.subtitle.is_none() {
        return 0.0;
    }
    let mut h = TITLE_TOP_PAD;
    if chrome.title.is_some() {
        h += FIGURE_TITLE_FONT_SIZE;
    }
    if chrome.subtitle.is_some() {
        if chrome.title.is_some() {
            h += TITLE_SUBTITLE_GAP;
        } else {
            // subtitle without title: treat subtitle as the first line
            h += TITLE_TOP_PAD; // additional top pad to mirror title case
        }
        h += FIGURE_SUBTITLE_FONT_SIZE;
    }
    h += HEADER_BOTTOM_PAD;
    h
}

fn compute_footer_height(chrome: &FigureChrome<'_>) -> f64 {
    if chrome.caption.is_none() {
        return 0.0;
    }
    CAPTION_TOP_PAD + FIGURE_CAPTION_FONT_SIZE + CAPTION_BOTTOM_PAD
}

/// Emit `<text>` elements for the header band (title + subtitle).
///
/// Both lines share the horizontal placement resolved from `chrome.anchor`
/// and the insets (governed uniformly with the caption). The y coordinates are
/// absolute within the outer SVG canvas.
fn emit_header(out: &mut String, chrome: &FigureChrome<'_>, panel_w: f64, _header_h: f64) {
    let (x, text_anchor) = chrome.resolve_anchor(panel_w);

    // title baseline
    let mut y = TITLE_TOP_PAD + FIGURE_TITLE_FONT_SIZE;

    if let Some(title) = chrome.title {
        out.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" fill=\"#1f2937\" font-family=\"Inter\" font-size=\"{}\" font-weight=\"600\" text-anchor=\"{}\">{}</text>",
            fmt_f(x),
            fmt_f(y),
            fmt_f(FIGURE_TITLE_FONT_SIZE),
            text_anchor,
            escape_text(title),
        ));
        y += TITLE_SUBTITLE_GAP + FIGURE_SUBTITLE_FONT_SIZE;
    } else if chrome.subtitle.is_some() {
        // subtitle-only: push down by a small extra top pad for visual symmetry
        y = TITLE_TOP_PAD + FIGURE_TITLE_FONT_SIZE + TITLE_SUBTITLE_GAP + FIGURE_SUBTITLE_FONT_SIZE;
    }

    if let Some(subtitle) = chrome.subtitle {
        let subtitle_y = if chrome.title.is_some() {
            y
        } else {
            // subtitle alone: use the y we computed above
            TITLE_TOP_PAD + FIGURE_SUBTITLE_FONT_SIZE
        };
        out.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" fill=\"#6b7280\" font-family=\"Inter\" font-size=\"{}\" text-anchor=\"{}\">{}</text>",
            fmt_f(x),
            fmt_f(subtitle_y),
            fmt_f(FIGURE_SUBTITLE_FONT_SIZE),
            text_anchor,
            escape_text(subtitle),
        ));
    }
}

/// Emit the `<text>` element for the footer caption band.
///
/// `panels_bottom_y` is the y-coordinate at the bottom of the composed panels
/// within the outer SVG (i.e., `header_h + panel_h`). The caption shares the
/// horizontal placement resolved from `chrome.anchor` and the insets, matching
/// the header lines.
fn emit_footer(out: &mut String, chrome: &FigureChrome<'_>, panel_w: f64, panels_bottom_y: f64) {
    let Some(caption) = chrome.caption else { return };
    let (x, text_anchor) = chrome.resolve_anchor(panel_w);
    let caption_y = panels_bottom_y + CAPTION_TOP_PAD + FIGURE_CAPTION_FONT_SIZE;
    out.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" fill=\"#6b7280\" font-family=\"Inter\" font-size=\"{}\" text-anchor=\"{}\">{}</text>",
        fmt_f(x),
        fmt_f(caption_y),
        fmt_f(FIGURE_CAPTION_FONT_SIZE),
        text_anchor,
        escape_text(caption),
    ));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_svg(w: f64, h: f64) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}"><rect x="0" y="0" width="{}" height="{}" fill="red"/></svg>"#,
            w, h, w, h, w, h,
        )
    }

    #[test]
    fn no_chrome_returns_input_unmodified() {
        let svg = make_svg(200.0, 100.0);
        let chrome = FigureChrome::default();
        let result = wrap_with_chrome(&svg, chrome).unwrap();
        assert_eq!(result, svg, "empty chrome must be byte-identical round-trip");
    }

    #[test]
    fn title_only_expands_height() {
        let svg = make_svg(200.0, 100.0);
        let chrome = FigureChrome { title: Some("My Figure"), ..Default::default() };
        let result = wrap_with_chrome(&svg, chrome).unwrap();
        // Height must exceed the original 100
        let parsed = parse_svg_root(&result).unwrap();
        assert!(parsed.height > 100.0, "height should grow with title: {}", parsed.height);
        assert_eq!(parsed.width, 200.0, "width unchanged");
        assert!(result.contains("My Figure"), "title text present");
    }

    #[test]
    fn caption_only_expands_height() {
        let svg = make_svg(200.0, 100.0);
        let chrome = FigureChrome { caption: Some("Source: foo"), ..Default::default() };
        let result = wrap_with_chrome(&svg, chrome).unwrap();
        let parsed = parse_svg_root(&result).unwrap();
        assert!(parsed.height > 100.0, "height should grow with caption: {}", parsed.height);
        assert_eq!(parsed.width, 200.0, "width unchanged");
        assert!(result.contains("Source: foo"), "caption text present");
    }

    #[test]
    fn title_subtitle_caption_all_present() {
        let svg = make_svg(300.0, 150.0);
        let chrome = FigureChrome {
            title: Some("Title"),
            subtitle: Some("Subtitle"),
            caption: Some("Caption"),
            ..Default::default()
        };
        let result = wrap_with_chrome(&svg, chrome).unwrap();
        let parsed = parse_svg_root(&result).unwrap();
        assert!(parsed.height > 150.0);
        assert_eq!(parsed.width, 300.0);
        assert!(result.contains("Title"));
        assert!(result.contains("Subtitle"));
        assert!(result.contains("Caption"));
    }

    #[test]
    fn title_height_expansion_is_deterministic() {
        // Same inputs produce exactly the same output (no randomness).
        let svg = make_svg(400.0, 200.0);
        let chrome = FigureChrome {
            title: Some("T"),
            subtitle: Some("S"),
            caption: Some("C"),
            ..Default::default()
        };
        let a = wrap_with_chrome(&svg, chrome).unwrap();
        let b = wrap_with_chrome(&svg, chrome).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn compute_header_height_title_only() {
        let chrome = FigureChrome { title: Some("x"), ..Default::default() };
        let h = compute_header_height(&chrome);
        let expected = TITLE_TOP_PAD + FIGURE_TITLE_FONT_SIZE + HEADER_BOTTOM_PAD;
        assert!((h - expected).abs() < 1e-9, "got {h}, expected {expected}");
    }

    #[test]
    fn compute_header_height_title_and_subtitle() {
        let chrome = FigureChrome { title: Some("x"), subtitle: Some("y"), ..Default::default() };
        let h = compute_header_height(&chrome);
        let expected = TITLE_TOP_PAD
            + FIGURE_TITLE_FONT_SIZE
            + TITLE_SUBTITLE_GAP
            + FIGURE_SUBTITLE_FONT_SIZE
            + HEADER_BOTTOM_PAD;
        assert!((h - expected).abs() < 1e-9, "got {h}, expected {expected}");
    }

    #[test]
    fn compute_footer_height_with_caption() {
        let chrome = FigureChrome { caption: Some("c"), ..Default::default() };
        let h = compute_footer_height(&chrome);
        let expected = CAPTION_TOP_PAD + FIGURE_CAPTION_FONT_SIZE + CAPTION_BOTTOM_PAD;
        assert!((h - expected).abs() < 1e-9, "got {h}, expected {expected}");
    }

    #[test]
    fn special_chars_in_text_are_escaped() {
        let svg = make_svg(100.0, 50.0);
        let chrome = FigureChrome {
            title: Some("Fish & Chips"),
            caption: Some("<Source>"),
            ..Default::default()
        };
        let result = wrap_with_chrome(&svg, chrome).unwrap();
        assert!(result.contains("Fish &amp; Chips"), "& must be escaped in title");
        assert!(result.contains("&lt;Source&gt;"), "< > must be escaped in caption");
    }

    #[test]
    fn inner_panels_wrapped_in_nested_svg_at_header_offset() {
        let svg = make_svg(200.0, 100.0);
        let chrome = FigureChrome { title: Some("T"), ..Default::default() };
        let result = wrap_with_chrome(&svg, chrome).unwrap();
        // The inner panels must be wrapped in <svg x="0" y="..." ...>
        let header_h = compute_header_height(&chrome);
        let needle = format!(r#"<svg x="0" y="{}"#, fmt_f(header_h));
        assert!(result.contains(&needle), "inner panels at y={header_h}: {result}");
    }

    #[test]
    fn default_inset_places_chrome_at_x16_start() {
        let svg = make_svg(200.0, 100.0);
        let chrome = FigureChrome {
            title: Some("T"),
            caption: Some("C"),
            ..Default::default()
        };
        // Default left_inset is DEFAULT_CHROME_INSET (16.0), anchor Start.
        assert_eq!(chrome.left_inset, 16.0);
        assert_eq!(chrome.right_inset, 16.0);
        assert_eq!(chrome.anchor, ChromeAnchor::Start);
        let result = wrap_with_chrome(&svg, chrome).unwrap();
        // Both the title and caption must sit at x="16" with text-anchor="start".
        let count_x16 = result.matches(r#"x="16""#).count();
        assert!(count_x16 >= 2, "title + caption at x=16: {result}");
        assert!(result.contains(r#"text-anchor="start""#), "start anchor: {result}");
        assert!(!result.contains(r#"text-anchor="middle""#));
        assert!(!result.contains(r#"text-anchor="end""#));
    }

    #[test]
    fn custom_left_inset_shifts_start_anchored_chrome() {
        let svg = make_svg(200.0, 100.0);
        let chrome = FigureChrome {
            title: Some("T"),
            caption: Some("C"),
            left_inset: 60.0,
            ..Default::default()
        };
        let result = wrap_with_chrome(&svg, chrome).unwrap();
        let count_x60 = result.matches(r#"x="60""#).count();
        assert!(count_x60 >= 2, "title + caption at x=60: {result}");
        assert!(!result.contains(r#"x="16""#), "no chrome left at default inset: {result}");
    }

    #[test]
    fn middle_anchor_centers_chrome_on_panel_width() {
        let panel_w = 300.0;
        let svg = make_svg(panel_w, 120.0);
        let chrome = FigureChrome {
            title: Some("T"),
            caption: Some("C"),
            anchor: ChromeAnchor::Middle,
            ..Default::default()
        };
        let result = wrap_with_chrome(&svg, chrome).unwrap();
        let expected_x = format!(r#"x="{}""#, fmt_f(panel_w / 2.0));
        let count = result.matches(expected_x.as_str()).count();
        assert!(count >= 2, "title + caption at {expected_x}: {result}");
        // text-anchor="middle" must appear for both lines.
        assert!(result.matches(r#"text-anchor="middle""#).count() >= 2, "{result}");
        assert!(!result.contains(r#"text-anchor="start""#));
    }

    #[test]
    fn end_anchor_right_aligns_chrome_using_right_inset() {
        let panel_w = 300.0;
        let right_inset = 40.0;
        let svg = make_svg(panel_w, 120.0);
        let chrome = FigureChrome {
            title: Some("T"),
            caption: Some("C"),
            right_inset,
            anchor: ChromeAnchor::End,
            ..Default::default()
        };
        let result = wrap_with_chrome(&svg, chrome).unwrap();
        let expected_x = format!(r#"x="{}""#, fmt_f(panel_w - right_inset));
        let count = result.matches(expected_x.as_str()).count();
        assert!(count >= 2, "title + caption at {expected_x}: {result}");
        assert!(result.matches(r#"text-anchor="end""#).count() >= 2, "{result}");
        assert!(!result.contains(r#"text-anchor="start""#));
    }

    #[test]
    fn empty_chrome_is_byte_identical_even_with_custom_geometry() {
        // Custom insets/anchor must not perturb the no-chrome early return.
        let svg = make_svg(200.0, 100.0);
        let chrome = FigureChrome {
            left_inset: 99.0,
            right_inset: 99.0,
            anchor: ChromeAnchor::End,
            ..Default::default()
        };
        assert!(chrome.is_empty());
        let result = wrap_with_chrome(&svg, chrome).unwrap();
        assert_eq!(result, svg, "empty chrome must be byte-identical round-trip");
    }
}
