//! Figure-level chrome bands for composite and flat single-chart renders.
//!
//! Wraps an already-composed SVG (the unified Rust composite renderer's output
//! for `HConcatChart`/`VConcatChart`/`RepeatChart`/facets, or a single chart's
//! own SVG via [`wrap_svg_with_chrome`]) with an optional title + subtitle band
//! above the panels and an optional caption band below the panels.
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
//! chrome bindings (which operate at the SVG string level, post-render).

use std::fmt;

use super::color::fmt_svg;
use super::svg::{
    escape_text, fmt_f, parse_svg_root, uniquify_clip_ids, write_svg_open, SvgParseError,
};
use ferrum_scene::{Color, FontWeight, SceneNode, TextAnchor, TextBaseline, TextStyle};

// ---------------------------------------------------------------------------
// Constants (consistent with ThemeTypography defaults + single-chart titles)
// ---------------------------------------------------------------------------

const FIGURE_TITLE_FONT_SIZE: f64 = 16.0;
const FIGURE_SUBTITLE_FONT_SIZE: f64 = 13.0;
const FIGURE_CAPTION_FONT_SIZE: f64 = 11.0;

/// Title text color (`#1f2937`, ferrum dark text). Matches the literal hex the
/// SVG header emits, so the interactive node band renders an identical color.
const FIGURE_TITLE_COLOR: Color = Color { r: 0x1f, g: 0x29, b: 0x37, a: 0xff };
/// Subtitle/caption text color (`#6b7280`, ferrum label gray). Matches the
/// literal hex the SVG bands emit.
const FIGURE_MUTED_COLOR: Color = Color { r: 0x6b, g: 0x72, b: 0x80, a: 0xff };
/// Font family for all three chrome lines (matches the SVG `font-family="Inter"`).
const FIGURE_FONT_FAMILY: &str = "Inter";

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
/// Governs all three chrome lines uniformly. Every caller (the PyO3
/// `wrap_svg_with_chrome` binding, `render/binding.rs`; the composite tree's
/// root `config` slot, `render/composite_render.rs`) parses the user-facing
/// anchor string once via [`FromStr`](std::str::FromStr) into this typed enum.
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

/// Error returned by [`ChromeAnchor::from_str`](std::str::FromStr::from_str)
/// for an unrecognized anchor string. Mirrors
/// [`crate::spec::composite::ParseCompositeLayoutError`]'s shape.
#[derive(Debug)]
pub struct ParseChromeAnchorError(pub String);

impl fmt::Display for ParseChromeAnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "anchor must be one of 'start'|'middle'|'end', got '{}'", self.0)
    }
}

impl std::error::Error for ParseChromeAnchorError {}

impl std::str::FromStr for ChromeAnchor {
    type Err = ParseChromeAnchorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "start" => Ok(ChromeAnchor::Start),
            "middle" => Ok(ChromeAnchor::Middle),
            "end" => Ok(ChromeAnchor::End),
            other => Err(ParseChromeAnchorError(other.to_string())),
        }
    }
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
    /// Title font size (px) override. `None` uses [`FIGURE_TITLE_FONT_SIZE`]
    /// (the figure-level chrome default). Set by a per-child composite label
    /// (Task 5d) that must match its owning call's theme rather than the
    /// figure-chrome constant — see `composite_render::apply_child_label`.
    /// Subtitle/caption sizes have no override slot: only the title-only band
    /// per-child labels use needs one.
    pub title_font_size: Option<f64>,
    /// Title color override. `None` uses [`FIGURE_TITLE_COLOR`]. Same
    /// per-child-label use case as `title_font_size`.
    pub title_color: Option<Color>,
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
            title_font_size: None,
            title_color: None,
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

    /// Compute the fully-resolved geometry for this chrome band against a
    /// composed panel of width `panel_w` and height `panel_h`.
    ///
    /// This is the single source of truth for chrome positioning: both the SVG
    /// emitter (`emit_header`/`emit_footer`) and the scene-node builder
    /// (`title_nodes`) consume the result, so the static and interactive renders
    /// place the title/subtitle/caption at byte-identical coordinates.
    fn layout(&self, panel_w: f64, panel_h: f64) -> ChromeLayout<'_> {
        let header_h = compute_header_height(self);
        let footer_h = compute_footer_height(self);
        let (x, text_anchor) = self.resolve_anchor(panel_w);
        let title_font_size = self.title_font_size.unwrap_or(FIGURE_TITLE_FONT_SIZE);

        let mut lines = Vec::new();

        // Header: title then subtitle. The y baselines below reproduce exactly
        // the inline math the SVG header used before this was extracted.
        if let Some(title) = self.title {
            lines.push(ChromeLine {
                role: ChromeRole::Title,
                content: title,
                x,
                y: TITLE_TOP_PAD + title_font_size,
                text_anchor,
                font_size: title_font_size,
                color_override: self.title_color,
            });
        }
        if let Some(subtitle) = self.subtitle {
            // With a title, the subtitle sits one line below the title baseline;
            // alone, it occupies the first header line.
            let y = if self.title.is_some() {
                TITLE_TOP_PAD
                    + title_font_size
                    + TITLE_SUBTITLE_GAP
                    + FIGURE_SUBTITLE_FONT_SIZE
            } else {
                TITLE_TOP_PAD + FIGURE_SUBTITLE_FONT_SIZE
            };
            lines.push(ChromeLine {
                role: ChromeRole::Subtitle,
                content: subtitle,
                x,
                y,
                text_anchor,
                font_size: FIGURE_SUBTITLE_FONT_SIZE,
                color_override: None,
            });
        }

        // Footer: caption below the panels.
        if let Some(caption) = self.caption {
            let panels_bottom_y = header_h + panel_h;
            lines.push(ChromeLine {
                role: ChromeRole::Caption,
                content: caption,
                x,
                y: panels_bottom_y + CAPTION_TOP_PAD + FIGURE_CAPTION_FONT_SIZE,
                text_anchor,
                font_size: FIGURE_CAPTION_FONT_SIZE,
                color_override: None,
            });
        }

        ChromeLayout { header_h, footer_h, lines }
    }
}

/// Which chrome slot a [`ChromeLine`] occupies. Drives the per-line styling
/// (font weight + color) so the node builder mirrors the SVG emitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChromeRole {
    Title,
    Subtitle,
    Caption,
}

/// Font weight for a chrome line. Bold renders `font-weight="600"` in the SVG
/// path and [`FontWeight::Custom("600")`] in the scene path; Normal omits the
/// SVG attribute entirely (matching the previous emitter) and maps to
/// [`FontWeight::Normal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChromeWeight {
    /// Title weight — `font-weight="600"`.
    Bold600,
    /// Subtitle/caption weight — no `font-weight` attribute.
    Normal,
}

/// The single source of truth for per-role chrome styling.
///
/// Both emitters read this struct so the static SVG band and the interactive
/// scene-node band stay byte-identical: the SVG path formats `color` via
/// [`crate::render::color::fmt_svg`] (yielding the same lowercase 6-digit hex
/// the band previously hardcoded) and consults `weight` for the optional
/// `font-weight="600"` attribute; the scene path consumes `color`/`weight`
/// directly. `text_anchor` is geometry, not role-keyed, so it stays on
/// [`ChromeLine`] — it is resolved per [`FigureChrome`] (the same value for all
/// three lines), not per role.
#[derive(Debug, Clone, Copy)]
struct ChromeRoleStyle {
    color: Color,
    weight: ChromeWeight,
}

impl ChromeRole {
    /// Resolve the color + weight for this chrome role. Editing the styling for
    /// a role here updates both the SVG and scene emitters at once.
    fn style(self) -> ChromeRoleStyle {
        match self {
            ChromeRole::Title => ChromeRoleStyle {
                color: FIGURE_TITLE_COLOR,
                weight: ChromeWeight::Bold600,
            },
            ChromeRole::Subtitle | ChromeRole::Caption => ChromeRoleStyle {
                color: FIGURE_MUTED_COLOR,
                weight: ChromeWeight::Normal,
            },
        }
    }
}

/// One fully-positioned chrome text line in the outer-canvas coordinate space.
#[derive(Debug, Clone, Copy)]
struct ChromeLine<'a> {
    role: ChromeRole,
    content: &'a str,
    x: f64,
    y: f64,
    text_anchor: &'static str,
    font_size: f64,
    /// Per-instance color override, consulted in place of [`ChromeRole::style`]'s
    /// default when set. Only the title line ever carries one (per-child
    /// composite labels, Task 5d); subtitle/caption always pass `None`.
    color_override: Option<Color>,
}

/// Fully-resolved chrome geometry: the reserved band heights plus every
/// positioned text line. Produced by [`FigureChrome::layout`] and consumed by
/// both the SVG path and the scene-node path.
#[derive(Debug, Clone)]
struct ChromeLayout<'a> {
    /// Vertical space reserved above the panels (title + subtitle band).
    header_h: f64,
    /// Vertical space reserved below the panels (caption band).
    footer_h: f64,
    /// Positioned chrome lines, in header-then-footer emit order.
    lines: Vec<ChromeLine<'a>>,
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
pub fn wrap_with_chrome(svg: &str, chrome: FigureChrome<'_>) -> Result<String, SvgParseError> {
    if chrome.is_empty() {
        return Ok(svg.to_string());
    }

    let parsed = parse_svg_root(svg)?;
    let panel_w = parsed.width;
    let panel_h = parsed.height;

    let layout = chrome.layout(panel_w, panel_h);
    let header_h = layout.header_h;

    // --- Total canvas ---
    let total_w = panel_w;
    let total_h = panel_h + header_h + layout.footer_h;

    let mut out = String::with_capacity(svg.len() + 512);
    write_svg_open(&mut out, total_w, total_h);

    // Emit the header band text nodes (title + subtitle), in layout order.
    for line in layout.lines.iter().filter(|l| l.role != ChromeRole::Caption) {
        emit_chrome_text(&mut out, line);
    }

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

    // Emit the footer band text node (caption).
    for line in layout.lines.iter().filter(|l| l.role == ChromeRole::Caption) {
        emit_chrome_text(&mut out, line);
    }

    out.push_str("</svg>");
    Ok(out)
}

/// Compose a single already-rendered SVG through the same single-cell wrap
/// step the deleted N-ary SVG compositor (`render/compositor.rs`, removed in
/// Task 10 stage 3) applied to a one-element input: re-emit the `<svg>` root
/// at the same dimensions and wrap the body in a
/// `<g transform="translate(0,0)">`, running it through [`uniquify_clip_ids`]
/// with `cell_idx = 0`.
///
/// The former compositor's per-cell emitter applied `uniquify_clip_ids`
/// unconditionally — even for the first/only cell — so this reproduces that
/// side effect exactly rather than skipping it as an optimization. This was
/// the flat single-chart counterpart of the deleted compositor's
/// vertical-stack entry point called with a one-element list and
/// `spacing=0.0`, extracted here so [`wrap_svg_with_chrome`] doesn't depend
/// on the (now-deleted) general N-ary SVG compositor.
fn compose_single_cell(svg: &str) -> Result<String, SvgParseError> {
    let parsed = parse_svg_root(svg)?;
    let mut out = String::with_capacity(svg.len() + 64);
    write_svg_open(&mut out, parsed.width, parsed.height);
    out.push_str(&format!(
        r#"<g transform="translate({},{})">"#,
        fmt_f(0.0), fmt_f(0.0),
    ));
    out.push_str(&uniquify_clip_ids(parsed.body, 0));
    out.push_str("</g></svg>");
    Ok(out)
}

/// Wrap a single (already-rendered) SVG with a figure-level chrome band.
///
/// This is the flat single-chart entry point that used to be reached via the
/// deleted compositor's vertical-stack entry point (one-element list,
/// `spacing=0.0`, `caption=..., **chrome_kwargs`) in `src/ferrum/_render.py`'s
/// `.properties(caption=)` post-wrap (Task 10 stage 3). It takes the same
/// code path — [`compose_single_cell`] followed by [`wrap_with_chrome`] —
/// without depending on the general N-ary SVG compositor, and is
/// byte-identical to that call for the same chrome parameters.
pub fn wrap_svg_with_chrome(svg: &str, chrome: FigureChrome<'_>) -> Result<String, SvgParseError> {
    let composed = compose_single_cell(svg)?;
    wrap_with_chrome(&composed, chrome)
}

/// Build figure-level chrome as positioned `SceneNode::Text` nodes (title,
/// subtitle, caption) plus the vertical band heights reserved above and below
/// the composed panels.
///
/// Positions come from the same [`FigureChrome::layout`] used by the SVG path,
/// so a composite's interactive on-canvas title band matches its static SVG
/// band exactly.
///
/// Returns `(nodes, header_h, footer_h)`:
/// - `nodes` — positioned `SceneNode::Text` items in outer-canvas coordinate
///   space for a panel region of size `(panel_w, panel_h)`.
/// - `header_h` — vertical band height (px) reserved **above** the panels
///   (title + subtitle). Python offsets merged child scenes down by this amount,
///   mirroring how `wrap_with_chrome` shifts the inner SVG down by `header_h`.
///   `0.0` when no title/subtitle is present.
/// - `footer_h` — vertical band height (px) reserved **below** the panels
///   (caption). Python should grow the merged canvas height by `footer_h` so the
///   interactive canvas matches the SVG canvas size that `wrap_with_chrome`
///   produces (which also grows total height by `footer_h`). `0.0` when no
///   caption is present.
///
/// When the chrome is empty, the nodes vector is empty and both band heights
/// are `0.0`.
pub fn title_nodes(chrome: FigureChrome<'_>, panel_w: f64, panel_h: f64) -> (Vec<SceneNode>, f64, f64) {
    if chrome.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }
    let layout = chrome.layout(panel_w, panel_h);
    let nodes = layout
        .lines
        .iter()
        .map(|line| SceneNode::Text {
            x: line.x,
            y: line.y,
            content: line.content.to_string(),
            style: chrome_text_style(line),
        })
        .collect();
    (nodes, layout.header_h, layout.footer_h)
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
        h += chrome.title_font_size.unwrap_or(FIGURE_TITLE_FONT_SIZE);
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

/// Format a chrome `Color` to its SVG `fill` hex string using the renderer's
/// canonical [`crate::render::color::fmt_svg`]. Chrome colors are always opaque,
/// so this yields the lowercase 6-digit form (`#1f2937` / `#6b7280`) the band
/// previously hardcoded — routing through `fmt_svg` makes the SVG fill track the
/// single styling source instead of a literal.
fn chrome_fill_hex(color: Color) -> String {
    fmt_svg(super::color::from_rgba(color.r, color.g, color.b, color.a))
}

/// Emit one positioned chrome `<text>` element.
///
/// Color and weight come from [`ChromeRole::style`] — the same source the scene
/// path ([`chrome_text_style`]) consumes — so the static and interactive renders
/// can never desync. Bold (title) emits `font-weight="600"`; Normal omits the
/// attribute entirely, reproducing the previous emitter's byte sequence.
fn emit_chrome_text(out: &mut String, line: &ChromeLine<'_>) {
    let style = line.role.style();
    let color = line.color_override.unwrap_or(style.color);
    let fill = chrome_fill_hex(color);
    let weight_attr = match style.weight {
        ChromeWeight::Bold600 => " font-weight=\"600\"",
        ChromeWeight::Normal => "",
    };
    out.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" fill=\"{}\" font-family=\"Inter\" font-size=\"{}\"{} text-anchor=\"{}\">{}</text>",
        fmt_f(line.x),
        fmt_f(line.y),
        fill,
        fmt_f(line.font_size),
        weight_attr,
        line.text_anchor,
        escape_text(line.content),
    ));
}

/// Build the `TextStyle` for a chrome line's scene node from the same
/// [`ChromeRole::style`] the SVG emitter reads (color + weight), so the static
/// SVG band and the interactive scene band carry identical per-role styling.
/// Baseline is `Alphabetic` to match SVG's default text baseline (the `y` is a
/// baseline, not a top edge).
fn chrome_text_style(line: &ChromeLine<'_>) -> TextStyle {
    let style = line.role.style();
    let font_weight = match style.weight {
        ChromeWeight::Bold600 => FontWeight::Custom("600".to_string()),
        ChromeWeight::Normal => FontWeight::Normal,
    };
    TextStyle {
        font_size: line.font_size,
        font_weight,
        anchor: match line.text_anchor {
            "middle" => TextAnchor::Middle,
            "end" => TextAnchor::End,
            _ => TextAnchor::Start,
        },
        baseline: TextBaseline::Alphabetic,
        angle: 0.0,
        color: line.color_override.unwrap_or(style.color),
        opacity: 1.0,
        font_family: FIGURE_FONT_FAMILY.to_string(),
    }
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

    // ── title_nodes (interactive scene-node path) ───────────────────────

    /// Pull the (x, y, content) out of a `SceneNode::Text`, panicking if the
    /// node is any other variant.
    fn text_node(node: &SceneNode) -> (f64, f64, &str) {
        match node {
            SceneNode::Text { x, y, content, .. } => (*x, *y, content.as_str()),
            other => panic!("expected Text node, got {other:?}"),
        }
    }

    #[test]
    fn title_nodes_empty_when_all_none() {
        let chrome = FigureChrome::default();
        let (nodes, header_h, footer_h) = title_nodes(chrome, 200.0, 100.0);
        assert!(nodes.is_empty(), "no chrome -> no nodes");
        assert_eq!(header_h, 0.0, "no chrome -> zero header band height");
        assert_eq!(footer_h, 0.0, "no chrome -> zero footer band height");
    }

    #[test]
    fn title_nodes_title_only_band_height_positive() {
        let chrome = FigureChrome { title: Some("Figure"), ..Default::default() };
        let (nodes, header_h, footer_h) = title_nodes(chrome, 200.0, 100.0);
        assert_eq!(nodes.len(), 1);
        assert!(header_h > 0.0, "title present -> band height > 0: {header_h}");
        assert_eq!(footer_h, 0.0, "no caption -> footer_h == 0");
        let (_, _, content) = text_node(&nodes[0]);
        assert_eq!(content, "Figure");
    }

    #[test]
    fn title_nodes_caption_only_has_zero_header_height() {
        // A caption is a footer band; it reserves no top space, so Python must
        // not offset the panels for a caption-only figure.
        let chrome = FigureChrome { caption: Some("Source: x"), ..Default::default() };
        let (nodes, header_h, footer_h) = title_nodes(chrome, 200.0, 100.0);
        assert_eq!(nodes.len(), 1);
        assert_eq!(header_h, 0.0, "caption is a footer band, no header offset");
        assert!(footer_h > 0.0, "caption present -> footer_h > 0: {footer_h}");
        let expected_footer_h = CAPTION_TOP_PAD + FIGURE_CAPTION_FONT_SIZE + CAPTION_BOTTOM_PAD;
        assert!((footer_h - expected_footer_h).abs() < 1e-9,
            "footer_h {footer_h} != expected {expected_footer_h}");
        let (_, _, content) = text_node(&nodes[0]);
        assert_eq!(content, "Source: x");
    }

    #[test]
    fn title_nodes_start_anchor_uses_left_inset() {
        let chrome = FigureChrome { title: Some("T"), ..Default::default() };
        let (nodes, _, _) = title_nodes(chrome, 200.0, 100.0);
        let (x, _, _) = text_node(&nodes[0]);
        assert_eq!(x, DEFAULT_CHROME_INSET, "start anchor -> x == left_inset");
    }

    #[test]
    fn title_nodes_middle_anchor_centers_on_width() {
        let panel_w = 300.0;
        let chrome = FigureChrome {
            title: Some("T"),
            anchor: ChromeAnchor::Middle,
            ..Default::default()
        };
        let (nodes, _, _) = title_nodes(chrome, panel_w, 100.0);
        let (x, _, _) = text_node(&nodes[0]);
        assert_eq!(x, panel_w / 2.0, "middle anchor -> x == width/2");
        match &nodes[0] {
            SceneNode::Text { style, .. } => assert_eq!(style.anchor, TextAnchor::Middle),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn title_nodes_end_anchor_uses_right_inset() {
        let panel_w = 300.0;
        let right_inset = 40.0;
        let chrome = FigureChrome {
            title: Some("T"),
            right_inset,
            anchor: ChromeAnchor::End,
            ..Default::default()
        };
        let (nodes, _, _) = title_nodes(chrome, panel_w, 100.0);
        let (x, _, _) = text_node(&nodes[0]);
        assert_eq!(x, panel_w - right_inset, "end anchor -> x == width - right_inset");
        match &nodes[0] {
            SceneNode::Text { style, .. } => assert_eq!(style.anchor, TextAnchor::End),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn title_nodes_positions_match_svg_band() {
        // Parity check: the node x/y for each line must equal the x/y the SVG
        // band emits for the same chrome. We assert on the shared `layout`
        // output, which both paths consume.
        let panel_w = 240.0;
        let panel_h = 120.0;
        let chrome = FigureChrome {
            title: Some("T"),
            subtitle: Some("S"),
            caption: Some("C"),
            ..Default::default()
        };
        let layout = chrome.layout(panel_w, panel_h);
        let (nodes, header_h, footer_h) = title_nodes(chrome, panel_w, panel_h);
        assert_eq!(nodes.len(), 3, "title + subtitle + caption");
        assert_eq!(header_h, layout.header_h);
        assert_eq!(footer_h, layout.footer_h);
        for (node, line) in nodes.iter().zip(layout.lines.iter()) {
            let (x, y, content) = text_node(node);
            assert_eq!(x, line.x, "node x matches layout line x");
            assert_eq!(y, line.y, "node y matches layout line y");
            assert_eq!(content, line.content);
        }
    }

    #[test]
    fn title_nodes_footer_h_zero_without_caption() {
        // title + subtitle only: footer_h must be 0.0; header_h must be > 0.
        let chrome = FigureChrome {
            title: Some("T"),
            subtitle: Some("S"),
            ..Default::default()
        };
        let (nodes, header_h, footer_h) = title_nodes(chrome, 300.0, 200.0);
        assert_eq!(nodes.len(), 2, "title + subtitle = 2 nodes");
        assert!(header_h > 0.0, "header_h > 0 with title+subtitle");
        assert_eq!(footer_h, 0.0, "footer_h == 0 without caption");
    }

    #[test]
    fn title_nodes_footer_h_matches_compute_footer_height() {
        // footer_h from title_nodes must equal compute_footer_height directly.
        let chrome = FigureChrome {
            title: Some("T"),
            caption: Some("C"),
            ..Default::default()
        };
        let (_, _, footer_h) = title_nodes(chrome, 400.0, 300.0);
        let expected = compute_footer_height(&chrome);
        assert!((footer_h - expected).abs() < 1e-9,
            "footer_h {footer_h} != compute_footer_height {expected}");
    }

    /// Byte tripwire: the chrome color consts, the canonical `fmt_svg` formatter
    /// the SVG path now routes through (`chrome_fill_hex`), and the manual
    /// `#{r:02x}{g:02x}{b:02x}` formula must all agree on the golden literals.
    ///
    /// Both `FIGURE_TITLE_COLOR` and `FIGURE_MUTED_COLOR` are fully opaque
    /// (`a == 0xff`), so `crate::render::color::fmt_svg` formats them as
    /// `#{r:02x}{g:02x}{b:02x}` (lowercase, 6-digit, no alpha). The goldens
    /// encode `#1f2937` / `#6b7280`; this test fails if any of the three
    /// representations drifts.
    #[test]
    fn color_consts_match_svg_emit_literals() {
        // Format using the same formula as `crate::render::color::fmt_svg` for
        // opaque colors: "#{r:02x}{g:02x}{b:02x}" (lowercase hex, no alpha).
        let title_hex = format!(
            "#{:02x}{:02x}{:02x}",
            FIGURE_TITLE_COLOR.r, FIGURE_TITLE_COLOR.g, FIGURE_TITLE_COLOR.b
        );
        let muted_hex = format!(
            "#{:02x}{:02x}{:02x}",
            FIGURE_MUTED_COLOR.r, FIGURE_MUTED_COLOR.g, FIGURE_MUTED_COLOR.b
        );
        // These must equal the golden fill="..." literals.
        assert_eq!(
            title_hex, "#1f2937",
            "FIGURE_TITLE_COLOR const diverged from the golden fill=\"#1f2937\""
        );
        assert_eq!(
            muted_hex, "#6b7280",
            "FIGURE_MUTED_COLOR const diverged from the golden fill=\"#6b7280\""
        );
        // The canonical formatter the SVG path uses must produce the same bytes.
        assert_eq!(chrome_fill_hex(FIGURE_TITLE_COLOR), "#1f2937");
        assert_eq!(chrome_fill_hex(FIGURE_MUTED_COLOR), "#6b7280");
    }

    /// Pins the single styling source ([`ChromeRole::style`]) to the exact
    /// byte contract the goldens encode: title is `#1f2937` + weight 600,
    /// subtitle/caption are `#6b7280` + normal. Both the SVG emitter
    /// (`chrome_fill_hex` + `weight_attr`) and the scene emitter
    /// (`chrome_text_style`) read this struct, so this test guards both paths.
    #[test]
    fn chrome_role_style_pins_color_weight_and_svg_fill() {
        // Title: dark text, bold-600, fill="#1f2937".
        let title = ChromeRole::Title.style();
        assert_eq!(title.color, FIGURE_TITLE_COLOR);
        assert_eq!(title.weight, ChromeWeight::Bold600);
        assert_eq!(
            chrome_fill_hex(title.color),
            "#1f2937",
            "title SVG fill must be exactly #1f2937 (golden byte contract)"
        );

        // Subtitle + caption: muted gray, normal weight, fill="#6b7280".
        for role in [ChromeRole::Subtitle, ChromeRole::Caption] {
            let s = role.style();
            assert_eq!(s.color, FIGURE_MUTED_COLOR, "{role:?} color");
            assert_eq!(s.weight, ChromeWeight::Normal, "{role:?} weight");
            assert_eq!(
                chrome_fill_hex(s.color),
                "#6b7280",
                "{role:?} SVG fill must be exactly #6b7280 (golden byte contract)"
            );
        }

        // The SVG weight attribute fragment: bold emits the attr, normal omits it.
        // (Mirrors `weight_attr` in `emit_chrome_text`; pins the byte sequence.)
        let title_weight_attr = match ChromeRole::Title.style().weight {
            ChromeWeight::Bold600 => " font-weight=\"600\"",
            ChromeWeight::Normal => "",
        };
        assert_eq!(title_weight_attr, " font-weight=\"600\"");
        let caption_weight_attr = match ChromeRole::Caption.style().weight {
            ChromeWeight::Bold600 => " font-weight=\"600\"",
            ChromeWeight::Normal => "",
        };
        assert_eq!(caption_weight_attr, "");

        // The scene anchor mapping stays Alphabetic baseline + role colors:
        // assert chrome_text_style derives FontWeight from the same struct.
        let line = ChromeLine {
            role: ChromeRole::Title,
            content: "x",
            x: 0.0,
            y: 0.0,
            text_anchor: "start",
            font_size: FIGURE_TITLE_FONT_SIZE,
            color_override: None,
        };
        let ts = chrome_text_style(&line);
        assert_eq!(ts.font_weight, FontWeight::Custom("600".to_string()));
        assert_eq!(ts.color, FIGURE_TITLE_COLOR);
    }

    #[test]
    fn title_nodes_carry_role_styling() {
        let chrome = FigureChrome {
            title: Some("T"),
            subtitle: Some("S"),
            caption: Some("C"),
            ..Default::default()
        };
        let (nodes, _, _) = title_nodes(chrome, 200.0, 100.0);
        // title: bold-600 dark; subtitle/caption: normal muted.
        match &nodes[0] {
            SceneNode::Text { style, .. } => {
                assert_eq!(style.font_weight, FontWeight::Custom("600".to_string()));
                assert_eq!(style.color, FIGURE_TITLE_COLOR);
                assert_eq!(style.font_size, FIGURE_TITLE_FONT_SIZE);
            }
            other => panic!("expected Text, got {other:?}"),
        }
        for node in &nodes[1..] {
            match node {
                SceneNode::Text { style, .. } => {
                    assert_eq!(style.font_weight, FontWeight::Normal);
                    assert_eq!(style.color, FIGURE_MUTED_COLOR);
                }
                other => panic!("expected Text, got {other:?}"),
            }
        }
    }

    /// A title-only chrome with `title_font_size`/`title_color` overrides must
    /// emit the override values, not the [`FIGURE_TITLE_FONT_SIZE`]/
    /// [`FIGURE_TITLE_COLOR`] constants — the per-child composite label
    /// (Task 5d) uses this override to match its owning call's theme. Also
    /// pins that the reserved header band height grows with the overridden
    /// font size (not a fixed 16px assumption), so a themed label never
    /// clips against its child's content.
    #[test]
    fn title_nodes_honor_font_size_and_color_overrides() {
        let custom_color = Color { r: 0x11, g: 0x22, b: 0x33, a: 0xff };
        let default_chrome = FigureChrome { title: Some("T"), ..Default::default() };
        let overridden = FigureChrome {
            title: Some("T"),
            title_font_size: Some(30.0),
            title_color: Some(custom_color),
            ..Default::default()
        };

        let (default_nodes, default_header_h, _) = title_nodes(default_chrome, 200.0, 100.0);
        let (nodes, header_h, _) = title_nodes(overridden, 200.0, 100.0);

        match &nodes[0] {
            SceneNode::Text { style, .. } => {
                assert_eq!(style.font_size, 30.0);
                assert_eq!(style.color, custom_color);
                // Weight/family are not part of this override (matches the
                // brief's two-field mirror of scene_build's title styling).
                assert_eq!(style.font_weight, FontWeight::Custom("600".to_string()));
            }
            other => panic!("expected Text, got {other:?}"),
        }
        assert_ne!(
            style_font_size(&default_nodes[0]),
            30.0,
            "sanity: default constant must differ from the override under test"
        );
        assert!(
            header_h > default_header_h,
            "a larger themed title_font_size must reserve more header height: \
             default={default_header_h} overridden={header_h}"
        );
    }

    fn style_font_size(node: &SceneNode) -> f64 {
        match node {
            SceneNode::Text { style, .. } => style.font_size,
            other => panic!("expected Text, got {other:?}"),
        }
    }

    // ── wrap_svg_with_chrome (flat single-chart chrome wrap) ────────────────
    //
    // Task 10 stage 3: `wrap_svg_with_chrome` replaced the flat single-chart
    // caption/title path that used to go through the deleted general N-ary
    // SVG compositor's vertical-stack entry point (one-element list,
    // `spacing=0.0`, before `render/compositor.rs` was deleted). While both
    // entries coexisted, a parity test suite (Rust unit tests here + a smoke
    // check through the compiled PyO3 bindings) pinned byte-identity between
    // them across every chrome kwarg combination the flat caption/title path
    // uses; it was deleted alongside the N-ary compositor's PyO3 bindings in
    // this same stage since there is no second implementation left to
    // compare against. These tests instead pin `wrap_svg_with_chrome`'s own
    // behavior directly.

    #[test]
    fn wrap_svg_with_chrome_no_chrome_matches_compose_single_cell() {
        // No-chrome early return happens inside `wrap_with_chrome`, on the
        // *composed* (single-cell-wrapped) string, not the raw input — so
        // `wrap_svg_with_chrome` with no chrome must equal `compose_single_cell`
        // directly, not the original `svg` argument.
        let svg = make_svg(200.0, 100.0);
        let chrome = FigureChrome::default();
        let wrapped = wrap_svg_with_chrome(&svg, chrome).unwrap();
        let composed = compose_single_cell(&svg).unwrap();
        assert_eq!(wrapped, composed, "no-chrome case must equal the single-cell composition");
    }

    #[test]
    fn wrap_svg_with_chrome_uniquifies_clip_ids_with_cell0_prefix() {
        // compose_single_cell (like the deleted compositor's write_cell) applies
        // uniquify_clip_ids unconditionally, even for the sole cell.
        let body = r#"<defs><clipPath id="ferrum-clip-0"><rect/></clipPath></defs><g clip-path="url(#ferrum-clip-0)"/>"#;
        let svg = make_root_svg(100.0, 50.0, body);
        let composed = compose_single_cell(&svg).unwrap();
        assert!(composed.contains(r#"id="cell0-ferrum-clip-0""#), "composed: {composed}");
        assert!(composed.contains("url(#cell0-ferrum-clip-0)"), "composed: {composed}");
        assert!(!composed.contains(r#"id="ferrum-clip-0""#), "unprefixed id leaked: {composed}");
    }

    fn make_root_svg(w: f64, h: f64, body: &str) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">{}</svg>"#,
            w, h, w, h, body,
        )
    }

    #[test]
    fn wrap_svg_with_chrome_caption_expands_height_and_preserves_width() {
        let svg = make_svg(200.0, 100.0);
        let chrome = FigureChrome { caption: Some("Source: note"), ..Default::default() };
        let result = wrap_svg_with_chrome(&svg, chrome).unwrap();
        let parsed = parse_svg_root(&result).unwrap();
        assert_eq!(parsed.width, 200.0, "width unchanged");
        assert!(parsed.height > 100.0, "height should grow with caption: {}", parsed.height);
        assert!(result.contains("Source: note"));
    }

    #[test]
    fn wrap_svg_with_chrome_left_right_inset_and_anchor_match_wrap_with_chrome() {
        // wrap_svg_with_chrome must honor the same insets/anchor resolution
        // wrap_with_chrome does for a composite, since it delegates to it.
        for (left_inset, right_inset, anchor) in [
            (40.0, 30.0, ChromeAnchor::Start),
            (16.0, 16.0, ChromeAnchor::Middle),
            (16.0, 16.0, ChromeAnchor::End),
        ] {
            let svg = make_svg(200.0, 100.0);
            let chrome = FigureChrome {
                caption: Some("Source: note"),
                left_inset,
                right_inset,
                anchor,
                ..Default::default()
            };
            let result = wrap_svg_with_chrome(&svg, chrome).unwrap();
            let expected_anchor = match anchor {
                ChromeAnchor::Start => "start",
                ChromeAnchor::Middle => "middle",
                ChromeAnchor::End => "end",
            };
            assert!(
                result.contains(&format!(r#"text-anchor="{expected_anchor}""#)),
                "expected anchor {expected_anchor}: {result}"
            );
        }
    }

    #[test]
    fn wrap_svg_with_chrome_empty_dataset_svg_caption_and_no_caption() {
        // The exact 97-byte placeholder `_render.py` emits for an empty dataset.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="600.0" height="400.0"><!-- empty dataset --></svg>"#;
        assert_eq!(svg.len(), 97, "pins the literal this test asserts against");

        let with_caption =
            wrap_svg_with_chrome(svg, FigureChrome { caption: Some("EmptyCap"), ..Default::default() })
                .unwrap();
        assert!(with_caption.contains("EmptyCap"));
        let parsed = parse_svg_root(&with_caption).unwrap();
        assert!(parsed.height > 400.0, "height should grow with caption: {}", parsed.height);

        let no_chrome = wrap_svg_with_chrome(svg, FigureChrome::default()).unwrap();
        assert_eq!(no_chrome, compose_single_cell(svg).unwrap());
    }
}
