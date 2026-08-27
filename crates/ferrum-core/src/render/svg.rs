//! Hand-rolled SVG buffer with deterministic float formatting and fixed attribute order.
//! Spec §4.4. The element vocabulary covers only what Phase 7 marks need.

use crate::layout::Rect;
use crate::layout::TextAnchor;

use super::color::{fmt_svg, Color};
use super::FLOAT_PRECISION;

pub struct SvgBuffer {
    buf: String,
}

#[derive(Debug, Clone)]
pub struct FillStroke {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    /// Per-element stroke opacity in [0, 1]. Omit attribute when == 1.0.
    pub stroke_opacity: f64,
    /// Per-element fill opacity in [0, 1]. Emitted as `fill-opacity` SVG attribute.
    /// Omit attribute when == 1.0. Distinct from `opacity` which bakes into fill RGBA alpha.
    pub fill_opacity: f64,
    /// Per-element stroke dash pattern. None = solid.
    pub stroke_dash: Option<Vec<f64>>,
    /// Rotation in degrees around (`angle_cx`, `angle_cy`). Omit when 0.0.
    pub angle: f64,
    /// Anchor x for `rotate(angle cx cy)` — typically element center.
    pub angle_cx: f64,
    /// Anchor y for `rotate(angle cx cy)` — typically element center.
    pub angle_cy: f64,
}

impl Default for FillStroke {
    fn default() -> Self {
        Self {
            fill: None,
            stroke: None,
            stroke_width: 0.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
            angle_cx: 0.0,
            angle_cy: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Stroke {
    pub stroke: Color,
    pub stroke_width: f64,
    pub stroke_dash: Option<Vec<f64>>,
    /// Per-element stroke opacity in [0, 1]. Omit attribute when == 1.0.
    pub stroke_opacity: f64,
}

#[derive(Debug, Clone)]
pub struct TextStyle<'a> {
    pub fill: Color,
    pub font_size: f64,
    pub anchor: TextAnchor,
    pub angle: f64,
    pub font_family: &'a str,
    pub font_weight: Option<&'a str>,
    pub dominant_baseline: Option<&'a str>,
}

impl SvgBuffer {
    pub fn new(viewport: Rect, background: Option<Color>, embed_font: bool) -> Self {
        let mut buf = String::with_capacity(8192);
        buf.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
            fmt_f(viewport.w), fmt_f(viewport.h), fmt_f(viewport.w), fmt_f(viewport.h),
        ));
        if embed_font {
            buf.push_str(&super::embed_font::inter_data_url_block());
        }
        if let Some(bg) = background {
            buf.push_str(&format!(
                "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"{}\"/>",
                fmt_f(viewport.w), fmt_f(viewport.h), fmt_svg(bg),
            ));
        }
        Self { buf }
    }

    pub fn finish(mut self) -> String {
        self.buf.push_str("</svg>");
        self.buf
    }

    pub fn g_open(&mut self, transform: Option<&str>) {
        match transform {
            Some(t) => self.buf.push_str(&format!("<g transform=\"{}\">", escape_attr(t))),
            None => self.buf.push_str("<g>"),
        }
    }

    pub fn g_close(&mut self) {
        self.buf.push_str("</g>");
    }

    /// Append raw SVG markup. Used for tightly-formatted output that doesn't
    /// fit a typed helper — e.g. a vertical `linearGradient` for colorbars.
    /// Callers are responsible for valid SVG (no escaping is performed).
    pub fn raw(&mut self, fragment: &str) {
        self.buf.push_str(fragment);
    }

    /// Emit an SVG `<title>` element (tooltip text on hover in browsers).
    /// The text is XML-escaped.
    pub fn title_elem(&mut self, text: &str) {
        self.buf.push_str("<title>");
        self.buf.push_str(&escape_text(text));
        self.buf.push_str("</title>");
    }

    /// Emit an SVG `<desc>` element (accessibility description).
    /// The text is XML-escaped.
    pub fn desc_elem(&mut self, text: &str) {
        self.buf.push_str("<desc>");
        self.buf.push_str(&escape_text(text));
        self.buf.push_str("</desc>");
    }

    /// Open an SVG `<a>` hyperlink wrapper with the given URL.
    pub fn a_open(&mut self, href: &str) {
        self.buf.push_str("<a href=\"");
        self.buf.push_str(&escape_attr(href));
        self.buf.push_str("\">");
    }

    /// Close an SVG `<a>` hyperlink wrapper.
    pub fn a_close(&mut self) {
        self.buf.push_str("</a>");
    }

    pub fn rect(&mut self, r: Rect, style: &FillStroke, corner_radius: Option<f64>) {
        self.buf.push_str("<rect");
        push_attr(&mut self.buf, "x", &fmt_f(r.x));
        push_attr(&mut self.buf, "y", &fmt_f(r.y));
        push_attr(&mut self.buf, "width", &fmt_f(r.w));
        push_attr(&mut self.buf, "height", &fmt_f(r.h));
        if let Some(rad) = corner_radius {
            if rad > 0.0 {
                push_attr(&mut self.buf, "rx", &fmt_f(rad));
                push_attr(&mut self.buf, "ry", &fmt_f(rad));
            }
        }
        push_fill_stroke(&mut self.buf, style);
        self.buf.push_str("/>");
    }

    pub fn circle(&mut self, cx: f64, cy: f64, radius: f64, style: &FillStroke) {
        self.buf.push_str("<circle");
        push_attr(&mut self.buf, "cx", &fmt_f(cx));
        push_attr(&mut self.buf, "cy", &fmt_f(cy));
        push_attr(&mut self.buf, "r", &fmt_f(radius));
        push_fill_stroke(&mut self.buf, style);
        self.buf.push_str("/>");
    }

    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, style: &Stroke) {
        self.buf.push_str("<line");
        push_attr(&mut self.buf, "x1", &fmt_f(x1));
        push_attr(&mut self.buf, "y1", &fmt_f(y1));
        push_attr(&mut self.buf, "x2", &fmt_f(x2));
        push_attr(&mut self.buf, "y2", &fmt_f(y2));
        push_stroke(&mut self.buf, style);
        self.buf.push_str("/>");
    }

    /// Emit a `<line>` for gridline use — accepts dash + opacity inline so
    /// theme-driven grid styling can flow without extending the shared
    /// `Stroke` struct (which would touch every mark call site).
    /// Attribute order matches `line` (x1, y1, x2, y2, stroke, stroke-width,
    /// stroke-dasharray, stroke-opacity) for SVG-diff stability.
    pub fn gridline(
        &mut self,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        color: Color,
        width: f64,
        dash: Option<&[f64]>,
        opacity: f64,
    ) {
        self.buf.push_str("<line");
        push_attr(&mut self.buf, "x1", &fmt_f(x1));
        push_attr(&mut self.buf, "y1", &fmt_f(y1));
        push_attr(&mut self.buf, "x2", &fmt_f(x2));
        push_attr(&mut self.buf, "y2", &fmt_f(y2));
        push_attr(&mut self.buf, "stroke", &fmt_svg(color));
        push_attr(&mut self.buf, "stroke-width", &fmt_f(width));
        if let Some(d) = dash {
            let v: Vec<String> = d.iter().map(|x| fmt_f(*x)).collect();
            push_attr(&mut self.buf, "stroke-dasharray", &v.join(","));
        }
        if opacity < 1.0 {
            push_attr(&mut self.buf, "stroke-opacity", &fmt_f(opacity));
        }
        self.buf.push_str("/>");
    }

    pub fn path(&mut self, d: &str, style: &FillStroke) {
        self.buf.push_str("<path");
        push_attr(&mut self.buf, "d", &escape_attr(d));
        push_fill_stroke(&mut self.buf, style);
        self.buf.push_str("/>");
    }

    pub fn polyline(&mut self, points: &[(f64, f64)], style: &Stroke) {
        let pts: Vec<String> = points.iter().map(|(x, y)| format!("{},{}", fmt_f(*x), fmt_f(*y))).collect();
        self.buf.push_str("<polyline");
        push_attr(&mut self.buf, "points", &pts.join(" "));
        push_attr(&mut self.buf, "fill", "none");
        push_stroke(&mut self.buf, style);
        self.buf.push_str("/>");
    }

    pub fn text(&mut self, x: f64, y: f64, content: &str, style: &TextStyle) {
        self.buf.push_str("<text");
        push_attr(&mut self.buf, "x", &fmt_f(x));
        push_attr(&mut self.buf, "y", &fmt_f(y));
        push_attr(&mut self.buf, "fill", &fmt_svg(style.fill));
        push_attr(&mut self.buf, "font-family", style.font_family);
        push_attr(&mut self.buf, "font-size", &fmt_f(style.font_size));
        if let Some(fw) = style.font_weight {
            push_attr(&mut self.buf, "font-weight", fw);
        }
        if let Some(db) = style.dominant_baseline {
            push_attr(&mut self.buf, "dominant-baseline", db);
        }
        push_attr(&mut self.buf, "text-anchor", anchor_str(style.anchor));
        if style.angle != 0.0 {
            let t = format!("rotate({} {} {})", fmt_f(style.angle), fmt_f(x), fmt_f(y));
            push_attr(&mut self.buf, "transform", &t);
        }
        self.buf.push('>');
        if content.contains('\n') {
            let x_str = fmt_f(x);
            for (i, line) in content.split('\n').enumerate() {
                self.buf.push_str("<tspan");
                push_attr(&mut self.buf, "x", &x_str);
                if i == 0 {
                    push_attr(&mut self.buf, "dy", "0");
                } else {
                    push_attr(&mut self.buf, "dy", "1.2em");
                }
                self.buf.push('>');
                self.buf.push_str(&escape_text(line));
                self.buf.push_str("</tspan>");
            }
        } else {
            self.buf.push_str(&escape_text(content));
        }
        self.buf.push_str("</text>");
    }

    /// Embed an image as `<image href="data:<mime>;base64,..." .../>`.
    /// `mime` controls whether the data URL uses `image/png` or `image/jpeg`.
    /// Attribute order is pinned: x, y, width, height, href.
    pub fn image(&mut self, x: f64, y: f64, w: f64, h: f64, bytes: &[u8], mime: ferrum_scene::ImageMime) {
        use base64::Engine;
        let mime_str = match mime {
            ferrum_scene::ImageMime::Png => "image/png",
            ferrum_scene::ImageMime::Jpeg => "image/jpeg",
        };
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        self.buf.push_str(&format!(
            r#"<image x="{}" y="{}" width="{}" height="{}" href="data:{};base64,{}"/>"#,
            fmt_f(x), fmt_f(y), fmt_f(w), fmt_f(h), mime_str, b64
        ));
    }

    /// Embed an image using a pre-formed data URL string (e.g. from mark_image URL tiles).
    /// `data_url` must be a complete `data:<mime>;base64,<payload>` string — it is
    /// embedded verbatim as the `href` attribute without re-encoding.
    pub fn image_data_url(&mut self, x: f64, y: f64, w: f64, h: f64, data_url: &str) {
        self.buf.push_str(&format!(
            r#"<image x="{}" y="{}" width="{}" height="{}" href="{}"/>"#,
            fmt_f(x), fmt_f(y), fmt_f(w), fmt_f(h), data_url
        ));
    }

    /// Emit a closed filled/stroked polygon as `<path d="M ... Z" fill-rule="evenodd"/>`.
    /// `paths`: each inner `Vec<(f64, f64)>` is one ring; multiple rings → first is outer,
    /// rest are holes. `fill-rule="evenodd"` handles winding automatically.
    pub fn polygon(&mut self, paths: &[Vec<(f64, f64)>], style: &FillStroke) {
        if paths.is_empty() {
            return;
        }
        let mut d = String::new();
        let mut first_ring = true;
        for ring in paths {
            if ring.is_empty() {
                continue;
            }
            if !first_ring {
                d.push(' ');
            }
            first_ring = false;
            d.push_str(&format!("M {} {}", fmt_f(ring[0].0), fmt_f(ring[0].1)));
            for (x, y) in &ring[1..] {
                d.push_str(&format!(" L {} {}", fmt_f(*x), fmt_f(*y)));
            }
            d.push_str(" Z");
        }
        self.buf.push_str("<path");
        push_attr(&mut self.buf, "d", &d);
        push_attr(&mut self.buf, "fill-rule", "evenodd");
        push_fill_stroke(&mut self.buf, style);
        self.buf.push_str("/>");
    }

    /// Emit a batch of <circle> elements at pre-resolved positions, wrapped in a <g>.
    /// Equivalent to N circle() calls with deterministic ordering, but more compact DOM.
    /// Used by mark_swarm to keep beeswarm SVG manageable.
    pub fn beeswarm(&mut self, points: &[(f64, f64)], radius: f64, style: &FillStroke) {
        if points.is_empty() { return; }
        self.buf.push_str("<g");
        push_fill_stroke(&mut self.buf, style);
        self.buf.push('>');
        let r = fmt_f(radius);
        for (x, y) in points {
            self.buf.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="{}"/>"#,
                fmt_f(*x), fmt_f(*y), r
            ));
        }
        self.buf.push_str("</g>");
    }

    pub fn clip_open(&mut self, id: &str, rect: Rect) {
        self.buf.push_str(&format!(
            "<defs><clipPath id=\"{}\"><rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/></clipPath></defs>",
            escape_attr(id), fmt_f(rect.x), fmt_f(rect.y), fmt_f(rect.w), fmt_f(rect.h),
        ));
    }

    pub fn use_clip_open(&mut self, clip_id: &str) {
        self.buf.push_str(&format!("<g clip-path=\"url(#{})\">", escape_attr(clip_id)));
    }

    pub fn use_clip_close(&mut self) {
        self.buf.push_str("</g>");
    }

    #[cfg(test)]
    pub(crate) fn as_str(&self) -> &str {
        &self.buf
    }
}

fn anchor_str(a: TextAnchor) -> &'static str {
    match a { TextAnchor::Start => "start", TextAnchor::Middle => "middle", TextAnchor::End => "end" }
}

fn push_attr(buf: &mut String, name: &str, value: &str) {
    buf.push(' ');
    buf.push_str(name);
    buf.push_str("=\"");
    buf.push_str(value);
    buf.push('"');
}

fn push_fill_stroke(buf: &mut String, s: &FillStroke) {
    match s.fill {
        Some(c) => push_attr(buf, "fill", &fmt_svg(c)),
        None => push_attr(buf, "fill", "none"),
    }
    if s.fill_opacity < 1.0 {
        push_attr(buf, "fill-opacity", &fmt_f(s.fill_opacity));
    }
    if let Some(stroke) = s.stroke {
        push_attr(buf, "stroke", &fmt_svg(stroke));
        if s.stroke_width > 0.0 {
            push_attr(buf, "stroke-width", &fmt_f(s.stroke_width));
        }
    }
    if s.stroke_opacity < 1.0 {
        push_attr(buf, "stroke-opacity", &fmt_f(s.stroke_opacity));
    }
    if let Some(ref dash) = s.stroke_dash {
        let v: Vec<String> = dash.iter().map(|x| fmt_f(*x)).collect();
        push_attr(buf, "stroke-dasharray", &v.join(","));
    }
    if s.angle != 0.0 {
        let t = format!("rotate({} {} {})", fmt_f(s.angle), fmt_f(s.angle_cx), fmt_f(s.angle_cy));
        push_attr(buf, "transform", &t);
    }
}

fn push_stroke(buf: &mut String, s: &Stroke) {
    push_attr(buf, "stroke", &fmt_svg(s.stroke));
    push_attr(buf, "stroke-width", &fmt_f(s.stroke_width));
    if let Some(dash) = &s.stroke_dash {
        let v: Vec<String> = dash.iter().map(|x| fmt_f(*x)).collect();
        push_attr(buf, "stroke-dasharray", &v.join(","));
    }
    if s.stroke_opacity < 1.0 {
        push_attr(buf, "stroke-opacity", &fmt_f(s.stroke_opacity));
    }
}

/// Format a float for SVG attribute output.
pub fn fmt_f(x: f64) -> String {
    if !x.is_finite() {
        debug_assert!(false, "non-finite float in SVG output: {x}");
        return "0".to_string();
    }
    let x = if x == 0.0 { 0.0 } else { x };
    let s = format!("{x:.*}", FLOAT_PRECISION);
    let trimmed = s.trim_end_matches('0').trim_end_matches('.').to_string();
    // After trimming, check for bare "-" or "-0" — both represent negative zero,
    // which is semantically wrong in SVG coordinates.
    if trimmed.is_empty() || trimmed == "-" || trimmed == "-0" { "0".to_string() } else { trimmed }
}

pub fn escape_text(s: &str) -> String {
    // Strip null bytes first: \x00 is invalid in XML and would produce malformed output.
    s.replace('\0', "").replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

pub fn escape_attr(s: &str) -> String {
    // Strip null bytes first: \x00 is invalid in XML and would produce malformed output.
    s.replace('\0', "").replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

// ---------------------------------------------------------------------------
// SVG root parsing + clip-id uniquification
//
// Moved from the now-deleted `render/compositor.rs` (Task 10 stage 3): these
// helpers parse/re-emit an SVG document's root element and rewrite embedded
// clip ids for cross-cell uniqueness. Surviving consumers are
// `figure_chrome::wrap_with_chrome`/`wrap_svg_with_chrome` (the composite
// chrome band and the flat single-chart chrome wrap) and
// `composite_render`'s raw-node clip uniquification (the unified Rust
// composite renderer) — the deleted N-ary SVG compositor's PyO3 bindings
// were the third consumer.
// ---------------------------------------------------------------------------

/// Error parsing an SVG document's root element.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SvgParseError {
    NoSvgRoot,
    MalformedRoot,
    NoClosingTag,
    MissingAttr(String),
    MalformedAttr(String),
    AttrNotNumeric(String),
}

impl std::fmt::Display for SvgParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SvgParseError::NoSvgRoot => write!(f, "input does not contain <svg ...>"),
            SvgParseError::MalformedRoot => write!(f, "<svg> open tag is malformed"),
            SvgParseError::NoClosingTag => write!(f, "input does not contain </svg>"),
            SvgParseError::MissingAttr(n) => write!(f, "<svg> missing required attr '{n}'"),
            SvgParseError::MalformedAttr(n) => write!(f, "<svg> attr '{n}' is malformed"),
            SvgParseError::AttrNotNumeric(n) => write!(f, "<svg> attr '{n}' is not numeric"),
        }
    }
}

impl std::error::Error for SvgParseError {}

#[derive(Debug)]
pub(crate) struct ParsedSvg<'a> {
    pub(crate) width: f64,
    pub(crate) height: f64,
    pub(crate) body: &'a str,
}

/// Parse the width and height attributes from an SVG root element.
/// Returns a `ParsedSvg` where `body` is the content between the opening
/// `<svg ...>` tag and the closing `</svg>`.
pub(crate) fn parse_svg_root(svg: &str) -> Result<ParsedSvg<'_>, SvgParseError> {
    let svg_open_start = svg.find("<svg").ok_or(SvgParseError::NoSvgRoot)?;
    let svg_open_end = svg[svg_open_start..]
        .find('>')
        .ok_or(SvgParseError::MalformedRoot)?
        + svg_open_start
        + 1;
    let svg_close = svg.rfind("</svg>").ok_or(SvgParseError::NoClosingTag)?;

    let attrs = &svg[svg_open_start..svg_open_end];
    let width = extract_attr_f64(attrs, "width")?;
    let height = extract_attr_f64(attrs, "height")?;

    Ok(ParsedSvg {
        width,
        height,
        body: &svg[svg_open_end..svg_close],
    })
}

fn extract_attr_f64(attrs: &str, name: &str) -> Result<f64, SvgParseError> {
    let needle = format!(r#"{}=""#, name);
    let start = attrs
        .find(&needle)
        .ok_or_else(|| SvgParseError::MissingAttr(name.into()))?;
    let val_start = start + needle.len();
    let val_end = attrs[val_start..]
        .find('"')
        .ok_or_else(|| SvgParseError::MalformedAttr(name.into()))?
        + val_start;
    attrs[val_start..val_end]
        .parse::<f64>()
        .map_err(|_| SvgParseError::AttrNotNumeric(name.into()))
}

/// Emit an outer `<svg ...>` opening tag with width/height/viewBox set to the
/// same `(w, h)` dimensions. The doc-comment on [`parse_svg_root`] notes that
/// attribute order matters for the round-trip; this helper is the single
/// source of truth for that order.
pub(crate) fn write_svg_open(out: &mut String, w: f64, h: f64) {
    out.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        fmt_f(w), fmt_f(h), fmt_f(w), fmt_f(h),
    ));
}

/// Rewrite every clipPath `id="…"` and `url(#…)` reference in a single SVG
/// body so each merged/composited cell uses globally-unique IDs. Covers the
/// three id families ferrum emits: `ferrum-clip-`, `ferrum-colorbar-`, and
/// `ferrum-legend-clip-`.
///
/// Each ferrum-rendered chart numbers its clipPaths per-panel starting at 0
/// (`ferrum-clip-0`, `ferrum-clip-1`, ...). When multiple single-panel charts
/// are merged into one document, they can all define `id="ferrum-clip-0"`,
/// and an SVG renderer would silently pick one definition — clipping every
/// cell by the wrong rect. The same collision hits `ferrum-colorbar-0`
/// colorbar gradients and `ferrum-legend-clip-0` legend `clip_height` clips.
/// Prefixing each cell's IDs with `cellNN-` makes them disjoint while
/// preserving each body's internal `id` ↔ `url(#id)` references.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::color::{from_rgb, from_rgba};

    fn vp() -> Rect { Rect { x: 0.0, y: 0.0, w: 100.0, h: 80.0 } }

    #[test]
    fn fmt_f_drops_trailing_zeros() {
        assert_eq!(fmt_f(1.5), "1.5");
        assert_eq!(fmt_f(1.0), "1");
        assert_eq!(fmt_f(1.500), "1.5");
        assert_eq!(fmt_f(0.0), "0");
        assert_eq!(fmt_f(-0.0), "0");
    }

    /// R1 port (bug_hunt_draw.rs / bug_hunt_render_pipeline.rs), corrected:
    /// the mirrors in both bug-hunt files reproduced `fmt_f` *without* its
    /// `debug_assert!(false, ...)` canary and asserted a silent `"0"` fallback
    /// — a drifted copy of the real function. The real `fmt_f` is guarded
    /// upstream so it should never see non-finite input; in debug/test builds
    /// the canary panics loudly instead of silently emitting "NaN"/"inf" into
    /// SVG output (the `"0"` fallback only executes in release builds, where
    /// `debug_assert!` compiles out).
    #[test]
    #[should_panic(expected = "non-finite float in SVG output")]
    #[cfg_attr(not(debug_assertions), ignore = "debug_assert! canary compiles out in release builds")]
    fn fmt_f_non_finite_panics_via_debug_assert_canary() {
        fmt_f(f64::NAN);
    }

    /// R1 port: the extended "-0" guard must catch every tiny-negative value
    /// that rounds to "-0.000" at the fixed 3-decimal precision, not just a
    /// bare "-" or literal `-0.0`.
    #[test]
    fn fmt_f_tiny_negatives_collapse_to_zero_not_minus_zero() {
        assert_eq!(fmt_f(-0.0004), "0");
        assert_eq!(fmt_f(-0.00049), "0");
        assert_eq!(fmt_f(-1e-10), "0");
        // A value that survives 3-decimal rounding must NOT be collapsed.
        assert_eq!(fmt_f(-0.001), "-0.001");
    }

    /// R1 port: extreme magnitudes (subnormal, MAX) must still produce a
    /// finite, non-empty, non-NaN/inf string.
    #[test]
    fn fmt_f_extreme_magnitudes_stay_finite_strings() {
        assert_eq!(fmt_f(5e-324), "0", "subnormal rounds to 0 at precision 3");
        let s = fmt_f(f64::MAX);
        assert!(!s.is_empty() && !s.contains("NaN") && !s.contains("inf"));
    }

    /// R1 port: null bytes are invalid in XML and must be stripped (not just
    /// left in place) before entity substitution, in both the text and
    /// attribute escapers.
    #[test]
    fn escape_text_and_attr_strip_null_bytes() {
        assert_eq!(escape_text("hello\x00world"), "helloworld");
        assert_eq!(escape_text("\x00\x00\x00"), "");
        assert_eq!(escape_attr("data\x00value"), "datavalue");
        // Null bytes strip independently of entity substitution ordering.
        assert_eq!(escape_text("a\x00&\x00b<c\x00>d"), "a&amp;b&lt;c&gt;d");
        assert_eq!(escape_attr("x\x00=\"y\"\x00"), "x=&quot;y&quot;");
    }

    #[test]
    fn empty_buffer_minimal_svg() {
        let buf = SvgBuffer::new(vp(), None, false);
        let out = buf.finish();
        assert!(out.starts_with("<svg "));
        assert!(out.ends_with("</svg>"));
        assert!(out.contains("width=\"100\""));
    }

    #[test]
    fn embed_font_includes_font_face_block() {
        let buf = SvgBuffer::new(vp(), None, true);
        let out = buf.finish();
        assert!(out.contains("@font-face"));
        assert!(out.contains("font-family:\"Inter\""));
    }

    #[test]
    fn background_emitted_when_some() {
        let buf = SvgBuffer::new(vp(), Some(from_rgb(0xFF, 0, 0)), false);
        let out = buf.finish();
        assert!(out.contains("<rect x=\"0\" y=\"0\""));
        assert!(out.contains("fill=\"#ff0000\""));
    }

    #[test]
    fn circle_attribute_order_is_fixed() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        buf.circle(10.5, 20.5, 3.0, &FillStroke { fill: Some(from_rgb(0, 0, 0)), stroke: None, stroke_width: 0.0, ..FillStroke::default() });
        let out = buf.finish();
        let needle = "<circle cx=\"10.5\" cy=\"20.5\" r=\"3\" fill=\"#000000\"/>";
        assert!(out.contains(needle), "missing exact element; got: {out}");
    }

    #[test]
    fn text_escapes_lt_gt_amp() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let style = TextStyle { fill: from_rgb(0,0,0), font_size: 11.0, anchor: TextAnchor::Start, angle: 0.0, font_family: "Inter", font_weight: None, dominant_baseline: None };
        buf.text(0.0, 0.0, "Price > 0 & < 1", &style);
        let out = buf.finish();
        assert!(out.contains("Price &gt; 0 &amp; &lt; 1"), "got: {out}");
    }

    #[test]
    fn text_single_line_no_tspan() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let style = TextStyle { fill: from_rgb(0,0,0), font_size: 11.0, anchor: TextAnchor::Middle, angle: 0.0, font_family: "Inter", font_weight: None, dominant_baseline: None };
        buf.text(50.0, 30.0, "hello", &style);
        let out = buf.finish();
        assert!(!out.contains("<tspan"), "single-line text must not emit <tspan>; got: {out}");
        assert!(out.contains(">hello<"), "content must appear directly in <text>; got: {out}");
    }

    #[test]
    fn text_multiline_emits_tspan_per_line() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let style = TextStyle { fill: from_rgb(0,0,0), font_size: 12.0, anchor: TextAnchor::Middle, angle: 0.0, font_family: "Inter", font_weight: None, dominant_baseline: None };
        buf.text(50.0, 30.0, "line1\nline2\nline3", &style);
        let out = buf.finish();
        assert_eq!(out.matches("<tspan").count(), 3, "expected 3 tspans; got: {out}");
        assert!(out.contains(">line1<"), "line1 content missing; got: {out}");
        assert!(out.contains(">line2<"), "line2 content missing; got: {out}");
        assert!(out.contains(">line3<"), "line3 content missing; got: {out}");
    }

    #[test]
    fn text_multiline_dy_sequence_is_correct() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let style = TextStyle { fill: from_rgb(0,0,0), font_size: 12.0, anchor: TextAnchor::Middle, angle: 0.0, font_family: "Inter", font_weight: None, dominant_baseline: None };
        buf.text(50.0, 30.0, "a\nb", &style);
        let out = buf.finish();
        // First tspan: dy="0"; second tspan: dy="1.2em"
        let tspan_a_idx = out.find(">a<").expect("tspan 'a' missing");
        let tspan_b_idx = out.find(">b<").expect("tspan 'b' missing");
        let first_region = &out[..tspan_a_idx];
        let second_region = &out[tspan_a_idx..tspan_b_idx];
        assert!(first_region.contains("dy=\"0\""), "first tspan must have dy=\"0\"; region: {first_region}");
        assert!(second_region.contains("dy=\"1.2em\""), "second tspan must have dy=\"1.2em\"; region: {second_region}");
    }

    #[test]
    fn text_multiline_x_attribute_repeated_on_each_tspan() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let style = TextStyle { fill: from_rgb(0,0,0), font_size: 12.0, anchor: TextAnchor::Start, angle: 0.0, font_family: "Inter", font_weight: None, dominant_baseline: None };
        buf.text(42.5, 20.0, "foo\nbar", &style);
        let out = buf.finish();
        // Both tspan elements must carry x="42.5"
        let x_in_tspan = out.matches(r#"<tspan x="42.5""#).count();
        assert_eq!(x_in_tspan, 2, "each tspan must carry the parent x; got: {out}");
    }

    #[test]
    fn rect_emits_corner_radius_when_positive() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let r = Rect { x: 0.0, y: 0.0, w: 10.0, h: 5.0 };
        buf.rect(r, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0, ..FillStroke::default() }, Some(2.0));
        assert!(buf.as_str().contains("rx=\"2\""));
        assert!(buf.as_str().contains("ry=\"2\""));
    }

    #[test]
    fn rect_omits_corner_radius_when_zero() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let r = Rect { x: 0.0, y: 0.0, w: 10.0, h: 5.0 };
        buf.rect(r, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0, ..FillStroke::default() }, Some(0.0));
        assert!(!buf.as_str().contains("rx="));
    }

    #[test]
    fn translucent_color_uses_rgba() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        buf.circle(0.0, 0.0, 1.0, &FillStroke { fill: Some(from_rgba(255, 0, 0, 128)), stroke: None, stroke_width: 0.0, ..FillStroke::default() });
        assert!(buf.as_str().contains("rgba(255,0,0,0.502)"));
    }

    #[test]
    fn determinism_two_calls_byte_identical() {
        let mut a = SvgBuffer::new(vp(), Some(from_rgb(255,255,255)), false);
        let mut b = SvgBuffer::new(vp(), Some(from_rgb(255,255,255)), false);
        a.circle(1.0, 2.0, 3.0, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0, ..FillStroke::default() });
        b.circle(1.0, 2.0, 3.0, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0, ..FillStroke::default() });
        assert_eq!(a.finish(), b.finish());
    }

    #[test]
    fn clip_open_close_is_balanced() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        buf.clip_open("c1", Rect { x: 0.0, y: 0.0, w: 50.0, h: 30.0 });
        buf.use_clip_open("c1");
        buf.circle(0.0, 0.0, 1.0, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0, ..FillStroke::default() });
        buf.use_clip_close();
        let out = buf.finish();
        assert!(out.contains("<clipPath id=\"c1\">"));
        assert!(out.contains("<g clip-path=\"url(#c1)\">"));
        assert_eq!(out.matches("</g>").count(), 1);
    }

    #[test]
    fn image_emits_data_url_with_fixed_attribute_order() {
        use base64::Engine;
        use ferrum_scene::ImageMime;
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let mut svg = SvgBuffer::new(viewport, None, false);
        let png_bytes: &[u8] = b"\x89PNG\r\n\x1a\n";  // PNG magic bytes (truncated)
        svg.image(10.0, 20.0, 50.0, 30.0, png_bytes, ImageMime::Png);
        let out = svg.finish();

        let b64 = base64::engine::general_purpose::STANDARD.encode(png_bytes);
        let expected_href = format!("data:image/png;base64,{b64}");

        // Attribute order: x, y, width, height, href (alphabetical NOT used; this is our pinned order)
        let needle = format!(r#"<image x="10" y="20" width="50" height="30" href="{expected_href}"/>"#);
        assert!(out.contains(&needle), "image markup missing or wrong order:\n{out}");
    }

    #[test]
    fn image_byte_identical_across_runs() {
        use ferrum_scene::ImageMime;
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let png: &[u8] = b"\x89PNG\r\n\x1a\nfoo";
        let make = || {
            let mut svg = SvgBuffer::new(viewport, None, false);
            svg.image(0.0, 0.0, 10.0, 10.0, png, ImageMime::Png);
            svg.finish()
        };
        assert_eq!(make(), make(), "image emission must be deterministic");
    }

    #[test]
    fn image_does_not_emit_whitespace_in_href() {
        use ferrum_scene::ImageMime;
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let mut svg = SvgBuffer::new(viewport, None, false);
        svg.image(0.0, 0.0, 10.0, 10.0, b"\x89PNG\r\n\x1a\nx", ImageMime::Png);
        let out = svg.finish();
        let href_start = out.find("data:image/png;base64,").expect("href missing");
        let href_end = out[href_start..].find('"').unwrap() + href_start;
        let href_body = &out[href_start..href_end];
        assert!(!href_body.contains('\n') && !href_body.contains(' '),
                "href body must be one line: {href_body}");
    }

    #[test]
    fn polygon_one_ring_emits_closed_path() {
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let mut svg = SvgBuffer::new(viewport, None, false);
        let ring = vec![(0.0_f64, 0.0_f64), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let style = FillStroke { fill: Some(from_rgb(0, 0, 0)), stroke: None, stroke_width: 0.0, ..FillStroke::default() };
        svg.polygon(&[ring], &style);
        let out = svg.finish();
        assert!(out.contains(r#"d="M 0 0 L 10 0 L 10 10 L 0 10 Z""#),
                "missing single-ring path data: {out}");
        assert!(out.contains(r#"fill-rule="evenodd""#),
                "missing fill-rule=evenodd: {out}");
    }

    #[test]
    fn polygon_multi_ring_concatenates_subpaths() {
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let mut svg = SvgBuffer::new(viewport, None, false);
        let outer = vec![(0.0_f64, 0.0_f64), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)];
        let hole  = vec![(5.0_f64, 5.0_f64), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)];
        let style = FillStroke { fill: Some(from_rgb(0, 0, 0)), stroke: None, stroke_width: 0.0, ..FillStroke::default() };
        svg.polygon(&[outer, hole], &style);
        let out = svg.finish();
        assert!(out.contains("M 0 0 L 20 0 L 20 20 L 0 20 Z M 5 5 L 15 5 L 15 15 L 5 15 Z"),
                "multi-ring path data wrong: {out}");
    }

    #[test]
    fn polygon_byte_identical_across_runs() {
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let style = FillStroke { fill: Some(from_rgba(50, 50, 50, 128)), stroke: None, stroke_width: 0.0, ..FillStroke::default() };
        let make = || {
            let mut svg = SvgBuffer::new(viewport, None, false);
            svg.polygon(&[vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]], &style);
            svg.finish()
        };
        assert_eq!(make(), make(), "polygon emission must be deterministic");
    }

    #[test]
    fn beeswarm_emits_group_with_n_circles() {
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let mut svg = SvgBuffer::new(viewport, None, false);
        let pts = vec![(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)];
        let style = FillStroke {
            fill: Some(from_rgb(0, 0, 0)),
            stroke: None,
            stroke_width: 0.0,
            ..FillStroke::default()
        };
        svg.beeswarm(&pts, 3.0, &style);
        let out = svg.finish();
        let circle_count = out.matches("<circle").count();
        assert_eq!(circle_count, 3, "expected 3 circles in beeswarm: {out}");
        assert!(out.contains("<g "), "beeswarm should wrap in <g>: {out}");
    }

    #[test]
    fn beeswarm_circles_in_input_order() {
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let mut svg = SvgBuffer::new(viewport, None, false);
        let pts = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let style = FillStroke {
            fill: Some(from_rgb(0, 0, 0)),
            stroke: None,
            stroke_width: 0.0,
            ..FillStroke::default()
        };
        svg.beeswarm(&pts, 1.0, &style);
        let out = svg.finish();
        let pos1 = out.find(r#"cx="1""#).unwrap();
        let pos2 = out.find(r#"cx="2""#).unwrap();
        let pos3 = out.find(r#"cx="3""#).unwrap();
        assert!(pos1 < pos2 && pos2 < pos3, "circles must emit in input order");
    }

    #[test]
    fn beeswarm_byte_identical_across_runs() {
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let style = FillStroke {
            fill: Some(from_rgb(50, 50, 50)),
            stroke: None,
            stroke_width: 0.0,
            ..FillStroke::default()
        };
        let make = || {
            let mut svg = SvgBuffer::new(viewport, None, false);
            svg.beeswarm(&[(1.0, 2.0), (3.0, 4.0)], 2.0, &style);
            svg.finish()
        };
        assert_eq!(make(), make(), "beeswarm must be deterministic");
    }

    // --- W13: image MIME type must match the ImageMime variant ---

    /// W13: image() with PNG mime must emit data:image/png;base64,...
    #[test]
    fn w13_image_png_mime_emitted_correctly() {
        use ferrum_scene::ImageMime;
        let mut buf = SvgBuffer::new(vp(), None, false);
        buf.image(10.0, 20.0, 30.0, 40.0, b"\x89PNG", ImageMime::Png);
        let out = buf.finish();
        assert!(
            out.contains("data:image/png;base64,"),
            "PNG mime type must appear in href; got: {out}"
        );
        assert!(
            !out.contains("data:image/jpeg;base64,"),
            "JPEG mime must not appear for PNG image; got: {out}"
        );
    }

    /// W13: image() with JPEG mime must emit data:image/jpeg;base64,...
    #[test]
    fn w13_image_jpeg_mime_emitted_correctly() {
        use ferrum_scene::ImageMime;
        let mut buf = SvgBuffer::new(vp(), None, false);
        buf.image(0.0, 0.0, 50.0, 50.0, b"\xFF\xD8\xFF", ImageMime::Jpeg);
        let out = buf.finish();
        assert!(
            out.contains("data:image/jpeg;base64,"),
            "JPEG mime type must appear in href; got: {out}"
        );
        assert!(
            !out.contains("data:image/png;base64,"),
            "PNG mime must not appear for JPEG image; got: {out}"
        );
    }

    // ── SVG root parsing + clip-id uniquification (moved from the deleted
    //    render/compositor.rs, Task 10 stage 3) ─────────────────────────

    fn make_root_svg(w: f64, h: f64, body: &str) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">{}</svg>"#,
            w, h, w, h, body,
        )
    }

    #[test]
    fn parse_svg_root_extracts_dimensions() {
        let s = make_root_svg(100.0, 50.0, "<rect/>");
        let parsed = parse_svg_root(&s).unwrap();
        assert_eq!(parsed.width, 100.0);
        assert_eq!(parsed.height, 50.0);
        assert_eq!(parsed.body, "<rect/>");
    }

    #[test]
    fn parse_svg_root_rejects_missing_root() {
        assert_eq!(parse_svg_root("<rect/>").unwrap_err(), SvgParseError::NoSvgRoot);
    }

    #[test]
    fn parse_svg_root_rejects_missing_dimension() {
        let s = r#"<svg xmlns="http://www.w3.org/2000/svg" height="50"></svg>"#;
        assert_eq!(
            parse_svg_root(s).unwrap_err(),
            SvgParseError::MissingAttr("width".to_string()),
        );
    }

    #[test]
    fn write_svg_open_matches_parse_svg_root_attribute_order() {
        let mut out = String::new();
        write_svg_open(&mut out, 200.0, 100.0);
        out.push_str("</svg>");
        let parsed = parse_svg_root(&out).unwrap();
        assert_eq!(parsed.width, 200.0);
        assert_eq!(parsed.height, 100.0);
    }

    /// Two merged bodies each carry a `clip_height` legend clip, so both
    /// share `id="ferrum-legend-clip-0"` + `clip-path="url(#ferrum-legend-clip-0)"`.
    /// `uniquify_clip_ids` must namespace each per its `cell_idx` so the two
    /// defs + references stay distinct (the JointChart/HConcat/composite
    /// merge collision this helper exists to prevent).
    #[test]
    fn uniquify_clip_ids_namespaces_clip_colorbar_and_legend_ids() {
        let body = concat!(
            r#"<defs><clipPath id="ferrum-clip-0"><rect/></clipPath></defs>"#,
            r#"<g clip-path="url(#ferrum-clip-0)">"#,
            r#"<linearGradient id="ferrum-colorbar-0"/>"#,
            r#"<rect fill="url(#ferrum-colorbar-0)"/>"#,
            r#"<clipPath id="ferrum-legend-clip-0"><rect/></clipPath>"#,
            r#"<g clip-path="url(#ferrum-legend-clip-0)"/></g>"#,
        );

        let cell0 = uniquify_clip_ids(body, 0);
        let cell1 = uniquify_clip_ids(body, 1);

        for (out, idx) in [(&cell0, 0), (&cell1, 1)] {
            assert!(!out.contains(r#"id="ferrum-clip-"#), "unprefixed clip def leaked: {out}");
            assert!(!out.contains("url(#ferrum-clip-0)"), "unprefixed clip ref leaked: {out}");
            for prefix in ["ferrum-clip-0", "ferrum-colorbar-0", "ferrum-legend-clip-0"] {
                let namespaced = format!("cell{idx}-{prefix}");
                assert!(out.contains(&format!(r#"id="{namespaced}""#)), "missing def for {namespaced}: {out}");
                assert!(out.contains(&format!("url(#{namespaced})")), "missing ref for {namespaced}: {out}");
            }
        }
        // Different cell indices must not collide with each other.
        assert_ne!(cell0, cell1);
    }
}
