//! Hand-rolled SVG buffer with deterministic float formatting and fixed attribute order.
//! Spec §4.4. The element vocabulary covers only what Phase 7 marks need.

use crate::layout::Rect;
use crate::layout::TextAnchor;

use super::color::{fmt_svg, Color};
use super::FLOAT_PRECISION;

pub struct SvgBuffer {
    buf: String,
}

#[derive(Debug, Clone, Copy)]
pub struct FillStroke {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
}

#[derive(Debug, Clone)]
pub struct Stroke {
    pub stroke: Color,
    pub stroke_width: f64,
    pub stroke_dash: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct TextStyle<'a> {
    pub fill: Color,
    pub font_size: f64,
    pub anchor: TextAnchor,
    pub angle: f64,
    pub font_family: &'a str,
    pub font_weight: Option<&'a str>,
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
        push_attr(&mut self.buf, "text-anchor", anchor_str(style.anchor));
        if style.angle != 0.0 {
            let t = format!("rotate({} {} {})", fmt_f(style.angle), fmt_f(x), fmt_f(y));
            push_attr(&mut self.buf, "transform", &t);
        }
        self.buf.push('>');
        self.buf.push_str(&escape_text(content));
        self.buf.push_str("</text>");
    }

    /// Embed a PNG as <image href="data:image/png;base64,..." x=... y=... width=... height=.../>.
    /// `png_bytes` must be a valid PNG-encoded buffer (use render::rasterize::encode_png).
    /// Attribute order is pinned: x, y, width, height, href.
    pub fn image(&mut self, x: f64, y: f64, w: f64, h: f64, png_bytes: &[u8]) {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(png_bytes);
        self.buf.push_str(&format!(
            r#"<image x="{}" y="{}" width="{}" height="{}" href="data:image/png;base64,{}"/>"#,
            fmt_f(x), fmt_f(y), fmt_f(w), fmt_f(h), b64
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
    if let Some(stroke) = s.stroke {
        push_attr(buf, "stroke", &fmt_svg(stroke));
        if s.stroke_width > 0.0 {
            push_attr(buf, "stroke-width", &fmt_f(s.stroke_width));
        }
    }
}

fn push_stroke(buf: &mut String, s: &Stroke) {
    push_attr(buf, "stroke", &fmt_svg(s.stroke));
    push_attr(buf, "stroke-width", &fmt_f(s.stroke_width));
    if let Some(dash) = &s.stroke_dash {
        let v: Vec<String> = dash.iter().map(|x| fmt_f(*x)).collect();
        push_attr(buf, "stroke-dasharray", &v.join(","));
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
    if trimmed.is_empty() || trimmed == "-" { "0".to_string() } else { trimmed }
}

pub fn escape_text(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

pub fn escape_attr(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
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
        buf.circle(10.5, 20.5, 3.0, &FillStroke { fill: Some(from_rgb(0, 0, 0)), stroke: None, stroke_width: 0.0 });
        let out = buf.finish();
        let needle = "<circle cx=\"10.5\" cy=\"20.5\" r=\"3\" fill=\"#000000\"/>";
        assert!(out.contains(needle), "missing exact element; got: {out}");
    }

    #[test]
    fn text_escapes_lt_gt_amp() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let style = TextStyle { fill: from_rgb(0,0,0), font_size: 11.0, anchor: TextAnchor::Start, angle: 0.0, font_family: "Inter", font_weight: None };
        buf.text(0.0, 0.0, "Price > 0 & < 1", &style);
        let out = buf.finish();
        assert!(out.contains("Price &gt; 0 &amp; &lt; 1"), "got: {out}");
    }

    #[test]
    fn rect_emits_corner_radius_when_positive() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let r = Rect { x: 0.0, y: 0.0, w: 10.0, h: 5.0 };
        buf.rect(r, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0 }, Some(2.0));
        assert!(buf.as_str().contains("rx=\"2\""));
        assert!(buf.as_str().contains("ry=\"2\""));
    }

    #[test]
    fn rect_omits_corner_radius_when_zero() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let r = Rect { x: 0.0, y: 0.0, w: 10.0, h: 5.0 };
        buf.rect(r, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0 }, Some(0.0));
        assert!(!buf.as_str().contains("rx="));
    }

    #[test]
    fn translucent_color_uses_rgba() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        buf.circle(0.0, 0.0, 1.0, &FillStroke { fill: Some(from_rgba(255, 0, 0, 128)), stroke: None, stroke_width: 0.0 });
        assert!(buf.as_str().contains("rgba(255,0,0,0.502)"));
    }

    #[test]
    fn determinism_two_calls_byte_identical() {
        let mut a = SvgBuffer::new(vp(), Some(from_rgb(255,255,255)), false);
        let mut b = SvgBuffer::new(vp(), Some(from_rgb(255,255,255)), false);
        a.circle(1.0, 2.0, 3.0, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0 });
        b.circle(1.0, 2.0, 3.0, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0 });
        assert_eq!(a.finish(), b.finish());
    }

    #[test]
    fn clip_open_close_is_balanced() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        buf.clip_open("c1", Rect { x: 0.0, y: 0.0, w: 50.0, h: 30.0 });
        buf.use_clip_open("c1");
        buf.circle(0.0, 0.0, 1.0, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0 });
        buf.use_clip_close();
        let out = buf.finish();
        assert!(out.contains("<clipPath id=\"c1\">"));
        assert!(out.contains("<g clip-path=\"url(#c1)\">"));
        assert_eq!(out.matches("</g>").count(), 1);
    }

    #[test]
    fn image_emits_data_url_with_fixed_attribute_order() {
        use base64::Engine;
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let mut svg = SvgBuffer::new(viewport, None, false);
        let png_bytes: &[u8] = b"\x89PNG\r\n\x1a\n";  // PNG magic bytes (truncated)
        svg.image(10.0, 20.0, 50.0, 30.0, png_bytes);
        let out = svg.finish();

        let b64 = base64::engine::general_purpose::STANDARD.encode(png_bytes);
        let expected_href = format!("data:image/png;base64,{b64}");

        // Attribute order: x, y, width, height, href (alphabetical NOT used; this is our pinned order)
        let needle = format!(r#"<image x="10" y="20" width="50" height="30" href="{expected_href}"/>"#);
        assert!(out.contains(&needle), "image markup missing or wrong order:\n{out}");
    }

    #[test]
    fn image_byte_identical_across_runs() {
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let png: &[u8] = b"\x89PNG\r\n\x1a\nfoo";
        let make = || {
            let mut svg = SvgBuffer::new(viewport, None, false);
            svg.image(0.0, 0.0, 10.0, 10.0, png);
            svg.finish()
        };
        assert_eq!(make(), make(), "image emission must be deterministic");
    }

    #[test]
    fn image_does_not_emit_whitespace_in_href() {
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let mut svg = SvgBuffer::new(viewport, None, false);
        svg.image(0.0, 0.0, 10.0, 10.0, b"\x89PNG\r\n\x1a\nx");
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
        let style = FillStroke { fill: Some(from_rgb(0, 0, 0)), stroke: None, stroke_width: 0.0 };
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
        let style = FillStroke { fill: Some(from_rgb(0, 0, 0)), stroke: None, stroke_width: 0.0 };
        svg.polygon(&[outer, hole], &style);
        let out = svg.finish();
        assert!(out.contains("M 0 0 L 20 0 L 20 20 L 0 20 Z M 5 5 L 15 5 L 15 15 L 5 15 Z"),
                "multi-ring path data wrong: {out}");
    }

    #[test]
    fn polygon_byte_identical_across_runs() {
        let viewport = Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 };
        let style = FillStroke { fill: Some(from_rgba(50, 50, 50, 128)), stroke: None, stroke_width: 0.0 };
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
        };
        let make = || {
            let mut svg = SvgBuffer::new(viewport, None, false);
            svg.beeswarm(&[(1.0, 2.0), (3.0, 4.0)], 2.0, &style);
            svg.finish()
        };
        assert_eq!(make(), make(), "beeswarm must be deterministic");
    }
}
