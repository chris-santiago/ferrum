//! Annotation rendering: converts annotation specs from the Python chart_config
//! into SceneNode elements positioned within a panel's plot area.
//!
//! Annotations are resolved against the panel's data-space scales (for data
//! coordinates) or directly against the pixel/normalized plot-area extent.

use ferrum_scene::{
    Color, FillStroke, FontWeight, PathCmd, SceneNode, StrokeStyle, TextAnchor, TextBaseline,
    TextStyle,
};
use serde::Deserialize;

use super::color::parse_color as parse_color_str;
use super::draw::to_scene_color;
use super::scale_resolve::ScaleKind;
use crate::layout::geometry::Rect;

// ── Coordinate types ────────────────────────────────────────────────────────

/// A coordinate value that can be expressed in data-space, pixel offset from
/// plot-area origin, or normalized [0,1] fraction of the plot-area extent.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum CoordValue {
    /// `{"px": 100.0}` — pixel offset from the plot-area origin.
    Pixel { px: f64 },
    /// `{"norm": 0.5}` — normalized fraction [0,1] of the plot-area extent.
    Norm { norm: f64 },
    /// Plain number — data-space value mapped through the panel's scale.
    Data(f64),
}

// ── Annotation spec ─────────────────────────────────────────────────────────

/// One annotation spec, deserialized from the chart_config `annotations` array.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum AnnotationSpec {
    #[serde(rename = "text")]
    Text {
        x: CoordValue,
        y: CoordValue,
        text: String,
        #[serde(default = "default_font_size")]
        font_size: f64,
        #[serde(default = "default_color_str")]
        color: String,
        #[serde(default = "default_anchor")]
        anchor: String,
        #[serde(default = "default_baseline")]
        baseline: String,
        #[serde(default)]
        angle: f64,
        #[serde(default)]
        dx: f64,
        #[serde(default)]
        dy: f64,
        /// Z-order: "front" or "back". Reserved for future layering support.
        #[serde(default = "default_z")]
        #[allow(dead_code)]
        z: String,
    },
    #[serde(rename = "arrow")]
    Arrow {
        x: CoordValue,
        y: CoordValue,
        x2: CoordValue,
        y2: CoordValue,
        #[serde(default = "default_color_str")]
        stroke: String,
        #[serde(default = "default_stroke_width")]
        stroke_width: f64,
        #[serde(default = "default_head_size")]
        head_size: f64,
        /// Curve type: "straight" or "curved". Reserved for future bezier support.
        #[serde(default = "default_curve")]
        #[allow(dead_code)]
        curve: String,
    },
    #[serde(rename = "rect")]
    Rect {
        x1: CoordValue,
        y1: CoordValue,
        x2: CoordValue,
        y2: CoordValue,
        #[serde(default = "default_fill_str")]
        fill: String,
        #[serde(default = "default_rect_opacity")]
        opacity: f64,
        #[serde(default)]
        corner_radius: f64,
        #[serde(default)]
        stroke: Option<String>,
    },
    #[serde(rename = "line")]
    Line {
        x1: CoordValue,
        y1: CoordValue,
        x2: CoordValue,
        y2: CoordValue,
        #[serde(default = "default_color_str")]
        stroke: String,
        #[serde(default = "default_stroke_width")]
        stroke_width: f64,
        #[serde(default)]
        dash: Option<Vec<f64>>,
    },
    #[serde(rename = "span")]
    Span {
        axis: String,
        start: CoordValue,
        end: CoordValue,
        #[serde(default = "default_fill_str")]
        fill: String,
        #[serde(default = "default_span_opacity")]
        opacity: f64,
        #[serde(default = "default_label_position")]
        #[allow(dead_code)]
        label_position: String,
        #[serde(default)]
        label: Option<String>,
    },
    #[serde(rename = "bracket")]
    Bracket {
        x1: CoordValue,
        y1: CoordValue,
        x2: CoordValue,
        y2: CoordValue,
        #[serde(default)]
        label: String,
        #[serde(default = "default_direction")]
        direction: String,
        #[serde(default = "default_color_str")]
        stroke: String,
        #[serde(default = "default_tip_length")]
        tip_length: f64,
    },
    #[serde(rename = "callout")]
    Callout {
        x: CoordValue,
        y: CoordValue,
        text: String,
        #[serde(default = "default_arrow_str")]
        arrow: String,
        #[serde(default = "default_callout_padding")]
        padding: f64,
        #[serde(default = "default_background_str")]
        background: String,
        #[serde(default = "default_border_color_str")]
        border_color: String,
        #[serde(default)]
        border_radius: f64,
        #[serde(default)]
        text_x: Option<CoordValue>,
        #[serde(default)]
        text_y: Option<CoordValue>,
    },
    #[serde(rename = "image")]
    Image {
        x: CoordValue,
        y: CoordValue,
        src: String,
        #[serde(default = "default_image_size")]
        width: f64,
        #[serde(default = "default_image_size")]
        height: f64,
        #[serde(default = "default_anchor")]
        anchor: String,
    },
}

// ── Serde defaults ──────────────────────────────────────────────────────────

fn default_font_size() -> f64 { 12.0 }
fn default_color_str() -> String { "#333333".to_string() }
fn default_anchor() -> String { "middle".to_string() }
fn default_baseline() -> String { "middle".to_string() }
fn default_z() -> String { "front".to_string() }
fn default_stroke_width() -> f64 { 1.5 }
fn default_head_size() -> f64 { 8.0 }
fn default_curve() -> String { "straight".to_string() }
fn default_fill_str() -> String { "#cccccc".to_string() }
fn default_rect_opacity() -> f64 { 0.3 }
fn default_span_opacity() -> f64 { 0.2 }
fn default_label_position() -> String { "center".to_string() }
fn default_direction() -> String { "up".to_string() }
fn default_tip_length() -> f64 { 6.0 }
fn default_arrow_str() -> String { "curved".to_string() }
fn default_callout_padding() -> f64 { 4.0 }
fn default_background_str() -> String { "#ffffff".to_string() }
fn default_border_color_str() -> String { "#333333".to_string() }
fn default_image_size() -> f64 { 50.0 }

// ── Scale context ───────────────────────────────────────────────────────────

/// Provides coordinate resolution from data/pixel/norm to absolute pixel position.
pub struct ScaleContext<'a> {
    pub plot_area: Rect,
    pub x_scale: &'a ScaleKind,
    pub y_scale: &'a ScaleKind,
}

impl<'a> ScaleContext<'a> {
    /// Resolve an x-axis coordinate value to absolute pixel position.
    pub fn resolve_x(&self, v: &CoordValue) -> f64 {
        match v {
            CoordValue::Data(d) => {
                self.x_scale.to_pixel_f64(*d).unwrap_or_else(|| {
                    // Fallback: linear interpolation using data domain.
                    if let Some((lo, hi)) = self.x_scale.data_domain() {
                        let frac = if (hi - lo).abs() < f64::EPSILON {
                            0.5
                        } else {
                            (d - lo) / (hi - lo)
                        };
                        self.plot_area.x + frac * self.plot_area.w
                    } else {
                        // Ordinal: cannot resolve numeric data to ordinal scale; center.
                        self.plot_area.x + self.plot_area.w * 0.5
                    }
                })
            }
            CoordValue::Pixel { px } => self.plot_area.x + px,
            CoordValue::Norm { norm } => self.plot_area.x + norm * self.plot_area.w,
        }
    }

    /// Resolve a y-axis coordinate value to absolute pixel position.
    ///
    /// The y-scale already maps domain [min, max] to pixel range [top, bottom]
    /// (SVG convention: Y increases downward), so `to_pixel_f64` returns the
    /// correct absolute pixel directly.
    pub fn resolve_y(&self, v: &CoordValue) -> f64 {
        match v {
            CoordValue::Data(d) => {
                self.y_scale.to_pixel_f64(*d).unwrap_or_else(|| {
                    // Fallback: linear interpolation using data domain.
                    // Scale range is [top, bottom], so frac maps top-to-bottom.
                    if let Some((lo, hi)) = self.y_scale.data_domain() {
                        let frac = if (hi - lo).abs() < f64::EPSILON {
                            0.5
                        } else {
                            (d - lo) / (hi - lo)
                        };
                        self.plot_area.y + frac * self.plot_area.h
                    } else {
                        self.plot_area.y + self.plot_area.h * 0.5
                    }
                })
            }
            CoordValue::Pixel { px } => self.plot_area.y + px,
            CoordValue::Norm { norm } => self.plot_area.y + norm * self.plot_area.h,
        }
    }
}

// ── Color helpers ───────────────────────────────────────────────────────────

/// Parse a color string to a scene Color. Falls back to dark gray (#333) on failure.
fn resolve_color(s: &str) -> Color {
    match parse_color_str(s) {
        Ok(c) => to_scene_color(c),
        Err(_) => Color::rgb(51, 51, 51),
    }
}

// ── Anchor / baseline parsing ───────────────────────────────────────────────

fn parse_anchor(s: &str) -> TextAnchor {
    match s {
        "start" | "left" => TextAnchor::Start,
        "end" | "right" => TextAnchor::End,
        _ => TextAnchor::Middle,
    }
}

fn parse_baseline(s: &str) -> TextBaseline {
    match s {
        "top" | "hanging" | "text-before-edge" => TextBaseline::Top,
        "bottom" | "ideographic" => TextBaseline::Bottom,
        "alphabetic" => TextBaseline::Alphabetic,
        _ => TextBaseline::Middle,
    }
}

// ── Build annotations ───────────────────────────────────────────────────────

/// Convert a slice of annotation specs into positioned SceneNodes.
pub fn build_annotations(specs: &[AnnotationSpec], ctx: &ScaleContext) -> Vec<SceneNode> {
    let mut nodes = Vec::with_capacity(specs.len());
    for spec in specs {
        build_one(spec, ctx, &mut nodes);
    }
    nodes
}

/// Dispatch a single annotation spec to its builder, appending nodes to `out`.
fn build_one(spec: &AnnotationSpec, ctx: &ScaleContext, out: &mut Vec<SceneNode>) {
    match spec {
        AnnotationSpec::Text { x, y, text, font_size, color, anchor, baseline, angle, dx, dy, .. } => {
            let px = ctx.resolve_x(x) + dx;
            let py = ctx.resolve_y(y) + dy;
            out.push(SceneNode::Text {
                x: px,
                y: py,
                content: text.to_string(),
                style: TextStyle {
                    font_size: *font_size,
                    font_weight: FontWeight::Normal,
                    anchor: parse_anchor(anchor),
                    baseline: parse_baseline(baseline),
                    angle: *angle,
                    color: resolve_color(color),
                    opacity: 1.0,
                    font_family: "Inter, system-ui, sans-serif".to_string(),
                },
            });
        }
        AnnotationSpec::Arrow { x, y, x2, y2, stroke, stroke_width, head_size, .. } => {
            let x1_px = ctx.resolve_x(x);
            let y1_px = ctx.resolve_y(y);
            let x2_px = ctx.resolve_x(x2);
            let y2_px = ctx.resolve_y(y2);
            emit_arrow(x1_px, y1_px, x2_px, y2_px, stroke, *stroke_width, *head_size, out);
        }
        AnnotationSpec::Rect { x1, y1, x2, y2, fill, opacity, corner_radius, stroke } => {
            let px1 = ctx.resolve_x(x1);
            let py1 = ctx.resolve_y(y1);
            let px2 = ctx.resolve_x(x2);
            let py2 = ctx.resolve_y(y2);
            let x = px1.min(px2);
            let y = py1.min(py2);
            let w = (px2 - px1).abs();
            let h = (py2 - py1).abs();
            let fill_color = resolve_color(fill);
            let stroke_color = stroke.as_deref().map(resolve_color);
            out.push(SceneNode::Rect {
                x, y, w, h,
                style: FillStroke {
                    fill: Some(fill_color),
                    stroke: stroke_color,
                    stroke_width: if stroke_color.is_some() { 1.0 } else { 0.0 },
                    opacity: 1.0,
                    stroke_dash: None,
                    stroke_opacity: 1.0,
                    // Store the caller-supplied opacity as fill_opacity so the SVG
                    // writer emits fill-opacity="<value>" on the <rect> element.
                    // The FillStroke.opacity field is not forwarded to SVG attributes
                    // by to_svg_fill_stroke_with_anchor; only fill_opacity is.
                    fill_opacity: *opacity,
                    angle: 0.0,
                },
                corner_radius: *corner_radius,
            });
        }
        AnnotationSpec::Line { x1, y1, x2, y2, stroke, stroke_width, dash } => {
            out.push(SceneNode::Line {
                x1: ctx.resolve_x(x1),
                y1: ctx.resolve_y(y1),
                x2: ctx.resolve_x(x2),
                y2: ctx.resolve_y(y2),
                style: StrokeStyle {
                    color: resolve_color(stroke),
                    width: *stroke_width,
                    opacity: 1.0,
                    dash: dash.as_deref().map(|d| d.to_vec()),
                    stroke_cap: None,
                    stroke_join: None,
                    stroke_opacity: 1.0,
                },
            });
        }
        AnnotationSpec::Span { axis, start, end, fill, opacity, label, .. } => {
            emit_span(axis, start, end, fill, *opacity, label.as_deref(), ctx, out);
        }
        AnnotationSpec::Bracket { x1, y1, x2, y2, label, direction, stroke, tip_length } => {
            emit_bracket(ctx.resolve_x(x1), ctx.resolve_y(y1), ctx.resolve_x(x2), ctx.resolve_y(y2),
                label, direction, stroke, *tip_length, out);
        }
        AnnotationSpec::Callout { x, y, text, arrow, padding, background, border_color, border_radius, text_x, text_y } => {
            emit_callout(ctx, x, y, text, arrow, *padding, background, border_color,
                *border_radius, text_x.as_ref(), text_y.as_ref(), out);
        }
        AnnotationSpec::Image { x, y, src, width, height, anchor } => {
            let mut px = ctx.resolve_x(x);
            let mut py = ctx.resolve_y(y);
            match anchor.as_str() {
                "start" | "left" => { /* px is already left edge */ }
                "end" | "right" => { px -= width; }
                _ => { px -= width * 0.5; py -= height * 0.5; }
            }
            // XML-escape src so that a URL containing '"' cannot break SVG structure.
            let escaped_src = src
                .replace('&', "&amp;")
                .replace('"', "&quot;")
                .replace('<', "&lt;")
                .replace('>', "&gt;");
            out.push(SceneNode::Raw {
                svg: format!(
                    r#"<image x="{}" y="{}" width="{}" height="{}" href="{}"/>"#,
                    px, py, width, height, escaped_src
                ),
            });
        }
    }
}

// ── Emitter helpers (extracted to stay under clippy's argument limit) ────────

#[allow(clippy::too_many_arguments)]
fn emit_arrow(
    x1: f64, y1: f64, x2: f64, y2: f64,
    stroke: &str, stroke_width: f64, head_size: f64,
    out: &mut Vec<SceneNode>,
) {
    let color = resolve_color(stroke);

    // Line shaft.
    out.push(SceneNode::Line {
        x1, y1, x2, y2,
        style: StrokeStyle {
            color,
            width: stroke_width,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        },
    });

    // Arrowhead: a small triangle at (x2, y2) pointing in the direction of travel.
    if head_size > 0.0 {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len = (dx * dx + dy * dy).sqrt();
        if len > 0.0 {
            let ux = dx / len;
            let uy = dy / len;
            let px = -uy;
            let py = ux;
            // Clamp head_size to the shaft length so an epsilon-length arrow
            // does not produce an arrowhead that vastly overshoots the start point.
            let head_size = head_size.min(len);
            let half = head_size * 0.5;
            let tip_x = x2;
            let tip_y = y2;
            let left_x = x2 - ux * head_size + px * half;
            let left_y = y2 - uy * head_size + py * half;
            let right_x = x2 - ux * head_size - px * half;
            let right_y = y2 - uy * head_size - py * half;

            out.push(SceneNode::Path {
                commands: vec![
                    PathCmd::MoveTo { x: tip_x, y: tip_y },
                    PathCmd::LineTo { x: left_x, y: left_y },
                    PathCmd::LineTo { x: right_x, y: right_y },
                    PathCmd::Close,
                ],
                style: FillStroke {
                    fill: Some(color),
                    stroke: None,
                    stroke_width: 0.0,
                    opacity: 1.0,
                    stroke_dash: None,
                    stroke_opacity: 1.0,
                    fill_opacity: 1.0,
                    angle: 0.0,
                },
                closed: true,
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_span(
    axis: &str, start: &CoordValue, end: &CoordValue,
    fill: &str, opacity: f64, label: Option<&str>,
    ctx: &ScaleContext, out: &mut Vec<SceneNode>,
) {
    let (x, y, w, h) = if axis == "x" {
        let x_start = ctx.resolve_x(start);
        let x_end = ctx.resolve_x(end);
        // Guard NaN: either coordinate being NaN produces NaN width — skip emission.
        if !x_start.is_finite() || !x_end.is_finite() {
            return;
        }
        let x = x_start.min(x_end);
        let w = (x_end - x_start).abs();
        (x, ctx.plot_area.y, w, ctx.plot_area.h)
    } else {
        let y_start = ctx.resolve_y(start);
        let y_end = ctx.resolve_y(end);
        // Guard NaN: either coordinate being NaN produces NaN height — skip emission.
        if !y_start.is_finite() || !y_end.is_finite() {
            return;
        }
        let y = y_start.min(y_end);
        let h = (y_end - y_start).abs();
        (ctx.plot_area.x, y, ctx.plot_area.w, h)
    };

    out.push(SceneNode::Rect {
        x, y, w, h,
        style: FillStroke {
            fill: Some(resolve_color(fill)),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: opacity,
            angle: 0.0,
        },
        corner_radius: 0.0,
    });

    if let Some(label_text) = label {
        if !label_text.is_empty() {
            out.push(SceneNode::Text {
                x: x + w * 0.5,
                y: y + h * 0.5,
                content: label_text.to_string(),
                style: TextStyle {
                    font_size: 11.0,
                    font_weight: FontWeight::Normal,
                    anchor: TextAnchor::Middle,
                    baseline: TextBaseline::Middle,
                    angle: 0.0,
                    color: Color::rgb(51, 51, 51),
                    opacity: 1.0,
                    font_family: "Inter, system-ui, sans-serif".to_string(),
                },
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_bracket(
    px1: f64, py1: f64, px2: f64, py2: f64,
    label: &str, direction: &str, stroke: &str, tip_length: f64,
    out: &mut Vec<SceneNode>,
) {
    let color = resolve_color(stroke);
    let stroke_style = StrokeStyle {
        color,
        width: 1.5,
        opacity: 1.0,
        dash: None,
        stroke_cap: None,
        stroke_join: None,
        stroke_opacity: 1.0,
    };

    // Baseline.
    out.push(SceneNode::Line {
        x1: px1, y1: py1, x2: px2, y2: py2,
        style: stroke_style.clone(),
    });

    let (tip_dx, tip_dy) = match direction {
        "up" => (0.0, -tip_length),
        "down" => (0.0, tip_length),
        "left" => (-tip_length, 0.0),
        "right" => (tip_length, 0.0),
        _ => (0.0, -tip_length),
    };

    // Left tip.
    out.push(SceneNode::Line {
        x1: px1, y1: py1,
        x2: px1 + tip_dx, y2: py1 + tip_dy,
        style: stroke_style.clone(),
    });

    // Right tip.
    out.push(SceneNode::Line {
        x1: px2, y1: py2,
        x2: px2 + tip_dx, y2: py2 + tip_dy,
        style: stroke_style,
    });

    // Label centered above/below the bracket.
    if !label.is_empty() {
        let mid_x = (px1 + px2) * 0.5;
        let mid_y = (py1 + py2) * 0.5;
        let label_offset = tip_length + 4.0;
        let (off_x, off_y) = match direction {
            "up" => (0.0, -label_offset),
            "down" => (0.0, label_offset),
            "left" => (-label_offset, 0.0),
            "right" => (label_offset, 0.0),
            _ => (0.0, -label_offset),
        };
        out.push(SceneNode::Text {
            x: mid_x + off_x,
            y: mid_y + off_y,
            content: label.to_string(),
            style: TextStyle {
                font_size: 11.0,
                font_weight: FontWeight::Normal,
                anchor: TextAnchor::Middle,
                baseline: TextBaseline::Middle,
                angle: 0.0,
                color: Color::rgb(51, 51, 51),
                opacity: 1.0,
                font_family: "Inter, system-ui, sans-serif".to_string(),
            },
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_callout(
    ctx: &ScaleContext, x: &CoordValue, y: &CoordValue,
    text: &str, arrow: &str, padding: f64,
    background: &str, border_color: &str, border_radius: f64,
    text_x: Option<&CoordValue>, text_y: Option<&CoordValue>,
    out: &mut Vec<SceneNode>,
) {
    let data_x = ctx.resolve_x(x);
    let data_y = ctx.resolve_y(y);

    let default_offset = 30.0;
    let tx = text_x.map(|v| ctx.resolve_x(v)).unwrap_or(data_x + default_offset);
    let ty = text_y.map(|v| ctx.resolve_y(v)).unwrap_or(data_y - default_offset);

    // Approximate text dimensions for the background box.
    // Use chars().count() not len() so multi-byte Unicode characters don't
    // inflate the estimated width (len() returns byte count, not char count).
    let char_width = 7.0;
    let text_w = text.chars().count() as f64 * char_width + padding * 2.0;
    let text_h = 14.0 + padding * 2.0;

    // Background rect.
    out.push(SceneNode::Rect {
        x: tx - text_w * 0.5,
        y: ty - text_h * 0.5,
        w: text_w,
        h: text_h,
        style: FillStroke {
            fill: Some(resolve_color(background)),
            stroke: Some(resolve_color(border_color)),
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        },
        corner_radius: border_radius,
    });

    // Leader line: draw for any style except "none".
    // Documented values: "curved", "straight", "none". Legacy "true"/"yes" also draw.
    if arrow != "none" {
        out.push(SceneNode::Line {
            x1: data_x,
            y1: data_y,
            x2: tx,
            y2: ty,
            style: StrokeStyle {
                color: resolve_color(border_color),
                width: 1.0,
                opacity: 1.0,
                dash: None,
                stroke_cap: None,
                stroke_join: None,
                stroke_opacity: 1.0,
            },
        });
    }

    // Text label.
    out.push(SceneNode::Text {
        x: tx,
        y: ty,
        content: text.to_string(),
        style: TextStyle {
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            anchor: TextAnchor::Middle,
            baseline: TextBaseline::Middle,
            angle: 0.0,
            color: Color::rgb(51, 51, 51),
            opacity: 1.0,
            font_family: "Inter, system-ui, sans-serif".to_string(),
        },
    });
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::scale_resolve::ScaleKind;
    use crate::scale::linear::LinearScale;

    /// Create a simple linear scale context for testing.
    /// Domain [0, 100], x pixel range [50, 550], y pixel range [20, 320].
    fn test_ctx() -> (ScaleKind, ScaleKind, Rect) {
        let plot_area = Rect { x: 50.0, y: 20.0, w: 500.0, h: 300.0 };
        let x_scale = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 100.0], vec![50.0, 550.0], false, false,
        ));
        let y_scale = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 100.0], vec![20.0, 320.0], false, false,
        ));
        (x_scale, y_scale, plot_area)
    }

    #[test]
    fn resolve_data_coord_x() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext { plot_area, x_scale: &x_scale, y_scale: &y_scale };
        // Data value 50 should map to the center of the x pixel range.
        let px = ctx.resolve_x(&CoordValue::Data(50.0));
        assert!((px - 300.0).abs() < 1.0, "expected ~300, got {px}");
    }

    #[test]
    fn resolve_pixel_coord() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext { plot_area, x_scale: &x_scale, y_scale: &y_scale };
        let px = ctx.resolve_x(&CoordValue::Pixel { px: 100.0 });
        assert!((px - 150.0).abs() < f64::EPSILON, "expected 150, got {px}");
    }

    #[test]
    fn resolve_norm_coord() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext { plot_area, x_scale: &x_scale, y_scale: &y_scale };
        let px = ctx.resolve_x(&CoordValue::Norm { norm: 0.5 });
        assert!((px - 300.0).abs() < f64::EPSILON, "expected 300, got {px}");
    }

    #[test]
    fn build_text_annotation() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext { plot_area, x_scale: &x_scale, y_scale: &y_scale };
        let specs = vec![AnnotationSpec::Text {
            x: CoordValue::Norm { norm: 0.5 },
            y: CoordValue::Norm { norm: 0.5 },
            text: "hello".to_string(),
            font_size: 14.0,
            color: "#ff0000".to_string(),
            anchor: "middle".to_string(),
            baseline: "middle".to_string(),
            angle: 0.0,
            dx: 0.0,
            dy: 0.0,
            z: "front".to_string(),
        }];
        let nodes = build_annotations(&specs, &ctx);
        assert_eq!(nodes.len(), 1);
        match &nodes[0] {
            SceneNode::Text { x, y, content, style } => {
                assert!((x - 300.0).abs() < f64::EPSILON);
                assert!((y - 170.0).abs() < f64::EPSILON);
                assert_eq!(content, "hello");
                assert_eq!(style.font_size, 14.0);
                assert_eq!(style.color, Color::rgb(255, 0, 0));
            }
            _ => panic!("expected Text node"),
        }
    }

    #[test]
    fn build_line_annotation() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext { plot_area, x_scale: &x_scale, y_scale: &y_scale };
        let specs = vec![AnnotationSpec::Line {
            x1: CoordValue::Norm { norm: 0.0 },
            y1: CoordValue::Norm { norm: 0.5 },
            x2: CoordValue::Norm { norm: 1.0 },
            y2: CoordValue::Norm { norm: 0.5 },
            stroke: "#000000".to_string(),
            stroke_width: 2.0,
            dash: Some(vec![4.0, 2.0]),
        }];
        let nodes = build_annotations(&specs, &ctx);
        assert_eq!(nodes.len(), 1);
        match &nodes[0] {
            SceneNode::Line { x1, y1, x2, y2, style } => {
                assert!((x1 - 50.0).abs() < f64::EPSILON);
                assert!((x2 - 550.0).abs() < f64::EPSILON);
                assert!((y1 - 170.0).abs() < f64::EPSILON);
                assert!((y2 - 170.0).abs() < f64::EPSILON);
                assert_eq!(style.width, 2.0);
                assert_eq!(style.dash.as_deref(), Some(&[4.0, 2.0][..]));
            }
            _ => panic!("expected Line node"),
        }
    }

    #[test]
    fn build_arrow_annotation() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext { plot_area, x_scale: &x_scale, y_scale: &y_scale };
        let specs = vec![AnnotationSpec::Arrow {
            x: CoordValue::Norm { norm: 0.0 },
            y: CoordValue::Norm { norm: 0.0 },
            x2: CoordValue::Norm { norm: 1.0 },
            y2: CoordValue::Norm { norm: 1.0 },
            stroke: "#333333".to_string(),
            stroke_width: 1.5,
            head_size: 8.0,
            curve: "straight".to_string(),
        }];
        let nodes = build_annotations(&specs, &ctx);
        // Arrow produces a line + arrowhead path.
        assert_eq!(nodes.len(), 2);
        assert!(matches!(&nodes[0], SceneNode::Line { .. }));
        assert!(matches!(&nodes[1], SceneNode::Path { .. }));
    }

    #[test]
    fn build_span_annotation() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext { plot_area, x_scale: &x_scale, y_scale: &y_scale };
        let specs = vec![AnnotationSpec::Span {
            axis: "x".to_string(),
            start: CoordValue::Norm { norm: 0.2 },
            end: CoordValue::Norm { norm: 0.8 },
            fill: "#aaaaaa".to_string(),
            opacity: 0.3,
            label_position: "center".to_string(),
            label: Some("span".to_string()),
        }];
        let nodes = build_annotations(&specs, &ctx);
        // Span produces a rect + label text.
        assert_eq!(nodes.len(), 2);
        assert!(matches!(&nodes[0], SceneNode::Rect { .. }));
        assert!(matches!(&nodes[1], SceneNode::Text { .. }));
    }

    #[test]
    fn deser_annotation_spec_text() {
        let json = r#"{"type": "text", "x": 50.0, "y": {"norm": 0.5}, "text": "hi"}"#;
        let spec: AnnotationSpec = serde_json::from_str(json).unwrap();
        assert!(matches!(spec, AnnotationSpec::Text { .. }));
    }

    #[test]
    fn deser_annotation_spec_line() {
        let json = r##"{"type": "line", "x1": {"px": 0}, "y1": {"norm": 0.5}, "x2": {"px": 100}, "y2": {"norm": 0.5}, "stroke": "#000"}"##;
        let spec: AnnotationSpec = serde_json::from_str(json).unwrap();
        assert!(matches!(spec, AnnotationSpec::Line { .. }));
    }

    #[test]
    fn color_fallback_on_invalid() {
        let c = resolve_color("not-a-valid-color-xyz");
        assert_eq!(c, Color::rgb(51, 51, 51));
    }

    #[test]
    fn empty_specs_returns_empty_vec() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext { plot_area, x_scale: &x_scale, y_scale: &y_scale };
        let nodes = build_annotations(&[], &ctx);
        assert!(nodes.is_empty());
    }
}
