//! Break indicator geometry — the slash, zigzag and wave glyphs drawn in the
//! pixel band a break reserves between two visible segments.
//!
//! Split out of `break_axis/mod.rs` (#229). The most self-contained of that
//! module's five jobs: its whole input is a finished
//! [`BreakResult`](super::BreakResult) plus a theme, its whole output is a
//! `Vec<SceneNode>` nothing in this subsystem reads back. Precedent for the
//! split: `render/scale_resolve/`, `projection/`, `transform/stats/`,
//! `transform/smooth/`.

use ferrum_scene::{FillStroke, PathCmd, SceneNode};

use crate::layout::{Rect, ThemeInputs};
use crate::render::draw::to_scene_stroke;
use crate::spec::axis::AlignAxis;

use super::BreakResult;

/// Render break indicators at `result.indicator_pixels` for the given axis and style.
///
/// `plot_area` provides the perpendicular extent (e.g. for a Y-axis break, the
/// x-extent of the plot area determines where the indicators are drawn).
pub fn build_break_indicators(
    result: &BreakResult,
    plot_area: &Rect,
    axis: AlignAxis,
    break_size: f64,
    break_style: &str,
    theme: &ThemeInputs,
) -> Vec<SceneNode> {
    let mut nodes = Vec::new();

    let stroke = to_scene_stroke(theme.colors.axis_line_color, theme.sizes.axis_line_width, 1.0, None, None, None);

    for &indicator_px in &result.indicator_pixels {
        let new_nodes = match break_style {
            "gap" => Vec::new(),
            "zigzag" => build_zigzag_indicator(indicator_px, plot_area, axis, break_size, stroke.clone()),
            "wave" => build_wave_indicator(indicator_px, plot_area, axis, break_size, stroke.clone()),
            _ => build_slash_indicator(indicator_px, plot_area, axis, break_size, stroke.clone()),
        };
        nodes.extend(new_nodes);
    }

    nodes
}

/// Two parallel diagonal slashes (`//`) at the break position.
fn build_slash_indicator(
    indicator_px: f64,
    plot_area: &Rect,
    axis: AlignAxis,
    break_size: f64,
    stroke: ferrum_scene::StrokeStyle,
) -> Vec<SceneNode> {
    let half = break_size / 2.0;
    let diag = 6.0; // diagonal extension in pixels

    if axis.is_y() {
        let x1 = plot_area.x + plot_area.w * 0.35;
        let x2 = plot_area.x + plot_area.w * 0.65;
        let x_mid = (x1 + x2) / 2.0;

        vec![
            // First slash
            SceneNode::Line {
                x1: x_mid - diag / 2.0,
                y1: indicator_px - half,
                x2: x_mid + diag / 2.0,
                y2: indicator_px + half,
                style: stroke.clone(),
            },
            // Second slash (offset to the right)
            SceneNode::Line {
                x1: x_mid - diag / 2.0 + diag,
                y1: indicator_px - half,
                x2: x_mid + diag / 2.0 + diag,
                y2: indicator_px + half,
                style: stroke,
            },
        ]
    } else {
        // x-axis break
        let y1 = plot_area.y + plot_area.h * 0.35;
        let y2 = plot_area.y + plot_area.h * 0.65;
        let y_mid = (y1 + y2) / 2.0;

        vec![
            SceneNode::Line {
                x1: indicator_px - half,
                y1: y_mid - diag / 2.0,
                x2: indicator_px + half,
                y2: y_mid + diag / 2.0,
                style: stroke.clone(),
            },
            SceneNode::Line {
                x1: indicator_px - half,
                y1: y_mid - diag / 2.0 + diag,
                x2: indicator_px + half,
                y2: y_mid + diag / 2.0 + diag,
                style: stroke,
            },
        ]
    }
}

/// A zigzag path at the break position.
fn build_zigzag_indicator(
    indicator_px: f64,
    plot_area: &Rect,
    axis: AlignAxis,
    break_size: f64,
    stroke: ferrum_scene::StrokeStyle,
) -> Vec<SceneNode> {
    let half = break_size / 2.0;
    let amplitude = 5.0;
    let n_zags = 4i32;

    let commands = if axis.is_y() {
        let x_center = plot_area.x + plot_area.w / 2.0;
        let mut cmds = Vec::new();
        cmds.push(PathCmd::MoveTo { x: x_center - amplitude, y: indicator_px - half });
        for i in 0..=n_zags {
            let t = i as f64 / n_zags as f64;
            let y = indicator_px - half + t * break_size;
            let x = if i % 2 == 0 {
                x_center - amplitude
            } else {
                x_center + amplitude
            };
            cmds.push(PathCmd::LineTo { x, y });
        }
        cmds
    } else {
        let y_center = plot_area.y + plot_area.h / 2.0;
        let mut cmds = Vec::new();
        cmds.push(PathCmd::MoveTo { x: indicator_px - half, y: y_center - amplitude });
        for i in 0..=n_zags {
            let t = i as f64 / n_zags as f64;
            let x = indicator_px - half + t * break_size;
            let y = if i % 2 == 0 {
                y_center - amplitude
            } else {
                y_center + amplitude
            };
            cmds.push(PathCmd::LineTo { x, y });
        }
        cmds
    };

    vec![SceneNode::Path {
        commands,
        style: FillStroke {
            fill: None,
            stroke: Some(stroke.color),
            stroke_width: stroke.width,
            opacity: stroke.opacity,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        },
        closed: false,
    }]
}

/// A wave/tilde path at the break position.
fn build_wave_indicator(
    indicator_px: f64,
    plot_area: &Rect,
    axis: AlignAxis,
    break_size: f64,
    stroke: ferrum_scene::StrokeStyle,
) -> Vec<SceneNode> {
    let half = break_size / 2.0;
    let amplitude = 4.0;
    let n_steps = 20i32;

    let commands = if axis.is_y() {
        let x_center = plot_area.x + plot_area.w / 2.0;
        let mut cmds = Vec::new();
        cmds.push(PathCmd::MoveTo {
            x: x_center,
            y: indicator_px - half,
        });
        for i in 1..=n_steps {
            let t = i as f64 / n_steps as f64;
            let y = indicator_px - half + t * break_size;
            let x = x_center + amplitude * (std::f64::consts::PI * 2.0 * t).sin();
            cmds.push(PathCmd::LineTo { x, y });
        }
        cmds
    } else {
        let y_center = plot_area.y + plot_area.h / 2.0;
        let mut cmds = Vec::new();
        cmds.push(PathCmd::MoveTo {
            x: indicator_px - half,
            y: y_center,
        });
        for i in 1..=n_steps {
            let t = i as f64 / n_steps as f64;
            let x = indicator_px - half + t * break_size;
            let y = y_center + amplitude * (std::f64::consts::PI * 2.0 * t).sin();
            cmds.push(PathCmd::LineTo { x, y });
        }
        cmds
    };

    vec![SceneNode::Path {
        commands,
        style: FillStroke {
            fill: None,
            stroke: Some(stroke.color),
            stroke_width: stroke.width,
            opacity: stroke.opacity,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        },
        closed: false,
    }]
}
