//! Axis input (caller-supplied) and axis layout output (engine-computed).
//! Per spec §14.1: tick labels are caller-pre-computed via Phase 4 scales;
//! Phase 6 never touches scale internals.

use serde::{Deserialize, Serialize};

use super::geometry::Rect;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisOrient {
    Top,
    Bottom,
    Left,
    Right,
}

/// Caller-supplied per-axis input. Phase 6 takes both x and y always.
#[derive(Debug, Clone, PartialEq)]
pub struct AxisInput {
    pub orient: AxisOrient,
    pub title: Option<String>,
    pub tick_labels: Vec<String>,
    pub label_angle_override: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AxesInput {
    pub x: AxisInput,
    pub y: AxisInput,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisLayout {
    pub orient: AxisOrient,
    pub panel_index: usize,
    pub axis_line: Rect,
    pub ticks: Vec<TickLayout>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub title: Option<AxisTitleLayout>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickLayout {
    pub position: f64,
    pub label: String,
    pub label_angle: f64,
    pub elided: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisTitleLayout {
    pub text: String,
    pub anchor_x: f64,
    pub anchor_y: f64,
    pub angle: f64,
}

use super::text_metrics::TextMetrics;

/// Returns the pixel width of the widest tick label on the y-axis. Used by the
/// orchestrator to reserve a left gutter before computing the plot rect.
pub fn compute_y_label_band_width(
    input: &AxisInput,
    label_font_size: f64,
    metrics: &dyn TextMetrics,
) -> f64 {
    input
        .tick_labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .fold(0.0_f64, f64::max)
}

/// Returns the title-row width contribution: title text height (rotated 90°,
/// so its "width" along the x-axis is its line height) plus axis_title_padding.
/// Returns 0 if there is no title.
pub fn compute_y_title_width(
    input: &AxisInput,
    title_font_size: f64,
    axis_title_padding: f64,
    metrics: &dyn TextMetrics,
) -> f64 {
    if input.title.is_some() {
        metrics.line_height(title_font_size) + axis_title_padding
    } else {
        0.0
    }
}

/// Build the AxisLayout for the y-axis (Left orient) of a single panel.
/// Tick positions are uniformly spaced across `panel_area.h`; no collision
/// policy applies to y-axis (spec §14.4).
pub fn layout_y_axis(
    input: &AxisInput,
    panel_area: Rect,
    panel_index: usize,
    label_font_size: f64,
    title_font_size: f64,
    axis_title_padding: f64,
    metrics: &dyn TextMetrics,
) -> AxisLayout {
    let n = input.tick_labels.len();
    let slot_h = if n > 0 { panel_area.h / n as f64 } else { 0.0 };
    let ticks: Vec<TickLayout> = input
        .tick_labels
        .iter()
        .enumerate()
        .map(|(i, label)| TickLayout {
            position: panel_area.y + (i as f64 + 0.5) * slot_h,
            label: label.clone(),
            label_angle: 0.0,
            elided: false,
        })
        .collect();

    let axis_line = Rect {
        x: panel_area.x,
        y: panel_area.y,
        w: 1.0,
        h: panel_area.h,
    };

    let title = input.title.as_ref().map(|text| {
        let label_band = compute_y_label_band_width(input, label_font_size, metrics);
        let title_h = metrics.line_height(title_font_size);
        AxisTitleLayout {
            text: text.clone(),
            anchor_x: panel_area.x - label_band - axis_title_padding - title_h / 2.0,
            anchor_y: panel_area.y + panel_area.h / 2.0,
            angle: -90.0,
        }
    });

    AxisLayout {
        orient: AxisOrient::Left,
        panel_index,
        axis_line,
        ticks,
        title,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_layout_round_trip() {
        let a = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 50.0, y: 350.0, w: 500.0, h: 1.0 },
            ticks: vec![TickLayout {
                position: 100.0,
                label: "0".into(),
                label_angle: 0.0,
                elided: false,
            }],
            title: Some(AxisTitleLayout {
                text: "Price".into(),
                anchor_x: 300.0,
                anchor_y: 380.0,
                angle: 0.0,
            }),
        };
        let json = serde_json::to_string(&a).unwrap();
        let parsed: AxisLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, a);
    }

    #[test]
    fn axis_layout_serde_lowercases_orient() {
        let a = AxisLayout {
            orient: AxisOrient::Left,
            panel_index: 0,
            axis_line: Rect::ZERO,
            ticks: vec![],
            title: None,
        };
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains(r#""orient":"left""#));
        assert!(!json.contains("title"));
    }

    use crate::layout::text_metrics::{fixed_width, MockMetrics};

    fn mock(per_char_px: f64) -> MockMetrics<impl Fn(&str, f64) -> f64> {
        MockMetrics { measure: fixed_width(per_char_px), line_h_factor: 1.2 }
    }

    #[test]
    fn y_axis_label_band_uses_longest_label() {
        let input = AxisInput {
            orient: AxisOrient::Left,
            title: None,
            tick_labels: vec!["0".into(), "100".into(), "10000".into()],
            label_angle_override: None,
        };
        let m = mock(10.0);
        let band = compute_y_label_band_width(&input, 11.0, &m);
        assert_eq!(band, 50.0);
    }

    #[test]
    fn y_axis_label_band_empty_labels_returns_zero() {
        let input = AxisInput {
            orient: AxisOrient::Left,
            title: None,
            tick_labels: vec![],
            label_angle_override: None,
        };
        let m = mock(10.0);
        assert_eq!(compute_y_label_band_width(&input, 11.0, &m), 0.0);
    }

    #[test]
    fn y_axis_layout_uniform_tick_positions() {
        let input = AxisInput {
            orient: AxisOrient::Left,
            title: Some("Price".into()),
            tick_labels: vec!["0".into(), "1".into(), "2".into(), "3".into()],
            label_angle_override: None,
        };
        let panel_area = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
        let m = mock(10.0);
        let axis = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        assert_eq!(axis.orient, AxisOrient::Left);
        assert_eq!(axis.panel_index, 0);
        assert_eq!(axis.ticks.len(), 4);
        assert!((axis.ticks[0].position - (50.0 + 25.0)).abs() < 1e-9);
        assert!((axis.ticks[3].position - (50.0 + 175.0)).abs() < 1e-9);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, 0.0);
            assert!(!t.elided);
        }
        let title = axis.title.unwrap();
        assert_eq!(title.text, "Price");
        assert!((title.angle - (-90.0)).abs() < 1e-9);
    }
}
