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
    /// When `false`, tick labels are suppressed (D7: `axis.labels`).
    /// Default `true` — preserves byte-identity for all existing goldens.
    pub show_labels: bool,
    /// When `false`, tick marks are suppressed (D7: `axis.ticks`).
    /// Default `true`.
    pub show_ticks: bool,
    /// When `false`, the axis domain line is suppressed (D7: `axis.domain`).
    /// Default `true`.
    pub show_domain: bool,
    /// When `false`, gridlines for this axis are suppressed even when the theme
    /// enables them globally (D7: `axis.grid`). Default `true`.
    pub show_grid: bool,
    /// Optional d3-format string applied to each tick label before layout
    /// (D12: `encoding.format` on x/y axes). `None` → use the scale's own
    /// default formatter (existing behavior).
    pub tick_format: Option<String>,
    /// When `Some("time")`, `tick_format` is a time format spec (D12:
    /// `encoding.format_type`). Currently unused by `layout_x_axis` /
    /// `layout_y_axis` — tick strings are already pre-formatted before this
    /// struct is built. Reserved for future granularity hints.
    pub tick_format_type: Option<String>,
}

impl AxisInput {
    /// Construct an `AxisInput` with all new D7/D12 fields at their
    /// backward-compatible defaults (all show_* = true, no tick_format).
    pub fn new(
        orient: AxisOrient,
        title: Option<String>,
        tick_labels: Vec<String>,
        label_angle_override: Option<f64>,
    ) -> Self {
        Self {
            orient,
            title,
            tick_labels,
            label_angle_override,
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
            tick_format: None,
            tick_format_type: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AxesInput {
    pub x: AxisInput,
    pub y: AxisInput,
    /// When false, the x axis line + ticks + labels + title are suppressed
    /// at layout time. Used by `ChartSpec.axis_x = Some(false)` (i.e.
    /// `Chart.axis(x=False)`) on clustermap dendrogram panels and JointChart
    /// marginal panels. Default `true`.
    pub show_x: bool,
    /// Y-axis variant of `show_x`. Default `true`.
    pub show_y: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisLayout {
    pub orient: AxisOrient,
    pub panel_index: usize,
    pub axis_line: Rect,
    pub ticks: Vec<TickLayout>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub title: Option<AxisTitleLayout>,
    /// D7: whether to render tick labels. Default `true`.
    #[serde(default = "default_true")]
    pub show_labels: bool,
    /// D7: whether to render tick marks. Default `true`.
    #[serde(default = "default_true")]
    pub show_ticks: bool,
    /// D7: whether to render the axis domain line. Default `true`.
    #[serde(default = "default_true")]
    pub show_domain: bool,
    /// D7: whether to render gridlines from this axis. Default `true`.
    #[serde(default = "default_true")]
    pub show_grid: bool,
}

fn default_true() -> bool { true }

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
        show_labels: input.show_labels,
        show_ticks: input.show_ticks,
        show_domain: input.show_domain,
        show_grid: input.show_grid,
    }
}

use crate::layout::{LABEL_OVERLAP_TOLERANCE, DEFAULT_LABEL_ANGLE};

/// Per-x-axis warning the orchestrator may emit. Internal — consumers translate
/// to `LayoutWarning`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum XAxisWarning {
    LabelsElided { count: u32 },
}

/// Truncate `label` by char prefix until the measured width plus the ellipsis
/// width fits in `max_width`. Returns the truncated label with "…" appended.
/// If even "…" alone exceeds max_width, returns "…" anyway (caller is already
/// in a degenerate state).
fn elide_to_fit(
    label: &str,
    max_width: f64,
    font_size: f64,
    metrics: &dyn TextMetrics,
) -> String {
    let ellipsis = '…';
    let ellipsis_w = metrics.measure_width(&ellipsis.to_string(), font_size);
    if ellipsis_w >= max_width {
        return ellipsis.to_string();
    }
    let budget = max_width - ellipsis_w;
    let mut out = String::new();
    for ch in label.chars() {
        let mut tentative = out.clone();
        tentative.push(ch);
        if metrics.measure_width(&tentative, font_size) > budget {
            break;
        }
        out = tentative;
    }
    out.push(ellipsis);
    out
}

/// Build the AxisLayout for the x-axis (Bottom orient) of a single panel.
/// Tick positions are uniformly spaced across `panel_area.w` (spec §14.3 step 7a).
/// Collision policy: rotate labels then elide if still colliding (spec §14.4).
pub fn layout_x_axis(
    input: &AxisInput,
    panel_area: Rect,
    panel_index: usize,
    label_font_size: f64,
    title_font_size: f64,
    axis_title_padding: f64,
    metrics: &dyn TextMetrics,
) -> (AxisLayout, Option<XAxisWarning>) {
    let n = input.tick_labels.len();
    let slot_w = if n > 0 { panel_area.w / n as f64 } else { 0.0 };

    // Step 1: measure all labels flat.
    let widths: Vec<f64> = input
        .tick_labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .collect();

    // Step 2: decide whether any label exceeds slot * (1 - tolerance).
    let threshold = slot_w * (1.0 - LABEL_OVERLAP_TOLERANCE);
    let any_collision = widths.iter().any(|w| *w > threshold);
    // When `label_angle_override` is explicitly set (from `axis.label_angle`),
    // force that angle regardless of collision detection. Otherwise apply the
    // default auto-rotation only when collision is detected.
    let forced_angle = input.label_angle_override;
    let angle = if let Some(override_angle) = forced_angle {
        override_angle
    } else if any_collision {
        DEFAULT_LABEL_ANGLE
    } else {
        0.0
    };

    // Step 3: collision recovery — rotation, then elision (Tasks 11 + 12).
    // Phase 1 of this task: produce flat ticks if no collision.
    // When a forced angle is set and there's no collision, still apply the angle.
    let (ticks, warning) = if !any_collision && forced_angle.is_none() {
        let ticks: Vec<TickLayout> = input
            .tick_labels
            .iter()
            .enumerate()
            .map(|(i, label)| TickLayout {
                position: panel_area.x + (i as f64 + 0.5) * slot_w,
                label: label.clone(),
                label_angle: 0.0,
                elided: false,
            })
            .collect();
        (ticks, None)
    } else if !any_collision && forced_angle.is_some() {
        // No collision but angle forced by `axis.label_angle`.
        let ticks: Vec<TickLayout> = input
            .tick_labels
            .iter()
            .enumerate()
            .map(|(i, label)| TickLayout {
                position: panel_area.x + (i as f64 + 0.5) * slot_w,
                label: label.clone(),
                label_angle: angle,
                elided: false,
            })
            .collect();
        (ticks, None)
    } else {
        // Rotated projection: |L * cos(angle)|. Spec §6 step 7c.
        let cos_factor = (angle.to_radians()).cos().abs();
        let any_still_colliding = widths.iter().any(|w| *w * cos_factor > slot_w);
        let mut elided_count: u32 = 0;
        let ticks: Vec<TickLayout> = input
            .tick_labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let w = widths[i];
                let needs_elide = any_still_colliding && (w * cos_factor > slot_w);
                let final_label = if needs_elide {
                    elided_count += 1;
                    // Available pixel budget for the rotated label projection is slot_w;
                    // the actual measured width budget is slot_w / cos(|angle|).
                    let budget = slot_w / cos_factor.max(1e-6);
                    elide_to_fit(label, budget, label_font_size, metrics)
                } else {
                    label.clone()
                };
                TickLayout {
                    position: panel_area.x + (i as f64 + 0.5) * slot_w,
                    label: final_label,
                    label_angle: angle,
                    elided: needs_elide,
                }
            })
            .collect();
        let warning = if elided_count > 0 {
            Some(XAxisWarning::LabelsElided { count: elided_count })
        } else {
            None
        };
        (ticks, warning)
    };

    let axis_line = Rect {
        x: panel_area.x,
        y: panel_area.y + panel_area.h,
        w: panel_area.w,
        h: 1.0,
    };

    let title = input.title.as_ref().map(|text| {
        let title_h = metrics.line_height(title_font_size);
        let label_h = metrics.line_height(label_font_size);
        AxisTitleLayout {
            text: text.clone(),
            anchor_x: panel_area.x + panel_area.w / 2.0,
            anchor_y: panel_area.y + panel_area.h + label_h + axis_title_padding + title_h / 2.0,
            angle: 0.0,
        }
    });

    (AxisLayout {
        orient: AxisOrient::Bottom,
        panel_index,
        axis_line,
        ticks,
        title,
        show_labels: input.show_labels,
        show_ticks: input.show_ticks,
        show_domain: input.show_domain,
        show_grid: input.show_grid,
    }, warning)
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
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
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
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
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
        let input = AxisInput::new(
            AxisOrient::Left,
            None,
            vec!["0".into(), "100".into(), "10000".into()],
            None,
        );
        let m = mock(10.0);
        let band = compute_y_label_band_width(&input, 11.0, &m);
        assert_eq!(band, 50.0);
    }

    #[test]
    fn y_axis_label_band_empty_labels_returns_zero() {
        let input = AxisInput::new(AxisOrient::Left, None, vec![], None);
        let m = mock(10.0);
        assert_eq!(compute_y_label_band_width(&input, 11.0, &m), 0.0);
    }

    #[test]
    fn y_axis_layout_uniform_tick_positions() {
        let input = AxisInput::new(
            AxisOrient::Left,
            Some("Price".into()),
            vec!["0".into(), "1".into(), "2".into(), "3".into()],
            None,
        );
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

    #[test]
    fn x_axis_no_collision_keeps_labels_flat() {
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 50.0, line_h_factor: 1.2 };
        let (axis, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        assert_eq!(axis.ticks.len(), 4);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, 0.0);
            assert!(!t.elided);
        }
        assert!(warning.is_none());
    }

    #[test]
    fn x_axis_uniform_tick_positions_along_axis() {
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            None,
        );
        let panel_area = Rect { x: 100.0, y: 50.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 10.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        assert!((axis.ticks[0].position - (100.0 + 50.0)).abs() < 1e-9);
        assert!((axis.ticks[1].position - (100.0 + 150.0)).abs() < 1e-9);
        assert!((axis.ticks[2].position - (100.0 + 250.0)).abs() < 1e-9);
        assert!((axis.ticks[3].position - (100.0 + 350.0)).abs() < 1e-9);
    }

    #[test]
    fn x_axis_collision_triggers_default_45_rotation() {
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..8).map(|i| format!("L{}", i)).collect(),
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
        }
    }

    #[test]
    fn x_axis_rotates_at_custom_angle_override() {
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..8).map(|i| format!("L{}", i)).collect(),
            Some(-90.0),
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -90.0);
        }
    }

    #[test]
    fn x_axis_rotation_only_no_elision_when_rotated_fits() {
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..6).map(|i| format!("L{}", i)).collect(),
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 600.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 95.0, line_h_factor: 1.2 };
        let (axis, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
            assert!(!t.elided, "rotated projection should fit; no elision");
        }
        assert!(warning.is_none());
    }

    #[test]
    fn x_axis_elides_with_ellipsis_when_rotated_still_collides() {
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..20).map(|i| format!("Label_{}", i)).collect(),
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let (axis, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
            assert!(t.elided, "expected all 20 labels to be elided");
            assert!(t.label.ends_with('…'), "expected ellipsis suffix; got {:?}", t.label);
        }
        match warning {
            Some(XAxisWarning::LabelsElided { count }) => assert_eq!(count, 20),
            other => panic!("expected LabelsElided{{count: 20}}, got {:?}", other),
        }
    }

    #[test]
    fn x_axis_elision_unicode_safe() {
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            vec!["héllo wörld".into(); 20],
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        for t in &axis.ticks {
            assert!(t.elided);
            assert!(t.label.is_char_boundary(t.label.len()));
        }
    }
}
