//! Internal: build axis and grid scene nodes from an AxisLayout.

use crate::layout::{AxisLayout, AxisOrient, Rect, TextAnchor, ThemeInputs};
use crate::render::draw::{to_scene_stroke, to_scene_text_style};
use ferrum_scene::SceneNode;

pub fn build_axis(axis: &AxisLayout, theme: &ThemeInputs) -> Vec<SceneNode> {
    let mut nodes = Vec::new();
    let r = axis.axis_line;

    // Domain line.
    if theme.axis.axis_line && axis.show_domain {
        nodes.push(SceneNode::Line {
            x1: r.x,
            y1: r.y,
            x2: r.x + r.w,
            y2: r.y + r.h,
            style: to_scene_stroke(theme.colors.axis_line_color, theme.sizes.axis_line_width, 1.0, None, None, None),
        });
    }

    let tick_stroke = to_scene_stroke(theme.colors.tick_color, theme.sizes.tick_width, 1.0, None, None, None);

    let label_fw: Option<&str> = if theme.typography.font_weight == "normal" {
        None
    } else {
        Some(&theme.typography.font_weight)
    };

    // Default per-orient gap between tick mark end and label baseline/edge.
    // L-2: guard negative label_padding — negative values would place labels
    // inside the tick area, producing overlapping or invisible labels.
    let label_pad = axis.label_padding.unwrap_or(2.0).max(0.0);

    for tick in &axis.ticks {
        let effective_font_size = tick.label_font_size.unwrap_or(theme.typography.label_font_size);

        let (tx1, ty1, tx2, ty2, label_x, label_y, anchor, angle) = match axis.orient {
            AxisOrient::Bottom => (
                tick.position, r.y, tick.position, r.y + theme.sizes.tick_size,
                tick.position, r.y + theme.sizes.tick_size + effective_font_size + label_pad,
                TextAnchor::Middle, tick.label_angle,
            ),
            AxisOrient::Top => (
                tick.position, r.y, tick.position, r.y - theme.sizes.tick_size,
                tick.position, r.y - theme.sizes.tick_size - label_pad - 2.0,
                TextAnchor::Middle, tick.label_angle,
            ),
            AxisOrient::Left => (
                r.x, tick.position, r.x - theme.sizes.tick_size, tick.position,
                r.x - theme.sizes.tick_size - label_pad, tick.position + effective_font_size / 3.0,
                TextAnchor::End, 0.0,
            ),
            AxisOrient::Right => (
                r.x, tick.position, r.x + theme.sizes.tick_size, tick.position,
                r.x + theme.sizes.tick_size + label_pad, tick.position + effective_font_size / 3.0,
                TextAnchor::Start, 0.0,
            ),
        };
        if axis.show_ticks {
            nodes.push(SceneNode::Line {
                x1: tx1,
                y1: ty1,
                x2: tx2,
                y2: ty2,
                style: tick_stroke.clone(),
            });
        }
        if axis.show_labels && !tick.culled {
            let lines: Vec<&str> = tick.label.split('\n').collect();
            if lines.len() == 1 || axis.orient != AxisOrient::Bottom {
                // Single-line label, or non-Bottom orient (no multi-line for those).
                nodes.push(SceneNode::Text {
                    x: label_x,
                    y: label_y,
                    content: tick.label.clone(),
                    style: to_scene_text_style(
                        theme.colors.label_color,
                        effective_font_size,
                        anchor,
                        angle,
                        &theme.typography.label_font_family,
                        label_fw,
                        None,
                        1.0,
                    ),
                });
            } else {
                // Multi-line label (Bottom orient only): emit one text node per line.
                let line_height = effective_font_size * 1.2;
                for (line_idx, line) in lines.iter().enumerate() {
                    let line_y = label_y + (line_idx as f64) * line_height;
                    nodes.push(SceneNode::Text {
                        x: label_x,
                        y: line_y,
                        content: line.to_string(),
                        style: to_scene_text_style(
                            theme.colors.label_color,
                            effective_font_size,
                            anchor,
                            angle,
                            &theme.typography.label_font_family,
                            label_fw,
                            None,
                            1.0,
                        ),
                    });
                }
            }
        }
    }

    if let Some(t) = &axis.title {
        let title_fw: Option<&str> = if theme.typography.title_font_weight == "normal" {
            None
        } else {
            Some(&theme.typography.title_font_weight)
        };
        let effective_title_color = axis
            .title_color_rgba
            .map(|[r, g, b, a]| palette::Srgba::new(r, g, b, a))
            .unwrap_or(theme.colors.title_color);
        let effective_title_font_size = axis.title_font_size.unwrap_or(theme.typography.title_font_size);
        nodes.push(SceneNode::Text {
            x: t.anchor_x,
            y: t.anchor_y,
            content: t.text.clone(),
            style: to_scene_text_style(
                effective_title_color,
                effective_title_font_size,
                TextAnchor::Middle,
                t.angle,
                &theme.typography.title_font_family,
                title_fw,
                None,
                1.0,
            ),
        });
    }

    nodes
}

pub fn build_grid(
    plot_area: Rect,
    x_axis: Option<&AxisLayout>,
    y_axis: Option<&AxisLayout>,
    theme: &ThemeInputs,
    band_colors: &[String],
) -> Vec<SceneNode> {
    if !theme.grid.grid {
        return Vec::new();
    }
    let mut nodes = Vec::new();
    // Major level — legacy single-level theme fields are the major level.
    let major_color = theme.colors.grid_color;
    let major_width = theme.sizes.grid_width;
    let major_dash: Option<&[f64]> = theme.grid.grid_dash.as_deref();
    let major_opacity = theme.grid.grid_opacity;
    // Minor level (Grid item 18). Emitted only when `theme.grid.minor` is on.
    let minor_enabled = theme.grid.minor;
    let minor_color = theme.colors.minor_grid_color;
    let minor_width = theme.sizes.minor_grid_width;
    let minor_dash: Option<&[f64]> = theme.grid.minor_grid_dash.as_deref();
    let minor_opacity = theme.grid.minor_grid_opacity;

    let y_baseline_x = y_axis.map(|a| a.axis_line.x).unwrap_or(plot_area.x);
    let x_baseline_y = x_axis
        .map(|a| a.axis_line.y)
        .unwrap_or(plot_area.y + plot_area.h);

    // Band fills: drawn before gridlines so lines appear on top.
    // Emit one rect per gap between consecutive y-tick positions.
    if !band_colors.is_empty() {
        if let Some(ay) = y_axis.filter(|a| a.show_grid) {
            let mut boundaries: Vec<f64> = Vec::with_capacity(ay.ticks.len() + 2);
            boundaries.push(plot_area.y);
            for tick in &ay.ticks {
                if (tick.position - x_baseline_y).abs() >= 0.5 {
                    boundaries.push(tick.position);
                }
            }
            boundaries.push(plot_area.y + plot_area.h);
            boundaries.dedup_by(|a, b| (*a - *b).abs() < 0.5);
            boundaries.sort_by(f64::total_cmp);

            for (band_idx, window) in boundaries.windows(2).enumerate() {
                let top = window[0];
                let bot = window[1];
                let band_color_str = &band_colors[band_idx % band_colors.len()];
                if band_color_str.eq_ignore_ascii_case("transparent") {
                    continue;
                }
                if let Ok(fill) = crate::render::color::from_hex_str(band_color_str) {
                    use crate::render::draw::to_scene_fill_stroke;
                    nodes.push(SceneNode::Rect {
                        x: plot_area.x,
                        y: top,
                        w: plot_area.w,
                        h: bot - top,
                        style: to_scene_fill_stroke(Some(fill), None, 0.0, 1.0, None),
                        corner_radius: 0.0,
                    });
                }
            }
        }
    }

    // Minor gridlines are drawn FIRST so the heavier major lines render on top
    // of them. Gated entirely on `theme.grid.minor`; when off, this block emits
    // nothing and the output below is byte-identical to the pre-minor renderer.
    if minor_enabled {
        if let Some(ax) = x_axis.filter(|a| a.show_grid) {
            for tick in &ax.minor_ticks {
                if (tick.position - y_baseline_x).abs() < 0.5 {
                    continue;
                }
                nodes.push(SceneNode::Line {
                    x1: tick.position,
                    y1: plot_area.y,
                    x2: tick.position,
                    y2: plot_area.y + plot_area.h,
                    style: to_scene_stroke(minor_color, minor_width, minor_opacity, minor_dash, None, None),
                });
            }
        }
        if let Some(ay) = y_axis.filter(|a| a.show_grid) {
            for tick in &ay.minor_ticks {
                if (tick.position - x_baseline_y).abs() < 0.5 {
                    continue;
                }
                nodes.push(SceneNode::Line {
                    x1: plot_area.x,
                    y1: tick.position,
                    x2: plot_area.x + plot_area.w,
                    y2: tick.position,
                    style: to_scene_stroke(minor_color, minor_width, minor_opacity, minor_dash, None, None),
                });
            }
        }
    }

    if let Some(ax) = x_axis.filter(|a| a.show_grid) {
        for tick in &ax.ticks {
            if (tick.position - y_baseline_x).abs() < 0.5 {
                continue;
            }
            nodes.push(SceneNode::Line {
                x1: tick.position,
                y1: plot_area.y,
                x2: tick.position,
                y2: plot_area.y + plot_area.h,
                style: to_scene_stroke(major_color, major_width, major_opacity, major_dash, None, None),
            });
        }
    }

    if let Some(ay) = y_axis.filter(|a| a.show_grid) {
        for tick in &ay.ticks {
            if (tick.position - x_baseline_y).abs() < 0.5 {
                continue;
            }
            nodes.push(SceneNode::Line {
                x1: plot_area.x,
                y1: tick.position,
                x2: plot_area.x + plot_area.w,
                y2: tick.position,
                style: to_scene_stroke(major_color, major_width, major_opacity, major_dash, None, None),
            });
        }
    }

    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{AxisLayout, AxisOrient, AxisTitleLayout, Rect, TickLayout};

    /// Construct a major `TickLayout` (`is_major = true`) for test fixtures.
    fn major_tick(position: f64, label: &str) -> TickLayout {
        TickLayout {
            position,
            label: label.into(),
            label_angle: 0.0,
            elided: false,
            culled: false,
            label_font_size: None,
            is_major: true,
        }
    }

    /// Construct a minor (unlabeled, `is_major = false`) `TickLayout`.
    fn minor_tick(position: f64) -> TickLayout {
        TickLayout {
            position,
            label: String::new(),
            label_angle: 0.0,
            elided: false,
            culled: false,
            label_font_size: None,
            is_major: false,
        }
    }

    /// A y-axis layout with three major ticks and two minor ticks between them.
    fn y_axis_with_minors() -> AxisLayout {
        AxisLayout {
            orient: AxisOrient::Left,
            panel_index: 0,
            axis_line: Rect { x: 50.0, y: 10.0, w: 1.0, h: 300.0 },
            ticks: vec![major_tick(60.0, "a"), major_tick(160.0, "b"), major_tick(260.0, "c")],
            minor_ticks: vec![minor_tick(110.0), minor_tick(210.0)],
            title: None,
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
            title_font_size: None,
            title_color_rgba: None,
            label_padding: None,
        }
    }

    #[test]
    fn build_grid_minor_disabled_emits_only_majors() {
        // Default theme has minor disabled → only major gridlines, byte-identical
        // to the pre-minor renderer (no minor lines emitted, major style unchanged).
        let y_axis = y_axis_with_minors();
        let plot_area = Rect { x: 50.0, y: 10.0, w: 400.0, h: 300.0 };
        let theme = ThemeInputs::default();
        assert!(!theme.grid.minor, "default theme must have minor off");
        let nodes = build_grid(plot_area, None, Some(&y_axis), &theme, &[]);

        let lines: Vec<&SceneNode> = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).collect();
        // 3 majors (none coincide with the x baseline since x_axis=None →
        // baseline at plot bottom y=310, far from 60/160/260). No minors.
        assert_eq!(lines.len(), 3, "minor off must emit only the 3 major lines");
        for n in &lines {
            if let SceneNode::Line { style, .. } = n {
                assert_eq!(style.width, theme.sizes.grid_width, "major width");
                assert_eq!(style.color.r, theme.colors.grid_color.red);
            }
        }
    }

    #[test]
    fn build_grid_minor_enabled_emits_minors_under_majors() {
        // Enable minor → 2 minor lines (minor style) followed by 3 major lines
        // (major style). Minors must precede majors in the node order (drawn under).
        let y_axis = y_axis_with_minors();
        let plot_area = Rect { x: 50.0, y: 10.0, w: 400.0, h: 300.0 };
        let mut theme = ThemeInputs::default();
        theme.grid.minor = true;
        let nodes = build_grid(plot_area, None, Some(&y_axis), &theme, &[]);

        let lines: Vec<&SceneNode> = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).collect();
        assert_eq!(lines.len(), 5, "2 minors + 3 majors");

        // First two lines are minors (minor style: thinner + fainter).
        for n in &lines[0..2] {
            if let SceneNode::Line { style, .. } = n {
                assert_eq!(style.width, theme.sizes.minor_grid_width, "minor width");
                assert!(style.width < theme.sizes.grid_width, "minor thinner than major");
                assert_eq!(style.color.r, theme.colors.minor_grid_color.red, "minor color");
            }
        }
        // Last three lines are majors.
        for n in &lines[2..5] {
            if let SceneNode::Line { style, .. } = n {
                assert_eq!(style.width, theme.sizes.grid_width, "major width");
                assert_eq!(style.color.r, theme.colors.grid_color.red, "major color");
            }
        }

        // Minor y-positions match the layout's minor_ticks (110, 210).
        let minor_ys: Vec<f64> = lines[0..2].iter().filter_map(|n| {
            if let SceneNode::Line { y1, .. } = n { Some(*y1) } else { None }
        }).collect();
        assert_eq!(minor_ys, vec![110.0, 210.0]);
    }

    #[test]
    fn build_grid_minor_enabled_no_minor_ticks_emits_only_majors() {
        // Categorical axis: minor enabled but minor_ticks empty (engine returns
        // none for categorical) → only majors emitted, no minor lines.
        let y_axis = AxisLayout {
            minor_ticks: vec![],
            ..y_axis_with_minors()
        };
        let plot_area = Rect { x: 50.0, y: 10.0, w: 400.0, h: 300.0 };
        let mut theme = ThemeInputs::default();
        theme.grid.minor = true;
        let nodes = build_grid(plot_area, None, Some(&y_axis), &theme, &[]);
        let line_count = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        assert_eq!(line_count, 3, "no minor_ticks → only the 3 majors even with minor enabled");
    }

    #[test]
    fn axis_builds_line_ticks_and_title() {
        let axis = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![major_tick(25.0, "0"), major_tick(75.0, "1")],
            minor_ticks: vec![],
            title: Some(AxisTitleLayout {
                text: "x".into(),
                anchor_x: 50.0,
                anchor_y: 95.0,
                angle: 0.0,
            }),
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
            title_font_size: None,
            title_color_rgba: None,
            label_padding: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme);
        // 1 domain line + 2 tick lines + 2 tick labels + 1 title = 6 nodes
        let line_count = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        let text_count = nodes.iter().filter(|n| matches!(n, SceneNode::Text { .. })).count();
        assert!(line_count >= 3, "expected >=3 lines (domain + ticks), got {line_count}");
        assert!(text_count >= 3, "expected >=3 texts (2 labels + title), got {text_count}");
    }

    #[test]
    fn culled_tick_skips_label() {
        let axis = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![
                major_tick(25.0, "visible"),
                TickLayout { position: 75.0, label: "culled".into(), label_angle: 0.0, elided: false, culled: true, label_font_size: None, is_major: true },
            ],
            minor_ticks: vec![],
            title: None,
            show_labels: true,
            show_ticks: true,
            show_domain: false,
            show_grid: false,
            title_font_size: None,
            title_color_rgba: None,
            label_padding: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme);

        // Both ticks should emit a tick mark line.
        let line_count = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        assert_eq!(line_count, 2, "expected 2 tick lines, got {line_count}");

        // Only the non-culled tick emits a text label.
        let texts: Vec<_> = nodes.iter().filter_map(|n| {
            if let SceneNode::Text { content, .. } = n { Some(content.as_str()) } else { None }
        }).collect();
        assert_eq!(texts, vec!["visible"], "culled tick must not emit a label; got {texts:?}");
    }

    #[test]
    fn multiline_label_emits_stacked_text() {
        let axis = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![
                TickLayout {
                    position: 50.0,
                    label: "trivial\nbaseline".into(),
                    label_angle: 0.0,
                    elided: false,
                    culled: false,
                    label_font_size: None,
                    is_major: true,
                },
            ],
            minor_ticks: vec![],
            title: None,
            show_labels: true,
            show_ticks: true,
            show_domain: false,
            show_grid: false,
            title_font_size: None,
            title_color_rgba: None,
            label_padding: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme);

        // One tick line + two text nodes (one per line).
        let text_contents: Vec<_> = nodes.iter().filter_map(|n| {
            if let SceneNode::Text { content, .. } = n { Some(content.as_str()) } else { None }
        }).collect();
        assert_eq!(
            text_contents, vec!["trivial", "baseline"],
            "multi-line label should emit one text node per line; got {text_contents:?}",
        );

        // The second line should be offset downward from the first.
        let text_ys: Vec<f64> = nodes.iter().filter_map(|n| {
            if let SceneNode::Text { y, .. } = n { Some(*y) } else { None }
        }).collect();
        assert_eq!(text_ys.len(), 2);
        assert!(
            text_ys[1] > text_ys[0],
            "second line y ({}) should be below first line y ({})",
            text_ys[1], text_ys[0],
        );
    }

    #[test]
    fn per_tick_font_size_override() {
        let axis = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![
                TickLayout {
                    position: 50.0,
                    label: "small".into(),
                    label_angle: 0.0,
                    elided: false,
                    culled: false,
                    label_font_size: Some(9.0),
                    is_major: true,
                },
            ],
            minor_ticks: vec![],
            title: None,
            show_labels: true,
            show_ticks: true,
            show_domain: false,
            show_grid: false,
            title_font_size: None,
            title_color_rgba: None,
            label_padding: None,
        };
        let theme = ThemeInputs::default(); // theme.typography.label_font_size == 11.0
        let nodes = build_axis(&axis, &theme);

        // The text node should use font_size 9.0, not the theme default.
        let text_node = nodes.iter().find(|n| matches!(n, SceneNode::Text { .. }));
        assert!(text_node.is_some(), "expected a text node");
        if let Some(SceneNode::Text { style, y, .. }) = text_node {
            assert_eq!(
                style.font_size, 9.0,
                "expected font_size 9.0 from per-tick override, got {}",
                style.font_size,
            );
            // label_y = r.y + tick_size + effective_font_size + label_pad
            // label_pad defaults to 2.0 when axis.label_padding is None.
            // With r.y=80, tick_size=4 (default), effective_font_size=9, label_pad=2: 80+4+9+2=95
            let expected_y = 80.0 + theme.sizes.tick_size + 9.0 + 2.0;
            assert!(
                (y - expected_y).abs() < 0.01,
                "label_y should use per-tick font size for positioning: expected {expected_y}, got {y}",
            );
        }
    }

    #[test]
    fn build_grid_band_colors_emits_rects() {
        // A y-axis with 3 ticks should produce 4 bands (before/between/after ticks).
        // With 2 band_colors, they alternate: fill, transparent, fill, transparent.
        let y_axis = AxisLayout {
            orient: AxisOrient::Left,
            panel_index: 0,
            axis_line: Rect { x: 50.0, y: 10.0, w: 1.0, h: 300.0 },
            ticks: vec![
                major_tick(60.0, "a"),
                major_tick(160.0, "b"),
                major_tick(260.0, "c"),
            ],
            minor_ticks: vec![],
            title: None,
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
            title_font_size: None,
            title_color_rgba: None,
            label_padding: None,
        };
        let plot_area = Rect { x: 50.0, y: 10.0, w: 400.0, h: 300.0 };
        let band_colors = vec!["#f0f0f0".to_string(), "transparent".to_string()];
        let theme = ThemeInputs::default();
        let nodes = build_grid(plot_area, None, Some(&y_axis), &theme, &band_colors);

        // Should emit Rect nodes for non-transparent bands only (every other band).
        let rect_count = nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();
        assert!(rect_count >= 1, "expected at least one band fill rect, got {rect_count}");

        // No rect should be emitted for "transparent" bands.
        // With 4 boundaries (plot_area.y=10, tick positions, plot_area.y+h=310):
        // bands: [10..60], [60..160], [160..260], [260..310]
        // colors cycling: #f0f0f0, transparent, #f0f0f0, transparent → 2 rects
        assert_eq!(rect_count, 2, "expected 2 rects (alternating with transparent), got {rect_count}");
    }

    #[test]
    fn build_grid_no_band_colors_emits_no_rects() {
        let y_axis = AxisLayout {
            orient: AxisOrient::Left,
            panel_index: 0,
            axis_line: Rect { x: 50.0, y: 10.0, w: 1.0, h: 300.0 },
            ticks: vec![major_tick(60.0, "a")],
            minor_ticks: vec![],
            title: None,
            show_labels: true, show_ticks: true, show_domain: true, show_grid: true,
            title_font_size: None, title_color_rgba: None, label_padding: None,
        };
        let plot_area = Rect { x: 50.0, y: 10.0, w: 400.0, h: 300.0 };
        let theme = ThemeInputs::default();
        let nodes = build_grid(plot_area, None, Some(&y_axis), &theme, &[]);
        let rect_count = nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();
        assert_eq!(rect_count, 0, "no band_colors means no rects");
    }

    #[test]
    fn build_axis_uses_per_axis_title_overrides() {
        // An AxisLayout with title_font_size=20 and title_color_rgba=#ff0000 should
        // render the title text with those values, not the theme defaults.
        let axis = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![],
            minor_ticks: vec![],
            title: Some(crate::layout::AxisTitleLayout {
                text: "My Title".into(),
                anchor_x: 50.0,
                anchor_y: 100.0,
                angle: 0.0,
            }),
            show_labels: true,
            show_ticks: true,
            show_domain: false,
            show_grid: false,
            title_font_size: Some(20.0),
            title_color_rgba: Some([0xff, 0x00, 0x00, 0xff]),
            label_padding: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme);
        let title_node = nodes.iter().find(|n| {
            if let SceneNode::Text { content, .. } = n { content == "My Title" } else { false }
        });
        assert!(title_node.is_some(), "expected a title text node");
        if let Some(SceneNode::Text { style, .. }) = title_node {
            assert_eq!(style.font_size, 20.0, "title should use overridden font size");
            assert_eq!(style.color.r, 0xff, "title should use overridden red color");
            assert_eq!(style.color.g, 0x00);
            assert_eq!(style.color.b, 0x00);
        }
    }
}
