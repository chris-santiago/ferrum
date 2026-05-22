//! Text-element JSON serialization for the WASM renderer overlay.
//!
//! The JavaScript rendering layer positions DOM text elements (axis labels,
//! tick labels, titles) from a JSON array emitted by the Rust side. This
//! module centralizes that serialization.

use crate::scene_load::TextElementData;

// ── Shared enum-to-str helpers ──────────────────────────────────────────────

fn font_weight_string(w: &ferrum_scene::FontWeight) -> String {
    match w {
        ferrum_scene::FontWeight::Normal => "normal".to_string(),
        ferrum_scene::FontWeight::Bold => "bold".to_string(),
        ferrum_scene::FontWeight::Custom(s) => s.clone(),
    }
}

fn text_anchor_str(a: &ferrum_scene::TextAnchor) -> &'static str {
    match a {
        ferrum_scene::TextAnchor::Start => "start",
        ferrum_scene::TextAnchor::Middle => "center",
        ferrum_scene::TextAnchor::End => "end",
    }
}

/// Owned variant that handles `Custom` baselines correctly.
fn text_baseline_string(b: &ferrum_scene::TextBaseline) -> String {
    match b {
        ferrum_scene::TextBaseline::Top => "top".to_string(),
        ferrum_scene::TextBaseline::Middle => "middle".to_string(),
        ferrum_scene::TextBaseline::Bottom => "bottom".to_string(),
        ferrum_scene::TextBaseline::Alphabetic => "alphabetic".to_string(),
        ferrum_scene::TextBaseline::Custom(s) => s.clone(),
    }
}

fn color_string(style: &ferrum_scene::TextStyle) -> String {
    format!(
        "rgba({},{},{},{})",
        style.color.r, style.color.g, style.color.b, style.opacity
    )
}

// ── Public serialization functions ──────────────────────────────────────────

/// Build the full text-element JSON array from scene data's text elements.
///
/// Called from `WasmRenderer::load_scene` and `reset_zoom` (wasm32-only paths).
#[cfg(target_arch = "wasm32")]
pub(crate) fn build_text_json(data: &crate::scene_load::SceneData) -> String {
    build_text_json_from(&data.text_elements)
}

/// Build the text-element JSON array from a slice of text elements.
pub(crate) fn build_text_json_from(all_text: &[TextElementData]) -> String {
    let elements: Vec<serde_json::Value> = all_text.iter().map(text_element_to_json).collect();
    serde_json::to_string(&elements).unwrap_or_else(|_| "[]".to_string())
}

/// Build text-element JSON for a zoomed panel.
///
/// Axis tick labels are identified by clustering text elements that share the
/// same y coordinate (x-axis row) or same x coordinate (y-axis column) and
/// whose content appears in the known tick-label set.  Each identified tick
/// label is repositioned by applying the affine zoom transform to its varying
/// coordinate (x for x-axis ticks, y for y-axis ticks).  All other text
/// (chart title, axis title, legend) is emitted at its original position.
///
/// Labels whose transformed position falls outside the panel's `plot_area`
/// are filtered out to prevent tick labels from extending past the plot
/// boundary during zoom.
pub(crate) fn build_zoomed_text_json(
    all_text: &[TextElementData],
    interaction: &ferrum_scene::InteractionConfig,
    panel_id: usize,
    transform: &crate::zoom_pan::Affine2,
    plot_area: Option<(f64, f64, f64, f64)>,
) -> String {
    use std::collections::{HashMap, HashSet};

    let Some(ptl) = interaction
        .tick_levels
        .iter()
        .find(|p| p.panel_id == panel_id)
    else {
        return build_text_json_from(all_text);
    };

    // Union of tick label strings across all zoom levels.
    let x_tick_labels: HashSet<&str> = ptl
        .x_levels
        .iter()
        .flat_map(|lvl| lvl.ticks.iter().map(|t| t.label.as_str()))
        .collect();
    let y_tick_labels: HashSet<&str> = ptl
        .y_levels
        .iter()
        .flat_map(|lvl| lvl.ticks.iter().map(|t| t.label.as_str()))
        .collect();

    // --- Identify x-axis tick row ------------------------------------------
    // All x-axis tick labels share the same y coordinate.  Find the most
    // common rounded-y among elements whose content is a known x-tick label.
    let mut x_y_freq: HashMap<i64, usize> = HashMap::new();
    for te in all_text
        .iter()
        .filter(|te| x_tick_labels.contains(te.content.as_str()))
    {
        *x_y_freq.entry((te.y * 10.0) as i64).or_insert(0) += 1;
    }
    let x_axis_y: Option<f64> = x_y_freq
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| k as f64 / 10.0);

    // --- Identify y-axis tick column ----------------------------------------
    // All y-axis tick labels share the same x coordinate.
    let mut y_x_freq: HashMap<i64, usize> = HashMap::new();
    for te in all_text
        .iter()
        .filter(|te| y_tick_labels.contains(te.content.as_str()))
    {
        *y_x_freq.entry((te.x * 10.0) as i64).or_insert(0) += 1;
    }
    let y_axis_x: Option<f64> = y_x_freq
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| k as f64 / 10.0);

    // 1 px tolerance covers label_font_size / 3.0 baseline offset and rounding.
    const COORD_TOL: f64 = 1.5;

    let is_x_tick = |te: &TextElementData| {
        x_tick_labels.contains(te.content.as_str())
            && x_axis_y
                .map(|ay| (te.y - ay).abs() < COORD_TOL)
                .unwrap_or(false)
    };
    let is_y_tick = |te: &TextElementData| {
        y_tick_labels.contains(te.content.as_str())
            && y_axis_x
                .map(|ax| (te.x - ax).abs() < COORD_TOL)
                .unwrap_or(false)
    };

    let mut elements: Vec<serde_json::Value> = Vec::new();
    for te in all_text {
        if is_x_tick(te) {
            // Apply sx + tx to the x coordinate; y stays at the axis level.
            let new_x = te.x * transform.sx + transform.tx;
            // Skip if outside plot area horizontally.
            if let Some((px, _py, pw, _ph)) = plot_area {
                if new_x < px || new_x > px + pw {
                    continue;
                }
            }
            elements.push(tick_label_json(
                new_x,
                te.y,
                &te.content,
                "center",
                Some(&te.style),
            ));
        } else if is_y_tick(te) {
            // Apply sy + ty to the y coordinate; x stays at the axis level.
            let new_y = te.y * transform.sy + transform.ty;
            // Skip if outside plot area vertically.
            if let Some((_px, py, _pw, ph)) = plot_area {
                if new_y < py || new_y > py + ph {
                    continue;
                }
            }
            elements.push(tick_label_json(
                te.x,
                new_y,
                &te.content,
                "end",
                Some(&te.style),
            ));
        } else {
            elements.push(text_element_to_json(te));
        }
    }

    serde_json::to_string(&elements).unwrap_or_else(|_| "[]".to_string())
}

/// Serialize a single text element to a JSON value.
pub(crate) fn text_element_to_json(t: &TextElementData) -> serde_json::Value {
    serde_json::json!({
        "x": t.x,
        "y": t.y,
        "content": t.content,
        "fontSize": t.style.font_size,
        "fontWeight": font_weight_string(&t.style.font_weight),
        "fontFamily": t.style.font_family,
        "anchor": text_anchor_str(&t.style.anchor),
        "baseline": text_baseline_string(&t.style.baseline),
        "angle": t.style.angle,
        "color": color_string(&t.style),
    })
}

/// Serialize a tick label to a JSON value, optionally inheriting style from a
/// `TextStyle`.  Falls back to sensible defaults when `style` is `None`.
pub(crate) fn tick_label_json(
    x: f64,
    y: f64,
    label: &str,
    anchor: &str,
    style: Option<&ferrum_scene::TextStyle>,
) -> serde_json::Value {
    let (font_size, font_weight, font_family, baseline, angle, color) = match style {
        Some(s) => (
            s.font_size,
            font_weight_string(&s.font_weight),
            s.font_family.clone(),
            text_baseline_string(&s.baseline),
            s.angle,
            color_string(s),
        ),
        None => (
            11.0,
            "normal".to_string(),
            "sans-serif".to_string(),
            "alphabetic".to_string(),
            0.0,
            "rgba(51,51,51,1)".to_string(),
        ),
    };
    serde_json::json!({
        "x": x, "y": y, "content": label,
        "fontSize": font_size, "fontWeight": font_weight,
        "fontFamily": font_family, "anchor": anchor,
        "baseline": baseline, "angle": angle, "color": color,
    })
}

/// Format a scene-graph `TooltipContent` to the same JSON structure as
/// `parse_tooltip_json`: `{"fields":[{"name":"x","value":"1.23"},…]}`.
///
/// Used by `get_tooltip` to serve tooltips from non-packed batches where
/// tooltip data lives in the scene graph rather than a binary sidecar.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn format_tooltip_content(tooltip: &ferrum_scene::TooltipContent) -> String {
    if tooltip.fields.is_empty() {
        return "{}".to_string();
    }
    let fields: Vec<serde_json::Value> = tooltip
        .fields
        .iter()
        .map(|f| {
            serde_json::json!({
                "name": f.name,
                "value": f.value,
            })
        })
        .collect();
    serde_json::json!({ "fields": fields }).to_string()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_scene::{Color, FontWeight, TextAnchor, TextBaseline, TextStyle};

    fn make_style() -> TextStyle {
        TextStyle {
            font_size: 11.0,
            font_weight: FontWeight::Normal,
            font_family: "sans-serif".to_string(),
            color: Color {
                r: 51,
                g: 51,
                b: 51,
                a: 255,
            },
            opacity: 1.0,
            anchor: TextAnchor::Middle,
            baseline: TextBaseline::Alphabetic,
            angle: 0.0,
        }
    }

    fn make_text(x: f64, y: f64, content: &str) -> TextElementData {
        TextElementData {
            x,
            y,
            content: content.to_string(),
            style: make_style(),
        }
    }

    // ── font_weight_string / text_anchor_str / text_baseline_string ────

    #[test]
    fn font_weight_string_normal_bold() {
        assert_eq!(font_weight_string(&FontWeight::Normal), "normal");
        assert_eq!(font_weight_string(&FontWeight::Bold), "bold");
    }

    #[test]
    fn font_weight_string_custom() {
        let s = font_weight_string(&FontWeight::Custom("600".into()));
        assert_eq!(s, "600");
    }

    #[test]
    fn text_anchor_str_variants() {
        assert_eq!(text_anchor_str(&TextAnchor::Start), "start");
        assert_eq!(text_anchor_str(&TextAnchor::Middle), "center");
        assert_eq!(text_anchor_str(&TextAnchor::End), "end");
    }

    #[test]
    fn text_baseline_string_variants() {
        assert_eq!(text_baseline_string(&TextBaseline::Top), "top");
        assert_eq!(text_baseline_string(&TextBaseline::Middle), "middle");
        assert_eq!(text_baseline_string(&TextBaseline::Bottom), "bottom");
        assert_eq!(
            text_baseline_string(&TextBaseline::Alphabetic),
            "alphabetic"
        );
        assert_eq!(
            text_baseline_string(&TextBaseline::Custom("hanging".into())),
            "hanging"
        );
    }

    // ── B7: zoomed tick labels clipped to plot area ─────────────────────

    #[test]
    fn test_zoomed_tick_labels_clipped_to_plot_area() {
        // Simulate a zoom that pushes some x-tick labels outside the plot area.
        //
        // Setup: plot_area is (50, 50, 400, 300) — x ranges from 50 to 450.
        // Two x-tick labels at x=100 and x=400, both at y=360 (below the plot area).
        // Zoom: sx=2.0, tx=-200.0
        //   - Label at x=100 transforms to: 100*2 - 200 = 0 (outside left edge at 50)
        //   - Label at x=400 transforms to: 400*2 - 200 = 600 (outside right edge at 450)
        //
        // The y-tick label at x=40, y=200 transforms y to: 200*1.0 + 0.0 = 200 (inside).

        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![
                        ferrum_scene::Tick {
                            value: 100.0,
                            label: "100".to_string(),
                            pixel: 100.0,
                        },
                        ferrum_scene::Tick {
                            value: 400.0,
                            label: "400".to_string(),
                            pixel: 400.0,
                        },
                    ],
                }],
                y_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 200.0,
                        label: "200".to_string(),
                        pixel: 200.0,
                    }],
                }],
            }],
        };

        let all_text = vec![
            make_text(100.0, 360.0, "100"), // x-tick label
            make_text(400.0, 360.0, "400"), // x-tick label
            make_text(40.0, 200.0, "200"),  // y-tick label
            make_text(250.0, 20.0, "Title"), // non-tick (title)
        ];

        let transform = crate::zoom_pan::Affine2 {
            sx: 2.0,
            sy: 1.0,
            tx: -200.0,
            ty: 0.0,
        };

        let plot_area = Some((50.0, 50.0, 400.0, 300.0));

        let json_str = build_zoomed_text_json(&all_text, &interaction, 0, &transform, plot_area);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("valid JSON");

        // The two x-tick labels should be clipped out (transformed to 0 and 600,
        // both outside plot_area x=[50, 450]).
        let contents: Vec<&str> = parsed
            .iter()
            .filter_map(|v| v["content"].as_str())
            .collect();

        assert!(
            !contents.contains(&"100"),
            "x-tick label '100' at transformed x=0 should be clipped (outside plot_area x=50..450)"
        );
        assert!(
            !contents.contains(&"400"),
            "x-tick label '400' at transformed x=600 should be clipped (outside plot_area x=50..450)"
        );

        // The y-tick label and title should still be present.
        assert!(
            contents.contains(&"200"),
            "y-tick label '200' at transformed y=200 should be kept (inside plot_area y=50..350)"
        );
        assert!(
            contents.contains(&"Title"),
            "non-tick text 'Title' should always be kept"
        );
    }

    #[test]
    fn test_zoomed_tick_labels_inside_plot_area_kept() {
        // All tick labels transform to positions inside the plot area.
        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![
                        ferrum_scene::Tick {
                            value: 100.0,
                            label: "A".to_string(),
                            pixel: 100.0,
                        },
                        ferrum_scene::Tick {
                            value: 200.0,
                            label: "B".to_string(),
                            pixel: 200.0,
                        },
                    ],
                }],
                y_levels: vec![],
            }],
        };

        let all_text = vec![
            make_text(100.0, 360.0, "A"), // x-tick
            make_text(200.0, 360.0, "B"), // x-tick
        ];

        // Identity-ish transform that keeps labels inside.
        let transform = crate::zoom_pan::Affine2 {
            sx: 1.0,
            sy: 1.0,
            tx: 0.0,
            ty: 0.0,
        };

        let plot_area = Some((0.0, 0.0, 500.0, 500.0));

        let json_str = build_zoomed_text_json(&all_text, &interaction, 0, &transform, plot_area);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("valid JSON");

        let contents: Vec<&str> = parsed
            .iter()
            .filter_map(|v| v["content"].as_str())
            .collect();

        assert!(contents.contains(&"A"), "label 'A' should be kept");
        assert!(contents.contains(&"B"), "label 'B' should be kept");
    }

    #[test]
    fn test_zoomed_text_no_plot_area_keeps_all() {
        // When plot_area is None, no clipping should occur.
        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 100.0,
                        label: "X".to_string(),
                        pixel: 100.0,
                    }],
                }],
                y_levels: vec![],
            }],
        };

        let all_text = vec![make_text(100.0, 360.0, "X")];

        // Transform that would push label far off-screen.
        let transform = crate::zoom_pan::Affine2 {
            sx: 10.0,
            sy: 1.0,
            tx: -5000.0,
            ty: 0.0,
        };

        let json_str =
            build_zoomed_text_json(&all_text, &interaction, 0, &transform, None);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("valid JSON");

        let contents: Vec<&str> = parsed
            .iter()
            .filter_map(|v| v["content"].as_str())
            .collect();

        assert!(
            contents.contains(&"X"),
            "without plot_area, no clipping should occur"
        );
    }
}
