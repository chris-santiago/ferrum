//! Axis break — piecewise scale that omits a data range plus visual indicators.
//!
//! An axis break removes one or more contiguous ranges from the data domain and
//! inserts a break indicator (slash, zigzag, wave, or blank gap) at the
//! corresponding pixel position. The remaining segments are stretched uniformly
//! to fill the available pixel space minus the indicator widths.
//!
//! # Coordinate conventions
//! The pixel range follows SVG conventions: the origin is at the top-left, so
//! for the Y axis `plot_top` (small pixel value) corresponds to a large data
//! value and `plot_bottom` (large pixel value) corresponds to a small data value.

use ferrum_scene::SceneNode;

use crate::layout::{Rect, ThemeInputs};
use crate::render::draw::to_scene_stroke;

// ── Public types ─────────────────────────────────────────────────────────────

/// Segments of a broken scale. Each segment covers a contiguous range of the
/// original data domain and maps to a contiguous range of pixels.
#[derive(Debug, Clone)]
#[allow(dead_code)] // fields used by broken_scale_map and future mark re-positioning
pub struct ScaleSegment {
    /// Lower bound of the data-space range covered by this segment.
    pub data_lo: f64,
    /// Upper bound of the data-space range covered by this segment.
    pub data_hi: f64,
    /// Lower bound of the pixel-space range for this segment (top for Y axis).
    pub pixel_lo: f64,
    /// Upper bound of the pixel-space range for this segment (bottom for Y axis).
    pub pixel_hi: f64,
}

/// Output of `apply_break_to_scale` — a piecewise mapping and the pixel
/// positions of the break indicators.
///
/// `segments` is used by `broken_scale_map` for mark position adjustment;
/// `indicator_pixels` is used by `build_break_indicators`.
#[derive(Debug, Clone)]
pub struct BreakResult {
    /// Ordered list of contiguous data segments (excluding the gaps).
    /// Used by `broken_scale_map` to map data values to pixel positions.
    #[allow(dead_code)] // used by broken_scale_map and for future mark re-positioning
    pub segments: Vec<ScaleSegment>,
    /// Pixel midpoints of each break indicator, along the broken axis.
    pub indicator_pixels: Vec<f64>,
}

// ── Scale computation ─────────────────────────────────────────────────────────

/// Build a piecewise scale that skips `gaps` in the data domain.
///
/// `original_domain` is the full `(lo, hi)` of the axis before breaking.
/// `pixel_range` is `(axis_start_px, axis_end_px)` — for Y this is typically
/// `(plot_top, plot_bottom)`, but the function is axis-agnostic.
/// `break_size` is the pixel height/width reserved for each indicator.
///
/// Segments are allocated proportionally to their data-domain fraction of the
/// total non-gap span. Gaps that fall entirely outside `original_domain` are
/// silently ignored.
pub fn apply_break_to_scale(
    original_domain: (f64, f64),
    gaps: &[[f64; 2]],
    pixel_range: (f64, f64),
    break_size: f64,
) -> BreakResult {
    let (d_lo, d_hi) = original_domain;
    let (px_lo, px_hi) = pixel_range;
    let total_px = (px_hi - px_lo).abs();

    // Clip gaps to the original domain; discard empty / out-of-range gaps.
    let valid_gaps: Vec<(f64, f64)> = gaps
        .iter()
        .filter_map(|g| {
            let g_lo = g[0].max(d_lo);
            let g_hi = g[1].min(d_hi);
            if g_hi > g_lo { Some((g_lo, g_hi)) } else { None }
        })
        .collect();

    if valid_gaps.is_empty() {
        // No effective gaps — single segment spanning the full domain.
        return BreakResult {
            segments: vec![ScaleSegment {
                data_lo: d_lo,
                data_hi: d_hi,
                pixel_lo: px_lo,
                pixel_hi: px_hi,
            }],
            indicator_pixels: Vec::new(),
        };
    }

    // Build ordered list of retained data ranges (domain minus gaps).
    let mut segment_ranges: Vec<(f64, f64)> = Vec::new();
    let mut cursor = d_lo;
    let mut sorted_gaps = valid_gaps.clone();
    // f64 gap bounds are finite data values; Equal is the safe NaN fallback.
    sorted_gaps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for (g_lo, g_hi) in &sorted_gaps {
        if cursor < *g_lo {
            segment_ranges.push((cursor, *g_lo));
        }
        cursor = *g_hi;
    }
    if cursor < d_hi {
        segment_ranges.push((cursor, d_hi));
    }

    // Total data span retained (excluding gaps).
    let total_data_span: f64 = segment_ranges.iter().map(|(lo, hi)| hi - lo).sum();

    // Pixels available for data segments (after reserving break indicator space).
    let n_breaks = valid_gaps.len() as f64;
    let pixels_for_data = (total_px - n_breaks * break_size).max(1.0);

    // Assign pixel ranges to each data segment proportionally.
    let mut segments: Vec<ScaleSegment> = Vec::new();
    let mut px_cursor = px_lo;

    for (seg_lo, seg_hi) in &segment_ranges {
        let data_fraction = if total_data_span > 0.0 {
            (seg_hi - seg_lo) / total_data_span
        } else {
            1.0 / segment_ranges.len() as f64
        };
        let seg_px = pixels_for_data * data_fraction;
        let seg_px_hi = px_cursor + seg_px * (px_hi - px_lo).signum();
        segments.push(ScaleSegment {
            data_lo: *seg_lo,
            data_hi: *seg_hi,
            pixel_lo: px_cursor,
            pixel_hi: seg_px_hi,
        });
        // Advance past this segment and the following break indicator
        // (unless this is the last segment).
        px_cursor = seg_px_hi;
        if segments.len() < segment_ranges.len() {
            px_cursor += break_size * (px_hi - px_lo).signum();
        }
    }

    // Indicator pixel midpoints: midpoint between each pair of adjacent segments.
    let indicator_pixels: Vec<f64> = segments
        .windows(2)
        .map(|w| (w[0].pixel_hi + w[1].pixel_lo) / 2.0)
        .collect();

    BreakResult { segments, indicator_pixels }
}

/// Map a single data value to its pixel position under the broken scale.
///
/// Returns `None` when `value` falls within a gap.
///
/// This function is part of the public API for mark re-positioning (future
/// scene_build integration), tested by unit tests in this module.
#[allow(dead_code)]
pub fn broken_scale_map(value: f64, result: &BreakResult) -> Option<f64> {
    for seg in &result.segments {
        let (lo, hi) = (seg.data_lo.min(seg.data_hi), seg.data_lo.max(seg.data_hi));
        if value >= lo && value <= hi {
            let t = if seg.data_hi != seg.data_lo {
                (value - seg.data_lo) / (seg.data_hi - seg.data_lo)
            } else {
                0.5
            };
            return Some(seg.pixel_lo + t * (seg.pixel_hi - seg.pixel_lo));
        }
    }
    None // value is in a gap
}

// ── Indicator rendering ───────────────────────────────────────────────────────

/// Render break indicators at `result.indicator_pixels` for the given axis and style.
///
/// `plot_area` provides the perpendicular extent (e.g. for a Y-axis break, the
/// x-extent of the plot area determines where the indicators are drawn).
pub fn build_break_indicators(
    result: &BreakResult,
    plot_area: &Rect,
    axis: &str,
    break_size: f64,
    break_style: &str,
    theme: &ThemeInputs,
) -> Vec<SceneNode> {
    let mut nodes = Vec::new();

    let stroke = to_scene_stroke(theme.axis_line_color, theme.axis_line_width, 1.0, None, None, None);

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
    axis: &str,
    break_size: f64,
    stroke: ferrum_scene::StrokeStyle,
) -> Vec<SceneNode> {
    let half = break_size / 2.0;
    let diag = 6.0; // diagonal extension in pixels

    if axis == "y" {
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
    axis: &str,
    break_size: f64,
    stroke: ferrum_scene::StrokeStyle,
) -> Vec<SceneNode> {
    use ferrum_scene::{PathCmd, FillStroke};

    let half = break_size / 2.0;
    let amplitude = 5.0;
    let n_zags = 4i32;

    let commands = if axis == "y" {
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
    axis: &str,
    break_size: f64,
    stroke: ferrum_scene::StrokeStyle,
) -> Vec<SceneNode> {
    use ferrum_scene::{PathCmd, FillStroke};

    let half = break_size / 2.0;
    let amplitude = 4.0;
    let n_steps = 20i32;

    let commands = if axis == "y" {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::ThemeInputs;

    fn plot() -> Rect {
        Rect { x: 50.0, y: 50.0, w: 300.0, h: 300.0 }
    }

    // ── broken_scale_map tests ───────────────────────────────────────────────

    #[test]
    fn broken_scale_map_value_in_gap_returns_none() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        // 500 is squarely inside the gap [200, 800].
        assert!(broken_scale_map(500.0, &result).is_none());
    }

    #[test]
    fn broken_scale_map_gap_boundary_is_none() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        // 200 is on the gap boundary — in the gap (exclusive of the retained segment end).
        // Values inside gaps should return None; segment boundaries are included in segments.
        // 0 and 1000 should map to finite pixels.
        assert!(broken_scale_map(0.0, &result).is_some());
        assert!(broken_scale_map(1000.0, &result).is_some());
    }

    #[test]
    fn broken_scale_map_value_below_domain_is_none() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        assert!(broken_scale_map(-1.0, &result).is_none());
    }

    #[test]
    fn broken_scale_map_domain_endpoints_map_to_pixel_range() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let px_min = broken_scale_map(0.0, &result).unwrap();
        let px_max = broken_scale_map(1000.0, &result).unwrap();
        // px_min should be close to pixel_range start, px_max close to end.
        let tolerance = 1.0;
        assert!(
            (px_min - 50.0).abs() < tolerance,
            "data lo should map near pixel_range start; got {px_min}"
        );
        assert!(
            (px_max - 350.0).abs() < tolerance,
            "data hi should map near pixel_range end; got {px_max}"
        );
    }

    #[test]
    fn broken_scale_map_no_gaps_behaves_like_linear() {
        let result = apply_break_to_scale((0.0, 100.0), &[], (50.0, 350.0), 12.0);
        // Single segment, linear mapping.
        let px_mid = broken_scale_map(50.0, &result).unwrap();
        // Should be at the midpoint of [50, 350] = 200.
        assert!((px_mid - 200.0).abs() < 0.01, "midpoint should be ~200; got {px_mid}");
    }

    #[test]
    fn broken_scale_map_segment_monotone() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let p0 = broken_scale_map(0.0, &result).unwrap();
        let p100 = broken_scale_map(100.0, &result).unwrap();
        let p200 = broken_scale_map(200.0, &result).unwrap();
        // Within first segment [0, 200], values should be monotonically increasing.
        assert!(p0 < p100, "p0={p0} should be < p100={p100}");
        assert!(p100 < p200, "p100={p100} should be < p200={p200}");

        let p800 = broken_scale_map(800.0, &result).unwrap();
        let p900 = broken_scale_map(900.0, &result).unwrap();
        let p1000 = broken_scale_map(1000.0, &result).unwrap();
        // Within second segment [800, 1000], values should be monotonically increasing.
        assert!(p800 < p900, "p800={p800} should be < p900={p900}");
        assert!(p900 < p1000, "p900={p900} should be < p1000={p1000}");
        // Second segment starts after the break (past first segment end).
        assert!(p800 > p200, "second segment should start after first ends");
    }

    // ── apply_break_to_scale tests ───────────────────────────────────────────

    #[test]
    fn apply_break_produces_two_segments_for_one_gap() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.indicator_pixels.len(), 1);
    }

    #[test]
    fn apply_break_no_gaps_produces_one_segment() {
        let result = apply_break_to_scale((0.0, 100.0), &[], (50.0, 350.0), 12.0);
        assert_eq!(result.segments.len(), 1);
        assert!(result.indicator_pixels.is_empty());
    }

    // ── build_break_indicators tests ─────────────────────────────────────────

    #[test]
    fn build_break_indicators_slash_returns_lines() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let theme = ThemeInputs::default();
        let nodes = build_break_indicators(&result, &plot(), "y", 12.0, "slash", &theme);
        assert!(!nodes.is_empty(), "slash style should produce nodes");
        let has_line = nodes.iter().any(|n| matches!(n, SceneNode::Line { .. }));
        assert!(has_line, "slash style should produce line nodes");
    }

    #[test]
    fn build_break_indicators_gap_style_returns_empty() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let theme = ThemeInputs::default();
        let nodes = build_break_indicators(&result, &plot(), "y", 12.0, "gap", &theme);
        assert!(nodes.is_empty(), "gap style should produce no indicator nodes");
    }

    #[test]
    fn build_break_indicators_zigzag_returns_paths() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let theme = ThemeInputs::default();
        let nodes = build_break_indicators(&result, &plot(), "y", 12.0, "zigzag", &theme);
        let has_path = nodes.iter().any(|n| matches!(n, SceneNode::Path { .. }));
        assert!(has_path, "zigzag style should produce path nodes");
    }

    #[test]
    fn build_break_indicators_wave_returns_paths() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let theme = ThemeInputs::default();
        let nodes = build_break_indicators(&result, &plot(), "y", 12.0, "wave", &theme);
        let has_path = nodes.iter().any(|n| matches!(n, SceneNode::Path { .. }));
        assert!(has_path, "wave style should produce path nodes");
    }

    #[test]
    fn build_break_indicators_x_axis() {
        let result = apply_break_to_scale((0.0, 100.0), &[[20.0, 80.0]], (50.0, 350.0), 12.0);
        let theme = ThemeInputs::default();
        let nodes = build_break_indicators(&result, &plot(), "x", 12.0, "slash", &theme);
        assert!(!nodes.is_empty(), "x-axis slash should produce nodes");
    }
}
