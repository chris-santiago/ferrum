//! Internal: build axis and grid scene nodes from an AxisLayout.

use crate::layout::{AxisLayout, AxisOrient, Rect, TextAnchor, ThemeInputs};
use crate::render::draw::{to_scene_stroke, to_scene_text_style};
use ferrum_scene::SceneNode;

/// Build an axis's scene nodes (domain line, ticks, tick labels, title).
///
/// `tick_slot` (secondary-y-axis, GH #60/#73): when `Some(k)`, every emitted
/// tick-label `SceneNode::Text` (but not the axis title) is tagged with
/// `slot: Some(k)` so the WASM zoomed-text relabeling path can select its
/// rescale affine by explicit slot instead of the retired column-frequency
/// heuristic. Callers pass `Some(k)` only for a y-axis routed through the
/// dual-axis `Group` wrapper (mirrors that wrapper's `y_slot` attr — see
/// `scene_build::route_panel_axes_and_grid`); every other caller (x-axes, and
/// y-axes on the byte-stable single-slot default path) passes `None`, leaving
/// tick-label nodes untagged exactly as before this field existed.
pub fn build_axis(axis: &AxisLayout, theme: &ThemeInputs, tick_slot: Option<usize>) -> Vec<SceneNode> {
    let mut nodes = Vec::new();
    let r = axis.axis_line;

    // Per-axis style overrides (B5): fall back to the shared theme when unset, so
    // chart-level styling stays byte-identical and only a per-channel/per-axis
    // spec lights these up.
    let rgba = |arr: [u8; 4]| palette::Srgba::new(arr[0], arr[1], arr[2], arr[3]);
    let label_color = axis.label_color_rgba.map(rgba).unwrap_or(theme.colors.label_color);
    let axis_line_color =
        axis.domain_color_rgba.map(rgba).unwrap_or(theme.colors.axis_line_color);
    let axis_line_width = axis.domain_width.unwrap_or(theme.sizes.axis_line_width);
    let axis_label_font_size = axis.label_font_size.unwrap_or(theme.typography.label_font_size);

    // Domain line.
    if theme.axis.axis_line && axis.show_domain {
        nodes.push(SceneNode::Line {
            x1: r.x,
            y1: r.y,
            x2: r.x + r.w,
            y2: r.y + r.h,
            style: to_scene_stroke(axis_line_color, axis_line_width, 1.0, None, None, None),
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

    // `label_flush` (B5): flush the first/last *rendered* tick labels at the axis
    // ends so edge labels align within the plot bounds instead of overflowing.
    // `None`/`false` keeps every label at its default anchor (byte-identical).
    // Edge ticks are the first and last non-culled ticks in index order (ticks are
    // emitted in axis order). Single-tick axes have first == last; flush still
    // applies a coherent edge anchor. Only flat labels are flushed — rotated
    // labels already carry edge anchors via the rotation handling below.
    let flush = axis.label_flush.unwrap_or(false);
    let first_label_idx = flush
        .then(|| axis.ticks.iter().position(|t| !t.culled))
        .flatten();
    let last_label_idx = flush
        .then(|| axis.ticks.iter().rposition(|t| !t.culled))
        .flatten();

    for (tick_idx, tick) in axis.ticks.iter().enumerate() {
        // Per-tick font size wins; else the per-axis override; else the theme.
        let effective_font_size = tick.label_font_size.unwrap_or(axis_label_font_size);

        let (tx1, ty1, tx2, ty2, label_x, mut label_y, mut anchor, angle) = match axis.orient {
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

        // Angle-aware anchor/offset for horizontal axes: when labels are rotated the
        // SVG rotation pivots around (label_x, label_y) with `text-anchor` deciding
        // where the text sits relative to that pivot before rotation.  Middle-anchored
        // text pivots at its horizontal centre, so half the label swings into the plot.
        // Switch to an edge anchor and remove the font-size baseline drop (the rotation
        // itself carries the text away from the axis) for the rotated case.
        if tick.label_angle != 0.0 {
            match axis.orient {
                AxisOrient::Bottom => {
                    // End-anchor pivots from the right edge of the text.  The label's
                    // top corner sits sin(|angle|)*ascent above the pivot, so only
                    // remove the portion of the font-size baseline drop not "used up"
                    // by the rotation.  At -90 this subtracts 0 (full clearance kept);
                    // at -45 it subtracts ~0.29*font_size.
                    //
                    // SYNC: the band reservation for rotated bottom labels in
                    // `crate::layout::axis::estimate_x_label_band` mirrors this
                    // pivot geometry (pivot at
                    // `r.y + tick_size + label_pad + sin(|angle|)·font_size`, label
                    // dropping `sin·max_label_w + cos·extent` below it). The two
                    // must be changed together or the x-axis title overlaps the
                    // labels.
                    anchor = TextAnchor::End;
                    label_y -= effective_font_size * (1.0 - tick.label_angle.to_radians().sin().abs());
                }
                AxisOrient::Top => {
                    // Start-anchor: with negative angles the label hangs up-and-away
                    // from the plot rather than swinging downward into it.
                    anchor = TextAnchor::Start;
                }
                AxisOrient::Left | AxisOrient::Right => {}
            }
        }

        // `label_flush` edge alignment (B5). Applies only to flat labels at the
        // first/last rendered tick — rotated labels already carry edge anchors.
        if tick.label_angle == 0.0 && (Some(tick_idx) == first_label_idx || Some(tick_idx) == last_label_idx) {
            let is_first = Some(tick_idx) == first_label_idx;
            match axis.orient {
                // Horizontal axes: the leftmost (first) label anchors at its start
                // edge, the rightmost (last) at its end edge, so neither spills past
                // the plot bounds. A lone tick (first == last) is anchored Start.
                AxisOrient::Bottom | AxisOrient::Top => {
                    anchor = if is_first { TextAnchor::Start } else { TextAnchor::End };
                }
                // Vertical axes: the top (first) label baseline drops so its cap sits
                // within the plot top; the bottom (last) label baseline rises so its
                // descender sits within the plot bottom. The default baseline is
                // tick-centered (`+ font/3`); flush shifts it to a within-bounds
                // edge alignment.
                AxisOrient::Left | AxisOrient::Right => {
                    label_y = if is_first {
                        // Top edge: baseline below the tick by the cap height.
                        tick.position + effective_font_size
                    } else {
                        // Bottom edge: baseline at the tick (no descender past it).
                        tick.position
                    };
                }
            }
        }

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
                    slot: tick_slot,
                    style: to_scene_text_style(
                        label_color,
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
                        slot: tick_slot,
                        style: to_scene_text_style(
                            label_color,
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
            slot: None,
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

    // `translate` + `offset` (B5): shift the whole axis group
    // (line/ticks/labels/title) perpendicular to its line, outward positive. Both
    // are perpendicular shifts away from the plot edge and **compose additively**
    // (`translate` is Vega's axis translate, `offset` its axis offset). `None`/`0`
    // on both is a no-op so default output is byte-identical.
    let shift = axis.translate.unwrap_or(0.0) + axis.offset.unwrap_or(0.0);
    if shift != 0.0 {
        let (dx, dy) = match axis.orient {
            AxisOrient::Bottom => (0.0, shift),  // outward = down
            AxisOrient::Top => (0.0, -shift),    // outward = up
            AxisOrient::Left => (-shift, 0.0),   // outward = left
            AxisOrient::Right => (shift, 0.0),   // outward = right
        };
        for node in &mut nodes {
            offset_axis_node(node, dx, dy);
        }
    }

    nodes
}

/// Translate an axis scene node (`Line`/`Text`) by `(dx, dy)`. Only the node
/// kinds `build_axis` emits are handled; any other kind is left untouched.
fn offset_axis_node(node: &mut SceneNode, dx: f64, dy: f64) {
    match node {
        SceneNode::Line { x1, y1, x2, y2, .. } => {
            *x1 += dx;
            *y1 += dy;
            *x2 += dx;
            *y2 += dy;
        }
        SceneNode::Text { x, y, .. } => {
            *x += dx;
            *y += dy;
        }
        _ => {}
    }
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
            let style = to_scene_stroke(minor_color, minor_width, minor_opacity, minor_dash, None, None);
            emit_gridlines(&mut nodes, &ax.minor_ticks, true, plot_area, y_baseline_x, &style);
        }
        if let Some(ay) = y_axis.filter(|a| a.show_grid) {
            let style = to_scene_stroke(minor_color, minor_width, minor_opacity, minor_dash, None, None);
            emit_gridlines(&mut nodes, &ay.minor_ticks, false, plot_area, x_baseline_y, &style);
        }
    }

    // Major gridlines honor per-axis style overrides (B5): the x gridlines pick up
    // the x-axis's `grid_color`/`grid_width`/`grid_dash` overrides, the y gridlines
    // the y-axis's; each falls back to the shared theme major level when unset, so
    // chart-level gridline styling (which mutates the theme) stays byte-identical.
    if let Some(ax) = x_axis.filter(|a| a.show_grid) {
        let style = major_grid_style(ax, major_color, major_width, major_dash, major_opacity);
        emit_gridlines(&mut nodes, &ax.ticks, true, plot_area, y_baseline_x, &style);
    }

    if let Some(ay) = y_axis.filter(|a| a.show_grid) {
        let style = major_grid_style(ay, major_color, major_width, major_dash, major_opacity);
        emit_gridlines(&mut nodes, &ay.ticks, false, plot_area, x_baseline_y, &style);
    }

    nodes
}

/// Major-gridline stroke style for one axis, honoring the per-axis `grid_color`,
/// `grid_width`, `grid_dash`, and `grid_opacity` overrides and falling back to
/// the supplied theme major-level values when an override is absent (B5).
fn major_grid_style(
    axis: &AxisLayout,
    theme_color: palette::Srgba<u8>,
    theme_width: f64,
    theme_dash: Option<&[f64]>,
    theme_opacity: f64,
) -> ferrum_scene::StrokeStyle {
    let color = axis
        .grid_color_rgba
        .map(|[r, g, b, a]| palette::Srgba::new(r, g, b, a))
        .unwrap_or(theme_color);
    let width = axis.grid_width.unwrap_or(theme_width);
    let dash: Option<&[f64]> = axis.grid_dash.as_deref().or(theme_dash);
    let opacity = axis.grid_opacity.unwrap_or(theme_opacity);
    to_scene_stroke(color, width, opacity, dash, None, None)
}

/// Emit one `SceneNode::Line` per tick spanning the plot area, skipping any
/// tick that coincides (within 0.5px) with the axis baseline. Shared by all
/// four x/y × minor/major gridline levels.
///
/// `is_x_orient` selects geometry: x-orient ticks emit vertical lines at
/// `tick.position` across the plot height; y-orient ticks emit horizontal lines
/// at `tick.position` across the plot width. `baseline_coord` is the axis-line
/// coordinate to skip against (`y_baseline_x` for x, `x_baseline_y` for y).
fn emit_gridlines(
    nodes: &mut Vec<SceneNode>,
    ticks: &[crate::layout::TickLayout],
    is_x_orient: bool,
    plot_area: Rect,
    baseline_coord: f64,
    style: &ferrum_scene::StrokeStyle,
) {
    for tick in ticks {
        if (tick.position - baseline_coord).abs() < 0.5 {
            continue;
        }
        let line = if is_x_orient {
            SceneNode::Line {
                x1: tick.position,
                y1: plot_area.y,
                x2: tick.position,
                y2: plot_area.y + plot_area.h,
                style: style.clone(),
            }
        } else {
            SceneNode::Line {
                x1: plot_area.x,
                y1: tick.position,
                x2: plot_area.x + plot_area.w,
                y2: tick.position,
                style: style.clone(),
            }
        };
        nodes.push(line);
    }
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
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        }
    }

    // ── B5 unit 2: grid_opacity + translate render ──────────────────────────

    #[test]
    fn build_grid_honors_per_axis_grid_opacity() {
        // A y-axis with grid_opacity=0.3 must emit its gridlines at opacity 0.3,
        // overriding the theme grid opacity.
        let mut y_axis = y_axis_with_minors();
        y_axis.grid_opacity = Some(0.3);
        let plot_area = Rect { x: 50.0, y: 10.0, w: 400.0, h: 300.0 };
        let theme = ThemeInputs::default();
        let nodes = build_grid(plot_area, None, Some(&y_axis), &theme, &[]);
        let lines: Vec<&SceneNode> =
            nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).collect();
        assert!(!lines.is_empty());
        for n in &lines {
            if let SceneNode::Line { style, .. } = n {
                assert!((style.opacity - 0.3).abs() < 1e-9, "grid opacity override not applied: {}", style.opacity);
            }
        }
    }

    #[test]
    fn build_grid_default_grid_opacity_unchanged() {
        // Regression guard: no grid_opacity override → theme opacity preserved.
        let y_axis = y_axis_with_minors();
        let plot_area = Rect { x: 50.0, y: 10.0, w: 400.0, h: 300.0 };
        let theme = ThemeInputs::default();
        let nodes = build_grid(plot_area, None, Some(&y_axis), &theme, &[]);
        for n in &nodes {
            if let SceneNode::Line { style, .. } = n {
                assert!((style.opacity - theme.grid.grid_opacity).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn build_axis_translate_shifts_bottom_axis_downward() {
        // A Bottom axis with translate=10 shifts every node down by 10px (outward).
        let base = bottom_axis_with_angle(0.0);
        let mut shifted = base.clone();
        shifted.translate = Some(10.0);
        let theme = ThemeInputs::default();
        let base_nodes = build_axis(&base, &theme, None);
        let shifted_nodes = build_axis(&shifted, &theme, None);
        assert_eq!(base_nodes.len(), shifted_nodes.len());
        // Compare each Line's y by 10px shift.
        for (b, s) in base_nodes.iter().zip(shifted_nodes.iter()) {
            if let (SceneNode::Line { y1: by1, .. }, SceneNode::Line { y1: sy1, .. }) = (b, s) {
                assert!((sy1 - (by1 + 10.0)).abs() < 1e-9, "translate must shift line down 10px");
            }
            if let (SceneNode::Text { y: by, x: bx, .. }, SceneNode::Text { y: sy, x: sx, .. }) = (b, s) {
                assert!((sy - (by + 10.0)).abs() < 1e-9, "translate must shift text down 10px");
                assert!((sx - bx).abs() < 1e-9, "bottom translate must not move x");
            }
        }
    }

    #[test]
    fn build_axis_translate_shifts_left_axis_leftward() {
        let mut base = bottom_axis_with_angle(0.0);
        base.orient = AxisOrient::Left;
        let mut shifted = base.clone();
        shifted.translate = Some(8.0);
        let theme = ThemeInputs::default();
        let base_nodes = build_axis(&base, &theme, None);
        let shifted_nodes = build_axis(&shifted, &theme, None);
        for (b, s) in base_nodes.iter().zip(shifted_nodes.iter()) {
            if let (SceneNode::Line { x1: bx1, .. }, SceneNode::Line { x1: sx1, .. }) = (b, s) {
                assert!((sx1 - (bx1 - 8.0)).abs() < 1e-9, "left translate must shift line left 8px");
            }
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
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme, None);
        // 1 domain line + 2 tick lines + 2 tick labels + 1 title = 6 nodes
        let line_count = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        let text_count = nodes.iter().filter(|n| matches!(n, SceneNode::Text { .. })).count();
        assert!(line_count >= 3, "expected >=3 lines (domain + ticks), got {line_count}");
        assert!(text_count >= 3, "expected >=3 texts (2 labels + title), got {text_count}");
    }

    /// GH #60/#73: `tick_slot: Some(k)` tags every tick-label text node with
    /// `slot: Some(k)` but leaves the axis title untagged (`slot: None`) —
    /// the title is not axis-tick text and must stay excluded from the wire
    /// contract this field serves (WASM slot-aware relabeling selects tick
    /// labels, never titles, by slot).
    #[test]
    fn tick_slot_tags_tick_labels_but_not_title() {
        let axis = AxisLayout {
            orient: AxisOrient::Right,
            panel_index: 0,
            axis_line: Rect { x: 400.0, y: 10.0, w: 0.0, h: 300.0 },
            ticks: vec![major_tick(50.0, "0"), major_tick(150.0, "1")],
            minor_ticks: vec![],
            title: Some(AxisTitleLayout {
                text: "right y".into(),
                anchor_x: 420.0,
                anchor_y: 160.0,
                angle: -90.0,
            }),
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
            title_font_size: None,
            title_color_rgba: None,
            label_padding: None,
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme, Some(2));
        let texts: Vec<&SceneNode> = nodes.iter().filter(|n| matches!(n, SceneNode::Text { .. })).collect();
        assert_eq!(texts.len(), 3, "2 tick labels + 1 title");
        // Tick labels (first two Text nodes, emitted before the title) carry
        // the tick_slot tag.
        for t in &texts[..2] {
            if let SceneNode::Text { content, slot, .. } = t {
                assert_eq!(*slot, Some(2), "tick label '{content}' must carry the tick_slot tag");
            }
        }
        // The title (last Text node) stays untagged.
        if let SceneNode::Text { content, slot, .. } = texts[2] {
            assert_eq!(content, "right y");
            assert_eq!(*slot, None, "axis title must never carry a slot tag");
        } else {
            panic!("expected title Text node");
        }
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
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme, None);

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
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme, None);

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
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        };
        let theme = ThemeInputs::default(); // theme.typography.label_font_size == 11.0
        let nodes = build_axis(&axis, &theme, None);

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
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
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
            label_color_rgba: None, label_font_size: None,
            grid_color_rgba: None, grid_dash: None, grid_width: None,
            domain_color_rgba: None, domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        };
        let plot_area = Rect { x: 50.0, y: 10.0, w: 400.0, h: 300.0 };
        let theme = ThemeInputs::default();
        let nodes = build_grid(plot_area, None, Some(&y_axis), &theme, &[]);
        let rect_count = nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();
        assert_eq!(rect_count, 0, "no band_colors means no rects");
    }

    // -----------------------------------------------------------------------
    // Rotated tick-label anchor / y-offset tests (the rendering-defaults fix).
    // -----------------------------------------------------------------------

    /// Helper: build a bottom AxisLayout with the given label_angle on every tick.
    fn bottom_axis_with_angle(label_angle: f64) -> AxisLayout {
        AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![
                TickLayout { position: 25.0, label: "a".into(), label_angle, elided: false, culled: false, label_font_size: None, is_major: true },
                TickLayout { position: 75.0, label: "b".into(), label_angle, elided: false, culled: false, label_font_size: None, is_major: true },
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
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        }
    }

    /// Helper: build a top AxisLayout with the given label_angle on every tick.
    fn top_axis_with_angle(label_angle: f64) -> AxisLayout {
        AxisLayout {
            orient: AxisOrient::Top,
            ..bottom_axis_with_angle(label_angle)
        }
    }

    /// Collect (anchor, angle) from every Text node in a node list.
    /// Returns `ferrum_scene::TextAnchor` — the type stored in `SceneNode::Text.style`.
    fn text_anchors_and_angles(nodes: &[SceneNode]) -> Vec<(ferrum_scene::TextAnchor, f64)> {
        nodes.iter().filter_map(|n| {
            if let SceneNode::Text { style, .. } = n {
                Some((style.anchor, style.angle))
            } else {
                None
            }
        }).collect()
    }

    #[test]
    fn bottom_rotated_tick_labels_use_end_anchor() {
        // Bottom axis with label_angle = -45: every tick-label Text node must
        // have anchor == End and the stored angle must equal -45.
        let axis = bottom_axis_with_angle(-45.0);
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme, None);
        let texts = text_anchors_and_angles(&nodes);
        assert_eq!(texts.len(), 2, "expected 2 tick-label text nodes");
        for (anchor, angle) in &texts {
            assert_eq!(*anchor, ferrum_scene::TextAnchor::End, "rotated bottom tick must use End anchor");
            assert!((angle - (-45.0_f64)).abs() < 0.001, "angle must be -45, got {angle}");
        }
    }

    #[test]
    fn bottom_flat_tick_labels_use_middle_anchor() {
        // Regression guard: the common flat (angle == 0) case must keep Middle.
        let axis = bottom_axis_with_angle(0.0);
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme, None);
        let texts = text_anchors_and_angles(&nodes);
        assert_eq!(texts.len(), 2, "expected 2 tick-label text nodes");
        for (anchor, _) in &texts {
            assert_eq!(*anchor, ferrum_scene::TextAnchor::Middle, "flat bottom tick must keep Middle anchor");
        }
    }

    #[test]
    fn bottom_rotated_label_y_angle_aware_geometry() {
        // Validates the angle-aware vertical gap formula:
        //   label_y = base - effective_font_size * (1 - sin(|angle|))
        // where base = r.y + tick_size + effective_font_size + label_pad.
        //
        // With r.y=80, tick_size=4 (default), font_size=11 (default), label_pad=2:
        //   flat (0°):  80 + 4 + 11 + 2             = 97.0
        //   -90°:  base - 11*(1-sin(90°)) = 97 - 0  = 97.0  (full clearance kept)
        //   -45°:  base - 11*(1-sin(45°)) ≈ 97 - 3.22 ≈ 93.78 (partial subtraction)
        let theme = ThemeInputs::default();
        let font_size = theme.typography.label_font_size; // 11.0

        let label_y_at = |angle: f64| -> f64 {
            build_axis(&bottom_axis_with_angle(angle), &theme, None)
                .into_iter()
                .find_map(|n| if let SceneNode::Text { y, .. } = n { Some(y) } else { None })
                .expect("axis must have a text node")
        };

        let flat_y = label_y_at(0.0);
        let y_neg90 = label_y_at(-90.0);
        let y_neg60 = label_y_at(-60.0);
        let y_neg45 = label_y_at(-45.0);
        let y_neg30 = label_y_at(-30.0);

        // 1. At -90° the rotated label_y equals the flat label_y (sin(90°)=1 → subtract 0).
        assert!(
            (y_neg90 - flat_y).abs() < 0.01,
            "at -90° label_y ({y_neg90}) must equal flat label_y ({flat_y})"
        );

        // 2. At -45° the rotated label_y is strictly between old too-tight value and flat.
        //    Old value: flat_y - font_size (dropped the full term).
        //    New value: flat_y - font_size*(1-sin(45°))  — greater than old, less than flat.
        let old_tight = flat_y - font_size;
        assert!(
            y_neg45 > old_tight && y_neg45 < flat_y,
            "at -45° label_y ({y_neg45}) must be between old tight ({old_tight}) and flat ({flat_y})"
        );

        // 3. Monotonic: steeper angle → larger sin → smaller subtraction → larger label_y.
        //    y(-30) < y(-45) < y(-60) < y(-90)
        assert!(
            y_neg30 < y_neg45,
            "label_y(-30°)={y_neg30} must be less than label_y(-45°)={y_neg45}"
        );
        assert!(
            y_neg45 < y_neg60,
            "label_y(-45°)={y_neg45} must be less than label_y(-60°)={y_neg60}"
        );
        assert!(
            y_neg60 < y_neg90,
            "label_y(-60°)={y_neg60} must be less than label_y(-90°)={y_neg90}"
        );
    }

    #[test]
    fn top_rotated_tick_labels_use_start_anchor() {
        // Top axis with label_angle = -45: every tick-label must use Start anchor.
        let axis = top_axis_with_angle(-45.0);
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme, None);
        let texts = text_anchors_and_angles(&nodes);
        assert_eq!(texts.len(), 2, "expected 2 tick-label text nodes");
        for (anchor, _) in &texts {
            assert_eq!(*anchor, ferrum_scene::TextAnchor::Start, "rotated top tick must use Start anchor");
        }
    }

    #[test]
    fn top_flat_tick_labels_use_middle_anchor() {
        // Regression guard: flat top ticks must keep Middle.
        let axis = top_axis_with_angle(0.0);
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme, None);
        let texts = text_anchors_and_angles(&nodes);
        assert_eq!(texts.len(), 2, "expected 2 tick-label text nodes");
        for (anchor, _) in &texts {
            assert_eq!(*anchor, ferrum_scene::TextAnchor::Middle, "flat top tick must keep Middle anchor");
        }
    }

    #[test]
    fn axis_title_anchor_unaffected_by_rotated_ticks() {
        // A bottom axis with rotated tick labels AND a title: the title Text node
        // must still use Middle anchor regardless of tick rotation.
        let mut axis = bottom_axis_with_angle(-45.0);
        axis.title = Some(crate::layout::AxisTitleLayout {
            text: "My Axis Title".into(),
            anchor_x: 50.0,
            anchor_y: 110.0,
            angle: 0.0,
        });
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme, None);

        let title_node = nodes.iter().find(|n| {
            if let SceneNode::Text { content, .. } = n { content == "My Axis Title" } else { false }
        }).expect("title text node must be present");

        if let SceneNode::Text { style, .. } = title_node {
            assert_eq!(
                style.anchor, ferrum_scene::TextAnchor::Middle,
                "axis title anchor must remain Middle even when tick labels are rotated"
            );
        }
    }

    #[test]
    fn left_and_right_tick_labels_unchanged() {
        // Left ticks → End anchor, angle 0.  Right ticks → Start anchor, angle 0.
        let left_axis = AxisLayout {
            orient: AxisOrient::Left,
            ..bottom_axis_with_angle(0.0)
        };
        let right_axis = AxisLayout {
            orient: AxisOrient::Right,
            ..bottom_axis_with_angle(0.0)
        };
        let theme = ThemeInputs::default();

        let left_texts  = text_anchors_and_angles(&build_axis(&left_axis,  &theme, None));
        let right_texts = text_anchors_and_angles(&build_axis(&right_axis, &theme, None));

        assert_eq!(left_texts.len(),  2, "expected 2 left tick-label nodes");
        assert_eq!(right_texts.len(), 2, "expected 2 right tick-label nodes");

        for (anchor, angle) in &left_texts {
            assert_eq!(*anchor, ferrum_scene::TextAnchor::End,   "left tick must use End anchor");
            assert!((angle).abs() < 0.001,                        "left tick angle must be 0");
        }
        for (anchor, angle) in &right_texts {
            assert_eq!(*anchor, ferrum_scene::TextAnchor::Start, "right tick must use Start anchor");
            assert!((angle).abs() < 0.001,                        "right tick angle must be 0");
        }
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
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme, None);
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

    // ── B5 unit 6b: offset (additive with translate) + label_flush ──────────

    #[test]
    fn build_axis_offset_shifts_bottom_axis_downward() {
        // A Bottom axis with offset=30 shifts every node down 30px (outward),
        // exactly like translate.
        let base = bottom_axis_with_angle(0.0);
        let mut shifted = base.clone();
        shifted.offset = Some(30.0);
        let theme = ThemeInputs::default();
        let base_nodes = build_axis(&base, &theme, None);
        let shifted_nodes = build_axis(&shifted, &theme, None);
        for (b, s) in base_nodes.iter().zip(shifted_nodes.iter()) {
            if let (SceneNode::Line { y1: by1, .. }, SceneNode::Line { y1: sy1, .. }) = (b, s) {
                assert!((sy1 - (by1 + 30.0)).abs() < 1e-9, "offset must shift line down 30px");
            }
        }
    }

    #[test]
    fn build_axis_offset_composes_additively_with_translate() {
        // offset=10 + translate=10 → a single combined 20px outward shift.
        let base = bottom_axis_with_angle(0.0);
        let mut both = base.clone();
        both.offset = Some(10.0);
        both.translate = Some(10.0);
        let theme = ThemeInputs::default();
        let base_nodes = build_axis(&base, &theme, None);
        let both_nodes = build_axis(&both, &theme, None);
        for (b, s) in base_nodes.iter().zip(both_nodes.iter()) {
            if let (SceneNode::Line { y1: by1, .. }, SceneNode::Line { y1: sy1, .. }) = (b, s) {
                assert!(
                    (sy1 - (by1 + 20.0)).abs() < 1e-9,
                    "offset+translate must compose additively to a 20px shift"
                );
            }
        }
    }

    #[test]
    fn build_axis_offset_none_is_byte_identical() {
        // Regression guard: no offset/translate → unchanged geometry.
        let base = bottom_axis_with_angle(0.0);
        let theme = ThemeInputs::default();
        let a = build_axis(&base, &theme, None);
        let b = build_axis(&base, &theme, None);
        assert_eq!(a, b);
    }

    /// A Bottom axis with three flat ticks (left/middle/right) for flush tests.
    fn three_tick_bottom_axis() -> AxisLayout {
        let mut axis = bottom_axis_with_angle(0.0);
        axis.ticks = vec![
            crate::layout::TickLayout {
                position: 10.0, label: "left".into(), label_angle: 0.0,
                elided: false, culled: false, label_font_size: None, is_major: true,
            },
            crate::layout::TickLayout {
                position: 50.0, label: "mid".into(), label_angle: 0.0,
                elided: false, culled: false, label_font_size: None, is_major: true,
            },
            crate::layout::TickLayout {
                position: 90.0, label: "right".into(), label_angle: 0.0,
                elided: false, culled: false, label_font_size: None, is_major: true,
            },
        ];
        axis
    }

    /// Collect (label content, anchor) for every tick-label Text node.
    fn label_anchors(nodes: &[SceneNode]) -> Vec<(String, ferrum_scene::TextAnchor)> {
        nodes.iter().filter_map(|n| {
            if let SceneNode::Text { content, style, .. } = n {
                Some((content.clone(), style.anchor))
            } else {
                None
            }
        }).collect()
    }

    #[test]
    fn build_axis_label_flush_anchors_edge_labels_on_bottom() {
        // label_flush=true: first (leftmost) label → Start, last (rightmost) →
        // End, interior → Middle.
        let mut axis = three_tick_bottom_axis();
        axis.label_flush = Some(true);
        let theme = ThemeInputs::default();
        let anchors = label_anchors(&build_axis(&axis, &theme, None));
        assert_eq!(anchors.len(), 3);
        assert_eq!(anchors[0], ("left".into(), ferrum_scene::TextAnchor::Start));
        assert_eq!(anchors[1], ("mid".into(), ferrum_scene::TextAnchor::Middle));
        assert_eq!(anchors[2], ("right".into(), ferrum_scene::TextAnchor::End));
    }

    #[test]
    fn build_axis_label_flush_off_keeps_all_middle() {
        // Regression guard: no flush → every bottom label stays Middle.
        let axis = three_tick_bottom_axis();
        let theme = ThemeInputs::default();
        let anchors = label_anchors(&build_axis(&axis, &theme, None));
        for (_, a) in &anchors {
            assert_eq!(*a, ferrum_scene::TextAnchor::Middle, "flush off keeps Middle");
        }
    }

    #[test]
    fn build_axis_label_flush_skips_culled_edges() {
        // With the leftmost tick culled, the first *rendered* label is the middle
        // tick, which then takes the Start edge anchor.
        let mut axis = three_tick_bottom_axis();
        axis.label_flush = Some(true);
        axis.ticks[0].culled = true;
        let theme = ThemeInputs::default();
        let anchors = label_anchors(&build_axis(&axis, &theme, None));
        // Culled tick emits no label; 2 labels remain: mid (now first→Start),
        // right (last→End).
        assert_eq!(anchors.len(), 2);
        assert_eq!(anchors[0], ("mid".into(), ferrum_scene::TextAnchor::Start));
        assert_eq!(anchors[1], ("right".into(), ferrum_scene::TextAnchor::End));
    }

    #[test]
    fn build_axis_label_flush_vertical_shifts_edge_baselines() {
        // On a Left axis, flush moves the first (top) and last (bottom) label
        // baselines off the tick-centered default to within-bounds edges.
        let mut axis = three_tick_bottom_axis();
        axis.orient = AxisOrient::Left;
        axis.label_flush = Some(true);
        let theme = ThemeInputs::default();
        let base = {
            let mut no_flush = axis.clone();
            no_flush.label_flush = None;
            build_axis(&no_flush, &theme, None)
        };
        let flushed = build_axis(&axis, &theme, None);
        let ys = |nodes: &[SceneNode]| -> Vec<f64> {
            nodes.iter().filter_map(|n| {
                if let SceneNode::Text { y, .. } = n { Some(*y) } else { None }
            }).collect()
        };
        let base_ys = ys(&base);
        let flush_ys = ys(&flushed);
        assert_eq!(base_ys.len(), 3);
        // Edge baselines move; the interior one is unchanged.
        assert!((flush_ys[0] - base_ys[0]).abs() > 1e-6, "top edge baseline must shift");
        assert!((flush_ys[1] - base_ys[1]).abs() < 1e-9, "interior baseline unchanged");
        assert!((flush_ys[2] - base_ys[2]).abs() > 1e-6, "bottom edge baseline must shift");
    }
}
