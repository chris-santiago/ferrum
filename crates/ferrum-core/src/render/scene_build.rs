use arrow::record_batch::RecordBatch;
use ferrum_scene::{
    BlendMode, CoordKind, InteractionConfig, MarkBatch, Panel, PanelTickLevels, SceneGraph,
    SceneNode, TickLevel,
};
use crate::spec::coord::to_scene_coord;

use crate::layout::{LayoutResult, ThemeInputs};
use crate::spec::chart::ChartSpec;

use super::arrow_cast::col_as_str;
use super::config::RenderConfig;
use super::draw::{self, to_scene_color, to_scene_text_style, DrawCtx};
use super::marks;
use super::prepare::PreparedInputs;
use super::{filter_batch_by_facet, position, scale_resolve, RenderError, RenderWarning, CLIP_ID_PREFIX};

pub fn build_scene(
    spec: &ChartSpec,
    prep: &PreparedInputs,
    layout: &LayoutResult,
    theme: &ThemeInputs,
    config: &RenderConfig,
    warnings: &mut Vec<RenderWarning>,
) -> Result<SceneGraph, RenderError> {
    let background = config.background.or(Some(theme.background_color));

    let mut title_nodes: Vec<SceneNode> = Vec::new();
    let mut legend_nodes: Vec<SceneNode> = Vec::new();

    // Chart title
    build_title(layout, spec, theme, &mut title_nodes);

    let mut panels: Vec<Panel> = Vec::new();
    let mut tick_levels: Vec<PanelTickLevels> = Vec::new();

    for (panel_idx, panel) in layout.panels.iter().enumerate() {
        if panel.plot_area.w <= 0.0 || panel.plot_area.h <= 0.0 {
            warnings.push(RenderWarning::EmptyPanel { panel_index: panel_idx });
            continue;
        }

        // Per-panel axes
        let panel_axes_layout: Vec<&crate::layout::AxisLayout> = layout
            .axes
            .iter()
            .filter(|a| a.panel_index == panel_idx)
            .collect();
        let panel_x_axis = panel_axes_layout
            .iter()
            .copied()
            .find(|a| matches!(a.orient,
                crate::layout::AxisOrient::Bottom | crate::layout::AxisOrient::Top));
        let panel_y_axis = panel_axes_layout
            .iter()
            .copied()
            .find(|a| matches!(a.orient,
                crate::layout::AxisOrient::Left | crate::layout::AxisOrient::Right));

        // Polar and Geo coordinates suppress Cartesian axes and gridlines.
        let suppress_axes = matches!(
            &spec.coord,
            Some(crate::spec::coord::CoordKind::Polar { .. })
            | Some(crate::spec::coord::CoordKind::Geo { .. })
        );

        let grid_nodes = if suppress_axes {
            Vec::new()
        } else {
            marks::axis::build_grid(panel.plot_area, panel_x_axis, panel_y_axis, theme)
        };

        // Axes
        let mut axes_nodes: Vec<SceneNode> = Vec::new();
        if !suppress_axes {
            for axis in &panel_axes_layout {
                axes_nodes.extend(marks::axis::build_axis(axis, theme));
            }
        }

        // Strip title — emitted as separate nodes in the panel, not a group
        let strip_title_nodes: Vec<SceneNode> = panel.strip_title.as_ref()
            .map(|strip| marks::strip_title::build_strip_title(strip, &panel.plot_area, theme))
            .unwrap_or_default();

        // Facet filter
        let panel_batch = if let Some(key) = &panel.facet_key {
            filter_batch_by_facet(prep.final_batch(), &key.field, &key.value)?
        } else {
            prep.final_batch().clone()
        };
        if panel_batch.num_rows() == 0 {
            continue;
        }

        // Layer batch resolution
        let layer_batches: Vec<RecordBatch> = prep
            .layers
            .iter()
            .map(|layer| match &layer.data_source {
                None => Ok(panel_batch.clone()),
                Some(name) => {
                    let src = prep.transform_outputs.get(name).expect(
                        "layer.data_source validated by prepare_render_inputs",
                    );
                    if let Some(key) = &panel.facet_key {
                        filter_batch_by_facet(src, &key.field, &key.value)
                    } else {
                        Ok(src.clone())
                    }
                }
            })
            .collect::<Result<Vec<_>, RenderError>>()?;

        // Encoding merge
        let mut merged_encoding = spec.encoding.clone();
        merged_encoding.overlay_from(&prep.layers[0].encoding);
        let rendering_spec_for_panel = ChartSpec {
            encoding: merged_encoding,
            ..spec.clone()
        };

        // Scale resolution
        let (scales, scale_warnings) = scale_resolve::resolve_scales_with_outputs(
            &rendering_spec_for_panel,
            &panel_batch,
            &prep.transform_outputs,
            (panel.plot_area.x, panel.plot_area.x + panel.plot_area.w),
            (panel.plot_area.y, panel.plot_area.y + panel.plot_area.h),
            theme,
        )?;
        warnings.extend(scale_warnings);

        tick_levels.push(build_tick_levels(&scales, panel_idx));

        // Polar axis: circular boundary + radial tick marks (replaces Cartesian axes)
        if matches!(&spec.coord, Some(crate::spec::coord::CoordKind::Polar { .. })) {
            let cx = panel.plot_area.x + panel.plot_area.w / 2.0;
            let cy = panel.plot_area.y + panel.plot_area.h / 2.0;
            let outer_r = polar_outer_radius(&panel.plot_area);
            axes_nodes.extend(build_polar_axes(cx, cy, outer_r, &scales, theme));
        }

        // Mark batches
        let mut mark_batches: Vec<MarkBatch> = Vec::new();

        for (li, layer) in prep.layers.iter().enumerate() {
            let layer_batch = &layer_batches[li];
            if layer_batch.num_rows() == 0 {
                continue;
            }

            // Position adjustment — always call apply_position; it is the
            // single authority for all adjustments (explicit layer.position
            // *and* encoding-level encoding.y.stack).  When neither is set
            // it returns a cheap reference-counted clone and is a no-op.
            let adjusted_owned = position::apply_position(
                layer_batch,
                layer.position.as_ref(),
                &scales,
                &layer.encoding,
            )?;
            let layer_batch: &RecordBatch = &adjusted_owned;

            // Synthetic ChartSpec for this layer
            let layer_spec = ChartSpec {
                mark: layer.mark,
                encoding: layer.encoding.clone(),
                ..spec.clone()
            };
            let mark_style = draw::resolve_mark_style(
                layer.mark_style.as_ref(), theme, &layer.mark,
            );
            let ctx = DrawCtx {
                spec: &layer_spec,
                panel,
                theme,
                scales: &scales,
                batch: layer_batch,
                mark_style: &mark_style,
            };

            let mut result = draw::dispatch_mark_build(&layer.mark, &ctx);

            // For CoordPolar, transform all mark nodes from Cartesian pixel
            // space to polar pixel space (arc marks handle their own transform).
            if matches!(&spec.coord, Some(crate::spec::coord::CoordKind::Polar { .. }))
                && !matches!(layer.mark, crate::spec::mark::Mark::Arc)
            {
                apply_polar_node_transform(&mut result.nodes, &panel.plot_area);
            }

            let keys = extract_keys(&layer.encoding, layer_batch, result.data_indices.as_deref());
            mark_batches.push(MarkBatch {
                kind: result.kind,
                nodes: result.nodes,
                data_indices: result.data_indices,
                tooltips: result.tooltips,
                hrefs: result.hrefs,
                descriptions: result.descriptions,
                keys,
                blend: layer.blend.unwrap_or(BlendMode::Normal),
                stroke_cap: mark_style.stroke_cap.as_deref().and_then(draw::parse_stroke_cap),
                stroke_join: mark_style.stroke_join.as_deref().and_then(draw::parse_stroke_join),
            });
        }

        let plot_area = ferrum_scene::Rect {
            x: panel.plot_area.x,
            y: panel.plot_area.y,
            w: panel.plot_area.w,
            h: panel.plot_area.h,
        };

        // Determine clip rect: clip=false expands to the full panel area.
        let panel_clip = match &spec.coord {
            Some(crate::spec::coord::CoordKind::Cartesian { clip: false, .. })
            | Some(crate::spec::coord::CoordKind::Fixed { clip: false, .. }) => {
                // Expand clip to the full viewport so marks render outside plot area.
                ferrum_scene::Rect {
                    x: 0.0,
                    y: 0.0,
                    w: layout.viewport.w,
                    h: layout.viewport.h,
                }
            }
            _ => plot_area,
        };

        // Convert spec-side CoordKind to scene-side CoordKind.
        // outer_radius_px defaults to half the smaller plot dimension for polar.
        let outer_radius_px = polar_outer_radius(&panel.plot_area);
        let scene_coord = spec.coord.as_ref()
            .map(|c| to_scene_coord(c, outer_radius_px))
            .unwrap_or(CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
            });

        // Inject computed axis domains into the scene coord so the JS zoom handler
        // can read the actual displayed domain even for auto-scaled charts.
        let scene_coord = match scene_coord {
            CoordKind::Cartesian { x_domain: None, y_domain: None, expand, clip } => {
                let x_dom = scales.x.data_domain();
                let y_dom = scales.y.data_domain();
                CoordKind::Cartesian { x_domain: x_dom, y_domain: y_dom, expand, clip }
            }
            CoordKind::Fixed { x_domain: None, y_domain: None, ratio, expand, clip } => {
                let x_dom = scales.x.data_domain();
                let y_dom = scales.y.data_domain();
                CoordKind::Fixed { x_domain: x_dom, y_domain: y_dom, ratio, expand, clip }
            }
            other => other,
        };

        panels.push(Panel {
            id: panel_idx,
            plot_area,
            clip: panel_clip,
            coord: scene_coord,
            grid: grid_nodes,
            marks: mark_batches,
            axes: axes_nodes,
            annotations: Vec::new(),
            strip_title: strip_title_nodes,
        });
    }

    // Legend
    build_legend_decorations(layout, spec, prep, theme, &mut legend_nodes)?;

    let interaction = InteractionConfig {
        zoom_enabled: !spec.selections.is_empty(),
        pan_enabled: !spec.selections.is_empty(),
        conditionals: spec.conditionals.clone(),
        linked_panels: Vec::new(),
        tick_levels,
    };

    Ok(SceneGraph {
        width: layout.viewport.w,
        height: layout.viewport.h,
        background: background.map(to_scene_color),
        title: title_nodes,
        panels,
        legend: legend_nodes,
        decorations: Vec::new(),
        selections: spec.selections.clone(),
        interaction,
        chart_description: spec.chart_description.clone(),
    })
}

fn build_title(
    layout: &LayoutResult,
    spec: &ChartSpec,
    theme: &ThemeInputs,
    out: &mut Vec<SceneNode>,
) {
    let Some(title) = &layout.chart_title else { return };
    let title_spec = spec.title.as_ref();
    let resolved_font_size = title_spec
        .and_then(|t| t.font_size)
        .unwrap_or(theme.title_font_size);
    let resolved_font_weight: String = title_spec
        .and_then(|t| t.font_weight.clone())
        .unwrap_or_else(|| theme.title_font_weight.clone());
    let resolved_color = title_spec
        .and_then(|t| t.color.as_deref())
        .and_then(|hex| super::color::from_hex_str(hex).ok())
        .unwrap_or(theme.title_color);
    let fw = if resolved_font_weight == "normal" { None } else { Some(resolved_font_weight.as_str()) };
    out.push(SceneNode::Text {
        x: title.x,
        y: title.y,
        content: title.text.clone(),
        style: to_scene_text_style(
            resolved_color, resolved_font_size, title.anchor, 0.0,
            &theme.title_font_family, fw, None, 1.0,
        ),
    });
    if let (Some(subtitle), Some(sy)) = (&title.subtitle, title.subtitle_y) {
        let resolved_sub_color = title_spec
            .and_then(|t| t.subtitle_color.as_deref())
            .and_then(|hex| super::color::from_hex_str(hex).ok())
            .unwrap_or(theme.font_color);
        let resolved_sub_font_size = title_spec
            .and_then(|t| t.subtitle_font_size)
            .unwrap_or(resolved_font_size * 0.85);
        out.push(SceneNode::Text {
            x: title.x,
            y: sy,
            content: subtitle.clone(),
            style: to_scene_text_style(
                resolved_sub_color, resolved_sub_font_size, title.anchor, 0.0,
                &theme.font_family, None, None, 1.0,
            ),
        });
    }
}

fn build_legend_decorations(
    layout: &LayoutResult,
    spec: &ChartSpec,
    prep: &PreparedInputs,
    theme: &ThemeInputs,
    out: &mut Vec<SceneNode>,
) -> Result<(), RenderError> {
    let Some(legend) = &layout.legend else { return Ok(()) };
    let rendering_spec_for_legend = ChartSpec {
        encoding: prep.layers[0].encoding.clone(),
        ..spec.clone()
    };
    let color_scale = if rendering_spec_for_legend.encoding.color.is_some() {
        let (gs, _) = scale_resolve::resolve_scales_with_outputs(
            &rendering_spec_for_legend,
            prep.final_batch(),
            &prep.transform_outputs,
            (0.0, 1.0),
            (0.0, 1.0),
            theme,
        )?;
        gs.color
    } else {
        None
    };
    out.extend(marks::legend::build_legend(legend, color_scale.as_ref(), theme));
    Ok(())
}

fn extract_keys(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    data_indices: Option<&[usize]>,
) -> Option<Vec<String>> {
    let key_enc = encoding.key.as_ref()?;
    let col = col_as_str(batch, &key_enc.field).ok()?;
    let indices = data_indices?;
    Some(
        indices
            .iter()
            .map(|&i| col.get(i).and_then(|v| v.clone()).unwrap_or_default())
            .collect(),
    )
}

pub fn build_tick_levels(
    scales: &scale_resolve::ResolvedScales,
    panel_idx: usize,
) -> PanelTickLevels {
    const ZOOM_BREAKPOINTS: &[(f64, f64, usize)] = &[
        (0.0, 0.5, 4),
        (0.5, 2.0, 8),
        (2.0, 4.0, 16),
        (4.0, 1e9, 32),
    ];

    let x_levels: Vec<TickLevel> = ZOOM_BREAKPOINTS
        .iter()
        .map(|&(min_z, max_z, count)| TickLevel {
            min_zoom: min_z,
            max_zoom: max_z,
            ticks: scales.x.tick_data(count),
        })
        .collect();

    let y_levels: Vec<TickLevel> = ZOOM_BREAKPOINTS
        .iter()
        .map(|&(min_z, max_z, count)| TickLevel {
            min_zoom: min_z,
            max_zoom: max_z,
            ticks: scales.y.tick_data(count),
        })
        .collect();

    PanelTickLevels {
        panel_id: panel_idx,
        x_levels,
        y_levels,
    }
}

/// Compute the outer radius for polar coordinates: half the smaller dimension.
fn polar_outer_radius(plot_area: &crate::layout::Rect) -> f64 {
    plot_area.w.min(plot_area.h) / 2.0
}

/// Transform SceneNodes from Cartesian pixel coordinates to polar pixel
/// coordinates. Used for all mark types under CoordPolar.
///
/// Cartesian interpretation: x → θ (0..2π mapped across plot width),
/// y → r (plot height mapped to outer_radius, y-inverted because SVG y grows down).
///
/// Covered node types:
/// - `Circle`: anchor point (cx, cy) is remapped.
/// - `Polyline`: every vertex is remapped.
/// - `Line`: both endpoints (x1,y1) and (x2,y2) are remapped.
/// - `Text`: anchor point (x, y) is remapped; content and style are unchanged.
/// - `Rect`: converted to a `Polygon` whose perimeter approximates the curved
///   arc the rect describes in polar space. Arc edges (constant-y) are sampled
///   with `RECT_ARC_SEGMENTS` points; radial edges (constant-x) use 2 points.
///   The Rect's `FillStroke` is preserved so fill colour is not lost.
fn apply_polar_node_transform(
    nodes: &mut Vec<SceneNode>,
    plot_area: &crate::layout::Rect,
) {
    use std::f64::consts::TAU;
    let plot_x = plot_area.x;
    let plot_y = plot_area.y;
    let plot_w = plot_area.w;
    let plot_h = plot_area.h;
    let center_x = plot_x + plot_w / 2.0;
    let center_y = plot_y + plot_h / 2.0;
    let outer_r = polar_outer_radius(plot_area);

    /// Number of sample points along each arc edge of a converted Rect.
    const RECT_ARC_SEGMENTS: usize = 12;

    /// Map a single Cartesian pixel point to polar pixel coordinates.
    #[inline]
    fn map_pt(px: f64, py: f64, plot_x: f64, plot_y: f64, plot_w: f64, plot_h: f64,
              center_x: f64, center_y: f64, outer_r: f64, tau: f64) -> (f64, f64) {
        let theta = (px - plot_x) / plot_w * tau;
        let r = (plot_y + plot_h - py) / plot_h * outer_r;
        (center_x + r * theta.sin(), center_y - r * theta.cos())
    }

    let mut replacements: Vec<(usize, SceneNode)> = Vec::new();

    for (idx, node) in nodes.iter_mut().enumerate() {
        match node {
            SceneNode::Circle { ref mut cx, ref mut cy, .. } => {
                let (nx, ny) = map_pt(*cx, *cy, plot_x, plot_y, plot_w, plot_h,
                                      center_x, center_y, outer_r, TAU);
                *cx = nx;
                *cy = ny;
            }
            SceneNode::Polyline { ref mut points, .. } => {
                for pt in points.iter_mut() {
                    let (nx, ny) = map_pt(pt.0, pt.1, plot_x, plot_y, plot_w, plot_h,
                                          center_x, center_y, outer_r, TAU);
                    pt.0 = nx;
                    pt.1 = ny;
                }
            }
            SceneNode::Line { ref mut x1, ref mut y1, ref mut x2, ref mut y2, .. } => {
                let (nx1, ny1) = map_pt(*x1, *y1, plot_x, plot_y, plot_w, plot_h,
                                        center_x, center_y, outer_r, TAU);
                let (nx2, ny2) = map_pt(*x2, *y2, plot_x, plot_y, plot_w, plot_h,
                                        center_x, center_y, outer_r, TAU);
                *x1 = nx1; *y1 = ny1;
                *x2 = nx2; *y2 = ny2;
            }
            SceneNode::Text { ref mut x, ref mut y, .. } => {
                let (nx, ny) = map_pt(*x, *y, plot_x, plot_y, plot_w, plot_h,
                                      center_x, center_y, outer_r, TAU);
                *x = nx;
                *y = ny;
            }
            SceneNode::Rect { x, y, w, h, style, .. } => {
                // Convert the Cartesian rect to a polar Polygon by sampling
                // its perimeter. Constant-y (arc) edges get RECT_ARC_SEGMENTS
                // points; constant-x (radial) edges get 2 points (straight).
                // Perimeter order: bottom-left → bottom-right (arc) →
                // top-right (radial) → top-left (arc, reversed) → close.
                let (rx, ry, rw, rh, fill_stroke) = (*x, *y, *w, *h, style.clone());
                let mut pts: Vec<[f64; 2]> = Vec::with_capacity(
                    2 * RECT_ARC_SEGMENTS + 2
                );
                // Bottom arc: y = ry + rh, x sweeps left → right.
                for i in 0..=RECT_ARC_SEGMENTS {
                    let t = i as f64 / RECT_ARC_SEGMENTS as f64;
                    let px = rx + t * rw;
                    let py = ry + rh;
                    let (nx, ny) = map_pt(px, py, plot_x, plot_y, plot_w, plot_h,
                                          center_x, center_y, outer_r, TAU);
                    pts.push([nx, ny]);
                }
                // Right radial edge: x = rx + rw, y sweeps bottom → top.
                let (nx, ny) = map_pt(rx + rw, ry, plot_x, plot_y, plot_w, plot_h,
                                      center_x, center_y, outer_r, TAU);
                pts.push([nx, ny]);
                // Top arc: y = ry, x sweeps right → left.
                for i in (0..=RECT_ARC_SEGMENTS).rev() {
                    let t = i as f64 / RECT_ARC_SEGMENTS as f64;
                    let px = rx + t * rw;
                    let py = ry;
                    let (nx, ny) = map_pt(px, py, plot_x, plot_y, plot_w, plot_h,
                                          center_x, center_y, outer_r, TAU);
                    pts.push([nx, ny]);
                }
                // Left radial edge closes back to start (polygon auto-closes).
                replacements.push((idx, SceneNode::Polygon { rings: vec![pts], style: fill_stroke }));
            }
            _ => {}
        }
    }

    // Apply Rect → Polygon replacements (done after the borrow-safe iteration above).
    for (idx, replacement) in replacements {
        nodes[idx] = replacement;
    }
}

/// Build polar axis nodes: a circular outer boundary and radial tick marks with labels.
fn build_polar_axes(
    cx: f64,
    cy: f64,
    outer_r: f64,
    scales: &scale_resolve::ResolvedScales,
    theme: &ThemeInputs,
) -> Vec<SceneNode> {
    use std::f64::consts::TAU;
    use ferrum_scene::PathCmd;

    let axis_color = draw::to_scene_color(theme.axis_line_color);
    let stroke = ferrum_scene::StrokeStyle {
        color: axis_color,
        width: theme.axis_line_width,
        dash: None,
        opacity: 1.0,
        stroke_opacity: 1.0,
        stroke_cap: None,
        stroke_join: None,
    };

    let mut nodes: Vec<SceneNode> = Vec::new();

    // Circular outer boundary (two 180° arcs to form a full circle).
    if outer_r > 0.0 {
        nodes.push(SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: cx - outer_r, y: cy },
                PathCmd::ArcTo { rx: outer_r, ry: outer_r, rotation: 0.0, large_arc: true,  sweep: true, x: cx + outer_r, y: cy },
                PathCmd::ArcTo { rx: outer_r, ry: outer_r, rotation: 0.0, large_arc: true,  sweep: true, x: cx - outer_r, y: cy },
            ],
            style: ferrum_scene::FillStroke { fill: None, stroke: Some(axis_color), stroke_width: theme.axis_line_width, opacity: 1.0, stroke_dash: None, stroke_opacity: 1.0, fill_opacity: 1.0, angle: 0.0 },
            closed: true,
        });
    }

    // Radial tick marks and labels at each x-axis tick position.
    let (px_lo, px_hi) = scales.x.pixel_range();
    let px_span = px_hi - px_lo;
    if px_span.abs() > 0.0 {
        let ticks = scales.x.tick_data(8);
        let tick_len = 5.0_f64;
        let label_pad = 10.0_f64;
        for tick in &ticks {
            let theta = (tick.pixel - px_lo) / px_span * TAU;
            // Radial line from center to outer_r + tick_len
            let x1 = cx + outer_r * theta.sin();
            let y1 = cy - outer_r * theta.cos();
            let x2 = cx + (outer_r + tick_len) * theta.sin();
            let y2 = cy - (outer_r + tick_len) * theta.cos();
            nodes.push(SceneNode::Line { x1, y1, x2, y2, style: stroke.clone() });

            // Label outside the tick
            let lx = cx + (outer_r + label_pad) * theta.sin();
            let ly = cy - (outer_r + label_pad) * theta.cos();
            nodes.push(SceneNode::Text {
                x: lx,
                y: ly,
                content: tick.label.clone(),
                style: draw::to_scene_text_style(
                    theme.label_color, theme.label_font_size,
                    crate::layout::TextAnchor::Middle, 0.0,
                    &theme.font_family, None, None, 1.0,
                ),
            });
        }
    }

    nodes
}
