use arrow::record_batch::RecordBatch;
use ferrum_scene::{
    BlendMode, CoordKind, InteractionConfig, MarkBatch, Panel, PanelTickLevels, SceneGraph,
    SceneNode, TickLevel,
};
use crate::spec::coord::to_scene_coord;

use crate::layout::{LayoutResult, ThemeInputs};
use crate::spec::chart::ChartSpec;

use super::arrow_cast::col_as_str;
use super::chart_config::StructuralSpec;
use super::config::RenderConfig;
use super::draw::{self, to_scene_color, to_scene_text_style, DrawCtx};
use super::marks;
use super::prepare::PreparedInputs;
use super::{
    break_axis, inset, secondary_axis,
    filter_batch_by_facet, position, scale_resolve, RenderError, RenderWarning, CLIP_ID_PREFIX,
};

pub fn build_scene(
    spec: &ChartSpec,
    prep: &PreparedInputs,
    layout: &LayoutResult,
    theme: &ThemeInputs,
    config: &RenderConfig,
    warnings: &mut Vec<RenderWarning>,
    chart_config: &super::chart_config::ChartConfig,
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

        let grid_band_colors: &[String] = chart_config.grid
            .as_ref()
            .and_then(|g| g.band_colors.as_deref())
            .unwrap_or(&[]);
        let mut grid_nodes = if suppress_axes {
            Vec::new()
        } else {
            marks::axis::build_grid(panel.plot_area, panel_x_axis, panel_y_axis, theme, grid_band_colors)
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
        let (mut scales, scale_warnings) = scale_resolve::resolve_scales_with_outputs(
            &rendering_spec_for_panel,
            &panel_batch,
            &prep.transform_outputs,
            (panel.plot_area.x, panel.plot_area.x + panel.plot_area.w),
            (panel.plot_area.y, panel.plot_area.y + panel.plot_area.h),
            theme,
        )?;
        warnings.extend(scale_warnings);

        // Apply chart_config color overrides (level 3) to the per-panel color scale.
        // This must run after scale resolution because resolve_scales_with_outputs
        // independently re-resolves the color scale for each panel, discarding any
        // provisional_scales override that was applied in render_svg/render_scene_json.
        if let Some(ref cfg) = chart_config.color {
            super::apply_color_config_to_color_scale(&mut scales.color, cfg);
        }

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
                prep.coord_flipped,
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
                packed_instances: None,
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

        // Annotations: render user-specified annotations on the first panel only.
        let annotation_nodes = if panel_idx == 0 && !chart_config.annotations.is_empty() {
            let ann_ctx = super::annotation::ScaleContext {
                plot_area: panel.plot_area,
                x_scale: &scales.x,
                y_scale: &scales.y,
            };
            super::annotation::build_annotations(&chart_config.annotations, &ann_ctx)
        } else {
            Vec::new()
        };

        // Structural features: secondary Y axis, axis breaks, insets.
        // Only applied to the first panel (non-faceted charts).
        let (structural_axes, structural_marks, structural_annotations, break_results) =
            if panel_idx == 0 && !chart_config.structural.is_empty() {
                build_structural_nodes(
                    &chart_config.structural,
                    &panel_batch,
                    &scales,
                    &panel.plot_area,
                    theme,
                    &rendering_spec_for_panel,
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };

        // Remap mark, axis, and grid pixel coordinates through any broken scales
        // so that elements inside a gap are hidden and elements outside the gap
        // are repositioned to their compressed pixel positions.
        for (axis, break_result) in &break_results {
            let (data_domain, pixel_range) = if axis == "y" {
                (scales.y.data_domain(), scales.y.pixel_range())
            } else {
                (scales.x.data_domain(), scales.x.pixel_range())
            };
            if let Some((d_lo, d_hi)) = data_domain {
                let (px_lo, px_hi) = pixel_range;
                for batch in &mut mark_batches {
                    remap_mark_batch_through_break(
                        &mut batch.nodes, axis, d_lo, d_hi, px_lo, px_hi, break_result,
                    );
                }
                // Remap axis nodes whose coordinate falls within the
                // broken axis's pixel range.  Cross-axis elements (e.g.
                // x-axis labels below the plot for a y-break) have
                // coordinates outside this range and are left untouched.
                let (range_lo, range_hi) = (px_lo.min(px_hi), px_lo.max(px_hi));
                for node in axes_nodes.iter_mut() {
                    if node_coord_in_range(node, axis, range_lo, range_hi) {
                        remap_node(node, axis, d_lo, d_hi, px_lo, px_hi, break_result);
                    }
                }
                // Grid lines within the plot area are remapped so they
                // align with the compressed scale segments.
                for node in grid_nodes.iter_mut() {
                    remap_node(node, axis, d_lo, d_hi, px_lo, px_hi, break_result);
                }
            }
        }

        let final_axes: Vec<SceneNode> = {
            let mut v = axes_nodes;
            v.extend(structural_axes);
            v
        };
        let final_marks: Vec<ferrum_scene::MarkBatch> = {
            let mut v = mark_batches;
            v.extend(structural_marks);
            v
        };
        let final_annotations: Vec<SceneNode> = {
            let mut v = annotation_nodes;
            v.extend(structural_annotations);
            v
        };

        panels.push(Panel {
            id: panel_idx,
            plot_area,
            clip: panel_clip,
            coord: scene_coord,
            grid: grid_nodes,
            marks: final_marks,
            axes: final_axes,
            annotations: final_annotations,
            strip_title: strip_title_nodes,
        });
    }

    // Legend
    build_legend_decorations(layout, spec, prep, theme, chart_config, &mut legend_nodes)?;

    let interaction = InteractionConfig {
        zoom_enabled: !spec.selections.is_empty(),
        pan_enabled: !spec.selections.is_empty(),
        conditionals: spec.conditionals.clone(),
        linked_panels: Vec::new(),
        tick_levels,
        toolbar: true,
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
    chart_config: &super::chart_config::ChartConfig,
    out: &mut Vec<SceneNode>,
) -> Result<(), RenderError> {
    let Some(legend) = &layout.legend else { return Ok(()) };
    let rendering_spec_for_legend = ChartSpec {
        encoding: prep.layers[0].encoding.clone(),
        ..spec.clone()
    };
    let mut color_scale = if rendering_spec_for_legend.encoding.color.is_some() {
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
    if let Some(ref cfg) = chart_config.color {
        super::apply_color_config_to_color_scale(&mut color_scale, cfg);
    }
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
    #[allow(clippy::too_many_arguments)]
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
            SceneNode::Path { ref mut commands, .. } => {
                // Transform each PathCmd endpoint and control point through the
                // polar projection. HLineTo/VLineTo change both axes under polar
                // (x offset → angle, y → radius) so they are converted to LineTo.
                // The current y for HLineTo / current x for VLineTo is unknown at
                // this level, so we instead perform the single-coordinate transform
                // treating the unchanged axis as its untransformed value — which
                // gives the correct polar result since map_pt() takes both coords.
                // To handle HLineTo/VLineTo we need to track current position.
                let mut cx_cur = 0.0_f64;
                let mut cy_cur = 0.0_f64;
                for cmd in commands.iter_mut() {
                    match cmd {
                        ferrum_scene::PathCmd::MoveTo { ref mut x, ref mut y } => {
                            let (nx, ny) = map_pt(*x, *y, plot_x, plot_y, plot_w, plot_h,
                                                  center_x, center_y, outer_r, TAU);
                            cx_cur = *x; cy_cur = *y;
                            *x = nx; *y = ny;
                        }
                        ferrum_scene::PathCmd::LineTo { ref mut x, ref mut y } => {
                            let (nx, ny) = map_pt(*x, *y, plot_x, plot_y, plot_w, plot_h,
                                                  center_x, center_y, outer_r, TAU);
                            cx_cur = *x; cy_cur = *y;
                            *x = nx; *y = ny;
                        }
                        ferrum_scene::PathCmd::QuadTo { ref mut cx, ref mut cy, ref mut x, ref mut y } => {
                            let (ncx, ncy) = map_pt(*cx, *cy, plot_x, plot_y, plot_w, plot_h,
                                                    center_x, center_y, outer_r, TAU);
                            let (nx, ny) = map_pt(*x, *y, plot_x, plot_y, plot_w, plot_h,
                                                  center_x, center_y, outer_r, TAU);
                            cx_cur = *x; cy_cur = *y;
                            *cx = ncx; *cy = ncy;
                            *x = nx; *y = ny;
                        }
                        ferrum_scene::PathCmd::CubicTo { ref mut c1x, ref mut c1y, ref mut c2x, ref mut c2y, ref mut x, ref mut y } => {
                            let (nc1x, nc1y) = map_pt(*c1x, *c1y, plot_x, plot_y, plot_w, plot_h,
                                                      center_x, center_y, outer_r, TAU);
                            let (nc2x, nc2y) = map_pt(*c2x, *c2y, plot_x, plot_y, plot_w, plot_h,
                                                      center_x, center_y, outer_r, TAU);
                            let (nx, ny) = map_pt(*x, *y, plot_x, plot_y, plot_w, plot_h,
                                                  center_x, center_y, outer_r, TAU);
                            cx_cur = *x; cy_cur = *y;
                            *c1x = nc1x; *c1y = nc1y;
                            *c2x = nc2x; *c2y = nc2y;
                            *x = nx; *y = ny;
                        }
                        ferrum_scene::PathCmd::ArcTo { ref mut x, ref mut y, .. } => {
                            let (nx, ny) = map_pt(*x, *y, plot_x, plot_y, plot_w, plot_h,
                                                  center_x, center_y, outer_r, TAU);
                            cx_cur = *x; cy_cur = *y;
                            *x = nx; *y = ny;
                        }
                        ferrum_scene::PathCmd::HLineTo { x: target_x } => {
                            // Polar changes both axes, so convert to LineTo using the
                            // current y position tracked above.
                            let old_x = *target_x;
                            let (nx, ny) = map_pt(old_x, cy_cur, plot_x, plot_y, plot_w, plot_h,
                                                  center_x, center_y, outer_r, TAU);
                            cx_cur = old_x;
                            *cmd = ferrum_scene::PathCmd::LineTo { x: nx, y: ny };
                        }
                        ferrum_scene::PathCmd::VLineTo { y: target_y } => {
                            // Polar changes both axes, so convert to LineTo using the
                            // current x position tracked above.
                            let old_y = *target_y;
                            let (nx, ny) = map_pt(cx_cur, old_y, plot_x, plot_y, plot_w, plot_h,
                                                  center_x, center_y, outer_r, TAU);
                            cy_cur = old_y;
                            *cmd = ferrum_scene::PathCmd::LineTo { x: nx, y: ny };
                        }
                        ferrum_scene::PathCmd::Close => {
                            // No coordinates to transform.
                        }
                    }
                }
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

// ── Structural feature processing ────────────────────────────────────────────

/// Process structural feature specs (secondary Y axis, axis breaks, insets).
///
/// Returns four values:
/// - `extra_axes` — additional axis scene nodes
/// - `extra_mark_batches` — additional mark batches (e.g. secondary Y marks)
/// - `extra_annotations` — additional annotation nodes (break indicators, insets)
/// - `break_results` — `(axis, BreakResult)` pairs for each BreakAxis spec, used
///   by the caller to remap primary mark pixel coordinates through the broken scale
type StructuralOutput = (
    Vec<SceneNode>,
    Vec<ferrum_scene::MarkBatch>,
    Vec<SceneNode>,
    Vec<(String, break_axis::BreakResult)>,
);

fn build_structural_nodes(
    structural: &[StructuralSpec],
    batch: &RecordBatch,
    scales: &scale_resolve::ResolvedScales,
    plot_area: &crate::layout::Rect,
    theme: &crate::layout::ThemeInputs,
    spec: &crate::spec::chart::ChartSpec,
) -> StructuralOutput {
    let mut extra_axes: Vec<SceneNode> = Vec::new();
    let mut extra_mark_batches: Vec<ferrum_scene::MarkBatch> = Vec::new();
    let mut extra_annotations: Vec<SceneNode> = Vec::new();
    let mut break_results: Vec<(String, break_axis::BreakResult)> = Vec::new();

    for item in structural {
        match item {
            StructuralSpec::SecondaryY(spec_y2) => {
                // Resolve secondary Y scale from the named field.
                let y_pixel_range = (plot_area.y, plot_area.y + plot_area.h);
                let y2_scale = match secondary_axis::resolve_secondary_y_scale(
                    &spec_y2.field,
                    batch,
                    y_pixel_range,
                ) {
                    Ok(s) => s,
                    Err(_) => continue, // missing column — skip silently
                };

                // Build right-side axis nodes.
                let axis_nodes =
                    secondary_axis::build_secondary_axis(&y2_scale, plot_area, theme, spec_y2);
                extra_axes.extend(axis_nodes);

                // Build secondary marks if there's an x encoding field.
                if let Some(x_enc) = spec.encoding.x.as_ref() {
                    let mark_nodes = secondary_axis::build_secondary_marks(
                        batch,
                        &x_enc.field,
                        &scales.x,
                        &y2_scale,
                        spec_y2,
                        plot_area,
                    );
                    if !mark_nodes.is_empty() {
                        extra_mark_batches.push(ferrum_scene::MarkBatch {
                            kind: ferrum_scene::MarkBatchKind::Line,
                            nodes: mark_nodes,
                            data_indices: None,
                            tooltips: None,
                            hrefs: None,
                            descriptions: None,
                            keys: None,
                            blend: ferrum_scene::BlendMode::Normal,
                            stroke_cap: None,
                            stroke_join: None,
                            packed_instances: None,
                        });
                    }
                }
            }

            StructuralSpec::BreakAxis(spec_brk) => {
                // Build break indicators and add them to annotations.
                let pixel_range = if spec_brk.axis == "y" {
                    (plot_area.y + plot_area.h, plot_area.y)
                } else {
                    (plot_area.x, plot_area.x + plot_area.w)
                };

                // Use the primary axis scale domain if available.
                let data_domain = if spec_brk.axis == "y" {
                    scales.y.data_domain().unwrap_or((0.0, 1.0))
                } else {
                    scales.x.data_domain().unwrap_or((0.0, 1.0))
                };

                let break_result = break_axis::apply_break_to_scale(
                    data_domain,
                    &spec_brk.gaps,
                    pixel_range,
                    spec_brk.break_size,
                );
                let indicator_nodes = break_axis::build_break_indicators(
                    &break_result,
                    plot_area,
                    &spec_brk.axis,
                    spec_brk.break_size,
                    &spec_brk.break_style,
                    theme,
                );
                extra_annotations.extend(indicator_nodes);

                // Stash the break result so the caller can remap primary marks.
                break_results.push((spec_brk.axis.clone(), break_result));
            }

            StructuralSpec::Inset(spec_inset) => {
                // `connect_to` is specified in data-space by the Python API.
                // Resolve [x, y] through the primary scales to pixel coordinates
                // before passing to build_inset_nodes (which treats them as pixels).
                let resolved_inset;
                let inset_to_build = if let Some([dx, dy]) = spec_inset.connect_to {
                    let px_x = scales.x.to_pixel_f64(dx)
                        .unwrap_or_else(|| {
                            // Fallback for ordinal / out-of-domain: linear interpolation.
                            if let Some((lo, hi)) = scales.x.data_domain() {
                                let frac = if (hi - lo).abs() < f64::EPSILON { 0.5 }
                                           else { (dx - lo) / (hi - lo) };
                                plot_area.x + frac * plot_area.w
                            } else {
                                plot_area.x + plot_area.w * 0.5
                            }
                        });
                    let px_y = scales.y.to_pixel_f64(dy)
                        .unwrap_or_else(|| {
                            if let Some((lo, hi)) = scales.y.data_domain() {
                                let frac = if (hi - lo).abs() < f64::EPSILON { 0.5 }
                                           else { (dy - lo) / (hi - lo) };
                                plot_area.y + frac * plot_area.h
                            } else {
                                plot_area.y + plot_area.h * 0.5
                            }
                        });
                    resolved_inset = super::chart_config::InsetSpec {
                        connect_to: Some([px_x, px_y]),
                        ..spec_inset.clone()
                    };
                    &resolved_inset
                } else {
                    spec_inset
                };
                let inset_nodes = inset::build_inset_nodes(inset_to_build, plot_area);
                extra_annotations.extend(inset_nodes);
            }
        }
    }

    (extra_axes, extra_mark_batches, extra_annotations, break_results)
}

// ── Break-axis mark remapping ────────────────────────────────────────────────

/// Coordinate used to hide marks that fall inside a break gap.
/// Using a large off-screen value instead of NaN avoids panics in the SVG serializer.
const BREAK_HIDDEN: f64 = -99999.0;

/// Remap the pixel coordinates of all nodes in a mark batch through a broken scale.
///
/// Marks whose data-space position falls inside a gap are hidden by moving them
/// far outside the viewport (rather than removed, to preserve data_indices alignment).
/// All other marks are repositioned to their compressed pixel coordinates.
///
/// `axis` — `"x"` or `"y"`, selects which coordinate axis is remapped.
/// Returns true when the node's primary coordinate on `axis` falls
/// within `[lo, hi]` (inclusive with 1px margin).  Used to distinguish
/// same-axis elements (y-axis ticks for a y-break) from cross-axis
/// elements (x-axis labels below the plot area) so that only same-axis
/// nodes are remapped through the broken scale.
fn node_coord_in_range(node: &SceneNode, axis: &str, lo: f64, hi: f64) -> bool {
    let margin = 1.0;
    let coord = match node {
        SceneNode::Text { x, y, .. } => if axis == "y" { *y } else { *x },
        SceneNode::Line { x1, y1, x2, y2, .. } => {
            if axis == "y" { (*y1).min(*y2) } else { (*x1).min(*x2) }
        }
        SceneNode::Rect { x, y, w, h, .. } => {
            if axis == "y" { *y } else { *x }
        }
        SceneNode::Group { children, .. } => {
            return children.iter().any(|c| node_coord_in_range(c, axis, lo, hi));
        }
        _ => return false,
    };
    coord >= lo - margin && coord <= hi + margin
}

/// `d_lo`/`d_hi` — data domain of the unbroken scale.
/// `px_lo`/`px_hi` — pixel range of the unbroken scale.
/// `break_result` — piecewise mapping from `apply_break_to_scale`.
fn remap_mark_batch_through_break(
    nodes: &mut [SceneNode],
    axis: &str,
    d_lo: f64,
    d_hi: f64,
    px_lo: f64,
    px_hi: f64,
    break_result: &break_axis::BreakResult,
) {
    for node in nodes.iter_mut() {
        remap_node(node, axis, d_lo, d_hi, px_lo, px_hi, break_result);
    }
}

/// Remap a single node's coordinates along the broken axis. Recurses into Group children.
fn remap_node(
    node: &mut SceneNode,
    axis: &str,
    d_lo: f64,
    d_hi: f64,
    px_lo: f64,
    px_hi: f64,
    br: &break_axis::BreakResult,
) {
    match node {
        SceneNode::Circle { cx, cy, .. } => {
            let coord = if axis == "y" { cy } else { cx };
            *coord = remap_coord(*coord, d_lo, d_hi, px_lo, px_hi, br)
                .unwrap_or(BREAK_HIDDEN);
        }
        SceneNode::Rect { x, y, w, h, .. } => {
            if axis == "y" {
                let top = remap_coord(*y, d_lo, d_hi, px_lo, px_hi, br);
                let bottom = remap_coord(*y + *h, d_lo, d_hi, px_lo, px_hi, br);
                match (top, bottom) {
                    (Some(t), Some(b)) => {
                        *y = t.min(b);
                        *h = (b - t).abs();
                    }
                    _ => { *h = 0.0; }
                }
            } else {
                let left = remap_coord(*x, d_lo, d_hi, px_lo, px_hi, br);
                let right = remap_coord(*x + *w, d_lo, d_hi, px_lo, px_hi, br);
                match (left, right) {
                    (Some(l), Some(r)) => {
                        *x = l.min(r);
                        *w = (r - l).abs();
                    }
                    _ => { *w = 0.0; }
                }
            }
        }
        SceneNode::Line { x1, y1, x2, y2, .. } => {
            if axis == "y" {
                *y1 = remap_coord(*y1, d_lo, d_hi, px_lo, px_hi, br).unwrap_or(BREAK_HIDDEN);
                *y2 = remap_coord(*y2, d_lo, d_hi, px_lo, px_hi, br).unwrap_or(BREAK_HIDDEN);
            } else {
                *x1 = remap_coord(*x1, d_lo, d_hi, px_lo, px_hi, br).unwrap_or(BREAK_HIDDEN);
                *x2 = remap_coord(*x2, d_lo, d_hi, px_lo, px_hi, br).unwrap_or(BREAK_HIDDEN);
            }
        }
        SceneNode::Path { commands, .. } => {
            for cmd in commands.iter_mut() {
                remap_path_cmd(cmd, axis, d_lo, d_hi, px_lo, px_hi, br);
            }
        }
        SceneNode::Text { x, y, .. } => {
            let coord = if axis == "y" { y } else { x };
            *coord = remap_coord(*coord, d_lo, d_hi, px_lo, px_hi, br)
                .unwrap_or(BREAK_HIDDEN);
        }
        SceneNode::Group { children, .. } => {
            for child in children.iter_mut() {
                remap_node(child, axis, d_lo, d_hi, px_lo, px_hi, br);
            }
        }
        // Polyline, Polygon, Image, Raw, Arc — leave untouched.
        _ => {}
    }
}

/// Remap a single PathCmd coordinate along the broken axis.
fn remap_path_cmd(
    cmd: &mut ferrum_scene::PathCmd,
    axis: &str,
    d_lo: f64,
    d_hi: f64,
    px_lo: f64,
    px_hi: f64,
    br: &break_axis::BreakResult,
) {
    use ferrum_scene::PathCmd;
    let remap = |v: &mut f64| {
        *v = remap_coord(*v, d_lo, d_hi, px_lo, px_hi, br).unwrap_or(BREAK_HIDDEN);
    };
    match cmd {
        PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => {
            if axis == "y" { remap(y); } else { remap(x); }
        }
        PathCmd::QuadTo { cx, cy, x, y } => {
            if axis == "y" { remap(cy); remap(y); } else { remap(cx); remap(x); }
        }
        PathCmd::CubicTo { c1x, c1y, c2x, c2y, x, y } => {
            if axis == "y" { remap(c1y); remap(c2y); remap(y); }
            else { remap(c1x); remap(c2x); remap(x); }
        }
        PathCmd::ArcTo { x, y, .. } => {
            if axis == "y" { remap(y); } else { remap(x); }
        }
        PathCmd::HLineTo { x } => { if axis != "y" { remap(x); } }
        PathCmd::VLineTo { y } => { if axis == "y" { remap(y); } }
        PathCmd::Close => {}
    }
}

/// Reverse-map a pixel coordinate to data-space, then forward-map through the
/// broken scale.  Returns `None` when the data value falls in a gap.
///
/// The reverse-map uses the unbroken (primary) scale's linear interpolation:
/// `data = d_lo + (px - px_lo) / (px_hi - px_lo) * (d_hi - d_lo)`.
/// This is correct for all continuous scale types when the underlying scale is
/// linear in pixel space (which is the case — all ferrum scales produce a linear
/// pixel mapping; the non-linearity lives in the data transform, not the range).
fn remap_coord(
    px: f64,
    d_lo: f64,
    d_hi: f64,
    px_lo: f64,
    px_hi: f64,
    br: &break_axis::BreakResult,
) -> Option<f64> {
    let span = px_hi - px_lo;
    if span.abs() < f64::EPSILON { return Some(px); }
    let data_val = d_lo + (px - px_lo) / span * (d_hi - d_lo);
    let data_val = data_val.clamp(d_lo.min(d_hi), d_lo.max(d_hi));
    break_axis::broken_scale_map(data_val, br)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_scene::{FillStroke, PathCmd, SceneNode};
    use crate::layout::Rect;

    fn default_fill_stroke() -> ferrum_scene::FillStroke {
        FillStroke {
            fill: None,
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        }
    }

    /// B5: Path nodes inside apply_polar_node_transform must have their
    /// x/y coordinates remapped through the polar projection. The previous
    /// catch-all `_ => {}` left Path nodes at their original Cartesian coords.
    #[test]
    fn b5_path_nodes_are_polar_transformed() {
        let plot_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        // A Path node placed at the top-left corner of the plot area in Cartesian space.
        // After polar transform this coordinate will NOT remain at (0, 0).
        let cartesian_x = 0.0_f64;
        let cartesian_y = 0.0_f64;

        let mut nodes = vec![
            SceneNode::Path {
                commands: vec![
                    PathCmd::MoveTo { x: cartesian_x, y: cartesian_y },
                    PathCmd::LineTo { x: 100.0, y: 100.0 },
                    PathCmd::Close,
                ],
                style: default_fill_stroke(),
                closed: true,
            }
        ];

        apply_polar_node_transform(&mut nodes, &plot_area);

        // The Path node must still be a Path node (no type change).
        match &nodes[0] {
            SceneNode::Path { commands, .. } => {
                // The MoveTo endpoint for (0, 0) in Cartesian maps to a non-zero polar
                // coordinate. Specifically: theta = 0, r = 1 * outer_r → (cx, cy - r).
                let outer_r = plot_area.w.min(plot_area.h) / 2.0; // 100.0
                let center_x = plot_area.x + plot_area.w / 2.0;   // 100.0
                let center_y = plot_area.y + plot_area.h / 2.0;   // 100.0
                // theta = 0 / 200 * TAU = 0; r = (0 + 200 - 0) / 200 * 100 = 100
                // nx = center_x + r * sin(0) = 100; ny = center_y - r * cos(0) = 0
                let expected_x = center_x + outer_r * 0_f64.sin(); // 100.0
                let expected_y = center_y - outer_r * 0_f64.cos(); // 0.0
                match &commands[0] {
                    PathCmd::MoveTo { x, y } => {
                        assert!(
                            (x - expected_x).abs() < 1e-9,
                            "MoveTo x after polar transform: expected {expected_x}, got {x}"
                        );
                        assert!(
                            (y - expected_y).abs() < 1e-9,
                            "MoveTo y after polar transform: expected {expected_y}, got {y}"
                        );
                    }
                    other => panic!("expected MoveTo, got {other:?}"),
                }
            }
            other => panic!("expected Path node after transform, got {other:?}"),
        }
    }

    /// B5: HLineTo and VLineTo inside a Path must be converted to LineTo after
    /// polar transform (since polar changes both x and y).
    #[test]
    fn b5_hlineto_vlineto_converted_to_lineto_under_polar() {
        let plot_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        let mut nodes = vec![
            SceneNode::Path {
                commands: vec![
                    PathCmd::MoveTo { x: 50.0, y: 100.0 },
                    PathCmd::HLineTo { x: 150.0 },
                    PathCmd::VLineTo { y: 50.0 },
                    PathCmd::Close,
                ],
                style: default_fill_stroke(),
                closed: true,
            }
        ];

        apply_polar_node_transform(&mut nodes, &plot_area);

        match &nodes[0] {
            SceneNode::Path { commands, .. } => {
                // HLineTo and VLineTo must not survive the polar transform.
                for cmd in commands {
                    assert!(
                        !matches!(cmd, PathCmd::HLineTo { .. } | PathCmd::VLineTo { .. }),
                        "HLineTo/VLineTo must be converted to LineTo under polar, found: {cmd:?}"
                    );
                }
                // The converted HLineTo/VLineTo must become LineTo variants.
                assert!(
                    matches!(commands[1], PathCmd::LineTo { .. }),
                    "HLineTo must become LineTo, got: {:?}", commands[1]
                );
                assert!(
                    matches!(commands[2], PathCmd::LineTo { .. }),
                    "VLineTo must become LineTo, got: {:?}", commands[2]
                );
            }
            other => panic!("expected Path node, got {other:?}"),
        }
    }

    /// B5: Control points in QuadTo/CubicTo must also be transformed.
    #[test]
    fn b5_quadto_control_points_are_transformed() {
        let plot_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        let mut nodes = vec![
            SceneNode::Path {
                commands: vec![
                    PathCmd::MoveTo { x: 0.0, y: 100.0 },
                    PathCmd::QuadTo { cx: 50.0, cy: 0.0, x: 100.0, y: 100.0 },
                    PathCmd::Close,
                ],
                style: default_fill_stroke(),
                closed: true,
            }
        ];

        apply_polar_node_transform(&mut nodes, &plot_area);

        match &nodes[0] {
            SceneNode::Path { commands, .. } => {
                match &commands[1] {
                    PathCmd::QuadTo { cx, cy, x, y } => {
                        // The control point (50, 0) in Cartesian must have been transformed.
                        // In Cartesian: cx=50, cy=0 → theta=TAU/4, r=outer_r → nx = center + r, ny = center.
                        // Just verify the control point differs from the original values.
                        assert!(
                            (*cx - 50.0).abs() > 1e-6 || (*cy - 0.0).abs() > 1e-6,
                            "QuadTo control point must be polar-transformed, still at ({cx},{cy})"
                        );
                        // Endpoint (100, 100) → theta=TAU/2, r=outer_r/2 → verify it changed.
                        assert!(
                            (*x - 100.0).abs() > 1e-6 || (*y - 100.0).abs() > 1e-6,
                            "QuadTo endpoint must be polar-transformed, still at ({x},{y})"
                        );
                    }
                    other => panic!("expected QuadTo, got {other:?}"),
                }
            }
            other => panic!("expected Path node, got {other:?}"),
        }
    }
}
