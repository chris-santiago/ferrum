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

        // Mark batches
        let mut mark_batches: Vec<MarkBatch> = Vec::new();

        for (li, layer) in prep.layers.iter().enumerate() {
            let layer_batch = &layer_batches[li];
            if layer_batch.num_rows() == 0 {
                continue;
            }

            // Position adjustment
            let adjusted_owned;
            let layer_batch: &RecordBatch = if layer.position.is_some() {
                adjusted_owned = position::apply_position(
                    layer_batch,
                    layer.position.as_ref(),
                    &scales,
                    &layer.encoding,
                )?;
                &adjusted_owned
            } else {
                layer_batch
            };

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

            let result = draw::dispatch_mark_build(&layer.mark, &ctx);
            let keys = extract_keys(&layer.encoding, layer_batch, result.data_indices.as_deref());
            mark_batches.push(MarkBatch {
                kind: result.kind,
                nodes: result.nodes,
                data_indices: result.data_indices,
                tooltips: result.tooltips,
                hrefs: result.hrefs,
                descriptions: result.descriptions,
                keys,
                blend: BlendMode::Normal,
                stroke_cap: mark_style.stroke_cap.as_deref().and_then(|s| match s {
                    "round" => Some(ferrum_scene::StrokeCap::Round),
                    "square" => Some(ferrum_scene::StrokeCap::Square),
                    "butt" => Some(ferrum_scene::StrokeCap::Butt),
                    _ => None,
                }),
                stroke_join: mark_style.stroke_join.as_deref().and_then(|s| match s {
                    "round" => Some(ferrum_scene::StrokeJoin::Round),
                    "bevel" => Some(ferrum_scene::StrokeJoin::Bevel),
                    "miter" => Some(ferrum_scene::StrokeJoin::Miter),
                    _ => None,
                }),
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
        let outer_radius_px = (panel.plot_area.w.min(panel.plot_area.h)) / 2.0;
        let scene_coord = spec.coord.as_ref()
            .map(|c| to_scene_coord(c, outer_radius_px))
            .unwrap_or(CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
            });

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
