use arrow::record_batch::RecordBatch;
use ferrum_scene::{
    BindingRole, BlendMode, CoordKind, InteractionConfig, LayoutScale, MarkBatch, Panel,
    ParamBinding, PanelTickLevels, SceneGraph, SceneNode, TickLevel,
};
use crate::spec::coord::to_scene_coord;

use crate::layout::{AxisLayout, LayoutResult, ResolveMode, ThemeInputs};
use crate::spec::chart::ChartSpec;

use super::arrow_cast::col_as_str;
use super::chart_config::StructuralSpec;
use super::config::RenderConfig;
use super::draw::{self, to_scene_color, to_scene_text_style, DrawCtx};
use super::marks;
use super::prepare::PreparedInputs;
use super::scale_resolve::LeafScaleContext;
use super::{
    break_axis, inset,
    filter_batch_by_facet, position, scale_resolve, RenderError, RenderWarning, CLIP_ID_PREFIX,
};

#[allow(clippy::too_many_arguments)]
pub fn build_scene(
    spec: &ChartSpec,
    prep: &PreparedInputs,
    layout: &LayoutResult,
    theme: &ThemeInputs,
    config: &RenderConfig,
    warnings: &mut Vec<RenderWarning>,
    chart_config: &super::chart_config::ChartConfig,
    // D4b composite seam: the shared-domain context for this leaf, forwarded to
    // the per-panel scale pass. `None` for every standalone (flat/facet) render
    // → byte-identical.
    leaf_scales: Option<&LeafScaleContext>,
) -> Result<SceneGraph, RenderError> {
    let background = config.background.or(Some(theme.colors.background_color));

    let mut title_nodes: Vec<SceneNode> = Vec::new();
    let mut legend_nodes: Vec<SceneNode> = Vec::new();

    // Chart title
    build_title(layout, spec, theme, &mut title_nodes);

    let mut panels: Vec<Panel> = Vec::new();
    let mut tick_levels: Vec<PanelTickLevels> = Vec::new();
    // Layer→slot map resolved for the domainParam binding pass (secondary-y,
    // GH #52). The mapping is structural (driven by `independent_y` flags), so
    // it is identical across panels; capture it once from the first panel that
    // resolves. Stays `None` — collapsing to the shared slot-0 path — when every
    // panel is empty (no marks, hence no bindings to route anyway).
    let mut resolved_y_slots: Option<scale_resolve::YScaleSlots> = None;

    // Heuristic text metrics for per-panel independent axis layout rebuilds.
    // Constructed once outside the panel loop; FontdueMetrics has no mutable
    // state so sharing it across panels is correct.
    let facet_metrics = super::font::FontdueMetrics::new();

    for (panel_idx, panel) in layout.panels.iter().enumerate() {
        if panel.plot_area.w <= 0.0 || panel.plot_area.h <= 0.0 {
            warnings.push(RenderWarning::EmptyPanel { panel_index: panel_idx });
            continue;
        }

        // Strip title — emitted as separate nodes in the panel, not a group.
        // Includes both the column-header strip (top) and, in grid mode, the
        // row-header strip (right side). Both are appended to the same vec so
        // the compositor's offset logic picks them up without a schema change.
        let mut strip_title_nodes: Vec<SceneNode> = panel.strip_title.as_ref()
            .map(|strip| marks::strip_title::build_strip_title(strip, &panel.plot_area, theme))
            .unwrap_or_default();
        if let Some(row_strip) = &panel.row_strip_title {
            strip_title_nodes.extend(
                marks::strip_title::build_row_strip_title(row_strip, panel.plot_area.h, theme)
            );
        }

        // Facet filter: filter the merged batch on col (and row in grid mode).
        let panel_batch = if let Some(key) = &panel.facet_key {
            let col_filtered = filter_batch_by_facet(prep.final_batch(), &key.field, &key.value)?;
            if let Some(rk) = &panel.row_facet_key {
                filter_batch_by_facet(&col_filtered, &rk.field, &rk.value)?
            } else {
                col_filtered
            }
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
                        let col_filtered =
                            filter_batch_by_facet(src, &key.field, &key.value)?;
                        if let Some(rk) = &panel.row_facet_key {
                            filter_batch_by_facet(&col_filtered, &rk.field, &rk.value)
                        } else {
                            Ok(col_filtered)
                        }
                    } else {
                        Ok(src.clone())
                    }
                }
            })
            .collect::<Result<Vec<_>, RenderError>>()?;

        // Per-panel scale build (encoding merge + param-domain substitution +
        // scale resolution + color-config re-apply) through the single
        // `resolve_panel_scales` seam, so the prepare provisional pass and this
        // per-panel pass cannot drift on what scales get built or on remembering
        // to re-apply the color config.
        let (rendering_spec_for_panel, scales) = resolve_panel_scales(
            spec,
            prep,
            panel,
            &panel_batch,
            &layer_batches,
            theme,
            chart_config,
            warnings,
            leaf_scales,
        )?;

        tick_levels.push(build_tick_levels(&scales, panel_idx));
        if resolved_y_slots.is_none() {
            resolved_y_slots = Some(scales.y_slots.clone());
        }

        // Per-panel axes — collected from the globally-computed layout.
        // When a facet channel requests independent scale resolution, the global
        // axis layout is replaced with a fresh per-panel layout derived from the
        // per-panel scales resolved above (`resolve_panel_axes`).
        let panel_axes_layout: Vec<&crate::layout::AxisLayout> = layout
            .axes
            .iter()
            .filter(|a| a.panel_index == panel_idx)
            .collect();
        let panel_x_axis_global = panel_axes_layout
            .iter()
            .copied()
            .find(|a| matches!(a.orient,
                crate::layout::AxisOrient::Bottom | crate::layout::AxisOrient::Top));
        let panel_y_axis_global = panel_axes_layout
            .iter()
            .copied()
            .find(|a| matches!(a.orient,
                crate::layout::AxisOrient::Left | crate::layout::AxisOrient::Right));

        // Secondary y-axes for this panel (secondary-y-axis, GH #52): one per
        // `independent_y` layer, orient Right, stacked outward. Empty
        // `layout.secondary_y_axes` (the pre-#52 default) keeps this empty —
        // byte-identical to the shared path.
        let panel_secondary_y: Vec<&crate::layout::AxisLayout> = layout
            .secondary_y_axes
            .iter()
            .filter(|a| a.panel_index == panel_idx)
            .collect();

        // Independent-axis rebuild (MOD-09): owns the per-panel layouts so the
        // effective references below stay valid for the rest of the block.
        let panel_axes = resolve_panel_axes(
            &rendering_spec_for_panel,
            spec,
            &scales,
            prep,
            panel,
            panel_idx,
            theme,
            &facet_metrics,
        );

        // Resolve the effective per-panel axis references: use the freshly-built
        // independent layout when available, otherwise the global shared one.
        let panel_x_axis: Option<&AxisLayout> = panel_axes
            .independent_x
            .as_ref()
            .or(panel_x_axis_global);
        let panel_y_axis: Option<&AxisLayout> = panel_axes
            .independent_y
            .as_ref()
            .or(panel_y_axis_global);

        // Grid + axis build and above/below routing (MOD-09).
        let PanelAxisGrid {
            axes_below: mut axes_nodes,
            axes_above: axes_above_nodes,
            grid: mut grid_nodes,
            grid_above,
        } = route_panel_axes_and_grid(
            spec,
            &scales,
            panel,
            &panel_axes_layout,
            panel_x_axis,
            panel_y_axis,
            &panel_secondary_y,
            panel_axes.x_independent,
            panel_axes.y_independent,
            theme,
            chart_config,
        );

        // Per-layer mark batches (MOD-09).
        let mut mark_batches = build_panel_mark_batches(
            spec,
            prep,
            &layer_batches,
            &scales,
            panel,
            theme,
            warnings,
        )?;

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
                y_domains: Vec::new(),
            });

        // Inject computed axis domains into the scene coord so the JS zoom handler
        // can read the actual displayed domain even for auto-scaled charts.
        let scene_coord = match scene_coord {
            CoordKind::Cartesian { x_domain: None, y_domain: None, expand, clip, y_domains } => {
                let x_dom = scales.x.data_domain();
                let y_dom = scales.y.data_domain();
                CoordKind::Cartesian { x_domain: x_dom, y_domain: y_dom, expand, clip, y_domains }
            }
            CoordKind::Fixed { x_domain: None, y_domain: None, ratio, expand, clip } => {
                let x_dom = scales.x.data_domain();
                let y_dom = scales.y.data_domain();
                CoordKind::Fixed { x_domain: x_dom, y_domain: y_dom, ratio, expand, clip }
            }
            other => other,
        };

        // Per-slot y-domains (secondary-y-axis, GH #52 Task 8): one domain per
        // slot, index = slot (slot 0 mirrors the primary `y_domain` injected
        // above). Applied as a distinct pass, after domain injection, so it
        // covers every Cartesian branch above (auto-scaled AND an explicit
        // `coord=` domain) uniformly. Populated only when this panel resolved
        // an `independent_y` layer; every other panel's `y_domains` stays
        // empty — the byte-stable default — so this pass is a no-op for every
        // existing chart. The interactive runtime (Task 9) reads slot `k`'s
        // domain to relabel/rescale that layer's own axis independently of the
        // panel-level zoom/pan affine (spec §6, §8.5).
        let scene_coord = if scales.y_slots.has_independent() {
            match scene_coord {
                CoordKind::Cartesian { x_domain, y_domain, expand, clip, .. } => {
                    let y_domains = scales.y_slots.slots()
                        .iter()
                        .map(|s| s.data_domain())
                        .collect();
                    CoordKind::Cartesian { x_domain, y_domain, expand, clip, y_domains }
                }
                other => other,
            }
        } else {
            scene_coord
        };

        // Annotations: render user-specified annotations on the first panel only.
        // `build_annotations` partitions nodes by the Text spec's `z` field:
        //   - `below_marks` → appended to the panel `grid` slot (pre-marks bucket)
        //   - `above_marks` → seeded into the `annotations` slot (post-marks bucket)
        // This mirrors how above-marks grid/axes (zindex >= 1) route into `annotations`.
        let annotation_nodes = if panel_idx == 0 && !chart_config.annotations.is_empty() {
            let ann_ctx = super::annotation::ScaleContext {
                plot_area: panel.plot_area,
                x_scale: &scales.x,
                y_scale: &scales.y,
            };
            super::annotation::build_annotations(&chart_config.annotations, &ann_ctx)
        } else {
            super::annotation::AnnotationNodes {
                below_marks: Vec::new(),
                above_marks: Vec::new(),
            }
        };

        // Structural features: axis breaks, insets.
        // Only applied to the first panel (non-faceted charts).
        let (structural_axes, structural_marks, structural_annotations, break_results) =
            if panel_idx == 0 && !chart_config.structural.is_empty() {
                build_structural_nodes(
                    &chart_config.structural,
                    &scales,
                    &panel.plot_area,
                    theme,
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

        // zindex (B5): an above-marks grid is moved out of the (before-marks)
        // `grid` slot into the `annotations` slot (emitted after marks). The
        // below-marks default keeps the grid in `grid` for byte-identical output.
        let (mut grid_below, grid_above_nodes) = if grid_above {
            (Vec::new(), grid_nodes)
        } else {
            (grid_nodes, Vec::new())
        };

        // z-routing for text annotations (XDEAD-03): Text annotations with
        // z="below_marks" are appended to the grid slot (the pre-marks "below"
        // bucket), painted on top of gridlines but below data marks.  All other
        // annotation nodes go into the post-marks `annotations` slot below.
        // Even when grid_above is true (grid_below is empty), below-marks
        // annotation nodes must still land in the grid slot.
        grid_below.extend(annotation_nodes.below_marks);

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
        // Annotation list (emitted after marks): user annotations (`above_marks`
        // bucket), then any above-marks grid + axes (zindex >= 1), then structural
        // annotations.  `below_marks` annotation nodes were routed into `grid_below`
        // above; they do not appear here.
        let final_annotations: Vec<SceneNode> = {
            let mut v = annotation_nodes.above_marks;
            v.extend(grid_above_nodes);
            v.extend(axes_above_nodes);
            v.extend(structural_annotations);
            v
        };

        panels.push(Panel {
            id: panel_idx,
            plot_area,
            clip: panel_clip,
            coord: scene_coord,
            grid: grid_below,
            marks: final_marks,
            axes: final_axes,
            annotations: final_annotations,
            strip_title: strip_title_nodes,
            layout_scale: LayoutScale::identity(),
        });
    }

    // Legend
    build_legend_decorations(layout, spec, prep, theme, chart_config, &mut legend_nodes)?;

    // zindex (B5 unit 3): coarse below/above-marks ordering for the legend,
    // mirroring the axis mechanism. Per-channel `Legend(zindex=...)` wins over
    // chart-level `configure_legend(zindex=...)`. `>= 1` routes the legend into
    // the first panel's annotation slot (drawn after that panel's marks); absent
    // or `<= 0` keeps it in the top-level `legend` slot (the byte-identical
    // default). Because the legend sits outside the plot area, both slots render
    // after the data marks, so this is usually a visual no-op — implemented for
    // parity with the axis `zindex` semantics rather than visible layering.
    let legend_zindex = prep
        .legend_overrides
        .zindex
        .or_else(|| chart_config.legend.as_ref().and_then(|l| l.style.zindex));
    if legend_zindex.is_some_and(|z| z >= 1) && !legend_nodes.is_empty() {
        if let Some(first_panel) = panels.first_mut() {
            first_panel.annotations.append(&mut legend_nodes);
        }
    }

    // Param→scene bindings (D6, 5e-2a). Computed from the ORIGINAL `spec`,
    // which still carries `domainParam`/transform `param`/selection `bind`:
    // the static resolver only mutated per-panel clones.
    let y_slots = resolved_y_slots.unwrap_or_default();
    let param_bindings =
        collect_param_bindings(spec, &prep.layers, &y_slots, layout.panels.len());

    let interaction = InteractionConfig {
        zoom_enabled: !spec.selections.is_empty(),
        pan_enabled: !spec.selections.is_empty(),
        conditionals: spec.conditionals.clone(),
        linked_panels: Vec::new(),
        tick_levels,
        toolbar: true,
        params: spec.params.clone(),
        param_bindings,
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

/// Static reactive-rescale substitution (D6).
///
/// Walks the spec's continuous positional/color/size/opacity scales; for each
/// one carrying a `domainParam` reference, substitutes the named variable's
/// static numeric-array value as the concrete `domain` (and clears the
/// reference). A reference to a selection (or a non-numeric variable) leaves
/// `domain = None`, so the renderer auto-infers from data — the correct static
/// semantics for an empty selection.
///
/// No-op when `spec.params` is empty (the byte-stability gate): the early return
/// keeps param-free specs on the exact pre-D6 code path.
/// The single per-panel scale-build seam: merge the layer-0 encoding onto the
/// chart encoding, substitute reactive `domainParam` references into concrete
/// domains, resolve the panel's scales over its pixel range, and re-apply the
/// chart-level color config to the resolved color scale.
///
/// This deliberately re-does work the prepare provisional pass already did once:
/// `prep.provisional_scales` exists **for tick-label generation only** (its pixel
/// ranges are not panel-specific), so the final per-panel scales — whose pixel
/// ranges differ per panel — must be resolved fresh here. The provisional-vs-final
/// duality is real and kept; what this seam removes is the silent coupling where
/// the encoding-merge, param-domain substitution, and color-config re-apply were
/// each hand-repeated next to the re-resolution (and the color-config re-apply was
/// a latent drift point — it had to be remembered both here and on the provisional
/// scales in `prepare_and_layout`).
///
/// Returns the merged-encoding `ChartSpec` (still needed by the caller for
/// independent-axis label formatting and structural-node building) and the
/// resolved scales. Scale warnings are appended to `warnings`. `panel_batch` is
/// the caller's already-facet-filtered batch for this panel (the same one used to
/// resolve layer batches), passed in to avoid re-running the facet filter.
#[allow(clippy::too_many_arguments)]
fn resolve_panel_scales(
    spec: &ChartSpec,
    prep: &PreparedInputs,
    panel: &crate::layout::PanelLayout,
    panel_batch: &RecordBatch,
    // Per-panel layer batches (one per `prep.layers`, facet-filtered). Slot 0 /
    // the primary y resolves against `panel_batch` exactly as before; each
    // independent layer's y-slot resolves against its own batch here.
    layer_batches: &[RecordBatch],
    theme: &ThemeInputs,
    chart_config: &super::chart_config::ChartConfig,
    warnings: &mut Vec<RenderWarning>,
    // D4b composite seam: shared-domain context for this leaf. `None` for
    // standalone renders → resolves exactly as before.
    leaf_scales: Option<&LeafScaleContext>,
) -> Result<(ChartSpec, scale_resolve::ResolvedScales), RenderError> {
    // Encoding merge: layer-0 encoding overlays the chart-level encoding.
    let mut merged_encoding = spec.encoding.clone();
    merged_encoding.overlay_from(&prep.layers[0].encoding);
    let mut rendering_spec_for_panel = ChartSpec {
        encoding: merged_encoding,
        ..spec.clone()
    };

    // Reactive-rescale substitution (D6): turn `domainParam` references into
    // concrete domains before scale resolution. No-op when `params` is empty.
    resolve_param_domains(&mut rendering_spec_for_panel);

    // Scale resolution over this panel's pixel range.
    let (mut scales, scale_warnings) = scale_resolve::resolve_scales_with_leaf_context(
        &rendering_spec_for_panel,
        panel_batch,
        &prep.transform_outputs,
        (panel.plot_area.x, panel.plot_area.x + panel.plot_area.w),
        (panel.plot_area.y, panel.plot_area.y + panel.plot_area.h),
        theme,
        leaf_scales,
    )?;
    warnings.extend(scale_warnings);

    // Apply chart_config color overrides (level 3) to the per-panel color scale.
    // Must run after scale resolution because `resolve_scales_with_outputs`
    // independently re-resolves the color scale for each panel, discarding the
    // provisional override applied to `prep.provisional_scales` in
    // `prepare_and_layout`.
    if let Some(ref cfg) = chart_config.color {
        super::apply_color_config_to_color_scale(&mut scales.color, cfg);
    }

    // Per-layer independent y-scale slots (secondary-y-axis, GH #52). Byte-stable
    // gate: only build slots when some non-primary layer requests `independent_y`;
    // otherwise leave `scales.y_slots` at its empty default so shared and
    // `y:"shared"` charts resolve exactly as before. Slot 0 stays the primary `y`
    // resolved above (against layer 0's / the panel batch), unchanged.
    if prep.layers.iter().skip(1).any(|l| l.independent_y) {
        let mut slots: Vec<scale_resolve::ScaleKind> = vec![scales.y.clone()];
        let mut layer_slot: Vec<usize> = vec![0; prep.layers.len()];
        for (li, layer) in prep.layers.iter().enumerate() {
            // Layer 0 is always the primary/left axis regardless of its flag.
            if li == 0 || !layer.independent_y {
                continue;
            }
            let y_scale = resolve_layer_y_scale(
                spec,
                layer,
                &layer_batches[li],
                prep,
                panel,
                theme,
                leaf_scales,
                warnings,
            )?;
            slots.push(y_scale);
            layer_slot[li] = slots.len() - 1;
        }
        scales.y_slots = scale_resolve::YScaleSlots::new(slots, layer_slot);
    }

    Ok((rendering_spec_for_panel, scales))
}

/// Resolve one independent layer's y-scale slot (secondary-y-axis, GH #52).
///
/// Reuses the exact per-layer resolution path the primary y uses: the layer's
/// encoding overlays the chart encoding, `domainParam` references are
/// substituted, and [`scale_resolve::resolve_scales_with_leaf_context`] applies
/// every rule the primary gets — explicit `scale=` on the layer's y encoding
/// wins, bar zero-anchor, y2 domain extension, nice-ing, and every `ScaleSpec`
/// type. The layer's own batch (its data + transform outputs) seeds the domain,
/// and its `mark` drives mark-dependent rules (e.g. bar zero-anchor). Only the
/// resolved `.y` is kept.
///
/// Warnings from this resolution ARE propagated into `warnings` — they can
/// concern the y channel itself (e.g. a non-finite/degenerate domain on this
/// layer's own field), which nothing else surfaces: the primary pass only
/// warns about layer-0's channels, and this slot's y-field is independent of
/// layer 0's. A genuine y-field error still propagates as `Err`.
#[allow(clippy::too_many_arguments)]
fn resolve_layer_y_scale(
    spec: &ChartSpec,
    layer: &super::prepare::LayerPrepared,
    layer_batch: &RecordBatch,
    prep: &PreparedInputs,
    panel: &crate::layout::PanelLayout,
    theme: &ThemeInputs,
    leaf_scales: Option<&LeafScaleContext>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<scale_resolve::ScaleKind, RenderError> {
    let mut layer_encoding = spec.encoding.clone();
    layer_encoding.overlay_from(&layer.encoding);
    let mut layer_spec = ChartSpec {
        mark: layer.mark,
        encoding: layer_encoding,
        // Resolve this slot from ITS OWN field only: dropping `layers` stops
        // `numeric_domain_union` from re-unioning sibling layers' y fields, so an
        // independent layer's y-scale spans exactly its own data.
        layers: None,
        ..spec.clone()
    };
    resolve_param_domains(&mut layer_spec);

    let (layer_scales, layer_warnings) = scale_resolve::resolve_scales_with_leaf_context(
        &layer_spec,
        layer_batch,
        &prep.transform_outputs,
        (panel.plot_area.x, panel.plot_area.x + panel.plot_area.w),
        (panel.plot_area.y, panel.plot_area.y + panel.plot_area.h),
        theme,
        leaf_scales,
    )?;
    warnings.extend(layer_warnings);
    Ok(layer_scales.y)
}

/// Per-panel axis layouts for the independent-resolve facet case (MOD-09).
///
/// When a facet channel requests [`ResolveMode::Independent`], the x and/or y
/// axis is rebuilt from this panel's freshly-resolved scales (its pixel range
/// and tick labels differ per panel). Shared channels keep the global layout, so
/// the matching field stays `None` and the caller falls back to the global axis.
///
/// The struct OWNS the rebuilt layouts so references into them stay valid for the
/// rest of the panel block; `*_independent` mirror the facet resolve mode so the
/// caller routes axes the same way the inline code did.
struct PanelAxes {
    independent_x: Option<AxisLayout>,
    independent_y: Option<AxisLayout>,
    x_independent: bool,
    y_independent: bool,
}

/// Rebuild the per-panel independent axes (MOD-09 extraction of the inline
/// independent-axis block). Pure extraction — same tick-input derivation through
/// the shared [`build_axis_tick_inputs`] helper (SPINE-07), same `Immediate`
/// format mode, same empty minor vec (independent panels do not rebuild minor
/// ticks), same `layout_x_axis`/`layout_y_axis` calls. A non-independent channel
/// yields `None`, so the caller keeps the global layout for it.
#[allow(clippy::too_many_arguments)]
fn resolve_panel_axes(
    rendering_spec_for_panel: &ChartSpec,
    spec: &ChartSpec,
    scales: &scale_resolve::ResolvedScales,
    prep: &PreparedInputs,
    panel: &crate::layout::PanelLayout,
    panel_idx: usize,
    theme: &ThemeInputs,
    facet_metrics: &super::font::FontdueMetrics,
) -> PanelAxes {
    // Independent-axis override: when the facet spec requests independent
    // resolution for x or y, rebuild that channel's AxisLayout from the
    // per-panel scales. Shared channels keep the global layout as-is.
    let x_independent = spec.facet.as_ref()
        .map(|f| f.resolve.x == ResolveMode::Independent)
        .unwrap_or(false);
    let y_independent = spec.facet.as_ref()
        .map(|f| f.resolve.y == ResolveMode::Independent)
        .unwrap_or(false);

    // Re-derive raw format specs from the merged rendering encoding so that
    // independent-axis label formatting uses the same precedence logic as
    // the shared path (Axis(label_format=) > encoding.format > none).
    // `resolve_axis_label_format` is the canonical single source of truth
    // for this precedence — calling it here avoids duplicating the logic
    // and ensures both paths stay in sync.
    let (x_fmt_spec, x_fmt_type) = super::prepare::resolve_axis_label_format(
        rendering_spec_for_panel.encoding.x.as_ref(),
    );
    let (y_fmt_spec, y_fmt_type) = super::prepare::resolve_axis_label_format(
        rendering_spec_for_panel.encoding.y.as_ref(),
    );

    let independent_x = if x_independent {
        // Use the global tick count as the hint so per-panel independent axes
        // produce a similar label density to what the layout engine chose globally.
        let x_tick_count = prep.axes.x.tick_labels.len().max(1);
        let mut x_input = prep.axes.x.clone();
        // Re-derive the per-panel labels + projection through the SAME
        // `build_axis_tick_inputs` helper the global axis path uses, so the
        // tick_labels → non-ordinal-y reverse → format → projection sequence
        // lives in one place. `Immediate` mode applies the encoding-level label
        // format to the fresh per-panel raw labels right away (the per-panel
        // layout is built directly below, bypassing `apply_label_format_to_axis`).
        // Independent panels keep `minor: Vec::new()` (no per-panel minor-tick
        // rebuild), passed as the empty minor vec.
        let (new_x_labels, x_projection, _threaded) = super::prepare::build_axis_tick_inputs(
            &scales.x,
            x_tick_count,
            super::prepare::TickFormatMode::Immediate {
                format: x_fmt_spec.as_deref(),
                format_type: x_fmt_type.as_deref(),
            },
            false,
            Vec::new(),
        );
        x_input.tick_labels = new_x_labels;
        x_input.tick_projection = x_projection;
        // Re-derive the explicit-range ordinal band centers from THIS panel's
        // scale (GH #39 phase 2): an explicit-range categorical facet axis places
        // its labels at the scale's band centers, matching the marks. `None` for
        // every other scale — keeps the global uniform-slot placement.
        x_input.categorical_positions = scales.x.explicit_band_centers();
        let x_label_fs = x_input
            .overrides
            .label_font_size
            .unwrap_or(theme.typography.label_font_size);
        let (new_x_layout, _warn) = crate::layout::axis::layout_x_axis(
            &x_input,
            panel.plot_area,
            panel_idx,
            x_label_fs,
            theme.typography.title_font_size,
            theme.padding.axis_title_padding,
            crate::layout::DEFAULT_CULL_THRESHOLD,
            theme.sizes.tick_size,
            facet_metrics,
        );
        Some(new_x_layout)
    } else {
        None
    };

    let independent_y = if y_independent {
        let y_tick_count = prep.axes.y.tick_labels.len().max(1);
        let mut y_input = prep.axes.y.clone();
        // Same `build_axis_tick_inputs` helper as the global/x path. `is_y =
        // true` carries the non-ordinal-y label/fraction reversal (high values
        // at the top of the inverted y pixel range) and the projection's
        // reversed fractions, in lockstep — so the carrier stays index-aligned
        // with the reversed labels. `Immediate` format mode; empty minor vec.
        let (new_y_labels, y_projection, _threaded) = super::prepare::build_axis_tick_inputs(
            &scales.y,
            y_tick_count,
            super::prepare::TickFormatMode::Immediate {
                format: y_fmt_spec.as_deref(),
                format_type: y_fmt_type.as_deref(),
            },
            true,
            Vec::new(),
        );
        y_input.tick_labels = new_y_labels;
        y_input.tick_projection = y_projection;
        // Same explicit-range ordinal band-center carrier as the x path above,
        // from THIS panel's y scale (GH #39 phase 2).
        y_input.categorical_positions = scales.y.explicit_band_centers();
        let y_label_fs = y_input
            .overrides
            .label_font_size
            .unwrap_or(theme.typography.label_font_size);
        let new_y_layout = crate::layout::axis::layout_y_axis(
            &y_input,
            panel.plot_area,
            panel_idx,
            y_label_fs,
            theme.typography.title_font_size,
            theme.padding.axis_title_padding,
            facet_metrics,
        );
        Some(new_y_layout)
    } else {
        None
    };

    PanelAxes {
        independent_x,
        independent_y,
        x_independent,
        y_independent,
    }
}

/// Routed axis + grid scene nodes for one panel (MOD-09).
///
/// `axes_below`/`grid` paint before the data marks; `axes_above` paints after
/// (the zindex >= 1 / `draws_above_marks()` case). `grid_above` reports whether
/// the grid block as a whole should follow an above-marks axis into the
/// annotation slot; the caller does the grid_below/grid_above split AFTER the
/// break-axis remap (which still needs to mutate the below-grid in place).
struct PanelAxisGrid {
    axes_below: Vec<SceneNode>,
    axes_above: Vec<SceneNode>,
    grid: Vec<SceneNode>,
    grid_above: bool,
}

/// Build and route this panel's grid + axis nodes (MOD-09 extraction of the
/// inline suppress/grid/route block). Pure extraction — same suppress test, same
/// `build_grid`, same `grid_above` computation, same above/below `route_axis`
/// dispatch, same independent-vs-shared axis selection, same polar-axis append.
#[allow(clippy::too_many_arguments)]
fn route_panel_axes_and_grid(
    spec: &ChartSpec,
    scales: &scale_resolve::ResolvedScales,
    panel: &crate::layout::PanelLayout,
    panel_axes_layout: &[&AxisLayout],
    panel_x_axis: Option<&AxisLayout>,
    panel_y_axis: Option<&AxisLayout>,
    // Secondary y-axes for this panel (secondary-y-axis, GH #52): one per
    // `independent_y` layer, orient Right, stacked outward beyond the
    // primary. Routed above/below marks the same way every other axis is
    // (via `draws_above_marks()`); never contributes gridlines — slot 0 (the
    // primary `panel_y_axis`) is the only gridline source, so these are never
    // passed to `build_grid`. Empty on the shared path — byte-identical.
    panel_secondary_y: &[&AxisLayout],
    x_independent: bool,
    y_independent: bool,
    theme: &ThemeInputs,
    chart_config: &super::chart_config::ChartConfig,
) -> PanelAxisGrid {
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
    let grid = if suppress_axes {
        Vec::new()
    } else {
        marks::axis::build_grid(panel.plot_area, panel_x_axis, panel_y_axis, theme, grid_band_colors)
    };
    // zindex (B5): gridlines follow their axis above/below the marks. When a
    // grid-bearing axis requests `zindex >= 1`, the whole grid block is routed
    // above the marks (into the annotation list) alongside that axis. Computed
    // here so the break-axis remap below still sees the grid before routing.
    let grid_above = !suppress_axes
        && [panel_x_axis, panel_y_axis]
            .into_iter()
            .flatten()
            .any(|a| a.show_grid && a.draws_above_marks());

    // Axes — draw from the effective (possibly per-panel) AxisLayout values.
    // `panel_x_axis` and `panel_y_axis` already point to either the
    // independent (per-panel) or the shared (global) layout. Emit them
    // first, then any additional axes (e.g. secondary Top/Right) from the
    // global layout that were not overridden.
    //
    // zindex (B5): an axis with `zindex >= 1` draws ABOVE the data marks. Its
    // nodes are routed into `axes_above` (appended to the panel's annotation
    // list, which the renderer emits after marks) instead of `axes_below`
    // (emitted before marks). `<= 0`/absent keeps the historical below-marks
    // behavior, so default output is byte-identical.
    let mut axes_below: Vec<SceneNode> = Vec::new();
    let mut axes_above: Vec<SceneNode> = Vec::new();
    let route_axis = |axis: &AxisLayout, above: &mut Vec<SceneNode>, below: &mut Vec<SceneNode>| {
        let nodes = marks::axis::build_axis(axis, theme);
        if axis.draws_above_marks() {
            above.extend(nodes);
        } else {
            below.extend(nodes);
        }
    };
    // Dual-axis scene contract (secondary-y-axis, GH #52 Task 8): when this
    // panel has one or more independent-y layers, every y-axis's nodes (the
    // primary left axis AND each stacked right axis) are wrapped in a
    // `SceneNode::Group` tagged with that axis's slot index (`y_slot` attr,
    // slot 0 = primary). This mirrors `MarkBatch::y_slot` and
    // `CoordKind::Cartesian::y_domains` — mesh, axis, and domain state all key
    // off the same slot number (spec §6) — so the interactive runtime can
    // relabel/rescale each axis from its own scale. Empty `panel_secondary_y`
    // (every existing chart) skips this wrapping entirely: y-axis nodes route
    // through the untagged `route_axis` exactly as before, byte-identical.
    let dual_axis = !panel_secondary_y.is_empty();
    let route_y_axis_slotted =
        |axis: &AxisLayout, slot: usize, above: &mut Vec<SceneNode>, below: &mut Vec<SceneNode>| {
            let nodes = marks::axis::build_axis(axis, theme);
            let tagged = vec![SceneNode::Group {
                attrs: vec![("y_slot".to_string(), slot.to_string())],
                children: nodes,
            }];
            if axis.draws_above_marks() {
                above.extend(tagged);
            } else {
                below.extend(tagged);
            }
        };
    if !suppress_axes {
        if x_independent || y_independent {
            // Emit the effective x and y axes (may be independent).
            if let Some(ax) = panel_x_axis {
                route_axis(ax, &mut axes_above, &mut axes_below);
            }
            if let Some(ay) = panel_y_axis {
                if dual_axis {
                    route_y_axis_slotted(ay, 0, &mut axes_above, &mut axes_below);
                } else {
                    route_axis(ay, &mut axes_above, &mut axes_below);
                }
            }
            // Also emit any other orientations (Top, Right) from the global
            // layout that are not covered by the independent overrides.
            for axis in panel_axes_layout {
                if !matches!(axis.orient,
                    crate::layout::AxisOrient::Bottom | crate::layout::AxisOrient::Top
                    | crate::layout::AxisOrient::Left | crate::layout::AxisOrient::Right)
                {
                    route_axis(axis, &mut axes_above, &mut axes_below);
                }
            }
        } else {
            for axis in panel_axes_layout {
                if dual_axis && matches!(axis.orient, crate::layout::AxisOrient::Left) {
                    // The primary y-axis (slot 0) in the shared/global layout.
                    route_y_axis_slotted(axis, 0, &mut axes_above, &mut axes_below);
                } else {
                    route_axis(axis, &mut axes_above, &mut axes_below);
                }
            }
        }
        // Secondary y-axes (secondary-y-axis, GH #52): emitted after the
        // primary x/y so they draw outward of (never occlude) the primary
        // axis. Same above/below zindex routing as every other axis; each one
        // is slot `i + 1` (slot 0 is always the primary, above).
        for (i, axis) in panel_secondary_y.iter().enumerate() {
            route_y_axis_slotted(axis, i + 1, &mut axes_above, &mut axes_below);
        }
    }

    // Polar axis: circular boundary + radial tick marks (replaces Cartesian axes)
    if matches!(&spec.coord, Some(crate::spec::coord::CoordKind::Polar { .. })) {
        let cx = panel.plot_area.x + panel.plot_area.w / 2.0;
        let cy = panel.plot_area.y + panel.plot_area.h / 2.0;
        let outer_r = polar_outer_radius(&panel.plot_area);
        axes_below.extend(build_polar_axes(cx, cy, outer_r, scales, theme));
    }

    PanelAxisGrid { axes_below, axes_above, grid, grid_above }
}

/// Build this panel's per-layer mark batches (MOD-09 extraction of the inline
/// mark-batch loop). Pure extraction — same position adjustment, same synthetic
/// per-layer ChartSpec, same polar node transform, same alignment guard, same
/// MarkBatch assembly. Returns the batches in layer order.
fn build_panel_mark_batches(
    spec: &ChartSpec,
    prep: &PreparedInputs,
    layer_batches: &[RecordBatch],
    scales: &scale_resolve::ResolvedScales,
    panel: &crate::layout::PanelLayout,
    theme: &ThemeInputs,
    warnings: &mut Vec<RenderWarning>,
) -> Result<Vec<MarkBatch>, RenderError> {
    let mut mark_batches: Vec<MarkBatch> = Vec::new();

    for (li, layer) in prep.layers.iter().enumerate() {
        let layer_batch = &layer_batches[li];
        if layer_batch.num_rows() == 0 {
            continue;
        }

        // Bind this layer to its y-slot (secondary-y-axis, GH #52). Shared layers
        // (and every layer on a chart with no independent slot) draw through the
        // primary `scales` unchanged — byte-stable. An independent layer draws
        // through a clone whose `.y` is its own slot scale, so mark geometry and
        // position adjustment map data → pixels through that layer's y-scale
        // without touching any mark renderer (all read `ctx.scales.y`).
        let layer_scales_owned: Option<scale_resolve::ResolvedScales> =
            if scales.y_slots.slot_for_layer(li) != 0 {
                let mut s = scales.clone();
                s.y = scales.y_for_layer(li).clone();
                Some(s)
            } else {
                None
            };
        let scales: &scale_resolve::ResolvedScales =
            layer_scales_owned.as_ref().unwrap_or(scales);

        // Position adjustment — always call apply_position; it is the
        // single authority for all adjustments (explicit layer.position
        // *and* encoding-level encoding.y.stack).  When neither is set
        // it returns a cheap reference-counted clone and is a no-op.
        let adjusted_owned = position::apply_position(
            layer_batch,
            layer.position.as_ref(),
            scales,
            &layer.encoding,
            prep.coord_flipped,
            warnings,
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
            scales,
            batch: layer_batch,
            mark_style: &mark_style,
        };

        validate_mark_encoding(&layer.mark, &layer.encoding)?;
        let mut result = draw::dispatch_mark_build(&layer.mark, &ctx);

        // For CoordPolar, transform all mark nodes from Cartesian pixel
        // space to polar pixel space. Arc marks (Mark::Arc) handle their
        // own polar geometry and must not be transformed again.  Bars under
        // CoordPolar route through `build_polar`, which also generates
        // arc-geometry nodes (MarkBatchKind::Arc) in polar space — those
        // must likewise be excluded from the transform, or the wedge
        // coordinates are corrupted by a second polar projection.
        let is_arc_geometry = matches!(result.kind, ferrum_scene::MarkBatchKind::Arc);
        if matches!(&spec.coord, Some(crate::spec::coord::CoordKind::Polar { .. }))
            && !matches!(layer.mark, crate::spec::mark::Mark::Arc)
            && !is_arc_geometry
        {
            apply_polar_node_transform(&mut result.nodes, &panel.plot_area);
        }

        let keys = extract_keys(&layer.encoding, layer_batch, result.data_indices.as_deref());
        // #6 alignment guard (spec §7): each present per-node vector must
        // have exactly one entry per node. All five channels — tooltips,
        // hrefs, descriptions, data_indices, and keys — are independent and
        // checked independently so any misaligned channel trips the guard
        // under a debug build. data_indices and keys are the alignment
        // vector and key channel; covering them here catches builders (like
        // the pre-fix label.rs) that diverge data_indices while leaving
        // metadata None (which the three-channel guard could not detect).
        crate::render::mark_nodes::debug_assert_nodes_metadata_aligned(
            result.nodes.len(),
            result.tooltips.as_ref().map(|t| t.len()),
            result.hrefs.as_ref().map(|h| h.len()),
            result.descriptions.as_ref().map(|d| d.len()),
            result.data_indices.as_ref().map(|d| d.len()),
            keys.as_ref().map(|k| k.len()),
        );
        mark_batches.push(MarkBatch {
            kind: result.kind,
            nodes: result.nodes,
            data_indices: result.data_indices,
            tooltips: result.tooltips,
            hrefs: result.hrefs,
            descriptions: result.descriptions,
            keys,
            blend: layer.blend.unwrap_or(BlendMode::Normal),
            stroke_cap: mark_style.line.stroke_cap.as_deref().and_then(draw::parse_stroke_cap),
            stroke_join: mark_style.line.stroke_join.as_deref().and_then(draw::parse_stroke_join),
            packed_instances: None,
            // Secondary-y-axis (GH #52 Task 8): tag this batch with the same
            // slot its marks were positioned through above. `0` on every
            // shared-path layer (the byte-stable default, omitted from JSON).
            y_slot: scales.y_slots.slot_for_layer(li),
        });
    }

    Ok(mark_batches)
}

fn resolve_param_domains(spec: &mut ChartSpec) {
    if spec.params.is_empty() {
        return;
    }
    let store = crate::spec::parameter::ParamStore::new(&spec.params);
    if store.is_empty() {
        return;
    }
    let enc = &mut spec.encoding;
    for channel in [
        enc.x.as_mut(),
        enc.y.as_mut(),
        enc.color.as_mut(),
        enc.size.as_mut(),
        enc.opacity.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        let Some(scale) = channel.scale.as_mut() else { continue };
        let Some(name) = scale.domain_param().map(str::to_owned) else { continue };
        if let Some(domain) = store.numeric_domain(&name) {
            scale.set_domain(domain);
        }
        // else: leave domain = None → auto-infer (empty-selection semantics).
    }
}

/// Collect param→scene bindings (D6, 5e-2a) from the original spec.
///
/// The static resolver substitutes `domainParam` into a concrete domain and
/// clears the reference before the scene exists, so the emitted scene has no
/// record of which panel/scale a param drives. This walks the original spec
/// (which still carries the markers) and emits one binding per (param, panel,
/// channel) connection:
///
/// - **Domain:** each `{x,y,color,size,opacity}` encoding scale with a
///   `domainParam` → one binding per panel, with the channel wire name.
/// - **Filter:** each `filter` transform carrying a `param` → one binding per
///   panel (channel `None`).
/// - **Legend:** each declared parameter whose `bind` is the string `"legend"`
///   → one panel-free binding (the selection name).
///
/// A `domainParam` on an `independent_y` layer's `y` encoding (secondary-y-axis,
/// GH #52) additionally emits one binding per panel tagged with that layer's
/// y-slot (`y_slots.slot_for_layer` — Task 2's contract, never re-derived), so
/// the WASM runtime rescales only that layer's marks. Charts with no
/// independent-y layer skip this walk entirely, so their bindings are
/// byte-identical to the pre-#52 chart-level-only collection.
///
/// Returns an empty vec when no markers apply, preserving param-free
/// byte-stability.
fn collect_param_bindings(
    spec: &ChartSpec,
    layers: &[super::prepare::LayerPrepared],
    y_slots: &scale_resolve::YScaleSlots,
    n_panels: usize,
) -> Vec<ParamBinding> {
    use crate::transform::core::TransformSpec;

    let mut bindings: Vec<ParamBinding> = Vec::new();
    let panel_count = n_panels.max(1);

    // Domain bindings: walk the positional/visual continuous channels.
    let enc = &spec.encoding;
    let channels: [(&str, Option<&crate::spec::encoding::EncodingSpec>); 5] = [
        ("x", enc.x.as_ref()),
        ("y", enc.y.as_ref()),
        ("color", enc.color.as_ref()),
        ("size", enc.size.as_ref()),
        ("opacity", enc.opacity.as_ref()),
    ];
    for (wire_name, channel) in channels {
        let Some(channel) = channel else { continue };
        let Some(scale) = channel.scale.as_ref() else { continue };
        let Some(param) = scale.domain_param() else { continue };
        for panel in 0..panel_count {
            bindings.push(ParamBinding {
                param: param.to_owned(),
                role: BindingRole::Domain,
                panel: Some(panel),
                channel: Some(wire_name.to_owned()),
                y_slot: 0,
            });
        }
    }

    // Secondary-y (#52): a `domainParam` on an `independent_y` layer's own `y`
    // encoding drives that layer's right-axis scale, not the shared primary. The
    // chart-level walk above only sees the primary/left `y`, so these bindings
    // are additive and only ever produced by dual-axis charts (charts with no
    // `independent_y` layer never enter this loop → byte-identical wire). Each
    // carries the layer's slot so the runtime routes the rescale into that
    // slot's affine, moving only that layer's marks.
    for (layer_idx, layer) in layers.iter().enumerate() {
        if !layer.independent_y {
            continue;
        }
        let Some(y_channel) = layer.encoding.y.as_ref() else { continue };
        let Some(scale) = y_channel.scale.as_ref() else { continue };
        let Some(param) = scale.domain_param() else { continue };
        let slot = y_slots.slot_for_layer(layer_idx);
        for panel in 0..panel_count {
            bindings.push(ParamBinding {
                param: param.to_owned(),
                role: BindingRole::Domain,
                panel: Some(panel),
                channel: Some("y".to_owned()),
                y_slot: slot,
            });
        }
    }

    // Filter bindings: each filter transform carrying a `param` marker.
    for transform in &spec.transforms {
        if let TransformSpec::Filter(filter) = transform {
            let Some(param) = filter.param.as_ref() else { continue };
            for panel in 0..panel_count {
                bindings.push(ParamBinding {
                    param: param.clone(),
                    role: BindingRole::Filter,
                    panel: Some(panel),
                    channel: None,
                    y_slot: 0,
                });
            }
        }
    }

    // Legend bindings: a selection declared with `bind="legend"`.
    for param in &spec.params {
        if matches!(&param.bind, Some(serde_json::Value::String(s)) if s == "legend") {
            bindings.push(ParamBinding {
                param: param.name.clone(),
                role: BindingRole::Legend,
                panel: None,
                channel: None,
                y_slot: 0,
            });
        }
    }

    bindings
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
        .unwrap_or(theme.typography.title_font_size);
    let resolved_font_weight: String = title_spec
        .and_then(|t| t.font_weight.clone())
        .unwrap_or_else(|| theme.typography.title_font_weight.clone());
    let resolved_color = title_spec
        .and_then(|t| t.color.as_deref())
        .and_then(|hex| super::color::from_hex_str(hex).ok())
        .unwrap_or(theme.colors.title_color);
    let fw = if resolved_font_weight == "normal" { None } else { Some(resolved_font_weight.as_str()) };
    out.push(SceneNode::Text {
        x: title.x,
        y: title.y,
        content: title.text.clone(),
        style: to_scene_text_style(
            resolved_color, resolved_font_size, title.anchor, 0.0,
            &theme.typography.title_font_family, fw, None, 1.0,
        ),
    });
    if let (Some(subtitle), Some(sy)) = (&title.subtitle, title.subtitle_y) {
        let resolved_sub_color = title_spec
            .and_then(|t| t.subtitle_color.as_deref())
            .and_then(|hex| super::color::from_hex_str(hex).ok())
            .or(theme.colors.subtitle_color)
            .unwrap_or(theme.colors.font_color);
        let resolved_sub_font_size = title_spec
            .and_then(|t| t.subtitle_font_size)
            .or(theme.typography.subtitle_font_size)
            .unwrap_or(resolved_font_size * 0.85);
        out.push(SceneNode::Text {
            x: title.x,
            y: sy,
            content: subtitle.clone(),
            style: to_scene_text_style(
                resolved_sub_color, resolved_sub_font_size, title.anchor, 0.0,
                &theme.typography.font_family, None, None, 1.0,
            ),
        });
    }
}

/// Resolve the color scale a legend draws against, from a leaf's prepared
/// inputs (`prep.layers[0].encoding` is the rendering encoding; the color scale
/// is re-resolved over the leaf's final batch, then any chart-level
/// `configure_color` override applied). Returns `None` when the leaf encodes no
/// color. Shared by [`build_legend_decorations`] (per-panel legend draw) and the
/// composite figure-legend band ([`super::composite_render`]), so both build the
/// legend's color scale through one code path (design §7 facet parity: the
/// figure legend reuses the same primitives, not a parallel implementation).
pub(crate) fn resolve_legend_color_scale(
    spec: &ChartSpec,
    prep: &PreparedInputs,
    theme: &ThemeInputs,
    chart_config: &super::chart_config::ChartConfig,
) -> Result<Option<scale_resolve::ColorScale>, RenderError> {
    let mut rendering_spec_for_legend = ChartSpec {
        encoding: prep.layers[0].encoding.clone(),
        ..spec.clone()
    };
    resolve_param_domains(&mut rendering_spec_for_legend);
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
    Ok(color_scale)
}

fn build_legend_decorations(
    layout: &LayoutResult,
    spec: &ChartSpec,
    prep: &PreparedInputs,
    theme: &ThemeInputs,
    chart_config: &super::chart_config::ChartConfig,
    out: &mut Vec<SceneNode>,
) -> Result<(), RenderError> {
    // Nothing to draw when neither a color legend nor any size/shape block is
    // present.
    if layout.legend.is_none() && layout.aux_legends.is_empty() {
        return Ok(());
    }
    let color_scale = resolve_legend_color_scale(spec, prep, theme, chart_config)?;
    if let Some(legend) = &layout.legend {
        out.extend(marks::legend::build_legend(legend, color_scale.as_ref(), theme));
    }
    // Auxiliary (size / shape) legend blocks stacked beneath the color legend.
    // Each carries its own per-entry color (color_hex) or falls back to the
    // theme mark color, so the color scale is unused but passed for uniformity.
    for aux in &layout.aux_legends {
        out.extend(marks::legend::build_legend(aux, color_scale.as_ref(), theme));
    }
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

    let tick_levels_for = |scale: &scale_resolve::ScaleKind| -> Vec<TickLevel> {
        ZOOM_BREAKPOINTS
            .iter()
            .map(|&(min_z, max_z, count)| TickLevel {
                min_zoom: min_z,
                max_zoom: max_z,
                ticks: scale.tick_data(count),
            })
            .collect()
    };

    let y_levels = tick_levels_for(&scales.y);

    // Secondary-y (#52): one tick-level list per right axis, generated from the
    // SAME `y_slots` ScaleKinds the axes and marks resolved against (one
    // resolution site, spec §6). `slots()[0]` mirrors the primary `y` already
    // emitted as `y_levels`, so skip it. Empty on the shared path → the
    // `skip_serializing_if` on `y_slot_levels` keeps the blob byte-identical.
    let y_slot_levels: Vec<Vec<TickLevel>> = scales
        .y_slots
        .slots()
        .iter()
        .skip(1)
        .map(tick_levels_for)
        .collect();

    PanelTickLevels {
        panel_id: panel_idx,
        x_levels,
        y_levels,
        y_slot_levels,
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

    let axis_color = draw::to_scene_color(theme.colors.axis_line_color);
    let stroke = ferrum_scene::StrokeStyle {
        color: axis_color,
        width: theme.sizes.axis_line_width,
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
            style: ferrum_scene::FillStroke { fill: None, stroke: Some(axis_color), stroke_width: theme.sizes.axis_line_width, opacity: 1.0, stroke_dash: None, stroke_opacity: 1.0, fill_opacity: 1.0, angle: 0.0 },
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
                    theme.colors.label_color, theme.typography.label_font_size,
                    crate::layout::TextAnchor::Middle, 0.0,
                    &theme.typography.font_family, None, None, 1.0,
                ),
            });
        }
    }

    nodes
}

// ── Structural feature processing ────────────────────────────────────────────

/// Process structural feature specs (axis breaks, insets).
///
/// Returns four values:
/// - `extra_axes` — additional axis scene nodes
/// - `extra_mark_batches` — additional mark batches
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
    scales: &scale_resolve::ResolvedScales,
    plot_area: &crate::layout::Rect,
    theme: &crate::layout::ThemeInputs,
) -> StructuralOutput {
    let extra_axes: Vec<SceneNode> = Vec::new();
    let extra_mark_batches: Vec<ferrum_scene::MarkBatch> = Vec::new();
    let mut extra_annotations: Vec<SceneNode> = Vec::new();
    let mut break_results: Vec<(String, break_axis::BreakResult)> = Vec::new();

    for item in structural {
        match item {
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

// The per-panel independent-axis `TickProjection` rebuild (formerly the
// `build_independent_{x,y}_projection` helpers here) is now derived through the
// shared `prepare::build_axis_tick_inputs`, which the global/shared axis path
// also drives — so the tick-derivation sequence (labels → non-ordinal-y reverse
// → format → fraction projection) lives in exactly one place.

/// Validate that the encoding channels supplied to a mark are a supported
/// combination for that mark type.  Called before `dispatch_mark_build` so that
/// unsupported combinations surface as clear errors rather than silently
/// producing wrong or empty geometry.
///
/// Currently enforced:
/// - `mark_area` with `x2` bound: `x2` is not a documented area channel.
///   Horizontal-band areas belong to `mark_rect`; vertical bands use `y2`.
/// - `mark_bar` with both `x2` AND `y2` bound: a 2-D extent (width AND height
///   from separate columns) defines a rectangle, not a bar.  Use `mark_rect`.
fn validate_mark_encoding(
    mark: &crate::spec::mark::Mark,
    encoding: &crate::spec::encoding::Encoding,
) -> Result<(), RenderError> {
    use crate::spec::mark::Mark;
    match mark {
        Mark::Area if encoding.x2.is_some() => {
            Err(RenderError::UnsupportedChannelCombination {
                mark: "mark_area",
                channel: "x2",
                hint: "use y2= for a vertical band area, or use mark_rect for a 2-D extent",
            })
        }
        Mark::Bar if encoding.x2.is_some() && encoding.y2.is_some() => {
            Err(RenderError::UnsupportedChannelCombination {
                mark: "mark_bar",
                channel: "x2 and y2",
                hint: "a 2-D extent (both x2= and y2=) is a rectangle; use mark_rect instead",
            })
        }
        _ => Ok(()),
    }
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

    // ── D6 reactive-rescale static resolver ──────────────────────────────

    use crate::spec::encoding::{ContinuousScaleCommon, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;
    use ferrum_scene::{ParamKind, ParameterSpec};

    fn linear_domain_param(name: &str) -> ScaleSpec {
        ScaleSpec::Linear {
            common: ContinuousScaleCommon {
                domain: None,
                range: None,
                clamp: false,
                padding: None,
                scheme: None,
                domain_param: Some(name.to_string()),
            },
            nice: false,
            zero: false,
        }
    }

    fn spec_with_x_domain_param(params: Vec<ParameterSpec>) -> ChartSpec {
        let mut spec = ChartSpec {
            data: crate::spec::data_ref::DataRef::default(),
            mark: Mark::Point,
            encoding: crate::spec::encoding::Encoding {
                x: Some(EncodingSpec {
                    field: "v".into(),
                    scale: Some(linear_domain_param("d")),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        spec.params = params;
        spec
    }

    fn x_domain(spec: &ChartSpec) -> Option<Vec<f64>> {
        match spec.encoding.x.as_ref()?.scale.as_ref()? {
            ScaleSpec::Linear { common, .. } => common.domain.clone(),
            _ => None,
        }
    }

    #[test]
    fn resolve_param_domains_substitutes_variable_array() {
        let mut spec = spec_with_x_domain_param(vec![ParameterSpec {
            name: "d".into(),
            kind: ParamKind::Variable,
            value: Some(serde_json::json!([10, 20])),
            bind: None,
            select: None,
        }]);
        resolve_param_domains(&mut spec);
        assert_eq!(x_domain(&spec), Some(vec![10.0, 20.0]));
        // domain_param cleared after substitution.
        assert_eq!(spec.encoding.x.unwrap().scale.unwrap().domain_param(), None);
    }

    #[test]
    fn resolve_param_domains_no_matching_param_leaves_auto() {
        // domainParam "d" referenced, but no such param declared → domain stays
        // None (auto-infer), and the reference is left in place.
        let mut spec = spec_with_x_domain_param(vec![ParameterSpec {
            name: "other".into(),
            kind: ParamKind::Variable,
            value: Some(serde_json::json!([1, 2])),
            bind: None,
            select: None,
        }]);
        resolve_param_domains(&mut spec);
        assert_eq!(x_domain(&spec), None);
    }

    #[test]
    fn resolve_param_domains_selection_leaves_auto() {
        // A selection (interval) yields no static numeric domain → auto-infer.
        let mut spec = spec_with_x_domain_param(vec![ParameterSpec {
            name: "d".into(),
            kind: ParamKind::Interval,
            value: None,
            bind: None,
            select: None,
        }]);
        resolve_param_domains(&mut spec);
        assert_eq!(x_domain(&spec), None);
    }

    #[test]
    fn resolve_param_domains_noop_when_no_params() {
        // The byte-stability gate: empty params → spec unchanged.
        let mut spec = spec_with_x_domain_param(Vec::new());
        let before = serde_json::to_string(&spec).unwrap();
        resolve_param_domains(&mut spec);
        let after = serde_json::to_string(&spec).unwrap();
        assert_eq!(before, after);
    }

    // ── D6 param→scene bindings (5e-2a) ──────────────────────────────────

    use ferrum_scene::BindingRole;

    #[test]
    fn collect_param_bindings_emits_domain_binding() {
        // An x-scale domainParam → a Domain binding on panel 0, channel "x".
        let spec = spec_with_x_domain_param(vec![ParameterSpec {
            name: "d".into(),
            kind: ParamKind::Variable,
            value: Some(serde_json::json!([0, 100])),
            bind: None,
            select: None,
        }]);
        let bindings = collect_param_bindings(&spec, &[], &scale_resolve::YScaleSlots::default(), 1);
        assert_eq!(bindings.len(), 1);
        let b = &bindings[0];
        assert_eq!(b.param, "d");
        assert_eq!(b.role, BindingRole::Domain);
        assert_eq!(b.panel, Some(0));
        assert_eq!(b.channel.as_deref(), Some("x"));
    }

    #[test]
    fn collect_param_bindings_domain_per_panel_for_facets() {
        let spec = spec_with_x_domain_param(vec![ParameterSpec {
            name: "d".into(),
            kind: ParamKind::Variable,
            value: Some(serde_json::json!([0, 100])),
            bind: None,
            select: None,
        }]);
        let bindings = collect_param_bindings(&spec, &[], &scale_resolve::YScaleSlots::default(), 3);
        assert_eq!(bindings.len(), 3);
        assert_eq!(
            bindings.iter().filter_map(|b| b.panel).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(bindings
            .iter()
            .all(|b| b.role == BindingRole::Domain && b.channel.as_deref() == Some("x")));
    }

    #[test]
    fn collect_param_bindings_emits_filter_binding() {
        let mut spec = spec_with_x_domain_param(Vec::new());
        // Drop the domainParam so only the filter contributes.
        spec.encoding.x = None;
        spec.transforms = vec![crate::transform::core::TransformSpec::Filter(
            crate::transform::filter::FilterSpec {
                predicate: "true".into(),
                name: None,
                param: Some("brush".into()),
            },
        )];
        let bindings = collect_param_bindings(&spec, &[], &scale_resolve::YScaleSlots::default(), 1);
        assert_eq!(bindings.len(), 1);
        let b = &bindings[0];
        assert_eq!(b.param, "brush");
        assert_eq!(b.role, BindingRole::Filter);
        assert_eq!(b.panel, Some(0));
        assert_eq!(b.channel, None);
    }

    #[test]
    fn collect_param_bindings_emits_legend_binding() {
        let mut spec = spec_with_x_domain_param(vec![ParameterSpec {
            name: "sel".into(),
            kind: ParamKind::Point,
            value: None,
            bind: Some(serde_json::json!("legend")),
            select: None,
        }]);
        spec.encoding.x = None;
        let bindings = collect_param_bindings(&spec, &[], &scale_resolve::YScaleSlots::default(), 1);
        assert_eq!(bindings.len(), 1);
        let b = &bindings[0];
        assert_eq!(b.param, "sel");
        assert_eq!(b.role, BindingRole::Legend);
        assert_eq!(b.panel, None);
        assert_eq!(b.channel, None);
    }

    #[test]
    fn collect_param_bindings_empty_for_param_free_spec() {
        let spec = spec_with_x_domain_param(Vec::new());
        // spec_with_x_domain_param sets an x domainParam "d", but param-free
        // here means no params declared; the marker still produces a Domain
        // binding because the reference exists. Strip it to assert true emptiness.
        let mut bare = spec;
        bare.encoding.x = None;
        assert!(collect_param_bindings(&bare, &[], &scale_resolve::YScaleSlots::default(), 1).is_empty());
    }

    /// Build a minimal `LayerPrepared` carrying a `y` domainParam scale.
    fn layer_with_y_domain_param(name: &str, independent_y: bool) -> crate::render::prepare::LayerPrepared {
        crate::render::prepare::LayerPrepared {
            mark: Mark::Line,
            encoding: crate::spec::encoding::Encoding {
                y: Some(EncodingSpec {
                    field: "w".into(),
                    scale: Some(linear_domain_param(name)),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            independent_y,
        }
    }

    #[test]
    fn collect_param_bindings_independent_y_layer_carries_slot() {
        // Secondary-y (#52): a domainParam on an `independent_y` layer's y
        // encoding emits a Domain binding tagged with that layer's slot, so the
        // WASM runtime rescales only that layer's marks.
        let mut spec = spec_with_x_domain_param(Vec::new());
        spec.encoding.x = None; // isolate the layer binding.
        let layers = [
            layer_with_y_domain_param("primary", false),
            layer_with_y_domain_param("d2", true),
        ];
        // slot_for_layer only reads `layer_slot`; an empty `slots` list is
        // sufficient for this isolated collection test.
        let y_slots = scale_resolve::YScaleSlots::new(Vec::new(), vec![0, 1]);
        let bindings = collect_param_bindings(&spec, &layers, &y_slots, 1);
        // Layer 0 is not independent → skipped; only the independent layer emits.
        assert_eq!(bindings.len(), 1);
        let b = &bindings[0];
        assert_eq!(b.param, "d2");
        assert_eq!(b.role, BindingRole::Domain);
        assert_eq!(b.channel.as_deref(), Some("y"));
        assert_eq!(b.panel, Some(0));
        assert_eq!(b.y_slot, 1);
    }

    #[test]
    fn collect_param_bindings_shared_y_layers_emit_no_slot_bindings() {
        // Byte-stability gate: with no `independent_y` layer, the per-layer walk
        // emits nothing — bindings are identical to the pre-#52 chart-level pass.
        let mut spec = spec_with_x_domain_param(Vec::new());
        spec.encoding.x = None;
        let layers = [
            layer_with_y_domain_param("a", false),
            layer_with_y_domain_param("b", false),
        ];
        let bindings =
            collect_param_bindings(&spec, &layers, &scale_resolve::YScaleSlots::default(), 1);
        assert!(bindings.is_empty());
    }

    #[test]
    fn collect_param_bindings_independent_y_layer_per_panel() {
        // The slot binding fans out one-per-panel like the chart-level domain
        // bindings, so faceted dual-axis charts route every panel.
        let mut spec = spec_with_x_domain_param(Vec::new());
        spec.encoding.x = None;
        let layers = [
            layer_with_y_domain_param("primary", false),
            layer_with_y_domain_param("d2", true),
        ];
        let y_slots = scale_resolve::YScaleSlots::new(Vec::new(), vec![0, 1]);
        let bindings = collect_param_bindings(&spec, &layers, &y_slots, 3);
        assert_eq!(bindings.len(), 3);
        assert_eq!(
            bindings.iter().filter_map(|b| b.panel).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(bindings.iter().all(|b| b.y_slot == 1 && b.channel.as_deref() == Some("y")));
    }

    fn layout_with_subtitle(subtitle: &str) -> LayoutResult {
        LayoutResult {
            viewport: Rect { x: 0.0, y: 0.0, w: 400.0, h: 300.0 },
            panels: Vec::new(),
            axes: Vec::new(),
            legend: None,
            aux_legends: Vec::new(),
            chart_title: Some(crate::layout::ChartTitleLayout {
                text: "Main Title".to_string(),
                subtitle: Some(subtitle.to_string()),
                x: 10.0,
                y: 20.0,
                subtitle_y: Some(36.0),
                anchor: crate::layout::TextAnchor::Start,
            }),
            warnings: Vec::new(),
            secondary_y_axes: Vec::new(),
        }
    }

    /// `configure_title(subtitle_font_size=…, subtitle_color=…)` flows through the
    /// chart-config → theme path and reaches the rendered subtitle. The chart-level
    /// subtitle styling lives on the theme (populated by `apply_chart_config`); the
    /// per-chart `spec.title` is `None` here, exactly the `configure_title` case.
    #[test]
    fn build_title_applies_chart_config_subtitle_styling() {
        let spec = spec_with_x_domain_param(Vec::new());
        assert!(spec.title.is_none(), "this test exercises the chart-config path, not spec.title");
        let layout = layout_with_subtitle("Styled subtitle");

        let mut theme = ThemeInputs::default();
        theme.typography.subtitle_font_size = Some(22.0);
        theme.colors.subtitle_color = Some(super::super::color::from_hex_str("#ff0000").unwrap());

        let mut nodes = Vec::new();
        build_title(&layout, &spec, &theme, &mut nodes);

        // [0] = title line, [1] = subtitle line.
        let subtitle_node = nodes
            .iter()
            .find_map(|n| match n {
                SceneNode::Text { content, style, .. } if content == "Styled subtitle" => Some(style),
                _ => None,
            })
            .expect("subtitle text node must be emitted");
        assert_eq!(subtitle_node.font_size, 22.0);
        assert_eq!(
            subtitle_node.color,
            to_scene_color(super::super::color::from_hex_str("#ff0000").unwrap()),
        );
    }

    /// Unset subtitle config → byte-identical default: font_color and 0.85× title size.
    #[test]
    fn build_title_subtitle_defaults_unchanged_when_unset() {
        let spec = spec_with_x_domain_param(Vec::new());
        let layout = layout_with_subtitle("Default subtitle");
        let theme = ThemeInputs::default();

        let mut nodes = Vec::new();
        build_title(&layout, &spec, &theme, &mut nodes);

        let subtitle_node = nodes
            .iter()
            .find_map(|n| match n {
                SceneNode::Text { content, style, .. } if content == "Default subtitle" => Some(style),
                _ => None,
            })
            .expect("subtitle text node must be emitted");
        assert_eq!(subtitle_node.font_size, theme.typography.title_font_size * 0.85);
        assert_eq!(subtitle_node.color, to_scene_color(theme.colors.font_color));
    }

    // ── SPINE-04: resolve_panel_scales per-panel seam ────────────────────────

    /// The per-panel scale seam (`resolve_panel_scales`) resolves each panel's
    /// scales over THAT panel's pixel range, so two panels with different
    /// `plot_area` widths get different resolved x pixel ranges. This pins the
    /// provisional/final duality the finding calls out as REAL: the prepare
    /// provisional pass is for tick labels only; per-panel ranges differ and must
    /// be resolved fresh here. (A regression that accidentally unified the two
    /// passes — e.g. by reusing `prep.provisional_scales` — would make both panels
    /// share one pixel range and fail this assertion.)
    #[test]
    fn resolve_panel_scales_differs_per_panel_pixel_range() {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use std::sync::Arc;

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            ],
        )
        .unwrap();

        let theme = ThemeInputs::default();
        let prep = super::super::prepare::prepare_render_inputs(&spec, &batch, &theme, None).unwrap();
        let chart_config = super::super::chart_config::ChartConfig::default();

        // Two panels with deliberately different plot_area widths.
        let panel_narrow = crate::layout::PanelLayout {
            plot_area: crate::layout::Rect { x: 0.0, y: 0.0, w: 100.0, h: 200.0 },
            ..Default::default()
        };
        let panel_wide = crate::layout::PanelLayout {
            plot_area: crate::layout::Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 },
            ..Default::default()
        };

        let mut warnings = Vec::new();
        // One implicit layer → one layer batch (the whole panel batch).
        let layer_batches = vec![batch.clone()];
        let (_spec_a, scales_a) = resolve_panel_scales(
            &spec, &prep, &panel_narrow, &batch, &layer_batches, &theme, &chart_config, &mut warnings, None,
        )
        .unwrap();
        let (_spec_b, scales_b) = resolve_panel_scales(
            &spec, &prep, &panel_wide, &batch, &layer_batches, &theme, &chart_config, &mut warnings, None,
        )
        .unwrap();

        let (a_lo, a_hi) = scales_a.x.pixel_range();
        let (b_lo, b_hi) = scales_b.x.pixel_range();
        // Each panel's x range spans its own plot_area width.
        assert!((a_hi - a_lo).abs() <= 100.0 + 1e-6, "narrow panel x range within 100px");
        assert!((b_hi - b_lo).abs() > 100.0 + 1e-6, "wide panel x range exceeds 100px");
        // The two panels do NOT share a pixel range — the per-panel resolution is real.
        assert!(
            (a_hi - a_lo - (b_hi - b_lo)).abs() > 1e-6,
            "per-panel x pixel ranges must differ ({a_lo}..{a_hi} vs {b_lo}..{b_hi})"
        );
    }

    // ── #52 Task 2: per-slot y-scale resolution ──────────────────────────────

    /// Build a two-layer ChartSpec sharing x, with layer 0 on `y0` and layer 1
    /// on `y1`, and a batch carrying `x`/`y0`/`y1`. `layer1_independent` sets the
    /// flag on the appended layer. Returns `(spec, batch)`.
    fn two_layer_dual_y_spec(layer1_independent: bool) -> (ChartSpec, RecordBatch) {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::layer::Layer;
        use std::sync::Arc;

        let y_enc = |field: &str| Layer {
            mark: Mark::Line,
            encoding: Encoding {
                y: Some(EncodingSpec { field: field.into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false,
        };
        let mut layer1 = y_enc("y1");
        layer1.independent_y = layer1_independent;

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: Some(vec![y_enc("y0"), layer1]),
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y0", DataType::Float64, false),
            Field::new("y1", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                // y0 ∈ [1,3] (small); y1 ∈ [100,300] (large) — clearly separable.
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
            ],
        )
        .unwrap();
        (spec, batch)
    }

    // ── #52 Task 10f: per-layer tooltip metadata ─────────────────────────────

    /// Variant of [`two_layer_dual_y_spec`] (always `layer1_independent =
    /// true`) that additionally sets `tooltip_fields` overrides at the chart
    /// level and on each layer's own encoding, so tests can exercise the
    /// per-layer tooltip-metadata contract without duplicating the whole
    /// two-layer fixture. `None` leaves that encoding's `tooltip_fields` unset
    /// (so it inherits per [`crate::spec::encoding::Encoding::inherit_from`]).
    fn two_layer_dual_y_spec_with_tooltips(
        chart_tooltip_field: Option<&str>,
        layer0_tooltip_field: Option<&str>,
        layer1_tooltip_field: Option<&str>,
    ) -> (ChartSpec, RecordBatch) {
        use crate::spec::encoding::EncodingSpec;

        let (mut spec, batch) = two_layer_dual_y_spec(true);
        let mk_fields = |f: Option<&str>| {
            f.map(|field| vec![EncodingSpec { field: field.into(), ..Default::default() }])
        };
        spec.encoding.tooltip_fields = mk_fields(chart_tooltip_field);
        let layers = spec.layers.as_mut().expect("two_layer_dual_y_spec always sets layers");
        layers[0].encoding.tooltip_fields = mk_fields(layer0_tooltip_field);
        layers[1].encoding.tooltip_fields = mk_fields(layer1_tooltip_field);
        (spec, batch)
    }

    /// Run the full `prepare → layout → build_scene` pipeline for an
    /// arbitrary `(spec, batch)` pair. Generalizes [`build_dual_y_scene`]
    /// (which is hardcoded to [`two_layer_dual_y_spec`]) for tests that need
    /// a customized spec.
    fn build_scene_for(spec: &ChartSpec, batch: &RecordBatch) -> ferrum_scene::SceneGraph {
        let theme = ThemeInputs::default();
        let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
        let config = super::super::config::RenderConfig::default();
        let chart_config = super::super::chart_config::ChartConfig::default();

        let prep = super::super::prepare::prepare_render_inputs(spec, batch, &theme, None).unwrap();
        let mut warnings = prep.warnings.clone();
        let metrics = super::super::font::FontdueMetrics::new();
        let layout = crate::layout::compute_layout(
            spec, &theme, viewport,
            &prep.axes, &prep.facet_groups, &prep.legend_entries,
            prep.legend_title.clone(), prep.colorbar.as_ref(), &metrics,
            &crate::layout::legend::LegendOverrides::default(), &prep.aux_legends,
            crate::layout::legend::LegendSuppression::default(),
        ).unwrap();

        build_scene(spec, &prep, &layout, &theme, &config, &mut warnings, &chart_config, None).unwrap()
    }

    /// A non-primary layer's mark batch must carry ITS OWN tooltip_fields —
    /// not the chart-level (auto-injected-from-the-primary-layer) fields.
    /// Regression test for GH #52 Task 10f: hovering the secondary layer used
    /// to show the primary layer's tooltip content
    /// (`.git/sdd/task-10-report.md` BUG FINDING #2). Each layer's own
    /// `tooltip_fields`, once Python emits them per layer, already flows
    /// through `LayerPrepared::from_chart_and_layer`'s
    /// `Encoding::inherit_from` merge and the per-layer synthetic `ChartSpec`
    /// built in `build_panel_mark_batches` — this test locks that contract in.
    #[test]
    fn layer_tooltip_metadata_uses_each_layers_own_fields() {
        let (spec, batch) =
            two_layer_dual_y_spec_with_tooltips(Some("y0"), Some("y0"), Some("y1"));
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        assert_eq!(panel.marks.len(), 2, "one mark batch per layer");

        let layer0 = panel.marks[0].tooltips.as_ref().expect("layer 0 must have tooltips");
        assert_eq!(layer0[0].fields.len(), 1);
        assert_eq!(layer0[0].fields[0].name, "y0");
        assert_eq!(layer0[0].fields[0].value, "1");

        let layer1 = panel.marks[1].tooltips.as_ref().expect("layer 1 must have tooltips");
        assert_eq!(layer1[0].fields.len(), 1);
        assert_eq!(
            layer1[0].fields[0].name, "y1",
            "layer 1's tooltip must report its OWN field (y1), not the primary layer's (y0)"
        );
        assert_eq!(layer1[0].fields[0].value, "100");
    }

    /// When neither layer carries its own `tooltip_fields` (legacy /
    /// auto-injected chart-level-only tooltips), every layer must still fall
    /// back to the chart-level `tooltip_fields` — today's
    /// `Encoding::inherit_from` behavior, preserved unchanged by the Task 10f
    /// fix (which only adds per-layer *preference*, never removes the
    /// fallback).
    #[test]
    fn layer_without_own_tooltip_fields_falls_back_to_chart_level() {
        let (spec, batch) = two_layer_dual_y_spec_with_tooltips(Some("y0"), None, None);
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];

        for (i, mark_batch) in panel.marks.iter().enumerate() {
            let tooltips = mark_batch.tooltips.as_ref()
                .unwrap_or_else(|| panic!("layer {i} must have tooltips via chart-level fallback"));
            assert_eq!(
                tooltips[0].fields[0].name, "y0",
                "layer {i} without its own tooltip_fields must fall back to the chart-level field"
            );
        }
    }

    /// A single-layer (unlayered) chart's tooltip resolution must be
    /// completely unaffected by the Task 10f per-layer tooltip fix: same
    /// content, and byte-identical scene JSON across two independent builds
    /// of the identical spec/batch.
    #[test]
    fn unlayered_chart_tooltip_scene_json_byte_identical() {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                tooltip_fields: Some(vec![EncodingSpec { field: "y".into(), ..Default::default() }]),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None, mark_style: None,
            position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
        ]).unwrap();

        let scene_a = build_scene_for(&spec, &batch);
        let scene_b = build_scene_for(&spec, &batch);
        assert_eq!(
            serde_json::to_string(&scene_a).unwrap(),
            serde_json::to_string(&scene_b).unwrap(),
            "identical inputs must produce byte-identical scene JSON"
        );

        let tooltips = scene_a.panels[0].marks[0].tooltips.as_ref().expect("must have tooltips");
        assert_eq!(tooltips[0].fields[0].name, "y");
        assert_eq!(tooltips[0].fields[0].value, "10");
    }

    fn resolve_dual_y(spec: &ChartSpec, batch: &RecordBatch) -> scale_resolve::ResolvedScales {
        let theme = ThemeInputs::default();
        let prep = super::super::prepare::prepare_render_inputs(spec, batch, &theme, None).unwrap();
        let chart_config = super::super::chart_config::ChartConfig::default();
        let panel = crate::layout::PanelLayout {
            plot_area: crate::layout::Rect { x: 0.0, y: 0.0, w: 300.0, h: 200.0 },
            ..Default::default()
        };
        // Both layers read the whole panel batch (no per-layer data_source).
        let layer_batches = vec![batch.clone(), batch.clone()];
        let mut warnings = Vec::new();
        let (_spec, scales) = resolve_panel_scales(
            spec, &prep, &panel, batch, &layer_batches, &theme, &chart_config, &mut warnings, None,
        )
        .unwrap();
        scales
    }

    /// An `independent_y` layer resolves its OWN y-slot from its own data: slot 0
    /// keeps the primary (layer-0) domain, slot 1 carries layer 1's much larger
    /// domain, and `y_for_layer` routes each layer to its slot.
    #[test]
    fn independent_y_layer_resolves_its_own_slot_domain() {
        let (spec, batch) = two_layer_dual_y_spec(true);
        let scales = resolve_dual_y(&spec, &batch);

        assert!(scales.y_slots.has_independent(), "independent layer must create a second slot");
        assert_eq!(scales.y_slots.slots().len(), 2, "one primary slot + one independent slot");
        assert_eq!(scales.y_slots.slot_for_layer(0), 0, "layer 0 is always the primary slot");
        assert_eq!(scales.y_slots.slot_for_layer(1), 1, "the independent layer binds slot 1");

        let (lo0, hi0) = scales.y_for_layer(0).data_domain().expect("primary y is continuous");
        let (lo1, hi1) = scales.y_for_layer(1).data_domain().expect("slot-1 y is continuous");

        // Slot 0 is the small y0 range; slot 1 is the large y1 range. Padding/nice
        // widen the exact bounds, so compare against a separating midpoint.
        assert!(hi0 < 50.0, "slot 0 must be layer 0's small y0 domain, got {lo0}..{hi0}");
        assert!(lo1 > 50.0, "slot 1 must be layer 1's large y1 domain, got {lo1}..{hi1}");

        // Slot 0 mirrors the primary `y` exactly (byte-stable primary resolution).
        assert_eq!(
            scales.y.data_domain(),
            scales.y_for_layer(0).data_domain(),
            "slot 0 must equal the primary y-scale"
        );
    }

    /// With both layers sharing y (flag false), no independent slot is built: the
    /// slot list stays empty and every layer maps through the primary `y` — the
    /// byte-stable pre-#52 path.
    #[test]
    fn shared_y_layers_leave_slots_empty_and_bind_primary() {
        let (spec, batch) = two_layer_dual_y_spec(false);
        let scales = resolve_dual_y(&spec, &batch);

        assert!(!scales.y_slots.has_independent(), "no independent layer → no extra slot");
        assert!(scales.y_slots.slots().is_empty(), "shared path leaves the slot list empty");
        assert_eq!(scales.y_slots.slot_for_layer(1), 0, "shared layers bind slot 0");

        // Every layer draws through the one primary y-scale.
        assert_eq!(scales.y_for_layer(0).data_domain(), scales.y.data_domain());
        assert_eq!(scales.y_for_layer(1).data_domain(), scales.y.data_domain());
    }

    /// Secondary-y (#52): `build_tick_levels` emits one `y_slot_levels` entry per
    /// right axis, generated from that slot's own scale, so the WASM overlay can
    /// recognize and reposition right-axis tick labels under zoom.
    #[test]
    fn build_tick_levels_emits_secondary_slot_levels() {
        let (spec, batch) = two_layer_dual_y_spec(true);
        let scales = resolve_dual_y(&spec, &batch);
        let ptl = build_tick_levels(&scales, 0);

        assert_eq!(ptl.y_slot_levels.len(), 1, "one independent slot → one right-axis tick list");
        // Same zoom-breakpoint structure as `y_levels`, with populated ticks.
        assert_eq!(ptl.y_slot_levels[0].len(), ptl.y_levels.len());
        assert!(
            ptl.y_slot_levels[0].iter().any(|lvl| !lvl.ticks.is_empty()),
            "the secondary slot must contribute tick labels"
        );
    }

    /// Byte-stability: a shared-y chart leaves `y_slot_levels` empty, so the
    /// `skip_serializing_if` keeps the tick-levels blob identical to pre-#52.
    #[test]
    fn build_tick_levels_shared_y_omits_slot_levels() {
        let (spec, batch) = two_layer_dual_y_spec(false);
        let scales = resolve_dual_y(&spec, &batch);
        let ptl = build_tick_levels(&scales, 0);
        assert!(ptl.y_slot_levels.is_empty(), "shared-y chart emits no secondary slot levels");

        let json = serde_json::to_string(&ptl).unwrap();
        assert!(!json.contains("y_slot_levels"), "empty slot levels must be omitted from JSON");
    }

    // ── #52 Task 3: layout + axis emission for independent-y layers ─────────

    /// Secondary y-axes never contribute gridlines — only slot 0 (the primary
    /// `panel_y_axis`) does (spec §4: "Right axes render ticks and labels but
    /// no gridlines"). A secondary `AxisLayout` with `show_grid: true` and a
    /// tick count that DIFFERS from the primary's must not change the emitted
    /// grid at all, while its own axis nodes (ticks/labels/domain line/title)
    /// DO appear in the routed axis list — one axis per slot.
    #[test]
    fn secondary_y_axes_do_not_contribute_gridlines_but_emit_their_own_axis_nodes() {
        use crate::layout::text_metrics::{fixed_width, MockMetrics};
        use crate::layout::{AxisInput, AxisOrient};

        let (spec, batch) = two_layer_dual_y_spec(true);
        let scales = resolve_dual_y(&spec, &batch);
        let panel = crate::layout::PanelLayout {
            plot_area: crate::layout::Rect { x: 0.0, y: 0.0, w: 300.0, h: 200.0 },
            ..Default::default()
        };
        let theme = ThemeInputs::default();
        let chart_config = super::super::chart_config::ChartConfig::default();
        let m = MockMetrics { measure: fixed_width(8.0), line_h_factor: 1.2 };

        let primary_y = crate::layout::axis::layout_y_axis(
            &AxisInput::new(
                AxisOrient::Left,
                Some("Primary".into()),
                vec!["0".into(), "5".into(), "10".into()],
                None,
            ),
            panel.plot_area, 0, 11.0, 13.0, 8.0, &m,
        );

        let mut secondary_input = AxisInput::new(
            AxisOrient::Right,
            Some("Secondary".into()),
            // Deliberately a DIFFERENT tick count than the primary's 3 — if the
            // grid leaked this axis's ticks, the counts below would diverge.
            vec!["0".into(), "25".into(), "50".into(), "75".into(), "100".into()],
            None,
        );
        secondary_input.show_grid = true; // deliberately try to leak into the grid
        let secondary_y = crate::layout::axis::layout_y_axis(
            &secondary_input, panel.plot_area, 0, 11.0, 13.0, 8.0, &m,
        );

        // Baseline: grid + axis nodes built from the primary alone.
        let baseline = route_panel_axes_and_grid(
            &spec, &scales, &panel, &[], None, Some(&primary_y), &[],
            false, false, &theme, &chart_config,
        );
        // With the secondary axis routed in alongside the primary.
        let with_secondary = route_panel_axes_and_grid(
            &spec, &scales, &panel, &[], None, Some(&primary_y), &[&secondary_y],
            false, false, &theme, &chart_config,
        );

        assert_eq!(
            with_secondary.grid.len(), baseline.grid.len(),
            "a secondary y-axis must not add or alter gridlines, even with show_grid=true"
        );
        // But it DOES contribute its own axis nodes (ticks + domain + labels +
        // title) — one axis per slot.
        let baseline_axis_nodes = baseline.axes_below.len() + baseline.axes_above.len();
        let with_secondary_axis_nodes = with_secondary.axes_below.len() + with_secondary.axes_above.len();
        assert!(
            with_secondary_axis_nodes > baseline_axis_nodes,
            "secondary axis must emit its own scene nodes: baseline={baseline_axis_nodes}, with={with_secondary_axis_nodes}"
        );
    }

    /// End-to-end (`render_svg`): a dual-axis LayerChart reserves BOTH margin
    /// bands (no plot-area overdraw — the plot area is strictly narrower than
    /// the shared-y equivalent at the same viewport), emits one Right-orient
    /// secondary axis, and titles it from layer 1's own y field — spec §4/§9
    /// acceptance criterion 1.
    #[test]
    fn render_svg_independent_y_reserves_right_band_and_emits_secondary_axis() {
        let theme = ThemeInputs::default();
        let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
        let config = super::super::config::RenderConfig::default();
        let chart_config = super::super::chart_config::ChartConfig::default();

        let (shared_spec, shared_batch) = two_layer_dual_y_spec(false);
        let shared = super::super::render_svg(
            &shared_spec, &shared_batch, &theme, viewport, &config, &chart_config,
        )
        .unwrap();
        assert!(
            shared.layout.secondary_y_axes.is_empty(),
            "the shared-y chart must not reserve any secondary axis"
        );

        let (dual_spec, dual_batch) = two_layer_dual_y_spec(true);
        let dual = super::super::render_svg(
            &dual_spec, &dual_batch, &theme, viewport, &config, &chart_config,
        )
        .unwrap();

        assert_eq!(dual.layout.secondary_y_axes.len(), 1, "one secondary axis for the one independent_y layer");
        let secondary = &dual.layout.secondary_y_axes[0];
        assert_eq!(secondary.orient, crate::layout::AxisOrient::Right);
        // No explicit title on layer 1's y encoding → falls back to the field
        // name ("y1"), same 3-way title resolution the primary axis uses.
        assert_eq!(secondary.title.as_ref().unwrap().text, "y1");

        // No plot-area overdraw: the dual-axis plot area is strictly narrower
        // than the shared-y plot area at the identical viewport — the right
        // band is genuinely reserved, not drawn over.
        let shared_w = shared.layout.panels[0].plot_area.w;
        let dual_w = dual.layout.panels[0].plot_area.w;
        assert!(
            dual_w < shared_w,
            "dual-axis plot area ({dual_w}) must be narrower than the shared-y plot area ({shared_w})"
        );

        // The secondary axis's title text renders into the SVG.
        assert!(dual.bytes.contains(">y1<"), "secondary axis title must appear in the SVG");
    }

    // ── #52 Task 8: scene contract — per-slot domains ────────────────────────

    /// Run the full `prepare → layout → build_scene` pipeline for
    /// `two_layer_dual_y_spec(layer1_independent)` and return the resulting
    /// `SceneGraph`, so tests can inspect the JSON-serializable scene contract
    /// directly (not just the SVG bytes `render_svg` exposes). Mirrors the
    /// recipe `scene_graph_path_matches_old_path_scatter` (render/mod.rs) uses
    /// to build a scene outside the `render_svg` entry point.
    fn build_dual_y_scene(layer1_independent: bool) -> ferrum_scene::SceneGraph {
        let (spec, batch) = two_layer_dual_y_spec(layer1_independent);
        let theme = ThemeInputs::default();
        let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
        let config = super::super::config::RenderConfig::default();
        let chart_config = super::super::chart_config::ChartConfig::default();

        let prep = super::super::prepare::prepare_render_inputs(&spec, &batch, &theme, None).unwrap();
        let mut warnings = prep.warnings.clone();
        let metrics = super::super::font::FontdueMetrics::new();
        let layout = crate::layout::compute_layout(
            &spec, &theme, viewport,
            &prep.axes, &prep.facet_groups, &prep.legend_entries,
            prep.legend_title.clone(), prep.colorbar.as_ref(), &metrics,
            &crate::layout::legend::LegendOverrides::default(), &prep.aux_legends,
            crate::layout::legend::LegendSuppression::default(),
        ).unwrap();

        build_scene(&spec, &prep, &layout, &theme, &config, &mut warnings, &chart_config, None).unwrap()
    }

    /// Find every `("y_slot", value)` attr pair carried by `SceneNode::Group`
    /// wrappers anywhere in `nodes` (axis nodes are wrapped one group per
    /// y-axis — see `route_y_axis_slotted`).
    fn collect_y_slot_group_tags(nodes: &[SceneNode]) -> Vec<String> {
        nodes.iter().filter_map(|n| {
            if let SceneNode::Group { attrs, .. } = n {
                attrs.iter().find(|(k, _)| k == "y_slot").map(|(_, v)| v.clone())
            } else {
                None
            }
        }).collect()
    }

    /// The dual-axis panel's coordinate state carries an ordered y-domain list
    /// (index = slot): slot 0 mirrors the primary (small y0) domain, slot 1
    /// carries the independent layer's own (large y1) domain — spec §6
    /// "Scene/WASM contract".
    #[test]
    fn scene_coord_dual_axis_carries_ordered_y_domains_per_slot() {
        let scene = build_dual_y_scene(true);
        let panel = &scene.panels[0];
        match &panel.coord {
            ferrum_scene::CoordKind::Cartesian { y_domain, y_domains, .. } => {
                assert_eq!(y_domains.len(), 2, "one y-domain per slot (primary + one independent)");
                let (slot0_lo, slot0_hi) = y_domains[0].expect("slot 0 domain must be Some");
                let (slot1_lo, slot1_hi) = y_domains[1].expect("slot 1 domain must be Some");
                assert!(slot0_hi < 50.0, "slot 0 must be the small y0 domain, got {slot0_lo}..{slot0_hi}");
                assert!(slot1_lo > 50.0, "slot 1 must be the large y1 domain, got {slot1_lo}..{slot1_hi}");
                // Slot 0 mirrors the panel's primary `y_domain` exactly.
                assert_eq!(Some((slot0_lo, slot0_hi)), *y_domain);
            }
            other => panic!("expected Cartesian coord, got {other:?}"),
        }
    }

    /// Shared-path back-compat (spec §7 byte-stability, brief Task 8): a chart
    /// with no `independent_y` layer must carry an empty `y_domains` list —
    /// omitted from JSON (`skip_serializing_if`) — and the serialized scene
    /// must not contain the new `y_domains`/`y_slot` keys anywhere, proving
    /// the addition is a true no-op for every pre-#52 chart.
    #[test]
    fn scene_coord_shared_path_leaves_y_domains_empty_and_byte_identical() {
        let scene = build_dual_y_scene(false);
        let panel = &scene.panels[0];
        match &panel.coord {
            ferrum_scene::CoordKind::Cartesian { y_domains, .. } => {
                assert!(y_domains.is_empty(), "shared path must leave the per-slot y-domain list empty");
            }
            other => panic!("expected Cartesian coord, got {other:?}"),
        }
        for batch in &panel.marks {
            assert_eq!(batch.y_slot, 0, "every mark batch binds slot 0 on the shared path");
        }
        assert!(
            collect_y_slot_group_tags(&panel.axes).is_empty(),
            "no axis Group should carry a y_slot tag on the shared path"
        );

        let json = serde_json::to_string(&scene).expect("serialize shared-path scene");
        assert!(!json.contains("y_domains"), "shared-path scene JSON must omit y_domains: {json}");
        assert!(!json.contains("y_slot"), "shared-path scene JSON must omit y_slot: {json}");
    }

    /// Dual-axis mark meshes carry the slot index their layer's marks were
    /// positioned through: layer 0 (shared/primary) is batch 0 → slot 0; the
    /// independent layer 1 is batch 1 → slot 1.
    #[test]
    fn mark_batches_carry_their_layers_y_slot() {
        let scene = build_dual_y_scene(true);
        let panel = &scene.panels[0];
        assert_eq!(panel.marks.len(), 2, "one mark batch per layer");
        assert_eq!(panel.marks[0].y_slot, 0, "layer 0 (primary) binds slot 0");
        assert_eq!(panel.marks[1].y_slot, 1, "layer 1 (independent) binds slot 1");
    }

    /// Dual-axis y-axis scene nodes are tagged with their slot index: the
    /// primary (left) axis is wrapped with `y_slot="0"`, the secondary (right)
    /// axis with `y_slot="1"` — so the interactive runtime can relabel/rescale
    /// each axis from its own scale (spec §6).
    #[test]
    fn y_axis_scene_nodes_tagged_with_slot_index() {
        let scene = build_dual_y_scene(true);
        let panel = &scene.panels[0];
        let tags = collect_y_slot_group_tags(&panel.axes);
        let mut sorted_tags = tags.clone();
        sorted_tags.sort();
        assert_eq!(
            sorted_tags, vec!["0".to_string(), "1".to_string()],
            "expected exactly one y_slot=0 (primary) and one y_slot=1 (secondary) axis group, got {tags:?}"
        );
    }
}
