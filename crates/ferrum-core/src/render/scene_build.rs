use crate::spec::coord::to_scene_coord;
use arrow::record_batch::RecordBatch;
use ferrum_scene::{
    BindingRole, BlendMode, CoordKind, InteractionConfig, LayoutScale, MarkBatch, Panel,
    PanelTickLevels, ParamBinding, SceneGraph, SceneNode, TickLevel,
};

use crate::layout::{AxisLayout, LayoutResult, ResolveMode, ThemeInputs};
use crate::spec::chart::ChartSpec;

use super::arrow_cast::{col_as_ordinal_category_str, col_as_temporal_epoch_str};
use super::chart_config::StructuralSpec;
use super::config::RenderConfig;
use super::draw::{self, to_scene_color, to_scene_text_style, DrawCtx};
use super::marks;
use super::prepare::PreparedInputs;
use super::scale_resolve::LeafScaleContext;
use super::{
    break_axis, filter_batch_by_facet, inset, position, scale_resolve, RenderError, RenderWarning,
    CLIP_ID_PREFIX,
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

    // GH #70: reference plot area for re-anchoring an explicit Band/Point/
    // positional-Ordinal pixel range per facet panel (see
    // `resolve_panel_scales` and `OrdinalScale::translate_explicit_range`).
    // `layout.panels[0]` for every standalone (non-faceted) render IS this
    // render's only panel, so `panel_offset` below is always `(0.0, 0.0)`
    // there — byte-identical to the pre-#70 behavior.
    let reference_plot_area = layout.panels.first().map(|p| p.plot_area);

    // Slot id for each secondary (right) y-axis, in axis-band order (GH #72).
    // Read from the one layer→slot plan rather than inferred from the axis's
    // position in the band list, so the axis router cannot drift from the slot
    // the marks and `y_domains` resolved against. Identical for every panel
    // (structural), so computed once here. Empty on the shared path.
    let secondary_slots: Vec<usize> = prep
        .y_slot_plan
        .secondary_layers()
        .iter()
        .map(|&layer_idx| prep.y_slot_plan.slot_for_layer(layer_idx))
        .collect();

    // Loop-invariant chart-level scale-resolution context (SPINE-04 follow-up,
    // T6): every panel this loop iterates resolves against the SAME spec,
    // prepared inputs, theme, chart config, and composite leaf-scale context —
    // constructing the bundle once here, rather than threading five loose
    // references through every `resolve_panel_scales` call, makes that
    // loop-invariance visible at the call site instead of implicit.
    let panel_resolve_ctx = PanelResolveCtx {
        spec,
        prep,
        theme,
        chart_config,
        leaf_scales,
    };

    for (panel_idx, panel) in layout.panels.iter().enumerate() {
        if panel.plot_area.w <= 0.0 || panel.plot_area.h <= 0.0 {
            warnings.push(RenderWarning::EmptyPanel {
                panel_index: panel_idx,
            });
            continue;
        }

        // Strip title — emitted as separate nodes in the panel, not a group.
        // Includes both the column-header strip (top) and, in grid mode, the
        // row-header strip (right side). Both are appended to the same vec so
        // the compositor's offset logic picks them up without a schema change.
        let mut strip_title_nodes: Vec<SceneNode> = panel
            .strip_title
            .as_ref()
            .map(|strip| marks::strip_title::build_strip_title(strip, &panel.plot_area, theme))
            .unwrap_or_default();
        if let Some(row_strip) = &panel.row_strip_title {
            strip_title_nodes.extend(marks::strip_title::build_row_strip_title(
                row_strip,
                panel.plot_area.h,
                theme,
            ));
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
                    let src = prep
                        .transform_outputs
                        .get(name)
                        .expect("layer.data_source validated by prepare_render_inputs");
                    if let Some(key) = &panel.facet_key {
                        let col_filtered = filter_batch_by_facet(src, &key.field, &key.value)?;
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

        // GH #70: this panel's displacement from the reference (panel 0)
        // plot-area origin, threaded into `resolve_panel_scales` so an
        // explicit Band/Point/positional-Ordinal range re-anchors inside
        // THIS panel instead of staying pinned to chart-absolute pixels.
        let panel_offset = reference_plot_area
            .map(|r| (panel.plot_area.x - r.x, panel.plot_area.y - r.y))
            .unwrap_or((0.0, 0.0));

        // Per-panel scale build (encoding merge + param-domain substitution +
        // scale resolution + color-config re-apply) through the single
        // `resolve_panel_scales` seam, so the prepare provisional pass and this
        // per-panel pass cannot drift on what scales get built or on remembering
        // to re-apply the color config.
        let (rendering_spec_for_panel, scales) = resolve_panel_scales(
            &panel_resolve_ctx,
            panel,
            &panel_batch,
            &layer_batches,
            warnings,
            panel_offset,
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
        let panel_x_axis_global = panel_axes_layout.iter().copied().find(|a| {
            matches!(
                a.orient,
                crate::layout::AxisOrient::Bottom | crate::layout::AxisOrient::Top
            )
        });
        let panel_y_axis_global = panel_axes_layout.iter().copied().find(|a| {
            matches!(
                a.orient,
                crate::layout::AxisOrient::Left | crate::layout::AxisOrient::Right
            )
        });

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
        let panel_x_axis: Option<&AxisLayout> =
            panel_axes.independent_x.as_ref().or(panel_x_axis_global);
        let panel_y_axis: Option<&AxisLayout> =
            panel_axes.independent_y.as_ref().or(panel_y_axis_global);

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
            &secondary_slots,
            panel_axes.x_independent,
            panel_axes.y_independent,
            theme,
            chart_config,
        );

        // Per-layer mark batches (MOD-09).
        let mut mark_batches =
            build_panel_mark_batches(spec, prep, &layer_batches, &scales, panel, theme, warnings)?;

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
        let scene_coord = spec
            .coord
            .as_ref()
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
            CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand,
                clip,
                y_domains,
            } => {
                let x_dom = scales.x.data_domain();
                let y_dom = scales.y.data_domain();
                CoordKind::Cartesian {
                    x_domain: x_dom,
                    y_domain: y_dom,
                    expand,
                    clip,
                    y_domains,
                }
            }
            CoordKind::Fixed {
                x_domain: None,
                y_domain: None,
                ratio,
                expand,
                clip,
            } => {
                let x_dom = scales.x.data_domain();
                let y_dom = scales.y.data_domain();
                CoordKind::Fixed {
                    x_domain: x_dom,
                    y_domain: y_dom,
                    ratio,
                    expand,
                    clip,
                }
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
                CoordKind::Cartesian {
                    x_domain,
                    y_domain,
                    expand,
                    clip,
                    ..
                } => {
                    let y_domains = scales
                        .y_slots
                        .slots()
                        .iter()
                        .map(|s| s.data_domain())
                        .collect();
                    CoordKind::Cartesian {
                        x_domain,
                        y_domain,
                        expand,
                        clip,
                        y_domains,
                    }
                }
                other => other,
            }
        } else {
            scene_coord
        };

        // Annotations: render user-specified annotations on the first panel only.
        // `build_annotations` partitions nodes by the Text spec's `z` field
        // (GH #89B): `below_marks` → the panel's typed `below_marks` slot
        // (pre-marks content bucket); `above_marks` → the panel's
        // `annotations` slot (post-marks content bucket). Chrome (above-marks
        // grid/axes, zindex >= 1) routes into the separate `chrome_above`
        // slot below — a typed sibling, not a prefix of `annotations`.
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
        let StructuralOutput {
            extra_annotations: structural_annotations,
            break_results,
        } = if panel_idx == 0 && !chart_config.structural.is_empty() {
            build_structural_nodes(&chart_config.structural, &scales, &panel.plot_area, theme)
        } else {
            StructuralOutput {
                extra_annotations: Vec::new(),
                break_results: Vec::new(),
            }
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
                        &mut batch.nodes,
                        axis,
                        d_lo,
                        d_hi,
                        px_lo,
                        px_hi,
                        break_result,
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
        // `grid` slot into the `chrome_above` slot (emitted after marks). The
        // below-marks default keeps the grid in `grid` for byte-identical output.
        let (grid_below, grid_above_nodes) = if grid_above {
            (Vec::new(), grid_nodes)
        } else {
            (grid_nodes, Vec::new())
        };

        // Typed chrome/content slots (GH #89B): `grid` (chrome) and
        // `below_marks` (content — text annotations with `z == "below_marks"`)
        // are distinct `Panel` fields rather than one commingled bucket, so a
        // later overlay merge can clear duplicate chrome without risk of also
        // dropping a user annotation. `below_marks` paints immediately after
        // `grid` (ferrum-scene `Panel::below_marks` doc), the same visual
        // position these nodes held when they were appended onto `grid`.
        let final_below_marks: Vec<SceneNode> = annotation_nodes.below_marks;

        let final_axes: Vec<SceneNode> = axes_nodes;
        let final_marks: Vec<ferrum_scene::MarkBatch> = mark_batches;
        // Typed chrome/content slots (GH #89B): above-marks axis/grid chrome
        // (zindex >= 1) routes into `chrome_above`, a typed sibling of
        // `annotations` rather than a prefix within it. `chrome_above` paints
        // immediately after `axes` and before `annotations` — the deliberate
        // z-order refinement: above-marks user annotations now always paint
        // above above-marks axis chrome (previously chrome was prefixed into
        // the same `annotations` list ahead of user content, so it painted
        // OVER user annotations).
        let final_chrome_above: Vec<SceneNode> = {
            let mut v = grid_above_nodes;
            v.extend(axes_above_nodes);
            v
        };
        // Annotation list (emitted after `chrome_above`): user annotations
        // (`above_marks` bucket), then structural annotations. Chrome (grid +
        // axis, zindex >= 1) no longer lives here — see `final_chrome_above`.
        let final_annotations: Vec<SceneNode> = {
            let mut v = annotation_nodes.above_marks;
            v.extend(structural_annotations);
            v
        };

        panels.push(Panel {
            id: panel_idx,
            plot_area,
            clip: panel_clip,
            coord: scene_coord,
            grid: grid_below,
            below_marks: final_below_marks,
            marks: final_marks,
            axes: final_axes,
            chrome_above: final_chrome_above,
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
    let param_bindings = collect_param_bindings(spec, &prep.layers, &y_slots, layout.panels.len());

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

/// The loop-invariant chart-level scale-resolution context: the chart spec,
/// prepared inputs, theme, chart config, and composite leaf-scale context.
/// These five references are identical across every panel a `build_scene`
/// call iterates (and across both `resolve_panel_scales` and its per-layer
/// helper `resolve_layer_y_scale`), so bundling them here removes the
/// same five loose parameters from both functions' signatures at once —
/// grouped because they travel together, not because they are merely
/// adjacent. `resolve_layer_y_scale` does not read `chart_config` (the color
/// override it carries applies once, to the primary y scale, inside
/// `resolve_panel_scales` itself) but takes the same bundle for uniformity
/// with its sibling, matching the `DrawCtx` precedent (`render/draw.rs`)
/// where not every consumer reads every field.
#[derive(Clone, Copy)]
pub(in crate::render) struct PanelResolveCtx<'a> {
    pub(in crate::render) spec: &'a ChartSpec,
    pub(in crate::render) prep: &'a PreparedInputs,
    pub(in crate::render) theme: &'a ThemeInputs,
    pub(in crate::render) chart_config: &'a super::chart_config::ChartConfig,
    // D4b composite seam: shared-domain context for this leaf. `None` for
    // standalone renders → resolves exactly as before.
    pub(in crate::render) leaf_scales: Option<&'a LeafScaleContext>,
}

/// The single per-panel scale-build seam: merge the layer-0 encoding onto the
/// chart encoding, substitute reactive `domainParam` references into concrete
/// domains (D6: a named variable's static numeric-array value becomes the
/// concrete `domain`; a selection or non-numeric reference leaves `domain =
/// None` so the renderer auto-infers — the correct static semantics for an
/// empty selection; param-free specs stay on the exact pre-D6 code path),
/// resolve the panel's scales over its pixel range, and re-apply the
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
fn resolve_panel_scales(
    ctx: &PanelResolveCtx,
    panel: &crate::layout::PanelLayout,
    panel_batch: &RecordBatch,
    // Per-panel layer batches (one per `prep.layers`, facet-filtered). Slot 0 /
    // the primary y resolves against `panel_batch` exactly as before; each
    // independent layer's y-slot resolves against its own batch here.
    layer_batches: &[RecordBatch],
    warnings: &mut Vec<RenderWarning>,
    // GH #70: this panel's `(x, y)` displacement from the reference (panel 0)
    // plot-area origin. `(0.0, 0.0)` for panel 0 and for every standalone
    // (non-faceted) render — byte-identical to the pre-#70 behavior. Applied
    // AFTER scale resolution to re-anchor an explicit Band/Point/positional-
    // Ordinal pixel range (chart-absolute by design, #39 phase 2) inside this
    // panel instead of leaving it pinned to the same absolute pixels every
    // panel would otherwise share.
    panel_offset: (f64, f64),
) -> Result<(ChartSpec, scale_resolve::ResolvedScales), RenderError> {
    let &PanelResolveCtx {
        spec,
        prep,
        theme,
        chart_config,
        leaf_scales,
    } = ctx;

    // Encoding merge: layer-0 encoding overlays the chart-level encoding.
    let mut merged_encoding = spec.encoding.clone();
    merged_encoding.overlay_from(&prep.layers[0].encoding);
    let mut rendering_spec_for_panel = ChartSpec {
        encoding: merged_encoding,
        ..spec.clone()
    };

    // Reactive-rescale substitution (D6): turn `domainParam` references into
    // concrete domains before scale resolution. No-op when `params` is empty.
    scale_resolve::resolve_param_domains(&mut rendering_spec_for_panel);

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

    // GH #70: re-anchor an explicit Band/Point/positional-Ordinal range onto
    // THIS panel. No-op (both components `0.0`) for panel 0 and for every
    // standalone render; a no-op per-axis when that axis's resolved scale
    // isn't a genuinely user-supplied explicit range (see
    // `OrdinalScale::translate_explicit_range`).
    scales.x.translate_explicit_ordinal_range(panel_offset.0);
    scales.y.translate_explicit_ordinal_range(panel_offset.1);

    // Apply chart_config color overrides (level 3) to the per-panel color scale.
    // Must run after scale resolution because `resolve_scales_with_outputs`
    // independently re-resolves the color scale for each panel, discarding the
    // provisional override applied to `prep.provisional_scales` in
    // `prepare_and_layout`.
    if let Some(ref cfg) = chart_config.color {
        // Warnings are emitted once, by the chart-level application in
        // `prepare_and_layout`; this is the same config against the same scale,
        // so reporting here would duplicate one warning per panel.
        let _ = super::apply_color_config_to_color_scale(&mut scales.color, cfg);
    }

    // Per-layer independent y-scale slots (secondary-y-axis, GH #52). Byte-stable
    // gate: only build slots when the prepared plan carries an independent-y
    // layer; otherwise leave `scales.y_slots` at its empty default so shared and
    // `y:"shared"` charts resolve exactly as before. Slot 0 stays the primary `y`
    // resolved above (against layer 0's / the panel batch), unchanged. The
    // layer→slot map is NOT re-derived here — it comes straight from
    // `prep.y_slot_plan` (GH #72), so this per-panel resolution, the axis-band
    // order, and the axis router cannot drift. Slots are pushed in the plan's
    // `secondary_layers` order, so `slots[k + 1]` is `secondary_layers[k]`,
    // matching `layer_slot`.
    if prep.y_slot_plan.has_independent() {
        let mut slots: Vec<scale_resolve::ScaleKind> = vec![scales.y.clone()];
        for &layer_idx in prep.y_slot_plan.secondary_layers() {
            let y_scale = resolve_layer_y_scale(
                ctx,
                &prep.layers[layer_idx],
                &layer_batches[layer_idx],
                panel,
                panel_offset.1,
                warnings,
            )?;
            slots.push(y_scale);
        }
        scales.y_slots =
            scale_resolve::YScaleSlots::new(slots, prep.y_slot_plan.layer_slot().to_vec());
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
fn resolve_layer_y_scale(
    ctx: &PanelResolveCtx,
    layer: &super::prepare::LayerPrepared,
    layer_batch: &RecordBatch,
    panel: &crate::layout::PanelLayout,
    // GH #70: this panel's y displacement from the reference panel's
    // plot-area origin (see `resolve_panel_scales`), applied to this slot's
    // resolved y scale the same way the primary y is re-anchored.
    y_offset: f64,
    warnings: &mut Vec<RenderWarning>,
) -> Result<scale_resolve::ScaleKind, RenderError> {
    let &PanelResolveCtx {
        spec,
        prep,
        theme,
        leaf_scales,
        ..
    } = ctx;

    // Shared param-aware per-layer y resolution (#72): the layer encoding
    // overlays the chart encoding, `layers: None` scopes the domain to this
    // layer's own field, and `domainParam` references are substituted before
    // resolution — the SAME resolution the prepare stage's axis-input builder
    // consumes, so ticks and marks cannot diverge. Only the pixel range differs
    // (panel-real here vs. prepare's placeholder).
    let layer_ctx = scale_resolve::LayerScaleCtx {
        spec,
        transform_outputs: &prep.transform_outputs,
        theme,
        leaf_scales,
    };
    let (mut y, layer_warnings) = scale_resolve::resolve_layer_y_slot_scale(
        &layer_ctx,
        layer.mark,
        &layer.encoding,
        layer_batch,
        (panel.plot_area.x, panel.plot_area.x + panel.plot_area.w),
        (panel.plot_area.y, panel.plot_area.y + panel.plot_area.h),
    )?;
    warnings.extend(layer_warnings);
    y.translate_explicit_ordinal_range(y_offset);
    Ok(y)
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
    let x_independent = spec
        .facet
        .as_ref()
        .map(|f| f.resolve.x == ResolveMode::Independent)
        .unwrap_or(false);
    let y_independent = spec
        .facet
        .as_ref()
        .map(|f| f.resolve.y == ResolveMode::Independent)
        .unwrap_or(false);

    // Re-derive raw format specs from the merged rendering encoding so that
    // independent-axis label formatting uses the same precedence logic as
    // the shared path (Axis(label_format=) > encoding.format > none).
    // `resolve_axis_label_format` is the canonical single source of truth
    // for this precedence — calling it here avoids duplicating the logic
    // and ensures both paths stay in sync.
    let (x_fmt_spec, x_fmt_type) =
        super::prepare::resolve_axis_label_format(rendering_spec_for_panel.encoding.x.as_ref());
    let (y_fmt_spec, y_fmt_type) =
        super::prepare::resolve_axis_label_format(rendering_spec_for_panel.encoding.y.as_ref());

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
        let (new_y_layout, _warn) = crate::layout::axis::layout_y_axis(
            &y_input,
            panel.plot_area,
            panel_idx,
            y_label_fs,
            theme.typography.title_font_size,
            theme.padding.axis_title_padding,
            theme.sizes.tick_size,
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
    // The y-slot id for each entry in `panel_secondary_y`, in the same order
    // (GH #72). Sourced from the one layer→slot plan so this router consumes the
    // slot rather than inferring it from the axis's list position. Same length
    // as `panel_secondary_y` (both follow the plan's `secondary_layers` order).
    secondary_slots: &[usize],
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

    let grid_band_colors: &[String] = chart_config
        .grid
        .as_ref()
        .and_then(|g| g.band_colors.as_deref())
        .unwrap_or(&[]);
    let grid = if suppress_axes {
        Vec::new()
    } else {
        marks::axis::build_grid(
            panel.plot_area,
            panel_x_axis,
            panel_y_axis,
            theme,
            grid_band_colors,
        )
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
        // Untagged: x-axes and y-axes on the byte-stable single-slot default
        // path never carry a `slot` tag on their tick-label text nodes.
        let nodes = marks::axis::build_axis(axis, theme, None);
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
            // Slot tag on tick-label text nodes (GH #60/#73): every y-axis
            // routed through this dual-axis wrapper tags its own tick labels
            // with the same slot carried by the enclosing `Group`'s `y_slot`
            // attr — including slot 0 (primary), for a uniform contract where
            // "this panel has 2+ y-slots" implies every y-axis tick label is
            // slot-tagged, not just the secondary ones.
            let nodes = marks::axis::build_axis(axis, theme, Some(slot));
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
                if !matches!(
                    axis.orient,
                    crate::layout::AxisOrient::Bottom
                        | crate::layout::AxisOrient::Top
                        | crate::layout::AxisOrient::Left
                        | crate::layout::AxisOrient::Right
                ) {
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
        // axis. Same above/below zindex routing as every other axis. Each axis's
        // slot comes from `secondary_slots` (the layer→slot plan, GH #72), not
        // from its position in the band list — so mesh, axis, and `y_domains`
        // key off the same slot number by construction.
        for (axis, &slot) in panel_secondary_y.iter().zip(secondary_slots) {
            route_y_axis_slotted(axis, slot, &mut axes_above, &mut axes_below);
        }
    }

    // Polar axis: circular boundary + radial tick marks (replaces Cartesian axes)
    if matches!(
        &spec.coord,
        Some(crate::spec::coord::CoordKind::Polar { .. })
    ) {
        let cx = panel.plot_area.x + panel.plot_area.w / 2.0;
        let cy = panel.plot_area.y + panel.plot_area.h / 2.0;
        let outer_r = polar_outer_radius(&panel.plot_area);
        axes_below.extend(build_polar_axes(cx, cy, outer_r, scales, theme));
    }

    PanelAxisGrid {
        axes_below,
        axes_above,
        grid,
        grid_above,
    }
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
        //
        // `layer_slot_idx` is captured against the panel-level `scales` before
        // any shadowing below, so the `MarkBatch.y_slot` tag at the end of this
        // loop iteration always reflects this layer's real slot — independent
        // of how the clone's own `y_slots` describes itself.
        let layer_slot_idx = scales.y_slots.slot_for_layer(li);
        let layer_scales_owned: Option<scale_resolve::ResolvedScales> = if layer_slot_idx != 0 {
            let mut s = scales.clone();
            s.y = scales.y_for_layer(li).clone();
            // Self-describing y_slots: the clone's `.y` now points at this
            // one layer's own scale, so its `y_slots` should describe just
            // that — a single slot — rather than keep the stale multi-slot
            // list carried over from `scales.clone()`, which would let a
            // reader of `ctx.scales.y_slots` disagree with `ctx.scales.y`.
            s.y_slots = scale_resolve::YScaleSlots::single(s.y.clone());
            Some(s)
        } else {
            None
        };
        let scales: &scale_resolve::ResolvedScales = layer_scales_owned.as_ref().unwrap_or(scales);

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

        // Synthetic ChartSpec for this layer. The `encoding` here is a
        // DrawCtx-LOCAL copy — everything else in this loop (position
        // grouping just above, key extraction below, and the legend's own
        // separate resolution in `resolve_legend_color_scale`) reads
        // `layer.encoding` directly and is never touched by what follows.
        //
        // Resolved before the own-color exemption below because that
        // exemption's non-Text branch needs to know whether THIS layer
        // carries its own literal stroke/fill override.
        // Refuses with `RenderError::InvalidColor` on an unparseable
        // `fill=`/`stroke=` (batch-A Task 8) — the same build-seam refusal
        // discipline as `validate_mark_encoding` below.
        let mark_style = draw::resolve_mark_style(layer.mark_style.as_ref(), theme, &layer.mark)?;

        // Own-color exemption (spec §4.4, 2026-08-28 T4 amendment; widened
        // batch-A T5d, 2026-08-28): a layer whose `color` channel came purely
        // from chart-level inheritance (`!layer.color_is_own`) must not have
        // it reach the mark builder's per-row color read when doing so would
        // override that layer's own declared paint — only the layer's OWN
        // declared `color=` may drive per-row/per-group color variation.
        // Originally scoped to `Mark::Text` unconditionally (the
        // `heatmap(annot=True)` colored-cells + colorless-labels shape,
        // where an un-colored label must NEVER borrow the cells' hue).
        // Widened here to any OTHER mark, but only when that layer ALSO
        // carries its own literal `stroke=`/`fill=` override that names a
        // real color (`*_is_user_set && !*_cleared` — see the clear carve-out
        // at the predicate below) — a chart-level
        // `color` set for one layer's legend (e.g. a diagnostic chart's
        // per-class curve) silently repainted a sibling layer's own literal
        // stroke override with the categorical palette's first color
        // whenever that sibling declared no color of its own (`fm.roc_chart`'s
        // grey dashed chance-diagonal `reference` line flipping to the first
        // class's blue, and fanning out into one duplicate polyline per
        // class). The literal-override gate is deliberately narrower than
        // "any non-owning mark": a layer with NO literal paint override that
        // relies on a genuinely shared chart-level `color` to vary ITS OWN
        // per-row/per-group fill — e.g. `catplot(kind="box", hue=x)`'s IQR
        // `rect`/tick-cap/outlier-`point` layers, none of which declare their
        // own `color=` because the desugar treats the shared x-axis category
        // as the hue and lets every layer inherit it — must keep inheriting;
        // an unconditional strip (mirroring the Text rule) silently flattened
        // every box to the first category's color instead of one color per
        // box. Clearing `color` on this DrawCtx-local copy (not on
        // `layer.encoding` itself) is what makes `resolve_fill_color`/
        // `resolve_stroke_color`/each mark's own per-row lookup fall through
        // to that mark's fill/stroke/theme-default precedence, without
        // starving the legend or dodge/stack position grouping the way an
        // earlier revision did by deleting the channel from
        // `LayerPrepared.encoding` itself (see `Encoding::inherit_from`'s doc
        // comment for the full history).
        //
        // For a GROUPING mark (line/area/ribbon — `marks/line.rs`'s
        // `build_color_detail_groups`), the cleared channel is not just a
        // paint input: it is also the per-group split key. Clearing it
        // therefore collapses whatever per-color groups that layer's OWN
        // batch would otherwise have fanned out into back down to a single
        // merged node/polyline over every row. This is exactly the desired
        // effect for a literal-stroke layer that is conceptually ONE curve
        // (the `roc_chart` chance-diagonal is collinear — `y = x` — so the
        // merge is invisible), but it is a real topology change in general:
        // a literal-stroke line layer drawn over a batch with real per-group
        // discontinuities would render one polyline with spurious connecting
        // segments between what used to be separate groups, not N separate
        // ones. There is no route back to per-group splitting for such a
        // layer without giving it its own `color=`/`detail=`, since the
        // exemption's whole point is that this layer's color channel was
        // never its own to group by.
        let mut layer_spec_encoding = layer.encoding.clone();
        if !layer.color_is_own {
            // A *cleared* paint (`fill="none"` / `stroke="transparent"`) is not
            // an own literal paint for this purpose (NF-A3 ribbon half, intent
            // gate). The exemption protects a color the layer declared from
            // being overwritten by an inherited one; a clear declares no color
            // at all — it declares the channel unpainted, and the mark builders
            // carry that intent separately in
            // `MarkPaint::{fill,stroke}_cleared`, which survive the color scale.
            // Reading a clear as "own paint" is what made
            // `mark_ribbon().encode(color=…)` draw ONE merged band in the theme
            // default fill under a full multi-category legend: `desugar_ribbon`
            // (and `desugar_errorband`) always pass `stroke="none"`, so every
            // ribbon layer tripped the exemption and lost the color channel it
            // groups its bands by — while `mark_area`, whose lowering is flat
            // and never reaches this seam, partitioned correctly from the same
            // data. The layers that motivated the widening (e.g. `roc_chart`'s
            // grey dashed chance diagonal, `stroke="#9ca3af"`) declare a real
            // color and are unaffected, as is a layer that clears one channel
            // and paints the other.
            let has_own_literal_paint = (mark_style.paint.stroke_is_user_set
                && !mark_style.paint.stroke_cleared)
                || (mark_style.paint.fill_is_user_set && !mark_style.paint.fill_cleared);
            if layer.mark == crate::spec::mark::Mark::Text || has_own_literal_paint {
                layer_spec_encoding.color = None;
            }
        }
        // Own-span-axis normalization for rule (spec §4.4, batch-A Task 13
        // spec c3, 2026-09-01). A reference-line layer declares the axis it
        // belongs to by declaring exactly one positional channel — the
        // diagnostic desugars all do this (`{"x": "_ref_zero"}` for a vertical
        // zero line, `{"y": "_ref_zero"}` for a horizontal one). When the
        // chart ALSO sets a chart-level `x`/`y`, `Encoding::inherit_from`
        // fills in the opposite channel, and rule's shape derivation would
        // then see a two-channel pattern that is not the layer's own shape:
        // `shap_chart(kind="beeswarm")`'s vertical zero line inherited the
        // chart's ordinal `y="feature"` and rendered as one full-width
        // HORIZONTAL span per row; `alpha_selection_chart`'s and
        // `pca_scree_chart`'s vertical markers did the same. Clearing the
        // inherited-only channel on this DrawCtx-LOCAL copy (never on
        // `layer.encoding`, exactly as the own-color exemption above) keeps
        // every other consumer — scale domains, axis chrome, position
        // grouping, the legend — reading the merged encoding unchanged, while
        // the geometry sees the shape the layer actually declared.
        //
        // Deliberately narrow: only when the layer declared exactly one of
        // `x`/`y` and the merged encoding binds neither `x2` nor `y2`. A rule
        // whose ranged/diagonal shape is assembled partly from inherited
        // channels keeps assembling it, and a layer that declares both
        // channels itself keeps `RuleShape::resolve`'s documented tie-break.
        if layer.mark == crate::spec::mark::Mark::Rule
            && layer_spec_encoding.x2.is_none()
            && layer_spec_encoding.y2.is_none()
        {
            match (layer.x_is_own, layer.y_is_own) {
                (true, false) => layer_spec_encoding.y = None,
                (false, true) => layer_spec_encoding.x = None,
                _ => {}
            }
        }
        let layer_spec = ChartSpec {
            mark: layer.mark,
            encoding: layer_spec_encoding,
            ..spec.clone()
        };
        let ctx = DrawCtx {
            spec: &layer_spec,
            panel,
            theme,
            scales,
            batch: layer_batch,
            mark_style: &mark_style,
        };

        // Validate the encoding the mark builder will actually see — the
        // DrawCtx-local copy, not the pre-normalization merge — so the refusal
        // gate and the geometry can never disagree about a layer's shape
        // (batch-A Task 13 spec c3). Only rule's own-span-axis normalization
        // above touches a positional channel, so every other mark's gate
        // decision is unchanged.
        validate_mark_encoding(&layer.mark, &layer_spec.encoding, prep.coord_flipped)?;
        let mut result = draw::dispatch_mark_build(&layer.mark, &ctx)?;

        // For CoordPolar, transform all mark nodes from Cartesian pixel
        // space to polar pixel space. Arc marks (Mark::Arc) handle their
        // own polar geometry and must not be transformed again.  Bars under
        // CoordPolar route through `build_polar`, which also generates
        // arc-geometry nodes (MarkBatchKind::Arc) in polar space — those
        // must likewise be excluded from the transform, or the wedge
        // coordinates are corrupted by a second polar projection.
        let is_arc_geometry = matches!(result.kind, ferrum_scene::MarkBatchKind::Arc);
        if matches!(
            &spec.coord,
            Some(crate::spec::coord::CoordKind::Polar { .. })
        ) && !matches!(layer.mark, crate::spec::mark::Mark::Arc)
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
            stroke_cap: mark_style
                .line
                .stroke_cap
                .as_deref()
                .and_then(draw::parse_stroke_cap),
            stroke_join: mark_style
                .line
                .stroke_join
                .as_deref()
                .and_then(draw::parse_stroke_join),
            packed_instances: None,
            // Secondary-y-axis (GH #52 Task 8): tag this batch with the same
            // slot its marks were positioned through above. `0` on every
            // shared-path layer (the byte-stable default, omitted from JSON).
            // Uses `layer_slot_idx` (captured pre-shadow above), not the
            // (possibly single-slot, self-describing) clone's `y_slots`.
            y_slot: layer_slot_idx,
        });
    }

    Ok(mark_batches)
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
        let Some(scale) = channel.scale.as_ref() else {
            continue;
        };
        let Some(param) = scale.domain_param() else {
            continue;
        };
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
        let Some(y_channel) = layer.encoding.y.as_ref() else {
            continue;
        };
        let Some(scale) = y_channel.scale.as_ref() else {
            continue;
        };
        let Some(param) = scale.domain_param() else {
            continue;
        };
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
            let Some(param) = filter.param.as_ref() else {
                continue;
            };
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
    let Some(title) = &layout.chart_title else {
        return;
    };
    let title_spec = spec.title.as_ref();
    let resolved_font_size = title_spec
        .and_then(|t| t.font_size)
        .unwrap_or(theme.typography.title_font_size);
    let resolved_font_weight: String = title_spec
        .and_then(|t| t.font_weight.clone())
        .unwrap_or_else(|| theme.typography.title_font_weight.clone());
    let resolved_color = title_spec
        .and_then(|t| t.color.as_deref())
        .and_then(|s| super::color::parse_color(s).ok())
        .unwrap_or(theme.colors.title_color);
    let fw = if resolved_font_weight == "normal" {
        None
    } else {
        Some(resolved_font_weight.as_str())
    };
    out.push(SceneNode::Text {
        x: title.x,
        y: title.y,
        content: title.text.clone(),
        slot: None,
        style: to_scene_text_style(
            resolved_color,
            resolved_font_size,
            title.anchor,
            0.0,
            &theme.typography.title_font_family,
            fw,
            None,
            1.0,
        ),
    });
    if let (Some(subtitle), Some(sy)) = (&title.subtitle, title.subtitle_y) {
        let resolved_sub_color = title_spec
            .and_then(|t| t.subtitle_color.as_deref())
            .and_then(|s| super::color::parse_color(s).ok())
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
            slot: None,
            style: to_scene_text_style(
                resolved_sub_color,
                resolved_sub_font_size,
                title.anchor,
                0.0,
                &theme.typography.font_family,
                None,
                None,
                1.0,
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
    scale_resolve::resolve_param_domains(&mut rendering_spec_for_legend);
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
        // See the per-panel site above: the chart-level application already
        // reported any refusal, so this legend-only re-application is silent.
        let _ = super::apply_color_config_to_color_scale(&mut color_scale, cfg);
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
        out.extend(marks::legend::build_legend(
            legend,
            color_scale.as_ref(),
            theme,
        ));
    }
    // Auxiliary (size / shape) legend blocks stacked beneath the color legend.
    // Each carries its own per-entry color (color_hex) or falls back to the
    // theme mark color, so the color scale is unused but passed for uniformity.
    for aux in &layout.aux_legends {
        out.extend(marks::legend::build_legend(
            aux,
            color_scale.as_ref(),
            theme,
        ));
    }
    Ok(())
}

/// Extract per-node object-constancy keys (spec §4.3 / GH #93) from
/// `encoding.key`, in node order.
///
/// Reads `key_enc.field` straight off the `RecordBatch` at the same
/// `data_indices`-indexed seam as tooltips/hrefs/descriptions — it never
/// looks at `result.nodes` (the JSON scene-node output). That means the
/// vector this returns is identical whether the batch later renders as JSON
/// `nodes` or gets binary-packed by `pack_instances::extract_packed_bytes`
/// (packing runs afterward, on the already-assembled `SceneGraph`, and moves
/// this vector into the `HAS_KEYS` sidecar for qualifying batches — see
/// `pack_instances.rs`). Packed-vs-JSON is purely a downstream serialization
/// choice; key extraction itself is decoupled from it.
///
/// **`data_indices` coupling (deliberate, not decoupled):** `let indices =
/// data_indices?;` short-circuits to `None` — keys silently absent — whenever
/// the caller passes `data_indices: None`. This is spec §4.3 as designed:
/// keys are defined by row identity, and `data_indices` is this codebase's
/// one canonical row-identity vector (the same one `MetadataColumns::
/// build_metadata_for_indices` consumes for tooltips/hrefs/descriptions), so
/// piggybacking on it rather than inventing a second row-tracking mechanism
/// is the intended design, not an oversight. It is safe **today** because
/// every mark builder that can produce a packing-eligible batch (circles via
/// `marks/point.rs`, rects via `marks/bar.rs`/`marks/rect.rs`) always
/// populates `data_indices` through the shared `MarkNodes`/`MetadataColumns`
/// accumulator — see `mark_nodes.rs`'s module doc. **What breaks if a future
/// builder doesn't:** a hypothetical new circle- or rect-emitting builder
/// that skips `data_indices` (returns `None` for it while still setting
/// `encoding.key`) would silently produce `MarkBatch.keys == None` for that
/// batch — no panic, no warning, object constancy quietly degrades to
/// index-zip fallback (spec §4.3's documented no-keys behavior) with no
/// signal that `key=` was ever requested. Any new packing-eligible builder
/// must set `data_indices` (already required for tooltips to work at all) or
/// this silent-drop failure mode reappears.
///
/// **Non-string key columns coerce to an injective identity string, not a
/// display string (GH #93).** Integer, unsigned, float, boolean, and
/// temporal key columns all produce keys via [`col_as_ordinal_category_str`]
/// — the crate's existing identity stringifier (Utf8/LargeUtf8 passthrough,
/// Int*/UInt* via native `.to_string()` with no magnitude ceiling, Float*
/// via `float_as_ordinal_str`, Boolean via `"true"`/`"false"`), already used
/// where `ScaleKind::Ordinal` needs per-row category strings that must match
/// a domain 1:1 — the same injectivity requirement object-constancy keys
/// have. It deliberately does not cover `Timestamp` (see its doc), so
/// temporal columns fall to [`col_as_temporal_epoch_str`]: the raw `i64`
/// epoch value's `.to_string()`, NOT an `f64` round-trip.
///
/// **Do not substitute a display formatter such as `format_numeric` here.**
/// It is not an identity function: it aliases a six-digit database id column
/// to 2 unique keys across 1200 rows, and any present-day Unix-epoch
/// timestamp to 1 unique key for the whole batch — worse than the silent
/// drop it would replace, since a colliding key makes a key-based matcher
/// pair up unrelated rows.
///
/// **Injectivity holds for Utf8/LargeUtf8, Int*/UInt*, and `Timestamp`** — n
/// distinct values in those dtypes produce n distinct key strings, with no
/// magnitude ceiling. It does **not** hold unconditionally for `Float64`/
/// `Float32` or `Boolean`: `float_as_ordinal_str` casts whole-valued floats
/// via `v as i64`, which **saturates** rather than wrapping — every value
/// `>= 2^63` (~9.2e18) collapses to the same `"9223372036854775807"` key,
/// and every `NaN` row collapses to the same `"NaN"` key, regardless of
/// which distinct `NaN` bit pattern produced it. This is a pre-existing
/// property of the shared ordinal stringifier (the same saturation already
/// merges ordinal *domain* categories elsewhere in the crate), not something
/// introduced here, and `>= 2^63` is not a realistic object-constancy key in
/// practice — but a consumer must not assume uniqueness in that corner.
/// `Boolean` is by construction non-injective above 2 rows (only
/// `"true"`/`"false"` exist) — correct coercion, not a defect, but a matcher
/// must not expect more than 2 distinct keys from a boolean column. Null key
/// values from any coercible dtype also collapse together:
/// `unwrap_or_default()` below maps every null to `""`, so multiple null
/// rows share one key. A key column of a dtype neither helper covers
/// (`Duration`, `List`, `Struct`) still falls through to `None`, and still
/// silently degrades to index-zip with no warning — the residual, now
/// narrower, instance of the short-circuit described above.
fn extract_keys(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    data_indices: Option<&[usize]>,
) -> Option<Vec<String>> {
    let key_enc = encoding.key.as_ref()?;
    let col = col_as_ordinal_category_str(batch, &key_enc.field)
        .ok()
        .or_else(|| col_as_temporal_epoch_str(batch, &key_enc.field).ok())?;
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
    const ZOOM_BREAKPOINTS: &[(f64, f64, usize)] =
        &[(0.0, 0.5, 4), (0.5, 2.0, 8), (2.0, 4.0, 16), (4.0, 1e9, 32)];

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
fn apply_polar_node_transform(nodes: &mut Vec<SceneNode>, plot_area: &crate::layout::Rect) {
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
    fn map_pt(
        px: f64,
        py: f64,
        plot_x: f64,
        plot_y: f64,
        plot_w: f64,
        plot_h: f64,
        center_x: f64,
        center_y: f64,
        outer_r: f64,
        tau: f64,
    ) -> (f64, f64) {
        let theta = (px - plot_x) / plot_w * tau;
        let r = (plot_y + plot_h - py) / plot_h * outer_r;
        (center_x + r * theta.sin(), center_y - r * theta.cos())
    }

    let mut replacements: Vec<(usize, SceneNode)> = Vec::new();

    for (idx, node) in nodes.iter_mut().enumerate() {
        match node {
            SceneNode::Circle {
                ref mut cx,
                ref mut cy,
                ..
            } => {
                let (nx, ny) = map_pt(
                    *cx, *cy, plot_x, plot_y, plot_w, plot_h, center_x, center_y, outer_r, TAU,
                );
                *cx = nx;
                *cy = ny;
            }
            SceneNode::Polyline { ref mut points, .. } => {
                for pt in points.iter_mut() {
                    let (nx, ny) = map_pt(
                        pt.0, pt.1, plot_x, plot_y, plot_w, plot_h, center_x, center_y, outer_r,
                        TAU,
                    );
                    pt.0 = nx;
                    pt.1 = ny;
                }
            }
            SceneNode::Line {
                ref mut x1,
                ref mut y1,
                ref mut x2,
                ref mut y2,
                ..
            } => {
                let (nx1, ny1) = map_pt(
                    *x1, *y1, plot_x, plot_y, plot_w, plot_h, center_x, center_y, outer_r, TAU,
                );
                let (nx2, ny2) = map_pt(
                    *x2, *y2, plot_x, plot_y, plot_w, plot_h, center_x, center_y, outer_r, TAU,
                );
                *x1 = nx1;
                *y1 = ny1;
                *x2 = nx2;
                *y2 = ny2;
            }
            SceneNode::Text {
                ref mut x,
                ref mut y,
                ..
            } => {
                let (nx, ny) = map_pt(
                    *x, *y, plot_x, plot_y, plot_w, plot_h, center_x, center_y, outer_r, TAU,
                );
                *x = nx;
                *y = ny;
            }
            SceneNode::Rect {
                x, y, w, h, style, ..
            } => {
                // Convert the Cartesian rect to a polar Polygon by sampling
                // its perimeter. Constant-y (arc) edges get RECT_ARC_SEGMENTS
                // points; constant-x (radial) edges get 2 points (straight).
                // Perimeter order: bottom-left → bottom-right (arc) →
                // top-right (radial) → top-left (arc, reversed) → close.
                let (rx, ry, rw, rh, fill_stroke) = (*x, *y, *w, *h, style.clone());
                let mut pts: Vec<[f64; 2]> = Vec::with_capacity(2 * RECT_ARC_SEGMENTS + 2);
                // Bottom arc: y = ry + rh, x sweeps left → right.
                for i in 0..=RECT_ARC_SEGMENTS {
                    let t = i as f64 / RECT_ARC_SEGMENTS as f64;
                    let px = rx + t * rw;
                    let py = ry + rh;
                    let (nx, ny) = map_pt(
                        px, py, plot_x, plot_y, plot_w, plot_h, center_x, center_y, outer_r, TAU,
                    );
                    pts.push([nx, ny]);
                }
                // Right radial edge: x = rx + rw, y sweeps bottom → top.
                let (nx, ny) = map_pt(
                    rx + rw,
                    ry,
                    plot_x,
                    plot_y,
                    plot_w,
                    plot_h,
                    center_x,
                    center_y,
                    outer_r,
                    TAU,
                );
                pts.push([nx, ny]);
                // Top arc: y = ry, x sweeps right → left.
                for i in (0..=RECT_ARC_SEGMENTS).rev() {
                    let t = i as f64 / RECT_ARC_SEGMENTS as f64;
                    let px = rx + t * rw;
                    let py = ry;
                    let (nx, ny) = map_pt(
                        px, py, plot_x, plot_y, plot_w, plot_h, center_x, center_y, outer_r, TAU,
                    );
                    pts.push([nx, ny]);
                }
                // Left radial edge closes back to start (polygon auto-closes).
                replacements.push((
                    idx,
                    SceneNode::Polygon {
                        rings: vec![pts],
                        style: fill_stroke,
                    },
                ));
            }
            SceneNode::Path {
                ref mut commands, ..
            } => {
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
                        ferrum_scene::PathCmd::MoveTo {
                            ref mut x,
                            ref mut y,
                        } => {
                            let (nx, ny) = map_pt(
                                *x, *y, plot_x, plot_y, plot_w, plot_h, center_x, center_y,
                                outer_r, TAU,
                            );
                            cx_cur = *x;
                            cy_cur = *y;
                            *x = nx;
                            *y = ny;
                        }
                        ferrum_scene::PathCmd::LineTo {
                            ref mut x,
                            ref mut y,
                        } => {
                            let (nx, ny) = map_pt(
                                *x, *y, plot_x, plot_y, plot_w, plot_h, center_x, center_y,
                                outer_r, TAU,
                            );
                            cx_cur = *x;
                            cy_cur = *y;
                            *x = nx;
                            *y = ny;
                        }
                        ferrum_scene::PathCmd::QuadTo {
                            ref mut cx,
                            ref mut cy,
                            ref mut x,
                            ref mut y,
                        } => {
                            let (ncx, ncy) = map_pt(
                                *cx, *cy, plot_x, plot_y, plot_w, plot_h, center_x, center_y,
                                outer_r, TAU,
                            );
                            let (nx, ny) = map_pt(
                                *x, *y, plot_x, plot_y, plot_w, plot_h, center_x, center_y,
                                outer_r, TAU,
                            );
                            cx_cur = *x;
                            cy_cur = *y;
                            *cx = ncx;
                            *cy = ncy;
                            *x = nx;
                            *y = ny;
                        }
                        ferrum_scene::PathCmd::CubicTo {
                            ref mut c1x,
                            ref mut c1y,
                            ref mut c2x,
                            ref mut c2y,
                            ref mut x,
                            ref mut y,
                        } => {
                            let (nc1x, nc1y) = map_pt(
                                *c1x, *c1y, plot_x, plot_y, plot_w, plot_h, center_x, center_y,
                                outer_r, TAU,
                            );
                            let (nc2x, nc2y) = map_pt(
                                *c2x, *c2y, plot_x, plot_y, plot_w, plot_h, center_x, center_y,
                                outer_r, TAU,
                            );
                            let (nx, ny) = map_pt(
                                *x, *y, plot_x, plot_y, plot_w, plot_h, center_x, center_y,
                                outer_r, TAU,
                            );
                            cx_cur = *x;
                            cy_cur = *y;
                            *c1x = nc1x;
                            *c1y = nc1y;
                            *c2x = nc2x;
                            *c2y = nc2y;
                            *x = nx;
                            *y = ny;
                        }
                        ferrum_scene::PathCmd::ArcTo {
                            ref mut x,
                            ref mut y,
                            ..
                        } => {
                            let (nx, ny) = map_pt(
                                *x, *y, plot_x, plot_y, plot_w, plot_h, center_x, center_y,
                                outer_r, TAU,
                            );
                            cx_cur = *x;
                            cy_cur = *y;
                            *x = nx;
                            *y = ny;
                        }
                        ferrum_scene::PathCmd::HLineTo { x: target_x } => {
                            // Polar changes both axes, so convert to LineTo using the
                            // current y position tracked above.
                            let old_x = *target_x;
                            let (nx, ny) = map_pt(
                                old_x, cy_cur, plot_x, plot_y, plot_w, plot_h, center_x, center_y,
                                outer_r, TAU,
                            );
                            cx_cur = old_x;
                            *cmd = ferrum_scene::PathCmd::LineTo { x: nx, y: ny };
                        }
                        ferrum_scene::PathCmd::VLineTo { y: target_y } => {
                            // Polar changes both axes, so convert to LineTo using the
                            // current x position tracked above.
                            let old_y = *target_y;
                            let (nx, ny) = map_pt(
                                cx_cur, old_y, plot_x, plot_y, plot_w, plot_h, center_x, center_y,
                                outer_r, TAU,
                            );
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
    use ferrum_scene::PathCmd;
    use std::f64::consts::TAU;

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
                PathCmd::MoveTo {
                    x: cx - outer_r,
                    y: cy,
                },
                PathCmd::ArcTo {
                    rx: outer_r,
                    ry: outer_r,
                    rotation: 0.0,
                    large_arc: true,
                    sweep: true,
                    x: cx + outer_r,
                    y: cy,
                },
                PathCmd::ArcTo {
                    rx: outer_r,
                    ry: outer_r,
                    rotation: 0.0,
                    large_arc: true,
                    sweep: true,
                    x: cx - outer_r,
                    y: cy,
                },
            ],
            style: ferrum_scene::FillStroke {
                fill: None,
                stroke: Some(axis_color),
                stroke_width: theme.sizes.axis_line_width,
                opacity: 1.0,
                stroke_dash: None,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
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
            nodes.push(SceneNode::Line {
                x1,
                y1,
                x2,
                y2,
                style: stroke.clone(),
            });

            // Label outside the tick
            let lx = cx + (outer_r + label_pad) * theta.sin();
            let ly = cy - (outer_r + label_pad) * theta.cos();
            nodes.push(SceneNode::Text {
                x: lx,
                y: ly,
                content: tick.label.clone(),
                slot: None,
                style: draw::to_scene_text_style(
                    theme.colors.label_color,
                    theme.typography.label_font_size,
                    crate::layout::TextAnchor::Middle,
                    0.0,
                    &theme.typography.font_family,
                    None,
                    None,
                    1.0,
                ),
            });
        }
    }

    nodes
}

// ── Structural feature processing ────────────────────────────────────────────

/// Process structural feature specs (axis breaks, insets).
struct StructuralOutput {
    /// Additional annotation nodes (break indicators, insets).
    extra_annotations: Vec<SceneNode>,
    /// `(axis, BreakResult)` pairs for each BreakAxis spec, used by the
    /// caller to remap primary mark pixel coordinates through the broken scale.
    break_results: Vec<(String, break_axis::BreakResult)>,
}

fn build_structural_nodes(
    structural: &[StructuralSpec],
    scales: &scale_resolve::ResolvedScales,
    plot_area: &crate::layout::Rect,
    theme: &crate::layout::ThemeInputs,
) -> StructuralOutput {
    let mut extra_annotations: Vec<SceneNode> = Vec::new();
    let mut break_results: Vec<(String, break_axis::BreakResult)> = Vec::new();
    // Distinct per Inset item so each embedded body gets its own clip-id
    // namespace (see `inset::build_inset_nodes`); only Inset variants advance it.
    let mut inset_idx: usize = 0;

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
                    let px_x = scales.x.to_pixel_f64(dx).unwrap_or_else(|| {
                        // Fallback for ordinal / out-of-domain: linear interpolation.
                        if let Some((lo, hi)) = scales.x.data_domain() {
                            let frac = if (hi - lo).abs() < f64::EPSILON {
                                0.5
                            } else {
                                (dx - lo) / (hi - lo)
                            };
                            plot_area.x + frac * plot_area.w
                        } else {
                            plot_area.x + plot_area.w * 0.5
                        }
                    });
                    let px_y = scales.y.to_pixel_f64(dy).unwrap_or_else(|| {
                        if let Some((lo, hi)) = scales.y.data_domain() {
                            let frac = if (hi - lo).abs() < f64::EPSILON {
                                0.5
                            } else {
                                (dy - lo) / (hi - lo)
                            };
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
                let inset_nodes = inset::build_inset_nodes(inset_to_build, plot_area, inset_idx);
                inset_idx += 1;
                extra_annotations.extend(inset_nodes);
            }
        }
    }

    StructuralOutput {
        extra_annotations,
        break_results,
    }
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
        SceneNode::Text { x, y, .. } => {
            if axis == "y" {
                *y
            } else {
                *x
            }
        }
        SceneNode::Line { x1, y1, x2, y2, .. } => {
            if axis == "y" {
                (*y1).min(*y2)
            } else {
                (*x1).min(*x2)
            }
        }
        SceneNode::Rect { x, y, w, h, .. } => {
            if axis == "y" {
                *y
            } else {
                *x
            }
        }
        SceneNode::Group { children, .. } => {
            return children
                .iter()
                .any(|c| node_coord_in_range(c, axis, lo, hi));
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
            *coord = remap_coord(*coord, d_lo, d_hi, px_lo, px_hi, br).unwrap_or(BREAK_HIDDEN);
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
                    _ => {
                        *h = 0.0;
                    }
                }
            } else {
                let left = remap_coord(*x, d_lo, d_hi, px_lo, px_hi, br);
                let right = remap_coord(*x + *w, d_lo, d_hi, px_lo, px_hi, br);
                match (left, right) {
                    (Some(l), Some(r)) => {
                        *x = l.min(r);
                        *w = (r - l).abs();
                    }
                    _ => {
                        *w = 0.0;
                    }
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
            *coord = remap_coord(*coord, d_lo, d_hi, px_lo, px_hi, br).unwrap_or(BREAK_HIDDEN);
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
            if axis == "y" {
                remap(y);
            } else {
                remap(x);
            }
        }
        PathCmd::QuadTo { cx, cy, x, y } => {
            if axis == "y" {
                remap(cy);
                remap(y);
            } else {
                remap(cx);
                remap(x);
            }
        }
        PathCmd::CubicTo {
            c1x,
            c1y,
            c2x,
            c2y,
            x,
            y,
        } => {
            if axis == "y" {
                remap(c1y);
                remap(c2y);
                remap(y);
            } else {
                remap(c1x);
                remap(c2x);
                remap(x);
            }
        }
        PathCmd::ArcTo { x, y, .. } => {
            if axis == "y" {
                remap(y);
            } else {
                remap(x);
            }
        }
        PathCmd::HLineTo { x } => {
            if axis != "y" {
                remap(x);
            }
        }
        PathCmd::VLineTo { y } => {
            if axis == "y" {
                remap(y);
            }
        }
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
    if span.abs() < f64::EPSILON {
        return Some(px);
    }
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
/// - `mark_rule` with a channel pattern rule has no geometry for: delegated in
///   full to [`crate::render::marks::rule::RuleShape::resolve`], the same
///   derivation the renderer uses to pick its shape (batch-A Task 13 spec c3,
///   ruled 2026-09-01). Running one function in both places is the point: the
///   gate cannot accept a pattern `marks/rule.rs::build` has no branch for,
///   which is how a presence-legal shape used to fall through to a blank
///   panel. Refused patterns are a second endpoint with no anchor to pair it
///   with (`x2`/`y2` bound with no `x`/`y` to range from) and nothing
///   positional bound at all. **Totality invariant (spec c2):** once a shape
///   resolves, `build` renders it or raises a typed `RenderError` — there is
///   no terminal fallback left to fall through to. See `marks/rule.rs`'s
///   module doc for the shape table and the scale-keyed positional read.
///
/// `channel`/`mark`'s bare-channel checks act on `encoding`, which is the
/// RESOLVED (post-`CoordFlip`) layer encoding — so is `channel`/`hint_alt_channel`
/// on the resulting error; `coord_flipped` (R3) lets `Display` un-flip them back
/// to the token the user actually wrote. `mark_bar`'s hint names both x2 AND y2
/// (flip-symmetric: whichever letters the user wrote, both are still bound after
/// the swap), so it carries no `hint_alt_channel`. `channel: "positional"` names
/// the whole family since no single bound channel is at fault, unlike
/// `mark_area`/`mark_bar` above.
fn validate_mark_encoding(
    mark: &crate::spec::mark::Mark,
    encoding: &crate::spec::encoding::Encoding,
    coord_flipped: bool,
) -> Result<(), RenderError> {
    use crate::spec::mark::Mark;
    match mark {
        Mark::Area if encoding.x2.is_some() => Err(RenderError::UnsupportedChannelCombination {
            mark: "mark_area",
            channel: "x2",
            hint: "use {alt}= for a vertical band area, or use mark_rect for a 2-D extent",
            hint_alt_channel: Some("y2"),
            coord_flipped,
        }),
        Mark::Bar if encoding.x2.is_some() && encoding.y2.is_some() => {
            Err(RenderError::UnsupportedChannelCombination {
                mark: "mark_bar",
                channel: "x2 and y2",
                hint: "a 2-D extent (both x2= and y2=) is a rectangle; use mark_rect instead",
                hint_alt_channel: None,
                coord_flipped,
            })
        }
        // Rule's gate IS its geometry derivation: the same
        // `RuleShape::resolve` the renderer calls, run here so an unsupported
        // channel pattern is refused before any mark is built (batch-A Task 13
        // spec c3). One function, so the gate cannot accept a pattern the
        // renderer has no branch for — the drift that let a presence-legal
        // shape fall through to a blank panel.
        Mark::Rule => {
            crate::render::marks::rule::RuleShape::resolve(encoding, coord_flipped)?;
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Rect;
    use ferrum_scene::{FillStroke, PathCmd, SceneNode};

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
        let plot_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        };
        // A Path node placed at the top-left corner of the plot area in Cartesian space.
        // After polar transform this coordinate will NOT remain at (0, 0).
        let cartesian_x = 0.0_f64;
        let cartesian_y = 0.0_f64;

        let mut nodes = vec![SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo {
                    x: cartesian_x,
                    y: cartesian_y,
                },
                PathCmd::LineTo { x: 100.0, y: 100.0 },
                PathCmd::Close,
            ],
            style: default_fill_stroke(),
            closed: true,
        }];

        apply_polar_node_transform(&mut nodes, &plot_area);

        // The Path node must still be a Path node (no type change).
        match &nodes[0] {
            SceneNode::Path { commands, .. } => {
                // The MoveTo endpoint for (0, 0) in Cartesian maps to a non-zero polar
                // coordinate. Specifically: theta = 0, r = 1 * outer_r → (cx, cy - r).
                let outer_r = plot_area.w.min(plot_area.h) / 2.0; // 100.0
                let center_x = plot_area.x + plot_area.w / 2.0; // 100.0
                let center_y = plot_area.y + plot_area.h / 2.0; // 100.0
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
        let plot_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        };
        let mut nodes = vec![SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 50.0, y: 100.0 },
                PathCmd::HLineTo { x: 150.0 },
                PathCmd::VLineTo { y: 50.0 },
                PathCmd::Close,
            ],
            style: default_fill_stroke(),
            closed: true,
        }];

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
                    "HLineTo must become LineTo, got: {:?}",
                    commands[1]
                );
                assert!(
                    matches!(commands[2], PathCmd::LineTo { .. }),
                    "VLineTo must become LineTo, got: {:?}",
                    commands[2]
                );
            }
            other => panic!("expected Path node, got {other:?}"),
        }
    }

    /// B5: Control points in QuadTo/CubicTo must also be transformed.
    #[test]
    fn b5_quadto_control_points_are_transformed() {
        let plot_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        };
        let mut nodes = vec![SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 0.0, y: 100.0 },
                PathCmd::QuadTo {
                    cx: 50.0,
                    cy: 0.0,
                    x: 100.0,
                    y: 100.0,
                },
                PathCmd::Close,
            ],
            style: default_fill_stroke(),
            closed: true,
        }];

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
        scale_resolve::resolve_param_domains(&mut spec);
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
        scale_resolve::resolve_param_domains(&mut spec);
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
        scale_resolve::resolve_param_domains(&mut spec);
        assert_eq!(x_domain(&spec), None);
    }

    #[test]
    fn resolve_param_domains_noop_when_no_params() {
        // The byte-stability gate: empty params → spec unchanged.
        let mut spec = spec_with_x_domain_param(Vec::new());
        let before = serde_json::to_string(&spec).unwrap();
        scale_resolve::resolve_param_domains(&mut spec);
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
        let bindings =
            collect_param_bindings(&spec, &[], &scale_resolve::YScaleSlots::default(), 1);
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
        let bindings =
            collect_param_bindings(&spec, &[], &scale_resolve::YScaleSlots::default(), 3);
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
        let bindings =
            collect_param_bindings(&spec, &[], &scale_resolve::YScaleSlots::default(), 1);
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
        let bindings =
            collect_param_bindings(&spec, &[], &scale_resolve::YScaleSlots::default(), 1);
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
        assert!(
            collect_param_bindings(&bare, &[], &scale_resolve::YScaleSlots::default(), 1)
                .is_empty()
        );
    }

    /// Build a minimal `LayerPrepared` carrying a `y` domainParam scale.
    fn layer_with_y_domain_param(
        name: &str,
        independent_y: bool,
    ) -> crate::render::prepare::LayerPrepared {
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
            color_is_own: false,
            x_is_own: false,
            y_is_own: true,
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
        assert!(bindings
            .iter()
            .all(|b| b.y_slot == 1 && b.channel.as_deref() == Some("y")));
    }

    fn layout_with_subtitle(subtitle: &str) -> LayoutResult {
        LayoutResult {
            viewport: Rect {
                x: 0.0,
                y: 0.0,
                w: 400.0,
                h: 300.0,
            },
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
            plot_region: Rect::ZERO,
        }
    }

    /// `configure_title(subtitle_font_size=…, subtitle_color=…)` flows through the
    /// chart-config → theme path and reaches the rendered subtitle. The chart-level
    /// subtitle styling lives on the theme (populated by `apply_chart_config`); the
    /// per-chart `spec.title` is `None` here, exactly the `configure_title` case.
    #[test]
    fn build_title_applies_chart_config_subtitle_styling() {
        let spec = spec_with_x_domain_param(Vec::new());
        assert!(
            spec.title.is_none(),
            "this test exercises the chart-config path, not spec.title"
        );
        let layout = layout_with_subtitle("Styled subtitle");

        let mut theme = ThemeInputs::default();
        theme.typography.subtitle_font_size = Some(22.0);
        theme.colors.subtitle_color = Some(super::super::color::parse_color("#ff0000").unwrap());

        let mut nodes = Vec::new();
        build_title(&layout, &spec, &theme, &mut nodes);

        // [0] = title line, [1] = subtitle line.
        let subtitle_node = nodes
            .iter()
            .find_map(|n| match n {
                SceneNode::Text { content, style, .. } if content == "Styled subtitle" => {
                    Some(style)
                }
                _ => None,
            })
            .expect("subtitle text node must be emitted");
        assert_eq!(subtitle_node.font_size, 22.0);
        assert_eq!(
            subtitle_node.color,
            to_scene_color(super::super::color::parse_color("#ff0000").unwrap()),
        );
    }

    /// Batch A Task 8 sweep: the per-chart `Title(color=…, subtitle_color=…)`
    /// strings were hex-only, so a CSS name or `rgb()` string silently fell back
    /// to the theme title color. All three spellings now paint the same text.
    #[test]
    fn build_title_colors_accept_named_and_rgb_forms_identically_to_hex() {
        let title_colors = |spelling: &str| -> (ferrum_scene::Color, ferrum_scene::Color) {
            let mut spec = spec_with_x_domain_param(Vec::new());
            spec.title = Some(crate::spec::title::TitleSpec {
                text: "Main Title".to_string(),
                subtitle: Some("Sub".to_string()),
                color: Some(spelling.to_string()),
                subtitle_color: Some(spelling.to_string()),
                ..Default::default()
            });
            let mut nodes = Vec::new();
            build_title(&layout_with_subtitle("Sub"), &spec, &ThemeInputs::default(), &mut nodes);
            let color_of = |want: &str| {
                nodes
                    .iter()
                    .find_map(|n| match n {
                        SceneNode::Text { content, style, .. } if content == want => {
                            Some(style.color)
                        }
                        _ => None,
                    })
                    .unwrap_or_else(|| panic!("no text node for {want:?}"))
            };
            (color_of("Main Title"), color_of("Sub"))
        };
        let expected = to_scene_color(super::super::color::parse_color("#4682b4").unwrap());
        for spelling in ["steelblue", "rgb(70, 130, 180)", "#4682b4"] {
            assert_eq!(
                title_colors(spelling),
                (expected, expected),
                "{spelling:?} must color both title lines"
            );
        }
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
                SceneNode::Text { content, style, .. } if content == "Default subtitle" => {
                    Some(style)
                }
                _ => None,
            })
            .expect("subtitle text node must be emitted");
        assert_eq!(
            subtitle_node.font_size,
            theme.typography.title_font_size * 0.85
        );
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
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "x".into(),
                    ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "y".into(),
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
        let prep =
            super::super::prepare::prepare_render_inputs(&spec, &batch, &theme, None).unwrap();
        let chart_config = super::super::chart_config::ChartConfig::default();

        // Two panels with deliberately different plot_area widths.
        let panel_narrow = crate::layout::PanelLayout {
            plot_area: crate::layout::Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 200.0,
            },
            ..Default::default()
        };
        let panel_wide = crate::layout::PanelLayout {
            plot_area: crate::layout::Rect {
                x: 0.0,
                y: 0.0,
                w: 400.0,
                h: 200.0,
            },
            ..Default::default()
        };

        let mut warnings = Vec::new();
        // One implicit layer → one layer batch (the whole panel batch).
        let layer_batches = vec![batch.clone()];
        let ctx = PanelResolveCtx {
            spec: &spec,
            prep: &prep,
            theme: &theme,
            chart_config: &chart_config,
            leaf_scales: None,
        };
        let (_spec_a, scales_a) = resolve_panel_scales(
            &ctx,
            &panel_narrow,
            &batch,
            &layer_batches,
            &mut warnings,
            (0.0, 0.0),
        )
        .unwrap();
        let (_spec_b, scales_b) = resolve_panel_scales(
            &ctx,
            &panel_wide,
            &batch,
            &layer_batches,
            &mut warnings,
            (0.0, 0.0),
        )
        .unwrap();

        let (a_lo, a_hi) = scales_a.x.pixel_range();
        let (b_lo, b_hi) = scales_b.x.pixel_range();
        // Each panel's x range spans its own plot_area width.
        assert!(
            (a_hi - a_lo).abs() <= 100.0 + 1e-6,
            "narrow panel x range within 100px"
        );
        assert!(
            (b_hi - b_lo).abs() > 100.0 + 1e-6,
            "wide panel x range exceeds 100px"
        );
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
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::layer::Layer;
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let y_enc = |field: &str| Layer {
            mark: Mark::Line,
            encoding: Encoding {
                y: Some(EncodingSpec {
                    field: field.into(),
                    ..Default::default()
                }),
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
                x: Some(EncodingSpec {
                    field: "x".into(),
                    ..Default::default()
                }),
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

    /// Three layers, layers 1 AND 2 independent-y (two secondary axes). Each
    /// slot's data is on a clearly separable magnitude — y0 ∈ [1,3],
    /// y1 ∈ [100,300], y2 ∈ [1000,3000] — so a slot cross-wire is observable in
    /// the resolved domain. Used to prove the ONE layer→slot plan (GH #72) keeps
    /// the prepare axis inputs, the per-panel slots, and the axis-group tags in
    /// lock-step for more than a single secondary axis.
    fn three_layer_two_independent_spec() -> (ChartSpec, RecordBatch) {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::layer::Layer;
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let y_layer = |field: &str, independent: bool| Layer {
            mark: Mark::Line,
            encoding: Encoding {
                y: Some(EncodingSpec {
                    field: field.into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: independent,
        };

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "x".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: Some(vec![
                y_layer("y0", false),
                y_layer("y1", true),
                y_layer("y2", true),
            ]),
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
            Field::new("y2", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
                Arc::new(Float64Array::from(vec![1000.0, 2000.0, 3000.0])),
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
            f.map(|field| {
                vec![EncodingSpec {
                    field: field.into(),
                    ..Default::default()
                }]
            })
        };
        spec.encoding.tooltip_fields = mk_fields(chart_tooltip_field);
        let layers = spec
            .layers
            .as_mut()
            .expect("two_layer_dual_y_spec always sets layers");
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
        let viewport = crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        };
        let config = super::super::config::RenderConfig::default();
        let chart_config = super::super::chart_config::ChartConfig::default();

        let prep = super::super::prepare::prepare_render_inputs(spec, batch, &theme, None).unwrap();
        let mut warnings = prep.warnings.clone();
        let metrics = super::super::font::FontdueMetrics::new();
        let layout = crate::layout::compute_layout(
            spec,
            &theme,
            viewport,
            &prep.axes,
            &prep.facet_groups,
            &prep.legend_entries,
            prep.legend_title.clone(),
            prep.colorbar.as_ref(),
            &metrics,
            &crate::layout::legend::LegendOverrides::default(),
            &prep.aux_legends,
            crate::layout::CompositeLayoutSeam::default(),
        )
        .unwrap();

        build_scene(
            spec,
            &prep,
            &layout,
            &theme,
            &config,
            &mut warnings,
            &chart_config,
            None,
        )
        .unwrap()
    }

    // ── Task 4 remediation (spec §4.4, 2026-08-28 T4 amendment): layered
    // mark_text color honesty ────────────────────────────────────────────────
    //
    // These exercise the FULL `prepare → layout → build_scene` pipeline (not
    // `marks::text::build` directly) because the bug lived at the layer-
    // inheritance seam (`LayerPrepared::from_chart_and_layer`,
    // `render/prepare/mod.rs`) and the mark_style-resolution seam
    // (`scene_build.rs`'s synthetic per-layer `ChartSpec`, whose `mark_style`
    // field is NOT overridden per layer) — a unit test that builds `DrawCtx`
    // by hand cannot see either seam.

    /// Two-layer (bar + text) spec/batch. `bar_fill`/`text_fill` map to each
    /// layer's own `mark_style.fill=`. `bar_own_color_field`/
    /// `text_own_color_field` each bind that layer's OWN `encoding.color`.
    /// `chart_color_field` sets the CHART-level `encoding.color` (inherited by
    /// whichever layer declares no color of its own, under normal
    /// `inherit_from` rules — mirrors `heatmap(annot=True)`'s colored-cells +
    /// colorless-annotation-labels shape). `chart_level_fill` sets the
    /// CHART-level `mark_style.fill=` directly, independently of `bar_fill`/
    /// `text_fill` — lets a test isolate the chart-level `mark_style`
    /// fallback (`LayerPrepared::from_chart_and_layer`'s `or_else`) from a
    /// layer's own `mark_style`.
    ///
    /// **What Python's `LayerChart` lowering actually emits** for
    /// `mark_bar(fill=...) + mark_text()`: it COPIES layer 0's own mark
    /// kwargs up onto `ChartSpec.mark_style` — layer 0 keeps its own
    /// `mark_style` too, both carry the fill (verified against the built
    /// extension's lowered JSON). So the real combined shape is `bar_fill`
    /// AND `chart_level_fill` both set to the same value, not
    /// `chart_level_fill` alone; a test wanting the actual Python-emittable
    /// shape must pass both.
    ///
    /// Panel-wide scale resolution (`resolve_panel_scales` in this module)
    /// only builds `ctx.scales.color` from the chart level merged with
    /// **layer 0's** (the bar layer's) encoding — a non-primary layer's color
    /// channel alone, with no chart-level or layer-0 counterpart, never gets
    /// a scale built for it at all. This is a real limitation at the Rust
    /// seam, but Python's `Chart.__add__` mirrors the mark_style-kwargs copy
    /// above: a layer's own `encode(color=...)` is ALSO copied up to the
    /// chart-level `encoding.color` (verified: `mark_bar() +
    /// mark_text().encode(color="cat")` lowers with chart-level `color="cat"`
    /// and renders correctly), so this Rust-only limitation is never actually
    /// reachable through Python's `+` — a test for "the text layer's own
    /// color IS honored" should use `chart_color_field` (mirroring what `+`
    /// really produces), not `bar_own_color_field`. `bar_own_color_field`
    /// stays available for directly exercising the Rust-seam-only shape
    /// (color declared on a non-primary layer with no chart-level
    /// counterpart at all), which is a real code path even though `+` cannot
    /// currently construct it.
    fn bar_text_layered_spec(
        bar_fill: Option<&str>,
        text_fill: Option<&str>,
        bar_own_color_field: Option<&str>,
        text_own_color_field: Option<&str>,
        chart_color_field: Option<&str>,
        chart_level_fill: Option<&str>,
    ) -> (ChartSpec, RecordBatch) {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::layer::Layer;
        use crate::spec::mark_style::MarkKwargsSpec;
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let fill_style = |hex: Option<&str>| {
            hex.map(|h| MarkKwargsSpec { fill: Some(h.into()), ..Default::default() })
        };
        let xy_encoding = || Encoding {
            x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
            ..Default::default()
        };

        let bar_layer = Layer {
            mark: Mark::Bar,
            encoding: Encoding {
                color: bar_own_color_field.map(|f| EncodingSpec { field: f.into(), ..Default::default() }),
                ..xy_encoding()
            },
            transforms: Vec::new(),
            mark_style: fill_style(bar_fill),
            data_source: None, position: None, blend: None, name: None, independent_y: false,
        };
        let text_layer = Layer {
            mark: Mark::Text,
            encoding: Encoding {
                color: text_own_color_field.map(|f| EncodingSpec { field: f.into(), ..Default::default() }),
                ..xy_encoding()
            },
            transforms: Vec::new(),
            mark_style: fill_style(text_fill),
            data_source: None, position: None, blend: None, name: None, independent_y: false,
        };

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Bar,
            encoding: Encoding {
                color: chart_color_field.map(|f| EncodingSpec { field: f.into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: Some(vec![bar_layer, text_layer]),
            coord: None,
            mark_style: fill_style(chart_level_fill),
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
            Field::new("cat", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ]).unwrap();
        (spec, batch)
    }

    /// Every `SceneNode::Text`'s resolved color, across every panel/mark-batch
    /// in the scene, in emission order.
    fn text_node_colors(scene: &ferrum_scene::SceneGraph) -> Vec<ferrum_scene::Color> {
        scene.panels.iter()
            .flat_map(|p| p.marks.iter())
            .filter(|m| m.kind == ferrum_scene::MarkBatchKind::Text)
            .flat_map(|m| m.nodes.iter())
            .filter_map(|n| if let SceneNode::Text { style, .. } = n { Some(style.color) } else { None })
            .collect()
    }

    /// `mark_bar() + mark_text(fill="#ff0000")`: the text layer's OWN
    /// `fill=` must be honored even though it is layer 1 of a `LayerChart`
    /// (the finding: `ctx.spec.mark_style` reads the CHART-level kwargs in a
    /// layered chart, not this layer's, so the old raw-kwarg gate silently
    /// dropped this).
    #[test]
    fn layered_bar_text_own_fill_is_honored() {
        let (spec, batch) = bar_text_layered_spec(None, Some("#ff0000"), None, None, None, None);
        let scene = build_scene_for(&spec, &batch);
        let colors = text_node_colors(&scene);
        assert_eq!(colors.len(), 2, "expected 2 text nodes");
        let red = ferrum_scene::Color { r: 0xff, g: 0x00, b: 0x00, a: 255 };
        assert!(colors.iter().all(|c| *c == red),
            "text layer's own fill='#ff0000' must be honored inside a LayerChart; got {colors:?}");
    }

    /// `mark_bar(fill="#00aa00") + mark_text()`, with the bar's fill placed
    /// on the BAR LAYER's own `mark_style` (not the chart level): the bar
    /// layer's fill must never leak into the text layer's color. Before the
    /// `fill_is_user_set` flag moved onto the layer-resolved `MarkPaint`,
    /// `ctx.spec.mark_style` (the chart-level kwargs) reported "fill was set"
    /// for every layer, wrongly triggering `ctx.mark_style.paint.fill` (the
    /// TEXT layer's own resolved paint, which theme-defaults to `mark_color`,
    /// not `font_color`) for a text layer that set no fill at all.
    ///
    /// NOTE: this shape alone does not discriminate the cycle-2 finding
    /// (`prepare/mod.rs`'s chart-level `mark_style` fallback) — see
    /// `layered_chart_level_hoisted_fill_does_not_leak_into_text` below for
    /// the shape Python's `LayerChart` lowering actually produces.
    #[test]
    fn layered_bar_fill_does_not_leak_into_text_color() {
        let (spec, batch) = bar_text_layered_spec(Some("#00aa00"), None, None, None, None, None);
        let scene = build_scene_for(&spec, &batch);
        let colors = text_node_colors(&scene);
        assert_eq!(colors.len(), 2);
        let font_color = to_scene_color(ThemeInputs::default().colors.font_color);
        assert!(colors.iter().all(|c| *c == font_color),
            "the bar layer's fill must not leak into text's color; expected theme font color, got {colors:?}");
        let green = ferrum_scene::Color { r: 0x00, g: 0xaa, b: 0x00, a: 255 };
        assert!(colors.iter().all(|c| *c != green), "text must not render the bar's green fill");
    }

    /// The REAL Python-lowered shape for `mark_bar(fill="#00aa00") +
    /// mark_text()`: `LayerChart` lowering COPIES layer 0's (the bar's) mark
    /// kwargs up onto the CHART-level `ChartSpec.mark_style` — layer 0 keeps
    /// its own `mark_style` too, so BOTH `bar_fill` and `chart_level_fill`
    /// are set here (only the TEXT layer's own `mark_style` stays `None`).
    /// Cycle-2 finding: `LayerPrepared::from_chart_and_layer`'s
    /// `layer.mark_style.clone().or_else(|| spec.mark_style.clone())`
    /// fallback let a kwarg-less text layer resolve that copied chart-level
    /// fill — passing `fill_is_user_set = true` with the BAR's green — even
    /// though `layered_bar_fill_does_not_leak_into_text_color` above (which
    /// sets ONLY `bar_fill`, with chart-level `mark_style` absent — a shape
    /// `+` never actually produces) already passed. This is the shape that
    /// must stay theme font color.
    #[test]
    fn layered_chart_level_hoisted_fill_does_not_leak_into_text() {
        let (spec, batch) =
            bar_text_layered_spec(Some("#00aa00"), None, None, None, None, Some("#00aa00"));
        let scene = build_scene_for(&spec, &batch);
        let colors = text_node_colors(&scene);
        assert_eq!(colors.len(), 2);
        let font_color = to_scene_color(ThemeInputs::default().colors.font_color);
        assert!(colors.iter().all(|c| *c == font_color),
            "the chart-level (hoisted-from-layer-0) fill must not leak into a kwarg-less text layer's color; expected theme font color, got {colors:?}");
        let green = ferrum_scene::Color { r: 0x00, g: 0xaa, b: 0x00, a: 255 };
        assert!(colors.iter().all(|c| *c != green),
            "text must not render the hoisted bar fill via the chart-level mark_style fallback");
    }

    /// Chart-level `color=` + a colorless text layer (the `heatmap(annot=True)`
    /// shape: colored cells, annotation labels with no color of their own):
    /// the inherited chart-level color channel must NOT color the text layer.
    /// Every label stays theme font color, not each row's inherited-legend
    /// color (which, for `heatmap`, sits the label on its own cell's fill —
    /// often invisible).
    #[test]
    fn layered_chart_level_color_does_not_inherit_into_text() {
        let (spec, batch) = bar_text_layered_spec(None, None, None, None, Some("cat"), None);
        let scene = build_scene_for(&spec, &batch);
        let colors = text_node_colors(&scene);
        assert_eq!(colors.len(), 2);
        let font_color = to_scene_color(ThemeInputs::default().colors.font_color);
        assert!(colors.iter().all(|c| *c == font_color),
            "an inherited chart-level color channel must not color a text layer with no color of its own; got {colors:?}");
        // Sanity: the two rows differ in `cat`, so if inheritance HAD leaked
        // through, the two labels would differ from each other too.
        assert_eq!(colors[0], colors[1]);
    }

    /// A text layer's OWN `encode(color=...)` — declared directly on that
    /// layer — IS honored: distinct categories resolve to distinct colors,
    /// none of them the font-color fallback. Uses the Python-emittable shape
    /// (`chart_color_field` + matching `text_own_color_field`, not
    /// `bar_own_color_field`): `Chart.__add__` copies a layer's own
    /// `encode(color=...)` up to the chart level too (see
    /// `bar_text_layered_spec`'s doc comment), so the chart-level channel is
    /// always present alongside the text layer's own declaration in
    /// practice. This guards the current exemption seam
    /// (`build_panel_mark_batches`, scene_build.rs:~1203-1206): the text
    /// layer's own `color` declaration means `LayerPrepared.color_is_own` is
    /// `true`, so the `!layer.color_is_own` branch that clears `color` on
    /// the DrawCtx-local encoding copy must NOT fire here — the per-row color
    /// read has to see the field and resolve it normally, not fall back to
    /// theme font color the way an over-eager exemption (one that fired
    /// regardless of `color_is_own`) would.
    #[test]
    fn layered_text_own_color_encoding_is_honored() {
        let (spec, batch) = bar_text_layered_spec(None, None, None, Some("cat"), Some("cat"), None);
        let scene = build_scene_for(&spec, &batch);
        let colors = text_node_colors(&scene);
        assert_eq!(colors.len(), 2);
        assert_ne!(colors[0], colors[1],
            "the text layer's own color-channel rows (distinct categories) must resolve to distinct colors; got {colors:?}");
        let font_color = to_scene_color(ThemeInputs::default().colors.font_color);
        assert!(colors.iter().all(|c| *c != font_color),
            "the text layer's own color channel must not fall back to theme font color; got {colors:?}");
    }

    // ── Batch-A T5d: the own-color exemption widened to every mark ─────────
    // Root-caused against `fm.roc_chart`: the desugar's `reference` (chance-
    // diagonal) `line` layer declares no `color` of its own and a literal
    // `stroke="#AAAAAA"` override, but shares the primary batch (no
    // `data_source`) with the `line` layer that DOES own `color="class"`.
    // The figure function additionally sets the CHART-level `encoding.color`
    // to the same field (for the shared legend), so `Encoding::inherit_from`
    // filled the reference layer's `LayerPrepared.encoding.color` in from
    // the chart level even though the layer never declared it — and, before
    // this fix, only `Mark::Text` had its inherited-only `color` cleared
    // before reaching the mark builder. `mark_line`'s own per-row grouping
    // then read the (inherited) `class` column straight off the reference
    // layer's shared batch: one dashed polyline per class, each stroked with
    // that class's legend color, instead of one dashed line in the layer's
    // own `#AAAAAA`.

    /// Chart-level `color="class"` (mirroring `roc_chart`'s shared-legend
    /// copy-up) + a two-layer `line`+`line` chart: layer 0 owns
    /// `color="class"` (the real per-class curves); layer 1 is the
    /// `reference`-shaped layer — no color of its own, a literal
    /// `stroke=`/`stroke_dash=` override, no `data_source` (shares layer 0's
    /// batch, which carries the `class` column). `n_classes` rows repeat per
    /// class so `build_color_detail_groups` has real multi-point groups to
    /// fan out over if the leak is present.
    fn roc_reference_line_layered_spec(n_classes: usize) -> (ChartSpec, RecordBatch) {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::layer::Layer;
        use crate::spec::mark_style::MarkKwargsSpec;
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let curve_layer = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                color: Some(EncodingSpec { field: "class".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), mark_style: None,
            data_source: None, position: None, blend: None, name: Some("line".into()),
            independent_y: false,
        };
        let reference_layer = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            mark_style: Some(MarkKwargsSpec {
                stroke: Some("#AAAAAA".into()),
                stroke_dash: Some(vec![4.0, 4.0]),
                ..Default::default()
            }),
            data_source: None, position: None, blend: None, name: Some("reference".into()),
            independent_y: false,
        };
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                // Mirrors `roc_chart`'s chart-level `color="class"` (set for
                // the shared legend, not declared by either raw layer above).
                color: Some(EncodingSpec { field: "class".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None,
            layers: Some(vec![curve_layer, reference_layer]),
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };

        let rows_per_class = 4;
        let n = n_classes * rows_per_class;
        let xs: Vec<f64> = (0..n).map(|i| (i % rows_per_class) as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| xs[i] * 2.0).collect();
        let classes: Vec<String> = (0..n).map(|i| (i / rows_per_class).to_string()).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("class", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(classes)),
        ]).unwrap();
        (spec, batch)
    }

    /// Every `SceneNode::Polyline`'s stroke color, in one mark batch, in
    /// emission order.
    fn polyline_strokes(batch: &ferrum_scene::MarkBatch) -> Vec<ferrum_scene::Color> {
        batch.nodes.iter()
            .filter_map(|n| if let SceneNode::Polyline { style, .. } = n { Some(style.color) } else { None })
            .collect()
    }

    /// Variant of [`roc_reference_line_layered_spec`] with the "reference"
    /// layer's literal `mark_style` (`stroke="#AAAAAA"`) removed — the
    /// `catplot(kind="box", hue=x)` shape: a layer with no color of its own
    /// AND no literal paint override, sharing the primary batch, relying
    /// entirely on the chart-level `color` it inherits to vary its own
    /// per-row/per-group paint. This is the NEGATIVE branch of the gate: it
    /// must keep fanning out and keep inheriting the legend colors, exactly
    /// like the curve layer. Both `layered_line_reference_layer_*` tests
    /// above, and every neighboring `layered_*` test, pass unchanged under a
    /// blanket `if !layer.color_is_own { color = None }` (the coder's
    /// rejected first attempt) — this fixture is what actually discriminates
    /// that regression from the correct, literal-paint-gated fix, so a
    /// reversion to the blanket form fails HERE in `cargo test`, not only in
    /// the Python golden (`tests/test_phase_9_e2e.py::test_catplot_box_golden`).
    fn roc_reference_line_layered_spec_no_own_paint(n_classes: usize) -> (ChartSpec, RecordBatch) {
        let (mut spec, batch) = roc_reference_line_layered_spec(n_classes);
        let layers = spec.layers.as_mut().expect("roc_reference_line_layered_spec always sets layers");
        layers[1].mark_style = None;
        (spec, batch)
    }

    /// Binary-shaped repro (2 classes): before the fix, the `reference`
    /// layer inherited `color="class"` from the chart level and rendered one
    /// polyline colored with the FIRST class's legend color (`#2563eb`) —
    /// the exact `roc_chart` symptom (stroke flips `#aaaaaa` → `#2563eb`).
    /// After the fix: exactly one polyline, stroked with the layer's own
    /// literal `#AAAAAA`.
    #[test]
    fn layered_line_reference_layer_keeps_own_stroke_not_first_class_color() {
        let (spec, batch) = roc_reference_line_layered_spec(2);
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        let line_batches: Vec<&ferrum_scene::MarkBatch> = panel.marks.iter()
            .filter(|m| m.kind == ferrum_scene::MarkBatchKind::Line)
            .collect();
        assert_eq!(line_batches.len(), 2, "expected one mark batch per layer (curve + reference)");

        let reference_strokes = polyline_strokes(line_batches[1]);
        assert_eq!(reference_strokes.len(), 1,
            "the reference layer must collapse to exactly one polyline (no color-driven fan-out); got {} polylines with strokes {:?}",
            reference_strokes.len(), reference_strokes);
        let gray = ferrum_scene::Color { r: 0xAA, g: 0xAA, b: 0xAA, a: 255 };
        assert_eq!(reference_strokes[0], gray,
            "the reference layer's own literal stroke='#AAAAAA' must win over the inherited chart-level \
             color scale's first-domain-entry color; got {:?}", reference_strokes[0]);
    }

    /// Multiclass-shaped repro (3 classes, coordinator amendment): before the
    /// fix the reference layer's per-row `class` read (inherited, not its
    /// own) fanned the SAME leak out across every class in the domain — one
    /// dashed polyline per class, each in that class's legend color — instead
    /// of collapsing to a single line. Guards both the polyline COUNT and the
    /// per-polyline stroke.
    #[test]
    fn layered_line_reference_layer_does_not_fan_out_per_class_on_multiclass() {
        let (spec, batch) = roc_reference_line_layered_spec(3);
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        let line_batches: Vec<&ferrum_scene::MarkBatch> = panel.marks.iter()
            .filter(|m| m.kind == ferrum_scene::MarkBatchKind::Line)
            .collect();
        assert_eq!(line_batches.len(), 2);

        // Control: the curve layer (its OWN color="class") keeps fanning out
        // one polyline per class — the HIT path must stay unchanged.
        let curve_strokes = polyline_strokes(line_batches[0]);
        assert_eq!(curve_strokes.len(), 3,
            "the curve layer's own color channel must still emit one polyline per class; got {curve_strokes:?}");

        let reference_strokes = polyline_strokes(line_batches[1]);
        assert_eq!(reference_strokes.len(), 1,
            "the reference layer must render exactly one polyline regardless of the chart's class \
             cardinality (no per-class fan-out from an inherited-only color channel); got {} with strokes {:?}",
            reference_strokes.len(), reference_strokes);
        let gray = ferrum_scene::Color { r: 0xAA, g: 0xAA, b: 0xAA, a: 255 };
        assert_eq!(reference_strokes[0], gray,
            "the reference layer's own literal stroke must win; got {:?}", reference_strokes[0]);
    }

    /// Negative branch of the gate (the `catplot(kind="box", hue=x)` shape):
    /// a layer with no color of its own AND no literal `stroke=`/`fill=`
    /// override must still inherit the chart-level `color` and fan out one
    /// polyline per class, in that class's legend color — same as its
    /// sibling curve layer. Proves the fix is gated on
    /// `mark_style.paint.{stroke,fill}_is_user_set`, not on `color_is_own`
    /// alone: temporarily blanking the gate back to
    /// `if !layer.color_is_own { color = None }` (no literal-paint check)
    /// made this test fail (`reference_strokes.len() == 1`, stroked the
    /// theme default) while leaving every other test in this module green —
    /// confirming this is the ONLY Rust-level guard against that regression.
    #[test]
    fn layered_line_reference_layer_without_own_paint_still_inherits_and_fans_out() {
        let (spec, batch) = roc_reference_line_layered_spec_no_own_paint(3);
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        let line_batches: Vec<&ferrum_scene::MarkBatch> = panel.marks.iter()
            .filter(|m| m.kind == ferrum_scene::MarkBatchKind::Line)
            .collect();
        assert_eq!(line_batches.len(), 2);

        let curve_strokes = polyline_strokes(line_batches[0]);
        assert_eq!(curve_strokes.len(), 3, "control: the curve layer's own color must still fan out");

        let reference_strokes = polyline_strokes(line_batches[1]);
        assert_eq!(reference_strokes.len(), 3,
            "a layer with no color of its own AND no literal paint override must keep inheriting the \
             chart-level color and fan out one polyline per class (the catplot(kind=\"box\") shape); \
             got {} polyline(s) with strokes {:?}",
            reference_strokes.len(), reference_strokes);

        // Every inherited stroke must be one of the legend-resolved class
        // colors the sibling curve layer actually used — not the theme
        // default/fallback a cleared color channel would produce.
        for c in &reference_strokes {
            assert!(curve_strokes.contains(c),
                "inherited stroke {c:?} is not one of the curve layer's legend colors {curve_strokes:?}");
        }
        let mut distinct: Vec<ferrum_scene::Color> = Vec::new();
        for c in &reference_strokes {
            if !distinct.contains(c) { distinct.push(*c); }
        }
        assert_eq!(distinct.len(), 3,
            "3 classes must paint 3 distinct inherited colors, matching the legend; got {reference_strokes:?}");
    }

    // ── Cycle-4 quality-review findings: the Text color-inheritance
    // exemption must gate the mark builder's per-row color READ only — never
    // delete `color` from the prepared `LayerPrepared.encoding` itself, which
    // the legend and dodge/stack position grouping also consume directly.
    // These two pipeline tests are what an earlier revision (which DID
    // delete the channel) lacked, per the quality-reviewer's coverage-gap
    // finding: no test placed text at layer 0, and none exercised a position
    // adjustment.

    /// `mark_text() + mark_bar().encode(color="cat")`, TEXT as layer 0: the
    /// legend must still render. `resolve_legend_color_scale` builds the
    /// legend's color scale from `prep.layers[0].encoding.color` — an
    /// earlier revision's exemption deleted that channel from the PREPARED
    /// encoding for any kwarg-less-color Text layer, so with text at index 0
    /// the legend silently vanished even though the bars stayed colored
    /// (their fill comes from the panel-wide scale, built from the chart
    /// level, not from `layers[0]`). Chart-level `color="cat"` mirrors what
    /// Python's `Chart.__add__` actually produces when the bar layer
    /// declares its own `color=` (see `bar_text_layered_spec`'s doc comment).
    #[test]
    fn layered_text_as_layer_zero_with_colored_sibling_still_renders_legend() {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::layer::Layer;
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let xy_encoding = || Encoding {
            x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
            ..Default::default()
        };
        let text_layer = Layer {
            mark: Mark::Text,
            encoding: xy_encoding(), // layer 0; NO own color
            transforms: Vec::new(), mark_style: None,
            data_source: None, position: None, blend: None, name: None, independent_y: false,
        };
        let bar_layer = Layer {
            mark: Mark::Bar,
            encoding: Encoding {
                color: Some(EncodingSpec { field: "cat".into(), ..Default::default() }),
                ..xy_encoding()
            },
            transforms: Vec::new(), mark_style: None,
            data_source: None, position: None, blend: None, name: None, independent_y: false,
        };
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Text,
            encoding: Encoding {
                // Chart-level color, mirroring Python's `+` copy-up of the
                // bar layer's own `encode(color="cat")`.
                color: Some(EncodingSpec { field: "cat".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None,
            layers: Some(vec![text_layer, bar_layer]),
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("cat", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ]).unwrap();

        let scene = build_scene_for(&spec, &batch);
        assert!(!scene.legend.is_empty(),
            "the legend must still render when the colored layer is NOT layer 0 \
             (text, which has no color of its own, is layer 0 here); legend was empty");
    }

    /// Shared bar+text spec/batch for the dodge/stack position-grouping
    /// regression tests. The bar layer OWNS `color="g"`; chart-level
    /// `color="g"` mirrors Python's `+` copy-up (see `bar_text_layered_spec`'s
    /// doc comment); the text layer declares NO color of its own — it only
    /// gets "g" through inheritance. `x="cat"` is ordinal with two
    /// categories, each holding one row per `g` group, so dodge/stack both
    /// have real grouping work to do. `position` is applied identically to
    /// both layers (the canonical value-labels-on-grouped-bars shape).
    fn text_over_bar_position_spec(
        position: crate::spec::position::PositionAdjust,
    ) -> (ChartSpec, RecordBatch) {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec};
        use crate::spec::layer::Layer;
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let xy_ordinal_encoding = || Encoding {
            x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
            ..Default::default()
        };
        let bar_layer = Layer {
            mark: Mark::Bar,
            encoding: Encoding {
                color: Some(EncodingSpec { field: "g".into(), ..Default::default() }),
                ..xy_ordinal_encoding()
            },
            transforms: Vec::new(), mark_style: None,
            data_source: None, position: Some(position.clone()), blend: None, name: None, independent_y: false,
        };
        let text_layer = Layer {
            mark: Mark::Text,
            encoding: xy_ordinal_encoding(), // no own color
            transforms: Vec::new(), mark_style: None,
            data_source: None, position: Some(position), blend: None, name: None, independent_y: false,
        };
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Bar,
            encoding: Encoding {
                color: Some(EncodingSpec { field: "g".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None,
            layers: Some(vec![bar_layer, text_layer]),
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["A", "A", "B", "B"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(StringArray::from(vec!["L1", "L2", "L1", "L2"])),
        ]).unwrap();
        (spec, batch)
    }

    /// `mark_bar(position=Dodge()).encode(color="g") +
    /// mark_text(position=Dodge())`: each label must sit at its OWN bar's
    /// dodged x position, not the undodged group center. `apply_dodge`'s
    /// grouping resolver (`resolve_group_channel`, `position.rs`) falls back
    /// to `encoding.color.field` when no explicit `by=` is given — an
    /// earlier revision's exemption deleted `color` from the text layer's
    /// PREPARED encoding whenever it was inherited-only, so the text layer's
    /// own dodge call saw no grouping channel at all and every label
    /// collapsed to the undodged band center (verified live by the
    /// quality-reviewer: labels at x=191.089/191.089 instead of
    /// 133.163/249.015, matching the bars).
    #[test]
    fn layered_dodged_text_over_colored_bar_tracks_bar_position() {
        use crate::spec::position::PositionAdjust;

        let position = PositionAdjust::Dodge { by: None, padding: 0.05 };
        let (spec, batch) = text_over_bar_position_spec(position);
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        assert_eq!(panel.marks.len(), 2, "one mark batch per layer");

        let bar_centers: Vec<f64> = panel.marks[0].nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { x, w, .. } = n { Some(x + w / 2.0) } else { None }
        }).collect();
        let text_xs: Vec<f64> = panel.marks[1].nodes.iter().filter_map(|n| {
            if let SceneNode::Text { x, .. } = n { Some(*x) } else { None }
        }).collect();
        assert_eq!(bar_centers.len(), 4, "expected 4 dodged bars");
        assert_eq!(text_xs.len(), 4, "expected 4 text labels");

        // Sanity: dodge actually separated the two groups within category
        // "A" — collapsed grouping (the regression) would put both at the
        // same undodged band center.
        assert_ne!(text_xs[0], text_xs[1],
            "dodged labels within the same x-category must land at different x; got {text_xs:?}");

        // The real assertion: each label tracks its own bar's dodge slot.
        for (i, (&bar_cx, &text_x)) in bar_centers.iter().zip(text_xs.iter()).enumerate() {
            assert!((bar_cx - text_x).abs() < 1e-6,
                "label {i} must sit at its own bar's dodged x position; bar center {bar_cx}, label x {text_x}");
        }
    }

    /// `mark_bar(position=Stack()).encode(color="g") +
    /// mark_text(position=Stack())`: each label must land at its OWN bar
    /// segment's stacked y position, not an unstacked/collapsed value.
    /// `apply_stack` has the identical `encoding.color.field` grouping
    /// fallback as dodge, and the identical S3 regression: an inherited-only
    /// `color` deleted from the text layer's prepared encoding meant its own
    /// stack call saw no grouping channel and every label used its raw
    /// (unstacked) y instead of the cumulative segment position.
    /// `StackAnchor::Top` for both layers so the resolved `y` column value
    /// (the segment TOP) is identical for the bar's rect and the text node —
    /// no anchor-specific geometry translation needed to compare them.
    #[test]
    fn layered_stacked_text_over_colored_bar_tracks_bar_position() {
        use crate::spec::position::{PositionAdjust, StackAnchor, StackOffset};

        let position = PositionAdjust::Stack {
            by: None,
            offset: StackOffset::Zero,
            anchor: StackAnchor::Top,
            value_axis: None,
        };
        let (spec, batch) = text_over_bar_position_spec(position);
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        assert_eq!(panel.marks.len(), 2, "one mark batch per layer");

        let bar_tops: Vec<f64> = panel.marks[0].nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { y, .. } = n { Some(*y) } else { None }
        }).collect();
        let text_ys: Vec<f64> = panel.marks[1].nodes.iter().filter_map(|n| {
            if let SceneNode::Text { y, .. } = n { Some(*y) } else { None }
        }).collect();
        assert_eq!(bar_tops.len(), 4, "expected 4 stacked bar segments");
        assert_eq!(text_ys.len(), 4, "expected 4 text labels");

        // Sanity: stacking actually separated the two segments within
        // category "A" — collapsed grouping (the regression) would leave
        // both labels at their raw (unstacked) y.
        assert_ne!(text_ys[0], text_ys[1],
            "stacked labels within the same x-category must land at different y; got {text_ys:?}");

        // The real assertion: each label tracks its own bar segment's
        // stacked (cumulative-top) position.
        for (i, (&bar_top, &text_y)) in bar_tops.iter().zip(text_ys.iter()).enumerate() {
            assert!((bar_top - text_y).abs() < 1e-6,
                "label {i} must sit at its own bar segment's stacked y position; bar top {bar_top}, label y {text_y}");
        }
    }

    // ── Scope extension (spec §4.0, 2026-08-28 user direction): the
    // hoisted-paint fix generalizes from Text-only to EVERY layered mark ────

    /// The mirror of `layered_chart_level_hoisted_fill_does_not_leak_into_text`:
    /// `mark_text(fill="#ff0000") + mark_bar()` — the TEXT layer's fill
    /// (present on its own `mark_style` AND hoisted to the chart level,
    /// matching Python's real lowered shape) must not leak into the BAR
    /// layer, which has no `mark_style` of its own. Before this scope
    /// extension, `without_paint()` was applied only for `Mark::Text`
    /// kwarg-less layers, so a kwarg-less BAR layer still inherited the
    /// hoisted-from-text fill in full — verified live pre-fix:
    /// `mark_text(fill='#ff0000') + mark_bar()` rendered red bars.
    #[test]
    fn layered_text_fill_hoisted_does_not_leak_into_bar_color() {
        let (spec, batch) = bar_text_layered_spec(None, Some("#ff0000"), None, None, None, Some("#ff0000"));
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        assert_eq!(panel.marks.len(), 2, "one mark batch per layer");
        let bar_fills: Vec<Option<ferrum_scene::Color>> = panel.marks[0].nodes.iter().filter_map(|n| {
            if let SceneNode::Rect { style, .. } = n { Some(style.fill) } else { None }
        }).collect();
        assert_eq!(bar_fills.len(), 2, "expected 2 bars");
        let mark_color = to_scene_color(ThemeInputs::default().colors.mark_color);
        assert!(bar_fills.iter().all(|c| *c == Some(mark_color)),
            "bars must render the theme default fill; expected {mark_color:?}, got {bar_fills:?}");
        let red = ferrum_scene::Color { r: 0xff, g: 0x00, b: 0x00, a: 255 };
        assert!(bar_fills.iter().all(|c| *c != Some(red)),
            "the text layer's hoisted fill must not leak into the bar layer's color; got {bar_fills:?}");
    }

    /// Two-layer (bar + point) spec/batch, mirroring `bar_text_layered_spec`
    /// but for `Mark::Point` — needed because point fill defaults differ
    /// from bar/text and the existing helper is bar+text-specific.
    /// `bar_fill`/`point_fill` map to each layer's own `mark_style.fill=`;
    /// `chart_level_fill` sets the CHART-level `mark_style.fill=` directly
    /// (the hoisted-paint shape).
    fn bar_point_layered_spec(
        bar_fill: Option<&str>,
        point_fill: Option<&str>,
        chart_level_fill: Option<&str>,
    ) -> (ChartSpec, RecordBatch) {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::layer::Layer;
        use crate::spec::mark_style::MarkKwargsSpec;
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let fill_style = |hex: Option<&str>| {
            hex.map(|h| MarkKwargsSpec { fill: Some(h.into()), ..Default::default() })
        };
        let xy_encoding = || Encoding {
            x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
            ..Default::default()
        };
        let bar_layer = Layer {
            mark: Mark::Bar,
            encoding: xy_encoding(),
            transforms: Vec::new(),
            mark_style: fill_style(bar_fill),
            data_source: None, position: None, blend: None, name: None, independent_y: false,
        };
        let point_layer = Layer {
            mark: Mark::Point,
            encoding: xy_encoding(),
            transforms: Vec::new(),
            mark_style: fill_style(point_fill),
            data_source: None, position: None, blend: None, name: None, independent_y: false,
        };
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Bar,
            encoding: Encoding::default(),
            transforms: Vec::new(),
            facet: None,
            layers: Some(vec![bar_layer, point_layer]),
            coord: None,
            mark_style: fill_style(chart_level_fill),
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
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ]).unwrap();
        (spec, batch)
    }

    /// `mark_bar(fill="#00aa00") + mark_point()`: the sanctioned behavior
    /// change (spec §4.0) — a kwarg-less POINT layer must now render
    /// default-colored, not the bar's hoisted-to-chart-level fill. Before
    /// this scope extension the Text-only gate meant every OTHER mark
    /// (point, line, area, ...) still leaked sibling paint; this pins the
    /// point case as the representative non-Text example.
    #[test]
    fn layered_bar_fill_hoisted_does_not_leak_into_point_color() {
        let (spec, batch) = bar_point_layered_spec(Some("#00aa00"), None, Some("#00aa00"));
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        assert_eq!(panel.marks.len(), 2, "one mark batch per layer");
        let point_fills: Vec<Option<ferrum_scene::Color>> = panel.marks[1].nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { style, .. } = n { Some(style.fill) } else { None }
        }).collect();
        assert_eq!(point_fills.len(), 2, "expected 2 points");
        let mark_color = to_scene_color(ThemeInputs::default().colors.mark_color);
        assert!(point_fills.iter().all(|c| *c == Some(mark_color)),
            "points must render the theme default fill; expected {mark_color:?}, got {point_fills:?}");
        let green = ferrum_scene::Color { r: 0x00, g: 0xaa, b: 0x00, a: 255 };
        assert!(point_fills.iter().all(|c| *c != Some(green)),
            "the bar layer's hoisted fill must not leak into the point layer's color; got {point_fills:?}");
    }

    /// Own-paint-wins for a NON-Text layer: `mark_bar(fill="#00aa00") +
    /// mark_point(fill="#0000ff")` — the point layer's OWN `mark_style`
    /// short-circuits the `or_else` chart-level fallback entirely (the
    /// `without_paint()` strip only ever applies to the FALLBACK value, never
    /// to a layer's own declared `mark_style`), so the point renders its own
    /// blue, not the bar's green and not a paint-stripped fallback.
    #[test]
    fn layered_point_own_fill_wins_over_hoisted_bar_fill() {
        let (spec, batch) = bar_point_layered_spec(Some("#00aa00"), Some("#0000ff"), Some("#00aa00"));
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        let point_fills: Vec<Option<ferrum_scene::Color>> = panel.marks[1].nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { style, .. } = n { Some(style.fill) } else { None }
        }).collect();
        assert_eq!(point_fills.len(), 2);
        let blue = ferrum_scene::Color { r: 0x00, g: 0x00, b: 0xff, a: 255 };
        assert!(point_fills.iter().all(|c| *c == Some(blue)),
            "the point layer's own fill must win over the hoisted chart-level fallback; got {point_fills:?}");
    }

    /// Flat (no-`layers`) chart byte-identity: a single-mark
    /// `mark_point(fill=...)` chart never reaches `from_chart_and_layer`'s
    /// `or_else` fallback at all (`LayerPrepared::from_chart_only` uses
    /// `spec.mark_style` directly), so the scope extension must not change
    /// anything about a flat chart's own paint.
    #[test]
    fn flat_point_chart_fill_unaffected_by_hoisted_paint_scope_extension() {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark_style::MarkKwargsSpec;
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None,
            mark_style: Some(MarkKwargsSpec { fill: Some("#ff00ff".into()), ..Default::default() }),
            position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ]).unwrap();

        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        let point_fills: Vec<Option<ferrum_scene::Color>> = panel.marks[0].nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { style, .. } = n { Some(style.fill) } else { None }
        }).collect();
        assert_eq!(point_fills.len(), 2);
        let magenta = ferrum_scene::Color { r: 0xff, g: 0x00, b: 0xff, a: 255 };
        assert!(point_fills.iter().all(|c| *c == Some(magenta)),
            "a flat chart's own mark_style.fill must render unaffected by the layered-only scope extension; got {point_fills:?}");
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
        let (spec, batch) = two_layer_dual_y_spec_with_tooltips(Some("y0"), Some("y0"), Some("y1"));
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        assert_eq!(panel.marks.len(), 2, "one mark batch per layer");

        let layer0 = panel.marks[0]
            .tooltips
            .as_ref()
            .expect("layer 0 must have tooltips");
        assert_eq!(layer0[0].fields.len(), 1);
        assert_eq!(layer0[0].fields[0].name, "y0");
        assert_eq!(layer0[0].fields[0].value, "1");

        let layer1 = panel.marks[1]
            .tooltips
            .as_ref()
            .expect("layer 1 must have tooltips");
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
            let tooltips = mark_batch
                .tooltips
                .as_ref()
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
                x: Some(EncodingSpec {
                    field: "x".into(),
                    ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "y".into(),
                    ..Default::default()
                }),
                tooltip_fields: Some(vec![EncodingSpec {
                    field: "y".into(),
                    ..Default::default()
                }]),
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
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();

        let scene_a = build_scene_for(&spec, &batch);
        let scene_b = build_scene_for(&spec, &batch);
        assert_eq!(
            serde_json::to_string(&scene_a).unwrap(),
            serde_json::to_string(&scene_b).unwrap(),
            "identical inputs must produce byte-identical scene JSON"
        );

        let tooltips = scene_a.panels[0].marks[0]
            .tooltips
            .as_ref()
            .expect("must have tooltips");
        assert_eq!(tooltips[0].fields[0].name, "y");
        assert_eq!(tooltips[0].fields[0].value, "10");
    }

    /// End-to-end (spec §9.3 / GH #93): a real
    /// `encode(key=...)`-driven chart with ≥1000 rows carries its keys into
    /// the `HAS_KEYS` packed sidecar through the ACTUAL production pipeline —
    /// `build_scene` (which internally calls `extract_keys`) followed by
    /// `pack_instances::extract_packed_bytes` — not a hand-constructed
    /// `MarkBatch`. Every other `HAS_KEYS` test builds `MarkBatch.keys`
    /// directly; this is the one test that proves `Encoding.key` actually
    /// reaches the packed wire format via the real producer↔consumer seam.
    #[test]
    fn keyed_point_chart_above_pack_threshold_carries_has_keys_sidecar() {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let n = 1200usize; // above PACK_THRESHOLD (1000)
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i as f64) * 2.0).collect();
        let keys: Vec<String> = (0..n).map(|i| format!("row-{i}")).collect();

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                key: Some(EncodingSpec { field: "k".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None, // default shape = Circle, required for packing eligibility
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
            Field::new("k", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(StringArray::from(keys.clone())),
            ],
        )
        .unwrap();

        let mut scene = build_scene_for(&spec, &batch);
        // Sanity: build_scene really did populate per-node keys pre-pack —
        // proves `extract_keys` ran on this real pipeline, not that the
        // packer merely no-ops on an absent/empty vector.
        assert_eq!(
            scene.panels[0].marks[0].keys.as_deref().map(<[String]>::len),
            Some(n),
            "build_scene must populate MarkBatch.keys for a >=1000-row key=-encoded batch \
             before packing runs"
        );

        let packed = crate::render::pack_instances::extract_packed_bytes(&mut scene);
        assert!(!packed.is_empty(), "a >=1000-row circle batch must pack");

        let flags = u32::from_le_bytes(packed[16..20].try_into().unwrap());
        assert_ne!(
            flags & crate::render::pack_instances::HAS_KEYS,
            0,
            "HAS_KEYS (0x4) must be set for a real key=-encoded chart above the pack threshold"
        );
        let count = u32::from_le_bytes(packed[12..16].try_into().unwrap()) as usize;
        assert_eq!(count, n);

        // This real-pipeline chart also auto-populates data_indices (every
        // packing-eligible builder does, per `extract_keys`'s doc above) and
        // may carry auto-tooltips too — `decode_packed_keys_section` skips
        // whichever of those two sections `flags` indicates are present.
        let decoded = decode_packed_keys_section(&packed, count, flags);
        assert_eq!(
            decoded, keys,
            "packed keys must match the encode(key=...) column, in row order"
        );
    }

    /// Decode the `HAS_KEYS` trailing section of a packed circle-batch buffer
    /// (GH #93), skipping whichever of `HAS_DATA_INDICES`/`HAS_TOOLTIPS`
    /// `flags` indicates are present first — the fixed wire order is
    /// data_indices -> tooltips -> keys (`pack_instances::extract_packed_bytes`).
    /// Shared by the packed `HAS_KEYS` e2e tests below so the flags-aware
    /// walker has exactly one definition; asserts the keys section runs
    /// exactly to the end of `packed` (no trailing bytes) before returning.
    /// Wire constants are imported from `pack_instances`, not re-declared.
    fn decode_packed_keys_section(packed: &[u8], count: usize, flags: u32) -> Vec<String> {
        use crate::render::pack_instances::{CIRCLE_STRIDE, HAS_DATA_INDICES, HAS_TOOLTIPS};

        let instance_size = count * CIRCLE_STRIDE;
        let mut cursor = 20 + instance_size;
        if flags & HAS_DATA_INDICES != 0 {
            cursor += count * 4; // count x u32 data indices
        }
        if flags & HAS_TOOLTIPS != 0 {
            let num_fields =
                u32::from_le_bytes(packed[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            for _ in 0..num_fields {
                let name_len =
                    u32::from_le_bytes(packed[cursor..cursor + 4].try_into().unwrap()) as usize;
                cursor += 4 + name_len;
            }
            for _ in 0..(count * num_fields) {
                let val_len =
                    u32::from_le_bytes(packed[cursor..cursor + 4].try_into().unwrap()) as usize;
                cursor += 4 + val_len;
            }
        }
        let mut decoded = Vec::with_capacity(count);
        for _ in 0..count {
            let len = u32::from_le_bytes(packed[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            decoded.push(
                std::str::from_utf8(&packed[cursor..cursor + len])
                    .unwrap()
                    .to_string(),
            );
            cursor += len;
        }
        assert_eq!(cursor, packed.len(), "no trailing bytes after the keys section");
        decoded
    }

    /// Build a minimal x/y point-chart `ChartSpec` + `RecordBatch` with a
    /// `key`-encoded column of caller-supplied dtype. Shared by the
    /// non-string-key coercion tests below (GH #93) so each one only
    /// supplies its key column's data.
    fn build_x_y_key_chart_spec_and_batch(
        n: usize,
        key_field_name: &str,
        key_col: arrow::array::ArrayRef,
    ) -> (ChartSpec, RecordBatch) {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i as f64) * 2.0).collect();

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                key: Some(EncodingSpec { field: key_field_name.into(), ..Default::default() }),
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
            Field::new(key_field_name, key_col.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                key_col,
            ],
        )
        .unwrap();
        (spec, batch)
    }

    /// End-to-end packed, **ids above 1e6** (GH #93): a display formatter
    /// such as `format_numeric` would collapse ~1200 six-digit ids to 2
    /// unique keys (see `extract_keys`'s doc). `col_as_ordinal_category_str`'s
    /// native `i64.to_string()` has no such ceiling, so this asserts full
    /// injectivity (1200 rows -> 1200 unique keys), not just "some key
    /// exists".
    #[test]
    fn keyed_point_chart_with_integer_key_above_pack_threshold_carries_has_keys_sidecar() {
        use arrow::array::Int64Array;
        use std::collections::HashSet;
        use std::sync::Arc;

        let n = 1200usize;
        let ids: Vec<i64> = (0..n as i64).map(|i| 1_000_000 + i).collect();
        let (spec, batch) =
            build_x_y_key_chart_spec_and_batch(n, "id", Arc::new(Int64Array::from(ids.clone())));

        let mut scene = build_scene_for(&spec, &batch);
        let pre_pack_keys = scene.panels[0].marks[0]
            .keys
            .clone()
            .expect("an integer key= column must populate MarkBatch.keys before packing runs");
        assert_eq!(pre_pack_keys.len(), n);
        let pre_pack_unique: HashSet<&String> = pre_pack_keys.iter().collect();
        assert_eq!(
            pre_pack_unique.len(), n,
            "1200 distinct six-digit ids must produce 1200 distinct keys, not 2 \
             (the format_numeric aliasing this fixture targets — see extract_keys's doc)"
        );

        let packed = crate::render::pack_instances::extract_packed_bytes(&mut scene);
        let flags = u32::from_le_bytes(packed[16..20].try_into().unwrap());
        assert_ne!(
            flags & crate::render::pack_instances::HAS_KEYS,
            0,
            "HAS_KEYS (0x4) must be set for an integer key= column above the pack threshold"
        );
        let count = u32::from_le_bytes(packed[12..16].try_into().unwrap()) as usize;
        assert_eq!(count, n);

        let decoded = decode_packed_keys_section(&packed, count, flags);
        let expected: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        assert_eq!(
            decoded, expected,
            "packed keys must be the lossless decimal id strings, no scientific-notation aliasing"
        );
        let packed_unique: HashSet<&String> = decoded.iter().collect();
        assert_eq!(
            packed_unique.len(), n,
            "packed keys must stay injective: 1200 rows -> 1200 unique keys"
        );
    }

    /// JSON path, **i64 past 2^53** (GH #93): `2^53 + 1`
    /// (`9_007_199_254_740_993`) is the first `i64` that cannot round-trip
    /// through `f64` exactly, so a display formatter or a `col_as_f64`
    /// fallback would collapse 1200 such ids to a single key (see
    /// `extract_keys`'s doc). `col_as_ordinal_category_str` reads the native
    /// `i64`, never touching `f64`, so this asserts lossless, injective
    /// decimal strings at that exact magnitude.
    #[test]
    fn keyed_point_chart_with_integer_key_below_pack_threshold_populates_json_keys() {
        use arrow::array::Int64Array;
        use std::collections::HashSet;
        use std::sync::Arc;

        let n = 10usize;
        let base: i64 = 9_007_199_254_740_993; // 2^53 + 1
        let ids: Vec<i64> = (0..n as i64).map(|i| base + i).collect();
        let (spec, batch) =
            build_x_y_key_chart_spec_and_batch(n, "id", Arc::new(Int64Array::from(ids.clone())));

        let scene = build_scene_for(&spec, &batch);
        let keys = scene.panels[0].marks[0].keys.as_ref().expect(
            "an integer key= column past 2^53 on a batch below the pack threshold must still \
             populate MarkBatch.keys on the JSON path",
        );
        let expected: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        assert_eq!(
            keys, &expected,
            "JSON-path keys must be the exact i64 decimal strings, lossless past f64's \
             53-bit mantissa"
        );
        let unique: HashSet<&String> = keys.iter().collect();
        assert_eq!(
            unique.len(), n,
            "distinct i64 ids past 2^53 must produce distinct keys, not collapse to one"
        );
    }

    /// **Realistic 2026 timestamps** (GH #93): a display formatter such as
    /// `format_numeric` would collapse an entire batch of present-day
    /// millisecond epochs to ONE key — every timestamp after
    /// 1970-01-01T00:16:40Z falls in its scientific-notation (4-sig-fig)
    /// regime (see `extract_keys`'s doc). `col_as_temporal_epoch_str` reads
    /// the raw `i64` epoch, so 50 one-second-apart 2026 timestamps must
    /// produce 50 unique keys, not 1 — the realistic-magnitude case a
    /// same-instant-only equality check would miss.
    #[test]
    fn extract_keys_coerces_realistic_2026_timestamp_key_column_injectively() {
        use arrow::array::TimestampMillisecondArray;
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use std::collections::HashSet;
        use std::sync::Arc;

        let n = 50usize;
        let base_ms: i64 = 1_767_225_600_000; // 2026-01-01T00:00:00Z
        let epochs: Vec<i64> = (0..n as i64).map(|i| base_ms + i * 1000).collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "t",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(TimestampMillisecondArray::from(epochs.clone()))],
        )
        .unwrap();
        let encoding = Encoding {
            key: Some(EncodingSpec { field: "t".into(), ..Default::default() }),
            ..Default::default()
        };
        let data_indices: Vec<usize> = (0..n).collect();

        let keys = extract_keys(&encoding, &batch, Some(&data_indices))
            .expect("a Timestamp key column must coerce, not silently drop");
        let expected: Vec<String> = epochs.iter().map(|e| e.to_string()).collect();
        assert_eq!(
            keys, expected,
            "temporal keys must be the exact epoch i64 strings, not a 4-sig-fig display value"
        );
        let unique: HashSet<&String> = keys.iter().collect();
        assert_eq!(
            unique.len(), n,
            "50 distinct present-day timestamps must produce 50 distinct keys, not 1"
        );
    }

    /// **Boolean key column** (GH #93): a residual silent-drop gap distinct
    /// from the aliasing defect above — neither `col_as_str` nor `col_as_f64`
    /// covers `Boolean`, so `key="b:N"` on a `Boolean` column produced no
    /// keys at all under either of those. `col_as_ordinal_category_str`
    /// covers it, closing that gap as a side effect.
    #[test]
    fn extract_keys_coerces_boolean_key_column_instead_of_silently_dropping() {
        use arrow::array::BooleanArray;
        use arrow::datatypes::{DataType, Field, Schema};
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new("flag", DataType::Boolean, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BooleanArray::from(vec![true, false, true]))],
        )
        .unwrap();
        let encoding = Encoding {
            key: Some(EncodingSpec { field: "flag".into(), ..Default::default() }),
            ..Default::default()
        };
        let data_indices = [0usize, 1, 2];

        let keys = extract_keys(&encoding, &batch, Some(&data_indices));
        assert_eq!(
            keys,
            Some(vec!["true".to_string(), "false".to_string(), "true".to_string()]),
            "a Boolean key= column must coerce to true/false strings, not silently drop"
        );
    }

    fn resolve_dual_y(spec: &ChartSpec, batch: &RecordBatch) -> scale_resolve::ResolvedScales {
        let theme = ThemeInputs::default();
        let prep = super::super::prepare::prepare_render_inputs(spec, batch, &theme, None).unwrap();
        let chart_config = super::super::chart_config::ChartConfig::default();
        let panel = crate::layout::PanelLayout {
            plot_area: crate::layout::Rect {
                x: 0.0,
                y: 0.0,
                w: 300.0,
                h: 200.0,
            },
            ..Default::default()
        };
        // Both layers read the whole panel batch (no per-layer data_source).
        let layer_batches = vec![batch.clone(), batch.clone()];
        let mut warnings = Vec::new();
        let ctx = PanelResolveCtx {
            spec,
            prep: &prep,
            theme: &theme,
            chart_config: &chart_config,
            leaf_scales: None,
        };
        let (_spec, scales) = resolve_panel_scales(
            &ctx,
            &panel,
            batch,
            &layer_batches,
            &mut warnings,
            (0.0, 0.0),
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

        assert!(
            scales.y_slots.has_independent(),
            "independent layer must create a second slot"
        );
        assert_eq!(
            scales.y_slots.slots().len(),
            2,
            "one primary slot + one independent slot"
        );
        assert_eq!(
            scales.y_slots.slot_for_layer(0),
            0,
            "layer 0 is always the primary slot"
        );
        assert_eq!(
            scales.y_slots.slot_for_layer(1),
            1,
            "the independent layer binds slot 1"
        );

        let (lo0, hi0) = scales
            .y_for_layer(0)
            .data_domain()
            .expect("primary y is continuous");
        let (lo1, hi1) = scales
            .y_for_layer(1)
            .data_domain()
            .expect("slot-1 y is continuous");

        // Slot 0 is the small y0 range; slot 1 is the large y1 range. Padding/nice
        // widen the exact bounds, so compare against a separating midpoint.
        assert!(
            hi0 < 50.0,
            "slot 0 must be layer 0's small y0 domain, got {lo0}..{hi0}"
        );
        assert!(
            lo1 > 50.0,
            "slot 1 must be layer 1's large y1 domain, got {lo1}..{hi1}"
        );

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

        assert!(
            !scales.y_slots.has_independent(),
            "no independent layer → no extra slot"
        );
        assert!(
            scales.y_slots.slots().is_empty(),
            "shared path leaves the slot list empty"
        );
        assert_eq!(
            scales.y_slots.slot_for_layer(1),
            0,
            "shared layers bind slot 0"
        );

        // Every layer draws through the one primary y-scale.
        assert_eq!(scales.y_for_layer(0).data_domain(), scales.y.data_domain());
        assert_eq!(scales.y_for_layer(1).data_domain(), scales.y.data_domain());
    }

    /// `build_panel_mark_batches` clones the panel-level `ResolvedScales` for an
    /// independent-y layer and reassigns `.y` to that layer's own slot scale
    /// (GH #61 T2). The clone's `y_slots` must describe just that one scale —
    /// self-describing — rather than keep the stale multi-slot list carried
    /// over from `scales.clone()`, which would let `.y` and `.y_slots`
    /// disagree on a clone whose `.y` only ever holds one layer's scale. This
    /// mirrors the exact clone-construction sequence in `build_panel_mark_batches`.
    #[test]
    fn independent_layer_clone_gets_self_describing_single_slot_y_slots() {
        let (spec, batch) = two_layer_dual_y_spec(true);
        let scales = resolve_dual_y(&spec, &batch);

        let mut layer_scales = scales.clone();
        layer_scales.y = scales.y_for_layer(1).clone();
        layer_scales.y_slots = scale_resolve::YScaleSlots::single(layer_scales.y.clone());

        assert!(
            !layer_scales.y_slots.has_independent(),
            "a single-slot clone must not report independent slots"
        );
        assert_eq!(
            layer_scales.y_slots.slots().len(),
            1,
            "the clone's y_slots must describe exactly one slot"
        );
        assert_eq!(
            layer_scales.y_slots.slot_for_layer(1),
            0,
            "slot_for_layer must fall back to 0 for any layer index on a single-slot clone"
        );
        assert_eq!(
            layer_scales.y.data_domain(),
            layer_scales.y_slots.slots()[0].data_domain(),
            "the clone's one slot must be the same scale as its .y"
        );

        // Sanity: the clone really carries layer 1's own (large) domain, not
        // the primary's — this isn't accidentally testing a no-op clone.
        let (lo, hi) = layer_scales
            .y
            .data_domain()
            .expect("layer 1 y is continuous");
        assert!(
            lo > 50.0,
            "clone must carry layer 1's large y1 domain, got {lo}..{hi}"
        );
    }

    /// Secondary-y (#52): `build_tick_levels` emits one `y_slot_levels` entry per
    /// right axis, generated from that slot's own scale, so the WASM overlay can
    /// recognize and reposition right-axis tick labels under zoom.
    #[test]
    fn build_tick_levels_emits_secondary_slot_levels() {
        let (spec, batch) = two_layer_dual_y_spec(true);
        let scales = resolve_dual_y(&spec, &batch);
        let ptl = build_tick_levels(&scales, 0);

        assert_eq!(
            ptl.y_slot_levels.len(),
            1,
            "one independent slot → one right-axis tick list"
        );
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
        assert!(
            ptl.y_slot_levels.is_empty(),
            "shared-y chart emits no secondary slot levels"
        );

        let json = serde_json::to_string(&ptl).unwrap();
        assert!(
            !json.contains("y_slot_levels"),
            "empty slot levels must be omitted from JSON"
        );
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
            plot_area: crate::layout::Rect {
                x: 0.0,
                y: 0.0,
                w: 300.0,
                h: 200.0,
            },
            ..Default::default()
        };
        let theme = ThemeInputs::default();
        let chart_config = super::super::chart_config::ChartConfig::default();
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let (primary_y, _warn) = crate::layout::axis::layout_y_axis(
            &AxisInput::new(
                AxisOrient::Left,
                Some("Primary".into()),
                vec!["0".into(), "5".into(), "10".into()],
                None,
            ),
            panel.plot_area,
            0,
            11.0,
            13.0,
            8.0,
            4.0,
            &m,
        );

        let mut secondary_input = AxisInput::new(
            AxisOrient::Right,
            Some("Secondary".into()),
            // Deliberately a DIFFERENT tick count than the primary's 3 — if the
            // grid leaked this axis's ticks, the counts below would diverge.
            vec![
                "0".into(),
                "25".into(),
                "50".into(),
                "75".into(),
                "100".into(),
            ],
            None,
        );
        secondary_input.show_grid = true; // deliberately try to leak into the grid
        let (secondary_y, _warn2) = crate::layout::axis::layout_y_axis(
            &secondary_input,
            panel.plot_area,
            0,
            11.0,
            13.0,
            8.0,
            4.0,
            &m,
        );

        // Baseline: grid + axis nodes built from the primary alone.
        let baseline = route_panel_axes_and_grid(
            &spec,
            &scales,
            &panel,
            &[],
            None,
            Some(&primary_y),
            &[],
            &[],
            false,
            false,
            &theme,
            &chart_config,
        );
        // With the secondary axis routed in alongside the primary (slot 1).
        let with_secondary = route_panel_axes_and_grid(
            &spec,
            &scales,
            &panel,
            &[],
            None,
            Some(&primary_y),
            &[&secondary_y],
            &[1],
            false,
            false,
            &theme,
            &chart_config,
        );

        assert_eq!(
            with_secondary.grid.len(),
            baseline.grid.len(),
            "a secondary y-axis must not add or alter gridlines, even with show_grid=true"
        );
        // But it DOES contribute its own axis nodes (ticks + domain + labels +
        // title) — one axis per slot.
        let baseline_axis_nodes = baseline.axes_below.len() + baseline.axes_above.len();
        let with_secondary_axis_nodes =
            with_secondary.axes_below.len() + with_secondary.axes_above.len();
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
        let viewport = crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        };
        let config = super::super::config::RenderConfig::default();
        let chart_config = super::super::chart_config::ChartConfig::default();

        let (shared_spec, shared_batch) = two_layer_dual_y_spec(false);
        let shared = super::super::render_svg(
            &shared_spec,
            &shared_batch,
            &theme,
            viewport,
            &config,
            &chart_config,
        )
        .unwrap();
        assert!(
            shared.layout.secondary_y_axes.is_empty(),
            "the shared-y chart must not reserve any secondary axis"
        );

        let (dual_spec, dual_batch) = two_layer_dual_y_spec(true);
        let dual = super::super::render_svg(
            &dual_spec,
            &dual_batch,
            &theme,
            viewport,
            &config,
            &chart_config,
        )
        .unwrap();

        assert_eq!(
            dual.layout.secondary_y_axes.len(),
            1,
            "one secondary axis for the one independent_y layer"
        );
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
        assert!(
            dual.bytes.contains(">y1<"),
            "secondary axis title must appear in the SVG"
        );
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
        let viewport = crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        };
        let config = super::super::config::RenderConfig::default();
        let chart_config = super::super::chart_config::ChartConfig::default();

        let prep =
            super::super::prepare::prepare_render_inputs(&spec, &batch, &theme, None).unwrap();
        let mut warnings = prep.warnings.clone();
        let metrics = super::super::font::FontdueMetrics::new();
        let layout = crate::layout::compute_layout(
            &spec,
            &theme,
            viewport,
            &prep.axes,
            &prep.facet_groups,
            &prep.legend_entries,
            prep.legend_title.clone(),
            prep.colorbar.as_ref(),
            &metrics,
            &crate::layout::legend::LegendOverrides::default(),
            &prep.aux_legends,
            crate::layout::CompositeLayoutSeam::default(),
        )
        .unwrap();

        build_scene(
            &spec,
            &prep,
            &layout,
            &theme,
            &config,
            &mut warnings,
            &chart_config,
            None,
        )
        .unwrap()
    }

    /// Find every `("y_slot", value)` attr pair carried by `SceneNode::Group`
    /// wrappers anywhere in `nodes` (axis nodes are wrapped one group per
    /// y-axis — see `route_y_axis_slotted`).
    fn collect_y_slot_group_tags(nodes: &[SceneNode]) -> Vec<String> {
        nodes
            .iter()
            .filter_map(|n| {
                if let SceneNode::Group { attrs, .. } = n {
                    attrs
                        .iter()
                        .find(|(k, _)| k == "y_slot")
                        .map(|(_, v)| v.clone())
                } else {
                    None
                }
            })
            .collect()
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
            ferrum_scene::CoordKind::Cartesian {
                y_domain,
                y_domains,
                ..
            } => {
                assert_eq!(
                    y_domains.len(),
                    2,
                    "one y-domain per slot (primary + one independent)"
                );
                let (slot0_lo, slot0_hi) = y_domains[0].expect("slot 0 domain must be Some");
                let (slot1_lo, slot1_hi) = y_domains[1].expect("slot 1 domain must be Some");
                assert!(
                    slot0_hi < 50.0,
                    "slot 0 must be the small y0 domain, got {slot0_lo}..{slot0_hi}"
                );
                assert!(
                    slot1_lo > 50.0,
                    "slot 1 must be the large y1 domain, got {slot1_lo}..{slot1_hi}"
                );
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
                assert!(
                    y_domains.is_empty(),
                    "shared path must leave the per-slot y-domain list empty"
                );
            }
            other => panic!("expected Cartesian coord, got {other:?}"),
        }
        for batch in &panel.marks {
            assert_eq!(
                batch.y_slot, 0,
                "every mark batch binds slot 0 on the shared path"
            );
        }
        assert!(
            collect_y_slot_group_tags(&panel.axes).is_empty(),
            "no axis Group should carry a y_slot tag on the shared path"
        );

        let json = serde_json::to_string(&scene).expect("serialize shared-path scene");
        assert!(
            !json.contains("y_domains"),
            "shared-path scene JSON must omit y_domains: {json}"
        );
        assert!(
            !json.contains("y_slot"),
            "shared-path scene JSON must omit y_slot: {json}"
        );
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
        assert_eq!(
            panel.marks[1].y_slot, 1,
            "layer 1 (independent) binds slot 1"
        );
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

    /// GH #60/#73: within each slot-tagged y-axis `Group`, every tick-label
    /// `SceneNode::Text` carries `slot: Some(k)` matching the Group's own
    /// `y_slot` attr (`k`) — including slot 0 (primary), making the contract
    /// uniform across every axis on a dual-axis panel. The axis title (at
    /// most one `Text` per group, distinguishable as the only untagged one)
    /// is never slot-tagged — it is not axis-tick text.
    #[test]
    fn y_axis_tick_text_nodes_tagged_with_slot_index() {
        let scene = build_dual_y_scene(true);
        let panel = &scene.panels[0];
        let mut checked_groups = 0;
        for node in &panel.axes {
            if let SceneNode::Group { attrs, children } = node {
                let Some((_, slot_str)) = attrs.iter().find(|(k, _)| k == "y_slot") else {
                    continue;
                };
                let expected_slot: usize = slot_str.parse().unwrap();
                let text_slots: Vec<Option<usize>> = children
                    .iter()
                    .filter_map(|c| {
                        if let SceneNode::Text { slot, .. } = c {
                            Some(*slot)
                        } else {
                            None
                        }
                    })
                    .collect();
                assert!(
                    !text_slots.is_empty(),
                    "y_slot={expected_slot} axis group must contain tick-label text"
                );
                let untagged = text_slots.iter().filter(|s| s.is_none()).count();
                assert!(
                    untagged <= 1,
                    "at most the axis title may be untagged, got {untagged} untagged texts"
                );
                for slot in text_slots.iter().filter(|s| s.is_some()) {
                    assert_eq!(
                        *slot, Some(expected_slot),
                        "tick-label text in the y_slot={expected_slot} group must carry the same slot"
                    );
                }
                checked_groups += 1;
            }
        }
        assert_eq!(
            checked_groups, 2,
            "expected one slot-tagged group each for primary and secondary y-axes"
        );
    }

    /// #72 de-risking (design-review S3-2): with TWO independent-y layers, the
    /// ONE layer→slot plan keeps every consumer in lock-step. For each secondary
    /// slot `k ∈ {1, 2}`: the prepared axis input `prep.axes.secondary_y[k-1]`,
    /// the resolved slot scale `y_slots.slots()[k]`, the scene `y_domains[k]`,
    /// the `y_slot="k"` axis-group tag, and the mark batch for that layer
    /// (`marks[k].y_slot`) all reference the SAME slot — proving the prepare
    /// axis-band order, the per-panel slots, and the axis-router tags cannot
    /// drift when there is more than one secondary axis.
    #[test]
    fn two_independent_layers_keep_all_slot_consumers_in_lockstep() {
        let (spec, batch) = three_layer_two_independent_spec();
        let theme = ThemeInputs::default();
        let viewport = crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        };
        let config = super::super::config::RenderConfig::default();
        let chart_config = super::super::chart_config::ChartConfig::default();

        let prep =
            super::super::prepare::prepare_render_inputs(&spec, &batch, &theme, None).unwrap();

        // The plan itself: two secondary layers (indices 1, 2) on slots 1, 2.
        assert!(prep.y_slot_plan.has_independent());
        assert_eq!(prep.y_slot_plan.secondary_layers(), &[1, 2]);
        assert_eq!(prep.y_slot_plan.slot_for_layer(0), 0);
        assert_eq!(prep.y_slot_plan.slot_for_layer(1), 1);
        assert_eq!(prep.y_slot_plan.slot_for_layer(2), 2);

        // Prepare built one axis input per secondary layer, in slot order.
        assert_eq!(
            prep.axes.secondary_y.len(),
            2,
            "one axis input per secondary slot"
        );

        let mut warnings = prep.warnings.clone();
        let metrics = super::super::font::FontdueMetrics::new();
        let layout = crate::layout::compute_layout(
            &spec,
            &theme,
            viewport,
            &prep.axes,
            &prep.facet_groups,
            &prep.legend_entries,
            prep.legend_title.clone(),
            prep.colorbar.as_ref(),
            &metrics,
            &crate::layout::legend::LegendOverrides::default(),
            &prep.aux_legends,
            crate::layout::CompositeLayoutSeam::default(),
        )
        .unwrap();

        assert_eq!(
            layout.secondary_y_axes.len(),
            2,
            "layout reserved one right band per secondary slot"
        );

        let scene = build_scene(
            &spec,
            &prep,
            &layout,
            &theme,
            &config,
            &mut warnings,
            &chart_config,
            None,
        )
        .unwrap();
        let panel = &scene.panels[0];

        // Every layer's mark batch binds its plan slot.
        assert_eq!(panel.marks.len(), 3, "one mark batch per layer");
        assert_eq!(panel.marks[0].y_slot, 0, "layer 0 (primary) binds slot 0");
        assert_eq!(panel.marks[1].y_slot, 1, "layer 1 binds slot 1");
        assert_eq!(panel.marks[2].y_slot, 2, "layer 2 binds slot 2");

        // One axis group per slot, tagged by slot id (primary + two secondary).
        let mut tags = collect_y_slot_group_tags(&panel.axes);
        tags.sort();
        assert_eq!(
            tags,
            vec!["0".to_string(), "1".to_string(), "2".to_string()],
            "expected exactly one axis group per slot 0/1/2"
        );

        // The per-slot scene y-domains: slot k carries slot-k's own magnitude
        // (y0 ∈ [1,3] < y1 ∈ [100,300] < y2 ∈ [1000,3000]). A slot cross-wire
        // between any consumer would land a domain on the wrong slot.
        match &panel.coord {
            ferrum_scene::CoordKind::Cartesian { y_domains, .. } => {
                assert_eq!(y_domains.len(), 3, "one y-domain per slot");
                let (_, hi0) = y_domains[0].expect("slot 0 domain");
                let (lo1, hi1) = y_domains[1].expect("slot 1 domain");
                let (lo2, _) = y_domains[2].expect("slot 2 domain");
                assert!(hi0 < 50.0, "slot 0 is the small y0 domain, got ..{hi0}");
                assert!(
                    lo1 > 50.0 && hi1 < 500.0,
                    "slot 1 is the y1 domain, got {lo1}..{hi1}"
                );
                assert!(lo2 > 500.0, "slot 2 is the large y2 domain, got {lo2}..");
            }
            other => panic!("expected Cartesian coord, got {other:?}"),
        }
    }

    /// #72 discriminating (spec §9.3): for a param-bound secondary (independent-y)
    /// layer, the static right-axis tick labels, the scene `y_domains[1]`, and the
    /// marks all reflect the param's *substituted* domain — proving prepare (axis
    /// ticks) and scene_build (marks / `y_domains`) share ONE param-aware per-layer
    /// y resolution (`scale_resolve::resolve_layer_y_slot_scale`).
    ///
    /// The independent layer's `y` scale carries a `domainParam` "d1" whose Variable
    /// value overrides the domain to `[0, 1000]`, DISJOINT from the data-inferred
    /// `y1` range `[100, 300]`. Before the unification, prepare's axis-input path
    /// did not substitute params, so the right-axis ticks spanned `[100, 300]` while
    /// scene_build's marks/`y_domains` used the substituted `[0, 1000]` — the two
    /// diverged. The spec is built directly with `params` populated, so the test is
    /// independent of the Python param-hoisting fix (Task 4).
    #[test]
    fn param_bound_secondary_axis_ticks_equal_substituted_domain() {
        let (mut spec, batch) = two_layer_dual_y_spec(true);
        // Bind the independent layer's y scale to a domain param whose declared
        // value is disjoint from the data range.
        spec.layers.as_mut().unwrap()[1]
            .encoding
            .y
            .as_mut()
            .unwrap()
            .scale = Some(linear_domain_param("d1"));
        spec.params = vec![ParameterSpec {
            name: "d1".into(),
            kind: ParamKind::Variable,
            value: Some(serde_json::json!([0.0, 1000.0])),
            bind: None,
            select: None,
        }];

        let theme = ThemeInputs::default();
        let viewport = crate::layout::Viewport {
            width: 600.0,
            height: 400.0,
        };
        let config = super::super::config::RenderConfig::default();
        let chart_config = super::super::chart_config::ChartConfig::default();

        let prep =
            super::super::prepare::prepare_render_inputs(&spec, &batch, &theme, None).unwrap();
        let mut warnings = prep.warnings.clone();
        let metrics = super::super::font::FontdueMetrics::new();
        let layout = crate::layout::compute_layout(
            &spec,
            &theme,
            viewport,
            &prep.axes,
            &prep.facet_groups,
            &prep.legend_entries,
            prep.legend_title.clone(),
            prep.colorbar.as_ref(),
            &metrics,
            &crate::layout::legend::LegendOverrides::default(),
            &prep.aux_legends,
            crate::layout::CompositeLayoutSeam::default(),
        )
        .unwrap();
        let scene = build_scene(
            &spec,
            &prep,
            &layout,
            &theme,
            &config,
            &mut warnings,
            &chart_config,
            None,
        )
        .unwrap();

        // Scene side (marks + y_domains): slot 1 carries the substituted domain.
        // Marks on layer 1 are positioned through this slot's scale, so the coord
        // `y_domains[1]` IS the domain the marks project through.
        let slot1 = match &scene.panels[0].coord {
            ferrum_scene::CoordKind::Cartesian { y_domains, .. } => {
                y_domains[1].expect("slot 1 domain must be Some")
            }
            other => panic!("expected Cartesian coord, got {other:?}"),
        };
        assert_eq!(
            slot1,
            (0.0, 1000.0),
            "scene y_domains[1] must be the substituted param domain, not the data [100,300]"
        );

        // Prepare side (axis ticks): the right axis's tick labels must reflect the
        // SAME substituted domain. Parse numeric tick labels (stripping thousands
        // separators).
        assert_eq!(
            layout.secondary_y_axes.len(),
            1,
            "one right axis for the independent layer"
        );
        let tick_vals: Vec<f64> = layout.secondary_y_axes[0]
            .ticks
            .iter()
            .filter_map(|t| t.label.replace(',', "").parse::<f64>().ok())
            .collect();
        assert!(
            !tick_vals.is_empty(),
            "secondary axis must have numeric tick labels"
        );
        let max_tick = tick_vals.iter().cloned().fold(f64::MIN, f64::max);
        let min_tick = tick_vals.iter().cloned().fold(f64::MAX, f64::min);
        // Discriminates: the data domain [100,300] cannot produce a tick at 1000 or
        // a tick at 0; the substituted [0,1000] domain produces both.
        assert_eq!(
            (min_tick, max_tick),
            slot1,
            "right-axis tick extent must equal scene y_domains[1] (ticks == marks == y_domains[1]); got {tick_vals:?}"
        );
    }

    // ── R3: user-facing channel names under CoordFlip ───────────────────────

    /// Build a single-layer `ChartSpec` with `x`/`y`/`y2` bound and, optionally,
    /// `coord: CoordFlip` — shared setup for the R3 `validate_mark_encoding`
    /// regression tests below.
    fn area_spec_with_y2(coord_flipped: bool) -> ChartSpec {
        use crate::spec::coord::CoordKind;
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Area,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "price".into(),
                    ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "weight".into(),
                    ..Default::default()
                }),
                y2: Some(EncodingSpec {
                    field: "weight2".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: if coord_flipped {
                Some(CoordKind::Flip)
            } else {
                None
            },
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        }
    }

    fn price_weight_weight2_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("price", DataType::Float64, false),
            Field::new("weight", DataType::Float64, false),
            Field::new("weight2", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
                Arc::new(Float64Array::from(vec![15.0, 25.0, 35.0])),
            ],
        )
        .unwrap()
    }

    /// R3 regression (the originally verified live bug): under `CoordFlip`, a
    /// user who writes `mark_area().encode(y2=...)` (a vertical band — the
    /// SUPPORTED spelling) gets an error naming `'y2'` — the channel they
    /// actually wrote — with a hint recommending `x2=`, the spelling that
    /// achieves the same supported vertical-band area post-flip. Before this
    /// fix the message named the resolved `'x2'` and advised `y2=`, neither of
    /// which the user wrote or should write.
    #[test]
    fn unsupported_channel_combination_names_users_channel_under_flip() {
        let spec = area_spec_with_y2(true);
        let batch = price_weight_weight2_batch();
        let prep = crate::render::prepare::prepare_render_inputs(
            &spec,
            &batch,
            &ThemeInputs::default(),
            None,
        )
        .unwrap();
        assert!(prep.coord_flipped);
        // Post-flip: the user's y2 (vertical-band area, supported) landed on x2
        // (unsupported for mark_area) — this is the resolved token validation acts on.
        assert!(prep.layers[0].encoding.x2.is_some());
        let err = validate_mark_encoding(&Mark::Area, &prep.layers[0].encoding, prep.coord_flipped)
            .expect_err("post-flip x2 must still be rejected for mark_area");
        let text = format!("{err}");
        assert_eq!(
            text,
            "mark_area: channel 'y2' is not supported; use x2= for a vertical band area, or use mark_rect for a 2-D extent"
        );
    }

    /// R3 byte-identity: the same unsupported combination, unflipped, renders
    /// exactly the pre-fix message text.
    #[test]
    fn unsupported_channel_combination_message_unchanged_when_not_flipped() {
        use crate::spec::encoding::EncodingSpec;
        let mut spec = area_spec_with_y2(false);
        // Directly bind x2 (the actually-unsupported channel) rather than y2.
        spec.encoding.y2 = None;
        spec.encoding.x2 = Some(EncodingSpec {
            field: "weight2".into(),
            ..Default::default()
        });
        let batch = price_weight_weight2_batch();
        let prep = crate::render::prepare::prepare_render_inputs(
            &spec,
            &batch,
            &ThemeInputs::default(),
            None,
        )
        .unwrap();
        assert!(!prep.coord_flipped);
        let err = validate_mark_encoding(&Mark::Area, &prep.layers[0].encoding, prep.coord_flipped)
            .expect_err("x2 must be rejected for mark_area");
        assert_eq!(
            format!("{err}"),
            "mark_area: channel 'x2' is not supported; use y2= for a vertical band area, or use mark_rect for a 2-D extent"
        );
    }

    /// A `mark_bar` spec with `x`, `y`, `x2`, AND `y2` all bound — the
    /// unsupported 2-D-extent combination `validate_mark_encoding` rejects for
    /// `Mark::Bar` (`x2.is_some() && y2.is_some()`).
    fn bar_spec_with_x2_y2(coord_flipped: bool) -> ChartSpec {
        use crate::spec::coord::CoordKind;
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "price".into(),
                    ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "weight".into(),
                    ..Default::default()
                }),
                x2: Some(EncodingSpec {
                    field: "price2".into(),
                    ..Default::default()
                }),
                y2: Some(EncodingSpec {
                    field: "weight2".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: if coord_flipped {
                Some(CoordKind::Flip)
            } else {
                None
            },
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        }
    }

    fn price_weight_price2_weight2_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("price", DataType::Float64, false),
            Field::new("weight", DataType::Float64, false),
            Field::new("price2", DataType::Float64, false),
            Field::new("weight2", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
                Arc::new(Float64Array::from(vec![5.0, 15.0, 25.0])),
                Arc::new(Float64Array::from(vec![15.0, 25.0, 35.0])),
            ],
        )
        .unwrap()
    }

    /// R3: `mark_bar`'s "both x2 and y2" hint is flip-symmetric — whichever
    /// letters the user wrote, both remain bound after the whole-encoding swap
    /// — so the message is identical flipped or not. Driven through the real
    /// `validate_mark_encoding` on a real flipped `Mark::Bar` spec (not a
    /// hand-built `RenderError` re-typing the production hint literal), so a
    /// future edit to the `Mark::Bar` arm in `validate_mark_encoding` — e.g.
    /// giving it a `hint_alt_channel` — would actually be exercised here.
    #[test]
    fn unsupported_channel_combination_bar_hint_is_flip_symmetric() {
        let batch = price_weight_price2_weight2_batch();
        let theme = ThemeInputs::default();

        let flipped_spec = bar_spec_with_x2_y2(true);
        let flipped_prep =
            crate::render::prepare::prepare_render_inputs(&flipped_spec, &batch, &theme, None)
                .unwrap();
        assert!(flipped_prep.coord_flipped);
        let flipped_err = validate_mark_encoding(
            &Mark::Bar,
            &flipped_prep.layers[0].encoding,
            flipped_prep.coord_flipped,
        )
        .expect_err("both x2 and y2 bound must be rejected for mark_bar, flipped");

        let unflipped_spec = bar_spec_with_x2_y2(false);
        let unflipped_prep =
            crate::render::prepare::prepare_render_inputs(&unflipped_spec, &batch, &theme, None)
                .unwrap();
        assert!(!unflipped_prep.coord_flipped);
        let unflipped_err = validate_mark_encoding(
            &Mark::Bar,
            &unflipped_prep.layers[0].encoding,
            unflipped_prep.coord_flipped,
        )
        .expect_err("both x2 and y2 bound must be rejected for mark_bar, unflipped");

        let flipped_text = format!("{flipped_err}");
        let unflipped_text = format!("{unflipped_err}");
        assert_eq!(flipped_text, unflipped_text);
        assert_eq!(
            flipped_text,
            "mark_bar: channel 'x2 and y2' is not supported; a 2-D extent (both x2= and y2=) is a rectangle; use mark_rect instead"
        );
    }

    // ── Batch-A Task 13: `Mark::Rule`'s presence-only channel validation ─────

    fn rule_encoding(
        x: Option<&str>,
        y: Option<&str>,
        x2: Option<&str>,
        y2: Option<&str>,
    ) -> crate::spec::encoding::Encoding {
        use crate::spec::encoding::EncodingSpec;
        crate::spec::encoding::Encoding {
            x: x.map(|f| EncodingSpec { field: f.into(), ..Default::default() }),
            y: y.map(|f| EncodingSpec { field: f.into(), ..Default::default() }),
            x2: x2.map(|f| EncodingSpec { field: f.into(), ..Default::default() }),
            y2: y2.map(|f| EncodingSpec { field: f.into(), ..Default::default() }),
            ..Default::default()
        }
    }

    /// The presence-invalid shapes rule.rs's `build` used to fall through to a
    /// silent `empty()` for (audit F-L06 class): a ranged-`y` shape (`y` +
    /// `y2`) with no `x` to anchor each row's segment, and no positional
    /// channel bound at all. Both are refused up front by
    /// `validate_mark_encoding` — which is now `RuleShape::resolve` itself
    /// (batch-A Task 13 spec c3) — with the message naming every supported
    /// shape.
    #[test]
    fn unsupported_channel_combination_rule_names_supported_shapes() {
        let y2_without_x = rule_encoding(None, Some("lo"), None, Some("hi"));
        let err = validate_mark_encoding(&Mark::Rule, &y2_without_x, false)
            .expect_err("y2 without x must be rejected for mark_rule");
        assert_eq!(
            format!("{err}"),
            "mark_rule: channel 'positional' is not supported; mark_rule supports: y= alone \
             (horizontal span), x= alone (vertical span), x=+y=+y2= (ranged vertical segment), \
             y=+x=+x2= (ranged horizontal segment), or x=+y=+x2=+y2= (diagonal segment)"
        );

        let nothing_bound = rule_encoding(None, None, None, None);
        let err2 = validate_mark_encoding(&Mark::Rule, &nothing_bound, false)
            .expect_err("no positional channel at all must be rejected for mark_rule");
        assert_eq!(format!("{err2}"), format!("{err}"), "both invalid shapes share one message");
    }

    // ── Batch-A Task 13 spec c3: a rule layer's span axis is its OWN ────────
    //
    // These exercise the FULL `prepare → layout → build_scene` pipeline
    // because the defect lived at the layer-inheritance seam
    // (`LayerPrepared::from_chart_and_layer` → the DrawCtx-local encoding copy
    // in `build_panel_mark_batches`), which a unit test that builds `DrawCtx`
    // by hand cannot see.

    /// `shap_chart(kind="beeswarm")`'s exact lowered shape, mirrored: a chart
    /// whose CHART-LEVEL encoding binds a numeric `x` and an ORDINAL `y`, a
    /// point layer declaring both, and a `rule` layer declaring ONLY
    /// `x="_ref_zero"` — the zero-line sentinel column
    /// (`ferrum._constant_columns._inject_constant`) with one non-null row and
    /// the rest null, so the layer draws exactly one reference line. Verified
    /// against the live `chart.to_dict()` of `ferrum.shap_chart(..., kind=
    /// "beeswarm")` before being written down here.
    fn beeswarm_shaped_zero_line_spec() -> (ChartSpec, RecordBatch) {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec};
        use crate::spec::layer::Layer;
        use crate::spec::mark_style::MarkKwargsSpec;
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let point_layer = Layer {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "shap_value".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "feature".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), mark_style: None,
            data_source: None, position: None, blend: None, name: Some("point".into()),
            independent_y: false,
        };
        let reference_layer = Layer {
            mark: Mark::Rule,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "_ref_zero".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            mark_style: Some(MarkKwargsSpec {
                stroke: Some("#AAAAAA".into()),
                stroke_dash: Some(vec![4.0, 4.0]),
                ..Default::default()
            }),
            data_source: None, position: None, blend: None, name: Some("reference".into()),
            independent_y: false,
        };
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "shap_value".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "feature".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None,
            layers: Some(vec![point_layer, reference_layer]),
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };

        let features = ["f0", "f0", "f1", "f1"];
        let shap = vec![-2.0_f64, 1.5, 0.5, -0.75];
        // `_inject_constant`: value in row 0, null everywhere else.
        let ref_zero = Float64Array::from(vec![Some(0.0_f64), None, None, None]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("shap_value", DataType::Float64, false),
            Field::new("feature", DataType::Utf8, false),
            Field::new("_ref_zero", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(shap)),
            Arc::new(StringArray::from(features.to_vec())),
            Arc::new(ref_zero),
        ]).unwrap();
        (spec, batch)
    }

    /// Every `SceneNode::Line`'s endpoints in one mark batch, in emission order.
    fn line_endpoints(batch: &ferrum_scene::MarkBatch) -> Vec<(f64, f64, f64, f64)> {
        batch.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { x1, y1, x2, y2, .. } => Some((*x1, *y1, *x2, *y2)),
            _ => None,
        }).collect()
    }

    /// The cycle-3 regression, pinned: a rule layer that declares only `x`
    /// must render ONE vertical span at that x, whatever the chart-level
    /// encoding inherits into it. Before the fix, the inherited ordinal
    /// `y="feature"` hijacked the shape into the horizontal-span mode and the
    /// layer drew one full-width horizontal span per row across the feature
    /// band centers, ignoring `x` entirely (in the real chart: 1000 spans and
    /// a 116 KB SVG on a blessed golden).
    ///
    /// Both halves of the ruling are load-bearing here, and each fails this
    /// test alone when reverted:
    /// - Relaxing `RuleShape::resolve`'s horizontal-span arm to accept a bound
    ///   `x` (the pre-fix `if let Some(yf)` gate) restores the hijack.
    /// - Dropping the own-span-axis normalization in
    ///   `build_panel_mark_batches` sends this shape to the `x`+`y` tie-break,
    ///   which is also a horizontal span.
    #[test]
    fn rule_layer_own_x_span_is_not_hijacked_by_inherited_ordinal_y() {
        let (spec, batch) = beeswarm_shaped_zero_line_spec();
        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        let rule_batches: Vec<&ferrum_scene::MarkBatch> = panel.marks.iter()
            .filter(|m| m.kind == ferrum_scene::MarkBatchKind::Rule)
            .collect();
        assert_eq!(rule_batches.len(), 1, "expected exactly one rule mark batch (the reference layer)");

        let lines = line_endpoints(rule_batches[0]);
        assert_eq!(lines.len(), 1,
            "the zero-line layer must draw exactly one line (its anchor column has one non-null row); \
             got {} — more than one means the inherited ordinal y drove the geometry",
            lines.len());
        let (x1, y1, x2, y2) = lines[0];
        assert_eq!(x1, x2, "a vertical zero line must keep one x for both endpoints; got x1={x1}, x2={x2}");
        assert_ne!(y1, y2, "a vertical zero line must span the panel height, not collapse");
        let plot = panel.plot_area;
        assert!((y1 - plot.y).abs() < 1e-9 && (y2 - (plot.y + plot.h)).abs() < 1e-9,
            "the span must run the full panel height ({}..{}); got {y1}..{y2}", plot.y, plot.y + plot.h);
        assert!(x1 > plot.x && x1 < plot.x + plot.w,
            "the line must sit inside the panel at the x=0 position; got x={x1} for plot {plot:?}");
    }

    /// The mirror, and the negative control for the normalization's direction:
    /// a rule layer declaring only `y` on a chart with a chart-level `x` keeps
    /// rendering a horizontal span (this is `residuals_chart`'s `y="_ref_zero"`
    /// layer, whose behavior must not change).
    #[test]
    fn rule_layer_own_y_span_is_not_flipped_by_inherited_x() {
        let (mut spec, batch) = beeswarm_shaped_zero_line_spec();
        {
            let layers = spec.layers.as_mut().expect("fixture always sets layers");
            let enc = &mut layers[1].encoding;
            enc.x = None;
            enc.y = Some(crate::spec::encoding::EncodingSpec {
                field: "_ref_zero".into(),
                ..Default::default()
            });
        }
        // A numeric y-scale for the horizontal span to land on.
        spec.encoding.y = Some(crate::spec::encoding::EncodingSpec {
            field: "shap_value".into(),
            ..Default::default()
        });
        spec.layers.as_mut().unwrap()[0].encoding.y = spec.encoding.y.clone();

        let scene = build_scene_for(&spec, &batch);
        let panel = &scene.panels[0];
        let rule_batch = panel.marks.iter()
            .find(|m| m.kind == ferrum_scene::MarkBatchKind::Rule)
            .expect("the reference layer must emit a rule batch");
        let lines = line_endpoints(rule_batch);
        assert_eq!(lines.len(), 1, "one non-null anchor row → one horizontal span");
        let (x1, y1, x2, y2) = lines[0];
        assert_eq!(y1, y2, "a horizontal span must keep one y for both endpoints");
        let plot = panel.plot_area;
        assert!((x1 - plot.x).abs() < 1e-9 && (x2 - (plot.x + plot.w)).abs() < 1e-9,
            "the span must run the full panel width ({}..{}); got {x1}..{x2}", plot.x, plot.x + plot.w);
    }

    // ── Task 13 spec c4 (spec §4.4, "Extended 2026-09-02"): the positional
    // provenance flags are coordinate-space-consistent ───────────────────────
    //
    // `prepare::build_layers` swaps `encoding.x`↔`y` (and `x2`↔`y2`) under
    // `CoordFlip`; `LayerPrepared::flip_coords` swaps `x_is_own`/`y_is_own`
    // WITH them, so a flag always describes the slot its channel now occupies
    // and a coord-flipped rule span renders exactly as its unflipped
    // counterpart does on the other axis. Pinned at BOTH `LayerPrepared`
    // constructors — `from_chart_only` (flat) and `from_chart_and_layer`
    // (layered) — because the flags are captured separately in each.
    //
    // Every `*_SPEC_JSON` below is the VERBATIM `chart.to_dict()` of the
    // public-API call named in its doc comment (captured 2026-09-02), parsed
    // through the same serde path `ChartSpec.from_json` uses, so these
    // fixtures cannot drift from what Python actually lowers.

    fn spec_from_lowered_json(json: &str) -> ChartSpec {
        serde_json::from_str(json).expect("lowered spec JSON must deserialize as a ChartSpec")
    }

    /// Every line emitted by every rule batch in panel 0, in emission order.
    fn rule_lines(scene: &ferrum_scene::SceneGraph) -> Vec<(f64, f64, f64, f64)> {
        scene.panels[0]
            .marks
            .iter()
            .filter(|m| m.kind == ferrum_scene::MarkBatchKind::Rule)
            .flat_map(line_endpoints)
            .collect()
    }

    /// `fm.Chart(df).mark_rule(stroke_dash=[4, 4]).encode(x="z").coord(fm.CoordFlip())`
    const FLIPPED_OWN_X_RULE_SPEC_JSON: &str = r#"{"data": {"kind": "named", "name": "default"}, "mark": "rule", "encoding": {"x": {"field": "z"}}, "coord": {"kind": "flip"}, "mark_style": {"stroke_dash": [4.0, 4.0]}}"#;

    /// `fm.Chart(df).mark_rule(stroke_dash=[4, 4]).encode(y="z")` — the
    /// unflipped counterpart of [`FLIPPED_OWN_X_RULE_SPEC_JSON`].
    const UNFLIPPED_OWN_Y_RULE_SPEC_JSON: &str = r#"{"data": {"kind": "named", "name": "default"}, "mark": "rule", "encoding": {"y": {"field": "z"}}, "mark_style": {"stroke_dash": [4.0, 4.0]}}"#;

    /// `pl.DataFrame({"z": [1.0, 2.0, 3.0, 4.0]})`.
    fn z_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![Field::new("z", DataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]))])
            .unwrap()
    }

    /// FLAT path (`LayerPrepared::from_chart_only`): a coord-flipped rule that
    /// declares only `x` renders exactly what its unflipped `y=` counterpart
    /// renders — four full-width horizontal spans, at identical pixels.
    ///
    /// RED (verified in place): drop `std::mem::swap(&mut self.x_is_own, &mut
    /// self.y_is_own)` from `LayerPrepared::flip_coords` and the flags still
    /// describe the PRE-flip slots — `(x_is_own, y_is_own) == (true, false)`
    /// against a post-flip encoding holding `y`. The own-span-axis
    /// normalization then clears the very channel the layer owns, handing
    /// `RuleShape::resolve` an all-absent encoding, and the render REFUSES the
    /// shape (`UnsupportedChannel`, whose own message advertises "x= alone").
    #[test]
    fn coord_flipped_rule_span_matches_its_unflipped_counterpart() {
        let batch = z_batch();
        let flipped = rule_lines(&build_scene_for(
            &spec_from_lowered_json(FLIPPED_OWN_X_RULE_SPEC_JSON),
            &batch,
        ));
        let unflipped = rule_lines(&build_scene_for(
            &spec_from_lowered_json(UNFLIPPED_OWN_Y_RULE_SPEC_JSON),
            &batch,
        ));
        assert_eq!(
            flipped.len(),
            4,
            "one horizontal span per row of `z`; got {flipped:?}"
        );
        for (x1, y1, x2, y2) in &flipped {
            assert_eq!(y1, y2, "a flipped `x=` span is HORIZONTAL; got {x1},{y1} → {x2},{y2}");
        }
        assert_eq!(
            flipped, unflipped,
            "a coord-flipped rule span must render exactly as its unflipped counterpart \
             does on the other axis (spec §4.4, 2026-09-02)"
        );
    }

    /// LAYERED path (`LayerPrepared::from_chart_and_layer`), the reviewer's
    /// cycle-4 repro: `fm.layer(fm.Chart(df).mark_bar(orient="horizontal")
    /// .encode(x=fm.X("cat", type_="ordinal"), y="v"), fm.Chart(df)
    /// .mark_rule(stroke_dash=[4, 4]).encode(x=fm.X("z", type_="ordinal")))`.
    /// `orient="horizontal"` is what sets `coord={"kind": "flip"}` here.
    ///
    /// The rule layer declares `x` and inherits `y="v"` from chart level, so
    /// it must draw ONE line at its OWN `z` value ("b", the second of four
    /// ordinal bands) and none at the bar's inherited `v` values. Under flip
    /// that own channel occupies the y slot, so the mirror spec (identical but
    /// unflipped) draws the same single line vertically on the x axis.
    ///
    /// RED (verified in place): without the flag swap this emits FOUR lines,
    /// one per row at the bar's `v` positions — `z` discarded entirely.
    #[test]
    fn coord_flipped_layered_rule_draws_its_own_channel_not_the_inherited_one() {
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        const FLIPPED: &str = r#"{"data": {"kind": "named", "name": "default"}, "mark": "bar", "encoding": {"x": {"field": "cat", "type": "ordinal"}, "y": {"field": "v", "scale": {"type": "linear", "clamp": false, "nice": false, "zero": true}}}, "layers": [{"mark": "bar", "encoding": {"x": {"field": "cat", "type": "ordinal"}, "y": {"field": "v"}}}, {"mark": "rule", "encoding": {"x": {"field": "z", "type": "ordinal"}}, "mark_style": {"stroke_dash": [4.0, 4.0]}}], "coord": {"kind": "flip"}}"#;

        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
            // `z` names one existing category, in one row: the reference line
            // the layer owns. The remaining rows are null, exactly as
            // `_inject_constant` leaves them.
            Field::new("z", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
                Arc::new(Float64Array::from(vec![9.0, 3.0, 6.0, 1.0])),
                Arc::new(StringArray::from(vec![Some("b"), None, None, None])),
            ],
        )
        .unwrap();

        let mut unflipped_spec = spec_from_lowered_json(FLIPPED);
        unflipped_spec.coord = None;

        let flipped_scene = build_scene_for(&spec_from_lowered_json(FLIPPED), &batch);
        let flipped = rule_lines(&flipped_scene);
        assert_eq!(
            flipped.len(),
            1,
            "the rule layer's own `z` has ONE non-null row → one line; {} means the \
             inherited `v` drove the geometry",
            flipped.len()
        );
        let (x1, y1, x2, y2) = flipped[0];
        assert_eq!(y1, y2, "under flip the layer's own channel spans HORIZONTALLY");
        let plot = flipped_scene.panels[0].plot_area;
        assert!(
            (x1 - plot.x).abs() < 1e-9 && (x2 - (plot.x + plot.w)).abs() < 1e-9,
            "the span must run the full panel width ({}..{}); got {x1}..{x2}",
            plot.x,
            plot.x + plot.w
        );
        assert!(
            y1 > plot.y + plot.h * 0.25 && y1 < plot.y + plot.h * 0.5,
            "the span must sit on the SECOND of four ordinal bands (`z` == \"b\"); got y={y1} \
             for plot {plot:?}"
        );

        let unflipped_scene = build_scene_for(&unflipped_spec, &batch);
        let unflipped = rule_lines(&unflipped_scene);
        assert_eq!(unflipped.len(), 1, "the mirror draws one line too; got {unflipped:?}");
        let (ux1, uy1, ux2, uy2) = unflipped[0];
        let uplot = unflipped_scene.panels[0].plot_area;
        assert_eq!(ux1, ux2, "unflipped, that same own channel spans VERTICALLY");
        assert!(
            (uy1 - uplot.y).abs() < 1e-9 && (uy2 - (uplot.y + uplot.h)).abs() < 1e-9,
            "the mirror span must run the full panel height; got {uy1}..{uy2}"
        );
        assert!(
            ux1 > uplot.x + uplot.w * 0.25 && ux1 < uplot.x + uplot.w * 0.5,
            "the mirror must sit on the same second band; got x={ux1} for plot {uplot:?}"
        );
    }

    /// A second endpoint with no anchor to pair it with is refused by name,
    /// not silently drawn as some other shape that ignores it (batch-A Task 13
    /// spec c3). `x`+`x2` with no `y` and `x`+`y2` with no `y` both used to
    /// slip past the presence gate (it only asked whether `x` was bound) and
    /// then render a plain vertical span, dropping the second endpoint
    /// entirely — the same "silently WRONG" class as the span/ranged hijacks.
    #[test]
    fn validate_mark_encoding_rule_refuses_second_endpoint_without_its_anchor() {
        let unsupported = [
            rule_encoding(Some("x"), None, Some("x2"), None),
            rule_encoding(Some("x"), None, None, Some("y2")),
            rule_encoding(Some("x"), None, Some("x2"), Some("y2")),
            rule_encoding(None, Some("y"), Some("x2"), None),
            rule_encoding(None, None, Some("x2"), Some("y2")),
        ];
        for enc in &unsupported {
            let err = validate_mark_encoding(&Mark::Rule, enc, false)
                .expect_err("a dangling second endpoint must be refused, not silently dropped");
            assert!(
                format!("{err}").contains("mark_rule supports:"),
                "the refusal must name the supported set; got {err}"
            );
        }
    }

    /// Every one of rule's five supported channel-presence shapes (module doc
    /// on `marks/rule.rs`) must pass validation — a regression guard against
    /// the presence-check above being drawn too tightly and newly rejecting a
    /// combination `marks/rule.rs::build` already knows how to render. Covers
    /// both flipped and unflipped, since the check reads only channel
    /// presence (`channel: "positional"` is not `x`/`y`/`x2`/`y2`, so
    /// `with_coord_flipped`'s un-flip is a no-op either way).
    #[test]
    fn validate_mark_encoding_rule_accepts_all_five_supported_shapes() {
        let shapes = [
            rule_encoding(None, Some("y"), None, None),          // horizontal span
            rule_encoding(Some("x"), None, None, None),          // vertical span
            rule_encoding(Some("x"), Some("y"), None, Some("y2")), // ordinal-x ranged
            rule_encoding(Some("x"), Some("y"), Some("x2"), None), // ordinal-y ranged
            rule_encoding(Some("x"), Some("y"), Some("x2"), Some("y2")), // diagonal
        ];
        for coord_flipped in [false, true] {
            for enc in &shapes {
                assert!(
                    validate_mark_encoding(&Mark::Rule, enc, coord_flipped).is_ok(),
                    "shape {enc:?} (coord_flipped={coord_flipped}) must be accepted for mark_rule"
                );
            }
        }
    }

    // ── R1 port (bug_hunt_draw.rs / bug_hunt_render_pipeline.rs): break-axis
    //    coordinate remapping (remap_coord/remap_node/remap_path_cmd) and the
    //    polar outer-radius helper. Zero prior in-src coverage of these fns.

    fn one_segment_break(d_lo: f64, d_hi: f64, px_lo: f64, px_hi: f64) -> break_axis::BreakResult {
        break_axis::apply_break_to_scale((d_lo, d_hi), &[], (px_lo, px_hi), 12.0)
    }

    fn two_segment_break() -> break_axis::BreakResult {
        // Domain [0,100] with a gap [40,60] -> two retained segments.
        break_axis::apply_break_to_scale((0.0, 100.0), &[[40.0, 60.0]], (50.0, 350.0), 12.0)
    }

    #[test]
    fn remap_coord_identity_maps_pixel_to_pixel() {
        let br = one_segment_break(0.0, 100.0, 50.0, 350.0);
        let px = remap_coord(200.0, 0.0, 100.0, 50.0, 350.0, &br).unwrap();
        assert!(
            (px - 200.0).abs() < 1e-6,
            "single-segment remap must be identity; got {px}"
        );
    }

    #[test]
    fn remap_coord_pixel_in_gap_returns_none() {
        let br = two_segment_break();
        // Pixel 200 maps back to data value 50, which falls inside the gap [40, 60].
        let result = remap_coord(200.0, 0.0, 100.0, 50.0, 350.0, &br);
        assert!(
            result.is_none(),
            "a pixel that reverse-maps into the gap must return None"
        );
    }

    #[test]
    fn remap_coord_nan_pixel_returns_none() {
        let br = one_segment_break(0.0, 100.0, 50.0, 350.0);
        assert!(remap_coord(f64::NAN, 0.0, 100.0, 50.0, 350.0, &br).is_none());
    }

    #[test]
    fn remap_node_circle_hides_at_break_hidden_when_in_gap() {
        let br = two_segment_break();
        let mut node = SceneNode::Circle {
            cx: 200.0,
            cy: 100.0,
            r: 4.0,
            style: default_fill_stroke(),
        };
        remap_node(&mut node, "x", 0.0, 100.0, 50.0, 350.0, &br);
        let SceneNode::Circle { cx, .. } = node else {
            panic!("expected Circle")
        };
        assert_eq!(
            cx, BREAK_HIDDEN,
            "a mark inside the break gap must be hidden off-screen"
        );
    }

    /// Discriminating fixture (quality-review kill-switch finding, cycle 2):
    /// `one_segment_break()` has no gaps, so remapping through it is the
    /// identity — an assertion of "unchanged" is satisfied whether or not the
    /// real remap dispatch runs at all, which is exactly how the reviewer's
    /// mutation (deleting the x-axis remap call) survived undetected the first
    /// time. `two_segment_break()`'s gap `[40,60]` makes `cx: 110.0` reverse-map
    /// to data value 20.0 (inside the retained segment `[0,40]`), which the
    /// broken scale compresses to pixel 122.0 — verified independently:
    /// `remap_coord(110.0, 0.0, 100.0, 50.0, 350.0, &two_segment_break())
    /// == Some(122.0)`. A disabled remap would leave `cx` at 110.0 instead.
    #[test]
    fn remap_node_circle_moves_to_compressed_pixel_across_a_gap() {
        let br = two_segment_break();
        let mut node = SceneNode::Circle {
            cx: 110.0,
            cy: 100.0,
            r: 4.0,
            style: default_fill_stroke(),
        };
        remap_node(&mut node, "x", 0.0, 100.0, 50.0, 350.0, &br);
        let SceneNode::Circle { cx, .. } = node else {
            panic!("expected Circle")
        };
        assert!(
            (cx - 122.0).abs() < 1e-6,
            "expected cx compressed to 122.0 across the break gap; got {cx}"
        );
    }

    #[test]
    fn remap_path_cmd_hlineto_only_remapped_on_x_axis() {
        // Same 110.0 -> 122.0 discriminating fixture as the Circle test above.
        let br = two_segment_break();
        let mut on_x = PathCmd::HLineTo { x: 110.0 };
        remap_path_cmd(&mut on_x, "x", 0.0, 100.0, 50.0, 350.0, &br);
        let PathCmd::HLineTo { x } = on_x else {
            panic!()
        };
        assert!(
            (x - 122.0).abs() < 1e-6,
            "HLineTo.x must compress to 122.0 through the broken scale when axis == x; got {x}"
        );

        let mut on_y = PathCmd::HLineTo { x: 110.0 };
        remap_path_cmd(&mut on_y, "y", 0.0, 100.0, 50.0, 350.0, &br);
        let PathCmd::HLineTo { x } = on_y else {
            panic!()
        };
        assert_eq!(x, 110.0, "HLineTo.x must NOT be touched when axis == y");
    }

    #[test]
    fn remap_path_cmd_vlineto_only_remapped_on_y_axis() {
        // Same 110.0 -> 122.0 discriminating fixture as the Circle test above.
        let br = two_segment_break();
        let mut on_y = PathCmd::VLineTo { y: 110.0 };
        remap_path_cmd(&mut on_y, "y", 0.0, 100.0, 50.0, 350.0, &br);
        let PathCmd::VLineTo { y } = on_y else {
            panic!()
        };
        assert!(
            (y - 122.0).abs() < 1e-6,
            "VLineTo.y must compress to 122.0 through the broken scale when axis == y; got {y}"
        );

        let mut on_x = PathCmd::VLineTo { y: 110.0 };
        remap_path_cmd(&mut on_x, "x", 0.0, 100.0, 50.0, 350.0, &br);
        let PathCmd::VLineTo { y } = on_x else {
            panic!()
        };
        assert_eq!(y, 110.0, "VLineTo.y must NOT be touched when axis == x");
    }

    #[test]
    fn polar_outer_radius_uses_smaller_dimension() {
        assert_eq!(
            polar_outer_radius(&Rect {
                x: 0.0,
                y: 0.0,
                w: 200.0,
                h: 200.0
            }),
            100.0
        );
        assert_eq!(
            polar_outer_radius(&Rect {
                x: 0.0,
                y: 0.0,
                w: 400.0,
                h: 200.0
            }),
            100.0,
            "outer radius uses the smaller of width/height"
        );
        assert_eq!(
            polar_outer_radius(&Rect {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 200.0
            }),
            0.0
        );
    }
}
