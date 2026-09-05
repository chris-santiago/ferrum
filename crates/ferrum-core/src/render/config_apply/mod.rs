//! Chart-level configuration application: the one ordered pipeline that layers
//! `ChartConfig` (the `configure_*()` surface, precedence level 3) onto the
//! prepared inputs, the effective theme and the legend overrides.
//!
//! `apply_chart_config_pipeline` performs, in order:
//!    1. Chart-level legend suppression (`configure_legend(orient="none")`).
//!    2. Per-axis `axis_x` / `axis_y` slot fill.
//!    3. `configure_grid(x=…, y=…)`'s per-axis slot fill.
//!    4. Shared `axis` slot fill.
//!    5. Concrete axis-side re-sync from the merged `overrides.orient`.
//!    6. `axis_y2` slot fill, for every secondary y axis.
//!    7. `tick_extra` / `tick_min_step` tick adjustment against the provisional scales.
//!    8. `label_format` re-formatting of the tick labels.
//!    9. Tick-placement re-sync for explicit `tick_values`: repairs whichever
//!       carrier the axis holds (the continuous projected-fraction carrier, or
//!       the categorical band-center placement when its length no longer
//!       matches the override) so layout never indexes a stale vec.
//!   10. Passes 7–9 again, for every secondary y axis.
//!   11. Color `domain`/`range` overrides on the resolved color scale (11a),
//!       then the categorical legend-entry rebuild from that new domain (11b).
//!       The halves are numbered because they are separately re-executed:
//!       `scene_build` re-runs 11a alone.
//!   12. Per-channel `Legend(values=[…])` legend-entry filter.
//!   13. Effective theme: `ChartConfig` over the caller's theme.
//!   14. The three theme-backed legend fields' per-channel > chart-level cascade.
//!   15. `grid`/`domain` show-defaults from the effective theme.
//!
//! Tier 6 then executes its three passes in the order **17, 18, 16** — not
//! 16, 17, 18. Written as a plain list because an ordered one would be
//! renumbered by the renderer, which would silently undo the statement:
//!
//! - **Pass 17** — `LegendOverrides` projection from the prepared inputs.
//! - **Pass 18** — `configure_legend` fill of the override slots still unset.
//! - **Pass 16** — effective legend title.
//!
//! The numbers label the precedence LAYERS (16 is the title layer, 17–18 the
//! overrides layer); the execution order is what this list shows. Resolving the
//! title last is safe because it only reads `prep`, which 17 and 18 do not
//! mutate — see [`resolve_legend_overrides_and_title`].
//!
//! That order is stated HERE and nowhere else. Each tier fn's name carries the
//! precedence rule its passes implement, so a reorder has to argue with a
//! function name rather than with a comment; three merge disciplines are in
//! play and the names distinguish them — fill-if-unset (2, 3, 4, 6, 15, 18),
//! unconditional overwrite (1, 8, 11a, 11b, 12, 13), and a single-expression
//! `.or()` cascade (14).
//!
//! Every partial re-execution outside this module goes through a named entry
//! rather than restating a slice of the order. In production there are exactly
//! two: `composite_render`'s figure-legend seam reads passes 16–18's result off
//! `PipelineOutput` (one value, computed once — it no longer recomputes them),
//! and `scene_build` re-applies the SCALE-EDIT HALF of pass 11 (11a alone, not
//! the legend-entry rebuild) after its own per-panel / per-legend scale
//! resolution, via [`apply_color_config_to_color_scale`]. Test code in
//! `scene_build` still hand-assembles a layout around `apply_show_defaults`
//! (pass 15); that is test plumbing, not a production re-execution.
//!
//! [`validate_chart_format_specs`] lives here too — it is a gate over the same
//! `ChartConfig` surface — but it is NOT part of the pipeline: `prepare_and_layout`
//! calls it before `prepare_render_inputs`, since it must refuse a malformed
//! format spec before any transform work begins.

#[cfg(test)]
mod tests;

use crate::layout::{
    LegendDirection, LegendOrient, LegendOverrides, TextAnchor, ThemeInputs,
};
use crate::spec::chart::ChartSpec;

use super::chart_config::{self, AxisConfigSpec, ChartConfig};
use super::{color, format, prepare, scale_resolve, RenderError, RenderWarning};

/// What [`apply_chart_config_pipeline`] produces beyond its in-place edits to
/// `prep`: the three values `compute_layout` needs that do not live on the
/// prepared inputs.
pub(in crate::render) struct AppliedChartConfig {
    /// The caller's theme with every `configure_*()` override layered in
    /// (passes 13–14).
    pub effective_theme: ThemeInputs,
    /// Per-channel `Legend(...)` overrides, filled from `configure_legend(...)`
    /// wherever the per-channel level left a slot unset (passes 17–18).
    pub legend_overrides: LegendOverrides,
    /// The resolved legend title — `None` for both "no legend" and the
    /// explicit `Legend(title=None)` suppress (pass 16).
    pub legend_title: Option<String>,
}

/// Apply every chart-level `configure_*()` override, in the one order the
/// module doc states.
///
/// Mutates `prep` in place (axis slots, tick products, color scale, legend
/// entries) and returns the products that outlive it in [`AppliedChartConfig`].
/// `warnings` accumulates what the overrides report — an `axis_y2` section with
/// no secondary axis to land on, color-domain/range degradations, unknown
/// `Legend(values=)` names.
pub(in crate::render) fn apply_chart_config_pipeline(
    prep: &mut prepare::PreparedInputs,
    theme: &ThemeInputs,
    chart_config: &ChartConfig,
    warnings: &mut Vec<RenderWarning>,
) -> Result<AppliedChartConfig, RenderError> {
    suppress_legend_if_chart_level_disabled(prep, chart_config);
    fill_axis_slots_specific_before_shared(&mut prep.axes, chart_config, warnings)?;
    resync_ticks_after_axis_merge(
        &mut prep.axes,
        TickResyncCtx {
            scales: &prep.provisional_scales,
            x_tick_count: prep.x_tick_count,
            y_tick_count: prep.y_tick_count,
            secondary_scales: &prep.secondary_y_scales,
            secondary_tick_counts: &prep.secondary_y_tick_counts,
        },
    );
    apply_color_config_then_filter_legend_entries(
        &mut prep.provisional_scales.color,
        &mut prep.legend_entries,
        prep.legend_overrides.values.as_deref(),
        chart_config,
        warnings,
    );
    let effective_theme = build_effective_theme_config_over_theme(prep, theme, chart_config);
    let (legend_overrides, legend_title) = resolve_legend_overrides_and_title(prep, chart_config);
    Ok(AppliedChartConfig { effective_theme, legend_overrides, legend_title })
}

// ── Tier 1 (pass 1): chart-level legend suppression ─────────────────────────

/// Chart-level `configure_legend(orient="none")` suppression (GH #74).
///
/// Python's `_resolve_chart_config` (`_render.py`) maps a fully-merged
/// `orient="none"` onto the same `disabled` signal `Color(legend=None)` sets at
/// the per-channel level ([`chart_config_legend_disabled`] reads it here) —
/// there is no `LegendOrient::None` variant on the Rust side.
/// `prepare::prepare_render_inputs` only reads the per-channel
/// `encoding.<channel>.legend.disabled` flag (Schwabish SB3), so the
/// chart-level signal is applied here by clearing the already-built legend
/// content — the exact same empty state that flag already produces.
///
/// One seam covers every consumer: `render_svg` / `render_scene_json` reach
/// this through `prepare_and_layout` for a standalone chart, and
/// `composite_render::render_leaf` reaches the SAME pipeline with the leaf's
/// own `chart_config` for every composite leaf. A suppressed leaf therefore
/// draws no per-panel legend AND yields an empty `LeafLegendBundle`, which
/// `apply_legend_band` already treats as "no content to capture" — the
/// identical degrade an all-disabled `Color(legend=None)` composite produces
/// today (design §4/§9.8).
fn suppress_legend_if_chart_level_disabled(
    prep: &mut prepare::PreparedInputs,
    chart_config: &ChartConfig,
) {
    if !chart_config_legend_disabled(chart_config) {
        return;
    }
    prep.legend_entries.clear();
    prep.colorbar = None;
    prep.aux_legends.clear();
    prep.legend_title = None;
    prep.legend_overrides.title = None;
}

// ── Tier 2 (passes 2–6): axis-slot fill, most specific first ────────────────

/// Apply the chart-level axis config (level 3) to the `AxisInput` slots, MOST
/// SPECIFIC FIRST — which is what makes `axis_x`/`axis_y` > `grid.x`/`grid.y` >
/// `axis` hold (level 2, the per-channel `fm.Axis(...)`, has already claimed
/// its slots and always wins).
///
/// These styling fields use fill-only-if-`None` — FIRST writer claims the slot
/// — so the more-specific source must run FIRST. That is the OPPOSITE order
/// from the overwrite-semantics theme path in [`apply_chart_config_to_theme`], where
/// last-writer-wins makes the shared `axis` key run first and `axis_x` second.
/// The inversion is the whole reason the two orders cannot be read off each
/// other, and is why this function's name states which one it implements.
///
/// R3 (chart-level `configure_axis` chain): deliberately EXEMPT from
/// `with_coord_flipped` — this is the mirror image of the `SortSpecIgnored`
/// exemption (`scale_resolve/domain.rs`'s `apply_channel_shorthand_sort`).
/// [`apply_axis_config_to_axis_input`] → [`apply_axis_style_to_axis_input`]
/// derives `channel` from the axis's PHYSICAL dimension
/// (`orient.dimension().channel_token()`: `Top|Bottom → "x"`, else `"y"`),
/// never from a user-written encoding channel.
/// And the config KEY the user actually typed here — `axis_x`/`axis_y` (or the
/// shared `axis`) — is itself resolved-slot vocabulary: nothing on the Python
/// side remaps `configure_axis(axis_x=…)` to `axis_y=` under `CoordFlip`
/// (`configure.py`, `_override_apply.py`, `_override_consume.py`). So for THIS
/// chain the resolved token already IS what the user wrote — `prep.axes.x` is
/// unconditionally the physical bottom axis, flip or not (flip is implemented
/// purely as the `x`/`y` encoding swap in `prepare::build_layers`; nothing
/// re-orients axes). Applying `with_coord_flipped` here would translate the
/// user's own typed config key AWAY from what they wrote — see
/// `chart_level_orient_error_names_resolved_axis_under_flip`.
fn fill_axis_slots_specific_before_shared(
    axes: &mut crate::layout::AxesInput,
    chart_config: &ChartConfig,
    warnings: &mut Vec<RenderWarning>,
) -> Result<(), RenderError> {
    apply_axis_config_to_axis_input(&mut axes.x, chart_config.axis_x.as_ref())?;
    apply_axis_config_to_axis_input(&mut axes.y, chart_config.axis_y.as_ref())?;
    // `configure_grid(x=…, y=…)` sits between the two axis layers: more
    // specific than the shared `axis` key, less specific than `axis_x`/`axis_y`
    // (D4, spec §4.3 — see `apply_grid_config_to_axis_inputs`).
    apply_grid_config_to_axis_inputs(axes, chart_config);
    apply_axis_config_to_axis_input(&mut axes.x, chart_config.axis.as_ref())?;
    apply_axis_config_to_axis_input(&mut axes.y, chart_config.axis.as_ref())?;
    // Re-sync the concrete axis side from the merged `overrides.orient`: a
    // per-channel `fm.Axis(orient=...)` already set it (so this is a no-op there
    // and per-channel wins), otherwise a chart-level `configure_axis(orient=...)`
    // filled it above and now takes effect.
    axes.x.resolve_orient();
    axes.y.resolve_orient();
    // `axis_y2` (D2/F-L07-06): applies ONLY to the secondary y axis, via the
    // SAME fill-only per-axis path `axis`/`axis_x`/`axis_y` use — deliberately
    // NOT the shared-theme path (`apply_axis_config_to_theme`) those three also
    // feed, since that path is genuinely global and would leak the "secondary y
    // only" override onto the primary x/y axes' fallback. A chart with no
    // `independent_y` layer has an empty `secondary_y` Vec (#52) — nothing to
    // fill, so the override is reported rather than silently dropped (spec §4.1).
    //
    // No `resolve_orient()` call here, unlike the primary x/y pair above:
    // `layout::layout_panel_axes` unconditionally overwrites every secondary
    // axis's concrete `orient` to `Right` on a CLONE of this input
    // (`layout/mod.rs`'s "orient forced `Right`" comment) — every secondary
    // axis always renders on the right, stacked outward, by design; a
    // per-channel `fm.Axis(orient=...)` on an `independent_y` layer is
    // overwritten the same way. So `axis_y2.orient` reaches
    // `AxisStyleOverrides.orient` (fill-only, like every other field here)
    // but has no concrete side left to resolve onto — a `resolve_orient()`
    // call would be a no-op, not a fix, so this block does not carry one.
    if let Some(ref axis_y2_cfg) = chart_config.axis_y2 {
        if axes.secondary_y.is_empty() {
            warnings.push(RenderWarning::ConfigSurfaceNotPresent { section: "axis_y2".to_string() });
        } else {
            for secondary in axes.secondary_y.iter_mut() {
                apply_axis_config_to_axis_input(secondary, Some(axis_y2_cfg))?;
            }
        }
    }
    Ok(())
}

// ── Tier 3 (passes 7–10): tick products, re-derived after the merge ─────────

/// Everything the tick re-sync READS, as one bundle: the resolved scales and
/// the tick counts each axis was built with. Every field is a by-axis
/// correspondence — `scales.x`/`x_tick_count` describe one axis,
/// `scales.y`/`y_tick_count` the other, and `secondary_scales[i]` /
/// `secondary_tick_counts[i]` the i-th secondary axis — which is exactly what
/// six positional parameters could not say: the two bare `usize` were
/// distinguishable only by argument order, and nothing but ordering tied each
/// count to its scale. Grouped because they travel together and must stay
/// index-aligned (`prepare` carries the secondary pair out of `build_axes` for
/// this one consumer), not because they are merely adjacent. `axes` stays a
/// separate `&mut` parameter: this bundle is the read-only half.
#[derive(Clone, Copy)]
struct TickResyncCtx<'a> {
    /// The provisional scales the primary tick labels were derived from.
    scales: &'a scale_resolve::ResolvedScales,
    /// `resolve_axis_tick_count`'s output for the primary x axis.
    x_tick_count: usize,
    /// `resolve_axis_tick_count`'s output for the primary y axis.
    y_tick_count: usize,
    /// One scale per `axes.secondary_y` entry, same order.
    secondary_scales: &'a [scale_resolve::ScaleKind],
    /// One tick count per `axes.secondary_y` entry, same order.
    secondary_tick_counts: &'a [usize],
}

/// Re-derive every tick product that the axis-slot merge above can have
/// changed — AFTER that merge, so each adjustment sees the effective value
/// (per-channel wins, chart-level fallback) rather than a half-merged one.
///
/// `tick_extra` / `tick_min_step` (B5 unit 2) adjust the generated ticks
/// against the provisional scale; `apply_label_format_to_axis` re-formats the
/// labels (it requires the axis config to be set first);
/// `sync_tick_placement_to_tick_values` re-pairs the axis's placement carrier
/// (continuous projection or categorical placement) with those new labels, so
/// neither is left index-addressed against a stale length. No-ops when none of
/// those fields is set, so default output is byte-identical.
///
/// The non-ordinal y labels/fractions were reversed in prepare, so the raw
/// values are reversed in lockstep. The EXPLICIT labels are not reversed
/// (unlike auto labels), so the value-order fractions align directly.
fn resync_ticks_after_axis_merge(axes: &mut crate::layout::AxesInput, ctx: TickResyncCtx<'_>) {
    let TickResyncCtx {
        scales,
        x_tick_count,
        y_tick_count,
        secondary_scales,
        secondary_tick_counts,
    } = ctx;
    let y_reversed = !matches!(scales.y, scale_resolve::ScaleKind::Ordinal(_));
    let (x_tc, y_tc) = (x_tick_count, y_tick_count);
    prepare::adjust_axis_ticks(&mut axes.x, &scales.x, x_tc, false);
    prepare::adjust_axis_ticks(&mut axes.y, &scales.y, y_tc, y_reversed);
    apply_label_format_to_axis(&mut axes.x, &scales.x, x_tc, false);
    apply_label_format_to_axis(&mut axes.y, &scales.y, y_tc, y_reversed);
    sync_tick_placement_to_tick_values(&mut axes.x, &scales.x);
    sync_tick_placement_to_tick_values(&mut axes.y, &scales.y);
    // The SAME three post-config tick adjustments for every secondary y axis
    // (spec §4.9, extended 2026-09-02). `axis_y2`'s `label_format`,
    // `label_format_type`, `tick_extra`, `tick_min_step` and `values` reached
    // `AxisStyleOverrides` before that task but had no consumer: this trio ran
    // on `prep.axes.x`/`.y` only, so a secondary axis carried the settings and
    // ignored them. Each secondary axis is paired with the scale it was built
    // from (`prep.secondary_y_scales`, same order) — the piece prepare now
    // carries out specifically so this loop can exist.
    //
    // `reversed`/`tick_count` follow the primary y's own rules, because
    // `build_secondary_y_axis_inputs` builds these through the identical
    // `build_axis_input(Channel::Y, …)` path — `tick_count` is that builder's
    // OWN `resolve_axis_tick_count` output, carried out beside the scales
    // (`prep.secondary_y_tick_counts`), exactly as the primary pair passes
    // `prep.x_tick_count`/`prep.y_tick_count`. Deriving it from
    // `tick_labels.len()` would only accidentally agree: `adjust_axis_ticks`
    // re-derives `tick_values_raw(tick_count)` and bails on a length
    // mismatch, so any data shape where the two differ would silently drop
    // `axis_y2`'s tick_extra/tick_min_step/values.
    for ((secondary, scale), &tc) in axes
        .secondary_y
        .iter_mut()
        .zip(secondary_scales.iter())
        .zip(secondary_tick_counts.iter())
    {
        let reversed = !matches!(scale, scale_resolve::ScaleKind::Ordinal(_));
        prepare::adjust_axis_ticks(secondary, scale, tc, reversed);
        apply_label_format_to_axis(secondary, scale, tc, reversed);
        sync_tick_placement_to_tick_values(secondary, scale);
    }
}

// ── Tier 4 (passes 11a, 11b, 12): color config, THEN the legend-entry filter ─

/// Apply `configure_color(domain=/range=)` to the resolved color scale and
/// rebuild the categorical legend entries from it — and only THEN apply the
/// per-channel `Legend(values=[…])` filter.
///
/// The order inside this tier is load-bearing in both directions, which is why
/// the two passes share one function:
///   - the entry rebuild ([`resync_categorical_legend_entries`]) must follow the
///     scale edit, or the swatch colors would follow the new domain order while
///     the labels kept the old one;
///   - the `values` filter (D6/F-L04-05) must follow the rebuild, because the
///     rebuild reconstructs `legend_entries` wholesale from the resolved domain
///     and would otherwise undo the filter (A-B-A). It must also precede
///     `compute_layout`, so the filtered set is what layout sizes, what the
///     overflow accounting counts, and what a compositor captures for a
///     figure-level legend.
///
/// This is the ONE reporting application of the color config: `scene_build`'s
/// per-panel and per-legend re-applications run the same config against the
/// same scale, so they discard their (identical) warnings rather than emit one
/// per panel.
fn apply_color_config_then_filter_legend_entries(
    color_scale: &mut Option<scale_resolve::ColorScale>,
    legend_entries: &mut Vec<crate::layout::LegendEntry>,
    legend_values: Option<&[String]>,
    chart_config: &ChartConfig,
    warnings: &mut Vec<RenderWarning>,
) {
    if let Some(ref cfg) = chart_config.color {
        warnings.extend(apply_color_config_to_color_scale(color_scale, cfg));
        resync_categorical_legend_entries(legend_entries, color_scale.as_ref());
    }
    apply_legend_values_to_entries(legend_entries, legend_values, warnings);
}

// ── Tier 5 (passes 13–15): effective theme, config over theme ──────────────

/// Build the effective theme: start from the caller-supplied theme, then layer
/// in the overrides from lowest to highest priority within the "render"
/// concern. Per-channel axis overrides (level 2) are NOT here — they live in
/// `AxisInput` and take effect at layout time.
///
/// The D7 cascade repair is expressed as a SPLIT OF OWNERSHIP, not as an
/// ordering: the three `ThemeInputs`-backed legend fields both levels write
/// (`orient`, `columns`, `title_font_size`) resolve together, per-channel
/// first, inside [`apply_legend_cascade_to_theme`], and
/// [`apply_chart_config_to_theme`] deliberately does not touch them — a second
/// unconditional writer for them is exactly what inverted their cascade before
/// D7.
///
/// Accuracy note (mutation review M7): BECAUSE that split holds, the two
/// writers' field sets are disjoint today, so swapping these two calls is
/// currently behavior-preserving and no test would catch it. The order is
/// therefore a statement of intent, not a live constraint — do not read it as
/// protection. What protects the cascade is the ownership split above: if a
/// field is ever added to BOTH writers, the order becomes load-bearing again
/// and nothing here will say so. Add it to one writer only.
///
/// Finally `apply_show_defaults` closes the grid/domain precedence chain (D4,
/// spec §4.3): every axis that expressed no opinion of its own now takes the
/// effective theme's. It runs AFTER [`apply_chart_config_to_theme`] so
/// `configure_grid(color=…)`'s both-axes shorthand is already in the theme, and
/// before `compute_layout` so `AxisLayout.show_grid`/`show_domain` carry the
/// FINAL per-axis answer — which is why `build_grid`/`build_axis` no longer
/// re-consult the theme themselves.
fn build_effective_theme_config_over_theme(
    prep: &mut prepare::PreparedInputs,
    theme: &ThemeInputs,
    chart_config: &ChartConfig,
) -> ThemeInputs {
    let mut effective_theme = theme.clone();
    apply_chart_config_to_theme(&mut effective_theme, chart_config);
    apply_legend_cascade_to_theme(&mut effective_theme, &prep.legend_overrides, chart_config);
    prep.axes.apply_show_defaults(&effective_theme);
    effective_theme
}

// ── Tier 6 (passes 17, 18, 16): the legend-overrides projection ────────────

/// Project one leaf's prepared legend state into the pair `compute_layout` (and
/// a compositor's figure-legend seam) consumes: the `LegendOverrides` bundle
/// with `configure_legend(...)` (level 3) filling whatever the per-channel
/// `Legend(...)` (level 2) left unset, plus the three-way-resolved title.
///
/// Runs 17 → 18 → 16 (see the module doc's note): the numbers label precedence
/// layers, the call order is what the code does, and the title resolution reads
/// only `prep`, which the other two do not mutate.
///
/// Private, and no longer `_leaf_`-named: `composite_render::capture_leaf_bundle`
/// used to call this directly (and, before #143, hand-copied its three lines),
/// but it now reads the result off `PipelineOutput`. The only caller left is
/// [`apply_chart_config_pipeline`], so there is no leaf-specific caller for the
/// name to describe — one value, one producer, one caller.
fn resolve_legend_overrides_and_title(
    prep: &prepare::PreparedInputs,
    chart_config: &ChartConfig,
) -> (LegendOverrides, Option<String>) {
    let mut overrides = legend_overrides_from_prep(prep);
    apply_chart_config_to_legend_overrides(&mut overrides, chart_config);
    // D13 + v0.15.1: the legend title override replaces the default field-name
    // title when `Some`.
    let title = effective_legend_title(prep);
    (overrides, title)
}

// ── Moved helpers: one `apply_<source>_to_<target>` per config surface ─────

/// Apply [`ChartConfig`] overrides to a [`ThemeInputs`] clone.
///
/// This implements "configure > theme" precedence (level 3 > level 4–5). It is
/// pass 13, the unconditional-overwrite half of
/// [`build_effective_theme_config_over_theme`].
///
/// It handles only the fields for which the theme slot is a genuine FALLBACK,
/// consulted by a later `.or(...)`. The three legend fields whose theme slot
/// also carries the resolved per-channel value are deliberately absent: they
/// resolve in [`apply_legend_cascade_to_theme`], which runs right after this
/// and sees both levels at once (D7 — a second unconditional writer here is
/// what inverted their cascade).
///
/// Per-channel `axis=Axis(...)` overrides live in `AxisInput` and are resolved
/// by `prepare_render_inputs` — they take effect at layout time (level 2) and
/// are never touched here.
fn apply_chart_config_to_theme(theme: &mut ThemeInputs, config: &ChartConfig) {
    // ── Grid overrides (both-axes shorthand only) ─────────────────────────────
    // `configure_grid(x=…, y=…)` no longer lands here at all (D4/F-L07-01,
    // spec §4.3): those are per-axis and are applied to each `AxisInput`'s own
    // override slots by `apply_grid_config_to_axis_inputs`. What remains is the
    // axis-unspecified shorthand — `color`/`width`/`dash`/`opacity` with no
    // axis named, which by definition means both axes, i.e. the theme.
    //
    // The deleted block could not express disagreement: it flipped the single
    // global `theme.grid.grid` only when x and y AGREED, so
    // `configure_grid(x=True, y=False)` was silently dropped in full — the
    // caller's whole request, gone, with no warning.
    if let Some(ref grid_cfg) = config.grid {
        if let Some(ref color_str) = grid_cfg.color {
            if let Ok(c) = color::parse_color(color_str) {
                theme.colors.grid_color = c;
            }
        }
        if let Some(w) = grid_cfg.width {
            theme.sizes.grid_width = w;
        }
        if let Some(ref d) = grid_cfg.dash {
            theme.grid.grid_dash = Some(d.clone());
        }
        if let Some(o) = grid_cfg.opacity {
            theme.grid.grid_opacity = o;
        }
    }

    // ── Padding overrides ─────────────────────────────────────────────────────
    // Per-side padding: map each supplied side directly to ThemeInputs.padding.*.
    // `auto=true` is not a guard — it does not block or disable explicit side
    // values (an explicit side always wins, spec §4.7). It flips
    // `theme.padding.padding_auto`, consumed at layout time by
    // `layout::compute_layout` (D10, spec §4.7, F-L07-08) to expand an UNSET
    // side enough to contain a continuous axis's edge-tick-label overhang
    // and/or recenter an overflowing axis title — see
    // `ThemePadding::padding_auto`'s own doc for the reader list.
    if let Some(ref pad) = config.padding {
        if let Some(top) = pad.top {
            theme.padding.padding_top = Some(top);
        }
        if let Some(right) = pad.right {
            theme.padding.padding_right = Some(right);
        }
        if let Some(bottom) = pad.bottom {
            theme.padding.padding_bottom = Some(bottom);
        }
        if let Some(left) = pad.left {
            theme.padding.padding_left = Some(left);
        }
        if let Some(auto) = pad.auto {
            theme.padding.padding_auto = auto;
        }
    }

    // ── Legend overrides ──────────────────────────────────────────────────────
    //
    // Only `direction` lands here. `direction`'s theme slot is genuinely the
    // chart-level fallback — `layout_color_legend` resolves the per-channel
    // override against it with `.or(...)`, so writing it unconditionally still
    // leaves per-channel winning. The other three ThemeInputs-backed legend
    // fields (`orient`, `columns`, `title_font_size`) cannot: their theme slots
    // are also where the per-channel values are written, so a write here
    // OVERWROTE the per-channel value that had already landed — the D7 cascade
    // inversion. They now resolve in `apply_legend_cascade_to_theme`, which
    // sees both levels at once. `label_font_size` left too: it wrote the
    // typography slot SHARED with axis labels, so a legend knob resized axis
    // labels; it now fills the legend-own `LegendStyleOpts.label_font_size`
    // slot in `apply_chart_config_to_legend_overrides`.
    if let Some(ref legend_cfg) = config.legend {
        if let Some(dir) = legend_cfg.style.direction.as_deref().and_then(LegendDirection::parse) {
            theme.legend.legend_direction = Some(dir);
        }
    }

    // ── Color scheme overrides ────────────────────────────────────────────────
    if let Some(ref color_cfg) = config.color {
        if let Some(ref scheme) = color_cfg.scheme {
            theme.palette.color_scheme = scheme.clone();
        }
        if let Some(ref seq) = color_cfg.sequential_scheme {
            theme.palette.sequential_scheme = seq.clone();
        }
        if let Some(ref div) = color_cfg.diverging_scheme {
            theme.palette.diverging_scheme = div.clone();
        }
    }

    // ── Axis overrides: the SHARED `axis` key only ────────────────────────────
    // `axis_x`/`axis_y` deliberately do not reach the theme any more (D12,
    // spec §4.9). `ThemeInputs` has no x/y split, so a per-axis section
    // writing it was writing BOTH axes — and, because the writes are
    // last-wins, `axis_y` then silently overwrote `axis_x` on every field
    // they shared. Every field `apply_axis_config_to_theme` still writes now
    // has a per-axis `AxisStyleOverrides` slot as well, which the per-axis
    // sections fill directly (`apply_axis_config_to_axis_input`), so nothing
    // is lost by not writing the theme from them — and the per-axis contract
    // becomes true rather than approximately true.
    //
    // This is the Rust half of retiring Python's `_redistribute_general_axis`:
    // that helper existed to stop a general `configure_axis(...)` from
    // pre-empting a per-axis `.override(x_axis_…)`, but it worked by re-pinning
    // the general value onto the OPPOSITE axis key — which, on a theme-global
    // field, made the general value the LAST writer and therefore the winner.
    // With no global write left for a per-axis section to lose to, the
    // ordering is settled in Rust and the redistribution has nothing to do.
    if let Some(ref axis_cfg) = config.axis {
        apply_axis_config_to_theme(theme, axis_cfg);
    }

    // ── Title overrides ───────────────────────────────────────────────────────
    if let Some(ref title) = config.title {
        if let Some(fs) = title.font_size {
            theme.typography.title_font_size = fs;
        }
        if let Some(ref fw) = title.font_weight {
            theme.typography.title_font_weight = fw.clone();
        }
        if let Some(ref anchor) = title.anchor {
            theme.typography.title_anchor = match anchor.as_str() {
                "middle" => TextAnchor::Middle,
                "end"    => TextAnchor::End,
                _        => TextAnchor::Start,
            };
        }
        if let Some(ref c) = title.color {
            if let Ok(parsed) = color::parse_color(c) {
                theme.colors.title_color = parsed;
            }
        }
        if let Some(o) = title.offset {
            theme.typography.title_offset = o;
        }
        if let Some(fs) = title.subtitle_font_size {
            theme.typography.subtitle_font_size = Some(fs);
        }
        if let Some(ref c) = title.subtitle_color {
            if let Ok(parsed) = color::parse_color(c) {
                theme.colors.subtitle_color = Some(parsed);
            }
        }
    }
}

/// Apply the SHARED `configure_axis(...)` key's styling to the theme.
///
/// Called for `config.axis` only. "Shared" is the whole justification: this
/// key means "every axis", which is exactly what a `ThemeInputs` write says,
/// and it is the only way the settings reach axes the per-axis application
/// never visits — the secondary y axes, which `apply_axis_config_to_axis_input`
/// fills from `axis_y2` alone.
///
/// The fields listed here all ALSO have a per-axis `AxisStyleOverrides` slot
/// that the same shared key fills (via `apply_axis_config_to_axis_input` on
/// both x and y). That is not a double application: the per-axis slot wins for
/// x/y and the theme write is the fallback everything else reads, so the two
/// agree by construction. The grid family and the two show toggles left this
/// function entirely (D4/D12, spec §4.3/§4.9) — for those the theme is the
/// bottom of a precedence chain, not a parallel writer, and it is reached
/// through `AxesInput::apply_show_defaults` instead.
fn apply_axis_config_to_theme(theme: &mut ThemeInputs, axis_cfg: &AxisConfigSpec) {
    let style = &axis_cfg.style;
    if let Some(fs) = style.label_font_size {
        theme.typography.label_font_size = fs;
    }
    if let Some(ref c) = style.label_color {
        if let Ok(parsed) = color::parse_color(c) {
            theme.colors.label_color = parsed;
        }
    }
    if let Some(ts) = style.tick_size {
        theme.sizes.tick_size = ts;
    }
    if let Some(ref c) = style.domain_color {
        if let Ok(parsed) = color::parse_color(c) {
            theme.colors.axis_line_color = parsed;
        }
    }
    if let Some(w) = style.domain_width {
        theme.sizes.axis_line_width = w;
    }
}

/// Resolve the three legend fields whose effective value lives on
/// [`ThemeInputs`] under the documented cascade: per-channel `Legend(...)` >
/// chart-level `configure_legend(...)` > theme (D7 cascade repair, spec §4.4).
///
/// These three share one hazard: the theme slot is the ONLY carrier for the
/// resolved value, so per-channel and chart-level both write the same field.
/// Sequencing two independent writers (per-channel first, then
/// `apply_chart_config_to_theme`) therefore inverted the contract — whoever ran last
/// won, and chart-level ran last. Resolving both levels HERE, with the
/// `or_else` chain `scene_build`'s `legend_zindex` already uses as the
/// chart-level-fallback exemplar, makes the precedence a property of one
/// expression instead of a property of call ordering.
///
/// **Disclosed behavior change:** a chart setting BOTH (e.g.
/// `Color(legend=Legend(orient="bottom"))` plus
/// `configure_legend(orient="right")`) now honors the per-channel value. That
/// is the documented contract asserting itself; the previous chart-level-wins
/// behavior was the bug.
fn apply_legend_cascade_to_theme(
    theme: &mut ThemeInputs,
    per_channel: &prepare::LegendPreparedOverrides,
    config: &ChartConfig,
) {
    let chart_legend = config.legend.as_ref().map(|l| &l.style);
    // `"none"` is a suppression, not a placement, and is consumed before this
    // point on both levels (per-channel: `LegendStyleSpec::suppresses` in
    // `prepare::legend`; chart-level: `chart_config_legend_disabled`, which
    // reads the same predicate, so a raw-dict `orient="none"` that never went
    // through Python's `_resolve_chart_config` conversion still suppresses).
    // An unrecognized token leaves the theme value standing.
    let chart_orient = chart_legend
        .and_then(|l| l.orient.as_deref())
        .and_then(LegendOrient::parse);
    if let Some(orient) = per_channel.orient.or(chart_orient) {
        theme.legend.legend_orient = orient;
    }
    if let Some(cols) = per_channel
        .columns
        .or_else(|| chart_legend.and_then(|l| l.columns))
    {
        theme.legend.legend_columns = Some(cols);
    }
    if let Some(fs) = per_channel
        .title_font_size
        .or_else(|| chart_legend.and_then(|l| l.title_font_size))
    {
        theme.typography.legend_title_font_size = fs;
    }
}

/// Build a [`LegendOverrides`] from a [`prepare::PreparedInputs`].
fn legend_overrides_from_prep(prep: &prepare::PreparedInputs) -> LegendOverrides {
    let lo = &prep.legend_overrides;
    LegendOverrides {
        tick_count:         lo.tick_count,
        gradient_length:    lo.gradient_length,
        gradient_thickness: lo.gradient_thickness,
        direction:          lo.direction,
        values:             lo.values.clone(),
        legend_type:        lo.legend_type.clone(),
        // Per-channel `Legend(symbol_type=...)` (B5); chart-level
        // `configure_legend` fills this only when the per-channel value is absent.
        symbol_type:        lo.symbol_type.clone(),
        tick_min_step:      lo.tick_min_step,
        // 380: the 11 categorical-style fields (B5 unit 3 / 6a orphans —
        // per-channel here; chart-level fills any still None) live nested on
        // `style`. The colorbar path also reads the shared `clip_height` /
        // `label_color` / `label_font_size` from here.
        style: crate::layout::LegendStyleOpts {
            symbol_stroke_width: lo.symbol_stroke_width,
            row_padding:         lo.row_padding,
            column_padding:      lo.column_padding,
            label_limit:         lo.label_limit,
            clip_height:         lo.clip_height,
            padding:             lo.padding,
            title_padding:       lo.title_padding,
            offset:              lo.offset,
            symbol_size:         lo.symbol_size,
            label_color:         lo.label_color.clone(),
            label_font_size:     lo.label_font_size,
        },
    }
}

/// Resolve the effective legend title from a [`prepare::PreparedInputs`].
///
/// Three-way resolution (D13 + v0.15.1), reached by both the single-chart path
/// and `composite_render::capture_leaf_bundle`'s figure-legend seam (design §6)
/// through [`resolve_legend_overrides_and_title`]: mirrors the axis-title contract in
/// `prepare.rs`.
///   - `legend_overrides.title` absent (`None`)   → fall through to field-name default
///   - `legend_overrides.title = Some("")`        → explicit suppress; no text node, no margin
///   - `legend_overrides.title = Some("Foo")`     → render "Foo" verbatim
///
/// Python forwards `""` only when `Legend(title=None)` is explicitly passed,
/// so `Some("")` here is always the caller's intentional suppress sentinel.
fn effective_legend_title(prep: &prepare::PreparedInputs) -> Option<String> {
    match prep.legend_overrides.title.as_deref() {
        Some(s) if s.trim().is_empty() => None, // explicit suppress — no fallback
        Some(s) => Some(s.to_owned()),           // explicit non-empty title
        None => prep.legend_title.clone(),       // absent — fall through to field-name default
    }
}

/// Whether chart-level `configure_legend(...)` fully suppresses the legend
/// (GH #74). Reads [`chart_config::LegendStyleSpec::suppressed_by`] on a
/// one-element chain — the SAME predicate the per-channel color legend reads
/// its whole cascade through and the size/shape aux blocks read their own
/// channel through. So both spellings work here too: `disabled: true` (which
/// Python's `_resolve_chart_config` derives from a fully-merged
/// `orient="none"`) and a raw-dict `orient="none"` that never went through
/// that conversion. Before the cycle-2 fix this asker was left on the bare
/// `disabled` field, so the second spelling parsed to no placement
/// (`LegendOrient::parse` rejects `"none"`) and the theme orient stood — a
/// legend drawn where the caller asked for none.
fn chart_config_legend_disabled(chart_config: &ChartConfig) -> bool {
    chart_config
        .legend
        .as_ref()
        .is_some_and(|l| chart_config::LegendStyleSpec::suppressed_by(&[&l.style]))
}

/// Apply `ChartConfig.legend` fields to a `LegendOverrides`.
///
/// Per-encoding `legend=Legend(...)` overrides (level 2) are already in the
/// `LegendOverrides` built by `legend_overrides_from_prep`. This function
/// fills in the `configure_legend(...)` overrides (level 3) only when the
/// per-encoding value is absent (level 2 wins over level 3).
fn apply_chart_config_to_legend_overrides(
    overrides: &mut LegendOverrides,
    config: &ChartConfig,
) {
    let Some(ref legend) = config.legend else { return };
    let legend = &legend.style;
    if overrides.gradient_length.is_none() {
        overrides.gradient_length = legend.gradient_length;
    }
    if overrides.symbol_type.is_none() {
        overrides.symbol_type = legend.symbol_type.clone();
    }
    // B5 unit 3 orphans: per-channel (level 2) already in `overrides.style`; fill
    // from `configure_legend` (level 3) only where still None, so per-channel wins.
    let style = &mut overrides.style;
    if style.symbol_stroke_width.is_none() {
        style.symbol_stroke_width = legend.symbol_stroke_width;
    }
    if style.row_padding.is_none() {
        style.row_padding = legend.row_padding;
    }
    if style.column_padding.is_none() {
        style.column_padding = legend.column_padding;
    }
    if style.label_limit.is_none() {
        style.label_limit = legend.label_limit;
    }
    if style.clip_height.is_none() {
        style.clip_height = legend.clip_height;
    }
    if overrides.tick_min_step.is_none() {
        overrides.tick_min_step = legend.tick_min_step;
    }
    // B5 unit 6a orphans: per-channel (level 2) already in `overrides.style`; fill
    // from `configure_legend` (level 3) only where still None, so per-channel wins.
    let style = &mut overrides.style;
    if style.symbol_size.is_none() {
        style.symbol_size = legend.symbol_size;
    }
    if style.label_color.is_none() {
        style.label_color = legend.label_color.clone();
    }
    if style.offset.is_none() {
        style.offset = legend.offset;
    }
    if style.padding.is_none() {
        style.padding = legend.padding;
    }
    if style.title_padding.is_none() {
        style.title_padding = legend.title_padding;
    }
    // D7: `configure_legend(label_font_size=)` fills the LEGEND-OWN slot, the
    // same one per-channel `Legend(label_font_size=)` writes and
    // `layout_color_legend` resolves against the theme. It used to write
    // `theme.typography.label_font_size`, which axis tick labels also read —
    // so a legend knob silently resized the axes. Axis label size is now
    // reachable only from `configure_axis(label_font_size=)` and the theme.
    if style.label_font_size.is_none() {
        style.label_font_size = legend.label_font_size;
    }
}

/// Apply `ChartConfig.axis` / `axis_x` / `axis_y` per-axis fields to the
/// `AxisInput`. Only fields absent from the input (higher-precedence per-channel
/// or earlier config) are filled, so the cascade is per-channel > axis_x/axis_y >
/// axis > theme.
///
/// The call ORDER that produces that cascade is
/// [`fill_axis_slots_specific_before_shared`]'s, not this fn's: because the
/// merge is fill-only-if-`None`, `axis_x`/`axis_y` win by running FIRST.
/// (An earlier revision of this line said they "win because they run last",
/// which is the overwrite-path rule, not this one.) Delegates the per-axis
/// style fields to
/// [`apply_axis_style_to_axis_input`] and handles the chart-only `label_format_raw`
/// d3-format key here (the per-channel path uses `label_format` inside the style).
pub(in crate::render) fn apply_axis_config_to_axis_input(
    axis: &mut crate::layout::AxisInput,
    config: Option<&chart_config::AxisConfigSpec>,
) -> Result<(), RenderError> {
    let Some(cfg) = config else { return Ok(()) };
    // Resolve the chart-level d3-format override BEFORE the shared style apply.
    // `effective_label_format()` is raw-first: `label_format_raw` (the chart-level
    // spelling) wins over the style's `label_format`. Applying it first ensures that
    // precedence holds — the style apply below only fills `label_format_override`
    // when still `None`, so it can no longer override the raw-first resolution. Both
    // keys are mutually exclusive at the Python boundary (`AxisConfig.__post_init__`
    // raises), so in practice at most one is set; this keeps the Rust side
    // self-consistent regardless. Fill only when nothing higher-precedence (a
    // per-channel spec) already set the override.
    //
    // `label_format.is_none()` ALONE is not a reliable "unclaimed" test (D8
    // cascade-inversion fix): a per-channel TEMPORAL
    // format is applied EAGERLY to `AxisInput.tick_labels` in
    // `prepare::build_axis_tick_inputs` and threads `label_format = None`
    // back (nothing left to defer) — indistinguishable, by `label_format`
    // alone, from "no per-channel format was ever set". `label_format_claimed`
    // (set in `prepare::build_axis_input` from `resolve_axis_label_format`'s
    // own result) carries that distinction; the mechanized
    // `fill_chart_level_label_format` checks
    // BOTH, so a per-channel-claimed axis is never touched, even when its
    // `label_format` slot happens to read `None`. See
    // `AxisStyleOverrides::label_format_claimed`'s doc for the full account
    // and `render::apply_label_format_to_axis` for why this alone is
    // sufficient (that fn no-ops whenever `label_format` stays `None`).
    axis.overrides.fill_chart_level_label_format(
        cfg.effective_label_format().map(str::to_owned),
        cfg.effective_label_format_type().map(str::to_owned),
    );
    apply_axis_style_to_axis_input(axis, &cfg.style)
}

/// Apply `configure_grid(x=…, y=…)`'s per-axis grid settings to the matching
/// `AxisInput` (D4/F-L07-01, spec §4.3).
///
/// Fill-only, like every other chart-level axis write, so a per-channel
/// `fm.Axis(grid=…)` still wins. Runs BETWEEN the per-axis `axis_x`/`axis_y`
/// section and the shared `axis` section, which is what gives the documented
/// precedence its full ordering: per-channel > `axis_x`/`axis_y` > `grid.x`/
/// `grid.y` > `axis` > `grid`'s flat both-axes shorthand (the theme) — most
/// specific first, and within one specificity level the axis section (which
/// carries the whole axis vocabulary) ahead of the grid section.
///
/// The flat `color`/`width`/`dash`/`opacity` keys are NOT applied here: they
/// are the axis-unspecified shorthand and stay on the theme, which is exactly
/// the fallback `build_grid` consults when an axis sets nothing of its own.
fn apply_grid_config_to_axis_inputs(
    axes: &mut crate::layout::AxesInput,
    config: &ChartConfig,
) {
    let Some(ref grid) = config.grid else { return };
    for (axis, spec) in [(&mut axes.x, grid.x.as_ref()), (&mut axes.y, grid.y.as_ref())] {
        let Some(spec) = spec else { continue };
        let o = &mut axis.overrides;
        if o.show_grid.is_none() {
            o.show_grid = spec.enabled;
        }
        if o.grid_color.is_none() {
            o.grid_color = spec.color.as_deref().and_then(|s| color::parse_color(s).ok());
        }
        if o.grid_width.is_none() {
            o.grid_width = spec.width;
        }
        if o.grid_dash.is_none() {
            o.grid_dash = spec.dash.clone();
        }
        if o.grid_opacity.is_none() {
            o.grid_opacity = spec.opacity;
        }
    }
}

/// Apply an [`AxisStyleSpec`](chart_config::AxisStyleSpec) to one `AxisInput`,
/// filling only fields the input has not already set (so a higher-precedence
/// source — a per-channel spec, or an earlier config layer — always wins).
///
/// Shared by the chart-level `configure_axis` path and the per-channel
/// `EncodingSpec.axis` path (B5 fix). Honored styling keys (grid color/dash/width,
/// label color/font-size, domain color/width) flow into per-axis override fields
/// on `AxisInput` that `build_axis`/`build_grid` consult with a theme fallback, so
/// they render per-axis instead of mutating the shared theme. The actual field
/// merge is the canonical [`prepare::axis_style_fill_from`] (chart-level discipline:
/// `fill_only_if_none = true`).
///
/// Show toggles (`grid`/`domain`/`labels`/`ticks`) route through the SAME merge
/// as of D12 (spec §4.9): they are `Option<bool>` slots on `AxisStyleOverrides`,
/// so the chart-level fill can no longer clobber a per-channel value and the
/// old carve-out (chart-level `grid`/`domain` reaching only the GLOBAL theme,
/// `labels`/`ticks` reaching nothing at all) is gone. The theme remains the
/// bottom of the `grid`/`domain` chain via `AxesInput::apply_show_defaults`.
///
/// The axis TITLE is filled here too, through `AxisInput::fill_chart_level_title`
/// — not via the style merge, because `AxisInput.title` is a resolved string
/// whose `None` is ambiguous between "unset" and "per-channel suppressed"
/// (see that method and `AxisStyleOverrides::title_claimed`).
pub(in crate::render) fn apply_axis_style_to_axis_input(
    axis: &mut crate::layout::AxisInput,
    style: &chart_config::AxisStyleSpec,
) -> Result<(), RenderError> {
    axis.fill_chart_level_title(style.title.as_deref());
    let channel = axis.orient.dimension().channel_token();
    // R3 EXEMPT chain: `channel` here is the axis's PHYSICAL dimension
    // (`AxisDimension::channel_token`), not a channel that traveled through
    // `build_layers`' swap — resolve with `false` explicitly. See the
    // three-chain account on `RenderError::InvalidAxisOrient` and the
    // `chart_level_orient_error_names_resolved_axis_under_flip` test pinning this.
    prepare::axis_style_fill_from(&mut axis.overrides, style, channel, true)
        .map_err(|e| e.resolve(false))
}

/// Re-format tick label strings using a d3-format/strftime string override.
///
/// When `tick_values_override` is set, tick_labels are replaced entirely
/// with formatted versions of the explicit tick values. If a
/// `label_format_override` is also provided the values are formatted using
/// that spec (`format_type`-aware — see [`prepare::apply_tick_format`]);
/// otherwise they are converted to plain decimal strings.
///
/// When only `label_format_override` is set (no explicit tick values): a
/// TIME format (`format_type == Some("time")`, D8/F-L07-05 fix) re-derives
/// the axis's RAW temporal tick values from `scale` and formats those
/// directly via `chrono` — `axis.tick_labels` at this point already holds
/// the DEFAULT spacing-keyed date strings ([`format::format_time`]'s output,
/// via `ScaleKind::tick_labels`), which cannot be re-parsed as epoch-ms, so
/// re-deriving from the scale is required (mirrors
/// `prepare::apply_axis_format_or_thread`'s identical per-channel handling
/// of the same problem). Falls through to the numeric string-reparse path
/// below when the scale isn't temporal or `values.len()` has drifted from
/// `tick_labels.len()` (e.g. `tick_extra`/`tick_min_step` ran between the
/// initial tick build and this call) — same guard
/// `apply_axis_format_or_thread` uses; `apply_tick_format` leaves labels
/// that fail to re-parse as a timestamp unchanged, so this fallback is safe
/// (no corruption), never a silent misformat.
///
/// Otherwise (numeric, or no explicit `format_type`), each existing tick
/// label is parsed as a float and reformatted via the d3-format spec.
/// Non-numeric labels (category names) are passed through unchanged.
fn apply_label_format_to_axis(
    axis: &mut crate::layout::AxisInput,
    scale: &scale_resolve::ScaleKind,
    tick_count: usize,
    reversed: bool,
) {
    let format_type = axis.overrides.label_format_type.clone();
    if let Some(tick_vals) = axis.overrides.tick_values.clone() {
        // Replace tick_labels with formatted versions of the explicit tick_values.
        let numeric_strings: Vec<String> = tick_vals.iter().map(|v| v.to_string()).collect();
        let fmt = axis.overrides.label_format.as_deref();
        axis.tick_labels = prepare::apply_tick_format(numeric_strings, fmt, format_type.as_deref());
        return;
    }
    let Some(fmt_str) = axis.overrides.label_format.clone() else { return };
    if format_type.as_deref() == Some("time") {
        if let Some(mut values) = scale.temporal_tick_values(tick_count) {
            if reversed {
                values.reverse();
            }
            if values.len() == axis.tick_labels.len() {
                axis.tick_labels = values
                    .into_iter()
                    .map(|ms| format::format_time_spec(ms, &fmt_str))
                    .collect();
                return;
            }
        }
    }
    axis.tick_labels = prepare::apply_tick_format(
        std::mem::take(&mut axis.tick_labels),
        Some(&fmt_str),
        format_type.as_deref(),
    );
}

/// Re-pair the axis's tick-placement carrier with its labels after a
/// `tick_values` override replaced them wholesale (`apply_label_format_to_axis`,
/// above).
///
/// An [`AxisInput`](crate::layout::AxisInput) carries at most one
/// index-addressed placement carrier beside `tick_labels` — a continuous
/// `tick_projection` or a categorical `categorical_placement` — and
/// `layout_*_axis` reads it as `carrier[i]` for `i` over `tick_labels`. An
/// explicit `tick_values` list replaces `tick_labels` at an arbitrary length,
/// so BOTH carriers go stale the same way. They are repaired in one function
/// rather than two siblings on purpose: F-L04-03 (GH #67) added the categorical
/// carrier with the identical hazard, and the fact that only the continuous one
/// had a sync is what let a stale-index panic ship (batch-C T7 review).
///
/// - **Continuous**: recompute `major` from the explicit values via the scale.
///   When the scale yields no fractions (degenerate domain), clear the carrier.
///   The minor carrier is dropped in lockstep — empty `value_fractions` implies
///   an axis with no continuum, so its minors are already empty.
/// - **Categorical**: whether there is anything to re-pair depends on whether
///   the override supplied one label per category. `apply_label_format_to_axis`
///   has already replaced `tick_labels` with the override's values, so at this
///   point `tick_labels.len()` IS the override's length.
///   - **Equal count** (`categorical_placement.len() == tick_labels.len()`):
///     the caller supplied exactly one relabel per category, so the pairing
///     between position `i` and category `i` still holds — keep the carrier
///     and let the (relabeled) tick land on its aligned center. For an
///     explicit `range=` scale this restores the pre-F-L04-03 (GH #67) band
///     centers exactly; for a padded no-`range=` scale it is a deliberate
///     improvement in this batch's own direction (relabeled ticks land on the
///     aligned centers rather than uniform slots).
///   - **Mismatched count**: the values have no correspondence to the scale's
///     category order (there is no category `i` for a label the caller never
///     paired one-to-one) — clear the carrier and let layout fall back to
///     uniform slots. That is the pre-F-L04-03 outcome for a mismatched-length
///     override on a categorical axis.
///
/// Either way the rule is this function's original one: never leave layout
/// indexing a stale vec. `AxisInput`'s debug canary
/// (`debug_assert_placement_invariants`) and layout's release-side
/// `paired_band_centers` guard both invariants regardless of which branch
/// this function takes.
fn sync_tick_placement_to_tick_values(
    axis: &mut crate::layout::AxisInput,
    scale: &scale_resolve::ScaleKind,
) {
    let Some(values) = axis.overrides.tick_values.clone() else {
        return;
    };
    if axis
        .categorical_placement
        .as_ref()
        .is_some_and(|placement| placement.len() != axis.tick_labels.len())
    {
        axis.categorical_placement = None;
    }
    if axis.tick_projection.is_none() {
        return;
    }
    let fractions = scale.value_fractions(&values);
    if fractions.is_empty() {
        axis.tick_projection = None;
    } else if let Some(proj) = axis.tick_projection.as_mut() {
        proj.major = fractions;
    }
}

/// Apply `ChartConfig.color.domain` and `color.range` overrides to the
/// resolved color scale (pass 11). Re-exported from `render` (see that
/// module's `config_apply` ladder) so `scene_build` can re-run it after
/// per-panel scale resolution, which re-resolves the scale independently of
/// the `provisional_scales` copy this pipeline patches.
///
/// Per-encoding `Color(scale=...)` overrides (level 2) have already been
/// applied during scale resolution. This function applies `configure_color(...)`
/// overrides (level 3) only when the per-encoding value is absent (level 2
/// wins over level 3).
///
/// Continuous scales: `domain` pins the data extent; `range` builds an evenly
/// spaced gradient from the provided hex stops.
///
/// Categorical scales: `domain` (a list of strings) replaces the resolved
/// category order — entries are drawn in the listed order, categories absent
/// from the list are dropped from the scale, and listed values absent from the
/// data are kept (an empty legend entry, matching positional explicit-domain
/// behavior). A dropped category's *marks still draw*, in the theme mark color
/// with no legend entry; the omitted names are reported as a
/// [`RenderWarning::ColorDomainOmitsCategories`] so that sanctioned degradation
/// is distinguishable from a rendering bug (spec §4.2, amended 2026-08-28).
/// `range` replaces the palette with a heap-allocated `Cow::Owned` slice of
/// parsed colors, all-or-nothing: one unparseable entry discards the whole
/// range and reports [`RenderWarning::ColorRangeParseFailure`], matching the
/// Discretizing arm and `build_color_scale`'s `explicit_string_range` path.
/// This arm cannot use the "silently skip" convention the Continuous arm keeps:
/// a categorical palette is indexed by domain position, so a dropped entry
/// re-points every later category rather than shortening a cycle (spec §4.2,
/// amended 2026-09-02).
///
/// Discretizing scales: `range` replaces the bucket swatches, all-or-nothing —
/// one unparseable entry discards the whole range and reports
/// [`RenderWarning::ColorRangeParseFailure`], matching `bucket_colors` and the
/// categorical explicit-range path. The bucket count is fixed by the scale's own
/// thresholds, so a range of any other length cannot describe the partition; it
/// is left unapplied and reported as a
/// [`RenderWarning::ColorRangeBucketCountMismatch`] naming both counts (spec
/// §4.2, amended 2026-08-28 — never a silent drop). `domain` is inapplicable
/// here: the boundaries come from the scale spec.
///
/// On the Continuous arm alone, invalid color strings are silently skipped and
/// the override is a no-op when fewer than two stops parse — a pre-existing
/// convention kept as-is, since a dropped stop there re-spaces a gradient
/// (`t = i / (n - 1)` over whatever parsed) rather than shifting a
/// position-indexed mapping. The Categorical and Discretizing arms are both
/// all-or-nothing with a `ColorRangeParseFailure`, because both index by
/// position.
///
/// Returns the warnings the override produced, for the caller to fold into its
/// accumulator. Empty for every chart with no `configure_color(...)`.
#[must_use]
pub(in crate::render) fn apply_color_config_to_color_scale(
    color_scale: &mut Option<scale_resolve::ColorScale>,
    cfg: &chart_config::ColorConfigSpec,
) -> Vec<RenderWarning> {
    let mut warnings = Vec::new();
    let Some(ref mut scale) = color_scale else { return warnings };
    match scale {
        scale_resolve::ColorScale::Continuous { ref mut domain, ref mut scheme, .. } => {
            if let Some(ref d) = cfg.domain {
                let floats: Vec<f64> = d.iter().filter_map(|v| v.as_f64()).collect();
                if floats.len() == 2 {
                    *domain = (floats[0], floats[1]);
                }
            }
            if let Some(ref r) = cfg.range {
                if r.len() >= 2 {
                    let parsed: Vec<color::Color> = r
                        .iter()
                        .filter_map(|s| color::parse_color(s).ok())
                        .collect();
                    if parsed.len() >= 2 {
                        // Build a gradient from the parsed stops, evenly spaced.
                        let stops: Vec<(f64, color::Color)> = parsed
                            .iter()
                            .enumerate()
                            .map(|(i, &c)| {
                                let t = i as f64 / (parsed.len() - 1) as f64;
                                (t, c)
                            })
                            .collect();
                        *scheme = color::ContinuousScheme::Gradient(stops);
                    }
                }
            }
        }
        scale_resolve::ColorScale::Categorical { ref mut domain, ref mut palette } => {
            if let Some(ref d) = cfg.domain {
                let categories: Vec<String> =
                    d.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect();
                if !categories.is_empty() {
                    // Categories the override leaves unlisted keep rendering —
                    // in the theme mark color, with no legend entry (spec §4.2,
                    // amended 2026-08-28). Sanctioned, but indistinguishable
                    // from a rendering bug on sight, so name them.
                    let omitted: Vec<String> = domain
                        .iter()
                        .filter(|c| !categories.contains(c))
                        .cloned()
                        .collect();
                    if !omitted.is_empty() {
                        warnings.push(RenderWarning::ColorDomainOmitsCategories {
                            categories: omitted,
                        });
                    }
                    *domain = categories;
                }
            }
            if let Some(ref range) = cfg.range {
                // All-or-nothing, mirroring the Discretizing arm below and
                // `build_color_scale`'s `explicit_string_range` path. A
                // categorical palette is indexed by DOMAIN POSITION
                // (`ColorScale::lookup` → `palette[i % palette.len()]`), so
                // dropping an unparseable entry does not merely shorten a
                // cycle: it re-points every category after the dropped one.
                // `range=["red", "notacolor", "blue"]` over domain `[a, b, c]`
                // silently rendered `a=red, b=blue, c=red` — a wrong mapping,
                // not a degraded one. One bad entry now discards the whole
                // range and reports the offending string, leaving the resolved
                // palette in place (spec §4.2, amended 2026-09-02).
                let parsed: Result<Vec<color::Color>, &String> = range
                    .iter()
                    .map(|s| color::parse_color(s).map_err(|_| s))
                    .collect();
                match parsed {
                    Ok(colors) if !colors.is_empty() => {
                        *palette = std::borrow::Cow::Owned(colors);
                    }
                    // An empty `range=[]` describes no palette at all; left
                    // unapplied with no warning, as before.
                    Ok(_) => {}
                    Err(entry) => warnings.push(RenderWarning::ColorRangeParseFailure {
                        entry: entry.clone(),
                    }),
                }
            }
        }
        scale_resolve::ColorScale::Discretizing(ref mut buckets) => {
            if let Some(ref range) = cfg.range {
                // All-or-nothing, mirroring every other explicit-range parse in
                // the crate (`bucket_colors`, the categorical explicit-range
                // path). Dropping unparseable entries first would make the
                // count check see a post-filter length: a 4-entry range with one
                // bad entry would "fit" a 3-bucket scale and silently repaint it
                // with a shifted mapping, and a 3-entry range with one bad entry
                // would report a count mismatch for what is really a parse
                // failure.
                let parsed: Result<Vec<color::Color>, &String> = range
                    .iter()
                    .map(|s| color::parse_color(s).map_err(|_| s))
                    .collect();
                match parsed {
                    Ok(colors) => {
                        if let Err(mismatch) = buckets.set_colors(colors) {
                            warnings.push(RenderWarning::ColorRangeBucketCountMismatch {
                                expected: mismatch.expected as u32,
                                received: mismatch.received as u32,
                            });
                        }
                    }
                    Err(entry) => warnings.push(RenderWarning::ColorRangeParseFailure {
                        entry: entry.clone(),
                    }),
                }
            }
        }
    }
    warnings
}

/// Re-derive the color legend's entry labels from the categorical scale domain.
///
/// `prepare_render_inputs` builds the entries from the domain as resolved, but
/// `configure_color(domain=[…])` reorders/subsets that domain afterwards
/// ([`apply_color_config_to_color_scale`]), so without this the swatch colors
/// would follow the new order while the labels kept the old one. Entry symbols
/// are preserved (a category that survives keeps its symbol; a newly listed one
/// takes the first entry's).
///
/// A no-op — and therefore byte-identical — whenever the domain still matches
/// the entries, which is every chart that does not set `configure_color(domain=)`
/// on a categorical color scale.
fn resync_categorical_legend_entries(
    entries: &mut Vec<crate::layout::LegendEntry>,
    color_scale: Option<&scale_resolve::ColorScale>,
) {
    let Some(domain) = color_scale.and_then(scale_resolve::ColorScale::categorical_domain) else {
        return;
    };
    let Some(first) = entries.first() else { return };
    if entries.len() == domain.len() && entries.iter().zip(domain).all(|(e, d)| &e.label == d) {
        return;
    }
    let default_symbol = first.symbol;
    *entries = domain
        .iter()
        .map(|label| crate::layout::LegendEntry {
            label: label.clone(),
            symbol: entries
                .iter()
                .find(|e| &e.label == label)
                .map_or(default_symbol, |e| e.symbol),
        })
        .collect();
}

/// Apply a per-channel `Legend(values=[...])` to a CATEGORICAL legend: keep
/// only the named entries, in the order named (D6/F-L04-05, spec §4.4).
///
/// Mirrors the colorbar arm's long-standing `values` semantics (an explicit
/// list replaces the computed tick labels) for the arm that never had them.
/// `values` absent → entries untouched, byte-identically. A colorbar chart has
/// no entries, so this is a no-op there and the colorbar path keeps reading
/// `values` itself, unchanged.
///
/// A name matching no entry cannot be drawn: there is no category behind it,
/// so no color-scale slot and no swatch. Per the batch's sanctioned-degradation
/// rule it is skipped and reported via [`RenderWarning::LegendValuesUnknown`],
/// never silently ignored and never invented as an empty swatch.
fn apply_legend_values_to_entries(
    entries: &mut Vec<crate::layout::LegendEntry>,
    values: Option<&[String]>,
    warnings: &mut Vec<RenderWarning>,
) {
    let Some(values) = values else { return };
    if entries.is_empty() {
        return;
    }
    let mut kept = Vec::with_capacity(values.len());
    let mut unknown = Vec::new();
    for wanted in values {
        match entries.iter().find(|e| &e.label == wanted) {
            Some(entry) => kept.push(entry.clone()),
            None => unknown.push(wanted.clone()),
        }
    }
    if !unknown.is_empty() {
        warnings.push(RenderWarning::LegendValuesUnknown { values: unknown });
    }
    *entries = kept;
}

// ── Pre-prepare gate: raw format-spec validation ───────────────────────────

/// Validate every raw d3-format/strftime spec string reachable from `spec`
/// and `chart_config` (NF-B1 residual, spec 2026-09-02 §4.5): a malformed
/// spec is refused with [`RenderError::InvalidFormatSpec`] before any
/// transform/layout work begins, rather than reaching `format.rs`'s
/// per-value tokenizer where trailing garbage from a typo'd preset name
/// (e.g. `"curency"`) was previously silently discarded, corrupting
/// rendered text with control characters (the tokenizer's `c` type char).
///
/// Every surface that accepts a raw format string funnels through here
/// exactly once per render — not once per tick/legend-entry:
/// `chart_config`'s `axis`/`axis_x`/`axis_y`/`axis_y2`
/// (`AxisConfigSpec::effective_label_format`/`_type`) and `legend.format`/
/// `format_type`, and every layer's (plus the chart-level shorthand)
/// `EncodingSpec.format`/`.format_type` for every channel whose `format` has
/// a real consumer today (x, y — the positional/axis-tick shorthand; text,
/// tooltip, tooltip_fields — draw.rs/marks/text.rs's label/tooltip
/// formatting; color, size, shape, opacity, x2, y2 are ALSO swept, even
/// though their own `.format` field has no consumer yet, since validating
/// an unused field is free and a future consumer then inherits validation
/// for free too), plus its nested `.axis.label_format`/`_type` (x/y only —
/// `AxisStyleSpec` has no consumer on any other channel) and
/// `.legend.format`/`format_type` (every channel's legend, chart-level and
/// per-channel). A spec classified as a TIME pattern is validated against
/// the `chrono` strftime grammar instead of the d3 one
/// ([`format::validate_strftime_spec`]) — NOT exempted from validation.
/// (Correction: an earlier revision of this
/// doc/code claimed `chrono` is "separately lenient" and skipped validating
/// time patterns entirely. That premise was false — `format::format_time_spec`
/// panics on a malformed pattern (`chrono`'s `DelayedFormat` `Display`
/// returns `Err`, and the blanket `ToString` impl `.expect()`s it never
/// does), so every raw-accepting surface's default `format_type: None` case
/// — which `is_time_format_spec` treats as time whenever the spec contains
/// `%` — was one typo away from a `PanicException` crossing the PyO3
/// boundary instead of the typed refusal this fn exists to give.)
///
/// **Two time-classification rules, matched to the two real runtime rules**
/// (correction — a single shared LOOSE rule here
/// falsely refused a valid raw d3 percent spec on the chart-level axis
/// surface): `check_loose` uses [`format::is_time_format_spec`] (the `%`
/// -containment heuristic) for the ONE surface that actually auto-detects
/// time that way at runtime — per-channel `x`/`y`'s `.format`/`.axis
/// .label_format`, consumed by `prepare::apply_axis_format_or_thread`,
/// which uses the identical heuristic. Every other surface's real consumer
/// (`render::apply_label_format_to_axis` for chart-level `AxisConfigSpec`;
/// `prepare::legend`'s `format_value_with_spec` for every `.legend.format`;
/// `draw.rs`/`marks/text.rs` for `text`/`tooltip`/`tooltip_fields`) checks
/// `format_type == Some("time")` STRICTLY, with no `%`-based auto-detection
/// — `check_strict` matches that, so this pre-pass never refuses a spec its
/// own real consumer would have accepted as numeric.
///
/// `check_loose`'s auto-detected case (no explicit `format_type`, spec
/// contains `%`) is genuinely ambiguous at validation time on its own: at
/// runtime, `apply_axis_format_or_thread` only actually commits to the
/// `chrono` strftime grammar when the RESOLVED scale turns out temporal
/// (`scale.temporal_tick_values` returns `Some`); on a non-temporal scale it
/// falls through to `apply_tick_format`'s own STRICT check, which sees
/// `format_type: None` and formats as NUMERIC d3 instead — so a spec valid
/// as d3 but not as strftime (e.g. `"*>8.1%"`, a real percent format) must
/// not be refused on a channel that will never resolve temporal. This fn
/// runs before scale resolution, but each channel's *declared* type
/// (`EncodingSpec.type_`, e.g. the `:T`/`:Q` shorthand suffix) is known
/// early and is exactly the signal `SpecDataType::Temporal` (`scale_resolve
/// /positional.rs`) uses to choose `ScaleKind::Time` in the first place —
/// so `check_loose` uses `type_` to resolve the ambiguity precisely for the
/// declared-type case, falling back to a tolerant "valid under either
/// grammar" check only when `type_` is unset (fully auto-inferred from data,
/// the one case this fn truly cannot know early).
pub(in crate::render) fn validate_chart_format_specs(spec: &ChartSpec, chart_config: &ChartConfig) -> Result<(), RenderError> {
    use crate::spec::encoding::DataType;

    fn check_with(f: &str, is_time: bool) -> Result<(), RenderError> {
        let result =
            if is_time { format::validate_strftime_spec(f) } else { format::validate_d3_format_spec(f) };
        result.map_err(|reason| RenderError::InvalidFormatSpec { spec: f.to_string(), reason })
    }

    /// Matches `prepare::apply_axis_format_or_thread`'s real per-channel
    /// axis time-detection rule (the `%`-containment heuristic), disambiguated
    /// by the channel's declared `encoding_type` where the auto-detected case
    /// would otherwise be ambiguous — see this fn's doc.
    fn check_loose(
        fmt: Option<&str>,
        format_type: Option<&str>,
        encoding_type: Option<DataType>,
    ) -> Result<(), RenderError> {
        let Some(f) = fmt else { return Ok(()) };
        if format_type == Some("time") {
            return check_with(f, true);
        }
        if !format::is_time_format_spec(f, format_type) {
            return check_with(f, false);
        }
        // Auto-detected via '%'-containment with no explicit format_type.
        match encoding_type {
            Some(DataType::Temporal) => check_with(f, true),
            Some(DataType::Quantitative | DataType::Nominal | DataType::Ordinal) => check_with(f, false),
            None => {
                // Fully auto-inferred type: genuinely ambiguous early.
                // Tolerate whichever grammar the spec actually validates
                // under, matching whichever one its eventually-resolved
                // scale will really use.
                if format::validate_strftime_spec(f).is_ok() || format::validate_d3_format_spec(f).is_ok()
                {
                    Ok(())
                } else {
                    check_with(f, true)
                }
            }
        }
    }

    /// Matches every OTHER real consumer's rule: `format_type == Some("time")`
    /// exactly, never auto-detected from the spec's own content.
    fn check_strict(fmt: Option<&str>, format_type: Option<&str>) -> Result<(), RenderError> {
        let Some(f) = fmt else { return Ok(()) };
        check_with(f, format_type == Some("time"))
    }

    fn check_axis_cfg(cfg: Option<&AxisConfigSpec>) -> Result<(), RenderError> {
        let Some(cfg) = cfg else { return Ok(()) };
        let fmt = cfg.effective_label_format();
        let format_type = cfg.effective_label_format_type();
        let Some(f) = fmt else { return Ok(()) };
        check_with(f, format_type == Some("time")).map_err(|err| {
            // When the STRICT check above failed
            // as d3 (format_type wasn't explicitly "time") but the SAME
            // string IS a valid strftime pattern, the true problem is NOT
            // "your d3 spec is malformed" — chart-level axis config
            // (`configure_axis` / `AxisConfig.label_format_raw`) has NO time
            // spelling at all; only a time preset name (`label_format=
            // "date_iso"`, resolved by Python before this fn ever runs) or
            // the per-channel `fm.Axis(label_format=...)` surface accept a
            // custom date/time pattern. Re-diagnose the message so it names
            // the real cause instead of restating the (also technically
            // true, but misleading) d3-grammar complaint.
            //
            // Correction (recurring
            // at this exact site): `validate_strftime_spec` alone is NOT a
            // "looks like a date pattern" test — `chrono`'s `StrftimeItems`
            // parses ANY `%`-free literal text successfully (there is
            // nothing to parse), so the bare `.is_ok()` check re-diagnosed
            // EVERY %-free typo (the batch's own headline repro, `"curency"`)
            // as "a valid date/time pattern". A string only counts as a
            // strftime CANDIDATE when it actually contains a `%` escape —
            // the one character every real specifier requires and no
            // literal-only string ever has — so `"curency"` (no `%`) now
            // keeps the d3-grammar message unconditionally, while `"%b %d"`
            // (has a `%`, and parses) still gets the re-diagnosis.
            if format_type != Some("time") && f.contains('%') && format::validate_strftime_spec(f).is_ok() {
                RenderError::InvalidFormatSpec {
                    spec: f.to_string(),
                    reason: format!(
                        "{f:?} is a valid date/time pattern, but this chart-level axis \
                         surface (configure_axis / AxisConfig.label_format_raw) only \
                         accepts numeric d3-format specs — use a time preset name \
                         (e.g. label_format=\"date_iso\") or the per-channel \
                         fm.Axis(label_format=...) surface for a custom date/time pattern"
                    ),
                }
            } else {
                err
            }
        })
    }

    fn check_legend_style(style: &chart_config::LegendStyleSpec) -> Result<(), RenderError> {
        check_strict(style.format.as_deref(), style.format_type.as_deref())
    }

    /// `format_is_strict`: whether THIS channel's own `.format`/`.format_type`
    /// (not the nested `.axis`/`.legend`, which always use their own fixed
    /// rule) is consumed by a strict-only reader (`text`/`tooltip`/
    /// `tooltip_fields` — `draw.rs`/`marks/text.rs`) versus the loose,
    /// auto-detecting `x`/`y` axis-shorthand reader. Channels with no
    /// consumer for their own `.format` today (color/size/shape/opacity/
    /// x2/y2) are swept with the loose rule — harmless either way since
    /// nothing reads it yet, and loose is the more permissive default.
    fn check_encoding(
        e: Option<&crate::spec::encoding::EncodingSpec>,
        format_is_strict: bool,
    ) -> Result<(), RenderError> {
        let Some(e) = e else { return Ok(()) };
        if format_is_strict {
            check_strict(e.format.as_deref(), e.format_type.as_deref())?;
        } else {
            check_loose(e.format.as_deref(), e.format_type.as_deref(), e.type_)?;
        }
        if let Some(axis) = e.axis.as_deref() {
            check_loose(axis.label_format.as_deref(), axis.label_format_type.as_deref(), e.type_)?;
        }
        if let Some(legend) = e.legend.as_deref() {
            check_legend_style(legend)?;
        }
        Ok(())
    }

    fn check_encoding_set(enc: &crate::spec::encoding::Encoding) -> Result<(), RenderError> {
        check_encoding(enc.x.as_ref(), false)?;
        check_encoding(enc.y.as_ref(), false)?;
        check_encoding(enc.color.as_ref(), false)?;
        check_encoding(enc.size.as_ref(), false)?;
        check_encoding(enc.shape.as_ref(), false)?;
        check_encoding(enc.opacity.as_ref(), false)?;
        check_encoding(enc.x2.as_ref(), false)?;
        check_encoding(enc.y2.as_ref(), false)?;
        check_encoding(enc.text.as_ref(), true)?;
        check_encoding(enc.tooltip.as_ref(), true)?;
        if let Some(fields) = enc.tooltip_fields.as_ref() {
            for f in fields {
                check_encoding(Some(f), true)?;
            }
        }
        Ok(())
    }

    check_axis_cfg(chart_config.axis.as_ref())?;
    check_axis_cfg(chart_config.axis_x.as_ref())?;
    check_axis_cfg(chart_config.axis_y.as_ref())?;
    check_axis_cfg(chart_config.axis_y2.as_ref())?;
    if let Some(legend) = chart_config.legend.as_ref() {
        check_legend_style(&legend.style)?;
    }

    check_encoding_set(&spec.encoding)?;
    if let Some(layers) = spec.layers.as_ref() {
        for layer in layers {
            check_encoding_set(&layer.encoding)?;
        }
    }
    Ok(())
}

