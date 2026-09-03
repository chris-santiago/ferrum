//! Composite render core — Phase B composite-render-unification (Task 5b).
//!
//! Passes 2 (layout) and 3 (scene) of the three-pass composite renderer (design
//! spec §5). Consumes a validated [`CompositeNode`] tree plus per-leaf render
//! inputs and produces **one** [`SceneGraph`] sized to the composite viewport —
//! the single Rust render call per composition that the Python cutovers
//! (Tasks 6–9) target through the PyO3 entries (Task 5c).
//!
//! # The three passes
//!
//! 1. **Resolve** (Task 4, [`super::composite`]): one shared domain per shared
//!    positional channel across the tree's leaves, yielding a per-leaf
//!    [`LeafScaleContext`].
//! 2. **Layout** (here): each leaf renders standalone (with its resolved-domain
//!    context threaded through the D4b seam), then the tree is walked to place
//!    every leaf scene — hconcat/vconcat (linear + spacing), grid (row-major with
//!    F20 ratio math absorbed from the deleted `grid_compose.rs`), wrap (`ncols`),
//!    and overlay (children share one region, z-order = child order).
//!    An overlay group's leaves additionally share one PLOT rect: a pre-pass
//!    ([`impose_shared_overlay_rects`]) intersects their natural plot regions
//!    and threads the result back through the same per-leaf context, so every
//!    layer's layout products describe one geometry and the merge below can
//!    drop the duplicate chrome (GH #89A).
//! 3. **Scene** (here): the placed leaf scenes are merged into one graph —
//!    panels renumbered `0..N` flat in pre-order (D4c), clip ids uniquified,
//!    figure chrome injected via [`super::figure_chrome::title_nodes`].
//!
//! # The layout-scale placement contract (amended-D4a)
//!
//! A leaf placed by pure translation with a slot matching its native size bakes
//! the offset straight into its panel geometry, leaving `Panel.layout_scale` at
//! identity — the same representation every flat and faceted panel uses today,
//! so a composite's non-ratio panels carry final `plot_area` rects. A
//! **ratio-fitted** grid cell instead keeps its content at native coordinates and
//! carries a **non-identity** `Panel.layout_scale`; the walkers (`walk_svg`,
//! WASM `scene_load`) apply that transform — the scene pass must **not** pre-bake
//! it (amended-D4a: pre-baking would double-apply under the WASM loader). Nested
//! non-identity panels under a pure-translate placement compose the translate
//! into their existing `layout_scale`.
//!
//! Non-panel nodes (figure title, legend, decorations, raw fragments) have no
//! per-node layout-scale slot, so they are translate-baked into final
//! coordinates (W4). The scale factor of a ratio cell is not applied to any
//! non-panel node it might carry — an approximation that is unreachable in the
//! real ratio producers (JointChart/ClusterMap marginals carry no legends),
//! consistent with amended-D4a's documented scalar-approximation gaps.
//!
//! # `hole` cells (Task 8a, sized holes Task 10-rust)
//!
//! A `grid`/`wrap` child may be [`CompositeNode::Hole`] — a placeholder cell
//! (JointChart's empty 2x2 corner, RepeatChart's `corner=True`). [`build_placed`]
//! renders it as a zero-size, empty [`Placed`] subtree: it consumes no leaf scene,
//! claims no panel ids, and carries no label. `plan_grid`/`plan_wrap` size each
//! row/column from the *max* native extent of its cells, so a hole's zero extent
//! never shrinks a lane that also holds a real cell — the hole's own slot is
//! simply left empty, and ratio/spacing math for the tree's other cells is
//! unaffected by its presence.
//!
//! A `hole` directly under `hconcat`/`vconcat` may additionally carry `width`/
//! `height` (spec §4, validated: both-or-neither under a linear layout — see
//! [`crate::spec::composite::CompositeSpecError::HoleSizeRequired`]). Faithful
//! to the legacy string-compositor's behavior for an empty-data child (a blank
//! SVG at the child's own viewport size, siblings unaffected), [`build_placed`]
//! reserves exactly that `width`x`height` of blank space in the flow when its
//! immediate parent is `hconcat`/`vconcat` — no panel, no leaf binding, no
//! chrome, normal spacing on both sides (the same placement math any other
//! child gets). Grid/wrap holes ignore these fields (cell math already governs
//! their slot, per the paragraph above); the size only takes effect under a
//! linear parent.

use std::collections::HashMap;

use arrow::record_batch::RecordBatch;
use ferrum_scene::{LayoutScale, MarkBatch, Panel, Rect, SceneGraph, SceneNode};

use crate::layout::facet::ResolveMode;
use crate::layout::legend::{
    layout_aux_legends, layout_color_legend, LEGEND_OUTER_PAD, LEGEND_PLOT_GAP,
};
use crate::layout::text_metrics::TextMetrics;
use crate::layout::{
    AuxLegendInput, ColorbarInput, LegendDirection, LegendEntry, LegendLayout, LegendOrient,
    LegendOverrides, Rect as LayoutRect, ThemeInputs, Viewport,
};
use crate::spec::chart::ChartSpec;

use super::chart_config::ChartConfig;
use super::composite::{
    effective_share, flatten_leaf_specs, resolve_composite_scales, CompositeResolveError,
    LeafResolveInput,
};
use super::config::RenderConfig;
use super::figure_chrome::{title_nodes, ChromeAnchor, FigureChrome, DEFAULT_CHROME_INSET};
use super::scale_resolve::{ColorScale, LeafScaleContext};
use super::svg::uniquify_clip_ids;
use super::{prepare, scene_build, RenderError, RenderWarning};
use crate::spec::composite::{CompositeLayout, CompositeNode};

/// Default pixel gap between adjacent cells, matching the deleted string
/// compositor's `spacing = 10.0` default (`render/compositor.rs`, removed in
/// Task 10 stage 3) so composites stay visually equivalent to the
/// string-compositor path they replaced.
const DEFAULT_SPACING: f64 = 10.0;

/// Slack for the "slot matches native" comparison — mirrors the deleted
/// `grid_compose.rs`'s `near_eq` (`1e-6`), so a ratio cell whose allocation equals its native size
/// bakes its offset (identity `layout_scale`) exactly as the string compositor
/// keeps such a cell on the lightweight `<g translate>` path.
const SLOT_MATCH_EPS: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// Per-leaf render inputs the composite core renders standalone before placement.
///
/// One per tree leaf, in the pre-order [`flatten_leaf_specs`] produces (leaf *i*
/// here is leaf *i* of the tree). `viewport`/`config`/`chart_config`/`theme` are
/// per-leaf so a composition of differently-sized or differently-themed children
/// renders each on its own terms; the composite entry (Task 5c) populates them
/// from the render call.
pub(crate) struct CompositeLeafInput<'a> {
    pub(crate) spec: &'a ChartSpec,
    pub(crate) batch: &'a RecordBatch,
    pub(crate) theme: &'a ThemeInputs,
    pub(crate) viewport: Viewport,
    pub(crate) config: &'a RenderConfig,
    pub(crate) chart_config: &'a ChartConfig,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A failure rendering a composite tree. Every variant names the offending node
/// kind and/or leaf index so a Python `ValueError` pinpoints it, matching the
/// `CompositeSpecError`/`CompositeResolveError`/`RenderError` idiom. No
/// warning-fallbacks.
#[derive(Debug)]
pub(crate) enum CompositeRenderError {
    /// The `leaves` slice length did not match the tree's leaf count — the
    /// caller assembled per-leaf inputs out of step with the tree.
    LeafCountMismatch { expected: usize, got: usize },
    /// The resolve pass (pass 1) failed unioning shared positional domains.
    Resolve(CompositeResolveError),
    /// Rendering one leaf standalone failed.
    LeafRender {
        kind: &'static str,
        index: usize,
        source: RenderError,
    },
    /// A `leaf` node's `data` index did not select a valid entry in the
    /// caller's Arrow payload list — the PyO3 entry boundary check (Task 5c),
    /// surfaced before any leaf is rendered so a malformed tree fails fast
    /// with the offending leaf pinpointed, never a Rust-side panic.
    LeafDataIndexOutOfBounds {
        kind: &'static str,
        index: usize,
        data: usize,
        payload_count: usize,
    },
    /// The composite tree root's `config` slot (spec §6 root-only figure
    /// chrome) failed to parse as `{"left_inset": f64?, "right_inset": f64?,
    /// "anchor": str?}` — an unrecognized key, wrong-typed value, or an
    /// `anchor` outside `start`/`middle`/`end`. No warn-fallback: a malformed
    /// root `config` is a typed error naming the tree's root kind.
    RootChromeConfigInvalid { kind: &'static str, message: String },
}

impl std::fmt::Display for CompositeRenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeafCountMismatch { expected, got } => write!(
                f,
                "composite leaf count mismatch: tree has {expected} leaves, got {got} leaf inputs"
            ),
            Self::Resolve(e) => write!(f, "composite scale resolution failed: {e}"),
            Self::LeafRender {
                kind,
                index,
                source,
            } => {
                write!(
                    f,
                    "failed to render composite {kind} leaf #{index}: {source}"
                )
            }
            Self::LeafDataIndexOutOfBounds {
                kind,
                index,
                data,
                payload_count,
            } => write!(
                f,
                "composite {kind} leaf #{index}: data index {data} out of bounds \
                 ({payload_count} payload(s) provided)"
            ),
            Self::RootChromeConfigInvalid { kind, message } => {
                write!(f, "{kind}: invalid root 'config': {message}")
            }
        }
    }
}

impl std::error::Error for CompositeRenderError {}

impl From<CompositeResolveError> for CompositeRenderError {
    fn from(e: CompositeResolveError) -> Self {
        CompositeRenderError::Resolve(e)
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Render a validated composite tree into one [`SceneGraph`], plus every
/// leaf's [`RenderWarning`]s aggregated in leaf pre-order.
///
/// `leaves` must be the tree's leaves in pre-order (the order
/// [`flatten_leaf_specs`] produces). Precondition: `tree` has passed
/// [`CompositeNode::validate`] (the sole Python→Rust construction path enforces
/// this; the PyO3 entry in Task 5c validates before calling).
///
/// `call_theme` is the theme passed to the *render call itself* (the value the
/// PyO3 entry decodes before resolving any per-leaf override, `binding.rs`'s
/// `t` in `decode_composite_inputs`) — distinct from any individual leaf's
/// (possibly overridden) `CompositeLeafInput::theme`. It styles per-child
/// composite labels ([`apply_child_label`]) only: a label belongs to the
/// composition, not to any one (possibly heterogeneous) leaf theme, so it
/// always follows the call-level theme even when leaves render under
/// different per-leaf themes.
///
/// Reachable from the `render_composite_svg` / `render_composite_interactive`
/// PyO3 entries (`render/binding.rs`) — mirrors `render_scene_json`'s tuple
/// return (this crate's idiom for "graph + side channel" outputs) rather than
/// introducing a bespoke output struct for a single extra field.
pub(crate) fn render_composite_scene(
    tree: &CompositeNode,
    leaves: &[CompositeLeafInput<'_>],
    call_theme: &ThemeInputs,
) -> Result<(SceneGraph, Vec<RenderWarning>), CompositeRenderError> {
    let n = flatten_leaf_specs(tree).len();
    if leaves.len() != n {
        return Err(CompositeRenderError::LeafCountMismatch {
            expected: n,
            got: leaves.len(),
        });
    }

    // Pass 1 (resolve) needs each leaf's transformed data + rendering encoding.
    // Prepare every leaf once with no context (context only seeds scale
    // resolution, never the transformed data), then union shared domains.
    let mut prepared: Vec<prepare::PreparedInputs> = Vec::with_capacity(n);
    for (i, leaf) in leaves.iter().enumerate() {
        let prep = prepare::prepare_render_inputs(leaf.spec, leaf.batch, leaf.theme, None)
            .map_err(|source| CompositeRenderError::LeafRender {
                kind: "leaf",
                index: i,
                source,
            })?;
        prepared.push(prep);
    }
    let resolve_inputs: Vec<LeafResolveInput<'_>> = leaves
        .iter()
        .zip(&prepared)
        .map(|(leaf, prep)| LeafResolveInput {
            spec: leaf.spec,
            encoding: &prep.layers[0].encoding,
            final_batch: prep.final_batch(),
            transform_outputs: &prep.transform_outputs,
        })
        .collect();
    let mut contexts = resolve_composite_scales(tree, &resolve_inputs)?;
    drop(resolve_inputs);
    // `prepared` stays alive through the line/ribbon exemption pre-pass below
    // (spec-review cycle-2 fix), which needs each leaf's REAL resolved
    // layers/marks (`prepared[i].layers`, from `LayerPrepared` — the same
    // fallback-to-top-level-mark-for-flat-specs / real-sub-layers-for-a-
    // desugared-mark logic `prepare_render_inputs` already runs) and each
    // leaf's own resolved color-scale kind (`prepared[i].provisional_scales.color`).
    // Dropped explicitly right after that pre-pass runs, same as before.

    // Figure-legend planning (design §5, GH #16 shared-legend Task 3): mark each
    // participating leaf's per-channel legend suppression and identify the
    // composite nodes that emit a single figure-level legend band. Writes the
    // suppression flags into `contexts` BEFORE pass 2 so a suppressed leaf's
    // layout reserves no gutter and draws no per-panel legend for that channel,
    // while its prepared legend bundle is still captured (below) as the figure
    // legend's content. A composite with all-independent legend resolution
    // produces an empty plan — no suppression, no band — so its output is
    // byte-identical to today (design §7 byte-stability invariant).
    let band_plan = plan_legend_bands(tree, &mut contexts);

    // Overlay shared-rect planning (GH #89A): for a `layout: Overlay`
    // composite node whose DIRECT children are ALL leaves (LayerChart's actual
    // shape — every layer is one leaf), `overlay_groups[i] = Some(group_start)`
    // for every non-primary child leaf index `i`, naming the group's first
    // (`group_start`) leaf. [`impose_shared_overlay_rects`] then lays every
    // group member out once to learn its natural plot region, intersects them,
    // and writes the result into each member's `LeafScaleContext` so the real
    // render below computes EVERY layout product against that one rect — and
    // PRUNES `overlay_groups` for any group it could not equalize. The merge
    // seam reads that same (pruned) map to decide chrome suppression, so the
    // group that shares a rect is exactly the group that dedups its chrome.
    // A cleared entry (a mixed leaf/composite Overlay, a singleton Overlay, or
    // a group whose intersection degenerated) leaves that leaf laying out on
    // its own terms and keeping its chrome.
    //
    // `warnings` is opened here rather than with the render loop below because
    // the pre-pass is the first stage that can produce one (a group whose
    // gutters diverge); leaf warnings still follow in leaf pre-order after it.
    let mut warnings: Vec<RenderWarning> = Vec::new();
    let mut overlay_groups = plan_overlay_groups(tree, n);
    impose_shared_overlay_rects(leaves, &mut overlay_groups, &mut contexts, &mut warnings);

    // Line/ribbon inert-continuous-color exemption for all-leaf Overlay
    // groups (T5b static-composite fix, spec §4.0's second bullet, spec-review
    // cycle-2 corrections): a sibling leaf in the SAME group that binds the
    // SAME color FIELD to a mark other than line/ribbon, via a Numeric-keyed
    // scale, genuinely renders that mapping, so the line/ribbon leaf should
    // not warn (or lose its colorbar) as if it were the only consumer. Field-
    // and scale-keyed (not "any sibling with any color binding") and
    // layers-aware (a leaf's real marks live in `prepared[i].layers`, which
    // correctly resolves a desugared mark like `mark_ribbon` — whose
    // top-level `spec.mark` Python leaves at the serde-default `point`
    // placeholder — to its real `Ribbon` sub-layer). See
    // `plan_line_ribbon_color_group_exemptions`'s doc comment.
    plan_line_ribbon_color_group_exemptions(tree, &prepared, &mut contexts);
    drop(prepared);

    // Pass 2/3 (per-leaf render): re-render each leaf with its resolved-domain
    // context so composite-shared channels land on the auto scale path (D4b). A
    // fully-empty context passes `None` so non-shared leaves render byte-identical
    // to a standalone chart. A suppressed leaf additionally returns its prepared
    // legend bundle so the compositor can build the figure legend from it.
    let mut leaf_scenes: Vec<SceneGraph> = Vec::with_capacity(n);
    let mut bundles: Vec<Option<LeafLegendBundle>> = Vec::with_capacity(n);
    for (i, leaf) in leaves.iter().enumerate() {
        let ctx = &contexts[i];
        let ctx_opt = (!ctx.is_empty()).then_some(ctx);
        let (mut scene, leaf_warnings, bundle) =
            render_leaf(leaf, ctx_opt).map_err(|source| CompositeRenderError::LeafRender {
                kind: "leaf",
                index: i,
                source,
            })?;
        // Uniquify each leaf's raw-fragment clip ids exactly once, keyed by the
        // leaf's global pre-order index so colorbar/legend-clip/inset def ids stay
        // disjoint across the composite (panel clips are auto-unique via the
        // global panel renumber below).
        uniquify_scene_raw_clips(&mut scene, i);
        leaf_scenes.push(scene);
        bundles.push(bundle);
        // Aggregated in leaf pre-order (the same order `leaves`/`flatten_leaf_specs`
        // produce), matching `render_svg`'s single-scene warning contract.
        warnings.extend(leaf_warnings);
    }

    let merge_ctx = MergeCtx {
        plan: &band_plan,
        contexts: &contexts,
        bundles: &bundles,
        overlay_groups: &overlay_groups,
    };

    // Pass 2/3 (place + merge): walk the tree, placing each leaf scene into the
    // composite frame. Panels are renumbered flat in pre-order as leaf scenes are
    // consumed (D4c); `leaf_cursor` tracks the same pre-order leaf index so a
    // figure-legend band node can capture the first participating leaf's bundle
    // from its subtree. `node_cursor` tracks the same pre-order `Composite`-node
    // index `plan_legend_bands` assigned, so `band_plan.band_nodes` (keyed by
    // that index — see `LegendBandPlan`) resolves to the right node here too.
    let mut scenes = leaf_scenes.into_iter();
    let mut panel_base = 0usize;
    let mut leaf_cursor = 0usize;
    let mut node_cursor = 0usize;
    let mut placed = build_placed(
        tree,
        &mut scenes,
        &mut panel_base,
        &mut leaf_cursor,
        &mut node_cursor,
        call_theme,
        &merge_ctx,
        None,
    );

    // Root figure chrome (title/subtitle/caption/config) — validated root-only.
    if let CompositeNode::Composite {
        title,
        subtitle,
        caption,
        config,
        ..
    } = tree
    {
        let (left_inset, right_inset, anchor) =
            resolve_root_chrome_config(config.as_ref(), tree.kind_name())?;
        inject_root_chrome(
            &mut placed.scene,
            title.as_deref(),
            subtitle.as_deref(),
            caption.as_deref(),
            left_inset,
            right_inset,
            anchor,
        );
    }

    Ok((placed.scene, warnings))
}

/// Render one leaf standalone (transforms → layout → scene) with an optional
/// resolved-domain context threaded through the D4b seam. Mirrors `render_svg`'s
/// prepare-and-layout → build-scene sequence, returning the leaf's warnings
/// alongside its scene rather than dropping `PipelineOutput::warnings` when
/// its owning `po` goes out of scope (the bug this fix closes).
///
/// A leaf in an all-leaves Overlay group carries its group's shared plot rect
/// on `ctx` ([`LeafScaleContext::imposed_plot_region`], written by
/// [`impose_shared_overlay_rects`]); `super::prepare_and_layout` hands it to
/// `compute_layout`, which lays the leaf out against that rect from the axis
/// -band stage onward. Nothing is patched afterwards here (GH #89A retired the
/// post-layout `panels[0].plot_area` overwrite this function used to perform),
/// so every layout product the leaf's scene draws from — panel rects, tick
/// pixel positions, axis titles, strip bands, legend placement — describes the
/// rect the marks actually land in.
fn render_leaf(
    leaf: &CompositeLeafInput<'_>,
    ctx: Option<&LeafScaleContext>,
) -> Result<(SceneGraph, Vec<RenderWarning>, Option<LeafLegendBundle>), RenderError> {
    // `prepare_and_layout` has no viewport guard of its own — `render_svg`/
    // `render_scene_json` each check this before calling it; a composite leaf
    // bypasses those entries, so the check is repeated here.
    if leaf.viewport.width <= 0.0 || leaf.viewport.height <= 0.0 {
        return Err(RenderError::InvalidViewport {
            width: leaf.viewport.width,
            height: leaf.viewport.height,
        });
    }
    let mut po = super::prepare_and_layout(
        leaf.spec,
        leaf.batch,
        leaf.theme,
        leaf.viewport,
        leaf.chart_config,
        ctx,
    )?;
    // A leaf whose color and/or size legend the compositor is suppressing (design
    // §6 seam) carries a figure-legend candidate bundle: the prepared legend
    // inputs (still fully built by `prepare_render_inputs` regardless of
    // suppression) plus the leaf's effective theme and color scale, so the
    // compositor can lay out and draw one figure legend from the SAME primitives
    // a per-panel legend uses. Non-suppressed leaves carry no bundle.
    let bundle = match ctx {
        Some(c) if c.suppress_color_legend || c.suppress_size_legend => {
            Some(capture_leaf_bundle(leaf, &po)?)
        }
        _ => None,
    };
    let scene = scene_build::build_scene(
        leaf.spec,
        &po.prep,
        &po.layout,
        &po.effective_theme,
        leaf.config,
        &mut po.warnings,
        leaf.chart_config,
        ctx,
    )?;
    Ok((scene, po.warnings, bundle))
}

/// Capture a suppressed leaf's prepared legend bundle as a figure-legend
/// candidate (design §6 seam contract). The bundle carries exactly what the
/// band assembler needs to lay out and draw one figure legend through the
/// existing legend primitives: the categorical entries / colorbar input, the
/// resolved (three-way) legend title, the per-channel style overrides, the
/// size/shape aux inputs, the leaf's effective theme (legend orient + fonts +
/// colors), the color scale the legend draws against, and whether this leaf
/// merged the color+size channels on a shared field (which folds size into the
/// color legend at prepare time — see [`LegendSuppression`]).
fn capture_leaf_bundle(
    leaf: &CompositeLeafInput<'_>,
    po: &super::PipelineOutput,
) -> Result<LeafLegendBundle, RenderError> {
    let color_scale = scene_build::resolve_legend_color_scale(
        leaf.spec,
        &po.prep,
        &po.effective_theme,
        leaf.chart_config,
    )?;
    let mut overrides = super::legend_overrides_from_prep(&po.prep);
    super::apply_chart_config_to_legend_overrides(&mut overrides, leaf.chart_config);
    let title = super::effective_legend_title(&po.prep);
    Ok(LeafLegendBundle {
        entries: po.prep.legend_entries.clone(),
        colorbar: po.prep.colorbar.clone(),
        title,
        overrides,
        aux: po.prep.aux_legends.clone(),
        color_scale,
        theme: po.effective_theme.clone(),
        merged_color_size: leaf_merges_color_size(leaf.spec),
    })
}

/// A leaf merges its color and size legends when both channels encode the SAME
/// field — `prepare_render_inputs` then folds size into the color legend (the
/// aux `Size` block carries the color, and the redundant colorbar is dropped),
/// so suppressing one channel must suppress the other (design §6 seam contract,
/// handoff item 3). Derived from the spec so it is known before pass 2 (the
/// suppression flags must be set before the leaf renders).
fn leaf_merges_color_size(spec: &ChartSpec) -> bool {
    match (spec.encoding.color.as_ref(), spec.encoding.size.as_ref()) {
        (Some(c), Some(s)) => c.field == s.field,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Figure-level shared legend (GH #16)
// ---------------------------------------------------------------------------

/// A sandbox `inner` extent the band legend is measured in. Large enough that the
/// legend-strip carve (which caps a strip at half the inner extent) never clips
/// the full legend; the measured content is then translated to the oriented edge
/// of the merged scene, so the sandbox size never leaks into the output.
const LEGEND_SANDBOX: f64 = 100_000.0;

/// Which channels a resolving composite node emits a figure legend for. A node
/// can band `color`, `size`, or both (a same-field color+size share stacks both
/// in one band); at least one is always `true` for a node present in
/// [`LegendBandPlan::band_nodes`].
#[derive(Debug, Clone, Copy)]
struct BandFlags {
    color: bool,
    size: bool,
}

/// The composite nodes that emit a figure-level legend band, keyed by node
/// identity. Empty for any composite with all-independent legend resolution
/// — the byte-stable path.
///
/// Keyed by a **pre-order composite-node index** (assigned only to
/// `Composite` nodes, incremented on entry — the same idiom `leaf_cursor`
/// uses for leaves) rather than the node's raw pointer: a pointer key is
/// sound only while [`plan_legend_walk`] and [`build_placed`] both borrow
/// the SAME tree value, which holds today (`render_composite_scene` walks
/// one borrowed `tree: &CompositeNode` throughout), but a future clone
/// between the two passes would silently drop every band (no error, no
/// test failure — a different address, `HashMap::get` just misses). The
/// node-index key is stable across any such clone because both walks
/// re-derive it structurally, from the tree shape rather than its address.
struct LegendBandPlan {
    band_nodes: HashMap<usize, BandFlags>,
}

/// The two pre-passes' output, as the place/merge walk consumes it.
///
/// For the figure legend band (GH #16): the plan (which nodes band which
/// channels), the per-leaf resolved contexts (to test participation), and the
/// per-leaf captured bundles (the legend content).
///
/// For overlay chrome dedup (GH #89A): `overlay_groups`, the leaf-index-space
/// map [`plan_overlay_groups`] produced **and [`impose_shared_overlay_rects`]
/// pruned**. [`build_placed`]'s `Composite` arm reads it (via each direct
/// child's leaf index at entry) to decide, per DIRECT child, whether that
/// child is a non-primary member of THIS node's overlay group. Because the
/// pre-pass equalized exactly the groups this map still names — clearing the
/// entries of any it could not — a child marked here has provably laid out
/// against the rect the surviving chrome describes.
struct MergeCtx<'a> {
    plan: &'a LegendBandPlan,
    contexts: &'a [LeafScaleContext],
    bundles: &'a [Option<LeafLegendBundle>],
    overlay_groups: &'a [Option<usize>],
}

/// A participating leaf's captured legend inputs — the figure legend's content
/// (design §6 seam contract). Built by [`capture_leaf_bundle`] for every leaf
/// the compositor suppresses; the band assembler picks the first non-empty one
/// in a resolving node's subtree (design §8.4 capture rule).
struct LeafLegendBundle {
    entries: Vec<LegendEntry>,
    colorbar: Option<ColorbarInput>,
    title: Option<String>,
    overrides: LegendOverrides,
    aux: Vec<AuxLegendInput>,
    color_scale: Option<ColorScale>,
    theme: ThemeInputs,
    /// The leaf folded color+size into one legend on a shared field (see
    /// [`leaf_merges_color_size`]).
    merged_color_size: bool,
}

impl LeafLegendBundle {
    /// A bundle is empty when the leaf produced no drawable legend content — a
    /// leaf whose legend the user disabled (`legend=None`) yields this (prepare-
    /// stage suppression), so it is skipped for figure-legend capture (§8.4).
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
            && self.colorbar.is_none()
            && self.aux.iter().all(aux_block_is_empty)
    }
}

/// True when an aux legend block carries no entries (nothing to draw).
fn aux_block_is_empty(a: &AuxLegendInput) -> bool {
    match a {
        AuxLegendInput::Size { entries, .. } => entries.is_empty(),
        AuxLegendInput::Shape { entries, .. } => entries.is_empty(),
        AuxLegendInput::StrokeDash { entries, .. } => entries.is_empty(),
    }
}

/// Per-channel legend-resolution state carried down the tree during planning.
#[derive(Debug, Clone, Copy)]
struct ChannelWalk {
    /// The effective scale-resolution mode inherited from ancestors for this
    /// channel (spec §6, GH #74): `Shared` once an ancestor established a
    /// shared union that covers this node, else `Independent`. The outermost
    /// effective-shared node is the one where `scale_eff == Shared` while
    /// `scale_inherited != Shared` — mirroring
    /// [`super::composite::resolve_nonpositional`]'s union gate bit-for-bit.
    scale_inherited: ResolveMode,
    /// We are inside a figure-legend band for this channel: participating leaves
    /// below are suppressed.
    band_active: bool,
}

/// Plan the figure-level legends for a composite tree (design §5). Writes the
/// per-leaf `suppress_color_legend`/`suppress_size_legend` flags into `contexts`
/// (read by pass 2's leaf layout) and returns the set of nodes that emit a band.
///
/// A node bands a channel when it is the outermost effective-shared-**scale**
/// node for that channel (the node where [`super::composite::resolve_nonpositional`]
/// unioned the domain across the whole leaf span — GH #74) AND its effective
/// legend resolution is `Shared`. A `legend`-independent override over a shared
/// scale therefore bands nothing and suppresses no leaf — today's per-panel
/// rendering (design §4). A leaf is suppressed only when it *participated* in
/// the shared domain (its context carries the channel), so a leaf with an
/// explicit per-chart `scale=`, or under an explicitly-independent nested node
/// (excluded from the union), keeps its own panel legend.
fn plan_legend_bands(tree: &CompositeNode, contexts: &mut [LeafScaleContext]) -> LegendBandPlan {
    let mut plan = LegendBandPlan {
        band_nodes: HashMap::new(),
    };
    let mut leaf_cursor = 0usize;
    let mut node_cursor = 0usize;
    let root = ChannelWalk {
        scale_inherited: ResolveMode::Independent,
        band_active: false,
    };
    plan_legend_walk(
        tree,
        contexts,
        &mut leaf_cursor,
        &mut node_cursor,
        root,
        root,
        &mut plan,
    );
    plan
}

/// `node_cursor` assigns each `Composite` node a pre-order index — incremented
/// once per `Composite` arm visited, on entry, before descending into
/// children — exactly mirroring how [`build_placed`] must increment its own
/// `node_cursor` at the same point in the same traversal order, so a node's
/// index here and its index there always agree (see [`LegendBandPlan`]).
fn plan_legend_walk(
    node: &CompositeNode,
    contexts: &mut [LeafScaleContext],
    leaf_cursor: &mut usize,
    node_cursor: &mut usize,
    color: ChannelWalk,
    size: ChannelWalk,
    plan: &mut LegendBandPlan,
) {
    match node {
        CompositeNode::Leaf { spec, .. } => {
            let i = *leaf_cursor;
            *leaf_cursor += 1;
            let merges = leaf_merges_color_size(spec);
            if color.band_active && contexts[i].color.is_some() {
                contexts[i].suppress_color_legend = true;
                // Same-field color+size merge: the size legend is folded into the
                // color block, so suppress both together (design §6, handoff 3).
                if merges {
                    contexts[i].suppress_size_legend = true;
                }
            }
            if size.band_active && contexts[i].size.is_some() {
                contexts[i].suppress_size_legend = true;
            }
        }
        CompositeNode::Hole { .. } => {}
        CompositeNode::Composite {
            children, resolve, ..
        } => {
            let node_idx = *node_cursor;
            *node_cursor += 1;
            // Must agree bit-for-bit with `resolve_nonpositional`'s union gate
            // (composite.rs): the band attaches at exactly the outermost
            // effective-shared node where the resolve pass unions the color/
            // size domain across the whole leaf span (GH #74). No congruence
            // gate here — the leaf-span union is congruence-agnostic (a nested
            // composite child no longer blocks sharing); congruence stays a
            // positional x/y concern only.
            let (color_next, color_band) =
                descend_channel(resolve.color, resolve.legend.color, color);
            let (size_next, size_band) = descend_channel(resolve.size, resolve.legend.size, size);
            if color_band || size_band {
                plan.band_nodes.insert(
                    node_idx,
                    BandFlags {
                        color: color_band,
                        size: size_band,
                    },
                );
            }
            for c in children {
                plan_legend_walk(
                    c,
                    contexts,
                    leaf_cursor,
                    node_cursor,
                    color_next,
                    size_next,
                    plan,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Overlay shared-rect planning (GH #89A)
// ---------------------------------------------------------------------------

/// For every `layout: Overlay` composite node whose DIRECT children are ALL
/// leaves (LayerChart's actual shape — every layer lowers to one leaf),
/// `result[i] = Some(group_start)` for each non-primary child leaf index `i`,
/// naming the leaf index of that Overlay node's first child. `None` for a
/// primary (index-0) child, a leaf outside any Overlay node, a singleton
/// Overlay (nothing to share with), or a leaf under an Overlay whose children
/// are NOT uniformly direct leaves.
///
/// That last case — a nested composite child — is the one shape this pass
/// leaves alone: such a child spans a whole subtree of leaves laid out in its
/// own frame, so there is no single "this child's plot rect" to intersect or
/// impose, and the node keeps every child's own rect and chrome (pre-#89A
/// behavior). It is unreachable from Python lowering: `LayerChart`, the sole
/// producer of an Overlay node (`composition.py`, `LayerChart._composite_tree`),
/// rejects any layer that is not a plain leaf `Chart` (`_is_leaf_chart`) with a
/// typed `ValueError` before a tree is ever built. The guard is kept because a
/// directly-constructed wire spec can still express it, and degrading to
/// per-child chrome is the honest response.
///
/// This map is the single source of truth for BOTH halves of the overlay
/// unification: [`impose_shared_overlay_rects`] equalizes exactly these
/// groups' plot rects, and [`build_placed`]'s merge arm suppresses chrome for
/// exactly the leaves this map marks non-primary — so a leaf can never have
/// its chrome dropped without having shared the rect that chrome describes.
///
/// Mirrors [`plan_legend_bands`]'s leaf-index pre-pass idiom: walked once,
/// entirely before any leaf is rendered, over the same tree shape
/// [`build_placed`] later re-walks.
fn plan_overlay_groups(tree: &CompositeNode, n_leaves: usize) -> Vec<Option<usize>> {
    let mut groups = vec![None; n_leaves];
    let mut leaf_cursor = 0usize;
    plan_overlay_group_walk(tree, &mut leaf_cursor, &mut groups);
    groups
}

fn plan_overlay_group_walk(
    node: &CompositeNode,
    leaf_cursor: &mut usize,
    groups: &mut [Option<usize>],
) {
    match node {
        CompositeNode::Leaf { .. } => {
            *leaf_cursor += 1;
        }
        CompositeNode::Hole { .. } => {}
        CompositeNode::Composite {
            layout, children, ..
        } => {
            if *layout == CompositeLayout::Overlay
                && children.len() > 1
                && children
                    .iter()
                    .all(|c| matches!(c, CompositeNode::Leaf { .. }))
            {
                let group_start = *leaf_cursor;
                for j in 1..children.len() {
                    groups[group_start + j] = Some(group_start);
                }
            }
            for c in children {
                plan_overlay_group_walk(c, leaf_cursor, groups);
            }
        }
    }
}

/// One color-consuming layer inside an Overlay group, for
/// [`plan_line_ribbon_color_group_exemptions`]'s field/scale-keyed check.
///
/// `mark` and `field` come from [`prepare::LayerPrepared`] — the SAME
/// authority `render/prepare/legend.rs`'s local (non-composite) inert-color
/// check reads — never straight off `ChartSpec.mark`/`encoding.color`,
/// which for a desugared mark (`mark_ribbon` chief among them) name a
/// serde-default placeholder (`point`) at the chart's top level while the
/// REAL mark lives in one of `spec.layers` (spec-review cycle-2 finding:
/// `fm.layer(line(color=v:Q), ribbon(color=v:Q))._composite_tree()`'s
/// second child is `ChartSpec(mark='point', layers=[{mark: 'ribbon', ...}])`).
/// `LayerPrepared::from_chart_only`/`from_chart_and_layer` already resolve
/// this correctly (flat spec → its own top-level mark; layered spec → each
/// real sub-layer), so consulting `prepared[i].layers` handles both shapes
/// uniformly with no special-casing here.
struct ColorConsumer {
    /// Index of this leaf within the group (0-based, group-local).
    member: usize,
    field: String,
    mark: crate::spec::mark::Mark,
    /// This leaf's OWN resolved color scale is `ColorInput::Numeric`
    /// (`Continuous`/`Discretizing`) — from `prepared[i].provisional_scales.color`,
    /// the same per-leaf resolution `render/prepare/legend.rs` itself reads.
    is_numeric: bool,
}

/// Mark, for every LINE/RIBBON leaf inside an all-leaf `Overlay` group (the
/// same #89A group shape [`plan_overlay_groups`] names — `LayerChart`'s
/// actual shape) whose own resolved color scale is Numeric-keyed AND which
/// has a sibling leaf in the SAME group binding the SAME color field to a
/// mark other than line/ribbon via ALSO a Numeric-keyed scale,
/// [`LeafScaleContext::color_scale_has_non_line_ribbon_sibling`] `= true`
/// (T5b static-composite fix, spec §4.0's second bullet, spec-review cycle-2
/// corrections). Field- and scale-keyed per spec §4.0's own justification
/// ("another mark SHARES the continuous scale") — a sibling on a DIFFERENT
/// field (even a genuinely-rendered categorical legend on that field) does
/// NOT exempt the line/ribbon leaf's own, unrelated, still-inert channel
/// (spec-review cycle-2: `layer(line(color=v:Q), point(color=g:N))` must
/// still warn about `v` while `g`'s own legend renders untouched).
///
/// That sibling genuinely renders the group's shared color mapping, so
/// `render::prepare::legend::build_color_legend`'s inert-color-on-line-or-
/// ribbon check — which sees only its OWN leaf's per-panel mark set under the
/// composite path, since each leaf renders through its own standalone
/// `prepare_render_inputs` and never the whole group — must not warn (or
/// suppress a colorbar) for that leaf.
///
/// Mirrors [`plan_overlay_groups`]'s leaf-index pre-pass idiom: walked once,
/// entirely before any leaf is rendered, over the same tree shape that pass
/// (and [`plan_legend_bands`]) also independently re-walk for STRUCTURE
/// (which leaves fall in which group) — but, unlike those two, reads leaf
/// FACTS (marks, color fields, scale kinds) from `prepared` rather than the
/// tree's own `ChartSpec`, for the `LayerPrepared`/desugar reason above.
///
/// An `hconcat`/`vconcat` of two independently line-colored leaves is NOT an
/// Overlay group — this pass's group-detection gate (`layout == Overlay`,
/// `children.len() > 1`, all-leaf children) never matches it — so both
/// leaves are left alone: each still renders, and warns, standalone
/// (reviewer-blessed: two offending hconcat leaves keep one warning each).
fn plan_line_ribbon_color_group_exemptions(
    tree: &CompositeNode,
    prepared: &[prepare::PreparedInputs],
    contexts: &mut [LeafScaleContext],
) {
    let mut leaf_cursor = 0usize;
    plan_line_ribbon_color_group_walk(tree, prepared, &mut leaf_cursor, contexts);
}

fn plan_line_ribbon_color_group_walk(
    node: &CompositeNode,
    prepared: &[prepare::PreparedInputs],
    leaf_cursor: &mut usize,
    contexts: &mut [LeafScaleContext],
) {
    use crate::spec::mark::Mark;
    use super::scale_resolve::{ColorInput, ColorScale};

    match node {
        CompositeNode::Leaf { .. } => {
            *leaf_cursor += 1;
        }
        CompositeNode::Hole { .. } => {}
        CompositeNode::Composite {
            layout, children, ..
        } => {
            if *layout == CompositeLayout::Overlay
                && children.len() > 1
                && children
                    .iter()
                    .all(|c| matches!(c, CompositeNode::Leaf { .. }))
            {
                let group_start = *leaf_cursor;
                let members = &prepared[group_start..group_start + children.len()];

                // Every color-bound layer across every member's REAL layers
                // (not the tree's possibly-placeholder top-level mark).
                let mut consumers: Vec<ColorConsumer> = Vec::new();
                for (member, prep) in members.iter().enumerate() {
                    let is_numeric = prep.provisional_scales.color.as_ref().map(ColorScale::input)
                        == Some(ColorInput::Numeric);
                    for layer in &prep.layers {
                        if let Some(enc) = &layer.encoding.color {
                            consumers.push(ColorConsumer {
                                member,
                                field: enc.field.clone(),
                                mark: layer.mark,
                                is_numeric,
                            });
                        }
                    }
                }

                // Exempt exactly the line/ribbon consumers with a same-field,
                // Numeric-keyed, non-line/ribbon sibling — never the whole
                // group indiscriminately.
                for c in &consumers {
                    if !matches!(c.mark, Mark::Line | Mark::Ribbon) || !c.is_numeric {
                        continue;
                    }
                    let exempted = consumers.iter().any(|other| {
                        other.member != c.member
                            && other.field == c.field
                            && other.is_numeric
                            && !matches!(other.mark, Mark::Line | Mark::Ribbon)
                    });
                    if exempted {
                        contexts[group_start + c.member].color_scale_has_non_line_ribbon_sibling = true;
                    }
                }
            }
            for c in children {
                plan_line_ribbon_color_group_walk(c, prepared, leaf_cursor, contexts);
            }
        }
    }
}

/// Give every leaf of every overlay group ONE plot rect to lay out against
/// (GH #89A), by writing it into each member's
/// [`LeafScaleContext::imposed_plot_region`] before pass 2 renders anything.
///
/// The shared rect is the INTERSECTION of the group members' natural plot
/// regions: per side, the largest gutter any member reserves for its legend
/// or axis bands. Intersecting (rather than adopting the primary leaf's rect)
/// is what lets the merge seam drop chrome at all — a non-primary leaf's own
/// legend box, which the seam never suppresses, always sits outside the
/// shared rect, so no leaf's marks can render across another leaf's chrome.
///
/// **Suppression-aware** (spec §4.2): a non-primary member's scene title WILL
/// be cleared at the merge seam, so its title band is excluded from its
/// natural region ([`LeafScaleContext::suppress_chart_title`], set here before
/// the region is measured and kept set for the real render). Reserving it
/// would push the whole group's chrome down by a band nothing is ever drawn
/// in. Legends get the opposite treatment for the same reason: they DO
/// render, so their gutters stay in.
///
/// **Coupled to the merge seam** (spec §4.2): this function mutates `groups`,
/// the map [`build_placed`] later reads to decide chrome suppression. Any
/// group it cannot equalize — a member whose layout failed, or an
/// intersection that degenerates to nothing — has its entries cleared here,
/// so those leaves keep BOTH their own geometry and their own chrome. Chrome
/// is never dropped for a leaf that did not lay out against the shared rect;
/// one decision, one source of truth, no re-derivation downstream.
///
/// Learning a member's natural region costs one extra `prepare_and_layout`
/// per overlay leaf: the region is the product of the leaf's own data, ticks,
/// theme and (composite-resolved) context, so it cannot be predicted without
/// running the layout that produces it. This runs only for leaves in an
/// all-leaves overlay group — LayerChart's two or three layers in practice,
/// never a grid/concat composite's leaves.
///
/// A degenerate intersection pushes a [`RenderWarning::OverlayGuttersDiverged`]
/// onto `warnings` (surfaced to Python by `binding::emit_warnings` like every
/// other render warning): the resulting doubled chrome is visible but hard to
/// attribute, and the cause — the layers' own gutter requests — is only known
/// here.
///
/// A member whose layout FAILS here produces no diagnostic beyond dropping
/// its group: `render_leaf` re-runs the identical `prepare_and_layout` call
/// for that leaf moments later, and surfaces the identical error at its
/// canonical position in leaf pre-order. Reporting from this pre-pass instead
/// would change WHICH leaf's error a multi-failure tree reports — and a
/// warning about chrome would be noise next to a render that is about to
/// fail outright.
fn impose_shared_overlay_rects(
    leaves: &[CompositeLeafInput<'_>],
    groups: &mut [Option<usize>],
    contexts: &mut [LeafScaleContext],
    warnings: &mut Vec<RenderWarning>,
) {
    // Each group's member leaf indices, leader (`group_start`) first, in leaf
    // order. One group's members are always consecutive leaf indices — an
    // all-leaves Overlay node's children are walked in order and each
    // contributes exactly one leaf — so a run of equal `Some(group_start)`
    // entries is exactly one group. Collected first so the natural-region
    // probes below can borrow `contexts` immutably.
    let mut members: Vec<Vec<usize>> = Vec::new();
    for (i, group) in groups.iter().enumerate() {
        let Some(group_start) = *group else { continue };
        match members.last_mut() {
            Some(g) if g[0] == group_start => g.push(i),
            _ => members.push(vec![group_start, i]),
        }
    }

    for group in members {
        // Measure every member as it will actually be RENDERED: the
        // non-primary ones without the chart-title band the merge seam is
        // about to clear. Set before the probes below, and left set for the
        // real render, so probe and render agree.
        for &i in &group[1..] {
            contexts[i].suppress_chart_title = true;
        }

        let mut rect: Option<LayoutRect> = None;
        for &i in &group {
            let Some(natural) = natural_plot_region(&leaves[i], &contexts[i]) else {
                rect = None;
                break;
            };
            rect = Some(match rect {
                Some(shared) => shared.intersect(natural),
                None => natural,
            });
        }

        // A collapsed intersection (disjoint regions — reachable when two
        // members reserve opposite-side gutters that together exceed the
        // viewport) would lay every member out on a zero rect.
        match rect {
            Some(rect) if rect.w > 0.0 && rect.h > 0.0 => {
                for i in group {
                    contexts[i].imposed_plot_region = Some(rect);
                }
            }
            // Nothing was equalized, so nothing may be suppressed: drop the
            // group from the map the merge seam reads, and undo the title
            // suppression that anticipated it. Every member keeps its own
            // geometry AND its own chrome, exactly as a nested-composite
            // overlay child does.
            //
            // `Some(degenerate)` warns, `None` does not: a degenerate
            // intersection is a real layout outcome the user can act on
            // (narrow the gutters, widen the canvas), whereas `None` means a
            // member's layout failed and `render_leaf` is about to raise that
            // failure as a typed error.
            outcome => {
                if outcome.is_some() {
                    warnings.push(RenderWarning::OverlayGuttersDiverged {
                        layers: group.len(),
                    });
                }
                for &i in &group[1..] {
                    contexts[i].suppress_chart_title = false;
                    groups[i] = None;
                }
            }
        }
    }
}

/// One leaf's plot region as it lays itself out under `ctx` — the region left
/// by its own chart-title, legend and axis-band reservations, minus whatever
/// `ctx` already suppresses (a non-primary overlay leaf's title band). `None`
/// when the leaf's layout fails (see [`impose_shared_overlay_rects`]'s error
/// contract).
fn natural_plot_region(
    leaf: &CompositeLeafInput<'_>,
    ctx: &LeafScaleContext,
) -> Option<LayoutRect> {
    let ctx_opt = (!ctx.is_empty()).then_some(ctx);
    super::prepare_and_layout(
        leaf.spec,
        leaf.batch,
        leaf.theme,
        leaf.viewport,
        leaf.chart_config,
        ctx_opt,
    )
    .ok()
    .map(|po| po.layout.plot_region)
}

/// Advance one channel's [`ChannelWalk`] across a composite node, returning the
/// state its children see and whether THIS node emits a band for the channel.
///
/// `node_scale` is the node's explicit color/size resolve (`None` = unset,
/// inherit — spec §6, GH #74); `legend_override` is its explicit
/// `resolve.legend` entry for the channel (`None` = follow the effective
/// scale mode). The effective scale mode and outermost-shared flag both come
/// from [`effective_share`](super::composite::effective_share) — the same gate
/// [`super::composite::resolve_nonpositional`] uses to place its union, so band
/// and union always coincide.
fn descend_channel(
    node_scale: Option<ResolveMode>,
    legend_override: Option<ResolveMode>,
    cur: ChannelWalk,
) -> (ChannelWalk, bool) {
    // Outermost effective-shared scale node for the channel: the resolve pass
    // unions the domain here, so this is the only node that may band it. Shared
    // with `resolve_nonpositional` through `effective_share` so band and union
    // always coincide.
    let (scale_eff, is_scale_resolver) = effective_share(node_scale, cur.scale_inherited);
    // Legend follows the effective scale mode unless explicitly overridden.
    let legend_eff = legend_override.unwrap_or(scale_eff);
    // A shared legend over a non-shared effective scale is rejected at lowering
    // (design §4, the Python `_lower_composite` guard) — a normal caller can
    // never build it. `CompositeNode::validate` does not re-check it Rust-side,
    // so a directly-constructed wire spec can still reach here; because
    // `is_scale_resolver` requires an effective-shared scale, such a spec
    // yields `band_here == false` regardless of `legend_eff` — the degrade is
    // "no band for this channel", not a misrender, mirroring how the resolve
    // pass treats the same spec (no union, per-panel fallback). Deliberately no
    // assert/log (an abort would make the documented degrade untestable). The
    // contract is pinned by
    // `invalid_wire_shared_legend_over_independent_scale_degrades_to_no_band`.
    let band_here = is_scale_resolver && legend_eff == ResolveMode::Shared;
    let next = ChannelWalk {
        scale_inherited: scale_eff,
        band_active: cur.band_active || band_here,
    };
    (next, band_here)
}

/// Draw one figure-level legend band on `scene` for the resolving node covering
/// `leaf_range`. Captures the first participating leaf's non-empty bundle in
/// pre-order (design §8.4); if every participating leaf is user-disabled (all
/// bundles empty) no band is emitted, matching design §4.
fn apply_legend_band(
    scene: &mut SceneGraph,
    merge_ctx: &MergeCtx<'_>,
    leaf_range: std::ops::Range<usize>,
    flags: BandFlags,
) {
    let captured = leaf_range.into_iter().find_map(|i| {
        let ctx = &merge_ctx.contexts[i];
        let participates =
            (flags.color && ctx.color.is_some()) || (flags.size && ctx.size.is_some());
        if !participates {
            return None;
        }
        match &merge_ctx.bundles[i] {
            Some(b) if !b.is_empty() => Some(b),
            _ => None,
        }
    });
    let Some(bundle) = captured else { return };
    draw_legend_band(scene, bundle, flags);
}

/// Lay out the band's legend content (color block + size aux) in a sandbox,
/// measure its true drawn extent, then translate it to the oriented edge of
/// `scene` and grow the scene. Reuses the single-chart legend layout + draw
/// primitives (design §7 facet parity — no parallel legend implementation).
fn draw_legend_band(scene: &mut SceneGraph, bundle: &LeafLegendBundle, flags: BandFlags) {
    let metrics = super::font::FontdueMetrics::new();
    let layouts = layout_band_legends(bundle, flags, &metrics);
    if layouts.is_empty() {
        return;
    }
    let theme = &bundle.theme;
    let label_fs = bundle
        .overrides
        .style
        .label_font_size
        .unwrap_or(theme.typography.label_font_size);
    let title_fs = theme.typography.legend_title_font_size;
    let Some((min_x, min_y, max_x, max_y)) =
        legend_layouts_extent(&layouts, label_fs, title_fs, &metrics)
    else {
        return;
    };
    let content_w = (max_x - min_x).max(0.0);
    let content_h = (max_y - min_y).max(0.0);
    let orient = theme.legend.legend_orient;
    let old_w = scene.width;
    let old_h = scene.height;

    // Placement origin of the measured content block. Left/Top first shift the
    // existing merged content over to make room (mirroring `apply_chrome_band`'s
    // header shift); Right/Bottom simply grow the far edge. Either way, the band's
    // OUTER edge — the one that borders the canvas boundary rather than the
    // `LEGEND_PLOT_GAP` toward the panels — gets `LEGEND_OUTER_PAD` of trailing
    // safety margin: `legend_layouts_extent` measures glyph *advance* widths via
    // `TextMetrics::measure_width`, which does not include the terminal glyph's
    // right side-bearing/ink overhang, so an exact-fit grow clips that sliver at
    // the canvas edge. `LEGEND_OUTER_PAD` is the same inner padding
    // `estimate_legend_size`/`estimate_colorbar_size` reserve around per-panel
    // legend content on every side (layout/legend.rs) — reused here rather than
    // inventing a second margin constant.
    let (target_x, target_y) = match orient {
        LegendOrient::Right => (old_w + LEGEND_PLOT_GAP, 0.0),
        LegendOrient::Left => {
            shift_scene(scene, LEGEND_OUTER_PAD + content_w + LEGEND_PLOT_GAP, 0.0);
            (LEGEND_OUTER_PAD, 0.0)
        }
        LegendOrient::Top => {
            shift_scene(scene, 0.0, LEGEND_OUTER_PAD + content_h + LEGEND_PLOT_GAP);
            (0.0, LEGEND_OUTER_PAD)
        }
        LegendOrient::Bottom => (0.0, old_h + LEGEND_PLOT_GAP),
    };
    let dx = target_x - min_x;
    let dy = target_y - min_y;

    let mut nodes: Vec<SceneNode> = Vec::new();
    for l in &layouts {
        nodes.extend(super::marks::legend::build_legend(
            l,
            bundle.color_scale.as_ref(),
            theme,
        ));
    }
    offset_nodes(&mut nodes, dx, dy);

    match orient {
        LegendOrient::Right | LegendOrient::Left => {
            scene.width = old_w + LEGEND_PLOT_GAP + content_w + LEGEND_OUTER_PAD;
            scene.height = old_h.max(content_h);
        }
        LegendOrient::Top | LegendOrient::Bottom => {
            scene.width = old_w.max(content_w);
            scene.height = old_h + LEGEND_PLOT_GAP + content_h + LEGEND_OUTER_PAD;
        }
    }
    scene.legend.extend(nodes);
}

/// Lay out the band's color block (categorical entries or colorbar) and its size
/// aux block against a sandbox `inner`. The color-block dispatch is the shared
/// [`layout_color_legend`] core also used by `layout::reserve_legends`; aux
/// stacking (below) is composite-band-specific. Shape aux blocks are excluded: a
/// same-field shape already folds into the color entries, and a differently-
/// keyed shape legend is not composite-shared, so it stays per-panel.
fn layout_band_legends(
    bundle: &LeafLegendBundle,
    flags: BandFlags,
    metrics: &dyn TextMetrics,
) -> Vec<LegendLayout> {
    let theme = &bundle.theme;
    let orient = theme.legend.legend_orient;
    let overrides = &bundle.overrides;
    let inner = LayoutRect {
        x: 0.0,
        y: 0.0,
        w: LEGEND_SANDBOX,
        h: LEGEND_SANDBOX,
    };

    // `!flags.color`: an empty-entries + no-colorbar input makes the shared
    // dispatch a no-op `(None, inner, ..)` via `layout_legend`'s empty-entries
    // early return — matching the pre-extraction "never attempted layout" skip.
    let (color_entries, color_colorbar): (&[LegendEntry], Option<&ColorbarInput>) = if flags.color {
        (&bundle.entries, bundle.colorbar.as_ref())
    } else {
        (&[], None)
    };
    let (color_legend, inner_after, effective_label_font_size) = layout_color_legend(
        inner,
        orient,
        theme.typography.label_font_size,
        theme.legend.legend_direction,
        theme.typography.legend_title_font_size,
        theme.legend.legend_columns,
        theme.padding.column_padding,
        color_entries,
        bundle.title.as_deref(),
        color_colorbar,
        metrics,
        overrides,
    );

    // Size aux is banded when the node shares size, or when a same-field color+
    // size merge folded size into the color block this node bands.
    let include_size = flags.size || (flags.color && bundle.merged_color_size);
    let aux_inputs: Vec<AuxLegendInput> = if include_size {
        bundle
            .aux
            .iter()
            .filter(|a| matches!(a, AuxLegendInput::Size { .. }))
            .cloned()
            .collect()
    } else {
        Vec::new()
    };
    let (aux_legends, _) = layout_aux_legends(
        &aux_inputs,
        color_legend.as_ref(),
        orient,
        inner,
        inner_after,
        effective_label_font_size,
        overrides.style.label_font_size,
        theme.typography.legend_title_font_size,
        metrics,
        theme.padding.column_padding,
    );

    let mut out: Vec<LegendLayout> = Vec::new();
    out.extend(color_legend);
    out.extend(aux_legends);
    out
}

/// The bounding box `(min_x, min_y, max_x, max_y)` of every drawn glyph across
/// `layouts` — entry swatches + labels, title, and colorbar bar + ticks. Used to
/// size the scene growth and position the band; `None` when nothing is drawn.
///
/// `title_fs` is the title's OWN font size (`theme.typography.legend_title_font_size`),
/// distinct from `label_fs`: the title is drawn at `title_fs` by `build_legend`
/// (via `layout_color_legend`), so measuring it at `label_fs` here would silently
/// under-size the band whenever a theme sets the two font sizes differently.
fn legend_layouts_extent(
    layouts: &[LegendLayout],
    label_fs: f64,
    title_fs: f64,
    metrics: &dyn TextMetrics,
) -> Option<(f64, f64, f64, f64)> {
    let line_h = metrics.line_height(label_fs);
    let title_line_h = metrics.line_height(title_fs);
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut any = false;
    let mut acc = |x0: f64, y0: f64, x1: f64, y1: f64| {
        min_x = min_x.min(x0);
        min_y = min_y.min(y0);
        max_x = max_x.max(x1);
        max_y = max_y.max(y1);
        any = true;
    };
    for l in layouts {
        for e in &l.entries {
            let lw = metrics.measure_width(&e.label, label_fs);
            let half = e.symbol_radius.unwrap_or(6.0).max(6.0);
            acc(
                e.symbol_anchor_x - half,
                e.symbol_anchor_y - line_h / 2.0,
                e.label_anchor_x + lw,
                e.symbol_anchor_y + line_h / 2.0,
            );
        }
        if let Some(t) = &l.title {
            let tw = metrics.measure_width(&t.text, title_fs);
            acc(t.x, t.y - title_line_h, t.x + tw, t.y);
        }
        if let Some(cb) = &l.colorbar {
            let mut max_tick = 0.0_f64;
            for tk in &cb.ticks {
                let tw = metrics.measure_width(&tk.label, label_fs);
                max_tick = max_tick.max(tw);
                match l.direction {
                    LegendDirection::Vertical => acc(
                        cb.bar_rect.x,
                        tk.y - line_h / 2.0,
                        cb.bar_rect.x,
                        tk.y + line_h / 2.0,
                    ),
                    // Horizontal (D5): the label is centered on the tick's `x`
                    // and drawn below the bar, so it overhangs the bar's ends by
                    // half its width on each side.
                    LegendDirection::Horizontal => {
                        let tx = tk.horizontal_x(&cb.bar_rect);
                        acc(tx - tw / 2.0, tk.y, tx + tw / 2.0, tk.y + 4.0 + line_h);
                    }
                }
            }
            match l.direction {
                LegendDirection::Vertical => acc(
                    cb.bar_rect.x,
                    cb.bar_rect.y,
                    cb.bar_rect.x + cb.bar_rect.w + 4.0 + max_tick,
                    cb.bar_rect.y + cb.bar_rect.h,
                ),
                LegendDirection::Horizontal => acc(
                    cb.bar_rect.x,
                    cb.bar_rect.y,
                    cb.bar_rect.x + cb.bar_rect.w,
                    cb.bar_rect.y + cb.bar_rect.h + 4.0 + line_h,
                ),
            }
        }
    }
    any.then_some((min_x, min_y, max_x, max_y))
}

/// Shift every panel and non-panel node in `scene` by `(dx, dy)` — the
/// make-room step for a Left/Top legend band (mirrors [`apply_chrome_band`]'s
/// header shift). No-op at `(0, 0)`.
fn shift_scene(scene: &mut SceneGraph, dx: f64, dy: f64) {
    if dx == 0.0 && dy == 0.0 {
        return;
    }
    let t = translate(dx, dy);
    for panel in &mut scene.panels {
        place_panel(panel, &t);
    }
    offset_nodes(&mut scene.title, dx, dy);
    offset_nodes(&mut scene.legend, dx, dy);
    offset_nodes(&mut scene.decorations, dx, dy);
}

// ---------------------------------------------------------------------------
// Tree walk: place + merge
// ---------------------------------------------------------------------------

/// A rendered subtree: its merged scene plus the bounding box the parent places.
struct Placed {
    scene: SceneGraph,
    width: f64,
    height: f64,
}

/// Recursively render `node` into one placed scene. Leaf scenes are consumed from
/// `scenes` in pre-order (a `Hole` consumes none — Task 8a); `panel_base` is the
/// running global panel-id offset, incremented as each leaf's panels are
/// renumbered. `call_theme` styles any per-child label encountered (see
/// [`apply_child_label`]). `parent_layout` is `node`'s immediate parent's
/// layout kind (`None` at the tree root — a hole can never be the root, so
/// this only matters for the `Hole` arm below); it is how a sized hole
/// (Task 10-rust) knows whether its parent is a linear (`hconcat`/`vconcat`)
/// layout, the only layout kind where its `width`/`height` take effect.
/// `node_cursor` assigns each `Composite` node the same pre-order index
/// [`plan_legend_walk`] assigned it — incremented at the identical point in
/// the identical traversal order — so the [`LegendBandPlan::band_nodes`]
/// lookup below always lands on the right node (see [`LegendBandPlan`]'s doc
/// for why this replaced a raw-pointer key).
#[allow(clippy::too_many_arguments)]
fn build_placed(
    node: &CompositeNode,
    scenes: &mut std::vec::IntoIter<SceneGraph>,
    panel_base: &mut usize,
    leaf_cursor: &mut usize,
    node_cursor: &mut usize,
    call_theme: &ThemeInputs,
    merge_ctx: &MergeCtx<'_>,
    parent_layout: Option<CompositeLayout>,
) -> Placed {
    let mut placed = match node {
        CompositeNode::Leaf { .. } => {
            let mut scene = scenes
                .next()
                .expect("leaf scenes count matches tree leaves (checked by entry)");
            renumber_panels(&mut scene, *panel_base);
            *panel_base += scene.panels.len();
            *leaf_cursor += 1;
            let (width, height) = (scene.width, scene.height);
            Placed {
                scene,
                width,
                height,
            }
        }
        // A hole is not a leaf: it consumes no scene from `scenes`, claims no
        // panel ids, and carries no label/chrome — it renders as an empty
        // placed subtree (Task 8a). Under grid/wrap its size is always zero:
        // `plan_grid`/`plan_wrap` size each row/column from the *max* native
        // extent per lane, so the slot's actual size is supplied by its
        // sibling cells there, leaving the hole's own cell visually blank
        // while ratio/spacing math for the *other* cells is unaffected — any
        // `width`/`height` on a grid/wrap hole is validated-legal but has no
        // effect (spec). Under a linear (hconcat/vconcat) parent, `validate`
        // guarantees both are present, and they size this placed subtree
        // directly — `plan_linear` then reserves exactly that much blank
        // space in the flow like any other child (Task 10-rust).
        CompositeNode::Hole { width, height } => {
            let linear = matches!(
                parent_layout,
                Some(CompositeLayout::Hconcat) | Some(CompositeLayout::Vconcat)
            );
            let (w, h) = if linear {
                (width.unwrap_or(0.0), height.unwrap_or(0.0))
            } else {
                (0.0, 0.0)
            };
            Placed {
                scene: empty_scene(w, h),
                width: w,
                height: h,
            }
        }
        CompositeNode::Composite {
            layout,
            children,
            spacing,
            row_ratios,
            col_ratios,
            ncols,
            nrows,
            ..
        } => {
            // Captured BEFORE recursing into children, at the same point
            // `plan_legend_walk`'s `Composite` arm captures its own
            // `node_idx` — keeping the two walks' pre-order numbering
            // bit-for-bit aligned (see `LegendBandPlan`'s doc).
            let node_idx = *node_cursor;
            *node_cursor += 1;
            let leaf_start = *leaf_cursor;
            // Each DIRECT child's leaf index AT ENTRY (before that child's own
            // subtree is walked) — captured alongside `build_placed` so
            // `child_leaf_starts[j]` names child `j`'s own leaf index (a
            // nested-composite child's is its FIRST descendant leaf's, which
            // never carries an overlay-group entry — see `plan_overlay_groups`
            // — so such a child is correctly never chosen for suppression
            // below).
            let mut child_leaf_starts: Vec<usize> = Vec::with_capacity(children.len());
            let child_placed: Vec<Placed> = children
                .iter()
                .map(|c| {
                    child_leaf_starts.push(*leaf_cursor);
                    build_placed(
                        c,
                        scenes,
                        panel_base,
                        leaf_cursor,
                        node_cursor,
                        call_theme,
                        merge_ctx,
                        Some(*layout),
                    )
                })
                .collect();
            let spacing = spacing.unwrap_or(DEFAULT_SPACING);
            let plan = plan_layout(
                *layout,
                &child_placed,
                spacing,
                row_ratios.as_deref(),
                col_ratios.as_deref(),
                *ncols,
                *nrows,
            );
            // Overlay chrome dedup (GH #89A, design §4.2): child 0's
            // grid/axes/above-marks chrome/title is the group's chrome; every
            // other member's duplicate of it is dropped. The condition is
            // exactly "this direct child is a non-primary leaf of THIS node's
            // overlay group", read off the overlay-group map AS THE
            // SHARED-RECT PRE-PASS LEFT IT — `impose_shared_overlay_rects`
            // clears the entries of any group it could not equalize, so a
            // child's chrome can only be dropped when its marks and the marks
            // under the surviving chrome were laid out against one identical
            // rect (spec §4.2, "suppression is coupled to imposition"). There
            // is no separate per-leaf "was it safe" flag: one decision, made
            // where the geometry is known, read here without re-derivation.
            // A child whose entry the pre-pass never set (a nested composite
            // child) or cleared (a member whose layout failed, an intersection
            // that degenerated) keeps its own chrome, matching the rect it
            // kept.
            //
            // Comparing against this node's own `leaf_start` needs no
            // `layout == Overlay` guard: a group entry exists only for the
            // direct leaf children of an all-leaves Overlay node, and a leaf
            // has exactly one parent — so `Some(leaf_start)` can only match at
            // the very node that owns the group.
            let suppress_chrome: Vec<bool> = child_leaf_starts
                .iter()
                .map(|&leaf_idx| {
                    merge_ctx.overlay_groups.get(leaf_idx) == Some(&Some(leaf_start))
                })
                .collect();
            let mut merged = merge_children(child_placed, plan, &suppress_chrome);
            // Figure-legend band (design §5 pass 3): if this composite node
            // resolved a channel as a shared figure legend, grow the merged scene
            // on the oriented edge and draw one legend built from the first
            // participating leaf's captured bundle. Applied AFTER children are
            // placed and BEFORE the per-child label / root chrome below, so a
            // title band stacks above the legend exactly as it does for a
            // per-panel legend.
            if let Some(flags) = merge_ctx.plan.band_nodes.get(&node_idx) {
                apply_legend_band(
                    &mut merged.scene,
                    merge_ctx,
                    leaf_start..*leaf_cursor,
                    *flags,
                );
                merged.width = merged.scene.width;
                merged.height = merged.scene.height;
            }
            merged
        }
    };
    // Per-child panel label (Task 5d): a title-only chrome band reserved above
    // this node's content at its top-left, so it moves with the child when the
    // parent places it. Mirrors the old path, where a titled composite compare
    // child was rendered standalone via `child.to_svg()` and wrapped with its
    // own figure-title chrome. Root labels are rejected by validation, so this
    // only ever fires on a non-root child.
    if let Some(label) = node.label() {
        apply_child_label(&mut placed, label, call_theme);
    }
    placed
}

/// Reserve a title-only chrome band above `placed`'s content for a per-child
/// label, shifting the child down and growing its bbox height. Reuses the
/// figure-chrome header band so a labeled child matches the old standalone
/// `child.to_svg()` title placement exactly.
///
/// Styled from `theme` — the call-level theme, not any per-leaf override
/// (composite labels belong to the composition, and per-leaf themes may be
/// heterogeneous across a tree's leaves, so there is no single "the" leaf
/// theme to pick). Mirrors the two field reads `scene_build::build_title`
/// (composite_render.rs's single-chart counterpart) uses for title styling —
/// `theme.typography.title_font_size` / `theme.colors.title_color` — rather
/// than reusing its full derivation, which also resolves a per-chart
/// `ChartSpec::title` override that composite labels have no equivalent of.
fn apply_child_label(placed: &mut Placed, label: &str, theme: &ThemeInputs) {
    let chrome = FigureChrome {
        title: Some(label),
        title_font_size: Some(theme.typography.title_font_size),
        title_color: Some(super::draw::to_scene_color(theme.colors.title_color)),
        ..Default::default()
    };
    apply_chrome_band(&mut placed.scene, chrome);
    // Only `height` grows: `apply_chrome_band` reserves a HEADER band (shifts
    // content down, grows `scene.height`); it never changes `scene.width`, so
    // re-reading `placed.scene.width` back into `placed.width` here would
    // always be a no-op assignment of the value already held.
    placed.height = placed.scene.height;
}

/// The computed placement of a composite node's children plus the node's bbox.
struct LayoutPlan {
    /// One placement transform per child, mapping native child coords into the
    /// composite frame.
    placements: Vec<LayoutScale>,
    width: f64,
    height: f64,
}

/// Compute per-child placements for one composite node. Cross-axis alignment
/// defaults to start (top/left), matching the composition binding's default
/// `align` for the concat composers.
fn plan_layout(
    layout: CompositeLayout,
    children: &[Placed],
    spacing: f64,
    row_ratios: Option<&[f64]>,
    col_ratios: Option<&[f64]>,
    ncols: Option<u32>,
    nrows: Option<u32>,
) -> LayoutPlan {
    match layout {
        CompositeLayout::Hconcat => plan_linear(children, spacing, true),
        CompositeLayout::Vconcat => plan_linear(children, spacing, false),
        CompositeLayout::Overlay => plan_overlay(children),
        CompositeLayout::Wrap => {
            let cols = ncols.unwrap_or(1).max(1) as usize;
            plan_wrap(children, spacing, cols)
        }
        CompositeLayout::Grid => {
            let cols = ncols.unwrap_or(1).max(1) as usize;
            let rows = nrows.unwrap_or(1).max(1) as usize;
            plan_grid(children, spacing, rows, cols, row_ratios, col_ratios)
        }
    }
}

/// Linear placement: `horizontal = true` lays children left-to-right (hconcat),
/// otherwise top-to-bottom (vconcat). Cross-axis offset is 0 (start-aligned).
fn plan_linear(children: &[Placed], spacing: f64, horizontal: bool) -> LayoutPlan {
    let mut placements = Vec::with_capacity(children.len());
    let mut cursor = 0.0_f64;
    let mut cross = 0.0_f64;
    for c in children {
        if horizontal {
            placements.push(translate(cursor, 0.0));
            cursor += c.width + spacing;
            cross = cross.max(c.height);
        } else {
            placements.push(translate(0.0, cursor));
            cursor += c.height + spacing;
            cross = cross.max(c.width);
        }
    }
    let main = (cursor - spacing).max(0.0);
    if horizontal {
        LayoutPlan {
            placements,
            width: main,
            height: cross,
        }
    } else {
        LayoutPlan {
            placements,
            width: cross,
            height: main,
        }
    }
}

/// Overlay: every child at the origin, bbox is the child extent max. Z-order is
/// child order (children merge in order, later children drawn on top).
fn plan_overlay(children: &[Placed]) -> LayoutPlan {
    let width = children.iter().map(|c| c.width).fold(0.0_f64, f64::max);
    let height = children.iter().map(|c| c.height).fold(0.0_f64, f64::max);
    let placements = children.iter().map(|_| translate(0.0, 0.0)).collect();
    LayoutPlan {
        placements,
        width,
        height,
    }
}

/// Wrap: children flow left-to-right into rows of `cols`, wrapping to the next
/// row. Each row is laid out like an hconcat; rows stack like a vconcat. Row
/// height is the tallest cell in the row (facet-style).
fn plan_wrap(children: &[Placed], spacing: f64, cols: usize) -> LayoutPlan {
    let mut placements = vec![translate(0.0, 0.0); children.len()];
    let mut total_w = 0.0_f64;
    let mut y = 0.0_f64;
    for (row_idx, row) in children.chunks(cols).enumerate() {
        let base = row_idx * cols;
        let row_h = row.iter().map(|c| c.height).fold(0.0_f64, f64::max);
        let mut x = 0.0_f64;
        for (j, c) in row.iter().enumerate() {
            placements[base + j] = translate(x, y);
            x += c.width + spacing;
        }
        total_w = total_w.max((x - spacing).max(0.0));
        y += row_h + spacing;
    }
    let total_h = (y - spacing).max(0.0);
    LayoutPlan {
        placements,
        width: total_w,
        height: total_h,
    }
}

/// Grid: row-major placement with F20 ratio math (absorbed from the deleted
/// `grid_compose.rs`). Each column's allocated width is `K_w * col_ratio[c]`,
/// where `K_w = min_c(native_col_w[c] / col_ratio[c])` keeps the dominant cell at
/// native size and shrinks smaller-share cells into their slots; rows are
/// symmetric. A cell whose native size matches its slot (within `SLOT_MATCH_EPS`)
/// is placed by pure translation; a mismatch produces a non-identity
/// `layout_scale` that stretches native content into the slot.
fn plan_grid(
    children: &[Placed],
    spacing: f64,
    rows: usize,
    cols: usize,
    row_ratios: Option<&[f64]>,
    col_ratios: Option<&[f64]>,
) -> LayoutPlan {
    // Ratios default to all-1 (uniform grid); congruent children then produce no
    // scaling (K == native), matching a plain grid.
    let col_r: Vec<f64> = col_ratios
        .map(<[f64]>::to_vec)
        .unwrap_or_else(|| vec![1.0; cols]);
    let row_r: Vec<f64> = row_ratios
        .map(<[f64]>::to_vec)
        .unwrap_or_else(|| vec![1.0; rows]);

    // Native max dimension per column / row.
    let mut native_col_w = vec![0.0_f64; cols];
    let mut native_row_h = vec![0.0_f64; rows];
    for (idx, c) in children.iter().enumerate() {
        let (r, col) = (idx / cols, idx % cols);
        native_col_w[col] = native_col_w[col].max(c.width);
        native_row_h[r] = native_row_h[r].max(c.height);
    }

    let k_w = fit_factor(&col_r, &native_col_w);
    let k_h = fit_factor(&row_r, &native_row_h);
    let col_w: Vec<f64> = col_r.iter().map(|r| k_w * r).collect();
    let row_h: Vec<f64> = row_r.iter().map(|r| k_h * r).collect();

    // Prefix offsets (cumulative slot extent + spacing) per column / row.
    let col_x = prefix_offsets(&col_w, spacing);
    let row_y = prefix_offsets(&row_h, spacing);

    let mut placements = vec![translate(0.0, 0.0); children.len()];
    for (idx, c) in children.iter().enumerate() {
        let (r, col) = (idx / cols, idx % cols);
        let mut sx = if c.width > 0.0 {
            col_w[col] / c.width
        } else {
            1.0
        };
        let mut sy = if c.height > 0.0 {
            row_h[r] / c.height
        } else {
            1.0
        };
        if (sx - 1.0).abs() < SLOT_MATCH_EPS && (sy - 1.0).abs() < SLOT_MATCH_EPS {
            sx = 1.0;
            sy = 1.0;
        }
        placements[idx] = LayoutScale {
            sx,
            sy,
            tx: col_x[col],
            ty: row_y[r],
        };
    }

    let total_w = col_w.iter().sum::<f64>() + spacing * cols.saturating_sub(1) as f64;
    let total_h = row_h.iter().sum::<f64>() + spacing * rows.saturating_sub(1) as f64;
    LayoutPlan {
        placements,
        width: total_w,
        height: total_h,
    }
}

/// `K = min over lanes of (native[i] / ratio[i])` for lanes with positive ratio
/// and native extent; `0.0` when no lane qualifies (all-empty grid). Mirrors
/// the deleted `grid_compose.rs`'s `k_w`/`k_h` derivation.
fn fit_factor(ratios: &[f64], native: &[f64]) -> f64 {
    let k = ratios
        .iter()
        .zip(native)
        .filter_map(|(r, n)| {
            if *r > 0.0 && *n > 0.0 {
                Some(n / r)
            } else {
                None
            }
        })
        .fold(f64::INFINITY, f64::min);
    if k.is_finite() {
        k
    } else {
        0.0
    }
}

/// Cumulative start offset of each slot: `offset[i] = sum(extent[0..i]) + i*spacing`.
fn prefix_offsets(extents: &[f64], spacing: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(extents.len());
    let mut acc = 0.0_f64;
    for (i, e) in extents.iter().enumerate() {
        out.push(acc + spacing * i as f64);
        acc += e;
    }
    out
}

/// Merge placed children into one scene. Panels are appended in child order with
/// their placement applied; non-panel nodes are translate-baked; interaction
/// state is unioned (panel refs are already global from [`renumber_panels`]).
///
/// `suppress_chrome[i]` drops child `i`'s duplicate overlay chrome — the
/// per-panel `grid`, `axes` and `chrome_above` slots plus the scene-level
/// `title` — keeping every other field (GH #89A, design §4.2). The caller
/// ([`build_placed`]'s `Composite` arm) is the ONLY place it is decided, and
/// decides it structurally: `true` exactly for a non-primary direct-leaf child
/// of an all-leaves `Overlay` node, i.e. exactly the leaves
/// [`impose_shared_overlay_rects`] laid out against ONE shared plot rect. That
/// coupling is what makes clearing sound — the surviving chrome describes the
/// same rect the dropped chrome did, so this is deduplication and not a
/// silent geometry mismatch.
///
/// The four cleared slots are chrome; every content slot is kept for every
/// child. `marks`/`strip_title` (panel content — the whole point of layering),
/// `below_marks` and `annotations` (user annotations at both z's, typed
/// siblings of the chrome slots as of GH #89B), `legend` (owned by the
/// separate shared/per-leaf legend machinery), `decorations`, `selections`,
/// `background`, `chart_description`, and the interaction fold all survive.
/// `chrome_above` is cleared BECAUSE it is chrome: GH #89B routes a
/// `zindex >= 1` axis and its gridlines there rather than leaving them
/// interleaved in `annotations`, so an above-marks axis on a non-primary leaf
/// dedups exactly like a below-marks one instead of surviving as a second
/// visible axis (the pre-#89A refusal door).
///
/// Overlay children do NOT share one panel rect "by construction": only their
/// PLACEMENT ORIGIN is (`plan_overlay` translates every child by `(0, 0)`);
/// their natural `plot_area` differs whenever one child reserves a chrome
/// gutter (a legend, a title) another doesn't — even though composite-shared
/// x/y forces identical DOMAINS, the reviewed defect measured real divergence
/// (e.g. `w=520.466` vs `w=567.201` with a legend on one layer only; a
/// `y`-offset shift with a title on one layer only). The shared-rect pre-pass,
/// not this function, is what removes that divergence. For every non-`Overlay`
/// layout, and for any `Overlay` node the pre-pass skipped, `suppress_chrome[i]`
/// is `false` and every child's fields merge exactly as they did before this
/// seam existed — the only invariant this function itself relies on is that a
/// cleared `Vec<SceneNode>` moves nothing, which is unconditionally true
/// regardless of geometry.
fn merge_children(children: Vec<Placed>, plan: LayoutPlan, suppress_chrome: &[bool]) -> Placed {
    let mut merged = empty_scene(plan.width, plan.height);
    let mut zoom = true;
    let mut pan = true;
    let mut toolbar = true;

    for (i, (child, t)) in children.into_iter().zip(plan.placements).enumerate() {
        let mut scene = child.scene;

        // The chrome slots, and only those: `grid` is gridlines-only and
        // `chrome_above` is axis/grid-only as of GH #89B, so clearing them
        // cannot reach a user annotation (`below_marks`/`annotations`, their
        // typed siblings, are content and always survive). See this
        // function's doc for the full kept/cleared enumeration.
        if suppress_chrome.get(i).copied().unwrap_or(false) {
            for panel in &mut scene.panels {
                panel.grid.clear();
                panel.axes.clear();
                panel.chrome_above.clear();
            }
            scene.title.clear();
        }

        for mut panel in scene.panels.drain(..) {
            place_panel(&mut panel, &t);
            merged.panels.push(panel);
        }

        offset_nodes(&mut scene.title, t.tx, t.ty);
        offset_nodes(&mut scene.legend, t.tx, t.ty);
        offset_nodes(&mut scene.decorations, t.tx, t.ty);
        merged.title.append(&mut scene.title);
        merged.legend.append(&mut scene.legend);
        merged.decorations.append(&mut scene.decorations);

        merged.selections.append(&mut scene.selections);
        if merged.background.is_none() {
            merged.background = scene.background;
        }
        if merged.chart_description.is_none() {
            merged.chart_description = scene.chart_description.take();
        }

        let ci = &mut scene.interaction;
        merged.interaction.conditionals.append(&mut ci.conditionals);
        merged.interaction.tick_levels.append(&mut ci.tick_levels);
        merged
            .interaction
            .linked_panels
            .append(&mut ci.linked_panels);
        for p in ci.params.drain(..) {
            if !merged.interaction.params.iter().any(|q| q.name == p.name) {
                merged.interaction.params.push(p);
            }
        }
        for b in ci.param_bindings.drain(..) {
            if !merged.interaction.param_bindings.contains(&b) {
                merged.interaction.param_bindings.push(b);
            }
        }
        zoom = zoom && ci.zoom_enabled;
        pan = pan && ci.pan_enabled;
        // Fold `toolbar` by the same AND-rule as `zoom_enabled`/`pan_enabled`
        // (burndown item 3 — it used to stay hardcoded `true` on `merged`
        // regardless of what any child carried). Every leaf resolves
        // `toolbar: true` by default (the single-chart default, `empty_scene`
        // below), so an all-defaults composition is unaffected; a leaf that
        // explicitly disabled its toolbar now turns the merged composite's
        // toolbar off too, instead of being silently overridden back on.
        toolbar = toolbar && ci.toolbar;
    }

    merged.interaction.zoom_enabled = zoom;
    merged.interaction.pan_enabled = pan;
    merged.interaction.toolbar = toolbar;
    Placed {
        scene: merged,
        width: plan.width,
        height: plan.height,
    }
}

/// A fresh empty merge target sized to `(w, h)`, carrying the single-chart
/// interaction defaults (`zoom_enabled`/`pan_enabled`/`toolbar` all `true`).
/// [`merge_children`] ANDs each child's flags into these, so an all-defaults
/// composition is unaffected while any child that disabled a flag flips it
/// off on the merged scene too.
fn empty_scene(w: f64, h: f64) -> SceneGraph {
    use ferrum_scene::InteractionConfig;
    SceneGraph {
        width: w,
        height: h,
        background: None,
        title: Vec::new(),
        panels: Vec::new(),
        legend: Vec::new(),
        decorations: Vec::new(),
        selections: Vec::new(),
        interaction: InteractionConfig {
            zoom_enabled: true,
            pan_enabled: true,
            conditionals: Vec::new(),
            linked_panels: Vec::new(),
            tick_levels: Vec::new(),
            toolbar: true,
            params: Vec::new(),
            param_bindings: Vec::new(),
        },
        chart_description: None,
    }
}

// ---------------------------------------------------------------------------
// Placement primitives
// ---------------------------------------------------------------------------

/// A pure-translation [`LayoutScale`].
fn translate(tx: f64, ty: f64) -> LayoutScale {
    LayoutScale {
        sx: 1.0,
        sy: 1.0,
        tx,
        ty,
    }
}

/// Compose two transforms: `outer ∘ inner` — apply `inner` then `outer`.
fn compose(outer: &LayoutScale, inner: &LayoutScale) -> LayoutScale {
    LayoutScale {
        sx: outer.sx * inner.sx,
        sy: outer.sy * inner.sy,
        tx: outer.sx * inner.tx + outer.tx,
        ty: outer.sy * inner.ty + outer.ty,
    }
}

/// Place one panel by transform `t`. A pure-translation placement of an
/// identity-scale panel bakes the offset into the panel geometry (keeping the
/// panel's `layout_scale` at identity, so non-ratio composites carry final
/// `plot_area` rects); a scaling placement — or a panel already carrying a
/// non-identity `layout_scale` — composes into `layout_scale`, leaving content at
/// native coordinates for the walkers to transform (amended-D4a).
fn place_panel(panel: &mut Panel, t: &LayoutScale) {
    let pure_translate = t.sx == 1.0 && t.sy == 1.0;
    if pure_translate && panel.layout_scale.is_identity() {
        offset_panel(panel, t.tx, t.ty);
    } else if pure_translate {
        panel.layout_scale.tx += t.tx;
        panel.layout_scale.ty += t.ty;
    } else {
        panel.layout_scale = compose(t, &panel.layout_scale);
    }
}

/// Renumber a leaf scene's panels to the global namespace starting at `base`, and
/// remap every interaction reference (tick levels, linked panels, param
/// bindings) that keys on a panel id in lockstep (D4c).
fn renumber_panels(scene: &mut SceneGraph, base: usize) {
    for panel in &mut scene.panels {
        panel.id += base;
    }
    for tl in &mut scene.interaction.tick_levels {
        tl.panel_id += base;
    }
    for lp in &mut scene.interaction.linked_panels {
        for p in lp {
            *p += base;
        }
    }
    for b in &mut scene.interaction.param_bindings {
        if let Some(p) = b.panel.as_mut() {
            *p += base;
        }
    }
}

/// Translate a panel's geometry by `(dx, dy)` — the bake path for identity-scale
/// panels placed by pure translation.
fn offset_panel(panel: &mut Panel, dx: f64, dy: f64) {
    offset_rect(&mut panel.plot_area, dx, dy);
    offset_rect(&mut panel.clip, dx, dy);
    offset_nodes(&mut panel.grid, dx, dy);
    // GH #89B: `below_marks`/`chrome_above` are typed siblings of `grid`/
    // `annotations`, not sub-buckets of them — they need the same
    // translation bake every other node slot gets, or a leaf placed anywhere
    // but the origin strands their content at the pre-placement offset while
    // its marks (correctly baked below) move to the real position.
    offset_nodes(&mut panel.below_marks, dx, dy);
    offset_nodes(&mut panel.axes, dx, dy);
    offset_nodes(&mut panel.chrome_above, dx, dy);
    offset_nodes(&mut panel.annotations, dx, dy);
    offset_nodes(&mut panel.strip_title, dx, dy);
    for batch in &mut panel.marks {
        offset_mark_batch(batch, dx, dy);
    }
}

fn offset_rect(rect: &mut Rect, dx: f64, dy: f64) {
    rect.x += dx;
    rect.y += dy;
}

fn offset_mark_batch(batch: &mut MarkBatch, dx: f64, dy: f64) {
    for node in &mut batch.nodes {
        offset_node(node, dx, dy);
    }
}

fn offset_nodes(nodes: &mut [SceneNode], dx: f64, dy: f64) {
    for node in nodes {
        offset_node(node, dx, dy);
    }
}

/// Translate a single scene node by `(dx, dy)`. Mirrors the Python
/// `_offset_node` contract (`_scene_merge.py`) across every node variant; a `Raw`
/// fragment is translated as a whole unit via a `<g transform="translate">`
/// wrapper, and a `Group` recurses into its children.
fn offset_node(node: &mut SceneNode, dx: f64, dy: f64) {
    if dx == 0.0 && dy == 0.0 {
        return;
    }
    match node {
        SceneNode::Rect { x, y, .. } => {
            *x += dx;
            *y += dy;
        }
        SceneNode::Circle { cx, cy, .. } => {
            *cx += dx;
            *cy += dy;
        }
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
        SceneNode::Image { x, y, .. } => {
            *x += dx;
            *y += dy;
        }
        SceneNode::Path { commands, .. } => {
            for cmd in commands {
                offset_path_cmd(cmd, dx, dy);
            }
        }
        SceneNode::Polygon { rings, .. } => {
            for ring in rings {
                for pt in ring {
                    pt[0] += dx;
                    pt[1] += dy;
                }
            }
        }
        SceneNode::Polyline { points, .. } => {
            for (px, py) in points {
                *px += dx;
                *py += dy;
            }
        }
        SceneNode::Group { children, .. } => {
            for child in children {
                offset_node(child, dx, dy);
            }
        }
        SceneNode::Raw { svg, .. } => {
            *svg = format!(
                r#"<g transform="translate({},{})">{svg}</g>"#,
                super::svg::fmt_f(dx),
                super::svg::fmt_f(dy),
            );
        }
    }
}

fn offset_path_cmd(cmd: &mut ferrum_scene::PathCmd, dx: f64, dy: f64) {
    use ferrum_scene::PathCmd;
    match cmd {
        PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => {
            *x += dx;
            *y += dy;
        }
        PathCmd::QuadTo { cx, cy, x, y } => {
            *cx += dx;
            *cy += dy;
            *x += dx;
            *y += dy;
        }
        PathCmd::CubicTo {
            c1x,
            c1y,
            c2x,
            c2y,
            x,
            y,
        } => {
            *c1x += dx;
            *c1y += dy;
            *c2x += dx;
            *c2y += dy;
            *x += dx;
            *y += dy;
        }
        PathCmd::HLineTo { x } => *x += dx,
        PathCmd::VLineTo { y } => *y += dy,
        PathCmd::ArcTo { x, y, .. } => {
            *x += dx;
            *y += dy;
        }
        PathCmd::Close => {}
    }
}

// ---------------------------------------------------------------------------
// Raw-fragment clip-id uniquification
// ---------------------------------------------------------------------------

/// Uniquify every `Raw` node's clip/colorbar/legend-clip ids in `scene` with the
/// per-leaf `cell_idx` prefix, mirroring [`super::svg::uniquify_clip_ids`] for the
/// scene-node world. Applied once per leaf (before placement) so colorbar and
/// legend-clip defs from different leaves stay disjoint in the merged scene.
fn uniquify_scene_raw_clips(scene: &mut SceneGraph, cell_idx: usize) {
    for node in scene
        .title
        .iter_mut()
        .chain(&mut scene.legend)
        .chain(&mut scene.decorations)
    {
        uniquify_node_raw_clips(node, cell_idx);
    }
    for panel in &mut scene.panels {
        // GH #89B: `below_marks`/`chrome_above` are typed siblings of the
        // other node slots here, not sub-buckets of `grid`/`annotations` —
        // include them in the chain so a leaf's `Raw` fragments in either
        // slot get the same per-leaf clip-id uniquification as every other
        // slot (latent today: only `AnnotationSpec::Text` reaches
        // `below_marks`, and axis/grid chrome is Line/Text, so no `Raw` node
        // currently lands in either — this closes the omission before a
        // future producer hits it).
        for node in panel
            .grid
            .iter_mut()
            .chain(&mut panel.below_marks)
            .chain(&mut panel.axes)
            .chain(&mut panel.chrome_above)
            .chain(&mut panel.annotations)
            .chain(&mut panel.strip_title)
        {
            uniquify_node_raw_clips(node, cell_idx);
        }
        for batch in &mut panel.marks {
            for node in &mut batch.nodes {
                uniquify_node_raw_clips(node, cell_idx);
            }
        }
    }
}

fn uniquify_node_raw_clips(node: &mut SceneNode, cell_idx: usize) {
    match node {
        SceneNode::Raw { svg, .. } => {
            *svg = uniquify_clip_ids(svg, cell_idx);
        }
        SceneNode::Group { children, .. } => {
            for child in children {
                uniquify_node_raw_clips(child, cell_idx);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Root figure chrome
// ---------------------------------------------------------------------------

/// Inject a root title/subtitle/caption band into the merged scene. Shifts every
/// panel and non-panel node down by the header band height, injects the chrome
/// text nodes (already in outer-canvas space), and grows the canvas by both band
/// heights — the scene-native counterpart of `figure_chrome::wrap_with_chrome`.
/// No-op when title/subtitle/caption are all `None` (an inset/anchor override
/// with no chrome text still emits nothing, matching the legacy compositor).
#[allow(clippy::too_many_arguments)]
fn inject_root_chrome(
    scene: &mut SceneGraph,
    title: Option<&str>,
    subtitle: Option<&str>,
    caption: Option<&str>,
    left_inset: f64,
    right_inset: f64,
    anchor: ChromeAnchor,
) {
    let chrome = FigureChrome {
        title,
        subtitle,
        caption,
        left_inset,
        right_inset,
        anchor,
        ..Default::default()
    };
    apply_chrome_band(scene, chrome);
}

/// Chrome config parsed from a composite tree root's `config` slot (spec §6,
/// Task 10-rust): `{"left_inset": f64?, "right_inset": f64?, "anchor": str?}`
/// — exactly the shape `_chrome.py::chrome_kwargs()` produces for the deleted
/// N-ary SVG compositor's PyO3 bindings, so `configure_padding(left=/right=)`
/// and `configure_title(anchor=)` reach the composite entry the same way they
/// reached the legacy path. `deny_unknown_fields` mirrors
/// [`crate::spec::composite::CompositeNode`]'s wire convention: a stray key is
/// a typed error, not a silent ignore.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RootChromeConfig {
    #[serde(default)]
    left_inset: Option<f64>,
    #[serde(default)]
    right_inset: Option<f64>,
    #[serde(default)]
    anchor: Option<String>,
}

/// Resolve `(left_inset, right_inset, anchor)` from the root's opaque `config`
/// `serde_json::Value` (absent keys, or an absent `config` entirely, keep the
/// same Rust default the deleted N-ary SVG compositor's bindings applied on
/// omission — [`DEFAULT_CHROME_INSET`] / [`ChromeAnchor::Start`]). `kind` is the tree
/// root's kind name (always a layout kind — `config` is root-only, and only
/// `Composite` nodes carry it), used to pinpoint a malformed `config` in the
/// returned error.
fn resolve_root_chrome_config(
    config: Option<&serde_json::Value>,
    kind: &'static str,
) -> Result<(f64, f64, ChromeAnchor), CompositeRenderError> {
    let parsed: RootChromeConfig = match config {
        None => RootChromeConfig {
            left_inset: None,
            right_inset: None,
            anchor: None,
        },
        Some(value) => serde_json::from_value(value.clone()).map_err(|source| {
            CompositeRenderError::RootChromeConfigInvalid {
                kind,
                message: source.to_string(),
            }
        })?,
    };
    let anchor = match parsed.anchor.as_deref() {
        None => ChromeAnchor::Start,
        Some(s) => s.parse::<ChromeAnchor>().map_err(|source| {
            CompositeRenderError::RootChromeConfigInvalid {
                kind,
                message: source.to_string(),
            }
        })?,
    };
    Ok((
        parsed.left_inset.unwrap_or(DEFAULT_CHROME_INSET),
        parsed.right_inset.unwrap_or(DEFAULT_CHROME_INSET),
        anchor,
    ))
}

/// Reserve a figure-chrome band (title/subtitle header above, caption footer
/// below) around `scene`'s content: shift every panel and non-panel node down
/// by the header height, inject the chrome text nodes (already in outer-canvas
/// space), and grow the canvas height by both band heights. The scene-native
/// counterpart of `figure_chrome::wrap_with_chrome`. No-op (returns `(0.0,
/// 0.0)`) when the chrome is empty. Shared by the root chrome
/// ([`inject_root_chrome`]) and the per-child label ([`apply_child_label`]) so
/// both place their bands identically.
fn apply_chrome_band(scene: &mut SceneGraph, chrome: FigureChrome<'_>) -> (f64, f64) {
    if chrome.is_empty() {
        return (0.0, 0.0);
    }
    let panel_w = scene.width;
    let panel_h = scene.height;
    let (nodes, header_h, footer_h) = title_nodes(chrome, panel_w, panel_h);

    if header_h != 0.0 {
        let shift = translate(0.0, header_h);
        for panel in &mut scene.panels {
            place_panel(panel, &shift);
        }
        offset_nodes(&mut scene.title, 0.0, header_h);
        offset_nodes(&mut scene.legend, 0.0, header_h);
        offset_nodes(&mut scene.decorations, 0.0, header_h);
    }

    scene.title.extend(nodes);
    scene.height = panel_h + header_h + footer_h;
    (header_h, footer_h)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::SymbolKind;
    use crate::render::config::RenderConfig;
    use crate::spec::composite::CompositeResolve;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::DataType as EncDataType;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    // -- builders -------------------------------------------------------------

    fn scatter_spec() -> ChartSpec {
        ChartSpec {
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
            params: Vec::new(),
            chart_description: None,
        }
    }

    fn xy_batch(xs: &[f64], ys: &[f64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs.to_vec())),
                Arc::new(Float64Array::from(ys.to_vec())),
            ],
        )
        .unwrap()
    }

    fn leaf_node(data: usize) -> CompositeNode {
        CompositeNode::Leaf {
            spec: Box::new(scatter_spec()),
            data,
            label: None,
        }
    }

    fn composite(layout: CompositeLayout, children: Vec<CompositeNode>) -> CompositeNode {
        CompositeNode::Composite {
            layout,
            children,
            label: None,
            resolve: CompositeResolve::default(),
            spacing: None,
            row_ratios: None,
            col_ratios: None,
            ncols: None,
            nrows: None,
            title: None,
            subtitle: None,
            caption: None,
            config: None,
        }
    }

    /// Hold owned per-leaf render inputs so `CompositeLeafInput` borrows stay valid.
    struct LeafHold {
        spec: ChartSpec,
        batch: RecordBatch,
        theme: ThemeInputs,
        config: RenderConfig,
        chart_config: ChartConfig,
    }

    fn hold() -> LeafHold {
        LeafHold {
            spec: scatter_spec(),
            batch: xy_batch(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]),
            theme: ThemeInputs::default(),
            config: RenderConfig::default(),
            chart_config: ChartConfig::default(),
        }
    }

    fn leaf_input(h: &LeafHold, w: f64, ht: f64) -> CompositeLeafInput<'_> {
        CompositeLeafInput {
            spec: &h.spec,
            batch: &h.batch,
            theme: &h.theme,
            viewport: Viewport {
                width: w,
                height: ht,
            },
            config: &h.config,
            chart_config: &h.chart_config,
        }
    }

    // -- dual-axis (#52) leaf builders ---------------------------------------

    /// A dual-axis LayerChart spec: layer 0 shares the primary y, layer 1 is
    /// `independent_y` (its own right-axis slot). Mirrors scene_build's
    /// `two_layer_dual_y_spec(true)` — the canonical secondary-y fixture.
    fn dual_axis_spec() -> ChartSpec {
        use crate::spec::layer::Layer;
        let y_layer = |field: &str, independent_y: bool| Layer {
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
            independent_y,
        };
        ChartSpec {
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
            layers: Some(vec![y_layer("y0", false), y_layer("y1", true)]),
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            params: Vec::new(),
            chart_description: None,
        }
    }

    /// Matching batch: `y0 ∈ [1,3]` (small, primary slot) and `y1 ∈ [100,300]`
    /// (large, independent slot) so the two slots are clearly separable.
    fn dual_axis_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y0", DataType::Float64, false),
            Field::new("y1", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
            ],
        )
        .unwrap()
    }

    fn dual_axis_hold() -> LeafHold {
        LeafHold {
            spec: dual_axis_spec(),
            batch: dual_axis_batch(),
            theme: ThemeInputs::default(),
            config: RenderConfig::default(),
            chart_config: ChartConfig::default(),
        }
    }

    fn dual_leaf_node(data: usize) -> CompositeNode {
        CompositeNode::Leaf {
            spec: Box::new(dual_axis_spec()),
            data,
            label: None,
        }
    }

    // -- layout math ----------------------------------------------------------

    fn placed_stub(w: f64, h: f64) -> Placed {
        Placed {
            scene: empty_scene(w, h),
            width: w,
            height: h,
        }
    }

    #[test]
    fn merge_children_all_defaults_keep_toolbar_enabled() {
        // Baseline: every leaf resolves the single-chart default
        // (toolbar/zoom/pan all `true`), so an all-defaults composition's
        // merged toolbar must stay `true` — the fold must not accidentally
        // flip the common case off.
        let children = vec![placed_stub(100.0, 50.0), placed_stub(80.0, 60.0)];
        let plan = plan_linear(&children, 10.0, true);
        let merged = merge_children(children, plan, &[false, false]);
        assert!(merged.scene.interaction.toolbar);
    }

    #[test]
    fn merge_children_folds_toolbar_like_zoom_and_pan() {
        // Discriminating counterpart: a child with `toolbar: false` must
        // disable the merged composite's toolbar too, the same AND-fold
        // `zoom_enabled`/`pan_enabled` already get — not silently overridden
        // back to the `empty_scene` default (burndown item 3).
        let a = placed_stub(100.0, 50.0);
        let mut b = placed_stub(80.0, 60.0);
        b.scene.interaction.toolbar = false;
        let children = vec![a, b];
        let plan = plan_linear(&children, 10.0, true);
        let merged = merge_children(children, plan, &[false, false]);
        assert!(!merged.scene.interaction.toolbar);
    }

    #[test]
    fn hconcat_places_children_left_to_right_with_spacing() {
        let children = vec![placed_stub(100.0, 50.0), placed_stub(80.0, 60.0)];
        let plan = plan_linear(&children, 10.0, true);
        assert_eq!(plan.placements[0], translate(0.0, 0.0));
        assert_eq!(plan.placements[1], translate(110.0, 0.0));
        assert_eq!(plan.width, 190.0); // 100 + 10 + 80
        assert_eq!(plan.height, 60.0); // max
    }

    #[test]
    fn vconcat_stacks_children_with_spacing() {
        let children = vec![placed_stub(100.0, 50.0), placed_stub(80.0, 60.0)];
        let plan = plan_linear(&children, 10.0, false);
        assert_eq!(plan.placements[0], translate(0.0, 0.0));
        assert_eq!(plan.placements[1], translate(0.0, 60.0));
        assert_eq!(plan.width, 100.0); // max
        assert_eq!(plan.height, 120.0); // 50 + 10 + 60
    }

    #[test]
    fn overlay_places_all_at_origin_bbox_is_max() {
        let children = vec![placed_stub(100.0, 50.0), placed_stub(80.0, 90.0)];
        let plan = plan_overlay(&children);
        assert_eq!(plan.placements[0], translate(0.0, 0.0));
        assert_eq!(plan.placements[1], translate(0.0, 0.0));
        assert_eq!(plan.width, 100.0);
        assert_eq!(plan.height, 90.0);
    }

    #[test]
    fn wrap_two_cols_flows_into_rows() {
        let children = vec![
            placed_stub(50.0, 40.0),
            placed_stub(50.0, 40.0),
            placed_stub(50.0, 40.0),
        ];
        let plan = plan_wrap(&children, 10.0, 2);
        // Row 0: (0,0), (60,0). Row 1: (0,50).
        assert_eq!(plan.placements[0], translate(0.0, 0.0));
        assert_eq!(plan.placements[1], translate(60.0, 0.0));
        assert_eq!(plan.placements[2], translate(0.0, 50.0));
        assert_eq!(plan.width, 110.0); // 50 + 10 + 50
        assert_eq!(plan.height, 90.0); // 40 + 10 + 40
    }

    #[test]
    fn grid_congruent_cells_have_identity_scale() {
        // 2x2 of equal 50x50 cells, no ratios → uniform grid, no scaling.
        let children: Vec<Placed> = (0..4).map(|_| placed_stub(50.0, 50.0)).collect();
        let plan = plan_grid(&children, 5.0, 2, 2, None, None);
        for p in &plan.placements {
            assert_eq!(p.sx, 1.0, "congruent grid must not scale");
            assert_eq!(p.sy, 1.0, "congruent grid must not scale");
        }
        assert_eq!(
            plan.placements[0],
            LayoutScale {
                sx: 1.0,
                sy: 1.0,
                tx: 0.0,
                ty: 0.0
            }
        );
        assert_eq!(
            plan.placements[3],
            LayoutScale {
                sx: 1.0,
                sy: 1.0,
                tx: 55.0,
                ty: 55.0
            }
        );
        assert_eq!(plan.width, 105.0); // 50 + 5 + 50
        assert_eq!(plan.height, 105.0);
    }

    #[test]
    fn grid_row_ratios_scale_smaller_share_cell() {
        // One column, two rows; row0 native 100 tall, row1 native 100 tall, but
        // row ratios [3, 1] → K_h = min(100/3, 100/1) = 33.33; slots 100, 33.33.
        // Row 0 slot == native (scale 1), row 1 slot 33.33 < native 100 → sy<1.
        let children = vec![placed_stub(80.0, 100.0), placed_stub(80.0, 100.0)];
        let plan = plan_grid(&children, 0.0, 2, 1, Some(&[3.0, 1.0]), None);
        // Row 0: dominant share, native size.
        assert!(
            (plan.placements[0].sy - 1.0).abs() < 1e-9,
            "row0 sy={}",
            plan.placements[0].sy
        );
        assert_eq!(plan.placements[0].ty, 0.0);
        // Row 1: shrunk to 1/3.
        assert!(
            (plan.placements[1].sy - (1.0 / 3.0)).abs() < 1e-9,
            "row1 sy={}",
            plan.placements[1].sy
        );
        assert!(
            (plan.placements[1].ty - 100.0).abs() < 1e-9,
            "row1 ty={}",
            plan.placements[1].ty
        );
        // Columns: single col, native 80, ratio 1 → K_w=80, slot 80, sx=1.
        assert_eq!(plan.placements[0].sx, 1.0);
        assert_eq!(plan.placements[1].sx, 1.0);
    }

    #[test]
    fn plan_grid_hole_cell_does_not_affect_lane_sizing() {
        // 2x2, row0 = [80x100, 0x0 (hole stub)], row1 = [80x100, 80x100].
        // Column/row sizing is unaffected by the hole: it never becomes the
        // dominant (max) extent in a lane it shares with a real cell, so this
        // plan is identical to a plain 2x2 uniform grid without a hole.
        let children = vec![
            placed_stub(80.0, 100.0),
            placed_stub(0.0, 0.0), // hole stub
            placed_stub(80.0, 100.0),
            placed_stub(80.0, 100.0),
        ];
        let plan = plan_grid(&children, 10.0, 2, 2, None, None);
        assert_eq!(
            plan.width, 170.0,
            "80 + 10 + 80, unaffected by the hole's 0 width"
        );
        assert_eq!(
            plan.height, 210.0,
            "100 + 10 + 100, unaffected by the hole's 0 height"
        );
        for (i, p) in plan.placements.iter().enumerate() {
            assert_eq!(p.sx, 1.0, "cell {i} should not scale");
            assert_eq!(p.sy, 1.0, "cell {i} should not scale");
        }
        assert_eq!(plan.placements[2], translate(0.0, 110.0));
        assert_eq!(plan.placements[3], translate(90.0, 110.0));
    }

    // -- placement primitives -------------------------------------------------

    #[test]
    fn compose_applies_inner_then_outer() {
        let inner = LayoutScale {
            sx: 2.0,
            sy: 3.0,
            tx: 1.0,
            ty: 1.0,
        };
        let outer = LayoutScale {
            sx: 10.0,
            sy: 10.0,
            tx: 5.0,
            ty: 5.0,
        };
        let c = compose(&outer, &inner);
        // point (1,1): inner -> (3,4); outer -> (35,45).
        assert_eq!(c.apply(1.0, 1.0), (35.0, 45.0));
    }

    #[test]
    fn place_panel_pure_translate_bakes_and_keeps_identity() {
        let mut panel = stub_panel();
        place_panel(&mut panel, &translate(20.0, 30.0));
        assert!(
            panel.layout_scale.is_identity(),
            "translate placement keeps identity ls"
        );
        assert_eq!(panel.plot_area.x, 20.0);
        assert_eq!(panel.plot_area.y, 30.0);
    }

    #[test]
    fn place_panel_scaling_sets_layout_scale_native_coords() {
        let mut panel = stub_panel();
        let t = LayoutScale {
            sx: 0.5,
            sy: 0.25,
            tx: 10.0,
            ty: 40.0,
        };
        place_panel(&mut panel, &t);
        // Native coords untouched; layout_scale carries the whole placement.
        assert_eq!(panel.plot_area.x, 0.0);
        assert_eq!(panel.layout_scale, t);
    }

    #[test]
    fn place_panel_translate_on_nonidentity_ls_adds_translation() {
        let mut panel = stub_panel();
        panel.layout_scale = LayoutScale {
            sx: 0.5,
            sy: 0.5,
            tx: 1.0,
            ty: 2.0,
        };
        place_panel(&mut panel, &translate(10.0, 20.0));
        assert_eq!(
            panel.layout_scale,
            LayoutScale {
                sx: 0.5,
                sy: 0.5,
                tx: 11.0,
                ty: 22.0
            }
        );
        assert_eq!(
            panel.plot_area.x, 0.0,
            "native coords untouched for non-identity ls"
        );
    }

    fn stub_panel() -> Panel {
        use ferrum_scene::CoordKind;
        Panel {
            id: 0,
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 100.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 100.0,
            },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: false,
                clip: false,
                y_domains: Vec::new(),
            },
            grid: Vec::new(),
            marks: Vec::new(),
            axes: Vec::new(),
            annotations: Vec::new(),
            strip_title: Vec::new(),
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        }
    }

    // -- offset_node ----------------------------------------------------------

    #[test]
    fn offset_node_translates_each_variant() {
        use ferrum_scene::{FillStroke, StrokeStyle};
        let fill = FillStroke {
            fill: None,
            stroke: None,
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        };
        let stroke = StrokeStyle {
            color: ferrum_scene::Color::rgb(0, 0, 0),
            width: 1.0,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        };
        let mut circle = SceneNode::Circle {
            cx: 1.0,
            cy: 2.0,
            r: 3.0,
            style: fill.clone(),
        };
        offset_node(&mut circle, 5.0, 7.0);
        assert!(matches!(circle, SceneNode::Circle { cx, cy, .. } if cx == 6.0 && cy == 9.0));

        let mut line = SceneNode::Line {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            style: stroke,
        };
        offset_node(&mut line, 2.0, 3.0);
        assert!(
            matches!(line, SceneNode::Line { x1, y1, x2, y2, .. } if x1 == 2.0 && y1 == 3.0 && x2 == 3.0 && y2 == 4.0)
        );

        let mut poly = SceneNode::Polygon {
            rings: vec![vec![[0.0, 0.0], [1.0, 1.0]]],
            style: fill,
        };
        offset_node(&mut poly, 1.0, 1.0);
        if let SceneNode::Polygon { rings, .. } = &poly {
            assert_eq!(rings[0][0], [1.0, 1.0]);
            assert_eq!(rings[0][1], [2.0, 2.0]);
        } else {
            panic!("expected polygon");
        }
    }

    #[test]
    fn offset_node_raw_wraps_in_translate_group() {
        let mut raw = SceneNode::Raw {
            svg: "<rect/>".into(),
            anchor: Default::default(),
        };
        offset_node(&mut raw, 5.0, 8.0);
        if let SceneNode::Raw { svg, .. } = &raw {
            assert!(
                svg.contains(r#"<g transform="translate(5,8)"><rect/></g>"#),
                "svg: {svg}"
            );
        } else {
            panic!("expected raw");
        }
    }

    // -- end-to-end -----------------------------------------------------------

    #[test]
    fn leaf_count_mismatch_errors() {
        let tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        let h = hold();
        let leaves = [leaf_input(&h, 300.0, 200.0)]; // only 1, tree has 2
        let err = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap_err();
        assert!(matches!(
            err,
            CompositeRenderError::LeafCountMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn hconcat_end_to_end_renumbers_panels_and_sizes_viewport() {
        let tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 2, "two leaves → two panels");
        // Panels globally renumbered 0..N in pre-order.
        assert_eq!(scene.panels[0].id, 0);
        assert_eq!(scene.panels[1].id, 1);
        // Composite viewport = 300 + 10 + 300 = 610 wide, 200 tall.
        assert_eq!(scene.width, 610.0);
        assert_eq!(scene.height, 200.0);
        // Second panel baked to the right of the first (identity ls, translated).
        assert!(scene.panels[1].layout_scale.is_identity());
        assert!(scene.panels[1].plot_area.x > scene.panels[0].plot_area.x + 300.0);
    }

    /// Cross-task verification (#52): a dual-axis LayerChart leaf nested inside a
    /// composite (an `overlay` under an `hconcat`) must keep its per-slot
    /// secondary-y state through the flatten/place/merge path — the final
    /// `SceneGraph` panel for that leaf still carries a non-empty `y_domains`
    /// (one entry per slot) and its mark batches still bind their layer's
    /// `y_slot`. If the composite placement stripped either, the WASM
    /// secondary-y relabel/rescale seam would silently degrade to a single axis
    /// for any dual-axis chart used inside a composition.
    #[test]
    fn nested_dual_axis_leaf_retains_y_domains_and_slotted_batches() {
        // hconcat([ overlay([dual_axis_leaf]), scatter_leaf ]) — the dual leaf is
        // two composite levels deep. Pre-order leaves: [dual, scatter].
        let inner = composite(CompositeLayout::Overlay, vec![dual_leaf_node(0)]);
        let tree = composite(CompositeLayout::Hconcat, vec![inner, leaf_node(1)]);

        let dh = dual_axis_hold();
        let sh = hold();
        let leaves = [leaf_input(&dh, 600.0, 400.0), leaf_input(&sh, 300.0, 400.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        // Exactly one panel (the dual-axis leaf) carries per-slot y-domains; the
        // scatter leaf's coord leaves the list empty.
        let dual_panels: Vec<&ferrum_scene::Panel> = scene
            .panels
            .iter()
            .filter(|p| match &p.coord {
                ferrum_scene::CoordKind::Cartesian { y_domains, .. } => !y_domains.is_empty(),
                _ => false,
            })
            .collect();
        assert_eq!(
            dual_panels.len(),
            1,
            "exactly one nested panel must retain per-slot y_domains, got {}",
            dual_panels.len()
        );

        let dual = dual_panels[0];
        match &dual.coord {
            ferrum_scene::CoordKind::Cartesian { y_domains, .. } => {
                assert_eq!(
                    y_domains.len(),
                    2,
                    "dual-axis leaf must keep one y-domain per slot through placement"
                );
                let (_, slot0_hi) = y_domains[0].expect("slot 0 domain preserved");
                let (slot1_lo, _) = y_domains[1].expect("slot 1 domain preserved");
                assert!(slot0_hi < 50.0, "slot 0 must stay the small y0 domain");
                assert!(
                    slot1_lo > 50.0,
                    "slot 1 must stay the large independent y1 domain"
                );
            }
            other => panic!("expected Cartesian coord, got {other:?}"),
        }

        // The independent layer's mark batch still binds slot 1 after the merge.
        assert_eq!(dual.marks.len(), 2, "one mark batch per layer");
        let slots: Vec<usize> = dual.marks.iter().map(|b| b.y_slot).collect();
        assert!(
            slots.contains(&0) && slots.contains(&1),
            "mark batches must retain both slot 0 (primary) and slot 1 (independent), got {slots:?}"
        );
    }

    #[test]
    fn global_panel_ids_are_unique_across_nested_tree() {
        // grid(2x1) of [ hconcat(leaf, leaf), leaf ] → 3 leaves, 3 panels.
        let inner = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        let mut tree = composite(CompositeLayout::Grid, vec![inner, leaf_node(2)]);
        if let CompositeNode::Composite { nrows, ncols, .. } = &mut tree {
            *nrows = Some(2);
            *ncols = Some(1);
        }
        let h = [hold(), hold(), hold()];
        let leaves = [
            leaf_input(&h[0], 200.0, 150.0),
            leaf_input(&h[1], 200.0, 150.0),
            leaf_input(&h[2], 200.0, 150.0),
        ];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 3);
        let ids: Vec<usize> = scene.panels.iter().map(|p| p.id).collect();
        assert_eq!(ids, vec![0, 1, 2], "panels renumbered 0..N pre-order");
    }

    #[test]
    fn grid_ratio_cell_emits_non_identity_layout_scale() {
        // 2 rows x 1 col with row ratios [3,1]: differently-native rows force the
        // small-share row to scale → non-identity layout_scale on that panel.
        let mut tree = composite(CompositeLayout::Grid, vec![leaf_node(0), leaf_node(1)]);
        if let CompositeNode::Composite {
            nrows,
            ncols,
            row_ratios,
            ..
        } = &mut tree
        {
            *nrows = Some(2);
            *ncols = Some(1);
            *row_ratios = Some(vec![3.0, 1.0]);
        }
        let h0 = hold();
        let h1 = hold();
        // Same native size so the ratio (not native disparity) drives scaling.
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 2);
        // Row 0 dominant share → identity (native). Row 1 shrunk → non-identity.
        assert!(
            scene.panels[0].layout_scale.is_identity(),
            "row0 should be native"
        );
        assert!(
            !scene.panels[1].layout_scale.is_identity(),
            "row1 must carry a ratio layout_scale"
        );
        assert!((scene.panels[1].layout_scale.sy - (1.0 / 3.0)).abs() < 1e-9);
    }

    // -- hole cells (Task 8a) --------------------------------------------------

    #[test]
    fn grid_with_hole_places_three_panels_and_ratio_math_is_unaffected() {
        // 2x2 grid: row0 = [leaf0, hole], row1 = [leaf1, leaf2]. The hole
        // occupies the top-right corner (JointChart's empty-corner shape) and
        // must render no panel; the other three panels land at the same rects
        // a plain uniform 2x2 grid would produce — the hole's zero native
        // extent never shrinks a row/column that also holds a real cell.
        let mut tree = composite(
            CompositeLayout::Grid,
            vec![
                leaf_node(0),
                CompositeNode::Hole {
                    width: None,
                    height: None,
                },
                leaf_node(1),
                leaf_node(2),
            ],
        );
        if let CompositeNode::Composite { nrows, ncols, .. } = &mut tree {
            *nrows = Some(2);
            *ncols = Some(2);
        }
        assert!(tree.validate().is_ok(), "hole is valid under grid");

        let h = [hold(), hold(), hold()];
        let leaves = [
            leaf_input(&h[0], 300.0, 200.0),
            leaf_input(&h[1], 300.0, 200.0),
            leaf_input(&h[2], 300.0, 200.0),
        ];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 3, "hole must not emit a panel");

        // No ratios and uniform native sizes → uniform grid, every placement
        // is identity-scale pure translation (mirrors
        // `grid_congruent_cells_have_identity_scale`).
        for p in &scene.panels {
            assert!(
                p.layout_scale.is_identity(),
                "uniform grid: no cell should scale"
            );
        }
        // Row 1 (leaf1) sits below row 0 (leaf0) by exactly row 0's native
        // height + spacing (200 + 10 = 210) — the hole sharing row 0 does not
        // change row 0's height.
        let row_offset = scene.panels[1].plot_area.y - scene.panels[0].plot_area.y;
        assert!(
            (row_offset - 210.0).abs() < 1e-9,
            "row offset: {row_offset}"
        );
        assert!(
            (scene.panels[1].plot_area.x - scene.panels[0].plot_area.x).abs() < 1e-9,
            "leaf1 shares leaf0's column"
        );
        // Column 1 (leaf2) sits right of column 0 (leaf1) by exactly column
        // 0's native width + spacing (300 + 10 = 310) — the hole's own
        // (empty) column does not change column 0's width.
        let col_offset = scene.panels[2].plot_area.x - scene.panels[1].plot_area.x;
        assert!(
            (col_offset - 310.0).abs() < 1e-9,
            "col offset: {col_offset}"
        );
        assert!(
            (scene.panels[2].plot_area.y - scene.panels[1].plot_area.y).abs() < 1e-9,
            "leaf2 shares leaf1's row"
        );
    }

    #[test]
    fn wrap_with_trailing_hole_renders_real_leaves_and_skips_the_hole() {
        // RepeatChart's `corner=True` shape: an odd leaf count padded to a
        // rectangle with a trailing hole. 3 leaves + 1 hole at ncols=2.
        let mut tree = composite(
            CompositeLayout::Wrap,
            vec![
                leaf_node(0),
                leaf_node(1),
                leaf_node(2),
                CompositeNode::Hole {
                    width: None,
                    height: None,
                },
            ],
        );
        if let CompositeNode::Composite { ncols, .. } = &mut tree {
            *ncols = Some(2);
        }
        assert!(tree.validate().is_ok(), "hole is valid under wrap");

        let h = [hold(), hold(), hold()];
        let leaves = [
            leaf_input(&h[0], 300.0, 200.0),
            leaf_input(&h[1], 300.0, 200.0),
            leaf_input(&h[2], 300.0, 200.0),
        ];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 3, "trailing hole must not emit a panel");

        // Row 1 (leaf2, alone — its sibling slot is the hole) sits below row 0
        // by exactly row 0's height + spacing, starting again at column 0.
        let row_offset = scene.panels[2].plot_area.y - scene.panels[0].plot_area.y;
        assert!(
            (row_offset - 210.0).abs() < 1e-9,
            "row offset: {row_offset}"
        );
        assert!(
            (scene.panels[2].plot_area.x - scene.panels[0].plot_area.x).abs() < 1e-9,
            "leaf2 starts row 1 at column 0"
        );
    }

    #[test]
    fn root_chrome_offsets_panels_down_by_header_height() {
        let mut tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        if let CompositeNode::Composite { title, .. } = &mut tree {
            *title = Some("Figure title".into());
        }
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];

        // Baseline: same tree without chrome.
        let bare = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        let (bare_scene, _warnings) =
            render_composite_scene(&bare, &leaves, &ThemeInputs::default()).unwrap();
        let bare_y = bare_scene.panels[0].plot_area.y;

        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        // A title band was reserved: canvas grew and panels shifted down.
        assert!(
            scene.height > bare_scene.height,
            "chrome must grow the canvas height"
        );
        let header_h = scene.height - bare_scene.height;
        assert!(header_h > 0.0);
        assert!(
            (scene.panels[0].plot_area.y - (bare_y + header_h)).abs() < 1e-9,
            "panel must shift down by exactly the header band height",
        );
        // Chrome text node injected into the title list.
        assert!(
            scene
                .title
                .iter()
                .any(|n| matches!(n, SceneNode::Text { content, .. } if content == "Figure title")),
            "figure title text node must be present",
        );
    }

    // -- root chrome `config` slot (Task 10-rust sub-task 1) --------------------

    /// Locate the title text node's `(x, style.anchor)` — the discriminator for
    /// every `config`-driven placement test below (mirrors
    /// `figure_chrome.rs`'s `title_nodes_*_anchor_uses_*` idiom, at the
    /// composite-scene level instead of calling `title_nodes` directly).
    fn title_node_x_anchor(scene: &SceneGraph) -> (f64, ferrum_scene::TextAnchor) {
        scene
            .title
            .iter()
            .find_map(|n| match n {
                SceneNode::Text {
                    x, style, content, ..
                } if content == "T" => Some((*x, style.anchor)),
                _ => None,
            })
            .expect("title text node must be present")
    }

    fn hconcat_two_leaves_titled() -> CompositeNode {
        let mut tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        if let CompositeNode::Composite { title, .. } = &mut tree {
            *title = Some("T".into());
        }
        tree
    }

    #[test]
    fn root_config_absent_uses_default_inset_and_start_anchor() {
        // No `config` at all: same default the deleted N-ary SVG compositor's
        // bindings applied on omission (`DEFAULT_CHROME_INSET`, `ChromeAnchor::Start`).
        let tree = hconcat_two_leaves_titled();
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        let (x, anchor) = title_node_x_anchor(&scene);
        assert_eq!(x, DEFAULT_CHROME_INSET);
        assert_eq!(anchor, ferrum_scene::TextAnchor::Start);
    }

    #[test]
    fn root_config_left_inset_shifts_start_anchored_chrome() {
        let mut tree = hconcat_two_leaves_titled();
        if let CompositeNode::Composite { config, .. } = &mut tree {
            *config = Some(serde_json::json!({"left_inset": 60.0}));
        }
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        let (x, anchor) = title_node_x_anchor(&scene);
        assert_eq!(
            x, 60.0,
            "custom left_inset must reposition the start-anchored chrome"
        );
        assert_eq!(anchor, ferrum_scene::TextAnchor::Start);
    }

    #[test]
    fn root_config_middle_anchor_centers_chrome() {
        let mut tree = hconcat_two_leaves_titled();
        if let CompositeNode::Composite { config, .. } = &mut tree {
            *config = Some(serde_json::json!({"anchor": "middle"}));
        }
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        let (x, anchor) = title_node_x_anchor(&scene);
        assert_eq!(
            x,
            scene.width / 2.0,
            "middle anchor must center on the composed width"
        );
        assert_eq!(anchor, ferrum_scene::TextAnchor::Middle);
    }

    #[test]
    fn root_config_end_anchor_uses_right_inset() {
        let mut tree = hconcat_two_leaves_titled();
        if let CompositeNode::Composite { config, .. } = &mut tree {
            *config = Some(serde_json::json!({"anchor": "end", "right_inset": 40.0}));
        }
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        let (x, anchor) = title_node_x_anchor(&scene);
        assert_eq!(
            x,
            scene.width - 40.0,
            "end anchor must use the custom right_inset"
        );
        assert_eq!(anchor, ferrum_scene::TextAnchor::End);
    }

    #[test]
    fn root_config_unknown_key_is_typed_error_naming_root_kind() {
        let mut tree = hconcat_two_leaves_titled();
        if let CompositeNode::Composite { config, .. } = &mut tree {
            *config = Some(serde_json::json!({"bogus": 1}));
        }
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let err = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap_err();
        match err {
            CompositeRenderError::RootChromeConfigInvalid { kind, message } => {
                assert_eq!(kind, "hconcat");
                assert!(message.contains("bogus"), "message: {message}");
            }
            other => panic!("expected RootChromeConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn root_config_invalid_anchor_is_typed_error() {
        let mut tree = hconcat_two_leaves_titled();
        if let CompositeNode::Composite { config, .. } = &mut tree {
            *config = Some(serde_json::json!({"anchor": "diagonal"}));
        }
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let err = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap_err();
        match err {
            CompositeRenderError::RootChromeConfigInvalid { kind, message } => {
                assert_eq!(kind, "hconcat");
                assert!(
                    message.contains("start")
                        && message.contains("middle")
                        && message.contains("end"),
                    "message: {message}"
                );
            }
            other => panic!("expected RootChromeConfigInvalid, got {other:?}"),
        }
    }

    // -- sized holes under hconcat/vconcat (Task 10-rust sub-task 2) -------------

    #[test]
    fn sized_hole_under_hconcat_reserves_blank_space_and_emits_no_panel() {
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![
                leaf_node(0),
                CompositeNode::Hole {
                    width: Some(50.0),
                    height: Some(150.0),
                },
                leaf_node(1),
            ],
        );
        assert!(
            tree.validate().is_ok(),
            "fully-sized hole is valid under hconcat"
        );

        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 2, "hole must not emit a panel");

        // leaf0 (300 wide) + spacing (10) + hole (50 wide) + spacing (10) = 370.
        let x_offset = scene.panels[1].plot_area.x - scene.panels[0].plot_area.x;
        assert!((x_offset - 370.0).abs() < 1e-9, "x offset: {x_offset}");
        assert!(
            (scene.panels[1].plot_area.y - scene.panels[0].plot_area.y).abs() < 1e-9,
            "hconcat: leaves share the same row"
        );
    }

    #[test]
    fn sized_hole_under_vconcat_reserves_blank_space_and_emits_no_panel() {
        let tree = composite(
            CompositeLayout::Vconcat,
            vec![
                leaf_node(0),
                CompositeNode::Hole {
                    width: Some(150.0),
                    height: Some(50.0),
                },
                leaf_node(1),
            ],
        );
        assert!(
            tree.validate().is_ok(),
            "fully-sized hole is valid under vconcat"
        );

        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 2, "hole must not emit a panel");

        // leaf0 (200 tall) + spacing (10) + hole (50 tall) + spacing (10) = 270.
        let y_offset = scene.panels[1].plot_area.y - scene.panels[0].plot_area.y;
        assert!((y_offset - 270.0).abs() < 1e-9, "y offset: {y_offset}");
        assert!(
            (scene.panels[1].plot_area.x - scene.panels[0].plot_area.x).abs() < 1e-9,
            "vconcat: leaves share the same column"
        );
    }

    #[test]
    fn grid_hole_size_fields_are_ignored_by_cell_math() {
        // Same shape as `grid_with_hole_places_three_panels_and_ratio_math_is_unaffected`
        // but the hole now carries (huge, wrong) `width`/`height` — the layout
        // pass must produce byte-identical offsets, proving grid/wrap holes
        // ignore the size fields (cell math governs, unaffected by Task 10-rust).
        let mut tree = composite(
            CompositeLayout::Grid,
            vec![
                leaf_node(0),
                CompositeNode::Hole {
                    width: Some(9999.0),
                    height: Some(9999.0),
                },
                leaf_node(1),
                leaf_node(2),
            ],
        );
        if let CompositeNode::Composite { nrows, ncols, .. } = &mut tree {
            *nrows = Some(2);
            *ncols = Some(2);
        }
        assert!(
            tree.validate().is_ok(),
            "sized hole is valid under grid (fields ignored)"
        );

        let h = [hold(), hold(), hold()];
        let leaves = [
            leaf_input(&h[0], 300.0, 200.0),
            leaf_input(&h[1], 300.0, 200.0),
            leaf_input(&h[2], 300.0, 200.0),
        ];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 3, "hole must not emit a panel");

        let row_offset = scene.panels[1].plot_area.y - scene.panels[0].plot_area.y;
        assert!(
            (row_offset - 210.0).abs() < 1e-9,
            "row offset unaffected by hole size: {row_offset}"
        );
        let col_offset = scene.panels[2].plot_area.x - scene.panels[1].plot_area.x;
        assert!(
            (col_offset - 310.0).abs() < 1e-9,
            "col offset unaffected by hole size: {col_offset}"
        );
    }

    #[test]
    fn shared_x_leaves_render_with_unioned_domain() {
        // Two hconcat leaves sharing x. The resolved shared domain must reach each
        // leaf, so both panels carry the SAME x extent spanning the union of the
        // two leaves' data — not each leaf's own [1..3] / [10..30]. This proves the
        // D4b seam propagates the shared domain end-to-end at the composite level
        // (the auto-path-vs-bypass padding discrimination is pinned by the 5a unit
        // tests in scale_resolve/tests.rs).
        let spec = scatter_spec();
        let b0 = xy_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        let b1 = xy_batch(&[10.0, 20.0, 30.0], &[1.0, 2.0, 3.0]);
        let h0 = LeafHold {
            batch: b0,
            ..hold()
        };
        let h1 = LeafHold {
            batch: b1,
            spec: spec.clone(),
            ..hold()
        };

        let mut tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.x = crate::layout::facet::ResolveMode::Shared;
        }
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let dom = |p: &Panel| match &p.coord {
            ferrum_scene::CoordKind::Cartesian { x_domain, .. } => *x_domain,
            _ => None,
        };
        let d0 = dom(&scene.panels[0]).expect("panel 0 x_domain");
        let d1 = dom(&scene.panels[1]).expect("panel 1 x_domain");
        assert_eq!(
            d0, d1,
            "shared-x panels must carry the identical resolved x domain"
        );
        // The shared extent spans BOTH leaves: panel 0's own data maxes at 3.0, so a
        // domain reaching 30.0 proves it absorbed leaf 1's extent (and vice versa) —
        // the discriminator against per-leaf independent resolution.
        assert!(
            d0.0 <= 1.0,
            "shared lower extent expected ~1.0, got {}",
            d0.0
        );
        assert!(
            d0.1 >= 30.0,
            "shared upper extent expected ~30.0, got {}",
            d0.1
        );
    }

    // -- GH #74 lockstep cross-check ------------------------------------------

    /// Build a composite node with an explicit `resolve.color` mode.
    fn composite_color(children: Vec<CompositeNode>, color: Option<ResolveMode>) -> CompositeNode {
        let mut node = composite(CompositeLayout::Hconcat, children);
        if let CompositeNode::Composite { resolve, .. } = &mut node {
            resolve.color = color;
        }
        node
    }

    /// Reference model of `composite.rs::resolve_nonpositional`'s color-union
    /// gate and recursion: records the pre-order composite-node index (the same
    /// numbering `plan_legend_walk` assigns via its `node_cursor`) at every node
    /// where the union fires. Shares the production `effective_share` helper, so
    /// this pins the *traversal structure* (how `inherited` threads down, where
    /// nodes are counted) rather than re-deriving the gate.
    fn collect_color_union_nodes(
        node: &CompositeNode,
        inherited: ResolveMode,
        cursor: &mut usize,
        acc: &mut Vec<usize>,
    ) {
        let CompositeNode::Composite {
            children, resolve, ..
        } = node
        else {
            return;
        };
        let node_idx = *cursor;
        *cursor += 1;
        let (eff, is_outermost) = effective_share(resolve.color, inherited);
        if is_outermost {
            acc.push(node_idx);
        }
        for child in children {
            collect_color_union_nodes(child, eff, cursor, acc);
        }
    }

    /// The set of nodes where the resolve pass fires a color union must equal
    /// the set where the legend pass attaches a color band (GH #74). Both walks
    /// gate through the shared `effective_share`; this exercises them on ONE
    /// nested tree and asserts the two node-sets coincide.
    ///
    /// Tree (color resolve in parens):
    /// ```text
    /// root(Shared) ┬ A(unset → inherits Shared) ┬ leaf ┬ leaf
    ///              ├ B(Independent) ─ G(Shared)  ┬ leaf ┬ leaf
    ///              └ leaf
    /// ```
    /// Pre-order composite indices: root=0, A=1, B=2, G=3. The union fires at
    /// the two outermost effective-shared nodes: root (0) and the re-sharing G
    /// (3) beneath B's independent boundary.
    #[test]
    fn color_union_nodes_equal_band_attach_nodes() {
        let tree = composite_color(
            vec![
                composite_color(vec![leaf_node(0), leaf_node(1)], None),
                composite_color(
                    vec![composite_color(
                        vec![leaf_node(2), leaf_node(3)],
                        Some(ResolveMode::Shared),
                    )],
                    Some(ResolveMode::Independent),
                ),
                leaf_node(4),
            ],
            Some(ResolveMode::Shared),
        );

        // Reference union-fire set (mirrors resolve_nonpositional).
        let mut union_nodes = Vec::new();
        let mut cursor = 0usize;
        collect_color_union_nodes(
            &tree,
            ResolveMode::Independent,
            &mut cursor,
            &mut union_nodes,
        );
        union_nodes.sort_unstable();

        // Real band-attach set from the production legend planner.
        let mut contexts = vec![LeafScaleContext::default(); 5];
        let plan = plan_legend_bands(&tree, &mut contexts);
        let mut band_nodes: Vec<usize> = plan
            .band_nodes
            .iter()
            .filter(|(_, flags)| flags.color)
            .map(|(idx, _)| *idx)
            .collect();
        band_nodes.sort_unstable();

        assert_eq!(union_nodes, vec![0, 3], "outermost effective-shared nodes");
        assert_eq!(
            union_nodes, band_nodes,
            "resolve-union nodes must equal legend-band nodes (GH #74 lockstep)"
        );
    }

    // -- gap-fix additions: clip uniquification / typed errors / overlay z-order

    /// A point spec with a continuous (quantitative) color encoding — the real
    /// producer of a `SceneNode::Raw` colorbar fragment (`marks::legend::
    /// build_legend`'s `ferrum-colorbar-0` gradient def), reached through the
    /// same `render_leaf` → `scene_build::build_scene` path every leaf uses.
    /// No facet/shared-color resolve involved: each leaf independently starts
    /// its own `ferrum-colorbar-0` counter at 0, which is exactly the collision
    /// `uniquify_scene_raw_clips` exists to prevent once two such leaves merge.
    fn color_spec() -> ChartSpec {
        ChartSpec {
            encoding: Encoding {
                color: Some(EncodingSpec {
                    field: "c".into(),
                    type_: Some(EncDataType::Quantitative),
                    ..Default::default()
                }),
                ..scatter_spec().encoding
            },
            ..scatter_spec()
        }
    }

    fn xyc_batch(xs: &[f64], ys: &[f64], cs: &[f64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("c", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs.to_vec())),
                Arc::new(Float64Array::from(ys.to_vec())),
                Arc::new(Float64Array::from(cs.to_vec())),
            ],
        )
        .unwrap()
    }

    /// `color_spec()`'s Point-mark twin with `mark: Line` — for the T5b
    /// static-composite tests (spec-review 2026-08-28 finding).
    fn line_color_spec() -> ChartSpec {
        ChartSpec { mark: Mark::Line, ..color_spec() }
    }

    /// A tree `Leaf` node embedding an arbitrary `spec` (rather than
    /// `leaf_node`'s hardcoded `scatter_spec()`) — for tests whose fixtures
    /// need the tree's own `spec` to match `CompositeLeafInput.spec` (some
    /// consumers, e.g. `plan_legend_bands`, read Leaf facts straight off the
    /// tree; `plan_line_ribbon_color_group_exemptions` instead reads
    /// `prepared[i].layers`/`.provisional_scales`, built from
    /// `CompositeLeafInput`, but keeping both in sync avoids two
    /// independently-maintained spec shapes per test).
    fn leaf_node_with(spec: ChartSpec, data: usize) -> CompositeNode {
        CompositeNode::Leaf {
            spec: Box::new(spec),
            data,
            label: None,
        }
    }

    // -- T5b static-composite fix (spec §4.0's second bullet, spec-review 2026-08-28) --

    /// The reviewer's exact static mixed probe:
    /// `fm.layer(line(color=v), point(color=v))` — both leaves bind the SAME
    /// continuous field under one `Overlay` group. The point leaf genuinely
    /// renders the shared mapping, so the line leaf must NOT warn (previously
    /// a spurious `UnsupportedColorScaleOnMark` fired even though a colorbar
    /// was, in fact, present). The fix must not depend on leaf order:
    /// `plan_line_ribbon_color_group_exemptions`'s group check is a
    /// symmetric `.any()` over the whole group by construction — swapping
    /// the two children pins that it stays order-independent (0 warnings,
    /// same surviving-colorbar count, both orders).
    #[test]
    fn overlay_mixed_line_and_point_sharing_continuous_color_never_warns_either_order() {
        fn run(marks_in_order: [ChartSpec; 2]) -> (usize, usize) {
            let batch = || xyc_batch(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], &[0.0, 5.0, 10.0, 15.0]);
            let h0 = LeafHold { spec: marks_in_order[0].clone(), batch: batch(), ..hold() };
            let h1 = LeafHold { spec: marks_in_order[1].clone(), batch: batch(), ..hold() };
            let tree = composite(
                CompositeLayout::Overlay,
                vec![
                    leaf_node_with(marks_in_order[0].clone(), 0),
                    leaf_node_with(marks_in_order[1].clone(), 1),
                ],
            );
            let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
            let (scene, warnings) =
                render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
            let gradient_count = scene
                .legend
                .iter()
                .filter(|n| matches!(n, SceneNode::Raw { svg, .. } if svg.contains("linearGradient")))
                .count();
            (warnings.len(), gradient_count)
        }

        let (warns_line_first, gradients_line_first) = run([line_color_spec(), color_spec()]);
        assert_eq!(warns_line_first, 0, "line+point sharing a continuous color scale must not warn");
        assert!(gradients_line_first > 0, "the colorbar must be kept, not blanked");

        let (warns_point_first, gradients_point_first) = run([color_spec(), line_color_spec()]);
        assert_eq!(warns_point_first, 0, "point+line (swapped order) must not warn either");
        assert_eq!(
            gradients_point_first, gradients_line_first,
            "surviving-colorbar count must be stable regardless of child order"
        );
    }

    /// Reviewer-blessed control: an `hconcat` (not `Overlay`) of two
    /// INDEPENDENT line-only continuous-color leaves is NOT a #89A group —
    /// `plan_line_ribbon_color_group_exemptions`'s gate (`layout == Overlay`)
    /// never matches `Hconcat`, so neither leaf is exempted. Each still
    /// renders, and warns, standalone — pinning that the fix is scoped to
    /// Overlay groups specifically, not "any leaf with any sibling."
    #[test]
    fn hconcat_two_independent_offending_line_leaves_each_warn_once() {
        let spec = line_color_spec();
        let batch = || xyc_batch(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], &[0.0, 5.0, 10.0, 15.0]);
        let h0 = LeafHold { spec: spec.clone(), batch: batch(), ..hold() };
        let h1 = LeafHold { spec: spec.clone(), batch: batch(), ..hold() };
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_with(spec.clone(), 0), leaf_node_with(spec, 1)],
        );
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (_scene, warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        let unsupported: Vec<_> = warnings
            .iter()
            .filter(|w| matches!(w, RenderWarning::UnsupportedColorScaleOnMark { .. }))
            .collect();
        assert_eq!(
            unsupported.len(),
            2,
            "each independent line leaf must warn on its own; hconcat is not an Overlay \
             group, so neither is exempted: {warnings:?}"
        );
    }

    // -- spec-review cycle-2: layers-aware + field/scale-keyed exemption ----

    /// The REAL shape Python's `mark_ribbon()` lowers to (spec-review
    /// cycle-2 finding, evidence dumped in the record):
    /// `fm.layer(...)._composite_tree()`'s ribbon child is
    /// `ChartSpec(mark='point', layers=[{mark: 'ribbon', ...}])` — a
    /// serde-default `Point` PLACEHOLDER at the top level (never actually
    /// drawn) with the true `Ribbon` mark living inside `spec.layers`.
    /// Building tests from a flat `mark: Mark::Ribbon` spec instead — the
    /// trap the reviewer flagged as having bitten twice — passes against
    /// BOTH a buggy top-level-mark read and the fixed layers-aware one, so
    /// this fixture mirrors the real lowered shape instead of a convenient
    /// flat one.
    fn ribbon_layered_spec(color_field: &str) -> ChartSpec {
        use crate::spec::layer::Layer;
        // Verified live against the real lowering (spec-review cycle-2):
        // `fm.Chart(df).mark_ribbon().encode(x=,y=,y2=,color="v:Q")`'s spec
        // JSON is `{"mark":"point","encoding":{"x":...,"y":...,
        // "color":{"field":"v","type":"quantitative"},"y2":...},
        // "layers":[{"mark":"ribbon","encoding":{"x":...,"y":...,"y2":...}}]}`
        // — `color` is hoisted to the CHART-level encoding (not present on
        // the layer's own encoding at all) and inherited down into the
        // layer at prepare time (`LayerPrepared::from_chart_and_layer`'s
        // `inherit_from`); `mark` at chart level is the `point` placeholder,
        // never `ribbon`. This exact shape is what tripped the reviewer's
        // finding: a top-level-mark read sees `(color: Some, mark: Point)`
        // and wrongly classifies the leaf as a non-line/ribbon consumer.
        let mut spec = scatter_spec(); // mark: Point (the real placeholder), x/y at chart level
        spec.encoding.color = Some(EncodingSpec {
            field: color_field.into(),
            type_: Some(EncDataType::Quantitative),
            ..Default::default()
        });
        spec.encoding.y2 = Some(EncodingSpec { field: "y2".into(), ..Default::default() });
        spec.layers = Some(vec![Layer {
            mark: Mark::Ribbon,
            encoding: Encoding {
                y2: Some(EncodingSpec { field: "y2".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false,
        }]);
        spec
    }

    /// `color_spec()`'s Point-mark twin bound to `field` on a DIFFERENT
    /// column than `"c"` — for the field-keyed exemption tests, which need
    /// a non-line/ribbon sibling whose color field can differ from the
    /// line/ribbon leaf's.
    fn point_color_spec(field: &str, quantitative: bool) -> ChartSpec {
        ChartSpec {
            encoding: Encoding {
                color: Some(EncodingSpec {
                    field: field.into(),
                    type_: Some(if quantitative { EncDataType::Quantitative } else { EncDataType::Nominal }),
                    ..Default::default()
                }),
                ..scatter_spec().encoding
            },
            ..scatter_spec()
        }
    }

    fn xyy2c_batch(xs: &[f64], ys: &[f64], y2s: &[f64], cs: &[f64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("c", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs.to_vec())),
                Arc::new(Float64Array::from(ys.to_vec())),
                Arc::new(Float64Array::from(y2s.to_vec())),
                Arc::new(Float64Array::from(cs.to_vec())),
            ],
        )
        .unwrap()
    }

    fn unsupported_warning_count(warnings: &[RenderWarning]) -> usize {
        warnings
            .iter()
            .filter(|w| matches!(w, RenderWarning::UnsupportedColorScaleOnMark { .. }))
            .count()
    }

    /// `fm.layer(ribbon(color=v:Q), ribbon(color=v:Q))`: both leaves are
    /// desugared `mark_ribbon` (real mark inside `spec.layers`, `point`
    /// placeholder at the top). Neither is a non-line/ribbon consumer, so
    /// NEITHER exempts the other — both must warn and both colorbars must
    /// be suppressed. Before the layers-aware fix, the top-level-mark read
    /// saw `point` for both, wrongly classified BOTH as "non-line/ribbon
    /// consumers", and exempted the whole group (spec-review cycle-2
    /// finding: live `fm.layer(ribbon(v:Q), ribbon(v:Q)).to_svg()` produced
    /// 0 warnings and TWO surviving colorbars).
    #[test]
    fn overlay_two_ribbon_leaves_sharing_continuous_color_both_warn_no_colorbars() {
        let spec = ribbon_layered_spec("c"); // matches xyy2c_batch's "c" column
        let batch = || xyy2c_batch(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], &[11.0, 21.0, 31.0, 41.0], &[0.0, 5.0, 10.0, 15.0]);
        let h0 = LeafHold { spec: spec.clone(), batch: batch(), ..hold() };
        let h1 = LeafHold { spec: spec.clone(), batch: batch(), ..hold() };
        let tree = composite(
            CompositeLayout::Overlay,
            vec![leaf_node_with(spec.clone(), 0), leaf_node_with(spec, 1)],
        );
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(unsupported_warning_count(&warnings), 2, "both ribbon leaves must warn: {warnings:?}");
        let gradient_count = scene
            .legend
            .iter()
            .filter(|n| matches!(n, SceneNode::Raw { svg, .. } if svg.contains("linearGradient")))
            .count();
        assert_eq!(gradient_count, 0, "neither colorbar may survive");
    }

    /// `fm.layer(line(color=v:Q), ribbon(color=v:Q))`: a line and a
    /// (desugared) ribbon share the same field. Both are line/ribbon marks,
    /// so neither exempts the other — both warn.
    #[test]
    fn overlay_line_and_ribbon_leaves_sharing_continuous_color_both_warn() {
        let line_spec = line_color_spec();
        let ribbon_spec = ribbon_layered_spec("c"); // matches line_color_spec()'s field ("c", via color_spec())
        let line_batch = xyc_batch(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], &[0.0, 5.0, 10.0, 15.0]);
        let ribbon_batch = xyy2c_batch(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], &[11.0, 21.0, 31.0, 41.0], &[0.0, 5.0, 10.0, 15.0]);
        let h0 = LeafHold { spec: line_spec.clone(), batch: line_batch, ..hold() };
        let h1 = LeafHold { spec: ribbon_spec.clone(), batch: ribbon_batch, ..hold() };
        let tree = composite(
            CompositeLayout::Overlay,
            vec![leaf_node_with(line_spec, 0), leaf_node_with(ribbon_spec, 1)],
        );
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (_scene, warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(
            unsupported_warning_count(&warnings), 2,
            "both line and ribbon must warn — neither is a non-line/ribbon consumer, so \
             neither can exempt the other (a top-level-mark misread of the ribbon leaf's \
             placeholder `point` mark would wrongly exempt both): {warnings:?}"
        );
    }

    /// The reviewer's live probe for the field-keyed correction:
    /// `fm.layer(line(color=v:Q), point(color=g:N))` — the point sibling
    /// binds a DIFFERENT field (`g`, nominal) than the line's (`v`,
    /// continuous). It must NOT exempt the line's inert `v` channel — that
    /// sibling never paints `v` at all — while `g`'s own categorical legend
    /// renders untouched (ordinary per-leaf legend building, unaffected by
    /// this fix either way).
    #[test]
    fn overlay_line_and_point_on_different_fields_warns_re_v_keeps_g_legend() {
        let line_spec = line_color_spec(); // color field "c" (color_spec()'s field)
        let point_spec = point_color_spec("g", false); // nominal, different field
        let line_batch = xyc_batch(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], &[0.0, 5.0, 10.0, 15.0]);
        let point_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let point_batch = RecordBatch::try_new(
            point_schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(arrow::array::StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap();
        let h0 = LeafHold { spec: line_spec.clone(), batch: line_batch, ..hold() };
        let h1 = LeafHold { spec: point_spec.clone(), batch: point_batch, ..hold() };
        let tree = composite(
            CompositeLayout::Overlay,
            vec![leaf_node_with(line_spec, 0), leaf_node_with(point_spec, 1)],
        );
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        match warnings.iter().find(|w| matches!(w, RenderWarning::UnsupportedColorScaleOnMark { .. })) {
            Some(RenderWarning::UnsupportedColorScaleOnMark { marks, .. }) => {
                assert_eq!(marks, &vec!["line".to_string()]);
            }
            _ => panic!("line's inert `v`-equivalent field must still warn even with an unrelated-field sibling: {warnings:?}"),
        }
        // g's own categorical legend must still render (ordinary per-leaf
        // legend building — the point leaf's own field, untouched by the
        // exemption logic either way).
        let has_g_legend_entry = scene
            .legend
            .iter()
            .any(|n| matches!(n, SceneNode::Text { content, .. } if content == "a" || content == "b"));
        assert!(has_g_legend_entry, "g's own categorical legend entries must render: {:?}", scene.legend);
    }

    /// Field-keying isolated from scale-kind: `layer(line(v:Q), point(w:Q))`
    /// — the sibling is ALSO Numeric-keyed (unlike the reviewer's `g:N`
    /// probe above, which conflates "different field" with "different scale
    /// kind" and so cannot alone catch a field-keying regression — an
    /// `is_numeric`-only check on the sibling would coincidentally still
    /// reject that probe). Two DIFFERENT numeric fields must still warn:
    /// the point sibling never paints `v`, so it cannot exempt line's own
    /// inert `v` channel just because SOME numeric scale exists nearby.
    #[test]
    fn overlay_line_and_point_on_different_numeric_fields_still_warns() {
        let line_spec = line_color_spec(); // color field "c"
        let point_spec = point_color_spec("w", true); // ALSO quantitative, different field
        let line_batch = xyc_batch(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], &[0.0, 5.0, 10.0, 15.0]);
        let point_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("w", DataType::Float64, false),
        ]));
        let point_batch = RecordBatch::try_new(
            point_schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0, 400.0])),
            ],
        )
        .unwrap();
        let h0 = LeafHold { spec: line_spec.clone(), batch: line_batch, ..hold() };
        let h1 = LeafHold { spec: point_spec.clone(), batch: point_batch, ..hold() };
        let tree = composite(
            CompositeLayout::Overlay,
            vec![leaf_node_with(line_spec, 0), leaf_node_with(point_spec, 1)],
        );
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (_scene, warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(
            unsupported_warning_count(&warnings), 1,
            "point's own numeric field `w` must not exempt line's unrelated numeric field `c`: {warnings:?}"
        );
    }

    #[test]
    fn colorbar_raw_clip_ids_uniquified_per_leaf_end_to_end() {
        // Two leaves each render an independent continuous-color colorbar — a
        // real `SceneNode::Raw` fragment carrying a bare `id="ferrum-colorbar-0"`
        // gradient def (legend.rs always starts each leaf's own counter at 0,
        // with no knowledge of sibling leaves). Composing them must not let
        // leaf 1's def collide with leaf 0's: `render_composite_scene`'s
        // per-leaf `uniquify_scene_raw_clips` call (this file, ~line 207) is the
        // only thing standing between this and a broken merged SVG (two
        // `<linearGradient id="ferrum-colorbar-0">` defs, the second silently
        // shadowing the first so BOTH rects end up painted from leaf 1's
        // gradient). Exercises the exact seam the spec-verdict flagged as
        // reachable-but-untested (no prior test's leaf ever emitted a Raw node).
        let spec = color_spec();
        let h0 = LeafHold {
            spec: spec.clone(),
            batch: xyc_batch(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0], &[0.0, 5.0, 10.0]),
            ..hold()
        };
        let h1 = LeafHold {
            spec,
            batch: xyc_batch(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0], &[0.0, 5.0, 10.0]),
            ..hold()
        };

        let tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let raw_svgs: Vec<&str> = scene
            .legend
            .iter()
            .filter_map(|n| match n {
                SceneNode::Raw { svg, .. } => Some(svg.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            raw_svgs.len() >= 2,
            "expected one colorbar raw fragment per leaf, got {}",
            raw_svgs.len()
        );

        // Both leaves' gradient DEF and its consuming `url(#...)` REFERENCE must
        // be rewritten together, each under its own leaf-indexed prefix — proving
        // uniquify_clip_ids rewrote both occurrences, not just the def.
        assert!(
            raw_svgs
                .iter()
                .any(|s| s.contains(r#"id="cell0-ferrum-colorbar-0""#)
                    && s.contains("url(#cell0-ferrum-colorbar-0)")),
            "leaf 0's colorbar def+ref must be namespaced cell0-...: {raw_svgs:?}"
        );
        assert!(
            raw_svgs
                .iter()
                .any(|s| s.contains(r#"id="cell1-ferrum-colorbar-0""#)
                    && s.contains("url(#cell1-ferrum-colorbar-0)")),
            "leaf 1's colorbar def+ref must be namespaced cell1-...: {raw_svgs:?}"
        );
        // No collision survives: the bare (un-namespaced) id must not leak
        // through, which is exactly what would happen if uniquification were a
        // no-op (the historical gap this test closes).
        assert!(
            !raw_svgs
                .iter()
                .any(|s| s.contains(r#"id="ferrum-colorbar-0""#)),
            "un-namespaced colorbar id leaked into the merged scene: {raw_svgs:?}"
        );
    }

    /// A minimal pre-rendered inset body: one `clipPath` def + its consuming
    /// `clip-path` reference, exactly the shape `render/inset.rs::build_inset_nodes`
    /// receives as `InsetSpec.svg` (a fully independent chart rendered on its
    /// own, so it always numbers its own ids from zero).
    fn inset_svg_fixture() -> String {
        concat!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="150">"#,
            r#"<defs><clipPath id="ferrum-clip-0"><rect width="200" height="150"/></clipPath></defs>"#,
            r#"<g clip-path="url(#ferrum-clip-0)"><circle cx="100" cy="75" r="50"/></g>"#,
            r#"</svg>"#,
        )
        .to_string()
    }

    fn inset_chart_config(svgs: &[String]) -> ChartConfig {
        use crate::render::chart_config::{InsetSpec, StructuralSpec};
        ChartConfig {
            structural: svgs
                .iter()
                .map(|svg| {
                    StructuralSpec::Inset(InsetSpec {
                        svg: svg.clone(),
                        bounds: [0.6, 0.1, 0.95, 0.55],
                        border: true,
                        border_color: "#999999".to_string(),
                        border_dash: None,
                        background: None,
                        shadow: false,
                        connect_to: None,
                        connect_style: "lines".to_string(),
                    })
                })
                .collect(),
            ..ChartConfig::default()
        }
    }

    fn raw_svgs_in(scene: &SceneGraph, panel_idx: usize) -> Vec<&str> {
        scene.panels[panel_idx]
            .annotations
            .iter()
            .filter_map(|n| match n {
                SceneNode::Raw { svg, .. } => Some(svg.as_str()),
                _ => None,
            })
            .collect()
    }

    /// S4 regression (closing rust-design-review finding): two composite
    /// leaves EACH embedding an inset must not collide with each other once
    /// merged. `render/inset.rs::build_inset_nodes` namespaces each leaf's
    /// inset under its own `inset_idx` (`inset0-ferrum-clip-0`), but that
    /// counter restarts at 0 independently per leaf (`scene_build.rs`'s
    /// `build_structural_nodes` is called once per leaf's own `build_scene`).
    /// Before the fix, `uniquify_scene_raw_clips`'s per-leaf `cellN` pass only
    /// matched the BARE `id="ferrum-clip-` literal, so it silently skipped
    /// the already-`inset0-`-prefixed ids — leaf 0's and leaf 1's insets both
    /// stayed `id="inset0-ferrum-clip-0"` and collided in the merged
    /// document, reopening the exact defect class the inset fix was written
    /// to close, just one level up.
    #[test]
    fn two_leaves_each_embedding_an_inset_stay_disjoint_end_to_end() {
        let fixture = inset_svg_fixture();
        let h0 = LeafHold {
            chart_config: inset_chart_config(std::slice::from_ref(&fixture)),
            ..hold()
        };
        let h1 = LeafHold {
            chart_config: inset_chart_config(std::slice::from_ref(&fixture)),
            ..hold()
        };

        let tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let panel0_raw = raw_svgs_in(&scene, 0);
        let panel1_raw = raw_svgs_in(&scene, 1);
        assert_eq!(panel0_raw.len(), 1, "leaf 0 must embed exactly one inset fragment");
        assert_eq!(panel1_raw.len(), 1, "leaf 1 must embed exactly one inset fragment");

        // Outermost-first composition: the leaf-level `cellN` pass runs LAST
        // (after `build_inset_nodes`'s embed-time `inset0-` pass), so it ends
        // up leftmost.
        assert!(
            panel0_raw[0].contains(r#"id="cell0-inset0-ferrum-clip-0""#)
                && panel0_raw[0].contains("url(#cell0-inset0-ferrum-clip-0)"),
            "leaf 0's inset def+ref must compose cell0- in front of inset0-: {panel0_raw:?}"
        );
        assert!(
            panel1_raw[0].contains(r#"id="cell1-inset0-ferrum-clip-0""#)
                && panel1_raw[0].contains("url(#cell1-inset0-ferrum-clip-0)"),
            "leaf 1's inset def+ref must compose cell1- in front of inset0-: {panel1_raw:?}"
        );
        assert_ne!(
            panel0_raw[0], panel1_raw[0],
            "the two leaves' inset fragments must not be byte-identical after namespacing"
        );
        // Neither the bare id nor the single-layer inset0- id (i.e. as if the
        // cell pass had skipped it) may survive into the merged document.
        for raw in panel0_raw.iter().chain(&panel1_raw) {
            assert!(!raw.contains(r#"id="ferrum-clip-0""#), "un-namespaced id leaked: {raw}");
            assert!(
                !raw.contains(r#"id="inset0-ferrum-clip-0""#),
                "cell pass skipped an already-namespaced inset id: {raw}"
            );
        }
    }

    /// Control for the test above: TWO insets inside the SAME (single, non-
    /// composite) chart must already stay disjoint from each other via their
    /// own `inset_idx` counter — no composite/cell-level pass involved at
    /// all. Pins that this simpler, older-fixed case still works after the
    /// `uniquify_clip_ids_with_prefix` rewrite above (composing instead of
    /// skipping) — it never depended on the skip behavior in the first place,
    /// since two insets in one chart get DIFFERENT `inset_idx` values
    /// (`inset0-...`, `inset1-...`), so neither ever collides with the other
    /// pre-cell-pass.
    #[test]
    fn two_insets_in_one_chart_stay_disjoint_without_any_composite_pass() {
        let fixture = inset_svg_fixture();
        let h = LeafHold {
            chart_config: inset_chart_config(&[fixture.clone(), fixture]),
            ..hold()
        };
        let (scene, _warnings, _bundle) = render_leaf(&leaf_input(&h, 300.0, 200.0), None).unwrap();

        let raw = raw_svgs_in(&scene, 0);
        assert_eq!(raw.len(), 2, "expected one Raw fragment per inset");
        assert!(
            raw.iter().any(|s| s.contains(r#"id="inset0-ferrum-clip-0""#)),
            "first inset must be namespaced inset0-: {raw:?}"
        );
        assert!(
            raw.iter().any(|s| s.contains(r#"id="inset1-ferrum-clip-0""#)),
            "second inset must be namespaced inset1-: {raw:?}"
        );
        assert_ne!(raw[0], raw[1], "the two insets' fragments must not be byte-identical");
    }

    #[test]
    fn shared_channel_unknown_field_surfaces_through_render_composite_scene() {
        // A leaf's spec encodes x on a field its batch does not carry, with the
        // tree sharing x — a semantically-broken composite tree that must
        // surface a typed error end-to-end through `render_composite_scene`'s
        // `?`-based wrapping, never panic or silently render garbage.
        //
        // NOTE on which typed error actually surfaces here (investigated, not
        // guessed): `render_composite_scene`'s pass 1 calls
        // `prepare::prepare_render_inputs` UNCONDITIONALLY for every leaf,
        // BEFORE `resolve_composite_scales` ever runs (this file, ~line 170).
        // That per-leaf prepare independently builds a provisional axis scale
        // for layer 0's encoding via `build_axis_scale`, which calls the exact
        // same `locate_field(&enc.field, primary_batch, transform_outputs)`
        // composite.rs's `leaf_channel_domain` calls (composite.rs:410,
        // positional.rs:87) — same field, same batch, same transform_outputs.
        // So a genuinely-missing x/y field always fails at the per-leaf
        // `prepare_render_inputs` step FIRST, as `RenderError::UnknownColumn`
        // wrapped into `CompositeRenderError::LeafRender { index, .. }` — never
        // reaching `resolve_composite_scales`'s own `CompositeResolveError::
        // UnknownField` (which has no leaf index at all, only channel+field).
        // `CompositeResolveError::UnknownField` therefore cannot be produced by
        // `render_composite_scene` for a missing-field leaf under the current
        // pass ordering; it remains reachable only via `resolve_composite_
        // scales` called directly (composite.rs's own Task-4 unit test,
        // `unknown_field_errors_with_channel_and_field`, which hand-builds
        // `LeafResolveInput` and skips the prepare gate). This is not a
        // regression — the error that DOES surface is strictly more precise
        // (it names the offending leaf index) — but it means gap #2's brief
        // literally asking for `CompositeResolveError::UnknownField` through
        // this entry point cannot be satisfied as written; this test instead
        // pins the error that IS reachable, with the same discriminating intent
        // (a semantically-broken tree surfaces a typed, indexed error, not a
        // panic or a silent skip). Flagged to the orchestrator in the gap-fix
        // addendum.
        let mut bad_spec = scatter_spec();
        bad_spec.encoding.x = Some(EncodingSpec {
            field: "missing".into(),
            ..Default::default()
        });
        let h0 = LeafHold {
            spec: bad_spec,
            ..hold()
        };
        let h1 = hold();

        let mut tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.x = crate::layout::facet::ResolveMode::Shared;
        }
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let err = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap_err();
        match err {
            CompositeRenderError::LeafRender {
                kind,
                index,
                ref source,
            } => {
                assert_eq!(kind, "leaf");
                assert_eq!(index, 0, "the broken leaf (index 0) must be pinpointed");
                assert!(
                    matches!(source, RenderError::UnknownColumn { name } if name == "missing"),
                    "expected UnknownColumn(\"missing\"), got {source:?}"
                );
            }
            ref other => panic!("expected LeafRender{{..UnknownColumn}}, got {other:?}"),
        }
    }

    #[test]
    fn overlay_end_to_end_preserves_child_declaration_order_as_zorder() {
        // Overlay places every child at the SAME rect (plan_overlay), so
        // position can never discriminate z-order — the z-order claim ("later
        // child drawn on top") lives entirely in the merged scene's PANEL
        // VECTOR ORDER, since SVG paints elements in document order
        // (painter's algorithm: later elements win visually). This proves
        // render_composite_scene's merge preserves child-declaration order
        // through `merge_children`'s `for (child, t) in children.into_iter()...`
        // loop, end to end — `overlay_places_all_at_origin_bbox_is_max` only
        // covers `plan_overlay`'s pure layout math, not this ordering property.
        let tree = composite(CompositeLayout::Overlay, vec![leaf_node(0), leaf_node(1)]);
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 2);
        // Panel ids are assigned by `renumber_panels` in pre-order BEFORE
        // placement/merge, so id 0 is unambiguously the first-declared child
        // (painted first / on the bottom) and id 1 the second-declared child
        // (painted last / on top) — the vec position IS the z-order.
        assert_eq!(
            scene.panels[0].id, 0,
            "first-declared child occupies slot 0 (bottom)"
        );
        assert_eq!(
            scene.panels[1].id, 1,
            "second-declared child occupies slot 1 (drawn on top)"
        );
        // Both share the identical overlay rect, confirming this genuinely
        // exercises the overlay (same-rect) case rather than a linear layout
        // where position alone would already prove ordering.
        assert_eq!(
            scene.panels[0].plot_area, scene.panels[1].plot_area,
            "overlay children share one rect; only vec order encodes z-order"
        );
    }

    // -- P2 (2026-08-27 findings): overlay chrome suppression -----------------

    #[test]
    fn overlay_merge_suppresses_chrome_on_children_after_the_first() {
        // P2 (design review, 2026-08-27): a shared-resolve overlay used to
        // render every child leaf as a full standalone panel (its own grid +
        // axes + chart title), so a 2-layer overlay emitted two full sets of
        // chrome sharing one rect. Only child 0's grid/axes must survive the
        // merge; both children's MARKS must still be present — layering
        // content is the whole point, only the duplicated chrome is dropped.
        let tree = composite(CompositeLayout::Overlay, vec![leaf_node(0), leaf_node(1)]);
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(
            scene.panels.len(),
            2,
            "both leaves still contribute a panel"
        );
        assert!(
            !scene.panels[0].axes.is_empty(),
            "child 0's axes must survive the merge"
        );
        assert!(
            !scene.panels[0].grid.is_empty(),
            "child 0's grid must survive the merge"
        );
        assert!(
            scene.panels[1].axes.is_empty(),
            "child 1's axes must be dropped by the overlay merge seam"
        );
        assert!(
            scene.panels[1].grid.is_empty(),
            "child 1's grid must be dropped by the overlay merge seam"
        );
        assert!(
            !scene.panels[0].marks.is_empty() && !scene.panels[1].marks.is_empty(),
            "chrome suppression must not drop either layer's mark content"
        );
    }

    #[test]
    fn overlay_merge_keeps_only_child_0_title_when_children_differ() {
        // The scene-level-title half of the P2 merge contract (design §6):
        // two overlay children carrying DIFFERENT chart titles (the shape a
        // LayerChart with mismatched per-layer titles hits) must merge to
        // exactly child 0's title text, not both overprinting at one origin.
        use crate::spec::title::TitleSpec;

        let mut spec0 = scatter_spec();
        spec0.title = Some(TitleSpec {
            text: "Left Y".into(),
            ..Default::default()
        });
        let mut spec1 = scatter_spec();
        spec1.title = Some(TitleSpec {
            text: "Right Y".into(),
            ..Default::default()
        });

        let h0 = LeafHold {
            spec: spec0.clone(),
            ..hold()
        };
        let h1 = LeafHold {
            spec: spec1.clone(),
            ..hold()
        };
        let tree = composite(
            CompositeLayout::Overlay,
            vec![
                CompositeNode::Leaf {
                    spec: Box::new(spec0),
                    data: 0,
                    label: None,
                },
                CompositeNode::Leaf {
                    spec: Box::new(spec1),
                    data: 1,
                    label: None,
                },
            ],
        );
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let title_text: Vec<&str> = scene
            .title
            .iter()
            .filter_map(|n| match n {
                SceneNode::Text { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            title_text,
            vec!["Left Y"],
            "merged overlay scene must carry only child 0's title text, got {title_text:?}"
        );
    }

    /// Build a 2-leaf shared-x/y Overlay tree from two specs.
    fn overlay_tree(spec0: ChartSpec, spec1: ChartSpec) -> CompositeNode {
        let mut tree = composite(
            CompositeLayout::Overlay,
            vec![
                CompositeNode::Leaf {
                    spec: Box::new(spec0),
                    data: 0,
                    label: None,
                },
                CompositeNode::Leaf {
                    spec: Box::new(spec1),
                    data: 1,
                    label: None,
                },
            ],
        );
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.x = crate::layout::facet::ResolveMode::Shared;
            resolve.y = crate::layout::facet::ResolveMode::Shared;
        }
        tree
    }

    #[test]
    fn plan_overlay_groups_names_non_primary_leaves_only_under_all_leaf_overlays() {
        // The map both halves of #89A read: rect sharing and chrome dedup.
        let overlay = composite(
            CompositeLayout::Overlay,
            vec![leaf_node(0), leaf_node(1), leaf_node(2)],
        );
        assert_eq!(
            plan_overlay_groups(&overlay, 3),
            vec![None, Some(0), Some(0)],
            "every non-primary leaf names the group leader"
        );

        // Nested composite child: the whole node is left alone.
        let nested = composite(
            CompositeLayout::Overlay,
            vec![
                composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]),
                leaf_node(2),
            ],
        );
        assert_eq!(plan_overlay_groups(&nested, 3), vec![None; 3]);

        // Singleton overlay: nothing to share with.
        let singleton = composite(CompositeLayout::Overlay, vec![leaf_node(0)]);
        assert_eq!(plan_overlay_groups(&singleton, 1), vec![None]);

        // Non-overlay layouts are never grouped, and an overlay nested inside
        // one is numbered from its own first leaf.
        let mixed = composite(
            CompositeLayout::Hconcat,
            vec![
                leaf_node(0),
                composite(CompositeLayout::Overlay, vec![leaf_node(1), leaf_node(2)]),
            ],
        );
        assert_eq!(plan_overlay_groups(&mixed, 3), vec![None, None, Some(1)]);
    }

    #[test]
    fn impose_shared_overlay_rects_writes_the_intersection_to_every_member() {
        // Unit-level statement of the shared-rect contract: BOTH members —
        // leader included — receive one identical rect, and it is the
        // intersection (the color leaf's legend gutter narrows it), not the
        // leader's own region.
        let tree = overlay_tree(color_spec(), scatter_spec());
        let h0 = LeafHold {
            spec: color_spec(),
            batch: xyc_batch(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0], &[1.0, 2.0, 3.0]),
            ..hold()
        };
        let h1 = hold();
        let leaves = [leaf_input(&h0, 400.0, 300.0), leaf_input(&h1, 400.0, 300.0)];

        let mut groups = plan_overlay_groups(&tree, 2);
        let mut contexts = vec![LeafScaleContext::default(); 2];
        let natural: Vec<_> = (0..2)
            .map(|i| natural_plot_region(&leaves[i], &contexts[i]).expect("leaf lays out"))
            .collect();
        assert_ne!(
            natural[0], natural[1],
            "the two leaves' natural regions must differ, or this test proves nothing"
        );

        let mut warnings: Vec<RenderWarning> = Vec::new();
        impose_shared_overlay_rects(&leaves, &mut groups, &mut contexts, &mut warnings);

        assert!(
            warnings.is_empty(),
            "an equalized group is the normal path and must warn about nothing: {warnings:?}"
        );
        let expected = natural[0].intersect(natural[1]);
        assert_eq!(contexts[0].imposed_plot_region, Some(expected));
        assert_eq!(
            contexts[1].imposed_plot_region, contexts[0].imposed_plot_region,
            "the leader shares the group's rect too — it is not the reference"
        );
        assert!(
            expected.w < natural[1].w,
            "the intersection keeps the color leaf's legend gutter for the whole group"
        );
        assert_eq!(
            groups,
            vec![None, Some(0)],
            "an equalized group survives in the map the merge seam reads"
        );
    }

    #[test]
    fn impose_shared_overlay_rects_clears_the_group_it_cannot_equalize() {
        // The coupling itself, stated at the unit level (spec §4.2): when the
        // intersection degenerates, the pre-pass imposes nothing AND removes
        // the group from the map `build_placed` reads — so suppression cannot
        // fire for a leaf that kept its own geometry.
        let tree = composite(
            CompositeLayout::Overlay,
            vec![color_leaf(0), color_leaf(1)],
        );
        let labels = [
            "a-very-long-category-label-one",
            "a-very-long-category-label-two",
            "a-very-long-category-label-one",
        ];
        let h0 = color_hold_oriented(&labels, LegendOrient::Left);
        let h1 = color_hold_oriented(&labels, LegendOrient::Right);
        let leaves = [leaf_input(&h0, 400.0, 300.0), leaf_input(&h1, 400.0, 300.0)];

        let mut groups = plan_overlay_groups(&tree, 2);
        assert_eq!(groups, vec![None, Some(0)], "planned as one group first");
        let mut contexts = vec![LeafScaleContext::default(); 2];
        let mut warnings: Vec<RenderWarning> = Vec::new();
        impose_shared_overlay_rects(&leaves, &mut groups, &mut contexts, &mut warnings);

        assert_eq!(
            groups,
            vec![None, None],
            "the un-equalized group must be cleared from the merge seam's map"
        );
        // The degradation is visible (doubled chrome) but hard to attribute,
        // so it is announced through the normal render-warning channel.
        assert_eq!(
            warnings,
            vec![RenderWarning::OverlayGuttersDiverged { layers: 2 }],
            "a collapsed intersection must warn, naming the group size"
        );
        assert!(contexts.iter().all(|c| c.imposed_plot_region.is_none()));
        assert!(
            contexts.iter().all(|c| !c.suppress_chart_title),
            "the title suppression that anticipated the dedup must be undone too"
        );
    }

    #[test]
    fn overlay_shared_rect_equalizes_heterogeneous_chrome_geometry() {
        // GH #89A: every leaf of an overlay group lays out against ONE shared
        // plot rect — the intersection of the leaves' natural regions. A
        // color-encoded layer 0 reserves a legend gutter a plain layer 1 does
        // not, so their STANDALONE `plot_area`s diverge even though
        // composite-shared x/y forces identical DOMAINS (the reviewer's pre-fix
        // repro: panel widths ~520 vs ~567, a ~47px cx divergence for the same
        // datum). Both leaves here share the SAME x/y data
        // (`[1,2,3]`/`[10,20,30]`) so any pixel divergence can only come from
        // the legend gutter, not the domain.
        let tree = overlay_tree(color_spec(), scatter_spec());
        let h0 = LeafHold {
            spec: color_spec(),
            batch: xyc_batch(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0], &[1.0, 2.0, 3.0]),
            ..hold()
        };
        let h1 = hold(); // scatter_spec, xy_batch same x/y domain, no color → no legend gutter.
        let leaves = [leaf_input(&h0, 400.0, 300.0), leaf_input(&h1, 400.0, 300.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(
            scene.panels.len(),
            2,
            "both leaves still contribute a panel"
        );
        assert_eq!(
            scene.panels[0].plot_area, scene.panels[1].plot_area,
            "both children must lay out against the group's one shared rect"
        );

        // Pixel-parity: the same datum (x=3, the third/last point in both
        // batches) must land at the identical cx in both panels — the exact
        // discriminating check (cx 569.265 vs 616.0) the reviewer's repro used.
        let cx_of = |panel: &ferrum_scene::Panel| -> f64 {
            let batch = panel.marks.first().expect("point mark batch present");
            match batch.nodes.get(2).expect("third point node present") {
                SceneNode::Circle { cx, .. } => *cx,
                other => panic!("expected a Circle point node, got {other:?}"),
            }
        };
        let cx0 = cx_of(&scene.panels[0]);
        let cx1 = cx_of(&scene.panels[1]);
        assert!(
            (cx0 - cx1).abs() < 1e-9,
            "same datum (x=3) must render at the same cx in both panels, \
             got panel0 cx={cx0}, panel1 cx={cx1}"
        );

        assert!(
            !scene.panels[0].axes.is_empty(),
            "child 0's axes must survive"
        );
        assert!(
            scene.panels[1].axes.is_empty(),
            "child 1's duplicate axes must be dropped"
        );
    }

    #[test]
    fn overlay_totality_non_primary_legend_gutter_narrows_the_shared_rect() {
        // GH #89A, former refusal door 1 (per-leaf legend). The legend sits on
        // the NON-PRIMARY child, the mirrored direction of the test above:
        // pre-#89A this refused imposition (a wider child-0 rect would have run
        // child 1's marks across its own unmoved legend box) and therefore
        // refused dedup too, leaving two full chromes. Now the shared rect is
        // the INTERSECTION, so child 0 adopts child 1's legend gutter, child 1's
        // duplicate chrome drops, AND the legend still renders.
        let tree = overlay_tree(scatter_spec(), color_spec());
        let h0 = hold();
        let h1 = LeafHold {
            spec: color_spec(),
            batch: xyc_batch(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0], &[1.0, 2.0, 3.0]),
            ..hold()
        };
        let leaves = [leaf_input(&h0, 400.0, 300.0), leaf_input(&h1, 400.0, 300.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 2);
        assert_eq!(
            scene.panels[0].plot_area, scene.panels[1].plot_area,
            "both children lay out against the group's one shared rect"
        );
        assert!(
            !scene.panels[0].axes.is_empty() && !scene.panels[0].grid.is_empty(),
            "child 0's chrome is the group's chrome and must survive"
        );
        assert!(
            scene.panels[1].axes.is_empty() && scene.panels[1].grid.is_empty(),
            "the legend-bearing child's duplicate chrome must be dropped (no refusal door)"
        );
        assert!(
            !scene.legend.is_empty(),
            "the non-primary child's own legend is content, not chrome — it must still render"
        );

        // The shared rect is the INTERSECTION, not child 0's rect: against a
        // legend-free overlay of the same two viewports, the rect here must be
        // narrower by child 1's legend gutter. This is what makes suppression
        // sound — child 1's legend box sits outside the rect BOTH children's
        // marks use, so nothing renders across it.
        let plain_tree = overlay_tree(scatter_spec(), scatter_spec());
        let p0 = hold();
        let p1 = hold();
        let plain_leaves = [leaf_input(&p0, 400.0, 300.0), leaf_input(&p1, 400.0, 300.0)];
        let (plain, _) =
            render_composite_scene(&plain_tree, &plain_leaves, &ThemeInputs::default()).unwrap();
        assert!(
            scene.panels[0].plot_area.w < plain.panels[0].plot_area.w,
            "the primary child must adopt the group's largest gutter per side: shared w={} \
             must be narrower than the legend-free overlay's w={}",
            scene.panels[0].plot_area.w,
            plain.panels[0].plot_area.w
        );
    }

    #[test]
    fn overlay_totality_non_primary_above_marks_axis_renders_once() {
        // GH #89A, former refusal door 2 (zindex >= 1 axis). A child whose
        // x-axis carries `zindex >= 1` routes that axis's nodes into the typed
        // `Panel.chrome_above` slot (GH #89B). Pre-#89A the merge seam did not
        // clear that slot, so suppressing the child's `axes`/`grid` would have
        // left a second, stale-position axis visible — the gate refused both,
        // and the overlay kept two full chromes. Now `chrome_above` is cleared
        // with the rest of the chrome: the axis renders exactly once, from the
        // primary child.
        use crate::render::chart_config::AxisStyleSpec;

        let mut zindex_spec = scatter_spec();
        zindex_spec.encoding.x = Some(EncodingSpec {
            field: "x".into(),
            axis: Some(Box::new(AxisStyleSpec {
                zindex: Some(1),
                ..Default::default()
            })),
            ..Default::default()
        });

        let tree = overlay_tree(scatter_spec(), zindex_spec.clone());
        let h0 = hold();
        let h1 = LeafHold {
            spec: zindex_spec.clone(),
            ..hold()
        };
        let leaves = [leaf_input(&h0, 400.0, 300.0), leaf_input(&h1, 400.0, 300.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 2);
        assert!(
            scene.panels[1].axes.is_empty() && scene.panels[1].grid.is_empty(),
            "the non-primary child's below-marks chrome must be dropped"
        );
        assert!(
            scene.panels[1].chrome_above.is_empty(),
            "and so must its ABOVE-marks axis chrome — otherwise the overlay draws a second axis"
        );
        assert!(
            !scene.panels[0].axes.is_empty(),
            "the primary child's axis is the one that renders"
        );
        assert!(
            !scene.panels[0].marks.is_empty() && !scene.panels[1].marks.is_empty(),
            "dedup must not drop either layer's marks"
        );

        // Control: the same leaf as the PRIMARY child really does populate
        // `chrome_above`, so the emptiness asserted above is this seam's
        // clearing and not an absence of above-marks routing.
        let primary_tree = overlay_tree(zindex_spec.clone(), scatter_spec());
        let c0 = LeafHold {
            spec: zindex_spec,
            ..hold()
        };
        let c1 = hold();
        let control_leaves = [leaf_input(&c0, 400.0, 300.0), leaf_input(&c1, 400.0, 300.0)];
        let (control, _) =
            render_composite_scene(&primary_tree, &control_leaves, &ThemeInputs::default())
                .unwrap();
        assert!(
            !control.panels[0].chrome_above.is_empty(),
            "control: an unsuppressed zindex>=1 leaf routes its axis into chrome_above"
        );
    }

    #[test]
    fn overlay_totality_non_primary_below_marks_annotation_survives_dedup() {
        // GH #89A, former refusal door 3 (below-marks annotation). Pre-#89B
        // `scene_build` folded a `z="below_marks"` text annotation into the same
        // bucket that became `Panel.grid`, so the merge seam's
        // `panel.grid.clear()` would have deleted the user's annotation along
        // with the gridlines; the gate refused imposition+dedup for any leaf
        // carrying one. GH #89B gave that content its own typed
        // `Panel.below_marks` slot, which this seam never clears — so the
        // chrome dedups and the annotation survives, both at once.
        use crate::render::annotation::{AnnotationSpec, CoordValue};

        let tree = overlay_tree(scatter_spec(), scatter_spec());
        let cc1 = ChartConfig {
            annotations: vec![AnnotationSpec::Text {
                x: CoordValue::Norm { norm: 0.5 },
                y: CoordValue::Norm { norm: 0.5 },
                text: "below marks note".into(),
                font_size: 14.0,
                color: "#ff0000".into(),
                anchor: "middle".into(),
                baseline: "middle".into(),
                angle: 0.0,
                dx: 0.0,
                dy: 0.0,
                z: "below_marks".into(),
            }],
            ..Default::default()
        };
        let h0 = hold();
        let h1 = LeafHold {
            chart_config: cc1,
            ..hold()
        };
        let leaves = [leaf_input(&h0, 400.0, 300.0), leaf_input(&h1, 400.0, 300.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 2);
        assert_eq!(
            scene.panels[0].plot_area, scene.panels[1].plot_area,
            "both children lay out against the group's one shared rect"
        );
        assert!(
            scene.panels[1].axes.is_empty() && scene.panels[1].grid.is_empty(),
            "the annotation-bearing child's duplicate chrome must be dropped"
        );
        let annotation_survived = scene.panels[1]
            .below_marks
            .iter()
            .any(|n| matches!(n, SceneNode::Text { content, .. } if content == "below marks note"));
        assert!(
            annotation_survived,
            "the below-marks text annotation is content: it must survive in panel.below_marks"
        );
    }

    #[test]
    fn overlay_group_whose_shared_rect_collapses_keeps_every_childs_chrome() {
        // Spec §4.2 "suppression is coupled to imposition" (amended
        // 2026-08-28). Two leaves whose legend gutters sit on OPPOSITE sides
        // of a narrow viewport reserve more than the whole inner width
        // between them, so their natural regions do not overlap and the
        // per-side-max intersection degenerates. The pre-pass declines to
        // impose there — and because it is the single source of truth for
        // BOTH halves, the merge seam must decline to suppress too. Dropping
        // chrome here would be the exact silent mismatch (chrome describing a
        // rect the marks never used) the retired gate existed to prevent.
        let tree = composite(
            CompositeLayout::Overlay,
            vec![color_leaf(0), color_leaf(1)],
        );
        // Labels long enough that each leaf's legend strip hits its
        // half-the-inner-extent cap, so the two plot regions land on opposite
        // halves of the viewport and share no area. Each leaf still lays out
        // healthily on its own (both contribute a panel below) — it is only
        // their INTERSECTION that degenerates.
        let labels = [
            "a-very-long-category-label-one",
            "a-very-long-category-label-two",
            "a-very-long-category-label-one",
        ];
        let h0 = color_hold_oriented(&labels, LegendOrient::Left);
        let h1 = color_hold_oriented(&labels, LegendOrient::Right);
        let leaves = [leaf_input(&h0, 400.0, 300.0), leaf_input(&h1, 400.0, 300.0)];
        let (scene, warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 2);
        // End to end, the degradation reaches the caller's warning channel
        // (`binding::emit_warnings` forwards it to `warnings.warn`).
        assert!(
            warnings.contains(&RenderWarning::OverlayGuttersDiverged { layers: 2 }),
            "the doubled chrome must be announced, not silent: {warnings:?}"
        );
        // The shape really is the collapse case: nothing was imposed, so the
        // two leaves kept their own (non-overlapping) regions.
        assert_ne!(
            scene.panels[0].plot_area, scene.panels[1].plot_area,
            "this test must exercise the collapse path — if the rects are equal the \
             intersection succeeded and it proves nothing"
        );
        for (i, panel) in scene.panels.iter().enumerate() {
            assert!(
                !panel.axes.is_empty(),
                "panel {i} must keep its own chrome: no leaf here laid out against a shared rect"
            );
        }
    }

    #[test]
    fn overlay_non_primary_title_band_does_not_inflate_the_shared_rect() {
        // Spec §4.2, suppression-aware pre-pass (amended 2026-08-28). Child 1
        // carries a chart title the merge seam CLEARS, so reserving its title
        // band into the shared rect would push both children's chrome down by
        // a band nothing is ever drawn in — a phantom top gutter. The
        // suppressed leaf's band must not participate: the group's geometry
        // must equal the same overlay with no title at all.
        use crate::spec::title::TitleSpec;

        let mut titled = scatter_spec();
        titled.title = Some(TitleSpec {
            text: "LAYER2".into(),
            ..Default::default()
        });

        let tree = overlay_tree(scatter_spec(), titled.clone());
        let h0 = hold();
        let h1 = LeafHold {
            spec: titled,
            ..hold()
        };
        let leaves = [leaf_input(&h0, 400.0, 300.0), leaf_input(&h1, 400.0, 300.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let plain_tree = overlay_tree(scatter_spec(), scatter_spec());
        let p0 = hold();
        let p1 = hold();
        let plain_leaves = [leaf_input(&p0, 400.0, 300.0), leaf_input(&p1, 400.0, 300.0)];
        let (plain, _) =
            render_composite_scene(&plain_tree, &plain_leaves, &ThemeInputs::default()).unwrap();

        assert!(
            scene.title.is_empty(),
            "child 1's title is cleared by the merge seam — that is what makes its band phantom"
        );
        assert_eq!(
            scene.panels[0].plot_area, plain.panels[0].plot_area,
            "a title on a NON-PRIMARY layer must not move the group's geometry at all, since \
             the title is never drawn"
        );
        assert_eq!(scene.panels[0].plot_area, scene.panels[1].plot_area);
    }

    #[test]
    fn overlay_primary_title_band_still_reserved_for_the_group() {
        // The other half of the same rule: the PRIMARY child's title IS drawn,
        // so its band must still be reserved — and, through the shared rect,
        // every other layer lays out below it.
        use crate::spec::title::TitleSpec;

        let mut titled = scatter_spec();
        titled.title = Some(TitleSpec {
            text: "LAYER1".into(),
            ..Default::default()
        });

        let tree = overlay_tree(titled.clone(), scatter_spec());
        let h0 = LeafHold {
            spec: titled,
            ..hold()
        };
        let h1 = hold();
        let leaves = [leaf_input(&h0, 400.0, 300.0), leaf_input(&h1, 400.0, 300.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let plain_tree = overlay_tree(scatter_spec(), scatter_spec());
        let p0 = hold();
        let p1 = hold();
        let plain_leaves = [leaf_input(&p0, 400.0, 300.0), leaf_input(&p1, 400.0, 300.0)];
        let (plain, _) =
            render_composite_scene(&plain_tree, &plain_leaves, &ThemeInputs::default()).unwrap();

        let title_text: Vec<&str> = scene
            .title
            .iter()
            .filter_map(|n| match n {
                SceneNode::Text { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(title_text, vec!["LAYER1"], "the primary child's title renders");
        assert!(
            scene.panels[0].plot_area.y > plain.panels[0].plot_area.y,
            "a drawn title reserves a real band: the group's rect must start below it"
        );
        assert_eq!(scene.panels[0].plot_area, scene.panels[1].plot_area);
    }

    #[test]
    fn overlay_with_nested_composite_child_keeps_every_childs_chrome() {
        // The one guarded shape (spec §4.2): an Overlay node with a nested
        // composite child spans more than one leaf per child, so there is no
        // single "this child's plot rect" to intersect or impose — the pre-pass
        // skips it and NO child's chrome is dropped. Unreachable from Python
        // (`LayerChart._composite_tree` rejects a non-leaf layer with a typed
        // ValueError), reachable from a directly-constructed wire spec, which is
        // why the guard exists.
        let inner = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(0), leaf_node(1)],
        );
        let tree = composite(CompositeLayout::Overlay, vec![inner, leaf_node(2)]);
        let h0 = hold();
        let h1 = hold();
        let h2 = hold();
        let leaves = [
            leaf_input(&h0, 300.0, 200.0),
            leaf_input(&h1, 300.0, 200.0),
            leaf_input(&h2, 300.0, 200.0),
        ];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 3);
        for (i, panel) in scene.panels.iter().enumerate() {
            assert!(
                !panel.axes.is_empty(),
                "panel {i} must keep its own chrome under an unequalized Overlay"
            );
        }
    }

    #[test]
    fn hconcat_merge_preserves_every_childs_chrome_with_heterogeneous_titles() {
        // S3 fix (rust-quality-reviewer, 2026-08-27 findings batch): the only
        // Rust-side guard for the byte-identity constraint on non-`Overlay`
        // layouts — an `Hconcat` of two DIFFERENTLY-titled leaves must keep
        // BOTH children's `panel.grid`, `panel.axes`, and BOTH title texts in
        // `scene.title`. This is the negative counterpart to
        // `overlay_merge_suppresses_chrome_on_children_after_the_first`:
        // nothing here should ever be dropped. If `build_placed` ever went
        // back to suppressing chrome unconditionally whenever `i > 0`
        // (instead of gating on proven overlay geometry parity), this would
        // catch it — `title_text` would collapse to `["Panel A"]` only.
        use crate::spec::title::TitleSpec;

        let mut spec0 = scatter_spec();
        spec0.title = Some(TitleSpec {
            text: "Panel A".into(),
            ..Default::default()
        });
        let mut spec1 = scatter_spec();
        spec1.title = Some(TitleSpec {
            text: "Panel B".into(),
            ..Default::default()
        });

        let h0 = LeafHold {
            spec: spec0.clone(),
            ..hold()
        };
        let h1 = LeafHold {
            spec: spec1.clone(),
            ..hold()
        };
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![
                CompositeNode::Leaf {
                    spec: Box::new(spec0),
                    data: 0,
                    label: None,
                },
                CompositeNode::Leaf {
                    spec: Box::new(spec1),
                    data: 1,
                    label: None,
                },
            ],
        );
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 2);
        assert!(
            !scene.panels[0].axes.is_empty(),
            "child 0's axes must be kept under hconcat"
        );
        assert!(
            !scene.panels[0].grid.is_empty(),
            "child 0's grid must be kept under hconcat"
        );
        assert!(
            !scene.panels[1].axes.is_empty(),
            "child 1's axes must be kept under hconcat"
        );
        assert!(
            !scene.panels[1].grid.is_empty(),
            "child 1's grid must be kept under hconcat"
        );

        let title_text: Vec<&str> = scene
            .title
            .iter()
            .filter_map(|n| match n {
                SceneNode::Text { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            title_text,
            vec!["Panel A", "Panel B"],
            "hconcat must keep BOTH children's titles — got {title_text:?}"
        );
    }

    /// GH #89B spec-review remediation (cycle 2): `offset_panel` bakes a
    /// non-primary leaf's geometry into its placed position under a
    /// non-`Overlay` layout — every panel node slot must be in that
    /// translation, or content in an un-translated slot renders at its
    /// pre-placement (solo) coordinates while the panel around it moved.
    /// `below_marks` was omitted when the slot was split out of `panel.grid`
    /// (which WAS translated), so a below-marks text annotation on the
    /// second `hconcat` child rendered stranded at its solo x, landing on
    /// top of child 0 instead of moving with child 1's marks.
    #[test]
    fn hconcat_translates_below_marks_annotation_with_its_panel() {
        use crate::render::annotation::{AnnotationSpec, CoordValue};

        let cc1 = ChartConfig {
            annotations: vec![AnnotationSpec::Text {
                x: CoordValue::Norm { norm: 0.5 },
                y: CoordValue::Norm { norm: 0.5 },
                text: "hconcat below marks".into(),
                font_size: 14.0,
                color: "#ff0000".into(),
                anchor: "middle".into(),
                baseline: "middle".into(),
                angle: 0.0,
                dx: 0.0,
                dy: 0.0,
                z: "below_marks".into(),
            }],
            ..Default::default()
        };
        let h0 = hold();
        let h1 = LeafHold {
            chart_config: cc1,
            ..hold()
        };
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![
                CompositeNode::Leaf {
                    spec: Box::new(scatter_spec()),
                    data: 0,
                    label: None,
                },
                CompositeNode::Leaf {
                    spec: Box::new(scatter_spec()),
                    data: 1,
                    label: None,
                },
            ],
        );
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 2);
        assert!(
            scene.panels[0].below_marks.is_empty(),
            "only child 1 carries the below_marks annotation"
        );
        let ann_x = scene.panels[1]
            .below_marks
            .iter()
            .find_map(|n| match n {
                SceneNode::Text { x, content, .. } if content == "hconcat below marks" => Some(*x),
                _ => None,
            })
            .expect("child 1's below_marks annotation must be present");

        let (p0_lo, p0_hi) = (
            scene.panels[0].plot_area.x,
            scene.panels[0].plot_area.x + scene.panels[0].plot_area.w,
        );
        let (p1_lo, p1_hi) = (
            scene.panels[1].plot_area.x,
            scene.panels[1].plot_area.x + scene.panels[1].plot_area.w,
        );
        assert!(
            p1_lo > p0_hi,
            "prerequisite: hconcat must place child 1 strictly to the right of child 0 \
             (child 0 x∈[{p0_lo},{p0_hi}], child 1 x∈[{p1_lo},{p1_hi}])"
        );
        assert!(
            ann_x >= p1_lo && ann_x <= p1_hi,
            "child 1's below_marks annotation must translate into child 1's plot area \
             (x∈[{p1_lo},{p1_hi}]), not stay at its un-translated solo x={ann_x} \
             (which would land inside child 0's x∈[{p0_lo},{p0_hi}])"
        );
    }

    /// GH #89B spec-review remediation (cycle 2): same omission class as
    /// above, for `chrome_above`. A non-primary hconcat child's `zindex >= 1`
    /// x-axis + gridlines route into `panel.chrome_above` at scene-build
    /// time; `offset_panel` must translate that slot too, or the second
    /// child's above-marks axis collapses onto child 0's x range instead of
    /// moving with child 1's own (translated) axes/marks.
    #[test]
    fn hconcat_translates_chrome_above_axis_with_its_panel() {
        use crate::render::chart_config::AxisStyleSpec;

        let mut spec1 = scatter_spec();
        spec1.encoding.x = Some(EncodingSpec {
            field: "x".into(),
            axis: Some(Box::new(AxisStyleSpec {
                zindex: Some(1),
                ..Default::default()
            })),
            ..Default::default()
        });

        let h0 = hold();
        let h1 = LeafHold {
            spec: spec1.clone(),
            ..hold()
        };
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![
                CompositeNode::Leaf {
                    spec: Box::new(scatter_spec()),
                    data: 0,
                    label: None,
                },
                CompositeNode::Leaf {
                    spec: Box::new(spec1),
                    data: 1,
                    label: None,
                },
            ],
        );
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 2);
        assert!(
            scene.panels[0].chrome_above.is_empty(),
            "only child 1 carries a zindex>=1 axis"
        );
        assert!(
            !scene.panels[1].chrome_above.is_empty(),
            "child 1's above-marks axis+grid must be present in chrome_above — otherwise \
             this test isn't exercising the routing the translation bug hits"
        );

        let chrome_xs: Vec<f64> = scene.panels[1]
            .chrome_above
            .iter()
            .filter_map(|n| match n {
                SceneNode::Line { x1, .. } => Some(*x1),
                _ => None,
            })
            .collect();
        assert!(
            !chrome_xs.is_empty(),
            "expected at least one Line node in child 1's chrome_above"
        );

        let (p0_lo, p0_hi) = (
            scene.panels[0].plot_area.x,
            scene.panels[0].plot_area.x + scene.panels[0].plot_area.w,
        );
        let (p1_lo, p1_hi) = (
            scene.panels[1].plot_area.x,
            scene.panels[1].plot_area.x + scene.panels[1].plot_area.w,
        );
        assert!(
            p1_lo > p0_hi,
            "prerequisite: hconcat must place child 1 strictly to the right of child 0 \
             (child 0 x∈[{p0_lo},{p0_hi}], child 1 x∈[{p1_lo},{p1_hi}])"
        );
        // Gridline endpoints may sit exactly on the plot-area boundary
        // (inclusive), so use an epsilon-widened window rather than a bare
        // `>=`/`<=` against child 1's rect.
        let eps = 1e-6;
        for x in &chrome_xs {
            assert!(
                *x >= p1_lo - eps && *x <= p1_hi + eps,
                "child 1's chrome_above axis/grid must translate into child 1's plot area \
                 (x∈[{p1_lo},{p1_hi}]), found x={x} — un-translated content collapses onto \
                 child 0's x∈[{p0_lo},{p0_hi}]"
            );
        }
    }

    // -- D4c: packed-buffer panel indexing (render_composite_interactive seam)

    /// The `render_composite_interactive` PyO3 entry (Task 5c, `binding.rs`)
    /// wires `render_composite_scene` directly into
    /// `pack_instances::extract_packed_bytes`. `extract_packed_bytes` derives
    /// its packed header's `panel_idx` from `scene.panels.iter_mut().
    /// enumerate()` — so the D4c contract ("panels numbered 0..N flat across
    /// the whole composite scene ... packed headers written at final
    /// positions ... no post-hoc header rewrite exists anywhere") holds
    /// automatically PROVIDED `render_composite_scene` has already renumbered
    /// panels into the one merged `SceneGraph` before extraction runs — which
    /// is exactly what this test pins end to end, using a real >=1000-node
    /// batch so packing actually triggers (a single merged-scene assertion
    /// alone cannot distinguish "renumbered correctly" from "never packed").
    #[test]
    fn packed_headers_use_final_flat_panel_indices_d4c() {
        // Leaf 0: a small (unpacked) 3-point batch. Leaf 1: 1200 points —
        // above `pack_instances::PACK_THRESHOLD` (1000) — so its Point batch
        // is extracted as packed binary instances.
        let n = 1200;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i % 50) as f64).collect();
        let h0 = hold();
        let h1 = LeafHold {
            batch: xy_batch(&xs, &ys),
            ..hold()
        };

        let tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (mut scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 2);
        // Panel 1 (leaf 1) is placed to the right of panel 0 by the hconcat
        // layout — its FINAL composite-space x, not leaf 1's own native
        // (x starting at 0) standalone position.
        let final_panel1_x = scene.panels[1].plot_area.x;
        assert!(
            final_panel1_x > scene.panels[0].plot_area.x,
            "panel 1 must be placed to the right of panel 0 in the merged scene"
        );

        let packed = crate::render::pack_instances::extract_packed_bytes(&mut scene);
        assert!(
            !packed.is_empty(),
            "leaf 1's 1200-point batch must trigger packing"
        );

        // Header layout (pack_instances.rs): [panel_idx: u32][batch_idx: u32]
        // [kind: u32][count: u32][flags: u32], all little-endian.
        let panel_idx = u32::from_le_bytes(packed[0..4].try_into().unwrap());
        let count = u32::from_le_bytes(packed[12..16].try_into().unwrap());
        assert_eq!(
            panel_idx, 1,
            "packed header must name panel 1 (the FINAL flat renumbered index), \
             not leaf 1's own leaf-local panel 0 — proving no post-hoc header \
             patch is needed (D4c)"
        );
        assert_eq!(
            count, n as u32,
            "packed instance count must match the source batch size"
        );

        // The header's panel_idx indexes directly into the merged scene's
        // panel vec at the SAME position whose plot_area was asserted above —
        // one flat namespace by construction, not a per-leaf offset table.
        assert_eq!(
            scene.panels[panel_idx as usize].plot_area.x, final_panel1_x,
            "packed header's panel_idx must resolve to the panel carrying the final placement"
        );
    }

    // -- Task 5d: per-leaf binding + per-child labels -------------------------

    /// Radius of the first `Circle` mark node in a panel (point marks render as
    /// circles; `point_size` maps to `sqrt(point_size/PI)`, so a per-leaf
    /// `point_size` override is observable here).
    fn first_circle_radius(panel: &Panel) -> f64 {
        for batch in &panel.marks {
            for node in &batch.nodes {
                if let SceneNode::Circle { r, .. } = node {
                    return *r;
                }
            }
        }
        panic!("panel has no circle mark node");
    }

    #[test]
    fn per_leaf_theme_applied_distinctly_per_leaf() {
        // Two hconcat leaves rendered under two DIFFERENT themes must carry the
        // distinct per-leaf styling end to end — proving the composite core
        // threads each leaf's own `CompositeLeafInput::theme`, never a single
        // collapsed value. `point_size` -> circle radius is the observable.
        let mut theme0 = ThemeInputs::default();
        theme0.sizes.point_size = 12.0;
        let mut theme1 = ThemeInputs::default();
        theme1.sizes.point_size = 300.0;
        let h0 = LeafHold {
            theme: theme0,
            ..hold()
        };
        let h1 = LeafHold {
            theme: theme1,
            ..hold()
        };

        let tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let r0 = first_circle_radius(&scene.panels[0]);
        let r1 = first_circle_radius(&scene.panels[1]);
        assert!(r0 > 0.0 && r1 > 0.0, "both leaves must render circle marks");
        assert!(
            (r0 - r1).abs() > 1e-6,
            "distinct per-leaf point_size must yield distinct radii: r0={r0} r1={r1}"
        );
        // Discriminator: leaf 1's larger point_size must render the larger radius,
        // proving the mapping applied per leaf (not one shared theme).
        assert!(
            r1 > r0,
            "larger per-leaf point_size must render the larger radius"
        );
    }

    /// A labeled leaf node — a per-child label attached to an otherwise-standard
    /// scatter leaf.
    fn labeled_leaf(data: usize, label: &str) -> CompositeNode {
        CompositeNode::Leaf {
            spec: Box::new(scatter_spec()),
            data,
            label: Some(label.to_string()),
        }
    }

    /// The `(x, y, content)` of the first title text node whose content equals
    /// `want`, if present.
    fn find_label(scene: &SceneGraph, want: &str) -> Option<(f64, f64)> {
        scene.title.iter().find_map(|n| match n {
            SceneNode::Text { x, y, content, .. } if content == want => Some((*x, *y)),
            _ => None,
        })
    }

    /// The `(font_size, color)` of the first title text node whose content
    /// equals `want`, if present.
    fn find_label_style(scene: &SceneGraph, want: &str) -> Option<(f64, ferrum_scene::Color)> {
        scene.title.iter().find_map(|n| match n {
            SceneNode::Text { content, style, .. } if content == want => {
                Some((style.font_size, style.color))
            }
            _ => None,
        })
    }

    #[test]
    fn child_labels_emit_at_child_origin_under_wrap() {
        use crate::render::figure_chrome::DEFAULT_CHROME_INSET;
        // Two labeled leaves in a 2-col wrap flow into one row: child 0 at the
        // left origin, child 1 offset to its right. Each label is a bold header
        // band at its child's top-left, so it moves with the child's placement.
        let mut tree = composite(
            CompositeLayout::Wrap,
            vec![labeled_leaf(0, "Model A"), labeled_leaf(1, "Model B")],
        );
        if let CompositeNode::Composite { ncols, .. } = &mut tree {
            *ncols = Some(2);
        }
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];

        // Baseline without labels: panels sit at their native top (~no header).
        let bare = composite(CompositeLayout::Wrap, vec![leaf_node(0), leaf_node(1)]);
        let bare = {
            let mut b = bare;
            if let CompositeNode::Composite { ncols, .. } = &mut b {
                *ncols = Some(2);
            }
            b
        };
        let (bare_scene, _) =
            render_composite_scene(&bare, &leaves, &ThemeInputs::default()).unwrap();

        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let (ax, _ay) = find_label(&scene, "Model A").expect("child 0 label present");
        let (bx, _by) = find_label(&scene, "Model B").expect("child 1 label present");
        // Child 0 is at placement tx=0, so its label sits at the default inset.
        assert!(
            (ax - DEFAULT_CHROME_INSET).abs() < 1e-6,
            "child 0 label must sit at the child origin inset, got x={ax}"
        );
        // Child 1 is placed to the right; its label is offset by that placement.
        assert!(
            bx > ax,
            "child 1 label must be offset right of child 0: ax={ax} bx={bx}"
        );

        // The label reserves headroom: panels shift DOWN vs. the unlabeled tree.
        assert!(
            scene.panels[0].plot_area.y > bare_scene.panels[0].plot_area.y + 1.0,
            "labeled panel must shift down by the reserved label band"
        );
    }

    #[test]
    fn child_labels_offset_down_with_row_under_grid() {
        // A 2-row x 1-col grid of labeled leaves: the second row's label must be
        // offset DOWN by the first row's placement, proving the label travels
        // with the child under grid placement (not baked at a fixed canvas y).
        let mut tree = composite(
            CompositeLayout::Grid,
            vec![labeled_leaf(0, "Top"), labeled_leaf(1, "Bottom")],
        );
        if let CompositeNode::Composite { nrows, ncols, .. } = &mut tree {
            *nrows = Some(2);
            *ncols = Some(1);
        }
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let (_tx, top_y) = find_label(&scene, "Top").expect("row 0 label present");
        let (_bx, bot_y) = find_label(&scene, "Bottom").expect("row 1 label present");
        assert!(
            bot_y > top_y + 100.0,
            "row 1 label must be offset far below row 0's (row placement): top={top_y} bottom={bot_y}"
        );
    }

    /// Under the DEFAULT theme, a child label must emit the DEFAULT
    /// `ThemeInputs`'s own title styling (`typography.title_font_size` /
    /// `colors.title_color`) — NOT the unrelated `figure_chrome::
    /// FIGURE_TITLE_FONT_SIZE` constant (16px). This is the byte-for-byte old
    /// path: `_compose_compare` (`plots/_helpers.py`) labels a composite child
    /// via `child.properties(title=name)`, i.e. the PER-CHART title pipeline
    /// (`scene_build::build_title`, which reads these same two theme fields),
    /// never the figure-level chrome constants — Task 5d's own report
    /// mischaracterized this as matching `child.to_svg()`'s figure-chrome
    /// wrap. `title_font_size` (13px, `DEFAULT_TITLE_FONT_SIZE`) genuinely
    /// differs from `FIGURE_TITLE_FONT_SIZE` (16px); `title_color` happens to
    /// also be `#1f2937` by coincidence of both defaults sharing the same
    /// "ferrum dark text" hex. Comparing against `ThemeInputs::default()`
    /// directly (not a hardcoded literal) keeps this test honest if either
    /// default ever changes.
    #[test]
    fn child_label_matches_call_level_theme_defaults() {
        let tree = composite(CompositeLayout::Hconcat, vec![labeled_leaf(0, "Model A")]);
        let h0 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0)];
        let default_theme = ThemeInputs::default();
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &default_theme).unwrap();

        let (font_size, color) =
            find_label_style(&scene, "Model A").expect("labeled leaf's label must be present");
        assert_eq!(font_size, default_theme.typography.title_font_size);
        assert_eq!(
            color,
            crate::render::draw::to_scene_color(default_theme.colors.title_color)
        );
    }

    /// The finding this test closes: `apply_child_label` used to style every
    /// label from `FigureChrome::default()` (the hardcoded figure-chrome
    /// constants) regardless of the theme passed to the render call, so a
    /// custom theme's title styling silently never reached composite labels.
    /// A labeled leaf rendered under a NON-default theme (distinct
    /// `title_font_size` + `title_color`) must reflect BOTH overridden values
    /// on the label's text node.
    #[test]
    fn child_label_reflects_call_level_theme_font_size_and_color() {
        let mut theme = ThemeInputs::default();
        theme.typography.title_font_size = 30.0;
        theme.colors.title_color = palette::Srgba::new(0x11, 0x22, 0x33, 0xFF);

        let tree = composite(CompositeLayout::Hconcat, vec![labeled_leaf(0, "Model A")]);
        let h0 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0)];
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &theme).unwrap();

        let (font_size, color) =
            find_label_style(&scene, "Model A").expect("labeled leaf's label must be present");
        assert_eq!(
            font_size, 30.0,
            "label must use the call-level theme's title_font_size"
        );
        assert_eq!(
            color,
            ferrum_scene::Color {
                r: 0x11,
                g: 0x22,
                b: 0x33,
                a: 0xff
            },
            "label must use the call-level theme's title_color"
        );
        assert_ne!(
            font_size, 16.0,
            "sanity: the themed value must differ from the figure-chrome constant"
        );
    }

    // -- figure-level shared legend (GH #16, Task 3) --------------------------

    use crate::layout::facet::ResolveMode as RM;

    /// A point spec with a categorical color encoding on field `g`.
    fn cat_color_spec() -> ChartSpec {
        let mut s = scatter_spec();
        s.encoding.color = Some(EncodingSpec {
            field: "g".into(),
            ..Default::default()
        });
        s
    }

    /// A point spec with categorical color on `g` AND numeric size on `s`
    /// (different fields → color+size do NOT merge; two stacked legends).
    fn color_size_spec() -> ChartSpec {
        let mut spec = cat_color_spec();
        spec.encoding.size = Some(EncodingSpec {
            field: "s".into(),
            ..Default::default()
        });
        spec
    }

    fn xyg_batch(xs: &[f64], ys: &[f64], gs: &[&str]) -> RecordBatch {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs.to_vec())),
                Arc::new(Float64Array::from(ys.to_vec())),
                Arc::new(StringArray::from(
                    gs.iter().map(|s| Some(*s)).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    }

    fn xycs_batch(xs: &[f64], ys: &[f64], gs: &[&str], ss: &[f64]) -> RecordBatch {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("s", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs.to_vec())),
                Arc::new(Float64Array::from(ys.to_vec())),
                Arc::new(StringArray::from(
                    gs.iter().map(|s| Some(*s)).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(ss.to_vec())),
            ],
        )
        .unwrap()
    }

    fn color_hold_with(spec: ChartSpec, batch: RecordBatch, theme: ThemeInputs) -> LeafHold {
        LeafHold {
            spec,
            batch,
            theme,
            config: RenderConfig::default(),
            chart_config: ChartConfig::default(),
        }
    }

    fn color_hold(gs: &[&str]) -> LeafHold {
        color_hold_with(
            cat_color_spec(),
            xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], gs),
            ThemeInputs::default(),
        )
    }

    fn color_hold_oriented(gs: &[&str], orient: LegendOrient) -> LeafHold {
        let mut theme = ThemeInputs::default();
        theme.legend.legend_orient = orient;
        color_hold_with(
            cat_color_spec(),
            xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], gs),
            theme,
        )
    }

    fn color_leaf(data: usize) -> CompositeNode {
        CompositeNode::Leaf {
            spec: Box::new(cat_color_spec()),
            data,
            label: None,
        }
    }

    /// An hconcat of `n` color leaves with the given color resolve mode and an
    /// optional explicit legend override.
    fn color_hconcat(n: usize, color: RM, legend_color: Option<RM>) -> CompositeNode {
        let mut node = composite(CompositeLayout::Hconcat, (0..n).map(color_leaf).collect());
        if let CompositeNode::Composite { resolve, .. } = &mut node {
            resolve.color = Some(color);
            resolve.legend.color = legend_color;
        }
        node
    }

    #[test]
    fn shared_color_hconcat_emits_one_figure_legend_not_per_panel() {
        // Two color leaves sharing color, legend follows scale (default) → one
        // figure band; the same tree forced legend-independent keeps N panel
        // legends. The band's legend-node count must be strictly fewer.
        let h0 = color_hold(&["a", "b", "a"]);
        let h1 = color_hold(&["a", "b", "a"]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];

        let shared = color_hconcat(2, RM::Shared, None);
        let (shared_scene, _) =
            render_composite_scene(&shared, &leaves, &ThemeInputs::default()).unwrap();

        let indep = color_hconcat(2, RM::Shared, Some(RM::Independent));
        let (indep_scene, _) =
            render_composite_scene(&indep, &leaves, &ThemeInputs::default()).unwrap();

        assert!(
            !shared_scene.legend.is_empty(),
            "shared color must emit one figure legend"
        );
        assert!(
            !indep_scene.legend.is_empty(),
            "legend-independent keeps per-panel legends"
        );
        assert!(
            shared_scene.legend.len() < indep_scene.legend.len(),
            "one figure legend ({}) must draw fewer nodes than two per-panel legends ({})",
            shared_scene.legend.len(),
            indep_scene.legend.len(),
        );
        assert_eq!(
            shared_scene.panels.len(),
            2,
            "band must not add or drop panels"
        );
    }

    #[test]
    fn shared_color_band_grows_scene_on_right_edge() {
        // Right orient (theme default): the figure legend grows the scene width
        // beyond the panel-grid width (300 + 10 + 300 = 610); the per-panel
        // (legend-independent) render keeps legends inside each panel, so its
        // width stays the grid width.
        let h0 = color_hold(&["a", "b", "a"]);
        let h1 = color_hold(&["a", "b", "a"]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];

        let shared = color_hconcat(2, RM::Shared, None);
        let (shared_scene, _) =
            render_composite_scene(&shared, &leaves, &ThemeInputs::default()).unwrap();
        assert!(
            shared_scene.width > 610.0,
            "right band must grow width past the grid: {}",
            shared_scene.width
        );
        assert!(
            (shared_scene.height - 200.0).abs() < 1e-6,
            "right band must not grow height"
        );

        let indep = color_hconcat(2, RM::Shared, Some(RM::Independent));
        let (indep_scene, _) =
            render_composite_scene(&indep, &leaves, &ThemeInputs::default()).unwrap();
        assert!(
            (indep_scene.width - 610.0).abs() < 1e-6,
            "per-panel legends must not grow the grid width: {}",
            indep_scene.width
        );
    }

    #[test]
    fn shared_color_band_grows_top_and_shifts_panels_down() {
        let h0 = color_hold_oriented(&["a", "b", "a"], LegendOrient::Top);
        let h1 = color_hold_oriented(&["a", "b", "a"], LegendOrient::Top);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let tree = color_hconcat(2, RM::Shared, None);
        let (scene, _) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert!(
            scene.height > 200.0,
            "top band must grow scene height: {}",
            scene.height
        );
        assert!(
            (scene.width - 610.0).abs() < 1e-6,
            "top band must not grow width: {}",
            scene.width
        );
        assert!(
            scene.panels[0].plot_area.y > 8.0,
            "top band must shift panels down: {}",
            scene.panels[0].plot_area.y
        );
    }

    #[test]
    fn shared_color_band_left_shifts_panels_right() {
        let h0 = color_hold_oriented(&["a", "b", "a"], LegendOrient::Left);
        let h1 = color_hold_oriented(&["a", "b", "a"], LegendOrient::Left);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let tree = color_hconcat(2, RM::Shared, None);
        let (scene, _) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert!(
            scene.width > 610.0,
            "left band must grow scene width: {}",
            scene.width
        );
        // Panel 0's plot area is pushed right by the legend gutter.
        let right_h = color_hold(&["a", "b", "a"]);
        let right_leaves = [
            leaf_input(&right_h, 300.0, 200.0),
            leaf_input(&right_h, 300.0, 200.0),
        ];
        let (right_scene, _) =
            render_composite_scene(&tree, &right_leaves, &ThemeInputs::default()).unwrap();
        assert!(
            scene.panels[0].plot_area.x > right_scene.panels[0].plot_area.x,
            "left band shifts panels right ({} vs right-orient {})",
            scene.panels[0].plot_area.x,
            right_scene.panels[0].plot_area.x,
        );
    }

    // -- legend band clip-safety padding (GH #16 follow-up) -------------------
    //
    // `draw_legend_band` used to grow the scene by the EXACT measured content
    // extent (`legend_layouts_extent`'s bounding box) with zero trailing safety
    // margin on the edge bordering the canvas boundary. `legend_layouts_extent`
    // measures glyph *advance* widths (`TextMetrics::measure_width`), which
    // don't include the terminal glyph's right side-bearing/ink overhang, so an
    // exact-fit grow clips that sliver at the canvas edge — visible on real SVG
    // goldens as a clipped title/label/tick-label glyph. These tests construct a
    // `LeafLegendBundle` directly (bypassing the full render pipeline, same as
    // the private-function access every other test in this module relies on)
    // so the assertions are exact rather than a loose end-to-end threshold.

    /// A minimal two-entry categorical bundle under the given theme/title.
    fn band_bundle(theme: ThemeInputs, title: Option<&str>) -> LeafLegendBundle {
        LeafLegendBundle {
            entries: vec![
                LegendEntry {
                    label: "a".into(),
                    symbol: SymbolKind::Circle,
                },
                LegendEntry {
                    label: "b".into(),
                    symbol: SymbolKind::Circle,
                },
            ],
            colorbar: None,
            title: title.map(str::to_owned),
            overrides: LegendOverrides::default(),
            aux: Vec::new(),
            color_scale: None,
            theme,
            merged_color_size: false,
        }
    }

    /// A single-stop-pair colorbar bundle (numeric legend content) under the
    /// given theme/title — exercises the colorbar arm of `legend_layouts_extent`
    /// (`cb.bar_rect` + tick labels), distinct from the categorical-entries arm
    /// every other band test drives.
    fn band_colorbar_bundle(theme: ThemeInputs, title: Option<&str>) -> LeafLegendBundle {
        LeafLegendBundle {
            entries: Vec::new(),
            colorbar: Some(ColorbarInput {
                stops: vec![(0.0, "#000000".into()), (1.0, "#ffffff".into())],
                tick_labels: vec!["1".into(), "7.75".into(), "10".into()],
                domain: Some((1.0, 10.0)),
            }),
            title: title.map(str::to_owned),
            overrides: LegendOverrides::default(),
            aux: Vec::new(),
            color_scale: None,
            theme,
            merged_color_size: false,
        }
    }

    /// For each of the four orients, the scene must grow past the band's own
    /// measured content extent (`legend_layouts_extent`, the same function
    /// `draw_legend_band` uses) by at least `LEGEND_OUTER_PAD` on the grown
    /// edge — the discriminating geometry assertion: on pre-fix code, the grown
    /// edge equals the content extent exactly (delta 0 < `LEGEND_OUTER_PAD`),
    /// so this fails RED on the unpadded implementation.
    #[test]
    fn legend_band_pads_trailing_edge_past_content_extent_all_orients() {
        let metrics = crate::render::font::FontdueMetrics::new();
        let flags = BandFlags {
            color: true,
            size: false,
        };

        for orient in [
            LegendOrient::Right,
            LegendOrient::Left,
            LegendOrient::Top,
            LegendOrient::Bottom,
        ] {
            let mut theme = ThemeInputs::default();
            theme.legend.legend_orient = orient;
            let bundle = band_bundle(theme.clone(), Some("grp"));

            let mut scene = empty_scene(300.0, 200.0);
            draw_legend_band(&mut scene, &bundle, flags);

            let layouts = layout_band_legends(&bundle, flags, &metrics);
            let (min_x, min_y, max_x, max_y) = legend_layouts_extent(
                &layouts,
                theme.typography.label_font_size,
                theme.typography.legend_title_font_size,
                &metrics,
            )
            .expect("band content must measure non-empty");
            let content_w = max_x - min_x;
            let content_h = max_y - min_y;

            let grown = match orient {
                LegendOrient::Right | LegendOrient::Left => scene.width - 300.0 - LEGEND_PLOT_GAP,
                LegendOrient::Top | LegendOrient::Bottom => scene.height - 200.0 - LEGEND_PLOT_GAP,
            };
            let content = match orient {
                LegendOrient::Right | LegendOrient::Left => content_w,
                LegendOrient::Top | LegendOrient::Bottom => content_h,
            };
            assert!(
                grown >= content + LEGEND_OUTER_PAD - 1e-9,
                "{orient:?} band must reserve at least LEGEND_OUTER_PAD ({LEGEND_OUTER_PAD}) \
                 past its measured content extent ({content}), got grown={grown}",
            );
        }
    }

    /// Same trailing-padding assertion, driven through the colorbar arm of
    /// `legend_layouts_extent` at `LegendOrient::Bottom` — the specific
    /// "horizontal colorbar band" case called out as a known risk area (the
    /// bottom-orient golden `shared_colorbar_orient_bottom.svg`).
    #[test]
    fn legend_band_pads_trailing_edge_colorbar_bottom_orient() {
        let metrics = crate::render::font::FontdueMetrics::new();
        let flags = BandFlags {
            color: true,
            size: false,
        };
        let mut theme = ThemeInputs::default();
        theme.legend.legend_orient = LegendOrient::Bottom;
        let bundle = band_colorbar_bundle(theme.clone(), Some("value"));

        let mut scene = empty_scene(300.0, 200.0);
        draw_legend_band(&mut scene, &bundle, flags);

        let layouts = layout_band_legends(&bundle, flags, &metrics);
        let (_, min_y, _, max_y) = legend_layouts_extent(
            &layouts,
            theme.typography.label_font_size,
            theme.typography.legend_title_font_size,
            &metrics,
        )
        .expect("colorbar band content must measure non-empty");
        let content_h = max_y - min_y;
        let grown = scene.height - 200.0 - LEGEND_PLOT_GAP;
        assert!(
            grown >= content_h + LEGEND_OUTER_PAD - 1e-9,
            "bottom colorbar band must reserve at least LEGEND_OUTER_PAD ({LEGEND_OUTER_PAD}) \
             past its measured content extent ({content_h}), got grown={grown}",
        );
    }

    /// Regression for the title-font-size bug: `legend_layouts_extent` used to
    /// measure the band title at `label_fs` (the *label* font size) instead of
    /// `theme.typography.legend_title_font_size` — a no-op under the default
    /// theme (both default to the same 11px), so it only manifests when a theme
    /// sets them differently and the title is the widest element. A 40px title
    /// longer than any 11px label must grow the band wide enough for the 40px
    /// glyphs, not the (much narrower) 11px estimate.
    #[test]
    fn legend_band_title_measured_at_title_font_size_not_label_font_size() {
        let mut theme = ThemeInputs::default();
        theme.typography.legend_title_font_size = 40.0;
        let bundle = band_bundle(theme.clone(), Some("A Very Long Legend Title"));

        let mut scene = empty_scene(300.0, 200.0);
        draw_legend_band(
            &mut scene,
            &bundle,
            BandFlags {
                color: true,
                size: false,
            },
        );

        let metrics = crate::render::font::FontdueMetrics::new();
        let title_w_at_title_fs = metrics.measure_width(
            "A Very Long Legend Title",
            theme.typography.legend_title_font_size,
        );
        let title_w_at_label_fs =
            metrics.measure_width("A Very Long Legend Title", theme.typography.label_font_size);
        assert!(
            title_w_at_title_fs > title_w_at_label_fs,
            "sanity: the 40px title must measure wider than the 11px label-font estimate \
             ({title_w_at_title_fs} vs {title_w_at_label_fs})",
        );

        let band_w = scene.width - 300.0;
        assert!(
            band_w >= title_w_at_title_fs,
            "band width ({band_w}) must fit the title measured at legend_title_font_size \
             ({title_w_at_title_fs}), not just the label-font estimate ({title_w_at_label_fs})",
        );
    }

    #[test]
    fn nested_shared_color_attaches_band_to_inner_subtree_only() {
        // Outer hconcat is independent; the inner hconcat of two color leaves
        // shares color. One band attaches at the inner node; forcing the inner
        // legend-independent removes the band (per-panel legends return).
        let inner_shared = {
            let mut inner = composite(CompositeLayout::Hconcat, vec![color_leaf(0), color_leaf(1)]);
            if let CompositeNode::Composite { resolve, .. } = &mut inner {
                resolve.color = Some(RM::Shared);
            }
            composite(CompositeLayout::Hconcat, vec![leaf_node(2), inner])
        };
        let inner_indep = {
            let mut inner = composite(CompositeLayout::Hconcat, vec![color_leaf(0), color_leaf(1)]);
            if let CompositeNode::Composite { resolve, .. } = &mut inner {
                resolve.color = Some(RM::Shared);
                resolve.legend.color = Some(RM::Independent);
            }
            composite(CompositeLayout::Hconcat, vec![leaf_node(2), inner])
        };
        let c0 = color_hold(&["a", "b", "a"]);
        let c1 = color_hold(&["a", "b", "a"]);
        let plain = hold(); // data index 2: a plain x/y leaf, no color
                            // Leaf inputs are in tree pre-order: the outer plain leaf, then the inner
                            // node's two color leaves.
        let ordered = [
            leaf_input(&plain, 300.0, 200.0),
            leaf_input(&c0, 300.0, 200.0),
            leaf_input(&c1, 300.0, 200.0),
        ];
        let (shared_scene, _) =
            render_composite_scene(&inner_shared, &ordered, &ThemeInputs::default()).unwrap();
        let (indep_scene, _) =
            render_composite_scene(&inner_indep, &ordered, &ThemeInputs::default()).unwrap();
        assert!(
            !shared_scene.legend.is_empty(),
            "inner-shared color must emit one band"
        );
        assert!(
            shared_scene.legend.len() < indep_scene.legend.len(),
            "nested band ({}) must draw fewer legend nodes than per-panel ({})",
            shared_scene.legend.len(),
            indep_scene.legend.len(),
        );
        assert_eq!(shared_scene.panels.len(), 3);
    }

    #[test]
    fn all_participating_leaves_disabled_emits_no_band() {
        // Both color leaves disable their color legend → empty bundles, so no
        // figure band is emitted even though color resolves shared. Suppression
        // still applies (nothing to draw either way), so scene.legend is empty.
        use crate::render::chart_config::LegendStyleSpec;
        let disabled_spec = || {
            let mut s = cat_color_spec();
            s.encoding.color.as_mut().unwrap().legend = Some(Box::new(LegendStyleSpec {
                disabled: Some(true),
                ..Default::default()
            }));
            s
        };
        let h0 = color_hold_with(
            disabled_spec(),
            xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &["a", "b", "a"]),
            ThemeInputs::default(),
        );
        let h1 = color_hold_with(
            disabled_spec(),
            xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &["a", "b", "a"]),
            ThemeInputs::default(),
        );
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        // Tree leaf specs must also carry the disabled legend so planning sees it.
        let mut tree = composite(
            CompositeLayout::Hconcat,
            vec![
                CompositeNode::Leaf {
                    spec: Box::new(disabled_spec()),
                    data: 0,
                    label: None,
                },
                CompositeNode::Leaf {
                    spec: Box::new(disabled_spec()),
                    data: 1,
                    label: None,
                },
            ],
        );
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.color = Some(RM::Shared);
        }
        let (scene, _) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert!(
            scene.legend.is_empty(),
            "all-disabled participants → no figure band"
        );
        assert!(
            (scene.width - 610.0).abs() < 1e-6,
            "no band → no width growth: {}",
            scene.width
        );
    }

    #[test]
    fn capture_skips_disabled_leaf_and_uses_next_participant() {
        // Leaf 0 disables its color legend; leaf 1 keeps it. The band captures
        // leaf 1's non-empty bundle, so exactly one band is still emitted.
        use crate::render::chart_config::LegendStyleSpec;
        let mut disabled = cat_color_spec();
        disabled.encoding.color.as_mut().unwrap().legend = Some(Box::new(LegendStyleSpec {
            disabled: Some(true),
            ..Default::default()
        }));
        let h0 = color_hold_with(
            disabled.clone(),
            xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &["a", "b", "a"]),
            ThemeInputs::default(),
        );
        let h1 = color_hold(&["a", "b", "a"]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let mut tree = composite(
            CompositeLayout::Hconcat,
            vec![
                CompositeNode::Leaf {
                    spec: Box::new(disabled),
                    data: 0,
                    label: None,
                },
                color_leaf(1),
            ],
        );
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.color = Some(RM::Shared);
        }
        let (scene, _) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert!(
            !scene.legend.is_empty(),
            "band must be captured from the non-disabled leaf 1"
        );
        assert!(
            scene.width > 610.0,
            "band from leaf 1 still grows the scene: {}",
            scene.width
        );
    }

    #[test]
    fn explicit_leaf_color_scale_keeps_its_own_panel_legend() {
        // Leaf 1 pins an explicit ordinal color scale → excluded from the shared
        // union, so it renders its own panel legend while leaf 0 (participant) is
        // deduped into the figure band. Result: MORE legend nodes than a tree
        // where both participate (band only).
        use crate::spec::encoding::ScaleSpec;
        let mut excl_spec = cat_color_spec();
        excl_spec.encoding.color.as_mut().unwrap().scale = Some(ScaleSpec::Ordinal {
            domain: None,
            range: None,
            padding: 0.0,
        });
        let h0 = color_hold(&["a", "b", "a"]);
        let h1 = color_hold_with(
            excl_spec.clone(),
            xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &["a", "b", "a"]),
            ThemeInputs::default(),
        );
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let mut tree = composite(
            CompositeLayout::Hconcat,
            vec![
                color_leaf(0),
                CompositeNode::Leaf {
                    spec: Box::new(excl_spec),
                    data: 1,
                    label: None,
                },
            ],
        );
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.color = Some(RM::Shared);
        }
        let (scene, _) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        // Baseline: both participate (no explicit scale) → single band only.
        let both = color_hold(&["a", "b", "a"]);
        let both_leaves = [
            leaf_input(&both, 300.0, 200.0),
            leaf_input(&both, 300.0, 200.0),
        ];
        let both_tree = color_hconcat(2, RM::Shared, None);
        let (both_scene, _) =
            render_composite_scene(&both_tree, &both_leaves, &ThemeInputs::default()).unwrap();

        assert!(
            scene.legend.len() > both_scene.legend.len(),
            "excluded leaf's own legend ({}) must add nodes beyond the lone band ({})",
            scene.legend.len(),
            both_scene.legend.len(),
        );
    }

    #[test]
    fn shared_color_and_size_stack_both_in_one_band() {
        // A leaf with color (categorical) + size (numeric) on DIFFERENT fields,
        // sharing both channels on one node → one band containing the color
        // legend AND the size aux block: more legend nodes than a color-only band.
        let batch = xycs_batch(
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0, 3.0],
            &["a", "b", "a"],
            &[2.0, 5.0, 9.0],
        );
        let h0 = color_hold_with(color_size_spec(), batch.clone(), ThemeInputs::default());
        let h1 = color_hold_with(color_size_spec(), batch, ThemeInputs::default());
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let cs_leaf = |data| CompositeNode::Leaf {
            spec: Box::new(color_size_spec()),
            data,
            label: None,
        };
        let mut tree = composite(CompositeLayout::Hconcat, vec![cs_leaf(0), cs_leaf(1)]);
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.color = Some(RM::Shared);
            resolve.size = Some(RM::Shared);
        }
        let (both_scene, _) =
            render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        // Color-only shared baseline (size independent → size legend stays per-panel,
        // not in the band).
        let color_only = color_hconcat(2, RM::Shared, None);
        let ch = color_hold(&["a", "b", "a"]);
        let color_leaves = [leaf_input(&ch, 300.0, 200.0), leaf_input(&ch, 300.0, 200.0)];
        let (color_scene, _) =
            render_composite_scene(&color_only, &color_leaves, &ThemeInputs::default()).unwrap();

        assert!(
            !both_scene.legend.is_empty(),
            "color+size share must emit a band"
        );
        assert!(
            both_scene.legend.len() > color_scene.legend.len(),
            "stacked color+size band ({}) must draw more nodes than a color-only band ({})",
            both_scene.legend.len(),
            color_scene.legend.len(),
        );
    }

    #[test]
    fn legend_independent_override_matches_per_panel_baseline() {
        // A shared color SCALE with legend={color: independent} must render the
        // per-panel legends today's shared-scale output produced — no band, no
        // suppression. Its legend-node count matches the independent-scale render
        // (both draw one legend per panel) and exceeds the shared-legend band.
        let h0 = color_hold(&["a", "b", "a"]);
        let h1 = color_hold(&["a", "b", "a"]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];

        let override_tree = color_hconcat(2, RM::Shared, Some(RM::Independent));
        let (override_scene, _) =
            render_composite_scene(&override_tree, &leaves, &ThemeInputs::default()).unwrap();

        let independent_scale = color_hconcat(2, RM::Independent, None);
        let (indep_scene, _) =
            render_composite_scene(&independent_scale, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(
            override_scene.legend.len(), indep_scene.legend.len(),
            "legend-independent over a shared scale must draw the same per-panel legends as an independent scale",
        );
        assert!(
            (override_scene.width - 610.0).abs() < 1e-6,
            "no band → no growth: {}",
            override_scene.width
        );
    }

    #[test]
    fn invalid_wire_shared_legend_over_independent_scale_degrades_to_no_band() {
        // A directly-constructed composite wire spec: `legend.color =
        // Some(Shared)` paired with an INDEPENDENT color scale. Python's
        // lowering-time guard rejects this exact combination (design §4)
        // before it ever reaches Rust, so no real caller can build it —
        // `CompositeNode::validate` does not re-check the pairing either.
        // This test constructs the wire-level spec directly to pin what
        // `descend_channel` does when that guard is bypassed: it computes
        // `legend_eff == Shared` while the effective scale mode is
        // `Independent`, but `is_scale_resolver` requires an effective-shared
        // scale, so `band_here` is always `false` regardless of the legend
        // override — no band, no suppression, per-panel legends stay exactly
        // as an independent-scale render's. No panic either (a `debug_assert!`
        // here would have aborted this test under the default debug/test
        // profile).
        let h0 = color_hold(&["a", "b", "a"]);
        let h1 = color_hold(&["a", "b", "a"]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];

        let mut invalid = composite(CompositeLayout::Hconcat, vec![color_leaf(0), color_leaf(1)]);
        if let CompositeNode::Composite { resolve, .. } = &mut invalid {
            resolve.color = Some(RM::Independent);
            resolve.legend.color = Some(RM::Shared);
        }
        let (scene, _) = render_composite_scene(&invalid, &leaves, &ThemeInputs::default())
            .expect("an invalid wire-level legend/scale pairing must degrade, not error");

        // Baseline: a plain independent-scale tree with no legend override —
        // today's per-panel rendering. The invalid pairing must match it
        // exactly: no band, same per-panel legend count, same panel count.
        let baseline = color_hconcat(2, RM::Independent, None);
        let (baseline_scene, _) =
            render_composite_scene(&baseline, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(
            scene.legend.len(),
            baseline_scene.legend.len(),
            "shared legend over an independent scale must degrade to the same per-panel \
             legends as a plain independent-scale render: got {} vs baseline {}",
            scene.legend.len(),
            baseline_scene.legend.len(),
        );
        assert_eq!(scene.panels.len(), 2, "degrade must not add or drop panels");
    }

    // -- extraction fix-round regression tests (Task 3, legend::layout_color_legend) --

    fn xys_batch(xs: &[f64], ys: &[f64], ss: &[f64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("s", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs.to_vec())),
                Arc::new(Float64Array::from(ys.to_vec())),
                Arc::new(Float64Array::from(ss.to_vec())),
            ],
        )
        .unwrap()
    }

    /// A point spec with numeric size on `s` and NO color channel at all.
    fn size_only_spec() -> ChartSpec {
        let mut s = scatter_spec();
        s.encoding.size = Some(EncodingSpec {
            field: "s".into(),
            ..Default::default()
        });
        s
    }

    fn size_leaf(data: usize) -> CompositeNode {
        CompositeNode::Leaf {
            spec: Box::new(size_only_spec()),
            data,
            label: None,
        }
    }

    #[test]
    fn shared_size_only_bands_and_suppresses_panel_size_legends() {
        // Size shared, color independent (no color channel at all): one figure
        // size legend band; per-panel size legends must be suppressed. Mirrors
        // `shared_color_hconcat_emits_one_figure_legend_not_per_panel` but drives
        // the size channel through `layout_color_legend`'s dispatch instead of the
        // color one — the shared helper's `legend_entries`/`colorbar` inputs are
        // both empty for a size-only leaf, so this exercises its early-return arm.
        let batch = xys_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[2.0, 5.0, 9.0]);
        let h0 = color_hold_with(size_only_spec(), batch.clone(), ThemeInputs::default());
        let h1 = color_hold_with(size_only_spec(), batch, ThemeInputs::default());
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];

        let mut shared = composite(CompositeLayout::Hconcat, vec![size_leaf(0), size_leaf(1)]);
        if let CompositeNode::Composite { resolve, .. } = &mut shared {
            resolve.size = Some(RM::Shared);
        }
        let (shared_scene, _) =
            render_composite_scene(&shared, &leaves, &ThemeInputs::default()).unwrap();

        let mut indep = composite(CompositeLayout::Hconcat, vec![size_leaf(0), size_leaf(1)]);
        if let CompositeNode::Composite { resolve, .. } = &mut indep {
            resolve.size = Some(RM::Independent);
        }
        let (indep_scene, _) =
            render_composite_scene(&indep, &leaves, &ThemeInputs::default()).unwrap();

        assert!(
            !shared_scene.legend.is_empty(),
            "shared size must emit one figure legend band"
        );
        assert!(
            !indep_scene.legend.is_empty(),
            "independent size keeps per-panel size legends"
        );
        assert!(
            shared_scene.legend.len() < indep_scene.legend.len(),
            "one figure size band ({}) must draw fewer nodes than two per-panel size legends ({})",
            shared_scene.legend.len(),
            indep_scene.legend.len(),
        );
        assert_eq!(
            shared_scene.panels.len(),
            2,
            "band must not add or drop panels"
        );
    }

    /// A point spec with color AND size mapped to the SAME numeric field `s` —
    /// `leaf_merges_color_size` folds size into the color legend for this leaf.
    fn merged_color_size_spec() -> ChartSpec {
        let mut s = scatter_spec();
        s.encoding.color = Some(EncodingSpec {
            field: "s".into(),
            ..Default::default()
        });
        s.encoding.size = Some(EncodingSpec {
            field: "s".into(),
            ..Default::default()
        });
        s
    }

    fn merged_leaf(data: usize) -> CompositeNode {
        CompositeNode::Leaf {
            spec: Box::new(merged_color_size_spec()),
            data,
            label: None,
        }
    }

    #[test]
    fn same_field_color_size_merge_bands_folded_size_and_suppresses_both_channels() {
        // color+size on the SAME field ("s") merges into one legend per leaf
        // (`leaf_merges_color_size`), so sharing color ALONE (size resolve stays
        // Independent, the default) must still band the folded size content —
        // `layout_band_legends`'s `include_size = flags.color && merged_color_size`
        // arm — and suppress BOTH panel channels together, not just color.
        let batch = xys_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[2.0, 5.0, 9.0]);
        let h0 = color_hold_with(
            merged_color_size_spec(),
            batch.clone(),
            ThemeInputs::default(),
        );
        let h1 = color_hold_with(merged_color_size_spec(), batch, ThemeInputs::default());
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];

        let mut shared = composite(
            CompositeLayout::Hconcat,
            vec![merged_leaf(0), merged_leaf(1)],
        );
        if let CompositeNode::Composite { resolve, .. } = &mut shared {
            resolve.color = Some(RM::Shared);
        }
        let (shared_scene, _) =
            render_composite_scene(&shared, &leaves, &ThemeInputs::default()).unwrap();

        // Baseline A: same tree, resolve stays default (Independent) → both
        // panels keep their OWN already-merged color+size legend (nothing
        // compositor-suppressed) — proves suppression collapsed two per-panel
        // merged blocks into one band.
        let unshared = composite(
            CompositeLayout::Hconcat,
            vec![merged_leaf(0), merged_leaf(1)],
        );
        let (unshared_scene, _) =
            render_composite_scene(&unshared, &leaves, &ThemeInputs::default()).unwrap();

        // Baseline B: a plain categorical color-only band (different field,
        // no size at all) — proves the merged band carries MORE than bare
        // color, i.e. the folded size swatches are actually present.
        let color_only = color_hconcat(2, RM::Shared, None);
        let ch = color_hold(&["a", "b", "a"]);
        let color_leaves = [leaf_input(&ch, 300.0, 200.0), leaf_input(&ch, 300.0, 200.0)];
        let (color_scene, _) =
            render_composite_scene(&color_only, &color_leaves, &ThemeInputs::default()).unwrap();

        assert!(
            !shared_scene.legend.is_empty(),
            "same-field color+size merge must emit a band"
        );
        assert!(
            shared_scene.legend.len() < unshared_scene.legend.len(),
            "one band folding both suppressed channels ({}) must draw fewer nodes than two \
             per-panel merged legends ({})",
            shared_scene.legend.len(),
            unshared_scene.legend.len(),
        );
        assert!(
            shared_scene.legend.len() > color_scene.legend.len(),
            "band with folded size content ({}) must draw more nodes than a color-only band ({})",
            shared_scene.legend.len(),
            color_scene.legend.len(),
        );
        assert_eq!(
            shared_scene.panels.len(),
            2,
            "band must not add or drop panels"
        );
    }

    #[test]
    fn nested_shared_color_band_at_top_orient_grows_height_and_shifts_panels_down() {
        // The flat two-leaf Left/Top orient geometry tests above put the shared
        // node at the tree root; this drives the same Top-orient geometry
        // through a NESTED tree (the shared node is an inner child, alongside an
        // unrelated sibling leaf), proving `layout_color_legend`'s dispatch still
        // grows/shifts on the correct edge when the band attaches below the root.
        let inner = {
            let mut inner = composite(CompositeLayout::Hconcat, vec![color_leaf(0), color_leaf(1)]);
            if let CompositeNode::Composite { resolve, .. } = &mut inner {
                resolve.color = Some(RM::Shared);
            }
            composite(CompositeLayout::Hconcat, vec![leaf_node(2), inner])
        };
        let plain = hold();
        let c0 = color_hold_oriented(&["a", "b", "a"], LegendOrient::Top);
        let c1 = color_hold_oriented(&["a", "b", "a"], LegendOrient::Top);
        let leaves = [
            leaf_input(&plain, 300.0, 200.0),
            leaf_input(&c0, 300.0, 200.0),
            leaf_input(&c1, 300.0, 200.0),
        ];
        let (scene, _) = render_composite_scene(&inner, &leaves, &ThemeInputs::default()).unwrap();

        assert!(
            !scene.legend.is_empty(),
            "nested Top-orient share must still emit a band"
        );
        assert!(
            scene.height > 200.0,
            "top band must grow scene height even when nested: {}",
            scene.height
        );
        assert!(
            scene.panels[1].plot_area.y > 8.0,
            "top band must shift the inner subtree's panels down: {}",
            scene.panels[1].plot_area.y,
        );
        assert_eq!(scene.panels.len(), 3, "band must not add or drop panels");
    }

    // -- figure-level shared legend + `Hole` cells (GH #16 hole fix) ---------

    /// A `Grid`-layout node with the given `nrows`/`ncols` and color resolve
    /// mode — mirrors `color_hconcat`'s shape for the `Grid` layout kind
    /// `pairplot`/`jointplot` actually lower to (a flat grid whose direct
    /// children mix real cells with `Hole` placeholders).
    fn color_grid(
        children: Vec<CompositeNode>,
        nrows: u32,
        ncols: u32,
        color: RM,
    ) -> CompositeNode {
        let mut node = composite(CompositeLayout::Grid, children);
        if let CompositeNode::Composite {
            resolve,
            nrows: nr,
            ncols: nc,
            ..
        } = &mut node
        {
            resolve.color = Some(color);
            *nr = Some(nrows);
            *nc = Some(ncols);
        }
        node
    }

    #[test]
    fn grid_with_hole_shares_color_and_bands_once() {
        // The pairplot(corner=True) / jointplot shape: a grid whose direct
        // children mix real color leaves with a `Hole` placeholder (the
        // upper-triangle / empty-corner cell). Before the fix,
        // `congruent_children` compared every child against a literal
        // `children[0]` reference, and a `Hole` is congruent only with
        // another `Hole` (`congruent`'s doc) — so ANY hole among a grid's
        // direct cells made the whole node "non-congruent", disabling the
        // domain union entirely: no band, every participating leaf kept its
        // own per-panel legend (the reported #16 bug — pairplot(corner=True)
        // rendered 3 legends instead of 1).
        let cells = || {
            vec![
                color_leaf(0),
                CompositeNode::Hole {
                    width: None,
                    height: None,
                },
                color_leaf(1),
                color_leaf(2),
            ]
        };
        let h0 = color_hold(&["a", "b", "a"]);
        let h1 = color_hold(&["a", "b", "a"]);
        let h2 = color_hold(&["a", "b", "a"]);
        let leaves = [
            leaf_input(&h0, 300.0, 200.0),
            leaf_input(&h1, 300.0, 200.0),
            leaf_input(&h2, 300.0, 200.0),
        ];

        let shared = color_grid(cells(), 2, 2, RM::Shared);
        let (scene, _) = render_composite_scene(&shared, &leaves, &ThemeInputs::default()).unwrap();

        let indep = color_grid(cells(), 2, 2, RM::Independent);
        let (indep_scene, _) =
            render_composite_scene(&indep, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 3, "the hole must not claim a panel");
        assert!(
            !scene.legend.is_empty(),
            "shared color across a grid with a hole must still emit one figure legend"
        );
        assert!(
            scene.legend.len() < indep_scene.legend.len(),
            "one figure legend ({}) must draw fewer nodes than three per-panel legends ({}) — \
             the real cells must still be recognized as sharing and suppressed",
            scene.legend.len(),
            indep_scene.legend.len(),
        );
    }

    #[test]
    fn hole_before_leaf_does_not_misalign_suppression() {
        // A hole at position 0 — `children[0]` IS the hole. The pre-fix
        // `congruent_children` compared every OTHER child against
        // `children[0]` directly, so a LEADING hole would disqualify the
        // whole node regardless of how many real cells followed it: the
        // sharpest form of the cursor/pairing-misalignment risk this fix
        // guards against. The two real leaves at positions 1 and 2 must
        // still union and band EXACTLY as the same two leaves do with no
        // hole present at all — same panel count, same legend content — so
        // the suppression flags provably land on the two real leaves, not
        // on the wrong (or no) leaves.
        let with_hole = color_grid(
            vec![
                CompositeNode::Hole {
                    width: None,
                    height: None,
                },
                color_leaf(0),
                color_leaf(1),
            ],
            1,
            3,
            RM::Shared,
        );
        let h0 = color_hold(&["a", "b", "a"]);
        let h1 = color_hold(&["a", "b", "a"]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (hole_scene, _) =
            render_composite_scene(&with_hole, &leaves, &ThemeInputs::default()).unwrap();

        let no_hole = color_hconcat(2, RM::Shared, None);
        let (baseline_scene, _) =
            render_composite_scene(&no_hole, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(
            hole_scene.panels.len(),
            2,
            "the leading hole must not claim a panel"
        );
        assert!(
            !hole_scene.legend.is_empty(),
            "the two real leaves after a leading hole must still band"
        );
        assert_eq!(
            hole_scene.legend.len(),
            baseline_scene.legend.len(),
            "a leading hole must not change which leaves get suppressed: legend content must \
             match the hole-free baseline exactly",
        );
    }
}
