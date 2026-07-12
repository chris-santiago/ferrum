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
use crate::layout::legend::{layout_aux_legends, layout_color_legend, LEGEND_OUTER_PAD, LEGEND_PLOT_GAP};
use crate::layout::text_metrics::TextMetrics;
use crate::layout::{
    AuxLegendInput, ColorbarInput, LegendEntry, LegendLayout, LegendOrient, LegendOverrides,
    Rect as LayoutRect, ThemeInputs, Viewport,
};
use crate::spec::chart::ChartSpec;

use super::chart_config::ChartConfig;
use super::composite::{
    congruent_non_hole, flatten_leaf_specs, resolve_composite_scales, CompositeResolveError,
    LeafResolveInput,
};
use super::scale_resolve::{ColorScale, LeafScaleContext};
use super::svg::uniquify_clip_ids;
use super::config::RenderConfig;
use super::figure_chrome::{title_nodes, ChromeAnchor, FigureChrome, DEFAULT_CHROME_INSET};
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
            Self::LeafRender { kind, index, source } => {
                write!(f, "failed to render composite {kind} leaf #{index}: {source}")
            }
            Self::LeafDataIndexOutOfBounds { kind, index, data, payload_count } => write!(
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
    drop(prepared);

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

    // Pass 2/3 (per-leaf render): re-render each leaf with its resolved-domain
    // context so composite-shared channels land on the auto scale path (D4b). A
    // fully-empty context passes `None` so non-shared leaves render byte-identical
    // to a standalone chart. A suppressed leaf additionally returns its prepared
    // legend bundle so the compositor can build the figure legend from it.
    let mut leaf_scenes: Vec<SceneGraph> = Vec::with_capacity(n);
    let mut bundles: Vec<Option<LeafLegendBundle>> = Vec::with_capacity(n);
    let mut warnings: Vec<RenderWarning> = Vec::new();
    for (i, leaf) in leaves.iter().enumerate() {
        let ctx = &contexts[i];
        let ctx_opt = (!ctx.is_empty()).then_some(ctx);
        let (mut scene, leaf_warnings, bundle) = render_leaf(leaf, ctx_opt).map_err(|source| {
            CompositeRenderError::LeafRender { kind: "leaf", index: i, source }
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

    let band_ctx = LegendBandCtx { plan: &band_plan, contexts: &contexts, bundles: &bundles };

    // Pass 2/3 (place + merge): walk the tree, placing each leaf scene into the
    // composite frame. Panels are renumbered flat in pre-order as leaf scenes are
    // consumed (D4c); `leaf_cursor` tracks the same pre-order leaf index so a
    // figure-legend band node can capture the first participating leaf's bundle
    // from its subtree.
    let mut scenes = leaf_scenes.into_iter();
    let mut panel_base = 0usize;
    let mut leaf_cursor = 0usize;
    let mut placed = build_placed(
        tree,
        &mut scenes,
        &mut panel_base,
        &mut leaf_cursor,
        call_theme,
        &band_ctx,
        None,
    );

    // Root figure chrome (title/subtitle/caption/config) — validated root-only.
    if let CompositeNode::Composite { title, subtitle, caption, config, .. } = tree {
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
/// identity (pointer into the borrowed tree the layout pass walks). Empty for
/// any composite with all-independent legend resolution — the byte-stable path.
struct LegendBandPlan {
    band_nodes: HashMap<*const CompositeNode, BandFlags>,
}

/// Everything the place/merge walk needs to draw a figure legend band: the plan
/// (which nodes band which channels), the per-leaf resolved contexts (to test
/// participation), and the per-leaf captured bundles (the legend content).
struct LegendBandCtx<'a> {
    plan: &'a LegendBandPlan,
    contexts: &'a [LeafScaleContext],
    bundles: &'a [Option<LeafLegendBundle>],
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
    }
}

/// Per-channel legend-resolution state carried down the tree during planning.
#[derive(Debug, Clone, Copy)]
struct ChannelWalk {
    /// A congruent-shared-scale ancestor already resolved this channel, so no
    /// inner node re-resolves it (the resolve pass stops at the outermost such
    /// node — [`super::composite::resolve_channel`]).
    resolved_above: bool,
    /// We are inside a figure-legend band for this channel: participating leaves
    /// below are suppressed.
    band_active: bool,
}

/// Plan the figure-level legends for a composite tree (design §5). Writes the
/// per-leaf `suppress_color_legend`/`suppress_size_legend` flags into `contexts`
/// (read by pass 2's leaf layout) and returns the set of nodes that emit a band.
///
/// A node bands a channel when it is the outermost congruent-shared-**scale**
/// node for that channel (the node where the resolve pass actually unioned the
/// domain) AND its effective legend resolution is `Shared`. A `legend`-
/// independent override over a shared scale therefore bands nothing and
/// suppresses no leaf — today's per-panel rendering (design §4). A leaf is
/// suppressed only when it *participated* in the shared domain (its context
/// carries the channel), so a leaf with an explicit per-chart `scale=` (excluded
/// from the union) keeps its own panel legend.
fn plan_legend_bands(tree: &CompositeNode, contexts: &mut [LeafScaleContext]) -> LegendBandPlan {
    let mut plan = LegendBandPlan { band_nodes: HashMap::new() };
    let mut cursor = 0usize;
    let root = ChannelWalk { resolved_above: false, band_active: false };
    plan_legend_walk(tree, contexts, &mut cursor, root, root, &mut plan);
    plan
}

fn plan_legend_walk(
    node: &CompositeNode,
    contexts: &mut [LeafScaleContext],
    cursor: &mut usize,
    color: ChannelWalk,
    size: ChannelWalk,
    plan: &mut LegendBandPlan,
) {
    match node {
        CompositeNode::Leaf { spec, .. } => {
            let i = *cursor;
            *cursor += 1;
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
        CompositeNode::Composite { children, resolve, .. } => {
            // Must agree bit-for-bit with `resolve_channel`'s congruence
            // gate (composite.rs): the band attaches at exactly the node
            // where the resolve pass unions the domain, and a `Hole` cell
            // (pairplot's corner triangle, jointplot's empty corner) must
            // not disqualify the node's real cells from sharing (see
            // `congruent_non_hole`'s doc).
            let congruent_children = congruent_non_hole(children);
            let (color_next, color_band) = descend_channel(
                resolve.color,
                resolve.effective_color_legend(),
                congruent_children,
                color,
            );
            let (size_next, size_band) = descend_channel(
                resolve.size,
                resolve.effective_size_legend(),
                congruent_children,
                size,
            );
            if color_band || size_band {
                plan.band_nodes.insert(
                    node as *const CompositeNode,
                    BandFlags { color: color_band, size: size_band },
                );
            }
            for c in children {
                plan_legend_walk(c, contexts, cursor, color_next, size_next, plan);
            }
        }
    }
}

/// Advance one channel's [`ChannelWalk`] across a composite node, returning the
/// state its children see and whether THIS node emits a band for the channel.
fn descend_channel(
    scale_mode: ResolveMode,
    eff_legend: ResolveMode,
    congruent_children: bool,
    cur: ChannelWalk,
) -> (ChannelWalk, bool) {
    // A shared legend over a non-shared scale is rejected at lowering (design
    // §4); if one somehow reaches here, treat it as independent (no band) rather
    // than misrender.
    debug_assert!(
        !(eff_legend == ResolveMode::Shared && scale_mode != ResolveMode::Shared),
        "shared legend over non-shared scale must be rejected at lowering"
    );
    let is_scale_resolver =
        !cur.resolved_above && scale_mode == ResolveMode::Shared && congruent_children;
    let band_here = is_scale_resolver && eff_legend == ResolveMode::Shared;
    let next = ChannelWalk {
        resolved_above: cur.resolved_above || is_scale_resolver,
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
    band_ctx: &LegendBandCtx<'_>,
    leaf_range: std::ops::Range<usize>,
    flags: BandFlags,
) {
    let captured = leaf_range.into_iter().find_map(|i| {
        let ctx = &band_ctx.contexts[i];
        let participates =
            (flags.color && ctx.color.is_some()) || (flags.size && ctx.size.is_some());
        if !participates {
            return None;
        }
        match &band_ctx.bundles[i] {
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
    let inner = LayoutRect { x: 0.0, y: 0.0, w: LEGEND_SANDBOX, h: LEGEND_SANDBOX };

    // `!flags.color`: an empty-entries + no-colorbar input makes the shared
    // dispatch a no-op `(None, inner, ..)` via `layout_legend`'s empty-entries
    // early return — matching the pre-extraction "never attempted layout" skip.
    let (color_entries, color_colorbar): (&[LegendEntry], Option<&ColorbarInput>) =
        if flags.color { (&bundle.entries, bundle.colorbar.as_ref()) } else { (&[], None) };
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
                max_tick = max_tick.max(metrics.measure_width(&tk.label, label_fs));
                acc(cb.bar_rect.x, tk.y - line_h / 2.0, cb.bar_rect.x, tk.y + line_h / 2.0);
            }
            acc(
                cb.bar_rect.x,
                cb.bar_rect.y,
                cb.bar_rect.x + cb.bar_rect.w + 4.0 + max_tick,
                cb.bar_rect.y + cb.bar_rect.h,
            );
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
fn build_placed(
    node: &CompositeNode,
    scenes: &mut std::vec::IntoIter<SceneGraph>,
    panel_base: &mut usize,
    leaf_cursor: &mut usize,
    call_theme: &ThemeInputs,
    band_ctx: &LegendBandCtx<'_>,
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
            Placed { scene, width, height }
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
            Placed { scene: empty_scene(w, h), width: w, height: h }
        }
        CompositeNode::Composite { layout, children, spacing, row_ratios, col_ratios, ncols, nrows, .. } => {
            let leaf_start = *leaf_cursor;
            let child_placed: Vec<Placed> = children
                .iter()
                .map(|c| build_placed(c, scenes, panel_base, leaf_cursor, call_theme, band_ctx, Some(*layout)))
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
            let mut merged = merge_children(child_placed, plan);
            // Figure-legend band (design §5 pass 3): if this composite node
            // resolved a channel as a shared figure legend, grow the merged scene
            // on the oriented edge and draw one legend built from the first
            // participating leaf's captured bundle. Applied AFTER children are
            // placed and BEFORE the per-child label / root chrome below, so a
            // title band stacks above the legend exactly as it does for a
            // per-panel legend.
            if let Some(flags) = band_ctx.plan.band_nodes.get(&(node as *const CompositeNode)) {
                apply_legend_band(&mut merged.scene, band_ctx, leaf_start..*leaf_cursor, *flags);
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
        LayoutPlan { placements, width: main, height: cross }
    } else {
        LayoutPlan { placements, width: cross, height: main }
    }
}

/// Overlay: every child at the origin, bbox is the child extent max. Z-order is
/// child order (children merge in order, later children drawn on top).
fn plan_overlay(children: &[Placed]) -> LayoutPlan {
    let width = children.iter().map(|c| c.width).fold(0.0_f64, f64::max);
    let height = children.iter().map(|c| c.height).fold(0.0_f64, f64::max);
    let placements = children.iter().map(|_| translate(0.0, 0.0)).collect();
    LayoutPlan { placements, width, height }
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
    LayoutPlan { placements, width: total_w, height: total_h }
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
        let mut sx = if c.width > 0.0 { col_w[col] / c.width } else { 1.0 };
        let mut sy = if c.height > 0.0 { row_h[r] / c.height } else { 1.0 };
        if (sx - 1.0).abs() < SLOT_MATCH_EPS && (sy - 1.0).abs() < SLOT_MATCH_EPS {
            sx = 1.0;
            sy = 1.0;
        }
        placements[idx] = LayoutScale { sx, sy, tx: col_x[col], ty: row_y[r] };
    }

    let total_w = col_w.iter().sum::<f64>() + spacing * cols.saturating_sub(1) as f64;
    let total_h = row_h.iter().sum::<f64>() + spacing * rows.saturating_sub(1) as f64;
    LayoutPlan { placements, width: total_w, height: total_h }
}

/// `K = min over lanes of (native[i] / ratio[i])` for lanes with positive ratio
/// and native extent; `0.0` when no lane qualifies (all-empty grid). Mirrors
/// the deleted `grid_compose.rs`'s `k_w`/`k_h` derivation.
fn fit_factor(ratios: &[f64], native: &[f64]) -> f64 {
    let k = ratios
        .iter()
        .zip(native)
        .filter_map(|(r, n)| if *r > 0.0 && *n > 0.0 { Some(n / r) } else { None })
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
fn merge_children(children: Vec<Placed>, plan: LayoutPlan) -> Placed {
    let mut merged = empty_scene(plan.width, plan.height);
    let mut zoom = true;
    let mut pan = true;
    let mut toolbar = true;

    for (child, t) in children.into_iter().zip(plan.placements) {
        let mut scene = child.scene;

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
        merged.interaction.linked_panels.append(&mut ci.linked_panels);
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
    Placed { scene: merged, width: plan.width, height: plan.height }
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
    LayoutScale { sx: 1.0, sy: 1.0, tx, ty }
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
    offset_nodes(&mut panel.axes, dx, dy);
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
        PathCmd::CubicTo { c1x, c1y, c2x, c2y, x, y } => {
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
    for node in scene.title.iter_mut().chain(&mut scene.legend).chain(&mut scene.decorations) {
        uniquify_node_raw_clips(node, cell_idx);
    }
    for panel in &mut scene.panels {
        for node in panel
            .grid
            .iter_mut()
            .chain(&mut panel.axes)
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
    let chrome =
        FigureChrome { title, subtitle, caption, left_inset, right_inset, anchor, ..Default::default() };
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
        None => RootChromeConfig { left_inset: None, right_inset: None, anchor: None },
        Some(value) => serde_json::from_value(value.clone()).map_err(|source| {
            CompositeRenderError::RootChromeConfigInvalid { kind, message: source.to_string() }
        })?,
    };
    let anchor = match parsed.anchor.as_deref() {
        None => ChromeAnchor::Start,
        Some(s) => s.parse::<ChromeAnchor>().map_err(|source| {
            CompositeRenderError::RootChromeConfigInvalid { kind, message: source.to_string() }
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
    use crate::render::config::RenderConfig;
    use crate::spec::composite::CompositeResolve;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::DataType as EncDataType;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use crate::layout::SymbolKind;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    // -- builders -------------------------------------------------------------

    fn scatter_spec() -> ChartSpec {
        ChartSpec {
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
        CompositeNode::Leaf { spec: Box::new(scatter_spec()), data, label: None }
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
            viewport: Viewport { width: w, height: ht },
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
                y: Some(EncodingSpec { field: field.into(), ..Default::default() }),
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
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
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
        CompositeNode::Leaf { spec: Box::new(dual_axis_spec()), data, label: None }
    }

    // -- layout math ----------------------------------------------------------

    fn placed_stub(w: f64, h: f64) -> Placed {
        Placed { scene: empty_scene(w, h), width: w, height: h }
    }

    #[test]
    fn merge_children_all_defaults_keep_toolbar_enabled() {
        // Baseline: every leaf resolves the single-chart default
        // (toolbar/zoom/pan all `true`), so an all-defaults composition's
        // merged toolbar must stay `true` — the fold must not accidentally
        // flip the common case off.
        let children = vec![placed_stub(100.0, 50.0), placed_stub(80.0, 60.0)];
        let plan = plan_linear(&children, 10.0, true);
        let merged = merge_children(children, plan);
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
        let merged = merge_children(children, plan);
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
        assert_eq!(plan.placements[0], LayoutScale { sx: 1.0, sy: 1.0, tx: 0.0, ty: 0.0 });
        assert_eq!(plan.placements[3], LayoutScale { sx: 1.0, sy: 1.0, tx: 55.0, ty: 55.0 });
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
        assert!((plan.placements[0].sy - 1.0).abs() < 1e-9, "row0 sy={}", plan.placements[0].sy);
        assert_eq!(plan.placements[0].ty, 0.0);
        // Row 1: shrunk to 1/3.
        assert!((plan.placements[1].sy - (1.0 / 3.0)).abs() < 1e-9, "row1 sy={}", plan.placements[1].sy);
        assert!((plan.placements[1].ty - 100.0).abs() < 1e-9, "row1 ty={}", plan.placements[1].ty);
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
        assert_eq!(plan.width, 170.0, "80 + 10 + 80, unaffected by the hole's 0 width");
        assert_eq!(plan.height, 210.0, "100 + 10 + 100, unaffected by the hole's 0 height");
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
        let inner = LayoutScale { sx: 2.0, sy: 3.0, tx: 1.0, ty: 1.0 };
        let outer = LayoutScale { sx: 10.0, sy: 10.0, tx: 5.0, ty: 5.0 };
        let c = compose(&outer, &inner);
        // point (1,1): inner -> (3,4); outer -> (35,45).
        assert_eq!(c.apply(1.0, 1.0), (35.0, 45.0));
    }

    #[test]
    fn place_panel_pure_translate_bakes_and_keeps_identity() {
        let mut panel = stub_panel();
        place_panel(&mut panel, &translate(20.0, 30.0));
        assert!(panel.layout_scale.is_identity(), "translate placement keeps identity ls");
        assert_eq!(panel.plot_area.x, 20.0);
        assert_eq!(panel.plot_area.y, 30.0);
    }

    #[test]
    fn place_panel_scaling_sets_layout_scale_native_coords() {
        let mut panel = stub_panel();
        let t = LayoutScale { sx: 0.5, sy: 0.25, tx: 10.0, ty: 40.0 };
        place_panel(&mut panel, &t);
        // Native coords untouched; layout_scale carries the whole placement.
        assert_eq!(panel.plot_area.x, 0.0);
        assert_eq!(panel.layout_scale, t);
    }

    #[test]
    fn place_panel_translate_on_nonidentity_ls_adds_translation() {
        let mut panel = stub_panel();
        panel.layout_scale = LayoutScale { sx: 0.5, sy: 0.5, tx: 1.0, ty: 2.0 };
        place_panel(&mut panel, &translate(10.0, 20.0));
        assert_eq!(panel.layout_scale, LayoutScale { sx: 0.5, sy: 0.5, tx: 11.0, ty: 22.0 });
        assert_eq!(panel.plot_area.x, 0.0, "native coords untouched for non-identity ls");
    }

    fn stub_panel() -> Panel {
        use ferrum_scene::CoordKind;
        Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            coord: CoordKind::Cartesian {
                x_domain: None, y_domain: None, expand: false, clip: false,
                y_domains: Vec::new(),
            },
            grid: Vec::new(),
            marks: Vec::new(),
            axes: Vec::new(),
            annotations: Vec::new(),
            strip_title: Vec::new(),
            layout_scale: LayoutScale::identity(),
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
        let mut circle = SceneNode::Circle { cx: 1.0, cy: 2.0, r: 3.0, style: fill.clone() };
        offset_node(&mut circle, 5.0, 7.0);
        assert!(matches!(circle, SceneNode::Circle { cx, cy, .. } if cx == 6.0 && cy == 9.0));

        let mut line = SceneNode::Line { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0, style: stroke };
        offset_node(&mut line, 2.0, 3.0);
        assert!(matches!(line, SceneNode::Line { x1, y1, x2, y2, .. } if x1 == 2.0 && y1 == 3.0 && x2 == 3.0 && y2 == 4.0));

        let mut poly = SceneNode::Polygon { rings: vec![vec![[0.0, 0.0], [1.0, 1.0]]], style: fill };
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
        let mut raw = SceneNode::Raw { svg: "<rect/>".into(), anchor: Default::default() };
        offset_node(&mut raw, 5.0, 8.0);
        if let SceneNode::Raw { svg, .. } = &raw {
            assert!(svg.contains(r#"<g transform="translate(5,8)"><rect/></g>"#), "svg: {svg}");
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
        assert!(matches!(err, CompositeRenderError::LeafCountMismatch { expected: 2, got: 1 }));
    }

    #[test]
    fn hconcat_end_to_end_renumbers_panels_and_sizes_viewport() {
        let tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
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
                assert!(slot1_lo > 50.0, "slot 1 must stay the large independent y1 domain");
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
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 3);
        let ids: Vec<usize> = scene.panels.iter().map(|p| p.id).collect();
        assert_eq!(ids, vec![0, 1, 2], "panels renumbered 0..N pre-order");
    }

    #[test]
    fn grid_ratio_cell_emits_non_identity_layout_scale() {
        // 2 rows x 1 col with row ratios [3,1]: differently-native rows force the
        // small-share row to scale → non-identity layout_scale on that panel.
        let mut tree = composite(CompositeLayout::Grid, vec![leaf_node(0), leaf_node(1)]);
        if let CompositeNode::Composite { nrows, ncols, row_ratios, .. } = &mut tree {
            *nrows = Some(2);
            *ncols = Some(1);
            *row_ratios = Some(vec![3.0, 1.0]);
        }
        let h0 = hold();
        let h1 = hold();
        // Same native size so the ratio (not native disparity) drives scaling.
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 2);
        // Row 0 dominant share → identity (native). Row 1 shrunk → non-identity.
        assert!(scene.panels[0].layout_scale.is_identity(), "row0 should be native");
        assert!(!scene.panels[1].layout_scale.is_identity(), "row1 must carry a ratio layout_scale");
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
                CompositeNode::Hole { width: None, height: None },
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
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 3, "hole must not emit a panel");

        // No ratios and uniform native sizes → uniform grid, every placement
        // is identity-scale pure translation (mirrors
        // `grid_congruent_cells_have_identity_scale`).
        for p in &scene.panels {
            assert!(p.layout_scale.is_identity(), "uniform grid: no cell should scale");
        }
        // Row 1 (leaf1) sits below row 0 (leaf0) by exactly row 0's native
        // height + spacing (200 + 10 = 210) — the hole sharing row 0 does not
        // change row 0's height.
        let row_offset = scene.panels[1].plot_area.y - scene.panels[0].plot_area.y;
        assert!((row_offset - 210.0).abs() < 1e-9, "row offset: {row_offset}");
        assert!(
            (scene.panels[1].plot_area.x - scene.panels[0].plot_area.x).abs() < 1e-9,
            "leaf1 shares leaf0's column"
        );
        // Column 1 (leaf2) sits right of column 0 (leaf1) by exactly column
        // 0's native width + spacing (300 + 10 = 310) — the hole's own
        // (empty) column does not change column 0's width.
        let col_offset = scene.panels[2].plot_area.x - scene.panels[1].plot_area.x;
        assert!((col_offset - 310.0).abs() < 1e-9, "col offset: {col_offset}");
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
                CompositeNode::Hole { width: None, height: None },
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
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 3, "trailing hole must not emit a panel");

        // Row 1 (leaf2, alone — its sibling slot is the hole) sits below row 0
        // by exactly row 0's height + spacing, starting again at column 0.
        let row_offset = scene.panels[2].plot_area.y - scene.panels[0].plot_area.y;
        assert!((row_offset - 210.0).abs() < 1e-9, "row offset: {row_offset}");
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
        let (bare_scene, _warnings) = render_composite_scene(&bare, &leaves, &ThemeInputs::default()).unwrap();
        let bare_y = bare_scene.panels[0].plot_area.y;

        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        // A title band was reserved: canvas grew and panels shifted down.
        assert!(scene.height > bare_scene.height, "chrome must grow the canvas height");
        let header_h = scene.height - bare_scene.height;
        assert!(header_h > 0.0);
        assert!(
            (scene.panels[0].plot_area.y - (bare_y + header_h)).abs() < 1e-9,
            "panel must shift down by exactly the header band height",
        );
        // Chrome text node injected into the title list.
        assert!(
            scene.title.iter().any(|n| matches!(n, SceneNode::Text { content, .. } if content == "Figure title")),
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
                SceneNode::Text { x, style, content, .. } if content == "T" => Some((*x, style.anchor)),
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
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
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
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        let (x, anchor) = title_node_x_anchor(&scene);
        assert_eq!(x, 60.0, "custom left_inset must reposition the start-anchored chrome");
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
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        let (x, anchor) = title_node_x_anchor(&scene);
        assert_eq!(x, scene.width / 2.0, "middle anchor must center on the composed width");
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
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        let (x, anchor) = title_node_x_anchor(&scene);
        assert_eq!(x, scene.width - 40.0, "end anchor must use the custom right_inset");
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
                    message.contains("start") && message.contains("middle") && message.contains("end"),
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
                CompositeNode::Hole { width: Some(50.0), height: Some(150.0) },
                leaf_node(1),
            ],
        );
        assert!(tree.validate().is_ok(), "fully-sized hole is valid under hconcat");

        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
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
                CompositeNode::Hole { width: Some(150.0), height: Some(50.0) },
                leaf_node(1),
            ],
        );
        assert!(tree.validate().is_ok(), "fully-sized hole is valid under vconcat");

        let h0 = hold();
        let h1 = hold();
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
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
                CompositeNode::Hole { width: Some(9999.0), height: Some(9999.0) },
                leaf_node(1),
                leaf_node(2),
            ],
        );
        if let CompositeNode::Composite { nrows, ncols, .. } = &mut tree {
            *nrows = Some(2);
            *ncols = Some(2);
        }
        assert!(tree.validate().is_ok(), "sized hole is valid under grid (fields ignored)");

        let h = [hold(), hold(), hold()];
        let leaves = [
            leaf_input(&h[0], 300.0, 200.0),
            leaf_input(&h[1], 300.0, 200.0),
            leaf_input(&h[2], 300.0, 200.0),
        ];
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert_eq!(scene.panels.len(), 3, "hole must not emit a panel");

        let row_offset = scene.panels[1].plot_area.y - scene.panels[0].plot_area.y;
        assert!((row_offset - 210.0).abs() < 1e-9, "row offset unaffected by hole size: {row_offset}");
        let col_offset = scene.panels[2].plot_area.x - scene.panels[1].plot_area.x;
        assert!((col_offset - 310.0).abs() < 1e-9, "col offset unaffected by hole size: {col_offset}");
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
        let h0 = LeafHold { batch: b0, ..hold() };
        let h1 = LeafHold { batch: b1, spec: spec.clone(), ..hold() };

        let mut tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.x = crate::layout::facet::ResolveMode::Shared;
        }
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let dom = |p: &Panel| match &p.coord {
            ferrum_scene::CoordKind::Cartesian { x_domain, .. } => *x_domain,
            _ => None,
        };
        let d0 = dom(&scene.panels[0]).expect("panel 0 x_domain");
        let d1 = dom(&scene.panels[1]).expect("panel 1 x_domain");
        assert_eq!(d0, d1, "shared-x panels must carry the identical resolved x domain");
        // The shared extent spans BOTH leaves: panel 0's own data maxes at 3.0, so a
        // domain reaching 30.0 proves it absorbed leaf 1's extent (and vice versa) —
        // the discriminator against per-leaf independent resolution.
        assert!(d0.0 <= 1.0, "shared lower extent expected ~1.0, got {}", d0.0);
        assert!(d0.1 >= 30.0, "shared upper extent expected ~30.0, got {}", d0.1);
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
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

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
            raw_svgs.iter().any(|s| s.contains(r#"id="cell0-ferrum-colorbar-0""#)
                && s.contains("url(#cell0-ferrum-colorbar-0)")),
            "leaf 0's colorbar def+ref must be namespaced cell0-...: {raw_svgs:?}"
        );
        assert!(
            raw_svgs.iter().any(|s| s.contains(r#"id="cell1-ferrum-colorbar-0""#)
                && s.contains("url(#cell1-ferrum-colorbar-0)")),
            "leaf 1's colorbar def+ref must be namespaced cell1-...: {raw_svgs:?}"
        );
        // No collision survives: the bare (un-namespaced) id must not leak
        // through, which is exactly what would happen if uniquification were a
        // no-op (the historical gap this test closes).
        assert!(
            !raw_svgs.iter().any(|s| s.contains(r#"id="ferrum-colorbar-0""#)),
            "un-namespaced colorbar id leaked into the merged scene: {raw_svgs:?}"
        );
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
        bad_spec.encoding.x =
            Some(EncodingSpec { field: "missing".into(), ..Default::default() });
        let h0 = LeafHold { spec: bad_spec, ..hold() };
        let h1 = hold();

        let mut tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.x = crate::layout::facet::ResolveMode::Shared;
        }
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let err = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap_err();
        match err {
            CompositeRenderError::LeafRender { kind, index, ref source } => {
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
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(scene.panels.len(), 2);
        // Panel ids are assigned by `renumber_panels` in pre-order BEFORE
        // placement/merge, so id 0 is unambiguously the first-declared child
        // (painted first / on the bottom) and id 1 the second-declared child
        // (painted last / on top) — the vec position IS the z-order.
        assert_eq!(scene.panels[0].id, 0, "first-declared child occupies slot 0 (bottom)");
        assert_eq!(scene.panels[1].id, 1, "second-declared child occupies slot 1 (drawn on top)");
        // Both share the identical overlay rect, confirming this genuinely
        // exercises the overlay (same-rect) case rather than a linear layout
        // where position alone would already prove ordering.
        assert_eq!(
            scene.panels[0].plot_area, scene.panels[1].plot_area,
            "overlay children share one rect; only vec order encodes z-order"
        );
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
        let h1 = LeafHold { batch: xy_batch(&xs, &ys), ..hold() };

        let tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (mut scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

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
        assert!(!packed.is_empty(), "leaf 1's 1200-point batch must trigger packing");

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
        assert_eq!(count, n as u32, "packed instance count must match the source batch size");

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
        let h0 = LeafHold { theme: theme0, ..hold() };
        let h1 = LeafHold { theme: theme1, ..hold() };

        let tree = composite(CompositeLayout::Hconcat, vec![leaf_node(0), leaf_node(1)]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let r0 = first_circle_radius(&scene.panels[0]);
        let r1 = first_circle_radius(&scene.panels[1]);
        assert!(r0 > 0.0 && r1 > 0.0, "both leaves must render circle marks");
        assert!(
            (r0 - r1).abs() > 1e-6,
            "distinct per-leaf point_size must yield distinct radii: r0={r0} r1={r1}"
        );
        // Discriminator: leaf 1's larger point_size must render the larger radius,
        // proving the mapping applied per leaf (not one shared theme).
        assert!(r1 > r0, "larger per-leaf point_size must render the larger radius");
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
            if let CompositeNode::Composite { ncols, .. } = &mut b { *ncols = Some(2); }
            b
        };
        let (bare_scene, _) = render_composite_scene(&bare, &leaves, &ThemeInputs::default()).unwrap();

        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        let (ax, _ay) = find_label(&scene, "Model A").expect("child 0 label present");
        let (bx, _by) = find_label(&scene, "Model B").expect("child 1 label present");
        // Child 0 is at placement tx=0, so its label sits at the default inset.
        assert!(
            (ax - DEFAULT_CHROME_INSET).abs() < 1e-6,
            "child 0 label must sit at the child origin inset, got x={ax}"
        );
        // Child 1 is placed to the right; its label is offset by that placement.
        assert!(bx > ax, "child 1 label must be offset right of child 0: ax={ax} bx={bx}");

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
        let (scene, _warnings) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

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
        let (scene, _warnings) =
            render_composite_scene(&tree, &leaves, &default_theme).unwrap();

        let (font_size, color) =
            find_label_style(&scene, "Model A").expect("labeled leaf's label must be present");
        assert_eq!(font_size, default_theme.typography.title_font_size);
        assert_eq!(color, crate::render::draw::to_scene_color(default_theme.colors.title_color));
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
        assert_eq!(font_size, 30.0, "label must use the call-level theme's title_font_size");
        assert_eq!(
            color,
            ferrum_scene::Color { r: 0x11, g: 0x22, b: 0x33, a: 0xff },
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
        s.encoding.color = Some(EncodingSpec { field: "g".into(), ..Default::default() });
        s
    }

    /// A point spec with categorical color on `g` AND numeric size on `s`
    /// (different fields → color+size do NOT merge; two stacked legends).
    fn color_size_spec() -> ChartSpec {
        let mut spec = cat_color_spec();
        spec.encoding.size = Some(EncodingSpec { field: "s".into(), ..Default::default() });
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
                Arc::new(StringArray::from(gs.iter().map(|s| Some(*s)).collect::<Vec<_>>())),
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
                Arc::new(StringArray::from(gs.iter().map(|s| Some(*s)).collect::<Vec<_>>())),
                Arc::new(Float64Array::from(ss.to_vec())),
            ],
        )
        .unwrap()
    }

    fn color_hold_with(spec: ChartSpec, batch: RecordBatch, theme: ThemeInputs) -> LeafHold {
        LeafHold { spec, batch, theme, config: RenderConfig::default(), chart_config: ChartConfig::default() }
    }

    fn color_hold(gs: &[&str]) -> LeafHold {
        color_hold_with(cat_color_spec(), xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], gs), ThemeInputs::default())
    }

    fn color_hold_oriented(gs: &[&str], orient: LegendOrient) -> LeafHold {
        let mut theme = ThemeInputs::default();
        theme.legend.legend_orient = orient;
        color_hold_with(cat_color_spec(), xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], gs), theme)
    }

    fn color_leaf(data: usize) -> CompositeNode {
        CompositeNode::Leaf { spec: Box::new(cat_color_spec()), data, label: None }
    }

    /// An hconcat of `n` color leaves with the given color resolve mode and an
    /// optional explicit legend override.
    fn color_hconcat(n: usize, color: RM, legend_color: Option<RM>) -> CompositeNode {
        let mut node = composite(CompositeLayout::Hconcat, (0..n).map(color_leaf).collect());
        if let CompositeNode::Composite { resolve, .. } = &mut node {
            resolve.color = color;
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
        let (shared_scene, _) = render_composite_scene(&shared, &leaves, &ThemeInputs::default()).unwrap();

        let indep = color_hconcat(2, RM::Shared, Some(RM::Independent));
        let (indep_scene, _) = render_composite_scene(&indep, &leaves, &ThemeInputs::default()).unwrap();

        assert!(!shared_scene.legend.is_empty(), "shared color must emit one figure legend");
        assert!(!indep_scene.legend.is_empty(), "legend-independent keeps per-panel legends");
        assert!(
            shared_scene.legend.len() < indep_scene.legend.len(),
            "one figure legend ({}) must draw fewer nodes than two per-panel legends ({})",
            shared_scene.legend.len(),
            indep_scene.legend.len(),
        );
        assert_eq!(shared_scene.panels.len(), 2, "band must not add or drop panels");
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
        let (shared_scene, _) = render_composite_scene(&shared, &leaves, &ThemeInputs::default()).unwrap();
        assert!(shared_scene.width > 610.0, "right band must grow width past the grid: {}", shared_scene.width);
        assert!((shared_scene.height - 200.0).abs() < 1e-6, "right band must not grow height");

        let indep = color_hconcat(2, RM::Shared, Some(RM::Independent));
        let (indep_scene, _) = render_composite_scene(&indep, &leaves, &ThemeInputs::default()).unwrap();
        assert!((indep_scene.width - 610.0).abs() < 1e-6, "per-panel legends must not grow the grid width: {}", indep_scene.width);
    }

    #[test]
    fn shared_color_band_grows_top_and_shifts_panels_down() {
        let h0 = color_hold_oriented(&["a", "b", "a"], LegendOrient::Top);
        let h1 = color_hold_oriented(&["a", "b", "a"], LegendOrient::Top);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let tree = color_hconcat(2, RM::Shared, None);
        let (scene, _) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert!(scene.height > 200.0, "top band must grow scene height: {}", scene.height);
        assert!((scene.width - 610.0).abs() < 1e-6, "top band must not grow width: {}", scene.width);
        assert!(scene.panels[0].plot_area.y > 8.0, "top band must shift panels down: {}", scene.panels[0].plot_area.y);
    }

    #[test]
    fn shared_color_band_left_shifts_panels_right() {
        let h0 = color_hold_oriented(&["a", "b", "a"], LegendOrient::Left);
        let h1 = color_hold_oriented(&["a", "b", "a"], LegendOrient::Left);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let tree = color_hconcat(2, RM::Shared, None);
        let (scene, _) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert!(scene.width > 610.0, "left band must grow scene width: {}", scene.width);
        // Panel 0's plot area is pushed right by the legend gutter.
        let right_h = color_hold(&["a", "b", "a"]);
        let right_leaves = [leaf_input(&right_h, 300.0, 200.0), leaf_input(&right_h, 300.0, 200.0)];
        let (right_scene, _) = render_composite_scene(&tree, &right_leaves, &ThemeInputs::default()).unwrap();
        assert!(
            scene.panels[0].plot_area.x > right_scene.panels[0].plot_area.x,
            "left band shifts panels right ({} vs right-orient {})",
            scene.panels[0].plot_area.x, right_scene.panels[0].plot_area.x,
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
                LegendEntry { label: "a".into(), symbol: SymbolKind::Circle },
                LegendEntry { label: "b".into(), symbol: SymbolKind::Circle },
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
        let flags = BandFlags { color: true, size: false };

        for orient in
            [LegendOrient::Right, LegendOrient::Left, LegendOrient::Top, LegendOrient::Bottom]
        {
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
                LegendOrient::Right | LegendOrient::Left => {
                    scene.width - 300.0 - LEGEND_PLOT_GAP
                }
                LegendOrient::Top | LegendOrient::Bottom => {
                    scene.height - 200.0 - LEGEND_PLOT_GAP
                }
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
        let flags = BandFlags { color: true, size: false };
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
        draw_legend_band(&mut scene, &bundle, BandFlags { color: true, size: false });

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
                resolve.color = RM::Shared;
            }
            composite(CompositeLayout::Hconcat, vec![leaf_node(2), inner])
        };
        let inner_indep = {
            let mut inner = composite(CompositeLayout::Hconcat, vec![color_leaf(0), color_leaf(1)]);
            if let CompositeNode::Composite { resolve, .. } = &mut inner {
                resolve.color = RM::Shared;
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
        let (shared_scene, _) = render_composite_scene(&inner_shared, &ordered, &ThemeInputs::default()).unwrap();
        let (indep_scene, _) = render_composite_scene(&inner_indep, &ordered, &ThemeInputs::default()).unwrap();
        assert!(!shared_scene.legend.is_empty(), "inner-shared color must emit one band");
        assert!(
            shared_scene.legend.len() < indep_scene.legend.len(),
            "nested band ({}) must draw fewer legend nodes than per-panel ({})",
            shared_scene.legend.len(), indep_scene.legend.len(),
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
            s.encoding.color.as_mut().unwrap().legend =
                Some(Box::new(LegendStyleSpec { disabled: Some(true), ..Default::default() }));
            s
        };
        let h0 = color_hold_with(disabled_spec(), xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &["a", "b", "a"]), ThemeInputs::default());
        let h1 = color_hold_with(disabled_spec(), xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &["a", "b", "a"]), ThemeInputs::default());
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        // Tree leaf specs must also carry the disabled legend so planning sees it.
        let mut tree = composite(
            CompositeLayout::Hconcat,
            vec![
                CompositeNode::Leaf { spec: Box::new(disabled_spec()), data: 0, label: None },
                CompositeNode::Leaf { spec: Box::new(disabled_spec()), data: 1, label: None },
            ],
        );
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.color = RM::Shared;
        }
        let (scene, _) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert!(scene.legend.is_empty(), "all-disabled participants → no figure band");
        assert!((scene.width - 610.0).abs() < 1e-6, "no band → no width growth: {}", scene.width);
    }

    #[test]
    fn capture_skips_disabled_leaf_and_uses_next_participant() {
        // Leaf 0 disables its color legend; leaf 1 keeps it. The band captures
        // leaf 1's non-empty bundle, so exactly one band is still emitted.
        use crate::render::chart_config::LegendStyleSpec;
        let mut disabled = cat_color_spec();
        disabled.encoding.color.as_mut().unwrap().legend =
            Some(Box::new(LegendStyleSpec { disabled: Some(true), ..Default::default() }));
        let h0 = color_hold_with(disabled.clone(), xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &["a", "b", "a"]), ThemeInputs::default());
        let h1 = color_hold(&["a", "b", "a"]);
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let mut tree = composite(
            CompositeLayout::Hconcat,
            vec![
                CompositeNode::Leaf { spec: Box::new(disabled), data: 0, label: None },
                color_leaf(1),
            ],
        );
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.color = RM::Shared;
        }
        let (scene, _) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();
        assert!(!scene.legend.is_empty(), "band must be captured from the non-disabled leaf 1");
        assert!(scene.width > 610.0, "band from leaf 1 still grows the scene: {}", scene.width);
    }

    #[test]
    fn explicit_leaf_color_scale_keeps_its_own_panel_legend() {
        // Leaf 1 pins an explicit ordinal color scale → excluded from the shared
        // union, so it renders its own panel legend while leaf 0 (participant) is
        // deduped into the figure band. Result: MORE legend nodes than a tree
        // where both participate (band only).
        use crate::spec::encoding::ScaleSpec;
        let mut excl_spec = cat_color_spec();
        excl_spec.encoding.color.as_mut().unwrap().scale =
            Some(ScaleSpec::Ordinal { domain: None, range: None, padding: 0.0 });
        let h0 = color_hold(&["a", "b", "a"]);
        let h1 = color_hold_with(excl_spec.clone(), xyg_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &["a", "b", "a"]), ThemeInputs::default());
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let mut tree = composite(
            CompositeLayout::Hconcat,
            vec![color_leaf(0), CompositeNode::Leaf { spec: Box::new(excl_spec), data: 1, label: None }],
        );
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.color = RM::Shared;
        }
        let (scene, _) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        // Baseline: both participate (no explicit scale) → single band only.
        let both = color_hold(&["a", "b", "a"]);
        let both_leaves = [leaf_input(&both, 300.0, 200.0), leaf_input(&both, 300.0, 200.0)];
        let both_tree = color_hconcat(2, RM::Shared, None);
        let (both_scene, _) = render_composite_scene(&both_tree, &both_leaves, &ThemeInputs::default()).unwrap();

        assert!(
            scene.legend.len() > both_scene.legend.len(),
            "excluded leaf's own legend ({}) must add nodes beyond the lone band ({})",
            scene.legend.len(), both_scene.legend.len(),
        );
    }

    #[test]
    fn shared_color_and_size_stack_both_in_one_band() {
        // A leaf with color (categorical) + size (numeric) on DIFFERENT fields,
        // sharing both channels on one node → one band containing the color
        // legend AND the size aux block: more legend nodes than a color-only band.
        let batch = xycs_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &["a", "b", "a"], &[2.0, 5.0, 9.0]);
        let h0 = color_hold_with(color_size_spec(), batch.clone(), ThemeInputs::default());
        let h1 = color_hold_with(color_size_spec(), batch, ThemeInputs::default());
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];
        let cs_leaf = |data| CompositeNode::Leaf { spec: Box::new(color_size_spec()), data, label: None };
        let mut tree = composite(CompositeLayout::Hconcat, vec![cs_leaf(0), cs_leaf(1)]);
        if let CompositeNode::Composite { resolve, .. } = &mut tree {
            resolve.color = RM::Shared;
            resolve.size = RM::Shared;
        }
        let (both_scene, _) = render_composite_scene(&tree, &leaves, &ThemeInputs::default()).unwrap();

        // Color-only shared baseline (size independent → size legend stays per-panel,
        // not in the band).
        let color_only = color_hconcat(2, RM::Shared, None);
        let ch = color_hold(&["a", "b", "a"]);
        let color_leaves = [leaf_input(&ch, 300.0, 200.0), leaf_input(&ch, 300.0, 200.0)];
        let (color_scene, _) = render_composite_scene(&color_only, &color_leaves, &ThemeInputs::default()).unwrap();

        assert!(!both_scene.legend.is_empty(), "color+size share must emit a band");
        assert!(
            both_scene.legend.len() > color_scene.legend.len(),
            "stacked color+size band ({}) must draw more nodes than a color-only band ({})",
            both_scene.legend.len(), color_scene.legend.len(),
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
        let (override_scene, _) = render_composite_scene(&override_tree, &leaves, &ThemeInputs::default()).unwrap();

        let independent_scale = color_hconcat(2, RM::Independent, None);
        let (indep_scene, _) = render_composite_scene(&independent_scale, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(
            override_scene.legend.len(), indep_scene.legend.len(),
            "legend-independent over a shared scale must draw the same per-panel legends as an independent scale",
        );
        assert!((override_scene.width - 610.0).abs() < 1e-6, "no band → no growth: {}", override_scene.width);
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
        s.encoding.size = Some(EncodingSpec { field: "s".into(), ..Default::default() });
        s
    }

    fn size_leaf(data: usize) -> CompositeNode {
        CompositeNode::Leaf { spec: Box::new(size_only_spec()), data, label: None }
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
            resolve.size = RM::Shared;
        }
        let (shared_scene, _) = render_composite_scene(&shared, &leaves, &ThemeInputs::default()).unwrap();

        let mut indep = composite(CompositeLayout::Hconcat, vec![size_leaf(0), size_leaf(1)]);
        if let CompositeNode::Composite { resolve, .. } = &mut indep {
            resolve.size = RM::Independent;
        }
        let (indep_scene, _) = render_composite_scene(&indep, &leaves, &ThemeInputs::default()).unwrap();

        assert!(!shared_scene.legend.is_empty(), "shared size must emit one figure legend band");
        assert!(!indep_scene.legend.is_empty(), "independent size keeps per-panel size legends");
        assert!(
            shared_scene.legend.len() < indep_scene.legend.len(),
            "one figure size band ({}) must draw fewer nodes than two per-panel size legends ({})",
            shared_scene.legend.len(), indep_scene.legend.len(),
        );
        assert_eq!(shared_scene.panels.len(), 2, "band must not add or drop panels");
    }

    /// A point spec with color AND size mapped to the SAME numeric field `s` —
    /// `leaf_merges_color_size` folds size into the color legend for this leaf.
    fn merged_color_size_spec() -> ChartSpec {
        let mut s = scatter_spec();
        s.encoding.color = Some(EncodingSpec { field: "s".into(), ..Default::default() });
        s.encoding.size = Some(EncodingSpec { field: "s".into(), ..Default::default() });
        s
    }

    fn merged_leaf(data: usize) -> CompositeNode {
        CompositeNode::Leaf { spec: Box::new(merged_color_size_spec()), data, label: None }
    }

    #[test]
    fn same_field_color_size_merge_bands_folded_size_and_suppresses_both_channels() {
        // color+size on the SAME field ("s") merges into one legend per leaf
        // (`leaf_merges_color_size`), so sharing color ALONE (size resolve stays
        // Independent, the default) must still band the folded size content —
        // `layout_band_legends`'s `include_size = flags.color && merged_color_size`
        // arm — and suppress BOTH panel channels together, not just color.
        let batch = xys_batch(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[2.0, 5.0, 9.0]);
        let h0 = color_hold_with(merged_color_size_spec(), batch.clone(), ThemeInputs::default());
        let h1 = color_hold_with(merged_color_size_spec(), batch, ThemeInputs::default());
        let leaves = [leaf_input(&h0, 300.0, 200.0), leaf_input(&h1, 300.0, 200.0)];

        let mut shared = composite(CompositeLayout::Hconcat, vec![merged_leaf(0), merged_leaf(1)]);
        if let CompositeNode::Composite { resolve, .. } = &mut shared {
            resolve.color = RM::Shared;
        }
        let (shared_scene, _) = render_composite_scene(&shared, &leaves, &ThemeInputs::default()).unwrap();

        // Baseline A: same tree, resolve stays default (Independent) → both
        // panels keep their OWN already-merged color+size legend (nothing
        // compositor-suppressed) — proves suppression collapsed two per-panel
        // merged blocks into one band.
        let unshared = composite(CompositeLayout::Hconcat, vec![merged_leaf(0), merged_leaf(1)]);
        let (unshared_scene, _) = render_composite_scene(&unshared, &leaves, &ThemeInputs::default()).unwrap();

        // Baseline B: a plain categorical color-only band (different field,
        // no size at all) — proves the merged band carries MORE than bare
        // color, i.e. the folded size swatches are actually present.
        let color_only = color_hconcat(2, RM::Shared, None);
        let ch = color_hold(&["a", "b", "a"]);
        let color_leaves = [leaf_input(&ch, 300.0, 200.0), leaf_input(&ch, 300.0, 200.0)];
        let (color_scene, _) = render_composite_scene(&color_only, &color_leaves, &ThemeInputs::default()).unwrap();

        assert!(!shared_scene.legend.is_empty(), "same-field color+size merge must emit a band");
        assert!(
            shared_scene.legend.len() < unshared_scene.legend.len(),
            "one band folding both suppressed channels ({}) must draw fewer nodes than two \
             per-panel merged legends ({})",
            shared_scene.legend.len(), unshared_scene.legend.len(),
        );
        assert!(
            shared_scene.legend.len() > color_scene.legend.len(),
            "band with folded size content ({}) must draw more nodes than a color-only band ({})",
            shared_scene.legend.len(), color_scene.legend.len(),
        );
        assert_eq!(shared_scene.panels.len(), 2, "band must not add or drop panels");
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
                resolve.color = RM::Shared;
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

        assert!(!scene.legend.is_empty(), "nested Top-orient share must still emit a band");
        assert!(scene.height > 200.0, "top band must grow scene height even when nested: {}", scene.height);
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
    fn color_grid(children: Vec<CompositeNode>, nrows: u32, ncols: u32, color: RM) -> CompositeNode {
        let mut node = composite(CompositeLayout::Grid, children);
        if let CompositeNode::Composite { resolve, nrows: nr, ncols: nc, .. } = &mut node {
            resolve.color = color;
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
                CompositeNode::Hole { width: None, height: None },
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
        let (indep_scene, _) = render_composite_scene(&indep, &leaves, &ThemeInputs::default()).unwrap();

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
                CompositeNode::Hole { width: None, height: None },
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
        let (hole_scene, _) = render_composite_scene(&with_hole, &leaves, &ThemeInputs::default()).unwrap();

        let no_hole = color_hconcat(2, RM::Shared, None);
        let (baseline_scene, _) = render_composite_scene(&no_hole, &leaves, &ThemeInputs::default()).unwrap();

        assert_eq!(hole_scene.panels.len(), 2, "the leading hole must not claim a panel");
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
