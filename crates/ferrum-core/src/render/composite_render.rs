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
//!    F20 ratio math absorbed from [`super::grid_compose`]), wrap (`ncols`), and
//!    overlay (children share one region, z-order = child order).
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

use arrow::record_batch::RecordBatch;
use ferrum_scene::{LayoutScale, MarkBatch, Panel, Rect, SceneGraph, SceneNode};

use crate::layout::{ThemeInputs, Viewport};
use crate::spec::chart::ChartSpec;

use super::chart_config::ChartConfig;
use super::composite::{
    flatten_leaf_specs, resolve_composite_scales, CompositeResolveError, LeafResolveInput,
    LeafScaleContext,
};
use super::compositor::uniquify_clip_ids;
use super::config::RenderConfig;
use super::figure_chrome::{title_nodes, FigureChrome};
use super::{prepare, scene_build, RenderError, RenderWarning};
use crate::spec::composite::{CompositeLayout, CompositeNode};

/// Default pixel gap between adjacent cells, matching the composition binding's
/// `spacing = 10.0` default (`compose_svg_horizontal`/`_vertical`) so composites
/// stay visually equivalent to the string-compositor path they replace.
const DEFAULT_SPACING: f64 = 10.0;

/// Slack for the "slot matches native" comparison — mirrors `grid_compose.rs`'s
/// `near_eq` (`1e-6`), so a ratio cell whose allocation equals its native size
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
    let contexts = resolve_composite_scales(tree, &resolve_inputs)?;
    drop(resolve_inputs);
    drop(prepared);

    // Pass 2/3 (per-leaf render): re-render each leaf with its resolved-domain
    // context so composite-shared channels land on the auto scale path (D4b). A
    // fully-empty context passes `None` so non-shared leaves render byte-identical
    // to a standalone chart.
    let mut leaf_scenes: Vec<SceneGraph> = Vec::with_capacity(n);
    let mut warnings: Vec<RenderWarning> = Vec::new();
    for (i, leaf) in leaves.iter().enumerate() {
        let ctx = &contexts[i];
        let ctx_opt = if ctx.x.is_some() || ctx.y.is_some() { Some(ctx) } else { None };
        let (mut scene, leaf_warnings) = render_leaf(leaf, ctx_opt).map_err(|source| {
            CompositeRenderError::LeafRender { kind: "leaf", index: i, source }
        })?;
        // Uniquify each leaf's raw-fragment clip ids exactly once, keyed by the
        // leaf's global pre-order index so colorbar/legend-clip/inset def ids stay
        // disjoint across the composite (panel clips are auto-unique via the
        // global panel renumber below).
        uniquify_scene_raw_clips(&mut scene, i);
        leaf_scenes.push(scene);
        // Aggregated in leaf pre-order (the same order `leaves`/`flatten_leaf_specs`
        // produce), matching `render_svg`'s single-scene warning contract.
        warnings.extend(leaf_warnings);
    }

    // Pass 2/3 (place + merge): walk the tree, placing each leaf scene into the
    // composite frame. Panels are renumbered flat in pre-order as leaf scenes are
    // consumed (D4c).
    let mut scenes = leaf_scenes.into_iter();
    let mut panel_base = 0usize;
    let mut placed = build_placed(tree, &mut scenes, &mut panel_base, call_theme);

    // Root figure chrome (title/subtitle/caption) — validated root-only.
    if let CompositeNode::Composite { title, subtitle, caption, .. } = tree {
        inject_root_chrome(
            &mut placed.scene,
            title.as_deref(),
            subtitle.as_deref(),
            caption.as_deref(),
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
) -> Result<(SceneGraph, Vec<RenderWarning>), RenderError> {
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
    Ok((scene, po.warnings))
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
/// `scenes` in pre-order; `panel_base` is the running global panel-id offset,
/// incremented as each leaf's panels are renumbered. `call_theme` styles any
/// per-child label encountered (see [`apply_child_label`]).
fn build_placed(
    node: &CompositeNode,
    scenes: &mut std::vec::IntoIter<SceneGraph>,
    panel_base: &mut usize,
    call_theme: &ThemeInputs,
) -> Placed {
    let mut placed = match node {
        CompositeNode::Leaf { .. } => {
            let mut scene = scenes
                .next()
                .expect("leaf scenes count matches tree leaves (checked by entry)");
            renumber_panels(&mut scene, *panel_base);
            *panel_base += scene.panels.len();
            let (width, height) = (scene.width, scene.height);
            Placed { scene, width, height }
        }
        CompositeNode::Composite { layout, children, spacing, row_ratios, col_ratios, ncols, nrows, .. } => {
            let child_placed: Vec<Placed> = children
                .iter()
                .map(|c| build_placed(c, scenes, panel_base, call_theme))
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
            merge_children(child_placed, plan)
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

/// Grid: row-major placement with F20 ratio math (absorbed from
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
/// `grid_compose.rs`'s `k_w`/`k_h` derivation.
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
    }

    merged.interaction.zoom_enabled = zoom;
    merged.interaction.pan_enabled = pan;
    Placed { scene: merged, width: plan.width, height: plan.height }
}

/// A fresh empty merge target sized to `(w, h)`.
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
/// per-leaf `cell_idx` prefix, mirroring `compositor::uniquify_clip_ids` for the
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
/// No-op when all three are `None`.
fn inject_root_chrome(
    scene: &mut SceneGraph,
    title: Option<&str>,
    subtitle: Option<&str>,
    caption: Option<&str>,
) {
    let chrome = FigureChrome { title, subtitle, caption, ..Default::default() };
    apply_chrome_band(scene, chrome);
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

    // -- layout math ----------------------------------------------------------

    fn placed_stub(w: f64, h: f64) -> Placed {
        Placed { scene: empty_scene(w, h), width: w, height: h }
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
            coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: false, clip: false },
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
}
