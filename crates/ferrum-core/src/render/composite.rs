//! Composite scale-resolution pass — Phase B composite-render-unification (Task 4).
//!
//! This is pass 1 of the three-pass composite renderer (design spec §5). It
//! walks a [`CompositeNode`] tree and computes, per shared positional channel,
//! one resolved domain across the tree's leaves — through the *same* mechanism
//! facets use ([`numeric_domain_union`] / [`positional_category_union`]),
//! with tree-path pairing for congruent grid children and skip for non-congruent
//! groups (spec §6, decision D2). The output is a per-leaf
//! [`LeafScaleContext`]; the layout/scene passes (Task 5) thread it into each
//! leaf's render so composite-shared leaves resolve on the **auto scale path**
//! (facet padding/`nice`), never the explicit-scale bypass (decision D4b).
//!
//! # What this pass does NOT do
//!
//! No layout, no panel placement, no scene construction — those are Task 5. This
//! pass is a pure function over *already-prepared* leaf data
//! ([`LeafResolveInput`]): the caller runs each leaf's transforms first (via the
//! existing `prepare_render_inputs`) and hands the resolve pass the leaf's
//! rendering encoding, final batch, and named transform outputs. Keeping the pass
//! pure over those pieces (rather than over the heavy `PreparedInputs`) is what
//! makes the congruence/union logic unit-testable in isolation.
//!
//! # Seam for Task 5 (D4b)
//!
//! [`SharedDomain`] is shaped to drop straight onto `build_axis_scale`'s auto
//! path: a [`SharedDomain::Numeric`] seeds the linear/temporal extent; a
//! [`SharedDomain::Ordinal`] seeds the category vector. Task 5 adds an
//! `Option<&LeafScaleContext>` parameter to `prepare_and_layout` /
//! `resolve_panel_scales` / `build_axis_scale`, consulted where `include_final`
//! is consulted today: when a channel carries `Some(domain)`, the leaf's
//! auto-inferred domain is replaced by the shared one (padding/`nice` still
//! applied). A leaf channel carrying a genuine user `enc.scale` is excluded here
//! (returns `None`), so it still short-circuits at the existing explicit-scale
//! bypass — user scale wins over sharing (spec §6).
//!
//! [`SharedDomain`], [`LeafScaleContext`], and [`Channel`] are homed in
//! `render::scale_resolve` (`scale_resolve::seam`), not here: they are the
//! lower layer's own vocabulary (both this resolve pass and
//! `scale_resolve`'s builders consume them), so the dependency points one
//! way — this module depends on `scale_resolve`, never the reverse (72h
//! findings burndown item 1; pure move, no behavior change).
//!
//! Everything in this module is consumed by Task 5b's composite render core
//! ([`crate::render::composite_render`]) and the D4b scale seam, which reach it
//! from within the crate.
use crate::render::scale_resolve::{
    distinct_values_in_order, extend_extent_to_zero, infer_spec_type, locate_field, locate_seed_field,
    establishing_caller_scale, numeric_domain_union_over, numeric_extent, positional_category_union,
    column_reads_categorical, warn_members_off_caller_scale, AxisMember, AxisMembers, Channel, LayerUnion, LeafScaleContext, SharedDomain,
};
use crate::render::{ExcludedLayerPlacement, RenderError, RenderWarning};
use crate::spec::chart::ChartSpec;
use crate::spec::composite::{CompositeLayout, CompositeNode};
use crate::spec::encoding::{DataType as SpecDataType, Encoding};
use crate::layout::facet::ResolveMode;

use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Public output types
// ---------------------------------------------------------------------------

/// Already-prepared per-leaf inputs the resolve pass unions over. The caller
/// (Task 5) builds one per leaf from that leaf's `PreparedInputs`:
/// `encoding` = the rendering (post-CoordFlip, layer-0-merged) encoding,
/// `final_batch` = `prep.final_batch()`, `transform_outputs` =
/// `prep.transform_outputs`. `spec` carries the leaf's layers/facet, which the
/// domain union consults (layer whisker fields, facet resolve mode).
pub(crate) struct LeafResolveInput<'a> {
    pub(crate) spec: &'a ChartSpec,
    pub(crate) encoding: &'a Encoding,
    pub(crate) final_batch: &'a RecordBatch,
    pub(crate) transform_outputs: &'a HashMap<String, RecordBatch>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A failure resolving composite shared scales. Every variant names the offending
/// composition node kind and/or channel so a Python `ValueError` pinpoints it,
/// matching the `CompositeSpecError` / `RenderError` idiom. No warning-fallbacks.
#[derive(Debug)]
pub(crate) enum CompositeResolveError {
    /// The flattened `leaves` slice length did not match the tree's leaf count —
    /// the caller assembled per-leaf inputs out of step with the tree.
    LeafCountMismatch {
        kind: &'static str,
        expected: usize,
        got: usize,
    },
    /// A shared channel referenced a field absent from the leaf's batches.
    /// `channel` is the RESOLVED (post-`CoordFlip`) token (`Channel::as_str()`)
    /// — `coord_flipped` (R3) lets `Display` un-flip `x`/`y` back to what the
    /// leaf's user actually wrote; `color`/`size` are unaffected either way.
    UnknownField {
        channel: &'static str,
        field: String,
        coord_flipped: bool,
    },
    /// The underlying scale resolver failed to union a leaf's channel domain.
    /// See [`Self::UnknownField`] for `channel`/`coord_flipped`.
    DomainResolve {
        channel: &'static str,
        field: String,
        source: RenderError,
        coord_flipped: bool,
    },
}

impl fmt::Display for CompositeResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LeafCountMismatch { kind, expected, got } => write!(
                f,
                "{kind}: composite leaf count mismatch: tree has {expected} leaves, got {got} prepared inputs"
            ),
            Self::UnknownField { channel, field, coord_flipped } => {
                let channel = crate::render::prepare::user_facing_channel(channel, *coord_flipped);
                write!(f, "shared {channel} channel references unknown field '{field}'")
            }
            Self::DomainResolve { channel, field, source, coord_flipped } => {
                let channel = crate::render::prepare::user_facing_channel(channel, *coord_flipped);
                write!(f, "failed to resolve shared {channel} domain for field '{field}': {source}")
            }
        }
    }
}

impl std::error::Error for CompositeResolveError {}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Read the positional resolve mode for `channel` off a composite node's
/// [`CompositeResolve`]. Positional channels (`x`/`y`) are node-local with a
/// hard `Independent` default; the non-positional channels (`color`/`size`)
/// carry an unset (`None`) state and are read through [`channel_resolve`]
/// instead, so this helper treats an unset color/size as `Independent`.
fn resolve_mode(resolve: &crate::spec::composite::CompositeResolve, channel: Channel) -> ResolveMode {
    match channel {
        Channel::X => resolve.x,
        Channel::Y => resolve.y,
        Channel::Color => resolve.color.unwrap_or(ResolveMode::Independent),
        Channel::Size => resolve.size.unwrap_or(ResolveMode::Independent),
    }
}

/// Read the *explicit* non-positional resolve mode for `channel`, or `None`
/// when the node left it unset (inherit the ancestor's effective mode — spec
/// §6, GH #74). Only `Color`/`Size` carry this unset state.
fn channel_resolve(
    resolve: &crate::spec::composite::CompositeResolve,
    channel: Channel,
) -> Option<ResolveMode> {
    match channel {
        Channel::Color => resolve.color,
        Channel::Size => resolve.size,
        Channel::X | Channel::Y => Some(resolve_mode(resolve, channel)),
    }
}

/// THE single source of the non-positional (color/size) effective-mode gate
/// (spec §6, GH #74).
///
/// Given a node's *explicit* resolve for a channel (`node_scale`: `None` = unset,
/// inherit) and the effective mode `inherited` from its ancestors, returns
/// `(eff, is_outermost)` where:
///
/// - `eff = node_scale.unwrap_or(inherited)` is the node's effective mode, threaded
///   down to its children.
/// - `is_outermost` is `true` exactly when this node is the *outermost*
///   effective-shared node for the channel (`eff == Shared` while
///   `inherited != Shared`).
///
/// Both walks over the composite tree must agree bit-for-bit on this gate: the
/// resolve pass ([`resolve_nonpositional`]) unions the channel's domain at every
/// `is_outermost` node, and the legend-band pass
/// ([`super::composite_render::plan_legend_walk`]'s `descend_channel`) attaches a
/// figure legend band at exactly the same nodes. Sharing one function makes that
/// lockstep compile-time, not a mirrored predicate maintained by discipline.
pub(crate) fn effective_share(
    node_scale: Option<ResolveMode>,
    inherited: ResolveMode,
) -> (ResolveMode, bool) {
    let eff = node_scale.unwrap_or(inherited);
    let is_outermost = eff == ResolveMode::Shared && inherited != ResolveMode::Shared;
    (eff, is_outermost)
}

/// Resolve shared domains across a composite tree, for both positional (x/y) and
/// non-positional (color/size) channels.
///
/// `leaves` must be the tree's leaves in pre-order (left-to-right depth-first),
/// the same order [`flatten_leaf_specs`] produces — leaf *i* in the returned
/// `Vec<LeafScaleContext>` is leaf *i* of the tree. Composition default is
/// independent (empty contexts); each `resolve={channel: shared}` node unions one
/// domain across its children per the congruence rules.
///
/// Precondition: `tree` has passed [`CompositeNode::validate`] (the sole
/// Python→Rust construction path enforces this). In particular every composite
/// node has non-empty `children` — `resolve_channel` indexes `children[0]`
/// after a vacuously-true congruence check and would panic on an unvalidated
/// empty node.
///
/// `warnings` receives one [`RenderWarning::LayerScaleExcluded`] per `Overlay`
/// member a caller's scale leaves out of the group's axis.
pub(crate) fn resolve_composite_scales(
    tree: &CompositeNode,
    leaves: &[LeafResolveInput<'_>],
    warnings: &mut Vec<RenderWarning>,
) -> Result<Vec<LeafScaleContext>, CompositeResolveError> {
    let n = leaf_count(tree);
    if leaves.len() != n {
        return Err(CompositeResolveError::LeafCountMismatch {
            kind: tree.kind_name(),
            expected: n,
            got: leaves.len(),
        });
    }
    let mut out = vec![LeafScaleContext::default(); n];
    let span: Vec<usize> = (0..n).collect();
    // Positional x/y: tree-path pairing across congruent direct children
    // (grids pair by column — pairplot/compare), node-local, no inheritance.
    for channel in [Channel::X, Channel::Y] {
        resolve_channel(tree, &span, channel, leaves, &mut out, warnings)?;
    }
    // Non-positional color/size: leaf-span union at each effective-shared node
    // (GH #74). A node's effective mode is its explicit setting if any, else
    // the nearest ancestor's effective mode (spec §6); the union spans every
    // descendant leaf whose effective mode stays shared, through nested
    // composites and spliced overlay subtrees.
    for channel in [Channel::Color, Channel::Size] {
        resolve_nonpositional(tree, &span, channel, ResolveMode::Independent, leaves, &mut out)?;
    }
    Ok(out)
}

/// Collect the tree's leaves in pre-order as `(spec, data-index)` pairs — the
/// canonical leaf order the resolve pass (and Task 5's panel emission) index by.
/// The PyO3 composite entries pair each leaf's `spec` with the Arrow payload its
/// `data` index selects; [`flatten_leaf_specs`] is the spec-only projection of
/// this same traversal.
pub(crate) fn flatten_leaves(tree: &CompositeNode) -> Vec<(&ChartSpec, usize)> {
    fn walk<'a>(node: &'a CompositeNode, acc: &mut Vec<(&'a ChartSpec, usize)>) {
        match node {
            CompositeNode::Leaf { spec, data, .. } => acc.push((spec, *data)),
            CompositeNode::Composite { children, .. } => {
                for c in children {
                    walk(c, acc);
                }
            }
            // A hole carries no spec/data — it is not a leaf and contributes
            // nothing to the flattened leaf list (Task 8a).
            CompositeNode::Hole { .. } => {}
        }
    }
    let mut acc = Vec::new();
    walk(tree, &mut acc);
    acc
}

/// Collect the tree's leaf `ChartSpec`s in pre-order — the canonical leaf order
/// the resolve pass (and Task 5's panel emission) index by. Exposed so the caller
/// can build [`LeafResolveInput`]s in the matching order without re-deriving the
/// traversal. The spec-only projection of [`flatten_leaves`].
pub(crate) fn flatten_leaf_specs(tree: &CompositeNode) -> Vec<&ChartSpec> {
    flatten_leaves(tree).into_iter().map(|(spec, _)| spec).collect()
}

// ---------------------------------------------------------------------------
// Tree walk
// ---------------------------------------------------------------------------

/// Leaves under `node`, in pre-order. A hole is not a leaf — it contributes 0
/// (Task 8a), so it occupies an empty leaf span when a parent partitions its
/// span among children by leaf count.
fn leaf_count(node: &CompositeNode) -> usize {
    match node {
        CompositeNode::Leaf { .. } => 1,
        CompositeNode::Composite { children, .. } => children.iter().map(leaf_count).sum(),
        CompositeNode::Hole { .. } => 0,
    }
}

/// Two nodes are congruent iff same node kind INCLUDING the layout kind for
/// composites, same child count, and children pairwise congruent (spec §6 —
/// whose "same node kind" predates Task 2's kind/layout field split and means
/// the layout kinds). Requiring layout equality is deliberately strict:
/// real producers (`compare=`) always emit identical layouts, and for
/// hand-built mixed-layout concats the safe non-congruent outcome is the
/// documented skip, not accidental cross-layout pairing.
///
/// A hole is its own node kind for this purpose (Task 8a): a hole matches a
/// hole only — never a leaf or a composite — so a hole in one congruent-grid
/// position never pairs its "domain" (it has none) with a real leaf at the
/// matching position in a sibling row/column.
pub(crate) fn congruent(a: &CompositeNode, b: &CompositeNode) -> bool {
    match (a, b) {
        (CompositeNode::Leaf { .. }, CompositeNode::Leaf { .. }) => true,
        (CompositeNode::Hole { .. }, CompositeNode::Hole { .. }) => true,
        (
            CompositeNode::Composite { layout: la, children: ca, .. },
            CompositeNode::Composite { layout: lb, children: cb, .. },
        ) => la == lb && ca.len() == cb.len() && ca.iter().zip(cb).all(|(x, y)| congruent(x, y)),
        _ => false,
    }
}

/// True when a composite node's DIRECT children qualify for tree-path domain
/// pairing: every non-hole child is congruent to the first non-hole child.
/// [`Hole`](CompositeNode::Hole) placeholders are transparent to this
/// check — a grid mixing real cells and holes (the corner triangle of
/// `pairplot(corner=True)`, a `jointplot`'s empty 2×2 corner) still qualifies
/// its real cells for pairing, so a shared channel unions across them exactly
/// as a hole-free grid would. This is deliberately narrower than requiring
/// literal `children.iter().all(|c| congruent(&children[0], c))`: that
/// formulation treats one interspersed hole as disqualifying the WHOLE node
/// (a hole is congruent only with another hole, by design — see
/// [`congruent`]'s doc), which is correct for a hole *nested inside* a
/// composite subtree (that mismatch is still caught by `congruent`'s
/// recursive Composite-vs-Composite comparison, since neither sibling
/// subtree is itself a `Hole` at ITS parent's level) but wrong for a hole
/// that is one of THIS node's own direct cells, which has no domain to pair
/// and should simply be skipped. Vacuously `true` for an all-hole child list
/// (nothing to pair, nothing to union).
///
/// Only the POSITIONAL walk ([`resolve_channel`], this module) consults
/// this gate now: since the #74 leaf-span union, color/size sharing goes
/// through [`resolve_nonpositional`] with no congruence requirement, and
/// the figure-legend planner (`composite_render::plan_legend_walk`) mirrors
/// that effective-mode rule instead — see the lockstep contract notes on
/// [`resolve_nonpositional`] and in `composite_render.rs`.
pub(crate) fn congruent_non_hole(children: &[CompositeNode]) -> bool {
    let mut real = children.iter().filter(|c| !matches!(c, CompositeNode::Hole { .. }));
    let Some(first) = real.next() else { return true };
    real.all(|c| congruent(first, c))
}

/// Resolve one *positional* channel (`x`/`y`) for `node`, whose leaves occupy
/// global indices `leaf_span` (in pre-order). At a `Shared` node the children's
/// domains are unioned by congruent tree-path pairing (grids pair by column);
/// at an `Independent` node (or a non-congruent `Shared` node) resolution
/// recurses into each child so inner sharing still applies. Color/size take the
/// leaf-span path in [`resolve_nonpositional`] instead.
fn resolve_channel(
    node: &CompositeNode,
    leaf_span: &[usize],
    channel: Channel,
    leaves: &[LeafResolveInput<'_>],
    out: &mut [LeafScaleContext],
    warnings: &mut Vec<RenderWarning>,
) -> Result<(), CompositeResolveError> {
    let CompositeNode::Composite { layout, children, resolve, .. } = node else {
        return Ok(()); // Leaf: assignment is decided by its parent.
    };

    // Partition this node's leaf span among its children by leaf count.
    let mut subspans: Vec<&[usize]> = Vec::with_capacity(children.len());
    let mut cursor = 0;
    for child in children {
        let lc = leaf_count(child);
        subspans.push(&leaf_span[cursor..cursor + lc]);
        cursor += lc;
    }

    let shared = resolve_mode(resolve, channel) == ResolveMode::Shared;

    let congruent_children = congruent_non_hole(children);

    if shared && congruent_children {
        // Congruent share: pair leaf i of each NON-HOLE child (tree-path
        // pairing). Every non-hole child has the same leaf count
        // (`congruent_non_hole` guarantees it), so `leaves_per_child` is
        // read off the first one; a `Hole` child's subspan is empty
        // (`leaf_count` 0) and is filtered out of each position's group
        // below rather than indexed, so a grid mixing real cells and holes
        // (pairplot's corner triangle, jointplot's empty corner) still
        // unions its real cells' domains.
        let leaves_per_child = children
            .iter()
            .find(|c| !matches!(c, CompositeNode::Hole { .. }))
            .map(leaf_count)
            .unwrap_or(0);
        for pos in 0..leaves_per_child {
            let group: Vec<usize> = subspans
                .iter()
                .filter(|s| pos < s.len())
                .map(|s| s[pos])
                .collect();
            if *layout == CompositeLayout::Overlay {
                let axis = overlay_axis(&group, channel, leaves);
                refuse_unplaceable_overlay_members(&group, channel, &axis, leaves)?;
                warn_members_off_caller_scale(
                    &axis,
                    channel.as_str(),
                    ExcludedLayerPlacement::OwnScale,
                    warnings,
                );
                if let Some((_, scale)) = establishing_caller_scale(&axis) {
                    resolve_caller_scale_group(&group, channel, scale, leaves, out)?;
                    continue;
                }
            }
            resolve_group(&group, channel, leaves, out)?;
        }
        Ok(())
    } else {
        // Independent node, or a non-congruent Shared node (skip sharing for the
        // group): fall back to per-child resolution so inner Shared nodes work.
        for (child, span) in children.iter().zip(&subspans) {
            resolve_channel(child, span, channel, leaves, out, warnings)?;
        }
        Ok(())
    }
}

/// Resolve one *non-positional* channel (`color`/`size`) for `node` (GH #74).
///
/// Unlike [`resolve_channel`]'s positional tree-path pairing, a shared color/
/// size scale unions across the node's **entire leaf span**, not just leaves at
/// matching grid positions — a single legend/scale must cover every panel,
/// however the panels nest. Effective-mode inheritance (spec §6): a node's
/// effective mode is its explicit [`channel_resolve`] setting if any, else the
/// inherited ancestor mode; `inherited` threads that down. A node is the
/// *outermost* effective-shared node exactly when `eff == Shared` while
/// `inherited != Shared`, and only there do we union — a deeper shared node
/// (unset child of a shared parent) is already covered, and an explicitly
/// `Independent` child resets the chain so its own subtree can re-share.
///
/// This is the resolve-side half of the bit-for-bit contract with
/// [`super::composite_render::plan_legend_walk`]: the figure legend band
/// attaches at exactly this outermost effective-shared node. Both walks compute
/// the gate through the shared [`effective_share`] helper, so the lockstep is
/// compile-time, not a mirrored predicate.
fn resolve_nonpositional(
    node: &CompositeNode,
    leaf_span: &[usize],
    channel: Channel,
    inherited: ResolveMode,
    leaves: &[LeafResolveInput<'_>],
    out: &mut [LeafScaleContext],
) -> Result<(), CompositeResolveError> {
    let CompositeNode::Composite { children, resolve, .. } = node else {
        return Ok(()); // Leaf/Hole: assignment is decided by an ancestor.
    };
    let (eff, is_outermost) = effective_share(channel_resolve(resolve, channel), inherited);

    if is_outermost {
        // Outermost effective-shared node: union every descendant leaf whose
        // effective mode stays shared (excluding explicit-independent
        // subtrees, which `collect_shared_leaves` skips and the recursion
        // below resolves on their own).
        let mut group: Vec<usize> = Vec::new();
        collect_shared_leaves(node, leaf_span, channel, inherited, &mut group);
        resolve_group(&group, channel, leaves, out)?;
    }

    // Always recurse: descendants under an explicit-independent boundary reset
    // `inherited` and may re-share among themselves; descendants already
    // covered by this node's union recurse harmlessly (their `eff == inherited
    // == Shared`, so no second union fires).
    let mut cursor = 0;
    for child in children {
        let lc = leaf_count(child);
        resolve_nonpositional(child, &leaf_span[cursor..cursor + lc], channel, eff, leaves, out)?;
        cursor += lc;
    }
    Ok(())
}

/// Collect the global leaf indices belonging to the shared union owned by the
/// outermost effective-shared node this is first called on. Descends only
/// through the *contiguous* shared region: the moment a descendant composite's
/// effective mode drops to `Independent` (an explicit opt-out — spec §6), that
/// whole branch is skipped — including any node that re-shares beneath it,
/// since a re-shared node under an independent boundary is a *separate* union
/// owner ([`resolve_nonpositional`]'s recursion resolves it on its own). Holes
/// contribute nothing.
fn collect_shared_leaves(
    node: &CompositeNode,
    leaf_span: &[usize],
    channel: Channel,
    inherited: ResolveMode,
    acc: &mut Vec<usize>,
) {
    match node {
        CompositeNode::Leaf { .. } => {
            if inherited == ResolveMode::Shared {
                // A leaf occupies exactly one global index in its span.
                acc.extend_from_slice(leaf_span);
            }
        }
        CompositeNode::Hole { .. } => {}
        CompositeNode::Composite { children, resolve, .. } => {
            let eff = channel_resolve(resolve, channel).unwrap_or(inherited);
            if eff != ResolveMode::Shared {
                // Independent boundary: nothing below belongs to this union
                // (a deeper re-share owns its own leaves).
                return;
            }
            let mut cursor = 0;
            for child in children {
                let lc = leaf_count(child);
                collect_shared_leaves(child, &leaf_span[cursor..cursor + lc], channel, eff, acc);
                cursor += lc;
            }
        }
    }
}

/// Union one channel's domain across a group of paired leaves and assign it to
/// each participating leaf. Non-participants (no channel encoding, explicit user
/// scale, or faceted-independent on this channel) are excluded from both the
/// union and the assignment, preserving their standalone resolution.
fn resolve_group(
    group: &[usize],
    channel: Channel,
    leaves: &[LeafResolveInput<'_>],
    out: &mut [LeafScaleContext],
) -> Result<(), CompositeResolveError> {
    let mut domains: Vec<(usize, SharedDomain)> = Vec::with_capacity(group.len());
    for &idx in group {
        if let Some(d) = leaf_channel_domain(&leaves[idx], channel)? {
            domains.push((idx, d));
        }
    }
    assign_union(&domains, channel, out);
    Ok(())
}

/// Union `domains` and assign the result to each leaf that contributed one.
/// A group mixing numeric and ordinal domains, or contributing none, assigns
/// nothing.
fn assign_union(domains: &[(usize, SharedDomain)], channel: Channel, out: &mut [LeafScaleContext]) {
    if domains.is_empty() {
        return;
    }

    let all_numeric = domains
        .iter()
        .all(|(_, d)| matches!(d, SharedDomain::Numeric { .. }));
    let all_ordinal = domains
        .iter()
        .all(|(_, d)| matches!(d, SharedDomain::Ordinal(_)));

    let unified = if all_numeric {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (_, d) in domains {
            if let SharedDomain::Numeric { lo: l, hi: h } = d {
                lo = lo.min(*l);
                hi = hi.max(*h);
            }
        }
        SharedDomain::Numeric { lo, hi }
    } else if all_ordinal {
        let mut cats: Vec<String> = Vec::new();
        for (_, d) in domains {
            if let SharedDomain::Ordinal(cs) = d {
                for c in cs {
                    if !cats.contains(c) {
                        cats.push(c.clone());
                    }
                }
            }
        }
        SharedDomain::Ordinal(cats)
    } else {
        // A shared group mixing a Numeric and an Ordinal domain (e.g. one
        // sibling's shared channel is quantitative, another's is
        // categorical) has no meaningful union. Spec §6's "Resolve
        // semantics" only spells out the skip for a *non-congruent* tree
        // shape ("non-congruent → skip that channel for the group"), but the
        // same skip-not-error philosophy applies here for a Hconcat/Vconcat/
        // Grid/Wrap group: there is no semantically right domain for
        // mismatched types any more than there is for mismatched tree
        // shapes, and erroring would reject a hand-built heterogeneous
        // concat that renders fine standalone — each sibling keeps its own
        // physical panel either way.
        //
        // An `Overlay` group is different: every member draws
        // onto the ONE panel the group shares, so a member that cannot serve
        // the axis type the group establishes has nowhere else to draw. That
        // refusal is asked by [`refuse_unplaceable_overlay_members`], from
        // `resolve_channel` — positional-only and layout-aware, unlike this
        // function, which stays generic over every layout and both the
        // positional and non-positional channels it serves.
        //
        // Skip sharing for this group; each participating leaf falls back to
        // its own standalone resolution (rendering is otherwise normal).
        // Reachable only via a hand-built tree with heterogeneous leaf dtypes
        // on the same shared channel — `compare=` producers always emit
        // uniform dtypes, so this is a defensive fallback, not a documented
        // user-facing feature.
        return;
    };

    for (idx, _) in domains {
        out[*idx].set(channel, unified.clone());
    }
}

/// The members of an `Overlay` group's axis on `channel`: each member, by
/// its position in the group, with its own scale there, in member order.
fn overlay_axis<'g, 'a>(
    group: &'g [usize],
    channel: Channel,
    leaves: &'g [LeafResolveInput<'a>],
) -> AxisMembers<impl Iterator<Item = AxisMember<'a>> + Clone + 'g> {
    let members = group
        .iter()
        .enumerate()
        .map(move |(index, &idx)| AxisMember { index, own_scale: leaf_own_scale(&leaves[idx], channel) });
    AxisMembers::new(members, |position| {
        let lead = &leaves[group[position]];
        leaf_post_flip_encoding(lead, channel)
            .is_some_and(|enc| {
                let located = locate_seed_field(&enc.field, lead.spec, lead.final_batch, lead.transform_outputs);
                column_reads_categorical(enc, located)
            })
    })
}

/// A leaf's own encoding for the positional `channel`, or `None` for a
/// non-positional channel. A leaf's encoding is already in its panel's
/// post-flip space, so no flip is applied, unlike
/// `scale_resolve::domain::layer_channel_encoding`, which applies a flat
/// layer's own coord flip.
fn leaf_post_flip_encoding<'a>(
    leaf: &LeafResolveInput<'a>,
    channel: Channel,
) -> Option<&'a crate::spec::encoding::EncodingSpec> {
    match channel {
        Channel::X => leaf.encoding.x.as_ref(),
        Channel::Y => leaf.encoding.y.as_ref(),
        Channel::Color | Channel::Size => None,
    }
}

/// A leaf's own scale for the positional `channel`, if its encoding binds one.
fn leaf_own_scale<'a>(
    leaf: &LeafResolveInput<'a>,
    channel: Channel,
) -> Option<&'a crate::spec::encoding::ScaleSpec> {
    leaf_post_flip_encoding(leaf, channel).and_then(|e| e.scale.as_ref())
}

/// Resolve an `Overlay` group whose axis is the caller's `scale`, the one
/// [`establishing_caller_scale`] names as establishing it over the group's
/// [`overlay_axis`]. The members are the leaves [`LayerUnion::CarryingScale`]
/// admits, as the flat layered route admits layers: those carrying this
/// same scale and those declaring none.
/// Every member resolves `scale`, so the panel's axis (drawn by the first
/// leaf, itself a member whenever it declares no scale) and every member's
/// marks describe one scale. When `scale` states no `domain`, the members'
/// data unions into the one extent they all resolve it over; a stated
/// `domain` is the extent as written. A leaf carrying a different scale, or
/// its own `domain`, stays out and resolves on its own.
fn resolve_caller_scale_group(
    group: &[usize],
    channel: Channel,
    scale: &crate::spec::encoding::ScaleSpec,
    leaves: &[LeafResolveInput<'_>],
    out: &mut [LeafScaleContext],
) -> Result<(), CompositeResolveError> {
    let selection = LayerUnion::CarryingScale(scale);
    let members: Vec<usize> = if scale.positional_extent().is_some() {
        group.iter().copied().filter(|&idx| selection.admits(leaf_own_scale(&leaves[idx], channel))).collect()
    } else {
        let mut domains: Vec<(usize, SharedDomain)> = Vec::with_capacity(group.len());
        for &idx in group {
            if let Some(d) = leaf_positional_domain(&leaves[idx], channel, selection)? {
                domains.push((idx, d));
            }
        }
        assign_union(&domains, channel, out);
        domains.into_iter().map(|(idx, _)| idx).collect()
    };
    for idx in members {
        out[idx].set_scale(channel, scale.clone());
    }
    Ok(())
}

/// The Overlay-only half of the shared-axis refusal: a member of an
/// `Overlay` group whose own column cannot serve the axis the group
/// establishes (through the caller scale [`establishing_caller_scale`] names over `axis`,
/// the group's [`overlay_axis`], else through the first member's own domain)
/// has nowhere else to draw, since every member of an `Overlay` shares one
/// panel. Called from [`resolve_channel`] (the one
/// caller that knows the node's `layout`, and resolves positional channels
/// only) immediately before [`resolve_group`], so `resolve_group` itself
/// stays generic over every layout and both callers it serves — a
/// `resolve_nonpositional` (color/size) call never reaches this question at
/// all, rather than carrying an unused parameter through the shared
/// function.
///
/// When no caller scale establishes the axis, the panel's own axis type is
/// decided by the group's first member; its flip state always is. The same
/// "Layer 0 is exempt... the panel's axes
/// are resolved from its own post-flip encoding" precedent
/// [`crate::render::LayerCoordUnplaceableReason::ExchangedChannelUnservable`]
/// already states. A member whose flip state matches the panel's is judged
/// only against a continuous axis: a mixed group whose first member is
/// categorical is not this defect at all (a categorical panel already
/// accepts a numeric sibling's column as a category key,
/// `ensure_category_keyable`).
///
/// A member whose flip state differs from the panel's draws its exchanged
/// column on the panel's axis, of either kind, and is refused when that axis
/// cannot draw it, by
/// [`crate::render::arrow_cast::require_exchanged_column_servable`], the
/// question the merged-flat routes' coord-exchange gate asks. The refusal
/// names the member's own coord, or `fm.CoordCartesian` for a member that
/// declares none beside a flipped first member: `+` stamps that coord on
/// such a member, so the merged-flat routes name it too. So `+`, `fm.layer`
/// and either output kind refuse a member whose flip state differs from the
/// first member's with the same reason.
///
/// Checked directly against each member's own encoding and column, not
/// against a `SharedDomain` [`resolve_group`] would separately compute: a
/// member whose `SharedDomain` is `Ordinal` because of an explicit
/// categorical type annotation over an otherwise-numeric dtype is not this
/// defect either (that column COULD serve a continuous axis, so the
/// group's mismatch is a rendering choice, not a malformed input) —
/// [`crate::render::arrow_cast::require_placeable_on_continuous_axis`]
/// draws exactly that distinction, the same predicate a single leaf's own
/// internal `spec.layers` union reads.
///
/// Every member of the group is judged, not only the ones
/// [`LayerUnion::admits`] lets into the axis's extent: the two ask different
/// questions. A member carrying a different scale, or its own `domain`,
/// stays out of the extent but still draws in the one shared panel, so its
/// column must sit on that axis all the same. The flat layered route's gate
/// (`scale_resolve::require_layers_placeable_on_caller_scale`) judges every
/// layer the same way.
///
/// At most `group[0]`'s own domain is resolved here (`resolve_group`, called
/// right after this returns, resolves every member's on its own): reading
/// a domain this function never uses would let a LATER member's unrelated
/// resolution failure pre-empt an EARLIER member's typed refusal below,
/// whichever came first in `leaf_channel_domain`'s own iteration rather
/// than in the caller's own member order.
fn refuse_unplaceable_overlay_members<'a>(
    group: &[usize],
    channel: Channel,
    axis: &AxisMembers<impl Iterator<Item = AxisMember<'a>> + Clone>,
    leaves: &[LeafResolveInput<'_>],
) -> Result<(), CompositeResolveError> {
    // The member whose caller scale establishes the continuous axis is the
    // one the others are judged against, whatever its position or its
    // column's own inferred type. Otherwise the first member's own domain
    // decides.
    use crate::render::arrow_cast::{require_exchanged_column_servable, PanelAxisKind};
    let (establishing, axis_kind) = match establishing_caller_scale(axis) {
        Some((position, _)) => (position, PanelAxisKind::Continuous),
        None => match leaf_channel_domain(&leaves[group[0]], channel)? {
            Some(SharedDomain::Numeric { .. }) => (0, PanelAxisKind::Continuous),
            Some(SharedDomain::Ordinal(_)) => (0, PanelAxisKind::Categorical),
            None => return Ok(()),
        },
    };
    let panel_flipped = crate::spec::coord::flips_positional_channels(
        leaves[group[0]].spec.coord.as_ref(),
    );
    let inherited_cartesian = crate::spec::coord::CoordKind::default();
    for (position, &idx) in group.iter().enumerate() {
        if position == establishing {
            continue; // the establishing member; trivially placeable
        }
        let leaf = &leaves[idx];
        let member_flipped =
            crate::spec::coord::flips_positional_channels(leaf.spec.coord.as_ref());
        // The coord that exchanges the member's channels relative to the
        // panel: its own, else the Cartesian a coord-less member keeps beside
        // a flipped first member, the coord `+` stamps on it and the
        // merged-flat routes name.
        let exchanged_by = (member_flipped != panel_flipped)
            .then(|| leaf.spec.coord.as_ref().unwrap_or(&inherited_cartesian));
        if exchanged_by.is_none() && axis_kind == PanelAxisKind::Categorical {
            continue;
        }
        // `resolve_channel`, the only caller, resolves positional channels
        // only, so a non-positional channel never reaches here; it would
        // judge no member rather than panic if that guard is ever loosened.
        let Some(enc) = leaf_post_flip_encoding(leaf, channel) else { continue };
        let Some(located) = locate_seed_field(&enc.field, leaf.spec, leaf.final_batch, leaf.transform_outputs)
        else {
            continue;
        };
        // `coord_flipped: false`, unconditionally: `channel` here already
        // names the panel's own resolved axis, the same deliberate choice
        // [`crate::render::arrow_cast::require_placeable_on_continuous_axis`]'s
        // own text makes (it is what a caller writes into `resolve=`, a
        // property of the shared panel, not of this member's own authored
        // encoding) — un-flipping only the OUTER wrapper's copy of the same
        // token would make one message disagree with itself about which
        // axis it means.
        let to_resolve_error = |source| CompositeResolveError::DomainResolve {
            channel: channel.as_str(),
            field: enc.field.clone(),
            source,
            coord_flipped: false,
        };
        if let Some(coord) = exchanged_by {
            require_exchanged_column_servable(
                position,
                coord,
                channel.as_str(),
                &enc.field,
                located.col.data_type(),
                axis_kind,
            )
            .map_err(to_resolve_error)?;
            continue;
        }
        crate::render::arrow_cast::require_placeable_on_continuous_axis(
            position,
            channel.as_str(),
            &enc.field,
            located.col.data_type(),
        )
        .map_err(to_resolve_error)?;
    }
    Ok(())
}

/// Compute one leaf's standalone domain for `channel`, or `None` if the leaf does
/// not participate in composite sharing on that channel. Dispatches to the
/// positional or non-positional resolver.
fn leaf_channel_domain(
    leaf: &LeafResolveInput<'_>,
    channel: Channel,
) -> Result<Option<SharedDomain>, CompositeResolveError> {
    match channel {
        Channel::X | Channel::Y => leaf_positional_domain(leaf, channel, LayerUnion::Every),
        Channel::Color => leaf_color_domain(leaf),
        Channel::Size => leaf_size_domain(leaf),
    }
}

/// One leaf's standalone positional (`x`/`y`) domain, or `None` if it does not
/// participate in sharing on that channel.
///
/// `selection` is the rule the leaf is admitted by. [`LayerUnion::Every`] is
/// the auto path's sharing; [`LayerUnion::CarryingScale`] is an `Overlay`
/// group under a caller's scale (see [`resolve_caller_scale_group`]), which
/// admits the leaf by its own scale, reads its data as a continuous extent
/// with its own layers selected the same way, and leaves the zero anchor to
/// that scale.
///
/// Non-participation (returns `None`):
/// - the channel has no encoding on this leaf;
/// - under `Every`, the channel carries an explicit user `enc.scale` (user
///   scale wins, spec §6) — a scale ferrum itself injected is NOT one of these
///   and DOES participate, carrying its zero anchor into its own contribution
///   (see `ScaleSpec::Linear`'s `system_injected`); under
///   `CarryingScale`, the selection does not admit the leaf's own scale;
/// - the leaf is faceted with `Independent` resolution on this channel (its
///   internal per-panel resolution is preserved — pdp `compare=` independent-x).
///
/// Otherwise the domain is unioned through the facet mechanism with
/// `include_final = false`, passing the leaf's full final batch as the primary:
/// for a faceted-shared leaf that batch already spans every panel (its outer
/// extent), and box/strip whisker fields are picked up via the layer union
/// inside [`numeric_domain_union`] (spec §6 D3 — shared domains derive from
/// transform-output batches).
fn leaf_positional_domain(
    leaf: &LeafResolveInput<'_>,
    channel: Channel,
    selection: LayerUnion<'_>,
) -> Result<Option<SharedDomain>, CompositeResolveError> {
    // R3: `leaf.spec` is the leaf's ORIGINAL ChartSpec (`coord` untouched by
    // `prepare::build_layers`'s encoding swap), while `leaf.encoding` is
    // already the RESOLVED (post-flip) rendering encoding (see
    // `LeafResolveInput`'s doc) — so `leaf.spec.coord` still tells us whether
    // this leaf is flipped, for un-flipping `channel` in any error below.
    let coord_flipped = matches!(leaf.spec.coord, Some(crate::spec::coord::CoordKind::Flip));
    let enc = match channel {
        Channel::X => leaf.encoding.x.as_ref(),
        Channel::Y => leaf.encoding.y.as_ref(),
        _ => unreachable!("leaf_positional_domain called with non-positional channel"),
    };
    let Some(enc) = enc else { return Ok(None) };

    // Explicit user scale bypasses sharing (must remain distinguishable so it
    // still short-circuits at the existing explicit-scale path — D4b).
    //
    // #247A: "user scale" is the whole point of the rule, and until this field
    // existed the question could only be asked as `is_some()`. A scale ferrum's
    // own spec build injected is not the user's, so it does NOT buy the leaf out
    // of sharing — see `ScaleSpec::Linear`'s `system_injected` doc for what
    // produces one and why the alternative (deleting the injection) was
    // rejected. The anchor it asked for travels into the union below instead of
    // being lost with the bypass it no longer takes.
    let injected_zero = match (selection, enc.scale.as_ref()) {
        (LayerUnion::CarryingScale(_), own) if !selection.admits(own) => return Ok(None),
        (LayerUnion::CarryingScale(_), _) => false,
        (LayerUnion::Every, Some(s)) if !s.is_system_injected() => return Ok(None),
        (LayerUnion::Every, Some(s)) => s.anchors_zero(),
        (LayerUnion::Every, None) => false,
    };

    // Faceted child keeps its internal facet resolution: an outer node shares
    // only the child's outer scales. A channel the leaf resolves `Independent`
    // internally has no single outer scale, so it does not participate.
    if let Some(fspec) = leaf.spec.facet.as_ref() {
        let mode = match channel {
            Channel::X => fspec.resolve.x,
            Channel::Y => fspec.resolve.y,
            _ => unreachable!(),
        };
        if mode == ResolveMode::Independent {
            return Ok(None);
        }
    }

    let located = locate_seed_field(&enc.field, leaf.spec, leaf.final_batch, leaf.transform_outputs)
        .ok_or_else(|| CompositeResolveError::UnknownField {
            channel: channel.as_str(),
            field: enc.field.clone(),
            coord_flipped,
        })?;
    let dtype = match selection {
        LayerUnion::CarryingScale(_) => SpecDataType::Quantitative,
        LayerUnion::Every => infer_spec_type(enc, located.col.data_type()),
    };

    let paired = match channel {
        Channel::X => leaf.encoding.x2.as_ref(),
        Channel::Y => leaf.encoding.y2.as_ref(),
        _ => None,
    }
    .map(|e| e.field.as_str());

    match dtype {
        SpecDataType::Quantitative | SpecDataType::Temporal => {
            let (lo, hi) = numeric_domain_union_over(
                channel.as_str(),
                &enc.field,
                paired,
                leaf.final_batch,
                leaf.transform_outputs,
                leaf.spec,
                false,
                selection,
            )
            .map_err(|source| CompositeResolveError::DomainResolve {
                channel: channel.as_str(),
                field: enc.field.clone(),
                source,
                coord_flipped,
            })?;
            // #247A: the leaf kept its injected scale's zero anchor by taking
            // the explicit bypass; now that it shares instead, the anchor has to
            // travel with its CONTRIBUTION or it would be silently dropped
            // (`fm.layer(bar(1..3), point(40..80))` would union to `1..80`, not
            // the `0..80` a bar promises). Applied to this leaf's extent, not to
            // the group union, so a sibling leaf never inherits an anchor its own
            // mark never asked for.
            //
            // Dtype-blind, like the auto path's own layered-bar anchor
            // (`domain::layer_wants_bar_zero_anchor`, which is reached from both
            // the Quantitative and Temporal arms of `build_axis_scale` and asks
            // nothing about dtype): one rule for "a bar's value axis reaches
            // zero", not one per dtype. A temporal value axis on a bar is a
            // latent shape either way — Python injects `type: "linear"` without
            // consulting the `:T` annotation — and the two mechanisms agreeing
            // matters more than special-casing it here.
            let (lo, hi) = if injected_zero { extend_extent_to_zero(lo, hi) } else { (lo, hi) };
            Ok(Some(SharedDomain::Numeric { lo, hi }))
        }
        SpecDataType::Ordinal | SpecDataType::Nominal => {
            // `#247`B's null-band occupancy rule needs the OPPOSITE positional
            // channel (the second coordinate a mark cannot be placed without),
            // which is a different pairing from the `paired` x2/y2 span partner
            // the numeric arm above unions over.
            let paired_positional = match channel {
                Channel::X => leaf.encoding.y.as_ref(),
                Channel::Y => leaf.encoding.x.as_ref(),
                _ => None,
            }
            .map(|e| e.field.as_str());
            // `leaf.spec` for the same reason the numeric arm above passes it: a
            // leaf can itself be a flat-merged layered chart (`hconcat(a + b, c)`),
            // and its own layers' categories belong in the domain it contributes
            // (task 12c). Routing through the same union as the flat path also
            // means a leaf whose ordinal field lives only in a named output now
            // locates it, where passing `final_batch` straight through used to
            // fail the leaf's whole domain resolve.
            let cats = positional_category_union(
                channel.as_str(),
                &enc.field,
                paired_positional,
                leaf.final_batch,
                leaf.transform_outputs,
                leaf.spec,
                false,
            )
            .map_err(|source| CompositeResolveError::DomainResolve {
                channel: channel.as_str(),
                field: enc.field.clone(),
                source,
                coord_flipped,
            })?;
            Ok(Some(SharedDomain::Ordinal(cats)))
        }
    }
}

/// One leaf's standalone color domain, or `None` if it does not participate.
///
/// Mirrors [`crate::render::scale_resolve::build_color_scale`]'s continuous-vs-
/// categorical determination so the resolved [`SharedDomain`] variant always
/// matches the scale the leaf render will build:
/// - continuous (`Quantitative`/`Temporal`, unless the mark forces categorical)
///   → [`SharedDomain::Numeric`] extent of the color column;
/// - categorical → [`SharedDomain::Ordinal`] first-appearance category vector.
///
/// Returns `None` when color is unencoded or carries an explicit user
/// `enc.scale` (user scale wins — the leaf render short-circuits on it, so the
/// resolve pass must exclude it to stay consistent). Color has no independent
/// facet option (`FacetResolve` only covers x/y), so faceted leaves participate.
fn leaf_color_domain(
    leaf: &LeafResolveInput<'_>,
) -> Result<Option<SharedDomain>, CompositeResolveError> {
    let Some(enc) = leaf.encoding.color.as_ref() else { return Ok(None) };
    if enc.scale.is_some() {
        return Ok(None);
    }

    let located = locate_field(&enc.field, leaf.final_batch, leaf.transform_outputs)
        .ok_or_else(|| CompositeResolveError::UnknownField {
            channel: Channel::Color.as_str(),
            field: enc.field.clone(),
            // "color" is never positional, so `CoordFlip` never touches it —
            // `user_facing_channel` is identity regardless (R3).
            coord_flipped: false,
        })?;

    // FA-5: area marks always group color discretely — treat any dtype as
    // categorical, exactly as `build_color_scale`'s `force_categorical` does, so
    // the shared domain lands on the categorical scale the area leaf builds.
    let force_categorical = matches!(leaf.spec.mark, crate::spec::mark::Mark::Area);
    let dtype = infer_spec_type(enc, located.col.data_type());
    let is_continuous = !force_categorical
        && matches!(dtype, SpecDataType::Quantitative | SpecDataType::Temporal)
        && crate::render::arrow_cast::is_numeric(located.col.data_type());

    if is_continuous {
        Ok(Some(numeric_extent_domain(located.col)))
    } else {
        let cats = distinct_values_in_order(located.batch, &enc.field).map_err(|source| {
            CompositeResolveError::DomainResolve {
                channel: Channel::Color.as_str(),
                field: enc.field.clone(),
                source,
                coord_flipped: false, // "color" is never positional — see above.
            }
        })?;
        Ok(Some(SharedDomain::Ordinal(cats)))
    }
}

/// One leaf's standalone size domain, or `None` if size is unencoded.
///
/// Size is always a numeric extent (`build_size_scale` derives `[min, max]` from
/// the column). An explicit user size `scale` only pins the pixel *range*, never
/// the data domain, so it does not exclude the leaf from domain sharing.
fn leaf_size_domain(
    leaf: &LeafResolveInput<'_>,
) -> Result<Option<SharedDomain>, CompositeResolveError> {
    let Some(enc) = leaf.encoding.size.as_ref() else { return Ok(None) };
    let located = locate_field(&enc.field, leaf.final_batch, leaf.transform_outputs)
        .ok_or_else(|| CompositeResolveError::UnknownField {
            channel: Channel::Size.as_str(),
            field: enc.field.clone(),
            // "size" is never positional — see `leaf_color_domain`'s note above.
            coord_flipped: false,
        })?;
    Ok(Some(numeric_extent_domain(located.col)))
}

/// A [`SharedDomain::Numeric`] over a located Arrow column's finite extent.
fn numeric_extent_domain(col: &dyn arrow::array::Array) -> SharedDomain {
    let (lo, hi) = numeric_extent(col);
    SharedDomain::Numeric { lo, hi }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::composite::CompositeResolve;
    use crate::spec::encoding::EncodingSpec;
    use crate::spec::layer::Layer;
    use crate::spec::mark::Mark;
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    use crate::transform::core::FINAL_OUTPUT_KEY;

    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    // -- builders -------------------------------------------------------------

    fn enc(field: &str) -> EncodingSpec {
        EncodingSpec { field: field.to_string(), ..Default::default() }
    }

    fn base_spec() -> ChartSpec {
        ChartSpec {
            mark: Mark::Point,
            encoding: Encoding::default(),
            ..Default::default()
        }
    }

    fn spec_with(encoding: Encoding) -> ChartSpec {
        ChartSpec { encoding, ..base_spec() }
    }

    /// A one-numeric-column batch on `field`.
    fn num_batch(field: &str, values: &[f64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(field, DataType::Float64, true)]));
        let arr = Float64Array::from(values.to_vec());
        RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
    }

    /// A one-string-column batch on `field`.
    fn cat_batch(field: &str, values: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(field, DataType::Utf8, true)]));
        let arr = StringArray::from(values.iter().map(|s| Some(*s)).collect::<Vec<_>>());
        RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
    }

    fn outputs(batch: &RecordBatch) -> HashMap<String, RecordBatch> {
        let mut m = HashMap::new();
        m.insert(FINAL_OUTPUT_KEY.to_string(), batch.clone());
        m
    }

    fn leaf_node(field_x: Option<&str>, field_y: Option<&str>) -> CompositeNode {
        let mut e = Encoding::default();
        if let Some(fx) = field_x {
            e.x = Some(enc(fx));
        }
        if let Some(fy) = field_y {
            e.y = Some(enc(fy));
        }
        CompositeNode::Leaf { spec: Box::new(spec_with(e)), data: 0, label: None }
    }

    fn composite(
        layout: CompositeLayout,
        children: Vec<CompositeNode>,
        resolve: CompositeResolve,
    ) -> CompositeNode {
        CompositeNode::Composite {
            layout,
            children,
            label: None,
            resolve,
            spacing: None,
            row_ratios: None,
            col_ratios: None,
            ncols: None,
            nrows: None,
            extent_alignments: Vec::new(),
            title: None,
            subtitle: None,
            caption: None,
            config: None,
        }
    }

    fn shared_x() -> CompositeResolve {
        CompositeResolve { x: ResolveMode::Shared, ..CompositeResolve::default() }
    }
    fn shared_both() -> CompositeResolve {
        CompositeResolve {
            x: ResolveMode::Shared,
            y: ResolveMode::Shared,
            ..CompositeResolve::default()
        }
    }
    fn shared_color() -> CompositeResolve {
        CompositeResolve { color: Some(ResolveMode::Shared), ..CompositeResolve::default() }
    }
    fn shared_size() -> CompositeResolve {
        CompositeResolve { size: Some(ResolveMode::Shared), ..CompositeResolve::default() }
    }

    /// Hold owned per-leaf pieces so borrows in `LeafResolveInput` stay valid.
    struct LeafOwned {
        spec: ChartSpec,
        encoding: Encoding,
        batch: RecordBatch,
        outputs: HashMap<String, RecordBatch>,
    }

    fn owned(spec: ChartSpec, batch: RecordBatch) -> LeafOwned {
        let encoding = spec.encoding.clone();
        let outputs = outputs(&batch);
        LeafOwned { spec, encoding, batch, outputs }
    }

    fn input(o: &LeafOwned) -> LeafResolveInput<'_> {
        LeafResolveInput {
            spec: &o.spec,
            encoding: &o.encoding,
            final_batch: &o.batch,
            transform_outputs: &o.outputs,
        }
    }

    // -- congruence -----------------------------------------------------------

    #[test]
    fn congruence_requires_matching_layout_kind() {
        let a = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("x"), None), leaf_node(Some("x"), None)],
            CompositeResolve::default(),
        );
        // Same structure, SAME layout → congruent.
        let a2 = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("x"), None), leaf_node(Some("x"), None)],
            CompositeResolve::default(),
        );
        assert!(congruent(&a, &a2));
        // Same structure, different layout kind → NOT congruent (spec §6's
        // "same node kind" means the layout kinds; skip is the safe outcome).
        let b = composite(
            CompositeLayout::Vconcat,
            vec![leaf_node(Some("x"), None), leaf_node(Some("x"), None)],
            CompositeResolve::default(),
        );
        assert!(!congruent(&a, &b));

        // Different child count → not congruent.
        let c = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("x"), None)],
            CompositeResolve::default(),
        );
        assert!(!congruent(&a, &c));

        // Leaf vs composite → not congruent.
        assert!(!congruent(&leaf_node(Some("x"), None), &a));
    }

    #[test]
    fn congruence_hole_matches_hole_only() {
        // A hole is its own node kind (Task 8a): hole-vs-hole is congruent,
        // but hole-vs-leaf and hole-vs-composite are not.
        let hole = || CompositeNode::Hole { width: None, height: None };
        assert!(congruent(&hole(), &hole()));
        assert!(!congruent(&hole(), &leaf_node(Some("x"), None)));
        assert!(!congruent(&leaf_node(Some("x"), None), &hole()));
        let comp = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("x"), None), leaf_node(Some("x"), None)],
            CompositeResolve::default(),
        );
        assert!(!congruent(&hole(), &comp));
        assert!(!congruent(&comp, &hole()));
    }

    /// Attach a per-child label to a node without altering its structure — the
    /// Task 5d field congruence must ignore. A no-op on `Hole` (it has no
    /// label field at all).
    fn labeled(mut node: CompositeNode, text: &str) -> CompositeNode {
        match &mut node {
            CompositeNode::Leaf { label, .. } | CompositeNode::Composite { label, .. } => {
                *label = Some(text.to_string());
            }
            CompositeNode::Hole { .. } => {}
        }
        node
    }

    #[test]
    fn congruence_ignores_per_child_labels() {
        // Per-child labels are decoration, never a structural discriminator, so
        // two otherwise-identical children with DIFFERENT labels must still be
        // congruent — a label must NOT re-trigger the non-congruent sharing skip
        // (Task 5d: labels must not affect `congruent`).
        let base = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("x"), None), leaf_node(Some("x"), None)],
            CompositeResolve::default(),
        );
        let a = labeled(base.clone(), "Model A");
        let b = labeled(base.clone(), "Model B");
        assert!(congruent(&a, &b), "differently-labeled congruent children must still pair");
        assert!(congruent(&base, &a), "labeled vs unlabeled must be congruent");
        // Labeled leaf vs unlabeled leaf are congruent too.
        assert!(congruent(
            &labeled(leaf_node(Some("x"), None), "L"),
            &leaf_node(Some("x"), None),
        ));
    }

    #[test]
    fn shared_resolve_still_unions_labeled_congruent_children() {
        // The #45 residuals shape: labeled compare children sharing an axis must
        // still union their domains. The labels (attached to each congruent
        // child) must not divert resolution to the per-child independent skip.
        let l0 = owned(spec_with_x("v"), num_batch("v", &[0.0, 10.0]));
        let l1 = owned(spec_with_x("v"), num_batch("v", &[5.0, 30.0]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![
                labeled(leaf_node(Some("v"), None), "Model A"),
                labeled(leaf_node(Some("v"), None), "Model B"),
            ],
            shared_x(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        let expect = Some(SharedDomain::Numeric { lo: 0.0, hi: 30.0 });
        assert_eq!(ctx[0].x, expect, "labeled child 0 must still receive the shared union");
        assert_eq!(ctx[1].x, expect, "labeled child 1 must still receive the shared union");
    }

    // -- flat children --------------------------------------------------------

    #[test]
    fn flat_children_shared_x_unions_numeric() {
        let l0 = owned(spec_with_x("v"), num_batch("v", &[0.0, 10.0]));
        let l1 = owned(spec_with_x("v"), num_batch("v", &[5.0, 30.0]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
            shared_x(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        assert_eq!(ctx.len(), 2);
        let expect = Some(SharedDomain::Numeric { lo: 0.0, hi: 30.0 });
        assert_eq!(ctx[0].x, expect);
        assert_eq!(ctx[1].x, expect);
        // y not encoded → untouched.
        assert_eq!(ctx[0].y, None);
    }

    #[test]
    fn mixed_numeric_ordinal_group_skips_sharing() {
        // One leaf's x is numeric, the sibling's is categorical: no meaningful
        // union exists, so the group's sharing is skipped (documented) and
        // both leaves keep independent (None) contexts — not an error.
        let l0 = owned(spec_with_x("v"), num_batch("v", &[0.0, 10.0]));
        let l1 = owned(spec_with_x("v"), cat_batch("v", &["a", "b"]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
            shared_x(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        assert_eq!(ctx[0].x, None);
        assert_eq!(ctx[1].x, None);
    }

    #[test]
    fn unknown_field_errors_with_channel_and_field() {
        // Leaf encodes x on a field its batch doesn't carry → typed error
        // naming the channel and field, never a silent skip.
        let l0 = owned(spec_with_x("v"), num_batch("w", &[0.0, 10.0]));
        let l1 = owned(spec_with_x("v"), num_batch("w", &[5.0, 30.0]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
            shared_x(),
        );
        let inputs = [input(&l0), input(&l1)];
        let err = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap_err();
        match err {
            CompositeResolveError::UnknownField { channel, field, coord_flipped } => {
                assert_eq!(channel, "x");
                assert_eq!(field, "v");
                assert!(!coord_flipped, "spec_with_x's leaves aren't flipped");
            }
            other => panic!("expected UnknownField, got {other:?}"),
        }
    }

    /// R3: under `CoordFlip`, a leaf's `UnknownField` must name the channel the
    /// LEAF's own user wrote, not the resolved one it landed on post-swap.
    /// `flipped_encoding.y` holds `enc("v")` — as `prepare::build_layers` would
    /// have left it after swapping the leaf's own `x="v"` into the y slot — so
    /// the message must still say `x`, not the resolved `y`.
    #[test]
    fn unknown_field_errors_names_leafs_user_channel_under_flip() {
        use crate::spec::coord::CoordKind;
        let mut flipped_encoding = Encoding::default();
        flipped_encoding.y = Some(enc("v")); // post-flip landing spot
        let flipped_spec = ChartSpec { coord: Some(CoordKind::Flip), ..spec_with(flipped_encoding.clone()) };
        let l0 = LeafOwned {
            encoding: flipped_encoding.clone(),
            spec: flipped_spec.clone(),
            batch: num_batch("w", &[0.0, 10.0]),
            outputs: outputs(&num_batch("w", &[0.0, 10.0])),
        };
        let l1 = LeafOwned {
            encoding: flipped_encoding,
            spec: flipped_spec,
            batch: num_batch("w", &[5.0, 30.0]),
            outputs: outputs(&num_batch("w", &[5.0, 30.0])),
        };
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(None, Some("v")), leaf_node(None, Some("v"))],
            shared_both(),
        );
        let inputs = [input(&l0), input(&l1)];
        let err = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap_err();
        match &err {
            CompositeResolveError::UnknownField { channel, field, coord_flipped } => {
                // Resolved/internal token: this is the slot actually looked up.
                assert_eq!(*channel, "y");
                assert_eq!(field, "v");
                assert!(*coord_flipped);
            }
            other => panic!("expected UnknownField, got {other:?}"),
        }
        assert_eq!(
            err.to_string(),
            "shared x channel references unknown field 'v'",
            "message must name the leaf's own channel ('x'), not the resolved one ('y')"
        );
    }

    #[test]
    fn empty_column_unions_with_auto_default_domain() {
        // A leaf with NO rows on the shared channel contributes the auto-path
        // default domain (same as a flat empty chart renders with), and the
        // union proceeds — no error, no silent skip of the whole group. Pins
        // the empirically-established semantics: [default 0..1] ∪ [5..30].
        let l0 = owned(spec_with_x("v"), num_batch("v", &[]));
        let l1 = owned(spec_with_x("v"), num_batch("v", &[5.0, 30.0]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
            shared_x(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        let expect = Some(SharedDomain::Numeric { lo: 0.0, hi: 30.0 });
        assert_eq!(ctx[0].x, expect);
        assert_eq!(ctx[1].x, expect);
    }

    #[test]
    fn domain_resolve_error_display_names_channel_and_field() {
        // DomainResolve is defensive plumbing for domain-helper failures with
        // no reachable trigger through validated construction paths today —
        // pin its message shape so error UX doesn't regress if one appears.
        let err = CompositeResolveError::DomainResolve {
            channel: "y",
            field: "score".to_string(),
            source: crate::render::RenderError::ScaleResolutionFailed("boom".to_string()),
            coord_flipped: false,
        };
        let msg = err.to_string();
        assert!(msg.contains('y') && msg.contains("score") && msg.contains("boom"), "{msg}");
    }

    /// R3: `DomainResolve`'s `channel` un-flips the same way `UnknownField`'s
    /// does — the resolved `"y"` token renders as `"x"` under `coord_flipped`.
    #[test]
    fn domain_resolve_error_names_users_channel_under_flip() {
        let err = CompositeResolveError::DomainResolve {
            channel: "y",
            field: "score".to_string(),
            source: crate::render::RenderError::ScaleResolutionFailed("boom".to_string()),
            coord_flipped: true,
        };
        assert_eq!(
            err.to_string(),
            "failed to resolve shared x domain for field 'score': scale resolution failed: boom"
        );
    }

    #[test]
    fn flat_children_independent_does_not_share() {
        let l0 = owned(spec_with_x("v"), num_batch("v", &[0.0, 10.0]));
        let l1 = owned(spec_with_x("v"), num_batch("v", &[5.0, 30.0]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
            CompositeResolve::default(), // both independent
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        assert_eq!(ctx[0].x, None);
        assert_eq!(ctx[1].x, None);
    }

    #[test]
    fn flat_children_shared_ordinal_order_preserving_union() {
        let l0 = owned(spec_with_x("g"), cat_batch("g", &["b", "a"]));
        let l1 = owned(spec_with_x("g"), cat_batch("g", &["a", "c"]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("g"), None), leaf_node(Some("g"), None)],
            shared_x(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        // First-appearance order across the group: b, a (leaf0), then c (new in leaf1).
        let expect = Some(SharedDomain::Ordinal(vec!["b".into(), "a".into(), "c".into()]));
        assert_eq!(ctx[0].x, expect);
        assert_eq!(ctx[1].x, expect);
    }

    // -- color / size sharing (10-pre-b) --------------------------------------

    fn spec_with_color(field: &str) -> ChartSpec {
        let mut e = Encoding::default();
        e.color = Some(enc(field));
        spec_with(e)
    }
    fn leaf_node_color(field: &str) -> CompositeNode {
        CompositeNode::Leaf { spec: Box::new(spec_with_color(field)), data: 0, label: None }
    }
    fn spec_with_size(field: &str) -> ChartSpec {
        let mut e = Encoding::default();
        e.size = Some(enc(field));
        spec_with(e)
    }
    fn leaf_node_size(field: &str) -> CompositeNode {
        CompositeNode::Leaf { spec: Box::new(spec_with_size(field)), data: 0, label: None }
    }

    #[test]
    fn shared_color_unions_continuous_extents() {
        let l0 = owned(spec_with_color("c"), num_batch("c", &[0.0, 10.0]));
        let l1 = owned(spec_with_color("c"), num_batch("c", &[5.0, 30.0]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_color("c"), leaf_node_color("c")],
            shared_color(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        let expect = Some(SharedDomain::Numeric { lo: 0.0, hi: 30.0 });
        assert_eq!(ctx[0].color, expect);
        assert_eq!(ctx[1].color, expect);
        // Other channels untouched.
        assert_eq!(ctx[0].x, None);
        assert_eq!(ctx[0].size, None);
    }

    #[test]
    fn shared_color_unions_categorical_order_preserving() {
        let l0 = owned(spec_with_color("g"), cat_batch("g", &["b", "a"]));
        let l1 = owned(spec_with_color("g"), cat_batch("g", &["a", "c"]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_color("g"), leaf_node_color("g")],
            shared_color(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        // First-appearance order across the group: b, a (leaf0), then c (new in leaf1).
        let expect = Some(SharedDomain::Ordinal(vec!["b".into(), "a".into(), "c".into()]));
        assert_eq!(ctx[0].color, expect);
        assert_eq!(ctx[1].color, expect);
    }

    #[test]
    fn shared_size_unions_extents() {
        let l0 = owned(spec_with_size("s"), num_batch("s", &[2.0, 4.0]));
        let l1 = owned(spec_with_size("s"), num_batch("s", &[1.0, 9.0]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_size("s"), leaf_node_size("s")],
            shared_size(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        let expect = Some(SharedDomain::Numeric { lo: 1.0, hi: 9.0 });
        assert_eq!(ctx[0].size, expect);
        assert_eq!(ctx[1].size, expect);
    }

    #[test]
    fn independent_color_size_is_noop() {
        let l0 = owned(spec_with_color("c"), num_batch("c", &[0.0, 10.0]));
        let l1 = owned(spec_with_color("c"), num_batch("c", &[5.0, 30.0]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_color("c"), leaf_node_color("c")],
            CompositeResolve::default(), // all independent
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        assert_eq!(ctx[0].color, None);
        assert_eq!(ctx[1].color, None);
    }

    // -- nested color/size: leaf-span union + inheritance (GH #74) -------------

    #[test]
    fn outer_shared_color_spans_nested_composite_leaves() {
        // Outer hconcat shares color; the inner hconcat leaves color UNSET. The
        // union must span the nested subtree's leaves AND the outer sibling —
        // congruence-agnostic, unlike positional x/y pairing.
        let l0 = owned(spec_with_color("c"), num_batch("c", &[0.0, 10.0]));
        let l1 = owned(spec_with_color("c"), num_batch("c", &[5.0, 30.0]));
        let l2 = owned(spec_with_color("c"), num_batch("c", &[20.0, 25.0]));
        let inner = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_color("c"), leaf_node_color("c")],
            CompositeResolve::default(), // inner color unset -> inherits shared
        );
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![inner, leaf_node_color("c")],
            shared_color(),
        );
        let inputs = [input(&l0), input(&l1), input(&l2)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        let expect = Some(SharedDomain::Numeric { lo: 0.0, hi: 30.0 });
        assert_eq!(ctx[0].color, expect, "nested leaf 0 must join the outer union");
        assert_eq!(ctx[1].color, expect, "nested leaf 1 must join the outer union");
        assert_eq!(ctx[2].color, expect, "outer sibling must join the union");
    }

    #[test]
    fn nested_and_outer_both_shared_union_once_not_twice() {
        // Sharing declared at BOTH nesting levels: the outer union already
        // covers every leaf, so the inner shared node must not re-resolve a
        // narrower domain — every leaf carries the full union.
        let l0 = owned(spec_with_color("c"), num_batch("c", &[0.0, 10.0]));
        let l1 = owned(spec_with_color("c"), num_batch("c", &[5.0, 30.0]));
        let l2 = owned(spec_with_color("c"), num_batch("c", &[20.0, 25.0]));
        let inner = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_color("c"), leaf_node_color("c")],
            shared_color(), // inner explicitly shared too
        );
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![inner, leaf_node_color("c")],
            shared_color(),
        );
        let inputs = [input(&l0), input(&l1), input(&l2)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        let expect = Some(SharedDomain::Numeric { lo: 0.0, hi: 30.0 });
        assert_eq!(ctx[0].color, expect);
        assert_eq!(ctx[1].color, expect);
        assert_eq!(ctx[2].color, expect);
    }

    #[test]
    fn explicit_independent_child_opts_out_of_outer_shared_union() {
        // The inheritance rule's explicit-wins half (spec §6): an inner
        // composite that explicitly resolves color Independent excludes its
        // leaves from the outer shared union, while the outer sibling still
        // participates. The opted-out leaves keep no shared context (own
        // standalone resolution).
        let l0 = owned(spec_with_color("c"), num_batch("c", &[0.0, 10.0]));
        let l1 = owned(spec_with_color("c"), num_batch("c", &[5.0, 30.0]));
        let l2 = owned(spec_with_color("c"), num_batch("c", &[20.0, 25.0]));
        let indep_inner = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_color("c"), leaf_node_color("c")],
            CompositeResolve { color: Some(ResolveMode::Independent), ..CompositeResolve::default() },
        );
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![indep_inner, leaf_node_color("c")],
            shared_color(),
        );
        let inputs = [input(&l0), input(&l1), input(&l2)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        // Opted-out inner leaves: excluded from the outer union.
        assert_eq!(ctx[0].color, None, "explicit-independent inner leaf 0 opts out");
        assert_eq!(ctx[1].color, None, "explicit-independent inner leaf 1 opts out");
        // Outer sibling: the only participant, unions only its own extent.
        assert_eq!(ctx[2].color, Some(SharedDomain::Numeric { lo: 20.0, hi: 25.0 }));
    }

    #[test]
    fn inner_shared_under_independent_boundary_reshares_locally() {
        // A shared node nested under an explicit-independent boundary re-shares
        // among ITS own leaves (the inheritance chain reset — spec §6), even
        // though those leaves were excluded from an outer shared union.
        let l0 = owned(spec_with_color("c"), num_batch("c", &[0.0, 10.0])); // outer sibling
        let l1 = owned(spec_with_color("c"), num_batch("c", &[5.0, 30.0])); // inner-inner leaf
        let l2 = owned(spec_with_color("c"), num_batch("c", &[6.0, 40.0])); // inner-inner leaf
        let reshared = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_color("c"), leaf_node_color("c")],
            shared_color(), // re-shares locally
        );
        let indep_mid = composite(
            CompositeLayout::Hconcat,
            vec![reshared],
            CompositeResolve { color: Some(ResolveMode::Independent), ..CompositeResolve::default() },
        );
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_color("c"), indep_mid],
            shared_color(),
        );
        let inputs = [input(&l0), input(&l1), input(&l2)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        // Outer sibling: only participant of the OUTER union (mid opted out).
        assert_eq!(ctx[0].color, Some(SharedDomain::Numeric { lo: 0.0, hi: 10.0 }));
        // Inner-inner leaves: re-shared among themselves = [5, 40].
        let inner_expect = Some(SharedDomain::Numeric { lo: 5.0, hi: 40.0 });
        assert_eq!(ctx[1].color, inner_expect);
        assert_eq!(ctx[2].color, inner_expect);
    }

    #[test]
    fn explicit_color_scale_leaf_excluded_from_sharing() {
        use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};
        // Leaf 1 pins an explicit linear color scale → excluded from sharing.
        let mut e1 = Encoding::default();
        let mut c1 = enc("c");
        c1.scale = Some(ScaleSpec::Linear {
            common: ContinuousScaleCommon {
                domain: None,
                range: None,
                clamp: false,
                padding: None,
                scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: false,
            zero: false,
            system_injected: false,
        });
        e1.color = Some(c1);
        let s1 = spec_with(e1);
        let l0 = owned(spec_with_color("c"), num_batch("c", &[0.0, 10.0]));
        let l1 = LeafOwned {
            encoding: s1.encoding.clone(),
            outputs: outputs(&num_batch("c", &[5.0, 30.0])),
            batch: num_batch("c", &[5.0, 30.0]),
            spec: s1,
        };
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_color("c"), leaf_node_color("c")],
            shared_color(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        // Leaf 0 participates → shares only its own extent [0, 10].
        assert_eq!(ctx[0].color, Some(SharedDomain::Numeric { lo: 0.0, hi: 10.0 }));
        // Leaf 1's explicit color scale wins → no shared context injected.
        assert_eq!(ctx[1].color, None);
    }

    #[test]
    fn shared_color_mixed_continuous_categorical_group_skips() {
        // One leaf's color is numeric (continuous), the sibling's is categorical:
        // no meaningful union — the group's color sharing is skipped, both stay None.
        let l0 = owned(spec_with_color("c"), num_batch("c", &[0.0, 10.0]));
        let l1 = owned(spec_with_color("c"), cat_batch("c", &["a", "b"]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node_color("c"), leaf_node_color("c")],
            shared_color(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        assert_eq!(ctx[0].color, None);
        assert_eq!(ctx[1].color, None);
    }

    // -- box-transform (stat-extent union via layer fields) -------------------

    #[test]
    fn box_transform_children_share_whisker_extent() {
        // Leaf encodes y on the raw "value" column, but a layer reads the
        // "upper" whisker column from a named output whose extent is wider than
        // the raw column. numeric_domain_union must union the layer field, so the
        // shared domain reaches the whisker, not just raw min/max.
        let l0 = box_leaf(&[1.0, 2.0], &[0.5, 8.0]);
        let l1 = box_leaf(&[3.0, 4.0], &[2.0, 12.0]);
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![box_leaf_node(), box_leaf_node()],
            shared_both(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        // Union of whisker extents [0.5, 8.0] ∪ [2.0, 12.0] = [0.5, 12.0].
        assert_eq!(ctx[0].y, Some(SharedDomain::Numeric { lo: 0.5, hi: 12.0 }));
        assert_eq!(ctx[1].y, Some(SharedDomain::Numeric { lo: 0.5, hi: 12.0 }));
    }

    // -- congruent grids pair per position ------------------------------------

    #[test]
    fn congruent_grids_pair_per_position_not_within_grid() {
        // Two hconcat children, each of two leaves. Outer shares x. Leaf position
        // 0 shares across children; position 1 shares across children; positions
        // 0 and 1 within a child do NOT share.
        let a0 = owned(spec_with_x("v"), num_batch("v", &[0.0, 1.0]));
        let a1 = owned(spec_with_x("v"), num_batch("v", &[100.0, 200.0]));
        let b0 = owned(spec_with_x("v"), num_batch("v", &[2.0, 3.0]));
        let b1 = owned(spec_with_x("v"), num_batch("v", &[150.0, 400.0]));
        let child = || {
            composite(
                CompositeLayout::Hconcat,
                vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
                CompositeResolve::default(),
            )
        };
        let tree = composite(CompositeLayout::Grid, vec![child(), child()], shared_x());
        let inputs = [input(&a0), input(&a1), input(&b0), input(&b1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        // Position 0: a0 ∪ b0 = [0, 3].
        assert_eq!(ctx[0].x, Some(SharedDomain::Numeric { lo: 0.0, hi: 3.0 }));
        assert_eq!(ctx[2].x, Some(SharedDomain::Numeric { lo: 0.0, hi: 3.0 }));
        // Position 1: a1 ∪ b1 = [100, 400] — NOT unioned with position 0.
        assert_eq!(ctx[1].x, Some(SharedDomain::Numeric { lo: 100.0, hi: 400.0 }));
        assert_eq!(ctx[3].x, Some(SharedDomain::Numeric { lo: 100.0, hi: 400.0 }));
    }

    // -- non-congruent skip ---------------------------------------------------

    #[test]
    fn non_congruent_children_skip_sharing() {
        // One child is a leaf, the other a composite → non-congruent. Shared x is
        // skipped for the group; nothing is assigned.
        let leaf_input = owned(spec_with_x("v"), num_batch("v", &[0.0, 10.0]));
        let g0 = owned(spec_with_x("v"), num_batch("v", &[5.0, 20.0]));
        let g1 = owned(spec_with_x("v"), num_batch("v", &[6.0, 30.0]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![
                leaf_node(Some("v"), None),
                composite(
                    CompositeLayout::Hconcat,
                    vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
                    CompositeResolve::default(),
                ),
            ],
            shared_x(),
        );
        let inputs = [input(&leaf_input), input(&g0), input(&g1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        assert_eq!(ctx[0].x, None);
        assert_eq!(ctx[1].x, None);
        assert_eq!(ctx[2].x, None);
    }

    #[test]
    fn non_congruent_outer_still_recurses_inner_sharing() {
        // Non-congruent at the outer node (leaf + composite), but the inner
        // composite shares x among its own leaves — that inner sharing must still
        // apply (skip is a fall-back to recursion, not a hard stop).
        let leaf_input = owned(spec_with_x("v"), num_batch("v", &[0.0, 10.0]));
        let g0 = owned(spec_with_x("v"), num_batch("v", &[5.0, 20.0]));
        let g1 = owned(spec_with_x("v"), num_batch("v", &[6.0, 30.0]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![
                leaf_node(Some("v"), None),
                composite(
                    CompositeLayout::Hconcat,
                    vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
                    shared_x(), // inner shares
                ),
            ],
            shared_x(), // outer shares but is non-congruent → recurse
        );
        let inputs = [input(&leaf_input), input(&g0), input(&g1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        // Outer leaf: no sharing.
        assert_eq!(ctx[0].x, None);
        // Inner two leaves: shared among themselves = [5, 30].
        assert_eq!(ctx[1].x, Some(SharedDomain::Numeric { lo: 5.0, hi: 30.0 }));
        assert_eq!(ctx[2].x, Some(SharedDomain::Numeric { lo: 5.0, hi: 30.0 }));
    }

    // -- explicit user scale wins ---------------------------------------------

    #[test]
    fn explicit_scale_leaf_excluded_from_sharing() {
        use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};
        // Leaf 1 pins an explicit linear scale on x → excluded from sharing.
        let mut e1 = Encoding::default();
        let mut x1 = enc("v");
        x1.scale = Some(ScaleSpec::Linear {
            common: ContinuousScaleCommon {
                domain: None,
                range: None,
                clamp: false,
                padding: None,
                scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: false,
            zero: false,
            system_injected: false,
        });
        e1.x = Some(x1);
        let s1 = spec_with(e1.clone());
        let l0 = owned(spec_with_x("v"), num_batch("v", &[0.0, 10.0]));
        let l1 = LeafOwned {
            encoding: s1.encoding.clone(),
            outputs: outputs(&num_batch("v", &[5.0, 30.0])),
            batch: num_batch("v", &[5.0, 30.0]),
            spec: s1,
        };
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
            shared_x(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        // Leaf 0 is the only participant → shares only its own extent [0, 10].
        assert_eq!(ctx[0].x, Some(SharedDomain::Numeric { lo: 0.0, hi: 10.0 }));
        // Leaf 1's explicit scale wins → no shared context injected.
        assert_eq!(ctx[1].x, None);
    }

    // -- system-injected scale participates (#247A) ----------------------------

    /// A leaf carrying the bar zero-anchor's injected scale on `x`, with its own
    /// extent and its own provenance flag.
    fn injected_zero_leaf(system_injected: bool, data: &[f64]) -> LeafOwned {
        use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};
        let mut e = Encoding::default();
        let mut x = enc("v");
        x.scale = Some(ScaleSpec::Linear {
            common: ContinuousScaleCommon {
                domain: None,
                range: None,
                clamp: false,
                padding: None,
                scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: false,
            zero: true,
            system_injected,
        });
        e.x = Some(x);
        let spec = spec_with(e);
        LeafOwned {
            encoding: spec.encoding.clone(),
            outputs: outputs(&num_batch("v", data)),
            batch: num_batch("v", data),
            spec,
        }
    }

    /// #247A: the injection is ferrum's, not the caller's, so it must not buy the
    /// leaf out of sharing — and the anchor it asked for has to travel with the
    /// leaf's contribution, since the leaf no longer resolves it through the
    /// explicit bypass.
    ///
    /// Leaf 0 spans `[12, 40]` and the injected leaf spans `[5, 30]`, so the
    /// plain union would be `[5, 40]`: a `lo` of `0` can only have come from the
    /// anchor, and `ctx[1]` being `Some` at all is what was broken (pre-fix both
    /// assertions failed — `ctx[0]` was leaf 0's own `[12, 40]` and `ctx[1]` was
    /// `None`).
    #[test]
    fn system_injected_scale_leaf_shares_and_carries_its_zero_anchor() {
        let l0 = owned(spec_with_x("v"), num_batch("v", &[12.0, 40.0]));
        let l1 = injected_zero_leaf(true, &[5.0, 30.0]);
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
            shared_x(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        let unioned = Some(SharedDomain::Numeric { lo: 0.0, hi: 40.0 });
        assert_eq!(ctx[0].x, unioned);
        assert_eq!(ctx[1].x, unioned);
    }

    /// The control that makes the assertion above about PROVENANCE rather than
    /// about `zero`: the identical scale the caller wrote keeps the documented
    /// "explicit user scale wins" exclusion, so leaf 0 shares only its own
    /// extent and the injected-scale leaf gets no context.
    #[test]
    fn user_written_zero_scale_leaf_stays_excluded_from_sharing() {
        let l0 = owned(spec_with_x("v"), num_batch("v", &[12.0, 40.0]));
        let l1 = injected_zero_leaf(false, &[5.0, 30.0]);
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
            shared_x(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        assert_eq!(ctx[0].x, Some(SharedDomain::Numeric { lo: 12.0, hi: 40.0 }));
        assert_eq!(ctx[1].x, None);
    }

    /// An injected scale with no anchor to carry (`zero: false` — a shape the
    /// current injection never emits, but the wire admits) participates with its
    /// raw extent: the provenance bit governs who may share, `zero` governs what
    /// is shared. Without this, reading the two as one flag would look correct on
    /// every test above.
    #[test]
    fn system_injected_scale_without_zero_shares_its_raw_extent() {
        use crate::spec::encoding::ScaleSpec;
        let l0 = owned(spec_with_x("v"), num_batch("v", &[12.0, 40.0]));
        let mut l1 = injected_zero_leaf(true, &[5.0, 30.0]);
        if let Some(ScaleSpec::Linear { zero, .. }) =
            l1.spec.encoding.x.as_mut().and_then(|e| e.scale.as_mut())
        {
            *zero = false;
        }
        l1.encoding = l1.spec.encoding.clone();
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
            shared_x(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        let unioned = Some(SharedDomain::Numeric { lo: 5.0, hi: 40.0 });
        assert_eq!(ctx[0].x, unioned);
        assert_eq!(ctx[1].x, unioned);
    }

    // -- faceted child outer/inner separation ---------------------------------

    #[test]
    fn faceted_independent_child_excluded_on_that_channel_only() {
        // Both leaves are faceted with x independent, y shared. Outer shares both.
        // x must stay None (internal independence preserved — pdp compare=), while
        // y shares the union across leaves.
        let mut e = Encoding::default();
        e.x = Some(enc("v"));
        e.y = Some(enc("w"));
        let facet = FacetSpec {
            field: "panel".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve { x: ResolveMode::Independent, y: ResolveMode::Shared },
        };
        let make = |vy: &[f64]| {
            let spec = ChartSpec { encoding: e.clone(), facet: Some(facet.clone()), ..base_spec() };
            // Batch carries both x and y columns.
            let schema = Arc::new(Schema::new(vec![
                Field::new("v", DataType::Float64, true),
                Field::new("w", DataType::Float64, true),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Float64Array::from(vec![0.0, 1.0])),
                    Arc::new(Float64Array::from(vy.to_vec())),
                ],
            )
            .unwrap();
            owned(spec, batch)
        };
        let l0 = make(&[0.0, 5.0]);
        let l1 = make(&[2.0, 20.0]);
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![
                leaf_node(Some("v"), Some("w")),
                leaf_node(Some("v"), Some("w")),
            ],
            shared_both(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        // x: faceted-independent → excluded.
        assert_eq!(ctx[0].x, None);
        assert_eq!(ctx[1].x, None);
        // y: faceted-shared → participates, unions [0, 20].
        assert_eq!(ctx[0].y, Some(SharedDomain::Numeric { lo: 0.0, hi: 20.0 }));
        assert_eq!(ctx[1].y, Some(SharedDomain::Numeric { lo: 0.0, hi: 20.0 }));
    }

    // -- leaf-count guard -----------------------------------------------------

    #[test]
    fn leaf_count_mismatch_errors() {
        let l0 = owned(spec_with_x("v"), num_batch("v", &[0.0, 1.0]));
        let tree = composite(
            CompositeLayout::Hconcat,
            vec![leaf_node(Some("v"), None), leaf_node(Some("v"), None)],
            shared_x(),
        );
        let inputs = [input(&l0)]; // only 1, tree has 2
        let err = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap_err();
        assert!(matches!(err, CompositeResolveError::LeafCountMismatch { expected: 2, got: 1, .. }));
    }

    #[test]
    fn flatten_leaf_specs_is_pre_order() {
        let tree = composite(
            CompositeLayout::Grid,
            vec![
                composite(
                    CompositeLayout::Hconcat,
                    vec![leaf_node(Some("a"), None), leaf_node(Some("b"), None)],
                    CompositeResolve::default(),
                ),
                leaf_node(Some("c"), None),
            ],
            CompositeResolve::default(),
        );
        let specs = flatten_leaf_specs(&tree);
        let fields: Vec<&str> = specs
            .iter()
            .map(|s| s.encoding.x.as_ref().unwrap().field.as_str())
            .collect();
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[test]
    fn flatten_leaves_skips_holes_and_resolve_composite_scales_ignores_them() {
        // A grid with a hole among its children: the hole contributes no
        // (spec, data) pair to `flatten_leaves`/`flatten_leaf_specs`, and
        // `resolve_composite_scales`'s leaf-count check only counts the real
        // leaves — 3, not 4.
        let tree = composite(
            CompositeLayout::Grid,
            vec![
                leaf_node(Some("a"), None),
                CompositeNode::Hole { width: None, height: None },
                leaf_node(Some("b"), None),
                leaf_node(Some("c"), None),
            ],
            CompositeResolve::default(),
        );
        let specs = flatten_leaf_specs(&tree);
        let fields: Vec<&str> = specs
            .iter()
            .map(|s| s.encoding.x.as_ref().unwrap().field.as_str())
            .collect();
        assert_eq!(fields, vec!["a", "b", "c"], "hole must not appear in the flattened leaf list");

        let l0 = owned(spec_with_x("a"), num_batch("a", &[0.0, 1.0]));
        let l1 = owned(spec_with_x("b"), num_batch("b", &[0.0, 1.0]));
        let l2 = owned(spec_with_x("c"), num_batch("c", &[0.0, 1.0]));
        let inputs = [input(&l0), input(&l1), input(&l2)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        assert_eq!(ctx.len(), 3, "leaf count excludes the hole");
    }

    // -- helpers for the box test --------------------------------------------

    fn spec_with_x(field: &str) -> ChartSpec {
        let mut e = Encoding::default();
        e.x = Some(enc(field));
        spec_with(e)
    }

    /// A leaf spec whose chart-level y is the raw "value" column and whose single
    /// layer reads the "upper" whisker column from the named output "box".
    fn box_leaf_node() -> CompositeNode {
        CompositeNode::Leaf { spec: Box::new(box_leaf_spec()), data: 0, label: None }
    }

    fn box_leaf_spec() -> ChartSpec {
        use crate::spec::layer::Layer;
        let mut chart_enc = Encoding::default();
        chart_enc.y = Some(enc("value"));
        let mut layer_enc = Encoding::default();
        layer_enc.y = Some(enc("upper"));
        // `Mark::Rect`, not `Mark::Bar` (T5r): mirrors what `desugar_boxplot`
        // actually emits for the IQR box body (`marks/composite.py`'s layer
        // 4, a `rect`) — `Mark::Bar` here was a synthetic mismatch with the
        // real production shape, which the T5r bar zero-anchor
        // (`scale_resolve/domain.rs::numeric_domain_union`) exposed: this
        // fixture's "upper" whisker extent (0.5..12.0) doesn't start at
        // zero, so under `Mark::Bar` the new rule zero-anchored it and this
        // test's asserted union stopped matching. A real boxplot never hits
        // that rule (its body layer is `rect`), so fixing the fixture's mark
        // to match reality is the correct fix, not a workaround.
        let layer = Layer {
            mark: Mark::Rect,
            encoding: layer_enc,
            data_source: Some("box".into()),
            ..Default::default()
        };
        ChartSpec {
            encoding: chart_enc,
            layers: Some(vec![layer]),
            ..base_spec()
        }
    }

    /// Owned leaf data for the box test: raw values in the final batch, whisker
    /// values in the named "box" output.
    fn box_leaf(raw: &[f64], whisker: &[f64]) -> LeafOwned {
        let spec = box_leaf_spec();
        let final_batch = num_batch("value", raw);
        let mut outs = outputs(&final_batch);
        outs.insert("box".to_string(), num_batch("upper", whisker));
        LeafOwned {
            encoding: spec.encoding.clone(),
            spec,
            batch: final_batch,
            outputs: outs,
        }
    }

    // -- #120(9) + N1: a layer column that cannot serve the shared numeric
    // axis is refused, at this module's own shared-scale site ---------------

    /// One leaf, `fm.layer(point(y="a"), line(y="s"))` lowered: chart-level
    /// `y` reads `a`, the second layer's own `y` reads `s`, a `Utf8` column.
    /// `leaf_positional_domain` is the static overlay route's own call into
    /// `numeric_domain_union` (see this function's doc above), so this is
    /// that route's own site, not a mirror of it.
    fn layer_utf8_y_leaf() -> LeafOwned {
        let mut chart_enc = Encoding::default();
        chart_enc.x = Some(enc("b"));
        chart_enc.y = Some(enc("a"));
        let mut point_layer_enc = Encoding::default();
        point_layer_enc.x = Some(enc("b"));
        point_layer_enc.y = Some(enc("a"));
        let mut line_layer_enc = Encoding::default();
        line_layer_enc.y = Some(enc("s"));
        let point_layer = Layer {
            mark: Mark::Point,
            encoding: point_layer_enc,
            ..Default::default()
        };
        let line_layer = Layer {
            mark: Mark::Line,
            encoding: line_layer_enc,
            ..Default::default()
        };
        let spec = ChartSpec {
            encoding: chart_enc,
            layers: Some(vec![point_layer, line_layer]),
            ..base_spec()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![3.0, 4.0])),
                Arc::new(StringArray::from(vec!["p", "q"])),
            ],
        )
        .unwrap();
        owned(spec, batch)
    }

    #[test]
    fn leaf_positional_domain_refuses_a_layer_utf8_column_on_the_shared_y_axis() {
        let leaf = layer_utf8_y_leaf();
        let err = leaf_positional_domain(&input(&leaf), Channel::Y, LayerUnion::Every)
            .expect_err("a Utf8 layer column must refuse the static overlay route too");
        match err {
            CompositeResolveError::DomainResolve { source, channel, field, .. } => {
                assert_eq!(channel, "y");
                assert_eq!(field, "a", "the LEAF's own chart-level field names the failure site");
                assert_eq!(
                    source,
                    RenderError::LayerAxisColumnUnplaceable {
                        layer: 1,
                        detail: Box::new(crate::render::LayerAxisColumnDetail {
                            channel: "y".to_string(),
                            field: "s".to_string(),
                            dtype: "Utf8".to_string(),
                        }),
                    },
                    "the wrapped source must be the LayerAxisColumnUnplaceable refusal, not a \
                     generic scale-resolution failure"
                );
            }
            other => panic!("expected CompositeResolveError::DomainResolve, got {other:?}"),
        }
    }

    // -- #120(9) + N1: two SEPARATE Overlay leaves, not one leaf's
    // own internal `layers` -----------------------------------------------
    //
    // `fm.layer(point_chart, line_chart)` (`LayerChart._composite_tree`,
    // `src/ferrum/composition.py`) lowers each member to its OWN leaf via
    // `_lower_overlay_leaf_node` — neither leaf's `ChartSpec` carries an
    // internal `layers` array, so `leaf_positional_domain_refuses_a_layer_
    // utf8_column_on_the_shared_y_axis` above (which pins a single leaf
    // whose OWN spec carries one) never exercises this shape. The real
    // cross-leaf reconciliation for two separate Overlay leaves is
    // `resolve_group`'s mixed-domain branch, reached through
    // `resolve_composite_scales` → `resolve_channel` → `resolve_group`.

    fn overlay_leaf(x_field: &str, y_field: &str, batch: RecordBatch) -> LeafOwned {
        let mut e = Encoding::default();
        e.x = Some(enc(x_field));
        e.y = Some(enc(y_field));
        owned(spec_with(e), batch)
    }

    #[test]
    fn overlay_of_two_leaves_refuses_a_utf8_column_on_the_shared_y_axis() {
        let numeric_leaf = overlay_leaf(
            "b",
            "a",
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("a", DataType::Float64, false),
                    Field::new("b", DataType::Float64, false),
                ])),
                vec![
                    Arc::new(Float64Array::from(vec![1.0, 2.0])),
                    Arc::new(Float64Array::from(vec![3.0, 4.0])),
                ],
            )
            .unwrap(),
        );
        let utf8_leaf = overlay_leaf(
            "b",
            "s",
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("b", DataType::Float64, false),
                    Field::new("s", DataType::Utf8, false),
                ])),
                vec![
                    Arc::new(Float64Array::from(vec![3.0, 4.0])),
                    Arc::new(StringArray::from(vec!["p", "q"])),
                ],
            )
            .unwrap(),
        );
        let tree = composite(
            CompositeLayout::Overlay,
            vec![leaf_node(Some("b"), Some("a")), leaf_node(Some("b"), Some("s"))],
            shared_both(),
        );
        let inputs = [input(&numeric_leaf), input(&utf8_leaf)];
        let err = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).expect_err(
            "a Utf8 y column beside a numeric y column on ONE shared Overlay panel must refuse",
        );
        match err {
            CompositeResolveError::DomainResolve { source, channel, field, .. } => {
                assert_eq!(channel, "y");
                assert_eq!(field, "s", "the offending member's own field names the failure site");
                assert_eq!(
                    source,
                    RenderError::LayerAxisColumnUnplaceable {
                        layer: 1,
                        detail: Box::new(crate::render::LayerAxisColumnDetail {
                            channel: "y".to_string(),
                            field: "s".to_string(),
                            dtype: "Utf8".to_string(),
                        }),
                    },
                    "layer 1 is this member's POSITION among the fm.layer(...) members, the same \
                     naming `numeric_domain_union_over`'s internal-layers gate uses"
                );
            }
            other => panic!("expected CompositeResolveError::DomainResolve, got {other:?}"),
        }
    }

    /// Control: two numeric leaves in an Overlay group still union normally
    /// — the new check only fires on a genuine mismatch.
    #[test]
    fn overlay_of_two_numeric_leaves_still_unions() {
        let mut e = Encoding::default();
        e.y = Some(enc("a"));
        let l0 = owned(spec_with(e.clone()), num_batch("a", &[0.0, 10.0]));
        let l1 = owned(spec_with(e), num_batch("a", &[5.0, 30.0]));
        let tree = composite(
            CompositeLayout::Overlay,
            vec![leaf_node(None, Some("a")), leaf_node(None, Some("a"))],
            shared_both(),
        );
        let inputs = [input(&l0), input(&l1)];
        let ctx = resolve_composite_scales(&tree, &inputs, &mut Vec::new()).unwrap();
        let expect = Some(SharedDomain::Numeric { lo: 0.0, hi: 30.0 });
        assert_eq!(ctx[0].y, expect);
        assert_eq!(ctx[1].y, expect);
    }

    /// `leaf.spec` carries the AUTHORED `CoordFlip` (`LeafResolveInput::spec`'s
    /// own doc: the leaf's ORIGINAL `ChartSpec`, untouched); `encoding` is the
    /// already-swapped RENDERING encoding (`LeafResolveInput::encoding`'s own
    /// doc: "rendering (post-CoordFlip...) encoding") — the shape this
    /// function's real caller (`prepare::build_layers`'s per-leaf swap) hands
    /// it, x/y exchanged from what `spec.encoding` authored.
    fn overlay_flipped_leaf(x_field: &str, y_field: &str, batch: RecordBatch) -> LeafOwned {
        let mut authored = Encoding::default();
        authored.x = Some(enc(x_field));
        authored.y = Some(enc(y_field));
        let spec = ChartSpec {
            encoding: authored,
            coord: Some(crate::spec::coord::CoordKind::Flip),
            ..base_spec()
        };
        let mut rendering = Encoding::default();
        rendering.x = Some(enc(y_field));
        rendering.y = Some(enc(x_field));
        let outs = outputs(&batch);
        LeafOwned { spec, encoding: rendering, batch, outputs: outs }
    }

    /// `fm.layer(_shared_bar(), _shared_point_flipped())`
    /// (`tests/test_layer_coord.py`'s own fixture) at the Rust boundary
    /// `resolve_composite_scales` owns: the flipped member's exchange lands
    /// its `cat` column on the bar's continuous y, which cannot draw it, so
    /// the overlay refuses with the coord-exchange reason the merged-flat
    /// routes raise, in the composite route's envelope.
    #[test]
    fn overlay_of_a_flipped_member_whose_category_lands_on_the_continuous_axis_refuses() {
        fn shared_frame() -> RecordBatch {
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("cat", DataType::Utf8, false),
                    Field::new("v", DataType::Float64, false),
                ])),
                vec![
                    Arc::new(StringArray::from(vec!["a", "b", "c"])),
                    Arc::new(Float64Array::from(vec![3.0, 1.0, 2.0])),
                ],
            )
            .unwrap()
        }
        let bar = overlay_leaf("cat", "v", shared_frame());
        let point_flipped = overlay_flipped_leaf("cat", "v", shared_frame());
        let tree = composite(
            CompositeLayout::Overlay,
            vec![leaf_node(Some("cat"), Some("v")), leaf_node(Some("v"), Some("cat"))],
            shared_both(),
        );
        let inputs = [input(&bar), input(&point_flipped)];
        let err = resolve_composite_scales(&tree, &inputs, &mut Vec::new())
            .expect_err("the flipped member's category cannot land on the continuous y");
        let CompositeResolveError::DomainResolve { channel: "y", field, source, coord_flipped: false } = &err else {
            panic!("unexpected composite error {err:?}");
        };
        assert_eq!(field, "cat");
        assert_eq!(
            *source,
            RenderError::LayerCoordUnplaceable {
                layer: 1,
                coord: "fm.CoordFlip",
                reason: Box::new(crate::render::LayerCoordUnplaceableReason::ExchangedChannelUnservable {
                    channel: "y",
                    column: "cat".to_string(),
                    dtype: "Utf8".to_string(),
                    axis_kind: "continuous",
                }),
            }
        );
    }

    /// Two members BOTH flipped the same way: the establishing member's
    /// (group[0]'s) own flip state decides the panel's, so a second member
    /// flipped identically is not deferred and its own genuinely non-placeable
    /// column still refuses. The refusal must name the RESOLVED axis
    /// consistently everywhere in the message — the outer composite envelope's
    /// `channel`, the inner text's "shared {channel} axis", and the
    /// `resolve={"{channel}": ...}` hint all name the same token, the one a
    /// caller would actually write into `resolve=` (a property of the shared
    /// panel, verified against `_composite_resolve_field`'s pass-through
    /// lowering: a `resolve=` key is never adjusted for any one member's own
    /// coord).
    #[test]
    fn overlay_refusal_on_a_flipped_panel_names_the_resolved_key_consistently() {
        let leaf0_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("v", DataType::Float64, false),
                Field::new("cat", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let leaf1_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("s", DataType::Utf8, false),
                Field::new("w", DataType::Float64, false),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["p", "q"])),
                Arc::new(Float64Array::from(vec![3.0, 4.0])),
            ],
        )
        .unwrap();
        // Both flipped: leaf0's authored (x="cat", y="v") renders x="v"
        // (numeric, establishing a continuous axis); leaf1's authored
        // (x="w", y="s") renders x="s" (Utf8, genuinely non-placeable).
        let leaf0 = overlay_flipped_leaf("cat", "v", leaf0_batch);
        let leaf1 = overlay_flipped_leaf("w", "s", leaf1_batch);
        let tree = composite(
            CompositeLayout::Overlay,
            vec![leaf_node(Some("cat"), Some("v")), leaf_node(Some("w"), Some("s"))],
            shared_both(),
        );
        let inputs = [input(&leaf0), input(&leaf1)];
        let err = resolve_composite_scales(&tree, &inputs, &mut Vec::new())
            .expect_err("a flip-matching sibling's genuinely non-placeable column must refuse");
        let message = err.to_string();
        assert!(
            message.contains("failed to resolve shared x domain for field 's'"),
            "the outer envelope must name the resolved x slot, not un-flip it: {message}"
        );
        assert!(
            message.contains("cannot be drawn on the shared x axis"),
            "the inner text must agree with the outer envelope: {message}"
        );
        assert!(
            message.contains("resolve={\"x\": \"independent\"}"),
            "the hint must name the resolved x slot, the key a caller writes into resolve=: {message}"
        );
        assert!(!message.contains("shared y"), "no y token may leak in: {message}");
    }

    /// A later group member's own, unrelated resolution defect (its
    /// `y` field absent from its own batch entirely) must not pre-empt the
    /// typed shared-axis refusal an EARLIER member's genuinely non-placeable
    /// column already owns. Only `group[0]`'s domain may be resolved before
    /// the refusal loop runs; resolving every member's (as `resolve_group`
    /// itself does) would let `leaf2`'s `UnknownField` surface first, purely
    /// because it happens to be resolved in the same pass, not because it
    /// comes first in the caller's own member order.
    #[test]
    fn an_unrelated_later_member_defect_does_not_preempt_the_shared_axis_refusal() {
        let leaf0 = overlay_leaf(
            "b",
            "a",
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("b", DataType::Float64, false),
                    Field::new("a", DataType::Float64, false),
                ])),
                vec![
                    Arc::new(Float64Array::from(vec![3.0, 4.0])),
                    Arc::new(Float64Array::from(vec![1.0, 2.0])),
                ],
            )
            .unwrap(),
        );
        let leaf1 = overlay_leaf(
            "b",
            "s",
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("b", DataType::Float64, false),
                    Field::new("s", DataType::Utf8, false),
                ])),
                vec![
                    Arc::new(Float64Array::from(vec![3.0, 4.0])),
                    Arc::new(StringArray::from(vec!["p", "q"])),
                ],
            )
            .unwrap(),
        );
        // leaf2's OWN batch carries no "y2" column at all -- an unrelated
        // resolution defect, distinct from N1's dtype refusal.
        let leaf2 = overlay_leaf(
            "b",
            "y2",
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("b", DataType::Float64, false)])),
                vec![Arc::new(Float64Array::from(vec![3.0, 4.0]))],
            )
            .unwrap(),
        );
        let tree = composite(
            CompositeLayout::Overlay,
            vec![
                leaf_node(Some("b"), Some("a")),
                leaf_node(Some("b"), Some("s")),
                leaf_node(Some("b"), Some("y2")),
            ],
            shared_both(),
        );
        let inputs = [input(&leaf0), input(&leaf1), input(&leaf2)];
        let err = resolve_composite_scales(&tree, &inputs, &mut Vec::new())
            .expect_err("leaf1's Utf8 column must still refuse the shared y axis");
        match err {
            CompositeResolveError::DomainResolve { source, .. } => {
                assert_eq!(
                    source,
                    RenderError::LayerAxisColumnUnplaceable {
                        layer: 1,
                        detail: Box::new(crate::render::LayerAxisColumnDetail {
                            channel: "y".to_string(),
                            field: "s".to_string(),
                            dtype: "Utf8".to_string(),
                        }),
                    },
                    "leaf2's unrelated UnknownField must not win the race"
                );
            }
            other => panic!(
                "expected the typed shared-axis refusal, got {other:?} \
                 (leaf2's unrelated defect pre-empted it)"
            ),
        }
    }

    // -- an overlay group under a caller's positional scale -------------------

    /// The composite-route mirror of the flat layered route's layer-selection
    /// matrix: an `Overlay` of two point leaves sharing y, the first binding
    /// `v = [1, 3]` under the caller's linear scale (domain-less or stating
    /// `[1, 4]`), the second binding `p = [40, 80]` and declaring one of five
    /// things on y. Each leaf's y axis is resolved through the real leaf-scale
    /// entry with the context the resolve pass hands it. A domain-less caller
    /// scale spans the sibling that declares no scale, carries an equal copy,
    /// or carries only the injected bar anchor, `(1, 80)`, and that sibling
    /// resolves the same axis; a sibling with a different scale or its own
    /// domain stays out, `(1, 3)`. A stated domain wins over every sibling.
    #[test]
    fn overlay_caller_scale_layer_selection_matrix() {
        use crate::layout::ThemeInputs;
        use crate::render::scale_resolve::resolve_scales_with_leaf_context;
        use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};

        let linear = |domain: Option<Vec<f64>>, zero: bool, system_injected: bool| ScaleSpec::Linear {
            common: ContinuousScaleCommon {
                domain,
                range: None,
                clamp: false,
                padding: None,
                scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: false,
            zero,
            system_injected,
        };
        let batch = |field: &str, values: [f64; 2]| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("i", DataType::Float64, false),
                Field::new(field, DataType::Float64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(Float64Array::from(vec![0.0, 1.0])), Arc::new(Float64Array::from(values.to_vec()))],
            )
            .unwrap()
        };
        let leaf = |field: &str, scale: Option<ScaleSpec>| {
            let mut e = Encoding::default();
            e.x = Some(enc("i"));
            e.y = Some(EncodingSpec { scale, ..enc(field) });
            spec_with(e)
        };
        let siblings: [(&str, Option<ScaleSpec>); 5] = [
            ("no scale", None),
            ("equal copy", Some(linear(None, false, false))),
            ("injected anchor", Some(linear(None, true, true))),
            ("different scale", Some(linear(None, true, false))),
            ("own domain", Some(linear(Some(vec![0.0, 100.0]), false, false))),
        ];
        let callers: [(&str, Option<Vec<f64>>); 2] =
            [("domain-less", None), ("stated domain", Some(vec![1.0, 4.0]))];
        let tree = composite(
            CompositeLayout::Overlay,
            vec![leaf_node(Some("i"), Some("v")), leaf_node(Some("i"), Some("p"))],
            shared_both(),
        );
        let theme = ThemeInputs::default();
        let pr = (0.0, 100.0);
        let mut failures = Vec::new();
        for (caller_name, caller_domain) in &callers {
            for (sibling_name, sibling_scale) in &siblings {
                let a = owned(leaf("v", Some(linear(caller_domain.clone(), false, false))), batch("v", [1.0, 3.0]));
                let b = owned(leaf("p", sibling_scale.clone()), batch("p", [40.0, 80.0]));
                let ctx = resolve_composite_scales(&tree, &[input(&a), input(&b)], &mut Vec::new()).unwrap();
                let y_domain = |o: &LeafOwned, c: &LeafScaleContext| {
                    let (scales, _) = resolve_scales_with_leaf_context(
                        &o.spec, &o.spec.encoding, &o.batch, &o.outputs, pr, pr, &theme, Some(c),
                    )
                    .unwrap();
                    scales.y.data_domain()
                };
                let axis = y_domain(&a, &ctx[0]);
                // The sibling adopts the caller's scale when it declares none
                // (or only the anchor) or carries an equal copy; the
                // domain-less "equal copy" is a different scale from a caller
                // stating a domain. Its data joins only a domain-less axis.
                let adopts = matches!(*sibling_name, "no scale" | "injected anchor")
                    || (caller_domain.is_none() && *sibling_name == "equal copy");
                let want = match (caller_domain, adopts) {
                    (Some(_), _) => (1.0, 4.0),
                    (None, true) => (1.0, 80.0),
                    (None, false) => (1.0, 3.0),
                };
                if axis != Some(want) {
                    failures.push(format!("{caller_name} x {sibling_name}: axis {axis:?}, want {want:?}"));
                }
                if ctx[1].y_scale.is_some() != adopts {
                    failures.push(format!("{caller_name} x {sibling_name}: sibling adopts the caller scale: {}", !adopts));
                }
                if adopts {
                    let sibling = y_domain(&b, &ctx[1]);
                    if sibling != axis {
                        failures.push(format!("{caller_name} x {sibling_name}: sibling {sibling:?}, axis {axis:?}"));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{} red cells:\n{}", failures.len(), failures.join("\n"));
    }

    /// An overlay whose first leaf carries the caller's linear scale with no
    /// `domain` establishes a continuous y axis, so a later member whose own
    /// column is `Utf8` refuses in the shared-axis voice, naming that member.
    /// (A `Utf8` first member decides its own categorical axis instead; see
    /// `a_later_continuous_caller_scale_does_not_take_a_categorical_layer_0_axis_on_both_routes`.)
    #[test]
    fn caller_scale_overlay_refuses_a_utf8_member_on_the_shared_y_axis() {
        use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};
        let mut scaled = overlay_leaf(
            "b",
            "a",
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("b", DataType::Float64, false),
                    Field::new("a", DataType::Float64, false),
                ])),
                vec![
                    Arc::new(Float64Array::from(vec![3.0, 4.0])),
                    Arc::new(Float64Array::from(vec![1.0, 2.0])),
                ],
            )
            .unwrap(),
        );
        let scale = ScaleSpec::Linear {
            common: ContinuousScaleCommon {
                domain: None,
                range: None,
                clamp: false,
                padding: None,
                scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: false,
            zero: false,
            system_injected: false,
        };
        scaled.spec.encoding.y.as_mut().unwrap().scale = Some(scale.clone());
        scaled.encoding.y.as_mut().unwrap().scale = Some(scale);
        let text = overlay_leaf(
            "b",
            "s",
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("b", DataType::Float64, false),
                    Field::new("s", DataType::Utf8, false),
                ])),
                vec![
                    Arc::new(Float64Array::from(vec![3.0, 4.0])),
                    Arc::new(StringArray::from(vec!["p", "q"])),
                ],
            )
            .unwrap(),
        );
        let members = [&scaled, &text];
        let fields = members.map(|m| m.spec.encoding.y.as_ref().unwrap().field.clone());
        let tree = composite(
            CompositeLayout::Overlay,
            fields.iter().map(|f| leaf_node(Some("b"), Some(f))).collect(),
            shared_both(),
        );
        let err = resolve_composite_scales(&tree, &members.map(input), &mut Vec::new())
            .expect_err("a Utf8 member cannot sit on the caller scale's continuous axis");
        match err {
            CompositeResolveError::DomainResolve { source, .. } => assert_eq!(
                source,
                RenderError::LayerAxisColumnUnplaceable {
                    layer: 1,
                    detail: Box::new(crate::render::LayerAxisColumnDetail {
                        channel: "y".to_string(),
                        field: "s".to_string(),
                        dtype: "Utf8".to_string(),
                    }),
                }
            ),
            other => panic!("expected the typed shared-axis refusal, got {other:?}"),
        }
    }

    /// One member of an overlay for [`overlay_routes`]: the `y` column it
    /// binds, that column's values, and its own `y` scale.
    struct Member {
        field: &'static str,
        values: arrow::array::ArrayRef,
        scale: Option<crate::spec::encoding::ScaleSpec>,
    }

    fn member(field: &'static str, values: &[f64], scale: Option<serde_json::Value>) -> Member {
        Member {
            field,
            values: Arc::new(Float64Array::from(values.to_vec())),
            scale: scale.map(|s| serde_json::from_value(s).unwrap()),
        }
    }

    /// How both layering routes resolve `members`' shared axis on the
    /// channel they bind, layered in the given order.
    struct Routes {
        /// Each composite overlay leaf's data domain on the channel, and
        /// whether it resolves with a scale its group adopted.
        leaves: Result<Vec<(Option<(f64, f64)>, bool)>, RenderError>,
        /// The flat layered route's data domain on the channel, its
        /// chart-level encoding being layer 0's as `+` lowers it.
        flat: Result<Option<(f64, f64)>, RenderError>,
        /// Each route's `LayerScaleExcluded` warnings: composite, then flat.
        excluded: (Vec<RenderWarning>, Vec<RenderWarning>),
    }

    impl Routes {
        fn unwrap(self) -> (Vec<(Option<(f64, f64)>, bool)>, Option<(f64, f64)>) {
            (self.leaves.expect("fm.layer resolves"), self.flat.expect("+ resolves"))
        }
    }

    fn overlay_routes(members: &[Member]) -> Routes {
        overlay_routes_on(Channel::Y, members)
    }

    /// [`overlay_routes`] with the members binding `channel` (`X` or `Y`)
    /// and the other positional channel bound to a shared index column.
    fn overlay_routes_on(channel: Channel, members: &[Member]) -> Routes {
        overlay_routes_with(channel, members, None)
    }

    /// [`overlay_routes_on`] with member 0's encoding typed `lead_type`.
    fn overlay_routes_with(
        channel: Channel,
        members: &[Member],
        lead_type: Option<crate::spec::encoding::DataType>,
    ) -> Routes {
        use crate::layout::ThemeInputs;
        use crate::render::scale_resolve::resolve_scales_with_leaf_context;

        let rows = members[0].values.len();
        let index: arrow::array::ArrayRef =
            Arc::new(Float64Array::from((0..rows).map(|i| i as f64).collect::<Vec<_>>()));
        let encoding_of = |position: usize, m: &Member| {
            let type_ = if position == 0 { lead_type } else { None };
            let own = Some(EncodingSpec { scale: m.scale.clone(), type_, ..enc(m.field) });
            let mut e = Encoding::default();
            (e.x, e.y) = match channel {
                Channel::X => (own, Some(enc("i"))),
                _ => (Some(enc("i")), own),
            };
            e
        };
        let domain_of = |scales: crate::render::scale_resolve::ResolvedScales| match channel {
            Channel::X => scales.x.data_domain(),
            _ => scales.y.data_domain(),
        };
        let batch_of = |ms: &[&Member]| {
            let mut fields = vec![Field::new("i", DataType::Float64, false)];
            let mut arrays = vec![index.clone()];
            for m in ms {
                fields.push(Field::new(m.field, m.values.data_type().clone(), true));
                arrays.push(m.values.clone());
            }
            RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
        };
        let theme = ThemeInputs::default();
        let pr = (0.0, 100.0);

        let owned_leaves: Vec<LeafOwned> =
            members.iter().enumerate().map(|(i, m)| owned(spec_with(encoding_of(i, m)), batch_of(&[m]))).collect();
        let inputs: Vec<LeafResolveInput<'_>> = owned_leaves.iter().map(input).collect();
        let tree = composite(
            CompositeLayout::Overlay,
            members
                .iter()
                .map(|m| match channel {
                    Channel::X => leaf_node(Some(m.field), Some("i")),
                    _ => leaf_node(Some("i"), Some(m.field)),
                })
                .collect(),
            shared_both(),
        );
        let mut leaf_warnings = Vec::new();
        let leaves = resolve_composite_scales(&tree, &inputs, &mut leaf_warnings)
            .map_err(|e| match e {
                CompositeResolveError::DomainResolve { source, .. } => source,
                other => panic!("unexpected composite error {other:?}"),
            })
            .and_then(|ctx| {
                owned_leaves
                    .iter()
                    .zip(&ctx)
                    .map(|(o, c)| {
                        let (scales, _) =
                            resolve_scales_with_leaf_context(&o.spec, &o.spec.encoding, &o.batch, &o.outputs, pr, pr, &theme, Some(c))?;
                        let adopted = match channel {
                            Channel::X => c.x_scale.is_some(),
                            _ => c.y_scale.is_some(),
                        };
                        Ok((domain_of(scales), adopted))
                    })
                    .collect()
            });

        let layers: Vec<Layer> = members
            .iter()
            .enumerate()
            .map(|(i, m)| Layer {
                mark: Mark::Point,
                encoding: encoding_of(i, m),
                ..Default::default()
            })
            .collect();
        let flat_spec = ChartSpec { encoding: layers[0].encoding.clone(), layers: Some(layers), ..base_spec() };
        let flat_batch = batch_of(&members.iter().collect::<Vec<_>>());
        // Through `prepare_render_inputs`, the one funnel a `+` render takes:
        // its spec-level layer checks run there, before scale resolution.
        let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
        let flat = crate::render::prepare::prepare_render_inputs(
            &flat_spec,
            &flat_batch,
            &theme,
            viewport,
            &crate::render::chart_config::ChartConfig::default(),
            None,
        );
        let mut flat_warnings = Vec::new();
        let flat = flat.map(|prep| {
            flat_warnings = prep.warnings;
            domain_of(prep.provisional_scales)
        });
        let excluded = |ws: Vec<RenderWarning>| {
            ws.into_iter().filter(|w| matches!(w, RenderWarning::LayerScaleExcluded { .. })).collect()
        };
        Routes { leaves, flat, excluded: (excluded(leaf_warnings), excluded(flat_warnings)) }
    }

    fn domainless(padding: Option<f64>) -> serde_json::Value {
        serde_json::json!({"type": "linear", "padding": padding, "nice": false, "zero": false})
    }

    /// Order independence of an overlay under a caller's scale: three point
    /// leaves, one (`v`) carrying the caller's domain-less linear scale and
    /// two (`p`, `q`) declaring none, with the caller leaf first, in the
    /// middle, and last. Every leaf resolves the one axis `(1, 80)` wherever
    /// the caller leaf sits, and that is the domain the flat layered route
    /// resolves for the same layers in the same order.
    #[test]
    fn overlay_caller_scale_join_is_order_independent() {
        let v = || member("v", &[1.0, 2.0, 3.0], Some(domainless(None)));
        let p = || member("p", &[40.0, 60.0, 80.0], None);
        let q = || member("q", &[20.0, 40.0, 60.0], None);
        let orders = [("caller first", [v(), p(), q()]), ("caller middle", [p(), v(), q()]), ("caller last", [p(), q(), v()])];
        let mut failures = Vec::new();
        for (name, members) in orders {
            let (leaves, flat) = overlay_routes(&members).unwrap();
            for (i, (d, _)) in leaves.iter().enumerate() {
                if *d != Some((1.0, 80.0)) {
                    failures.push(format!("{name}: fm.layer leaf {i} resolved {d:?}, want (1, 80)"));
                }
            }
            if leaves[0].0 != flat {
                failures.push(format!("{name}: fm.layer axis {:?}, + axis {flat:?}", leaves[0].0));
            }
        }
        assert!(failures.is_empty(), "{} red cells:\n{}", failures.len(), failures.join("\n"));
    }

    /// A first member whose caller scale is categorical owns the axis: a
    /// later member's domain-less continuous scale does not turn the panel
    /// continuous, so neither route refuses the categorical member's column
    /// and no member adopts the later scale.
    #[test]
    fn overlay_categorical_first_member_owns_the_axis_on_both_routes() {
        let members = [
            Member {
                field: "c",
                values: Arc::new(StringArray::from(vec!["a", "b", "c"])),
                scale: Some(serde_json::from_value(serde_json::json!({"type": "band"})).unwrap()),
            },
            member("n", &[1.0, 2.0, 3.0], Some(domainless(None))),
        ];
        let (leaves, flat) = overlay_routes(&members).unwrap();
        assert_eq!((leaves[0].0, flat), (None, None), "both panel axes stay categorical");
        assert!(!leaves[1].1, "the continuous member keeps its own scale");
    }

    /// A first member stating its own `domain` establishes the axis on both
    /// routes as written: the scale-less member resolves on it, and a later
    /// member's domain-less scale is a different scale that stays out.
    #[test]
    fn overlay_stated_domain_first_member_establishes_the_axis_on_both_routes() {
        let members = [
            member("a", &[1.0, 2.0, 3.0], Some(serde_json::json!({"type": "linear", "domain": [0.0, 10.0]}))),
            member("b", &[40.0, 50.0, 60.0], Some(domainless(None))),
            member("s", &[20.0, 30.0, 40.0], None),
        ];
        let (leaves, flat) = overlay_routes(&members).unwrap();
        assert_eq!(leaves[0].0, Some((0.0, 10.0)), "fm.layer panel axis");
        assert_eq!(flat, Some((0.0, 10.0)), "+ axis");
        assert_eq!(leaves[2], (Some((0.0, 10.0)), true), "the scale-less member resolves on the stated axis");
        assert!(!leaves[1].1, "the different-scale member stays out");

        // Established by a later member, the stated axis is the same.
        let members = [
            member("s", &[20.0, 30.0, 40.0], None),
            member("a", &[1.0, 2.0, 3.0], Some(serde_json::json!({"type": "linear", "domain": [0.0, 10.0]}))),
        ];
        let (leaves, flat) = overlay_routes(&members).unwrap();
        assert_eq!(leaves[0], (Some((0.0, 10.0)), true), "fm.layer panel axis, drawn by the scale-less member");
        assert_eq!(flat, Some((0.0, 10.0)), "+ axis");
    }

    /// The caller's scale on `x`: a scale-less sibling joins it on both
    /// routes exactly as on `y`. The scale's own `zero` makes adopting it
    /// visible in the sibling's domain, `(0, 80)`, where the auto path over
    /// the same union would give `(1, 80)`.
    #[test]
    fn overlay_caller_scale_on_x_joins_a_scale_less_sibling_on_both_routes() {
        let zeroed = serde_json::json!({"type": "linear", "nice": false, "zero": true});
        let members = [member("v", &[1.0, 2.0, 3.0], Some(zeroed)), member("p", &[40.0, 60.0, 80.0], None)];
        let (leaves, flat) = overlay_routes_on(Channel::X, &members).unwrap();
        assert_eq!(leaves, vec![(Some((0.0, 80.0)), true); 2], "fm.layer leaves");
        assert_eq!(flat, Some((0.0, 80.0)), "+ axis");
    }

    /// Which layers each route reports as left out of the axis a caller's
    /// scale establishes: a layer carrying a different scale, or its own
    /// `domain`, warns once, naming itself, the establishing layer and the
    /// channel, with the route's placement (its own scale on `fm.layer`, the
    /// shared axis on `+`). A categorical first scale owns the axis the same
    /// way. The owning layer, a scale-less or anchor-only layer, and an
    /// equal copy do not warn; the equal-copy rows are the controls that must
    /// differ from the different-scale rows.
    #[test]
    fn a_layer_left_out_of_the_caller_scale_warns_once_on_both_routes() {
        let c = || member("c", &[1.0, 2.0, 3.0], Some(domainless(Some(0.2))));
        let s = || member("s", &[10.0, 20.0, 30.0], None);
        let d = || member("d", &[50.0, 60.0, 70.0], Some(domainless(Some(0.6))));
        let anchored = || {
            member("a", &[5.0, 6.0, 7.0], Some(serde_json::json!({"type": "linear", "zero": true, "systemInjected": true})))
        };
        let own_domain =
            || member("o", &[5.0, 6.0, 7.0], Some(serde_json::json!({"type": "linear", "domain": [0.0, 10.0]})));
        let categorical_as = |kind: &str| Member {
            field: "k",
            values: Arc::new(StringArray::from(vec!["a", "b", "c"])),
            scale: Some(serde_json::from_value(serde_json::json!({"type": kind})).unwrap()),
        };
        let categorical = || categorical_as("band");
        // Each shape, its channel, and the (layer, establishing) pair it
        // reports, if any.
        let shapes: Vec<(&str, Channel, Vec<Member>, Option<(usize, usize)>)> = vec![
            ("different scale last", Channel::Y, vec![c(), s(), d()], Some((2, 0))),
            ("caller scale established later", Channel::Y, vec![s(), d(), c()], Some((2, 1))),
            ("own domain", Channel::Y, vec![c(), own_domain()], Some((1, 0))),
            ("different scale on x", Channel::X, vec![c(), d()], Some((1, 0))),
            ("equal copy", Channel::Y, vec![c(), c()], None),
            ("scale-less sibling", Channel::Y, vec![c(), s()], None),
            ("anchor-only sibling", Channel::Y, vec![c(), anchored()], None),
            ("different scale after a categorical first scale", Channel::Y, vec![categorical(), d()], Some((1, 0))),
            ("different categorical scale", Channel::Y, vec![categorical(), categorical_as("point")], Some((1, 0))),
            ("scale-less after a categorical first scale", Channel::Y, vec![categorical(), s()], None),
            ("categorical equal copy", Channel::Y, vec![categorical(), categorical()], None),
        ];
        let mut failures = Vec::new();
        for (name, channel, members, want) in shapes {
            let expect = |placement| {
                want.map(|(layer, establishing)| RenderWarning::LayerScaleExcluded {
                    layer,
                    establishing,
                    channel: channel.as_str().to_string(),
                    placement,
                })
                .into_iter()
                .collect::<Vec<_>>()
            };
            let routes = overlay_routes_on(channel, &members);
            let (composite, flat) = &routes.excluded;
            if *composite != expect(ExcludedLayerPlacement::OwnScale) {
                failures.push(format!("{name}: fm.layer warned {composite:?}"));
            }
            if *flat != expect(ExcludedLayerPlacement::SharedAxis) {
                failures.push(format!("{name}: + warned {flat:?}"));
            }
            routes.unwrap();
        }
        assert!(failures.is_empty(), "{} red cells:\n{}", failures.len(), failures.join("\n"));
    }

    /// A later layer under `CoordFlip` whose own caller y scale lands on the
    /// panel's x axis does not take that axis from layer 0, whose string x
    /// column makes it categorical: on both routes the chart resolves, the
    /// panel x is layer 0's band axis over its categories, and the flipped
    /// layer is reported left out of the x axis once, with the route's
    /// placement. The flipped layer binds a numeric column on each axis, so
    /// its exchanged channels are servable and only the establishing rule is
    /// in question.
    #[test]
    fn a_flipped_layers_caller_scale_does_not_take_a_categorical_layer_0_axis_on_both_routes() {
        use crate::layout::{ThemeInputs, Viewport};
        use crate::render::chart_config::ChartConfig;
        use crate::render::scale_resolve::{resolve_scales_with_leaf_context, ScaleKind};

        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
            Field::new("w", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 5.0])),
                Arc::new(Float64Array::from(vec![2.0, 3.0])),
            ],
        )
        .unwrap();
        let caller: crate::spec::encoding::ScaleSpec =
            serde_json::from_value(serde_json::json!({"type": "linear", "nice": false, "zero": false})).unwrap();
        let encoding = |x: &str, scale: Option<crate::spec::encoding::ScaleSpec>| {
            let mut e = Encoding::default();
            e.x = Some(enc(x));
            e.y = Some(EncodingSpec { scale, ..enc("v") });
            e
        };
        let plain = spec_with(encoding("cat", None));
        let flipped =
            ChartSpec { coord: Some(crate::spec::coord::CoordKind::Flip), ..spec_with(encoding("w", Some(caller))) };
        let excluded = |placement| {
            vec![RenderWarning::LayerScaleExcluded { layer: 1, establishing: 0, channel: "x".to_string(), placement }]
        };
        let only_excluded = |ws: Vec<RenderWarning>| -> Vec<RenderWarning> {
            ws.into_iter().filter(|w| matches!(w, RenderWarning::LayerScaleExcluded { .. })).collect()
        };
        let band_categories = |x: &ScaleKind| match x {
            ScaleKind::Ordinal(_) => x.tick_labels(5),
            other => panic!("panel x is {other:?}, want layer 0's band axis"),
        };
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let prepare = |spec: &ChartSpec| {
            crate::render::prepare::prepare_render_inputs(spec, &batch, &theme, viewport, &ChartConfig::default(), None)
        };

        // `+`: layer 1 carries the flip and the scale, as the lowering writes it.
        let layer = |spec: &ChartSpec| Layer {
            mark: Mark::Point,
            encoding: spec.encoding.clone(),
            coord: spec.coord.clone(),
            ..Default::default()
        };
        let flat = ChartSpec { layers: Some(vec![layer(&plain), layer(&flipped)]), ..plain.clone() };
        let prep = prepare(&flat).expect("+ resolves");
        assert_eq!(band_categories(&prep.provisional_scales.x), ["a", "b"], "+ panel x");
        assert_eq!(only_excluded(prep.warnings), excluded(ExcludedLayerPlacement::SharedAxis), "+");

        // `fm.layer`: each leaf resolves with its own post-flip encoding.
        let leaves: Vec<LeafOwned> = [plain, flipped]
            .into_iter()
            .map(|spec| {
                let encoding = prepare(&spec).unwrap().layers[0].encoding.clone();
                LeafOwned { encoding, outputs: outputs(&batch), batch: batch.clone(), spec }
            })
            .collect();
        let tree = composite(
            CompositeLayout::Overlay,
            vec![leaf_node(Some("cat"), Some("v")), leaf_node(Some("v"), Some("w"))],
            shared_both(),
        );
        let mut warnings = Vec::new();
        let ctx = resolve_composite_scales(&tree, &leaves.iter().map(input).collect::<Vec<_>>(), &mut warnings)
            .expect("fm.layer resolves");
        let pr = (0.0, 100.0);
        let (scales, _) =
            resolve_scales_with_leaf_context(&leaves[0].spec, &leaves[0].spec.encoding, &batch, &leaves[0].outputs, pr, pr, &theme, Some(&ctx[0]))
                .expect("fm.layer panel resolves");
        assert_eq!(band_categories(&scales.x), ["a", "b"], "fm.layer panel x");
        assert_eq!(only_excluded(warnings), excluded(ExcludedLayerPlacement::OwnScale), "fm.layer");
    }

    /// One overlay of `lead` and `member` over `batch` on both layering
    /// routes. `+`: the full scene build of one layered spec whose chart
    /// coord is `lead`'s and whose member layer carries `member_layer_coord`,
    /// the coord `+` stamps on it. `fm.layer`: the overlay's shared-scale
    /// resolution over the two leaves as authored, a composite refusal
    /// unwrapped to the render error it carries.
    fn render_on_both_layering_routes(
        batch: &RecordBatch,
        lead: ChartSpec,
        member: ChartSpec,
        member_layer_coord: Option<crate::spec::coord::CoordKind>,
    ) -> (Result<(), RenderError>, Result<(), RenderError>) {
        use crate::layout::{ThemeInputs, Viewport};
        use crate::render::chart_config::ChartConfig;
        use crate::render::config::RenderConfig;

        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let layer = |spec: &ChartSpec, coord| Layer {
            mark: Mark::Point,
            encoding: spec.encoding.clone(),
            coord,
            ..Default::default()
        };
        let flat_spec =
            ChartSpec { layers: Some(vec![layer(&lead, None), layer(&member, member_layer_coord)]), ..lead.clone() };
        let flat = crate::render::render_svg(
            &flat_spec,
            batch,
            &theme,
            viewport,
            &RenderConfig::default(),
            &ChartConfig::default(),
        )
        .map(|_| ());
        let leaves: Vec<LeafOwned> = [lead, member]
            .into_iter()
            .map(|spec| {
                let encoding =
                    crate::render::prepare::prepare_render_inputs(&spec, batch, &theme, viewport, &ChartConfig::default(), None)
                        .unwrap()
                        .layers[0]
                        .encoding
                        .clone();
                LeafOwned { encoding, outputs: outputs(batch), batch: batch.clone(), spec }
            })
            .collect();
        let field = |e: &Option<EncodingSpec>| e.as_ref().map(|e| e.field.clone());
        let nodes = leaves
            .iter()
            .map(|leaf| leaf_node(field(&leaf.encoding.x).as_deref(), field(&leaf.encoding.y).as_deref()))
            .collect();
        let tree = composite(CompositeLayout::Overlay, nodes, shared_both());
        let layered = resolve_composite_scales(&tree, &leaves.iter().map(input).collect::<Vec<_>>(), &mut Vec::new())
            .map(|_| ())
            .map_err(|e| match e {
                CompositeResolveError::DomainResolve { source, .. } => source,
                other => panic!("unexpected composite error {other:?}"),
            });
        (flat, layered)
    }

    /// A batch with a string `s`, numeric `v` and `w`, and a timestamp `t`.
    fn exchange_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
            Field::new("w", DataType::Float64, false),
            Field::new("t", DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None), false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0])),
                Arc::new(arrow::array::TimestampMillisecondArray::from(vec![0_i64, 1, 2])),
            ],
        )
        .unwrap()
    }

    fn xy_spec(x: &str, y: &str, coord: Option<crate::spec::coord::CoordKind>) -> ChartSpec {
        let mut e = Encoding::default();
        e.x = Some(enc(x));
        e.y = Some(enc(y));
        ChartSpec { coord, ..spec_with(e) }
    }

    fn exchanged_unservable(
        channel: &'static str,
        column: &str,
        dtype: &str,
        axis_kind: &'static str,
    ) -> crate::render::LayerCoordUnplaceableReason {
        crate::render::LayerCoordUnplaceableReason::ExchangedChannelUnservable {
            channel,
            column: column.to_string(),
            dtype: dtype.to_string(),
            axis_kind,
        }
    }

    /// A later layer whose `CoordFlip` lands a column on an axis of layer
    /// 0's that cannot draw it refuses on both routes with the same refusal,
    /// the exchanged channel naming the column and the axis kind it cannot
    /// serve, on either kind of panel axis: a string column on the continuous
    /// y, a timestamp on the categorical x. A flipped layer whose exchanged
    /// columns suit the axes they land on renders on both: numeric columns
    /// (servable on the continuous y and keyable on the categorical x), and
    /// a string column landing on the categorical x, which only a categorical
    /// judgment of that axis accepts.
    #[test]
    fn a_flipped_layers_unservable_exchanged_column_refuses_on_both_routes() {
        use crate::spec::coord::CoordKind;
        let batch = exchange_batch();
        let routes = |x1: &str, y1: &str| {
            render_on_both_layering_routes(
                &batch,
                xy_spec("s", "v", None),
                xy_spec(x1, y1, Some(CoordKind::Flip)),
                Some(CoordKind::Flip),
            )
        };

        for ((x1, y1), channel, column, dtype, axis_kind) in [
            (("s", "v"), "y", "s", "Utf8", "continuous"),
            (("v", "t"), "x", "t", "Timestamp(Millisecond, None)", "categorical"),
        ] {
            let (flat, layered) = routes(x1, y1);
            let flat = flat.expect_err("+ refuses the exchanged column the panel axis cannot draw");
            assert_eq!(layered, Err(flat.clone()), "fm.layer refuses with the + refusal ({axis_kind} {channel})");
            match &flat {
                RenderError::LayerCoordUnplaceable { layer: 1, coord: "fm.CoordFlip", reason } => {
                    assert_eq!(**reason, exchanged_unservable(channel, column, dtype, axis_kind))
                }
                other => panic!("+ refused with {other:?}"),
            }
        }

        for (x1, y1) in [("w", "v"), ("v", "s")] {
            let (flat, layered) = routes(x1, y1);
            assert_eq!(flat, Ok(()), "+ renders a flipped layer x={x1}, y={y1} whose columns suit the panel");
            assert_eq!(layered, Ok(()), "fm.layer renders a flipped layer x={x1}, y={y1} whose columns suit the panel");
        }
    }

    /// The mirror: layer 0 flipped and a later layer declaring no coord. `+`
    /// stamps `fm.CoordCartesian` on that layer, and `fm.layer` names the
    /// same coord for it, so both routes refuse a column the exchange lands
    /// on an axis that cannot draw it with the same refusal, on either kind
    /// of panel axis: a string column on a continuous x, a timestamp on a
    /// categorical x. A later layer whose columns suit the axes they land on
    /// renders on both.
    #[test]
    fn a_coordless_layer_beside_a_flipped_first_layer_refuses_on_both_routes() {
        use crate::spec::coord::CoordKind;
        let batch = exchange_batch();
        let routes = |lead: (&str, &str), member: (&str, &str)| {
            render_on_both_layering_routes(
                &batch,
                xy_spec(lead.0, lead.1, Some(CoordKind::Flip)),
                xy_spec(member.0, member.1, None),
                Some(CoordKind::default()),
            )
        };

        for (lead, member, column, dtype, axis_kind) in [
            (("s", "v"), ("s", "v"), "s", "Utf8", "continuous"),
            (("v", "s"), ("t", "v"), "t", "Timestamp(Millisecond, None)", "categorical"),
        ] {
            let (flat, layered) = routes(lead, member);
            let flat = flat.expect_err("+ refuses the exchanged column the panel axis cannot draw");
            assert_eq!(layered, Err(flat.clone()), "fm.layer refuses with the + refusal ({axis_kind} x)");
            match &flat {
                RenderError::LayerCoordUnplaceable { layer: 1, coord: "fm.CoordCartesian", reason } => {
                    assert_eq!(**reason, exchanged_unservable("x", column, dtype, axis_kind))
                }
                other => panic!("+ refused with {other:?}"),
            }
        }

        let (flat, layered) = routes(("s", "v"), ("v", "s"));
        assert_eq!(flat, Ok(()), "+ renders a coord-less layer whose columns suit the flipped panel");
        assert_eq!(layered, Ok(()), "fm.layer renders a coord-less layer whose columns suit the flipped panel");
    }

    /// A member admitted to a caller scale's axis reads its column as the
    /// continuous extent it draws on, whatever type it annotates: a numeric
    /// column annotated nominal, with no scale of its own, widens the
    /// domain-less caller scale's axis like any other admitted member.
    #[test]
    fn an_admitted_member_annotated_nominal_joins_the_caller_scale_extent() {
        use crate::layout::ThemeInputs;
        use crate::render::scale_resolve::resolve_scales_with_leaf_context;
        use crate::spec::encoding::DataType as SpecType;

        let batch = |field: &str, values: [f64; 2]| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("i", DataType::Float64, false),
                Field::new(field, DataType::Float64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(Float64Array::from(vec![0.0, 1.0])), Arc::new(Float64Array::from(values.to_vec()))],
            )
            .unwrap()
        };
        let leaf = |y: EncodingSpec| {
            let mut e = Encoding::default();
            e.x = Some(enc("i"));
            e.y = Some(y);
            spec_with(e)
        };
        let caller = serde_json::from_value(domainless(None)).unwrap();
        let a = owned(leaf(EncodingSpec { scale: Some(caller), ..enc("v") }), batch("v", [1.0, 3.0]));
        let b = owned(
            leaf(EncodingSpec { type_: Some(SpecType::Nominal), ..enc("p") }),
            batch("p", [40.0, 80.0]),
        );
        let tree = composite(
            CompositeLayout::Overlay,
            vec![leaf_node(Some("i"), Some("v")), leaf_node(Some("i"), Some("p"))],
            shared_both(),
        );
        let ctx = resolve_composite_scales(&tree, &[input(&a), input(&b)], &mut Vec::new()).unwrap();
        let pr = (0.0, 100.0);
        let (scales, _) =
            resolve_scales_with_leaf_context(&a.spec, &a.spec.encoding, &a.batch, &a.outputs, pr, pr, &ThemeInputs::default(), Some(&ctx[0]))
                .unwrap();
        assert_eq!(scales.y.data_domain(), Some((1.0, 80.0)));
    }

    /// Two different domain-less caller scales and a scale-less member, the
    /// caller scale `c` first, in the middle, and last. The first caller
    /// scale in member order establishes the axis on both routes: the
    /// scale-less member joins it and the member carrying the other scale
    /// stays out. With `c` last, `d` is that first scale.
    #[test]
    fn overlay_different_scale_member_stays_out_in_every_order() {
        let c = || member("c", &[1.0, 2.0, 3.0], Some(domainless(Some(0.2))));
        let s = || member("s", &[10.0, 20.0, 30.0], None);
        let d = || member("d", &[50.0, 60.0, 70.0], Some(domainless(Some(0.6))));
        // Each order, the axis it establishes, and the position of the
        // member that stays out.
        let orders = [
            ("c first", [c(), s(), d()], (1.0, 30.0), 2),
            ("c middle", [s(), c(), d()], (1.0, 30.0), 2),
            ("c last", [s(), d(), c()], (10.0, 70.0), 2),
        ];
        let mut failures = Vec::new();
        for (name, members, axis, out) in orders {
            let (leaves, flat) = overlay_routes(&members).unwrap();
            for (i, (d, adopted)) in leaves.iter().enumerate() {
                let joined = i != out;
                if joined && *d != Some(axis) {
                    failures.push(format!("{name}: fm.layer leaf {i} resolved {d:?}, want {axis:?}"));
                }
                if *adopted != joined {
                    failures.push(format!("{name}: fm.layer leaf {i} adopted the group scale: {adopted}"));
                }
            }
            if flat != Some(axis) {
                failures.push(format!("{name}: + axis {flat:?}, want {axis:?}"));
            }
        }
        assert!(failures.is_empty(), "{} red cells:\n{}", failures.len(), failures.join("\n"));
    }

    /// Layer 0 decides its own axis when its column there reads categorical:
    /// a later layer's continuous caller scale establishes nothing, so neither
    /// route refuses layer 0's column, the panel axis stays categorical, and
    /// the later layer is reported left out once, naming layer 0 as the
    /// establishing layer, with the route's placement. Layer 0 reads
    /// categorical through a `Utf8` column, or through a numeric column it
    /// types nominal. The controls, layer 0 numeric and scale-less beside the
    /// same later scale, keep the rule as it was: the later scale establishes
    /// the axis, layer 0 joins it, and nothing warns.
    #[test]
    fn a_later_continuous_caller_scale_does_not_take_a_categorical_layer_0_axis_on_both_routes() {
        use crate::spec::encoding::DataType as SpecType;

        let utf8 = || Member { field: "s", values: Arc::new(StringArray::from(vec!["a", "b", "c"])), scale: None };
        let later = || member("n", &[1.0, 2.0, 3.0], Some(domainless(None)));
        let numeric = || member("p", &[40.0, 60.0, 80.0], None);
        let mut failures = Vec::new();
        for channel in [Channel::X, Channel::Y] {
            let excluded = |placement| {
                vec![RenderWarning::LayerScaleExcluded {
                    layer: 1,
                    establishing: 0,
                    channel: channel.as_str().to_string(),
                    placement,
                }]
            };
            for (name, routes) in [
                ("Utf8 layer 0", overlay_routes_on(channel, &[utf8(), later()])),
                ("nominal layer 0", overlay_routes_with(channel, &[numeric(), later()], Some(SpecType::Nominal))),
            ] {
                let name = format!("{name} on {}", channel.as_str());
                if routes.excluded != (excluded(ExcludedLayerPlacement::OwnScale), excluded(ExcludedLayerPlacement::SharedAxis)) {
                    failures.push(format!("{name}: warned {:?}", routes.excluded));
                }
                match (&routes.leaves, &routes.flat) {
                    (Ok(leaves), Ok(flat)) => {
                        if (leaves[0].0, *flat) != (None, None) {
                            failures.push(format!("{name}: panel axes {:?} / {flat:?}, want categorical", leaves[0].0));
                        }
                        if leaves[1].1 {
                            failures.push(format!("{name}: the later leaf adopted a group scale"));
                        }
                    }
                    (leaves, flat) => failures.push(format!(
                        "{name}: fm.layer {:?}, + {:?}",
                        leaves.as_ref().err(),
                        flat.as_ref().err()
                    )),
                }
            }

            // The control: layer 0 numeric and scale-less.
            let routes = overlay_routes_on(channel, &[numeric(), later()]);
            let name = format!("numeric layer 0 on {}", channel.as_str());
            if routes.excluded != (Vec::new(), Vec::new()) {
                failures.push(format!("{name}: warned {:?}", routes.excluded));
            }
            match (routes.leaves, routes.flat) {
                (Ok(leaves), Ok(flat)) => {
                    if leaves != vec![(Some((1.0, 80.0)), true); 2] || flat != Some((1.0, 80.0)) {
                        failures.push(format!("{name}: fm.layer {leaves:?}, + {flat:?}, want the joined (1, 80)"));
                    }
                }
                (leaves, flat) => failures.push(format!("{name}: fm.layer {:?}, + {:?}", leaves.err(), flat.err())),
            }

            // A layer 0 typed nominal that carries its own continuous scale
            // does not decide a categorical axis: its scale establishes the
            // axis, the later equal copy joins it, and nothing warns.
            let scaled_nominal = || member("p", &[40.0, 60.0, 80.0], Some(domainless(None)));
            let routes = overlay_routes_with(channel, &[scaled_nominal(), later()], Some(SpecType::Nominal));
            let name = format!("nominal layer 0 with its own scale on {}", channel.as_str());
            if routes.excluded != (Vec::new(), Vec::new()) {
                failures.push(format!("{name}: warned {:?}", routes.excluded));
            }
            match (routes.leaves, routes.flat) {
                (Ok(leaves), Ok(flat)) => {
                    if leaves != vec![(Some((1.0, 80.0)), true); 2] || flat != Some((1.0, 80.0)) {
                        failures.push(format!("{name}: fm.layer {leaves:?}, + {flat:?}, want the joined (1, 80)"));
                    }
                }
                (leaves, flat) => failures.push(format!("{name}: fm.layer {:?}, + {:?}", leaves.err(), flat.err())),
            }
        }
        assert!(failures.is_empty(), "{} red cells:\n{}", failures.len(), failures.join("\n"));
    }

    /// A caller scale stating its own `domain` needs no data union, and a
    /// `Utf8` member of the group it establishes still refuses in the
    /// shared-axis voice on both routes, with the establishing member first
    /// and after a scale-less member.
    #[test]
    fn stated_domain_caller_scale_refuses_a_utf8_member_on_both_routes() {
        let stated = || member("n", &[1.0, 2.0, 3.0], Some(serde_json::json!({"type": "linear", "domain": [0.0, 10.0]})));
        let utf8 = || Member { field: "s", values: Arc::new(StringArray::from(vec!["a", "b", "c"])), scale: None };
        let orders = [
            ("stated first", vec![stated(), utf8()], 1),
            ("stated after a scale-less member", vec![member("p", &[4.0, 5.0, 6.0], None), stated(), utf8()], 2),
        ];
        let mut failures = Vec::new();
        for (name, members, layer) in orders {
            let want = RenderError::LayerAxisColumnUnplaceable {
                layer,
                detail: Box::new(crate::render::LayerAxisColumnDetail {
                    channel: "y".to_string(),
                    field: "s".to_string(),
                    dtype: "Utf8".to_string(),
                }),
            };
            let routes = overlay_routes(&members);
            if routes.leaves.as_ref().err() != Some(&want) {
                failures.push(format!("{name}: fm.layer {:?}", routes.leaves.map(|_| "rendered")));
            }
            if routes.flat.as_ref().err() != Some(&want) {
                failures.push(format!("{name}: + {:?}", routes.flat.map(|_| "rendered")));
            }
        }
        assert!(failures.is_empty(), "{} red cells:\n{}", failures.len(), failures.join("\n"));
    }

    /// A `Utf8` member carrying its own categorical scale beside a
    /// continuous establishing scale still refuses in the shared-axis
    /// voice on both routes, naming the `Utf8` member's own position,
    /// whether the establishing scale is domain-less or states its own
    /// `domain`.
    #[test]
    fn own_scale_utf8_member_refuses_beside_a_continuous_establishing_scale_on_both_routes() {
        let utf8_band = || Member {
            field: "s",
            values: Arc::new(StringArray::from(vec!["a", "b", "c"])),
            scale: Some(serde_json::from_value(serde_json::json!({"type": "band"})).unwrap()),
        };
        let cases = [
            ("domain-less establishing scale", vec![member("n", &[1.0, 2.0, 3.0], Some(domainless(None))), utf8_band()]),
            (
                "stated-domain establishing scale",
                vec![member("n", &[1.0, 2.0, 3.0], Some(serde_json::json!({"type": "linear", "domain": [0.0, 10.0]}))), utf8_band()],
            ),
        ];
        let mut failures = Vec::new();
        for (name, members) in cases {
            let want = RenderError::LayerAxisColumnUnplaceable {
                layer: 1,
                detail: Box::new(crate::render::LayerAxisColumnDetail {
                    channel: "y".to_string(),
                    field: "s".to_string(),
                    dtype: "Utf8".to_string(),
                }),
            };
            let routes = overlay_routes(&members);
            if routes.leaves.as_ref().err() != Some(&want) {
                failures.push(format!("{name}: fm.layer {:?}", routes.leaves.map(|_| "rendered")));
            }
            if routes.flat.as_ref().err() != Some(&want) {
                failures.push(format!("{name}: + {:?}", routes.flat.map(|_| "rendered")));
            }
        }
        assert!(failures.is_empty(), "{} red cells:\n{}", failures.len(), failures.join("\n"));
    }

    /// On `+`, a layer's `y2` endpoint on an axis a domain-less caller scale
    /// establishes is read into the axis's extent, and a `Utf8` endpoint
    /// refuses in the shared-axis voice naming that layer. The spec-level
    /// gate judges value columns only, so the union's read guard is the one
    /// owner of this refusal.
    #[test]
    fn caller_scale_union_refuses_a_utf8_paired_endpoint_on_the_flat_route() {
        let layer = |mark: Mark, encoding: Encoding| Layer {
            mark,
            encoding,
            ..Default::default()
        };
        let scaled = Encoding {
            x: Some(enc("i")),
            y: Some(EncodingSpec { scale: Some(serde_json::from_value(domainless(None)).unwrap()), ..enc("v") }),
            ..Encoding::default()
        };
        let paired = Encoding { x: Some(enc("i")), y: Some(enc("p")), y2: Some(enc("s")), ..Encoding::default() };
        let spec = ChartSpec {
            encoding: scaled.clone(),
            layers: Some(vec![layer(Mark::Point, scaled), layer(Mark::Rule, paired)]),
            ..base_spec()
        };
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("i", DataType::Float64, false),
                Field::new("v", DataType::Float64, false),
                Field::new("p", DataType::Float64, false),
                Field::new("s", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();
        let err = crate::render::prepare::prepare_render_inputs(
            &spec,
            &batch,
            &crate::layout::ThemeInputs::default(),
            crate::layout::Viewport { width: 600.0, height: 400.0 },
            &crate::render::chart_config::ChartConfig::default(),
            None,
        )
        .err()
        .expect("+ refuses");
        assert_eq!(
            err,
            RenderError::LayerAxisColumnUnplaceable {
                layer: 1,
                detail: Box::new(crate::render::LayerAxisColumnDetail {
                    channel: "y".to_string(),
                    field: "s".to_string(),
                    dtype: "Utf8".to_string(),
                }),
            }
        );
    }

    // -- per-layer domain source ------------------------------------------------

    /// A layered overlay leaf contributes each layer's data from the batch
    /// that layer draws. The leaf's chart data carries a same-named column
    /// far outside it, which the rejected reading would take in.
    #[test]
    fn an_overlay_leaf_contributes_each_layers_own_batch() {
        let spec: ChartSpec = serde_json::from_str(
            r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"ci_lower"},"y2":{"field":"ci_upper"}},"layers":[{"mark":"ribbon","encoding":{"x":{"field":"x"},"y":{"field":"ci_lower"},"y2":{"field":"ci_upper"}},"data_source":"smooth"},{"mark":"line","encoding":{"x":{"field":"x"},"y":{"field":"y"}},"data_source":"smooth"}]}"#,
        )
        .expect("layered spec JSON must deserialize");
        let f64_col = |v: &[f64]| -> arrow::array::ArrayRef { Arc::new(Float64Array::from(v.to_vec())) };
        let batch_of = |cols: Vec<(&str, arrow::array::ArrayRef)>| {
            let fields: Vec<Field> =
                cols.iter().map(|(n, c)| Field::new(*n, c.data_type().clone(), true)).collect();
            RecordBatch::try_new(Arc::new(Schema::new(fields)), cols.into_iter().map(|(_, c)| c).collect())
                .unwrap()
        };
        let raw = batch_of(vec![("x", f64_col(&[0.0, 5.0, 10.0])), ("y", f64_col(&[-4.15, 10.0, 22.0]))]);
        let smooth = batch_of(vec![
            ("x", f64_col(&[0.0, 5.0, 10.0])),
            ("y", f64_col(&[0.5, 10.0, 19.5])),
            ("ci_lower", f64_col(&[-0.2, 9.0, 18.0])),
            ("ci_upper", f64_col(&[1.0, 11.0, 20.0])),
        ]);
        let mut leaf = owned(spec, raw);
        leaf.outputs.insert("smooth".to_string(), smooth);
        let domain = leaf_positional_domain(&input(&leaf), Channel::Y, LayerUnion::Every)
            .expect("every column is Float64");
        assert_eq!(domain, Some(SharedDomain::Numeric { lo: -0.2, hi: 20.0 }));
    }

    /// The ordinal overlay union: each operand's categories from its own rows,
    /// and no `null` band for the merged frame's padding.
    #[test]
    fn an_overlay_leaf_contributes_each_layers_own_categories() {
        let spec: ChartSpec = serde_json::from_str(
            r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"g","type":"nominal"},"y":{"field":"v"}},"layers":[{"mark":"point","encoding":{"x":{"field":"g","type":"nominal"},"y":{"field":"v"}},"data_source":"lhs"},{"mark":"point","encoding":{"x":{"field":"g","type":"nominal"},"y":{"field":"v"}},"data_source":"rhs"}]}"#,
        )
        .expect("layered spec JSON must deserialize");
        let batch_of = |g: Vec<Option<&str>>, v: Vec<f64>| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, true),
                Field::new("v", DataType::Float64, true),
            ]));
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(g)), Arc::new(Float64Array::from(v))])
                .unwrap()
        };
        let merged = batch_of(vec![Some("a"), Some("b"), None, None], vec![1.0, 2.0, 3.0, 4.0]);
        let mut leaf = owned(spec, merged);
        leaf.outputs.insert("lhs".to_string(), batch_of(vec![Some("a"), Some("b")], vec![1.0, 2.0]));
        leaf.outputs.insert("rhs".to_string(), batch_of(vec![Some("x"), Some("y")], vec![3.0, 4.0]));
        let domain = leaf_positional_domain(&input(&leaf), Channel::X, LayerUnion::Every)
            .expect("every category column is Utf8");
        let expected: Vec<String> = ["a", "b", "x", "y"].iter().map(|s| s.to_string()).collect();
        assert_eq!(domain, Some(SharedDomain::Ordinal(expected)));
    }

    /// The other live `+` lowering on the composite path: a chainless left
    /// operand on the chart's data and the right operand on the merged
    /// frame's `_ident_0` under renamed fields. The ladder is the left
    /// categories then the right, with no band for the merge padding.
    #[test]
    fn an_overlay_leaf_contributes_chart_and_output_layer_categories() {
        let spec: ChartSpec = serde_json::from_str(
            r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}},"layers":[{"mark":"point","encoding":{"x":{"field":"cat","type":"nominal"},"y":{"field":"val"}}},{"mark":"point","encoding":{"x":{"field":"cat__rhs_0001","type":"nominal"},"y":{"field":"val__rhs_0001"}},"data_source":"_ident_0"}]}"#,
        )
        .expect("layered spec JSON must deserialize");
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, true),
            Field::new("val", DataType::Float64, true),
            Field::new("cat__rhs_0001", DataType::Utf8, true),
            Field::new("val__rhs_0001", DataType::Float64, true),
        ]));
        let merged = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("c"), Some("a"), None])),
                Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), None])),
                Arc::new(StringArray::from(vec![None, None, Some("b")])),
                Arc::new(Float64Array::from(vec![None, None, Some(40.0)])),
            ],
        )
        .unwrap();
        let mut leaf = owned(spec, merged.clone());
        leaf.outputs.insert("_ident_0".to_string(), merged);
        let domain = leaf_positional_domain(&input(&leaf), Channel::X, LayerUnion::Every)
            .expect("every category column is Utf8");
        let expected: Vec<String> = ["c", "a", "b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(domain, Some(SharedDomain::Ordinal(expected)));
    }
}
