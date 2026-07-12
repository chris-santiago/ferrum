//! Composite scale-resolution pass — Phase B composite-render-unification (Task 4).
//!
//! This is pass 1 of the three-pass composite renderer (design spec §5). It
//! walks a [`CompositeNode`] tree and computes, per shared positional channel,
//! one resolved domain across the tree's leaves — through the *same* mechanism
//! facets use ([`numeric_domain_union`] / [`distinct_positional_categories_shared`]),
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
    distinct_positional_categories_shared, distinct_values_in_order, infer_spec_type, locate_field,
    numeric_domain_union, numeric_extent, Channel, LeafScaleContext, SharedDomain,
};
use crate::render::RenderError;
use crate::spec::chart::ChartSpec;
use crate::spec::composite::CompositeNode;
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
    UnknownField {
        channel: &'static str,
        field: String,
    },
    /// The underlying scale resolver failed to union a leaf's channel domain.
    DomainResolve {
        channel: &'static str,
        field: String,
        source: RenderError,
    },
}

impl fmt::Display for CompositeResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LeafCountMismatch { kind, expected, got } => write!(
                f,
                "{kind}: composite leaf count mismatch: tree has {expected} leaves, got {got} prepared inputs"
            ),
            Self::UnknownField { channel, field } => write!(
                f,
                "shared {channel} channel references unknown field '{field}'"
            ),
            Self::DomainResolve { channel, field, source } => write!(
                f,
                "failed to resolve shared {channel} domain for field '{field}': {source}"
            ),
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
pub(crate) fn resolve_composite_scales(
    tree: &CompositeNode,
    leaves: &[LeafResolveInput<'_>],
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
        resolve_channel(tree, &span, channel, leaves, &mut out)?;
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
) -> Result<(), CompositeResolveError> {
    let CompositeNode::Composite { children, resolve, .. } = node else {
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
            resolve_group(&group, channel, leaves, out)?;
        }
        Ok(())
    } else {
        // Independent node, or a non-congruent Shared node (skip sharing for the
        // group): fall back to per-child resolution so inner Shared nodes work.
        for (child, span) in children.iter().zip(&subspans) {
            resolve_channel(child, span, channel, leaves, out)?;
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
/// attaches at exactly this outermost effective-shared node (see that walk's
/// `descend_channel`, which computes the identical `eff`/`inherited` gate).
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
    let eff = channel_resolve(resolve, channel).unwrap_or(inherited);

    if eff == ResolveMode::Shared && inherited != ResolveMode::Shared {
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
    if domains.is_empty() {
        return Ok(());
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
        for (_, d) in &domains {
            if let SharedDomain::Numeric { lo: l, hi: h } = d {
                lo = lo.min(*l);
                hi = hi.max(*h);
            }
        }
        SharedDomain::Numeric { lo, hi }
    } else if all_ordinal {
        let mut cats: Vec<String> = Vec::new();
        for (_, d) in &domains {
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
        // same skip-not-error philosophy applies here: there is no
        // semantically right domain for mismatched types any more than
        // there is for mismatched tree shapes, and erroring would reject a
        // hand-built heterogeneous concat that renders fine standalone.
        // Skip sharing for this group; each participating leaf falls back to
        // its own standalone resolution (rendering is otherwise normal).
        // Reachable only via a hand-built tree with heterogeneous leaf dtypes
        // on the same shared channel — `compare=` producers always emit
        // uniform dtypes, so this is a defensive fallback, not a documented
        // user-facing feature.
        return Ok(());
    };

    for (idx, _) in &domains {
        out[*idx].set(channel, unified.clone());
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
        Channel::X | Channel::Y => leaf_positional_domain(leaf, channel),
        Channel::Color => leaf_color_domain(leaf),
        Channel::Size => leaf_size_domain(leaf),
    }
}

/// One leaf's standalone positional (`x`/`y`) domain, or `None` if it does not
/// participate in sharing on that channel.
///
/// Non-participation (returns `None`):
/// - the channel has no encoding on this leaf;
/// - the channel carries an explicit user `enc.scale` (user scale wins, spec §6);
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
) -> Result<Option<SharedDomain>, CompositeResolveError> {
    let enc = match channel {
        Channel::X => leaf.encoding.x.as_ref(),
        Channel::Y => leaf.encoding.y.as_ref(),
        _ => unreachable!("leaf_positional_domain called with non-positional channel"),
    };
    let Some(enc) = enc else { return Ok(None) };

    // Explicit user scale bypasses sharing (must remain distinguishable so it
    // still short-circuits at the existing explicit-scale path — D4b).
    if enc.scale.is_some() {
        return Ok(None);
    }

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

    let located = locate_field(&enc.field, leaf.final_batch, leaf.transform_outputs)
        .ok_or_else(|| CompositeResolveError::UnknownField {
            channel: channel.as_str(),
            field: enc.field.clone(),
        })?;
    let dtype = infer_spec_type(enc, located.col.data_type());

    let paired = match channel {
        Channel::X => leaf.encoding.x2.as_ref(),
        Channel::Y => leaf.encoding.y2.as_ref(),
        _ => None,
    }
    .map(|e| e.field.as_str());

    match dtype {
        SpecDataType::Quantitative | SpecDataType::Temporal => {
            let (lo, hi) = numeric_domain_union(
                channel.as_str(),
                &enc.field,
                paired,
                leaf.final_batch,
                leaf.transform_outputs,
                leaf.spec,
                false,
            )
            .map_err(|source| CompositeResolveError::DomainResolve {
                channel: channel.as_str(),
                field: enc.field.clone(),
                source,
            })?;
            Ok(Some(SharedDomain::Numeric { lo, hi }))
        }
        SpecDataType::Ordinal | SpecDataType::Nominal => {
            let cats = distinct_positional_categories_shared(
                leaf.final_batch,
                &enc.field,
                leaf.transform_outputs,
                false,
            )
            .map_err(|source| CompositeResolveError::DomainResolve {
                channel: channel.as_str(),
                field: enc.field.clone(),
                source,
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
    use crate::spec::composite::{CompositeLayout, CompositeResolve};
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::EncodingSpec;
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
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding::default(),
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let err = resolve_composite_scales(&tree, &inputs).unwrap_err();
        match err {
            CompositeResolveError::UnknownField { channel, field } => {
                assert_eq!(channel, "x");
                assert_eq!(field, "v");
            }
            other => panic!("expected UnknownField, got {other:?}"),
        }
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        };
        let msg = err.to_string();
        assert!(msg.contains('y') && msg.contains("score") && msg.contains("boom"), "{msg}");
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
            },
            nice: false,
            zero: false,
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
            },
            nice: false,
            zero: false,
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
        // Leaf 0 is the only participant → shares only its own extent [0, 10].
        assert_eq!(ctx[0].x, Some(SharedDomain::Numeric { lo: 0.0, hi: 10.0 }));
        // Leaf 1's explicit scale wins → no shared context injected.
        assert_eq!(ctx[1].x, None);
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let err = resolve_composite_scales(&tree, &inputs).unwrap_err();
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
        let ctx = resolve_composite_scales(&tree, &inputs).unwrap();
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
        let layer = Layer {
            mark: Mark::Bar,
            encoding: layer_enc,
            transforms: Vec::new(),
            mark_style: None,
            data_source: Some("box".into()),
            position: None,
            blend: None,
            name: None,
            independent_y: false,
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
}
