//! Composite spec tree — Phase B composite-render-unification (Task 2).
//!
//! See `design-docs/superpowers/specs/2026-07-02-composite-render-unification-design.md`
//! §6 for the canonical contract. A composite spec tree is self-contained: it
//! crosses the PyO3 boundary once per render call, leaves carry a
//! [`ChartSpec`] plus an index into the caller's Arrow payload list (no data
//! of their own), and composite nodes carry layout/resolve fields but never
//! data. Types only in this module — no renderer; Task 4/5 consumes
//! [`CompositeNode`] to build the composite `SceneGraph`.
//!
//! # Wire shape and a deliberate naming deviation from the design doc
//!
//! The design doc's tree grammar (§6) names the per-composite-node layout
//! selector `kind: hconcat | vconcat | grid | wrap | overlay` and separately
//! describes two shapes, `Leaf` and `Composite`. Taken literally that would
//! put two different concepts named `kind` at two nesting levels of the same
//! object, which collides with this crate's tagged-enum convention (every
//! tagged spec enum discriminates on a field literally named `"kind"`; see
//! `spec/mod.rs`). This module keeps the convention for the *shape*
//! discriminant — [`CompositeNode`] tags on `"kind"` with three values,
//! `"leaf"`, `"composite"`, and `"hole"` — and renames the *layout* selector
//! to `"layout"` on the `Composite` variant. The design doc's contract note
//! ("exact field names/types are plan-level") licenses this; flagged here
//! for Task 4/5 and the orchestrator in case downstream Python-side dict
//! construction assumed the literal `kind` nesting.
//!
//! `hole` (Task 8a) is a third, field-less shape added after the design doc
//! was written: a placeholder cell in a `grid`/`wrap` layout that reserves
//! its row/column slot but renders no panel/label/chrome — the wire covering
//! for JointChart's empty 2x2 corner and RepeatChart's `corner=True`.
//!
//! ```text
//! {"kind": "leaf", "spec": <ChartSpec>, "data": 0}
//! {"kind": "composite", "layout": "grid", "children": [...], "nrows": 2, "ncols": 2,
//!  "extent_alignments": [{"follower": 0, "leader": 2, "axis": "x"}], ...}
//! {"kind": "hole"}
//! ```
//!
//! The tree types + validation are consumed by the composite render core
//! ([`crate::render::composite_render`], Task 5b) and the resolve pass; the PyO3
//! coercion ([`composite_tree_from_py`]) is consumed by the
//! `render_composite_svg`/`render_composite_interactive` entries (Task 5c).
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

use crate::layout::facet::ResolveMode;
use crate::spec::chart::ChartSpec;

// ---------------------------------------------------------------------------
// CompositeLayout
// ---------------------------------------------------------------------------

/// The five composition forms a `Composite` node can lay out its children
/// with (spec §1: hconcat/vconcat/concat-grid, joint/clustermap/repeat lower
/// to `Grid`/`Wrap`, layer overlay lowers to `Overlay`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompositeLayout {
    Hconcat,
    Vconcat,
    Grid,
    Wrap,
    Overlay,
}

impl CompositeLayout {
    /// The layout kind as used in error messages and the wire value.
    pub fn as_str(&self) -> &'static str {
        match self {
            CompositeLayout::Hconcat => "hconcat",
            CompositeLayout::Vconcat => "vconcat",
            CompositeLayout::Grid => "grid",
            CompositeLayout::Wrap => "wrap",
            CompositeLayout::Overlay => "overlay",
        }
    }
}

/// Error returned by [`CompositeLayout::from_str`] for an unrecognized
/// layout string. Mirrors [`crate::spec::mark::ParseMarkError`]'s shape —
/// same "unknown X; expected one of [...]" idiom.
#[derive(Debug)]
pub struct ParseCompositeLayoutError(pub String);

impl fmt::Display for ParseCompositeLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown composite layout '{}'; expected one of [hconcat, vconcat, grid, wrap, overlay]",
            self.0
        )
    }
}

impl std::error::Error for ParseCompositeLayoutError {}

impl FromStr for CompositeLayout {
    type Err = ParseCompositeLayoutError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "hconcat" => Ok(CompositeLayout::Hconcat),
            "vconcat" => Ok(CompositeLayout::Vconcat),
            "grid" => Ok(CompositeLayout::Grid),
            "wrap" => Ok(CompositeLayout::Wrap),
            "overlay" => Ok(CompositeLayout::Overlay),
            other => Err(ParseCompositeLayoutError(other.to_string())),
        }
    }
}

// ---------------------------------------------------------------------------
// ExtentAlignment
// ---------------------------------------------------------------------------

/// The axis an [`ExtentAlignment`] equalizes. `X` makes two cells in the same
/// grid COLUMN span the same pixels horizontally; `Y` does the same for two
/// cells in one ROW.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlignAxis {
    X,
    Y,
}

impl AlignAxis {
    /// The axis as it appears on the wire — and, for that reason, as it
    /// appears in the warning a caller reads back
    /// (`RenderWarning::ExtentAlignmentUnresolvable`): the token it is told
    /// about is the token it wrote.
    pub fn wire_name(&self) -> &'static str {
        match self {
            AlignAxis::X => "x",
            AlignAxis::Y => "y",
        }
    }
}

/// One grid cell whose plot extent follows another cell's on one axis —
/// **the composition declaring, on the wire, that two of its panels are meant
/// to read against the same ruler** (`#182` follow-up: a jointplot's marginals
/// span the joint panel's axis; a clustermap's dendrograms span their
/// heatmap's).
///
/// `follower` and `leader` are **cell positions** in the grid's own
/// `children` (row-major, so they count holes), not leaf indices: the
/// declaration is local to the node that makes it and stays correct when that
/// node is nested inside another composition.
/// [`crate::render::composite_render`] resolves them to leaf indices at render
/// time.
///
/// Why a declaration and not a shape predicate: the first implementation of
/// this alignment recognized a jointplot by its grid geometry (2x2, hole in
/// the top-right corner, axis-free corner cells). That predicate cannot be
/// made safe, because a clustermap is *also* a 2x2 grid with a hole and
/// axis-free cells — it distinguishes two different figures by which corner
/// the hole is in. This wave took the same decision for the area baseline
/// (`mark_style.value_axis`, `9c570342`) and deleted the geometric heuristic
/// that preceded it; carrying a second heuristic here would leave the
/// codebase holding two answers to one question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtentAlignment {
    /// Cell position (in the grid's `children`) of the panel that adopts the
    /// other's extent.
    pub follower: usize,
    /// Cell position of the panel whose extent is adopted.
    pub leader: usize,
    /// The axis on which the two must agree. The follower keeps its own
    /// extent on the other axis.
    pub axis: AlignAxis,
}

// ---------------------------------------------------------------------------
// CompositeResolve
// ---------------------------------------------------------------------------

/// Per-channel scale resolution policy for a composite node (spec §6).
///
/// Reuses [`ResolveMode`] (the same enum facets use) rather than duplicating
/// it — the *default* differs by design: a facet's `FacetResolve` defaults
/// to `Shared` (today's facet behavior), while a composite node defaults to
/// `Independent` (today's composition behavior, spec §6 "default
/// independent"). Keeping the default on this wrapper struct instead of on
/// `ResolveMode` itself preserves `ResolveMode::default()` for the facet
/// call sites that still depend on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositeResolve {
    #[serde(default = "default_independent")]
    pub x: ResolveMode,
    #[serde(default = "default_independent")]
    pub y: ResolveMode,
    /// Color scale sharing (10-pre-b). Continuous color unions a `[lo, hi]`
    /// extent; categorical color unions an order-preserving category vector.
    ///
    /// Unlike the positional `x`/`y` channels, color/size resolution is
    /// `Option`: `None` is *unset* (inherit the nearest ancestor's effective
    /// mode — spec §6 nested-resolve rule, GH #74), distinct from
    /// `Some(Independent)` (an *explicit* opt-out that wins over an ancestor's
    /// shared mode). Positional channels stay node-local (no inheritance —
    /// grids pair by tree path), so they need no such distinction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<ResolveMode>,
    /// Size scale sharing (10-pre-b): unions the numeric `[min, max]` extent.
    /// `Option` with the same unset-inherits / explicit-wins semantics as
    /// [`CompositeResolve::color`] (spec §6, GH #74).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<ResolveMode>,
    /// Legend resolution override for `color`/`size` (shared-legend design
    /// spec §6 "wire contract"). Absent (both channels `None`, the common
    /// case) means *follow the scale mode* for both channels; the figure-legend
    /// band planner resolves the effective legend mode against the *inherited*
    /// effective scale mode (`render::composite_render::descend_channel`, spec
    /// §6 nested-resolve rule).
    #[serde(default, skip_serializing_if = "CompositeLegendResolve::is_default")]
    pub legend: CompositeLegendResolve,
}

fn default_independent() -> ResolveMode {
    ResolveMode::Independent
}

impl Default for CompositeResolve {
    fn default() -> Self {
        CompositeResolve {
            x: ResolveMode::Independent,
            y: ResolveMode::Independent,
            color: None,
            size: None,
            legend: CompositeLegendResolve::default(),
        }
    }
}

impl CompositeResolve {
    /// `true` when every scale channel is `Independent` and no legend
    /// override is set — the `skip_serializing_if` predicate that keeps the
    /// common (independent) case out of canonical JSON, mirroring
    /// `FacetResolve::is_all_shared`.
    pub fn is_default(r: &CompositeResolve) -> bool {
        r.x == ResolveMode::Independent
            && r.y == ResolveMode::Independent
            && r.color.is_none()
            && r.size.is_none()
            && CompositeLegendResolve::is_default(&r.legend)
    }

}

/// Legend resolution override for the `color`/`size` channels (shared-legend
/// design spec §6 "wire contract"). Each field is `None` when unset — the
/// common case, meaning "follow the channel's scale resolution mode" — and
/// `Some(mode)` when the caller explicitly overrides legend resolution
/// independently of scale resolution (e.g. a shared color scale rendered
/// with per-panel legends via `legend={"color": "independent"}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CompositeLegendResolve {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<ResolveMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<ResolveMode>,
}

impl CompositeLegendResolve {
    /// `true` when both channels are unset (follow scale) — the
    /// `skip_serializing_if` predicate that keeps the common case out of
    /// canonical JSON.
    pub fn is_default(r: &CompositeLegendResolve) -> bool {
        r.color.is_none() && r.size.is_none()
    }
}

// ---------------------------------------------------------------------------
// CompositeNode
// ---------------------------------------------------------------------------

/// One node in a composite render tree (spec §6). Tagged on `"kind"` per the
/// crate convention (`spec/mod.rs`), with three shapes:
///
/// - `Leaf`: a single chart. `spec` is the compiled `ChartSpec`; `data` is
///   an index into the render call's Arrow payload list (leaves carry no
///   data of their own — spec §6 "leaves carry no layout, nodes carry no
///   data").
/// - `Composite`: a composition of `children`, laid out per `layout`.
///   `title`/`subtitle`/`caption`/`config` are root-only figure chrome
///   (spec §6); [`CompositeNode::validate`] rejects them on non-root nodes.
/// - `Hole` (Task 8a): a field-less placeholder cell, valid only as a direct
///   child of a `grid`/`wrap` `Composite` (never at the tree root). Covers
///   JointChart's empty 2x2 corner and RepeatChart's `corner=True`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum CompositeNode {
    Leaf {
        /// Boxed so `CompositeNode` (recursed into via `Vec<CompositeNode>`
        /// on every `Composite` node) doesn't inflate to `ChartSpec`'s full
        /// inline size (clippy `large_enum_variant`); serde is transparent
        /// through `Box<T>` so the wire shape is unaffected.
        spec: Box<ChartSpec>,
        data: usize,
        /// Optional per-child panel label (spec §6, Task 5d). Rendered as a
        /// bold header band at the child's placement origin. Valid on any
        /// non-root node; rejected at the root (the root uses `title`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    Composite {
        layout: CompositeLayout,
        children: Vec<CompositeNode>,
        /// Optional per-child panel label (spec §6, Task 5d). Rendered as a
        /// bold header band at the child's placement origin. Valid on any
        /// non-root node; rejected at the root (the root uses `title`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        /// Per-channel scale resolution. Omitted (or both channels
        /// `independent`) is skipped from canonical JSON.
        #[serde(default, skip_serializing_if = "CompositeResolve::is_default")]
        resolve: CompositeResolve,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        spacing: Option<f64>,
        /// Grid-only: relative row heights. Length must equal `nrows`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        row_ratios: Option<Vec<f64>>,
        /// Grid-only: relative column widths. Length must equal `ncols`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        col_ratios: Option<Vec<f64>>,
        /// Grid/Wrap-only: column count. Required for both; for `Wrap`,
        /// `nrows` is derived from `children.len()` (facet-style) and must
        /// not be set explicitly.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ncols: Option<u32>,
        /// Grid-only: row count. Required for `Grid`; `children.len()` must
        /// equal `nrows * ncols`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        nrows: Option<u32>,
        /// Grid-only: cells whose plot extent follows another cell's on one
        /// axis (`#182` follow-up — a jointplot's marginals against its joint
        /// panel, a clustermap's dendrograms against its heatmap). Empty for
        /// every other composition and skipped from canonical JSON, so a
        /// payload that declares none serializes byte-identically to one
        /// written before this field existed.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        extent_alignments: Vec<ExtentAlignment>,
        /// Root-only figure chrome (spec §6).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        subtitle: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caption: Option<String>,
        /// Root-only opaque render-config passthrough (chart-level config,
        /// not the per-call `viewport`/`theme`/`config` render-entry
        /// kwargs — those remain separate render-call arguments per spec
        /// §6's render-entry signature).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        config: Option<serde_json::Value>,
    },
    /// A placeholder cell — occupies a row/column slot but renders no panel
    /// or label (spec: JointChart's empty 2x2 corner, RepeatChart's
    /// `corner=True`). Valid only as a direct child of a `grid`/`wrap`
    /// composite, or (Task 10-rust) a direct child of `hconcat`/`vconcat`
    /// when both `width` and `height` are present ([`CompositeNode::validate`]
    /// rejects it elsewhere — `overlay`, and any layout at the tree root).
    ///
    /// - Under `grid`/`wrap`: `width`/`height` are ignored (cell math governs
    ///   the slot's size from its sibling cells; the layout pass
    ///   ([`crate::render::composite_render`]) reserves the slot and merges
    ///   no content for it) — both fields absent is the common case.
    /// - Under `hconcat`/`vconcat`: `width`/`height` (both required) reserve
    ///   exactly that much blank space in the flow, mirroring the legacy
    ///   string-compositor's placement of an empty-data child's blank SVG at
    ///   its own viewport size.
    Hole {
        /// Linear-layout-only sized-hole width (px). `None` under grid/wrap
        /// (ignored there); required alongside `height` under hconcat/vconcat.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        width: Option<f64>,
        /// Linear-layout-only sized-hole height (px). See `width`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        height: Option<f64>,
    },
}

impl CompositeNode {
    /// The node kind as used in error messages: `"leaf"`, `"hole"`, or the
    /// layout kind (`"hconcat"`, `"vconcat"`, `"grid"`, `"wrap"`, `"overlay"`)
    /// for a `Composite` node.
    pub fn kind_name(&self) -> &'static str {
        match self {
            CompositeNode::Leaf { .. } => "leaf",
            CompositeNode::Composite { layout, .. } => layout.as_str(),
            CompositeNode::Hole { .. } => "hole",
        }
    }

    /// This node's optional per-child panel label (Task 5d), if any. Present on
    /// `Leaf`/`Composite`; `None` when unset (the common case) and always
    /// `None` for `Hole` (a hole carries no fields at all).
    pub fn label(&self) -> Option<&str> {
        match self {
            CompositeNode::Leaf { label, .. } | CompositeNode::Composite { label, .. } => {
                label.as_deref()
            }
            CompositeNode::Hole { .. } => None,
        }
    }

    /// Validate structural invariants for the whole tree (spec §4): empty
    /// `children`, ratio arity vs. `nrows`/`ncols`, kind-specific field
    /// misuse (ratios/`ncols`/`nrows` on a layout that doesn't use them),
    /// root-only chrome fields (`title`/`subtitle`/`caption`) set on a
    /// non-root node, and `hole` children (valid only under `grid`/`wrap`,
    /// never as the tree root). Call once on the tree's root; recurses into
    /// every composite descendant. No warning-fallbacks — every violation is
    /// a typed [`CompositeSpecError`] naming the offending node's kind.
    pub fn validate(&self) -> Result<(), CompositeSpecError> {
        self.validate_node(true)
    }

    fn validate_node(&self, is_root: bool) -> Result<(), CompositeSpecError> {
        let kind = self.kind_name();
        // Per-child labels are root-forbidden (the root titles via `title`);
        // valid on any node below the root. Checked for both shapes before the
        // Composite-only structural checks so a labeled leaf root also fails.
        if is_root && self.label().is_some() {
            return Err(CompositeSpecError::LabelAtRoot { kind });
        }
        // A hole is a placeholder cell within a grid/wrap/linear parent — it
        // has no content of its own, so it can never stand alone as the tree
        // root.
        if is_root && matches!(self, CompositeNode::Hole { .. }) {
            return Err(CompositeSpecError::HoleAtRoot);
        }
        let CompositeNode::Composite {
            layout,
            children,
            row_ratios,
            col_ratios,
            ncols,
            nrows,
            extent_alignments,
            title,
            subtitle,
            caption,
            config,
            ..
        } = self
        else {
            return Ok(()); // Leaf or Hole: nothing further to validate structurally.
        };

        if children.is_empty() {
            return Err(CompositeSpecError::EmptyChildren { kind });
        }

        // Holes are valid as children of grid/wrap composites (spec: they
        // occupy an unused cell in a rectangular layout; `width`/`height` are
        // ignored there — cell math governs the slot) and, as of Task
        // 10-rust, as children of hconcat/vconcat when they carry BOTH
        // `width` and `height` (a sizeless hole under a linear layout has no
        // cell math to size it from, so it is rejected). Overlay never
        // supports holes, sized or not — an overlay has no notion of a cell
        // to leave empty.
        if !matches!(layout, CompositeLayout::Grid | CompositeLayout::Wrap) {
            for child in children {
                let CompositeNode::Hole { width, height } = child else { continue };
                if matches!(layout, CompositeLayout::Hconcat | CompositeLayout::Vconcat) {
                    if width.is_none() || height.is_none() {
                        return Err(CompositeSpecError::HoleSizeRequired { kind });
                    }
                } else {
                    return Err(CompositeSpecError::HoleNotSupported { kind });
                }
            }
        }

        // Extent alignment is cell-to-cell geometry, so it means nothing
        // without the row/column addressing only `grid` has (same reason
        // `row_ratios`/`col_ratios` are grid-only).
        if !matches!(layout, CompositeLayout::Grid) && !extent_alignments.is_empty() {
            return Err(CompositeSpecError::GridOnlyField {
                kind,
                field: "extent_alignments",
            });
        }

        match layout {
            CompositeLayout::Hconcat | CompositeLayout::Vconcat | CompositeLayout::Overlay => {
                if row_ratios.is_some() {
                    return Err(CompositeSpecError::GridOnlyField { kind, field: "row_ratios" });
                }
                if col_ratios.is_some() {
                    return Err(CompositeSpecError::GridOnlyField { kind, field: "col_ratios" });
                }
                if ncols.is_some() {
                    return Err(CompositeSpecError::WrapParamsNotSupported { kind, field: "ncols" });
                }
                if nrows.is_some() {
                    return Err(CompositeSpecError::WrapParamsNotSupported { kind, field: "nrows" });
                }
            }
            CompositeLayout::Wrap => {
                if row_ratios.is_some() {
                    return Err(CompositeSpecError::GridOnlyField { kind, field: "row_ratios" });
                }
                if col_ratios.is_some() {
                    return Err(CompositeSpecError::GridOnlyField { kind, field: "col_ratios" });
                }
                if nrows.is_some() {
                    return Err(CompositeSpecError::WrapParamsNotSupported { kind, field: "nrows" });
                }
                if ncols.is_none() {
                    return Err(CompositeSpecError::MissingField { kind, field: "ncols" });
                }
            }
            CompositeLayout::Grid => {
                let (Some(nrows), Some(ncols)) = (nrows, ncols) else {
                    let field = if nrows.is_none() { "nrows" } else { "ncols" };
                    return Err(CompositeSpecError::MissingField { kind, field });
                };
                let expected = *nrows as usize * *ncols as usize;
                if children.len() != expected {
                    return Err(CompositeSpecError::ChildCountMismatch {
                        kind,
                        expected,
                        got: children.len(),
                    });
                }
                if let Some(rr) = row_ratios {
                    if rr.len() != *nrows as usize {
                        return Err(CompositeSpecError::RatioArityMismatch {
                            kind,
                            field: "row_ratios",
                            expected: *nrows as usize,
                            got: rr.len(),
                        });
                    }
                }
                if let Some(cr) = col_ratios {
                    if cr.len() != *ncols as usize {
                        return Err(CompositeSpecError::RatioArityMismatch {
                            kind,
                            field: "col_ratios",
                            expected: *ncols as usize,
                            got: cr.len(),
                        });
                    }
                }
                validate_extent_alignments(kind, children, *ncols as usize, extent_alignments)?;
            }
        }

        if !is_root {
            if title.is_some() {
                return Err(CompositeSpecError::RootOnlyField { kind, field: "title" });
            }
            if subtitle.is_some() {
                return Err(CompositeSpecError::RootOnlyField { kind, field: "subtitle" });
            }
            if caption.is_some() {
                return Err(CompositeSpecError::RootOnlyField { kind, field: "caption" });
            }
            if config.is_some() {
                return Err(CompositeSpecError::RootOnlyField { kind, field: "config" });
            }
        }

        for child in children {
            child.validate_node(false)?;
        }
        Ok(())
    }
}

/// Check a grid's `extent_alignments` against the grid it is declared on
/// (see [`ExtentAlignment`]). Called from [`CompositeNode::validate`] once
/// `nrows`/`ncols` and the child count are known, so the row/column
/// addressing here is sound.
///
/// Every rejection is a malformed declaration — a statement the grid cannot
/// honor, which no lowering emits — so it refuses rather than degrading.
/// Whether the named cell is a *leaf* is deliberately NOT checked here: a
/// nested composite cell (a `LayerChart` as a jointplot's centre panel) is a
/// legitimate figure, and the render pass drops such an alignment rather
/// than failing the render. See `render::composite_render`'s
/// `resolve_extent_alignments`.
fn validate_extent_alignments(
    kind: &'static str,
    children: &[CompositeNode],
    ncols: usize,
    alignments: &[ExtentAlignment],
) -> Result<(), CompositeSpecError> {
    let mut claimed: HashSet<(usize, AlignAxis)> = HashSet::new();
    for (index, a) in alignments.iter().enumerate() {
        let invalid = |reason| CompositeSpecError::ExtentAlignmentInvalid { kind, index, reason };
        for cell in [a.follower, a.leader] {
            match children.get(cell) {
                None => return Err(invalid("names a cell outside this grid's children")),
                Some(CompositeNode::Hole { .. }) => {
                    return Err(invalid("names a 'hole' cell, which has no plot extent to share"));
                }
                Some(_) => {}
            }
        }
        if a.follower == a.leader {
            return Err(invalid("makes a cell follow itself"));
        }
        // An x alignment equalizes horizontal extents, which is only
        // meaningful for cells stacked in one column; y is the transpose.
        let same_column = a.follower % ncols == a.leader % ncols;
        let same_row = a.follower / ncols == a.leader / ncols;
        match a.axis {
            AlignAxis::X if !same_column => {
                return Err(invalid("aligns x across cells that are not in one column"));
            }
            AlignAxis::Y if !same_row => {
                return Err(invalid("aligns y across cells that are not in one row"));
            }
            _ => {}
        }
        if !claimed.insert((a.follower, a.axis)) {
            return Err(invalid("gives a cell a second leader on the same axis"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CompositeSpecError
// ---------------------------------------------------------------------------

/// Validation error for a malformed composite spec tree (spec §4). Every
/// variant names the offending node's `kind` (`"leaf"` never appears here —
/// leaves have no structural invariants to violate) so a Python
/// `ValueError` message pinpoints the failing node directly, matching the
/// existing `RenderError`/`SvgParseError` idiom (`Display` + `Error`,
/// mapped to `PyValueError` at the PyO3 boundary).
#[derive(Debug, Clone, PartialEq)]
pub enum CompositeSpecError {
    /// A composite node's `children` list is empty.
    EmptyChildren { kind: &'static str },
    /// A grid-only field (`row_ratios`/`col_ratios`/`extent_alignments`) was
    /// set on a layout that isn't `grid`.
    GridOnlyField { kind: &'static str, field: &'static str },
    /// An `extent_alignments` entry is not a statement this grid can honor:
    /// a cell index outside `children`, a cell following itself, a `hole`
    /// endpoint (a hole has no plot extent to share), two cells that are not
    /// in the same column for an `"x"` alignment (or the same row for `"y"`),
    /// or a second leader for a cell's axis. `index` is the entry's position
    /// in the list so the caller can find the one it wrote.
    ExtentAlignmentInvalid {
        kind: &'static str,
        index: usize,
        reason: &'static str,
    },
    /// `row_ratios`/`col_ratios` length didn't match `nrows`/`ncols`.
    RatioArityMismatch {
        kind: &'static str,
        field: &'static str,
        expected: usize,
        got: usize,
    },
    /// `ncols`/`nrows` was set on a layout that isn't `grid`/`wrap` (or,
    /// for `wrap`, `nrows` was set at all — it is always derived).
    WrapParamsNotSupported { kind: &'static str, field: &'static str },
    /// A field required by the node's layout was omitted (`grid` needs
    /// both `nrows`/`ncols`; `wrap` needs `ncols`).
    MissingField { kind: &'static str, field: &'static str },
    /// `grid`'s `children.len()` didn't equal `nrows * ncols`.
    ChildCountMismatch {
        kind: &'static str,
        expected: usize,
        got: usize,
    },
    /// `title`/`subtitle`/`caption` was set on a non-root node (spec §6:
    /// these fields are root-only figure chrome).
    RootOnlyField { kind: &'static str, field: &'static str },
    /// `label` was set on the root node — labels are per-child chrome (Task
    /// 5d); the root titles via `title`.
    LabelAtRoot { kind: &'static str },
    /// A `hole` child was placed under a composite layout other than `grid`/
    /// `wrap`/`hconcat`/`vconcat` — holes are unsupported there (e.g. `overlay`
    /// has no notion of a cell to leave empty).
    HoleNotSupported { kind: &'static str },
    /// A `hole` child under `hconcat`/`vconcat` omitted `width` and/or
    /// `height` — a linear layout has no cell math to size a hole from, so
    /// both are required there (grid/wrap holes have no such requirement).
    HoleSizeRequired { kind: &'static str },
    /// A `hole` node was used as the tree root — a hole is a placeholder
    /// cell within a grid/wrap parent and carries no content of its own.
    HoleAtRoot,
}

impl fmt::Display for CompositeSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyChildren { kind } => write!(f, "{kind}: children must not be empty"),
            Self::GridOnlyField { kind, field } => {
                write!(f, "{kind}: '{field}' is only supported on 'grid' nodes")
            }
            Self::ExtentAlignmentInvalid { kind, index, reason } => write!(
                f,
                "{kind}: extent_alignments[{index}] {reason}"
            ),
            Self::RatioArityMismatch { kind, field, expected, got } => write!(
                f,
                "{kind}: '{field}' length mismatch: expected {expected}, got {got}"
            ),
            Self::WrapParamsNotSupported { kind, field } => write!(
                f,
                "{kind}: '{field}' is only supported on 'grid'/'wrap' nodes"
            ),
            Self::MissingField { kind, field } => {
                write!(f, "{kind}: missing required field '{field}'")
            }
            Self::ChildCountMismatch { kind, expected, got } => write!(
                f,
                "{kind}: children length {got} does not match nrows*ncols ({expected})"
            ),
            Self::RootOnlyField { kind, field } => write!(
                f,
                "{kind}: '{field}' is only valid at the root of a composite tree"
            ),
            Self::LabelAtRoot { kind } => write!(
                f,
                "{kind}: 'label' is a per-child field and is not valid at the root \
                 of a composite tree (the root uses 'title')"
            ),
            Self::HoleNotSupported { kind } => write!(
                f,
                "{kind}: 'hole' children are only supported on 'grid'/'wrap'/'hconcat'/'vconcat' nodes"
            ),
            Self::HoleSizeRequired { kind } => write!(
                f,
                "{kind}: a 'hole' under 'hconcat'/'vconcat' requires both 'width' and 'height'"
            ),
            Self::HoleAtRoot => write!(
                f,
                "'hole' is not a valid composite tree root; holes are only valid \
                 as children of 'grid'/'wrap' nodes"
            ),
        }
    }
}

impl std::error::Error for CompositeSpecError {}

// ---------------------------------------------------------------------------
// PyO3 coercion
// ---------------------------------------------------------------------------

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Coerce the Python-side composite tree (nested dicts; see the module-level
/// wire-shape doc) into a validated [`CompositeNode`]. Call once on the
/// tree's root argument of a future `render_composite_svg`/
/// `render_composite_interactive` entry point (Task 4/5) — recurses through
/// `children` internally, then validates the whole tree (spec §4) before
/// returning, mapping any [`CompositeSpecError`] to a `ValueError` the same
/// way `render_err_to_py` does for `RenderError` (`render/binding.rs`).
pub(crate) fn composite_tree_from_py(obj: &Bound<'_, PyAny>) -> PyResult<CompositeNode> {
    let node = composite_node_from_py(obj)?;
    node.validate().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(node)
}

fn composite_node_from_py(obj: &Bound<'_, PyAny>) -> PyResult<CompositeNode> {
    let dict: &Bound<'_, PyDict> = obj
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("composite tree node must be a dict"))?;
    let kind = require_str(dict, "kind")?;

    match kind.as_str() {
        "leaf" => {
            let spec: ChartSpec = dict
                .get_item("spec")?
                .ok_or_else(|| PyValueError::new_err("leaf node missing required key 'spec'"))?
                .extract()?;
            let data: usize = dict
                .get_item("data")?
                .ok_or_else(|| PyValueError::new_err("leaf node missing required key 'data'"))?
                .extract()?;
            let label = opt_string(dict, "label")?;
            Ok(CompositeNode::Leaf { spec: Box::new(spec), data, label })
        }
        "composite" => {
            let layout_str = require_str(dict, "layout")?;
            let layout = CompositeLayout::from_str(&layout_str)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let children_obj = dict.get_item("children")?.ok_or_else(|| {
                PyValueError::new_err("composite node missing required key 'children'")
            })?;
            let children_list: &Bound<'_, PyList> = children_obj
                .cast::<PyList>()
                .map_err(|_| PyValueError::new_err("composite node: 'children' must be a list"))?;
            let children = children_list
                .iter()
                .map(|item| composite_node_from_py(&item))
                .collect::<PyResult<Vec<_>>>()?;

            let resolve = match opt_item(dict, "resolve")? {
                Some(v) => crate::pyo3_serde::from_py(&v, "resolve")?,
                None => CompositeResolve::default(),
            };
            let config = match opt_item(dict, "config")? {
                Some(v) => Some(crate::pyo3_serde::from_py(&v, "config")?),
                None => None,
            };

            let spacing: Option<f64> = match opt_item(dict, "spacing")? {
                Some(v) => Some(v.extract()?),
                None => None,
            };
            let row_ratios: Option<Vec<f64>> = match opt_item(dict, "row_ratios")? {
                Some(v) => Some(v.extract()?),
                None => None,
            };
            let col_ratios: Option<Vec<f64>> = match opt_item(dict, "col_ratios")? {
                Some(v) => Some(v.extract()?),
                None => None,
            };
            let ncols: Option<u32> = match opt_item(dict, "ncols")? {
                Some(v) => Some(v.extract()?),
                None => None,
            };
            let nrows: Option<u32> = match opt_item(dict, "nrows")? {
                Some(v) => Some(v.extract()?),
                None => None,
            };
            // Routed through the serde bridge (like `resolve`) rather than a
            // hand-written dict walk: a misspelled key or an axis outside
            // {"x","y"} then refuses with the field named, instead of
            // silently dropping the alignment.
            let extent_alignments: Vec<ExtentAlignment> = match opt_item(dict, "extent_alignments")?
            {
                Some(v) => crate::pyo3_serde::from_py(&v, "extent_alignments")?,
                None => Vec::new(),
            };
            let title: Option<String> = match opt_item(dict, "title")? {
                Some(v) => Some(v.extract()?),
                None => None,
            };
            let subtitle: Option<String> = match opt_item(dict, "subtitle")? {
                Some(v) => Some(v.extract()?),
                None => None,
            };
            let caption: Option<String> = match opt_item(dict, "caption")? {
                Some(v) => Some(v.extract()?),
                None => None,
            };
            let label = opt_string(dict, "label")?;

            Ok(CompositeNode::Composite {
                layout,
                children,
                label,
                resolve,
                spacing,
                row_ratios,
                col_ratios,
                ncols,
                nrows,
                extent_alignments,
                title,
                subtitle,
                caption,
                config,
            })
        }
        "hole" => {
            let width: Option<f64> = match opt_item(dict, "width")? {
                Some(v) => Some(v.extract()?),
                None => None,
            };
            let height: Option<f64> = match opt_item(dict, "height")? {
                Some(v) => Some(v.extract()?),
                None => None,
            };
            Ok(CompositeNode::Hole { width, height })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown composite tree node kind: '{other}'; expected 'leaf', 'composite', or 'hole'"
        ))),
    }
}

fn require_str(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    dict.get_item(key)?
        .ok_or_else(|| {
            PyValueError::new_err(format!("composite tree node missing required key '{key}'"))
        })?
        .extract()
}

/// Fetch `key` from `dict`, treating a present-but-`None` Python value the
/// same as an absent key (Python callers commonly build these dicts with
/// every key present and unset fields set to `None`).
fn opt_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    match dict.get_item(key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v)),
    }
}

/// Extract an optional string field, treating an absent-or-`None` value as
/// `None` (same present-but-`None` handling as [`opt_item`]).
fn opt_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match opt_item(dict, key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::Encoding;
    use crate::spec::mark::Mark;

    fn dummy_chart_spec() -> ChartSpec {
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

    fn leaf(data: usize) -> CompositeNode {
        CompositeNode::Leaf { spec: Box::new(dummy_chart_spec()), data, label: None }
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
            extent_alignments: Vec::new(),
            title: None,
            subtitle: None,
            caption: None,
            config: None,
        }
    }

    // -- serde round trip -----------------------------------------------

    #[test]
    fn leaf_round_trips() {
        let node = leaf(3);
        let json = serde_json::to_string(&node).unwrap();
        assert!(json.contains(r#""kind":"leaf""#), "json: {json}");
        assert!(json.contains(r#""data":3"#), "json: {json}");
        let parsed: CompositeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, node);
    }

    #[test]
    fn hole_round_trips() {
        let node = CompositeNode::Hole { width: None, height: None };
        let json = serde_json::to_string(&node).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"hole"}"#,
            "unsized hole omits both fields, same wire shape as before sized holes: {json}"
        );
        let parsed: CompositeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, node);
    }

    #[test]
    fn sized_hole_round_trips() {
        let node = CompositeNode::Hole { width: Some(40.0), height: Some(30.0) };
        let json = serde_json::to_string(&node).unwrap();
        assert_eq!(json, r#"{"kind":"hole","width":40.0,"height":30.0}"#, "json: {json}");
        let parsed: CompositeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, node);
    }

    #[test]
    fn hole_with_unknown_field_is_rejected() {
        // An extra key beyond `kind`/`width`/`height` on the wire (e.g. a
        // stray `label`) must be rejected the same way `deny_unknown_fields`
        // rejects it on `leaf`/`composite`, not silently ignored.
        let json = r#"{"kind":"hole","label":"nope"}"#;
        let err = serde_json::from_str::<CompositeNode>(json).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "err: {err}");
    }

    #[test]
    fn hconcat_of_two_leaves_round_trips() {
        let node = composite(CompositeLayout::Hconcat, vec![leaf(0), leaf(1)]);
        let json = serde_json::to_string(&node).unwrap();
        assert!(json.contains(r#""kind":"composite""#), "json: {json}");
        assert!(json.contains(r#""layout":"hconcat""#), "json: {json}");
        let parsed: CompositeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, node);
    }

    #[test]
    fn nested_grid_of_hconcat_round_trips_idempotent_json() {
        let inner_a = composite(CompositeLayout::Hconcat, vec![leaf(0), leaf(1)]);
        let inner_b = composite(CompositeLayout::Hconcat, vec![leaf(2), leaf(3)]);
        let mut root = composite(CompositeLayout::Grid, vec![inner_a, inner_b]);
        if let CompositeNode::Composite { nrows, ncols, .. } = &mut root {
            *nrows = Some(2);
            *ncols = Some(1);
        }
        let json1 = serde_json::to_string(&root).unwrap();
        let parsed: CompositeNode = serde_json::from_str(&json1).unwrap();
        assert_eq!(parsed, root);
        let json2 = serde_json::to_string(&parsed).unwrap();
        assert_eq!(json1, json2, "two-pass JSON differed");
    }

    #[test]
    fn resolve_defaults_to_independent_when_omitted() {
        let json = r#"{"kind":"composite","layout":"hconcat","children":[
            {"kind":"leaf","spec":{"mark":"point","encoding":{}},"data":0}
        ]}"#;
        let parsed: CompositeNode = serde_json::from_str(json).unwrap();
        let CompositeNode::Composite { resolve, .. } = parsed else { panic!("expected Composite") };
        assert_eq!(resolve.x, ResolveMode::Independent);
        assert_eq!(resolve.y, ResolveMode::Independent);
    }

    #[test]
    fn default_resolve_omitted_from_canonical_json() {
        let node = composite(CompositeLayout::Hconcat, vec![leaf(0)]);
        let json = serde_json::to_string(&node).unwrap();
        assert!(!json.contains("resolve"), "default resolve should be omitted: {json}");
    }

    #[test]
    fn explicit_shared_resolve_round_trips() {
        let mut node = composite(CompositeLayout::Hconcat, vec![leaf(0)]);
        if let CompositeNode::Composite { resolve, .. } = &mut node {
            resolve.x = ResolveMode::Shared;
        }
        let json = serde_json::to_string(&node).unwrap();
        assert!(
            json.contains(r#""resolve":{"x":"shared","y":"independent"}"#),
            "unset color/size are Option-None and omitted from canonical JSON: {json}"
        );
        let parsed: CompositeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, node);
    }

    #[test]
    fn resolve_color_size_round_trip_and_default_omission() {
        // color/size default to independent and are omitted from canonical JSON
        // when the whole resolve is default; a shared color/size request survives
        // a serde round trip.
        let default_node = composite(CompositeLayout::Hconcat, vec![leaf(0)]);
        let default_json = serde_json::to_string(&default_node).unwrap();
        assert!(!default_json.contains("resolve"), "default resolve omitted: {default_json}");

        let mut node = composite(CompositeLayout::Hconcat, vec![leaf(0)]);
        if let CompositeNode::Composite { resolve, .. } = &mut node {
            resolve.color = Some(ResolveMode::Shared);
            resolve.size = Some(ResolveMode::Shared);
        }
        let json = serde_json::to_string(&node).unwrap();
        assert!(json.contains(r#""color":"shared""#), "json: {json}");
        assert!(json.contains(r#""size":"shared""#), "json: {json}");
        let parsed: CompositeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, node);
    }

    // -- legend resolution (GH #16 wire contract) --------------------------

    #[test]
    fn legend_defaults_to_follow_scale_when_omitted() {
        // Existing (pre-legend) payloads must deserialize identically: no
        // "legend" key means both channels are unset ("follow scale").
        let json = r#"{"kind":"composite","layout":"hconcat","children":[
            {"kind":"leaf","spec":{"mark":"point","encoding":{}},"data":0}
        ],"resolve":{"color":"shared"}}"#;
        let parsed: CompositeNode = serde_json::from_str(json).unwrap();
        let CompositeNode::Composite { resolve, .. } = parsed else { panic!("expected Composite") };
        assert_eq!(resolve.legend, CompositeLegendResolve::default());
        assert_eq!(resolve.legend.color, None);
        assert_eq!(resolve.legend.size, None);
    }

    #[test]
    fn default_legend_omitted_from_canonical_json() {
        let node = composite(CompositeLayout::Hconcat, vec![leaf(0)]);
        let json = serde_json::to_string(&node).unwrap();
        assert!(!json.contains("legend"), "default legend should be omitted: {json}");
    }

    #[test]
    fn explicit_legend_override_round_trips() {
        let mut node = composite(CompositeLayout::Hconcat, vec![leaf(0)]);
        if let CompositeNode::Composite { resolve, .. } = &mut node {
            resolve.color = Some(ResolveMode::Shared);
            resolve.legend.color = Some(ResolveMode::Independent);
        }
        let json = serde_json::to_string(&node).unwrap();
        assert!(
            json.contains(r#""legend":{"color":"independent"}"#),
            "unset size must not appear in the legend object: {json}"
        );
        let parsed: CompositeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, node);
        let CompositeNode::Composite { resolve, .. } = parsed else { panic!("expected Composite") };
        assert_eq!(resolve.legend.color, Some(ResolveMode::Independent));
        assert_eq!(resolve.legend.size, None);
    }

    #[test]
    fn explicit_legend_both_channels_round_trips() {
        let mut node = composite(CompositeLayout::Hconcat, vec![leaf(0)]);
        if let CompositeNode::Composite { resolve, .. } = &mut node {
            resolve.color = Some(ResolveMode::Shared);
            resolve.size = Some(ResolveMode::Shared);
            resolve.legend.color = Some(ResolveMode::Shared);
            resolve.legend.size = Some(ResolveMode::Independent);
        }
        let json = serde_json::to_string(&node).unwrap();
        assert!(
            json.contains(r#""legend":{"color":"shared","size":"independent"}"#),
            "json: {json}"
        );
        let parsed: CompositeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, node);
    }

    #[test]
    fn composite_tree_from_py_reads_legend_override() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let leaf_a = PyDict::new(py);
            leaf_a.set_item("kind", "leaf").unwrap();
            leaf_a.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
            leaf_a.set_item("data", 0usize).unwrap();

            let children = pyo3::types::PyList::new(py, [leaf_a]).unwrap();

            let legend = PyDict::new(py);
            legend.set_item("color", "independent").unwrap();

            let resolve = PyDict::new(py);
            resolve.set_item("color", "shared").unwrap();
            resolve.set_item("legend", legend).unwrap();

            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "vconcat").unwrap();
            root.set_item("children", children).unwrap();
            root.set_item("resolve", resolve).unwrap();

            let node = composite_tree_from_py(root.as_any()).unwrap();
            let CompositeNode::Composite { resolve, .. } = node else { panic!("expected Composite") };
            assert_eq!(resolve.color, Some(ResolveMode::Shared));
            assert_eq!(resolve.legend.color, Some(ResolveMode::Independent));
            assert_eq!(resolve.legend.size, None);
        });
    }

    #[test]
    fn composite_tree_from_py_omitted_legend_defaults_to_follow_scale() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let leaf_a = PyDict::new(py);
            leaf_a.set_item("kind", "leaf").unwrap();
            leaf_a.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
            leaf_a.set_item("data", 0usize).unwrap();

            let children = pyo3::types::PyList::new(py, [leaf_a]).unwrap();

            let resolve = PyDict::new(py);
            resolve.set_item("color", "shared").unwrap();

            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "vconcat").unwrap();
            root.set_item("children", children).unwrap();
            root.set_item("resolve", resolve).unwrap();

            let node = composite_tree_from_py(root.as_any()).unwrap();
            let CompositeNode::Composite { resolve, .. } = node else { panic!("expected Composite") };
            assert_eq!(resolve.legend, CompositeLegendResolve::default());
            assert_eq!(resolve.color, Some(ResolveMode::Shared));
        });
    }

    #[test]
    fn composite_tree_from_py_reads_color_size_resolve() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let leaf_a = PyDict::new(py);
            leaf_a.set_item("kind", "leaf").unwrap();
            leaf_a.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
            leaf_a.set_item("data", 0usize).unwrap();

            let children = pyo3::types::PyList::new(py, [leaf_a]).unwrap();

            let resolve = PyDict::new(py);
            resolve.set_item("color", "shared").unwrap();
            resolve.set_item("size", "shared").unwrap();

            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "vconcat").unwrap();
            root.set_item("children", children).unwrap();
            root.set_item("resolve", resolve).unwrap();

            let node = composite_tree_from_py(root.as_any()).unwrap();
            let CompositeNode::Composite { resolve, .. } = node else { panic!("expected Composite") };
            assert_eq!(resolve.color, Some(ResolveMode::Shared));
            assert_eq!(resolve.size, Some(ResolveMode::Shared));
            assert_eq!(resolve.x, ResolveMode::Independent);
            assert_eq!(resolve.y, ResolveMode::Independent);
        });
    }

    #[test]
    fn unknown_top_level_field_is_rejected() {
        let json = r#"{"kind":"leaf","spec":{"mark":"point","encoding":{}},"data":0,"typo":1}"#;
        let err = serde_json::from_str::<CompositeNode>(json).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "err: {err}");
    }

    // -- CompositeLayout::from_str ----------------------------------------

    #[test]
    fn composite_layout_from_str_accepts_all_five() {
        for (s, expected) in [
            ("hconcat", CompositeLayout::Hconcat),
            ("vconcat", CompositeLayout::Vconcat),
            ("grid", CompositeLayout::Grid),
            ("wrap", CompositeLayout::Wrap),
            ("overlay", CompositeLayout::Overlay),
        ] {
            assert_eq!(CompositeLayout::from_str(s).unwrap(), expected);
        }
    }

    #[test]
    fn composite_layout_from_str_rejects_unknown() {
        let err = CompositeLayout::from_str("spaghetti").unwrap_err();
        assert!(err.to_string().contains("spaghetti"));
    }

    // -- validate() ---------------------------------------------------------

    #[test]
    fn validate_rejects_empty_children() {
        for layout in [
            CompositeLayout::Hconcat,
            CompositeLayout::Vconcat,
            CompositeLayout::Grid,
            CompositeLayout::Wrap,
            CompositeLayout::Overlay,
        ] {
            let node = composite(layout, vec![]);
            let err = node.validate().unwrap_err();
            assert_eq!(err, CompositeSpecError::EmptyChildren { kind: layout.as_str() });
        }
    }

    // -- extent alignment (#182 follow-up) ---------------------------------

    /// A jointplot's grid, built exactly as `JointChart._composite_tree`
    /// lowers it: `[top, HOLE, centre, right]`, each marginal declared against
    /// the centre CELL (2, not leaf 1 — the wire counts cells).
    fn joint_grid_with(alignments: Vec<ExtentAlignment>) -> CompositeNode {
        let mut node = composite(
            CompositeLayout::Grid,
            vec![
                leaf(0),
                CompositeNode::Hole { width: None, height: None },
                leaf(1),
                leaf(2),
            ],
        );
        if let CompositeNode::Composite {
            nrows,
            ncols,
            extent_alignments,
            ..
        } = &mut node
        {
            *nrows = Some(2);
            *ncols = Some(2);
            *extent_alignments = alignments;
        }
        node
    }

    fn joint_alignments() -> Vec<ExtentAlignment> {
        vec![
            ExtentAlignment { follower: 0, leader: 2, axis: AlignAxis::X },
            ExtentAlignment { follower: 3, leader: 2, axis: AlignAxis::Y },
        ]
    }

    /// **Wire byte-stability.** A payload captured before this field existed
    /// parses, reads as declaring nothing, and re-serializes to the SAME
    /// BYTES — proved by round-tripping the captured string, not by reading
    /// the `skip_serializing_if` attribute (the same proof
    /// `ScaleSpec::Linear::system_injected` took in `2299d31b` and
    /// `MarkKwargsSpec::value_axis` in `9c570342`).
    #[test]
    fn pre_alignment_grid_payload_round_trips_to_identical_bytes() {
        // A jointplot-shaped grid with a hole, written with every key a
        // pre-change producer emitted and nothing else. (The leaves carry
        // `ChartSpec`'s explicit default `data` ref because that struct
        // serializes it — unrelated to this field, and included so the
        // equality below is byte-for-byte rather than approximate.)
        const CAPTURED: &str = r#"{"kind":"composite","layout":"grid","children":[{"kind":"leaf","spec":{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{}},"data":0},{"kind":"hole"},{"kind":"leaf","spec":{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{}},"data":1}],"spacing":8.0,"ncols":2,"nrows":2}"#;
        let parsed: CompositeNode = serde_json::from_str(CAPTURED).unwrap();
        let CompositeNode::Composite { ref extent_alignments, .. } = parsed else {
            panic!("expected Composite")
        };
        assert!(
            extent_alignments.is_empty(),
            "an absent field must read as 'declares nothing', never as a default alignment"
        );
        assert_eq!(serde_json::to_string(&parsed).unwrap(), CAPTURED);
    }

    /// A declaration survives the wire in both directions, keyed the way the
    /// Python lowering writes it (`_extent_alignment`).
    #[test]
    fn declared_alignments_round_trip_on_the_wire() {
        let node = joint_grid_with(joint_alignments());
        let json = serde_json::to_string(&node).unwrap();
        assert!(
            json.contains(
                r#""extent_alignments":[{"follower":0,"leader":2,"axis":"x"},{"follower":3,"leader":2,"axis":"y"}]"#
            ),
            "json: {json}"
        );
        let parsed: CompositeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, node);
        assert_eq!(serde_json::to_string(&parsed).unwrap(), json);
    }

    /// The shapes the lowering actually emits validate — the guard against a
    /// refusal rule that is too strict to let the real figures through.
    #[test]
    fn the_lowerings_own_declarations_validate() {
        joint_grid_with(joint_alignments()).validate().unwrap();
        // ClusterMap: [HOLE, col_dendro, row_dendro, heatmap], dendrograms
        // declared against the heatmap cell (3).
        let mut clustermap = composite(
            CompositeLayout::Grid,
            vec![
                CompositeNode::Hole { width: None, height: None },
                leaf(0),
                leaf(1),
                leaf(2),
            ],
        );
        if let CompositeNode::Composite {
            nrows,
            ncols,
            extent_alignments,
            ..
        } = &mut clustermap
        {
            *nrows = Some(2);
            *ncols = Some(2);
            *extent_alignments = vec![
                ExtentAlignment { follower: 1, leader: 3, axis: AlignAxis::X },
                ExtentAlignment { follower: 2, leader: 3, axis: AlignAxis::Y },
            ];
        }
        clustermap.validate().unwrap();
    }

    /// Every way a declaration can be unhonorable is REFUSED naming the
    /// entry, not silently dropped: the grid would otherwise render as if the
    /// caller had said nothing, which is indistinguishable from the feature
    /// being broken.
    #[test]
    fn validate_refuses_every_unhonorable_alignment() {
        for (alignment, reason) in [
            (
                ExtentAlignment { follower: 0, leader: 9, axis: AlignAxis::X },
                "names a cell outside this grid's children",
            ),
            (
                ExtentAlignment { follower: 0, leader: 1, axis: AlignAxis::X },
                "names a 'hole' cell, which has no plot extent to share",
            ),
            (
                ExtentAlignment { follower: 2, leader: 2, axis: AlignAxis::X },
                "makes a cell follow itself",
            ),
            (
                // Cells 2 and 3 are in one ROW, so they can agree on y, never x.
                ExtentAlignment { follower: 3, leader: 2, axis: AlignAxis::X },
                "aligns x across cells that are not in one column",
            ),
            (
                // Cells 0 and 2 are in one COLUMN, so they can agree on x, never y.
                ExtentAlignment { follower: 0, leader: 2, axis: AlignAxis::Y },
                "aligns y across cells that are not in one row",
            ),
        ] {
            let node = joint_grid_with(vec![alignment]);
            assert_eq!(
                node.validate().unwrap_err(),
                CompositeSpecError::ExtentAlignmentInvalid { kind: "grid", index: 0, reason }
            );
        }

        // A second leader for the same cell+axis: which one wins would be
        // whichever the resolver happened to visit last.
        let node = joint_grid_with(vec![
            ExtentAlignment { follower: 0, leader: 2, axis: AlignAxis::X },
            ExtentAlignment { follower: 0, leader: 2, axis: AlignAxis::X },
        ]);
        assert_eq!(
            node.validate().unwrap_err(),
            CompositeSpecError::ExtentAlignmentInvalid {
                kind: "grid",
                index: 1,
                reason: "gives a cell a second leader on the same axis",
            }
        );

        // One cell may still follow different leaders on DIFFERENT axes: the
        // two declarations describe independent extents, so the duplicate rule
        // must key on cell AND axis, not on the cell alone. Needs a dense grid
        // (a jointplot's row-mate is the hole).
        let mut dense = composite(
            CompositeLayout::Grid,
            vec![leaf(0), leaf(1), leaf(2), leaf(3)],
        );
        if let CompositeNode::Composite {
            nrows,
            ncols,
            extent_alignments,
            ..
        } = &mut dense
        {
            *nrows = Some(2);
            *ncols = Some(2);
            *extent_alignments = vec![
                ExtentAlignment { follower: 0, leader: 2, axis: AlignAxis::X },
                ExtentAlignment { follower: 0, leader: 1, axis: AlignAxis::Y },
            ];
        }
        dense.validate().unwrap();
    }

    /// Cell-to-cell geometry needs the row/column addressing only `grid` has,
    /// the same reason `row_ratios`/`col_ratios` are grid-only.
    #[test]
    fn validate_rejects_alignments_on_a_non_grid_layout() {
        for layout in [
            CompositeLayout::Hconcat,
            CompositeLayout::Vconcat,
            CompositeLayout::Overlay,
            CompositeLayout::Wrap,
        ] {
            let mut node = composite(layout, vec![leaf(0), leaf(1)]);
            if let CompositeNode::Composite { extent_alignments, ncols, .. } = &mut node {
                *extent_alignments =
                    vec![ExtentAlignment { follower: 0, leader: 1, axis: AlignAxis::Y }];
                if layout == CompositeLayout::Wrap {
                    *ncols = Some(2); // otherwise wrap fails its own required-field check first
                }
            }
            assert_eq!(
                node.validate().unwrap_err(),
                CompositeSpecError::GridOnlyField {
                    kind: layout.as_str(),
                    field: "extent_alignments",
                }
            );
        }
    }

    #[test]
    fn validate_rejects_ratios_on_hconcat() {
        let mut node = composite(CompositeLayout::Hconcat, vec![leaf(0), leaf(1)]);
        if let CompositeNode::Composite { col_ratios, .. } = &mut node {
            *col_ratios = Some(vec![1.0, 2.0]);
        }
        let err = node.validate().unwrap_err();
        assert_eq!(
            err,
            CompositeSpecError::GridOnlyField { kind: "hconcat", field: "col_ratios" }
        );
    }

    #[test]
    fn validate_rejects_ncols_on_vconcat() {
        let mut node = composite(CompositeLayout::Vconcat, vec![leaf(0)]);
        if let CompositeNode::Composite { ncols, .. } = &mut node {
            *ncols = Some(2);
        }
        let err = node.validate().unwrap_err();
        assert_eq!(
            err,
            CompositeSpecError::WrapParamsNotSupported { kind: "vconcat", field: "ncols" }
        );
    }

    #[test]
    fn validate_grid_requires_nrows_and_ncols() {
        let node = composite(CompositeLayout::Grid, vec![leaf(0)]);
        let err = node.validate().unwrap_err();
        assert_eq!(err, CompositeSpecError::MissingField { kind: "grid", field: "nrows" });
    }

    #[test]
    fn validate_wrap_requires_ncols() {
        let node = composite(CompositeLayout::Wrap, vec![leaf(0)]);
        let err = node.validate().unwrap_err();
        assert_eq!(err, CompositeSpecError::MissingField { kind: "wrap", field: "ncols" });
    }

    #[test]
    fn validate_wrap_rejects_explicit_nrows() {
        let mut node = composite(CompositeLayout::Wrap, vec![leaf(0)]);
        if let CompositeNode::Composite { ncols, nrows, .. } = &mut node {
            *ncols = Some(2);
            *nrows = Some(1);
        }
        let err = node.validate().unwrap_err();
        assert_eq!(
            err,
            CompositeSpecError::WrapParamsNotSupported { kind: "wrap", field: "nrows" }
        );
    }

    #[test]
    fn validate_grid_rejects_child_count_mismatch() {
        let mut node = composite(CompositeLayout::Grid, vec![leaf(0), leaf(1), leaf(2)]);
        if let CompositeNode::Composite { nrows, ncols, .. } = &mut node {
            *nrows = Some(2);
            *ncols = Some(2);
        }
        let err = node.validate().unwrap_err();
        assert_eq!(err, CompositeSpecError::ChildCountMismatch { kind: "grid", expected: 4, got: 3 });
    }

    #[test]
    fn validate_grid_rejects_row_ratio_arity_mismatch() {
        let mut node = composite(CompositeLayout::Grid, vec![leaf(0), leaf(1), leaf(2), leaf(3)]);
        if let CompositeNode::Composite { nrows, ncols, row_ratios, .. } = &mut node {
            *nrows = Some(2);
            *ncols = Some(2);
            *row_ratios = Some(vec![1.0, 2.0, 3.0]);
        }
        let err = node.validate().unwrap_err();
        assert_eq!(
            err,
            CompositeSpecError::RatioArityMismatch { kind: "grid", field: "row_ratios", expected: 2, got: 3 }
        );
    }

    #[test]
    fn validate_grid_accepts_matching_ratios() {
        let mut node = composite(CompositeLayout::Grid, vec![leaf(0), leaf(1), leaf(2), leaf(3)]);
        if let CompositeNode::Composite { nrows, ncols, row_ratios, col_ratios, .. } = &mut node {
            *nrows = Some(2);
            *ncols = Some(2);
            *row_ratios = Some(vec![1.0, 2.0]);
            *col_ratios = Some(vec![1.0, 1.0]);
        }
        assert!(node.validate().is_ok());
    }

    #[test]
    fn validate_rejects_title_on_nested_non_root_node() {
        let mut inner = composite(CompositeLayout::Hconcat, vec![leaf(0), leaf(1)]);
        if let CompositeNode::Composite { title, .. } = &mut inner {
            *title = Some("nested title".into());
        }
        let root = composite(CompositeLayout::Vconcat, vec![inner]);
        let err = root.validate().unwrap_err();
        assert_eq!(
            err,
            CompositeSpecError::RootOnlyField { kind: "hconcat", field: "title" }
        );
    }

    #[test]
    fn validate_accepts_minimal_hconcat_vconcat_overlay() {
        for layout in [
            CompositeLayout::Hconcat,
            CompositeLayout::Vconcat,
            CompositeLayout::Overlay,
        ] {
            let node = composite(layout, vec![leaf(0), leaf(1)]);
            assert!(node.validate().is_ok(), "minimal {layout:?} should validate");
        }
    }

    // -- hole (Task 8a) -------------------------------------------------------

    #[test]
    fn validate_rejects_hole_at_root() {
        let err = CompositeNode::Hole { width: None, height: None }.validate().unwrap_err();
        assert_eq!(err, CompositeSpecError::HoleAtRoot);
    }

    #[test]
    fn validate_rejects_sizeless_hole_child_on_overlay() {
        // Overlay never supports holes, sized or not.
        let node = composite(
            CompositeLayout::Overlay,
            vec![leaf(0), CompositeNode::Hole { width: None, height: None }],
        );
        let err = node.validate().unwrap_err();
        assert_eq!(err, CompositeSpecError::HoleNotSupported { kind: "overlay" });
    }

    #[test]
    fn validate_rejects_sized_hole_child_on_overlay() {
        // Even a fully-sized hole is illegal under overlay (Task 10-rust: only
        // hconcat/vconcat gained sized-hole support).
        let node = composite(
            CompositeLayout::Overlay,
            vec![leaf(0), CompositeNode::Hole { width: Some(10.0), height: Some(10.0) }],
        );
        let err = node.validate().unwrap_err();
        assert_eq!(err, CompositeSpecError::HoleNotSupported { kind: "overlay" });
    }

    #[test]
    fn validate_rejects_sizeless_hole_child_on_hconcat_vconcat() {
        for layout in [CompositeLayout::Hconcat, CompositeLayout::Vconcat] {
            let node = composite(layout, vec![leaf(0), CompositeNode::Hole { width: None, height: None }]);
            let err = node.validate().unwrap_err();
            assert_eq!(
                err,
                CompositeSpecError::HoleSizeRequired { kind: layout.as_str() },
                "sizeless hole should be rejected under {layout:?}"
            );
        }
    }

    #[test]
    fn validate_rejects_partially_sized_hole_child_on_hconcat_vconcat() {
        for layout in [CompositeLayout::Hconcat, CompositeLayout::Vconcat] {
            let width_only =
                composite(layout, vec![leaf(0), CompositeNode::Hole { width: Some(10.0), height: None }]);
            assert_eq!(
                width_only.validate().unwrap_err(),
                CompositeSpecError::HoleSizeRequired { kind: layout.as_str() },
                "width-only hole should be rejected under {layout:?}"
            );

            let height_only =
                composite(layout, vec![leaf(0), CompositeNode::Hole { width: None, height: Some(10.0) }]);
            assert_eq!(
                height_only.validate().unwrap_err(),
                CompositeSpecError::HoleSizeRequired { kind: layout.as_str() },
                "height-only hole should be rejected under {layout:?}"
            );
        }
    }

    #[test]
    fn validate_accepts_fully_sized_hole_child_on_hconcat_vconcat() {
        for layout in [CompositeLayout::Hconcat, CompositeLayout::Vconcat] {
            let node = composite(
                layout,
                vec![leaf(0), CompositeNode::Hole { width: Some(40.0), height: Some(30.0) }, leaf(1)],
            );
            assert!(node.validate().is_ok(), "fully-sized hole should validate under {layout:?}");
        }
    }

    #[test]
    fn validate_accepts_hole_in_grid() {
        let mut node = composite(
            CompositeLayout::Grid,
            vec![leaf(0), CompositeNode::Hole { width: None, height: None }, leaf(1), leaf(2)],
        );
        if let CompositeNode::Composite { nrows, ncols, .. } = &mut node {
            *nrows = Some(2);
            *ncols = Some(2);
        }
        assert!(node.validate().is_ok());
    }

    #[test]
    fn validate_accepts_sized_fields_ignored_under_grid() {
        // Task 10-rust: grid/wrap holes accept width/height on the wire (no
        // validation error) but the fields have no effect on cell math — the
        // layout pass ignores them there, tested in composite_render.rs.
        let mut node = composite(
            CompositeLayout::Grid,
            vec![
                leaf(0),
                CompositeNode::Hole { width: Some(999.0), height: Some(999.0) },
                leaf(1),
                leaf(2),
            ],
        );
        if let CompositeNode::Composite { nrows, ncols, .. } = &mut node {
            *nrows = Some(2);
            *ncols = Some(2);
        }
        assert!(node.validate().is_ok());
    }

    #[test]
    fn validate_accepts_hole_in_wrap() {
        let mut node = composite(
            CompositeLayout::Wrap,
            vec![leaf(0), leaf(1), CompositeNode::Hole { width: None, height: None }],
        );
        if let CompositeNode::Composite { ncols, .. } = &mut node {
            *ncols = Some(2);
        }
        assert!(node.validate().is_ok(), "trailing hole should validate under wrap");
    }

    #[test]
    fn validate_accepts_wrap_with_ncols() {
        let mut node = composite(CompositeLayout::Wrap, vec![leaf(0), leaf(1), leaf(2)]);
        if let CompositeNode::Composite { ncols, .. } = &mut node {
            *ncols = Some(2);
        }
        assert!(node.validate().is_ok());
    }

    #[test]
    fn validate_rejects_config_on_nested_non_root_node() {
        let mut inner = composite(CompositeLayout::Hconcat, vec![leaf(0), leaf(1)]);
        if let CompositeNode::Composite { config, .. } = &mut inner {
            *config = Some(serde_json::json!({"background": "#fff"}));
        }
        let root = composite(CompositeLayout::Vconcat, vec![inner]);
        let err = root.validate().unwrap_err();
        assert_eq!(
            err,
            CompositeSpecError::RootOnlyField { kind: "hconcat", field: "config" }
        );
    }

    #[test]
    fn validate_accepts_title_at_root() {
        let mut root = composite(CompositeLayout::Hconcat, vec![leaf(0), leaf(1)]);
        if let CompositeNode::Composite { title, subtitle, caption, config, .. } = &mut root {
            *title = Some("t".into());
            *subtitle = Some("s".into());
            *caption = Some("c".into());
            *config = Some(serde_json::json!({"background": "#fff"}));
        }
        assert!(root.validate().is_ok());
    }

    // -- per-child labels (Task 5d) -----------------------------------------

    #[test]
    fn validate_rejects_label_on_composite_root() {
        let mut root = composite(CompositeLayout::Hconcat, vec![leaf(0), leaf(1)]);
        if let CompositeNode::Composite { label, .. } = &mut root {
            *label = Some("root label".into());
        }
        let err = root.validate().unwrap_err();
        assert_eq!(err, CompositeSpecError::LabelAtRoot { kind: "hconcat" });
    }

    #[test]
    fn validate_rejects_label_on_leaf_root() {
        // A bare leaf used as the tree root may not carry a label either.
        let mut root = leaf(0);
        if let CompositeNode::Leaf { label, .. } = &mut root {
            *label = Some("leaf root".into());
        }
        let err = root.validate().unwrap_err();
        assert_eq!(err, CompositeSpecError::LabelAtRoot { kind: "leaf" });
    }

    #[test]
    fn validate_accepts_label_on_non_root_children() {
        // Labels are valid anywhere below the root — on both leaf and composite
        // children.
        let mut inner = composite(CompositeLayout::Vconcat, vec![leaf(0), leaf(1)]);
        if let CompositeNode::Composite { label, .. } = &mut inner {
            *label = Some("composite child".into());
        }
        let mut labeled_leaf = leaf(2);
        if let CompositeNode::Leaf { label, .. } = &mut labeled_leaf {
            *label = Some("leaf child".into());
        }
        let mut root = composite(CompositeLayout::Wrap, vec![inner, labeled_leaf]);
        if let CompositeNode::Composite { ncols, .. } = &mut root {
            *ncols = Some(2);
        }
        assert!(root.validate().is_ok(), "labels valid on non-root children");
    }

    #[test]
    fn label_round_trips_in_json_and_is_omitted_when_absent() {
        // Present label survives a serde round trip; absent label is skipped
        // from canonical JSON (so Task 6/7 label-less trees stay byte-identical).
        let mut node = composite(CompositeLayout::Hconcat, vec![leaf(0), leaf(1)]);
        if let CompositeNode::Composite { children, .. } = &mut node {
            if let CompositeNode::Leaf { label, .. } = &mut children[0] {
                *label = Some("A".into());
            }
        }
        let json = serde_json::to_string(&node).unwrap();
        assert!(json.contains(r#""label":"A""#), "labeled child must serialize: {json}");
        // The unlabeled sibling + the root must NOT emit a label key.
        assert_eq!(json.matches("label").count(), 1, "only the one label present: {json}");
        let parsed: CompositeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, node);
    }

    #[test]
    fn composite_tree_from_py_reads_child_label() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let child = PyDict::new(py);
            child.set_item("kind", "leaf").unwrap();
            child.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
            child.set_item("data", 0usize).unwrap();
            child.set_item("label", "Model A").unwrap();

            let children = pyo3::types::PyList::new(py, [child]).unwrap();
            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "hconcat").unwrap();
            root.set_item("children", children).unwrap();

            let node = composite_tree_from_py(root.as_any()).unwrap();
            let CompositeNode::Composite { children, label, .. } = node else {
                panic!("expected Composite")
            };
            assert_eq!(label, None, "root carries no label");
            assert_eq!(children[0].label(), Some("Model A"), "child label read from dict");
        });
    }

    #[test]
    fn composite_tree_from_py_rejects_label_at_root() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let child = PyDict::new(py);
            child.set_item("kind", "leaf").unwrap();
            child.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
            child.set_item("data", 0usize).unwrap();

            let children = pyo3::types::PyList::new(py, [child]).unwrap();
            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "hconcat").unwrap();
            root.set_item("children", children).unwrap();
            root.set_item("label", "not allowed here").unwrap();

            let err = composite_tree_from_py(root.as_any()).unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("label") && msg.contains("root"), "msg: {msg}");
        });
    }

    // -- PyO3 coercion --------------------------------------------------

    #[test]
    fn composite_tree_from_py_leaf_extracts_spec_and_data() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("kind", "leaf").unwrap();
            d.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
            d.set_item("data", 2usize).unwrap();
            let node = composite_tree_from_py(d.as_any()).unwrap();
            match node {
                CompositeNode::Leaf { spec, data, .. } => {
                    assert_eq!(data, 2);
                    assert_eq!(*spec, dummy_chart_spec());
                }
                other => panic!("expected Leaf, got {other:?}"),
            }
        });
    }

    #[test]
    fn composite_tree_from_py_nested_hconcat() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let leaf_a = PyDict::new(py);
            leaf_a.set_item("kind", "leaf").unwrap();
            leaf_a.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
            leaf_a.set_item("data", 0usize).unwrap();

            let leaf_b = PyDict::new(py);
            leaf_b.set_item("kind", "leaf").unwrap();
            leaf_b.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
            leaf_b.set_item("data", 1usize).unwrap();

            let children = pyo3::types::PyList::new(py, [leaf_a, leaf_b]).unwrap();

            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "hconcat").unwrap();
            root.set_item("children", children).unwrap();

            let node = composite_tree_from_py(root.as_any()).unwrap();
            match node {
                CompositeNode::Composite { layout, children, resolve, .. } => {
                    assert_eq!(layout, CompositeLayout::Hconcat);
                    assert_eq!(children.len(), 2);
                    assert_eq!(resolve, CompositeResolve::default());
                }
                other => panic!("expected Composite, got {other:?}"),
            }
        });
    }

    #[test]
    fn composite_tree_from_py_resolve_dict_round_trips() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let leaf_a = PyDict::new(py);
            leaf_a.set_item("kind", "leaf").unwrap();
            leaf_a.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
            leaf_a.set_item("data", 0usize).unwrap();

            let children = pyo3::types::PyList::new(py, [leaf_a]).unwrap();

            let resolve = PyDict::new(py);
            resolve.set_item("x", "shared").unwrap();

            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "vconcat").unwrap();
            root.set_item("children", children).unwrap();
            root.set_item("resolve", resolve).unwrap();

            let node = composite_tree_from_py(root.as_any()).unwrap();
            let CompositeNode::Composite { resolve, .. } = node else { panic!("expected Composite") };
            assert_eq!(resolve.x, ResolveMode::Shared);
            assert_eq!(resolve.y, ResolveMode::Independent);
        });
    }

    #[test]
    fn composite_tree_from_py_grid_extracts_all_layout_fields() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mk_leaf = |idx: usize| {
                let d = PyDict::new(py);
                d.set_item("kind", "leaf").unwrap();
                d.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
                d.set_item("data", idx).unwrap();
                d
            };
            let children =
                pyo3::types::PyList::new(py, [mk_leaf(0), mk_leaf(1), mk_leaf(2), mk_leaf(3)])
                    .unwrap();

            let config = PyDict::new(py);
            config.set_item("background", "#fff").unwrap();

            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "grid").unwrap();
            root.set_item("children", children).unwrap();
            root.set_item("nrows", 2usize).unwrap();
            root.set_item("ncols", 2usize).unwrap();
            root.set_item("row_ratios", vec![1.0f64, 3.0]).unwrap();
            root.set_item("col_ratios", vec![2.0f64, 1.0]).unwrap();
            root.set_item("spacing", 12.0f64).unwrap();
            root.set_item("config", config).unwrap();
            root.set_item("title", "T").unwrap();
            root.set_item("subtitle", "S").unwrap();
            root.set_item("caption", "C").unwrap();

            let node = composite_tree_from_py(root.as_any()).unwrap();
            let CompositeNode::Composite {
                layout,
                children,
                nrows,
                ncols,
                row_ratios,
                col_ratios,
                spacing,
                config,
                title,
                subtitle,
                caption,
                ..
            } = node
            else {
                panic!("expected Composite")
            };
            assert_eq!(layout, CompositeLayout::Grid);
            assert_eq!(children.len(), 4);
            assert_eq!(nrows, Some(2));
            assert_eq!(ncols, Some(2));
            assert_eq!(row_ratios, Some(vec![1.0, 3.0]));
            assert_eq!(col_ratios, Some(vec![2.0, 1.0]));
            assert_eq!(spacing, Some(12.0));
            let cfg = config.expect("config should round-trip");
            assert_eq!(cfg["background"], serde_json::json!("#fff"));
            assert_eq!(title.as_deref(), Some("T"));
            assert_eq!(subtitle.as_deref(), Some("S"));
            assert_eq!(caption.as_deref(), Some("C"));
        });
    }

    /// The declaration must survive the PyO3 boundary in the dict shape
    /// `composition.py`'s `_extent_alignment` writes — otherwise the Python
    /// emit is dead and the panels silently go back to laying out
    /// independently, which is a defect no test of either side alone catches.
    #[test]
    fn composite_tree_from_py_reads_extent_alignments() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mk_leaf = |idx: usize| {
                let d = PyDict::new(py);
                d.set_item("kind", "leaf").unwrap();
                d.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
                d.set_item("data", idx).unwrap();
                d
            };
            let hole = PyDict::new(py);
            hole.set_item("kind", "hole").unwrap();
            // JointChart's lowering: [top, HOLE, centre, right].
            let children = pyo3::types::PyList::new(
                py,
                [mk_leaf(0), hole.clone(), mk_leaf(1), mk_leaf(2)],
            )
            .unwrap();

            let mk_alignment = |follower: usize, leader: usize, axis: &str| {
                let d = PyDict::new(py);
                d.set_item("follower", follower).unwrap();
                d.set_item("leader", leader).unwrap();
                d.set_item("axis", axis).unwrap();
                d
            };
            let alignments =
                pyo3::types::PyList::new(py, [mk_alignment(0, 2, "x"), mk_alignment(3, 2, "y")])
                    .unwrap();

            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "grid").unwrap();
            root.set_item("children", children).unwrap();
            root.set_item("nrows", 2usize).unwrap();
            root.set_item("ncols", 2usize).unwrap();
            root.set_item("extent_alignments", alignments).unwrap();

            let node = composite_tree_from_py(root.as_any()).unwrap();
            let CompositeNode::Composite { extent_alignments, .. } = node else {
                panic!("expected Composite")
            };
            assert_eq!(extent_alignments, joint_alignments());

            // Absent is "declares nothing", never an error or a default.
            let bare = PyDict::new(py);
            bare.set_item("kind", "composite").unwrap();
            bare.set_item("layout", "hconcat").unwrap();
            bare.set_item("children", pyo3::types::PyList::new(py, [mk_leaf(0)]).unwrap())
                .unwrap();
            let node = composite_tree_from_py(bare.as_any()).unwrap();
            let CompositeNode::Composite { extent_alignments, .. } = node else {
                panic!("expected Composite")
            };
            assert!(extent_alignments.is_empty());
        });
    }

    /// An axis outside `{"x","y"}` (or a misspelled key) refuses naming the
    /// field, rather than dropping the alignment and rendering as if the
    /// caller had declared nothing.
    #[test]
    fn composite_tree_from_py_refuses_a_malformed_alignment() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mk_leaf = |idx: usize| {
                let d = PyDict::new(py);
                d.set_item("kind", "leaf").unwrap();
                d.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
                d.set_item("data", idx).unwrap();
                d
            };
            let bad = PyDict::new(py);
            bad.set_item("follower", 0usize).unwrap();
            bad.set_item("leader", 1usize).unwrap();
            bad.set_item("axis", "diagonal").unwrap();

            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "grid").unwrap();
            root.set_item("children", pyo3::types::PyList::new(py, [mk_leaf(0), mk_leaf(1)]).unwrap())
                .unwrap();
            root.set_item("nrows", 2usize).unwrap();
            root.set_item("ncols", 1usize).unwrap();
            root.set_item("extent_alignments", pyo3::types::PyList::new(py, [bad]).unwrap())
                .unwrap();

            let err = composite_tree_from_py(root.as_any()).unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("extent_alignments"), "msg: {msg}");
        });
    }

    #[test]
    fn composite_tree_from_py_unknown_kind_errors() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("kind", "spaghetti").unwrap();
            let err = composite_tree_from_py(d.as_any()).unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("spaghetti"), "msg: {msg}");
        });
    }

    #[test]
    fn composite_tree_from_py_missing_children_key_errors() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("kind", "composite").unwrap();
            d.set_item("layout", "hconcat").unwrap();
            let err = composite_tree_from_py(d.as_any()).unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("children"), "msg: {msg}");
        });
    }

    #[test]
    fn composite_tree_from_py_empty_children_surfaces_validate_error() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let children = pyo3::types::PyList::empty(py);
            let d = PyDict::new(py);
            d.set_item("kind", "composite").unwrap();
            d.set_item("layout", "hconcat").unwrap();
            d.set_item("children", children).unwrap();
            let err = composite_tree_from_py(d.as_any()).unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("hconcat") && msg.contains("empty"), "msg: {msg}");
        });
    }

    #[test]
    fn composite_tree_from_py_hole_child_in_grid() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mk_leaf = |idx: usize| {
                let d = PyDict::new(py);
                d.set_item("kind", "leaf").unwrap();
                d.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
                d.set_item("data", idx).unwrap();
                d
            };
            let hole = PyDict::new(py);
            hole.set_item("kind", "hole").unwrap();

            let children =
                pyo3::types::PyList::new(py, [mk_leaf(0).into_any(), hole.into_any()]).unwrap();
            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "grid").unwrap();
            root.set_item("children", children).unwrap();
            root.set_item("nrows", 1usize).unwrap();
            root.set_item("ncols", 2usize).unwrap();

            let node = composite_tree_from_py(root.as_any()).unwrap();
            let CompositeNode::Composite { children, .. } = node else {
                panic!("expected Composite")
            };
            assert_eq!(children.len(), 2);
            assert!(matches!(children[1], CompositeNode::Hole { .. }));
        });
    }

    #[test]
    fn composite_tree_from_py_sizeless_hole_rejected_under_hconcat() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mk_leaf = |idx: usize| {
                let d = PyDict::new(py);
                d.set_item("kind", "leaf").unwrap();
                d.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
                d.set_item("data", idx).unwrap();
                d
            };
            let hole = PyDict::new(py);
            hole.set_item("kind", "hole").unwrap();

            let children =
                pyo3::types::PyList::new(py, [mk_leaf(0).into_any(), hole.into_any()]).unwrap();
            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "hconcat").unwrap();
            root.set_item("children", children).unwrap();

            let err = composite_tree_from_py(root.as_any()).unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("hole") && msg.contains("hconcat"), "msg: {msg}");
        });
    }

    #[test]
    fn composite_tree_from_py_sized_hole_accepted_under_hconcat() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mk_leaf = |idx: usize| {
                let d = PyDict::new(py);
                d.set_item("kind", "leaf").unwrap();
                d.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
                d.set_item("data", idx).unwrap();
                d
            };
            let hole = PyDict::new(py);
            hole.set_item("kind", "hole").unwrap();
            hole.set_item("width", 40.0f64).unwrap();
            hole.set_item("height", 30.0f64).unwrap();

            let children =
                pyo3::types::PyList::new(py, [mk_leaf(0).into_any(), hole.into_any()]).unwrap();
            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "hconcat").unwrap();
            root.set_item("children", children).unwrap();

            let node = composite_tree_from_py(root.as_any()).unwrap();
            let CompositeNode::Composite { children, .. } = node else {
                panic!("expected Composite")
            };
            assert_eq!(children[1], CompositeNode::Hole { width: Some(40.0), height: Some(30.0) });
        });
    }

    #[test]
    fn composite_tree_from_py_hole_rejected_under_overlay() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mk_leaf = |idx: usize| {
                let d = PyDict::new(py);
                d.set_item("kind", "leaf").unwrap();
                d.set_item("spec", Py::new(py, dummy_chart_spec()).unwrap()).unwrap();
                d.set_item("data", idx).unwrap();
                d
            };
            let hole = PyDict::new(py);
            hole.set_item("kind", "hole").unwrap();
            hole.set_item("width", 40.0f64).unwrap();
            hole.set_item("height", 30.0f64).unwrap();

            let children =
                pyo3::types::PyList::new(py, [mk_leaf(0).into_any(), hole.into_any()]).unwrap();
            let root = PyDict::new(py);
            root.set_item("kind", "composite").unwrap();
            root.set_item("layout", "overlay").unwrap();
            root.set_item("children", children).unwrap();

            let err = composite_tree_from_py(root.as_any()).unwrap_err();
            let msg = err.value(py).to_string();
            assert!(msg.contains("hole") && msg.contains("overlay"), "msg: {msg}");
        });
    }
}
