# Follow-up: Python-layer axis sharing for grid-composed charts

**Surfaced:** 2026-05-11 during the rust-coherence-pass (F21 cleanup).
**Verified:** 2026-05-11 by reading
`crates/ferrum-core/src/render/binding.rs::compose_svg_grid_py` (pre-F21).
**Resolved:** _open_.
**Severity:** S3 / design gap. Affects JointChart, ClusterMapChart,
RepeatChart, and any future grid-composed multi-chart figure that
wants shared x/y axes across cells.

## TL;DR

The Rust grid compositor (`compose_svg_grid`) used to accept
`share_x` / `share_y` parameters — lists of cell-index groups whose
axes the user wanted aligned. The Rust side never honored them; F21
removed the dead parameters. Axis sharing must live in the Python
layer, before each cell renders, so the resulting SVGs come out
already aligned.

## Why axis sharing can't live in the compositor

The Rust grid compositor sees opaque SVG strings. By the time a cell
arrives at the compositor:

- Its scale resolution has already happened (domain → pixel range).
- Its tick labels are baked into the SVG.
- Its plot area / margins / axes are at fixed coordinates inside the
  cell's viewBox.

Sharing an x-axis across two cells means giving them the SAME domain
and the SAME pixel range. That decision has to happen during
`scale_resolve::resolve_scales_with_outputs` — i.e. before either
cell is rendered. The compositor has no scale metadata to enforce
sharing against; the dead `share_x` / `share_y` parameters were
inherited from a phase-9a placeholder that never got wired through.

## What ferrum has today (post-F21)

- **JointChart**: marginals share their data axis with the centre via
  `axis(show=False)` suppression at the spec level. The Rust
  compositor scales the marginal cell into its narrow strip via
  `preserveAspectRatio="none"`, so the data axis stays aligned with
  the centre by construction. The non-shared marginal axis is
  suppressed (count/density on a thin strip is illegible).
- **ClusterMapChart**: dendrograms are pre-resized via
  `properties(width=hm_w, height=dendro_h)` and have axes suppressed.
  Alignment is structural — the dendrogram tree leaves correspond
  one-to-one with the heatmap row/column ordering.
- **RepeatChart**: cells use the same encoding spec by construction;
  axis ranges match because the underlying data fields match.

So the current grid combinators don't need explicit axis sharing —
each handles its alignment via spec-level construction.

## What's missing

A general API for "make these N cells render with identical axes."
Use cases:

1. **Faceted small multiples** with a free-but-shared scale.
2. **Cross-chart annotation** where two unrelated charts in a grid
   should share an axis for visual comparison.
3. **A heatmap + a per-column boxplot strip** where the column axis
   should match exactly.

Today users would have to compute the shared domain in Python and
pass it as an explicit `scale=fr.LinearScale(domain=[...])` to every
participating cell.

## Proposed design

`Chart.share_scale(other_chart, channel="x"|"y"|...)` that pre-renders
the union extent and injects `scale=fr.LinearScale(domain=union)` into
both specs, then a `Figure.shared(x=[(r, c), ...])`-style declarative
API for grid-laid-out charts. Belongs in ferrum's Python composition
layer, not the Rust compositor. Out of scope for the coherence pass.

## Out of scope (for this follow-up)

- Coordinate-system rebinding within already-rendered SVG strings
  (the original phase-9a intent). Structurally wrong; capture removed.
