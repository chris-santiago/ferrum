# Follow-up: Python-layer axis sharing for grid-composed charts

**Surfaced:** 2026-05-11 during the rust-coherence-pass (F21 cleanup).
**Verified:** 2026-05-11 by reading
`crates/ferrum-core/src/render/binding.rs::compose_svg_grid_py` (pre-F21).
**Resolved:** 2026-05-12, Python-pass P2.8 (K16).  Shipped as
`_ChartLike.share_scale(x="shared", y="shared", ...)` so every
composition (HConcat / VConcat / Joint / Repeat / ClusterMap) gets
it for free.  `RepeatChart` additionally supports the same semantics
at construction time via `resolve={"x": "shared", ...}` (P2.5d).  The
sketched `Figure.shared(...)` API was discarded — ferrum has no
`Figure` class and the use cases the section below names are all
covered by the composition-level method.
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

## Proposed design (superseded)

The original sketch proposed `Chart.share_scale(other_chart, channel)`
plus a `Figure.shared(x=[(r, c), ...])` declarative API.  Both ideas
were re-examined during P2.8 implementation:

- **Pair-wise `Chart.share_scale(other, channel)`** — dropped.  Charts
  are immutable, so the call must return a tuple `(new_a, new_b)`,
  which reads asymmetrically ("a does the sharing to b") at the call
  site.  Every concrete use case flowed through a composition anyway;
  the composition-level method covers it.
- **`Figure.shared(...)`** — dropped.  Ferrum has no `Figure` class,
  and inventing one for a single method would add a sixth composition
  type.  `_ChartLike` already owns `save` / `show` / `_repr_*` for all
  five compositions, so the method lands there.

## Shipped design (2026-05-12)

`_ChartLike.share_scale(**channels)` walks every member chart (and
every layer of layered cells) via `_scale_share.compute_union_domain`,
then re-emits the composition with each chart cloned and given an
explicit `scale={"type": ..., "domain": [...]}` dict on the shared
channels.  Independent channels are no-ops.  `RepeatChart.share_scale`
overrides this to merge into its `resolve=` config so the union pass
runs once at `expand()` time (consistent with `resolve=` at
construction).

The union-domain primitive lives in `src/ferrum/_scale_share.py`:

- `compute_union_domain(charts, channel) -> dict | None` — handles
  linear / ordinal / time scale types, polars-typed.
- `inject_scale(chart, channel, scale_dict) -> Chart` — preserves
  every other ChannelBase kwarg (type_, aggregate, bin, title, ...),
  handles single-mark and layered charts symmetrically.

## Out of scope (for this follow-up)

- Coordinate-system rebinding within already-rendered SVG strings
  (the original phase-9a intent). Structurally wrong; capture removed.
