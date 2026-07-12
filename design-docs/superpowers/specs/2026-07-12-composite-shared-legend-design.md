# Composite Figure-Level Shared Legend + `Resolve(legend=)` Design Spec

GH #16 (C4-residual). Companion decision record: the coherent-change defended
choice (2026-07-12, in-session) — compositor legend band chosen over
Python keep-one/suppress-rest, merge-time node dedup, and facet-conversion.

## 1. Scope

When a composite (concat/grid/wrap) resolves a `color` or `size` scale as
shared across its leaves, the compositor renders **one figure-level legend**
built from the unioned domain and suppresses the participating leaves'
per-panel legends. Legend resolution becomes a user-controllable axis via the
spec §3.9 `Resolve` contract: it **defaults to following scale resolution**
and can be forced back to per-panel with `legend={"color": "independent"}`.
`pairplot(hue=)` and `jointplot(hue=)` get the shared legend by default.

## 2. Goals

- `pairplot(df, hue=...)` renders exactly one legend, placed outside the
  panel grid, instead of one per color-bearing cell.
- Any `HConcat`/`VConcat`/`Concat`/`RepeatChart` with
  `resolve={"color": "shared"}` (or `"size"`) renders one figure-level
  legend/colorbar for that channel; `jointplot(hue=)` opts in internally.
- `fm.Resolve(scale=..., legend=...)` is accepted wherever `resolve=` is
  accepted today; the flat-dict form keeps meaning scale resolution.
- Static SVG and interactive HTML render the same single legend (one
  mechanism: the shared `SceneGraph.legend` slot).
- Independent-resolve output is byte-identical to today.

## 3. Non-goals

- `Resolve(axis=...)` — axis sharing/dedup across composite panels is a
  separate layout feature; §3.9 gets a dated note and a follow-up issue.
- Legend resolution for channels without scale resolve support (`shape`,
  `opacity`) — their legends stay per-panel.
- `Overlay` composite nodes (LayerChart semantics): a layer stack is one
  panel; its legend behavior is unchanged. Overlay *leaves* inside a shared
  concat/grid still participate like any other leaf.
- `bind="legend"` interactive toggles across composite panels (per-chart
  runtime, orthogonal).
- Facet (`.facet()`, `catplot(col=)`) — already renders one legend via
  single-spec layout; untouched.

## 4. System behavior

**Default (legend follows scale).** For each composite node whose resolve
marks `color` (or `size`) shared:

- Every leaf that participated in that channel's domain union (i.e. received
  the shared domain; leaves with an explicit user `scale=` on the channel are
  excluded, as today) has its per-panel legend for that channel suppressed —
  no gutter space reserved, nothing drawn.
- The compositor emits one legend for the channel on that node's merged
  scene, outside the panel grid: categorical domain → swatch legend,
  continuous domain → colorbar, shared size → size legend. Content is
  identical to what any single participating panel would have shown
  (unioned domain, same scheme, same title derivation, same per-channel
  `Legend()` styling — taken from the first participating leaf in pre-order).
- Placement honors the effective legend orient (composition-level
  `configure_legend(orient=)`, else theme default `right`): right/left grow
  the figure horizontally; top/bottom vertically. The figure title band, when
  present, stacks above it exactly as it does for per-panel legends today.
- Non-participating leaves (explicit `scale=`) keep their own per-panel
  legend. A leaf whose channel legend the user disabled (`legend=None`)
  contributes nothing and stays suppressed; if **all** participating leaves
  are user-disabled, no figure legend is emitted.
- Nesting: the legend attaches at the composite node that declared the
  sharing, so a shared-color subtree inside an independent parent gets one
  legend local to that subtree.

**Explicit legend resolution.**

- `legend={"color": "independent"}` with a shared color scale → today's
  rendering: unified domain, per-panel legends, no figure legend.
- `legend={"color": "shared"}` with an independent (or absent) color scale →
  typed `ValueError` at lowering (domains differ; nothing coherent to dedup),
  matching the existing cannot-lower convention.
- `legend={"color": "shared"}` with a shared scale → same as default.

**Unchanged:** default (all-independent) composites, LayerChart,
single charts, facet charts, clustermap (single colored panel).

## 5. Architecture

All new work lives in the existing three-pass composite renderer
(`render_composite_scene`); Python only widens the `resolve=` surface.

- **Pass 1 (resolve)** already computes the per-leaf shared-domain contexts;
  participation in a shared group is observable per leaf per channel. No new
  resolution logic.
- **Pass 2 (per-leaf render)** gains a per-leaf legend-suppression signal for
  channels resolved shared (when legend resolution is not `independent`).
  Suppression acts at the layout stage — legend *inputs* are still prepared
  (entries/colorbar/title/aux and style overrides) so the compositor can
  capture the first participating leaf's non-empty bundle as the figure
  legend's content; no gutter is reserved and no nodes are drawn per panel.
  User-level `legend={"disabled": true}` keeps suppressing at prepare, so a
  disabled leaf yields an empty bundle and is skipped for capture.
- **Pass 3 (place/merge)** gains a legend-band step, the width/height
  analogue of the existing root-chrome band: measure the legend with the
  existing legend layout primitives against the merged extent, grow the
  scene on the oriented edge, append the drawn nodes to the merged scene's
  `legend` slot. Applied at each resolving node after its children are
  placed; ordered before root-chrome injection so the title band offsets it
  like any other legend node.
- **Interactive** consumes the same `SceneGraph.legend` slot as static mesh;
  no WASM-side changes.
- **Python** maps `Resolve` onto the composite node's resolve field,
  validates the scale/legend mode matrix, and threads pairplot's existing
  shared-color resolve unchanged; jointplot sets an internal shared-color
  resolve on its grid node when `hue` is given.

## 6. Canonical interfaces / data contracts

**Python public API** (new value class, exported as `fm.Resolve`):

```python
Resolve(scale=None, legend=None)
# scale:  dict[channel, "shared"|"independent"]  — channels: x, y, color, size
# legend: dict[channel, "shared"|"independent"]  — channels: color, size
```

Accepted by every `resolve=` parameter and by `share_scale()`-adjacent
construction paths that exist today. A plain dict remains valid and is
equivalent to `Resolve(scale=<dict>)`. §3.9's `Resolve(scale, axis, legend)`
is narrowed to `(scale, legend)` with a dated spec note; `axis` is filed as
its own issue.

**Wire contract** (composite node `"resolve"` field, Python → Rust):

```json
{"color": "shared", "legend": {"color": "independent"}}
```

The existing flat channel→mode entries keep meaning scale resolution. The
optional `"legend"` sub-object carries legend resolution for `color`/`size`;
an absent key means *follow the scale mode*. Rust's `CompositeResolve` gains
a corresponding optional legend sub-struct with serde defaults such that
today's payloads deserialize identically and all-default resolve still
serializes to nothing.

**Semantic rule (per channel):** effective legend resolution =
`legend[channel]` if present, else `scale[channel]`. `Shared` legend
resolution requires `Shared` scale resolution; violations raise at lowering
with a message naming the channel and both modes.

**Seam contract (compositor ↔ leaf render):** the per-leaf render accepts a
set of channels whose legends are compositor-suppressed, and returns —
alongside the scene — the leaf's prepared legend bundle (color entries or
colorbar input, title, size/shape aux inputs, per-channel style overrides)
for those channels. Bundle content reflects the resolved (shared) domain.

## 7. Invariants and constraints

- **One mechanism per output kind:** the figure legend is built once into
  `SceneGraph.legend`; SVG and WASM both consume it. No WASM-side legend
  logic.
- **Byte stability:** any composite with all-independent legend resolution
  (including every composite that doesn't opt into scale sharing) renders
  byte-identically to today, static and interactive.
- **Facet parity:** the figure legend must visually match what the same
  data would produce on a faceted single chart (same measurement, symbols,
  fonts, orient handling) — it reuses the same legend build/measure/draw
  primitives, not a parallel implementation.
- **Explicit per-chart scale wins** (spec §6): leaves excluded from the
  domain union are also excluded from legend suppression and keep their own
  legend.
- **No silent drops:** an unsatisfiable legend resolution raises a typed
  `ValueError`; it never falls back to per-panel rendering silently.
- **Goldens are not blessed until visually inspected** (CLAUDE.md): every
  new/regenerated golden is rasterized and inspected before commit.
- New public symbol `fm.Resolve` is added to `__all__`, homed by
  `scripts/gen_api_pages.py`, and cross-linked from the composition docs.

## 8. Key decisions and tradeoffs

1. **Compositor legend band, not Python leaf mutation** *(defended choice,
   decision-only coherent-change run, 2026-07-12).* Rejected: injecting
   `legend=None` on all-but-one leaf during lowering (surviving legend sits
   in one panel's gutter, shrinking that panel and mis-placing the legend);
   merge-time scene-node dedup (post-layout, can't reclaim reserved gutters,
   brittle node comparison); converting pairplot to facet (repeat cells
   encode different x/y fields — facet cannot express them). Chosen approach
   reuses the facet "one legend reserved at figure scope" model and the
   root-chrome band placement idiom.
2. **Legend resolution defaults to following scale resolution** — matches
   Vega-Lite's default and makes #16's fix automatic for `pairplot(hue=)`
   with zero API change; the explicit axis exists purely as an opt-out.
3. **Suppress at layout, capture from prepare.** The compositor suppresses
   panel legends *after* inputs are prepared so the figure legend's content
   is produced by the same code path as panel legends (guaranteed-identical
   content), while user-level `disabled` keeps its earlier prepare-stage
   suppression semantics.
4. **Legend content and styling come from the first participating leaf
   (pre-order) with a non-empty bundle.** All participating leaves share the
   unioned domain, so bundles agree on entries; taking the first makes
   per-channel styling deterministic. Cross-leaf styling conflicts are not
   reconciled — composition-level `configure_legend` is the supported way to
   style a shared legend.
5. **`shared` legend over `independent` scale is an error, not a union.**
   Deduping legends whose domains differ would fabricate a mapping no panel
   uses. Typed error mirrors the existing unlowerable-composition convention.
6. **JointChart/ClusterMap keep no user-facing `resolve=`** (their panel
   alignment is fixed geometry); `jointplot` sets the shared-color resolve
   internally on the grid node it builds when `hue` is given.
7. **`Resolve(axis=)` deliberately excluded**; §3.9 narrowed with a dated
   note plus a follow-up issue, honoring the never-silently-drift rule
   without pulling an unrelated layout subsystem into #16.

## 9. Acceptance criteria

Static SVG unless stated; each golden visually inspected.

1. `pairplot(df, hue=)` (categorical): exactly one legend group in the SVG,
   right of the grid; all panels equal-sized; colors consistent panel-to-
   panel. Regression test proven RED against current main.
2. `hconcat`/`vconcat`/`ConcatChart` with `resolve={"color": "shared"}`:
   one legend; with default resolve: byte-identical to today (per-panel
   legends remain).
3. Shared **continuous** color: one figure colorbar with the unioned extent.
4. `resolve={"size": "shared"}`: one figure size legend; combined
   color+size sharing stacks both in one band (same-field color+size merge
   behavior preserved).
5. `Resolve(scale={"color": "shared"}, legend={"color": "independent"})`:
   unified domain, per-panel legends — matches today's shared-scale output.
6. `Resolve(legend={"color": "shared"})` without shared color scale: typed
   `ValueError` naming the channel and modes.
7. Leaf with explicit `Color(..., scale=...)` inside a shared composite:
   that leaf keeps its own legend; others are deduped.
8. Leaf with `legend=None` participates in the domain union but never in
   capture; all-leaves-disabled emits no figure legend.
9. `jointplot(df, hue=)`: one legend. `clustermap`: unchanged.
10. Nested composite: sharing declared on an inner node renders one legend
    attached to that subtree, outer panels untouched.
11. Figure title + shared legend co-render: title band above, legend
    offset correctly (no overlap).
12. `pairplot(hue=, markers=)`: shape glyphs still collapse into the single
    color legend (per-leaf same-field collapse preserved at figure level).
13. Interactive export of case 1: one legend in the rendered HTML, verified
    by headless capture; panel content unaffected.
14. Full suites green: `uv run pytest -n auto`, `cargo test`; API pages
    regenerated with `fm.Resolve` homed.

## 10. Validation strategy

- **Behavioral tests** at the seam: Rust unit tests for the resolve-mode
  matrix (follow/override/error), leaf suppression sets, band geometry
  (scene growth per orient), and capture-from-first-nonempty; Python tests
  for `Resolve` normalization, wire shape, error surfaces, and
  pairplot/jointplot defaults. Legend-count assertions on SVG output
  discriminate one-figure-legend from N-panel-legends.
- **Byte-stability guard:** existing composite goldens with independent
  resolve must not change; the shared-resolve goldens change deliberately
  and are re-blessed via `regen_and_verify` + PNG inspection.
- **Interactive:** headless WASM capture of the pairplot case, compared
  before/after for legend count and panel geometry.
- **RED proof:** the pairplot one-legend regression test runs against
  unpatched main to confirm it fails there.

## 11. Open questions

None blocking. (Overlay-leaf participation inside shared concats is defined
by §4; the pre-existing question of duplicate legends *within* a plain
LayerChart stack is out of scope and untouched.)
