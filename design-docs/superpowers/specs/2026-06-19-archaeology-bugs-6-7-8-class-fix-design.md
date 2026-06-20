# Archaeology Bugs #6 / #7 / #8 — Class-Level Fix Design Spec

*Date: 2026-06-19. Source issues: GH #6 (D7), #7 (D2), #8 (D10). Companion: `.claude/output/2026-06-19-session-postmortem-agentic-coding.md`.*

## 1. Scope

Fix three code-archaeology bugs as **defect classes**, not as the single instances named in their issues. The prior attempt (deleted branch `fix/archaeology-bugs-5-8`) fixed one instance of each and passed every cheap gate while leaving the class live; this spec defines the complete surface and a structural fix for each so the class cannot silently recur. The three classes are: (#6) mark metadata/node-index misalignment in the Rust renderer, including its packed/interactive face; (#7) faceted shared-extent pinning that is currently KDE-only and single-group-only; (#8) figure-level title/subtitle/caption placement across composite charts in both the SVG and interactive paths.

## 2. Goals

- **#6:** Every mark builder that can skip rows, emit multiple nodes per row, or group rows attaches tooltip/href/description metadata to the *correct* source row in both the SVG and packed/interactive render paths. Divergence between node count and metadata length is impossible by construction, enforced by an assertion.
- **#6/N1:** The packed-instance interactive path (>1000 nodes) shows correct tooltips and correct selection matching for reordered, row-skipping, and multi-node-per-row batches — achieved by fixing #6 at the source, with no compensating index lookup in the WASM consumer.
- **#7:** Faceted histograms (`Bin`), KDE, and violins share one comparable value-axis extent across all panels, including the multi-group (hue) case. Extent computation lives in the transform layer, not the render-prepare layer.
- **#8:** `.properties(title=…, subtitle=…, caption=…)` on every composite chart places chrome on the composed *figure*, not on an inner panel, in both the SVG and interactive paths. The interactive on-canvas title band and the HTML document `<title>` both reflect the figure title. Figure-chrome handling lives in one shared base, not copy-pasted per class.

## 3. Non-goals

- **N2 (selection `stroke_width`-family scene rejection) is out of scope** — investigation could not reproduce it from source; the contract is sound end-to-end (`ChannelName::StrokeWidth` / `EncodingValue::StrokeWidth`, matching snake_case serde, conditional application, existing W7 tests). It is recorded as a non-bug, optionally filed as a `needs-repro` issue. No code change.
- No change to the packed binary format, the `data_indices` semantics as consumed by cross-filter/linking, or the WASM tooltip-decode call signature.
- No redesign of the facet partitioning mechanism itself; only where the extent pin is computed and which transforms it covers.
- No new public Python API surface beyond the already-documented `.properties()` chrome kwargs behaving correctly on composites.

## 4. System behavior

**#6 metadata alignment.** When a chart's mark skips rows (null/degenerate values), emits multiple primitives per row (point `Cross` shape → 2 line nodes/row), or aggregates rows into per-group nodes (line/area/ribbon/polygon), hovering any rendered element shows the tooltip/href/description of the *row that produced that element*. This holds identically in the static SVG output and in the interactive/WASM output, including batches that cross the 1000-node packing threshold and batches whose render order differs from data order.

**#7 facet extent.** A faceted chart whose panels use `Bin`, `Kde`, or `Violin` with auto extent renders every panel (and every hue group within a panel) on the same value-axis range, so panel-to-panel and group-to-group comparison is visually valid. A user does not see one panel's histogram stretched to a different x-range than its neighbor.

**#8 figure title.** `joint.properties(title="T")`, `clustermap.properties(title="T")`, `repeat.properties(title="T")`, and the concat composites all render `T` as a figure-level chrome band wrapping the whole composition (not on the center/heatmap/template sub-panel), honoring layout padding/anchor. The interactive export shows the same on-canvas band and sets the browser-tab `<title>` to `T`.

## 5. Architecture

**#6 — node+index accumulator (Rust, `crates/ferrum-core/src/render`).** Introduce a small accumulator type that owns scene nodes and their source-row indices together; a node cannot be added without supplying its source row. Multi-node shapes add several nodes against one row; group marks add one node against the group's representative row. Builders finalize by calling `meta.build_metadata_for_indices(&indices)` (the convention already correct in `arc.rs`) and setting `batch.data_indices` from the same index vector. The full-row `build_metadata(ctx)` path is removed from row-skipping/multi-node/group builders. Result: `batch.tooltips` is always in **node order**, which both the SVG renderer and the packed-path `get_tooltip(node_idx)` already index correctly — so N1 is resolved at the source with no WASM-side change.

**#7 — transform-layer extent pin (Rust, `crates/ferrum-core/src/transform` + render-prepare seam).** Global-extent computation moves from a hand-rolled fold in `render/prepare.rs` into the owning transform modules. Each extent-carrying transform (`Kde`, `Bin`, `Violin`) exposes a global-extent helper. The prepare seam dispatches generically over any transform carrying the `(extent, shared_extent)` pair when faceting, computing the extent across the entire pre-facet dataset for the value field and pinning it regardless of facet/group partitioning. `ViolinSpec` gains the `extent`/`shared_extent` fields it currently lacks.

**#8 — single chrome home (Python, `src/ferrum/composition.py` + `_chrome.py` + interactive scene merge).** Figure-chrome storage, `.properties()` interception, and chrome threading consolidate into `_CompositeBase`; `JointChart`, `ClusterMapChart`, `RepeatChart` inherit it (they currently extend `_ChartLike`). The SVG path threads `chrome_kwargs(merge_configure_layers(...))` into each composite's grid composition. The interactive path threads figure title/subtitle/caption through the scene-merge functions into the merged scene's title representation so the WASM renderer draws the on-canvas band; `to_html` reads a canonical title accessor on the base so the document `<title>` is correct for all composites.

## 6. Canonical interfaces / data contracts

**Metadata/node alignment contract (#6).** For any mark batch: `batch.nodes.len() == build_metadata_for_indices(&batch.data_indices).len()`, and `batch.data_indices[k]` is the source row of `batch.nodes[k]`. The accumulator enforces lockstep:

```rust
// one node, one source row
fn push(&mut self, node: SceneNode, row: usize);
// N nodes from one row (e.g. Cross → 2 nodes), all mapped to that row
fn push_many(&mut self, nodes: impl IntoIterator<Item = SceneNode>, row: usize);
```

`build_metadata_for_indices(&[usize]) -> MetadataOutput` is the sole metadata entry point for these builders; `build_metadata(ctx)` is reserved for true 1:1 no-skip builders only.

**Transform extent contract (#7).** `ViolinSpec` carries `extent: Option<(f64, f64)>` and `shared_extent: bool`, matching `KdeSpec`/`BinSpec`. Faceted pinning applies to any transform exposing this pair; the pinned extent is the niced global range of the value field over the full pre-facet batch, independent of `groupby`.

**Composite chrome contract (#8).** `_CompositeBase` stores `_figure_title/_figure_subtitle/_figure_caption`, intercepts those kwargs in `.properties()` (they never reach inner panels), and exposes a canonical title-text accessor consumed by `to_html`. Scene-merge functions accept and propagate `title`/`subtitle`/`caption` into the merged scene so the interactive on-canvas band matches the SVG band.

## 7. Invariants and constraints

- **Node-order metadata is the one canonical convention.** All render paths (SVG, packed/WASM) index metadata by node position; no path compensates with a `data_indices` remap. A `debug_assert_eq!(nodes.len(), metadata.len())` (when metadata present) guards batch construction.
- `data_indices` always has one entry per emitted node (repeated for multi-node shapes), preserving correct node→source-row mapping for all consumers.
- **#7 pin lives in the transform layer**; the render-prepare layer orchestrates but does not re-derive extents (removes the existing layering violation).
- **#8 chrome lives in exactly one base class**; no per-class copy of the figure-chrome logic.
- CLAUDE.md hard constraints hold: no matplotlib; no global mutable state; `cargo test` passes before done; goldens visually inspected (rasterize → Read PNG) before commit.
- Backward compatibility: existing correct charts (1:1 no-skip marks, non-faceted transforms, single `Chart` titles, `LayerChart`) render byte-identically where they were already correct.

## 8. Key decisions and tradeoffs

- **N1 folds into #6 (not a separate WASM fix).** The packer serializes `batch.tooltips` verbatim; correctness is determined by builder output order. Fixing builders to node-order metadata makes the existing `get_tooltip(node_idx)` correct. The investigated WASM-side `data_indices[node_idx]` lookup is explicitly *rejected*: it would patch only the packed path, leave SVG broken, and double-map once #6 lands.
- **N2 dropped as not-real** — fixing a non-bug is itself the overclaim failure mode the postmortem documents.
- **Structural over surgical** (user decision): accumulator (#6), transform-layer pin (#7), single chrome base (#8). Cost: larger blast radius across ~14 builders / 3 transforms / 7 composites. Benefit: the class becomes unrepresentable, satisfying the postmortem's "structural guard so it can't recur."
- **#8 interactive: full parity** (user decision) — on-canvas band *and* document `<title>`, not document-title-only. Avoids re-introducing the "SVG-only" silent feature-drop the postmortem flagged as a falsely-deliberate design note.
- **#7 multi-group included** — the prior fix handled only single-group facets; the global-extent-over-full-dataset approach covers multi-group/hue without special-casing.

## 9. Acceptance criteria

- **#6:** A per-builder-family alignment test (row-skip case + point `Cross` multi-node case + group-mark case) passes; each is proven to fail on current `main` and pass after. The batch-construction assertion is present and trips on a deliberately misaligned builder.
- **#6/N1:** A packed-path test (>1000 nodes, render order ≠ data order, plus a Cross batch) shows tooltips and selection field-matching aligned to source rows. No change to WASM tooltip-decode signatures.
- **#7:** Faceted `Bin` and faceted `Violin` tests (single-group *and* multi-group/hue) assert all panels/groups share the pinned value extent; KDE behavior unchanged where already correct. Goldens regenerated and visually inspected.
- **#8:** For all 7 composites (incl. `RepeatChart`), `.properties(title/subtitle/caption=…)` renders figure-level chrome in SVG; the interactive path renders the same on-canvas band and sets the HTML `<title>`; inner panels carry no stray title. Goldens regenerated and visually inspected.
- `cargo test` and `uv run pytest -n auto` green; `/regression-test` run per fix at the class level.
- No regression in previously-correct charts (byte-stable where applicable).

## 10. Validation strategy

- **Class-level, not instance-level:** tests parametrize across the full enumerated surface (every affected builder family, every affected composite), so "green" certifies the class is closed, not that one example passes. This directly answers the postmortem's "what would still be broken if every test passed?" gap.
- **Fail-before/pass-after** proven for each class via `/regression-test`, using disciplined isolation (git or single targeted edits — no over-broad scripted string-replaces).
- **Visual golden inspection** (rasterize SVG → Read PNG) for every regenerated golden, per CLAUDE.md; sanity-check SVG path counts before concluding a render is broken (resvg-py truncation caveat).
- **Heavyweight review + audits as gates** before marking done: `rust-review` on the marks/transform subsystems, `python-review` on the composition family, and the relevant scene-pipeline / interactive audits — the controls that caught the prior incompleteness.
- **Claims discipline:** commit messages assert only what was verified at the class level; no "eradicated the class" language without the class-level test to back it.

## 11. Open questions

- **Interactive figure-title representation (#8).** The exact scene-graph representation a single `Chart` uses to carry its on-canvas title into the WASM renderer must be confirmed before composites reuse it, so the merged-scene title band is built from the same mechanism rather than a parallel one. This is a bounded implementation spike, not a design fork; it does not change any contract above.

## 12. Implementation notes (2026-06-19, as-built)

Resolved during implementation; recorded here so the contract matches the code (per CLAUDE.md "spec is the API contract").

- **§6 extent contract — niced applies to Bin only.** The pinned faceted extent is the **niced** global range for `Bin` (so panels align to comparable bin edges) but the **raw** global min/max for `Kde` and `Violin` (no bins to align). §6's "niced global range" phrasing is Bin-specific; KDE/Violin pin the unrounded global extent. Guarded by `global_extent_nices_for_bin_but_raw_for_kde_and_violin`.
- **§11 spike outcome — no WASM change.** A single `Chart`'s title is built in Rust into `SceneGraph.title` (`Vec<SceneNode>`) and WASM already renders it via `collect_static(&scene.title, …)`. Composites reuse this by injecting figure-title nodes (produced by the Rust `figure_title_nodes` PyO3 helper, which shares `FigureChrome::layout` with the SVG path) into the merged scene's `title`. No WASM edit.
- **§8 full interactive parity — caption caveat (W5).** Title and subtitle render byte-identically to SVG for **all** composites. The **caption** absolute-y matches SVG for the concat family (HConcat/VConcat/Concat); for `JointChart`/`ClusterMapChart` the caption sits relative to the interactive body, which differs from the SVG body per the **pre-existing W5 limitation** (interactive nonuniform-grid native-size vs SVG ratio-viewBox). Closing it requires a W5 body-layout fix, out of this effort's scope. Title/subtitle parity (the chrome §8 names) is exact for all composites.
