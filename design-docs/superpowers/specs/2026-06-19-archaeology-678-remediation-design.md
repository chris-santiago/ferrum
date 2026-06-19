# Archaeology #6/#7/#8 Heavyweight-Findings Remediation — Design Spec

*Date: 2026-06-19. Follows the #6/#7/#8 class-fix (branch `fix/archaeology-bugs-6-7-8-class`). Source: the full heavyweight review + audit pass (`.git/sdd/heavyweight-{rust-6,rust-7,python-8}.md` + scene-pipeline/interactive audits). Companion post-mortem: `.claude/output/2026-06-19-session-postmortem-agentic-coding.md`.*

## 1. Scope

Remediate every issue the heavyweight pass surfaced over the #6/#7/#8 fix surface, as a deliberate spec-first effort rather than reactive patching. The pass confirmed #8 and N1 are genuinely closed but found residual **class-gaps**: a #6-class builder missed by the migration (with a confirmed data-corruption bug), the #6 structural guard not covering its own alignment vector, three marks that silently drop metadata, the #7 class unclosed for 2-D transforms, a dead field this effort introduced, and several cohesion/drift hazards. This spec defines the complete remediation surface and the contract for each fix.

## 2. Goals

- **R6 (metadata reaches every node, in lockstep):** every mark builder that emits nodes pairs each node with its source row AND attaches user-set tooltip/href/description; `nodes.len() == data_indices.len()` is enforced by construction for all builders.
- **R7 (faceted shared-extent is complete):** the faceted extent pin covers 2-D transforms (`Kde2D`/`Bin2D`) as well as 1-D; no transform carrying an extent field drifts per-panel when faceted. No inert/misleading extent field remains; the bin extent/nice logic has one source.
- **R8 (composite chrome cohesion):** the figure-title contract holds for every composite including `LayerChart`'s HTML `<title>`; structural parity (offset key-sets) cannot silently drift; no over-broad exception handling.
- **Honesty:** items that require a separate subsystem redesign are recorded as explicit, named limitations — never silently dropped or overclaimed as fixed.

## 3. Non-goals (deferred with rationale, recorded as known limitations)

- **W5 — Joint/ClusterMap interactive caption-y body-layout parity.** The interactive nonuniform-grid uses native panel sizes; SVG uses ratio-viewBox scaling, so the composed body heights differ. Title/subtitle parity is exact; only the caption baseline diverges. A true fix is a W5 interactive grid-layout change, out of this remediation's surface. Tracked as W5 (CLAUDE.md).
- **Text/Label tooltip hit-testing in the WASM path.** `hit_test.rs` matches only `Circle`/`Rect`; Text/Line nodes are not hit-tested interactively (SVG path renders them correctly). Pre-existing renderer-design limitation in a different subsystem.
- **`keys` channel WASM consumer.** `batch.keys` is built and serialized but no node-indexed `getKey` consumer exists yet (the "Key encoding interactive-only" archaeology item). The guard will cover `keys` length (R6.3) so it is *safe by construction* when a consumer lands, but wiring the consumer is out of scope.
- No change to the packed binary format, the `data_indices` cross-filter/linking semantics, or matplotlib-free / no-global-state constraints.

## 4. System behavior

- A `mark_label(leader_line=True)` chart with a selection/crossfilter/`key=` encoding highlights and links the correct rows (today: misaligned because the leader Line node has no `data_indices` entry).
- `mark_geoshape`, `mark_label`, `mark_image` with a `tooltip=`/`href=`/`description=` encoding show that metadata on hover (today: silently dropped).
- A faceted 2-D density / contour / 2-D-binned heatmap renders every panel on the same x AND y value extents (today: per-panel drift).
- A non-faceted violin split into groups (hue), with shared extent requested, renders all groups on one comparable value axis like KDE/histogram do (today: `shared_extent` is ignored).
- `LayerChart(...).properties(title="T")` sets the interactive/HTML document `<title>` to `T` (today: falls back to the default string).

## 5. Architecture

**R6 — Rust render.** `label.rs` migrates to the `MarkNodes` accumulator, emitting the leader-line `Line` and `Text` as a `push_many([text, line], row)` pair so both map to the row. `geoshape.rs`/`image.rs` (and `label.rs`) call `build_metadata_for_indices(&data_indices)` instead of hardcoding `None`, reusing the indices they already track. The construction-seam guard (`scene_build.rs` + `mark_nodes.rs`) gains `data_indices` and `keys` length checks alongside the existing three metadata channels, so the root `nodes.len() == data_indices.len()` invariant is asserted for every batch (this is what would have caught the label bug).

**R7 — Rust transform.** `Kde2D`/`Bin2D` expose 2-D `global_extent` helpers (4-tuple `(x_lo,x_hi,y_lo,y_hi)`, Bin2D niced per axis like 1-D Bin) in their transform modules; `fix_transform_extents_for_facet` dispatches over them in addition to the 1-D trio. `ViolinSpec::shared_extent` is wired into `violin::apply` to mirror `kde::apply_grouped` (when set and multiple groups exist, compute and pin the cross-group global extent), making the field load-bearing. The triplicated bin cast→clean→fold→nice logic is extracted into shared `bin` module helpers used by `apply_one_group`, `apply_grouped`, and `global_extent`.

**R8 — Python composition.** `LayerChart` resolves its document title from the title it was given (`.properties(title=)` or ctor), not from a child-fanned `_title`. The panel-node offset key-set used by `_inject_figure_chrome`, `_merge_scene_panels`, and `_merge_one_child` is consolidated into one shared definition so the three paths cannot drift. The `_overrides._apply_overrides` try/except is narrowed to wrap only the rebuild call. The internal figure-chrome payload becomes a `TypedDict`.

## 6. Canonical interfaces / data contracts

- **Alignment guard (extended):** for every mark batch, `nodes.len()` must equal the length of each present per-node channel — `tooltips`, `hrefs`, `descriptions`, **`data_indices`**, and **`keys`**. The seam guard asserts all five; `data_indices`/`keys` are the additions.
- **2-D extent contract:** `Kde2D::global_extent`/`Bin2D::global_extent` return `Option<(f64,f64,f64,f64)>` over the full pre-facet batch; `Bin2D` nices each axis to match its grouped bin edges, `Kde2D` returns raw per-axis min/max. The faceted pin sets the spec's `extent_x`/`extent_y` (or 4-tuple `extent`) only when unset (never clobbers a user value).
- **Violin shared_extent:** when `shared_extent == true` and the batch has >1 group, every group's internal KDE evaluates on the cross-group global extent; when false, per-group extent (today's behavior). Mirrors `KdeSpec`/`BinSpec`.
- **Offset key-set:** a single shared list of panel-node keys (`plot_area`, `clip`, `marks`/`nodes`, `axes`, `grid`, `annotations`, `strip_title`) consumed by every offset path; adding a panel node type updates one place.

## 7. Invariants and constraints

- Node-order metadata remains the single canonical convention; no consumer remaps tooltips via `data_indices`.
- `data_indices` has exactly one entry per emitted node (multi-node shapes repeat the row); now guard-enforced.
- Extent computation stays in the transform layer; `prepare.rs` orchestrates. No render-layer extent math.
- Bin extent/nice logic has exactly one expression site.
- Figure-chrome stays single-homed; no per-class copy; LayerChart's fix must not reintroduce inner-panel title leakage.
- No WASM source change (the guard additions and 2-D pin are Rust-core/transform; the interactive title already works). No packed-format change. No matplotlib, no global mutable state.
- Backward compat: every chart already rendering correctly (incl. all #6/#7/#8 fixes just landed) renders byte-identically; the additions only *add* correct metadata/extents where they were missing.

## 8. Key decisions and tradeoffs

- **Spec-first remediation, not reactive patching** (user decision). The findings are treated as a defect surface to be scoped completely, mirroring the original effort — the post-mortem's core lesson applied to its own follow-ups.
- **Violin `shared_extent`: wire, not remove.** Wiring it (symmetric with KDE/Bin) makes the Task-8 field load-bearing and closes the non-faceted multi-group violin comparability gap, vs. removing it which would lose parity. Do-it-right per CLAUDE.md.
- **2-D facet extent is in scope** (user decision: "remediate all"). It is the #7 class in 2-D; the 4-tuple shape is more work but the same defect.
- **`geoshape`/`label`/`image` metadata-drop is in scope.** Same user-visible symptom as #6 (per-row metadata not reaching nodes); cheap given correct `data_indices` already exist.
- **Guard covers `data_indices` + `keys`.** The guard must assert the alignment vector itself, not just the metadata derived from it — otherwise it cannot catch a builder (like label) that diverges `data_indices` while leaving metadata `None`.
- **W5 / WASM-hit-test / keys-consumer explicitly deferred** with named rationale (§3); not silently dropped.

## 9. Acceptance criteria

- **R6.1:** `mark_label(leader_line=True)` produces `nodes.len() == data_indices.len()` (test: multi-row leader-line batch; was 2N nodes / N indices). Selection/conditional on a leader-line label matches the correct rows.
- **R6.2:** geoshape/label/image with tooltip/href/description render `<title>`/href in SVG and carry the metadata in the scene; a per-row test asserts the correct value per node.
- **R6.3:** the seam guard trips on a deliberately `data_indices`-misaligned and a `keys`-misaligned batch (debug build); all builders pass it.
- **R7.1:** faceted Kde2D and Bin2D share x+y extents across panels (discriminating test with disjoint per-panel ranges); 1-D behavior unchanged.
- **R7.2:** a non-faceted multi-group violin with `shared_extent` spans one extent across groups; without it, per-group (regression test pins both).
- **R7.3:** bin extent/nice logic has one source (the 3 call sites delegate); existing bin tests + niced-vs-raw guard still pass.
- **R8.1:** `LayerChart(...).properties(title="T")` → HTML `<title>` is `T`; inner layers carry no stray title.
- **R8.2/R8.3/R8.4:** one shared offset key-set; narrowed except; TypedDict — no behavior change, existing composite tests green.
- Full suite green (`cargo test`, `uv run pytest -n auto`) in a consistent environment; per-fix regression tests proven fail-before/pass-after; goldens (where added) rasterized + visually inspected.

## 10. Validation strategy

- **Class-level + adversarial:** each fix gets a test that fails on the pre-fix code; the guard extension is the structural backstop that makes the R6 class unrepresentable. The 2-D facet and violin tests use disjoint per-panel/per-group ranges so they fail if the pin is absent.
- **Run the suite directly in a consistent environment** (build and test against the same libpython) — the prior effort's masked regression came from a dyld load-abort that hid real failures; the close-out must not rely on a per-task "green" that never executed.
- **Heavyweight re-review of the remediated surfaces** before done, plus golden visual inspection per CLAUDE.md.
- **Claims discipline:** commit messages assert only what was verified; deferred items (§3) are stated as limitations, not fixed.

## 11. Open questions

- **Kde2D/Bin2D extent field shape.** Confirm whether the 2-D specs carry a single 4-tuple `extent` or separate `extent_x`/`extent_y` (the pin and helpers must match the actual field shape). Bounded implementation check, not a design fork.
- **LayerChart title storage.** Whether to give LayerChart a `_figure_title` or resolve the document title from the merged child — pick the option that does not reintroduce inner-layer leakage and keeps LayerChart a single-plot overlay. Bounded; does not change the §6 contract.
