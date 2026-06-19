# Archaeology #6/#7/#8 Remediation — Round 3 (convergence) Spec + Plan

*Date: 2026-06-19. Third pass of the autonomous review→remediate loop on branch `fix/archaeology-bugs-6-7-8-class`. Round 1 fixed the three defect classes; round 2's full re-review confirmed R1–R8 closed and the #6 pipeline clean, but surfaced three residual real issues (this doc) plus one deferred renderer limitation. Sources: `.git/sdd/round2-{rust-6,rust-7,python-8}.md` + round-2 scene-pipeline/interactive audits.*

## Scope

Close the three real issues the round-2 re-review found, all within the already-named #7/#8 classes or the #8 feature itself. Then re-run the full review+audit loop (round 4); converge when no confirmed correctness/class issue remains.

## Findings → tasks

### T1 — DensityData faceted extent (#7 class, 3rd transform) — S3 real
`fix_transform_extents_for_facet` dispatches Kde/Bin/Violin/Kde2D/Bin2D but NOT `DensityData`, which carries a value-axis `extent: Option<(f64,f64)>` (density_data.rs) and a grouped apply path. A faceted `transform_density(extent=None)` (public Python API) computes a per-panel KDE extent → cross-panel value-axis drift — the identical #7 defect. **Fix:** add `density_data::global_extent` (raw per-axis min/max, like kde — DensityData doesn't nice), dispatch it in `fix_transform_extents_for_facet` (pin only when unset). Resolve the dead `DensityData::apply_one_group` `_shared_extent` param (round-2 N2): wire it to mirror kde/violin intra-batch sharing, or remove it if DensityData has no grouped-sharing path — pick the honest option (no inert field left).
**Contract:** faceted DensityData shares one value extent across panels; non-faceted unchanged; raw (unniced).

### T2 — factory-dict chrome split for LayerChart (#8 class, factory path) — S4 real
`_overrides._apply_overrides` splits the figure-chrome keys out of `properties={...}` only when the target `isinstance(_CompositeBase)`. `LayerChart` is `_ChartLike` (single-plot overlay, intentionally not reparented), so a figure function returning a LayerChart with `properties={"title":...}` fans the title into every inner layer's `_title` (leak) and the HTML `<title>` stays default — the same bug R7 fixed for LayerChart's *chained* `.properties(title=)`. **Fix:** in `_apply_overrides`, split the chrome keys for any chart-like that intercepts them (LayerChart now has the override too), not only `_CompositeBase` — e.g. gate on "has a chrome-intercepting `.properties`" or include LayerChart. Route chrome through `target.properties(**chrome)` (intercepted), fan only non-chrome to children. Subtitle/caption keep fanning to the merged chart (LayerChart has no figure band; document `<title>` is title-only — round-2 verdict). **Also (round-2 NEW-2, S5):** consolidate the chrome-key set into one shared `_FIGURE_CHROME_KEYS` used by BOTH `_overrides` and `_CompositeBase.properties` (currently parallel hand-maintained — the smell R8(a) fixed for offset keys, reintroduced for chrome keys).
**Contract:** `properties={title/subtitle/caption=}` behaves identically to chained `.properties(...)` for every chart-like (composite AND LayerChart); no inner leak; HTML `<title>` correct; chrome-key set single-sourced.

### T3 — packed instance offset under the figure-title band (#8 interactive bug) — S3 real
`_inject_figure_chrome` shifts scene-graph mark nodes and `plot_area.y` down by `header_h`, but the **packed instance data** (`>1000`-mark circle/rect batches) is never offset — `_merge_packed_data` rewrites only `panel_idx`. So a composite with a figure title + a packed child renders its GPU marks `header_h` px above the (shifted-down) axes and clips into the title band. Scene-graph (small) children offset correctly; only the packed path is broken. **Fix:** offset the packed instances' y-coordinate by `header_h` wherever `_inject_figure_chrome` shifts a panel's nodes — add a packed-coordinate y-offset (parse the packed circle/rect instance records and add `header_h` to the y field) alongside the existing scene-node offset, so GPU marks and axes stay aligned. Confirm the per-panel GPU scissor/plot_area and the packed y use the same post-offset frame.
**Contract:** a composite figure title + a >1000-mark child renders GPU marks aligned with their axes (no `header_h` drift, no clipping into the band); small-data composites unchanged; non-titled composites byte-identical.

## Non-goals (deferred, documented)
- **Hover-tooltip for geoshape/image/label marks.** `nearest_in_batch` (WASM hover hit-test) handles only Circle/Rect; Text/Polygon/Image are skipped, so hover-tooltips don't fire for these marks even though R2/R3 now give them metadata and the CLICK path (`hit_test_batch`) handles them. Pre-existing renderer-subsystem limitation, same class as the already-deferred Text/Label WASM hit-test (parent spec §3). Extend that deferral note to cover geoshape/image hover. Not a regression; out of this loop's surface.
- W5 (Joint/ClusterMap interactive caption-y), W4 (raw-node offset), keys WASM consumer — unchanged from the parent spec §3.

## Invariants / constraints
- Extent computation in the transform layer (T1); chrome single-homed + chrome-key single-sourced (T2); packed and scene-graph marks share one post-offset frame (T3).
- No matplotlib; no global mutable state. T1/T3 may touch Rust (density transform) / Python (packed offset); T2 Python. No WASM source change expected (T3 offsets the packed bytes Python already assembles).
- Backward compat: non-faceted DensityData, non-LayerChart factory calls, small-data + non-titled composites all byte-identical.

## Acceptance
- T1: faceted DensityData shares value extent across panels (discriminating disjoint-range test); no inert field.
- T2: `properties={title=}` → figure-level for LayerChart + composites, no inner leak, HTML `<title>` correct; one shared `_FIGURE_CHROME_KEYS`.
- T3: a >1000-mark titled composite has packed marks aligned with axes (test the packed-instance y offset == header_h; assert against the scene-node offset).
- Full suite green (consistent env); per-fix fail-before/pass-after; round-4 re-review + audits surface no confirmed correctness/class issue → converged.

## Validation
- Same discipline: three-gate per task (spec → quality → review-lite), run tests directly in a consistent env (matching libpython), goldens visually inspected if added, claims-disciplined commits.
- After T1–T3: round-4 full rust+python heavyweight review + scene-pipeline + interactive audits. Loop if real issues remain.
