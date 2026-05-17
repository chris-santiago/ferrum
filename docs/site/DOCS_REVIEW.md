# Docs Site Review — Practitioner Perspective

> Five parallel agents reviewed `docs/site/` as first-time users, mid-level practitioners,
> API-reference consumers, library evaluators, and ML engineers. This report consolidates
> their findings.
>
> Date: 2026-05-17

---

## HIGH Severity (blocks users or documents non-existent behavior)

### H1. First tutorial requires undeclared dependency
- **File:** `getting-started/first-plot.md` lines 10–13
- **Problem:** The very first code example imports `from sklearn.datasets import load_iris`, but the base install (`pip install ferrum-viz`) does not include sklearn. A user following the "lean install" path in `install.md` hits `ModuleNotFoundError` on the first page they try.
- **Fix:** Use synthetic data (e.g. `polars.DataFrame(...)`) OR add a prominent prerequisite callout stating the tutorial requires `pip install ferrum-viz[all]`.

### H2. Wrong package name in interactive guide
- **File:** `guide/interactive.md` line 17
- **Problem:** Says `pip install ferrum[jupyter]`. The PyPI package is `ferrum-viz`. Users get "package not found."
- **Fix:** Change to `pip install ferrum-viz[jupyter]`.

### H3. No API page for 24 Visualizer classes
- **File:** `api/` directory + `zensical.toml` nav
- **Problem:** `ROCVisualizer`, `CalibrationVisualizer`, `ModelSource`, `ComparedModelSource`, and 20+ other public classes from `ferrum._diagnostics` have no entry in the API Reference nav. Users of the OO `.fit()/.show()` workflow have nowhere to go.
- **Fix:** Add an `api/visualizers.md` with `::: ferrum._diagnostics.visualizers` directive and a `ModelSource` section.

### H4. No marks reference page
- **File:** `api/` directory
- **Problem:** 20+ `mark_*` methods are the primary user-facing API for choosing chart type, but they're only discoverable buried inside the `Chart` class page. A user searching "how do I make a violin plot" has no landing page.
- **Fix:** Add `api/marks.md` that documents each mark method with its unique parameters (or at minimum, a table with links into the Chart page).

### H5. Changelog/version mismatch
- **File:** `changelog.md` vs `src/ferrum/__init__.py` line 252
- **Problem:** Changelog documents v0.9.0 (dated 2026-05-17) with Phase 12 features. `__version__` still reads `"0.8.2"`. Users running `fm.__version__` see a stale number.
- **Fix:** Bump the version string in `__init__.py` to match.

---

## MEDIUM Severity (causes confusion or is misleading)

### M1. `type_` vs `type` parameter inconsistency
- **File:** `guide/recipes.md` line 391 vs `guide/marks-encodings.md` line 119
- **Problem:** Recipes uses `fm.X("size", type_="N")` — incorrect at the call site (should be `type="N"`). The marks-encodings guide correctly uses `type="Q"`. The mismatch teaches contradictory idioms and the wrong one silently fails.
- **Fix:** Change recipes to use `type="N"`.

### M2. Unimplemented marks listed as available
- **File:** `guide/marks-encodings.md` line 309+
- **Problem:** `mark_label` and `mark_arc` appear in the primitives table but both raise `NotImplementedError` at runtime.
- **Fix:** Remove them from the table or add a "planned" badge.

### M3. Interactivity page oversells WASM availability
- **File:** `guide/concepts/interactivity.md`, status table
- **Problem:** Every row says "Shipping ✓" including WASM/GPU renderer and linked views. Doesn't mention that WASM requires a separate `wasm-pack build` step — it is NOT included in `pip install`.
- **Fix:** Add a "Deployment requirements" note clarifying: Jupyter via anywidget (pip-installed), standalone HTML via `.save()`, full WASM interactive requires build step.

### M4. Top-level API page will be unreadable
- **File:** `api/ferrum.md`
- **Problem:** The `::: ferrum` directive renders 130+ symbols on a single page with no grouping — an overwhelming wall of text.
- **Fix:** Add `members` filtering or replace with a curated overview that links to submodule pages.

### M5. Hand-written `scales.md` without type stubs
- **File:** `api/scales.md`
- **Problem:** Only hand-written API page (all others use mkdocstrings). Documents 8 Rust-backed classes (`PowScale`, `SqrtScale`, `BandScale`, `PointScale`, `SequentialScale`, `DivergingScale`, `QuantizeScale`, `BinOrdinalScale`) that have no `.pyi` stubs — will drift from Rust implementation without any check.
- **Fix:** Add stubs to `_core.pyi`; convert to mkdocstrings directive.

### M6. `schemes.md` — Gradient class undocumented
- **File:** `api/schemes.md`
- **Problem:** `Gradient` is a Rust class with no `.pyi` stub. mkdocstrings will render nothing useful.
- **Fix:** Add `Gradient` stub to `_core.pyi`.

### M7. Comparison pages are one-directional
- **Files:** `comparison/seaborn.md`, `comparison/scikit-plot.md`, `comparison/yellowbrick.md`
- **Problem:** Only ferrum advantages are listed. No limitations, trade-offs, or honest "where seaborn still wins" section. Skeptical readers will distrust the entire page.
- **Fix:** Add a short "Current limitations" or "Where [library] may still be preferable" section to each.

### M8. Comparison pages link to non-existent guide pages
- **Files:** All 3 comparison pages, "Where to go next" sections
- **Problem:** Links to `../guide/figure-helpers.md`, `../guide/themes.md`, `../guide/composition.md`, `../guide/model-diagnostics.md` — these exist only on the unmerged `docs/continue` branch. Live site would show 404s.
- **Fix:** Don't deploy comparisons until the guide pages land, or remove the links.

### M9. Performance claims lack evidence
- **File:** `getting-started/why-ferrum.md` line 13
- **Problem:** "Altair breaks around 5,000 rows" (it's a configurable client-side limit, not a hard break), "matplotlib around 100,000 marks" (debatable). No benchmarks cited.
- **Fix:** Link to the performance-scale concepts page (which has real benchmarks), or soften the claims.

### M10. Unexplained parameter in first tutorial
- **File:** `getting-started/first-plot.md` line 51
- **Problem:** `mark_smooth(method="loess", groupby="species")` — `groupby` is never explained or linked to docs. User doesn't know if it's universal or mark-specific.
- **Fix:** Add a one-line note or link.

### M11. Interactive comparison oversells ease-of-use
- **File:** `comparison/seaborn.md`, interactive output row
- **Problem:** Claims WASM/GPU renderer via `.interactive()` — implies matplotlib-backend-like ease, but requires browser context and setup beyond pip install.
- **Fix:** Add qualifying note about environment requirements.

### M12. Two chaining idioms taught without explanation
- **Files:** `guide/data-transforms.md` vs `guide/recipes.md` line 445
- **Problem:** One page chains `.transform(a).transform(b)`, the other uses `.transform(a, b)`. Both work (variadic signature) but the equivalence is never stated.
- **Fix:** Add a one-line note in whichever page introduces the second form.

---

## LOW Severity (polish / quality of life)

| # | Location | Issue |
|---|---|---|
| L1 | `guide/interactive.md` | Only guide page with zero screenshots — all output described in prose. |
| L2 | `guide/concepts/` (all 6 pages) | No runnable code snippets — pure philosophy. |
| L3 | `getting-started/first-plot.md` line 117 | Hard-codes "54 marks" — will go stale. |
| L4 | `guide/concepts/performance-scale.md` | Benchmark numbers lack ferrum version or date. |
| L5 | `index.md` | Multiple links to gallery/API/guide pages — 404 risk if targets are stubs. |
| L6 | All comparison pages | Code examples wrapped in `<!--pytest.mark.skip-->` — never CI-verified. |
| L7 | `comparison/yellowbrick.md` | `ClassBalance` mapping doesn't note ferrum requires a fitted model where yellowbrick doesn't. |
| L8 | `guide/model-diagnostics.md` line 81 | Clustering helpers listed without cross-reference links (unlike peers in same table). |
| L9 | `getting-started/why-ferrum.md` line 15 | "Auto-raster and GPU rendering happen transparently" — unverifiable claim with no link. |
| L10 | `guide/concepts/dataframe-pluralism.md` | Claims modin/cuDF/dask/ibis support but no runnable example showing those paths. |
| L11 | `getting-started/first-plot.md` line 108 | "Twelve built-in themes" — count will go stale. |

---

## Priority Recommendations

### Immediate (pre-deploy blockers)
1. Fix first-plot tutorial to not require sklearn on base install
2. Fix package name typo in `guide/interactive.md`
3. Bump `__version__` in `__init__.py` to match changelog

### Before public launch
4. Add Visualizer API reference page
5. Add Marks reference page (or at minimum a discovery table)
6. Remove `mark_label`/`mark_arc` from documented primitives
7. Add deployment-requirement notes to interactivity page
8. Fix `type_` → `type` in recipes

### Quality pass
9. Add limitations sections to comparison pages
10. Fix dead guide links in comparison pages (or gate deployment)
11. Add `.pyi` stubs for scale/scheme Rust classes
12. Add `members` filtering to `api/ferrum.md`
13. Add at least one code snippet to each concepts page
