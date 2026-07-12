# Changelog

All notable changes to Ferrum are documented here.

## Unreleased

*No unreleased changes.*

## 0.20.0 — 2026-07-12

Two secondary-axis and composite-legend feature streams ([#52](https://github.com/chris-santiago/ferrum/issues/52), [#16](https://github.com/chris-santiago/ferrum/issues/16)), the dodge-by-model `compare=` diagnostics ([#42](https://github.com/chris-santiago/ferrum/issues/42)), and Band/Point range handling ([#39](https://github.com/chris-santiago/ferrum/issues/39)/[#65](https://github.com/chris-santiago/ferrum/issues/65)/[#40](https://github.com/chris-santiago/ferrum/issues/40)) land alongside a release-scoped bug sweep that consolidated 23 edge-case defects into eight remediating issues ([#69](https://github.com/chris-santiago/ferrum/issues/69)–[#76](https://github.com/chris-santiago/ferrum/issues/76)).

### Added

- `chart + SecondaryY(...)` secondary y-axis: per-layer independent y-scale slots, dual-side stacked axis bands, and a per-(panel, slot) interactive runtime for dual-axis charts ([#52](https://github.com/chris-santiago/ferrum/issues/52))
- One figure-level legend for a composition: `fm.Resolve(scale=, legend=)`, a compositor legend band for shared color/size resolve, and `pairplot`/`jointplot` rendering a single legend on `hue` ([#16](https://github.com/chris-santiago/ferrum/issues/16))
- Shared color/size resolve now spans nested composites via a leaf-span union, so an outer `resolve={"color": "shared"}` reaches leaves inside nested concats and layered overlays ([#74](https://github.com/chris-santiago/ferrum/issues/74))
- `configure_legend(orient="none")` suppresses a chart's legends and the figure-level band ([#74](https://github.com/chris-santiago/ferrum/issues/74))
- `importance_chart` / `shap_bar_chart` / `cv_scores_chart` render dodge-by-model panels under `compare=`; dodge eligibility extended to importance/shap_bar/cv_scores/text marks ([#42](https://github.com/chris-santiago/ferrum/issues/42))
- `SecondaryY` validates its `mark` against the primitive-mark set with a typed error ([#71](https://github.com/chris-santiago/ferrum/issues/71))
- Axis-tick text nodes carry an explicit slot tag on the scene wire ([#60](https://github.com/chris-santiago/ferrum/issues/60), [#73](https://github.com/chris-santiago/ferrum/issues/73))

### Fixed

- Scale constructors no longer silently fabricate placeholder ranges or accept non-finite values: `BandScale`/`PointScale`/`DivergingScale`/`SequentialScale` and the continuous scales raise typed errors on short or non-finite input (closing a `LinearScale(range=[5.0])` panic), and inverted band ranges yield non-negative bandwidth ([#69](https://github.com/chris-santiago/ferrum/issues/69))
- An explicit Band/Point pixel range is re-anchored per facet panel, so marks and ticks stay within each panel instead of pinning to one absolute window ([#70](https://github.com/chris-santiago/ferrum/issues/70))
- Independent-y conflict detection is spelling-independent — both the `LayerChart(resolve={"y": "independent"})` and `chart + SecondaryY` forms raise the same conflict under `resolve={"y": "shared"}` ([#71](https://github.com/chris-santiago/ferrum/issues/71))
- The internal column-collision rename is deterministic and never leaks a `__rhs_` sentinel into axis titles or tooltips; the original column name is shown ([#71](https://github.com/chris-santiago/ferrum/issues/71))
- An explicit tooltip on the primary layer of a layered chart no longer leaks to other layers; a scale-domain `param` declared on a layer reaches the wire so the secondary axis reflects its declared domain ([#71](https://github.com/chris-santiago/ferrum/issues/71), [#72](https://github.com/chris-santiago/ferrum/issues/72))
- Interactive dual-axis charts: hit-testing finds rescaled secondary-layer marks at their displayed position, and a single-tick secondary axis relabels under a slot rescale ([#73](https://github.com/chris-santiago/ferrum/issues/73))
- `jointplot(kind="reg")` and `jointplot(marginal_kind="box")` render instead of crashing; `pairplot`/`jointplot` accept an encoding object for `hue`; `hist`/`hex` + `hue` render one figure legend ([#75](https://github.com/chris-santiago/ferrum/issues/75))
- `compare=` diagnostics: a reserved `"base"` key raises instead of overwriting the primary model, `top_k=0` gives a legible error, infinite/NaN importances no longer corrupt the domain or ranking, and tied scores order deterministically ([#76](https://github.com/chris-santiago/ferrum/issues/76))
- Dodge selects its band axis by resolved scale kind, fixing overlapping/desynced boxes for natively-horizontal marks ([#75](https://github.com/chris-santiago/ferrum/issues/75))
- `PointScale(reverse=)` is honored at the positional resolver; categorical axis ticks sit at explicit-range band centers; a 3-element `Diverging` domain no longer truncates positional axes ([#65](https://github.com/chris-santiago/ferrum/issues/65), [#39](https://github.com/chris-santiago/ferrum/issues/39), [#40](https://github.com/chris-santiago/ferrum/issues/40))

### Changed

- Per-layer y-domain resolution is unified: one param-aware resolver feeds both axis ticks and mark placement, and the layer→slot mapping is computed once at prepare time (`YSlotPlan`) ([#72](https://github.com/chris-santiago/ferrum/issues/72))
- The nested-resolve effective-mode gate is a single shared function across the domain-union and legend-band walks; the legacy secondary-axis silo and the dead spatial-index API were removed ([#74](https://github.com/chris-santiago/ferrum/issues/74), [#52](https://github.com/chris-santiago/ferrum/issues/52))
- Diagnostic domain/ranking/field-name helpers were de-duplicated into shared functions across the `plots` builders

## 0.19.0 — 2026-07-05

### Breaking changes

- `compose_svg_horizontal` / `compose_svg_vertical` / `compose_svg_grid` removed from the public API — every composition renders through the unified Rust composite path; there is no public low-level SVG-stitching surface
- All-empty compositions (e.g. zero-row `pairplot`/`jointplot`) raise a typed `ValueError` instead of rendering a blank grid
- `JointChart.share_scale()` / `ClusterMapChart.share_scale()` raise a typed `ValueError` — their panels share axes by construction ([#53](https://github.com/chris-santiago/ferrum/issues/53) tracks native `resolve=` support)
- `LayerChart` with explicit `resolve={"x"/"y": "independent"}` raises a typed `ValueError` (overlay one-panel contract; [#52](https://github.com/chris-santiago/ferrum/issues/52) tracks per-layer scales) — previously silently ignored
- `share_scale()` now resolves at render time through the composite tree with transform-aware unions (box whisker reach, KDE support); it no longer injects explicit `scale=` dicts onto child charts

### Added

- composite spec tree types for the Rust composite render path (#45)
- per-panel LayoutScale for ratio-fitted composite cells (#45, D4a as amended)
- composite scale-resolution pass (#45, pass 1 of 3)
- thread D4b resolved-domain context through the scale seam (#45, Task 5a)
- composite render core — layout + scene passes (#45, passes 2+3)
- PyO3 composite entries + WASM baked-geometry single source of truth (#45, Task 5c)
- route HConcat/VConcat through the Rust composite render path (#45, Task 6)
- ConcatChart wrap grids through the Rust composite path — compare= rides automatically (#45, Task 7)
- per-leaf binding inputs + themed per-child panel labels (#45, Task 5d)
- residuals compare= shares axes position-wise — the #45 headline
- grid/wrap hole children on the composite wire (#45, Task 8a)
- JointChart + ClusterMapChart through the Rust composite path as ratio grids (#45, Task 8b)
- RepeatChart + LayerChart through the Rust composite path — all forms unified (#45, Task 9)
- color/size channels in the composite resolve pass (#45)
- Python composite-path gate drops ahead of legacy deletion (#45)
- root chrome config + sized linear holes on the composite wire (#45)
- delete the Python legacy composition paths (#45)
- compose_svg_* no longer exist (#45)

### Fixed

- clear ValueError for leverage-only charts on no-hat-matrix models (#44)
- per-class cumulative walk + facet for shap_waterfall per_class=True (#46)
- chart-level explicit x/y scales survive composite-mark desugar (#45)
- sanitize zero/non-finite log-domain endpoints at construction (#49)
- split render passes at scissor boundaries — wgpu 29.0.3 workaround: interactive scenes with 3+ mark-bearing panels drew only one panel's marks (#51 tracks upstream)
- full 72h-review findings burn-down — every logged item resolved; interactive renders now emit RenderWarnings like static output (#50)

### Changed

- share_scale unified onto resolve= — one sharing mechanism (#45 close)
- collapse whole-change gate findings (#45)

### Other

- per-class waterfall golden + SHAP compare= shared-x hardening (#46)
- remediation spec + plan for issues #44/#45/#46; Phase B spec, plan, and sub-decision records
- Phase B close — composite rendering documented, W4/W5 retired
- track Phase A design-review S2 deferrals
- 72h full-span review close — exclusion semantics documented, findings logged
- ruff-format the wrap_svg_with_chrome stub

## 0.18.2

*2026-06-27*

### Added

- render compare= small multiples for 4 clustering charts; refine 2 sweep-chart rejections (#35)
- render compare= small multiples for regression charts + fix pooled-residual band (#35)
- render compare= small multiples for 4 model-selection charts (#35)
- render compare= small multiples for 6 explanation charts (#35)
- add `_compose_compare` helper for multi-model small multiples (#35)
- add Quantile/Threshold ScaleSpec wire variants (SPEC-04, #38)

### Fixed

- pdp compare= must use independent per-feature x, not shared (#35)
- order-preserving ordinal scale sharing (#35)
- positional Quantile/Threshold use data extent, not domain-as-extent (SPEC-04)
- single-source `_scale_to_dict` via Rust delegation; fix Quantile/Threshold encode (SPEC-04)

### Changed

- de-duplicate compare= gate kwargs + public ComparedModelSource accessor (#35)
- single-source the scale wire form via Rust `to_scale_spec` (SPEC-04)

### Other

- ferrum-spec compare= note + representative compose goldens (#35)
- align ComparedModelSource import order in explanation.py (#35)
- design spec + plan for compare= aggregate rendering (#35)

## 0.18.1

*2026-06-25*

### Added

- None

### Fixed

- Unify cmap-vocabulary holdouts on the canonical `scheme=` keyword. (#32)
- Offset inset `<svg>` and data-anchored `<image>` raw nodes in composed interactive renders. (#34)

## 0.18.0

*2026-06-23*

This release lands the **cohesion campaign**: a 20-agent audit of the whole
codebase surfaced 193 findings (sibling drift, duplicated logic, silent-failure
seams, naming drift, dead code), and every one is now addressed across the Python
declaration layer and the Rust compute/render engine. The refactors are
overwhelmingly **byte-identical** — the full SVG/PNG golden suite is unchanged — so
the bulk of the work is invisible at the output level. The user-visible changes
below are correctness and no-silent-failure improvements, API-vocabulary
unification (old spellings keep working as deprecated aliases), and one breaking
module-path change (`ferrum._diagnostics` → `ferrum.diagnostics`).

### Added

- **`compare=` and `random_state` are now uniform across the model-diagnostic figure families.** `roc_chart`, `pr_chart`, `prediction_error_chart`, `residuals_chart`, and their siblings accept `compare={"name": model, ...}` for multi-model overlays and a `random_state=` for reproducible resampling, where before only one module exposed them. The 17 single-model-aggregate paths that cannot overlay raise a documented `ValueError` instead of silently ignoring `compare=`. (PLOT-01, XSIB-08)

### Fixed

- **Degenerate hand-authored `Bin` JSON now errors instead of silently coercing (behavior change).** A `ChartSpec.from_json` carrying a hand-built bin spec with `bin_count: 0` or a non-finite / non-positive `bin_width` previously fell back to a self-inconsistent Sturges default; it now raises. Unreachable through the `Bin` / `transform_bin` constructors (which already validate) — only hand-authored JSON is affected. (XFORM-04)

- **A mistyped data-transform key now raises instead of being silently dropped (behavior change).** The `transform_*` family (`transform_bin`, `transform_aggregate`, `transform_window`, ...) serializes to a dict that crosses into the Rust engine. Previously an unknown key on that dict — e.g. `transform_bin("x", maxbin=3)` with the `maxbins` typo, or a hand-built `{"type": "data_bin", "field": "x", "maxbin": 3}` — was dropped without warning, so the transform silently ran with defaults. Each per-transform spec now carries `#[serde(deny_unknown_fields)]`, matching the strictness the theme and per-channel paths already enforce. An unrecognized key now raises a `ValueError` that names the bad field and lists the valid ones (e.g. ``unknown field `maxbin`, expected one of `field`, `as_`, `maxbins`, `step`, `nice`, `name`, `extent` ``). Valid transforms are unchanged and render byte-identically. (SEAM-04)
- **A mistyped encoding-channel or top-level chart key now raises instead of being silently dropped (behavior change).** Completing the no-silent-drop seam, `EncodingSpec` and `ChartSpec` now carry `#[serde(deny_unknown_fields)]`. A hand-authored spec with a typo'd channel key (e.g. `{"field": "a", "typ": "Q"}` instead of `"type"`) or a typo'd top-level key (e.g. `{"marrk": "point", ...}` instead of `"mark"`) fed to `ChartSpec.from_json` now raises a `ValueError` naming the unknown field, rather than dropping it and rendering with the value silently missing. Only reachable via hand-authored JSON — ferrum's own `to_json`/`to_spec` never emits an unknown key, so every valid chart round-trips byte-identically. The one documented exception is a `scale` sub-dict: `ScaleSpec` is an internally-tagged enum whose variants flatten `ContinuousScaleCommon`, and serde cannot enforce `deny_unknown_fields` through a flattened/internally-tagged shape, so a typo'd scale key (e.g. `clammp`) is still tolerated and dropped. This structural constraint is documented on the `ScaleSpec` doc comment. (SEAM-04)
- **`Bin2D` (2-D histogram binning) now errors on a reversed or degenerate explicit extent instead of silently dropping every row (behavior change).** A reversed `extent_x=(hi, lo)` (or `extent_y`) made the in-cell row filter exclude every in-range value, so `jointplot(kind="hist")` and other 2-D histograms produced an empty grid with no error or warning. `Bin2D` now validates each explicit extent (`lo < hi`, both finite) at both construction and apply, raising a `ValueError` that names the offending axis — exactly as the 1-D `Bin` already did. Fixes a pre-existing sibling-drift between the 1-D and 2-D binners. Valid extents are unchanged and render byte-identically.

### Changed

- **Render-warning text is now a stable Display contract instead of the Rust enum's Debug shape (behavior change).** Warnings emitted during rendering (palette overflow, out-of-domain rows, empty/collapsed panels, dropped facet panels, ignored sort specs, color-range parse failures, elided tick labels, overflowed legends) cross into Python via `warnings.warn`. Previously the message text was the Rust `RenderWarning`/`LayoutWarning` *Debug* representation (e.g. `SortSpecIgnored { reason: "missing field" }`), which leaked the internal variant/field shape and was not a stable contract — any field rename silently changed the string. Each variant now has an intentional `Display` message (e.g. `sort spec could not be applied (missing field); categories fall back to insertion order`), so the user-facing warning text is a deliberate, stable sentence. Code that string-matches on warning messages should match on the new wording. SVG/PNG output is unaffected — warnings are not part of the rendered document. (SEAM-07)
- **The unused typed `Data*` transform classes were removed from the Rust extension (internal).** Every Phase-12 data transform existed twice across the Python/Rust seam: the public dict-emitting `transform_*` functions (the only path in `ferrum.__all__`) and a parallel set of typed PyO3 classes (`DataAggregate`, `DataBin`, `DataCalculate`, `DataFilter`, `DataFold`, `DataPivot`, `DataStack`, `DataWindow`) that were registered into `ferrum._core` and declared in the type stub but never imported, exported, or constructed anywhere. The dead typed classes are gone, leaving one transform-construction API (the `transform_*` functions). The `transform_*` public surface and behavior are unchanged. (SEAM-02)
- **Diagnostic visualizer `score()` now delegates to the wrapped estimator (behavior change).** The `FerrumVisualizer` base `score(X, y)` previously returned a meaningless `0.0` stub for every visualizer that did not hand-override it. It now returns `float(self.model.score(X, y))` whenever the wrapped model exposes a `.score` method, and keeps the `0.0` fallback only for genuinely no-model visualizers (rank / parallel-coordinates / class-balance / elbow, all constructed with `model=None`). This means model-backed visualizers that previously inherited the stub — `PRVisualizer`, `ConfusionMatrixVisualizer`, `ClassificationReportVisualizer`, `ClassPredictionErrorVisualizer`, `DiscriminationThresholdVisualizer`, `CalibrationVisualizer`, `CooksDistanceVisualizer`, `FeatureImportancesVisualizer`, `PCAVarianceVisualizer`, `InterclusterDistanceVisualizer`, `SilhouetteVisualizer`, `ManifoldVisualizer`, the SHAP visualizers, and the deprecated `SHAPVisualizer` — now report the estimator's own metric instead of `0.0`. `ROCVisualizer` keeps its `roc_auc_score` override; the four model-selection visualizers and the two regression visualizers shed their identical copy-pasted overrides and inherit the base. The `has_score` class flag is replaced by a **derived** read-only property: `viz.has_score` is `True` exactly when `score()` returns a real metric (i.e. the wrapped model exposes `.score`), so the flag can no longer drift from the method. Two derivation gaps in that property are now closed so it holds for the non-standard subclass shapes too: (1) `ROCVisualizer.has_score` mirrors its own `score()` condition (a model with `predict_proba` but no `.score` — a valid sklearn shape — now reports `has_score=True` and `score()` returns the AUC, where before the inherited flag said `False`); and (2) a multi-model `CalibrationVisualizer` overlay (two or more positional models, or a single dict-of-models like `{"a": m_a, "b": m_b}`) now reports `has_score=False` and `score()` returns the documented `0.0` no-single-score fallback, instead of silently scoring only the first model (positional) or returning the stub (dict). Single-model calibration is unchanged. No SVG output changes — `score()` and `has_score` are not part of rendering.
- **`annotate_text` anchor/align vocabulary reconciled (additive, non-breaking).** [`annotate_text`][ferrum.annotate_text] now accepts the canonical `anchor=` keyword in the SVG vocabulary (`"start"`/`"middle"`/`"end"`), matching [`annotation.text`][]. The former `align=` keyword (`"left"`/`"center"`/`"right"`) keeps working as a **deprecated alias**, mapped via `{left: start, center: middle, right: end}`. With neither supplied, the anchor defaults to `"middle"` (identical render to the old `align="center"` default). Supplying both `anchor=` and `align=` raises `ValueError`. Internally, the coordinate-coercion logic that was duplicated across `annotations.py` and `annotation/primitives.py` is consolidated into one home (`annotation/coords.py`), and each `annotate_*` now captures user-supplied style once and derives both the mark and the annotation primitive from it so they cannot drift. Render output is byte-identical (verified against the annotation golden suite).
- **Dodge position adjustment now accepts non-string grouping columns and signals an unresolvable grouping channel (behavior change).** `Dodge` resolves its grouping channel (the `by=` field, else the `color` encoding) through the same category-coercion the rest of the renderer uses, so an integer, float, or boolean grouping column now dodges correctly. Previously only string columns worked; a non-string `by`/color column raised an error. The two contradictory failure policies are unified: a grouping channel that cannot yield categories — a missing/typo'd named column, or an un-categorizable dtype such as a timestamp — now emits one warning (`dodge could not be applied (...); marks were not offset`) and leaves the marks un-offset, instead of either hard-crashing (non-string) or silently producing an un-dodged plot (missing column). The one remaining silent no-op is a dodge requested with no grouping channel at all (no `by` and no color), which has nothing to group by. Existing string-column charts render byte-identically. (RSUP-05)
- **Breaking (defining-module path only):** the model-diagnostics implementation package was renamed from the private `ferrum._diagnostics` to the public `ferrum.diagnostics`. The public import surface is **unchanged** — `from ferrum import ROCVisualizer`, `import ferrum; ferrum.ModelSource`, and every other `ferrum.<Name>` spelling work exactly as before. What changes is the *defining* module path: each diagnostic class's `__module__` (and any direct submodule import such as `from ferrum._diagnostics.source import ModelSource`) now reads `ferrum.diagnostics.*` instead of `ferrum._diagnostics.*`. Code that imported a diagnostic class from the canonical `ferrum` namespace is unaffected; only code reaching into the private `ferrum._diagnostics.*` path needs to update to `ferrum.diagnostics.*`. Heavy sklearn-boundary internals moved under `ferrum.diagnostics._internal`, the `visualizers/` submodules were underscore-prefixed to match `sources/`, and the visualizer surface is otherwise identical.
- **The deprecated dispatcher shims `shap_chart` and `rank_chart` are no longer in `__all__` (advertised-surface reduction, non-breaking for direct imports).** Both have long been deprecated dispatchers that warn and forward: `shap_chart(kind=...)` to `shap_beeswarm_chart` / `shap_bar_chart` / `shap_waterfall_chart`, and `rank_chart(rank=...)` to `rank1d_chart` / `rank2d_chart`. They are dropped from `ferrum.__all__` and `ferrum.plots.__all__` so the canonical split functions are the only advertised surface (`from ferrum.plots import *` and the API docs no longer list the two shims). The shims remain fully importable (`from ferrum import shap_chart, rank_chart`) and still warn-and-forward unchanged; their removal target is the next major release. The duplicated warn idiom shared by both shims was factored into one internal helper, leaving render output byte-identical. (PLOT-09)
- **Color-set vocabulary unified on `scheme=`, validated at construction (behavior change).** `scheme=` is the canonical color-palette keyword on `Color` / `Fill` / `Stroke` (with `cmap`, `continuous_palette`, `sequential`, and `diverging` as deprecated aliases), and a bogus palette name now raises at construction instead of at render time. The Rust palette registry is the single source of truth: `color.palette()` / `sequential()` / `diverging()` return the true rendered colors for the colorous-backed palettes (`viridis`, `plasma`, `magma`, `inferno`, `cividis`, `blues`, `rdbu`) instead of hand-picked approximations. (ENC-06, XNAME-02, XSIB-07, ENC-11)
- **`color.to_hex` auto-detects byte vs. unit by value range and warns on an ambiguous overshoot (behavior change).** A tuple with a component `> 1` is read as byte (`[0, 255]`), so `(1, 0, 0)` → `#ff0000`; the previous type-based heuristic is gone. A *unit* color that overshoots `1.0` from float color math (e.g. `(0.9, 0.9, 1.1)`) is still read as byte but now emits a `UserWarning` pointing at `scale="unit"` / `"byte"` to disambiguate, instead of silently producing a near-black color. Pass `scale=` explicitly to suppress. (ENC-05)
- **`apply_stack` (stacking) now accepts non-string grouping columns and warns on an unresolvable channel — matching `Dodge` (behavior change).** An integer, float, or boolean `by` / `color` grouping column now stacks correctly (was silently un-stacked), and a missing or un-categorizable grouping channel emits one warning and leaves marks un-stacked instead of silently no-op'ing. Existing string-column charts render byte-identically. (RSUP-05)
- **`format_type` is the canonical number/date format keyword on every channel.** `formatType` is honored as an alias on positional channels, fixing an asymmetry where text channels dropped `format_type` and positional channels dropped `formatType`. (ENC-03)
- **`mark_text` now emits valid SVG for `baseline="top"`, `"bottom"`, and `"alphabetic"` (behavior change).** The three text-baseline parsers were unified; those spellings previously produced an invalid `dominant-baseline` attribute, which adjusts the vertical placement of text marks that used them. (RSUP-04)
- **Annotations honor per-annotation z-ordering** (draw above or below the marks layer), and the dead `curve` flag on arrow annotations was removed. (XDEAD-03)
- **Precomputed 1-D binary gain/lift curves now rank the negative class by `1 − p`,** consistent with the precomputed roc/pr path. (DIAG-01)
- **Interactive (WASM) click-selection uses typed field comparison,** so a numeric selection co-selects `42` and `42.0`, consistent with conditional highlighting. Browser-only; no static-output change. (WASM-04)
- **numpy-array figure inputs name columns `col_0`, `col_1`, …** (the ferrum-wide convention) instead of `f0`, `f1`, …. (PLOT-03)

### Deprecated

- **`min_extent=` / `max_extent=` → `min_band=` / `max_band=`** on `Axis`, `AxisConfig`, and `configure_axis` (axis-band sizing). The old names warn and forward; "extent" is reserved for data-domain bounds. (XNAME-01)
- **`extent=` → `method=` (errorbar / errorband) and `whisker_mult=` (boxplot).** The overloaded `extent=` keyword is split by mark; old usage warns and forwards. (XNAME-01)
- **`align=` → `anchor=`** on `annotate_text`, **`cmap=` / `continuous_palette=` → `scheme=`**, and **`shap_chart` / `rank_chart`** dropped from `__all__` — see the corresponding entries above. All keep working as before.

## 0.17.1

*2026-06-20*

Closes the remaining known-gap and bug issues from the v0.17.0 review and lands a
large internal cohesion pass (full Python + Rust review → 13 findings, all
behavior-preserving). The user-visible changes are a handful of default-behavior
and channel fixes; the bulk of the release is internal refactoring that leaves
render output byte-identical (verified against the golden suite). No breaking API
changes.

### Added

- [`silhouette_chart`][ferrum.silhouette_chart] and [`pca_scree_chart`][ferrum.pca_scree_chart] gained an opt-in `subtitle=` parameter (default `None`, fully backward-compatible), matching the classification/regression chart family.
- `EncodingSpec.condition` is now a readable attribute (previously write-only at construction).

### Fixed

- **`facet(col=X)`** without an explicit `ncols` now lays panels **side-by-side** (a single horizontal row) instead of stacking them vertically, matching Altair/seaborn. `facet(row=X)`, the generic `facet(field=X)` wrap, and any explicit `ncols=` are unchanged. ([#24](https://github.com/chris-santiago/ferrum/issues/24))
- **`mark_line`** now honors the `fill_opacity` encoding channel (previously silently dropped). ([#5](https://github.com/chris-santiago/ferrum/issues/5))
- The **`shape`** encoding now honors `sort` (alphabetical, data-aware `"x"/"-x"/"y"/"-y"`, sort-field object, and explicit array). ([#26](https://github.com/chris-santiago/ferrum/issues/26))
- A **conditional-only color encoding** (base `color` unset + a `when(Color(...))` selection) now builds a categorical legend, so a `bind="legend"` toggle has entries to toggle. ([#9](https://github.com/chris-santiago/ferrum/issues/9))
- Interactive **data-label z-order** no longer flips on the first zoom/pan tick — labels keep their paint order relative to annotations/axes (the placement now uses a DOM-order-preserving update). ([#10](https://github.com/chris-santiago/ferrum/issues/10))
- Setting both `color=` and `stroke=` now emits a one-time warning that `stroke` is ignored (previously dropped silently).

### Changed

- Large internal cohesion refactor across the Python and Rust layers, all behavior-preserving (byte-identical output): a shared opacity resolver across the five core marks; one serde→Python getter helper and one appearance-channel honored-kwargs contract; `chart.py` facet machinery extracted to `_facet.py` with a type-safe `_Facet`; `prepare.rs` split into `prepare/{mod,legend,extent}`; shared figure-family annotation/facet helpers; and the symmetric concat composites deduped onto a parameterized `_CompositeBase`. No public API or render-output change.

### Known gaps

- `ShapeKind::Square` glyphs can exceed the panel clip in facets ([#25](https://github.com/chris-santiago/ferrum/issues/25), investigated as not-reproducible/cosmetic); completing the `chart.py` decomposition needs `_NamedTransform` co-relocation ([#28](https://github.com/chris-santiago/ferrum/issues/28)).

## 0.17.0

*2026-06-20*

This release lands the archaeology **#6/#7/#8** remediation to convergence (a
multi-round review→remediate loop that ended with a 5-agent sweep finding zero
in-class defects) plus two heavyweight-review cohesion refactors. The headline
behavior change: **faceted charts now share scales correctly** across every
data-driven channel, **composite figure titles render**, and the interactive
(WASM) packed-instance path's metadata/positioning bugs are fixed. The work is
predominantly behavior fixes; the new public surface is internal PyO3/transform
plumbing. All output is byte-verified against the golden suite (5774 tests).

### Added

- `MarkNodes` node+index accumulator with a 5-channel alignment guard (tooltips, hrefs, descriptions, data_indices, keys), so every mark builder emits metadata aligned to its source rows.
- `ViolinSpec` extent fields and a per-transform `global_extent` so faceted density/violin panels can pin a shared value-axis extent.
- `figure_title_nodes` PyO3 helper backing interactive composite figure titles.

### Fixed

- **Faceted shared scales.** Faceted marks now match the global axis/legend across panels for *every* data-driven channel — positional x/y (raw fields under `resolve="shared"`), categorical data-aware sort, continuous and categorical color, size, opacity, and shape. Previously each panel resolved its own domain, so a mark could render with the wrong position, color, glyph, or normalization relative to the shared legend.
- **Faceted transform extents.** `Kde`/`Bin`/`Violin`/`Kde2D`/`Bin2D`/`DensityData` and the extent-*deriving* `Hex`/`Raster`/`DataBin` transforms now pin a shared value-axis extent across panels when one is not given explicitly (previously KDE-only).
- **Composite figure titles** (`title`/`subtitle`/`caption`) render in both SVG and interactive output across all composite families; `LayerChart` titles reach the document `<title>`; packed GPU marks are offset under the title band.
- **Interactive (WASM) packed path.** Packed instances are offset by the per-panel `(dx, dy)` composition translation and the figure-title band; the tooltip string table is parsed field-by-field; conditional/crossfilter instances index from the packed-first base; and `scene_load` hardens its tooltip-scan and instance-count handling against malformed buffers.
- **Mark metadata alignment** (`bar`/`rect`/`point`/`segment`/`text`/`tick`/`rule`/group/`geoshape`/`image`): tooltips, hrefs, descriptions, data_indices, and keys now align to the true source row when rows are skipped or reordered.
- **`_core.pyi` stub parity.** Corrected `EncodingSpec`/`TimeScale`/`hat_matrix_stats` signature drift and added a programmatic live-vs-stub signature-parity test so the class cannot silently recur.

### Changed

- Cohesion refactors (behavior-preserving, byte-identical): collapsed the duplicated packed/scene merge tail into a `_PlacedChild` record + `_assemble_placed_children`; extracted `build_auxiliary_scales` to dedup the triplicated scale dispatch; deduped `share_scale` mode validation; consolidated figure-chrome handling into `_CompositeBase`; single-sourced bin extent/nice logic.
- Packed GPU wire-format is now enforced: named stride/offset consts + a producer stride test (ferrum-core) and compile-time `size_of`/`offset_of` assertions (ferrum-wasm), so any future layout drift fails the build instead of silently corrupting the interactive render.

### Packaging

- Broader wheel matrix: manylinux2014 + aarch64 + musllinux + Intel macOS, with macOS built as universal2 on Apple Silicon.

### Known gaps

- Four pre-existing, out-of-scope gaps surfaced during the review were filed as issues rather than fixed here: [#24](https://github.com/chris-santiago/ferrum/issues/24) (`facet(col=)` defaults to a single column), [#25](https://github.com/chris-santiago/ferrum/issues/25) (square-shape glyphs can exceed the panel clip in facets), [#26](https://github.com/chris-santiago/ferrum/issues/26) (shape encoding ignores `sort`), [#27](https://github.com/chris-santiago/ferrum/issues/27) (PyO3 stub fidelity nits).

## 0.16.2

*2026-06-15*

A focused patch fixing figure-level title/caption positioning on concatenated
charts and single-chart captions ([#1](https://github.com/chris-santiago/ferrum/issues/1)).
No new public API.

### Fixed

- Figure-level **title, subtitle, and caption** on concatenated charts (`a | b`, `a & b`, and `concat(...)`) and single-chart captions no longer render flush against the left edge. They now default to a 16 px left inset matching the single-chart title, and [`configure_padding(left=…, right=…)`][ferrum.Chart.configure_padding] and [`configure_title(anchor="start"|"middle"|"end")`][ferrum.Chart.configure_title] reposition them (the anchor governs title, subtitle, and caption together). Previously the chrome was pinned to `x=0` and both knobs were silently ignored at the figure level.

## 0.16.1

*2026-06-15*

This release makes several documented-but-broken features actually render. No new
public API was added; methods and parameters that previously stored their values
without effect now take effect, and silently-dropped inputs now fail loud.

### Added

- [`Chart.override(**kwargs)`][ferrum.Chart.override] now applies at render. It is a fail-loud escape hatch that injects presentation-spec values (per-axis/legend config, per-channel scales, mark style, coord, `width`/`height`) and **wins the cascade** over `configure_*`/theme. Previously the overrides were stored but never read. Unknown or misspelled paths now raise `FerrumOverrideError` with a closest-match suggestion.
- Per-channel axis and legend styling now renders. Every documented field on [`fm.Axis`][ferrum.Axis] and [`fm.Legend`][ferrum.Legend] — label/grid/domain/title colors and sizes, `orient`, `translate`, `min_extent`/`max_extent`, `tick_extra`/`tick_min_step`, `grid_opacity`, `title_orient`, `zindex`, `label_flush`, `label_overlap`; legend `symbol_size`/`symbol_stroke_width`, `label_color`, `label_limit`, `clip_height`, `row_padding`/`column_padding`, `padding`/`title_padding`, `offset` — now takes effect when set on an encoding, matching `configure_axis`/`configure_legend`. Previously most of these silently dropped. A misspelled per-channel key now raises.
- `configure_axis(...)` and `configure_legend(...)` gained the same orphan styling fields, so they can be set chart-wide too.

### Fixed

- Rotated x-axis tick labels are now end-anchored (were center-anchored and overlapped the plot) with an angle-aware gap, and the x-axis title clears them.
- `configure(axis_x=…)` per-axis styling is no longer clobbered by a general `configure(axis=…)`; a per-axis/legend `label_font_size` now sizes the reserved layout (not just the drawn text); `fm.Legend(label_color=…)` applies to continuous colorbars; `configure_title(subtitle_font_size=…, subtitle_color=…)` styling is honored.
- [`clustermap`][ferrum.clustermap] accepts the documented integer `z_score`/`standard_scale` values (`0`/`1`); they previously raised `TypeError`.

### Changed

- Per-channel axis/legend options are now typed structs with `deny_unknown_fields` (replacing an opaque pass-through map), so typos fail at render instead of silently dropping. Internal: axis style overrides were unified into one struct and the `orient` precedence moved to an `Option` sentinel.

### Deprecated

- `AxisConfig.x` / `configure_axis(x=…, y=…)` are deprecated and emit a `DeprecationWarning` — they never had any effect. Use [`Chart.axis(x=False)`][ferrum.Chart.axis] / `Chart.axis(y=False)` to show or hide an axis.

## 0.16.0

*2026-06-04*

### Added

- [`Chart.to_html()`][ferrum.Chart.to_html] returns the self-contained interactive HTML page as a string, byte-identical to what `save("out.html")` writes. It accepts the same `embed_wasm=`, `toolbar=`, and `raster=` keywords as the HTML save path. This completes the `to_*` converter family (`to_svg`, `to_png`, `to_html`).

### Changed

- Renamed the output converters `Chart.show_svg()` → [`Chart.to_svg()`][ferrum.Chart.to_svg] and `Chart.show_png()` → [`Chart.to_png()`][ferrum.Chart.to_png], establishing a clear convention across the export surface: `to_*` returns an in-memory value, [`save(path)`][ferrum.Chart.save] writes to disk, and [`show()`][ferrum.Chart.show] displays. The old `show_svg`/`show_png` names remain as deprecated aliases that emit a `DeprecationWarning` and will be removed after 0.16.0.

### Fixed

- Statistical transforms now accept **integer columns**. `mark_density` (1-D and 2-D KDE), `mark_qq`, `mark_hex`, and 2-D binning previously raised `column '<x>' must be Float64` on integer data (counts, ages, years, ranks); they now coerce numeric columns to floating point. Genuinely non-numeric columns still raise a clear error.
- [`jointplot(kind="hist")`][ferrum.jointplot] now renders a 2-D histogram instead of raising `unknown column 'bin_x_start'`. The center panel is encoded against the correct `Bin2D` output columns (`x_lo`/`x_hi`/`y_lo`/`y_hi`/`count`) so the bins tile the plane instead of collapsing.
- Removed two panic-prone internal downcasts in the 2-D binning transform; bad input now returns a normal error rather than risking a panic.

## 0.15.3

*2026-06-03*

### Fixed

- Interactive (WASM) reactive **domain-rescale** (focus+context brushing) no longer shears sibling panels or makes the overview line disappear during the brush. Each panel's marks now render with their own transform, so rescaling one panel leaves the others untouched.
- Interactive **box-zoom** (magnifying-glass tool) on a focus+context overview now rescales the detail panel instead of making it vanish. Wheel, pan, and box-zoom operate on the focus panel; a box-zoom drawn on the overview drives the detail's domain like a brush.
- Interactive **axis lines and stroked marks are antialiased** via 4× MSAA on the render pass, removing the occasional 1px axis-line seam/gap where ticks cross the baseline or adjacent facet panels meet. Silently falls back to the previous (non-MSAA) path on GPUs/browsers without 4× support — no regression there.

## 0.15.2

*2026-06-03*

### Fixed

- Interactive (WASM) line and area **stroke width is now constant under reactive domain-rescale** — the "ribbon"/variable-width artifact when a focus+context brush rescaled an axis non-uniformly is gone. Stroke width is applied in screen space and is invariant to the zoom/rescale transform. As part of this, interactive line joins are now beveled (the static SVG renderer is unchanged and keeps miter joins).

### Known limitations

- A few interactive-renderer issues remain tracked for a follow-up pass: the reactive-rescale brush-mode default, sibling panels shearing during a cross-panel rescale, and occasional axis-line segment breaks. These do not affect the static SVG renderer.

## 0.15.1

*2026-06-03*

A post-v0.15.0 audit (bug-hunters, seam auditors, and heavyweight cohesion reviews) found a cluster of incomplete-unification, sibling-drift, and silent-failure bugs in the flexibility-campaign surface. All are fixed here, each with a regression test.

### Fixed

- Finished the group-by-key unification: `transform_window`, `data_stack`, `stat_aggregate`'s join/pivot/aggregate paths now accept integer/uint/bool groupby columns (they previously collapsed every row into one group). A null group key now stays distinct from a real `0`/`false`/empty (FA-9), which also fixes a crash in `summary`/`boxplot`/`violin`/`errorband` when a groupby column contained a null
- Cross-panel reactive rescale (focus+context brushing) now zooms the detail panel **in-bounds** — the rescale is reprojected through the shared data domain instead of the source panel's pixels, which previously sent the detail marks off-screen
- Polar bars, ranged rects, and arc marks now honor the per-row `opacity`/`stroke_width`/`stroke_opacity`/`stroke_dash`/`fill_opacity`/`angle`/`tooltip`/`href`/`description` channels their sibling marks already supported
- `Axis(title=None)` and `Legend(title=None)` truly suppress the title (no phantom text node), matching the channel-level `fm.X(title=None)`
- `add_params`/`add_selection` raise a clear `TypeError` on wrong-typed arguments, and a selection colliding with a same-named parameter raises instead of silently shadowing it
- A non-finite (Inf/NaN) value in an `fm.param` domain raises a legible error naming the parameter; a `bind="legend"` selection registered via `add_params` is now wired correctly; the `ChartSpec` type stub covers all constructor arguments
- `mark_area` with an `x2` channel, and `mark_bar` with both `x2` and `y2`, now raise a clear error pointing to `mark_rect` instead of silently dropping the second extent

### Known limitations

- A single-panel reactive-rescale chart may still default to pan mode; click the box-select tool to brush (FA-17). The line "ribbon" appearance under a non-uniform rescale (FA-16) is tracked separately.

## 0.15.0

*2026-06-02*

### Added

- Reactive parameters: `fm.param`, `fm.when`, and unified `selection_interval`/`selection_point`, with `bind="legend"` toggles. Parameters drive `scale.domain`, `transform_filter`, and conditional encodings; they statically resolve to their initial value for SVG export and power a live WASM/JS runtime in interactive HTML (domain rescale, crossfilter, legend toggling)
- Polar `theta2`/`radius2` second extents with radial stacking, enabling annular wedges, coxcombs, and sunburst-style charts
- Figure-level `title`/`subtitle`/`caption` on composite charts via `.properties()`, rendered once around the composite rather than per panel
- Typed continuous scales (`LinearScale`, `LogScale`, etc.) auto-infer their domain from the data, matching the dict form
- `resolve=` on `vconcat`/`hconcat`; `pairplot` shares a single color domain across panels
- 2-D density (`kde_2d` / `mark_contour`) splits by categorical hue, drawing one contour set per group

### Changed

- **Breaking:** `transform_window(..., frame=(preceding, following))` now follows the documented Vega/Altair convention, where a negative `preceding` counts rows *before* the current row. A trailing window is `frame=(-k, 0)` (the `k` preceding rows through the current row). The sign was previously inverted, so trailing rolling aggregates silently produced all-null or forward-looking results. Pipelines that compensated for the inverted sign must drop the workaround.

### Fixed

- Continuous color scales bound through `SequentialScale`/`DivergingScale` now honor the named scheme, `domain`, and `clamp` (every scheme rendered blue before); added `reds`/`greens`/`oranges`/`purples`; diverging midpoint placed via piecewise normalization
- `mark_bar`/`mark_area` honor a second positional extent (`y2`/`x2`) for candlestick, floating, and diverging bars; `mark_bar(zero=False)` opts out of the zero-anchored y-domain
- Integer and nominal columns render on categorical channels (integer-keyed heatmaps, nominal bar `y`) and stack correctly; integer storage still defaults to quantitative
- Annotation span `label_position` (`top`/`middle`/`bottom`) is honored; unsupported `stack=` on non-stackable marks warns instead of silently dropping marks; empty facet partitions name the dropped key in the warning
- Two-way faceting (`.facet(row=, col=)`) renders a true grid with per-partition transforms, layered marks, and independent/shared scales — the row dimension was previously dropped
- Silent failures across the render pipeline now surface as warnings or errors instead of producing blank or wrong output
- `mark_line` `detail=` groups by non-Utf8 columns; axis titles use the source field name rather than the internal transform column
- `mark_violin` and `mark_area` honor color/hue instead of collapsing groups; per-layer `aggregate=` and `bin=` are no longer dropped on layered charts; `transform_top_k` aggregates integer columns instead of counting rows
- `fm.when` numeric conditionals apply instead of silently no-op'ing; radial bars stack outward and `stack=` is normalized across `x`/`y`/`theta`/`radius`
- `mark_arc` with nominal theta renders a Nightingale coxcomb; polar bars render equal full-circle angular bands; `stat_aggregate` and all stat transforms accept integer/uint/bool groupby keys; ordinal-color area legend swatches match fills; box-inner layers color-encode by hue
- `title=None` and `Axis(title="")` truly suppress axis titles; annotations anchor to categorical/ordinal axes
- errorbar/errorband compute per-hue extents instead of pooling across groups; unsupported mark shapes and stack offsets raise instead of silently defaulting
- Zoomed lines and areas stay clipped within their panel in interactive exports; two-way facet row-strip labels render on the right edge

### Other

- Expanded the docs-site Showcase with composed charts unlocked by the flexibility work; flattened the Gallery/Showcase navigation

## 0.14.0

*2026-05-31*

### Added

- `catplot(height=, aspect=)` for per-panel facet sizing, matching `displot`
- Automatic temporal scale inference from `Datetime`/`Date` dtypes (no explicit `:T` needed); microsecond/nanosecond timestamps normalized to milliseconds
- Size and shape legends, with multi-legend stacking and same-field legend merging

### Fixed

- Float ordinal domains, displot facet width, and boxen sort on pandas/pyarrow inputs
- Explicit constant stroke now beats an inherited color encoding on rule/segment marks
- 3- and 4-digit hex colors, `~e`/`~s` scientific-notation trimming, ranged-mark per-row color, and raster Y-axis orientation
- Deterministic batch selection in `locate_field` for named transform outputs
- Integer/float ordinal columns render correctly; per-panel facet sizing on `displot`
- Channel-level `axis=None` hides the axis
- Date/datetime/ISO coordinates accepted in annotations
- Color routes to stroke on line-family marks
- Value-sort and the full sort spec honored on axes and composite marks
- Order-independent color-scale merge on layered charts
- Per-channel axis `label_format` with the full d3-format grammar and chrono time formatting
- Categorical color-string ranges; rect/heatmap scheme routing

### Changed

- Boxen sort moved into the Rust `LetterValue` transform (new `sort=` kwarg); Python shim removed
- Typed ordinal `scale.range` wire format
- Unified categorical-axis sort resolution; removed the dead errorbar `y_sort`
- New `ChannelBase.option()` accessor; dropped private `_kwargs` reach-ins
- Deduplicated inferred-type application; reconstruct the promoted color channel on layer merge
- Shared fill/stroke color-resolution helpers across marks
- Colorbar tick formatting routes through the full d3-format grammar
- Extracted `build_default_categorical_scale`

### Other

- Fixed docstring parameter drift across the public API; added a docstring-drift auditor (`scripts/audit_docstring_drift.py`)
- Added a capabilities showcase docs page for power-user chart designs
- Regenerated configure goldens for corrected 3-digit hex colors

## 0.13.0

*2026-05-30*

### Added

- `ferrum.Grid` value class for theme-level gridline control: major/minor levels with per-level styling (`major_color`/`minor_color`, `major_width`/`minor_width`, dash, opacity) plus bare `color`/`width`/`dash`/`opacity` shorthand that sets both levels
- Minor gridlines on continuous axes (linear, log, time, pow, sqrt, symlog); log axes place minors at the 2–9 intra-decade multiples; categorical/discretizing axes have no minors
- Constant mark-style kwargs on composite/statistical marks — `mark_density(opacity=0.4)`, `mark_smooth(stroke_width=2)`, `mark_boxplot(fill=...)`, etc. now work like simple marks, applied to every emitted layer
- `mark_hex(stroke=, stroke_width=)` renders hex-cell borders
- Native Rust diagnostic curve kernels (ROC, precision-recall, calibration, confusion matrix, threshold sweep); raw-array (`y_true=`/`y_pred=`) diagnostics are now fully scikit-learn-free
- `SceneNode::Raw` rendering in the WASM interactive renderer (chrome-vs-data anchoring) and a dedicated `MarkBatchKind::Label` scene-graph kind

### Fixed

- Continuous-axis gridlines and ticks now coincide with the data marks they label (previously used uniform-slot placement, causing visible misalignment on continuous scales)
- WASM colorbar gradient id collision when a colorbar chart was placed in an inset (outer and inset colorbars collapsed to one id)
- Diagnostic kernels reject null-containing Arrow inputs instead of silently mishandling them
- Degenerate single-class ROC matches scikit-learn's NaN convention

### Changed

- Diagnostic chart paths (precomputed and model-backed) route through the shared Rust kernels; scikit-learn is now required only when a fitted model is passed
- Internal cohesion refactors (no API change): post-review cleanups in the scale-projection/grid code, consolidated diagnostic frame assembly into a shared `_curve_frames` module, and unified precision/recall cores in the PR kernel

### Other

- Documented `ferrum.Grid`, minor gridlines, and composite mark-style kwargs across the guides and API reference; clarified that raw-array diagnostics need no scikit-learn; regenerated guide PNGs for the gridline-coincidence rendering

## 0.12.0

*2026-05-24*

### Added

- Declarative configuration surface: `configure_axis`, `configure_legend`, `configure_title`, `configure_grid`, `configure_padding`, `configure_color`, and unified `configure()` methods on `Chart` and all composition types
- Annotation rendering module with coordinate resolution (text, arrow, rect, line, span, bracket, callout, image)
- Structural feature rendering: `SecondaryY`, `BreakAxis`, `Inset` with full composition support
- ChartConfig domain fields wired into render pipeline
- `label_padding` theme key, per-side padding support

### Fixed

- Break-axis x-label disappearance and y2 right-margin clipping
- Inset viewBox scaling and span fill_opacity
- Format-presets revenue chart uses nominal encoding for monthly bars
- Break-axis remaps same-axis ticks, SecondaryY bar marks with nominal x
- Inset recipe PNG referenced wrong image
- ColorConfigSpec.domain accepts string and float values
- Annotation text font-family rendering
- 10 bug-hunt R3 bugs (26 failing tests) and 21 bug-hunt bugs across coercion, composition, figures, interactive, scale, projection, render, and layout
- 31 bug-hunt bugs from configure field wiring

### Changed

- **Major internal refactoring (no API changes):**
  - `chart.py` split from 5,779 to 2,794 lines via `StatisticalMarksMixin` (15 methods) and `DiagnosticMarksMixin` (26 methods)
  - `configure_*` methods unified into `ConfigureMixin`, eliminating ~635 lines of duplication between `Chart` and `_ChartLike`
  - `Chart._clone` auto-generated from `__slots__` (prevents silent data loss)
  - `to_spec()` decomposed into 3 focused helpers
  - Rust `ThemeInputs` decomposed from 42 flat fields into 9 sub-structs
  - Rust `PreparedInputs` legend overrides grouped into `LegendPreparedOverrides` sub-struct
  - Rust `render_svg`/`render_scene_json` deduplicated via shared `prepare_and_layout` pipeline (fixes missing secondary-Y padding in interactive renders)
  - Rust `scale_resolve.rs` (2,205 lines) split into 6 sub-modules
  - Rust `ScaleSpec` common fields factored into `ContinuousScaleCommon`
  - Rust `Encoding::inherit_*` duplication eliminated
- Added `cargo_test` nox session with correct macOS DYLD paths

### Other

- Cohesion/complexity audit report and refactor plan
- Docs: customization guide, 7 concept pages, 12 recipes, 16 golden SVGs
- Regression tests for break-axis labels, y2 right-margin, font-family, and auditor-found bugs

## 0.11.2

*2026-05-23*

### Added

- Graduated axis label collision cascade — replaces flat→-45°→elide with wrap → font shrink → graduated rotate → tick cull → elide, keeping categorical labels legible without user intervention
- Multi-line label wrapping: snake_case, space, and camelCase labels split at natural boundaries instead of truncating
- Dynamic bottom margin: rotation-aware gutter estimation prevents rotated labels from clipping at the viewport boundary
- Theme-configurable `cull_threshold` for tick label density reduction (`Theme(cull_threshold=N)`)

### Fixed

- Y-axis title duplicated across all facet columns — now suppressed on non-leftmost columns (mirrors existing x-axis bottom-row suppression)
- Panel annotations silently dropped from SVG output — now emitted after marks, matching WASM z-order

## 0.11.1

*2026-05-22*

### Fixed

- PNG export: DPR-aware canvas with pHYs DPI metadata replaces the old DPR-upscale approach, fixing gridline artifacts in composed charts (HConcat) caused by stale pixel-snap positions after canvas resize
- PNG export: full Retina resolution via 2D offscreen canvas — wide charts clamped by GPU max texture size now export at correct @2x dimensions with proper DPI metadata
- Canvas initialization: safe three-phase init (CSS size → GPU create → DPR-clamped resize) prevents `Surface::configure` crash on wide HConcat charts exceeding GPU max texture size
- Mouse coordinate mapping uses scene dimensions instead of canvas backing store, fixing hit-test and tooltip positioning on DPR-aware canvases

## 0.11.0

*2026-05-22*

### Added

- R-tree spatial indexing (`rstar` crate) for O(log n) hit-testing on large point clouds
- Interactive toolbar: Pan, Box Zoom, Box Select, Reset, Save PNG buttons (Bokeh-style)
- `Chart.interactive(toolbar=True/False)` and `Chart.save(toolbar=True/False)` kwargs
- Auto-tooltips: interactive renders automatically inject tooltip fields from encoded channels
- `WasmRenderer.maxTextureSize()` WASM export for adaptive DPR capture
- Keyboard shortcuts for toolbar modes (P/Z/S/R/Escape)

### Fixed

- Opacity semantics: `fill_opacity` used correctly for packed batches (>1000 marks), `stroke_opacity` applied in tessellated paths/polygons, no double-apply on strokes
- Grid pixel-snap in WASM renderer eliminates sub-pixel aliasing
- Mark clip region: GPU scissor rect clips zoomed marks to panel plot area (DPR-aware)
- Tick label clipping: zoomed tick labels filtered by panel plot area
- Save PNG: adaptive DPR clamping to GPU max texture size (prevents crash on wide charts)
- Save PNG: ResizeObserver disconnected during capture (prevents race condition)
- Save PNG: SVG viewBox uses actual canvas dimensions (not stale scene dimensions)
- Annotation z-order: reference lines drawn above data marks (matching SVG renderer)
- Reset button forwards selection state to Jupyter kernel
- Cursor CSS targets SVG overlay in select/boxzoom modes
- HTML title XSS: `html.escape()` applied to `<title>` tag
- Hex color shorthand: 3-char (`#ccc`) and 4-char (`#abcd`) correctly expanded
- Malformed hex colors now emit a warning instead of silently returning black
- `_render_scene_json` collapsed into `_render_scene` (eliminated duplication)

### Changed

- `SceneCollector` accumulator replaces 8-parameter `collect_nodes`/`emit_draw_commands`
- Uniform clip vec4 removed from shaders — GPU scissor rect handles clipping (32-byte uniforms)
- `text_json.rs` extracted from `lib.rs` with deduplicated text serialization helpers
- `lib.rs` split: tooltip formatting, conditional render, transform upload moved to focused modules
- Partial GPU buffer update for conditional re-render (only re-uploads instance data)
- Dead packed-batch linear fallback removed from hit-test (spatial index covers it)
- Hand-rolled JSON replaced with `serde_json` in hit-test and tooltip paths
- `_auto_tooltips` removed from public `Chart.to_spec()` signature
- `_ChartLike.save()` uses explicit `toolbar` parameter instead of `kwargs.pop`
- `ResizeObserver` reads CSS layout size (not backing-store size)
- Transition RAF loop cancels stale animations on rapid scene changes

## 0.10.0

*2026-05-20*

### Added

- `.labs()` fluent method for post-hoc axis labels and titles
- `.xlim()` / `.ylim()` convenience methods for axis limits
- `.to_dict()` on Chart (returns parsed JSON spec as a Python dict)
- `mark_circle()` / `mark_square()` convenience methods (aliases for `mark_point(shape=...)`)
- `mark_line(point=True)` overlays points on lines (altair parity)
- `color=` and `alpha=` mark kwarg aliases (maps to `fill` and `opacity`)
- `linetype=` mark kwarg alias with named values (`"dashed"`, `"dotted"`, `"dashdot"`, `"longdash"`, `"solid"`)
- `Axis(label_map={"a": "Group A"})` categorical label remapping
- `annotate_abline(slope, intercept)` for slope+intercept reference lines
- PNG `scale=` parameter on `show_png()` and `save()` for DPI control
- PDF export via `save("chart.pdf")` (zero-dependency minimal PDF writer)
- `reverse=` on continuous positional scales
- 5 new aggregate functions: `variance`, `stdev`, `q1`, `q3`, `distinct`
- 4 KDE kernels: `epanechnikov`, `tophat`, `cosine` (plus existing `gaussian`)
- Smooth method aliases: `"linear"`, `"quadratic"`, `"cubic"`, `"log"`, `"sqrt"`
- `"|"` / `"-"` (vline/hline) point shapes
- 148 CSS named colors in theme overrides (e.g. `Theme(mark_color="steelblue")`)
- Altair migration guide (`docs/site/comparison/altair.md`)
- Pandas/polars `Series` accepted as data input
- Polars `Categorical` / `Enum` columns auto-cast to string
- Polars `Duration` columns auto-cast to int64 nanoseconds
- PyArrow `Date32` / `Date64` columns auto-cast to timestamp
- Composite mark kwargs forwarding (errorbar, boxplot, errorband, ribbon, boxen)
- Mark tick rendering for ordinal-y-only and ordinal-x-only encodings

### Fixed

- `type_="Q"` kwarg on channel constructors was silently dropped
- `count():Q` aggregate shorthand crashed (empty field rejected by Rust)
- Aggregate shorthand `mean(val):Q` missing auto-groupby from sibling channels
- `nice=True` on Scale constructors silently dropped during serialization
- `mark_smooth(method="linear")` documented but crashed (Rust only accepted `"lm"`)
- Non-Int64 integer columns failed when encoded as nominal (pyarrow/pandas users)
- Config defaults (640x480) didn't match render defaults (was hardcoded 600x400)
- Shorthand parser rejected hyphens, dots, and spaces in column names
- `ChannelName` enum missing `StrokeOpacity` / `FillOpacity` / `Angle` variants (conditional encodings on extended channels were broken)
- WASM conditional resolver missing match arms for 3 extended channels
- `mark_density(kernel=...)` never forwarded kernel to Kde transform
- `_apply_label_maps` broad `except Exception` narrowed to specific types with warning

### Other

- Plotnine/Altair migration gap audit (`design-docs/superpowers/audits/`)
- 226 new regression tests across 6 test files
- Overhaul README with hero strip, benchmarks, comparison table, and theme grid
- Embed interactive demos in docs
- Add plotnine to benchmarks, comparison table, and migration guides
- 10M-row stress tests for scale ceiling validation

## 0.9.1

*2026-05-18*

### Added

- Auto-inject selection fields into tooltip for cross-panel linking
- D3 interaction layer, SVG text rendering, and cross-panel linked selection
- Interactive HTML export with interval selection, composition support, and adapter pattern
- Nox `docs` session for docs site build verification

### Fixed

- Resolve 13 scene-pipeline audit findings across all 4 stages
- Include tooltip field names in SVG output and escape Group attrs
- Wire `strip_text_size` and `strip_padding` theme keys end-to-end
- Resolve 4 PyO3 binding audit findings (B1, B2, W1, W4)
- Bug-hunt across 5 interactive subsystems — 189 tests, 7 bugs fixed
- Resolve 9 wiring-audit findings in interactive HTML export (B1-B5, W1, W3, W6, W8)
- Deduplicate and harden interactive HTML export pipeline (Rust + Python)
- Eagerly populate `LAYER_NAME_CATALOG` from all mark modules
- Embed Inter font in shared CSS for Jupyter and HTML export
- sRGB-to-linear color conversion for correct GPU rendering
- Rebuild WASM artifacts with B4 `startTransition` fix

### Changed

- Rename audit skills/agents to prefix pattern (`audit-*`, `auditor-*`)
- Docs site: add classification meta tags, update author metadata, plain display names for API nav

## 0.9.0

*2026-05-17*

### Added

- **Phase 12: Spec completeness** — full implementation of all remaining `ferrum-spec.md` features
- **Data transforms API** — 17 transform constructors (`FilterTransform`, `AggregateTransform`, `WindowTransform`, `BinTransform`, `CalculateTransform`, `FoldTransform`, `PivotTransform`, `StackTransform`, `FlattenTransform`, `SampleTransform`, `ImputeTransform`, `DensityTransform`, `RegressionTransform`, `LoessTransform`, `QuantileTransform`, `LookupTransform`, `TimeUnitTransform`) with Rust-backed execution and expression evaluator
- **New scale types** — `PowScale`, `BandScale`, `PointScale`, `SequentialScale`, `DivergingScale`, `QuantizeScale`, `BinOrdinalScale`
- **Power/sqrt transform** — `PowScale(exponent=0.5)` for square-root position resolution
- **`Axis` and `Legend` value classes** — full configuration objects for axis and legend customization
- **`ferrum.color` and `ferrum.config` modules** — color utilities and configuration API
- **`LayerChart` and `ConcatChart`** — first-class composition types for programmatic multi-view construction
- **9 infinity domain regression tests** — continuous-x bar width, infinity filtering across mark types
- **Docs: 4 recipe pages** — transforms, ConcatChart, Axis/Legend, PowScale with rendered PNGs
- **Docs: API reference pages** — transforms, scales, color, config, axis, legend
- **Docs: data transforms guide**

### Fixed

- **Infinity domain filtering** — `±inf` values no longer corrupt scale domain auto-computation
- **Continuous-x bar width** — bars with continuous x-axis and no `x2` now auto-compute width
- **Safe Rust downcasts** — temporal type handling and Time bar rendering hardened
- **Python type annotations and dead code** — review-lite findings addressed across Python surface

### Changed

- **Design-docs reorganization** — `design-docs/` split into `architecture/` and `narrative/` subdirectories
- **Code archaeology updated** — Phase 12 items marked resolved, deferred marks inventory refreshed
- **Documentation expanded** — Phase 12 pages, cross-reference links, missing docstrings added

## 0.8.3

### Added

- pytest-xdist for parallel test execution (`-n auto`)
- 132 new tests from 5-round test-sweep campaign (scale, transform, composition, facet, theme, coord, encoding dimensions)
- Scale tests for SHAP, ICE, and PDP figure functions

### Fixed

- Multi-layer scale domain union — layers now merge domains correctly instead of using only the first layer's range
- TickLevel infinity serde — tick-level values no longer panic on serialize/deserialize with infinity bounds
- Coord system × position adjustment interaction (dodge offsets under CoordFlip)
- Figure function × data shapes gap (sweep-1)
- Missing `packed_instances` field in scale-stat test fixture
- Stale `docs/superpowers` paths updated to `design-docs/superpowers` in CLAUDE.md

### Changed

- Test-sweep skill: parallel Rust test track, coding agent dispatch enforcement

## 0.8.2

### Added

- `/test-sweep` skill — iterative combinatorial test-and-fix campaign
- `python-coder` and `rust-coder` coding agents for language-specific dispatch
- Plotly added to scatter benchmark suite
- Social card generation script with iris lmplot

### Fixed

- **Per-instance channel rendering** — `opacity`, `stroke_width`, `fill_opacity`, `size`, and `angle` encoding channels now affect SVG output for all 9 primitive marks (previously only `mark_point` had full wiring)
- **All-null data handling** — charts with entirely null/NaN columns now render an empty chart gracefully instead of crashing with `ValueError`
- **Histogram auto-groupby** — `displot(hue=)` and `jointplot(hue=)` no longer crash; the Bin transform now preserves the color-groupby column
- **RepeatPlaceholder guard** — `pairplot` mark overrides no longer crash when encodings contain unresolved repeat placeholders
- **Chained layer column-overlap** — `mark_rule` layering with overlapping column names resolved
- **Scale constructors** — `range` parameter now optional on all scale constructors
- **License packaging** — NOTICE file declared in `license-files` for sdist inclusion

### Changed

- Benchmark tables updated with Plotly results (4-library comparison)
- Documentation expanded: 25 new-user gaps addressed, API reflinks, stale gallery PNGs regenerated

## 0.8.1

### Fixed

- **Scale `range` now optional** — `LinearScale`, `LogScale`, `SymlogScale`, `TimeScale`, and `OrdinalScale` no longer require `range=` in their constructors. The renderer auto-fills from the plot-area dimensions, so users can write `fm.LogScale(domain=[100, 100000], base=10)` without pixel math.
- **`mark_rule` layering with overlapping columns** — layering a rule on a scatter (`scatter + hline`) no longer draws a line per scatter row when both DataFrames share a column name (e.g. `"y"`).
- **Chained `+` layering** — `scatter + hline + vline + label` now renders correctly. Added `Identity` transform and `inherit_non_positional` to prevent chart-level positional encoding from polluting routed layers.
- **Docs code examples** — fixed `CoordFlip` encoding order, `stroke_dash` string→list, `Stack` Int64→Float64, `to_theme_inputs_dict()`→`to_spec_dict()`, `chart.render_config()`→`chart.properties(render_config=)`, `feature_importances_chart`→`importance_chart`, `ferrum[...]`→`ferrum-viz[...]`.

### Added

- **25 new documentation sections** — type suffix explanation, position adjustments (Dodge/Stack/Jitter), axis customization (log scale, limits, reversed), smooth method table, legend control, palette cycling, shared scales, regplot, multi-model comparison, precomputed scores, output methods summary, DPI note, PDF note, interactive compatibility, CoordFlip recipe, annotation recipe, chart sizing recipe, category sorting recipe, time-series recipe.
- **13 guide PNGs** — visuals for every new section (position adjustments, axis customization, legend suppression, shared scales, regplot, multi-model ROC, CoordFlip, annotations, sizing, category order, time-series).
- **12-theme visual grid** — side-by-side comparison of all built-in themes on the same chart.
- **Complete 54-mark table** — all mark methods listed with reflinks in marks-encodings guide.
- **Consistent API reflinks** — every public symbol mentioned in prose now links to its API reference across all doc pages.
- **`Identity` transform** — Rust pass-through transform for named-output routing in layered compositions.
- **11 regression tests** — covering scale range, rule layering, and chained layer fixes.
- **PyPI metadata** — keywords, classifiers, and project URLs added to `pyproject.toml`.

### Changed

- **README** — updated install section with `ferrum-viz[all]`, added docs link, interactive rendering feature, corrected dev commands.

## 0.8.0

Grammar-of-graphics core with Rust rendering engine.

### Added

- **12 built-in themes** — Paper Ink (default), Slate Citrus, Arctic Signal, Observable, Minimal, Dark, Publication, Economist, FiveThirtyEight, Solarized Light, Solarized Dark, plus explicit `paper_ink` for derivation.
- **3 original categorical palettes** — `paper_ink`, `slate_citrus`, `arctic_signal` (8 colors each).
- **9 continuous color schemes** — `cool_blue`, `warm_ochre`, `night_blue`, `electric_lime`, `signal_blue`, `ember_orange`, `blue_to_red`, `cyan_to_amber`, `blue_to_violet`.
- **[`fm.hconcat()`][ferrum.hconcat] / [`fm.vconcat()`][ferrum.vconcat]** — top-level convenience functions for multi-chart concatenation.
- **[`Chart.axis()`][ferrum.Chart.axis]** — method for suppressing or restoring axis decorations.
- **Facet-before-transform** — faceting now partitions data before statistical transforms run, so each panel gets its own transform subset.
- **Grouped smooth** — `mark_smooth` supports group-by; no explicit per-group layering needed.
- **Continuous colorbar** — rendered alongside heatmaps and continuous-color charts.
- **Full Theme key wiring** — all `ferrum-spec.md` §3.13 keys plumbed end-to-end from Python through Rust renderer. Unknown keys raise `ValueError` at construction.
- **Model diagnostics (Phase 10)** — [`ModelSource`][ferrum.ModelSource], [`ComparedModelSource`][ferrum.ComparedModelSource], 32 visualizer classes, 31 figure-level helpers covering classification, regression, feature explanation, model selection, and clustering.
- **9 figure-level helpers** — [`displot`][ferrum.displot], [`catplot`][ferrum.catplot], [`relplot`][ferrum.relplot], [`lmplot`][ferrum.lmplot], [`residplot`][ferrum.residplot], [`pairplot`][ferrum.pairplot], [`heatmap`][ferrum.heatmap], [`clustermap`][ferrum.clustermap], [`jointplot`][ferrum.jointplot].
- **54 mark methods** — primitives, statistical, distribution, uncertainty, scale-aware, and diagnostic marks.
- **DataFrame pluralism** — polars, pandas, modin, cuDF, dask, ibis, and pyarrow all accepted via `Chart(data)`.
- **Title rendering** — `Chart.properties(title="...")` rendered in the SVG with theme-controlled typography.
- **Grid lines** — theme-controlled grid rendering with configurable color, width, dash, and opacity.
- **Legend** — categorical and continuous legends with theme-controlled positioning and styling.
- **Interactive rendering (Phase 11)** — [`Chart.interactive()`][ferrum.Chart.interactive] switches to a GPU-backed WASM renderer with selections, zoom/pan, linked views, and tooltips. Backed by `anywidget` for Jupyter integration.
- **Selection API** — [`selection_point`][ferrum.selection_point], [`selection_interval`][ferrum.selection_interval], [`selection_single`][ferrum.selection_single], [`selection_multi`][ferrum.selection_multi] for declaring interactive state. Conditional encodings via `sel.when(...).otherwise(...)`.
- **[`SelectionMark`][ferrum.SelectionMark]** — configurable brush styling for interval selections.
- **`InteractiveChart`** — anywidget-based Jupyter widget with `on_selection_change` callback and self-contained HTML export via `.save()`.
- **Scene graph renderer** — Rust-side `render_interactive` produces a SceneGraph JSON consumed by the WASM GPU renderer.
- **`compose_svg_horizontal` / `compose_svg_vertical` / `compose_svg_grid`** — low-level Rust SVG composition helpers.
- **t-SNE and UMAP in pure Rust** — [`ManifoldVisualizer`][ferrum.ManifoldVisualizer] runs both via `manifolds-rs`, no Python `umap-learn` dependency.
- **7 new diagnostic helpers** — [`classification_report_chart`][ferrum.classification_report_chart], [`class_balance_chart`][ferrum.class_balance_chart], [`cooks_distance_chart`][ferrum.cooks_distance_chart], [`prediction_error_chart`][ferrum.prediction_error_chart], [`silhouette_chart`][ferrum.silhouette_chart], [`elbow_chart`][ferrum.elbow_chart], [`manifold_chart`][ferrum.manifold_chart].
- **[`mark_label`][ferrum.Chart.mark_label]** — text labels with collision avoidance (`avoid_overlap=True`).
- **[`mark_image`][ferrum.Chart.mark_image]** — image tiles from URL fields.
- **[`RenderConfig`][ferrum.RenderConfig]** — per-chart auto-raster policy configuration (threshold, behavior, aggregate, colormap).
- **`raster=` keyword** — one-off auto-raster override on `.show()`, `.save()`, `.show_svg()`, `.show_png()`.
- **`score()` on visualizers** — all Group A visualizers implement `.score()` for sklearn-protocol compatibility.
- **`width=` on [`mark_boxplot`][ferrum.Chart.mark_boxplot]** — API symmetry with other marks.
- **`stroke`, `angle`, `fill_opacity` channels** — wired end-to-end to SVG attribute emission and WASM renderer.
- **Legend `format=` and `columns=`** — kwargs now wired through to legend rendering.
- **Packed tooltips** — field-level tooltip content sent via binary buffer for interactive performance.
- **Binary instance bridge** — GPU data bypasses JSON serialization in interactive rendering.

### Fixed

- **Facet encoding** — `facet=`, `facet_col=`, `facet_row=` now correctly partition data into panels.
- **`mark_tick` y-rug** — single-axis tick marks render correctly in both x and y orientations.
- **CoordFlip rendering** — no longer drops violin paths or boxplot rects under coordinate flip.
- **Histogram `multiple='stack'`/`'fill'`** — bin edges now align across groups.
- **[`calibration_chart`][ferrum.calibration_chart] rendering** — layering wiring gap resolved; renders correctly via `.show_svg()`.
- **Chart decomposition** — `chart.py` split into rendering, encoding helpers, and composition helpers for maintainability.

### Changed

- **`+` operator always layers** — no longer ambiguous between layering and concatenation. Use `|`, `&`, [`fm.hconcat()`][ferrum.hconcat], or [`fm.vconcat()`][ferrum.vconcat] for concatenation.
- **Default theme identity is Paper Ink** — warm cream background, blue marks, warm grid. Previously was Observable-style white background.
- **Default categorical palette is `paper_ink`** — previously `okabe_ito`.
