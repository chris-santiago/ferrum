# Cohesion Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: chris-code:subagent-driven-development. Dispatch python-coder/rust-coder per task; spec-compliance → quality → review-lite gates each; stage by file-footprint disjointness; track progress in the SDD ledger.

## 1. Objective

Close all 193 cohesion-audit findings by closing their underlying patterns, across 49 tasks in 6 tiers, behavior-preserving except the 5 regression-tested Tier-0 bug fixes and the additive public-API aliases.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-21-cohesion-campaign-design.md` — canonical decisions (D-*), target seams, invariants. **The ⚠️ decisions there are the pre-execution confirmation gate.**
- `design-docs/superpowers/followups/2026-06-21-cohesion-audit.md` — patterns + tiered roadmap.
- `design-docs/superpowers/followups/2026-06-21-cohesion-audit-findings.md` — all 193 findings, full detail, keyed by the same `CODE-NN` IDs used below.
- `CLAUDE.md` hard constraints; `ferrum-spec.md` API contract (update with dated notes where vocabulary changes).

## 3. Constraints

- **No matplotlib; no new global mutable state.** (CLAUDE.md hard constraints.)
- **Behavior-preserving** for every non-Tier-0 task: `uv run pytest -n auto` green; goldens byte-identical. Any intentionally-regenerated golden is rasterized via `scripts/snapshot-goldens.py` and **visually inspected by the orchestrator** before commit (coders never self-bless goldens).
- **Public API stays backward-compatible:** every vocabulary change ships canonical + alias. The single accepted module-path change (T4.4, D-MOD-1) is recorded in the changelog.
- **Tier-0 tasks require `/regression-test`** before commit (each test must fail on pre-fix code). Enforced by the PreToolUse hook on this `fix/` branch.
- **Per code task:** dispatch `python-coder` (`.py`) / `rust-coder` (`.rs`); spec-compliance → quality → review-lite gates; orchestrator re-runs the test command independently and verifies subagent reports.
- **Rust:** `cargo test` + `cargo clippy -D warnings` clean on every touched crate before any Rust task is marked done. `cargo test` uses the `DYLD_LIBRARY_PATH` form from CLAUDE.md.
- **Build:** Rust changes require `unset CONDA_PREFIX && uv run --no-sync maturin develop` before pytest sees them; WASM changes require the `wasm-pack build` form from CLAUDE.md.

## 4. Execution order & staging

Execute **tier by tier** (T0 → T1 → T2 → T3 → T4 → T6); run the matching heavyweight review (`python-review`/`rust-review`) on each tier's subsystems before advancing (CLAUDE.md escalation trigger #1). Within a tier, dispatch in parallel only tasks with **disjoint file footprints**; serialize any that share a file.

**Known serialization hotspots** (tasks sharing a file must not run concurrently):
- `src/ferrum/chart.py`: T1.11, T2.4, T3.6, T4.1, T4.3 → serialize; do T4.1 (the split) before T4.3 where possible.
- `src/ferrum/composition.py`: T1.4, T3.6, T3.7, T4.2, T4.5 → serialize.
- `src/ferrum/transforms.py`: T2.4, T3.2, T4.11 → serialize.
- `crates/.../transform/bin.rs`: T0.3, T1.2, T3.8 → serialize.
- `crates/.../render/prepare/mod.rs`: T1.9, T2.6, T4.9 → serialize.
- `crates/.../render/scene_build.rs`: T2.6, T4.9 → serialize.
- `crates/.../render/draw.rs`: T1.3, T1.8, T4.12 → serialize.
- `crates/.../scale/mod.rs`: T2.5, T4.14, T6.2 → serialize.
- `src/ferrum/encoding/base.py`: T2.3, T3.1 → serialize.
- `src/ferrum/plots/*`: T1.7, T3.4, T3.5, T4.7, T6.3 → check per-file overlap.

Tasks touching only one unique file (T0.1, T1.6, T3.3 partially, etc.) parallelize freely against non-overlapping peers.

## 5. Tasks

Each task lists the finding IDs it closes (see §6 for the inverse map) and its file footprint. Requirements per finding live in the findings doc — read the cited `CODE-NN` entries; do not restate them here.

### Tier 0 — Latent bugs (regression-tested)

#### T0.1 [rs] **[regression]** Point line-shapes honor opacity channel
- **Closes (1):** RMARK-01  (S4)
- **Files:** crates/ferrum-core/src/render/marks/point.rs

#### T0.2 [both] **[regression]** One boundary dtype-normalization + one Rust numeric predicate (D-DTYPE-1)
- **Closes (2):** SEAM-01, RSUP-02  (S4)
- **Files:** src/ferrum/_coerce.py; src/ferrum/_render.py; crates/ferrum-core/src/render/arrow_cast.rs

#### T0.3 [rs] **[regression]** group_partition() unifies bin/kde/kde_2d/smooth grouping (D-GROUPBY-1; fixes int/bool groupby)
- **Closes (3):** XFORM-01, XFORM-07, XFORM-09  (S2,S4)  [XFORM-03 reassigned to T3.8 during execution — it is purely the Option<String>→Vec<String> groupby field-shape API change, requiring PyO3/.pyi/Python edits outside T0.3's bug-fix scope]
- **Files:** crates/ferrum-core/src/transform/group_key.rs; crates/ferrum-core/src/transform/kde.rs; crates/ferrum-core/src/transform/kde_2d.rs; crates/ferrum-core/src/transform/smooth.rs; crates/ferrum-core/src/transform/bin.rs

#### T0.4 [both] **[regression]** Canonical `orient` across marks; implement mark_violin flip (D-ORIENT-1)
- **Closes (2):** XSIB-01, XSIB-02  (S3,S4)
- **Files:** src/ferrum/_marks_statistical.py; src/ferrum/marks/heavy_stat.py; src/ferrum/marks/composite.py; crates/ferrum-core/src/transform/violin.rs

#### T0.5 [py] **[regression]** Unify the three save()/HTML/title dispatchers; InteractiveChart.save honors extension (D-… REND)
- **Closes (6):** REND-01, REND-02, REND-03, REND-07, REND-09, REND-10  (S2,S3,S4)
- **Files:** src/ferrum/display.py; src/ferrum/composition.py; src/ferrum/_interactive.py; src/ferrum/_render.py


### Tier 1 — Finish started unifications (behavior-preserving)

#### T1.1 [rs] Adopt MarkNodes + shared color/stroke/opacity resolvers in all mark builders (MOD-06)
- **Closes (8):** RMARK-02, RMARK-03, RMARK-04, RMARK-05, RMARK-06, RMARK-07, RMARK-08, MOD-06  (S2,S3)
- **Files:** crates/ferrum-core/src/render/marks/point.rs; crates/ferrum-core/src/render/marks/rect.rs; crates/ferrum-core/src/render/marks/area.rs; crates/ferrum-core/src/render/marks/arc.rs; crates/ferrum-core/src/render/marks/image.rs; crates/ferrum-core/src/render/marks/channels.rs

#### T1.2 [rs] One finite-extent helper; delete duplicate min_max wrappers (XFORM-02)
- **Closes (4):** XFORM-02, XFORM-05, XFORM-08, SPINE-09  (S2,S3)
- **Files:** crates/ferrum-core/src/transform/numeric_util.rs; crates/ferrum-core/src/transform/bin.rs; crates/ferrum-core/src/render/prepare/extent.rs

#### T1.3 [rs] One DASH_PALETTE const drives both dash maps (RSUP-01)
- **Closes (1):** RSUP-01  (S4)
- **Files:** crates/ferrum-core/src/render/draw.rs; crates/ferrum-core/src/render/pack_instances.rs

#### T1.4 [py] Route Repeat/Layer through _validate_resolve; _empty_scene() everywhere; dedup merge loop (COMP)
- **Closes (4):** COMP-02, COMP-03, COMP-04, COMP-05  (S2,S3)
- **Files:** src/ferrum/composition.py

#### T1.5 [py] gain/lift into _curve_frames; one X→RecordBatch helper; classification-curve boilerplate (DIAG dup)
- **Closes (4):** DIAG-01, DIAG-03, DIAG-05, DIAG-09  (S2,S3)
- **Files:** src/ferrum/_diagnostics/_curve_frames.py; src/ferrum/_diagnostics/sources/_classification.py; src/ferrum/_diagnostics/precomputed.py; src/ferrum/_diagnostics/sources/_clustering.py; src/ferrum/_diagnostics/sources/_ranking.py; src/ferrum/_diagnostics/sources/_predictions.py; src/ferrum/_diagnostics/_rank_helpers.py

#### T1.6 [py] position.py value classes → @dataclass(frozen=True) (CHART-03)
- **Closes (1):** CHART-03  (S3)
- **Files:** src/ferrum/position.py

#### T1.7 [py] Route plots DataFrame coercion through _coerce; move _merge_layers to _helpers (PLOT dup)
- **Closes (3):** PLOT-03, PLOT-04, PLOT-10  (S2,S3)
- **Files:** src/ferrum/plots/_helpers.py; src/ferrum/plots/regression.py; src/ferrum/plots/matrix.py; src/ferrum/plots/distribution.py; src/ferrum/plots/ranking.py

#### T1.8 [rs] Rust render-support dedup: stroke cap/join, decode preamble, compose tails, baseline parser (RSUP)
- **Closes (4):** RSUP-04, RSUP-07, RSUP-08, RSUP-09  (S2,S3)
- **Files:** crates/ferrum-core/src/render/draw.rs; crates/ferrum-core/src/render/binding.rs; crates/ferrum-core/src/render/format.rs

#### T1.9 [rs] Collapse facet-extent-pin + facet partition/group to one generic helper each (SPINE dup)
- **Closes (4):** SPINE-05, SPINE-06, SPINE-10, SPINE-11  (S2,S3)
- **Files:** crates/ferrum-core/src/render/prepare/extent.rs; crates/ferrum-core/src/render/prepare/mod.rs

#### T1.10 [rs] WASM dedup: one tooltip parser, one membership predicate, one indexed-kind, MarkEntry/data_idx (WASM dup)
- **Closes (6):** WASM-01, WASM-03, WASM-04, WASM-06, WASM-09, WASM-10  (S1,S2,S3)
- **Files:** crates/ferrum-wasm/src/scene_load.rs; crates/ferrum-wasm/src/conditional.rs; crates/ferrum-wasm/src/hit_test.rs; crates/ferrum-wasm/src/spatial_index.rs

#### T1.11 [py] Unify name-dedup + reactive-param-collision helpers; drop dead _NamedTransform._inner_eq (CHART dup)
- **Closes (5):** CHART-04, CHART-05, CHART-06, CHART-09, CHART-10  (S1,S2)
- **Files:** src/ferrum/chart.py


### Tier 2 — Collapse dual sources of truth

#### T2.1 [both] Theme key contract derived from one manifest; resolve font_size/grid/_FALLBACKS drift (D-THEME-1)
- **Closes (4):** THEME-01, THEME-02, THEME-05, THEME-07  (S2,S3)  [XDEAD-03 CARVED OUT → T2.1b: it is annotation dead-flags (render/annotation.rs + annotation/primitives.py), NOT a theme file, and an [API] wire-vs-drop fork the "non-breaking aliases" answer doesn't cover — needs feasibility assessment / user decision]
- **Files:** src/ferrum/themes/__init__.py; src/ferrum/themes/_defaults.py; crates/ferrum-core/src/render/binding.rs  [corrected: ThemeOverridesSpec lives in render/binding.rs:195, not spec/theme.rs]

#### T2.1b [both] **[CARVED FROM T2.1 — pending decision]** Annotation z/curve dead FFI flags (XDEAD-03, [API])
- **Closes (1):** XDEAD-03  (S3)
- **Files:** crates/ferrum-core/src/render/annotation.rs; src/ferrum/annotation/primitives.py
- **Decision needed:** wire per-annotation z-ordering (below_marks → pre-mark scene slot, align Rust enum to Python above/below_marks vocab) + curve bezier, OR drop z/curve from the Python API + Rust structs. No-defer principle favors wiring z (infra exists in scene_build draws_above_marks); curve=bezier is a real geometry feature. Assess feasibility, then wire or surface.

#### T2.2 [both] Rust palette registry is sole source; color.py consumes it; scheme= validated at declaration (D-COLOR-1)
- **Closes (4):** ENC-06, XNAME-02, XSIB-07, ENC-11  (S2,S3)
- **Files:** src/ferrum/color.py; src/ferrum/schemes.py; src/ferrum/encoding/{appearance,_scale,base}.py; src/ferrum/marks/{heavy_stat,statistical}.py; src/ferrum/_marks_statistical.py; crates/ferrum-core/src/render/palette.rs + render/color/{mod,categorical,continuous}.rs; crates/ferrum-core/src/lib.rs  [corrected: palette registry is render/palette.rs + render/color/, not render/color/palette.rs; XNAME-02 merges schemes.py; XSIB-07 cmap→scheme spans marks/*; scheme= validation in encoding/base.py]

#### T2.3 [py] Honored-kwargs is the single truth; serializer iterates it; one honored vocab module (D-HONORED-1)
- **Closes (7):** ENC-01, ENC-04, ENC-08, ENC-09, ENC-10, XSIB-03, XSIB-05  (S2,S3)
- **Files:** src/ferrum/encoding/{base,positional,appearance,text,facet,__init__}.py; NEW src/ferrum/encoding/_honored.py; NEW src/ferrum/encoding/_aliases.py  [corrected during execution: ENC-10 lives in __init__.py; XSIB-05 spans facet.py; the honored-vocab + alias-table get their own modules]

#### T2.4 [both] One STACK_OFFSETS + validator; one stack capability registry (D-STACK-1)
- **Closes (3):** ENC-02, ENC-07, CHART-08  (S2,S3)
- **Files:** src/ferrum/position.py; src/ferrum/encoding/positional.py; src/ferrum/transforms.py; src/ferrum/chart.py

#### T2.5 [rs] Continuous scales → named-field structs; one domain/range/utc representation (D-SCALE-1)
- **Closes (6):** SPEC-01, SPEC-02, SPEC-03, SPEC-06, SPEC-07, SEAM-08  (S2,S3,S4)
- **Files:** crates/ferrum-core/src/scale/{linear,log,time,pow,symlog,band,core,mod}.rs  [corrected during execution: the 6 continuous scales + BandScale live in per-type files, not just time/linear/mod]

#### T2.6 [rs] One AxisStyleOverrides::fill_from merge; resolve_panel_scales seam (SPINE-01/02)
- **Closes (3):** SPINE-01, SPINE-02, SPINE-03  (S3,S4)
- **Files:** crates/ferrum-core/src/render/prepare/mod.rs; crates/ferrum-core/src/render/mod.rs; crates/ferrum-core/src/render/scene_build.rs


### Tier 3 — Unify public vocabulary (canonical + alias)

#### T3.1 [both] format_type canonical, formatType alias (D-FMT-1)
- **Closes (1):** ENC-03  (S3)
- **Files:** src/ferrum/encoding/base.py; src/ferrum/encoding/text.py; src/ferrum/encoding/positional.py

#### T3.2 [py] as_ canonical across transform_* wrappers; validation parity with channels (D-ASNAME-1)
- **Closes (2):** ENC-05, SEAM-03  (S3)
- **Files:** src/ferrum/transforms.py

#### T3.3 [both] Disambiguate `extent`: band / whisker_mult / method renames with aliases (D-EXTENT-1)
- **Closes (1):** XNAME-01  (S3)
- **Files:** src/ferrum/marks/heavy_stat.py; src/ferrum/marks/composite.py; crates/ferrum-core/src/layout/mod.rs; crates/ferrum-core/src/transform/letter_value.rs

#### T3.4 [py] First-positional param canonicalization across figure families (D-FIRSTPARAM-1)
- **Closes (4):** PLOT-02, PLOT-06, PLOT-08, PLOT-11  (S1,S2,S3)
- **Files:** src/ferrum/plots/clustering.py; src/ferrum/plots/ranking.py; src/ferrum/plots/_helpers.py

#### T3.5 [py] Add compare=/random_state to every model-diagnostic function (D-COMPARE-1)
- **Closes (2):** PLOT-01, XSIB-08  (S2,S3)
- **Files:** src/ferrum/plots/regression.py; src/ferrum/plots/explanation.py; src/ferrum/plots/model_selection.py; src/ferrum/plots/clustering.py

#### T3.6 [py] share_scale() sibling signatures unified; one scale-sharing mechanism (D-PANEL/share)
- **Closes (1):** XNAME-03  (S3)
- **Files:** src/ferrum/composition.py; src/ferrum/chart.py

#### T3.7 [both] Naming: panel/chrome/sentinel/mark_text-label terminology alignment (D-PANEL-1/D-CHROME-1)
- **Closes (4):** XNAME-04, XNAME-05, XNAME-06, XNAME-07  (S1,S2)
- **Files:** src/ferrum/composition.py; src/ferrum/_render.py; crates/ferrum-core/src/render/figure_chrome.rs

#### T3.8 [rs+py] groupby field one shape (XFORM-03 Option<String>→Vec<String>, canonical+alias); Bin → typed BinSpecAxis enum like bin_2d
- **Closes (3):** XFORM-03, XFORM-04, XFORM-06  (S2,S3)  [XFORM-03 reassigned here from T0.3 — public-API field-shape change, staged with the non-breaking-alias discipline per D-GROUPBY-1]
- **Files:** crates/ferrum-core/src/transform/bin.rs; crates/ferrum-core/src/transform/core.rs; crates/ferrum-core/src/transform/{kde,kde_2d,smooth}.rs; src/ferrum/_core.pyi; src/ferrum/transforms.py; src/ferrum/marks/ (desugar call sites)


### Tier 4 — Structural splits (pure moves behind goldens)

#### T4.1 [py] Split chart.py → _desugar.py + _layer_transforms.py + SpecBuildMixin (CHART/MOD god-module)
- **Closes (4):** CHART-01, CHART-02, CHART-07, MOD-03  (S2,S3)
- **Files:** src/ferrum/chart.py; src/ferrum/_desugar.py; src/ferrum/_layer_transforms.py

#### T4.2 [py] Split composition.py → _scene_merge.py; unify the 4 grid variants (COMP/MOD god-module)
- **Closes (3):** COMP-07, COMP-08, MOD-04  (S2,S3)
- **Files:** src/ferrum/composition.py; src/ferrum/_scene_merge.py

#### T4.3 [py] Move mark mixins into marks/ package; split the diagnostic monolith by domain (MARK/MOD)
- **Closes (3):** MARK-03, MOD-07, XDEAD-07  (S2,S3)
- **Files:** src/ferrum/_marks_statistical.py; src/ferrum/_marks_diagnostic.py; src/ferrum/marks/diagnostic/__init__.py; src/ferrum/chart.py

#### T4.4 [py] Promote _diagnostics → public diagnostics/ package; align taxonomy naming (D-MOD-1)
- **Closes (2):** MOD-01, MOD-02  (S3)
- **Files:** src/ferrum/_diagnostics/; src/ferrum/__init__.py; scripts/gen_api_pages.py

#### T4.5 [py] annotations.py → thin layer over annotation/; one coord-coercion, aligned vocabulary (COMP-01)
- **Closes (2):** COMP-01, COMP-06  (S2,S3)
- **Files:** src/ferrum/annotations.py; src/ferrum/annotation/primitives.py; src/ferrum/chart.py

#### T4.6 [py] Diagnostics visualizer family: builders for manifold/elbow; has_score property; score() shared (DIAG/XSIB)
- **Closes (9):** DIAG-02, DIAG-04, DIAG-06, DIAG-07, DIAG-08, DIAG-10, DIAG-11, XSIB-04, XSIB-09  (S1,S2,S3)
- **Files:** src/ferrum/_diagnostics/visualizers/clustering.py; src/ferrum/_diagnostics/visualizers/base.py; src/ferrum/plots/clustering.py; src/ferrum/_diagnostics/visualizers/regression.py

#### T4.7 [py] ROC/PR/calibration single annotation path; seaborn family builder split (PLOT/MOD-05)
- **Closes (2):** PLOT-05, MOD-05  (S3)
- **Files:** src/ferrum/plots/classification.py; src/ferrum/marks/diagnostic/_classification.py; src/ferrum/plots/matrix.py; src/ferrum/plots/distribution.py

#### T4.8 [rs] Decompose compute_layout; shared measure/carve core; AxisLayout/LegendLayout builders (LAYOUT)
- **Closes (11):** LAYOUT-01, LAYOUT-02, LAYOUT-03, LAYOUT-04, LAYOUT-05, LAYOUT-06, LAYOUT-07, LAYOUT-08, LAYOUT-09, LAYOUT-10, LAYOUT-11  (S1,S2,S3)
- **Files:** crates/ferrum-core/src/layout/mod.rs; crates/ferrum-core/src/layout/axis.rs; crates/ferrum-core/src/layout/legend.rs

#### T4.9 [rs] Decompose prepare_render_inputs + build_scene; loop x/y axis-input over a channel (SPINE god)
- **Closes (5):** SPINE-04, SPINE-07, SPINE-08, SPINE-12, MOD-09  (S2,S3)
- **Files:** crates/ferrum-core/src/render/prepare/mod.rs; crates/ferrum-core/src/render/scene_build.rs

#### T4.10 [rs] WASM circle-vs-rect one typed representation; ConditionalEncoding channel-xor-value; remove dead bindgen methods (WASM state)
- **Closes (4):** WASM-02, WASM-05, WASM-07, WASM-08  (S2,S3)
- **Files:** crates/ferrum-wasm/src/scene_load.rs; crates/ferrum-wasm/src/conditional.rs; crates/ferrum-wasm/src/lib.rs

#### T4.11 [rs] Transform/seam: dict path serde-validated; one transform representation; PyO3 thin bindings (SEAM)
- **Closes (7):** SEAM-02, SEAM-04, SEAM-05, SEAM-06, SEAM-07, SPEC-08, SPEC-05  (S2,S3)
- **Files:** crates/ferrum-core/src/lib.rs; crates/ferrum-core/src/spec/chart.rs; crates/ferrum-core/src/spec/encoding.rs; src/ferrum/transforms.py

#### T4.12 [rs] MarkStyle god-struct → typed per-family sub-structs; apply_dodge error consistency; color module rename (RSUP)
- **Closes (4):** RSUP-03, RSUP-05, RSUP-06, RSUP-10  (S2,S3)
- **Files:** crates/ferrum-core/src/render/draw.rs; crates/ferrum-core/src/render/position.rs; crates/ferrum-core/src/render/color/categorical.rs; crates/ferrum-core/src/render/figure_chrome.rs

#### T4.13 [py] Render-surface module consolidation; _render.py annotation-coord split; typed render fork (REND)
- **Closes (7):** REND-04, REND-05, REND-06, REND-08, REND-11, REND-12, MOD-10  (S2,S3)
- **Files:** src/ferrum/_render.py; src/ferrum/display.py; src/ferrum/_interactive.py

#### T4.14 [rs] ScaleSpec ↔ PyO3 scale-class reconciliation; remove unused Band/Point classes (SPEC)
- **Closes (3):** SPEC-04, SPEC-09, SPEC-10  (S1,S3)
- **Files:** crates/ferrum-core/src/scale/mod.rs; crates/ferrum-core/src/spec/encoding.rs


### Tier 6 — Scar-tissue sweep + docs

#### T6.1 [py] Rewrite MarkDesugarResult annotations+docstrings+doctests; enable scoped doctests (D-MARKRESULT-1)
- **Closes (10):** MARK-01, MARK-02, MARK-04, MARK-05, MARK-06, MARK-07, MARK-08, MARK-09, XDEAD-04, XSIB-06  (S1,S2,S3)
- **Files:** src/ferrum/marks/composite.py; src/ferrum/marks/statistical.py; src/ferrum/marks/heavy_stat.py; src/ferrum/marks/diagnostic/_classification.py; src/ferrum/marks/diagnostic/_regression.py; src/ferrum/marks/diagnostic/_selection.py; src/ferrum/_layer.py; noxfile.py

#### T6.2 [rs] Remove stale #[allow(dead_code)] on live code; delete truly-dead items (XDEAD Rust)
- **Closes (5):** XDEAD-01, XDEAD-02, XDEAD-05, XDEAD-06, XDEAD-08  (S1,S2,S3)
- **Files:** crates/ferrum-core/src/scale/mod.rs; crates/ferrum-core/src/render/format.rs; crates/ferrum-core/src/render/pack_instances.rs; crates/ferrum-core/src/render/annotation.rs

#### T6.3 [py] Remove dead Python scar tissue: _apply_overrides imports, dead exceptions, deprecated shims (PLOT/REND/DIAG)
- **Closes (4):** PLOT-07, PLOT-09, DIAG-12, THEME-03  (S1,S2)
- **Files:** src/ferrum/plots/classification.py; src/ferrum/plots/regression.py; src/ferrum/plots/explanation.py; src/ferrum/plots/clustering.py; src/ferrum/plots/ranking.py; src/ferrum/_diagnostics/_metric_labels.py

#### T6.4 [py] Themes: 12 builtins from shared base; resolve import cycle; set_default/theme_context alias (THEME)
- **Closes (2):** THEME-04, THEME-06  (S2)
- **Files:** src/ferrum/themes/__init__.py; src/ferrum/themes/_defaults.py

#### T6.5 [py] Docs/API-page coverage: homeless __all__ symbols, Grid page, regression module docstring (MOD docs)
- **Closes (2):** MOD-08, MOD-11  (S1,S2)
- **Files:** scripts/gen_api_pages.py; docs/site/api/; src/ferrum/plots/regression.py

## 6. Coverage matrix (every finding → its task)

All 193 findings are assigned to exactly one task (verified: 0 missing, 0 duplicate).

| Finding | Sev | Task |
|---|---|---|
| CHART-01 | S3 | T4.1 |
| CHART-02 | S3 | T4.1 |
| CHART-03 | S3 | T1.6 |
| CHART-04 | S2 | T1.11 |
| CHART-05 | S2 | T1.11 |
| CHART-06 | S2 | T1.11 |
| CHART-07 | S2 | T4.1 |
| CHART-08 | S2 | T2.4 |
| CHART-09 | S2 | T1.11 |
| CHART-10 | S1 | T1.11 |
| COMP-01 | S3 | T4.5 |
| COMP-02 | S3 | T1.4 |
| COMP-03 | S3 | T1.4 |
| COMP-04 | S2 | T1.4 |
| COMP-05 | S2 | T1.4 |
| COMP-06 | S2 | T4.5 |
| COMP-07 | S2 | T4.2 |
| COMP-08 | S2 | T4.2 |
| DIAG-01 | S3 | T1.5 |
| DIAG-02 | S3 | T4.6 |
| DIAG-03 | S3 | T1.5 |
| DIAG-04 | S3 | T4.6 |
| DIAG-05 | S2 | T1.5 |
| DIAG-06 | S2 | T4.6 |
| DIAG-07 | S2 | T4.6 |
| DIAG-08 | S2 | T4.6 |
| DIAG-09 | S2 | T1.5 |
| DIAG-10 | S2 | T4.6 |
| DIAG-11 | S2 | T4.6 |
| DIAG-12 | S1 | T6.3 |
| ENC-01 | S3 | T2.3 |
| ENC-02 | S3 | T2.4 |
| ENC-03 | S3 | T3.1 |
| ENC-04 | S3 | T2.3 |
| ENC-05 | S3 | T3.2 |
| ENC-06 | S3 | T2.2 |
| ENC-07 | S2 | T2.4 |
| ENC-08 | S2 | T2.3 |
| ENC-09 | S2 | T2.3 |
| ENC-10 | S2 | T2.3 |
| ENC-11 | S2 | T2.2 |
| LAYOUT-01 | S3 | T4.8 |
| LAYOUT-02 | S3 | T4.8 |
| LAYOUT-03 | S3 | T4.8 |
| LAYOUT-04 | S3 | T4.8 |
| LAYOUT-05 | S3 | T4.8 |
| LAYOUT-06 | S2 | T4.8 |
| LAYOUT-07 | S2 | T4.8 |
| LAYOUT-08 | S2 | T4.8 |
| LAYOUT-09 | S2 | T4.8 |
| LAYOUT-10 | S2 | T4.8 |
| LAYOUT-11 | S1 | T4.8 |
| MARK-01 | S3 | T6.1 |
| MARK-02 | S3 | T6.1 |
| MARK-03 | S3 | T4.3 |
| MARK-04 | S2 | T6.1 |
| MARK-05 | S2 | T6.1 |
| MARK-06 | S2 | T6.1 |
| MARK-07 | S2 | T6.1 |
| MARK-08 | S2 | T6.1 |
| MARK-09 | S1 | T6.1 |
| MOD-01 | S3 | T4.4 |
| MOD-02 | S3 | T4.4 |
| MOD-03 | S3 | T4.1 |
| MOD-04 | S3 | T4.2 |
| MOD-05 | S3 | T4.7 |
| MOD-06 | S3 | T1.1 |
| MOD-07 | S2 | T4.3 |
| MOD-08 | S2 | T6.5 |
| MOD-09 | S2 | T4.9 |
| MOD-10 | S2 | T4.13 |
| MOD-11 | S1 | T6.5 |
| PLOT-01 | S3 | T3.5 |
| PLOT-02 | S3 | T3.4 |
| PLOT-03 | S3 | T1.7 |
| PLOT-04 | S3 | T1.7 |
| PLOT-05 | S3 | T4.7 |
| PLOT-06 | S2 | T3.4 |
| PLOT-07 | S2 | T6.3 |
| PLOT-08 | S2 | T3.4 |
| PLOT-09 | S2 | T6.3 |
| PLOT-10 | S2 | T1.7 |
| PLOT-11 | S1 | T3.4 |
| REND-01 | S4 | T0.5 |
| REND-02 | S3 | T0.5 |
| REND-03 | S3 | T0.5 |
| REND-04 | S3 | T4.13 |
| REND-05 | S3 | T4.13 |
| REND-06 | S2 | T4.13 |
| REND-07 | S2 | T0.5 |
| REND-08 | S2 | T4.13 |
| REND-09 | S2 | T0.5 |
| REND-10 | S2 | T0.5 |
| REND-11 | S2 | T4.13 |
| REND-12 | S2 | T4.13 |
| RMARK-01 | S4 | T0.1 |
| RMARK-02 | S3 | T1.1 |
| RMARK-03 | S3 | T1.1 |
| RMARK-04 | S3 | T1.1 |
| RMARK-05 | S2 | T1.1 |
| RMARK-06 | S2 | T1.1 |
| RMARK-07 | S2 | T1.1 |
| RMARK-08 | S2 | T1.1 |
| RSUP-01 | S4 | T1.3 |
| RSUP-02 | S4 | T0.2 |
| RSUP-03 | S3 | T4.12 |
| RSUP-04 | S3 | T1.8 |
| RSUP-05 | S3 | T4.12 |
| RSUP-06 | S3 | T4.12 |
| RSUP-07 | S2 | T1.8 |
| RSUP-08 | S2 | T1.8 |
| RSUP-09 | S2 | T1.8 |
| RSUP-10 | S2 | T4.12 |
| SEAM-01 | S4 | T0.2 |
| SEAM-02 | S3 | T4.11 |
| SEAM-03 | S3 | T3.2 |
| SEAM-04 | S3 | T4.11 |
| SEAM-05 | S2 | T4.11 |
| SEAM-06 | S2 | T4.11 |
| SEAM-07 | S2 | T4.11 |
| SEAM-08 | S2 | T2.5 |
| SPEC-01 | S4 | T2.5 |
| SPEC-02 | S3 | T2.5 |
| SPEC-03 | S3 | T2.5 |
| SPEC-04 | S3 | T4.14 |
| SPEC-05 | S3 | T4.11 |
| SPEC-06 | S2 | T2.5 |
| SPEC-07 | S2 | T2.5 |
| SPEC-08 | S2 | T4.11 |
| SPEC-09 | S1 | T4.14 |
| SPEC-10 | S1 | T4.14 |
| SPINE-01 | S4 | T2.6 |
| SPINE-02 | S4 | T2.6 |
| SPINE-03 | S3 | T2.6 |
| SPINE-04 | S3 | T4.9 |
| SPINE-05 | S3 | T1.9 |
| SPINE-06 | S3 | T1.9 |
| SPINE-07 | S3 | T4.9 |
| SPINE-08 | S3 | T4.9 |
| SPINE-09 | S2 | T1.2 |
| SPINE-10 | S2 | T1.9 |
| SPINE-11 | S2 | T1.9 |
| SPINE-12 | S2 | T4.9 |
| THEME-01 | S3 | T2.1 |
| THEME-02 | S2 | T2.1 |
| THEME-03 | S2 | T6.3 |
| THEME-04 | S2 | T6.4 |
| THEME-05 | S2 | T2.1 |
| THEME-06 | S2 | T6.4 |
| THEME-07 | S2 | T2.1 |
| WASM-01 | S3 | T1.10 |
| WASM-02 | S3 | T4.10 |
| WASM-03 | S3 | T1.10 |
| WASM-04 | S3 | T1.10 |
| WASM-05 | S3 | T4.10 |
| WASM-06 | S2 | T1.10 |
| WASM-07 | S2 | T4.10 |
| WASM-08 | S2 | T4.10 |
| WASM-09 | S2 | T1.10 |
| WASM-10 | S1 | T1.10 |
| XDEAD-01 | S3 | T6.2 |
| XDEAD-02 | S3 | T6.2 |
| XDEAD-03 | S3 | T2.1 |
| XDEAD-04 | S2 | T6.1 |
| XDEAD-05 | S2 | T6.2 |
| XDEAD-06 | S2 | T6.2 |
| XDEAD-07 | S2 | T4.3 |
| XDEAD-08 | S1 | T6.2 |
| XFORM-01 | S4 | T0.3 |
| XFORM-02 | S3 | T1.2 |
| XFORM-03 | S3 | T3.8 |
| XFORM-04 | S3 | T3.8 |
| XFORM-05 | S2 | T1.2 |
| XFORM-06 | S2 | T3.8 |
| XFORM-07 | S2 | T0.3 |
| XFORM-08 | S2 | T1.2 |
| XFORM-09 | S2 | T0.3 |
| XNAME-01 | S3 | T3.3 |
| XNAME-02 | S3 | T2.2 |
| XNAME-03 | S3 | T3.6 |
| XNAME-04 | S2 | T3.7 |
| XNAME-05 | S2 | T3.7 |
| XNAME-06 | S2 | T3.7 |
| XNAME-07 | S1 | T3.7 |
| XSIB-01 | S4 | T0.4 |
| XSIB-02 | S3 | T0.4 |
| XSIB-03 | S3 | T2.3 |
| XSIB-04 | S3 | T4.6 |
| XSIB-05 | S2 | T2.3 |
| XSIB-06 | S2 | T6.1 |
| XSIB-07 | S2 | T2.2 |
| XSIB-08 | S2 | T3.5 |
| XSIB-09 | S1 | T4.6 |

## 7. Acceptance checks

- `uv run nox` green (lint + pytest + cargo test + build + docs).
- `cargo clippy -D warnings` clean on ferrum-core + ferrum-wasm.
- Goldens byte-identical except intentionally-regenerated dirs (each visually inspected).
- Every `CODE-NN` in §6 closed by a committed task (re-run the coverage check).
- Tier-0 regression tests present and red-on-old-code.
- Step 4: heavyweight `python-review` + `rust-review` over the branch report no residual instance of the six patterns in touched subsystems and no new cohesion regression; `bug-hunt`/`test-sweep` find no new defects.

## 8. Open questions

The ⚠️ decisions in the design spec §6 (D-COLOR-1, D-ORIENT-1, D-FMT-1, D-EXTENT-1, D-FIRSTPARAM-1, D-COMPARE-1, D-MOD-1) gate execution. All have recommended resolutions; confirm or override before Tier 0 begins.
