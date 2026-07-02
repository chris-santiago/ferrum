# Remediation #44/#45/#46 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Fix #44 (cooks_distance IndexError on no-hat-matrix models), #46 (shap_waterfall per_class multiclass mis-render), and the #45 desugar-drops-explicit-scale bug, per the approved design spec **as amended 2026-07-02**: #45's composition-sharing itself ships separately as the Phase B Rust composite-render unification (own spec/plan/branch); this plan no longer builds the Python `_scale_share` extension.

## 2. Spec references

- `design-docs/superpowers/specs/2026-07-02-remediation-44-45-46-design.md` (all sections; §6 contracts, §7 byte-identity list, §9 acceptance)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/plots/regression.py` | #44 ValueError at leverage-drop site (spec §4/§6) |
| Modify | `src/ferrum/plots/_helpers.py` | #44 `_grid_panels` 0-and->4 guards |
| Test | `tests/diagnostics/test_regression.py` | #44 regression tests (non-linear cooks; auto-panels guard) |
| Test | `tests/diagnostics/test_compare_exclusions.py` | #44 compare= path error test |
| Modify | `src/ferrum/plots/explanation.py` | #46 per-class x0/x1 + facet gate in `_shap_waterfall_chart_from_source` |
| Modify | `src/ferrum/marks/_chart_mixins/_explanation.py` | #46 `_shap_waterfall_filter` class-aware (spec §4) |
| Test | `tests/diagnostics/test_explanation.py` | #46 discriminating per-class tests + beeswarm/bar hardening test |
| Create | `tests/goldens/phase_10/shap_chart_waterfall_per_class.svg` | #46 new golden (PNG-inspected) |
| Modify | `tests/diagnostics/test_goldens_phase_10.py` | #46 golden byte-equality test |
| Modify | `src/ferrum/chart.py` | #45 prerequisite: scale propagation at pending-composite-mark resolution |
| Test | `tests/test_scale_through_desugar.py` | scale-through-desugar tests (new file) |

## 4. Constraints

- Branch: create `fix/issues-44-46-remediation` off `main`; never commit to main directly.
- Order: Task 1 (#44) → Tasks 2–3 (#46) → Task 4 (desugar propagation) → Task 5 (close sweep). #45's composition sharing is out of this plan's scope (Phase B).
- Python-only: no changes under `crates/`. If a Rust change appears necessary, stop and surface to the orchestrator.
- Test-first per task: write the regression tests, run them, confirm RED (fail with the pre-fix symptom, e.g. IndexError for #44), then implement. If tests are written after implementation for any reason, prove RED via `git stash push -- <changed source>` → test fails → `git stash pop`, **before** `git add`.
- Byte-identity (spec §7): `per_class=False` waterfall, single-class `per_class=True`, all linear cooks paths, `residuals_chart` auto-panel degradation, flat-chart/ordinal sharing, pdp compare= independent-x, and composite marks without explicit chart-level positional scales must render identically. Existing goldens must pass un-regenerated.
- Scale propagation rule (spec §6, exact): positional channels x/y only; attach chart-level scale when layer channel has none; merge `domain` only when layer scale lacks `domain` (never overwrite `type`/`range`); skip when layer already has a `domain`.
- Any new golden SVG: run `python scripts/snapshot-goldens.py <name>` (or `regen_and_verify` from `tests/_snapshots.py`), Read the PNG, confirm correct render before committing. Orchestrator performs the Read.
- Errors raise; no warnings-as-fallback.
- Commits via `commit-commands:commit` skill only; `python-review-lite` gate before every commit; no Claude authorship trailers.
- All implementation dispatched to `python-coder` agents.

## 5. Tasks

### Task 1: #44 — leverage-only rejection + `_grid_panels` guards
- [ ] Create branch `fix/issues-44-46-remediation` off main
- [ ] RED tests: non-linear `cooks_distance_chart` raises ValueError naming `coef_`/hat-matrix + estimator type (`tests/diagnostics/test_regression.py`, reuse RandomForest fixture pattern at :247-254); compare= mixed-set raises identifying the offending member (`tests/diagnostics/test_compare_exclusions.py` near :188); `residuals_chart(panels=["residuals_vs_leverage"])` non-linear raises; `residuals_chart(panels="auto")` non-linear still renders 3 panels; `_grid_panels([])` and 5-chart input raise ValueError naming count
- [ ] Implement per spec §4/§6: drop-site ValueError in `_residuals_chart_from_source` (fires only when panel list empties; predicate stays `df["leverage"].is_nan().all()`); `_grid_panels` guards
- [ ] Verify: `uv run pytest tests/diagnostics/test_regression.py tests/diagnostics/test_compare_exclusions.py tests/test_bug_hunt_model_diagnostics.py -n auto`
- [ ] Commit (review-lite gated)

### Task 2: #46 — per-class waterfall computation + facet
- Consumes: per-class waterfall data contract from spec §6
- [ ] RED tests in `tests/diagnostics/test_explanation.py`: replace weak `"<svg" in svg` assertion at :294-297 with discriminating ones — panel per class, total bars = kept features × classes, per-class cumulative chain (x0 first = 0; x0[i] = x1[i-1] within class), shared x domain across panels; byte-identity tests for `per_class=False` multiclass and `per_class=True` binary (capture pre-fix SVG in-test via building both paths — assert unchanged output shape/equality vs single-class path)
- [ ] Implement per spec §4: per-class `x0`/`x1` (`cum_sum().over("class_label")`, shifted x0), x domain = union across classes, `_should_facet_by_class` gate + `facet(col="class_label")`, docstring stays accurate; make `_shap_waterfall_filter` keep the globally-ranked feature set for every class
- [ ] Verify: `uv run pytest tests/diagnostics/test_explanation.py -n auto`
- [ ] Commit (review-lite gated)

### Task 3: #46 — golden + SHAP-family hardening test
- Consumes: Task 2 implementation
- [ ] Add `shap_chart_waterfall_per_class` golden: generate via `regen_and_verify`, byte-equality test in `tests/diagnostics/test_goldens_phase_10.py` mirroring existing waterfall golden test
- [ ] Orchestrator gate: Read the rasterized PNG, confirm per-class panels render correctly before commit
- [ ] Hardening test (spec §9): shap_beeswarm + shap_bar with `per_class=True` + `compare=` — assert class facets share x within each model panel (discriminating, per issue #46 note)
- [ ] Verify: `uv run pytest tests/diagnostics/test_goldens_phase_10.py tests/diagnostics/test_explanation.py -n auto` and confirm existing `shap_chart_waterfall_sample3.svg` golden untouched (`git status tests/goldens/`)
- [ ] Commit (review-lite gated)

### Task 4: scale survives composite-mark desugar (#45 prerequisite)
- Consumes: propagation rule from spec §6 (repeated verbatim in Constraints)
- [ ] RED tests in `tests/test_scale_through_desugar.py`: explicit y-domain on a box/cv_scores chart renders that domain (SVG axis-extent inspection, reuse the extent-parsing pattern from `tests/test_facet_shared_extent.py:120-163`); log-scale validation_curve layer keeps `type: log` when a domain merges in; mark-computed domain (shap `x_scale_domain`) untouched by a chart-level scale; composite mark with no chart-level scale renders byte-identically (compare SVG before/after via unscaled chart)
- [ ] Implement propagation at pending-mark resolution in `src/ferrum/chart.py` (`_resolve_pending_stat` seam), generic across all 26 composite marks
- [ ] Verify: `uv run pytest -n auto` (full suite — shared contract touching every composite mark)
- [ ] Commit (review-lite gated)

### Task 5: batch close sweep
- [ ] Run the pinned repro scripts from the scratchpad: #44 raises clear ValueError; #46 renders faceted per-class panels (#45's repro stays red by design — Phase B delivers it)
- [ ] `uv run pytest -n auto` green; `nox -s lint` clean; `uv run pytest -m slow` if any touched path has slow coverage
- [ ] Check `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` for overlapping open items; update status if any
- [ ] Log remaining follow-ups as GitHub issues: shap base_value; `_grid_panels` >4 generalization (resolve=/facet unification is Phase B, in-round — do NOT file it as a deferral)
- [ ] Hand back to orchestrator for verification-before-completion + finishing-a-development-branch + issue close-out (#44/#46 close; #45 stays open pending Phase B)

## 6. Acceptance checks

- `uv run pytest -n auto` — full suite green
- `nox -s lint` — clean
- Spec §9 criteria for #44, #46, and the desugar-propagation slice of #45 observably met (composition-sharing criteria transfer to the Phase B spec)
- Existing goldens pass without regeneration; new per-class waterfall golden PNG-inspected before commit
- Each fix's regression tests demonstrated RED pre-fix
- Issues #44/#45/#46 referenced in commits; closed after user confirmation

## 7. Open questions

None.
