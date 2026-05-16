# Python Coherence Pass — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Address all 21 findings from the full Python review (Q1–Q7, M1–M8, H1–H6): eliminate dead code, fix silent data loss, deduplicate boilerplate, standardize naming, and decompose the 5231-line `chart.py` god file.

## 2. Spec references

- Review findings F1–F21 in the 2026-05-15 Python review conversation (no external doc — the review IS the spec)
- `CLAUDE.md` §Hard constraints — no global mutable state, `ferrum-spec.md` is API contract
- `design-docs/ARCHITECTURE.md` — Python vs. Rust responsibility boundary

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/marks/diagnostic/_selection.py` | Delete 3 duplicate desugar functions (F2) |
| Modify | `src/ferrum/marks/diagnostic/_clustering.py` | Ensure canonical copies remain (F2) |
| Modify | `src/ferrum/marks/diagnostic/__init__.py` | Fix imports after dedup (F2) |
| Modify | `src/ferrum/marks/base.py` | Forward `blend`, remove unused `ClassVar` (F3, F19) |
| Modify | `src/ferrum/marks/composite.py` | Fix return annotations (F4) |
| Modify | `src/ferrum/marks/heavy_stat.py` | Fix return annotations (F4) |
| Modify | `src/ferrum/encoding/base.py` | Remove `_renders_in_phase_8a` (F9) |
| Modify | `src/ferrum/encoding/positional.py` | Remove `_renders_in_phase_8a` (F9) |
| Modify | `src/ferrum/encoding/appearance.py` | Remove `_renders_in_phase_8a` (F9) |
| Modify | `src/ferrum/encoding/text.py` | Remove `_renders_in_phase_8a` (F9) |
| Modify | `src/ferrum/encoding/facet.py` | Remove `_renders_in_phase_8a` (F9) |
| Modify | `tests/test_encoding.py` | Delete `_renders_in_phase_8a` assertions (F9) |
| Modify | `src/ferrum/_interactive.py` | Delete dead `merge_scene_graphs`, `_offset_nodes`, `_offset_path_cmds`, unused `_render_scene_json` (F12) |
| Modify | `tests/test_bug_hunt_phase_11_interactive.py` | Remove import/tests for deleted functions (F12) |
| Modify | `src/ferrum/chart.py` | Narrow bare `except Exception: pass` (F20); extract `_warn_large_chart`, `_apply_remap` helpers (F1/F15); extract rendering block to `_render.py` (H1); extract encoding helpers to `encoding/` (H2); extract composition helpers (H3) |
| Create | `src/ferrum/_render.py` | Rendering methods extracted from `chart.py` (H1) |
| Modify | `src/ferrum/plots/_helpers.py` | Add `_finalize_chart` helper (M1) |
| Modify | `src/ferrum/plots/classification.py` | Use `_finalize_chart`, remove `_resolve_source` wrapper (M1, M2) |
| Modify | `src/ferrum/plots/regression.py` | Same (M1, M2) |
| Modify | `src/ferrum/plots/clustering.py` | Same (M1, M2) |
| Modify | `src/ferrum/plots/explanation.py` | Same (M1, M2) |
| Modify | `src/ferrum/plots/ranking.py` | Same (M1, M2) |
| Modify | `src/ferrum/plots/model_selection.py` | Same (M1) |
| Modify | `src/ferrum/plots/distribution.py` | Same (M1) |
| Modify | `src/ferrum/plots/matrix.py` | Same (M1) |
| Create | `src/ferrum/_metric_labels.py` | Metric-label subsystem split from annotations.py (M5) |
| Modify | `src/ferrum/annotations.py` | Remove metric-label code, import from `_metric_labels` (M5) |
| Modify | `src/ferrum/_warn.py` | Replace global `_seen` set with `contextvars.ContextVar` (M6) |
| Modify | `src/ferrum/marks/diagnostic/_regression.py` | Rename `identity_line` → `reference_line` (M7) |
| Modify | `src/ferrum/marks/diagnostic/_classification.py` | Rename `reference_lines` → `reference_line` (M7) |
| Modify | `src/ferrum/coord.py` | Rename `_to_spec_dict` → `to_spec_dict` (M8) |
| Modify | `src/ferrum/themes/__init__.py` | Rename `to_theme_inputs_dict` → `to_spec_dict` (M8) |
| Modify | `src/ferrum/marks/diagnostic/_classification.py` | Add `**mark_kwargs` (H6) |
| Modify | `src/ferrum/marks/diagnostic/_regression.py` | Add `**mark_kwargs` (H6) |
| Modify | `src/ferrum/marks/diagnostic/_explanation.py` | Add `**mark_kwargs` (H6) |
| Modify | `src/ferrum/composition.py` | Receive extracted helpers from chart.py (H3) |
| Modify | `src/ferrum/encoding/__init__.py` | Receive extracted helpers from chart.py (H2) |
| Modify | `src/ferrum/_diagnostics/visualizers/ranking.py` | Delete dead `_columns_and_array` (F12) |
| Create | `src/ferrum/encoding/_scale.py` | Extract `_scale_to_dict` from base.py (F18) |
| Modify | `src/ferrum/themes/_defaults.py` | Replace `_DefaultThemeCM` class with contextmanager |
| Modify | all `src/ferrum/plots/*.py` | Standardize first-positional param names (F17) |

## 4. Constraints

- **No public API changes** in Tasks 1–4. All are internal-only.
- **Task 5 (chart.py decomposition)** moves methods but must preserve every import path — `from ferrum.chart import Chart` and `Chart.show_svg()` etc. must keep working.
- **Task 6 (`reference_line` rename)** is a parameter rename on diagnostic marks. Library is not public yet — no deprecation shims needed, just rename.
- **Task 7 (serialization rename)** touches internal method names only — `to_spec_dict()` is not user-facing. But `chart.py` and `composition.py` call these methods; all call sites must update.
- **Task 8 (`**mark_kwargs` addition)** is additive — new optional parameter, no breaking change.
- **Test suite must stay green after every task.** Run `uv run pytest` after each.
- Do **not** rebuild Rust extension unless `blend` forwarding requires Rust-side changes (it should not — `to_mark_kwargs_dict()` output is consumed by `ChartSpec` which already accepts `blend`).

## 5. Tasks

### Task 1: Dead code & fossil metadata deletion (Q4, Q5, Q6, F12, F9)
- [ ] Delete `merge_scene_graphs`, `_offset_nodes`, `_offset_path_cmds`, unused `_render_scene_json` from `_interactive.py`
- [ ] Delete corresponding imports/tests from `tests/test_bug_hunt_phase_11_interactive.py`
- [ ] Remove `_renders_in_phase_8a` from `ChannelBase` and all subclasses in `encoding/`
- [ ] Delete `_renders_in_phase_8a` assertions from `tests/test_encoding.py`
- [ ] Remove unused `ClassVar` import from `marks/base.py`
- [ ] Delete dead `_columns_and_array` function from `_diagnostics/visualizers/ranking.py`
- [ ] Delete stale `_diagnostics/__pycache__/stats.cpython-310.pyc` if present
- [ ] Verify: `uv run pytest tests/test_encoding.py tests/test_bug_hunt_phase_11_interactive.py tests/marks/ -v`

### Task 2: Fix silent data loss & lying annotations (Q2, Q3, F3, F4)
- [ ] Add `"blend"` to the forwarding iteration list in `marks/base.py:to_mark_kwargs_dict()`
- [ ] Fix all `-> tuple` return annotations to `-> MarkDesugarResult` in `composite.py` and `heavy_stat.py`
- [ ] Narrow bare `except Exception: pass` in `chart.py` (lines ~530, ~2022, ~2037) to specific expected exceptions (`TypeError`, `ValueError`, `AttributeError` as appropriate)
- [ ] Verify: `uv run pytest tests/marks/ tests/test_chart.py -v`

### Task 3: Diagnostic mark dedup & naming (Q1, M7, H6, F2, F7, F16)
- [ ] Delete duplicate `desugar_silhouette`, `desugar_pca_scree`, `desugar_pca_scree_with_threshold` from `_selection.py`; keep canonical copies in `_clustering.py`
- [ ] Delete duplicate `desugar_class_prediction_error` from `_selection.py`; keep canonical in `_classification.py`
- [ ] Update `marks/diagnostic/__init__.py` imports to source all 4 from their canonical modules
- [ ] Rename `identity_line` → `reference_line` in `_regression.py:desugar_prediction_error` (no deprecation needed — not public yet)
- [ ] Rename `reference_lines` → `reference_line` in `_classification.py:desugar_gain` (no deprecation needed)
- [ ] Add `**mark_kwargs` with `validate_user_mark_kwargs` / `apply_user_mark_kwargs` to all desugar functions in `_classification.py`, `_regression.py`, `_explanation.py` that lack it — match the pattern in `_selection.py` / `_clustering.py`
- [ ] Verify: `uv run pytest tests/marks/diagnostic/ tests/test_pipeline_regression.py -v`

### Task 4: Plots boilerplate dedup (M1, M2, F5, F6)
- [ ] Add `_finalize_chart(chart, *, mark=None, encode=None, properties=None, layers=None, theme=None) -> Chart` to `plots/_helpers.py`
- [ ] Replace all 66 closing sequences across 8 plot modules with `return _finalize_chart(chart, ...)`
- [ ] Delete the 4 duplicate `_resolve_source` wrappers from `classification.py`, `regression.py`, `clustering.py`, `explanation.py`; replace with direct `from ferrum.plots._helpers import _resolve_source`
- [ ] Verify: `uv run pytest tests/ -x -q`

### Task 5: chart.py decomposition (H1, H2, H3, M3, M4, F1, F15)
- [ ] Extract `_warn_large_chart(mark_count)` helper to deduplicate 3 identical warning strings in `_apply_auto_raster`
- [ ] Extract `_apply_remap(encoding, remap, preserve_title=True)` helper to deduplicate 4 remap-application blocks
- [ ] Create `src/ferrum/_render.py`: move `_with_raster_override`, `_render_inputs`, `_apply_auto_raster`, `show_svg`, `show_png`, `show`, `save`, `_repr_svg_`, `_repr_html_` out of `Chart` class — implement as mixin class or standalone functions that `Chart` delegates to
- [ ] Move `_channel_class_map`, `_channel_class_for`, `_apply_channel_aliases` from `chart.py` module-level into `encoding/__init__.py`
- [ ] Move `_expand_layers`, `_merge_top_transforms`, `_warn_on_layer_conflicts` from `chart.py` module-level into `composition.py`
- [ ] Update all import sites in `chart.py` to import from new locations
- [ ] Verify: `uv run pytest tests/ -x -q` — full suite green
- [ ] Verify: `python -c "from ferrum.chart import Chart; print(Chart.show_svg)"` — import path preserved

### Task 6: Supporting module cleanup (M5, M6, M8, F8, F13, F14)
- [ ] Create `src/ferrum/_metric_labels.py`: move `AUCLabel`, `APLabel`, `BrierLabel`, `OutlierLabel`, `_apply_metric_label`, `_apply_metric_label_explicit`, `_trapezoid_auc`, `_ap_step`, `_brier_score` from `annotations.py`
- [ ] Update `annotations.py` to re-export the label classes from `_metric_labels` for backward compat
- [ ] Replace `_warn.py` global `_seen: set` with a `contextvars.ContextVar[set]` — same API surface but scoped, not process-global
- [ ] Standardize serialization method names: `coord.py:_to_spec_dict` → `to_spec_dict`; `themes/__init__.py:to_theme_inputs_dict` → `to_spec_dict`; `marks/base.py:to_mark_kwargs_dict` → `to_spec_dict`; `selection.py:SelectionMark.to_dict` → `to_spec_dict`
- [ ] Update all call sites for renamed methods in `chart.py`, `composition.py`, `_interactive.py`
- [ ] Verify: `uv run pytest tests/ -x -q`

### Task 7: Return type annotations & docstrings (H5, F10)
- [ ] Add return type annotations to all `*_chart` figure functions in `plots/`
- [ ] Document polymorphic returns (`pairplot` → `RepeatChart`, `clustermap` → `ClusterMapChart`, `jointplot` → `JointChart`, `cluster_diagnostics("both")` → `HConcatChart`) in their docstrings
- [ ] Add return annotations to `mark_arc`, `mark_image`, `mark_geoshape`, `mark_label` in `chart.py`
- [ ] Verify: `uv run pytest tests/ -x -q`

### Task 8: Figure function signature standardization (F17, F18, misc)
- [ ] Standardize first positional param name across all `*_chart` figure functions: model-backed functions use `model`, data-only functions use `data`. Rename `model_or_source` → `model`, `data_or_source` → `data`, `model_class` → `model` throughout `plots/`
- [ ] Update corresponding `_*_from_source` builders and their docstrings
- [ ] Move `_scale_to_dict` from `encoding/base.py` to `encoding/_scale.py`; update the one import in `encoding/base.py`
- [ ] Replace `_DefaultThemeCM` class in `themes/_defaults.py` with a `@contextlib.contextmanager` function
- [ ] Verify: `uv run pytest tests/ -x -q`

## 6. Acceptance checks

- `uv run pytest tests/ -x -q` — all pass, zero new failures
- `grep -rn "except Exception" src/ferrum/chart.py` — zero bare catches remain
- `grep -rn "_renders_in_phase_8a" src/ferrum/` — zero results
- `grep -rn "merge_scene_graphs" src/ferrum/` — zero results
- `wc -l src/ferrum/chart.py` — under 4200 lines (down from 5231)
- `grep -c "_finalize_chart" src/ferrum/plots/*.py` — ≥60 (replaced closing sequences)
- `python -c "from ferrum import Chart; c = Chart({'x':[1],'y':[2]}).mark_point().encode(x='x',y='y'); print(c.show_svg()[:20])"` — prints `<svg`

## 7. Open questions

- **`blend` Rust forwarding**: does `ChartSpec` on the Rust side already accept `blend` in its mark kwargs dict? If not, Task 2 needs a Rust-side one-liner too. Verify with `grep -rn "blend" crates/ferrum-core/src/`.
- **`mark_function` refactor (H4/F11)**: omitted from this plan — it requires rethinking the data-creation escape hatch, which is better as a standalone design decision. Flag for a follow-up session.
- **Elbow sweep duplication** between `cluster_diagnostics` and `ElbowVisualizer.fit`: both independently implement model-fit-per-k loops. Straddles plots/diagnostics boundary — dedup requires deciding which layer owns the loop. Flag for follow-up.
