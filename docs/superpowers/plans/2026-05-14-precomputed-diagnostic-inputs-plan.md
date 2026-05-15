# Precomputed Diagnostic Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Add `y_true` + `y_pred` keyword-only args to the nine prediction-evaluation diagnostic figure functions so callers can bypass the model entirely and pass precomputed arrays.

## 2. Spec references

- `docs/superpowers/specs/2026-05-14-precomputed-diagnostic-inputs-design.md §4 System behavior`
- `§6 Canonical interfaces` — `_PrecomputedSource` protocol and per-function `y_pred` semantics table
- `§7 Invariants` — exactly-one-path rule, `compare=` exclusion, no `residuals=` kwarg
- `§9 Acceptance criteria`

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `src/ferrum/_diagnostics/precomputed.py` | `_PrecomputedSource` class |
| Modify | `src/ferrum/plots/_helpers.py` | add precomputed branch + validation to `_resolve_source` |
| Modify | `src/ferrum/plots/classification.py` | update 7 function signatures |
| Modify | `src/ferrum/plots/regression.py` | update `residuals_chart` signature |
| Modify | `ferrum-spec.md` | update §3.14 signatures for all nine functions |
| Create | `tests/diagnostics/test_precomputed_inputs.py` | new test file |

## 4. Constraints

- `model_or_source` default `None` must not break existing positional calls `fn(model, X, y)`.
- `_PrecomputedSource` is never exported — not in `ferrum/__init__.py`, not in `ferrum-spec.md §3.1`.
- `stat_*` spec-level transforms do not exist yet; `_PrecomputedSource` calls `sklearn.metrics.*` directly.
- `compare=` + precomputed path → `ValueError`. See spec §7.
- `residuals_chart` precomputed: residuals = `y_true − y_pred` inside the function; no `residuals=` kwarg.
- `y_pred` is the sole new array param — no `y_score`, `y_proba`, or `y_prob` aliases.

## 5. Tasks

### Task 1: Create `_PrecomputedSource`

- [ ] Create `src/ferrum/_diagnostics/precomputed.py` with `_PrecomputedSource(y_true, y_pred)`
- [ ] Implement all eight methods per spec §6 protocol, delegating to `sklearn.metrics.*`; each returns a `polars.DataFrame` with column names matching `ModelSource` equivalents
- [ ] `predictions()` for `residuals_chart` / `class_prediction_error`: emits `y_true`, `y_pred`, `residual` (= `y_true − y_pred`); `studentized_residual`, `cooks_distance`, `leverage` as all-NaN columns (matching `ModelSource` NaN behavior for linear models without matrix info)
- [ ] Verify: `python -c "from ferrum._diagnostics.precomputed import _PrecomputedSource; print('OK')"`

### Task 2: Update `_resolve_source` in `_helpers.py`

- [ ] Add `y_true` and `y_pred` params to `_resolve_source` signature
- [ ] Add validation for exactly-one-path rule (neither / both / incomplete pair) — all `ValueError`
- [ ] Add `compare=` + precomputed guard → `ValueError`
- [ ] When `y_true` is not `None`, return `_PrecomputedSource(y_true, y_pred)`
- [ ] Verify: validation errors raise correctly via `python -c` smoke tests

### Task 3: Update classification figure function signatures

- [ ] In `classification.py`: add `y_true=None, y_pred=None` keyword-only args to `roc_chart`, `pr_chart`, `calibration_chart`, `gain_chart`, `lift_chart`, `discrimination_threshold_chart`, `confusion_matrix_chart`, `class_prediction_error_chart`
- [ ] Each function passes `y_true`/`y_pred` through to `_resolve_source`
- [ ] Update each function's docstring with `y_pred` semantics (see spec §6 table)
- [ ] Verify: `uv run python -c "import ferrum; help(ferrum.roc_chart)"` shows new params

### Task 4: Update `residuals_chart` signature

- [ ] In `regression.py`: add `y_true=None, y_pred=None` keyword-only args to `residuals_chart`
- [ ] Pass through to `_resolve_source`; confirm residuals column is computed inside `_PrecomputedSource.predictions()`
- [ ] Verify: `uv run python -c "import ferrum; help(ferrum.residuals_chart)"` shows new params

### Task 5: Write tests

- [ ] Create `tests/diagnostics/test_precomputed_inputs.py`
- [ ] For each of the nine functions: one model-path test (existing call shape) + one precomputed-path test
- [ ] Parametrize binary vs. multiclass for curve functions (`roc_chart`, `pr_chart`, `calibration_chart`) and matrix functions
- [ ] Three `ValueError` tests: neither path, both paths, incomplete pair (y_true without y_pred)
- [ ] `residuals_chart` precomputed test: assert all four panels present in returned `Chart`
- [ ] Verify: `uv run pytest tests/diagnostics/test_precomputed_inputs.py -v`

### Task 6: Update `ferrum-spec.md`

- [ ] Update §3.14 signatures for all nine functions to show `y_true=None, y_pred=None`
- [ ] Add a brief note explaining the precomputed path and linking to `y_pred` semantics table
- [ ] Verify: `grep -n "y_true" ferrum-spec.md` shows all nine functions updated

## 6. Acceptance checks

- `uv run pytest tests/diagnostics/test_precomputed_inputs.py -v` — all pass
- `uv run pytest tests/diagnostics/ -v` — existing tests unmodified and passing
- `uv run pytest -x` — full suite green
