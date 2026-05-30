# Diagnostic Curve Kernels in Rust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Move the five scikit-learn metric computations (ROC, PR, calibration, confusion matrix, threshold sweep) into Rust kernels returning Arrow, so the precomputed `(y_true, y_pred)` chart path needs no scikit-learn and the model-backed path shares the same curve math.

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-29-diagnostics-rust-curve-kernels-design.md §5 Architecture`
- `…§6 Canonical interfaces / data contracts` (kernel signatures + DataFrame schemas)
- `…§7 Invariants and constraints` (byte-parity conventions)
- `…§8 Key decisions and tradeoffs`

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/diagnostics.rs` | Add 5 curve kernels + 2 scalar-metric fns |
| Modify | `crates/ferrum-core/src/lib.rs` | Register new pyo3 functions |
| Modify | `src/ferrum/_diagnostics/precomputed.py` | Route roc/pr/calibration/confusion/threshold through kernels; columnar gain/lift |
| Modify | `src/ferrum/_diagnostics/sources/_classification.py` | Route model-path curve math through kernels; drop `sklearn.metrics` on metric step |
| Test | `crates/ferrum-core/src/diagnostics.rs` (`#[cfg(test)]`) | Kernel unit tests |
| Test | `tests/diagnostics/test_rust_curve_parity.py` | Parity vs sklearn across binary/multiclass/degenerate/tie inputs |
| Test | `tests/diagnostics/test_no_sklearn_precomputed.py` | Precomputed curves run with sklearn absent |

## 4. Constraints

- **Byte-parity with scikit-learn** — existing goldens must not shift. Replicate sklearn conventions per spec §7: roc leading-threshold sentinel + `drop_intermediate` collinear pruning; pr reversed-cumsum + trailing `(1,0)` endpoint + threshold one shorter (adapter NaN-pads); `average_precision` step sum not trapezoidal; `roc_auc` trapezoidal; calibration empty-bin drop + uniform/quantile edges; confusion sorted labels + `normalize ∈ {true,pred,all,None}`.
- **No `sklearn` import on the precomputed curve path** — not at module level, not in method bodies for the five functions.
- **No `list[dict]`** in ported functions or in gain/lift/cumulative-gain.
- Averaging (micro/macro/weighted) + grid interpolation stay in Python, columnar (spec §8). Only per-curve kernels + scalar metrics in Rust.
- Confusion kernel takes integer-encoded labels + sorted `labels` array; encode/decode + `value_fmt` stay in Python adapter.
- Kernels emit numeric core only; adapter attaches `class`, broadcasts scalar metric.
- Python-facing DataFrame schemas unchanged (spec §6).
- `models`/`shap`/`all` extras keep pinning `scikit-learn`; do not remove. Coding tasks → `rust-coder` (`.rs`) / `python-coder` (`.py`).

## 5. Tasks

### Task 1: Rust curve kernels
- [ ] Add to `diagnostics.rs`: `roc_curve_kernel`, `roc_auc`, `pr_curve_kernel`, `average_precision`, `calibration_kernel`, `confusion_kernel`, `prf_at_thresholds` — signatures + output columns per spec §6; Arrow boundary mirrors `hat_matrix_stats`/`studentized_residual_no_x`.
- [ ] Replicate sklearn conventions per spec §7 (Constraints above).
- [ ] Unit tests covering ties, single-class, all-correct, empty quantile bins, `drop_intermediate` on/off.
- [ ] Register all in `lib.rs`.
- [ ] Verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` and `cargo clippy -p ferrum-core -- -D warnings`

### Task 2: Precomputed path → kernels
- [ ] Rebuild extension: `unset CONDA_PREFIX && uv run --no-sync maturin develop`
- [ ] In `precomputed.py`, replace sklearn calls in `roc_curve`/`pr_curve`/`calibration_curve`/`confusion_matrix`/`discrimination_threshold` with kernel calls; remove `require_sklearn` + `from sklearn…` on these methods; build frames from Arrow / columnar.
- [ ] Rewrite `cumulative_gain`/`lift_curve` row-dict loops to vectorized numpy → polars (no math change).
- [ ] Verify: `uv run pytest tests/diagnostics -n auto`

### Task 3: Model path → kernels
- [ ] In `_classification.py`, route curve math (after score extraction) through the same kernels; keep `predict_proba`/`decision_function` + averaging helpers; no `list[dict]`.
- [ ] Verify: `uv run pytest tests/diagnostics tests/test_render_smoke.py -n auto`

### Task 4: Parity + dependency-isolation tests
- [ ] `test_rust_curve_parity.py`: assert each kernel == sklearn under spec §7 conventions across binary/multiclass/degenerate/tie inputs; assert model-path and precomputed-path frames identical for same `(y_true, scores)`.
- [ ] `test_no_sklearn_precomputed.py`: drop `sklearn` from `sys.modules`, exercise all 5 precomputed curves end-to-end, assert no re-import.
- [ ] Verify: `uv run pytest tests/diagnostics/test_rust_curve_parity.py tests/diagnostics/test_no_sklearn_precomputed.py -v`

### Task 5: Golden + full-suite confirmation
- [ ] Run golden suite; expect zero shifted goldens. If any shift, `python scripts/snapshot-goldens.py <name>`, Read PNG, confirm correct before accepting.
- [ ] Verify: `uv run pytest -n auto` and `DYLD_LIBRARY_PATH=… cargo test`

## 6. Acceptance checks

- `uv run pytest tests/diagnostics -n auto` — all pass
- `uv run pytest tests/diagnostics/test_no_sklearn_precomputed.py -v` — 5 precomputed charts run with sklearn absent
- `cargo test` + `cargo clippy -p ferrum-core -- -D warnings` — clean
- `test_no_sklearn_at_import.py` still passes
- Existing goldens unchanged (byte-identical)
- No `list[dict]` in ported functions or gain/lift/cumulative

## 7. Open questions

- (none — exact sklearn threshold sentinel + `drop_intermediate` rule resolved empirically by the parity harness against the pinned sklearn version, per spec §11)
