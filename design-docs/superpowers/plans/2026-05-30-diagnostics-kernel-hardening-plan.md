# Diagnostic Kernel Hardening + Source Consolidation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Close the review/audit findings on the diagnostic-kernel branch: two behavior-neutral Rust hardening changes (null-reject boundary guard, unified PR core), and two Python cohesion changes (drop the error-masking `except Exception` guards, then collapse the duplicated per-source assembly glue into one shared module that both diagnostic sources delegate to).

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-29-diagnostics-rust-curve-kernels-design.md §5 (architecture; single-source-of-truth), §6 (kernel contracts), §7 (byte-parity)`
- Findings: rust-review (S3 null bitmap, S2 PR duplication), python-review (S3 parallel-API drift, S3 error-masking guards), PyO3 audit WARN (`.values()` ignores validity bitmap; broad `except` masks dtype errors).

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/diagnostics.rs` | Null guard in extraction helpers; extract `reversed_pr` |
| Modify | `src/ferrum/_diagnostics/precomputed.py` | Drop `except` guards; delegate assembly to shared module |
| Modify | `src/ferrum/_diagnostics/sources/_classification.py` | Drop `except` guards; delegate assembly to shared module |
| Create | `src/ferrum/_diagnostics/_curve_frames.py` | Single home for kernel-calling frame assembly shared by both sources |

No `lib.rs` / `_core.pyi` changes. No public-API change — both source classes keep their method signatures and output schemas.

## 4. Constraints

- **Byte-parity preserved throughout.** No change to any frame's values/columns for valid inputs. The existing Rust unit tests, Python parity tests (`test_rust_curve_parity.py`), golden tests (`test_goldens_phase_10.py`), and `test_bug_hunt_model_diagnostics.py` are the byte-parity guard and must pass unchanged after every task.
- **Only two intentional behavior deltas, both isolated and named:**
  - Task 1: null-containing Arrow input → explicit `PyValueError` instead of silent misread (unreachable from current callers).
  - Task 3: a genuine kernel error (dtype/length) now surfaces instead of becoming a silent NaN. Degenerate single-class still yields NaN AUC / 0.0 AP — via the kernel's *native* return, not a guard.
- **Fix 2 recall convention** must stay `recall = if total_pos > 0 { tps/total_pos } else { 1.0 }`; `average_precision_core` early-returns `0.0` on `total_pos <= 0` so it never hits the `else` — the guarded form is exact for both callers.
- **Consolidation is a pure structural move** (Task 4). Label dtype-coercion stays source-side (each source knows its label dtype — `_coerce_class_label` for the model path, `str(class)` for precomputed); the shared module receives already-resolved class labels + arrays. NaN is a valid value, never rejected.
- **No re-drift:** the shared module defines ONE canonical name per concept (below); neither source may keep a private duplicate afterward.
- Rust task → `rust-coder`; Python tasks → `python-coder`. After Rust changes, rebuild with `unset CONDA_PREFIX && uv run --no-sync maturin develop` before running Python tests.

## 5. Tasks

### Task 1: Null-reject guard at the Arrow boundary (Rust)
- [ ] Reject null-containing inputs in `as_f64_slice`/`as_i64_slice` (a shared `check_no_nulls(arr, name)` is fine): `PyValueError("{name} must not contain nulls")` when `null_count() > 0`.
- [ ] Tests: null-containing array → error path; NaN-but-no-null → still succeeds.
- [ ] Verify: `cd /Users/chrissantiago/Dropbox/GitHub/ferrum && source ~/.cargo/env && unset CONDA_PREFIX PYTHONPATH; VENV="$(uv run python -c 'import sys; print(sys.prefix)')"; BASE="$(uv run python -c 'import sys; print(sys.base_prefix)')"; PATH="$VENV/bin:$PATH" PYO3_PYTHON="$VENV/bin/python3" PYTHONHOME="$BASE" RUSTFLAGS="-L $BASE/lib" DYLD_LIBRARY_PATH="$BASE/lib" cargo test -p ferrum-core --lib diagnostics`

### Task 2: Extract `reversed_pr` shared core (Rust)
- [ ] Add `fn reversed_pr(fps, tps, total_pos) -> (Vec<f64>, Vec<f64>)` (reversed precision/recall, no endpoint), with the recall convention in §4.
- [ ] `pr_curve_core` consumes it (builds reversed `threshold` itself, appends `(1.0, 0.0, NaN)` endpoint); `average_precision_core` consumes it (keeps `total_pos<=0 → 0.0` early-return, appends `(1.0, 0.0)`, sums). Outputs unchanged.
- [ ] Verify: same `cargo test … --lib diagnostics`. (Same file as Task 1 — serialize after Task 1, or one agent does both.)

### Task 3: Remove error-masking `except Exception` guards (Python)
- [ ] In `precomputed.py` and `_classification.py`, remove the `try: float(roc_auc/average_precision(...)) except Exception: nan` wrappers around the per-class and binary AUC/AP calls (precomputed.py:390-393, 435-438, 477-480, 525-528; _classification.py:424-427, 452-455, 480-483, 500-503, 541-544, 568-571, 604, 624-627). Call the kernels directly, matching the already-unguarded micro path. Degenerate cases rely on the kernel's native NaN/0.0.
- [ ] Verify: `uv run pytest tests/diagnostics tests/test_bug_hunt_model_diagnostics.py -n auto` — all pass (degenerate single-class AUC must still be NaN).

### Task 4: Consolidate assembly into shared `_curve_frames.py` (Python)
- [ ] Create `src/ferrum/_diagnostics/_curve_frames.py` exposing canonically-named builders that take resolved arrays + class labels and return the diagnostic frames: `one_hot`, `roc_frame`, `pr_frame`, `calibration_frame`, `confusion_frame` (owns integer encode/decode + `value_fmt`), `threshold_sweep_frame`. These call the `ferrum._core` kernels and produce the exact existing output schemas (§ spec 6).
- [ ] Rewrite `_PrecomputedSource` and `ClassificationCurvesMixin` to obtain their arrays (model path keeps sklearn for `probabilities()`/refit) then delegate all frame assembly to `_curve_frames`. Delete the per-source duplicate helpers (`_one_hot`/`_label_binarize`, `_roc_frame_binary`/`_roc_binary`/`_roc_one_class`/`_roc_average`, `_pr_*`, `_confusion_matrix_columnar`, `_sweep_thresholds`, the inlined confusion/threshold blocks). Pick one weighted-scalar form (`np.average(weights=)`).
- [ ] Verify: `uv run pytest tests/diagnostics tests/test_bug_hunt_model_diagnostics.py tests/test_render_smoke.py -n auto` — all pass; `grep -rn "_label_binarize\|_roc_frame_binary\|_roc_binary\|_avg_roc_frame\|_roc_average" src/ferrum/_diagnostics/precomputed.py src/ferrum/_diagnostics/sources/_classification.py` returns nothing (helpers gone). (Same two files as Task 3 — runs after Task 3.)

## 6. Acceptance checks

- `cargo test -p ferrum-core --lib diagnostics` + `cargo clippy -p ferrum-core -- -D warnings` — clean (incl. new null-guard tests).
- Full `uv run pytest -n auto` — all pass; **zero golden SVG diffs** (byte-parity).
- No `except Exception`→NaN guard remains around the AUC/AP kernel calls.
- Each diagnostic concept (one-hot, roc/pr/calibration/confusion/threshold assembly) has exactly ONE implementation in `_curve_frames.py`; no per-source duplicate.
- `precomputed.py` still imports zero sklearn; `_classification.py` keeps sklearn only for `probabilities()`/CV-refit.

## 7. Open questions

- (none — the NaN-score-in-`binary_clf_curve`-sort robustness concern from the audit remains a separate, distinct item, out of scope here.)
