# Polars Cleanup + Rust Linear Algebra Consolidation

**Date:** 2026-05-13
**Status:** Proposed

## Problem

Three separate issues converge:

1. **`charts.py`, `annotations.py`, and `_direct_label.py` still use numpy for trivial polars-replaceable arithmetic** — R²/RMSE/MAE corner metrics, SHAP waterfall cumsum, metric-label endpoint detection, outlier z-scoring, and direct-label collision staggering all operate on column-level data via a triple-copy anti-pattern (`Arrow → Python list → numpy`) when polars handles every operation natively.

2. **`stats.py` is 381 lines of numpy linear algebra** that should live in Rust — hat matrix, Pearson/Spearman correlation, rankdata, Shapiro-Wilk, and the rank1d/rank2d orchestrators. Every call requires copying the full DataFrame to a contiguous numpy array, and `_predictions.py` computes the hat matrix three separate times for the same data. At 500K+ rows this is the bottleneck.

3. **The Rust crate has three identical `invert_2x2` functions** across `glm.rs`, `logistic.rs`, and `robust.rs`, plus a hand-rolled 3x3 Cholesky in `linalg.rs`, and hand-rolled distance metrics in `linkage.rs`. These should use a shared linear algebra foundation.

## Decision: `faer` (pure-Rust linalg)

Use [`faer`](https://crates.io/crates/faer) instead of `ndarray-linalg`. `faer` provides Cholesky, QR, SVD, eigendecomposition, and matrix operations in pure Rust with zero external dependencies — no LAPACK, no OpenBLAS, no platform-specific linking, no wheel-size increase. At the matrix sizes in ferrum (2x2 through ~100x100 for rank2d correlation), `faer` matches BLAS performance. The build stays simple: one more `Cargo.toml` line, zero CI changes.

The hat matrix diagonal `h_ii = diag(X (X'X)⁻¹ X')` does NOT require pseudoinverse. Compute via QR decomposition: `QR(X) → h_ii = ||Q[i,:]||²` (row norms of the Q factor). This is numerically superior to the current `pinv + einsum` approach and avoids SVD entirely.

## Structure

Three independent tiers. Each can be committed and tested separately.

---

## Tier 1 — Polars cleanup in `charts.py` (Python-only, no Rust)

Replace remaining numpy arithmetic with polars-native operations. No dependency on Tiers 2–3.

### Pattern A: R²/RMSE/MAE corner metrics (lines 66–119)

Before:
```python
import numpy as np
y_true = np.asarray(df["y_true"].to_list(), dtype=float)
y_pred = np.asarray(df["y_pred"].to_list(), dtype=float)
ss_res = float(np.sum((y_true - y_pred) ** 2))
ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
r2 = 1.0 - ss_res / ss_tot
rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
mae = float(np.mean(np.abs(y_true - y_pred)))
anchor_idx = int(np.argmax(y_pred))
resid_arr = np.asarray(df[y_col].to_list(), dtype=float)
y_col_vals[anchor_idx] = float(np.max(resid_arr))
```

After:
```python
diff = df["y_pred"] - df["y_true"]
ss_res = float((diff ** 2).sum())
mean_y = float(df["y_true"].mean())
ss_tot = float(((df["y_true"] - mean_y) ** 2).sum())
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
rmse = float((diff ** 2).mean() ** 0.5)
mae = float(diff.abs().mean())
anchor_idx = df["y_pred"].arg_max()
y_col_vals[anchor_idx] = float(df[y_col].max())
```

The `_r2_score` helper becomes a pure-polars two-liner. Remove `import numpy as np` from both functions.

### Pattern C: SHAP waterfall cumsum (lines 1363–1379)

Before:
```python
sv_arr = np.asarray(ordered["shap_value"])
cum = np.concatenate([[0.0], np.cumsum(sv_arr)])
plot_df = ordered.with_columns([
    pl.Series("x0", cum[:-1]),
    pl.Series("x1", cum[1:]),
    ...
])
x_lo = float(min(cum.min(), 0.0))
x_hi = float(max(cum.max(), 0.0))
```

After:
```python
cumsum = ordered["shap_value"].cum_sum()
x0 = pl.concat([pl.Series([0.0]), cumsum.head(cumsum.len() - 1)])
x1 = cumsum
plot_df = ordered.with_columns([
    x0.alias("x0"),
    x1.alias("x1"),
    ...
])
x_lo = float(min(x0.min(), x1.min(), 0.0))
x_hi = float(max(x0.max(), x1.max(), 0.0))
```

### Pattern B: `annotations.py` metric-label endpoint detection + outlier z-scoring (lines 289–421)

`_apply_metric_label` and `OutlierLabel.__radd__` pull Arrow columns into numpy via the triple-copy anti-pattern `np.asarray(tbl.column(col).to_pylist(), dtype=float)`. All operations are column-level: `argmax`, boolean masking, `where`, z-score, `argsort`.

Before (`_apply_metric_label`, lines 289–316):
```python
x_arr = np.asarray(tbl.column(x_col).to_pylist(), dtype=float)
y_arr = np.asarray(tbl.column(y_col).to_pylist(), dtype=float)
y_range = float(np.nanmax(y_arr) - np.nanmin(y_arr))
y_top = float(np.nanmax(y_arr))
color_vals = np.asarray(tbl.column(color_col).to_pylist())
mask = color_vals == cls
mask_idxs = np.where(mask)[0]
idx_in_mask = int(np.argmax(x_arr[mask]))
global_idx = int(mask_idxs[idx_in_mask])
```

After:
```python
df = pl.from_arrow(tbl)
y_range = float(df[y_col].max() - df[y_col].min())
y_top = float(df[y_col].max())
for cls in df[color_col].unique().sort():
    group = df.filter(pl.col(color_col) == cls)
    if group.is_empty():
        continue
    idx_in_group = group[x_col].arg_max()
    global_idx = int(group.row_nr()[idx_in_group])  # or track via with_row_index
    ...
```

Use `df.with_row_index("_idx")` before filtering to preserve the mapping from group-local position back to the original row index.

Before (`OutlierLabel.__radd__`, lines 413–421):
```python
values = np.asarray(tbl.column(field).to_pylist(), dtype=float)
mu = float(np.mean(values))
sigma = float(np.std(values, ddof=1)) or 1.0
z = np.abs((values - mu) / sigma)
mask = z > self.threshold
candidate_idx = np.where(mask)[0]
ordered = candidate_idx[np.argsort(-z[candidate_idx])][: self.max_labels]
```

After:
```python
df = pl.from_arrow(tbl).with_row_index("_idx")
mu = float(df[field].mean())
sigma = float(df[field].std(ddof=1)) or 1.0
df = df.with_columns(
    ((pl.col(field) - mu).abs() / sigma).alias("_z")
)
outliers = (
    df.filter(pl.col("_z") > self.threshold)
    .sort("_z", descending=True)
    .head(self.max_labels)
)
```

The AUC/AP/Brier helper functions (`_trapezoid_auc`, `_ap_step`, `_brier_score`) operate on small curve-resolution arrays (100–500 points) where numpy reads naturally and performance is irrelevant. Leave them as-is — they stay as lazy-imported numpy inside `annotations.py`.

### Pattern D: `_direct_label.py` endpoint collision detection (lines 76–103)

Same triple-copy anti-pattern. Pulls Arrow columns into numpy for per-series `argmax`/`argmin`, boolean masking, and y-range stagger computation.

Before:
```python
series_arr = np.asarray(tbl.column(label_field).to_pylist())
x_all = np.asarray(tbl.column(x_col).to_pylist(), dtype=float)
y_all = np.asarray(tbl.column(y_col).to_pylist(), dtype=float)
for series in series_list:
    mask = series_arr == series
    masked_x = x_all[mask]
    idx_in_mask = int(np.argmax(masked_x))
    global_idx = int(np.where(mask)[0][idx_in_mask])
    ep_y = float(y_all[global_idx])
y_range = float(np.nanmax(y_all) - np.nanmin(y_all))
```

After:
```python
df = pl.from_arrow(tbl).with_row_index("_idx")
y_range = float(df[y_col].max() - df[y_col].min()) if df.height > 0 else 1.0
series_endpoints = []
for series in df[label_field].unique().sort():
    group = df.filter(pl.col(label_field) == series)
    if group.is_empty():
        continue
    if position == "end":
        row = group.sort(x_col, descending=True).row(0, named=True)
    else:
        row = group.sort(x_col).row(0, named=True)
    series_endpoints.append((
        str(series), int(row["_idx"]), float(row[x_col]), float(row[y_col])
    ))
```

After both `annotations.py` and `_direct_label.py` are converted, remove top-level `import numpy as np` from each file if no remaining usage. `annotations.py` keeps a lazy numpy import inside the AUC/AP/Brier helpers only.

### Import cleanup

After all patterns, check each file for remaining numpy usage:
- `charts.py` — remove top-level import; decision-boundary mesh grid uses lazy `import numpy as np` inside its function body.
- `annotations.py` — remove top-level import; AUC/AP/Brier helpers use lazy `import numpy as np` inside their function bodies.
- `_direct_label.py` — remove `import numpy as np` entirely.

---

## Tier 2 — Add `faer` to the Rust crate and consolidate hand-rolled linalg

### 2a. Add `faer` dependency

In `Cargo.toml` (workspace):
```toml
faer = "0.22"
```

In `crates/ferrum-core/Cargo.toml`:
```toml
faer = { workspace = true }
```

### 2b. Replace `linalg.rs` with a shared linalg module

Replace the hand-rolled 3x3 Cholesky in `linalg.rs` with a generic `faer`-backed module that provides:

```rust
// crates/ferrum-core/src/transform/linalg.rs

use faer::prelude::*;

/// Solve M x = b via Cholesky for an NxN SPD system.
/// Returns None if M is not positive-definite.
pub(crate) fn solve_spd(m: &Mat<f64>, b: &Col<f64>) -> Option<Col<f64>>;

/// Invert a 2x2 matrix via Cramer's rule.
/// Keep as-is — 2x2 Cramer is faster than faer dispatch overhead.
pub(crate) fn invert_2x2(a: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]>;

/// Hat matrix diagonal h_ii via QR decomposition.
/// Given X (n x p), computes h_ii = ||Q[i,:]||² where X = QR.
/// Returns Vec<f64> of length n.
pub(crate) fn hat_diagonal(x: &Mat<f64>) -> Vec<f64>;

/// Pearson correlation matrix for X (n x p).
/// Returns p x p Mat<f64>.
pub(crate) fn corrcoef(x: &Mat<f64>) -> Mat<f64>;

/// Pseudoinverse via thin SVD. Used only if QR path is insufficient
/// (rank-deficient X).
pub(crate) fn pinv(x: &Mat<f64>) -> Mat<f64>;
```

**Design note on `invert_2x2`:** Keep the hand-rolled Cramer's rule for 2x2 — it's 6 multiplies and a branch. Routing through `faer` for 2x2 adds dispatch overhead that exceeds the computation. But consolidate the three copies into a single `pub(crate)` function in `linalg.rs` and call it from `glm.rs`, `logistic.rs`, and `robust.rs`.

### 2c. Consolidate `invert_2x2` across modules

Delete the private `invert_2x2` from:
- `glm.rs:513–521`
- `logistic.rs:245–253`
- `robust.rs:307–315`

Replace with `use crate::transform::linalg::invert_2x2;` in each. Also move `scale_2x2` and `xtx_unweighted` from `robust.rs` to `linalg.rs` — they're general-purpose 2x2 utilities.

### 2d. Replace 3x3 Cholesky in `smooth.rs`

`smooth.rs:678` calls `solve_3x3_spd`. Replace the fixed-size implementation with `solve_spd` which handles any NxN SPD system via `faer::Cholesky`. This future-proofs for higher-degree LOESS (degree > 2) without adding another hand-rolled solver.

### 2e. Vectorize `linkage.rs` distance metrics with `faer`

Replace the hand-rolled loops in `condensed_distances` (lines 165–242) for Cosine and Correlation metrics with `faer` matrix operations:

- **Cosine:** Compute norms via `faer::Col::norm_l2()` and dot products via column dot.
- **Correlation:** Center columns, compute `X_centered.transpose() * X_centered`, normalize by column norms.
- **Euclidean / Manhattan / Chebyshev:** Keep as hand-rolled — these are element-wise and `faer` adds no value.

For n > ~2000 observations, precompute the Gram matrix `X X'` and derive distances from it: `d_euclidean(i,j)² = G_ii + G_jj - 2G_ij`. This turns O(n² × feat) into O(n × feat + n²) — a significant win at scale when feat > 10.

---

## Tier 3 — Move `stats.py` into Rust

Eliminate `src/ferrum/_diagnostics/stats.py` entirely. All functions move to a new Rust module `crates/ferrum-core/src/transform/stats.rs` exposed through PyO3.

### 3a. PyO3 boundary: Arrow in, Arrow out

The existing `kendall_tau_b` uses the worst boundary pattern — Python lists. All new stats functions accept and return Arrow arrays via `pyo3-arrow`, matching ferrum's CDI architecture. The Python-side callers pass polars DataFrames/Series directly; pyo3-arrow handles the zero-copy handoff.

```rust
// Exposed to Python via #[pyfunction]
#[pyfunction]
fn hat_matrix_stats(
    py: Python<'_>,
    x_table: PyArrowType<RecordBatch>,   // X design matrix (n x p)
    y_true: PyArrowType<ArrayData>,       // Float64Array
    y_pred: PyArrowType<ArrayData>,       // Float64Array
    has_intercept: bool,
) -> PyResult<PyArrowType<RecordBatch>>;
// Returns: studentized_residual, cooks_distance, leverage (all Float64)
```

This replaces three separate Python functions (`studentized_residual`, `cooks_distance`, and the inline leverage computation in `_predictions.py:55–62`) with a single Rust function that computes the hat matrix diagonal once.

### 3b. `hat_matrix_stats` — single-pass hat matrix

The current Python code in `_predictions.py:32–66` computes `pinv(X'X)` + `einsum` three times:
1. Inside `studentized_residual()` (line 49)
2. Inside `cooks_distance()` (line 50)
3. Inline for leverage (lines 55–62)

The Rust implementation computes the hat diagonal once via QR:

```rust
pub fn hat_matrix_stats(
    x: &Mat<f64>,       // n x p design matrix (intercept column included by caller)
    y_true: &[f64],
    y_pred: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // 1. QR(X) → h_ii = ||Q[i,:]||²
    let h = hat_diagonal(x);

    // 2. Residuals
    let r: Vec<f64> = y_true.iter().zip(y_pred).map(|(t, p)| t - p).collect();

    // 3. sigma² = SSE / (n - p)
    let n = x.nrows();
    let p = x.ncols();
    let sse: f64 = r.iter().map(|ri| ri * ri).sum();
    let sigma_sq = sse / (n - p).max(1) as f64;

    // 4. Studentized: r_i / (σ * sqrt(1 - h_ii))
    let sigma = sigma_sq.sqrt();
    let stud: Vec<f64> = ...;

    // 5. Cook's: (r_i² / (p · σ²)) · (h_ii / (1 - h_ii)²)
    let cooks: Vec<f64> = ...;

    // 6. Leverage = h_ii (already computed)
    (stud, cooks, h)
}
```

Complexity: O(n·p²) for QR, O(n·p) for row norms. One pass, not three.

### 3c. Correlation and ranking functions

New Rust functions exposed via PyO3:

```rust
/// Per-column Pearson correlation of X columns vs y.
/// X: Arrow RecordBatch (n x p Float64), y: Float64Array.
/// Returns: Float64Array of length p.
#[pyfunction]
fn pearson_r(x_table: ..., y: ...) -> PyResult<...>;

/// Per-column Spearman rho (Pearson on tied-average ranks).
#[pyfunction]
fn spearman_rho(x_table: ..., y: ...) -> PyResult<...>;

/// Pairwise correlation/covariance matrix in long form.
/// Returns RecordBatch with columns (feature_x, feature_y, correlation).
#[pyfunction]
fn rank2d(x_table: ..., algorithm: &str) -> PyResult<...>;

/// Univariate feature ranking.
/// Returns RecordBatch with columns (feature, score, rank).
#[pyfunction]
fn rank1d(x_table: ..., algorithm: &str, top_k: Option<usize>) -> PyResult<...>;

/// Shapiro-Wilk W statistic for a single Float64 column.
#[pyfunction]
fn shapiro_w(x: PyArrowType<ArrayData>) -> PyResult<f64>;

/// Average-rank tied rankdata for a Float64 array.
/// Returns Float64Array of same length.
#[pyfunction]
fn rankdata_average(x: PyArrowType<ArrayData>) -> PyResult<...>;

/// Variance per column (ddof=0). Returns Float64Array of length p.
#[pyfunction]
fn variance_rank(x_table: ...) -> PyResult<...>;

/// Abs covariance per column vs y (ddof=1). Returns Float64Array of length p.
#[pyfunction]
fn covariance_rank(x_table: ..., y: ...) -> PyResult<...>;
```

### 3d. Fix `kendall_tau_b` boundary

The existing Rust `kendall_tau_b` accepts Python lists. Refactor to accept Arrow arrays:

Before (Python caller):
```python
x64 = np.ascontiguousarray(x, dtype=np.float64).tolist()
y64 = np.ascontiguousarray(y, dtype=np.float64).tolist()
return float(_rust_ktb(x64, y64)["tau"])
```

After:
```python
return float(_core.kendall_tau_b(x_series, y_series))
```

The Rust side accepts `PyArrowType<ArrayData>` for both inputs.

### 3e. Move Kendall rank2d loop into Rust

The current `rank2d_compute(algorithm="kendall")` in Python loops p*(p-1)/2 times, each iteration round-tripping through Python→Rust. Move the entire pairwise loop into the Rust `rank2d` function — the Kendall branch iterates over column pairs in Rust and calls the existing Knight's O(n log n) algorithm directly, with optional rayon parallelism for p > 10.

### 3f. Eliminate `stats.py`

After Tiers 3a–3e, `stats.py` has no remaining functions. Delete it. Update callers:

| Caller | Before | After |
|---|---|---|
| `_predictions.py:32–66` | `studentized_residual()`, `cooks_distance()`, inline `pinv+einsum` for leverage | `_core.hat_matrix_stats(X, y_true, y_pred)` — single call returns all three |
| `_predictions.py:90` | `np.asarray(self._model.predict_proba(...))` | stays (sklearn boundary — unchanged) |
| `_ranking.py:28–51` | `rank1d_compute()`, `covariance_rank()` | `_core.rank1d(X, algorithm)`, `_core.covariance_rank(X, y)` |
| `_ranking.py:66–72` | `rank2d_compute()` | `_core.rank2d(X, algorithm)` |
| `figures.py:71` | `from ferrum._diagnostics.stats import rank1d_compute, rank2d_compute` | `from ferrum._core import rank1d, rank2d` |
| `figures.py:1947,2000` | same lazy imports | same pattern, new import source |

### 3g. Reduce numpy imports in `_predictions.py`

After `hat_matrix_stats` moves to Rust, `_predictions.py` still needs numpy for:
- `np.asarray(self._model.predict(...))` — sklearn returns numpy (unavoidable)
- `np.asarray(self._model.predict_proba(...))` — same
- `np.full_like(y_pred, np.nan)` — fallback for non-linear estimators

The numpy import stays, but the linear algebra imports (`from ferrum._diagnostics.stats import ...`) disappear.

`_ranking.py` loses its numpy import entirely — all numpy usage was for `covariance_rank` and `argsort`, both of which move to Rust.

---

## New Rust module layout

```
crates/ferrum-core/src/transform/
├── linalg.rs          # shared: invert_2x2, solve_spd, hat_diagonal, corrcoef, pinv
├── stats.rs           # NEW: hat_matrix_stats, pearson_r, spearman_rho, rank1d, rank2d,
│                      #       shapiro_w, rankdata_average, variance_rank, covariance_rank,
│                      #       kendall_tau_b (refactored boundary)
├── glm.rs             # uses linalg::invert_2x2 (deduped)
├── logistic.rs        # uses linalg::invert_2x2 (deduped)
├── robust.rs          # uses linalg::invert_2x2, linalg::xtx_unweighted, linalg::scale_2x2
├── smooth.rs          # uses linalg::solve_spd (replaces solve_3x3_spd)
├── linkage.rs         # uses linalg::corrcoef for Correlation metric, faer norms for Cosine
└── ...
```

---

## Verification plan

### Tier 1 (Python-only)
- `uv run pytest -x -q` — full suite, verify 0 failures.
- `grep -rn 'np\.' src/ferrum/_diagnostics/charts.py` — confirm only decision-boundary mesh remains.
- `grep -rn 'np\.' src/ferrum/annotations.py` — confirm only AUC/AP/Brier helpers (lazy imports inside function bodies) remain.
- `grep -rn 'np\.' src/ferrum/_direct_label.py` — confirm zero remaining numpy usage.

### Tier 2 (Rust consolidation)
- `cargo test` — existing linalg, glm, logistic, robust, smooth, linkage tests must pass.
- Add parity tests: `faer`-backed `solve_spd` vs hand-rolled `solve_3x3_spd` on existing test fixtures — results must be bit-identical (both are Cholesky on the same data).
- Regenerate goldens affected by smooth/robust/glm changes. Expect byte-identical SVGs — the math is the same, only the code path changes.

### Tier 3 (stats.py → Rust)
- Port `tests/diagnostics/test_stats.py` assertions to validate Rust implementations against the same scipy parity tolerances.
- Validate `hat_matrix_stats` produces identical `studentized_residual`, `cooks_distance`, and `leverage` columns as the current triple-pass Python implementation, at 1e-10 tolerance.
- Run `rank2d(algorithm="kendall")` on a p=50 dataset and verify results match current Python implementation.
- Benchmark: time `hat_matrix_stats` at n=500K, p=20. Expect >3x speedup from single-pass + zero-copy.
- Benchmark: time `rank2d(algorithm="kendall")` at n=100K, p=50. Expect >10x from eliminating 1,225 Python→Rust boundary crossings.
- `uv run pytest -x -q` — full suite, verify 0 failures.
- Confirm `stats.py` is deleted and no Python file imports from it.

## Risk

**Low for Tier 1** — trivial polars arithmetic replacements.

**Low for Tier 2** — `faer` is a well-tested crate (used by polars internally); the consolidation is mechanical deduplication with identical math.

**Medium for Tier 3** — new PyO3 boundary functions need careful Arrow type handling. Mitigated by parity tests against the existing numpy implementations before deleting `stats.py`.

## Non-goal

This spec does not propose removing numpy as a project dependency. Numpy remains needed for sklearn interop in the diagnostics subsystem (`predict`, `predict_proba`, `decision_function` all return numpy arrays) and for the decision-boundary mesh grid.
