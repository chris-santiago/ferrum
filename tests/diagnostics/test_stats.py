"""Scipy-parity tests for the Rust-native statistics in ferrum._core.

`scipy` is a dev/test dependency only — ferrum never imports it at
runtime. These tests validate that the Rust implementations match
scipy on a sweep of distributions / sample sizes / tie densities.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from ferrum._core import (
    kendall_tau_b,
    py_shapiro_w,
)
from ferrum._diagnostics._rank_helpers import _coerce_to_arrow_batch


def _pearson_r_rust(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Call Rust rank2d pearson to extract per-column Pearson r vs y."""
    # rank2d gives the full p×p matrix; we compare column-by-column with scipy
    # For per-column r vs y, we append y as an extra column, run pearson rank2d,
    # and read off the last row (y vs each X column).
    n, p = X.shape
    combined = np.column_stack([X, y])
    batch = _coerce_to_arrow_batch(combined)
    from ferrum._core import py_rank2d
    import polars as pl

    result = pl.from_arrow(py_rank2d(batch, "pearson"))
    # Extract correlation of each feature with the last column (y)
    y_col_name = f"f{p}"
    corrs = result.filter(pl.col("feature_y") == y_col_name).sort("feature_x")
    r_vals = [
        float(corrs.filter(pl.col("feature_x") == f"f{j}")["correlation"][0]) for j in range(p)
    ]
    return np.array(r_vals)


def _spearman_rho_rust(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, p = X.shape
    combined = np.column_stack([X, y])
    batch = _coerce_to_arrow_batch(combined)
    from ferrum._core import py_rank2d
    import polars as pl

    result = pl.from_arrow(py_rank2d(batch, "spearman"))
    y_col_name = f"f{p}"
    corrs = result.filter(pl.col("feature_y") == y_col_name).sort("feature_x")
    r_vals = [
        float(corrs.filter(pl.col("feature_x") == f"f{j}")["correlation"][0]) for j in range(p)
    ]
    return np.array(r_vals)


@pytest.fixture
def rng():
    return np.random.RandomState(0)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_pearson_parity_vs_scipy(n, rng):
    import scipy.stats as ss

    X = rng.randn(n, 4)
    y = rng.randn(n)
    ours = _pearson_r_rust(X, y)
    theirs = np.array([ss.pearsonr(X[:, j], y).statistic for j in range(X.shape[1])])
    np.testing.assert_allclose(ours, theirs, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_spearman_parity_vs_scipy(n, rng):
    import scipy.stats as ss

    X = rng.randn(n, 4)
    y = rng.randn(n)
    ours = _spearman_rho_rust(X, y)
    theirs = np.array([ss.spearmanr(X[:, j], y).statistic for j in range(X.shape[1])])
    np.testing.assert_allclose(ours, theirs, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("n", [10, 50, 200, 1000])
@pytest.mark.parametrize("dist", ["normal", "uniform", "exponential", "bimodal"])
def test_shapiro_parity_vs_scipy(n, dist, rng):
    import scipy.stats as ss

    if dist == "normal":
        x = rng.randn(n)
    elif dist == "uniform":
        x = rng.uniform(0.0, 1.0, n)
    elif dist == "exponential":
        x = rng.exponential(1.0, n)
    else:  # bimodal
        x = np.concatenate([rng.randn(n // 2) - 3.0, rng.randn(n - n // 2) + 3.0])
    x_arrow = pa.array(x, type=pa.float64())
    ours = py_shapiro_w(x_arrow)
    theirs = float(ss.shapiro(x).statistic)
    assert abs(ours - theirs) < 1e-6, f"W mismatch n={n} dist={dist}: ours={ours}, scipy={theirs}"


@pytest.mark.parametrize("n", [10, 100, 1000])
@pytest.mark.parametrize("tie_density", [0.0, 0.1, 0.5])
def test_kendall_parity_vs_scipy(n, tie_density, rng):
    import scipy.stats as ss

    x = rng.randn(n)
    y = rng.randn(n)
    if tie_density > 0:
        scale = max(1, int(1.0 / max(tie_density, 0.01)))
        x = np.round(x * scale) / scale
        y = np.round(y * scale) / scale
    result = kendall_tau_b(x.tolist(), y.tolist())
    ours = result["tau"]
    theirs = float(ss.kendalltau(x, y).statistic)
    assert abs(ours - theirs) < 1e-12, (
        f"tau mismatch n={n} ties={tie_density}: ours={ours}, scipy={theirs}"
    )
