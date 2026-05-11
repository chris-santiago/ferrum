"""Scipy-parity tests for `_diagnostics/stats.py`.

`scipy` is a dev/test dependency only — ferrum never imports it at
runtime. These tests validate that the in-house implementations match
scipy on a sweep of distributions / sample sizes / tie densities.
"""
from __future__ import annotations

import numpy as np
import pytest

from ferrum._diagnostics.stats import (
    kendall_tau_b,
    pearson_r,
    shapiro_w,
    spearman_rho,
)


@pytest.fixture
def rng():
    return np.random.RandomState(0)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_pearson_parity_vs_scipy(n, rng):
    import scipy.stats as ss

    X = rng.randn(n, 4)
    y = rng.randn(n)
    ours = pearson_r(X, y)
    theirs = np.array([ss.pearsonr(X[:, j], y).statistic for j in range(X.shape[1])])
    np.testing.assert_allclose(ours, theirs, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_spearman_parity_vs_scipy(n, rng):
    import scipy.stats as ss

    X = rng.randn(n, 4)
    y = rng.randn(n)
    ours = spearman_rho(X, y)
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
    ours = shapiro_w(x)
    theirs = float(ss.shapiro(x).statistic)
    # Royston 1992 with BSM inverse-normal seeding matches scipy's
    # swilk.f Fortran implementation to ~1e-8 on the standard
    # n in {10, 50, 200, 1000} × {normal, uniform, exponential, bimodal}
    # sweep. 1e-6 is a comfortable budget.
    assert abs(ours - theirs) < 1e-6, (
        f"W mismatch n={n} dist={dist}: ours={ours}, scipy={theirs}"
    )


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
    ours = kendall_tau_b(x, y)
    theirs = float(ss.kendalltau(x, y).statistic)
    assert abs(ours - theirs) < 1e-12, (
        f"tau mismatch n={n} ties={tie_density}: ours={ours}, scipy={theirs}"
    )
