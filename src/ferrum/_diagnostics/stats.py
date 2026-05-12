"""Vectorized NumPy in-house statistics for Phase 10.

Phase 10a provides ``studentized_residual`` for ``.predictions()``.
Phase 10g adds Pearson, Spearman, Shapiro-Wilk, Kendall (Rust-backed),
variance/covariance ranking, and the rank1d/rank2d compute helpers.

scipy is **not** imported at runtime. scipy is used only by
``tests/diagnostics/test_stats.py`` for parity validation.
"""

from __future__ import annotations

from math import log, sqrt

import numpy as np
import polars as pl


def studentized_residual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray | None = None,
) -> np.ndarray:
    """Compute studentized residuals.

    For linear estimators (X provided), uses the hat matrix diagonal:
        r_i / (sigma_hat * sqrt(1 - h_ii))
    where h = X (X' X)^{-1} X' and sigma_hat^2 = sum(r^2) / (n - p).

    For non-linear estimators (X=None), falls back to internally
    studentized residuals using the raw standard deviation of residuals.
    """
    r = y_true - y_pred
    if X is None:
        sigma = float(np.std(r, ddof=1)) if len(r) > 1 else 1.0
        return r / sigma if sigma > 0 else r * 0.0

    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    h_diag = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    h_diag = np.clip(h_diag, 0.0, 1.0 - 1e-12)
    sigma_sq = float((r * r).sum() / max(n - p, 1))
    sigma = float(np.sqrt(sigma_sq)) if sigma_sq > 0 else 0.0
    if sigma == 0.0:
        return r * 0.0
    return r / (sigma * np.sqrt(1.0 - h_diag))


def cooks_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Per-observation Cook's distance for a linear estimator.

    Formula
    -------
    D_i = (r_i² / (p · σ̂²)) · (h_ii / (1 − h_ii)²)

    where r_i = y_true − y_pred at observation i, p = column count of
    X (caller is responsible for adding an intercept column when the
    model fits one), σ̂² = SSE / (n − p), and h_ii is the i-th diagonal
    of the hat matrix H = X (Xᵀ X)⁻¹ Xᵀ. Returns a non-negative array
    of length n.
    """
    r = y_true - y_pred
    n, p = X.shape
    if n <= p:
        return np.zeros(n, dtype=np.float64)
    XtX_inv = np.linalg.pinv(X.T @ X)
    h_diag = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    # Clamp h_ii to (0, 1) — h_ii = 1 means the observation is its own
    # model and Cook's D is undefined.
    h_diag = np.clip(h_diag, 0.0, 1.0 - 1e-12)
    sse = float((r * r).sum())
    sigma_sq = sse / max(n - p, 1)
    if sigma_sq <= 0.0:
        return np.zeros(n, dtype=np.float64)
    one_minus_h_sq = (1.0 - h_diag) ** 2
    safe_denom = np.where(one_minus_h_sq > 0.0, one_minus_h_sq, 1e-24)
    return (r * r / (p * sigma_sq)) * (h_diag / safe_denom)


# ---------- Phase 10g: feature-ranking statistics ----------


def pearson_r(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlation between each X column and y."""
    Xm = X - X.mean(axis=0, keepdims=True)
    ym = y - y.mean()
    num = Xm.T @ ym
    denom = np.sqrt((Xm**2).sum(axis=0) * (ym**2).sum())
    return np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0)


def _rankdata_average(arr: np.ndarray) -> np.ndarray:
    """Scipy-compatible average-rank rankdata for 1D arrays."""
    arr = np.asarray(arr, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, arr.size + 1, dtype=np.float64)
    # Average ties: walk the sorted array, replace each tied run with its
    # mean rank.
    sorted_arr = arr[order]
    i = 0
    n = arr.size
    while i < n:
        j = i + 1
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        if j - i > 1:
            mean_rank = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            ranks[order[i:j]] = mean_rank
        i = j
    return ranks


def spearman_rho(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column Spearman rho = Pearson on average ranks (with ties)."""
    Xr = np.column_stack([_rankdata_average(X[:, i]) for i in range(X.shape[1])])
    yr = _rankdata_average(y)
    return pearson_r(Xr, yr)


def variance_rank(X: np.ndarray) -> np.ndarray:
    """Variance per feature (population variance, ddof=0).

    Used purely for ranking, so the absolute normalization is
    irrelevant — using ``var()`` keeps the output stable for n=1.
    """
    return X.var(axis=0)


def covariance_rank(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """abs(cov(X_col, y)) per feature (sample covariance, ddof=1)."""
    n = len(y)
    if n < 2:
        return np.abs(X * 0.0).sum(axis=0)
    Xm = X - X.mean(axis=0, keepdims=True)
    ym = y - y.mean()
    return np.abs((Xm.T @ ym) / (n - 1))


def _phi_inv(p: float) -> float:
    """Beasley-Springer-Moro inverse-normal CDF approximation.

    Used to seed Royston's Shapiro-Wilk normal-order-statistic
    coefficients ``m_i``. Sufficient precision (~1e-9) for the parity
    test tolerance against ``scipy.stats.shapiro``.
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = sqrt(-2.0 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = sqrt(-2.0 * log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def shapiro_w(x: np.ndarray) -> float:
    """Shapiro-Wilk W statistic via Royston 1992.

    Numerically stable for n in [3, 5000]. Returns W only; p-value is
    omitted (W alone is sufficient for ranking).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 3:
        raise ValueError(f"shapiro_w requires n >= 3; got n={n}")
    x_sorted = np.sort(x)

    # m_i = Phi^{-1}((i - 0.375) / (n + 0.25)), i = 1..n.
    i = np.arange(1, n + 1, dtype=np.float64)
    m = np.array([_phi_inv(((k - 0.375) / (n + 0.25))) for k in i])
    m_norm_sq = float((m * m).sum())
    u = 1.0 / sqrt(n)

    a_n = (
        -2.706056 * u**5
        + 4.434685 * u**4
        - 2.071190 * u**3
        - 0.147981 * u**2
        + 0.221157 * u
        + m[-1] / sqrt(m_norm_sq)
    )
    a_nm1 = (
        -3.582633 * u**5
        + 5.682633 * u**4
        - 1.752460 * u**3
        - 0.293762 * u**2
        + 0.042981 * u
        + m[-2] / sqrt(m_norm_sq)
    )

    eps = (m_norm_sq - 2.0 * m[-1] ** 2 - 2.0 * m[-2] ** 2) / (1.0 - 2.0 * a_n**2 - 2.0 * a_nm1**2)
    a = np.zeros(n, dtype=np.float64)
    a[-1] = a_n
    a[-2] = a_nm1
    a[0] = -a_n
    a[1] = -a_nm1
    if n > 4:
        a[2:-2] = m[2:-2] / sqrt(eps)

    numer = float((a * x_sorted).sum()) ** 2
    denom = float(((x_sorted - x_sorted.mean()) ** 2).sum())
    if denom <= 0.0:
        return 1.0
    return float(numer / denom)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b — wraps the Rust ``ferrum._core.kendall_tau_b``."""
    from ferrum._core import kendall_tau_b as _rust_ktb

    x64 = np.ascontiguousarray(x, dtype=np.float64).tolist()
    y64 = np.ascontiguousarray(y, dtype=np.float64).tolist()
    return float(_rust_ktb(x64, y64)["tau"])


# ---------- rank1d / rank2d compute helpers ----------


def _columns_and_array(X) -> tuple[list[str], np.ndarray]:
    """Coerce X (polars DataFrame / pandas / 2D numpy) to (cols, ndarray)."""
    if hasattr(X, "to_numpy") and hasattr(X, "columns"):
        cols = list(X.columns)
        X_np = np.asarray(X.to_numpy(), dtype=np.float64)
        return cols, X_np
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {arr.shape}")
    return [f"f{j}" for j in range(arr.shape[1])], arr


def rank1d_compute(
    X,
    *,
    algorithm: str = "shapiro",
    top_k: int | None = None,
) -> pl.DataFrame:
    """Univariate feature ranking. ``algorithm`` in {shapiro, variance}.

    For ``algorithm='covariance'``, the caller must use
    ``ModelSource.rank1d(algorithm='covariance')`` which has access to
    ``y``; ``rank1d_compute`` here raises.
    """
    cols, X_np = _columns_and_array(X)
    if algorithm == "shapiro":
        scores = np.array(
            [shapiro_w(X_np[:, j]) for j in range(X_np.shape[1])],
            dtype=np.float64,
        )
    elif algorithm == "variance":
        scores = variance_rank(X_np)
    elif algorithm == "covariance":
        raise ValueError(
            "rank1d_compute(algorithm='covariance') requires y; call via "
            "ModelSource.rank1d(algorithm='covariance')."
        )
    else:
        raise ValueError(
            f"rank1d algorithm must be one of 'shapiro', 'variance', "
            f"'covariance'; got {algorithm!r}"
        )
    order = np.argsort(-scores, kind="mergesort")
    df = pl.DataFrame(
        {
            "feature": [str(cols[i]) for i in order],
            "score": [float(scores[i]) for i in order],
            "rank": list(range(1, len(order) + 1)),
        }
    )
    if top_k is not None:
        df = df.head(int(top_k))
    return df


def rank2d_compute(X, *, algorithm: str = "pearson") -> pl.DataFrame:
    """Pairwise feature correlation matrix in long form.

    ``algorithm`` in {pearson, spearman, kendall, covariance}.
    """
    cols, X_np = _columns_and_array(X)
    p = X_np.shape[1]
    if algorithm == "pearson":
        if p == 1:
            C = np.array([[1.0]])
        else:
            C = np.corrcoef(X_np, rowvar=False)
            C = np.nan_to_num(C, nan=0.0)
            # np.corrcoef can return 1.0 ± O(eps) on the diagonal due to
            # accumulated rounding in the variance ratio; pin to exact 1.0
            # so the heatmap diagonal renders identically across platforms.
            np.fill_diagonal(C, 1.0)
    elif algorithm == "spearman":
        if p == 1:
            C = np.array([[1.0]])
        else:
            Xr = np.column_stack([_rankdata_average(X_np[:, j]) for j in range(p)])
            C = np.corrcoef(Xr, rowvar=False)
            C = np.nan_to_num(C, nan=0.0)
            np.fill_diagonal(C, 1.0)
    elif algorithm == "kendall":
        from ferrum._core import kendall_tau_b as _rust_ktb

        C = np.eye(p, dtype=np.float64)
        for i in range(p):
            for j in range(i + 1, p):
                tau = _rust_ktb(
                    X_np[:, i].tolist(),
                    X_np[:, j].tolist(),
                )["tau"]
                if not np.isfinite(tau):
                    tau = 0.0
                C[i, j] = C[j, i] = float(tau)
    elif algorithm == "covariance":
        if p == 1:
            C = np.array([[float(np.var(X_np[:, 0], ddof=1))]])
        else:
            C = np.cov(X_np, rowvar=False)
    else:
        raise ValueError(
            f"rank2d algorithm must be 'pearson', 'spearman', 'kendall', or "
            f"'covariance'; got {algorithm!r}"
        )
    feat_x: list[str] = []
    feat_y: list[str] = []
    corr: list[float] = []
    for i in range(p):
        for j in range(p):
            feat_x.append(str(cols[i]))
            feat_y.append(str(cols[j]))
            corr.append(float(C[i, j]))
    return pl.DataFrame(
        {
            "feature_x": feat_x,
            "feature_y": feat_y,
            "correlation": corr,
        }
    )
