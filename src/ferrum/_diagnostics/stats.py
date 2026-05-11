"""Vectorized NumPy in-house statistics for Phase 10.

Phase 10a provides `studentized_residual` for `.predictions()`.
Phase 10g adds Pearson, Spearman, Shapiro-Wilk, Kendall (Rust-backed),
variance/covariance ranking, and the rank1d/rank2d compute helpers.
"""
from __future__ import annotations

import numpy as np


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
