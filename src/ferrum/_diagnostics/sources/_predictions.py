"""Phase 10a — predictions and probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..deps import require_sklearn
from ..stats import cooks_distance, studentized_residual


class PredictionsMixin:
    """Phase 10a — predictions and probabilities."""

    # --- 10a: predictions, probabilities ---------------------------------

    def predictions(self) -> pl.DataFrame:
        """Return y_true, y_pred, residual, studentized_residual, cooks_distance, leverage.

        ``leverage`` is the diagonal of the hat matrix
        ``H = X (XᵀX)⁻¹ Xᵀ`` for linear estimators (those exposing
        ``coef_``); NaN otherwise. Used by the residuals-vs-leverage
        panel of multi-panel residuals charts.
        """
        key = self._cache_key("predictions")
        if key in self._cache:
            return self._cache[key]

        self._require_capability("predict", "predictions")
        X_np = self._X.to_numpy()
        y_pred = np.asarray(self._model.predict(X_np), dtype=np.float64)
        y_true = (
            np.asarray(self._y.to_numpy(), dtype=np.float64)
            if self._y is not None
            else np.full_like(y_pred, np.nan)
        )
        residual = y_true - y_pred

        # Studentized residual + Cook's distance + leverage: hat-matrix
        # quantities require the design matrix. Linear-estimator path
        # uses `coef_` as the gate; non-linear estimators fall back to
        # the no-X studentized residual and report NaN for Cook's D and
        # leverage (both undefined without a hat matrix).
        if "coef_" in self._capabilities and self._y is not None:
            X_with_intercept = np.column_stack([np.ones(len(X_np)), X_np])
            stud = studentized_residual(y_true, y_pred, X_with_intercept)
            cooks = cooks_distance(y_true, y_pred, X_with_intercept)
            # leverage h_ii = diag(X (XᵀX)⁻¹ Xᵀ). Recomputed here rather
            # than threaded out of studentized_residual/cooks_distance to
            # keep those helpers single-output; the redundant pinv() per
            # call is O(p³) at most and negligible for typical p.
            XtX_inv = np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
            lev = np.einsum(
                "ij,jk,ik->i",
                X_with_intercept,
                XtX_inv,
                X_with_intercept,
            )
            lev = np.clip(lev, 0.0, 1.0 - 1e-12)
        else:
            stud = studentized_residual(y_true, y_pred, X=None)
            cooks = np.full_like(y_pred, np.nan)
            lev = np.full_like(y_pred, np.nan)

        df = pl.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": residual,
                "studentized_residual": stud,
                "cooks_distance": cooks,
                "leverage": lev,
            }
        )
        self._cache[key] = df
        return df

    def probabilities(self) -> pl.DataFrame:
        """Return y_true + one column per class with predicted probability."""
        key = self._cache_key("probabilities")
        if key in self._cache:
            return self._cache[key]

        require_sklearn("probabilities")
        X_np = self._X.to_numpy()

        if "predict_proba" in self._capabilities:
            proba = np.asarray(self._model.predict_proba(X_np), dtype=np.float64)
        elif "decision_function" in self._capabilities:
            scores = np.asarray(self._model.decision_function(X_np), dtype=np.float64)
            if scores.ndim == 1:
                # Binary classifier — apply sigmoid.
                p1 = 1.0 / (1.0 + np.exp(-scores))
                proba = np.column_stack([1.0 - p1, p1])
            else:
                # Multiclass — softmax.
                exp = np.exp(scores - scores.max(axis=1, keepdims=True))
                proba = exp / exp.sum(axis=1, keepdims=True)
        else:
            raise AttributeError(
                "ModelSource.probabilities() requires the wrapped model to "
                "implement 'predict_proba' or 'decision_function'. Got "
                f"{type(self._model).__name__!r} which implements neither."
            )

        # Determine class labels.
        if self._class_names is not None:
            classes = list(self._class_names)
        elif hasattr(self._model, "classes_"):
            classes = [str(c) for c in self._model.classes_]
        else:
            classes = [f"class_{i}" for i in range(proba.shape[1])]

        data: dict[str, Any] = {}
        if self._y is not None:
            data["y_true"] = self._y.to_numpy()
        for i, c in enumerate(classes):
            data[f"proba_{c}"] = proba[:, i]
        df = pl.DataFrame(data)
        self._cache[key] = df
        return df
