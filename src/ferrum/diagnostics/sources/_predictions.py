"""Phase 10a — predictions and probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa

from .._internal.deps import require_sklearn


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
        y_pred = np.asarray(self._model.predict(self._X), dtype=np.float64)
        y_true = (
            np.asarray(self._y, dtype=np.float64)
            if self._y is not None
            else np.full_like(y_pred, np.nan)
        )
        residual = y_true - y_pred

        if "coef_" in self._capabilities and self._y is not None:
            from ferrum._core import hat_matrix_stats

            x_arrow = self._x_record_batch()
            yt_arrow = pa.array(y_true, type=pa.float64())
            yp_arrow = pa.array(y_pred, type=pa.float64())
            result = hat_matrix_stats(x_arrow, yt_arrow, yp_arrow)
            result_pl = pl.from_arrow(result)
            stud = result_pl["studentized_residual"].to_numpy()
            cooks = result_pl["cooks_distance"].to_numpy()
            lev = result_pl["leverage"].to_numpy()
        else:
            from ferrum._core import studentized_residual_no_x

            yt_arrow = pa.array(y_true, type=pa.float64())
            yp_arrow = pa.array(y_pred, type=pa.float64())
            stud_arro3 = studentized_residual_no_x(yt_arrow, yp_arrow)
            stud = np.asarray(pa.array(stud_arro3), dtype=np.float64)
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

        if "predict_proba" in self._capabilities:
            proba = np.asarray(self._model.predict_proba(self._X), dtype=np.float64)
        elif "decision_function" in self._capabilities:
            scores = np.asarray(self._model.decision_function(self._X), dtype=np.float64)
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
            data["y_true"] = self._y
        for i, c in enumerate(classes):
            data[f"proba_{c}"] = proba[:, i]
        df = pl.DataFrame(data)
        self._cache[key] = df
        return df
