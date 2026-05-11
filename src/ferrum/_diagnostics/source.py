"""ModelSource adapter — wraps a fitted estimator + data, exposes derived data.

Phase 10a: constructor, protocol detection, cache, .predictions(),
.probabilities(). Other methods land in 10b-10g; `ComparedModelSource`
in 10h.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from .deps import require_sklearn
from .stats import studentized_residual


_PROTOCOL_ATTRS: tuple[str, ...] = (
    "predict", "predict_proba", "decision_function", "transform",
    "fit_transform", "fit_predict", "score",
    "feature_importances_", "coef_", "explained_variance_ratio_",
    "cluster_centers_", "labels_", "classes_",
)


def _coerce_X_y(X: Any, y: Any) -> tuple[pl.DataFrame, pl.Series | None]:
    """Coerce X to polars.DataFrame and y to polars.Series (or None)."""
    if isinstance(X, pl.DataFrame):
        X_df = X
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        X_df = pl.from_numpy(X, schema=[f"f{i}" for i in range(X.shape[1])])
    else:
        # Route through ferrum's existing input-normalization to a pyarrow Table,
        # then convert to polars.
        from ferrum._coerce import to_arrow_table
        X_df = pl.from_arrow(to_arrow_table(X))
        if not isinstance(X_df, pl.DataFrame):
            raise TypeError(f"Could not coerce X to a polars DataFrame; got {type(X_df).__name__}")

    y_ser: pl.Series | None = None
    if y is not None:
        if isinstance(y, pl.Series):
            y_ser = y
        elif isinstance(y, np.ndarray):
            y_ser = pl.Series("y", y)
        elif isinstance(y, pl.DataFrame):
            if y.width != 1:
                raise ValueError(f"y DataFrame must have exactly 1 column; got {y.width}")
            y_ser = y.to_series()
        else:
            y_ser = pl.Series("y", list(y))
    return X_df, y_ser


class ModelSource:
    """Wraps a fitted estimator + dataset; exposes derived data as DataFrames.

    Constructor is sklearn-free: pure attribute introspection.
    Methods that need sklearn / shap / umap lazy-import on call.
    """

    def __init__(
        self,
        model: Any,
        X: Any,
        y: Any = None,
        *,
        feature_names: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        sample_weight: Any = None,
        random_state: int | None = None,
    ):
        self._model = model
        self._X, self._y = _coerce_X_y(X, y)
        self._feature_names: list[str] = (
            list(feature_names) if feature_names is not None
            else list(self._X.columns)
        )
        self._class_names: list[str] | None = (
            list(class_names) if class_names is not None else None
        )
        self._sample_weight = sample_weight
        self._random_state = random_state

        self._capabilities = frozenset(
            attr for attr in _PROTOCOL_ATTRS if hasattr(self._model, attr)
        )
        self._cache: dict[tuple, pl.DataFrame] = {}

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    def _require_capability(self, attr: str, method_name: str) -> None:
        if attr not in self._capabilities:
            raise AttributeError(
                f"ModelSource.{method_name}() requires the wrapped model to "
                f"implement '{attr}'. Got {type(self._model).__name__!r} which "
                f"does not."
            )

    def _cache_key(self, method: str, **kwargs) -> tuple:
        return (method, tuple(sorted(kwargs.items())))

    # --- 10a: predictions, probabilities ---------------------------------

    def predictions(self) -> pl.DataFrame:
        """Return y_true, y_pred, residual, studentized_residual."""
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

        # Studentized residual: linear-estimator path if model exposes coef_.
        if "coef_" in self._capabilities and self._y is not None:
            X_with_intercept = np.column_stack([np.ones(len(X_np)), X_np])
            stud = studentized_residual(y_true, y_pred, X_with_intercept)
        else:
            stud = studentized_residual(y_true, y_pred, X=None)

        df = pl.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": residual,
            "studentized_residual": stud,
        })
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
