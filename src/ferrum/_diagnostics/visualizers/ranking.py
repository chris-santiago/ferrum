"""Phase 10g — feature-ranking visualizers (no-model variants).

Mirrors yellowbrick's ``Rank1D`` / ``Rank2D`` / ``ParallelCoordinates``
surfaces. None of these need a fitted estimator — they operate on the
raw feature matrix X (with optional y for ``Rank1D(algorithm="covariance")``)
and skip the ``ModelSource`` round-trip. The chart is materialized
directly from the in-house compute helpers in ``stats.py``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .base import FerrumVisualizer
from ..charts import (
    _parallel_coords_chart_from_dataframe,
    _rank1d_chart_from_dataframe,
    _rank2d_chart_from_dataframe,
)
from ..stats import (
    covariance_rank,
    rank1d_compute,
    rank2d_compute,
)


class Rank1DVisualizer(FerrumVisualizer):
    """Univariate feature ranking — see ferrum-spec.md §3.15.

    ``algorithm`` in ``{"shapiro", "variance", "covariance"}``. The
    ``"covariance"`` variant requires ``y`` to be supplied at ``fit()``
    time. Records ``top_feature_score`` (the score of the highest-ranked
    feature) as the headline metric.
    """

    def __init__(
        self,
        *,
        algorithm: str = "shapiro",
        orient: str = "horizontal",
        top_k: int | None = None,
        color_field: str | None = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.algorithm = algorithm
        self.orient = orient
        self.top_k = top_k
        self.color_field = color_field

    def fit(self, X: Any, y: Any = None) -> "Rank1DVisualizer":
        if self.algorithm == "covariance":
            if y is None:
                raise ValueError(
                    "Rank1DVisualizer(algorithm='covariance') requires y."
                )
            cols, X_np = _columns_and_array(X)
            y_np = np.asarray(
                y.to_numpy() if hasattr(y, "to_numpy") else y,
                dtype=np.float64,
            )
            scores = covariance_rank(X_np, y_np)
            order = np.argsort(-scores, kind="mergesort")
            df = pl.DataFrame({
                "feature": [str(cols[int(i)]) for i in order],
                "score": [float(scores[int(i)]) for i in order],
                "rank": list(range(1, len(order) + 1)),
            })
        else:
            df = rank1d_compute(X, algorithm=self.algorithm)
        self._metrics["top_feature_score"] = float(df["score"][0])
        self._chart = _rank1d_chart_from_dataframe(
            df,
            algorithm=self.algorithm,
            orient=self.orient,
            top_k=self.top_k,
            color_field=self.color_field,
            theme=self.theme,
        )
        self._fitted = True
        return self


class Rank2DVisualizer(FerrumVisualizer):
    """Pairwise feature ranking — see ferrum-spec.md §3.15.

    ``algorithm`` in ``{"pearson", "spearman", "kendall", "covariance"}``.
    Records ``max_abs_corr`` (the largest absolute off-diagonal
    correlation) as the headline metric — useful for detecting
    multicollinearity at a glance.
    """

    def __init__(
        self,
        *,
        algorithm: str = "pearson",
        annot: bool = True,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.algorithm = algorithm
        self.annot = annot

    def fit(self, X: Any, y: Any = None) -> "Rank2DVisualizer":
        del y
        df = rank2d_compute(X, algorithm=self.algorithm)
        off_diag = df.filter(pl.col("feature_x") != pl.col("feature_y"))
        if off_diag.height > 0:
            self._metrics["max_abs_corr"] = float(
                off_diag["correlation"].abs().max() or 0.0
            )
        else:
            self._metrics["max_abs_corr"] = 0.0
        self._chart = _rank2d_chart_from_dataframe(
            df,
            algorithm=self.algorithm,
            annot=self.annot,
            theme=self.theme,
        )
        self._fitted = True
        return self


class ParallelCoordinatesVisualizer(FerrumVisualizer):
    """Parallel coordinates — see ferrum-spec.md §3.15.

    ``hue`` is the column name to color samples by; ``features`` selects
    a subset of feature columns. Records ``n_samples`` and ``n_features``
    so the repr surfaces the chart's shape.
    """

    def __init__(
        self,
        *,
        features: list[str] | None = None,
        hue: str | None = None,
        rescale: str | None = "minmax",
        alpha: float = 0.5,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.features = features
        self.hue = hue
        self.rescale = rescale
        self.alpha = alpha

    def fit(self, X: Any, y: Any = None) -> "ParallelCoordinatesVisualizer":
        del y
        # n_samples / n_features bookkeeping for the repr.
        if isinstance(X, pl.DataFrame):
            n_samples = X.height
            n_features = len(self.features) if self.features else (
                X.width - (1 if self.hue in X.columns else 0)
            )
        elif hasattr(X, "shape"):
            n_samples = int(X.shape[0])
            n_features = (
                len(self.features) if self.features else int(X.shape[1])
            )
        else:
            arr = np.asarray(X)
            n_samples = int(arr.shape[0])
            n_features = (
                len(self.features) if self.features else int(arr.shape[1])
            )
        self._metrics["n_samples"] = float(n_samples)
        self._metrics["n_features"] = float(n_features)
        self._chart = _parallel_coords_chart_from_dataframe(
            X,
            features=self.features,
            hue=self.hue,
            rescale=self.rescale,
            alpha=self.alpha,
            theme=self.theme,
        )
        self._fitted = True
        return self


def _columns_and_array(X: Any) -> tuple[list[str], np.ndarray]:
    """Coerce X (polars / pandas DataFrame / 2D numpy) to (cols, ndarray)."""
    if hasattr(X, "to_numpy") and hasattr(X, "columns"):
        return list(X.columns), np.asarray(X.to_numpy(), dtype=np.float64)
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {arr.shape}")
    return [f"f{j}" for j in range(arr.shape[1])], arr
