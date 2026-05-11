"""10b/10c extra classification visualizers.

Houses visualizers that don't naturally cluster with the curve-based ones
in ``classification.py``: threshold sweep, per-class error stack, and the
model-free class-balance bar.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import polars as pl

from ..charts import (
    _class_balance_chart_from_dataframe,
    _class_prediction_error_chart_from_source,
    _discrimination_threshold_chart_from_source,
)
from .base import FerrumVisualizer


class DiscriminationThresholdVisualizer(FerrumVisualizer):
    """Sweeps a probability threshold for a binary classifier; reports the
    F1-maximizing threshold + F1 as scalar metrics and renders the four
    per-threshold metric curves.
    """

    def __init__(
        self,
        model: Any,
        *,
        n_thresholds: int = 50,
        metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
        cv: Any = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.n_thresholds = n_thresholds
        self.metrics = metrics
        self.cv = cv

    def _materialize(self) -> None:
        dt = self._source.discrimination_threshold(
            n_thresholds=self.n_thresholds, cv=self.cv,
        )
        f1 = dt["f1"].to_numpy()
        idx = int(np.argmax(f1)) if f1.size else 0
        self._metrics["best_threshold"] = float(dt["threshold"][idx])
        self._metrics["best_f1"] = float(dt["f1"][idx])

    def _build_chart(self) -> Any:
        return _discrimination_threshold_chart_from_source(
            self._source,
            n_thresholds=self.n_thresholds,
            metrics=self.metrics,
            cv=self.cv,
            theme=self.theme,
        )


class ClassPredictionErrorVisualizer(FerrumVisualizer):
    """Per-predicted-class stacked-bar of actual-class composition.

    Reports overall ``accuracy`` (raw-count basis) as a scalar metric.
    The visual stack is rendered via ``mark_class_prediction_error``; the
    100% stack variant is available via ``normalize=True``.
    """

    def __init__(
        self,
        model: Any,
        *,
        normalize: bool = False,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.normalize = normalize

    def _materialize(self) -> None:
        cm = self._source.confusion_matrix(normalize=None)
        n_correct = float(
            cm.filter(pl.col("actual") == pl.col("predicted"))["value"].sum()
        )
        n_total = float(cm["value"].sum())
        self._metrics["accuracy"] = n_correct / max(n_total, 1.0)

    def _build_chart(self) -> Any:
        return _class_prediction_error_chart_from_source(
            self._source, normalize=self.normalize, theme=self.theme,
        )


class ClassBalanceVisualizer(FerrumVisualizer):
    """Per-class count bar chart computed from ``y`` alone (no model).

    Accepts the sklearn ``.fit(X, y)`` shape as well as the ``.fit(y)``
    shorthand — when ``y`` is omitted, the first positional argument is
    treated as the labels. Reports ``n_classes`` and ``imbalance_ratio``
    (``max_count / max(min_count, 1)``).
    """

    def __init__(
        self,
        *,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self._y: pl.Series | None = None

    def fit(self, X: Any, y: Any = None) -> "ClassBalanceVisualizer":
        if y is None:
            if X is None:
                raise TypeError(
                    "ClassBalanceVisualizer.fit() needs either fit(X, y) or fit(y)."
                )
            y = X

        if isinstance(y, pl.Series):
            self._y = y
        else:
            try:
                arr = np.asarray(y).ravel()
            except Exception:
                arr = list(y)
            self._y = pl.Series(arr.tolist() if isinstance(arr, np.ndarray) else arr)

        counts = Counter(self._y.to_list())
        max_n = max(counts.values()) if counts else 0
        min_n = min(counts.values()) if counts else 1
        self._metrics["n_classes"] = float(len(counts))
        self._metrics["imbalance_ratio"] = float(max_n) / float(max(min_n, 1))

        self._chart = _class_balance_chart_from_dataframe(
            self._y, theme=self.theme,
        )
        self._fitted = True
        return self

    def _materialize(self) -> None:  # pragma: no cover - bypassed by fit()
        raise NotImplementedError(
            "ClassBalanceVisualizer overrides fit() — _materialize unused."
        )

    def _build_chart(self) -> Any:  # pragma: no cover - bypassed by fit()
        raise NotImplementedError(
            "ClassBalanceVisualizer overrides fit() — _build_chart unused."
        )
