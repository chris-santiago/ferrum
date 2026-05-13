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

from ferrum.plots.classification import (
    _class_balance_chart_from_dataframe,
    _class_prediction_error_chart_from_source,
    _discrimination_threshold_chart_from_source,
)
from .base import FerrumVisualizer


class DiscriminationThresholdVisualizer(FerrumVisualizer):
    """Sweep a decision threshold for a binary classifier and plot four metric curves.

    Evaluates ``precision``, ``recall``, ``f1``, and ``queue_rate`` (or a
    caller-supplied subset) at ``n_thresholds`` evenly-spaced probability
    thresholds between 0 and 1. After ``fit``, the F1-maximising threshold
    and its F1 score are available via ``_metrics``.

    Parameters
    ----------
    model : Any
        Fitted binary sklearn estimator that exposes ``predict_proba``.
    n_thresholds : int, default 50
        Number of probability thresholds to evaluate across ``[0, 1]``.
    metrics : tuple of str, default ("precision", "recall", "f1", "queue_rate")
        Which per-threshold metrics to include as curves in the chart.
        Passed through to the underlying ``ModelSource.discrimination_threshold``
        call and to ``mark_discrimination_threshold``.
    cv : Any, optional
        Cross-validation strategy forwarded to ``ModelSource.discrimination_threshold``.
        ``None`` uses the model as-is (no CV averaging).
    threshold_line : bool, default False
        When ``True``, overlays a vertical rule at the F1-maximising
        threshold (matches the figure-level
        ``discrimination_threshold_chart(threshold_line=True)``).
    random_state : int, optional
        Seed forwarded to ``ModelSource`` for any randomness in CV splits.
    theme : Theme, optional
        Per-chart theme override. Falls back to the global default when ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.DiscriminationThresholdVisualizer(model).fit(X, y)
    >>> viz.show()                   # returns the four-curve Chart
    >>> viz._metrics["best_threshold"], viz._metrics["best_f1"]
    """

    def __init__(
        self,
        model: Any,
        *,
        n_thresholds: int = 50,
        metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
        cv: Any = None,
        threshold_line: bool = False,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.n_thresholds = n_thresholds
        self.metrics = metrics
        self.cv = cv
        self.threshold_line = threshold_line

    def _materialize(self) -> None:
        dt = self._source.discrimination_threshold(
            n_thresholds=self.n_thresholds,
            cv=self.cv,
        )
        f1 = dt["f1"]
        idx = f1.arg_max() if f1.len() else 0
        self._metrics["best_threshold"] = float(dt["threshold"][idx])
        self._metrics["best_f1"] = float(dt["f1"][idx])

    def _build_chart(self) -> Any:
        return _discrimination_threshold_chart_from_source(
            self._source,
            n_thresholds=self.n_thresholds,
            metrics=self.metrics,
            cv=self.cv,
            threshold_line=self.threshold_line,
            theme=self.theme,
        )


class ClassPredictionErrorVisualizer(FerrumVisualizer):
    """Stacked-bar chart of actual-class composition per predicted class.

    For each predicted class label on the x-axis, the bar is stacked by
    the true class, showing how often a predicted class is correct vs.
    confused with another class. When ``normalize=True`` every bar is
    scaled to 100 % so proportions are comparable across imbalanced
    classes.

    After ``fit``, overall accuracy (total correct / total predictions,
    computed from raw counts regardless of ``normalize``) is stored in
    ``_metrics["accuracy"]``.

    Parameters
    ----------
    model : Any
        Fitted sklearn classifier that exposes ``predict``.
    normalize : bool, default False
        When ``True``, each predicted-class bar is scaled to sum to 1
        (100 % stacked view). When ``False``, bars show absolute counts.
    random_state : int, optional
        Seed forwarded to ``ModelSource`` for any randomness in data prep.
    theme : Theme, optional
        Per-chart theme override. Falls back to the global default when ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.ClassPredictionErrorVisualizer(model).fit(X, y)
    >>> viz.show()                   # returns the stacked-bar Chart
    >>> viz._metrics["accuracy"]     # proportion of correct predictions
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
        n_correct = float(cm.filter(pl.col("actual") == pl.col("predicted"))["value"].sum())
        n_total = float(cm["value"].sum())
        self._metrics["accuracy"] = n_correct / max(n_total, 1.0)

    def _build_chart(self) -> Any:
        return _class_prediction_error_chart_from_source(
            self._source,
            normalize=self.normalize,
            theme=self.theme,
        )


class ClassBalanceVisualizer(FerrumVisualizer):
    """Bar chart of per-class label counts, computed from target labels alone.

    Accepts both the standard sklearn ``fit(X, y)`` signature and the
    label-only shorthand ``fit(y)`` — when the second argument is omitted,
    the first positional argument is treated as the label array. No model
    is required; pass nothing for the ``model`` argument (it is always
    ``None`` internally).

    After ``fit``, ``_metrics`` contains:

    - ``n_classes`` — number of unique class labels.
    - ``imbalance_ratio`` — ``max_count / max(min_count, 1)``, where 1.0
      indicates perfectly balanced classes and larger values indicate
      increasing imbalance.

    Parameters
    ----------
    random_state : int, optional
        Accepted for API symmetry with model-backed visualizers but
        intentionally never consumed — ``ClassBalanceVisualizer`` overrides
        ``fit()`` and never constructs a ``ModelSource``. Documented as
        a permanent no-op so callers that script over visualizers don't
        have to special-case which ones accept the kwarg.
    theme : Theme, optional
        Per-chart theme override. Falls back to the global default when ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.ClassBalanceVisualizer().fit(X, y)
    >>> viz.show()                   # returns the per-class count bar Chart
    >>> viz._metrics["imbalance_ratio"]
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
                raise TypeError("ClassBalanceVisualizer.fit() needs either fit(X, y) or fit(y).")
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
            self._y,
            theme=self.theme,
        )
        self._fitted = True
        return self
