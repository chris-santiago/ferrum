"""10b/10c classification visualizers.

Curve-based: ROC, PR, Calibration, ConfusionMatrix, ClassificationReport.
Plus the threshold sweep, per-class error stack, and model-free class-balance
bar (folded in from the former ``classification_extra`` module).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import polars as pl

from ferrum.plots.classification import (
    _calibration_chart_from_source,
    _class_balance_chart_from_dataframe,
    _class_prediction_error_chart_from_source,
    _classification_report_chart,
    _confusion_chart_from_source,
    _discrimination_threshold_chart_from_source,
    _pr_chart_from_source,
    _roc_chart_from_source,
)
from ._base import FerrumVisualizer


class ROCVisualizer(FerrumVisualizer):
    """ROC curve(s) for binary or multiclass classifiers.

    Wraps ``ModelSource.roc_curve()``. ``per_class=True`` (default)
    draws one curve per class; pass ``per_class=False`` to plot a
    single averaged curve. Records ``auc_mean`` as the headline metric.

    Parameters
    ----------
    model : Any
        Fitted classifier (must implement ``predict_proba`` or
        ``decision_function``).
    micro : bool, default True
        Compute the micro-averaged AUC.
    macro : bool, default True
        Compute the macro-averaged AUC. Takes precedence over ``micro``
        when choosing which averaged curve to plot at
        ``per_class=False``.
    per_class : bool, default True
        Render one curve per class. When False, only the averaged
        curve is rendered.
    random_state : int, optional
    theme : Theme, optional

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.ROCVisualizer(clf).fit(X, y)
    >>> viz._metrics["auc_mean"]
    0.92
    """

    def __init__(
        self,
        model: Any,
        *,
        micro: bool = True,
        macro: bool = True,
        per_class: bool = True,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.micro = micro
        self.macro = macro
        self.per_class = per_class

    def _materialize(self) -> None:
        avg = "macro" if self.macro else ("micro" if self.micro else None)
        roc = self._source.roc_curve(average=avg)
        aucs = roc["auc"].drop_nulls().unique().to_list()
        self._metrics["auc_mean"] = float(np.nanmean(aucs)) if aucs else float("nan")

    def _build_chart(self) -> Any:
        avg = "macro" if self.macro else ("micro" if self.micro else None)
        return _roc_chart_from_source(
            self._source,
            per_class=self.per_class,
            average=avg,
            theme=self.theme,
        )

    @property
    def has_score(self) -> bool:
        """True iff :meth:`score` returns a real AUC rather than the fallback.

        ROC scores from class probabilities, so unlike the base class it is
        scoreable when the model exposes ``predict_proba`` even with no
        ``.score`` method. Mirrors the exact branch taken in :meth:`score`,
        so the two can never drift.
        """
        return hasattr(self.model, "predict_proba") or callable(getattr(self.model, "score", None))

    def score(self, X: Any, y: Any) -> float:
        from sklearn.metrics import roc_auc_score

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.shape[1] == 2:
                return float(roc_auc_score(y, proba[:, 1]))
            return float(roc_auc_score(y, proba, multi_class="ovr"))
        if callable(getattr(self.model, "score", None)):
            return float(self.model.score(X, y))
        return 0.0


class PRVisualizer(FerrumVisualizer):
    """Precision-recall curve(s) for binary or multiclass classifiers.

    Wraps ``ModelSource.pr_curve()``. Records ``ap_mean`` (average
    precision averaged across classes) as the headline metric.

    Parameters
    ----------
    model : Any
        Fitted classifier exposing ``predict_proba`` or
        ``decision_function``.
    random_state : int, optional
    theme : Theme, optional

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.PRVisualizer(clf).fit(X, y)
    >>> viz._metrics["ap_mean"]
    0.88
    """

    def __init__(
        self,
        model: Any,
        *,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)

    def _materialize(self) -> None:
        pr = self._source.pr_curve()
        aps = pr["ap"].drop_nulls().unique().to_list()
        self._metrics["ap_mean"] = float(np.nanmean(aps)) if aps else float("nan")

    def _build_chart(self) -> Any:
        return _pr_chart_from_source(self._source, theme=self.theme)


class CalibrationVisualizer(FerrumVisualizer):
    """Calibration (reliability) diagram for a probability classifier.

    Wraps ``ModelSource.calibration_curve()``. Records the mean squared
    deviation between ``mean_predicted`` and ``fraction_positive`` as
    ``calibration_error``.

    Parameters
    ----------
    *models : Any
        One or more fitted classifiers. Pass a single model for a
        single-curve diagram; pass two or more (or a single dict
        positional argument like ``{"a": m_a, "b": m_b}``) to overlay
        multiple curves via ``ComparedModelSource``.
    n_bins : int, default 10
        Number of bins for the calibration histogram.
    strategy : {"uniform", "quantile"}, default "uniform"
        Bin-edge strategy (matches ``sklearn.calibration``).
    random_state : int, optional
    theme : Theme, optional

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.CalibrationVisualizer(clf, n_bins=5).fit(X, y)
    >>> viz_overlay = fm.CalibrationVisualizer(clf_a, clf_b).fit(X, y)
    """

    def __init__(
        self,
        *models: Any,
        n_bins: int = 10,
        strategy: str = "uniform",
        random_state: int | None = None,
        theme: Any = None,
    ):
        if len(models) == 0:
            raise TypeError("CalibrationVisualizer requires at least one model")
        # Stash the raw inputs; fit() resolves them into a single
        # ModelSource or ComparedModelSource (the parent class's fit()
        # always builds a single-model ModelSource, so we override fit
        # for the multi-model path).
        super().__init__(models[0], random_state=random_state, theme=theme)
        self._models = models
        self.n_bins = n_bins
        self.strategy = strategy

    @property
    def _is_multi_model(self) -> bool:
        """Whether this overlays multiple models (no single well-defined score).

        True for two or more positional models, or for a single dict-of-models
        positional argument. False for a single fitted model, where the
        inherited single-model :meth:`score` / :meth:`has_score` apply.
        """
        return len(self._models) > 1 or isinstance(self._models[0], dict)

    @property
    def has_score(self) -> bool:
        """True iff :meth:`score` returns a real metric rather than the fallback.

        A multi-model overlay has no single ``.score`` to report, so
        ``has_score`` is ``False`` and :meth:`score` returns ``0.0`` — the two
        stay mutually consistent. A single-model calibration keeps the base
        ``model.score`` behavior unchanged.
        """
        if self._is_multi_model:
            return False
        return super().has_score

    def score(self, X: Any, y: Any) -> float:
        """Single model's ``.score``; ``0.0`` for a multi-model overlay.

        Overlaying several models has no single well-defined score, so rather
        than silently returning only the first model's metric this returns the
        documented ``0.0`` no-single-score fallback (consistent with
        :attr:`has_score` being ``False``).
        """
        if self._is_multi_model:
            return 0.0
        return super().score(X, y)

    def fit(self, X: Any, y: Any = None) -> "CalibrationVisualizer":
        import ferrum
        from ferrum.diagnostics.source import ComparedModelSource

        if len(self._models) == 1:
            m = self._models[0]
            if isinstance(m, dict):
                # Dict-of-models form.
                self._source = ferrum.ModelSource.compare(
                    m,
                    X,
                    y,
                    random_state=self.random_state,
                )
            else:
                self._source = ferrum.ModelSource(
                    m,
                    X,
                    y,
                    random_state=self.random_state,
                )
        else:
            # Positional N-model overlay.
            if all(isinstance(m, ferrum.ModelSource) for m in self._models):
                self._source = ComparedModelSource(
                    {f"model_{i}": m for i, m in enumerate(self._models)},
                )
            else:
                self._source = ferrum.ModelSource.compare(
                    {f"model_{i}": m for i, m in enumerate(self._models)},
                    X,
                    y,
                    random_state=self.random_state,
                )
        self._materialize()
        self._chart = self._build_chart()
        self._fitted = True
        return self

    def _materialize(self) -> None:
        cal = self._source.calibration_curve(
            n_bins=self.n_bins,
            strategy=self.strategy,
        )
        # When the source is a ComparedModelSource the frame carries one
        # row per (model, bin). Per-model calibration_error then
        # surfaces as a dict in the repr via the dispatched mean.
        diff = cal["fraction_positive"] - cal["mean_predicted"]
        self._metrics["calibration_error"] = float((diff**2).mean())

    def _build_chart(self) -> Any:
        return _calibration_chart_from_source(
            self._source,
            n_bins=self.n_bins,
            strategy=self.strategy,
            theme=self.theme,
        )


class ConfusionMatrixVisualizer(FerrumVisualizer):
    """Confusion-matrix heatmap with per-cell counts or normalized fractions.

    Wraps ``ModelSource.confusion_matrix()``. Renders a rect-mark heatmap
    with cell-value text overlaid. Records ``accuracy`` (diagonal sum / total,
    always computed from raw counts regardless of the ``normalize`` setting).

    Parameters
    ----------
    model : Any
        Fitted classifier implementing ``predict``.
    normalize : {"true", "pred", "all"} or None, default "true"
        Row-normalization strategy passed to the chart builder.
        ``"true"`` normalizes each row by the true-class total (recall
        fractions); ``"pred"`` by the predicted-class total (precision
        fractions); ``"all"`` by the grand total; ``None`` shows raw counts.
    random_state : int, optional
        Seed forwarded to ``ModelSource`` for reproducible train/test splits.
    theme : Theme, optional
        Ferrum theme applied to the output chart.

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.ConfusionMatrixVisualizer(clf).fit(X, y)
    >>> viz._metrics["accuracy"]
    0.94
    >>> viz.show()
    """

    def __init__(
        self,
        model: Any,
        *,
        normalize: str | None = "true",
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.normalize = normalize

    def _materialize(self) -> None:
        # accuracy is a per-sample headline metric and must be computed
        # from raw integer counts, never from a row/col/all-normalized
        # frame — so this call is intentionally hardcoded to
        # normalize=None, independent of self.normalize (which controls
        # only the rendered heatmap's cell values). The docstring
        # documents this; the comment lives here so a future reader of
        # the code doesn't "fix" it by passing self.normalize through.
        cm = self._source.confusion_matrix(normalize=None)
        n_correct = float(cm.filter(pl.col("actual") == pl.col("predicted"))["value"].sum())
        n_total = float(cm["value"].sum())
        self._metrics["accuracy"] = n_correct / max(n_total, 1.0)

    def _build_chart(self) -> Any:
        return _confusion_chart_from_source(
            self._source,
            normalize=self.normalize,
            theme=self.theme,
        )


class ClassificationReportVisualizer(FerrumVisualizer):
    """Per-class precision, recall, and F1-score heatmap with text overlay.

    Wraps ``_classification_report_chart()``. Produces a rect-mark heatmap
    with one row per class and columns for precision, recall, and F1.
    Records ``f1_macro`` (macro-averaged F1 across all classes, computed via
    ``sklearn.metrics.f1_score``) as the headline scalar metric.

    Parameters
    ----------
    model : Any
        Fitted classifier implementing ``predict``.
    random_state : int, optional
        Seed forwarded to ``ModelSource`` for reproducible train/test splits.
    theme : Theme, optional
        Ferrum theme applied to the output chart.

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.ClassificationReportVisualizer(clf).fit(X, y)
    >>> viz._metrics["f1_macro"]
    0.91
    >>> viz.show()
    """

    def __init__(
        self,
        model: Any,
        *,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)

    def _materialize(self) -> None:
        from .._internal.deps import require_sklearn

        require_sklearn("ClassificationReportVisualizer")
        from sklearn.metrics import f1_score

        y_true = self._source.y
        y_pred = self._source.model.predict(self._source.X)
        self._metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

    def _build_chart(self) -> Any:
        return _classification_report_chart(self._source, theme=self.theme)


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
