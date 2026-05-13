"""10b/10c classification visualizers — ROC, PR, Calibration, ConfusionMatrix, ClassificationReport."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..charts import (
    _calibration_chart_from_source,
    _classification_report_chart,
    _confusion_chart_from_source,
    _pr_chart_from_source,
    _roc_chart_from_source,
)
from .base import FerrumVisualizer


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

    has_score: bool = True

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

    def score(self, X: Any, y: Any) -> float:
        from sklearn.metrics import roc_auc_score

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.shape[1] == 2:
                return float(roc_auc_score(y, proba[:, 1]))
            return float(roc_auc_score(y, proba, multi_class="ovr"))
        return float(self.model.score(X, y))


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

    def fit(self, X: Any, y: Any = None) -> "CalibrationVisualizer":
        import ferrum
        from ferrum._diagnostics.source import ComparedModelSource

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
        from ..deps import require_sklearn

        require_sklearn("ClassificationReportVisualizer")
        from sklearn.metrics import f1_score

        y_true = self._source.y
        y_pred = self._source.model.predict(self._source.X)
        self._metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

    def _build_chart(self) -> Any:
        return _classification_report_chart(self._source, theme=self.theme)
