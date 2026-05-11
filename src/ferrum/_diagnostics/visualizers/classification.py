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

    ``per_class=True`` (default) draws one curve per class. ``micro`` /
    ``macro`` toggle which averaged curve is reported by
    ``_metrics["auc_mean"]`` and overlaid when ``per_class=False``.
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

    def score(self, X: Any, y: Any) -> float:
        from sklearn.metrics import roc_auc_score
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.shape[1] == 2:
                return float(roc_auc_score(y, proba[:, 1]))
            return float(roc_auc_score(y, proba, multi_class="ovr"))
        return float(self.model.score(X, y))


class PRVisualizer(FerrumVisualizer):
    """Precision-recall curve(s)."""

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
    """Calibration (reliability) diagram. Variadic in models for Phase 10h;
    10b accepts a single model only.
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
        if len(models) != 1:
            raise NotImplementedError(
                "Multi-model CalibrationVisualizer ships in Phase 10h."
            )
        super().__init__(models[0], random_state=random_state, theme=theme)
        self.n_bins = n_bins
        self.strategy = strategy

    def _materialize(self) -> None:
        cal = self._source.calibration_curve(
            n_bins=self.n_bins, strategy=self.strategy,
        )
        diff = (
            cal["fraction_positive"].to_numpy()
            - cal["mean_predicted"].to_numpy()
        )
        self._metrics["calibration_error"] = float(np.mean(diff ** 2))

    def _build_chart(self) -> Any:
        return _calibration_chart_from_source(
            self._source,
            n_bins=self.n_bins,
            strategy=self.strategy,
            theme=self.theme,
        )


class ConfusionMatrixVisualizer(FerrumVisualizer):
    """Confusion-matrix heatmap with per-cell counts (or normalized fractions).

    ``accuracy`` is reported as a scalar metric (computed on raw counts
    regardless of the ``normalize`` setting used for rendering).
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
        cm = self._source.confusion_matrix(normalize=None)
        n_correct = float(
            cm.filter(pl.col("actual") == pl.col("predicted"))["value"].sum()
        )
        n_total = float(cm["value"].sum())
        self._metrics["accuracy"] = n_correct / max(n_total, 1.0)

    def _build_chart(self) -> Any:
        return _confusion_chart_from_source(
            self._source, normalize=self.normalize, theme=self.theme,
        )


class ClassificationReportVisualizer(FerrumVisualizer):
    """Per-class precision/recall/F1 heatmap (rect + text overlay).

    Reports ``f1_macro`` as a scalar metric.
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

        y_true = self._source._y.to_numpy()
        y_pred = self._source._model.predict(self._source._X.to_numpy())
        self._metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

    def _build_chart(self) -> Any:
        return _classification_report_chart(self._source, theme=self.theme)
