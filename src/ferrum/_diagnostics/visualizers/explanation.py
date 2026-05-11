"""10d explanation visualizers — feature importance, SHAP family, PDP."""
from __future__ import annotations

from typing import Any

import polars as pl

from ..charts import (
    _importance_chart_from_source,
    _shap_bar_chart_from_source,
    _shap_beeswarm_chart_from_source,
    _shap_waterfall_chart_from_source,
)
from .base import FerrumVisualizer


class FeatureImportancesVisualizer(FerrumVisualizer):
    """Sklearn-protocol visualizer for ``importance_chart``.

    ``method``: ``"builtin"`` reads the estimator's ``feature_importances_``
    or ``coef_`` (std=0); ``"permutation"`` runs sklearn
    ``permutation_importance`` with the supplied ``random_state``.
    ``top_k`` is informational at the materialization layer and threaded
    into the chart builder so the rendered chart shows only the most
    important features.
    """

    def __init__(
        self,
        model: Any,
        *,
        method: str = "builtin",
        top_k: int | None = 20,
        orient: str = "horizontal",
        error_bars: bool = True,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.method = method
        self.top_k = top_k
        self.orient = orient
        self.error_bars = error_bars

    def _materialize(self) -> None:
        df = self._source.importances(
            method=self.method, random_state=self.random_state,
        )
        if df.height:
            self._metrics["top_feature_importance"] = float(df["importance"][0])
        else:
            self._metrics["top_feature_importance"] = 0.0

    def _build_chart(self) -> Any:
        return _importance_chart_from_source(
            self._source,
            method=self.method,
            top_k=self.top_k,
            orient=self.orient,
            error_bars=self.error_bars,
            random_state=self.random_state,
            theme=self.theme,
        )


class SHAPVisualizer(FerrumVisualizer):
    """Sklearn-protocol visualizer for ``shap_chart``.

    ``kind`` selects the underlying chart: ``"beeswarm"`` (default,
    per-sample scatter), ``"bar"`` (mean-|shap| aggregated), or
    ``"waterfall"`` (single-sample cumulative — requires ``sample_idx``).
    The visualizer records ``top_abs_shap`` (max mean-|shap| across
    features) in ``_metrics`` for the repr.
    """

    def __init__(
        self,
        model: Any,
        *,
        kind: str = "beeswarm",
        max_display: int = 20,
        sample_idx: int | None = None,
        order: str = "abs_mean",
        background: Any = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.kind = kind
        self.max_display = max_display
        self.sample_idx = sample_idx
        self.order = order
        self.background = background

    def _materialize(self) -> None:
        sv = self._source.shap_values(background=self.background)
        agg = sv.group_by("feature").agg(
            pl.col("shap_value").abs().mean().alias("v")
        )
        if agg.height:
            self._metrics["top_abs_shap"] = float(agg["v"].max())
        else:
            self._metrics["top_abs_shap"] = 0.0

    def _build_chart(self) -> Any:
        if self.kind == "beeswarm":
            return _shap_beeswarm_chart_from_source(
                self._source,
                max_display=self.max_display,
                order=self.order,
                background=self.background,
                theme=self.theme,
            )
        if self.kind == "bar":
            return _shap_bar_chart_from_source(
                self._source,
                max_display=self.max_display,
                background=self.background,
                theme=self.theme,
            )
        if self.kind == "waterfall":
            if self.sample_idx is None:
                raise ValueError(
                    "SHAPVisualizer(kind='waterfall') requires sample_idx=<int>."
                )
            return _shap_waterfall_chart_from_source(
                self._source,
                sample_idx=self.sample_idx,
                max_display=self.max_display,
                background=self.background,
                theme=self.theme,
            )
        raise ValueError(
            f"SHAPVisualizer(kind={self.kind!r}) — expected "
            "'beeswarm', 'bar', or 'waterfall'."
        )
