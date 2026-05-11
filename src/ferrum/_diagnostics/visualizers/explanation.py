"""10d explanation visualizers — feature importance (SHAP/PDP land later in 10d)."""
from __future__ import annotations

from typing import Any

from ..charts import _importance_chart_from_source
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
