"""10a regression visualizers — ResidualsVisualizer, PredictionErrorVisualizer,
CooksDistanceVisualizer.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..charts import (
    _prediction_error_chart_from_source,
    _residuals_chart_from_source,
)
from .base import FerrumVisualizer


class ResidualsVisualizer(FerrumVisualizer):
    def __init__(
        self,
        model: Any,
        *,
        kind: str = "studentized",
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.kind = kind

    def _materialize(self) -> None:
        df = self._source.predictions()
        resid = df["residual"].to_numpy()
        self._metrics["rmse"] = float(np.sqrt((resid ** 2).mean()))
        self._metrics["mae"] = float(np.abs(resid).mean())

    def _build_chart(self) -> Any:
        return _residuals_chart_from_source(
            self._source, kind=self.kind, theme=self.theme,
        )

    def score(self, X: Any, y: Any) -> float:
        return float(self.model.score(X, y))


class PredictionErrorVisualizer(FerrumVisualizer):
    def __init__(
        self,
        model: Any,
        *,
        identity_line: bool = True,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.identity_line = identity_line

    def _materialize(self) -> None:
        df = self._source.predictions()
        resid = df["residual"].to_numpy()
        self._metrics["rmse"] = float(np.sqrt((resid ** 2).mean()))

    def _build_chart(self) -> Any:
        return _prediction_error_chart_from_source(
            self._source, identity_line=self.identity_line, theme=self.theme,
        )

    def score(self, X: Any, y: Any) -> float:
        return float(self.model.score(X, y))


class CooksDistanceVisualizer(FerrumVisualizer):
    """Cook's distance via studentized residuals. The leverage-aware variant
    (true Cook's D = stud^2 * h_ii / (p * (1 - h_ii))) lands in 10h alongside
    the multi-panel residuals_chart; the 10a build surfaces max-|studentized|
    as a proxy metric.
    """

    def __init__(
        self,
        model: Any,
        *,
        threshold: float | None = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.threshold = threshold

    def _materialize(self) -> None:
        df = self._source.predictions()
        stud = df["studentized_residual"].to_numpy()
        self._metrics["max_studentized"] = float(np.max(np.abs(stud)))

    def _build_chart(self) -> Any:
        return _residuals_chart_from_source(
            self._source,
            kind="studentized",
            panels=["residuals_vs_leverage"],
            theme=self.theme,
        )
