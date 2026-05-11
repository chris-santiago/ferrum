"""10b/10c extra classification visualizers — DiscriminationThresholdVisualizer."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..charts import _discrimination_threshold_chart_from_source
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
