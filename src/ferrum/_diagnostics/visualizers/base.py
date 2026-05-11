"""FerrumVisualizer base — see ferrum-spec.md §3.15.

fit() materializes derived data on a ModelSource and builds the chart.
show() returns the Chart for rendering.
"""
from __future__ import annotations

from typing import Any


class FerrumVisualizer:
    """Base class for sklearn-protocol model-diagnostic visualizers.

    Concrete visualizers override ``_materialize`` (compute and record
    metrics on a ``ModelSource``) and ``_build_chart`` (assemble the
    Chart). The base ``fit`` constructs a ``ModelSource`` from the
    supplied ``model``, ``X``, ``y`` and dispatches both hooks; pass
    ``model=None`` for no-model variants like ``Rank1DVisualizer`` /
    ``ParallelCoordinatesVisualizer`` and override ``fit`` directly.
    """

    def __init__(
        self,
        model: Any = None,
        *,
        random_state: int | None = None,
        theme: Any = None,
        **_extra: Any,
    ):
        self.model = model
        self.random_state = random_state
        self.theme = theme
        self._fitted = False
        self._source: Any = None
        self._chart: Any = None
        self._metrics: dict[str, float] = {}

    def fit(self, X: Any, y: Any = None) -> "FerrumVisualizer":
        import ferrum
        self._source = ferrum.ModelSource(
            self.model, X, y, random_state=self.random_state
        )
        self._materialize()
        self._chart = self._build_chart()
        self._fitted = True
        return self

    def _materialize(self) -> None:
        raise NotImplementedError

    def _build_chart(self) -> Any:
        raise NotImplementedError

    def score(self, X: Any, y: Any) -> float:
        raise NotImplementedError(f"{type(self).__name__}.score() is not implemented")

    def show(self) -> Any:
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fit before .show(); call .fit(X, y) first."
            )
        return self._chart

    def __repr__(self) -> str:
        if not self._fitted:
            return f"{type(self).__name__}(unfit)"
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in self._metrics.items())
        return f"{type(self).__name__}({metric_str})"
