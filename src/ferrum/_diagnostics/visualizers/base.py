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

    Parameters
    ----------
    model : Any, optional
        Fitted estimator that will be wrapped in a ``ModelSource`` at
        ``fit`` time. Pass ``None`` for no-model visualizers (rank /
        parallel coordinates).
    random_state : int, optional
        Seed forwarded to the underlying ``ModelSource``. Ignored when
        the wrapped derived-data compute is deterministic.
    theme : Theme, optional
        Per-chart theme override. Falls back to the global default
        when ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.ROCVisualizer(model, random_state=0).fit(X, y)
    >>> viz.show()                # returns a Chart
    >>> viz._metrics              # headline metric(s)
    """

    has_score: bool = False

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
        """Materialize derived data from ``X`` / ``y`` and build the chart.

        Constructs a ``ModelSource`` from the wrapped ``model``, calls
        ``_materialize`` to populate ``_metrics``, then ``_build_chart`` to
        assemble the ferrum ``Chart``. Returns ``self`` for method chaining.

        Parameters
        ----------
        X : array-like
            Feature matrix. Accepted types match ``ModelSource.__init__``.
        y : array-like, optional
            Target vector. Required by most classification / regression
            diagnostics; optional for unsupervised variants.

        Returns
        -------
        FerrumVisualizer
            ``self`` — the fitted visualizer instance.
        """
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
        """Compute a scalar model-quality score on ``(X, y)``.

        Subclasses override this to return an appropriate metric (e.g.
        ``roc_auc_score`` for ``ROCVisualizer``, ``r2_score`` for
        ``ResidualsVisualizer``). The base implementation always raises
        ``NotImplementedError`` — do not call it directly.

        Parameters
        ----------
        X : array-like
            Feature matrix, same type accepted by ``fit``.
        y : array-like
            True target values.

        Returns
        -------
        float
            A scalar quality metric (higher is conventionally better).

        Raises
        ------
        NotImplementedError
            When the concrete subclass has not overridden this method.
        """
        raise NotImplementedError(f"{type(self).__name__}.score() is not implemented")

    def show(self) -> Any:
        """Return the ferrum ``Chart`` for this visualizer.

        Must be called after ``fit``; raises ``RuntimeError`` otherwise.
        The returned ``Chart`` can be rendered in a notebook (``_repr_svg_``),
        saved with ``.save(path)``, or composed with other charts via ``+``
        / ``/``.

        Returns
        -------
        Chart
            The assembled diagnostic chart.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called yet.
        """
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fit before .show(); call .fit(X, y) first."
            )
        return self._chart

    def __repr__(self) -> str:
        """Return a concise string representation of the visualizer state.

        Returns ``ClassName(unfit)`` before ``fit`` is called and
        ``ClassName(metric=value, ...)`` afterwards, where the metrics are
        those recorded in ``_metrics`` during ``_materialize``.

        Returns
        -------
        str
            Human-readable one-liner summarising fit status and headline
            metrics.
        """
        if not self._fitted:
            return f"{type(self).__name__}(unfit)"
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in self._metrics.items())
        return f"{type(self).__name__}({metric_str})"
