"""FerrumVisualizer base — see ferrum-spec.md §3.15.

fit() materializes derived data on a ModelSource and builds the chart.
show() returns the Chart for rendering.

Two override patterns:

1. **Model-backed visualizers (the common case)** — override
   ``_materialize`` and ``_build_chart``; the inherited ``fit``
   constructs a ``ModelSource`` and calls both hooks in order.
2. **No-model / non-standard visualizers** — override ``fit`` directly
   and skip ``ModelSource`` construction.  ``ClassBalanceVisualizer``,
   ``Rank1DVisualizer`` / ``Rank2DVisualizer`` /
   ``ParallelCoordinatesVisualizer``, ``CalibrationVisualizer``, and
   ``ElbowVisualizer`` all take this path.

The two hooks default to no-ops on the base class so option-2
subclasses don't need to provide trivial stubs.  Calling ``show()``
on a fitted-but-unbuilt visualizer simply returns ``None`` instead
of raising; that's a programming error in the subclass, not a
runtime condition to guard against.
"""

from __future__ import annotations

from typing import Any


class FerrumVisualizer:
    """Base class for sklearn-protocol model-diagnostic visualizers.

    Concrete visualizers either override ``_materialize`` +
    ``_build_chart`` (the standard model-backed flow) or override
    ``fit`` directly (no-model / multi-fit / label-only flow).

    Parameters
    ----------
    model : Any, optional
        Fitted estimator that will be wrapped in a ``ModelSource`` at
        ``fit`` time. Pass ``None`` for no-model visualizers (rank /
        parallel coordinates / class balance).
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

    @property
    def has_score(self) -> bool:
        """Whether [score][ferrum.FerrumVisualizer.score] returns a real test-set metric.

        Derived from behavior rather than hand-maintained: ``True`` exactly
        when the wrapped model exposes a ``.score`` method (so the inherited
        [score][ferrum.FerrumVisualizer.score] delegates to it), ``False``
        for genuinely no-model visualizers (rank / parallel-coordinates /
        class-balance / elbow) whose [score][ferrum.FerrumVisualizer.score]
        returns the ``0.0`` fallback. Mirrors the guard in
        [score][ferrum.FerrumVisualizer.score], so the two can never drift.

        Subclasses that override [score][ferrum.FerrumVisualizer.score] with
        a different scoreability condition (e.g.
        [ROCVisualizer][ferrum.ROCVisualizer], which scores from
        ``predict_proba``) or that wrap multiple models with no single
        metric (e.g. a multi-model overlay) override this property to keep
        the two in lockstep — ``has_score`` is ``True`` for an instance iff
        its [score][ferrum.FerrumVisualizer.score] returns a real metric
        rather than the ``0.0`` fallback.
        """
        return callable(getattr(self.model, "score", None))

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

        self._source = ferrum.ModelSource(self.model, X, y, random_state=self.random_state)
        self._materialize()
        self._chart = self._build_chart()
        self._fitted = True
        return self

    def _materialize(self) -> None:
        """Compute derived data and populate ``_metrics``.

        Default implementation is a no-op so subclasses that override
        ``fit`` directly (no-model variants) don't have to provide a
        trivial stub.  Standard model-backed subclasses override this
        to read off ``self._source`` and write headline metrics into
        ``self._metrics``.
        """

    def _build_chart(self) -> Any:
        """Assemble and return the ferrum ``Chart``.

        Default implementation returns ``None`` so subclasses that
        override ``fit`` directly (and set ``self._chart`` themselves)
        don't have to provide a trivial stub.  Standard model-backed
        subclasses override this to construct the chart from
        ``self._source``.
        """
        return None

    def score(self, X: Any, y: Any) -> float:
        """Delegate to ``self.model.score(X, y)`` when the model supports it.

        For model-backed visualizers (the common case) this returns the
        wrapped estimator's own ``.score`` metric — ``r2_score`` for a
        regressor, accuracy for a classifier, etc. Genuinely no-model
        visualizers (rank / parallel-coordinates / class-balance / elbow,
        all constructed with ``model=None``) fall through to the ``0.0``
        fallback so they satisfy the sklearn visualizer protocol without
        raising.

        Subclasses whose score is *not* the estimator's own ``.score``
        (e.g. ``ROCVisualizer`` returns ``roc_auc_score``) override this.

        Parameters
        ----------
        X : array-like
            Feature matrix, same type accepted by ``fit``.
        y : array-like
            True target values.

        Returns
        -------
        float
            ``self.model.score(X, y)`` when the wrapped model exposes a
            ``.score`` method; ``0.0`` otherwise.
        """
        if callable(getattr(self.model, "score", None)):
            return float(self.model.score(X, y))
        return 0.0

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
