"""FerrumVisualizer base — see ferrum-spec.md §3.15.

fit() materializes derived data and builds the chart; show() returns the
Chart for rendering.

``fit`` is one template over three hooks, which subclasses override; no
subclass overrides ``fit`` itself, so every visualizer's chart is built in
its class's voice:

1. ``_build_source(X, y)`` returns what the chart is computed from. By
   default a ``ModelSource`` wrapping ``self.model``. A subclass may build
   it differently (``ROCVisualizer``, ``PRVisualizer``,
   ``DiscriminationThresholdVisualizer`` and ``CalibrationVisualizer``
   route ``compare=`` / ``*models`` to a ``ComparedModelSource``), or, with
   no model, return the prepared data itself (``ClassBalanceVisualizer``,
   ``Rank1DVisualizer`` / ``Rank2DVisualizer`` /
   ``ParallelCoordinatesVisualizer``, ``ElbowVisualizer``).
2. ``_materialize()`` reads ``self._source`` and writes ``_metrics``.
3. ``_build_chart()`` returns the chart built from ``self._source``.
"""

from __future__ import annotations

from typing import Any

from ferrum._entry import stamp_entry, voiced_as


class FerrumVisualizer:
    """Base class for sklearn-protocol model-diagnostic visualizers.

    Concrete visualizers override the ``_build_source``, ``_materialize``
    and ``_build_chart`` hooks, never ``fit``. ``fit`` runs all three in the
    class's voice (:func:`ferrum._entry.voiced_as`) and stamps the class
    name on the chart (:func:`ferrum._entry.stamp_entry`), so a malformed
    ``mark=`` refuses at ``fit``, the first point the chart's layers exist,
    and every warning and refusal about the chart names the class.

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

        Builds the source (``_build_source``), calls ``_materialize`` to
        populate ``_metrics``, then ``_build_chart`` to assemble the ferrum
        ``Chart``, all speaking as the class. Returns ``self`` for method
        chaining.

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
        entry = type(self).__name__
        with voiced_as(entry):
            self._source = self._build_source(X, y)
            self._materialize()
            chart = self._build_chart()
        self._chart = stamp_entry(chart, entry)
        self._fitted = True
        return self

    def _build_source(self, X: Any, y: Any) -> Any:
        """Return what the chart is computed from, stored as ``self._source``.

        Default implementation for the standard model-backed flow: a
        ``ModelSource`` wrapping ``self.model``. Subclasses with other
        source-construction options (e.g. a ``compare=`` kwarg that should
        build a ``ComparedModelSource``), or with no model at all, override
        this hook.
        """
        import ferrum

        return ferrum.ModelSource(self.model, X, y, random_state=self.random_state)

    def _materialize(self) -> None:
        """Compute derived data and populate ``_metrics``.

        Default implementation is a no-op, for a visualizer with no
        headline metric. Subclasses override this to read off
        ``self._source`` and write headline metrics into ``self._metrics``.
        """

    def _build_chart(self) -> Any:
        """Assemble and return the ferrum ``Chart`` from ``self._source``.

        Every visualizer overrides it.
        """
        raise TypeError(f"{type(self).__name__} defines no _build_chart")

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
