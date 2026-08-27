"""10d explanation visualizers — feature importance, SHAP family, PDP."""

from __future__ import annotations

from typing import Any

import polars as pl

from ferrum._validate import validate_choice
from ferrum.plots.explanation import (
    _importance_chart_from_source,
    _shap_bar_chart_from_source,
    _shap_beeswarm_chart_from_source,
    _shap_waterfall_chart_from_source,
)
from ._base import FerrumVisualizer


class FeatureImportancesVisualizer(FerrumVisualizer):
    """Visualize feature importances from a fitted sklearn estimator.

    Wraps ``importance_chart`` in the sklearn-protocol visualizer interface.
    ``method="builtin"`` reads ``feature_importances_`` or ``coef_`` directly
    from the estimator (std is 0 in this case). ``method="permutation"`` runs
    ``sklearn.inspection.permutation_importance`` using the supplied
    ``random_state`` seed. The headline metric recorded in ``_metrics`` is
    ``top_feature_importance`` — the importance of the highest-ranked feature
    after sorting.

    Parameters
    ----------
    model : Any
        Fitted sklearn-compatible estimator. Must expose
        ``feature_importances_`` or ``coef_`` (``method="builtin"``) or
        support ``predict`` / ``predict_proba`` (``method="permutation"``).
    method : {"builtin", "permutation"}, default "builtin"
        Strategy for extracting importances. ``"builtin"`` reads the
        estimator attribute directly (zero standard deviation). ``"permutation"``
        shuffles each feature and measures the drop in score.
    top_k : int or None, default 20
        Maximum number of features to display, ranked by importance. Pass
        ``None`` to show all features.
    orient : {"horizontal", "vertical"}, default "horizontal"
        Bar orientation in the rendered chart.
    error_bars : bool, default True
        Whether to draw ±1 std error bars. Has no visual effect when
        ``method="builtin"`` because std is always 0 in that case.
    random_state : int or None, optional
        RNG seed forwarded to ``permutation_importance``. Ignored when
        ``method="builtin"``.
    theme : Theme or None, optional
        Per-chart theme override. Falls back to the global default when
        ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
    >>> viz = fm.FeatureImportancesVisualizer(model).fit(X_train, y_train)
    >>> viz.show()
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
            method=self.method,
            random_state=self.random_state,
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
    """Visualize SHAP values from a fitted sklearn estimator.

    Wraps the ``shap_chart`` family in the sklearn-protocol visualizer
    interface. Requires the ``shap`` library (``pip install ferrum[shap]``).
    Three chart kinds are supported: ``"beeswarm"`` shows per-sample SHAP
    scatter colored by feature value; ``"bar"`` shows mean absolute SHAP
    aggregated per feature; ``"waterfall"`` shows the cumulative contribution
    for a single sample selected by ``sample_idx``. The headline metric
    recorded in ``_metrics`` is ``top_abs_shap`` — the maximum mean absolute
    SHAP value across all features.

    Parameters
    ----------
    model : Any
        Fitted sklearn-compatible estimator supported by the ``shap`` library
        (e.g. tree ensembles, linear models with a ``shap.Explainer``).
    kind : {"beeswarm", "bar", "waterfall"}, default "beeswarm"
        Chart style to render. ``"waterfall"`` requires ``sample_idx`` to be
        set; a ``ValueError`` is raised at ``.show()`` time if it is omitted.
    max_display : int, default 20
        Maximum number of features to include in the chart, ranked by mean
        absolute SHAP value.
    sample_idx : int or None, optional
        Row index of the sample to explain. Required when ``kind="waterfall"``;
        ignored for ``"beeswarm"`` and ``"bar"``.
    order : {"abs_mean", "mean", "max", "none"}, default "abs_mean"
        Feature ordering strategy applied to all three ``kind`` values
        (shared vocabulary with ``Chart.mark_shap_beeswarm(order=)``).
        ``"abs_mean"`` ranks features by descending ``mean(|shap_value|)``
        across the dataset; ``"max"`` ranks by descending
        ``max(|shap_value|)`` (surfaces high-impact-outlier features; also
        the ``kind="bar"`` x-value aggregation for this one setting);
        ``"mean"`` ranks by descending signed ``mean(shap_value)``;
        ``"none"`` keeps the first ``max_display`` features in
        row-encounter order, unranked. For ``kind="bar"`` with any other
        ``order``, the bar's x-value is ``mean(|shap_value|)``; for
        ``kind="waterfall"`` ``order`` selects which features are shown for
        the single sample and the order they appear in.
    background : Any or None, optional
        Background dataset passed to the SHAP explainer for models that
        require a reference distribution (e.g. kernel SHAP). Pass ``None``
        to use the explainer's default.
    random_state : int or None, optional
        RNG seed forwarded to the underlying ``ModelSource``. Ignored when
        SHAP computation is deterministic.
    theme : Theme or None, optional
        Per-chart theme override. Falls back to the global default when
        ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> model = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
    >>> viz = fm.SHAPVisualizer(model, kind="beeswarm").fit(X_train, y_train)
    >>> viz.show()
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
        import warnings

        warnings.warn(
            "SHAPVisualizer(kind=...) is deprecated; use "
            "SHAPBeeswarmVisualizer / SHAPBarVisualizer / "
            "SHAPWaterfallVisualizer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(model, random_state=random_state, theme=theme)
        self.kind = kind
        self.max_display = max_display
        self.sample_idx = sample_idx
        self.order = order
        self.background = background

    def _materialize(self) -> None:
        sv = self._source.shap_values(background=self.background)
        # For waterfall, the chart only shows a single sample, so the
        # headline metric reflects that sample — not the global aggregation.
        # When sample_idx is None the metric is 0.0 (the ValueError is
        # deferred to _build_chart so the message stays consistent); since
        # _top_abs_shap(sample_idx=None) would aggregate the whole dataset,
        # short-circuit to 0.0 here before delegating.
        if self.kind == "waterfall" and self.sample_idx is None:
            self._metrics["top_abs_shap"] = 0.0
            return
        sample_idx = self.sample_idx if self.kind == "waterfall" else None
        self._metrics["top_abs_shap"] = _top_abs_shap(sv, self.order, sample_idx=sample_idx)

    def _build_chart(self) -> Any:
        validate_choice("SHAPVisualizer", "kind", self.kind, ("beeswarm", "bar", "waterfall"))
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
                order=self.order,
                background=self.background,
                theme=self.theme,
            )
        # kind == "waterfall" (validated above)
        if self.sample_idx is None:
            raise ValueError("SHAPVisualizer(kind='waterfall') requires sample_idx=<int>.")
        return _shap_waterfall_chart_from_source(
            self._source,
            sample_idx=self.sample_idx,
            max_display=self.max_display,
            order=self.order,
            background=self.background,
            theme=self.theme,
        )


def _top_abs_shap(
    frame: "pl.DataFrame",
    order: str,
    *,
    sample_idx: int | None = None,
) -> float:
    """Headline SHAP metric for a long-form ``shap_values`` frame.

    The single home for the abs-mean / abs-max aggregation that the
    deprecated ``SHAPVisualizer`` and the three dedicated SHAP visualizers
    all need. ``order="max"`` restores this function's original pre-batch
    behavior byte-for-byte (2026-08-27 close-out: `order`'s vocabulary was
    briefly narrowed to drop `"max"`, which broke this; restored to the
    union `{"abs_mean", "mean", "max", "none"}` -- see
    ``_shap_order_features``). Every other ``order`` value aggregates by
    mean, matching this class's own docstring ("the maximum **mean**
    absolute SHAP value").

    - ``sample_idx is None`` (beeswarm / bar): aggregate ``|shap_value|`` per
      feature by mean (default, and every ``order`` except ``"max"``) or max
      (``order="max"``), then return the largest per-feature value across
      all features.
    - ``sample_idx`` given (waterfall): return the max ``|shap_value|`` over
      the single explained sample's rows, regardless of ``order``.

    Returns ``0.0`` when the relevant rows are empty (no features, or the
    requested ``sample_idx`` is absent), matching the prior per-visualizer
    behavior exactly.
    """
    if sample_idx is not None:
        one = frame.filter(pl.col("sample_id") == sample_idx)
        return float(one["shap_value"].abs().max()) if one.height else 0.0
    expr = pl.col("shap_value").abs()
    agg_expr = expr.max() if order == "max" else expr.mean()
    agg = frame.group_by("feature").agg(agg_expr.alias("v"))
    return float(agg["v"].max()) if agg.height else 0.0


class _SHAPBaseMixin:
    """Shared ``_materialize`` for the three SHAP sibling visualizers.

    Computes ``shap_values`` once on the underlying ``ModelSource`` and
    records ``top_abs_shap`` in ``self._metrics``.  Subclasses select
    which subset of the SHAP frame drives the metric (whole-dataset
    aggregation for beeswarm / bar, single-sample max for waterfall) by
    passing ``sample_idx`` to :func:`_top_abs_shap`.
    """

    def _shap_dataframe(self) -> "pl.DataFrame":
        return self._source.shap_values(background=self.background)

    def _record_top_abs_shap(self, *, sample_idx: int | None = None) -> None:
        """Compute and store ``top_abs_shap`` from a freshly-read SHAP frame."""
        self._metrics["top_abs_shap"] = _top_abs_shap(
            self._shap_dataframe(),
            self.order,
            sample_idx=sample_idx,
        )


class SHAPBeeswarmVisualizer(_SHAPBaseMixin, FerrumVisualizer):
    """Per-sample SHAP scatter colored by z-scored feature value.

    Parameters
    ----------
    model : Any
        Fitted sklearn-compatible estimator supported by the ``shap``
        library (e.g. tree ensembles, linear models).
    max_display : int, default 20
        Maximum number of features ranked by ``order``.
    order : {"abs_mean", "mean", "max", "none"}, default "abs_mean"
        Feature ranking criterion (shared vocabulary with
        ``Chart.mark_shap_beeswarm(order=)``).  ``"abs_mean"`` ranks by
        descending mean absolute SHAP; ``"max"`` by descending max
        absolute SHAP (surfaces high-impact-outlier features); ``"mean"``
        by descending signed mean SHAP; ``"none"`` keeps the first
        ``max_display`` features in row-encounter order, unranked.
    background : Any, optional
        Background dataset passed to the SHAP explainer for kernel-
        SHAP models.  Tree SHAP ignores this.
    per_class : bool, default False
        Facet by class on multi-class classifiers.
    random_state, theme : forwarded to ``FerrumVisualizer``.

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.SHAPBeeswarmVisualizer(model, max_display=15).fit(X, y)
    >>> viz.show()
    """

    def __init__(
        self,
        model: Any,
        *,
        max_display: int = 20,
        order: str = "abs_mean",
        background: Any = None,
        per_class: bool = False,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.max_display = max_display
        self.order = order
        self.background = background
        self.per_class = per_class

    def _materialize(self) -> None:
        self._record_top_abs_shap()

    def _build_chart(self) -> Any:
        return _shap_beeswarm_chart_from_source(
            self._source,
            max_display=self.max_display,
            order=self.order,
            background=self.background,
            per_class=self.per_class,
            theme=self.theme,
        )


class SHAPBarVisualizer(_SHAPBaseMixin, FerrumVisualizer):
    """Mean-absolute SHAP per feature as a horizontal bar chart.

    Parameters mirror [SHAPBeeswarmVisualizer][ferrum.SHAPBeeswarmVisualizer]; see that class
    for the full parameter list.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> model = GradientBoostingClassifier().fit(X_train, y_train)
    >>> viz = fm.SHAPBarVisualizer(model).fit(X_test, y_test).show()
    """

    def __init__(
        self,
        model: Any,
        *,
        max_display: int = 20,
        order: str = "abs_mean",
        background: Any = None,
        per_class: bool = False,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.max_display = max_display
        self.order = order
        self.background = background
        self.per_class = per_class

    def _materialize(self) -> None:
        self._record_top_abs_shap()

    def _build_chart(self) -> Any:
        return _shap_bar_chart_from_source(
            self._source,
            max_display=self.max_display,
            order=self.order,
            background=self.background,
            per_class=self.per_class,
            theme=self.theme,
        )


class SHAPWaterfallVisualizer(_SHAPBaseMixin, FerrumVisualizer):
    """Cumulative per-feature SHAP contributions for one sample.

    Parameters
    ----------
    model : Any
        Fitted sklearn-compatible estimator supported by the ``shap``
        library.
    sample_idx : int
        Row index (0-based) of the sample to explain.  Required.
    max_display : int, default 20
        Maximum number of features to include in the waterfall, ranked
        by ``order``.
    order : {"abs_mean", "mean", "max", "none"}, default "abs_mean"
        Feature ranking criterion (shared vocabulary with
        ``Chart.mark_shap_beeswarm(order=)``; drives both the
        top-``max_display`` selection and the bar order).
    background : Any, optional
        Background dataset passed to the SHAP explainer.
    per_class : bool, default False
        Facet by class on multi-class classifiers.
    random_state, theme : forwarded to ``FerrumVisualizer``.

    Raises
    ------
    ValueError
        If ``sample_idx`` is missing at ``__init__`` time.

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.SHAPWaterfallVisualizer(model, sample_idx=3).fit(X, y)
    >>> viz.show()
    """

    def __init__(
        self,
        model: Any,
        *,
        sample_idx: int,
        max_display: int = 20,
        order: str = "abs_mean",
        background: Any = None,
        per_class: bool = False,
        random_state: int | None = None,
        theme: Any = None,
    ):
        if sample_idx is None:
            raise ValueError("SHAPWaterfallVisualizer requires sample_idx=<int>.")
        super().__init__(model, random_state=random_state, theme=theme)
        self.sample_idx = int(sample_idx)
        self.max_display = max_display
        self.order = order
        self.background = background
        self.per_class = per_class

    def _materialize(self) -> None:
        self._record_top_abs_shap(sample_idx=self.sample_idx)

    def _build_chart(self) -> Any:
        return _shap_waterfall_chart_from_source(
            self._source,
            sample_idx=self.sample_idx,
            max_display=self.max_display,
            order=self.order,
            background=self.background,
            per_class=self.per_class,
            theme=self.theme,
        )
