"""10a regression visualizers — ResidualsVisualizer, PredictionErrorVisualizer,
CooksDistanceVisualizer.
"""

from __future__ import annotations

from typing import Any

from ferrum.plots.regression import (
    _prediction_error_chart_from_source,
    _residuals_chart_from_source,
)
from .base import FerrumVisualizer


class ResidualsVisualizer(FerrumVisualizer):
    """Residuals-vs-fitted diagnostic for a regression estimator.

    Wraps ``ModelSource.predictions()``. Records ``rmse`` and ``mae`` as
    headline metrics; the chart shows ``y_pred`` on x and the chosen
    residual variant on y with a horizontal reference line at zero.

    Parameters
    ----------
    model : Any
        Fitted regression estimator (must implement ``predict``).
    kind : {"studentized", "raw", "scaled"}, default "studentized"
        Which residual variant to plot. ``"studentized"`` uses the
        leverage-aware hat-matrix form when ``model`` exposes
        ``coef_``; otherwise falls back to internally-studentized
        (residual / std).
    random_state : int, optional
        Forwarded to the underlying ``ModelSource``.
    theme : Theme, optional

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.ResidualsVisualizer(model).fit(X, y)
    >>> viz._metrics["rmse"]
    1.234
    >>> viz.show()
    """

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

    has_score: bool = True

    def _materialize(self) -> None:
        df = self._source.predictions()
        resid = df["residual"]
        self._metrics["rmse"] = float((resid**2).mean() ** 0.5)
        self._metrics["mae"] = float(resid.abs().mean())

    def _build_chart(self) -> Any:
        return _residuals_chart_from_source(
            self._source,
            kind=self.kind,
            theme=self.theme,
        )

    def score(self, X: Any, y: Any) -> float:
        return float(self.model.score(X, y))


class PredictionErrorVisualizer(FerrumVisualizer):
    """Actual-vs-predicted scatter for a regression estimator.

    Wraps ``ModelSource.predictions()``. Records ``rmse`` as the headline
    metric; the chart shows ``y_true`` on x and ``y_pred`` on y with an
    optional identity line and CI / reference band overlays.

    Parameters
    ----------
    model : Any
        Fitted regression estimator.
    reference_line : bool, default True
        Overlay the dashed ``y = x`` diagonal.
    ci : float, optional
        Confidence level in ``(0, 1)``. When set, overlays a ribbon
        spanning the central ``ci`` fraction of residuals around the
        reference line. Raises ``ValueError`` if not in ``(0, 1)``.
    reference_band : bool, default False
        When ``True`` (and ``ci`` is None), overlays a ±1 RMSE ribbon
        around the reference line.
    random_state : int, optional
    theme : Theme, optional

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.PredictionErrorVisualizer(model, ci=0.95).fit(X, y)
    >>> viz.show()
    """

    def __init__(
        self,
        model: Any,
        *,
        reference_line: bool = True,
        ci: float | None = None,
        reference_band: bool = False,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.reference_line = reference_line
        self.ci = ci
        self.reference_band = reference_band

    has_score: bool = True

    def _materialize(self) -> None:
        df = self._source.predictions()
        resid = df["residual"]
        self._metrics["rmse"] = float((resid**2).mean() ** 0.5)

    def _build_chart(self) -> Any:
        return _prediction_error_chart_from_source(
            self._source,
            reference_line=self.reference_line,
            ci=self.ci,
            reference_band=self.reference_band,
            theme=self.theme,
        )

    def score(self, X: Any, y: Any) -> float:
        return float(self.model.score(X, y))


class CooksDistanceVisualizer(FerrumVisualizer):
    """Cook's distance diagnostic for a regression estimator.

    Surfaces ``max_studentized`` (the largest absolute studentized
    residual) as a quick proxy for influential observations; the
    leverage-aware Cook's D itself is materialized by
    ``ModelSource.predictions().cooks_distance`` for linear estimators
    and is rendered by ``mark_residuals(cook_threshold=...)``. This
    visualizer plots the residuals chart with the leverage panel
    enabled.

    Parameters
    ----------
    model : Any
        Fitted regression estimator (must expose ``coef_`` for the
        leverage-aware Cook's distance).
    threshold : float or "auto", optional
        Cook's-distance threshold for highlighting outliers on the
        residuals-vs-leverage panel. Forwarded to
        ``_residuals_chart_from_source(cook_threshold=...)``, which
        injects ``_cook_outlier_x`` / ``_cook_outlier_y`` columns
        (keyed on leverage and the studentized residual) and the
        leverage panel renders them as a red-filled overlay layer.
        ``"auto"`` uses the conventional ``4 / n`` rule (Hair et al.).
    random_state : int, optional
    theme : Theme, optional

    Examples
    --------
    >>> import ferrum as fm
    >>> viz = fm.CooksDistanceVisualizer(linear_model).fit(X, y)
    >>> viz._metrics["max_studentized"]
    3.42
    """

    def __init__(
        self,
        model: Any,
        *,
        threshold: float | str | None = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.threshold = threshold

    def _materialize(self) -> None:
        df = self._source.predictions()
        stud = df["studentized_residual"]
        self._metrics["max_studentized"] = float(stud.abs().max())

    def _build_chart(self) -> Any:
        return _residuals_chart_from_source(
            self._source,
            kind="studentized",
            cook_threshold=self.threshold,
            panels=["residuals_vs_leverage"],
            theme=self.theme,
        )
