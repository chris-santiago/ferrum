"""10a regression visualizers — ResidualsVisualizer, PredictionErrorVisualizer,
CooksDistanceVisualizer.
"""

from __future__ import annotations

from typing import Any

from ferrum.plots.regression import (
    _prediction_error_chart_from_source,
    _residuals_chart_from_source,
    _resolve_residuals_panels,
)
from ._base import FerrumVisualizer

# CooksDistanceVisualizer's historical default -- exactly the leverage panel,
# regardless of what `_resolve_residuals_panels` would do with `None`/"single"
# (which mean "just residuals_vs_fitted" for `ResidualsVisualizer`). Kept as
# an immutable sentinel so a caller who explicitly passes `panels=` (e.g.
# `panels="auto"` or an explicit list) still routes through the shared
# expansion helper, while the unset default stays byte-identical to the
# pre-batch-G behavior.
_COOKS_DEFAULT_PANELS: tuple[str, ...] = ("residuals_vs_leverage",)


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
    cook_threshold : float, "auto", or None, default None
        Cook's-distance threshold for outlier highlighting on the
        residuals-vs-fitted (and, in a multi-panel layout,
        residuals-vs-leverage) view. See ``residuals_chart``.
    panels : "auto", None, "single", or list of str, default None
        Panel selection, matching ``residuals_chart(panels=...)``.
        ``"auto"`` renders the canonical 4-panel diagnostic grid
        (residuals-vs-fitted, QQ, scale-location, residuals-vs-leverage);
        ``None`` or ``"single"`` (the default) renders just the
        residuals-vs-fitted panel. The ``"auto"`` expansion is shared
        with ``residuals_chart`` via ``_resolve_residuals_panels`` so
        ``ResidualsVisualizer(panels="auto")`` renders the same 4-panel
        grid.
    annotate_metrics : bool, default True
        Overlay a top-right corner annotation showing R^2/RMSE/MAE.
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
    mark : dict, optional
        Per-layer mark overrides. See ``residuals_chart``.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``.
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
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

    Canonical 4-panel diagnostic grid, matching
    ``residuals_chart(panels="auto")``:

    >>> fm.ResidualsVisualizer(model, panels="auto").fit(X, y).show()
    """

    def __init__(
        self,
        model: Any,
        *,
        kind: str = "studentized",
        cook_threshold: float | str | None = None,
        panels: Any = None,
        annotate_metrics: bool = True,
        subtitle: str | None = None,
        mark: dict | None = None,
        encode: dict | None = None,
        properties: dict | None = None,
        layers: list | None = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.kind = kind
        self.cook_threshold = cook_threshold
        self.panels = panels
        self.annotate_metrics = annotate_metrics
        self.subtitle = subtitle
        self.mark = mark
        self.encode = encode
        self.properties = properties
        self.layers = layers

    def _materialize(self) -> None:
        df = self._source.predictions()
        resid = df["residual"]
        self._metrics["rmse"] = float((resid**2).mean() ** 0.5)
        self._metrics["mae"] = float(resid.abs().mean())

    def _build_chart(self) -> Any:
        return _residuals_chart_from_source(
            self._source,
            kind=self.kind,
            cook_threshold=self.cook_threshold,
            panels=_resolve_residuals_panels(self.panels),
            annotate_metrics=self.annotate_metrics,
            subtitle=self.subtitle,
            mark=self.mark,
            encode=self.encode,
            properties=self.properties,
            layers=self.layers,
            theme=self.theme,
        )


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
    mark : dict, optional
        Per-layer mark overrides. See ``prediction_error_chart``.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``.
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
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
        mark: dict | None = None,
        encode: dict | None = None,
        properties: dict | None = None,
        layers: list | None = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.reference_line = reference_line
        self.ci = ci
        self.reference_band = reference_band
        self.mark = mark
        self.encode = encode
        self.properties = properties
        self.layers = layers

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
            mark=self.mark,
            encode=self.encode,
            properties=self.properties,
            layers=self.layers,
            theme=self.theme,
        )


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
        Kept as ``threshold=`` (not ``cook_threshold=``, the builder's own
        spelling) for back-compat -- an adjudicated shape drift, not
        unified (batch G decision doc, G5).
    kind : {"studentized", "raw", "scaled"}, default "studentized"
        Residual variant underlying the leverage panel's y axis.
    panels : "auto", None, "single", or list of str, default the leverage panel alone
        Panel selection, matching ``residuals_chart(panels=...)``. The
        default renders just ``residuals_vs_leverage`` (this
        visualizer's historical behavior); pass ``"auto"`` for the
        canonical 4-panel grid or an explicit list to customize it.
    annotate_metrics : bool, default True
        Overlay a top-right corner annotation showing R^2/RMSE/MAE on
        the residuals-vs-fitted panel, when present.
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
    mark : dict, optional
        Per-layer mark overrides. See ``cooks_distance_chart``.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``.
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
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
        kind: str = "studentized",
        panels: Any = _COOKS_DEFAULT_PANELS,
        annotate_metrics: bool = True,
        subtitle: str | None = None,
        mark: dict | None = None,
        encode: dict | None = None,
        properties: dict | None = None,
        layers: list | None = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.threshold = threshold
        self.kind = kind
        self.panels = panels
        self.annotate_metrics = annotate_metrics
        self.subtitle = subtitle
        self.mark = mark
        self.encode = encode
        self.properties = properties
        self.layers = layers

    def _materialize(self) -> None:
        df = self._source.predictions()
        stud = df["studentized_residual"]
        self._metrics["max_studentized"] = float(stud.abs().max())

    def _build_chart(self) -> Any:
        panels = self.panels
        panel_list = (
            list(panels) if isinstance(panels, (list, tuple)) else _resolve_residuals_panels(panels)
        )
        return _residuals_chart_from_source(
            self._source,
            kind=self.kind,
            cook_threshold=self.threshold,
            panels=panel_list,
            annotate_metrics=self.annotate_metrics,
            subtitle=self.subtitle,
            mark=self.mark,
            encode=self.encode,
            properties=self.properties,
            layers=self.layers,
            theme=self.theme,
        )
