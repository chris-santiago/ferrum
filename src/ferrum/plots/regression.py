"""Regression domain figure functions and their private builders.

Public API
----------
lmplot, regplot, residplot, residuals_chart, prediction_error_chart,
cooks_distance_chart.

Each public function wraps ``_resolve_source`` (shared across all figure
functions) and dispatches to a co-located ``_*_from_source`` builder that
produces a fully-formed ``Chart``.

Internal-only builders
----------------------
_residuals_chart_from_source, _residuals_panel,
_prediction_error_chart_from_source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from ferrum import (
    Chart,
    Glm,
    Jitter,
    Logistic,
    Robust,
    Smooth,
)
from ferrum._validate import validate_choice
from ferrum.diagnostics.source import ComparedModelSource
from ferrum.encoding import X, Y
from ferrum.plots._helpers import (
    _color_field_for,
    _compose_compare,
    _field_name,
    _finalize_chart,
    _grid_panels,
    _inject_cook_outliers,
    _inject_metrics_corner,
    _merge_layers,
    _overlay_metrics_corner,
    _resolve_source,
    _to_polars,
)

if TYPE_CHECKING:
    from ferrum import ConcatChart, HConcatChart, VConcatChart


_VALID_METHODS = frozenset({"lm", "logistic", "glm", "loess", "robust"})


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _residuals_chart_from_source(
    source: Any,
    *,
    kind: str = "studentized",
    cook_threshold: float | str | None = None,
    panels: Any = None,  # None / "single" / list of panel names
    annotate_metrics: bool = True,
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Build a residuals diagnostic chart from a ModelSource.

    Schwabish SB3 (2026-05-11): the corner R^2/RMSE/MAE annotation rides
    on the residuals-vs-fitted view in both single-panel and 4-panel
    layouts. ``_inject_metrics_corner`` augments the shared DataFrame
    once at the top of this function; the single-panel chart and the
    ``residuals_vs_fitted`` cell of the 4-panel grid then call
    ``_overlay_metrics_corner`` which detects the injected columns and
    adds a same-data ``mark_text`` layer. The other 4-panel cells (QQ,
    scale-location, residuals-vs-leverage) inherit the extra columns but
    do not reference them, so their renderers are unaffected.
    """
    import ferrum

    df = source.predictions()
    # The corner R²/RMSE/MAE metrics conflate models (a single R²/RMSE over all
    # rows, anchored at one ``y_pred.arg_max()`` row), so skip the injection when
    # comparing. This mirrors how classification gates its active title to the
    # single-model case.
    is_single_model = "model" not in df.columns
    if annotate_metrics and is_single_model:
        df = _inject_metrics_corner(df, kind=kind)

    if panels in (None, "single"):
        if cook_threshold is not None:
            df = _inject_cook_outliers(df, kind=kind, threshold=cook_threshold)
        # When a ``model`` column is present (compare-source path), drive
        # per-model colour on the residual scatter; absent it, ``None`` is
        # byte-identical to the single-model default.
        chart = ferrum.Chart(df).mark_residuals(
            kind=kind,
            cook_threshold=cook_threshold,
            color_field=_color_field_for(df, None),
        )
        chart = chart.properties(
            title=ferrum.Title("Residuals", subtitle=subtitle),
        )
        # ``_overlay_metrics_corner`` is a no-op when the injected
        # columns are absent, so the call is unconditional.
        chart = _overlay_metrics_corner(chart)
        return _finalize_chart(
            chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
        )

    # Multi-panel path: each panel injects its own outlier columns with
    # the right x-axis encoding for that panel's coordinate system. The
    # leverage panel depends on hat-matrix quantities; for non-linear
    # estimators ModelSource.predictions emits NaN for every leverage row,
    # in which case we drop the panel so the remaining panels still render.
    # If the drop would leave nothing to render (leverage was the only
    # panel requested), we raise instead (GH #44).
    panel_list = panels if isinstance(panels, list) else ["residuals_vs_fitted"]
    if "residuals_vs_leverage" in panel_list and df["leverage"].is_nan().all():
        panel_list = [p for p in panel_list if p != "residuals_vs_leverage"]
        if not panel_list:
            model = getattr(source, "model", None)
            estimator_clause = f" Got {type(model).__name__!r}." if model is not None else ""
            raise ValueError(
                "Cook's distance and residuals-vs-leverage are hat-matrix "
                "quantities that require a linear estimator exposing "
                f"`coef_`.{estimator_clause} Pass a linear estimator (e.g. "
                "LinearRegression, Ridge, Lasso), or drop "
                "residuals_vs_leverage from panels= and use the other "
                "diagnostic panels instead."
            )
    charts = [
        _residuals_panel(df, name, kind=kind, cook_threshold=cook_threshold) for name in panel_list
    ]
    # _grid_panels applies theme internally; apply overrides before theme.
    chart = _grid_panels(charts)
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def _residuals_panel(
    df: pl.DataFrame,
    name: str,
    *,
    kind: str = "studentized",
    cook_threshold: float | str | None = None,
):
    """One sub-panel of a multi-panel residuals chart.

    When ``cook_threshold`` is set, the residuals_vs_fitted and
    residuals_vs_leverage panels overlay an outlier-highlight layer
    keyed on each panel's own x axis (y_pred for residuals_vs_fitted,
    leverage for residuals_vs_leverage).
    """
    import ferrum

    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"

    if name == "residuals_vs_fitted":
        if cook_threshold is not None:
            df = _inject_cook_outliers(df, kind=kind, threshold=cook_threshold)
            chart = ferrum.Chart(df).mark_residuals(
                kind=kind,
                cook_threshold=cook_threshold,
            )
        else:
            chart = ferrum.Chart(df).mark_residuals(kind=kind)
        chart = chart.encode(
            x=X("y_pred", title="Fitted values"),
            y=Y(y_col, title="Studentized residual"),
        )
        chart = chart.properties(title=ferrum.Title("Residuals vs Fitted"))
        # Schwabish SB3: detect the ``_metrics_text``/``_metrics_y``
        # columns injected upstream by ``_inject_metrics_corner`` and
        # overlay the corner annotation. No-op when the columns are
        # absent, so an externally-built panel (e.g. a user invoking
        # ``_residuals_panel`` directly without metrics injection)
        # renders identically to the pre-SB3 behavior.
        return _overlay_metrics_corner(chart)

    if name == "qq":
        return (
            ferrum.Chart(df)
            .mark_qq()
            .encode(
                x=X("studentized_residual", title="Theoretical quantiles"),
                y=Y("sample", title="Sample quantiles"),
            )
            .properties(title=ferrum.Title("Normal Q–Q"))
        )

    if name == "scale_location":
        d2 = df.with_columns(pl.col("studentized_residual").abs().sqrt().alias("sqrt_abs_resid"))
        return (
            ferrum.Chart(d2)
            .mark_point()
            .encode(
                x=X("y_pred", title="Fitted values"),
                y=Y("sqrt_abs_resid", title="√|Studentized residual|"),
            )
            .properties(title=ferrum.Title("Scale–Location"))
        )

    if name == "residuals_vs_leverage":
        # Inject outlier columns BEFORE constructing the base chart so
        # base and overlay layers share the same DataFrame identity
        # (avoids unnecessary null-pad merge in Chart.__add__).
        # x_col="leverage" so the overlay sits at the right
        # x coordinate for this panel.
        if cook_threshold is not None:
            df = _inject_cook_outliers(
                df,
                kind=kind,
                threshold=cook_threshold,
                x_col="leverage",
            )
        base = (
            ferrum.Chart(df)
            .mark_point()
            .encode(
                x=X("leverage", title="Leverage"),
                y=Y(y_col, title="Studentized residual"),
            )
            .properties(title=ferrum.Title("Residuals vs Leverage"))
        )
        if cook_threshold is not None and "_cook_outlier_x" in df.columns:
            overlay = (
                ferrum.Chart(df)
                .mark_point(
                    fill="#e15759",  # tableau red, matches mark_residuals overlay
                    stroke="#000000",
                    stroke_width=1.0,
                    size=80.0,
                )
                .encode(x="_cook_outlier_x", y="_cook_outlier_y")
            )
            return base + overlay
        return base

    raise ValueError(f"unknown residuals panel: {name!r}")


def _prediction_error_chart_from_source(
    source: Any,
    *,
    reference_line: bool = True,
    ci: float | None = None,
    reference_band: bool = False,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Build an actual-vs-predicted error chart from a ModelSource.

    When ``ci`` or ``reference_band`` is set, computes a residual band
    and injects ``_pe_band_lo`` / ``_pe_band_hi`` columns:

    - ``ci`` (float in (0, 1)): band spans the central ``ci`` fraction
      of residuals -- ``y_true + quantile_low`` to ``y_true +
      quantile_high`` where the quantiles bracket ``(1 - ci) / 2``.
    - ``reference_band=True`` (without ``ci``): band spans +/-1 RMSE
      around the identity line.
    """
    import ferrum

    df = source.predictions().sort("y_true")
    # Internal invariant: the residual band (``ci`` / ``reference_band``) is a
    # single-model aggregate, so this builder must be handed a single-model
    # source when a band is requested. ``prediction_error_chart`` enforces this
    # by routing the multi-model band case through ``_compose_compare`` (one
    # single-model panel per model). A pooled ``model`` frame reaching here with
    # a band set is therefore a routing bug, not user error — fail loudly rather
    # than conflate residuals across models.
    if "model" in df.columns and (ci is not None or reference_band):
        raise ValueError(
            "internal: _prediction_error_chart_from_source received a pooled "
            "multi-model frame with ci=/reference_band= set; the multi-model "
            "band path must be routed through _compose_compare (one single-model "
            "panel per model) by prediction_error_chart."
        )
    if ci is not None or reference_band:
        residuals = df["y_pred"] - df["y_true"]
        if ci is not None:
            if not 0.0 < ci < 1.0:
                raise ValueError(f"mark_prediction_error(ci={ci!r}) — expected a value in (0, 1).")
            alpha = (1.0 - float(ci)) / 2.0
            q_lo = float(residuals.quantile(alpha, interpolation="linear"))
            q_hi = float(residuals.quantile(1.0 - alpha, interpolation="linear"))
        else:
            sigma = float((residuals**2).mean() ** 0.5)
            q_lo = -sigma
            q_hi = sigma
        df = df.with_columns(
            (pl.col("y_true") + q_lo).alias("_pe_band_lo"),
            (pl.col("y_true") + q_hi).alias("_pe_band_hi"),
        )
    # When a ``model`` column is present (compare-source path), drive per-model
    # colour on the scatter; absent it, ``None`` is byte-identical to the
    # single-model default.
    chart = ferrum.Chart(df).mark_prediction_error(
        reference_line=reference_line,
        ci=ci,
        reference_band=reference_band,
        color_field=_color_field_for(df, None),
    )
    chart = chart.encode(
        x=X("y_pred", title="Predicted value"),
        y=Y("y_true", title="True value"),
    )
    chart = chart.properties(title=ferrum.Title("Prediction Error"))
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


# ---------------------------------------------------------------------------
# Public figure functions
# ---------------------------------------------------------------------------


def lmplot(
    data: Any,
    *,
    x: str,
    y: str,
    hue: Any = None,
    col: Any = None,
    row: Any = None,
    method: str = "lm",
    ci: Any = 95,
    order: int = 1,
    scatter: bool = True,
    scatter_kws: Any = None,
    line_kws: Any = None,
    truncate: bool = True,
    x_bins: Any = None,
    x_estimator: Any = None,
    x_jitter: Any = None,
    logx: bool = False,
    show_metrics: bool = True,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Linear (and non-linear) regression scatter overlay.

    Builds a layered chart with an optional scatter (``mark_point``) and a
    regression fit line, dispatching to the appropriate transform:

    * ``"lm"``       -- ``mark_smooth(method="lm")`` (polynomial degree
      controlled by ``order``).
    * ``"loess"``    -- ``mark_smooth(method="loess")``.
    * ``"logistic"`` -- ``Logistic`` transform + ``mark_line``.
    * ``"glm"``      -- ``Glm`` transform + ``mark_line``.
    * ``"robust"``   -- ``Robust`` transform + ``mark_line``.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str
        Column name for the horizontal (predictor) axis (required).
    y : str
        Column name for the vertical (response) axis (required).
    hue : str or encoding, optional
        Column name to map to color; fit lines are drawn per hue level.
    col : str, optional
        Column name for faceting across columns.
    row : str, optional
        Column name for faceting across rows.
    method : {"lm", "logistic", "glm", "loess", "robust"}, default "lm"
        Fitting method.
    ci : int or None, default 95
        Confidence interval level (0--100) shown as a band around the fit
        line.  Pass ``None`` to suppress.
    order : int, default 1
        Polynomial degree forwarded to ``mark_smooth`` when
        ``method="lm"``.
    scatter : bool, default True
        Include a scatter layer (``mark_point``).  Set to ``False`` to
        show only the fit line.
    scatter_kws : dict, optional
        Extra keyword arguments forwarded to the scatter ``mark_point``
        call (e.g. ``{"opacity": 0.3, "size": 20}``).
    line_kws : dict, optional
        Extra keyword arguments forwarded to the regression-line mark
        call (e.g. ``{"stroke_width": 3}``).
    truncate : bool, default True
        When ``True`` (default), the fit line spans only the observed
        data range (min to max of ``x``).  When ``False``, raises
        ``ValueError`` because extending the fit line beyond the data
        range requires Rust-side ``x_range`` support (tracked in the
        design spec WI-7).
    x_bins : any, optional
        Forwarded as ``x_bins`` to ``mark_smooth`` for binning the
        x-axis before fitting (``method="lm"`` only).
    x_estimator : any, optional
        Forwarded as ``x_estimator`` to ``mark_smooth`` (``method="lm"``
        only).
    x_jitter : float or None, optional
        When set, applies ``Jitter(axis="x", width=x_jitter)`` to the
        scatter layer.
    logx : bool, default False
        Apply a ``log`` scale to the x-axis on both scatter and fit layers.
    show_metrics : bool, default True
        Schwabish SB-followup (2026-05-12): overlay a top-right corner
        annotation with ``R^2`` / ``RMSE`` / ``MAE`` computed from the
        OLS fit (``method="lm"``, no ``hue``). Silently skipped for
        non-LM methods (loess, robust, logistic, glm -- different metric
        space) or when ``hue`` is set (per-group corners would crowd).
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()``.

    Returns
    -------
    Chart
        Layered chart (scatter + fit) or fit-only when ``scatter=False``.
        May be faceted.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported values.
    ValueError
        If ``truncate=False`` (extending the fit line beyond the data range
        is not yet supported).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.lmplot(df, x="total_bill", y="tip")

    Logistic regression with per-sex fit lines:

    >>> fm.lmplot(df, x="total_bill", y="smoker_int", method="logistic", hue="sex")

    Polynomial fit (degree 2) with no confidence band:

    >>> fm.lmplot(df, x="size", y="tip", order=2, ci=None)
    """
    validate_choice("lmplot", "method", method, _VALID_METHODS)

    # Rust Smooth/Robust transforms require Float64 columns. Auto-cast
    # integer columns to Float64, matching catplot's pattern (chart.py).
    _INT_DTYPES = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )
    data = _to_polars(data)
    casts = []
    for fld in (x, y):
        col_name = _field_name(fld)
        if col_name in data.columns and data[col_name].dtype in _INT_DTYPES:
            casts.append(pl.col(col_name).cast(pl.Float64))
    if casts:
        data = data.with_columns(casts)

    # Normalize CI: spec accepts ci=95 (percent) or ci=None.
    ci_frac = (ci / 100.0) if ci is not None else None

    # Schwabish SB-followup (2026-05-12): the corner-metrics overlay is
    # supplied by the Smooth Rust transform via ``inject_metrics=True``
    # (threaded through ``mark_smooth(show_metrics=True)``), so the
    # OLS R²/RMSE/MAE is computed once in Rust and surfaced as
    # ``_metrics_text`` / ``_metrics_y`` columns on the fit-grid
    # output. The text layer reads the same Smooth-named source as
    # the ribbon + line layers — no Python-side OLS duplication.
    # Restricted to LM-without-hue: per-group corners would crowd, and
    # the metric is well-defined only for the OLS path.
    metrics_applied = show_metrics and method == "lm" and hue is None

    # truncate: True = clip to data range (current Smooth default).
    # False = extend fit line to the full x-axis domain boundary via
    # SmoothSpec/RobustSpec.x_range (spec §4 WI-7 — now wired end-to-end).
    # The x_range is resolved from the chart's x column min/max at Python
    # desugar time; the Rust transform uses it as the evaluation domain.
    # When truncate=True (default) x_range stays None → clips to observed data.
    x_range = None
    if not truncate:
        # Compute the axis-domain extent here so lmplot does not need a
        # provisional scale pass. The Rust Smooth/Robust transform will
        # evaluate the fit line over [x_min, x_max] of the full dataset
        # regardless of which rows each facet panel sees.
        # seaborn behaviour: extend to the axis limits, which are the data
        # range padded by 5% on each side (matching the typical auto-scale margin).
        try:
            _df = data.collect() if hasattr(data, "collect") else data
            x_col_name = _field_name(x)
            if hasattr(_df, "to_arrow"):
                _x_series = _df[x_col_name]
                _x_min = float(_x_series.min())
                _x_max = float(_x_series.max())
                _margin = (_x_max - _x_min) * 0.05
                x_range = [_x_min - _margin, _x_max + _margin]
        except (AttributeError, KeyError):
            x_range = None

    # Shared encoding.
    enc: dict = {"x": x, "y": y}
    if hue is not None:
        enc["color"] = hue
    enc.update(encode_kwargs)

    # ---- Scatter layer (optional) --------------------------------------
    scatter_layer = None
    if scatter:
        s = Chart(data)
        skw = dict(scatter_kws) if scatter_kws else {}
        if x_jitter is not None:
            s = s.mark_point(position=Jitter(axis="x", width=float(x_jitter)), **skw)
        else:
            s = s.mark_point(**skw)
        s = s.encode(**enc)
        scatter_layer = s

    # ---- Fit layer (per method) ----------------------------------------
    lkw = dict(line_kws) if line_kws else {}

    # When hue is set and the method is smooth-based (lm, loess), use
    # Smooth's groupby support to fit per-group smoothing in one pass.
    if hue is not None and method in ("lm", "loess"):
        hue_col = _field_name(hue)
        if method == "lm":
            fit = (
                Chart(data)
                .mark_smooth(
                    method="lm",
                    ci=ci_frac,
                    degree=order,
                    x_bins=x_bins,
                    x_estimator=x_estimator,
                    groupby=hue_col,
                    **({"x_range": x_range} if x_range is not None else {}),
                )
                .encode(x=x, y=y, color=hue)
            )
        else:
            fit = (
                Chart(data)
                .mark_smooth(
                    method="loess",
                    ci=ci_frac,
                    groupby=hue_col,
                    **({"x_range": x_range} if x_range is not None else {}),
                )
                .encode(x=x, y=y, color=hue)
            )
    elif method == "lm":
        fit = (
            Chart(data)
            .mark_smooth(
                method="lm",
                ci=ci_frac,
                degree=order,
                x_bins=x_bins,
                x_estimator=x_estimator,
                show_metrics=metrics_applied,
                **({"x_range": x_range} if x_range is not None else {}),
            )
            .encode(x=x, y=y)
        )
    elif method == "loess":
        fit = (
            Chart(data)
            .mark_smooth(
                method="loess",
                ci=ci_frac,
                **({"x_range": x_range} if x_range is not None else {}),
            )
            .encode(x=x, y=y)
        )
    elif method == "logistic":
        fit = (
            Chart(data)
            .transform(Logistic(x=x, y=y, n_grid=200, ci=ci_frac, name="logistic"))
            .mark_line(**lkw)
            .encode(x=x, y="fitted")
        )
    elif method == "glm":
        fit = (
            Chart(data)
            .transform(Glm(x=x, y=y, family="gaussian", n_grid=200, ci=ci_frac, name="glm"))
            .mark_line(**lkw)
            .encode(x=x, y="fitted")
        )
    elif method == "robust":
        fit = (
            Chart(data)
            .transform(Robust(x=x, y=y, n_grid=200, ci=ci_frac, name="robust"))
            .mark_line(**lkw)
            .encode(x=x, y="fitted")
        )
    else:  # pragma: no cover — guarded above
        raise ValueError(f"unreachable: method={method!r}")

    if hue is not None and method not in ("lm", "loess"):
        # Ensure fit also carries color encoding so per-group fits render
        # with the same palette as scatter.
        fit = fit.encode(color=hue)

    # logx → log scale on x.
    if logx:
        fit = fit.encode(x=X(x, scale={"type": "log"}))
        if scatter_layer is not None:
            scatter_layer = scatter_layer.encode(x=X(x, scale={"type": "log"}))

    # Compose scatter + fit. (When ``metrics_applied``, the fit chart
    # already carries the metrics-text layer reading the Smooth output.)
    if scatter_layer is None:
        out = fit
    else:
        out = _merge_layers(scatter_layer, fit, scatter_name="scatter", fit_name="fit")

    # Apply line_kws to smooth-based methods (lm, loess) where line_kws
    # can't be passed through mark_smooth directly. Non-smooth methods
    # (logistic, glm, robust) already applied lkw at mark_line construction.
    if lkw and method in ("lm", "loess") and out._layers:
        from dataclasses import replace as _dc_replace

        out._layers = [
            _dc_replace(layer, mark_kwargs={**(layer.mark_kwargs or {}), **lkw})
            if getattr(layer, "mark", None) == "line"
            else layer
            for layer in out._layers
        ]

    # Faceting.
    if col is not None or row is not None:
        if col is not None and row is not None:
            out = out.facet(row=row, col=col)
        elif col is not None:
            out = out.facet(col=col)
        else:
            out = out.facet(row=row)

    return _finalize_chart(
        out, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def residplot(
    data: Any,
    *,
    x: str,
    y: str,
    lowess: bool = False,
    order: int = 1,
    robust: bool = False,
    dropna: bool = True,
    show_metrics: bool = True,
    zero_line: bool = True,
    label: Any = None,
    color: Any = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Residual-diagnostic scatter plot.

    Computes regression residuals via ``Smooth(output="residuals")`` (or
    ``Robust(output="residuals")`` when ``robust=True``) and plots
    ``(x, residual)`` with ``mark_point``. When ``lowess=True``, a
    ``mark_line`` lowess smoother is layered over the residuals to help
    diagnose non-linearity.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str
        Column name for the horizontal axis (predictor; required).
    y : str
        Column name used to compute residuals (response; required).
    lowess : bool, default False
        Overlay a lowess smoother on the residuals using
        ``Smooth(method="loess")``.
    order : int, default 1
        Polynomial degree of the regression used to compute residuals.
    robust : bool, default False
        Use ``Robust`` regression (MM-estimator) instead of OLS when
        computing residuals. Compatible with ``show_metrics=True`` and
        ``zero_line=True`` (annotations flow through the Robust
        transform's same opt-in kwargs).
    dropna : bool, default True
        Drop rows where ``x`` or ``y`` is null before fitting.
    show_metrics : bool, default True
        Schwabish SB-followup (2026-05-12): overlay a top-right corner
        annotation with ``R^2`` / ``RMSE`` / ``MAE`` computed inside the
        Rust Smooth/Robust transform via the ``inject_metrics=True``
        kwarg -- same single execution model as fitted residuals, no
        Python-side regression duplication.
    zero_line : bool, default True
        Schwabish SB-followup: draw a dashed horizontal reference at
        ``y=0`` via the Rust transform's ``inject_zero_ref=True`` opt-in.
    label : str, optional
        Legend label for the residual series.  When set, a constant
        ``_label`` column is injected and mapped to color, producing a
        single-entry legend.
    color : str or encoding, optional
        Column name or constant color forwarded to ``Chart.encode(color=)``.
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()``.

    Returns
    -------
    Chart
        Scatter of residuals (possibly with a lowess layer, zero
        reference line, and corner R^2/RMSE/MAE annotation).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.residplot(df, x="total_bill", y="tip")

    Robust residuals with annotations:

    >>> fm.residplot(df, x="size", y="tip", robust=True)
    """
    data = _to_polars(data)
    if dropna:
        data = data.drop_nulls(subset=[x, y])

    # Rust Smooth/Robust transforms require Float64 columns.  Auto-cast
    # integer columns, matching lmplot and catplot.
    _INT_DTYPES = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )
    casts = []
    for fld in (x, y):
        if fld in data.columns and data[fld].dtype in _INT_DTYPES:
            casts.append(pl.col(fld).cast(pl.Float64))
    if casts:
        data = data.with_columns(casts)

    # The Rust Smooth/Robust transforms inject ``_ref_zero``,
    # ``_metrics_text``, and ``_metrics_y`` columns when the corresponding
    # ``inject_*`` kwargs are set — single source of truth for residual
    # computation lives in Rust; Python only declares the spec.
    # When ``label`` is set, inject a constant ``_label`` column into the data
    # and use ``groupby="_label"`` on the Smooth transform so the column
    # survives the Rust transform's output schema (which otherwise only emits
    # its declared columns).  Robust does not support groupby, so the label
    # column cannot be preserved through that path — a Rust-side ``groupby``
    # addition to the Robust transform would be needed.
    if label is not None:
        data = data.with_columns(pl.lit(str(label)).alias("_label"))

    if robust:
        resid_transform = Robust(
            x=x,
            y=y,
            output="residuals",
            inject_zero_ref=zero_line,
            inject_metrics=show_metrics,
        )
    else:
        smooth_kwargs: dict = dict(
            method="lm",
            degree=order,
            ci=None,
            output="residuals",
            inject_zero_ref=zero_line,
            inject_metrics=show_metrics,
        )
        if label is not None:
            smooth_kwargs["groupby"] = "_label"
        resid_transform = Smooth(x=x, y=y, **smooth_kwargs)

    chart = Chart(data).transform(resid_transform).mark_point()
    enc: dict = {"x": "x", "y": "residual"}
    if label is not None and color is None and not robust:
        enc["color"] = "_label"
    if color is not None:
        enc["color"] = color
    enc.update(encode_kwargs)
    chart = chart.encode(**enc)

    # Build the layered chart when any overlay is requested (zero rule,
    # metrics text, or lowess smoother). The augmented-DataFrame pattern
    # works because all overlays reference columns the Rust transform
    # already emitted into the same residuals batch.
    layered = lowess or zero_line or show_metrics
    if layered:
        from ferrum._layer import _Layer

        chart = chart._clone()
        internal_layers: list = [_Layer(name="point", mark="point", encoding=dict(enc))]
        if zero_line:
            internal_layers.append(
                _Layer(
                    name="reference",
                    mark="rule",
                    encoding={"y": "_ref_zero"},
                    mark_kwargs={"stroke_dash": [3, 3], "stroke": "#8a8a8a"},
                )
            )
        if show_metrics:
            internal_layers.append(
                _Layer(
                    name="metrics",
                    mark="text",
                    encoding={"x": "x", "y": "_metrics_y", "text": "_metrics_text"},
                    mark_kwargs={"align": "right", "dx": -4, "dy": 4},
                )
            )
        if lowess:
            chart._transforms = [
                resid_transform,
                Smooth(
                    x="x",
                    y="residual",
                    method="loess",
                    ci=None,
                    name="lowess",
                ),
            ]
            internal_layers.append(
                _Layer(
                    name="lowess",
                    mark="line",
                    encoding={"x": "x", "y": "y"},
                    data_source="lowess",
                )
            )
        chart._layers = internal_layers
        chart._mark = None

    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def regplot(
    data: Any,
    *,
    x: str,
    y: str,
    hue: Any = None,
    method: str = "lm",
    ci: Any = 95,
    order: int = 1,
    scatter: bool = True,
    scatter_kws: Any = None,
    line_kws: Any = None,
    truncate: bool = True,
    x_jitter: Any = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Axes-level regression scatter plot.

    The axes-level equivalent of ``lmplot`` — identical API except
    ``col=`` and ``row=`` are excluded because ``regplot`` does not
    facet.  Every parameter is forwarded to ``lmplot`` unchanged.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str
        Column name for the horizontal (predictor) axis (required).
    y : str
        Column name for the vertical (response) axis (required).
    hue : str or encoding, optional
        Column name to map to color; fit lines are drawn per hue level.
    method : {"lm", "logistic", "glm", "loess", "robust"}, default "lm"
        Fitting method forwarded to ``lmplot``.
    ci : int or None, default 95
        Confidence interval level (0--100) shown as a band around the fit
        line.  Pass ``None`` to suppress.
    order : int, default 1
        Polynomial degree forwarded to ``mark_smooth`` when
        ``method="lm"``.
    scatter : bool, default True
        Include a scatter layer (``mark_point``).
    scatter_kws : dict, optional
        Extra keyword arguments forwarded to the scatter ``mark_point`` call.
    line_kws : dict, optional
        Extra keyword arguments forwarded to the regression-line mark call.
    truncate : bool, default True
        When ``True``, the fit line spans only the observed data range.
    x_jitter : float or None, optional
        When set, applies ``Jitter(axis="x", width=x_jitter)`` to the
        scatter layer.
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()``.

    Returns
    -------
    Chart
        Layered chart (scatter + fit) or fit-only when ``scatter=False``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported values.
    ValueError
        If ``truncate=False`` (extending the fit line beyond the data range
        is not yet supported).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.regplot(df, x="total_bill", y="tip")

    Robust regression:

    >>> fm.regplot(df, x="size", y="tip", method="robust")
    """
    return lmplot(
        data,
        x=x,
        y=y,
        hue=hue,
        method=method,
        ci=ci,
        order=order,
        scatter=scatter,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        truncate=truncate,
        x_jitter=x_jitter,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
        **encode_kwargs,
    )


def residuals_chart(
    model: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    kind: str = "studentized",
    cook_threshold: float | str | None = None,
    panels: Any = "auto",
    annotate_metrics: bool = True,
    subtitle: str | None = None,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart | HConcatChart | VConcatChart | ConcatChart":
    """Residuals diagnostic chart for a regression estimator.

    Plots residuals vs. fitted values. Optional Cook's distance
    highlighting marks observations whose leverage-adjusted influence
    exceeds a user-supplied threshold. ``panels="auto"`` returns the
    canonical 4-panel diagnostic layout (residuals-vs-fitted, QQ,
    scale-location, residuals-vs-leverage) as a 2x2 grid.

    Parameters
    ----------
    model : estimator or ModelSource
        A fitted sklearn-compatible regression estimator, an explicit
        ``ferrum.ModelSource``, or a dict of named estimators.
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y : array-like, optional
        Target vector. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y_true : array-like, optional
        Ground-truth target values for the precomputed path.  Must be
        paired with ``y_pred``; mutually exclusive with ``model``.
    y_pred : array-like, optional
        Predicted target values for the precomputed path.
    kind : {"studentized", "raw"}, default "studentized"
        Residual type to plot on the y axis. ``"studentized"`` uses
        internally studentized residuals; ``"raw"`` uses raw residuals.
    cook_threshold : float, "auto", or None, default None
        Threshold for Cook's distance outlier highlighting. A float is
        used as an absolute cutoff; ``"auto"`` applies the ``4 / n``
        rule (Hair et al.); ``None`` disables highlighting. Cook's
        distance is only defined for estimators that expose ``coef_``;
        non-linear models surface ``NaN`` and produce no outliers.
    panels : "auto", None, "single", or list of str, default "auto"
        Panel selection. ``"auto"`` ships the canonical 4-panel layout
        -- ``residuals_vs_fitted``, ``qq``, ``scale_location``, and
        ``residuals_vs_leverage`` -- laid out as a 2x2 grid. ``None`` or
        ``"single"`` returns just the residuals-vs-fitted panel. Pass an
        explicit list such as ``["residuals_vs_fitted", "qq"]`` to
        customize the panel set.
    annotate_metrics : bool, default True
        Overlay a top-right corner annotation showing R^2/RMSE/MAE
        computed from the fit. Pass ``False`` to render the residual
        scatter alone.
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
    compare : dict of str -> estimator or None, default None
        Mapping of label -> fitted estimator to compare against ``model``.
        Behavior depends on ``panels``:

        - ``panels="single"`` (or ``None``): the residuals-vs-fitted scatter
          carries a ``model`` colour group with one overlaid series per model
          (unchanged single-panel overlay).
        - any multi-panel value (``"auto"`` or an explicit list): returns a
          [ConcatChart][ferrum.ConcatChart] of small multiples, one panel per model.
          Each model renders its full multi-panel diagnostic grid as one
          small-multiple panel, labeled with the model name. Scale sharing
          across models is not applied to the multi-panel grid (each panel
          carries its own faceted scales). The per-model hat-matrix leverage
          and Cook's-distance computations are correct by construction because
          each panel sees one model's data only.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``; does not affect deterministic
        residuals computation.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.

    Returns
    -------
    Chart or composite
        Residuals-vs-fitted scatter chart with optional Cook's-distance
        outlier overlay, the multi-panel diagnostic grid, or a
        small-multiples ``ConcatChart`` (one panel per model) when
        ``compare=`` is supplied with a multi-panel ``panels`` value.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import Ridge
    >>> fm.residuals_chart(Ridge().fit(X_train, y_train), X_test, y_test)

    Precomputed path — residuals are computed as ``y_true − y_pred`` internally.
    Leverage and Cook's distance are unavailable, so the leverage panel is
    omitted when ``panels="auto"``:

    >>> fm.residuals_chart(y_true=y_test, y_pred=reg.predict(X_test))
    """
    source = _resolve_source(
        model, X, y, y_true=y_true, y_pred=y_pred, random_state=random_state, compare=compare
    )
    if panels in (None, "single"):
        panel_list: Any = None
    elif panels == "auto":
        panel_list = [
            "residuals_vs_fitted",
            "qq",
            "scale_location",
            "residuals_vs_leverage",
        ]
    else:
        panel_list = list(panels)
    # Both the compose path and the single-model fall-through pass identical
    # kwargs to the same builder (``panels=panel_list`` in both); only the
    # routing condition differs. Spell the kwargs once and route both paths
    # through it.
    builder_kwargs = dict(
        kind=kind,
        cook_threshold=cook_threshold,
        panels=panel_list,
        annotate_metrics=annotate_metrics,
        subtitle=subtitle,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )
    # Multi-panel comparison renders small multiples (one full diagnostic grid
    # per model) so each model's leverage / Cook's-distance computations stay
    # single-model. The single-panel residuals-vs-fitted view keeps its
    # ``model`` colour-group overlay (``panel_list is None``), so only the
    # multi-panel + compare case routes through ``_compose_compare``.
    if isinstance(source, ComparedModelSource) and panel_list is not None:
        return _compose_compare(
            source,
            _residuals_chart_from_source,
            builder_kwargs=builder_kwargs,
            resolve={"x": "shared", "y": "shared"},
        )
    return _residuals_chart_from_source(source, **builder_kwargs)


def prediction_error_chart(
    model: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    reference_line: bool = True,
    ci: float | None = None,
    reference_band: bool = False,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart | ConcatChart":
    """Actual-vs-predicted scatter for a regression estimator.

    Plots ``y_true`` on the y axis against ``y_pred`` on the x axis with
    an optional reference line (``y = x`` diagonal) and an optional
    residual-based confidence ribbon.

    Parameters
    ----------
    model : estimator or ModelSource
        A fitted sklearn-compatible regression estimator, an explicit
        ``ferrum.ModelSource``, or ``None`` when pre-computed arrays are
        supplied via ``y_true`` / ``y_pred``.
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y : array-like, optional
        Target vector. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y_true : array-like, optional
        Pre-computed true labels. Use with ``y_pred`` to bypass model
        inference entirely.
    y_pred : array-like, optional
        Pre-computed predictions. Use with ``y_true`` to bypass model
        inference entirely.
    reference_line : bool, default True
        Overlay the dashed ``y = x`` diagonal.
    ci : float or None, default None
        Confidence level in ``(0, 1)``. When set, overlays a ribbon
        spanning the central ``ci`` fraction of residuals around the
        reference line. Raises ``ValueError`` if not in ``(0, 1)``.
    reference_band : bool, default False
        When ``True`` (and ``ci`` is ``None``), overlays a ±1 RMSE
        ribbon around the reference line.
    compare : dict of str -> estimator or None, default None
        Mapping of label -> fitted estimator to compare against ``model``.
        Behavior depends on ``ci`` / ``reference_band``:

        - default scatter path (no band): the chart carries a ``model`` colour
          group with one actual-vs-predicted series per model and a shared y=x
          reference line (unchanged single-panel overlay).
        - with ``ci`` or ``reference_band``: returns a
          [ConcatChart][ferrum.ConcatChart] of small multiples, one panel per model,
          with shared x/y scales. Each panel's residual band is computed from
          that model's residuals only — never pooled across models.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.

    Returns
    -------
    Chart or ConcatChart
        Actual-vs-predicted scatter with optional reference line and
        confidence ribbon, or a small-multiples ``ConcatChart`` (one panel per
        model) when ``compare=`` is combined with ``ci`` / ``reference_band``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.prediction_error_chart(model, X_test, y_test, reference_line=True)
    """
    source = _resolve_source(
        model,
        X,
        y,
        y_true=y_true,
        y_pred=y_pred,
        random_state=random_state,
        compare=compare,
    )
    # Both the compose path and the single-model fall-through pass identical
    # kwargs to the same builder; only the routing condition differs.
    builder_kwargs = dict(
        reference_line=reference_line,
        ci=ci,
        reference_band=reference_band,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )
    # The residual confidence band (``ci`` / ``reference_band``) is a
    # single-model aggregate. Composing one panel per model gives the builder a
    # single-model source for each panel, so every band is computed from that
    # model's residuals only — never pooled across models. The default scatter
    # path (no band) keeps its single-panel ``model`` colour-group overlay.
    if isinstance(source, ComparedModelSource) and (ci is not None or reference_band):
        return _compose_compare(
            source,
            _prediction_error_chart_from_source,
            builder_kwargs=builder_kwargs,
            resolve={"x": "shared", "y": "shared"},
        )
    return _prediction_error_chart_from_source(source, **builder_kwargs)


def cooks_distance_chart(
    model: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    threshold: float | str | None = None,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart | ConcatChart":
    """Cook's distance / residuals-vs-leverage diagnostic for a linear estimator.

    Renders the residuals-vs-leverage panel with optional Cook's-distance
    outlier highlighting. Useful for identifying influential observations
    in a linear regression fit.

    Parameters
    ----------
    model : estimator or ModelSource
        A fitted sklearn-compatible regression estimator that exposes
        ``coef_`` (required for leverage-aware Cook's distance), or an
        explicit ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y : array-like, optional
        Target vector. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    threshold : float, "auto", or None, default None
        Cook's-distance threshold for outlier highlighting. A float is
        used as an absolute cutoff; ``"auto"`` applies the ``4 / n``
        rule (Hair et al.); ``None`` disables highlighting.
    compare : dict of str -> estimator or None, default None
        Mapping of label -> fitted estimator to compare against ``model``.
        When supplied, returns a [ConcatChart][ferrum.ConcatChart] of small
        multiples, one residuals-vs-leverage panel per model, with shared x/y
        scales. Each panel's Cook's distance and leverage are derived from that
        model's own hat matrix only. The single-model path (no ``compare=``) is
        unchanged.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.

    Returns
    -------
    Chart or ConcatChart
        Residuals-vs-leverage panel with optional Cook's-distance outlier
        overlay, or a small-multiples ``ConcatChart`` (one panel per model)
        when ``compare=`` is supplied.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.cooks_distance_chart(linear_model, X_test, y_test, threshold="auto")
    """
    source = _resolve_source(
        model,
        X,
        y,
        random_state=random_state,
        compare=compare,
    )
    builder_kwargs = dict(
        kind="studentized",
        cook_threshold=threshold,
        panels=["residuals_vs_leverage"],
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )
    if isinstance(source, ComparedModelSource):
        return _compose_compare(
            source,
            _residuals_chart_from_source,
            builder_kwargs=builder_kwargs,
            resolve={"x": "shared", "y": "shared"},
        )
    return _residuals_chart_from_source(source, **builder_kwargs)
