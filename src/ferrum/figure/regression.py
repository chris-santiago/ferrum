"""Regression-plot convenience functions (lmplot, residplot)."""
from __future__ import annotations
from typing import Any

from ferrum import (
    Chart, Glm, Jitter, Logistic, Robust, Smooth,
)


_VALID_METHODS = {"lm", "logistic", "glm", "loess", "robust"}


def _merge_layers(scatter_chart: Chart, fit_chart: Chart) -> Chart:
    """Compose a scatter Chart and a fit Chart into a multi-layer Chart.

    Returns a new Chart with ``_layers`` = scatter-layer + fit-layers,
    with transforms accumulated from both inputs.
    """
    s_resolved = scatter_chart._resolve_pending()
    f_resolved = fit_chart._resolve_pending()

    new = s_resolved._clone()
    new._pending_stat_mark = None

    # Collect top-level transforms shared by both charts. To avoid duplicate
    # transforms in the output (e.g. an Unpivot pipeline that both layers
    # reference), we dedupe by class+constructor-equality on best-effort.
    shared_transforms: list = []
    seen_ids = set()
    for t in list(s_resolved._transforms) + list(f_resolved._transforms):
        key = id(t)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        shared_transforms.append(t)

    # Build scatter layer dict.
    scatter_layer = {
        "mark": s_resolved._mark,
        "encoding": dict(s_resolved._encoding),
        "transforms": [],
        "mark_style": dict(s_resolved._mark_kwargs),
        "position": s_resolved._position,
    }

    # Collect fit layers (may be single-mark or multi-layer).
    if f_resolved._layers is not None:
        fit_layers = list(f_resolved._layers)
    else:
        fit_layers = [{
            "mark": f_resolved._mark,
            "encoding": dict(f_resolved._encoding),
            "transforms": [],
            "mark_style": dict(f_resolved._mark_kwargs),
            "position": f_resolved._position,
        }]

    new._mark = None
    new._layers = [scatter_layer] + fit_layers
    new._transforms = shared_transforms
    return new


def lmplot(
    data: Any, *, x: str, y: str,
    hue: Any = None, col: Any = None, row: Any = None,
    method: str = "lm",
    ci: Any = 95, order: int = 1,
    scatter: bool = True,
    scatter_kws: Any = None, line_kws: Any = None,
    truncate: bool = False,
    x_bins: Any = None, x_estimator: Any = None, x_jitter: Any = None,
    logx: bool = False, theme: Any = None,
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
        Confidence interval level (0–100) shown as a band around the fit
        line.  Pass ``None`` to suppress.
    order : int, default 1
        Polynomial degree forwarded to ``mark_smooth`` when
        ``method="lm"``.
    scatter : bool, default True
        Include a scatter layer (``mark_point``).  Set to ``False`` to
        show only the fit line.
    scatter_kws : dict, optional
        Reserved for future use (no-op today). When wired, will forward extra
        keyword arguments to the scatter ``mark_point`` call.
    line_kws : dict, optional
        Reserved for future use (no-op today). When wired, will forward extra
        keyword arguments to the regression-line mark call.
    truncate : bool, default False
        Reserved for future line-truncation support (no-op today; the
        fit line already extends to the data range via the Smooth grid).
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
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.lmplot(df, x="total_bill", y="tip")

    Logistic regression with per-sex fit lines:

    >>> fm.lmplot(df, x="total_bill", y="smoker_int", method="logistic", hue="sex")

    Polynomial fit (degree 2) with no confidence band:

    >>> fm.lmplot(df, x="size", y="tip", order=2, ci=None)
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"lmplot: method must be one of {sorted(_VALID_METHODS)}; got {method!r}"
        )

    # Normalize CI: spec accepts ci=95 (percent) or ci=None.
    ci_frac = (ci / 100.0) if ci is not None else None

    # Shared encoding.
    enc: dict = {"x": x, "y": y}
    if hue is not None:
        enc["color"] = hue
    enc.update(encode_kwargs)

    # ---- Scatter layer (optional) --------------------------------------
    scatter_layer = None
    if scatter:
        s = Chart(data)
        if x_jitter is not None:
            s = s.mark_point(position=Jitter(axis="x", width=float(x_jitter)))
        else:
            s = s.mark_point()
        s = s.encode(**enc)
        scatter_layer = s

    # ---- Fit layer (per method) ----------------------------------------
    if method == "lm":
        fit = Chart(data).mark_smooth(
            method="lm", ci=ci_frac, degree=order,
            x_bins=x_bins, x_estimator=x_estimator,
        ).encode(x=x, y=y)
    elif method == "loess":
        fit = Chart(data).mark_smooth(method="loess", ci=ci_frac).encode(x=x, y=y)
    elif method == "logistic":
        fit = Chart(data).transform(
            Logistic(x=x, y=y, n_grid=200, ci=ci_frac, name="logistic")
        ).mark_line().encode(x=x, y="fitted")
    elif method == "glm":
        fit = Chart(data).transform(
            Glm(x=x, y=y, family="gaussian", n_grid=200, ci=ci_frac, name="glm")
        ).mark_line().encode(x=x, y="fitted")
    elif method == "robust":
        fit = Chart(data).transform(
            Robust(x=x, y=y, n_grid=200, ci=ci_frac, name="robust")
        ).mark_line().encode(x=x, y="fitted")
    else:  # pragma: no cover — guarded above
        raise ValueError(f"unreachable: method={method!r}")

    if hue is not None:
        # Ensure fit also carries color encoding so per-group fits render
        # with the same palette as scatter.
        fit = fit.encode(color=hue)

    # logx → log scale on x.
    if logx:
        from ferrum.encoding import X
        fit = fit.encode(x=X(x, scale={"type": "log"}))
        if scatter_layer is not None:
            scatter_layer = scatter_layer.encode(x=X(x, scale={"type": "log"}))

    # Compose scatter + fit.
    if scatter_layer is None:
        out = fit
    else:
        out = _merge_layers(scatter_layer, fit)

    # truncate=True: deferred (line extends to data range by Smooth's grid).
    # Faceting.
    if col is not None or row is not None:
        if col is not None and row is not None:
            out = out.facet(row=row, col=col)
        elif col is not None:
            out = out.facet(col=col)
        else:
            out = out.facet(row=row)

    if theme is not None:
        out = out.theme(theme)

    return out


def residplot(
    data: Any, *, x: str, y: str,
    lowess: bool = False, order: int = 1,
    robust: bool = False, dropna: bool = True,
    label: Any = None, color: Any = None, theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Residual-diagnostic scatter plot.

    Computes regression residuals via ``Smooth(output="residuals")`` (or
    ``Robust(output="residuals")`` when ``robust=True``) and plots
    ``(x, residual)`` with ``mark_point``.  When ``lowess=True``, a
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
        Polynomial degree of the regression used to compute residuals
        (forwarded to ``Smooth`` when ``robust=False``).
    robust : bool, default False
        Use ``Robust`` regression (MM-estimator) instead of OLS when
        computing residuals.
    dropna : bool, default True
        Reserved for future NaN-dropping logic (no-op today).
    label : any, optional
        Reserved for future legend-label support (no-op today).
    color : str or encoding, optional
        Column name or constant color forwarded to ``Chart.encode(color=)``.
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()``.

    Returns
    -------
    Chart
        Scatter of residuals (possibly with a lowess layer).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.residplot(df, x="total_bill", y="tip")

    Robust residuals with a lowess overlay:

    >>> fm.residplot(df, x="size", y="tip", robust=True, lowess=True)
    """
    # Build the residuals transform — unnamed so it advances the chained
    # output. Smooth/Robust's residuals output schema is literal [x, residual]
    # (per design §5.3), so the downstream layers encode against "x" and
    # "residual" rather than the original column names.
    if robust:
        resid_transform = Robust(x=x, y=y, output="residuals")
    else:
        resid_transform = Smooth(
            x=x, y=y, method="lm", degree=order, ci=None,
            output="residuals",
        )

    # Base scatter against residuals.
    chart = Chart(data).transform(resid_transform).mark_point()
    enc: dict = {"x": "x", "y": "residual"}
    if color is not None:
        enc["color"] = color
    enc.update(encode_kwargs)
    chart = chart.encode(**enc)

    # Optional lowess smoother of the residuals (overlaid as a second layer).
    # Chart-level chain: [Smooth(residuals), Smooth(loess, name="lowess")].
    # First Smooth advances FINAL → residuals batch [x, residual].
    # Second Smooth is named, so it doesn't advance FINAL but publishes its
    # output ([x, y, ci_lower, ci_upper]) under "lowess". (After Phase 9's
    # named-on-chained-current semantics fix, the named Smooth runs on the
    # chained residuals batch, not the original data.)
    # Layer 0 (point) consumes FINAL = residuals → encoding y="residual" works.
    # Layer 1 (line) consumes data_source="lowess" → encoding y="y" works.
    if lowess:
        loess_transform = Smooth(
            x="x", y="residual", method="loess",
            ci=None, name="lowess",
        )
        # Build layered chart manually to control transform placement.
        chart = chart._clone()
        chart._transforms = [resid_transform, loess_transform]
        chart._layers = [
            {
                "mark": "point",
                "encoding": dict(enc),  # {"x": "x", "y": "residual", maybe color}
            },
            {
                "mark": "line",
                "encoding": {"x": "x", "y": "y"},
                "data_source": "lowess",
            },
        ]
        chart._mark = None  # signals layered mode in to_spec

    if theme is not None:
        chart = chart.theme(theme)
    return chart
