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

    from ferrum._layer import _Layer

    # Build scatter layer.
    scatter_layer = _Layer(
        mark=s_resolved._mark,
        encoding=dict(s_resolved._encoding),
        mark_kwargs=dict(s_resolved._mark_kwargs) if s_resolved._mark_kwargs else None,
        position=s_resolved._position,
    )

    # Collect fit layers (may be single-mark or multi-layer).
    if f_resolved._layers is not None:
        fit_layers = list(f_resolved._layers)
    else:
        fit_layers = [_Layer(
            mark=f_resolved._mark,
            encoding=dict(f_resolved._encoding),
            mark_kwargs=dict(f_resolved._mark_kwargs) if f_resolved._mark_kwargs else None,
            position=f_resolved._position,
        )]

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
    show_metrics: bool = True, zero_line: bool = True,
    label: Any = None, color: Any = None, theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Residual-diagnostic scatter plot.

    Computes regression residuals and plots ``(x, residual)`` with
    ``mark_point``. When ``lowess=True``, a ``mark_line`` lowess
    smoother is layered over the residuals to help diagnose
    non-linearity.

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
        computing residuals. Currently incompatible with
        ``show_metrics=True`` / ``zero_line=True``; pass both False to
        keep the legacy transform path, or switch to
        :func:`residuals_chart` for the model-diagnostics surface.
    dropna : bool, default True
        Reserved for future NaN-dropping logic (no-op today).
    show_metrics : bool, default True
        Schwabish SB-followup (2026-05-12): overlay a top-right corner
        annotation with ``R²`` / ``RMSE`` / ``MAE`` computed from the
        underlying ``(x, y)`` regression. Bypasses the Smooth-transform
        pipeline (residuals computed Python-side via ``np.polyfit``).
        Set ``False`` to keep the legacy transform-based path.
    zero_line : bool, default True
        Schwabish SB-followup: draw a dashed horizontal reference at
        ``y=0`` so deviations are immediately visible.
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
        Scatter of residuals (possibly with a lowess layer, zero
        reference line, and corner R²/RMSE/MAE annotation).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.residplot(df, x="total_bill", y="tip")

    Robust residuals (legacy path; no annotations):

    >>> fm.residplot(df, x="size", y="tip", robust=True,
    ...              show_metrics=False, zero_line=False)
    """
    del dropna, label  # reserved kwargs — not yet honored

    annotated = show_metrics or zero_line
    if annotated and robust:
        raise NotImplementedError(
            "residplot(show_metrics=True | zero_line=True) currently "
            "requires robust=False (OLS via np.polyfit). For robust "
            "residual diagnostics, either pass show_metrics=False, "
            "zero_line=False (legacy Robust-transform path) or use "
            "ferrum.residuals_chart() on a fitted estimator."
        )

    if annotated:
        return _residplot_annotated(
            data, x=x, y=y, lowess=lowess, order=order,
            show_metrics=show_metrics, zero_line=zero_line,
            color=color, theme=theme, **encode_kwargs,
        )

    # Legacy transform path — preserved byte-identical for callers that
    # opt out of annotations or need the Robust-transform residuals.
    if robust:
        resid_transform = Robust(x=x, y=y, output="residuals")
    else:
        resid_transform = Smooth(
            x=x, y=y, method="lm", degree=order, ci=None,
            output="residuals",
        )

    chart = Chart(data).transform(resid_transform).mark_point()
    enc: dict = {"x": "x", "y": "residual"}
    if color is not None:
        enc["color"] = color
    enc.update(encode_kwargs)
    chart = chart.encode(**enc)

    if lowess:
        loess_transform = Smooth(
            x="x", y="residual", method="loess",
            ci=None, name="lowess",
        )
        chart = chart._clone()
        chart._transforms = [resid_transform, loess_transform]
        from ferrum._layer import _Layer
        chart._layers = [
            _Layer(mark="point", encoding=dict(enc)),
            _Layer(
                mark="line",
                encoding={"x": "x", "y": "y"},
                data_source="lowess",
            ),
        ]
        chart._mark = None

    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _residplot_annotated(
    data: Any, *, x: str, y: str,
    lowess: bool, order: int,
    show_metrics: bool, zero_line: bool,
    color: Any, theme: Any,
    **encode_kwargs: Any,
) -> Chart:
    """Schwabish-annotated residplot path.

    Bypasses the Smooth/Robust transforms — computes residuals in Python
    via ``np.polyfit`` so the augmented-DataFrame pattern (same-data
    overlays for ``_ref_zero`` and ``_metrics_text``) works the same way
    it does in ``_residuals_chart_from_source``.
    """
    import numpy as np
    import polars as pl

    from ferrum._coerce import to_arrow_table
    from ferrum._layer import _Layer

    tbl = to_arrow_table(data)
    if x not in tbl.column_names or y not in tbl.column_names:
        raise ValueError(
            f"residplot(x={x!r}, y={y!r}): both columns must exist on the input."
        )
    x_arr = np.asarray(tbl.column(x).to_pylist(), dtype=float)
    y_arr = np.asarray(tbl.column(y).to_pylist(), dtype=float)
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]
    if x_arr.size < order + 1:
        raise ValueError(
            f"residplot needs at least order+1={order + 1} finite points; "
            f"got {x_arr.size}."
        )

    coeffs = np.polyfit(x_arr, y_arr, deg=order)
    y_pred = np.polyval(coeffs, x_arr)
    resid_arr = y_arr - y_pred

    n = x_arr.size
    df_cols: dict[str, Any] = {
        "x": pl.Series("x", x_arr, dtype=pl.Float64),
        "residual": pl.Series("residual", resid_arr, dtype=pl.Float64),
    }
    # Carry over any extra columns the user might encode against
    # (e.g. ``color="species"``).
    for col in tbl.column_names:
        if col in (x, y, "x", "residual"):
            continue
        try:
            df_cols[col] = pl.Series(col, tbl.column(col).to_pylist())
        except Exception:
            pass

    df = pl.DataFrame(df_cols)

    if zero_line:
        zero_col: list = [0.0] + [None] * (n - 1)
        df = df.with_columns(pl.Series("_ref_zero", zero_col, dtype=pl.Float64))

    if show_metrics:
        ss_res = float(np.sum(resid_arr ** 2))
        ss_tot = float(np.sum((y_arr - float(np.mean(y_arr))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean(resid_arr ** 2)))
        mae = float(np.mean(np.abs(resid_arr)))
        anchor_idx = int(np.argmax(x_arr))
        text_col: list = [None] * n
        text_col[anchor_idx] = f"R² {r2:.3f}\nRMSE {rmse:.3f}\nMAE {mae:.3f}"
        metrics_y_col: list = [None] * n
        metrics_y_col[anchor_idx] = float(np.max(resid_arr))
        df = df.with_columns(
            pl.Series("_metrics_text", text_col, dtype=pl.Utf8),
            pl.Series("_metrics_y", metrics_y_col, dtype=pl.Float64),
        )

    chart = Chart(df).mark_point()
    enc: dict = {"x": "x", "y": "residual"}
    if color is not None:
        enc["color"] = color
    enc.update(encode_kwargs)
    chart = chart.encode(**enc)

    layers: list = [_Layer(mark="point", encoding=dict(enc))]
    if zero_line:
        layers.append(_Layer(
            mark="rule",
            encoding={"y": "_ref_zero"},
            mark_kwargs={"stroke_dash": [3, 3], "stroke": "#8a8a8a"},
        ))
    if show_metrics:
        layers.append(_Layer(
            mark="text",
            encoding={"x": "x", "y": "_metrics_y", "text": "_metrics_text"},
            mark_kwargs={"align": "right", "dx": -4, "dy": 4},
        ))
    if lowess:
        # Run a fresh loess smoother on the residuals batch. The Smooth
        # transform is named so its output is available as a separate
        # data source; the chart's "default" data is the residuals df.
        chart = chart._clone()
        chart._transforms = [
            Smooth(x="x", y="residual", method="loess", ci=None, name="lowess"),
        ]
        layers.append(_Layer(
            mark="line",
            encoding={"x": "x", "y": "y"},
            data_source="lowess",
        ))

    chart = chart._clone()
    chart._data = df
    chart._layers = layers
    chart._mark = None

    if theme is not None:
        chart = chart.theme(theme)
    return chart
