"""Regression-plot convenience functions (lmplot, residplot)."""

from __future__ import annotations
from typing import Any

from ferrum import (
    Chart,
    Glm,
    Jitter,
    Logistic,
    Robust,
    Smooth,
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
        fit_layers = [
            _Layer(
                mark=f_resolved._mark,
                encoding=dict(f_resolved._encoding),
                mark_kwargs=dict(f_resolved._mark_kwargs) if f_resolved._mark_kwargs else None,
                position=f_resolved._position,
            )
        ]

    new._mark = None
    new._layers = [scatter_layer] + fit_layers
    new._transforms = shared_transforms
    return new


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
        Confidence interval level (0–100) shown as a band around the fit
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
        annotation with ``R²`` / ``RMSE`` / ``MAE`` computed from the
        OLS fit (``method="lm"``, no ``hue``). Silently skipped for
        non-LM methods (loess, robust, logistic, glm — different metric
        space) or when ``hue`` is set (per-group corners would crowd).
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
    if method not in _VALID_METHODS:
        raise ValueError(f"lmplot: method must be one of {sorted(_VALID_METHODS)}; got {method!r}")

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
    # False = extend beyond data range — requires Rust-side x_range
    # support (SmoothSpec.x_range, tracked in design spec WI-7).
    if not truncate:
        raise ValueError(
            "lmplot: truncate=False is not yet supported; the fit line always "
            "clips to the observed data range. Set truncate=True or omit the "
            "parameter. Rust-side x_range support is tracked in design spec WI-7."
        )

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
    if method == "lm":
        fit = (
            Chart(data)
            .mark_smooth(
                method="lm",
                ci=ci_frac,
                degree=order,
                x_bins=x_bins,
                x_estimator=x_estimator,
                show_metrics=metrics_applied,
            )
            .encode(x=x, y=y)
        )
    elif method == "loess":
        fit = Chart(data).mark_smooth(method="loess", ci=ci_frac).encode(x=x, y=y)
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

    # Compose scatter + fit. (When ``metrics_applied``, the fit chart
    # already carries the metrics-text layer reading the Smooth output.)
    if scatter_layer is None:
        out = fit
    else:
        out = _merge_layers(scatter_layer, fit)

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

    if theme is not None:
        out = out.theme(theme)

    return out


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
        annotation with ``R²`` / ``RMSE`` / ``MAE`` computed inside the
        Rust Smooth/Robust transform via the ``inject_metrics=True``
        kwarg — same single execution model as fitted residuals, no
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

    Robust residuals with annotations:

    >>> fm.residplot(df, x="size", y="tip", robust=True)
    """
    if dropna:
        import polars as pl

        data = pl.DataFrame(data) if not isinstance(data, pl.DataFrame) else data
        data = data.drop_nulls(subset=[x, y])

    # The Rust Smooth/Robust transforms inject ``_ref_zero``,
    # ``_metrics_text``, and ``_metrics_y`` columns when the corresponding
    # ``inject_*`` kwargs are set — single source of truth for residual
    # computation lives in Rust; Python only declares the spec.
    if robust:
        resid_transform = Robust(
            x=x,
            y=y,
            output="residuals",
            inject_zero_ref=zero_line,
            inject_metrics=show_metrics,
        )
    else:
        resid_transform = Smooth(
            x=x,
            y=y,
            method="lm",
            degree=order,
            ci=None,
            output="residuals",
            inject_zero_ref=zero_line,
            inject_metrics=show_metrics,
        )

    if label is not None:
        import polars as pl

        data = pl.DataFrame(data) if not isinstance(data, pl.DataFrame) else data
        data = data.with_columns(pl.lit(label).alias("_label"))

    chart = Chart(data).transform(resid_transform).mark_point()
    enc: dict = {"x": "x", "y": "residual"}
    if label is not None and color is None:
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
        layers: list = [_Layer(mark="point", encoding=dict(enc))]
        if zero_line:
            layers.append(
                _Layer(
                    mark="rule",
                    encoding={"y": "_ref_zero"},
                    mark_kwargs={"stroke_dash": [3, 3], "stroke": "#8a8a8a"},
                )
            )
        if show_metrics:
            layers.append(
                _Layer(
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
            layers.append(
                _Layer(
                    mark="line",
                    encoding={"x": "x", "y": "y"},
                    data_source="lowess",
                )
            )
        chart._layers = layers
        chart._mark = None

    if theme is not None:
        chart = chart.theme(theme)
    return chart
