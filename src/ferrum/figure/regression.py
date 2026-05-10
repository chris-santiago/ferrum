"""Phase 9e — lmplot and residplot."""
from __future__ import annotations
from typing import Any

from ferrum import (
    Chart, Glm, Jitter, Logistic, Robust, Smooth,
)


_VALID_METHODS = {"lm", "logistic", "glm", "loess", "robust"}


def _merge_layers(scatter_chart: Chart, fit_chart: Chart) -> Chart:
    """Compose a scatter Chart and a (possibly layered) fit Chart into one
    multi-layer Chart sharing the same data.

    Returns a new Chart with `_layers` = scatter-layer + fit-layers,
    transforms accumulated.
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
    """Regression figure-level function — see ferrum-spec.md §3.14.

    Builds a scatter + fit overlay using the appropriate Phase 9b transform:
    Smooth (lm/loess), Logistic, Glm, or Robust.
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
    """Residual-diagnostic figure-level function — see ferrum-spec.md §3.14.

    Builds a scatter plot of (x, residual) using ``Smooth(output='residuals')``
    (or ``Robust(output='residuals')`` when ``robust=True``); optionally
    layers a lowess smoother over the residuals.
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
    # NOTE: full implementation requires per-layer transform chaining so that
    # layer 1 sees the residuals output and layer 2 sees the loess output
    # without _merge_layers consolidating both Smooth transforms chart-level.
    # The current overlay path fails because chart-level chain advances past
    # the residuals output. Tracked as xfail in tests/test_phase_9_e2e.py.
    if lowess:
        lo = (
            Chart(data)
            .transform(resid_transform)
            .encode(x="x", y="residual")
            .mark_smooth(method="loess")
        )
        chart = _merge_layers(chart, lo)

    if theme is not None:
        chart = chart.theme(theme)
    return chart
