"""Distribution convenience functions (displot)."""

from __future__ import annotations
from typing import Any

from ferrum import Bin, Chart, Identity, Dodge, Stack
from ferrum._overrides import _apply_overrides


_VALID_KINDS = {"hist", "kde", "ecdf", "rug"}
_VALID_MULTIPLE = {"layer", "stack", "fill", "dodge"}


def displot(
    data: Any,
    *,
    x: Any = None,
    y: Any = None,
    hue: Any = None,
    col: Any = None,
    row: Any = None,
    kind: str = "hist",
    fill: bool = True,
    cumulative: bool = False,
    log_scale: bool = False,
    stat: str = "count",
    bins: Any = "sturges",
    bandwidth: Any = "scott",
    bw_adjust: float = 1.0,
    multiple: str = "layer",
    kde: bool = False,
    rug: bool = False,
    height: float | None = None,
    aspect: float | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Univariate distribution plot.

    Convenience wrapper that dispatches to ``mark_histogram``, ``mark_density``,
    or ``mark_tick`` based on ``kind``.  The ``multiple`` parameter controls how
    overlapping groups (from ``hue``) are positioned, and the ``kde`` / ``rug``
    flags optionally layer additional marks on top of the primary kind.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str or encoding, optional
        Column name for the distribution variable (horizontal axis).
    y : str or encoding, optional
        Column name for the distribution variable (vertical axis).
    hue : str or encoding, optional
        Column name to map to color (one distribution per level).
    col : str, optional
        Column name for faceting across columns.
    row : str, optional
        Column name for faceting across rows.
    kind : {"hist", "kde", "ecdf", "rug"}, default "hist"
        Which distribution mark to draw.  ``"hist"`` calls
        ``mark_histogram``; ``"kde"`` calls ``mark_density`` (filled by
        default); ``"ecdf"`` builds a cumulative frequency line via ``Bin``
        + ``mark_line``; ``"rug"`` calls ``mark_tick``.
    fill : bool, default True
        Fill the area under the KDE curve (``kind="kde"`` only).
    cumulative : bool, default False
        Produce a cumulative histogram or density (``kind="hist"`` and
        ``kind="kde"``).
    log_scale : bool, default False
        Apply a ``log`` scale to the ``x`` axis.
    stat : {"count", "density"}, default "count"
        Statistic to plot on the value axis for ``kind="hist"``.
        ``"density"`` normalises so the total area integrates to 1.
    bins : int or str, default "sturges"
        Binning rule for ``kind="hist"``.  An integer is forwarded as
        ``bin_count``; a string (``"sturges"``, ``"fd"``, etc.) lets the
        Rust engine decide the count automatically.
    bandwidth : str, default "scott"
        Bandwidth selector for ``kind="kde"`` (``"scott"`` or ``"silverman"``).
    bw_adjust : float, default 1.0
        Multiplicative bandwidth adjustment for ``kind="kde"``.
    multiple : {"layer", "stack", "fill", "dodge"}, default "layer"
        How to render multiple distributions (one per ``hue`` level).
        ``"layer"`` overlays them (``Identity``); ``"dodge"`` places them
        side by side; ``"stack"`` and ``"fill"`` use ``Stack`` with
        ``offset="zero"`` or ``offset="normalize"`` respectively.
    kde : bool, default False
        When ``True`` and ``kind != "kde"``, layer a ``mark_density`` on
        top of the primary mark.
    rug : bool, default False
        When ``True`` and ``kind != "rug"``, layer a ``mark_tick`` rug
        on top of the primary mark.
    height : float or None, optional
        Height of the chart in pixels.  Width is derived from ``aspect``.
    aspect : float or None, optional
        Aspect ratio (width = height * aspect).  Requires ``height``.
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()``.

    Returns
    -------
    Chart
        Configured chart (possibly layered, faceted, or sized).

    Raises
    ------
    ValueError
        If ``kind`` or ``multiple`` is not one of the supported values.
    ValueError
        If ``kind="ecdf"`` is used without specifying ``x=``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.displot(df, x="sepal_length")

    KDE with per-species coloring:

    >>> fm.displot(df, x="sepal_length", hue="species", kind="kde")

    Stacked histogram with an overlaid rug:

    >>> fm.displot(df, x="tip", hue="sex", multiple="stack", rug=True)
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"displot: kind must be one of {sorted(_VALID_KINDS)}; got {kind!r}")
    if multiple not in _VALID_MULTIPLE:
        raise ValueError(
            f"displot: multiple must be one of {sorted(_VALID_MULTIPLE)}; got {multiple!r}"
        )

    # Position adjustment from `multiple`.
    position = _multiple_to_position(multiple, hue)

    # Build the base chart.
    chart = Chart(data)

    # Encoding: x (required for most kinds), color from hue.
    enc: dict = {}
    if x is not None:
        enc["x"] = x
    if y is not None:
        enc["y"] = y
    if hue is not None:
        enc["color"] = hue
    enc.update(encode_kwargs)

    # Mark + transforms by kind.
    if kind == "hist":
        bin_count = bins if isinstance(bins, int) else None
        # When `multiple` requires per-group binning (stack/fill/dodge) and a
        # hue is bound, thread `groupby=hue` so the Bin transform emits per-
        # (bin, group) rows preserving the hue column for color encoding +
        # position adjustment.
        hist_kwargs: dict = dict(
            bin_count=bin_count,
            cumulative=cumulative,
            density=(stat == "density"),
            position=position,
        )
        if hue is not None and multiple in ("stack", "fill", "dodge"):
            hist_kwargs["groupby"] = hue
        chart = chart.mark_histogram(**hist_kwargs)
    elif kind == "kde":
        # When hue is bound, thread `groupby=hue` so the Kde transform emits
        # per-(grid, group) rows preserving the hue column for color encoding.
        # Position adjustment is informational for KDE (continuous curves
        # overlay regardless of `multiple`), but groupby is required whenever
        # hue is set so each level gets its own curve.
        kde_kwargs: dict = dict(
            bandwidth=bandwidth,
            bw_adjust=bw_adjust,
            fill=fill,
            position=position,
        )
        if hue is not None:
            kde_kwargs["groupby"] = hue
        chart = chart.mark_density(**kde_kwargs)
    elif kind == "ecdf":
        # ECDF: cumulative bin → step line.
        if x is None:
            raise ValueError("displot(kind='ecdf') requires x=")
        from ferrum.encoding import X as _X_ecdf
        from ferrum.encoding.base import ChannelBase as _CB_ecdf

        bin_count = bins if isinstance(bins, int) else None
        # Extract the original field name for the axis title.
        _ecdf_x_name = x.field if isinstance(x, _CB_ecdf) else str(x)
        chart = chart.transform(Bin(field=_ecdf_x_name, bin_count=bin_count, cumulative=True))
        chart = chart.mark_line()
        # Re-route encoding to bin output columns, preserving the original
        # variable name as the x-axis title.
        enc["x"] = _X_ecdf("bin_start", title=_ecdf_x_name)
        enc["y"] = "count"
    elif kind == "rug":
        chart = chart.mark_tick()

    chart = chart.encode(**enc)

    # Optional kde/rug layers (only when not already that kind).
    if kde and kind != "kde":
        kde_layer = (
            Chart(data)
            .mark_density(bandwidth=bandwidth, bw_adjust=bw_adjust, fill=False)
            .encode(x=x)
        )
        chart = chart + kde_layer
    if rug and kind != "rug":
        rug_layer = Chart(data).mark_tick().encode(x=x)
        chart = chart + rug_layer

    # Name the layers so override passthrough can target them.
    if chart._layers is not None:
        from dataclasses import replace as _dc_replace

        _kind_names = {"bar": "histogram", "area": "kde", "line": "kde",
                       "tick": "rug", "point": "scatter"}
        seen: dict[str, int] = {}
        named: list = []
        for ly in chart._layers:
            if ly.name is not None:
                named.append(ly)
            else:
                base = _kind_names.get(ly.mark or "", ly.mark or "layer")
                count = seen.get(base, 0)
                seen[base] = count + 1
                lname = base if count == 0 else f"{base}_{count + 1}"
                named.append(_dc_replace(ly, name=lname))
        chart._layers = named

    # log_scale on x.
    if log_scale and x is not None:
        from ferrum.encoding import X

        chart = chart.encode(x=X(x, scale={"type": "log"}))

    # Faceting.
    if col is not None or row is not None:
        if col is not None and row is not None:
            chart = chart.facet(row=row, col=col)
        elif col is not None:
            chart = chart.facet(col=col)
        else:
            chart = chart.facet(row=row)

    # Properties.
    if height is not None or aspect is not None:
        h = height if height is not None else 300.0
        w = h * aspect if aspect is not None else h
        chart = chart.properties(width=w, height=h)

    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)

    if theme is not None:
        chart = chart.theme(theme)

    return chart


def _multiple_to_position(multiple: str, hue: Any):
    if multiple == "layer":
        return Identity()
    if multiple == "dodge":
        return Dodge(by=hue)
    if multiple == "stack":
        return Stack(by=hue, offset="zero")
    if multiple == "fill":
        return Stack(by=hue, offset="normalize")
    raise ValueError(f"unknown multiple {multiple!r}")
