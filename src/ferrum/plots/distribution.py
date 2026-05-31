"""Distribution domain figure functions.

Public API
----------
displot, catplot, relplot.

``displot`` dispatches to histogram, KDE, ECDF, or rug marks.
``catplot`` dispatches to strip, swarm, box, violin, boxen, point, bar,
or count marks.
``relplot`` dispatches to scatter or line marks with optional faceting.
"""

from __future__ import annotations
from typing import Any

from ferrum import (
    Aggregate,
    AggregateOp,
    Bin,
    Chart,
    CoordFlip,
    Dodge,
    Identity,
    Jitter,
    Stack,
)
from ferrum._overrides import _apply_overrides
from ferrum.plots._helpers import _finalize_chart


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISPLOT_VALID_KINDS = {"hist", "kde", "ecdf", "rug"}
_DISPLOT_VALID_MULTIPLE = {"layer", "stack", "fill", "dodge"}

_CATPLOT_VALID_KINDS = {"strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"}
_RELPLOT_VALID_KINDS = {"scatter", "line"}

# Minimum per-panel height (px) used by the auto-sizing path in displot/catplot
# when `height` is not set and the chart is faceted across rows.  Derived from
# the Rust layout minimum: EmptyPanel triggers at ~55 px; 150 px gives comfortable
# axes + mark area and matches seaborn's default facet height.
_FACET_DEFAULT_PANEL_HEIGHT_PX: float = 150.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_facet_levels(data: Any, field: str) -> int:
    """Return the number of distinct non-null values in *field* within *data*.

    Used to auto-scale chart height when faceting across rows.  Returns 1 on
    any error (field absent, data not a supported type, etc.) so the caller
    degrades gracefully to the default size rather than raising.
    """
    try:
        import polars as pl
        from ferrum._coerce import to_arrow_table

        if isinstance(data, pl.DataFrame):
            df = data
        elif isinstance(data, pl.LazyFrame):
            df = data.collect()
        else:
            df = pl.from_arrow(to_arrow_table(data))
        if field not in df.columns:
            return 1
        return df[field].n_unique()
    except Exception:
        return 1


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


# ---------------------------------------------------------------------------
# displot
# ---------------------------------------------------------------------------


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
    if kind not in _DISPLOT_VALID_KINDS:
        raise ValueError(
            f"displot: kind must be one of {sorted(_DISPLOT_VALID_KINDS)}; got {kind!r}"
        )
    if multiple not in _DISPLOT_VALID_MULTIPLE:
        raise ValueError(
            f"displot: multiple must be one of {sorted(_DISPLOT_VALID_MULTIPLE)}; got {multiple!r}"
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

    # Optional kde/rug overlay layers (only when not already that kind).
    #
    # These overlays must NOT use Chart.__add__ because the histogram's Bin
    # transform replaces the original data columns (e.g. "sepal_length" →
    # "bin_start"/"bin_end"/"count").  __add__ merges all transforms into a
    # single sequential pipeline, so the Kde overlay would run on the Bin
    # output and fail to find the original column.
    #
    # Instead we resolve the chart, then manually prepend named transforms
    # for the overlays.  Named transforms read from the current chain head
    # (the original data, since they run before the unnamed Bin) without
    # advancing it.  The overlay _Layer objects set data_source to the named
    # transform key so they read the correct output.
    if (kde and kind != "kde") or (rug and kind != "rug"):
        from ferrum._layer import _Layer as _OverlayLayer
        from ferrum.encoding.base import ChannelBase as _CB_overlay

        chart = chart._resolve_pending()
        chart = chart._clone()
        x_field = x.field if isinstance(x, _CB_overlay) else str(x) if x is not None else None

        # Convert single-mark chart to layered mode: promote the existing
        # mark + encoding into a primary layer so overlay layers can be
        # appended alongside it.
        if chart._layers is None and chart._mark is not None:
            primary_layer = _OverlayLayer(
                mark=chart._mark,
                encoding=dict(chart._encoding),
                mark_kwargs=dict(chart._mark_kwargs) if chart._mark_kwargs else None,
                position=chart._position,
            )
            chart._layers = [primary_layer]
            chart._mark = None

        if kde and kind != "kde" and x_field is not None:
            from ferrum import Kde as _KdeTransform
            import polars as pl

            # KDE transform requires Float64; auto-cast integer x columns.
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
            if hasattr(chart._data, "columns") and x_field in chart._data.columns:
                if chart._data[x_field].dtype in _INT_DTYPES:
                    chart._data = chart._data.with_columns(pl.col(x_field).cast(pl.Float64))

            kde_xform = _KdeTransform(
                x_field, bandwidth=bandwidth, bw_adjust=bw_adjust, name="kde_overlay"
            )
            # Prepend the named Kde before the unnamed Bin so it reads from
            # the original (pre-Bin) data batch.
            chart._transforms = [kde_xform] + list(chart._transforms or [])
            chart._layers.append(
                _OverlayLayer(
                    name="kde",
                    mark="line",
                    encoding={"x": "value", "y": "density"},
                    data_source="kde_overlay",
                )
            )

        if rug and kind != "rug" and x_field is not None:
            from ferrum._core import PyIdentity as _IdentityTransform

            rug_xform = _IdentityTransform("rug_data")
            # Prepend the named Identity before the unnamed Bin so it
            # captures the original data for the rug tick marks.
            chart._transforms = [rug_xform] + list(chart._transforms or [])
            chart._layers.append(
                _OverlayLayer(
                    name="rug",
                    mark="tick",
                    encoding={"x": x_field},
                    data_source="rug_data",
                )
            )

    # Name the layers so override passthrough can target them.
    if chart._layers is not None:
        from dataclasses import replace as _dc_replace

        _kind_names = {
            "bar": "histogram",
            "area": "kde",
            "line": "kde",
            "tick": "rug",
            "point": "scatter",
        }
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
    # When row-faceting is active, `height` is treated as per-panel height
    # (seaborn semantics).  The total canvas height is per-panel height ×
    # number of row facet levels.  When `height` is absent and row-faceting
    # is active, auto-compute a total height so every panel is at least
    # _FACET_DEFAULT_PANEL_HEIGHT_PX tall.
    if row is not None:
        n_row_panels = _count_facet_levels(data, str(row))
        if height is not None:
            # height= is per-panel: total = height × n_panels
            total_h = height * n_row_panels
        else:
            # No explicit height: ensure each panel meets the minimum.
            total_h = _FACET_DEFAULT_PANEL_HEIGHT_PX * n_row_panels
        w = total_h * aspect if aspect is not None else None
        props: dict = {"height": total_h}
        if w is not None:
            props["width"] = w
        chart = chart.properties(**props)
    elif height is not None or aspect is not None:
        h = height if height is not None else 300.0
        w = h * aspect if aspect is not None else h
        chart = chart.properties(width=w, height=h)

    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


# ---------------------------------------------------------------------------
# catplot
# ---------------------------------------------------------------------------


def catplot(
    data: Any,
    *,
    x: Any = None,
    y: Any = None,
    hue: Any = None,
    col: Any = None,
    row: Any = None,
    kind: str = "strip",
    order: Any = None,
    hue_order: Any = None,
    orient: Any = None,
    dodge: bool = False,
    jitter: bool = True,
    native_scale: bool = False,
    ci: Any = 95,
    n_boot: int = 1000,
    seed: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Categorical figure-level function.

    Dispatches to the appropriate mark based on ``kind``:

    * ``"strip"``  -- ``mark_point`` with ``Jitter`` (when ``jitter=True``, default) or ``Dodge`` (when ``dodge=True`` and ``hue`` is set).
    * ``"swarm"``  -- ``mark_swarm``.
    * ``"box"``    -- ``mark_boxplot`` (box + whiskers + outliers).
    * ``"violin"`` -- ``mark_violin`` (kernel-density outline).
    * ``"boxen"``  -- ``mark_boxen`` (letter-value / extended box).
    * ``"point"``  -- ``mark_point`` per observation on the categorical axis.
    * ``"bar"``    -- ``mark_bar`` per observation on the categorical axis.
    * ``"count"``  -- ``Aggregate(count)`` + ``mark_bar``.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str or encoding, optional
        Column name for the horizontal axis (categorical by default).
    y : str or encoding, optional
        Column name for the vertical axis (value by default).
    hue : str or encoding, optional
        Column name to map to color (one visual group per level).
    col : str, optional
        Column name for faceting across columns.
    row : str, optional
        Column name for faceting across rows.
    kind : {"strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"}, default "strip"
        Which categorical mark to draw.
    order : list of str, optional
        Explicit ordering for the categorical axis levels.  Passed as
        ``sort=order`` on the categorical-axis encoding so the domain
        renders in the given order.
    hue_order : list of str, optional
        Explicit ordering for hue levels.  Passed as ``sort=hue_order``
        on the color encoding.
    orient : {"h", "v", None}, optional
        ``"h"`` flips the axes (``x`` becomes the value axis, ``y`` the category
        axis) and applies ``CoordFlip``.  ``"v"`` and ``None`` are both treated
        as vertical (the default); no error is raised for other values.
    dodge : bool, default False
        When ``True`` and ``hue`` is set, apply ``Dodge`` so each hue level
        is drawn side-by-side rather than overlaid.
    jitter : bool, default True
        For ``kind="strip"``, add ``Jitter`` on the categorical axis.
        Ignored when ``dodge=True``.
    native_scale : bool, default False
        When ``True``, treat the categorical axis as quantitative instead
        of ordinal (preserves numeric spacing rather than equal-spacing
        categories).  Currently raises ``ValueError`` because the renderer
        does not support quantitative categorical axes.
    ci : int or float, default 95
        Confidence-interval level (0--100) for ``"point"`` and ``"bar"``
        kinds.  Currently raises ``ValueError`` because the Summary
        transform is not yet wired into catplot.
    n_boot : int, default 1000
        Bootstrap iteration count used to compute ``ci``.  Currently
        raises ``ValueError`` alongside ``ci``.
    seed : int or None, optional
        Random seed forwarded to ``Jitter`` for reproducible strip positions.
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
        Configured chart (possibly faceted or coord-flipped).

    Raises
    ------
    ValueError
        If ``kind`` is not one of the supported values.
    ValueError
        If ``kind="count"`` is used without specifying ``x`` (or ``y`` when
        ``orient="h"``).
    ValueError
        If ``native_scale=True`` is passed (not yet supported by the renderer).
    ValueError
        If ``ci`` is not the default value ``95`` and the kind is ``"point"``
        or ``"bar"`` (Summary transform not yet wired).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.catplot(df, x="species", y="sepal_length", kind="box")

    Group by a hue variable with dodged bars:

    >>> fm.catplot(df, x="day", y="tip", hue="sex", kind="bar", dodge=True)

    Horizontal violin plot:

    >>> fm.catplot(df, x="total_bill", y="day", kind="violin", orient="h")
    """
    if kind not in _CATPLOT_VALID_KINDS:
        raise ValueError(
            f"catplot: kind must be one of {sorted(_CATPLOT_VALID_KINDS)}; got {kind!r}"
        )

    if native_scale:
        raise ValueError(
            "catplot: native_scale=True is not supported; the renderer does not "
            "support quantitative categorical axes"
        )

    if ci != 95 and kind in ("point", "bar"):
        raise ValueError(
            f"catplot: ci={ci!r} with kind={kind!r} is not yet supported; "
            "the Summary transform is not wired into catplot"
        )

    if n_boot != 1000 and kind in ("point", "bar"):
        raise ValueError(
            f"catplot: n_boot={n_boot!r} with kind={kind!r} is not yet supported; "
            "the Summary transform is not wired into catplot"
        )

    # Determine the categorical and value axes. By default x is categorical,
    # y is value; orient="h" flips to y categorical / x value (and we add
    # CoordFlip to the chart).
    horizontal = orient == "h"
    cat_field = x if not horizontal else y
    val_field = y if not horizontal else x

    # Position adjustment.
    position = None
    if dodge and hue is not None:
        position = Dodge(by=hue)

    # Encoding shared across all kinds.
    from ferrum.encoding import Color as _Color
    from ferrum.encoding import X as _X, Y as _Y

    # Build the encoding with x=categorical, y=value regardless of orientation.
    # CoordFlip (added below when horizontal=True) handles the visual axis swap.
    # Stat-mark desugars (boxplot, violin, swarm, etc.) always expect x=cat, y=val,
    # so we normalise the encoding here rather than propagating the orientation
    # through every desugar function.
    enc: dict = {}
    if not horizontal:
        if x is not None:
            enc["x"] = x
        if y is not None:
            enc["y"] = y
    else:
        # orient="h": user passed x=val, y=cat — swap so x=cat, y=val for desugars.
        if y is not None:
            enc["x"] = y  # cat_field goes on x
        if x is not None:
            enc["y"] = x  # val_field goes on y
    if hue is not None:
        enc["color"] = hue

    # Wire order → sort on the categorical axis encoding.
    # After the swap above, cat is always on x, so cat_channel is always "x".
    if order is not None and cat_field is not None:
        enc["x"] = _X(cat_field, sort=list(order))

    # Wire hue_order → sort on the color encoding.
    if hue_order is not None and hue is not None:
        enc["color"] = _Color(hue, sort=list(hue_order))

    enc.update(encode_kwargs)

    chart = Chart(data)

    if kind == "strip":
        if position is not None:
            # dodge=True with hue overrides jitter (per spec — single-position
            # adjustments aren't composable in Phase 9c).
            chart = chart.mark_point(position=position)
        elif jitter:
            # After the enc normalisation above, cat is always on x, so jitter
            # always acts on the x axis (the categorical band axis).
            chart = chart.mark_point(
                position=Jitter(axis="x", width=0.4, seed=seed),
            )
        else:
            chart = chart.mark_point(position=Identity())
    elif kind == "swarm":
        if position is not None:
            chart = chart.mark_swarm(position=position)
        else:
            chart = chart.mark_swarm()
    elif kind == "box":
        if position is not None:
            chart = chart.mark_boxplot(position=position)
        else:
            chart = chart.mark_boxplot()
    elif kind == "violin":
        if position is not None:
            chart = chart.mark_violin(position=position)
        else:
            chart = chart.mark_violin()
    elif kind == "boxen":
        if position is not None:
            chart = chart.mark_boxen(position=position)
        else:
            chart = chart.mark_boxen()
    elif kind == "point":
        chart = chart.mark_point(position=position)
    elif kind == "bar":
        chart = chart.mark_bar(position=position) if position is not None else chart.mark_bar()
    elif kind == "count":
        # Aggregate(count of cat_field) → bar.
        if cat_field is None:
            raise ValueError("catplot(kind='count') requires x= (or y= when orient='h')")
        op = AggregateOp(cat_field, "count", "n")
        chart = chart.transform(Aggregate([op], groupby=[cat_field]))
        chart = chart.mark_bar(position=position) if position is not None else chart.mark_bar()
        # Remap value axis to the count column.
        # After enc normalisation, val is always on y (cat on x), so "n" always
        # goes on y regardless of orientation.
        enc["y"] = "n"

    chart = chart.encode(**enc)

    # orient="h" → CoordFlip.
    if horizontal:
        chart = chart.coord(CoordFlip())

    # Faceting.
    if col is not None or row is not None:
        if col is not None and row is not None:
            chart = chart.facet(row=row, col=col)
        elif col is not None:
            chart = chart.facet(col=col)
        else:
            chart = chart.facet(row=row)

    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


# ---------------------------------------------------------------------------
# relplot
# ---------------------------------------------------------------------------


def relplot(
    data: Any,
    *,
    x: Any,
    y: Any,
    hue: Any = None,
    size: Any = None,
    style: Any = None,
    col: Any = None,
    row: Any = None,
    kind: str = "scatter",
    height: float | None = None,
    aspect: float | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Relational figure-level function for scatter and line plots.

    Dispatches to the appropriate mark based on ``kind``:

    * ``"scatter"`` -- ``mark_point()``.  ``style`` maps to ``Shape``.
    * ``"line"``    -- ``mark_line()``.  ``style`` maps to ``StrokeDash``.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str or encoding
        Column name for the horizontal axis (required).
    y : str or encoding
        Column name for the vertical axis (required).
    hue : str or encoding, optional
        Column name to map to color (one group per level).
    size : str or encoding, optional
        Column name to map to the ``Size`` channel (point area or line width).
    style : str or encoding, optional
        Column name to map to ``Shape`` (scatter) or ``StrokeDash`` (line).
    col : str, optional
        Column name for faceting across columns.
    row : str, optional
        Column name for faceting across rows.
    kind : {"scatter", "line"}, default "scatter"
        Which relational mark to draw.
    height : float or None, optional
        Height of the chart in pixels.  Width is derived from ``aspect``.
    aspect : float or None, optional
        Aspect ratio (width = height * aspect).  Requires ``height``.
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
        Configured chart (possibly faceted or sized).

    Raises
    ------
    ValueError
        If ``kind`` is not one of ``"scatter"`` or ``"line"``.

    Examples
    --------
    Scatter with hue grouping and column faceting:

    >>> import ferrum as fm
    >>> fm.relplot(df, x="total_bill", y="tip", hue="sex", col="time")

    Line plot with per-level dash styling:

    >>> fm.relplot(df, x="timepoint", y="signal", hue="region",
    ...            style="region", kind="line")
    """
    if kind not in _RELPLOT_VALID_KINDS:
        raise ValueError(
            f"relplot: kind must be one of {sorted(_RELPLOT_VALID_KINDS)}; got {kind!r}"
        )

    chart = Chart(data)

    enc: dict = {}
    if x is not None:
        enc["x"] = x
    if y is not None:
        enc["y"] = y
    if hue is not None:
        enc["color"] = hue
    if size is not None:
        enc["size"] = size
    if style is not None:
        if kind == "scatter":
            enc["shape"] = style
        else:
            enc["stroke_dash"] = style
    enc.update(encode_kwargs)

    if kind == "scatter":
        chart = chart.mark_point()
    else:
        chart = chart.mark_line()

    chart = chart.encode(**enc)

    if col is not None or row is not None:
        if col is not None and row is not None:
            chart = chart.facet(row=row, col=col)
        elif col is not None:
            chart = chart.facet(col=col)
        else:
            chart = chart.facet(row=row)

    if height is not None or aspect is not None:
        h = height if height is not None else 300.0
        w = h * aspect if aspect is not None else h
        chart = chart.properties(width=w, height=h)

    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )
