"""Categorical convenience functions (catplot)."""

from __future__ import annotations
from typing import Any

from ferrum import (
    Aggregate,
    AggregateOp,
    Chart,
    CoordFlip,
    Dodge,
    Identity,
    Jitter,
)


_VALID_KINDS = {"strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"}


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
        Reserved for future use (no-op today). Explicit ordering for the
        categorical axis levels.
    hue_order : list of str, optional
        Reserved for future use (no-op today). Explicit ordering for hue levels.
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
        Reserved for future use (no-op in the current renderer).
    ci : int or float, default 95
        Reserved for future use (no-op today). When wired, will set the
        confidence-interval level for ``"point"`` and ``"bar"`` kinds.
    n_boot : int, default 1000
        Reserved for future use (no-op today). When wired, will set the
        bootstrap iteration count used to compute ``ci``.
    seed : int or None, optional
        Random seed forwarded to ``Jitter`` for reproducible strip positions.
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.catplot(df, x="species", y="sepal_length", kind="box")

    Group by a hue variable with dodged bars:

    >>> fm.catplot(df, x="day", y="tip", hue="sex", kind="bar", dodge=True)

    Horizontal violin plot:

    >>> fm.catplot(df, x="total_bill", y="day", kind="violin", orient="h")
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"catplot: kind must be one of {sorted(_VALID_KINDS)}; got {kind!r}")

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
    enc: dict = {}
    if x is not None:
        enc["x"] = x
    if y is not None:
        enc["y"] = y
    if hue is not None:
        enc["color"] = hue
    enc.update(encode_kwargs)

    chart = Chart(data)

    if kind == "strip":
        if position is not None:
            # dodge=True with hue overrides jitter (per spec — single-position
            # adjustments aren't composable in Phase 9c).
            chart = chart.mark_point(position=position)
        elif jitter:
            jit_axis = "x" if not horizontal else "y"
            chart = chart.mark_point(
                position=Jitter(axis=jit_axis, width=0.4, seed=seed),
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
        if not horizontal:
            enc["y"] = "n"
        else:
            enc["x"] = "n"

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

    if theme is not None:
        chart = chart.theme(theme)

    return chart
