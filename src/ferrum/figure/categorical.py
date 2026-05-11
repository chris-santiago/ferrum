"""Phase 9e — catplot (categorical figure-level function)."""
from __future__ import annotations
from typing import Any

from ferrum import (
    Aggregate, AggregateOp, Chart, CoordFlip, Dodge, Identity, Jitter,
)


_VALID_KINDS = {"strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"}


def catplot(
    data: Any, *,
    x: Any = None, y: Any = None,
    hue: Any = None, col: Any = None, row: Any = None,
    kind: str = "strip",
    order: Any = None, hue_order: Any = None, orient: Any = None,
    dodge: bool = False, jitter: bool = True, native_scale: bool = False,
    ci: Any = 95, n_boot: int = 1000, seed: int | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Categorical figure-level function — see ferrum-spec.md §3.14.

    Per-kind desugar:
      strip   -> mark_point [+ Jitter if jitter=True]
      swarm   -> mark_swarm
      box     -> mark_boxplot
      violin  -> mark_violin
      boxen   -> mark_boxen
      point   -> mark_point  (CI ribbon deferred — single-layer point chart)
      bar     -> mark_bar    (CI ribbon deferred — single-layer bar chart)
      count   -> Aggregate(count) + mark_bar
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"catplot: kind must be one of {sorted(_VALID_KINDS)}; got {kind!r}"
        )

    # Determine the categorical and value axes. By default x is categorical,
    # y is value; orient="h" flips to y categorical / x value (and we add
    # CoordFlip to the chart).
    horizontal = (orient == "h")
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
