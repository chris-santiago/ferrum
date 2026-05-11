"""Phase 9e — displot (distribution figure-level function)."""
from __future__ import annotations
from typing import Any

from ferrum import Bin, Chart, Identity, Dodge, Stack


_VALID_KINDS = {"hist", "kde", "ecdf", "rug"}
_VALID_MULTIPLE = {"layer", "stack", "fill", "dodge"}


def displot(
    data: Any, *,
    x: Any = None, y: Any = None,
    hue: Any = None, col: Any = None, row: Any = None,
    kind: str = "hist",
    fill: bool = True, cumulative: bool = False, log_scale: bool = False,
    stat: str = "count",
    bins: Any = "sturges",
    bandwidth: Any = "scott", bw_adjust: float = 1.0,
    multiple: str = "layer",
    kde: bool = False, rug: bool = False,
    height: float | None = None, aspect: float | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Distribution figure-level function — see ferrum-spec.md §3.14.

    Builds a Chart for histogram / KDE / ECDF / rug plots. The ``multiple``
    parameter routes to a position adjustment (Identity/Dodge/Stack); the
    ``kde``/``rug`` flags optionally layer additional marks on top.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"displot: kind must be one of {sorted(_VALID_KINDS)}; got {kind!r}"
        )
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
            bin_count=bin_count, cumulative=cumulative,
            density=(stat == "density"),
            position=position,
        )
        if hue is not None and multiple in ("stack", "fill", "dodge"):
            hist_kwargs["groupby"] = hue
        chart = chart.mark_histogram(**hist_kwargs)
    elif kind == "kde":
        chart = chart.mark_density(
            bandwidth=bandwidth, bw_adjust=bw_adjust, fill=fill,
            position=position,
        )
    elif kind == "ecdf":
        # ECDF: cumulative bin → step line.
        if x is None:
            raise ValueError("displot(kind='ecdf') requires x=")
        bin_count = bins if isinstance(bins, int) else None
        chart = chart.transform(Bin(field=x, bin_count=bin_count, cumulative=True))
        chart = chart.mark_line()
        # Re-route encoding to bin output columns.
        enc["x"] = "bin_start"
        enc["y"] = "count"
    elif kind == "rug":
        chart = chart.mark_tick()

    chart = chart.encode(**enc)

    # Optional kde/rug layers (only when not already that kind).
    if kde and kind != "kde":
        kde_layer = Chart(data).mark_density(
            bandwidth=bandwidth, bw_adjust=bw_adjust, fill=False
        ).encode(x=x)
        chart = chart + kde_layer
    if rug and kind != "rug":
        rug_layer = Chart(data).mark_tick().encode(x=x)
        chart = chart + rug_layer

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
