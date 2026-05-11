"""Reference-line, rectangle, and text annotation helpers."""
from __future__ import annotations

from typing import Optional

import polars as pl

from ferrum.chart import Chart


def annotate_hline(y: float, *, label: Optional[str] = None,
                   stroke: Optional[str] = None, stroke_dash=None) -> Chart:
    """Horizontal reference line at a fixed y position.

    Returns a single-mark ``Chart`` suitable for ``|`` / ``&`` concatenation
    composition; for true overlay/layer, use ``+`` with a chart that shares
    the same DataFrame.

    Parameters
    ----------
    y : float
        Y position of the line in data coordinates.
    label : str, optional
        Reserved for future use (no-op today).
    stroke : str, optional
        Line color as a CSS color string. Defaults to the mark default when
        omitted.
    stroke_dash : list of float, optional
        SVG dash array, e.g. ``[4, 4]`` for evenly dashed.

    Returns
    -------
    Chart
        Annotation chart suitable for ``|`` / ``&`` composition.

    Examples
    --------
    >>> import ferrum as fm
    >>> ref = fm.annotate_hline(y=0.0, stroke="red", stroke_dash=[4, 4])
    >>> chart = fm.Chart(df).encode(x="t", y="r").mark_line() & ref
    """
    df = pl.DataFrame({"_y": [y]})
    kwargs: dict = {}
    if stroke is not None:
        kwargs["stroke"] = stroke
    if stroke_dash is not None:
        kwargs["stroke_dash"] = stroke_dash
    return Chart(df).mark_rule(**kwargs).encode(y="_y")


def annotate_vline(x: float, *, label: Optional[str] = None,
                   stroke: Optional[str] = None, stroke_dash=None) -> Chart:
    """Vertical reference line at a fixed x position.

    Returns a single-mark ``Chart`` suitable for ``|`` / ``&`` concatenation
    composition; for true overlay/layer, use ``+`` with a chart that shares
    the same DataFrame.

    Parameters
    ----------
    x : float
        X position of the line in data coordinates.
    label : str, optional
        Reserved for future use (no-op today).
    stroke : str, optional
        Line color as a CSS color string.
    stroke_dash : list of float, optional
        SVG dash array, e.g. ``[4, 4]``.

    Returns
    -------
    Chart
        Annotation chart suitable for ``|`` / ``&`` composition.

    Examples
    --------
    >>> import ferrum as fm
    >>> ref = fm.annotate_vline(x=2020, stroke="#888")
    >>> chart = fm.Chart(df).encode(x="year", y="val").mark_line() & ref
    """
    df = pl.DataFrame({"_x": [x]})
    kwargs: dict = {}
    if stroke is not None:
        kwargs["stroke"] = stroke
    if stroke_dash is not None:
        kwargs["stroke_dash"] = stroke_dash
    return Chart(df).mark_rule(**kwargs).encode(x="_x")


def annotate_rect(x1: float, x2: float, y1: float, y2: float, *,
                  fill: Optional[str] = None, opacity: float = 0.1,
                  label: Optional[str] = None) -> Chart:
    """Shaded rectangle region spanning (x1, y1) to (x2, y2).

    Returns a ``mark_rect`` annotation chart for ``|`` / ``&`` concatenation
    composition; for true overlay/layer, use ``+`` with a chart that shares
    the same DataFrame.

    .. note::
       ``x2`` and ``y2`` are accepted but deferred — the renderer currently
       anchors the rect at ``(x1, y1)`` only. Full X2/Y2 support lands in
       Phase 9.

    Parameters
    ----------
    x1 : float
        Left x boundary in data coordinates.
    x2 : float
        Right x boundary in data coordinates. Reserved for future use
        (no-op today — see note above).
    y1 : float
        Bottom y boundary in data coordinates.
    y2 : float
        Top y boundary in data coordinates. Reserved for future use
        (no-op today).
    fill : str, optional
        Fill color as a CSS color string.
    opacity : float, default 0.1
        Fill opacity in ``[0, 1]``.
    label : str, optional
        Reserved for future use (no-op today).

    Returns
    -------
    Chart
        Annotation chart suitable for ``|`` / ``&`` composition.

    Examples
    --------
    >>> import ferrum as fm
    >>> shade = fm.annotate_rect(x1=2018, x2=2020, y1=0, y2=100,
    ...                          fill="#ffcc00", opacity=0.2)
    >>> chart = fm.Chart(df).encode(x="year", y="val").mark_line() & shade
    """
    df = pl.DataFrame({"_x1": [x1], "_x2": [x2], "_y1": [y1], "_y2": [y2]})
    kwargs: dict = {"opacity": opacity}
    if fill is not None:
        kwargs["fill"] = fill
    return Chart(df).mark_rect(**kwargs).encode(x="_x1", y="_y1")


def annotate_text(x: float, y: float, text: str, *, dx: float = 0, dy: float = 0,
                  align: str = "center", baseline: str = "middle",
                  font_size: Optional[float] = None, color: Optional[str] = None,
                  angle: Optional[float] = None) -> Chart:
    """Free-floating text annotation at a fixed (x, y) position.

    Returns a ``mark_text`` chart for ``|`` / ``&`` concatenation composition;
    for true overlay/layer, use ``+`` with a chart that shares the same
    DataFrame.

    .. note::
       Text content is not rendered in the current phase; the ``text``
       argument is stored in a ``_text`` column but that column is never
       bound to an encoding channel, so the mark renders as a positioned but
       empty text element. Full Text channel support lands in Phase 9+.

    Parameters
    ----------
    x : float
        X position in data coordinates.
    y : float
        Y position in data coordinates.
    text : str
        Text string to display.
    dx : float, default 0
        Horizontal pixel offset from ``(x, y)``.
    dy : float, default 0
        Vertical pixel offset from ``(x, y)``.
    align : str, default "center"
        Horizontal text alignment (SVG ``text-anchor``): ``"left"``,
        ``"center"``, or ``"right"``.
    baseline : str, default "middle"
        Vertical text baseline: ``"top"``, ``"middle"``, or ``"bottom"``.
    font_size : float, optional
        Font size in points.
    color : str, optional
        Text fill color as a CSS color string.
    angle : float, optional
        Rotation angle in degrees (clockwise).

    Returns
    -------
    Chart
        Annotation chart suitable for ``|`` / ``&`` composition.

    Examples
    --------
    >>> import ferrum as fm
    >>> label = fm.annotate_text(x=2020, y=95, text="peak", dy=-8,
    ...                          color="#333", font_size=11)
    >>> chart = fm.Chart(df).encode(x="year", y="val").mark_line() & label
    """
    df = pl.DataFrame({"_x": [x], "_y": [y], "_text": [text]})
    kwargs: dict = {"dx": dx, "dy": dy, "align": align, "baseline": baseline}
    if font_size is not None:
        kwargs["font_size"] = font_size
    if color is not None:
        kwargs["fill"] = color
    if angle is not None:
        kwargs["angle"] = angle
    return Chart(df).mark_text(**kwargs).encode(x="_x", y="_y")
