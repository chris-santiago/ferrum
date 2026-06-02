"""Reference-line, rectangle, and text annotation helpers.

Metric-label classes (AUCLabel, APLabel, BrierLabel, OutlierLabel) and their
private helpers live in ``ferrum._metric_labels``; they are re-exported here
for backward compatibility.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional

import polars as pl

from ferrum.annotation.coords import (
    CoordValue,
    OrdinalCategoryCoord,
    PixelCoord,
    NormCoord,
    _is_iso8601_string,
    temporal_coord_to_epoch_ms,
)
from ferrum.chart import Chart
from ferrum._metric_labels import AUCLabel, APLabel, BrierLabel, OutlierLabel  # noqa: F401

# Type accepted by positional annotation parameters.  This is the same as
# CoordValue: numeric, temporal, ISO-8601 string, ordinal category string,
# or explicit PixelCoord/NormCoord wrapper.
_AnnotationCoord = CoordValue


def _coerce_coord(v: CoordValue) -> CoordValue:
    """Normalize an annotation positional coordinate for storage in a primitive.

    Rules:
    - ``PixelCoord`` / ``NormCoord`` / ``OrdinalCategoryCoord`` pass through
      unchanged — the caller already expressed precise intent.
    - ``datetime.date`` / ``datetime.datetime`` → epoch-milliseconds float.
    - String that IS an ISO-8601 date/datetime → epoch-milliseconds float.
    - String that is NOT ISO-8601 → ``OrdinalCategoryCoord`` (resolved against
      the chart's ordinal domain at render time).
    - Numeric (float, int) → float (data-space value).

    The returned value is always a type that ``_coord()`` in
    ``annotation/primitives.py`` knows how to serialize.
    """
    if isinstance(v, (PixelCoord, NormCoord, OrdinalCategoryCoord)):
        return v
    if isinstance(v, _dt.datetime):
        return temporal_coord_to_epoch_ms(v)
    if isinstance(v, _dt.date):
        return temporal_coord_to_epoch_ms(v)
    if isinstance(v, str):
        if _is_iso8601_string(v):
            return temporal_coord_to_epoch_ms(v)
        return OrdinalCategoryCoord(v)
    return float(v)


def _coerce_coord_to_numeric(v: CoordValue) -> float:
    """Convert a coordinate to a plain float for use in a polars DataFrame column.

    ``PixelCoord`` and ``NormCoord`` do not have a meaningful data-space value,
    so they are mapped to 0.0 as a placeholder (the primitive carries the real
    coord; the DataFrame column for the mark_rule is only used as a shape hint
    and is not used for rendering when the primitive is present).

    ``OrdinalCategoryCoord`` is mapped to 0.0 similarly — its true pixel
    position is resolved at render time from the ordinal domain.
    """
    if isinstance(v, (PixelCoord, NormCoord, OrdinalCategoryCoord)):
        return 0.0
    return float(v)  # already a float after _coerce_coord


def _attach_annotation_primitive(chart: Chart, primitive: object) -> Chart:
    """Set the annotation primitive on a Chart so ``+`` routes through annotations."""
    chart._annotation_primitive = primitive
    return chart


def annotate_hline(
    y: _AnnotationCoord,
    *,
    label: Optional[str] = None,
    stroke: Optional[str] = None,
    stroke_dash=None,
) -> Chart:
    """Horizontal reference line at a fixed y position.

    Returns a single-mark ``Chart`` suitable for ``|`` / ``&`` concatenation
    composition; for true overlay/layer, use ``+`` with a chart that shares
    the same DataFrame.

    Parameters
    ----------
    y : float, datetime.date, datetime.datetime, or str
        Y position of the line in data coordinates.  Temporal values
        (``date``, ``datetime``, ISO-8601 strings) are converted to
        epoch-milliseconds (UTC) to align with ferrum's temporal axis scale.
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
    from ferrum.annotation.coords import norm
    from ferrum.annotation.primitives import AnnotationLine

    y_coord = _coerce_coord(y)
    y_num = _coerce_coord_to_numeric(y_coord)
    df = pl.DataFrame({"_y": [y_num]})
    kwargs: dict = {}
    if stroke is not None:
        kwargs["stroke"] = stroke
    if stroke_dash is not None:
        kwargs["stroke_dash"] = stroke_dash
    chart = Chart(df).mark_rule(**kwargs).encode(y="_y")
    # Attach annotation primitive: horizontal line spanning the full x-axis
    prim = AnnotationLine(
        x1=norm(0),
        y1=y_coord,
        x2=norm(1),
        y2=y_coord,
        stroke=stroke or "#333",
        stroke_width=1,
        dash=stroke_dash,
    )
    return _attach_annotation_primitive(chart, prim)


def annotate_vline(
    x: _AnnotationCoord,
    *,
    label: Optional[str] = None,
    stroke: Optional[str] = None,
    stroke_dash=None,
) -> Chart:
    """Vertical reference line at a fixed x position.

    Returns a single-mark ``Chart`` suitable for ``|`` / ``&`` concatenation
    composition; for true overlay/layer, use ``+`` with a chart that shares
    the same DataFrame.

    Parameters
    ----------
    x : float, datetime.date, datetime.datetime, or str
        X position of the line in data coordinates.  Temporal values
        (``date``, ``datetime``, ISO-8601 strings) are converted to
        epoch-milliseconds (UTC) to align with ferrum's temporal axis scale.
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
    from ferrum.annotation.coords import norm
    from ferrum.annotation.primitives import AnnotationLine

    x_coord = _coerce_coord(x)
    x_num = _coerce_coord_to_numeric(x_coord)
    df = pl.DataFrame({"_x": [x_num]})
    kwargs: dict = {}
    if stroke is not None:
        kwargs["stroke"] = stroke
    if stroke_dash is not None:
        kwargs["stroke_dash"] = stroke_dash
    chart = Chart(df).mark_rule(**kwargs).encode(x="_x")
    # Attach annotation primitive: vertical line spanning the full y-axis
    prim = AnnotationLine(
        x1=x_coord,
        y1=norm(0),
        x2=x_coord,
        y2=norm(1),
        stroke=stroke or "#333",
        stroke_width=1,
        dash=stroke_dash,
    )
    return _attach_annotation_primitive(chart, prim)


def annotate_rect(
    x1: _AnnotationCoord,
    x2: _AnnotationCoord,
    y1: _AnnotationCoord,
    y2: _AnnotationCoord,
    *,
    fill: Optional[str] = None,
    opacity: float = 0.1,
    label: Optional[str] = None,
) -> Chart:
    """Shaded rectangle region spanning (x1, y1) to (x2, y2).

    Returns a ``mark_rect`` annotation chart for ``|`` / ``&`` concatenation
    composition; for true overlay/layer, use ``+`` with a chart that shares
    the same DataFrame.

    Parameters
    ----------
    x1 : float, datetime.date, datetime.datetime, or str
        Left x boundary in data coordinates.
    x2 : float, datetime.date, datetime.datetime, or str
        Right x boundary in data coordinates.
    y1 : float, datetime.date, datetime.datetime, or str
        Bottom y boundary in data coordinates.
    y2 : float, datetime.date, datetime.datetime, or str
        Top y boundary in data coordinates.
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
    from ferrum.annotation.primitives import AnnotationRect

    x1_coord, x2_coord = _coerce_coord(x1), _coerce_coord(x2)
    y1_coord, y2_coord = _coerce_coord(y1), _coerce_coord(y2)
    x1_num, x2_num = _coerce_coord_to_numeric(x1_coord), _coerce_coord_to_numeric(x2_coord)
    y1_num, y2_num = _coerce_coord_to_numeric(y1_coord), _coerce_coord_to_numeric(y2_coord)
    df = pl.DataFrame({"_x1": [x1_num], "_x2": [x2_num], "_y1": [y1_num], "_y2": [y2_num]})
    kwargs: dict = {"opacity": opacity}
    if fill is not None:
        kwargs["fill"] = fill
    chart = Chart(df).mark_rect(**kwargs).encode(x="_x1", y="_y1", x2="_x2", y2="_y2")
    # Attach annotation primitive
    prim = AnnotationRect(
        x1=x1_coord,
        y1=y1_coord,
        x2=x2_coord,
        y2=y2_coord,
        fill=fill or "#cccccc",
        opacity=opacity,
        stroke=None,
        corner_radius=0,
    )
    return _attach_annotation_primitive(chart, prim)


def annotate_text(
    x: _AnnotationCoord,
    y: _AnnotationCoord,
    text: str,
    *,
    dx: float = 0,
    dy: float = 0,
    align: str = "center",
    baseline: str = "middle",
    font_size: Optional[float] = None,
    color: Optional[str] = None,
    angle: Optional[float] = None,
) -> Chart:
    """Free-floating text annotation at a fixed (x, y) position.

    Returns a ``mark_text`` chart for ``|`` / ``&`` concatenation composition;
    for true overlay/layer, use ``+`` with a chart that shares the same
    DataFrame.

    Parameters
    ----------
    x : float, datetime.date, datetime.datetime, or str
        X position in data coordinates.  Temporal values are converted to
        epoch-milliseconds (UTC).
    y : float, datetime.date, datetime.datetime, or str
        Y position in data coordinates.  Temporal values are converted to
        epoch-milliseconds (UTC).
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
    from ferrum.annotation.primitives import AnnotationText

    x_coord, y_coord = _coerce_coord(x), _coerce_coord(y)
    x_num, y_num = _coerce_coord_to_numeric(x_coord), _coerce_coord_to_numeric(y_coord)
    df = pl.DataFrame({"_x": [x_num], "_y": [y_num], "_text": [text]})
    kwargs: dict = {"dx": dx, "dy": dy, "align": align, "baseline": baseline}
    if font_size is not None:
        kwargs["font_size"] = font_size
    if color is not None:
        kwargs["fill"] = color
    if angle is not None:
        kwargs["angle"] = angle
    chart = Chart(df).mark_text(**kwargs).encode(x="_x", y="_y", text="_text")
    # Map align to annotation anchor
    anchor_map = {"left": "start", "center": "middle", "right": "end"}
    prim = AnnotationText(
        x=x_coord,
        y=y_coord,
        text=text,
        font_size=font_size or 12,
        color=color or "#333",
        anchor=anchor_map.get(align, align),
        baseline=baseline,
        angle=angle or 0,
        dx=dx,
        dy=dy,
        z="above_marks",
    )
    return _attach_annotation_primitive(chart, prim)


def annotate_abline(
    slope: float,
    intercept: float,
    *,
    stroke: str = "black",
    stroke_width: float = 1.0,
    stroke_dash: "list[float] | None" = None,
    opacity: float = 1.0,
) -> Chart:
    """Draw the line y = slope * x + intercept across the full x extent.

    Returns a two-point ``mark_line`` chart whose x range spans ``[-1e6,
    1e6]``; the Rust renderer clips the line to the plot area automatically.
    The chart is suitable for layering with ``+``::

        scatter + fm.annotate_abline(slope=1.0, intercept=0.0, stroke="gray")

    Parameters
    ----------
    slope : float
        Line slope (rise over run).
    intercept : float
        Y-intercept (value of y when x = 0).
    stroke : str, default "black"
        Line color as a CSS color string.
    stroke_width : float, default 1.0
        Line width in pixels.
    stroke_dash : list of float, optional
        SVG dash array, e.g. ``[4, 4]`` for evenly dashed.
    opacity : float, default 1.0
        Line opacity in ``[0, 1]``.

    Returns
    -------
    Chart
        Two-point line chart suitable for ``+`` layering.

    Examples
    --------
    >>> import ferrum as fm
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.1, 0.9]})
    >>> scatter = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    >>> identity = fm.annotate_abline(slope=1.0, intercept=0.0, stroke="gray")
    >>> chart = scatter + identity
    """
    from ferrum.annotation.primitives import AnnotationLine

    x_lo, x_hi = -1e6, 1e6
    df = pl.DataFrame(
        {
            "__abline_x": [x_lo, x_hi],
            "__abline_y": [slope * x_lo + intercept, slope * x_hi + intercept],
        }
    )
    mark_kwargs: dict = {"stroke": stroke, "stroke_width": stroke_width, "opacity": opacity}
    if stroke_dash is not None:
        mark_kwargs["stroke_dash"] = stroke_dash
    chart = Chart(df).mark_line(**mark_kwargs).encode(x="__abline_x:Q", y="__abline_y:Q")
    # Attach annotation primitive: line from extreme data coords (clipped by renderer)
    prim = AnnotationLine(
        x1=x_lo,
        y1=slope * x_lo + intercept,
        x2=x_hi,
        y2=slope * x_hi + intercept,
        stroke=stroke,
        stroke_width=stroke_width,
        dash=stroke_dash,
    )
    return _attach_annotation_primitive(chart, prim)


def annotate_arrow(
    x1: _AnnotationCoord,
    y1: _AnnotationCoord,
    x2: _AnnotationCoord,
    y2: _AnnotationCoord,
    *,
    label: Optional[str] = None,
    label_side: str = "start",
    stroke: Optional[str] = None,
) -> Chart:
    """Draw an arrow from ``(x1, y1)`` to ``(x2, y2)`` with an optional text label.

    Composes a ``mark_segment`` (the arrow shaft) with an optional
    ``annotate_text`` placed at the ``label_side`` endpoint.

    Parameters
    ----------
    x1 : float, datetime.date, datetime.datetime, or str
        Horizontal data coordinate of the arrow start.
    y1 : float, datetime.date, datetime.datetime, or str
        Vertical data coordinate of the arrow start.
    x2 : float, datetime.date, datetime.datetime, or str
        Horizontal data coordinate of the arrow end (tip).
    y2 : float, datetime.date, datetime.datetime, or str
        Vertical data coordinate of the arrow end (tip).
    label : str, optional
        Text to display alongside the arrow.  When omitted, no text is
        rendered.
    label_side : {"start", "end"}, default "start"
        Which end of the arrow to place the label.  ``"start"`` anchors
        the text at ``(x1, y1)``; ``"end"`` places it at ``(x2, y2)``.
    stroke : str, optional
        Hex colour string for the arrow line (e.g. ``"#ff0000"``).
        Inherits the theme's foreground colour when omitted.

    Returns
    -------
    Chart
        Layered chart containing the arrow segment and, when *label* is
        provided, the annotation text.

    Examples
    --------
    Simple unlabelled arrow:

    >>> import ferrum as fm
    >>> fm.annotate_arrow(0.1, 0.5, 0.8, 0.9)

    Arrow with a label at the tip:

    >>> fm.annotate_arrow(0.1, 0.5, 0.8, 0.9, label="threshold", label_side="end")
    """
    # Spec §3.3 lists `arrow=True` for mark_segment but the validator does
    # not yet accept it; emit a plain segment for now.
    x1_coord, y1_coord = _coerce_coord(x1), _coerce_coord(y1)
    x2_coord, y2_coord = _coerce_coord(x2), _coerce_coord(y2)
    x1_num = _coerce_coord_to_numeric(x1_coord)
    y1_num = _coerce_coord_to_numeric(y1_coord)
    x2_num = _coerce_coord_to_numeric(x2_coord)
    y2_num = _coerce_coord_to_numeric(y2_coord)
    df = pl.DataFrame({"_x1": [x1_num], "_y1": [y1_num], "_x2": [x2_num], "_y2": [y2_num]})
    seg_kwargs: dict = {}
    if stroke is not None:
        seg_kwargs["stroke"] = stroke
    arrow_chart = (
        Chart(df)
        .mark_segment(**seg_kwargs)
        .encode(
            x="_x1",
            y="_y1",
            x2="_x2",
            y2="_y2",
        )
    )
    if label is None:
        return arrow_chart
    lx: CoordValue = x1_coord if label_side == "start" else x2_coord
    ly: CoordValue = y1_coord if label_side == "start" else y2_coord
    dx = -6 if label_side == "start" else 6
    align = "right" if label_side == "start" else "left"
    return arrow_chart & annotate_text(lx, ly, label, dx=dx, align=align)
