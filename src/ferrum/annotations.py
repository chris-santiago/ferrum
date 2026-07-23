"""Reference-line, rectangle, and text annotation helpers.

Metric-label classes (AUCLabel, APLabel, BrierLabel, OutlierLabel) and their
private helpers live in ``ferrum._metric_labels``; they are re-exported here
for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import polars as pl

from ferrum.annotation.coords import (
    CoordValue,
    _coerce_coord,
    _coerce_coord_to_numeric,
)
from ferrum.chart import Chart
from ferrum._metric_labels import AUCLabel, APLabel, BrierLabel, OutlierLabel  # noqa: F401

# Type accepted by positional annotation parameters.  This is the same as
# CoordValue: numeric, temporal, ISO-8601 string, ordinal category string,
# or explicit PixelCoord/NormCoord wrapper.
_AnnotationCoord = CoordValue


@dataclass(frozen=True)
class _LineStyle:
    """A single capture of line style feeding both the mark and the primitive.

    ``mark_kwargs`` carries only the style the user explicitly supplied (so the
    mark falls back to its own defaults when omitted, exactly as before).
    ``stroke`` / ``dash`` carry the same style with the primitive's documented
    defaults applied.  Deriving both from one capture keeps the mark-Chart and
    the annotation primitive from drifting.
    """

    mark_kwargs: dict
    stroke: str
    dash: Optional[list]


def _capture_line_style(*, stroke: Optional[str], stroke_dash) -> _LineStyle:
    """Capture the user-supplied reference-line style once.

    The mark receives ``stroke``/``stroke_dash`` only when supplied; the
    primitive receives the defaulted ``stroke`` (``"#333"``) and the raw
    ``stroke_dash`` as its ``dash``.
    """
    mark_kwargs: dict = {}
    if stroke is not None:
        mark_kwargs["stroke"] = stroke
    if stroke_dash is not None:
        mark_kwargs["stroke_dash"] = stroke_dash
    return _LineStyle(mark_kwargs=mark_kwargs, stroke=stroke or "#333", dash=stroke_dash)


def _attach_annotation_primitive(chart: Chart, primitive: object) -> Chart:
    """Set the annotation primitive on a Chart so ``+`` routes through annotations.

    ``primitive`` is usually a single annotation primitive (``AnnotationLine``,
    ``AnnotationText``, etc.), but may also be a ``list`` of primitives — see
    :func:`_attach_primitive_with_label`, which sets a two-element list so a
    labelled reference line or arrow's ``+`` overlay carries both the
    mark-equivalent primitive and its label. ``Chart.__add__`` reads this
    field through the module-level ``ferrum.chart._annotations_from_primitive``
    helper, which accepts either shape.
    """
    chart._annotation_primitive = primitive
    return chart


def _attach_primitive_with_label(
    chart: Chart,
    primitive: object,
    *,
    label: Optional[str],
    label_x: _AnnotationCoord,
    label_y: _AnnotationCoord,
    label_dx: float,
    label_dy: float,
    label_align: str,
) -> Chart:
    """Wire an annotate_* ``Chart`` for both standalone and ``+`` overlay rendering.

    ``primitive`` is the annotation-primitive equivalent of the chart's own
    mark layer (``AnnotationLine`` for a ``mark_rule`` reference line,
    ``AnnotationArrow`` for a ``mark_segment`` arrow, etc.).

    Unlabelled (``label is None``): attaches ``primitive`` as the sole
    ``_annotation_primitive`` — unchanged from the pre-label behavior.

    Labelled: the chart's own mark layer is kept (so standalone rendering
    still draws it), the label is added as a real ``_annotations`` entry (so
    it also renders standalone), and ``_annotation_primitive`` is set to
    ``[primitive, text_primitive]`` so the ``+`` overlay path — which
    discards the helper chart's mark layers and reads only
    ``_annotation_primitive`` — carries both the mark and its label. Neither
    path draws the mark twice: standalone draws it via the chart's own mark
    layer, not ``_annotation_primitive`` (dormant until ``+``); overlay
    draws it via ``_annotation_primitive``, not the mark layer (dropped by
    ``Chart.__add__``).
    """
    if label is None:
        return _attach_annotation_primitive(chart, primitive)

    from ferrum.annotation.container import Annotate

    text_primitive = annotate_text(
        label_x, label_y, label, dx=label_dx, dy=label_dy, align=label_align
    )._annotation_primitive
    chart._annotations = chart._annotations + [Annotate(text_primitive)]
    chart._annotation_primitive = [primitive, text_primitive]
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
        Text to display alongside the line, near its right end, inset from
        the axis edge and offset above the rule.  Renders both standalone
        (``.to_svg()``) and when layered with ``+`` onto another chart.
        When omitted, no text is rendered.
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

    Labelled reference line:

    >>> fm.annotate_hline(y=0.0, label="baseline")
    """
    from ferrum.annotation.coords import norm
    from ferrum.annotation.primitives import AnnotationLine

    y_coord = _coerce_coord(y)
    y_num = _coerce_coord_to_numeric(y_coord)
    df = pl.DataFrame({"_y": [y_num]})
    # Single style capture: collect only the explicitly-supplied style once,
    # then derive the mark kwargs (only-if-supplied) and the primitive fields
    # (defaulted) from it so the two representations cannot drift.
    style = _capture_line_style(stroke=stroke, stroke_dash=stroke_dash)
    chart = Chart(df).mark_rule(**style.mark_kwargs).encode(y="_y")
    # Attach annotation primitive: horizontal line spanning the full x-axis
    prim = AnnotationLine(
        x1=norm(0),
        y1=y_coord,
        x2=norm(1),
        y2=y_coord,
        stroke=style.stroke,
        stroke_width=1,
        dash=style.dash,
    )
    # Label near the right end of the line: inset from the right axis edge
    # (dx=-6, align="right") and offset above the rule (dy=-6).
    return _attach_primitive_with_label(
        chart,
        prim,
        label=label,
        label_x=norm(1),
        label_y=y_coord,
        label_dx=-6,
        label_dy=-6,
        label_align="right",
    )


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
        Text to display alongside the line, near its top, inset from the
        axis edge and offset to the side of the rule so it doesn't overlap.
        Renders both standalone (``.to_svg()``) and when layered with ``+``
        onto another chart.  When omitted, no text is rendered.
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

    Labelled reference line:

    >>> fm.annotate_vline(x=2020, label="cutoff")
    """
    from ferrum.annotation.coords import norm
    from ferrum.annotation.primitives import AnnotationLine

    x_coord = _coerce_coord(x)
    x_num = _coerce_coord_to_numeric(x_coord)
    df = pl.DataFrame({"_x": [x_num]})
    style = _capture_line_style(stroke=stroke, stroke_dash=stroke_dash)
    chart = Chart(df).mark_rule(**style.mark_kwargs).encode(x="_x")
    # Attach annotation primitive: vertical line spanning the full y-axis
    prim = AnnotationLine(
        x1=x_coord,
        y1=norm(0),
        x2=x_coord,
        y2=norm(1),
        stroke=style.stroke,
        stroke_width=1,
        dash=style.dash,
    )
    # Label near the top of the line: inset from the top axis edge (dy=6)
    # and offset to the right of the rule (dx=6, align="left") so the text
    # doesn't overlap the line.
    return _attach_primitive_with_label(
        chart,
        prim,
        label=label,
        label_x=x_coord,
        label_y=norm(1),
        label_dx=6,
        label_dy=6,
        label_align="left",
    )


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
    # Single style capture: mark gets fill only-if-supplied; primitive defaults.
    mark_kwargs: dict = {"opacity": opacity}
    if fill is not None:
        mark_kwargs["fill"] = fill
    chart = Chart(df).mark_rect(**mark_kwargs).encode(x="_x1", y="_y1", x2="_x2", y2="_y2")
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


# Horizontal-alignment vocabularies.  The mark_text mark speaks the
# ``left/center/right`` ``align`` vocabulary; the annotation primitive (and
# ``annotation.text``) speak the SVG ``start/middle/end`` ``anchor`` vocabulary.
# ``annotate_text`` accepts either keyword and resolves to both.
_ALIGN_TO_ANCHOR = {"left": "start", "center": "middle", "right": "end"}
_ANCHOR_TO_ALIGN = {"start": "left", "middle": "center", "end": "right"}


def _resolve_text_alignment(*, anchor: Optional[str], align: Optional[str]) -> tuple[str, str]:
    """Resolve the ``anchor``/``align`` pair to ``(mark_align, primitive_anchor)``.

    ``anchor`` is the canonical SVG vocabulary (``start``/``middle``/``end``,
    matching ``annotation.text``); ``align`` is the back-compat alias
    (``left``/``center``/``right``).  Exactly one vocabulary drives the result:

    - Neither supplied → ``("center", "middle")`` (the historical default).
    - ``align`` only → mark gets ``align``; primitive gets ``_ALIGN_TO_ANCHOR``.
    - ``anchor`` only → mark gets ``_ANCHOR_TO_ALIGN``; primitive gets ``anchor``.
    - Both supplied → ``ValueError`` (the canonical and the alias conflict).
    """
    if anchor is not None and align is not None:
        raise ValueError(
            "annotate_text() received both 'anchor' and 'align'; pass only one "
            "('anchor' is the canonical SVG vocabulary, 'align' is the alias)."
        )
    if anchor is not None:
        return _ANCHOR_TO_ALIGN.get(anchor, anchor), anchor
    if align is not None:
        return align, _ALIGN_TO_ANCHOR.get(align, align)
    return "center", "middle"


def annotate_text(
    x: _AnnotationCoord,
    y: _AnnotationCoord,
    text: str,
    *,
    dx: float = 0,
    dy: float = 0,
    anchor: Optional[str] = None,
    align: Optional[str] = None,
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
    anchor : str, optional
        Horizontal text anchor in the SVG vocabulary (matching
        `text`): ``"start"``, ``"middle"``, or
        ``"end"``.  This is the canonical keyword.  When neither ``anchor`` nor
        ``align`` is supplied the anchor defaults to ``"middle"`` (centered).
    align : str, optional
        Deprecated alias for ``anchor`` in the ``"left"``/``"center"``/
        ``"right"`` vocabulary.  Mapped to ``anchor`` via
        ``{left: start, center: middle, right: end}``.  Supplying both
        ``anchor`` and ``align`` raises ``ValueError``.
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

    mark_align, prim_anchor = _resolve_text_alignment(anchor=anchor, align=align)

    x_coord, y_coord = _coerce_coord(x), _coerce_coord(y)
    x_num, y_num = _coerce_coord_to_numeric(x_coord), _coerce_coord_to_numeric(y_coord)
    df = pl.DataFrame({"_x": [x_num], "_y": [y_num], "_text": [text]})
    kwargs: dict = {"dx": dx, "dy": dy, "align": mark_align, "baseline": baseline}
    if font_size is not None:
        kwargs["font_size"] = font_size
    if color is not None:
        kwargs["fill"] = color
    if angle is not None:
        kwargs["angle"] = angle
    chart = Chart(df).mark_text(**kwargs).encode(x="_x", y="_y", text="_text")
    prim = AnnotationText(
        x=x_coord,
        y=y_coord,
        text=text,
        font_size=font_size or 12,
        color=color or "#333",
        anchor=prim_anchor,
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

    Returns a single ``mark_segment`` (the arrow shaft) ``Chart``, with an
    optional ``annotate_text`` label placed at the ``label_side`` endpoint.
    Renders both standalone (``.to_svg()``) and when layered with ``+`` onto
    another chart; when omitted, no text is rendered.

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
        Single-panel chart containing the arrow segment and, when *label*
        is provided, the annotation text — suitable for ``+`` layering.

    Examples
    --------
    Simple unlabelled arrow:

    >>> import ferrum as fm
    >>> fm.annotate_arrow(0.1, 0.5, 0.8, 0.9)

    Arrow with a label at the tip:

    >>> fm.annotate_arrow(0.1, 0.5, 0.8, 0.9, label="threshold", label_side="end")
    """
    from ferrum.annotation.primitives import AnnotationArrow

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
        # No annotation primitive attached: unlabelled `+` overlay merges the
        # `mark_segment` layer directly, same as before this primitive
        # machinery existed. `AnnotationArrow` hardcodes a stroke default
        # ("#333") the way `AnnotationLine` does for hline/vline, whereas the
        # mark falls back to the chart's *themed* stroke color when `stroke`
        # is omitted — routing the unlabelled case through the primitive
        # would silently override that themed default, so it stays on the
        # mark-merge path.
        return arrow_chart
    lx: CoordValue = x1_coord if label_side == "start" else x2_coord
    ly: CoordValue = y1_coord if label_side == "start" else y2_coord
    dx = -6 if label_side == "start" else 6
    align = "right" if label_side == "start" else "left"
    prim = AnnotationArrow(
        x=x1_coord,
        y=y1_coord,
        x2=x2_coord,
        y2=y2_coord,
        stroke=stroke or "#333",
        stroke_width=1.5,
        head_size=8,
    )
    return _attach_primitive_with_label(
        arrow_chart,
        prim,
        label=label,
        label_x=lx,
        label_y=ly,
        label_dx=dx,
        label_dy=0,
        label_align=align,
    )
