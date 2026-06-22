"""Appearance encoding channels (Color, Fill, Stroke, Opacity, Size, Shape, Angle).

Each channel references one named role from :mod:`ferrum.encoding._honored`;
``_honored_kwargs`` is the single, machine-readable source of truth for which
kwargs the channel honors (see the ``ChannelBase`` docstring for the contract).
"""

from __future__ import annotations

from ferrum.encoding._honored import (
    APPEARANCE_BASE,
    APPEARANCE_COLOR,
    APPEARANCE_CONDITION,
    APPEARANCE_FULL,
    APPEARANCE_SCHEME,
    APPEARANCE_SORT,
)
from ferrum.encoding.base import ChannelBase


# Phase 8a renders these (added to scale_resolve in Task 8):
class Color(ChannelBase):
    """Color encoding channel — maps a field to mark color (fill and stroke).

    Parameters
    ----------
    field : str
        Column name to map to color.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type: quantitative, nominal, ordinal, temporal. Inferred from
        the column dtype when omitted.
    scheme : str, optional
        Named categorical color scheme (e.g. ``"tableau10"``, ``"set1"``).
    scale : Scale, optional
        Explicit scale override (e.g. ``ColorScale.Continuous("viridis")``).
    title : str, optional
        Legend title override.  When omitted the field name is used.

    Notes
    -----
    ``legend`` is honored: passing ``legend={"disabled": True}``,
    ``legend=None``, or ``legend=False`` suppresses the color legend in the
    rendered SVG.  ``sort`` is honored: pass a list of strings to set the
    explicit domain order for the color scale.  ``condition`` is accepted but
    reserved for future use.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", color=fm.Color("cyl"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", color=fm.Color("origin", scheme="set1"))
    """

    _channel_name = "color"

    # {"type","legend","scale","title","condition","sort","scheme"}
    _honored_kwargs = APPEARANCE_COLOR


class Size(ChannelBase):
    """Size channel — maps a field to mark size (point area in square pixels).

    For point marks the size value corresponds to the area of the mark in
    square pixels.

    Parameters
    ----------
    field : str
        Column name to map to size.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.
    scale : Scale, optional
        Explicit scale override.
    title : str, optional
        Legend title override.

    Notes
    -----
    ``legend`` is honored: passing ``legend=None`` or ``legend=False`` suppresses
    the size legend in the rendered SVG.  ``legend=Legend(title=..., format=...)``
    customizes the legend title and numeric format string.  When ``legend`` is
    omitted, a graduated size legend is rendered automatically showing ~5 round
    representative values from the field's domain.  ``condition`` is accepted but
    reserved for future use.

    When ``size`` and ``color`` are both mapped to the same field, ferrum merges
    them into a single combined legend block.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", size=fm.Size("weight"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", size=fm.Size("pop", legend=None))
    """

    _channel_name = "size"

    _honored_kwargs = APPEARANCE_CONDITION  # {"type","legend","scale","title","condition"}


class Shape(ChannelBase):
    """Shape channel — maps a categorical field to point shape.

    Supported shapes include ``"circle"``, ``"square"``, ``"triangle"``,
    ``"cross"``, ``"diamond"``, and ``"star"``.

    Parameters
    ----------
    field : str
        Column name (should be nominal or ordinal) to map to shape.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted; ``"N"`` is
        the most common choice.
    scale : Scale, optional
        Explicit scale override.
    title : str, optional
        Legend title override.

    Notes
    -----
    ``legend`` is honored: passing ``legend=None`` or ``legend=False`` suppresses
    the shape legend in the rendered SVG.  ``legend=Legend(title=..., format=...)``
    customizes the legend title.  When ``legend`` is omitted, a categorical shape
    legend is rendered automatically showing one glyph per distinct category value.
    ``sort`` is honored: pass ``"ascending"``, ``"descending"``, a list of strings
    for an explicit domain order, or a channel shorthand such as ``"-y"`` to sort
    by the per-category aggregate of another channel.  ``condition`` is accepted
    but reserved for future use.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", shape=fm.Shape("origin", type_="N"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", shape=fm.Shape("origin", sort="descending"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", shape=fm.Shape("origin", legend=None))
    """

    _channel_name = "shape"

    _honored_kwargs = APPEARANCE_SORT  # {"type","legend","scale","title","condition","sort"}


class Opacity(ChannelBase):
    """Opacity channel — maps a field to overall mark opacity.

    Controls both fill and stroke opacity simultaneously.  Values are mapped
    to the range ``[0, 1]``.

    Parameters
    ----------
    field : str
        Column name to map to opacity.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.
    scale : Scale, optional
        Explicit scale override.
    title : str, optional
        Legend title override.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", opacity=fm.Opacity("year"))
    """

    _channel_name = "opacity"

    _honored_kwargs = APPEARANCE_CONDITION  # {"type","legend","scale","title","condition"}


# Deferred to Phase 9:
class Fill(ChannelBase):
    """Fill color channel — maps a field to filled-mark fill color.

    Distinct from ``stroke``, which controls only the outline color.  When
    both ``Fill`` and ``Color`` are encoded, ``Fill`` takes precedence for the
    interior of the mark.

    Parameters
    ----------
    field : str
        Column name to map to fill color.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", fill=fm.Fill("origin"))
    """

    _channel_name = "fill"

    _honored_kwargs = APPEARANCE_SCHEME  # {"type","legend","scale","title","condition","scheme"}


class Stroke(ChannelBase):
    """Stroke color channel — maps a field to mark stroke (outline) color.

    Distinct from ``fill``, which controls only the interior color.

    Parameters
    ----------
    field : str
        Column name to map to stroke color.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", stroke=fm.Stroke("origin"))
    """

    _channel_name = "stroke"

    _honored_kwargs = APPEARANCE_SCHEME  # {"type","legend","scale","title","condition","scheme"}


class FillOpacity(ChannelBase):
    """Fill-opacity channel — maps a field to fill opacity.

    Independent of stroke opacity; controls only how transparent the interior
    of a mark is.  Values are mapped to the range ``[0, 1]``.

    Parameters
    ----------
    field : str
        Column name to map to fill opacity.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", fill_opacity=fm.FillOpacity("density"))
    """

    _channel_name = "fill_opacity"

    _honored_kwargs = APPEARANCE_FULL  # {"type","legend","scale","title"}


class StrokeOpacity(ChannelBase):
    """Stroke-opacity channel — maps a field to stroke opacity.

    Independent of fill opacity; controls only how transparent the outline of
    a mark is.  Values are mapped to the range ``[0, 1]``.

    Parameters
    ----------
    field : str
        Column name to map to stroke opacity.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", stroke_opacity=fm.StrokeOpacity("year"))
    """

    _channel_name = "stroke_opacity"

    _honored_kwargs = APPEARANCE_BASE  # {"type","legend"}


class StrokeWidth(ChannelBase):
    """Stroke-width channel — maps a field to stroke width in pixels.

    Larger values produce thicker mark outlines.

    Parameters
    ----------
    field : str
        Column name to map to stroke width.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", stroke_width=fm.StrokeWidth("importance"))
    """

    _channel_name = "stroke_width"

    _honored_kwargs = APPEARANCE_BASE  # {"type","legend"}


class StrokeDash(ChannelBase):
    """Stroke-dash channel — maps a field to a dash pattern.

    Each level of the field maps to a distinct dash-gap pattern for the mark
    stroke (e.g. solid, dashed, dotted).

    Parameters
    ----------
    field : str
        Column name (typically nominal or ordinal) to map to dash pattern.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", stroke_dash=fm.StrokeDash("group", type_="N"))
    """

    _channel_name = "stroke_dash"

    _honored_kwargs = APPEARANCE_BASE  # {"type","legend"}


class Angle(ChannelBase):
    """Angle channel — maps a field to mark rotation in degrees.

    Rotates each mark around its center point.  A value of ``0`` is
    upright; values increase clockwise.

    Parameters
    ----------
    field : str
        Column name to map to rotation angle.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", angle=fm.Angle("direction"))
    """

    _channel_name = "angle"

    _honored_kwargs = APPEARANCE_BASE  # {"type","legend"}
