"""Appearance encoding channels (Color, Fill, Stroke, Opacity, Size, Shape, Angle)."""
from __future__ import annotations

from ferrum.encoding.base import ChannelBase


_RENDERED_HONORED = frozenset(["type", "scale", "title"])


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
        Named color scheme for categorical data (e.g. ``"tableau10"``,
        ``"set1"``).  Only honored for the ``Color`` channel; other channels
        treat this kwarg as a no-op.
    scale : Scale, optional
        Explicit scale override (e.g. ``ColorScale.Continuous("viridis")``).
    title : str, optional
        Legend title override.  When omitted the field name is used.

    Notes
    -----
    ``legend`` and ``condition`` kwargs are accepted but are reserved for
    future use (no-op today) — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", color=fm.Color("cyl"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", color=fm.Color("origin", scheme="set1"))
    """

    # `scheme` is honored ONLY for Color in Phase 8a (Task 10 wires it through
    # palette.rs::categorical_palette into the renderer's color-scale construction).
    # All other channels treat `scheme` as deferred → warn-once.
    _channel_name = "color"
    _renders_in_phase_8a = True
    # ``legend`` honored in Schwabish SB3 (2026-05-11): passing
    # ``legend=None`` (or ``False``) suppresses the categorical color legend
    # at the renderer level. Used by direct-label diagnostic charts to
    # avoid a redundant legend alongside endpoint-anchored series labels.
    _honored_kwargs = frozenset(["type", "scheme", "scale", "title", "legend"])


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
    ``legend`` and ``condition`` kwargs are accepted but are reserved for
    future use (no-op today) — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", size=fm.Size("weight"))
    """

    _channel_name = "size"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


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
    ``legend`` and ``condition`` kwargs are accepted but are reserved for
    future use (no-op today) — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", shape=fm.Shape("origin", type_="N"))
    """

    _channel_name = "shape"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


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

    Notes
    -----
    ``legend`` and ``condition`` kwargs are accepted but are reserved for
    future use (no-op today) — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", opacity=fm.Opacity("year"))
    """

    _channel_name = "opacity"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


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

    Notes
    -----
    ``scale``, ``title``, ``legend``, and ``condition`` kwargs are accepted
    but are reserved for future use (no-op today) — they trigger a one-time
    deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", fill=fm.Fill("origin"))
    """

    _channel_name = "fill"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Stroke(ChannelBase):
    """Stroke color channel — maps a field to mark stroke (outline) color.

    Distinct from ``fill``, which controls only the interior color.

    Parameters
    ----------
    field : str
        Column name to map to stroke color.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    ``scale``, ``title``, ``legend``, and ``condition`` kwargs are accepted
    but are reserved for future use (no-op today) — they trigger a one-time
    deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", stroke=fm.Stroke("origin"))
    """

    _channel_name = "stroke"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


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

    Notes
    -----
    ``scale``, ``title``, ``legend``, and ``condition`` kwargs are accepted
    but are reserved for future use (no-op today) — they trigger a one-time
    deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", fill_opacity=fm.FillOpacity("density"))
    """

    _channel_name = "fill_opacity"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


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

    Notes
    -----
    ``scale``, ``title``, ``legend``, and ``condition`` kwargs are accepted
    but are reserved for future use (no-op today) — they trigger a one-time
    deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", stroke_opacity=fm.StrokeOpacity("year"))
    """

    _channel_name = "stroke_opacity"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class StrokeWidth(ChannelBase):
    """Stroke-width channel — maps a field to stroke width in pixels.

    Larger values produce thicker mark outlines.

    Parameters
    ----------
    field : str
        Column name to map to stroke width.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    ``scale``, ``title``, ``legend``, and ``condition`` kwargs are accepted
    but are reserved for future use (no-op today) — they trigger a one-time
    deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", stroke_width=fm.StrokeWidth("importance"))
    """

    _channel_name = "stroke_width"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


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

    Notes
    -----
    ``scale``, ``title``, ``legend``, and ``condition`` kwargs are accepted
    but are reserved for future use (no-op today) — they trigger a one-time
    deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", stroke_dash=fm.StrokeDash("group", type_="N"))
    """

    _channel_name = "stroke_dash"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


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

    Notes
    -----
    ``scale``, ``title``, ``legend``, and ``condition`` kwargs are accepted
    but are reserved for future use (no-op today) — they trigger a one-time
    deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", angle=fm.Angle("direction"))
    """

    _channel_name = "angle"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])
