"""Appearance encoding channels (Color, Fill, Stroke, Opacity, Size, Shape, Angle).

Each channel references one named role from `_honored`;
``_honored_kwargs`` is the single, machine-readable source of truth for which
kwargs the channel honors (see the ``ChannelBase`` docstring for the contract).
"""

from __future__ import annotations

from ferrum.encoding._honored import (
    APPEARANCE_BASE,
    APPEARANCE_COLOR,
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
        Explicit scale override (e.g. ``fm.ContinuousScheme("viridis")`` or
        ``fm.continuous_palette("viridis")``).
    title : str, optional
        Legend title override.  When omitted the field name is used.

    Notes
    -----
    ``legend`` is honored: passing ``legend={"disabled": True}``,
    ``legend=None``, or ``legend=False`` suppresses the color legend in the
    rendered SVG.  ``sort`` is honored: pass a list of strings to set the
    explicit domain order for the color scale.  ``condition`` is honored:
    pass a ``ConditionalSpec`` built via ``sel.when(...).otherwise(...)`` (or
    ``fm.when(sel).then(...).otherwise(...)``) to switch color based on
    selection state -- see ``Chart.encode()``'s condition= translation
    (``ferrum-spec.md`` §3.10).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", color=fm.Color("cyl"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", color=fm.Color("origin", scheme="set1"))
    """

    _channel_name = "color"

    # {"type","legend","scale","title","sort","scheme"} plus "condition",
    # added by ChannelBase.__init_subclass__ -- "color" is a real conditional
    # channel (ferrum.encoding._channel_policy._CONDITIONAL_CHANNELS).
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
    representative values from the field's domain.  ``condition`` is honored:
    pass a ``ConditionalSpec`` built via ``sel.when(...).otherwise(...)`` (or
    ``fm.when(sel).then(...).otherwise(...)``) to switch size based on
    selection state -- see ``Chart.encode()``'s condition= translation
    (``ferrum-spec.md`` §3.10).

    When ``size`` and ``color`` are both mapped to the same field, ferrum merges
    them into a single combined legend block.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", size=fm.Size("weight"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", size=fm.Size("pop", legend=None))
    """

    _channel_name = "size"

    # {"type","legend","scale","title"} plus "condition", added by
    # ChannelBase.__init_subclass__ -- "size" is a real conditional channel.
    _honored_kwargs = APPEARANCE_FULL


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
    by the per-category aggregate of another channel.

    ``condition`` is **not** honored: ``shape`` is not a conditional target,
    and ``encode(shape=<ConditionalSpec>)`` / ``Shape(field, condition=...)``
    raise ``ValueError`` at ``encode()`` time naming the channels that are.
    The wire has a ``shape`` channel name but no shape *value*: a shape
    conditional could only carry a field reference, which the runtime applies
    as a no-op, so it would serialize cleanly and then do nothing. An earlier
    version of this docstring said ``condition`` was honored, written on the
    assumption that reaching the wire meant working; it did not. Refusing
    beats a kwarg that quietly does nothing. Filed follow-up:
    ``EncodingValue::Shape`` in Rust, after which this channel can honor
    ``condition`` for real.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", shape=fm.Shape("origin", type_="N"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", shape=fm.Shape("origin", sort="descending"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", shape=fm.Shape("origin", legend=None))
    """

    _channel_name = "shape"

    # {"type","legend","scale","title","sort"} — no "condition": a shape
    # conditional serializes and then applies as a no-op (no
    # EncodingValue::Shape), so ChannelBase.__init_subclass__ does not add
    # the bit (_channel_policy._CONDITIONAL_REFUSALS["shape"]) and encode()
    # refuses it. StrokeDash shares this same role set and DOES gain the bit
    # (a real EncodingValue::StrokeDash exists) -- see APPEARANCE_SORT's
    # docstring; the derivation, not a second role set, is what keeps the two
    # channels apart.
    _honored_kwargs = APPEARANCE_SORT


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
    ``condition`` is honored: pass a ``ConditionalSpec`` built via
    ``sel.when(...).otherwise(...)`` (or ``fm.when(sel).then(...).otherwise(...)``)
    to switch opacity based on selection state -- see ``Chart.encode()``'s
    condition= translation (``ferrum-spec.md`` §3.10).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", opacity=fm.Opacity("year"))
    """

    _channel_name = "opacity"

    # {"type","legend","scale","title"} plus "condition", added by
    # ChannelBase.__init_subclass__ -- "opacity" is a real conditional channel.
    _honored_kwargs = APPEARANCE_FULL


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
    scheme : str, optional
        Named categorical or continuous color scheme (e.g. ``"tableau10"``,
        ``"viridis"``), validated against the palette registry.
    scale : Scale, optional
        Explicit scale override (e.g. ``fm.ContinuousScheme("viridis")``).
    title : str, optional
        Legend title override.  When omitted the field name is used.

    Notes
    -----
    ``legend`` is honored: passing ``legend={"disabled": True}``, ``legend=None``,
    or ``legend=False`` suppresses the fill legend in the rendered SVG.
    ``condition`` is honored: pass a ``ConditionalSpec`` built via
    ``sel.when(...).otherwise(...)`` (or ``fm.when(sel).then(...).otherwise(...)``)
    to switch fill based on selection state -- see ``Chart.encode()``'s
    condition= translation (``ferrum-spec.md`` §3.10).  The conditional lands
    on the ``color`` channel, the same target ``fill`` aliases to for its base
    encoding, so ``encode(fill=fm.Fill("c", condition=cond))`` and
    ``encode(fill=cond)`` produce the identical wire conditional.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", fill=fm.Fill("origin"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", fill=fm.Fill("origin", scheme="set1"))
    """

    _channel_name = "fill"

    # {"type","legend","scale","title","scheme"} plus "condition", added by
    # ChannelBase.__init_subclass__ -- "fill" aliases to "color" and gains
    # the bit. Stroke shares this same role set and does NOT gain it -- see
    # APPEARANCE_SCHEME's docstring.
    _honored_kwargs = APPEARANCE_SCHEME


class Stroke(ChannelBase):
    """Stroke color channel — maps a field to mark stroke (outline) color.

    Distinct from ``fill``, which controls only the interior color.

    Parameters
    ----------
    field : str
        Column name to map to stroke color.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.
    scheme : str, optional
        Named categorical or continuous color scheme (e.g. ``"tableau10"``,
        ``"viridis"``), validated against the palette registry.
    scale : Scale, optional
        Explicit scale override (e.g. ``fm.ContinuousScheme("viridis")``).
    title : str, optional
        Legend title override.  When omitted the field name is used.

    Notes
    -----
    ``legend`` is honored: passing ``legend={"disabled": True}``, ``legend=None``,
    or ``legend=False`` suppresses the stroke legend in the rendered SVG.

    ``condition`` is **not** honored: ``stroke`` is not a conditional target,
    and ``encode(stroke=<ConditionalSpec>)`` / ``Stroke(field, condition=...)``
    raise ``ValueError`` at ``encode()`` time naming the channels that are.
    The reason is a missing renderer capability rather than a policy choice:
    a conditional's only colour value is ``EncodingValue::Color``, which the
    WASM runtime applies to the mark's **fill** (``apply_value_common``;
    ``FillStrokeView`` has no stroke-colour member), so carrying a stroke
    conditional onto ``color`` -- the channel ``stroke`` aliases to for its
    base encoding -- would recolour the interior instead of the outline.
    Refusing beats silently changing the wrong property. ``Fill`` is
    unaffected and does honor ``condition`` (``Color`` *is* the fill).
    Filed follow-up: ``EncodingValue::StrokeColor`` in Rust, after which this
    channel can honor ``condition`` for real.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", stroke=fm.Stroke("origin"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", stroke=fm.Stroke("origin", scheme="set1"))
    """

    _channel_name = "stroke"

    # {"type","legend","scale","title","scheme"} — no "condition": a stroke
    # conditional would repaint the fill (no EncodingValue::StrokeColor), so
    # ChannelBase.__init_subclass__ does not add the bit
    # (_channel_policy._CONDITIONAL_REFUSALS["stroke"]) and encode() refuses
    # it. Fill shares this same role set and DOES gain the bit -- the
    # derivation, not a second role set, is what keeps the two channels apart.
    _honored_kwargs = APPEARANCE_SCHEME


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
    ``legend`` is honored: passing ``legend=None`` or ``legend=False`` suppresses
    the legend in the rendered SVG.  ``scale`` (explicit scale override) and
    ``title`` (legend title) are also honored.  ``condition`` is honored: pass a
    ``ConditionalSpec`` built via ``sel.when(...).otherwise(...)`` (or
    ``fm.when(sel).then(...).otherwise(...)``) to switch fill opacity based on
    selection state -- see ``Chart.encode()``'s condition= translation
    (``ferrum-spec.md`` §3.10).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", fill_opacity=fm.FillOpacity("density"))
    """

    _channel_name = "fill_opacity"

    # {"type","legend","scale","title"} plus "condition", added by
    # ChannelBase.__init_subclass__ -- fill_opacity is a conditional channel
    # on the wire and its condition= works end to end.
    _honored_kwargs = APPEARANCE_FULL


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
    scale : Scale, optional
        Explicit scale override.
    title : str, optional
        Legend title override.

    Notes
    -----
    ``legend`` is honored: passing ``legend=None`` or ``legend=False`` suppresses
    the legend in the rendered SVG.  ``scale`` (explicit scale override) and
    ``title`` (legend title) are also honored.  ``condition`` is honored: pass a
    ``ConditionalSpec`` built via ``sel.when(...).otherwise(...)`` (or
    ``fm.when(sel).then(...).otherwise(...)``) to switch stroke opacity based on
    selection state -- see ``Chart.encode()``'s condition= translation
    (``ferrum-spec.md`` §3.10).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", stroke_opacity=fm.StrokeOpacity("year"))
    """

    _channel_name = "stroke_opacity"

    # {"type","legend","scale","title"} plus "condition", added by
    # ChannelBase.__init_subclass__ -- stroke_opacity is a conditional channel
    # on the wire and its condition= works end to end.
    _honored_kwargs = APPEARANCE_FULL


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
    ``legend`` is honored: passing ``legend=None`` or ``legend=False`` suppresses
    the legend in the rendered SVG.  ``condition`` is honored: pass a
    ``ConditionalSpec`` built via ``sel.when(...).otherwise(...)`` (or
    ``fm.when(sel).then(...).otherwise(...)``) to switch stroke width based on
    selection state -- see ``Chart.encode()``'s condition= translation
    (``ferrum-spec.md`` §3.10).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", stroke_width=fm.StrokeWidth("importance"))
    """

    _channel_name = "stroke_width"

    # {"type","legend"} plus "condition", added by
    # ChannelBase.__init_subclass__ -- stroke_width is a conditional channel
    # on the wire and its condition= works end to end.
    _honored_kwargs = APPEARANCE_BASE


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
    scale : Scale, optional
        Explicit scale override.
    title : str, optional
        Legend title override.

    Notes
    -----
    ``legend`` is honored: passing ``legend=None`` or ``legend=False`` suppresses
    the legend in the rendered SVG.  ``scale`` (explicit scale override) and
    ``title`` (legend title) are also honored.  ``sort`` is honored: pass
    ``"ascending"``, ``"descending"``, a list of strings for an explicit domain
    order, or a channel shorthand such as ``"-y"`` to sort by the per-category
    aggregate of another channel — mirroring ``Shape``, since the Rust
    ``build_stroke_dash_scale`` domain builder already reads it.  ``condition``
    is honored: pass a ``ConditionalSpec`` built via ``sel.when(...).otherwise(...)``
    (or ``fm.when(sel).then(...).otherwise(...)``) to switch stroke dash based
    on selection state -- see ``Chart.encode()``'s condition= translation
    (``ferrum-spec.md`` §3.10).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", stroke_dash=fm.StrokeDash("group", type_="N"))
    >>> fm.Chart(df).encode(x="x", y="y", stroke_dash=fm.StrokeDash("group", sort="descending"))
    """

    _channel_name = "stroke_dash"

    # {"type","legend","scale","title","sort"} plus "condition", added by
    # ChannelBase.__init_subclass__ -- stroke_dash is a real conditional
    # channel. sort added 2026-09-01 (batch-A T12 spec-review addendum, spec
    # §4.3) so a user's sort= reaches the wire; Rust's build_stroke_dash_scale
    # already read it. Shape shares this same role set and does NOT gain
    # "condition" -- see APPEARANCE_SORT's docstring.
    _honored_kwargs = APPEARANCE_SORT


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
    ``legend`` is honored: passing ``legend=None`` or ``legend=False`` suppresses
    the legend in the rendered SVG.  ``condition`` is honored: pass a
    ``ConditionalSpec`` built via ``sel.when(...).otherwise(...)`` (or
    ``fm.when(sel).then(...).otherwise(...)``) to switch rotation based on
    selection state -- see ``Chart.encode()``'s condition= translation
    (``ferrum-spec.md`` §3.10).

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", angle=fm.Angle("direction"))
    """

    _channel_name = "angle"

    # {"type","legend"} plus "condition", added by
    # ChannelBase.__init_subclass__ -- angle is a conditional channel on the
    # wire and its condition= works end to end.
    _honored_kwargs = APPEARANCE_BASE
