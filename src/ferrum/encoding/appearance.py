"""Appearance encoding channels (Color, Fill, Stroke, Opacity, Size, Shape, Angle).

Each channel references one named role from `_honored`;
``_honored_kwargs`` is the single, machine-readable source of truth for which
kwargs the channel honors (see the ``ChannelBase`` docstring for the contract).
"""

from __future__ import annotations

from ferrum.encoding._honored import (
    AGGREGATE,
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
    _legend_surface = "own"

    # {"type","legend","scale","title","sort","scheme"} plus "condition",
    # added by ChannelBase.__init_subclass__ -- "color" is a real conditional
    # channel (ferrum.encoding._channel_policy._CONDITIONAL_CHANNELS).
    _honored_kwargs = APPEARANCE_COLOR | AGGREGATE


class Size(ChannelBase):
    """Size channel — maps a field to a point's area or a line's width.

    The size scale maps each value to an area in square pixels. A point mark
    draws that area. A line mark draws the diameter of a dot with that area,
    ``2 * sqrt(area / pi)`` pixels, at each vertex, so the line tapers along
    its length and a line and a point sharing one size encoding agree: the
    line is as wide as the dot drawn for the same value. A line that binds
    ``size`` cannot also bind ``stroke_width``; rendering such a line raises
    ``ValueError``.

    Parameters
    ----------
    field : str
        Column name to map to size.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.
    scale : LinearScale or dict, optional
        Explicit scale override. Its ``range`` is in area units (square
        pixels) for every mark, a line included; an endpoint below 0 draws
        as 0. A linear scale only: ``domain``, ``range``, ``reverse``, ``nice`` and
        ``zero`` are honored, and values outside the domain clamp to the
        nearer end of the range. Any other scale type, ``padding``,
        ``scheme``, or a domain or range whose length is not 2 raises
        ``ValueError`` at construction.
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
    (``ferrum-spec.md`` §3.10). On a line, a conditional size draws the base
    width in static output; in the interactive view each vertex redraws at
    the dot-diameter width of its row's active branch. Every branch must be
    positive on such a line, and a conditional size plus ``stroke_width``
    raises ``ValueError`` as any other size does.

    A field ``size`` and a conditional ``size`` follow one rule under
    ``chart + chart``: each reaches only the layers of the chart it was
    declared on. ``point.encode(size="v") + line`` sizes the points and
    leaves the line untapered, and a point beside a line that declared one
    is not resized by it; the panel still draws one size scale and legend
    over every layer that draws size. To size both, declare it on the
    composition: ``(point + line).encode(size="v")``. A size declared on a
    layered chart as a whole, on a composite such as
    ``mark_line(point=True)``, or on the host of ``Chart.layer()`` reaches
    every layer of that chart that binds none, lines included; one declared
    on a ``Layer``'s own encoding reaches that layer only.

    When ``size`` and ``color`` are both mapped to the same field, ferrum merges
    them into a single combined legend block.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", size=fm.Size("weight"))
    >>> fm.Chart(df).encode(x="hp", y="mpg", size=fm.Size("pop", legend=None))
    >>> fm.Chart(df).mark_line().encode(x="year", y="price", size=fm.Size("volume"))
    """

    _channel_name = "size"
    _legend_surface = "own"
    _numeric_aux_scale = True

    # {"type","legend","scale","title"} plus "condition", added by
    # ChannelBase.__init_subclass__ -- "size" is a real conditional channel.
    _honored_kwargs = APPEARANCE_FULL | AGGREGATE


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
        Explicit scale override.  ``domain`` reorders the category-to-glyph
        assignment, mirroring ``Color``.  ``range`` picks the glyphs
        directly: a list of point-shape names (``"square"``, ``"cross"``,
        ``"diamond"``, ``"triangle-up"``, ``"triangle-down"``, ``"|"``,
        ``"-"``, ...; see ``ferrum.marks.base._VALID_POINT_SHAPES`` for the
        exact vocabulary) parsed all-or-nothing -- an unrecognized entry
        raises ``ValueError`` naming ``Shape(scale=...)`` at construction,
        and a hand-built spec that reaches Rust with the same problem raises
        a typed ``RenderError`` at render (2026-09-23, #123).
    title : str, optional
        Legend title override.

    Notes
    -----
    ``legend`` is honored: passing ``legend=None`` or ``legend=False`` suppresses
    the shape legend in the rendered SVG.  ``legend=Legend(title=..., format=...)``
    customizes the legend title.  When ``legend`` is omitted, a categorical shape
    legend is rendered automatically showing one glyph per distinct category value.
    A ``legend=`` value whose ``orient=`` is anything other than ``"none"`` is
    refused (2026-09-23, #123): ``Shape``'s legend is an auxiliary block that
    always shares the chart's legend position -- place it with
    ``Color(legend=Legend(orient=...))`` or ``configure_legend(orient=...)``.
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
    _legend_surface = "own"

    # {"type","legend","scale","title","sort"} — no "condition": a shape
    # conditional serializes and then applies as a no-op (no
    # EncodingValue::Shape), so ChannelBase.__init_subclass__ does not add
    # the bit (_channel_policy._CONDITIONAL_REFUSALS["shape"]) and encode()
    # refuses it. StrokeDash shares this same role set and DOES gain the bit
    # (a real EncodingValue::StrokeDash exists) -- see APPEARANCE_SORT's
    # docstring; the derivation, not a second role set, is what keeps the two
    # channels apart.
    # Shape's legend is an auxiliary block (2026-09-23, #123,
    # encoding.base._aux_legend_channels(), read from Rust): it always
    # shares the chart's legend position, so an explicit orient= other than
    # "none" is refused by ChannelBase._validate_legend_value.
    _honored_kwargs = APPEARANCE_SORT

    def _validate(self) -> None:
        """Eagerly validate an explicit ``scale=`` range against the point-shape table.

        Mirrors ``mark_point(shape=...)``'s own gate
        (``ferrum.marks.base._VALID_POINT_SHAPES``) so an unrecognized shape
        name in ``Shape(scale=...)``'s range is refused at construction,
        naming ``Shape(scale=...)``, rather than reaching Rust silently
        (2026-09-23, #123, user ruling Q1). Domain-only scales, and scales
        with no range at all, pass through unchanged -- Rust's
        ``build_shape_scale`` resolves the categorical domain and assigns
        palette glyphs the same way it always has when no explicit range is
        given.

        Each entry goes through :func:`~ferrum._validate.validate_choice`, the
        same membership check ``mark_point(shape=...)`` runs, against a sorted
        tuple so a non-string entry is refused with the same wording instead
        of an unhashable-operand ``TypeError``.
        """
        super()._validate()
        range_ = self._explicit_scale_range()
        if not isinstance(range_, (list, tuple)):
            return
        from ferrum._validate import validate_choice
        from ferrum.marks.base import _VALID_POINT_SHAPES

        shapes = tuple(sorted(_VALID_POINT_SHAPES))
        for entry in range_:
            validate_choice("Shape(scale=...)", "range entry", entry, shapes)


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
    scale : LinearScale or dict, optional
        Explicit scale override: ``domain`` sets the data extent and
        ``range`` the alphas it maps onto, each endpoint clamped to
        ``[0, 1]``. A linear scale only: ``domain``, ``range``, ``reverse``, ``nice`` and
        ``zero`` are honored, and values outside the domain clamp to the
        nearer end of the range. Any other scale type, ``padding``,
        ``scheme``, or a domain or range whose length is not 2 raises
        ``ValueError`` at construction.
    title : str, optional
        Legend title override.  When omitted the field name is used.

    Notes
    -----
    ``condition`` is honored: pass a ``ConditionalSpec`` built via
    ``sel.when(...).otherwise(...)`` (or ``fm.when(sel).then(...).otherwise(...)``)
    to switch opacity based on selection state -- see ``Chart.encode()``'s
    condition= translation (``ferrum-spec.md`` §3.10).

    ``legend`` is honored: a field bound to opacity draws a legend of
    swatches at round values of the domain, each drawn at the alpha the
    scale maps it to. ``legend=None`` or ``legend=False`` suppresses it and
    keeps the mapped alphas; ``legend=Legend(title=..., format=...)``
    customizes it. The legend is an auxiliary block that shares the chart's
    legend position, so a ``legend=`` value whose ``orient=`` is anything
    other than ``"none"`` is refused.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", opacity=fm.Opacity("year"))
    """

    _channel_name = "opacity"
    _legend_surface = "own"
    _numeric_aux_scale = True

    # {"type","legend","scale","title"} plus "condition", derived from the
    # conditional vocabulary.
    _honored_kwargs = APPEARANCE_FULL | AGGREGATE


# Deferred to Phase 9:
class Fill(ChannelBase):
    """Fill color channel — maps a field to filled-mark fill color.

    Distinct from ``stroke``, which controls only the outline color.  When
    both ``Fill`` and ``Color`` are encoded, ``Color`` decides the interior of
    the mark and the ``Fill`` encoding is dropped without a warning.

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
    _legend_surface = "own"

    # {"type","legend","scale","title","scheme"} plus "condition", added by
    # ChannelBase.__init_subclass__ -- "fill" aliases to "color" and gains
    # the bit. Stroke shares this same role set and does NOT gain it -- see
    # APPEARANCE_SCHEME's docstring.
    _honored_kwargs = APPEARANCE_SCHEME | AGGREGATE


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
    _legend_surface = "own"

    # {"type","legend","scale","title","scheme"} — no "condition": a stroke
    # conditional would repaint the fill (no EncodingValue::StrokeColor), so
    # ChannelBase.__init_subclass__ does not add the bit
    # (_channel_policy._CONDITIONAL_REFUSALS["stroke"]) and encode() refuses
    # it. Fill shares this same role set and DOES gain the bit -- the
    # derivation, not a second role set, is what keeps the two channels apart.
    _honored_kwargs = APPEARANCE_SCHEME | AGGREGATE


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
    scale : LinearScale or dict, optional
        Explicit scale override: ``domain`` sets the data extent and
        ``range`` the alphas it maps onto, each endpoint clamped to
        ``[0, 1]``. A linear scale only: ``domain``, ``range``, ``reverse``, ``nice`` and
        ``zero`` are honored, and values outside the domain clamp to the
        nearer end of the range. Any other scale type, ``padding``,
        ``scheme``, or a domain or range whose length is not 2 raises
        ``ValueError`` at construction.
    title : str, optional
        Legend title override.  When omitted the field name is used.

    Notes
    -----
    ``condition`` is honored: pass a ``ConditionalSpec`` built via
    ``sel.when(...).otherwise(...)`` (or ``fm.when(sel).then(...).otherwise(...)``)
    to switch fill opacity based on selection state -- see ``Chart.encode()``'s
    condition= translation (``ferrum-spec.md`` §3.10).

    ``legend`` is honored: a field bound to fill opacity draws a legend of
    swatches at round values of the domain, each drawn at the alpha the
    scale maps it to. ``legend=None`` or ``legend=False`` suppresses it and
    keeps the mapped alphas; ``legend=Legend(title=..., format=...)``
    customizes it. The legend is an auxiliary block that shares the chart's
    legend position, so a ``legend=`` value whose ``orient=`` is anything
    other than ``"none"`` is refused.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", fill_opacity=fm.FillOpacity("density"))
    """

    _channel_name = "fill_opacity"
    _legend_surface = "own"
    _numeric_aux_scale = True

    # {"type","legend","scale","title"} plus "condition", derived from the
    # conditional vocabulary.
    _honored_kwargs = APPEARANCE_FULL | AGGREGATE


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
    scale : LinearScale or dict, optional
        Explicit scale override: ``domain`` sets the data extent and
        ``range`` the alphas it maps onto, each endpoint clamped to
        ``[0, 1]``. A linear scale only: ``domain``, ``range``, ``reverse``, ``nice`` and
        ``zero`` are honored, and values outside the domain clamp to the
        nearer end of the range. Any other scale type, ``padding``,
        ``scheme``, or a domain or range whose length is not 2 raises
        ``ValueError`` at construction.
    title : str, optional
        Legend title override.  When omitted the field name is used.

    Notes
    -----
    ``condition`` is honored: pass a ``ConditionalSpec`` built via
    ``sel.when(...).otherwise(...)`` (or ``fm.when(sel).then(...).otherwise(...)``)
    to switch stroke opacity based on selection state -- see ``Chart.encode()``'s
    condition= translation (``ferrum-spec.md`` §3.10).

    ``legend`` is honored: a field bound to stroke opacity draws a legend of
    swatches at round values of the domain, each drawn at the alpha the
    scale maps it to. ``legend=None`` or ``legend=False`` suppresses it and
    keeps the mapped alphas; ``legend=Legend(title=..., format=...)``
    customizes it. The legend is an auxiliary block that shares the chart's
    legend position, so a ``legend=`` value whose ``orient=`` is anything
    other than ``"none"`` is refused.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg", stroke_opacity=fm.StrokeOpacity("year"))
    """

    _channel_name = "stroke_opacity"
    _legend_surface = "own"
    _numeric_aux_scale = True

    # {"type","legend","scale","title"} plus "condition", derived from the
    # conditional vocabulary.
    _honored_kwargs = APPEARANCE_FULL | AGGREGATE


class StrokeWidth(ChannelBase):
    """Stroke-width channel — maps a field to stroke width in pixels.

    A linear scale maps the field's extent onto a 1 to 4 pixel band, so
    larger values draw thicker strokes; values outside an explicit domain
    draw at the nearer end of the range, and a null value draws the mark's
    default width. To draw a column of literal pixel widths, pass a scale
    whose domain equals its range, e.g.
    ``StrokeWidth("w", scale=fm.LinearScale(domain=[1, 6], range=[1, 6]))``.

    Parameters
    ----------
    field : str
        Column name to map to stroke width.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.
    scale : LinearScale or dict, optional
        Explicit scale override: ``domain`` sets the data extent and
        ``range`` the pixel widths it maps onto; an endpoint below 0 draws
        as 0. A linear scale only: ``domain``, ``range``, ``reverse``, ``nice`` and
        ``zero`` are honored, and values outside the domain clamp to the
        nearer end of the range. Any other scale type, ``padding``,
        ``scheme``, or a domain or range whose length is not 2 raises
        ``ValueError`` at construction.
    title : str, optional
        Legend title override.  When omitted the field name is used.

    Notes
    -----
    ``condition`` is honored: pass a ``ConditionalSpec`` built via
    ``sel.when(...).otherwise(...)`` (or ``fm.when(sel).then(...).otherwise(...)``)
    to switch stroke width based on selection state -- see ``Chart.encode()``'s
    condition= translation (``ferrum-spec.md`` §3.10).

    ``legend`` is honored: a field bound to stroke width draws a legend of
    line swatches at round values of the domain, each drawn at the width
    the scale maps it to. ``legend=None`` or ``legend=False`` suppresses it
    and keeps the mapped widths; ``legend=Legend(title=..., format=...)``
    customizes it. The legend is an auxiliary block that shares the chart's
    legend position, so a ``legend=`` value whose ``orient=`` is anything
    other than ``"none"`` is refused.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", stroke_width=fm.StrokeWidth("importance"))
    """

    _channel_name = "stroke_width"
    _legend_surface = "own"
    _numeric_aux_scale = True

    # {"type","legend","scale","title"} plus "condition", derived from the
    # conditional vocabulary.
    _honored_kwargs = APPEARANCE_FULL | AGGREGATE


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
        Explicit scale override.  ``domain`` reorders the category-to-dash
        assignment, mirroring ``Shape``.  ``range`` is **not** honored: the
        StrokeDash categorical scale's range is a fixed set of palette
        patterns, which cannot carry an arbitrary domain-to-range mapping,
        so a ``range`` key raises ``ValueError`` at construction naming the
        fixed palette.
    title : str, optional
        Legend title override.

    Notes
    -----
    ``legend`` is honored: passing ``legend=None`` or ``legend=False`` suppresses
    the legend in the rendered SVG.  ``scale`` (explicit scale override, domain
    only -- see above) and ``title`` (legend title) are also honored.  A
    ``legend=`` value whose ``orient=`` is anything other than ``"none"`` is
    refused (2026-09-23, #123): ``StrokeDash``'s legend is an auxiliary block
    that always shares the chart's legend position -- place it with
    ``Color(legend=Legend(orient=...))`` or ``configure_legend(orient=...)``.
    ``sort`` is honored: pass
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
    _legend_surface = "own"

    # {"type","legend","scale","title","sort"} plus "condition", added by
    # ChannelBase.__init_subclass__ -- stroke_dash is a real conditional
    # channel. sort added 2026-09-01 (batch-A T12 spec-review addendum, spec
    # §4.3) so a user's sort= reaches the wire; Rust's build_stroke_dash_scale
    # already read it. Shape shares this same role set and does NOT gain
    # "condition" -- see APPEARANCE_SORT's docstring. Legend is an auxiliary
    # block (2026-09-23, #123, encoding.base._aux_legend_channels(), read
    # from Rust): a non-"none" orient= is refused by
    # ChannelBase._validate_legend_value.
    _honored_kwargs = APPEARANCE_SORT

    def _validate(self) -> None:
        """Refuse an explicit ``scale=`` range naming the fixed dash palette.

        Mirrors ``Shape._validate``'s eager gate, but StrokeDash's range
        cannot be wired at all: the dash scale resolves to a palette index
        (``crates/ferrum-scene/src/dash.rs``), which the GPU-packed
        interactive path reads as a fixed-slot index -- there is no way to
        carry an arbitrary domain-to-range mapping through that
        representation. Domain-only scales pass through unchanged.
        """
        super()._validate()
        range_ = self._explicit_scale_range()
        if range_ is not None:
            raise ValueError(
                f"StrokeDash(scale=...): range={range_!r} is not supported -- "
                "the StrokeDash categorical scale's range is a fixed set of "
                "palette patterns, not an arbitrary domain-to-range mapping. "
                "Wire only the domain: "
                "StrokeDash('field', scale=OrdinalScale(domain=[...]))."
            )


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
    ``condition`` is honored: pass a ``ConditionalSpec`` built via
    ``sel.when(...).otherwise(...)`` (or ``fm.when(sel).then(...).otherwise(...)``)
    to switch rotation based on selection state -- see ``Chart.encode()``'s
    condition= translation (``ferrum-spec.md`` §3.10).

    ``Angle`` draws no legend of its own. ``legend=None``/``legend=False`` are
    accepted silently; ``title=`` and a legend dict/``Legend(...)`` raise
    ``ValueError`` at construction, since there is nothing for them to label.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="x", y="y", angle=fm.Angle("direction"))
    """

    _channel_name = "angle"

    # Plus "condition", derived from the conditional vocabulary.
    _honored_kwargs = APPEARANCE_BASE | AGGREGATE
    _legend_surface = "none"
