"""Selection API for interactive charts (Phase 11c).

Constructors
------------
selection_point   — point selection (click on marks)
selection_interval — interval selection (drag to brush)
selection_single  — point selection with toggle=False
selection_multi   — point selection with toggle="event.shiftKey"

Classes
-------
Selection          — immutable selection descriptor, with .when() conditional builder
SelectionMark      — visual style for interval selection brush

Module-level builders
---------------------
when               — ``fm.when(sel).then(v_if).otherwise(v_else)`` conditional builder
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

from ferrum._validate import is_none_color_sentinel
from ferrum.parameter import Parameter, _normalize_bind


@dataclass(frozen=True)
class SelectionMark:
    """Visual style for the interval selection brush rectangle.

    Pass an instance to ``selection_interval(mark=...)`` to override the
    default brush appearance.

    Parameters
    ----------
    fill : str, optional
        Color string for the brush fill — a CSS name, hex
        (e.g. ``"#4287f5"``), or ``rgb()``/``rgba()`` form.  Defaults to the
        renderer's built-in blue.
    stroke : str, optional
        Color string for the brush border (same vocabulary as *fill*).
        Defaults to the renderer's built-in grey.
    fill_opacity : float, default 0.3
        Opacity of the fill (0.0 = transparent, 1.0 = opaque).
    stroke_opacity : float, default 1.0
        Opacity of the border.
    stroke_width : float, default 1.0
        Border line width in pixels.
    stroke_dash : list of float, optional
        Dash pattern for the border (e.g. ``[4, 2]``).  Solid when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> brush_style = fm.SelectionMark(fill="#ff9900", fill_opacity=0.2, stroke_dash=[4, 2])
    >>> brush = fm.selection_interval(mark=brush_style)
    """

    fill: str | None = None
    stroke: str | None = None
    fill_opacity: float = 0.3
    stroke_opacity: float = 1.0
    stroke_width: float = 1.0
    stroke_dash: list[float] | None = None

    def to_spec_dict(self) -> dict:
        """Serialize to a dict matching the Rust ``SelectionMarkStyle`` shape."""
        d: dict[str, Any] = {
            "fill_opacity": self.fill_opacity,
            "stroke_opacity": self.stroke_opacity,
            "stroke_width": self.stroke_width,
        }
        if self.fill is not None:
            d["fill"] = _hex_to_color_dict(self.fill, context=f"SelectionMark: fill={self.fill!r}")
        if self.stroke is not None:
            d["stroke"] = _hex_to_color_dict(
                self.stroke, context=f"SelectionMark: stroke={self.stroke!r}"
            )
        if self.stroke_dash is not None:
            d["stroke_dash"] = self.stroke_dash
        return d


@dataclass(frozen=True)
class Selection(Parameter):
    """Immutable selection descriptor.

    Created by ``selection_point()``, ``selection_interval()``,
    ``selection_single()``, or ``selection_multi()`` — do not construct
    directly.

    ``Selection`` is a subtype of ``Parameter``, so ``isinstance(sel,
    Parameter)`` is ``True`` and ``sel.ref()`` returns
    ``{"param": self.name}``.

    Attributes
    ----------
    name : str
        Stable identifier used to link this selection to conditional encodings
        and to the WASM renderer.
    kind : {"point", "interval"}
        Selection type.
    params : dict
        Resolved parameters forwarded to the Rust ``SelectionSpec``.
    bind : str or None, default None
        Optional bind target.  ``"legend"`` wires legend-entry clicks to
        toggle the point selection.  When ``None`` the ``"bind"`` key is
        omitted from ``to_param_spec_dict()`` output.

    Examples
    --------
    Build a conditional encoding from a selection:

    >>> import ferrum as fm
    >>> sel = fm.selection_point()
    >>> color_enc = sel.when(fm.Color("category")).otherwise(fm.value("#cccccc"))
    >>> fm.Chart(df).mark_point().encode(
    ...     x=fm.X("x"), y=fm.Y("y"), color=color_enc
    ... ).add_selection(sel)
    """

    name: str
    kind: Literal["point", "interval"]
    params: dict = field(default_factory=dict)
    bind: str | None = None

    def when(self, if_encoding: Any) -> _SelectionCondition:
        """Start a conditional encoding: ``sel.when(Color("x")).otherwise(value("#ccc"))``."""
        return _SelectionCondition(selection=self, if_encoding=if_encoding)

    def to_spec_dict(self) -> dict:
        """Serialize to a dict matching the Rust ``SelectionSpec`` shape.

        This is the existing Phase 11c wire format consumed by the WASM
        renderer.  The ``bind`` field is intentionally excluded here — it
        belongs only in ``to_param_spec_dict()``.

        Returns
        -------
        dict
            ``{"type": kind, "name": name, ...params}``
        """
        d: dict[str, Any] = {"type": self.kind, "name": self.name}
        d.update(self.params)
        return d

    def to_param_spec_dict(self) -> dict:
        """Serialize to the params-array wire dict (D6 reactive-parameter format).

        This is the unified declaration consumed by the static resolver and
        the new WASM parameter wiring.  Selections also continue to
        serialize into the existing ``selections`` key via ``to_spec_dict()``
        for backward-compatible WASM reads.

        ``"bind"`` is emitted only when set to avoid null clutter.

        Returns
        -------
        dict
            Wire shape::

                {"name": str, "kind": "point"|"interval",
                 "select": {<params dict>}}

                # with bind:
                {"name": str, "kind": "point"|"interval",
                 "select": {<params dict>}, "bind": str}

        Examples
        --------
        >>> import ferrum as fm
        >>> brush = fm.selection_interval(name="brush", encodings=["x"])
        >>> brush.to_param_spec_dict()
        {'name': 'brush', 'kind': 'interval', 'select': {'translate': True, ...}}
        """
        d: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "select": dict(self.params),
        }
        normalized = _normalize_bind(self.bind)
        if normalized is not None:
            d["bind"] = normalized
        return d


@dataclass(frozen=True)
class _SelectionCondition:
    selection: Selection
    if_encoding: Any

    def otherwise(self, else_encoding: Any) -> ConditionalSpec:
        return ConditionalSpec(
            selection_name=self.selection.name,
            if_selected=self.if_encoding,
            if_not=else_encoding,
            selection=self.selection,
        )


@dataclass(frozen=True)
class ConditionalSpec:
    """Resolved conditional encoding — produced by ``sel.when(...).otherwise(...)``.

    Do not construct directly.  Build one through the selection fluent API::

        sel.when(<if_encoding>).otherwise(<else_encoding>)

    Attributes
    ----------
    selection_name : str
        Name of the ``Selection`` that drives the condition.
    if_selected : encoding channel or value(...)
        Encoding applied when a datum falls inside the selection.
    if_not : encoding channel or value(...)
        Encoding applied when a datum falls outside the selection.
    channel : str or None, default None
        Explicit wire channel (e.g. ``"opacity"``, ``"size"``, ``"color"``).
        Set by ``Chart.encode(<channel>=cond)`` from the encode key, so the
        wire ``channel`` and the value-kind resolution match even when the
        branches are bare numbers (which carry no channel of their own). When
        ``None`` the channel is inferred from ``if_selected`` via
        ``_resolve_channel``.
    selection : Selection or None, default None
        The originating ``Selection``, carried so callers
        (``Chart.encode`` / ``Chart.conditional``) can auto-register it without
        a separate ``.add_selection()`` call.

    Examples
    --------
    >>> import ferrum as fm
    >>> sel = fm.selection_point()
    >>> cond = sel.when(fm.Color("category")).otherwise(fm.value("#cccccc"))
    >>> type(cond).__name__
    'ConditionalSpec'
    """

    selection_name: str
    if_selected: Any
    if_not: Any
    channel: str | None = None
    selection: Selection | None = None

    def to_spec_dict(self) -> dict:
        """Serialize to a dict matching the Rust ``ConditionalEncoding`` shape.

        When ``self.channel`` is set (the ``encode(<channel>=cond)`` path) it is
        used as both the wire ``channel`` and the value-kind hint for
        ``if_selected``/``if_not``. Otherwise the channel is inferred from
        ``if_selected`` (the ``.conditional(...)`` path with an encoding object).
        """
        channel = self.channel if self.channel is not None else _resolve_channel(self.if_selected)
        return {
            "selection_name": self.selection_name,
            "channel": channel,
            "if_selected": _resolve_encoding_value(self.if_selected, channel=channel),
            "if_not": _resolve_encoding_value(self.if_not, channel=channel),
        }


def selection_point(
    *,
    fields: list[str] | None = None,
    encodings: list[str] | None = None,
    nearest: bool = False,
    toggle: str = "event.shiftKey",
    on: str = "click",
    clear: str = "mouseout",
    resolve: Literal["global", "union", "intersect"] = "global",
    name: str | None = None,
    bind: str | None = None,
) -> Selection:
    """Create a point selection activated by clicking on marks.

    A point selection highlights individual marks.  Shift-click adds to the
    selection by default (``toggle="event.shiftKey"``).  Use
    ``selection_single`` to disable toggle, or ``selection_multi`` to keep
    the shift-click default explicitly.

    Parameters
    ----------
    fields : list of str, optional
        Data fields to project the selection onto.  When omitted the
        selection binds to all fields.
    encodings : list of str, optional
        Encoding channels to project onto (e.g. ``["x", "color"]``).
    nearest : bool, default False
        When ``True``, clicking between marks selects the nearest one.
    toggle : str, default "event.shiftKey"
        JavaScript event expression controlling when clicking *adds* to the
        selection instead of replacing it.  Pass ``"false"`` to disable
        toggling (see also ``selection_single``).
    on : str, default "click"
        Event that triggers the selection (e.g. ``"mouseover"``).
    clear : str, default "mouseout"
        Event that clears the selection.
    resolve : {"global", "union", "intersect"}, default "global"
        How to resolve this selection when the chart is faceted.
    name : str, optional
        Stable identifier for the selection.  Auto-generated when omitted.
    bind : str or None, default None
        Optional bind target.  ``"legend"`` wires legend-entry clicks to
        toggle the point selection (and thus any series linked to it via a
        conditional encoding).  When ``None`` no bind is configured.

    Returns
    -------
    Selection
        Immutable selection descriptor.  Pass to ``Chart.add_selection()``
        and use ``sel.when(...).otherwise(...)`` to build conditional encodings.

    Examples
    --------
    Highlight selected marks by colour:

    >>> import ferrum as fm
    >>> sel = fm.selection_point()
    >>> fm.Chart(df).mark_point().encode(
    ...     x=fm.X("x"), y=fm.Y("y"),
    ...     color=sel.when(fm.Color("category")).otherwise(fm.value("#cccccc")),
    ... ).add_selection(sel)

    Nearest-mark selection on mouse-over:

    >>> sel = fm.selection_point(nearest=True, on="mouseover", clear="mouseout")

    Legend-toggle selection:

    >>> sel = fm.selection_point(bind="legend")
    """
    sel_name = name or f"sel_{uuid.uuid4().hex[:8]}"
    params: dict[str, Any] = {
        "nearest": nearest,
        "toggle": _to_event_expr(toggle),
        "on": _to_event_expr(on),
        "clear": _to_event_expr(clear),
        "resolve": resolve,
    }
    if fields is not None:
        params["fields"] = fields
    if encodings is not None:
        params["encodings"] = [e.lower() for e in encodings]
    return Selection(name=sel_name, kind="point", params=params, bind=bind)


def selection_interval(
    *,
    fields: list[str] | None = None,
    encodings: list[str] | None = None,
    translate: bool = True,
    zoom: bool = True,
    mark: SelectionMark | None = None,
    resolve: Literal["global", "union", "intersect"] = "global",
    name: str | None = None,
) -> Selection:
    """Create an interval (brush) selection activated by dragging.

    An interval selection lets the user drag a rectangular brush over the
    chart.  All marks inside the brush are considered selected.  The brush
    supports panning (``translate=True``) and scroll-to-zoom (``zoom=True``)
    after it is drawn.

    Parameters
    ----------
    fields : list of str, optional
        Data fields to project the brush onto.  When omitted the brush
        applies to all fields represented by the brushed encodings.
    encodings : list of str, optional
        Encoding channels to constrain the brush to (e.g. ``["x"]`` for a
        1-D horizontal brush).
    translate : bool, default True
        Allow the user to reposition the brush by dragging inside it.
    zoom : bool, default True
        Allow the user to scroll inside the brush to zoom the view.
    mark : SelectionMark, optional
        Visual style of the brush rectangle.  Defaults to a semi-transparent
        blue fill with a solid grey border.
    resolve : {"global", "union", "intersect"}, default "global"
        How to resolve this selection when the chart is faceted.
    name : str, optional
        Stable identifier for the selection.  Auto-generated when omitted.

    Returns
    -------
    Selection
        Immutable selection descriptor.  Pass to ``Chart.add_selection()``
        and use ``sel.when(...).otherwise(...)`` to build conditional encodings.

    Examples
    --------
    Brush that greys out unselected points:

    >>> import ferrum as fm
    >>> brush = fm.selection_interval()
    >>> fm.Chart(df).mark_point().encode(
    ...     x=fm.X("x"), y=fm.Y("y"),
    ...     color=brush.when(fm.Color("category")).otherwise(fm.value("#cccccc")),
    ... ).add_selection(brush)

    Horizontal-only brush:

    >>> brush = fm.selection_interval(encodings=["x"])

    Custom brush style:

    >>> brush = fm.selection_interval(
    ...     mark=fm.SelectionMark(fill="#4287f5", fill_opacity=0.2, stroke_dash=[4, 2])
    ... )
    """
    sel_name = name or f"sel_{uuid.uuid4().hex[:8]}"
    params: dict[str, Any] = {
        "translate": translate,
        "zoom": zoom,
        "resolve": resolve,
    }
    if fields is not None:
        params["fields"] = fields
    if encodings is not None:
        params["encodings"] = [e.lower() for e in encodings]
    if mark is not None:
        params["mark"] = mark.to_spec_dict()
    return Selection(name=sel_name, kind="interval", params=params)


selection_single = partial(selection_point, toggle="false")
selection_single.__doc__ = """Create a single-select point selection (no toggle).

    A convenience alias for ``selection_point(toggle="false")``.  Clicking a
    new mark *replaces* the current selection rather than adding to it.

    All parameters are identical to ``selection_point`` except *toggle*,
    which is fixed to ``"false"`` and cannot be overridden.

    Returns
    -------
    Selection
        Immutable selection descriptor.

    Examples
    --------
    >>> import ferrum as fm
    >>> sel = fm.selection_single()
    >>> fm.Chart(df).mark_point().encode(
    ...     x=fm.X("x"), y=fm.Y("y"),
    ...     opacity=sel.when(fm.Opacity("density")).otherwise(fm.value(0.2)),
    ... ).add_selection(sel)
    """

selection_multi = partial(selection_point, toggle="event.shiftKey")
selection_multi.__doc__ = """Create a multi-select point selection (shift-click to add).

    A convenience alias for ``selection_point(toggle="event.shiftKey")``,
    making the default toggle behaviour explicit.  Shift-clicking a mark
    adds it to the selection; clicking without shift replaces the selection.

    All parameters are identical to ``selection_point``.

    Returns
    -------
    Selection
        Immutable selection descriptor.

    Examples
    --------
    >>> import ferrum as fm
    >>> sel = fm.selection_multi()
    >>> fm.Chart(df).mark_point().encode(
    ...     x=fm.X("x"), y=fm.Y("y"),
    ...     color=sel.when(fm.Color("category")).otherwise(fm.value("#cccccc")),
    ... ).add_selection(sel)
    """


def value(v: Any) -> "_LiteralValue":
    """Wrap a literal value for use in conditional encodings.

    Returns an opaque literal wrapper that can be passed as the
    ``if_selected`` or ``if_not`` argument of
    ``sel.when(...).otherwise(...)`` when a constant (rather than a field
    mapping) is needed.

    Parameters
    ----------
    v : str, int, or float
        The constant to embed.  Color strings (a CSS name, hex, e.g.
        ``"#cccccc"``, or ``rgb()``/``rgba()`` form — validated and
        normalized by ferrum's one Rust color parser) are interpreted as
        colours; an unparseable string raises ``ValueError`` naming the
        accepted forms.  Numbers are interpreted as opacity values.

    Returns
    -------
    _LiteralValue
        Opaque wrapper consumed by ``ConditionalSpec``.

    Examples
    --------
    Grey-out unselected marks:

    >>> import ferrum as fm
    >>> sel = fm.selection_point()
    >>> fm.Chart(df).mark_point().encode(
    ...     x=fm.X("x"), y=fm.Y("y"),
    ...     color=sel.when(fm.Color("category")).otherwise(fm.value("#cccccc")),
    ... ).add_selection(sel)

    Fade unselected marks with low opacity (numeric branches resolve to opacity
    values, not colours, so assign to the ``opacity`` channel):

    >>> opacity = fm.when(sel).then(1.0).otherwise(0.1)
    >>> # or, equivalently, with explicit value(...) wrappers:
    >>> opacity = sel.when(fm.value(1.0)).otherwise(fm.value(0.1))
    """
    return _LiteralValue(v)


@dataclass(frozen=True)
class _LiteralValue:
    val: Any


# ── Module-level when() builder ───────────────────────────────────────────────


@dataclass(frozen=True)
class _When:
    """Intermediate builder produced by ``fm.when(parameter)``.

    Call ``.then(v)`` to continue the chain.
    """

    _parameter: Parameter

    def then(self, v: Any) -> "_WhenThen":
        """Provide the value to use when the parameter's selection is active.

        Parameters
        ----------
        v : Any
            Value applied when a datum falls inside the selection.  Plain
            values are wrapped via ``value()``; passing an already-wrapped
            ``value(...)`` is also accepted.  Color strings (CSS name, hex,
            or ``rgb()``/``rgba()``) are interpreted as colours at
            serialization time (in ``_resolve_encoding_value``), not at this
            layer.

        Returns
        -------
        _WhenThen
            Intermediate builder; call ``.otherwise(v)`` to finalize.
        """
        return _WhenThen(_parameter=self._parameter, _if_val=_ensure_value(v))


@dataclass(frozen=True)
class _WhenThen:
    """Intermediate builder produced by ``fm.when(parameter).then(v)``.

    Call ``.otherwise(v)`` to produce the final ``ConditionalSpec``.
    """

    _parameter: Parameter
    _if_val: "_LiteralValue"

    def otherwise(self, v: Any) -> ConditionalSpec:
        """Provide the value to use when the selection is NOT active.

        Parameters
        ----------
        v : Any
            Value applied when a datum falls outside the selection.  Plain
            numbers and color strings (CSS name, hex, or ``rgb()``/``rgba()``)
            are auto-wrapped via ``value()``.

        Returns
        -------
        ConditionalSpec
            Resolved conditional encoding spec.  Passes through the existing
            ``ConditionalEncoding`` wire (``selection_name``, ``channel``,
            ``if_selected``, ``if_not``).
        """
        selection = self._parameter if isinstance(self._parameter, Selection) else None
        return ConditionalSpec(
            selection_name=self._parameter.name,
            if_selected=self._if_val,
            if_not=_ensure_value(v),
            selection=selection,
        )


def _ensure_value(v: Any) -> "_LiteralValue":
    """Wrap *v* in a ``_LiteralValue`` unless it already is one."""
    if isinstance(v, _LiteralValue):
        return v
    return _LiteralValue(v)


def when(parameter: Parameter) -> _When:
    """Start a module-level conditional encoding builder.

    A more explicit alternative to ``sel.when(enc).otherwise(enc)`` for
    cases where the conditional test is on a selection parameter and the
    branches are literal values (opacity, colour) rather than encoding
    channels.

    Usage::

        fm.when(sel).then(v_if).otherwise(v_else)

    The ``parameter`` argument must be a ``Selection`` instance — the
    conditional test is "datum ∈ selection".  Variable parameters
    (``fm.param``) drive value/domain/bind references, not the conditional
    predicate; passing one raises ``TypeError``.

    ``v_if`` and ``v_else`` may be plain numbers or ``fm.value(...)``
    wrappers; they are wrapped via ``value()`` when not already a
    ``_LiteralValue``.  Color strings (CSS name, hex, or ``rgb()``/``rgba()``)
    are interpreted as colours at serialization time (in
    ``_resolve_encoding_value``), not at this layer.

    Parameters
    ----------
    parameter : Selection
        The selection whose active/inactive state drives the condition.
        Produced by ``selection_point()`` or ``selection_interval()``.

    Returns
    -------
    _When
        Intermediate builder.  Call ``.then(v_if).otherwise(v_else)`` to
        finalize.

    Raises
    ------
    TypeError
        If ``parameter`` is not a ``Selection`` instance.  Variable
        parameters from ``fm.param`` drive value/domain/bind references,
        not the conditional predicate.

    Examples
    --------
    Fade unselected marks — assign the conditional to the ``opacity`` channel so
    the numeric branches resolve to opacity values (``encode`` stamps the channel
    from its key and auto-registers ``sel``):

    >>> import ferrum as fm
    >>> sel = fm.selection_point()
    >>> chart = fm.Chart(df).mark_point().encode(
    ...     x=fm.X("x"), y=fm.Y("y"),
    ...     opacity=fm.when(sel).then(1.0).otherwise(0.2),
    ... )

    Colour toggle on legend selection:

    >>> sel = fm.selection_point(bind="legend")
    >>> cond = fm.when(sel).then(fm.value("#1f77b4")).otherwise(fm.value("#cccccc"))
    """
    if not isinstance(parameter, Selection):
        raise TypeError(
            f"fm.when(...) requires a selection-kind parameter (a Selection produced by "
            f"selection_point() or selection_interval()); got {type(parameter).__name__}. "
            f"Variable parameters from fm.param() drive value/domain/bind references, "
            f"not the conditional predicate."
        )
    return _When(_parameter=parameter)


# ── helpers ──────────────────────────────────────────────────────────

_EVENT_MAP = {
    "click": "click",
    "mouseout": "mouseout",
    "mouseover": "mouseover",
    "event.shiftKey": "shift_key",
    "dblclick": "dblclick",
    "false": "click",
}


def _to_event_expr(s: str) -> str | dict[str, str]:
    canon = _EVENT_MAP.get(s)
    if canon:
        return canon
    return {"custom": s}


def _hex_to_color_dict(color_str: str, *, context: str) -> dict[str, int]:
    """Parse a color string into the wire color dict via the one Rust parser.

    Routes through :func:`ferrum.color.to_hex`, so this accepts the exact
    vocabulary every other ferrum color boundary accepts (CSS names, hex,
    ``rgb()``/``rgba()``) — not just hex. Raises ``ValueError`` on anything
    unparseable; there is no silent fallback.

    Parameters
    ----------
    context:
        Human-readable description of the failing call site (e.g.
        ``"SelectionMark: fill='nonsense'"`` or
        ``"fm.value('nonsense') for channel='color'"``), prefixed onto any
        raised message. Mirrors ``marks/base.py``'s
        ``_validate_literal_color`` prefix shape, so a bad literal raised
        several frames inside ``.to_svg()`` (this is a construction-time
        surface, not a construction-time gate — the failure only surfaces
        when the conditional/mark is serialized) is traceable back to the
        call that produced it.

    ``"none"``/``"transparent"`` are refusals here, not the paint-clear they
    are on the mark-style surface (spec §4.1, amended 2026-09-01, extended to
    ``"transparent"`` in the 2026-09-01 T8 quality-review supersession): the
    selection wire's ``{r, g, b, a}`` color dict has no representation for a
    cleared paint, so a bound selection value can't express either spelling
    (logged follow-up). Both spellings get the same dedicated message rather
    than the generic accepted-forms text, since neither is a typo or an
    unrecognized vocabulary item — each is a real color-clearing request
    this surface can't fulfil yet. The match is trimmed and case-insensitive
    (``"None"``, ``"NONE"``, ``" none "``, ``"Transparent"``, ``" transparent "``
    all count) via the shared ``ferrum._validate.is_none_color_sentinel``
    predicate — the same one ``marks/base.py``'s ``_is_paint_sentinel``
    composes from, matching Rust's
    ``trimmed.eq_ignore_ascii_case("none") || trimmed.eq_ignore_ascii_case("transparent")``
    (``draw.rs``). This does not extend to ``"theme:label"``, an unrelated
    sentinel with its own refusal path.
    """
    if is_none_color_sentinel(color_str):
        raise ValueError(
            f"{context}: selection styling cannot express a cleared paint "
            "('none'/'transparent'); provide a color"
        )

    from ferrum.color import to_hex

    try:
        h = to_hex(color_str).lstrip("#")
    except ValueError as exc:
        raise ValueError(f"{context} is not a valid color ({exc})") from exc
    if len(h) == 8:
        return {
            "r": int(h[0:2], 16),
            "g": int(h[2:4], 16),
            "b": int(h[4:6], 16),
            "a": int(h[6:8], 16),
        }
    return {
        "r": int(h[0:2], 16),
        "g": int(h[2:4], 16),
        "b": int(h[4:6], 16),
        "a": 255,
    }


def _resolve_channel(enc: Any) -> str:
    from ferrum.encoding.appearance import (
        Angle,
        Color,
        FillOpacity,
        Opacity,
        Shape,
        Size,
        StrokeDash,
        StrokeOpacity,
        StrokeWidth,
    )

    if isinstance(enc, Color):
        return "color"
    if isinstance(enc, Opacity):
        return "opacity"
    if isinstance(enc, Size):
        return "size"
    if isinstance(enc, Shape):
        return "shape"
    if isinstance(enc, StrokeWidth):
        return "stroke_width"
    if isinstance(enc, StrokeOpacity):
        return "stroke_opacity"
    if isinstance(enc, StrokeDash):
        return "stroke_dash"
    if isinstance(enc, FillOpacity):
        return "fill_opacity"
    if isinstance(enc, Angle):
        return "angle"
    if isinstance(enc, _LiteralValue) and isinstance(enc.val, (int, float)):
        # A bare number carries no channel of its own. Default to "opacity":
        # it is the sensible default for a numeric literal and the only numeric
        # channel whose (channel, kind) tuple the WASM matcher handles when no
        # explicit channel is supplied via encode(<channel>=...).
        return "opacity"
    return "color"


def _resolve_encoding_value(enc: Any, *, channel: str | None = None) -> dict:
    """Serialize a conditional encoding value to the Rust wire dict.

    Parameters
    ----------
    enc:
        A ``_LiteralValue`` (from ``fm.value(...)``) or a ``ChannelBase`` subclass.
    channel:
        The resolved channel string (e.g. ``"size"``, ``"opacity"``).  Used to
        tag numeric literals correctly so the Rust resolver maps them to the right
        ``EncodingValue`` variant.  When ``None``, numeric literals fall back to
        the ``"opacity"`` tag (backward-compatible default).
    """
    if isinstance(enc, _LiteralValue):
        v = enc.val
        if isinstance(v, str):
            # Routes through the one Rust color parser (ferrum.color.to_hex);
            # a parseable string is a color, an unparseable one raises
            # ValueError naming the accepted forms — no silent fallback.
            context = f"fm.value({v!r}) for channel={channel!r}"
            return {"kind": "color", "value": _hex_to_color_dict(v, context=context)}
        if isinstance(v, (int, float)):
            fv = float(v)
            if channel == "size":
                return {"kind": "size", "value": fv}
            if channel == "stroke_width":
                return {"kind": "stroke_width", "value": fv}
            if channel == "stroke_dash":
                return {"kind": "stroke_dash", "value": [fv]}
            if channel == "stroke_opacity":
                return {"kind": "stroke_opacity", "value": fv}
            if channel == "fill_opacity":
                return {"kind": "fill_opacity", "value": fv}
            if channel == "angle":
                return {"kind": "angle", "value": fv}
            return {"kind": "opacity", "value": fv}
        raise TypeError(
            f"fm.value(...) accepts a color string or a number, got {type(v).__name__}: {v!r}"
        )
    from ferrum.encoding.base import ChannelBase

    if isinstance(enc, ChannelBase):
        return {"kind": "field", "name": enc.field}
    raise TypeError(
        f"conditional encoding value must be a channel object (Color, Shape, etc.) "
        f"or a literal value(...) — got {type(enc).__name__}"
    )
