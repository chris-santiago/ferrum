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

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

from ferrum._validate import is_none_color_sentinel, is_real_number
from ferrum.encoding._channel_policy import _CONDITIONAL_CHANNELS
from ferrum.marks.base import _normalize_stroke_dash
from ferrum.parameter import Parameter, VariableParameter, _decode_param_ref, _normalize_bind


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
    stroke_dash : list of float, comma-separated str, or named linetype, optional
        Dash pattern for the border (e.g. ``[4, 2]``).  Solid when omitted.
        Also accepts a comma-separated numeric string (``"4,2"``) or a named
        linetype (``"solid"``, ``"dashed"``, ``"dotted"``, ``"dashdot"``,
        ``"longdash"``) -- normalized and validated at construction time
        (#124); an unrecognized string raises ``ValueError`` rather than
        constructing silently.

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

    def __post_init__(self) -> None:
        # Normalize stroke_dash through the shared helper (#124) so a named
        # linetype or comma-split string resolves identically to MarkBase's
        # and the annotate_* constructors' stroke_dash=, and so junk (e.g.
        # "wiggly") refuses at construction rather than constructing
        # silently and reaching the wire unvalidated. Frozen dataclass, so
        # the normalized value is written back via object.__setattr__
        # (matches axis.py's / _facet.py's existing pattern).
        normalized = _normalize_stroke_dash(self.stroke_dash, context="SelectionMark: stroke_dash")
        object.__setattr__(self, "stroke_dash", normalized)

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

    def to_spec_dict(self, *, params: dict[str, Parameter] | None = None) -> dict:
        """Serialize to a dict matching the Rust ``ConditionalEncoding`` shape.

        When ``self.channel`` is set (the ``encode(<channel>=cond)`` path) it is
        used as both the wire ``channel`` and the value-kind hint for
        ``if_selected``/``if_not``. Otherwise the channel is inferred from
        ``if_selected`` (the ``.conditional(...)`` path with an encoding object).

        Parameters
        ----------
        params:
            Name-keyed registry of the chart's registered reactive parameters
            (selections and ``fm.param()`` variables), used to resolve a
            ``Parameter.ref()`` value at ``if_selected``/``if_not`` (F-L13-01).
            ``Chart.to_spec()`` supplies this at spec-build time. ``None``
            (the default, used by every other caller of this method) means no
            registry is available -- see :func:`_resolve_encoding_value` for
            what that means for an unresolved ref.
        """
        channel = (
            self.channel
            if self.channel is not None
            else _infer_conditional_channel(self.if_selected, self.if_not)
        )
        return {
            "selection_name": self.selection_name,
            "channel": channel,
            "if_selected": _resolve_encoding_value(
                self.if_selected, channel=channel, params=params
            ),
            "if_not": _resolve_encoding_value(self.if_not, channel=channel, params=params),
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
    v : str, int, float, list, or tuple
        The constant to embed. What forms are accepted depends on the
        channel this value ends up assigned to (#128) -- the channel is
        resolved from the encode key (``encode(<channel>=cond)``) or, for
        ``.conditional(...)``, inferred from the ``if_selected`` branch:

        - ``color`` (and an unresolved/absent channel): a color string (a
          CSS name, hex, e.g. ``"#cccccc"``, or ``rgb()``/``rgba()`` form —
          validated and normalized by ferrum's one Rust color parser); an
          unparseable string raises ``ValueError`` naming the accepted
          forms. A number is interpreted as an opacity value.
        - ``stroke_dash``: a named linetype (``"dashed"``, ``"dotted"``,
          ...), a comma-separated numeric string (``"4,2"``), a number, or
          a list/tuple of numbers — the same vocabulary every other
          ``stroke_dash`` surface accepts, via
          :func:`ferrum.marks.base._normalize_stroke_dash`.
        - ``size``, ``opacity``, ``stroke_width``, ``stroke_opacity``,
          ``fill_opacity``, ``angle``: a number only. A color string here
          raises ``ValueError`` naming the channel and this call (e.g.
          ``fm.value('red') for channel='size' needs a number``) rather
          than silently repainting the mark's fill.

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
            ``value(...)`` is also accepted.  What forms are accepted, and
            what they mean, is typed by the resolved conditional channel
            (#128) at serialization time (in ``_resolve_encoding_value``),
            not at this layer -- see :func:`value` for the per-channel
            table.

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
            values are auto-wrapped via ``value()``; see :func:`value` for
            the per-channel table of what forms are accepted (#128).

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

    ``v_if`` and ``v_else`` may be plain values or ``fm.value(...)``
    wrappers; they are wrapped via ``value()`` when not already a
    ``_LiteralValue``.  What forms are accepted, and what they mean, is
    typed by the resolved conditional channel (#128) at serialization time
    (in ``_resolve_encoding_value``), not at this layer -- see
    :func:`value` for the per-channel table.

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


def _resolve_channel(enc: Any) -> str | None:
    """Return the channel one conditional branch names, or ``None`` if it names none.

    An encoding channel names its own channel. A literal names one by its
    type: a real number is an opacity, anything else a color. A
    ``Parameter.ref()`` names none.
    """
    from ferrum.encoding.base import ChannelBase

    if isinstance(enc, ChannelBase):
        return enc._channel_name
    if isinstance(enc, _LiteralValue):
        return "opacity" if is_real_number(enc.val) else "color"
    return None


def _infer_conditional_channel(if_selected: Any, if_not: Any) -> str:
    """Return the wire channel of a conditional declared without one.

    ``sel.when(...).otherwise(...)`` handed to ``Chart.conditional()`` names no
    channel, so it is read off the branches: an encoding-channel branch on
    either side wins (``sel.when(value("red")).otherwise(fm.Color("g"))`` is a
    color conditional); otherwise the first branch whose literal names one
    decides; two ``Parameter.ref()`` branches default to ``color``. The name
    goes through the conditional vocabulary, so a channel no conditional can
    target (``shape``, ``x``) is refused by name here.
    """
    from ferrum.encoding._channel_policy import conditional_wire_channel
    from ferrum.encoding.base import ChannelBase

    branches = (if_selected, if_not)
    named = next((b for b in branches if isinstance(b, ChannelBase)), None)
    if named is not None:
        return conditional_wire_channel(named._channel_name)
    for branch in branches:
        channel = _resolve_channel(branch)
        if channel is not None:
            return channel
    return "color"


def _resolve_param_ref(
    name: str, *, channel: str | None, params: dict[str, Parameter] | None
) -> dict:
    """Resolve a decoded ``Parameter.ref()`` name to its wire literal (F-L13-01).

    ``params`` is the chart's registered-parameter registry (name -> the
    registered ``Selection``/``VariableParameter``), supplied by
    ``Chart.to_spec()`` at spec-build time -- the D6 static-resolver
    semantics: only an ``fm.param()`` variable carries a static value to
    resolve as a literal; a ``Selection`` (or any other non-
    ``VariableParameter``) has none.

    Every failure to resolve -- no registry at all, an unregistered name, a
    registered-but-non-``VariableParameter`` name, or a registered variable
    with no static value -- raises the same typed-``ValueError`` family
    rather than emitting a placeholder. There is no wire value that is safe
    to fabricate here: every ``EncodingValue`` numeric variant *replaces*
    the targeted channel on the instance (e.g. ``Opacity{1.0}`` overwrites
    ``opacity`` unconditionally -- see ``ferrum-wasm``'s
    ``apply_value_common``, which is a `*view.opacity = *o` assignment, not
    a no-op), so any fabricated value would silently change the render of
    marks whose real value differs from the placeholder. (This helper
    previously fabricated ``{"kind": "opacity", "value": 1.0}`` for the
    no-registry case on the mistaken belief that arm was inert for every
    channel; it is not, and the mistake is corrected here.)

    Raises
    ------
    ValueError
        Always, unless *params* is supplied and *name* resolves to a
        registered ``VariableParameter`` with a non-``None`` static value.
    """
    if params is None:
        # This helper reached outside spec-build -- e.g.
        # ConditionalSpec.to_spec_dict() invoked directly, as every call
        # site that predates .ref() support still does -- so there is no
        # registry to resolve against at all. Chart.to_spec() always
        # supplies *params*, so a real chart never takes this branch.
        raise ValueError(
            f"parameter reference {{'param': {name!r}}} cannot be resolved without "
            f"the chart's registered parameters -- attach the parameter via "
            f"Chart.add_params() and resolve this conditional through "
            f"Chart.to_spec() (e.g. Chart.to_dict()/to_json()), not by calling "
            f"to_spec_dict() directly"
        )
    resolved = params.get(name)
    if resolved is None:
        raise ValueError(
            f"conditional encoding references unregistered parameter {name!r}; "
            f"register it via Chart.add_params()/add_selection() before referencing "
            f"it with .ref()"
        )
    if not isinstance(resolved, VariableParameter):
        raise ValueError(
            f"conditional encoding references parameter {name!r}, which is already "
            f"registered as a {type(resolved).__name__}, not an fm.param() variable; "
            f"only an fm.param() variable carries a static value to resolve at a "
            f"conditional site -- {name!r} has none, so add_params() would not help"
        )
    if resolved.value is None:
        raise ValueError(
            f"conditional encoding references parameter {name!r}, which has no static "
            f"value (fm.param({name!r}, value=None)); give it a literal value to use "
            f"at a conditional site"
        )
    return _resolve_encoding_value(_LiteralValue(resolved.value), channel=channel)


def _conditional_literal_type_error(v: Any, *, channel: str | None, needs: str) -> ValueError:
    """Build the one refusal shape every conditional-literal handler raises
    for a value of the wrong type/content for *channel* ("the literal doesn't
    fit this channel"). Every such refusal is a ``ValueError`` reading exactly ``fm.value(<repr>) for channel=<repr>
    needs <needs>, got <type>: <repr>`` -- the same shape the matrix in
    ``tests/test_condition_translation.py`` pins byte-for-byte, not a
    substring.
    """
    return ValueError(
        f"fm.value({v!r}) for channel={channel!r} needs {needs}, got {type(v).__name__}: {v!r}"
    )


#: The dash-pattern description shared by every stroke_dash refusal and
#: imported by the test matrix so the prose lives in exactly one place.
_DASH_PATTERN_NEEDS = (
    "a dash pattern (a named linetype, a comma-separated numeric string, a "
    "number, or a list/tuple of numbers)"
)


def _literal_color(v: str, *, channel: str | None) -> dict:
    """Parse a color string via the one Rust color parser (``ferrum.color.to_hex``).

    A parseable string is a color; an unparseable one raises ``ValueError``
    naming the accepted forms -- no silent fallback.
    """
    context = f"fm.value({v!r}) for channel={channel!r}"
    return {"kind": "color", "value": _hex_to_color_dict(v, context=context)}


def _literal_color_only(v: Any, *, channel: str) -> dict:
    """``color``: a color string only (see :func:`_literal_color`).

    A number on a color conditional is refused by name rather than tagged as
    an opacity: the literal is typed by the channel the conditional targets.
    """
    if isinstance(v, str):
        return _literal_color(v, channel=channel)
    raise _conditional_literal_type_error(v, channel=channel, needs="a color string")


def _literal_numeric_only(v: Any, *, channel: str) -> dict:
    """Resolve a literal for one of the six plain-numeric channels: a finite number only.

    A string is refused by channel name rather than reaching the color parser
    (a ``kind=color`` value would repaint the fill whatever the declared
    channel). A number is :func:`ferrum._validate.is_real_number` (never
    ``bool``), and ``nan``/``inf`` are refused.
    """
    if is_real_number(v):
        fv = float(v)
        if math.isfinite(fv):
            return {"kind": channel, "value": fv}
    raise _conditional_literal_type_error(v, channel=channel, needs="a number")


def _literal_stroke_dash(v: Any, *, channel: str) -> dict:
    """``stroke_dash``: the dash vocabulary every other ``stroke_dash`` surface
    accepts, through the shared :func:`ferrum.marks.base._normalize_stroke_dash`
    (a named linetype, a comma-separated numeric string, a number, or a
    list/tuple of numbers). Any value it refuses raises the conditional-literal
    refusal shape (:func:`_conditional_literal_type_error`).
    """
    try:
        dash = _normalize_stroke_dash(v, context=f"fm.value({v!r}) for channel={channel!r}")
    except ValueError as exc:
        raise _conditional_literal_type_error(
            v, channel=channel, needs=_DASH_PATTERN_NEEDS
        ) from exc
    if dash is None:
        raise _conditional_literal_type_error(v, channel=channel, needs=_DASH_PATTERN_NEEDS)
    return {"kind": "stroke_dash", "value": dash}


# One channel-keyed literal handler per conditional channel: the literal is
# typed by the channel it targets. The numeric-only entries are derived from
# `_CONDITIONAL_CHANNELS`, so a newly added conditional channel defaults to
# number-only rather than silently accepting a color string.
_CONDITIONAL_LITERAL_HANDLERS: dict[str, Callable[..., dict]] = {
    **{channel: _literal_numeric_only for channel in _CONDITIONAL_CHANNELS},
    "color": _literal_color_only,
    "stroke_dash": _literal_stroke_dash,
}


def _resolve_encoding_value(
    enc: Any, *, channel: str | None = None, params: dict[str, Parameter] | None = None
) -> dict:
    """Serialize a conditional encoding value to the Rust wire dict.

    Parameters
    ----------
    enc:
        A ``_LiteralValue`` (from ``fm.value(...)``), a ``ChannelBase``
        subclass, or a ``Parameter.ref()`` marker dict (F-L13-01).
    channel:
        The resolved channel string (e.g. ``"size"``, ``"opacity"``). Selects
        the literal handler from :data:`_CONDITIONAL_LITERAL_HANDLERS`:
        ``color`` accepts a color string; ``stroke_dash`` a name, comma
        string, number, or list; the six plain-numeric channels a number.
        ``None`` (a direct call with no channel) takes the channel the
        literal names (:func:`_resolve_channel`). A channel no conditional
        can target is refused by name.
    params:
        The chart's registered-parameter registry, forwarded to
        :func:`_resolve_param_ref` when *enc* is a ``Parameter.ref()``
        marker. See that function for what ``None`` means here.
    """
    ref_name = _decode_param_ref(enc)
    if ref_name is not None:
        return _resolve_param_ref(ref_name, channel=channel, params=params)
    if isinstance(enc, _LiteralValue):
        if channel is None:
            channel = _resolve_channel(enc)
        handler = _CONDITIONAL_LITERAL_HANDLERS.get(channel)
        if handler is None:
            raise ValueError(
                f"fm.value({enc.val!r}) for channel={channel!r}: no selection conditional "
                f"can target {channel!r}"
            )
        return handler(enc.val, channel=channel)
    if isinstance(enc, str):
        # A bare string means a FIELD everywhere in ferrum -- literals require
        # fm.value() (encode()'s own rule; see H10). Parse via the same
        # shorthand grammar encode() uses, then build the real channel object
        # implied by the branch's resolved wire channel so shorthand errors
        # (e.g. a bad type suffix) match encode()'s own behavior, and fall
        # through to the ChannelBase handling below.
        from ferrum._shorthand import parse_shorthand
        from ferrum.encoding import _channel_class_for

        shorthand_field, type_, agg = parse_shorthand(enc)
        if shorthand_field is None:
            raise TypeError(
                f"conditional encoding value {enc!r} has no field to encode -- an "
                f"aggregate-only shorthand like 'count()' needs an explicit channel "
                f"object (e.g. fm.Color(...)), not a bare string"
            )
        branch_cls = _channel_class_for(channel) if channel is not None else None
        if branch_cls is None:
            # Reached only by a direct call with no channel (every chart path
            # resolves one first -- see `ConditionalSpec.to_spec_dict`), so
            # there is no channel to imply a branch class. Color is the
            # stand-in for "some field-carrying channel"; the wire dict below
            # reads `enc.field` only, so the class does not change the
            # serialized output. It IS observable in the channel's own
            # honored-kwarg warning, though -- an aggregate-carrying shorthand
            # warns `color(aggregate=...)` here and would warn
            # `shape(aggregate=...)` under a different stand-in -- so the
            # choice is pinned by
            # `tests/test_condition_translation.py::test_channelless_shorthand_branch_warns_as_color`
            # rather than left to a comment. (A previous version of this
            # comment claimed the two were "indistinguishable by
            # construction"; that was false, and it told the next reader not
            # to test something testable.)
            from ferrum.encoding.appearance import Color

            branch_cls = Color
        shorthand_kwargs: dict[str, Any] = {}
        if type_:
            shorthand_kwargs["type"] = type_
        if agg:
            shorthand_kwargs["aggregate"] = agg
        enc = branch_cls(shorthand_field, **shorthand_kwargs)
    from ferrum.encoding.base import ChannelBase

    if isinstance(enc, ChannelBase):
        return {"kind": "field", "name": enc.field}
    raise TypeError(
        f"conditional encoding value must be a channel object (Color, Shape, etc.), "
        f"a field shorthand string (e.g. 'category:N'), or a literal value(...) — "
        f"got {type(enc).__name__}"
    )
