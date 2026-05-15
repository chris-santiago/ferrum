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
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal


@dataclass(frozen=True)
class SelectionMark:
    """Visual style for the interval selection brush rectangle.

    Pass an instance to ``selection_interval(mark=...)`` to override the
    default brush appearance.

    Parameters
    ----------
    fill : str, optional
        Hex colour string for the brush fill (e.g. ``"#4287f5"``).  Defaults
        to the renderer's built-in blue.
    stroke : str, optional
        Hex colour string for the brush border.  Defaults to the renderer's
        built-in grey.
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

    def to_dict(self) -> dict:
        """Serialize to a dict matching the Rust ``SelectionMarkStyle`` shape."""
        d: dict[str, Any] = {
            "fill_opacity": self.fill_opacity,
            "stroke_opacity": self.stroke_opacity,
            "stroke_width": self.stroke_width,
        }
        if self.fill is not None:
            d["fill"] = _hex_to_color_dict(self.fill)
        if self.stroke is not None:
            d["stroke"] = _hex_to_color_dict(self.stroke)
        if self.stroke_dash is not None:
            d["stroke_dash"] = self.stroke_dash
        return d


@dataclass(frozen=True)
class Selection:
    """Immutable selection descriptor.

    Created by ``selection_point()``, ``selection_interval()``,
    ``selection_single()``, or ``selection_multi()`` — do not construct
    directly.

    Attributes
    ----------
    name : str
        Stable identifier used to link this selection to conditional encodings
        and to the WASM renderer.
    kind : {"point", "interval"}
        Selection type.
    params : dict
        Resolved parameters forwarded to the Rust ``SelectionSpec``.

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

    def when(self, if_encoding: Any) -> _SelectionCondition:
        """Start a conditional encoding: ``sel.when(Color("x")).otherwise(value("#ccc"))``."""
        return _SelectionCondition(selection=self, if_encoding=if_encoding)

    def to_spec_dict(self) -> dict:
        """Serialize to a dict matching the Rust ``SelectionSpec`` shape."""
        d: dict[str, Any] = {"type": self.kind, "name": self.name}
        d.update(self.params)
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

    def to_spec_dict(self) -> dict:
        """Serialize to a dict matching the Rust ``ConditionalEncoding`` shape."""
        return {
            "selection_name": self.selection_name,
            "channel": _resolve_channel(self.if_selected),
            "if_selected": _resolve_encoding_value(self.if_selected),
            "if_not": _resolve_encoding_value(self.if_not),
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
    return Selection(name=sel_name, kind="point", params=params)


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
        params["mark"] = mark.to_dict()
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
        The constant to embed.  Hex colour strings (e.g. ``"#cccccc"``) are
        interpreted as colours.  Numbers are interpreted as opacity values.

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

    Fade unselected marks with low opacity:

    >>> color=sel.when(fm.Color("category")).otherwise(fm.value(0.1))
    """
    return _LiteralValue(v)


@dataclass(frozen=True)
class _LiteralValue:
    val: Any


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


def _hex_to_color_dict(hex_str: str) -> dict[str, int]:
    h = hex_str.lstrip("#")
    if len(h) == 6:
        return {
            "r": int(h[0:2], 16),
            "g": int(h[2:4], 16),
            "b": int(h[4:6], 16),
            "a": 255,
        }
    if len(h) == 8:
        return {
            "r": int(h[0:2], 16),
            "g": int(h[2:4], 16),
            "b": int(h[4:6], 16),
            "a": int(h[6:8], 16),
        }
    return {"r": 0, "g": 0, "b": 0, "a": 255}


def _resolve_channel(enc: Any) -> str:
    from ferrum.encoding.appearance import Color, Opacity, Size

    if isinstance(enc, Color) or (isinstance(enc, _LiteralValue) and isinstance(enc.val, str) and enc.val.startswith("#")):
        return "color"
    if isinstance(enc, Opacity):
        return "opacity"
    if isinstance(enc, Size):
        return "size"
    return "color"


def _resolve_encoding_value(enc: Any) -> dict:
    if isinstance(enc, _LiteralValue):
        v = enc.val
        if isinstance(v, str) and v.startswith("#"):
            return {"kind": "color", "value": _hex_to_color_dict(v)}
        if isinstance(v, (int, float)):
            return {"kind": "opacity", "value": float(v)}
        return {"kind": "opacity", "value": 1.0}
    from ferrum.encoding.base import ChannelBase
    if isinstance(enc, ChannelBase):
        return {"kind": "field", "name": enc.field}
    raise TypeError(
        f"conditional encoding value must be a channel object (Color, Shape, etc.) "
        f"or a literal value(...) — got {type(enc).__name__}"
    )
