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
    """Visual style for the interval selection brush rectangle."""

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
    """Immutable selection descriptor."""

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
    """Resolved conditional encoding — produced by ``sel.when(...).otherwise(...)``."""

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
    """Create a point selection."""
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
    """Create an interval selection (brush)."""
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
selection_multi = partial(selection_point, toggle="event.shiftKey")


def value(v: Any) -> _LiteralValue:
    """Wrap a literal value for use in conditional encodings."""
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
