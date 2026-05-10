"""Chart — the user-facing top-level value class.

Immutability rule: every fluent method returns a new Chart. The internal
spec is deep-copied on each call so chains compose without aliasing surprises.
"""
from __future__ import annotations

from typing import Any, Optional, Union

from ferrum._coerce import to_arrow_table
from ferrum._shorthand import parse_shorthand
from ferrum.encoding.base import ChannelBase
from ferrum.marks.base import MarkBase
from ferrum.marks.deferred import deferred_mark_error, PHASE_8B_MARKS, PHASE_9_PLUS_MARKS
from ferrum.marks.statistical import desugar_density, desugar_histogram, desugar_smooth


_PRIMITIVE_MARKS = frozenset(["point", "line", "bar", "area", "rule", "text", "tick", "rect"])

_CHANNEL_CLASSES_BY_NAME: dict = {}


def _channel_class_for(name: str):
    """Return the channel-class for a given parameter name (lazy import to avoid cycles)."""
    if not _CHANNEL_CLASSES_BY_NAME:
        from ferrum.encoding import (
            X, Y, X2, Y2, XError, YError, XError2, YError2, Theta, Radius,
            Color, Fill, Stroke, Opacity, FillOpacity, StrokeOpacity,
            StrokeWidth, StrokeDash, Size, Shape, Angle,
            Text, Detail, Tooltip, TooltipField, Href, Description, Key,
            Facet, FacetRow, FacetCol,
        )
        _CHANNEL_CLASSES_BY_NAME.update({
            "x": X, "y": Y, "x2": X2, "y2": Y2,
            "x_error": XError, "y_error": YError, "x_error2": XError2, "y_error2": YError2,
            "theta": Theta, "radius": Radius,
            "color": Color, "fill": Fill, "stroke": Stroke,
            "opacity": Opacity, "fill_opacity": FillOpacity, "stroke_opacity": StrokeOpacity,
            "stroke_width": StrokeWidth, "stroke_dash": StrokeDash,
            "size": Size, "shape": Shape, "angle": Angle,
            "text": Text, "detail": Detail, "tooltip": Tooltip, "tooltip_field": TooltipField,
            "href": Href, "description": Description, "key": Key,
            "facet": Facet, "facet_row": FacetRow, "facet_col": FacetCol,
        })
    return _CHANNEL_CLASSES_BY_NAME.get(name)


class Chart:
    """Top-level chart value class. Immutable — every method returns a new Chart."""

    __slots__ = (
        "_data", "_mark", "_mark_kwargs", "_encoding", "_transforms",
        "_facet", "_coord", "_theme", "_layers",
        "_width", "_height", "_title", "_description",
    )

    def __init__(
        self,
        data: Any = None,
        *,
        width: Optional[Union[int, str]] = None,
        height: Optional[Union[int, str]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        self._data = data
        self._mark = None
        self._mark_kwargs = {}
        self._encoding: dict = {}
        self._transforms: list = []
        self._facet = None
        self._coord = None
        self._theme = None
        self._layers: Optional[list] = None
        self._width = width
        self._height = height
        self._title = title
        self._description = description

    def _clone(self) -> "Chart":
        new = object.__new__(Chart)
        new._data = self._data
        new._mark = self._mark
        new._mark_kwargs = dict(self._mark_kwargs)
        new._encoding = dict(self._encoding)
        new._transforms = list(self._transforms)
        new._facet = self._facet
        new._coord = self._coord
        new._theme = self._theme
        new._layers = None if self._layers is None else list(self._layers)
        new._width = self._width
        new._height = self._height
        new._title = self._title
        new._description = self._description
        return new

    # ---- Marks (primitives) ----

    def _set_mark(self, name: str, **kwargs: Any) -> "Chart":
        m = MarkBase(name, **kwargs)
        new = self._clone()
        new._mark = name
        new._mark_kwargs = m.to_mark_kwargs_dict()
        return new

    def mark_point(self, **kwargs):  return self._set_mark("point", **kwargs)
    def mark_line(self, **kwargs):   return self._set_mark("line", **kwargs)
    def mark_bar(self, **kwargs):    return self._set_mark("bar", **kwargs)
    def mark_area(self, **kwargs):   return self._set_mark("area", **kwargs)
    def mark_rule(self, **kwargs):   return self._set_mark("rule", **kwargs)
    def mark_text(self, **kwargs):   return self._set_mark("text", **kwargs)
    def mark_tick(self, **kwargs):   return self._set_mark("tick", **kwargs)
    def mark_rect(self, **kwargs):   return self._set_mark("rect", **kwargs)

    # ---- Marks (statistical) ----

    def mark_density(self, **kwargs) -> "Chart":
        # Field comes from .encode(x=...) chain; call after .encode() typically
        x_field = self._encoding.get("x")
        if x_field is None:
            raise ValueError("mark_density() requires .encode(x=...) to specify the density field")
        field = x_field.field if isinstance(x_field, ChannelBase) else x_field
        mark, transforms, remap = desugar_density(field, **kwargs)
        new = self._clone()
        new._mark = mark
        new._transforms = list(self._transforms) + transforms
        # Remap encoding
        from ferrum.encoding import Y
        new._encoding["y"] = Y(remap["y"], type="Q")
        return new

    def mark_histogram(self, **kwargs) -> "Chart":
        x_field = self._encoding.get("x")
        if x_field is None:
            raise ValueError("mark_histogram() requires .encode(x=...)")
        field = x_field.field if isinstance(x_field, ChannelBase) else x_field
        mark, transforms, remap = desugar_histogram(field, **kwargs)
        new = self._clone()
        new._mark = mark
        new._transforms = list(self._transforms) + transforms
        from ferrum.encoding import X, X2, Y
        new._encoding["x"] = X(remap["x"], type="Q")
        new._encoding["x2"] = X2(remap["x2"], type="Q")
        new._encoding["y"] = Y(remap["y"], type="Q")
        return new

    def mark_smooth(self, **kwargs) -> "Chart":
        x_enc = self._encoding.get("x")
        y_enc = self._encoding.get("y")
        if x_enc is None or y_enc is None:
            raise ValueError("mark_smooth() requires .encode(x=..., y=...)")
        x_field = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
        y_field = y_enc.field if isinstance(y_enc, ChannelBase) else y_enc
        mark, transforms, remap = desugar_smooth(x_field, y_field, **kwargs)
        new = self._clone()
        new._mark = mark
        new._transforms = list(self._transforms) + transforms
        return new

    # ---- Marks (deferred) ----

    def mark_boxplot(self, **kwargs):       raise deferred_mark_error("boxplot")
    def mark_errorbar(self, **kwargs):      raise deferred_mark_error("errorbar")
    def mark_errorband(self, **kwargs):     raise deferred_mark_error("errorband")
    def mark_ribbon(self, **kwargs):        raise deferred_mark_error("ribbon")
    def mark_contour(self, **kwargs):       raise deferred_mark_error("contour")
    def mark_violin(self, **kwargs):        raise deferred_mark_error("violin")
    def mark_qq(self, **kwargs):            raise deferred_mark_error("qq")
    def mark_raster(self, **kwargs):        raise deferred_mark_error("raster")
    def mark_swarm(self, **kwargs):         raise deferred_mark_error("swarm")
    def mark_hex(self, **kwargs):           raise deferred_mark_error("hex")
    def mark_function(self, fn, **kwargs):  raise deferred_mark_error("function")
    def mark_arc(self, **kwargs):           raise deferred_mark_error("arc")
    def mark_image(self, **kwargs):         raise deferred_mark_error("image")
    def mark_geoshape(self, **kwargs):      raise deferred_mark_error("geoshape")
    def mark_segment(self, **kwargs):       raise deferred_mark_error("segment")
    def mark_label(self, **kwargs):         raise deferred_mark_error("label")

    # ---- Encoding ----

    def encode(self, **channels: Any) -> "Chart":
        new = self._clone()
        for name, value in channels.items():
            cls = _channel_class_for(name)
            if cls is None:
                raise ValueError(f"unknown encoding channel: {name!r}")

            if isinstance(value, ChannelBase):
                channel = value
            elif isinstance(value, str):
                field, type_, agg = parse_shorthand(value)
                kw = {}
                if type_: kw["type"] = type_
                if agg: kw["aggregate"] = agg
                channel = cls(field, **kw)
            else:
                raise TypeError(
                    f"encode({name}=...) expects str or {cls.__name__} instance, "
                    f"got {type(value).__name__}"
                )

            new._encoding[name] = channel
            new._transforms.extend(channel.to_implicit_transforms())
        return new

    def transform(self, *transforms) -> "Chart":
        new = self._clone()
        new._transforms = list(self._transforms) + list(transforms)
        return new

    # ---- Properties ----

    def properties(self, *, width=None, height=None, title=None, description=None) -> "Chart":
        new = self._clone()
        if width is not None: new._width = width
        if height is not None: new._height = height
        if title is not None: new._title = title
        if description is not None: new._description = description
        return new

    # ---- Spec output ----

    def to_spec(self):
        from ferrum import ChartSpec, EncodingSpec
        # Build full EncodingSpec instances per channel so honored kwargs
        # (scale, title) and deferred kwargs (axis, legend, sort, ...) flow to Rust.
        # Phase 7 + 8a's ChartSpec(...) accepts EncodingSpec instances or strings.
        kw = {"mark": self._mark or "point", "data": "default"}
        for axis in ("x", "y", "color", "size", "shape", "opacity"):
            if axis in self._encoding:
                ch = self._encoding[axis]
                if ch.field is None:
                    continue   # Tooltip(*fields) etc. with no single field
                d = ch.to_encoding_spec_dict()
                # `field` is positional; rest are keyword-only on EncodingSpec.__new__.
                # The Python-visible param name is `type_` (Rust signature `type_: Option<&str>`).
                field = d.pop("field")
                kw[axis] = EncodingSpec(field, **d)
        if self._transforms:
            kw["transforms"] = list(self._transforms)
        return ChartSpec(**kw)

    def to_json(self, *, indent=None) -> str:
        spec = self.to_spec()
        return spec.to_json()

    def show_svg(self) -> str:
        # Stub — full impl in Task 32
        from ferrum._core import render_svg
        spec = self.to_spec()
        data = to_arrow_table(self._data)
        viewport = (self._width or 600.0, self._height or 400.0)
        theme_dict = (self._theme.to_theme_inputs_dict() if self._theme else {})
        return render_svg(spec, data, viewport=viewport, theme=theme_dict)

    def show_png(self) -> bytes:
        from ferrum._core import render_png
        spec = self.to_spec()
        data = to_arrow_table(self._data)
        viewport = (self._width or 600.0, self._height or 400.0)
        theme_dict = (self._theme.to_theme_inputs_dict() if self._theme else {})
        return render_png(spec, data, viewport=viewport, theme=theme_dict)

    # Stubs for Phase 11
    def add_selection(self, *selections):
        raise NotImplementedError("selections require .interactive() — Phase 11")

    def interactive(self):
        raise NotImplementedError("interactive renderer — Phase 11")

    def __repr__(self) -> str:
        return f"Chart(mark={self._mark!r}, encoding={list(self._encoding.keys())})"
