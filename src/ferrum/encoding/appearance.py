"""Appearance encoding channels (Color, Fill, Stroke, Opacity, Size, Shape, Angle)."""
from __future__ import annotations

from ferrum.encoding.base import ChannelBase


_RENDERED_HONORED = frozenset(["type", "scale", "title"])


# Phase 8a renders these (added to scale_resolve in Task 8):
class Color(ChannelBase):
    # `scheme` is honored ONLY for Color in Phase 8a (Task 10 wires it through
    # palette.rs::categorical_palette into the renderer's color-scale construction).
    # All other channels treat `scheme` as deferred → warn-once.
    _channel_name = "color"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type", "scheme", "scale", "title"])


class Size(ChannelBase):
    _channel_name = "size"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


class Shape(ChannelBase):
    _channel_name = "shape"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


class Opacity(ChannelBase):
    _channel_name = "opacity"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


# Deferred to Phase 9:
class Fill(ChannelBase):
    _channel_name = "fill"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Stroke(ChannelBase):
    _channel_name = "stroke"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class FillOpacity(ChannelBase):
    _channel_name = "fill_opacity"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class StrokeOpacity(ChannelBase):
    _channel_name = "stroke_opacity"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class StrokeWidth(ChannelBase):
    _channel_name = "stroke_width"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class StrokeDash(ChannelBase):
    _channel_name = "stroke_dash"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Angle(ChannelBase):
    _channel_name = "angle"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])
