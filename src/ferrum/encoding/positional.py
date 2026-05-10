"""Positional encoding channels (X, Y, X2, Y2, errors, polar)."""
from __future__ import annotations

from ferrum.encoding.base import ChannelBase


_RENDERED_HONORED = frozenset(["type", "bin", "aggregate", "scale", "title"])


class X(ChannelBase):
    _channel_name = "x"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


class Y(ChannelBase):
    _channel_name = "y"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


class X2(ChannelBase):
    _channel_name = "x2"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Y2(ChannelBase):
    _channel_name = "y2"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class XError(ChannelBase):
    _channel_name = "x_error"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class YError(ChannelBase):
    _channel_name = "y_error"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class XError2(ChannelBase):
    _channel_name = "x_error2"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class YError2(ChannelBase):
    _channel_name = "y_error2"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])


class Theta(ChannelBase):
    _channel_name = "theta"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type", "stack"])


class Radius(ChannelBase):
    _channel_name = "radius"
    _renders_in_phase_8a = False
    _honored_kwargs = frozenset(["type"])
