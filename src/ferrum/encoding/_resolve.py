"""The one reading of an encoding value as a channel, shared by ``encode()`` and ``Layer(encoding=)``.

A channel object is taken as written, a string is read as encoding shorthand
(``"sum(y)"``, ``"r:Q"``, ``"count()"``) into the channel object it spells,
and a ``Repeat`` placeholder becomes a channel over the placeholder. So a
string means the same channel on every door that accepts one. Every door
checks the key with :func:`channel_class` before it reads the value, and
``None`` binds nothing.
"""

from __future__ import annotations

from typing import Any

from ferrum._shorthand import parse_shorthand
from ferrum.encoding.base import ChannelBase


def channel_class(name: str) -> type[ChannelBase]:
    """Return the channel class an encoding key names.

    Checked before anything reads the key's value, so an unknown name is
    refused whatever value it carries, ``None`` and a conditional included.

    Raises
    ------
    ValueError
        If *name* is not an encoding channel.
    """
    from ferrum.encoding import _channel_class_for

    cls = _channel_class_for(name)
    if cls is None:
        raise ValueError(f"unknown encoding channel: {name!r}")
    return cls


def channel_for_value(name: str, value: Any, *, declare: str) -> ChannelBase:
    """Return the channel object an encoding value for channel *name* stands for.

    *declare* is the public spelling of the binding that received *value*
    (``"encode(y=...)"``, ``"Layer(encoding={'y': ...})"``), named by the
    refusal of a value that is no binding.

    Raises
    ------
    ValueError
        If *name* is not an encoding channel.
    TypeError
        If *value* is not a string, a channel object or a ``Repeat`` placeholder.
    """
    from ferrum.repeat import _RepeatPlaceholder

    cls = channel_class(name)
    if isinstance(value, ChannelBase):
        return value
    if isinstance(value, str):
        field, type_, agg = parse_shorthand(value)
        kw = {}
        if type_:
            kw["type"] = type_
        if agg:
            kw["aggregate"] = agg
        return cls(field, **kw)
    if isinstance(value, _RepeatPlaceholder):
        return cls(value)
    raise TypeError(
        f"{declare} expects str, {cls.__name__} instance, or Repeat placeholder; "
        f"got {type(value).__name__}"
    )
