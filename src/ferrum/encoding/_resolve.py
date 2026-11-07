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


def _is_column_shorthand(value: str) -> bool:
    """Whether *value* is genuine encoding shorthand, not a column's own literal name.

    ``parse_shorthand`` matches a string like ``"f(h)"`` structurally (an
    aggregate prefix and a field), but ``"f"`` is not a real aggregate --
    the column is named ``"f(h)"``, not an unknown-function error waiting
    to happen. A string parses as shorthand here only when
    ``parse_shorthand`` succeeds AND, when it reads an aggregate prefix,
    that prefix is a member of the aggregate vocabulary Rust publishes
    (:func:`ferrum._registry.accepts_aggregate`). A string
    ``parse_shorthand`` cannot read at all (unbalanced parens, a
    non-shorthand ``":"``) is not shorthand either.
    """
    from ferrum._registry import accepts_aggregate

    try:
        _, _, agg = parse_shorthand(value)
    except ValueError:
        return False
    return agg is None or accepts_aggregate("aggregate", agg)


def channel_for_column(name: str, value: Any, *, declare: str) -> ChannelBase:
    """Return the channel object naming the column *value* for channel *name*.

    The reading for a keyword that names a column of the caller's data --
    a figure's ``hue=``, ``Chart.facet``'s ``field=``/``row=``/``col=``,
    ``residplot``'s ``color=`` once a literal color has been ruled out --
    as opposed to :func:`channel_for_value`'s strict ``encode()`` grammar.
    A non-string *value* (a channel object, a ``Repeat`` placeholder, or
    anything :func:`channel_for_value` would refuse) reads exactly as
    :func:`channel_for_value` reads it. A string that IS encoding
    shorthand (:func:`_is_column_shorthand`) reads the same way too, so
    ``"g:N"`` and ``"mean(k)"`` mean what they mean on ``encode()``. A
    string that is not shorthand -- the grammar cannot read it
    (``"Weight (kg)"``), or its prefix names no real aggregate
    (``"f(h)"``) -- is the column it names, bound exactly as
    ``channel_class(name)(value)`` binds it (the same literal-string
    reading :class:`~ferrum.encoding.base.ChannelBase` already falls back
    to for a non-shorthand ``":"``).

    Raises
    ------
    ValueError
        If *name* is not an encoding channel.
    TypeError
        If *value* is not a string, a channel object or a ``Repeat`` placeholder.
    """
    if isinstance(value, str) and not _is_column_shorthand(value):
        return channel_class(name)(value)
    return channel_for_value(name, value, declare=declare)
