"""Public exception types raised by Ferrum's declaration API."""

from __future__ import annotations

from collections.abc import Callable

from ferrum._entry import EntryRefusal


class FerrumOverrideError(ValueError):
    """Raised when a [override][ferrum.Chart.override] path or value cannot be applied.

    A ``ValueError``, like every other refusal of malformed input, built
    from its message alone: ``FerrumOverrideError("...")``. Its text names
    the public call, ``Chart.override``; an override a figure renders on the
    caller's behalf names that figure instead.

    ``Chart.override(**kwargs)`` is a flat snake_case escape hatch onto the
    chart's presentation spec.  Because the Rust spec deserializers silently
    drop unknown fields, a misspelled or unsupported path would otherwise be a
    silent no-op.  Override paths are therefore validated Python-side against a
    registry generated from the live schemas; any path that does not resolve
    (unknown prefix, or a known prefix with a leaf that is not a valid field of
    its target) raises this error at render time, naming the offending path and
    — when a close match exists — suggesting the nearest valid path. A
    ``mark_*`` leaf the chart's mark does not take, or a value of the wrong
    type, raises it too, with the text the style doors refuse it with.

    Examples
    --------
    >>> import ferrum as fm
    >>> chart.override(x_axis_lable_angle=-45).to_svg()  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ferrum.exceptions.FerrumOverrideError: Chart.override: Unknown override path
    'x_axis_lable_angle'. Did you mean: 'x_axis_label_angle'?
    """


class _OverrideRefusal(FerrumOverrideError, EntryRefusal):
    """A :class:`FerrumOverrideError` that ``Chart.override`` raises, re-voiceable as any call.

    Carries the refusal's text for any voice, as every
    :class:`~ferrum._entry.EntryRefusal` does, so a figure that renders an
    override on the caller's behalf re-voices it as itself
    (:func:`ferrum._entry.voiced_as`), while a caller still catches it as
    :class:`FerrumOverrideError`. Private, so the voice protocol stays out
    of the public constructor.
    """

    def __init__(self, message_for: Callable[[str], str], entry: str = "Chart.override") -> None:
        EntryRefusal.__init__(self, message_for, entry)
