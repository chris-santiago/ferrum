"""Public exception types raised by Ferrum's declaration API."""

from __future__ import annotations


class FerrumOverrideError(Exception):
    """Raised when a [override][ferrum.Chart.override] path cannot be resolved.

    ``Chart.override(**kwargs)`` is a flat snake_case escape hatch onto the
    chart's presentation spec.  Because the Rust spec deserializers silently
    drop unknown fields, a misspelled or unsupported path would otherwise be a
    silent no-op.  Override paths are therefore validated Python-side against a
    registry generated from the live schemas; any path that does not resolve
    (unknown prefix, or a known prefix with a leaf that is not a valid field of
    its target) raises this error at render time, naming the offending path and
    — when a close match exists — suggesting the nearest valid path.

    Examples
    --------
    >>> import ferrum as fm
    >>> chart.override(x_axis_lable_angle=-45).to_svg()  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ferrum.exceptions.FerrumOverrideError: Unknown override path
    'x_axis_lable_angle'. Did you mean: 'x_axis_label_angle'?
    """
