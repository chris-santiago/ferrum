"""Warn-once registry: each (channel, kwarg) tuple emits at most one
UserWarning per process. Tests use reset_warnings() to clear state.
"""
from __future__ import annotations

import warnings
from typing import Optional

_seen: set[tuple[str, str]] = set()


def warn_once(channel: str, kwarg: str, message: Optional[str] = None) -> None:
    """Emit a ``UserWarning`` the first time a ``(channel, kwarg)`` pair is seen.

    Subsequent calls with the same ``(channel, kwarg)`` key are silently
    suppressed.  Used by desugar functions to warn about accepted-but-
    not-yet-honored kwargs without spamming per-render.

    Parameters
    ----------
    channel : str
        The mark or feature name (e.g. ``"mark_raster"``).
    kwarg : str
        The specific kwarg or feature being warned about
        (e.g. ``"blend_additive"``).
    message : str or None, default None
        Custom warning text.  If ``None``, a default message is
        constructed from ``channel`` and ``kwarg``.

    Examples
    --------
    >>> warn_once("mark_foo", "bar", "mark_foo bar= not yet supported")
    >>> warn_once("mark_foo", "bar")  # second call — silently suppressed
    """
    key = (channel, kwarg)
    if key in _seen:
        return
    _seen.add(key)
    msg = message or (
        f"{channel}({kwarg}=...) is accepted but not honored in Phase 8a; "
        f"planned for Phase 9."
    )
    warnings.warn(msg, UserWarning, stacklevel=3)


def reset_warnings() -> None:
    """Clear the warn-once registry.

    Intended for use in tests so that ``warn_once`` side effects from one
    test case do not bleed into the next.

    Examples
    --------
    >>> reset_warnings()
    """
    _seen.clear()
