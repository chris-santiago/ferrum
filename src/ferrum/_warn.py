"""Warn-once registry: each (channel, kwarg) tuple emits at most one
UserWarning per context. Tests use reset_warnings() to clear state.

State is stored in a ``contextvars.ContextVar`` so it is scoped to the
current execution context (thread-safe and async-safe) rather than
process-global.
"""

from __future__ import annotations

import contextvars
import warnings
from typing import Optional

_seen_var: contextvars.ContextVar[set | None] = contextvars.ContextVar(
    "_ferrum_warn_seen", default=None
)


def _get_seen() -> set:
    """Return the context-local seen set, initializing it on first access."""
    s = _seen_var.get()
    if s is None:
        s = set()
        _seen_var.set(s)
    return s


def warn_once(
    channel: str, kwarg: str, message: Optional[str] = None, *, stacklevel: int = 3
) -> None:
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
    stacklevel : int, default 3
        Passed through to ``warnings.warn``. The default of ``3`` is tuned
        for the common shape "user code -> one ferrum helper -> `warn_once`"
        (1 = inside this function, 2 = the helper, 3 = the helper's caller).
        A caller with an *extra* frame between the user and this function
        (e.g. a small wrapper that itself calls `warn_once`) should pass a
        correspondingly higher value so the emitted warning still points at
        the user's own call site rather than into ferrum's source.

    Examples
    --------
    >>> warn_once("mark_foo", "bar", "mark_foo bar= not yet supported")
    >>> warn_once("mark_foo", "bar")  # second call — silently suppressed
    """
    key = (channel, kwarg)
    seen = _get_seen()
    if key in seen:
        return
    seen.add(key)
    msg = message or (
        f"{channel}({kwarg}=...) is accepted but not yet honored; planned for a future Phase."
    )
    warnings.warn(msg, UserWarning, stacklevel=stacklevel)


def reset_warnings() -> None:
    """Clear the warn-once registry for the current context.

    Intended for use in tests so that ``warn_once`` side effects from one
    test case do not bleed into the next.

    Examples
    --------
    >>> reset_warnings()
    """
    _seen_var.set(None)
