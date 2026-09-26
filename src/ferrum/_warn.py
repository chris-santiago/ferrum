"""Warn-once registry: each (channel, kwarg, message) triple emits at most
one UserWarning per context. Tests use reset_warnings() to clear state.

``user_stacklevel`` is the ``stacklevel`` for a warning raised at any depth
inside ``ferrum``, naming the first caller frame outside the package.

State is stored in a ``contextvars.ContextVar`` so it is scoped to the
current execution context (thread-safe and async-safe) rather than
process-global.

Identity is ``(channel, kwarg, message)``, not just ``(channel, kwarg)``
(#240, 2026-09-11 wave-1 hardening). Before this, a second caller warning
about the same pair with a DIFFERENT message was silently dropped -- the
first caller's text permanently won that slot, so distinct facts (e.g. two
different marks both warning on ``encoding("detail", ...)``, each naming its
own mark) collapsed into one, wrong-for-the-second-caller message. Including
the resolved message in the key means "same pair, same message" still dedups
(the common case: a loop re-warning about the identical fact) while "same
pair, different message" reaches the user once per distinct message.
"""

from __future__ import annotations

import contextvars
import os
import sys
import warnings
from typing import Optional

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep

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
    """Emit a ``UserWarning`` the first time a ``(channel, kwarg, message)`` triple is seen.

    Subsequent calls with the same ``(channel, kwarg)`` pair AND the same
    resolved message are silently suppressed. A later call for the same pair
    with a DIFFERENT message is NOT suppressed -- two distinct one-time facts
    about the same ``(channel, kwarg)`` slot both reach the user, once each.
    Used by desugar functions to warn about accepted-but-not-yet-honored
    kwargs without spamming per-render.

    Parameters
    ----------
    channel : str
        The mark or feature name (e.g. ``"mark_raster"``).
    kwarg : str
        The specific kwarg or feature being warned about
        (e.g. ``"blend_additive"``).
    message : str or None, default None
        Custom warning text. If ``None``, a default message is constructed
        from ``channel`` and ``kwarg``; this default text is itself part of
        the key, so a caller that always passes ``None`` still dedups
        against itself the same way it always has.
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
    >>> warn_once("mark_foo", "bar", "mark_foo bar= not yet supported")  # dup — suppressed
    >>> warn_once("mark_foo", "bar", "a different fact about bar=")  # distinct message — emits
    """
    msg = message or (
        f"{channel}({kwarg}=...) is accepted but not yet honored; planned for a future Phase."
    )
    key = (channel, kwarg, msg)
    seen = _get_seen()
    if key in seen:
        return
    seen.add(key)
    warnings.warn(msg, UserWarning, stacklevel=stacklevel)


def user_stacklevel() -> int:
    """Return the ``stacklevel`` naming the first frame outside ``ferrum``.

    For a ``warnings.warn`` called directly by the function that calls this
    one, when that function runs at a depth that depends on the public entry
    the user called (a desugar reached from ``to_json``, ``to_svg``, a
    composition render, ...), so no fixed ``stacklevel`` names the user's
    line.
    """
    frame = sys._getframe(1)  # the function that will call warnings.warn
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame = frame.f_back
        level += 1
    return level


def reset_warnings() -> None:
    """Clear the warn-once registry for the current context.

    Intended for use in tests so that ``warn_once`` side effects from one
    test case do not bleed into the next.

    Examples
    --------
    >>> reset_warnings()
    """
    _seen_var.set(None)
