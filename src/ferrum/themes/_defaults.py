"""Built-in theme defaults and process-default theme controls."""

from __future__ import annotations

import contextlib
import contextvars
from contextlib import AbstractContextManager

from ferrum.themes import Theme, default as _ferrum_default


_default_theme: contextvars.ContextVar[Theme] = contextvars.ContextVar(
    "_ferrum_default_theme",
    default=_ferrum_default,
)


@contextlib.contextmanager
def _default_theme_cm(token: contextvars.Token):
    """Context manager that resets the default theme token on exit."""
    try:
        yield
    finally:
        _default_theme.reset(token)


def set_default_theme(theme: Theme) -> AbstractContextManager:
    """Set the process-default theme for all subsequent charts.

    The returned object is a context manager. Use it with ``with`` to scope
    the default to a block (previous default is restored on ``__exit__``).
    Fire-and-forget usage (without ``with``) is also supported.

    Per-chart ``Chart.theme(t)`` always overrides this process default.

    Parameters
    ----------
    theme : Theme
        Theme to install as the process default.

    Returns
    -------
    contextlib.AbstractContextManager
        Context manager that restores the prior default on ``__exit__``.

    Raises
    ------
    TypeError
        If ``theme`` is not a ``Theme`` instance.

    Examples
    --------
    Fire-and-forget:

    >>> import ferrum as fm
    >>> fm.set_default_theme(fm.themes.dark)

    Scoped to a block:

    >>> with fm.set_default_theme(fm.themes.dark):
    ...     chart = fm.Chart(df).mark_point().encode(x="hp", y="mpg")
    """
    if not isinstance(theme, Theme):
        raise TypeError(f"theme must be a Theme instance, got {type(theme).__name__}")
    token = _default_theme.set(theme)
    return _default_theme_cm(token)


def get_default_theme() -> Theme:
    """Return the current process-default theme.

    Returns
    -------
    Theme
        The currently active process-default theme.  Starts as
        ``ferrum.themes.default`` (Rust renderer defaults); changes after
        each ``set_default_theme()`` call.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.get_default_theme()
    Theme()
    """
    return _default_theme.get()


def theme_context(theme: Theme) -> AbstractContextManager:
    """Scope a theme to a ``with`` block — alias for ``set_default_theme()``.

    Prefer this spelling over ``set_default_theme()`` when the intent is
    always context-manager usage (e.g. in tests or notebook cells).

    Parameters
    ----------
    theme : Theme
        Theme to activate for the duration of the ``with`` block.

    Returns
    -------
    contextlib.AbstractContextManager
        Context manager; restores the prior default on ``__exit__``.

    Examples
    --------
    >>> import ferrum as fm
    >>> with fm.theme_context(fm.themes.dark):
    ...     chart = fm.Chart(df).mark_point().encode(x="hp", y="mpg")
    """
    return set_default_theme(theme)
