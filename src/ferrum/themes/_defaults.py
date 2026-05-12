"""Built-in theme defaults and process-default theme controls."""

from __future__ import annotations

import contextvars

from ferrum.themes import Theme, default as _ferrum_default


_default_theme: contextvars.ContextVar[Theme] = contextvars.ContextVar(
    "_ferrum_default_theme",
    default=_ferrum_default,
)


class _DefaultThemeCM:
    """Context manager returned by ``set_default_theme()``.

    Restores the prior process-default theme on ``__exit__``. Also usable as
    a plain object for fire-and-forget ``set_default_theme(t)`` calls.
    """

    __slots__ = ("_token",)

    def __init__(self, token: contextvars.Token) -> None:
        self._token = token

    def __enter__(self) -> "_DefaultThemeCM":
        """Enter the context manager — returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore the previous process-default theme."""
        _default_theme.reset(self._token)


def set_default_theme(theme: Theme) -> _DefaultThemeCM:
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
    _DefaultThemeCM
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
    return _DefaultThemeCM(token)


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


def theme_context(theme: Theme) -> _DefaultThemeCM:
    """Scope a theme to a ``with`` block — alias for ``set_default_theme()``.

    Prefer this spelling over ``set_default_theme()`` when the intent is
    always context-manager usage (e.g. in tests or notebook cells).

    Parameters
    ----------
    theme : Theme
        Theme to activate for the duration of the ``with`` block.

    Returns
    -------
    _DefaultThemeCM
        Context manager; restores the prior default on ``__exit__``.

    Examples
    --------
    >>> import ferrum as fm
    >>> with fm.theme_context(fm.themes.dark):
    ...     chart = fm.Chart(df).mark_point().encode(x="hp", y="mpg")
    """
    return set_default_theme(theme)
