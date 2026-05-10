"""Process-default theme stack backed by contextvars.

Per spec §10 row 11: the only sanctioned global theme state.
Per-chart Chart.theme(t) always wins over this default.
"""
from __future__ import annotations

import contextvars

from ferrum.themes import Theme, default as _ferrum_default


_default_theme: contextvars.ContextVar[Theme] = contextvars.ContextVar(
    "_ferrum_default_theme", default=_ferrum_default,
)


class _DefaultThemeCM:
    """Context manager returned by set_default_theme(). Restores prior default on __exit__.
    Also acts as a plain object for fire-and-forget set_default_theme(t) usage."""

    __slots__ = ("_token",)

    def __init__(self, token: contextvars.Token) -> None:
        self._token = token

    def __enter__(self) -> "_DefaultThemeCM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _default_theme.reset(self._token)


def set_default_theme(theme: Theme) -> _DefaultThemeCM:
    """Set the process-default theme. Per-chart Chart.theme(t) overrides this.

    Returns a context manager that restores the previous default on __exit__.
    Fire-and-forget usage (without `with`) is also supported — the previous
    theme stays restorable via the returned token.
    """
    if not isinstance(theme, Theme):
        raise TypeError(f"theme must be a Theme instance, got {type(theme).__name__}")
    token = _default_theme.set(theme)
    return _DefaultThemeCM(token)


def get_default_theme() -> Theme:
    """Return the current process-default theme."""
    return _default_theme.get()


def theme_context(theme: Theme) -> _DefaultThemeCM:
    """Alias for set_default_theme() — explicit context-manager spelling."""
    return set_default_theme(theme)
