"""Process-level configuration store for ferrum defaults.

Uses ``contextvars.ContextVar`` for thread-safe, scope-bounded defaults.
The same pattern as ``set_default_theme()`` in ``ferrum.themes._defaults``.

Precedence (highest wins):
    1. Explicit per-chart ``.properties()``
    2. ``ferrum.config`` overrides
    3. Built-in defaults
"""

from __future__ import annotations

import contextlib
import contextvars
from contextlib import AbstractContextManager
from typing import Any

__all__ = ["set", "get", "defaults", "reset"]

_BUILTIN_DEFAULTS: dict[str, Any] = {
    "width": 640,
    "height": 480,
    "renderer": "svg",
    "max_rows": 5000,
    "raster_threshold": 500_000,
}

_config_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "ferrum_config",
    default={},
)


def set(**kwargs: Any) -> None:
    """Set process-level configuration defaults.

    Parameters
    ----------
    **kwargs
        Key-value pairs to store.  Recognized keys: ``width``, ``height``,
        ``renderer``, ``max_rows``, ``raster_threshold``.

    Examples
    --------
    >>> import ferrum
    >>> ferrum.config.set(width=800, height=600)
    >>> ferrum.config.get("width")
    800
    """
    current = _config_var.get()
    merged = {**current, **kwargs}
    _config_var.set(merged)


def get(key: str) -> Any:
    """Get a configuration value (user-set or built-in default).

    Parameters
    ----------
    key : str
        Configuration key.

    Returns
    -------
    Any
        The configured value, or the built-in default if not overridden.

    Raises
    ------
    KeyError
        If *key* is not a recognized configuration key and has not been
        set by the user.

    Examples
    --------
    >>> import ferrum
    >>> ferrum.config.get("width")
    640
    """
    current = _config_var.get()
    if key in current:
        return current[key]
    if key in _BUILTIN_DEFAULTS:
        return _BUILTIN_DEFAULTS[key]
    raise KeyError(f"Unknown config key: {key!r}")


@contextlib.contextmanager
def defaults(**kwargs: Any) -> AbstractContextManager:  # type: ignore[misc]
    """Context manager for temporary config overrides.

    Restores the prior configuration state on exit.

    Parameters
    ----------
    **kwargs
        Temporary configuration overrides active within the ``with`` block.

    Yields
    ------
    None

    Examples
    --------
    >>> import ferrum
    >>> with ferrum.config.defaults(width=1000):
    ...     assert ferrum.config.get("width") == 1000
    >>> assert ferrum.config.get("width") == 640
    """
    old = _config_var.get()
    merged = {**old, **kwargs}
    token = _config_var.set(merged)
    try:
        yield
    finally:
        _config_var.reset(token)


def reset() -> None:
    """Reset all config to built-in defaults.

    Removes all user-set overrides so subsequent ``get()`` calls return
    the built-in defaults.  This only affects the current context — child
    async tasks that already inherited a prior ``ContextVar`` snapshot
    retain their own copy and are not affected by this call.

    Examples
    --------
    >>> import ferrum
    >>> ferrum.config.set(width=1200)
    >>> ferrum.config.reset()
    >>> ferrum.config.get("width")
    640
    """
    _config_var.set({})
