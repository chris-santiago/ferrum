"""Warn-once registry: each (channel, kwarg) tuple emits at most one
UserWarning per process. Tests use reset_warnings() to clear state.
"""
from __future__ import annotations

import warnings
from typing import Optional

_seen: set[tuple[str, str]] = set()


def warn_once(channel: str, kwarg: str, message: Optional[str] = None) -> None:
    """Emit a UserWarning the first time this (channel, kwarg) pair is seen.
    Subsequent calls with the same key are silent.
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
    """Clear the warn-once registry. For tests."""
    _seen.clear()
