"""Color scheme lookups (Phase 8a categorical + Phase 8b continuous).

Public surface:

- :func:`continuous_palette(name)` — look up one of the built-in continuous
  colormaps (``viridis``, ``plasma``, ``magma``, ``inferno``, ``cividis``).
- :func:`continuous_palette.list()` — return the list of built-in names.
- :class:`Gradient` — construct a custom gradient scheme from a list of
  ``(t, color)`` stops.
"""
from __future__ import annotations

from ferrum._core import ContinuousScheme as _ContinuousScheme
from ferrum._core import Gradient as _Gradient


def continuous_palette(name: str):
    """Look up a built-in continuous colormap by name.

    Parameters
    ----------
    name : str
        One of ``"viridis"``, ``"plasma"``, ``"magma"``, ``"inferno"``,
        ``"cividis"``.

    Returns
    -------
    ContinuousScheme

    Raises
    ------
    ValueError
        If ``name`` is not one of the known colormaps.
    """
    return _ContinuousScheme.from_name(name)


def _list_continuous() -> list[str]:
    return ["viridis", "plasma", "magma", "inferno", "cividis"]


# Attach `.list()` as a function attribute so callers can do
# `fe.continuous_palette.list()` without a separate import.
continuous_palette.list = _list_continuous  # type: ignore[attr-defined]

# Re-export the Rust-backed Gradient factory at the Python level.
Gradient = _Gradient

__all__ = ["continuous_palette", "Gradient"]
