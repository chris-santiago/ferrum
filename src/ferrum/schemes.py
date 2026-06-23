"""Color scheme lookups — ``continuous_palette`` and the ``Gradient`` factory.

This module exposes the ``ContinuousScheme`` factory used by ``Color(scale=...)``.
For hex-string palette lookups (``palette``/``sequential``/``diverging``) see
:mod:`ferrum.color`, which is the single name-keyed entry point over the Rust
palette registry.  Both consume the same registry (``ferrum._core``); there are
no hand-mirrored color tables in Python.
"""

from __future__ import annotations

from ferrum._core import ContinuousScheme as _ContinuousScheme
from ferrum._core import Gradient as _Gradient
from ferrum._core import list_palettes as _list_palettes
from ferrum._core import palette_kind as _palette_kind


def continuous_palette(name: str):
    """Look up a built-in continuous colormap by name.

    Parameters
    ----------
    name : str
        Built-in continuous colormap name (e.g. ``"viridis"``, ``"plasma"``,
        ``"magma"``, ``"inferno"``, ``"cividis"``, ``"blues"``, ``"rdbu"``).
        See ``continuous_palette.list()`` for the full set.

    Returns
    -------
    ContinuousScheme
        A ferrum continuous scheme suitable for ``Color(scale=...)``.

    Raises
    ------
    ValueError
        If ``name`` is not one of the built-in colormaps.

    Examples
    --------
    >>> import ferrum as fm
    >>> scheme = fm.continuous_palette("viridis")
    >>> fm.Chart(df).encode(x="x", y="y", color=fm.Color("val", scale=scheme))
    """
    return _ContinuousScheme.from_name(name)


def _list_continuous() -> list[str]:
    """Return all continuous (sequential + diverging) palette names.

    Derived from the Rust palette registry so the list never drifts from the
    schemes ``ContinuousScheme.from_name`` actually accepts.
    """
    return [n for n in _list_palettes() if _palette_kind(n) in ("sequential", "diverging")]


# Attach `.list()` as a function attribute so callers can do
# `fe.continuous_palette.list()` without a separate import.
continuous_palette.list = _list_continuous  # type: ignore[attr-defined]

# Re-export the Rust-backed Gradient factory at the Python level.
Gradient = _Gradient

__all__ = ["continuous_palette", "Gradient"]
