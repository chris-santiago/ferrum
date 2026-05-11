"""Color scheme lookups — ``continuous_palette`` and the ``Gradient`` factory."""
from __future__ import annotations

from ferrum._core import ContinuousScheme as _ContinuousScheme
from ferrum._core import Gradient as _Gradient


def continuous_palette(name: str):
    """Look up a built-in continuous colormap by name.

    Parameters
    ----------
    name : {"viridis", "plasma", "magma", "inferno", "cividis"}
        Built-in colormap name.

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
    return ["viridis", "plasma", "magma", "inferno", "cividis"]


# Attach `.list()` as a function attribute so callers can do
# `fe.continuous_palette.list()` without a separate import.
continuous_palette.list = _list_continuous  # type: ignore[attr-defined]

# Re-export the Rust-backed Gradient factory at the Python level.
Gradient = _Gradient

__all__ = ["continuous_palette", "Gradient"]
