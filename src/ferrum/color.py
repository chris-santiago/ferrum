"""Programmatic access to ferrum's color palettes.

Provides functions to retrieve hex color arrays from named categorical,
sequential, and diverging palettes.  The Rust palette registry
(``crates/ferrum-core/src/render/palette.rs`` +
``crates/ferrum-core/src/render/color/continuous.rs``) is the single source of
truth: this module consumes it through the ``ferrum._core`` accessors
(``list_palettes``, ``palette_kind``, ``palette_colors``, ``palette_sample``)
rather than re-declaring hex tables.  Continuous palettes are sampled from the
same interpolator the renderer uses, so ``color.palette("viridis")`` returns the
colors that actually render.
"""

from __future__ import annotations

import warnings
from typing import Literal

from ferrum._core import (
    list_palettes as _list_palettes,
    palette_colors as _palette_colors,
    palette_kind as _palette_kind,
    palette_sample as _palette_sample,
)

__all__ = ["palette", "to_hex", "sequential", "diverging"]


# ---------------------------------------------------------------------------
# Registry access helpers (consume the Rust palette registry)
# ---------------------------------------------------------------------------


def _available_palettes() -> list[str]:
    """Return all known palette names, sorted, for error messages."""
    return sorted(_list_palettes())


def _unknown_palette_error(name: str) -> ValueError:
    """Build the canonical ``Unknown palette`` ValueError for *name*."""
    available = ", ".join(_available_palettes())
    return ValueError(f"Unknown palette: {name!r}. Available: {available}")


def _validate_scheme(name: str, *, where: str) -> None:
    """Raise ``ValueError`` if *name* is not a known palette.

    Shared declaration-time validator for ``scheme=`` kwargs.  Routes the
    failure through :func:`_unknown_palette_error` so the message body is the
    canonical ``Unknown palette: 'x'. Available: ...`` shape that :func:`palette`
    raises, then prepends *where* (e.g. ``"Color(scheme=...)"``) as a
    channel-context prefix.  One message body everywhere; only the prefix
    distinguishes the entry point.
    """
    if _palette_kind(name) is None:
        canonical = _unknown_palette_error(name)
        raise ValueError(f"{where}: {canonical}")


def _continuous_at(name: str, n: int) -> list[str]:
    """Sample *n* render-truth colors evenly across a continuous palette.

    Samples ``palette_sample(name, t)`` at ``t = i / (n - 1)`` for
    ``i in 0..n`` so the endpoints are inclusive and the interior matches the
    renderer's interpolation exactly.  ``n == 1`` returns the midpoint sample,
    preserving the previous single-color behaviour.
    """
    if n <= 0:
        return []
    ts = [0.5] if n == 1 else [i / (n - 1) for i in range(n)]
    out: list[str] = []
    for t in ts:
        color = _palette_sample(name, t)
        if color is None:  # pragma: no cover - continuous kind guarantees membership
            raise _unknown_palette_error(name)
        out.append(color)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def palette(name: str, n: int | None = None) -> list[str]:
    """Return hex colors from a named palette.

    Parameters
    ----------
    name : str
        Palette name (e.g., "tableau10", "okabe_ito", "viridis").
    n : int, optional
        Number of colors.  For categorical palettes, wraps cyclically if
        *n* exceeds the palette length.  For continuous palettes,
        interpolates *n* evenly-spaced colors (render-truth samples).

    Returns
    -------
    list[str]
        Hex color strings (e.g., ``["#1f77b4", ...]``).

    Raises
    ------
    ValueError
        If *name* is not a recognized palette.

    Examples
    --------
    >>> import ferrum
    >>> ferrum.color.palette("tableau10")[:3]
    ['#4c78a8', '#f58e18', '#e45756']
    >>> ferrum.color.palette("viridis", n=3)
    ['#440154', '#20908c', '#fde725']
    """
    kind = _palette_kind(name)
    if kind is None:
        raise _unknown_palette_error(name)

    if kind == "categorical":
        colors = _palette_colors(name)
        if colors is None:  # pragma: no cover - kind guarantees membership
            raise _unknown_palette_error(name)
        if n is None:
            return list(colors)
        return [colors[i % len(colors)] for i in range(n)]

    # Sequential or diverging: render-truth continuous samples.
    return _continuous_at(name, n if n is not None else 7)


def to_hex(
    color: tuple[float, ...] | str,
    *,
    scale: Literal["unit", "byte"] | None = None,
) -> str:
    """Convert a color to a hex string.

    Parameters
    ----------
    color : tuple or str
        An RGB tuple with values in [0, 1] (unit) or [0, 255] (byte),
        or a hex string (returned as-is after normalization).
    scale : {"unit", "byte"}, optional
        Explicit interpretation of an RGB tuple's component range.  ``"unit"``
        treats components as floats in ``[0, 1]``; ``"byte"`` treats them as
        integers in ``[0, 255]``.  When ``None`` (default) the range is
        inferred: any component greater than 1 forces byte interpretation;
        otherwise unit interpretation is used.  This makes integer-valued
        floats (``1.0``) and integers (``1``) behave identically.

    Returns
    -------
    str
        Hex string like ``"#1f77b4"``.

    Raises
    ------
    ValueError
        If the input format is not recognized, or *scale* is not one of
        ``"unit"``, ``"byte"``, or ``None``.

    Examples
    --------
    >>> import ferrum
    >>> ferrum.color.to_hex((1.0, 0.0, 0.0))
    '#ff0000'
    >>> ferrum.color.to_hex((255, 0, 0))
    '#ff0000'
    >>> ferrum.color.to_hex((128, 128, 128), scale="byte")
    '#808080'
    """
    if isinstance(color, str):
        # Normalize: strip whitespace, ensure lowercase
        s = color.strip().lower()
        if not s.startswith("#"):
            raise ValueError(f"String colors must be hex format (#rrggbb), got: {color!r}")
        return s

    if not isinstance(color, (tuple, list)) or len(color) < 3:
        raise ValueError(f"Expected an RGB tuple (r, g, b) or hex string, got: {color!r}")

    if scale not in (None, "unit", "byte"):
        raise ValueError(f"scale must be 'unit', 'byte', or None, got: {scale!r}")

    r, g, b = color[0], color[1], color[2]

    if scale is None:
        # Range-based heuristic: any component above 1 means a byte ([0, 255])
        # encoding; otherwise treat as unit ([0, 1]).  Inspecting the value
        # range (not the Python type) keeps 1.0 and 1 in the same bucket.
        if max(r, g, b) > 1:
            scale = "byte"
            # Warn when the input looks like a fractional unit-range overshoot
            # (e.g. (0.9, 0.9, 1.1) from float color math) rather than an
            # unambiguous integer-valued byte tuple like (255, 0, 0).  The
            # discriminator: at least one component is non-integer-valued.
            # Integer-valued byte tuples are unambiguous and stay silent.
            if any(c != int(c) for c in (r, g, b) if isinstance(c, float)):
                warnings.warn(
                    f"to_hex: ambiguous color tuple {(r, g, b)!r} — "
                    "at least one component is non-integer and exceeds the unit range [0, 1]. "
                    "Defaulting to byte interpretation ([0, 255]). "
                    'Pass scale="unit" or scale="byte" to suppress this warning.',
                    UserWarning,
                    stacklevel=2,
                )
        else:
            scale = "unit"

    if scale == "unit":
        ri = int(round(float(r) * 255))
        gi = int(round(float(g) * 255))
        bi = int(round(float(b) * 255))
    else:  # "byte"
        ri, gi, bi = int(r), int(g), int(b)

    ri = max(0, min(255, ri))
    gi = max(0, min(255, gi))
    bi = max(0, min(255, bi))
    return f"#{ri:02x}{gi:02x}{bi:02x}"


def sequential(name: str, n: int = 256) -> list[str]:
    """Return *n* interpolated colors from a sequential palette.

    Parameters
    ----------
    name : str
        Sequential palette name (e.g., "viridis", "plasma", "magma",
        "inferno", "cividis", "blues", "cool_blue", "warm_ochre",
        "night_blue", "electric_lime", "signal_blue", "ember_orange").
    n : int, default 256
        Number of interpolated colors.

    Returns
    -------
    list[str]
        Hex color strings (render-truth samples).

    Raises
    ------
    ValueError
        If *name* is not a recognized sequential palette.

    Examples
    --------
    >>> import ferrum
    >>> colors = ferrum.color.sequential("viridis", n=5)
    >>> len(colors)
    5
    """
    if _palette_kind(name) != "sequential":
        available = ", ".join(
            sorted(p for p in _list_palettes() if _palette_kind(p) == "sequential")
        )
        raise ValueError(f"Unknown sequential palette: {name!r}. Available: {available}")
    return _continuous_at(name, n)


def diverging(name: str, n: int = 11) -> list[str]:
    """Return *n* colors from a diverging palette, centered.

    Parameters
    ----------
    name : str
        Diverging palette name (e.g., "rdbu", "blue_to_red",
        "cyan_to_amber", "blue_to_violet").
    n : int, default 11
        Number of colors (odd recommended for a distinct center point).

    Returns
    -------
    list[str]
        Hex color strings (render-truth samples).

    Raises
    ------
    ValueError
        If *name* is not a recognized diverging palette.

    Examples
    --------
    >>> import ferrum
    >>> colors = ferrum.color.diverging("rdbu", n=5)
    >>> len(colors)
    5
    """
    if _palette_kind(name) != "diverging":
        available = ", ".join(
            sorted(p for p in _list_palettes() if _palette_kind(p) == "diverging")
        )
        raise ValueError(f"Unknown diverging palette: {name!r}. Available: {available}")
    return _continuous_at(name, n)
