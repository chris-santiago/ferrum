"""Programmatic access to ferrum's color palettes.

Provides functions to retrieve hex color arrays from named categorical,
sequential, and diverging palettes.  Categorical palettes mirror the Rust
palette registry; sequential/diverging palettes interpolate using the same
stop definitions as the Rust renderer.
"""

from __future__ import annotations

__all__ = ["palette", "to_hex", "sequential", "diverging"]

# ---------------------------------------------------------------------------
# Categorical palette definitions (mirrored from palette.rs)
# ---------------------------------------------------------------------------

_CATEGORICAL: dict[str, list[str]] = {
    "okabe_ito": [
        "#e69f00",
        "#56b4e9",
        "#009e73",
        "#f0e442",
        "#0072b2",
        "#d55e00",
        "#cc79a7",
        "#000000",
    ],
    "tableau10": [
        "#4c78a8",
        "#f58e18",
        "#e45756",
        "#72b7b2",
        "#54a24b",
        "#eeca3b",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
        "#bab0ac",
    ],
    "set1": [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ],
    "set2": [
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
    ],
    "paired": [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
    ],
    "pastel": [
        "#fbb4ae",
        "#b3cde3",
        "#ccebc5",
        "#decbe4",
        "#fed9a6",
        "#ffffcc",
        "#e5d8bd",
        "#fddaec",
        "#f2f2f2",
    ],
    "dark2": [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ],
    "paper_ink": [
        "#2563eb",
        "#dc2626",
        "#d4a017",
        "#0f766e",
        "#7c3aed",
        "#ea580c",
        "#4b5563",
        "#db2777",
    ],
    "slate_citrus": [
        "#60a5fa",
        "#a78bfa",
        "#a3e635",
        "#f59e0b",
        "#34d399",
        "#f472b6",
        "#f87171",
        "#22d3ee",
    ],
    "arctic_signal": [
        "#0284c7",
        "#7c3aed",
        "#ea580c",
        "#16a34a",
        "#dc2626",
        "#0891b2",
        "#ca8a04",
        "#db2777",
    ],
}

# ---------------------------------------------------------------------------
# Sequential/diverging stop definitions (mirrored from continuous.rs)
# ---------------------------------------------------------------------------

# Each entry maps a palette name to a list of hex stops evenly spaced in [0,1].
# The "colorous-backed" palettes (viridis, plasma, magma, inferno, cividis,
# blues, rdbu) use pre-sampled 7-stop approximations adequate for Python-side
# interpolation at reasonable n (up to ~256).

_SEQUENTIAL_STOPS: dict[str, list[int]] = {
    "viridis": [0x440154, 0x443A83, 0x31688E, 0x21918C, 0x35B779, 0x90D743, 0xFDE725],
    "plasma": [0x0D0887, 0x6A00A8, 0xB12A90, 0xE16462, 0xFCA636, 0xEFF821, 0xFCFFA4],
    "magma": [0x000004, 0x221150, 0x5F187F, 0xB5367A, 0xFB8861, 0xFCFDBF, 0xFCFDBF],
    "inferno": [0x000004, 0x210C4A, 0x57106E, 0x9E2F7F, 0xF1605D, 0xFEBE2D, 0xFCFFA4],
    "cividis": [0x002051, 0x255166, 0x526D7A, 0x7B8A68, 0xA7A94B, 0xD3C836, 0xFDE725],
    "blues": [0xF7FBFF, 0xDEEBF7, 0xC6DBEF, 0x9ECAE1, 0x6BAED6, 0x3182BD, 0x08519C],
    "cool_blue": [0xEFF6FF, 0xDBEAFE, 0x93C5FD, 0x60A5FA, 0x2563EB, 0x1D4ED8, 0x1E3A8A],
    "warm_ochre": [0xFFF7E6, 0xFDECC8, 0xF8D88A, 0xD4A017, 0xB45309, 0x92400E, 0x78350F],
    "night_blue": [0x1E293B, 0x1D4ED8, 0x2563EB, 0x60A5FA, 0x93C5FD, 0xBFDBFE, 0xE0F2FE],
    "electric_lime": [0x365314, 0x4D7C0F, 0x65A30D, 0x84CC16, 0xA3E635, 0xBEF264, 0xD9F99D],
    "signal_blue": [0xF0F9FF, 0xE0F2FE, 0x7DD3FC, 0x38BDF8, 0x0284C7, 0x0369A1, 0x0C4A6E],
    "ember_orange": [0xFFF7ED, 0xFED7AA, 0xFDBA74, 0xEA580C, 0xC2410C, 0x9A3412, 0x7C2D12],
}

_DIVERGING_STOPS: dict[str, list[int]] = {
    "rdbu": [0x67001F, 0xD6604D, 0xFDDBC7, 0xF7F7F7, 0xD1E5F0, 0x4393C3, 0x053061],
    "blue_to_red": [0x1E3A8A, 0x60A5FA, 0xDBEAFE, 0xFAF7F2, 0xFDE68A, 0xDC2626, 0x7F1D1D],
    "cyan_to_amber": [0x155E75, 0x0891B2, 0x67E8F9, 0x111827, 0xFDE68A, 0xF59E0B, 0xB45309],
    "blue_to_violet": [0x0C4A6E, 0x38BDF8, 0xBAE6FD, 0xF8FAFC, 0xE9D5FF, 0xA78BFA, 0x6D28D9],
}


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(h: int) -> tuple[int, int, int]:
    """Convert a 24-bit integer to (r, g, b) tuple."""
    return ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)


def _lerp(a: int, b: int, t: float) -> int:
    """Linear interpolation between two 0-255 values."""
    return int(round(a + (b - a) * t))


def _interpolate_stops(stops: list[int], n: int) -> list[str]:
    """Interpolate n evenly-spaced colors from a list of hex stops."""
    if n <= 0:
        return []
    if n == 1:
        r, g, b = _hex_to_rgb(stops[len(stops) // 2])
        return [f"#{r:02x}{g:02x}{b:02x}"]

    rgbs = [_hex_to_rgb(s) for s in stops]
    result: list[str] = []
    for i in range(n):
        # Map i to a position t in [0, 1]
        t = i / (n - 1)
        # Map t to the stop space
        pos = t * (len(rgbs) - 1)
        idx = int(pos)
        if idx >= len(rgbs) - 1:
            r, g, b = rgbs[-1]
        else:
            frac = pos - idx
            r0, g0, b0 = rgbs[idx]
            r1, g1, b1 = rgbs[idx + 1]
            r = _lerp(r0, r1, frac)
            g = _lerp(g0, g1, frac)
            b = _lerp(b0, b1, frac)
        result.append(f"#{r:02x}{g:02x}{b:02x}")
    return result


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
        interpolates *n* evenly-spaced colors.

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
    ['#440154', '#21918c', '#fde725']
    """
    # Check categorical first
    if name in _CATEGORICAL:
        colors = _CATEGORICAL[name]
        if n is None:
            return list(colors)
        # Wrap cyclically
        return [colors[i % len(colors)] for i in range(n)]

    # Check sequential
    if name in _SEQUENTIAL_STOPS:
        return _interpolate_stops(_SEQUENTIAL_STOPS[name], n if n is not None else 7)

    # Check diverging
    if name in _DIVERGING_STOPS:
        return _interpolate_stops(_DIVERGING_STOPS[name], n if n is not None else 7)

    available = sorted(set(_CATEGORICAL) | set(_SEQUENTIAL_STOPS) | set(_DIVERGING_STOPS))
    raise ValueError(f"Unknown palette: {name!r}. Available: {', '.join(available)}")


def to_hex(color: tuple[float, ...] | str) -> str:
    """Convert a color to a hex string.

    Parameters
    ----------
    color : tuple or str
        An RGB tuple with values in [0, 1] (floats) or [0, 255] (ints),
        or a hex string (returned as-is after normalization).

    Returns
    -------
    str
        Hex string like ``"#1f77b4"``.

    Raises
    ------
    ValueError
        If the input format is not recognized.

    Examples
    --------
    >>> import ferrum
    >>> ferrum.color.to_hex((1.0, 0.0, 0.0))
    '#ff0000'
    >>> ferrum.color.to_hex((255, 0, 0))
    '#ff0000'
    """
    if isinstance(color, str):
        # Normalize: strip whitespace, ensure lowercase
        s = color.strip().lower()
        if not s.startswith("#"):
            raise ValueError(f"String colors must be hex format (#rrggbb), got: {color!r}")
        return s

    if not isinstance(color, (tuple, list)) or len(color) < 3:
        raise ValueError(f"Expected an RGB tuple (r, g, b) or hex string, got: {color!r}")

    r, g, b = color[0], color[1], color[2]

    # Detect whether values are in [0, 1] float range or [0, 255] int range
    if isinstance(r, float) or isinstance(g, float) or isinstance(b, float):
        # Float range [0, 1]
        ri = int(round(float(r) * 255))
        gi = int(round(float(g) * 255))
        bi = int(round(float(b) * 255))
    else:
        # Integer range [0, 255]
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
        Hex color strings.

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
    if name not in _SEQUENTIAL_STOPS:
        available = sorted(_SEQUENTIAL_STOPS)
        raise ValueError(f"Unknown sequential palette: {name!r}. Available: {', '.join(available)}")
    return _interpolate_stops(_SEQUENTIAL_STOPS[name], n)


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
        Hex color strings.

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
    if name not in _DIVERGING_STOPS:
        available = sorted(_DIVERGING_STOPS)
        raise ValueError(f"Unknown diverging palette: {name!r}. Available: {', '.join(available)}")
    return _interpolate_stops(_DIVERGING_STOPS[name], n)
