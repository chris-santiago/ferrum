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

import numbers
import warnings
from typing import Literal

from ferrum._core import (
    list_palettes as _list_palettes,
    palette_colors as _palette_colors,
    palette_kind as _palette_kind,
    palette_sample as _palette_sample,
    parse_color_to_hex as _parse_color_to_hex,
)
from ferrum._validate import is_none_color_sentinel

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
        An RGB tuple with values in [0, 1] (unit) or [0, 255] (byte) — the
        first 3 components are R/G/B; an optional 4th component is alpha,
        in the same scale as R/G/B, and is honored into the hex output (see
        Returns) rather than silently dropped, EXCEPT when it is fully
        opaque -- a fully-opaque alpha normalizes away on every input form
        (matching the historical, still-current string behavior), so
        ``to_hex((1.0, 0.0, 0.0, 1.0))`` and ``to_hex((1.0, 0.0, 0.0))``
        both return ``"#ff0000"`` — or a color string: a CSS
        named color (e.g. ``"steelblue"``), hex
        (``#rgb``/``#rgba``/``#rrggbb``/``#rrggbbaa``), or a functional
        ``rgb()``/``rgba()`` form. String input is validated and normalized
        by ferrum's single Rust color parser (`parse_color`), so this
        function accepts the exact vocabulary every other ferrum color
        boundary accepts, alpha included.
    scale : {"unit", "byte"}, optional
        Explicit interpretation of an RGB tuple's component range.  ``"unit"``
        treats components as floats in ``[0, 1]``; ``"byte"`` treats them as
        integers in ``[0, 255]``.  When ``None`` (default) the range is
        inferred: any component greater than 1 forces byte interpretation;
        otherwise unit interpretation is used.  This makes integer-valued
        floats (``1.0``) and integers (``1``) behave identically.  Ignored
        for string input.

    Returns
    -------
    str
        Normalized hex string: ``"#rrggbb"`` when there is no alpha
        component, or the alpha is fully opaque, and ``"#rrggbbaa"`` when
        there is a non-opaque alpha. This is one rule shared by both input
        forms with no per-form drift (#122 residual, close-remediation
        cycle 2): a *string* carrying alpha (``rgba()``, 4/8-digit hex)
        keeps it in the output only when it is not fully opaque, and a
        *tuple*'s 4th component follows the identical rule — so
        ``to_hex((1.0, 0.0, 0.0, 1.0))``, ``to_hex("#ff0000ff")``,
        ``to_hex("#f00f")``, and ``to_hex("rgba(255,0,0,1)")`` all return
        ``"#ff0000"``, and ``to_hex`` is idempotent over its own output
        (``to_hex(to_hex(x)) == to_hex(x)``) for every opaque color.

    Raises
    ------
    ValueError
        If a string input is not a recognized CSS color name, hex literal, or
        ``rgb()``/``rgba()`` form (the message names the accepted forms); if
        the string is a paint-clearing sentinel (``"none"``/``"transparent"``,
        trimmed and case-insensitive) — those spellings clear paint at the
        mark/selection boundaries (``MarkBase.__init__``, ``selection.py``)
        and have no hex form here, so the message says that instead of the
        generic accepted-forms text, which would be self-contradicting for
        ``"transparent"`` (a real CSS Color 4 keyword); if a tuple input's
        shape is not recognized, or any of its read components is not
        numeric (e.g. ``("a", "b", "c")`` — previously an unexplained
        ``TypeError`` escaped from the internal range comparison; it is now
        this typed ``ValueError``); or if *scale* is not one of ``"unit"``,
        ``"byte"``, or ``None``.

    Examples
    --------
    >>> import ferrum
    >>> ferrum.color.to_hex((1.0, 0.0, 0.0))
    '#ff0000'
    >>> ferrum.color.to_hex((255, 0, 0))
    '#ff0000'
    >>> ferrum.color.to_hex((128, 128, 128), scale="byte")
    '#808080'
    >>> ferrum.color.to_hex((1.0, 0.0, 0.0, 0.5))
    '#ff000080'
    >>> ferrum.color.to_hex("steelblue")
    '#4682b4'
    >>> ferrum.color.to_hex("rgb(70, 130, 180)")
    '#4682b4'
    """
    if isinstance(color, str):
        if is_none_color_sentinel(color):
            raise ValueError(
                f"to_hex({color!r}): {color!r} clears paint at mark/selection "
                "boundaries (fill=/stroke=/color=), not a color value — it has "
                "no hex form here"
            )
        return _parse_color_to_hex(color)

    if not isinstance(color, (tuple, list)) or len(color) < 3:
        raise ValueError(f"Expected an RGB tuple (r, g, b) or hex string, got: {color!r}")

    if scale not in (None, "unit", "byte"):
        raise ValueError(f"scale must be 'unit', 'byte', or None, got: {scale!r}")

    r, g, b = color[0], color[1], color[2]
    # A 4th component is alpha; a 5th+ is read past, unchanged from before.
    alpha = color[3] if len(color) >= 4 else None

    # Every component actually read must be numeric. Previously a
    # non-numeric component (e.g. `("a", "b", "c")`) escaped as a bare
    # `TypeError` from the `max(r, g, b) > 1` comparison below — an
    # unexplained internal error at any caller that only expected
    # `ValueError` (see `plots/_helpers.py`'s `_column_or_literal_color`).
    # `bool` is intentionally NOT excluded here (unlike e.g.
    # `_validate_domain_bounds`): a `(True, False, True)` tuple already
    # rendered `"#ff00ff"` before this change, and closing only the
    # genuinely-non-numeric escape must not also start refusing that.
    for component in (r, g, b, alpha):
        if component is not None and not isinstance(component, numbers.Real):
            raise ValueError(
                f"Expected numeric RGB(A) components, got: {color!r} "
                f"({component!r} is not a number)"
            )

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
        ai = int(round(float(alpha) * 255)) if alpha is not None else None
    else:  # "byte"
        ri, gi, bi = int(r), int(g), int(b)
        ai = int(alpha) if alpha is not None else None

    ri = max(0, min(255, ri))
    gi = max(0, min(255, gi))
    bi = max(0, min(255, bi))
    if ai is None:
        return f"#{ri:02x}{gi:02x}{bi:02x}"
    ai = max(0, min(255, ai))
    if ai == 255:
        # Fully-opaque alpha normalizes away on EVERY input form (#122
        # residual, close-remediation cycle 2): the string path already
        # elides it here (`parse_color_to_hex`'s Rust parser normalizes
        # "rgba(255,0,0,1)"/"#ff0000ff"/"#f00f" to "#ff0000"), so the tuple
        # path must match rather than keep `ff` just because a 4th
        # component was structurally present. This is what makes "one
        # keep-alpha rule shared by both input forms" (see Returns) true,
        # and restores idempotence: to_hex(to_hex(x)) == to_hex(x) for
        # every opaque color, tuple or string. Non-opaque alpha is
        # unaffected -- it is kept on both forms exactly as before.
        return f"#{ri:02x}{gi:02x}{bi:02x}"
    return f"#{ri:02x}{gi:02x}{bi:02x}{ai:02x}"


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
