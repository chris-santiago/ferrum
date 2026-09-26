"""``ferrum._core.stroke_dash_palette()`` is the one dash-pattern table.

Rust facts reach Python through this thin binding rather than a hand-mirrored
Python literal (the ``scale_accepted_keys`` precedent): Python's
``_LINETYPE_MAP`` names must all resolve into the palette Rust actually
renders from, and the palette's legacy index positions must stay put so a
``StrokeDash``-encoded integer index keeps pointing at the same pattern.
"""

from __future__ import annotations

from ferrum._core import stroke_dash_palette
from ferrum.marks.base import _LINETYPE_MAP

# The four indices that predate the named linetypes: index 0 is solid, then
# the two two-segment patterns, then their concatenation.
_LEGACY_PALETTE_PREFIX: list[list[float]] = [
    [],
    [6.0, 3.0],
    [2.0, 3.0],
    [6.0, 3.0, 2.0, 3.0],
]


def test_every_linetype_value_is_a_palette_member() -> None:
    palette = stroke_dash_palette()
    for name, lengths in _LINETYPE_MAP.items():
        assert [float(x) for x in lengths] in palette, (
            f"_LINETYPE_MAP[{name!r}] = {lengths!r} is not a member of "
            f"stroke_dash_palette() = {palette!r}"
        )


def test_legacy_prefix_indices_are_unchanged() -> None:
    palette = stroke_dash_palette()
    assert palette[:4] == _LEGACY_PALETTE_PREFIX, (
        f"palette's first four entries {palette[:4]!r} must stay exactly "
        f"{_LEGACY_PALETTE_PREFIX!r} so a StrokeDash-encoded index still "
        "points at the same pattern"
    )


def test_nonpalette_pattern_is_absent() -> None:
    palette = stroke_dash_palette()
    assert [7.0, 2.0, 1.0, 2.0] not in palette
