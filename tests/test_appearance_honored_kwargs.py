"""Guard: each appearance channel's _honored_kwargs equals its expected explicit set.

This test exists so that any accidental membership change (kwarg added or
removed from a channel's contract) fails loudly rather than silently.  The
expected sets below are the canonical contract; they must match exactly what
each channel declared before the C5 declaration de-duplication refactor.

**Deliberate contract change (2026-08-28, batch-A appearance-resolution
spec §4.3):** StrokeOpacity and StrokeDash move from ``APPEARANCE_BASE`` to
``APPEARANCE_FULL`` — ``scale=`` and ``title=`` are now honored (serialized
to the wire) instead of warn-and-drop. This opens the wire gate for the
Rust-side ``OpacityScale``/``StrokeDashScale`` builders (Task 6, concurrent).
StrokeWidth and Angle are unaffected: both stay on ``APPEARANCE_BASE``
because their scale resolution is out of scope by design (per-row constants,
``spec/encoding.rs:746``).
"""

from __future__ import annotations

import pytest

from ferrum.encoding.appearance import (
    Angle,
    Color,
    Fill,
    FillOpacity,
    Opacity,
    Shape,
    Size,
    Stroke,
    StrokeDash,
    StrokeOpacity,
    StrokeWidth,
)


# Canonical expected sets — these are the authoritative contract.
# When a channel's membership intentionally changes, update this table and
# add a dated comment explaining why.
_EXPECTED: dict[type, frozenset[str]] = {
    Color: frozenset({"type", "scale", "title", "legend", "condition", "sort", "scheme"}),
    Size: frozenset({"type", "scale", "title", "legend", "condition"}),
    Shape: frozenset({"type", "scale", "title", "legend", "condition", "sort"}),
    Opacity: frozenset({"type", "scale", "title", "legend", "condition"}),
    Fill: frozenset({"type", "scale", "title", "legend", "condition", "scheme"}),
    Stroke: frozenset({"type", "scale", "title", "legend", "condition", "scheme"}),
    FillOpacity: frozenset({"type", "scale", "title", "legend"}),
    # StrokeOpacity/StrokeDash gained "scale"/"title" 2026-08-28 (batch-A §4.3).
    StrokeOpacity: frozenset({"type", "scale", "title", "legend"}),
    StrokeWidth: frozenset({"type", "legend"}),
    StrokeDash: frozenset({"type", "scale", "title", "legend"}),
    Angle: frozenset({"type", "legend"}),
}


@pytest.mark.parametrize(
    "channel_cls,expected",
    list(_EXPECTED.items()),
    ids=[cls.__name__ for cls in _EXPECTED],
)
def test_appearance_channel_honored_kwargs(channel_cls: type, expected: frozenset[str]) -> None:
    """Each appearance channel's _honored_kwargs must equal the expected set exactly."""
    actual = channel_cls._honored_kwargs
    assert actual == expected, (
        f"{channel_cls.__name__}._honored_kwargs mismatch.\n"
        f"  Missing from actual : {expected - actual}\n"
        f"  Extra in actual     : {actual - expected}"
    )
