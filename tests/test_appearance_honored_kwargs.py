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

**Further deliberate contract change (2026-09-01, batch-A T12 spec-review
addendum):** StrokeDash moves a second time, from ``APPEARANCE_FULL`` to
``APPEARANCE_SORT`` — mirroring Shape, ``sort=`` is now honored too. (At the
time this note was written, ``condition`` was still a hand-placed role-set
member and StrokeDash's move onto ``APPEARANCE_SORT`` looked like the reason
it also gained ``condition`` — "as a consequence of the role hierarchy". The
fifth change below replaced that mechanism entirely: ``condition`` is now
derived from the conditional vocabulary, independent of which role set a
channel declares, so the role hierarchy is no longer why StrokeDash honors
it.) The Rust-side ``build_stroke_dash_scale`` domain builder (T6,
``auxiliary.rs:394``) already read ``dash_enc.sort``; the Python channel was
silently dropping it before this change. StrokeOpacity is unaffected and
stays on ``APPEARANCE_FULL``.

**Third deliberate contract change (2026-09-11, batch-H close remediation):**
StrokeWidth, StrokeOpacity, FillOpacity and Angle gain ``condition``.
All four are conditional channels on the wire
(``ferrum.encoding._channel_policy._CONDITIONAL_CHANNELS``, mirrored from
Rust's ``EncodingValue`` variants) and their ``condition=`` translates and
renders end to end -- so the warn guard was telling users the kwarg was
ignored on a call that changed the render.
``tests/test_condition_translation.py`` pins the other half of this
invariant: a channel declaring ``condition`` honored must translate to a
channel the Rust enum accepts, and every other channel must refuse.

**Fourth deliberate contract change (2026-09-11, batch-H close re-review):**
``Stroke`` and ``Shape`` LOSE ``condition``. Both are refused at
``encode()`` because the wire cannot express their conditional: a stroke
conditional would repaint the fill (``EncodingValue::Color`` → ``fill_color``),
and a shape conditional has no value variant at all, so it serialized and
then applied as a no-op. ``Fill`` and ``StrokeDash`` — their role-set
partners — keep ``condition``, because both DO have a working wire value.

**Fifth change, structural not contractual (2026-09-11, batch-H close cycle
3):** every expected set below is unchanged from the fourth change's table —
this row is documented because HOW each set is produced changed, not any
value in it. ``condition`` is no longer hand-placed in a channel's role set
at all (the ``APPEARANCE_BASE_CONDITION``/``APPEARANCE_CONDITION``/
``APPEARANCE_SORT_NO_CONDITION``/``APPEARANCE_SCHEME_NO_CONDITION`` role
constants the third and fourth changes introduced are deleted).
``ChannelBase.__init_subclass__`` (``ferrum.encoding.base``) now derives the
bit per class from ``_channel_policy.channel_honors_condition(cls._channel_name)``,
so this test and the conditional vocabulary in
``ferrum.encoding._channel_policy`` cannot drift apart the way they did
across the third and fourth changes above.

**Sixth deliberate contract change (2026-09-23, #123):** ``legend`` and a
legend's ``title`` are derived from ``_legend_surface`` (``base._derive_honored``),
not declared by role sets. A channel with no legend of its own
(``_legend_surface == "none"``: Opacity, FillOpacity, StrokeOpacity,
StrokeWidth, Angle) honors neither: ``title=`` and a legend dict/``Legend(...)``
are construction-time ``ValueError``s, and ``legend=None``/``False`` is
accepted silently and puts nothing on the wire. ``scale`` stays honored on the
opacity family: it IS wired. No legend-owning channel's set changes.

**Seventh deliberate contract change (2026-10-15):** every channel whose mark
draws the aggregated value honors ``aggregate``: Color, Fill, Stroke, Size,
Opacity, FillOpacity, StrokeOpacity, StrokeWidth and Angle here (Text,
Tooltip and TooltipField in ``ferrum.encoding.text``). Any channel's
``aggregate=`` already lowered into an ``Aggregate`` step (one per flat chart,
one per layer) that re-points the channel at its output column, so the warn
guard was telling users the kwarg was ignored on a call that drew the
aggregated value. Shape and StrokeDash keep warning: an aggregated category
draws nothing a shape or dash palette means.
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
    Color: frozenset(
        {"type", "scale", "title", "legend", "condition", "sort", "scheme", "aggregate"}
    ),
    Size: frozenset({"type", "scale", "title", "legend", "condition", "aggregate"}),
    # Shape LOST "condition" 2026-09-11 (batch-H close re-review), the same
    # shape of change as Stroke's below and for the same class of reason: the
    # wire has a `shape` channel name but no EncodingValue::Shape, so a shape
    # conditional serialized as a Field value the runtime applies as a no-op --
    # accepted and inert. encode() refuses it, so this table must not advertise
    # it. StrokeDash keeps "condition" (it HAS an EncodingValue::StrokeDash).
    # As of cycle 3 (2026-09-11), Shape and StrokeDash both declare the SAME
    # role set (APPEARANCE_SORT) -- the derivation, not a second role set, is
    # what keeps Shape without the bit. Gains "condition" for real once Rust
    # grows EncodingValue::Shape (no table change needed here then either --
    # only the derivation's answer for "shape" flips).
    Shape: frozenset({"type", "scale", "title", "legend", "sort"}),
    # Opacity LOST "title" and "legend" 2026-09-23 (#123, sixth change
    # above): it draws no legend.
    Opacity: frozenset({"type", "scale", "condition", "aggregate"}),
    Fill: frozenset({"type", "scale", "title", "legend", "condition", "scheme", "aggregate"}),
    # Stroke LOST "condition" 2026-09-11 (batch-H close re-review): a stroke
    # conditional would repaint the fill (ferrum-wasm's apply_value_common
    # writes EncodingValue::Color to fill_color; FillStrokeView has no
    # stroke-colour member), so encode() refuses it and this table must not
    # advertise it. Fill keeps it -- Color *is* the fill. As of cycle 3
    # (2026-09-11), Fill and Stroke both declare the SAME role set
    # (APPEARANCE_SCHEME) -- see Shape's comment above for the pattern.
    Stroke: frozenset({"type", "scale", "title", "legend", "scheme", "aggregate"}),
    # FillOpacity/StrokeOpacity/StrokeWidth/Angle gained "condition"
    # 2026-09-11 (batch-H close remediation) — all four are conditional
    # channels on the wire; StrokeWidth/Angle still carry no "scale"/"title".
    # All four LOST "legend" (and FillOpacity/StrokeOpacity "title")
    # 2026-09-23 (#123, sixth change above), the same reason as Opacity's.
    FillOpacity: frozenset({"type", "scale", "condition", "aggregate"}),
    # StrokeOpacity gained "scale"/"title" 2026-08-28 (batch-A §4.3).
    StrokeOpacity: frozenset({"type", "scale", "condition", "aggregate"}),
    StrokeWidth: frozenset({"type", "condition", "aggregate"}),
    # Every channel above but Shape and StrokeDash gained "aggregate" 2026-10-15
    # (seventh change above).
    # StrokeDash gained "sort"/"condition" 2026-09-01 (batch-A T12 spec-review
    # addendum) — moved to APPEARANCE_SORT, mirroring Shape.
    StrokeDash: frozenset({"type", "scale", "title", "legend", "condition", "sort"}),
    Angle: frozenset({"type", "condition", "aggregate"}),
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
