"""Regression tests for finding R3 (2026-08-27 design-review remediation).

Spec §9.3: under `CoordFlip`, user-facing render error/warning text must name
the channel the USER wrote in `.encode(...)`, not the RESOLVED (post-flip)
internal slot. Before the fix, a flipped `mark_area(y2=...)` said `channel
'x2' is not supported; use y2=...` -- naming the wrong channel twice, since
the resolved slot (`x2`) is what validation acted on, not what the user typed
(`y2`). The fix is `render::prepare::user_facing_channel`, applied at
`Display`-time only to the *user-encoding-channel* message families
(`EncodingTypeMismatch`, `UnsupportedChannelCombination`,
`InvalidAxisOrient`'s two per-channel `Axis(orient=...)` call chains) -- never
to physical-slot or user-literal vocabularies (see `RenderError::
InvalidAxisOrient`'s field doc for the chart-level `configure_axis` exemption).

These tests are Python-side proof the Rust fix (reviewed, already built into
the extension this session) actually changes the strings ferrum raises. The
RED pre-fix behavior was captured once via a `git stash` of `crates/` +
`maturin develop` rebuild + rerun (see the companion report); it is not
re-derived here since these tests must pass against the fixed extension.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# 1. mark_area + y2= under CoordFlip: flagged channel names 'y2' (what the
#    user wrote), and the hint names 'x2' (what the user should write instead
#    to land the field in the resolved-y2 slot mark_area actually supports).
# ---------------------------------------------------------------------------


def test_flipped_mark_area_y2_names_user_written_channel():
    """Flipped mark_area(y2=...) must say the flagged channel is 'y2' (what
    the user wrote), not 'x2' (the resolved post-flip slot the Rust
    validation actually inspected)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "y2": [0.0, 0.0, 0.0]})
    chart = fm.Chart(df).mark_area().encode(x="x", y="y", y2="y2").coord(fm.CoordFlip())

    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()

    assert str(exc_info.value) == (
        "mark_area: channel 'y2' is not supported; "
        "use x2= for a vertical band area, or use mark_rect for a 2-D extent"
    )


# ---------------------------------------------------------------------------
# 2. Flipped missing-y (EncodingTypeMismatch family): a chart that only binds
#    x= must be reported as missing 'y' -- the channel the user actually left
#    unbound -- not 'x' (the resolved slot the post-flip swap left empty).
# ---------------------------------------------------------------------------


def test_flipped_missing_y_encoding_names_user_written_channel():
    """A flipped mark_point with only x= bound must report the missing
    channel as 'y' (what the user never wrote), not 'x' (what the user DID
    write, which the post-flip swap moved into the resolved y slot)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_point().encode(x="x").coord(fm.CoordFlip())

    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()

    assert str(exc_info.value) == "encoding 'y' expected EncodingSpec, got None"


# ---------------------------------------------------------------------------
# 3. Unflipped control: byte-identical to pre-fix text. `user_facing_channel`
#    is identity when `!coord_flipped`, so these two error families must be
#    completely unaffected by the R3 fix on an unflipped chart.
# ---------------------------------------------------------------------------


def test_unflipped_mark_area_x2_message_unaffected_by_r3_fix():
    """The unflipped mirror of test 1: mark_area(x2=...) without CoordFlip
    must still name the literal channel the user wrote ('x2') -- R3 only
    changes flipped-chart text."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "x2": [0.0, 0.0, 0.0]})
    chart = fm.Chart(df).mark_area().encode(x="x", y="y", x2="x2")

    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()

    assert str(exc_info.value) == (
        "mark_area: channel 'x2' is not supported; "
        "use y2= for a vertical band area, or use mark_rect for a 2-D extent"
    )


def test_unflipped_missing_y_encoding_message_unaffected_by_r3_fix():
    """The unflipped mirror of test 2: an unflipped mark_point with only x=
    bound must still report the missing channel as 'y' -- the literal
    channel absent from `.encode(...)`."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_point().encode(x="x")

    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()

    assert str(exc_info.value) == "encoding 'y' expected EncodingSpec, got None"


# ---------------------------------------------------------------------------
# 4. A second message family reachable from Python: per-channel
#    fm.Axis(orient=...) validation (InvalidAxisOrient), flipped vs.
#    unflipped pair.
# ---------------------------------------------------------------------------


def test_unflipped_axis_orient_names_literal_channel():
    """Unflipped: an invalid orient on the x channel's own Axis(...) must
    name the x axis -- the literal channel the user configured it on."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_point().encode(x=fm.X("x", axis=fm.Axis(orient="left")), y="y")

    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()

    assert str(exc_info.value) == (
        "axis orient 'left' is invalid for the x axis (expected 'top' or 'bottom')"
    )


def test_flipped_axis_orient_names_user_written_channel():
    """Flipped: the SAME invalid orient value, configured via the user's own
    y-channel Axis(...), must name the y axis -- what the user wrote -- even
    though CoordFlip's data-swap moves that Axis config into the resolved x
    slot (which is what the top/bottom-vs-left/right constraint actually
    validates against, per InvalidAxisOrient's physical-slot doc)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y=fm.Y("y", axis=fm.Axis(orient="left")))
        .coord(fm.CoordFlip())
    )

    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()

    assert str(exc_info.value) == (
        "axis orient 'left' is invalid for the y axis "
        "(expected 'top' or 'bottom' — under CoordFlip, y renders as the horizontal axis)"
    )
