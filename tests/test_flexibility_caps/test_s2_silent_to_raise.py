"""S2: unknown shape / offset values must raise clear errors, not silently produce
plausible-but-wrong output.

SITE 1: unknown constant shape in mark_point(shape=...).
  - A typo'd shape name silently became a circle. Now it must raise ValueError
    naming the bad value and listing the valid set.
  - Every valid shape name must still work exactly as before.

SITE 2: unknown offset in transform_stack(offset=...).
  - The DataStackSpec in Rust stored offset as a raw String, so an unknown
    value silently fell through the match arm to zero-offset behavior.
  - transform_stack() now validates offset at the Python boundary before the
    dict ever reaches Rust. Valid offsets still work.
  - The PyDataStack PyO3 constructor also validates offset.

SITE 2 REACHABILITY FINDING (documented here):
  - fm.Stack(offset="bogus") already raises ValueError (validated in
    position.py __init__ with _VALID_STACK_OFFSETS).
  - fm.transform_stack(offset="bogus") was NOT validated — the dict was
    forwarded to Rust verbatim. The Rust DataStackSpec stores offset as a
    plain String; the apply() match arm fell through to _ => {} (treat as
    zero). This is reachable from the public API.
  - The fix adds validation in transform_stack() at the Python boundary
    (same layer where Stack.offset is validated), plus changes the dead
    Rust _ => {} arm to an explicit unreachable! comment (it remains dead
    because all Python paths now validate before reaching Rust).
"""

from __future__ import annotations

import pytest
import polars as pl
import ferrum as fm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DF = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
_DF_STACK = pl.DataFrame(
    {
        "cat": ["a", "a", "b", "b"],
        "grp": ["x", "y", "x", "y"],
        "val": [1.0, 2.0, 3.0, 4.0],
    }
)

_VALID_SHAPES = [
    "circle",
    "square",
    "cross",
    "diamond",
    "triangle-up",
    "triangle_up",
    "triangle-down",
    "triangle_down",
    "|",
    "vline",
    "-",
    "hline",
]


# ---------------------------------------------------------------------------
# SITE 1: shape — RED tests (these must pass AFTER the fix)
# ---------------------------------------------------------------------------


def test_mark_point_bogus_shape_raises():
    """mark_point(shape='not_a_real_shape') must raise a clear error.

    This was a silent-failure: the bogus shape became a circle with no error.
    """
    with pytest.raises((ValueError, TypeError)) as exc_info:
        fm.Chart(_DF).mark_point(shape="not_a_real_shape").encode(x="x:Q", y="y:Q").show_svg()

    msg = str(exc_info.value).lower()
    assert "not_a_real_shape" in msg, (
        f"Error message must name the bad value; got: {exc_info.value!r}"
    )


def test_mark_point_typo_shape_raises_early():
    """A typo in shape= must be caught at mark_point() call time, not at render."""
    with pytest.raises((ValueError, TypeError)) as exc_info:
        fm.Chart(_DF).mark_point(shape="dimond")  # typo: "dimond" not "diamond"

    msg = str(exc_info.value).lower()
    assert "dimond" in msg, (
        f"Error message must name the bad value 'dimond'; got: {exc_info.value!r}"
    )


def test_mark_point_bogus_shape_message_lists_valid_shapes():
    """The error message for a bad shape must list valid shape names."""
    with pytest.raises((ValueError, TypeError)) as exc_info:
        fm.Chart(_DF).mark_point(shape="hexagon")

    msg = str(exc_info.value)
    # At minimum, some well-known valid shapes should appear in the message.
    listed = any(s in msg for s in ("circle", "square", "diamond", "cross"))
    assert listed, f"Error message must list valid shape names; got: {exc_info.value!r}"


# ---------------------------------------------------------------------------
# SITE 1: shape — guard tests (valid shapes must still work)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_name", _VALID_SHAPES)
def test_mark_point_valid_shape_does_not_raise(shape_name):
    """Every documented valid shape name must be accepted without error."""
    # mark_point(shape=...) construction must not raise.
    chart = fm.Chart(_DF).mark_point(shape=shape_name).encode(x="x:Q", y="y:Q")
    # Rendering must also succeed.
    svg = chart.show_svg()
    assert svg, f"Expected non-empty SVG for shape={shape_name!r}"


def test_mark_circle_shorthand_still_works():
    """mark_circle() (shape='circle' shorthand) must still render without error."""
    svg = fm.Chart(_DF).mark_circle().encode(x="x:Q", y="y:Q").show_svg()
    assert svg, "mark_circle() must produce non-empty SVG"


def test_mark_square_shorthand_still_works():
    """mark_square() (shape='square' shorthand) must still render without error."""
    svg = fm.Chart(_DF).mark_square().encode(x="x:Q", y="y:Q").show_svg()
    assert svg, "mark_square() must produce non-empty SVG"


# ---------------------------------------------------------------------------
# SITE 2: stack offset — reachability + fix tests
# ---------------------------------------------------------------------------


def test_stack_position_bogus_offset_already_raises():
    """fm.Stack(offset='bogus') already raises ValueError — guard against regression."""
    with pytest.raises(ValueError) as exc_info:
        fm.Stack(offset="bogus_offset")
    assert "bogus_offset" in str(exc_info.value) or "offset" in str(exc_info.value).lower()


def test_transform_stack_bogus_offset_raises():
    """fm.transform_stack(offset='bogus') must raise a clear error.

    Previously this was silently treated as zero offset because the dict was
    forwarded to Rust verbatim and the match arm fell through to _ => {}.
    """
    with pytest.raises((ValueError, TypeError)) as exc_info:
        fm.transform_stack("val", groupby=["cat"], offset="not_a_real_offset")

    msg = str(exc_info.value).lower()
    assert "not_a_real_offset" in msg or "offset" in msg, (
        f"Error message must reference the bad offset; got: {exc_info.value!r}"
    )


def test_transform_stack_bogus_offset_message_lists_valid():
    """The transform_stack error for a bad offset must list valid offset values."""
    with pytest.raises((ValueError, TypeError)) as exc_info:
        fm.transform_stack("val", groupby=["cat"], offset="streamgraph")

    msg = str(exc_info.value)
    listed = any(s in msg for s in ("zero", "normalize", "center"))
    assert listed, f"Error must list valid offsets; got: {exc_info.value!r}"


def test_transform_stack_offset_zero_still_works():
    """transform_stack(offset='zero') must produce correct stacked output."""
    chart = (
        fm.Chart(_DF_STACK)
        .mark_bar()
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .transform(fm.transform_stack("val", groupby=["cat"], offset="zero"))
    )
    svg = chart.show_svg()
    assert svg, "offset='zero' must produce non-empty SVG"


def test_transform_stack_offset_normalize_still_works():
    """transform_stack(offset='normalize') must produce correct stacked output."""
    chart = (
        fm.Chart(_DF_STACK)
        .mark_bar()
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .transform(fm.transform_stack("val", groupby=["cat"], offset="normalize"))
    )
    svg = chart.show_svg()
    assert svg, "offset='normalize' must produce non-empty SVG"


def test_transform_stack_offset_center_still_works():
    """transform_stack(offset='center') must produce correct stacked output."""
    chart = (
        fm.Chart(_DF_STACK)
        .mark_bar()
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .transform(fm.transform_stack("val", groupby=["cat"], offset="center"))
    )
    svg = chart.show_svg()
    assert svg, "offset='center' must produce non-empty SVG"
