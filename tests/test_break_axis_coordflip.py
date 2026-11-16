"""Characterization tests: BreakAxis + CoordFlip axis token semantics (#231).

BreakAxis.axis names the CALLER's channel: on a flipped chart, axis="y" breaks
the column the caller bound to y, wherever the flip draws it (horizontally).
This test pins that behavior and asserts chrome ownership consistency: the
broken axis's ticks/labels remap, the cross axis's don't. ``tests/
test_break_axis_caller_channel.py`` pins the same reading against the
transposed chart, on every door.
"""

from __future__ import annotations

import warnings

import polars as pl

import ferrum as fm
from _svg_extents import axis_tick_dash_label_pairs, tick_along
from _svg_rects import fill_rects
from test_null_group_warning import _render_capturing


_ORDINAL = (
    "BreakAxis on the {axis} axis had no effect: the axis is ordinal, so its gaps have no "
    "numeric data range to remove"
)


def _rects(svg: str, fill: str = "#2563eb") -> list[tuple[float, float, float, float]]:
    """Extract (x, y, w, h) for every mark rect painted in fill.

    Binds to `tests/_svg_rects.py`'s `fill_rects` (the home for this
    fill-filtered rect extraction, since it was identically duplicated across
    this module, `test_break_axis_geometry.py` and `test_break_axis_facet.py`).
    """
    return fill_rects(svg, fill)


def test_break_x_axis_on_flipped_chart_warns_on_the_callers_categorical_x():
    """BreakAxis(axis="x") on a flipped chart names the caller's x channel.

    On a chart with x="category:N", y="value:Q" the caller's x is the ordinal
    category channel (drawn vertically after the flip). Breaks only work on
    quantitative axes, so a break on it warns, naming ``x`` (the caller's
    letter), with no effect. This pins that the axis parameter is the caller's
    channel, not the drawn side.
    """
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 100.0, 50.0, 500.0, 75.0],
        }
    )

    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")

    # Flipped chart with a break on the caller's x channel (category:N).
    # Breaks don't apply to ordinal axes, so this should warn.
    flipped_with_break = base_chart.coord(fm.CoordFlip()) + fm.BreakAxis(axis="x", gap=(2.5, 3.5))

    # Capture the warning that the break has no effect on the ordinal axis.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg_flipped = flipped_with_break.to_svg()

    # Verify a warning was issued about the ordinal axis.
    messages = [str(w.message) for w in caught]
    assert messages == [_ORDINAL.format(axis="x")], messages

    # The SVG should still render without error, just without the break effect.
    assert "<svg" in svg_flipped
    rects = _rects(svg_flipped)
    assert rects, "flipped chart must still render bars"


def test_break_y_axis_on_flipped_chart_clips_the_value_bar_crossing_the_gap():
    """BreakAxis(axis="y") on a flipped chart clips marks on the caller's y (value) channel.

    Mirrors ``test_break_axis_geometry.py``'s vertical clipping pin, rotated: on
    a flipped chart the bars are horizontal and vary along the value encoding.
    A bar whose value crosses the gap must render as two clipped pieces sharing
    the same row (y), not as one continuous (unclipped) rect. An SVG coming
    back, or some rect having y > 0, would also hold if the break were refused
    as ineffective, so the pieces are what is asserted.
    """
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 100.0, 50.0, 500.0, 75.0],
        }
    )

    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")

    # Flip the chart, then break the caller's y channel (value, drawn horizontally).
    # The value=500 bar crosses the gap (150, 400) and must split into two
    # clipped horizontal pieces sharing the same row.
    flipped_with_break = base_chart.coord(fm.CoordFlip()) + fm.BreakAxis(
        axis="y", gap=(150.0, 400.0)
    )
    svg_flipped = flipped_with_break.to_svg()

    rects = _rects(svg_flipped)
    assert len(rects) == 6, f"four unclipped bars + the crossing bar's two pieces; got {rects}"

    by_y: dict[float, list[tuple[float, float, float, float]]] = {}
    for r in rects:
        by_y.setdefault(round(r[1], 3), []).append(r)
    split = [group for group in by_y.values() if len(group) == 2]
    assert len(split) == 1, f"exactly one row splits; got {by_y}"
    left, right = sorted(split[0], key=lambda r: r[0])
    assert left[3] == right[3], "both pieces keep the bar's height"
    assert left[2] > 0.5 and right[2] > 0.5, "neither piece is a zero-width artifact"
    assert left[0] + left[2] < right[0], "the pieces are separated by the break"


def test_break_x_axis_effective_on_a_transposed_flip_while_y_warns_ordinal():
    """The positive case: the quantitative channel is the caller's x.

    Every other test here fixes x="category:N", y="value:Q". This is the
    mirror: with the encodings swapped before the flip (x="value:Q",
    y="category:N"), the caller's x is the quantitative value channel, so
    BreakAxis(axis="x") lands there effectively, with no warning, while
    BreakAxis(axis="y") on the SAME flipped chart warns that the caller's y
    (the ordinal category) had no effect.
    """
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 100.0, 50.0, 500.0, 75.0],
        }
    )

    base_chart = fm.Chart(df).mark_bar().encode(x="value:Q", y="category:N")
    flipped = base_chart.coord(fm.CoordFlip())
    gap = (150.0, 400.0)

    svg_plain = flipped.to_svg()

    svg_break_x, messages_x = _render_capturing((flipped + fm.BreakAxis(axis="x", gap=gap)).to_svg)
    assert messages_x == [], (
        f"axis='x' must break the caller's x (value) channel effectively with no "
        f"warning; got {messages_x}"
    )
    assert svg_break_x != svg_plain, "an effective break must change the render"

    y_ticks = axis_tick_dash_label_pairs(svg_break_x, axis="y")
    hidden = [t for t in y_ticks if tick_along(t.dash, axis="y") < -1000]
    visible = [t for t in y_ticks if tick_along(t.dash, axis="y") > 0]
    # The native horizontal bar anchors its value axis at zero, so the visible
    # ticks run from 0 up to the gap and on past it.
    assert len(hidden) == 5 and len(visible) == 6, (
        f"expected 5 hidden + 6 visible physical-y tick dashes under the gap; "
        f"got hidden={len(hidden)} visible={len(visible)} total={len(y_ticks)}"
    )
    assert any(label[2] == "0" for t in visible for label in t.labels), visible

    svg_break_y, messages_y = _render_capturing((flipped + fm.BreakAxis(axis="y", gap=gap)).to_svg)
    assert messages_y == [_ORDINAL.format(axis="y")], (
        f"axis='y' on this flipped chart (caller's y = ordinal category) must "
        f"warn that the break had no effect; got {messages_y}"
    )
    assert svg_break_y == svg_plain, (
        "the ordinal no-op must render byte-identical to the plain chart"
    )


def test_break_x_axis_is_a_no_op_while_break_y_axis_clips_on_the_same_flipped_chart():
    """The caller's-channel reading predicts an ASYMMETRIC result on one fixture.

    On the x="category:N", y="value:Q" fixture flipped, axis="x" (the caller's
    ordinal category) must be a byte-identical no-op while axis="y" (the
    caller's quantitative value) must clip. The drawn-side reading predicts
    the OPPOSITE asymmetry, so pinning WHICH SIDE is effective, not merely
    that the two renders differ, is what discriminates the two readings.
    """
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 100.0, 50.0, 500.0, 75.0],
        }
    )

    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")
    flipped = base_chart.coord(fm.CoordFlip())
    gap = (150.0, 400.0)

    svg_plain = flipped.to_svg()
    svg_break_x, messages_x = _render_capturing((flipped + fm.BreakAxis(axis="x", gap=gap)).to_svg)
    svg_break_y, messages_y = _render_capturing((flipped + fm.BreakAxis(axis="y", gap=gap)).to_svg)

    assert messages_x == [_ORDINAL.format(axis="x")]
    assert messages_y == []

    assert svg_break_x == svg_plain, (
        "axis='x' (the caller's ordinal category) on the flipped chart must be a "
        "byte-identical no-op"
    )
    assert len(_rects(svg_break_y)) == 6, (
        f"axis='y' on the same flipped chart must clip the value-crossing "
        f"bar; got {_rects(svg_break_y)}"
    )


def test_chrome_ownership_break_y_affects_value_axis_on_flipped():
    """On a flipped chart, BreakAxis(axis="y") breaks the caller's y (value axis).

    Chrome ownership consistency: the BROKEN axis's ticks/labels remap
    (move/hide per the break), while the cross axis's ticks/labels stay in
    their original positions.

    On a non-flipped chart with x="category:N", y="value:Q":
    - BreakAxis(axis="y", gap) breaks the value axis, drawn vertically

    On a flipped version of the same chart:
    - BreakAxis(axis="y", gap) breaks the value axis, drawn horizontally

    These two should produce similar results since they break the same
    encoding dimension (value), just on different drawn sides due to flip.
    """
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 100.0, 50.0, 200.0, 75.0],
        }
    )

    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")

    # Non-flipped baseline: break on y (value axis).
    svg_unflipped_break_y = (base_chart + fm.BreakAxis(axis="y", gap=(60.0, 150.0))).to_svg()

    # Flipped chart: break on the caller's y (value axis, drawn on the x side).
    svg_flipped_break_y = (
        base_chart.coord(fm.CoordFlip()) + fm.BreakAxis(axis="y", gap=(60.0, 150.0))
    ).to_svg()

    # Both should have ticks remapped on the broken axis.
    unflipped_y_ticks = axis_tick_dash_label_pairs(svg_unflipped_break_y, axis="y")
    flipped_x_ticks = axis_tick_dash_label_pairs(svg_flipped_break_y, axis="x")

    # The broken axis should have some ticks hidden (due to the gap).
    # We expect at least 1 tick to be hidden (y < -1000) on the unflipped y-axis.
    unflipped_hidden_y = [t for t in unflipped_y_ticks if tick_along(t.dash, axis="y") < -1000]
    assert unflipped_hidden_y, (
        "BreakAxis(axis='y', gap=(60, 150)) on the value axis should hide some ticks"
    )

    # Similarly, the flipped chart's x-axis should have some ticks hidden.
    flipped_hidden_x = [t for t in flipped_x_ticks if tick_along(t.dash, axis="x") < -1000]
    assert flipped_hidden_x, (
        "BreakAxis(axis='y', gap=(60, 150)) on a flipped chart (caller's y = value) "
        "should hide some ticks"
    )


def test_chrome_ownership_consistency_break_y_on_flipped():
    """On a flipped chart with BreakAxis(axis="y"), the drawn x-axis chrome remaps.

    With BreakAxis(axis="y") on a flipped chart, the caller's y (drawn on the
    x side) is broken.
    So x-axis ticks/labels should remap (some hidden, others compressed), while
    y-axis ticks/labels should stay untouched.
    """
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 100.0, 50.0, 200.0, 75.0],
        }
    )

    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")

    # Unbroken flipped chart: baseline for comparison.
    svg_unbroken = base_chart.coord(fm.CoordFlip()).to_svg()

    # Flipped chart with a break on the caller's y (value dimension).
    svg_broken = (
        base_chart.coord(fm.CoordFlip()) + fm.BreakAxis(axis="y", gap=(80.0, 180.0))
    ).to_svg()

    # Extract x-axis ticks and labels from both SVGs.
    # The x-axis is the bottom row (largest y).
    unbroken_x_ticks = axis_tick_dash_label_pairs(svg_unbroken, axis="x")
    broken_x_ticks = axis_tick_dash_label_pairs(svg_broken, axis="x")

    # With a break on the caller's y (drawn on x), some ticks/labels should be hidden
    # while others are compressed. The set should differ.
    unbroken_visible_x = [
        (tick_along(t.dash, axis="x"), t.labels)
        for t in unbroken_x_ticks
        if tick_along(t.dash, axis="x") > 0
    ]
    broken_visible_x = [
        (tick_along(t.dash, axis="x"), t.labels)
        for t in broken_x_ticks
        if tick_along(t.dash, axis="x") > 0
    ]

    # The break should affect the drawn x-axis ticks.
    # At least some x-ticks should be hidden or have different positions.
    assert len(broken_visible_x) != len(unbroken_visible_x) or any(
        tick_along(t.dash, axis="x") != tick_along(u.dash, axis="x")
        for t, u in zip(broken_x_ticks, unbroken_x_ticks)
        if tick_along(t.dash, axis="x") > 0 and tick_along(u.dash, axis="x") > 0
    ), "the x-axis (which carries the broken channel) should differ in tick positions or visibility"


def test_indicator_on_broken_axis_on_flipped_chart():
    """Break indicator is drawn on the broken axis of a flipped chart.

    Mirrors ``test_break_axis_geometry.py``'s break-style parity check: with a
    bar crossing the gap, ``break_style="slash"`` must render strictly MORE
    ``<line>`` elements than ``break_style="gap"`` (which draws no indicator at
    all) on the SAME flipped chart, axis="y" break -- proving an indicator is
    actually drawn on the broken axis, not merely that some SVG came back
    (``"<svg" in svg`` and ``len(svg) > 100`` both hold on an SVG with no break
    indicator at all).
    """
    df = pl.DataFrame({"category": ["A", "B"], "value": [72.0, 900.0]})
    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")
    flipped = base_chart.coord(fm.CoordFlip())
    gap = (150.0, 850.0)

    svg_no_indicator = (flipped + fm.BreakAxis(axis="y", gap=gap, break_style="gap")).to_svg()
    svg_slash = (flipped + fm.BreakAxis(axis="y", gap=gap, break_style="slash")).to_svg()

    assert len(_rects(svg_slash)) == 3, "one short bar + the crossing bar's two clipped pieces"
    assert svg_slash.count("<line") > svg_no_indicator.count("<line"), (
        "break_style='slash' must draw indicator lines the 'gap' style omits, "
        "on the flipped chart's value axis"
    )


def test_flipped_chart_without_break_same_as_break_on_ordinal_callers_x():
    """A flipped chart with BreakAxis on the caller's ordinal x is unchanged.

    Sanity check: applying a break to the caller's ordinal x (the
    categories, drawn vertically) has no effect and produces a warning, so the SVG
    should be identical to the unbroken case.
    """
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 100.0, 50.0, 200.0, 75.0],
        }
    )

    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")
    flipped = base_chart.coord(fm.CoordFlip())

    svg_no_break = flipped.to_svg()

    svg_with_break, messages = _render_capturing(
        (flipped + fm.BreakAxis(axis="x", gap=(2.0, 3.5))).to_svg
    )

    assert messages == [_ORDINAL.format(axis="x")]

    # Since the break has no effect on an ordinal axis, the SVGs should be identical.
    assert svg_no_break == svg_with_break, (
        "BreakAxis(axis='x') on a flipped chart with an ordinal caller's x "
        "should have no effect and produce identical SVG"
    )


def test_cross_axis_chrome_unchanged_when_breaking_on_flipped():
    """The cross axis's chrome is not affected when breaking the other channel on flipped.

    When BreakAxis(axis="y") is applied to a flipped chart:
    - the value channel (drawn on x) is broken → x ticks remap
    - the category channel (drawn on y, ordinal) is NOT broken → y ticks stay unchanged

    This test verifies that y-axis ticks are in the same positions in both
    the unbroken and broken charts.
    """
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 100.0, 50.0, 200.0, 75.0],
        }
    )

    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")
    flipped = base_chart.coord(fm.CoordFlip())

    svg_unbroken = flipped.to_svg()
    svg_broken = (flipped + fm.BreakAxis(axis="y", gap=(60.0, 150.0))).to_svg()

    # Extract y-axis ticks (the cross/unbroken axis) from both.
    unbroken_y_ticks = axis_tick_dash_label_pairs(svg_unbroken, axis="y")
    broken_y_ticks = axis_tick_dash_label_pairs(svg_broken, axis="y")

    # The y-axis (cross axis, carrying ordinal categories) should be unchanged.
    unbroken_y_positions = sorted([tick_along(t.dash, axis="y") for t in unbroken_y_ticks])
    broken_y_positions = sorted([tick_along(t.dash, axis="y") for t in broken_y_ticks])

    # Both should have the same number of ticks (all visible, none hidden).
    assert len(unbroken_y_positions) == len(broken_y_positions), (
        "the y-axis (cross axis, not broken) should have the same number of ticks "
        "before and after applying BreakAxis(axis='x')"
    )

    # Positions should be approximately the same (within 1 pixel).
    for unbroken_pos, broken_pos in zip(unbroken_y_positions, broken_y_positions):
        assert abs(unbroken_pos - broken_pos) < 1.0, (
            f"y-axis tick positions should be unchanged; "
            f"unbroken {unbroken_pos} vs broken {broken_pos}"
        )
