"""Characterization tests: BreakAxis + CoordFlip axis token semantics (#231).

BreakAxis.axis is a PHYSICAL-axis token: on a flipped chart, axis="y"
breaks the PHYSICAL y axis (which carries the flipped encoding). This test
pins that behavior and asserts chrome ownership consistency: the broken axis's
ticks/labels remap, the cross axis's don't.

Acceptance criterion: Break × CoordFlip characterization test pins
physical-token semantics (spec §4, acceptance criterion 5).
"""

from __future__ import annotations

import warnings

import polars as pl

import ferrum as fm
from _svg_extents import axis_tick_dash_label_pairs, tick_along
from _svg_rects import fill_rects


def _rects(svg: str, fill: str = "#2563eb") -> list[tuple[float, float, float, float]]:
    """Extract (x, y, w, h) for every mark rect painted in fill.

    Binds to `tests/_svg_rects.py`'s `fill_rects` (the home for this
    fill-filtered rect extraction, since it was identically duplicated across
    this module, `test_break_axis_geometry.py` and `test_break_axis_facet.py`).
    """
    return fill_rects(svg, fill)


def test_break_y_axis_on_flipped_chart_breaks_physical_y():
    """BreakAxis(axis="y") on a flipped chart breaks the PHYSICAL y axis.

    When a chart is flipped with CoordFlip(), the physical x and y axes swap.
    A BreakAxis(axis="y") is a PHYSICAL-axis token: it breaks the physical y
    axis regardless of what encoding it carries.

    On a chart with x="category:N", y="value:Q":
    - Without flip: BreakAxis(axis="y") breaks physical y, which carries value:Q
    - With flip: BreakAxis(axis="y") breaks physical y, which now carries category:N
                 (from the original x encoding)

    Since breaks only work on quantitative axes, a break on the ordinal physical y
    axis (the flipped x, carrying categories) produces a warning with no effect.
    This test pins that the axis parameter is interpreted as a PHYSICAL token,
    proven by the warning that the axis is ordinal.
    """
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 100.0, 50.0, 500.0, 75.0],
        }
    )

    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")

    # Flipped chart with a break on the PHYSICAL y axis.
    # Physical y now carries the category:N encoding (from original x).
    # Breaks don't apply to ordinal axes, so this should warn.
    flipped_with_break = base_chart.coord(fm.CoordFlip()) + fm.BreakAxis(axis="y", gap=(2.5, 3.5))

    # Capture the warning that the break has no effect on the ordinal axis.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg_flipped = flipped_with_break.to_svg()

    # Verify a warning was issued about the ordinal axis.
    messages = [str(w.message) for w in caught]
    assert any("ordinal" in m.lower() or "no effect" in m.lower() for m in messages), (
        f"Expected a warning that BreakAxis on the ordinal (flipped) y axis has no effect; "
        f"got {messages}"
    )

    # The SVG should still render without error, just without the break effect.
    assert "<svg" in svg_flipped
    rects = _rects(svg_flipped)
    assert rects, "flipped chart must still render bars"


def test_break_x_axis_on_flipped_chart_clips_the_value_bar_crossing_the_gap():
    """BreakAxis(axis="x") on a flipped chart clips marks against the PHYSICAL x axis.

    Mirrors ``test_break_axis_geometry.py``'s vertical clipping pin, rotated: on
    a flipped chart the bars are horizontal and vary along x (the value
    encoding). A bar whose value crosses the gap must render as two clipped
    pieces sharing the same row (y), not as one continuous (unclipped) rect --
    the assertion the predecessor test never made (spec review cycle 1, finding
    #1: the old test's only assertions -- an SVG came back, and some rect had
    y > 0 -- hold identically under the rejected logical-token reading, where
    axis="x" would name the still-ordinal category channel and the break would
    be refused as ineffective).
    """
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 100.0, 50.0, 500.0, 75.0],
        }
    )

    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")

    # Flip the chart, then break the PHYSICAL x axis (value dimension).
    # The value=500 bar crosses the gap (150, 400) and must split into two
    # clipped horizontal pieces sharing the same row.
    flipped_with_break = base_chart.coord(fm.CoordFlip()) + fm.BreakAxis(
        axis="x", gap=(150.0, 400.0)
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


def test_break_y_axis_effective_on_a_transposed_flip_while_x_warns_ordinal():
    """The spec's headline positive case (spec review cycle 1, finding #4).

    Every other test in this file fixes x="category:N", y="value:Q", so
    flipping always makes physical y ORDINAL: axis="y" is only ever shown to
    warn/no-op, never to break effectively. This is the missing positive
    mirror: with the quantitative/ordinal encodings swapped before the flip
    (x="value:Q", y="category:N"), flipping puts the quantitative value
    encoding on physical y -- so BreakAxis(axis="y") lands there EFFECTIVELY,
    with no warning, while BreakAxis(axis="x") on the SAME flipped chart warns
    that the physical x axis (now ordinal) had no effect.
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

    with warnings.catch_warnings(record=True) as caught_y:
        warnings.simplefilter("always")
        svg_break_y = (flipped + fm.BreakAxis(axis="y", gap=gap)).to_svg()
    assert not caught_y, (
        f"axis='y' must break the physical y (value) axis effectively with no "
        f"warning; got {[str(w.message) for w in caught_y]}"
    )
    assert svg_break_y != svg_plain, "an effective break must change the render"

    y_ticks = axis_tick_dash_label_pairs(svg_break_y, axis="y")
    hidden = [t for t in y_ticks if tick_along(t.dash, axis="y") < -1000]
    visible = [t for t in y_ticks if tick_along(t.dash, axis="y") > 0]
    assert len(hidden) == 5 and len(visible) == 5, (
        f"expected 5 hidden + 5 visible physical-y tick dashes under the gap; "
        f"got hidden={len(hidden)} visible={len(visible)} total={len(y_ticks)}"
    )

    with warnings.catch_warnings(record=True) as caught_x:
        warnings.simplefilter("always")
        svg_break_x = (flipped + fm.BreakAxis(axis="x", gap=gap)).to_svg()
    messages = [str(w.message) for w in caught_x]
    assert any("ordinal" in m for m in messages), (
        f"axis='x' on this flipped chart (physical x = ordinal category) must "
        f"warn that the break had no effect; got {messages}"
    )
    assert svg_break_x == svg_plain, (
        "the ordinal no-op must render byte-identical to the plain chart"
    )


def test_break_y_axis_is_a_no_op_while_break_x_axis_clips_on_the_same_flipped_chart():
    """The physical-token reading predicts an ASYMMETRIC result on one fixture.

    On the x="category:N", y="value:Q" fixture flipped, axis="y" (physical y,
    carrying the ordinal category encoding) must be a byte-identical no-op
    while axis="x" (physical x, carrying the quantitative value encoding) must
    clip. The rejected logical-token reading predicts the OPPOSITE asymmetry
    (axis="y" would name the still-quantitative value channel; axis="x" would
    name the still-ordinal category channel), so pinning WHICH SIDE is
    effective -- not merely that the two renders differ -- is what
    discriminates the two readings (spec review cycle 1, finding #3: the
    predecessor test's bare ``svg_break_y != svg_break_x`` held under both
    readings, since each reading makes exactly one of the two axes effective).
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
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        svg_break_y = (flipped + fm.BreakAxis(axis="y", gap=gap)).to_svg()
    svg_break_x = (flipped + fm.BreakAxis(axis="x", gap=gap)).to_svg()

    assert svg_break_y == svg_plain, (
        "axis='y' on the flipped (ordinal-physical-y) chart must be a byte-identical no-op"
    )
    assert len(_rects(svg_break_x)) == 6, (
        f"axis='x' on the same flipped chart must clip the value-crossing "
        f"bar; got {_rects(svg_break_x)}"
    )


def test_chrome_ownership_break_x_affects_value_axis_on_flipped():
    """On a flipped chart, BreakAxis(axis="x") breaks the physical x (value axis).

    Chrome ownership consistency: the BROKEN axis's ticks/labels remap
    (move/hide per the break), while the cross axis's ticks/labels stay in
    their original positions.

    On a non-flipped chart with x="category:N", y="value:Q":
    - BreakAxis(axis="y", gap) breaks the physical y (value axis)

    On a flipped version of the same chart:
    - BreakAxis(axis="x", gap) breaks the physical x (value axis)

    These two should produce similar results since they break the same
    encoding dimension (value), just on different physical axes due to flip.
    """
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 100.0, 50.0, 200.0, 75.0],
        }
    )

    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")

    # Non-flipped baseline: break on physical y (value axis).
    svg_unflipped_break_y = (base_chart + fm.BreakAxis(axis="y", gap=(60.0, 150.0))).to_svg()

    # Flipped chart: break on physical x (value axis, which was physical y before flip).
    svg_flipped_break_x = (
        base_chart.coord(fm.CoordFlip()) + fm.BreakAxis(axis="x", gap=(60.0, 150.0))
    ).to_svg()

    # Both should have ticks remapped on the broken axis.
    unflipped_y_ticks = axis_tick_dash_label_pairs(svg_unflipped_break_y, axis="y")
    flipped_x_ticks = axis_tick_dash_label_pairs(svg_flipped_break_x, axis="x")

    # The broken axis should have some ticks hidden (due to the gap).
    # We expect at least 1 tick to be hidden (y < -1000) on the unflipped y-axis.
    unflipped_hidden_y = [t for t in unflipped_y_ticks if tick_along(t.dash, axis="y") < -1000]
    assert unflipped_hidden_y, (
        "BreakAxis(axis='y', gap=(60, 150)) on the value axis should hide some ticks"
    )

    # Similarly, the flipped chart's x-axis should have some ticks hidden.
    flipped_hidden_x = [t for t in flipped_x_ticks if tick_along(t.dash, axis="x") < -1000]
    assert flipped_hidden_x, (
        "BreakAxis(axis='x', gap=(60, 150)) on a flipped chart (physical x = value) "
        "should hide some ticks"
    )


def test_chrome_ownership_consistency_break_x_on_flipped():
    """On a flipped chart with BreakAxis(axis="x"), the x-axis chrome remaps.

    With BreakAxis(axis="x") on a flipped chart, the PHYSICAL x axis is broken.
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

    # Flipped chart with break on physical x (value dimension).
    svg_broken = (
        base_chart.coord(fm.CoordFlip()) + fm.BreakAxis(axis="x", gap=(80.0, 180.0))
    ).to_svg()

    # Extract x-axis ticks and labels from both SVGs.
    # The x-axis is the bottom row (largest y).
    unbroken_x_ticks = axis_tick_dash_label_pairs(svg_unbroken, axis="x")
    broken_x_ticks = axis_tick_dash_label_pairs(svg_broken, axis="x")

    # With a break on the physical x axis, some ticks/labels should be hidden
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

    # The break on physical x should affect x-axis ticks.
    # At least some x-ticks should be hidden or have different positions.
    assert len(broken_visible_x) != len(unbroken_visible_x) or any(
        tick_along(t.dash, axis="x") != tick_along(u.dash, axis="x")
        for t, u in zip(broken_x_ticks, unbroken_x_ticks)
        if tick_along(t.dash, axis="x") > 0 and tick_along(u.dash, axis="x") > 0
    ), "the x-axis (which is broken on physical x) should differ in tick positions or visibility"


def test_indicator_on_broken_axis_on_flipped_chart():
    """Break indicator is drawn on the PHYSICAL broken axis.

    Mirrors ``test_break_axis_geometry.py``'s break-style parity check: with a
    bar crossing the gap, ``break_style="slash"`` must render strictly MORE
    ``<line>`` elements than ``break_style="gap"`` (which draws no indicator at
    all) on the SAME flipped chart, axis="x" break -- proving an indicator is
    actually drawn on the physical x axis, not merely that some SVG came back
    (the vacuous predecessor's only assertions, spec review cycle 1, finding
    #2: ``"<svg" in svg`` and ``len(svg) > 100`` both hold on an SVG with no
    break indicator at all).
    """
    df = pl.DataFrame({"category": ["A", "B"], "value": [72.0, 900.0]})
    base_chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")
    flipped = base_chart.coord(fm.CoordFlip())
    gap = (150.0, 850.0)

    svg_no_indicator = (flipped + fm.BreakAxis(axis="x", gap=gap, break_style="gap")).to_svg()
    svg_slash = (flipped + fm.BreakAxis(axis="x", gap=gap, break_style="slash")).to_svg()

    assert len(_rects(svg_slash)) == 3, "one short bar + the crossing bar's two clipped pieces"
    assert svg_slash.count("<line") > svg_no_indicator.count("<line"), (
        "break_style='slash' must draw indicator lines the 'gap' style omits, "
        "on the flipped chart's physical x axis"
    )


def test_flipped_chart_without_break_same_as_break_on_ordinal_physical_y():
    """A flipped chart with BreakAxis on the ordinal physical y axis is unchanged.

    Sanity check: applying a break to the ordinal physical y axis (the flipped
    x carrying categories) has no effect and produces a warning, so the SVG
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

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        svg_with_break = (flipped + fm.BreakAxis(axis="y", gap=(2.0, 3.5))).to_svg()

    # Since the break has no effect on an ordinal axis, the SVGs should be identical.
    assert svg_no_break == svg_with_break, (
        "BreakAxis(axis='y') on a flipped chart with ordinal physical y "
        "should have no effect and produce identical SVG"
    )


def test_cross_axis_chrome_unchanged_when_breaking_on_flipped():
    """The cross axis's chrome is not affected when breaking the other physical axis on flipped.

    When BreakAxis(axis="x") is applied to a flipped chart:
    - x-axis (physical, carrying values) is broken → x ticks remap
    - y-axis (physical, carrying categories, ordinal) is NOT broken → y ticks stay unchanged

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
    svg_broken = (flipped + fm.BreakAxis(axis="x", gap=(60.0, 150.0))).to_svg()

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
