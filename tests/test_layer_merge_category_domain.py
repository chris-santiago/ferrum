"""`#247`B — a cross-DataFrame layer merge must not grow a phantom `null` band.

`Chart + Chart` over two different DataFrames (the merged-flat lowering, which
`.interactive()` and `LayerChart(resolve={"y": "independent"})` also route
through) concatenates the two frames diagonally, padding each frame's columns
with nulls over the other frame's rows. Those padded rows are structural: no
mark of either layer occupies them (asserted below by counting the drawn
marks). Banding them costs the axis a slot no mark stands in and makes every
real band narrower than the equivalent single-layer chart's.

The sibling module for `#247`A (the numeric union) is
`tests/test_layer_union_provenance.py`; this one is scoped to the categorical
domain because the two halves of `#247` have different mechanisms and only one
of them is bar-specific.

**The banding policy is not what this pins against.** The campaign's standing
rule is that a genuine null ordinal-anchor row bands at its own `null`
category (`arrow_cast::distinct_positional_categories`, FA-9), and stage 3 of
this wave deliberately strengthened it. `genuine_null_category_still_bands_*`
are the controls for that: a fix that stopped banding nulls, or that stripped
nulls from the column wholesale, passes every phantom test here and fails
those two.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm

from _svg_extents import axis_tick_dash_label_pairs, axis_tick_labels, tick_along
from _svg_rects import data_bar_rects, mark_layer, plot_rect

#: Two frames' worth of the same three categories, each with its own value
#: column scale, so a layer clipped out of view is visible as a missing mark
#: rather than as an overlapping one.
LOW = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
HIGH = pl.DataFrame({"cat": ["a", "b", "c"], "val": [40.0, 50.0, 60.0]})


def band_center_fractions(svg: str) -> list[float]:
    """Each drawn `<circle>`'s centre as a fraction of the plot rect's width.

    Reported as a fraction rather than as a pixel `cx` so a merged chart and
    its single-layer twin are comparable: the two have different y tick-label
    widths, so their plot rects start at different x. The *fraction* is a
    function of the band ladder alone, which is the thing under test -- three
    bands put mark centres at 1/6, 3/6, 5/6 of the plot width; a fourth
    (phantom) band moves them to 1/8, 3/8, 5/8.
    """
    x, _, w, _ = plot_rect(svg)
    return sorted(
        round((float(m.group(1)) - x) / w, 4)
        for m in re.finditer(r'<circle[^>]*\bcx="([-\d.]+)"', mark_layer(svg))
    )


def bar_width_fraction(svg: str) -> float:
    """The single bar width every bar in *svg* shares, as a fraction of plot width.

    Asserts the bars really do share one width: a partial fix that left the
    phantom band in the ladder but sized bars off something else would
    otherwise read as a pass.
    """
    _, _, w, _ = plot_rect(svg)
    widths = {round(float(a["width"]) / w, 4) for a in data_bar_rects(mark_layer(svg))}
    assert len(widths) == 1, f"expected one shared bar width, got {sorted(widths)}"
    return widths.pop()


def circle_count(svg: str) -> int:
    return len(re.findall(r"<circle", mark_layer(svg)))


def test_point_point_merge_carries_only_the_categories_its_marks_occupy():
    """point+point, not just bar+point: the padding is the merge's, not `mark_bar`'s.

    A bar-only regression test would pass for a fix scoped to the bar
    zero-anchor path, which is `#247`A's mechanism and not this one.
    """
    merged = fm.Chart(LOW).mark_point().encode(x=fm.X("cat", type_="N"), y="val") + fm.Chart(
        HIGH
    ).mark_point().encode(x=fm.X("cat", type_="N"), y="val")
    svg = merged.to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c"]
    # Six marks, three per layer, and none of them at a fourth band: the
    # padded rows carry no marks, which is why banding them is wrong.
    assert circle_count(svg) == 6
    assert band_center_fractions(svg) == pytest.approx(
        [1 / 6, 1 / 6, 0.5, 0.5, 5 / 6, 5 / 6], abs=2e-3
    )


def test_point_point_merge_band_centres_match_the_single_layer_chart():
    """Band widths match the equivalent single-layer chart, measured not assumed."""
    single = fm.Chart(LOW).mark_point().encode(x=fm.X("cat", type_="N"), y="val")
    merged = single + fm.Chart(HIGH).mark_point().encode(x=fm.X("cat", type_="N"), y="val")
    assert band_center_fractions(merged.to_svg()) == pytest.approx(
        sorted(band_center_fractions(single.to_svg()) * 2), abs=2e-3
    )


def test_bar_point_merge_bars_are_as_wide_as_the_single_layer_chart():
    """The reported shape. Band width is the observable the phantom slot steals from."""
    single = fm.Chart(LOW).mark_bar().encode(x=fm.X("cat", type_="N"), y="val")
    merged = single + fm.Chart(HIGH).mark_point().encode(x=fm.X("cat", type_="N"), y="val")
    svg = merged.to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c"]
    assert bar_width_fraction(svg) == pytest.approx(bar_width_fraction(single.to_svg()), abs=2e-3)


def test_disjoint_column_names_merge_carries_no_phantom_band():
    """The no-rename arm of the merge (`chart.py`'s `else` branch) pads too.

    Distinct column names skip the collision rename, so this exercises the
    second of the two `pl.concat(how="diagonal")` call sites rather than
    re-testing the first.
    """
    left = pl.DataFrame({"catA": ["a", "b", "c"], "valA": [1.0, 2.0, 3.0]})
    right = pl.DataFrame({"catB": ["a", "b", "c"], "valB": [40.0, 50.0, 60.0]})
    merged = fm.Chart(left).mark_point().encode(x=fm.X("catA", type_="N"), y="valA") + fm.Chart(
        right
    ).mark_point().encode(x=fm.X("catB", type_="N"), y="valB")

    assert axis_tick_labels(merged.to_svg(), axis="x") == ["a", "b", "c"]


def test_unequal_row_counts_merge_carries_no_phantom_band():
    """Frames of different heights: the padding survives any row-alignment fix.

    Two frames of the same height can be composed with no padding at all, so a
    fix that only re-shapes the concat passes the tests above and fails here --
    a merged batch must be at least as tall as the taller frame, so the shorter
    frame's own columns necessarily carry entries no row of its data supplied.
    """
    short = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
    tall = pl.DataFrame({"cat": ["a", "b", "c", "d", "e"], "val": [40.0, 50.0, 60.0, 70.0, 80.0]})
    merged = fm.Chart(short).mark_point().encode(x=fm.X("cat", type_="N"), y="val") + fm.Chart(
        tall
    ).mark_point().encode(x=fm.X("cat", type_="N"), y="val")

    assert "null" not in axis_tick_labels(merged.to_svg(), axis="x")


def test_row_null_in_both_positional_channels_claims_no_band_unmerged():
    """The same defect with no merge in sight, which is what places the fix.

    One row null in BOTH the category channel and its paired value channel.
    No mark is drawn for it -- a point needs a y -- so the band it claims is
    empty for exactly the reason the merge's padded rows' band is. The merge
    manufactures rows of this shape in bulk; it does not create the shape. A
    fix scoped to `chart.py`'s concat leaves this case standing.
    """
    frame = pl.DataFrame({"cat": ["a", "b", "c", None], "val": [1.0, 2.0, 3.0, None]})
    svg = fm.Chart(frame).mark_point().encode(x=fm.X("cat", type_="N"), y="val").to_svg()

    assert circle_count(svg) == 3
    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c"]


def test_genuine_null_category_still_bands_unlayered():
    """Control: the standing FA-9 policy, unlayered. A null the user supplied bands."""
    frame = pl.DataFrame({"cat": ["a", None, "c"], "val": [1.0, 2.0, 3.0]})
    svg = fm.Chart(frame).mark_point().encode(x=fm.X("cat", type_="N"), y="val").to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "c", "null"]


def test_genuine_null_category_still_bands_inside_a_merge():
    """Control: the policy holds *through* the merge, which is the whole distinction.

    The left frame supplies a real null category carrying a real value. It
    bands, and the merge's own padding does not -- so "no phantom" cannot be
    satisfied by dropping nulls from the column, only by distinguishing a null
    a mark occupies from one manufactured to square off the batch.
    """
    nullish = pl.DataFrame({"cat": ["a", None, "c"], "val": [1.0, 2.0, 3.0]})
    # The right frame's categories are a SUBSET of the left's real ones, so no
    # mark goes missing for the unrelated reason that a merged-flat chart's
    # ordinal domain is read off the chart-level column alone and never unions
    # the other layer's categories in (a separate defect, reported alongside
    # this one). Keeping that out of this control's way is what makes the mark
    # count below a statement about the null band.
    right = pl.DataFrame({"cat": ["a", "c"], "val": [40.0, 60.0]})
    merged = fm.Chart(nullish).mark_point().encode(x=fm.X("cat", type_="N"), y="val") + fm.Chart(
        right
    ).mark_point().encode(x=fm.X("cat", type_="N"), y="val")
    svg = merged.to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "c", "null"]
    # Five marks at three bands: the left layer draws all three of its rows
    # (including the null-category one, at the null band -- that is the
    # policy), the right layer draws its two, and the merge's own padded rows
    # draw nothing. The null band is occupied, not phantom.
    assert circle_count(svg) == 5


# ---------------------------------------------------------------------------
# The OTHER overlay spelling. `fm.layer(a, b)` lowers to a composite overlay
# tree of single-mark leaves and resolves its shared ordinal domain through
# `composite::leaf_positional_domain`, a different arm from the flat `a + b`
# merge above. The two arms pick the paired positional channel independently,
# so a fix to one is not a fix to the other and an x/y mix-up in either is
# silent: it would either keep every phantom band or drop every genuine one.
# Both outcomes are pinned, which is what makes the pair discriminating.
# ---------------------------------------------------------------------------

BOTH_NULL = pl.DataFrame({"cat": ["a", "b", "c", None], "val": [1.0, 2.0, 3.0, None]})
GENUINE_NULL = pl.DataFrame({"cat": ["a", None, "c"], "val": [1.0, 2.0, 3.0]})


def overlay_svg(left: pl.DataFrame, right: pl.DataFrame) -> str:
    return fm.layer(
        fm.Chart(left).mark_point().encode(x=fm.X("cat", type_="N"), y="val"),
        fm.Chart(right).mark_point().encode(x=fm.X("cat", type_="N"), y="val"),
    ).to_svg()


def test_composite_overlay_drops_an_unoccupied_null_band():
    """A leaf's own row null in both positional channels bands in neither lowering."""
    svg = overlay_svg(BOTH_NULL, LOW)

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c"]
    # Three per leaf and none at a fourth band, as in the flat spelling.
    assert circle_count(svg) == 6


def test_composite_overlay_keeps_a_genuine_null_band():
    """Control: the FA-9 policy holds through the composite arm too.

    Order is not asserted: the shared ordinal domain is a cross-leaf union whose
    ordering is `#247`A's concern, not this one. Membership is the property under
    test, and a paired-channel mix-up in `leaf_positional_domain` would empty the
    null band out of it.
    """
    svg = overlay_svg(GENUINE_NULL, LOW)

    labels = axis_tick_labels(svg, axis="x")
    assert sorted(labels) == ["a", "b", "c", "null"]
    # Three marks per leaf again -- the left leaf's null-category row draws, at
    # the null band, because it carries a real value.
    assert circle_count(svg) == 6


# ---------------------------------------------------------------------------
# The stack ladder moves with the axis ladder.
#
# `position::apply_stack` bins its rows through the same category reader the
# ordinal band ladder is enumerated with, so a band the axis stops drawing and a
# bin the stack keeps cumulating could disagree -- and a stacked bar drawn off
# its tick is exactly the silent failure that would look like a layout bug rather
# than a domain bug. These two pin that they agree, as a measurement of where the
# bars landed rather than as a claim about which call sites were edited.
# ---------------------------------------------------------------------------

#: Three categories x two stack groups, plus one structural null row: null
#: category AND null value, so the band is unoccupied and no segment is drawn in
#: it. `grp` is non-null so the row still reaches the stack's binning -- the
#: point is that the stack keeps a bin the ladder has no band for.
STACK_ROWS: dict[str, list] = {
    "cat": ["a", "a", "b", "b", "c", "c"],
    "grp": ["g1", "g2", "g1", "g2", "g1", "g2"],
    "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
STACK_WITH_STRUCTURAL_NULL = pl.DataFrame(
    {
        "cat": [*STACK_ROWS["cat"], None],
        "grp": [*STACK_ROWS["grp"], "g1"],
        "val": [*STACK_ROWS["val"], None],
    }
)


def stacked_bar_svg(frame: pl.DataFrame) -> str:
    return (
        fm.Chart(frame)
        .mark_bar(position=fm.Stack())
        .encode(x=fm.X("cat", type_="N"), y="val", color="grp")
        .to_svg()
    )


def test_stacked_segments_sit_on_the_ticks_the_axis_actually_draws():
    """Measured: every bar segment's centre is a tick the x axis drew.

    Compared against the ticks read back out of THIS chart's own rendered axis,
    not against computed band arithmetic -- if the ladder and the stack disagreed
    the bars would be at four-band centres while the axis drew three ticks, and
    no amount of internally consistent band math would notice.
    """
    svg = stacked_bar_svg(STACK_WITH_STRUCTURAL_NULL)

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c"]
    ticks = sorted(
        tick_along(dash, axis="x") for dash, _ in axis_tick_dash_label_pairs(svg, axis="x")
    )
    centres = sorted(
        {float(r["x"]) + float(r["width"]) / 2 for r in data_bar_rects(mark_layer(svg))}
    )
    assert len(centres) == 3, f"three bands, three stacks; got centres {centres}"
    for centre, tick in zip(centres, ticks, strict=True):
        # Pixel coordinates are emitted rounded to 3 decimals, so the tolerance
        # is the formatter's resolution and not a fudge factor.
        assert centre == pytest.approx(tick, abs=1e-3)


def test_a_structural_null_row_changes_nothing_about_a_stacked_chart():
    """The whole claim in one byte comparison: the row is structure, not data.

    A stacked bar chart whose category column carries a null-with-no-value must
    render byte-for-byte as the same chart with that row deleted. That covers the
    band ladder (same ticks, same widths), the stack cumulation (same segment
    heights, so the null row contributed nothing to any real bin) and the node
    count (no invisible zero-height rect left behind) in a single assertion --
    each of which could regress independently.
    """
    assert stacked_bar_svg(STACK_WITH_STRUCTURAL_NULL) == stacked_bar_svg(pl.DataFrame(STACK_ROWS))
