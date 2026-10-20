"""The shared ordinal axis covers every layer's categories, not just layer 0's.

A layered chart's ordinal axis used to read its categories off the chart-level
column alone. The numeric side has unioned every layer's extent since long
before (`scale_resolve::domain::numeric_domain_union`'s `spec.layers` loop); the
categorical side had no equivalent, so a cross-DataFrame `a + b` merge -- whose
RHS columns are *renamed*, leaving the chart-level column covering the LHS data
alone -- lost every RHS category. No band existed for them, so every RHS mark was
silently dropped: measured, three circles for six rows.

**Ordering is the ruled part.** Union by first appearance, in layer order:
`distinct_values_in_order` already orders categories by first appearance within a
column, and the union extends that rule across the layer boundary rather than
introducing a sort. Nothing sorts, so the assertions below are on exact label
ORDER, not on membership -- a sorting "fix" would put `['a','b','c','x','y','z']`
in the same place for the disjoint case and fail `partial_overlap` and
`right_layer_first`.

Scoped to the cross-layer union. The sibling module `test_layer_merge_category_domain.py`
owns `#247`B (when a null claims the `null` band) and its policy controls; this
module asserts the union preserves that rule rather than re-litigating it.

Two later sections extend the same subject from the wave-5 close review: which
coordinate space a layer's column is read in when the LAYER carries its own
`CoordFlip` (F1), and which layer columns the ladder will serve at all (F2).
Both are properties of the union loop. `test_layer_coord.py` owns the
complementary statement -- that a member's own coord is honored on each of the
three merged-flat routes -- and its ordinal row is the route-level sibling of
F1's section here.
"""

from __future__ import annotations

import datetime
import re

import polars as pl
import pytest

import ferrum as fm

from _svg_extents import axis_tick_dash_label_pairs, axis_tick_labels, tick_along
from _svg_rects import data_bar_rects, mark_layer, plot_rect

LEFT = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
#: Disjoint categories, and a value range well clear of LEFT's so a dropped
#: layer cannot hide behind an overlapping mark.
RIGHT_DISJOINT = pl.DataFrame({"cat": ["x", "y", "z"], "val": [40.0, 50.0, 60.0]})


def points(frame: pl.DataFrame) -> fm.Chart:
    return fm.Chart(frame).mark_point().encode(x=fm.X("cat", type_="N"), y="val")


def circle_count(svg: str) -> int:
    return len(re.findall(r"<circle", mark_layer(svg)))


def circle_centre_fractions(svg: str) -> list[float]:
    """Each drawn circle's `cx` as a fraction of the plot rect width.

    A fraction rather than a pixel so charts with different y tick-label widths
    stay comparable, mirroring `test_layer_merge_category_domain.py`'s helper of
    the same shape. Six bands put centres at 1/12, 3/12, ..., 11/12.
    """
    x, _, w, _ = plot_rect(svg)
    return sorted(
        round((float(m.group(1)) - x) / w, 4)
        for m in re.finditer(r'<circle[^>]*\bcx="([-\d.]+)"', mark_layer(svg))
    )


def x_tick_positions(svg: str) -> list[float]:
    return sorted(
        tick_along(dash, axis="x") for dash, _ in axis_tick_dash_label_pairs(svg, axis="x")
    )


# ---------------------------------------------------------------------------
# The measured defect.
# ---------------------------------------------------------------------------


def test_disjoint_layer_categories_all_band_and_all_marks_draw():
    """The reported shape: six rows, six bands, six marks.

    Before the union this rendered `['a','b','c']` and THREE circles. The mark
    count is the load-bearing assertion -- the pre-fix render was a perfectly
    valid SVG that simply contained half the data, which is why the defect
    survived.
    """
    svg = (points(LEFT) + points(RIGHT_DISJOINT)).to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c", "x", "y", "z"]
    assert circle_count(svg) == 6


def test_every_band_is_placed_not_merely_counted():
    """Six evenly spaced band centres, and every mark sits on a tick the axis drew.

    Counting labels proves the ladder has six entries; this proves the six are
    laid out as six bands and that the marks agree with them. The ticks are read
    back out of THIS chart's own rendered axis, so an internally consistent but
    wrong band arithmetic could not satisfy both halves.
    """
    svg = (points(LEFT) + points(RIGHT_DISJOINT)).to_svg()

    assert circle_centre_fractions(svg) == pytest.approx(
        [1 / 12, 3 / 12, 5 / 12, 7 / 12, 9 / 12, 11 / 12], abs=2e-3
    )
    ticks = x_tick_positions(svg)
    assert len(ticks) == 6
    centres = sorted(
        float(m.group(1)) for m in re.finditer(r'<circle[^>]*\bcx="([-\d.]+)"', mark_layer(svg))
    )
    for centre, tick in zip(centres, ticks, strict=True):
        # 1e-3 is the SVG coordinate formatter's resolution, not a fudge factor.
        assert centre == pytest.approx(tick, abs=1e-3)


def test_partially_overlapping_categories_union_without_duplicating():
    """Shared categories appear once, in the position they first appeared in.

    `b`/`c` come from the left layer and are not re-appended for the right, and
    `d` lands after `c` because that is where it first appears. A sort would give
    the same answer here only by coincidence of the alphabet; `right_layer_first`
    below is the case it cannot fake.
    """
    right = pl.DataFrame({"cat": ["b", "c", "d"], "val": [10.0, 20.0, 30.0]})
    svg = (points(LEFT) + points(right)).to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c", "d"]
    assert circle_count(svg) == 6


def test_first_appearance_in_layer_order_is_not_a_sort():
    """The discriminating ordering case: unsorted categories, in layer order.

    Left layer contributes `c, a`; right contributes `b`. First appearance in
    layer order is `['c', 'a', 'b']`. Any sorting union answers `['a','b','c']`,
    and a union that walked the layers backwards answers `['b','c','a']`.
    """
    left = pl.DataFrame({"cat": ["c", "a"], "val": [1.0, 2.0]})
    right = pl.DataFrame({"cat": ["b"], "val": [40.0]})
    svg = (points(left) + points(right)).to_svg()

    # Mark count first: `axis_tick_labels` reports `[]` for a tick row of fewer
    # than three labels (it cannot tell a two-tick row from an axis title), so
    # the two-band pre-fix ladder reads as empty rather than as `['c','a']`. The
    # count is unambiguous in both directions.
    assert circle_count(svg) == 3
    assert axis_tick_labels(svg, axis="x") == ["c", "a", "b"]


def test_three_layers_each_contribute_in_spec_order():
    """Not just the two-layer case: the loop walks every layer, in order."""
    third = pl.DataFrame({"cat": ["p", "q"], "val": [5.0, 6.0]})
    svg = (points(LEFT) + points(RIGHT_DISJOINT) + points(third)).to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c", "x", "y", "z", "p", "q"]
    assert circle_count(svg) == 8


def test_a_subset_layer_adds_no_band_and_still_draws():
    """A layer whose categories are a strict subset contributes nothing new.

    The union is a widening, never a rewrite: a subset layer must not append a
    duplicate band, and its marks must land on the existing ones.
    """
    right = pl.DataFrame({"cat": ["b"], "val": [40.0]})
    svg = (points(LEFT) + points(right)).to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c"]
    assert circle_count(svg) == 4
    # The subset layer's mark shares the middle band with the left layer's.
    assert circle_centre_fractions(svg) == pytest.approx([1 / 6, 0.5, 0.5, 5 / 6], abs=2e-3)


# ---------------------------------------------------------------------------
# The byte-identical control. Identical categories is the common case, and the
# evidence that this change widens the ladder rather than rebuilding it.
# ---------------------------------------------------------------------------


def test_identical_categories_render_byte_identically_to_the_unlayered_twin():
    """Two layers over the same three categories: the ladder is unchanged.

    Asserted against the SAME chart's own pre-union ladder via the single-layer
    twin's band centres, plus an exact byte comparison of the merged chart
    against a merged chart whose right layer is a re-spelling of the same data --
    if the union appended anything, the six centres would move to eighths.
    """
    same = pl.DataFrame({"cat": ["a", "b", "c"], "val": [7.0, 8.0, 9.0]})
    merged = points(LEFT) + points(same)
    svg = merged.to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c"]
    assert circle_count(svg) == 6
    single = points(LEFT).to_svg()
    assert circle_centre_fractions(svg) == pytest.approx(
        sorted(circle_centre_fractions(single) * 2), abs=2e-3
    )


def test_disjoint_column_names_with_identical_categories_still_band_once():
    """The no-rename arm of the merge: distinct column names, same categories.

    Distinct names skip the collision rename, so the layer field differs from the
    chart-level field and the union loop actually runs -- and must still produce
    three bands, because the categories themselves coincide.
    """
    left = pl.DataFrame({"catA": ["a", "b", "c"], "valA": [1.0, 2.0, 3.0]})
    right = pl.DataFrame({"catB": ["a", "b", "c"], "valB": [40.0, 50.0, 60.0]})
    merged = fm.Chart(left).mark_point().encode(x=fm.X("catA", type_="N"), y="valA") + fm.Chart(
        right
    ).mark_point().encode(x=fm.X("catB", type_="N"), y="valB")
    svg = merged.to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c"]
    assert circle_count(svg) == 6


# ---------------------------------------------------------------------------
# `#247`B's null rule, through the union. The union must not become a second way
# to manufacture a `null` band, and must not suppress a genuine one.
# ---------------------------------------------------------------------------


def test_the_merge_padding_still_bands_nothing_when_categories_are_disjoint():
    """`#247`B holds where the union now runs.

    Both layers' columns are padded with nulls over the other's rows. With the
    union in place those padded nulls are read on BOTH columns rather than one,
    so a union that ignored the occupancy rule would grow not one phantom band
    but one per layer.
    """
    svg = (points(LEFT) + points(RIGHT_DISJOINT)).to_svg()

    assert "null" not in axis_tick_labels(svg, axis="x")


def test_a_genuine_null_in_the_right_layer_bands_and_draws():
    """A null the RIGHT layer's own data supplies is not padding, and bands.

    This is the case the union makes reachable: before it, the right layer's
    categories -- null included -- never reached the ladder at all. The layer
    binds both positional channels, so `#247`B can tell its authored
    null-with-a-value from the merge's padding, and does: one null band, and a
    mark drawn in it.
    """
    right = pl.DataFrame({"cat": ["x", None], "val": [40.0, 50.0]})
    svg = (points(LEFT) + points(right)).to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c", "x", "null"]
    # Five bands, five marks: three left, two right (the null-category row draws
    # at the null band because it carries a real value).
    assert circle_count(svg) == 5


def test_a_reference_line_layers_structural_nulls_open_no_band():
    """A layer binding ONE positional channel cannot open the null band.

    `mark_rule(y=…)` over a constant-injected column is one authored value in a
    column of structural nulls, with no opposite coordinate of its own -- it
    inherits the chart-level one, whose values say nothing about this layer's
    rows. Judging the band against that inherited channel would open it and draw
    a rule per null row. One rule is the answer.
    """
    frame = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
    chart = fm.Chart(frame).mark_bar().encode(x=fm.X("cat", type_="N"), y="val") + fm.Chart(
        pl.DataFrame({"ref": ["b", None, None]})
    ).mark_rule().encode(x=fm.X("ref", type_="N"))
    svg = chart.to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c"]
    assert len(data_bar_rects(mark_layer(svg))) == 3


# ---------------------------------------------------------------------------
# Whose coordinate space the layer's column lives in (wave-5 close review F1),
# and which columns the ladder will serve at all (F2). Both are properties of
# the union loop, so they live here rather than in `test_layer_coord.py` --
# which owns the complementary statement, that a member's own coord is honored
# on each of the three merged-flat routes.
# ---------------------------------------------------------------------------

#: The flipped member binds its own (renamed, cross-frame) columns, so its
#: contribution is observable: authored `x=val2` / `y=cat2`, i.e. post-flip its
#: CATEGORIES occupy x and its values occupy y.
FLIPPED_MEMBER = pl.DataFrame({"cat2": ["x", "y", "z"], "val2": [40.0, 50.0, 60.0]})


def test_a_members_own_flip_decides_which_of_its_columns_reaches_the_ladder():
    """The flip is per LAYER, not per chart.

    A member carrying its own `CoordFlip` inside an unflipped chart has its
    authored y on the rendered x axis, so THAT is the column the x ladder takes.
    Measured pre-fix (the loop read the chart's flip, hoisted above the loop
    where a per-layer answer cannot be expressed): the ladder was
    `['a','b','c','40','50','60']` -- the member's VALUE column enumerated as
    three spurious bands, stealing band width from the real categories -- and the
    member drew 0 of its 3 circles, because the categories it needed a band for
    never reached the ladder. Both halves are asserted; either alone passes for a
    render that is still wrong.
    """
    left = fm.Chart(LEFT).mark_bar().encode(x=fm.X("cat", type_="N"), y="val")
    flipped = (
        fm.Chart(FLIPPED_MEMBER)
        .mark_point()
        .encode(x="val2", y=fm.Y("cat2", type_="N"))
        .coord(fm.CoordFlip())
    )
    svg = (left + flipped).to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c", "x", "y", "z"], (
        "the member's post-flip x (its `cat2` categories) is what widens the x ladder"
    )
    assert circle_count(svg) == 3, "…and with bands to occupy, all three of its marks draw"


def test_an_unflipped_twin_of_the_same_member_is_the_control():
    """The same member without the flip: its AUTHORED x is the categorical one.

    The two tests differ by one `.coord()` call, so a "fix" that ignored coords
    entirely would have to fail one of them.
    """
    left = fm.Chart(LEFT).mark_bar().encode(x=fm.X("cat", type_="N"), y="val")
    plain = fm.Chart(FLIPPED_MEMBER).mark_point().encode(x=fm.X("cat2", type_="N"), y="val2")
    svg = (left + plain).to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c", "x", "y", "z"]
    assert circle_count(svg) == 3


def test_a_temporal_layer_field_beside_a_categorical_axis_renders():
    """A layer column the ladder cannot enumerate is skipped, not refused (F2).

    Measured pre-fix: this raised `column 'when' has unsupported dtype:
    Timestamp(Millisecond, None) (cannot enumerate distinct values)` -- the
    ordinal side refusing where its numeric sibling
    (`numeric_domain_union`'s layer loop, "the field may not exist in all
    batches") shrugs. A temporal-x layer beside a categorical-x layer is not
    malformed, so REFUSE MALFORMED does not reach it: the bar layer draws, and the
    line layer places nothing on a band axis, which is what it did before the
    cross-layer ladder existed at all.
    """
    cat = pl.DataFrame({"cat": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    stamps = pl.DataFrame(
        {
            "when": [
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2020, 2, 1),
                datetime.datetime(2020, 3, 1),
            ],
            "v2": [4.0, 5.0, 6.0],
        }
    )
    svg = (
        fm.Chart(cat).mark_bar().encode(x=fm.X("cat", type_="N"), y="v")
        + fm.Chart(stamps).mark_line().encode(x=fm.X("when", type_="T"), y="v2")
    ).to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c"], (
        "no band is invented for a column no mark could key against"
    )
    assert len(data_bar_rects(mark_layer(svg))) == 3, "the categorical layer still draws"


def test_a_genuinely_unreadable_axis_column_still_refuses():
    """The skip is for a LAYER's contribution, not for the axis's own column.

    The same `Timestamp` column bound as a nominal axis — i.e. as the thing the
    ladder is made of — must still raise, naming the column. This is the half that
    keeps F2's fix from being a blanket swallow: only the layer-contribution read
    is gated, and only by a property (can this be a category key at all) that
    makes the skip lossless.
    """
    stamps = pl.DataFrame(
        {
            "when": [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 2, 1)],
            "v2": [4.0, 5.0],
        }
    )
    with pytest.raises(ValueError, match="when"):
        fm.Chart(stamps).mark_bar().encode(x=fm.X("when", type_="N"), y="v2").to_svg()


def test_a_quantitative_layer_column_the_ladder_can_serve_still_bands_and_draws():
    """The gate asks about the COLUMN, not about the declared type.

    A quantitative layer channel over a numeric column gets bands and its marks
    land on them — so gating on `Quantitative`/`Temporal` instead would silently
    drop these two points (and would also drop an unannotated integer
    categorical, which infers Quantitative). Byte-identical to before the F2 gate,
    which is the point.
    """
    cat = pl.DataFrame({"cat": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    numeric = pl.DataFrame({"n": [10, 20], "v2": [4.0, 5.0]})
    svg = (
        fm.Chart(cat).mark_bar().encode(x=fm.X("cat", type_="N"), y="v")
        + fm.Chart(numeric).mark_point().encode(x="n", y="v2")
    ).to_svg()

    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c", "10", "20"]
    assert circle_count(svg) == 2


# ---------------------------------------------------------------------------
# The stack ladder moves with the axis ladder. `position::apply_stack` bins rows
# per batch on the layer's OWN category column, while the ladder is the unioned
# one -- so a band the union added and a bin the stack keeps must agree, or a
# stacked bar lands off its tick. Measured, not asserted from call sites.
#
# One thing these deliberately do NOT assert: a segment per row. A merged-flat
# layered stack whose `color` column collides across the two frames loses the
# right layer's non-first colour group, and that loss is not this union's --
# it reproduces identically when the two frames share every category (so the
# ladder is unchanged), and disappears when `position=` is dropped (so the bands
# are present and the marks place). Reported separately rather than pinned here
# as if it were correct.
# ---------------------------------------------------------------------------

STACK_LEFT = pl.DataFrame(
    {"cat": ["a", "a", "b", "b"], "grp": ["g1", "g2", "g1", "g2"], "val": [1.0, 2.0, 3.0, 4.0]}
)
STACK_RIGHT = pl.DataFrame(
    {"cat": ["c", "c", "d", "d"], "grp": ["g1", "g2", "g1", "g2"], "val": [5.0, 6.0, 7.0, 8.0]}
)


def stacked_bars(frame: pl.DataFrame) -> fm.Chart:
    return (
        fm.Chart(frame)
        .mark_bar(position=fm.Stack())
        .encode(x=fm.X("cat", type_="N"), y="val", color="grp")
    )


def test_stacked_bars_across_layers_sit_on_the_ticks_the_axis_draws():
    """Every drawn segment's centre is a tick this chart's own x axis drew.

    Read back out of the rendered axis rather than recomputed from band
    arithmetic: if the ladder the axis ticks and the bins the stack cumulates
    disagreed, the bars would sit at two-band centres while the axis drew four
    ticks, and internally consistent band math would not notice.
    """
    svg = (stacked_bars(STACK_LEFT) + stacked_bars(STACK_RIGHT)).to_svg()

    centres = sorted(
        {round(float(r["x"]) + float(r["width"]) / 2, 3) for r in data_bar_rects(mark_layer(svg))}
    )
    # Four occupied bands: before the union the right layer's `c`/`d` had no band
    # and drew nothing at all, so this was two. Asserted before the label read
    # because that helper reports `[]` for a tick row of fewer than three labels.
    assert len(centres) == 4, f"four bands, four stacks; got {centres}"
    assert axis_tick_labels(svg, axis="x") == ["a", "b", "c", "d"]
    ticks = x_tick_positions(svg)
    assert len(ticks) == 4
    for centre, tick in zip(centres, ticks, strict=True):
        # 1e-3 is the coordinate formatter's resolution, not a fudge factor.
        assert centre == pytest.approx(tick, abs=1e-3)


def test_the_union_leaves_the_stack_cumulation_alone():
    """The left layer's two segments still stack: shared top, no re-baselining.

    A ladder change that perturbed the stack would show up as segments drawn
    from the axis baseline instead of from each other.
    """
    svg = (stacked_bars(STACK_LEFT) + stacked_bars(STACK_RIGHT)).to_svg()
    rects = data_bar_rects(mark_layer(svg))
    band_x = min(float(r["x"]) for r in rects)
    first_band = [r for r in rects if float(r["x"]) == pytest.approx(band_x, abs=1e-3)]

    assert len(first_band) == 2, f"band `a` holds two groups; got {first_band}"
    # The two segments tile: the upper one's bottom edge is the lower one's top.
    upper, lower = sorted(first_band, key=lambda r: float(r["y"]))
    assert float(upper["y"]) + float(upper["height"]) == pytest.approx(
        float(lower["y"]), abs=1e-3
    ), "a stacked segment starts where the one below it ends; got {upper}, {lower}"
    # 1:2 value ratio, so 1:2 pixel heights — the cumulation used the real values.
    # The SVG writes coordinates to 3 decimals: each height is off by up to
    # 0.0005, so twice the lower one and the upper one differ by up to 0.0015.
    assert float(upper["height"]) == pytest.approx(2 * float(lower["height"]), abs=2e-3)
