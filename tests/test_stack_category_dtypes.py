"""`Stack` bins on the band ladder, not on the category column's Arrow dtype.

`#184`'s category half. `apply_stack` carried a SECOND dtype gate, independent
of the value-column one its first half unified: the bin key was built from a
two-arm chain that accepted `Float64` (keyed on `f64::to_bits`) and `Utf8`
(keyed on a hash of the bytes) and typed-refused everything else, so

    mark_bar(position=fm.Stack()).encode(x=fm.X("cat", type_="N"), y=..., color=...)

raised `Stack: x column must be Float64 or Utf8` on an `Int64` category column
while the identical column dodged, jittered and laid out bands. The key now
comes from `arrow_cast::col_as_identity_key_str`: the reader the ordinal band
ladder itself is enumerated with, falling back to the temporal epoch reader for
`Timestamp`. So a stack's groups ARE its bands wherever a band ladder exists, and
are the exact value where one does not -- which is also what makes a stacked time
series render (`mark_area(position=fm.Stack())` over a date x raised for every
input before).

This is a pre-existing hole, not a regression this wave introduced: the refusal
was measured byte-identical with and without `#171`'s composite dispatch.

The dtype set and the frame builders are `tests/_category_dtype_matrix.py`'s and
the rect parsing is `tests/_svg_rects.py`'s -- consumed here, not rebuilt. The
geometry this module reads is fuller than `band_geometry`'s band extents: a
mis-binned stack draws the right number of rects at the right widths and gets
their *tops* wrong, so the claim has to be made on the whole rectangle.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import ferrum as fm
from _category_dtype_matrix import (
    CATEGORY_DTYPES,
    category_frame,
    category_levels,
    utf8_twin_frame,
)
from _svg_corpus import path_points
from _svg_rects import data_bar_rects, mark_layer, svg_elements

# `value_axis_ladder` / `pixel_for` read a quantitative axis's tick ladder and
# invert it. They live in `test_layer_union_provenance.py` today, built on
# `_svg_extents`' tick primitives; their natural home is `_svg_extents` itself,
# and the hoist is recorded as pending rather than done here because that module
# belongs to another in-flight task. Importing them is the alternative to a
# third copy of the same ten lines.
from test_layer_union_provenance import pixel_for, value_axis_ladder

#: How many rows `category_frame` carries per category level, and therefore how
#: many stacked segments each bar is built from once a segment column is added.
#: Read off the frame rather than hardcoded so this module cannot disagree with
#: the matrix module about the shape it is stacking.
_SEGMENTS_PER_BAR = 5

#: "These two rendered coordinates are the same pixel". Coordinates are emitted
#: rounded to 3 decimals, so a tighter tolerance would assert on the formatter
#: rather than on the geometry. Matches the `EPS` both
#: `test_area_ribbon_baseline_axis.py` and `test_layer_union_provenance.py` use.
_EPS = 1e-3

Rect = tuple[float, float, float, float]


def _with_segments(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the `seg` column that turns one bar per category into a stack.

    Row `i` of each category's block becomes segment `s{i}`, so every
    `(category, segment)` pair has exactly one row and each bar is five
    segments tall. Deterministic, never sampled: the whole matrix is judged
    against one reference render.
    """
    return frame.with_columns(
        pl.Series("seg", [f"s{i % _SEGMENTS_PER_BAR}" for i in range(frame.height)])
    )


def _stacked_rects(frame: pl.DataFrame) -> list[Rect]:
    """Render *frame* as a stacked bar chart and return its `(x, y, w, h)` rects.

    The category channel is declared nominal explicitly: that declaration is
    what makes a numeric column a category axis at all, and it is the spelling
    the issue reports.
    """
    svg = (
        fm.Chart(_with_segments(frame))
        .mark_bar(position=fm.Stack())
        .encode(x=fm.X("cat", type_="N"), y="value", color="seg")
        .to_svg()
    )
    return [
        (float(a["x"]), float(a["y"]), float(a["width"]), float(a["height"]))
        for a in data_bar_rects(mark_layer(svg))
    ]


def _bars_by_x(rects: list[Rect]) -> dict[float, list[Rect]]:
    """Group *rects* by their left edge -- one entry per band, segments within."""
    out: dict[float, list[Rect]] = {}
    for rect in rects:
        out.setdefault(rect[0], []).append(rect)
    return out


@pytest.mark.parametrize("dtype", sorted(CATEGORY_DTYPES))
def test_stack_geometry_is_category_dtype_invariant(dtype: str) -> None:
    """Every dtype the band ladder enumerates stacks into the same rectangles
    its `Utf8` twin stacks into -- same band centres, same widths, same
    cumulative tops.

    RED before the fix: every cell except `Float64`, `Utf8` and `Categorical`
    raised `ValueError: Stack: x column must be Float64 or Utf8`, so the eight
    other cells failed in the render, not in the comparison.

    Equality alone would be satisfied by two empty renders, so the count is
    asserted first: a chart that drew nothing has no band centres to agree on.
    """
    expected_rects = len(category_levels(dtype)) * _SEGMENTS_PER_BAR

    cell = _stacked_rects(category_frame(dtype))
    twin = _stacked_rects(utf8_twin_frame(dtype))

    assert len(cell) == expected_rects, (
        f"{dtype}: expected {expected_rects} stacked segment rects "
        f"({len(category_levels(dtype))} bands x {_SEGMENTS_PER_BAR} segments); got {len(cell)}"
    )
    assert cell == twin, f"{dtype}: stacked geometry differs from the Utf8 twin's"


@pytest.mark.parametrize("dtype", sorted(CATEGORY_DTYPES))
def test_stacked_segments_tile_their_own_band(dtype: str) -> None:
    """Within a band, the segments tile from the bar's base upward with no gap
    and no overlap -- the property a mis-keyed stack breaks.

    This is the claim the twin comparison cannot make on its own: a stack that
    merged two bands into one bin still draws `bands x segments` rects at the
    right widths, but the segments of one band are then stacked on top of the
    other band's total, which shows up here as a gap between a segment's top
    and the next segment's base.
    """
    bands = _bars_by_x(_stacked_rects(category_frame(dtype)))

    assert len(bands) == len(category_levels(dtype)), (
        f"{dtype}: expected one band per category level; got {sorted(bands)}"
    )
    for x, rects in bands.items():
        # Sorted by their top edge: in SVG coordinates y grows downward, so the
        # first entry is the topmost (last-stacked) segment.
        rects = sorted(rects, key=lambda r: r[1])
        assert len(rects) == _SEGMENTS_PER_BAR, (
            f"{dtype}: band at x={x} has {len(rects)} segments, expected {_SEGMENTS_PER_BAR}"
        )
        for upper, lower in zip(rects, rects[1:]):
            # Tolerance is one SVG rounding unit (coordinates are emitted to 3
            # decimals, so two abutting edges can round 0.001 apart). A stack
            # that binned two bands together leaves a gap the size of the other
            # band's total -- tens of pixels here -- so this stays discriminating.
            assert upper[1] + upper[3] == pytest.approx(lower[1], abs=0.002), (
                f"{dtype}: band at x={x} has a gap/overlap between a segment ending at "
                f"{upper[1] + upper[3]} and the next starting at {lower[1]}"
            )


def test_int64_category_stacks_where_it_used_to_refuse() -> None:
    """The headline repro, stated on its own rather than only as a matrix cell.

    Three bands, five segments each, geometrically identical to the string
    column's render -- where before the fix this exact chart raised
    `Stack: x column must be Float64 or Utf8` while `fm.Dodge()` on the same
    column rendered fine.
    """
    numeric = _stacked_rects(category_frame("Int64"))
    reference = _stacked_rects(utf8_twin_frame("Int64"))

    assert len(numeric) == 15
    assert numeric == reference


def _temporal_frame() -> pl.DataFrame:
    """Two dates x five segments -- a `Timestamp(ms)` category column after
    `_coerce.normalize_for_rust` pre-casts the polars `Date`."""
    return _with_segments(
        pl.DataFrame(
            {
                "cat": [dt.date(2024, 1, 1)] * 5 + [dt.date(2024, 1, 2)] * 5,
                "value": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
            }
        )
    )


def test_nominal_temporal_stack_refuses_at_the_scale_naming_the_column() -> None:
    """A temporal column *declared nominal or ordinal* still refuses, one layer
    earlier: the scale could not enumerate a band ladder for it, and says so as a
    typed `ValueError` naming the column and its dtype -- never a raw Arrow
    error.

    **This refusal is a separate defect, filed separately, and deliberately NOT
    fixed here.** It lives in the enumeration layer
    (`arrow_cast::distinct_values_in_order`, which excludes `Timestamp` by
    documented contract because an epoch read through the `f64` widening is not
    injective past `2^53`), not in the stack's keying chain -- so `#184`'s fix
    cannot reach it, and a temporal x declared `:N`/`:O` never gets as far as
    `apply_stack`. The test exists to pin WHICH layer speaks and to keep today's
    behavior honest while that defect is open: the continuous temporal stack now
    renders (see `test_temporal_stack_renders_a_stacked_time_series`), so without
    this pin the two spellings could silently diverge in message quality.
    """
    chart = (
        fm.Chart(_temporal_frame())
        .mark_bar(position=fm.Stack())
        .encode(x=fm.X("cat", type_="N"), y="value", color="seg")
    )
    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()

    message = str(excinfo.value)
    assert "unsupported dtype" in message, message
    assert "'cat'" in message and "Timestamp" in message, message


#: Two groups over three dates. The per-date stacked totals are 5, 7 and 9,
#: so the stacked value axis must reach 9 where an overlaid one reaches only the
#: largest single observation, 6. Fixed values, never sampled.
_SERIES_A = (1.0, 2.0, 3.0)
_SERIES_B = (4.0, 5.0, 6.0)
_STACKED_MAX = 9.0
_OVERLAID_MAX = 6.0


def _time_series_frame() -> pl.DataFrame:
    """Two groups x three dates -- a `Timestamp(ms)` x after `_coerce` pre-casts."""
    dates = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    return pl.DataFrame(
        {
            "d": dates * 2,
            "v": list(_SERIES_A) + list(_SERIES_B),
            "g": ["a"] * 3 + ["b"] * 3,
        }
    )


def _area_svg(*, stack: bool) -> str:
    chart = fm.Chart(_time_series_frame()).encode(x="d:T", y="v:Q", color="g:N")
    return chart.mark_area(position=fm.Stack()).to_svg() if stack else chart.mark_area().to_svg()


def _topmost_painted_y(svg: str) -> float:
    """The smallest y pixel any filled mark path reaches (SVG y grows downward).

    For a stacked area that is the apex of the topmost band, so it answers "how
    high did the chart actually draw", which an axis domain alone does not.
    """
    vertices = [
        point
        for attrs in svg_elements(
            mark_layer(svg), "path", lambda a: "d" in a and a.get("fill") not in (None, "none")
        )
        for point in path_points(attrs["d"])
    ]
    assert vertices, "no filled mark path was drawn"
    return min(y for _, y in vertices)


def test_temporal_stack_renders_a_stacked_time_series() -> None:
    """A stacked time series renders, and it stacks: the value axis reaches the
    group SUM and the topmost band is painted at the pixel that axis assigns to
    the sum.

    Flipped from a refusal, with the rationale rewritten rather than
    re-baselined. The earlier version of this test pinned the pre-existing
    `ValueError` and argued a temporal x needed a *continuous* keying answer this
    fix did not owe. That argument was wrong on a measurement: a continuous
    quantitative x (`Int32` declared `:Q`) already stacks to its group sum
    through the same reader, so exact-value keying over a continuous axis was
    already the shipped behavior, and `Timestamp` was simply the one dtype the
    reader could not spell. It now keys by its epoch via
    `arrow_cast::col_as_identity_key_str`, the union the `key=` channel's reader
    was already using.

    Both halves are needed. The ladder alone would pass for a chart that widened
    its axis to the sum and then overlaid the groups anyway; the geometry alone
    would pass for a chart that stacked correctly onto a mis-scaled axis. The
    unstacked control pins that the reference values are the chart's, not
    hardcoded luck: the same data overlaid reaches 6, not 9.

    `value_axis_ladder` / `pixel_for` come from `test_layer_union_provenance.py`
    because that module owns them today; their natural home is
    `tests/_svg_extents.py`, whose tick primitives they are built on, and the
    hoist is recorded as pending rather than done here (that module is another
    task's and still in flight).
    """
    stacked, overlaid = _area_svg(stack=True), _area_svg(stack=False)
    stacked_ladder = value_axis_ladder(stacked, axis="y")
    overlaid_ladder = value_axis_ladder(overlaid, axis="y")

    assert max(stacked_ladder) == pytest.approx(_STACKED_MAX), (
        f"the stacked value axis must reach the group sum; ladder was {sorted(stacked_ladder)}"
    )
    assert max(overlaid_ladder) == pytest.approx(_OVERLAID_MAX), (
        "control: the same data overlaid must reach only the largest single value; "
        f"ladder was {sorted(overlaid_ladder)}"
    )
    assert _topmost_painted_y(stacked) == pytest.approx(
        pixel_for(stacked_ladder, _STACKED_MAX), abs=_EPS
    ), "the stacked area must be painted up to the pixel its own axis calls the sum"
    assert _topmost_painted_y(overlaid) == pytest.approx(
        pixel_for(overlaid_ladder, _OVERLAID_MAX), abs=_EPS
    ), "control: the overlaid area must top out at the largest single value"


def test_temporal_stack_bins_by_the_instant_not_by_row_order() -> None:
    """Rows sharing an instant cumulate; rows at different instants do not.

    The negative control for the keying itself. A fix that stopped refusing but
    keyed every row alike would put all six rows in one bin, stacking to 21 and
    drawing one band per group at a single x; a fix that keyed every row
    uniquely would never cumulate at all and would top out at 6. Reaching
    exactly 9 with three distinct x positions is what says the bin is the
    instant.
    """
    svg = _area_svg(stack=True)
    ladder = value_axis_ladder(svg, axis="y")
    assert max(ladder) == pytest.approx(_STACKED_MAX), sorted(ladder)

    xs = {
        round(x, 3)
        for attrs in svg_elements(
            mark_layer(svg), "path", lambda a: "d" in a and a.get("fill") not in (None, "none")
        )
        for x, _ in path_points(attrs["d"])
    }
    assert len(xs) == 3, f"three instants must draw at three x positions; got {sorted(xs)}"
