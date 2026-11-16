"""``BreakAxis(axis=...)`` under a caller's ``CoordFlip`` names the caller's channel.

A user who wrote ``encode(x='c', y='v')``, flipped the chart and broke ``y`` means
``v``, the numeric column they bound to ``y``, wherever the flip draws it. The
break, every warning it raises (in the caller's letter) and the annotations it
clips follow that channel, the same way annotations and facet-sort messages
already read the caller's letter under a flip.

Every positive case is pinned two ways. The warning list is compared whole, and
the render is compared with the transposed chart (the same data bound to the
exchanged channels, no flip) carrying the break on the channel it is drawn on:
the transposed chart is the unambiguous statement of what the break should do,
and it differs from what a break on the wrong channel draws.

A figure that applies its own flip (``catplot(orient='h')``) keeps the axis it
draws; those renders and the unflipped ones are pinned by digests measured
before the break routing changed.
"""

from __future__ import annotations

import hashlib
import re

import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation as ann
from _svg_rects import fill_rects
from test_null_group_warning import _render_capturing

_BAR_FILL = "#2563eb"
_RED = "#ff0000"

# The data every bar case shares: one numeric column over five categories.
_BARS = pl.DataFrame({"c": list("abcde"), "v": [10.0, 20.0, 50.0, 60.0, 70.0]})
# A gap that bars b..e's bodies cross (they run from 0), inside (0, 70).
_GAP = (25.0, 45.0)

# Two numeric columns whose ranges are far apart, so a break on the wrong one
# either misses the domain or removes the wrong range.
_NUM = pl.DataFrame(
    {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [100.0, 110.0, 120.0, 300.0, 310.0, 320.0],
    }
)
_NUM_GAP_A = (2.5, 4.5)

_ORDINAL = (
    "BreakAxis on the {axis} axis had no effect: the axis is ordinal, so its gaps have no "
    "numeric data range to remove"
)
_NO_OVERLAP = "BreakAxis on the {axis} axis had no effect: no gap overlaps the axis domain {domain}"
_CONFLICT = (
    "multiple BreakAxis specs on the {axis} axis disagree on break_style; the first "
    "declaration's value was kept"
)


def _break(axis: str, gap: tuple[float, float] = _GAP, **kwargs) -> fm.BreakAxis:
    """A ``BreakAxis`` naming the caller's ``axis`` letter."""
    return fm.BreakAxis(axis=axis, gap=gap, **kwargs)


def _bars() -> fm.Chart:
    """Bars of ``v`` over the categories ``c``, as the caller bound them (x=c, y=v)."""
    return fm.Chart(_BARS).mark_bar().encode(x="c:N", y="v:Q")


def _bars_transposed() -> fm.Chart:
    """The same bars bound the other way round (x=v, y=c): what ``_bars()`` flipped draws."""
    return fm.Chart(_BARS).mark_bar().encode(x="v:Q", y="c:N")


def _points() -> fm.Chart:
    """Points of ``b`` over ``a`` (x=a, y=b), both numeric."""
    return fm.Chart(_NUM).mark_point().encode(x="a:Q", y="b:Q")


def _points_transposed() -> fm.Chart:
    """The same points bound the other way round (x=b, y=a)."""
    return fm.Chart(_NUM).mark_point().encode(x="b:Q", y="a:Q")


def _flip(chart: fm.Chart) -> fm.Chart:
    return chart.coord(fm.CoordFlip())


def _svg(chart: fm.Chart) -> tuple[str, list[str]]:
    return _render_capturing(chart.to_svg)


def _digest(svg: str) -> str:
    return hashlib.sha256(svg.encode()).hexdigest()


def _bar_rects(svg: str) -> list[tuple[float, float, float, float]]:
    return fill_rects(svg, _BAR_FILL)


def _red_rects(svg: str) -> list[tuple[float, float, float, float]]:
    """``(x, y, w, h)`` of every annotation rect painted in the test's red."""
    pattern = (
        r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)"'
        r'[^>]*fill="' + re.escape(_RED) + '"'
    )
    return [tuple(float(g) for g in m.groups()) for m in re.finditer(pattern, svg)]


# ---------------------------------------------------------------------------
# bars: the caller's numeric y is broken, the caller's categorical x is not
# ---------------------------------------------------------------------------


def test_the_callers_numeric_y_breaks_with_no_warning_when_the_chart_is_flipped():
    flipped = _flip(_bars())
    plain, _ = _svg(flipped)

    svg, messages = _svg(flipped + _break("y"))

    assert messages == []
    assert svg != plain
    # b..e's bars cross the gap: three of them split into two pieces each, and
    # the two bars below the gap stay whole.
    assert len(_bar_rects(svg)) == 8


def test_the_callers_numeric_y_break_draws_what_the_transposed_chart_draws():
    svg, _ = _svg(_flip(_bars()) + _break("y"))
    twin, twin_messages = _svg(_bars_transposed() + _break("x"))

    assert twin_messages == []
    assert svg == twin


def test_the_callers_categorical_x_has_no_numeric_range_so_its_break_warns_in_the_callers_letter():
    flipped = _flip(_bars())
    plain, _ = _svg(flipped)

    svg, messages = _svg(flipped + _break("x"))

    assert messages == [_ORDINAL.format(axis="x")]
    assert svg == plain


def test_a_gap_outside_the_callers_numeric_domain_warns_in_the_callers_letter():
    flipped = _flip(_bars())
    plain, _ = _svg(flipped)

    svg, messages = _svg(flipped + _break("y", (200.0, 300.0)))

    assert messages == [_NO_OVERLAP.format(axis="y", domain="(0, 70)")]
    assert svg == plain


# ---------------------------------------------------------------------------
# numeric on both channels: the caller's letter picks the column, not the side
# ---------------------------------------------------------------------------


def test_the_callers_numeric_x_breaks_when_the_chart_is_flipped():
    flipped = _flip(_points())
    plain, _ = _svg(flipped)

    svg, messages = _svg(flipped + _break("x", _NUM_GAP_A))

    assert messages == []
    assert svg != plain


def test_the_callers_numeric_x_break_draws_what_the_transposed_chart_draws():
    svg, _ = _svg(_flip(_points()) + _break("x", _NUM_GAP_A))
    twin, twin_messages = _svg(_points_transposed() + _break("y", _NUM_GAP_A))

    assert twin_messages == []
    assert svg == twin


def test_the_callers_y_names_the_column_bound_to_y_not_the_one_drawn_on_the_physical_y():
    """``b`` is bound to y and drawn horizontally; a gap in ``a``'s range misses it."""
    flipped = _flip(_points())
    plain, _ = _svg(flipped)

    svg, messages = _svg(flipped + _break("y", _NUM_GAP_A))

    assert messages == [_NO_OVERLAP.format(axis="y", domain="(100, 320)")]
    assert svg == plain


def test_a_gap_outside_the_callers_x_domain_warns_in_the_callers_letter():
    flipped = _flip(_points())
    plain, _ = _svg(flipped)

    svg, messages = _svg(flipped + _break("x", (50.0, 60.0)))

    assert messages == [_NO_OVERLAP.format(axis="x", domain="(1, 6)")]
    assert svg == plain


# ---------------------------------------------------------------------------
# two flipped breaks on one caller axis that disagree
# ---------------------------------------------------------------------------


def test_conflicting_breaks_on_the_callers_numeric_axis_warn_once_and_nothing_else():
    flipped = _flip(_bars())
    chart = (
        flipped
        + _break("y", (12.0, 15.0), break_style="gap")
        + _break("y", (25.0, 45.0), break_style="slash")
    )

    svg, messages = _svg(chart)
    twin, twin_messages = _svg(
        _bars_transposed()
        + _break("x", (12.0, 15.0), break_style="gap")
        + _break("x", (25.0, 45.0), break_style="slash")
    )

    assert messages == [_CONFLICT.format(axis="y")]
    assert twin_messages == [_CONFLICT.format(axis="x")]
    assert svg == twin


def test_conflicting_breaks_on_the_callers_x_name_x_with_the_ordinal_notice():
    chart = (
        _flip(_bars())
        + _break("x", (12.0, 15.0), break_style="gap")
        + _break("x", (25.0, 45.0), break_style="slash")
    )

    _, messages = _svg(chart)

    assert messages == [_CONFLICT.format(axis="x"), _ORDINAL.format(axis="x")]


# ---------------------------------------------------------------------------
# annotations clip with the marks on the caller's axis
# ---------------------------------------------------------------------------


def test_an_annotation_rect_on_the_callers_broken_axis_clips_into_pieces():
    rect = ann.rect(x1="a", x2="e", y1=0.0, y2=70.0, fill=_RED, opacity=0.5)

    svg, messages = _svg(_flip(_bars()) + _break("y") + rect)

    assert messages == []
    pieces = _red_rects(svg)
    assert len(pieces) == 2, pieces
    left, right = sorted(pieces)
    # The caller's y runs horizontally: the pieces share a row and a height and
    # are separated along x by the break.
    assert left[1] == right[1] and left[3] == right[3]
    assert left[0] + left[2] < right[0]


def test_an_annotation_on_the_callers_broken_axis_matches_the_transposed_chart():
    rect = ann.rect(x1="a", x2="e", y1=0.0, y2=70.0, fill=_RED, opacity=0.5)
    twin_rect = ann.rect(x1=0.0, x2=70.0, y1="a", y2="e", fill=_RED, opacity=0.5)

    svg, _ = _svg(_flip(_bars()) + _break("y") + rect)
    twin, _ = _svg(_bars_transposed() + _break("x") + twin_rect)

    assert svg == twin


def test_an_annotation_wholly_below_the_gap_stays_one_piece():
    rect = ann.rect(x1="b", x2="d", y1=0.0, y2=20.0, fill=_RED, opacity=0.5)

    svg, messages = _svg(_flip(_bars()) + _break("y") + rect)

    assert messages == []
    assert len(_red_rects(svg)) == 1


_TEXT_ROLLUP = (
    "1 annotation element (text) on the {axis} axis fell inside a BreakAxis gap and was suppressed"
)
_BRACKET_ROLLUP = (
    "1 annotation element (bracket) on the {axis} axis spans a BreakAxis gap and was suppressed"
)


def test_a_text_annotation_in_the_gap_is_reported_on_the_callers_axis():
    text = ann.text(x="b", y=35.0, text="t")
    twin_text = ann.text(x=35.0, y="b", text="t")

    _, messages = _svg(_flip(_bars()) + _break("y") + text)
    _, twin_messages = _svg(_bars_transposed() + _break("x") + twin_text)

    assert messages == [_TEXT_ROLLUP.format(axis="y")]
    assert twin_messages == [_TEXT_ROLLUP.format(axis="x")]


def test_a_bracket_spanning_the_gap_is_reported_on_the_callers_axis():
    bracket = ann.bracket(x1="b", y1=20.0, x2="b", y2=50.0, label="br")
    twin_bracket = ann.bracket(x1=20.0, y1="b", x2=50.0, y2="b", label="br")

    _, messages = _svg(_flip(_bars()) + _break("y") + bracket)
    _, twin_messages = _svg(_bars_transposed() + _break("x") + twin_bracket)

    assert messages == [_BRACKET_ROLLUP.format(axis="y")]
    assert twin_messages == [_BRACKET_ROLLUP.format(axis="x")]


# ---------------------------------------------------------------------------
# every route reads the caller's letter: faceted, composite leaf, `+`, `Chart.layer`,
# `LayerChart` members, interactive. `LayerChart` itself takes no `coord` and refuses
# `+ BreakAxis`; its member charts carry the flip and the break.
# ---------------------------------------------------------------------------

_FACET_DF = _BARS.with_columns(pl.Series("g", ["p", "q", "p", "q", "p"]))


def _faceted(x: str, y: str) -> fm.Chart:
    return fm.Chart(_FACET_DF).mark_bar().encode(x=x, y=y).facet(col="g")


def test_a_faceted_flipped_chart_breaks_the_callers_numeric_axis_in_every_panel():
    flipped = _flip(_faceted("c:N", "v:Q"))
    plain, _ = _svg(flipped)

    svg, messages = _svg(flipped + _break("y"))
    twin, twin_messages = _svg(_faceted("v:Q", "c:N") + _break("x"))

    assert messages == []
    assert twin_messages == []
    assert svg != plain
    assert svg == twin


def test_a_faceted_flipped_chart_warns_once_for_the_callers_categorical_axis():
    flipped = _flip(_faceted("c:N", "v:Q"))
    plain, _ = _svg(flipped)

    svg, messages = _svg(flipped + _break("x"))

    assert messages == [_ORDINAL.format(axis="x")]
    assert svg == plain


def test_a_composite_leaf_reads_the_callers_letter():
    flipped_leaf = _flip(_bars()) + _break("y")
    twin_leaf = _bars_transposed() + _break("x")

    svg, messages = _render_capturing((flipped_leaf | _bars()).to_svg)
    twin, twin_messages = _render_capturing((twin_leaf | _bars()).to_svg)

    assert messages == []
    assert twin_messages == []
    assert svg == twin


def test_a_composite_leaf_warns_in_the_callers_letter():
    _, messages = _render_capturing(((_flip(_bars()) + _break("x")) | _bars()).to_svg)

    assert messages == [_ORDINAL.format(axis="x")]


def test_a_layered_flipped_chart_reads_the_callers_letter():
    pts = fm.Chart(_BARS).mark_point().encode(x="c:N", y="v:Q")
    twin_pts = fm.Chart(_BARS).mark_point().encode(x="v:Q", y="c:N")

    svg, messages = _svg(_flip(_bars() + pts) + _break("y"))
    twin, twin_messages = _svg((_bars_transposed() + twin_pts) + _break("x"))

    assert messages == []
    assert twin_messages == []
    assert svg == twin


def _layered(x: str, y: str) -> fm.Chart:
    """Bars with a point layer added through ``Chart.layer``."""
    return fm.Chart(_BARS).encode(x=x, y=y).mark_bar().layer(fm.Layer(mark="point"))


def test_a_chart_layer_flipped_chart_reads_the_callers_letter():
    flipped = _flip(_layered("c:N", "v:Q"))
    plain, _ = _svg(flipped)

    svg, messages = _svg(flipped + _break("y"))
    twin, twin_messages = _svg(_layered("v:Q", "c:N") + _break("x"))

    assert messages == []
    assert twin_messages == []
    assert svg != plain
    assert svg == twin


def test_a_chart_layer_flipped_chart_warns_in_the_callers_letter():
    flipped = _flip(_layered("c:N", "v:Q"))
    plain, _ = _svg(flipped)

    svg, messages = _svg(flipped + _break("x"))

    assert messages == [_ORDINAL.format(axis="x")]
    assert svg == plain


# ---------------------------------------------------------------------------
# every warning the exchange switched names the caller's letter, and breaks on
# both caller axes stay separate
# ---------------------------------------------------------------------------

_BREAK_SIZE_CONFLICT = (
    "multiple BreakAxis specs on the {axis} axis disagree on break_size; the first "
    "declaration's value was kept"
)


def test_conflicting_break_sizes_on_the_callers_axis_name_the_callers_letter():
    svg, messages = _svg(
        _flip(_bars())
        + _break("y", (12.0, 15.0), break_size=12.0)
        + _break("y", (25.0, 45.0), break_size=40.0)
    )
    twin, twin_messages = _svg(
        _bars_transposed()
        + _break("x", (12.0, 15.0), break_size=12.0)
        + _break("x", (25.0, 45.0), break_size=40.0)
    )

    assert messages == [_BREAK_SIZE_CONFLICT.format(axis="y")]
    assert twin_messages == [_BREAK_SIZE_CONFLICT.format(axis="x")]
    assert svg == twin


def _log_points(x: str | fm.X, y: str | fm.Y) -> fm.Chart:
    return fm.Chart(_BARS).mark_point().encode(x=x, y=y)


def _nonlinear_message(axis: str) -> str:
    return (
        f"BreakAxis on the {axis} axis targets a log scale; break placement is computed with "
        "a linear pixel-to-data map, so the removed range and the indicator position can "
        "differ from the gap you asked for, and annotation suppression (which reads the "
        "data value directly) can disagree with where marks are placed"
    )


def test_a_break_on_the_callers_log_axis_warns_in_the_callers_letter():
    log_y = _log_points("c:N", fm.Y("v:Q", scale=fm.LogScale()))
    twin_log_x = _log_points(fm.X("v:Q", scale=fm.LogScale()), "c:N")

    svg, messages = _svg(_flip(log_y) + _break("y"))
    twin, twin_messages = _svg(twin_log_x + _break("x"))

    assert messages == [_nonlinear_message("y")]
    assert twin_messages == [_nonlinear_message("x")]
    assert svg == twin


def test_breaks_on_both_callers_axes_of_a_flipped_numeric_chart_match_the_transposed_chart():
    both = _flip(_points()) + _break("x", _NUM_GAP_A) + _break("y", (150.0, 250.0))
    twin = _points_transposed() + _break("y", _NUM_GAP_A) + _break("x", (150.0, 250.0))

    svg, messages = _svg(both)
    twin_svg, twin_messages = _svg(twin)
    only_x, _ = _svg(_flip(_points()) + _break("x", _NUM_GAP_A))
    only_y, _ = _svg(_flip(_points()) + _break("y", (150.0, 250.0)))

    assert messages == []
    assert twin_messages == []
    assert svg == twin_svg
    assert svg != only_x
    assert svg != only_y


def _layer_chart(flipped: bool) -> fm.LayerChart:
    """A ``LayerChart`` of bars and points whose members carry the flip and the break.

    The flipped form breaks the caller's y; the transposed form binds the same data the
    other way round and breaks its x.
    """
    if flipped:
        bars = _flip(_bars()) + _break("y")
        points = _flip(fm.Chart(_BARS).mark_point().encode(x="c:N", y="v:Q"))
    else:
        bars = _bars_transposed() + _break("x")
        points = fm.Chart(_BARS).mark_point().encode(x="v:Q", y="c:N")
    return fm.LayerChart(bars, points)


def test_a_layer_chart_member_reads_the_callers_letter_on_both_routes():
    svg, messages = _render_capturing(_layer_chart(True).to_svg)
    twin, twin_messages = _render_capturing(_layer_chart(False).to_svg)
    page, page_messages = _interactive_page(_layer_chart(True))
    unbroken = fm.LayerChart(
        _flip(_bars()), _flip(fm.Chart(_BARS).mark_point().encode(x="c:N", y="v:Q"))
    )
    plain, _ = _render_capturing(unbroken.to_svg)

    assert messages == []
    assert twin_messages == []
    assert page_messages == []
    assert svg != plain
    assert svg == twin
    assert page


def test_a_layer_chart_member_warns_in_the_callers_letter_on_both_routes():
    points = _flip(fm.Chart(_BARS).mark_point().encode(x="c:N", y="v:Q"))
    chart = fm.LayerChart(_flip(_bars()) + _break("x"), points)

    plain, _ = _render_capturing(
        fm.LayerChart(
            _flip(_bars()), _flip(fm.Chart(_BARS).mark_point().encode(x="c:N", y="v:Q"))
        ).to_svg
    )

    svg, messages = _render_capturing(chart.to_svg)
    _, page_messages = _interactive_page(chart)

    assert messages == [_ORDINAL.format(axis="x")]
    assert page_messages == [_ORDINAL.format(axis="x")]
    assert svg == plain


def test_an_independent_y_layer_chart_refuses_the_flip_a_break_would_ride_on():
    """A break cannot be attached to an independent-y layer under a flip: the flip is refused."""
    points = _flip(fm.Chart(_BARS).mark_point().encode(x="c:N", y="v:Q"))
    chart = fm.LayerChart(_flip(_bars()) + _break("y"), points, resolve={"y": "independent"})

    refusal = (
        "SecondaryY (a layer with an independent y scale) is not supported under fm.CoordFlip: "
        "its right y axis has no place once the coord moves y off the vertical axis; draw the "
        "series on a Cartesian panel (put the categories on x and drop the coord), or draw it "
        "on the shared y axis"
    )
    with pytest.raises(ValueError, match=re.escape(refusal)):
        chart.to_svg()


def _interactive_page(chart: fm.Chart) -> tuple[str, list[str]]:
    return _render_capturing(lambda: chart.interactive().to_html())


def test_the_interactive_route_breaks_the_callers_numeric_axis_with_no_warning():
    flipped = _flip(_bars())
    plain, _ = _interactive_page(flipped)

    page, messages = _interactive_page(flipped + _break("y"))

    assert messages == []
    assert page != plain


def test_the_interactive_route_warns_for_the_callers_categorical_axis_as_static_does():
    flipped = _flip(_bars())
    _, static_messages = _svg(flipped + _break("x"))

    _, messages = _interactive_page(flipped + _break("x"))

    assert static_messages == [_ORDINAL.format(axis="x")]
    assert messages == static_messages


# ---------------------------------------------------------------------------
# unchanged: a figure-applied flip and an unflipped chart
# ---------------------------------------------------------------------------
# Digests measured on the build before the break routing changed.


def test_catplot_box_with_a_horizontal_orient_keeps_breaking_its_drawn_numeric_axis():
    chart = fm.catplot(_BARS, x="v", y="c", kind="box", orient="h")

    svg, messages = _svg(chart + _break("x"))

    assert messages == []
    assert _digest(svg) == "e9e9069513e82eb83d0a9fab74d0bb886c73adb0f7723bc56c27d6bc39ef486a"


def test_catplot_strip_with_a_horizontal_orient_keeps_breaking_its_drawn_numeric_axis():
    chart = fm.catplot(_BARS, x="v", y="c", kind="strip", orient="h")

    svg, messages = _svg(chart + _break("x"))

    assert messages == []
    assert _digest(svg) == "a68eff4ac28ebab57620b29427a414c73ef3c2679d6ac38f14889d42a6739895"


def test_an_unflipped_chart_breaks_the_axis_it_names():
    svg, messages = _svg(_bars() + _break("y"))

    assert messages == []
    assert _digest(svg) == "aafaaacfb3fd68eab961c657607af58e3569942a598dd8026bcfac41282f3369"
