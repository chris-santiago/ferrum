"""End-to-end coverage for the one rule deciding whether a layer draws a non-positional channel.

A size, opacity or color a layer's mark cannot draw must never widen that
channel's chart-level scale, never build a legend from it, and must warn in
the same words regardless of which door reached it: a flat chart, a ``+``
composition (either operand order), an attached ``Chart.layer``, or a
``fm.LayerChart``, rendered statically (each member its own overlay leaf) or
through ``.interactive()`` (the merged scene). Each test names the one wrong
behavior it would catch: a channel judged by its own per-site rule, with its
own answer for whether a mark draws it, instead of the one rule every
reader shares.

Every legend is read off the output the door in question actually renders:
``legend_set`` on the SVG a static route (``to_svg()``) produced,
``scene_legend_set`` on the scene an ``.interactive()`` route built. Static
and interactive ``fm.LayerChart`` render through genuinely different paths
(the static overlay tree versus the merged flat chart), so a legend read
from the wrong one would silently miss a static-only regression. Every test
below renders its chart exactly once inside its warnings capture and reads
the legend from that same artifact, never a second render, so a warning the
render raised is never duplicated or left to leak past the test.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum._layer import _Layer
from tests._legends import legend_set, scene_legend_set


def _rendered(build: Callable[[], str]) -> tuple[list[str], str]:
    """Runs ``build()`` (a chart's ``to_svg``) once inside one warnings capture.

    Returns the captured warning messages together with the rendered SVG,
    so a legend assertion reads the exact artifact that warned instead of
    rendering the chart a second time outside the capture -- a second
    render would both duplicate the warning and let it leak past this
    test, visible as a stray ``UserWarning`` in the pytest summary or,
    under ``-W error::UserWarning``, as a failure of an unrelated
    assertion.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = build()
    return [str(w.message) for w in caught], svg


def _rendered_door(
    chart: fm.Chart | fm.LayerChart, kind: str
) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    """A door's one render -- ``to_svg`` for a static door, the interactive scene otherwise.

    Captures the warnings that render raises and reads the legend off that
    same render, never a second one.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if kind == "static":
            legend = legend_set(chart.to_svg())
        else:
            legend = scene_legend_set(json.loads(chart.interactive().scene_json))
    return [str(w.message) for w in caught], legend


# ---------------------------------------------------------------------------
# A line's own stroke_width outranks an inherited size: the line never
# establishes or widens the chart-level size scale, on either `+` order, and
# the same precedence holds whether the line owns the winning width or only
# inherits it.
# ---------------------------------------------------------------------------

LINE_WIDE_DF = pl.DataFrame(
    {"v": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "s": [100.0, 200.0, 300.0]}
)
LINE_WIDE_DF_2 = pl.DataFrame({"v": [1.0, 2.0, 3.0], "y": [3.0, 1.0, 2.0], "s": [5.0, 6.0, 7.0]})
POINT_NARROW_DF = pl.DataFrame({"v": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 1.0], "s": [1.0, 2.0, 3.0]})
# The point's own size legend, pinned literally: the discriminating value
# both `+` orders and the `Chart.layer` attach below must match exactly, not
# merely "whatever a separately built reference chart happens to draw" (a
# regression that deleted every size legend would make both sides {} too).
POINT_SIZE_LEGEND = {"s": ("1", "1.5", "2", "2.5", "3")}


def _wide_line(**encoding: Any) -> fm.Chart:
    """A line whose own ``stroke_width`` outranks any inherited ``size``."""
    return fm.Chart(LINE_WIDE_DF).mark_line(stroke_width=3).encode(x="v", y="y", **encoding)


def _second_wide_line(**encoding: Any) -> fm.Chart:
    """A second, differently width-owning line, over a different frame."""
    return fm.Chart(LINE_WIDE_DF_2).mark_line(stroke_width=5).encode(x="v", y="y", **encoding)


def _narrow_point(**encoding: Any) -> fm.Chart:
    return fm.Chart(POINT_NARROW_DF).mark_point().encode(x="v", y="y", **encoding)


@pytest.mark.parametrize(
    "build",
    [
        lambda: (_wide_line() + _narrow_point()).encode(size="s"),
        lambda: (_narrow_point() + _wide_line()).encode(size="s"),
    ],
    ids=["line-first", "point-first"],
)
def test_a_yielding_lines_size_never_widens_the_chart_level_scale(build):
    """Rejects a size legend spanning the line's 100..300 beside the point's 1..3.

    Both ``+`` orders draw the pinned point-only size legend, never the
    line's wide values: the line's own ``stroke_width`` means it never reads
    into the chart-level size scale's domain, and nothing warns since the
    point still draws it.
    """
    chart = build()
    messages, svg = _rendered(chart.to_svg)
    assert messages == []
    assert legend_set(svg) == POINT_SIZE_LEGEND


def test_an_attached_lines_own_width_does_not_widen_the_hosts_size_domain():
    """Rejects a `Chart.layer` attach reaching the same leak through a different door.

    The attached line's own ``stroke_width`` means it never reads into the
    host's inherited ``size`` scale either, so the legend stays the pinned
    point-only one.
    """
    host = _narrow_point(size="s")
    chart = host.layer(fm.Layer(mark="line", data=LINE_WIDE_DF, mark_kwargs={"stroke_width": 3}))
    messages, svg = _rendered(chart.to_svg)
    assert messages == []
    assert legend_set(svg) == POINT_SIZE_LEGEND


INHERITED_WIDTH_LINE_DF = pl.DataFrame(
    {"v": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "s": [10.0, 20.0, 30.0], "w": [2.0, 3.0, 4.0]}
)
INHERITED_WIDTH_LINE_DF_2 = pl.DataFrame(
    {"v": [1.0, 2.0, 3.0], "y": [3.0, 1.0, 2.0], "s": [40.0, 50.0, 60.0], "w": [5.0, 6.0, 7.0]}
)


def _plain_line(data: pl.DataFrame, **encoding: Any) -> fm.Chart:
    return fm.Chart(data).mark_line().encode(x="v", y="y", **encoding)


# Two voices for the same precedence rule: a line whose own stroke_width
# kwarg outranks an inherited size names that kwarg; two lines that merely
# inherit both widths (neither owns either) name the inherited field
# instead, since there is no owned kwarg to name. Each case's expected
# legend is pinned exactly: own-width draws no legend at all (no mark
# anywhere binds stroke_width as a channel); inherited-width's own
# stroke_width legend is drawn normally and unaffected -- only size is
# excluded from it.
_EVERY_LAYER_YIELDS: dict[str, tuple[Callable[[], fm.Chart], str, dict[str, tuple[str, ...]]]] = {
    "own-width": (
        lambda: (_wide_line() + _second_wide_line()).encode(size="s"),
        "layer 0: mark_line: size 's' has no effect: the line draws its own width "
        "(stroke_width=3), which outranks an inherited size, so size gets no scale "
        "and no legend; it was ignored",
        {},
    ),
    "inherited-width": (
        lambda: (
            _plain_line(INHERITED_WIDTH_LINE_DF) + _plain_line(INHERITED_WIDTH_LINE_DF_2)
        ).encode(size="s", stroke_width="w"),
        "layer 0: mark_line: size 's' has no effect: the line draws the width it also "
        "inherits (stroke_width='w'), which outranks an inherited size, so size gets "
        "no scale and no legend; it was ignored",
        {"w": ("2", "3", "4", "5", "6", "7")},
    ),
}


@pytest.mark.parametrize("case", sorted(_EVERY_LAYER_YIELDS))
def test_every_width_owning_layer_yielding_size_warns_once_and_builds_no_scale(case):
    """Rejects a size scale or legend built from rows no mark actually sizes by.

    Two lines, each yielding size to a width (its own kwarg, or one it only
    inherits), both yield: neither reads into the scale, so the chart-level
    binding is reported ignored exactly once, naming whichever width reason
    applies. The pinned legend is a guard, not the rejection: it is the
    same with or without this rule (no mark ever painted by size here), so
    the warning text above is what a regression would actually break.
    """
    build, message, expected_legend = _EVERY_LAYER_YIELDS[case]
    chart = build()
    messages, svg = _rendered(chart.to_svg)
    assert messages == [message]
    assert legend_set(svg) == expected_legend


# ---------------------------------------------------------------------------
# An arrowless segment fills nothing: a bound fill_opacity never builds a
# scale or legend, and warns once naming the arrowhead rule, on every door.
# An arrow=True sibling on each same door is the discriminating pair: it
# proves that door's pipeline can draw this legend, so the arrowless door's
# empty legend means something was actually suppressed, not that the route
# never draws a fill_opacity legend at all.
# ---------------------------------------------------------------------------

SEGMENT_DF = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 1.5],
        "x2": [1.5, 2.5, 3.5],
        "y2": [1.2, 2.2, 1.8],
        "o": [0.2, 0.5, 0.9],
    }
)
OTHER_POINT_DF = pl.DataFrame({"x": [4.0, 5.0], "y": [2.0, 3.0]})

_ARROWLESS_FILL_OPACITY = (
    "{entry}: fill_opacity 'o' has no effect: a segment fills only the arrowheads "
    "arrow=True adds, so fill_opacity gets no scale and no legend; it was ignored"
)
# The legend arrow=True draws, pinned literally and shared by every door's
# control below.
ARROWED_FILL_OPACITY_LEGEND = {"o": ("0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9")}


def _segment(*, arrow: bool = False, **encoding: Any) -> fm.Chart:
    style = {"arrow": True} if arrow else {}
    return (
        fm.Chart(SEGMENT_DF)
        .mark_segment(**style)
        .encode(x="x", y="y", x2="x2", y2="y2", **encoding)
    )


def _other_point(**encoding: Any) -> fm.Chart:
    return fm.Chart(OTHER_POINT_DF).mark_point().encode(x="x", y="y", **encoding)


# Each door: how to build the arrowless chart, which output it renders, and
# the exact warning it must raise. The entry a layer-voiced message names
# differs only by door: an unlayered chart or a static overlay member (this
# door's overlay-unlayered-member case, an arrowless segment beside a point)
# name the mark alone; a `+` layer list, or the merged scene the interactive
# `LayerChart` renders through, name the layer index.
_ARROWLESS_DOORS: dict[str, tuple[Callable[[], fm.Chart | fm.LayerChart], str, str]] = {
    "flat": (
        lambda: _segment(fill_opacity="o"),
        "static",
        _ARROWLESS_FILL_OPACITY.format(entry="mark_segment"),
    ),
    "+": (
        lambda: _segment(fill_opacity="o") + _other_point(),
        "static",
        _ARROWLESS_FILL_OPACITY.format(entry="layer 0: mark_segment"),
    ),
    "LayerChart-static": (
        lambda: fm.LayerChart(_segment(fill_opacity="o"), _other_point()),
        "static",
        _ARROWLESS_FILL_OPACITY.format(entry="mark_segment"),
    ),
    "LayerChart-interactive": (
        lambda: fm.LayerChart(_segment(fill_opacity="o"), _other_point()),
        "interactive",
        _ARROWLESS_FILL_OPACITY.format(entry="layer 0: mark_segment"),
    ),
}

# The same doors, built with arrow=True instead: the control proving each
# door's legend machinery draws when the mark actually paints by the bound
# channel, so the arrowless door's "no legend" above is a real suppression
# rather than a route that never draws any fill_opacity legend.
_ARROW_HEADED_CONTROLS: dict[str, tuple[Callable[[], fm.Chart | fm.LayerChart], str]] = {
    "flat": (lambda: _segment(arrow=True, fill_opacity="o"), "static"),
    "+": (lambda: _segment(arrow=True, fill_opacity="o") + _other_point(), "static"),
    "LayerChart-static": (
        lambda: fm.LayerChart(_segment(arrow=True, fill_opacity="o"), _other_point()),
        "static",
    ),
    "LayerChart-interactive": (
        lambda: fm.LayerChart(_segment(arrow=True, fill_opacity="o"), _other_point()),
        "interactive",
    ),
}


@pytest.mark.parametrize("door", sorted(_ARROWLESS_DOORS))
def test_an_arrowless_segments_fill_opacity_warns_instead_of_drawing_on_every_door(door):
    """Rejects a silently built fill_opacity scale or legend for a mark that paints nothing with it.

    The warning text is the rejection: a door that built the fill_opacity
    scale silently instead of warning would show no message here. The
    empty legend is a guard -- on its own it would pass on a regression
    that dropped fill_opacity legends everywhere -- made meaningful by this
    door's arrow=True control below, which draws the same legend on the
    same route.
    """
    build, kind, message = _ARROWLESS_DOORS[door]
    chart = build()
    messages, legend = _rendered_door(chart, kind)
    assert messages == [message]
    assert legend == {}


@pytest.mark.parametrize("door", sorted(_ARROW_HEADED_CONTROLS))
def test_an_arrow_headed_segments_fill_opacity_draws_on_every_door(door):
    """Control: ``arrow=True`` draws the arrowheads, so ``fill_opacity`` still builds its legend.

    Rejects the drawn-channel rule over-reaching into the drawing case it
    must leave alone, and gives the matching arrowless door's empty legend
    something to be empty in contrast to.
    """
    build, kind = _ARROW_HEADED_CONTROLS[door]
    chart = build()
    messages, legend = _rendered_door(chart, kind)
    assert messages == []
    assert legend == ARROWED_FILL_OPACITY_LEGEND


# ---------------------------------------------------------------------------
# A conditional branch on a channel a layer's mark does not draw is never
# read, column-validated or otherwise.
# ---------------------------------------------------------------------------

CONDITIONAL_SEGMENT_DF = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 1.5],
        "x2": [1.5, 2.5, 3.5],
        "y2": [1.2, 2.2, 1.8],
    }
)
# The conditional names this column. It is present on the attached point
# layer but never on the segment's own rows, so a read that reaches the
# segment's layer raises "unknown column" there -- a clean render proves the
# field was never read for that layer.
CONDITIONAL_POINT_DF = pl.DataFrame({"x": [4.0, 5.0], "y": [2.0, 3.0], "unseen": [0.1, 0.2]})


def _conditional_fill_opacity_chart(*, arrow: bool) -> fm.Chart:
    pick = fm.selection_point(fields=["x"], name="pick")
    conditional = pick.when(fm.FillOpacity("unseen")).otherwise(fm.value(0.5))
    style = {"arrow": True} if arrow else {}
    host = (
        fm.Chart(CONDITIONAL_SEGMENT_DF)
        .mark_segment(**style)
        .encode(x="x", y="y", x2="x2", y2="y2", fill_opacity=conditional)
        .add_selection(pick)
    )
    return host.layer(
        fm.Layer(mark="point", data=CONDITIONAL_POINT_DF, encoding={"x": "x", "y": "y"})
    )


def test_a_conditional_fill_opacity_on_an_arrowless_segment_is_not_a_column_read():
    """Rejects validating a conditional branch's field against a layer that never reads it.

    The segment's own rows lack ``unseen``: arrowless, the conditional's
    field is never read, so the chart renders. With ``arrow=True`` (the
    control, same field, same layer), the segment does read it, and the
    layer is refused for the missing column -- proving the field really
    would be invalid if the arrowless case read it too.
    """
    assert _conditional_fill_opacity_chart(arrow=False).to_svg()
    with pytest.raises(ValueError) as refused:
        _conditional_fill_opacity_chart(arrow=True).to_svg()
    assert str(refused.value) == (
        "layer 0: encoding 'fill_opacity' references unknown column 'unseen' "
        "(not present in that layer's data)"
    )


# ---------------------------------------------------------------------------
# Controls: a contrast-ink text, a raster's own-field cell color, a text
# declining an inherited color, a literal-paint layer's inherited color, and
# a line with its own size and no own width -- each warns, scales or draws
# its legend by its own site's rule, independent of the drawn-channel rule
# above. Each warning control also asserts the chart renders
# byte-identically to the same chart with the ignored binding removed, so
# "unchanged" is checked in output, not only in warning text.
# ---------------------------------------------------------------------------

CONTRAST_CELL_DF = pl.DataFrame(
    {"row": ["p0", "p1"], "col": ["q0", "q0"], "value": [0.0, 10.0], "label": ["L0", "L1"]}
)


def _contrast_chart(*, with_color: bool) -> fm.Chart:
    base = fm.Chart(CONTRAST_CELL_DF).mark_rect().encode(x="col", y="row")
    encoding: dict[str, str] = {"x": "col", "y": "row", "text": "label"}
    if with_color:
        encoding["color"] = "value"
    # The "theme:contrast" fill is engine-internal, reached by constructing
    # the layer directly rather than through the public mark_text(fill=...)
    # door, which refuses it like any other unrecognized paint string.
    ink_text = _Layer(
        mark="text", encoding=encoding, mark_kwargs={"fill": "theme:contrast"}, data_source=None
    )
    return base.layer(ink_text)


def test_a_contrast_ink_text_color_warning_stays_byte_identical():
    """Control: an ink-only text color warns the engine's contrast-ink warning text, unchanged.

    The chart also renders byte-identically to the same chart with the
    ignored color binding removed, so "unchanged" covers the drawn output,
    not only the warning text.
    """
    chart = _contrast_chart(with_color=True)
    messages, svg = _rendered(chart.to_svg)
    assert messages == [
        "layer 1: mark_text: color 'value' has no effect: the labels read color only to "
        "pick an ink against the cells under them, no layer draws that color, so it gets "
        "no scale and no legend; each label is judged against the theme mark color, the "
        "fill an unstyled cell paints"
    ]
    assert svg == _contrast_chart(with_color=False).to_svg()


RASTER_ROWS = pl.DataFrame(
    {
        "x": [float(i % 10) for i in range(60)],
        "y": [float(i // 10) for i in range(60)],
        "f": ["m", "n"] * 30,
    }
)


def _raster_chart(*, with_color: bool) -> fm.Chart:
    encoding: dict[str, str] = {"x": "x", "y": "y"}
    if with_color:
        encoding["color"] = "f"
    return fm.Chart(RASTER_ROWS).mark_raster().encode(**encoding)


def test_a_rasters_cell_color_warning_stays_byte_identical():
    """Control: color binding a raster to a field other than its cell value warns unchanged.

    The chart also renders byte-identically to the same chart with the
    ignored color binding removed.
    """
    chart = _raster_chart(with_color=True)
    messages, svg = _rendered(chart.to_svg)
    assert messages == [
        "mark_raster: color 'f' has no effect: the image colors its cells by their 'count' "
        "values (field=) and does not read color='f', so it gets no scale and no legend; "
        "it was ignored"
    ]
    assert svg == _raster_chart(with_color=False).to_svg()


DECLINED_TEXT_DF = pl.DataFrame(
    {"g": ["a", "b", "a", "b"], "y": [1.0, 2.0, 3.0, 4.0], "c": ["p", "q", "p", "q"]}
)


def _declined_text_chart(*, with_color: bool) -> fm.Chart:
    def text(field: str) -> fm.Chart:
        return fm.Chart(DECLINED_TEXT_DF).mark_text().encode(x="g", y="y", text=field)

    composed = text("y") + text("c")
    return composed.encode(color="c") if with_color else composed


def test_a_text_declining_an_inherited_color_warning_stays_byte_identical():
    """Control: a text that draws only a color it binds itself warns unchanged when it inherits one.

    The chart also renders byte-identically to the same composition with
    the ignored color binding removed.
    """
    chart = _declined_text_chart(with_color=True)
    messages, svg = _rendered(chart.to_svg)
    assert messages == [
        "layer 0: mark_text: color 'c' has no effect: no layer's mark draws color (a text "
        "mark draws only a color it binds itself), so it gets no scale and no legend; it "
        "was ignored"
    ]
    assert svg == _declined_text_chart(with_color=False).to_svg()


PAINT_DF = pl.DataFrame({"v": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "g": ["a", "b", "a"]})
# The attached line's own frame, carrying a category ("c") the host lacks:
# the union legend only lists it if the line's inherited color really reads
# into the chart-level scale.
PAINT_LINE_DF = pl.DataFrame({"v": [4.0, 5.0], "y": [4.0, 5.0], "g": ["c", "c"]})


def test_a_literal_paint_layer_inheriting_color_still_reads_into_the_scale():
    """Control: a line's own literal stroke still reads its inherited color into the chart's scale.

    Rejects excluding a layer from the color domain just because it paints a
    literal color instead of the mapped one: that per-row paint rule is
    untouched by the drawn-channel rule. The line's own frame carries a
    category ('c') the host alone never draws, so the union legend ('a',
    'b', 'c') appears only if the line's color really reads in; the host
    alone, pinned beside it, draws only ('a', 'b').
    """
    host = fm.Chart(PAINT_DF).mark_point().encode(x="v", y="y", color="g")
    chart = host.layer(fm.Layer(mark="line", data=PAINT_LINE_DF, mark_kwargs={"stroke": "red"}))
    messages, svg = _rendered(chart.to_svg)
    assert messages == []
    assert legend_set(svg) == {"g": ("a", "b", "c")}
    assert legend_set(host.to_svg()) == {"g": ("a", "b")}


OWN_SIZE_DF = pl.DataFrame({"v": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "s": [10.0, 20.0, 30.0]})


def test_a_lines_own_size_with_no_own_width_still_draws_and_establishes():
    """Control: a line with its own size binding and no stroke_width still draws and scales by it."""
    chart = fm.Chart(OWN_SIZE_DF).mark_line().encode(x="v", y="y", size="s")
    messages, svg = _rendered(chart.to_svg)
    assert messages == []
    assert legend_set(svg) == {"s": ("10", "15", "20", "25", "30")}
