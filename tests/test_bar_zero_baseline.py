"""Bars and areas start at zero.

Whenever 0 is in a bar's value domain, every bar grows from the 0 tick: mixed
sign, all negative, horizontal (native, ``orient="horizontal"``, ``CoordFlip``),
reversed, dodged, overlapping, histogram, layered and ``SecondaryY`` bars alike,
and an area closes on that tick. When 0 is outside the domain (log, ``zero=False``,
an explicit non-zero domain) the base is the plot edge on zero's side.

Stacks split by sign: positive values accumulate up from 0 and negative values
down from 0, each in group order. With ``P`` the positive sum and ``N`` the
absolute negative sum of one stack, ``zero`` spans ``[-N, P]``, ``normalize``
divides every segment by ``P + N`` and ``center`` shifts the stack by
``(N - P) / 2``. A zero-value bar is a zero-height rect at its base.

Every assertion reads the rendered geometry against the value axis's own tick
ladder, so a 0 tick placed by the axis and a base drawn by the bar builder are
compared directly.
"""

from __future__ import annotations

import datetime
import json
from typing import NamedTuple

import polars as pl
import pyarrow as pa
import pytest

import ferrum as fm
from ferrum._spec_build import _column_reads_categorical
from ferrum.layer import Layer

from _svg_corpus import path_points
from _svg_extents import pixel_for, value_axis_ladder, value_axis_ladders
from _svg_rects import data_bar_rects, mark_layer, plot_rect, svg_elements

# Pixel coordinates are emitted to 3 decimals; the 0.05 tolerance is the
# acceptance bound for "the base sits on the 0 tick".
TOL_PX = 0.05

C = fm.Chart
MIXED = pl.DataFrame({"x": ["a", "b", "c"], "y": [3.0, -2.0, 1.0]})
POSITIVE = pl.DataFrame({"x": ["a", "b", "c"], "y": [3.0, 2.0, 1.0]})
NEGATIVE = pl.DataFrame({"x": ["a", "b", "c"], "y": [-3.0, -2.0, -1.0]})
HIGH = pl.DataFrame({"x": ["a", "b", "c"], "y": [50.0, 60.0, 70.0]})
QUANT_MIXED = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, -2.0, 1.0]})
GROUPED_MIXED = pl.DataFrame(
    {"x": ["a", "a", "b", "b"], "g": ["p", "q", "p", "q"], "y": [2.0, -1.0, -2.0, 3.0]}
)
GROUPED_POSITIVE = pl.DataFrame(
    {"x": ["a", "a", "b", "b"], "g": ["p", "q", "p", "q"], "y": [2.0, 1.0, 3.0, 4.0]}
)
QUANT_GROUPED_MIXED = pl.DataFrame(
    {"x": [1.0, 1.0, 2.0, 2.0], "g": ["p", "q", "p", "q"], "y": [2.0, -1.0, -2.0, 3.0]}
)
SAMPLE = pl.DataFrame({"v": [1.0, 1.5, 2.0, 2.2, 3.0, 3.1, 3.2, 4.0]})


class Bar(NamedTuple):
    x: float
    y: float
    w: float
    h: float

    def span(self, axis: str) -> tuple[float, float]:
        """The rect's two edges along the value ``axis``, low pixel first."""
        return (self.y, self.y + self.h) if axis == "y" else (self.x, self.x + self.w)


def render(chart) -> str:
    return chart.properties(width=300, height=200).to_svg()


def bars(svg: str) -> list[Bar]:
    """Every data rect inside the plot area, in document order."""
    return [
        Bar(float(a["x"]), float(a["y"]), float(a["width"]), float(a["height"]))
        for a in data_bar_rects(mark_layer(svg))
    ]


def zero_px(svg: str, axis: str) -> float:
    ladder = value_axis_ladder(svg, axis=axis)
    assert 0.0 in ladder, f"the {axis} axis draws no 0 tick: {sorted(ladder)}"
    return ladder[0.0]


def value_at(ladder: dict[float, float], px: float) -> float:
    """The data value a linear axis with this tick ladder places at ``px``."""
    lo, hi = min(ladder), max(ladder)
    return lo + (px - ladder[lo]) * (hi - lo) / (ladder[hi] - ladder[lo])


def assert_grow_from_zero(svg: str, axis: str, *, n_positive: int, n_negative: int) -> None:
    """Every bar has one edge on the 0 tick; its far edge lies on its value's side.

    A positive value draws up (y) or right (x) of the 0 tick, a negative one
    down or left; ``n_positive``/``n_negative`` count the bars on each side.
    """
    z = zero_px(svg, axis)
    rects = bars(svg)
    assert len(rects) == n_positive + n_negative, rects
    positive_side = 0
    for rect in rects:
        lo, hi = rect.span(axis)
        assert min(abs(lo - z), abs(hi - z)) <= TOL_PX, f"{rect} does not start at the 0 tick {z}"
        far = lo if abs(hi - z) <= TOL_PX else hi
        positive_side += (far < z) if axis == "y" else (far > z)
    assert positive_side == n_positive, f"{positive_side} bars on the positive side of 0: {rects}"


VERTICAL = {
    "ordinal mixed": (C(MIXED).mark_bar().encode(x="x", y="y"), 2, 1),
    "ordinal all positive": (C(POSITIVE).mark_bar().encode(x="x", y="y"), 3, 0),
    "ordinal all negative": (C(NEGATIVE).mark_bar().encode(x="x", y="y"), 0, 3),
    "quantitative x": (C(QUANT_MIXED).mark_bar().encode(x="x", y="y"), 2, 1),
    "histogram": (C(SAMPLE).mark_histogram().encode(x="v"), None, 0),
    "dodged": (
        C(GROUPED_MIXED).mark_bar(position=fm.Dodge()).encode(x="x", y="y", color="g"),
        2,
        2,
    ),
    "overlapping colored": (
        C(GROUPED_MIXED).mark_bar().encode(x="x", y="y", color="g"),
        2,
        2,
    ),
    "bar + point": (
        C(MIXED).mark_bar().encode(x="x", y="y") + C(MIXED).mark_point().encode(x="x", y="y"),
        2,
        1,
    ),
    "explicit domain [-5, 5]": (
        C(MIXED).mark_bar().encode(x="x", y=fm.Y("y", scale={"domain": [-5, 5]})),
        2,
        1,
    ),
    "symlog": (C(MIXED).mark_bar().encode(x="x", y=fm.Y("y", scale={"type": "symlog"})), 2, 1),
    "Diverging positional": (
        C(MIXED)
        .mark_bar()
        .encode(x="x", y=fm.Y("y", scale={"type": "diverging", "domain": [-4, 0, 4]})),
        2,
        1,
    ),
}

HORIZONTAL = {
    "native horizontal mixed": (C(MIXED).mark_bar().encode(y="x:N", x="y"), 2, 1),
    "native horizontal all positive": (C(POSITIVE).mark_bar().encode(y="x:N", x="y"), 3, 0),
    'orient="horizontal" mixed': (
        C(MIXED).mark_bar(orient="horizontal").encode(x="x", y="y"),
        2,
        1,
    ),
    'orient="horizontal" all positive': (
        C(POSITIVE).mark_bar(orient="horizontal").encode(x="x", y="y"),
        3,
        0,
    ),
    "CoordFlip mixed": (
        C(MIXED).mark_bar().encode(x="x", y="y").coord(fm.CoordFlip()),
        2,
        1,
    ),
    "CoordFlip all positive": (
        C(POSITIVE).mark_bar().encode(x="x", y="y").coord(fm.CoordFlip()),
        3,
        0,
    ),
}


@pytest.mark.parametrize("name", list(VERTICAL))
def test_vertical_bars_grow_from_the_zero_tick(name):
    chart, n_positive, n_negative = VERTICAL[name]
    svg = render(chart)
    if n_positive is None:
        n_positive = len(bars(svg))
        assert n_positive > 0
    assert_grow_from_zero(svg, "y", n_positive=n_positive, n_negative=n_negative)


@pytest.mark.parametrize("name", list(HORIZONTAL))
def test_horizontal_bars_grow_from_the_zero_tick(name):
    chart, n_positive, n_negative = HORIZONTAL[name]
    assert_grow_from_zero(render(chart), "x", n_positive=n_positive, n_negative=n_negative)


def test_a_native_horizontal_all_negative_bar_hangs_left_from_zero():
    """Its x ticks stop short of 0 (-3 .. -0.2), so 0's pixel is read off the ladder."""
    svg = render(C(NEGATIVE).mark_bar().encode(y="x:N", x="y"))
    ladder = value_axis_ladder(svg, axis="x")
    z = pixel_for(ladder, 0.0)
    px, _, pw, _ = plot_rect(svg)
    assert z < px + pw - 1.0, "0 sits on the plot edge; the inset beside it is gone"
    rects = bars(svg)
    assert len(rects) == 3
    assert all(abs(r.x + r.w - z) <= TOL_PX for r in rects)
    assert all(r.x < z - 1.0 for r in rects)


def test_a_negative_bar_hangs_from_the_zero_tick_not_from_the_plot_bottom():
    """The "b" bar (-2) spans the 0 tick to -2's pixel.

    Moving the padding inset alone would still draw it from the plot bottom up
    to -2's pixel, so both its edges are pinned.
    """
    svg = render(C(MIXED).mark_bar().encode(x="x", y="y"))
    ladder = value_axis_ladder(svg, axis="y")
    b = sorted(bars(svg), key=lambda r: r.x)[1]
    assert abs(b.y - ladder[0.0]) <= TOL_PX
    assert abs(b.y + b.h - pixel_for(ladder, -2.0)) <= TOL_PX
    _, py, _, ph = plot_rect(svg)
    assert b.y + b.h < py + ph - 1.0, "the negative bar reaches the plot bottom"


def test_an_all_positive_bar_keeps_the_padding_inset_below_its_zero_tick():
    svg = render(C(POSITIVE).mark_bar().encode(x="x", y="y"))
    z = zero_px(svg, "y")
    _, py, _, ph = plot_rect(svg)
    assert py + ph - z > 1.0, "the 0 tick sits on the plot edge; the inset below it is gone"
    assert all(abs(r.y + r.h - z) <= TOL_PX for r in bars(svg))


def test_a_reversed_axis_hangs_bars_from_the_zero_tick_at_the_top():
    svg = render(C(POSITIVE).mark_bar().encode(x="x", y=fm.Y("y", scale={"reverse": True})))
    z = zero_px(svg, "y")
    _, py, _, _ = plot_rect(svg)
    assert z < py + 20.0, "a reversed axis puts 0 near the top"
    rects = sorted(bars(svg), key=lambda r: r.x)
    assert all(abs(r.y - z) <= TOL_PX for r in rects)
    heights = [r.h for r in rects]
    assert heights[0] == pytest.approx(3 * heights[2], abs=TOL_PX)
    assert heights[1] == pytest.approx(2 * heights[2], abs=TOL_PX)


def test_a_mixed_sign_area_closes_on_the_zero_tick():
    data = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [3.0, -2.0, 1.0, 2.0]})
    svg = render(C(data).mark_area().encode(x="x", y="y"))
    z = zero_px(svg, "y")
    [path] = svg_elements(
        mark_layer(svg), "path", lambda a: "d" in a and a.get("fill") not in (None, "none")
    )
    points = path_points(path["d"])
    closing = [p for p in points if abs(p[1] - z) <= TOL_PX]
    xs = sorted({round(p[0], 3) for p in points})
    assert {round(p[0], 3) for p in closing} >= {xs[0], xs[-1]}, (
        f"the fill does not close on the 0 tick {z}: {points}"
    )


# -- the plot edge on zero's side when 0 is outside the domain ---------------

EDGE_CONTROLS = {
    "log": C(pl.DataFrame({"x": ["a", "b", "c"], "y": [1.0, 10.0, 100.0]}))
    .mark_bar()
    .encode(x="x", y=fm.Y("y", scale={"type": "log"})),
    "zero=False": C(HIGH).mark_bar(zero=False).encode(x="x", y="y"),
    "explicit domain [40, 80]": C(HIGH)
    .mark_bar()
    .encode(x="x", y=fm.Y("y", scale={"domain": [40, 80]})),
}


@pytest.mark.parametrize("name", list(EDGE_CONTROLS))
def test_bars_whose_domain_excludes_zero_start_at_the_plot_bottom(name):
    svg = render(EDGE_CONTROLS[name])
    _, py, _, ph = plot_rect(svg)
    rects = bars(svg)
    assert len(rects) == 3
    assert all(abs(r.y + r.h - (py + ph)) <= TOL_PX for r in rects)


def test_all_negative_bars_whose_domain_excludes_zero_hang_from_the_plot_top():
    """The plot edge on zero's side of an all-negative domain is the top."""
    data = pl.DataFrame({"x": ["a", "b", "c"], "y": [-50.0, -60.0, -70.0]})
    svg = render(C(data).mark_bar(zero=False).encode(x="x", y="y"))
    _, py, _, _ = plot_rect(svg)
    rects = bars(svg)
    assert len(rects) == 3
    assert all(abs(r.y - py) <= TOL_PX for r in rects)


def test_an_area_whose_domain_excludes_zero_closes_on_the_plot_bottom():
    data = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [50.0, 60.0, 55.0, 70.0]})
    svg = render(C(data).mark_area().encode(x="x", y="y"))
    _, py, _, ph = plot_rect(svg)
    [path] = svg_elements(
        mark_layer(svg), "path", lambda a: "d" in a and a.get("fill") not in (None, "none")
    )
    assert sum(abs(p[1] - (py + ph)) <= TOL_PX for p in path_points(path["d"])) >= 2


# -- stacks split by sign ----------------------------------------------------


def stack_spans(svg: str, axis: str) -> list[float]:
    """Every segment's low and high data value, segments sorted, flattened.

    Read through the axis ladder; flat so ``pytest.approx`` compares it.
    """
    ladder = value_axis_ladder(svg, axis=axis)
    spans = []
    for rect in bars(svg):
        a, b = (value_at(ladder, px) for px in rect.span(axis))
        spans.append((round(min(a, b), 3), round(max(a, b), 3)))
    return [value for span in sorted(spans) for value in span]


def flat(spans: list[tuple[float, float]]) -> list[float]:
    return [value for span in spans for value in span]


MIXED_STACK_SPANS = {
    # a: p=2, q=-1 -> P=2, N=1; b: p=-2, q=3 -> P=3, N=2.
    "zero": [(-2.0, 0.0), (-1.0, 0.0), (0.0, 2.0), (0.0, 3.0)],
    "normalize": [(-0.4, 0.0), (-1 / 3, 0.0), (0.0, 0.6), (0.0, 2 / 3)],
    "center": [(-2.5, -0.5), (-1.5, -0.5), (-0.5, 1.5), (-0.5, 2.5)],
}


@pytest.mark.parametrize("offset", list(MIXED_STACK_SPANS))
def test_a_mixed_sign_stack_splits_at_zero_under_every_offset(offset):
    chart = (
        C(GROUPED_MIXED).mark_bar(position=fm.Stack(offset=offset)).encode(x="x", y="y", color="g")
    )
    spans = stack_spans(render(chart), "y")
    assert spans == pytest.approx(flat(MIXED_STACK_SPANS[offset]), abs=0.01)


def test_a_normalized_mixed_stack_fills_one_unit_across_zero():
    chart = (
        C(GROUPED_MIXED)
        .mark_bar(position=fm.Stack(offset="normalize"))
        .encode(x="x", y="y", color="g")
    )
    # Sorted segments: b's negative, a's negative, b's positive, a's positive.
    lows = stack_spans(render(chart), "y")[0::2]
    highs = stack_spans(render(chart), "y")[1::2]
    assert highs[2] - lows[0] == pytest.approx(1.0, abs=0.01)
    assert highs[3] - lows[1] == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize(
    "name, chart, axis",
    [
        (
            "horizontal stack",
            C(GROUPED_MIXED)
            .mark_bar(position=fm.Stack(), orient="horizontal")
            .encode(x="x", y="y", color="g"),
            "x",
        ),
        (
            "quantitative-x stack",
            C(QUANT_GROUPED_MIXED).mark_bar(position=fm.Stack()).encode(x="x", y="y", color="g"),
            "y",
        ),
    ],
)
def test_a_mixed_sign_stack_draws_every_segment(name, chart, axis):
    spans = stack_spans(render(chart), axis)
    assert spans == pytest.approx(flat(MIXED_STACK_SPANS["zero"]), abs=0.01), name


def test_an_all_positive_center_stack_draws_every_segment_at_its_true_extent():
    chart = (
        C(GROUPED_POSITIVE)
        .mark_bar(position=fm.Stack(offset="center"))
        .encode(x="x", y="y", color="g")
    )
    svg = render(chart)
    # a: P=3 shifted by -1.5; b: P=7 shifted by -3.5.
    assert stack_spans(svg, "y") == pytest.approx(
        flat([(-3.5, -0.5), (-1.5, 0.5), (-0.5, 3.5), (0.5, 1.5)]), abs=0.01
    )
    heights = sorted(r.h for r in bars(svg))
    unit = heights[0]
    assert heights == pytest.approx([unit, 2 * unit, 3 * unit, 4 * unit], abs=TOL_PX)


@pytest.mark.parametrize("offset", ["zero", "normalize"])
def test_a_stack_summing_to_zero_draws_zero_height_segments_at_zero(offset):
    data = pl.DataFrame(
        {"x": ["a", "a", "b", "b"], "g": ["p", "q", "p", "q"], "y": [0.0, 0.0, 1.0, 2.0]}
    )
    chart = C(data).mark_bar(position=fm.Stack(offset=offset)).encode(x="x", y="y", color="g")
    svg = render(chart)
    rects = bars(svg)
    assert len(rects) == 4, rects
    z = zero_px(svg, "y")
    zero_height = [r for r in rects if r.h == 0.0]
    assert len(zero_height) == 2
    assert all(abs(r.y - z) <= TOL_PX for r in zero_height)


# -- a zero-value bar is a zero-height rect ----------------------------------


def test_a_zero_value_bar_is_a_zero_height_rect_on_the_zero_tick():
    data = pl.DataFrame({"x": ["a", "b", "c"], "y": [3.0, 0.0, 1.0]})
    svg = render(C(data).mark_bar().encode(x="x", y="y"))
    rects = sorted(bars(svg), key=lambda r: r.x)
    assert len(rects) == 3
    assert rects[1].h == 0.0
    assert abs(rects[1].y - zero_px(svg, "y")) <= TOL_PX


def test_an_empty_histogram_bin_keeps_its_rect():
    sample = pl.DataFrame({"v": [0.0, 0.2, 3.8, 4.0]})
    svg = render(C(sample).mark_histogram(bins=4).encode(x="v"))
    rects = bars(svg)
    assert len(rects) == 4, rects
    z = zero_px(svg, "y")
    empty = [r for r in rects if r.h == 0.0]
    assert len(empty) == 2
    assert all(abs(r.y - z) <= TOL_PX for r in empty)


# -- native horizontal and SecondaryY anchors --------------------------------


def _encoding(chart) -> dict:
    return json.loads(chart.to_spec().to_json())["encoding"]


def test_a_native_horizontal_bar_anchors_its_x_value_axis_at_zero():
    enc = _encoding(C(POSITIVE).mark_bar().encode(y="x:N", x="y"))
    assert enc["x"]["scale"]["zero"] is True
    assert "scale" not in enc["y"]


def test_a_native_horizontal_bar_under_coord_flip_anchors_its_authored_x():
    enc = _encoding(C(POSITIVE).mark_bar().encode(y="x:N", x="y").coord(fm.CoordFlip()))
    assert enc["x"]["scale"]["zero"] is True
    assert "scale" not in enc["y"]


def test_a_vertical_bar_anchors_y_and_leaves_x_alone():
    enc = _encoding(C(POSITIVE).mark_bar().encode(x="x", y="y"))
    assert enc["y"]["scale"]["zero"] is True
    assert "scale" not in enc["x"]


@pytest.mark.parametrize(
    "chart",
    [
        C(POSITIVE).mark_bar(zero=False).encode(y="x:N", x="y"),
        C(POSITIVE.with_columns(lo=pl.lit(0.5))).mark_bar().encode(y="x:N", x="lo", x2="y"),
    ],
    ids=["zero=False", "x2 bound"],
)
def test_a_native_horizontal_bar_without_an_anchor_leaves_x_unscaled(chart):
    assert "scale" not in _encoding(chart)["x"]


def test_a_native_horizontal_bar_draws_a_zero_tick():
    svg = render(C(POSITIVE).mark_bar().encode(y="x:N", x="y"))
    z = zero_px(svg, "x")
    assert all(abs(r.x - z) <= TOL_PX for r in bars(svg))


def test_a_secondary_y_bar_grows_from_its_own_zero_tick():
    data = pl.DataFrame({"x": ["a", "b", "c"], "y": [3.0, 2.0, 1.0], "v": [1.85e6, 2.0e6, 2.4e6]})
    svg = render(C(data).mark_line().encode(x="x", y="y") + fm.SecondaryY("v", mark="bar"))
    ladders = value_axis_ladders(svg, axis="y")
    assert len(ladders) == 2, ladders
    secondary = ladders[-1][1]
    assert 0.0 in secondary, f"the secondary axis does not reach 0: {sorted(secondary)}"
    rects = bars(svg)
    assert len(rects) == 3
    assert all(abs(r.y + r.h - secondary[0.0]) <= TOL_PX for r in rects)


# -- all-negative stacks and the edge on zero's side -------------------------

GROUPED_NEGATIVE = pl.DataFrame(
    {"x": ["a", "a", "b", "b"], "g": ["p", "q", "p", "q"], "y": [-2.0, -1.0, -3.0, -4.0]}
)
NEGATIVE_STACK_SPANS = {
    # a: p=-2, q=-1 -> N=3; b: p=-3, q=-4 -> N=7.
    "zero": [(-7.0, -3.0), (-3.0, -2.0), (-3.0, 0.0), (-2.0, 0.0)],
    "normalize": [(-1.0, -2 / 3), (-1.0, -3 / 7), (-2 / 3, 0.0), (-3 / 7, 0.0)],
    "center": [(-3.5, 0.5), (-1.5, -0.5), (-0.5, 1.5), (0.5, 3.5)],
}


@pytest.mark.parametrize("offset", list(NEGATIVE_STACK_SPANS))
def test_an_all_negative_bar_stack_hangs_below_zero(offset):
    """``zero`` spans ``[-N, 0]``, ``normalize`` fills ``[-1, 0]``, ``center`` is centered on 0."""
    chart = (
        C(GROUPED_NEGATIVE)
        .mark_bar(position=fm.Stack(offset=offset))
        .encode(x="x", y="y", color="g")
    )
    spans = stack_spans(render(chart), "y")
    assert spans == pytest.approx(flat(NEGATIVE_STACK_SPANS[offset]), abs=0.01)


AREA_NEGATIVE = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "g": ["p"] * 3 + ["q"] * 3,
        "y": [-2.0, -1.0, -3.0, -1.0, -2.0, -2.5],
    }
)


def filled_paths(svg: str) -> list[list[tuple[float, float]]]:
    """Every filled path's vertices inside the plot area, in document order."""
    return [
        path_points(a["d"])
        for a in svg_elements(
            mark_layer(svg), "path", lambda a: "d" in a and a.get("fill") not in (None, "none")
        )
    ]


@pytest.mark.parametrize("offset", ["zero", "normalize"])
def test_an_all_negative_stacked_area_closes_its_first_layer_on_the_top_edge(offset):
    """Its domain excludes 0, so the first layer closes on the edge on zero's side."""
    chart = (
        C(AREA_NEGATIVE).mark_area(position=fm.Stack(offset=offset)).encode(x="x", y="y", color="g")
    )
    svg = render(chart)
    _, py, _, ph = plot_rect(svg)
    first, second = filled_paths(svg)
    assert sum(abs(y - py) <= TOL_PX for _, y in first) >= 2, first
    assert all(abs(y - (py + ph)) > 1.0 for _, y in first + second), "a layer closes on the bottom"


def test_an_all_negative_centered_area_stack_is_centered_on_zero():
    chart = (
        C(AREA_NEGATIVE)
        .mark_area(position=fm.Stack(offset="center"))
        .encode(x="x", y="y", color="g")
    )
    svg = render(chart)
    z = zero_px(svg, "y")
    ys = [y for path in filled_paths(svg) for _, y in path]
    assert z - min(ys) == pytest.approx(max(ys) - z, abs=TOL_PX)


def test_a_reversed_zero_false_bar_hangs_from_the_top_edge():
    """``reverse=True`` puts zero's side of a 50..70 domain at the top edge."""
    svg = render(C(HIGH).mark_bar(zero=False).encode(x="x", y=fm.Y("y", scale={"reverse": True})))
    _, py, _, _ = plot_rect(svg)
    rects = sorted(bars(svg), key=lambda r: r.x)
    assert len(rects) == 3
    assert all(abs(r.y - py) <= TOL_PX for r in rects)
    assert rects[0].h < rects[1].h < rects[2].h


# -- orientation read from the column when y is undeclared -------------------

UNDECLARED_CATEGORICAL_Y = {
    "Utf8": pl.Series(["a", "b", "c"]),
    "Categorical": pl.Series(["a", "b", "c"], dtype=pl.Categorical),
    "Enum": pl.Series(["a", "b", "c"], dtype=pl.Enum(["a", "b", "c"])),
    "Boolean": pl.Series([True, False, None]),
}


@pytest.mark.parametrize("dtype", list(UNDECLARED_CATEGORICAL_Y))
def test_an_undeclared_categorical_y_draws_a_horizontal_bar_from_the_zero_tick(dtype):
    data = pl.DataFrame({"val": [3.0, -2.0, 1.0], "c": UNDECLARED_CATEGORICAL_Y[dtype]})
    chart = C(data).mark_bar().encode(x="val", y="c")
    enc = _encoding(chart)
    assert enc["x"]["scale"]["zero"] is True
    assert "scale" not in enc["y"]
    assert_grow_from_zero(render(chart), "x", n_positive=2, n_negative=1)


@pytest.mark.parametrize("dtype", list(UNDECLARED_CATEGORICAL_Y))
def test_a_layered_undeclared_categorical_y_bar_agrees(dtype):
    """``bar + point`` reaches the same orientation through Rust's layered rule."""
    data = pl.DataFrame({"val": [3.0, -2.0, 1.0], "c": UNDECLARED_CATEGORICAL_Y[dtype]})
    chart = C(data).mark_bar().encode(x="val", y="c") + C(data).mark_point().encode(x="val", y="c")
    assert_grow_from_zero(render(chart), "x", n_positive=2, n_negative=1)


def test_an_undeclared_date_y_keeps_its_value_on_y():
    """A date column reads numeric to the render, so the bar stays vertical."""
    data = pl.DataFrame(
        {"val": [3.0, 2.0, 1.0], "d": [datetime.date(2020, 1, day) for day in (1, 2, 3)]}
    )
    enc = _encoding(C(data).mark_bar().encode(x="val", y="d"))
    assert enc["y"]["scale"]["zero"] is True
    assert "scale" not in enc["x"]


# -- independent-y bars take the standalone bar's anchor ---------------------

SLOT_DATA = pl.DataFrame({"x": ["a", "b", "c"], "y": [3.0, 2.0, 1.0], "v": [1.85e6, 2.0e6, 2.4e6]})
#: Each caller-scale shape, and whether the standalone bar's axis reaches 0.
SLOT_SCALES = {
    "no scale": (None, True),
    "nice": ({"nice": True}, True),
    "padding": ({"padding": 0.1}, True),
    "range": ({"range": [180, 20]}, True),
    "clamp": ({"clamp": True}, True),
    "reverse": ({"reverse": True}, True),
    "type-less dict": ({"nice": False}, True),
    "explicit domain": ({"domain": [1e6, 3e6]}, False),
    "log": ({"type": "log"}, False),
}


def _reaches_zero(svg: str, *, secondary: bool) -> bool:
    ladders = value_axis_ladders(svg, axis="y")
    return 0.0 in ladders[-1 if secondary else 0][1]


def _bar_v(scale) -> fm.Chart:
    y = fm.Y("v") if scale is None else fm.Y("v", scale=scale)
    return C(SLOT_DATA).mark_bar().encode(x="x", y=y)


def _line() -> fm.Chart:
    return C(SLOT_DATA).mark_line().encode(x="x", y="y")


@pytest.mark.parametrize("name", list(SLOT_SCALES))
def test_a_secondary_y_bar_reaches_zero_exactly_when_the_standalone_bar_does(name):
    scale, reaches = SLOT_SCALES[name]
    assert _reaches_zero(render(_bar_v(scale)), secondary=False) is reaches
    kwargs = {} if scale is None else {"scale": scale}
    slot = _line() + fm.SecondaryY("v", mark="bar", **kwargs)
    assert _reaches_zero(render(slot), secondary=True) is reaches


@pytest.mark.parametrize("name", list(SLOT_SCALES))
def test_an_independent_y_layer_chart_bar_reaches_zero_exactly_when_the_standalone_bar_does(name):
    scale, reaches = SLOT_SCALES[name]
    chart = fm.LayerChart(_line(), _bar_v(scale), resolve={"y": "independent"})
    assert _reaches_zero(render(chart), secondary=True) is reaches


def test_an_independent_y_bar_that_opts_out_keeps_its_data_domain():
    bar = C(SLOT_DATA).mark_bar(zero=False).encode(x="x", y="v")
    chart = fm.LayerChart(_line(), bar, resolve={"y": "independent"})
    assert not _reaches_zero(render(chart), secondary=True)


def test_a_secondary_y_line_slot_takes_no_anchor():
    assert not _reaches_zero(render(_line() + fm.SecondaryY("v", mark="line")), secondary=True)


# -- every encoding spelling reaches the same anchor -------------------------


def _layer_member(y) -> fm.Chart:
    """A bar member whose encoding is a layer's raw ``encoding=`` dict."""
    return C(SLOT_DATA).layer(Layer(mark="bar", encoding={"x": "x", "y": y}))


@pytest.mark.parametrize("y", ["v", fm.Y("v")], ids=["plain string", "channel object"])
def test_an_independent_y_layer_bar_reaches_zero_in_every_spelling(y):
    chart = fm.LayerChart(_line(), _layer_member(y), resolve={"y": "independent"})
    assert _reaches_zero(render(chart), secondary=True)


def _independent_bar_layer(chart: fm.Chart) -> dict:
    layers = json.loads(chart.to_spec().to_json())["layers"]
    [layer] = [layer for layer in layers if layer.get("independent_y")]
    return layer["encoding"]


@pytest.mark.parametrize(
    "y, anchored",
    [("v", True), ("v:Q", True), ("v:O", False), (fm.Y("v"), True)],
    ids=["plain string", "shorthand Q", "shorthand O", "channel object"],
)
def test_an_independent_y_layer_bar_reads_the_type_its_spelling_declares(y, anchored):
    member = _layer_member(y)
    merged = fm.LayerChart(_line(), member, resolve={"y": "independent"})._build_merged()
    scale = _independent_bar_layer(merged)["y"].get("scale") or {}
    assert (scale.get("zero") is True) is anchored


def test_a_secondary_y_bar_over_a_string_encoded_base_layer_reaches_zero():
    base = (
        C(SLOT_DATA)
        .mark_line()
        .encode(x="x", y="y")
        .layer(Layer(mark="point", encoding={"x": "x", "y": "y"}))
    )
    assert _reaches_zero(render(base + fm.SecondaryY("v", mark="bar")), secondary=True)


# -- the categorical reading follows the schema the render receives ----------


def test_an_undeclared_all_null_y_bar_stays_vertical():
    """An all-null column reaches the render as float64, so it is no category."""
    data = pl.DataFrame({"x": ["a", "b"], "y": pl.Series([None, None], dtype=pl.Null)})
    chart = C(data).mark_bar().encode(x="x", y="y")
    enc = _encoding(chart)
    assert enc["y"]["scale"]["zero"] is True
    assert "scale" not in enc["x"]
    render(chart)


@pytest.mark.parametrize(
    "column, categorical",
    [
        (pa.array([None, None], type=pa.null()), False),
        (pa.array([1, 2]).dictionary_encode(), False),
        (
            pa.array([datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)]).dictionary_encode(),
            False,
        ),
        (pa.array(["a", "b"]).dictionary_encode(), True),
    ],
    ids=["null", "dictionary<int>", "dictionary<date>", "dictionary<string>"],
)
def test_a_column_reads_categorical_as_its_rendered_type(column, categorical):
    """A dictionary reads as its value type and null as float64, as the render decodes them."""
    assert _column_reads_categorical(pa.table({"y": column}), "y") is categorical
