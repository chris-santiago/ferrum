"""An infinite value is unusable on every statistic and every data transform, as a null is.

A row whose value is ``+inf`` or ``-inf`` is skipped by every statistic (a histogram, a density,
a box, a violin, an error bar, a fit, a 2-D density, a hexbin, an ECDF, a QQ, a boxen) and by every
transform that computes from a column's values (``transform_aggregate`` and its join, the rolling
window ops, ``transform_stack``, ``transform_density``, ``transform_bin``, ``transform_pivot``,
``transform_top_k``, the fill statistic of ``transform_impute``), exactly as a null is, with no
warning for the skip itself. The chart then draws from the finite rows. A group, panel or call left
with no usable value takes the outcome and warnings its all-null twin takes.

The matrix renders each door over a frame holding an infinity and over the same frame with the
infinity replaced by a null (its null twin), and requires the same chart and the same warnings. The
ordering, selection and pass-through transforms (a filter, a rank, a sort key) keep an infinity as
the value it is, and a plain mark skips it on a linear axis as it always has.
"""

from __future__ import annotations

import hashlib
import json
import datetime as dt
from collections import Counter
from collections.abc import Sequence
from typing import Any, Callable, NamedTuple

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._pixel_capture import parse_circles
from tests._svg_extents import text_elements
from tests.test_null_group_warning import (
    _NO_ROWS_TO_DRAW,
    _layer_message,
    FIT_BANDWIDTH,
    FIT_FEW,
    FIT_FEW_LOESS_POINTS,
    FIT_FEW_PAIRS,
    FIT_FEW_POINTS,
    FIT_NONE,
    FIT_TIED,
    _fit_message,
    _infinite_warning,
    _panel_fit_message,
    _render_capturing,
    _log_message,
    _slot_message,
    _statistic_infinite_cut_warning,
    _whole_fit_message,
)

INF = float("inf")
NAN = float("nan")

# Nine rows: ``c``, ``g`` and ``f`` cycle a, b, c / a, b, c / p, q, r, so group ``c`` and facet
# panel ``r`` hold rows 2, 5 and 8, the rows the inputs below put an infinity in. ``h`` splits the
# rows into two blocks for the pivot.
_FINITE = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.5, 3.5, 4.5]
_W = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 2.0, 4.0, 5.0]
_B = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
_N = [1.0, 2.0, 2.0, 3.0, 5.0, 8.0, 3.0, 4.0, 6.0]


def _cycled(values: Sequence[Any], n: int) -> list[Any]:
    return [values[k % len(values)] for k in range(n)]


def _frame(v: Sequence[float | None]) -> pl.DataFrame:
    """One row per value of ``v`` (a multiple of three), with the group, panel and companion
    columns cycling as described above."""
    n = len(v)
    return pl.DataFrame(
        {
            "v": list(v),
            "w": _cycled(_W, n),
            "b": _cycled(_B, n),
            "n": _cycled(_N, n),
            "i": [float(k) for k in range(1, n + 1)],
            "c": _cycled(["a", "b", "c"], n),
            "g": _cycled(["a", "b", "c"], n),
            "f": _cycled(["p", "q", "r"], n),
            "h": ["m"] * ((n + 1) // 2) + ["n"] * (n // 2),
        },
        schema={
            "v": pl.Float64,
            "w": pl.Float64,
            "b": pl.Float64,
            "n": pl.Float64,
            "i": pl.Float64,
            "c": pl.Utf8,
            "g": pl.Utf8,
            "f": pl.Utf8,
            "h": pl.Utf8,
        },
    )


#: The inputs: an infinity of either sign, an infinity beside a null and a NaN (where group ``b``
#: keeps one usable value, and where every group keeps three of five), one group and panel left
#: with only infinities, and every value infinite.
_INPUTS: dict[str, pl.DataFrame] = {
    "pos-inf": _frame(_FINITE[:8] + [INF]),
    "neg-inf": _frame(_FINITE[:8] + [-INF]),
    "inf-in-group-c": _frame([1.0, 2.0, INF, 4.0, 5.0, 3.0, 2.5, 3.5, 4.5]),
    "inf-beside-null-and-nan": _frame([1.0, None, INF, 4.0, NAN, 3.0, -INF, 3.5, 4.5]),
    "inf-beside-null-and-nan-wide": _frame(
        [1.0, None, INF, -INF, NAN, 3.0, 2.0, 3.5, 4.5, 2.5, 6.0, 1.5, 5.0, 4.0, 3.2]
    ),
    "group-c-all-inf": _frame([1.0, 2.0, INF, 4.0, 5.0, -INF, 2.5, 3.5, INF]),
    "all-inf": _frame([INF, -INF, INF, INF, -INF, INF, INF, -INF, INF]),
}
_INPUT_CASES = [pytest.param(name, id=name) for name in _INPUTS]
#: The inputs that leave every group with enough usable values that the null twin is silent, so
#: a skipped infinity warns nothing.
_SILENT_INPUTS = ["pos-inf", "neg-inf", "inf-in-group-c", "inf-beside-null-and-nan-wide"]


def _null_twin(frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` with every infinite value in a float column replaced by a null."""
    return frame.with_columns(
        pl.when(pl.col(col).is_infinite()).then(None).otherwise(pl.col(col)).alias(col)
        for col, dtype in frame.schema.items()
        if dtype.is_float()
    )


class _Outcome(NamedTuple):
    """What a render gave: the output (or the refusal), and every warning in order."""

    output: str
    messages: list[str]


def _outcome(render: Callable[[], str]) -> _Outcome:
    try:
        svg, messages = _render_capturing(render)
    except Exception as exc:  # noqa: BLE001 - a refusal is an outcome the twin must share
        return _Outcome(f"RAISE {type(exc).__name__}: {exc}", [])
    return _Outcome(svg, messages)


def _static(chart: Any) -> Callable[[], str]:
    return chart.to_svg


def _scene_of(chart: Any) -> Callable[[], str]:
    """The interactive scene's geometry and axes, without the tooltips that carry the raw values."""

    def render() -> str:
        scene = json.loads(chart.interactive().scene_json)
        return json.dumps(
            [
                {
                    "axes": panel["axes"],
                    "marks": [(batch["kind"], batch["nodes"]) for batch in panel["marks"]],
                }
                for panel in scene["panels"]
            ],
            sort_keys=True,
        )

    return render


# ---------------------------------------------------------------------------
# The doors: one builder per statistic, with the point chart a layering door pairs it with.
# ---------------------------------------------------------------------------

C = fm.Chart


class _Door(NamedTuple):
    build: Callable[[pl.DataFrame], Any]
    #: the x and y of the point chart a layering door composes it with
    base: tuple[str, str] = ("c", "v")


_STATISTIC_DOORS: dict[str, _Door] = {
    # 1-D distributions
    "mark_histogram": _Door(lambda d: C(d).mark_histogram().encode(x="v"), ("v", "w")),
    "mark_histogram hue": _Door(
        lambda d: C(d).mark_histogram().encode(x="v", color="g"), ("v", "w")
    ),
    "mark_histogram facet": _Door(
        lambda d: C(d).mark_histogram().encode(x="v").facet(col="f"), ("v", "w")
    ),
    "mark_bar binned count": _Door(
        lambda d: C(d).mark_bar().encode(x=fm.X("v", bin=True), y="count()"), ("v", "w")
    ),
    "displot hist": _Door(lambda d: fm.displot(d, x="v"), ("v", "w")),
    "displot hist hue": _Door(lambda d: fm.displot(d, x="v", hue="g"), ("v", "w")),
    "mark_density": _Door(lambda d: C(d).mark_density().encode(x="v"), ("v", "w")),
    "mark_density hue": _Door(lambda d: C(d).mark_density().encode(x="v", color="g"), ("v", "w")),
    "mark_density facet": _Door(
        lambda d: C(d).mark_density().encode(x="v").facet(col="f"), ("v", "w")
    ),
    "displot kde": _Door(lambda d: fm.displot(d, x="v", kind="kde"), ("v", "w")),
    "displot kde hue": _Door(lambda d: fm.displot(d, x="v", kind="kde", hue="g"), ("v", "w")),
    "displot ecdf": _Door(lambda d: fm.displot(d, x="v", kind="ecdf"), ("v", "w")),
    "displot ecdf hue": _Door(lambda d: fm.displot(d, x="v", hue="g", kind="ecdf"), ("v", "w")),
    "mark_qq": _Door(lambda d: C(d).mark_qq().encode(x="v"), ("v", "w")),
    "mark_qq hue": _Door(lambda d: C(d).mark_qq().encode(x="v", color="g"), ("v", "w")),
    # distributions by category
    "mark_boxplot": _Door(lambda d: C(d).mark_boxplot().encode(x="c", y="v")),
    "mark_violin": _Door(lambda d: C(d).mark_violin().encode(x="c", y="v")),
    "mark_boxen": _Door(lambda d: C(d).mark_boxen().encode(x="c", y="v")),
    "mark_errorbar": _Door(lambda d: C(d).mark_errorbar().encode(x="c", y="v")),
    "catplot box": _Door(lambda d: fm.catplot(d, x="c", y="v", kind="box")),
    "catplot point": _Door(lambda d: fm.catplot(d, x="c", y="v", kind="point")),
    "catplot bar": _Door(lambda d: fm.catplot(d, x="c", y="v", kind="bar")),
    # aggregates
    **{
        f"mark_bar {op}": _Door(
            lambda d, op=op: C(d).mark_bar().encode(x="c", y=fm.Y("v", aggregate=op))
        )
        for op in ("mean", "sum", "median", "min", "max", "stdev", "count", "distinct")
    },
    # fits (an infinity in x, and in y)
    "mark_smooth lm x": _Door(
        lambda d: C(d).mark_smooth(method="lm").encode(x="v", y="w"), ("v", "w")
    ),
    "mark_smooth lm y": _Door(
        lambda d: C(d).mark_smooth(method="lm").encode(x="w", y="v"), ("w", "v")
    ),
    "mark_smooth loess": _Door(lambda d: C(d).mark_smooth().encode(x="v", y="w"), ("v", "w")),
    "mark_smooth hue": _Door(
        lambda d: C(d).mark_smooth(method="lm").encode(x="v", y="w", color="g"), ("v", "w")
    ),
    "regplot x": _Door(lambda d: fm.regplot(d, x="v", y="w"), ("v", "w")),
    "regplot y": _Door(lambda d: fm.regplot(d, x="w", y="v"), ("w", "v")),
    "lmplot": _Door(lambda d: fm.lmplot(d, x="v", y="w", hue="g"), ("v", "w")),
    "residplot": _Door(lambda d: fm.residplot(d, x="v", y="w"), ("v", "w")),
    "Robust": _Door(
        lambda d: C(d).transform(fm.Robust(x="v", y="w")).mark_line().encode(x="x", y="y"),
        ("v", "w"),
    ),
    "Logistic": _Door(
        lambda d: C(d).transform(fm.Logistic("v", "b", n_grid=20)).mark_line().encode(x="x", y="y"),
        ("v", "w"),
    ),
    "Glm": _Door(
        lambda d: (
            C(d)
            .transform(fm.Glm("v", "n", family="poisson", n_grid=20))
            .mark_line()
            .encode(x="x", y="y")
        ),
        ("v", "w"),
    ),
    "transform_regression": _Door(
        lambda d: (
            C(d).transform(fm.transform_regression("v", "w")).mark_line().encode(x="x", y="y")
        ),
        ("v", "w"),
    ),
    "transform_loess": _Door(
        lambda d: C(d).transform(fm.transform_loess("v", "w")).mark_line().encode(x="x", y="y"),
        ("v", "w"),
    ),
    # 2-D statistics
    "mark_contour": _Door(lambda d: C(d).mark_contour().encode(x="v", y="w"), ("v", "w")),
    "mark_hex": _Door(lambda d: C(d).mark_hex().encode(x="v", y="w"), ("v", "w")),
    "mark_raster": _Door(lambda d: C(d).mark_raster().encode(x="v", y="w"), ("v", "w")),
    "jointplot hex": _Door(lambda d: fm.jointplot(d, x="v", y="w", kind="hex"), ("v", "w")),
    "jointplot hist": _Door(lambda d: fm.jointplot(d, x="v", y="w", kind="hist"), ("v", "w")),
    "jointplot kde": _Door(lambda d: fm.jointplot(d, x="v", y="w", kind="kde"), ("v", "w")),
    "jointplot scatter": _Door(lambda d: fm.jointplot(d, x="v", y="w"), ("v", "w")),
    # plain marks, unchanged: already identical to the null twin
    "mark_point": _Door(lambda d: C(d).mark_point().encode(x="v", y="w"), ("v", "w")),
    "mark_swarm": _Door(lambda d: C(d).mark_swarm().encode(x="c", y="v")),
    # statistics and plain marks pinned beside the rest
    "mark_errorband": _Door(lambda d: C(d).mark_errorband().encode(x="c", y="v")),
    "mark_bar stacked": _Door(
        lambda d: C(d).mark_bar().encode(x="h", y="v", color="c"), ("h", "v")
    ),
    "mark_area stacked": _Door(
        lambda d: C(d).mark_area().encode(x="w", y="v", color="h"), ("w", "v")
    ),
    "mark_line": _Door(lambda d: C(d).mark_line().encode(x="i", y="v"), ("i", "v")),
    "mark_bar": _Door(lambda d: C(d).mark_bar().encode(x="c", y="v")),
    "mark_smooth gam": _Door(
        lambda d: C(d).mark_smooth(method="gam", k=4).encode(x="v", y="w"), ("v", "w")
    ),
    "mark_bar binned count of a field": _Door(
        lambda d: C(d).mark_bar().encode(x=fm.X("w", bin=True), y=fm.Y("v", aggregate="count")),
        ("w", "v"),
    ),
}

#: The data transforms that compute from a column's values, drawn through a mark that reads
#: their output.
_TRANSFORM_DOORS: dict[str, _Door] = {
    "transform_aggregate sum": _Door(
        lambda d: (
            C(d)
            .transform(
                fm.transform_aggregate({"field": "v", "fn": "sum", "as": "s"}, groupby=["c"])
            )
            .mark_bar()
            .encode(x="c", y="s")
        ),
        ("c", "s"),
    ),
    "transform_aggregate mean": _Door(
        lambda d: (
            C(d)
            .transform(
                fm.transform_aggregate({"field": "v", "fn": "mean", "as": "s"}, groupby=["c"])
            )
            .mark_bar()
            .encode(x="c", y="s")
        ),
        ("c", "s"),
    ),
    "transform_aggregate count": _Door(
        lambda d: (
            C(d)
            .transform(
                fm.transform_aggregate({"field": "v", "fn": "count", "as": "s"}, groupby=["c"])
            )
            .mark_bar()
            .encode(x="c", y="s")
        ),
        ("c", "s"),
    ),
    "transform_join_aggregate": _Door(
        lambda d: (
            C(d)
            .transform(
                fm.transform_join_aggregate({"field": "v", "fn": "mean", "as": "m"}, groupby=["c"])
            )
            .mark_point()
            .encode(x="i", y="m")
        ),
        ("i", "m"),
    ),
    "transform_window rolling sum": _Door(
        lambda d: (
            C(d)
            .transform(
                fm.transform_window(
                    {"op": "sum", "field": "v", "as": "s"}, sort=["i"], frame=(None, 0)
                )
            )
            .mark_line()
            .encode(x="i", y="s")
        ),
        ("i", "s"),
    ),
    "transform_window rolling mean": _Door(
        lambda d: (
            C(d)
            .transform(
                fm.transform_window(
                    {"op": "mean", "field": "v", "as": "s"}, sort=["i"], frame=(None, 0)
                )
            )
            .mark_line()
            .encode(x="i", y="s")
        ),
        ("i", "s"),
    ),
    "transform_window rolling count": _Door(
        lambda d: (
            C(d)
            .transform(
                fm.transform_window(
                    {"op": "count", "field": "v", "as": "s"}, sort=["i"], frame=(None, 0)
                )
            )
            .mark_line()
            .encode(x="i", y="s")
        ),
        ("i", "s"),
    ),
    "transform_window rolling min": _Door(
        lambda d: (
            C(d)
            .transform(
                fm.transform_window(
                    {"op": "min", "field": "v", "as": "s"}, sort=["i"], frame=(None, 0)
                )
            )
            .mark_line()
            .encode(x="i", y="s")
        ),
        ("i", "s"),
    ),
    "transform_window rolling max": _Door(
        lambda d: (
            C(d)
            .transform(
                fm.transform_window(
                    {"op": "max", "field": "v", "as": "s"}, sort=["i"], frame=(None, 0)
                )
            )
            .mark_line()
            .encode(x="i", y="s")
        ),
        ("i", "s"),
    ),
    "transform_join_aggregate count": _Door(
        lambda d: (
            C(d)
            .transform(
                fm.transform_join_aggregate({"field": "v", "fn": "count", "as": "m"}, groupby=["c"])
            )
            .mark_point()
            .encode(x="i", y="m")
        ),
        ("i", "m"),
    ),
    "transform_stack": _Door(
        lambda d: (
            C(d)
            .transform(fm.transform_stack("v", groupby=["c"], sort=["i"]))
            .mark_bar()
            .encode(x="c", y="y0", y2="y1")
        ),
        ("c", "y1"),
    ),
    "transform_density": _Door(
        lambda d: (
            C(d).transform(fm.transform_density("v")).mark_line().encode(x="value", y="density")
        ),
        ("value", "density"),
    ),
    "transform_bin": _Door(
        lambda d: (
            C(d).transform(fm.transform_bin("v", as_="vb")).mark_point().encode(x="vb", y="w")
        ),
        ("vb", "w"),
    ),
    "transform_top_k": _Door(
        lambda d: (
            C(d)
            .transform(fm.transform_top_k(1, field="v", op="sum", groupby=["c"]))
            .mark_point()
            .encode(x="i", y="w")
        ),
        ("i", "w"),
    ),
    "transform_pivot": _Door(
        lambda d: (
            C(d)
            .transform(fm.transform_pivot("c", "v", op="sum", groupby=["h"]))
            .mark_point()
            .encode(x="a", y="c")
        ),
        ("a", "c"),
    ),
}

_ALL_DOORS = {**_STATISTIC_DOORS, **_TRANSFORM_DOORS}
_DOOR_CASES = [pytest.param(name, id=name) for name in _ALL_DOORS]


def _guard_against_a_refusal(twin: _Outcome, door: str) -> None:
    """The twin is the reference, so it must be a chart: a door both frames refuse proves nothing."""
    assert not twin.output.startswith("RAISE"), f"{door} refuses its null twin: {twin.output}"


# ---------------------------------------------------------------------------
# The matrix: every door over every input, against its null twin.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_name", _INPUT_CASES)
@pytest.mark.parametrize("name", _DOOR_CASES)
def test_an_infinite_value_is_skipped_as_a_null_one_is(name: str, input_name: str):
    """The chart over a frame holding an infinity, or holding only infinities, is the chart over
    the same frame with a null in its place, and warns what that chart warns."""
    door, frame = _ALL_DOORS[name], _INPUTS[input_name]
    got = _outcome(_static(door.build(frame)))
    twin = _outcome(_static(door.build(_null_twin(frame))))
    _guard_against_a_refusal(twin, name)
    assert got.messages == twin.messages
    assert got.output == twin.output


_LMPLOT_C_BAND_UNDEFINED = (
    "Smooth: the ci=0.95 band of group g=c is undefined (2 points leave no residual degrees of "
    "freedom for a 2-parameter fit; the band needs at least 3); the fit line is drawn without it"
)


#: The one pair whose null twin names a hue x category cell: row 2 is the only ``h=m, c=c`` row of
#: the stacked bar, so that cell draws nothing while ``c=c`` draws under ``h=n``.
_STACKED_CELL = ("mark_bar stacked", "inf-in-group-c")
_SILENT_CASES = [
    pytest.param(name, input_name, id=f"{name}-{input_name}")
    for input_name in _SILENT_INPUTS
    for name in _ALL_DOORS
    if (name, input_name) != _STACKED_CELL
]


@pytest.mark.parametrize("name,input_name", _SILENT_CASES)
def test_a_skipped_infinite_value_raises_no_warning(name: str, input_name: str):
    """Skipping the infinity is not itself reported: on a frame where every group keeps enough
    usable values for its null twin to be silent, the chart warns nothing."""
    frame = _INPUTS[input_name]
    twin = _outcome(_static(_ALL_DOORS[name].build(_null_twin(frame))))
    got = _outcome(_static(_ALL_DOORS[name].build(frame)))
    if name == "lmplot" and input_name != "inf-beside-null-and-nan-wide":
        # Its group ``c`` keeps two points over nine rows, too few for a confidence band: the
        # twin says so, and the infinity adds nothing to that.
        assert twin.messages == [_LMPLOT_C_BAND_UNDEFINED]
        assert got.messages == twin.messages
    else:
        assert twin.messages == [], "the fixture's groups keep enough usable values"
        assert got.messages == []


def test_a_skipped_infinite_value_in_a_stacked_bar_names_the_cell_its_twin_names():
    """The infinity empties the cell ``h=m, c=c`` exactly as the null does, so both charts name
    that cell, and name nothing else."""
    name, input_name = _STACKED_CELL
    frame = _INPUTS[input_name]
    twin = _outcome(_static(_ALL_DOORS[name].build(_null_twin(frame))))
    got = _outcome(_static(_ALL_DOORS[name].build(frame)))
    expected = ["mark_bar: every row of h=m, c=c has a null or non-finite y, so it draws nothing"]
    assert twin.messages == expected
    assert got.messages == expected


def test_a_density_group_left_with_one_usable_value_names_the_count():
    frame = _INPUTS["inf-beside-null-and-nan"]
    _, shown = _render_capturing(C(frame).mark_density().encode(x="v", color="g").to_svg)
    assert shown == [_fit_message("Kde", "g=b", FIT_FEW)]


# ---------------------------------------------------------------------------
# Every layering door and the interactive scene read an infinity the same way.
# ---------------------------------------------------------------------------

_LAYERED_DOORS = [
    pytest.param(name, id=name)
    for name in (
        "mark_histogram",
        "mark_density",
        "mark_boxplot",
        "mark_violin",
        "mark_boxen",
        "mark_errorbar",
        "mark_errorband",
        "mark_bar mean",
        "mark_bar median",
        "mark_smooth lm x",
        "mark_qq",
        "mark_contour",
        "mark_hex",
    )
]
_LAYERINGS = {
    "+": lambda base, stat: base + stat,
    "LayerChart": lambda base, stat: fm.LayerChart(base, stat),
}
_LAYERED_INPUTS = [
    pytest.param(name, id=name)
    for name in ("pos-inf", "neg-inf", "inf-in-group-c", "group-c-all-inf")
]


def _layered(layering: str, name: str, frame: pl.DataFrame) -> Any:
    door = _ALL_DOORS[name]
    base_x, base_y = door.base
    base = C(frame).mark_point().encode(x=base_x, y=base_y)
    return _LAYERINGS[layering](base, door.build(frame))


@pytest.mark.parametrize("input_name", _LAYERED_INPUTS)
@pytest.mark.parametrize("layering", list(_LAYERINGS))
@pytest.mark.parametrize("name", _LAYERED_DOORS)
def test_a_statistic_layered_over_a_point_chart_skips_an_infinity_as_a_null(
    name: str, layering: str, input_name: str
):
    """``+`` and ``LayerChart`` compose the statistic with a point chart of the same frame, and
    the layered chart reads the infinity as the flat one does."""
    frame = _INPUTS[input_name]
    got = _outcome(_static(_layered(layering, name, frame)))
    twin = _outcome(_static(_layered(layering, name, _null_twin(frame))))
    _guard_against_a_refusal(twin, name)
    assert got == twin


#: Each aggregate a ``Chart.layer`` door reaches, as the layer it adds to a point chart.
_LAYER_DOOR_AGGREGATES = {
    **{
        f"bar {op}": fm.Layer(mark="bar", encoding={"x": "c", "y": fm.Y("v", aggregate=op)})
        for op in ("mean", "sum", "median", "min", "max", "stdev", "count", "distinct")
    },
    "binned count": fm.Layer(
        mark="bar", encoding={"x": fm.X("v", bin=True), "y": fm.Y(aggregate="count")}
    ),
}


@pytest.mark.parametrize("input_name", _LAYERED_INPUTS)
@pytest.mark.parametrize("name", list(_LAYER_DOOR_AGGREGATES))
def test_an_aggregate_layer_attached_with_chart_layer_skips_an_infinity_as_a_null(
    name: str, input_name: str
):
    """``Chart.layer`` attaches an aggregate to a host chart of the same frame: the layer reads
    the infinity as the flat chart does."""
    frame = _INPUTS[input_name]

    def chart(data: pl.DataFrame) -> Any:
        x, y = ("v", "w") if name == "binned count" else ("c", "v")
        return C(data).mark_point().encode(x=x, y=y).layer(_LAYER_DOOR_AGGREGATES[name])

    got = _outcome(_static(chart(frame)))
    twin = _outcome(_static(chart(_null_twin(frame))))
    _guard_against_a_refusal(twin, name)
    assert got == twin


_INTERACTIVE_DOORS = [
    pytest.param(name, id=name)
    for name in (
        "mark_histogram",
        "mark_histogram facet",
        "mark_density",
        "mark_boxplot",
        "mark_violin",
        "mark_boxen",
        "mark_errorbar",
        "mark_errorband",
        "mark_bar mean",
        "mark_bar median",
        "mark_smooth lm x",
        "mark_qq",
        "mark_contour",
        "mark_hex",
        "mark_raster",
        "transform_aggregate sum",
        "transform_window rolling sum",
    )
]


@pytest.mark.parametrize("input_name", _LAYERED_INPUTS)
@pytest.mark.parametrize("name", _INTERACTIVE_DOORS)
def test_the_interactive_scene_skips_an_infinity_as_a_null(name: str, input_name: str):
    """``.interactive()`` builds the scene from the same statistic: its marks and axes, and the
    warnings the build raises, are the null twin's."""
    door, frame = _ALL_DOORS[name], _INPUTS[input_name]
    got = _outcome(_scene_of(door.build(frame)))
    twin = _outcome(_scene_of(door.build(_null_twin(frame))))
    _guard_against_a_refusal(twin, name)
    assert got == twin


# ---------------------------------------------------------------------------
# What an infinity changes, measured directly: the values the statistic reads.
# ---------------------------------------------------------------------------


def _rect_heights(svg: str) -> list[float]:
    from tests._svg_rects import data_bar_rects, mark_layer

    return [float(r["height"]) for r in data_bar_rects(mark_layer(svg))]


@pytest.mark.parametrize("value", [INF, -INF], ids=["pos", "neg"])
def test_a_histogram_over_finite_values_and_an_infinity_bins_the_finite_values(value: float):
    """Six finite values and an infinity bin as the six: no warning, the bars of the six values
    alone. Nothing says the layer draws nothing."""
    infinite = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, value]})
    six = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    svg, shown = _render_capturing(C(infinite).mark_histogram().encode(x="v").to_svg)
    assert shown == []
    assert svg == C(six).mark_histogram().encode(x="v").to_svg()
    assert _rect_heights(svg), "the histogram draws bars"


@pytest.mark.parametrize(
    "op", ["mean", "sum", "median", "min", "max", "stdev", "count", "distinct"]
)
def test_an_aggregate_bar_over_a_group_holding_an_infinity_is_the_statistic_of_its_finite_values(
    op: str,
):
    """Group ``c`` holds ``[3, 6, inf]``: its bar is the statistic of ``[3, 6]``, drawn as the
    frame without the infinite row draws it."""
    with_inf = _INPUTS["inf-in-group-c"]
    dropped = with_inf.filter(pl.col("v").is_finite())
    chart = lambda d: C(d).mark_bar().encode(x="c", y=fm.Y("v", aggregate=op))  # noqa: E731
    svg, shown = _render_capturing(chart(with_inf).to_svg)
    assert shown == []
    group_c = svg.count("<rect")
    assert group_c, "the bars draw"
    reference, _ = _render_capturing(chart(dropped).to_svg)
    assert _rect_heights(svg) == _rect_heights(reference)


def test_a_median_bar_reads_the_finite_values_only():
    """The median of ``[3, 6, inf]`` is 4.5, the median of its finite values, not 6."""
    frame = pl.DataFrame({"c": ["c", "c", "c"], "v": [3.0, 6.0, INF]})
    twin = pl.DataFrame({"c": ["c", "c"], "v": [3.0, 6.0]})
    chart = lambda d: C(d).mark_bar().encode(x="c", y=fm.Y("v", aggregate="median"))  # noqa: E731
    assert chart(frame).to_svg() == chart(twin).to_svg()


def test_count_of_a_field_skips_an_infinity_and_count_of_rows_does_not():
    """``count(v)`` counts the usable values of ``v`` (six of seven); ``count()`` counts rows."""
    frame = pl.DataFrame({"c": ["a"] * 7, "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, INF]})
    six = frame.head(6)
    by_field = lambda d: C(d).mark_bar().encode(x="c", y=fm.Y("v", aggregate="count"))  # noqa: E731
    assert by_field(frame).to_svg() == by_field(six).to_svg()
    by_rows = lambda d: C(d).mark_bar().encode(x="c", y="count()")  # noqa: E731
    assert by_rows(frame).to_svg() != by_rows(six).to_svg()


def test_a_boxen_over_values_and_an_infinity_is_the_boxen_of_the_values():
    """Forty values and an infinity: the depths and letter values are computed over the forty, so
    the same outlier circles draw as the null twin's."""
    rng = np.random.default_rng(7)
    values = [float(v) for v in rng.normal(0, 1, 40)] + [-6.0, 7.0]
    frame = pl.DataFrame({"c": ["a"] * 43, "v": [*values, INF]})
    chart = lambda d: C(d).mark_boxen().encode(x="c", y="v")  # noqa: E731
    got = _outcome(_static(chart(frame)))
    twin_got = _outcome(_static(chart(_null_twin(frame))))
    assert got == twin_got
    assert got.output.count("<circle") == twin_got.output.count("<circle") > 0


@pytest.mark.parametrize("value", [INF, -INF], ids=["pos", "neg"])
def test_a_boxen_whose_ramp_depth_depends_on_the_count_skips_an_infinity(value: float):
    """Group ``a`` holds 200 values and 200 infinities: its depth is that of 200 values, and a
    count of all 400 rows would draw one more level. The chart is the null twin's."""
    frame = pl.DataFrame(
        {"c": ["a"] * 400, "v": [float(k) for k in range(200)] + [value] * 200},
        schema={"c": pl.Utf8, "v": pl.Float64},
    )
    chart = lambda d: C(d).mark_boxen().encode(x="c", y="v")  # noqa: E731
    got, twin = _outcome(_static(chart(frame))), _outcome(_static(chart(_null_twin(frame))))
    _guard_against_a_refusal(twin, "mark_boxen")
    assert got == twin


@pytest.mark.parametrize("value", [INF, -INF], ids=["pos", "neg"])
def test_an_ecdf_counts_the_finite_values_only(value: float):
    """The cumulative count reaches six over six finite values and an infinity, not seven."""
    frame = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, value]})
    six = frame.filter(pl.col("v").is_finite())
    got = _outcome(_static(fm.displot(frame, x="v", kind="ecdf")))
    assert got == _outcome(_static(fm.displot(six, x="v", kind="ecdf")))


@pytest.mark.parametrize("value", [INF, -INF], ids=["pos", "neg"])
def test_a_qq_places_the_finite_values_only(value: float):
    """Plotting positions are computed over the six finite values, not over seven."""
    frame = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, value]})
    six = frame.filter(pl.col("v").is_finite())
    got = _outcome(_static(C(frame).mark_qq().encode(x="v")))
    assert got == _outcome(_static(C(six).mark_qq().encode(x="v")))


# ---------------------------------------------------------------------------
# A group, panel or call with no usable value: the outcome its all-null twin gets, in the
# words that name the usable values (non-null, finite).
# ---------------------------------------------------------------------------


def _hue_frame(values: Sequence[float | None]) -> pl.DataFrame:
    """Hue ``a`` of six finite values beside hue ``c`` of ``values``."""
    return pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, *values],
            "w": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, *[float(k) for k in range(len(values))]],
            "g": ["a"] * 6 + ["c"] * len(values),
        },
        schema={"v": pl.Float64, "w": pl.Float64, "g": pl.Utf8},
    )


@pytest.mark.parametrize("blank", [INF, -INF, None, NAN], ids=["pos", "neg", "null", "nan"])
def test_a_density_group_of_only_unusable_values_is_named_with_the_count(blank: float | None):
    """Every value of hue ``c`` is infinite (or null, or NaN): ``Kde`` names it with the count of
    usable values, never the bandwidth, and the group keeps its legend entry."""
    frame = _hue_frame([blank] * 3)
    svg, shown = _render_capturing(C(frame).mark_density().encode(x="v", color="g").to_svg)
    assert shown == [_fit_message("Kde", "g=c", FIT_FEW)]
    assert ">c<" in svg


@pytest.mark.parametrize("blank", [INF, None], ids=["inf", "null"])
def test_a_density_call_of_only_unusable_values_is_named_with_the_count(blank: float | None):
    frame = pl.DataFrame({"v": [blank] * 4}, schema={"v": pl.Float64})
    _, shown = _render_capturing(C(frame).mark_density().encode(x="v").to_svg)
    assert shown == [_whole_fit_message("Kde", FIT_FEW)]


def test_a_density_facet_panel_of_only_infinities_is_named_with_the_count():
    frame = pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0, INF, -INF, INF, INF], "f": ["p"] * 4 + ["q"] * 4}
    )
    _, shown = _render_capturing(C(frame).mark_density().encode(x="v").facet(col="f").to_svg)
    assert shown == [_panel_fit_message("Kde", FIT_FEW, "f=q")]


def _kde_and_violin(values: Sequence[float | None], kind: str) -> Any:
    frame = _hue_frame(values)
    if kind == "density":
        return C(frame).mark_density().encode(x="v", color="g")
    return C(frame.rename({"g": "c"})).mark_violin().encode(x="c", y="v")


_VIOLIN_INNER_BOX = _slot_message("mark_violin", "y", "c", "c")


@pytest.mark.parametrize(
    "values,expected_density,expected_violin",
    [
        pytest.param(
            [5.0, INF],
            [_fit_message("Kde", "g=c", FIT_FEW)],
            [_fit_message("Violin", "c=c", FIT_FEW)],
            id="one-finite-and-an-infinity",
        ),
        pytest.param(
            [INF, INF],
            [_fit_message("Kde", "g=c", FIT_FEW)],
            [_fit_message("Violin", "c=c", FIT_FEW)] + [_VIOLIN_INNER_BOX] * 5,
            id="only-infinities",
        ),
        pytest.param(
            [3.0, 3.0, INF],
            [_fit_message("Kde", "g=c", FIT_TIED)],
            [_fit_message("Violin", "c=c", FIT_TIED)],
            id="tied-and-an-infinity",
        ),
        pytest.param([-INF, 2.0, 7.0, INF], [], [], id="two-distinct-finite-values"),
    ],
)
@pytest.mark.parametrize("kind", ["density", "violin"])
def test_a_kde_never_names_the_bandwidth_for_an_infinity(
    kind: str,
    values: list[float],
    expected_density: list[str],
    expected_violin: list[str],
):
    """Whatever the group holds beside an infinity, the reason is the count or a tie, in the
    words of the same chart over nulls, and never the bandwidth."""
    expected = expected_density if kind == "density" else expected_violin
    _, shown = _render_capturing(_kde_and_violin(values, kind).to_svg)
    nulled = [None if abs(v) == INF else v for v in values]
    _, twin = _render_capturing(_kde_and_violin(nulled, kind).to_svg)
    assert shown == expected
    assert shown == twin


@pytest.mark.parametrize("scale", [1e308, 1e300], ids=["1e308", "1e300"])
def test_a_density_over_finite_values_whose_spread_overflows_keeps_the_bandwidth_reason(
    scale: float,
):
    """The bandwidth text stays for what it is true of: finite values whose spread overflows."""
    frame = pl.DataFrame({"v": [-scale, 0.0, scale]})
    _, shown = _render_capturing(C(frame).mark_density().encode(x="v").to_svg)
    assert shown == [_whole_fit_message("Kde", FIT_BANDWIDTH)]


def test_a_qq_group_of_only_infinities_is_named_as_an_all_null_group_is():
    frame = _hue_frame([INF, -INF, INF])
    svg, shown = _render_capturing(C(frame).mark_qq().encode(x="v", color="g").to_svg)
    assert shown == [_fit_message("QQ", "g=c", FIT_NONE)]
    assert ">c<" in svg


def test_a_qq_call_of_only_infinities_is_named_as_an_all_null_call_is():
    frame = pl.DataFrame({"v": [INF, -INF, INF]})
    _, shown = _render_capturing(C(frame).mark_qq().encode(x="v").to_svg)
    assert shown == [_whole_fit_message("QQ", FIT_NONE)]


def test_a_smooth_group_of_only_infinities_is_named_with_the_usable_point_count():
    frame = _hue_frame([INF, -INF, INF, INF])
    svg, shown = _render_capturing(
        C(frame).mark_smooth(method="lm").encode(x="v", y="w", color="g").to_svg
    )
    assert shown == [_fit_message("Smooth", "g=c", FIT_FEW_POINTS)]
    assert ">c<" in svg


def test_a_smooth_call_of_only_infinities_is_named_with_the_usable_point_count():
    frame = pl.DataFrame({"v": [INF, -INF, INF], "w": [1.0, 2.0, 3.0]})
    _, shown = _render_capturing(C(frame).mark_smooth(method="lm").encode(x="v", y="w").to_svg)
    assert shown == [_whole_fit_message("Smooth", FIT_FEW_POINTS)]


@pytest.mark.parametrize("method", ["lm", "loess"])
@pytest.mark.parametrize("column", ["v", "w"])
def test_a_fit_over_an_infinity_in_either_column_fits_the_finite_rows(column: str, method: str):
    """An infinity in x or in y skips that row: the chart is the one fitted over the other rows,
    and nothing is warned."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.5, 3.5]
    y = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 2.0, 4.0]
    finite = pl.DataFrame({"v": x, "w": y})
    row = {"v": {"v": [INF], "w": [3.0]}, "w": {"v": [3.0], "w": [INF]}}[column]
    infinite = pl.concat([finite, pl.DataFrame(row)])
    chart = lambda d: C(d).mark_smooth(method=method).encode(x="v", y="w")  # noqa: E731
    got = _outcome(_static(chart(infinite)))
    assert got.messages == []
    assert got.output == _outcome(_static(chart(finite))).output


@pytest.mark.parametrize(
    "build,expected",
    [
        pytest.param(
            lambda d: C(d).transform(fm.Robust(x="v", y="w")).mark_line().encode(x="x", y="y"),
            _whole_fit_message("Robust", FIT_FEW_POINTS),
            id="Robust",
        ),
        pytest.param(
            lambda d: (
                C(d).transform(fm.transform_regression("v", "w")).mark_line().encode(x="x", y="y")
            ),
            _whole_fit_message("RegressionData", FIT_FEW_POINTS),
            id="transform_regression",
        ),
        pytest.param(
            lambda d: C(d).transform(fm.transform_loess("v", "w")).mark_line().encode(x="x", y="y"),
            _whole_fit_message("LoessData", FIT_FEW_LOESS_POINTS),
            id="transform_loess",
        ),
    ],
)
def test_a_fit_transform_over_only_infinite_x_names_the_usable_point_count(
    build: Callable[[pl.DataFrame], Any], expected: str
):
    """Every fit counts its usable points the same way: a call whose x values are all infinite
    names the count, in the words that say what a usable value is."""
    _, shown = _render_capturing(build(_INPUTS["all-inf"]).to_svg)
    assert shown == [expected]


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("name", ["Logistic", "Glm"])
def test_a_logistic_or_glm_call_of_only_infinities_warns_and_draws_as_an_all_null_call_does(
    name: str, output: str
):
    """These two fits degrade a call with no usable observation as ``Robust`` does: an all-infinite
    call and its all-null twin each draw their frame with no fit and name the fit once, in the
    shared usable wording, on both output kinds."""
    frame = _INPUTS["all-inf"]
    render = {"static": _static, "interactive": _scene_of}[output]
    got = _outcome(render(_STATISTIC_DOORS[name].build(frame)))
    twin = _outcome(render(_STATISTIC_DOORS[name].build(_null_twin(frame))))
    robust = _outcome(render(_STATISTIC_DOORS["Robust"].build(frame)))
    _guard_against_a_refusal(twin, name)
    assert got.messages == [_whole_fit_message(name, FIT_FEW_POINTS)]
    assert twin.messages == got.messages
    assert robust.messages == [_whole_fit_message("Robust", FIT_FEW_POINTS)]
    assert got == twin
    assert got.output == robust.output


def test_a_two_dimensional_density_group_of_only_infinities_is_named_with_the_usable_count():
    frame = pl.DataFrame(
        {
            "x": [0.1, 1.2, 2.3, 3.1, 4.2, 5.3, INF, -INF, INF, INF],
            "y": [1.0, 0.2, 3.3, 2.1, 5.2, 4.1, 1.0, 2.0, INF, 3.0],
            "g": ["a"] * 6 + ["c"] * 4,
        }
    )
    _, shown = _render_capturing(C(frame).mark_contour().encode(x="x", y="y", color="g").to_svg)
    assert shown == [_fit_message("Kde2D", "g=c", FIT_FEW_PAIRS)]


@pytest.mark.parametrize("mark", ["mark_hex", "mark_raster"])
def test_a_hex_or_raster_call_of_only_infinities_is_named_with_the_usable_count(mark: str):
    frame = pl.DataFrame({"x": [INF, -INF, 1.0], "y": [1.0, INF, INF]})
    _, shown = _render_capturing(getattr(C(frame), mark)().encode(x="x", y="y").to_svg)
    assert shown == [_whole_fit_message("Hex" if mark == "mark_hex" else "Raster", FIT_NONE)]


@pytest.mark.parametrize(
    "mark,census",
    [
        pytest.param("mark_histogram", "mark_histogram", id="histogram"),
        pytest.param("mark_boxplot", "mark_boxplot", id="boxplot"),
    ],
)
def test_a_band_slot_or_hue_group_of_only_infinities_is_named_as_an_all_null_one_is(
    mark: str, census: str
):
    """The census sentence of an all-null group is the one an all-infinite group gets."""
    frame = _hue_frame([INF, -INF, INF])
    if mark == "mark_histogram":
        chart = C(frame).mark_histogram().encode(x="v", color="g")
        expected = _slot_message(census, "x", "c", "g")
    else:
        frame = frame.rename({"g": "c"})
        chart = C(frame).mark_boxplot().encode(x="c", y="v")
        expected = _slot_message(census, "y", "c", "c")
    # The box's layers each raise the one sentence; Python's own filter shows it once per call.
    svg, shown = _render_capturing(
        chart.to_svg, action="default" if mark == "mark_boxplot" else "always"
    )
    assert shown == [expected]
    assert ">c<" in svg


# ---------------------------------------------------------------------------
# A facet panel holding one infinity draws, and the frame's shared extent is kept.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda d: C(d).mark_histogram().encode(x="v").facet(col="f"), id="histogram"),
        pytest.param(lambda d: C(d).mark_density().encode(x="v").facet(col="f"), id="density"),
        pytest.param(lambda d: fm.displot(d, x="v", col="f"), id="displot"),
    ],
)
def test_a_facet_panel_holding_one_infinity_draws_and_keeps_the_shared_extent(
    build: Callable[[pl.DataFrame], Any],
):
    """Panel ``r`` holds one infinity beside finite values: it draws, nothing is warned, and the
    shared count (or density) axis is the one the null twin's panels share, which an infinity in
    the extent fold used to drop."""
    frame = _INPUTS["inf-in-group-c"]
    svg, shown = _render_capturing(build(frame).to_svg)
    twin_svg, twin_shown = _render_capturing(build(_null_twin(frame)).to_svg)
    assert shown == [] and twin_shown == []
    assert svg == twin_svg
    assert svg.count(">r<") >= 1


# ---------------------------------------------------------------------------
# Impute fills a null or NaN cell, never an infinite one, with the statistic of the finite values.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mean", "median", "min", "max"])
def test_impute_fills_a_null_cell_from_the_finite_values_and_keeps_an_infinite_cell(method: str):
    """The fill statistic skips the infinity, and the infinite cell is the caller's value, not a
    missing one: it is not filled."""
    frame = pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0, None, INF], "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        schema={"v": pl.Float64, "w": pl.Float64},
    )
    fill = {"mean": 2.5, "median": 2.5, "min": 1.0, "max": 4.0}[method]
    manual = frame.with_columns(pl.col("v").fill_null(fill))
    imputed = C(frame).transform(fm.transform_impute("v", method=method)).mark_point()
    assert _outcome(_static(imputed.encode(x="v", y="w"))) == _outcome(
        _static(C(manual).mark_point().encode(x="v", y="w"))
    )


# ---------------------------------------------------------------------------
# What keeps an infinity: ordering and selection, the plain marks, the log axis.
# ---------------------------------------------------------------------------


def test_a_filter_keeps_an_infinite_row_as_the_value_it_is():
    """``datum.v > 3`` is true of ``+inf``, so the filter keeps that row: its x is drawn."""
    frame = pl.DataFrame({"v": [1.0, 4.0, INF], "w": [1.0, 2.0, 3.0]})
    chart = C(frame).transform(fm.transform_filter("datum.v > 3")).mark_point().encode(x="w", y="w")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert svg.count("<circle") == 2


def test_a_rank_orders_an_infinite_value_last():
    """An infinity is an ordered value to a rank: every row has a rank, so every row draws."""
    frame = pl.DataFrame({"v": [3.0, 1.0, INF, 2.0], "w": [1.0, 2.0, 3.0, 4.0]})
    chart = (
        C(frame)
        .transform(fm.transform_window({"op": "rank", "as": "r"}, sort=["v"]))
        .mark_point()
        .encode(x="w", y="r")
    )
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert svg.count("<circle") == 4


def test_a_plain_mark_skips_an_infinity_on_a_linear_axis_without_a_word():
    frame = pl.DataFrame({"v": [1.0, 2.0, INF], "w": [1.0, 2.0, 3.0]})
    chart = lambda d: C(d).mark_point().encode(x="v", y="w")  # noqa: E731
    assert _outcome(_static(chart(frame))) == _outcome(_static(chart(_null_twin(frame))))


def test_an_infinity_on_a_log_scale_is_called_infinite():
    frame = pl.DataFrame({"v": [1.0, 2.0, 3.0, INF], "w": [1.0, 2.0, 3.0, 4.0]})
    chart = C(frame).mark_point().encode(x=fm.X("v", scale=fm.LogScale()), y="w")
    _, shown = _render_capturing(chart.to_svg)
    assert shown == [_infinite_warning("x", "v", 1)]


_LOG_DOORS: dict[str, Callable[[pl.DataFrame, fm.LogScale], Any]] = {
    "mark_boxplot": lambda d, sc: C(d).mark_boxplot().encode(x="c", y=fm.Y("v", scale=sc)),
    "mark_violin": lambda d, sc: C(d).mark_violin().encode(x="c", y=fm.Y("v", scale=sc)),
    "mark_boxen": lambda d, sc: C(d).mark_boxen().encode(x="c", y=fm.Y("v", scale=sc)),
    "mark_errorbar": lambda d, sc: C(d).mark_errorbar().encode(x="c", y=fm.Y("v", scale=sc)),
    "mark_smooth": lambda d, sc: C(d).mark_smooth(method="lm").encode(x=fm.X("v", scale=sc), y="w"),
    "mark_histogram": lambda d, sc: C(d).mark_histogram().encode(x=fm.X("v", scale=sc)),
    "mark_density": lambda d, sc: C(d).mark_density().encode(x=fm.X("v", scale=sc)),
}


#: The doors whose value axis is x, where the log axis itself reports a ``-inf`` row it cannot
#: place (an off-side value, as a negative one is). A category door's infinite row is no mark.
_LOG_X_DOORS = {"mark_smooth", "mark_histogram", "mark_density"}


@pytest.mark.parametrize("clamp", [False, True], ids=["unclamped", "clamped"])
@pytest.mark.parametrize("value", [INF, -INF], ids=["pos-inf", "neg-inf"])
@pytest.mark.parametrize("name", list(_LOG_DOORS))
def test_a_statistic_on_a_log_axis_over_an_infinity_equals_its_null_twin(
    name: str, value: float, clamp: bool
):
    """On a log axis an infinity is no usable value to the statistic, under a clamped scale too:
    the chart is the null twin's. The only difference in what is said is the axis's own count of
    a ``-inf`` row on an x axis, an off-side value the scale cannot place."""
    frame = _frame([*_FINITE[:8], value])
    scale = fm.LogScale(clamp=clamp)
    got = _outcome(_static(_LOG_DOORS[name](frame, scale)))
    twin = _outcome(_static(_LOG_DOORS[name](_null_twin(frame), scale)))
    _guard_against_a_refusal(twin, name)
    axis_report = [_log_message("x", "v", 1)] if value == -INF and name in _LOG_X_DOORS else []
    assert got.output == twin.output
    assert got.messages == twin.messages + axis_report


@pytest.mark.parametrize("value", [INF, -INF], ids=["pos-inf", "neg-inf"])
def test_an_ecdf_on_a_log_axis_over_an_infinity_equals_its_null_twin(value: float):
    """The ECDF keeps the infinite row as a step of its own, which the log axis reports (a
    ``-inf`` as off-side, a ``+inf`` as reaching infinity); the chart is the null twin's."""
    frame = _frame([*_FINITE[:8], value])
    chart = lambda d: fm.displot(d, x="v", kind="ecdf", log_scale=True)  # noqa: E731
    got, twin = _outcome(_static(chart(frame))), _outcome(_static(chart(_null_twin(frame))))
    _guard_against_a_refusal(twin, "displot ecdf")
    axis_report = (
        [_statistic_infinite_cut_warning("displot", "x", "v", 1)]
        if value == INF
        else [_log_message("x", "v", 1)]
    )
    assert got.output == twin.output
    assert got.messages == twin.messages + axis_report


def test_a_plain_mark_on_a_clamped_log_scale_draws_an_infinity_at_the_range_end():
    """The standing clamp ruling: ``+inf`` on a clamped log scale draws a plain mark at the end
    of the range, where the statistics above skip it."""
    frame = pl.DataFrame({"v": [1.0, 2.0, 3.0, INF], "w": [1.0, 2.0, 3.0, 4.0]})
    chart = C(frame).mark_point().encode(x=fm.X("v", scale=fm.LogScale(clamp=True)), y="w")
    svg, shown = _render_capturing(chart.to_svg)
    xs = [cx for cx, _, _ in parse_circles(svg)]
    assert shown == [] and len(xs) == 4
    assert xs[3] == max(xs) == xs[2], "the infinity sits at the end the largest value reaches"


# ---------------------------------------------------------------------------
# The standalone Outliers transform, the degree refusal and top_k's ranking read the same rule.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [INF, -INF], ids=["pos-inf", "neg-inf"])
def test_the_outliers_transform_skips_an_infinite_value_and_never_flags_its_row(value: float):
    """An infinite value is no usable value: the quartiles read the finite ones and an infinite
    row is not an outlier row, so the output is the null twin's and the real outlier remains."""
    frame = pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 2.0, 3.0, 2.0, value, 2.0, 3.0, 50.0], "g": ["a"] * 10},
        schema={"v": pl.Float64, "g": pl.Utf8},
    )
    chart = lambda d: C(d).transform(fm.Outliers("v")).mark_point().encode(x="g", y="v")  # noqa: E731
    got, twin = _outcome(_static(chart(frame))), _outcome(_static(chart(_null_twin(frame))))
    _guard_against_a_refusal(twin, "Outliers")
    assert got == twin
    assert got.output.count("<circle") == twin.output.count("<circle") > 0


def test_the_degree_refusal_counts_usable_rows_and_names_them_so():
    """Five of six rows are usable once the infinite x is skipped, so degree 5 is refused with the
    count of the rows that remain, in the words the other reasons use; the null twin is refused
    in the same words."""
    frame = pl.DataFrame({"v": [1.0, 2.0, 3.0, INF, 5.0, 6.0], "w": [1.0, 3.0, 2.0, 4.0, 6.0, 5.0]})
    chart = lambda d: C(d).mark_smooth(method="lm", degree=5).encode(x="v", y="w")  # noqa: E731
    got, twin = _outcome(_static(chart(frame))), _outcome(_static(chart(_null_twin(frame))))
    assert got == twin
    assert got.output == (
        "RAISE ValueError: transform failed: ValueError: Smooth: degree=5 needs at least 6 data "
        "points to fit a degree-5 polynomial; only 5 remain after dropping rows with no usable "
        "(non-null, finite) x/y (and any x_bins aggregation). A degree this high relative to the "
        "data is a rank-deficient design, not a valid fit — lower degree or provide more data"
    )


#: With no groupby the field is its own category, so an all-infinite call's two infinities are two
#: categories where its null twin has one: those two frames are not twins and are not compared.
_TOP_K_CASES = [
    pytest.param(
        input_name, k, groupby, id=f"{input_name}-{k}-{'grouped' if groupby else 'ungrouped'}"
    )
    for input_name in ("pos-inf", "inf-in-group-c", "group-c-all-inf", "all-inf")
    for k in (1, 2)
    for groupby in (None, ["c"])
    if not (groupby is None and input_name == "all-inf")
]


@pytest.mark.parametrize("input_name,k,groupby", _TOP_K_CASES)
def test_top_k_ranks_a_group_of_infinite_values_as_a_group_with_no_usable_value(
    input_name: str, k: int, groupby: list[str] | None
):
    """A group whose values are all infinite ranks as an all-null group does, and a group holding
    one infinity ranks by its finite values: the chart is the null twin's."""
    chart = lambda d: (  # noqa: E731
        C(d)
        .transform(fm.transform_top_k(k, field="v", op="sum", groupby=groupby))
        .mark_point()
        .encode(x="i", y="w")
    )
    frame = _INPUTS[input_name]
    got, twin = _outcome(_static(chart(frame))), _outcome(_static(chart(_null_twin(frame))))
    assert got == twin


# ---------------------------------------------------------------------------
# The two folds Python runs over a caller's column read the same usable values.
# ---------------------------------------------------------------------------


_FUNCTION_Y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0]


def _function_chart(x: Sequence[float | None], door: str) -> Any:
    """The point chart and the function line over ``x``, composed with ``+`` or chained."""
    frame = pl.DataFrame(
        {"x": list(x), "y": _FUNCTION_Y}, schema={"x": pl.Float64, "y": pl.Float64}
    )
    if door == "plus":
        return C(frame).mark_point().encode(x="x", y="y") + C(frame).mark_function(np.sin)
    return C(frame).mark_point().encode(x="x", y="y").mark_function(np.sin)


def _function_outcome(x: Sequence[float | None], door: str) -> _Outcome:
    """Build and render inside the capture, so a warning raised while the chart is built (a
    numpy one from evaluating the function over the domain) is seen."""
    return _outcome(lambda: _function_chart(x, door).to_svg())


@pytest.mark.parametrize("door", ["plus", "chained"])
@pytest.mark.parametrize("value", [INF, -INF, NAN], ids=["pos", "neg", "nan"])
def test_mark_function_takes_its_domain_from_the_finite_values(value: float, door: str):
    """One unusable value in the parent column leaves the curve over the finite values' range, as
    a null does, with no numpy warning and none from the render."""
    finite = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    got = _function_outcome([*finite, value], door)
    twin = _function_outcome([*finite, None], door)
    assert got.messages == [] and twin.messages == []
    assert got.output == twin.output
    assert "<polyline" in got.output


@pytest.mark.parametrize("door", ["plus", "chained"])
@pytest.mark.parametrize(
    "parent",
    [
        pytest.param([INF, -INF, INF, INF, -INF, INF, INF], id="all-infinite"),
        pytest.param([NAN] * 7, id="all-nan"),
    ],
)
def test_mark_function_over_a_parent_of_only_unusable_values_draws_nothing_and_says_so(
    parent: list[float], door: str
):
    """The curve has no domain to evaluate over: it draws nothing, as an all-null parent's does,
    and the census names the point layer and the function layer, once each."""
    got = _function_outcome(parent, door)
    twin = _function_outcome([None] * 7, door)
    assert got == twin
    assert got.messages == [
        _layer_message("layer 0: mark_point", "x"),
        _layer_message("mark_function", "x and y"),
    ]
    assert "<polyline" not in got.output


_T0 = dt.datetime(2020, 1, 1)
_DAYS = [_T0 + dt.timedelta(days=k) for k in range(5)]
_EPOCH_MS = (1577836800000.0, 1578182400000.0)


@pytest.mark.parametrize(
    "parent,domain",
    [
        pytest.param([1, 2, 3, 4, 5], (1.0, 5.0), id="int"),
        pytest.param(_DAYS, _EPOCH_MS, id="datetime"),
        pytest.param([d.date() for d in _DAYS], _EPOCH_MS, id="date"),
        pytest.param([_DAYS[0], None, *_DAYS[2:]], _EPOCH_MS, id="datetime-with-null"),
        pytest.param(
            [dt.timedelta(days=k) for k in range(1, 6)], (86400000.0, 432000000.0), id="duration"
        ),
    ],
)
def test_mark_function_reads_its_domain_in_the_parent_axis_units(
    parent: list[Any], domain: tuple[float, float]
):
    """A numeric parent is read in its own units and a temporal one in epoch milliseconds, the
    units a temporal axis reads: the chart is the one given that domain explicitly."""
    frame = pl.DataFrame({"t": parent, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})

    def chart(**kwargs: Any) -> Any:
        return C(frame).mark_point().encode(x="t", y="y").mark_function(np.sin, **kwargs)

    got = _outcome(lambda: chart().to_svg())
    explicit = _outcome(lambda: chart(domain=domain).to_svg())
    assert got.messages == []
    assert got == explicit


@pytest.mark.parametrize(
    "parent",
    [
        pytest.param(["a", "b", "c", "d", "e"], id="text"),
        pytest.param(["1", "2", "3", "4", "5"], id="numeric-text"),
        pytest.param([True, False, True, False, True], id="boolean"),
    ],
)
def test_mark_function_refuses_a_parent_it_cannot_read_a_domain_from(parent: list[Any]):
    frame = pl.DataFrame({"t": parent, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(ValueError) as refused:
        C(frame).mark_point().encode(x="t", y="y").mark_function(np.sin)
    assert str(refused.value) == (
        "mark_function: x column 't' holds neither numbers nor times, so no domain can be read "
        "from it; pass domain=(x_min, x_max)"
    )


def test_mark_function_with_a_domain_over_a_text_parent_still_draws():
    """The refusal names ``domain=``, so that door renders."""
    frame = pl.DataFrame({"t": ["a", "b", "c", "d", "e"], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    chart = C(frame).mark_point().encode(x="t", y="y").mark_function(np.sin, domain=(0.0, 4.0))
    assert "<polyline" in chart.to_svg()


# ---------------------------------------------------------------------------
# OutlierLabel reads the usable values of its field.
# ---------------------------------------------------------------------------


def _added_texts(base: Any, label: Any) -> list[str]:
    """The sorted texts ``label`` adds to ``base``'s render."""
    with_label = Counter(text for _, text in text_elements((base + label).to_svg()))
    without = Counter(text for _, text in text_elements(base.to_svg()))
    return sorted((with_label - without).elements())


def _outlier_base(y: Sequence[float | None]) -> Any:
    frame = pl.DataFrame(
        {"x": [float(k) for k in range(len(y))], "y": list(y)},
        schema={"x": pl.Float64, "y": pl.Float64},
    )
    return C(frame).mark_point().encode(x="x", y="y")


@pytest.mark.parametrize("max_labels", [10, 1])
@pytest.mark.parametrize("value", [INF, -INF, NAN], ids=["pos", "neg", "nan"])
def test_outlier_label_labels_the_outlier_and_never_an_unusable_row(value: float, max_labels: int):
    """The mean and deviation read the usable values, so the real outlier is labeled whether an
    infinity or a NaN sits beside it or a null does, and the unusable row never takes the label's
    place (at ``max_labels=1`` it would take the only one)."""
    label = fm.OutlierLabel(field="y", threshold=1.5, max_labels=max_labels)
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 30.0]
    assert _added_texts(_outlier_base([*values, value]), label) == ["30.0"]
    assert _added_texts(_outlier_base([*values, None]), label) == ["30.0"]


@pytest.mark.parametrize(
    "y",
    [
        pytest.param([None] * 5, id="all-null"),
        pytest.param([NAN] * 5, id="all-nan"),
        pytest.param([INF, -INF, INF, INF, -INF], id="all-infinite"),
        pytest.param([], id="zero-row"),
    ],
)
def test_outlier_label_over_a_field_with_no_usable_value_returns_the_chart_unlabeled(
    y: list[float | None],
):
    """With no usable value there is nothing to score: the chart is the base chart's, in the
    words the base chart's own render says."""
    got = _outcome(lambda: (_outlier_base(y) + fm.OutlierLabel(field="y", threshold=1.5)).to_svg())
    base = _outcome(_outlier_base(y).to_svg)
    assert got == base
    assert got.messages == ([_layer_message("mark_point", "y")] if y else [_NO_ROWS_TO_DRAW])


def test_outlier_label_over_one_usable_value_returns_the_chart_unlabeled():
    """One usable value has no deviation: it reads as 1.0, so nothing scores above a positive
    threshold and the chart is the base chart's."""
    y = [3.0, None, INF, NAN, None]
    got = _outcome(lambda: (_outlier_base(y) + fm.OutlierLabel(field="y", threshold=1.5)).to_svg())
    assert got == _outcome(_outlier_base(y).to_svg)
    assert got.messages == []


_NON_FLOAT_FIELDS = {
    "int": (pl.Int64, [1, 2, 3, 4, 5, 30]),
    "uint8": (pl.UInt8, [1, 2, 3, 4, 5, 30]),
    "int16": (pl.Int16, [1, 2, 3, 4, 5, 30]),
    "float32": (pl.Float32, [1.0, 2.0, 3.0, 4.0, 5.0, 30.0]),
    "decimal": (pl.Decimal(10, 2), [1, 2, 3, 4, 5, 30]),
    "boolean": (pl.Boolean, [True, True, True, True, True, False]),
}


@pytest.mark.parametrize("kind", list(_NON_FLOAT_FIELDS))
def test_outlier_label_reads_a_field_of_any_dtype(kind: str):
    """A field that is not a float64 has its own usable values (non-null, and finite on a float
    width): the outlier is labeled in the field's own text, and nothing raises."""
    dtype, values = _NON_FLOAT_FIELDS[kind]
    frame = pl.DataFrame(
        {
            "x": [float(k) for k in range(6)],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 30.0],
            "d": pl.Series(values).cast(dtype),
        }
    )
    base = C(frame).mark_point().encode(x="x", y="y")
    label = fm.OutlierLabel(field="d", threshold=1.5)
    assert _added_texts(base, label) == [str(frame["d"][5])]


# ---------------------------------------------------------------------------
# AUCLabel and APLabel read a caller's curve through its usable points.
# ---------------------------------------------------------------------------


def _curve_frame(
    *, b_x: Sequence[float | None] | None = None, b_y: Sequence[float | None] | None = None
):
    """Two grouped curves: ``a`` of three points, ``b`` of four, ``b``'s own values overridable."""
    bx = b_x if b_x is not None else [0.0, 0.5, 0.7, 1.0]
    by = b_y if b_y is not None else [0.0, 0.6, 0.8, 1.0]
    return pl.DataFrame(
        {
            "x": [0.0, 0.5, 1.0, *bx],
            "y": [0.0, 0.7, 1.0, *by],
            "g": ["a"] * 3 + ["b"] * 4,
        },
        schema={"x": pl.Float64, "y": pl.Float64, "g": pl.Utf8},
    )


def _curve_base(frame: pl.DataFrame) -> Any:
    return C(frame).mark_line().encode(x="x", y="y", color="g")


_LABEL_DOORS = [
    pytest.param(fm.AUCLabel(position="end"), "y", ["AUC = 0.550", "AUC = 0.600"], id="auc-end-y"),
    pytest.param(
        fm.AUCLabel(position="corner"), "y", ["AUC = 0.550", "AUC = 0.600"], id="auc-corner-y"
    ),
    pytest.param(fm.APLabel(position="end"), "y", ["AP = 0.800", "AP = 0.850"], id="ap-end-y"),
    pytest.param(fm.AUCLabel(position="end"), "x", ["AUC = 0.550", "AUC = 0.600"], id="auc-end-x"),
]


@pytest.mark.parametrize("label,column,texts", _LABEL_DOORS)
@pytest.mark.parametrize("value", [INF, -INF], ids=["pos-inf", "neg-inf"])
def test_a_metric_label_over_a_curve_holding_an_infinity_is_its_null_twins(
    label: Any, column: str, texts: list[str], value: float
):
    """One infinite point of series ``b`` is left out of its integral and its anchor, as a null
    one is: both curves draw their label, in the words of the null twin, with nothing warned."""
    bad = {"b_x": [0.0, 0.5, value, 1.0]} if column == "x" else {"b_y": [0.0, 0.6, value, 1.0]}
    frame = _curve_frame(**bad)
    nulled = _curve_frame(
        **{k: [None if abs(v) == INF else v for v in vs] for k, vs in bad.items()}
    )
    got = _outcome(lambda: (_curve_base(frame) + label).to_svg())
    twin = _outcome(lambda: (_curve_base(nulled) + label).to_svg())
    assert got == twin
    assert got.messages == []
    assert _added_texts(_curve_base(frame), label) == texts


def test_a_metric_label_reads_an_infinite_x_as_a_null_one() -> None:
    """An ungrouped curve with an extra ``(inf, 0.5)`` point labels its integral over the
    usable points, where the label used to be anchored at the infinite x and not drawn."""
    frame = pl.DataFrame(
        {"x": [0.0, 0.1, 0.2, 0.4, 0.6, 1.0, INF], "y": [0.0, 0.4, 0.6, 0.8, 0.9, 1.0, 0.5]}
    )
    base = C(frame).mark_line().encode(x="x", y="y")
    label = fm.AUCLabel()
    got = _outcome(lambda: (base + label).to_svg())
    twin = _outcome(
        lambda: (C(_null_twin(frame)).mark_line().encode(x="x", y="y") + label).to_svg()
    )
    assert got == twin
    assert got.messages == []
    assert _added_texts(base, label) == ["AUC = 0.760"]


@pytest.mark.parametrize(
    "y",
    [
        pytest.param([None, None, None], id="all-null"),
        pytest.param([INF, INF, -INF], id="all-infinite"),
    ],
)
def test_a_metric_label_over_a_curve_with_no_usable_point_shows_nan(y: list[float | None]):
    """A curve with no usable point has no integral: the label reads ``nan`` where it raised or
    blanked, and the census names the empty line once."""
    frame = pl.DataFrame({"x": [0.0, 0.5, 1.0], "y": y}, schema={"x": pl.Float64, "y": pl.Float64})
    base = C(frame).mark_line().encode(x="x", y="y")
    label = fm.AUCLabel()
    got = _outcome(lambda: (base + label).to_svg())
    null_frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("y"))
    twin = _outcome(lambda: (C(null_frame).mark_line().encode(x="x", y="y") + label).to_svg())
    assert got == twin
    assert got.messages == [_layer_message("layer 0: mark_line", "y")]
    # An empty line has no axis of its own, so the label's layer also moves the tick texts: the
    # metric text is the one that carries its words.
    assert [t for t in _added_texts(base, label) if " = " in t] == ["AUC = nan"]


def _label_rows(chart: Any) -> list[list[int]]:
    """The data rows each text batch of the interactive scene draws."""
    scene = json.loads(chart.interactive().scene_json)
    return [
        batch["data_indices"]
        for panel in scene["panels"]
        for batch in panel["marks"]
        if batch["kind"] == "text"
    ]


@pytest.mark.parametrize("value", [INF, NAN, None], ids=["infinite", "nan", "null"])
def test_an_ungrouped_metric_label_is_carried_by_the_row_of_the_largest_usable_x(
    value: float | None,
):
    """The label rides the first row of the largest usable x, so an unusable x beside it (which a
    plain maximum would pick when infinite) never carries it."""
    frame = pl.DataFrame(
        {"x": [0.0, 0.5, value, 1.0], "y": [0.0, 0.6, 0.7, 1.0]},
        schema={"x": pl.Float64, "y": pl.Float64},
    )
    chart = C(frame).mark_line().encode(x="x", y="y") + fm.AUCLabel()
    assert _label_rows(chart) == [[3]]


_UNUSABLE_X = {
    "null": [None, None, None],
    "infinite": [INF, -INF, INF],
    "nan": [NAN, NAN, NAN],
}


def _no_x_frame(x: list[float | None], *, grouped: bool) -> pl.DataFrame:
    frame = pl.DataFrame({"x": x, "y": [0.0, 0.5, 1.0]}, schema={"x": pl.Float64, "y": pl.Float64})
    return frame.with_columns(pl.lit("a").alias("g")) if grouped else frame


@pytest.mark.parametrize("label", [fm.AUCLabel(), fm.APLabel()], ids=["auc", "ap"])
@pytest.mark.parametrize("grouped", [False, True], ids=["ungrouped", "grouped"])
@pytest.mark.parametrize("kind", list(_UNUSABLE_X))
def test_a_metric_label_over_a_curve_with_no_usable_x_shows_nan_at_the_default_anchor(
    kind: str, grouped: bool, label: Any
):
    """A curve whose every x is null, infinite or NaN has no usable point: its label reads ``nan``
    at the anchor an unplaced curve takes, grouped or not, and the only message is the curve's
    own census, never a sentence naming the label's text layer."""
    frame = _no_x_frame(_UNUSABLE_X[kind], grouped=grouped)
    base = (
        C(frame).mark_line().encode(x="x", y="y", color="g")
        if grouped
        else C(frame).mark_line().encode(x="x", y="y")
    )
    got = _outcome(lambda: (base + label).to_svg())
    word = "AUC" if isinstance(label, fm.AUCLabel) else "AP"
    texts = [(attrs, text) for attrs, text in text_elements(got.output) if " = " in text]
    assert [text for _, text in texts] == [f"{word} = nan"]
    assert got.messages == [
        _layer_message("layer 0: mark_line", "x")
        if not grouped
        else _slot_message("layer 0: mark_line", "x", "a", "g")
    ]
    # One position for the three kinds of unusable x: the label sits at the anchor of an
    # unplaced curve, not wherever a row happens to carry it.
    other = _outcome(
        lambda: (
            C(_no_x_frame(_UNUSABLE_X["null"], grouped=grouped))
            .mark_line()
            .encode(x="x", y="y", **({"color": "g"} if grouped else {}))
            + label
        ).to_svg()
    )
    assert [t for t in text_elements(other.output) if " = " in t[1]] == texts


@pytest.mark.parametrize("kind", list(_UNUSABLE_X))
@pytest.mark.parametrize("mark", ["mark_roc", "mark_pr"])
@pytest.mark.parametrize("grouped", [False, True], ids=["ungrouped", "grouped"])
def test_a_curve_annotation_over_a_curve_with_no_usable_x_shows_nan(
    grouped: bool, mark: str, kind: str
):
    x_name, y_name = ("fpr", "tpr") if mark == "mark_roc" else ("recall", "precision")
    frame = pl.DataFrame(
        {x_name: _UNUSABLE_X[kind], y_name: [0.0, 0.5, 1.0], "class": ["0"] * 3},
        schema={x_name: pl.Float64, y_name: pl.Float64, "class": pl.Utf8},
    )
    if not grouped:
        frame = frame.drop("class")
    kwargs = {"reference_line": False} if mark == "mark_roc" else {}
    got = _outcome(
        lambda: (
            getattr(C(frame), mark)(annotate_auc=True, **kwargs).to_svg()
            if mark == "mark_roc"
            else getattr(C(frame), mark)(annotate_ap=True).to_svg()
        )
    )
    word = "AUC" if mark == "mark_roc" else "AP"
    texts = [text for _, text in text_elements(got.output) if " = " in text]
    # An annotation names a curve by its class, so a frame with no class column draws none, with
    # usable points or without: that is unchanged.
    assert texts == ([f"0: {word} = nan"] if grouped else [])
    assert not any("mark_text" in m for m in got.messages)


def test_a_curve_annotation_integrates_the_usable_points():
    """``mark_roc(annotate_auc=True)`` over a hand-built frame integrates each class's usable
    points: an infinite tpr is left out, as a null one is."""
    frame = pl.DataFrame(
        {
            "fpr": [0.0, 0.5, 1.0, 0.0, 0.25, 0.5, 1.0],
            "tpr": [0.0, 0.6, 1.0, 0.0, INF, 0.7, 1.0],
            "class": ["0"] * 3 + ["1"] * 4,
        }
    )
    chart = lambda d: C(d).mark_roc(reference_line=False, annotate_auc=True)  # noqa: E731
    got = _outcome(lambda: chart(frame).to_svg())
    twin = _outcome(lambda: chart(_null_twin(frame)).to_svg())
    assert got == twin
    texts = Counter(text for _, text in text_elements(got.output))
    assert texts["0: AUC = 0.550"] == 1 and texts["1: AUC = 0.600"] == 1


# ---------------------------------------------------------------------------
# parallel_coordinates_chart's rescale reads each feature's usable values.
# ---------------------------------------------------------------------------


def _features_frame(a_last: float | None) -> pl.DataFrame:
    return pl.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, a_last], "b": [5.0, 3.0, 4.0, 1.0, 2.0]},
        schema={"a": pl.Float64, "b": pl.Float64},
    )


@pytest.mark.parametrize("value", [INF, -INF, NAN], ids=["pos-inf", "neg-inf", "nan"])
@pytest.mark.parametrize("rescale", ["minmax", "zscore", None])
def test_parallel_coordinates_rescale_skips_an_unusable_value_as_a_null_one_is(
    rescale: str | None, value: float
):
    chart = lambda d: fm.parallel_coordinates_chart(d, features=["a", "b"], rescale=rescale)  # noqa: E731
    got = _outcome(lambda: chart(_features_frame(value)).to_svg())
    twin = _outcome(lambda: chart(_features_frame(None)).to_svg())
    _guard_against_a_refusal(twin, "parallel_coordinates_chart")
    assert got == twin


def test_the_parallel_coordinates_visualizer_rescales_the_usable_values():
    chart = lambda d: fm.ParallelCoordinatesVisualizer(rescale="zscore").fit(d).show()  # noqa: E731
    got = _outcome(lambda: chart(_features_frame(INF)).to_svg())
    twin = _outcome(lambda: chart(_features_frame(None)).to_svg())
    _guard_against_a_refusal(twin, "ParallelCoordinatesVisualizer")
    assert got == twin


def test_pairplot_dropna_keeps_an_infinite_row():
    """``dropna`` is a row selection: it drops a row holding a null and keeps an infinite one, so
    over a frame with no null it changes nothing."""
    frame = _INPUTS["pos-inf"]
    chart = lambda dropna: fm.pairplot(frame, vars=["v", "w"], dropna=dropna)  # noqa: E731
    assert _outcome(lambda: chart(True).to_svg()) == _outcome(lambda: chart(False).to_svg())


# ---------------------------------------------------------------------------
# An ordering or pass-through transform keeps an infinity as the value it is.
# ---------------------------------------------------------------------------

_PASS_THROUGH = pl.DataFrame(
    {
        "i": [1.0, 2.0, 3.0, 4.0, 5.0],
        "v": [INF, 1.0, 3.0, 4.0, -INF],
        "g": ["a", "a", "b", "b", "b"],
    }
)


@pytest.mark.parametrize(
    "transform,filters,rows",
    [
        pytest.param(
            fm.transform_window({"op": "lag", "field": "v", "as": "o"}, sort=["i"]),
            ["datum.o > 100"],
            1,
            id="lag",
        ),
        pytest.param(
            fm.transform_window({"op": "lead", "field": "v", "as": "o"}, sort=["i"]),
            ["datum.o < -100"],
            1,
            id="lead",
        ),
        pytest.param(
            fm.transform_window(
                {"op": "first_value", "field": "v", "as": "o"}, sort=["i"], groupby=["g"]
            ),
            ["datum.o > 100"],
            2,
            id="first_value",
        ),
        pytest.param(
            fm.transform_window(
                {"op": "last_value", "field": "v", "as": "o"}, sort=["i"], groupby=["g"]
            ),
            ["datum.o < -100"],
            3,
            id="last_value",
        ),
        pytest.param(
            fm.transform_window({"op": "row_number", "as": "o"}, sort=["v"]),
            ["datum.o == 1", "datum.v < -100"],
            1,
            id="row_number-first",
        ),
        pytest.param(
            fm.transform_window({"op": "row_number", "as": "o"}, sort=["v"]),
            ["datum.o == 5", "datum.v > 100"],
            1,
            id="row_number-last",
        ),
        pytest.param(
            fm.transform_window({"op": "dense_rank", "as": "o"}, sort=["v"]),
            ["datum.o == 5", "datum.v > 100"],
            1,
            id="dense_rank-last",
        ),
        pytest.param(
            fm.transform_calculate("o", "datum.v * 2"), ["datum.o > 100"], 1, id="calculate-pos"
        ),
        pytest.param(
            fm.transform_calculate("o", "datum.v * 2"), ["datum.o < -100"], 1, id="calculate-neg"
        ),
    ],
)
def test_a_pass_through_transform_keeps_an_infinity_as_the_value_it_is(
    transform: dict, filters: list[str], rows: int
):
    """The filter reads the transform's own output, so a row whose infinite value was skipped
    (read as a null) would not pass it and would draw nothing."""
    chart = C(_PASS_THROUGH).transform(transform)
    for predicate in filters:
        chart = chart.transform(fm.transform_filter(predicate))
    got = _outcome(_static(chart.mark_point().encode(x="i", y="i")))
    assert got.messages == []
    assert got.output.count("<circle") == rows


# ---------------------------------------------------------------------------
# A numeric column of any float width skips an infinity, and a count of a non-numeric one is
# unchanged.
# ---------------------------------------------------------------------------

_FLOAT32_DOORS = [
    "mark_bar count",
    "mark_bar distinct",
    "mark_bar mean",
    "transform_aggregate count",
    "transform_join_aggregate count",
    "mark_bar binned count of a field",
]


@pytest.mark.parametrize(
    "input_name", ["pos-inf", "inf-in-group-c", "inf-beside-null-and-nan", "group-c-all-inf"]
)
@pytest.mark.parametrize("name", _FLOAT32_DOORS)
def test_a_float32_column_skips_an_infinity_as_a_null_one_is(name: str, input_name: str):
    """The count, like every other statistic, reads a ``Float32`` column's usable values only."""
    frame = _INPUTS[input_name].with_columns(pl.col("v").cast(pl.Float32))
    got = _outcome(_static(_ALL_DOORS[name].build(frame)))
    twin = _outcome(_static(_ALL_DOORS[name].build(_null_twin(frame))))
    _guard_against_a_refusal(twin, name)
    assert got == twin


def test_a_count_of_a_text_column_counts_its_non_null_values():
    """A column that holds no NaN or infinity counts its non-null values: group ``a`` holds one
    text and one null, so it counts one, the bar the frame without the null row draws."""
    frame = pl.DataFrame({"c": ["a", "a", "b"], "t": ["x", None, "y"]})
    chart = lambda d: C(d).mark_bar().encode(x="c", y=fm.Y("t", aggregate="count"))  # noqa: E731
    got = _outcome(_static(chart(frame)))
    dropped = _outcome(_static(chart(frame.filter(pl.col("t").is_not_null()))))
    assert got.messages == []
    assert got == dropped
    assert _rect_heights(got.output)[0] == _rect_heights(got.output)[1]


# ---------------------------------------------------------------------------
# A chart with no infinite value is unchanged, byte for byte.
# ---------------------------------------------------------------------------


def _digest(svg: str) -> str:
    return hashlib.sha256(svg.encode()).hexdigest()


#: SHA-256 of each door's render over the finite frame, recorded before the rule was added: a chart
#: with no infinity renders the bytes it always did.
_FINITE_DIGESTS: dict[str, str] = {
    "mark_histogram": "391cacb4e1581190afeea25a267e7b809b70a6c7e2222180fad6b3f14527c528",
    "mark_histogram hue": "cda6fdb502f0e3802520165278b2dde6ee6b71e6e01e428bd96bfaa048dd2fe6",
    "mark_histogram facet": "4566485d5fcc26e481c581d83168f9e22634123ca4734a394aba840edfd4bc97",
    "mark_bar binned count": "391cacb4e1581190afeea25a267e7b809b70a6c7e2222180fad6b3f14527c528",
    "displot hist": "391cacb4e1581190afeea25a267e7b809b70a6c7e2222180fad6b3f14527c528",
    "displot hist hue": "cda6fdb502f0e3802520165278b2dde6ee6b71e6e01e428bd96bfaa048dd2fe6",
    "mark_density": "bebeeaa98619f3736d922fa342e1949b129525f406d49ab94984698b92792531",
    "mark_density hue": "a06637d39ea209f21ec694914e1b45c4d4d7636b1906a37c7eb6a51849ad774e",
    "mark_density facet": "189c2ba11040a94a2f949d2ec72ec6ef671fc5e6bfa435cccb2f101e2c92a4b5",
    "displot kde": "bebeeaa98619f3736d922fa342e1949b129525f406d49ab94984698b92792531",
    "displot kde hue": "a06637d39ea209f21ec694914e1b45c4d4d7636b1906a37c7eb6a51849ad774e",
    "displot ecdf": "dc9cbcc559497252aae64be00027c0f4ec206b1609f65a566aff0c6b9f0ec59d",
    "displot ecdf hue": "17a7d3c7a277360afa077aee852e291f38f16fc61c56a4d10867258a6f5c2593",
    "mark_qq": "b50e23f73972cd553432efdb43be9fbd5bd88ace838dd8a8a556c6e6a2b4c60b",
    "mark_qq hue": "c231802fdbe00ec7c8c18dc082bc75b40e8e263f3ac16b57cd829f545d78b0f7",
    "mark_boxplot": "358102d79010255b20efe5144c737b3c311be1a18b2e7c8e90a9727864df9cb6",
    "mark_violin": "72edb65a01cc02816d6f466d4299d2d0045c99ce47caf303f27beacf990ec2aa",
    "mark_boxen": "3aa011fe6c8b4c026bf8415fd4ea9d13cd46280865af0e7bd4b2ca71301722bb",
    "mark_errorbar": "f24096c4275f1015c775d3148b3b54e34b130b15eedd599df830b35c7dcd8a74",
    "catplot box": "358102d79010255b20efe5144c737b3c311be1a18b2e7c8e90a9727864df9cb6",
    "catplot point": "b45cd57e7ae4c457c0764a1521b7f304dbf409b954594faa4ac300e22c8e62d7",
    "catplot bar": "cd36832218ae5c9117a1ccf4369979ca332bb23b5b3a8981e321bffa37e28daa",
    "mark_bar mean": "c0fb32a4b3934f2de5b87cd3c5fec8ea96462843f590e507eb739b66f8daed0c",
    "mark_bar sum": "698b3d1ca52934dbc36f4182a635910ac383d8d94d7202f165dccff3cb732416",
    "mark_bar median": "a34971fc0f34740aa3c14b723b9308e5fab99aa72d0f515374b490bb13caa2c5",
    "mark_bar min": "4e09b1efad24502b5843d02dba90d0eaa27ad74c5c6ab0ad5ab7ea3884319174",
    "mark_bar max": "093b7f52cfeb5816d5a2eed1f8999a8722d42c28916b6c3b716932c6b20f7a39",
    "mark_bar stdev": "72bd822445fe386a160f865a086befc4f6f1d70070fdb737939fc197276c6c68",
    "mark_bar count": "95a928d5502d0a7516be758538fb36abf32405ea24bc0a809ee5a72d4f4aa18f",
    "mark_bar distinct": "7723f5980ea34c5f12e735b091b346ab7d677e23d5be45711c4d7cb1a2cd0fd7",
    "mark_smooth lm x": "7efeb1451de23758f78dadfd3acc74e4a9cdbd24bf7ca16e33afeb1be71b0980",
    "mark_smooth lm y": "0513bbd6053d592b850b3235bfe22baf361461c758d67a7b3e465413098a43cf",
    "mark_smooth loess": "2395e62610e8e358da4017ba62011b2fcca428eca78bb544b4ca51ccdc6aff41",
    "mark_smooth hue": "f7b782b2853bc70992cb54272ac6493b417ac840ec63b8500f10a88ce5fe1668",
    "regplot x": "664a257ee73d5cc5819d64c5cf89e08879d5ebb955f4c6ab8331a864f35e4eca",
    "regplot y": "f2ef5b754a9e4d3c3637d76283474499ac2abf8e8f2203882c90917ef2812cf5",
    "lmplot": "d4c63aaba622b51c6677a14878081cfe76a2675ccdd0caa5fae9f94add123d72",
    "residplot": "ccb32191ab4407243cee8f6cb1990ee861dfc050f8fe8169980af79bfe539a9e",
    "Robust": "e25173df11c02b0be86636ab303aaf6139c45bed8afa97e7b32db50de3c0c546",
    "Logistic": "16deb4480541db40245ce04afbeb6ed871316034f387740c0f8d08b2aea32f2c",
    "Glm": "89c29502e2a8ba91bb737571152e1bee167156174e1c62dfaef86eab6ba70b55",
    "transform_regression": "e25173df11c02b0be86636ab303aaf6139c45bed8afa97e7b32db50de3c0c546",
    "transform_loess": "2dbf27d7b690778202fcc46b8e4c4b7a6ce19a945c2bc08abc6a663585ea6b7b",
    "mark_contour": "473503049c3bfd3ff3e4929db7c733226ddf012b811f4d609184c92d40c15097",
    "mark_hex": "d62f8bbaaad9dea9223b4ec4a79f021f5a1d328494e6b3d250c54c62bd851059",
    "mark_raster": "288b36880b3cd516d594aadee62940ee6a2dd0383e406efaba133c52a3c47a6b",
    "jointplot hex": "3f345cada87622415194ff9072a03fecce66358f53f9533e4c340dddaf58f1f0",
    "jointplot hist": "dbc86e737029a059b4f456b42d0370fa280716679cea9fb3c710bf06912eafae",
    "jointplot kde": "d015368dd10d790e82af783c9fb74f410c3b3c41017ef9ba9716af98f8ac445c",
    "jointplot scatter": "733b83ee4a0a7ed83ff06413689c53aa92be0191f0310545cfd045d5ab11e6bf",
    "mark_point": "a1dd556295fd02004bccf04453e548d4e71a05a47c860f120485d68a58be12d9",
    "mark_swarm": "bcfe2022722055b243e8c6b359fd7074d347a60db1c490020223acc85e43eee4",
    "transform_aggregate sum": "504786a819e855efa38ea52763bac8fc4ef647ce328916b8a1553465c57152c0",
    "transform_aggregate mean": "517920cc1d300047947d543ac9f6877080ba17434c82820e8f6b00946807d049",
    "transform_aggregate count": "589859f5ef1844bca0ef120198b28fe04cb757420f49d45e9790b813dcd331ec",
    "transform_join_aggregate": "89463004c78beb035927eadfff1a074ad4216086fecda992781745c85ce4219f",
    "transform_window rolling sum": "09d6014a62be0f181eacc135e10fdafb96b55832777fe482ff4d8f652650eca3",
    "transform_window rolling mean": "6bb8a123bf0f38a951f92dd6978623a83453d824e36873fe3659e191c39cc4be",
    "transform_window rolling count": "365279ecf1fb8f714a4c5323b9cb49d6b848ba3b158fb862779e6d56c79ef52b",
    "transform_stack": "101f401884ae3e04b997116ca53655751e50cca9712f39ec939466f8a51ee2dd",
    "transform_density": "8d12c4d5e66b1fd6b8f77ef2450aa1664e0891958580dd03a1b083d85aafc4bb",
    "transform_bin": "08a95a1d2c8be65580b42e633c90a3c7d90d832d69af9be5f3c988cabed9a011",
    "transform_top_k": "039165f1c9b0df448dd831f712320ee840bf4b4c588fa224ad8c945c5c059cc0",
    "transform_pivot": "a5f729e87ad7e55eda62e2f6d9b1ded5a8bf87993daf9de39a1d39a8c788b698",
}


#: A door added after the rule was recorded has no digest from before it, so the guard covers the
#: doors that do.
_DIGEST_CASES = [pytest.param(name, id=name) for name in _FINITE_DIGESTS]


@pytest.mark.parametrize("name", _DIGEST_CASES)
def test_a_chart_with_no_infinite_value_is_byte_identical(name: str):
    svg, shown = _render_capturing(_static(_ALL_DOORS[name].build(_frame(_FINITE))))
    assert shown == []
    assert _digest(svg) == _FINITE_DIGESTS[name], f"{name} renders different bytes"
