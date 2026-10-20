"""Every door that adds a layer answers which rows it reads and which x/y it takes the same way.

A layer attached to a chart (``Chart.layer(fm.Layer(...))``, ``chart +
fm.SecondaryY(...)``, a figure's ``layers=``) reads the chart's data: its rows
after the caller's own ``.transform(...)`` steps, never the statistics a
composite mark computes for its own layers. A ``fm.Layer(data=...)`` reads its
own rows and composes as a ``+`` operand does. A layer that binds no x/y takes
the binding the caller wrote, latest per channel.

Each rendering case is compared with the ``+`` or ``fm.LayerChart`` spelling
that already reads the intended rows, so a door reading the wrong batch fails
on mark counts or positions, not on "did not raise". Each test names the
alternative it rejects:

- *tail reading*: an attached layer reads the pipeline tail, so a histogram's
  bins or a density grid;
- *source-only check*: ``SecondaryY`` validates against the source columns and
  reads the tail;
- *resolved binding*: an attached layer inherits the x/y a composite rewrote
  (``bin_start``) instead of the written one;
- *host-chain merge*: ``Layer(data=)`` rows are concatenated into the host and
  run through the host's chain;
- *composed tail*: ``SecondaryY`` on a composition reads the combined rows of
  both operands;
- *snapshot x*: ``SecondaryY`` copies the base's x when it is added, so a later
  ``.encode(x=...)`` does not reach it.

The controls pin the SHA-256 of their SVG and of their normalized spec JSON:
charts no layer door changes render exactly these bytes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import warnings
from typing import Any, Callable

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import value_axis_ladder, value_axis_ladders
from tests._svg_rects import data_bar_rects, mark_groups, mark_layer, polylines, svg_elements
from tests._wire_steps import all_steps
from tests.test_plus_composes_transforms import DF, _N, H as H_VALUE, _box, _violin

SY = fm.SecondaryY

# `v` is the sample column, `y` and `r` two series, `g` a group.
df = pl.DataFrame(
    {
        "v": [1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 6.0, 7.0],
        "y": [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0],
        "r": [10.0, 20.0, 15.0, 30.0, 25.0, 35.0, 40.0, 45.0],
        "g": ["a", "b"] * 4,
    }
)
# A second frame sharing `v` and `y`: `datum.v > 3` keeps 5 of its 8 rows.
df2 = pl.DataFrame(
    {
        "v": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        "y": [2.0, 3.0, 4.0, 2.5, 3.5, 1.0, 5.0, 4.0],
        "w": [5.0, 6.0, 7.0, 8.0, 6.5, 5.5, 7.5, 6.0],
    }
)
_FILTER = "datum.v > 3"
_N_DF_FILTERED = df.filter(pl.col("v") > 3).height
_N_DF2_FILTERED = df2.filter(pl.col("v") > 3).height


def H() -> fm.Chart:
    return fm.Chart(df).mark_histogram().encode(x="v")


def P(data: pl.DataFrame = df) -> fm.Chart:
    return fm.Chart(data).mark_point().encode(x="v", y="y")


def P2(data: pl.DataFrame = df) -> fm.Chart:
    return fm.Chart(data).mark_line().encode(x="v", y="y")


def filtered(chart: fm.Chart) -> fm.Chart:
    return chart.transform(fm.transform_filter(_FILTER))


def fP(data: pl.DataFrame = df) -> fm.Chart:
    return fm.Chart(data).transform(fm.transform_filter(_FILTER)).mark_point().encode(x="v", y="y")


def tick(**encoding: Any) -> fm.Layer:
    return fm.Layer(mark="tick", encoding=encoding)


def _mean_by_v(*ops: dict) -> fm.Chart:
    return fm.Chart(df).transform(fm.transform_aggregate(*ops, groupby=["v"])).mark_line()


_MEAN_R = {"field": "r", "fn": "mean", "as": "mean_r"}
_MEAN_Y = {"field": "y", "fn": "mean", "as": "mean_y"}


def _census(marks: str) -> dict[str, int]:
    return {
        tag: len(re.findall(rf"<{tag} ", marks))
        for tag in ("rect", "path", "line", "circle", "polyline", "text")
    }


def _wire(chart: Any) -> dict:
    """The chart's spec JSON, with the identity-derived names of its routed steps made stable."""
    return json.loads(re.sub(r"_(auto|ident)_[0-9a-f]+", r"_\1_", chart.to_spec().to_json()))


def _secondary_series(svg: str) -> list[tuple[float, float]]:
    """The points of the last polyline the chart draws: the appended secondary series."""
    return polylines(mark_layer(svg))[-1][1]


def _right_axis_ticks(svg: str) -> dict[float, float]:
    ladders = value_axis_ladders(svg, axis="y")
    assert len(ladders) == 2, f"expected a left and a right y axis; got {len(ladders)}"
    return ladders[-1][1]


def _digest(svg: str) -> str:
    return hashlib.sha256(svg.encode()).hexdigest()


def _texts(svg: str) -> list[str]:
    return re.findall(r"<text[^>]*>([^<]*)</text>", svg)


# ---------------------------------------------------------------------------
# SecondaryY over a composite mark reads the chart's rows.
# ---------------------------------------------------------------------------


def test_a_secondary_bar_over_a_histogram_draws_the_written_x_from_its_own_zero():
    """Rejects tail reading: the secondary bars draw ``r`` over the written ``x="v"``."""
    svg = (H() + SY("r", mark="bar")).to_svg()
    oracle = fm.LayerChart(
        H(), fm.Chart(df).mark_bar().encode(x="v", y="r"), resolve={"y": "independent"}
    ).to_svg()
    rects = data_bar_rects(mark_layer(svg))
    secondary = rects[-df.height :]
    assert rects == data_bar_rects(mark_layer(oracle))
    zero = _right_axis_ticks(svg)[0.0]
    for rect in secondary:
        base = float(rect["y"]) + float(rect["height"])
        assert abs(base - zero) <= 0.05, f"bar {rect} starts at {base}, not on the 0 tick {zero}"


_STATISTIC_SY = (
    "SecondaryY: field 'count' is a statistic the base chart's mark_histogram computes for "
    "its own layers, not a column of the base chart's data (SecondaryY draws the base "
    "chart's rows after its .transform(...) steps). Compute it with .transform(...) "
    "(e.g. fm.Bin, fm.transform_aggregate), or draw it from its own chart with "
    "LayerChart(base, other, resolve={'y': 'independent'}). "
    "Available columns: ['v', 'y', 'r', 'g']"
)


def test_a_secondary_y_naming_a_histogram_statistic_is_refused_with_the_transform_remedy():
    """Rejects the source-only check's plain message for a statistic the mark computes."""
    with pytest.raises(ValueError) as exc:
        H() + SY("count")
    assert str(exc.value) == _STATISTIC_SY


def test_a_secondary_y_typo_on_a_histogram_keeps_the_plain_message():
    with pytest.raises(ValueError) as exc:
        H() + SY("nope")
    assert str(exc.value) == (
        "SecondaryY: field 'nope' is not a column of the base chart's data (SecondaryY "
        "draws from the same table as the base chart); available columns: "
        "['v', 'y', 'r', 'g']"
    )


def test_a_secondary_y_field_a_primitive_base_also_lacks_keeps_the_plain_message():
    """Only a composite mark computes statistics, so a primitive's missing field is a typo."""
    with pytest.raises(ValueError) as exc:
        fm.Chart(df).mark_point().encode(x="v", y="nope") + SY("nope")
    assert str(exc.value) == (
        "SecondaryY: field 'nope' is not a column of the base chart's data (SecondaryY "
        "draws from the same table as the base chart); available columns: "
        "['v', 'y', 'r', 'g']"
    )


def test_smooth_and_smooth_with_a_band_draw_the_same_secondary_series():
    """Rejects tail reading: an unnamed fit step must not hide ``r`` from the secondary layer."""
    plain = (fm.Chart(df).mark_smooth().encode(x="v", y="y") + SY("r")).to_svg()
    banded = (fm.Chart(df).mark_smooth(ci=0.95).encode(x="v", y="y") + SY("r")).to_svg()
    # The left axis's width differs with the fit's own y range, so the series
    # is compared on the right axis it is drawn against.
    assert len(_secondary_series(plain)) == df.height
    assert [y for _x, y in _secondary_series(plain)] == [y for _x, y in _secondary_series(banded)]
    assert _right_axis_ticks(plain) == _right_axis_ticks(banded)


def test_a_point_then_smooth_chart_takes_a_secondary_y():
    chart = fm.Chart(df).mark_point().encode(x="v", y="y").mark_smooth() + SY("r")
    assert len(_secondary_series(chart.to_svg())) == df.height


# ---------------------------------------------------------------------------
# Chart.layer over a composite mark reads the chart's rows.
# ---------------------------------------------------------------------------


def test_an_unbound_rug_on_a_histogram_takes_the_written_x():
    """Rejects resolved binding: the rug draws at ``v``, not at the bins' ``bin_start``."""
    rug = H().layer(fm.Layer(mark="tick"))
    oracle = H() + fm.Chart(df).mark_tick().encode(x="v")
    assert _census(mark_layer(rug.to_svg()))["line"] == df.height
    assert mark_layer(rug.to_svg()) == mark_layer(oracle.to_svg())


_STATISTIC_SHARED = (
    "layer 1: encoding '{channel}' references unknown column '{name}' (not present in that "
    "layer's data) — '{name}' is a statistic another layer's mark computes for itself; "
    "compute it in the chart's .transform(...) (e.g. fm.Bin, fm.transform_aggregate) so "
    "every layer reads it, or draw it from its own chart composed with `+`"
)
_STATISTIC_INDEPENDENT = (
    "layer 1: encoding '{channel}' references unknown column '{name}' (not present in that "
    "layer's data) — '{name}' is a statistic another layer's mark computes for itself; "
    "compute it in the chart's .transform(...) (e.g. fm.Bin, fm.transform_aggregate) so "
    "every layer reads it, or draw it from its own chart with "
    'fm.LayerChart(base, other, resolve={{"y": "independent"}})'
)


@pytest.mark.parametrize(
    "build, name",
    [
        pytest.param(
            lambda: H().layer(fm.Layer(mark="point", encoding={"y": "count"})),
            "count",
            id="histogram-count",
        ),
        pytest.param(
            lambda: (
                fm.Chart(df)
                .mark_density()
                .encode(x="v")
                .layer(fm.Layer(mark="line", encoding={"y": "density"}))
            ),
            "density",
            id="density-grid",
        ),
    ],
)
def test_an_attached_layer_naming_a_composite_statistic_is_refused_with_the_remedy(build, name):
    """Rejects tail reading: a composite's statistic is private to its own layers."""
    with pytest.raises(ValueError) as exc:
        build().to_svg()
    assert str(exc.value) == _STATISTIC_SHARED.format(channel="y", name=name)


@pytest.mark.parametrize(
    "build",
    [pytest.param(_box, id="boxplot"), pytest.param(_violin, id="violin")],
)
def test_a_tick_on_a_named_only_composite_reads_the_source_rows(build):
    chart = build().layer(tick(y="value"))
    assert (
        _census(mark_layer(chart.to_svg()))["line"] > _census(mark_layer(build().to_svg()))["line"]
    )


# ---------------------------------------------------------------------------
# The caller's own steps are the chart's chain.
# ---------------------------------------------------------------------------


def test_a_calculated_column_reaches_a_secondary_y():
    """Rejects the source-only check: ``z`` is in the chart's data, not its source."""
    calc = fm.transform_calculate("z", "datum.r * 2")
    chart = fm.Chart(df).transform(calc).mark_point().encode(x="v", y="y") + SY("z")
    oracle = fm.LayerChart(
        fm.Chart(df).transform(calc).mark_point().encode(x="v", y="y"),
        fm.Chart(df).transform(calc).mark_line().encode(x="v", y="z"),
        resolve={"y": "independent"},
    )
    assert _secondary_series(chart.to_svg()) == _secondary_series(oracle.to_svg())


def test_an_aggregated_column_reaches_a_secondary_y():
    """Rejects the source-only check: ``mean_r`` is in the chart's data, not its source."""
    chart = _mean_by_v(_MEAN_R, _MEAN_Y).encode(x="v", y="mean_y") + SY("mean_r")
    oracle = fm.LayerChart(
        _mean_by_v(_MEAN_R, _MEAN_Y).encode(x="v", y="mean_y"),
        _mean_by_v(_MEAN_R, _MEAN_Y).encode(x="v", y="mean_r"),
        resolve={"y": "independent"},
    )
    assert _secondary_series(chart.to_svg()) == _secondary_series(oracle.to_svg())


_SOURCE_INDEPENDENT = (
    "layer 1: encoding 'y' references unknown column 'r' (not present in that layer's "
    "data) — this layer reads the chart's transformed rows, not its source data; draw 'r' "
    'from the source with fm.LayerChart(base, other, resolve={"y": "independent"}) instead'
)
_SOURCE_SHARED = (
    "layer 1: encoding 'y' references unknown column 'r' (not present in that layer's "
    "data) — this layer reads the chart's transformed rows, not its source data; draw 'r' "
    "from the source by composing a separate chart with `+` instead"
)


def test_a_secondary_y_naming_a_column_the_callers_aggregate_drops_names_the_layer_chart_remedy():
    """Rejects a door-blind remedy: an independent-y layer's remedy is ``fm.LayerChart``."""
    with pytest.raises(ValueError) as exc:
        (_mean_by_v(_MEAN_Y).encode(x="v", y="mean_y") + SY("r")).to_svg()
    assert str(exc.value) == _SOURCE_INDEPENDENT


def test_a_layer_naming_a_column_the_callers_aggregate_drops_names_the_plus_remedy():
    with pytest.raises(ValueError) as exc:
        _mean_by_v(_MEAN_Y).encode(x="v", y="mean_y").layer(tick(y="r")).to_svg()
    assert str(exc.value) == _SOURCE_SHARED


def _plain(channel: str, name: str, layer: int = 1) -> str:
    return (
        f"layer {layer}: encoding '{channel}' references unknown column '{name}' "
        "(not present in that layer's data)"
    )


@pytest.mark.parametrize(
    "build, message",
    [
        pytest.param(lambda: H().layer(tick(x="nope")), _plain("x", "nope"), id="layer-histogram"),
        pytest.param(lambda: fP().layer(tick(x="nope")), _plain("x", "nope"), id="layer-filtered"),
        pytest.param(lambda: fP() + SY("nope"), _plain("y", "nope"), id="secondary-filtered"),
        pytest.param(
            lambda: fP() + filtered(fm.Chart(df2).mark_point().encode(x="v", y="missing")),
            _plain("y", "missing"),
            id="plus-operand-branch-reader",
        ),
    ],
)
def test_a_typo_on_every_door_gets_the_plain_message(build, message):
    """Rejects classifying every reader: a typo, or a ``+`` operand's own column, has no remedy."""
    with pytest.raises(ValueError) as exc:
        build().to_svg()
    assert str(exc.value) == message


# ---------------------------------------------------------------------------
# A composition's data is its left operand's data.
# ---------------------------------------------------------------------------


def test_a_secondary_y_on_a_composition_reads_the_left_operands_filtered_rows():
    """Rejects composed tail: the secondary series draws the left's 5 filtered rows."""
    chart = (fP() + P(df2)) + SY("r")
    oracle = fm.LayerChart(
        fP(),
        filtered(fm.Chart(df).mark_line().encode(x="v", y="r")),
        resolve={"y": "independent"},
    )
    assert len(_secondary_series(chart.to_svg())) == _N_DF_FILTERED
    assert len(_secondary_series(oracle.to_svg())) == _N_DF_FILTERED


def test_a_secondary_y_used_as_a_left_operand_reads_its_own_rows():
    """Rejects tail reading: the secondary series draws ``df``'s ``r``, not the bins."""
    chart = (H() + SY("r")) + P(df2)
    assert len(_secondary_series(chart.to_svg())) == df.height
    assert _right_axis_ticks(chart.to_svg()) == _right_axis_ticks((H() + SY("r")).to_svg())


def test_a_rug_on_a_right_operand_reads_that_operands_rows():
    """Rejects resolved binding: the tick draws ``df``'s ``v`` beside ``df2``'s points.

    The tick writes no positional channel, so it takes its host's written
    ``x="v"`` and layer 0's ``y="y"``, both drawn by a tick, read in ``df``.
    """
    chart = P(df2) + H().layer(fm.Layer(mark="tick"))
    oracle = P(df2) + (H() + fm.Chart(df).mark_tick().encode(x="v", y="y"))
    assert _census(mark_layer(chart.to_svg()))["line"] == df.height
    assert mark_layer(chart.to_svg()) == mark_layer(oracle.to_svg())


def test_a_layer_on_a_different_frame_composition_reads_the_left_operands_filtered_rows():
    """Rejects composed tail: the tick draws the left's filtered rows only."""
    chart = (fP() + P(df2)).layer(tick(x="v"))
    assert _census(mark_layer(chart.to_svg()))["line"] == _N_DF_FILTERED


# ---------------------------------------------------------------------------
# A Layer with its own data composes as a `+` operand does.
# ---------------------------------------------------------------------------


def test_a_data_layer_reads_all_of_its_own_rows():
    """Rejects host-chain merge: the host's filter does not reach the layer's rows."""
    layer = fm.Layer(data=df2, mark="line", encoding={"x": "v", "y": "w"})
    chart = fP().layer(layer)
    oracle = fP() + fm.Chart(df2).mark_line().encode(x="v", y="w")
    assert len(_secondary_series(chart.to_svg())) == df2.height
    assert mark_layer(chart.to_svg()) == mark_layer(oracle.to_svg())


def test_a_data_layer_over_the_hosts_own_frame_follows_the_same_frame_rule():
    """Rejects host-chain merge: the frame is not concatenated with itself and filtered."""
    layer = fm.Layer(data=df, mark="line", encoding={"x": "v", "y": "r"})
    chart = fP().layer(layer)
    oracle = fP() + fm.Chart(df).mark_line().encode(x="v", y="r")
    assert len(_secondary_series(chart.to_svg())) == df.height
    assert mark_layer(chart.to_svg()) == mark_layer(oracle.to_svg())


def test_a_data_layer_keeps_its_name_color_and_conditional_as_plus_does():
    """V1: the layer's own color stays on its layer, as on the ``+`` spelling."""
    sel = fm.selection_point(fields=["w"], name="pick")
    cond = sel.when(fm.Color("w")).otherwise(fm.value("#cccccc"))
    encoding = {"x": "v", "y": "w", "color": fm.Color("w", condition=cond)}
    chart = (
        P()
        .add_selection(sel)
        .layer(fm.Layer(data=df2, mark="point", name="other", encoding=encoding))
    )
    oracle = P().add_selection(sel) + fm.Chart(df2).mark_point().encode(**encoding)
    wire, oracle_wire = _wire(chart), _wire(oracle)
    assert wire["layers"][1]["name"] == "other"
    assert "color" not in wire["encoding"] and "color" not in oracle_wire["encoding"]
    assert wire["layers"][1]["encoding"]["color"] == oracle_wire["layers"][1]["encoding"]["color"]
    assert wire["layers"][1]["encoding"]["color"]["field"] == "w"
    assert "color" not in wire["layers"][0]["encoding"]
    assert wire["conditionals"] == oracle_wire["conditionals"]
    assert mark_layer(chart.to_svg()) == mark_layer(oracle.to_svg())


def test_layers_attached_in_one_call_keep_their_order():
    chart = P().layer(
        tick(x="v"),
        fm.Layer(data=df2, mark="line", encoding={"x": "v", "y": "w"}),
        fm.Layer(mark="rule", encoding={"y": "y"}),
    )
    assert [layer["mark"] for layer in _wire(chart)["layers"]] == ["point", "tick", "line", "rule"]


# ---------------------------------------------------------------------------
# `.transform()` and `.encode()` after attaching.
# ---------------------------------------------------------------------------


def test_a_transform_after_attaching_runs_as_it_would_before():
    """Rejects tail placement: the filter runs before the histogram's bins, either way round."""
    late = H().layer(fm.Layer(mark="tick")).transform(fm.transform_filter(_FILTER))
    early = H().transform(fm.transform_filter(_FILTER)).layer(fm.Layer(mark="tick"))
    assert late.to_svg() == early.to_svg()
    assert _census(mark_layer(late.to_svg()))["line"] == _N_DF_FILTERED


def test_a_transform_after_attaching_to_a_named_only_composite_runs_as_it_would_before():
    late = _box().layer(tick(y="value")).transform(fm.transform_filter("datum.value > 0"))
    early = _box().transform(fm.transform_filter("datum.value > 0")).layer(tick(y="value"))
    assert late.to_svg() == early.to_svg()


def test_an_encode_after_attaching_a_rug_moves_it():
    """Rejects resolved binding: the rug takes the later written ``x``."""
    chart = H().layer(fm.Layer(mark="tick")).encode(x="y")
    oracle = H().layer(tick(x="y"))
    assert mark_layer(chart.to_svg()) == mark_layer(oracle.to_svg())


def _secondary_wire(chart: Any) -> dict:
    return _wire(chart)["layers"][-1]


# Each case pairs `.encode(x="y")` after SecondaryY with x="y" written before it.
_REENCODE_CASES = {
    "histogram": (lambda: (H() + SY("r")).encode(x="y"), lambda: H().encode(x="y") + SY("r")),
    "point": (lambda: (P() + SY("r")).encode(x="y"), lambda: P().encode(x="y") + SY("r")),
    "composition": (
        lambda: ((P() + P(df2)) + SY("r")).encode(x="y"),
        lambda: (P().encode(x="y") + P(df2)) + SY("r"),
    ),
    "filtered-composition": (
        lambda: ((fP() + P(df2)) + SY("r")).encode(x="y"),
        lambda: (fP().encode(x="y") + P(df2)) + SY("r"),
    ),
    "nested-left": (
        lambda: ((H() + SY("r")) + P(df2)).encode(x="y"),
        lambda: (H().encode(x="y") + SY("r")) + P(df2),
    ),
}


def test_a_right_operands_secondary_y_keeps_its_hosts_written_x():
    """Spec §4.1 parent rule, cell X1: a secondary layer on a right operand takes its host's written ``x``.

    The histogram host wrote ``x="v"``, so a later ``.encode(x="y")`` on the
    composition, the host's own parent, does not reach it. Rejects filling
    the right operand's attached layer from the composition's x.
    """
    late = (P(df2) + (H() + SY("r"))).encode(x="y")
    early = P(df2).encode(x="y") + (H() + SY("r"))
    alone = H() + SY("r")
    assert _secondary_of(late)["encoding"]["x"]["field"] == "v"
    assert _secondary_of(late)["encoding"]["x"] == _secondary_of(early)["encoding"]["x"]
    assert _secondary_of(late)["encoding"]["x"] == _secondary_of(alone)["encoding"]["x"]
    assert "x" in _secondary_of(late)["inherited_positional"]


def _secondary_of(chart: Any) -> dict:
    return next(layer for layer in _wire(chart)["layers"] if layer.get("independent_y"))


@pytest.mark.parametrize("case", list(_REENCODE_CASES))
def test_an_encode_after_secondary_y_reaches_the_secondary_layer(case):
    """Rejects snapshot x: the secondary series draws at the latest written ``x``."""
    late, early = (build() for build in _REENCODE_CASES[case])
    assert _secondary_of(late)["encoding"]["x"]["field"] == "y"
    assert _secondary_of(late)["encoding"]["x"] == _secondary_of(early)["encoding"]["x"]
    assert "x" in _secondary_of(late)["inherited_positional"]
    late_svg, early_svg = late.to_svg(), early.to_svg()
    assert len(_secondary_series(late_svg)) == len(_secondary_series(early_svg))
    assert _right_axis_ticks(late_svg) == _right_axis_ticks(early_svg)


def test_a_retitle_after_secondary_y_does_not_rebind_its_x():
    chart = (H() + SY("r")).labs(x="T")
    assert _secondary_of(chart)["encoding"]["x"]["field"] == "v"
    assert "T" in _texts(chart.to_svg())


# ---------------------------------------------------------------------------
# LayerChart, facets, interactive output, figures.
# ---------------------------------------------------------------------------


def _rug_on_value_histogram() -> fm.Chart:
    return H_VALUE().layer(tick(x="value"))


def _value_points() -> fm.Chart:
    return fm.Chart(DF).mark_point().encode(x="value", y="y")


@pytest.mark.parametrize(
    "render",
    [
        pytest.param(
            lambda: fm.LayerChart(_rug_on_value_histogram(), _value_points()), id="static"
        ),
        pytest.param(
            lambda: fm.LayerChart(_rug_on_value_histogram(), _value_points())._build_merged(),
            id="merged",
        ),
        pytest.param(
            lambda: fm.LayerChart(
                _rug_on_value_histogram(), _value_points(), resolve={"y": "independent"}
            ),
            id="independent-y",
        ),
    ],
)
def test_a_layer_chart_member_draws_its_rug(render):
    """Rejects tail reading: the member's rug draws the sample values on every route."""
    assert _census(mark_layer(render().to_svg()))["line"] == _N


def _groups() -> list[str]:
    return sorted(df["g"].unique().to_list())


def _per_group(build: Callable[[pl.DataFrame], fm.Chart]) -> list[dict[str, int]]:
    return [_census(mark_layer(build(df.filter(pl.col("g") == g)).to_svg())) for g in _groups()]


def test_a_faceted_secondary_y_over_a_histogram_draws_each_groups_rows():
    def build(data: pl.DataFrame) -> fm.Chart:
        return fm.Chart(data).mark_histogram().encode(x="v") + SY("r")

    panels = mark_groups((H() + SY("r")).facet("g").to_svg())
    assert [_census(p)["polyline"] for p in panels] == [c["polyline"] for c in _per_group(build)]
    assert [len(points) for _stroke, points in polylines("".join(panels))] == [
        df.filter(pl.col("g") == g).height for g in _groups()
    ]


def test_a_faceted_rug_on_a_histogram_draws_each_groups_rows():
    panels = mark_groups(H_VALUE().layer(tick(x="value")).facet("g").to_svg())
    expected = [DF.filter(pl.col("g") == g).height for g in DF["g"].unique(maintain_order=True)]
    assert [_census(p)["line"] for p in panels] == expected


def test_a_figures_layers_door_attaches_as_chart_layer_does():
    """Rejects resolved binding: ``displot``'s rug draws at ``v`` as ``.layer`` draws it."""
    through_door = fm.displot(df, x="v", layers=[fm.Layer(mark="tick")])
    through_layer = fm.displot(df, x="v").layer(fm.Layer(mark="tick"))
    assert _census(mark_layer(through_door.to_svg()))["line"] == df.height
    assert mark_layer(through_door.to_svg()) == mark_layer(through_layer.to_svg())


#: Rows whose (g, h) and (g, f) groups have distinct counts.
_COUNTED_G = pl.DataFrame(
    {
        "g": ["a", "b", "a", "a", "b", "a", "a", "b", "c", "c"],
        "h": ["u", "v", "u", "v", "v", "u", "u", "u", "v", "u"],
        "f": ["p", "q"] * 5,
    }
)
_COUNT_LABELS: dict[str, Any] = {
    "shorthand": {"text": "count()"},
    "channel": {"text": fm.Text(aggregate="count")},
}
#: Each count plot door, and the columns each panel counts by.
_COUNT_DOORS: dict[str, tuple[dict, list[str], str | None]] = {
    "plain": ({}, ["g"], None),
    "hue": ({"hue": "h"}, ["g", "h"], None),
    "hue-dodged": ({"hue": "h", "dodge": True}, ["g", "h"], None),
    "faceted": ({"col": "f"}, ["g"], "f"),
}


def _panel_counts(chart: Any) -> list[tuple[list[float], list[float]]]:
    """Per interactive panel: the bars' heights and the labels' values, each sorted."""
    out = []
    for panel in json.loads(chart.interactive().scene_json)["panels"]:
        bars: list[float] = []
        labels: list[float] = []
        for batch in panel["marks"]:
            for tooltip in batch.get("tooltips") or []:
                fields = {f["name"]: f["value"] for f in tooltip["fields"]}
                if batch["kind"] == "bar":
                    bars.append(float(fields["n"]))
                elif batch["kind"] == "text":
                    labels.append(float(fields["count_all"]))
        out.append((sorted(bars), sorted(labels)))
    return out


@pytest.mark.parametrize("label", sorted(_COUNT_LABELS))
@pytest.mark.parametrize("door", sorted(_COUNT_DOORS))
def test_a_count_plot_label_draws_each_bars_count(door, label):
    """``catplot(kind="count", layers=[text=count()])``: each panel's labels are its bars' heights.

    The oracle is the row count per drawn group, from the frame. Rejects a
    label counting at a coarser grain than the bars (one label per category
    beside per-hue bars) or reading the bars' output column.
    """
    kwargs, by, facet = _COUNT_DOORS[door]
    chart = fm.catplot(
        _COUNTED_G,
        x="g",
        kind="count",
        layers=[fm.Layer(mark="text", encoding=_COUNT_LABELS[label])],
        **kwargs,
    )
    panels = (
        [_COUNTED_G]
        if facet is None
        else [_COUNTED_G.filter(pl.col(facet) == value) for value in ("p", "q")]
    )
    expected = [sorted(float(n) for n in rows.group_by(by).len()["len"]) for rows in panels]
    assert _panel_counts(chart) == [(counts, counts) for counts in expected]


def test_a_count_plot_layer_naming_n_is_refused_with_the_count_label_remedy():
    """``text="n"`` names no column: the count is ``y=count()``, so the remedy is ``text="count()"``."""
    with pytest.raises(ValueError) as exc:
        fm.catplot(
            _COUNTED_G, x="g", kind="count", layers=[fm.Layer(mark="text", encoding={"text": "n"})]
        )
    assert str(exc.value) == (
        "catplot(kind='count'): layers= binds text='n', but the count plot draws each bar's "
        "count as y=count(), so there is no column 'n'; label each bar with text='count()' "
        "(or fm.Text(aggregate='count')), which counts the rows the layer reads as each bar does"
    )


#: Other rows holding their own ``n``, one row per category.
_OTHER_WITH_N = pl.DataFrame({"g": ["a", "b", "c"], "n": [7, 8, 9]})


def test_a_count_plot_data_layer_whose_rows_hold_n_draws_them():
    """Control: a ``Layer(data=)`` whose own rows hold ``n`` draws those rows as written."""
    chart = fm.catplot(
        _COUNTED_G,
        x="g",
        kind="count",
        layers=[
            fm.Layer(data=_OTHER_WITH_N, mark="text", encoding={"x": "g", "y": "n", "text": "n"})
        ],
    )
    # The labels draw above the plot clip, after every axis and legend text.
    labels = re.findall(r"<text[^>]*>([^<]*)</text>", chart.to_svg())[-3:]
    assert sorted(labels) == ["7", "8", "9"]


def test_a_count_plot_data_layer_lacking_n_is_refused_though_the_figures_rows_hold_it():
    """The rows the layer reads are its own frame's: the figure's ``n`` is not theirs."""
    with pytest.raises(ValueError) as exc:
        fm.catplot(
            _COUNTED_G.with_columns(pl.lit(1).alias("n")),
            x="g",
            kind="count",
            layers=[
                fm.Layer(
                    data=_OTHER_WITH_N.drop("n"), mark="text", encoding={"x": "g", "text": "n"}
                )
            ],
        )
    assert str(exc.value).startswith("catplot(kind='count'): layers= binds text='n'")


@pytest.mark.parametrize(("channel", "mark"), [("detail", "line"), ("shape", "point")])
def test_a_count_plot_layer_binding_n_on_a_channel_taking_no_aggregate_gets_the_plain_refusal(
    channel, mark
):
    """No ``aggregate=`` remedy where the channel refuses ``aggregate=``: the render's plain message."""
    chart = fm.catplot(
        _COUNTED_G, x="g", kind="count", layers=[fm.Layer(mark=mark, encoding={channel: "n"})]
    )
    with pytest.raises(ValueError) as exc:
        chart.to_svg()
    assert str(exc.value) == (
        f"catplot: encoding '{channel}' references unknown column 'n' "
        "(not present in that layer's data)"
    )


def test_a_count_plot_internal_layer_binding_n_is_refused_like_any_attached_layer():
    """An internal ``_Layer`` passed to ``layers=`` is checked as a public ``Layer`` is."""
    from ferrum._layer import _Layer

    with pytest.raises(ValueError) as exc:
        fm.catplot(
            _COUNTED_G, x="g", kind="count", layers=[_Layer(mark="text", encoding={"text": "n"})]
        )
    assert str(exc.value).startswith("catplot(kind='count'): layers= binds text='n'")


def test_a_count_plot_layer_naming_a_column_n_the_data_holds_draws_it():
    """Control: a frame's own ``n`` column is the caller's, drawn as written."""
    data = _COUNTED_G.with_columns(pl.lit(7.0).alias("n"))
    chart = fm.catplot(
        data, x="g", kind="count", layers=[fm.Layer(mark="text", encoding={"text": "n"})]
    )
    assert set(re.findall(r"<text[^>]*>([^<]*)</text>", chart.to_svg())) >= {"7"}


_NO_MUTATION_CALLS: dict[str, Callable[[fm.Chart], Any]] = {
    "layer": lambda c: c.layer(fm.Layer(mark="tick")),
    "data-layer": lambda c: c.layer(fm.Layer(data=df2, mark="line", encoding={"x": "v", "y": "w"})),
    "secondary-y": lambda c: c + SY("r"),
    "plus": lambda c: c + P(df2),
    "transform": lambda c: c.transform(fm.transform_filter(_FILTER)),
    "encode": lambda c: c.encode(x="y"),
}


@pytest.mark.parametrize("call", list(_NO_MUTATION_CALLS))
@pytest.mark.parametrize(
    "base",
    [
        pytest.param(H, id="histogram"),
        pytest.param(lambda: H() + SY("r"), id="secondary"),
        pytest.param(lambda: fP() + P(df2), id="composition"),
    ],
)
def test_no_door_mutates_the_callers_chart(base, call):
    chart = base()
    before = (
        _wire(chart),
        copy.copy(chart._transforms),
        chart._operands,
        chart._frames,
        chart._layers,
    )
    _NO_MUTATION_CALLS[call](chart).to_svg()
    assert (
        _wire(chart),
        chart._transforms,
        chart._operands,
        chart._frames,
        chart._layers,
    ) == before


# ---------------------------------------------------------------------------
# Byte-identical controls: the SVG and spec JSON the commit before the layer
# data model rendered for the same charts.
# ---------------------------------------------------------------------------


def _each_with_own_conditional() -> fm.Chart:
    """Two ``+`` operands, each with its own conditional, which stays on its own layer."""
    own = fm.selection_point(fields=["y"], name="own")
    left = P().encode(color=_PICK.when(fm.Color("g")).otherwise(fm.value("#cccccc")))
    right = P2().encode(size=own.when(fm.Size("y")).otherwise(fm.value(10)))
    return left.add_selection(_PICK) + right.add_selection(own)


def _calculated_points() -> fm.Chart:
    return (
        fm.Chart(df)
        .transform(fm.transform_calculate("z", "datum.r * 2"))
        .mark_point()
        .encode(x="v", y="y")
    )


_CONTROLS: dict[str, Callable[[], Any]] = {
    "point-layer-line": lambda: P().layer(fm.Layer(mark="line")),
    "filtered-point-layer-line": lambda: fP().layer(fm.Layer(mark="line")),
    "filtered-point+secondary": lambda: fP() + SY("r"),
    "line+secondary-bar": lambda: P2() + SY("r", mark="bar"),
    "boxplot-layer-tick": lambda: _box().layer(tick(y="value")),
    "violin-layer-tick": lambda: _violin().layer(tick(y="value")),
    "smooth-band+secondary": lambda: (
        fm.Chart(df).mark_smooth(ci=0.95).encode(x="v", y="y") + SY("r")
    ),
    "swarm-layer-text": lambda: (
        fm.Chart(df)
        .transform(fm.Swarm(category="g", value="y"))
        .mark_point(size=200.0)
        .encode(x="g:N", y="y")
        .layer(fm.Layer(mark="text", encoding={"text": "y"}))
    ),
    "same-frame-composition+secondary": lambda: (P() + P2()) + SY("r"),
    "same-frame-composition-layer-line": lambda: (P() + P2()).layer(
        fm.Layer(mark="line", encoding={"y": "r"})
    ),
    "faceted-point+secondary": lambda: (P() + SY("r")).facet("g"),
    "layer-chart-independent-bar": lambda: fm.LayerChart(
        P(), fm.Chart(df).mark_bar().encode(x="v", y="r"), resolve={"y": "independent"}
    ),
    "composition-transform": lambda: (fP() + P(df2)).transform(fm.transform_filter("datum.y > 1")),
    "same-frame-composition-transform": lambda: (P() + P2()).transform(
        fm.transform_filter("datum.y > 1")
    ),
    "flat-mean-bar": lambda: _aggregated_bar("mean", None),
    "mean-bar+mean-point": lambda: (
        _aggregated_bar("mean", None) + fm.Chart(_BARS).mark_point().encode(x="g", y="mean(y)")
    ),
    "plus-each-own-conditional": lambda: _each_with_own_conditional(),
    "ungrouped-binned-count": lambda: _binned_count(),
    # Renders since encoding statistics became per layer (the commit before
    # the layer data model refused it), so its pins are this cycle's own.
    "ungrouped-binned-count-layer-rule": lambda: _binned_count().layer(
        fm.Layer(mark="rule", encoding={"y": fm.Y("v", aggregate="mean")})
    ),
    "no-data-host-data-layer": lambda: (
        fm.Chart(None).mark_point().encode(x="v", y="y").layer(fm.Layer(data=df, mark="line"))
    ),
}

# The SHA-256 of each control's SVG.
_SVG_DIGESTS = {
    "point-layer-line": "62dd601259a1fcb244f892679826d98b92b5871135c3963423854aafe4628a3e",
    "filtered-point-layer-line": "b9812d35fd16f18696cb34788be6f840decfbce7cdd04c80bce0785456255ee0",
    "filtered-point+secondary": "5330d3d1cb9a4953bf107963c9bb6f2db6356b357b7e1dc3ebe030b84d4f2660",
    "line+secondary-bar": "3009fe88ea10f950ae9177f10fef14b6bdbc54c77ab7af4915c5e11253f865a8",
    "boxplot-layer-tick": "6feaa81d1bf58b524d20707a83ba1ac52369cfe831000530a6d2976562cd6ae0",
    "violin-layer-tick": "6792083281b6cac82ecae8e7f6b92d6bf0efca6ee8663b7e177c7ac3edb95ff5",
    "smooth-band+secondary": "fa3455c4455ae9130b17c6645851c1b9c70b7a36bd75d1c7b7d3351866b4d003",
    "swarm-layer-text": "a75aec0e2f713fefcf632e17af067f2bf0ff39e5115c86269011a450e1395ab9",
    "same-frame-composition+secondary": "3262c72b3e7f9287cf22d5cc1dca8f1be56c38a873f2217bd796a17e7589c1b9",
    "same-frame-composition-layer-line": "42d6493e6a5c992ca98792618386cd7548d5ffd8777217ea211d3c0315210923",
    "faceted-point+secondary": "95cd5af2f5808ea74de483e73bf798f70d90f9dd98b622707413643093dd8b45",
    "layer-chart-independent-bar": "40324f95225eaf94a82eff4ea3d2666101a0facfbacde35b3d9b1a1863dd3ecc",
    "composition-transform": "fc269064f36ff115068e31ee8d1af2bcce8eab06f60c51d43c033633b14b033a",
    "same-frame-composition-transform": "d3465223e51ce48ef3f537dd2627a6575cb85fe4e56ab8047b65dc9f21098a7e",
    "flat-mean-bar": "3ac447c87ee5187ba0fbfa0526da0285a69e73a1d1f3b87ff86b8676f3f216da",
    "mean-bar+mean-point": "68bffa6e84c1df86d6fd8462a5544f741ae56441dcf51a55d999593b8e629386",
    "plus-each-own-conditional": "dc55ef9989488878bf3165eadef306b8c26af3d04bdda764e889ba40a502a331",
    "ungrouped-binned-count": "f9b2041050d7eae26d615b4580ad7e8ee6590d15ff33c726c8244776a3838368",
    # The y-only rule draws no x, so it averages every row once: one rule, not one per bin.
    "ungrouped-binned-count-layer-rule": "a5b26fffb1ce82174edbde5ba08eb6208f834ae39c4ffd74e96c5f3c31c1b855",
    "no-data-host-data-layer": "62dd601259a1fcb244f892679826d98b92b5871135c3963423854aafe4628a3e",
}
_SECONDARY_CONTROLS = {
    "filtered-point+secondary",
    "line+secondary-bar",
    "smooth-band+secondary",
    "same-frame-composition+secondary",
    "faceted-point+secondary",
}


# The SHA-256 of each control's normalized spec JSON (see _normalized_json).
_JSON_DIGESTS = {
    "point-layer-line": "e9efaf8931203b579be8a46e221aa8fcbfcbf35c89319637170f3ae2dac2f232",
    "filtered-point-layer-line": "59f04ad9a65008584a7c10db1fb9bf1036a8a2498321c7971c12ce83229c8af6",
    "filtered-point+secondary": "6db2f7ac79e16b913d048e1e41c6cd2c20f0698d974dab068f63ff1ea628d908",
    "line+secondary-bar": "3120a59a390e3ded3ee669c3b491b521756aff6ef63bf29e8f9b90bec5e073d4",
    "boxplot-layer-tick": "837624cb5090256ff8d481b1b7ff259159f73192c1eb9083b282a3583a22c5af",
    "violin-layer-tick": "cbb20d5c76b7fb8f684d3d5b78c93d7e1517cb1ba1ebd42048f96ff4b6d56f47",
    "smooth-band+secondary": "c4cec8f19c26803966c4f21c39766bde63db237dee84d18362f58d13c43487a6",
    "swarm-layer-text": "018eaa033fcd4648f86e543d1f1a06f03ab9a09acdccb5438fe13ad5f2ecc443",
    "same-frame-composition+secondary": "df270ca1ec6dd0e0d247533786598d40290d58f52172be5dad45cada89c4fa71",
    "same-frame-composition-layer-line": "b3ff99269a0c13339e4ad6190900f526ad504bb9b6f80eef5ebce86d7a8fbe4f",
    "faceted-point+secondary": "aaf6cef5c01f57d17c064daedf33f5fa989f071b89ee5671b620efdf7b6e8853",
    "layer-chart-independent-bar": "9a23f515bc5a2c70e15b5cb918a532d5c3f8810aa1ffd91b88e8f47d874f92da",
    # Each operand runs as one branch over its own frame. A composition's
    # filter is a composed step copied into each operand chain it reaches,
    # the left's here and not the other frame's; its render is pinned
    # beside an oracle in
    # test_a_controls_composition_step_draws_what_its_parts_draw_filtered.
    "composition-transform": "1cbe1ae047b86c92c20d4999d0880fd955858c988ab5d49e7db0f8887cd1caca",
    "same-frame-composition-transform": "cab22b75964ad0cb84897be582b99ee5c950980505533b638273c8bef09a6fad",
    "flat-mean-bar": "2d162c1ff7e599abcb6e4880d6b6edd1c05829847402650c6a16da1317a6e0d4",
    "mean-bar+mean-point": "1aee8ae29a47aa292be1d571d327729e54d6037489efac8bde22dae25f8bc48e",
    # The spec carries no selection fields in its tooltip: the render's
    # tooltip reader adds them where the rows hold them.
    "plus-each-own-conditional": "1eac3e0ab559a6ced86598eca5dfdb6fb03c6153e2138e6215c168652dc288cb",
    "ungrouped-binned-count": "74ff073ed84e9a1416e76addcba55b61060cdbb75126878303a502f5f3d44233",
    # The y-only rule takes no binned x: an aggregate, not a second bin.
    "ungrouped-binned-count-layer-rule": "c5669a521646f12a78dd6fc3f799bcfedcfdacef9b4ad88c326c8ae4b98096e0",
    "no-data-host-data-layer": "e9efaf8931203b579be8a46e221aa8fcbfcbf35c89319637170f3ae2dac2f232",
}


def _normalized_json(case: str) -> str:
    """A control's spec JSON with its identity-derived step names made stable.

    A ``SecondaryY`` control's secondary layer lists its ``x`` in
    ``inherited_positional``, which these digests leave out: the layer takes
    its ``x`` from the written bindings when the spec is built, and draws it
    as the chart's own.
    """
    chart = _CONTROLS[case]()
    merged = chart._build_merged() if isinstance(chart, fm.LayerChart) else chart
    wire = _wire(merged)
    if case in _SECONDARY_CONTROLS:
        for layer in wire["layers"]:
            if layer.get("independent_y"):
                layer.pop("inherited_positional", None)
    return json.dumps(wire, sort_keys=True)


@pytest.mark.parametrize("case", list(_CONTROLS))
def test_a_control_renders_its_pinned_svg(case):
    assert _digest(_CONTROLS[case]().to_svg()) == _SVG_DIGESTS[case]


@pytest.mark.parametrize("case", list(_CONTROLS))
def test_a_control_serializes_its_pinned_spec(case):
    assert _digest(_normalized_json(case)) == _JSON_DIGESTS[case]


def _drawn_marks(chart: Any) -> tuple[int, list[int]]:
    """The points a chart draws, and each line's vertex count, sorted."""
    marks = mark_layer(_quiet_svg(chart))
    return marks.count("<circle"), sorted(len(points) for _, points in polylines(marks))


def _quiet_svg(chart: Any) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return chart.to_svg()


def _composition_filter() -> dict:
    return fm.transform_filter("datum.y > 1")


#: Each control that writes a step on a ``+`` composition, and its oracle:
#: the standalone parts it draws. The step reaches the left operand and every
#: operand over the composition's frame, never one over a different frame.
_COMPOSED_STEP_PARTS: dict[str, Callable[[], tuple]] = {
    "composition-transform": lambda: (fP().transform(_composition_filter()), P(df2)),
    "same-frame-composition-transform": lambda: (
        P().transform(_composition_filter()),
        P2().transform(_composition_filter()),
    ),
}


@pytest.mark.parametrize("case", sorted(_COMPOSED_STEP_PARTS))
def test_a_controls_composition_step_draws_what_its_parts_draw_filtered(case):
    """The pinned render is the oracle's: each reached part filtered, a different-frame part as written."""
    got = _drawn_marks(_CONTROLS[case]())
    parts = [_drawn_marks(part) for part in _COMPOSED_STEP_PARTS[case]()]
    assert got == (sum(c for c, _ in parts), sorted(n for _, lines in parts for n in lines))


#: The controls whose layer attached to a ``+`` composition reads the
#: composition's data: its left operand's rows, published inside that
#: operand's branch.
_COMPOSITION_ATTACHED = {"same-frame-composition+secondary", "same-frame-composition-layer-line"}


@pytest.mark.parametrize("case", sorted(set(_CONTROLS) - _COMPOSITION_ATTACHED))
def test_a_control_never_flags_a_chart_data_reader(case):
    chart = _CONTROLS[case]()
    merged = chart._build_merged() if isinstance(chart, fm.LayerChart) else chart
    assert "reads_chart_data" not in merged.to_spec().to_json()


@pytest.mark.parametrize("case", sorted(_COMPOSITION_ATTACHED))
def test_a_layer_attached_to_a_composition_reads_its_left_operands_rows(case):
    """The attached layer reads the rows the left operand's branch publishes, flagged as the chart's data."""
    layers = _wire(_CONTROLS[case]())["layers"]
    attached = layers[-1]
    assert attached["data_source"] == "_operand_0__rows"
    assert attached["reads_chart_data"] is True
    assert all(not layer.get("reads_chart_data") for layer in layers[:-1])


@pytest.mark.parametrize("case", sorted(_SECONDARY_CONTROLS))
def test_a_secondary_control_lists_its_x_as_inherited(case):
    secondary = _secondary_of(_CONTROLS[case]())
    assert secondary["inherited_positional"] == ["x"]
    if case in _COMPOSITION_ATTACHED:
        assert secondary["data_source"] == "_operand_0__rows"
    else:
        assert "data_source" not in secondary


# ---------------------------------------------------------------------------
# Encoding aggregates and bins are per layer.
# ---------------------------------------------------------------------------

# Two rows per `g`, one per (`g`, `c`): every aggregate below is a round number.
_BARS = pl.DataFrame(
    {
        "g": ["a", "a", "b", "b", "c", "c"],
        "c": ["x", "y", "x", "y", "x", "y"],
        "y": [1.0, 3.0, 2.0, 6.0, 4.0, 4.0],
    }
)
# Each aggregate: the bar's shorthand, the label's (field, op) and the oracle.
_AGGREGATES = {
    "mean": ("mean(y)", ("y", "mean"), pl.col("y").mean()),
    "sum": ("sum(y)", ("y", "sum"), pl.col("y").sum()),
    "median": ("median(y)", ("y", "median"), pl.col("y").median()),
    "count": ("count()", ("", "count"), pl.len()),
}


def _aggregated_bar(agg: str, color: str | None) -> fm.Chart:
    shorthand = _AGGREGATES[agg][0]
    encoding = {"x": "g", "y": shorthand}
    if color is not None:
        encoding["color"] = color
    return fm.Chart(_BARS).mark_bar().encode(**encoding)


def _label(agg: str) -> fm.Layer:
    field, op = _AGGREGATES[agg][1]
    return fm.Layer(mark="text", encoding={"text": fm.Text(field, aggregate=op)})


def _labels(chart: Any, bare: Any) -> list[str]:
    """The text a chart draws beyond what *bare* (the same chart without its labels) draws."""
    remaining = list(_texts(bare.to_svg()))
    labels = []
    for text in _texts(chart.to_svg()):
        if text in remaining:
            remaining.remove(text)
        else:
            labels.append(text)
    return sorted(labels)


def _expected(agg: str, by: list[str]) -> list[str]:
    values = _BARS.group_by(by).agg(_AGGREGATES[agg][2].alias("value"))["value"].to_list()
    return sorted(f"{float(v):g}" for v in values)


def _steps(chart: Any, kind: str) -> list[dict]:
    """The chart's pipeline steps of type *kind*, at any depth, in the order they are written."""
    merged = chart._build_merged() if isinstance(chart, fm.LayerChart) else chart
    return [step for step in all_steps(_wire(merged)) if step["type"] == kind]


@pytest.mark.parametrize("color", [None, "c"], ids=["by-g", "by-g-and-c"])
@pytest.mark.parametrize("agg", list(_AGGREGATES))
def test_a_label_on_an_aggregated_bar_computes_the_value_the_bar_draws(agg, color):
    """Each layer computes its own aggregate over the chart's rows, grouped as the bars are."""
    bare = _aggregated_bar(agg, color)
    chart = bare.layer(_label(agg))
    assert _labels(chart, bare) == _expected(agg, ["g"] if color is None else ["g", "c"])
    steps = _steps(chart, "aggregate")
    assert len(steps) == 2 and all(step["name"].startswith("_layer_agg_") for step in steps)


# Each grouping channel and a mark that draws it. Color is a channel object,
# so it builds the color scale that tells the groups apart.
_GROUPINGS = {
    "color": ("point", {"color": fm.Color("c")}),
    "size": ("point", {"size": "c"}),
    "shape": ("point", {"shape": "c"}),
    "opacity": ("point", {"opacity": "c"}),
    "detail": ("line", {"detail": "c"}),
}

# size/opacity are value channels: `c` is a string column, so binding it
# refuses at render exactly as `encode()` does -- the exact strings the
# matrix below pins.
_REFUSED_VALUE_CHANNELS = {
    "size": "size: column 'c' has unsupported dtype: Utf8",
    "opacity": "opacity: column 'c' has unsupported dtype: Utf8",
}


def _y_values(svg: str, pixels: list[float]) -> list[float]:
    """Data values of y *pixels* on the chart's one y axis."""
    ladder = value_axis_ladders(svg, axis="y")[0][1]
    lo, hi = min(ladder), max(ladder)
    per_px = (hi - lo) / (ladder[hi] - ladder[lo])
    return sorted(round(lo + (px - ladder[lo]) * per_px, 3) for px in pixels)


def _point_glyph_ys(marks: str) -> list[float]:
    """y-pixel center of every point-mark glyph: a circle, or a square shape-scale glyph.

    A shape scale draws its second category as a square `<rect>`, not a
    `<circle>`, so a caller after every glyph a point layer drew (not only
    circles) reads both tags. The square glyph is `width == height`, which a
    bar's `<rect>` never is, so the filter does not pick up bars sharing the
    same clip group.
    """
    centers = [float(cy) for cy in re.findall(r'<circle [^>]*cy="([\d.]+)"', marks)]
    for attrs in svg_elements(marks, "rect", lambda a: a["width"] == a["height"]):
        centers.append(float(attrs["y"]) + float(attrs["height"]) / 2)
    return centers


@pytest.mark.parametrize("channel", list(_GROUPINGS))
def test_a_finer_layer_statistic_groups_by_its_own_channels(channel):
    """A layer's mean per (g, c) is computed over the raw rows, not over the bars' means.

    G6: the layer's own grouping channel draws, with a legend, exactly as the
    same point chart draws it alone -- `shape` draws a square glyph for its
    second category, not only circles, and its own legend. V12/X4: `size`
    and `opacity` bind the string column `c` exactly as `encode()` does, so
    they refuse at render with the same message a standalone chart raises;
    the statistic still plans its groupby over `c` before that refusal, so
    this row also pins that the grouping itself does not depend on whether
    the channel can be drawn.
    """
    mark, grouping = _GROUPINGS[channel]
    layer = fm.Layer(mark=mark, encoding={"y": fm.Y("y", aggregate="mean"), **grouping})
    chart = _aggregated_bar("mean", None).layer(layer)
    groupbys = [step["groupby"] for step in _steps(chart, "aggregate")]
    # V3: the layer's own channel never reaches the host, whose bars keep
    # their grain (rejects D1-B and promotion: the bars would split by c).
    assert groupbys == [["g"], ["g", "c"]]
    if channel in _REFUSED_VALUE_CHANNELS:
        with pytest.raises(ValueError) as exc:
            chart.to_svg()
        assert str(exc.value) == _REFUSED_VALUE_CHANNELS[channel]
        return
    svg = chart.to_svg()
    marks = mark_layer(svg)
    if mark == "line":
        lines = polylines(marks)
        assert [len(points) for _stroke, points in lines] == [3, 3]
        pixels = [y for _stroke, points in lines for _x, y in points]
    elif channel == "shape":
        pixels = _point_glyph_ys(marks)
    else:
        pixels = [float(cy) for cy in re.findall(r'<circle [^>]*cy="([\d.]+)"', marks)]
    # One row per (g, c), so each mean is that row's y.
    assert _y_values(svg, pixels) == sorted(_BARS["y"].to_list())
    if channel == "color":
        fills = re.findall(r'<circle [^>]*fill="([^"]+)"', marks)
        assert len(set(fills)) == _BARS["c"].n_unique()
    if channel == "shape":
        # One glyph kind per `c` category, three rows each (g in a, b, c).
        circles = len(re.findall(r'<circle [^>]*cy="[\d.]+"', marks))
        squares = len(svg_elements(marks, "rect", lambda a: a["width"] == a["height"]))
        assert (circles, squares) == (3, 3), (circles, squares)
        # A legend titled `c` with entries `x`, `y` -- the text no bare bar chart draws.
        bare = _aggregated_bar("mean", None)
        assert {"c", "x", "y"} <= set(_labels(chart, bare))


@pytest.mark.parametrize("channel", ["size", "opacity"])
def test_a_finer_layer_statistic_groups_by_a_numeric_value_channel(channel):
    """X2 (L, finer channel; spec §4.2 groupby): a layer's own size/opacity groups its statistic.

    Beside the string-column refusal above: a `size`/`opacity` binding on a
    numeric column groups the layer's statistic the same way, and draws it
    -- `size` with a legend; `opacity` draws no legend on any door, so this
    case checks the rendered value only.
    """
    bars = pl.DataFrame(
        {
            "g": ["a", "a", "b", "b", "c", "c"],
            "n": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "y": [1.0, 3.0, 2.0, 6.0, 4.0, 4.0],
        }
    )
    bare = fm.Chart(bars).mark_bar().encode(x="g", y="mean(y)")
    layer = fm.Layer(mark="point", encoding={"y": fm.Y("y", aggregate="mean"), channel: "n"})
    chart = bare.layer(layer)
    groupbys = [step["groupby"] for step in _steps(chart, "aggregate")]
    assert groupbys == [["g"], ["g", "n"]]
    svg = chart.to_svg()
    marks = mark_layer(svg)
    if channel == "size":
        radii = [float(r) for r in re.findall(r'<circle [^>]*r="([\d.]+)"', marks)]
        assert len(set(radii)) > 1, radii
        # A legend titled `n`, the text no bare bar chart draws.
        assert "n" in _labels(chart, bare)
    else:
        fills = re.findall(r'<circle [^>]*fill="([^"]+)"', marks)
        alphas = [
            1.0 if f.startswith("#") else float(f.rstrip(")").rsplit(",", 1)[1]) for f in fills
        ]
        assert len(set(alphas)) > 1, alphas


def test_a_faceted_layer_statistic_groups_by_the_facet_too():
    layer = fm.Layer(mark="point", encoding={"y": fm.Y("y", aggregate="mean")})
    chart = _aggregated_bar("mean", None).layer(layer).facet("c")
    assert [step["groupby"] for step in _steps(chart, "aggregate")] == [["g", "c"], ["g", "c"]]
    assert sum(_census(p)["circle"] for p in mark_groups(chart.to_svg())) == _BARS.height


def test_a_different_layer_statistic_reads_the_raw_rows():
    """A ``sum(y)`` beside a mean bar sums the rows, per ``g``."""
    layer = fm.Layer(mark="point", encoding={"y": fm.Y("y", aggregate="sum")})
    chart = _aggregated_bar("mean", None).layer(layer)
    steps = _steps(chart, "aggregate")
    assert [(step["ops"][0]["fn"], step["groupby"]) for step in steps] == [
        ("mean", ["g"]),
        ("sum", ["g"]),
    ]
    assert _census(mark_layer(chart.to_svg()))["circle"] == _BARS["g"].n_unique()


def test_a_right_operand_statistic_never_groups_by_the_left_operands_color():
    """V3/G4: the left operand's color is its own, so the right keeps its grain.

    Rejects D1-C (the left's encoding as the composition's): the right
    operand's sum would group by c too.
    """
    chart = _aggregated_bar("mean", "c") + fm.Chart(_BARS).mark_point().encode(x="g", y="sum(y)")
    assert [step["groupby"] for step in _steps(chart, "aggregate")] == [["g", "c"], ["g"]]


def _labelled_mean_bar() -> fm.Chart:
    return _aggregated_bar("mean", None).layer(_label("mean"))


def _bars_beside_points(labelled: bool) -> Callable[..., Any]:
    member = _labelled_mean_bar() if labelled else _aggregated_bar("mean", None)
    points = fm.Chart(_BARS).mark_point().encode(x="g", y="y")
    return lambda **kw: fm.LayerChart(member, points, **kw)


_LAYER_CHART_ROUTES = {
    "static": lambda make: make(),
    "merged": lambda make: make()._build_merged(),
    "independent-y": lambda make: make(resolve={"y": "independent"}),
}


@pytest.mark.parametrize("route", list(_LAYER_CHART_ROUTES))
def test_a_layer_chart_member_labels_its_aggregated_bars(route):
    build = _LAYER_CHART_ROUTES[route]
    chart = build(_bars_beside_points(labelled=True))
    bare = build(_bars_beside_points(labelled=False))
    assert _labels(chart, bare) == _expected("mean", ["g"])


def test_a_layer_chart_member_labels_render_interactive():
    html = _bars_beside_points(labelled=True)().interactive().to_html()
    assert "<html" in html.lower()


@pytest.mark.parametrize("data_first", [False, True], ids=["label-first", "data-first"])
def test_a_label_and_a_data_layer_attach_in_either_order(data_first):
    data_layer = fm.Layer(data=df2, mark="line", encoding={"x": "v", "y": "w"})
    layers = (data_layer, _label("mean")) if data_first else (_label("mean"), data_layer)
    chart = _aggregated_bar("mean", None).layer(*layers)
    bare = _aggregated_bar("mean", None).layer(data_layer)
    assert _labels(chart, bare) == _expected("mean", ["g"])
    assert len(polylines(mark_layer(chart.to_svg()))[-1][1]) == df2.height


def test_a_faceted_label_computes_each_panels_aggregate():
    bare = _aggregated_bar("mean", None).facet("c")
    chart = _labelled_mean_bar().facet("c")
    assert _labels(chart, bare) == _expected("mean", ["g", "c"])


def test_a_label_on_a_binned_count_bins_its_own_rows():
    data = pl.DataFrame({"v": [1.0, 1.5, 2.0, 4.0, 4.5, 6.0, 7.0, 7.5]})
    bare = fm.Chart(data).mark_bar().encode(x=fm.X("v", bin=True), y="count()")
    chart = bare.layer(fm.Layer(mark="text", encoding={"text": "count"}))
    assert [t.get("name", "")[:11] for t in _steps(chart, "bin")] == ["_layer_bin_"] * 2
    assert sum(float(t) for t in _labels(chart, bare)) == data.height


# ---------------------------------------------------------------------------
# A missing column of a layer's statistic is refused in the render's words.
# ---------------------------------------------------------------------------


_ENCODING_STATISTIC_SY = (
    "SecondaryY: field 'mean_y' is a statistic the base chart's encoding computes for its "
    "own layers, not a column of the base chart's data (SecondaryY draws the base chart's "
    "rows after its .transform(...) steps). Compute it with .transform(...) (e.g. "
    "fm.transform_aggregate), or draw it from its own chart with LayerChart(base, other, "
    "resolve={'y': 'independent'}). Available columns: ['g', 'c', 'y']"
)

_STATISTIC_REFUSALS: dict[str, tuple[Callable[[], Any], str]] = {
    "sum-of-a-typo": (
        lambda: _aggregated_bar("mean", None).layer(
            fm.Layer(mark="point", encoding={"y": fm.Y("nope", aggregate="sum")})
        ),
        _plain("y", "nope"),
    ),
    "label-naming-another-layers-statistic": (
        lambda: _aggregated_bar("mean", None).layer(
            fm.Layer(mark="text", encoding={"text": "mean_y"})
        ),
        _STATISTIC_SHARED.format(channel="text", name="mean_y"),
    ),
    "independent-y-label-naming-another-layers-statistic": (
        lambda: fm.LayerChart(
            _aggregated_bar("mean", None),
            fm.Chart(_BARS).mark_text().encode(x="g", y="sum(y)", text="mean_y"),
            resolve={"y": "independent"},
        ),
        _STATISTIC_INDEPENDENT.format(channel="text", name="mean_y"),
    ),
    "grouping-typo": (
        lambda: _aggregated_bar("mean", None).layer(
            fm.Layer(mark="point", encoding={"y": fm.Y("y", aggregate="sum"), "color": "nope"})
        ),
        _plain("color", "nope"),
    ),
    "plus-operand-over-a-named-only-chain": (
        lambda: _box() + fm.Chart(DF).mark_point().encode(x="g", y=fm.Y("nope", aggregate="sum")),
        _plain("y", "nope", layer=6),
    ),
}


@pytest.mark.parametrize("case", list(_STATISTIC_REFUSALS))
def test_a_layer_statistic_on_a_missing_column_is_refused(case):
    build, message = _STATISTIC_REFUSALS[case]
    with pytest.raises(ValueError) as exc:
        build().to_svg()
    assert str(exc.value) == message


def test_a_secondary_y_naming_the_charts_encoding_statistic_is_refused():
    with pytest.raises(ValueError) as exc:
        _aggregated_bar("mean", None) + SY("mean_y")
    assert str(exc.value) == _ENCODING_STATISTIC_SY


def test_a_secondary_y_over_an_aggregated_bar_draws_the_raw_column():
    chart = _aggregated_bar("mean", None) + SY("y")
    assert len(_secondary_series(chart.to_svg())) == _BARS.height


def test_a_layer_statistic_after_a_users_aggregate_keeps_the_transform_error():
    """The columns after a user's own ``.transform(...)`` are the engine's, so it reports them."""
    chart = (
        fm.Chart(_BARS)
        .transform(
            fm.transform_aggregate({"field": "y", "fn": "mean", "as": "mean_y"}, groupby=["g"])
        )
        .mark_line()
        .encode(x="g", y="mean_y")
        .layer(fm.Layer(mark="point", encoding={"y": fm.Y("y", aggregate="sum")}))
    )
    with pytest.raises(ValueError, match="^transform failed: "):
        chart.to_svg()


# ---------------------------------------------------------------------------
# An independent-y layer's y scale is its own.
# ---------------------------------------------------------------------------


def _bar_of(field: str) -> fm.Chart:
    return fm.Chart(df).mark_bar().encode(x="g", y=field)


@pytest.mark.parametrize(
    "build, oracle",
    [
        pytest.param(
            lambda: _bar_of("r") + SY("r"),
            lambda: fm.LayerChart(
                _bar_of("r"),
                fm.Chart(df).mark_line().encode(x="g", y="r"),
                resolve={"y": "independent"},
            ),
            id="same-field-line",
        ),
        pytest.param(
            lambda: _mean_by_g().mark_bar().encode(x="g", y="mean_y") + SY("mean_r"),
            lambda: fm.LayerChart(
                _mean_by_g().mark_bar().encode(x="g", y="mean_y"),
                _mean_by_g().mark_line().encode(x="g", y="mean_r"),
                resolve={"y": "independent"},
            ),
            id="transformed-aggregate",
        ),
    ],
)
def test_a_secondary_line_spans_its_own_data(build, oracle):
    """The secondary axis is the independent-y oracle's, not the bar's zero-anchored one."""
    svg, expected = build().to_svg(), oracle().to_svg()
    assert _right_axis_ticks(svg) == _right_axis_ticks(expected)
    assert [y for _x, y in _secondary_series(svg)] == [y for _x, y in _secondary_series(expected)]
    assert "<html" in build().interactive().to_html().lower()


def _mean_by_g() -> fm.Chart:
    return fm.Chart(df).transform(fm.transform_aggregate(_MEAN_R, _MEAN_Y, groupby=["g"]))


def test_a_secondary_bar_over_a_same_field_bar_starts_at_its_own_zero():
    svg = (_bar_of("r") + SY("r", mark="bar")).to_svg()
    zero = _right_axis_ticks(svg)[0.0]
    for rect in data_bar_rects(mark_layer(svg))[-df["g"].n_unique() :]:
        assert abs(float(rect["y"]) + float(rect["height"]) - zero) <= 0.05


# ---------------------------------------------------------------------------
# A host's conditionals reach every layer attached to it.
# ---------------------------------------------------------------------------

_PICK = fm.selection_point(fields=["g"], name="pick")


def _host() -> fm.Chart:
    cond = _PICK.when(fm.Color("g")).otherwise(fm.value("#cccccc"))
    return fm.Chart(df).mark_point().encode(x="v", y="y", color=cond).add_selection(_PICK)


# df2 with the host's conditional field `g`, so the conditional reaches a
# layer whose rows carry it (a frame lacking it is refused by the column gate).
_DF2_G = df2.with_columns(g=pl.Series(["a", "b"] * 4))


def _data_layer() -> fm.Layer:
    return fm.Layer(data=_DF2_G, mark="line", encoding={"x": "v", "y": "w"})


def _text_layer() -> fm.Layer:
    return fm.Layer(mark="text", encoding={"text": "g"})


def _scoped_batches(chart: Any) -> list[tuple[str, Any]]:
    scene = json.loads(chart.interactive().scene_json)
    return [
        (batch["kind"], batch.get("conditionals")) for p in scene["panels"] for batch in p["marks"]
    ]


_HOST_ORDERS = {
    "data-first": lambda: _host().layer(_data_layer(), _text_layer()),
    "data-middle": lambda: _host().layer(_text_layer(), _data_layer(), fm.Layer(mark="line")),
    "data-last": lambda: _host().layer(_text_layer(), _data_layer()),
    "separate-calls": lambda: _host().layer(_data_layer()).layer(fm.Layer(mark="line")),
}


@pytest.mark.parametrize("order", list(_HOST_ORDERS))
def test_a_hosts_conditional_reaches_every_layer_in_any_order(order):
    """Rejects binding the host's conditional to the layers it had when a data layer merged."""
    chart = _HOST_ORDERS[order]()
    wire = _wire(chart)
    assert [c["layers"] for c in wire["conditionals"]] == [list(range(len(wire["layers"])))]
    assert all(scope == [0] for _kind, scope in _scoped_batches(chart))


def test_a_composed_hosts_own_conditional_stays_chart_wide():
    cond = _PICK.when(fm.Color("g")).otherwise(fm.value("#cccccc"))
    own = fm.selection_point(fields=["y"], name="own")
    left = P().encode(size=own.when(fm.Size("y")).otherwise(fm.value(10))).add_selection(own)
    composed = (left + P2()).encode(color=cond).add_selection(_PICK)
    chart = composed.layer(_data_layer())
    scopes = {c["channel"]: c["layers"] for c in _wire(chart)["conditionals"]}
    assert scopes == {"size": [0], "color": [0, 1, 2]}


def test_a_data_layers_own_conditional_stays_on_its_layer():
    sel = fm.selection_point(fields=["w"], name="mine")
    cond = sel.when(fm.Color("w")).otherwise(fm.value("#cccccc"))
    layer = fm.Layer(data=df2, mark="point", encoding={"x": "v", "y": "w", "color": cond})
    chart = P().add_selection(sel).layer(_text_layer(), layer)
    assert [c["layers"] for c in _wire(chart)["conditionals"]] == [[2]]


def test_a_data_layer_takes_the_hosts_coordinate_system():
    chart = P().coord(fm.CoordFlip()).layer(_data_layer())
    assert "coord" not in _wire(chart)["layers"][1]
    assert _wire(chart)["coord"] == {"kind": "flip"}


# ---------------------------------------------------------------------------
# A facet partitions every operand's rows, whenever it is declared.
# ---------------------------------------------------------------------------

# A second frame with the facet column `g`, two rows per group.
_GROUPED = pl.DataFrame(
    {"v": [1.5, 2.5, 3.5, 4.5], "w": [2.0, 1.5, 3.0, 1.0], "g": ["a", "b", "a", "b"]}
)


def _faceted_plus() -> fm.Chart:
    """A faceted chart ``+`` an unfaceted one: the right operand has no facet to ignore, so nothing warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return P().facet("g") + fm.Chart(_GROUPED).mark_line().encode(x="v", y="w")


def _grouped_line(data: pl.DataFrame = _GROUPED) -> fm.Layer:
    return fm.Layer(data=data, mark="line", encoding={"x": "v", "y": "w"})


@pytest.mark.parametrize(
    "faceted_first, facet_last",
    [
        pytest.param(
            lambda: P().facet("g").layer(_grouped_line()),
            lambda: P().layer(_grouped_line()).facet("g"),
            id="data-layer",
        ),
        pytest.param(
            lambda: _faceted_plus(),
            lambda: (P() + fm.Chart(_GROUPED).mark_line().encode(x="v", y="w")).facet("g"),
            id="plus",
        ),
    ],
)
def test_a_facet_declared_before_the_merge_partitions_the_right_rows(faceted_first, facet_last):
    svg = faceted_first().to_svg()
    assert svg == facet_last().to_svg()
    assert [len(polylines(p)) for p in mark_groups(svg)] == [1, 1]


def test_a_faceted_plus_operand_with_its_own_facet_draws_its_rows():
    chart = P().facet("g") + fm.Chart(_GROUPED).mark_line().encode(x="v", y="w").facet("g")
    assert [len(polylines(p)) for p in mark_groups(chart.to_svg())] == [1, 1]


def test_rows_without_the_facet_column_join_no_panel_in_either_order():
    ungrouped = _GROUPED.drop("g")
    first = P().facet("g").layer(_grouped_line(ungrouped)).to_svg()
    last = P().layer(_grouped_line(ungrouped)).facet("g").to_svg()
    assert first == last


def test_a_data_layer_on_a_faceted_host_warns_nothing(recwarn):
    P().facet("g").layer(_grouped_line()).to_svg()
    assert not [w for w in recwarn if "theme/facet is ignored" in str(w.message)]


# ---------------------------------------------------------------------------
# The layering conflict warning names the caller's line.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "compose",
    [
        pytest.param(lambda: P() + P2().facet("g"), id="plus"),
        pytest.param(lambda: fm.LayerChart(P(), P2().facet("g"))._build_merged(), id="layer-chart"),
    ],
)
def test_the_layering_conflict_warning_names_the_callers_file(compose):
    with pytest.warns(UserWarning, match="theme/facet is ignored") as record:
        compose()
    assert {w.filename for w in record if "theme/facet" in str(w.message)} == {__file__}


# ---------------------------------------------------------------------------
# Every case renders interactively, and every refusal says the same thing there.
# ---------------------------------------------------------------------------

_RENDERING_CASES: dict[str, Callable[[], Any]] = {
    "histogram+secondary-bar": lambda: H() + SY("r", mark="bar"),
    "histogram+secondary-sample-line": lambda: H_VALUE() + SY("value", mark="line"),
    "histogram-rug": lambda: H().layer(fm.Layer(mark="tick")),
    "histogram-rug-at-sample-values": lambda: _rug_on_value_histogram(),
    "faceted-histogram-rug": lambda: _rug_on_value_histogram().facet("g"),
    "smooth+secondary": lambda: fm.Chart(df).mark_smooth().encode(x="v", y="y") + SY("r"),
    "smooth-band+secondary": lambda: (
        fm.Chart(df).mark_smooth(ci=0.95).encode(x="v", y="y") + SY("r")
    ),
    "point-then-smooth+secondary": lambda: (
        fm.Chart(df).mark_point().encode(x="v", y="y").mark_smooth() + SY("r")
    ),
    "calculate+secondary": lambda: _calculated_points() + SY("z"),
    "aggregate+secondary": lambda: (
        _mean_by_v(_MEAN_R, _MEAN_Y).encode(x="v", y="mean_y") + SY("mean_r")
    ),
    "composition+secondary": lambda: (fP() + P(df2)) + SY("r"),
    "nested-left": lambda: (H() + SY("r")) + P(df2),
    "nested-right": lambda: P(df2) + H().layer(fm.Layer(mark="tick")),
    "composition-layer": lambda: (fP() + P(df2)).layer(tick(x="v")),
    "data-layer": lambda: fP().layer(_data_layer()),
    "data-layer-over-the-hosts-frame": lambda: fP().layer(
        fm.Layer(data=df, mark="line", encoding={"x": "v", "y": "r"})
    ),
    "data-layer-orders": lambda: _HOST_ORDERS["data-first"](),
    "data-layer-own-conditional": lambda: P().layer(
        fm.Layer(
            data=df2,
            mark="point",
            encoding={
                "x": "v",
                "y": "w",
                "color": fm.selection_point(fields=["w"], name="mine")
                .when(fm.Color("w"))
                .otherwise(fm.value("#cccccc")),
            },
        )
    ),
    "transform-after-layer": lambda: (
        H().layer(fm.Layer(mark="tick")).transform(fm.transform_filter(_FILTER))
    ),
    "encode-after-layer": lambda: H().layer(fm.Layer(mark="tick")).encode(x="y"),
    **{f"encode-after-secondary-{name}": cases[0] for name, cases in _REENCODE_CASES.items()},
    "layer-chart-member": lambda: fm.LayerChart(_rug_on_value_histogram(), _value_points()),
    "layer-chart-member-merged": lambda: fm.LayerChart(
        _rug_on_value_histogram(), _value_points()
    )._build_merged(),
    "layer-chart-member-independent-y": lambda: fm.LayerChart(
        _rug_on_value_histogram(), _value_points(), resolve={"y": "independent"}
    ),
    "displot-layers": lambda: fm.displot(df, x="v", layers=[fm.Layer(mark="tick")]),
    "faceted-secondary": lambda: (H() + SY("r")).facet("g"),
    "aggregated-label": lambda: _labelled_mean_bar(),
    "finer-layer-statistic": lambda: _aggregated_bar("mean", None).layer(
        fm.Layer(mark="point", encoding={"y": fm.Y("y", aggregate="mean"), "color": "c"})
    ),
    "different-layer-statistic": lambda: _aggregated_bar("mean", None).layer(
        fm.Layer(mark="point", encoding={"y": fm.Y("y", aggregate="sum")})
    ),
    "faceted-aggregated-label": lambda: _labelled_mean_bar().facet("c"),
    "same-field-secondary": lambda: _bar_of("r") + SY("r"),
    "faceted-data-layer": lambda: P().facet("g").layer(_grouped_line()),
    "faceted-plus": lambda: _faceted_plus(),
}


@pytest.mark.parametrize("case", list(_RENDERING_CASES))
def test_every_rendering_case_renders_interactive(case):
    assert "<html" in _RENDERING_CASES[case]().interactive().to_html().lower()


_REFUSALS: dict[str, tuple[Callable[[], Any], str]] = {
    "histogram-statistic": (
        lambda: H().layer(fm.Layer(mark="point", encoding={"y": "count"})),
        _STATISTIC_SHARED.format(channel="y", name="count"),
    ),
    "density-statistic": (
        lambda: (
            fm.Chart(df)
            .mark_density()
            .encode(x="v")
            .layer(fm.Layer(mark="line", encoding={"y": "density"}))
        ),
        _STATISTIC_SHARED.format(channel="y", name="density"),
    ),
    "source-independent-y": (
        lambda: _mean_by_v(_MEAN_Y).encode(x="v", y="mean_y") + SY("r"),
        _SOURCE_INDEPENDENT,
    ),
    "source-shared": (
        lambda: _mean_by_v(_MEAN_Y).encode(x="v", y="mean_y").layer(tick(y="r")),
        _SOURCE_SHARED,
    ),
    "plain": (lambda: fP() + SY("nope"), _plain("y", "nope")),
    **_STATISTIC_REFUSALS,
}


@pytest.mark.parametrize("case", list(_REFUSALS))
def test_every_refusal_raises_the_same_string_interactive(case):
    build, message = _REFUSALS[case]
    with pytest.raises(ValueError) as exc:
        build().interactive().to_html()
    assert str(exc.value) == message


@pytest.mark.parametrize(
    "build, message",
    [
        pytest.param(lambda: H() + SY("count"), _STATISTIC_SY, id="histogram-statistic"),
        pytest.param(
            lambda: _aggregated_bar("mean", None) + SY("mean_y"),
            _ENCODING_STATISTIC_SY,
            id="encoding-statistic",
        ),
    ],
)
def test_a_secondary_y_refusal_raises_before_any_output(build, message):
    """``SecondaryY`` refuses a field it can check when it is added, so no output door is reached."""
    with pytest.raises(ValueError) as exc:
        build()
    assert str(exc.value) == message


# ---------------------------------------------------------------------------
# An encoding bin is grouped like the layer's aggregate, over shared edges.
# ---------------------------------------------------------------------------

# Two groups of different sizes: a grouped bin shares the pooled edges.
_BINNED = pl.DataFrame(
    {
        "v": [1.2, 2.7, 3.1, 4.8, 5.5, 6.1, 7.9, 8.4, 2.2, 6.6],
        "g": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "a"],
    }
)


def _binned_count(**grouping: Any) -> fm.Chart:
    return fm.Chart(_BINNED).mark_bar().encode(x=fm.X("v", bin=True), y="count()", **grouping)


def _inverse(ladder: dict[float, float], px: float) -> float:
    lo, hi = min(ladder), max(ladder)
    return lo + (px - ladder[lo]) * (hi - lo) / (ladder[hi] - ladder[lo])


def _bars_by_fill(svg: str) -> dict[str, list[tuple[float, float, float]]]:
    """Each fill's bars as ``(start, end, height)`` in data units, in drawing order."""
    x, y = value_axis_ladder(svg, axis="x"), value_axis_ladder(svg, axis="y")
    out: dict[str, list[tuple[float, float, float]]] = {}
    for rect in data_bar_rects(mark_layer(svg)):
        left, top = float(rect["x"]), float(rect["y"])
        right, base = left + float(rect["width"]), top + float(rect["height"])
        out.setdefault(rect["fill"], []).append(
            (
                round(_inverse(x, left), 3),
                round(_inverse(x, right), 3),
                round(_inverse(y, top) - _inverse(y, base), 3),
            )
        )
    return out


def test_a_grouped_binned_count_counts_each_group_into_the_pooled_edges():
    """Every group is counted into one edge set: the ungrouped Bin's over all rows."""
    grouped = _bars_by_fill(_binned_count(color="g").to_svg())
    pooled = next(iter(_bars_by_fill(_binned_count().to_svg()).values()))
    edges = [start for start, _end, _h in pooled] + [pooled[-1][1]]
    assert len(grouped) == _BINNED["g"].n_unique()
    counted = sorted([h for _s, _e, h in bars] for bars in grouped.values())
    expected = sorted(
        np.histogram(_BINNED.filter(pl.col("g") == g)["v"].to_numpy(), bins=edges)[0].tolist()
        for g in _BINNED["g"].unique()
    )
    assert counted == expected
    for bars in grouped.values():
        assert [(start, end) for start, end, _h in bars] == [(s, e) for s, e, _h in pooled]


def test_a_grouped_histogram_stack_draws_aligned_bins():
    chart = fm.Chart(_BINNED).mark_histogram(multiple="stack").encode(x="v", color="g")
    spans = {
        fill: [(s, e) for s, e, _h in bars] for fill, bars in _bars_by_fill(chart.to_svg()).items()
    }
    assert len(set(map(tuple, spans.values()))) == 1


@pytest.mark.parametrize(
    "grouping",
    [
        pytest.param({"color": "g"}, id="color"),
        pytest.param({"size": "g"}, id="size"),
        pytest.param({"opacity": "g"}, id="opacity"),
        pytest.param({"detail": "g"}, id="detail"),
    ],
)
def test_a_binned_count_groups_by_its_grouping_channel(grouping):
    bin_step = _steps(_binned_count(**grouping), "bin")[0]
    assert (bin_step["groupby"], bin_step["shared_extent"]) == (["g"], True)


def test_a_faceted_binned_count_groups_by_the_facet():
    bin_step = _steps(_binned_count().facet("g"), "bin")[0]
    assert bin_step["groupby"] == ["g"]


def test_an_attached_layer_keeps_the_binned_bars_grouped():
    rule = fm.Layer(mark="rule", encoding={"y": fm.Y("v", aggregate="mean")})
    chart = _binned_count(color="g").layer(rule)
    # The y-only rule draws no x, so it takes no binned x: only the bars bin,
    # per group, and the rule averages v per group of the color it inherits.
    assert [step["groupby"] for step in _steps(chart, "bin")] == [["g"]]
    assert [step["groupby"] for step in _steps(chart, "aggregate")] == [["g"]]
    assert len(_bars_by_fill(chart.to_svg())) == _BINNED["g"].n_unique()


def test_a_plus_operand_keeps_the_binned_bars_grouped():
    chart = _binned_count(color="g") + fm.Chart(_BINNED).mark_point().encode(x="v", y="v")
    assert [step["groupby"] for step in _steps(chart, "bin")] == [["g"]]
    assert len(_bars_by_fill(chart.to_svg())) == _BINNED["g"].n_unique()


def test_a_count_scaled_density_overlay_matches_its_stacked_bars():
    """Each group's overlay covers that group's count of bars of the shared width."""
    rng = np.random.default_rng(3)
    data = pl.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, 200), rng.normal(1.5, 1, 80)]),
            "g": ["a"] * 200 + ["b"] * 80,
        }
    )
    chart = fm.displot(data, x="x", hue="g", kind="hist", kde=True, stat="count", multiple="stack")
    svg = chart.to_svg()
    widths = {
        round(end - start, 6) for bars in _bars_by_fill(svg).values() for start, end, _h in bars
    }
    assert len(widths) == 1
    width = widths.pop()
    x, y = value_axis_ladder(svg, axis="x"), value_axis_ladder(svg, axis="y")
    areas = []
    for _stroke, points in polylines(mark_layer(svg)):
        xs = [_inverse(x, px) for px, _py in points]
        ys = [_inverse(y, py) for _px, py in points]
        areas.append(float(np.trapezoid(ys, xs)) / width)
    assert sorted(areas) == pytest.approx([80, 200], rel=0.03)


# ---------------------------------------------------------------------------
# The render gate's message names any channel encode() accepts.
# ---------------------------------------------------------------------------


def test_the_gate_message_names_every_encode_channel_as_written():
    from ferrum._core import layer_unknown_column_message
    from ferrum.encoding._channel_policy import _SPEC_KNOWN_CHANNELS

    for channel in sorted(_SPEC_KNOWN_CHANNELS):
        message = layer_unknown_column_message(1, channel, "nope", "plain", False)
        assert f"encoding '{channel}' references unknown column 'nope'" in message
    with pytest.raises(ValueError):
        layer_unknown_column_message(1, "not_a_channel", "nope", "plain", False)


def test_the_gate_message_names_a_layer_a_public_call_built_by_that_call():
    """With ``entry``, the formatter speaks as the render gate does for a desugared layer."""
    from ferrum._core import layer_unknown_column_message

    assert layer_unknown_column_message(1, "size", "nope", "plain", False, entry="lmplot") == (
        "lmplot: encoding 'size' references unknown column 'nope' (not present in that "
        "layer's data)"
    )
    assert layer_unknown_column_message(1, "size", "nope", "plain", False) == _plain("size", "nope")


@pytest.mark.parametrize("interactive", [False, True], ids=["static", "interactive"])
def test_a_figure_built_layers_missing_statistic_input_is_named_by_the_figure(interactive):
    """A figure's scatter layer aggregating a column its rows lack is refused as ``lmplot``.

    The check runs in Python, where the layer's rows are known, with the
    gate's own words. Rejects naming the figure's internal ``layer 0``,
    which the caller never placed.
    """
    rng = np.random.default_rng(0)
    frame = pl.DataFrame({"x": rng.normal(0, 1, 30), "y": rng.normal(0, 1, 30)})
    chart = fm.lmplot(frame, x="x", y="y", encode={"size": fm.Size("nope", aggregate="mean")})
    with pytest.raises(ValueError) as exc:
        chart.interactive().to_html() if interactive else chart.to_svg()
    assert str(exc.value) == (
        "lmplot: encoding 'size' references unknown column 'nope' (not present in that "
        "layer's data)"
    )


def test_the_gate_channels_are_the_encode_channels_plus_the_wire_only_ones():
    """Both directions: the gate names exactly the channels a layer's wire encoding can carry.

    Every channel ``encode()`` accepts, except ``tooltip_field`` (warned and
    dropped before the wire), plus the two wire-only channels Rust's
    encoding carries itself (``order``, the ``tooltip_fields`` list).
    """
    from ferrum._core import layer_unknown_column_message
    from ferrum.encoding import _channel_class_map

    with pytest.raises(ValueError) as exc:
        layer_unknown_column_message(1, "not_a_channel", "nope", "plain", False)
    named = set(re.findall(r"'(\w+)'", str(exc.value).split("expected one of", 1)[1]))
    assert named == (set(_channel_class_map()) - {"tooltip_field"}) | {"order", "tooltip_fields"}


# ``order`` is a wire-only channel: ``Layer(encoding=)`` refuses it as an
# unknown channel at construction, as ``encode()`` does, so it never reaches
# the gate from a layer.
_GROUPING_TYPOS = ["fill", "stroke", "detail", "color", "tooltip", "stroke_dash"]


def _typo_grouping(channel: str) -> fm.Chart:
    layer = fm.Layer(mark="point", encoding={"y": fm.Y("y", aggregate="sum"), channel: "nope"})
    return _aggregated_bar("mean", None).layer(layer)


@pytest.mark.parametrize("interactive", [False, True], ids=["static", "interactive"])
@pytest.mark.parametrize("channel", _GROUPING_TYPOS)
def test_a_grouping_typo_names_its_channel(channel, interactive):
    chart = _typo_grouping(channel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError) as exc:
            chart.interactive().to_html() if interactive else chart.to_svg()
    assert str(exc.value) == _plain(channel, "nope")


# ---------------------------------------------------------------------------
# The statistic check reads the rows the layer reads.
# ---------------------------------------------------------------------------


def _calculated_left() -> fm.Chart:
    return (
        fm.Chart(df)
        .transform(fm.transform_calculate("z", "datum.r * 2"))
        .mark_point()
        .encode(x="v", y="y")
    )


def test_a_layer_statistic_reads_a_column_the_left_operands_transform_made():
    chart = (_calculated_left() + P(df2)).layer(
        fm.Layer(mark="point", encoding={"y": fm.Y("z", aggregate="sum")})
    )
    assert "<svg" in chart.to_svg()
    assert '"field":"z","fn":"sum"' in chart.to_spec().to_json().replace(" ", "")


def test_a_nested_composition_layer_statistic_reads_the_calculated_column():
    chart = ((_calculated_left() + P(df2)) + P(df2)).layer(
        fm.Layer(mark="point", encoding={"y": fm.Y("z", aggregate="sum")})
    )
    assert "<svg" in chart.to_svg()


def test_a_typo_after_a_users_transform_keeps_the_transform_error():
    chart = (_calculated_left() + P(df2)).layer(
        fm.Layer(mark="point", encoding={"y": fm.Y("nope", aggregate="sum")})
    )
    with pytest.raises(ValueError, match="^transform failed: "):
        chart.to_svg()


def test_a_chart_data_reader_over_source_rows_gets_the_plain_message():
    chart = H().layer(fm.Layer(mark="point", encoding={"y": fm.Y("nope", aggregate="sum")}))
    with pytest.raises(ValueError) as exc:
        chart.to_svg()
    assert str(exc.value) == _plain("y", "nope")


# ---------------------------------------------------------------------------
# A drawn channel's aggregate is honored, flat and per layer.
# ---------------------------------------------------------------------------

_HONORED = {
    "text": ("text", fm.Text),
    "tooltip": ("point", fm.Tooltip),
    "color": ("point", fm.Color),
    "fill": ("point", fm.Fill),
    "stroke": ("point", fm.Stroke),
    "size": ("point", fm.Size),
    "opacity": ("point", fm.Opacity),
    "fill_opacity": ("point", fm.FillOpacity),
    "stroke_opacity": ("point", fm.StrokeOpacity),
    "stroke_width": ("line", fm.StrokeWidth),
    "angle": ("point", fm.Angle),
}


def _numeric_bars() -> pl.DataFrame:
    return _BARS.with_columns(z=pl.col("y") * 10)


@pytest.mark.parametrize("channel", list(_HONORED))
def test_a_drawn_channels_aggregate_is_computed_flat(channel):
    mark, cls = _HONORED[channel]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        value = cls("z", aggregate="mean")
    chart = fm.Chart(_numeric_bars()).mark_point().encode(x="g", y="mean(y)", **{channel: value})
    steps = _steps(chart, "aggregate")
    assert len(steps) == 1
    assert sorted(op["as"] for op in steps[0]["ops"]) == ["mean_y", "mean_z"]
    assert "mean_z" in [ch.get("field") for ch in _wire(chart)["encoding"].values()]
    assert "<svg" in chart.to_svg()


@pytest.mark.parametrize("channel", list(_HONORED))
def test_a_drawn_channels_aggregate_is_computed_per_layer(channel):
    mark, cls = _HONORED[channel]
    layer = fm.Layer(mark=mark, encoding={channel: cls("z", aggregate="mean")})
    chart = fm.Chart(_numeric_bars()).mark_bar().encode(x="g", y="mean(y)").layer(layer)
    # The attached layer computes its own aggregate; the bars, which never
    # see the layer's binding (spec §4.6 V3), compute only their mean y.
    steps = _steps(chart, "aggregate")
    assert [sorted(op["as"] for op in step["ops"]) for step in steps] == [
        ["mean_y"],
        ["mean_y", "mean_z"],
    ]
    assert "<svg" in chart.to_svg()


def test_a_flat_bar_with_a_text_and_a_color_aggregate_runs_one_aggregate():
    chart = (
        fm.Chart(_numeric_bars())
        .mark_bar()
        .encode(
            x="g",
            y="mean(y)",
            text=fm.Text("y", aggregate="mean"),
            color=fm.Color("z", aggregate="mean"),
        )
    )
    steps = _steps(chart, "aggregate")
    assert len(steps) == 1
    assert [op["as"] for op in steps[0]["ops"]] == ["mean_y", "mean_z"]
    assert len(data_bar_rects(mark_layer(chart.to_svg()))) == _BARS["g"].n_unique()


def test_a_drawn_aggregate_label_is_warning_free():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _labelled_mean_bar().to_svg()
        fm.Chart(_BARS).mark_text().encode(
            x="g", y="g", text=fm.Text("y", aggregate="mean")
        ).to_svg()


# ---------------------------------------------------------------------------
# A chart-level scale channel's domain spans every layer that draws it.
# ---------------------------------------------------------------------------

# Each row is its own (g, y), so a point layer's per-(g, y) mean of z is z.
_SCALED = pl.DataFrame(
    {
        "g": ["a", "a", "b", "b", "c", "c"],
        "y": [1.0, 3.0, 2.0, 6.0, 4.0, 5.0],
        "z": [10.0, 30.0, 20.0, 60.0, 40.0, 90.0],
    }
)


def _points_beside(channel: str, cls: type, agg: str = "mean") -> fm.Chart:
    """Raw points ``+`` a mean-y layer whose ``channel`` carries ``agg`` of ``z``."""
    points = fm.Chart(_SCALED).mark_point().encode(x="g", y="y")
    aggregated = (
        fm.Chart(_SCALED)
        .mark_point()
        .encode(x="g", y="mean(y)", **{channel: cls("z", aggregate=agg)})
    )
    return points + aggregated


_OPACITY_CLASSES = {
    "opacity": fm.Opacity,
    "fill_opacity": fm.FillOpacity,
    "stroke_opacity": fm.StrokeOpacity,
}


def _mark_styles(svg: str) -> list[str]:
    """Each plot-area mark element with its geometry removed: what it draws, not where.

    Sorted, so two renders compare as multisets.
    """
    geometry = r' (?:cx|cy|x|y|x1|x2|y1|y2|width|height|points|d|transform)="[^"]*"'
    return sorted(
        re.sub(geometry, "", element)
        for element in re.findall(
            r"<(?:circle|rect|path|line|polyline|text) [^>]*/?>", mark_layer(svg)
        )
    )


def _assert_draws_its_layers_alone(chart: Any, *parts: fm.Chart) -> None:
    """Each layer draws what it draws alone: the marks equal the parts' marks, styles included."""
    assert _mark_styles(chart.to_svg()) == sorted(
        style for part in parts for style in _mark_styles(part.to_svg())
    )


def _raw_beside_parts() -> fm.Chart:
    return fm.Chart(_SCALED).mark_point().encode(x="g", y="y")


def _aggregated_part(channel: str, cls: type, agg: str = "mean") -> fm.Chart:
    return (
        fm.Chart(_SCALED)
        .mark_point()
        .encode(x="g", y="mean(y)", **{channel: cls("z", aggregate=agg)})
    )


@pytest.mark.parametrize("channel", list(_OPACITY_CLASSES))
def test_an_aggregated_opacity_beside_raw_points_leaves_the_raw_points_at_the_default(channel):
    """V5: the raw points draw as they do alone; only the aggregating layer varies its opacity.

    Rejects promotion: the raw points would draw their own z through the
    aggregating layer's opacity scale.
    """
    chart = _points_beside(channel, _OPACITY_CLASSES[channel])
    _assert_draws_its_layers_alone(
        chart, _raw_beside_parts(), _aggregated_part(channel, _OPACITY_CLASSES[channel])
    )
    assert "<html" in chart.interactive().to_html().lower()


def _legend_values(svg: str) -> list[float]:
    """The numeric labels drawn after the last bold (legend title) text."""
    end = [m.end() for m in re.finditer(r'font-weight="600"[^>]*>[^<]*<', svg)][-1]
    return [float(v) for v in re.findall(r">(-?[0-9.]+)<", svg[end:])]


_LEGEND_TITLE = 'font-weight="600" text-anchor="start">{}<'


def test_an_aggregated_size_beside_raw_points_is_titled_by_its_column():
    """V5/V11: the raw points keep the default radius; the size legend is titled ``mean_z``."""
    chart = _points_beside("size", fm.Size)
    svg = chart.to_svg()
    assert svg.count(_LEGEND_TITLE.format("mean_z")) == 1
    _assert_draws_its_layers_alone(chart, _raw_beside_parts(), _aggregated_part("size", fm.Size))
    assert "<html" in chart.interactive().to_html().lower()


def test_an_aggregated_color_beside_raw_points_spans_only_the_drawing_layer():
    """V5/V10: the legend's domain is the one drawing layer's means per g (20..65).

    Rejects promotion: the raw points' own z (10..90) would widen it.
    """
    chart = _points_beside("color", fm.Color)
    svg = chart.to_svg()
    assert svg.count(_LEGEND_TITLE.format("mean_z")) == 1
    values = _legend_values(svg)
    means = _SCALED.group_by("g").agg(pl.col("z").mean())["z"]
    assert (min(values), max(values)) == (means.min(), means.max())
    _assert_draws_its_layers_alone(chart, _raw_beside_parts(), _aggregated_part("color", fm.Color))
    assert "<html" in chart.interactive().to_html().lower()


def test_a_color_domain_spans_two_layers_that_compute_different_values():
    """V10: each layer binds its own sum; the union spans per (g, y) 10..90 and per g 40..130."""
    raw = fm.Chart(_SCALED).mark_point().encode(x="g", y="y", color=fm.Color("z", aggregate="sum"))
    chart = raw + _aggregated_part("color", fm.Color, agg="sum")
    values = _legend_values(chart.to_svg())
    assert min(values) <= 10 and max(values) >= 130


def test_a_faceted_aggregated_size_renders_every_panels_points():
    chart = _points_beside("size", fm.Size).facet("g")
    panels = mark_groups(chart.to_svg())
    per_group = _SCALED.group_by("g").len()["len"].to_list()
    assert sum(_census(p)["circle"] for p in panels) == _SCALED.height + len(per_group)


# ---------------------------------------------------------------------------
# A channel that draws no aggregated value refuses aggregate=.
# ---------------------------------------------------------------------------

# Each channel class and the encode() key that builds it. The span ends and the
# polar channels draw the value they are bound to, so they honor aggregate=
# (test_layer_encoding_model.py's S2 and P2 cells).
_NO_VALUE_CHANNELS = {
    "Detail": "detail",
    "Shape": "shape",
    "StrokeDash": "stroke_dash",
    "Key": "key",
    "Href": "href",
    "Description": "description",
    "Url": "url",
    "Facet": "facet",
    "FacetRow": "facet_row",
    "FacetCol": "facet_col",
    "XError": "x_error",
    "XError2": "x_error2",
    "YError": "y_error",
    "YError2": "y_error2",
}


def _no_value_message(name: str) -> str:
    return (
        f"{name}(aggregate='mean') is not supported: {name} draws no aggregated value. "
        "Compute the aggregate with .transform(fm.transform_aggregate(...)) and bind its "
        "output column, or drop aggregate=."
    )


def _channel_class(name: str) -> type:
    import ferrum.encoding as encoding

    return getattr(fm, name, None) or getattr(encoding, name)


@pytest.mark.parametrize("name", list(_NO_VALUE_CHANNELS))
def test_a_channel_object_that_draws_no_value_refuses_aggregate(name):
    with pytest.raises(ValueError) as exc:
        _channel_class(name)("z", aggregate="mean")
    assert str(exc.value) == _no_value_message(name)


@pytest.mark.parametrize("name", list(_NO_VALUE_CHANNELS))
def test_encode_shorthand_on_a_channel_that_draws_no_value_refuses_aggregate(name):
    with pytest.raises(ValueError) as exc:
        fm.Chart(df).mark_point().encode(**{_NO_VALUE_CHANNELS[name]: "mean(z)"})
    assert str(exc.value) == _no_value_message(name)


# ---------------------------------------------------------------------------
# A layer whose field type differs from the layered scale's stays out of it.
# ---------------------------------------------------------------------------

_MIXED = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 3.0, 2.0, 4.0],
        "z": [1.0, 2.0, 3.0, 4.0],
        "g": ["a", "b", "a", "b"],
    }
)


def _mixed_point(**encoding: Any) -> fm.Chart:
    return fm.Chart(_MIXED).mark_point().encode(x="x", y="y", **encoding)


def _mixed_line(**encoding: Any) -> fm.Chart:
    return fm.Chart(_MIXED).mark_line().encode(x="x", y="y", **encoding)


_MIXED_CASES: dict[str, tuple[Callable[[], Any], str]] = {
    "color-quantitative-then-nominal": (
        lambda: _mixed_point(color="z") + _mixed_line(color="g"),
        "color",
    ),
    "color-quantitative-then-nominal-layer": (
        lambda: _mixed_point(color="z").layer(fm.Layer(mark="line", encoding={"color": "g"})),
        "color",
    ),
    "color-nominal-then-quantitative": (
        lambda: _mixed_point(color="g") + _mixed_line(color="z"),
        "color",
    ),
    "size": (lambda: _mixed_point(size="z") + _mixed_point(size="g"), "size"),
    # The opacity family draws a legend the commit before the layer data model
    # did not; with it suppressed, the marks render those same bytes.
    "opacity": (
        lambda: _mixed_point(opacity=fm.Opacity("z", legend=None)) + _mixed_point(opacity="g"),
        "opacity",
    ),
    "fill_opacity": (
        lambda: (
            _mixed_point(fill_opacity=fm.FillOpacity("z", legend=None))
            + _mixed_point(fill_opacity="g")
        ),
        "fill_opacity",
    ),
    "stroke_opacity": (
        lambda: (
            _mixed_point(stroke_opacity=fm.StrokeOpacity("z", legend=None))
            + _mixed_point(stroke_opacity="g")
        ),
        "stroke_opacity",
    ),
}

# The SVG each mixed chart renders, as the commit before the layer data model
# rendered it.
_MIXED_SVG_DIGESTS = {
    "color-quantitative-then-nominal": "e7cff10ee6b066b24ba9cc095246333b33b7e1a6af51943f2e9af6156846eef5",
    "color-quantitative-then-nominal-layer": "e7cff10ee6b066b24ba9cc095246333b33b7e1a6af51943f2e9af6156846eef5",
    "color-nominal-then-quantitative": "b349d656a5c1b3664c2339577c835179034db8c618bf7ac5f2164f794f9d3d7d",
    "size": "b44851036d2c30d2c3decf9efc1b73a646e2fd875f50659b68f2b0bbb0c8090f",
    "opacity": "9461e2f784d9b8d0cdf26a5f5bdc9a2030c4a26b1998dc3ac1f340b0689f2cfd",
    "fill_opacity": "04b6b258f339783cf68ce2b379c17adfc149c6d62598b6bd9600f3d4b9d37cec",
    "stroke_opacity": "88d704b7e8b507cfd115e8aaec461fe4e2c30d528c3c28699effe7a8fcd41c08",
}


def _excluded(channel: str) -> str:
    return (
        f"layer 1: its {channel} scale differs from the one layer 0 establishes the {channel} "
        "scale with, so it stays out of that scale; its marks draw through that scale and may "
        "fall outside its legend"
    )


def _scale_warnings(build: Callable[[], Any]) -> tuple[str, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = build().to_svg()
    return svg, [str(w.message) for w in caught if "establishes the" in str(w.message)]


@pytest.mark.parametrize("case", list(_MIXED_CASES))
def test_a_layer_of_another_field_type_stays_out_of_the_layered_scale(case):
    build, channel = _MIXED_CASES[case]
    svg, messages = _scale_warnings(build)
    assert messages == [_excluded(channel)]
    assert _digest(svg) == _MIXED_SVG_DIGESTS[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert "<html" in build().interactive().to_html().lower()


@pytest.mark.parametrize("channel", ["opacity", "fill_opacity", "stroke_opacity"])
def test_an_excluded_opacity_layer_leaves_one_legend_over_the_establishing_values(channel):
    """The string layer stays out of the scale and its legend; layer 0's ``z`` titles the one legend."""
    svg, messages = _scale_warnings(
        lambda: _mixed_point(**{channel: "z"}) + _mixed_point(**{channel: "g"})
    )
    assert messages == [_excluded(channel)]
    texts = _texts(svg)
    assert texts.count("z") == 1 and "g" not in texts
    assert texts[-7:] == ["1", "1.5", "2", "2.5", "3", "3.5", "4"]


def test_a_nominal_color_legend_lists_only_the_establishing_categories():
    svg, _messages = _scale_warnings(_MIXED_CASES["color-nominal-then-quantitative"][0])
    assert _texts(svg)[-2:] == ["a", "b"]


def test_a_type_compatible_color_layer_joins_the_scale_without_a_warning():
    svg, messages = _scale_warnings(lambda: _mixed_point(color="z") + _mixed_line(color="z"))
    assert messages == []
    assert _digest(svg) == _MIXED_SVG_DIGESTS["color-quantitative-then-nominal"]


def test_a_nominal_size_establishing_the_size_scale_is_refused():
    """Layer 0 establishes the size scale, and a size scale cannot read a string column."""
    with pytest.raises(ValueError) as exc:
        (_mixed_point(size="g") + _mixed_point(size="z")).to_svg()
    assert str(exc.value) == "size: column 'g' has unsupported dtype: Utf8"


# ---------------------------------------------------------------------------
# A + operand's statistic stays on that operand's own layers.
# ---------------------------------------------------------------------------

# Duplicate (g, y) rows: a raw layer that aggregated by (g, y) would collapse them.
_DUPLICATED = pl.DataFrame(
    {
        "g": ["a", "a", "a", "b", "b", "c", "c", "c"],
        "y": [1.0, 1.0, 3.0, 2.0, 2.0, 4.0, 5.0, 5.0],
        "z": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    }
)


def _raw_points() -> fm.Chart:
    return fm.Chart(_DUPLICATED).mark_point().encode(x="g", y="y")


def _mean_points(**encoding: Any) -> fm.Chart:
    return fm.Chart(_DUPLICATED).mark_point().encode(x="g", y="mean(y)", **encoding)


_RAW_BESIDE = {
    "color": lambda: _raw_points() + _mean_points(color=fm.Color("z", aggregate="mean")),
    "size": lambda: _raw_points() + _mean_points(size=fm.Size("z", aggregate="mean")),
}
_RAW_BESIDE_PARTS = {
    "color": lambda: _mean_points(color=fm.Color("z", aggregate="mean")),
    "size": lambda: _mean_points(size=fm.Size("z", aggregate="mean")),
}


@pytest.mark.parametrize("channel", list(_RAW_BESIDE))
def test_a_right_operands_statistic_leaves_the_left_operands_rows_raw(channel):
    """V5: the left draws its raw rows uncolored and unsized; one legend titled ``mean_z``.

    Rejects promotion: the left would draw its raw z through the scale, and
    rejects D1-B's grain leak: the left would aggregate by (g, y).
    """
    chart = _RAW_BESIDE[channel]()
    svg = chart.to_svg()
    assert svg.count(_LEGEND_TITLE.format("mean_z")) == 1
    _assert_draws_its_layers_alone(chart, _raw_points(), _RAW_BESIDE_PARTS[channel]())
    assert [step for step in _steps(chart, "aggregate") if step["groupby"] == ["g", "y"]] == []
    assert "<html" in chart.interactive().to_html().lower()


def test_a_layer_statistic_leaves_the_hosts_rows_raw():
    layer = fm.Layer(
        mark="point",
        encoding={"y": fm.Y("y", aggregate="mean"), "size": fm.Size("z", aggregate="mean")},
    )
    chart = _raw_points().layer(layer)
    groups = _DUPLICATED["g"].n_unique()
    assert _census(mark_layer(chart.to_svg()))["circle"] == _DUPLICATED.height + groups
    assert [step["groupby"] for step in _steps(chart, "aggregate")] == [["g"]]
    assert "<html" in chart.interactive().to_html().lower()


def test_a_layered_operand_keeps_its_chart_level_statistic_on_its_own_layers():
    """Parent rule: a binding written on ``(a + b)`` moves onto that operand's layers."""
    layered = (
        _mean_points() + fm.Chart(_DUPLICATED).mark_line().encode(x="g", y="mean(y)")
    ).encode(color=fm.Color("z", aggregate="mean"))
    chart = _raw_points() + layered
    wire = _wire(chart)
    assert "color" not in wire["encoding"]
    assert [layer["encoding"].get("color", {}).get("field") for layer in wire["layers"]] == [
        None,
        "mean_z",
        "mean_z",
    ]
    groups = _DUPLICATED["g"].n_unique()
    assert _census(mark_layer(chart.to_svg()))["circle"] == _DUPLICATED.height + groups


def test_a_left_operands_statistic_in_the_other_order_leaves_the_raw_rows_raw():
    """V5 reversed: the legend spans only the means; the raw points draw uncolored."""
    chart = _mean_points(color=fm.Color("z", aggregate="mean")) + _raw_points()
    svg = chart.to_svg()
    assert svg.count(_LEGEND_TITLE.format("mean_z")) == 1
    groups = _DUPLICATED["g"].n_unique()
    assert _census(mark_layer(svg))["circle"] == _DUPLICATED.height + groups
    _assert_draws_its_layers_alone(
        chart, _mean_points(color=fm.Color("z", aggregate="mean")), _raw_points()
    )
    values = _legend_values(svg)
    means = _DUPLICATED.group_by("g").agg(pl.col("z").mean())["z"]
    assert (min(values), max(values)) == (means.min(), means.max())
    assert "<html" in chart.interactive().to_html().lower()


# ---------------------------------------------------------------------------
# SecondaryY's statistic check reads the base mark's own layers.
# ---------------------------------------------------------------------------


def test_a_secondary_y_typo_beside_an_attached_layer_keeps_the_plain_message():
    with pytest.raises(ValueError) as exc:
        H().layer(fm.Layer(mark="tick", encoding={"x": "nope"})) + SY("nope")
    assert str(exc.value) == (
        "SecondaryY: field 'nope' is not a column of the base chart's data (SecondaryY "
        "draws from the same table as the base chart); available columns: "
        "['v', 'y', 'r', 'g']"
    )
