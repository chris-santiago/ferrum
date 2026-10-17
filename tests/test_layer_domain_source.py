"""Each layer's positional domain comes from the batch that layer draws.

A layer reading a named output (a stat layer's own result, a filtered
operand's rows) contributes that output's values to its axis, and a layer on
the chart's data contributes the chart's rows. A column of the same name in
some other batch never widens the axis: a smoothing band does not stretch to
the raw minimum, a user column named ``lower`` does not leak into an error
bar's extent, and a filtered operand of ``+`` spans only the rows it keeps.

Every assertion reads the rendered tick labels, the observable proxy for the
domain Rust resolves.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import axis_tick_labels, y_axis_extents

_RNG = np.random.default_rng(7)
_X = _RNG.uniform(0, 10, 120)
_Y = 2 * _X + _RNG.normal(0, 3, 120)
# One far-low point: the raw minimum a stat layer's own output never reaches.
_RAW_MIN = -40.0
_Y[0] = _RAW_MIN
SMOOTH = pl.DataFrame({"x": _X, "y": _Y})

_GROUPS = pl.DataFrame(
    {
        "g": ["a"] * 40 + ["b"] * 40,
        "v": np.concatenate([_RNG.normal(0, 1, 40), _RNG.normal(1, 1, 40)]),
    }
)
# A user column that shares its name with a field the error bar's own output carries.
_USER_LOWER = -100.0
GROUPS_WITH_LOWER = _GROUPS.with_columns(lower=pl.lit(_USER_LOWER))


def _ticks(svg: str, axis: str) -> list[float]:
    return sorted(float(label) for label in axis_tick_labels(svg, axis=axis))


def _y_ticks(chart) -> list[float]:
    return _ticks(chart.to_svg(), "y")


def _x_ticks(chart) -> list[float]:
    return _ticks(chart.to_svg(), "x")


# ---------------------------------------------------------------------------
# A stat layer spans its own output.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smooth",
    [
        pytest.param(dict(ci=0.95), id="loess-ci"),
        pytest.param(dict(method="lm", ci=0.95), id="lm-ci"),
        pytest.param(dict(method="lm", ci=None), id="lm-no-ci"),
    ],
)
def test_a_smooth_spans_its_fit_not_the_raw_minimum(smooth):
    """The fit and its band are far above the one raw point at -40."""
    ticks = _y_ticks(fm.Chart(SMOOTH).mark_smooth(**smooth).encode(x="x", y="y"))
    assert ticks[0] > _RAW_MIN / 2, ticks


def test_a_scatter_beside_the_smooth_still_reaches_the_raw_minimum():
    """The point layer draws the raw rows, so the axis keeps their minimum."""
    chart = fm.Chart(SMOOTH).mark_point().encode(x="x", y="y") + fm.Chart(SMOOTH).mark_smooth(
        ci=0.95
    ).encode(x="x", y="y")
    assert _y_ticks(chart)[0] <= _RAW_MIN


def test_faceted_smooths_share_the_y_axis_of_every_panel():
    """A facet-shared axis still reads every panel's fit."""
    faceted = SMOOTH.with_columns(
        panel=pl.Series(["p", "q"] * (SMOOTH.height // 2)),
        y=pl.when(pl.int_range(pl.len()) % 2 == 0).then(pl.col("y")).otherwise(pl.col("y") + 30),
    )
    chart = fm.Chart(faceted).mark_smooth(ci=0.95).encode(x="x", y="y").facet(col="panel")
    extents = y_axis_extents(chart.to_svg())
    assert len(extents) == 2
    assert extents[0] == extents[1]
    # The shared axis spans the higher panel's fit too.
    assert extents[0].hi >= 40


# ---------------------------------------------------------------------------
# A user column never widens a layer whose own output has a same-named field.
# ---------------------------------------------------------------------------


def _errorbar(data):
    return fm.Chart(data).mark_errorbar(extent="stdev").encode(x="g", y="v")


def _points(data):
    return fm.Chart(data).mark_point().encode(x="g", y="v")


def test_a_user_column_named_lower_does_not_reach_the_error_bar_axis():
    assert _y_ticks(_errorbar(GROUPS_WITH_LOWER)) == _y_ticks(_errorbar(_GROUPS))


def test_a_layer_on_the_charts_rows_does_not_open_them_to_the_error_bar():
    """A mixed chart: the points read the chart's rows, the error bar its own output.

    Reading the chart's rows for every layer once any layer reads them would
    take the error bar's ``lower`` from the user column and stretch to -100.
    """
    mixed = _errorbar(GROUPS_WITH_LOWER) + _points(GROUPS_WITH_LOWER)
    plain = _errorbar(_GROUPS) + _points(_GROUPS)
    ticks = _y_ticks(mixed)
    assert ticks[0] > _USER_LOWER / 2, ticks
    assert ticks == _y_ticks(plain)


# ---------------------------------------------------------------------------
# Distribution marks span what they draw.
# ---------------------------------------------------------------------------


def test_a_boxen_spans_its_boxes_and_its_drawn_outliers():
    """The letter-value boxes plus the outliers beyond them cover exactly the raw range."""
    boxen = fm.Chart(_GROUPS).mark_boxen().encode(x="g", y="v")
    assert _y_ticks(boxen) == _y_ticks(_points(_GROUPS))


def test_a_boxplot_spans_its_drawn_outlier():
    data = _GROUPS.with_columns(
        v=pl.when(pl.int_range(pl.len()) == 0).then(60.0).otherwise(pl.col("v"))
    )
    ticks = _y_ticks(fm.Chart(data).mark_boxplot().encode(x="g", y="v"))
    assert ticks[-1] >= 60


def test_a_histogram_and_a_density_cover_the_raw_range_they_bin():
    for chart in (
        fm.Chart(SMOOTH).mark_histogram().encode(x="y"),
        fm.Chart(SMOOTH).mark_density().encode(x="y"),
    ):
        ticks = _x_ticks(chart)
        assert ticks[0] <= _RAW_MIN and ticks[-1] >= float(np.max(_Y)) - 5, ticks


def test_an_ecdf_climbs_to_the_row_count():
    ticks = _y_ticks(fm.displot(SMOOTH, x="y", kind="ecdf"))
    assert ticks[-1] == SMOOTH.height, ticks


# ---------------------------------------------------------------------------
# Layer 0's batch seeds the axis, and an empty batch adds nothing.
# ---------------------------------------------------------------------------

_SEED = pl.DataFrame(
    {"a": [1.0, 2.0, 3.0, 4.0], "v": [1.0, 2.0, 3.0, 200.0], "w": [2.0, 3.0, 4.0, 5.0]}
)


def test_layer_zeros_seed_field_is_read_from_its_own_batch():
    """Layer 0 draws only the rows it keeps, so its 200 never reaches the axis."""
    kept = fm.Chart(_SEED).transform(fm.transform_filter("datum.v < 100"))
    layer0 = kept.mark_point().encode(x="a", y="v")
    other = fm.Chart(_SEED).mark_point().encode(x="a", y="w")
    ticks = _y_ticks(layer0 + other)
    assert ticks[-1] < 100, ticks
    assert ticks == _y_ticks(fm.LayerChart(layer0, other))


def test_an_empty_batch_contributes_nothing():
    base = fm.Chart(_SEED).mark_point().encode(x="a", y="w")
    empty = (
        fm.Chart(_SEED)
        .transform(fm.transform_filter("datum.v > 1000"))
        .mark_point()
        .encode(x="a", y="v")
    )
    assert _y_ticks(base + empty) == _y_ticks(base)


# ---------------------------------------------------------------------------
# `+` over different frames spans what each layer draws.
# ---------------------------------------------------------------------------

_LEFT = pl.DataFrame({"a": [1.0, 2.0, 3.0, 40.0]})
_RIGHT = pl.DataFrame({"a": [5.0, 6.0]})


def test_a_filtered_operand_spans_only_the_rows_it_keeps():
    """The left frame's 40 is filtered out, so the x axis ends near the right frame's 6."""
    left = fm.Chart(_LEFT).transform(fm.transform_filter("datum.a < 3")).mark_tick().encode(x="a")
    right = fm.Chart(_RIGHT).mark_tick().encode(x="a")
    ticks = _x_ticks(left + right)
    assert ticks[-1] < 10, ticks
    assert ticks == _x_ticks(fm.LayerChart(left, right))


def test_two_unfiltered_frames_span_both():
    left = fm.Chart(_LEFT).mark_tick().encode(x="a")
    right = fm.Chart(_RIGHT).mark_tick().encode(x="a")
    ticks = _x_ticks(left + right)
    assert ticks[0] <= 5 and ticks[-1] >= 40, ticks


@pytest.mark.parametrize("mark, widens", [("point", True), ("tick", False)])
def test_a_layer_inheriting_a_re_encoded_chart_field_widens_the_axis_only_where_it_draws_x(
    mark, widens
):
    """A layer binding only y takes the composition's x="c" only when its mark draws x.

    A point draws x, so it draws ``c`` and the axis covers 300. A y-only tick
    is a one-axis mark: it takes no inherited x, so the axis stays layer
    0's ``a`` (1 to 3).
    """
    frame = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0], "c": [100.0, 200.0, 300.0]})
    other = getattr(fm.Chart(frame), f"mark_{mark}")().encode(y="b")
    chart = (fm.Chart(frame).mark_point().encode(x="a", y="b") + other).encode(x="c")
    ticks = _x_ticks(chart)
    assert (ticks[-1] >= 300) is widens, ticks
