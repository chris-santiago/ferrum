"""`a + b` draws each operand's own transform chain, as `fm.LayerChart(a, b)` does.

A composed chart runs one Rust pipeline: unnamed steps chain, named steps
publish an output a layer reads through ``data_source``. ``+`` routes each
operand's chain so it reads the operand's own source rows: a right-hand chain
becomes a named step (one step) or one ``Branch`` (several steps) ahead of the
left-hand chain, and a right-hand layer with no chain of its own reads a named
identity of the source whenever the left-hand chain has unnamed steps. A
per-layer aggregate or bin reads the layer's own named input. A layer built
on the chart itself (``Chart.layer``, ``SecondaryY``) reads that chart's
data instead: its rows after the chart's own ``.transform(...)`` steps, never
a composite mark's statistics.

The static ``fm.LayerChart`` overlay tree lowers every operand on its own
pipeline, so its marks are the oracle: each composed chart below draws the
same mark geometry as the LayerChart of the same operands.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import axis_tick_labels

_N = 241
_VALUES = [sum(math.sin(i * k * 0.7311 + k) for k in range(1, 7)) for i in range(_N)]
DF = pl.DataFrame(
    {
        "value": _VALUES,
        "y": [v * 0.5 + math.cos(i) for i, v in enumerate(_VALUES)],
        "g": ["A" if i % 2 else "B" for i in range(_N)],
    }
)
# The same columns shifted, so a composed chart over two frames has overlapping names.
DF_SHIFTED = DF.with_columns(pl.col("value") + 3.0)
# Rows with value > 0: the filtered operand's own data.
_N_POSITIVE = sum(v > 0 for v in _VALUES)


def H(data=DF, **kwargs):
    return fm.Chart(data).mark_histogram(**kwargs).encode(x="value")


def D(data=DF, **kwargs):
    return fm.Chart(data).mark_density(**kwargs).encode(x="value")


def T(data=DF):
    return fm.Chart(data).mark_tick().encode(x="value")


def P(data=DF):
    return fm.Chart(data).mark_point().encode(x="value", y="y")


def S(data=DF):
    return fm.Chart(data).mark_smooth().encode(x="value", y="y")


def _positive():
    return fm.transform_filter("datum.value > 0")


def _binned_bar(data=DF):
    return (
        fm.Chart(data)
        .mark_bar()
        .encode(x=fm.X("value", bin=True), y=fm.Y("value", aggregate="count"))
    )


def _mean_rule(data=DF):
    return fm.Chart(data).mark_rule().encode(x=fm.X("value", aggregate="mean"))


def _contour(data=DF):
    return fm.Chart(data).mark_contour().encode(x="value", y="y")


def _box(data=DF):
    return fm.Chart(data).mark_boxplot().encode(x="g", y="value")


def _plot_marks(svg: str) -> str:
    """The markup inside the plot's clipped mark groups (axes and legends excluded)."""
    body = re.sub(r"<defs>.*?</defs>", "", svg, flags=re.S)
    return "".join(
        re.findall(r'<g clip-path="url\(#ferrum-clip-\d+\)">(.*?)</g>', body, flags=re.S)
    )


def _census(svg: str) -> dict:
    marks = _plot_marks(svg)
    return {
        "rects": len(re.findall(r"<rect ", marks)),
        "paths": len(re.findall(r"<path ", marks)),
        "lines": len(re.findall(r"<line ", marks)),
        "circles": len(re.findall(r"<circle ", marks)),
        "polylines": len(re.findall(r"<polyline ", marks)),
    }


def _paths(svg: str) -> list[str]:
    return re.findall(r'<path d="([^"]*)"', _plot_marks(svg))


def _rect_heights(svg: str) -> list[float]:
    return [
        round(float(h), 1) for h in re.findall(r'<rect [^>]*height="([\d.]+)"', _plot_marks(svg))
    ]


# ---------------------------------------------------------------------------
# Each composed chart draws the LayerChart oracle's marks.
# ---------------------------------------------------------------------------

_ORACLE_CASES = [
    pytest.param(lambda: (H(), D()), id="histogram+density"),
    pytest.param(lambda: (D(), H()), id="density+histogram"),
    pytest.param(lambda: (H(bins=5), H(bins=20)), id="histogram5+histogram20"),
    pytest.param(lambda: (D(bandwidth=0.5), D(bandwidth=2.0)), id="density-bw0.5+density-bw2"),
    pytest.param(lambda: (H(), T()), id="histogram+tick"),
    pytest.param(lambda: (D(), T()), id="density+tick"),
    pytest.param(lambda: (S(), P()), id="smooth+point"),
    pytest.param(
        lambda: (P(), fm.Chart(DF).transform(_positive()).mark_density().encode(x="value")),
        id="point+filtered-density",
    ),
    pytest.param(
        lambda: (fm.Chart(DF).transform(_positive()).mark_point().encode(x="value", y="y"), D()),
        id="filtered-point+density",
    ),
    pytest.param(lambda: (H(), _binned_bar()), id="histogram+binned-bar"),
    pytest.param(lambda: (D(), _binned_bar()), id="density+binned-bar"),
    pytest.param(lambda: (H(), D(DF_SHIFTED)), id="histogram+density-of-overlapping-frame"),
    pytest.param(lambda: (D(DF_SHIFTED), T()), id="density-of-overlapping-frame+tick"),
    # A composite whose chain mixes an unnamed step with a named output its
    # layers read (contour's Kde2D then Contour), on either side.
    pytest.param(lambda: (P(), _contour()), id="point+contour"),
    pytest.param(lambda: (_contour(), P()), id="contour+point"),
    pytest.param(
        lambda: (
            fm.Chart(DF).transform(_positive()).mark_point().encode(x="value", y="y"),
            _contour(),
        ),
        id="filtered-point+contour",
    ),
    # A composite whose layers each read one named output (boxplot), after a filter.
    pytest.param(
        lambda: (fm.Chart(DF).transform(_positive()).mark_tick().encode(x="g", y="value"), _box()),
        id="filtered-tick+boxplot",
    ),
]


@pytest.mark.parametrize("operands", _ORACLE_CASES)
def test_plus_draws_the_layerchart_marks(operands):
    """Every mark of ``a + b`` equals the overlay tree's mark for the same operands."""
    a, b = operands()
    composed = (a + b).to_svg()
    oracle = fm.LayerChart(a, b).to_svg()
    assert _plot_marks(composed) == _plot_marks(oracle)


def test_three_operands_draw_each_chain_once():
    """``H + D + T`` draws the histogram, one density curve and every sample tick."""
    census = _census((H() + D() + T()).to_svg())
    assert census == {"rects": 9, "paths": 1, "lines": _N, "circles": 0, "polylines": 0}


def test_a_tick_layer_after_two_chains_draws_the_sample():
    """``T + H + D``: the tick layer reads the source rows, not the density grid."""
    census = _census((T() + H() + D()).to_svg())
    assert census == {"rects": 9, "paths": 1, "lines": _N, "circles": 0, "polylines": 0}


def test_two_bandwidths_draw_two_distinct_curves():
    first, second = _paths((D(bandwidth=0.5) + D(bandwidth=2.0)).to_svg())
    assert first != second


def test_a_right_hand_tick_reads_its_own_rows_not_the_left_filter():
    """The left operand's filter narrows only the left operand's ticks."""
    left = fm.Chart(DF).transform(_positive()).mark_tick().encode(x="value")
    census = _census((left + T()).to_svg())
    assert census["lines"] == _N_POSITIVE + _N


def test_a_right_hand_filter_is_applied_before_its_density():
    """A two-step right-hand chain runs both steps: the curve is the filtered density."""
    filtered = fm.Chart(DF).transform(_positive()).mark_density().encode(x="value")
    unfiltered = fm.LayerChart(P(), D()).to_svg()
    composed = (P() + filtered).to_svg()
    assert _paths(composed) != _paths(unfiltered)
    assert _paths(composed) == _paths(fm.LayerChart(P(), filtered).to_svg())


def test_binned_bar_heights_after_a_density_equal_the_standalone_bars():
    """The per-layer bin reads the source rows, not the density grid."""
    standalone = _rect_heights(_binned_bar().to_svg())
    composed = _rect_heights((D() + _binned_bar()).to_svg())
    assert composed == standalone


def test_an_aggregated_layer_after_a_histogram_draws_the_sample_mean():
    """The per-layer aggregate reads the source rows, not the histogram's bins."""
    composed = _plot_marks((H() + _mean_rule()).to_svg())
    oracle = _plot_marks(fm.LayerChart(H(), _mean_rule()).to_svg())
    assert len(re.findall(r"<line ", composed)) == 1
    assert composed == oracle


def test_two_contours_over_different_frames_each_draw_their_own_output():
    """Each side's contour publishes one fixed output name from its own rows.

    The right side's chain runs as a branch, which publishes its outputs under
    its own name, so neither side's contours replace the other's.
    """
    a, b = _contour(), _contour(DF_SHIFTED)
    assert _plot_marks((a + b).to_svg()) == _plot_marks(fm.LayerChart(a, b).to_svg())


def _tick_labels(svg: str) -> list[str]:
    """Every text label the chart draws: axis ticks, axis titles and legend entries."""
    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


def _point_by_group(data=DF):
    return fm.Chart(data).mark_point().encode(x="g", y="value")


def _mean_bar(data=DF):
    return fm.Chart(data).mark_bar().encode(x="g", y=fm.Y("value", aggregate="mean"))


def _violin(data=DF):
    return fm.Chart(data).mark_violin().encode(x="g", y="value")


def _imputed_ticks(data=DF):
    return (
        fm.Chart(data)
        .transform(fm.transform_impute("value", value=0.0), fm.transform_impute("y", value=0.0))
        .mark_tick()
        .encode(x="value")
    )


_OTHER_COLUMNS = DF_SHIFTED.rename({"value": "w", "y": "y2"}).with_columns(
    pl.Series("c", ["p", "q"] * (_N // 2) + ["p"])
)


@pytest.mark.parametrize(
    "operands",
    [
        pytest.param(lambda: (_box(), _box(DF_SHIFTED)), id="boxplot+boxplot-other-frame"),
        pytest.param(
            lambda: (_box(), _box(DF_SHIFTED.rename({"y": "w"}))),
            id="boxplot+boxplot-overlapping-frame",
        ),
        pytest.param(lambda: (_violin(), _violin(DF_SHIFTED)), id="violin+violin"),
        pytest.param(lambda: (_box(), _point_by_group(DF_SHIFTED)), id="boxplot+point"),
        pytest.param(lambda: (_mean_bar(), _mean_bar(DF_SHIFTED)), id="mean-bar+mean-bar"),
        # A nested branch from an earlier `+` reads only its operand's rows:
        # imputing over the combined rows would turn the other frame's
        # padding into extra ticks.
        pytest.param(
            lambda: (P() + _imputed_ticks(), T(DF_SHIFTED)), id="nested-imputed-ticks+tick"
        ),
        pytest.param(
            lambda: (H(), fm.Chart(_OTHER_COLUMNS).mark_point().encode(x="w", y="y2", color="c")),
            id="histogram+colored-point",
        ),
        pytest.param(lambda: (_contour(), _contour(DF_SHIFTED)), id="contour+contour"),
        pytest.param(lambda: (H(), D(DF_SHIFTED)), id="histogram+density-overlapping-frame"),
    ],
)
def test_operands_over_different_frames_each_read_their_own_rows(operands):
    """Each operand's chain runs over only the rows it brought, as LayerChart draws it."""
    a, b = operands()
    composed = (a + b).to_svg()
    oracle = fm.LayerChart(a, b).to_svg()
    assert _plot_marks(composed) == _plot_marks(oracle)
    assert _tick_labels(composed) == _tick_labels(oracle)


def test_two_contours_render_on_the_interactive_route():
    """LayerChart's interactive output renders the merged chart ``+`` builds."""
    a, b = _contour(), _contour(DF_SHIFTED)
    merged = fm.LayerChart(a, b)._build_merged()
    assert _plot_marks(merged.to_svg()) == _plot_marks(fm.LayerChart(a, b).to_svg())
    assert "<html" in fm.LayerChart(a, b).interactive().to_html().lower()


def test_two_boxplots_render_on_the_interactive_route():
    a, b = _box(), _box(DF_SHIFTED)
    merged = fm.LayerChart(a, b)._build_merged().to_svg()
    oracle = fm.LayerChart(a, b).to_svg()
    assert "<html" in fm.LayerChart(a, b).interactive().to_html().lower()
    assert _plot_marks(merged) == _plot_marks(oracle)
    assert _tick_labels(merged) == _tick_labels(oracle)


def test_two_contours_render_with_independent_y():
    """Each member's contour layer reads its own output and draws its own segments."""
    a, b = _contour(), _contour(DF_SHIFTED)
    chart = fm.LayerChart(a, b, resolve={"y": "independent"})
    sources = [
        layer["data_source"]
        for layer in json.loads(chart._build_merged().to_spec().to_json())["layers"]
    ]
    assert len(set(sources)) == 2
    assert _census(chart.to_svg()) == _census(fm.LayerChart(a, b).to_svg())


def test_two_boxplots_with_independent_y_meet_only_the_single_y_layer_rule():
    """A boxplot member is refused by the independent-y member rule, not by an output-name clash."""
    with pytest.raises(
        ValueError, match="member charts under independent y must be a single y-layer"
    ):
        fm.LayerChart(_box(), _box(DF_SHIFTED), resolve={"y": "independent"}).to_svg()


def test_nested_composites_over_three_frames_each_draw_their_own_output():
    """An inner ``+``'s branch is addressed again under the outer one."""
    third = DF.with_columns(pl.col("value") - 3.0)
    a, b, c = P(), _contour(DF_SHIFTED), _contour(third)
    assert _plot_marks((a + (b + c)).to_svg()) == _plot_marks(fm.LayerChart(a, b, c).to_svg())


# ---------------------------------------------------------------------------
# A right-hand layer that inherits its own chart's x/y keeps them when routed.
# ---------------------------------------------------------------------------


def _line_with_points(data=DF):
    return fm.Chart(data).mark_line(point=True).encode(x="value", y="y")


def _filtered_line_with_points(data=DF):
    return fm.Chart(data).transform(_positive()).mark_line(point=True).encode(x="value", y="y")


def _line_and_points_sharing_y():
    """A layered operand whose layers own x and inherit y from the chart."""
    keep = _positive()
    line = fm.Chart(DF).transform(keep).mark_line().encode(x="value")
    points = fm.Chart(DF).transform(keep).mark_point().encode(x="value")
    return (line + points).encode(y="y")


def _residual_frame() -> pl.DataFrame:
    y_true = [math.sin(i) for i in range(40)]
    y_pred = [t + 0.3 * math.cos(3 * i) for i, t in enumerate(y_true)]
    residual = [t - p for t, p in zip(y_true, y_pred)]
    return pl.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": residual,
            "studentized_residual": [r / 0.3 for r in residual],
            "cooks_distance": [abs(r) / 10 for r in residual],
        }
    )


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "operands",
    [
        pytest.param(lambda: (H(), _line_with_points()), id="histogram+line-with-points"),
        pytest.param(lambda: (D(), _line_with_points()), id="density+line-with-points"),
        pytest.param(
            lambda: (P(), _filtered_line_with_points()), id="point+filtered-line-with-points"
        ),
        pytest.param(
            lambda: (P(), _line_with_points(DF_SHIFTED)),
            id="point+line-with-points-overlapping-frame",
        ),
        pytest.param(
            lambda: (P(), _filtered_line_with_points(DF_SHIFTED)),
            id="point+filtered-line-with-points-overlapping-frame",
        ),
        pytest.param(lambda: (H(), _line_and_points_sharing_y()), id="histogram+layers-sharing-y"),
        pytest.param(
            lambda: (
                H(),
                P().layer(fm.Layer(mark="rule", encoding={"x": "value"})),
            ),
            id="histogram+points-with-x-only-rules",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_residual_frame()).mark_histogram().encode(x="y_pred"),
                fm.Chart(_residual_frame())
                .mark_residuals()
                .encode(x="y_pred", y="studentized_residual"),
            ),
            id="histogram+residuals-reference-rule",
        ),
    ],
)
def test_a_routed_layer_keeps_its_operands_positional_channels(operands):
    """A layer routed to a named output does not inherit the chart's x/y.

    ``+`` gives it its own operand's channels instead, marked as inherited, so
    a reference rule that owns y alone still spans the panel's x.
    """
    a, b = operands()
    composed = _plot_marks((a + b).to_svg())
    assert composed == _plot_marks(fm.LayerChart(a, b).to_svg())
    assert "<circle" in composed or "<line" in composed


# ---------------------------------------------------------------------------
# A left-hand layer reading its own rows keeps taking the chart's x/y.
# ---------------------------------------------------------------------------

_UV = pl.DataFrame({"u": [1.0, 2.0, 3.0], "v": [3.0, 2.0, 1.0]})


def _unencoded_filtered_points():
    return fm.Chart(DF).transform(_positive()).mark_point()


def _uv_line():
    return fm.Chart(_UV).mark_line().encode(x="u", y="v")


def test_a_chart_level_encode_after_the_merge_reaches_the_left_operands_rows():
    """The left chain runs over its own rows, and its layer still reads the composed chart's x/y."""
    after = (_unencoded_filtered_points() + _uv_line()).encode(x="value", y="y")
    before = _unencoded_filtered_points().encode(x="value", y="y") + _uv_line()
    svg = after.to_svg()
    assert svg.count("<circle") == _N_POSITIVE
    assert _plot_marks(svg) == _plot_marks(before.to_svg())


def test_a_composed_chart_as_an_operand_keeps_its_own_x_and_y():
    """Once a composed chart is an operand, its own encoding stays with the layers it gave x/y."""
    composed = _unencoded_filtered_points() + _uv_line()
    a = composed.encode(x="value", y="y")
    b = composed.encode(x="value", y="value")
    assert _plot_marks((a + b).to_svg()) == _plot_marks(fm.LayerChart(a, b).to_svg())


# ---------------------------------------------------------------------------
# A chart-level x/y bound on the composition reaches a routed right layer too.
# ---------------------------------------------------------------------------

_AB_LEFT = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
_AB_RIGHT = pl.DataFrame({"a": [10.0, 20.0], "b": [10.0, 20.0]})


def _bare_points(data, *chain):
    chart = fm.Chart(data)
    if chain:
        chart = chart.transform(*chain)
    return chart.mark_point()


@pytest.mark.parametrize(
    "right",
    [
        pytest.param(lambda: _bare_points(_AB_RIGHT), id="overlapping-frame-no-chain"),
        pytest.param(
            lambda: _bare_points(_AB_RIGHT, fm.transform_filter("datum.a > 0")),
            id="overlapping-frame-with-chain",
        ),
    ],
)
def test_an_encode_after_a_cross_frame_merge_reaches_the_right_layer(right):
    """The right layer draws the right frame's own a/b, as LayerChart draws it."""
    composed = (_bare_points(_AB_LEFT) + right()).encode(x="a", y="b")
    oracle = fm.LayerChart(
        _bare_points(_AB_LEFT).encode(x="a", y="b"), right().encode(x="a", y="b")
    )
    svg = composed.to_svg()
    assert svg.count("<circle") == _AB_LEFT.height + _AB_RIGHT.height
    assert _plot_marks(svg) == _plot_marks(oracle.to_svg())
    assert _tick_labels(svg) == _tick_labels(oracle.to_svg())


def test_an_encode_after_a_same_data_merge_reaches_the_routed_right_layer():
    right = _bare_points(DF, _positive())
    composed = (_bare_points(DF) + right).encode(x="value", y="y")
    oracle = _bare_points(DF).encode(x="value", y="y") + right.encode(x="value", y="y")
    svg = composed.to_svg()
    assert svg.count("<circle") == _N + _N_POSITIVE
    assert _plot_marks(svg) == _plot_marks(oracle.to_svg())


# ---------------------------------------------------------------------------
# The left operand's written x/y reaches a right layer that binds none; a
# binding a composite placed reaches it only once the composition binds it.
# ---------------------------------------------------------------------------

_RIGHT_ROUTES = {
    "overlapping-frame-no-chain": lambda: _bare_points(_AB_RIGHT),
    "overlapping-frame-with-chain": lambda: _bare_points(
        _AB_RIGHT, fm.transform_filter("datum.a > 0")
    ),
    "same-data-with-chain": lambda: _bare_points(_AB_LEFT, fm.transform_filter("datum.a > 1")),
}
_RIGHT_ROWS = {
    "overlapping-frame-no-chain": _AB_RIGHT.height,
    "overlapping-frame-with-chain": _AB_RIGHT.height,
    "same-data-with-chain": 2,
}


def _left_ab():
    return _bare_points(_AB_LEFT).encode(x="a", y="b")


@pytest.fixture(params=sorted(_RIGHT_ROUTES))
def route(request):
    return request.param


def test_the_left_operands_written_x_y_reaches_a_right_layer_that_binds_none(route):
    right = _RIGHT_ROUTES[route]
    svg = (_left_ab() + right()).to_svg()
    oracle = fm.LayerChart(_left_ab(), right().encode(x="a", y="b")).to_svg()
    assert svg.count("<circle") == _AB_LEFT.height + _RIGHT_ROWS[route]
    assert _plot_marks(svg) == _plot_marks(oracle)


def test_a_same_data_right_operand_without_a_chain_inherits_natively():
    svg = (_left_ab() + _bare_points(_AB_LEFT)).to_svg()
    assert svg.count("<circle") == 2 * _AB_LEFT.height


@pytest.mark.parametrize(
    "encode",
    [
        pytest.param(lambda c: c.encode(x="a", y="b"), id="exact-re-encode"),
        pytest.param(lambda c: c.encode(x="b").encode(x="a", y="b"), id="after-another-binding"),
    ],
)
def test_re_encoding_the_left_operands_binding_draws_every_right_point(route, encode):
    right = _RIGHT_ROUTES[route]
    svg = encode(_left_ab() + right()).to_svg()
    assert svg.count("<circle") == _AB_LEFT.height + _RIGHT_ROWS[route]
    assert _plot_marks(svg) == _plot_marks(
        fm.LayerChart(_left_ab(), right().encode(x="a", y="b")).to_svg()
    )


def test_a_partial_re_encode_reaches_both_channels(route):
    right = _RIGHT_ROUTES[route]
    svg = (_left_ab() + right()).encode(x="a", y="a").to_svg()
    assert svg.count("<circle") == _AB_LEFT.height + _RIGHT_ROWS[route]
    oracle = fm.LayerChart(_left_ab().encode(y="a"), right().encode(x="a", y="a")).to_svg()
    assert _plot_marks(svg) == _plot_marks(oracle)


def _histogram_left():
    return fm.Chart(_AB_LEFT).mark_histogram().encode(x="a")


def _y_only_rule(route):
    chart = fm.Chart(_AB_LEFT if route == "same-data-with-chain" else _AB_RIGHT)
    if route != "overlapping-frame-no-chain":
        chart = chart.transform(fm.transform_filter("datum.a > 0"))
    return chart.mark_rule().encode(y="b")


def test_a_right_layer_takes_the_left_binding_as_written_even_after_a_retitle(route):
    """The y-only rule takes no x at all, never the histogram's binned bin_start.

    A one-axis rule draws no x, so it takes neither the histogram's written
    ``x="a"`` nor its resolved ``bin_start``, before or after a retitle.
    """
    composed = _histogram_left() + _y_only_rule(route)
    for chart in (composed, composed.labs(x="T")):
        rule = json.loads(chart.to_spec().to_json())["layers"][-1]
        assert "x" not in rule["encoding"], rule
    assert _plot_marks(composed.labs(x="T").to_svg()) == _plot_marks(composed.to_svg())


def test_a_placed_binding_reaches_a_right_layer_once_the_composition_binds_it(route):
    right = _RIGHT_ROUTES[route]
    composed = (_histogram_left() + right()).encode(x="a", y="b")
    assert composed.to_svg().count("<circle") == _RIGHT_ROWS[route]


@pytest.mark.parametrize("ci", [None, 0.95], ids=["smooth", "smooth-ci"])
def test_a_smooth_left_operand_hands_its_written_x_y_to_a_right_layer(route, ci):
    """mark_smooth rewrites its x/y to the fit's columns; the right layer gets the written a/b."""
    right = _RIGHT_ROUTES[route]
    left = fm.Chart(_AB_LEFT).mark_smooth(ci=ci).encode(x="a", y="b")
    svg = (left + right()).to_svg()
    assert svg.count("<circle") == _RIGHT_ROWS[route]
    assert _plot_marks(svg) == _plot_marks(
        fm.LayerChart(left, right().encode(x="a", y="b")).to_svg()
    )


@pytest.mark.parametrize(
    "compose",
    [
        pytest.param(lambda right: _histogram_left() + right(), id="bare-right"),
        pytest.param(
            lambda right: (_histogram_left() + right()).encode(x="a"), id="composition-binds-x"
        ),
    ],
)
def test_a_right_layer_left_without_y_is_refused_as_layerchart_refuses_it(route, compose):
    """The histogram's count y was never written, so the right point has no y to plot."""
    right = _RIGHT_ROUTES[route]
    with pytest.raises(ValueError, match="needs both positional channels, but no 'y'"):
        compose(right).to_svg()
    with pytest.raises(ValueError):
        fm.LayerChart(_histogram_left(), right()).to_svg()


def test_an_encode_on_an_outer_composition_reaches_every_nested_right_layer():
    inner = _histogram_left() + _bare_points(_AB_RIGHT, fm.transform_filter("datum.a > 0"))
    third = pl.DataFrame({"a": [5.0], "b": [5.0]})
    composed = (inner + _bare_points(third)).encode(x="a", y="b")
    assert composed.to_svg().count("<circle") == _AB_RIGHT.height + third.height


# ---------------------------------------------------------------------------
# The composition's written bindings survive nesting and later encodes.
# ---------------------------------------------------------------------------

_ABC_LEFT = pl.DataFrame(
    {"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 3.0, 2.0, 4.0], "c": [100.0, 200.0, 300.0, 400.0]}
)
_ABC_RIGHT = pl.DataFrame({"a": [10.0, 20.0], "b": [10.0, 20.0], "c": [500.0, 600.0]})
_ABC_THIRD = pl.DataFrame({"a": [30.0, 40.0, 50.0], "b": [30.0, 40.0, 50.0]})


def _abc_right(route):
    if route == "overlapping-frame-no-chain":
        return _bare_points(_ABC_RIGHT)
    if route == "overlapping-frame-with-chain":
        return _bare_points(_ABC_RIGHT, fm.transform_filter("datum.a > 0"))
    return _bare_points(_ABC_LEFT, fm.transform_filter("datum.a > 1"))


def _abc_right_rows(route):
    return 3 if route == "same-data-with-chain" else _ABC_RIGHT.height


_WRITING_LEFTS = {
    "point": lambda: _bare_points(_ABC_LEFT).encode(x="a", y="b"),
    "line": lambda: fm.Chart(_ABC_LEFT).mark_line().encode(x="a", y="b"),
    "smooth": lambda: fm.Chart(_ABC_LEFT).mark_smooth().encode(x="a", y="b"),
    "smooth-ci": lambda: fm.Chart(_ABC_LEFT).mark_smooth(ci=0.95).encode(x="a", y="b"),
}


def _right_field(chart, channel="x"):
    """The field the last layer reads on *channel*, without the merge's rename suffix."""
    layer = json.loads(chart.to_spec().to_json())["layers"][-1]
    return layer["encoding"][channel]["field"].split("__rhs_")[0]


@pytest.mark.parametrize("left", ["point", "smooth", "histogram"])
def test_an_encode_after_a_rewriting_left_reaches_the_right_layer(route, left):
    """A later x binding replaces the left's written one for the right layer."""
    lefts = {
        **_WRITING_LEFTS,
        "histogram": lambda: fm.Chart(_ABC_LEFT).mark_histogram().encode(x="a"),
    }
    composed = (lefts[left]() + _abc_right(route)).encode(x="c")
    assert _right_field(composed) == "c"


@pytest.mark.parametrize("left", sorted(_WRITING_LEFTS))
def test_both_groupings_of_three_operands_draw_every_right_point(left):
    """``a + b + c`` and ``a + (b + c)`` both hand the left's written x/y on."""
    third = _bare_points(_ABC_THIRD)
    left_points = _ABC_LEFT.height if left == "point" else 0
    for composed in (
        _WRITING_LEFTS[left]() + _bare_points(_ABC_RIGHT) + third,
        _WRITING_LEFTS[left]() + (_bare_points(_ABC_RIGHT) + third),
    ):
        assert composed.to_svg().count("<circle") == (
            left_points + _ABC_RIGHT.height + _ABC_THIRD.height
        )


def test_an_encode_on_an_inner_composition_reaches_the_outer_right_layer():
    inner = (_WRITING_LEFTS["smooth"]() + _bare_points(_ABC_RIGHT)).encode(x="c")
    composed = inner + _bare_points(_ABC_THIRD.with_columns(c=pl.col("a") * 10))
    layers = json.loads(composed.to_spec().to_json())["layers"]
    assert [layer["encoding"]["x"]["field"].split("__rhs_")[0] for layer in layers[-2:]] == [
        "c",
        "c",
    ]
    assert composed.to_svg().count("<circle") == _ABC_RIGHT.height + _ABC_THIRD.height


def test_a_composed_right_operand_hands_its_own_written_x_y_to_its_layers():
    right = _bare_points(_ABC_RIGHT).encode(x="a", y="b") + _bare_points(_ABC_THIRD)
    composed = fm.Chart(_ABC_LEFT).mark_smooth().encode(x="c", y="c") + right
    assert _right_field(composed) == "a"
    assert _right_field(composed, "y") == "b"
    assert composed.to_svg().count("<circle") == _ABC_RIGHT.height + _ABC_THIRD.height


def test_a_right_layer_drawn_once_the_composition_binds_the_missing_y(route):
    composed = (fm.Chart(_ABC_LEFT).mark_histogram().encode(x="a") + _abc_right(route)).encode(
        y="b"
    )
    assert composed.to_svg().count("<circle") == _abc_right_rows(route)


def test_a_written_x2_reaches_a_bare_right_rule():
    """A left rule's written x/x2/y reach the right rule, as LayerChart draws both encoded."""
    left = fm.Chart(_ABC_LEFT).mark_rule().encode(x="a", x2="b", y="c")
    right = fm.Chart(_ABC_RIGHT).mark_rule()
    composed = left + right
    assert _right_field(composed, "x2") == "b"
    oracle = fm.LayerChart(left, right.encode(x="a", x2="b", y="c"))
    assert _plot_marks(composed.to_svg()) == _plot_marks(oracle.to_svg())


def test_a_histograms_binned_x2_never_reaches_a_right_layer(route):
    composed = (fm.Chart(_ABC_LEFT).mark_histogram().encode(x="a") + _abc_right(route)).encode(
        y="b"
    )
    layer = json.loads(composed.to_spec().to_json())["layers"][-1]
    assert "x2" not in layer["encoding"], layer


def test_a_repeat_placeholder_bound_on_a_composition_reaches_the_right_layer():
    template = (
        _bare_points(_ABC_LEFT) + _bare_points(_ABC_RIGHT, fm.transform_filter("datum.a > 0"))
    ).encode(x=fm.Repeat.column, y="b")
    svg = fm.RepeatChart(template, column=["a", "c"]).to_svg()
    assert svg.count("<circle") == 2 * (_ABC_LEFT.height + _ABC_RIGHT.height)


_D1 = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 2.5, 4.0]})
_D2 = pl.DataFrame({"a": [10.0, 20.0], "b": [10.0, 20.0]})


def _bare_composition():
    """Two filtered frames composed with no x/y of their own."""
    keep = fm.transform_filter("datum.a > 0")
    return _bare_points(_D1, keep) + _bare_points(_D2, keep)


@pytest.mark.parametrize("mark", ["point", "smooth"])
@pytest.mark.parametrize(
    "shares_chain",
    [
        pytest.param(False, id="routed-named-steps"),
        pytest.param(True, id="left-runs-the-right-chain"),
    ],
)
def test_a_composed_same_data_right_operand_takes_the_left_written_x_y(mark, shares_chain):
    """The composition's layers draw the left's written a/b, as if bound on the composition."""
    inner = _bare_composition()
    base = fm.Chart(inner._data)
    if shares_chain:
        base = base.transform(*inner._transforms)
    left = getattr(base, f"mark_{mark}")().encode(x="a", y="b")
    svg = (left + inner).to_svg()
    oracle = (left + inner.encode(x="a", y="b")).to_svg()
    assert svg.count("<circle") >= _D1.height + _D2.height
    assert _plot_marks(svg) == _plot_marks(oracle)


# ---------------------------------------------------------------------------
# A flipped, colored layered chart renders its legend.
# ---------------------------------------------------------------------------


def test_a_flipped_colored_bar_with_inheriting_labels_renders():
    """The value axis spans v, in static and interactive output."""
    frame = pl.DataFrame({"g": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    chart = (
        fm.Chart(frame).mark_bar().encode(x="g", y="v", color="g")
        + fm.Chart(frame).mark_text().encode(text="v")
    ).coord(fm.CoordFlip())
    svg = chart.to_svg()
    value_ticks = [float(t) for t in axis_tick_labels(svg, axis="x")]
    assert value_ticks[0] <= 1.0 and value_ticks[-1] >= 3.0, value_ticks
    assert len(re.findall(r"<rect [^>]*fill=\"#", _plot_marks(svg))) == frame.height
    assert "<html" in chart.interactive().to_html().lower()


# ---------------------------------------------------------------------------
# A step added to the composition reaches the layers reading the chart's rows.
# ---------------------------------------------------------------------------

_V = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
_W = pl.DataFrame({"w": [10.0, 20.0, 30.0]})


def _filtered_v_points():
    return (
        fm.Chart(_V).transform(fm.transform_filter("datum.v > 1")).mark_point().encode(x="v", y="v")
    )


def _own_chain_w_points(data=_W):
    return (
        fm.Chart(data)
        .transform(fm.transform_filter("datum.w > 0"))
        .mark_point()
        .encode(x="w", y="w")
    )


def test_a_step_after_a_cross_frame_merge_reaches_the_left_operands_rows():
    """The post-merge filter runs over the left rows; the right chain is its own."""
    composed = (_filtered_v_points() + _own_chain_w_points()).transform(
        fm.transform_filter("datum.v < 4")
    )
    # v in {2, 3} on the left, and every right point.
    assert composed.to_svg().count("<circle") == 2 + _W.height


def test_steps_after_nested_merges_reach_the_innermost_left_rows_in_order():
    third = pl.DataFrame({"w": [40.0, 50.0]})
    inner = (_filtered_v_points() + _own_chain_w_points()).transform(
        fm.transform_filter("datum.v < 5")
    )
    composed = (inner + _own_chain_w_points(third)).transform(fm.transform_filter("datum.v > 2"))
    # v > 1, then v < 5, then v > 2: v in {3, 4}; both right operands keep every point.
    assert composed.to_svg().count("<circle") == 2 + _W.height + third.height


def test_a_composition_with_a_step_used_as_a_right_operand_draws_as_it_does_alone():
    inner = (_filtered_v_points() + _own_chain_w_points()).transform(
        fm.transform_filter("datum.v < 4")
    )
    left = fm.Chart(_UV).mark_line().encode(x="u", y="v")
    svg = (left + inner).to_svg()
    assert svg.count("<circle") == inner.to_svg().count("<circle") == 2 + _W.height
    assert _plot_marks(svg) == _plot_marks(fm.LayerChart(left, inner).to_svg())


def test_a_left_per_layer_count_sees_a_step_added_after_the_merge():
    """The count runs over the left rows the post-merge filter keeps: v in {2, 3}."""
    rule = fm.Chart(_V).transform(fm.transform_filter("datum.v > 1")).mark_rule()
    composed = (rule.encode(y="count()") + _own_chain_w_points()).transform(
        fm.transform_filter("datum.v < 4")
    )
    expected = (
        fm.Chart(_V)
        .transform(fm.transform_filter("datum.v > 1"), fm.transform_filter("datum.v < 4"))
        .mark_rule()
        .encode(y="count()")
        + _own_chain_w_points()
    )
    assert _plot_marks(composed.to_svg()) == _plot_marks(expected.to_svg())


# ---------------------------------------------------------------------------
# A per-layer aggregate over different frames counts only its own rows.
# ---------------------------------------------------------------------------

_COUNTED = pl.DataFrame({"g": ["a", "b", "a", "b"], "value": [1.0, 2.0, 3.0, 4.0]})


def _many_points(n, column="h"):
    return (
        fm.Chart(pl.DataFrame({column: ["c"] * n, "w": [0.5] * n}))
        .mark_point()
        .encode(x="w", y="w")
    )


@pytest.mark.parametrize("column", ["h", "g"], ids=["disjoint-columns", "overlapping-columns"])
@pytest.mark.parametrize("order", ["count-first", "count-second"])
def test_a_count_beside_another_frame_counts_only_its_own_rows(column, order):
    """The count is 4 whether the other frame has 3 rows or 50."""
    rule = fm.Chart(_COUNTED).mark_rule().encode(y="count()")

    def compose(n):
        other = _many_points(n, column)
        return rule + other if order == "count-first" else other + rule

    assert _tick_labels(compose(50).to_svg()) == _tick_labels(compose(3).to_svg())
    assert max(float(t) for t in _tick_labels(compose(50).to_svg()) if t[0].isdigit()) < 10


def test_a_binned_count_bar_beside_another_frame_counts_only_its_own_rows():
    """The bars are the same whether the other frame holds its rows once or twice."""
    other = DF_SHIFTED.rename({"value": "w"})

    def compose(frame):
        return _binned_bar() + fm.Chart(frame).mark_point().encode(x="w", y="y")

    assert _rect_heights(compose(other).to_svg()) == _rect_heights(
        compose(pl.concat([other, other])).to_svg()
    )


# ---------------------------------------------------------------------------
# LayerChart's shared-scale injection keeps every other field of a layer.
# ---------------------------------------------------------------------------


def test_a_shared_scale_keeps_an_encode_after_reaching_the_left_rows():
    """Every kept left point lies inside the shared x domain, so all of them draw."""
    left = pl.DataFrame({"value": [-2.0, -1.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    composed = (fm.Chart(left).transform(_positive()).mark_point() + _uv_line()).encode(
        x="value", y="y"
    )
    other = fm.Chart(_UV).mark_point().encode(x="u", y="v")
    chart = fm.LayerChart(composed, other, resolve={"x": "shared"})
    assert chart._build_merged().to_svg().count("<circle") == 3 + _UV.height
    assert "<html" in chart.interactive().to_html().lower()


def test_injecting_a_shared_scale_changes_only_the_channel_it_sets():
    from dataclasses import fields

    from ferrum.composition import inject_scale

    member = P() + fm.SecondaryY("value")
    injected = inject_scale(member, "x", {"domain": [0.0, 1.0]})
    for before, after in zip(member._layers, injected._layers, strict=True):
        for f in fields(before):
            if f.name != "encoding":
                assert getattr(after, f.name) == getattr(before, f.name), f.name
    assert any(layer.independent_y for layer in injected._layers)


# ---------------------------------------------------------------------------
# A routed step is named per routing, so operands may share transform objects.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chain",
    [
        pytest.param(lambda: [_positive()], id="one-step"),
        pytest.param(
            lambda: [_positive(), fm.transform_filter("datum.value < 4")], id="two-step-branch"
        ),
    ],
)
@pytest.mark.parametrize("frames", ["three-frames", "one-frame"])
def test_operands_sharing_a_transform_object_route_separately(chain, frames):
    steps = chain()
    second = DF_SHIFTED if frames == "three-frames" else DF
    third = DF.with_columns(pl.col("value") - 3.0) if frames == "three-frames" else DF
    a = P()
    b = fm.Chart(second).transform(*steps).mark_tick().encode(x="value")
    c = fm.Chart(third).transform(*steps).mark_tick().encode(x="value")
    assert _plot_marks((a + b + c).to_svg()) == _plot_marks(fm.LayerChart(a, b, c).to_svg())


# ---------------------------------------------------------------------------
# Routes that lower through `+`: SecondaryY, facets, LayerChart's merged path.
# ---------------------------------------------------------------------------


def _filtered_point():
    return fm.Chart(DF).transform(_positive()).mark_point().encode(x="value", y="y")


def test_secondary_y_reads_the_base_charts_transformed_rows():
    """A SecondaryY series is built on the base chart, so it reads the chain's tail."""
    census = _census((_filtered_point() + fm.SecondaryY("y", mark="point")).to_svg())
    assert census["circles"] == 2 * _N_POSITIVE


def test_a_faceted_composition_draws_each_chain_per_panel():
    composed = _census((H() + D()).facet("g").to_svg())
    assert composed["rects"] == _census(H().facet("g").to_svg())["rects"]
    assert composed["paths"] == _census(D().facet("g").to_svg())["paths"] == 2


def test_layerchart_independent_y_renders_both_chains():
    census = _census(fm.LayerChart(H(), D(), resolve={"y": "independent"}).to_svg())
    assert census == {"rects": 9, "paths": 1, "lines": 0, "circles": 0, "polylines": 0}


def test_layerchart_interactive_renders_both_chains():
    """The interactive route renders LayerChart's merged chart: its density layer reads the routed KDE."""
    merged = fm.LayerChart(H(), D())._build_merged()
    wire = json.loads(merged.to_spec().to_json())
    assert len(wire["layers"]) == 2
    source = wire["layers"][1]["data_source"]
    published = {t["name"]: t for t in wire["transforms"] if "name" in t}
    assert published[source]["type"] == "kde"
    assert _plot_marks(merged.to_svg()) == _plot_marks(fm.LayerChart(H(), D()).to_svg())
    assert "<html" in fm.LayerChart(H(), D()).interactive().to_html().lower()


# ---------------------------------------------------------------------------
# Chart.layer: a layer built on the chart reads the chart's transformed rows.
# ---------------------------------------------------------------------------


def test_a_public_layer_reads_the_charts_transformed_rows():
    chart = _filtered_point().layer(fm.Layer(mark="tick", encoding={"x": "value"}))
    census = _census(chart.to_svg())
    assert census["circles"] == _N_POSITIVE
    assert census["lines"] == _N_POSITIVE


def test_a_public_layer_on_a_histogram_draws_a_rug_at_the_sample_values():
    """A histogram's bins are its mark's statistics; an attached layer reads the chart's rows.

    The rug draws as the same tick composed with ``+``, which reads its own
    source rows.
    """
    chart = H().layer(fm.Layer(mark="tick", encoding={"x": "value"}))
    oracle = H() + fm.Chart(DF).mark_tick().encode(x="value")
    assert _census(chart.to_svg())["lines"] == _N
    assert _plot_marks(chart.to_svg()) == _plot_marks(oracle.to_svg())


def test_a_secondary_y_naming_a_source_column_of_a_histogram_draws():
    """``SecondaryY`` reads the histogram's rows, not its bins, so a sample column draws."""
    chart = H() + fm.SecondaryY("value", mark="line")
    oracle = fm.LayerChart(
        H(), fm.Chart(DF).mark_line().encode(x="value", y="value"), resolve={"y": "independent"}
    )
    assert _census(chart.to_svg()) == _census(oracle.to_svg())
    assert _paths(chart.to_svg()) == _paths(oracle.to_svg())


def test_an_unknown_column_on_a_chart_without_transforms_names_no_remedy():
    """With no chain, the layer reads the source, so `+` would not help."""
    chart = P().layer(fm.Layer(mark="tick", encoding={"x": "missing"}))
    with pytest.raises(ValueError) as exc:
        chart.to_svg()
    assert str(exc.value) == (
        "layer 1: encoding 'x' references unknown column 'missing' "
        "(not present in that layer's data)"
    )


@pytest.mark.parametrize(
    "build, message",
    [
        pytest.param(
            lambda: (
                fm.Chart(DF)
                .transform(fm.transform_calculate("w", "datum.value * 2"))
                .mark_point()
                .encode(x="value", y="y")
                .layer(fm.Layer(mark="tick", encoding={"x": "nope"}))
            ),
            "layer 1: encoding 'x' references unknown column 'nope' "
            "(not present in that layer's data)",
            id="typo-on-a-transformed-chart",
        ),
        pytest.param(
            lambda: _box().layer(fm.Layer(mark="tick", encoding={"x": "missing"})),
            "layer 6: encoding 'x' references unknown column 'missing' "
            "(not present in that layer's data)",
            id="named-only-chain",
        ),
        pytest.param(
            lambda: fm.LayerChart(
                P(),
                fm.Chart(DF).transform(_positive()).mark_line().encode(x="nope", y="y"),
                resolve={"y": "independent"},
            ),
            "layer 1: encoding 'x' references unknown column 'nope' "
            "(not present in that layer's data)",
            id="own-x-miss-on-an-independent-y-layer",
        ),
    ],
)
def test_the_plus_remedy_is_named_only_for_a_source_column_the_chain_drops(build, message):
    """A column that is not in the source, or a chain whose tail is the source, gets no remedy."""
    with pytest.raises(ValueError) as exc:
        build().to_svg()
    assert str(exc.value) == message


def test_a_faceted_histogram_rug_draws_each_panels_sample_values():
    """Faceted, the attached rug reads each panel's partition of the chart's rows."""
    chart = H().layer(fm.Layer(mark="tick", encoding={"x": "value"})).facet("g")
    assert _census(chart.to_svg())["lines"] == _N


def test_a_layer_naming_a_source_column_on_a_named_only_chain_draws():
    """A boxplot's chain is named steps only, so its tail is the source rows."""
    chart = _box().layer(fm.Layer(mark="tick", encoding={"y": "value"}))
    assert _census(chart.to_svg())["lines"] > _census(_box().to_svg())["lines"]


def test_a_path_chart_with_a_layer_frame_is_refused():
    chart = fm.Chart(Path("data.csv"))
    with pytest.raises(TypeError, match="Unsupported data type: PosixPath"):
        chart.layer(fm.Layer(data=DF, mark="tick", encoding={"x": "value"}))


# ---------------------------------------------------------------------------
# Routing shape on the wire.
# ---------------------------------------------------------------------------


def _wire_transforms(chart) -> list[dict]:
    return json.loads(chart.to_spec().to_json())["transforms"]


def test_a_two_step_right_hand_chain_is_one_branch_over_the_source():
    filtered = fm.Chart(DF).transform(_positive()).mark_density().encode(x="value")
    transforms = _wire_transforms(H() + filtered)
    branch = transforms[0]
    assert branch["type"] == "branch"
    assert branch["input"] is None
    assert [t["type"] for t in branch["transforms"]] == ["filter", "kde"]
    assert transforms[1]["type"] == "bin" and "name" not in transforms[1]


def test_a_one_step_right_hand_chain_is_a_named_step_ahead_of_the_left_chain():
    transforms = _wire_transforms(H() + D())
    assert [t["type"] for t in transforms] == ["kde", "bin"]
    assert "name" in transforms[0]
    assert "name" not in transforms[1]


# ---------------------------------------------------------------------------
# Overlapping frames: the right-hand chain sees its own columns' types.
# ---------------------------------------------------------------------------

_TYPED_N = 12
# The left frame shares every column name with the right one, with other types,
# so the right operand's columns are renamed in the merge and restored by name
# ahead of its chain.
LEFT_TYPED = pl.DataFrame(
    {
        "d": [float(i) for i in range(_TYPED_N)],
        "k": pl.Series([i % 3 + 7 for i in range(_TYPED_N)], dtype=pl.Int64),
        "c": pl.Series(["c", "b", "a"] * 4, dtype=pl.Categorical),
        "v": [float(i % 5) for i in range(_TYPED_N)],
    }
)
RIGHT_TYPED = pl.DataFrame(
    {
        "d": [date(2020, 1, 1) + timedelta(days=17 * i) for i in range(_TYPED_N)],
        "k": pl.Series([(i + 1) % 3 + 7 for i in range(_TYPED_N)], dtype=pl.Int64),
        "c": pl.Series(["b", "a", "c"] * 4, dtype=pl.Categorical),
        "v": [float(i) for i in range(_TYPED_N)],
    }
)
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _right_typed():
    return fm.Chart(RIGHT_TYPED).transform(fm.transform_filter("datum.v > 2"))


def _labels(svg: str) -> list[str]:
    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


def test_a_date_column_reaches_the_right_hand_chain_as_a_date():
    """The right operand's own y axis draws the dates its filtered rows span, as standalone.

    The left frame's ``d`` is a float, so the date axis can only come from the
    right operand's restored column keeping its date type.
    """
    right = _right_typed().mark_line().encode(x="v", y="d")
    left = fm.Chart(LEFT_TYPED).mark_point().encode(x="v", y="d")
    composed = _labels(fm.LayerChart(left, right, resolve={"y": "independent"}).to_svg())
    standalone = _labels(right.to_svg())
    dates = [t for t in standalone if _ISO_DATE.fullmatch(t)]
    assert dates
    assert [t for t in composed if _ISO_DATE.fullmatch(t)] == dates


@pytest.mark.parametrize(
    "operands",
    [
        pytest.param(
            lambda: (
                fm.Chart(LEFT_TYPED).mark_point().encode(x="k:N", y="v"),
                _right_typed().mark_point().encode(x="k:N", y="v"),
            ),
            id="int64-nominal-axis",
        ),
        pytest.param(
            lambda: (
                fm.Chart(LEFT_TYPED).mark_point().encode(x="c", y="v"),
                _right_typed().mark_point().encode(x="c", y="v"),
            ),
            id="categorical-axis",
        ),
        pytest.param(
            lambda: (
                fm.Chart(LEFT_TYPED).mark_point().encode(x="v", y="v"),
                _right_typed().mark_point().encode(x="v", y="v", color="k:N"),
            ),
            id="int64-nominal-color-stays-on-the-right-operand",
        ),
    ],
)
def test_overlapping_typed_columns_draw_the_layerchart_marks(operands):
    a, b = operands()
    composed = (a + b).to_svg()
    oracle = fm.LayerChart(a, b).to_svg()
    assert _plot_marks(composed) == _plot_marks(oracle)
    assert _labels(composed) == _labels(oracle)
