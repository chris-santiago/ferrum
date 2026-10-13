"""`a + b` draws each operand's own transform chain, as `fm.LayerChart(a, b)` does.

A composed chart runs one Rust pipeline: unnamed steps chain, named steps
publish an output a layer reads through ``data_source``. ``+`` routes each
operand's chain so it reads the operand's own source rows: a right-hand chain
becomes a named step (one step) or one ``Branch`` (several steps) ahead of the
left-hand chain, and a right-hand layer with no chain of its own reads a named
identity of the source whenever the left-hand chain has unnamed steps. A
per-layer aggregate or bin reads the layer's own named input. A layer built
on the chart itself (``Chart.layer``, ``SecondaryY``) reads that chart's
transformed rows instead.

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


@pytest.mark.parametrize(
    "right",
    [
        pytest.param(lambda: _box(DF_SHIFTED), id="other-frame"),
        pytest.param(
            lambda: _box(DF_SHIFTED.rename({"y": "w"})), id="overlapping-frame-with-chain"
        ),
    ],
)
def test_two_boxplots_over_different_frames_each_draw_their_own_boxes(right):
    """Each boxplot reads its own statistics: the same box, whisker and outlier marks as LayerChart."""
    a, b = _box(), right()
    assert _census((a + b).to_svg()) == _census(fm.LayerChart(a, b).to_svg())


def test_two_contours_render_on_the_interactive_route():
    """LayerChart's interactive output renders the merged chart ``+`` builds."""
    a, b = _contour(), _contour(DF_SHIFTED)
    merged = fm.LayerChart(a, b)._build_merged()
    assert _plot_marks(merged.to_svg()) == _plot_marks(fm.LayerChart(a, b).to_svg())
    assert "<html" in fm.LayerChart(a, b).interactive().to_html().lower()


def test_two_boxplots_render_on_the_interactive_route():
    a, b = _box(), _box(DF_SHIFTED)
    merged = fm.LayerChart(a, b)._build_merged()
    assert _census(merged.to_svg()) == _census(fm.LayerChart(a, b).to_svg())
    assert "<html" in fm.LayerChart(a, b).interactive().to_html().lower()


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


_TAIL_REMEDY = (
    "this layer reads the chart's transformed rows, not its source data; "
    "draw 'value' from the source by composing a separate chart with `+` instead"
)


def test_a_public_layer_naming_a_column_the_chain_drops_is_refused():
    """A histogram's chain outputs bins, so a layer naming the sample column cannot draw."""
    chart = H().layer(fm.Layer(mark="tick", encoding={"x": "value"}))
    with pytest.raises(ValueError) as exc:
        chart.to_svg()
    assert str(exc.value) == (
        "layer 1: encoding 'x' references unknown column 'value' "
        f"(not present in that layer's data) — {_TAIL_REMEDY}"
    )


def test_a_secondary_y_naming_a_column_the_chain_drops_is_refused():
    with pytest.raises(ValueError) as exc:
        (H() + fm.SecondaryY("value", mark="line")).to_svg()
    assert str(exc.value) == (
        "layer 1: encoding 'y' references unknown column 'value' "
        f"(not present in that layer's data) — {_TAIL_REMEDY}"
    )


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
        pytest.param(
            lambda: H().layer(fm.Layer(mark="tick", encoding={"x": "value"})).facet("g"),
            "layer 1: encoding 'x' references unknown column 'value' "
            f"(not present in that layer's data) — {_TAIL_REMEDY}",
            id="faceted-chain-tail",
        ),
    ],
)
def test_the_plus_remedy_is_named_only_for_a_source_column_the_chain_drops(build, message):
    """A column that is not in the source, or a chain whose tail is the source, gets no remedy."""
    with pytest.raises(ValueError) as exc:
        build().to_svg()
    assert str(exc.value) == message


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
