"""The two-dimensional statistics name a group, panel or call they cannot estimate or place.

A contour (``Kde2D`` then ``Contour``), a 2-D density figure, a hexbin or a raster whose points
cannot be estimated or placed keeps its legend entry or facet panel and is named once, by its
statistic's ``DegenerateFit``. ``Kde2D`` names a group or panel with fewer than two usable points
(a row with a null, NaN or infinite coordinate is no point), identical points, points tied on one
axis (under any bandwidth), or a density that is zero at every grid point; ``Hex`` and ``Raster``
name a panel or call with rows and no non-null, finite point (or, for an aggregate that reads one, no
usable ``field=`` value). The render's census defers to the statistic that named the unit, so every
chart here raises exactly the listed messages. A frame with no rows reports that the chart has no
rows to draw, and a well-formed 2-D chart warns nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests.test_null_group_warning import (
    FIT_FEW_PAIRS,
    FIT_TIED_POINTS,
    FIT_TIED_X,
    FIT_TIED_Y,
    _NO_FACET_ROWS_WARNING,
    _NO_ROWS_TO_DRAW,
    _facet_key_dropped_message,
    _fit_message,
    _panel_fit_message,
    _render_capturing,
    _whole_fit_message,
)

# Kde2D counts points, not values: its count reason is the shared one (Kde keeps "values").
_FEW_POINTS = FIT_FEW_PAIRS
_ZERO_GRID = "its density is zero across the evaluation grid"
# The reason Hex and Raster give for a panel or call with rows and no usable value.
_NO_FINITE = "no usable (non-null, finite) values"
_INF = float("inf")
_NAN = float("nan")

# Level ``c`` is the one the statistic cannot estimate; level ``a`` holds ten distinct points.
_C_CASES = [
    pytest.param([(None, None)] * 4, _FEW_POINTS, id="all-null"),
    pytest.param([(_NAN, _NAN)] * 4, _FEW_POINTS, id="nan"),
    pytest.param([(1.0, 1.0)], _FEW_POINTS, id="one"),
    pytest.param([(1.0, 2.0)] * 4, FIT_TIED_POINTS, id="tied"),
    pytest.param([(0.0, 2.0), (1.0, 2.0), (2.0, 2.0), (3.0, 2.0)], FIT_TIED_Y, id="tied-y"),
    pytest.param([(2.0, 0.0), (2.0, 1.0), (2.0, 2.0), (2.0, 3.0)], FIT_TIED_X, id="tied-x"),
    pytest.param([(_INF, _INF), (-_INF, 1.0), (2.0, _INF), (_INF, 3.0)], _FEW_POINTS, id="all-inf"),
    pytest.param(
        [(0.0, 0.0), (_INF, 1.0), (_INF, 2.0), (-_INF, 3.0)], _FEW_POINTS, id="one-finite"
    ),
]
_CONTOUR_FILLS = [pytest.param(False, id="isoline"), pytest.param(True, id="isoband")]

# The cause of a facet panel ``q`` (or of a whole call) having rows and no usable value.
# ``x``, ``y`` and ``z`` are the rows' columns; ``z`` is read only by a field-reading aggregate.
_CAUSE_ROWS: dict[str, dict[str, list[float | None]]] = {
    "x-and-y-null": {"x": [None] * 10, "y": [None] * 10, "z": [1.0] * 10},
    "x-null": {"x": [None] * 10, "y": [float(i) for i in range(10)], "z": [1.0] * 10},
    "y-null": {"x": [float(i) for i in range(10)], "y": [None] * 10, "z": [1.0] * 10},
    "nan": {"x": [_NAN] * 10, "y": [_NAN] * 10, "z": [1.0] * 10},
    "inf": {"x": [_INF, _INF, -_INF], "y": [1.0, _INF, 2.0], "z": [1.0, 1.0, 1.0]},
    "field-null": {
        "x": [float(i) for i in range(10)],
        "y": [float(i) for i in range(10)],
        "z": [None] * 10,
    },
}
_FIELD_KWARGS: dict[str, dict[str, Any]] = {
    "field-null-mean": {"aggregate": "mean", "field": "z"},
    "field-null-sum": {"aggregate": "sum", "field": "z"},
}


def _cause(case: str) -> tuple[str, dict[str, Any]]:
    """The rows' cause and the mark kwargs a test case id stands for."""
    if case in _FIELD_KWARGS:
        return "field-null", _FIELD_KWARGS[case]
    return case, {}


def _points(n: int, *, seed: int = 0) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    return list(rng.normal(0, 1, n)), list(rng.normal(0, 1, n))


def _hue_frame(c_points: list[tuple[float | None, float | None]]) -> pl.DataFrame:
    """Hue level ``a`` of ten distinct points beside level ``c`` of ``c_points``."""
    xs, ys = _points(10)
    return pl.DataFrame(
        {
            "x": xs + [p[0] for p in c_points],
            "y": ys + [p[1] for p in c_points],
            "g": ["a"] * 10 + ["c"] * len(c_points),
        },
        schema={"x": pl.Float64, "y": pl.Float64, "g": pl.Utf8},
    )


def _cause_frame(case: str, *, faceted: bool) -> pl.DataFrame:
    """Rows of ``case``'s cause alone, or beside panel ``p`` of ten finite rows, interleaved into
    the facet column ``f`` when ``faceted``."""
    cause, _ = _cause(case)
    q = _CAUSE_ROWS[cause]
    schema = {"x": pl.Float64, "y": pl.Float64, "z": pl.Float64}
    if not faceted:
        return pl.DataFrame(q, schema=schema)
    xs, ys = _points(10, seed=3)
    return pl.DataFrame(
        {
            "f": ["p"] * 10 + ["q"] * len(q["x"]),
            "x": xs + q["x"],
            "y": ys + q["y"],
            "z": [1.0] * 10 + q["z"],
        },
        schema={"f": pl.Utf8, **schema},
    )


def _panel_frame(*, nan: bool = False) -> pl.DataFrame:
    """Facet panels ``p`` (distinct points) and ``q`` (every x and y null, or NaN), interleaved."""
    n = 90
    xs, ys = _points(n, seed=1)
    panels = ["p", "q"] * (n // 2)
    blank = _NAN if nan else None
    return pl.DataFrame(
        {
            "f": panels,
            "x": [blank if f == "q" else v for f, v in zip(panels, xs)],
            "y": [blank if f == "q" else v for f, v in zip(panels, ys)],
        },
        schema={"f": pl.Utf8, "x": pl.Float64, "y": pl.Float64},
    )


def _null_frame(rows: int = 5, *, nan: bool = False) -> pl.DataFrame:
    blank = _NAN if nan else None
    return pl.DataFrame(
        {"x": [blank] * rows, "y": [blank] * rows}, schema={"x": pl.Float64, "y": pl.Float64}
    )


def _zero_row_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"x": [], "y": [], "f": []}, schema={"x": pl.Float64, "y": pl.Float64, "f": pl.Utf8}
    )


def _well_formed_frame() -> pl.DataFrame:
    xs, ys = _points(40, seed=2)
    return pl.DataFrame({"x": xs, "y": ys, "f": ["p", "q"] * 20})


def _chart(mark_method: str, df: pl.DataFrame, **mark_kwargs: Any) -> fm.Chart:
    return getattr(fm.Chart(df), mark_method)(**mark_kwargs).encode(x="x", y="y")


def _density_door(door: str, df: pl.DataFrame) -> fm.Chart:
    """The 2-D density of ``x`` by ``y``, one group per ``g``, through ``door``."""
    if door == "mark_contour":
        return fm.Chart(df).mark_contour().encode(x="x", y="y", color="g")
    if door == "mark_density":
        return fm.Chart(df).mark_density().encode(x="x", y="y", color="g")
    return fm.displot(df, x="x", y="y", hue="g", kind="kde")


_DENSITY_DOORS = ["mark_contour", "mark_density", "displot"]


# ---------------------------------------------------------------------------
# Contour and the 2-D density doors: a hue group Kde2D cannot estimate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("door", _DENSITY_DOORS)
@pytest.mark.parametrize("points,reason", _C_CASES)
def test_a_contour_group_it_cannot_estimate_is_named(
    points: list[tuple[float | None, float | None]], reason: str, door: str
):
    """The group keeps its legend entry and ``Kde2D`` names it once, in ``Kde``'s wording for the
    same cause. A grouped contour draws isolines, so the isoline path is the one pinned."""
    svg, shown = _render_capturing(_density_door(door, _hue_frame(points)).to_svg)
    assert shown == [_fit_message("Kde2D", "g=c", reason)]
    assert ">c<" in svg


_FIXED_BANDWIDTH_TIED_CASES = [
    pytest.param([(1.0, 2.0)] * 4, FIT_TIED_POINTS, id="tied"),
    pytest.param([(0.0, 2.0), (1.0, 2.0), (2.0, 2.0), (3.0, 2.0)], FIT_TIED_Y, id="tied-y"),
    pytest.param([(2.0, 0.0), (2.0, 1.0), (2.0, 2.0), (2.0, 3.0)], FIT_TIED_X, id="tied-x"),
]


@pytest.mark.parametrize("points,reason", _FIXED_BANDWIDTH_TIED_CASES)
def test_a_contour_group_it_cannot_estimate_under_a_fixed_bandwidth_names_its_cause(
    points: list[tuple[float | None, float | None]], reason: str
):
    """A caller's bandwidth is always usable, so the data is the cause: the group is named with
    the tied reason, instead of the census blaming null x and y (identical points) or nothing
    being said for a body that draws nothing (a tie on one axis)."""
    chart = fm.Chart(_hue_frame(points)).mark_contour(bandwidth=0.5).encode(x="x", y="y", color="g")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_fit_message("Kde2D", "g=c", reason)]
    assert ">c<" in svg


_INFINITE_COORDINATE_ROWS = [
    pytest.param([(0.0, 0.0), (1.0, 1.0), (2.0, 2.5), (_INF, 3.0)], id="inf-x"),
    pytest.param([(0.0, 0.0), (1.0, 1.0), (2.0, 2.5), (3.0, -_INF)], id="inf-y"),
    pytest.param([(0.0, 0.0), (1.0, 1.0), (2.0, 2.5), (_INF, -_INF)], id="inf-both"),
]


def _nullified(points: list[tuple[float, float]]) -> list[tuple[float | None, float | None]]:
    """``points`` with every infinite coordinate replaced by a null."""
    return [tuple(None if abs(v) == _INF else v for v in p) for p in points]


def _density_door_at(door: str, bandwidth: float | None, df: pl.DataFrame) -> fm.Chart:
    """``_density_door``, or a contour under the fixed ``bandwidth`` when one is given."""
    if bandwidth is None:
        return _density_door(door, df)
    return fm.Chart(df).mark_contour(bandwidth=bandwidth).encode(x="x", y="y", color="g")


@pytest.mark.parametrize(
    "door,bandwidth",
    [*((door, None) for door in _DENSITY_DOORS), ("mark_contour", 0.5)],
    ids=[*_DENSITY_DOORS, "mark_contour-fixed-bandwidth"],
)
@pytest.mark.parametrize("points", _INFINITE_COORDINATE_ROWS)
def test_a_contour_group_with_an_infinite_coordinate_draws_from_its_finite_points(
    points: list[tuple[float, float]], door: str, bandwidth: float | None
):
    """A row with an infinite coordinate is skipped as one with a null coordinate is: the group
    keeps three usable points, draws as its null twin does and warns nothing, under the default
    bandwidth and a fixed one."""
    svg, shown = _render_capturing(_density_door_at(door, bandwidth, _hue_frame(points)).to_svg)
    twin_svg, twin_shown = _render_capturing(
        _density_door_at(door, bandwidth, _hue_frame(_nullified(points))).to_svg
    )
    assert shown == [] and twin_shown == []
    assert svg == twin_svg
    assert ">c<" in svg


def test_a_contour_group_with_a_null_row_and_enough_points_stays_silent():
    """A null row in a group that still has distinct points to estimate from is no degenerate
    group: nothing is named and the level stays in the legend."""
    points = [(None, None), (None, 1.0), (2.0, 3.0), (-1.0, 0.5), (0.3, -0.7), (1.1, 1.4)]
    svg, shown = _render_capturing(_density_door("mark_contour", _hue_frame(points)).to_svg)
    assert shown == []
    assert ">c<" in svg


# ---------------------------------------------------------------------------
# A facet panel a 2-D statistic cannot estimate or place.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fill", _CONTOUR_FILLS)
@pytest.mark.parametrize("blank", ["null", "nan"])
def test_a_contour_facet_panel_a_null_column_empties_is_named(blank: str, fill: bool):
    """Panel ``q`` holds only nulls (or only NaN): it is drawn empty (its strip title stays) and
    ``Kde2D`` names it once, with the panel clause."""
    chart = _chart("mark_contour", _panel_frame(nan=blank == "nan"), fill=fill).facet(col="f")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_panel_fit_message("Kde2D", _FEW_POINTS)]
    assert ">p<" in svg and ">q<" in svg


@pytest.mark.parametrize("fill", _CONTOUR_FILLS)
def test_a_contour_facet_panel_whose_density_vanishes_on_the_shared_grid_is_named(fill: bool):
    """Panel ``q`` holds six distinct points packed far tighter than the grid shared with panel
    ``p``, so its density is zero at every grid point: ``Kde2D`` names it, and the panel is drawn."""
    xs, ys = _points(40, seed=1)
    tight_x = [0.3 + 1e-7 * k for k in (0, 1, -1, 2, -2, 3)]
    tight_y = [0.2 + 1e-7 * k for k in (0, 2, 1, -1, -2, 3)]
    df = pl.DataFrame({"x": xs + tight_x, "y": ys + tight_y, "f": ["p"] * 40 + ["q"] * 6})
    svg, shown = _render_capturing(_chart("mark_contour", df, fill=fill).facet(col="f").to_svg)
    assert shown == [_panel_fit_message("Kde2D", _ZERO_GRID)]
    assert ">p<" in svg and ">q<" in svg


_HEX_CAUSES = [
    "x-and-y-null",
    "x-null",
    "y-null",
    "nan",
    "inf",
    "field-null-mean",
    "field-null-sum",
]


@pytest.mark.parametrize("case", _HEX_CAUSES)
def test_a_hex_facet_panel_it_cannot_bin_is_named(case: str):
    """Panel ``q`` has rows and no non-null, finite point (or no usable ``field=``): ``Hex`` names
    it once with the panel clause, the census adds nothing, and both panels are drawn."""
    _, kwargs = _cause(case)
    chart = _chart("mark_hex", _cause_frame(case, faceted=True), **kwargs).facet(col="f")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_panel_fit_message("Hex", _NO_FINITE)]
    assert ">p<" in svg and ">q<" in svg


_RASTER_CAUSES = ["x-and-y-null", "nan", "inf", "field-null-mean"]


@pytest.mark.parametrize("case", _RASTER_CAUSES)
def test_a_raster_facet_panel_it_cannot_bin_is_named(case: str):
    """``Raster`` names the panel once with the panel clause, and both panels are drawn: panel
    ``q`` keeps its transparent image, so two images are placed."""
    _, kwargs = _cause(case)
    chart = _chart("mark_raster", _cause_frame(case, faceted=True), **kwargs).facet(col="f")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_panel_fit_message("Raster", _NO_FINITE)]
    assert ">p<" in svg and ">q<" in svg
    assert svg.count("<image") == 2


def test_a_hex_facet_panel_with_one_usable_row_stays_silent():
    """Rows with a null x and y beside a usable one are no emptied panel."""
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, None, 4.0, None],
            "y": [1.0, 2.0, 3.0, None, 5.0, None],
            "f": ["p"] * 3 + ["q"] * 3,
        }
    )
    svg, shown = _render_capturing(_chart("mark_hex", df).facet(col="f").to_svg)
    assert shown == []
    assert ">p<" in svg and ">q<" in svg


# ---------------------------------------------------------------------------
# A whole unfaceted call with nothing to estimate or place.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["x-and-y-null", "x-null", "field-null-mean"])
def test_an_unfaceted_hex_it_cannot_bin_names_its_cause(case: str):
    """The call has rows and no usable point: ``Hex`` names it, instead of the chart having no rows
    to draw."""
    _, kwargs = _cause(case)
    _, shown = _render_capturing(
        _chart("mark_hex", _cause_frame(case, faceted=False), **kwargs).to_svg
    )
    assert shown == [_whole_fit_message("Hex", _NO_FINITE)]


@pytest.mark.parametrize("case", ["x-and-y-null", "nan", "field-null-mean"])
def test_an_unfaceted_raster_it_cannot_bin_is_named(case: str):
    """``Raster`` names the call, and a call with no usable value places no image."""
    _, kwargs = _cause(case)
    svg, shown = _render_capturing(
        _chart("mark_raster", _cause_frame(case, faceted=False), **kwargs).to_svg
    )
    assert shown == [_whole_fit_message("Raster", _NO_FINITE)]
    assert "<image" not in svg


@pytest.mark.parametrize("fill", _CONTOUR_FILLS)
@pytest.mark.parametrize("blank", ["null", "nan"])
def test_an_all_null_contour_names_its_cause_not_the_empty_chart(blank: str, fill: bool):
    """``Kde2D`` names the cause, and nothing says the chart has no rows."""
    _, shown = _render_capturing(
        _chart("mark_contour", _null_frame(nan=blank == "nan"), fill=fill).to_svg
    )
    assert shown == [_whole_fit_message("Kde2D", _FEW_POINTS)]


@pytest.mark.parametrize(
    "x,y,reason",
    [
        pytest.param([1.0], [1.0], _FEW_POINTS, id="one"),
        pytest.param([1.0] * 4, [2.0] * 4, FIT_TIED_POINTS, id="tied"),
        pytest.param([0.0, 1.0, 2.0, 3.0], [4.0] * 4, FIT_TIED_Y, id="tied-y"),
        pytest.param([4.0] * 4, [0.0, 1.0, 2.0, 3.0], FIT_TIED_X, id="tied-x"),
    ],
)
def test_a_contour_of_one_point_or_identical_points_names_its_cause(
    x: list[float], y: list[float], reason: str
):
    """A call whose points cannot be estimated is named, not reported as a chart with no rows;
    a bandwidth that collapses on one axis alone counts."""
    _, shown = _render_capturing(_chart("mark_contour", pl.DataFrame({"x": x, "y": y})).to_svg)
    assert shown == [_whole_fit_message("Kde2D", reason)]


# ---------------------------------------------------------------------------
# Infinite rows, emptied panels and frames with no rows.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mark_method", ["mark_hex", "mark_raster"])
def test_an_infinite_row_is_skipped_as_a_null_one_is(mark_method: str):
    """Three rows with an infinite x or y beside ten finite ones are skipped like null rows: no
    warning, and the same chart as the ten finite rows alone."""
    finite = pl.DataFrame(dict(zip("xy", _points(10, seed=3))))
    infinite = pl.DataFrame({"x": [_INF, _INF, -_INF], "y": [1.0, _INF, 2.0]})
    svg, shown = _render_capturing(_chart(mark_method, pl.concat([finite, infinite])).to_svg)
    assert shown == []
    assert svg == _chart(mark_method, finite).to_svg()


@pytest.mark.parametrize("mark_method", ["mark_hex", "mark_raster", "mark_contour"])
def test_a_frame_whose_facet_key_is_all_null_draws_no_panel(mark_method: str):
    """Rows with a null facet key belong to no panel: the facet drops them and says so, and the
    chart then reports that it has no facet panel to draw."""
    xs, ys = _points(20, seed=5)
    df = pl.DataFrame(
        {"x": xs, "y": ys, "f": [None] * 20},
        schema={"x": pl.Float64, "y": pl.Float64, "f": pl.Utf8},
    )
    _, shown = _render_capturing(_chart(mark_method, df).facet(col="f").to_svg)
    assert shown == [
        _facet_key_dropped_message(20, "f"),
        _NO_FACET_ROWS_WARNING,
    ]


@pytest.mark.parametrize("mark_method", ["mark_hex", "mark_raster", "mark_contour"])
def test_a_facet_panel_a_filter_empties_is_left_out(mark_method: str):
    """A panel with no row left after a filter is not a panel with rows and no usable value: it is
    left out without a word, and the other panel is drawn."""
    xs, ys = _points(40, seed=5)
    df = pl.DataFrame({"x": xs, "y": ys, "f": ["p", "q"] * 20, "k": [1, 0] * 20})
    filtered = fm.Chart(df).transform(fm.transform_filter("datum.k > 0"))
    chart = getattr(filtered, mark_method)().encode(x="x", y="y").facet(col="f")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert ">p<" in svg and ">q<" not in svg
    if mark_method == "mark_raster":
        assert svg.count("<image") == 1


_ZERO_ROW_DOORS = ["mark_hex", "mark_raster", "mark_contour", "displot"]


def _zero_row_chart(door: str, *, faceted: bool) -> fm.Chart:
    df = _zero_row_frame()
    if door == "displot":
        return fm.displot(df, x="x", y="y", kind="kde", **({"row": "f"} if faceted else {}))
    chart = _chart(door, df)
    return chart.facet(row="f") if faceted else chart


@pytest.mark.parametrize(
    "faceted", [pytest.param(False, id="unfaceted"), pytest.param(True, id="faceted")]
)
@pytest.mark.parametrize("door", _ZERO_ROW_DOORS)
def test_a_zero_row_frame_still_warns_no_rows_to_draw(door: str, faceted: bool):
    """A frame with no rows reports that the chart has no rows to draw (a faceted one, that it
    draws no facet panel), on every 2-D door."""
    _, shown = _render_capturing(_zero_row_chart(door, faceted=faceted).to_svg)
    assert shown == [_NO_FACET_ROWS_WARNING if faceted else _NO_ROWS_TO_DRAW]


# ---------------------------------------------------------------------------
# A well-formed chart warns nothing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fill", _CONTOUR_FILLS)
def test_a_well_formed_contour_warns_nothing(fill: bool):
    df = _well_formed_frame()
    assert _render_capturing(_chart("mark_contour", df, fill=fill).to_svg)[1] == []
    faceted = _chart("mark_contour", df, fill=fill).facet(col="f")
    svg, shown = _render_capturing(faceted.to_svg)
    assert shown == []
    assert ">p<" in svg and ">q<" in svg


@pytest.mark.parametrize("mark_method", ["mark_hex", "mark_raster"])
def test_a_well_formed_hex_or_raster_warns_nothing(mark_method: str):
    df = _well_formed_frame()
    assert _render_capturing(_chart(mark_method, df).to_svg)[1] == []
    svg, shown = _render_capturing(_chart(mark_method, df).facet(col="f").to_svg)
    assert shown == []
    assert ">p<" in svg and ">q<" in svg


def test_a_well_formed_2d_density_figure_warns_nothing():
    chart = fm.displot(_well_formed_frame(), x="x", y="y", hue="f", kind="kde")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert ">p<" in svg and ">q<" in svg
