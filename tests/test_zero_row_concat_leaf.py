"""A zero-row leaf of a concat beside a populated leaf draws its frame and warns once.

A leaf whose frame has no rows lowers like any other leaf: its panel's frame and axes draw and the
chart reports that it has no rows to draw, once per empty leaf, on the static and interactive
routes, as a single zero-row chart and a filter-emptied leaf already do. A composition every one
of whose leaves is zero-row keeps its refusal, and a filter-emptied leaf and a ``LayerChart``
keep the bytes they had.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import panel_extents
from tests.test_log_remedies_render import _interactive as _scene_json
from tests.test_null_group_warning import _NO_ROWS_TO_DRAW, _render_capturing
from tests.test_zero_row_statistics import _MARKS
from tests.test_zero_row_statistics import _empty as _empty_stat_frame
from tests.test_zero_row_statistics import _populated as _populated_stat_frame

_SCHEMA = {"x": pl.Float64, "y": pl.Float64}
_EMPTY_SCHEMA = {"u": pl.Float64, "w": pl.Float64}


def _populated() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}, schema=_SCHEMA)


def _empty() -> pl.DataFrame:
    return pl.DataFrame({"u": [], "w": []}, schema=_EMPTY_SCHEMA)


def _leaf(frame: pl.DataFrame, x: str = "x", y: str = "y") -> fm.Chart:
    return fm.Chart(frame).mark_point().encode(x=x, y=y)


def _populated_leaf() -> fm.Chart:
    return _leaf(_populated())


def _zero_row_leaf() -> fm.Chart:
    """Its own axis titles (``u``, ``w``) let a page show whether the panel's axes drew."""
    return _leaf(_empty(), "u", "w")


def _filter_emptied_leaf() -> fm.Chart:
    chart = fm.Chart(_populated()).transform(fm.transform_filter("datum.x > 100"))
    return chart.mark_point().encode(x="x", y="y")


def _static(chart: Any) -> Callable[[], str]:
    return chart.to_svg


def _interactive(chart: Any) -> Callable[[], Any]:
    return lambda: _scene_json(chart)


_OUTPUTS = {"static": _static, "interactive": _interactive}

_DOORS: dict[str, Callable[[fm.Chart, fm.Chart], Any]] = {
    "hconcat": lambda a, b: a | b,
    "vconcat": lambda a, b: a & b,
    "concat": lambda a, b: fm.concat(a, b, columns=2),
    "hconcat_fn": lambda a, b: fm.hconcat(a, b),
    "vconcat_fn": lambda a, b: fm.vconcat(a, b),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("door", list(_DOORS))
def test_a_zero_row_leaf_beside_a_populated_leaf_draws_its_frame_and_warns_once(door: str):
    grid = _DOORS[door](_populated_leaf(), _zero_row_leaf())
    svg, messages = _render_capturing(grid.to_svg)
    assert messages == [_NO_ROWS_TO_DRAW]
    # the empty panel's own axis titles are on the page, as the populated panel's are
    assert ">u<" in svg and ">w<" in svg
    assert ">x<" in svg and ">y<" in svg


@pytest.mark.parametrize("door", list(_DOORS))
def test_the_same_composition_interactive_warns_once_and_draws_the_empty_frame(door: str):
    grid = _DOORS[door](_populated_leaf(), _zero_row_leaf())
    scene, messages = _render_capturing(_interactive(grid))
    assert messages == [_NO_ROWS_TO_DRAW]
    # the empty panel is a panel of the scene beside the populated one
    assert len(json.loads(scene)["panels"]) == 2
    assert '"u"' in scene


def test_a_zero_row_leaf_first_in_a_pair_draws_its_frame_and_warns_once():
    svg, messages = _render_capturing((_zero_row_leaf() | _populated_leaf()).to_svg)
    assert messages == [_NO_ROWS_TO_DRAW]
    assert ">u<" in svg and ">x<" in svg


@pytest.mark.parametrize("output", list(_OUTPUTS))
def test_two_zero_row_leaves_beside_a_populated_leaf_warn_once_each(output: str):
    grid = _populated_leaf() | _zero_row_leaf() | _zero_row_leaf()
    page, messages = _render_capturing(_OUTPUTS[output](grid))
    assert messages == [_NO_ROWS_TO_DRAW, _NO_ROWS_TO_DRAW]
    if output == "static":
        assert page.count(">u<") == 2 and page.count(">w<") == 2


@pytest.mark.parametrize("output", list(_OUTPUTS))
@pytest.mark.parametrize("mark", sorted(_MARKS))
def test_a_zero_row_statistic_leaf_beside_a_populated_leaf_draws_its_frame_and_warns_once(
    mark: str, output: str
):
    grid = _MARKS[mark](_empty_stat_frame()) | _MARKS[mark](_populated_stat_frame())
    page, messages = _render_capturing(_OUTPUTS[output](grid))
    assert messages == [_NO_ROWS_TO_DRAW]
    if output == "static":
        # each leaf draws its own axis titles: the populated leaf alone gives one of each
        assert page.count(">k<") == 2 and page.count(">v<") == 2
    else:
        assert len(json.loads(page)["panels"]) == 2


@pytest.mark.parametrize("output", list(_OUTPUTS))
def test_a_zero_row_histogram_leaf_beside_a_populated_leaf_warns_once(output: str):
    def histogram(frame: pl.DataFrame) -> fm.Chart:
        return fm.Chart(frame).mark_bar().encode(x=fm.X("x", bin=True), y="count()")

    grid = histogram(_populated().clear()) | histogram(_populated())
    page, messages = _render_capturing(_OUTPUTS[output](grid))
    assert messages == [_NO_ROWS_TO_DRAW]
    if output == "static":
        assert page.count(">x<") == 2
    else:
        assert len(json.loads(page)["panels"]) == 2


def test_a_zero_row_leaf_inside_a_nested_concat_draws_and_warns_once():
    grid = (_populated_leaf() | _zero_row_leaf()) & _populated_leaf()
    svg, messages = _render_capturing(grid.to_svg)
    assert messages == [_NO_ROWS_TO_DRAW]
    assert ">u<" in svg


# a filter-emptied leaf and a LayerChart keep the bytes they had: digests measured on the
# build before the lowering change
_FILTER_EMPTIED_STATIC = "c1bae2df847ab9dd2447cf566602f44fc7aeea56de1f5160b7927d0213a01458"
_FILTER_EMPTIED_INTERACTIVE = "34e0fb92ba464966570384629d159384015f5ae8ee147a4f3d61894983bb7d1c"
_LAYER_STATIC = "4f4034c67a84a324c58c53764db3e6adc3aa9475310c7d89c0abd166f2a84d5e"
_LAYER_INTERACTIVE = "d00b1c2234f8704778441b0a66f9b3fbb6cbd9ac3ddb0fb3260ace641196da62"


def test_a_filter_emptied_leaf_beside_a_populated_leaf_keeps_its_bytes():
    grid = _populated_leaf() | _filter_emptied_leaf()
    svg, messages = _render_capturing(grid.to_svg)
    assert messages == [_NO_ROWS_TO_DRAW]
    assert _sha(svg) == _FILTER_EMPTIED_STATIC
    scene, interactive = _render_capturing(_interactive(grid))
    assert interactive == [_NO_ROWS_TO_DRAW]
    assert _sha(scene) == _FILTER_EMPTIED_INTERACTIVE


def test_a_layer_chart_with_a_zero_row_member_keeps_its_bytes_and_stays_silent():
    z = _leaf(_populated().clear())
    layered = fm.LayerChart(_populated_leaf(), z)
    svg, messages = _render_capturing(layered.to_svg)
    assert messages == []
    assert _sha(svg) == _LAYER_STATIC
    scene, interactive = _render_capturing(_interactive(layered))
    assert interactive == []
    assert _sha(scene) == _LAYER_INTERACTIVE


@pytest.mark.parametrize(
    "build",
    [
        lambda: _zero_row_leaf() | _zero_row_leaf(),
        lambda: _zero_row_leaf() & _zero_row_leaf(),
        lambda: fm.concat(_zero_row_leaf(), _zero_row_leaf()),
    ],
    ids=["hconcat", "vconcat", "concat"],
)
def test_a_composition_of_only_zero_row_leaves_still_raises(build):
    kind = type(build()).__name__
    with pytest.raises(ValueError) as caught:
        build().to_svg()
    assert str(caught.value) == f"{kind}: every child's data is empty; nothing to render"


def test_a_zero_row_pairplot_still_raises():
    frame = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Float64, "b": pl.Float64})
    with pytest.raises(ValueError) as caught:
        fm.pairplot(frame).to_svg()
    assert str(caught.value) == (
        "RepeatChart (2x2): every panel's data is empty; nothing to render"
    )


def test_a_grid_cell_with_no_chart_stays_a_layout_hole():
    """A jointplot-shaped grid with both marginals has a ``None`` corner cell: it stays a hole,
    draws no panel and warns nothing."""
    marginal = fm.Chart(_populated()).mark_histogram().encode(x="x")
    side = fm.Chart(_populated()).mark_histogram().encode(x="y")
    joint = fm.JointChart(_populated_leaf(), top=marginal, right=side)
    lowered = joint._composite_tree(auto_tooltips=False)
    kinds = [c["kind"] for c in lowered.tree["children"]]
    assert kinds.count("hole") == 1 and kinds.count("leaf") == 3
    svg, messages = _render_capturing(joint.to_svg)
    assert messages == []
    assert len(panel_extents(svg)) == 3
    scene, interactive = _render_capturing(_interactive(joint))
    assert interactive == []
    assert len(json.loads(scene)["panels"]) == 3


def _refusal(render: Callable[[], Any]) -> str:
    with pytest.raises(ValueError) as caught:
        render()
    return str(caught.value)


# A concat or grid whose every leaf has zero rows is refused however its layers were combined: an
# overlay LayerChart, a shared or independent y resolve, ``+`` and ``Chart.layer`` all raise the
# same all-empty refusal.
def _empty_layer_chart(**kwargs: Any) -> fm.LayerChart:
    return fm.LayerChart(_leaf(_populated().clear()), _leaf(_populated().clear()), **kwargs)


_NESTED_ALL_EMPTY = {
    "overlay": lambda: fm.hconcat(_empty_layer_chart(), _empty_layer_chart()),
    "overlay_alone": lambda: fm.hconcat(_empty_layer_chart()),
    "shared": lambda: fm.hconcat(
        _empty_layer_chart(resolve={"y": "shared"}), _empty_layer_chart(resolve={"y": "shared"})
    ),
    "independent_y": lambda: fm.hconcat(
        _empty_layer_chart(resolve={"y": "independent"}),
        _empty_layer_chart(resolve={"y": "independent"}),
    ),
    "plus": lambda: fm.hconcat(_leaf(_populated().clear()) + _leaf(_populated().clear())),
    "chart_layer": lambda: fm.hconcat(_leaf(_populated().clear()).layer(fm.Layer(mark="line"))),
}


@pytest.mark.parametrize("output", list(_OUTPUTS))
@pytest.mark.parametrize("spelling", list(_NESTED_ALL_EMPTY))
def test_a_concat_whose_every_leaf_is_an_all_zero_row_layered_chart_is_refused(
    spelling: str, output: str
):
    chart = _NESTED_ALL_EMPTY[spelling]()
    assert _refusal(_OUTPUTS[output](chart)) == (
        "HConcatChart: every child's data is empty; nothing to render"
    )


@pytest.mark.parametrize("output", list(_OUTPUTS))
def test_a_grid_whose_every_leaf_is_zero_row_with_a_layered_centre_is_refused(output: str):
    marginal = fm.Chart(_populated().clear()).mark_histogram().encode(x="x")
    joint = fm.JointChart(_empty_layer_chart(), top=marginal)
    assert _refusal(_OUTPUTS[output](joint)) == (
        "JointChart (2x1): every panel's data is empty; nothing to render"
    )


@pytest.mark.parametrize("output", list(_OUTPUTS))
@pytest.mark.parametrize("resolve", [None, {"y": "shared"}, {"y": "independent"}])
def test_a_top_level_all_zero_row_layer_chart_still_draws_its_frame_and_warns(
    resolve: Any, output: str
):
    chart = _empty_layer_chart(resolve=resolve) if resolve else _empty_layer_chart()
    _, messages = _render_capturing(_OUTPUTS[output](chart))
    assert messages == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("output", list(_OUTPUTS))
def test_a_nested_layer_chart_with_a_populated_layer_still_draws(output: str):
    layered = fm.LayerChart(_populated_leaf(), _leaf(_populated().clear()))
    page, messages = _render_capturing(_OUTPUTS[output](fm.hconcat(layered, _populated_leaf())))
    assert messages == []
    assert page


@pytest.mark.parametrize("output", list(_OUTPUTS))
def test_a_zero_row_layer_chart_beside_a_populated_leaf_draws_and_warns_once(
    output: str,
):
    grid = _empty_layer_chart() | _populated_leaf()
    _, messages = _render_capturing(_OUTPUTS[output](grid))
    assert messages == [_NO_ROWS_TO_DRAW]


# A zero-row frame that lacks a column its encoding names is refused with the single chart's
# sentence; a composite leaf carries it behind the renderer's leaf prefix.
_UNKNOWN_COLUMN = "unknown column 'x' referenced by an encoding"


def _columnless_leaf() -> fm.Chart:
    return fm.Chart(pl.DataFrame()).mark_point().encode(x="x", y="y")


def test_a_zero_row_frame_without_an_encoded_column_is_refused_by_a_single_chart():
    assert _refusal(_columnless_leaf().to_svg) == _UNKNOWN_COLUMN


@pytest.mark.parametrize("output", list(_OUTPUTS))
def test_a_zero_row_concat_leaf_without_an_encoded_column_is_refused_with_the_charts_sentence(
    output: str,
):
    grid = _populated_leaf() | _columnless_leaf()
    assert _refusal(_OUTPUTS[output](grid)) == (
        f"failed to render composite leaf leaf #1: {_UNKNOWN_COLUMN}"
    )


@pytest.mark.parametrize("output", list(_OUTPUTS))
def test_a_zero_row_grid_leaf_without_an_encoded_column_is_refused_with_the_charts_sentence(
    output: str,
):
    joint = fm.JointChart(_populated_leaf(), top=_columnless_leaf())
    assert _refusal(_OUTPUTS[output](joint)) == (
        f"failed to render composite leaf leaf #0: {_UNKNOWN_COLUMN}"
    )
