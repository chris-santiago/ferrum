"""A fallback notice an ``impute=`` raises is reported once per chart for layers that share the
same completed rows, and once per layer for layers that complete separately.

Two layers whose data and ``impute=`` are identical complete the same series from the same rows, so
the whole-column fallback of the series with no usable value is one fact about the chart, and every
layering door reports it once: ``+``, ``Chart.layer`` and ``LayerChart`` (shared and independent
y, static and interactive). Layers that impute differently (here a different ``frame``) each
complete their own series and each report their own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from tests.test_impute_every_dtype import _no_usable_notice
from tests.test_null_group_warning import _render_capturing

#: Series ``y`` holds no usable value, so its fill falls back to the whole column's statistic.
_FRAME = pl.DataFrame(
    {
        "t": [1, 2, 3, 1, 2],
        "h": ["x", "x", "x", "y", "y"],
        "v": pl.Series("v", [1.0, 2.0, 3.0, None, None], dtype=pl.Float64),
    }
)
_NOTICE = _no_usable_notice("h=y")

_SAME: dict[str, Any] = {"method": "mean"}
_OTHER: dict[str, Any] = {"method": "mean", "frame": [1, 1]}


def _layer(mark: str, options: dict[str, Any]) -> fm.Chart:
    return getattr(fm.Chart(_FRAME), mark)().encode(x="t", y=fm.Y("v", impute=options), color="h")


def _static(chart: Any) -> list[str]:
    return _render_capturing(chart.to_svg)[1]


def _interactive(chart: Any) -> list[str]:
    return _render_capturing(lambda: chart.interactive().scene_json)[1]


_DOORS: dict[str, Callable[[dict[str, Any], dict[str, Any]], Any]] = {
    "plus": lambda a, b: _layer("mark_line", a) + _layer("mark_point", b),
    "layer-chart-shared": lambda a, b: fm.LayerChart(
        _layer("mark_line", a), _layer("mark_point", b)
    ),
    "layer-chart-independent-y": lambda a, b: fm.LayerChart(
        _layer("mark_line", a), _layer("mark_point", b), resolve={"y": "independent"}
    ),
}
_OUTPUTS = {"static": _static, "interactive": _interactive}


@pytest.mark.parametrize("output", sorted(_OUTPUTS))
@pytest.mark.parametrize("door", sorted(_DOORS))
def test_layers_that_share_their_completed_rows_report_the_fallback_once(door: str, output: str):
    assert _OUTPUTS[output](_DOORS[door](_SAME, _SAME)) == [_NOTICE]


@pytest.mark.parametrize("output", sorted(_OUTPUTS))
@pytest.mark.parametrize("door", sorted(_DOORS))
def test_layers_that_impute_separately_each_report_their_own_fallback(door: str, output: str):
    assert _OUTPUTS[output](_DOORS[door](_SAME, _OTHER)) == [_NOTICE, _NOTICE]


@pytest.mark.parametrize("output", sorted(_OUTPUTS))
def test_a_chart_layer_door_reports_the_shared_fallback_once(output: str):
    chart = _layer("mark_line", _SAME).layer(fm.Layer(mark="point"))
    assert _OUTPUTS[output](chart) == [_NOTICE]
