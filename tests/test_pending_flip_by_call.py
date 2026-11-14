"""A pending composite mark records that its call applied the flip on every branch that resolves it.

``Chart._with_flip_by_call`` stamps the pending mark, and resolving it stamps the layers it emits
(or, for a single mark, the chart itself) with the same fact, so a message about any of them names
the channels the caller wrote. The layered branch, the single mark beside a prior layer and the
flat single mark all carry it; a prior layer the caller drew keeps its own naming.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm

_FRAME = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        "g": ["a", "a", "b", "b", "c", "c"],
    }
)


def _layers(chart: fm.Chart) -> list[tuple[str, bool]]:
    resolved = chart._resolve_pending()
    return [(layer.mark, layer.flip_by_call) for layer in resolved._layers or []]


def test_a_composite_that_emits_layers_stamps_each_of_them():
    chart = fm.Chart(_FRAME).mark_boxplot().encode(x="g", y="y")._with_flip_by_call()
    layers = _layers(chart)
    assert len(layers) == 6
    assert all(flipped for _, flipped in layers)


def test_a_single_mark_beside_a_prior_layer_is_stamped_and_the_prior_layer_is_not():
    chart = fm.Chart(_FRAME).mark_point().encode(x="x", y="y").mark_smooth()._with_flip_by_call()
    assert _layers(chart) == [("point", False), ("line", True)]


@pytest.mark.parametrize(
    ("mark", "encoding", "drawn"),
    [
        pytest.param("mark_histogram", {"x": "x"}, "bar", id="histogram"),
        pytest.param("mark_smooth", {"x": "x", "y": "y"}, "line", id="smooth"),
    ],
)
def test_a_flat_single_mark_is_stamped(mark: str, encoding: dict[str, str], drawn: str):
    chart = getattr(fm.Chart(_FRAME), mark)().encode(**encoding)._with_flip_by_call()
    resolved = chart._resolve_pending()
    assert not resolved._layers
    assert resolved._mark == drawn
    assert resolved._flip_by_call is True


def test_a_flat_single_mark_with_no_stamp_records_none():
    resolved = fm.Chart(_FRAME).mark_histogram().encode(x="x")._resolve_pending()
    assert resolved._flip_by_call is False
