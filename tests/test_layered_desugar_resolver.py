"""Tests for the layered desugar protocol and _Layer conversion.
Uses a fake desugar so we don't depend on any composite mark being implemented yet."""
import polars as pl
import pytest
import ferrum as fe
from ferrum._layer import MarkDesugarResult, _Layer, _PendingMark


def _fake_layered_desugar(x_field, y_field, **_kwargs):
    """Returns a MarkDesugarResult with layers."""
    layers = [
        _Layer(mark="rule", encoding={"x": x_field, "y": y_field}),
        _Layer(mark="point", encoding={"x": x_field, "y": y_field},
               mark_kwargs={"size": 5}),
    ]
    return MarkDesugarResult(layers=layers)


@pytest.fixture
def df():
    return pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})


def test_layered_sentinel_produces_multi_layer_spec(df):
    chart = fe.Chart(df)
    chart._pending_stat_mark = _PendingMark("__fake__", {}, _fake_layered_desugar)
    spec = chart.encode(x="x", y="y")._build_spec()
    assert spec.layers is not None, "expected multi-layer ChartSpec"
    assert len(spec.layers) == 2
    layer_names = [l.mark.name for l in spec.layers]
    assert layer_names == ["rule", "point"]


def test_layered_layer_carries_mark_kwargs(df):
    chart = fe.Chart(df)
    chart._pending_stat_mark = _PendingMark("__fake__", {}, _fake_layered_desugar)
    spec = chart.encode(x="x", y="y")._build_spec()
    point_layer = spec.layers[1]
    assert point_layer.mark_kwargs is not None
    json = spec.to_json()
    assert '"size":5' in json or '"size": 5' in json


def test_layered_layer_carries_data_source_when_set(df):
    def desugar_with_named_source(x_field, y_field, **_kw):
        layers = [
            _Layer(mark="rule", encoding={"x": x_field, "y": y_field}, data_source="box"),
        ]
        return MarkDesugarResult(layers=layers)
    chart = fe.Chart(df)
    chart._pending_stat_mark = _PendingMark("__fake__", {}, desugar_with_named_source)
    spec = chart.encode(x="x", y="y")._build_spec()
    assert spec.layers[0].data_source == "box"


def test_layered_encoding_y2_supported(df):
    """Layered desugar may emit y2 in encoding (used by ribbon/errorband layers)."""
    def desugar_with_y2(x_field, y_field, **_kw):
        layers = [
            _Layer(mark="ribbon", encoding={"x": x_field, "y": y_field, "y2": "y"}),
        ]
        return MarkDesugarResult(layers=layers)
    chart = fe.Chart(df)
    chart._pending_stat_mark = _PendingMark("__fake__", {}, desugar_with_y2)
    spec = chart.encode(x="x", y="y")._build_spec()
    json = spec.to_json()
    assert '"y2"' in json or "y2" in json
