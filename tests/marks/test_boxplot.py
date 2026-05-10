import polars as pl
import pytest
import ferrum as fe


@pytest.fixture
def df():
    return pl.DataFrame({
        "group": ["a", "a", "a", "b", "b", "b"],
        "value": [1.0, 2.0, 100.0, 4.0, 5.0, 6.0],
    })


def test_boxplot_smoke_4_layers(df):
    chart = fe.Chart(df).mark_boxplot().encode(x="group", y="value")
    spec = chart._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 4  # rule + rect + tick + outliers point


def test_boxplot_no_outliers_3_layers(df):
    chart = fe.Chart(df).mark_boxplot(outliers=False).encode(x="group", y="value")
    spec = chart._build_spec()
    assert len(spec.layers) == 3


def test_boxplot_extent_min_max(df):
    chart = fe.Chart(df).mark_boxplot(extent="min-max").encode(x="group", y="value")
    json_str = chart._build_spec().to_json()
    assert "min-max" in json_str


def test_boxplot_horizontal_swaps_x_y(df):
    chart = fe.Chart(df).mark_boxplot(horizontal=True).encode(x="value", y="group")
    spec = chart._build_spec()
    json_str = spec.to_json()
    assert "lower_whisker" in json_str  # whisker layer present
    assert '"q1"' in json_str  # box stats present


def test_boxplot_render_smoke(df):
    chart = fe.Chart(df).mark_boxplot().encode(x="group", y="value")
    svg = chart.show_svg()
    assert svg.startswith("<?xml") or svg.startswith("<svg")
