import polars as pl
import pytest
import ferrum as fe


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
            "y": [1.1, 2.2, 3.3, 4.4, 5.5, 1.0, 2.0, 3.0, 4.0, 5.0, 1.2, 2.1, 3.4, 4.3, 5.2],
        }
    )


def test_errorband_default_1_layer(df):
    chart = fe.Chart(df).mark_errorband().encode(x="x", y="y")
    spec = chart._build_spec()
    assert len(spec.layers) == 1  # ribbon only


def test_errorband_borders_3_layers(df):
    chart = fe.Chart(df).mark_errorband(borders=True).encode(x="x", y="y")
    spec = chart._build_spec()
    assert len(spec.layers) == 3  # ribbon + 2 lines


def test_errorband_extent_stdev(df):
    chart = fe.Chart(df).mark_errorband(method="stdev").encode(x="x", y="y")
    json_str = chart._build_spec().to_json()
    assert "stdev" in json_str


def test_errorband_ribbon_has_y2(df):
    chart = fe.Chart(df).mark_errorband().encode(x="x", y="y")
    json_str = chart._build_spec().to_json()
    assert '"y2"' in json_str


def test_errorband_render_smoke(df):
    chart = fe.Chart(df).mark_errorband().encode(x="x", y="y")
    svg = chart.to_svg()
    assert "<svg" in svg
