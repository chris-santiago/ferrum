import polars as pl
import pytest
import ferrum as fe


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "x": ["a", "b"] * 5,
            "y": [1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.8, 2.8, 1.1, 2.1],
        }
    )


def test_errorbar_default_3_layers(df):
    chart = fe.Chart(df).mark_errorbar().encode(x="x", y="y")
    spec = chart._build_spec()
    assert len(spec.layers) == 3  # rule + 2 ticks


def test_errorbar_no_ticks_1_layer(df):
    chart = fe.Chart(df).mark_errorbar(ticks=False).encode(x="x", y="y")
    spec = chart._build_spec()
    assert len(spec.layers) == 1


def test_errorbar_extent_stderr(df):
    chart = fe.Chart(df).mark_errorbar(method="stderr").encode(x="x", y="y")
    json_str = chart._build_spec().to_json()
    assert "stderr" in json_str


def test_errorbar_uses_error_extent_transform(df):
    chart = fe.Chart(df).mark_errorbar().encode(x="x", y="y")
    json_str = chart._build_spec().to_json()
    assert "error_extent" in json_str  # transform type tag


def test_errorbar_render_smoke(df):
    chart = fe.Chart(df).mark_errorbar().encode(x="x", y="y")
    svg = chart.to_svg()
    assert "<svg" in svg
