import polars as pl
import pytest
import ferrum as fe


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "t": [1.0, 2.0, 3.0, 4.0, 5.0],
            "lo": [0.5, 1.5, 2.5, 3.5, 4.5],
            "hi": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )


def test_ribbon_basic(df):
    chart = fe.Chart(df).mark_ribbon().encode(x="t", y="lo", y2="hi")
    spec = chart._build_spec()
    assert len(spec.layers) == 1


def test_ribbon_missing_y2_raises(df):
    chart = fe.Chart(df).mark_ribbon().encode(x="t", y="lo")
    with pytest.raises(ValueError, match="y2"):
        chart._build_spec()


def test_ribbon_opacity_threaded(df):
    chart = fe.Chart(df).mark_ribbon(opacity=0.5).encode(x="t", y="lo", y2="hi")
    json_str = chart._build_spec().to_json()
    assert "0.5" in json_str  # opacity in mark_style


def test_ribbon_render_smoke(df):
    chart = fe.Chart(df).mark_ribbon().encode(x="t", y="lo", y2="hi")
    svg = chart.to_svg()
    assert "<svg" in svg
    assert "<path" in svg  # ribbon emits a path


def test_ribbon_y2_field_propagates(df):
    chart = fe.Chart(df).mark_ribbon().encode(x="t", y="lo", y2="hi")
    json_str = chart._build_spec().to_json()
    assert '"y2"' in json_str
    assert '"hi"' in json_str  # y2 field name
