"""Phase 8b end-to-end smoke tests: every new mark renders without panic."""
import polars as pl
import numpy as np
import pytest
import ferrum as fe


@pytest.fixture
def df_grouped():
    np.random.seed(42)
    return pl.DataFrame({
        "g": ["a"] * 30 + ["b"] * 30,
        "v": np.concatenate([np.random.normal(0, 1, 30), np.random.normal(2, 1, 30)]),
    })


@pytest.fixture
def df_2d():
    np.random.seed(42)
    n = 200
    return pl.DataFrame({"x": np.random.normal(0, 1, n), "y": np.random.normal(0, 1, n)})


def test_boxplot_renders(df_grouped):
    svg = fe.Chart(df_grouped).mark_boxplot().encode(x="g", y="v").show_svg()
    assert "<svg" in svg


def test_errorbar_renders(df_grouped):
    svg = fe.Chart(df_grouped).mark_errorbar(extent="ci").encode(x="g", y="v").show_svg()
    assert "<svg" in svg


def test_errorband_renders():
    df_lines = pl.DataFrame({
        "g": ["a"] * 10 + ["b"] * 10,
        "x": [float(i) for i in range(10)] * 2,
        "v": [float(i) + 0.1 for i in range(10)] + [float(i) + 0.5 for i in range(10)],
    })
    svg = fe.Chart(df_lines).mark_errorband(extent="ci").encode(x="x", y="v").show_svg()
    assert "<svg" in svg


def test_ribbon_renders():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "lo": [0.5, 1.5, 2.5], "hi": [1.5, 2.5, 3.5]})
    svg = fe.Chart(df).mark_ribbon().encode(x="x", y="lo", y2="hi").show_svg()
    assert "<svg" in svg


def test_smooth_with_ci_renders():
    np.random.seed(42)
    n = 50
    x = np.linspace(0.0, 10.0, n)
    y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, n)
    df = pl.DataFrame({"x": x, "y": y})
    svg = fe.Chart(df).mark_smooth(ci=0.95).encode(x="x", y="y").show_svg()
    assert "<svg" in svg


def test_contour_renders(df_2d):
    svg = fe.Chart(df_2d).mark_contour().encode(x="x", y="y").show_svg()
    assert "<svg" in svg


def test_violin_renders(df_grouped):
    svg = fe.Chart(df_grouped).mark_violin(inner="box").encode(x="g", y="v").show_svg()
    assert "<svg" in svg


def test_qq_renders(df_2d):
    svg = fe.Chart(df_2d).mark_qq(distribution="normal").encode(x="x").show_svg()
    assert "<svg" in svg


def test_raster_renders(df_2d):
    svg = fe.Chart(df_2d).mark_raster(resolution=64).encode(x="x", y="y").show_svg()
    assert "<svg" in svg
    assert "data:image/png;base64," in svg


def test_swarm_renders(df_grouped):
    svg = fe.Chart(df_grouped).mark_swarm().encode(x="g", y="v").show_svg()
    assert "<svg" in svg


def test_hex_renders(df_2d):
    svg = fe.Chart(df_2d).mark_hex().encode(x="x", y="y").show_svg()
    assert "<svg" in svg


def test_function_renders():
    df = pl.DataFrame({"t": np.linspace(0, np.pi * 2, 50)})
    chart = fe.Chart(df).encode(x="t", y="t").mark_function(np.sin)
    svg = chart.show_svg()
    assert "<svg" in svg


def test_layered_violin_plus_swarm(df_grouped):
    svg = (fe.Chart(df_grouped).mark_violin(inner=None).encode(x="g", y="v") +
           fe.Chart(df_grouped).mark_swarm().encode(x="g", y="v")).show_svg()
    assert "<svg" in svg


def test_layered_errorband_plus_smooth():
    df = pl.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0) + np.random.randn(20)})
    svg = fe.Chart(df).mark_smooth(ci=0.95).encode(x="x", y="y").show_svg()
    assert "<svg" in svg
