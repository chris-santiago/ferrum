"""Smoke and validation tests for relplot and regplot."""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum import Chart


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.2],
            "group": ["a", "b"] * 4,
            "time": ["am", "am", "am", "am", "pm", "pm", "pm", "pm"],
        }
    )


# ---------------------------------------------------------------------------
# relplot — scatter
# ---------------------------------------------------------------------------


def test_relplot_scatter_returns_chart(df):
    chart = fm.relplot(df, x="x", y="y")
    assert isinstance(chart, Chart)


def test_relplot_scatter_with_hue(df):
    chart = fm.relplot(df, x="x", y="y", hue="group")
    assert isinstance(chart, Chart)


def test_relplot_scatter_with_col_facet(df):
    chart = fm.relplot(df, x="x", y="y", hue="group", col="time")
    assert isinstance(chart, Chart)


def test_relplot_scatter_with_size_and_style(df):
    chart = fm.relplot(df, x="x", y="y", size="x", style="group")
    assert isinstance(chart, Chart)


def test_relplot_scatter_with_row_facet(df):
    chart = fm.relplot(df, x="x", y="y", row="group")
    assert isinstance(chart, Chart)


def test_relplot_scatter_height_aspect(df):
    chart = fm.relplot(df, x="x", y="y", height=300.0, aspect=1.5)
    assert isinstance(chart, Chart)


# ---------------------------------------------------------------------------
# relplot — line
# ---------------------------------------------------------------------------


def test_relplot_line_returns_chart(df):
    chart = fm.relplot(df, x="x", y="y", kind="line")
    assert isinstance(chart, Chart)


def test_relplot_line_with_hue(df):
    chart = fm.relplot(df, x="x", y="y", hue="group", kind="line")
    assert isinstance(chart, Chart)


def test_relplot_line_with_style_maps_stroke_dash(df):
    chart = fm.relplot(df, x="x", y="y", style="group", kind="line")
    assert isinstance(chart, Chart)


# ---------------------------------------------------------------------------
# relplot — invalid kind
# ---------------------------------------------------------------------------


def test_relplot_invalid_kind_raises(df):
    with pytest.raises(ValueError, match="kind must be one of"):
        fm.relplot(df, x="x", y="y", kind="hist")


def test_relplot_invalid_kind_message_lists_valid_values(df):
    with pytest.raises(ValueError, match="scatter"):
        fm.relplot(df, x="x", y="y", kind="bad")


# ---------------------------------------------------------------------------
# regplot
# ---------------------------------------------------------------------------


def test_regplot_basic_returns_chart(df):
    chart = fm.regplot(df, x="x", y="y")
    assert isinstance(chart, Chart)


def test_regplot_with_hue(df):
    chart = fm.regplot(df, x="x", y="y", hue="group")
    assert isinstance(chart, Chart)


def test_regplot_loess_method(df):
    chart = fm.regplot(df, x="x", y="y", method="loess")
    assert isinstance(chart, Chart)


def test_regplot_no_scatter(df):
    chart = fm.regplot(df, x="x", y="y", scatter=False)
    assert isinstance(chart, Chart)


def test_regplot_scatter_kws(df):
    chart = fm.regplot(df, x="x", y="y", scatter_kws={"opacity": 0.3})
    assert isinstance(chart, Chart)


def test_regplot_invalid_method_raises(df):
    with pytest.raises(ValueError, match="method must be one of"):
        fm.regplot(df, x="x", y="y", method="bad_method")


# ---------------------------------------------------------------------------
# Export surface checks
# ---------------------------------------------------------------------------


def test_relplot_in_ferrum_namespace():
    assert hasattr(fm, "relplot")
    assert fm.relplot is not None


def test_regplot_in_ferrum_namespace():
    assert hasattr(fm, "regplot")
    assert fm.regplot is not None


def test_relplot_has_docstring():
    assert fm.relplot.__doc__ is not None
    assert len(fm.relplot.__doc__) > 0


def test_regplot_has_docstring():
    assert fm.regplot.__doc__ is not None
    assert len(fm.regplot.__doc__) > 0
