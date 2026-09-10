"""Smoke and validation tests for relplot and regplot."""

from __future__ import annotations

import inspect

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
# regplot — col/row are lmplot-only (F-L10-09)
# ---------------------------------------------------------------------------


def test_regplot_col_raises_typeerror_naming_lmplot(df):
    with pytest.raises(TypeError, match="lmplot"):
        fm.regplot(df, x="x", y="y", col="time")


def test_regplot_row_raises_typeerror_naming_lmplot(df):
    with pytest.raises(TypeError, match="lmplot"):
        fm.regplot(df, x="x", y="y", row="group")


# ---------------------------------------------------------------------------
# regplot — leaked params are now real named parameters (F-L10-09)
# ---------------------------------------------------------------------------


def test_regplot_logx_is_named_param_and_matches_lmplot(df):
    chart = fm.regplot(df, x="x", y="y", logx=True)
    equivalent = fm.lmplot(df, x="x", y="y", logx=True)
    assert chart.to_json() == equivalent.to_json()
    assert chart.to_json() != fm.regplot(df, x="x", y="y").to_json()


def test_regplot_show_metrics_is_named_param_and_matches_lmplot(df):
    chart = fm.regplot(df, x="x", y="y", show_metrics=False)
    equivalent = fm.lmplot(df, x="x", y="y", show_metrics=False)
    assert chart.to_json() == equivalent.to_json()
    assert chart.to_json() != fm.regplot(df, x="x", y="y").to_json()


def test_regplot_x_bins_is_named_param_and_matches_lmplot(df):
    chart = fm.regplot(df, x="x", y="y", x_bins=5)
    equivalent = fm.lmplot(df, x="x", y="y", x_bins=5)
    assert chart.to_json() == equivalent.to_json()
    assert chart.to_json() != fm.regplot(df, x="x", y="y").to_json()


def test_regplot_x_estimator_is_named_param_and_matches_lmplot(df):
    # "max" (not "mean") is the discriminating value here: "mean" is the
    # transform's own default, so x_estimator="mean" round-trips to a
    # byte-identical render on this 8-row fixture (bins hold <=2 points)
    # and would still pass even if x_estimator were silently dropped
    # downstream of the spec.
    chart = fm.regplot(df, x="x", y="y", x_bins=5, x_estimator="max")
    equivalent = fm.lmplot(df, x="x", y="y", x_bins=5, x_estimator="max")
    assert chart.to_json() == equivalent.to_json()
    assert chart.to_json() != fm.regplot(df, x="x", y="y", x_bins=5).to_json()


# ---------------------------------------------------------------------------
# regplot — signature parity with lmplot minus {col, row} (spec §6)
# ---------------------------------------------------------------------------


def test_regplot_signature_matches_lmplot_minus_col_row():
    lmplot_params = set(inspect.signature(fm.lmplot).parameters) - {"col", "row"}
    regplot_params = set(inspect.signature(fm.regplot).parameters)
    assert regplot_params == lmplot_params

    # Names alone don't pin the "identical API" contract -- a future default
    # drift on either side (e.g. lmplot's show_metrics=True -> False without
    # a matching regplot change) would leave the name-only check green.
    # Compare defaults too, so both sides of the parity contract are pinned.
    lmplot_defaults = {
        name: param.default
        for name, param in inspect.signature(fm.lmplot).parameters.items()
        if name not in {"col", "row"}
    }
    regplot_defaults = {
        name: param.default for name, param in inspect.signature(fm.regplot).parameters.items()
    }
    assert regplot_defaults == lmplot_defaults


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
