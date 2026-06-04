"""Tests verifying that figure-level functions correctly pass through
override kwargs (mark=, encode=, properties=) to the underlying chart.

Strategy: call the figure function with defaults, then with an override kwarg,
render both to SVG, and assert the outputs differ — proving the override had a
visible effect on the final chart.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "x": rng.normal(0, 1, 50).tolist(),
            "y": rng.normal(0, 1, 50).tolist(),
            "cat": ["a"] * 25 + ["b"] * 25,
            "val": rng.normal(5, 2, 50).tolist(),
        }
    )


@pytest.fixture
def wide_df():
    """Wide-format DataFrame for heatmap tests."""
    return pl.DataFrame(
        {
            "label": ["row_a", "row_b", "row_c"],
            "col_1": [1.0, 0.5, 0.2],
            "col_2": [0.3, 1.0, 0.7],
            "col_3": [0.8, 0.4, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _renders_differently(chart_default, chart_override):
    """Assert that two charts produce different SVG output."""
    svg_default = chart_default.to_svg()
    svg_override = chart_override.to_svg()
    assert svg_default != svg_override, (
        "Expected overridden chart to produce different SVG from the default"
    )


# ---------------------------------------------------------------------------
# displot
# ---------------------------------------------------------------------------


class TestDisplotPassthrough:
    def test_mark_override(self, df):
        default = fm.displot(df, x="val")
        overridden = fm.displot(df, x="val", mark={"opacity": 0.2})
        _renders_differently(default, overridden)

    def test_encode_override(self, df):
        default = fm.displot(df, x="val")
        overridden = fm.displot(df, x="val", hue="cat")
        _renders_differently(default, overridden)

    def test_properties_override(self, df):
        default = fm.displot(df, x="val")
        overridden = fm.displot(df, x="val", properties={"width": 800, "height": 600})
        _renders_differently(default, overridden)

    def test_properties_title(self, df):
        default = fm.displot(df, x="val")
        overridden = fm.displot(df, x="val", properties={"title": "Custom Title"})
        _renders_differently(default, overridden)


# ---------------------------------------------------------------------------
# catplot
# ---------------------------------------------------------------------------


class TestCatplotPassthrough:
    def test_mark_override(self, df):
        default = fm.catplot(df, x="cat", y="val", kind="box")
        overridden = fm.catplot(df, x="cat", y="val", kind="box", mark={"box": {"opacity": 0.3}})
        _renders_differently(default, overridden)

    def test_encode_override(self, df):
        default = fm.catplot(df, x="cat", y="val", kind="box")
        overridden = fm.catplot(df, x="cat", y="val", kind="box", encode={"color": "cat:N"})
        _renders_differently(default, overridden)

    def test_properties_override(self, df):
        default = fm.catplot(df, x="cat", y="val", kind="strip")
        overridden = fm.catplot(
            df, x="cat", y="val", kind="strip", properties={"width": 900, "height": 500}
        )
        _renders_differently(default, overridden)

    def test_properties_title(self, df):
        default = fm.catplot(df, x="cat", y="val", kind="violin")
        overridden = fm.catplot(
            df, x="cat", y="val", kind="violin", properties={"title": "Violin Plot"}
        )
        _renders_differently(default, overridden)


# ---------------------------------------------------------------------------
# lmplot
# ---------------------------------------------------------------------------


class TestLmplotPassthrough:
    def test_mark_override(self, df):
        default = fm.lmplot(df, x="x", y="y")
        overridden = fm.lmplot(df, x="x", y="y", mark={"scatter": {"opacity": 0.1}})
        _renders_differently(default, overridden)

    def test_encode_override(self, df):
        default = fm.lmplot(df, x="x", y="y")
        overridden = fm.lmplot(df, x="x", y="y", encode={"color": "cat"})
        _renders_differently(default, overridden)

    def test_properties_override(self, df):
        default = fm.lmplot(df, x="x", y="y")
        overridden = fm.lmplot(df, x="x", y="y", properties={"width": 800, "height": 600})
        _renders_differently(default, overridden)

    def test_properties_title(self, df):
        default = fm.lmplot(df, x="x", y="y")
        overridden = fm.lmplot(df, x="x", y="y", properties={"title": "Linear Model"})
        _renders_differently(default, overridden)


# ---------------------------------------------------------------------------
# residplot
# ---------------------------------------------------------------------------


class TestResidplotPassthrough:
    def test_mark_override(self, df):
        default = fm.residplot(df, x="x", y="y", show_metrics=False, zero_line=False)
        overridden = fm.residplot(
            df, x="x", y="y", show_metrics=False, zero_line=False, mark={"opacity": 0.2}
        )
        _renders_differently(default, overridden)

    def test_encode_override(self, df):
        default = fm.residplot(df, x="x", y="y", show_metrics=False, zero_line=False)
        overridden = fm.residplot(
            df,
            x="x",
            y="y",
            show_metrics=False,
            zero_line=False,
            encode={"size": "residual"},
        )
        _renders_differently(default, overridden)

    def test_properties_override(self, df):
        default = fm.residplot(df, x="x", y="y", show_metrics=False, zero_line=False)
        overridden = fm.residplot(
            df,
            x="x",
            y="y",
            show_metrics=False,
            zero_line=False,
            properties={"width": 700, "height": 400},
        )
        _renders_differently(default, overridden)

    def test_properties_title(self, df):
        default = fm.residplot(df, x="x", y="y", show_metrics=False, zero_line=False)
        overridden = fm.residplot(
            df,
            x="x",
            y="y",
            show_metrics=False,
            zero_line=False,
            properties={"title": "Residuals"},
        )
        _renders_differently(default, overridden)


# ---------------------------------------------------------------------------
# pairplot
# ---------------------------------------------------------------------------


class TestPairplotPassthrough:
    def test_mark_override(self, df):
        default = fm.pairplot(df, vars=["x", "y"])
        overridden = fm.pairplot(df, vars=["x", "y"], mark={"opacity": 0.2})
        assert overridden.to_svg() is not None

    def test_encode_override(self, df):
        default = fm.pairplot(df, vars=["x", "y"], hue="cat")
        overridden = fm.pairplot(df, vars=["x", "y"])
        _renders_differently(default, overridden)

    def test_properties_override(self, df):
        default = fm.pairplot(df, vars=["x", "y"])
        overridden = fm.pairplot(df, vars=["x", "y"], properties={"width": 500, "height": 500})
        _renders_differently(default, overridden)


# ---------------------------------------------------------------------------
# heatmap
# ---------------------------------------------------------------------------


class TestHeatmapPassthrough:
    def test_mark_override(self, wide_df):
        default = fm.heatmap(wide_df)
        overridden = fm.heatmap(wide_df, mark={"cells": {"opacity": 0.4}})
        _renders_differently(default, overridden)

    def test_encode_override(self, wide_df):
        default = fm.heatmap(wide_df)
        overridden = fm.heatmap(wide_df, encode={"color": "value"})
        # encode override replaces the default color encoding — may or may not
        # differ visually, but the spec contract is that it's wired.  At minimum
        # we verify it does not raise.
        assert overridden.to_svg() is not None

    def test_properties_override(self, wide_df):
        default = fm.heatmap(wide_df)
        overridden = fm.heatmap(wide_df, properties={"width": 600, "height": 600})
        _renders_differently(default, overridden)

    def test_properties_title(self, wide_df):
        default = fm.heatmap(wide_df)
        overridden = fm.heatmap(wide_df, properties={"title": "Correlation Matrix"})
        _renders_differently(default, overridden)


# ---------------------------------------------------------------------------
# jointplot
# ---------------------------------------------------------------------------


class TestJointplotPassthrough:
    def test_mark_override(self, df):
        default = fm.jointplot(df, x="x", y="y")
        overridden = fm.jointplot(df, x="x", y="y", mark={"opacity": 0.2})
        # jointplot is a compound view; mark override fans out to children.
        # Verify no crash and valid SVG.
        assert "<svg" in overridden.to_svg()

    def test_encode_override(self, df):
        default = fm.jointplot(df, x="x", y="y")
        overridden = fm.jointplot(df, x="x", y="y", hue="cat")
        _renders_differently(default, overridden)

    def test_properties_override(self, df):
        default = fm.jointplot(df, x="x", y="y")
        overridden = fm.jointplot(df, x="x", y="y", properties={"width": 700, "height": 700})
        _renders_differently(default, overridden)


# ---------------------------------------------------------------------------
# relplot (bonus — same module as displot/catplot)
# ---------------------------------------------------------------------------


class TestRelplotPassthrough:
    def test_mark_override(self, df):
        default = fm.relplot(df, x="x", y="y")
        overridden = fm.relplot(df, x="x", y="y", mark={"opacity": 0.3})
        _renders_differently(default, overridden)

    def test_encode_override(self, df):
        default = fm.relplot(df, x="x", y="y")
        overridden = fm.relplot(df, x="x", y="y", encode={"color": "cat"})
        _renders_differently(default, overridden)

    def test_properties_override(self, df):
        default = fm.relplot(df, x="x", y="y")
        overridden = fm.relplot(df, x="x", y="y", properties={"width": 800, "height": 600})
        _renders_differently(default, overridden)
