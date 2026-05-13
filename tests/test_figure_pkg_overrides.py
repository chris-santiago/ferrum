"""Override passthrough tests for the src/ferrum/figure/ package functions.

Covers lmplot, residplot, catplot, displot, heatmap, pairplot, jointplot,
clustermap with SVG round-trip verification and edge cases.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum.layer import Layer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scatter_df():
    rng = np.random.RandomState(42)
    x = rng.randn(50)
    return pl.DataFrame({
        "x": x,
        "y": x * 2 + rng.randn(50) * 0.5,
        "group": ["a", "b"] * 25,
    })


@pytest.fixture(scope="module")
def cat_df():
    return pl.DataFrame({
        "category": ["A", "B", "C", "D"] * 10,
        "value": list(range(40)),
        "hue": ["x", "y"] * 20,
    })


@pytest.fixture(scope="module")
def heatmap_df():
    return pl.DataFrame({
        "row": ["a", "b", "c"],
        "x": [1.0, 0.5, 0.3],
        "y": [0.5, 1.0, 0.7],
        "z": [0.3, 0.7, 1.0],
    })


@pytest.fixture(scope="module")
def iris_df():
    rng = np.random.RandomState(42)
    return pl.DataFrame({
        "sl": rng.randn(30),
        "sw": rng.randn(30),
        "pl": rng.randn(30),
        "species": (["a"] * 10 + ["b"] * 10 + ["c"] * 10),
    })


# ---------------------------------------------------------------------------
# lmplot
# ---------------------------------------------------------------------------

class TestLmplot:
    def test_properties_width_reaches_svg(self, scatter_df):
        svg = fm.lmplot(
            scatter_df, x="x", y="y", properties={"width": 800}
        ).show_svg()
        assert 'width="800"' in svg

    def test_properties_title_reaches_svg(self, scatter_df):
        svg = fm.lmplot(
            scatter_df, x="x", y="y", properties={"title": "LM Custom"}
        ).show_svg()
        assert "LM Custom" in svg

    def test_encode_override_title(self, scatter_df):
        svg = fm.lmplot(
            scatter_df, x="x", y="y",
            encode={"x": fm.X("x", title="My X Axis")},
        ).show_svg()
        assert "My X Axis" in svg

    def test_empty_overrides_noop(self, scatter_df):
        default = fm.lmplot(scatter_df, x="x", y="y").show_svg()
        overridden = fm.lmplot(
            scatter_df, x="x", y="y",
            mark={}, encode={}, properties={}, layers=[],
        ).show_svg()
        assert default == overridden

    def test_mark_override_stroke_width(self, scatter_df):
        svg = fm.lmplot(
            scatter_df, x="x", y="y",
            mark={"line": {"stroke_width": 5}},
        ).show_svg()
        assert 'stroke-width="5"' in svg


# ---------------------------------------------------------------------------
# residplot
# ---------------------------------------------------------------------------

class TestResidplot:
    def test_properties_reaches_svg(self, scatter_df):
        svg = fm.residplot(
            scatter_df, x="x", y="y", properties={"width": 700}
        ).show_svg()
        assert 'width="700"' in svg

    def test_mark_suppress_reference(self, scatter_df):
        default_svg = fm.residplot(scatter_df, x="x", y="y").show_svg()
        suppressed_svg = fm.residplot(
            scatter_df, x="x", y="y", mark={"reference": False}
        ).show_svg()
        default_dashes = default_svg.count("stroke-dasharray")
        suppressed_dashes = suppressed_svg.count("stroke-dasharray")
        assert suppressed_dashes < default_dashes

    def test_mark_suppress_metrics(self, scatter_df):
        default_svg = fm.residplot(scatter_df, x="x", y="y").show_svg()
        suppressed_svg = fm.residplot(
            scatter_df, x="x", y="y", mark={"metrics": False}
        ).show_svg()
        default_texts = default_svg.count("<text ")
        suppressed_texts = suppressed_svg.count("<text ")
        assert suppressed_texts < default_texts

    def test_encode_override(self, scatter_df):
        svg = fm.residplot(
            scatter_df, x="x", y="y",
            encode={"x": fm.X("x", title="Predictor")},
        ).show_svg()
        assert "Predictor" in svg


# ---------------------------------------------------------------------------
# displot
# ---------------------------------------------------------------------------

class TestDisplot:
    def test_properties_width(self, scatter_df):
        svg = fm.displot(scatter_df, x="x", properties={"width": 600}).show_svg()
        assert 'width="600"' in svg

    def test_properties_title(self, scatter_df):
        svg = fm.displot(
            scatter_df, x="x", properties={"title": "Dist Custom"}
        ).show_svg()
        assert "Dist Custom" in svg

    def test_empty_overrides_noop(self, scatter_df):
        default = fm.displot(scatter_df, x="x").show_svg()
        overridden = fm.displot(
            scatter_df, x="x", mark={}, encode={}, properties={}, layers=[],
        ).show_svg()
        assert default == overridden


# ---------------------------------------------------------------------------
# catplot
# ---------------------------------------------------------------------------

class TestCatplot:
    def test_properties_width(self, cat_df):
        svg = fm.catplot(
            cat_df, x="category", y="value", properties={"width": 600}
        ).show_svg()
        assert 'width="600"' in svg

    def test_properties_title(self, cat_df):
        svg = fm.catplot(
            cat_df, x="category", y="value",
            properties={"title": "Cat Custom"},
        ).show_svg()
        assert "Cat Custom" in svg


# ---------------------------------------------------------------------------
# heatmap
# ---------------------------------------------------------------------------

class TestHeatmap:
    def test_properties_title(self, heatmap_df):
        svg = fm.heatmap(
            heatmap_df, properties={"title": "Heat Custom"}
        ).show_svg()
        assert "Heat Custom" in svg

    def test_mark_flat_kwarg_on_single_mark(self, heatmap_df):
        svg = fm.heatmap(
            heatmap_df, annot=False, mark={"opacity": 0.5}
        ).show_svg()
        assert "rgba(" in svg

    def test_annot_false_vs_true_fewer_texts(self, heatmap_df):
        no_annot = fm.heatmap(heatmap_df, annot=False).show_svg()
        with_annot = fm.heatmap(heatmap_df, annot=True).show_svg()
        assert no_annot.count("<text ") < with_annot.count("<text ")


# ---------------------------------------------------------------------------
# pairplot (RepeatChart — properties only, mark/encode/layers not fanned out)
# ---------------------------------------------------------------------------

class TestPairplot:
    def test_properties_title(self, iris_df):
        chart = fm.pairplot(iris_df, vars=["sl", "sw"])
        updated = chart.properties(title="Pair Custom")
        svg = updated.show_svg()
        assert "Pair Custom" in svg

    def test_override_properties_on_pairplot(self, iris_df):
        svg = fm.pairplot(
            iris_df, vars=["sl", "sw"],
            properties={"title": "Pairplot Override"},
        ).show_svg()
        assert "Pairplot Override" in svg


# ---------------------------------------------------------------------------
# jointplot
# ---------------------------------------------------------------------------

class TestJointplot:
    def test_properties_title(self, scatter_df):
        svg = fm.jointplot(
            scatter_df, x="x", y="y",
            properties={"title": "Joint Custom"},
        ).show_svg()
        assert "Joint Custom" in svg


# ---------------------------------------------------------------------------
# clustermap
# ---------------------------------------------------------------------------

class TestClustermap:
    def test_properties_title(self, heatmap_df):
        svg = fm.clustermap(
            heatmap_df, properties={"title": "Cluster Custom"}
        ).show_svg()
        assert "Cluster Custom" in svg


# ---------------------------------------------------------------------------
# Edge cases across figure/ package
# ---------------------------------------------------------------------------

class TestFigurePkgEdgeCases:
    def test_all_four_overrides_lmplot(self, scatter_df):
        svg = fm.lmplot(
            scatter_df, x="x", y="y",
            mark={"line": {"stroke_width": 4}},
            encode={"x": fm.X("x", title="Overridden X")},
            properties={"width": 750, "title": "All Four"},
            layers=[Layer(mark="rule", encoding={"y": "y"})],
        ).show_svg()
        assert 'stroke-width="4"' in svg
        assert "Overridden X" in svg
        assert 'width="750"' in svg
        assert "All Four" in svg

    def test_post_hoc_chaining_lmplot(self, scatter_df):
        chart = fm.lmplot(
            scatter_df, x="x", y="y",
            mark={"line": {"stroke_width": 3}},
        ).properties(width=500)
        svg = chart.show_svg()
        assert 'width="500"' in svg
        assert 'stroke-width="3"' in svg

    def test_residplot_suppress_optional_layers(self, scatter_df):
        svg = fm.residplot(
            scatter_df, x="x", y="y",
            mark={"reference": False, "metrics": False},
        ).show_svg()
        assert "<svg" in svg
        assert svg.count("stroke-dasharray") == 0

    def test_displot_properties_combined(self, scatter_df):
        svg = fm.displot(
            scatter_df, x="x",
            properties={"width": 800, "title": "Dist Combined"},
        ).show_svg()
        assert 'width="800"' in svg
        assert "Dist Combined" in svg


# ---------------------------------------------------------------------------
# Named layer verification — figure/ package functions
# ---------------------------------------------------------------------------

class TestFigurePkgLayerNames:
    def test_lmplot_layer_names(self, scatter_df):
        chart = fm.lmplot(scatter_df, x="x", y="y")
        names = chart.layer_names
        assert "scatter" in names
        assert "line" in names

    def test_lmplot_suppress_scatter(self, scatter_df):
        default_svg = fm.lmplot(scatter_df, x="x", y="y").show_svg()
        suppressed_svg = fm.lmplot(
            scatter_df, x="x", y="y", mark={"scatter": False}
        ).show_svg()
        default_circles = default_svg.count("<circle")
        suppressed_circles = suppressed_svg.count("<circle")
        assert suppressed_circles < default_circles

    def test_lmplot_override_line_stroke_width(self, scatter_df):
        svg = fm.lmplot(
            scatter_df, x="x", y="y",
            mark={"line": {"stroke_width": 5}},
        ).show_svg()
        assert 'stroke-width="5"' in svg

    def test_residplot_layer_names(self, scatter_df):
        chart = fm.residplot(scatter_df, x="x", y="y")
        names = chart.layer_names
        assert "point" in names
        assert "reference" in names
        assert "metrics" in names

    def test_displot_single_kind_no_layers(self, scatter_df):
        chart = fm.displot(scatter_df, x="x")
        assert chart.layer_names == []

    def test_displot_with_kde_has_named_layers(self, scatter_df):
        chart = fm.displot(scatter_df, x="x", kde=True)
        names = chart.layer_names
        assert "histogram" in names
        assert "kde" in names

    def test_heatmap_layer_names(self, heatmap_df):
        chart = fm.heatmap(heatmap_df, annot=True)
        names = chart.layer_names
        assert "cells" in names
        assert "label" in names

    def test_heatmap_suppress_label_svg(self, heatmap_df):
        default_svg = fm.heatmap(heatmap_df, annot=True).show_svg()
        suppressed_svg = fm.heatmap(
            heatmap_df, annot=True, mark={"label": False}
        ).show_svg()
        default_texts = default_svg.count("<text ")
        suppressed_texts = suppressed_svg.count("<text ")
        assert suppressed_texts < default_texts

    def test_heatmap_override_cells_opacity(self, heatmap_df):
        svg = fm.heatmap(
            heatmap_df, annot=True, mark={"cells": {"opacity": 0.3}}
        ).show_svg()
        assert "rgba(" in svg
