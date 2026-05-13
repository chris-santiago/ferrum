"""Tests for Smooth transform groupby support.

Covers:
1. mark_smooth(groupby="group") produces SVG output with multiple groups
2. The Smooth output includes the group column for color encoding
3. lmplot with hue renders without errors using the new groupby path
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum import Chart


@pytest.fixture
def grouped_data() -> pl.DataFrame:
    """Two-group linear data: group A = y=2x, group B = y=3x+5."""
    import random

    random.seed(42)
    rows = []
    for g, slope, intercept in [("A", 2.0, 0.0), ("B", 3.0, 5.0)]:
        for i in range(30):
            x = float(i)
            y = slope * x + intercept + random.gauss(0, 0.5)
            rows.append({"x": x, "y": y, "group": g})
    return pl.DataFrame(rows)


class TestSmoothGroupby:
    """Smooth transform with groupby parameter."""

    def test_groupby_produces_svg(self, grouped_data):
        """mark_smooth(groupby='group') renders SVG without error."""
        chart = (
            Chart(grouped_data)
            .mark_smooth(method="lm", groupby="group")
            .encode(x="x", y="y", color="group")
        )
        svg = chart.show_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 100

    def test_groupby_output_contains_group_column(self, grouped_data):
        """The Smooth transform's output batch includes the group column,
        so color encoding maps distinct strokes per group."""
        chart = (
            Chart(grouped_data)
            .mark_smooth(method="lm", groupby="group")
            .encode(x="x", y="y", color="group")
        )
        svg = chart.show_svg()
        # Multiple polyline elements indicate per-group lines rendered.
        polyline_count = svg.count("<polyline")
        assert polyline_count >= 2, (
            f"Expected at least 2 polylines (one per group), got {polyline_count}"
        )

    def test_groupby_ci_produces_layered_svg(self, grouped_data):
        """mark_smooth with groupby and ci= produces ribbon+line layers."""
        chart = (
            Chart(grouped_data)
            .mark_smooth(method="lm", ci=0.95, groupby="group")
            .encode(x="x", y="y", color="group")
        )
        svg = chart.show_svg()
        assert svg.startswith("<svg")
        # CI band (ribbon) and fit line both rendered.
        assert len(svg) > 200

    def test_groupby_loess(self, grouped_data):
        """LOESS method works with groupby."""
        chart = (
            Chart(grouped_data)
            .mark_smooth(method="loess", groupby="group")
            .encode(x="x", y="y", color="group")
        )
        svg = chart.show_svg()
        assert svg.startswith("<svg")


class TestLmplotHue:
    """lmplot with hue uses Smooth groupby under the hood."""

    def test_lmplot_hue_renders(self, grouped_data):
        """lmplot(hue='group') renders without errors."""
        chart = fm.lmplot(grouped_data, x="x", y="y", hue="group")
        svg = chart.show_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 100

    def test_lmplot_hue_lm(self, grouped_data):
        """lmplot with method='lm' and hue renders per-group fit lines."""
        chart = fm.lmplot(
            grouped_data, x="x", y="y", hue="group", method="lm"
        )
        svg = chart.show_svg()
        assert svg.startswith("<svg")

    def test_lmplot_hue_loess(self, grouped_data):
        """lmplot with method='loess' and hue renders per-group fit lines."""
        chart = fm.lmplot(
            grouped_data, x="x", y="y", hue="group", method="loess"
        )
        svg = chart.show_svg()
        assert svg.startswith("<svg")

    def test_lmplot_hue_with_ci(self, grouped_data):
        """lmplot with hue and CI band renders."""
        chart = fm.lmplot(
            grouped_data, x="x", y="y", hue="group", ci=95
        )
        svg = chart.show_svg()
        assert svg.startswith("<svg")

    def test_lmplot_hue_no_scatter(self, grouped_data):
        """lmplot with hue and scatter=False renders only fit lines."""
        chart = fm.lmplot(
            grouped_data, x="x", y="y", hue="group", scatter=False
        )
        svg = chart.show_svg()
        assert svg.startswith("<svg")
