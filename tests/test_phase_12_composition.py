"""Tests for Phase 12 composition classes: LayerChart and ConcatChart."""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm


@pytest.fixture
def sample_df():
    return pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 1, 5, 3]})


@pytest.fixture
def chart_a(sample_df):
    return fm.Chart(sample_df).mark_point().encode(x="x", y="y")


@pytest.fixture
def chart_b(sample_df):
    return fm.Chart(sample_df).mark_line().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# LayerChart tests
# ---------------------------------------------------------------------------


class TestLayerChart:
    """Tests for LayerChart composition class."""

    def test_to_svg_produces_valid_svg(self, chart_a, chart_b):
        """LayerChart.to_svg() produces SVG without error."""
        layered = fm.LayerChart(chart_a, chart_b)
        svg = layered.to_svg()
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_single_chart(self, chart_a):
        """LayerChart with a single chart works."""
        layered = fm.LayerChart(chart_a)
        svg = layered.to_svg()
        assert "<svg" in svg

    def test_chart_plus_produces_layered_chart(self, chart_a, chart_b):
        """chart_a + chart_b still produces a multi-layer Chart (existing behavior)."""
        result = chart_a + chart_b
        # Existing behavior: __add__ produces a Chart, not a LayerChart
        assert isinstance(result, fm.Chart)
        svg = result.to_svg()
        assert "<svg" in svg

    def test_layer_function(self, chart_a, chart_b):
        """fm.layer() convenience function produces a LayerChart."""
        layered = fm.layer(chart_a, chart_b)
        assert isinstance(layered, fm.LayerChart)
        svg = layered.to_svg()
        assert "<svg" in svg

    def test_theme_propagation(self, chart_a, chart_b):
        """LayerChart.theme() propagates to children."""
        layered = fm.LayerChart(chart_a, chart_b)
        themed = layered.theme(fm.themes.dark)
        assert isinstance(themed, fm.LayerChart)
        svg = themed.to_svg()
        assert "<svg" in svg

    def test_properties_propagation(self, chart_a, chart_b):
        """LayerChart.properties() propagates to children."""
        layered = fm.LayerChart(chart_a, chart_b)
        resized = layered.properties(width=800, height=600)
        assert isinstance(resized, fm.LayerChart)
        svg = resized.to_svg()
        assert "<svg" in svg

    def test_save_svg(self, chart_a, chart_b, tmp_path):
        """LayerChart.save() writes SVG to disk."""
        layered = fm.LayerChart(chart_a, chart_b)
        path = tmp_path / "layered.svg"
        layered.save(str(path))
        assert path.exists()
        content = path.read_text()
        assert content.startswith("<svg")

    def test_repr(self, chart_a, chart_b):
        """LayerChart has a meaningful repr."""
        layered = fm.LayerChart(chart_a, chart_b)
        assert "LayerChart" in repr(layered)
        assert "2 layers" in repr(layered)

    def test_resolve_parameter(self, sample_df):
        """LayerChart accepts resolve= parameter."""
        c1 = fm.Chart(sample_df).mark_point().encode(x="x", y="y")
        c2 = fm.Chart(sample_df).mark_line().encode(x="x", y="y")
        layered = fm.LayerChart(c1, c2, resolve={"color": "independent"})
        svg = layered.to_svg()
        assert "<svg" in svg

    def test_title_parameter(self, chart_a, chart_b):
        """LayerChart accepts title= parameter."""
        layered = fm.LayerChart(chart_a, chart_b, title="My Layer")
        svg = layered.to_svg()
        assert "<svg" in svg

    def test_empty_raises(self):
        """LayerChart with no charts raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            fm.LayerChart()

    def test_invalid_resolve_raises(self, chart_a):
        """LayerChart with invalid resolve value raises ValueError."""
        with pytest.raises(ValueError, match=r"LayerChart: resolve\['x'\] must be one of"):
            fm.LayerChart(chart_a, resolve={"x": "bad"})

    def test_charts_property(self, chart_a, chart_b):
        """LayerChart.charts returns the member charts."""
        layered = fm.LayerChart(chart_a, chart_b)
        assert len(layered.charts) == 2

    def test_repr_svg(self, chart_a, chart_b):
        """LayerChart._repr_svg_ returns SVG for Jupyter."""
        layered = fm.LayerChart(chart_a, chart_b)
        svg = layered._repr_svg_()
        assert "<svg" in svg

    def test_heterogeneous_data(self):
        """LayerChart with different data sources works."""
        df1 = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        df2 = pl.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1]})
        c1 = fm.Chart(df1).mark_point().encode(x="x", y="y")
        c2 = fm.Chart(df2).mark_line().encode(x="x", y="y")
        layered = fm.LayerChart(c1, c2)
        svg = layered.to_svg()
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# ConcatChart tests
# ---------------------------------------------------------------------------


class TestConcatChart:
    """Tests for ConcatChart composition class."""

    def test_single_row(self, chart_a, chart_b):
        """ConcatChart with columns=None renders a single row."""
        grid = fm.ConcatChart(chart_a, chart_b)
        svg = grid.to_svg()
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_two_columns(self, sample_df):
        """ConcatChart with columns=2 renders a 2-column grid."""
        charts = [
            fm.Chart(sample_df).mark_point().encode(x="x", y="y"),
            fm.Chart(sample_df).mark_line().encode(x="x", y="y"),
            fm.Chart(sample_df).mark_bar().encode(x="x", y="y"),
        ]
        grid = fm.ConcatChart(*charts, columns=2)
        svg = grid.to_svg()
        assert "<svg" in svg

    def test_columns_none_single_row(self, sample_df):
        """ConcatChart with columns=None arranges all charts in one row."""
        charts = [
            fm.Chart(sample_df).mark_point().encode(x="x", y="y"),
            fm.Chart(sample_df).mark_line().encode(x="x", y="y"),
            fm.Chart(sample_df).mark_bar().encode(x="x", y="y"),
        ]
        grid = fm.ConcatChart(*charts)
        # columns=None defaults to len(charts) → single row
        assert grid.columns is None
        svg = grid.to_svg()
        assert "<svg" in svg

    def test_theme_propagation(self, chart_a, chart_b):
        """ConcatChart.theme() propagates to children."""
        grid = fm.ConcatChart(chart_a, chart_b, columns=2)
        themed = grid.theme(fm.themes.dark)
        assert isinstance(themed, fm.ConcatChart)
        svg = themed.to_svg()
        assert "<svg" in svg

    def test_properties_propagation(self, chart_a, chart_b):
        """ConcatChart.properties() propagates to children."""
        grid = fm.ConcatChart(chart_a, chart_b)
        resized = grid.properties(width=400, height=300)
        assert isinstance(resized, fm.ConcatChart)
        svg = resized.to_svg()
        assert "<svg" in svg

    def test_save_svg(self, chart_a, chart_b, tmp_path):
        """ConcatChart.save() writes SVG to disk."""
        grid = fm.ConcatChart(chart_a, chart_b, columns=1)
        path = tmp_path / "grid.svg"
        grid.save(str(path))
        assert path.exists()
        content = path.read_text()
        assert content.startswith("<svg")

    def test_repr(self, chart_a, chart_b):
        """ConcatChart has a meaningful repr."""
        grid = fm.ConcatChart(chart_a, chart_b, columns=2)
        assert "ConcatChart" in repr(grid)
        assert "2 charts" in repr(grid)
        assert "columns=2" in repr(grid)

    def test_resolve_parameter(self, sample_df):
        """ConcatChart accepts resolve= for shared scales."""
        charts = [
            fm.Chart(sample_df).mark_point().encode(x="x", y="y"),
            fm.Chart(sample_df).mark_line().encode(x="x", y="y"),
        ]
        grid = fm.ConcatChart(*charts, columns=2, resolve={"x": "shared"})
        svg = grid.to_svg()
        assert "<svg" in svg

    def test_concat_function(self, chart_a, chart_b):
        """fm.concat() convenience function produces a ConcatChart."""
        grid = fm.concat(chart_a, chart_b, columns=2)
        assert isinstance(grid, fm.ConcatChart)
        svg = grid.to_svg()
        assert "<svg" in svg

    def test_empty_raises(self):
        """ConcatChart with no charts raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            fm.ConcatChart()

    def test_invalid_columns_raises(self, chart_a):
        """ConcatChart with columns <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="columns must be > 0"):
            fm.ConcatChart(chart_a, columns=0)

    def test_invalid_resolve_raises(self, chart_a):
        """ConcatChart with invalid resolve value raises ValueError."""
        with pytest.raises(ValueError, match=r"ConcatChart: resolve\['x'\] must be one of"):
            fm.ConcatChart(chart_a, resolve={"x": "wrong"})

    def test_spacing_parameter(self, chart_a, chart_b):
        """ConcatChart respects spacing parameter."""
        grid = fm.ConcatChart(chart_a, chart_b, spacing=20.0)
        assert grid.spacing == 20.0
        svg = grid.to_svg()
        assert "<svg" in svg

    def test_or_operator(self, chart_a, chart_b):
        """ConcatChart supports | operator for further horizontal concat."""
        grid = fm.ConcatChart(chart_a, chart_b, columns=1)
        result = grid | chart_a
        assert isinstance(result, fm.HConcatChart)

    def test_and_operator(self, chart_a, chart_b):
        """ConcatChart supports & operator for further vertical concat."""
        grid = fm.ConcatChart(chart_a, chart_b, columns=2)
        result = grid & chart_a
        assert isinstance(result, fm.VConcatChart)
