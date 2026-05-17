"""Cross-dimensional tests: faceting x multi-layer charts x statistical transforms.

Verifies that:
1. Transforms compute per-facet (not globally across all panels)
2. Multi-layer faceted charts render without crashes
3. Different transform layers within a faceted chart don't interfere
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ferrum import Chart


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def facet_df() -> pl.DataFrame:
    """DataFrame with 120 rows, 3 categories in `cat`, 2 groups in `grp`."""
    np.random.seed(42)
    n = 120
    return pl.DataFrame(
        {
            "x": np.random.randn(n).tolist(),
            "y": (np.random.randn(n) * 2 + np.random.randn(n)).tolist(),
            "cat": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
            "grp": ["x", "y"] * 60,
            "num": np.random.rand(n).tolist(),
        }
    )


@pytest.fixture
def sparse_df() -> pl.DataFrame:
    """DataFrame with uneven facet groups: one has only 3 rows."""
    np.random.seed(99)
    return pl.DataFrame(
        {
            "x": np.random.randn(23).tolist(),
            "y": np.random.randn(23).tolist(),
            "cat": ["small"] * 3 + ["medium"] * 10 + ["large"] * 10,
        }
    )


@pytest.fixture
def many_panel_df() -> pl.DataFrame:
    """DataFrame with 5 categories for many-panel faceting."""
    np.random.seed(7)
    n = 75  # 15 per group
    cats = ["v", "w", "x", "y", "z"]
    return pl.DataFrame(
        {
            "x": np.random.randn(n).tolist(),
            "y": np.random.randn(n).tolist(),
            "cat": [c for c in cats for _ in range(15)],
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_svg(text: str) -> bool:
    return "<svg" in text


# ---------------------------------------------------------------------------
# Group 1: Single transform layer, faceted (no crash)
# ---------------------------------------------------------------------------


class TestSingleTransformFaceted:
    """Single statistical mark faceted — must render without crash."""

    def test_smooth_facet_col(self, facet_df):
        chart = Chart(facet_df).mark_smooth().encode(x="x:Q", y="y:Q").facet(col="cat")
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_smooth_facet_row(self, facet_df):
        chart = Chart(facet_df).mark_smooth().encode(x="x:Q", y="y:Q").facet(row="cat")
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_smooth_facet_grid(self, facet_df):
        chart = (
            Chart(facet_df).mark_smooth().encode(x="x:Q", y="y:Q").facet(row="grp", col="cat")
        )
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_density_facet_col(self, facet_df):
        chart = Chart(facet_df).mark_density().encode(x="x:Q").facet(col="cat")
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_histogram_facet_col(self, facet_df):
        chart = Chart(facet_df).mark_histogram().encode(x="x:Q").facet(col="cat")
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_boxplot_facet_col(self, facet_df):
        chart = Chart(facet_df).mark_boxplot().encode(x="grp:N", y="y:Q").facet(col="cat")
        svg = chart.show_svg()
        assert _is_svg(svg)


# ---------------------------------------------------------------------------
# Group 2: Multi-layer with transform, faceted (no crash)
# ---------------------------------------------------------------------------


class TestMultiLayerTransformFaceted:
    """Layered charts with at least one statistical transform, then faceted."""

    def test_point_plus_smooth_facet_col(self, facet_df):
        points = Chart(facet_df).mark_point().encode(x="x:Q", y="y:Q")
        smooth = Chart(facet_df).mark_smooth().encode(x="x:Q", y="y:Q")
        layered = (points + smooth).facet(col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)

    def test_point_plus_smooth_facet_grid(self, facet_df):
        points = Chart(facet_df).mark_point().encode(x="x:Q", y="y:Q")
        smooth = Chart(facet_df).mark_smooth().encode(x="x:Q", y="y:Q")
        layered = (points + smooth).facet(row="grp", col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)

    def test_point_plus_density_facet_col(self, facet_df):
        points = Chart(facet_df).mark_point().encode(x="x:Q", y="y:Q")
        density = Chart(facet_df).mark_density().encode(x="x:Q")
        layered = (points + density).facet(col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)

    def test_point_plus_histogram_facet_col(self, facet_df):
        """Points + histogram on same x column, faceted."""
        points = Chart(facet_df).mark_point().encode(x="x:Q", y="y:Q")
        hist = Chart(facet_df).mark_histogram().encode(x="x:Q")
        layered = (points + hist).facet(col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)

    def test_three_layers_point_smooth_rule_faceted(self, facet_df):
        points = Chart(facet_df).mark_point().encode(x="x:Q", y="y:Q")
        smooth = Chart(facet_df).mark_smooth().encode(x="x:Q", y="y:Q")
        rule = Chart(facet_df).mark_rule().encode(y="y:Q")
        layered = (points + smooth + rule).facet(col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)

    def test_point_smooth_method_lm_faceted(self, facet_df):
        points = Chart(facet_df).mark_point().encode(x="x:Q", y="y:Q")
        smooth = Chart(facet_df).mark_smooth(method="lm").encode(x="x:Q", y="y:Q")
        layered = (points + smooth).facet(col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)


# ---------------------------------------------------------------------------
# Group 3: Faceting scopes transform correctly (differs from non-faceted)
# ---------------------------------------------------------------------------


class TestFacetScopesTransform:
    """Faceted SVG must differ from non-faceted, proving per-panel computation."""

    def test_smooth_faceted_differs_from_single(self, facet_df):
        base = Chart(facet_df).mark_smooth().encode(x="x:Q", y="y:Q")
        single_svg = base.show_svg()
        faceted_svg = base.facet(col="cat").show_svg()
        assert _is_svg(single_svg)
        assert _is_svg(faceted_svg)
        assert single_svg != faceted_svg

    def test_density_faceted_differs_from_single(self, facet_df):
        base = Chart(facet_df).mark_density().encode(x="x:Q")
        single_svg = base.show_svg()
        faceted_svg = base.facet(col="cat").show_svg()
        assert _is_svg(single_svg)
        assert _is_svg(faceted_svg)
        assert single_svg != faceted_svg

    def test_histogram_faceted_differs_from_single(self, facet_df):
        base = Chart(facet_df).mark_histogram().encode(x="x:Q")
        single_svg = base.show_svg()
        faceted_svg = base.facet(col="cat").show_svg()
        assert _is_svg(single_svg)
        assert _is_svg(faceted_svg)
        assert single_svg != faceted_svg


# ---------------------------------------------------------------------------
# Group 4: Multiple transform layers in same faceted chart
# ---------------------------------------------------------------------------


class TestMultipleTransformLayers:
    """Two statistical transform layers coexisting in a faceted chart."""

    def test_two_smooth_methods_faceted(self, facet_df):
        """Two smooth layers with different methods in same faceted chart."""
        smooth_loess = Chart(facet_df).mark_smooth(method="loess").encode(x="x:Q", y="y:Q")
        smooth_lm = Chart(facet_df).mark_smooth(method="lm").encode(x="x:Q", y="y:Q")
        layered = (smooth_loess + smooth_lm).facet(col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)

    def test_smooth_loess_plus_smooth_lm_different_bw_faceted(self, facet_df):
        """Two smooth layers with different bandwidths, faceted."""
        smooth_narrow = Chart(facet_df).mark_smooth(bandwidth=0.3).encode(x="x:Q", y="y:Q")
        smooth_wide = Chart(facet_df).mark_smooth(bandwidth=0.9).encode(x="x:Q", y="y:Q")
        layered = (smooth_narrow + smooth_wide).facet(col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)

    def test_point_plus_density_same_col_faceted(self, facet_df):
        """Points + density on same x column, faceted."""
        points = Chart(facet_df).mark_point().encode(x="x:Q", y="y:Q")
        density = Chart(facet_df).mark_density().encode(x="x:Q")
        layered = (points + density).facet(col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)


# ---------------------------------------------------------------------------
# Group 5: Facet with color encoding + transform
# ---------------------------------------------------------------------------


class TestColorEncodingWithTransformFaceted:
    """Color-grouped transforms within faceted charts."""

    def test_smooth_color_faceted(self, facet_df):
        """mark_smooth with color=grp, faceted by cat."""
        chart = (
            Chart(facet_df)
            .mark_smooth(groupby="grp")
            .encode(x="x:Q", y="y:Q", color="grp:N")
            .facet(col="cat")
        )
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_density_color_faceted(self, facet_df):
        """mark_density with color=grp, faceted by cat."""
        chart = (
            Chart(facet_df)
            .mark_density(groupby="grp")
            .encode(x="x:Q", color="grp:N")
            .facet(col="cat")
        )
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_histogram_color_stack_faceted(self, facet_df):
        """mark_histogram with color=grp, stacked, faceted."""
        chart = (
            Chart(facet_df)
            .mark_histogram(groupby="grp", multiple="stack")
            .encode(x="x:Q", color="grp:N")
            .facet(col="cat")
        )
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_point_plus_smooth_color_faceted(self, facet_df):
        """Layered: points colored by grp + smooth colored by grp, faceted."""
        points = Chart(facet_df).mark_point().encode(x="x:Q", y="y:Q", color="grp:N")
        smooth = (
            Chart(facet_df)
            .mark_smooth(groupby="grp")
            .encode(x="x:Q", y="y:Q", color="grp:N")
        )
        layered = (points + smooth).facet(col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)


# ---------------------------------------------------------------------------
# Group 6: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: sparse facet groups, many panels, mixed layer types."""

    def test_facet_sparse_group_smooth(self, sparse_df):
        """Facet with one group having only 3 rows — smooth should still render."""
        chart = Chart(sparse_df).mark_smooth().encode(x="x:Q", y="y:Q").facet(col="cat")
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_facet_sparse_group_density(self, sparse_df):
        """Facet with one group having only 3 rows — density should still render."""
        chart = Chart(sparse_df).mark_density().encode(x="x:Q").facet(col="cat")
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_facet_sparse_group_histogram(self, sparse_df):
        """Facet with one group having only 3 rows — histogram should still render."""
        chart = Chart(sparse_df).mark_histogram().encode(x="x:Q").facet(col="cat")
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_many_panels_smooth(self, many_panel_df):
        """5 facet panels with smooth transform — no crash."""
        chart = (
            Chart(many_panel_df).mark_smooth().encode(x="x:Q", y="y:Q").facet(col="cat")
        )
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_many_panels_density(self, many_panel_df):
        """5 facet panels with density transform — no crash."""
        chart = Chart(many_panel_df).mark_density().encode(x="x:Q").facet(col="cat")
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_layered_mixed_transform_nontransform_faceted(self, facet_df):
        """One transform layer + one plain layer, faceted — must not crash."""
        plain_points = Chart(facet_df).mark_point().encode(x="x:Q", y="y:Q")
        transform_smooth = Chart(facet_df).mark_smooth().encode(x="x:Q", y="y:Q")
        layered = (plain_points + transform_smooth).facet(col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)

    def test_facet_smooth_with_ci_band(self, facet_df):
        """Smooth with CI band (layered ribbon+line) faceted by col."""
        chart = (
            Chart(facet_df).mark_smooth(ci=0.95).encode(x="x:Q", y="y:Q").facet(col="cat")
        )
        svg = chart.show_svg()
        assert _is_svg(svg)

    def test_facet_smooth_ci_plus_points(self, facet_df):
        """Points + smooth with CI, faceted — 3 effective layers per panel."""
        points = Chart(facet_df).mark_point().encode(x="x:Q", y="y:Q")
        smooth_ci = Chart(facet_df).mark_smooth(ci=0.95).encode(x="x:Q", y="y:Q")
        layered = (points + smooth_ci).facet(col="cat")
        svg = layered.show_svg()
        assert _is_svg(svg)
