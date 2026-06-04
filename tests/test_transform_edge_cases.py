"""Test sweep: transform x edge cases.

Combinatorial test module exercising ferrum transforms against degenerate
inputs: single-row data, all-constant values, all-null/NaN columns, high
cardinality groupby, numeric extremes, and degenerate binary targets.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _renders(chart) -> None:
    """Assert chart produces valid SVG without crashing."""
    svg = chart.to_svg()
    assert "<svg" in svg


def _graceful_failure(chart) -> None:
    """Assert edge-case chart either renders or raises a clean error."""
    try:
        svg = chart.to_svg()
        assert isinstance(svg, str)
    except (ValueError, RuntimeError):
        pass  # Clean error is acceptable for degenerate input.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def df_normal():
    """10-row DataFrame with numeric x/y and a categorical group column."""
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [2.0, 4.0, 1.0, 5.0, 3.0, 7.0, 6.0, 8.0, 9.0, 10.0],
            "g": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        }
    )


@pytest.fixture()
def df_single_row():
    """Single-row DataFrame."""
    return pl.DataFrame({"x": [5.0], "y": [3.0], "g": ["a"]})


@pytest.fixture()
def df_constant():
    """All-identical numeric values."""
    return pl.DataFrame({"x": [5.0] * 10, "y": [3.0] * 10, "g": ["a"] * 10})


@pytest.fixture()
def df_all_null():
    """All-null numeric columns with a categorical grouper."""
    return pl.DataFrame({"x": [None] * 5, "y": [None] * 5, "g": ["a", "a", "b", "b", "b"]}).cast(
        {"x": pl.Float64, "y": pl.Float64}
    )


@pytest.fixture()
def df_all_nan():
    """All-NaN numeric columns."""
    return pl.DataFrame({"x": [float("nan")] * 5, "y": [float("nan")] * 5, "g": ["a"] * 5})


@pytest.fixture()
def df_empty():
    """Zero-row DataFrame with correct schema."""
    return pl.DataFrame(
        {
            "x": pl.Series([], dtype=pl.Float64),
            "y": pl.Series([], dtype=pl.Float64),
            "g": pl.Series([], dtype=pl.Utf8),
        }
    )


# ---------------------------------------------------------------------------
# 1. Single-row input
# ---------------------------------------------------------------------------


class TestSingleRow:
    """Transforms that need 2+ rows should degrade gracefully with 1 row."""

    def test_kde_single_row(self, df_single_row):
        chart = fm.Chart(df_single_row).mark_density().encode(x="x:Q")
        _graceful_failure(chart)

    def test_smooth_single_row(self, df_single_row):
        chart = fm.Chart(df_single_row).mark_smooth().encode(x="x:Q", y="y:Q")
        _graceful_failure(chart)

    def test_boxplot_single_row(self, df_single_row):
        chart = fm.Chart(df_single_row).mark_boxplot().encode(x="g:N", y="y:Q")
        _graceful_failure(chart)

    def test_violin_single_row(self, df_single_row):
        chart = fm.Chart(df_single_row).mark_violin().encode(x="g:N", y="y:Q")
        _graceful_failure(chart)

    def test_qq_single_row(self, df_single_row):
        chart = fm.Chart(df_single_row).mark_qq().encode(x="x:Q")
        _graceful_failure(chart)

    def test_smooth_two_rows(self):
        df = pl.DataFrame({"x": [1.0, 10.0], "y": [1.0, 10.0]})
        chart = fm.Chart(df).mark_smooth().encode(x="x:Q", y="y:Q")
        _renders(chart)

    def test_smooth_loess_two_rows(self):
        df = pl.DataFrame({"x": [1.0, 10.0], "y": [1.0, 10.0]})
        chart = (
            fm.Chart(df)
            .mark_smooth()
            .encode(x="x:Q", y="y:Q")
            .transform(fm.Smooth(x="x", y="y", method="loess"))
        )
        _graceful_failure(chart)

    def test_robust_two_rows(self):
        df = pl.DataFrame({"x": [1.0, 10.0], "y": [1.0, 10.0]})
        chart = (
            fm.Chart(df).mark_smooth().encode(x="x:Q", y="y:Q").transform(fm.Robust(x="x", y="y"))
        )
        _graceful_failure(chart)


# ---------------------------------------------------------------------------
# 2. All-identical values (zero variance)
# ---------------------------------------------------------------------------


class TestConstantValues:
    """Transforms should handle zero-variance input gracefully."""

    def test_kde_constant(self, df_constant):
        chart = fm.Chart(df_constant).mark_density().encode(x="x:Q")
        _graceful_failure(chart)

    def test_smooth_constant_x(self, df_constant):
        chart = fm.Chart(df_constant).mark_smooth().encode(x="x:Q", y="y:Q")
        _graceful_failure(chart)

    def test_smooth_constant_y(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [3.0] * 5})
        chart = fm.Chart(df).mark_smooth().encode(x="x:Q", y="y:Q")
        _graceful_failure(chart)

    def test_boxplot_constant(self, df_constant):
        chart = fm.Chart(df_constant).mark_boxplot().encode(x="g:N", y="y:Q")
        _graceful_failure(chart)

    def test_histogram_constant(self, df_constant):
        chart = fm.Chart(df_constant).mark_histogram().encode(x="x:Q")
        _graceful_failure(chart)

    def test_violin_constant(self, df_constant):
        chart = fm.Chart(df_constant).mark_violin().encode(x="g:N", y="y:Q")
        _graceful_failure(chart)

    def test_glm_constant_x(self, df_constant):
        """GLM with zero-variance x should raise cleanly."""
        chart = (
            fm.Chart(df_constant)
            .mark_smooth()
            .encode(x="x:Q", y="y:Q")
            .transform(fm.Glm(x="x", y="y", family="gaussian"))
        )
        with pytest.raises(ValueError, match="singular|zero variance"):
            chart.to_svg()


# ---------------------------------------------------------------------------
# 3. All-null / all-NaN input
# ---------------------------------------------------------------------------


class TestAllNullNaN:
    """Transforms should produce empty output or raise cleanly for null data."""

    def test_histogram_all_null(self, df_all_null):
        chart = fm.Chart(df_all_null).mark_histogram().encode(x="x:Q")
        _graceful_failure(chart)

    def test_kde_all_nan(self, df_all_nan):
        chart = fm.Chart(df_all_nan).mark_density().encode(x="x:Q")
        _graceful_failure(chart)

    def test_smooth_all_nan(self, df_all_nan):
        chart = fm.Chart(df_all_nan).mark_smooth().encode(x="x:Q", y="y:Q")
        _graceful_failure(chart)

    def test_aggregate_all_null(self, df_all_null):
        chart = (
            fm.Chart(df_all_null)
            .mark_bar()
            .encode(x="g:N", y="mean_y:Q")
            .transform(
                fm.Aggregate(
                    ops=[fm.AggregateOp("y", "mean", "mean_y")],
                    groupby=["g"],
                )
            )
        )
        _graceful_failure(chart)

    def test_violin_all_null(self, df_all_null):
        chart = fm.Chart(df_all_null).mark_violin().encode(x="g:N", y="y:Q")
        _graceful_failure(chart)

    def test_glm_all_nan(self, df_all_nan):
        """GLM with all-NaN y should raise cleanly."""
        chart = (
            fm.Chart(df_all_nan)
            .mark_smooth()
            .encode(x="x:Q", y="y:Q")
            .transform(fm.Glm(x="x", y="y", family="gaussian"))
        )
        with pytest.raises(ValueError, match="non-null observations"):
            chart.to_svg()


# ---------------------------------------------------------------------------
# 4. Empty DataFrame
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Zero-row DataFrames should render empty charts or degrade cleanly."""

    def test_histogram_empty(self, df_empty):
        chart = fm.Chart(df_empty).mark_histogram().encode(x="x:Q")
        _graceful_failure(chart)

    def test_density_empty(self, df_empty):
        chart = fm.Chart(df_empty).mark_density().encode(x="x:Q")
        _graceful_failure(chart)

    def test_smooth_empty(self, df_empty):
        chart = fm.Chart(df_empty).mark_smooth().encode(x="x:Q", y="y:Q")
        _graceful_failure(chart)

    def test_boxplot_empty(self, df_empty):
        chart = fm.Chart(df_empty).mark_boxplot().encode(x="g:N", y="y:Q")
        _graceful_failure(chart)


# ---------------------------------------------------------------------------
# 5. Large groupby cardinality
# ---------------------------------------------------------------------------


class TestHighCardinality:
    """Stress-test groupby paths with many unique groups."""

    def test_aggregate_100_groups(self):
        df = pl.DataFrame({"x": [f"g{i}" for i in range(100)], "y": [float(i) for i in range(100)]})
        chart = (
            fm.Chart(df)
            .mark_bar()
            .encode(x="x:N", y="mean_y:Q")
            .transform(
                fm.Aggregate(
                    ops=[fm.AggregateOp("y", "mean", "mean_y")],
                    groupby=["x"],
                )
            )
        )
        _renders(chart)

    def test_boxplot_many_groups(self):
        groups = [f"g{i}" for i in range(50)]
        df = pl.DataFrame(
            {
                "x": groups * 10,
                "y": [float(i % 20) for i in range(500)],
            }
        )
        chart = fm.Chart(df).mark_boxplot().encode(x="x:N", y="y:Q")
        _renders(chart)


# ---------------------------------------------------------------------------
# 6. Binary data edge cases (Logistic transform)
# ---------------------------------------------------------------------------


class TestLogistic:
    """Logistic regression with degenerate binary targets."""

    def test_logistic_all_zero(self):
        """All y=0 should raise (no separation possible)."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.0] * 5})
        chart = (
            fm.Chart(df).mark_smooth().encode(x="x:Q", y="y:Q").transform(fm.Logistic(x="x", y="y"))
        )
        with pytest.raises(ValueError, match="singular"):
            chart.to_svg()

    def test_logistic_all_one(self):
        """All y=1 should raise (no separation possible)."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0] * 5})
        chart = (
            fm.Chart(df).mark_smooth().encode(x="x:Q", y="y:Q").transform(fm.Logistic(x="x", y="y"))
        )
        with pytest.raises(ValueError, match="singular"):
            chart.to_svg()

    def test_logistic_single_unique_x(self):
        """Constant x with mixed y should raise (singular design)."""
        df = pl.DataFrame({"x": [5.0] * 10, "y": [0.0] * 5 + [1.0] * 5})
        chart = (
            fm.Chart(df).mark_smooth().encode(x="x:Q", y="y:Q").transform(fm.Logistic(x="x", y="y"))
        )
        with pytest.raises(ValueError, match="singular|zero variance"):
            chart.to_svg()

    def test_logistic_valid_overlapping(self):
        """Non-degenerate binary target with overlap should render."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "y": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            }
        )
        chart = (
            fm.Chart(df).mark_smooth().encode(x="x:Q", y="y:Q").transform(fm.Logistic(x="x", y="y"))
        )
        _renders(chart)


# ---------------------------------------------------------------------------
# 7. Specific Bin edge cases
# ---------------------------------------------------------------------------


class TestBinEdgeCases:
    """Histogram/Bin boundary conditions."""

    def test_bin_count_one(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        chart = fm.Chart(df).mark_histogram(bin_count=1).encode(x="x:Q")
        _renders(chart)

    def test_bin_count_exceeds_rows(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        chart = fm.Chart(df).mark_histogram(bin_count=100).encode(x="x:Q")
        _renders(chart)

    def test_bin_large_values(self):
        df = pl.DataFrame({"x": [1e15, 2e15, 3e15, 4e15, 5e15]})
        chart = fm.Chart(df).mark_histogram().encode(x="x:Q")
        _renders(chart)

    def test_bin_tiny_values(self):
        df = pl.DataFrame({"x": [1e-15, 2e-15, 3e-15, 4e-15, 5e-15]})
        chart = fm.Chart(df).mark_histogram().encode(x="x:Q")
        _renders(chart)


# ---------------------------------------------------------------------------
# 8. Hex edge cases
# ---------------------------------------------------------------------------


class TestHexEdgeCases:
    """Hex transform with minimal or degenerate input."""

    def test_hex_single_point(self):
        df = pl.DataFrame({"x": [3.0], "y": [4.0]})
        chart = fm.Chart(df).mark_hex().encode(x="x:Q", y="y:Q")
        _renders(chart)

    def test_hex_collinear(self):
        """All points on a line — zero y-spread."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.0] * 5})
        chart = fm.Chart(df).mark_hex().encode(x="x:Q", y="y:Q")
        _graceful_failure(chart)


# ---------------------------------------------------------------------------
# 9. Aggregate edge cases
# ---------------------------------------------------------------------------


class TestAggregateEdgeCases:
    """Aggregate transform boundary conditions."""

    def test_aggregate_count_on_utf8(self):
        """Count op on a string column — regression check."""
        df = pl.DataFrame({"category": ["a", "b", "a", "c", "b", "a"]})
        chart = (
            fm.Chart(df)
            .mark_bar()
            .encode(x="category:N", y="n:Q")
            .transform(
                fm.Aggregate(
                    ops=[fm.AggregateOp("category", "count", "n")],
                    groupby=["category"],
                )
            )
        )
        _renders(chart)

    def test_aggregate_multiple_ops(self):
        df = pl.DataFrame({"x": ["a", "a", "b", "b"], "y": [1.0, 2.0, 3.0, 4.0]})
        chart = (
            fm.Chart(df)
            .mark_bar()
            .encode(x="x:N", y="mean_y:Q")
            .transform(
                fm.Aggregate(
                    ops=[
                        fm.AggregateOp("y", "mean", "mean_y"),
                        fm.AggregateOp("y", "sum", "sum_y"),
                        fm.AggregateOp("y", "min", "min_y"),
                        fm.AggregateOp("y", "max", "max_y"),
                        fm.AggregateOp("y", "count", "count_y"),
                        fm.AggregateOp("y", "median", "med_y"),
                    ],
                    groupby=["x"],
                )
            )
        )
        _renders(chart)


# ---------------------------------------------------------------------------
# 10. Contour (Kde2D) edge cases
# ---------------------------------------------------------------------------


class TestContourEdgeCases:
    """2D KDE / contour with degenerate input."""

    def test_contour_single_point(self):
        df = pl.DataFrame({"x": [3.0], "y": [4.0]})
        chart = fm.Chart(df).mark_contour().encode(x="x:Q", y="y:Q")
        _graceful_failure(chart)

    def test_contour_collinear(self):
        """All points along x-axis (zero y-spread)."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.0] * 5})
        chart = fm.Chart(df).mark_contour().encode(x="x:Q", y="y:Q")
        _graceful_failure(chart)


# ---------------------------------------------------------------------------
# 11. Miscellaneous transform robustness
# ---------------------------------------------------------------------------


class TestMiscTransformRobustness:
    """Additional transform edge cases not covered above."""

    def test_boxen_small_group(self):
        """LetterValue (boxen) with only 3 rows per group."""
        df = pl.DataFrame({"x": ["a"] * 3, "y": [1.0, 2.0, 3.0]})
        chart = fm.Chart(df).mark_boxen().encode(x="x:N", y="y:Q")
        _renders(chart)

    def test_boxen_single_row_per_group(self):
        df = pl.DataFrame({"x": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})
        chart = fm.Chart(df).mark_boxen().encode(x="x:N", y="y:Q")
        _renders(chart)

    def test_swarm_single_per_group(self):
        df = pl.DataFrame({"x": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})
        chart = fm.Chart(df).mark_swarm().encode(x="x:N", y="y:Q")
        _renders(chart)

    def test_errorbar_single_per_group(self):
        """Summary transform with only 1 observation per group."""
        df = pl.DataFrame({"x": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})
        chart = fm.Chart(df).mark_errorbar().encode(x="x:N", y="y:Q")
        _graceful_failure(chart)

    def test_kde_with_inf(self):
        """Infinite values mixed with finite should not crash."""
        df = pl.DataFrame({"x": [float("inf"), float("-inf"), 1.0, 2.0, 3.0]})
        chart = fm.Chart(df).mark_density().encode(x="x:Q")
        _graceful_failure(chart)

    def test_qq_two_rows(self):
        df = pl.DataFrame({"x": [1.0, 2.0]})
        chart = fm.Chart(df).mark_qq().encode(x="x:Q")
        _renders(chart)
