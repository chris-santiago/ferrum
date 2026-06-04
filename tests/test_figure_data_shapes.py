"""Test sweep: figure function x data shapes.

Tests figure functions with edge-case data shapes that might surface bugs:
tiny DataFrames (2 rows), single unique value columns, all-constant numeric
columns, zero-row DataFrames, and wide-format inputs with many columns.

Run with: uv run pytest tests/test_figure_data_shapes.py --noconftest -v
"""

from __future__ import annotations

import random

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_df():
    """2-row DataFrame -- minimal viable input."""
    return pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "cat": ["a", "b"]})


@pytest.fixture
def constant_df():
    """All values identical in numeric columns (zero variance)."""
    return pl.DataFrame({"x": [5.0] * 10, "y": [5.0] * 10, "cat": ["a"] * 5 + ["b"] * 5})


@pytest.fixture
def single_group_df():
    """Single unique value in the categorical grouping column."""
    random.seed(99)
    return pl.DataFrame(
        {
            "x": [random.gauss(0, 1) for _ in range(15)],
            "y": [random.gauss(0, 1) for _ in range(15)],
            "cat": ["only"] * 15,
        }
    )


@pytest.fixture
def zero_row_df():
    """Empty DataFrame with correct schema."""
    return pl.DataFrame(
        {
            "x": pl.Series([], dtype=pl.Float64),
            "y": pl.Series([], dtype=pl.Float64),
            "cat": pl.Series([], dtype=pl.Utf8),
        }
    )


@pytest.fixture
def wide_df():
    """Wide-format with 10 numeric columns for pairplot/heatmap."""
    random.seed(42)
    data: dict[str, list] = {f"v{i}": [random.gauss(0, 1) for _ in range(20)] for i in range(10)}
    data["label"] = ["a"] * 10 + ["b"] * 10
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _produces_svg(chart) -> None:
    """Assert chart renders to valid SVG without crashing."""
    svg = chart.to_svg()
    assert "<svg" in svg, "Expected SVG output containing '<svg' tag"


def _empty_input_safe(fn, *args, **kwargs) -> None:
    """Assert zero-row input either produces SVG or raises clean ValueError."""
    try:
        chart = fn(*args, **kwargs)
        svg = chart.to_svg()
        # Either empty SVG or valid SVG is acceptable
        assert isinstance(svg, str)
    except (ValueError, RuntimeError):
        # Clean error for empty data is acceptable
        pass


# ===========================================================================
# Distribution functions
# ===========================================================================


class TestDisplot:
    """displot x data shapes."""

    def test_tiny(self, tiny_df):
        _produces_svg(fm.displot(tiny_df, x="x"))

    def test_constant_values(self, constant_df):
        _produces_svg(fm.displot(constant_df, x="x"))

    def test_single_group_hue(self, single_group_df):
        _produces_svg(fm.displot(single_group_df, x="x", hue="cat"))

    def test_zero_rows(self, zero_row_df):
        _empty_input_safe(fm.displot, zero_row_df, x="x")

    def test_kde_tiny(self, tiny_df):
        _produces_svg(fm.displot(tiny_df, x="x", kind="kde"))

    def test_ecdf_constant(self, constant_df):
        _produces_svg(fm.displot(constant_df, x="x", kind="ecdf"))


class TestCatplot:
    """catplot x data shapes."""

    def test_tiny_strip(self, tiny_df):
        _produces_svg(fm.catplot(tiny_df, x="cat", y="x"))

    def test_constant_box(self, constant_df):
        _produces_svg(fm.catplot(constant_df, x="cat", y="x", kind="box"))

    def test_single_group_violin(self, single_group_df):
        _produces_svg(fm.catplot(single_group_df, x="cat", y="x", kind="violin"))

    def test_zero_rows(self, zero_row_df):
        _empty_input_safe(fm.catplot, zero_row_df, x="cat", y="x")

    def test_tiny_count(self, tiny_df):
        _produces_svg(fm.catplot(tiny_df, x="cat", kind="count"))


class TestRelplot:
    """relplot x data shapes."""

    def test_tiny(self, tiny_df):
        _produces_svg(fm.relplot(tiny_df, x="x", y="y"))

    def test_constant_values(self, constant_df):
        _produces_svg(fm.relplot(constant_df, x="x", y="y"))

    def test_single_group_hue(self, single_group_df):
        _produces_svg(fm.relplot(single_group_df, x="x", y="y", hue="cat"))

    def test_zero_rows(self, zero_row_df):
        _empty_input_safe(fm.relplot, zero_row_df, x="x", y="y")

    def test_line_kind_tiny(self, tiny_df):
        _produces_svg(fm.relplot(tiny_df, x="x", y="y", kind="line"))


# ===========================================================================
# Regression functions
# ===========================================================================


class TestLmplot:
    """lmplot x data shapes."""

    def test_tiny(self, tiny_df):
        _produces_svg(fm.lmplot(tiny_df, x="x", y="y"))

    def test_constant_x(self, constant_df):
        _produces_svg(fm.lmplot(constant_df, x="x", y="y"))

    def test_single_group_hue(self, single_group_df):
        _produces_svg(fm.lmplot(single_group_df, x="x", y="y", hue="cat"))

    def test_zero_rows(self, zero_row_df):
        _empty_input_safe(fm.lmplot, zero_row_df, x="x", y="y")


class TestResidplot:
    """residplot x data shapes."""

    def test_tiny(self, tiny_df):
        _produces_svg(fm.residplot(tiny_df, x="x", y="y"))

    def test_constant_values(self, constant_df):
        _produces_svg(fm.residplot(constant_df, x="x", y="y"))

    def test_zero_rows(self, zero_row_df):
        _empty_input_safe(fm.residplot, zero_row_df, x="x", y="y")

    def test_lowess_tiny(self, tiny_df):
        _produces_svg(fm.residplot(tiny_df, x="x", y="y", lowess=True))


# ===========================================================================
# Matrix functions
# ===========================================================================


class TestPairplot:
    """pairplot x data shapes."""

    def test_tiny(self, tiny_df):
        _produces_svg(fm.pairplot(tiny_df))

    def test_constant_values(self, constant_df):
        _produces_svg(fm.pairplot(constant_df))

    def test_wide_many_columns(self, wide_df):
        _produces_svg(fm.pairplot(wide_df, vars=["v0", "v1", "v2", "v3"]))

    def test_single_group_hue(self, single_group_df):
        _produces_svg(fm.pairplot(single_group_df, hue="cat"))

    def test_zero_rows(self, zero_row_df):
        _empty_input_safe(fm.pairplot, zero_row_df)


class TestHeatmap:
    """heatmap x data shapes."""

    def test_wide_normal(self, wide_df):
        _produces_svg(fm.heatmap(wide_df.drop("label")))

    def test_wide_with_label(self, wide_df):
        _produces_svg(fm.heatmap(wide_df))

    def test_constant_values(self):
        df = pl.DataFrame({"a": [1.0] * 3, "b": [1.0] * 3, "c": [1.0] * 3})
        _produces_svg(fm.heatmap(df))

    def test_single_row(self):
        df = pl.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        _produces_svg(fm.heatmap(df))

    def test_zero_rows(self):
        df = pl.DataFrame(
            {
                "a": pl.Series([], dtype=pl.Float64),
                "b": pl.Series([], dtype=pl.Float64),
            }
        )
        _empty_input_safe(fm.heatmap, df)


class TestJointplot:
    """jointplot x data shapes."""

    def test_tiny(self, tiny_df):
        _produces_svg(fm.jointplot(tiny_df, x="x", y="y"))

    def test_constant_values(self, constant_df):
        _produces_svg(fm.jointplot(constant_df, x="x", y="y"))

    def test_single_group_hue(self, single_group_df):
        _produces_svg(fm.jointplot(single_group_df, x="x", y="y", hue="cat"))

    def test_zero_rows(self, zero_row_df):
        _empty_input_safe(fm.jointplot, zero_row_df, x="x", y="y")


# ===========================================================================
# Classification functions (array-accepting, no model required)
# ===========================================================================


class TestConfusionMatrixChart:
    """confusion_matrix_chart with precomputed y_true/y_pred arrays."""

    def test_tiny(self):
        _produces_svg(fm.confusion_matrix_chart(y_true=["a", "b"], y_pred=["a", "b"]))

    def test_single_class(self):
        _produces_svg(fm.confusion_matrix_chart(y_true=["a", "a", "a"], y_pred=["a", "a", "a"]))

    def test_many_classes(self):
        classes = list("abcdefgh")
        y_true = classes * 3
        y_pred = (classes[1:] + [classes[0]]) * 3
        _produces_svg(fm.confusion_matrix_chart(y_true=y_true, y_pred=y_pred))

    def test_empty(self):
        _empty_input_safe(fm.confusion_matrix_chart, y_true=[], y_pred=[])


class TestClassBalanceChart:
    """class_balance_chart with raw label arrays."""

    def test_tiny(self):
        _produces_svg(fm.class_balance_chart(y=["a", "b"]))

    def test_single_class(self):
        _produces_svg(fm.class_balance_chart(y=["x"] * 10))

    def test_many_classes(self):
        _produces_svg(fm.class_balance_chart(y=list("abcdefghij") * 5))

    def test_empty(self):
        _empty_input_safe(fm.class_balance_chart, y=[])


# ===========================================================================
# Ranking functions (DataFrame-accepting, no model required)
# ===========================================================================


class TestRankChart:
    """rank_chart with raw DataFrames (no model)."""

    def test_2d_wide(self, wide_df):
        _produces_svg(fm.rank_chart(wide_df.drop("label"), rank="2d"))

    def test_1d_wide(self, wide_df):
        _produces_svg(fm.rank_chart(wide_df.drop("label"), rank="1d"))

    def test_2d_constant(self):
        df = pl.DataFrame({"a": [1.0] * 10, "b": [2.0] * 10, "c": [3.0] * 10})
        _produces_svg(fm.rank_chart(df, rank="2d"))

    def test_2d_tiny(self, tiny_df):
        _produces_svg(fm.rank_chart(tiny_df[["x", "y"]], rank="2d"))
