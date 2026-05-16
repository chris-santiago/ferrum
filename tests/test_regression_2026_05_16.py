"""Regression tests for 2026-05-16 bug fixes.

Bug 1: Scale constructors required `range` (now optional).
    The `range` parameter on LinearScale, LogScale, SymlogScale, TimeScale, and
    OrdinalScale was required. Users setting axis limits via `domain=` had to
    also pass `range=` with a pixel value. Now `range` is optional -- when
    omitted, the renderer auto-fills from the plot-area dimensions.

Bug 2: mark_rule renders incorrectly when layered with overlapping column names.
    When `Chart(df_scatter) + Chart(df_rule)` where both DataFrames have a "y"
    column, `pl.concat(..., how="diagonal")` merged them into one column, causing
    the rule layer to see all scatter rows and draw a line per row instead of just
    one line. Fix: `__add__` now renames overlapping RHS columns before diagonal
    concat, and updates RHS layer encodings to reference the renamed columns.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import X, Y


# ---------------------------------------------------------------------------
# Bug 1: Scale constructors -- range is now optional
# ---------------------------------------------------------------------------


class TestScaleRangeOptional:
    """Verify that scale constructors accept domain-only (no range)."""

    def test_linear_scale_domain_only(self):
        """LinearScale(domain=[0, 10]) should have range=None and render."""
        scale = fm.LinearScale(domain=[0, 10])
        assert scale.range is None

        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]})
        chart = fm.Chart(df).mark_point().encode(
            x=X("x", scale=scale),
            y="y",
        )
        svg = chart.show_svg()
        assert svg.strip().startswith("<svg")
        assert len(svg) > 100

    def test_log_scale_domain_only(self):
        """LogScale(domain=[1, 1000], base=10) should have range=None and render."""
        scale = fm.LogScale(domain=[1, 1000], base=10)
        assert scale.range is None

        df = pl.DataFrame({"x": [1.0, 10.0, 100.0, 1000.0], "y": [1.0, 2.0, 3.0, 4.0]})
        chart = fm.Chart(df).mark_point().encode(
            x=X("x", scale=scale),
            y="y",
        )
        svg = chart.show_svg()
        assert svg.strip().startswith("<svg")
        assert len(svg) > 100

    def test_scale_range_explicit_preserved(self):
        """LinearScale(domain=[0, 10], range=[0, 500]) preserves both values."""
        scale = fm.LinearScale(domain=[0, 10], range=[0, 500])
        assert scale.range == [0.0, 500.0]
        assert scale.domain == [0.0, 10.0]


# ---------------------------------------------------------------------------
# Bug 2: mark_rule layering with overlapping column names
# ---------------------------------------------------------------------------


class TestRuleLayerColumnRename:
    """Verify that layering with overlapping column names renames RHS columns."""

    @pytest.fixture()
    def scatter_and_rule(self):
        """50-point scatter + 1-row rule, both with a 'y' column."""
        rng = np.random.default_rng(42)
        df_scatter = pl.DataFrame(
            {"x": rng.standard_normal(50).tolist(), "y": rng.standard_normal(50).tolist()}
        )
        df_rule = pl.DataFrame({"y": [0.0]})

        scatter = fm.Chart(df_scatter).mark_point().encode(x="x", y="y")
        rule = fm.Chart(df_rule).mark_rule().encode(y="y")
        return scatter, rule, df_scatter, df_rule

    def test_rule_layer_single_line(self, scatter_and_rule):
        """Layered rule should produce exactly 1 extra <line element."""
        scatter, rule, _, _ = scatter_and_rule

        # Baseline: count <line elements in scatter-only SVG (axis ticks etc.)
        svg_scatter = scatter.show_svg()
        baseline_lines = len(re.findall(r"<line ", svg_scatter))

        # Layered: should have exactly 1 additional <line from the rule
        layered = scatter + rule
        svg_layered = layered.show_svg()
        layered_lines = len(re.findall(r"<line ", svg_layered))

        rule_lines = layered_lines - baseline_lines
        assert rule_lines == 1, (
            f"Expected exactly 1 rule line, got {rule_lines} extra <line elements "
            f"(baseline={baseline_lines}, layered={layered_lines})"
        )

    def test_rule_layer_column_rename(self, scatter_and_rule):
        """RHS 'y' column should be renamed with a suffix in the merged DataFrame."""
        scatter, rule, _, _ = scatter_and_rule
        layered = scatter + rule

        columns = layered._data.columns
        # LHS "x" and "y" should be present unchanged
        assert "x" in columns
        assert "y" in columns
        # RHS "y" should be renamed with __rhs_ suffix
        rhs_y_cols = [c for c in columns if c.startswith("y__rhs_")]
        assert len(rhs_y_cols) == 1, f"Expected 1 renamed 'y' column, got: {columns}"

    def test_text_layer_overlapping_columns(self):
        """Layering scatter + text callout with shared x/y should render without error."""
        df_scatter = pl.DataFrame(
            {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 1.0, 5.0, 3.0]}
        )
        df_text = pl.DataFrame({"x": [3.0], "y": [5.0], "label": ["peak"]})

        scatter = fm.Chart(df_scatter).mark_point().encode(x="x", y="y")
        text = fm.Chart(df_text).mark_text().encode(x="x", y="y", text="label")
        layered = scatter + text

        svg = layered.show_svg()
        assert svg.strip().startswith("<svg")
        assert len(svg) > 100

    def test_layer_disjoint_columns_unchanged(self):
        """Layering charts with no overlapping columns should not rename anything."""
        df1 = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        df2 = pl.DataFrame({"c": [7.0, 8.0, 9.0], "d": [10.0, 11.0, 12.0]})

        c1 = fm.Chart(df1).mark_point().encode(x="a", y="b")
        c2 = fm.Chart(df2).mark_point().encode(x="c", y="d")
        layered = c1 + c2

        columns = set(layered._data.columns)
        assert columns == {"a", "b", "c", "d"}, f"Unexpected columns: {columns}"

    def test_layer_same_data_unchanged(self):
        """Layering two charts sharing the same DataFrame should not rename columns."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})

        c1 = fm.Chart(df).mark_point().encode(x="x", y="y")
        c2 = fm.Chart(df).mark_line().encode(x="x", y="y")
        layered = c1 + c2

        columns = layered._data.columns
        assert columns == ["x", "y"], f"Expected unchanged columns, got: {columns}"
        assert layered._data.shape == df.shape


# ---------------------------------------------------------------------------
# Bug 2 (extended): chained 3+ layer `+` operations with overlapping columns
# ---------------------------------------------------------------------------


class TestChainedLayerFix:
    """Verify that chaining 3+ layers with overlapping column names produces correct SVG.

    Previously, chaining `scatter + hline + vline + label` where each layer's
    DataFrame shared column names (e.g., "x", "y") caused the renderer to
    duplicate lines from other layers. The fix ensures each rule layer draws
    exactly one line and scatter points are preserved.
    """

    @pytest.fixture()
    def four_layer_chart(self):
        """Build a 4-layer chart: scatter + hline + vline + text label."""
        rng = np.random.default_rng(42)
        df = pl.DataFrame(
            {"x": rng.standard_normal(50).tolist(), "y": rng.standard_normal(50).tolist()}
        )
        scatter = fm.Chart(df).mark_point(opacity=0.6).encode(x="x", y="y")
        hline = (
            fm.Chart(pl.DataFrame({"y": [0.0]}))
            .mark_rule(stroke="red", stroke_dash=[4, 2])
            .encode(y="y")
        )
        vline = (
            fm.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(stroke="blue", stroke_dash=[4, 2])
            .encode(x="x")
        )
        label = (
            fm.Chart(pl.DataFrame({"x": [1.5], "y": [2.0], "label": ["outlier"]}))
            .mark_text(fill="gray")
            .encode(x="x", y="y", text="label")
        )
        return scatter, hline, vline, label, df

    def test_chained_3_layer_hline_vline(self, four_layer_chart):
        """scatter + hline + vline should produce exactly 2 dashed rule lines."""
        scatter, hline, vline, _, _ = four_layer_chart

        layered = scatter + hline + vline
        svg = layered.show_svg()

        assert svg.strip().startswith("<svg")
        dashed_lines = len(re.findall(r"<line[^>]*stroke-dasharray[^>]*>", svg))
        assert dashed_lines == 2, (
            f"Expected exactly 2 dashed <line elements (hline + vline), got {dashed_lines}"
        )

    def test_chained_4_layer_with_text(self, four_layer_chart):
        """scatter + hline + vline + text should render valid SVG with exactly 2 dashed lines."""
        scatter, hline, vline, label, _ = four_layer_chart

        layered = scatter + hline + vline + label
        svg = layered.show_svg()

        assert svg.strip().startswith("<svg")
        assert len(svg) > 100
        dashed_lines = len(re.findall(r"<line[^>]*stroke-dasharray[^>]*>", svg))
        assert dashed_lines == 2, (
            f"Expected exactly 2 dashed <line elements (hline + vline), got {dashed_lines}"
        )

    def test_chained_preserves_scatter_points(self, four_layer_chart):
        """4-layer chart should preserve scatter <circle elements matching input row count."""
        scatter, hline, vline, label, df = four_layer_chart

        layered = scatter + hline + vline + label
        svg = layered.show_svg()

        circles = len(re.findall(r"<circle", svg))
        assert circles == len(df), (
            f"Expected {len(df)} <circle elements for scatter points, got {circles}"
        )
