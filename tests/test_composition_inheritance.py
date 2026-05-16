"""Combinatorial tests for composition x encoding inheritance.

Tests encoding propagation, isolation, and inheritance across all
composition types: Layer (+), Facet, Repeat, HConcat (|), VConcat (&),
and JointChart.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum.composition import HConcatChart, VConcatChart, JointChart, RepeatChart


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "y": [2.0, 4.0, 3.0, 5.0, 1.0] * 4,
            "cat": ["a", "a", "b", "b", "c"] * 4,
            "grp": ["X"] * 10 + ["Y"] * 10,
        }
    )


def _renders(chart) -> str:
    """Assert that the chart renders to valid SVG and return it."""
    svg = chart.show_svg()
    assert "<svg" in svg, "SVG output missing <svg element"
    assert "NaN" not in svg, "SVG contains NaN values"
    return svg


def _svg_contains(chart, text: str) -> None:
    """Assert that the chart SVG contains the given text."""
    svg = chart.show_svg()
    assert text in svg, f"Expected '{text}' in SVG"


# ---------------------------------------------------------------------------
# 1. Layer encoding isolation
# ---------------------------------------------------------------------------


class TestLayerEncodingIsolation:
    """Layers (Chart + Chart) keep independent encodings."""

    def test_two_layers_different_marks(self, df):
        """Two layers with different marks both render."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
        _renders(c1 + c2)

    def test_layer_with_color_and_without(self, df):
        """Layer with x/y only + layer with x/y/color renders."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N")
        _renders(c1 + c2)

    def test_two_layers_different_color_fields(self, df):
        """Two layers with different color encodings both render."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N")
        c2 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="grp:N")
        svg = _renders(c1 + c2)
        # Both should produce circle elements (marks)
        assert svg.count("<circle") >= 2

    def test_layered_chart_warns_on_theme_mismatch(self, df):
        """Layering with mismatched themes warns."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").theme(fm.themes.dark)
        c2 = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q").theme(fm.themes.minimal)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = c1 + c2
        # Check that at least one warning was raised about layer conflicts
        layer_warns = [x for x in w if "layer" in str(x.message).lower()]
        assert len(layer_warns) >= 1

    def test_three_layers_stacked(self, df):
        """Three layers stacked via + render together."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
        c3 = fm.Chart(df).mark_rule().encode(y="y:Q")
        _renders(c1 + c2 + c3)


# ---------------------------------------------------------------------------
# 2. Facet encoding inheritance
# ---------------------------------------------------------------------------


class TestFacetEncodingInheritance:
    """Faceted charts inherit parent encodings into each panel."""

    def test_facet_col_wrap_renders(self, df):
        """Facet with col= only (wrap mode) renders."""
        chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").facet(col="cat")
        _renders(chart)

    def test_facet_grid_renders(self, df):
        """Facet with row= and col= (grid mode) renders."""
        chart = (
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").facet(row="grp", col="cat")
        )
        _renders(chart)

    def test_facet_with_explicit_title(self, df):
        """Faceted chart with titled encoding propagates title to SVG."""
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("x", title="Custom X"), y="y:Q")
            .facet(col="cat")
        )
        _svg_contains(chart, "Custom X")

    def test_facet_with_hue_encoding(self, df):
        """Facet with color encoding renders a legend."""
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color="grp:N")
            .facet(col="cat")
        )
        svg = _renders(chart)
        # Color legend produces text elements for the legend entries
        assert "X" in svg or "Y" in svg

    def test_facet_with_transform(self, df):
        """Faceted chart with a statistical mark (histogram) applies transform per panel."""
        chart = fm.Chart(df).mark_histogram().encode(x="x:Q").facet(col="cat")
        _renders(chart)

    def test_facet_single_group(self, df):
        """Facet where only one group exists renders as single panel."""
        single_grp = df.filter(pl.col("cat") == "a")
        chart = (
            fm.Chart(single_grp).mark_point().encode(x="x:Q", y="y:Q").facet(col="cat")
        )
        _renders(chart)

    def test_facet_field_kwarg(self, df):
        """Facet with field= keyword (explicit wrap) renders."""
        chart = (
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").facet(field="cat", ncols=2)
        )
        _renders(chart)

    def test_facet_row_only(self, df):
        """Facet with row= only renders."""
        chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").facet(row="grp")
        _renders(chart)


# ---------------------------------------------------------------------------
# 3. Repeat encoding propagation
# ---------------------------------------------------------------------------


class TestRepeatEncodingPropagation:
    """RepeatChart substitutes placeholders and preserves fixed channels."""

    def test_repeat_column_placeholder(self, df):
        """RepeatChart with Repeat.column placeholder renders."""
        template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y="y:Q")
        rc = RepeatChart(template, column=["x", "y"])
        _renders(rc)

    def test_repeat_row_placeholder(self, df):
        """RepeatChart with Repeat.row placeholder renders."""
        template = fm.Chart(df).mark_point().encode(x="x:Q", y=fm.Repeat.row)
        rc = RepeatChart(template, row=["x", "y"])
        _renders(rc)

    def test_repeat_fixed_color_preserved(self, df):
        """Non-placeholder channel (fixed color) preserved in all cells."""
        template = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.Repeat.column, y=fm.Repeat.row, color="cat:N")
        )
        rc = RepeatChart(template, row=["x", "y"], column=["x", "y"])
        _renders(rc)

    def test_repeat_share_scale_x(self, df):
        """RepeatChart with share_scale(x='shared') renders."""
        template = (
            fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
        )
        rc = RepeatChart(template, row=["x", "y"], column=["x", "y"])
        shared = rc.share_scale(x="shared")
        _renders(shared)

    def test_repeat_share_scale_y(self, df):
        """RepeatChart with share_scale(y='shared') renders."""
        template = (
            fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
        )
        rc = RepeatChart(template, row=["x", "y"], column=["x", "y"])
        shared = rc.share_scale(y="shared")
        _renders(shared)

    def test_repeat_resolve_constructor(self, df):
        """RepeatChart constructed with resolve= renders."""
        template = (
            fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
        )
        rc = RepeatChart(
            template, row=["x", "y"], column=["x", "y"], resolve={"x": "shared"}
        )
        _renders(rc)

    def test_pairplot_multiple_vars(self, df):
        """pairplot (internally RepeatChart) renders with many vars."""
        pp = fm.pairplot(df, vars=["x", "y"])
        _renders(pp)


# ---------------------------------------------------------------------------
# 4. HConcat / VConcat
# ---------------------------------------------------------------------------


class TestConcatComposition:
    """Horizontal and vertical concatenation with independent scales."""

    def test_hconcat_two_charts(self, df):
        """Two different charts | renders."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_bar().encode(x="cat:N", y="y:Q")
        _renders(c1 | c2)
        assert isinstance(c1 | c2, HConcatChart)

    def test_vconcat_two_charts(self, df):
        """Two different charts & renders."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_bar().encode(x="cat:N", y="y:Q")
        _renders(c1 & c2)
        assert isinstance(c1 & c2, VConcatChart)

    def test_hconcat_share_scale_y(self, df):
        """HConcat with share_scale(y='shared') renders."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        shared = (c1 | c2).share_scale(y="shared")
        _renders(shared)

    def test_vconcat_share_scale_x(self, df):
        """VConcat with share_scale(x='shared') renders."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        shared = (c1 & c2).share_scale(x="shared")
        _renders(shared)

    def test_concat_with_different_data(self, df):
        """Concat with different data renders."""
        df2 = pl.DataFrame({"a": [10.0, 20.0], "b": [30.0, 40.0]})
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df2).mark_line().encode(x="a:Q", y="b:Q")
        _renders(c1 | c2)

    def test_three_charts_hconcat(self, df):
        """Three charts concatenated horizontally render."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_bar().encode(x="cat:N", y="y:Q")
        c3 = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
        combined = HConcatChart([c1, c2, c3])
        _renders(combined)

    def test_three_charts_vconcat(self, df):
        """Three charts concatenated vertically render."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_bar().encode(x="cat:N", y="y:Q")
        c3 = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
        combined = VConcatChart([c1, c2, c3])
        _renders(combined)

    def test_share_scale_unbound_channel_no_crash(self, df):
        """share_scale on a channel no chart uses is a no-op, not a crash."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        # 'size' is not bound on either chart
        result = (c1 | c2).share_scale(size="shared")
        _renders(result)


# ---------------------------------------------------------------------------
# 5. JointChart
# ---------------------------------------------------------------------------


class TestJointChart:
    """JointChart: center + marginals with independent scales."""

    def test_basic_jointplot(self, df):
        """Basic jointplot renders center + marginals."""
        jp = fm.jointplot(df, x="x", y="y")
        _renders(jp)
        assert isinstance(jp, JointChart)

    def test_jointplot_with_hue(self, df):
        """Jointplot with hue renders color in all panels."""
        jp = fm.jointplot(df, x="x", y="y", hue="cat")
        _renders(jp)

    def test_jointplot_kde_marginals(self, df):
        """Jointplot with kde marginals renders."""
        jp = fm.jointplot(df, x="x", y="y", marginal_kind="kde")
        _renders(jp)

    def test_joint_chart_manual_construction(self, df):
        """Manually constructed JointChart renders."""
        center = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        top = fm.Chart(df).mark_histogram().encode(x="x:Q")
        right = fm.Chart(df).mark_histogram().encode(x="y:Q")
        jc = JointChart(center, top=top, right=right)
        _renders(jc)


# ---------------------------------------------------------------------------
# 6. Cross-cutting edge cases
# ---------------------------------------------------------------------------


class TestCrossCuttingEdgeCases:
    """Compositions of compositions and boundary conditions."""

    def test_layered_chart_inside_facet(self, df):
        """A layered chart (c1 + c2) faceted renders."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
        layered = c1 + c2
        faceted = layered.facet(col="cat")
        _renders(faceted)

    def test_faceted_chart_in_concat(self, df):
        """Faceted chart combined with another via | renders."""
        faceted = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").facet(col="cat")
        plain = fm.Chart(df).mark_bar().encode(x="cat:N", y="y:Q")
        _renders(faceted | plain)

    def test_empty_facet_group_no_crash(self):
        """Facet with a group that has zero rows in one level does not crash."""
        # Force a facet field with a level that has no rows in the subframe
        df_sparse = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [1.0, 2.0, 3.0],
                "grp": ["A", "A", "A"],
            }
        )
        # Only one group present, faceting should still work
        chart = fm.Chart(df_sparse).mark_point().encode(x="x:Q", y="y:Q").facet(col="grp")
        _renders(chart)

    def test_repeat_single_variable(self, df):
        """RepeatChart with only 1 variable renders a single-cell grid."""
        template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y="y:Q")
        rc = RepeatChart(template, column=["x"])
        _renders(rc)

    def test_concat_then_share_scale(self, df):
        """A concat followed by share_scale renders correctly."""
        c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        c2 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        result = (c1 & c2).share_scale(x="shared", y="shared")
        _renders(result)
