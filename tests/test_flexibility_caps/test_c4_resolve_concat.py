"""C4 — resolve= on vconcat/hconcat and pairplot shared color domain.

Tests for:
1. HConcatChart / VConcatChart accept and store resolve=.
2. fm.hconcat() / fm.vconcat() free functions forward resolve=.
3. resolve={"color":"shared"} unifies the color scale domain across panels.
4. pairplot(hue=...) produces a RepeatChart with resolve={"color":"shared"}.
5. Regression: hconcat/vconcat without resolve= behaves as before.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.composition import HConcatChart, VConcatChart


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_two_groups():
    """DataFrame with a numeric and a categorical column for color encoding."""
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [2, 4, 1, 5, 3, 6],
            "cat": ["a", "a", "b", "b", "c", "c"],
        }
    )


@pytest.fixture
def chart_left(df_two_groups):
    return fm.Chart(df_two_groups).mark_point().encode(x="x", y="y", color="cat")


@pytest.fixture
def chart_right(df_two_groups):
    return fm.Chart(df_two_groups).mark_bar().encode(x="x", y="y", color="cat")


@pytest.fixture
def chart_top(df_two_groups):
    return fm.Chart(df_two_groups).mark_point().encode(x="x", y="y", color="cat")


@pytest.fixture
def chart_bottom(df_two_groups):
    return fm.Chart(df_two_groups).mark_line().encode(x="x", y="y", color="cat")


# ---------------------------------------------------------------------------
# HConcatChart — resolve= storage and validation
# ---------------------------------------------------------------------------


class TestHConcatChartResolve:
    """HConcatChart accepts, stores, and validates resolve=."""

    def test_hconcat_stores_resolve(self, chart_left, chart_right):
        """HConcatChart stores the resolve dict."""
        rc = HConcatChart([chart_left, chart_right], resolve={"color": "shared"})
        assert rc._resolve == {"color": "shared"}

    def test_hconcat_resolve_none_by_default(self, chart_left, chart_right):
        """HConcatChart._resolve is None when resolve= is not passed."""
        rc = HConcatChart([chart_left, chart_right])
        assert rc._resolve is None

    def test_hconcat_invalid_resolve_mode_raises(self, chart_left, chart_right):
        """HConcatChart rejects unknown resolve mode values."""
        with pytest.raises(ValueError, match="expected 'shared' or 'independent'"):
            HConcatChart([chart_left, chart_right], resolve={"color": "bad"})

    def test_hconcat_invalid_resolve_type_raises(self, chart_left, chart_right):
        """HConcatChart rejects non-dict resolve argument."""
        with pytest.raises(ValueError):
            HConcatChart([chart_left, chart_right], resolve="shared")

    def test_hconcat_resolve_shared_svg_renders(self, chart_left, chart_right):
        """HConcatChart with resolve=shared renders without error."""
        rc = HConcatChart([chart_left, chart_right], resolve={"color": "shared"})
        svg = rc.to_svg()
        assert "<svg" in svg

    def test_hconcat_resolve_unifies_color_domain(self, df_two_groups):
        """Resolved charts share a unified color scale domain."""
        # Build charts with a subset of data each — different color domains.
        c1 = (
            fm.Chart(df_two_groups.filter(pl.col("cat").is_in(["a", "b"])))
            .mark_point()
            .encode(x="x", y="y", color="cat")
        )
        c2 = (
            fm.Chart(df_two_groups.filter(pl.col("cat").is_in(["b", "c"])))
            .mark_point()
            .encode(x="x", y="y", color="cat")
        )
        rc = HConcatChart([c1, c2], resolve={"color": "shared"})
        resolved = rc._resolved_charts()
        # After resolution all charts must share a domain covering all three categories.
        domains = []
        for chart in resolved:
            enc = chart._encoding if hasattr(chart, "_encoding") else {}
            color_enc = enc.get("color")
            if color_enc is not None:
                scale = getattr(color_enc, "_kwargs", {}).get("scale") or {}
                domain = scale.get("domain", [])
                domains.append(set(domain))
        # All panels must have the same domain.
        assert len(domains) >= 2
        assert domains[0] == domains[1]
        assert {"a", "b", "c"} == domains[0]


# ---------------------------------------------------------------------------
# VConcatChart — resolve= storage and validation
# ---------------------------------------------------------------------------


class TestVConcatChartResolve:
    """VConcatChart accepts, stores, and validates resolve=."""

    def test_vconcat_stores_resolve(self, chart_top, chart_bottom):
        """VConcatChart stores the resolve dict."""
        rc = VConcatChart([chart_top, chart_bottom], resolve={"color": "shared"})
        assert rc._resolve == {"color": "shared"}

    def test_vconcat_resolve_none_by_default(self, chart_top, chart_bottom):
        """VConcatChart._resolve is None when resolve= is not passed."""
        rc = VConcatChart([chart_top, chart_bottom])
        assert rc._resolve is None

    def test_vconcat_invalid_resolve_mode_raises(self, chart_top, chart_bottom):
        """VConcatChart rejects unknown resolve mode values."""
        with pytest.raises(ValueError, match="expected 'shared' or 'independent'"):
            VConcatChart([chart_top, chart_bottom], resolve={"color": "bad"})

    def test_vconcat_invalid_resolve_type_raises(self, chart_top, chart_bottom):
        """VConcatChart rejects non-dict resolve argument."""
        with pytest.raises(ValueError):
            VConcatChart([chart_top, chart_bottom], resolve="shared")

    def test_vconcat_resolve_shared_svg_renders(self, chart_top, chart_bottom):
        """VConcatChart with resolve=shared renders without error."""
        rc = VConcatChart([chart_top, chart_bottom], resolve={"color": "shared"})
        svg = rc.to_svg()
        assert "<svg" in svg

    def test_vconcat_resolve_unifies_color_domain(self, df_two_groups):
        """Resolved charts share a unified color scale domain."""
        c1 = (
            fm.Chart(df_two_groups.filter(pl.col("cat").is_in(["a", "b"])))
            .mark_point()
            .encode(x="x", y="y", color="cat")
        )
        c2 = (
            fm.Chart(df_two_groups.filter(pl.col("cat").is_in(["b", "c"])))
            .mark_point()
            .encode(x="x", y="y", color="cat")
        )
        rc = VConcatChart([c1, c2], resolve={"color": "shared"})
        resolved = rc._resolved_charts()
        domains = []
        for chart in resolved:
            enc = chart._encoding if hasattr(chart, "_encoding") else {}
            color_enc = enc.get("color")
            if color_enc is not None:
                scale = getattr(color_enc, "_kwargs", {}).get("scale") or {}
                domain = scale.get("domain", [])
                domains.append(set(domain))
        assert len(domains) >= 2
        assert domains[0] == domains[1]
        assert {"a", "b", "c"} == domains[0]


# ---------------------------------------------------------------------------
# Free functions: fm.hconcat() and fm.vconcat()
# ---------------------------------------------------------------------------


class TestFreeFunctions:
    """fm.hconcat() and fm.vconcat() thread resolve= through."""

    def test_hconcat_free_function_forwards_resolve(self, chart_left, chart_right):
        """fm.hconcat() passes resolve= to HConcatChart."""
        rc = fm.hconcat(chart_left, chart_right, resolve={"color": "shared"})
        assert isinstance(rc, HConcatChart)
        assert rc._resolve == {"color": "shared"}

    def test_hconcat_free_function_no_resolve(self, chart_left, chart_right):
        """fm.hconcat() without resolve= leaves _resolve=None."""
        rc = fm.hconcat(chart_left, chart_right)
        assert isinstance(rc, HConcatChart)
        assert rc._resolve is None

    def test_vconcat_free_function_forwards_resolve(self, chart_top, chart_bottom):
        """fm.vconcat() passes resolve= to VConcatChart."""
        rc = fm.vconcat(chart_top, chart_bottom, resolve={"color": "shared"})
        assert isinstance(rc, VConcatChart)
        assert rc._resolve == {"color": "shared"}

    def test_vconcat_free_function_no_resolve(self, chart_top, chart_bottom):
        """fm.vconcat() without resolve= leaves _resolve=None."""
        rc = fm.vconcat(chart_top, chart_bottom)
        assert isinstance(rc, VConcatChart)
        assert rc._resolve is None

    def test_hconcat_free_function_shared_svg_renders(self, chart_left, chart_right):
        """fm.hconcat(..., resolve=...) renders without error."""
        rc = fm.hconcat(chart_left, chart_right, resolve={"color": "shared"})
        svg = rc.to_svg()
        assert "<svg" in svg

    def test_vconcat_free_function_shared_svg_renders(self, chart_top, chart_bottom):
        """fm.vconcat(..., resolve=...) renders without error."""
        rc = fm.vconcat(chart_top, chart_bottom, resolve={"color": "shared"})
        svg = rc.to_svg()
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# pairplot — shared color domain when hue= is set
# ---------------------------------------------------------------------------


class TestPairplotSharedColorDomain:
    """pairplot(hue=...) must produce a RepeatChart with resolve={"color":"shared"}."""

    @pytest.fixture
    def df_iris(self):
        return pl.DataFrame(
            {
                "sepal_length": [5.1, 4.9, 4.7, 6.4, 6.3, 6.7, 7.0, 6.4, 6.9],
                "sepal_width": [3.5, 3.0, 3.2, 3.2, 2.5, 3.1, 3.2, 3.2, 3.1],
                "petal_length": [1.4, 1.4, 1.3, 4.5, 4.9, 5.6, 4.7, 4.5, 4.9],
                "species": [
                    "setosa",
                    "setosa",
                    "setosa",
                    "versicolor",
                    "versicolor",
                    "versicolor",
                    "virginica",
                    "virginica",
                    "virginica",
                ],
            }
        )

    def test_pairplot_hue_resolve_shared(self, df_iris):
        """pairplot with hue= sets resolve={"color":"shared"} on RepeatChart."""
        rc = fm.pairplot(
            df_iris,
            vars=["sepal_length", "sepal_width", "petal_length"],
            hue="species",
        )
        assert isinstance(rc, fm.RepeatChart)
        assert rc.resolve is not None
        assert rc.resolve.get("color") == "shared"

    def test_pairplot_no_hue_no_color_resolve(self, df_iris):
        """pairplot without hue= does not force a color resolve."""
        rc = fm.pairplot(
            df_iris,
            vars=["sepal_length", "sepal_width", "petal_length"],
        )
        assert isinstance(rc, fm.RepeatChart)
        # Without hue, either resolve is None or color is not "shared".
        if rc.resolve is not None:
            assert rc.resolve.get("color") != "shared"

    def test_pairplot_hue_svg_renders(self, df_iris):
        """pairplot with hue= and shared resolve renders without error."""
        rc = fm.pairplot(
            df_iris,
            vars=["sepal_length", "sepal_width"],
            hue="species",
        )
        svg = rc.to_svg()
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# Regression: no-resolve behavior is unchanged
# ---------------------------------------------------------------------------


class TestRegressionNoResolve:
    """Without resolve=, hconcat/vconcat behave exactly as before."""

    def test_hconcat_no_resolve_svg(self, chart_left, chart_right):
        """HConcatChart without resolve= still renders valid SVG."""
        rc = fm.hconcat(chart_left, chart_right)
        svg = rc.to_svg()
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_vconcat_no_resolve_svg(self, chart_top, chart_bottom):
        """VConcatChart without resolve= still renders valid SVG."""
        rc = fm.vconcat(chart_top, chart_bottom)
        svg = rc.to_svg()
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_pipe_operator_no_resolve(self, chart_left, chart_right):
        """The | operator produces an HConcatChart with _resolve=None."""
        rc = chart_left | chart_right
        assert isinstance(rc, HConcatChart)
        assert rc._resolve is None

    def test_and_operator_no_resolve(self, chart_top, chart_bottom):
        """The & operator produces a VConcatChart with _resolve=None."""
        rc = chart_top & chart_bottom
        assert isinstance(rc, VConcatChart)
        assert rc._resolve is None

    def test_hconcat_resolved_charts_passthrough(self, chart_left, chart_right):
        """_resolved_charts() without resolve= returns the original chart list."""
        rc = HConcatChart([chart_left, chart_right])
        assert rc._resolved_charts() is rc.charts or rc._resolved_charts() == rc.charts

    def test_vconcat_resolved_charts_passthrough(self, chart_top, chart_bottom):
        """_resolved_charts() without resolve= returns the original chart list."""
        rc = VConcatChart([chart_top, chart_bottom])
        assert rc._resolved_charts() is rc.charts or rc._resolved_charts() == rc.charts
