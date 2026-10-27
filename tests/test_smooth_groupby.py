"""Tests for Smooth transform groupby support.

Covers:
1. mark_smooth(groupby="group") produces SVG output with multiple groups
2. The Smooth output includes the group column for color encoding
3. lmplot with hue renders without errors using the new groupby path
"""

from __future__ import annotations

import re
import warnings

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
        svg = chart.to_svg()
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
        svg = chart.to_svg()
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
        svg = chart.to_svg()
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
        svg = chart.to_svg()
        assert svg.startswith("<svg")


class TestLmplotHue:
    """lmplot with hue uses Smooth groupby under the hood."""

    def test_lmplot_hue_renders(self, grouped_data):
        """lmplot(hue='group') renders without errors."""
        chart = fm.lmplot(grouped_data, x="x", y="y", hue="group")
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 100

    def test_lmplot_hue_lm(self, grouped_data):
        """lmplot with method='lm' and hue renders per-group fit lines."""
        chart = fm.lmplot(grouped_data, x="x", y="y", hue="group", method="lm")
        svg = chart.to_svg()
        assert svg.startswith("<svg")

    def test_lmplot_hue_loess(self, grouped_data):
        """lmplot with method='loess' and hue renders per-group fit lines."""
        chart = fm.lmplot(grouped_data, x="x", y="y", hue="group", method="loess")
        svg = chart.to_svg()
        assert svg.startswith("<svg")

    def test_lmplot_hue_with_ci(self, grouped_data):
        """lmplot with hue and CI band renders."""
        chart = fm.lmplot(grouped_data, x="x", y="y", hue="group", ci=95)
        svg = chart.to_svg()
        assert svg.startswith("<svg")

    def test_lmplot_hue_no_scatter(self, grouped_data):
        """lmplot with hue and scatter=False renders only fit lines."""
        chart = fm.lmplot(grouped_data, x="x", y="y", hue="group", scatter=False)
        svg = chart.to_svg()
        assert svg.startswith("<svg")


class TestSmoothGroupbyUndefinedBand:
    """An undefined confidence band warns, naming the group it belongs to --
    one warning per undersized group, and a well-supported sibling group's
    own band is byte-identical whether it is fit alongside an undersized
    group or alone."""

    def test_mixed_group_sizes_warn_only_for_the_undersized_group(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [1.0, 2.0, 1.0, 2.0, 3.0, 5.0, 4.0],
                "g": ["a", "a", "b", "b", "b", "b", "b"],
            }
        )
        chart = (
            Chart(df).mark_smooth(method="lm", ci=0.95, groupby="g").encode(x="x", y="y", color="g")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        msgs = [str(w.message) for w in caught]
        assert len(msgs) == 1
        assert "g=a" in msgs[0]
        assert "g=b" not in msgs[0]

        # The ribbon is absent for the sparse group but present for the
        # well-supported one: exactly one <path> (group b's band).
        assert svg.count("<path") == 1

        # Group b's own band GEOMETRY renders byte-identically whether fit
        # mixed with the undersized group a or alone -- the undersized
        # sibling never perturbs it (the fill differs between the two
        # renders because "b" is the first color in the solo render and
        # the second in the mixed one, which is the ordinal color scale's
        # own behavior, so only the path's own geometry is compared here).
        df_b_alone = pl.DataFrame(
            {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 2.0, 3.0, 5.0, 4.0], "g": ["b"] * 5}
        )
        chart_b_alone = (
            Chart(df_b_alone)
            .mark_smooth(method="lm", ci=0.95, groupby="g")
            .encode(x="x", y="y", color="g")
        )
        svg_b_alone = chart_b_alone.to_svg()
        mixed_match = re.search(r'<path d="([^"]+)"', svg)
        alone_match = re.search(r'<path d="([^"]+)"', svg_b_alone)
        assert mixed_match is not None, "expected a <path> element in the mixed render"
        assert alone_match is not None, "expected a <path> element in the solo render"
        assert mixed_match.group(1) == alone_match.group(1)

    def test_two_groups_both_n2_warn_once_each(self):
        df = pl.DataFrame(
            {"x": [1.0, 2.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]}
        )
        chart = (
            Chart(df).mark_smooth(method="lm", ci=0.95, groupby="g").encode(x="x", y="y", color="g")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart.to_svg()
        msgs = [str(w.message) for w in caught]
        assert len(msgs) == 2
        assert any("g=a" in m for m in msgs)
        assert any("g=b" in m for m in msgs)
