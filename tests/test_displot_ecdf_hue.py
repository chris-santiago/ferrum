"""Pins that ``hue=`` reaches the ecdf statistic's grouping.

A bound ``hue=`` reaches the per-observation rolling count
(``fm.transform_window``) the ecdf branch builds, so each level keeps its
own count and its own color, drawn as a step-after line (a ``<path>`` of
``M``/``H``/``V`` commands).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import value_axis_ladder, value_for_pixel
from tests._svg_rects import polylines, rug_tick_x1_by_stroke, step_after_paths

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_group_df():
    """Two equal-size, well-separated normal samples tagged by ``g``.

    Equal group sizes make "each level reaches its own row count" a plain
    equality (150 and 150) rather than two different numbers to track.
    """
    rng = np.random.default_rng(0)
    n = 150
    return pl.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, n), rng.normal(6, 1, n)]),
            "g": ["a"] * n + ["b"] * n,
        }
    )


@pytest.fixture
def two_facet_df():
    """Two hue levels crossed with two facet levels, equal counts throughout."""
    rng = np.random.default_rng(0)
    n = 100
    return pl.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, n), rng.normal(6, 1, n)]),
            "g": ["a"] * n + ["b"] * n,
            "c": (["p1"] * (n // 2) + ["p2"] * (n // 2)) * 2,
        }
    )


def _assert_finite_svg(svg: str) -> None:
    assert "NaN" not in svg
    assert "Infinity" not in svg


# ---------------------------------------------------------------------------
# Distinct-colored per-group step lines, each reaching its own count.
# ---------------------------------------------------------------------------


class TestEcdfHueCarriesGroupKey:
    def test_ecdf_hue_renders_distinct_colored_lines_reaching_each_groups_own_count(
        self, two_group_df
    ):
        chart = fm.displot(two_group_df, x="x", hue="g", kind="ecdf")
        svg = chart.to_svg()
        _assert_finite_svg(svg)

        lines = step_after_paths(svg)
        assert len(lines) == 2, f"expected one step line per hue level, got {len(lines)}"
        strokes = {stroke for stroke, _ in lines}
        assert len(strokes) == 2, f"step lines are not distinctly colored: {strokes}"

        # Each level's own count, from `groupby=` carrying the hue column
        # into the window: a dropped `groupby=` would instead count across
        # both groups pooled, reaching 300 on one or both lines rather than
        # 150 on each. Each line also rises from 0 at its first
        # observation, so its corners are that riser plus 150 steps (151).
        y_ladder = value_axis_ladder(svg, axis="y")
        for _stroke, corners in lines:
            heights = sorted(round(value_for_pixel(y_ladder, py)) for _px, py in corners)
            assert len(corners) == 151, f"expected the zero riser plus 150 steps, got {len(corners)}"
            assert heights[0] == 0, f"expected the line to rise from 0, got {heights[0]}"
            assert heights[-1] == 150, f"expected this level's own row count, got {heights[-1]}"

    def test_ecdf_hue_channel_object_matches_bare_string(self, two_group_df):
        """`hue=fm.Color("g")` must carry the group exactly like `hue="g"` --
        `_field_name` is the one unwrap point for this documented trap
        (see its docstring in `plots/_helpers.py`)."""
        svg_str = fm.displot(two_group_df, x="x", hue="g", kind="ecdf").to_svg()
        svg_channel = fm.displot(two_group_df, x="x", hue=fm.Color("g"), kind="ecdf").to_svg()
        assert svg_str == svg_channel

    def test_ecdf_hue_with_col_facets_each_panel_colored(self, two_facet_df):
        chart = fm.displot(two_facet_df, x="x", hue="g", col="c", kind="ecdf")
        svg = chart.to_svg()
        _assert_finite_svg(svg)
        lines = step_after_paths(svg)
        # Two hue levels x two facet panels.
        assert len(lines) == 4, f"expected 4 step lines (2 hue x 2 facets), got {len(lines)}"
        strokes = {stroke for stroke, _ in lines}
        assert len(strokes) == 2, f"facet panels do not reuse the same 2 hue colors: {strokes}"

    def test_ecdf_hue_with_kde_overlay_both_colored_by_group(self, two_group_df):
        chart = fm.displot(two_group_df, x="x", hue="g", kind="ecdf", kde=True)
        svg = chart.to_svg()
        _assert_finite_svg(svg)
        # One ECDF step line (`<path>`) + one KDE overlay curve (`<polyline>`)
        # per hue level.
        ecdf_lines = step_after_paths(svg)
        kde_lines = polylines(svg)
        assert len(ecdf_lines) == 2, f"expected 2 ecdf step lines, got {len(ecdf_lines)}"
        assert len(kde_lines) == 2, f"expected 2 kde overlay curves, got {len(kde_lines)}"
        strokes = {stroke for stroke, _ in ecdf_lines} | {stroke for stroke, _ in kde_lines}
        assert len(strokes) == 2, (
            f"kde overlay does not share the ecdf lines' 2 hue colors: {strokes}"
        )

    def test_ecdf_hue_with_rug_overlay_ticks_match_line_colors(self, two_group_df):
        chart = fm.displot(two_group_df, x="x", hue="g", kind="ecdf", rug=True)
        svg = chart.to_svg()
        _assert_finite_svg(svg)
        lines = step_after_paths(svg)
        assert len(lines) == 2, f"expected 2 ecdf step lines, got {len(lines)}"
        line_strokes = {stroke for stroke, _ in lines}

        rug_ticks = rug_tick_x1_by_stroke(svg)
        assert set(rug_ticks) == line_strokes, (
            f"rug tick colors {set(rug_ticks)} do not match the ecdf lines' "
            f"hue colors {line_strokes}"
        )


# ---------------------------------------------------------------------------
# Clean-input contract: the no-hue path carries no stray grouping.
# ---------------------------------------------------------------------------


class TestEcdfNoHueUnchanged:
    def test_ecdf_no_hue_wire_has_no_groupby(self):
        """With no grouping channel bound, the window's `groupby=` is
        unset, so no grouping channel's binding can leak onto an ungrouped
        render."""
        df = pl.DataFrame({"x": np.arange(20.0)})
        chart = fm.displot(df, x="x", kind="ecdf")
        spec = chart.to_dict()
        (window,) = [t for t in spec["transforms"] if t["type"] == "data_window"]
        assert "groupby" not in window
        assert window["sort"] == ["x"]

    def test_ecdf_no_hue_renders_one_step_line_reaching_the_row_count(self):
        df = pl.DataFrame({"x": np.arange(20.0)})
        svg = fm.displot(df, x="x", kind="ecdf").to_svg()
        lines = step_after_paths(svg)
        assert len(lines) == 1, f"expected one step line, got {len(lines)}"
        _stroke, corners = lines[0]
        # The line rises from 0 at its first observation, so its corners
        # are that riser plus the 20 steps (21).
        assert len(corners) == 21
        y_ladder = value_axis_ladder(svg, axis="y")
        heights = sorted(round(value_for_pixel(y_ladder, py)) for _px, py in corners)
        assert heights == [0, *range(1, 21)], heights


# ---------------------------------------------------------------------------
# A grouping a step wrote before the ECDF
# ---------------------------------------------------------------------------


def _rank_within(groupby: list[str]) -> dict:
    return fm.transform_window({"op": "rank", "as": "r"}, sort=["x"], groupby=groupby)


def test_a_window_grouped_before_the_ecdf_does_not_split_its_line(two_group_df):
    """The ECDF counts across every row, so a ``g`` the caller's window grouped
    by does not split its line: one step line, byte-equal to the plain ECDF.
    """
    plain = fm.displot(two_group_df, x="x", kind="ecdf").to_svg()
    windowed = fm.displot(two_group_df, x="x", kind="ecdf").transform(_rank_within(["g"])).to_svg()
    assert len(step_after_paths(windowed)) == 1
    assert windowed == plain


def test_a_window_grouped_by_the_hue_before_the_ecdf_matches_the_hue_alone(two_group_df):
    """With ``hue="g"`` the ECDF partitions by ``g`` itself, so the same window
    changes nothing: one step line per level, byte-equal to the hue alone.
    """
    hue = fm.displot(two_group_df, x="x", kind="ecdf", hue="g").to_svg()
    windowed = (
        fm.displot(two_group_df, x="x", kind="ecdf", hue="g")
        .transform(_rank_within(["g"]))
        .to_svg()
    )
    assert len(step_after_paths(windowed)) == 2
    assert windowed == hue
