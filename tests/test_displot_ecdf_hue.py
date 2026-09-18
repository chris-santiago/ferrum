"""Regression coverage for #200 (batch F wave 4 task 8).

``fm.displot(df, x=..., kind="ecdf", hue=...)`` raised ``ValueError: unknown
column 'g' referenced by an encoding`` at render: the ecdf branch built a raw
``Bin(...)`` transform without threading ``groupby=``, the one figure-layer
site not on the composite-mark auto-carry road
(``_desugar._prep_groupby_from_color``), so ``Bin``'s output batch silently
dropped the hue column the ``color`` channel still named.

Fix: ``src/ferrum/plots/distribution.py``'s ecdf branch threads
``groupby=_field_name(hue)`` into the ``Bin`` transform (mirroring the
``kind="kde"`` branch's explicit ``groupby=hue``), and sets
``shared_extent=True`` whenever a hue is bound -- unconditionally, not gated
on ``multiple=`` -- since an ECDF is always a per-group comparison and a
common x-grid is the only reading under which the curves are comparable (the
same contract a sibling task applies to ``multiple="stack"``/``"fill"``
histogram binning).
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import polylines, rug_tick_x1_by_stroke

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_group_df():
    """Two equal-size, well-separated normal samples tagged by ``g``.

    Equal group sizes keep each group's auto (Sturges) bin count identical,
    so under ``shared_extent=True`` both groups land on the exact same
    bin-edge grid -- the fixture that makes "shared x-grid" a checkable fact
    rather than an approximation.
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


# ---------------------------------------------------------------------------
# SVG parsing helpers. `_polylines` and `_rug_tick_x1_by_stroke` were written
# here as local copies with a DUPLICATION-PENDING note; they now come from
# `tests/_svg_rects.py`, the module that already owned the element regex for
# the sibling site (`test_displot_kde_overlay.py`). The copies deleted here
# read attributes positionally, so they also depended on the renderer's
# attribute order; the owner's parser reads them by name.
# ---------------------------------------------------------------------------


def _assert_finite_svg(svg: str) -> None:
    assert "NaN" not in svg
    assert "Infinity" not in svg


# ---------------------------------------------------------------------------
# #200: distinct-colored per-group step lines, sharing one x-grid.
# ---------------------------------------------------------------------------


class TestEcdfHueCarriesGroupKey:
    def test_ecdf_hue_renders_distinct_colored_lines_on_shared_grid(self, two_group_df):
        chart = fm.displot(two_group_df, x="x", hue="g", kind="ecdf")
        svg = chart.to_svg()
        _assert_finite_svg(svg)

        lines = polylines(svg)
        assert len(lines) == 2, f"expected one step line per hue level, got {len(lines)}"
        strokes = {stroke for stroke, _ in lines}
        assert len(strokes) == 2, f"step lines are not distinctly colored: {strokes}"

        # Shared x-grid: `shared_extent=True` forces both groups onto the
        # same bin edges, so (with this fixture's equal group sizes) the two
        # lines' x coordinates land on the exact same set of screen
        # positions. Without it (the pre-fix default `shared_extent=False`,
        # or a regression that drops it again), each group would bin over
        # its own data range -- and this fixture's two groups (means 0 and
        # 6) have non-overlapping ranges, so their bin edges would land at
        # entirely different x positions.
        xs_by_stroke = {stroke: {round(x, 2) for x, _ in points} for stroke, points in lines}
        stroke_a, stroke_b = sorted(xs_by_stroke)
        assert xs_by_stroke[stroke_a] == xs_by_stroke[stroke_b], (
            "per-group ECDF step lines do not share an x-grid: "
            f"{sorted(xs_by_stroke[stroke_a])} != {sorted(xs_by_stroke[stroke_b])}"
        )

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
        lines = polylines(svg)
        # Two hue levels x two facet panels.
        assert len(lines) == 4, f"expected 4 step lines (2 hue x 2 facets), got {len(lines)}"
        strokes = {stroke for stroke, _ in lines}
        assert len(strokes) == 2, f"facet panels do not reuse the same 2 hue colors: {strokes}"

    def test_ecdf_hue_with_kde_overlay_both_colored_by_group(self, two_group_df):
        chart = fm.displot(two_group_df, x="x", hue="g", kind="ecdf", kde=True)
        svg = chart.to_svg()
        _assert_finite_svg(svg)
        lines = polylines(svg)
        # One ECDF step line + one KDE overlay curve per hue level.
        assert len(lines) == 4, f"expected 4 lines (2 ecdf + 2 kde overlay), got {len(lines)}"
        strokes = {stroke for stroke, _ in lines}
        assert len(strokes) == 2, (
            f"kde overlay does not share the ecdf lines' 2 hue colors: {strokes}"
        )

    def test_ecdf_hue_with_rug_overlay_ticks_match_line_colors(self, two_group_df):
        chart = fm.displot(two_group_df, x="x", hue="g", kind="ecdf", rug=True)
        svg = chart.to_svg()
        _assert_finite_svg(svg)
        lines = polylines(svg)
        assert len(lines) == 2, f"expected 2 ecdf step lines, got {len(lines)}"
        line_strokes = {stroke for stroke, _ in lines}

        rug_ticks = rug_tick_x1_by_stroke(svg)
        assert set(rug_ticks) == line_strokes, (
            f"rug tick colors {set(rug_ticks)} do not match the ecdf lines' "
            f"hue colors {line_strokes}"
        )


# ---------------------------------------------------------------------------
# Clean-input contract: the no-hue path is unchanged by this fix.
# ---------------------------------------------------------------------------


class TestEcdfNoHueUnchanged:
    def test_ecdf_no_hue_wire_has_no_groupby_and_shared_extent_false(self):
        df = pl.DataFrame({"x": np.arange(20.0)})
        chart = fm.displot(df, x="x", kind="ecdf")
        spec = chart.to_dict()
        bin_xform = next(t for t in spec["transforms"] if t["type"] == "bin")
        assert "groupby" not in bin_xform
        assert bin_xform["shared_extent"] is False

    def test_ecdf_no_hue_render_byte_identical(self):
        """Pinned baseline for the no-hue ecdf render -- must stay
        byte-identical across this fix: with no hue, `_field_name(None)` is
        `None`, so `groupby=None, shared_extent=False` are exactly what the
        pre-fix raw `Bin(...)` call always produced.
        """
        df = pl.DataFrame({"x": np.arange(20.0)})
        svg = fm.displot(df, x="x", kind="ecdf").to_svg()
        digest = hashlib.sha256(svg.encode()).hexdigest()
        assert digest == ("983b966e9905d7504ca2692fdf2e0c6213dac6314206d8847c7d103757ae7251"), (
            "no-hue ecdf render moved -- expected byte-identity with the pre-fix baseline"
        )
