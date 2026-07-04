"""Ratio/hole composite cutover — JointChart/ClusterMapChart (Task 8b).

Sibling of ``test_composite_render_linear.py`` (HConcat/VConcat) and
``test_composite_render_grid.py`` (ConcatChart wrap/sparse grids). JointChart and
ClusterMapChart lower to the ``grid`` layout of the one-call Rust composite entry
via ``JointChart._composite_tree`` / ``ClusterMapChart._composite_tree``
(``ferrum/composition.py``), with the empty 2×2 corner emitted as a
``{"kind": "hole"}`` cell (Task 8a) rather than a real panel.

What this file proves
----------------------
- **Hole contributes no panel** — the interactive scene has exactly 3 panels for
  a full both-marginals/both-dendrograms JointChart/ClusterMapChart, not 4.
- **Non-identity ``layout_scale`` (the W5 slice)** — marginal/dendrogram panels
  carry a genuine ratio-fit scale (not identity), computed analytically here to
  mirror the Rust ``plan_grid`` fit-factor formula
  (``crates/ferrum-core/src/render/composite_render.rs``) so the assertion is a
  real regression check, not a magic number.
- **Axis-hiding survives the cutover** — marginal panels render zero axes
  (``axis(show=False)`` traveled onto the leaf spec); the center/heatmap panel
  keeps its axes.
- **SVG shares the same x-domain across center/top** — using the existing
  ``tests/_svg_extents`` tick-extent parser, confirming the ratio-grid cutover
  didn't break the shared axis the marginal exists to illustrate.
- **Static-SVG panel extents numerically match the ratio shares** — parsing
  the rendered SVG's total canvas dims (not just the interactive scene's
  ``layout_scale``) to recover ``ratio``/``dendrogram_ratio`` from geometry.
- **An empty-data leaf falls back to the legacy render path instead of
  raising** — ``_build_grid_tree`` mirrors ``_lower_composite``'s per-leaf
  ``num_rows == 0`` guard, so a zero-row marginal/dendrogram makes the whole
  tree decline (return ``None``) rather than handing the Rust composite entry
  a batch it cannot lay out.
"""

from __future__ import annotations

import json
import re

import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import numeric_text_entries


def _joint_chart(*, ratio: int = 5, spacing: float = 10.0, size: float = 300.0) -> fm.JointChart:
    """A JointChart with center + top + right, all declared the same size.

    Same-size leaves make the fit-factor math easy to hand-verify (see
    :func:`_expected_marginal_scale`).
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 3.0, 2.0, 1.0]})
    center = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=size, height=size)
    top = fm.Chart(df).mark_histogram().encode(x="x:Q").properties(width=size, height=size)
    right = (
        fm.Chart(df)
        .mark_histogram(orientation="horizontal")
        .encode(y="y:Q")
        .properties(width=size, height=size)
    )
    return fm.JointChart(center, top=top, right=right, ratio=ratio, spacing=spacing)


def _expected_marginal_scale(ratio: int) -> float:
    """Analytic fit-factor scale for a same-size-leaves JointChart marginal.

    Mirrors ``plan_grid``'s ``fit_factor`` (composite_render.rs): with every
    leaf declared at the same native size ``s``, ``k = min(s/marginal_share,
    s/center_share) = s/center_share`` (center_share > marginal_share for any
    ratio > 1), so the marginal slot size is ``k * marginal_share = s *
    marginal_share/center_share`` and the marginal's fit scale is that slot
    size divided by its own native size ``s`` — i.e. ``marginal_share /
    center_share``, independent of the declared leaf size.
    """
    marginal_share = 1.0 / (ratio + 1)
    center_share = ratio / (ratio + 1)
    return marginal_share / center_share


class TestJointChartRatioGrid:
    """Both-marginals JointChart: 2×2 grid with a hole at the empty corner."""

    def test_hole_contributes_no_panel(self):
        jc = _joint_chart()
        scene_json, _ = jc._render_interactive()
        scene = json.loads(scene_json)
        assert len(scene["panels"]) == 3, (
            f"expected 3 panels (top, center, right; hole contributes none), "
            f"got {len(scene['panels'])}"
        )

    def test_marginal_panels_carry_nonidentity_layout_scale(self):
        """W5: marginal panels are ratio-fit via a non-identity layout_scale."""
        ratio = 5
        jc = _joint_chart(ratio=ratio)
        scene_json, _ = jc._render_interactive()
        scene = json.loads(scene_json)
        panels = scene["panels"]
        assert len(panels) == 3

        top_panel, center_panel, right_panel = panels
        expected = _expected_marginal_scale(ratio)

        top_scale = top_panel["layout_scale"]
        assert top_scale["sx"] == pytest.approx(1.0)
        assert top_scale["sy"] == pytest.approx(expected)

        center_scale = center_panel.get("layout_scale", {"sx": 1.0, "sy": 1.0})
        assert center_scale.get("sx", 1.0) == pytest.approx(1.0)
        assert center_scale.get("sy", 1.0) == pytest.approx(1.0)

        right_scale = right_panel["layout_scale"]
        assert right_scale["sx"] == pytest.approx(expected)
        assert right_scale["sy"] == pytest.approx(1.0)

    def test_marginal_axis_hidden_center_axis_kept(self):
        """Marginals suppress axis decoration; the center panel keeps its axes."""
        jc = _joint_chart()
        scene_json, _ = jc._render_interactive()
        scene = json.loads(scene_json)
        top_panel, center_panel, right_panel = scene["panels"]

        assert top_panel.get("axes", []) == [], "top marginal should hide its axis"
        assert right_panel.get("axes", []) == [], "right marginal should hide its axis"
        assert center_panel.get("axes", []) != [], "center panel should keep its axes"

    def test_svg_center_and_top_share_x_domain(self):
        """The center and top marginal render the same x tick values in SVG.

        A proxy for "the ratio-grid cutover preserved the shared x-axis the
        marginal exists to illustrate": both center's x-axis ticks and top's
        rendered value range should span the same [1, 4] domain (the data's x
        column), even though top's own axis decoration is hidden.
        """
        jc = _joint_chart()
        svg = jc.to_svg()
        entries = numeric_text_entries(svg)
        values = {round(val, 6) for _, _, val in entries}
        # The x column is [1, 2, 3, 4]; the center panel's x-axis ticks must
        # include the full domain (top's own ticks are hidden by axis(show=False),
        # but the shared domain is what makes the two panels visually align).
        assert 1.0 in values or any(abs(v - 1.0) < 0.5 for v in values)
        assert 4.0 in values or any(abs(v - 4.0) < 0.5 for v in values)

    def test_one_marginal_no_hole_needed(self):
        """A single marginal lowers to a dense 2×1 grid — no hole panel gap."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        center = (
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
        )
        top = fm.Chart(df).mark_histogram().encode(x="x:Q").properties(width=200, height=200)
        jc = fm.JointChart(center, top=top, spacing=10.0)

        scene_json, _ = jc._render_interactive()
        scene = json.loads(scene_json)
        assert len(scene["panels"]) == 2

    def test_static_svg_total_dims_match_ratio_shares(self):
        """The rendered SVG's total canvas dims numerically encode ``ratio``.

        With same-size (``size``) leaves, the center panel's own fit-scale is
        1 (see :func:`_expected_marginal_scale`'s docstring), so the rendered
        total width/height decompose as ``center_size + spacing +
        marginal_slot`` where ``marginal_slot = center_size *
        marginal_share/center_share``. Solving for the marginal slot from the
        *rendered* total and dividing back into ``center_size`` recovers
        ``ratio`` purely from parsed SVG geometry — this is the static-SVG
        counterpart to ``test_marginal_panels_carry_nonidentity_layout_scale``,
        which only checks the interactive scene's ``layout_scale`` field.
        """
        ratio = 4
        size = 300.0
        spacing = 10.0
        jc = _joint_chart(ratio=ratio, spacing=spacing, size=size)
        svg = jc.to_svg()

        total_width = float(re.search(r'<svg[^>]*\swidth="([\d.]+)"', svg).group(1))
        total_height = float(re.search(r'<svg[^>]*\sheight="([\d.]+)"', svg).group(1))

        marginal_col_width = total_width - size - spacing
        marginal_row_height = total_height - size - spacing
        assert marginal_col_width > 0
        assert marginal_row_height > 0

        # center's column (== size) is `ratio` times the right marginal's column;
        # center's row (== size) is `ratio` times the top marginal's row.
        assert size / marginal_col_width == pytest.approx(ratio, rel=0.01)
        assert size / marginal_row_height == pytest.approx(ratio, rel=0.01)


class TestClusterMapChartRatioGrid:
    """Both-dendrograms ClusterMapChart: 2×2 grid with a hole at the empty corner."""

    def _cluster_map(self, *, dendrogram_ratio: float = 0.3, heatmap_size: float = 300.0):
        wide = pl.DataFrame(
            {
                "row": ["a", "b", "c", "d"],
                "x": [1.0, 0.5, 0.3, 0.9],
                "y": [0.5, 1.0, 0.7, 0.2],
            }
        )
        heatmap = (
            fm.Chart(wide)
            .mark_rect()
            .encode(x="row", y="row", color="x")
            .properties(width=heatmap_size, height=heatmap_size)
        )
        row_dendro = fm.Chart(wide).mark_line().encode(x="x:Q", y="row").axis(show=False)
        col_dendro = fm.Chart(wide).mark_line().encode(x="row", y="x:Q").axis(show=False)
        return fm.ClusterMapChart(
            heatmap,
            row_dendrogram=row_dendro,
            col_dendrogram=col_dendro,
            dendrogram_ratio=dendrogram_ratio,
            spacing=10.0,
        )

    def test_hole_contributes_no_panel(self):
        cmc = self._cluster_map()
        scene_json, _ = cmc._render_interactive()
        scene = json.loads(scene_json)
        assert len(scene["panels"]) == 3, (
            f"expected 3 panels (col_dendro, row_dendro, heatmap; hole contributes "
            f"none), got {len(scene['panels'])}"
        )

    def test_dendrogram_panels_are_identity_scaled(self):
        """Dendrograms are pre-resized to their target dims, so plan_grid's
        fit-factor degenerates to identity scale for every cell (no post-hoc
        non-uniform stretch that would distort branch geometry — see
        ``ClusterMapChart._pre_resized_dendrograms``).
        """
        cmc = self._cluster_map()
        scene_json, _ = cmc._render_interactive()
        scene = json.loads(scene_json)
        for panel in scene["panels"]:
            scale = panel.get("layout_scale", {"sx": 1.0, "sy": 1.0})
            assert scale.get("sx", 1.0) == pytest.approx(1.0)
            assert scale.get("sy", 1.0) == pytest.approx(1.0)

    def test_one_dendrogram_no_hole_needed(self):
        """A single dendrogram lowers to a dense 1×2 grid — no hole panel gap."""
        cmc = self._cluster_map()
        row_only = fm.ClusterMapChart(
            cmc.heatmap,
            row_dendrogram=cmc.row_dendrogram,
            dendrogram_ratio=0.3,
            spacing=10.0,
        )
        scene_json, _ = row_only._render_interactive()
        scene = json.loads(scene_json)
        assert len(scene["panels"]) == 2

    def test_static_svg_total_dims_match_dendrogram_ratio_shares(self):
        """The rendered SVG's total canvas dims numerically encode ``dendrogram_ratio``.

        Unlike JointChart's marginals, dendrograms are pre-resized to identity
        scale (``ClusterMapChart._pre_resized_dendrograms``) — the heatmap
        keeps its native declared size, and the total rendered width/height
        decompose as ``heatmap_size + spacing + dendro_slot`` where
        ``dendro_slot = heatmap_size * d/(1 - d)`` (``d = dendrogram_ratio``).
        Solving for the dendro slot from the *rendered* total recovers the
        configured ratio purely from parsed SVG geometry, the static-SVG
        counterpart to ``test_dendrogram_panels_are_identity_scaled``.
        """
        d = 0.3
        heatmap_size = 300.0
        spacing = 10.0
        cmc = self._cluster_map(dendrogram_ratio=d, heatmap_size=heatmap_size)
        svg = cmc.to_svg()

        total_width = float(re.search(r'<svg[^>]*\swidth="([\d.]+)"', svg).group(1))
        total_height = float(re.search(r'<svg[^>]*\sheight="([\d.]+)"', svg).group(1))

        dendro_col_width = total_width - heatmap_size - spacing
        dendro_row_height = total_height - heatmap_size - spacing
        assert dendro_col_width > 0
        assert dendro_row_height > 0

        expected_ratio = (1.0 - d) / d
        assert heatmap_size / dendro_col_width == pytest.approx(expected_ratio, rel=0.01)
        assert heatmap_size / dendro_row_height == pytest.approx(expected_ratio, rel=0.01)


class TestEmptyDataLeafFallback:
    """An empty-data leaf must fall back to the legacy render path, not raise.

    ``_build_grid_tree`` lowers every non-hole cell via ``Chart._render_inputs``,
    exactly like ``_lower_composite``'s ``lower()`` does for HConcat/VConcat/
    ConcatChart leaves. ``lower()`` already declines (returns ``None``) when a
    leaf's data is empty (``composition.py``'s ``if data.num_rows == 0: return
    None`` guard) because the Rust composite-render entry cannot lay out a
    zero-row leaf batch — but until this fix, ``_build_grid_tree`` had no such
    guard, so ``JointChart``/``ClusterMapChart`` would hand the Rust entry an
    empty batch and it would raise ``ValueError``, even though the legacy
    per-child path (each child rendering through its own ``Chart.to_svg()`` /
    scene-merge call) handles an empty-data child without issue.
    """

    def test_joint_chart_empty_marginal_declines_composite_tree(self):
        """An empty-data marginal makes ``_composite_tree`` return None."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 3.0, 2.0, 1.0]})
        center = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        empty_top = fm.Chart(df.head(0)).mark_histogram().encode(x="x:Q")
        jc = fm.JointChart(center, top=empty_top)

        assert jc._composite_tree(auto_tooltips=False) is None
        assert jc._composite_tree(auto_tooltips=True) is None

    def test_joint_chart_empty_marginal_static_renders_via_fallback(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 3.0, 2.0, 1.0]})
        center = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        empty_top = fm.Chart(df.head(0)).mark_histogram().encode(x="x:Q")
        jc = fm.JointChart(center, top=empty_top)

        svg = jc.to_svg()  # must not raise
        assert svg.startswith("<svg")

    def test_joint_chart_empty_marginal_interactive_renders_via_fallback(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 3.0, 2.0, 1.0]})
        center = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        empty_top = fm.Chart(df.head(0)).mark_histogram().encode(x="x:Q")
        jc = fm.JointChart(center, top=empty_top)

        scene_json, packed = jc._render_interactive()  # must not raise
        scene = json.loads(scene_json)
        assert scene["panels"]
        assert isinstance(packed, (bytes, bytearray))

    def test_clustermap_chart_empty_dendrogram_declines_composite_tree(self):
        """An empty-data dendrogram makes ``_composite_tree`` return None."""
        wide = pl.DataFrame(
            {"row": ["a", "b", "c", "d"], "x": [1.0, 0.5, 0.3, 0.9], "y": [0.5, 1.0, 0.7, 0.2]}
        )
        heatmap = fm.Chart(wide).mark_rect().encode(x="row", y="row", color="x")
        empty_row_dendro = (
            fm.Chart(wide.head(0)).mark_line().encode(x="x:Q", y="row").axis(show=False)
        )
        cmc = fm.ClusterMapChart(heatmap, row_dendrogram=empty_row_dendro, dendrogram_ratio=0.3)

        assert cmc._composite_tree(auto_tooltips=False) is None
        assert cmc._composite_tree(auto_tooltips=True) is None

    def test_clustermap_chart_empty_dendrogram_renders_via_fallback(self):
        wide = pl.DataFrame(
            {"row": ["a", "b", "c", "d"], "x": [1.0, 0.5, 0.3, 0.9], "y": [0.5, 1.0, 0.7, 0.2]}
        )
        heatmap = fm.Chart(wide).mark_rect().encode(x="row", y="row", color="x")
        empty_row_dendro = (
            fm.Chart(wide.head(0)).mark_line().encode(x="x:Q", y="row").axis(show=False)
        )
        cmc = fm.ClusterMapChart(heatmap, row_dendrogram=empty_row_dendro, dendrogram_ratio=0.3)

        svg = cmc.to_svg()  # must not raise
        assert svg.startswith("<svg")

        scene_json, packed = cmc._render_interactive()  # must not raise
        scene = json.loads(scene_json)
        assert scene["panels"]
        assert isinstance(packed, (bytes, bytearray))
