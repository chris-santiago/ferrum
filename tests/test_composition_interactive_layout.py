"""Tests for JointChart and ClusterMapChart interactive grid layouts.

B7: JointChart._render_interactive should render a 2x2 grid, not flat horizontal.
W24: ClusterMapChart._render_interactive should render a 2x2 grid, not flat horizontal.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm


def _simple_chart(width: float = 200.0, height: float = 200.0):
    """Return a minimal chart for composition tests."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=width, height=height)


# ---------------------------------------------------------------------------
# B7: JointChart interactive grid layout
# ---------------------------------------------------------------------------


class TestJointChartInteractiveGrid:
    """JointChart._render_interactive must produce a 2x2 grid, not flat horizontal."""

    def test_grid_dimensions_all_three(self):
        """With center + top + right, merged scene must reflect the 2x2 ratio grid.

        Post-8b, JointChart's interactive path routes through the composite grid
        entry and applies the same F20 ratio math as the SVG path
        (``marginal_share = 1/(ratio+1)``, ``center_share = ratio/(ratio+1)``;
        default ``ratio=5`` -> marginal_share=1/6, center_share=5/6) via each
        panel's non-identity ``layout_scale`` — this is the W5 fix: previously the
        interactive body kept every panel at native size (no ratio scaling at all).

        All three panels are 200x200 here:
          native_col_w = [max(top_w, center_w)=200, right_w=200]
          native_row_h = [top_h=200, max(center_h, right_h)=200]
          k_w = min(200/(5/6), 200/(1/6)) = min(240, 1200) = 240
          k_h = min(200/(1/6), 200/(5/6)) = min(1200, 240) = 240
          col_w = [240*5/6, 240*1/6] = [200, 40]
          row_h = [240*1/6, 240*5/6] = [40, 200]
          width  = col_w[0] + spacing + col_w[1] = 200 + 10 + 40  = 250
          height = row_h[0] + spacing + row_h[1] = 40  + 10 + 200 = 250

        A flat-horizontal layout would give width=620, height=200; a non-ratio
        dense grid (pre-8b interactive behavior) would give width=410, height=410.
        """
        center = _simple_chart(200, 200)
        top = _simple_chart(200, 200)
        right = _simple_chart(200, 200)
        jc = fm.JointChart(center, top=top, right=right, spacing=10.0)

        scene_json, packed = jc._render_interactive()
        scene = json.loads(scene_json)

        assert scene["width"] == pytest.approx(250.0, abs=1.0), (
            f"Width {scene['width']} does not match the ratio-scaled 2x2 grid"
        )
        assert scene["height"] == pytest.approx(250.0, abs=1.0), (
            f"Height {scene['height']} does not match the ratio-scaled 2x2 grid"
        )

    def test_panel_offsets_grid_placement(self):
        """Verify panels are placed at grid positions, not in a flat row.

        Grid layout (row, col):
          - top     at (0, 0): dx=0, dy=0
          - center  at (1, 0): dx=0, dy=top_h + spacing
          - right   at (1, 1): dx=center_w + spacing, dy=top_h + spacing

        In a flat horizontal, all panels would have dy=0.
        """
        center = _simple_chart(200, 200)
        top = _simple_chart(200, 100)  # shorter top marginal
        right = _simple_chart(100, 200)  # narrower right marginal
        jc = fm.JointChart(center, top=top, right=right, spacing=10.0)

        scene_json, _ = jc._render_interactive()
        scene = json.loads(scene_json)
        panels = scene["panels"]

        # We need at least 3 panels (one from each child).
        assert len(panels) >= 3, f"Expected >= 3 panels, got {len(panels)}"

        # Panel offsets are stored in panel["plot_area"]["y"].
        # In a flat horizontal, all panels would be at the same y.
        # In a grid, center and right panels (row 1) should have non-zero y.
        y_positions = [p.get("plot_area", {}).get("y", 0) for p in panels]
        assert max(y_positions) > 0, (
            "All panels at plot_area.y=0 suggests flat layout; grid should offset row-1 panels"
        )

    def test_center_only(self):
        """JointChart with only center (no marginals) renders directly."""
        center = _simple_chart(300, 250)
        jc = fm.JointChart(center)

        scene_json, packed = jc._render_interactive()
        scene = json.loads(scene_json)

        # Should render the center chart directly.
        assert len(scene["panels"]) >= 1

    def test_center_plus_top_only(self):
        """JointChart with center + top (no right) should stack vertically.

        Grid positions: top at (0, 0), center at (1, 0). Single column.
        Height = top_h + spacing + center_h. Width = max(top_w, center_w).
        """
        center = _simple_chart(200, 200)
        top = _simple_chart(200, 100)
        jc = fm.JointChart(center, top=top, spacing=10.0)

        scene_json, _ = jc._render_interactive()
        scene = json.loads(scene_json)

        # Height should reflect vertical stacking, not flat horizontal
        # top(100) + spacing(10) + center(200) ≈ 310-ish (scenes include axes/padding)
        assert scene["height"] > 200, (
            f"Height {scene['height']} too small for vertical stack of top + center"
        )

    def test_center_plus_right_only(self):
        """JointChart with center + right (no top) should stack horizontally.

        Grid positions: center at (0, 0), right at (0, 1). Single row.
        Width = center_w + spacing + right_w. Height = max(center_h, right_h).
        """
        center = _simple_chart(200, 200)
        right = _simple_chart(100, 200)
        jc = fm.JointChart(center, right=right, spacing=10.0)

        scene_json, _ = jc._render_interactive()
        scene = json.loads(scene_json)

        # Width should reflect horizontal arrangement
        # center(200) + spacing(10) + right(100) ≈ 310-ish
        assert scene["width"] > 200, (
            f"Width {scene['width']} too small for horizontal stack of center + right"
        )


# ---------------------------------------------------------------------------
# W24: ClusterMapChart interactive grid layout
# ---------------------------------------------------------------------------


class TestClusterMapChartInteractiveGrid:
    """ClusterMapChart._render_interactive must produce a 2x2 ratio grid, not flat horizontal.

    Post-8b, ``ClusterMapChart`` pre-resizes each dendrogram from the heatmap's own
    declared size and ``dendrogram_ratio`` — ``dendro_dim = heatmap_dim * d / (1 - d)``
    — *before* lowering to a leaf spec (see ``composition.py``
    ``_pre_resized_dendrograms``), exactly mirroring the pre-cutover SVG path. This
    means whatever width/height a test passes to ``row_dendrogram=``/
    ``col_dendrogram=`` directly is **discarded and overridden** by that derived
    value — only the heatmap's size and ``dendrogram_ratio`` matter. These fixtures
    use ``heatmap=300x300`` with an explicit ``dendrogram_ratio=0.3`` (rather than
    relying on the 0.2 default) so the derived dendrogram dimension
    (``300 * 0.3/0.7 ≈ 128.6``) comfortably clears the render pipeline's ~80px
    minimum-viewport floor (a pre-existing, unrelated limitation — a viewport
    dimension below roughly 60-80px renders zero panels; see the 8a report).
    """

    _HM = 300.0
    _RATIO = 0.3
    _DENDRO = _HM * _RATIO / (1.0 - _RATIO)  # ~128.57

    def test_grid_dimensions(self):
        """With heatmap + row_dendro + col_dendro, layout should be a 2x2 ratio grid.

        SVG layout:
          - col_dendrogram at (0, 1) -- above heatmap
          - row_dendrogram at (1, 0) -- left of heatmap
          - heatmap        at (1, 1) -- main content

        Both dendrograms are pre-resized to the same derived dimension (see class
        docstring), so the ratio grid's fit-factor math degenerates to identity
        scale (native extents already match their ratio-derived slots exactly):
          width  = dendro_dim + spacing + heatmap_w = 128.57 + 10 + 300 = 438.57
          height = dendro_dim + spacing + heatmap_h = 128.57 + 10 + 300 = 438.57

        Flat horizontal would give width = heatmap_w + row_w + col_w + 2*spacing
        (much larger); a non-ratio dense grid would give width=height=310.
        """
        heatmap = _simple_chart(self._HM, self._HM)
        row_dendro = _simple_chart(100, 200)
        col_dendro = _simple_chart(200, 100)
        cmc = fm.ClusterMapChart(
            heatmap,
            row_dendrogram=row_dendro,
            col_dendrogram=col_dendro,
            dendrogram_ratio=self._RATIO,
            spacing=10.0,
        )

        scene_json, packed = cmc._render_interactive()
        scene = json.loads(scene_json)

        expected = self._DENDRO + 10.0 + self._HM
        assert scene["width"] == pytest.approx(expected, abs=2.0), (
            f"Width {scene['width']} does not match the ratio-scaled 2x2 grid"
        )
        assert scene["height"] == pytest.approx(expected, abs=2.0), (
            f"Height {scene['height']} does not match the ratio-scaled 2x2 grid"
        )

    def test_panel_offsets_grid(self):
        """Panels must be placed at grid positions, not in a flat row."""
        heatmap = _simple_chart(self._HM, self._HM)
        row_dendro = _simple_chart(100, 200)
        col_dendro = _simple_chart(200, 100)
        cmc = fm.ClusterMapChart(
            heatmap,
            row_dendrogram=row_dendro,
            col_dendrogram=col_dendro,
            dendrogram_ratio=self._RATIO,
            spacing=10.0,
        )

        scene_json, _ = cmc._render_interactive()
        scene = json.loads(scene_json)
        panels = scene["panels"]

        assert len(panels) >= 3, f"Expected >= 3 panels, got {len(panels)}"

        # Panel offsets are stored in panel["plot_area"]["y"].
        # In a grid, heatmap and row_dendro (row 1) should have non-zero y.
        y_positions = [p.get("plot_area", {}).get("y", 0) for p in panels]
        assert max(y_positions) > 0, (
            "All panels at plot_area.y=0 suggests flat layout; grid should offset row-1 panels"
        )

    def test_heatmap_only(self):
        """ClusterMapChart with only heatmap renders directly."""
        heatmap = _simple_chart(300, 250)
        cmc = fm.ClusterMapChart(heatmap)

        scene_json, packed = cmc._render_interactive()
        scene = json.loads(scene_json)
        assert len(scene["panels"]) >= 1

    def test_heatmap_plus_row_only(self):
        """ClusterMapChart with heatmap + row_dendro (no col) arranges horizontally.

        Grid positions: row_dendro at (0, 0), heatmap at (0, 1). Single row.
        """
        heatmap = _simple_chart(self._HM, self._HM)
        row_dendro = _simple_chart(100, 200)
        cmc = fm.ClusterMapChart(
            heatmap,
            row_dendrogram=row_dendro,
            dendrogram_ratio=self._RATIO,
            spacing=10.0,
        )

        scene_json, _ = cmc._render_interactive()
        scene = json.loads(scene_json)

        # Width should reflect side-by-side arrangement
        assert scene["width"] > self._HM, (
            f"Width {scene['width']} too small for horizontal arrangement"
        )

    def test_heatmap_plus_col_only(self):
        """ClusterMapChart with heatmap + col_dendro (no row) arranges vertically.

        Grid positions: col_dendro at (0, 0), heatmap at (1, 0). Single column.
        """
        heatmap = _simple_chart(self._HM, self._HM)
        col_dendro = _simple_chart(200, 100)
        cmc = fm.ClusterMapChart(
            heatmap,
            col_dendrogram=col_dendro,
            dendrogram_ratio=self._RATIO,
            spacing=10.0,
        )

        scene_json, _ = cmc._render_interactive()
        scene = json.loads(scene_json)

        # Height should reflect vertical stacking
        assert scene["height"] > self._HM, f"Height {scene['height']} too small for vertical stack"
