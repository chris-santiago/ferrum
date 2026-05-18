"""Tests for JointChart and ClusterMapChart interactive grid layouts.

B7: JointChart._render_interactive should render a 2x2 grid, not flat horizontal.
W24: ClusterMapChart._render_interactive should render a 2x2 grid, not flat horizontal.
"""

from __future__ import annotations

import json

import polars as pl

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
        """With center + top + right, merged scene must reflect 2x2 grid dimensions.

        In a flat horizontal layout, width = center_w + top_w + right_w + 2*spacing.
        In a proper grid:
          width  = max(top_w, center_w) + spacing + right_w
          height = top_h + spacing + max(center_h, right_h)

        Since all children are 200x200 here, the grid is 2 cols and 2 rows:
          width  = 200 + spacing + 200  (col0 = max(top,center)=200, col1 = right=200)
          height = 200 + spacing + 200  (row0 = top=200, row1 = max(center,right)=200)

        The flat horizontal would give width = 200+200+200+2*spacing = 620 (spacing=10).
        """
        center = _simple_chart(200, 200)
        top = _simple_chart(200, 200)
        right = _simple_chart(200, 200)
        jc = fm.JointChart(center, top=top, right=right, spacing=10.0)

        scene_json, packed = jc._render_interactive()
        scene = json.loads(scene_json)

        # Grid layout: width should NOT be 3*200 + 2*10 = 620 (flat horizontal)
        # It should be closer to 200 + 10 + 200 = 410 (2 cols)
        assert scene["width"] < 600, (
            f"Width {scene['width']} suggests flat horizontal layout, not 2x2 grid"
        )
        # Height should NOT be max(200, 200, 200) = 200 (flat horizontal)
        # It should be closer to 200 + 10 + 200 = 410 (2 rows)
        assert scene["height"] > 300, (
            f"Height {scene['height']} suggests flat horizontal layout, not 2x2 grid"
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
    """ClusterMapChart._render_interactive must produce a 2x2 grid, not flat horizontal."""

    def test_grid_dimensions(self):
        """With heatmap + row_dendro + col_dendro, layout should be a 2x2 grid.

        SVG layout:
          - col_dendrogram at (0, 1) -- above heatmap
          - row_dendrogram at (1, 0) -- left of heatmap
          - heatmap        at (1, 1) -- main content

        Grid dimensions:
          width  = row_dendro_w + spacing + max(col_dendro_w, heatmap_w)
          height = col_dendro_h + spacing + max(row_dendro_h, heatmap_h)

        Flat horizontal would give width = heatmap_w + row_w + col_w + 2*spacing.
        """
        heatmap = _simple_chart(200, 200)
        row_dendro = _simple_chart(100, 200)
        col_dendro = _simple_chart(200, 100)
        cmc = fm.ClusterMapChart(
            heatmap,
            row_dendrogram=row_dendro,
            col_dendrogram=col_dendro,
            spacing=10.0,
        )

        scene_json, packed = cmc._render_interactive()
        scene = json.loads(scene_json)

        # In flat horizontal: width = 200+100+200+20 = 520, height = 200
        # In grid: width = 100+10+200 = 310, height = 100+10+200 = 310
        assert scene["width"] < 500, (
            f"Width {scene['width']} suggests flat horizontal, not 2x2 grid"
        )
        assert scene["height"] > 200, (
            f"Height {scene['height']} suggests flat horizontal, not 2x2 grid"
        )

    def test_panel_offsets_grid(self):
        """Panels must be placed at grid positions, not in a flat row."""
        heatmap = _simple_chart(200, 200)
        row_dendro = _simple_chart(100, 200)
        col_dendro = _simple_chart(200, 100)
        cmc = fm.ClusterMapChart(
            heatmap,
            row_dendrogram=row_dendro,
            col_dendrogram=col_dendro,
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
        heatmap = _simple_chart(200, 200)
        row_dendro = _simple_chart(100, 200)
        cmc = fm.ClusterMapChart(
            heatmap,
            row_dendrogram=row_dendro,
            spacing=10.0,
        )

        scene_json, _ = cmc._render_interactive()
        scene = json.loads(scene_json)

        # Width should reflect side-by-side arrangement
        assert scene["width"] > 200, f"Width {scene['width']} too small for horizontal arrangement"

    def test_heatmap_plus_col_only(self):
        """ClusterMapChart with heatmap + col_dendro (no row) arranges vertically.

        Grid positions: col_dendro at (0, 0), heatmap at (1, 0). Single column.
        """
        heatmap = _simple_chart(200, 200)
        col_dendro = _simple_chart(200, 100)
        cmc = fm.ClusterMapChart(
            heatmap,
            col_dendrogram=col_dendro,
            spacing=10.0,
        )

        scene_json, _ = cmc._render_interactive()
        scene = json.loads(scene_json)

        # Height should reflect vertical stacking
        assert scene["height"] > 200, f"Height {scene['height']} too small for vertical stack"
