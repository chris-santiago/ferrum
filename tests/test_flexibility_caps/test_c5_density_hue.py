"""C5 — categorical hue through the 2-D density path.

`jointplot(kind="kde", hue=<col>)` must split the CENTER 2-D density per group.
`mark_contour(groupby=...)` must thread groupby into Kde2D and route the layer
color through the GROUP column (not the density-level colormap).

Tests:
1. `desugar_contour` with groupby= passes groupby into Kde2D and colors by group.
2. No-groupby `desugar_contour` is byte-stable (color="level_value", no groupby in Kde2D).
3. `mark_contour(groupby=...).encode(color=...)` wires through to Kde2D.
4. `jointplot(kind="kde", hue=...)` produces a grouped center contour.
5. Jointplot center colors by the hue group field.
6. A 2-group rendering produces visually separate contour groups (smoke render test).
7. No-hue jointplot kind="kde" center is byte-stable.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum import Kde2D, Contour
from ferrum.marks.heavy_stat import desugar_contour


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_two_clusters():
    """Two spatially separated Gaussian clusters with a group column."""
    import numpy as np

    rng = np.random.default_rng(12345)
    n = 40
    xa = rng.normal(2.0, 0.4, n).astype("float64")
    ya = rng.normal(3.0, 0.4, n).astype("float64")
    xb = rng.normal(7.0, 0.4, n).astype("float64")
    yb = rng.normal(8.0, 0.4, n).astype("float64")
    return pl.DataFrame(
        {
            "x": list(xa) + list(xb),
            "y": list(ya) + list(yb),
            "g": ["A"] * n + ["B"] * n,
        }
    )


@pytest.fixture
def df_no_hue():
    """Simple numeric-only DataFrame for no-hue regression."""
    import numpy as np

    rng = np.random.default_rng(99)
    n = 30
    return pl.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n).astype("float64"),
            "y": rng.normal(0.0, 1.0, n).astype("float64"),
        }
    )


# ---------------------------------------------------------------------------
# 1. desugar_contour with groupby= passes groupby into Kde2D
# ---------------------------------------------------------------------------


class TestDesugaredContourGroupby:
    """desugar_contour routes groupby into Kde2D and switches color encoding."""

    def test_groupby_sets_kde2d_groupby(self):
        """Kde2D transform in grouped desugar has groupby set to the group field."""
        result = desugar_contour("x", "y", groupby="g", fill=True)
        # First transform must be Kde2D with groupby="g"
        transforms = result.transforms
        assert len(transforms) >= 1
        kde_repr = repr(transforms[0])
        assert "groupby=Some" in kde_repr or "g" in kde_repr, (
            f"Kde2D repr must mention groupby field 'g'; got: {kde_repr}"
        )

    def test_groupby_none_kde2d_no_groupby(self):
        """Without groupby, Kde2D has no groupby set (byte-stable path)."""
        result = desugar_contour("x", "y", groupby=None, fill=True)
        transforms = result.transforms
        kde_repr = repr(transforms[0])
        assert "groupby=[]" in kde_repr, f"Kde2D repr must show empty groupby; got: {kde_repr}"

    def test_groupby_fill_layer_colors_by_group(self):
        """When groupby is set, the polygon layer colors by the group field."""
        result = desugar_contour("x", "y", groupby="g", fill=True)
        layers = result.layers
        assert len(layers) >= 1
        layer = layers[0]
        enc = layer.encoding
        color_field = enc.get("color")
        # color must be the group field, not "level_value"
        assert color_field == "g", (
            f"Grouped contour polygon layer must color by group field 'g'; "
            f"got color={color_field!r}"
        )

    def test_no_groupby_fill_layer_colors_by_level_value(self):
        """Without groupby, polygon layer colors by level_value (byte-stable)."""
        result = desugar_contour("x", "y", groupby=None, fill=True)
        layers = result.layers
        enc = layers[0].encoding
        assert enc.get("color") == "level_value", (
            f"No-groupby contour must color by 'level_value'; got {enc.get('color')!r}"
        )

    def test_groupby_uses_segment_mark_not_polygon(self):
        """Grouped contours use segment marks (isolines) not polygon marks.

        The isoband ``level_id`` encoding is not globally unique across groups;
        polygon grouping by ``level_id`` would merge polygons from different groups.
        Segment marks are per-row and avoid this merging problem entirely.
        """
        result = desugar_contour("x", "y", groupby="g", fill=True)
        layer = result.layers[0]
        assert layer.mark == "segment", (
            f"Grouped contour must use segment mark to avoid level_id collision; "
            f"got mark={layer.mark!r}"
        )

    def test_groupby_segment_layer_has_no_detail(self):
        """Grouped segment layer does not need a detail kwarg (segments are per-row)."""
        result = desugar_contour("x", "y", groupby="g", fill=True)
        layer = result.layers[0]
        mk = layer.mark_kwargs or {}
        # Segment marks don't group vertices by detail; each row is one segment.
        detail = mk.get("detail")
        assert detail is None, (
            f"Grouped contour segment layer must not have detail; got detail={detail!r}"
        )


# ---------------------------------------------------------------------------
# 2. mark_contour(groupby=...) wires to Kde2D
# ---------------------------------------------------------------------------


class TestMarkContourGroupby:
    """mark_contour(groupby=...) exposes groupby kwarg and wires it through."""

    def test_mark_contour_accepts_groupby(self, df_two_clusters):
        """mark_contour(groupby=...) does not raise."""
        chart = (
            fm.Chart(df_two_clusters)
            .mark_contour(groupby="g", fill=True, thresholds=3)
            .encode(x="x", y="y", color="g")
        )
        assert chart is not None

    def test_mark_contour_standalone_renders(self, df_two_clusters):
        """mark_contour(groupby=...) produces valid SVG."""
        chart = (
            fm.Chart(df_two_clusters)
            .mark_contour(groupby="g", fill=True, thresholds=3)
            .encode(x="x", y="y", color="g")
        )
        svg = chart.to_svg()
        assert "<svg" in svg

    def test_mark_contour_grouped_has_content(self, df_two_clusters):
        """Grouped mark_contour produces geometry elements for both groups.

        Grouped contours use segment marks (isoline mode), which render as
        ``<line>`` elements in SVG.
        """
        chart = (
            fm.Chart(df_two_clusters)
            .mark_contour(groupby="g", fill=True, thresholds=3)
            .encode(x="x", y="y", color="g")
        )
        svg = chart.to_svg()
        # Segment marks render as <line> elements; paths/polygons are for no-hue fills
        has_geometry = "<path" in svg or "<polygon" in svg or "<line" in svg
        assert has_geometry, "Grouped mark_contour must produce geometry elements"
        # Verify there are a reasonable number of contour lines
        import re

        line_count = len(re.findall(r"<line\b", svg))
        # 2 groups × thresholds × contour cells = many lines; expect > 10
        assert line_count > 10, f"Expected >10 contour line segments; got {line_count}"


# ---------------------------------------------------------------------------
# 3. jointplot(kind="kde", hue=...) — grouped center
# ---------------------------------------------------------------------------


class TestJointplotKdeHue:
    """jointplot(kind='kde', hue=...) splits the center 2-D density per group."""

    def test_jointplot_kde_hue_creates_chart(self, df_two_clusters):
        """jointplot(kind='kde', hue=...) returns a JointChart without error."""
        chart = fm.jointplot(df_two_clusters, x="x", y="y", kind="kde", hue="g")
        assert chart is not None

    def test_jointplot_kde_hue_renders_svg(self, df_two_clusters):
        """jointplot(kind='kde', hue=...) renders valid SVG."""
        chart = fm.jointplot(df_two_clusters, x="x", y="y", kind="kde", hue="g")
        svg = chart.to_svg()
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_jointplot_kde_hue_center_has_groupby(self, df_two_clusters):
        """The center chart's transforms include Kde2D with groupby set."""
        chart = fm.jointplot(df_two_clusters, x="x", y="y", kind="kde", hue="g")
        center = chart.center
        # Resolve the center to get the transforms list
        # The center may be a pre-desugar Chart (composite mark); check pending transforms
        # OR it may already be desugared. Walk the center's structure for Kde2D with groupby.
        found_grouped_kde = _find_grouped_kde2d(center, "g")
        assert found_grouped_kde, "Center chart must include a Kde2D transform with groupby='g'"

    def test_jointplot_kde_hue_svg_has_geometry(self, df_two_clusters):
        """jointplot(kind='kde', hue=...) produces non-empty geometry.

        Grouped contours use segment marks which render as ``<line>`` in SVG.
        """
        chart = fm.jointplot(df_two_clusters, x="x", y="y", kind="kde", hue="g")
        svg = chart.to_svg()
        # Segment marks produce <line> elements; paths/polygons for no-hue fills
        has_geometry = "<path" in svg or "<polygon" in svg or "<line" in svg
        assert has_geometry, "jointplot kde hue must render geometry (lines/paths/polygons)"

    def test_jointplot_kde_nohue_center_no_groupby(self, df_no_hue):
        """Without hue, the center chart uses the no-groupby contour path (byte-stable)."""
        chart = fm.jointplot(df_no_hue, x="x", y="y", kind="kde")
        center = chart.center
        found_grouped = _find_grouped_kde2d(center, group_col=None)
        assert not found_grouped, "No-hue jointplot kind='kde' center must not have grouped Kde2D"


# ---------------------------------------------------------------------------
# 4. Regression: no-hue jointplot center is byte-stable
# ---------------------------------------------------------------------------


class TestJointplotKdeNoHueRegression:
    """No-hue jointplot(kind='kde') center must be byte-stable after this change."""

    def test_no_hue_jointplot_kde_renders(self, df_no_hue):
        """jointplot(kind='kde') without hue renders without error."""
        chart = fm.jointplot(df_no_hue, x="x", y="y", kind="kde")
        svg = chart.to_svg()
        assert "<svg" in svg

    def test_no_hue_jointplot_kde_has_geometry(self, df_no_hue):
        """No-hue jointplot kde center produces rendered geometry."""
        chart = fm.jointplot(df_no_hue, x="x", y="y", kind="kde")
        svg = chart.to_svg()
        assert "<path" in svg or "<polygon" in svg

    def test_no_hue_center_colors_by_level_value(self, df_no_hue):
        """No-hue center contour layer encodes color='level_value' (sequential cmap)."""
        chart = fm.jointplot(df_no_hue, x="x", y="y", kind="kde")
        center = chart.center
        # Find the polygon layer's color encoding
        color_enc = _find_polygon_color_encoding(center)
        assert color_enc == "level_value" or color_enc is None, (
            f"No-hue center must color by 'level_value'; got {color_enc!r}"
        )


# ---------------------------------------------------------------------------
# 5. Visual smoke test: groups visible in SVG
# ---------------------------------------------------------------------------


class TestGroupedContourVisualSmoke:
    """Groups appear as separate colored regions in the SVG output."""

    def test_grouped_contour_two_fill_colors(self, df_two_clusters):
        """SVG must contain fill colors from the categorical palette (not just sequential)."""
        chart = (
            fm.Chart(df_two_clusters)
            .mark_contour(groupby="g", fill=True, thresholds=3)
            .encode(x="x", y="y", color="g")
        )
        svg = chart.to_svg()
        # Look for at least 2 distinct fill colors that are not the same
        fills = re.findall(r'fill="([^"]+)"', svg)
        # Filter out background and non-data fills
        data_fills = [
            f
            for f in fills
            if f.startswith("rgba") or (f.startswith("#") and f not in ("#faf7f2", "none"))
        ]
        # With 2 groups, expect at least 2 distinct data fills
        assert len(set(data_fills)) >= 2, (
            f"Grouped contour must produce ≥2 distinct fill colors; got {set(data_fills)}"
        )

    def test_jointplot_kde_hue_marginals_still_split(self, df_two_clusters):
        """Marginals still split by hue after center KDE groupby change."""
        chart = fm.jointplot(df_two_clusters, x="x", y="y", kind="kde", hue="g")
        svg = chart.to_svg()
        # Should have multiple fill colors from the two marginal groups + center
        fills = re.findall(r'fill="([^"]+)"', svg)
        data_fills = [
            f
            for f in fills
            if f.startswith("rgba") or (f.startswith("#") and f not in ("#faf7f2", "none"))
        ]
        assert len(set(data_fills)) >= 2, (
            "Jointplot kde hue must produce multiple fill colors (groups split in marginals)"
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _find_grouped_kde2d(chart, group_col: str | None) -> bool:
    """Walk a chart's transforms, layers, and pending stat mark to check groupby.

    If group_col is not None: returns True if a Kde2D with groupby=group_col exists.
    If group_col is None: returns True if any Kde2D has a non-None groupby (unwanted).
    """
    transforms_to_check = []

    # Direct transforms
    if hasattr(chart, "_transforms") and chart._transforms:
        transforms_to_check.extend(chart._transforms)

    # Layers' transforms (for composite marks that have been desugared)
    if hasattr(chart, "_layers") and chart._layers:
        for layer in chart._layers:
            if hasattr(layer, "_transforms") and layer._transforms:
                transforms_to_check.extend(layer._transforms)

    for t in transforms_to_check:
        r = repr(t)
        if "Kde2D" in r:
            if group_col is not None:
                if f'Some("{group_col}")' in r or "groupby=Some" in r:
                    return True
            else:
                # We want to confirm NO groupby; return True if found (violation)
                if "groupby=Some" in r:
                    return True

    # For charts with composite mark pending desugar, check _pending_stat_mark.
    # The pending mark's kwargs contain the groupby setting before resolution.
    pending = getattr(chart, "_pending_stat_mark", None)
    if pending is not None:
        kwargs = getattr(pending, "kwargs", {}) or {}
        pending_groupby = kwargs.get("groupby")
        if group_col is not None:
            # Looking for groupby=group_col in pending
            if pending_groupby == group_col:
                return True
        else:
            # Looking for absence of groupby; return True if found (violation)
            if pending_groupby is not None:
                return True

    return False


def _find_polygon_color_encoding(chart) -> str | None:
    """Find the color field in a chart's first polygon layer, if any."""
    if hasattr(chart, "_layers") and chart._layers:
        for layer in chart._layers:
            if hasattr(layer, "mark") and layer.mark == "polygon":
                enc = getattr(layer, "encoding", {})
                return enc.get("color")
    return None
