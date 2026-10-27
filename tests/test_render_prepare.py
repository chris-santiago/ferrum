"""Tests for ``ferrum._render_prepare``'s per-channel field introspection
helpers (``_classify_field``) -- shared by
``Chart._resolve_chart_config``/``_render_inputs`` and by this module's own
``_extract_ordinal_domain`` single-chart lookup.
"""

from __future__ import annotations

import polars as pl

import ferrum
from ferrum._render_prepare import _classify_field


class TestLayerChartMergedSharedTemporalDomain:
    """``LayerChart``'s interactive render (always the merged flat chart,
    see ``composition.py::LayerChart._build_merged``) must share a temporal
    x/y across layers exactly as ``chart_a + chart_b`` does -- both routes
    resolve the shared scale through the same ``+`` reduction, so their
    scenes are byte-identical.
    """

    def _temporal_charts(self):
        df1 = pl.DataFrame({"t": ["2020-01-01", "2020-06-01"], "v": [1.0, 2.0]}).with_columns(
            pl.col("t").str.to_datetime()
        )
        df2 = pl.DataFrame({"t": ["2019-01-01", "2021-01-01"], "v": [3.0, 4.0]}).with_columns(
            pl.col("t").str.to_datetime()
        )
        c1 = ferrum.Chart(df1).mark_line().encode(x="t", y="v")
        c2 = ferrum.Chart(df2).mark_point().encode(x="t", y="v")
        return c1, c2

    def test_render_interactive_matches_the_plus_chart_on_shared_temporal_x(self):
        """A temporal shared x on the merged path matches the plain ``+``
        chart's own scene exactly, byte for byte."""
        from ferrum._scene import _render_scene

        c1, c2 = self._temporal_charts()
        layer = ferrum.LayerChart(c1, c2, resolve={"x": "shared"})
        scene_json, packed = layer._render_interactive()
        plus_scene_json, plus_packed = _render_scene(c1 + c2)
        assert scene_json == plus_scene_json
        assert bytes(packed) == bytes(plus_packed)


def test_introspection_helper_tolerates_missing_columns():
    """A genuinely missing column returns ``None`` rather than raising --
    polars raises ``ColumnNotFoundError`` (not a ``KeyError`` subclass) on
    ``df["missing"]``, so the lookup must catch that type by name."""
    df = pl.DataFrame({"present": [1.0, 2.0]})
    assert _classify_field(df, "absent") is None
