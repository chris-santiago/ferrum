"""Phase 9d new-mark tests."""
import pytest
import polars as pl
import ferrum as fe


class TestMarkSegment:
    def test_segment_no_longer_in_deferred(self):
        from ferrum.marks import PHASE_9_PLUS_MARKS
        assert "segment" not in PHASE_9_PLUS_MARKS

    def test_mark_segment_accepts_x2_y2(self):
        df = pl.DataFrame({
            "x": [0.0, 1.0], "y": [0.0, 1.0],
            "x2": [1.0, 2.0], "y2": [1.0, 2.0],
        })
        chart = fe.Chart(df).mark_segment().encode(x="x", y="y", x2="x2", y2="y2")
        spec = chart.to_spec()
        # Mark name in JSON.
        import json
        d = json.loads(spec.to_json())
        assert d["mark"] == "segment"

    def test_mark_segment_renders_diagonal_line(self):
        df = pl.DataFrame({
            "x": [0.0, 1.0], "y": [0.0, 1.0],
            "x2": [1.0, 2.0], "y2": [2.0, 0.0],
        })
        chart = fe.Chart(df).mark_segment().encode(x="x", y="y", x2="x2", y2="y2")
        svg = chart.show_svg()
        assert "<svg" in svg
        # Two segments → at least 2 <line> elements (or path equivalents).
        assert svg.count("<line") + svg.count("<path") >= 2

    def test_mark_segment_position_only_identity(self):
        from ferrum import Identity, Dodge
        df = pl.DataFrame({"x": [0.0], "y": [0.0], "x2": [1.0], "y2": [1.0]})
        # Identity is fine.
        fe.Chart(df).mark_segment(position=Identity()).encode(x="x", y="y", x2="x2", y2="y2")
        # Dodge is not.
        with pytest.raises(TypeError, match="Dodge"):
            fe.Chart(df).mark_segment(position=Dodge())
