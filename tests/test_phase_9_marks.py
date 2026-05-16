"""Phase 9d new-mark tests."""

import pytest
import polars as pl
import ferrum as fe


class TestMarkSegment:
    def test_mark_segment_accepts_x2_y2(self):
        df = pl.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "x2": [1.0, 2.0],
                "y2": [1.0, 2.0],
            }
        )
        chart = fe.Chart(df).mark_segment().encode(x="x", y="y", x2="x2", y2="y2")
        spec = chart.to_spec()
        # Mark name in JSON.
        import json

        d = json.loads(spec.to_json())
        assert d["mark"] == "segment"

    def test_mark_segment_renders_diagonal_line(self):
        df = pl.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "x2": [1.0, 2.0],
                "y2": [2.0, 0.0],
            }
        )
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


class TestMarkBoxen:
    @pytest.fixture
    def df_grouped(self):
        import numpy as np

        np.random.seed(42)
        return pl.DataFrame(
            {
                "g": ["a"] * 100 + ["b"] * 100,
                "v": np.concatenate(
                    [np.random.normal(0, 1, 100), np.random.normal(2, 1, 100)]
                ).tolist(),
            }
        )

    def test_mark_boxen_renders(self, df_grouped):
        chart = fe.Chart(df_grouped).mark_boxen().encode(x="g", y="v")
        svg = chart.show_svg()
        assert "<svg" in svg

    def test_mark_boxen_spec_has_letter_value_transform(self, df_grouped):
        chart = fe.Chart(df_grouped).mark_boxen().encode(x="g", y="v")
        import json

        d = json.loads(chart.to_spec().to_json())
        # Either at top-level or in a layer's transforms.
        all_transforms = list(d.get("transforms", []) or [])
        for layer in d.get("layers", []) or []:
            all_transforms.extend(layer.get("transforms", []) or [])
        assert any(t.get("type") == "letter_value" for t in all_transforms)

    def test_mark_boxen_layered_spec(self, df_grouped):
        chart = fe.Chart(df_grouped).mark_boxen().encode(x="g", y="v")
        spec = chart._build_spec()
        # Multiple layers: rects per depth + median + outliers.
        assert len(spec.layers) >= 3

    def test_mark_boxen_position_dodge_eligible(self, df_grouped):
        from ferrum import Dodge, Jitter

        # Dodge accepted on boxen.
        fe.Chart(df_grouped).mark_boxen(position=Dodge(by="g")).encode(x="g", y="v")
        # Jitter rejected.
        with pytest.raises(TypeError, match="Jitter"):
            fe.Chart(df_grouped).mark_boxen(position=Jitter())

    def test_mark_boxen_k_depth_param_threads_through(self, df_grouped):
        chart = fe.Chart(df_grouped).mark_boxen(k_depth="full").encode(x="g", y="v")
        import json

        d = json.loads(chart.to_spec().to_json())
        all_t = list(d.get("transforms", []) or [])
        for layer in d.get("layers", []) or []:
            all_t.extend(layer.get("transforms", []) or [])
        lv = next(t for t in all_t if t.get("type") == "letter_value")
        # KDepth serializes via serde(tag = "type"): {"type": "full"}.
        assert lv["k_depth"]["type"] == "full"
