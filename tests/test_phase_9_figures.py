"""Phase 9e figure-level function tests."""
import json
import numpy as np
import polars as pl
import pytest

import ferrum as fe


def test_all_8_functions_importable():
    assert callable(fe.displot)
    assert callable(fe.catplot)
    assert callable(fe.lmplot)
    assert callable(fe.residplot)
    assert callable(fe.pairplot)
    assert callable(fe.heatmap)
    assert callable(fe.clustermap)
    assert callable(fe.jointplot)


def test_figure_submodule_accessible():
    assert hasattr(fe, "figure")
    assert callable(fe.figure.displot)


@pytest.fixture
def iris_like():
    np.random.seed(0)
    return pl.DataFrame({
        "sepal_length": np.random.normal(5.0, 0.5, 60).tolist(),
        "sepal_width":  np.random.normal(3.0, 0.3, 60).tolist(),
        "species":      ["a"] * 30 + ["b"] * 30,
    })


# ---------------------------------------------------------------------------
# Task 28 — displot
# ---------------------------------------------------------------------------

class TestDisplot:
    def test_hist_default(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length")
        assert isinstance(chart, fe.Chart)
        d = json.loads(chart.to_spec().to_json())
        # Histogram desugars to mark_bar with a Bin transform.
        assert d["mark"] == "bar"
        assert any(t.get("type") == "bin" for t in d.get("transforms", []))

    def test_kde(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", kind="kde")
        d = json.loads(chart.to_spec().to_json())
        assert any(t.get("type") == "kde" for t in d.get("transforms", []))

    def test_ecdf_uses_cumulative_bin(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", kind="ecdf")
        d = json.loads(chart.to_spec().to_json())
        bin_t = next((t for t in d.get("transforms", []) if t.get("type") == "bin"), None)
        assert bin_t is not None
        assert bin_t.get("cumulative") is True

    def test_rug_kind(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", kind="rug")
        d = json.loads(chart.to_spec().to_json())
        assert d["mark"] == "tick"

    @pytest.mark.parametrize("multiple,expected_position_type", [
        ("layer", "identity"),
        ("dodge", "dodge"),
        ("stack", "stack"),
        ("fill", "stack"),    # normalize stack
    ])
    def test_multiple_param_sets_position(self, iris_like, multiple, expected_position_type):
        chart = fe.displot(iris_like, x="sepal_length", hue="species", multiple=multiple)
        d = json.loads(chart.to_spec().to_json())
        assert d.get("position", {}).get("type") == expected_position_type
        if multiple == "fill":
            assert d["position"].get("offset") == "normalize"

    def test_cumulative_param_threads_to_bin(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", cumulative=True)
        d = json.loads(chart.to_spec().to_json())
        bin_t = next(t for t in d.get("transforms", []) if t.get("type") == "bin")
        assert bin_t["cumulative"] is True

    def test_renders_e2e(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", kind="hist")
        svg = chart.show_svg()
        assert "<svg" in svg

    def test_invalid_kind_errors(self, iris_like):
        with pytest.raises(ValueError, match="kind"):
            fe.displot(iris_like, x="sepal_length", kind="bogus")

    def test_facet_col_row(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", col="species")
        d = json.loads(chart.to_spec().to_json())
        assert d.get("facet") is not None
