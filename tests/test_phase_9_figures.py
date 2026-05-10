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


# ---------------------------------------------------------------------------
# Task 29 — catplot
# ---------------------------------------------------------------------------

@pytest.fixture
def cat_data():
    np.random.seed(1)
    return pl.DataFrame({
        "group":    ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
        "subgroup": (["x", "y"] * 30),
        "value":    np.random.normal(0, 1, 60).tolist(),
    })


class TestCatplot:
    def test_strip_with_jitter(self, cat_data):
        chart = fe.catplot(cat_data, x="group", y="value", kind="strip")
        d = json.loads(chart.to_spec().to_json())
        assert d["mark"] == "point"
        assert d.get("position", {}).get("type") == "jitter"

    def test_strip_without_jitter(self, cat_data):
        chart = fe.catplot(cat_data, x="group", y="value", kind="strip", jitter=False)
        d = json.loads(chart.to_spec().to_json())
        assert d["mark"] == "point"
        assert d.get("position", {}).get("type") == "identity"

    def test_swarm(self, cat_data):
        chart = fe.catplot(cat_data, x="group", y="value", kind="swarm")
        d = json.loads(chart.to_spec().to_json())
        assert any(t.get("type") == "swarm" for t in d.get("transforms", []))

    def test_box(self, cat_data):
        chart = fe.catplot(cat_data, x="group", y="value", kind="box")
        d = json.loads(chart.to_spec().to_json())
        assert any(t.get("type") == "box_stats" for t in d.get("transforms", []))

    def test_violin(self, cat_data):
        chart = fe.catplot(cat_data, x="group", y="value", kind="violin")
        d = json.loads(chart.to_spec().to_json())
        assert any(t.get("type") == "violin" for t in d.get("transforms", []))

    def test_boxen(self, cat_data):
        chart = fe.catplot(cat_data, x="group", y="value", kind="boxen")
        d = json.loads(chart.to_spec().to_json())
        assert any(t.get("type") == "letter_value" for t in d.get("transforms", []))

    def test_point(self, cat_data):
        chart = fe.catplot(cat_data, x="group", y="value", kind="point")
        d = json.loads(chart.to_spec().to_json())
        assert d["mark"] == "point"

    def test_bar(self, cat_data):
        chart = fe.catplot(cat_data, x="group", y="value", kind="bar")
        d = json.loads(chart.to_spec().to_json())
        assert d["mark"] == "bar"

    def test_count(self, cat_data):
        chart = fe.catplot(cat_data, x="group", kind="count")
        d = json.loads(chart.to_spec().to_json())
        assert d["mark"] == "bar"
        assert any(t.get("type") == "aggregate" for t in d.get("transforms", []))

    def test_dodge_with_hue(self, cat_data):
        chart = fe.catplot(
            cat_data, x="group", y="value", hue="subgroup",
            kind="bar", dodge=True,
        )
        d = json.loads(chart.to_spec().to_json())
        assert d.get("position", {}).get("type") == "dodge"
        assert d["position"].get("by") == "subgroup"

    def test_orient_horizontal(self, cat_data):
        chart = fe.catplot(cat_data, y="group", x="value", kind="box", orient="h")
        d = json.loads(chart.to_spec().to_json())
        assert d.get("coord", {}).get("kind") == "flip"

    def test_invalid_kind_errors(self, cat_data):
        with pytest.raises(ValueError, match="kind"):
            fe.catplot(cat_data, x="group", y="value", kind="bogus")


# ---------------------------------------------------------------------------
# Task 30 — lmplot
# ---------------------------------------------------------------------------

@pytest.fixture
def reg_data():
    rng = np.random.default_rng(2)
    n = 50
    x = np.linspace(0, 10, n)
    y = 2.0 + 0.5 * x + rng.normal(0, 1, n)
    return pl.DataFrame({"x": x.tolist(), "y": y.tolist()})


class TestLmplot:
    def test_lm_default(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y")
        d = json.loads(chart.to_spec().to_json())
        # scatter + ribbon + line (3 layers when ci is set).
        assert d.get("layers") is not None
        assert len(d["layers"]) == 3

    def test_lm_no_ci(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", ci=None)
        d = json.loads(chart.to_spec().to_json())
        # scatter + line (no ribbon) → 2 layers.
        assert d.get("layers") is not None
        assert len(d["layers"]) == 2

    def test_lm_no_scatter(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", scatter=False)
        d = json.loads(chart.to_spec().to_json())
        # ribbon + line only.
        assert d.get("layers") is not None
        assert len(d["layers"]) == 2

    def test_loess_method(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", method="loess")
        d = json.loads(chart.to_spec().to_json())
        smooth_t = next(
            (t for t in d.get("transforms", []) if t.get("type") == "smooth"),
            None,
        )
        assert smooth_t is not None
        assert smooth_t["method"] == "loess"

    def test_logistic_method(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", method="logistic")
        d = json.loads(chart.to_spec().to_json())
        assert any(t.get("type") == "logistic" for t in d.get("transforms", []))

    def test_glm_method(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", method="glm")
        d = json.loads(chart.to_spec().to_json())
        assert any(t.get("type") == "glm" for t in d.get("transforms", []))

    def test_robust_method(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", method="robust")
        d = json.loads(chart.to_spec().to_json())
        assert any(t.get("type") == "robust" for t in d.get("transforms", []))

    def test_x_bins_estimator(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", x_bins=5, x_estimator="mean")
        d = json.loads(chart.to_spec().to_json())
        smooth_t = next(
            (t for t in d.get("transforms", []) if t.get("type") == "smooth"),
            None,
        )
        assert smooth_t is not None
        assert smooth_t.get("x_bins") == 5
        assert smooth_t.get("x_estimator") == "mean"

    def test_invalid_method_errors(self, reg_data):
        with pytest.raises(ValueError, match="method"):
            fe.lmplot(reg_data, x="x", y="y", method="bogus")

    def test_renders_e2e(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", method="lm", ci=None)
        svg = chart.show_svg()
        assert "<svg" in svg


