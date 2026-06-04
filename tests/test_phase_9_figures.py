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


def test_plots_submodule_accessible():
    assert hasattr(fe, "plots")
    assert callable(fe.plots.displot)


@pytest.fixture
def iris_like():
    np.random.seed(0)
    return pl.DataFrame(
        {
            "sepal_length": np.random.normal(5.0, 0.5, 60).tolist(),
            "sepal_width": np.random.normal(3.0, 0.3, 60).tolist(),
            "species": ["a"] * 30 + ["b"] * 30,
        }
    )


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

    @pytest.mark.parametrize(
        "multiple,expected_position_type",
        [
            ("layer", "identity"),
            ("dodge", "dodge"),
            ("stack", "stack"),
            ("fill", "stack"),  # normalize stack
        ],
    )
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
        svg = chart.to_svg()
        assert "<svg" in svg

    def test_invalid_kind_errors(self, iris_like):
        with pytest.raises(ValueError, match="kind"):
            fe.displot(iris_like, x="sepal_length", kind="bogus")

    def test_facet_col_row(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", col="species")
        d = json.loads(chart.to_spec().to_json())
        assert d.get("facet") is not None

    def test_kde_with_hue_threads_groupby(self, iris_like):
        # Regression: displot(kind="kde", hue=...) must forward the hue
        # column as Kde.groupby so the output retains it for color encoding.
        chart = fe.displot(iris_like, x="sepal_length", hue="species", kind="kde")
        d = json.loads(chart.to_spec().to_json())
        kde_t = next(t for t in d.get("transforms", []) if t.get("type") == "kde")
        assert kde_t.get("groupby") == "species"
        # End-to-end render must not raise and must produce SVG.
        assert "<svg" in chart.to_svg()

    def test_mark_density_groupby_passthrough(self, iris_like):
        # Direct API path mirrors the displot wiring.
        chart = (
            fe.Chart(iris_like)
            .mark_density(groupby="species")
            .encode(
                x="sepal_length",
                color="species",
            )
        )
        d = json.loads(chart.to_spec().to_json())
        kde_t = next(t for t in d.get("transforms", []) if t.get("type") == "kde")
        assert kde_t.get("groupby") == "species"

    def test_mark_histogram_horizontal_orientation(self, iris_like):
        # JointChart's right-marginal pattern: bind the data column to y and
        # request horizontal orientation; encoding remap flips x ↔ y.
        chart = (
            fe.Chart(iris_like).mark_histogram(orientation="horizontal").encode(y="sepal_length")
        )
        resolved = chart._resolve_pending()
        # Bin is on the y field (the data dimension), output goes on y/y2/x.
        assert resolved._encoding["y"].field == "bin_start"
        assert resolved._encoding["y2"].field == "bin_end"
        assert resolved._encoding["x"].field == "count"

    def test_mark_density_horizontal_orientation(self, iris_like):
        chart = fe.Chart(iris_like).mark_density(orientation="horizontal").encode(y="sepal_length")
        resolved = chart._resolve_pending()
        assert resolved._encoding["y"].field == "value"
        assert resolved._encoding["x"].field == "density"


# ---------------------------------------------------------------------------
# Task 29 — catplot
# ---------------------------------------------------------------------------


@pytest.fixture
def cat_data():
    np.random.seed(1)
    return pl.DataFrame(
        {
            "group": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
            "subgroup": (["x", "y"] * 30),
            "value": np.random.normal(0, 1, 60).tolist(),
        }
    )


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
            cat_data,
            x="group",
            y="value",
            hue="subgroup",
            kind="bar",
            dodge=True,
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
    def test_lm_default_layers(self, reg_data):
        # Schwabish SB-followup (2026-05-12): lmplot defaults to
        # show_metrics=True which adds a metrics-corner mark_text layer.
        # 3 (scatter + ribbon + line) → 4 with the corner overlay.
        chart = fe.lmplot(reg_data, x="x", y="y")
        d = json.loads(chart.to_spec().to_json())
        assert d.get("layers") is not None
        assert len(d["layers"]) == 4

    def test_lm_no_ci_layers(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", ci=None)
        d = json.loads(chart.to_spec().to_json())
        # scatter + line + metrics-corner text → 3 layers.
        assert d.get("layers") is not None
        assert len(d["layers"]) == 3

    def test_lm_no_metrics_layers(self, reg_data):
        # Opting out of the corner restores the pre-Schwabish layer count.
        chart = fe.lmplot(reg_data, x="x", y="y", show_metrics=False)
        d = json.loads(chart.to_spec().to_json())
        assert d.get("layers") is not None
        assert len(d["layers"]) == 3  # scatter + ribbon + line

    def test_lm_show_metrics_emits_text(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y")
        svg = chart.to_svg()
        assert "R²" in svg
        assert "RMSE" in svg

    def test_lm_no_scatter(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", scatter=False)
        d = json.loads(chart.to_spec().to_json())
        # Schwabish SB-followup (3a rework, 2026-05-12): scatter=False
        # still emits the metrics-corner text overlay because the
        # corner rides on the Smooth-transform fit-grid output, not
        # the scatter data. 3 layers = ribbon + line + metrics-text.
        assert d.get("layers") is not None
        assert len(d["layers"]) == 3

    def test_lm_no_scatter_no_metrics(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", scatter=False, show_metrics=False)
        d = json.loads(chart.to_spec().to_json())
        # Opting out of metrics restores the pre-Schwabish 2-layer shape.
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

    def test_scatter_kws_forwarded(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", scatter_kws={"opacity": 0.3})
        d = json.loads(chart.to_spec().to_json())
        point_layer = next(
            (l for l in d["layers"] if l.get("mark") == "point"),
            None,
        )
        assert point_layer is not None
        assert point_layer.get("mark_style", {}).get("opacity") == 0.3

    def test_line_kws_forwarded(self, reg_data):
        chart = fe.lmplot(
            reg_data,
            x="x",
            y="y",
            line_kws={"stroke_width": 4},
            show_metrics=False,
        )
        d = json.loads(chart.to_spec().to_json())
        layers = d.get("layers", [])
        line_layer = next(
            (l for l in layers if l.get("mark") == "line"),
            None,
        )
        assert line_layer is not None
        assert line_layer.get("mark_style", {}).get("stroke_width") == 4

    def test_renders_e2e(self, reg_data):
        chart = fe.lmplot(reg_data, x="x", y="y", method="lm", ci=None)
        svg = chart.to_svg()
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# Task 31 — residplot
# ---------------------------------------------------------------------------


class TestResidplot:
    def test_default_legacy_transform_path(self, reg_data):
        # Schwabish SB-followup (2026-05-12): the default annotated path
        # computes residuals Python-side via np.polyfit and emits no
        # smooth transform. Opting out of both annotations restores the
        # legacy Smooth(output="residuals") transform path.
        chart = fe.residplot(
            reg_data,
            x="x",
            y="y",
            show_metrics=False,
            zero_line=False,
        )
        d = json.loads(chart.to_spec().to_json())
        smooth_t = next(
            (t for t in d.get("transforms", []) if t.get("type") == "smooth"),
            None,
        )
        assert smooth_t is not None
        assert smooth_t.get("output") == "residuals"

    def test_robust_legacy_path(self, reg_data):
        # Robust path only available without annotations (raises
        # NotImplementedError when paired with show_metrics/zero_line).
        chart = fe.residplot(
            reg_data,
            x="x",
            y="y",
            robust=True,
            show_metrics=False,
            zero_line=False,
        )
        d = json.loads(chart.to_spec().to_json())
        robust_t = next(
            (t for t in d.get("transforms", []) if t.get("type") == "robust"),
            None,
        )
        assert robust_t is not None
        assert robust_t.get("output") == "residuals"

    def test_robust_with_annotations_works(self, reg_data):
        # Schwabish SB-followup (2026-05-12, Option B): annotations are
        # injected by the Rust Smooth/Robust transforms, so the
        # robust+annotations path now ships R²/RMSE/MAE alongside Huber
        # residuals.
        chart = fe.residplot(reg_data, x="x", y="y", robust=True)
        svg = chart.to_svg()
        assert "R²" in svg
        assert "#8a8a8a" in svg

    def test_lowess_adds_layer(self, reg_data):
        chart = fe.residplot(reg_data, x="x", y="y", lowess=True)
        d = json.loads(chart.to_spec().to_json())
        # base (point) + zero_line + metrics + lowess → 4 layers
        assert d.get("layers") is not None
        assert len(d["layers"]) >= 2

    def test_dropna_removes_null_rows(self, reg_data):
        data_with_nulls = reg_data.with_columns(
            pl.when(pl.col("x") < 2).then(None).otherwise(pl.col("y")).alias("y")
        )
        n_nulls = data_with_nulls.filter(pl.col("y").is_null()).height
        assert n_nulls > 0
        chart = fe.residplot(
            data_with_nulls, x="x", y="y", dropna=True, show_metrics=False, zero_line=False
        )
        # dropna=True should have removed the null rows before the chart
        # received them. The chart's internal data should have fewer rows.
        assert chart._data.height == reg_data.height - n_nulls

    def test_label_adds_color_encoding(self, reg_data):
        chart = fe.residplot(
            reg_data, x="x", y="y", label="Residuals", show_metrics=False, zero_line=False
        )
        d = json.loads(chart.to_spec().to_json())
        assert d.get("encoding", {}).get("color", {}).get("field") == "_label"

    def test_no_lowess_legacy_single_layer(self, reg_data):
        chart = fe.residplot(
            reg_data,
            x="x",
            y="y",
            lowess=False,
            show_metrics=False,
            zero_line=False,
        )
        d = json.loads(chart.to_spec().to_json())
        assert d.get("layers") is None or len(d.get("layers", [])) <= 1

    def test_renders_e2e(self, reg_data):
        chart = fe.residplot(reg_data, x="x", y="y")
        svg = chart.to_svg()
        assert "<svg" in svg

    def test_show_metrics_emits_corner_annotation(self, reg_data):
        chart = fe.residplot(reg_data, x="x", y="y")
        svg = chart.to_svg()
        assert "R²" in svg

    def test_zero_line_renders_dashed_rule(self, reg_data):
        chart = fe.residplot(reg_data, x="x", y="y")
        svg = chart.to_svg()
        # Dashed rule at y=0 ships in stroke #8a8a8a per the
        # _inject_metrics_corner / _inject_zero_reference idiom.
        assert "#8a8a8a" in svg


# ---------------------------------------------------------------------------
# Task 32 — pairplot
# ---------------------------------------------------------------------------


@pytest.fixture
def iris_pair():
    rng = np.random.default_rng(3)
    n = 20
    return pl.DataFrame(
        {
            "a": rng.normal(0, 1, n).tolist(),
            "b": rng.normal(0, 1, n).tolist(),
            "c": rng.normal(0, 1, n).tolist(),
            "species": (["x"] * 10 + ["y"] * 10),
        }
    )


class TestPairplot:
    def test_default_returns_repeatchart(self, iris_pair):
        rc = fe.pairplot(iris_pair, vars=["a", "b"])
        assert isinstance(rc, fe.RepeatChart)
        assert rc.row == ["a", "b"]
        assert rc.column == ["a", "b"]

    def test_vars_explicit(self, iris_pair):
        rc = fe.pairplot(iris_pair, vars=["a", "b", "c"])
        assert rc.row == ["a", "b", "c"]
        assert rc.column == ["a", "b", "c"]

    def test_x_vars_y_vars(self, iris_pair):
        rc = fe.pairplot(iris_pair, x_vars=["a"], y_vars=["b", "c"])
        assert rc.column == ["a"]
        assert rc.row == ["b", "c"]

    def test_vars_and_x_vars_errors(self, iris_pair):
        with pytest.raises(ValueError, match="vars"):
            fe.pairplot(iris_pair, vars=["a"], x_vars=["b"], y_vars=["c"])

    def test_corner_true(self, iris_pair):
        rc = fe.pairplot(iris_pair, vars=["a", "b"], corner=True)
        assert rc.corner is True

    def test_diagonal_hist(self, iris_pair):
        rc = fe.pairplot(iris_pair, vars=["a", "b"], diag_kind="hist")
        assert rc.diagonal is not None

    def test_diagonal_kde(self, iris_pair):
        rc = fe.pairplot(iris_pair, vars=["a", "b"], diag_kind="kde")
        assert rc.diagonal is not None

    def test_diagonal_none(self, iris_pair):
        rc = fe.pairplot(iris_pair, vars=["a", "b"], diag_kind=None)
        assert rc.diagonal is None

    def test_hue_propagates_to_template_encoding(self, iris_pair):
        rc = fe.pairplot(iris_pair, vars=["a", "b"], hue="species")
        # The template chart's encoding includes color=species.
        assert "color" in rc.template._encoding

    def test_invalid_kind_errors(self, iris_pair):
        with pytest.raises(ValueError, match="kind"):
            fe.pairplot(iris_pair, vars=["a", "b"], kind="bogus")

    def test_auto_detects_numeric_columns(self, iris_pair):
        rc = fe.pairplot(iris_pair)
        # species is non-numeric; should be excluded.
        assert "species" not in rc.row
        assert set(rc.row) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Task 33 — heatmap
# ---------------------------------------------------------------------------


@pytest.fixture
def heat_data():
    return pl.DataFrame(
        {
            "row": ["r1", "r2", "r3"],
            "A": [1.0, 2.0, 3.0],
            "B": [4.0, 5.0, 6.0],
            "C": [7.0, 8.0, 9.0],
        }
    )


class TestHeatmap:
    def test_default(self, heat_data):
        chart = fe.heatmap(heat_data)
        d = json.loads(chart.to_spec().to_json())
        assert any(t.get("type") == "unpivot" for t in d.get("transforms", []))

    def test_annot_true_layers(self, heat_data):
        chart = fe.heatmap(heat_data, annot=True)
        d = json.loads(chart.to_spec().to_json())
        # rect + text layers
        assert d.get("layers") is not None
        assert len(d["layers"]) == 2

    def test_annot_false_single_layer(self, heat_data):
        chart = fe.heatmap(heat_data, annot=False)
        d = json.loads(chart.to_spec().to_json())
        assert d.get("layers") is None
        assert d["mark"] == "rect"

    def test_robust_sets_vmin_vmax_from_percentiles(self, heat_data):
        chart = fe.heatmap(heat_data, annot=False, robust=True)
        d = json.loads(chart.to_spec().to_json())
        # robust path should populate domain bounds.
        scale = d["encoding"]["color"].get("scale", {})
        assert "domain" in scale

    def test_square(self, heat_data):
        chart = fe.heatmap(heat_data, annot=False, square=True)
        d = json.loads(chart.to_spec().to_json())
        # Square heatmap → width == height.
        # Spec width/height land in the top-level "width" / "height" keys.
        assert chart._width == chart._height

    def test_vmin_vmax_explicit(self, heat_data):
        chart = fe.heatmap(heat_data, annot=False, vmin=0.0, vmax=10.0)
        d = json.loads(chart.to_spec().to_json())
        scale = d["encoding"]["color"].get("scale", {})
        assert scale.get("domain") == [0.0, 10.0]

    def test_errors_on_no_numeric(self):
        df = pl.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="numeric"):
            fe.heatmap(df)


# ---------------------------------------------------------------------------
# Task 34 — clustermap
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_data():
    rng = np.random.default_rng(4)
    return pl.DataFrame(
        {
            "row": [f"r{i}" for i in range(5)],
            "A": rng.normal(0, 1, 5).tolist(),
            "B": rng.normal(0, 1, 5).tolist(),
            "C": rng.normal(0, 1, 5).tolist(),
            "D": rng.normal(0, 1, 5).tolist(),
        }
    )


class TestClustermap:
    def test_returns_clustermap_chart(self, cluster_data):
        cm = fe.clustermap(cluster_data)
        assert isinstance(cm, fe.ClusterMapChart)

    def test_heatmap_center_has_linkage_transforms(self, cluster_data):
        cm = fe.clustermap(cluster_data)
        d = json.loads(cm.heatmap.to_spec().to_json())
        # row_link + col_link + 1 reorder (rows, consuming row_link_order via
        # the Phase-9-finalize Reorder.from='<named_output>' mechanism) + unpivot.
        # Column ordering is handled by the column-dendrogram layer separately;
        # the heatmap's visible row ordering matches the row-linkage cluster order.
        types = [t.get("type") for t in d.get("transforms", [])]
        assert types.count("linkage") == 2
        assert types.count("reorder") == 1
        assert "unpivot" in types

    def test_dendrograms_present(self, cluster_data):
        cm = fe.clustermap(cluster_data)
        assert cm.row_dendrogram is not None
        assert cm.col_dendrogram is not None

    def test_col_dendrogram_reads_segments(self, cluster_data):
        cm = fe.clustermap(cluster_data)
        d = json.loads(cm.col_dendrogram.to_spec().to_json())
        layers = d.get("layers") or []
        assert any(L.get("data_source") == "col_link_segments" for L in layers)

    def test_row_dendrogram_reads_segments(self, cluster_data):
        cm = fe.clustermap(cluster_data)
        d = json.loads(cm.row_dendrogram.to_spec().to_json())
        layers = d.get("layers") or []
        assert any(L.get("data_source") == "row_link_segments" for L in layers)

    def test_dendrogram_ratio_passes_through(self, cluster_data):
        cm = fe.clustermap(cluster_data, dendrogram_ratio=0.3)
        assert cm.dendrogram_ratio == 0.3

    def test_z_score_threads_to_linkage(self, cluster_data):
        cm = fe.clustermap(cluster_data, z_score="rows")
        d = json.loads(cm.heatmap.to_spec().to_json())
        link_t = next(t for t in d["transforms"] if t.get("type") == "linkage")
        assert link_t.get("z_score") == "rows"

    def test_method_metric_threads_to_linkage(self, cluster_data):
        cm = fe.clustermap(cluster_data, method="average", metric="cosine")
        d = json.loads(cm.heatmap.to_spec().to_json())
        link_t = next(t for t in d["transforms"] if t.get("type") == "linkage")
        assert link_t.get("method") == "average"
        assert link_t.get("metric") == "cosine"

    def test_cmap_forwarded_to_color_encoding(self, cluster_data):
        """BUG-4 regression: cmap must be forwarded to the center heatmap color scheme."""
        cm = fe.clustermap(cluster_data, cmap="rdbu")
        d = json.loads(cm.heatmap.to_spec().to_json())
        color_enc = d.get("encoding", {}).get("color", {})
        assert color_enc.get("scheme") == "rdbu", (
            "cmap='rdbu' must be forwarded to the color encoding scheme"
        )

    def test_cmap_theme_default(self, cluster_data):
        """clustermap default cmap defers to theme (no explicit scheme in spec)."""
        cm = fe.clustermap(cluster_data)
        d = json.loads(cm.heatmap.to_spec().to_json())
        color_enc = d.get("encoding", {}).get("color", {})
        assert "scheme" not in color_enc

    def test_dendrogram_panels_have_grid_disabled(self, cluster_data):
        """BUG-5 regression: dendrogram panels must have grid=False to suppress gridlines."""
        cm = fe.clustermap(cluster_data)
        assert cm.col_dendrogram._theme is not None, "col_dendrogram must have a theme"
        assert cm.row_dendrogram._theme is not None, "row_dendrogram must have a theme"
        assert cm.col_dendrogram._theme._props.get("grid") is False
        assert cm.row_dendrogram._theme._props.get("grid") is False

    def test_dendrogram_grid_false_preserved_with_user_theme(self, cluster_data):
        """User-supplied theme is merged with grid=False for dendrogram panels."""
        from ferrum.themes import Theme

        user_theme = Theme(mark_color="#ff0000")
        cm = fe.clustermap(cluster_data, theme=user_theme)
        assert cm.col_dendrogram._theme._props.get("grid") is False
        assert cm.col_dendrogram._theme._props.get("mark_color") == "#ff0000"

    def test_renders_when_data_has_no_id_column(self):
        """All-numeric input must render without referencing a non-existent column.

        Regression for: ``ValueError: unknown column '_row_id'``. The original
        ``clustermap`` referenced ``_row_id`` in the y encoding when no
        non-numeric id column was present, but never synthesized the column —
        unlike ``heatmap``, which materialises ``_row_id`` via
        ``with_row_index``. Mirrors a real iris-style call:
        ``clustermap(iris.drop("target"))``.
        """
        rng = np.random.default_rng(7)
        all_numeric = pl.DataFrame(
            {
                "A": rng.normal(0, 1, 6).tolist(),
                "B": rng.normal(0, 1, 6).tolist(),
                "C": rng.normal(0, 1, 6).tolist(),
                "D": rng.normal(0, 1, 6).tolist(),
            }
        )
        cm = fe.clustermap(all_numeric)
        svg = cm.to_svg()
        assert svg.startswith("<svg") or "<svg" in svg[:200]


# ---------------------------------------------------------------------------
# Task 35 — jointplot
# ---------------------------------------------------------------------------


@pytest.fixture
def joint_data():
    rng = np.random.default_rng(5)
    return pl.DataFrame(
        {
            "x": rng.normal(0, 1, 50).tolist(),
            "y": rng.normal(0, 1, 50).tolist(),
            "g": (["a"] * 25 + ["b"] * 25),
        }
    )


class TestJointplot:
    def test_returns_jointchart(self, joint_data):
        jc = fe.jointplot(joint_data, x="x", y="y")
        assert isinstance(jc, fe.JointChart)

    def test_scatter_center(self, joint_data):
        jc = fe.jointplot(joint_data, x="x", y="y", kind="scatter")
        d = json.loads(jc.center.to_spec().to_json())
        assert d["mark"] == "point"

    def test_kde_center(self, joint_data):
        jc = fe.jointplot(joint_data, x="x", y="y", kind="kde")
        d = json.loads(jc.center.to_spec().to_json())
        # 2D kde routes through desugar_contour → layered output.
        assert d.get("layers") is not None or any(
            t.get("type") in ("kde2d", "contour") for t in d.get("transforms", [])
        )

    def test_hist_center_uses_bin2d(self, joint_data):
        jc = fe.jointplot(joint_data, x="x", y="y", kind="hist")
        d = json.loads(jc.center.to_spec().to_json())
        assert any(t.get("type") == "bin_2d" for t in d.get("transforms", []))

    def test_hex_center(self, joint_data):
        jc = fe.jointplot(joint_data, x="x", y="y", kind="hex")
        d = json.loads(jc.center.to_spec().to_json())
        assert any(t.get("type") == "hex" for t in d.get("transforms", []))

    def test_reg_center(self, joint_data):
        jc = fe.jointplot(joint_data, x="x", y="y", kind="reg")
        d = json.loads(jc.center.to_spec().to_json())
        # scatter + smoothing line → 2 layers.
        assert d.get("layers") is not None
        assert len(d["layers"]) >= 2

    def test_marginal_hist(self, joint_data):
        jc = fe.jointplot(joint_data, x="x", y="y", marginal_kind="hist")
        d_top = json.loads(jc.top.to_spec().to_json())
        assert d_top["mark"] == "bar"

    def test_marginal_kde(self, joint_data):
        jc = fe.jointplot(joint_data, x="x", y="y", marginal_kind="kde")
        d_top = json.loads(jc.top.to_spec().to_json())
        assert any(t.get("type") == "kde" for t in d_top.get("transforms", []))

    def test_invalid_kind_errors(self, joint_data):
        with pytest.raises(ValueError, match="kind"):
            fe.jointplot(joint_data, x="x", y="y", kind="bogus")

    def test_invalid_marginal_kind_errors(self, joint_data):
        with pytest.raises(ValueError, match="marginal"):
            fe.jointplot(joint_data, x="x", y="y", marginal_kind="bogus")
