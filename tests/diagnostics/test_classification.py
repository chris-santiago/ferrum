"""Phase 10b tests: classification curve marks + figure functions + visualizers."""
from __future__ import annotations

import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture


@pytest.fixture(scope="module")
def binary_source():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    return ferrum.ModelSource(model, X, df["y"])


@pytest.fixture(scope="module")
def multi_source():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    return ferrum.ModelSource(model, X, df["y"])


# --- Mark-layer tests (Task 15) -------------------------------------


def test_mark_roc_renders_binary(binary_source):
    roc = binary_source.roc_curve()
    svg = ferrum.Chart(roc).mark_roc().show_svg()
    assert "<svg" in svg


def test_mark_roc_renders_multiclass(multi_source):
    roc = multi_source.roc_curve()
    svg = ferrum.Chart(roc).mark_roc().show_svg()
    assert "<svg" in svg


def test_mark_roc_annotate_auc_renders_text_label(binary_source):
    """annotate_auc=True emits a per-class text layer at fixed (x, y)
    coordinates with the formatted AUC value. The chart builder injects
    `_auc_label_x/_y/text` columns one-non-null-per-class, and Rust's
    mark_text skips null rows so exactly N labels render for N classes.
    """
    # Direct mark_roc(...) on a raw DataFrame doesn't go through the
    # chart builder, so the annotation columns won't be present and the
    # text layer will render zero labels. The figure-function path is
    # the canonical entry: build via ferrum.roc_chart(annotate_auc=True).
    svg = ferrum.roc_chart(
        binary_source, annotate_auc=True,
    ).show_svg()
    assert "<svg" in svg
    assert "<text" in svg
    # The formatted "AUC = 0." literal should appear in the SVG text.
    assert "AUC = 0." in svg


def test_mark_pr_renders(binary_source):
    pr = binary_source.pr_curve()
    svg = ferrum.Chart(pr).mark_pr().show_svg()
    assert "<svg" in svg


def test_mark_pr_iso_lines_raises(binary_source):
    pr = binary_source.pr_curve()
    with pytest.raises(NotImplementedError, match="iso_lines"):
        ferrum.Chart(pr).mark_pr(iso_lines=True).show_svg()


def test_mark_pr_annotate_ap_raises(binary_source):
    pr = binary_source.pr_curve()
    with pytest.raises(NotImplementedError, match="annotate_ap"):
        ferrum.Chart(pr).mark_pr(annotate_ap=True).show_svg()


def test_mark_calibration_renders(binary_source):
    cal = binary_source.calibration_curve(n_bins=10)
    svg = ferrum.Chart(cal).mark_calibration().show_svg()
    assert "<svg" in svg


def test_mark_gain_renders(binary_source):
    gain = binary_source.cumulative_gain()
    svg = ferrum.Chart(gain).mark_gain().show_svg()
    assert "<svg" in svg


def test_mark_lift_renders(binary_source):
    lift = binary_source.lift_curve()
    svg = ferrum.Chart(lift).mark_lift().show_svg()
    assert "<svg" in svg


def test_mark_discrimination_threshold_renders(binary_source):
    dt = binary_source.discrimination_threshold(n_thresholds=20)
    long = dt.unpivot(
        index="threshold",
        on=["precision", "recall", "f1", "queue_rate"],
        variable_name="metric",
        value_name="value",
    )
    svg = ferrum.Chart(long).mark_discrimination_threshold().show_svg()
    assert "<svg" in svg


def test_mark_discrimination_threshold_threshold_line_raises(binary_source):
    dt = binary_source.discrimination_threshold(n_thresholds=10)
    long = dt.unpivot(
        index="threshold",
        on=["precision", "recall", "f1", "queue_rate"],
        variable_name="metric",
        value_name="value",
    )
    with pytest.raises(NotImplementedError, match="threshold_line"):
        ferrum.Chart(long).mark_discrimination_threshold(
            threshold_line=True,
        ).show_svg()


# --- Figure-function tests (Task 16) --------------------------------


def test_roc_chart_figure_function(binary_source):
    svg = ferrum.roc_chart(binary_source).show_svg()
    assert "<svg" in svg


def test_roc_chart_from_model():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    svg = ferrum.roc_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
    ).show_svg()
    assert "<svg" in svg


def test_pr_chart_figure_function(binary_source):
    svg = ferrum.pr_chart(binary_source).show_svg()
    assert "<svg" in svg


def test_calibration_chart_figure_function(binary_source):
    svg = ferrum.calibration_chart(binary_source, n_bins=5).show_svg()
    assert "<svg" in svg


def test_calibration_chart_multi_model_variadic_overlay():
    """Multi-model calibration_chart (Phase 10h) routes through
    ComparedModelSource, auto-naming each positional model 'model_0',
    'model_1', etc., and overlays the resulting reliability curves.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    src = ferrum.ModelSource(model, X, df["y"])
    chart = ferrum.calibration_chart(src, src)
    assert "<svg" in chart.show_svg()


def test_gain_chart_figure_function(binary_source):
    svg = ferrum.gain_chart(binary_source).show_svg()
    assert "<svg" in svg


def test_lift_chart_figure_function(binary_source):
    svg = ferrum.lift_chart(binary_source).show_svg()
    assert "<svg" in svg


def test_discrimination_threshold_chart_figure_function(binary_source):
    svg = ferrum.discrimination_threshold_chart(
        binary_source, n_thresholds=20,
    ).show_svg()
    assert "<svg" in svg


# --- Visualizer tests (Task 17) --------------------------------------


def test_roc_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.ROCVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "auc_mean=" in repr(viz)
    assert 0.0 <= viz._metrics["auc_mean"] <= 1.0
    assert "<svg" in viz.show().show_svg()


def test_roc_visualizer_score():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    viz = ferrum.ROCVisualizer(model).fit(X, df["y"])
    auc = viz.score(X.to_numpy(), df["y"].to_numpy())
    assert 0.0 <= auc <= 1.0


def test_pr_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.PRVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "ap_mean=" in repr(viz)
    assert 0.0 <= viz._metrics["ap_mean"] <= 1.0
    assert "<svg" in viz.show().show_svg()


def test_calibration_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.CalibrationVisualizer(model, n_bins=5).fit(
        df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "calibration_error=" in repr(viz)
    assert viz._metrics["calibration_error"] >= 0.0
    assert "<svg" in viz.show().show_svg()


def test_calibration_visualizer_multi_model_variadic():
    """Multi-model CalibrationVisualizer (Phase 10h) routes through
    ComparedModelSource, auto-naming each positional model 'model_0',
    'model_1', etc. and rendering an overlay reliability diagram.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    viz = ferrum.CalibrationVisualizer(model, model, n_bins=5).fit(X, df["y"])
    assert "calibration_error=" in repr(viz)
    assert "<svg" in viz.show().show_svg()


def test_discrimination_threshold_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.DiscriminationThresholdVisualizer(
        model, n_thresholds=20,
    ).fit(df.select(["f0", "f1", "f2", "f3"]), df["y"])
    assert "best_threshold=" in repr(viz)
    assert 0.0 <= viz._metrics["best_threshold"] <= 1.0
    assert 0.0 <= viz._metrics["best_f1"] <= 1.0
    assert "<svg" in viz.show().show_svg()


def test_visualizer_unfit_repr():
    model = load_fixture("binary_logistic")
    viz = ferrum.ROCVisualizer(model)
    assert repr(viz) == "ROCVisualizer(unfit)"


# --- 10c: confusion matrix (Task 18) ----------------------------------


def test_confusion_matrix_schema(binary_source):
    cm = binary_source.confusion_matrix()
    assert set(cm.columns) == {"actual", "predicted", "value", "value_fmt"}
    assert cm.height == 4  # 2x2 for binary


def test_confusion_matrix_normalized(binary_source):
    import polars as pl
    cm = binary_source.confusion_matrix(normalize="true")
    assert cm.height == 4
    row_sums = (
        cm.group_by("actual")
        .agg(pl.col("value").sum())
        .sort("actual")["value"]
        .to_list()
    )
    for s in row_sums:
        assert abs(s - 1.0) < 1e-9


def test_confusion_matrix_multiclass(multi_source):
    cm = multi_source.confusion_matrix()
    assert cm.height == 9  # 3x3 for 3-class


def test_confusion_matrix_caching(binary_source):
    assert binary_source.confusion_matrix() is binary_source.confusion_matrix()


def test_mark_confusion_renders(binary_source):
    cm = binary_source.confusion_matrix(normalize="true")
    svg = ferrum.Chart(cm).mark_confusion(normalize="true").show_svg()
    assert "<svg" in svg
    # Annotated by default — text labels appear (data labels + axis labels).
    assert svg.count("<text ") >= 4


def test_mark_confusion_annotate_off(binary_source):
    cm = binary_source.confusion_matrix(normalize=None)
    svg = ferrum.Chart(cm).mark_confusion(annotate=False).show_svg()
    assert "<svg" in svg


def test_confusion_matrix_chart_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.confusion_matrix_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "<svg" in chart.show_svg()


def test_confusion_matrix_chart_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.confusion_matrix_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
        normalize="true",
    )
    assert "<svg" in chart.show_svg()


# --- 10c: class prediction error (Task 19) ----------------------------


def test_mark_class_prediction_error_renders(multi_source):
    cm = multi_source.confusion_matrix(normalize=None)
    svg = ferrum.Chart(cm).mark_class_prediction_error().show_svg()
    assert "<svg" in svg


def test_class_prediction_error_chart_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.class_prediction_error_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "<svg" in chart.show_svg()


def test_class_prediction_error_chart_normalized():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.class_prediction_error_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
        normalize=True,
    )
    assert "<svg" in chart.show_svg()


# --- 10c: visualizers (Task 20) ---------------------------------------


def test_confusion_matrix_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ConfusionMatrixVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "accuracy=" in repr(viz)
    assert 0.0 <= viz._metrics["accuracy"] <= 1.0
    assert "<svg" in viz.show().show_svg()


def test_classification_report_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassificationReportVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "f1_macro=" in repr(viz)
    assert 0.0 <= viz._metrics["f1_macro"] <= 1.0
    assert "<svg" in viz.show().show_svg()


def test_class_prediction_error_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassPredictionErrorVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "accuracy=" in repr(viz)
    assert "<svg" in viz.show().show_svg()


def test_class_balance_visualizer_y_only():
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassBalanceVisualizer().fit(df["y"])
    assert "n_classes=" in repr(viz)
    assert "imbalance_ratio=" in repr(viz)
    assert viz._metrics["n_classes"] >= 2.0
    assert viz._metrics["imbalance_ratio"] >= 1.0
    assert "<svg" in viz.show().show_svg()


def test_class_balance_visualizer_xy_signature():
    df = load_dataset("multiclass_classification")
    # sklearn-shape .fit(X, y) should also work; X is ignored.
    X = df.select(["f0", "f1", "f2", "f3"])
    viz = ferrum.ClassBalanceVisualizer().fit(X, df["y"])
    assert "<svg" in viz.show().show_svg()
