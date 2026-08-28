"""Phase 10b tests: classification curve marks + figure functions + visualizers."""

from __future__ import annotations

import re

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
    svg = ferrum.Chart(roc).mark_roc().to_svg()
    assert "<svg" in svg


def test_mark_roc_renders_multiclass(multi_source):
    roc = multi_source.roc_curve()
    svg = ferrum.Chart(roc).mark_roc().to_svg()
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
        binary_source,
        annotate_auc=True,
    ).to_svg()
    assert "<svg" in svg
    assert "<text" in svg
    # The formatted "AUC = 0." literal should appear in the SVG text.
    assert "AUC = 0." in svg


def test_mark_pr_renders(binary_source):
    pr = binary_source.pr_curve()
    svg = ferrum.Chart(pr).mark_pr().to_svg()
    assert "<svg" in svg


def test_mark_pr_iso_lines_renders_4_iso_curves(binary_source):
    """iso_lines=True appends F-score iso curves for F = 0.2, 0.4, 0.6, 0.8
    plus a text-label layer with the F-value at each curve's rightmost
    point. The chart builder owns the iso-row injection; the desugar
    emits one extra dashed line layer + one text layer."""
    svg = ferrum.pr_chart(binary_source, iso_lines=True).to_svg()
    assert "<svg" in svg
    for f in ("F=0.2", "F=0.4", "F=0.6", "F=0.8"):
        assert f in svg, f"iso label {f!r} missing from SVG"


def test_mark_pr_annotate_ap_renders_text_label(binary_source):
    """annotate_ap=True emits a per-class text layer at fixed (x, y)
    with the formatted AP value, color-coded to match each curve."""
    svg = ferrum.pr_chart(binary_source, annotate_ap=True).to_svg()
    assert "<svg" in svg
    assert "AP = 0." in svg


def test_mark_pr_annotate_ap_and_iso_lines_compose(binary_source):
    """The two annotation features compose: AP labels + iso curves
    render in the same chart without collision."""
    svg = ferrum.pr_chart(
        binary_source,
        annotate_ap=True,
        iso_lines=True,
    ).to_svg()
    assert "AP = 0." in svg
    assert "F=0.6" in svg


def test_mark_calibration_renders(binary_source):
    cal = binary_source.calibration_curve(n_bins=10)
    svg = ferrum.Chart(cal).mark_calibration().to_svg()
    assert "<svg" in svg


def test_mark_gain_renders(binary_source):
    gain = binary_source.cumulative_gain()
    svg = ferrum.Chart(gain).mark_gain().to_svg()
    assert "<svg" in svg


def test_mark_lift_renders(binary_source):
    lift = binary_source.lift_curve()
    svg = ferrum.Chart(lift).mark_lift().to_svg()
    assert "<svg" in svg


def test_mark_discrimination_threshold_renders(binary_source):
    dt = binary_source.discrimination_threshold(n_thresholds=20)
    long = dt.unpivot(
        index="threshold",
        on=["precision", "recall", "f1", "queue_rate"],
        variable_name="metric",
        value_name="value",
    )
    svg = ferrum.Chart(long).mark_discrimination_threshold().to_svg()
    assert "<svg" in svg


def test_mark_discrimination_threshold_renders_threshold_line(binary_source):
    """threshold_line=True emits exactly one vertical mark_rule at the
    F1-best threshold -- not one horizontal rule per (threshold, metric)
    data row.

    Regression for the discrimination-threshold rule layer inheriting the
    chart-level y="value" encoding through the layered/desugar path: with
    n_thresholds=20 and 4 metrics, an un-fixed render draws 80 horizontal
    dashed rules (one per data row, spanning the full plot width) and zero
    vertical ones. This asserts the dashed threshold-line layer's own
    `<line>` element (identified by its distinctive
    ``stroke-dasharray="4,4"`` style, which no gridline/tick/axis line
    carries) is emitted exactly once, is vertical (x1 == x2), and spans a
    non-zero extent (y1 != y2) rather than being a degenerate point.
    """
    svg = ferrum.discrimination_threshold_chart(
        binary_source,
        threshold_line=True,
        n_thresholds=20,
    ).to_svg()
    assert "<svg" in svg

    dashed_lines = re.findall(r'<line\b[^>]*stroke-dasharray="4,4"[^>]*/>', svg)
    assert len(dashed_lines) == 1, (
        f"expected exactly 1 dashed threshold-line rule, got {len(dashed_lines)}: {dashed_lines}"
    )
    (line,) = dashed_lines
    x1, y1, x2, y2 = (
        float(re.search(rf'{attr}="([^"]+)"', line).group(1)) for attr in ("x1", "y1", "x2", "y2")
    )
    assert x1 == x2, f"threshold line must be vertical (x1 == x2); got x1={x1}, x2={x2}"
    assert y1 != y2, f"threshold line must span a non-zero height; got y1={y1}, y2={y2}"


def test_discrimination_threshold_chart_optimum_label_present_vs_absent(binary_source):
    """optimum_label=True renders the max-F1 annotation text on the
    figure-function path; optimum_label=False renders none of it.

    Regression for the F1-row case-mismatch bug (#96): `discrimination_
    threshold_chart` relabels the metric column to display names ("f1" ->
    "F1") before calling `mark_discrimination_threshold`, so the F1-optimum
    lookup in `_disc_threshold_prep` (`pl.col("metric") == "f1"`) matched
    zero rows and silently skipped the `_optimum_x`/`_optimum_y`/
    `_optimum_text` sentinel-column injection for every
    `discrimination_threshold_chart()` call -- not just `threshold_line=
    True` ones. Asserts on the actual rendered label text (not merely the
    presence of some `<text>` element, which the axis/legend/title already
    guarantee), and that the same text is genuinely absent when
    `optimum_label=False`.
    """
    svg_on = ferrum.discrimination_threshold_chart(
        binary_source,
        optimum_label=True,
        n_thresholds=20,
    ).to_svg()
    svg_off = ferrum.discrimination_threshold_chart(
        binary_source,
        optimum_label=False,
        n_thresholds=20,
    ).to_svg()

    label = re.search(r"max F1 = [\d.]+ @ t=[\d.]+", svg_on)
    assert label is not None, "optimum_label=True must render the max-F1 annotation text"
    assert svg_on.count(label.group(0)) == 1

    assert "max F1" not in svg_off, "optimum_label=False must not render the max-F1 annotation text"


# --- Figure-function tests (Task 16) --------------------------------


def test_roc_chart_figure_function(binary_source):
    svg = ferrum.roc_chart(binary_source).to_svg()
    assert "<svg" in svg


def test_roc_chart_from_model():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    svg = ferrum.roc_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    ).to_svg()
    assert "<svg" in svg


def test_pr_chart_figure_function(binary_source):
    svg = ferrum.pr_chart(binary_source).to_svg()
    assert "<svg" in svg


def test_pr_chart_per_class_false_routes_to_macro(multi_source):
    """``per_class=False`` should render a single averaged curve. The
    routed DataFrame from ``ModelSource.pr_curve(average='macro')``
    carries a single ``class='macro'`` value (no per-class rows), so the
    chart contains exactly one line. Regression test for the silent-
    discard bug fixed in the post-merge cleanup sweep.
    """
    chart = ferrum.pr_chart(multi_source, per_class=False)
    svg = chart.to_svg()
    assert "<svg" in svg
    # Verify the underlying DataFrame holds one summary class (the macro
    # summary) rather than several per-class curves.
    # After legend-label renaming, the single class value carries the AP score.
    classes = set(chart._data["class"].unique().to_list())
    assert len(classes) == 1
    assert next(iter(classes)).startswith("macro (AP =")


def test_pr_chart_per_class_false_micro(multi_source):
    chart = ferrum.pr_chart(multi_source, per_class=False, average="micro")
    classes = set(chart._data["class"].unique().to_list())
    assert len(classes) == 1
    assert next(iter(classes)).startswith("micro (AP =")


def test_calibration_chart_figure_function(binary_source):
    svg = ferrum.calibration_chart(binary_source, n_bins=5).to_svg()
    assert "<svg" in svg


def test_calibration_chart_multi_model_compared_source_passthrough():
    """calibration_chart accepts a pre-built ComparedModelSource as the
    sole positional argument, mirroring the sibling figure-function
    passthrough path (see test_compared_source_passthrough_via_figure in
    test_compare.py).
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    cms = ferrum.ModelSource.compare({"a": model, "b": model}, X, df["y"])
    chart = ferrum.calibration_chart(cms)
    assert "<svg" in chart.to_svg()


def test_gain_chart_figure_function(binary_source):
    svg = ferrum.gain_chart(binary_source).to_svg()
    assert "<svg" in svg


def test_lift_chart_figure_function(binary_source):
    svg = ferrum.lift_chart(binary_source).to_svg()
    assert "<svg" in svg


def test_discrimination_threshold_chart_figure_function(binary_source):
    svg = ferrum.discrimination_threshold_chart(
        binary_source,
        n_thresholds=20,
    ).to_svg()
    assert "<svg" in svg


# --- Visualizer tests (Task 17) --------------------------------------


def test_roc_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.ROCVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "auc_mean=" in repr(viz)
    assert 0.0 <= viz._metrics["auc_mean"] <= 1.0
    assert "<svg" in viz.show().to_svg()


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
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "ap_mean=" in repr(viz)
    assert 0.0 <= viz._metrics["ap_mean"] <= 1.0
    assert "<svg" in viz.show().to_svg()


def test_calibration_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.CalibrationVisualizer(model, n_bins=5).fit(
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "calibration_error=" in repr(viz)
    assert viz._metrics["calibration_error"] >= 0.0
    assert "<svg" in viz.show().to_svg()


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
    assert "<svg" in viz.show().to_svg()


def test_discrimination_threshold_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.DiscriminationThresholdVisualizer(
        model,
        n_thresholds=20,
    ).fit(df.select(["f0", "f1", "f2", "f3"]), df["y"])
    assert "best_threshold=" in repr(viz)
    assert 0.0 <= viz._metrics["best_threshold"] <= 1.0
    assert 0.0 <= viz._metrics["best_f1"] <= 1.0
    assert "<svg" in viz.show().to_svg()


def test_discrimination_threshold_visualizer_threshold_line():
    """threshold_line=True on the visualizer should render the same
    vertical F1-best rule as the figure-level function. Without this
    kwarg sklearn-protocol users could not get the rule at all.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.DiscriminationThresholdVisualizer(
        model,
        n_thresholds=20,
        threshold_line=True,
    ).fit(df.select(["f0", "f1", "f2", "f3"]), df["y"])
    svg = viz.show().to_svg()
    # mark_rule emits an SVG <line> for the vertical span.
    assert "<line " in svg


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
    row_sums = cm.group_by("actual").agg(pl.col("value").sum()).sort("actual")["value"].to_list()
    for s in row_sums:
        assert abs(s - 1.0) < 1e-9


def test_confusion_matrix_multiclass(multi_source):
    cm = multi_source.confusion_matrix()
    assert cm.height == 9  # 3x3 for 3-class


def test_confusion_matrix_caching(binary_source):
    assert binary_source.confusion_matrix() is binary_source.confusion_matrix()


def test_mark_confusion_renders(binary_source):
    cm = binary_source.confusion_matrix(normalize="true")
    svg = ferrum.Chart(cm).mark_confusion(normalize="true").to_svg()
    assert "<svg" in svg
    # Annotated by default — text labels appear (data labels + axis labels).
    assert svg.count("<text ") >= 4


def test_mark_confusion_annotate_off(binary_source):
    cm = binary_source.confusion_matrix(normalize=None)
    svg = ferrum.Chart(cm).mark_confusion(annotate=False).to_svg()
    assert "<svg" in svg


def test_confusion_matrix_chart_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.confusion_matrix_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "<svg" in chart.to_svg()


def test_confusion_matrix_chart_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.confusion_matrix_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
        normalize="true",
    )
    assert "<svg" in chart.to_svg()


# --- 10c: class prediction error (Task 19) ----------------------------


def test_mark_class_prediction_error_renders(multi_source):
    cm = multi_source.confusion_matrix(normalize=None)
    svg = ferrum.Chart(cm).mark_class_prediction_error().to_svg()
    assert "<svg" in svg


def test_class_prediction_error_chart_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.class_prediction_error_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "<svg" in chart.to_svg()


def test_class_prediction_error_chart_normalized():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.class_prediction_error_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
        normalize=True,
    )
    assert "<svg" in chart.to_svg()


# --- 10c: visualizers (Task 20) ---------------------------------------


def test_confusion_matrix_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ConfusionMatrixVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "accuracy=" in repr(viz)
    assert 0.0 <= viz._metrics["accuracy"] <= 1.0
    assert "<svg" in viz.show().to_svg()


def test_classification_report_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassificationReportVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "f1_macro=" in repr(viz)
    assert 0.0 <= viz._metrics["f1_macro"] <= 1.0
    assert "<svg" in viz.show().to_svg()


def test_class_prediction_error_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassPredictionErrorVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "accuracy=" in repr(viz)
    assert "<svg" in viz.show().to_svg()


def test_class_balance_visualizer_y_only():
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassBalanceVisualizer().fit(df["y"])
    assert "n_classes=" in repr(viz)
    assert "imbalance_ratio=" in repr(viz)
    assert viz._metrics["n_classes"] >= 2.0
    assert viz._metrics["imbalance_ratio"] >= 1.0
    assert "<svg" in viz.show().to_svg()


def test_class_balance_visualizer_xy_signature():
    df = load_dataset("multiclass_classification")
    # sklearn-shape .fit(X, y) should also work; X is ignored.
    X = df.select(["f0", "f1", "f2", "f3"])
    viz = ferrum.ClassBalanceVisualizer().fit(X, df["y"])
    assert "<svg" in viz.show().to_svg()


# ---------------------------------------------------------------------------
# Public figure-function smoke tests
# ---------------------------------------------------------------------------


def test_classification_report_chart_returns_chart():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    chart = ferrum.classification_report_chart(model, X, df["y"])
    assert "<svg" in chart.to_svg()


def test_class_balance_chart_returns_chart():
    df = load_dataset("multiclass_classification")
    chart = ferrum.class_balance_chart(df["y"])
    assert "<svg" in chart.to_svg()


def test_class_balance_chart_accepts_list():
    chart = ferrum.class_balance_chart([0, 1, 1, 2, 0, 2, 2])
    assert "<svg" in chart.to_svg()


def test_class_balance_chart_bars_colored_by_class():
    """Regression: bars must use distinct colors per class, not a single fill."""
    import re

    chart = ferrum.class_balance_chart([0, 1, 1, 2, 0, 2, 2])
    svg = chart.to_svg()
    fills = set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg))
    assert len(fills) >= 3, f"Expected ≥3 distinct fill colors for 3 classes; got {fills}"
