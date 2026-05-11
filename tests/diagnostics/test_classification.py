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


def test_mark_roc_annotate_auc_raises(binary_source):
    roc = binary_source.roc_curve()
    with pytest.raises(NotImplementedError, match="annotate_auc"):
        ferrum.Chart(roc).mark_roc(annotate_auc=True).show_svg()


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


def test_calibration_chart_multi_model_rejected():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    src = ferrum.ModelSource(model, X, df["y"])
    with pytest.raises(NotImplementedError, match="Multi-model calibration"):
        ferrum.calibration_chart(src, src)


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
