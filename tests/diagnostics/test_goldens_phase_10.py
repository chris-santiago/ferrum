"""Phase 10 SVG golden tests.

Single tier — all goldens render at the renderer's default 3-decimal-place
quantization (``fmt_f`` in ``crates/ferrum-core/src/render/svg.rs``). The
original plan proposed a tiered byte-identical / quantized split; that was
collapsed once it became clear the renderer already quantizes everything
via ``FLOAT_PRECISION = 3``.

Regenerate with ``FERRUM_REGENERATE_GOLDENS=1 pytest tests/diagnostics/test_goldens_phase_10.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture

_GOLDEN_ROOT = Path(__file__).parent.parent / "goldens" / "phase_10"
_REGENERATE = bool(os.environ.get("FERRUM_REGENERATE_GOLDENS"))


def _check_golden(svg: str, name: str) -> None:
    path = _GOLDEN_ROOT / f"{name}.svg"
    if _REGENERATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)
        if not _REGENERATE:
            pytest.skip(f"created new golden at {path}; rerun to verify")
        return
    expected = path.read_text()
    assert svg == expected, (
        f"Golden mismatch for {name!r}. "
        f"Set FERRUM_REGENERATE_GOLDENS=1 to regenerate after intentional changes."
    )


# --- 10a goldens ---


def test_golden_residuals_chart_regression_cook_outliers():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.residuals_chart(model, X, df["y"], cook_threshold="auto")
    _check_golden(chart.show_svg(), "residuals_chart_regression_cook_outliers")


def test_golden_residuals_chart_regression():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.residuals_chart(model, X, df["y"])
    svg = chart.show_svg()
    _check_golden(svg, "residuals_chart_regression")


def test_golden_prediction_error_regression_ci95():
    from ferrum._diagnostics.charts import _prediction_error_chart_from_source
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])
    chart = _prediction_error_chart_from_source(source, ci=0.95)
    _check_golden(chart.show_svg(), "prediction_error_regression_ci95")


def test_golden_prediction_error_regression_reference_band():
    from ferrum._diagnostics.charts import _prediction_error_chart_from_source
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])
    chart = _prediction_error_chart_from_source(source, reference_band=True)
    _check_golden(chart.show_svg(), "prediction_error_regression_reference_band")


def test_golden_prediction_error_regression():
    from ferrum._diagnostics.charts import _prediction_error_chart_from_source
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])
    chart = _prediction_error_chart_from_source(source)
    svg = chart.show_svg()
    _check_golden(svg, "prediction_error_regression")


# --- 10b goldens (classification curves) ---


def _binary_xy():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    return model, X, df["y"]


def _multi_xy():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    return model, X, df["y"]


def test_golden_roc_chart_binary():
    model, X, y = _binary_xy()
    chart = ferrum.roc_chart(model, X, y)
    _check_golden(chart.show_svg(), "roc_chart_binary")


def test_golden_roc_chart_binary_annotate_auc():
    model, X, y = _binary_xy()
    chart = ferrum.roc_chart(model, X, y, annotate_auc=True)
    _check_golden(chart.show_svg(), "roc_chart_binary_annotate_auc")


def test_golden_roc_chart_multiclass():
    model, X, y = _multi_xy()
    chart = ferrum.roc_chart(model, X, y)
    _check_golden(chart.show_svg(), "roc_chart_multiclass")


def test_golden_roc_chart_multiclass_annotate_auc():
    model, X, y = _multi_xy()
    chart = ferrum.roc_chart(model, X, y, annotate_auc=True)
    _check_golden(chart.show_svg(), "roc_chart_multiclass_annotate_auc")


def test_golden_pr_chart_binary_annotate_ap():
    model, X, y = _binary_xy()
    chart = ferrum.pr_chart(model, X, y, annotate_ap=True)
    _check_golden(chart.show_svg(), "pr_chart_binary_annotate_ap")


def test_golden_pr_chart_binary_iso_lines():
    model, X, y = _binary_xy()
    chart = ferrum.pr_chart(model, X, y, iso_lines=True)
    _check_golden(chart.show_svg(), "pr_chart_binary_iso_lines")


def test_golden_pr_chart_multiclass_annotate_ap_iso():
    model, X, y = _multi_xy()
    chart = ferrum.pr_chart(model, X, y, annotate_ap=True, iso_lines=True)
    _check_golden(chart.show_svg(), "pr_chart_multiclass_annotate_ap_iso")


def test_golden_pr_chart_multiclass_macro_average():
    """Multiclass PR chart with ``per_class=False`` — renders a single
    macro-averaged summary curve instead of per-class one-vs-rest curves.
    """
    model, X, y = _multi_xy()
    chart = ferrum.pr_chart(model, X, y, per_class=False)
    _check_golden(chart.show_svg(), "pr_chart_multiclass_macro_average")


def test_golden_pr_chart_binary():
    model, X, y = _binary_xy()
    chart = ferrum.pr_chart(model, X, y)
    _check_golden(chart.show_svg(), "pr_chart_binary")


def test_golden_calibration_chart_binary():
    model, X, y = _binary_xy()
    chart = ferrum.calibration_chart(model, X=X, y=y, n_bins=5)
    _check_golden(chart.show_svg(), "calibration_chart_binary")


def test_golden_gain_chart_binary():
    model, X, y = _binary_xy()
    chart = ferrum.gain_chart(model, X, y)
    _check_golden(chart.show_svg(), "gain_chart_binary")


def test_golden_lift_chart_binary():
    model, X, y = _binary_xy()
    chart = ferrum.lift_chart(model, X, y)
    _check_golden(chart.show_svg(), "lift_chart_binary")


def test_golden_discrimination_threshold_binary_threshold_line():
    model, X, y = _binary_xy()
    chart = ferrum.discrimination_threshold_chart(
        model, X, y, n_thresholds=20, threshold_line=True,
    )
    _check_golden(
        chart.show_svg(), "discrimination_threshold_binary_threshold_line",
    )


def test_golden_discrimination_threshold_binary():
    model, X, y = _binary_xy()
    chart = ferrum.discrimination_threshold_chart(model, X, y, n_thresholds=20)
    _check_golden(chart.show_svg(), "discrimination_threshold_binary")


# --- 10c goldens (classification matrices) ---


def test_golden_confusion_matrix_binary():
    model, X, y = _binary_xy()
    chart = ferrum.confusion_matrix_chart(model, X, y)
    _check_golden(chart.show_svg(), "confusion_matrix_binary")


def test_golden_confusion_matrix_multiclass():
    model, X, y = _multi_xy()
    chart = ferrum.confusion_matrix_chart(model, X, y, normalize="true")
    _check_golden(chart.show_svg(), "confusion_matrix_multiclass")


# The Task 19 golden was held back pending the mark_bar + Stack rendering
# fix (see handoff-phase9-golden-bugs.md). That fix landed alongside Task
# 20: scale_resolve.rs now reads the post-Stack y extent so cumulative
# tops are inside the y-scale domain, apply_stack emits a
# __stack_y_base__ column with per-segment lower bounds, and bar.rs
# draws each segment from base→top instead of every segment from y=0.
# Goldens visually verified before locking (rect+text PNG snapshot).


def test_golden_class_prediction_error_multiclass():
    model, X, y = _multi_xy()
    chart = ferrum.class_prediction_error_chart(model, X, y)
    _check_golden(chart.show_svg(), "class_prediction_error_multiclass")


# --- 10d goldens (feature importance + SHAP + PDP) ---


def _regression_xy():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return model, X, df["y"]


def test_golden_importance_chart_builtin():
    model, X, y = _regression_xy()
    chart = ferrum.importance_chart(model, X, y)
    _check_golden(chart.show_svg(), "importance_chart_builtin")


def test_golden_importance_chart_permutation():
    model, X, y = _regression_xy()
    chart = ferrum.importance_chart(
        model, X, y, method="permutation", random_state=0,
    )
    _check_golden(chart.show_svg(), "importance_chart_permutation")


def _ridge_xy():
    """Ridge regressor for SHAP goldens — LinearExplainer is deterministic
    across runs and platforms."""
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return model, X, df["y"]


def test_golden_shap_chart_beeswarm():
    model, X, _y = _ridge_xy()
    chart = ferrum.shap_chart(model, X, kind="beeswarm")
    _check_golden(chart.show_svg(), "shap_chart_beeswarm")


def test_golden_shap_chart_bar():
    model, X, _y = _ridge_xy()
    chart = ferrum.shap_chart(model, X, kind="bar")
    _check_golden(chart.show_svg(), "shap_chart_bar")


def test_golden_shap_chart_waterfall():
    model, X, _y = _ridge_xy()
    chart = ferrum.shap_chart(model, X, kind="waterfall", sample_idx=3)
    _check_golden(chart.show_svg(), "shap_chart_waterfall_sample3")


def test_golden_pdp_chart_individual_ice():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.pdp_chart(
        model, X, df["y"],
        features=["f0", "f1"], grid_resolution=15, kind="individual",
    )
    _check_golden(chart.show_svg(), "pdp_chart_individual_ice")


def test_golden_pdp_chart_both_centered():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.pdp_chart(
        model, X, df["y"],
        features=["f0", "f1"], grid_resolution=15,
        kind="both", center=True,
    )
    _check_golden(chart.show_svg(), "pdp_chart_both_centered")


def test_golden_pdp_chart_three_features():
    model, X, y = _regression_xy()
    chart = ferrum.pdp_chart(
        model, X, y, features=["f0", "f1", "f2"], grid_resolution=20,
    )
    _check_golden(chart.show_svg(), "pdp_chart_three_features")


# --- 10e goldens (model selection / CV curves) ---


def _ridge_for_selection():
    """Fresh Ridge — deterministic across runs given fixed random_state."""
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return Ridge(random_state=0), X, df["y"]


def test_golden_learning_curve_ridge():
    model, X, y = _ridge_for_selection()
    chart = ferrum.learning_curve_chart(model, X, y, cv=3, random_state=0)
    _check_golden(chart.show_svg(), "learning_curve_ridge")


def test_golden_validation_curve_ridge_alpha():
    model, X, y = _ridge_for_selection()
    chart = ferrum.validation_curve_chart(
        model, X, y, param="alpha", values=[0.1, 1.0, 10.0], cv=3,
    )
    _check_golden(chart.show_svg(), "validation_curve_ridge_alpha")


def test_golden_cv_scores_ridge_box():
    model, X, y = _ridge_for_selection()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="box")
    _check_golden(chart.show_svg(), "cv_scores_ridge_box")


def test_golden_alpha_selection_ridge():
    model, X, y = _ridge_for_selection()
    chart = ferrum.alpha_selection_chart(
        model, X, y, alphas=[0.01, 0.1, 1.0, 10.0], cv=3,
    )
    _check_golden(chart.show_svg(), "alpha_selection_ridge")


# --- 10f goldens (clustering / manifold / decision boundary) ---


def test_golden_silhouette_kmeans():
    from ferrum._diagnostics.charts import _silhouette_chart_from_source
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    source = ferrum.ModelSource(model, df)
    chart = _silhouette_chart_from_source(source)
    _check_golden(chart.show_svg(), "silhouette_kmeans_3cluster")


def test_golden_pca_scree_4comp():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.pca_scree_chart(model, df, threshold=0.95)
    _check_golden(chart.show_svg(), "pca_scree_4comp")


def test_golden_intercluster_distance_kmeans():
    # MDS is deterministic under a fixed random_state and an Arrow/PyO3
    # build pinned to a single platform; we ship this golden as
    # byte-identical (no quantization tier needed — FLOAT_PRECISION = 3
    # already collapses solver micro-jitter).
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    chart = ferrum.intercluster_distance_chart(
        model, df, k=3, method="mds", random_state=0,
    )
    _check_golden(chart.show_svg(), "intercluster_distance_mds_3cluster")


def test_golden_decision_boundary_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.decision_boundary_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
        features=(0, 1), grid_resolution=50, proba=True,
    )
    _check_golden(chart.show_svg(), "decision_boundary_binary_logistic")


def test_golden_decision_boundary_binary_with_scatter():
    """scatter=True overlay golden — true class colors and boundary
    colors share the same continuous viridis scale so misclassifications
    pop.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.decision_boundary_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
        features=(0, 1), grid_resolution=40, scatter=True,
    )
    _check_golden(
        chart.show_svg(), "decision_boundary_binary_with_scatter",
    )


# --- 10g goldens (feature ranking + parallel coordinates) ---


def test_golden_rank1d_shapiro_regression():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.rank_chart(df, rank="1d", algorithm="shapiro")
    _check_golden(chart.show_svg(), "rank1d_shapiro_regression")


def test_golden_rank2d_kendall_regression():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.rank_chart(df, rank="2d", algorithm="kendall")
    _check_golden(chart.show_svg(), "rank2d_kendall_regression")


def test_golden_parallel_coordinates_multiclass():
    df = load_dataset("multiclass_classification")
    chart = ferrum.parallel_coordinates_chart(
        df, features=["f0", "f1", "f2", "f3"], hue="y",
        rescale="minmax",
    )
    _check_golden(chart.show_svg(), "parallel_coordinates_multiclass")


# --- 10h goldens (multi-model compare) ---


def test_golden_roc_chart_compare_two_models():
    """ROC overlay with two named models. Uses the same fixture for both
    so the two curves overlap — the value here is regression detection on
    the multi-model serialization path, not a visual demonstration of
    diverging curves.
    """
    model, X, y = _binary_xy()
    chart = ferrum.roc_chart({"a": model, "b": model}, X, y)
    _check_golden(chart.show_svg(), "roc_chart_compare_two_models")
