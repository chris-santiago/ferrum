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


def test_golden_residuals_chart_regression():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.residuals_chart(model, X, df["y"])
    svg = chart.show_svg()
    _check_golden(svg, "residuals_chart_regression")


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


def test_golden_roc_chart_multiclass():
    model, X, y = _multi_xy()
    chart = ferrum.roc_chart(model, X, y)
    _check_golden(chart.show_svg(), "roc_chart_multiclass")


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


# NOTE: The Task 19 golden (class_prediction_error_multiclass) is held back
# pending the mark_bar + Stack rendering fix tracked in
# handoff-phase9-golden-bugs.md. Current renderer emits only 1 segment per
# stacked bar instead of one per (actual, predicted) cell, and
# offset='normalize' emits 0 rects. Locking a golden against that broken
# output would repeat the earlier mistake from heatmap_annot.svg.
