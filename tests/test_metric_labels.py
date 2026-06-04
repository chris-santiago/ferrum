"""Schwabish SB1 — AUCLabel / APLabel / BrierLabel + OutlierLabel + annotate_arrow.

Tests use small synthetic data so the SVG inspection is deterministic.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ferrum import (
    APLabel,
    AUCLabel,
    BrierLabel,
    Chart,
    OutlierLabel,
    annotate_arrow,
)


# ---- AUCLabel ----------------------------------------------------------------


def _roc_data(label: str = "c0", scale: float = 1.0):
    """Synthetic ROC: tpr = sqrt(fpr) yields AUC = 2/3 ≈ 0.667."""
    fpr = np.linspace(0, 1, 50)
    tpr = np.sqrt(fpr) * scale
    return pl.DataFrame({"fpr": fpr, "tpr": tpr, "class": [label] * len(fpr)})


def test_auc_label_default_format():
    df = _roc_data()
    chart = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line() + AUCLabel()
    svg = chart.to_svg()
    # tpr = sqrt(fpr) → AUC = 2/3 → 0.667
    assert "AUC = 0.667" in svg or "AUC = 0.666" in svg


def test_auc_label_custom_format_and_prefix():
    df = _roc_data()
    chart = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line() + AUCLabel(
        format=".2f", prefix="auc:"
    )
    svg = chart.to_svg()
    assert "auc:0.67" in svg


def test_auc_label_multi_class_emits_one_per_class():
    df = pl.concat([_roc_data("c0"), _roc_data("c1")])
    chart = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line() + AUCLabel()
    svg = chart.to_svg()
    assert svg.count("AUC =") == 2


# ---- APLabel -----------------------------------------------------------------


def _pr_data():
    """Synthetic PR: precision = 1 - 0.5 * recall yields AP ≈ 0.745."""
    recall = np.linspace(0, 1, 50)
    precision = 1.0 - 0.5 * recall
    return pl.DataFrame({"recall": recall, "precision": precision, "class": ["c0"] * len(recall)})


def test_ap_label_default():
    df = _pr_data()
    chart = Chart(df).encode(x="recall", y="precision", color="class").mark_line() + APLabel()
    svg = chart.to_svg()
    assert "AP = " in svg


# ---- BrierLabel --------------------------------------------------------------


def _calibration_data():
    """Perfectly-calibrated curve: predicted == observed → Brier ≈ 0."""
    p = np.linspace(0.05, 0.95, 19)
    obs = p.copy()
    return pl.DataFrame({"predicted": p, "observed": obs, "model": ["m0"] * len(p)})


def test_brier_label_default():
    df = _calibration_data()
    chart = Chart(df).encode(x="predicted", y="observed", color="model").mark_line() + BrierLabel()
    svg = chart.to_svg()
    assert "Brier = " in svg


# ---- OutlierLabel ------------------------------------------------------------


def test_outlier_label_threshold_3_emits_only_high_z():
    rng = np.random.default_rng(0)
    n = 1000
    residuals = rng.normal(0, 1, n)
    residuals[[100, 200, 300]] = 5.0
    residuals[[400, 500]] = -4.5
    obs_ids = [f"id_{i}" for i in range(n)]
    df = pl.DataFrame(
        {
            "fitted": np.linspace(0, 10, n),
            "residual": residuals,
            "obs_id": obs_ids,
        }
    )
    chart = Chart(df).encode(x="fitted", y="residual").mark_point() + OutlierLabel(
        threshold=3.0, field="residual", label_field="obs_id"
    )
    svg = chart.to_svg()
    for idx in (100, 200, 300, 400, 500):
        assert f">id_{idx}<" in svg


def test_outlier_label_max_labels_caps():
    rng = np.random.default_rng(0)
    n = 1000
    residuals = rng.normal(0, 1, n)
    residuals[:20] = 5.0
    df = pl.DataFrame(
        {
            "fitted": np.linspace(0, 10, n),
            "residual": residuals,
            "obs_id": [f"id_{i}" for i in range(n)],
        }
    )
    chart = Chart(df).encode(x="fitted", y="residual").mark_point() + OutlierLabel(
        threshold=3.0, field="residual", label_field="obs_id", max_labels=3
    )
    svg = chart.to_svg()
    count = sum(1 for i in range(20) if f">id_{i}<" in svg)
    assert count == 3


def test_outlier_label_no_outliers_returns_base_chart():
    rng = np.random.default_rng(1)
    n = 100
    df = pl.DataFrame(
        {
            "fitted": np.linspace(0, 10, n),
            "residual": rng.normal(0, 1, n),  # nothing above z=10
        }
    )
    chart = Chart(df).encode(x="fitted", y="residual").mark_point()
    out = chart + OutlierLabel(threshold=10.0)
    assert out is chart


# ---- annotate_arrow ----------------------------------------------------------


def test_annotate_arrow_without_label_uses_mark_segment():
    """Plain annotate_arrow yields a mark_segment chart."""
    chart = annotate_arrow(0.5, 1.5, 1.5, 3.5)
    assert chart._mark == "segment"
    assert "x" in chart._encoding and "y" in chart._encoding
    assert "x2" in chart._encoding and "y2" in chart._encoding


def test_annotate_arrow_with_label_is_vconcat_with_label_panel():
    """annotate_arrow(..., label=...) returns a VConcatChart whose right-hand
    chart is a mark_text annotation carrying the label string.

    (Single-row annotation charts cannot reliably render standalone due to
    degenerate scale domains under the post-themes-T4 5% padding default;
    real usage composes annotate_arrow with a richer chart that supplies
    scale extents. The structural test confirms the wiring is correct.)
    """
    from ferrum.composition import VConcatChart

    chart = annotate_arrow(0.5, 1.5, 1.5, 3.5, label="trend")
    assert isinstance(chart, VConcatChart)
    # Locate the mark_text label chart among the VConcat children.
    label_charts = [c for c in chart.charts if getattr(c, "_mark", None) == "text"]
    assert len(label_charts) == 1
    label_chart = label_charts[0]
    text_value = label_chart._data["_text"][0]
    assert text_value == "trend"


def test_annotate_arrow_stroke_passes_through():
    chart = annotate_arrow(0, 0, 1, 1, stroke="#ff0000")
    assert chart._mark_kwargs.get("stroke") == "#ff0000"
