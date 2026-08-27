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


# ---- Single-source metric-kind table (T4.7b) ---------------------------------


def test_metric_label_specs_single_source():
    """The metric-kind table is the one source shared by the direct-mark
    ``__radd__`` path and the figure-function explicit-field path.

    Both surfaces must read the same ``(value_fn, prefix)`` per kind: the
    per-class dataclass defaults derive their prefix from the table, and
    ``_apply_metric_label_explicit`` looks the kind up in the same table.
    A drift here is exactly the two-code-paths hazard MOD-05 flagged.
    """
    from ferrum._metric_labels import (
        APLabel,
        AUCLabel,
        BrierLabel,
        _METRIC_LABEL_SPECS,
        _ap_step,
        _brier_score,
        _trapezoid_auc,
    )

    assert _METRIC_LABEL_SPECS == {
        "auc": (_trapezoid_auc, "AUC = "),
        "ap": (_ap_step, "AP = "),
        "brier": (_brier_score, "Brier = "),
    }
    expected_fns = {"auc": _trapezoid_auc, "ap": _ap_step, "brier": _brier_score}
    # The dataclass prefix defaults are the table's prefixes (no second copy).
    for cls, kind in ((AUCLabel, "auc"), (APLabel, "ap"), (BrierLabel, "brier")):
        assert cls().prefix == _METRIC_LABEL_SPECS[kind][1]
        assert cls._kind == kind
        # __radd__ resolves its value function FROM the table: i.e. the lookup
        # ``_METRIC_LABEL_SPECS[self._kind][0]`` returns the expected fn for this
        # class. (Pins the resolution, not just that ``_kind`` matches a key.)
        assert _METRIC_LABEL_SPECS[cls._kind][0] is expected_fns[kind]


def test_metric_label_classes_registry_derived_consistent():
    """``_METRIC_LABEL_CLASSES`` is derived from the classes and stays in lockstep
    with ``_METRIC_LABEL_SPECS``: every spec kind has a class, every class's
    ``_kind`` is a spec key, and the explicit-field path constructs the right
    class per kind.
    """
    from ferrum._metric_labels import (
        APLabel,
        AUCLabel,
        BrierLabel,
        _METRIC_LABEL_CLASSES,
        _METRIC_LABEL_SPECS,
        _apply_metric_label_explicit,
    )

    # Registry is keyed by each class's own ``_kind`` (cannot desync).
    assert _METRIC_LABEL_CLASSES == {"auc": AUCLabel, "ap": APLabel, "brier": BrierLabel}
    # Bidirectional key agreement with the spec table.
    assert set(_METRIC_LABEL_CLASSES) == set(_METRIC_LABEL_SPECS)
    for kind, cls in _METRIC_LABEL_CLASSES.items():
        assert cls._kind == kind

    # The explicit-field path constructs the registry's class for each kind.
    base = Chart(_roc_data()).encode(x="fpr", y="tpr").mark_line()
    constructed: list[type] = []

    import ferrum._metric_labels as ml

    real_apply = ml._apply_metric_label

    def _spy(base_chart, label_obj, **kwargs):
        constructed.append(type(label_obj))
        return real_apply(base_chart, label_obj, **kwargs)

    ml._apply_metric_label = _spy
    try:
        for kind, cls in _METRIC_LABEL_CLASSES.items():
            constructed.clear()
            _apply_metric_label_explicit(base, kind, x_col="fpr", y_col="tpr")
            assert constructed == [cls], (kind, constructed, cls)
    finally:
        ml._apply_metric_label = real_apply


def test_metric_label_kind_is_classvar_not_a_field():
    """``_kind`` is a ``ClassVar`` so the frozen dataclass excludes it from
    ``fields()``/``eq``/``hash``; adding the annotation must not turn it into a
    dataclass field.
    """
    import dataclasses

    from ferrum._metric_labels import APLabel, AUCLabel, BrierLabel

    for cls in (AUCLabel, APLabel, BrierLabel):
        names = [f.name for f in dataclasses.fields(cls)]
        assert names == ["position", "format", "prefix"], (cls.__name__, names)
        assert "_kind" not in names


def test_metric_label_explicit_rejects_unknown_kind():
    """Figure-path entry point validates against the shared table."""
    import pytest

    from ferrum import Chart
    from ferrum._metric_labels import _apply_metric_label_explicit

    base = Chart(_roc_data()).encode(x="fpr", y="tpr").mark_line()
    with pytest.raises(
        ValueError, match=r"_apply_metric_label_explicit: label_kind must be one of"
    ):
        _apply_metric_label_explicit(base, "f1", x_col="fpr", y_col="tpr")


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


def test_annotate_arrow_with_label_is_plain_chart_with_label_annotation():
    """annotate_arrow(..., label=...) returns a plain Chart (GH #84) whose
    ``mark_segment`` layer draws the arrow standalone and whose
    ``_annotations`` carries the label as a real ``AnnotationText`` entry
    (so it renders standalone too), and whose ``_annotation_primitive`` is
    ``[AnnotationArrow, AnnotationText]`` so a ``+`` overlay carries both.

    Previously this composed via ``&`` and returned a ``VConcatChart``,
    which broke ``chart + annotate_arrow(..., label=...)`` (TypeError) and
    rendered the label in a separate panel below the arrow instead of
    overlaid in one panel.
    """
    from ferrum.annotation.primitives import AnnotationArrow, AnnotationText
    from ferrum.composition import VConcatChart

    chart = annotate_arrow(0.5, 1.5, 1.5, 3.5, label="trend")
    assert isinstance(chart, Chart)
    assert not isinstance(chart, VConcatChart)
    assert chart._mark == "segment"

    # Label renders standalone via a real `_annotations` entry.
    assert len(chart._annotations) == 1
    text_items = [item for item in chart._annotations[0].items if isinstance(item, AnnotationText)]
    assert len(text_items) == 1
    assert text_items[0].text == "trend"

    # `+` overlay carries both the arrow and its label via `_annotation_primitive`.
    assert isinstance(chart._annotation_primitive, list)
    assert len(chart._annotation_primitive) == 2
    assert isinstance(chart._annotation_primitive[0], AnnotationArrow)
    assert isinstance(chart._annotation_primitive[1], AnnotationText)
    assert chart._annotation_primitive[1].text == "trend"


def test_annotate_arrow_stroke_passes_through():
    chart = annotate_arrow(0, 0, 1, 1, stroke="#ff0000")
    assert chart._mark_kwargs.get("stroke") == "#ff0000"
