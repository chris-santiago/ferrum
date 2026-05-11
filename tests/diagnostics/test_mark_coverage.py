"""Phase 10 mark coverage assertion (Task 42).

Locks down the contract that all 26 Phase 10 marks ship as ``Chart``
methods, none of them appears in the deferred-marks lists in
``src/ferrum/marks/deferred.py``, and the Phase 9+ deferred list is
unchanged from its post-9d state.
"""
from __future__ import annotations

import ferrum


PHASE_10_MARKS = frozenset({
    "residuals", "prediction_error", "confusion", "roc", "pr", "calibration",
    "gain", "lift", "importance", "shap_beeswarm", "shap_bar", "shap_waterfall",
    "pdp", "silhouette", "learning_curve", "validation_curve",
    "decision_boundary", "discrimination_threshold", "parallel_coordinates",
    "class_prediction_error", "pca_scree", "rank1d", "rank2d",
    "intercluster_distance", "cv_scores", "alpha_selection",
})


def test_phase_10_marks_count():
    assert len(PHASE_10_MARKS) == 26


def test_phase_10_marks_not_in_deferred():
    from ferrum.marks.deferred import PHASE_8B_MARKS, PHASE_9_PLUS_MARKS

    overlap = PHASE_10_MARKS & (PHASE_8B_MARKS | PHASE_9_PLUS_MARKS)
    assert not overlap, (
        f"Phase 10 marks still appear in a deferred list: {overlap}"
    )


def test_phase_10_marks_available_on_chart():
    # All Phase 10 marks ship as ``Chart.mark_<name>`` methods (not
    # module-level functions). Verify each method exists and is callable.
    missing: list[str] = []
    for mark_name in sorted(PHASE_10_MARKS):
        method_name = f"mark_{mark_name}"
        attr = getattr(ferrum.Chart, method_name, None)
        if attr is None or not callable(attr):
            missing.append(method_name)
    assert not missing, (
        f"Phase 10 marks not available as Chart methods: {missing}"
    )


def test_phase_9_plus_marks_unchanged():
    from ferrum.marks.deferred import PHASE_9_PLUS_MARKS

    assert PHASE_9_PLUS_MARKS == frozenset({"arc", "image", "geoshape", "label"})


def test_phase_8b_marks_empty():
    from ferrum.marks.deferred import PHASE_8B_MARKS

    assert PHASE_8B_MARKS == frozenset()
