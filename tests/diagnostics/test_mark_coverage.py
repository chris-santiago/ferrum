"""Phase 10 mark coverage assertion (Task 42).

Locks down the contract that all 26 Phase 10 marks ship as ``Chart``
methods.  The deferred-marks module was deleted once all marks shipped;
these tests verify the marks are available on Chart.
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
