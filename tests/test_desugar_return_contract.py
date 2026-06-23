"""Regression test for MARK-04: all desugar functions return MarkDesugarResult.

Regression: MARK-04 (T6.1a cohesion campaign) standardized 42 desugar
return annotations to ``-> MarkDesugarResult``. Annotations are not
runtime-enforced, so this test guards against any desugar regressing to
a tuple return or an annotation that lies about its runtime type.
"""

from __future__ import annotations

import numpy as np
import pytest

from ferrum._layer import MarkDesugarResult
from ferrum.marks.composite import (
    desugar_boxen,
    desugar_boxplot,
    desugar_errorband,
    desugar_errorbar,
    desugar_ribbon,
)
from ferrum.marks.diagnostic._classification import (
    desugar_calibration,
    desugar_class_prediction_error,
    desugar_confusion,
    desugar_discrimination_threshold,
    desugar_gain,
    desugar_lift,
    desugar_pr,
    desugar_roc,
)
from ferrum.marks.diagnostic._clustering import (
    desugar_decision_boundary,
    desugar_intercluster_distance,
    desugar_pca_scree,
    desugar_pca_scree_with_threshold,
    desugar_silhouette,
)
from ferrum.marks.diagnostic._explanation import (
    desugar_importance,
    desugar_pdp,
    desugar_shap_bar,
    desugar_shap_beeswarm,
    desugar_shap_waterfall,
)
from ferrum.marks.diagnostic._ranking import (
    desugar_parallel_coordinates,
    desugar_rank1d,
    desugar_rank2d,
)
from ferrum.marks.diagnostic._regression import (
    desugar_prediction_error,
    desugar_residuals,
)
from ferrum.marks.diagnostic._selection import (
    desugar_alpha_selection,
    desugar_cv_scores,
    desugar_learning_curve,
    desugar_validation_curve,
)
from ferrum.marks.heavy_stat import (
    desugar_contour,
    desugar_function,
    desugar_hex,
    desugar_qq,
    desugar_raster,
    desugar_swarm,
    desugar_violin,
)
from ferrum.marks.statistical import (
    desugar_density,
    desugar_histogram,
    desugar_smooth,
)

# ---------------------------------------------------------------------------
# Parametrize table: (desugar_callable, positional_args, keyword_args)
# Each entry mirrors the minimal-valid invocation shown in the module's doctest.
# ---------------------------------------------------------------------------

DESUGAR_CASES: list[tuple] = [
    # --- composite ----------------------------------------------------------
    ("desugar_boxplot", desugar_boxplot, ("species", "sepal_length"), {}),
    ("desugar_errorbar", desugar_errorbar, ("day", "tip"), {}),
    ("desugar_errorband", desugar_errorband, ("x", "y"), {}),
    ("desugar_ribbon", desugar_ribbon, ("x", "lower"), {"y2_field": "upper"}),
    ("desugar_boxen", desugar_boxen, ("species", "sepal_length"), {}),
    # --- statistical --------------------------------------------------------
    ("desugar_density", desugar_density, ("value",), {}),
    ("desugar_histogram", desugar_histogram, ("tip",), {}),
    ("desugar_smooth", desugar_smooth, ("x", "y"), {}),
    # --- heavy_stat ---------------------------------------------------------
    ("desugar_contour", desugar_contour, ("x", "y"), {}),
    ("desugar_violin", desugar_violin, ("species", "petal_length"), {"inner": None}),
    ("desugar_qq", desugar_qq, ("residuals",), {}),
    ("desugar_raster", desugar_raster, ("x", "y"), {}),
    ("desugar_hex", desugar_hex, ("x", "y"), {}),
    ("desugar_swarm", desugar_swarm, ("species", "petal_length"), {}),
    ("desugar_function", desugar_function, (np.sin,), {"domain": (0, 6.28)}),
    # --- diagnostic: classification -----------------------------------------
    ("desugar_roc", desugar_roc, (None, None), {}),
    ("desugar_pr", desugar_pr, (None, None), {}),
    ("desugar_calibration", desugar_calibration, (None, None), {}),
    ("desugar_gain", desugar_gain, (None, None), {}),
    ("desugar_lift", desugar_lift, (None, None), {}),
    (
        "desugar_discrimination_threshold",
        desugar_discrimination_threshold,
        (None, None),
        {"optimum_label": False},
    ),
    ("desugar_confusion", desugar_confusion, (None, None), {}),
    ("desugar_class_prediction_error", desugar_class_prediction_error, (None, None), {}),
    # --- diagnostic: regression ---------------------------------------------
    ("desugar_residuals", desugar_residuals, (None, None), {}),
    ("desugar_prediction_error", desugar_prediction_error, (None, None), {}),
    # --- diagnostic: selection (model selection / CV) -----------------------
    ("desugar_learning_curve", desugar_learning_curve, (None, None), {}),
    ("desugar_validation_curve", desugar_validation_curve, (None, None), {}),
    ("desugar_cv_scores_box", desugar_cv_scores, (None, None), {"kind": "box"}),
    ("desugar_cv_scores_bar", desugar_cv_scores, (None, None), {"kind": "bar"}),
    ("desugar_cv_scores_strip", desugar_cv_scores, (None, None), {"kind": "strip"}),
    ("desugar_alpha_selection", desugar_alpha_selection, (None, None), {}),
    # --- diagnostic: clustering ---------------------------------------------
    ("desugar_silhouette", desugar_silhouette, (None, None), {}),
    ("desugar_pca_scree", desugar_pca_scree, (None, None), {}),
    (
        "desugar_pca_scree_with_threshold",
        desugar_pca_scree_with_threshold,
        (None, None),
        {},
    ),
    ("desugar_intercluster_distance", desugar_intercluster_distance, (None, None), {}),
    ("desugar_decision_boundary", desugar_decision_boundary, (None, None), {}),
    # --- diagnostic: explanation -------------------------------------------
    ("desugar_importance", desugar_importance, (None, None), {}),
    ("desugar_shap_beeswarm", desugar_shap_beeswarm, (None, None), {}),
    ("desugar_shap_bar", desugar_shap_bar, (None, None), {}),
    (
        "desugar_shap_waterfall",
        desugar_shap_waterfall,
        (None, None),
        {"sample_idx": 0},
    ),
    ("desugar_pdp", desugar_pdp, (None, None), {}),
    # --- diagnostic: ranking -----------------------------------------------
    ("desugar_rank1d", desugar_rank1d, (None, None), {}),
    ("desugar_rank2d", desugar_rank2d, (None, None), {}),
    ("desugar_parallel_coordinates", desugar_parallel_coordinates, (None, None), {}),
]


@pytest.mark.parametrize(
    "name, fn, args, kwargs",
    [(name, fn, args, kw) for name, fn, args, kw in DESUGAR_CASES],
    ids=[name for name, *_ in DESUGAR_CASES],
)
def test_desugar_returns_mark_desugar_result(
    name: str,
    fn,
    args: tuple,
    kwargs: dict,
) -> None:
    """Each desugar function must return a MarkDesugarResult, not a tuple.

    Regression: MARK-04 (T6.1a cohesion campaign).
    """
    result = fn(*args, **kwargs)
    assert isinstance(result, MarkDesugarResult), (
        f"{name} returned {type(result).__name__!r}, expected MarkDesugarResult"
    )
    assert not isinstance(result, tuple), (
        f"{name} returned a tuple — regression to the old N-tuple protocol"
    )
