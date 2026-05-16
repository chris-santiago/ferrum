"""Model-diagnostic mark desugars, split by domain.

Re-exports every public desugar function so existing imports
(``from ferrum.marks.diagnostic import desugar_roc``) continue to work.
"""

from ferrum.marks.diagnostic._regression import (
    desugar_residuals,
    desugar_prediction_error,
)
from ferrum.marks.diagnostic._classification import (
    desugar_roc,
    desugar_pr,
    desugar_calibration,
    desugar_gain,
    desugar_lift,
    desugar_discrimination_threshold,
    desugar_confusion,
    desugar_class_prediction_error,
)
from ferrum.marks.diagnostic._explanation import (
    desugar_importance,
    desugar_shap_beeswarm,
    desugar_shap_bar,
    desugar_shap_waterfall,
    desugar_pdp,
)
from ferrum.marks.diagnostic._selection import (
    desugar_learning_curve,
    desugar_validation_curve,
    desugar_cv_scores,
    desugar_alpha_selection,
)
from ferrum.marks.diagnostic._clustering import (
    desugar_silhouette,
    desugar_pca_scree,
    desugar_pca_scree_with_threshold,
    desugar_intercluster_distance,
    desugar_decision_boundary,
)
from ferrum.marks.diagnostic._ranking import (
    desugar_rank1d,
    desugar_rank2d,
    desugar_parallel_coordinates,
)

__all__ = [
    "desugar_residuals",
    "desugar_prediction_error",
    "desugar_roc",
    "desugar_pr",
    "desugar_calibration",
    "desugar_gain",
    "desugar_lift",
    "desugar_discrimination_threshold",
    "desugar_confusion",
    "desugar_class_prediction_error",
    "desugar_importance",
    "desugar_shap_beeswarm",
    "desugar_shap_bar",
    "desugar_shap_waterfall",
    "desugar_pdp",
    "desugar_learning_curve",
    "desugar_validation_curve",
    "desugar_cv_scores",
    "desugar_alpha_selection",
    "desugar_pca_scree",
    "desugar_pca_scree_with_threshold",
    "desugar_silhouette",
    "desugar_intercluster_distance",
    "desugar_decision_boundary",
    "desugar_rank1d",
    "desugar_rank2d",
    "desugar_parallel_coordinates",
]
