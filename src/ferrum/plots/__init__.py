"""Unified plot function package — all ferrum figure-level functions.

Organized by domain: classification, regression, distribution, matrix,
model_selection, clustering, explanation, ranking.
"""

from ferrum.plots.classification import (
    roc_chart,
    pr_chart,
    calibration_chart,
    gain_chart,
    lift_chart,
    discrimination_threshold_chart,
    confusion_matrix_chart,
    class_prediction_error_chart,
    classification_report_chart,
    class_balance_chart,
)
from ferrum.plots.regression import (
    residuals_chart,
    prediction_error_chart,
    cooks_distance_chart,
    lmplot,
    residplot,
    regplot,
)
from ferrum.plots.explanation import (
    importance_chart,
    shap_chart,
    shap_beeswarm_chart,
    shap_bar_chart,
    shap_waterfall_chart,
    pdp_chart,
)
from ferrum.plots.model_selection import (
    learning_curve_chart,
    validation_curve_chart,
    cv_scores_chart,
    alpha_selection_chart,
)
from ferrum.plots.clustering import (
    cluster_diagnostics,
    intercluster_distance_chart,
    pca_scree_chart,
    silhouette_chart,
    manifold_chart,
    elbow_chart,
)
from ferrum.plots.ranking import (
    rank_chart,
    rank1d_chart,
    rank2d_chart,
    parallel_coordinates_chart,
    decision_boundary_chart,
)
from ferrum.plots.distribution import (
    displot,
    catplot,
    relplot,
)
from ferrum.plots.matrix import (
    pairplot,
    heatmap,
    clustermap,
    jointplot,
)

__all__ = [
    "roc_chart",
    "pr_chart",
    "calibration_chart",
    "gain_chart",
    "lift_chart",
    "discrimination_threshold_chart",
    "confusion_matrix_chart",
    "class_prediction_error_chart",
    "classification_report_chart",
    "class_balance_chart",
    "residuals_chart",
    "prediction_error_chart",
    "cooks_distance_chart",
    "lmplot",
    "residplot",
    "regplot",
    "importance_chart",
    "shap_chart",
    "shap_beeswarm_chart",
    "shap_bar_chart",
    "shap_waterfall_chart",
    "pdp_chart",
    "learning_curve_chart",
    "validation_curve_chart",
    "cv_scores_chart",
    "alpha_selection_chart",
    "cluster_diagnostics",
    "intercluster_distance_chart",
    "pca_scree_chart",
    "silhouette_chart",
    "manifold_chart",
    "elbow_chart",
    "rank_chart",
    "rank1d_chart",
    "rank2d_chart",
    "parallel_coordinates_chart",
    "decision_boundary_chart",
    "displot",
    "catplot",
    "relplot",
    "pairplot",
    "heatmap",
    "clustermap",
    "jointplot",
]
