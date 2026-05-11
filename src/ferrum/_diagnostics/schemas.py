"""Polars schema constants for every derived-data DataFrame.

Every schema documents an optional `model: str` column appended by
`ComparedModelSource` and absent on plain `ModelSource`. Chart builders
check `"model" in df.columns` to add a `color="model"` encoding.

Schemas are added per sub-batch.
"""
from __future__ import annotations

import polars as pl

# Phase 10a — regression
SCHEMA_PREDICTIONS = pl.Schema({
    "y_true": pl.Float64,
    "y_pred": pl.Float64,
    "residual": pl.Float64,
    "studentized_residual": pl.Float64,
    "cooks_distance": pl.Float64,  # NaN for non-linear estimators
    # "model": pl.Utf8 (optional, present in ComparedModelSource output)
})

# Phase 10b — classification curves
SCHEMA_ROC_CURVE = pl.Schema({
    "fpr": pl.Float64,
    "tpr": pl.Float64,
    "threshold": pl.Float64,
    "class": pl.Utf8,
    "auc": pl.Float64,
    # "model": pl.Utf8 (optional)
})

SCHEMA_PR_CURVE = pl.Schema({
    "precision": pl.Float64,
    "recall": pl.Float64,
    "threshold": pl.Float64,
    "class": pl.Utf8,
    "ap": pl.Float64,
    # "model": pl.Utf8 (optional)
})

SCHEMA_CALIBRATION = pl.Schema({
    "mean_predicted": pl.Float64,
    "fraction_positive": pl.Float64,
    "count": pl.Int64,
    # "model": pl.Utf8 (optional)
})

SCHEMA_GAIN = pl.Schema({
    "percent_population": pl.Float64,
    "gain": pl.Float64,
    "class": pl.Utf8,
    # "model": pl.Utf8 (optional)
})

SCHEMA_LIFT = pl.Schema({
    "percent_population": pl.Float64,
    "lift": pl.Float64,
    "class": pl.Utf8,
    # "model": pl.Utf8 (optional)
})

SCHEMA_DISCRIMINATION_THRESHOLD = pl.Schema({
    "threshold": pl.Float64,
    "precision": pl.Float64,
    "recall": pl.Float64,
    "f1": pl.Float64,
    "queue_rate": pl.Float64,
    # "model": pl.Utf8 (optional)
})

# Phase 10c — classification matrices
SCHEMA_CONFUSION = pl.Schema({
    "actual": pl.Utf8,
    "predicted": pl.Utf8,
    "value": pl.Float64,
    "value_fmt": pl.Utf8,
    # "model": pl.Utf8 (optional)
})

# Phase 10d — feature importance
SCHEMA_IMPORTANCES = pl.Schema({
    "feature": pl.Utf8,
    "importance": pl.Float64,
    "std": pl.Float64,
    "rank": pl.Int64,
    # "model": pl.Utf8 (optional)
})

SCHEMA_PDP = pl.Schema({
    "feature": pl.Utf8,
    "feature_value": pl.Float64,
    "pd_value": pl.Float64,
    "sample_id": pl.Int64,
})

SCHEMA_SHAP_VALUES = pl.Schema({
    "sample_id": pl.Int64,
    "feature": pl.Utf8,
    "shap_value": pl.Float64,
    "feature_value": pl.Float64,
    "feature_value_normalized": pl.Float64,
})

# Phase 10e — model selection / CV curves
SCHEMA_LEARNING_CURVE = pl.Schema({
    "train_size": pl.Int64,
    "split": pl.Utf8,        # "train" | "test"
    "score": pl.Float64,
    "mean_score": pl.Float64,
    "std_score": pl.Float64,
    "lower": pl.Float64,
    "upper": pl.Float64,
    # "model": pl.Utf8 (optional)
})

SCHEMA_VALIDATION_CURVE = pl.Schema({
    "param_value": pl.Float64,
    "split": pl.Utf8,
    "score": pl.Float64,
    "mean_score": pl.Float64,
    "std_score": pl.Float64,
    "lower": pl.Float64,
    "upper": pl.Float64,
    # "model": pl.Utf8 (optional)
})

SCHEMA_CV_SCORES = pl.Schema({
    "fold": pl.Int64,
    "split": pl.Utf8,        # "train" | "test"
    "score": pl.Float64,
    # "model": pl.Utf8 (optional)
})

SCHEMA_ALPHA_SELECTION = pl.Schema({
    "alpha": pl.Float64,
    "fold": pl.Int64,
    "score": pl.Float64,
    "mean_score": pl.Float64,
    "std_score": pl.Float64,
    # "model": pl.Utf8 (optional)
})

# Phase 10f — clustering / manifold
# `y_position` is a sequential 0..n-1 stack-order index for the Rousseeuw
# silhouette plot. `sample_id` preserves the original X index (debug-useful;
# matches every other 10d-pre per-sample schema).
SCHEMA_SILHOUETTE = pl.Schema({
    "sample_id": pl.Int64,
    "y_position": pl.Int64,
    "cluster": pl.Utf8,
    "silhouette_value": pl.Float64,
})

SCHEMA_PCA_VARIANCE = pl.Schema({
    "component": pl.Int64,
    "explained_variance_ratio": pl.Float64,
    "cumulative_variance_ratio": pl.Float64,
})

# Embeddings: dim_0..dim_{k-1} + label (label is `y` from ModelSource when
# provided, else zeros). The number of dim columns depends on n_components,
# so we document the contract here rather than enumerating in a Schema.

SCHEMA_INTERCLUSTER_DISTANCE = pl.Schema({
    "cluster": pl.Utf8,
    "x": pl.Float64,
    "y": pl.Float64,
    "size": pl.Int64,
})

# Phase 10g — feature ranking
SCHEMA_RANK1D = pl.Schema({
    "feature": pl.Utf8,
    "score": pl.Float64,
    "rank": pl.Int64,
})

SCHEMA_RANK2D = pl.Schema({
    "feature_x": pl.Utf8,
    "feature_y": pl.Utf8,
    "correlation": pl.Float64,
})
