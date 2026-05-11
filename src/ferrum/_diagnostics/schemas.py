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
