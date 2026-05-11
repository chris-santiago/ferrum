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
