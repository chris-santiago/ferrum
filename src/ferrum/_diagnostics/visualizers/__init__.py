"""§3.15 sklearn-protocol visualizers — added incrementally per sub-batch."""
from __future__ import annotations

from .base import FerrumVisualizer
from .classification import (
    CalibrationVisualizer,
    ClassificationReportVisualizer,
    ConfusionMatrixVisualizer,
    PRVisualizer,
    ROCVisualizer,
)
from .classification_extra import (
    ClassBalanceVisualizer,
    ClassPredictionErrorVisualizer,
    DiscriminationThresholdVisualizer,
)
from .explanation import FeatureImportancesVisualizer, SHAPVisualizer
from .regression import (
    CooksDistanceVisualizer,
    PredictionErrorVisualizer,
    ResidualsVisualizer,
)
from .clustering import (
    ElbowVisualizer,
    InterclusterDistanceVisualizer,
    ManifoldVisualizer,
    PCAVarianceVisualizer,
    SilhouetteVisualizer,
)
from .selection import (
    AlphaSelectionVisualizer,
    CVScoresVisualizer,
    LearningCurveVisualizer,
    ValidationCurveVisualizer,
)

__all__ = [
    "FerrumVisualizer",
    "ResidualsVisualizer",
    "PredictionErrorVisualizer",
    "CooksDistanceVisualizer",
    "ROCVisualizer",
    "PRVisualizer",
    "CalibrationVisualizer",
    "DiscriminationThresholdVisualizer",
    "ConfusionMatrixVisualizer",
    "ClassificationReportVisualizer",
    "ClassPredictionErrorVisualizer",
    "ClassBalanceVisualizer",
    "FeatureImportancesVisualizer",
    "SHAPVisualizer",
    "LearningCurveVisualizer",
    "ValidationCurveVisualizer",
    "CVScoresVisualizer",
    "AlphaSelectionVisualizer",
    "SilhouetteVisualizer",
    "ElbowVisualizer",
    "ManifoldVisualizer",
    "InterclusterDistanceVisualizer",
    "PCAVarianceVisualizer",
]
