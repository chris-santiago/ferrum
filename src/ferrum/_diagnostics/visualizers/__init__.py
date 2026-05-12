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
from .explanation import (
    FeatureImportancesVisualizer,
    SHAPBarVisualizer,
    SHAPBeeswarmVisualizer,
    SHAPVisualizer,
    SHAPWaterfallVisualizer,
)
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
from .ranking import (
    ParallelCoordinatesVisualizer,
    Rank1DVisualizer,
    Rank2DVisualizer,
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
    "SHAPBeeswarmVisualizer",
    "SHAPBarVisualizer",
    "SHAPWaterfallVisualizer",
    "LearningCurveVisualizer",
    "ValidationCurveVisualizer",
    "CVScoresVisualizer",
    "AlphaSelectionVisualizer",
    "SilhouetteVisualizer",
    "ElbowVisualizer",
    "ManifoldVisualizer",
    "InterclusterDistanceVisualizer",
    "PCAVarianceVisualizer",
    "Rank1DVisualizer",
    "Rank2DVisualizer",
    "ParallelCoordinatesVisualizer",
]
