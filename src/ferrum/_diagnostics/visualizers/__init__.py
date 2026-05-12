"""§3.15 sklearn-protocol visualizers — one diagnostic per class.

This module is one of two parallel diagnostic surfaces ferrum exposes on
top of the same chart-builder layer. Both are documented in
``ferrum-spec.md §3.14`` (figures) and ``§3.15`` (visualizers); both are
first-class API.

- **Visualizers** (this module). A ``Viz(model).fit(X, y).show()``
  sklearn-protocol object. Each visualizer subclasses
  ``FerrumVisualizer`` (``base.py``), implements ``_materialize`` to
  compute its derived data + headline ``_metrics``, and ``_build_chart``
  to render. The ``.fit`` / ``.predict`` / ``get_params`` /
  ``set_params`` surface keeps visualizers composable inside sklearn
  pipelines, ``GridSearchCV``, and external evaluators.
- **Figure functions** (``ferrum.figures``). A direct
  ``f(model, X, y) -> Chart`` call. No state, no separate ``.fit()``.
  Best for exploratory work, notebook one-liners, and any path where
  the user already has a fitted estimator and just wants a chart.

Both surfaces dispatch to the same ``_*_chart_from_source`` builders in
``ferrum._diagnostics.charts``; the dual API is sugar over a single
implementation tree, not a parallel pipeline.

Each visualizer wraps a ``ModelSource`` internally — the derived-data
layer is the same `.predictions() / .roc_curve() / .shap_values()`
contract that both surfaces consume.
"""

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
