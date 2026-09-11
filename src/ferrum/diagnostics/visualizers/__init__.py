"""§3.15 sklearn-protocol visualizers — one diagnostic per class.

This module is one of two parallel diagnostic surfaces ferrum exposes on
top of the same chart-builder layer. Both are documented in
``ferrum-spec.md §3.14`` (figures) and ``§3.15`` (visualizers); both are
first-class API.

- **Visualizers** (this module). A ``Viz(model).fit(X, y).show()``
  sklearn-protocol object. Each visualizer subclasses
  ``FerrumVisualizer`` (``_base.py``), implements ``_materialize`` to
  compute its derived data + headline ``_metrics``, and ``_build_chart``
  to render. The public surface is exactly ``fit`` / ``score`` /
  ``has_score`` / ``show``: ``fit(X, y)`` builds the source and the
  chart, ``score(X, y)`` delegates to the wrapped estimator's own
  ``score`` (with ``has_score`` saying whether that number is real or
  the documented fallback), and ``show()`` returns the ``Chart``.
  There is deliberately no ``predict``; ``get_params`` / ``set_params``
  are not implemented either, so a visualizer is **not** clonable by
  ``sklearn.base.clone`` and cannot be dropped into a ``Pipeline`` or
  ``GridSearchCV`` — wrap the estimator, fit the visualizer on the
  search's best estimator instead.
- **Figure functions** (``ferrum.plots``). A direct
  ``f(model, X, y) -> Chart`` call. No state, no separate ``.fit()``.
  Best for exploratory work, notebook one-liners, and any path where
  the user already has a fitted estimator and just wants a chart.

Both surfaces dispatch to the same ``_*_chart_from_source`` builders in
``ferrum.plots.*`` domain modules; the dual API is sugar over a single
implementation tree, not a parallel pipeline.

Each visualizer wraps a ``ModelSource`` internally — the derived-data
layer is the same `.predictions() / .roc_curve() / .shap_values()`
contract that both surfaces consume.
"""

from __future__ import annotations

from ._base import FerrumVisualizer
from ._classification import (
    CalibrationVisualizer,
    ClassBalanceVisualizer,
    ClassificationReportVisualizer,
    ClassPredictionErrorVisualizer,
    ConfusionMatrixVisualizer,
    DiscriminationThresholdVisualizer,
    PRVisualizer,
    ROCVisualizer,
)
from ._explanation import (
    FeatureImportancesVisualizer,
    SHAPBarVisualizer,
    SHAPBeeswarmVisualizer,
    SHAPVisualizer,
    SHAPWaterfallVisualizer,
)
from ._regression import (
    CooksDistanceVisualizer,
    PredictionErrorVisualizer,
    ResidualsVisualizer,
)
from ._clustering import (
    ElbowVisualizer,
    InterclusterDistanceVisualizer,
    ManifoldVisualizer,
    PCAVarianceVisualizer,
    SilhouetteVisualizer,
)
from ._selection import (
    AlphaSelectionVisualizer,
    CVScoresVisualizer,
    LearningCurveVisualizer,
    ValidationCurveVisualizer,
)
from ._ranking import (
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
