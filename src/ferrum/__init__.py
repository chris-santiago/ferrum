"""Ferrum — a statistical visualization library with a Rust core."""

from ferrum._core import (
    Aggregate,
    AggregateOp,
    Bin,
    Bin2D,
    BoxStats,
    ChartSpec,
    ContinuousScheme,
    Contour,
    EncodingSpec,
    ErrorExtent,
    Hex,
    Kde,
    Kde2D,
    Glm,
    LinearScale,
    LetterValue,
    Linkage,
    Logistic,
    LogScale,
    TimeScale,
    SymlogScale,
    OrdinalScale,
    Outliers,
    QQ,
    QuantileScale,
    Raster,
    Reorder,
    Robust,
    Smooth,
    Summary,
    Swarm,
    ThresholdScale,
    Unpivot,
    Violin,
    compute_layout,
    process_batch,
    render_png,
    render_svg,
    compose_svg_horizontal,
    compose_svg_vertical,
    compose_svg_grid,
)
from ferrum.schemes import continuous_palette, Gradient

from ferrum.themes import (
    Theme, set_default_theme, get_default_theme, theme_context,
)
import ferrum.themes as themes  # so users can write ferrum.themes.dark

from ferrum.chart import Chart
from ferrum.position import Identity, Dodge, Jitter, Stack
from ferrum.coord import (
    CoordFlip, CoordCartesian, CoordPolar, CoordGeo, CoordFixed,
)
from ferrum.layer import Layer
from ferrum.composition import HConcatChart, VConcatChart, JointChart, RepeatChart, ClusterMapChart
from ferrum.repeat import Repeat
from ferrum.annotations import annotate_hline, annotate_vline, annotate_rect, annotate_text

# Phase 10 — model diagnostics
from ferrum._diagnostics import ModelSource
from ferrum._diagnostics.visualizers import (
    CalibrationVisualizer,
    ClassBalanceVisualizer,
    ClassificationReportVisualizer,
    ClassPredictionErrorVisualizer,
    ConfusionMatrixVisualizer,
    CooksDistanceVisualizer,
    DiscriminationThresholdVisualizer,
    FerrumVisualizer,
    PRVisualizer,
    PredictionErrorVisualizer,
    ResidualsVisualizer,
    ROCVisualizer,
)
from ferrum.figures import (
    residuals_chart,
    roc_chart,
    pr_chart,
    calibration_chart,
    gain_chart,
    lift_chart,
    discrimination_threshold_chart,
    confusion_matrix_chart,
    class_prediction_error_chart,
)

# Phase 9e — figure-level convenience functions
from ferrum.figure import (
    displot, catplot, lmplot, residplot,
    pairplot, heatmap, clustermap, jointplot,
)
import ferrum.figure as figure  # so users can also do ferrum.figure.displot

import ferrum.encoding as encoding
from ferrum.encoding import (
    X, Y, X2, Y2, XError, YError, XError2, YError2, Theta, Radius,
    Color, Fill, Stroke, Opacity, FillOpacity, StrokeOpacity,
    StrokeWidth, StrokeDash, Size, Shape, Angle,
    Text, Detail, Tooltip, TooltipField, Href, Description, Key,
    Facet, FacetRow, FacetCol,
)

__version__ = "0.1.0"

__all__ = [
    # Phase 1-7 core
    "Aggregate", "AggregateOp", "Bin", "Bin2D", "BoxStats", "ChartSpec", "Contour", "EncodingSpec", "ErrorExtent", "Hex", "Kde", "Kde2D",
    "Glm", "LetterValue", "LinearScale", "Linkage", "Logistic", "LogScale", "TimeScale", "SymlogScale", "OrdinalScale",
    "Outliers", "QQ", "QuantileScale", "Raster", "Reorder", "Robust", "ThresholdScale", "Smooth", "Summary", "Swarm", "Unpivot", "Violin",
    "compute_layout", "process_batch", "render_png", "render_svg",
    "compose_svg_horizontal", "compose_svg_vertical", "compose_svg_grid",
    # Phase 8a
    "Chart", "Layer", "HConcatChart", "VConcatChart",
    # Phase 9
    "Repeat", "JointChart", "RepeatChart", "ClusterMapChart",
    "CoordFlip", "CoordCartesian", "CoordPolar", "CoordGeo", "CoordFixed",
    "Theme", "themes", "set_default_theme", "get_default_theme", "theme_context",
    "encoding",
    "X", "Y", "X2", "Y2", "XError", "YError", "XError2", "YError2",
    "Theta", "Radius",
    "Color", "Fill", "Stroke", "Opacity", "FillOpacity", "StrokeOpacity",
    "StrokeWidth", "StrokeDash", "Size", "Shape", "Angle",
    "Text", "Detail", "Tooltip", "TooltipField", "Href", "Description", "Key",
    "Facet", "FacetRow", "FacetCol",
    "annotate_hline", "annotate_vline", "annotate_rect", "annotate_text",
    # Phase 8b
    "ContinuousScheme", "continuous_palette", "Gradient",
    # Phase 9c position adjustments
    "Identity", "Dodge", "Jitter", "Stack",
    # Phase 9e figure-level convenience
    "figure",
    "displot", "catplot", "lmplot", "residplot",
    "pairplot", "heatmap", "clustermap", "jointplot",
    # Phase 10 — model diagnostics
    "ModelSource",
    "FerrumVisualizer",
    "ResidualsVisualizer", "PredictionErrorVisualizer", "CooksDistanceVisualizer",
    "ROCVisualizer", "PRVisualizer", "CalibrationVisualizer",
    "DiscriminationThresholdVisualizer",
    "ConfusionMatrixVisualizer", "ClassificationReportVisualizer",
    "ClassPredictionErrorVisualizer", "ClassBalanceVisualizer",
    "residuals_chart",
    "roc_chart", "pr_chart", "calibration_chart",
    "gain_chart", "lift_chart", "discrimination_threshold_chart",
    "confusion_matrix_chart", "class_prediction_error_chart",
]
