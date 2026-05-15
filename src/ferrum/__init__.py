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
    ReferenceLine,
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
    Theme,
    set_default_theme,
    get_default_theme,
    theme_context,
)
import ferrum.themes as themes  # so users can write ferrum.themes.dark

from ferrum.chart import Chart
from ferrum.position import Identity, Dodge, Jitter, Stack
from ferrum.coord import (
    CoordFlip,
    CoordCartesian,
    CoordPolar,
    CoordGeo,
    CoordFixed,
)
from ferrum.layer import Layer
from ferrum.composition import HConcatChart, VConcatChart, JointChart, RepeatChart, ClusterMapChart
from ferrum.repeat import Repeat
from ferrum.annotations import (
    annotate_hline,
    annotate_vline,
    annotate_rect,
    annotate_text,
    annotate_arrow,
    AUCLabel,
    APLabel,
    BrierLabel,
    OutlierLabel,
)
from ferrum.title import Title
from ferrum.selection import (
    Selection,
    SelectionMark,
    ConditionalSpec,
    selection_point,
    selection_interval,
    selection_single,
    selection_multi,
    value,
)

# Phase 10 — model diagnostics
from ferrum._diagnostics import ComparedModelSource, ModelSource
from ferrum._diagnostics.visualizers import (
    CalibrationVisualizer,
    ClassBalanceVisualizer,
    ClassificationReportVisualizer,
    ClassPredictionErrorVisualizer,
    ConfusionMatrixVisualizer,
    CooksDistanceVisualizer,
    DiscriminationThresholdVisualizer,
    FeatureImportancesVisualizer,
    FerrumVisualizer,
    PRVisualizer,
    PredictionErrorVisualizer,
    ResidualsVisualizer,
    ROCVisualizer,
    SHAPVisualizer,
    SHAPBeeswarmVisualizer,
    SHAPBarVisualizer,
    SHAPWaterfallVisualizer,
    LearningCurveVisualizer,
    ValidationCurveVisualizer,
    CVScoresVisualizer,
    AlphaSelectionVisualizer,
    SilhouetteVisualizer,
    ElbowVisualizer,
    ManifoldVisualizer,
    InterclusterDistanceVisualizer,
    PCAVarianceVisualizer,
    Rank1DVisualizer,
    Rank2DVisualizer,
    ParallelCoordinatesVisualizer,
)
from ferrum.plots import (
    residuals_chart,
    roc_chart,
    pr_chart,
    calibration_chart,
    gain_chart,
    lift_chart,
    discrimination_threshold_chart,
    confusion_matrix_chart,
    class_prediction_error_chart,
    importance_chart,
    shap_chart,
    shap_beeswarm_chart,
    shap_bar_chart,
    shap_waterfall_chart,
    pdp_chart,
    learning_curve_chart,
    validation_curve_chart,
    cv_scores_chart,
    alpha_selection_chart,
    pca_scree_chart,
    cluster_diagnostics,
    intercluster_distance_chart,
    decision_boundary_chart,
    rank_chart,
    rank1d_chart,
    rank2d_chart,
    parallel_coordinates_chart,
    displot,
    catplot,
    relplot,
    lmplot,
    residplot,
    regplot,
    pairplot,
    heatmap,
    clustermap,
    jointplot,
)
import ferrum.plots as plots

import ferrum.encoding as encoding
from ferrum.encoding import (
    X,
    Y,
    X2,
    Y2,
    XError,
    YError,
    XError2,
    YError2,
    Theta,
    Radius,
    Color,
    Fill,
    Stroke,
    Opacity,
    FillOpacity,
    StrokeOpacity,
    StrokeWidth,
    StrokeDash,
    Size,
    Shape,
    Angle,
    Text,
    Detail,
    Tooltip,
    TooltipField,
    Href,
    Description,
    Key,
    Facet,
    FacetRow,
    FacetCol,
)

__version__ = "0.1.0"


def hconcat(*charts, spacing=10.0):
    """Horizontal concatenation of charts.

    Parameters
    ----------
    *charts : Chart or _ChartLike
        Two or more charts to place side-by-side.
    spacing : float, default 10.0
        Pixel gap between adjacent charts.

    Returns
    -------
    HConcatChart

    Examples
    --------
    >>> import ferrum as fm
    >>> left = fm.Chart(df).mark_point().encode(x="x", y="y")
    >>> right = fm.Chart(df).mark_bar().encode(x="category", y="count")
    >>> fm.hconcat(left, right)
    """
    from ferrum.composition import HConcatChart

    return HConcatChart(list(charts), spacing=spacing)


def vconcat(*charts, spacing=10.0):
    """Vertical concatenation of charts.

    Parameters
    ----------
    *charts : Chart or _ChartLike
        Two or more charts to stack top-to-bottom.
    spacing : float, default 10.0
        Pixel gap between adjacent charts.

    Returns
    -------
    VConcatChart

    Examples
    --------
    >>> import ferrum as fm
    >>> top = fm.Chart(df).mark_point().encode(x="x", y="y")
    >>> bottom = fm.Chart(df).mark_line().encode(x="time", y="value")
    >>> fm.vconcat(top, bottom)
    """
    from ferrum.composition import VConcatChart

    return VConcatChart(list(charts), spacing=spacing)

__all__ = [
    # Phase 1-7 core
    "Aggregate",
    "AggregateOp",
    "Bin",
    "Bin2D",
    "BoxStats",
    "ChartSpec",
    "Contour",
    "EncodingSpec",
    "ErrorExtent",
    "Hex",
    "Kde",
    "Kde2D",
    "Glm",
    "LetterValue",
    "LinearScale",
    "Linkage",
    "Logistic",
    "LogScale",
    "TimeScale",
    "SymlogScale",
    "OrdinalScale",
    "Outliers",
    "QQ",
    "QuantileScale",
    "Raster",
    "ReferenceLine",
    "Reorder",
    "Robust",
    "ThresholdScale",
    "Smooth",
    "Summary",
    "Swarm",
    "Unpivot",
    "Violin",
    "compute_layout",
    "process_batch",
    "render_png",
    "render_svg",
    "compose_svg_horizontal",
    "compose_svg_vertical",
    "compose_svg_grid",
    # Phase 8a
    "Chart",
    "Layer",
    "HConcatChart",
    "VConcatChart",
    "hconcat",
    "vconcat",
    # Phase 9
    "Repeat",
    "JointChart",
    "RepeatChart",
    "ClusterMapChart",
    "CoordFlip",
    "CoordCartesian",
    "CoordPolar",
    "CoordGeo",
    "CoordFixed",
    "Theme",
    "themes",
    "set_default_theme",
    "get_default_theme",
    "theme_context",
    "encoding",
    "X",
    "Y",
    "X2",
    "Y2",
    "XError",
    "YError",
    "XError2",
    "YError2",
    "Theta",
    "Radius",
    "Color",
    "Fill",
    "Stroke",
    "Opacity",
    "FillOpacity",
    "StrokeOpacity",
    "StrokeWidth",
    "StrokeDash",
    "Size",
    "Shape",
    "Angle",
    "Text",
    "Detail",
    "Tooltip",
    "TooltipField",
    "Href",
    "Description",
    "Key",
    "Facet",
    "FacetRow",
    "FacetCol",
    "annotate_hline",
    "annotate_vline",
    "annotate_rect",
    "annotate_text",
    "annotate_arrow",
    "AUCLabel",
    "APLabel",
    "BrierLabel",
    "OutlierLabel",
    "Title",
    # Phase 11c — selections
    "Selection",
    "SelectionMark",
    "ConditionalSpec",
    "selection_point",
    "selection_interval",
    "selection_single",
    "selection_multi",
    "value",
    # Phase 8b
    "ContinuousScheme",
    "continuous_palette",
    "Gradient",
    # Phase 9c position adjustments
    "Identity",
    "Dodge",
    "Jitter",
    "Stack",
    # Phase 9e figure-level convenience
    "plots",
    "displot",
    "catplot",
    "relplot",
    "lmplot",
    "residplot",
    "regplot",
    "pairplot",
    "heatmap",
    "clustermap",
    "jointplot",
    # Phase 10 — model diagnostics
    "ModelSource",
    "ComparedModelSource",
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
    "residuals_chart",
    "roc_chart",
    "pr_chart",
    "calibration_chart",
    "gain_chart",
    "lift_chart",
    "discrimination_threshold_chart",
    "confusion_matrix_chart",
    "class_prediction_error_chart",
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
    "pca_scree_chart",
    "cluster_diagnostics",
    "intercluster_distance_chart",
    "decision_boundary_chart",
    "rank_chart",
    "rank1d_chart",
    "rank2d_chart",
    "parallel_coordinates_chart",
]
