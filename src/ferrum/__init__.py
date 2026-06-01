"""Ferrum — a statistical visualization library with a Rust core."""

from ferrum._core import (
    Aggregate,
    AggregateOp,
    BandScale,
    Bin,
    Bin2D,
    BinOrdinalScale,
    BoxStats,
    ChartSpec,
    ContinuousScheme,
    Contour,
    DivergingScale,
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
    PointScale,
    PowScale,
    QQ,
    QuantileScale,
    QuantizeScale,
    Raster,
    ReferenceLine,
    Reorder,
    Robust,
    SequentialScale,
    Smooth,
    SqrtScale,
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
from ferrum.render_config import RenderConfig
from ferrum.position import Identity, Dodge, Jitter, Stack
from ferrum.coord import (
    CoordFlip,
    CoordCartesian,
    CoordPolar,
    CoordGeo,
    CoordFixed,
)
from ferrum.layer import Layer
from ferrum.composition import (
    HConcatChart,
    VConcatChart,
    LayerChart,
    ConcatChart,
    JointChart,
    RepeatChart,
    ClusterMapChart,
)
from ferrum.repeat import Repeat
from ferrum.annotations import (
    annotate_hline,
    annotate_vline,
    annotate_rect,
    annotate_text,
    annotate_arrow,
    annotate_abline,
    AUCLabel,
    APLabel,
    BrierLabel,
    OutlierLabel,
)
from ferrum.axis import Axis
from ferrum.grid import Grid
from ferrum.legend import Legend
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

# Phase 12 — data transforms
from ferrum.transforms import (
    transform_filter,
    transform_calculate,
    transform_aggregate,
    transform_bin,
    transform_fold,
    transform_pivot,
    transform_join_aggregate,
    transform_window,
    transform_density,
    transform_regression,
    transform_loess,
    transform_impute,
    transform_flatten,
    transform_sample,
    transform_top_k,
    transform_stack,
    transform_timeunit,
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
    prediction_error_chart,
    cooks_distance_chart,
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
    silhouette_chart,
    manifold_chart,
    elbow_chart,
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

import ferrum.color as color
import ferrum.config as config

from ferrum.configure import (
    AxisConfig,
    LegendConfig,
    TitleConfig,
    GridConfig,
    PaddingConfig,
    ColorConfig,
    Configure,
)
from ferrum.format_presets import resolve_format
from ferrum.structural import SecondaryY, BreakAxis, Inset
from ferrum.annotation.coords import px, norm, PixelCoord, NormCoord
import ferrum.annotation as annotation
from ferrum.annotation import Annotate

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
    Theta2,
    Radius2,
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
    Url,
    Facet,
    FacetRow,
    FacetCol,
)

from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("ferrum-viz")


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


def layer(*charts, resolve=None, title=None):
    """Overlay charts on shared axes (same coordinate space).

    Parameters
    ----------
    *charts : Chart
        Two or more charts to overlay bottom-to-top.
    resolve : dict, optional
        Per-channel scale-sharing overrides — e.g.
        ``resolve={"color": "independent"}``.
    title : str, optional
        Title applied to the combined chart.

    Returns
    -------
    LayerChart

    Examples
    --------
    >>> import ferrum as fm
    >>> scatter = fm.Chart(df).mark_point().encode(x="x", y="y")
    >>> line = fm.Chart(df).mark_line().encode(x="x", y="y")
    >>> fm.layer(scatter, line)
    """
    from ferrum.composition import LayerChart

    return LayerChart(*charts, resolve=resolve, title=title)


def concat(*charts, columns=None, spacing=10.0, resolve=None):
    """Wrapping concatenation of charts in a grid.

    Parameters
    ----------
    *charts : Chart or _ChartLike
        Charts to arrange in a wrapping grid.
    columns : int, optional
        Maximum number of columns before wrapping.  Defaults to
        ``len(charts)`` (single row).
    spacing : float, default 10.0
        Pixel gap between adjacent cells.
    resolve : dict, optional
        Per-channel scale-sharing overrides.

    Returns
    -------
    ConcatChart

    Examples
    --------
    >>> import ferrum as fm
    >>> charts = [fm.Chart(df).mark_point().encode(x=c, y="y") for c in cols]
    >>> fm.concat(*charts, columns=2)
    """
    from ferrum.composition import ConcatChart

    return ConcatChart(*charts, columns=columns, spacing=spacing, resolve=resolve)


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
    "BandScale",
    "BinOrdinalScale",
    "DivergingScale",
    "LinearScale",
    "Linkage",
    "Logistic",
    "LogScale",
    "PointScale",
    "PowScale",
    "QuantizeScale",
    "SequentialScale",
    "SqrtScale",
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
    "RenderConfig",
    "Layer",
    "HConcatChart",
    "VConcatChart",
    "LayerChart",
    "ConcatChart",
    "hconcat",
    "vconcat",
    "layer",
    "concat",
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
    "Theta2",
    "Radius2",
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
    "Url",
    "Facet",
    "FacetRow",
    "FacetCol",
    "annotate_hline",
    "annotate_vline",
    "annotate_rect",
    "annotate_text",
    "annotate_arrow",
    "annotate_abline",
    "AUCLabel",
    "APLabel",
    "BrierLabel",
    "OutlierLabel",
    "Title",
    # Phase 12 — Axis, Grid, and Legend value classes
    "Axis",
    "Grid",
    "Legend",
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
    "prediction_error_chart",
    "cooks_distance_chart",
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
    "silhouette_chart",
    "manifold_chart",
    "elbow_chart",
    "decision_boundary_chart",
    "rank_chart",
    "rank1d_chart",
    "rank2d_chart",
    "parallel_coordinates_chart",
    # Phase 12 — data transforms
    "transform_filter",
    "transform_calculate",
    "transform_aggregate",
    "transform_bin",
    "transform_fold",
    "transform_pivot",
    "transform_join_aggregate",
    "transform_window",
    "transform_density",
    "transform_regression",
    "transform_loess",
    "transform_impute",
    "transform_flatten",
    "transform_sample",
    "transform_top_k",
    "transform_stack",
    "transform_timeunit",
    # Phase 12 — color and config modules
    "color",
    "config",
    # Declarative config surface
    "AxisConfig",
    "LegendConfig",
    "TitleConfig",
    "GridConfig",
    "PaddingConfig",
    "ColorConfig",
    "Configure",
    "resolve_format",
    "SecondaryY",
    "BreakAxis",
    "Inset",
    "px",
    "norm",
    "PixelCoord",
    "NormCoord",
    "annotation",
    "Annotate",
]
