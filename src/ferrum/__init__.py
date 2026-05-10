"""Ferrum — a statistical visualization library with a Rust core."""

from ferrum._core import (
    Aggregate,
    AggregateOp,
    Bin,
    BoxStats,
    ChartSpec,
    Contour,
    EncodingSpec,
    ErrorExtent,
    Hex,
    Kde,
    Kde2D,
    LinearScale,
    LogScale,
    TimeScale,
    SymlogScale,
    OrdinalScale,
    Outliers,
    QQ,
    QuantileScale,
    Raster,
    Smooth,
    Summary,
    ThresholdScale,
    Violin,
    compute_layout,
    process_batch,
    render_png,
    render_svg,
    compose_svg_horizontal,
    compose_svg_vertical,
)

from ferrum.themes import (
    Theme, set_default_theme, get_default_theme, theme_context,
)
import ferrum.themes as themes  # so users can write ferrum.themes.dark

from ferrum.chart import Chart
from ferrum.coord import (
    CoordFlip, CoordCartesian, CoordPolar, CoordGeo, CoordFixed,
)
from ferrum.layer import Layer
from ferrum.composition import HConcatChart, VConcatChart
from ferrum.annotations import annotate_hline, annotate_vline, annotate_rect, annotate_text

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
    "Aggregate", "AggregateOp", "Bin", "BoxStats", "ChartSpec", "Contour", "EncodingSpec", "ErrorExtent", "Hex", "Kde", "Kde2D",
    "LinearScale", "LogScale", "TimeScale", "SymlogScale", "OrdinalScale",
    "Outliers", "QQ", "QuantileScale", "Raster", "ThresholdScale", "Smooth", "Summary", "Violin",
    "compute_layout", "process_batch", "render_png", "render_svg",
    "compose_svg_horizontal", "compose_svg_vertical",
    # Phase 8a
    "Chart", "Layer", "HConcatChart", "VConcatChart",
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
]
