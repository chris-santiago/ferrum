"""Ferrum — a statistical visualization library with a Rust core."""

from ferrum._core import (
    Aggregate,
    AggregateOp,
    Bin,
    ChartSpec,
    EncodingSpec,
    Kde,
    LinearScale,
    LogScale,
    TimeScale,
    SymlogScale,
    OrdinalScale,
    QuantileScale,
    Smooth,
    Summary,
    ThresholdScale,
    compute_layout,
    process_batch,
    render_png,
    render_svg,
)

from ferrum.themes import (
    Theme, set_default_theme, get_default_theme, theme_context,
)
import ferrum.themes as themes  # so users can write ferrum.themes.dark

__version__ = "0.1.0"

__all__ = [
    "Aggregate",
    "AggregateOp",
    "Bin",
    "ChartSpec",
    "EncodingSpec",
    "Kde",
    "LinearScale",
    "LogScale",
    "TimeScale",
    "SymlogScale",
    "OrdinalScale",
    "QuantileScale",
    "Smooth",
    "Summary",
    "ThresholdScale",
    "compute_layout",
    "process_batch",
    "render_png",
    "render_svg",
    "Theme",
    "themes",
    "set_default_theme",
    "get_default_theme",
    "theme_context",
]
