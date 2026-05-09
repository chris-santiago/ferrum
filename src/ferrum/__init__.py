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
    process_batch,
)

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
    "process_batch",
]
