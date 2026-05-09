"""Ferrum — a statistical visualization library with a Rust core."""

from ferrum._core import (
    ChartSpec,
    EncodingSpec,
    LinearScale,
    LogScale,
    TimeScale,
    SymlogScale,
    OrdinalScale,
    QuantileScale,
    ThresholdScale,
    process_batch,
)

__version__ = "0.1.0"

__all__ = [
    "ChartSpec",
    "EncodingSpec",
    "LinearScale",
    "LogScale",
    "TimeScale",
    "SymlogScale",
    "OrdinalScale",
    "QuantileScale",
    "ThresholdScale",
    "process_batch",
]
