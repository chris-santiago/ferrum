"""§3.15 sklearn-protocol visualizers — added incrementally per sub-batch."""
from __future__ import annotations

from .base import FerrumVisualizer
from .regression import (
    CooksDistanceVisualizer,
    PredictionErrorVisualizer,
    ResidualsVisualizer,
)

__all__ = [
    "FerrumVisualizer",
    "ResidualsVisualizer",
    "PredictionErrorVisualizer",
    "CooksDistanceVisualizer",
]
