"""Phase 10 — model-diagnostics adapter layer.

Public surface:
    ferrum.ModelSource           (re-exported here)
    ferrum.ComparedModelSource   (added in 10h)
"""

from __future__ import annotations

from .source import ComparedModelSource, ModelSource

__all__ = ["ComparedModelSource", "ModelSource"]
