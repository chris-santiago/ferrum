"""Per-domain split of the ``ModelSource`` adapter.

``ModelSource`` is too large to live in a single file; the seven
phase-10 domains (predictions, classification curves, importance,
model selection, clustering, ranking, plus ``ComparedModelSource``)
become independent mixin modules whose seam already exists as
``# --- 10x ---`` comment headers.  ``_base.BaseSource`` owns the
constructor, capability detection, and cache infrastructure; each
mixin contributes domain methods that use those bits via ``self``.

External callers still import from ``ferrum._diagnostics.source``
(re-export shim) — this package is the implementation home.
"""
from __future__ import annotations

from ._base import BaseSource

__all__ = ["BaseSource"]
