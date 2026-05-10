"""Marks deferred to Phase 8b or Phase 9+. These exist as Chart methods that
raise NotImplementedError with a clear forward-pointer."""
from __future__ import annotations

# Phase 8b marks
PHASE_8B_MARKS = frozenset([
    # composite (Sub-batch E done)
    # heavy stat
    "contour", "violin", "qq", "raster", "swarm", "hex", "function",
])

# Phase 9+ marks
PHASE_9_PLUS_MARKS = frozenset([
    "arc", "image", "geoshape", "segment", "label",
])


def deferred_mark_error(mark_name: str) -> NotImplementedError:
    """Build an informative NotImplementedError for a deferred mark."""
    if mark_name in PHASE_8B_MARKS:
        return NotImplementedError(
            f"mark_{mark_name} is planned for Phase 8b. "
            f"See docs/superpowers/ferrum-phases.md for the roadmap."
        )
    if mark_name in PHASE_9_PLUS_MARKS:
        return NotImplementedError(
            f"mark_{mark_name} is planned for Phase 9+. "
            f"See docs/superpowers/ferrum-phases.md for the roadmap."
        )
    return NotImplementedError(f"mark_{mark_name} is not implemented.")
