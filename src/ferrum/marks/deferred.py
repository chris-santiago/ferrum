"""Marks deferred to Phase 9+. Phase 8b's PHASE_8B_MARKS is empty; Phase 9
removes `segment` from PHASE_9_PLUS_MARKS (now in 9d). The remaining four marks
(`arc`, `image`, `geoshape`, `label`) are not blocked by any §3.14 Group A figure
function and stay deferred consistent with the no-defer rule applying to spec
contracts ferrum currently advertises (see ferrum-spec.md §3.3 — these aren't
referenced by any §3.14 figure-level signature)."""
from __future__ import annotations

# Phase 8b marks (Sub-batches E + F landed; list now empty).
PHASE_8B_MARKS: frozenset[str] = frozenset()

# Phase 9+ marks
PHASE_9_PLUS_MARKS = frozenset([
    "arc", "image", "geoshape", "label",
])


def deferred_mark_error(mark_name: str) -> NotImplementedError:
    """Build an informative ``NotImplementedError`` for a deferred mark.

    Parameters
    ----------
    mark_name : str
        The mark name without the ``mark_`` prefix (e.g. ``"arc"``).

    Returns
    -------
    NotImplementedError
        Exception with a human-readable message indicating which phase the
        mark is planned for.

    Examples
    --------
    >>> err = deferred_mark_error("arc")
    >>> "Phase 9+" in str(err)
    True
    """
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
