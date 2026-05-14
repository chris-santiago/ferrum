"""Marks deferred beyond Phase 9.

Phase 8b's PHASE_8B_MARKS is empty.  Phase 9 removed `segment`.
Phase 11d closed arc (11d3), label (11d4), geoshape (11d5), and
added coord-awareness to image (11d6 — no longer deferred).
PHASE_9_PLUS_MARKS is now empty.
"""

from __future__ import annotations

# Phase 8b marks (Sub-batches E + F landed; list now empty).
PHASE_8B_MARKS: frozenset[str] = frozenset()

# All four Phase 9+ deferred marks are closed by Phase 11d.
PHASE_9_PLUS_MARKS: frozenset[str] = frozenset()


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
    >>> err = deferred_mark_error("unknown_mark")
    >>> "not implemented" in str(err)
    True
    """
    if mark_name in PHASE_8B_MARKS:
        return NotImplementedError(
            f"mark_{mark_name} is planned for Phase 8b. "
            f"See docs/superpowers/ferrum-phases.md for the roadmap."
        )
    if mark_name in PHASE_9_PLUS_MARKS:
        return NotImplementedError(
            f"mark_{mark_name} is planned for Phase 11+. "
            f"See docs/superpowers/ferrum-phases.md for the roadmap."
        )
    return NotImplementedError(f"mark_{mark_name} is not implemented.")
