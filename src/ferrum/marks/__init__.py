"""Marks — primitive + statistical (Phase 8a). Composite + heavy stat = Phase 8b.

Marks are normally accessed as Chart methods: chart.mark_point(...). The
mark functions below exist for direct construction in figure-level code paths.
"""
from ferrum.marks.base import MarkBase
from ferrum.marks.deferred import deferred_mark_error, PHASE_8B_MARKS, PHASE_9_PLUS_MARKS

__all__ = ["MarkBase", "deferred_mark_error", "PHASE_8B_MARKS", "PHASE_9_PLUS_MARKS"]
