"""Private chart-builder functions used by figure functions + visualizers.

Each builder takes a ModelSource (or ComparedModelSource), calls the
appropriate derived-data method, and returns a fully-formed Chart over
the resulting DataFrame.

Builders are added incrementally per sub-batch. This module also hosts
small data-prep helpers shared across diagnostic marks (reference-line
injection, sort-by-axis), since several Phase 10 marks draw fixed-position
reference lines that Rust's mark_rule renders one-line-per-row.
"""
from __future__ import annotations

import polars as pl


def _inject_constant(df: pl.DataFrame, name: str, value: float) -> pl.DataFrame:
    """Append a column with one non-null row at `value`, rest null.

    Used so a downstream ``mark_rule(y=name)`` (or x=name) draws exactly one
    reference line. Rust's ``rule.rs`` skips non-finite values, so the
    N-1 nulls produce no overdraw.
    """
    if name in df.columns:
        return df
    n = df.height
    if n == 0:
        return df.with_columns(pl.Series(name, [], dtype=pl.Float64))
    series = pl.Series(name, [value] + [None] * (n - 1), dtype=pl.Float64)
    return df.with_columns(series)


def _sort_by(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Sort the frame ascending by `col` so a downstream ``mark_line`` over
    that column draws a monotonic polyline.
    """
    if col not in df.columns:
        return df
    return df.sort(col, nulls_last=True)
