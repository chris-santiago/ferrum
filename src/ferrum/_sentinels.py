"""Shared sentinel-column helpers used by mark-side and chart-builder paths.

A *sentinel column* is a per-row column whose only purpose is to feed a
downstream same-data overlay layer (``mark_rule``, ``mark_text``, …) so
that the renderer draws exactly one reference element per chart panel
instead of one per data row. The pattern is:

1. The caller appends a Float64 (or Utf8) column with one non-null
   value at row 0 and ``null`` on every other row.
2. A layer encoding that column reads it and emits a single mark — the
   nulls are skipped by the Rust mark renderers (``rule.rs``,
   ``text.rs`` skip non-finite values and null strings).

This module exists so the helpers live independently of any one
caller. Both chart-builder code (``_diagnostics/charts.py``) and mark
methods (``chart.py: mark_silhouette``, ``mark_shap_beeswarm``, …) reach
for the same helpers.
"""
from __future__ import annotations

import polars as pl


def _inject_constant(df: pl.DataFrame, name: str, value: float) -> pl.DataFrame:
    """Append a Float64 column with one non-null row at ``value``, rest null.

    Used so a downstream ``mark_rule(y=name)`` (or x=name) draws exactly
    one reference line. Rust's ``rule.rs`` skips non-finite values, so
    the ``N-1`` nulls produce no overdraw. Idempotent: returns ``df``
    unchanged when ``name`` is already a column.
    """
    if name in df.columns:
        return df
    n = df.height
    if n == 0:
        return df.with_columns(pl.Series(name, [], dtype=pl.Float64))
    series = pl.Series(name, [value] + [None] * (n - 1), dtype=pl.Float64)
    return df.with_columns(series)
