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

from typing import Any

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


# ---------------------------------------------------------------------------
# 10a builders
# ---------------------------------------------------------------------------


def _residuals_chart_from_source(
    source: Any,
    *,
    kind: str = "studentized",
    panels: Any = None,  # None / "single" / list of panel names
    theme: Any = None,
):
    """Build a residuals diagnostic chart from a ModelSource."""
    import ferrum
    df = source.predictions()
    if panels in (None, "single"):
        chart = ferrum.Chart(df).mark_residuals(kind=kind)
        if theme is not None:
            chart = chart.theme(theme)
        return chart

    panel_list = panels if isinstance(panels, list) else ["residuals_vs_fitted"]
    charts = [_residuals_panel(df, name) for name in panel_list]
    return _grid_panels(charts, theme=theme)


def _residuals_panel(df: pl.DataFrame, name: str):
    """One sub-panel of a multi-panel residuals chart. 10a ships only the
    canonical residuals_vs_fitted; the rest land in 10h alongside the
    leverage-aware Cook's distance path.
    """
    import ferrum
    if name == "residuals_vs_fitted":
        return ferrum.Chart(df).mark_residuals()
    if name == "qq":
        return ferrum.Chart(df).mark_qq().encode(x="studentized_residual")
    if name == "scale_location":
        d2 = df.with_columns(
            pl.col("studentized_residual").abs().sqrt().alias("sqrt_abs_resid")
        )
        return ferrum.Chart(d2).mark_point().encode(x="y_pred", y="sqrt_abs_resid")
    if name == "residuals_vs_leverage":
        return ferrum.Chart(df).mark_point().encode(x="y_pred", y="residual")
    raise ValueError(f"unknown residuals panel: {name!r}")


def _grid_panels(charts: list, theme: Any = None):
    """Compose up to 4 panels into a grid using Phase 8a hstack/vstack."""
    if len(charts) == 1:
        c = charts[0]
    elif len(charts) == 2:
        c = charts[0] | charts[1]
    elif len(charts) == 3:
        c = (charts[0] | charts[1]) & charts[2]
    else:
        c = (charts[0] | charts[1]) & (charts[2] | charts[3])
    if theme is not None:
        c = c.theme(theme)
    return c


def _prediction_error_chart_from_source(
    source: Any,
    *,
    identity_line: bool = True,
    theme: Any = None,
):
    """Build an actual-vs-predicted error chart from a ModelSource."""
    import ferrum
    df = source.predictions()
    chart = ferrum.Chart(df).mark_prediction_error(identity_line=identity_line)
    if theme is not None:
        chart = chart.theme(theme)
    return chart
