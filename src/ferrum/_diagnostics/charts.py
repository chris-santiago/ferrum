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


# ---------------------------------------------------------------------------
# 10b builders — classification curves
# ---------------------------------------------------------------------------


def _color_field_for(df: pl.DataFrame, default: str) -> str:
    """Return ``'model'`` if a ``model`` column is present (compare-source
    path), otherwise the supplied default.
    """
    return "model" if "model" in df.columns else default


def _roc_chart_from_source(
    source: Any,
    *,
    per_class: bool = True,
    average: str | None = "macro",
    annotate_auc: bool = False,
    theme: Any = None,
):
    """Build an ROC chart from a ModelSource."""
    import ferrum
    df = source.roc_curve(average=None if per_class else average)
    chart = ferrum.Chart(df).mark_roc(
        average=None if per_class else average,
        annotate_auc=annotate_auc,
        color_field=_color_field_for(df, "class"),
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _pr_chart_from_source(
    source: Any,
    *,
    per_class: bool = True,
    annotate_ap: bool = False,
    iso_lines: bool = False,
    theme: Any = None,
):
    """Build a precision-recall chart from a ModelSource."""
    import ferrum
    del per_class  # ModelSource.pr_curve() always returns per-class for now
    df = source.pr_curve()
    chart = ferrum.Chart(df).mark_pr(
        annotate_ap=annotate_ap,
        iso_lines=iso_lines,
        color_field=_color_field_for(df, "class"),
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _calibration_chart_from_source(
    source: Any,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    theme: Any = None,
):
    """Build a calibration (reliability) chart from a ModelSource."""
    import ferrum
    df = source.calibration_curve(n_bins=n_bins, strategy=strategy)
    color = "model" if "model" in df.columns else None
    chart = ferrum.Chart(df).mark_calibration(
        n_bins=n_bins,
        strategy=strategy,
        color_field=color,
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _gain_chart_from_source(
    source: Any,
    *,
    theme: Any = None,
):
    """Build a cumulative-gain chart from a ModelSource."""
    import ferrum
    df = source.cumulative_gain()
    chart = ferrum.Chart(df).mark_gain(
        color_field=_color_field_for(df, "class"),
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _lift_chart_from_source(
    source: Any,
    *,
    theme: Any = None,
):
    """Build a lift chart from a ModelSource."""
    import ferrum
    df = source.lift_curve()
    chart = ferrum.Chart(df).mark_lift(
        color_field=_color_field_for(df, "class"),
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _confusion_chart_from_source(
    source: Any,
    *,
    normalize: str | None = "true",
    annotate: bool = True,
    theme: Any = None,
):
    """Build a confusion-matrix heatmap chart from a ModelSource."""
    import ferrum
    df = source.confusion_matrix(normalize=normalize)
    chart = ferrum.Chart(df).mark_confusion(
        normalize=normalize, annotate=annotate,
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _discrimination_threshold_chart_from_source(
    source: Any,
    *,
    n_thresholds: int = 50,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
    cv: Any = None,
    threshold_line: bool = False,
    theme: Any = None,
):
    """Build a discrimination-threshold chart from a ModelSource.

    The underlying DataFrame is unpivoted to long form
    ``(threshold, metric, value)`` for plotting.
    """
    import ferrum
    df = source.discrimination_threshold(n_thresholds=n_thresholds, cv=cv)
    long_df = df.unpivot(
        index="threshold",
        on=list(metrics),
        variable_name="metric",
        value_name="value",
    )
    chart = ferrum.Chart(long_df).mark_discrimination_threshold(
        metrics=metrics,
        n_thresholds=n_thresholds,
        threshold_line=threshold_line,
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart
