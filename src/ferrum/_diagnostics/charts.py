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


def _class_prediction_error_chart_from_source(
    source: Any,
    *,
    normalize: bool = False,
    theme: Any = None,
):
    """Build a stacked-bar class-prediction-error chart from a ModelSource.

    Reuses the unnormalized confusion-matrix output as the underlying
    long-form data.
    """
    import ferrum
    df = source.confusion_matrix(normalize=None)
    chart = ferrum.Chart(df).mark_class_prediction_error(normalize=normalize)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _classification_report_chart(source: Any, *, theme: Any = None):
    """Heatmap of per-class precision/recall/F1 (rect + text overlay).

    Long-form data: one row per (class, metric) cell with ``value`` and
    ``value_fmt``. Renders via the same rect-plus-text pattern as
    ``mark_confusion``.
    """
    from .deps import require_sklearn
    require_sklearn("ClassificationReportVisualizer")
    from sklearn.metrics import classification_report

    import ferrum

    y_true = source._y.to_numpy()
    y_pred = source._model.predict(source._X.to_numpy())
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0,
    )

    rows: list[dict] = []
    for cls_label, metrics in report.items():
        if cls_label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(metrics, dict):
            continue
        for m_name in ("precision", "recall", "f1-score"):
            val = float(metrics[m_name])
            rows.append({
                "class": str(cls_label),
                "metric": m_name,
                "value": val,
                "value_fmt": f"{val:.2f}",
            })
    df = pl.DataFrame(rows)

    heatmap = ferrum.Chart(df).mark_rect().encode(
        x="metric", y="class", color="value",
    )
    text = ferrum.Chart(df).mark_text().encode(
        x="metric", y="class", text="value_fmt",
    )
    chart = heatmap + text
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _class_balance_chart_from_dataframe(y_series: Any, *, theme: Any = None):
    """Bar chart of class counts.

    Operates on ``y`` alone (no model required). Computes per-class
    counts via polars ``group_by``.
    """
    import ferrum

    if isinstance(y_series, pl.Series):
        series = y_series
    else:
        series = pl.Series(list(y_series))

    counts = (
        pl.DataFrame({"y": series.cast(pl.Utf8, strict=False).to_list()})
        .group_by("y")
        .len()
        .rename({"len": "count"})
        .sort("y")
    )
    chart = ferrum.Chart(counts).mark_bar().encode(x="y", y="count")
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _importance_chart_from_source(
    source: Any,
    *,
    method: str = "builtin",
    top_k: int | None = 20,
    orient: str = "horizontal",
    error_bars: bool = True,
    random_state: int | None = None,
    theme: Any = None,
):
    """Build a feature-importance chart from a ModelSource.

    Computes ``imp_lower``/``imp_upper`` from ``importance`` ± ``std`` and
    truncates to the top-k rows by absolute importance. The value-axis
    scale domain is set to ``[0, max(imp_upper) * 1.05]`` so bars start
    at zero (the conventional bar-chart anchor) and the rightmost error
    bar has a small visual margin.
    """
    import ferrum

    df = source.importances(method=method, random_state=random_state)
    if top_k is not None:
        df = df.head(top_k)
    df = df.with_columns([
        (pl.col("importance") - pl.col("std")).alias("imp_lower"),
        (pl.col("importance") + pl.col("std")).alias("imp_upper"),
    ])

    upper_max = float(df["imp_upper"].max())
    lower_min = float(df["imp_lower"].min())
    domain_lo = min(0.0, lower_min)
    domain_hi = max(upper_max, 0.0) * 1.05 if upper_max > 0 else 1.0

    chart = ferrum.Chart(df).mark_importance(
        orient=orient,
        error_bars=error_bars,
        top_k=top_k,
        x_scale_domain=(domain_lo, domain_hi),
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


# ---------------------------------------------------------------------------
# 10d builders — SHAP family
# ---------------------------------------------------------------------------


def _shap_order_features(
    sv: pl.DataFrame, *, order: str, max_display: int,
) -> list[str]:
    """Return the top-`max_display` feature names ordered by `order`."""
    expr = pl.col("shap_value").abs()
    agg = expr.mean() if order == "abs_mean" else expr.max()
    ranked = (
        sv.group_by("feature")
        .agg(agg.alias("score"))
        .sort("score", descending=True)
        .head(max_display)
    )
    return ranked["feature"].to_list()


def _shap_beeswarm_chart_from_source(
    source: Any,
    *,
    max_display: int = 20,
    order: str = "abs_mean",
    background: Any = None,
    theme: Any = None,
):
    """Beeswarm chart: per-sample shap values colored by feature value."""
    import ferrum

    sv = source.shap_values(background=background)
    keep = _shap_order_features(sv, order=order, max_display=max_display)
    plot_df = sv.filter(pl.col("feature").is_in(keep))

    x_min = float(plot_df["shap_value"].min())
    x_max = float(plot_df["shap_value"].max())
    pad = max(abs(x_min), abs(x_max)) * 0.05 if (x_min < x_max) else 1.0
    domain = (x_min - pad, x_max + pad)

    chart = ferrum.Chart(plot_df).mark_shap_beeswarm(
        max_display=max_display, order=order, x_scale_domain=domain,
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _shap_bar_chart_from_source(
    source: Any,
    *,
    max_display: int = 20,
    background: Any = None,
    theme: Any = None,
):
    """SHAP aggregated bar chart: mean(|shap_value|) per feature."""
    import ferrum

    sv = source.shap_values(background=background)
    agg = (
        sv.group_by("feature")
        .agg(pl.col("shap_value").abs().mean().alias("abs_mean_shap"))
        .sort("abs_mean_shap", descending=True)
        .head(max_display)
    )
    x_max = float(agg["abs_mean_shap"].max())
    domain = (0.0, x_max * 1.05 if x_max > 0 else 1.0)
    chart = ferrum.Chart(agg).mark_shap_bar(
        max_display=max_display, x_scale_domain=domain,
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _shap_waterfall_chart_from_source(
    source: Any,
    *,
    sample_idx: int,
    max_display: int = 20,
    background: Any = None,
    theme: Any = None,
):
    """Waterfall chart for a single sample's SHAP contributions."""
    import ferrum
    import numpy as np

    sv = source.shap_values(background=background)
    one = sv.filter(pl.col("sample_id") == sample_idx)
    if one.height == 0:
        raise ValueError(
            f"shap_waterfall: sample_idx={sample_idx} not found in shap output "
            f"(have {sv['sample_id'].n_unique()} samples)."
        )
    # Order by descending |shap| and trim.
    ordered = one.sort(pl.col("shap_value").abs(), descending=True).head(max_display)
    sv_arr = ordered["shap_value"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(sv_arr)])
    plot_df = ordered.with_columns([
        pl.Series("x0", cum[:-1]),
        pl.Series("x1", cum[1:]),
        pl.when(pl.col("shap_value") >= 0)
        .then(pl.lit("positive"))
        .otherwise(pl.lit("negative"))
        .alias("shap_sign"),
    ])

    x_lo = float(min(cum.min(), 0.0))
    x_hi = float(max(cum.max(), 0.0))
    pad = max(abs(x_lo), abs(x_hi)) * 0.05 if (x_lo < x_hi) else 1.0
    domain = (x_lo - pad, x_hi + pad)

    chart = ferrum.Chart(plot_df).mark_shap_waterfall(
        sample_idx=sample_idx, max_display=max_display, x_scale_domain=domain,
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


# ---------------------------------------------------------------------------
# 10d builders — partial dependence
# ---------------------------------------------------------------------------


def _pdp_chart_from_source(
    source: Any,
    features: list,
    *,
    grid_resolution: int = 100,
    kind: str = "average",
    ice_alpha: float = 0.2,
    center: bool = False,
    theme: Any = None,
):
    """Partial-dependence chart: one polyline per feature."""
    import ferrum

    df = source.partial_dependence(
        features, grid_resolution=grid_resolution, kind=kind,
    )
    # Pre-sort ascending by feature_value within each feature so the line
    # layer renders monotonically (line.rs groups rows by color in batch
    # order, so the sort order matters).
    df = df.sort(["feature", "feature_value"])
    chart = ferrum.Chart(df).mark_pdp(
        kind=kind, ice_alpha=ice_alpha, center=center,
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


# ---------------------------------------------------------------------------
# 10e builders — model selection / CV curves
# ---------------------------------------------------------------------------


def _dedupe_aggregated(df: pl.DataFrame, *group_keys: str) -> pl.DataFrame:
    """Drop per-fold duplicate rows when only the aggregated (mean/lower/upper)
    columns are needed. Sorts ascending by the primary group key so a
    downstream line layer renders a monotonic polyline.
    """
    keep = df.unique(subset=list(group_keys), keep="first", maintain_order=True)
    return keep.sort(list(group_keys), nulls_last=True)


def _learning_curve_chart_from_source(
    source: Any,
    *,
    cv: int = 5,
    scoring: Any = None,
    train_sizes: Any = None,
    ci_style: str = "band",
    theme: Any = None,
):
    """Learning-curve chart: dedupe per (train_size, split), then ribbon+line."""
    import ferrum

    df = source.learning_curve(cv=cv, scoring=scoring, train_sizes=train_sizes)
    df = _dedupe_aggregated(df, "train_size", "split")
    chart = ferrum.Chart(df).mark_learning_curve(ci_style=ci_style)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _validation_curve_chart_from_source(
    source: Any,
    param: str,
    values: Any,
    *,
    cv: int = 5,
    scoring: Any = None,
    log_scale: Any = "auto",
    ci_style: str = "band",
    theme: Any = None,
):
    """Validation-curve chart: dedupe per (param_value, split), then ribbon+line.

    ``log_scale="auto"`` enables log when the parameter range spans more
    than two orders of magnitude (max / max(min, 1e-12) > 100).
    """
    import ferrum

    df = source.validation_curve(param, values, cv=cv, scoring=scoring)
    df = _dedupe_aggregated(df, "param_value", "split")
    vals = [float(v) for v in values]
    if log_scale == "auto":
        non_zero = [v for v in vals if v > 0]
        if len(non_zero) >= 2 and max(non_zero) / min(non_zero) > 100:
            is_log = True
        else:
            is_log = False
    else:
        is_log = bool(log_scale)
    chart = ferrum.Chart(df).mark_validation_curve(
        log_scale=is_log, ci_style=ci_style, param_label=param,
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _cv_scores_chart_from_source(
    source: Any,
    *,
    cv: int = 5,
    scoring: Any = None,
    kind: str = "box",
    split: str = "both",
    theme: Any = None,
):
    """Per-fold CV-score chart. ``kind="bar"`` pre-aggregates per split;
    ``"box"``/``"strip"`` leave raw per-fold rows for the mark layer.
    """
    import ferrum

    df = source.cv_scores(cv=cv, scoring=scoring)
    if split != "both":
        df = df.filter(pl.col("split") == split)
    if kind == "bar":
        df = (
            df.group_by("split")
            .agg(pl.col("score").mean())
            .sort("split")
        )
    chart = ferrum.Chart(df).mark_cv_scores(kind=kind, split=split)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _alpha_selection_chart_from_source(
    source: Any,
    alphas: Any,
    *,
    cv: int = 5,
    scoring: Any = None,
    log_scale: bool = True,
    highlight_best: bool = True,
    theme: Any = None,
):
    """Alpha-selection chart: dedupe per alpha (one row per alpha holds the
    aggregated mean_score). The Chart method injects ``_best_alpha`` when
    ``highlight_best=True``.
    """
    import ferrum

    df = source.alpha_selection(alphas, cv=cv, scoring=scoring)
    df = _dedupe_aggregated(df, "alpha")
    chart = ferrum.Chart(df).mark_alpha_selection(
        log_scale=log_scale, highlight_best=highlight_best,
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


# ---------------------------------------------------------------------------
# 10f builders — clustering / manifold / decision boundary
# ---------------------------------------------------------------------------


def _silhouette_chart_from_source(
    source: Any,
    *,
    k: int | None = None,
    theme: Any = None,
):
    """Silhouette chart from a ModelSource. The source method packs
    samples into a 0..n-1 ``y_position`` stack order so the bars render
    tightly per cluster.
    """
    import ferrum

    df = source.silhouette(k=k)
    chart = ferrum.Chart(df).mark_silhouette()
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _pca_scree_chart_from_source(
    source: Any,
    *,
    n_components: int | None = None,
    cumulative_line: bool = True,
    threshold: float | None = 0.95,
    theme: Any = None,
):
    """PCA scree chart with optional cumulative line + threshold rule."""
    import ferrum

    df = source.pca_variance(n_components=n_components)
    chart = ferrum.Chart(df).mark_pca_scree(
        cumulative_line=cumulative_line,
        threshold_line=threshold,
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _intercluster_distance_chart_from_source(
    source: Any,
    *,
    k: int,
    method: str = "mds",
    theme: Any = None,
):
    """Cluster-center 2D scatter sized by cluster count."""
    import ferrum

    df = source.intercluster_distance(k, method=method)
    chart = ferrum.Chart(df).mark_intercluster_distance()
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _decision_boundary_chart_from_source(
    source: Any,
    *,
    features: tuple = (0, 1),
    grid_resolution: int = 200,
    proba: bool = False,
    scatter: bool = True,
    theme: Any = None,
):
    """Decision-boundary heatmap + optional scatter overlay of (X, y).

    Pre-computes a grid_resolution × grid_resolution grid of x/x2/y/y2
    cell bounds and the model's prediction (class index when
    ``proba=False``, probability when ``proba=True``). The grid is fed to
    ``mark_decision_boundary`` (rect-based); when ``scatter=True`` and
    the source has ``y``, a ``mark_point`` layer is composed on top via
    the multi-data ``+`` compositor.
    """
    import ferrum
    import numpy as np

    X_np = source._X.to_numpy()
    feat_idx = tuple(
        source._feature_names.index(f) if isinstance(f, str) else int(f)
        for f in features
    )
    if len(feat_idx) != 2:
        raise ValueError(
            "decision_boundary_chart requires exactly 2 features; got "
            f"{len(feat_idx)}."
        )
    x_col = X_np[:, feat_idx[0]].astype(np.float64)
    y_col = X_np[:, feat_idx[1]].astype(np.float64)
    pad_x = (x_col.max() - x_col.min()) * 0.05
    pad_y = (y_col.max() - y_col.min()) * 0.05
    xs = np.linspace(
        x_col.min() - pad_x, x_col.max() + pad_x, int(grid_resolution),
    )
    ys = np.linspace(
        y_col.min() - pad_y, y_col.max() + pad_y, int(grid_resolution),
    )
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    xx, yy = np.meshgrid(xs, ys)
    grid = np.tile(X_np.mean(axis=0), (xx.size, 1))
    grid[:, feat_idx[0]] = xx.ravel()
    grid[:, feat_idx[1]] = yy.ravel()
    if proba and "predict_proba" in source._capabilities:
        z = source._model.predict_proba(grid)[:, 1].astype(np.float64)
    else:
        z = np.asarray(source._model.predict(grid)).astype(np.float64)
    flat_x = xx.ravel()
    flat_y = yy.ravel()
    grid_df = pl.DataFrame({
        "x": [float(v) - dx / 2 for v in flat_x],
        "x2": [float(v) + dx / 2 for v in flat_x],
        "y": [float(v) - dy / 2 for v in flat_y],
        "y2": [float(v) + dy / 2 for v in flat_y],
        "z": [float(v) for v in z],
    })
    chart = ferrum.Chart(grid_df).mark_decision_boundary(proba=proba)
    if scatter and source._y is not None:
        scatter_df = pl.DataFrame({
            "x": [float(v) for v in x_col],
            "y": [float(v) for v in y_col],
            "label": source._y.to_numpy().tolist(),
        })
        overlay = ferrum.Chart(scatter_df).mark_point().encode(
            x="x", y="y", color="label",
        )
        chart = chart + overlay
    if theme is not None:
        chart = chart.theme(theme)
    return chart
