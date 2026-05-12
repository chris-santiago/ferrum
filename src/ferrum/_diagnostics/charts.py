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


def _inject_cook_outliers(
    df: pl.DataFrame,
    *,
    kind: str = "studentized",
    threshold: float | str = "auto",
    x_col: str = "y_pred",
) -> pl.DataFrame:
    """Inject ``_cook_outlier_x`` / ``_cook_outlier_y`` columns.

    For each row whose ``cooks_distance`` exceeds ``threshold``, the
    outlier-coordinate columns hold the ``(x_col, y_col)`` pair; all
    other rows are null so the residual mark's outlier-overlay layer
    skips them. ``x_col`` defaults to ``"y_pred"`` (the residuals-vs-
    fitted view); passing ``"leverage"`` produces an outlier overlay
    keyed on the leverage panel's x axis.

    ``threshold`` accepts:
    - a float — use that value directly.
    - ``"auto"`` — use the conventional ``4 / n`` rule (Hair et al.).
    """
    if df.height == 0 or "cooks_distance" not in df.columns:
        return df
    if threshold == "auto":
        thr = 4.0 / df.height
    else:
        thr = float(threshold)
    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"
    # Polars treats NaN > anything as True (NaN sorts after Infinity), so
    # the comparison must be guarded with `is_not_nan()` to keep non-linear
    # estimators (whose cooks_distance is NaN for every row) from
    # silently flagging every observation as an outlier.
    is_outlier = (
        pl.col("cooks_distance").is_not_nan()
        & pl.col("cooks_distance").is_not_null()
        & (pl.col("cooks_distance") > thr)
    )
    return df.with_columns(
        pl.when(is_outlier).then(pl.col(x_col)).otherwise(None).alias("_cook_outlier_x"),
        pl.when(is_outlier).then(pl.col(y_col)).otherwise(None).alias("_cook_outlier_y"),
    )


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
    cook_threshold: float | str | None = None,
    panels: Any = None,  # None / "single" / list of panel names
    theme: Any = None,
):
    """Build a residuals diagnostic chart from a ModelSource.

    When ``cook_threshold`` is set, identify observations whose
    ``cooks_distance`` exceeds the threshold and inject
    ``_cook_outlier_x`` / ``_cook_outlier_y`` columns (one per outlier
    row, null elsewhere). ``cook_threshold`` may be a float, or the
    literal ``"auto"`` to use the conventional ``4 / n`` rule, or
    ``None`` to skip outlier highlighting. The wrapped estimator must
    expose ``coef_`` for Cook's distance to be defined; non-linear
    estimators surface ``NaN`` Cook's D and silently produce no
    outliers (the threshold comparison is False for NaN).
    """
    import ferrum
    df = source.predictions()
    if panels in (None, "single"):
        if cook_threshold is not None:
            # Single-panel path uses mark_residuals' built-in overlay
            # layer keyed on the y_pred-based outlier columns.
            df = _inject_cook_outliers(df, kind=kind, threshold=cook_threshold)
        chart = ferrum.Chart(df).mark_residuals(
            kind=kind, cook_threshold=cook_threshold,
        )
        if theme is not None:
            chart = chart.theme(theme)
        return chart

    # Multi-panel path: each panel injects its own outlier columns with
    # the right x-axis encoding for that panel's coordinate system.
    panel_list = panels if isinstance(panels, list) else ["residuals_vs_fitted"]
    charts = [
        _residuals_panel(df, name, kind=kind, cook_threshold=cook_threshold)
        for name in panel_list
    ]
    return _grid_panels(charts, theme=theme)


def _residuals_panel(
    df: pl.DataFrame,
    name: str,
    *,
    kind: str = "studentized",
    cook_threshold: float | str | None = None,
):
    """One sub-panel of a multi-panel residuals chart.

    When ``cook_threshold`` is set, the residuals_vs_fitted and
    residuals_vs_leverage panels overlay an outlier-highlight layer
    keyed on each panel's own x axis (y_pred for residuals_vs_fitted,
    leverage for residuals_vs_leverage).
    """
    import ferrum

    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"

    if name == "residuals_vs_fitted":
        if cook_threshold is not None:
            df = _inject_cook_outliers(df, kind=kind, threshold=cook_threshold)
            return ferrum.Chart(df).mark_residuals(
                kind=kind, cook_threshold=cook_threshold,
            )
        return ferrum.Chart(df).mark_residuals(kind=kind)

    if name == "qq":
        return ferrum.Chart(df).mark_qq().encode(x="studentized_residual")

    if name == "scale_location":
        d2 = df.with_columns(
            pl.col("studentized_residual").abs().sqrt().alias("sqrt_abs_resid")
        )
        return ferrum.Chart(d2).mark_point().encode(x="y_pred", y="sqrt_abs_resid")

    if name == "residuals_vs_leverage":
        # Inject outlier columns BEFORE constructing the base chart so
        # base and overlay layers share the same DataFrame identity
        # (Chart.__add__ falls back to HConcat when data objects
        # differ). x_col="leverage" so the overlay sits at the right
        # x coordinate for this panel.
        if cook_threshold is not None:
            df = _inject_cook_outliers(
                df, kind=kind, threshold=cook_threshold, x_col="leverage",
            )
        base = ferrum.Chart(df).mark_point().encode(x="leverage", y=y_col)
        if cook_threshold is not None and "_cook_outlier_x" in df.columns:
            overlay = ferrum.Chart(df).mark_point(
                fill="#e15759",  # tableau red, matches mark_residuals overlay
                stroke="#000000",
                stroke_width=1.0,
                size=80.0,
            ).encode(x="_cook_outlier_x", y="_cook_outlier_y")
            return base + overlay
        return base

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
    ci: float | None = None,
    reference_band: bool = False,
    theme: Any = None,
):
    """Build an actual-vs-predicted error chart from a ModelSource.

    When ``ci`` or ``reference_band`` is set, computes a residual band
    and injects ``_pe_band_lo`` / ``_pe_band_hi`` columns:

    - ``ci`` (float in (0, 1)): band spans the central ``ci`` fraction
      of residuals — ``y_true + quantile_low`` to ``y_true +
      quantile_high`` where the quantiles bracket ``(1 - ci) / 2``.
    - ``reference_band=True`` (without ``ci``): band spans ±1 RMSE
      around the identity line.
    """
    import ferrum
    import numpy as np

    df = source.predictions().sort("y_true")
    if ci is not None or reference_band:
        residuals = (df["y_pred"] - df["y_true"]).to_numpy()
        if ci is not None:
            if not 0.0 < ci < 1.0:
                raise ValueError(
                    f"mark_prediction_error(ci={ci!r}) — expected a value in (0, 1)."
                )
            alpha = (1.0 - float(ci)) / 2.0
            q_lo = float(np.quantile(residuals, alpha))
            q_hi = float(np.quantile(residuals, 1.0 - alpha))
        else:
            sigma = float(np.sqrt(np.mean(residuals ** 2)))
            q_lo = -sigma
            q_hi = sigma
        df = df.with_columns(
            (pl.col("y_true") + q_lo).alias("_pe_band_lo"),
            (pl.col("y_true") + q_hi).alias("_pe_band_hi"),
        )
    chart = ferrum.Chart(df).mark_prediction_error(
        identity_line=identity_line,
        ci=ci,
        reference_band=reference_band,
    )
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
    if annotate_auc:
        df = _inject_curve_annotation(
            df,
            group_field="class",
            metric_field="auc",
            metric_label="AUC",
            prefix="_auc",
        )
    chart = ferrum.Chart(df).mark_roc(
        average=None if per_class else average,
        annotate_auc=annotate_auc,
        color_field=_color_field_for(df, "class"),
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _inject_curve_annotation(
    df: pl.DataFrame,
    *,
    group_field: str,
    metric_field: str,
    metric_label: str,
    prefix: str,
) -> pl.DataFrame:
    """Inject one-non-null-row-per-group annotation columns.

    Adds three columns:
    - ``{prefix}_label_x: Float64`` — x coordinate of the label anchor
      (fixed at 0.65 on the data axis so labels live in the lower-right
      region where ROC / PR curves typically asymptote).
    - ``{prefix}_label_y: Float64`` — staggered y per group (one slot
      every 0.06 of axis range, starting at 0.05) so multi-class labels
      don't overlap.
    - ``{prefix}_label: Utf8`` — formatted string
      ``"{group_value}: {metric_label} = {value:.3f}"``.

    All rows except one per group are null; ``mark_text`` skips nulls
    so exactly N labels render for N groups. Used by ``mark_roc``'s
    ``annotate_auc`` and ``mark_pr``'s ``annotate_ap`` paths.
    """
    n_rows = df.height
    if n_rows == 0:
        return df
    group_values = df[group_field].to_list()
    metric_values = df[metric_field].to_list()
    groups: list[str] = []
    for g in group_values:
        if g is None:
            continue
        if g not in groups:
            groups.append(str(g))
    label_x: list[float | None] = [None] * n_rows
    label_y: list[float | None] = [None] * n_rows
    label_text: list[str | None] = [None] * n_rows
    for i, g in enumerate(groups):
        # First row matching this group anchors the label.
        idx = group_values.index(g)
        label_x[idx] = 0.65
        label_y[idx] = 0.05 + i * 0.06
        value = float(metric_values[idx])
        label_text[idx] = f"{g}: {metric_label} = {value:.3f}"
    return df.with_columns(
        pl.Series(f"{prefix}_label_x", label_x, dtype=pl.Float64),
        pl.Series(f"{prefix}_label_y", label_y, dtype=pl.Float64),
        pl.Series(f"{prefix}_label", label_text, dtype=pl.Utf8),
    )


def _pr_chart_from_source(
    source: Any,
    *,
    per_class: bool = True,
    average: str | None = "macro",
    annotate_ap: bool = False,
    iso_lines: bool = False,
    theme: Any = None,
):
    """Build a precision-recall chart from a ModelSource.

    When ``per_class=True`` (default) the chart shows one curve per class
    via ``ModelSource.pr_curve(average=None)``. When ``per_class=False``
    the chart shows a single summary curve averaged across classes
    (``"macro"`` by default, configurable to ``"micro"`` or
    ``"weighted"``). Binary classifiers ignore ``per_class`` because
    ``pr_curve`` returns a single curve in either case.

    Pins both axes to ``[0, 1.05]`` (canonical sklearn / yellowbrick
    convention for PR diagrams) so curves render against the full
    precision-recall space, AP-annotation labels in the bottom-right
    corner stay in-bounds, and visual comparisons across models with
    very different baseline precisions remain calibrated.
    """
    import ferrum
    from ferrum.encoding import X, Y

    df = source.pr_curve(average=None if per_class else average)
    # Order matters: annotate first (so the per-class label rows live in
    # the original schema), then append iso-curve rows. `_inject_pr_iso_lines`
    # fills the new rows with nulls for every original column including the
    # annotation columns, which is exactly what mark_text wants.
    if annotate_ap:
        df = _inject_curve_annotation(
            df,
            group_field="class",
            metric_field="ap",
            metric_label="AP",
            prefix="_ap",
        )
    if iso_lines:
        df = _inject_pr_iso_lines(df)
    color_field = _color_field_for(df, "class")
    chart = (
        ferrum.Chart(df)
        .mark_pr(
            annotate_ap=annotate_ap,
            iso_lines=iso_lines,
            color_field=color_field,
        )
        .encode(
            x=X("recall", scale={"type": "linear", "domain": [0.0, 1.05]}),
            y=Y("precision", scale={"type": "linear", "domain": [0.0, 1.05]}),
        )
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _inject_pr_iso_lines(
    df: pl.DataFrame,
    *,
    f_scores: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
    n_points: int = 50,
) -> pl.DataFrame:
    """Append F-score iso-curve rows to a PR-curve DataFrame.

    For each F in ``f_scores``, parameterizes the iso curve
    ``precision = F * recall / (2 * recall - F)`` (the locus of points
    with the chosen F1) over the recall range ``(F/2, 1]`` where
    precision stays in ``(0, 1]``. Each iso curve gets ``n_points``
    rows with synthetic ``_iso_recall`` / ``_iso_precision`` /
    ``_iso_f`` columns plus one ``_iso_label_x`` / ``_iso_label_y`` /
    ``_iso_label`` row near the curve's right end for the F-value
    annotation. The original PR columns are extended with nulls on the
    new rows so ``mark_line`` skips them on the PR-curve layer (and
    vice-versa on the iso layer).
    """
    import numpy as np

    # Build the iso-curve rows.
    iso_rows: list[dict] = []
    for f in f_scores:
        recalls = np.linspace(f / 2 + 1e-3, 1.0, n_points)
        # precision = F * recall / (2 * recall - F)
        precisions = f * recalls / (2 * recalls - f)
        valid = (precisions > 0.0) & (precisions <= 1.0)
        for r, p in zip(recalls[valid], precisions[valid]):
            iso_rows.append({
                "_iso_recall": float(r),
                "_iso_precision": float(p),
                "_iso_f": f"F={f:.1f}",
                "_iso_label_x": None,
                "_iso_label_y": None,
                "_iso_label": None,
            })
        # Label anchor at the rightmost (recall ≈ 1) point of the iso curve.
        if iso_rows:
            iso_rows[-1]["_iso_label_x"] = float(recalls[valid][-1]) - 0.02
            iso_rows[-1]["_iso_label_y"] = float(precisions[valid][-1]) + 0.02
            iso_rows[-1]["_iso_label"] = f"F={f:.1f}"

    iso_df = pl.DataFrame(iso_rows)
    # Extend the original PR DataFrame with null iso columns.
    null_iso = pl.DataFrame({
        "_iso_recall": [None] * df.height,
        "_iso_precision": [None] * df.height,
        "_iso_f": [None] * df.height,
        "_iso_label_x": [None] * df.height,
        "_iso_label_y": [None] * df.height,
        "_iso_label": [None] * df.height,
    }, schema={
        "_iso_recall": pl.Float64,
        "_iso_precision": pl.Float64,
        "_iso_f": pl.Utf8,
        "_iso_label_x": pl.Float64,
        "_iso_label_y": pl.Float64,
        "_iso_label": pl.Utf8,
    })
    df = pl.concat([df, null_iso], how="horizontal")

    # Extend iso_df with null PR columns so the schemas align for vertical concat.
    pr_columns = [c for c in df.columns if c not in iso_df.columns]
    null_pr_for_iso: dict[str, list] = {}
    schema_for_iso: dict[str, pl.DataType] = {}
    for col in pr_columns:
        null_pr_for_iso[col] = [None] * iso_df.height
        schema_for_iso[col] = df.schema[col]
    iso_df = pl.concat(
        [iso_df, pl.DataFrame(null_pr_for_iso, schema=schema_for_iso)],
        how="horizontal",
    )

    # Vertical concat: original PR rows first, then iso rows.
    return pl.concat(
        [df.select(df.columns), iso_df.select(df.columns)],
        how="vertical_relaxed",
    )


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

    y_true = source.y.to_numpy()
    y_pred = source.model.predict(source.X.to_numpy())
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


_SHAP_ORDER_VALUES: frozenset[str] = frozenset({"abs_mean", "max"})


def _shap_order_features(
    sv: pl.DataFrame, *, order: str, max_display: int,
) -> list[str]:
    """Return the top-`max_display` feature names ordered by `order`."""
    if order not in _SHAP_ORDER_VALUES:
        raise ValueError(
            f"Unknown order {order!r}. "
            f"Accepted values: {sorted(_SHAP_ORDER_VALUES)}"
        )
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
    order: str = "abs_mean",
    background: Any = None,
    theme: Any = None,
):
    """SHAP aggregated bar chart: mean(|shap_value|) per feature.

    ``order`` selects the aggregation used both for sorting and as the
    bar's x-value. ``"abs_mean"`` aggregates by ``mean(|shap_value|)``
    (the conventional summary); ``"max"`` aggregates by
    ``max(|shap_value|)`` to emphasize features with high-impact outliers.
    The output column is named ``abs_mean_shap`` regardless of the
    aggregation so the downstream mark's data contract stays stable.
    """
    if order not in _SHAP_ORDER_VALUES:
        raise ValueError(
            f"Unknown order {order!r}. "
            f"Accepted values: {sorted(_SHAP_ORDER_VALUES)}"
        )
    import ferrum

    expr = pl.col("shap_value").abs()
    agg_expr = expr.mean() if order == "abs_mean" else expr.max()
    sv = source.shap_values(background=background)
    agg = (
        sv.group_by("feature")
        .agg(agg_expr.alias("abs_mean_shap"))
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
    order: str = "abs_mean",
    background: Any = None,
    theme: Any = None,
):
    """Waterfall chart for a single sample's SHAP contributions.

    Feature display order follows the global aggregation chosen by
    ``order``: ``"abs_mean"`` ranks features by ``mean(|shap_value|)``
    across the full dataset (matching the beeswarm and bar paths);
    ``"max"`` ranks by ``max(|shap_value|)``. The top ``max_display``
    features are kept and rendered in descending rank order; the
    waterfall's cumulative sum follows that same order.
    """
    import ferrum
    import numpy as np

    sv = source.shap_values(background=background)
    one = sv.filter(pl.col("sample_id") == sample_idx)
    if one.height == 0:
        raise ValueError(
            f"shap_waterfall: sample_idx={sample_idx} not found in shap output "
            f"(have {sv['sample_id'].n_unique()} samples)."
        )
    # Use the global aggregation (over all samples) to rank features —
    # matches the beeswarm / bar ordering so all three views agree on
    # which features matter for the model overall. Then project that
    # ranking onto the single sample's rows.
    ranked = _shap_order_features(sv, order=order, max_display=max_display)
    rank_map = {name: i for i, name in enumerate(ranked)}
    ordered = (
        one.filter(pl.col("feature").is_in(ranked))
        .with_columns(
            pl.col("feature").replace_strict(rank_map, return_dtype=pl.Int64).alias("_rank"),
        )
        .sort("_rank")
        .drop("_rank")
    )
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


def _pdp_center_curves(df: pl.DataFrame) -> pl.DataFrame:
    """Subtract each curve's value at its smallest feature_value so every
    polyline starts at zero. Anchors per ``(feature, sample_id)`` because
    each (feature, sample) has its own grid; the smallest grid value is
    the anchor.
    """
    anchor = (
        df.group_by(["feature", "sample_id"], maintain_order=True)
          .agg(pl.col("pd_value").first().alias("_pd_anchor"))
    )
    return (
        df.join(anchor, on=["feature", "sample_id"], how="left")
          .with_columns(
              (pl.col("pd_value") - pl.col("_pd_anchor")).alias("pd_value"),
          )
          .drop("_pd_anchor")
    )


def _pdp_split_kind_both(df: pl.DataFrame) -> pl.DataFrame:
    """Split ``pd_value`` into two layer-specific columns for ``kind="both"``.

    - ``_pd_ice_value``: per-sample value on ICE rows (sample_id >= 0),
      null on the average row so the ICE line layer skips it.
    - ``_pd_avg_value``: the average curve broadcast to every row (joined
      from the sample_id=-1 rows by ``(feature, feature_value)``). The
      average layer reads this on ALL rows but mark_line groups by color
      (feature) so duplicated rows merge into one polyline.

    Drops the original ``sample_id == -1`` rows so the duplicate-avg
    polyline collapses to one rendering; ``_pd_avg_value`` on the ICE
    rows already carries the full average curve.
    """
    avg_only = (
        df.filter(pl.col("sample_id") == -1)
          .select(["feature", "feature_value", "pd_value"])
          .rename({"pd_value": "_pd_avg_value"})
    )
    df = df.with_columns(
        pl.when(pl.col("sample_id") >= 0)
          .then(pl.col("pd_value"))
          .otherwise(None)
          .alias("_pd_ice_value"),
    )
    df = df.join(avg_only, on=["feature", "feature_value"], how="left")
    return df.filter(pl.col("sample_id") >= 0)


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
    """Partial-dependence chart: faceted, one polyline per feature panel.

    Each feature occupies its own facet panel because PDP curves for
    different features span unrelated x-axis ranges and units — sharing
    a single x axis collapses every curve onto the union range.

    For ``kind="individual"``: injects ``_sample_id_str`` (Utf8 cast of
    ``sample_id``) so ``mark_line``'s ``mark_style.detail`` grouping
    can produce one polyline per sample.

    For ``kind="both"``: injects two y-source columns —
    ``_pd_ice_value`` (per-sample value on ICE rows, null on the average
    row) and ``_pd_avg_value`` (the average curve broadcast to every
    row; non-null on ICE rows too so the average overlay renders the
    full grid). Each layer reads its own y column.

    When ``center=True``: subtracts the value at the smallest
    ``feature_value`` per ``(feature, sample_id)`` group so every
    polyline starts at 0.
    """
    import ferrum

    df = source.partial_dependence(
        features, grid_resolution=grid_resolution, kind=kind,
    )
    # Sort by (feature, sample_id, feature_value) so each polyline's
    # rows are contiguous and monotonic in feature_value within its
    # group — line.rs renders rows in batch order within each detail /
    # color group.
    df = df.sort(["feature", "sample_id", "feature_value"])

    if center:
        df = _pdp_center_curves(df)
    if kind in ("individual", "both"):
        # mark_line reads mark_style.detail as a Utf8 column name. Cast
        # sample_id to Utf8 so the detail grouping works.
        df = df.with_columns(
            pl.col("sample_id").cast(pl.Utf8).alias("_sample_id_str"),
        )
    if kind == "both":
        df = _pdp_split_kind_both(df)

    chart = (
        ferrum.Chart(df)
        .mark_pdp(
            kind=kind, ice_alpha=ice_alpha, center=center,
            color_field=None,  # one feature per facet — color is redundant
        )
        .facet(col="feature")
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
    if threshold_line:
        # Compute argmax(f1) from the un-melted DataFrame so we get the
        # exact threshold value (not the per-melted-row value of the F1
        # series). Inject _threshold_best as a sentinel column on the
        # long-form frame: one non-null row at the best threshold so
        # mark_rule renders exactly one vertical line.
        best_idx = int(df["f1"].arg_max() or 0)
        best_threshold = float(df["threshold"][best_idx])
        long_df = _inject_constant(
            long_df, "_threshold_best", best_threshold,
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
    """Cluster-center 2D scatter sized by cluster count.

    Adds 15% padding around the data x/y range so the large
    size-encoded points don't clip at the axis edges (MDS pushes
    cluster centers to the extreme corners of the embedded space).
    """
    import ferrum
    from ferrum.encoding import X, Y

    df = source.intercluster_distance(k, method=method)
    x_lo = float(df["x"].min() or 0.0)
    x_hi = float(df["x"].max() or 0.0)
    y_lo = float(df["y"].min() or 0.0)
    y_hi = float(df["y"].max() or 0.0)
    x_pad = max((x_hi - x_lo) * 0.15, 0.5)
    y_pad = max((y_hi - y_lo) * 0.15, 0.5)
    chart = (
        ferrum.Chart(df)
        .mark_intercluster_distance()
        .encode(
            x=X("x", scale={
                "type": "linear",
                "domain": [x_lo - x_pad, x_hi + x_pad],
            }),
            y=Y("y", scale={
                "type": "linear",
                "domain": [y_lo - y_pad, y_hi + y_pad],
            }),
        )
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _rank1d_chart_from_dataframe(
    df: pl.DataFrame,
    *,
    algorithm: str = "shapiro",
    orient: str = "horizontal",
    top_k: int | None = None,
    color_field: str | None = None,
    theme: Any = None,
):
    """Univariate rank1d chart over a pre-computed rank1d DataFrame.

    Truncates to ``top_k`` rows before rendering (the ModelSource has
    already sorted rows by descending score). ``algorithm`` is accepted
    for signature parity with ``rank_chart``; it doesn't affect the
    render — the DataFrame already carries the scores.

    Pins the score axis to ``[0, max_score * 1.05]`` (or ``[0, 1]`` for
    shapiro — W is bounded above by 1). Without an explicit zero
    baseline, ``mark_bar``'s ordinal-y path renders bars from the
    panel's left edge to ``to_pixel_f64(score)`` — when scores are
    tightly clustered above zero (typical for Shapiro W ∈ [0.98, 1.0]
    on well-behaved features), the auto-derived x domain starts at
    ``min(score)`` and the smallest-scored feature has a zero-width
    bar. Anchoring the domain at zero matches yellowbrick's default
    and makes relative magnitudes legible.
    """
    import ferrum
    from ferrum.encoding import X, Y

    if top_k is not None:
        df = df.head(int(top_k))
    max_score = float(df["score"].max() or 0.0)
    min_score = float(df["score"].min() or 0.0)
    if algorithm == "shapiro":
        x_domain = [0.0, 1.0]
    elif min_score < 0.0:
        # Negative scores (rare — only seen for raw covariance without
        # abs); anchor at ``min - pad`` instead.
        pad = max(abs(min_score), abs(max_score)) * 0.05
        x_domain = [min_score - pad, max_score + pad]
    else:
        x_domain = [0.0, max_score * 1.05 if max_score > 0.0 else 1.0]

    chart = ferrum.Chart(df).mark_rank1d(
        orient=orient, color_field=color_field,
    )
    if orient == "horizontal":
        chart = chart.encode(
            x=X("score", scale={"type": "linear", "domain": x_domain}),
            y=Y("feature"),
        )
    else:
        chart = chart.encode(
            x=X("feature"),
            y=Y("score", scale={"type": "linear", "domain": x_domain}),
        )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _rank2d_chart_from_dataframe(
    df: pl.DataFrame,
    *,
    algorithm: str = "pearson",
    annot: bool = True,
    theme: Any = None,
):
    """Pairwise rank2d heatmap chart over a pre-computed rank2d DataFrame.

    When ``annot=True``, appends a ``correlation_fmt`` (Utf8) column
    holding ``"{:.2f}".format(correlation)`` per row so the text-overlay
    layer can render compact 2-dp labels without invoking Rust-side
    number formatting per cell.
    """
    import ferrum

    del algorithm
    if annot and "correlation_fmt" not in df.columns:
        df = df.with_columns(
            pl.col("correlation").map_elements(
                lambda v: f"{v:.2f}", return_dtype=pl.Utf8,
            ).alias("correlation_fmt"),
        )
    chart = ferrum.Chart(df).mark_rank2d(annot=annot)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _coerce_to_polars(data: Any) -> pl.DataFrame:
    """Coerce a polars / pandas / 2D-numpy input into a polars DataFrame."""
    import numpy as np

    if isinstance(data, pl.DataFrame):
        return data
    if hasattr(data, "to_numpy") and hasattr(data, "columns"):
        return pl.from_pandas(data)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"data must be a 2D array; got shape {arr.shape}")
    return pl.DataFrame({
        f"f{j}": arr[:, j].tolist() for j in range(arr.shape[1])
    })


def _resolve_pc_features(
    df: pl.DataFrame, features: list[str] | None, hue: str | None,
) -> list[str]:
    """Resolve the parallel-coordinates feature list and validate it exists."""
    if features is None:
        features = [c for c in df.columns if c != hue]
    else:
        features = [str(c) for c in features]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"parallel_coordinates: features {missing!r} are not in the "
            f"data (available columns: {df.columns!r})."
        )
    return features


def _apply_pc_rescale(
    df: pl.DataFrame, features: list[str], rescale: str | None,
) -> pl.DataFrame:
    """Apply per-feature minmax / zscore rescale (or pass through on None)."""
    if rescale is None:
        return df
    if rescale == "minmax":
        for c in features:
            col = df[c]
            vmin = col.min()
            vmax = col.max()
            if vmin is None or vmax is None or vmin == vmax:
                continue
            df = df.with_columns(
                ((pl.col(c) - float(vmin)) / (float(vmax) - float(vmin))).alias(c),
            )
        return df
    if rescale == "zscore":
        for c in features:
            col = df[c]
            mu = col.mean()
            sd = col.std()
            if sd is None or sd == 0.0:
                continue
            df = df.with_columns(
                ((pl.col(c) - float(mu)) / float(sd)).alias(c),
            )
        return df
    raise ValueError(
        f"parallel_coordinates(rescale={rescale!r}) — expected "
        "'minmax', 'zscore', or None."
    )


def _parallel_coords_chart_from_dataframe(
    data,
    *,
    features: list[str] | None = None,
    hue: str | None = None,
    rescale: str | None = "minmax",
    alpha: float = 0.5,
    theme: Any = None,
):
    """Parallel coordinates chart from a wide DataFrame.

    Reshapes ``data`` (a polars DataFrame, pandas DataFrame, or 2D
    numpy array) into long form (``sample_id``, ``feature``, ``value``)
    plus an optional ``hue`` column when provided, then renders one
    polyline per sample with ``mark_parallel_coordinates``.

    ``rescale`` in ``{"minmax", "zscore", None}``: per-feature rescaling
    applied before unpivot so all features share a common y axis.
    """
    import ferrum

    df = _coerce_to_polars(data)
    features = _resolve_pc_features(df, features, hue)
    df = _apply_pc_rescale(df, features, rescale)

    # Reshape to long form.
    id_cols = ["sample_id"] + ([hue] if hue is not None else [])
    df = df.with_row_index("sample_id").with_columns(
        pl.col("sample_id").cast(pl.Utf8),
    )
    long = df.unpivot(
        index=id_cols,
        on=features,
        variable_name="feature",
        value_name="value",
    )
    # Preserve feature order so the ordinal x scale lays out features in
    # the user-supplied (or default) sequence rather than alphabetical.
    long = long.with_columns(
        pl.col("feature").cast(pl.Enum(features)),
    ).sort("sample_id", "feature").with_columns(
        pl.col("feature").cast(pl.Utf8),
    )

    if hue is not None:
        # Cast hue to Utf8 so the categorical color scale routes it
        # correctly (same Int64→continuous gotcha as silhouette.cluster).
        long = long.with_columns(pl.col(hue).cast(pl.Utf8))

    chart = ferrum.Chart(long).mark_parallel_coordinates(
        alpha=alpha,
        color_field=hue,
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _decision_boundary_chart_from_source(
    source: Any,
    *,
    features: tuple = (0, 1),
    grid_resolution: int = 200,
    proba: bool = False,
    scatter: bool = False,
    theme: Any = None,
):
    """Decision-boundary heatmap of model predictions over a 2D feature grid.

    Pre-computes a ``grid_resolution × grid_resolution`` grid of
    ``x / x2 / y / y2`` cell bounds and the model's prediction
    (``z`` = class index when ``proba=False``, ``P(class=1)`` when
    ``proba=True``). The grid is fed to ``mark_decision_boundary``
    (rect-based).

    When ``scatter=True`` and ``y`` is available, the training scatter
    is composed as a true overlay layer (not horizontal concat) by
    constructing a single unified DataFrame: grid rows hold
    ``x/x2/y/y2/z`` with the scatter coordinates null, and scatter rows
    hold ``scatter_x/scatter_y/scatter_z`` (= true class label mapped to
    the same numeric domain as ``z``) with the grid columns null. Both
    layers share the same DataFrame identity so ``Chart.__add__``
    produces a true layered chart and both color encodings resolve
    against the same continuous color scale — matching boundary and
    point colors means the cell-color directly says "predicted class /
    probability" while the point's color says "true class", so a
    misclassified point pops against its neighborhood.

    Non-numeric class labels (e.g. string labels) are mapped to a
    zero-based class-index float via the model's ``classes_`` attribute
    when present, otherwise via lexicographic order. The black point
    stroke is uniform; ``size=80`` ensures visibility against any
    background color including same-color cells.
    """
    import ferrum

    feat_idx = _resolve_decision_boundary_features(source, features)
    grid_info = _build_decision_boundary_grid(
        source, feat_idx, grid_resolution, proba=proba,
    )

    if not (scatter and source.y is not None):
        # Pure-boundary path: no overlay, no padding columns, no row mixing.
        grid_df = pl.DataFrame({
            "x": [v - grid_info["dx"] / 2 for v in grid_info["flat_x"]],
            "x2": [v + grid_info["dx"] / 2 for v in grid_info["flat_x"]],
            "y": [v - grid_info["dy"] / 2 for v in grid_info["flat_y"]],
            "y2": [v + grid_info["dy"] / 2 for v in grid_info["flat_y"]],
            "z": [float(v) for v in grid_info["z"]],
        })
        chart = ferrum.Chart(grid_df).mark_decision_boundary(proba=proba)
        if theme is not None:
            chart = chart.theme(theme)
        return chart

    unified = _build_decision_boundary_unified(source, grid_info)
    grid_chart = ferrum.Chart(unified).mark_decision_boundary(proba=proba)
    overlay = ferrum.Chart(unified).mark_point(
        stroke="#000000",
        stroke_width=1.0,
        size=80.0,
    ).encode(x="scatter_x", y="scatter_y", color="scatter_z")
    chart = grid_chart + overlay
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _resolve_decision_boundary_features(
    source: Any, features: tuple,
) -> tuple[int, int]:
    feat_idx = tuple(
        source.feature_names.index(f) if isinstance(f, str) else int(f)
        for f in features
    )
    if len(feat_idx) != 2:
        raise ValueError(
            "decision_boundary_chart requires exactly 2 features; got "
            f"{len(feat_idx)}."
        )
    return feat_idx  # type: ignore[return-value]


def _build_decision_boundary_grid(
    source: Any,
    feat_idx: tuple[int, int],
    grid_resolution: int,
    *,
    proba: bool,
) -> dict:
    """Compute the prediction grid + cell bounds for a 2-feature
    decision-boundary chart. Returns a dict with ``flat_x``, ``flat_y``,
    ``dx``, ``dy``, ``z``, and the two feature vectors ``x_col``, ``y_col``.
    """
    import numpy as np

    X_np = source.X.to_numpy()
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
    if proba and "predict_proba" in source.capabilities:
        z = source.model.predict_proba(grid)[:, 1].astype(np.float64)
    else:
        z = np.asarray(source.model.predict(grid)).astype(np.float64)
    return {
        "flat_x": [float(v) for v in xx.ravel()],
        "flat_y": [float(v) for v in yy.ravel()],
        "dx": dx,
        "dy": dy,
        "z": z,
        "x_col": x_col,
        "y_col": y_col,
    }


def _build_decision_boundary_unified(source: Any, g: dict) -> pl.DataFrame:
    """Build the layered grid+scatter DataFrame for the overlay path.

    Maps ``y`` labels to the same numeric domain as ``z``. For
    ``proba=False`` ``z`` is the predicted class index (integer-cast
    prediction); ``scatter_z`` is the true class index in the same
    encoding so matching colors = correct prediction. For ``proba=True``
    ``z`` is ``P(class=1) ∈ [0, 1]`` and the true label is 0 or 1.
    Prefers the model's ``classes_`` attribute when present; falls back
    to lex sort.

    Grid rows hold ``x/x2/y/y2/z`` (scatter columns null); scatter rows
    hold ``scatter_x/scatter_y/scatter_z`` (grid columns null). mark_rect
    skips null x/x2/y/y2 cells and mark_point skips null
    scatter_x/scatter_y, so each layer renders only its intended rows.
    The shared ``z`` color scale resolves coherently because both ``z``
    and ``scatter_z`` live on the same numeric domain.
    """
    import numpy as np

    y_raw = source.y.to_numpy()
    if hasattr(source.model, "classes_"):
        class_order = list(source.model.classes_)
    else:
        class_order = sorted({v for v in y_raw.tolist()})
    label_to_idx = {c: float(i) for i, c in enumerate(class_order)}
    scatter_z = np.array(
        [label_to_idx.get(v, float("nan")) for v in y_raw.tolist()],
        dtype=np.float64,
    )

    flat_x, flat_y = g["flat_x"], g["flat_y"]
    dx, dy, z = g["dx"], g["dy"], g["z"]
    x_col, y_col = g["x_col"], g["y_col"]
    n_grid = len(flat_x)
    n_scatter = len(x_col)
    nulls_grid = [None] * n_grid
    nulls_scatter = [None] * n_scatter

    return pl.DataFrame({
        "x": [v - dx / 2 for v in flat_x] + nulls_scatter,
        "x2": [v + dx / 2 for v in flat_x] + nulls_scatter,
        "y": [v - dy / 2 for v in flat_y] + nulls_scatter,
        "y2": [v + dy / 2 for v in flat_y] + nulls_scatter,
        "z": [float(v) for v in z] + nulls_scatter,
        "scatter_x": nulls_grid + [float(v) for v in x_col],
        "scatter_y": nulls_grid + [float(v) for v in y_col],
        "scatter_z": nulls_grid + [float(v) for v in scatter_z],
    }, schema={
        "x": pl.Float64, "x2": pl.Float64, "y": pl.Float64, "y2": pl.Float64,
        "z": pl.Float64,
        "scatter_x": pl.Float64, "scatter_y": pl.Float64,
        "scatter_z": pl.Float64,
    })
