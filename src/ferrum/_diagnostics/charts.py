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

from ferrum.encoding import X, Y

from ferrum._sentinels import _inject_constant  # noqa: F401  re-export


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


def _r2_score(y_true: pl.Series, y_pred: pl.Series) -> float:
    """Coefficient of determination — Schwabish SB3 corner-metrics helper."""
    diff = y_true - y_pred
    ss_res = float((diff ** 2).sum())
    mean_y = float(y_true.mean())
    ss_tot = float(((y_true - mean_y) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _inject_metrics_corner(
    df: pl.DataFrame,
    *,
    kind: str = "studentized",
) -> pl.DataFrame:
    """Augment a residuals DataFrame with ``_metrics_text`` and ``_metrics_y``.

    Reads ``y_true`` / ``y_pred`` to compute R² / RMSE / MAE, then writes
    the formatted corner string on the single row with the largest
    ``y_pred`` (other rows null, so ``mark_text`` skips them). ``_metrics_y``
    anchors the text to the top of the residual-axis range so the
    annotation lands in the top-right corner of the plot.

    Schwabish SB3 (2026-05-11). Used by both the single-panel residuals
    chart and the residuals-vs-fitted cell of the 4-panel layout; the
    column is harmless on the other panels (their renderers do not read
    it). ``kind`` selects ``studentized_residual`` vs ``residual`` for
    the y-anchor — must match what ``mark_residuals`` will render so the
    text lands at the correct corner.
    """
    if df.height == 0:
        return df
    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"
    from ferrum._metrics_fmt import format_corner_metrics

    diff = df["y_pred"] - df["y_true"]
    r2 = _r2_score(df["y_true"], df["y_pred"])
    rmse = float((diff ** 2).mean() ** 0.5)
    mae = float(diff.abs().mean())
    corner_text = format_corner_metrics(r2, rmse, mae)

    n = df.height
    anchor_idx = df["y_pred"].arg_max()
    text_col: list[str | None] = [None] * n
    text_col[anchor_idx] = corner_text
    y_col_vals: list[float | None] = [None] * n
    y_col_vals[anchor_idx] = float(df[y_col].max())
    return df.with_columns(
        pl.Series("_metrics_text", text_col, dtype=pl.Utf8),
        pl.Series("_metrics_y", y_col_vals, dtype=pl.Float64),
    )


def _overlay_metrics_corner(chart):
    """Add a same-data ``mark_text`` overlay reading the injected columns.

    Assumes ``chart._data`` already carries ``_metrics_text`` / ``_metrics_y``
    (see :func:`_inject_metrics_corner`). The overlay layer reuses
    ``chart._data`` so ``+`` produces a true layer rather than the
    HConcat fallback.

    Schwabish SB3 (2026-05-11). Returns the original chart unchanged
    when the columns are absent, so callers can invoke unconditionally.
    """
    from ferrum.layer import Layer

    data = chart._data
    if data is None or "_metrics_text" not in data.columns:
        return chart
    return chart.layer(
        Layer(
            mark="text",
            encoding={"x": "y_pred", "y": "_metrics_y", "text": "_metrics_text"},
            mark_kwargs={"align": "right", "dx": -4, "dy": 4},
        )
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
    annotate_metrics: bool = True,
    subtitle: str | None = None,
    theme: Any = None,
):
    """Build a residuals diagnostic chart from a ModelSource.

    Schwabish SB3 (2026-05-11): the corner R²/RMSE/MAE annotation rides
    on the residuals-vs-fitted view in both single-panel and 4-panel
    layouts. ``_inject_metrics_corner`` augments the shared DataFrame
    once at the top of this function; the single-panel chart and the
    ``residuals_vs_fitted`` cell of the 4-panel grid then call
    ``_overlay_metrics_corner`` which detects the injected columns and
    adds a same-data ``mark_text`` layer. The other 4-panel cells (QQ,
    scale-location, residuals-vs-leverage) inherit the extra columns but
    do not reference them, so their renderers are unaffected.
    """
    import ferrum

    df = source.predictions()
    if annotate_metrics:
        df = _inject_metrics_corner(df, kind=kind)

    if panels in (None, "single"):
        if cook_threshold is not None:
            df = _inject_cook_outliers(df, kind=kind, threshold=cook_threshold)
        chart = ferrum.Chart(df).mark_residuals(
            kind=kind,
            cook_threshold=cook_threshold,
        )
        chart = chart.properties(
            title=ferrum.Title("Residuals", subtitle=subtitle),
        )
        # ``_overlay_metrics_corner`` is a no-op when the injected
        # columns are absent, so the call is unconditional.
        chart = _overlay_metrics_corner(chart)
        if theme is not None:
            chart = chart.theme(theme)
        return chart

    # Multi-panel path: each panel injects its own outlier columns with
    # the right x-axis encoding for that panel's coordinate system. The
    # leverage panel depends on hat-matrix quantities; for non-linear
    # estimators ModelSource.predictions emits NaN for every leverage row,
    # in which case we silently drop the panel rather than crashing on
    # "no usable values for field 'leverage'".
    panel_list = panels if isinstance(panels, list) else ["residuals_vs_fitted"]
    if "residuals_vs_leverage" in panel_list and df["leverage"].is_nan().all():
        panel_list = [p for p in panel_list if p != "residuals_vs_leverage"]
    charts = [
        _residuals_panel(df, name, kind=kind, cook_threshold=cook_threshold) for name in panel_list
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
            chart = ferrum.Chart(df).mark_residuals(
                kind=kind,
                cook_threshold=cook_threshold,
            )
        else:
            chart = ferrum.Chart(df).mark_residuals(kind=kind)
        chart = chart.encode(
            x=X("y_pred", title="Fitted values"),
            y=Y(y_col, title="Studentized residual"),
        )
        chart = chart.properties(title=ferrum.Title("Residuals vs Fitted"))
        # Schwabish SB3: detect the ``_metrics_text``/``_metrics_y``
        # columns injected upstream by ``_inject_metrics_corner`` and
        # overlay the corner annotation. No-op when the columns are
        # absent, so an externally-built panel (e.g. a user invoking
        # ``_residuals_panel`` directly without metrics injection)
        # renders identically to the pre-SB3 behavior.
        return _overlay_metrics_corner(chart)

    if name == "qq":
        return (
            ferrum.Chart(df)
            .mark_qq()
            .encode(
                x=X("studentized_residual", title="Theoretical quantiles"),
                y=Y("sample", title="Sample quantiles"),
            )
            .properties(title=ferrum.Title("Normal Q–Q"))
        )

    if name == "scale_location":
        d2 = df.with_columns(pl.col("studentized_residual").abs().sqrt().alias("sqrt_abs_resid"))
        return (
            ferrum.Chart(d2)
            .mark_point()
            .encode(
                x=X("y_pred", title="Fitted values"),
                y=Y("sqrt_abs_resid", title="√|Studentized residual|"),
            )
            .properties(title=ferrum.Title("Scale–Location"))
        )

    if name == "residuals_vs_leverage":
        # Inject outlier columns BEFORE constructing the base chart so
        # base and overlay layers share the same DataFrame identity
        # (avoids unnecessary null-pad merge in Chart.__add__).
        # x_col="leverage" so the overlay sits at the right
        # x coordinate for this panel.
        if cook_threshold is not None:
            df = _inject_cook_outliers(
                df,
                kind=kind,
                threshold=cook_threshold,
                x_col="leverage",
            )
        base = (
            ferrum.Chart(df)
            .mark_point()
            .encode(
                x=X("leverage", title="Leverage"),
                y=Y(y_col, title="Studentized residual"),
            )
            .properties(title=ferrum.Title("Residuals vs Leverage"))
        )
        if cook_threshold is not None and "_cook_outlier_x" in df.columns:
            overlay = (
                ferrum.Chart(df)
                .mark_point(
                    fill="#e15759",  # tableau red, matches mark_residuals overlay
                    stroke="#000000",
                    stroke_width=1.0,
                    size=80.0,
                )
                .encode(x="_cook_outlier_x", y="_cook_outlier_y")
            )
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
        residuals = df["y_pred"] - df["y_true"]
        if ci is not None:
            if not 0.0 < ci < 1.0:
                raise ValueError(f"mark_prediction_error(ci={ci!r}) — expected a value in (0, 1).")
            alpha = (1.0 - float(ci)) / 2.0
            q_lo = float(residuals.quantile(alpha, interpolation="linear"))
            q_hi = float(residuals.quantile(1.0 - alpha, interpolation="linear"))
        else:
            sigma = float((residuals**2).mean() ** 0.5)
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
    chart = chart.encode(
        x=X("y_pred", title="Predicted value"),
        y=Y("y_true", title="True value"),
    )
    chart = chart.properties(title=ferrum.Title("Prediction Error"))
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
    annotate_auc: bool = True,
    subtitle: str | None = None,
    theme: Any = None,
):
    """Build an ROC chart from a ModelSource.

    Schwabish SB3 (2026-05-11): emits an active title carrying the AUC
    value when the chart has exactly one curve, and overlays
    :class:`ferrum.AUCLabel` text per curve via the explicit-field helper
    (so the figure builder does not need a chart-level color encoding
    that would leak into mark_roc's diagonal reference layer).
    """
    import numpy as np

    import ferrum
    from ferrum.annotations import _apply_metric_label_explicit, _trapezoid_auc

    df = source.roc_curve(average=None if per_class else average)
    color_field = _color_field_for(df, "class")

    if annotate_auc and color_field is not None:
        groups = df[color_field].unique().to_list()
        rename_map = {}
        for g in groups:
            subset = df.filter(pl.col(color_field) == g)
            fpr_g = np.asarray(subset["fpr"].to_list(), dtype=float)
            tpr_g = np.asarray(subset["tpr"].to_list(), dtype=float)
            auc_val = _trapezoid_auc(fpr_g, tpr_g)
            rename_map[str(g)] = f"{g} (AUC = {auc_val:.3f})"
        df = df.with_columns(
            pl.col(color_field).cast(pl.Utf8).replace(rename_map).alias(color_field)
        )

    chart = ferrum.Chart(df).mark_roc(
        average=None if per_class else average,
        annotate_auc=False,
        color_field=color_field,
    )
    chart = chart.encode(
        x=X("fpr", title="False Positive Rate"),
        y=Y("tpr", title="True Positive Rate"),
    )

    n_curves = len(set(df[color_field].to_list())) if color_field is not None else 1
    if n_curves == 1:
        fpr = np.asarray(df["fpr"].to_list(), dtype=float)
        tpr = np.asarray(df["tpr"].to_list(), dtype=float)
        auc_value = _trapezoid_auc(fpr, tpr)
        chart = chart.properties(
            title=ferrum.Title(f"ROC Curve — AUC {auc_value:.3f}", subtitle=subtitle),
        )
    else:
        chart = chart.properties(title=ferrum.Title("ROC Curve", subtitle=subtitle))

    if annotate_auc:
        chart = _apply_metric_label_explicit(
            chart,
            "auc",
            x_col="fpr",
            y_col="tpr",
            color_col=color_field,
            position="end",
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
    annotate_ap: bool = True,
    iso_lines: bool = False,
    subtitle: str | None = None,
    theme: Any = None,
):
    """Build a precision-recall chart from a ModelSource.

    Schwabish SB3 (2026-05-11): active title with AP for single curve,
    positive-class-prevalence baseline hline for binary classifiers,
    and per-curve :class:`ferrum.APLabel` overlay via the explicit-field
    helper.
    """
    import numpy as np

    import ferrum
    from ferrum.annotations import _apply_metric_label_explicit, _ap_step

    df = source.pr_curve(average=None if per_class else average)
    if iso_lines:
        df = _inject_pr_iso_lines(df)

    # Rename color-field values to include per-class AP before legend is built.
    _pr_color_field = _color_field_for(df, "class")
    if annotate_ap and _pr_color_field is not None:
        _curve_df_pre = df.filter(pl.col("precision").is_not_null())
        _pr_groups = _curve_df_pre[_pr_color_field].unique().to_list()
        _pr_rename_map = {}
        for _g in _pr_groups:
            _subset = _curve_df_pre.filter(pl.col(_pr_color_field) == _g)
            _recall_g = np.asarray(_subset["recall"].to_list(), dtype=float)
            _prec_g = np.asarray(_subset["precision"].to_list(), dtype=float)
            _ap_val = _ap_step(_recall_g, _prec_g)
            _pr_rename_map[str(_g)] = f"{_g} (AP = {_ap_val:.3f})"
        df = df.with_columns(
            pl.col(_pr_color_field).cast(pl.Utf8).replace(_pr_rename_map).alias(_pr_color_field)
        )

    # Inject the positive-class-prevalence baseline as a same-data column
    # (one non-null entry, rest null) so the mark_rule overlay shares
    # ``chart._data`` and renders as a true layer rather than HConcat
    # fallback. Binary classifiers only — multi-class baselines would
    # need one rule per class which is information-dense without payoff.
    baseline_prevalence: float | None = None
    y_series = getattr(source, "y", None)
    if y_series is None:
        y_series = getattr(source, "_y", None)
    if y_series is not None:
        unique_y = y_series.unique().sort()
        if len(unique_y) == 2:
            positive = unique_y[1]
            p = float((y_series == positive).mean())
            if 0.0 < p < 1.0:
                baseline_prevalence = p
                n = df.height
                df = df.with_columns(
                    pl.Series(
                        "_baseline_y",
                        [p] + [None] * (n - 1),
                        dtype=pl.Float64,
                    ),
                )

    color_field = _color_field_for(df, "class")
    chart = (
        ferrum.Chart(df)
        .mark_pr(
            annotate_ap=False,
            iso_lines=iso_lines,
            color_field=color_field,
        )
        .encode(
            x=X("recall", title="Recall", scale={"type": "linear", "domain": [0.0, 1.05]}),
            y=Y("precision", title="Precision", scale={"type": "linear", "domain": [0.0, 1.05]}),
        )
    )

    # Active title on single-curve PR charts. Drop iso-curve sentinel rows
    # (precision==null) before computing AP.
    curve_df = df.filter(pl.col("precision").is_not_null())
    if color_field is not None and color_field in curve_df.columns:
        n_curves = len(set(curve_df[color_field].to_list()))
    else:
        n_curves = 1
    if n_curves == 1:
        recall = np.asarray(curve_df["recall"].to_list(), dtype=float)
        precision = np.asarray(curve_df["precision"].to_list(), dtype=float)
        ap_value = _ap_step(recall, precision)
        chart = chart.properties(
            title=ferrum.Title(
                f"Precision–Recall Curve — AP {ap_value:.3f}",
                subtitle=subtitle,
            ),
        )
    else:
        chart = chart.properties(
            title=ferrum.Title("Precision–Recall Curve", subtitle=subtitle),
        )

    if baseline_prevalence is not None:
        from ferrum.layer import Layer

        chart = chart.layer(
            Layer(
                mark="rule",
                encoding={"y": "_baseline_y"},
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
            )
        )

    if annotate_ap:
        chart = _apply_metric_label_explicit(
            chart,
            "ap",
            x_col="recall",
            y_col="precision",
            color_col=color_field,
            position="end",
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
            iso_rows.append(
                {
                    "_iso_recall": float(r),
                    "_iso_precision": float(p),
                    "_iso_f": f"F={f:.1f}",
                    "_iso_label_x": None,
                    "_iso_label_y": None,
                    "_iso_label": None,
                }
            )
        # Label anchor at the rightmost (recall ≈ 1) point of the iso curve.
        if iso_rows:
            iso_rows[-1]["_iso_label_x"] = float(recalls[valid][-1]) - 0.02
            iso_rows[-1]["_iso_label_y"] = float(precisions[valid][-1]) + 0.02
            iso_rows[-1]["_iso_label"] = f"F={f:.1f}"

    iso_df = pl.DataFrame(iso_rows)
    # Extend the original PR DataFrame with null iso columns.
    null_iso = pl.DataFrame(
        {
            "_iso_recall": [None] * df.height,
            "_iso_precision": [None] * df.height,
            "_iso_f": [None] * df.height,
            "_iso_label_x": [None] * df.height,
            "_iso_label_y": [None] * df.height,
            "_iso_label": [None] * df.height,
        },
        schema={
            "_iso_recall": pl.Float64,
            "_iso_precision": pl.Float64,
            "_iso_f": pl.Utf8,
            "_iso_label_x": pl.Float64,
            "_iso_label_y": pl.Float64,
            "_iso_label": pl.Utf8,
        },
    )
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
    annotate_brier: bool = True,
    subtitle: str | None = None,
    theme: Any = None,
):
    """Build a calibration (reliability) chart from a ModelSource.

    Schwabish SB3 (2026-05-11): active title carrying the per-sample
    Brier score when the chart shows a single model; per-series
    :class:`ferrum.BrierLabel` overlay when ``annotate_brier=True``.
    """
    import numpy as np

    import ferrum
    from ferrum.annotations import _apply_metric_label_explicit, _brier_score

    df = source.calibration_curve(n_bins=n_bins, strategy=strategy)
    color = "model" if "model" in df.columns else None
    chart = ferrum.Chart(df).mark_calibration(
        n_bins=n_bins,
        strategy=strategy,
        color_field=color,
    )
    chart = chart.encode(
        x=X("mean_predicted", title="Mean predicted probability"),
        y=Y("fraction_positive", title="Fraction of positives"),
    )

    # Active title fires when exactly one model. Compute the true
    # per-sample Brier from ``source.model.predict_proba`` and
    # ``source.y``; the compared-model case skips the active title.
    is_single_model = not hasattr(source, "models")
    brier_value: float | None = None
    if is_single_model:
        try:
            model_attr = getattr(source, "model", None) or getattr(source, "_model", None)
            X_attr = getattr(source, "X", None)
            if X_attr is None:
                X_attr = getattr(source, "_X", None)
            y_attr = getattr(source, "y", None)
            if y_attr is None:
                y_attr = getattr(source, "_y", None)
            if model_attr is not None and X_attr is not None and y_attr is not None:
                proba = model_attr.predict_proba(X_attr)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    p = np.asarray(proba[:, 1], dtype=float)
                    obs = np.asarray(y_attr, dtype=float)
                    brier_value = _brier_score(p, obs)
        except Exception:
            brier_value = None

    if brier_value is not None:
        chart = chart.properties(
            title=ferrum.Title(
                f"Calibration Curve — Brier {brier_value:.3f}",
                subtitle=subtitle,
            ),
        )
    else:
        chart = chart.properties(
            title=ferrum.Title("Calibration Curve", subtitle=subtitle),
        )

    if annotate_brier:
        chart = _apply_metric_label_explicit(
            chart,
            "brier",
            x_col="mean_predicted",
            y_col="fraction_positive",
            color_col=color,
            position="corner",
        )

    from ferrum.layer import Layer

    point_enc: dict = {"x": "mean_predicted", "y": "fraction_positive"}
    if color is not None:
        point_enc["color"] = color
    chart = chart.layer(
        Layer(mark="point", encoding=point_enc, mark_kwargs={"size": 40, "filled": True})
    )

    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _gain_chart_from_source(
    source: Any,
    *,
    subtitle: str | None = None,
    theme: Any = None,
):
    """Build a cumulative-gain chart from a ModelSource.

    Replaces the categorical color legend with endpoint-anchored
    direct labels via ``_direct_label_endpoint`` and adds a
    ``Cumulative gain`` active title — unconditional, matching the
    learning_curve / validation_curve / lift sibling figures
    (Schwabish C8 audit-rework, 2026-05-12). Chart-API users who
    want a categorical legend can call ``Chart(df).mark_gain(...)``
    directly without going through this builder.
    """
    import ferrum
    from ferrum._direct_label import _direct_label_endpoint

    df = source.cumulative_gain()
    color_field = _color_field_for(df, "class")
    if color_field is not None:
        df = df.with_columns(
            pl.col(color_field).cast(pl.Utf8).replace({
                "0": "Class 0", "1": "Class 1", "baseline": "Baseline",
            }).alias(color_field)
        )
    chart = ferrum.Chart(df)
    if color_field is not None:
        chart = chart.encode(color=ferrum.Color(color_field, legend=None))
    chart = chart.mark_gain(color_field=color_field)
    chart = chart.encode(
        x=X("percent_population", title="Percentage of sample"),
        y=Y("gain", title="Gain"),
    )
    chart = chart.properties(
        title=ferrum.Title("Cumulative Gains Curve", subtitle=subtitle),
    )
    if color_field is not None:
        chart = _direct_label_endpoint(
            chart,
            label_field=color_field,
            x_col="percent_population",
            y_col="gain",
        )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _lift_chart_from_source(
    source: Any,
    *,
    subtitle: str | None = None,
    theme: Any = None,
):
    """Build a lift chart from a ModelSource.

    Direct labels + legend suppression + ``Lift`` active title —
    unconditional, matching ``_gain_chart_from_source`` (Schwabish C8
    audit-rework, 2026-05-12).
    """
    import ferrum
    from ferrum._direct_label import _direct_label_endpoint

    df = source.lift_curve()
    color_field = _color_field_for(df, "class")
    if color_field is not None:
        df = df.with_columns(
            pl.col(color_field).cast(pl.Utf8).replace({
                "0": "Class 0", "1": "Class 1", "baseline": "Baseline",
            }).alias(color_field)
        )
    chart = ferrum.Chart(df)
    if color_field is not None:
        chart = chart.encode(color=ferrum.Color(color_field, legend=None))
    chart = chart.mark_lift(color_field=color_field)
    chart = chart.encode(
        x=X("percent_population", title="Percentage of sample"),
        y=Y("lift", title="Lift"),
    )
    chart = chart.properties(title=ferrum.Title("Lift Curve", subtitle=subtitle))
    if color_field is not None:
        chart = _direct_label_endpoint(
            chart,
            label_field=color_field,
            x_col="percent_population",
            y_col="lift",
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
        normalize=normalize,
        annotate=annotate,
    )
    chart = chart.encode(
        x=X("predicted", title="Predicted label"),
        y=Y("actual", title="True label"),
    )
    chart = chart.properties(title=ferrum.Title("Confusion Matrix"))
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _class_prediction_error_chart_from_source(
    source: Any,
    *,
    normalize: bool = False,
    show_counts: bool = True,
    theme: Any = None,
):
    """Build a stacked-bar class-prediction-error chart from a ModelSource.

    Reuses the unnormalized confusion-matrix output as the underlying
    long-form data.

    Schwabish SB-followup (2026-05-12): ``show_counts=True`` (default)
    routes through ``Chart.mark_class_prediction_error(show_counts=...)``
    which owns both the ``_count_text`` sentinel column (via its
    ``data_transform``) and the text layer emitted by
    ``desugar_class_prediction_error``. The text layer carries the
    same ``Stack`` position adjustment as the bar layer; the renderer
    maps text-mark y values to segment MIDPOINTS so labels land
    centred without any Python-side cumsum duplication. Chart-API
    users get the same behaviour via
    ``Chart(df).mark_class_prediction_error(show_counts=True)``.
    """
    import ferrum

    df = source.confusion_matrix(normalize=None)
    chart = ferrum.Chart(df).mark_class_prediction_error(
        normalize=normalize,
        show_counts=show_counts,
    )
    chart = chart.encode(
        x=X("actual", title="Actual class"),
        y=Y("value", title="Number of predictions"),
    )
    chart = chart.properties(title=ferrum.Title("Class Prediction Error"))
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

    y_true = source.y
    y_pred = source.model.predict(source.X)
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    rows: list[dict] = []
    for cls_label, metrics in report.items():
        if cls_label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(metrics, dict):
            continue
        for m_name in ("precision", "recall", "f1-score"):
            val = float(metrics[m_name])
            rows.append(
                {
                    "class": str(cls_label),
                    "metric": m_name,
                    "value": val,
                    "value_fmt": f"{val:.2f}",
                }
            )
    df = pl.DataFrame(rows)

    from ferrum.layer import Layer

    chart = (
        ferrum.Chart(df)
        .mark_rect()
        .encode(
            x="metric",
            y="class",
            color="value",
        )
        .layer(
            Layer(
                mark="text",
                encoding={"x": "metric", "y": "class", "text": "value_fmt"},
            )
        )
    )
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
    show_values: bool = True,
    subtitle: str | None = None,
    random_state: int | None = None,
    theme: Any = None,
):
    """Build a feature-importance chart from a ModelSource.

    Schwabish SB3 (2026-05-11): ``Feature importance`` title and, when
    ``show_values=True``, overlays the formatted importance value at the
    end of each bar via a same-data text layer.
    """
    import ferrum

    df = source.importances(method=method, random_state=random_state)
    if top_k is not None:
        df = df.head(top_k)
    df = df.with_columns(
        [
            (pl.col("importance") - pl.col("std")).alias("imp_lower"),
            (pl.col("importance") + pl.col("std")).alias("imp_upper"),
        ]
    )

    if show_values:
        df = df.with_columns(
            pl.col("importance")
            .map_elements(lambda v: f"{v:.3f}", return_dtype=pl.Utf8)
            .alias("_value_text"),
        )

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
    chart = chart.properties(
        title=ferrum.Title("Feature importance", subtitle=subtitle),
    )

    if show_values:
        from ferrum.layer import Layer

        if orient == "horizontal":
            text_ly = Layer(
                mark="text",
                encoding={"x": "importance", "y": "feature", "text": "_value_text"},
                mark_kwargs={"align": "left", "dx": 4},
            )
        else:
            text_ly = Layer(
                mark="text",
                encoding={"x": "feature", "y": "importance", "text": "_value_text"},
                mark_kwargs={"baseline": "bottom", "dy": -4},
            )
        chart = chart.layer(text_ly)

    if theme is not None:
        chart = chart.theme(theme)
    return chart


# ---------------------------------------------------------------------------
# 10d builders — SHAP family
# ---------------------------------------------------------------------------


_SHAP_ORDER_VALUES: frozenset[str] = frozenset({"abs_mean", "max"})


def _shap_select_class(sv: pl.DataFrame, *, per_class: bool) -> pl.DataFrame:
    """Either return all rows (``per_class=True``) or filter to the first
    ``class_label`` group (``per_class=False``). On regression and binary
    classifiers the single-group filter is a no-op."""
    if per_class:
        return sv
    classes = sv["class_label"].unique(maintain_order=True)
    if classes.len() <= 1:
        return sv
    first_class = classes[0]
    return sv.filter(pl.col("class_label") == first_class)


def _shap_order_features(
    sv: pl.DataFrame,
    *,
    order: str,
    max_display: int,
) -> list[str]:
    """Return the top-`max_display` feature names ordered by `order`."""
    if order not in _SHAP_ORDER_VALUES:
        raise ValueError(f"Unknown order {order!r}. Accepted values: {sorted(_SHAP_ORDER_VALUES)}")
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
    per_class: bool = False,
    zero_line: bool = True,
    theme: Any = None,
):
    """Beeswarm chart: per-sample shap values colored by feature value.

    ``per_class=True`` on a multi-class classifier returns a faceted chart
    with one beeswarm panel per class; ``per_class=False`` (default)
    renders a single panel using the first class_label group (the only
    group on regression and binary classifiers).

    Schwabish SB-followup (2026-05-12): ``zero_line=True`` (default)
    routes through ``Chart.mark_shap_beeswarm(zero_line=...)`` which
    owns both the sentinel ``_ref_zero`` column injection (via its
    ``data_transform``) and the dashed-rule layer (emitted by
    ``desugar_shap_beeswarm``). Chart-API callers (
    ``Chart(df).mark_shap_beeswarm(zero_line=True)``) get the same
    behaviour. Disabled automatically on the faceted ``per_class``
    multi-panel path — each facet would need its own non-null sentinel
    row.
    """
    import ferrum

    sv = source.shap_values(background=background)
    sv = _shap_select_class(sv, per_class=per_class)
    keep = _shap_order_features(sv, order=order, max_display=max_display)
    plot_df = sv.filter(pl.col("feature").is_in(keep))
    # Sort rows so features appear in descending mean-|SHAP| order (keep[0]
    # is most important). Rust's distinct_values_in_order uses encounter order
    # to build the ordinal-y domain, so row order drives axis ordering.
    plot_df = plot_df.with_columns(
        pl.col("feature").cast(pl.Enum(keep)).alias("_feature_order")
    ).sort("_feature_order").drop("_feature_order")

    is_faceted = per_class and plot_df["class_label"].n_unique() > 1
    x_min = float(plot_df["shap_value"].min())
    x_max = float(plot_df["shap_value"].max())
    pad = max(abs(x_min), abs(x_max)) * 0.05 if (x_min < x_max) else 1.0
    domain = (x_min - pad, x_max + pad)

    chart = ferrum.Chart(plot_df).mark_shap_beeswarm(
        max_display=max_display,
        order=order,
        zero_line=zero_line and not is_faceted,
        x_scale_domain=domain,
    )
    chart = chart.encode(
        x=X("shap_value", title="SHAP value"),
        y=Y("feature", title="Feature"),
    )
    chart = chart.properties(title=ferrum.Title("SHAP Summary"))
    if is_faceted:
        chart = chart.facet(col="class_label")
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _shap_bar_chart_from_source(
    source: Any,
    *,
    max_display: int = 20,
    order: str = "abs_mean",
    background: Any = None,
    per_class: bool = False,
    theme: Any = None,
):
    """SHAP aggregated bar chart: mean(|shap_value|) per feature.

    ``order`` selects the aggregation used both for sorting and as the
    bar's x-value. ``"abs_mean"`` aggregates by ``mean(|shap_value|)``
    (the conventional summary); ``"max"`` aggregates by
    ``max(|shap_value|)`` to emphasize features with high-impact outliers.
    The output column is named ``abs_mean_shap`` regardless of the
    aggregation so the downstream mark's data contract stays stable.

    ``per_class=True`` on a multi-class classifier facets the chart by
    ``class_label`` (one bar panel per class). ``per_class=False``
    (default) renders a single panel using the first class_label group
    (the only group on regression and binary classifiers).
    """
    if order not in _SHAP_ORDER_VALUES:
        raise ValueError(f"Unknown order {order!r}. Accepted values: {sorted(_SHAP_ORDER_VALUES)}")
    import ferrum

    expr = pl.col("shap_value").abs()
    agg_expr = expr.mean() if order == "abs_mean" else expr.max()
    sv = source.shap_values(background=background)
    sv = _shap_select_class(sv, per_class=per_class)
    group_keys = ["class_label", "feature"] if per_class else ["feature"]
    agg = (
        sv.group_by(group_keys)
        .agg(agg_expr.alias("abs_mean_shap"))
        .sort("abs_mean_shap", descending=True)
        .head(max_display * sv["class_label"].n_unique() if per_class else max_display)
    )
    x_max = float(agg["abs_mean_shap"].max())
    domain = (0.0, x_max * 1.05 if x_max > 0 else 1.0)
    chart = ferrum.Chart(agg).mark_shap_bar(
        max_display=max_display,
        x_scale_domain=domain,
    )
    if per_class and agg["class_label"].n_unique() > 1:
        chart = chart.facet(col="class_label")
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
    per_class: bool = False,
    theme: Any = None,
):
    """Waterfall chart for a single sample's SHAP contributions.

    Feature display order follows the global aggregation chosen by
    ``order``: ``"abs_mean"`` ranks features by ``mean(|shap_value|)``
    across the full dataset (matching the beeswarm and bar paths);
    ``"max"`` ranks by ``max(|shap_value|)``. The top ``max_display``
    features are kept and rendered in descending rank order; the
    waterfall's cumulative sum follows that same order.

    ``per_class=True`` on a multi-class classifier facets the chart by
    ``class_label`` (one waterfall panel per class for the same sample).
    ``per_class=False`` (default) renders a single panel using the first
    class_label group.
    """
    import ferrum

    sv = source.shap_values(background=background)
    sv = _shap_select_class(sv, per_class=per_class)
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
    cumsum = ordered["shap_value"].cum_sum()
    x0 = pl.concat([pl.Series("x0", [0.0]), cumsum.head(cumsum.len() - 1).alias("x0")])
    x1 = cumsum.alias("x1")
    plot_df = ordered.with_columns(
        [
            x0,
            x1,
            pl.when(pl.col("shap_value") >= 0)
            .then(pl.lit("positive"))
            .otherwise(pl.lit("negative"))
            .alias("shap_sign"),
        ]
    )

    x_lo = float(min(x0.min(), x1.min(), 0.0))
    x_hi = float(max(x0.max(), x1.max(), 0.0))
    pad = max(abs(x_lo), abs(x_hi)) * 0.05 if (x_lo < x_hi) else 1.0
    domain = (x_lo - pad, x_hi + pad)

    chart = ferrum.Chart(plot_df).mark_shap_waterfall(
        sample_idx=sample_idx,
        max_display=max_display,
        x_scale_domain=domain,
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
    anchor = df.group_by(["feature", "sample_id"], maintain_order=True).agg(
        pl.col("pd_value").first().alias("_pd_anchor")
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
        features,
        grid_resolution=grid_resolution,
        kind=kind,
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
            kind=kind,
            ice_alpha=ice_alpha,
            center=center,
            color_field=None,  # one feature per facet — color is redundant
        )
        .encode(
            x=X("feature_value", title="Feature value"),
            y=Y("pd_value", title="Partial dependence"),
        )
        .facet(col="feature")
    )
    chart = chart.properties(title=ferrum.Title("Partial Dependence"))
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
    optimum_label: bool = True,
    subtitle: str | None = None,
    theme: Any = None,
):
    """Build a discrimination-threshold chart from a ModelSource.

    The underlying DataFrame is unpivoted to long form
    ``(threshold, metric, value)`` for plotting.

    Schwabish (post-C7 audit-rework, 2026-05-12): adds a
    ``Discrimination threshold`` active title and threads
    ``threshold_line`` + ``optimum_label`` through to
    ``Chart.mark_discrimination_threshold``. The mark owns the
    sentinel-column injection (via ``data_transform``) and the
    text layer (via ``desugar_discrimination_threshold``), so
    chart-API users get the same defaults via
    ``Chart(df).mark_discrimination_threshold(optimum_label=True)``.
    """
    import ferrum

    df = source.discrimination_threshold(n_thresholds=n_thresholds, cv=cv)
    long_df = df.unpivot(
        index="threshold",
        on=list(metrics),
        variable_name="metric",
        value_name="value",
    )
    # Rename internal metric names to human-readable legend labels.
    _metric_labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "queue_rate": "Queue rate",
    }
    long_df = long_df.with_columns(
        pl.col("metric").replace(_metric_labels).alias("metric")
    )
    chart = ferrum.Chart(long_df).mark_discrimination_threshold(
        metrics=metrics,
        n_thresholds=n_thresholds,
        threshold_line=threshold_line,
        optimum_label=optimum_label,
    )
    chart = chart.encode(
        x=X("threshold", title="Discrimination threshold"),
        y=Y("value", title="Score"),
    )
    chart = chart.properties(
        title=ferrum.Title("Discrimination Threshold", subtitle=subtitle),
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
    subtitle: str | None = None,
    theme: Any = None,
):
    """Learning-curve chart: dedupe per (train_size, split), then ribbon+line.

    Schwabish SB3 (2026-05-11): replaces the categorical legend with
    endpoint-anchored direct labels (``train`` / ``test``), suppresses
    the redundant color legend via ``Color(legend=None)``, and adds a
    ``Learning curve`` active title.
    """
    import ferrum
    from ferrum._direct_label import _direct_label_endpoint

    df = source.learning_curve(cv=cv, scoring=scoring, train_sizes=train_sizes)
    df = _dedupe_aggregated(df, "train_size", "split")
    df = df.with_columns(
        pl.col("split").replace({"train": "Training Score", "test": "Cross-Validation Score"})
    )
    chart = (
        ferrum.Chart(df)
        .encode(color=ferrum.Color("split", legend=None))
        .mark_learning_curve(ci_style=ci_style)
    )
    chart = chart.encode(
        x=X("train_size", title="Training instances"),
        y=Y("mean_score", title="Score"),
    )
    chart = chart.properties(
        title=ferrum.Title("Learning Curve", subtitle=subtitle),
    )
    chart = _direct_label_endpoint(
        chart,
        label_field="split",
        x_col="train_size",
        y_col="mean_score",
    )
    from ferrum.layer import Layer

    chart = chart.layer(
        Layer(
            mark="point",
            encoding={"x": "train_size", "y": "mean_score", "color": "split"},
            mark_kwargs={"size": 40, "filled": True},
        )
    )
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
    subtitle: str | None = None,
    theme: Any = None,
):
    """Validation-curve chart: dedupe per (param_value, split), then ribbon+line.

    Schwabish SB3 (2026-05-11): direct labels + legend suppression +
    parameterized active title (``"Validation curve — <param>"``).
    """
    import ferrum
    from ferrum._direct_label import _direct_label_endpoint

    df = source.validation_curve(param, values, cv=cv, scoring=scoring)
    df = _dedupe_aggregated(df, "param_value", "split")
    df = df.with_columns(
        pl.col("split").replace({"train": "Training Score", "test": "Cross-Validation Score"})
    )
    vals = [float(v) for v in values]
    if log_scale == "auto":
        non_zero = [v for v in vals if v > 0]
        if len(non_zero) >= 2 and max(non_zero) / min(non_zero) > 100:
            is_log = True
        else:
            is_log = False
    else:
        is_log = bool(log_scale)
    chart = (
        ferrum.Chart(df)
        .encode(color=ferrum.Color("split", legend=None))
        .mark_validation_curve(
            log_scale=is_log,
            ci_style=ci_style,
            param_label=param,
        )
    )
    chart = chart.encode(
        y=Y("mean_score", title="Score"),
    )
    chart = chart.properties(
        title=ferrum.Title(f"Validation Curve — {param}", subtitle=subtitle),
    )
    chart = _direct_label_endpoint(
        chart,
        label_field="split",
        x_col="param_value",
        y_col="mean_score",
    )
    from ferrum.layer import Layer

    chart = chart.layer(
        Layer(
            mark="point",
            encoding={"x": "param_value", "y": "mean_score", "color": "split"},
            mark_kwargs={"size": 40, "filled": True},
        )
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
    """Per-fold CV-score chart.

    ``kind="bar"`` shows one bar per fold with a dashed mean-score reference
    line per split; ``"box"``/``"strip"`` leave raw per-fold rows for the
    mark layer.
    """
    import ferrum

    df = source.cv_scores(cv=cv, scoring=scoring)
    if split != "both":
        df = df.filter(pl.col("split") == split)
    if kind == "bar":
        # Broadcast per-split mean so the rule layer draws one line per split.
        df = df.with_columns(pl.col("score").mean().over("split").alias("_mean_score"))
    chart = ferrum.Chart(df).mark_cv_scores(kind=kind, split=split)
    chart = chart.encode(y=Y("score", title="Score"))
    chart = chart.properties(title=ferrum.Title("Cross-Validation Scores"))
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
        log_scale=log_scale,
        highlight_best=highlight_best,
    )
    chart = chart.encode(y=Y("mean_score", title="Score"))
    chart = chart.properties(title=ferrum.Title("Alpha Selection"))
    from ferrum.layer import Layer

    chart = chart.layer(
        Layer(
            mark="point",
            encoding={"x": "alpha", "y": "mean_score"},
            mark_kwargs={"size": 40, "filled": True},
        )
    )
    # Schwabish C11: annotate the best-alpha point with a text label.
    # The desugar already draws a vertical rule via the _best_alpha sentinel
    # (injected by the data_transform at render time); here we inject
    # companion sentinel columns for a text layer that reads the same
    # _best_alpha x position.
    if highlight_best and "mean_score" in df.columns:
        best_idx = int(df["mean_score"].arg_max())
        best_alpha = float(df["alpha"][best_idx])
        best_score = float(df["mean_score"][best_idx])
        label_text = f"α = {best_alpha:.3f}"
        n = df.height
        df = df.with_columns(
            pl.Series("_best_alpha_y", [best_score] + [None] * (n - 1), dtype=pl.Float64),
            pl.Series("_best_alpha_text", [label_text] + [None] * (n - 1), dtype=pl.Utf8),
        )
        # Rebuild the chart with the augmented DataFrame so that the
        # text layer shares the same data identity.
        chart = ferrum.Chart(df).mark_alpha_selection(
            log_scale=log_scale,
            highlight_best=highlight_best,
        )
        chart = chart.encode(y=Y("mean_score", title="Score"))
        chart = chart.properties(title=ferrum.Title("Alpha Selection"))
        chart = chart.layer(
            Layer(
                mark="point",
                encoding={"x": "alpha", "y": "mean_score"},
                mark_kwargs={"size": 40, "filled": True},
            ),
            Layer(
                mark="text",
                encoding={"x": "_best_alpha", "y": "_best_alpha_y", "text": "_best_alpha_text"},
                mark_kwargs={"dx": 8, "dy": -8, "align": "left"},
            ),
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
    chart = chart.properties(title=ferrum.Title("PCA Explained Variance"))
    if cumulative_line:
        from ferrum.layer import Layer

        chart = chart.layer(
            Layer(
                mark="point",
                encoding={"x": "component", "y": "cumulative_variance_ratio"},
                mark_kwargs={"size": 40, "filled": True},
            )
        )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _pca_scree_chart_from_variance_df(
    df: Any,
    *,
    cumulative_line: bool = True,
    threshold: float | None = 0.95,
    theme: Any = None,
):
    """PCA scree chart from a pre-computed variance DataFrame (Rust SVD path)."""
    import ferrum
    from ferrum.layer import Layer

    chart = ferrum.Chart(df).mark_pca_scree(
        cumulative_line=cumulative_line,
        threshold_line=threshold,
    )
    chart = chart.properties(title=ferrum.Title("PCA Explained Variance"))
    if cumulative_line:
        chart = chart.layer(
            Layer(
                mark="point",
                encoding={"x": "component", "y": "cumulative_variance_ratio"},
                mark_kwargs={"size": 40, "filled": True},
            )
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
            x=X(
                "x",
                title=f"{method.upper()}1",
                scale={
                    "type": "linear",
                    "domain": [x_lo - x_pad, x_hi + x_pad],
                },
            ),
            y=Y(
                "y",
                title=f"{method.upper()}2",
                scale={
                    "type": "linear",
                    "domain": [y_lo - y_pad, y_hi + y_pad],
                },
            ),
        )
    )
    chart = chart.properties(title=ferrum.Title("Intercluster Distance Map"))
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
        orient=orient,
        color_field=color_field,
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
            pl.col("correlation")
            .map_elements(
                lambda v: f"{v:.2f}",
                return_dtype=pl.Utf8,
            )
            .alias("correlation_fmt"),
        )
    chart = ferrum.Chart(df).mark_rank2d(annot=annot)
    chart = chart.properties(title=ferrum.Title("Feature Correlation"))
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
    return pl.DataFrame({f"f{j}": arr[:, j].tolist() for j in range(arr.shape[1])})


def _resolve_pc_features(
    df: pl.DataFrame,
    features: list[str] | None,
    hue: str | None,
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
    df: pl.DataFrame,
    features: list[str],
    rescale: str | None,
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
        f"parallel_coordinates(rescale={rescale!r}) — expected 'minmax', 'zscore', or None."
    )


def _parallel_coords_chart_from_dataframe(
    data,
    *,
    features: list[str] | None = None,
    hue: str | None = None,
    rescale: str | None = "minmax",
    alpha: float = 0.3,
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
    long = (
        long.with_columns(
            pl.col("feature").cast(pl.Enum(features)),
        )
        .sort("sample_id", "feature")
        .with_columns(
            pl.col("feature").cast(pl.Utf8),
        )
    )

    if hue is not None:
        # Cast hue to Utf8 so the categorical color scale routes it
        # correctly (same Int64→continuous gotcha as silhouette.cluster).
        long = long.with_columns(pl.col(hue).cast(pl.Utf8))

    chart = ferrum.Chart(long).mark_parallel_coordinates(
        alpha=alpha,
        color_field=hue,
    )
    chart = chart.properties(title=ferrum.Title("Parallel Coordinates"))
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
    layers share the same DataFrame identity (avoiding unnecessary
    null-pad merge) so ``Chart.__add__`` layers them and both color
    encodings resolve
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
        source,
        feat_idx,
        grid_resolution,
        proba=proba,
    )

    if not (scatter and source.y is not None):
        # Pure-boundary path: no overlay, no padding columns, no row mixing.
        grid_df = pl.DataFrame(
            {
                "x": [v - grid_info["dx"] / 2 for v in grid_info["flat_x"]],
                "x2": [v + grid_info["dx"] / 2 for v in grid_info["flat_x"]],
                "y": [v - grid_info["dy"] / 2 for v in grid_info["flat_y"]],
                "y2": [v + grid_info["dy"] / 2 for v in grid_info["flat_y"]],
                "z": [float(v) for v in grid_info["z"]],
            }
        )
        # Non-proba: cast class indices to String for categorical color scale.
        if not proba:
            grid_df = grid_df.with_columns(
                pl.col("z").cast(pl.Int64).cast(pl.Utf8).alias("z"),
            )
        chart = ferrum.Chart(grid_df).mark_decision_boundary(proba=proba)
        chart = chart.properties(title=ferrum.Title("Decision Boundary"))
        if theme is not None:
            chart = chart.theme(theme)
        return chart

    from ferrum.layer import Layer

    unified = _build_decision_boundary_unified(source, grid_info)
    # Non-proba: cast class indices to String for categorical color scale.
    if not proba:
        unified = unified.with_columns(
            pl.col("z").cast(pl.Int64).cast(pl.Utf8).alias("z"),
            pl.col("scatter_z").cast(pl.Int64).cast(pl.Utf8).alias("scatter_z"),
        )
    chart = ferrum.Chart(unified).mark_decision_boundary(proba=proba)
    chart = chart.layer(
        Layer(
            mark="point",
            encoding={"x": "scatter_x", "y": "scatter_y", "color": "scatter_z"},
            mark_kwargs={"stroke": "#000000", "stroke_width": 1.0, "size": 80.0},
        )
    )
    chart = chart.properties(title=ferrum.Title("Decision Boundary"))
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _resolve_decision_boundary_features(
    source: Any,
    features: tuple,
) -> tuple[int, int]:
    feat_idx = tuple(
        source.feature_names.index(f) if isinstance(f, str) else int(f) for f in features
    )
    if len(feat_idx) != 2:
        raise ValueError(
            f"decision_boundary_chart requires exactly 2 features; got {len(feat_idx)}."
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

    # numpy required: 2D positional column slicing (X_np[:, i]) and meshgrid operations.
    X_np = source.X.to_numpy()
    x_col = X_np[:, feat_idx[0]].astype(np.float64)
    y_col = X_np[:, feat_idx[1]].astype(np.float64)
    pad_x = (x_col.max() - x_col.min()) * 0.05
    pad_y = (y_col.max() - y_col.min()) * 0.05
    xs = np.linspace(
        x_col.min() - pad_x,
        x_col.max() + pad_x,
        int(grid_resolution),
    )
    ys = np.linspace(
        y_col.min() - pad_y,
        y_col.max() + pad_y,
        int(grid_resolution),
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

    y_raw = np.asarray(source.y)
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

    return pl.DataFrame(
        {
            "x": [v - dx / 2 for v in flat_x] + nulls_scatter,
            "x2": [v + dx / 2 for v in flat_x] + nulls_scatter,
            "y": [v - dy / 2 for v in flat_y] + nulls_scatter,
            "y2": [v + dy / 2 for v in flat_y] + nulls_scatter,
            "z": [float(v) for v in z] + nulls_scatter,
            "scatter_x": nulls_grid + [float(v) for v in x_col],
            "scatter_y": nulls_grid + [float(v) for v in y_col],
            "scatter_z": nulls_grid + [float(v) for v in scatter_z],
        },
        schema={
            "x": pl.Float64,
            "x2": pl.Float64,
            "y": pl.Float64,
            "y2": pl.Float64,
            "z": pl.Float64,
            "scatter_x": pl.Float64,
            "scatter_y": pl.Float64,
            "scatter_z": pl.Float64,
        },
    )


# ---------------------------------------------------------------------------
# 10f builders — sweep-k cluster diagnostics
# ---------------------------------------------------------------------------


def _cluster_diagnostics_chart(
    X: Any,
    *,
    ks: Any,
    method: str,
    scoring: str,
    n_init: int,
    random_state: int | None,
    theme: Any,
):
    """Build the elbow / silhouette / both panel(s) for cluster_diagnostics.

    Unlike the other builders, this one sweeps the clusterer over the
    requested ``ks`` rather than wrapping a single fitted ``ModelSource``.
    See ``ferrum.cluster_diagnostics`` for the user-facing parameter docs.
    """
    import ferrum
    import numpy as np
    import pyarrow as pa
    from ferrum import _core
    from sklearn.cluster import KMeans, AgglomerativeClustering

    # numpy required: manual inertia computation uses 2D positional indexing and mask ops.
    X_np = np.asarray(X, dtype=np.float64)
    X_np = np.ascontiguousarray(X_np)
    x_arrow = pa.RecordBatch.from_pydict(
        {f"f{j}": X_np[:, j].tolist() for j in range(X_np.shape[1])}
    )
    seed = 0 if random_state is None else int(random_state)
    rows: list[dict] = []
    for k in ks:
        if method == "kmeans":
            m = KMeans(
                n_clusters=int(k),
                n_init=int(n_init),
                random_state=seed,
            ).fit(X_np)
            inertia = float(m.inertia_)
            labels = m.labels_
        else:  # hierarchical
            m = AgglomerativeClustering(n_clusters=int(k), linkage="ward").fit(X_np)
            labels = m.labels_
            inertia = 0.0
            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                centroid = X_np[mask].mean(axis=0)
                inertia += float(np.sum((X_np[mask] - centroid) ** 2))
        labels_arrow = pa.array(labels.astype(int).tolist(), type=pa.int64())
        rows.append(
            {
                "k": int(k),
                "inertia": inertia,
                "silhouette": float(_core.silhouette_score(x_arrow, labels_arrow, "euclidean")),
            }
        )
    df = pl.DataFrame(rows)
    from ferrum.encoding import X as XEnc, Y as YEnc

    from ferrum.layer import Layer

    # Schwabish C11: elbow detection via second derivative of inertia.
    # Inject sentinel columns for the elbow vline + text before building
    # the chart so both layers share the same DataFrame.
    inertias = np.array([r["inertia"] for r in rows])
    ks_list = list(ks)
    elbow_layers: list[Layer] = []
    if len(inertias) >= 3:
        second_diff = np.diff(inertias, n=2)
        elbow_detect_idx = int(np.argmax(second_diff)) + 1
        elbow_k_val = int(ks_list[elbow_detect_idx])
        elbow_score_val = float(inertias[elbow_detect_idx])
        n = df.height
        df = df.with_columns(
            pl.Series("_elbow_k", [float(elbow_k_val)] + [None] * (n - 1), dtype=pl.Float64),
            pl.Series("_elbow_y", [elbow_score_val] + [None] * (n - 1), dtype=pl.Float64),
            pl.Series(
                "_elbow_text",
                [f"elbow at k={elbow_k_val}"] + [None] * (n - 1),
                dtype=pl.Utf8,
            ),
        )
        elbow_layers = [
            Layer(
                mark="rule",
                encoding={"x": "_elbow_k"},
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
            ),
            Layer(
                mark="text",
                encoding={"x": "_elbow_k", "y": "_elbow_y", "text": "_elbow_text"},
                mark_kwargs={"dx": 8, "dy": -8, "align": "left"},
            ),
        ]
    elbow = (
        ferrum.Chart(df)
        .mark_line()
        .encode(x=XEnc("k", title="k"), y=YEnc("inertia", title="Distortion score"))
        .properties(title=ferrum.Title("Distortion Score Elbow"))
        .layer(
            Layer(
                mark="point",
                encoding={"x": "k", "y": "inertia"},
                mark_kwargs={"size": 40, "filled": True},
            ),
            *elbow_layers,
        )
    )
    sil = (
        ferrum.Chart(df)
        .mark_line()
        .encode(x=XEnc("k", title="k"), y=YEnc("silhouette", title="Silhouette score"))
        .properties(title=ferrum.Title("Silhouette Score"))
        .layer(
            Layer(
                mark="point",
                encoding={"x": "k", "y": "silhouette"},
                mark_kwargs={"size": 40, "filled": True},
            )
        )
    )
    if scoring == "elbow":
        chart = elbow
    elif scoring == "silhouette":
        chart = sil
    else:
        chart = elbow | sil
    if theme is not None:
        chart = chart.theme(theme)
    return chart
