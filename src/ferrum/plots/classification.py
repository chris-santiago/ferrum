"""Classification domain figure functions and their private builders.

Public API
----------
roc_chart, pr_chart, calibration_chart, gain_chart, lift_chart,
discrimination_threshold_chart, confusion_matrix_chart,
class_prediction_error_chart, classification_report_chart, class_balance_chart.

Each public function wraps ``_resolve_source`` (shared across all figure
functions) and dispatches to a co-located ``_*_from_source`` builder that
produces a fully-formed ``Chart``.

Internal-only builders
----------------------
_classification_report_chart, _class_balance_chart_from_dataframe.
These are consumed by visualizers but have no matching public figure
function today.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ferrum.encoding import X, Y
from ferrum._overrides import _apply_overrides, register_layer_names
from ferrum.plots._helpers import _color_field_for

# Register the calibration layer-name catalog entry.
register_layer_names("calibration", frozenset({"line", "reference", "point"}))


# ---------------------------------------------------------------------------
# Domain-specific helpers (classification curves only)
# ---------------------------------------------------------------------------


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
    - ``{prefix}_label_x: Float64`` -- x coordinate of the label anchor
      (fixed at 0.65 on the data axis so labels live in the lower-right
      region where ROC / PR curves typically asymptote).
    - ``{prefix}_label_y: Float64`` -- staggered y per group (one slot
      every 0.06 of axis range, starting at 0.05) so multi-class labels
      don't overlap.
    - ``{prefix}_label: Utf8`` -- formatted string
      ``"{group_value}: {metric_label} = {value:.3f}"``.

    All rows except one per group are null; ``mark_text`` skips nulls
    so exactly N labels render for N groups.  Used by ``mark_roc``'s
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
    precision stays in ``(0, 1]``.  Each iso curve gets ``n_points``
    rows with synthetic ``_iso_recall`` / ``_iso_precision`` /
    ``_iso_f`` columns plus one ``_iso_label_x`` / ``_iso_label_y`` /
    ``_iso_label`` row near the curve's right end for the F-value
    annotation.  The original PR columns are extended with nulls on the
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
        # Label anchor at the rightmost (recall ~ 1) point of the iso curve.
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


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _roc_chart_from_source(
    source: Any,
    *,
    per_class: bool = True,
    average: str | None = "macro",
    annotate_auc: bool = True,
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
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

    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _pr_chart_from_source(
    source: Any,
    *,
    per_class: bool = True,
    average: str | None = "macro",
    annotate_ap: bool = True,
    iso_lines: bool = False,
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
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
    # fallback.  Binary classifiers only -- multi-class baselines would
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

    # Active title on single-curve PR charts.  Drop iso-curve sentinel rows
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
        from ferrum._layer import _Layer

        chart = chart.layer(
            _Layer(
                mark="rule",
                encoding={"y": "_baseline_y"},
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
                name="baseline",
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

    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _calibration_chart_from_source(
    source: Any,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    annotate_brier: bool = True,
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
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

    # Active title fires when exactly one model.  Compute the true
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

    from ferrum._layer import _Layer

    point_enc: dict = {"x": "mean_predicted", "y": "fraction_positive"}
    if color is not None:
        point_enc["color"] = color
    chart = chart.layer(
        _Layer(mark="point", encoding=point_enc, mark_kwargs={"size": 40, "filled": True}, name="point")
    )

    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _gain_chart_from_source(
    source: Any,
    *,
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Build a cumulative-gain chart from a ModelSource.

    Replaces the categorical color legend with endpoint-anchored
    direct labels via ``_direct_label_endpoint`` and adds a
    ``Cumulative gain`` active title -- unconditional, matching the
    learning_curve / validation_curve / lift sibling figures
    (Schwabish C8 audit-rework, 2026-05-12).  Chart-API users who
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
    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _lift_chart_from_source(
    source: Any,
    *,
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Build a lift chart from a ModelSource.

    Direct labels + legend suppression + ``Lift`` active title --
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
    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _confusion_chart_from_source(
    source: Any,
    *,
    normalize: str | None = "true",
    annotate: bool = True,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
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
    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _class_prediction_error_chart_from_source(
    source: Any,
    *,
    normalize: bool = False,
    show_counts: bool = True,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Build a stacked-bar class-prediction-error chart from a ModelSource.

    Reuses the unnormalized confusion-matrix output as the underlying
    long-form data.

    Schwabish SB-followup (2026-05-12): ``show_counts=True`` (default)
    routes through ``Chart.mark_class_prediction_error(show_counts=...)``
    which owns both the ``_count_text`` sentinel column (via its
    ``data_transform``) and the text layer emitted by
    ``desugar_class_prediction_error``.  The text layer carries the
    same ``Stack`` position adjustment as the bar layer; the renderer
    maps text-mark y values to segment MIDPOINTS so labels land
    centred without any Python-side cumsum duplication.  Chart-API
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
    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
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
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Build a discrimination-threshold chart from a ModelSource.

    The underlying DataFrame is unpivoted to long form
    ``(threshold, metric, value)`` for plotting.

    Schwabish (post-C7 audit-rework, 2026-05-12): adds a
    ``Discrimination threshold`` active title and threads
    ``threshold_line`` + ``optimum_label`` through to
    ``Chart.mark_discrimination_threshold``.  The mark owns the
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
    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _classification_report_chart(
    source: Any,
    *,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Heatmap of per-class precision/recall/F1 (rect + text overlay).

    Long-form data: one row per (class, metric) cell with ``value`` and
    ``value_fmt``.  Renders via the same rect-plus-text pattern as
    ``mark_confusion``.
    """
    from ferrum._diagnostics.deps import require_sklearn

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
    for cls_label, metrics_dict in report.items():
        if cls_label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(metrics_dict, dict):
            continue
        for m_name in ("precision", "recall", "f1-score"):
            val = float(metrics_dict[m_name])
            rows.append(
                {
                    "class": str(cls_label),
                    "metric": m_name,
                    "value": val,
                    "value_fmt": f"{val:.2f}",
                }
            )
    df = pl.DataFrame(rows)

    from ferrum._layer import _Layer

    chart = (
        ferrum.Chart(df)
        .mark_rect()
        .encode(
            x="metric",
            y="class",
            color="value",
        )
        .layer(
            _Layer(
                mark="text",
                encoding={"x": "metric", "y": "class", "text": "value_fmt"},
                name="annotation",
            )
        )
    )
    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _class_balance_chart_from_dataframe(
    y_series: Any,
    *,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Bar chart of class counts.

    Operates on ``y`` alone (no model required).  Computes per-class
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
    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


# ---------------------------------------------------------------------------
# Public figure functions
# ---------------------------------------------------------------------------


def _resolve_source(
    model_or_source: Any,
    X_data: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    random_state: int | None = None,
    compare: dict[str, Any] | None = None,
) -> Any:
    """Resolve a figure-function input into a ModelSource, ComparedModelSource,
    or _PrecomputedSource.

    Thin wrapper delegating to ``ferrum.plots._helpers._resolve_source``.
    """
    from ferrum.plots._helpers import _resolve_source as _resolve

    return _resolve(
        model_or_source, X_data, y,
        y_true=y_true, y_pred=y_pred,
        random_state=random_state, compare=compare,
    )


def roc_chart(
    model_or_source: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    per_class: bool = True,
    average: str | None = "macro",
    annotate_auc: bool = True,
    subtitle: str | None = None,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """ROC curve chart for a classifier.

    Plots true-positive rate vs. false-positive rate, one curve per
    class (default) or a single macro/micro/weighted-averaged curve.
    Supports multi-model comparison via ``compare=``.

    Parameters
    ----------
    model_or_source : estimator, ModelSource, or dict of str -> estimator, optional
        Fitted sklearn-compatible classifier, an explicit
        ``ferrum.ModelSource``, or a dict of named estimators for
        comparison.  When a dict is passed, each estimator is evaluated
        and curves are overlaid.  Mutually exclusive with ``y_true``/``y_pred``.
    X : array-like, optional
        Feature matrix.  Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels.  Required when ``model_or_source`` is a raw
        estimator.
    y_true : array-like, optional
        Ground-truth class labels for the precomputed path.  Must be
        paired with ``y_pred``; mutually exclusive with ``model_or_source``.
    y_pred : array-like, optional
        Soft scores / probabilities for the precomputed path.  1-D for
        binary classifiers (positive-class scores); 2-D
        ``(n_samples, n_classes)`` for multiclass.
    per_class : bool, default True
        When ``True``, one ROC curve per class is drawn using the
        one-vs-rest scheme.  When ``False``, a single curve averaged
        per ``average`` is drawn.
    average : {"macro", "micro", "weighted"} or None, default "macro"
        Averaging method used when ``per_class=False``.  Ignored when
        ``per_class=True``.
    annotate_auc : bool, default True
        When ``True``, injects one text label per class showing the AUC
        value to 3 decimal places, anchored in the lower-right corner
        of the plot.
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
    compare : dict of str -> estimator or None, default None
        Additional estimators to overlay.  Keys become model labels.
        ``model_or_source`` is treated as the base model (label
        ``"base"``).
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        ROC curve chart with one line per class (or averaged curve).

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.roc_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)

    Precomputed path — bypass the model entirely:

    >>> fm.roc_chart(y_true=y_test, y_pred=clf.predict_proba(X_test))
    """
    source = _resolve_source(
        model_or_source,
        X,
        y,
        y_true=y_true,
        y_pred=y_pred,
        random_state=random_state,
        compare=compare,
    )
    return _roc_chart_from_source(
        source,
        per_class=per_class,
        average=average,
        annotate_auc=annotate_auc,
        subtitle=subtitle,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def pr_chart(
    model_or_source: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    per_class: bool = True,
    average: str | None = "macro",
    annotate_ap: bool = True,
    iso_lines: bool = False,
    subtitle: str | None = None,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Precision-recall curve chart for a classifier.

    Plots precision vs. recall, one curve per class (default) or a single
    averaged curve.  Both axes are pinned to ``[0, 1.05]`` so curves
    render against the full precision-recall space (same convention as
    sklearn and yellowbrick).  Supports multi-model comparison via
    ``compare=``.

    Parameters
    ----------
    model_or_source : estimator, ModelSource, or dict of str -> estimator
        Fitted sklearn-compatible classifier, an explicit
        ``ferrum.ModelSource``, or a dict of named estimators for
        comparison.
    X : array-like, optional
        Feature matrix.  Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels.  Required when ``model_or_source`` is a raw
        estimator.
    per_class : bool, default True
        When ``True``, one PR curve per class is drawn using the
        one-vs-rest scheme.  When ``False``, a single curve averaged per
        ``average`` is drawn.  Binary classifiers always render a single
        curve regardless.
    average : {"macro", "micro", "weighted"} or None, default "macro"
        Averaging method used when ``per_class=False``.  Ignored when
        ``per_class=True``.  Macro and weighted variants interpolate
        per-class precision over a shared recall grid; micro ravels the
        binarized labels into a single curve.
    annotate_ap : bool, default True
        When ``True``, injects one text label per class near the lower-
        right corner of the plot showing average precision (AP) to 3
        decimal places.
    iso_lines : bool, default False
        When ``True``, overlays F-score iso-contours at
        F={0.2, 0.4, 0.6, 0.8} so users can read off combined
        precision-recall quality directly from the chart.
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
    compare : dict of str -> estimator or None, default None
        Additional estimators to overlay.  Keys become model labels.
        ``model_or_source`` is treated as the base model (label
        ``"base"``).
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Precision-recall curve chart with one line per class (or an
        averaged summary curve).

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.pr_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)

    Precomputed path:

    >>> fm.pr_chart(y_true=y_test, y_pred=clf.predict_proba(X_test))
    """
    source = _resolve_source(
        model_or_source,
        X,
        y,
        y_true=y_true,
        y_pred=y_pred,
        random_state=random_state,
        compare=compare,
    )
    return _pr_chart_from_source(
        source,
        per_class=per_class,
        average=average,
        annotate_ap=annotate_ap,
        iso_lines=iso_lines,
        subtitle=subtitle,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def calibration_chart(
    model_or_source: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    annotate_brier: bool = True,
    subtitle: str | None = None,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Calibration (reliability) curve for one or more classifiers.

    Plots mean predicted probability vs. fraction of positives in each
    bin, following sklearn's ``calibration_curve`` convention.  Supports
    multi-model comparison via ``compare=``.

    Parameters
    ----------
    model_or_source : estimator, ModelSource, or dict of str -> estimator
        Fitted sklearn-compatible classifier, an explicit
        ``ferrum.ModelSource``, or a dict of named estimators for
        comparison.  When a dict is passed, each estimator is evaluated
        and curves are overlaid.
    X : array-like, optional
        Feature matrix.  Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True binary labels.  Required when ``model_or_source`` is a raw
        estimator.
    n_bins : int, default 10
        Number of probability bins for the reliability diagram.
    strategy : {"uniform", "quantile"}, default "uniform"
        Binning strategy forwarded to ``sklearn.calibration.calibration_curve``.
        ``"uniform"`` uses equally-spaced bins; ``"quantile"`` uses
        equal-frequency bins.
    annotate_brier : bool, default True
        When ``True``, attaches the :class:`BrierLabel` composite (spec
        3.11) showing the Brier score per series.  The chart title also
        encodes the Brier value when exactly one model is shown.
    subtitle : str or None, default None
        Optional one-line subtitle drawn beneath the chart title.
    compare : dict of str -> estimator or None, default None
        Additional estimators to overlay.  Keys become model labels.
        ``model_or_source`` is treated as the base model (label
        ``"base"``).
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Reliability diagram with one curve per model plus a perfect-
        calibration diagonal reference.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.calibration_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)

    Precomputed path (``y_pred`` = 1-D predicted probabilities for positive class):

    >>> fm.calibration_chart(y_true=y_test, y_pred=clf.predict_proba(X_test)[:, 1])
    """
    source = _resolve_source(
        model_or_source,
        X,
        y,
        y_true=y_true,
        y_pred=y_pred,
        random_state=random_state,
        compare=compare,
    )
    return _calibration_chart_from_source(
        source,
        n_bins=n_bins,
        strategy=strategy,
        annotate_brier=annotate_brier,
        subtitle=subtitle,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def gain_chart(
    model_or_source: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    compare: dict[str, Any] | None = None,
    subtitle: str | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Cumulative-gain curve for a classifier.

    Plots the fraction of positive cases captured vs. the fraction of
    samples scored, one curve per class.  Useful for evaluating the
    benefit of targeting a top-ranked subset.  The categorical legend
    is replaced with endpoint-anchored direct labels -- unconditional,
    matching the learning_curve / validation_curve / lift sibling
    figures (Schwabish C8 audit-rework, 2026-05-12).  Use
    ``Chart(df).mark_gain(...)`` directly to keep the legend.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix.  Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels.  Required when ``model_or_source`` is a raw
        estimator.
    compare : dict[str, estimator] or None, default None
        Multi-model overlay.  Keys are display names; values are fitted
        estimators.  Routes through ``_resolve_source`` ->
        ``ComparedModelSource``.
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Cumulative-gain curve with one line per class.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.gain_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)

    Precomputed path (``y_pred`` = soft scores, 1-D binary or 2-D multiclass):

    >>> fm.gain_chart(y_true=y_test, y_pred=clf.predict_proba(X_test))
    """
    source = _resolve_source(model_or_source, X, y, y_true=y_true, y_pred=y_pred, random_state=random_state, compare=compare)
    return _gain_chart_from_source(source, subtitle=subtitle, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme)


def lift_chart(
    model_or_source: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    compare: dict[str, Any] | None = None,
    subtitle: str | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Lift curve for a classifier.

    Plots the ratio of positive-hit rate in the scored top-n vs. random
    baseline, one curve per class.  Values above 1 indicate the model
    outperforms random selection at that depth.  The categorical legend
    is replaced with endpoint-anchored direct labels -- unconditional
    (Schwabish C8 audit-rework, 2026-05-12).  Use
    ``Chart(df).mark_lift(...)`` directly to keep the legend.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix.  Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels.  Required when ``model_or_source`` is a raw
        estimator.
    compare : dict[str, estimator] or None, default None
        Multi-model overlay.  Keys are display names; values are fitted
        estimators.  Routes through ``_resolve_source`` ->
        ``ComparedModelSource``.
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Lift curve with one line per class.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.lift_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)

    Precomputed path (``y_pred`` = soft scores, 1-D binary or 2-D multiclass):

    >>> fm.lift_chart(y_true=y_test, y_pred=clf.predict_proba(X_test))
    """
    source = _resolve_source(model_or_source, X, y, y_true=y_true, y_pred=y_pred, random_state=random_state, compare=compare)
    return _lift_chart_from_source(source, subtitle=subtitle, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme)


def confusion_matrix_chart(
    model_or_source: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    normalize: str | None = "true",
    annotate: bool = True,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Confusion matrix heatmap for a classifier.

    Renders an ordinal heatmap of (actual class, predicted class) cell
    values with optional per-cell text annotations.  Normalization
    follows sklearn's ``confusion_matrix`` conventions.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix.  Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels.  Required when ``model_or_source`` is a raw
        estimator.
    normalize : {"true", "pred", "all"} or None, default "true"
        Normalization scheme for cell values.  ``None`` shows raw
        counts; ``"true"`` normalizes over actual classes (rows);
        ``"pred"`` normalizes over predicted classes (columns);
        ``"all"`` normalizes over the total number of samples.
    annotate : bool, default True
        When ``True``, overlays the numeric cell value as a text
        label in each cell.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Confusion matrix heatmap with optional cell annotations.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.confusion_matrix_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)

    Precomputed path (``y_pred`` = 1-D hard class labels):

    >>> fm.confusion_matrix_chart(y_true=y_test, y_pred=clf.predict(X_test))
    """
    source = _resolve_source(model_or_source, X, y, y_true=y_true, y_pred=y_pred, random_state=random_state)
    return _confusion_chart_from_source(
        source,
        normalize=normalize,
        annotate=annotate,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def class_prediction_error_chart(
    model_or_source: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    normalize: bool = False,
    show_counts: bool = True,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Class prediction-error stacked bar chart for a classifier.

    One bar per predicted class, stacked by actual class so misclassified
    segments are visually distinct.  The data source is the unnormalized
    confusion matrix reshaped to long form.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix.  Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels.  Required when ``model_or_source`` is a raw
        estimator.
    normalize : bool, default False
        When ``True``, each bar is normalized to 100% (relative
        composition).  When ``False``, bars show raw sample counts.
    show_counts : bool, default True
        When ``True``, overlays per-segment count text at the vertical
        centre of each bar segment (empty segments -- ``value == 0`` --
        are skipped).  Raw counts are shown regardless of ``normalize``.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Stacked bar chart with one bar per predicted class.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.class_prediction_error_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)

    Precomputed path (``y_pred`` = 1-D hard class labels):

    >>> fm.class_prediction_error_chart(y_true=y_test, y_pred=clf.predict(X_test))
    """
    source = _resolve_source(model_or_source, X, y, y_true=y_true, y_pred=y_pred, random_state=random_state)
    return _class_prediction_error_chart_from_source(
        source,
        normalize=normalize,
        show_counts=show_counts,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def discrimination_threshold_chart(
    model_or_source: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    n_thresholds: int = 50,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
    cv: Any = None,
    threshold_line: bool = False,
    optimum_label: bool = True,
    compare: dict[str, Any] | None = None,
    subtitle: str | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Discrimination-threshold sweep chart for a binary classifier.

    Plots multiple classification metrics (precision, recall, F1,
    queue rate) as a function of the decision threshold, allowing users
    to select an operating point that balances competing objectives.
    The underlying data is unpivoted to long form for multi-metric
    rendering.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible binary classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix.  Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True binary labels.  Required when ``model_or_source`` is a raw
        estimator.
    n_thresholds : int, default 50
        Number of evenly spaced threshold values in ``[0, 1]`` to
        evaluate.
    metrics : tuple of str, default ("precision", "recall", "f1", "queue_rate")
        Metric names to plot.  Each must be a column in the threshold
        sweep DataFrame.
    cv : int, cross-validator, or None, default None
        When provided, the threshold sweep is computed via cross-
        validation rather than a single train/test split.
    threshold_line : bool, default False
        When ``True``, injects a vertical reference rule at the
        threshold that maximises F1.
    optimum_label : bool, default True
        When ``True``, overlays a text annotation at the F1-optimum
        point showing the threshold and F1 value
        (e.g. ``"max F1 = 0.872 @ t=0.43"``).  Composes with
        ``threshold_line``; either can be enabled independently.
    compare : dict[str, estimator] or None, default None
        Multi-model overlay.  Keys are display names; values are fitted
        estimators.  Routes through ``_resolve_source`` ->
        ``ComparedModelSource``.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Multi-metric line chart over the threshold range.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.discrimination_threshold_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)


    Precomputed path (``y_pred`` = 1-D positive-class scores; ``cv=`` not supported):

    >>> fm.discrimination_threshold_chart(y_true=y_test, y_pred=clf.predict_proba(X_test)[:, 1])
    """
    source = _resolve_source(model_or_source, X, y, y_true=y_true, y_pred=y_pred, random_state=random_state, compare=compare)
    return _discrimination_threshold_chart_from_source(
        source,
        n_thresholds=n_thresholds,
        metrics=metrics,
        cv=cv,
        threshold_line=threshold_line,
        optimum_label=optimum_label,
        subtitle=subtitle,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def classification_report_chart(
    model_or_source: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Per-class precision / recall / F1 heatmap for a classifier.

    Renders a rect-plus-text heatmap where rows are class labels, columns
    are metrics (``precision``, ``recall``, ``f1-score``), and cell color
    encodes the metric value. Each cell is annotated with its value to two
    decimal places.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        A fitted sklearn-compatible classifier that exposes ``predict``,
        or an explicit ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y : array-like, optional
        True labels. Required when ``model_or_source`` is a raw estimator;
        ignored when it is already a ``ModelSource``.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Heatmap with per-class precision / recall / F1-score cells.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.classification_report_chart(clf, X_test, y_test)
    """
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _classification_report_chart(
        source,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def class_balance_chart(
    y: Any,
    *,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Bar chart of per-class label counts.

    Computes the count of each unique class label and renders a vertical
    bar chart. No model is required — operates on the target array alone.

    Parameters
    ----------
    y : array-like
        Target label array (1-D). Accepts polars ``Series``, numpy
        arrays, and any iterable convertible to a flat list.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Vertical bar chart with class labels on x and counts on y.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.class_balance_chart(y_train)
    """
    return _class_balance_chart_from_dataframe(
        y,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )
