"""Phase 10 model-diagnostic mark desugars (Python-side).

Each ``desugar_<name>(x_field, y_field, **kwargs)`` returns the 5-tuple
``("__layered__", transforms: list, None, None, layers: list[dict])``
consumed by ``Chart._resolve_pending`` when the user calls
``chart.mark_<name>(...)``.

These desugars operate on DataFrames whose columns are committed by the
``ModelSource`` method that produced them (e.g. ``predictions()`` →
``y_true``, ``y_pred``, ``residual``, ``studentized_residual``). They
ignore the positional ``x_field``/``y_field`` arguments — the calling
``Chart.mark_*`` method knows the diagnostic schema and references the
columns literally.

Reference lines: Rust's ``mark_rule`` renders one line per row. To draw
a single reference line, the corresponding ``Chart.mark_*`` method
injects a one-non-null-row column (see
``ferrum._diagnostics.charts._inject_constant``); the desugar references
that column by name. No new Rust marks or transforms.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from ferrum._layer import _Layer
from ferrum.marks._mark_kwargs import (
    apply_user_mark_kwargs as _apply,
    validate_user_mark_kwargs as _validate,
)


def desugar_residuals(
    x_field: str | None,
    y_field: str | None,
    *,
    kind: str = "studentized",
    reference_line: bool = True,
    cook_threshold: float | None = None,
    color_field: str | None = None,
) -> tuple:
    """Residuals diagnostic: scatter of (y_pred, residual) plus optional y=0 rule.

    Data contract: the chart's data must carry columns ``y_pred`` and either
    ``residual`` (kind="raw") or ``studentized_residual`` (kind in
    "studentized"/"scaled"). When ``reference_line=True`` the data must
    also carry the injected ``_ref_zero`` column (the ``Chart.mark_residuals``
    method takes care of this).

    When ``cook_threshold`` is set (a float, or the literal ``"auto"`` for
    the conventional ``4 / n`` rule), the chart builder injects
    ``_cook_outlier_x`` / ``_cook_outlier_y`` columns that hold the
    (y_pred, residual) coordinates only for observations whose leverage-
    aware Cook's distance exceeds the threshold (all other rows null).
    This desugar then overlays a second ``mark_point`` layer keyed on
    those columns; Rust's mark_point skips null rows so exactly K outlier
    markers render, drawn in red with a black outline to stand out
    against the base scatter. Requires the wrapped estimator to expose
    ``coef_`` so the hat matrix is computable (the
    ``ModelSource.predictions()`` step has already filled
    ``cooks_distance`` with NaN for non-linear estimators).
    """
    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"
    point_enc: dict[str, Any] = {"x": "y_pred", "y": y_col}
    if color_field is not None:
        point_enc["color"] = color_field
    layers: list = [_Layer(mark="point", encoding=point_enc)]
    if reference_line:
        layers.append(_Layer(
            mark="rule",
            encoding={"y": "_ref_zero"},
            mark_kwargs={"stroke_dash": [4, 4]},
        ))
    if cook_threshold is not None:
        layers.append(_Layer(
            mark="point",
            encoding={
                "x": "_cook_outlier_x",
                "y": "_cook_outlier_y",
            },
            mark_kwargs={
                "fill": "#e15759",  # tableau red
                "stroke": "#000000",
                "stroke_width": 1.0,
                "size": 80.0,
            },
        ))
    return ("__layered__", [], None, None, layers)


def desugar_prediction_error(
    x_field: str | None,
    y_field: str | None,
    *,
    identity_line: bool = True,
    ci: float | None = None,
    reference_band: bool = False,
    color_field: str | None = None,
) -> tuple:
    """Actual vs predicted: scatter of (y_true, y_pred) + optional identity line.

    Data contract: columns ``y_true`` and ``y_pred``. When
    ``identity_line=True`` the data must be sorted ascending by ``y_true`` so
    the line layer renders as a clean y=x diagonal (handled by
    ``Chart.mark_prediction_error``).

    When ``ci is not None`` or ``reference_band=True``, the data must also
    carry the injected ``_pe_band_lo`` / ``_pe_band_hi`` columns (the chart
    builder pre-computes these as the ``ci``-width band around the identity
    line), and this desugar emits a ``ribbon`` layer between those bounds with
    ``opacity=0.2`` so the underlying scatter remains visible.
    """
    point_enc: dict[str, Any] = {"x": "y_true", "y": "y_pred"}
    if color_field is not None:
        point_enc["color"] = color_field
    # Layer ordering: point first so layer-0-driven axis-scale resolution
    # picks up `y_pred` as the y-axis title. The ribbon and identity-line
    # layers paint on top; opacity=0.2 on the ribbon keeps the underlying
    # points visible.
    layers: list = [_Layer(mark="point", encoding=point_enc)]
    if ci is not None or reference_band:
        layers.append(_Layer(
            mark="ribbon",
            encoding={
                "x": "y_true",
                "y": "_pe_band_lo",
                "y2": "_pe_band_hi",
            },
            mark_kwargs={"opacity": 0.2},
        ))
    if identity_line:
        layers.append(_Layer(
            mark="line",
            encoding={"x": "y_true", "y": "y_true"},
            mark_kwargs={"stroke_dash": [4, 4]},
        ))
    return ("__layered__", [], None, None, layers)


# --- 10b: classification curves --------------------------------------


def desugar_roc(
    x_field: str | None,
    y_field: str | None,
    *,
    average: str | None = None,
    reference_line: bool = True,
    annotate_auc: bool = False,
    color_field: str | None = "class",
) -> tuple:
    """ROC curve mark.

    Data contract: columns ``fpr``, ``tpr``, ``class``, ``auc`` as emitted
    by ``ModelSource.roc_curve()``. When ``reference_line=True`` the
    calling ``Chart.mark_roc`` method pre-sorts the data ascending by
    ``fpr`` so the diagonal line layer is monotonic.

    When ``annotate_auc=True`` the chart builder
    (``_roc_chart_from_source``) injects ``_auc_label_x`` / ``_auc_label_y``
    / ``_auc_label`` columns — one non-null row per class — and this
    desugar emits a ``mark_text`` layer that references them. Rust's
    ``mark_text`` skips null rows, so exactly one label renders per
    class. ``average`` is informational at the mark layer — the figure
    builder is responsible for shaping the data appropriately before
    constructing the chart.
    """
    del average  # informational at the mark layer
    line_enc: dict[str, Any] = {"x": "fpr", "y": "tpr"}
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list = [_Layer(mark="line", encoding=line_enc)]
    if reference_line:
        layers.append(_Layer(
            mark="line",
            encoding={"x": "fpr", "y": "fpr"},
            mark_kwargs={"stroke_dash": [4, 4]},
        ))
    if annotate_auc:
        text_enc: dict[str, Any] = {
            "x": "_auc_label_x", "y": "_auc_label_y", "text": "_auc_label",
        }
        if color_field is not None:
            text_enc["color"] = color_field
        layers.append(_Layer(
            mark="text",
            encoding=text_enc,
            mark_kwargs={"align": "left"},
        ))
    return ("__layered__", [], None, None, layers)


def desugar_pr(
    x_field: str | None,
    y_field: str | None,
    *,
    average: str | None = None,
    annotate_ap: bool = False,
    iso_lines: bool = False,
    color_field: str | None = "class",
) -> tuple:
    """Precision-recall curve mark.

    Data contract: ``recall``, ``precision``, ``class``, ``ap`` as emitted
    by ``ModelSource.pr_curve()``.

    When ``annotate_ap=True`` the chart builder
    (``_pr_chart_from_source``) injects ``_ap_label_x`` / ``_ap_label_y`` /
    ``_ap_label`` columns — one non-null row per class — and this
    desugar emits a ``mark_text`` layer that references them.

    When ``iso_lines=True`` the chart builder appends F-score iso-curve rows
    for F in {0.2, 0.4, 0.6, 0.8} with synthetic columns ``_iso_recall``,
    ``_iso_precision``, ``_iso_f`` (Utf8 F-score label used as the line color
    grouping key), ``_iso_label_x``, ``_iso_label_y``, and ``_iso_label``; the
    desugar emits a grey dashed line layer grouped by ``_iso_f`` plus a text
    layer at ``(_iso_label_x, _iso_label_y)`` for the iso labels.
    """
    del average  # informational at the mark layer
    line_enc: dict[str, Any] = {"x": "recall", "y": "precision"}
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list = [_Layer(mark="line", encoding=line_enc)]
    if iso_lines:
        # Iso-F lines are rendered as a separate line layer grouped by
        # `_iso_f` (Utf8 string of the F-score). The chart builder appends
        # one row per (F, recall_step) point along each iso curve.
        layers.append(_Layer(
            mark="line",
            encoding={
                "x": "_iso_recall",
                "y": "_iso_precision",
                "color": "_iso_f",
            },
            mark_kwargs={"stroke_dash": [2, 4], "opacity": 0.6},
        ))
        layers.append(_Layer(
            mark="text",
            encoding={
                "x": "_iso_label_x",
                "y": "_iso_label_y",
                "text": "_iso_label",
            },
            mark_kwargs={"align": "left", "font_size": 9.0},
        ))
    if annotate_ap:
        text_enc: dict[str, Any] = {
            "x": "_ap_label_x", "y": "_ap_label_y", "text": "_ap_label",
        }
        if color_field is not None:
            text_enc["color"] = color_field
        layers.append(_Layer(
            mark="text",
            encoding=text_enc,
            mark_kwargs={"align": "left"},
        ))
    return ("__layered__", [], None, None, layers)


def desugar_calibration(
    x_field: str | None,
    y_field: str | None,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    reference_line: bool = True,
    color_field: str | None = None,
) -> tuple:
    """Calibration (reliability) curve mark.

    Data contract: ``mean_predicted``, ``fraction_positive``, ``count`` as
    emitted by ``ModelSource.calibration_curve()``. When
    ``reference_line=True`` the calling ``Chart.mark_calibration`` method
    pre-sorts data ascending by ``mean_predicted`` so the y=x line is
    monotonic. ``n_bins``/``strategy`` are informational at the mark layer
    (the data is already binned).

    Layer wiring (Phase 8a-compliant). The calibration curve reads from the
    primary input (one row per (model, bin)).  The y=x reference diagonal
    reads from a named ``ReferenceLine`` transform that emits exactly two
    rows for the line endpoints — so the diagonal renders once per chart
    regardless of how many models are layered on top.
    """
    del n_bins, strategy
    line_enc: dict[str, Any] = {
        "x": "mean_predicted", "y": "fraction_positive",
    }
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list = [_Layer(mark="line", encoding=line_enc)]
    transforms: list = []
    if reference_line:
        from ferrum import ReferenceLine
        transforms.append(ReferenceLine(
            "mean_predicted", "fraction_positive",
            x=(0.0, 1.0), y=(0.0, 1.0),
            name="calibration_ref",
        ))
        layers.append(_Layer(
            mark="line",
            encoding={"x": "mean_predicted", "y": "fraction_positive"},
            mark_kwargs={"stroke_dash": [4, 4]},
            data_source="calibration_ref",
        ))
    return ("__layered__", transforms, None, None, layers)


def desugar_gain(
    x_field: str | None,
    y_field: str | None,
    *,
    reference_lines: bool = True,
    color_field: str | None = "class",
) -> tuple:
    """Cumulative-gain mark.

    Data contract: ``percent_population``, ``gain``, ``class`` per
    ``ModelSource.cumulative_gain()``. The data already carries
    ``class='baseline'`` rows that render as the diagonal reference when
    ``color_field='class'``; ``reference_lines`` is informational.
    """
    del reference_lines  # baseline already in data
    line_enc: dict[str, Any] = {"x": "percent_population", "y": "gain"}
    if color_field is not None:
        line_enc["color"] = color_field
    return ("__layered__", [], None, None, [
        _Layer(mark="line", encoding=line_enc),
    ])


def desugar_lift(
    x_field: str | None,
    y_field: str | None,
    *,
    reference_line: bool = True,
    color_field: str | None = "class",
) -> tuple:
    """Lift curve mark.

    Data contract: ``percent_population``, ``lift``, ``class`` per
    ``ModelSource.lift_curve()``. The ``class='baseline'`` rows render as
    the lift=1 reference line when ``color_field='class'``;
    ``reference_line`` is informational.
    """
    del reference_line  # baseline already in data
    line_enc: dict[str, Any] = {"x": "percent_population", "y": "lift"}
    if color_field is not None:
        line_enc["color"] = color_field
    return ("__layered__", [], None, None, [
        _Layer(mark="line", encoding=line_enc),
    ])


def desugar_discrimination_threshold(
    x_field: str | None,
    y_field: str | None,
    *,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
    n_thresholds: int = 50,
    threshold_line: bool = False,
) -> tuple:
    """Discrimination-threshold sweep mark.

    Data contract (long form): ``threshold``, ``metric``, ``value`` — the
    figure builder is responsible for unpivoting
    ``ModelSource.discrimination_threshold()`` output into this shape.

    When ``threshold_line=True`` the chart builder injects a
    ``_threshold_best`` column with one non-null row at the F1-best
    threshold (argmax of the ``f1`` series in the un-melted source
    output). The desugar emits a vertical ``mark_rule`` layer on
    ``x=_threshold_best``; Rust's mark_rule renders one vertical span
    per non-null row, so exactly one rule appears.
    """
    del metrics, n_thresholds  # informational; data is pre-melted
    layers: list = [
        _Layer(mark="line",
               encoding={"x": "threshold", "y": "value", "color": "metric"}),
    ]
    if threshold_line:
        layers.append(_Layer(
            mark="rule",
            encoding={"x": "_threshold_best"},
            mark_kwargs={"stroke_dash": [4, 4], "opacity": 0.6},
        ))
    return ("__layered__", [], None, None, layers)


# --- 10c: classification matrices ------------------------------------


def desugar_confusion(
    x_field: str | None,
    y_field: str | None,
    *,
    normalize: str | None = None,
    annotate: bool = True,
    color_field: str = "value",
) -> tuple:
    """Confusion-matrix mark: ordinal heatmap + per-cell value labels.

    Data contract: ``actual``, ``predicted``, ``value``, ``value_fmt`` as
    emitted by ``ModelSource.confusion_matrix()``. The heatmap layer
    encodes ``color=value`` (continuous color scale, see Phase 10c-pre
    mark_rect fix); the optional text layer encodes ``text=value_fmt``
    via the Phase 10c-pre ``text`` channel.

    ``normalize`` is informational at the mark layer (the chart builder
    is responsible for shaping the data); the user-visible normalization
    happens upstream in ``ModelSource.confusion_matrix``.
    """
    del normalize, x_field, y_field
    layers: list = [
        _Layer(
            mark="rect",
            encoding={"x": "predicted", "y": "actual", "color": color_field},
        ),
    ]
    if annotate:
        layers.append(_Layer(
            mark="text",
            encoding={"x": "predicted", "y": "actual", "text": "value_fmt"},
        ))
    return ("__layered__", [], None, None, layers)


# --- 10d: feature importance / SHAP / PDP -----------------------------


def desugar_importance(
    x_field: str | None,
    y_field: str | None,
    *,
    orient: str = "horizontal",
    error_bars: bool = True,
    top_k: int | None = None,
    color_field: str | None = None,
    x_scale_domain: tuple[float, float] | list[float] | None = None,
) -> tuple:
    """Feature-importance mark: bars (per feature) + optional error bars.

    Data contract: ``feature`` (Utf8), ``importance`` (Float64), ``std``
    (Float64) as emitted by ``ModelSource.importances()``. The calling
    chart builder pre-computes ``imp_lower = importance - std`` and
    ``imp_upper = importance + std`` so the rule layer can reference them
    directly. ``top_k`` is informational at the mark layer — the chart
    builder truncates the DataFrame to the top-k rows before constructing
    the chart so the scale domain reflects only the visible rows.

    ``orient="horizontal"`` (default) renders horizontal bars with the
    Phase 10d-pre quantitative-x + ordinal-y bar path and horizontal
    ranged-rule error bars. ``orient="vertical"`` flips the axes (ordinal
    x, quantitative y), using the original boxplot-whisker-style ranged
    rule.

    ``x_scale_domain`` is supplied by the chart builder so the value axis
    starts at 0 (bar charts conventionally include zero) and extends
    slightly past the max error-bar upper bound; without it bars look
    truncated by the auto-derived [min, max] domain.
    """
    del x_field, y_field, top_k
    if orient not in ("horizontal", "vertical"):
        raise ValueError(
            f"mark_importance(orient={orient!r}) — expected 'horizontal' or 'vertical'."
        )

    if orient == "horizontal":
        value_axis, group_axis, err_axis2 = "x", "y", "x2"
    else:
        value_axis, group_axis, err_axis2 = "y", "x", "y2"
    value_field, group_field = "importance", "feature"
    err_lower, err_upper = "imp_lower", "imp_upper"

    def _value_channel(field: str) -> Any:
        if x_scale_domain is None:
            return field
        from ferrum.encoding import X, Y

        ch_cls = X if value_axis == "x" else Y
        return ch_cls(field, scale={"type": "linear", "domain": list(x_scale_domain)})

    bar_enc: dict[str, Any] = {
        value_axis: _value_channel(value_field),
        group_axis: group_field,
    }
    if color_field is not None:
        bar_enc["color"] = color_field
    layers: list = [_Layer(mark="bar", encoding=bar_enc)]

    if error_bars:
        err_enc: dict[str, Any] = {
            value_axis: _value_channel(err_lower),
            err_axis2: err_upper,
            group_axis: group_field,
        }
        layers.append(_Layer(mark="rule", encoding=err_enc))

    return ("__layered__", [], None, None, layers)


def desugar_shap_beeswarm(
    x_field: str | None,
    y_field: str | None,
    *,
    max_display: int = 20,
    color_bar: bool = True,
    order: str = "abs_mean",
    x_scale_domain: tuple[float, float] | list[float] | None = None,
) -> tuple:
    """SHAP beeswarm mark: categorical scatter of per-sample shap values.

    Data contract: ``feature`` (Utf8), ``shap_value`` (Float64),
    ``feature_value_normalized`` (Float64) as emitted by
    ``ModelSource.shap_values()`` and pre-filtered by the chart builder
    to the top ``max_display`` features.

    Renders one point per (sample, feature) cell with feature on the
    ordinal y-axis, shap_value on the quantitative x-axis, and the
    z-scored feature value on the continuous color scale. Vertical
    spread within each feature band uses the Phase 10d-pre Jitter
    ordinal-axis path; ``width=0.6`` keeps the band well-contained.

    ``color_bar`` and ``order`` are informational at the mark layer —
    the chart builder is responsible for any reordering / aggregation
    before constructing the chart.
    """
    del x_field, y_field, max_display, color_bar, order

    def _x_channel(field: str) -> Any:
        if x_scale_domain is None:
            return field
        from ferrum.encoding import X

        return X(field, scale={"type": "linear", "domain": list(x_scale_domain)})

    from ferrum.position import Jitter

    layers: list = [
        _Layer(
            mark="point",
            encoding={
                "x": _x_channel("shap_value"),
                "y": "feature",
                "color": "feature_value_normalized",
            },
            position=Jitter(axis="y", width=0.6, seed=42),
        ),
    ]
    return ("__layered__", [], None, None, layers)


def desugar_shap_bar(
    x_field: str | None,
    y_field: str | None,
    *,
    max_display: int = 20,
    x_scale_domain: tuple[float, float] | list[float] | None = None,
) -> tuple:
    """Aggregated-SHAP bar mark: mean(|shap_value|) per feature.

    Data contract: ``feature`` (Utf8), ``abs_mean_shap`` (Float64) — the
    chart builder aggregates ``ModelSource.shap_values()`` and selects
    the top ``max_display`` features.
    """
    del x_field, y_field, max_display

    def _x_channel(field: str) -> Any:
        if x_scale_domain is None:
            return field
        from ferrum.encoding import X

        return X(field, scale={"type": "linear", "domain": list(x_scale_domain)})

    return ("__layered__", [], None, None, [
        _Layer(
            mark="bar",
            encoding={"x": _x_channel("abs_mean_shap"), "y": "feature"},
        ),
    ])


def desugar_shap_waterfall(
    x_field: str | None,
    y_field: str | None,
    *,
    sample_idx: int = -1,
    max_display: int = 20,
    x_scale_domain: tuple[float, float] | list[float] | None = None,
) -> tuple:
    """SHAP waterfall mark: per-feature contribution segments for one sample.

    Data contract: ``feature`` (Utf8), ``x0`` (cumulative start),
    ``x1`` (cumulative end), ``shap_sign`` (Utf8: 'positive'|'negative')
    pre-computed by the chart builder. Renders a horizontal-ranged bar
    per feature via the Phase 10d-pre quantitative-x + x2 + ordinal-y
    bar path.
    """
    del x_field, y_field, max_display
    if sample_idx < 0:
        raise ValueError(
            "mark_shap_waterfall(sample_idx=...) is required. Pass an "
            "explicit non-negative sample index."
        )

    def _x_channel(field: str) -> Any:
        if x_scale_domain is None:
            return field
        from ferrum.encoding import X

        return X(field, scale={"type": "linear", "domain": list(x_scale_domain)})

    return ("__layered__", [], None, None, [
        _Layer(
            mark="bar",
            encoding={
                "x": _x_channel("x0"),
                "x2": "x1",
                "y": "feature",
                "color": "shap_sign",
            },
        ),
    ])


def desugar_pdp(
    x_field: str | None,
    y_field: str | None,
    *,
    kind: str = "average",
    ice_alpha: float = 0.2,
    center: bool = False,
    color_field: str | None = "feature",
) -> tuple:
    """Partial-dependence mark.

    Data contract (from ``ModelSource.partial_dependence``): ``feature``
    (Utf8), ``feature_value`` (Float64), ``pd_value`` (Float64),
    ``sample_id`` (Int64 — ``-1`` for the average row).

    ``kind="average"`` (default): single PD curve per feature (the
    underlying data has only ``sample_id=-1`` rows).

    ``kind="individual"``: per-sample ICE polylines via ``mark_style.detail``
    routing on a Utf8-cast ``sample_id`` column (one polyline per
    sample, no categorical color collision because the color encoding
    stays on ``feature``).

    ``kind="both"``: ICE polylines + a thicker average overlay. The chart
    builder injects an ``_pd_ice_value`` column that holds the per-sample
    value on ICE rows and ``None`` on the average row (mark_line skips
    null rows). The original ``pd_value`` column holds the average curve
    only.

    When ``center=True``, the chart builder pre-subtracts the value at
    the smallest ``feature_value`` per ``(feature, sample_id)`` group so
    every polyline starts at 0.
    """
    del x_field, y_field

    if kind == "average":
        # Single polyline per feature, color-coded by feature when faceted.
        line_enc: dict[str, Any] = {"x": "feature_value", "y": "pd_value"}
        if color_field is not None:
            line_enc["color"] = color_field
        return ("__layered__", [], None, None, [
            _Layer(mark="line", encoding=line_enc),
        ])

    from ferrum.encoding import Y as _Y

    if kind == "individual":
        # One polyline per sample via mark_style.detail. Color stays on
        # the feature (each facet gets its own hue). Opacity controls
        # the visual density of overlapping ICE lines.
        line_enc: dict[str, Any] = {
            "x": "feature_value",
            "y": _Y("pd_value", title="pd_value"),
        }
        if color_field is not None:
            line_enc["color"] = color_field
        return ("__layered__", [], None, None, [
            _Layer(
                mark="line",
                encoding=line_enc,
                mark_kwargs={
                    "detail": "_sample_id_str",
                    "opacity": float(ice_alpha),
                },
            ),
        ])

    if kind == "both":
        # ICE layer: y = _pd_ice_value (null on average row → skipped).
        # Override the y-axis title to 'pd_value' since the underlying
        # column name is a layer-internal artifact.
        ice_enc = {
            "x": "feature_value",
            "y": _Y("_pd_ice_value", title="pd_value"),
        }
        if color_field is not None:
            ice_enc["color"] = color_field
        avg_enc: dict[str, Any] = {
            "x": "feature_value", "y": "_pd_avg_value",
        }
        if color_field is not None:
            avg_enc["color"] = color_field
        return ("__layered__", [], None, None, [
            _Layer(
                mark="line",
                encoding=ice_enc,
                mark_kwargs={
                    "detail": "_sample_id_str",
                    "opacity": float(ice_alpha),
                },
            ),
            _Layer(
                mark="line",
                encoding=avg_enc,
                mark_kwargs={"stroke_width": 2.5},
            ),
        ])

    raise ValueError(
        f"mark_pdp(kind={kind!r}) — expected 'average', 'individual', or "
        "'both'."
    )


# --- 10e: model selection / CV curves --------------------------------


def _log_x_channel(field: str, log_scale: bool) -> Any:
    """Wrap ``field`` in an ``X`` channel with a log scale when requested,
    otherwise return the bare string field. Used by validation-curve and
    alpha-selection desugars whose x-axis is conventionally log-scaled.
    """
    if not log_scale:
        return field
    from ferrum.encoding import X

    return X(field, scale={"type": "log"})


def desugar_learning_curve(
    x_field: str | None,
    y_field: str | None,
    *,
    ci_style: str = "band",
    color_field: str | None = "split",
    **mark_kwargs: Any,
) -> tuple:
    """Learning-curve mark: per-split CI band/errorbar + mean line.

    Data contract: ``train_size`` (Int64), ``split`` (Utf8: "train"|"test"),
    ``mean_score``, ``lower``, ``upper`` (Float64) as emitted by
    ``ModelSource.learning_curve()`` and deduped per (train_size, split)
    by the chart builder. The CI columns are pre-computed in
    ``ModelSource.learning_curve``; mark_errorband would re-aggregate
    via ErrorExtent against a single column and is the wrong tool here
    (see plan-CORRECTIONS §7).

    ``ci_style="band"`` (default) renders a translucent ribbon between
    ``lower`` and ``upper``; ``ci_style="errorbar"`` renders a vertical
    rule per train_size from ``lower`` to ``upper``.

    The ribbon layer is layer-0 so the line draws on top; layer-0's y
    title therefore drives the chart's y-axis label — set to "score" so
    the axis does not read "lower" (the bottom-edge column name).
    """
    del x_field, y_field
    from ferrum.encoding import X, Y

    user_kw = _validate("learning_curve", mark_kwargs)
    y_axis = Y("lower", title="score")
    x_axis = X("train_size", title="training samples")
    if ci_style == "band":
        ci_layer = _Layer(
            mark="ribbon",
            encoding={
                "x": x_axis, "y": y_axis, "y2": "upper", "color": color_field,
            },
            mark_kwargs={"opacity": 0.3},
        )
    elif ci_style == "errorbar":
        ci_layer = _Layer(
            mark="rule",
            encoding={
                "x": x_axis, "y": y_axis, "y2": "upper", "color": color_field,
            },
        )
    else:
        raise ValueError(
            f"mark_learning_curve(ci_style={ci_style!r}) — expected "
            "'band' or 'errorbar'."
        )
    line_enc: dict[str, Any] = {
        "x": "train_size", "y": "mean_score", "color": color_field,
    }
    layers = [ci_layer, _Layer(mark="line", encoding=line_enc)]
    return ("__layered__", [], None, None,
            _apply(layers, user_kw))


def desugar_validation_curve(
    x_field: str | None,
    y_field: str | None,
    *,
    log_scale: bool = False,
    ci_style: str = "band",
    color_field: str | None = "split",
    param_label: str | None = None,
    **mark_kwargs: Any,
) -> tuple:
    """Validation-curve mark: same shape as learning_curve over a
    hyperparameter sweep.

    Data contract: ``param_value`` (Float64), ``split``, ``mean_score``,
    ``lower``, ``upper`` — deduped per (param_value, split) by the
    builder. When ``log_scale=True`` the x channel uses a log scale,
    appropriate when ``values`` span more than two orders of magnitude.

    ``param_label`` (passed by the chart builder) names the swept
    hyperparameter so the x-axis reads e.g. "alpha" rather than the
    generic ``param_value`` column name.
    """
    del x_field, y_field
    from ferrum.encoding import X, Y

    user_kw = _validate("validation_curve", mark_kwargs)
    scale = {"type": "log"} if log_scale else None
    x_kwargs: dict[str, Any] = {}
    if param_label is not None:
        x_kwargs["title"] = param_label
    if scale is not None:
        x_kwargs["scale"] = scale
    x_ch = X("param_value", **x_kwargs) if x_kwargs else "param_value"
    y_axis = Y("lower", title="score")
    if ci_style == "band":
        ci_layer = _Layer(
            mark="ribbon",
            encoding={
                "x": x_ch, "y": y_axis, "y2": "upper", "color": color_field,
            },
            mark_kwargs={"opacity": 0.3},
        )
    elif ci_style == "errorbar":
        ci_layer = _Layer(
            mark="rule",
            encoding={
                "x": x_ch, "y": y_axis, "y2": "upper", "color": color_field,
            },
        )
    else:
        raise ValueError(
            f"mark_validation_curve(ci_style={ci_style!r}) — expected "
            "'band' or 'errorbar'."
        )
    layers = [
        ci_layer,
        _Layer(mark="line",
               encoding={"x": "param_value", "y": "mean_score", "color": color_field}),
    ]
    return ("__layered__", [], None, None,
            _apply(layers, user_kw))


def desugar_cv_scores(
    x_field: str | None,
    y_field: str | None,
    *,
    kind: str = "box",
    split: str = "both",
    **mark_kwargs: Any,
) -> tuple:
    """Per-fold CV score distribution.

    Data contract: ``fold`` (Int64), ``split`` (Utf8), ``score`` (Float64)
    as emitted by ``ModelSource.cv_scores()``. For ``kind="box"`` the
    builder leaves raw per-fold rows (BoxStats groups by split); for
    ``kind="bar"`` the builder pre-aggregates to mean score per split;
    for ``kind="strip"`` the builder leaves raw rows and Jitter spreads
    them along the categorical axis.
    """
    del x_field, y_field, split
    from ferrum.encoding import Y

    user_kw = _validate("cv_scores", mark_kwargs)
    if kind == "box":
        from ferrum.marks.composite import desugar_boxplot

        prefix, transforms, _ig1, _ig2, layers = desugar_boxplot("split", "score")
        # Override layer-0's y field-name (BoxStats output column) with a
        # titled channel so the chart's y-axis reads "score" rather than
        # the internal column name "lower_whisker".
        if layers:
            first = layers[0]
            enc = dict(first.encoding)
            y_val = enc.get("y")
            if isinstance(y_val, str):
                enc["y"] = Y(y_val, title="score")
                layers = [replace(first, encoding=enc), *layers[1:]]
        return (prefix, transforms, _ig1, _ig2,
                _apply(layers, user_kw))
    if kind == "bar":
        layers = [
            _Layer(mark="bar",
                   encoding={"x": "split", "y": Y("score", title="score"),
                             "color": "split"}),
        ]
        return ("__layered__", [], None, None,
                _apply(layers, user_kw))
    if kind == "strip":
        from ferrum.position import Jitter

        layers = [
            _Layer(mark="point",
                   encoding={"x": "split", "y": Y("score", title="score"),
                             "color": "split"},
                   position=Jitter(axis="x", width=0.3, seed=42)),
        ]
        return ("__layered__", [], None, None,
                _apply(layers, user_kw))
    raise ValueError(
        f"mark_cv_scores(kind={kind!r}) — expected 'box', 'bar', or 'strip'."
    )


def desugar_alpha_selection(
    x_field: str | None,
    y_field: str | None,
    *,
    log_scale: bool = True,
    highlight_best: bool = True,
    ci_style: str = "band",
    **mark_kwargs: Any,
) -> tuple:
    """Regularization-strength selection mark: mean-score line + best-alpha rule.

    Data contract: ``alpha`` (Float64), ``mean_score`` (Float64) — the
    builder dedupes per alpha and (when ``highlight_best=True``) the
    ``Chart.mark_alpha_selection`` method injects a ``_best_alpha``
    sentinel column with one non-null row at ``argmax(mean_score)`` for
    a single vertical rule.

    ``ci_style`` is informational at the mark layer — alpha_selection
    renders a single curve without CI bands; the multi-fold spread
    surfaces in the companion ``cv_scores_chart``.
    """
    del x_field, y_field, ci_style
    from ferrum.encoding import Y

    user_kw = _validate("alpha_selection", mark_kwargs)
    x_ch = _log_x_channel("alpha", log_scale)
    layers: list = [
        _Layer(mark="line",
               encoding={"x": x_ch, "y": Y("mean_score", title="score")}),
    ]
    if highlight_best:
        layers.append(_Layer(
            mark="rule",
            encoding={"x": "_best_alpha"},
            mark_kwargs={"stroke_dash": [4, 4]},
        ))
    return ("__layered__", [], None, None,
            _apply(layers, user_kw))


def desugar_class_prediction_error(
    x_field: str | None,
    y_field: str | None,
    *,
    normalize: bool = False,
    color_field: str = "actual",
) -> tuple:
    """Stacked-bar diagnostic of predicted-class composition.

    Data contract: ``actual``, ``predicted``, ``value`` (same shape as
    ``ModelSource.confusion_matrix(normalize=None)``). One bar per
    ``predicted`` value, segments colored by ``actual``. ``normalize=True``
    switches to a per-bar 100% stack via the Stack position adjustment.
    """
    del x_field, y_field
    from ferrum.position import Stack

    stack = Stack(by=color_field, offset="normalize" if normalize else "zero")
    return ("__layered__", [], None, None, [
        _Layer(
            mark="bar",
            encoding={"x": "predicted", "y": "value", "color": color_field},
            position=stack,
        ),
    ])


# --- 10f: clustering / manifold / decision boundary -------------------


def desugar_silhouette(
    x_field: str | None,
    y_field: str | None,
    *,
    zero_line: bool = True,
    color_field: str | None = "cluster",
    **mark_kwargs: Any,
) -> tuple:
    """Rousseeuw silhouette plot: one horizontal bar per sample.

    Data contract: ``y_position`` (Int64 stack order, 0..n-1 — packed by
    ``ModelSource.silhouette()``), ``silhouette_value`` (Float64),
    ``cluster`` (Int64), plus the per-bar bound columns
    ``_silhouette_x_lo``, ``_silhouette_x_hi``, ``_silhouette_y_lo``,
    ``_silhouette_y_hi`` (Float64), and (when ``zero_line=True``)
    ``_ref_zero``. The bound columns are computed by
    ``Chart.mark_silhouette``: mark_bar has no quantitative-x /
    quantitative-y rendering path, so mark_rect with explicit cell
    bounds is the native primitive for the Rousseeuw layout.
    """
    del x_field, y_field
    from ferrum.encoding import X, Y

    user_kw = _validate("silhouette", mark_kwargs)
    rect_enc: dict[str, Any] = {
        "x": X("_silhouette_x_lo", title="silhouette coefficient"),
        "x2": "_silhouette_x_hi",
        "y": Y("_silhouette_y_lo", title="sample"),
        "y2": "_silhouette_y_hi",
    }
    if color_field is not None:
        rect_enc["color"] = color_field
    layers: list = [_Layer(mark="rect", encoding=rect_enc)]
    if zero_line:
        layers.append(_Layer(
            mark="rule",
            encoding={"x": "_ref_zero"},
            mark_kwargs={"stroke_dash": [4, 4]},
        ))
    return ("__layered__", [], None, None,
            _apply(layers, user_kw))


def desugar_pca_scree(
    x_field: str | None,
    y_field: str | None,
    *,
    cumulative_line: bool = True,
    threshold_line: float | None = None,
    **mark_kwargs: Any,
) -> tuple:
    """PCA scree plot: bar of per-component variance + optional cumulative
    line and threshold rule.

    Data contract: ``component`` (Int64), ``explained_variance_ratio``
    (Float64), ``cumulative_variance_ratio`` (Float64) as emitted by
    ``ModelSource.pca_variance()`` plus the injected bar-bound columns
    ``_pca_bar_x_lo``, ``_pca_bar_x_hi``, ``_pca_bar_y_lo``,
    ``_pca_bar_y_hi`` (Float64) computed by ``Chart.mark_pca_scree``.
    mark_bar has no quant-x / quant-y path; mark_rect with explicit
    bounds renders the bars while preserving quantitative axes for the
    cumulative-line and threshold-rule layers.

    When ``threshold_line`` is non-None the data also carries
    ``_threshold_line`` (a sentinel single-non-null column for the
    horizontal reference rule).
    """
    del x_field, y_field, threshold_line
    from ferrum.encoding import X, Y

    user_kw = _validate("pca_scree", mark_kwargs)
    # Layer-0 drives axis-scale resolution (see render/prepare.rs:265 —
    # only the first layer's encoding feeds resolve_scales). The
    # cumulative line spans the widest y range, so emit it first so the
    # y axis covers [0, max(cum)] rather than [0, max(evr)]. The rect
    # bar layer follows and renders within the established axis.
    if cumulative_line:
        layers: list = [
            _Layer(
                mark="line",
                encoding={
                    "x": X("component", title="component"),
                    "x2": "_x_axis_anchor",
                    "y": Y("cumulative_variance_ratio",
                           title="explained variance ratio"),
                    # x2/y2 here are scale-resolution hints — mark_line
                    # ignores both when drawing, but
                    # scale_resolve::build_axis_scale unions the paired
                    # channel's extent into the axis domain. The anchor
                    # columns hold [bar_x_lo_min, bar_x_hi_max] and
                    # [0, max(cum, threshold)] so the axes cover the
                    # bar baselines, the first/last bar edges, and any
                    # threshold rule introduced by sibling layers.
                    "y2": "_y_axis_anchor",
                },
            ),
            _Layer(
                mark="rect",
                encoding={
                    "x": "_pca_bar_x_lo",
                    "x2": "_pca_bar_x_hi",
                    "y": "_pca_bar_y_lo",
                    "y2": "_pca_bar_y_hi",
                },
            ),
        ]
    else:
        # No cumulative line — the rect bar is the only y signal and
        # leads the scale-resolution.
        layers = [
            _Layer(
                mark="rect",
                encoding={
                    "x": X("_pca_bar_x_lo", title="component"),
                    "x2": "_pca_bar_x_hi",
                    "y": Y("_pca_bar_y_lo",
                           title="explained variance ratio"),
                    "y2": "_pca_bar_y_hi",
                },
            ),
        ]
    return ("__layered__", [], None, None,
            _apply(layers, user_kw))


def desugar_pca_scree_with_threshold(
    x_field: str | None,
    y_field: str | None,
    *,
    cumulative_line: bool = True,
    **mark_kwargs: Any,
) -> tuple:
    """Variant of ``desugar_pca_scree`` that appends a threshold rule.

    Used by ``Chart.mark_pca_scree`` when ``threshold_line`` is non-None;
    references the injected ``_threshold_line`` sentinel column.
    """
    # Validate kwargs here so the AST guardrail (test_mark_kwargs_no_silent_drop)
    # sees a call to validate_user_mark_kwargs at this function level. The
    # nested desugar_pca_scree call validates the same set independently, so
    # the second validation is a no-op for well-formed inputs.
    _validate("pca_scree", mark_kwargs)
    prefix, transforms, _ig1, _ig2, layers = desugar_pca_scree(
        x_field, y_field, cumulative_line=cumulative_line, **mark_kwargs,
    )
    layers = list(layers) + [_Layer(
        mark="rule",
        encoding={"y": "_threshold_line"},
        mark_kwargs={"stroke_dash": [4, 4]},
    )]
    return (prefix, transforms, _ig1, _ig2, layers)


def desugar_intercluster_distance(
    x_field: str | None,
    y_field: str | None,
    *,
    label_clusters: bool = True,
    color_field: str | None = "cluster",
    min_size: float = 60.0,
    max_size: float = 600.0,
    **mark_kwargs: Any,
) -> tuple:
    """Cluster-center 2D embedding: one point per cluster, sized by count.

    Data contract: ``cluster`` (Utf8), ``x`` (Float64), ``y`` (Float64),
    ``size`` (Int64). When ``label_clusters=True`` a text layer overlays
    the cluster id at each point. ``min_size`` and ``max_size`` set the
    point-area range (in pixel² units, before sqrt → radius); the
    defaults ``min_size=60.0`` / ``max_size=600.0`` produce ~8–24 px radii,
    well above the theme default ``[3, 30]`` that collapses to a 1–3 px
    speck on typical KMeans cluster counts.
    """
    del x_field, y_field
    from ferrum.encoding import Size

    user_kw = _validate("intercluster_distance", mark_kwargs)
    # Pass scale as a dict so we can specify range without needing a
    # domain (the renderer infers domain from the data). The `_core`
    # LinearScale constructor requires domain explicitly.
    size_channel = Size(
        "size",
        scale={"type": "linear", "range": [float(min_size), float(max_size)]},
    )
    point_enc: dict[str, Any] = {"x": "x", "y": "y", "size": size_channel}
    if color_field is not None:
        point_enc["color"] = color_field
    layers: list = [_Layer(mark="point", encoding=point_enc)]
    if label_clusters:
        layers.append(_Layer(
            mark="text",
            encoding={"x": "x", "y": "y", "text": "cluster"},
        ))
    return ("__layered__", [], None, None,
            _apply(layers, user_kw))


def desugar_rank1d(
    x_field: str | None,
    y_field: str | None,
    *,
    orient: str = "horizontal",
    color_field: str | None = None,
    **mark_kwargs: Any,
) -> tuple:
    """Univariate feature ranking — one bar per feature, sized by score.

    Data contract: ``feature`` (Utf8), ``score`` (Float64), ``rank``
    (Int64) — the schema emitted by ``ModelSource.rank1d()``. Bars use
    the renderer's ordinal-x / quant-y or ordinal-y / quant-x ``mark_bar``
    path (no per-row bound injection required — the existing horizontal /
    vertical bar dispatch handles both orientations natively).
    """
    del x_field, y_field

    if orient not in ("horizontal", "vertical"):
        raise ValueError(
            f"mark_rank1d(orient={orient!r}) — expected 'horizontal' or 'vertical'."
        )
    user_kw = _validate("rank1d", mark_kwargs)
    if orient == "horizontal":
        enc: dict[str, Any] = {"x": "score", "y": "feature"}
    else:
        enc = {"x": "feature", "y": "score"}
    if color_field is not None:
        enc["color"] = color_field
    layers: list = [_Layer(mark="bar", encoding=enc)]
    return ("__layered__", [], None, None,
            _apply(layers, user_kw))


def desugar_rank2d(
    x_field: str | None,
    y_field: str | None,
    *,
    annot: bool = True,
    color_field: str = "correlation",
    text_field: str = "correlation_fmt",
    **mark_kwargs: Any,
) -> tuple:
    """Pairwise feature ranking — long-form correlation matrix heatmap.

    Data contract: ``feature_x`` (Utf8), ``feature_y`` (Utf8),
    ``correlation`` (Float64) — schema from ``ModelSource.rank2d()``.
    When ``annot=True``, the data must also carry ``correlation_fmt``
    (Utf8 — preformatted to 2 decimal places by the chart builder so
    the renderer can lay out short labels without invoking
    Rust-side number formatting per cell).
    """
    del x_field, y_field

    user_kw = _validate("rank2d", mark_kwargs)
    rect_enc: dict[str, Any] = {
        "x": "feature_x", "y": "feature_y", "color": color_field,
    }
    layers: list = [_Layer(mark="rect", encoding=rect_enc)]
    if annot:
        layers.append(_Layer(
            mark="text",
            encoding={
                "x": "feature_x", "y": "feature_y", "text": text_field,
            },
        ))
    return ("__layered__", [], None, None,
            _apply(layers, user_kw))


def desugar_parallel_coordinates(
    x_field: str | None,
    y_field: str | None,
    *,
    alpha: float = 0.5,
    color_field: str | None = None,
    **mark_kwargs: Any,
) -> tuple:
    """Parallel coordinates — one polyline per sample, x = feature, y = value.

    Data contract: ``feature`` (Utf8 — ordinal x axis), ``value``
    (Float64), ``sample_id`` (Utf8) and, when ``color_field`` is set,
    a hue column (Utf8 categorical). ``alpha`` becomes the line layer's
    ``opacity`` mark kwarg.

    ``mark_kwargs.detail`` routes through ``MarkKwargsSpec.detail`` on
    the line layer; mark_line then groups by (color, detail) producing
    one polyline per (hue, sample) pair (or one per sample when
    ``color_field`` is None). Requires the line renderer's ordinal-x
    + detail support added in this sub-batch.
    """
    del x_field, y_field

    user_kw = _validate("parallel_coordinates", mark_kwargs)
    line_enc: dict[str, Any] = {"x": "feature", "y": "value"}
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list = [_Layer(
        mark="line",
        encoding=line_enc,
        mark_kwargs={
            "detail": "sample_id",
            "opacity": float(alpha),
        },
    )]
    return ("__layered__", [], None, None,
            _apply(layers, user_kw))


def desugar_decision_boundary(
    x_field: str | None,
    y_field: str | None,
    *,
    proba: bool = False,
    color_field: str = "z",
    **mark_kwargs: Any,
) -> tuple:
    """Decision-boundary background heatmap: one rect per grid cell.

    Data contract: pre-computed grid with columns ``x``, ``x2``, ``y``,
    ``y2`` (cell bounds) and ``z`` (the prediction value — class index
    when ``proba=False``, probability when ``proba=True``). The chart
    builder produces these columns from a ``ModelSource``.

    ``proba`` is informational at the mark layer — the chart builder
    chooses the data and the renderer's continuous-color scale handles
    both kinds of ``z`` identically. Recorded for future overrides.
    """
    del x_field, y_field, proba

    user_kw = _validate("decision_boundary", mark_kwargs)
    layers: list = [
        _Layer(
            mark="rect",
            encoding={
                "x": "x", "x2": "x2", "y": "y", "y2": "y2",
                "color": color_field,
            },
            mark_kwargs={"opacity": 0.5},
        ),
    ]
    return ("__layered__", [], None, None,
            _apply(layers, user_kw))
