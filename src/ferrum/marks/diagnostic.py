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

from typing import Any


def desugar_residuals(
    x_field: str | None,
    y_field: str | None,
    *,
    kind: str = "studentized",
    reference_line: bool = True,
    cook_threshold: float | None = None,
    color_field: str | None = None,
    **mark_kwargs: Any,
) -> tuple:
    """Residuals diagnostic: scatter of (y_pred, residual) plus optional y=0 rule.

    Data contract: the chart's data must carry columns ``y_pred`` and either
    ``residual`` (kind="raw") or ``studentized_residual`` (kind in
    "studentized"/"scaled"). When ``reference_line=True`` the data must also
    carry the injected ``_ref_zero`` column (the ``Chart.mark_residuals``
    method takes care of this).

    ``cook_threshold`` is reserved for the multi-panel residuals_chart in
    Phase 10h; passing a non-default value raises ``NotImplementedError``
    rather than silently ignoring it (per the Phase 9+ no-defer principle).
    """
    if cook_threshold is not None:
        raise NotImplementedError(
            "mark_residuals(cook_threshold=...) lands in Phase 10h alongside "
            "the leverage-aware Cook's D path."
        )
    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"
    point_enc: dict[str, Any] = {"x": "y_pred", "y": y_col}
    if color_field is not None:
        point_enc["color"] = color_field
    layers: list[dict] = [{"mark": "point", "encoding": point_enc}]
    if reference_line:
        layers.append({
            "mark": "rule",
            "encoding": {"y": "_ref_zero"},
            "mark_kwargs": {"stroke_dash": [4, 4]},
        })
    return ("__layered__", [], None, None, layers)


def desugar_prediction_error(
    x_field: str | None,
    y_field: str | None,
    *,
    identity_line: bool = True,
    ci: float | None = None,
    reference_band: bool = False,
    color_field: str | None = None,
    **mark_kwargs: Any,
) -> tuple:
    """Actual vs predicted: scatter of (y_true, y_pred) + optional identity line.

    Data contract: columns ``y_true`` and ``y_pred``. When
    ``identity_line=True`` the data must be sorted ascending by ``y_true`` so
    the line layer renders as a clean y=x diagonal (handled by
    ``Chart.mark_prediction_error``).

    ``ci`` and ``reference_band`` are reserved for Phase 10h; passing
    non-default values raises ``NotImplementedError`` (per the Phase 9+
    no-defer principle).
    """
    if ci is not None:
        raise NotImplementedError(
            "mark_prediction_error(ci=...) lands in Phase 10h."
        )
    if reference_band:
        raise NotImplementedError(
            "mark_prediction_error(reference_band=True) lands in Phase 10h."
        )
    point_enc: dict[str, Any] = {"x": "y_true", "y": "y_pred"}
    if color_field is not None:
        point_enc["color"] = color_field
    layers: list[dict] = [{"mark": "point", "encoding": point_enc}]
    if identity_line:
        layers.append({
            "mark": "line",
            "encoding": {"x": "y_true", "y": "y_true"},
            "mark_kwargs": {"stroke_dash": [4, 4]},
        })
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
    **mark_kwargs: Any,
) -> tuple:
    """ROC curve mark.

    Data contract: columns ``fpr``, ``tpr``, and (typically) ``class`` and
    ``auc`` as emitted by ``ModelSource.roc_curve()``. When
    ``reference_line=True`` the calling ``Chart.mark_roc`` method pre-sorts
    the data ascending by ``fpr`` so the diagonal line layer is monotonic.

    ``annotate_auc`` is reserved for Phase 10h (text annotation); passing
    ``True`` raises ``NotImplementedError``. ``average`` is informational
    at the mark layer — the figure builder is responsible for shaping the
    data appropriately before constructing the chart.
    """
    del average  # informational at the mark layer
    if annotate_auc:
        raise NotImplementedError(
            "mark_roc(annotate_auc=True) lands in Phase 10h alongside text "
            "annotations on diagnostic curves."
        )
    line_enc: dict[str, Any] = {"x": "fpr", "y": "tpr"}
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list[dict] = [{"mark": "line", "encoding": line_enc}]
    if reference_line:
        layers.append({
            "mark": "line",
            "encoding": {"x": "fpr", "y": "fpr"},
            "mark_kwargs": {"stroke_dash": [4, 4]},
        })
    return ("__layered__", [], None, None, layers)


def desugar_pr(
    x_field: str | None,
    y_field: str | None,
    *,
    average: str | None = None,
    annotate_ap: bool = False,
    iso_lines: bool = False,
    color_field: str | None = "class",
    **mark_kwargs: Any,
) -> tuple:
    """Precision-recall curve mark.

    Data contract: ``recall``, ``precision``, ``class``, ``ap`` as emitted
    by ``ModelSource.pr_curve()``. ``annotate_ap`` and ``iso_lines`` are
    reserved for Phase 10h; passing non-default values raises
    ``NotImplementedError``.
    """
    del average  # informational at the mark layer
    if annotate_ap:
        raise NotImplementedError(
            "mark_pr(annotate_ap=True) lands in Phase 10h."
        )
    if iso_lines:
        raise NotImplementedError(
            "mark_pr(iso_lines=True) lands in Phase 10h."
        )
    line_enc: dict[str, Any] = {"x": "recall", "y": "precision"}
    if color_field is not None:
        line_enc["color"] = color_field
    return ("__layered__", [], None, None, [
        {"mark": "line", "encoding": line_enc},
    ])


def desugar_calibration(
    x_field: str | None,
    y_field: str | None,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    reference_line: bool = True,
    color_field: str | None = None,
    **mark_kwargs: Any,
) -> tuple:
    """Calibration (reliability) curve mark.

    Data contract: ``mean_predicted``, ``fraction_positive``, ``count`` as
    emitted by ``ModelSource.calibration_curve()``. When
    ``reference_line=True`` the calling ``Chart.mark_calibration`` method
    pre-sorts data ascending by ``mean_predicted`` so the y=x line is
    monotonic. ``n_bins``/``strategy`` are informational at the mark layer
    (the data is already binned).
    """
    del n_bins, strategy
    line_enc: dict[str, Any] = {
        "x": "mean_predicted", "y": "fraction_positive",
    }
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list[dict] = [{"mark": "line", "encoding": line_enc}]
    if reference_line:
        layers.append({
            "mark": "line",
            "encoding": {"x": "mean_predicted", "y": "mean_predicted"},
            "mark_kwargs": {"stroke_dash": [4, 4]},
        })
    return ("__layered__", [], None, None, layers)


def desugar_gain(
    x_field: str | None,
    y_field: str | None,
    *,
    reference_lines: bool = True,
    color_field: str | None = "class",
    **mark_kwargs: Any,
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
        {"mark": "line", "encoding": line_enc},
    ])


def desugar_lift(
    x_field: str | None,
    y_field: str | None,
    *,
    reference_line: bool = True,
    color_field: str | None = "class",
    **mark_kwargs: Any,
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
        {"mark": "line", "encoding": line_enc},
    ])


def desugar_discrimination_threshold(
    x_field: str | None,
    y_field: str | None,
    *,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
    n_thresholds: int = 50,
    threshold_line: bool = False,
    **mark_kwargs: Any,
) -> tuple:
    """Discrimination-threshold sweep mark.

    Data contract (long form): ``threshold``, ``metric``, ``value`` — the
    figure builder is responsible for unpivoting
    ``ModelSource.discrimination_threshold()`` output into this shape.
    ``threshold_line`` is reserved for Phase 10h (rule at the F1-best
    threshold); passing ``True`` raises ``NotImplementedError``.
    """
    if threshold_line:
        raise NotImplementedError(
            "mark_discrimination_threshold(threshold_line=True) lands in "
            "Phase 10h alongside rule-annotation support."
        )
    del metrics, n_thresholds  # informational; data is pre-melted
    return ("__layered__", [], None, None, [
        {"mark": "line",
         "encoding": {"x": "threshold", "y": "value", "color": "metric"}},
    ])


# --- 10c: classification matrices ------------------------------------


def desugar_confusion(
    x_field: str | None,
    y_field: str | None,
    *,
    normalize: str | None = None,
    annotate: bool = True,
    color_field: str = "value",
    **mark_kwargs: Any,
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
    layers: list[dict] = [
        {
            "mark": "rect",
            "encoding": {"x": "predicted", "y": "actual", "color": color_field},
        },
    ]
    if annotate:
        layers.append({
            "mark": "text",
            "encoding": {"x": "predicted", "y": "actual", "text": "value_fmt"},
        })
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
    **mark_kwargs: Any,
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
    layers: list[dict] = [{"mark": "bar", "encoding": bar_enc}]

    if error_bars:
        err_enc: dict[str, Any] = {
            value_axis: _value_channel(err_lower),
            err_axis2: err_upper,
            group_axis: group_field,
        }
        layers.append({"mark": "rule", "encoding": err_enc})

    return ("__layered__", [], None, None, layers)


def desugar_shap_beeswarm(
    x_field: str | None,
    y_field: str | None,
    *,
    max_display: int = 20,
    color_bar: bool = True,
    order: str = "abs_mean",
    x_scale_domain: tuple[float, float] | list[float] | None = None,
    **mark_kwargs: Any,
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

    layers: list[dict] = [
        {
            "mark": "point",
            "encoding": {
                "x": _x_channel("shap_value"),
                "y": "feature",
                "color": "feature_value_normalized",
            },
            "position": Jitter(axis="y", width=0.6, seed=42),
        },
    ]
    return ("__layered__", [], None, None, layers)


def desugar_shap_bar(
    x_field: str | None,
    y_field: str | None,
    *,
    max_display: int = 20,
    x_scale_domain: tuple[float, float] | list[float] | None = None,
    **mark_kwargs: Any,
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
        {
            "mark": "bar",
            "encoding": {"x": _x_channel("abs_mean_shap"), "y": "feature"},
        },
    ])


def desugar_shap_waterfall(
    x_field: str | None,
    y_field: str | None,
    *,
    sample_idx: int = -1,
    max_display: int = 20,
    x_scale_domain: tuple[float, float] | list[float] | None = None,
    **mark_kwargs: Any,
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
        raise TypeError(
            "mark_shap_waterfall(sample_idx=...) is required. Pass an "
            "explicit non-negative sample index."
        )

    def _x_channel(field: str) -> Any:
        if x_scale_domain is None:
            return field
        from ferrum.encoding import X

        return X(field, scale={"type": "linear", "domain": list(x_scale_domain)})

    return ("__layered__", [], None, None, [
        {
            "mark": "bar",
            "encoding": {
                "x": _x_channel("x0"),
                "x2": "x1",
                "y": "feature",
                "color": "shap_sign",
            },
        },
    ])


def desugar_pdp(
    x_field: str | None,
    y_field: str | None,
    *,
    kind: str = "average",
    ice_alpha: float = 0.2,
    center: bool = False,
    color_field: str | None = "feature",
    **mark_kwargs: Any,
) -> tuple:
    """Partial-dependence mark: one polyline per feature.

    Data contract: ``feature`` (Utf8), ``feature_value`` (Float64),
    ``pd_value`` (Float64) as emitted by
    ``ModelSource.partial_dependence()``. The chart builder is
    responsible for sorting per feature so the line layer renders as a
    monotonic curve in ``feature_value``.

    ``kind="individual"``/``"both"`` and ``center=True`` are reserved for
    a later sub-batch that adds the ``detail`` encoding channel (needed
    for per-sample ICE polylines without categorical color collisions);
    passing non-default values raises ``NotImplementedError``.
    """
    del x_field, y_field, ice_alpha
    if kind != "average":
        raise NotImplementedError(
            f"mark_pdp(kind={kind!r}) requires the 'detail' encoding channel "
            "for per-sample ICE polylines (Phase 9-deferred). Use "
            "kind='average' for now."
        )
    if center:
        raise NotImplementedError(
            "mark_pdp(center=True) lands alongside ICE support (Phase 9+)."
        )

    line_enc: dict[str, Any] = {"x": "feature_value", "y": "pd_value"}
    if color_field is not None:
        line_enc["color"] = color_field
    return ("__layered__", [], None, None, [
        {"mark": "line", "encoding": line_enc},
    ])


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

    y_axis = Y("lower", title="score")
    x_axis = X("train_size", title="training samples")
    if ci_style == "band":
        ci_layer = {
            "mark": "ribbon",
            "encoding": {
                "x": x_axis, "y": y_axis, "y2": "upper", "color": color_field,
            },
            "mark_kwargs": {"opacity": 0.3},
        }
    elif ci_style == "errorbar":
        ci_layer = {
            "mark": "rule",
            "encoding": {
                "x": x_axis, "y": y_axis, "y2": "upper", "color": color_field,
            },
        }
    else:
        raise ValueError(
            f"mark_learning_curve(ci_style={ci_style!r}) — expected "
            "'band' or 'errorbar'."
        )
    line_enc: dict[str, Any] = {
        "x": "train_size", "y": "mean_score", "color": color_field,
    }
    return ("__layered__", [], None, None, [
        ci_layer,
        {"mark": "line", "encoding": line_enc},
    ])


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

    scale = {"type": "log"} if log_scale else None
    x_kwargs: dict[str, Any] = {}
    if param_label is not None:
        x_kwargs["title"] = param_label
    if scale is not None:
        x_kwargs["scale"] = scale
    x_ch = X("param_value", **x_kwargs) if x_kwargs else "param_value"
    y_axis = Y("lower", title="score")
    if ci_style == "band":
        ci_layer = {
            "mark": "ribbon",
            "encoding": {
                "x": x_ch, "y": y_axis, "y2": "upper", "color": color_field,
            },
            "mark_kwargs": {"opacity": 0.3},
        }
    elif ci_style == "errorbar":
        ci_layer = {
            "mark": "rule",
            "encoding": {
                "x": x_ch, "y": y_axis, "y2": "upper", "color": color_field,
            },
        }
    else:
        raise ValueError(
            f"mark_validation_curve(ci_style={ci_style!r}) — expected "
            "'band' or 'errorbar'."
        )
    return ("__layered__", [], None, None, [
        ci_layer,
        {"mark": "line",
         "encoding": {"x": "param_value", "y": "mean_score", "color": color_field}},
    ])


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

    if kind == "box":
        from ferrum.marks.composite import desugar_boxplot

        prefix, transforms, _ig1, _ig2, layers = desugar_boxplot("split", "score")
        # Override layer-0's y field-name (BoxStats output column) with a
        # titled channel so the chart's y-axis reads "score" rather than
        # the internal column name "lower_whisker".
        if layers:
            first = dict(layers[0])
            enc = dict(first.get("encoding", {}))
            y_val = enc.get("y")
            if isinstance(y_val, str):
                enc["y"] = Y(y_val, title="score")
                first["encoding"] = enc
                layers = [first, *layers[1:]]
        return (prefix, transforms, _ig1, _ig2, layers)
    if kind == "bar":
        return ("__layered__", [], None, None, [
            {"mark": "bar",
             "encoding": {"x": "split", "y": Y("score", title="score"),
                          "color": "split"}},
        ])
    if kind == "strip":
        from ferrum.position import Jitter

        return ("__layered__", [], None, None, [
            {"mark": "point",
             "encoding": {"x": "split", "y": Y("score", title="score"),
                          "color": "split"},
             "position": Jitter(axis="x", width=0.3, seed=42)},
        ])
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

    x_ch = _log_x_channel("alpha", log_scale)
    layers: list[dict] = [
        {"mark": "line",
         "encoding": {"x": x_ch, "y": Y("mean_score", title="score")}},
    ]
    if highlight_best:
        layers.append({
            "mark": "rule",
            "encoding": {"x": "_best_alpha"},
            "mark_kwargs": {"stroke_dash": [4, 4]},
        })
    return ("__layered__", [], None, None, layers)


def desugar_class_prediction_error(
    x_field: str | None,
    y_field: str | None,
    *,
    normalize: bool = False,
    color_field: str = "actual",
    **mark_kwargs: Any,
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
        {
            "mark": "bar",
            "encoding": {"x": "predicted", "y": "value", "color": color_field},
            "position": stack,
        },
    ])
