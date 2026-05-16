"""Model-diagnostic mark desugars — classification domain."""

from __future__ import annotations

from typing import Any

from ferrum._layer import MarkDesugarResult, _Layer
from ferrum._overrides import register_layer_names
from ferrum.marks._mark_kwargs import (
    apply_user_mark_kwargs as _apply,
    validate_user_mark_kwargs as _validate,
)

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
    _ = average  # informational at the mark layer
    user_kw = _validate("roc", mark_kwargs)
    line_enc: dict[str, Any] = {"x": "fpr", "y": "tpr"}
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list = [_Layer(name="line", mark="line", encoding=line_enc)]
    if reference_line:
        layers.append(
            _Layer(
                name="reference",
                mark="line",
                encoding={"x": "fpr", "y": "fpr"},
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
            )
        )
    if annotate_auc:
        text_enc: dict[str, Any] = {
            "x": "_auc_label_x",
            "y": "_auc_label_y",
            "text": "_auc_label",
        }
        if color_field is not None:
            text_enc["color"] = color_field
        layers.append(
            _Layer(
                name="auc_label",
                mark="text",
                encoding=text_enc,
                mark_kwargs={"align": "left"},
            )
        )
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("roc", frozenset({"line", "reference", "auc_label"}))


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
    _ = average  # informational at the mark layer
    user_kw = _validate("pr", mark_kwargs)
    line_enc: dict[str, Any] = {"x": "recall", "y": "precision"}
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list = [_Layer(name="line", mark="line", encoding=line_enc)]
    if iso_lines:
        # Iso-F lines are rendered as a separate line layer grouped by
        # `_iso_f` (Utf8 string of the F-score). The chart builder appends
        # one row per (F, recall_step) point along each iso curve.
        layers.append(
            _Layer(
                name="iso_line",
                mark="line",
                encoding={
                    "x": "_iso_recall",
                    "y": "_iso_precision",
                    "color": "_iso_f",
                },
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4], "opacity": 0.6},
            )
        )
        layers.append(
            _Layer(
                name="iso_label",
                mark="text",
                encoding={
                    "x": "_iso_label_x",
                    "y": "_iso_label_y",
                    "text": "_iso_label",
                },
                mark_kwargs={"align": "left", "font_size": 9.0},
            )
        )
    if annotate_ap:
        text_enc: dict[str, Any] = {
            "x": "_ap_label_x",
            "y": "_ap_label_y",
            "text": "_ap_label",
        }
        if color_field is not None:
            text_enc["color"] = color_field
        layers.append(
            _Layer(
                name="ap_label",
                mark="text",
                encoding=text_enc,
                mark_kwargs={"align": "left"},
            )
        )
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("pr", frozenset({"line", "iso_line", "iso_label", "ap_label"}))


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

    Layer wiring (Phase 8a-compliant). The calibration curve reads from the
    primary input (one row per (model, bin)).  The y=x reference diagonal
    reads from a named ``ReferenceLine`` transform that emits exactly two
    rows for the line endpoints — so the diagonal renders once per chart
    regardless of how many models are layered on top.
    """
    # n_bins and strategy are consumed upstream by the chart builder
    # (source.calibration_curve(n_bins=..., strategy=...)); informational
    # at the mark layer — the data is already binned.
    _ = n_bins, strategy
    user_kw = _validate("calibration", mark_kwargs)
    line_enc: dict[str, Any] = {
        "x": "mean_predicted",
        "y": "fraction_positive",
    }
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list = [_Layer(name="line", mark="line", encoding=line_enc)]
    transforms: list = []
    if reference_line:
        from ferrum import ReferenceLine

        transforms.append(
            ReferenceLine(
                "mean_predicted",
                "fraction_positive",
                x=(0.0, 1.0),
                y=(0.0, 1.0),
                name="calibration_ref",
            )
        )
        layers.append(
            _Layer(
                name="reference",
                mark="line",
                encoding={"x": "mean_predicted", "y": "fraction_positive"},
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
                data_source="calibration_ref",
            )
        )
    return MarkDesugarResult(transforms=transforms, layers=_apply(layers, user_kw))


register_layer_names("calibration", frozenset({"line", "reference"}))


def desugar_gain(
    x_field: str | None,
    y_field: str | None,
    *,
    reference_line: bool = True,
    color_field: str | None = "class",
    **mark_kwargs: Any,
) -> tuple:
    """Cumulative-gain mark.

    Data contract: ``percent_population``, ``gain``, ``class`` per
    ``ModelSource.cumulative_gain()``. The data already carries
    ``class='baseline'`` rows that render as the diagonal reference when
    ``color_field='class'``; ``reference_line`` is informational.
    """
    _ = reference_line  # baseline already in data
    user_kw = _validate("gain", mark_kwargs)
    line_enc: dict[str, Any] = {"x": "percent_population", "y": "gain"}
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list = [_Layer(name="line", mark="line", encoding=line_enc)]
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("gain", frozenset({"line"}))


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
    _ = reference_line  # baseline already in data
    user_kw = _validate("lift", mark_kwargs)
    line_enc: dict[str, Any] = {"x": "percent_population", "y": "lift"}
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list = [_Layer(name="line", mark="line", encoding=line_enc)]
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("lift", frozenset({"line"}))


def desugar_discrimination_threshold(
    x_field: str | None,
    y_field: str | None,
    *,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
    n_thresholds: int = 50,
    threshold_line: bool = False,
    optimum_label: bool = True,
    **mark_kwargs: Any,
) -> tuple:
    """Discrimination-threshold sweep mark.

    Data contract (long form): ``threshold``, ``metric``, ``value`` — the
    figure builder is responsible for unpivoting
    ``ModelSource.discrimination_threshold()`` output into this shape.

    When ``threshold_line=True`` the data must carry a sentinel
    ``_threshold_best`` column (one non-null row at the F1-best
    threshold). The desugar emits a vertical ``mark_rule`` layer on
    ``x=_threshold_best``.

    When ``optimum_label=True`` the data must also carry
    ``_optimum_x`` / ``_optimum_y`` / ``_optimum_text`` sentinel columns
    (one non-null row at the F1-best point with the caption
    ``"max F1 = {f1:.3f} @ t={threshold:.2f}"``). The desugar emits a
    ``mark_text`` layer reading those columns. Column injection lives in
    ``Chart.mark_discrimination_threshold``'s ``data_transform`` so the
    feature works for both chart-API and figure-function entry points
    (Schwabish C7 audit-rework, 2026-05-12).
    """
    _ = metrics, n_thresholds  # informational; data is pre-melted
    user_kw = _validate("discrimination_threshold", mark_kwargs)
    layers: list = [
        _Layer(
            name="line", mark="line", encoding={"x": "threshold", "y": "value", "color": "metric"}
        ),
    ]
    if threshold_line:
        layers.append(
            _Layer(
                name="threshold",
                mark="rule",
                encoding={"x": "_threshold_best"},
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4], "opacity": 0.6},
            )
        )
    if optimum_label:
        layers.append(
            _Layer(
                name="optimum_label",
                mark="text",
                encoding={"x": "_optimum_x", "y": "_optimum_y", "text": "_optimum_text"},
                mark_kwargs={"align": "left", "dx": 4, "dy": -4},
            )
        )
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("discrimination_threshold", frozenset({"line", "threshold", "optimum_label"}))


# --- 10c: classification matrices ------------------------------------


def desugar_confusion(
    x_field: str | None,
    y_field: str | None,
    *,
    normalize: str | None = None,
    annotate: bool = True,
    color_field: str = "value",
    cmap: str | None = None,
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

    ``cmap`` selects the sequential colormap applied to the heat cells.
    ``None`` (default) defers to the theme's sequential scheme.
    """
    from ferrum.encoding import Color

    del x_field, y_field
    user_kw = _validate("confusion", mark_kwargs)
    color_enc = Color(color_field, scheme=cmap) if cmap is not None else Color(color_field)
    layers: list = [
        _Layer(
            name="rect",
            mark="rect",
            encoding={"x": "predicted", "y": "actual", "color": color_enc},
        ),
    ]
    if annotate:
        layers.append(
            _Layer(
                name="label",
                mark="text",
                encoding={"x": "predicted", "y": "actual", "text": "value_fmt"},
            )
        )
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("confusion", frozenset({"rect", "label"}))


def desugar_class_prediction_error(
    x_field: str | None,
    y_field: str | None,
    *,
    normalize: bool = False,
    color_field: str = "predicted",
    show_counts: bool = True,
    **mark_kwargs: Any,
) -> tuple:
    """Stacked-bar diagnostic of predicted-class composition.

    Data contract: ``actual``, ``predicted``, ``value`` (same shape as
    ``ModelSource.confusion_matrix(normalize=None)``). One bar per
    ``actual`` class, segments colored by ``predicted`` class. This
    orientation (x = actual, color = predicted) surfaces which classes
    are confused with which — the standard Class Prediction Error layout.
    ``normalize=True`` switches to a per-bar 100% stack via the Stack
    position adjustment.

    Schwabish SB-followup (2026-05-12): ``show_counts=True`` (default)
    appends a same-data ``mark_text`` layer. The text layer carries
    its own Stack adjustment with ``anchor="mid"`` so the renderer
    maps each row's y to the segment MIDPOINT (the bar layer's Stack
    uses default ``anchor="top"``). C6 audit-rework (2026-05-12)
    moved this decision into the position spec; the renderer is
    mark-agnostic. Data must carry a ``_count_text`` Utf8 column
    (the chart builder formats it from ``value``; null on empty
    segments).
    """
    del x_field, y_field
    from ferrum.position import Stack

    user_kw = _validate("class_prediction_error", mark_kwargs)
    offset = "normalize" if normalize else "zero"
    bar_stack = Stack(by=color_field, offset=offset)  # anchor="top" by default
    text_stack = Stack(by=color_field, offset=offset, anchor="mid")
    layers: list = [
        _Layer(
            name="bar",
            mark="bar",
            encoding={"x": "actual", "y": "value", "color": color_field},
            position=bar_stack,
        ),
    ]
    if show_counts:
        layers.append(
            _Layer(
                name="label",
                mark="text",
                encoding={"x": "actual", "y": "value", "text": "_count_text"},
                position=text_stack,
            )
        )
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("class_prediction_error", frozenset({"bar", "label"}))
