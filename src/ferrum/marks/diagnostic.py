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
