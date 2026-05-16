"""Model-diagnostic mark desugars — regression domain."""

from __future__ import annotations

from typing import Any

from ferrum._layer import MarkDesugarResult, _Layer
from ferrum._overrides import register_layer_names
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
    **mark_kwargs: Any,
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
    user_kw = _validate("residuals", mark_kwargs)
    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"
    point_enc: dict[str, Any] = {"x": "y_pred", "y": y_col}
    if color_field is not None:
        point_enc["color"] = color_field
    layers: list = [_Layer(name="point", mark="point", encoding=point_enc)]
    if reference_line:
        layers.append(
            _Layer(
                name="reference",
                mark="rule",
                encoding={"y": "_ref_zero"},
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
            )
        )
    if cook_threshold is not None:
        layers.append(
            _Layer(
                name="outlier",
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
            )
        )
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("residuals", frozenset({"point", "reference", "outlier"}))


def desugar_prediction_error(
    x_field: str | None,
    y_field: str | None,
    *,
    reference_line: bool = True,
    ci: float | None = None,
    reference_band: bool = False,
    color_field: str | None = None,
    **mark_kwargs: Any,
) -> tuple:
    """Actual vs predicted: scatter of (y_true, y_pred) + optional identity line.

    Data contract: columns ``y_true`` and ``y_pred``. When
    ``reference_line=True`` the data must be sorted ascending by ``y_true`` so
    the line layer renders as a clean y=x diagonal (handled by
    ``Chart.mark_prediction_error``).

    When ``ci is not None`` or ``reference_band=True``, the data must also
    carry the injected ``_pe_band_lo`` / ``_pe_band_hi`` columns (the chart
    builder pre-computes these as the ``ci``-width band around the identity
    line), and this desugar emits a ``ribbon`` layer between those bounds with
    ``opacity=0.2`` so the underlying scatter remains visible.
    """
    user_kw = _validate("prediction_error", mark_kwargs)
    point_enc: dict[str, Any] = {"x": "y_true", "y": "y_pred"}
    if color_field is not None:
        point_enc["color"] = color_field
    # Layer ordering: point first so layer-0-driven axis-scale resolution
    # picks up `y_pred` as the y-axis title. The ribbon and identity-line
    # layers paint on top; opacity=0.2 on the ribbon keeps the underlying
    # points visible.
    layers: list = [_Layer(name="point", mark="point", encoding=point_enc)]
    if ci is not None or reference_band:
        layers.append(
            _Layer(
                name="band",
                mark="ribbon",
                encoding={
                    "x": "y_true",
                    "y": "_pe_band_lo",
                    "y2": "_pe_band_hi",
                },
                mark_kwargs={"opacity": 0.2},
            )
        )
    if reference_line:
        layers.append(
            _Layer(
                name="identity",
                mark="line",
                encoding={"x": "y_true", "y": "y_true"},
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
            )
        )
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("prediction_error", frozenset({"point", "band", "identity"}))
