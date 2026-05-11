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
