"""Model-diagnostic mark desugars — explanation (SHAP / PDP) domain."""

from __future__ import annotations

from typing import Any

from ferrum._layer import MarkDesugarResult, _Layer
from ferrum._overrides import register_layer_names
from ferrum._validate import validate_choice
from ferrum.marks._mark_kwargs import (
    apply_user_mark_kwargs as _apply,
    validate_user_mark_kwargs as _validate,
)

_ORIENT_CHOICES = ("horizontal", "vertical")

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
) -> MarkDesugarResult:
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
    del x_field, y_field
    # top_k is wired via data_transform in mark_importance; informational
    # at the desugar layer.
    del top_k
    user_kw = _validate("importance", mark_kwargs)
    validate_choice("mark_importance", "orient", orient, _ORIENT_CHOICES)

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
    layers: list = [_Layer(name="bar", mark="bar", encoding=bar_enc)]

    if error_bars:
        err_enc: dict[str, Any] = {
            value_axis: _value_channel(err_lower),
            err_axis2: err_upper,
            group_axis: group_field,
        }
        layers.append(_Layer(name="errorbar", mark="rule", encoding=err_enc))

    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("importance", frozenset({"bar", "errorbar"}))


def desugar_shap_beeswarm(
    x_field: str | None,
    y_field: str | None,
    *,
    max_display: int = 20,
    color_bar: bool = True,
    order: str = "abs_mean",
    zero_line: bool = True,
    x_scale_domain: tuple[float, float] | list[float] | None = None,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
    """SHAP beeswarm mark: categorical scatter of per-sample shap values.

    Data contract: ``feature`` (Utf8), ``shap_value`` (Float64),
    ``feature_value_normalized`` (Float64) as emitted by
    ``ModelSource.shap_values()`` and pre-filtered by the chart builder
    to the top ``max_display`` features. When ``zero_line=True`` the
    data must also carry a sentinel ``_ref_zero`` Float64 column (one
    ``0.0`` value, rest null) — ``Chart.mark_shap_beeswarm`` injects
    that column via its ``data_transform``.

    Renders one point per (sample, feature) cell with feature on the
    ordinal y-axis, shap_value on the quantitative x-axis, and the
    z-scored feature value on the continuous color scale. Vertical
    spread within each feature band uses the Phase 10d-pre Jitter
    ordinal-axis path; ``width=0.6`` keeps the band well-contained.
    ``zero_line=True`` (Schwabish SB-followup 2026-05-12) appends a
    dashed ``mark_rule`` layer at ``x=0`` so the sign of each feature's
    SHAP impact is immediately legible.

    ``color_bar`` and ``order`` are informational at the mark layer —
    the chart builder is responsible for any reordering / aggregation
    before constructing the chart.
    """
    del x_field, y_field
    # max_display is wired via data_transform in mark_shap_beeswarm;
    # color_bar and order are consumed upstream by the chart builder.
    del max_display, color_bar, order
    user_kw = _validate("shap_beeswarm", mark_kwargs)

    def _x_channel(field: str) -> Any:
        if x_scale_domain is None:
            return field
        from ferrum.encoding import X

        return X(field, scale={"type": "linear", "domain": list(x_scale_domain)})

    from ferrum.encoding import Color
    from ferrum.position import Jitter

    layers: list = [
        _Layer(
            name="point",
            mark="point",
            encoding={
                "x": _x_channel("shap_value"),
                "y": "feature",
                "color": Color(
                    "feature_value_normalized",
                    scheme="rdbu",
                    title="Feature value",
                    legend={"tickLabels": ["Low", "", "", "", "High"]},
                ),
            },
            position=Jitter(axis="y", width=0.6, seed=42),
        ),
    ]
    if zero_line:
        layers.append(
            _Layer(
                name="reference",
                mark="rule",
                encoding={"x": "_ref_zero"},
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
            )
        )
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("shap_beeswarm", frozenset({"point", "reference"}))


def desugar_shap_bar(
    x_field: str | None,
    y_field: str | None,
    *,
    max_display: int = 20,
    orient: str = "horizontal",
    color_field: str | None = None,
    x_scale_domain: tuple[float, float] | list[float] | None = None,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
    """Aggregated-SHAP bar mark: mean(|shap_value|) per feature.

    Data contract: ``feature`` (Utf8), ``abs_mean_shap`` (Float64) — the
    chart builder aggregates ``ModelSource.shap_values()`` and selects
    the top ``max_display`` features.

    ``orient="horizontal"`` (default) renders value-on-x, ordinal-feature-on-y
    — the always-horizontal single-model layout. ``orient="vertical"`` swaps
    the axes (ordinal feature on x, value on y): the ``compare=`` dodge-by-
    model builder uses this form because dodge requires an ordinal-x band
    axis, then re-applies ``CoordFlip`` to restore the horizontal visual
    (spec D2, mirrors ``desugar_importance``).
    """
    del x_field, y_field
    # max_display is wired via data_transform in mark_shap_bar.
    del max_display
    user_kw = _validate("shap_bar", mark_kwargs)
    validate_choice("mark_shap_bar", "orient", orient, _ORIENT_CHOICES)

    if orient == "horizontal":
        value_axis, group_axis = "x", "y"
    else:
        value_axis, group_axis = "y", "x"
    value_field, group_field = "abs_mean_shap", "feature"

    def _value_channel(field: str) -> Any:
        from ferrum.encoding import X, Y

        kw: dict[str, Any] = {"title": "Mean |SHAP value|"}
        if x_scale_domain is not None:
            kw["scale"] = {"type": "linear", "domain": list(x_scale_domain)}
        ch_cls = X if value_axis == "x" else Y
        return ch_cls(field, **kw)

    bar_enc: dict[str, Any] = {
        value_axis: _value_channel(value_field),
        group_axis: group_field,
    }
    if color_field is not None:
        bar_enc["color"] = color_field
    layers: list = [_Layer(name="bar", mark="bar", encoding=bar_enc)]
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("shap_bar", frozenset({"bar"}))


def desugar_shap_waterfall(
    x_field: str | None,
    y_field: str | None,
    *,
    sample_idx: int = -1,
    max_display: int = 20,
    x_scale_domain: tuple[float, float] | list[float] | None = None,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
    """SHAP waterfall mark: per-feature contribution segments for one sample.

    Data contract: ``feature`` (Utf8), ``x0`` (cumulative start),
    ``x1`` (cumulative end), ``shap_sign`` (Utf8: 'positive'|'negative')
    pre-computed by the chart builder. Renders a horizontal-ranged bar
    per feature via the Phase 10d-pre quantitative-x + x2 + ordinal-y
    bar path.
    """
    del x_field, y_field
    # max_display is wired via data_transform in mark_shap_waterfall.
    del max_display
    user_kw = _validate("shap_waterfall", mark_kwargs)
    if sample_idx < 0:
        raise ValueError(
            "mark_shap_waterfall(sample_idx=...) is required. Pass an "
            "explicit non-negative sample index."
        )

    def _x_channel(field: str, title: str | None = None) -> Any:
        from ferrum.encoding import X

        kw: dict[str, Any] = {}
        if title is not None:
            kw["title"] = title
        if x_scale_domain is not None:
            kw["scale"] = {"type": "linear", "domain": list(x_scale_domain)}
        if kw:
            return X(field, **kw)
        return field

    layers: list = [
        _Layer(
            name="bar",
            mark="bar",
            encoding={
                "x": _x_channel("x0", title="SHAP value"),
                "x2": "x1",
                "y": "feature",
                "color": "shap_sign",
            },
        ),
    ]
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("shap_waterfall", frozenset({"bar"}))


def desugar_pdp(
    x_field: str | None,
    y_field: str | None,
    *,
    kind: str = "average",
    ice_alpha: float = 0.2,
    center: bool = False,
    color_field: str | None = "feature",
    **mark_kwargs: Any,
) -> MarkDesugarResult:
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
    validate_choice("mark_pdp", "kind", kind, ("average", "individual", "both"))
    user_kw = _validate("pdp", mark_kwargs)

    if kind == "average":
        # Single polyline per feature, color-coded by feature when faceted.
        line_enc: dict[str, Any] = {"x": "feature_value", "y": "pd_value"}
        if color_field is not None:
            line_enc["color"] = color_field
        layers: list = [_Layer(name="line", mark="line", encoding=line_enc)]
        return MarkDesugarResult(layers=_apply(layers, user_kw))

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
        layers = [
            _Layer(
                name="ice",
                mark="line",
                encoding=line_enc,
                mark_kwargs={
                    "detail": "_sample_id_str",
                    "opacity": float(ice_alpha),
                },
            ),
        ]
        return MarkDesugarResult(layers=_apply(layers, user_kw))

    # kind == "both" (validated above)
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
        "x": "feature_value",
        "y": "_pd_avg_value",
    }
    if color_field is not None:
        avg_enc["color"] = color_field
    layers = [
        _Layer(
            name="ice",
            mark="line",
            encoding=ice_enc,
            mark_kwargs={
                "detail": "_sample_id_str",
                "opacity": float(ice_alpha),
            },
        ),
        _Layer(
            name="average",
            mark="line",
            encoding=avg_enc,
            mark_kwargs={"stroke_width": 2.5},
        ),
    ]
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("pdp", frozenset({"line", "ice", "average"}))
