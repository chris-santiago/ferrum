"""Model-diagnostic mark desugars — ranking domain."""

from __future__ import annotations

from typing import Any

from ferrum._layer import MarkDesugarResult, _Layer
from ferrum._overrides import register_layer_names
from ferrum._validate import validate_choice
from ferrum.marks._mark_kwargs import (
    apply_user_mark_kwargs as _apply,
    validate_user_mark_kwargs as _validate,
)


def desugar_rank1d(
    x_field: str | None,
    y_field: str | None,
    *,
    orient: str = "horizontal",
    color_field: str | None = None,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
    """Univariate feature ranking — one bar per feature, sized by score.

    Data contract: ``feature`` (Utf8), ``score`` (Float64), ``rank``
    (Int64) — the schema emitted by ``ModelSource.rank1d()``. Bars use
    the renderer's ordinal-x / quant-y or ordinal-y / quant-x ``mark_bar``
    path (no per-row bound injection required — the existing horizontal /
    vertical bar dispatch handles both orientations natively).
    """
    del x_field, y_field

    validate_choice("mark_rank1d", "orient", orient, ("horizontal", "vertical"))
    user_kw = _validate("rank1d", mark_kwargs)
    if orient == "horizontal":
        enc: dict[str, Any] = {"x": "score", "y": "feature"}
    else:
        enc = {"x": "feature", "y": "score"}
    if color_field is not None:
        enc["color"] = color_field
    layers: list = [_Layer(name="bar", mark="bar", encoding=enc)]
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("rank1d", frozenset({"bar"}))


def desugar_rank2d(
    x_field: str | None,
    y_field: str | None,
    *,
    annot: bool = True,
    color_field: str = "correlation",
    text_field: str = "correlation_fmt",
    cmap: str | None = None,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
    """Pairwise feature ranking — long-form correlation matrix heatmap.

    Data contract: ``feature_x`` (Utf8), ``feature_y`` (Utf8),
    ``correlation`` (Float64) — schema from ``ModelSource.rank2d()``.
    When ``annot=True``, the data must also carry ``correlation_fmt``
    (Utf8 — preformatted to 2 decimal places by the chart builder so
    the renderer can lay out short labels without invoking
    Rust-side number formatting per cell).

    ``cmap`` selects the diverging colormap applied to correlation cells.
    ``None`` (default) defers to the theme's diverging scheme.
    """
    from ferrum.encoding import Color

    del x_field, y_field

    from ferrum.encoding import X, Y

    user_kw = _validate("rank2d", mark_kwargs)
    color_enc = Color(color_field, scheme=cmap) if cmap is not None else Color(color_field)
    rect_enc: dict[str, Any] = {
        "x": X("feature_x", title="Feature"),
        "y": Y("feature_y", title="Feature"),
        "color": color_enc,
    }
    layers: list = [_Layer(name="rect", mark="rect", encoding=rect_enc)]
    if annot:
        layers.append(
            _Layer(
                name="label",
                mark="text",
                encoding={
                    "x": "feature_x",
                    "y": "feature_y",
                    "text": text_field,
                },
            )
        )
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("rank2d", frozenset({"rect", "label"}))


def desugar_parallel_coordinates(
    x_field: str | None,
    y_field: str | None,
    *,
    alpha: float = 0.5,
    color_field: str | None = None,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
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
    layers: list = [
        _Layer(
            name="line",
            mark="line",
            encoding=line_enc,
            mark_kwargs={
                "detail": "sample_id",
                "opacity": float(alpha),
            },
        )
    ]
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("parallel_coordinates", frozenset({"line"}))
