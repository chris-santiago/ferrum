"""Model-diagnostic mark desugars — model selection domain."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ferrum._layer import MarkDesugarResult, _Layer
from ferrum._overrides import register_layer_names
from ferrum.marks._mark_kwargs import (
    apply_user_mark_kwargs as _apply,
    validate_user_mark_kwargs as _validate,
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
            name="band",
            mark="ribbon",
            encoding={
                "x": x_axis,
                "y": y_axis,
                "y2": "upper",
                "color": color_field,
            },
            mark_kwargs={"opacity": 0.3},
        )
    elif ci_style == "errorbar":
        ci_layer = _Layer(
            name="band",
            mark="rule",
            encoding={
                "x": x_axis,
                "y": y_axis,
                "y2": "upper",
                "color": color_field,
            },
        )
    else:
        raise ValueError(
            f"mark_learning_curve(ci_style={ci_style!r}) — expected 'band' or 'errorbar'."
        )
    line_enc: dict[str, Any] = {
        "x": "train_size",
        "y": "mean_score",
        "color": color_field,
    }
    layers = [ci_layer, _Layer(name="line", mark="line", encoding=line_enc)]
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("learning_curve", frozenset({"band", "line"}))


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
            name="band",
            mark="ribbon",
            encoding={
                "x": x_ch,
                "y": y_axis,
                "y2": "upper",
                "color": color_field,
            },
            mark_kwargs={"opacity": 0.3},
        )
    elif ci_style == "errorbar":
        ci_layer = _Layer(
            name="band",
            mark="rule",
            encoding={
                "x": x_ch,
                "y": y_axis,
                "y2": "upper",
                "color": color_field,
            },
        )
    else:
        raise ValueError(
            f"mark_validation_curve(ci_style={ci_style!r}) — expected 'band' or 'errorbar'."
        )
    layers = [
        ci_layer,
        _Layer(name="line", mark="line", encoding={"x": "param_value", "y": "mean_score", "color": color_field}),
    ]
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("validation_curve", frozenset({"band", "line"}))


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
    del x_field, y_field
    # split is consumed upstream by the chart builder (filters DataFrame
    # to the requested split); informational at the mark layer.
    _ = split
    from ferrum.encoding import Y

    user_kw = _validate("cv_scores", mark_kwargs)
    if kind == "box":
        from ferrum.marks.composite import desugar_boxplot

        box_result = desugar_boxplot("split", "score")
        layers = list(box_result.layers)
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
        return MarkDesugarResult(transforms=box_result.transforms, layers=_apply(layers, user_kw))
    if kind == "bar":
        layers = [
            _Layer(
                name="bar",
                mark="bar",
                encoding={"x": "fold", "y": Y("score", title="score"), "color": "split"},
            ),
            _Layer(
                name="mean",
                mark="rule",
                encoding={"y": "_mean_score", "color": "split"},
                mark_kwargs={"stroke_dash": [4, 4]},
            ),
        ]
        return MarkDesugarResult(layers=_apply(layers, user_kw))
    if kind == "strip":
        from ferrum.position import Jitter

        layers = [
            _Layer(
                name="point",
                mark="point",
                encoding={"x": "split", "y": Y("score", title="score"), "color": "split"},
                position=Jitter(axis="x", width=0.3, seed=42),
            ),
        ]
        return MarkDesugarResult(layers=_apply(layers, user_kw))
    raise ValueError(f"mark_cv_scores(kind={kind!r}) — expected 'box', 'bar', or 'strip'.")


register_layer_names("cv_scores", frozenset({"point", "bar", "mean"}))


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
    del x_field, y_field
    # ci_style is informational at the mark layer — alpha_selection
    # renders a single curve without CI bands.
    _ = ci_style
    from ferrum.encoding import Y

    user_kw = _validate("alpha_selection", mark_kwargs)
    x_ch = _log_x_channel("alpha", log_scale)
    layers: list = [
        _Layer(name="line", mark="line", encoding={"x": x_ch, "y": Y("mean_score", title="score")}),
    ]
    if highlight_best:
        layers.append(
            _Layer(
                name="best",
                mark="rule",
                encoding={"x": "_best_alpha"},
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
            )
        )
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("alpha_selection", frozenset({"line", "best"}))
