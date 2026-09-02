"""Model-diagnostic mark desugars — clustering / decision boundary domain."""

from __future__ import annotations

from typing import Any

from ferrum._layer import MarkDesugarResult, _Layer
from ferrum._overrides import register_layer_names
from ferrum.marks._desugar_helpers import nominal_color_channel
from ferrum.marks._mark_kwargs import (
    apply_user_mark_kwargs as _apply,
    validate_user_mark_kwargs as _validate,
)

# --- 10f: clustering / manifold / decision boundary -------------------


def desugar_silhouette(
    x_field: str | None,
    y_field: str | None,
    *,
    zero_line: bool = True,
    color_field: str | None = "cluster",
    **mark_kwargs: Any,
) -> MarkDesugarResult:
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
        # `cluster` is documented Int64 (KMeans-style integer labels), so a
        # bare string here infers Continuous and silently renders a
        # colorbar instead of a per-cluster swatch legend -- no warning,
        # since UnsupportedColorScaleOnMark only fires for line/ribbon.
        # Bind Nominal explicitly (see nominal_color_channel's docstring).
        rect_enc["color"] = nominal_color_channel(color_field)
    layers: list = [_Layer(name="rect", mark="rect", encoding=rect_enc)]
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


register_layer_names("silhouette", frozenset({"rect", "reference"}))


def desugar_pca_scree(
    x_field: str | None,
    y_field: str | None,
    *,
    cumulative_line: bool = True,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
    """PCA scree plot: bar of per-component variance + optional cumulative
    line.

    Data contract: ``component`` (Utf8, cast from Int64 by
    ``_pca_scree_prep``), ``explained_variance_ratio`` (Float64),
    ``cumulative_variance_ratio`` (Float64) as emitted by
    ``ModelSource.pca_variance()``.

    Component is cast to String by ``_pca_scree_prep`` so the x scale
    resolves as ordinal — ``mark_bar`` uses native band positioning and
    the axis shows integer-only labels ("1", "2", …) with no fractional
    ticks.

    This desugar has no ``threshold_line``/``n_components`` parameters:
    ``Chart.mark_pca_scree`` routes a non-None ``threshold_line`` to the
    sibling ``desugar_pca_scree_with_threshold`` instead (this function is
    only ever reached when ``threshold_line`` is ``None``), and
    ``n_components`` belongs to the ``pca_scree_chart`` figure function
    (consumed by ``ModelSource.pca_variance(n_components=...)`` before the
    mark ever sees the data) -- neither reaches this mark layer.
    """
    del x_field, y_field
    from ferrum.encoding import X, Y

    user_kw = _validate("pca_scree", mark_kwargs)

    if cumulative_line:
        layers: list = [
            _Layer(
                name="cumulative",
                mark="line",
                encoding={
                    "x": X("component", title="Component"),
                    "y": Y("cumulative_variance_ratio", title="Explained variance"),
                    "y2": "_y_axis_anchor",
                },
            ),
            _Layer(
                name="bar",
                mark="bar",
                encoding={
                    "x": "component",
                    "y": "explained_variance_ratio",
                },
            ),
        ]
    else:
        layers = [
            _Layer(
                name="bar",
                mark="bar",
                encoding={
                    "x": X("component", title="Component"),
                    "y": Y("explained_variance_ratio", title="Explained variance"),
                },
            ),
        ]
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("pca_scree", frozenset({"bar", "cumulative"}))


def desugar_pca_scree_with_threshold(
    x_field: str | None,
    y_field: str | None,
    *,
    cumulative_line: bool = True,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
    """Variant of ``desugar_pca_scree`` that appends a threshold rule.

    Used by ``Chart.mark_pca_scree`` when ``threshold_line`` is non-None;
    references the injected ``_threshold_line`` sentinel column. Like its
    sibling, this desugar has no ``n_components`` parameter -- that belongs
    to the ``pca_scree_chart`` figure function and is consumed upstream of
    the mark layer (see ``desugar_pca_scree``'s docstring).
    """
    # Validate kwargs here so the AST guardrail (test_mark_kwargs_no_silent_drop)
    # sees a call to validate_user_mark_kwargs at this function level. The
    # nested desugar_pca_scree call validates the same set independently, so
    # the second validation is a no-op for well-formed inputs.
    _validate("pca_scree", mark_kwargs)
    scree_result = desugar_pca_scree(
        x_field,
        y_field,
        cumulative_line=cumulative_line,
        **mark_kwargs,
    )
    layers = list(scree_result.layers) + [
        _Layer(
            name="threshold",
            mark="rule",
            encoding={"y": "_threshold_line"},
            mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
        )
    ]
    return MarkDesugarResult(transforms=scree_result.transforms, layers=layers)


register_layer_names("pca_scree_with_threshold", frozenset({"bar", "cumulative", "threshold"}))


def desugar_intercluster_distance(
    x_field: str | None,
    y_field: str | None,
    *,
    label_clusters: bool = True,
    color_field: str | None = "cluster",
    min_size: float = 60.0,
    max_size: float = 600.0,
    **mark_kwargs: Any,
) -> MarkDesugarResult:
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
    layers: list = [_Layer(name="point", mark="point", encoding=point_enc)]
    if label_clusters:
        layers.append(
            _Layer(
                name="label",
                mark="text",
                encoding={"x": "x", "y": "y", "text": "cluster"},
            )
        )
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("intercluster_distance", frozenset({"point", "label"}))


def desugar_decision_boundary(
    x_field: str | None,
    y_field: str | None,
    *,
    proba: bool = False,
    color_field: str = "z",
    **mark_kwargs: Any,
) -> MarkDesugarResult:
    """Decision-boundary background heatmap: one rect per grid cell.

    Data contract: pre-computed grid with columns ``x``, ``x2``, ``y``,
    ``y2`` (cell bounds) and ``z`` (the prediction value — class index
    when ``proba=False``, probability when ``proba=True``). The chart
    builder produces these columns from a ``ModelSource``.

    ``proba`` is informational at the mark layer — the chart builder
    chooses the data and the renderer's continuous-color scale handles
    both kinds of ``z`` identically. It is registered in
    ``ferrum.marks._informational_kwargs.INFORMATIONAL_KWARGS`` under
    ``"decision_boundary"``; ``Chart.mark_decision_boundary`` warns once if
    it is passed directly with a truthy value.
    """
    del x_field, y_field
    # proba is informational at the mark layer -- see the docstring above
    # and ferrum.marks._informational_kwargs.INFORMATIONAL_KWARGS, which is
    # what Chart.mark_decision_boundary's warn_once is keyed against.
    del proba

    user_kw = _validate("decision_boundary", mark_kwargs)
    layers: list = [
        _Layer(
            name="rect",
            mark="rect",
            encoding={
                "x": "x",
                "x2": "x2",
                "y": "y",
                "y2": "y2",
                "color": color_field,
            },
            mark_kwargs={"opacity": 0.5},
        ),
    ]
    return MarkDesugarResult(layers=_apply(layers, user_kw))


register_layer_names("decision_boundary", frozenset({"rect"}))
