"""Explanation domain figure functions and their private builders.

Public API
----------
importance_chart, shap_chart, shap_beeswarm_chart, shap_bar_chart,
shap_waterfall_chart, pdp_chart.

Each public function wraps ``_resolve_source`` (shared across all figure
functions) and dispatches to a co-located ``_*_from_source`` builder that
produces a fully-formed ``Chart``.

Internal-only builders
----------------------
_importance_chart_from_source, _shap_select_class, _shap_order_features,
_shap_beeswarm_chart_from_source, _shap_bar_chart_from_source,
_shap_waterfall_chart_from_source, _pdp_center_curves,
_pdp_split_kind_both, _pdp_chart_from_source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from ferrum import Chart

from ferrum.encoding import X, Y
from ferrum._overrides import _apply_overrides
from ferrum.plots._helpers import (
    _finalize_chart,
    _require,
    _resolve_source,
    _should_facet_by_class,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SHAP_ORDER_VALUES: frozenset[str] = frozenset({"abs_mean", "max"})


# ---------------------------------------------------------------------------
# Private builders — feature importance
# ---------------------------------------------------------------------------


def _importance_chart_from_source(
    source: Any,
    *,
    method: str = "builtin",
    top_k: int | None = 20,
    orient: str = "horizontal",
    error_bars: bool = True,
    show_values: bool = True,
    subtitle: str | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Build a feature-importance chart from a ModelSource.

    Schwabish SB3 (2026-05-11): ``Feature importance`` title and, when
    ``show_values=True``, overlays the formatted importance value at the
    end of each bar via a same-data text layer.
    """
    import ferrum

    df = source.importances(method=method, random_state=random_state)
    if top_k is not None:
        df = df.head(top_k)
    df = df.with_columns(
        [
            (pl.col("importance") - pl.col("std")).alias("imp_lower"),
            (pl.col("importance") + pl.col("std")).alias("imp_upper"),
        ]
    )

    if show_values:
        df = df.with_columns(
            pl.col("importance")
            .map_elements(lambda v: f"{v:.3f}", return_dtype=pl.Utf8)
            .alias("_value_text"),
        )

    upper_max = float(df["imp_upper"].max())
    lower_min = float(df["imp_lower"].min())
    domain_lo = min(0.0, lower_min)
    domain_hi = max(upper_max, 0.0) * 1.05 if upper_max > 0 else 1.0

    chart = ferrum.Chart(df).mark_importance(
        orient=orient,
        error_bars=error_bars,
        top_k=top_k,
        x_scale_domain=(domain_lo, domain_hi),
    )
    chart = chart.properties(
        title=ferrum.Title("Feature importance", subtitle=subtitle),
    )

    if show_values:
        from ferrum._layer import _Layer

        if orient == "horizontal":
            text_ly = _Layer(
                mark="text",
                encoding={"x": "importance", "y": "feature", "text": "_value_text"},
                mark_kwargs={"align": "left", "dx": 4},
                name="value_text",
            )
        else:
            text_ly = _Layer(
                mark="text",
                encoding={"x": "feature", "y": "importance", "text": "_value_text"},
                mark_kwargs={"baseline": "bottom", "dy": -4},
                name="value_text",
            )
        chart = chart.layer(text_ly)

    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


# ---------------------------------------------------------------------------
# Private builders — SHAP family
# ---------------------------------------------------------------------------


def _shap_select_class(sv: pl.DataFrame, *, per_class: bool) -> pl.DataFrame:
    """Either return all rows (``per_class=True``) or filter to the first
    ``class_label`` group (``per_class=False``). On regression and binary
    classifiers the single-group filter is a no-op.
    """
    if per_class:
        return sv
    classes = sv["class_label"].unique(maintain_order=True)
    if classes.len() <= 1:
        return sv
    first_class = classes[0]
    return sv.filter(pl.col("class_label") == first_class)


def _shap_order_features(
    sv: pl.DataFrame,
    *,
    order: str,
    max_display: int,
) -> list[str]:
    """Return the top-`max_display` feature names ordered by `order`."""
    if order not in _SHAP_ORDER_VALUES:
        raise ValueError(f"Unknown order {order!r}. Accepted values: {sorted(_SHAP_ORDER_VALUES)}")
    expr = pl.col("shap_value").abs()
    agg = expr.mean() if order == "abs_mean" else expr.max()
    ranked = (
        sv.group_by("feature")
        .agg(agg.alias("score"))
        .sort("score", descending=True)
        .head(max_display)
    )
    return ranked["feature"].to_list()


def _shap_beeswarm_chart_from_source(
    source: Any,
    *,
    max_display: int = 20,
    order: str = "abs_mean",
    background: Any = None,
    per_class: bool = False,
    zero_line: bool = True,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Beeswarm chart: per-sample shap values colored by feature value.

    ``per_class=True`` on a multi-class classifier returns a faceted chart
    with one beeswarm panel per class; ``per_class=False`` (default)
    renders a single panel using the first class_label group (the only
    group on regression and binary classifiers).

    Schwabish SB-followup (2026-05-12): ``zero_line=True`` (default)
    routes through ``Chart.mark_shap_beeswarm(zero_line=...)`` which
    owns both the sentinel ``_ref_zero`` column injection (via its
    ``data_transform``) and the dashed-rule layer (emitted by
    ``desugar_shap_beeswarm``). Chart-API callers (
    ``Chart(df).mark_shap_beeswarm(zero_line=True)``) get the same
    behaviour. Disabled automatically on the faceted ``per_class``
    multi-panel path — each facet would need its own non-null sentinel
    row.
    """
    import ferrum

    sv = source.shap_values(background=background)
    sv = _shap_select_class(sv, per_class=per_class)
    keep = _shap_order_features(sv, order=order, max_display=max_display)
    plot_df = sv.filter(pl.col("feature").is_in(keep))
    # Sort rows so features appear in descending mean-|SHAP| order (keep[0]
    # is most important). Rust's distinct_values_in_order uses encounter order
    # to build the ordinal-y domain, so row order drives axis ordering.
    plot_df = (
        plot_df.with_columns(pl.col("feature").cast(pl.Enum(keep)).alias("_feature_order"))
        .sort("_feature_order")
        .drop("_feature_order")
    )

    is_faceted = _should_facet_by_class(plot_df, per_class=per_class)
    x_min = float(plot_df["shap_value"].min())
    x_max = float(plot_df["shap_value"].max())
    pad = max(abs(x_min), abs(x_max)) * 0.05 if (x_min < x_max) else 1.0
    domain = (x_min - pad, x_max + pad)

    chart = ferrum.Chart(plot_df).mark_shap_beeswarm(
        max_display=max_display,
        order=order,
        zero_line=zero_line and not is_faceted,
        x_scale_domain=domain,
    )
    chart = chart.encode(
        x=X("shap_value", title="SHAP value"),
        y=Y("feature", title="Feature"),
    )
    chart = chart.properties(title=ferrum.Title("SHAP Summary"))
    if is_faceted:
        chart = chart.facet(col="class_label")
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def _shap_bar_chart_from_source(
    source: Any,
    *,
    max_display: int = 20,
    order: str = "abs_mean",
    background: Any = None,
    per_class: bool = False,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """SHAP aggregated bar chart: mean(|shap_value|) per feature.

    ``order`` selects the aggregation used both for sorting and as the
    bar's x-value. ``"abs_mean"`` aggregates by ``mean(|shap_value|)``
    (the conventional summary); ``"max"`` aggregates by
    ``max(|shap_value|)`` to emphasize features with high-impact outliers.
    The output column is named ``abs_mean_shap`` regardless of the
    aggregation so the downstream mark's data contract stays stable.

    ``per_class=True`` on a multi-class classifier facets the chart by
    ``class_label`` (one bar panel per class). ``per_class=False``
    (default) renders a single panel using the first class_label group
    (the only group on regression and binary classifiers).
    """
    if order not in _SHAP_ORDER_VALUES:
        raise ValueError(f"Unknown order {order!r}. Accepted values: {sorted(_SHAP_ORDER_VALUES)}")
    import ferrum

    expr = pl.col("shap_value").abs()
    agg_expr = expr.mean() if order == "abs_mean" else expr.max()
    sv = source.shap_values(background=background)
    sv = _shap_select_class(sv, per_class=per_class)
    group_keys = ["class_label", "feature"] if per_class else ["feature"]
    agg = (
        sv.group_by(group_keys)
        .agg(agg_expr.alias("abs_mean_shap"))
        .sort("abs_mean_shap", descending=True)
        .head(max_display * sv["class_label"].n_unique() if per_class else max_display)
    )
    x_max = float(agg["abs_mean_shap"].max())
    domain = (0.0, x_max * 1.05 if x_max > 0 else 1.0)
    chart = ferrum.Chart(agg).mark_shap_bar(
        max_display=max_display,
        x_scale_domain=domain,
    )
    if _should_facet_by_class(agg, per_class=per_class):
        chart = chart.facet(col="class_label")
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def _shap_waterfall_chart_from_source(
    source: Any,
    *,
    sample_idx: int,
    max_display: int = 20,
    order: str = "abs_mean",
    background: Any = None,
    per_class: bool = False,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Waterfall chart for a single sample's SHAP contributions.

    Feature display order follows the global aggregation chosen by
    ``order``: ``"abs_mean"`` ranks features by ``mean(|shap_value|)``
    across the full dataset (matching the beeswarm and bar paths);
    ``"max"`` ranks by ``max(|shap_value|)``. The top ``max_display``
    features are kept and rendered in descending rank order; the
    waterfall's cumulative sum follows that same order.

    ``per_class=True`` on a multi-class classifier facets the chart by
    ``class_label`` (one waterfall panel per class for the same sample).
    ``per_class=False`` (default) renders a single panel using the first
    class_label group.
    """
    import ferrum

    sv = source.shap_values(background=background)
    sv = _shap_select_class(sv, per_class=per_class)
    one = sv.filter(pl.col("sample_id") == sample_idx)
    if one.height == 0:
        raise ValueError(
            f"shap_waterfall: sample_idx={sample_idx} not found in shap output "
            f"(have {sv['sample_id'].n_unique()} samples)."
        )
    # Use the global aggregation (over all samples) to rank features —
    # matches the beeswarm / bar ordering so all three views agree on
    # which features matter for the model overall. Then project that
    # ranking onto the single sample's rows.
    ranked = _shap_order_features(sv, order=order, max_display=max_display)
    rank_map = {name: i for i, name in enumerate(ranked)}
    ordered = (
        one.filter(pl.col("feature").is_in(ranked))
        .with_columns(
            pl.col("feature").replace_strict(rank_map, return_dtype=pl.Int64).alias("_rank"),
        )
        .sort("_rank")
        .drop("_rank")
    )
    cumsum = ordered["shap_value"].cum_sum()
    x0 = pl.concat([pl.Series("x0", [0.0]), cumsum.head(cumsum.len() - 1).alias("x0")])
    x1 = cumsum.alias("x1")
    plot_df = ordered.with_columns(
        [
            x0,
            x1,
            pl.when(pl.col("shap_value") >= 0)
            .then(pl.lit("positive"))
            .otherwise(pl.lit("negative"))
            .alias("shap_sign"),
        ]
    )

    x_lo = float(min(x0.min(), x1.min(), 0.0))
    x_hi = float(max(x0.max(), x1.max(), 0.0))
    pad = max(abs(x_lo), abs(x_hi)) * 0.05 if (x_lo < x_hi) else 1.0
    domain = (x_lo - pad, x_hi + pad)

    chart = ferrum.Chart(plot_df).mark_shap_waterfall(
        sample_idx=sample_idx,
        max_display=max_display,
        x_scale_domain=domain,
    )
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


# ---------------------------------------------------------------------------
# Private builders — partial dependence
# ---------------------------------------------------------------------------


def _pdp_center_curves(df: pl.DataFrame) -> pl.DataFrame:
    """Subtract each curve's value at its smallest feature_value so every
    polyline starts at zero. Anchors per ``(feature, sample_id)`` because
    each (feature, sample) has its own grid; the smallest grid value is
    the anchor.
    """
    anchor = df.group_by(["feature", "sample_id"], maintain_order=True).agg(
        pl.col("pd_value").first().alias("_pd_anchor")
    )
    return (
        df.join(anchor, on=["feature", "sample_id"], how="left")
        .with_columns(
            (pl.col("pd_value") - pl.col("_pd_anchor")).alias("pd_value"),
        )
        .drop("_pd_anchor")
    )


def _pdp_split_kind_both(df: pl.DataFrame) -> pl.DataFrame:
    """Split ``pd_value`` into two layer-specific columns for ``kind="both"``.

    - ``_pd_ice_value``: per-sample value on ICE rows (sample_id >= 0),
      null on the average row so the ICE line layer skips it.
    - ``_pd_avg_value``: the average curve broadcast to every row (joined
      from the sample_id=-1 rows by ``(feature, feature_value)``). The
      average layer reads this on ALL rows but mark_line groups by color
      (feature) so duplicated rows merge into one polyline.

    Drops the original ``sample_id == -1`` rows so the duplicate-avg
    polyline collapses to one rendering; ``_pd_avg_value`` on the ICE
    rows already carries the full average curve.
    """
    avg_only = (
        df.filter(pl.col("sample_id") == -1)
        .select(["feature", "feature_value", "pd_value"])
        .rename({"pd_value": "_pd_avg_value"})
    )
    df = df.with_columns(
        pl.when(pl.col("sample_id") >= 0)
        .then(pl.col("pd_value"))
        .otherwise(None)
        .alias("_pd_ice_value"),
    )
    df = df.join(avg_only, on=["feature", "feature_value"], how="left")
    return df.filter(pl.col("sample_id") >= 0)


def _pdp_chart_from_source(
    source: Any,
    features: list,
    *,
    grid_resolution: int = 100,
    kind: str = "average",
    ice_alpha: float = 0.2,
    center: bool = False,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Partial-dependence chart: faceted, one polyline per feature panel.

    Each feature occupies its own facet panel because PDP curves for
    different features span unrelated x-axis ranges and units — sharing
    a single x axis collapses every curve onto the union range.

    For ``kind="individual"``: injects ``_sample_id_str`` (Utf8 cast of
    ``sample_id``) so ``mark_line``'s ``mark_style.detail`` grouping
    can produce one polyline per sample.

    For ``kind="both"``: injects two y-source columns —
    ``_pd_ice_value`` (per-sample value on ICE rows, null on the average
    row) and ``_pd_avg_value`` (the average curve broadcast to every
    row; non-null on ICE rows too so the average overlay renders the
    full grid). Each layer reads its own y column.

    When ``center=True``: subtracts the value at the smallest
    ``feature_value`` per ``(feature, sample_id)`` group so every
    polyline starts at 0.
    """
    import ferrum

    df = source.partial_dependence(
        features,
        grid_resolution=grid_resolution,
        kind=kind,
    )
    # Sort by (feature, sample_id, feature_value) so each polyline's
    # rows are contiguous and monotonic in feature_value within its
    # group — line.rs renders rows in batch order within each detail /
    # color group.
    df = df.sort(["feature", "sample_id", "feature_value"])

    if center:
        df = _pdp_center_curves(df)
    if kind in ("individual", "both"):
        # mark_line reads mark_style.detail as a Utf8 column name. Cast
        # sample_id to Utf8 so the detail grouping works.
        df = df.with_columns(
            pl.col("sample_id").cast(pl.Utf8).alias("_sample_id_str"),
        )
    if kind == "both":
        df = _pdp_split_kind_both(df)

    chart = (
        ferrum.Chart(df)
        .mark_pdp(
            kind=kind,
            ice_alpha=ice_alpha,
            center=center,
            color_field=None,  # one feature per facet — color is redundant
        )
        .encode(
            x=X("feature_value", title="Feature value"),
            y=Y("pd_value", title="Partial dependence"),
        )
        .facet(col="feature")
        # Each feature panel is a DIFFERENT variable with its own x-units, so the
        # x-axis must resolve per-panel. The facet default is shared (which is
        # correct when every panel shows the same axis variable); PDP is the one
        # case where the faceting field IS the x-variable, so we opt out
        # explicitly. (Before the round-7 T4 shared-faceted-scale fix, raw
        # positional domains resolved per-panel regardless — PDP relied on that
        # bug; this makes the per-panel intent explicit and paradigm-correct.)
        .share_scale(x="independent")
    )
    chart = chart.properties(title=ferrum.Title("Partial Dependence"))
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


# ---------------------------------------------------------------------------
# Public figure functions
# ---------------------------------------------------------------------------


def importance_chart(
    model: Any,
    X: Any = None,
    y: Any = None,
    *,
    method: str = "builtin",
    top_k: int | None = 20,
    orient: str = "horizontal",
    error_bars: bool = True,
    show_values: bool = True,
    subtitle: str | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Feature-importance bar chart for an estimator.

    Extracts feature importances from the model via the selected method
    and renders a ranked bar chart. Error bars are drawn from
    ``importance +/- std`` when available.

    Parameters
    ----------
    model : estimator or ModelSource
        Fitted sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``. The estimator must expose
        ``feature_importances_`` or ``coef_`` for ``method="builtin"``.
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        estimator. Also required for ``method="permutation"``.
    y : array-like, optional
        Target vector. Required when ``model`` is a raw
        estimator. Also required for ``method="permutation"``.
    method : {"builtin", "permutation"}, default "builtin"
        Importance extraction method. ``"builtin"`` reads
        ``feature_importances_`` or ``coef_`` directly (``std=0``).
        ``"permutation"`` runs ``sklearn.inspection.permutation_importance``
        and populates ``std`` for the error bars.
    top_k : int or None, default 20
        Maximum number of features to display, ordered by descending
        absolute importance. Pass ``None`` to show all features.
    orient : {"horizontal", "vertical"}, default "horizontal"
        Bar orientation. ``"horizontal"`` maps feature names to the y
        axis; ``"vertical"`` maps them to the x axis.
    error_bars : bool, default True
        When ``True``, draws +/-1 std error bars around each bar. Has no
        visual effect when ``method="builtin"`` (``std=0``).
    show_values : bool, default True
        When ``True``, overlays the numeric importance value as a text
        label on each bar.
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource`` and to
        ``permutation_importance`` when ``method="permutation"``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Feature-importance bar chart ranked by absolute importance.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> fm.importance_chart(RandomForestClassifier().fit(X_train, y_train), X_test, y_test)
    """
    source = _resolve_source(model, X, y, random_state=random_state)
    return _importance_chart_from_source(
        source,
        method=method,
        top_k=top_k,
        orient=orient,
        error_bars=error_bars,
        show_values=show_values,
        subtitle=subtitle,
        random_state=random_state,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def shap_beeswarm_chart(
    model: Any,
    X: Any = None,
    y: Any = None,
    *,
    max_display: int = 20,
    order: str = "abs_mean",
    background: Any = None,
    per_class: bool = False,
    zero_line: bool = True,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """SHAP beeswarm chart -- per-sample SHAP scatter colored by z-scored value.

    ``per_class=True`` on a multi-class classifier facets the chart by
    class. ``per_class=False`` (default) renders a single panel using
    the first class (the only group on regression and binary).

    Parameters
    ----------
    model : estimator or ModelSource
        Fitted sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix.  Required when ``model`` is a raw estimator.
    y : array-like, optional
        Target vector.  Required when ``model`` is a raw estimator.
    max_display : int, default 20
        Maximum number of features to display.
    order : {"abs_mean", "max"}, default "abs_mean"
        Feature ranking criterion.  ``"abs_mean"`` ranks by mean absolute
        SHAP value; ``"max"`` by max absolute SHAP value.
    background : array-like or None, default None
        Background dataset for kernel SHAP explainers.  Ignored for
        tree SHAP.
    per_class : bool, default False
        Facet by class on multi-class classifiers.
    zero_line : bool, default True
        Overlay a dashed vertical reference rule at ``shap_value = 0``
        so the sign of each feature's contribution is immediately
        legible.  Automatically skipped on the multi-panel ``per_class``
        path.  Pass ``False`` to suppress.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        SHAP beeswarm chart.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> fm.shap_beeswarm_chart(GradientBoostingClassifier().fit(X_train, y_train), X_test, y_test)
    """
    source = _resolve_source(model, X, y, random_state=random_state)
    return _shap_beeswarm_chart_from_source(
        source,
        max_display=max_display,
        order=order,
        background=background,
        per_class=per_class,
        zero_line=zero_line,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def shap_bar_chart(
    model: Any,
    X: Any = None,
    y: Any = None,
    *,
    max_display: int = 20,
    order: str = "abs_mean",
    background: Any = None,
    per_class: bool = False,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """SHAP bar chart -- mean absolute SHAP per feature.

    ``per_class=True`` on a multi-class classifier facets the chart by
    class. ``per_class=False`` (default) renders a single panel using
    the first class.

    Parameters
    ----------
    model : estimator or ModelSource
        Fitted sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix.  Required when ``model`` is a raw estimator.
    y : array-like, optional
        Target vector.  Required when ``model`` is a raw estimator.
    max_display : int, default 20
        Maximum number of features to display.
    order : {"abs_mean", "max"}, default "abs_mean"
        Feature ranking criterion.  ``"abs_mean"`` ranks by mean absolute
        SHAP value; ``"max"`` by max absolute SHAP value.
    background : array-like or None, default None
        Background dataset for kernel SHAP explainers.  Ignored for
        tree SHAP.
    per_class : bool, default False
        Facet by class on multi-class classifiers.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        SHAP bar chart (features x mean |SHAP|).

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> fm.shap_bar_chart(GradientBoostingClassifier().fit(X_train, y_train), X_test, y_test)
    """
    source = _resolve_source(model, X, y, random_state=random_state)
    return _shap_bar_chart_from_source(
        source,
        max_display=max_display,
        order=order,
        background=background,
        per_class=per_class,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def shap_waterfall_chart(
    model: Any,
    X: Any = None,
    y: Any = None,
    *,
    sample_idx: int,
    max_display: int = 20,
    order: str = "abs_mean",
    background: Any = None,
    per_class: bool = False,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """SHAP waterfall chart -- cumulative per-feature contributions for one sample.

    ``per_class=True`` on a multi-class classifier facets the chart by
    class (one waterfall panel per class for the same sample).
    ``per_class=False`` (default) renders a single panel using the first
    class.

    Parameters
    ----------
    model : estimator or ModelSource
        Fitted sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix.  Required when ``model`` is a raw estimator.
    y : array-like, optional
        Target vector.  Required when ``model`` is a raw estimator.
    sample_idx : int
        Row index (0-based) of the sample to explain.  Required.
    max_display : int, default 20
        Maximum number of features to display.
    order : {"abs_mean", "max"}, default "abs_mean"
        Feature ranking criterion.  ``"abs_mean"`` ranks by mean absolute
        SHAP value; ``"max"`` by max absolute SHAP value.
    background : array-like or None, default None
        Background dataset for kernel SHAP explainers.  Ignored for
        tree SHAP.
    per_class : bool, default False
        Facet by class on multi-class classifiers.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        SHAP waterfall chart for the sample at ``sample_idx``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> fm.shap_waterfall_chart(
    ...     GradientBoostingClassifier().fit(X_train, y_train), X_test, y_test,
    ...     sample_idx=0,
    ... )
    """
    source = _resolve_source(model, X, y, random_state=random_state)
    return _shap_waterfall_chart_from_source(
        source,
        sample_idx=sample_idx,
        max_display=max_display,
        order=order,
        background=background,
        per_class=per_class,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def shap_chart(
    model: Any,
    X: Any = None,
    y: Any = None,
    *,
    kind: str = "beeswarm",
    max_display: int = 20,
    sample_idx: int | None = None,
    order: str = "abs_mean",
    background: Any = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """SHAP value chart for an estimator.

    Dispatches to one of three chart types based on ``kind``. The
    beeswarm (default) shows per-sample per-feature SHAP values colored
    by z-scored feature magnitude. The bar chart aggregates mean absolute
    SHAP per feature. The waterfall chart shows cumulative contributions
    for a single sample.

    Parameters
    ----------
    model : estimator or ModelSource
        Fitted sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``. SHAP computation requires a tree-based
        or kernel-explainer-compatible model.
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        estimator.
    y : array-like, optional
        Target vector. Required when ``model`` is a raw
        estimator; not used by the SHAP computation itself.
    kind : {"beeswarm", "bar", "waterfall"}, default "beeswarm"
        Chart type. ``"beeswarm"`` renders one point per (sample,
        feature) colored by z-scored feature value. ``"bar"`` renders
        mean(|shap|) per feature as a horizontal bar. ``"waterfall"``
        renders cumulative per-feature contributions for the sample at
        ``sample_idx``.
    max_display : int, default 20
        Maximum number of features to display, selected by the ``order``
        ranking criterion.
    sample_idx : int or None, default None
        Row index (0-based) of the sample to explain. Required when
        ``kind="waterfall"``; ignored for other kinds.
    order : {"abs_mean", "max"}, default "abs_mean"
        Feature ranking criterion across all three kinds. ``"abs_mean"``
        ranks by mean absolute SHAP value; ``"max"`` by max absolute
        SHAP value. Drives both the top-``max_display`` selection and
        the bar/waterfall layout order so all three chart types agree
        on "most important".
    background : array-like or None, default None
        Background dataset for kernel SHAP explainers. When ``None``,
        the full training set is used. Ignored for tree SHAP.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``; SHAP computation itself is
        deterministic for tree explainers.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        SHAP beeswarm, bar, or waterfall chart depending on ``kind``.

    Raises
    ------
    ValueError
        If ``kind="waterfall"`` and ``sample_idx`` is not provided.
    ValueError
        If ``kind`` is not one of ``"beeswarm"``, ``"bar"``,
        ``"waterfall"``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> fm.shap_chart(GradientBoostingClassifier().fit(X_train, y_train), X_test, y_test)
    """
    import warnings

    warnings.warn(
        "shap_chart(kind=...) is deprecated; use shap_beeswarm_chart / "
        "shap_bar_chart / shap_waterfall_chart instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    source = _resolve_source(model, X, y, random_state=random_state)
    if kind == "beeswarm":
        return _shap_beeswarm_chart_from_source(
            source,
            max_display=max_display,
            order=order,
            background=background,
            mark=mark,
            encode=encode,
            properties=properties,
            layers=layers,
            theme=theme,
        )
    if kind == "bar":
        return _shap_bar_chart_from_source(
            source,
            max_display=max_display,
            order=order,
            background=background,
            mark=mark,
            encode=encode,
            properties=properties,
            layers=layers,
            theme=theme,
        )
    if kind == "waterfall":
        if sample_idx is None:
            raise ValueError("shap_chart(kind='waterfall') requires sample_idx=<int>.")
        return _shap_waterfall_chart_from_source(
            source,
            sample_idx=sample_idx,
            max_display=max_display,
            order=order,
            background=background,
            mark=mark,
            encode=encode,
            properties=properties,
            layers=layers,
            theme=theme,
        )
    raise ValueError(f"shap_chart(kind={kind!r}) — expected 'beeswarm', 'bar', or 'waterfall'.")


def pdp_chart(
    model: Any,
    X: Any = None,
    y: Any = None,
    *,
    features: list | None = None,
    grid_resolution: int = 100,
    kind: str = "average",
    ice_alpha: float = 0.2,
    center: bool = False,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Partial-dependence plot (PDP) for one or more features.

    Renders one facet panel per feature, each showing how the model
    prediction changes as that feature varies across its observed range
    while all other features are held at their column means. Supports
    average PDP, individual conditional expectation (ICE), and both
    overlaid.

    Parameters
    ----------
    model : estimator or ModelSource
        Fitted sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        estimator.
    y : array-like, optional
        Target vector. Required when ``model`` is a raw
        estimator; not used by PDP computation.
    features : list of str or int, required
        Column names or integer indices of the features to plot. Each
        feature gets its own facet panel. Must be provided; raises
        ``ValueError`` when ``None``.
    grid_resolution : int, default 100
        Number of evenly spaced grid points along each feature's range.
    kind : {"average", "individual", "both"}, default "average"
        ``"average"`` renders the mean PDP line. ``"individual"`` renders
        one ICE line per sample. ``"both"`` overlays the average curve on
        top of the ICE lines.
    ice_alpha : float, default 0.2
        Opacity of individual ICE lines when ``kind`` is ``"individual"``
        or ``"both"``.
    center : bool, default False
        When ``True``, each curve is anchored at zero by subtracting the
        value at the smallest grid point, making relative changes across
        features directly comparable.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Faceted PDP chart with one panel per feature.

    Raises
    ------
    ValueError
        If ``features`` is ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> fm.pdp_chart(GradientBoostingRegressor().fit(X_train, y_train), X_test, features=["age", "income"])
    """
    _require(
        "pdp_chart",
        "features",
        features,
        hint="pass a list of column names or indices",
    )
    source = _resolve_source(model, X, y, random_state=random_state)
    return _pdp_chart_from_source(
        source,
        list(features),
        grid_resolution=grid_resolution,
        kind=kind,
        ice_alpha=ice_alpha,
        center=center,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )
