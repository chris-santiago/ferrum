"""Shared helper functions used across diagnostic and plot builders.

Shared by the ``ferrum.plots.*`` domain modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import polars as pl

from ferrum._coerce import to_arrow_table
from ferrum._overrides import _apply_overrides

if TYPE_CHECKING:
    from ferrum.chart import Chart
    from ferrum.composition import ConcatChart


def _inject_cook_outliers(
    df: pl.DataFrame,
    *,
    kind: str = "studentized",
    threshold: float | str = "auto",
    x_col: str = "y_pred",
) -> pl.DataFrame:
    """Inject ``_cook_outlier_x`` / ``_cook_outlier_y`` columns.

    For each row whose ``cooks_distance`` exceeds ``threshold``, the
    outlier-coordinate columns hold the ``(x_col, y_col)`` pair; all
    other rows are null so the residual mark's outlier-overlay layer
    skips them. ``x_col`` defaults to ``"y_pred"`` (the residuals-vs-
    fitted view); passing ``"leverage"`` produces an outlier overlay
    keyed on the leverage panel's x axis.

    ``threshold`` accepts:
    - a float — use that value directly.
    - ``"auto"`` — use the conventional ``4 / n`` rule (Hair et al.).
    """
    if df.height == 0 or "cooks_distance" not in df.columns:
        return df
    if threshold == "auto":
        thr = 4.0 / df.height
    else:
        thr = float(threshold)
    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"
    # Polars treats NaN > anything as True (NaN sorts after Infinity), so
    # the comparison must be guarded with `is_not_nan()` to keep non-linear
    # estimators (whose cooks_distance is NaN for every row) from
    # silently flagging every observation as an outlier.
    is_outlier = (
        pl.col("cooks_distance").is_not_nan()
        & pl.col("cooks_distance").is_not_null()
        & (pl.col("cooks_distance") > thr)
    )
    return df.with_columns(
        pl.when(is_outlier).then(pl.col(x_col)).otherwise(None).alias("_cook_outlier_x"),
        pl.when(is_outlier).then(pl.col(y_col)).otherwise(None).alias("_cook_outlier_y"),
    )


def _r2_score(y_true: pl.Series, y_pred: pl.Series) -> float:
    """Coefficient of determination — Schwabish SB3 corner-metrics helper."""
    diff = y_true - y_pred
    ss_res = float((diff**2).sum())
    mean_y = float(y_true.mean())
    ss_tot = float(((y_true - mean_y) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _inject_metrics_corner(
    df: pl.DataFrame,
    *,
    kind: str = "studentized",
) -> pl.DataFrame:
    """Augment a residuals DataFrame with ``_metrics_text`` and ``_metrics_y``.

    Reads ``y_true`` / ``y_pred`` to compute R² / RMSE / MAE, then writes
    the formatted corner string on the single row with the largest
    ``y_pred`` (other rows null, so ``mark_text`` skips them). ``_metrics_y``
    anchors the text to the top of the residual-axis range so the
    annotation lands in the top-right corner of the plot.

    Schwabish SB3 (2026-05-11). Used by both the single-panel residuals
    chart and the residuals-vs-fitted cell of the 4-panel layout; the
    column is harmless on the other panels (their renderers do not read
    it). ``kind`` selects ``studentized_residual`` vs ``residual`` for
    the y-anchor — must match what ``mark_residuals`` will render so the
    text lands at the correct corner.
    """
    if df.height == 0:
        return df
    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"
    from ferrum._metrics_fmt import format_corner_metrics

    diff = df["y_pred"] - df["y_true"]
    r2 = _r2_score(df["y_true"], df["y_pred"])
    rmse = float((diff**2).mean() ** 0.5)
    mae = float(diff.abs().mean())
    corner_text = format_corner_metrics(r2, rmse, mae)

    n = df.height
    anchor_idx = df["y_pred"].arg_max()
    text_col: list[str | None] = [None] * n
    text_col[anchor_idx] = corner_text
    y_col_vals: list[float | None] = [None] * n
    y_col_vals[anchor_idx] = float(df[y_col].max())
    return df.with_columns(
        pl.Series("_metrics_text", text_col, dtype=pl.Utf8),
        pl.Series("_metrics_y", y_col_vals, dtype=pl.Float64),
    )


def _overlay_metrics_corner(chart):
    """Add a same-data ``mark_text`` overlay reading the injected columns.

    Assumes ``chart._data`` already carries ``_metrics_text`` / ``_metrics_y``
    (see :func:`_inject_metrics_corner`). The overlay layer reuses
    ``chart._data`` so ``+`` produces a true layer rather than the
    HConcat fallback.

    Schwabish SB3 (2026-05-11). Returns the original chart unchanged
    when the columns are absent, so callers can invoke unconditionally.
    """
    from ferrum._layer import _Layer

    data = chart._data
    if data is None or "_metrics_text" not in data.columns:
        return chart
    return chart.layer(
        _Layer(
            mark="text",
            encoding={"x": "y_pred", "y": "_metrics_y", "text": "_metrics_text"},
            mark_kwargs={"align": "right", "dx": -4, "dy": 4},
            name="metrics",
        )
    )


def _sort_by(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Sort the frame ascending by `col` so a downstream ``mark_line`` over
    that column draws a monotonic polyline.
    """
    if col not in df.columns:
        return df
    return df.sort(col, nulls_last=True)


def _grid_panels(charts: list, theme: Any = None):
    """Compose up to 4 panels into a grid using Phase 8a hstack/vstack."""
    if len(charts) == 1:
        c = charts[0]
    elif len(charts) == 2:
        c = charts[0] | charts[1]
    elif len(charts) == 3:
        c = (charts[0] | charts[1]) & charts[2]
    else:
        c = (charts[0] | charts[1]) & (charts[2] | charts[3])
    if theme is not None:
        c = c.theme(theme)
    return c


def _color_field_for(df: pl.DataFrame, default: str | None) -> str | None:
    """Return ``'model'`` if a ``model`` column is present (compare-source
    path), otherwise the supplied default (which may be ``None`` for marks
    whose single-model colour field is unset).
    """
    return "model" if "model" in df.columns else default


def _reject_compare(compare: dict | None, *, chart: str, reason: str) -> None:
    """Raise a clear ``ValueError`` when ``compare=`` is passed to a chart whose
    builder cannot render a multi-model :class:`ComparedModelSource`.

    Mirrors the loud rejection ``_resolve_source`` already applies on the
    precomputed path (D-COMPARE-1): the parameter exists for signature
    uniformity across the model-diagnostic family, but a non-``None`` value is
    surfaced as an explicit error rather than silently dropped.
    """
    if compare is not None:
        raise ValueError(f"compare= is not supported for {chart}: {reason}")


def _dedupe_aggregated(df: pl.DataFrame, *group_keys: str) -> pl.DataFrame:
    """Drop per-fold duplicate rows when only the aggregated (mean/lower/upper)
    columns are needed. Sorts ascending by the primary group key so a
    downstream line layer renders a monotonic polyline.
    """
    keep = df.unique(subset=list(group_keys), keep="first", maintain_order=True)
    return keep.sort(list(group_keys), nulls_last=True)


def _inject_metric_into_legend_labels(
    df: pl.DataFrame,
    *,
    color_field: str,
    x_col: str,
    y_col: str,
    metric_fn: Callable[[Any, Any], float],
    metric_name: str,
    curve_filter: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Rename each ``color_field`` value to embed its per-series metric.

    Computes ``metric_fn`` over each color-group's ``(x_col, y_col)`` and
    rewrites the group's legend value to ``"{group} ({metric_name} = {v:.3f})"``,
    so the legend itself carries the AUC/AP value. Shared by the ROC
    (``metric_name="AUC"``, no filter) and PR (``metric_name="AP"``,
    ``curve_filter`` drops null-precision iso-curve sentinel rows) builders;
    each caller's current text/format is reproduced exactly.

    ``curve_filter`` (when given) restricts the rows used to enumerate
    groups and compute the metric, but the rename is always applied to the
    full ``df`` — so any group present only in filtered-out rows is left
    unrenamed, matching the original per-builder behavior.
    """
    import numpy as np

    group_df = curve_filter(df) if curve_filter is not None else df
    groups = group_df[color_field].unique().to_list()
    rename_map: dict[str, str] = {}
    for g in groups:
        subset = group_df.filter(pl.col(color_field) == g)
        x_g = np.asarray(subset[x_col].to_list(), dtype=float)
        y_g = np.asarray(subset[y_col].to_list(), dtype=float)
        value = metric_fn(x_g, y_g)
        rename_map[str(g)] = f"{g} ({metric_name} = {value:.3f})"
    return df.with_columns(pl.col(color_field).cast(pl.Utf8).replace(rename_map).alias(color_field))


def _charts_with_endpoint_labels(
    chart: "Chart",
    *,
    label_field: str,
    x_col: str,
    y_col: str,
) -> "Chart":
    """Append endpoint direct labels to ``chart`` for each ``label_field`` series.

    Thin wrapper over :func:`ferrum._direct_label._direct_label_endpoint`
    that the gain / lift / learning-curve builders share. The underlying
    helper is unchanged, so output is byte-identical to a direct call.
    """
    from ferrum._direct_label import _direct_label_endpoint

    return _direct_label_endpoint(
        chart,
        label_field=label_field,
        x_col=x_col,
        y_col=y_col,
    )


def _should_facet_by_class(df: pl.DataFrame, *, per_class: bool) -> bool:
    """Decide whether a SHAP chart facets by ``class_label``.

    A chart facets only when ``per_class`` is requested *and* the data
    carries more than one class. Shared by the SHAP beeswarm and bar
    builders, which previously inlined ``per_class and
    df["class_label"].n_unique() > 1`` at separate sites.
    """
    return per_class and df["class_label"].n_unique() > 1


def _to_polars(data: Any) -> pl.DataFrame:
    """Coerce any supported input to a ``polars.DataFrame``.

    Backed by :func:`ferrum._coerce.to_arrow_table`, so all input types
    accepted by ``Chart(data=...)`` are accepted here: polars, pyarrow,
    narwhals-compatible (pandas/modin/cuDF/dask/ibis), dict, list, numpy 2D.

    Returns the input unchanged when it is already a ``polars.DataFrame``.
    """
    if isinstance(data, pl.DataFrame):
        return data
    return pl.from_arrow(to_arrow_table(data))


def _coerce_to_polars(data: Any) -> pl.DataFrame:
    """Coerce a polars / pandas / 2D-numeric input into a polars DataFrame.

    A bare 2D numeric array or list-of-lists is auto-named ``col_0, col_1, ...``
    to match the ferrum-wide :func:`ferrum._coerce.to_arrow_table` convention
    (same prefix, same 0-based indexing) used by ``Chart(numpy_array)``: those
    names become the parallel-coordinates axis labels when ``features=None``.
    Every other input type (pyarrow, narwhals frames, dict, ...) delegates to
    :func:`_to_polars` for the widened CDI-backed coercion.
    """
    import numpy as np

    if isinstance(data, pl.DataFrame):
        return data
    if hasattr(data, "to_numpy") and hasattr(data, "columns"):
        return pl.from_pandas(data)
    try:
        arr = np.asarray(data, dtype=np.float64)
    except (ValueError, TypeError):
        return _to_polars(data)
    if arr.ndim == 2:
        return pl.DataFrame({f"col_{j}": arr[:, j].tolist() for j in range(arr.shape[1])})
    return _to_polars(data)


def _validate_choice(
    func_name: str,
    param: str,
    value: Any,
    choices: "frozenset[Any] | set[Any] | tuple[Any, ...] | list[Any]",
) -> None:
    """Raise ``ValueError`` when ``value`` is not in ``choices``.

    Produces a canonical error message across all figure functions::

        {func_name}: {param} must be one of {sorted(choices)}; got {value!r}

    This replaces three inconsistent idioms that existed across the plots
    package (``"must be one of …; got"``, ``"Unknown … . Accepted values:"``,
    and tail-raises with ``"— expected a or b"``).
    """
    if value not in choices:
        raise ValueError(
            f"{func_name}: {param} must be one of {sorted(str(c) for c in choices)}; got {value!r}"
        )


def _warn_deprecated_dispatcher(old_name: str, param_name: str, replacements: str) -> None:
    """Emit the canonical ``DeprecationWarning`` for a split dispatcher shim.

    Both ``shap_chart(kind=...)`` and ``rank_chart(rank=...)`` are deprecated
    dispatchers that warn then forward to their split siblings. This factors
    out the duplicated warn idiom so each shim is a thin validate-then-delegate
    wrapper; the heterogeneous per-kind dispatch stays explicit in each shim::

        {old_name}({param_name}=...) is deprecated; use {replacements} instead.

    ``stacklevel=3`` (this helper -> the shim -> the user) points the warning at
    the user's call site, preserving the pre-refactor behavior where each shim
    called ``warnings.warn(..., stacklevel=2)`` inline.
    """
    import warnings

    warnings.warn(
        f"{old_name}({param_name}=...) is deprecated; use {replacements} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _require(func_name: str, arg_name: str, value: Any, *, hint: str) -> Any:
    """Raise ``ValueError`` when a required figure-function argument is ``None``."""
    if value is None:
        raise ValueError(
            f"{func_name}({arg_name}=...) is required — {hint}.",
        )
    return value


# Sentinel distinguishing "argument omitted" from an explicit ``None`` so that a
# deprecated keyword alias can be ``None`` and still count as supplied.
_UNSET: Any = object()


def _resolve_first_param(
    canonical_value: Any,
    alias_value: Any,
    *,
    canonical_name: str,
    alias_name: str,
    func_name: str,
) -> Any:
    """Resolve a renamed first positional parameter against its deprecated alias.

    The single keyword-alias mechanism shared by every figure function whose
    first positional parameter was renamed to its family-canonical name
    (D-FIRSTPARAM-1). The canonical parameter keeps the positional slot, so
    positional callers are unaffected; the old name is accepted as a deprecated
    keyword whose default is :data:`_UNSET` (so an explicit ``alias=None`` is
    still detected as "supplied").

    Returns the value the function should use. Raises ``TypeError`` when both
    the canonical and alias names are supplied, mirroring Python's own
    "got multiple values for argument" error shape.
    """
    if alias_value is _UNSET:
        return canonical_value
    if canonical_value is not _UNSET:
        raise TypeError(
            f"{func_name}() got both {canonical_name}= and {alias_name}=; "
            f"{alias_name}= is a deprecated alias for {canonical_name}=, "
            "supply only one."
        )
    return alias_value


def _finalize_chart(chart, *, mark=None, encode=None, properties=None, layers=None, theme=None):
    """Apply overrides and optional theme to a chart, then return it.

    Encapsulates the identical 3-4 line closing pattern shared by every
    ``_*_from_source`` builder across the ``ferrum.plots.*`` domain modules.
    """
    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _compose_compare(
    source,
    builder,
    *,
    builder_kwargs: dict,
    resolve: dict[str, str],
    columns: int | None = None,
) -> "ConcatChart":
    """Build one panel per model via ``builder(model_source, **builder_kwargs)``,
    label each panel with its model name, and compose as small multiples.

    The caller must have already confirmed ``isinstance(source,
    ComparedModelSource)`` before calling this helper — it does not re-resolve
    the source.

    Parameters
    ----------
    source : ComparedModelSource
        Multi-model wrapper whose ``.items()`` ``(name, source)`` pairs are iterated.
    builder : callable
        The chart's ``_<name>_chart_from_source`` builder; called once per model
        as ``builder(model_source, **builder_kwargs)``.  May return a ``Chart``
        or any ``_ChartLike`` composite (e.g. a nested ``ConcatChart`` from
        ``pdp`` or ``residuals``).
    builder_kwargs : dict
        Forwarded verbatim to *builder* for every model.
    resolve : dict[str, str]
        Per-channel scale-sharing policy passed through to the outer
        ``ConcatChart`` (e.g. ``{"x": "shared", "y": "shared"}`` for
        supervised aggregates; ``{"x": "independent", "y": "independent"}``
        for unsupervised diagnostics).
    columns : int, optional
        Grid columns for the outer ``ConcatChart``.  Defaults to the number of
        models so all panels appear in a single row.

    Returns
    -------
    ConcatChart
        Small-multiples composition with one labeled panel per model.
    """
    from ferrum.composition import ConcatChart

    children = []
    for name, model_source in source.items():
        child = builder(model_source, **builder_kwargs)
        child = child.properties(title=name)
        children.append(child)

    n_cols = columns if columns is not None else len(children)
    return ConcatChart(*children, columns=n_cols, resolve=resolve)


def _resolve_source(
    model: Any,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    random_state: int | None = None,
    compare: dict[str, Any] | None = None,
) -> Any:
    """Resolve a figure-function input into a ``ModelSource``, ``ComparedModelSource``,
    or ``_PrecomputedSource``.

    Exactly one input path must be active:
    - **Model path**: ``model`` is not ``None``; ``y_true``/``y_pred`` must both be ``None``.
    - **Precomputed path**: both ``y_true`` and ``y_pred`` are not ``None``; ``model`` must be ``None``.
    - **Neither**: ``ValueError``.

    The precomputed path is incompatible with ``compare=``.
    """
    import ferrum
    from ferrum.diagnostics.source import ComparedModelSource

    has_precomputed = y_true is not None or y_pred is not None
    has_model = model is not None

    if has_precomputed:
        if has_model:
            raise ValueError(
                "Supply either a model/source (model=) or precomputed arrays "
                "(y_true=, y_pred=), not both."
            )
        if compare is not None:
            raise ValueError(
                "compare= is not supported with precomputed y_true/y_pred inputs.  "
                "Multi-model comparison requires a fitted model on each path."
            )
        if y_true is None or y_pred is None:
            missing = "y_pred" if y_true is not None else "y_true"
            raise ValueError(
                f"Precomputed path requires both y_true= and y_pred=; {missing}= is missing."
            )
        from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

        return _PrecomputedSource(y_true, y_pred)

    if not has_model:
        raise ValueError(
            "Supply either a fitted model/source (model=) or precomputed arrays (y_true=, y_pred=)."
        )

    if isinstance(model, ComparedModelSource):
        return model
    if compare is not None:
        if not isinstance(compare, dict):
            raise TypeError(
                f"compare= must be dict[str, model] or None; got {type(compare).__name__}."
            )
        models = {"base": model, **compare}
        return ferrum.ModelSource.compare(
            models,
            X,
            y,
            random_state=random_state,
        )
    if isinstance(model, dict):
        return ferrum.ModelSource.compare(
            model,
            X,
            y,
            random_state=random_state,
        )
    if isinstance(model, ferrum.ModelSource):
        return model
    return ferrum.ModelSource(model, X, y, random_state=random_state)


# ---------------------------------------------------------------------------
# _merge_layers — general-purpose layer composer (moved from regression.py)
# ---------------------------------------------------------------------------


def _merge_layers(
    scatter_chart: "Chart",
    fit_chart: "Chart",
    *,
    scatter_name: str | None = None,
    fit_name: str | None = None,
) -> "Chart":
    """Compose a scatter Chart and a fit Chart into a multi-layer Chart.

    Returns a new Chart with ``_layers`` = scatter-layer + fit-layers,
    with transforms accumulated from both inputs.

    Originally defined in ``plots/regression.py``; relocated here because
    it is domain-agnostic and is imported by both ``regression.py`` and
    ``matrix.py``.
    """
    from dataclasses import replace as _replace

    from ferrum._layer import _Layer

    s_resolved = scatter_chart._resolve_pending()
    f_resolved = fit_chart._resolve_pending()

    new = s_resolved._clone()
    new._pending_stat_mark = None

    shared_transforms: list = []
    seen_ids: set = set()
    for t in list(s_resolved._transforms) + list(f_resolved._transforms):
        key = id(t)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        shared_transforms.append(t)

    scatter_layer = _Layer(
        name=scatter_name,
        mark=s_resolved._mark,
        encoding=dict(s_resolved._encoding),
        mark_kwargs=dict(s_resolved._mark_kwargs) if s_resolved._mark_kwargs else None,
        position=s_resolved._position,
    )

    if f_resolved._layers is not None:
        fit_layers = list(f_resolved._layers)
        if fit_name and fit_layers and fit_layers[0].name is None:
            fit_layers[0] = _replace(fit_layers[0], name=fit_name)
    else:
        fit_layers = [
            _Layer(
                name=fit_name,
                mark=f_resolved._mark,
                encoding=dict(f_resolved._encoding),
                mark_kwargs=dict(f_resolved._mark_kwargs) if f_resolved._mark_kwargs else None,
                position=f_resolved._position,
            )
        ]

    new._mark = None
    new._layers = [scatter_layer] + fit_layers
    new._transforms = shared_transforms
    return new
