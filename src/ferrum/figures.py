"""§3.14 Group B figure-level functions.

Each function is a thin facade over a ``_*_chart_from_source`` builder in
``ferrum._diagnostics.charts``. The ``_resolve_source`` helper accepts a
fitted model, an explicit ``ModelSource``, or (10h) a dict of named models
for comparison.
"""
from __future__ import annotations

from typing import Any


def _resolve_source(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    random_state: int | None = None,
    compare: dict[str, Any] | None = None,
) -> Any:
    """Resolve a figure-function input into a ``ModelSource`` or
    ``ComparedModelSource``.

    Dispatch order:
    - If ``model_or_source`` is already a ``ComparedModelSource``, return it.
    - If ``compare`` is a dict, build a ``ComparedModelSource`` over
      ``{"base": model_or_source, **compare}``.
    - If ``model_or_source`` itself is a dict (one-arg compare form), build
      a ``ComparedModelSource`` from it.
    - If ``model_or_source`` is a ``ModelSource``, return it.
    - Otherwise wrap ``model_or_source`` in a fresh ``ModelSource``.
    """
    import ferrum
    from ferrum._diagnostics.source import ComparedModelSource

    if isinstance(model_or_source, ComparedModelSource):
        return model_or_source
    if compare is not None:
        if not isinstance(compare, dict):
            raise TypeError(
                f"compare= must be dict[str, model] or None; got "
                f"{type(compare).__name__}."
            )
        models = {"base": model_or_source, **compare}
        return ferrum.ModelSource.compare(
            models, X, y, random_state=random_state,
        )
    if isinstance(model_or_source, dict):
        return ferrum.ModelSource.compare(
            model_or_source, X, y, random_state=random_state,
        )
    if isinstance(model_or_source, ferrum.ModelSource):
        return model_or_source
    return ferrum.ModelSource(model_or_source, X, y, random_state=random_state)


def residuals_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    kind: str = "studentized",
    cook_threshold: float | str | None = None,
    panels: Any = "auto",
    random_state: int | None = None,
    theme: Any = None,
):
    """Residuals diagnostic chart — see ferrum-spec.md §3.14.

    ``cook_threshold`` highlights observations whose leverage-aware
    Cook's distance exceeds the threshold. Accepts a float (absolute
    threshold), the literal ``"auto"`` (the ``4 / n`` rule), or
    ``None`` (no highlighting). Requires the wrapped estimator to
    expose ``coef_`` so the hat matrix is computable.

    ``panels="auto"`` ships only the residuals-vs-fitted panel in 10a;
    the QQ / scale-location / leverage panels join in 10h. Pass an
    explicit list (e.g. ``panels=["residuals_vs_fitted", "qq"]``) to
    force the multi-panel path.
    """
    from ferrum._diagnostics.charts import _residuals_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    if panels in (None, "single"):
        panel_list: Any = None
    elif panels == "auto":
        panel_list = None  # 10a auto == single; 10h expands this to 2x2
    else:
        panel_list = list(panels)
    return _residuals_chart_from_source(
        source,
        kind=kind,
        cook_threshold=cook_threshold,
        panels=panel_list,
        theme=theme,
    )


# --- 10b: classification curves --------------------------------------


def roc_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    per_class: bool = True,
    average: str | None = "macro",
    annotate_auc: bool = False,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    theme: Any = None,
):
    """ROC curve(s) — see ferrum-spec.md §3.14.

    ``per_class=True`` (default) plots one curve per class; pass ``False``
    along with ``average="macro"`` (or "micro"/"weighted") to plot a single
    averaged curve. ``annotate_auc=True`` is reserved for Phase 10h.
    """
    from ferrum._diagnostics.charts import _roc_chart_from_source
    source = _resolve_source(
        model_or_source, X, y, random_state=random_state, compare=compare,
    )
    return _roc_chart_from_source(
        source,
        per_class=per_class,
        average=average,
        annotate_auc=annotate_auc,
        theme=theme,
    )


def pr_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    per_class: bool = True,
    annotate_ap: bool = False,
    iso_lines: bool = False,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    theme: Any = None,
):
    """Precision-recall curve(s) — see ferrum-spec.md §3.14.

    ``annotate_ap=True`` and ``iso_lines=True`` are reserved for Phase 10h.
    """
    from ferrum._diagnostics.charts import _pr_chart_from_source
    source = _resolve_source(
        model_or_source, X, y, random_state=random_state, compare=compare,
    )
    return _pr_chart_from_source(
        source,
        per_class=per_class,
        annotate_ap=annotate_ap,
        iso_lines=iso_lines,
        theme=theme,
    )


def calibration_chart(
    *model_or_sources: Any,
    X: Any = None,
    y: Any = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    random_state: int | None = None,
    theme: Any = None,
):
    """Calibration (reliability) curve — see ferrum-spec.md §3.14.

    Variadic in ``*model_or_sources``: pass a single model / ``ModelSource``
    for a one-curve diagram, or two or more for an overlay. Multi-model
    inputs may be supplied as separate positional arguments
    (``calibration_chart(model_a, model_b, X=..., y=...)``) or as a single
    dict positional argument
    (``calibration_chart({"a": model_a, "b": model_b}, X=..., y=...)``).
    The dict form preserves user-supplied model names; the positional form
    auto-names them ``"model_0"``, ``"model_1"``, etc.
    """
    if len(model_or_sources) == 0:
        raise TypeError(
            "calibration_chart requires at least one model or ModelSource"
        )
    from ferrum._diagnostics.charts import _calibration_chart_from_source
    if len(model_or_sources) == 1:
        source = _resolve_source(
            model_or_sources[0], X, y, random_state=random_state,
        )
    else:
        # Multiple positional inputs → build a ComparedModelSource. When
        # the inputs are already ModelSource instances, wrap them directly
        # (X/y are already bound on each source). When they're raw fitted
        # models, build new ModelSources sharing the supplied X/y via
        # ModelSource.compare.
        import ferrum
        from ferrum._diagnostics.source import ComparedModelSource

        if all(isinstance(m, ferrum.ModelSource) for m in model_or_sources):
            sources = {
                f"model_{i}": m for i, m in enumerate(model_or_sources)
            }
            source = ComparedModelSource(sources)
        else:
            models = {
                f"model_{i}": m for i, m in enumerate(model_or_sources)
            }
            source = _resolve_source(
                models, X, y, random_state=random_state,
            )
    return _calibration_chart_from_source(
        source, n_bins=n_bins, strategy=strategy, theme=theme,
    )


def gain_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    random_state: int | None = None,
    theme: Any = None,
):
    """Cumulative-gain curve — see ferrum-spec.md §3.14."""
    from ferrum._diagnostics.charts import _gain_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _gain_chart_from_source(source, theme=theme)


def lift_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    random_state: int | None = None,
    theme: Any = None,
):
    """Lift curve — see ferrum-spec.md §3.14."""
    from ferrum._diagnostics.charts import _lift_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _lift_chart_from_source(source, theme=theme)


# --- 10c: classification matrices ------------------------------------


def confusion_matrix_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    normalize: str | None = "true",
    annotate: bool = True,
    random_state: int | None = None,
    theme: Any = None,
):
    """Confusion matrix heatmap — see ferrum-spec.md §3.14.

    Ordinal heatmap of (actual, predicted) cell values with optional
    per-cell text labels via the Phase 10c text channel. ``normalize``
    follows sklearn semantics: ``None`` for raw counts,
    ``"true"``/``"pred"``/``"all"`` for normalized fractions.
    """
    from ferrum._diagnostics.charts import _confusion_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _confusion_chart_from_source(
        source, normalize=normalize, annotate=annotate, theme=theme,
    )


def class_prediction_error_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    normalize: bool = False,
    random_state: int | None = None,
    theme: Any = None,
):
    """Class prediction-error stacked bar — see ferrum-spec.md §3.14.

    One bar per predicted class, segments stacked by actual class. Pass
    ``normalize=True`` for a per-bar 100% stack (relative composition).
    """
    from ferrum._diagnostics.charts import (
        _class_prediction_error_chart_from_source,
    )
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _class_prediction_error_chart_from_source(
        source, normalize=normalize, theme=theme,
    )


# --- 10d: feature importance / SHAP / PDP ----------------------------


def importance_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    method: str = "builtin",
    top_k: int | None = 20,
    orient: str = "horizontal",
    error_bars: bool = True,
    random_state: int | None = None,
    theme: Any = None,
):
    """Feature-importance chart — see ferrum-spec.md §3.14.

    ``method="builtin"`` reads ``feature_importances_`` / ``coef_`` from
    the wrapped estimator (std=0). ``method="permutation"`` runs sklearn's
    ``permutation_importance`` and populates ``std`` for the error bars.
    """
    from ferrum._diagnostics.charts import _importance_chart_from_source

    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _importance_chart_from_source(
        source,
        method=method,
        top_k=top_k,
        orient=orient,
        error_bars=error_bars,
        random_state=random_state,
        theme=theme,
    )


def shap_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    kind: str = "beeswarm",
    max_display: int = 20,
    sample_idx: int | None = None,
    order: str = "abs_mean",
    background: Any = None,
    random_state: int | None = None,
    theme: Any = None,
):
    """SHAP value chart dispatcher — see ferrum-spec.md §3.14.

    ``kind="beeswarm"`` (default) renders one point per (sample, feature)
    with shap_value on x and feature on ordinal y, colored by z-scored
    feature value. ``kind="bar"`` aggregates mean(|shap|) per feature.
    ``kind="waterfall"`` requires ``sample_idx`` and renders cumulative
    per-feature contributions for that one sample.
    """
    from ferrum._diagnostics.charts import (
        _shap_bar_chart_from_source,
        _shap_beeswarm_chart_from_source,
        _shap_waterfall_chart_from_source,
    )

    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    if kind == "beeswarm":
        return _shap_beeswarm_chart_from_source(
            source,
            max_display=max_display,
            order=order,
            background=background,
            theme=theme,
        )
    if kind == "bar":
        return _shap_bar_chart_from_source(
            source,
            max_display=max_display,
            background=background,
            theme=theme,
        )
    if kind == "waterfall":
        if sample_idx is None:
            raise ValueError(
                "shap_chart(kind='waterfall') requires sample_idx=<int>."
            )
        return _shap_waterfall_chart_from_source(
            source,
            sample_idx=sample_idx,
            max_display=max_display,
            background=background,
            theme=theme,
        )
    raise ValueError(
        f"shap_chart(kind={kind!r}) — expected 'beeswarm', 'bar', or 'waterfall'."
    )


def pdp_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    features: list | None = None,
    grid_resolution: int = 100,
    kind: str = "average",
    ice_alpha: float = 0.2,
    center: bool = False,
    random_state: int | None = None,
    theme: Any = None,
):
    """Partial-dependence chart — see ferrum-spec.md §3.14.

    Pass ``features`` as a list of column names or indices. ``kind="average"``
    is the only supported value today; ``kind="individual"``/``"both"``
    require the deferred ``detail`` encoding channel.
    """
    from ferrum._diagnostics.charts import _pdp_chart_from_source

    if features is None:
        raise ValueError(
            "pdp_chart requires features=<list of column names or indices>."
        )
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _pdp_chart_from_source(
        source,
        list(features),
        grid_resolution=grid_resolution,
        kind=kind,
        ice_alpha=ice_alpha,
        center=center,
        theme=theme,
    )


def discrimination_threshold_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    n_thresholds: int = 50,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
    cv: Any = None,
    threshold_line: bool = False,
    random_state: int | None = None,
    theme: Any = None,
):
    """Discrimination-threshold sweep — see ferrum-spec.md §3.14.

    ``threshold_line=True`` (rule at the F1-best threshold) is reserved
    for Phase 10h.
    """
    from ferrum._diagnostics.charts import (
        _discrimination_threshold_chart_from_source,
    )
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _discrimination_threshold_chart_from_source(
        source,
        n_thresholds=n_thresholds,
        metrics=metrics,
        cv=cv,
        threshold_line=threshold_line,
        theme=theme,
    )


def learning_curve_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    cv: int = 5,
    scoring: Any = None,
    train_sizes: Any = None,
    ci_style: str = "band",
    random_state: int | None = None,
    theme: Any = None,
):
    """Learning curve — see ferrum-spec.md §3.14."""
    from ferrum._diagnostics.charts import _learning_curve_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _learning_curve_chart_from_source(
        source,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        ci_style=ci_style,
        theme=theme,
    )


def validation_curve_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    param: str = "alpha",
    values: Any = None,
    cv: int = 5,
    scoring: Any = None,
    log_scale: Any = "auto",
    ci_style: str = "band",
    random_state: int | None = None,
    theme: Any = None,
):
    """Validation curve — see ferrum-spec.md §3.14.

    ``param`` and ``values`` are required; the defaults exist only to
    keep the signature ergonomic for ``param="alpha"`` Ridge sweeps.
    """
    if values is None:
        raise ValueError(
            "validation_curve_chart(values=...) is required — pass an explicit "
            "list of values to sweep for the given param."
        )
    from ferrum._diagnostics.charts import _validation_curve_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _validation_curve_chart_from_source(
        source, param, values,
        cv=cv, scoring=scoring,
        log_scale=log_scale, ci_style=ci_style, theme=theme,
    )


def cv_scores_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    cv: int = 5,
    scoring: Any = None,
    kind: str = "box",
    split: str = "both",
    random_state: int | None = None,
    theme: Any = None,
):
    """Per-fold CV-score chart — see ferrum-spec.md §3.14."""
    from ferrum._diagnostics.charts import _cv_scores_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _cv_scores_chart_from_source(
        source, cv=cv, scoring=scoring,
        kind=kind, split=split, theme=theme,
    )


def alpha_selection_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    alphas: Any = None,
    cv: int = 5,
    scoring: Any = None,
    log_scale: bool = True,
    highlight_best: bool = True,
    random_state: int | None = None,
    theme: Any = None,
):
    """Regularization-strength selection chart — see ferrum-spec.md §3.14."""
    if alphas is None:
        raise ValueError(
            "alpha_selection_chart(alphas=...) is required — pass an explicit "
            "list of regularization-strength values to sweep."
        )
    from ferrum._diagnostics.charts import _alpha_selection_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _alpha_selection_chart_from_source(
        source, alphas,
        cv=cv, scoring=scoring,
        log_scale=log_scale, highlight_best=highlight_best, theme=theme,
    )


# --- 10f: clustering / manifold / decision boundary ------------------


def pca_scree_chart(
    model_or_source: Any,
    X: Any = None,
    *,
    n_components: int | None = None,
    cumulative_line: bool = True,
    threshold: float | None = 0.95,
    random_state: int | None = None,
    theme: Any = None,
):
    """PCA scree chart — see ferrum-spec.md §3.14.

    ``threshold`` draws a horizontal reference at the named cumulative-
    variance level (default 0.95). Pass ``threshold=None`` to omit.
    """
    from ferrum._diagnostics.charts import _pca_scree_chart_from_source
    source = _resolve_source(model_or_source, X, None, random_state=random_state)
    return _pca_scree_chart_from_source(
        source,
        n_components=n_components,
        cumulative_line=cumulative_line,
        threshold=threshold,
        theme=theme,
    )


def cluster_diagnostics(
    X: Any,
    *,
    ks: Any,
    method: str = "kmeans",
    n_init: int = 10,
    random_state: int | None = None,
    theme: Any = None,
):
    """Elbow + silhouette per k for a given clusterer — see ferrum-spec.md §3.14.

    Fits one ``KMeans(n_clusters=k)`` per k and emits a side-by-side
    HConcatChart: distortion (inertia) vs k, silhouette mean vs k.
    Distinct from the other figure functions in that it sweeps the
    model class itself rather than wrapping a single ``ModelSource``.
    """
    from ferrum._diagnostics.deps import require_sklearn
    require_sklearn("cluster_diagnostics")
    if method != "kmeans":
        raise NotImplementedError(
            f"cluster_diagnostics(method={method!r}) — Phase 10f ships "
            "'kmeans' only; other estimators land in 10h."
        )
    import polars as pl
    import numpy as np
    import ferrum
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    seed = 0 if random_state is None else int(random_state)
    rows = []
    for k in ks:
        m = KMeans(
            n_clusters=int(k), n_init=int(n_init), random_state=seed,
        ).fit(X_np)
        rows.append({
            "k": int(k),
            "inertia": float(m.inertia_),
            "silhouette": float(silhouette_score(X_np, m.labels_)),
        })
    df = pl.DataFrame(rows)
    elbow = ferrum.Chart(df).mark_line().encode(x="k", y="inertia")
    sil = ferrum.Chart(df).mark_line().encode(x="k", y="silhouette")
    chart = elbow | sil
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def intercluster_distance_chart(
    model_or_source: Any,
    X: Any = None,
    *,
    k: int | None = None,
    method: str = "mds",
    random_state: int | None = None,
    theme: Any = None,
):
    """Cluster-center 2D embedding — see ferrum-spec.md §3.14."""
    from ferrum._diagnostics.charts import (
        _intercluster_distance_chart_from_source,
    )
    source = _resolve_source(model_or_source, X, None, random_state=random_state)
    if k is None:
        if hasattr(source._model, "n_clusters"):
            k = int(source._model.n_clusters)
        elif hasattr(source._model, "cluster_centers_"):
            k = int(source._model.cluster_centers_.shape[0])
        else:
            raise ValueError(
                "intercluster_distance_chart(k=...) is required when the "
                "wrapped model exposes neither n_clusters nor "
                "cluster_centers_."
            )
    return _intercluster_distance_chart_from_source(
        source, k=int(k), method=method, theme=theme,
    )


def rank_chart(
    data_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    rank: str = "2d",
    algorithm: str | None = None,
    top_k: int | None = None,
    annot: bool = True,
    orient: str = "horizontal",
    color_field: str | None = None,
    random_state: int | None = None,
    theme: Any = None,
):
    """Feature-ranking chart — see ferrum-spec.md §3.14.

    Accepts either a ``ModelSource`` / fitted model + X (covariance
    routes through ``ModelSource.rank1d(algorithm='covariance')``,
    which requires ``y``) or a raw DataFrame / 2D array of features
    (no model needed).

    ``rank='1d'`` produces a univariate ranking bar chart; ``rank='2d'``
    produces a pairwise heatmap. Algorithm defaults: ``shapiro`` for 1d,
    ``pearson`` for 2d.
    """
    import ferrum

    if rank == "1d":
        algo = algorithm or "shapiro"
        if isinstance(data_or_source, ferrum.ModelSource):
            df = data_or_source.rank1d(algorithm=algo)
        elif algo == "covariance":
            source = _resolve_source(
                data_or_source, X, y, random_state=random_state,
            )
            df = source.rank1d(algorithm=algo)
        else:
            from ferrum._diagnostics.stats import rank1d_compute
            data = data_or_source if X is None else X
            df = rank1d_compute(data, algorithm=algo)
        from ferrum._diagnostics.charts import _rank1d_chart_from_dataframe
        return _rank1d_chart_from_dataframe(
            df, algorithm=algo, orient=orient, top_k=top_k,
            color_field=color_field, theme=theme,
        )
    if rank == "2d":
        algo = algorithm or "pearson"
        if isinstance(data_or_source, ferrum.ModelSource):
            df = data_or_source.rank2d(algorithm=algo)
        else:
            from ferrum._diagnostics.stats import rank2d_compute
            data = data_or_source if X is None else X
            df = rank2d_compute(data, algorithm=algo)
        from ferrum._diagnostics.charts import _rank2d_chart_from_dataframe
        return _rank2d_chart_from_dataframe(
            df, algorithm=algo, annot=annot, theme=theme,
        )
    raise ValueError(f"rank_chart(rank={rank!r}) — expected '1d' or '2d'.")


def parallel_coordinates_chart(
    data: Any,
    *,
    features: list[str] | None = None,
    hue: str | None = None,
    rescale: str | None = "minmax",
    alpha: float = 0.5,
    theme: Any = None,
):
    """Parallel coordinates chart — see ferrum-spec.md §3.14.

    ``data`` is a polars / pandas DataFrame or a 2D numpy array of
    features. ``hue`` is the column name to color samples by (typically
    the target / cluster id); pass None for monochrome.
    """
    from ferrum._diagnostics.charts import _parallel_coords_chart_from_dataframe
    return _parallel_coords_chart_from_dataframe(
        data, features=features, hue=hue, rescale=rescale,
        alpha=alpha, theme=theme,
    )


def decision_boundary_chart(
    model: Any,
    X: Any,
    y: Any = None,
    *,
    features: tuple = (0, 1),
    grid_resolution: int = 200,
    proba: bool = False,
    scatter: bool = False,
    random_state: int | None = None,
    theme: Any = None,
):
    """Decision-boundary heatmap — see ferrum-spec.md §3.14.

    Requires exactly 2 features (passed via ``features=(i, j)``); other
    features are fixed at their column means while the grid sweeps the
    two named columns. ``proba=True`` uses ``predict_proba[:, 1]`` when
    available; otherwise ``predict`` returns the class index colored
    discretely.
    """
    source = _resolve_source(model, X, y, random_state=random_state)
    from ferrum._diagnostics.charts import _decision_boundary_chart_from_source
    return _decision_boundary_chart_from_source(
        source,
        features=tuple(features),
        grid_resolution=int(grid_resolution),
        proba=bool(proba),
        scatter=bool(scatter),
        theme=theme,
    )

