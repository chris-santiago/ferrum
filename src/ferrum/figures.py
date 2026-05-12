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
    """Residuals diagnostic chart for a regression estimator.

    Plots residuals vs. fitted values. Optional Cook's distance
    highlighting marks observations whose leverage-adjusted influence
    exceeds a user-supplied threshold. Multi-panel layout (QQ,
    scale-location, leverage) is reserved for Phase 10h.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        A fitted sklearn-compatible regression estimator, an explicit
        ``ferrum.ModelSource``, or a dict of named estimators.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y : array-like, optional
        Target vector. Required when ``model_or_source`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    kind : {"studentized", "raw"}, default "studentized"
        Residual type to plot on the y axis. ``"studentized"`` uses
        internally studentized residuals; ``"raw"`` uses raw residuals.
    cook_threshold : float, "auto", or None, default None
        Threshold for Cook's distance outlier highlighting. A float is
        used as an absolute cutoff; ``"auto"`` applies the ``4 / n``
        rule (Hair et al.); ``None`` disables highlighting. Cook's
        distance is only defined for estimators that expose ``coef_``;
        non-linear models surface ``NaN`` and produce no outliers.
    panels : "auto", None, "single", or list of str, default "auto"
        Panel selection. ``"auto"`` and ``None`` ship the single
        residuals-vs-fitted panel (Phase 10a behavior). Pass an
        explicit list such as ``["residuals_vs_fitted", "qq"]`` to
        force a multi-panel layout (Phase 10h).
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``; does not affect deterministic
        residuals computation.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Residuals-vs-fitted scatter chart with optional Cook's-distance
        outlier overlay.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import Ridge
    >>> fm.residuals_chart(Ridge().fit(X_train, y_train), X_test, y_test)
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
    """ROC curve chart for a classifier.

    Plots true-positive rate vs. false-positive rate, one curve per
    class (default) or a single macro/micro/weighted-averaged curve.
    Supports multi-model comparison via ``compare=``.

    Parameters
    ----------
    model_or_source : estimator, ModelSource, or dict of str -> estimator
        Fitted sklearn-compatible classifier, an explicit
        ``ferrum.ModelSource``, or a dict of named estimators for
        comparison. When a dict is passed, each estimator is evaluated
        and curves are overlaid.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels. Required when ``model_or_source`` is a raw
        estimator.
    per_class : bool, default True
        When ``True``, one ROC curve per class is drawn using the
        one-vs-rest scheme. When ``False``, a single curve averaged
        per ``average`` is drawn.
    average : {"macro", "micro", "weighted"} or None, default "macro"
        Averaging method used when ``per_class=False``. Ignored when
        ``per_class=True``.
    annotate_auc : bool, default False
        When ``True``, injects one text label per class showing the AUC
        value to 3 decimal places, anchored in the lower-right corner
        of the plot.
    compare : dict of str -> estimator or None, default None
        Additional estimators to overlay. Keys become model labels.
        ``model_or_source`` is treated as the base model (label
        ``"base"``).
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        ROC curve chart with one line per class (or averaged curve).

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.roc_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)
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
    average: str | None = "macro",
    annotate_ap: bool = False,
    iso_lines: bool = False,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    theme: Any = None,
):
    """Precision-recall curve chart for a classifier.

    Plots precision vs. recall, one curve per class (default) or a single
    averaged curve. Both axes are pinned to ``[0, 1.05]`` so curves
    render against the full precision-recall space (same convention as
    sklearn and yellowbrick). Supports multi-model comparison via
    ``compare=``.

    Parameters
    ----------
    model_or_source : estimator, ModelSource, or dict of str -> estimator
        Fitted sklearn-compatible classifier, an explicit
        ``ferrum.ModelSource``, or a dict of named estimators for
        comparison.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels. Required when ``model_or_source`` is a raw
        estimator.
    per_class : bool, default True
        When ``True``, one PR curve per class is drawn using the
        one-vs-rest scheme. When ``False``, a single curve averaged per
        ``average`` is drawn. Binary classifiers always render a single
        curve regardless.
    average : {"macro", "micro", "weighted"} or None, default "macro"
        Averaging method used when ``per_class=False``. Ignored when
        ``per_class=True``. Macro and weighted variants interpolate
        per-class precision over a shared recall grid; micro ravels the
        binarized labels into a single curve.
    annotate_ap : bool, default False
        When ``True``, injects one text label per class near the lower-
        right corner of the plot showing average precision (AP) to 3
        decimal places.
    iso_lines : bool, default False
        When ``True``, overlays F-score iso-contours at
        F={0.2, 0.4, 0.6, 0.8} so users can read off combined
        precision-recall quality directly from the chart.
    compare : dict of str -> estimator or None, default None
        Additional estimators to overlay. Keys become model labels.
        ``model_or_source`` is treated as the base model (label
        ``"base"``).
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Precision-recall curve chart with one line per class (or an
        averaged summary curve).

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.pr_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)
    """
    from ferrum._diagnostics.charts import _pr_chart_from_source
    source = _resolve_source(
        model_or_source, X, y, random_state=random_state, compare=compare,
    )
    return _pr_chart_from_source(
        source,
        per_class=per_class,
        average=average,
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
    """Calibration (reliability) curve for one or more classifiers.

    Plots mean predicted probability vs. fraction of positives in each
    bin, following sklearn's ``calibration_curve`` convention. Accepts
    one or more models; multi-model inputs are overlaid on a single
    chart.

    Parameters
    ----------
    *model_or_sources : estimator, ModelSource, or dict
        One or more fitted classifiers or ``ModelSource`` objects. A
        single dict positional argument is treated as a named-model map
        (keys become model labels). Multiple positional arguments are
        auto-named ``"model_0"``, ``"model_1"``, etc. At least one
        positional argument is required.
    X : array-like, optional
        Feature matrix. Required when positional arguments are raw
        estimators; ignored when they are ``ModelSource`` instances.
    y : array-like, optional
        True binary labels. Required when positional arguments are raw
        estimators.
    n_bins : int, default 10
        Number of probability bins for the reliability diagram.
    strategy : {"uniform", "quantile"}, default "uniform"
        Binning strategy forwarded to ``sklearn.calibration.calibration_curve``.
        ``"uniform"`` uses equally-spaced bins; ``"quantile"`` uses
        equal-frequency bins.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Reliability diagram with one curve per model plus a perfect-
        calibration diagonal reference.

    Raises
    ------
    TypeError
        If no positional arguments are provided.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.calibration_chart(LogisticRegression().fit(X_train, y_train), X=X_test, y=y_test)
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
    """Cumulative-gain curve for a classifier.

    Plots the fraction of positive cases captured vs. the fraction of
    samples scored, one curve per class. Useful for evaluating the
    benefit of targeting a top-ranked subset.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels. Required when ``model_or_source`` is a raw
        estimator.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Cumulative-gain curve with one line per class.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.gain_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)
    """
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
    """Lift curve for a classifier.

    Plots the ratio of positive-hit rate in the scored top-n vs. random
    baseline, one curve per class. Values above 1 indicate the model
    outperforms random selection at that depth.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels. Required when ``model_or_source`` is a raw
        estimator.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Lift curve with one line per class.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.lift_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)
    """
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
    """Confusion matrix heatmap for a classifier.

    Renders an ordinal heatmap of (actual class, predicted class) cell
    values with optional per-cell text annotations. Normalization
    follows sklearn's ``confusion_matrix`` conventions.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels. Required when ``model_or_source`` is a raw
        estimator.
    normalize : {"true", "pred", "all"} or None, default "true"
        Normalization scheme for cell values. ``None`` shows raw
        counts; ``"true"`` normalizes over actual classes (rows);
        ``"pred"`` normalizes over predicted classes (columns);
        ``"all"`` normalizes over the total number of samples.
    annotate : bool, default True
        When ``True``, overlays the numeric cell value as a text
        label in each cell.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Confusion matrix heatmap with optional cell annotations.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.confusion_matrix_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)
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
    """Class prediction-error stacked bar chart for a classifier.

    One bar per predicted class, stacked by actual class so misclassified
    segments are visually distinct. The data source is the unnormalized
    confusion matrix reshaped to long form.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True class labels. Required when ``model_or_source`` is a raw
        estimator.
    normalize : bool, default False
        When ``True``, each bar is normalized to 100% (relative
        composition). When ``False``, bars show raw sample counts.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Stacked bar chart with one bar per predicted class.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.class_prediction_error_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)
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
    """Feature-importance bar chart for an estimator.

    Extracts feature importances from the model via the selected method
    and renders a ranked bar chart. Error bars are drawn from
    ``importance ± std`` when available.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``. The estimator must expose
        ``feature_importances_`` or ``coef_`` for ``method="builtin"``.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator. Also required for ``method="permutation"``.
    y : array-like, optional
        Target vector. Required when ``model_or_source`` is a raw
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
        When ``True``, draws ±1 std error bars around each bar. Has no
        visual effect when ``method="builtin"`` (``std=0``).
    random_state : int or None, default None
        Seed forwarded to ``ModelSource`` and to
        ``permutation_importance`` when ``method="permutation"``.
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


def shap_beeswarm_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    max_display: int = 20,
    order: str = "abs_mean",
    background: Any = None,
    random_state: int | None = None,
    theme: Any = None,
):
    """SHAP beeswarm chart — per-sample SHAP scatter colored by z-scored value.

    See :func:`shap_chart` for shared parameter docstring (this function
    is the dedicated sibling for ``kind="beeswarm"``).

    Returns
    -------
    Chart
        SHAP beeswarm chart.
    """
    from ferrum._diagnostics.charts import _shap_beeswarm_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _shap_beeswarm_chart_from_source(
        source,
        max_display=max_display,
        order=order,
        background=background,
        theme=theme,
    )


def shap_bar_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    max_display: int = 20,
    order: str = "abs_mean",
    background: Any = None,
    random_state: int | None = None,
    theme: Any = None,
):
    """SHAP bar chart — mean absolute SHAP per feature.

    See :func:`shap_chart` for shared parameter docstring (this function
    is the dedicated sibling for ``kind="bar"``).

    Returns
    -------
    Chart
        SHAP bar chart.
    """
    from ferrum._diagnostics.charts import _shap_bar_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _shap_bar_chart_from_source(
        source,
        max_display=max_display,
        order=order,
        background=background,
        theme=theme,
    )


def shap_waterfall_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    sample_idx: int,
    max_display: int = 20,
    order: str = "abs_mean",
    background: Any = None,
    random_state: int | None = None,
    theme: Any = None,
):
    """SHAP waterfall chart — cumulative per-feature contributions for one sample.

    See :func:`shap_chart` for shared parameter docstring (this function
    is the dedicated sibling for ``kind="waterfall"``).  ``sample_idx``
    is required.

    Returns
    -------
    Chart
        SHAP waterfall chart for the sample at ``sample_idx``.
    """
    from ferrum._diagnostics.charts import _shap_waterfall_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _shap_waterfall_chart_from_source(
        source,
        sample_idx=sample_idx,
        max_display=max_display,
        order=order,
        background=background,
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
    """SHAP value chart for an estimator.

    Dispatches to one of three chart types based on ``kind``. The
    beeswarm (default) shows per-sample per-feature SHAP values colored
    by z-scored feature magnitude. The bar chart aggregates mean absolute
    SHAP per feature. The waterfall chart shows cumulative contributions
    for a single sample.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``. SHAP computation requires a tree-based
        or kernel-explainer-compatible model.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        Target vector. Required when ``model_or_source`` is a raw
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
        DeprecationWarning, stacklevel=2,
    )
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
            order=order,
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
            order=order,
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
    """Partial-dependence plot (PDP) for one or more features.

    Renders one facet panel per feature, each showing how the model
    prediction changes as that feature varies across its observed range
    while all other features are held at their column means. Supports
    average PDP, individual conditional expectation (ICE), and both
    overlaid.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        Target vector. Required when ``model_or_source`` is a raw
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
    """Discrimination-threshold sweep chart for a binary classifier.

    Plots multiple classification metrics (precision, recall, F1,
    queue rate) as a function of the decision threshold, allowing users
    to select an operating point that balances competing objectives.
    The underlying data is unpivoted to long form for multi-metric
    rendering.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted sklearn-compatible binary classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        True binary labels. Required when ``model_or_source`` is a raw
        estimator.
    n_thresholds : int, default 50
        Number of evenly spaced threshold values in ``[0, 1]`` to
        evaluate.
    metrics : tuple of str, default ("precision", "recall", "f1", "queue_rate")
        Metric names to plot. Each must be a column in the threshold
        sweep DataFrame.
    cv : int, cross-validator, or None, default None
        When provided, the threshold sweep is computed via cross-
        validation rather than a single train/test split.
    threshold_line : bool, default False
        When ``True``, injects a vertical reference rule at the
        threshold that maximises F1.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Multi-metric line chart over the threshold range.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> fm.discrimination_threshold_chart(LogisticRegression().fit(X_train, y_train), X_test, y_test)
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
    """Learning curve chart showing score vs. training set size.

    Plots cross-validated train and validation scores as training size
    grows, revealing overfitting, underfitting, and data-hunger. The CI
    band is drawn from per-fold score variance.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted (or unfitted) sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        Target vector. Required when ``model_or_source`` is a raw
        estimator.
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str, callable, or None, default None
        Scoring metric forwarded to
        ``sklearn.model_selection.learning_curve``. When ``None``,
        the estimator's default scorer is used.
    train_sizes : array-like or None, default None
        Relative or absolute training sizes to evaluate. When ``None``,
        sklearn's default ``np.linspace(0.1, 1.0, 5)`` is used.
    ci_style : {"band", "errorbar"}, default "band"
        Visual style of the confidence interval. ``"band"`` draws a
        shaded ribbon; ``"errorbar"`` draws error bars.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Learning curve with train and validation score lines plus CI.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.svm import SVC
    >>> fm.learning_curve_chart(SVC(), X_train, y_train, cv=5)
    """
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
    """Validation curve chart: score vs. a single hyperparameter value.

    Sweeps one hyperparameter over a supplied value range and plots
    cross-validated train and validation scores, revealing the bias-
    variance tradeoff for that parameter. The x axis is log-scaled
    automatically when the value range spans more than two orders of
    magnitude.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted (or unfitted) sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        Target vector. Required when ``model_or_source`` is a raw
        estimator.
    param : str, default "alpha"
        Name of the hyperparameter to sweep, passed to
        ``sklearn.model_selection.validation_curve`` as
        ``param_name``.
    values : array-like, required
        Values of ``param`` to evaluate. Must be provided; raises
        ``ValueError`` when ``None``.
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str, callable, or None, default None
        Scoring metric. When ``None``, the estimator's default scorer
        is used.
    log_scale : bool or "auto", default "auto"
        Whether to use a log scale on the x axis. ``"auto"`` enables
        log scale when ``max(values) / min(non-zero values) > 100``.
    ci_style : {"band", "errorbar"}, default "band"
        Visual style of the confidence interval.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Validation curve with train and validation score lines plus CI.

    Raises
    ------
    ValueError
        If ``values`` is ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import Ridge
    >>> fm.validation_curve_chart(Ridge(), X_train, y_train, param="alpha", values=[0.01, 0.1, 1, 10])
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
    """Per-fold cross-validation score distribution chart.

    Visualizes the distribution of scores across folds for train and/or
    validation splits, making variance and consistency across folds
    immediately visible.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted (or unfitted) sklearn-compatible estimator or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        Target vector. Required when ``model_or_source`` is a raw
        estimator.
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str, callable, or None, default None
        Scoring metric. When ``None``, the estimator's default scorer
        is used.
    kind : {"box", "strip", "bar"}, default "box"
        Chart type. ``"box"`` renders a box-and-whisker per split;
        ``"strip"`` renders individual fold points; ``"bar"`` renders
        the mean score per split as a bar (pre-aggregated).
    split : {"both", "train", "test"}, default "both"
        Which CV split to show. ``"both"`` shows train and validation
        side by side; ``"train"`` or ``"test"`` restricts to one split.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Per-fold CV score distribution chart.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> fm.cv_scores_chart(RandomForestClassifier(), X_train, y_train, cv=10)
    """
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
    """Regularization-strength (alpha) selection chart.

    Plots cross-validated mean score as a function of regularization
    strength, helping users identify the optimal alpha for penalized
    estimators such as Ridge, Lasso, and ElasticNet.

    Parameters
    ----------
    model_or_source : estimator or ModelSource
        Fitted (or unfitted) sklearn-compatible penalized estimator or
        an explicit ``ferrum.ModelSource``. The estimator must accept an
        ``alpha`` constructor parameter.
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        estimator.
    y : array-like, optional
        Target vector. Required when ``model_or_source`` is a raw
        estimator.
    alphas : array-like, required
        Regularization-strength values to sweep. Must be provided;
        raises ``ValueError`` when ``None``.
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str, callable, or None, default None
        Scoring metric. When ``None``, the estimator's default scorer
        is used.
    log_scale : bool, default True
        When ``True``, the alpha axis is log-scaled.
    highlight_best : bool, default True
        When ``True``, injects a vertical reference rule at the alpha
        that maximises mean CV score.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        CV-score-vs-alpha line chart with optional best-alpha rule.

    Raises
    ------
    ValueError
        If ``alphas`` is ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import Ridge
    >>> fm.alpha_selection_chart(Ridge(), X_train, y_train, alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    """
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
    """PCA scree chart showing explained variance per component.

    Plots per-component explained variance ratio as bars, with an
    optional cumulative-variance overlay line and a horizontal
    reference rule at a target cumulative-variance threshold.

    Parameters
    ----------
    model_or_source : PCA estimator or ModelSource
        A fitted ``sklearn.decomposition.PCA`` instance, an explicit
        ``ferrum.ModelSource`` wrapping one, or an unfitted PCA
        estimator (fit is run on ``X``).
    X : array-like, optional
        Feature matrix. Required when ``model_or_source`` is a raw
        (unfitted) estimator; ignored when it is already a
        ``ModelSource`` with data bound.
    n_components : int or None, default None
        Number of components to display. When ``None``, all available
        components from the fitted PCA are shown.
    cumulative_line : bool, default True
        When ``True``, overlays a cumulative explained-variance line.
    threshold : float or None, default 0.95
        When a float is given, draws a horizontal reference rule at
        that cumulative-variance level (e.g. ``0.95`` marks where 95%
        of variance is explained). Pass ``None`` to omit the rule.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        PCA scree bar chart with optional cumulative line and threshold
        rule.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.decomposition import PCA
    >>> fm.pca_scree_chart(PCA(n_components=10).fit(X_train), threshold=0.90)
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
    scoring: str = "both",
    n_init: int = 10,
    random_state: int | None = None,
    theme: Any = None,
):
    """Elbow and silhouette diagnostics over a range of cluster counts.

    Fits one clusterer per value of k and renders the requested
    diagnostic panel(s): distortion (inertia) vs. k, mean silhouette
    score vs. k, or both side-by-side. Unlike the other figure
    functions, this sweeps the model class itself rather than wrapping
    a single pre-fitted ``ModelSource``.

    Parameters
    ----------
    X : array-like
        Feature matrix. All samples are used for fitting and scoring.
        Polars DataFrames, pandas DataFrames, and 2D numpy arrays are
        accepted.
    ks : iterable of int
        Values of k (number of clusters) to evaluate.
    method : {"kmeans", "hierarchical"}, default "kmeans"
        Clustering algorithm.

        - ``"kmeans"`` — ``sklearn.cluster.KMeans``. Uses the estimator's
          native ``inertia_`` attribute.
        - ``"hierarchical"`` — ``sklearn.cluster.AgglomerativeClustering``
          (Ward linkage). Inertia is computed manually as the sum of
          squared distances from each sample to its cluster centroid,
          since AgglomerativeClustering does not expose ``inertia_``.

        DBSCAN is intentionally not supported — its number of clusters
        is determined by ``eps`` / ``min_samples``, not by a swept k,
        so the chart's elbow / silhouette-vs-k axis doesn't apply.
    scoring : {"elbow", "silhouette", "both"}, default "both"
        Which diagnostic panel(s) to render.

        - ``"elbow"`` — inertia-vs-k line chart only.
        - ``"silhouette"`` — mean silhouette-vs-k line chart only.
        - ``"both"`` — side-by-side HConcatChart of the two.
    n_init : int, default 10
        Number of KMeans initializations per k; forwarded to
        ``sklearn.cluster.KMeans(n_init=...)``. Ignored for hierarchical
        clustering, which is deterministic given the linkage strategy.
    random_state : int or None, default None
        Random seed for KMeans initialisation. When ``None``, defaults
        to seed 0 for deterministic results. Ignored for hierarchical
        clustering.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Single-panel line chart when ``scoring`` is ``"elbow"`` or
        ``"silhouette"``; HConcatChart of both panels when
        ``scoring="both"``.

    Raises
    ------
    ValueError
        If ``method`` is unknown or ``scoring`` is not in the supported
        set.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.cluster_diagnostics(X_train, ks=range(2, 11))
    >>> fm.cluster_diagnostics(X_train, ks=range(2, 11),
    ...                         method="hierarchical", scoring="silhouette")
    """
    from ferrum._diagnostics.deps import require_sklearn
    require_sklearn("cluster_diagnostics")
    if method not in ("kmeans", "hierarchical"):
        raise ValueError(
            f"cluster_diagnostics(method={method!r}) — expected one of "
            "'kmeans', 'hierarchical'. DBSCAN and other density-based "
            "methods don't fit the sweep-k framework and are not supported."
        )
    if scoring not in ("elbow", "silhouette", "both"):
        raise ValueError(
            f"cluster_diagnostics(scoring={scoring!r}) — expected one of "
            "'elbow', 'silhouette', 'both'."
        )
    import polars as pl
    import numpy as np
    import ferrum
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    X_np = np.ascontiguousarray(X_np, dtype=np.float64)
    seed = 0 if random_state is None else int(random_state)
    rows = []
    for k in ks:
        if method == "kmeans":
            m = KMeans(
                n_clusters=int(k), n_init=int(n_init), random_state=seed,
            ).fit(X_np)
            inertia = float(m.inertia_)
            labels = m.labels_
        else:  # hierarchical
            m = AgglomerativeClustering(n_clusters=int(k), linkage="ward").fit(X_np)
            labels = m.labels_
            # AgglomerativeClustering doesn't expose inertia_; compute
            # manually as the sum of squared distances from each sample
            # to its cluster centroid (the same definition KMeans uses).
            inertia = 0.0
            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                centroid = X_np[mask].mean(axis=0)
                inertia += float(np.sum((X_np[mask] - centroid) ** 2))
        rows.append({
            "k": int(k),
            "inertia": inertia,
            "silhouette": float(silhouette_score(X_np, labels)),
        })
    df = pl.DataFrame(rows)
    elbow = ferrum.Chart(df).mark_line().encode(x="k", y="inertia")
    sil = ferrum.Chart(df).mark_line().encode(x="k", y="silhouette")
    if scoring == "elbow":
        chart = elbow
    elif scoring == "silhouette":
        chart = sil
    else:
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
    """Intercluster distance map: 2D embedding of cluster centers.

    Embeds cluster centers into 2D using MDS so their pairwise distances
    are approximately preserved. Each center is rendered as a
    size-encoded circle (bubble area proportional to cluster count).
    A 15% padding is added around the data range so large bubbles do
    not clip at axis edges.

    Parameters
    ----------
    model_or_source : fitted clusterer or ModelSource
        A fitted sklearn-compatible clusterer (must expose
        ``cluster_centers_`` or ``n_clusters``) or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix used to compute cluster member counts.
        Required when ``model_or_source`` is a raw estimator.
    k : int or None, default None
        Number of clusters. When ``None``, inferred from
        ``model.n_clusters`` or ``len(model.cluster_centers_)``; raises
        ``ValueError`` if neither attribute is present.
    method : {"mds", "tsne"}, default "mds"
        Dimensionality-reduction method for embedding cluster centers.
        ``"mds"`` (default) uses sklearn MDS to preserve pairwise
        distances; ``"tsne"`` uses t-SNE (perplexity clamped to
        ``min(5, k-1)`` for small cluster counts).
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        2D scatter chart of embedded cluster centers sized by cluster
        population.

    Raises
    ------
    ValueError
        If ``k`` is ``None`` and the model exposes neither
        ``n_clusters`` nor ``cluster_centers_``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.cluster import KMeans
    >>> fm.intercluster_distance_chart(KMeans(n_clusters=5).fit(X_train), X_train)
    """
    from ferrum._diagnostics.charts import (
        _intercluster_distance_chart_from_source,
    )
    source = _resolve_source(model_or_source, X, None, random_state=random_state)
    if k is None:
        if hasattr(source.model, "n_clusters"):
            k = int(source.model.n_clusters)
        elif hasattr(source.model, "cluster_centers_"):
            k = int(source.model.cluster_centers_.shape[0])
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
    """Feature-ranking chart: univariate bar or pairwise heatmap.

    Computes a ranking score for each feature (or each feature pair)
    and renders either a ranked bar chart (``rank="1d"``) or a pairwise
    correlation heatmap (``rank="2d"``). Accepts a fitted estimator,
    ``ModelSource``, or a raw DataFrame / 2D array (no model required
    for most algorithms).

    Parameters
    ----------
    data_or_source : estimator, ModelSource, DataFrame, or array-like
        Input data. When a fitted estimator or ``ModelSource`` is
        supplied, the feature matrix is taken from the bound data.
        When a DataFrame or 2D array is supplied, ``X`` is used as
        the feature matrix if provided.
    X : array-like, optional
        Feature matrix. Used when ``data_or_source`` is a raw estimator
        (not a ``ModelSource``) or when ``data_or_source`` is a raw
        DataFrame and ``X`` overrides it.
    y : array-like, optional
        Target vector. Required only for
        ``algorithm="covariance"`` which routes through
        ``ModelSource.rank1d``.
    rank : {"1d", "2d"}, default "2d"
        Ranking mode. ``"1d"`` computes a univariate score per feature
        and renders a horizontal bar chart (or vertical with
        ``orient="vertical"``). ``"2d"`` computes pairwise scores and
        renders a heatmap.
    algorithm : str or None, default None
        Ranking algorithm. When ``None``, defaults to ``"shapiro"``
        for ``rank="1d"`` and ``"pearson"`` for ``rank="2d"``.
    top_k : int or None, default None
        For ``rank="1d"``, truncate to the top-k features by score.
        Has no effect for ``rank="2d"``.
    annot : bool, default True
        For ``rank="2d"``, overlays the correlation value (2 decimal
        places) as a text label in each heatmap cell. Has no effect for
        ``rank="1d"``.
    orient : {"horizontal", "vertical"}, default "horizontal"
        Bar orientation for ``rank="1d"``; ignored for ``rank="2d"``.
    color_field : str or None, default None
        Column name to use for bar fill color in ``rank="1d"``; when
        ``None``, a single color is used.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Ranked bar chart (``rank="1d"``) or pairwise heatmap
        (``rank="2d"``).

    Raises
    ------
    ValueError
        If ``rank`` is not ``"1d"`` or ``"2d"``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.rank_chart(X_train, rank="2d")
    >>> fm.rank_chart(X_train, rank="1d", algorithm="shapiro", top_k=10)
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
    """Parallel coordinates chart for multivariate data.

    Renders one polyline per sample, with each feature mapped to a
    vertical axis. Features are optionally rescaled to a common range
    before plotting so all axes are visually comparable. Samples are
    colored by a grouping column when ``hue`` is provided.

    Parameters
    ----------
    data : polars.DataFrame, pandas.DataFrame, or array-like
        Input data. Polars and pandas DataFrames are used directly;
        2D numpy arrays are auto-named ``f0``, ``f1``, etc.
    features : list of str or None, default None
        Column names to use as parallel axes. When ``None``, all
        columns except ``hue`` are used.
    hue : str or None, default None
        Column name to color samples by (e.g. a target class or cluster
        id). Pass ``None`` for monochrome lines.
    rescale : {"minmax", "zscore"} or None, default "minmax"
        Per-feature rescaling applied before rendering so axes share a
        common visual range. ``"minmax"`` maps to ``[0, 1]``;
        ``"zscore"`` standardizes to zero mean and unit variance;
        ``None`` uses raw feature values.
    alpha : float, default 0.5
        Opacity of individual polylines; lower values reduce overplot
        in dense datasets.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Parallel coordinates chart with one polyline per sample.

    Raises
    ------
    ValueError
        If any name in ``features`` is not a column in ``data``.
    ValueError
        If ``rescale`` is not one of ``"minmax"``, ``"zscore"``, or
        ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.parallel_coordinates_chart(X_df, hue="species", rescale="minmax")
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
    """Decision-boundary heatmap for a classifier over a 2D feature slice.

    Builds a ``grid_resolution × grid_resolution`` grid over two
    selected features, holds all other features fixed at their column
    means, and colors each cell by the model's predicted class (or
    probability). Optionally overlays training-point scatter.

    Parameters
    ----------
    model : estimator or ModelSource
        Fitted sklearn-compatible classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like
        Feature matrix. Must be provided (not optional) so the grid
        bounds and column means can be computed.
    y : array-like or None, default None
        True labels. Used only for the scatter overlay when
        ``scatter=True``; not required otherwise.
    features : tuple of (int or str, int or str), default (0, 1)
        Two feature indices or column names to use for the x and y axes
        of the grid. All other features are fixed at their column means.
        Exactly 2 features are required.
    grid_resolution : int, default 200
        Number of grid points along each axis; total cells =
        ``grid_resolution²``.
    proba : bool, default False
        When ``True`` and the model exposes ``predict_proba``, the color
        channel uses ``predict_proba[:, 1]`` (continuous probability).
        When ``False``, the color channel uses ``predict`` (discrete
        class index).
    scatter : bool, default False
        When ``True``, overlays a scatter of training points colored by
        ``y`` on top of the boundary heatmap via the ``+`` compositor.
        Note: the overlay currently renders as horizontal concatenation
        per the ``ChartSpec`` one-batch contract.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Decision-boundary heatmap, optionally with training-point
        scatter overlay.

    Raises
    ------
    ValueError
        If ``features`` does not contain exactly 2 elements.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.svm import SVC
    >>> fm.decision_boundary_chart(SVC().fit(X_train, y_train), X_train, y_train, features=(0, 1))
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

