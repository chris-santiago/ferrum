"""Model-selection domain figure functions and their private builders.

Public API
----------
learning_curve_chart, validation_curve_chart, cv_scores_chart,
alpha_selection_chart.

Each public function wraps ``_resolve_source`` (shared across all figure
functions) and dispatches to a co-located ``_*_from_source`` builder that
produces a fully-formed ``Chart``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from ferrum import Chart

from ferrum.encoding import X, Y
from ferrum._overrides import _apply_overrides
from ferrum.plots._helpers import _dedupe_aggregated, _finalize_chart
from ferrum.plots._helpers import _resolve_source, _require


# ---------------------------------------------------------------------------
# Builders (private)
# ---------------------------------------------------------------------------


def _learning_curve_chart_from_source(
    source: Any,
    *,
    cv: int = 5,
    scoring: Any = None,
    train_sizes: Any = None,
    ci_style: str = "band",
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Learning-curve chart: dedupe per (train_size, split), then ribbon+line.

    Schwabish SB3 (2026-05-11): replaces the categorical legend with
    endpoint-anchored direct labels (``train`` / ``test``), suppresses
    the redundant color legend via ``Color(legend=None)``, and adds a
    ``Learning curve`` active title.
    """
    import ferrum
    from ferrum._direct_label import _direct_label_endpoint

    df = source.learning_curve(cv=cv, scoring=scoring, train_sizes=train_sizes)
    df = _dedupe_aggregated(df, "train_size", "split")
    df = df.with_columns(
        pl.col("split").replace({"train": "Training Score", "test": "Cross-Validation Score"})
    )
    chart = (
        ferrum.Chart(df)
        .encode(color=ferrum.Color("split", legend=None))
        .mark_learning_curve(ci_style=ci_style)
    )
    chart = chart.encode(
        x=X("train_size", title="Training instances"),
        y=Y("mean_score", title="Score"),
    )
    chart = chart.properties(
        title=ferrum.Title("Learning Curve", subtitle=subtitle),
    )
    chart = _direct_label_endpoint(
        chart,
        label_field="split",
        x_col="train_size",
        y_col="mean_score",
    )
    from ferrum._layer import _Layer

    chart = chart.layer(
        _Layer(
            mark="point",
            encoding={"x": "train_size", "y": "mean_score", "color": "split"},
            mark_kwargs={"size": 40, "filled": True},
            name="point",
        )
    )
    return _finalize_chart(chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme)


def _validation_curve_chart_from_source(
    source: Any,
    param: str,
    values: Any,
    *,
    cv: int = 5,
    scoring: Any = None,
    log_scale: Any = "auto",
    ci_style: str = "band",
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Validation-curve chart: dedupe per (param_value, split), then ribbon+line.

    Schwabish SB3 (2026-05-11): direct labels + legend suppression +
    parameterized active title (``"Validation curve — <param>"``).
    """
    import ferrum
    from ferrum._direct_label import _direct_label_endpoint

    df = source.validation_curve(param, values, cv=cv, scoring=scoring)
    df = _dedupe_aggregated(df, "param_value", "split")
    df = df.with_columns(
        pl.col("split").replace({"train": "Training Score", "test": "Cross-Validation Score"})
    )
    vals = [float(v) for v in values]
    if log_scale == "auto":
        non_zero = [v for v in vals if v > 0]
        if len(non_zero) >= 2 and max(non_zero) / min(non_zero) > 100:
            is_log = True
        else:
            is_log = False
    else:
        is_log = bool(log_scale)
    chart = (
        ferrum.Chart(df)
        .encode(color=ferrum.Color("split", legend=None))
        .mark_validation_curve(
            log_scale=is_log,
            ci_style=ci_style,
            param_label=param,
        )
    )
    chart = chart.encode(
        y=Y("mean_score", title="Score"),
    )
    chart = chart.properties(
        title=ferrum.Title(f"Validation Curve — {param}", subtitle=subtitle),
    )
    chart = _direct_label_endpoint(
        chart,
        label_field="split",
        x_col="param_value",
        y_col="mean_score",
    )
    from ferrum._layer import _Layer

    chart = chart.layer(
        _Layer(
            mark="point",
            encoding={"x": "param_value", "y": "mean_score", "color": "split"},
            mark_kwargs={"size": 40, "filled": True},
            name="point",
        )
    )
    return _finalize_chart(chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme)


def _cv_scores_chart_from_source(
    source: Any,
    *,
    cv: int = 5,
    scoring: Any = None,
    kind: str = "box",
    split: str = "both",
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Per-fold CV-score chart.

    ``kind="bar"`` shows one bar per fold with a dashed mean-score reference
    line per split; ``"box"``/``"strip"`` leave raw per-fold rows for the
    mark layer.
    """
    import ferrum

    df = source.cv_scores(cv=cv, scoring=scoring)
    if split != "both":
        df = df.filter(pl.col("split") == split)
    if kind == "bar":
        # Broadcast per-split mean so the rule layer draws one line per split.
        df = df.with_columns(pl.col("score").mean().over("split").alias("_mean_score"))
    chart = ferrum.Chart(df).mark_cv_scores(kind=kind, split=split)
    chart = chart.encode(y=Y("score", title="Score"))
    chart = chart.properties(title=ferrum.Title("Cross-Validation Scores"))
    return _finalize_chart(chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme)


def _alpha_selection_chart_from_source(
    source: Any,
    alphas: Any,
    *,
    cv: int = 5,
    scoring: Any = None,
    log_scale: bool = True,
    highlight_best: bool = True,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Alpha-selection chart: dedupe per alpha (one row per alpha holds the
    aggregated mean_score). The Chart method injects ``_best_alpha`` when
    ``highlight_best=True``.
    """
    import ferrum

    df = source.alpha_selection(alphas, cv=cv, scoring=scoring)
    df = _dedupe_aggregated(df, "alpha")
    chart = ferrum.Chart(df).mark_alpha_selection(
        log_scale=log_scale,
        highlight_best=highlight_best,
    )
    chart = chart.encode(y=Y("mean_score", title="Score"))
    chart = chart.properties(title=ferrum.Title("Alpha Selection"))
    from ferrum._layer import _Layer

    chart = chart.layer(
        _Layer(
            mark="point",
            encoding={"x": "alpha", "y": "mean_score"},
            mark_kwargs={"size": 40, "filled": True},
            name="point",
        )
    )
    # Schwabish C11: annotate the best-alpha point with a text label.
    # The desugar already draws a vertical rule via the _best_alpha sentinel
    # (injected by the data_transform at render time); here we inject
    # companion sentinel columns for a text layer that reads the same
    # _best_alpha x position.
    if highlight_best and "mean_score" in df.columns:
        best_idx = int(df["mean_score"].arg_max())
        best_alpha = float(df["alpha"][best_idx])
        best_score = float(df["mean_score"][best_idx])
        label_text = f"α = {best_alpha:.3f}"
        n = df.height
        df = df.with_columns(
            pl.Series("_best_alpha_y", [best_score] + [None] * (n - 1), dtype=pl.Float64),
            pl.Series("_best_alpha_text", [label_text] + [None] * (n - 1), dtype=pl.Utf8),
        )
        # Rebuild the chart with the augmented DataFrame so that the
        # text layer shares the same data identity.
        chart = ferrum.Chart(df).mark_alpha_selection(
            log_scale=log_scale,
            highlight_best=highlight_best,
        )
        chart = chart.encode(y=Y("mean_score", title="Score"))
        chart = chart.properties(title=ferrum.Title("Alpha Selection"))
        chart = chart.layer(
            _Layer(
                mark="point",
                encoding={"x": "alpha", "y": "mean_score"},
                mark_kwargs={"size": 40, "filled": True},
                name="point",
            ),
            _Layer(
                mark="text",
                encoding={"x": "_best_alpha", "y": "_best_alpha_y", "text": "_best_alpha_text"},
                mark_kwargs={"dx": 8, "dy": -8, "align": "left"},
                name="best_alpha_text",
            ),
        )
    return _finalize_chart(chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme)


# ---------------------------------------------------------------------------
# Public figure functions
# ---------------------------------------------------------------------------


def learning_curve_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    cv: int = 5,
    scoring: Any = None,
    train_sizes: Any = None,
    ci_style: str = "band",
    subtitle: str | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
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
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
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
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _learning_curve_chart_from_source(
        source,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        ci_style=ci_style,
        subtitle=subtitle,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
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
    subtitle: str | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Plot score vs. a single hyperparameter value.

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
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
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
    _require(
        "validation_curve_chart",
        "values",
        values,
        hint="pass an explicit list of values to sweep for the given param",
    )
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _validation_curve_chart_from_source(
        source,
        param,
        values,
        cv=cv,
        scoring=scoring,
        log_scale=log_scale,
        ci_style=ci_style,
        subtitle=subtitle,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
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
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
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
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _cv_scores_chart_from_source(
        source,
        cv=cv,
        scoring=scoring,
        kind=kind,
        split=split,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
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
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
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
    _require(
        "alpha_selection_chart",
        "alphas",
        alphas,
        hint="pass an explicit list of regularization-strength values to sweep",
    )
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _alpha_selection_chart_from_source(
        source,
        alphas,
        cv=cv,
        scoring=scoring,
        log_scale=log_scale,
        highlight_best=highlight_best,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )
