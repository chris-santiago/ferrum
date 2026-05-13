"""10e model-selection visualizers — LearningCurve, ValidationCurve,
CVScores, AlphaSelection.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ..charts import (
    _alpha_selection_chart_from_source,
    _cv_scores_chart_from_source,
    _learning_curve_chart_from_source,
    _validation_curve_chart_from_source,
)
from .base import FerrumVisualizer


class LearningCurveVisualizer(FerrumVisualizer):
    """Visualize training and cross-validation scores as training size grows.

    Wraps ``learning_curve_chart`` in the sklearn visualizer protocol.
    Call ``fit(X, y)`` to run cross-validation across the requested
    training-size grid; call ``show()`` to retrieve the rendered
    ``Chart``.  The headline metric ``final_test_score`` (mean test score
    at the largest training size) is recorded in ``_metrics`` and shown
    in ``repr``.

    Parameters
    ----------
    model : Any
        Unfitted or fitted sklearn-compatible estimator.
    cv : int, default 5
        Number of cross-validation folds passed to
        ``sklearn.model_selection.learning_curve``.
    scoring : str or callable, optional
        Sklearn scoring string or callable.  Defaults to the estimator's
        own ``score`` method when ``None``.
    train_sizes : array-like, optional
        Relative or absolute training-set sizes to evaluate.  Passed
        directly to ``learning_curve``; defaults to ``np.linspace(0.1,
        1.0, 5)`` when ``None``.
    ci_style : {"band", "bar"}, default "band"
        How to render the cross-validation confidence interval.
        ``"band"`` draws a shaded ribbon; ``"bar"`` draws error bars.
    random_state : int, optional
        Seed forwarded to ``ModelSource`` and from there to
        ``learning_curve`` for reproducible CV splits (ChaCha8 RNG).
    theme : Theme, optional
        Per-chart theme override.  Falls back to the global default
        when ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> viz = fm.LearningCurveVisualizer(
    ...     LogisticRegression(), cv=5, scoring="accuracy"
    ... ).fit(X_train, y_train)
    >>> chart = viz.show()
    """

    def __init__(
        self,
        model: Any,
        *,
        cv: int = 5,
        scoring: Any = None,
        train_sizes: Any = None,
        ci_style: str = "band",
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.cv = cv
        self.scoring = scoring
        self.train_sizes = train_sizes
        self.ci_style = ci_style

    def _materialize(self) -> None:
        df = self._source.learning_curve(
            cv=self.cv,
            scoring=self.scoring,
            train_sizes=self.train_sizes,
        )
        # mean_score is already aggregated per (train_size, split) in
        # the source schema, so collapsing per-fold rows is explicit
        # deduplication — not a group-then-aggregate. Use
        # df.unique(...maintain_order=True) for that intent (same pattern
        # as ValidationCurveVisualizer / AlphaSelectionVisualizer).
        test_rows = (
            df.filter(pl.col("split") == "test")
            .unique(subset=["train_size"], keep="first", maintain_order=True)
            .sort("train_size")
        )
        if test_rows.height:
            self._metrics["final_test_score"] = float(test_rows["mean_score"][-1])
        else:
            self._metrics["final_test_score"] = float("nan")

    def _build_chart(self) -> Any:
        return _learning_curve_chart_from_source(
            self._source,
            cv=self.cv,
            scoring=self.scoring,
            train_sizes=self.train_sizes,
            ci_style=self.ci_style,
            theme=self.theme,
        )


class ValidationCurveVisualizer(FerrumVisualizer):
    """Visualize train/test scores as a single hyperparameter is swept.

    Wraps ``validation_curve_chart`` in the sklearn visualizer protocol.
    For each value in ``values``, ``sklearn.model_selection.validation_curve``
    runs ``cv`` folds and records the mean score.  The value that
    maximises the mean test score is stored as ``best_param`` in
    ``_metrics``; the corresponding mean score is stored as
    ``best_test_score``.

    Parameters
    ----------
    model : Any
        Unfitted or fitted sklearn-compatible estimator.
    param : str
        Name of the hyperparameter to sweep, e.g. ``"C"`` for
        ``LogisticRegression`` or ``"max_depth"`` for tree estimators.
        Must be a valid ``__init__`` parameter of ``model``.
    values : array-like
        Grid of candidate values for ``param``.  Passed directly to
        ``sklearn.model_selection.validation_curve`` as ``param_range``.
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str or callable, optional
        Sklearn scoring string or callable.  Defaults to the estimator's
        own ``score`` method when ``None``.
    log_scale : bool or "auto", default "auto"
        Whether to render the x-axis (param values) on a log scale.
        ``"auto"`` enables log scale when the ratio between the largest
        and smallest value exceeds 100.
    ci_style : {"band", "bar"}, default "band"
        How to render the cross-validation confidence interval.
        ``"band"`` draws a shaded ribbon; ``"bar"`` draws error bars.
    random_state : int, optional
        Seed forwarded to ``ModelSource`` for reproducible CV splits
        (ChaCha8 RNG).
    theme : Theme, optional
        Per-chart theme override.  Falls back to the global default
        when ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.svm import SVC
    >>> import numpy as np
    >>> viz = fm.ValidationCurveVisualizer(
    ...     SVC(), param="C", values=np.logspace(-3, 3, 7), cv=5
    ... ).fit(X_train, y_train)
    >>> chart = viz.show()
    """

    def __init__(
        self,
        model: Any,
        param: str,
        values: Any,
        *,
        cv: int = 5,
        scoring: Any = None,
        log_scale: Any = "auto",
        ci_style: str = "band",
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.param = param
        self.values = values
        self.cv = cv
        self.scoring = scoring
        self.log_scale = log_scale
        self.ci_style = ci_style

    def _materialize(self) -> None:
        df = self._source.validation_curve(
            self.param,
            self.values,
            cv=self.cv,
            scoring=self.scoring,
        )
        # The source emits one (split, param_value) row per combination;
        # dedupe explicitly on param_value to guard against schema changes.
        test_rows = (
            df.filter(pl.col("split") == "test")
            .unique(subset=["param_value"], keep="first", maintain_order=True)
            .sort("param_value")
        )
        if test_rows.height:
            idx = test_rows["mean_score"].arg_max()
            self._metrics["best_param"] = float(test_rows["param_value"][idx])
            self._metrics["best_test_score"] = float(test_rows["mean_score"][idx])
        else:
            self._metrics["best_param"] = float("nan")
            self._metrics["best_test_score"] = float("nan")

    def _build_chart(self) -> Any:
        return _validation_curve_chart_from_source(
            self._source,
            self.param,
            self.values,
            cv=self.cv,
            scoring=self.scoring,
            log_scale=self.log_scale,
            ci_style=self.ci_style,
            theme=self.theme,
        )


class CVScoresVisualizer(FerrumVisualizer):
    """Visualize the distribution of per-fold cross-validation scores.

    Wraps ``cv_scores_chart`` in the sklearn visualizer protocol.  After
    ``fit(X, y)``, ``_metrics`` contains ``test_mean`` (mean test-fold
    score across all folds) and ``test_std`` (standard deviation of
    test-fold scores).

    Parameters
    ----------
    model : Any
        Unfitted or fitted sklearn-compatible estimator.
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str or callable, optional
        Sklearn scoring string or callable.  Defaults to the estimator's
        own ``score`` method when ``None``.
    kind : {"box", "bar", "strip"}, default "box"
        Chart geometry used to display the fold-score distributions.
    split : {"both", "test", "train"}, default "both"
        Which CV splits to include in the chart.  ``"both"`` shows train
        and test folds side by side; ``"test"`` or ``"train"`` shows
        only one.
    random_state : int, optional
        Seed forwarded to ``ModelSource`` for reproducible CV splits
        (ChaCha8 RNG).
    theme : Theme, optional
        Per-chart theme override.  Falls back to the global default
        when ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> viz = fm.CVScoresVisualizer(
    ...     RandomForestClassifier(n_estimators=100), cv=10, kind="box"
    ... ).fit(X_train, y_train)
    >>> chart = viz.show()
    """

    def __init__(
        self,
        model: Any,
        *,
        cv: int = 5,
        scoring: Any = None,
        kind: str = "box",
        split: str = "both",
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.cv = cv
        self.scoring = scoring
        self.kind = kind
        self.split = split

    def _materialize(self) -> None:
        df = self._source.cv_scores(cv=self.cv, scoring=self.scoring)
        test = df.filter(pl.col("split") == "test")["score"]
        if test.len():
            self._metrics["test_mean"] = float(test.mean())
            self._metrics["test_std"] = float(test.std(ddof=0))
        else:
            self._metrics["test_mean"] = float("nan")
            self._metrics["test_std"] = float("nan")

    def _build_chart(self) -> Any:
        return _cv_scores_chart_from_source(
            self._source,
            cv=self.cv,
            scoring=self.scoring,
            kind=self.kind,
            split=self.split,
            theme=self.theme,
        )


class AlphaSelectionVisualizer(FerrumVisualizer):
    """Visualize cross-validated scores over a regularization-strength grid.

    Wraps ``alpha_selection_chart`` in the sklearn visualizer protocol.
    Designed for regularized regressors such as ``Ridge``, ``Lasso``, and
    ``ElasticNet`` — estimators that expose an ``alpha`` hyperparameter
    controlling L1/L2 penalty strength.  For each value in ``alphas``,
    cross-validation produces a mean score; the alpha that maximises the
    mean test score is stored as ``best_alpha`` in ``_metrics``, and the
    corresponding score as ``best_score``.

    Parameters
    ----------
    model : Any
        Unfitted regularized regressor with an ``alpha`` hyperparameter
        (e.g. ``sklearn.linear_model.Ridge``, ``Lasso``, ``ElasticNet``).
        Passing an estimator without an ``alpha`` parameter will raise at
        ``fit`` time.
    alphas : array-like
        Grid of regularization-strength values to evaluate.  Passed
        directly to the underlying ``alpha_selection`` compute; log-spaced
        grids (e.g. ``np.logspace(-4, 4, 50)``) are typical.
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str or callable, optional
        Sklearn scoring string or callable.  Defaults to the estimator's
        own ``score`` method when ``None``.
    log_scale : bool, default True
        Whether to render the x-axis (alpha values) on a log scale.
        Defaults to ``True`` because alpha grids are almost always
        log-spaced.
    highlight_best : bool, default True
        When ``True``, draws a vertical rule at the alpha value that
        maximises the mean test score.
    random_state : int, optional
        Seed forwarded to ``ModelSource`` for reproducible CV splits
        (ChaCha8 RNG).
    theme : Theme, optional
        Per-chart theme override.  Falls back to the global default
        when ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> viz = fm.AlphaSelectionVisualizer(
    ...     Ridge(), alphas=np.logspace(-4, 4, 40), cv=5
    ... ).fit(X_train, y_train)
    >>> chart = viz.show()
    """

    def __init__(
        self,
        model: Any,
        alphas: Any,
        *,
        cv: int = 5,
        scoring: Any = None,
        log_scale: bool = True,
        highlight_best: bool = True,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.alphas = alphas
        self.cv = cv
        self.scoring = scoring
        self.log_scale = log_scale
        self.highlight_best = highlight_best

    def _materialize(self) -> None:
        df = self._source.alpha_selection(
            self.alphas,
            cv=self.cv,
            scoring=self.scoring,
        )
        # The source emits one row per alpha; dedupe explicitly to guard
        # against schema changes that might add a (split, alpha) breakdown.
        agg = df.unique(subset=["alpha"], keep="first", maintain_order=True).sort("alpha")
        if agg.height:
            idx = agg["mean_score"].arg_max()
            self._metrics["best_alpha"] = float(agg["alpha"][idx])
            self._metrics["best_score"] = float(agg["mean_score"][idx])
        else:
            self._metrics["best_alpha"] = float("nan")
            self._metrics["best_score"] = float("nan")

    def _build_chart(self) -> Any:
        return _alpha_selection_chart_from_source(
            self._source,
            self.alphas,
            cv=self.cv,
            scoring=self.scoring,
            log_scale=self.log_scale,
            highlight_best=self.highlight_best,
            theme=self.theme,
        )
