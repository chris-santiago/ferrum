"""10e model-selection visualizers — LearningCurve, ValidationCurve,
CVScores, AlphaSelection.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..charts import (
    _alpha_selection_chart_from_source,
    _cv_scores_chart_from_source,
    _learning_curve_chart_from_source,
    _validation_curve_chart_from_source,
)
from .base import FerrumVisualizer


class LearningCurveVisualizer(FerrumVisualizer):
    """Sklearn-protocol visualizer for ``learning_curve_chart``.

    Materializes per-(train_size, split) mean scores and records the
    final-training-size test mean as ``final_test_score`` for the repr.
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
            cv=self.cv, scoring=self.scoring, train_sizes=self.train_sizes,
        )
        test_rows = (
            df.filter(pl.col("split") == "test")
            .group_by("train_size")
            .agg(pl.col("mean_score").first())
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
    """Sklearn-protocol visualizer for ``validation_curve_chart``.

    Records the test-score-maximizing ``param`` value as ``best_param``
    and the corresponding mean as ``best_test_score``.
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
            self.param, self.values, cv=self.cv, scoring=self.scoring,
        )
        test_rows = (
            df.filter(pl.col("split") == "test")
            .group_by("param_value")
            .agg(pl.col("mean_score").first())
            .sort("param_value")
        )
        if test_rows.height:
            scores = test_rows["mean_score"].to_numpy()
            idx = int(np.argmax(scores))
            self._metrics["best_param"] = float(test_rows["param_value"][idx])
            self._metrics["best_test_score"] = float(scores[idx])
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
    """Sklearn-protocol visualizer for ``cv_scores_chart``.

    Records the test-fold mean and std as ``test_mean`` / ``test_std``.
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
        test = df.filter(pl.col("split") == "test")["score"].to_numpy()
        if test.size:
            self._metrics["test_mean"] = float(test.mean())
            self._metrics["test_std"] = float(test.std())
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
    """Sklearn-protocol visualizer for ``alpha_selection_chart``.

    Records the test-score-maximizing alpha as ``best_alpha``.
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
            self.alphas, cv=self.cv, scoring=self.scoring,
        )
        agg = (
            df.group_by("alpha")
            .agg(pl.col("mean_score").first())
            .sort("alpha")
        )
        if agg.height:
            scores = agg["mean_score"].to_numpy()
            idx = int(np.argmax(scores))
            self._metrics["best_alpha"] = float(agg["alpha"][idx])
            self._metrics["best_score"] = float(scores[idx])
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
