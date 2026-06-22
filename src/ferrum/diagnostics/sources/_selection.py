"""Phase 10e — model selection / CV curves (learning, validation, cv scores, alpha selection)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .._internal.deps import require_sklearn


def _ci_agg_rows(key_name: str, key_value, split_name: str, arr: np.ndarray) -> list[dict]:
    """Build per-fold row dicts with CI/mean/std aggregates for one (key, split, fold) group.

    Used by both ``learning_curve`` (key = train_size) and
    ``validation_curve`` (key = param_value).  Returns one dict per element
    in ``arr`` — callers extend their row accumulator with the result.
    """
    mean = float(arr.mean())
    std = float(arr.std())
    n = len(arr)
    ci = 1.96 * std / np.sqrt(n) if n > 0 else 0.0
    return [
        {
            key_name: key_value,
            "split": split_name,
            "score": float(s),
            "mean_score": mean,
            "std_score": std,
            "lower": mean - ci,
            "upper": mean + ci,
        }
        for s in arr
    ]


class ModelSelectionMixin:
    """Phase 10e — model selection / CV curves (learning, validation, cv scores, alpha selection)."""

    # --- 10e: model selection / CV curves --------------------------------

    def learning_curve(
        self,
        *,
        cv: int = 5,
        scoring: Any = None,
        train_sizes: Any = None,
    ) -> pl.DataFrame:
        """Learning curve: score per (train_size, fold, split).

        Returns long-form rows — one per (train_size, fold, split). Each
        row carries the per-fold ``score`` plus the per-(train_size, split)
        aggregates ``mean_score``, ``std_score``, ``lower``, ``upper`` (95%
        CI on the mean). Chart builders dedupe by (train_size, split) to
        render a ribbon + line; the per-fold rows enable per-fold strip
        overlays if a future caller wants them.
        """
        if self._y is None:
            raise ValueError("ModelSource.learning_curve() requires y to be provided.")
        key = self._cache_key(
            "learning_curve",
            cv=cv,
            scoring=scoring,
            train_sizes=train_sizes,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("learning_curve")
        from sklearn.model_selection import learning_curve as _lc

        sizes = train_sizes if train_sizes is not None else np.linspace(0.1, 1.0, 5)
        ts, tr_scores, te_scores = _lc(
            self._model,
            self._X,
            self._y,
            train_sizes=sizes,
            cv=cv,
            scoring=scoring,
            random_state=self._random_state if self._random_state is not None else 0,
            shuffle=True,
        )
        rows: list[dict] = []
        for i, t in enumerate(ts):
            for split_name, arr in (("train", tr_scores[i]), ("test", te_scores[i])):
                rows.extend(_ci_agg_rows("train_size", int(t), split_name, arr))
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def validation_curve(
        self,
        param: str,
        values: Any,
        *,
        cv: int = 5,
        scoring: Any = None,
    ) -> pl.DataFrame:
        """Validation curve: score per (param_value, fold, split).

        Same shape as ``learning_curve`` but parameterized by an
        estimator hyperparameter sweep. ``param`` is the kwarg name on
        the wrapped estimator (e.g. ``"alpha"`` for ``Ridge``).
        """
        if self._y is None:
            raise ValueError("ModelSource.validation_curve() requires y to be provided.")
        vals = np.asarray(list(values), dtype=np.float64)
        key = self._cache_key(
            "validation_curve",
            param=param,
            values=tuple(float(v) for v in vals),
            cv=cv,
            scoring=scoring,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("validation_curve")
        from sklearn.model_selection import validation_curve as _vc

        tr, te = _vc(
            self._model,
            self._X,
            self._y,
            param_name=param,
            param_range=vals,
            cv=cv,
            scoring=scoring,
        )
        rows: list[dict] = []
        for i, v in enumerate(vals):
            for split_name, arr in (("train", tr[i]), ("test", te[i])):
                rows.extend(_ci_agg_rows("param_value", float(v), split_name, arr))
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def cv_scores(
        self,
        *,
        cv: int = 5,
        scoring: Any = None,
    ) -> pl.DataFrame:
        """Per-fold cross-validation scores.

        Returns one row per (fold, split) — train and test scores for each
        cross-validation fold. Chart builders use this for boxplot / bar /
        strip distributions across folds.
        """
        if self._y is None:
            raise ValueError("ModelSource.cv_scores() requires y to be provided.")
        key = self._cache_key(
            "cv_scores",
            cv=cv,
            scoring=scoring,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("cv_scores")
        from sklearn.model_selection import cross_validate

        result = cross_validate(
            self._model,
            self._X,
            self._y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
        )
        rows: list[dict] = []
        for fold, s in enumerate(result["train_score"]):
            rows.append({"fold": int(fold), "split": "train", "score": float(s)})
        for fold, s in enumerate(result["test_score"]):
            rows.append({"fold": int(fold), "split": "test", "score": float(s)})
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def alpha_selection(
        self,
        alphas: Any,
        *,
        cv: int = 5,
        scoring: Any = None,
    ) -> pl.DataFrame:
        """Regularization-strength sweep for linear models.

        Returns one row per (alpha, fold) — the per-fold test score on the
        held-out split — plus per-alpha ``mean_score`` / ``std_score``
        aggregates. Chart builders dedupe by alpha to render a single
        line, and use ``argmax(mean_score)`` to mark the best alpha.
        """
        if self._y is None:
            raise ValueError("ModelSource.alpha_selection() requires y to be provided.")
        vals = np.asarray(list(alphas), dtype=np.float64)
        key = self._cache_key(
            "alpha_selection",
            alphas=tuple(float(v) for v in vals),
            cv=cv,
            scoring=scoring,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("alpha_selection")
        from sklearn.model_selection import validation_curve as _vc

        _, te = _vc(
            self._model,
            self._X,
            self._y,
            param_name="alpha",
            param_range=vals,
            cv=cv,
            scoring=scoring,
        )
        rows: list[dict] = []
        for i, a in enumerate(vals):
            mean = float(te[i].mean())
            std = float(te[i].std())
            for fold_idx, s in enumerate(te[i]):
                rows.append(
                    {
                        "alpha": float(a),
                        "fold": int(fold_idx),
                        "score": float(s),
                        "mean_score": mean,
                        "std_score": std,
                    }
                )
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df
