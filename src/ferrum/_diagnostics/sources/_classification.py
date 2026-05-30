"""Phase 10b — classification curves (ROC, PR, calibration, gain, lift, discrimination threshold, confusion matrix)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .. import _curve_frames
from ..deps import require_sklearn


class ClassificationCurvesMixin:
    """Phase 10b — classification curves (ROC, PR, calibration, gain, lift, discrimination threshold, confusion matrix)."""

    # --- 10b: classification curves --------------------------------------

    def roc_curve(
        self,
        *,
        average: str | None = None,
        drop_intermediate: bool = True,
    ) -> pl.DataFrame:
        """ROC curve(s). One row per (class, threshold). ``auc`` repeats per class.

        For binary classifiers with ``average=None`` (default), returns a
        single curve on the positive (second) class. For multiclass,
        returns one-vs-rest curves per class; pass ``average`` in
        {"micro", "macro", "weighted"} to additionally include a summary
        curve under ``class="<average>"``.
        """
        key = self._cache_key(
            "roc_curve",
            average=average,
            drop_intermediate=drop_intermediate,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("roc_curve")

        if self._y is None:
            raise ValueError("ModelSource.roc_curve() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        y_true = np.asarray(self._y)
        labels = [c[len("proba_") :] for c in proba_cols]
        class_values = [_coerce_class_label(c, y_true.dtype) for c in labels]
        score_matrix = proba_df[proba_cols].to_numpy()

        df = _curve_frames.roc_frame(
            y_true,
            score_matrix,
            class_values,
            labels,
            average=average,
            drop_intermediate=drop_intermediate,
        )
        self._cache[key] = df
        return df

    def pr_curve(self, *, average: str | None = None) -> pl.DataFrame:
        """Precision-recall curve(s). One row per (class, threshold).

        For binary classifiers, returns a single curve on the positive
        (second) class — ``average`` is accepted for API symmetry with
        the multiclass path but has no effect because binary classifiers
        have only one curve to draw. For multiclass:

        - ``average=None`` (default) — returns one-vs-rest curves per
          class.
        - ``average in {"micro", "macro", "weighted"}`` — returns a
          single summary curve with ``class="<average>"`` and no
          per-class rows. Macro / weighted variants interpolate per-
          class precision over a shared recall grid (100 points); micro
          ravels the binarized labels into one curve. ``threshold`` is
          NaN on every row of macro / weighted summaries (recall-grid
          interpolation discards thresholds) and follows sklearn's
          padding convention for micro.

        ``threshold`` is NaN at the final (recall=0) point of every
        per-class curve per sklearn's convention.
        """
        if average is not None and average not in ("micro", "macro", "weighted"):
            raise ValueError(
                f"pr_curve(average={average!r}) — expected one of "
                "'micro', 'macro', 'weighted', or None."
            )
        key = self._cache_key("pr_curve", average=average)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("pr_curve")

        if self._y is None:
            raise ValueError("ModelSource.pr_curve() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        y_true = np.asarray(self._y)
        labels = [c[len("proba_") :] for c in proba_cols]
        class_values = [_coerce_class_label(c, y_true.dtype) for c in labels]
        score_matrix = proba_df[proba_cols].to_numpy()

        df = _curve_frames.pr_frame(
            y_true,
            score_matrix,
            class_values,
            labels,
            average=average,
        )
        self._cache[key] = df
        return df

    def calibration_curve(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> pl.DataFrame:
        """Calibration (reliability) curve for binary classifiers.

        Returns one row per non-empty bin with ``mean_predicted``,
        ``fraction_positive``, and ``count``. Delegates to the
        ``calibration_kernel`` Rust kernel.
        """
        key = self._cache_key(
            "calibration_curve",
            n_bins=n_bins,
            strategy=strategy,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("calibration_curve")

        if self._y is None:
            raise ValueError("ModelSource.calibration_curve() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if len(proba_cols) != 2:
            raise ValueError(
                f"calibration_curve() is binary-classifier only; got {len(proba_cols)} classes."
            )
        y_true = np.asarray(self._y, dtype=np.float64)
        y_score = proba_df[proba_cols[1]].to_numpy().astype(np.float64)

        df = _curve_frames.calibration_frame(y_true, y_score, n_bins, strategy)
        self._cache[key] = df
        return df

    def cumulative_gain(self) -> pl.DataFrame:
        """Cumulative-gain curve per class. Appends a 2-row ``class='baseline'``
        diagonal for plotting reference.
        """
        key = self._cache_key("cumulative_gain")
        if key in self._cache:
            return self._cache[key]

        if self._y is None:
            raise ValueError("ModelSource.cumulative_gain() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        y_true = np.asarray(self._y)
        classes = [c[len("proba_") :] for c in proba_cols]
        n = len(y_true)

        frames: list[pl.DataFrame] = []
        for i, cls in enumerate(classes):
            y_bin = (y_true == _coerce_class_label(cls, y_true.dtype)).astype(int)
            order = np.argsort(-np.asarray(proba_df[proba_cols[i]]))
            cum_pos = np.cumsum(y_bin[order])
            total_pos = max(int(cum_pos[-1]), 1) if n else 1
            pct_pop = np.arange(1, n + 1) / max(n, 1)
            gain = cum_pos / total_pos
            xs = np.concatenate([[0.0], pct_pop])
            ys = np.concatenate([[0.0], gain])
            frames.append(
                pl.DataFrame(
                    {
                        "percent_population": xs,
                        "gain": ys,
                        "class": [str(cls)] * len(xs),
                    }
                )
            )

        frames.append(
            pl.DataFrame(
                {
                    "percent_population": [0.0, 1.0],
                    "gain": [0.0, 1.0],
                    "class": ["baseline", "baseline"],
                }
            )
        )
        df = pl.concat(frames, how="vertical")
        self._cache[key] = df
        return df

    def lift_curve(self) -> pl.DataFrame:
        """Lift curve per class. Appends a 2-row ``class='baseline'`` line at
        lift=1.0.
        """
        key = self._cache_key("lift_curve")
        if key in self._cache:
            return self._cache[key]

        if self._y is None:
            raise ValueError("ModelSource.lift_curve() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        y_true = np.asarray(self._y)
        classes = [c[len("proba_") :] for c in proba_cols]
        n = len(y_true)

        frames: list[pl.DataFrame] = []
        for i, cls in enumerate(classes):
            y_bin = (y_true == _coerce_class_label(cls, y_true.dtype)).astype(int)
            base_rate = float(y_bin.mean()) if n else 0.0
            if base_rate == 0.0:
                continue
            order = np.argsort(-np.asarray(proba_df[proba_cols[i]]))
            cum_pos = np.cumsum(y_bin[order])
            denom = np.arange(1, n + 1)
            cum_rate = cum_pos / denom
            lift = cum_rate / base_rate
            pct_pop = denom / n
            frames.append(
                pl.DataFrame(
                    {
                        "percent_population": pct_pop,
                        "lift": lift,
                        "class": [str(cls)] * n,
                    }
                )
            )

        frames.append(
            pl.DataFrame(
                {
                    "percent_population": [0.0, 1.0],
                    "lift": [1.0, 1.0],
                    "class": ["baseline", "baseline"],
                }
            )
        )
        df = pl.concat(frames, how="vertical")
        self._cache[key] = df
        return df

    def discrimination_threshold(
        self,
        *,
        n_thresholds: int = 50,
        cv: Any = None,
    ) -> pl.DataFrame:
        """Discrimination threshold sweep — binary classifiers only.

        Sweeps ``n_thresholds`` evenly-spaced thresholds in [0, 1] and
        reports precision, recall, F1, and queue_rate at each. ``queue_rate``
        is the hand-computed fraction ``(y_score >= t).mean()``.

        When ``cv`` is an int, runs the same sweep on each fold's held-out
        scores from a freshly-cloned + re-fit estimator and averages
        per-threshold metrics across folds. Pass a splitter object with a
        ``.split()`` method to override.
        """
        key = self._cache_key(
            "discrimination_threshold",
            n_thresholds=n_thresholds,
            cv=cv,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("discrimination_threshold")

        if self._y is None:
            raise ValueError("ModelSource.discrimination_threshold() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if len(proba_cols) != 2:
            raise ValueError(
                "discrimination_threshold() is binary-classifier only; "
                f"got {len(proba_cols)} classes."
            )
        y_true = np.asarray(self._y)
        positive_class = _coerce_class_label(
            proba_cols[1][len("proba_") :],
            y_true.dtype,
        )
        thresholds = np.linspace(0.0, 1.0, n_thresholds)

        if cv is None:
            y_score = np.asarray(proba_df[proba_cols[1]])
            y_true_bin = (y_true == positive_class).astype(int)
            df = _curve_frames.threshold_sweep_frame(y_true_bin, y_score, thresholds)
        else:
            df = self._discrimination_threshold_cv(
                cv,
                y_true,
                thresholds,
                positive_class,
            )

        self._cache[key] = df
        return df

    def _discrimination_threshold_cv(
        self,
        cv: Any,
        y_true: np.ndarray,
        thresholds: np.ndarray,
        positive_class: object,
    ) -> pl.DataFrame:
        """Cross-validated threshold sweep — average per-fold metrics."""
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        # numpy required: CV split uses integer-array row indexing (X_np[tr], X_np[te])
        # which polars does not support.
        X_np = self._X.to_numpy()
        splitter = (
            cv
            if hasattr(cv, "split")
            else KFold(
                n_splits=int(cv),
                shuffle=True,
                random_state=self._random_state or 0,
            )
        )
        fold_dfs: list[pl.DataFrame] = []
        for tr, te in splitter.split(X_np):
            m = clone(self._model).fit(X_np[tr], y_true[tr])
            s = _score_fold(m, X_np[te])
            y_te_bin = (y_true[te] == positive_class).astype(int)
            fold_dfs.append(_curve_frames.threshold_sweep_frame(y_te_bin, s, thresholds))
        return (
            pl.concat(fold_dfs, how="vertical")
            .group_by("threshold")
            .agg(
                [
                    pl.col("precision").mean(),
                    pl.col("recall").mean(),
                    pl.col("f1").mean(),
                    pl.col("queue_rate").mean(),
                ]
            )
            .sort("threshold")
        )

    def confusion_matrix(self, *, normalize: str | None = None) -> pl.DataFrame:
        """Confusion matrix in long form: one row per (actual, predicted) cell.

        ``normalize``: ``None`` for raw counts, ``"true"``/``"pred"``/``"all"``
        for sklearn-style normalization. ``value`` is the (possibly
        normalized) count; ``value_fmt`` is a stringified label suitable for
        ``mark_text`` overlay (integer counts when unnormalized, two-decimal
        fractions when normalized).
        """
        key = self._cache_key("confusion_matrix", normalize=normalize)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("confusion_matrix")

        if self._y is None:
            raise ValueError("ModelSource.confusion_matrix() requires y to be provided.")
        y_true = np.asarray(self._y)
        y_pred = np.asarray(self._model.predict(self._X))

        if self._class_names is not None:
            labels: list = list(self._class_names)
        elif hasattr(self._model, "classes_"):
            labels = list(self._model.classes_)
        else:
            labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

        df = _curve_frames.confusion_frame(y_true, y_pred, labels, normalize=normalize)
        self._cache[key] = df
        return df


# ---------------------------------------------------------------------------
# Model-path helpers (sklearn-bound: label coercion + per-fold scoring)
# ---------------------------------------------------------------------------


def _coerce_class_label(label_str: str, target_dtype) -> object:
    """Coerce a stringified class label back to y's dtype for equality
    comparison. Falls back to the original string if conversion fails.
    """
    if np.issubdtype(target_dtype, np.integer):
        try:
            return int(label_str)
        except ValueError:
            return label_str
    if np.issubdtype(target_dtype, np.floating):
        try:
            return float(label_str)
        except ValueError:
            return label_str
    return label_str


def _score_fold(model, X_te: np.ndarray) -> np.ndarray:
    """Return positive-class scores for one CV fold's held-out X."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X_te), dtype=np.float64)[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X_te), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-raw))
    raise AttributeError(
        "discrimination_threshold(cv=...) requires the wrapped "
        "model to implement 'predict_proba' or 'decision_function'."
    )
