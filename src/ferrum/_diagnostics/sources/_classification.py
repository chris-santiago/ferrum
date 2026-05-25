"""Phase 10b — classification curves (ROC, PR, calibration, gain, lift, discrimination threshold, confusion matrix)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

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
        from sklearn.metrics import roc_curve as _sk_roc_curve, roc_auc_score

        if self._y is None:
            raise ValueError("ModelSource.roc_curve() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        y_true = np.asarray(self._y)
        classes = [c[len("proba_") :] for c in proba_cols]
        n_classes = len(classes)

        rows: list[dict] = []
        # Binary: a single curve is the only meaningful output. ``average`` is
        # accepted for API symmetry with the multiclass path but treated as a
        # no-op (there is only one class to average over).
        if n_classes == 2:
            y_score = proba_df[proba_cols[1]]
            fpr, tpr, thr = _sk_roc_curve(
                y_true,
                y_score,
                drop_intermediate=drop_intermediate,
            )
            try:
                auc = float(roc_auc_score(y_true, y_score))
            except ValueError:
                auc = float("nan")
            for f, t, h in zip(fpr, tpr, thr):
                rows.append(
                    {
                        "fpr": float(f),
                        "tpr": float(t),
                        "threshold": float(h),
                        "class": classes[1],
                        "auc": auc,
                    }
                )
        else:
            for i, cls in enumerate(classes):
                y_bin = (y_true == _coerce_class_label(cls, y_true.dtype)).astype(int)
                y_score = proba_df[proba_cols[i]]
                fpr, tpr, thr = _sk_roc_curve(
                    y_bin,
                    y_score,
                    drop_intermediate=drop_intermediate,
                )
                try:
                    auc = float(roc_auc_score(y_bin, y_score))
                except ValueError:
                    auc = float("nan")
                for f, t, h in zip(fpr, tpr, thr):
                    rows.append(
                        {
                            "fpr": float(f),
                            "tpr": float(t),
                            "threshold": float(h),
                            "class": str(cls),
                            "auc": auc,
                        }
                    )

            if average in ("micro", "macro", "weighted"):
                rows.extend(
                    _compute_avg_roc(
                        y_true,
                        proba_df[proba_cols].to_numpy(),
                        classes,
                        average,
                        drop_intermediate,
                    )
                )

        df = pl.DataFrame(rows)
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
        classes = [c[len("proba_") :] for c in proba_cols]
        n_classes = len(classes)

        if n_classes == 2:
            rows = _pr_rows_binary(y_true, proba_df, proba_cols, classes)
        elif average in ("micro", "macro", "weighted"):
            # Multiclass + average requested: return ONLY the summary curve.
            # The user has explicitly opted into a single-curve view, so the
            # per-class one-vs-rest rows would just be visual noise.
            rows = _compute_avg_pr(
                y_true,
                proba_df[proba_cols].to_numpy(),
                classes,
                average,
            )
        else:
            rows = _pr_rows_per_class(y_true, proba_df, proba_cols, classes)

        df = pl.DataFrame(rows)
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
        ``fraction_positive``, and ``count``. Uses sklearn's
        ``calibration_curve`` for the means/fractions and a parallel pass
        over ``y_score`` to count rows per bin.
        """
        key = self._cache_key(
            "calibration_curve",
            n_bins=n_bins,
            strategy=strategy,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("calibration_curve")
        from sklearn.calibration import calibration_curve as _ccurve

        if self._y is None:
            raise ValueError("ModelSource.calibration_curve() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if len(proba_cols) != 2:
            raise ValueError(
                f"calibration_curve() is binary-classifier only; got {len(proba_cols)} classes."
            )
        y_true = np.asarray(self._y)
        y_score = proba_df[proba_cols[1]].to_numpy()

        frac_pos, mean_pred = _ccurve(
            y_true,
            y_score,
            n_bins=n_bins,
            strategy=strategy,
        )

        if strategy == "uniform":
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        else:  # "quantile" — sklearn has already validated strategy above
            edges = np.quantile(y_score, np.linspace(0.0, 1.0, n_bins + 1))
        bin_idx = np.clip(np.digitize(y_score, edges[1:-1]), 0, n_bins - 1)
        counts_all = np.bincount(bin_idx, minlength=n_bins)
        centers = edges[:-1] + np.diff(edges) / 2.0
        used_bins = np.array([int(np.argmin(np.abs(centers - mp))) for mp in mean_pred], dtype=int)
        counts = counts_all[used_bins] if used_bins.size else np.empty(0, dtype=int)

        df = pl.DataFrame(
            {
                "mean_predicted": [float(x) for x in mean_pred],
                "fraction_positive": [float(x) for x in frac_pos],
                "count": [int(x) for x in counts],
            }
        )
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

        rows: list[dict] = []
        for i, cls in enumerate(classes):
            y_bin = (y_true == _coerce_class_label(cls, y_true.dtype)).astype(int)
            order = np.argsort(-np.asarray(proba_df[proba_cols[i]]))
            cum_pos = np.cumsum(y_bin[order])
            total_pos = max(int(cum_pos[-1]), 1) if n else 1
            pct_pop = np.arange(1, n + 1) / max(n, 1)
            gain = cum_pos / total_pos
            xs = np.concatenate([[0.0], pct_pop])
            ys = np.concatenate([[0.0], gain])
            for pp, g in zip(xs, ys):
                rows.append(
                    {
                        "percent_population": float(pp),
                        "gain": float(g),
                        "class": str(cls),
                    }
                )

        rows.append({"percent_population": 0.0, "gain": 0.0, "class": "baseline"})
        rows.append({"percent_population": 1.0, "gain": 1.0, "class": "baseline"})

        df = pl.DataFrame(rows)
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

        rows: list[dict] = []
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
            for pp, lv in zip(pct_pop, lift):
                rows.append(
                    {
                        "percent_population": float(pp),
                        "lift": float(lv),
                        "class": str(cls),
                    }
                )

        rows.append({"percent_population": 0.0, "lift": 1.0, "class": "baseline"})
        rows.append({"percent_population": 1.0, "lift": 1.0, "class": "baseline"})

        df = pl.DataFrame(rows)
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
            df = self._sweep_thresholds(
                y_true,
                y_score,
                thresholds,
                positive_class,
            )
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
            fold_dfs.append(
                self._sweep_thresholds(
                    y_true[te],
                    s,
                    thresholds,
                    positive_class,
                )
            )
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
        from sklearn.metrics import confusion_matrix as _cm

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

        cm = _cm(y_true, y_pred, labels=labels, normalize=normalize)
        rows: list[dict] = []
        for i, a in enumerate(labels):
            for j, p in enumerate(labels):
                val = float(cm[i, j])
                fmt = f"{val:.2f}" if normalize is not None else f"{int(val)}"
                rows.append(
                    {
                        "actual": str(a),
                        "predicted": str(p),
                        "value": val,
                        "value_fmt": fmt,
                    }
                )
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def _sweep_thresholds(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        thresholds: np.ndarray,
        positive_class: object,
    ) -> pl.DataFrame:
        from sklearn.metrics import precision_recall_fscore_support

        y_true_bin = (y_true == positive_class).astype(int)
        rows: list[dict] = []
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                y_true_bin,
                y_pred,
                average="binary",
                zero_division=0,
            )
            queue_rate = float((y_score >= t).mean()) if y_score.size else 0.0
            rows.append(
                {
                    "threshold": float(t),
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f1),
                    "queue_rate": queue_rate,
                }
            )
        return pl.DataFrame(rows)


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


def _pr_rows_binary(y_true, proba_df, proba_cols, classes) -> list[dict]:
    """Per-row PR curve for a binary classifier on the positive class."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    y_score = proba_df[proba_cols[1]]
    p, r, thr = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))
    thresholds_padded = np.concatenate([thr, [float("nan")]])
    return [
        {
            "precision": float(pi),
            "recall": float(ri),
            "threshold": float(ti),
            "class": classes[1],
            "ap": ap,
        }
        for pi, ri, ti in zip(p, r, thresholds_padded)
    ]


def _pr_rows_per_class(y_true, proba_df, proba_cols, classes) -> list[dict]:
    """Per-row PR curves for a multiclass classifier (one-vs-rest)."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    rows: list[dict] = []
    for i, cls in enumerate(classes):
        y_bin = (y_true == _coerce_class_label(cls, y_true.dtype)).astype(int)
        y_score = proba_df[proba_cols[i]]
        p, r, thr = precision_recall_curve(y_bin, y_score)
        try:
            ap = float(average_precision_score(y_bin, y_score))
        except ValueError:
            ap = float("nan")
        thresholds_padded = np.concatenate([thr, [float("nan")]])
        for pi, ri, ti in zip(p, r, thresholds_padded):
            rows.append(
                {
                    "precision": float(pi),
                    "recall": float(ri),
                    "threshold": float(ti),
                    "class": str(cls),
                    "ap": ap,
                }
            )
    return rows


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


def _compute_avg_pr(y_true, y_score_matrix, classes, average):
    """Return micro/macro/weighted-averaged PR-curve rows.

    Mirrors ``_compute_avg_roc`` for the precision-recall axes. ``micro``
    ravels the binarized labels + score matrix and computes a single PR
    curve via ``precision_recall_curve``. ``macro`` and ``weighted``
    interpolate each per-class precision at a shared recall grid (100
    points on ``[0, 1]``) and reduce with equal-weight (macro) or
    support-weighted (weighted) means. ``threshold`` is reported as NaN
    on every row of these summary curves — recall-grid interpolation
    doesn't preserve thresholds.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    coerced_classes = [_coerce_class_label(c, y_true.dtype) for c in classes]
    y_bin = label_binarize(y_true, classes=coerced_classes)
    if average == "micro":
        p, r, thr = precision_recall_curve(y_bin.ravel(), y_score_matrix.ravel())
        ap = float(average_precision_score(y_bin, y_score_matrix, average="micro"))
        thresholds_padded = np.concatenate([thr, [float("nan")]])
        return [
            {
                "precision": float(pi),
                "recall": float(ri),
                "threshold": float(ti),
                "class": "micro",
                "ap": ap,
            }
            for pi, ri, ti in zip(p, r, thresholds_padded)
        ]
    # macro / weighted: interpolate precision at a shared recall grid.
    # sklearn's precision_recall_curve returns recall descending from 1
    # to 0, so reverse before passing to np.interp (which requires
    # monotonically increasing xp).
    grid = np.linspace(0.0, 1.0, 100)
    precisions = []
    for i in range(y_bin.shape[1]):
        p_i, r_i, _ = precision_recall_curve(y_bin[:, i], y_score_matrix[:, i])
        order = np.argsort(r_i)
        precisions.append(np.interp(grid, r_i[order], p_i[order]))
    if average == "macro":
        weights = np.ones(len(classes)) / len(classes)
    else:  # weighted
        total = max(int(y_bin.sum()), 1)
        weights = y_bin.sum(axis=0) / total
    precision_avg = (np.array(precisions).T * weights).sum(axis=1)
    ap = float(average_precision_score(y_bin, y_score_matrix, average=average))
    return [
        {
            "precision": float(p),
            "recall": float(r),
            "threshold": float("nan"),
            "class": average,
            "ap": ap,
        }
        for p, r in zip(precision_avg, grid)
    ]


def _compute_avg_roc(y_true, y_score_matrix, classes, average, drop_intermediate):
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import label_binarize

    coerced_classes = [_coerce_class_label(c, y_true.dtype) for c in classes]
    y_bin = label_binarize(y_true, classes=coerced_classes)
    if average == "micro":
        fpr, tpr, thr = roc_curve(
            y_bin.ravel(),
            y_score_matrix.ravel(),
            drop_intermediate=drop_intermediate,
        )
        auc = float(roc_auc_score(y_bin, y_score_matrix, average="micro"))
        return [
            {"fpr": float(f), "tpr": float(t), "threshold": float(h), "class": "micro", "auc": auc}
            for f, t, h in zip(fpr, tpr, thr)
        ]
    grid = np.linspace(0.0, 1.0, 100)
    tprs = []
    for i in range(y_bin.shape[1]):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_score_matrix[:, i])
        tprs.append(np.interp(grid, fpr_i, tpr_i))
    if average == "macro":
        weights = np.ones(len(classes)) / len(classes)
    else:  # weighted
        total = max(int(y_bin.sum()), 1)
        weights = y_bin.sum(axis=0) / total
    tpr_avg = (np.array(tprs).T * weights).sum(axis=1)
    auc = float(roc_auc_score(y_bin, y_score_matrix, average=average))
    return [
        {"fpr": float(f), "tpr": float(t), "threshold": float("nan"), "class": average, "auc": auc}
        for f, t in zip(grid, tpr_avg)
    ]
