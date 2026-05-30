"""Phase 10b — classification curves (ROC, PR, calibration, gain, lift, discrimination threshold, confusion matrix)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow as pa

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
        classes = [c[len("proba_") :] for c in proba_cols]
        n_classes = len(classes)

        if n_classes == 2:
            df = _roc_binary(y_true, proba_df, proba_cols, classes, drop_intermediate)
        else:
            per_class_frames = [
                _roc_one_class(y_true, proba_df, proba_cols, i, cls, drop_intermediate)
                for i, cls in enumerate(classes)
            ]
            if average in ("micro", "macro", "weighted"):
                avg_frame = _roc_average(
                    y_true,
                    proba_df[proba_cols].to_numpy(),
                    classes,
                    average,
                    drop_intermediate,
                )
                df = pl.concat([*per_class_frames, avg_frame], how="vertical")
            else:
                df = pl.concat(per_class_frames, how="vertical")

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
            df = _pr_binary(y_true, proba_df, proba_cols, classes)
        elif average in ("micro", "macro", "weighted"):
            df = _pr_average(
                y_true,
                proba_df[proba_cols].to_numpy(),
                classes,
                average,
            )
        else:
            frames = [
                _pr_one_class(y_true, proba_df, proba_cols, i, cls) for i, cls in enumerate(classes)
            ]
            df = pl.concat(frames, how="vertical")

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
        from ferrum._core import calibration_kernel

        y_true = np.asarray(self._y, dtype=np.float64)
        y_score = proba_df[proba_cols[1]].to_numpy().astype(np.float64)

        rb = calibration_kernel(
            pa.array(y_true, type=pa.float64()),
            pa.array(y_score, type=pa.float64()),
            n_bins,
            strategy,
        )
        df = cast("pl.DataFrame", pl.from_arrow(rb))
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
            df = _sweep_thresholds(y_true, y_score, thresholds, positive_class)
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
            fold_dfs.append(_sweep_thresholds(y_true[te], s, thresholds, positive_class))
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

        df = _confusion_matrix_columnar(y_true, y_pred, labels, normalize)
        self._cache[key] = df
        return df


# ---------------------------------------------------------------------------
# Module-level kernel-backed helpers
# ---------------------------------------------------------------------------


def _roc_binary(
    y_true: np.ndarray,
    proba_df: pl.DataFrame,
    proba_cols: list[str],
    classes: list[str],
    drop_intermediate: bool,
) -> pl.DataFrame:
    """ROC curve for a binary classifier — calls roc_curve_kernel."""
    from ferrum._core import roc_curve_kernel, roc_auc

    y_score = proba_df[proba_cols[1]].to_numpy().astype(np.float64)
    yt = pa.array(y_true.astype(np.float64), type=pa.float64())
    ys = pa.array(y_score, type=pa.float64())

    rb = roc_curve_kernel(yt, ys, drop_intermediate)
    base = cast("pl.DataFrame", pl.from_arrow(rb))
    try:
        auc = float(roc_auc(yt, ys))
    except Exception:
        auc = float("nan")
    return base.with_columns(
        pl.lit(classes[1]).alias("class"),
        pl.lit(auc).alias("auc"),
    )


def _roc_one_class(
    y_true: np.ndarray,
    proba_df: pl.DataFrame,
    proba_cols: list[str],
    idx: int,
    cls: str,
    drop_intermediate: bool,
) -> pl.DataFrame:
    """One-vs-rest ROC curve for a single class in a multiclass problem."""
    from ferrum._core import roc_curve_kernel, roc_auc

    y_bin = (y_true == _coerce_class_label(cls, y_true.dtype)).astype(np.float64)
    y_score = proba_df[proba_cols[idx]].to_numpy().astype(np.float64)
    yt = pa.array(y_bin, type=pa.float64())
    ys = pa.array(y_score, type=pa.float64())

    rb = roc_curve_kernel(yt, ys, drop_intermediate)
    base = cast("pl.DataFrame", pl.from_arrow(rb))
    try:
        auc = float(roc_auc(yt, ys))
    except Exception:
        auc = float("nan")
    return base.with_columns(
        pl.lit(str(cls)).alias("class"),
        pl.lit(auc).alias("auc"),
    )


def _roc_average(
    y_true: np.ndarray,
    y_score_matrix: np.ndarray,
    classes: list[str],
    average: str,
    drop_intermediate: bool,
) -> pl.DataFrame:
    """Micro/macro/weighted-averaged ROC curve — sklearn-free, columnar."""
    from ferrum._core import roc_curve_kernel, roc_auc

    coerced = [_coerce_class_label(c, y_true.dtype) for c in classes]
    y_bin = _label_binarize(y_true, coerced)

    if average == "micro":
        yt = pa.array(y_bin.ravel().astype(np.float64), type=pa.float64())
        ys = pa.array(y_score_matrix.ravel().astype(np.float64), type=pa.float64())
        rb = roc_curve_kernel(yt, ys, drop_intermediate)
        base = cast("pl.DataFrame", pl.from_arrow(rb))
        try:
            auc = float(roc_auc(yt, ys))
        except Exception:
            auc = float("nan")
        return base.with_columns(
            pl.lit("micro").alias("class"),
            pl.lit(auc).alias("auc"),
        )

    # macro / weighted: interpolate TPR at shared FPR grid, weighted mean.
    grid = np.linspace(0.0, 1.0, 100)
    tprs: list[np.ndarray] = []
    per_class_auc: list[float] = []
    for i in range(y_bin.shape[1]):
        yt_i = pa.array(y_bin[:, i].astype(np.float64), type=pa.float64())
        ys_i = pa.array(y_score_matrix[:, i].astype(np.float64), type=pa.float64())
        rb_i = cast("pl.DataFrame", pl.from_arrow(roc_curve_kernel(yt_i, ys_i, drop_intermediate)))
        fpr_i = rb_i["fpr"].to_numpy()
        tpr_i = rb_i["tpr"].to_numpy()
        tprs.append(np.interp(grid, fpr_i, tpr_i))
        try:
            per_class_auc.append(float(roc_auc(yt_i, ys_i)))
        except Exception:
            per_class_auc.append(float("nan"))

    if average == "macro":
        weights = np.ones(len(classes)) / len(classes)
    else:  # weighted
        support = y_bin.sum(axis=0)
        total = max(int(support.sum()), 1)
        weights = support / total

    tpr_avg = (np.array(tprs).T * weights).sum(axis=1)
    auc = float(np.dot(np.array(per_class_auc), weights))
    n = len(grid)
    return pl.DataFrame(
        {
            "fpr": grid,
            "tpr": tpr_avg,
            "threshold": np.full(n, float("nan")),
            "class": [average] * n,
            "auc": np.full(n, auc),
        }
    )


def _pr_binary(
    y_true: np.ndarray,
    proba_df: pl.DataFrame,
    proba_cols: list[str],
    classes: list[str],
) -> pl.DataFrame:
    """PR curve for a binary classifier — calls pr_curve_kernel."""
    from ferrum._core import pr_curve_kernel, average_precision

    y_score = proba_df[proba_cols[1]].to_numpy().astype(np.float64)
    yt = pa.array(y_true.astype(np.float64), type=pa.float64())
    ys = pa.array(y_score, type=pa.float64())

    rb = pr_curve_kernel(yt, ys)
    base = cast("pl.DataFrame", pl.from_arrow(rb))
    try:
        ap = float(average_precision(yt, ys))
    except Exception:
        ap = float("nan")
    return base.with_columns(
        pl.lit(classes[1]).alias("class"),
        pl.lit(ap).alias("ap"),
    )


def _pr_one_class(
    y_true: np.ndarray,
    proba_df: pl.DataFrame,
    proba_cols: list[str],
    idx: int,
    cls: str,
) -> pl.DataFrame:
    """One-vs-rest PR curve for a single class in a multiclass problem."""
    from ferrum._core import pr_curve_kernel, average_precision

    y_bin = (y_true == _coerce_class_label(cls, y_true.dtype)).astype(np.float64)
    y_score = proba_df[proba_cols[idx]].to_numpy().astype(np.float64)
    yt = pa.array(y_bin, type=pa.float64())
    ys = pa.array(y_score, type=pa.float64())

    rb = pr_curve_kernel(yt, ys)
    base = cast("pl.DataFrame", pl.from_arrow(rb))
    try:
        ap = float(average_precision(yt, ys))
    except Exception:
        ap = float("nan")
    return base.with_columns(
        pl.lit(str(cls)).alias("class"),
        pl.lit(ap).alias("ap"),
    )


def _pr_average(
    y_true: np.ndarray,
    y_score_matrix: np.ndarray,
    classes: list[str],
    average: str,
) -> pl.DataFrame:
    """Micro/macro/weighted-averaged PR curve — sklearn-free, columnar.

    Micro: ravel binarized labels + score matrix → one kernel call.
    Macro/weighted: interpolate per-class precision at a shared recall
    grid (100 points on [0, 1]) → weighted mean.
    ``threshold`` is NaN on every row of macro/weighted summaries.
    """
    from ferrum._core import pr_curve_kernel, average_precision

    coerced = [_coerce_class_label(c, y_true.dtype) for c in classes]
    y_bin = _label_binarize(y_true, coerced)

    if average == "micro":
        yt = pa.array(y_bin.ravel().astype(np.float64), type=pa.float64())
        ys = pa.array(y_score_matrix.ravel().astype(np.float64), type=pa.float64())
        rb = pr_curve_kernel(yt, ys)
        base = cast("pl.DataFrame", pl.from_arrow(rb))
        try:
            # Micro AP: average_precision on raveled arrays (matches sklearn).
            ap = float(average_precision(yt, ys))
        except Exception:
            ap = float("nan")
        return base.with_columns(
            pl.lit("micro").alias("class"),
            pl.lit(ap).alias("ap"),
        )

    # macro / weighted: interpolate precision at a shared recall grid.
    grid = np.linspace(0.0, 1.0, 100)
    precisions: list[np.ndarray] = []
    per_class_ap: list[float] = []
    for i in range(y_bin.shape[1]):
        yt_i = pa.array(y_bin[:, i].astype(np.float64), type=pa.float64())
        ys_i = pa.array(y_score_matrix[:, i].astype(np.float64), type=pa.float64())
        rb_i = cast("pl.DataFrame", pl.from_arrow(pr_curve_kernel(yt_i, ys_i)))
        r_i = rb_i["recall"].to_numpy()
        p_i = rb_i["precision"].to_numpy()
        # sklearn's curve has recall descending from 1 to 0; sort ascending for interp.
        order = np.argsort(r_i)
        precisions.append(np.interp(grid, r_i[order], p_i[order]))
        try:
            per_class_ap.append(float(average_precision(yt_i, ys_i)))
        except Exception:
            per_class_ap.append(float("nan"))

    if average == "macro":
        weights = np.ones(len(classes)) / len(classes)
    else:  # weighted
        support = y_bin.sum(axis=0)
        total = max(int(support.sum()), 1)
        weights = support / total

    precision_avg = (np.array(precisions).T * weights).sum(axis=1)
    ap = float(np.dot(np.array(per_class_ap), weights))
    n = len(grid)
    return pl.DataFrame(
        {
            "precision": precision_avg,
            "recall": grid,
            "threshold": np.full(n, float("nan")),
            "class": [average] * n,
            "ap": np.full(n, ap),
        }
    )


def _confusion_matrix_columnar(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    normalize: str | None,
) -> pl.DataFrame:
    """Build a long-form confusion matrix DataFrame via the confusion_kernel.

    The kernel takes integer-encoded labels; this function encodes/decodes
    so arbitrary string/categorical labels work correctly.
    """
    from ferrum._core import confusion_kernel

    label_strs = [str(lbl) for lbl in labels]
    # Build a lookup from label value → integer code.
    label_to_code: dict = {lbl: i for i, lbl in enumerate(labels)}
    codes = np.array([label_to_code.get(v, -1) for v in y_true.tolist()], dtype=np.int64)
    pred_codes = np.array([label_to_code.get(v, -1) for v in y_pred.tolist()], dtype=np.int64)
    label_codes = np.arange(len(labels), dtype=np.int64)

    norm_arg = "" if normalize is None else normalize
    rb = confusion_kernel(
        pa.array(codes, type=pa.int64()),
        pa.array(pred_codes, type=pa.int64()),
        pa.array(label_codes, type=pa.int64()),
        norm_arg,
    )
    cm_df = cast("pl.DataFrame", pl.from_arrow(rb))

    # Map integer row/col indices back to original label strings.
    n = len(labels)
    actual_col = [label_strs[int(r)] for r in cm_df["row"].to_list()]
    predicted_col = [label_strs[int(c)] for c in cm_df["col"].to_list()]
    value_col = cm_df["value"].to_list()
    value_fmt_col = [f"{v:.2f}" if normalize is not None else f"{int(v)}" for v in value_col]

    return pl.DataFrame(
        {
            "actual": actual_col,
            "predicted": predicted_col,
            "value": value_col,
            "value_fmt": value_fmt_col,
        }
    )


def _sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    positive_class: object,
) -> pl.DataFrame:
    """Threshold sweep via prf_at_thresholds Rust kernel.

    The kernel returns precision, recall, f1, queue_rate in threshold order.
    We prepend the threshold column to match the expected output schema
    (threshold, precision, recall, f1, queue_rate).
    """
    from ferrum._core import prf_at_thresholds

    y_true_bin = (y_true == positive_class).astype(np.float64)
    rb = prf_at_thresholds(
        pa.array(y_true_bin, type=pa.float64()),
        pa.array(y_score.astype(np.float64), type=pa.float64()),
        pa.array(thresholds.astype(np.float64), type=pa.float64()),
    )
    metrics = cast("pl.DataFrame", pl.from_arrow(rb))
    return metrics.with_columns(pl.Series("threshold", thresholds.astype(np.float64))).select(
        ["threshold", "precision", "recall", "f1", "queue_rate"]
    )


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


def _label_binarize(y: np.ndarray, classes: list) -> np.ndarray:
    """One-hot encode y against an ordered list of class values.

    Returns an (n_samples, n_classes) int array. Rows for values not in
    ``classes`` will be all-zero.
    """
    n = len(y)
    k = len(classes)
    out = np.zeros((n, k), dtype=int)
    for j, cls in enumerate(classes):
        out[:, j] = (y == cls).astype(int)
    return out


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
