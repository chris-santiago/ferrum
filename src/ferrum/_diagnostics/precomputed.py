"""Internal precomputed source adapter for diagnostic figure functions.

``_PrecomputedSource`` is never exported — not in ``ferrum.__init__`` or
``ferrum-spec.md §3.1``.  It adapts ``(y_true, y_pred)`` arrays to the same
method protocol ``ModelSource`` exposes, so the existing chart builders in
``ferrum.plots.classification`` and ``ferrum.plots.regression`` work without
modification.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow as pa

from ferrum._core import (
    average_precision,
    calibration_kernel,
    confusion_kernel,
    pr_curve_kernel,
    prf_at_thresholds,
    roc_auc,
    roc_curve_kernel,
    studentized_residual_no_x,
)


class _PrecomputedSource:
    """Lightweight adapter that wraps raw (y_true, y_pred) arrays.

    Satisfies the subset of ``ModelSource``'s method protocol required by the
    nine in-scope diagnostic chart builders.  Delegates computation to Rust
    kernels in ``ferrum._core``; no scikit-learn is required.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels or continuous targets.
    y_pred : array-like
        Predictions.  Semantics are caller-defined:
        - Soft scores / probabilities (1-D binary or 2-D multiclass) for
          ``roc_curve``, ``pr_curve``, ``calibration_curve``,
          ``cumulative_gain``, ``lift_curve``, ``discrimination_threshold``.
        - Hard class labels (1-D) for ``confusion_matrix``.
        - Fitted values (1-D) for ``predictions``.
    """

    def __init__(self, y_true: Any, y_pred: Any) -> None:
        self._y_true_np = np.asarray(y_true)
        self._y_pred_np = np.asarray(y_pred)
        # Polars Series stored as ._y so _pr_chart_from_source can access
        # source.y / source._y for the baseline-prevalence hline.
        self._y = pl.Series(self._y_true_np.tolist())

    @property
    def y(self) -> pl.Series:
        return self._y

    # ------------------------------------------------------------------
    # Classification curves
    # ------------------------------------------------------------------

    def roc_curve(
        self,
        *,
        average: str | None = None,
        drop_intermediate: bool = True,
    ) -> pl.DataFrame:
        """ROC curve(s).  Mirrors ``ModelSource.roc_curve`` column schema."""
        y_true = self._y_true_np
        y_pred = self._y_pred_np

        if y_pred.ndim == 1:
            # Binary: y_pred is 1-D scores for the positive class.
            pos_class = str(np.unique(y_true)[-1]) if len(np.unique(y_true)) >= 2 else "1"
            frame = _roc_frame_binary(y_true, y_pred, pos_class, drop_intermediate)
            return frame
        else:
            # Multiclass: y_pred is (n_samples, n_classes); columns map to
            # sorted unique classes (sorted ascending, matching the per-class score column order).
            classes = list(np.unique(y_true))
            per_class_frames = [
                _roc_frame_binary(
                    (y_true == cls).astype(int),
                    y_pred[:, i],
                    str(cls),
                    drop_intermediate,
                )
                for i, cls in enumerate(classes)
            ]
            frames = per_class_frames
            if average in ("micro", "macro", "weighted"):
                frames = per_class_frames + [
                    _avg_roc_frame(y_true, y_pred, classes, average, drop_intermediate)
                ]
            return pl.concat(frames)

    def pr_curve(self, *, average: str | None = None) -> pl.DataFrame:
        """Precision-recall curve(s).  Mirrors ``ModelSource.pr_curve`` schema."""
        y_true = self._y_true_np
        y_pred = self._y_pred_np

        if y_pred.ndim == 1:
            return _pr_frame_binary(y_true, y_pred)
        elif average in ("micro", "macro", "weighted"):
            classes = list(np.unique(y_true))
            return _avg_pr_frame(y_true, y_pred, classes, average)
        else:
            classes = list(np.unique(y_true))
            frames = [
                _pr_frame_binary(
                    (y_true == cls).astype(int),
                    y_pred[:, i],
                    str(cls),
                )
                for i, cls in enumerate(classes)
            ]
            return pl.concat(frames)

    def calibration_curve(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> pl.DataFrame:
        """Reliability diagram bins.  Mirrors ``ModelSource.calibration_curve`` schema."""
        y_true = self._y_true_np
        y_pred = self._y_pred_np  # 1-D probabilities for positive class

        yt_arrow = pa.array(y_true.astype(np.float64), type=pa.float64())
        yp_arrow = pa.array(y_pred.astype(np.float64), type=pa.float64())
        rb = calibration_kernel(yt_arrow, yp_arrow, n_bins, strategy)
        return cast("pl.DataFrame", pl.from_arrow(rb))

    def cumulative_gain(self) -> pl.DataFrame:
        """Cumulative-gain curve.  Mirrors ``ModelSource.cumulative_gain`` schema."""
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        n = len(y_true)
        classes = list(np.unique(y_true))

        per_class_parts: list[pl.DataFrame] = []
        if y_pred.ndim == 1:
            for cls in classes:
                y_bin = (y_true == cls).astype(int)
                scores = y_pred
                order = np.argsort(-scores)
                cum_pos = np.cumsum(y_bin[order])
                total_pos = max(int(cum_pos[-1]), 1) if n else 1
                pct_pop = np.arange(1, n + 1) / max(n, 1)
                gain = cum_pos / total_pos
                xs = np.concatenate([[0.0], pct_pop])
                ys = np.concatenate([[0.0], gain])
                per_class_parts.append(
                    pl.DataFrame(
                        {
                            "percent_population": xs,
                            "gain": ys,
                            "class": [str(cls)] * len(xs),
                        }
                    )
                )
        else:
            for i, cls in enumerate(classes):
                y_bin = (y_true == cls).astype(int)
                order = np.argsort(-y_pred[:, i])
                cum_pos = np.cumsum(y_bin[order])
                total_pos = max(int(cum_pos[-1]), 1) if n else 1
                pct_pop = np.arange(1, n + 1) / max(n, 1)
                gain = cum_pos / total_pos
                xs = np.concatenate([[0.0], pct_pop])
                ys = np.concatenate([[0.0], gain])
                per_class_parts.append(
                    pl.DataFrame(
                        {
                            "percent_population": xs,
                            "gain": ys,
                            "class": [str(cls)] * len(xs),
                        }
                    )
                )

        baseline = pl.DataFrame(
            {
                "percent_population": [0.0, 1.0],
                "gain": [0.0, 1.0],
                "class": ["baseline", "baseline"],
            }
        )
        return pl.concat(per_class_parts + [baseline])

    def lift_curve(self) -> pl.DataFrame:
        """Lift curve.  Mirrors ``ModelSource.lift_curve`` schema."""
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        n = len(y_true)
        classes = list(np.unique(y_true))

        per_class_parts: list[pl.DataFrame] = []
        if y_pred.ndim == 1:
            for cls in classes:
                y_bin = (y_true == cls).astype(int)
                base_rate = float(y_bin.mean()) if n else 0.0
                if base_rate == 0.0:
                    continue
                order = np.argsort(-y_pred)
                cum_pos = np.cumsum(y_bin[order])
                denom = np.arange(1, n + 1)
                cum_rate = cum_pos / denom
                lift = cum_rate / base_rate
                pct_pop = denom / n
                per_class_parts.append(
                    pl.DataFrame(
                        {
                            "percent_population": pct_pop.astype(float),
                            "lift": lift.astype(float),
                            "class": [str(cls)] * len(pct_pop),
                        }
                    )
                )
        else:
            for i, cls in enumerate(classes):
                y_bin = (y_true == cls).astype(int)
                base_rate = float(y_bin.mean()) if n else 0.0
                if base_rate == 0.0:
                    continue
                order = np.argsort(-y_pred[:, i])
                cum_pos = np.cumsum(y_bin[order])
                denom = np.arange(1, n + 1)
                cum_rate = cum_pos / denom
                lift = cum_rate / base_rate
                pct_pop = denom / n
                per_class_parts.append(
                    pl.DataFrame(
                        {
                            "percent_population": pct_pop.astype(float),
                            "lift": lift.astype(float),
                            "class": [str(cls)] * len(pct_pop),
                        }
                    )
                )

        baseline = pl.DataFrame(
            {
                "percent_population": [0.0, 1.0],
                "lift": [1.0, 1.0],
                "class": ["baseline", "baseline"],
            }
        )
        return pl.concat(per_class_parts + [baseline])

    def confusion_matrix(self, *, normalize: str | None = None) -> pl.DataFrame:
        """Confusion matrix in long form.  Mirrors ``ModelSource.confusion_matrix`` schema.

        ``y_pred`` must be 1-D hard class labels.
        """
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)

        # Encode labels as integer codes for the Rust kernel.
        # Use .get(v, -1) so that out-of-vocabulary values (including NaN) map
        # to code -1, which confusion_kernel drops — matching the behaviour of
        # _confusion_matrix_columnar in sources/_classification.py.
        label_to_code = {lbl: i for i, lbl in enumerate(labels)}
        yt_codes = np.array([label_to_code.get(v, -1) for v in y_true.tolist()], dtype=np.int64)
        yp_codes = np.array([label_to_code.get(v, -1) for v in y_pred.tolist()], dtype=np.int64)
        label_codes = np.arange(len(labels), dtype=np.int64)

        yt_arrow = pa.array(yt_codes, type=pa.int64())
        yp_arrow = pa.array(yp_codes, type=pa.int64())
        labels_arrow = pa.array(label_codes, type=pa.int64())
        normalize_str = normalize if normalize is not None else ""

        rb = confusion_kernel(yt_arrow, yp_arrow, labels_arrow, normalize_str)
        cm_df = cast("pl.DataFrame", pl.from_arrow(rb))

        # Map integer row/col indices back to original label strings and
        # build the long-form output with value_fmt.
        label_strs = [str(lbl) for lbl in labels]
        actual_col = [label_strs[r] for r in cm_df["row"].to_list()]
        predicted_col = [label_strs[c] for c in cm_df["col"].to_list()]
        values = cm_df["value"].to_list()
        fmt_col = [f"{v:.2f}" if normalize is not None else f"{int(v)}" for v in values]
        return pl.DataFrame(
            {
                "actual": actual_col,
                "predicted": predicted_col,
                "value": values,
                "value_fmt": fmt_col,
            }
        )

    def predictions(self) -> pl.DataFrame:
        """Return y_true, y_pred, residual, studentized_residual, cooks_distance, leverage.

        ``y_pred`` must be 1-D fitted values.  ``studentized_residual`` is
        computed without the design matrix (``ferrum._core.studentized_residual_no_x``),
        matching the non-linear estimator path in ``ModelSource``.
        ``cooks_distance`` and ``leverage`` are NaN (no design matrix available).
        """
        y_true = self._y_true_np.astype(np.float64)
        y_pred = self._y_pred_np.astype(np.float64)
        residual = y_true - y_pred

        yt_arrow = pa.array(y_true, type=pa.float64())
        yp_arrow = pa.array(y_pred, type=pa.float64())
        stud_arrow = studentized_residual_no_x(yt_arrow, yp_arrow)
        stud = np.asarray(pa.array(stud_arrow), dtype=np.float64)
        nan_col = np.full_like(y_pred, np.nan)

        return pl.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": residual,
                "studentized_residual": stud,
                "cooks_distance": nan_col,
                "leverage": nan_col,
            }
        )

    def discrimination_threshold(
        self,
        *,
        n_thresholds: int = 50,
        cv: Any = None,
    ) -> pl.DataFrame:
        """Threshold sweep.  Mirrors ``ModelSource.discrimination_threshold`` schema.

        ``y_pred`` must be 1-D soft scores for the positive class.
        ``cv`` is not supported on the precomputed path (no model to re-fit).
        """
        if cv is not None:
            raise ValueError(
                "discrimination_threshold_chart: cv= is not supported with "
                "precomputed y_true/y_pred inputs — cross-validation requires "
                "a fitted model.  Pass cv=None or use the model-backed path."
            )

        y_true = self._y_true_np
        y_pred = self._y_pred_np  # 1-D positive-class scores

        unique = np.unique(y_true)
        if len(unique) != 2:
            raise ValueError(
                "discrimination_threshold_chart with precomputed inputs requires "
                f"binary y_true; got {len(unique)} unique classes."
            )
        positive_class = unique[-1]
        y_true_bin = (y_true == positive_class).astype(int)
        thresholds_np = np.linspace(0.0, 1.0, n_thresholds)

        yt_arrow = pa.array(y_true_bin.astype(np.float64), type=pa.float64())
        yp_arrow = pa.array(y_pred.astype(np.float64), type=pa.float64())
        thr_arrow = pa.array(thresholds_np, type=pa.float64())

        rb = prf_at_thresholds(yt_arrow, yp_arrow, thr_arrow)
        df = cast("pl.DataFrame", pl.from_arrow(rb))

        return pl.DataFrame(
            {
                "threshold": thresholds_np,
                "precision": df["precision"].to_list(),
                "recall": df["recall"].to_list(),
                "f1": df["f1"].to_list(),
                "queue_rate": df["queue_rate"].to_list(),
            }
        )


# ---------------------------------------------------------------------------
# Module-private helpers (not part of any public API)
# ---------------------------------------------------------------------------


def _roc_frame_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_label: str,
    drop_intermediate: bool,
) -> pl.DataFrame:
    """Return a ROC DataFrame for one binary class."""
    yt_arrow = pa.array(y_true.astype(np.float64), type=pa.float64())
    ys_arrow = pa.array(y_score.astype(np.float64), type=pa.float64())
    rb = roc_curve_kernel(yt_arrow, ys_arrow, drop_intermediate)
    curve = cast("pl.DataFrame", pl.from_arrow(rb))
    auc_val = float(roc_auc(yt_arrow, ys_arrow))
    return curve.with_columns(
        [
            pl.lit(class_label).alias("class"),
            pl.lit(auc_val).alias("auc"),
        ]
    ).select(["fpr", "tpr", "threshold", "class", "auc"])


def _avg_roc_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    average: str,
    drop_intermediate: bool,
) -> pl.DataFrame:
    """Return a single averaged ROC curve row-set (micro, macro, or weighted)."""
    y_bin = _one_hot(y_true, classes)

    if average == "micro":
        yt_arrow = pa.array(y_bin.ravel().astype(np.float64), type=pa.float64())
        ys_arrow = pa.array(y_pred.ravel().astype(np.float64), type=pa.float64())
        rb = roc_curve_kernel(yt_arrow, ys_arrow, drop_intermediate)
        curve = cast("pl.DataFrame", pl.from_arrow(rb))
        auc_val = float(roc_auc(yt_arrow, ys_arrow))
        return curve.with_columns(
            [
                pl.lit("micro").alias("class"),
                pl.lit(auc_val).alias("auc"),
            ]
        ).select(["fpr", "tpr", "threshold", "class", "auc"])

    # macro / weighted: interpolate each per-class curve onto a shared FPR grid
    grid = np.linspace(0.0, 1.0, 100)
    tprs = []
    auc_per_class = []
    for i in range(y_bin.shape[1]):
        yt_i = pa.array(y_bin[:, i].astype(np.float64), type=pa.float64())
        ys_i = pa.array(y_pred[:, i].astype(np.float64), type=pa.float64())
        rb_i = roc_curve_kernel(yt_i, ys_i, drop_intermediate)
        c_i = cast("pl.DataFrame", pl.from_arrow(rb_i))
        tprs.append(np.interp(grid, c_i["fpr"].to_numpy(), c_i["tpr"].to_numpy()))
        auc_per_class.append(float(roc_auc(yt_i, ys_i)))

    if average == "macro":
        weights = np.ones(len(classes)) / len(classes)
    else:  # weighted
        support = y_bin.sum(axis=0)
        total = max(int(support.sum()), 1)
        weights = support / total

    tpr_avg = (np.array(tprs).T * weights).sum(axis=1)
    auc_val = float(np.average(auc_per_class, weights=weights))

    n = len(grid)
    return pl.DataFrame(
        {
            "fpr": grid,
            "tpr": tpr_avg,
            "threshold": [float("nan")] * n,
            "class": [average] * n,
            "auc": [auc_val] * n,
        }
    )


def _pr_frame_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_label: str | None = None,
) -> pl.DataFrame:
    """Return a PR DataFrame for one binary class.

    The Rust kernel already emits the final NaN threshold; do not pad again.
    """
    if class_label is None:
        class_label = str(np.unique(y_true)[-1]) if len(np.unique(y_true)) >= 2 else "1"
    yt_arrow = pa.array(y_true.astype(np.float64), type=pa.float64())
    ys_arrow = pa.array(y_score.astype(np.float64), type=pa.float64())
    rb = pr_curve_kernel(yt_arrow, ys_arrow)
    curve = cast("pl.DataFrame", pl.from_arrow(rb))
    ap_val = float(average_precision(yt_arrow, ys_arrow))
    return curve.with_columns(
        [
            pl.lit(class_label).alias("class"),
            pl.lit(ap_val).alias("ap"),
        ]
    ).select(["precision", "recall", "threshold", "class", "ap"])


def _avg_pr_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    average: str,
) -> pl.DataFrame:
    """Return a single averaged PR curve row-set (micro, macro, or weighted)."""
    y_bin = _one_hot(y_true, classes)

    if average == "micro":
        yt_arrow = pa.array(y_bin.ravel().astype(np.float64), type=pa.float64())
        ys_arrow = pa.array(y_pred.ravel().astype(np.float64), type=pa.float64())
        rb = pr_curve_kernel(yt_arrow, ys_arrow)
        curve = cast("pl.DataFrame", pl.from_arrow(rb))
        ap_val = float(average_precision(yt_arrow, ys_arrow))
        return curve.with_columns(
            [
                pl.lit("micro").alias("class"),
                pl.lit(ap_val).alias("ap"),
            ]
        ).select(["precision", "recall", "threshold", "class", "ap"])

    # macro / weighted: interpolate each per-class P-R curve onto a shared
    # recall grid, then take a weighted mean.
    grid = np.linspace(0.0, 1.0, 100)
    precisions = []
    ap_per_class = []
    for i in range(y_bin.shape[1]):
        yt_i = pa.array(y_bin[:, i].astype(np.float64), type=pa.float64())
        ys_i = pa.array(y_pred[:, i].astype(np.float64), type=pa.float64())
        rb_i = pr_curve_kernel(yt_i, ys_i)
        c_i = cast("pl.DataFrame", pl.from_arrow(rb_i))
        r_i = c_i["recall"].to_numpy()
        p_i = c_i["precision"].to_numpy()
        order = np.argsort(r_i)
        precisions.append(np.interp(grid, r_i[order], p_i[order]))
        ap_per_class.append(float(average_precision(yt_i, ys_i)))

    if average == "macro":
        weights = np.ones(len(classes)) / len(classes)
    else:  # weighted
        support = y_bin.sum(axis=0)
        total = max(int(support.sum()), 1)
        weights = support / total

    precision_avg = (np.array(precisions).T * weights).sum(axis=1)
    ap_val = float(np.average(ap_per_class, weights=weights))

    n = len(grid)
    return pl.DataFrame(
        {
            "precision": precision_avg,
            "recall": grid,
            "threshold": [float("nan")] * n,
            "class": [average] * n,
            "ap": [ap_val] * n,
        }
    )


def _one_hot(y_true: np.ndarray, classes: list) -> np.ndarray:
    """One-hot encode y_true for the given sorted class list.

    Equivalent to one-hot encoding with columns ordered by ``classes``
    for a fully multiclass (len(classes) >= 3) input.
    """
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    codes = np.array([class_to_idx[v] for v in y_true.tolist()], dtype=int)
    onehot = np.zeros((len(y_true), n_classes), dtype=int)
    onehot[np.arange(len(y_true)), codes] = 1
    return onehot
