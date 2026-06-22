"""Shared kernel-calling frame assembly for classification diagnostics.

This module is the single home for building the seven diagnostic curve/matrix
frames (ROC, PR, calibration, confusion, threshold sweep, cumulative gain, lift)
from raw numpy arrays via the Rust kernels in ``ferrum._core``. Both diagnostic
source adapters — ``_PrecomputedSource`` (array inputs) and
``ClassificationCurvesMixin`` (model inputs) — delegate all frame assembly here,
so the Python-side curve math has exactly one implementation per concept.

The callers own input concerns the kernels cannot: obtaining the score arrays
(the model path runs ``predict_proba``/``decision_function`` through
scikit-learn) and coercing class labels to ``y``'s dtype. They pass resolved
class labels plus float arrays into these builders.

No scikit-learn import lives here — the kernels are sklearn-free.

Python-facing DataFrame schemas (the chart-builder contract, unchanged):

- ROC: ``fpr, tpr, threshold, class, auc``
- PR: ``precision, recall, threshold, class, ap``
- Calibration: ``mean_predicted, fraction_positive, count``
- Confusion: ``actual, predicted, value, value_fmt``
- Discrimination threshold: ``threshold, precision, recall, f1, queue_rate``
- Cumulative gain: ``percent_population, gain, class``
- Lift: ``percent_population, lift, class``
"""

from __future__ import annotations

from typing import cast

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
)

_AVG_GRID_POINTS = 100


def one_hot(y_true: np.ndarray, classes: list) -> np.ndarray:
    """One-hot encode ``y_true`` against an ordered list of class values.

    Returns an ``(n_samples, n_classes)`` int array with column ``j`` set where
    ``y_true == classes[j]``. Rows whose value is not in ``classes`` are
    all-zero (out-of-vocabulary values are dropped, not raised on).
    """
    n = len(y_true)
    k = len(classes)
    out = np.zeros((n, k), dtype=int)
    for j, cls in enumerate(classes):
        out[:, j] = (y_true == cls).astype(int)
    return out


def _f64(arr: np.ndarray) -> pa.Array:
    """Build a float64 Arrow array from a numpy array."""
    return pa.array(np.asarray(arr).astype(np.float64), type=pa.float64())


def _avg_weights(y_bin: np.ndarray, average: str, n_classes: int) -> np.ndarray:
    """Per-class weights for macro (uniform) or weighted (support) averaging."""
    if average == "macro":
        return np.ones(n_classes) / n_classes
    support = y_bin.sum(axis=0)
    total = max(int(support.sum()), 1)
    return support / total


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------


def _roc_one(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    class_label: str,
    *,
    drop_intermediate: bool,
) -> pl.DataFrame:
    """ROC frame for one binary/one-vs-rest curve with broadcast AUC."""
    yt = _f64(y_true_bin)
    ys = _f64(y_score)
    curve = cast("pl.DataFrame", pl.from_arrow(roc_curve_kernel(yt, ys, drop_intermediate)))
    auc = float(roc_auc(yt, ys))
    return curve.with_columns(
        pl.lit(class_label).alias("class"),
        pl.lit(auc).alias("auc"),
    ).select(["fpr", "tpr", "threshold", "class", "auc"])


def _roc_average(
    y_bin: np.ndarray,
    y_score_matrix: np.ndarray,
    average: str,
    *,
    drop_intermediate: bool,
) -> pl.DataFrame:
    """Micro/macro/weighted-averaged ROC summary curve."""
    if average == "micro":
        return _roc_one(
            y_bin.ravel(),
            y_score_matrix.ravel(),
            "micro",
            drop_intermediate=drop_intermediate,
        )

    # macro / weighted: interpolate TPR onto a shared FPR grid, weighted mean.
    grid = np.linspace(0.0, 1.0, _AVG_GRID_POINTS)
    tprs: list[np.ndarray] = []
    per_class_auc: list[float] = []
    for i in range(y_bin.shape[1]):
        yt_i = _f64(y_bin[:, i])
        ys_i = _f64(y_score_matrix[:, i])
        c_i = cast("pl.DataFrame", pl.from_arrow(roc_curve_kernel(yt_i, ys_i, drop_intermediate)))
        tprs.append(np.interp(grid, c_i["fpr"].to_numpy(), c_i["tpr"].to_numpy()))
        per_class_auc.append(float(roc_auc(yt_i, ys_i)))

    weights = _avg_weights(y_bin, average, y_bin.shape[1])
    tpr_avg = (np.array(tprs).T * weights).sum(axis=1)
    auc = float(np.average(per_class_auc, weights=weights))
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


def roc_frame(
    y_true: np.ndarray,
    y_score_matrix: np.ndarray,
    class_values: list,
    class_labels: list[str],
    *,
    average: str | None,
    drop_intermediate: bool,
) -> pl.DataFrame:
    """Assemble the ROC frame for binary or multiclass inputs.

    Parameters
    ----------
    y_true
        Raw ground-truth labels (caller's dtype).
    y_score_matrix
        ``(n_samples, n_classes)`` per-class scores. Column ``i`` aligns with
        ``class_values[i]`` / ``class_labels[i]``.
    class_values
        Per-class comparison values in score-column order, in ``y_true``'s
        dtype. Used to binarize ``y_true`` (``y_true == value``). The caller
        owns label-to-value coercion.
    class_labels
        Per-class string labels for the ``class`` discriminator column, in
        score-column order. Length 2 selects the binary path (curve drawn on
        the positive/second class); length >= 3 selects the multiclass
        one-vs-rest path.
    average
        ``None`` for per-class curves; ``"micro"``/``"macro"``/``"weighted"``
        to additionally append a summary curve (multiclass only).
    drop_intermediate
        Forwarded to ``roc_curve_kernel``.
    """
    if len(class_labels) == 2:
        return _roc_one(
            y_true,
            y_score_matrix[:, 1],
            class_labels[1],
            drop_intermediate=drop_intermediate,
        )

    y_bin = one_hot(y_true, class_values)
    per_class = [
        _roc_one(
            y_bin[:, i],
            y_score_matrix[:, i],
            label,
            drop_intermediate=drop_intermediate,
        )
        for i, label in enumerate(class_labels)
    ]
    if average in ("micro", "macro", "weighted"):
        per_class.append(
            _roc_average(y_bin, y_score_matrix, average, drop_intermediate=drop_intermediate)
        )
    return pl.concat(per_class, how="vertical")


# ---------------------------------------------------------------------------
# PR
# ---------------------------------------------------------------------------


def _pr_one(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    class_label: str,
) -> pl.DataFrame:
    """PR frame for one binary/one-vs-rest curve with broadcast AP.

    The Rust kernel already emits the trailing NaN threshold; no padding here.
    """
    yt = _f64(y_true_bin)
    ys = _f64(y_score)
    curve = cast("pl.DataFrame", pl.from_arrow(pr_curve_kernel(yt, ys)))
    ap = float(average_precision(yt, ys))
    return curve.with_columns(
        pl.lit(class_label).alias("class"),
        pl.lit(ap).alias("ap"),
    ).select(["precision", "recall", "threshold", "class", "ap"])


def _pr_average(
    y_bin: np.ndarray,
    y_score_matrix: np.ndarray,
    average: str,
) -> pl.DataFrame:
    """Micro/macro/weighted-averaged PR summary curve.

    Micro ravels the binarized labels into one kernel call. Macro/weighted
    interpolate per-class precision onto a shared recall grid (100 points),
    then take a weighted mean; ``threshold`` is NaN on every summary row.
    """
    if average == "micro":
        return _pr_one(y_bin.ravel(), y_score_matrix.ravel(), "micro")

    grid = np.linspace(0.0, 1.0, _AVG_GRID_POINTS)
    precisions: list[np.ndarray] = []
    per_class_ap: list[float] = []
    for i in range(y_bin.shape[1]):
        yt_i = _f64(y_bin[:, i])
        ys_i = _f64(y_score_matrix[:, i])
        c_i = cast("pl.DataFrame", pl.from_arrow(pr_curve_kernel(yt_i, ys_i)))
        r_i = c_i["recall"].to_numpy()
        p_i = c_i["precision"].to_numpy()
        # Curve recall descends from 1 to 0; sort ascending for np.interp.
        order = np.argsort(r_i)
        precisions.append(np.interp(grid, r_i[order], p_i[order]))
        per_class_ap.append(float(average_precision(yt_i, ys_i)))

    weights = _avg_weights(y_bin, average, y_bin.shape[1])
    precision_avg = (np.array(precisions).T * weights).sum(axis=1)
    ap = float(np.average(per_class_ap, weights=weights))
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


def pr_frame(
    y_true: np.ndarray,
    y_score_matrix: np.ndarray,
    class_values: list,
    class_labels: list[str],
    *,
    average: str | None,
) -> pl.DataFrame:
    """Assemble the PR frame for binary or multiclass inputs.

    ``class_values`` and ``class_labels`` follow the same contract as
    :func:`roc_frame`. Binary (``len(class_labels) == 2``) draws a single
    curve on the positive (second) class. Multiclass with
    ``average in {micro, macro, weighted}`` returns a single summary curve;
    otherwise one-vs-rest curves per class.
    """
    if len(class_labels) == 2:
        return _pr_one(y_true, y_score_matrix[:, 1], class_labels[1])

    y_bin = one_hot(y_true, class_values)
    if average in ("micro", "macro", "weighted"):
        return _pr_average(y_bin, y_score_matrix, average)
    per_class = [
        _pr_one(y_bin[:, i], y_score_matrix[:, i], label) for i, label in enumerate(class_labels)
    ]
    return pl.concat(per_class, how="vertical")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_frame(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
    strategy: str,
) -> pl.DataFrame:
    """Reliability-diagram bins via ``calibration_kernel`` (empty bins dropped)."""
    rb = calibration_kernel(
        _f64(y_true),
        _f64(y_prob),
        n_bins,
        strategy,
    )
    return cast("pl.DataFrame", pl.from_arrow(rb))


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def confusion_frame(
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    labels: list,
    *,
    normalize: str | None,
) -> pl.DataFrame:
    """Long-form confusion matrix via ``confusion_kernel``.

    Owns the integer encode -> kernel -> decode round-trip plus ``value_fmt``
    so arbitrary string/categorical labels work where their dtype is known.
    Values not in ``labels`` (including NaN) map to code ``-1``, which the
    kernel drops.
    """
    label_strs = [str(lbl) for lbl in labels]
    label_to_code = {lbl: i for i, lbl in enumerate(labels)}
    yt_codes = np.array([label_to_code.get(v, -1) for v in y_true_labels.tolist()], dtype=np.int64)
    yp_codes = np.array([label_to_code.get(v, -1) for v in y_pred_labels.tolist()], dtype=np.int64)
    label_codes = np.arange(len(labels), dtype=np.int64)

    norm_arg = "" if normalize is None else normalize
    rb = confusion_kernel(
        pa.array(yt_codes, type=pa.int64()),
        pa.array(yp_codes, type=pa.int64()),
        pa.array(label_codes, type=pa.int64()),
        norm_arg,
    )
    cm_df = cast("pl.DataFrame", pl.from_arrow(rb))

    actual_col = [label_strs[int(r)] for r in cm_df["row"].to_list()]
    predicted_col = [label_strs[int(c)] for c in cm_df["col"].to_list()]
    values = cm_df["value"].to_list()
    value_fmt = [f"{v:.2f}" if normalize is not None else f"{int(v)}" for v in values]
    return pl.DataFrame(
        {
            "actual": actual_col,
            "predicted": predicted_col,
            "value": values,
            "value_fmt": value_fmt,
        }
    )


# ---------------------------------------------------------------------------
# Cumulative gain
# ---------------------------------------------------------------------------


def gain_frame(
    y_true: np.ndarray,
    y_score_matrix: np.ndarray,
    class_values: list,
    class_labels: list[str],
) -> pl.DataFrame:
    """Cumulative-gain curve per class with a 2-row baseline diagonal.

    Parameters
    ----------
    y_true
        Raw ground-truth labels (caller's dtype).
    y_score_matrix
        ``(n_samples, n_classes)`` per-class scores. Column ``i`` aligns with
        ``class_values[i]`` / ``class_labels[i]``.
    class_values
        Per-class comparison values in ``y_true``'s dtype. Used to binarize
        ``y_true`` (``y_true == value``).
    class_labels
        Per-class string labels for the ``class`` discriminator column.

    Returns
    -------
    pl.DataFrame
        Columns ``percent_population``, ``gain``, ``class``. Per-class curves
        prepend a ``(0.0, 0.0)`` origin point. A 2-row ``class='baseline'``
        diagonal (from 0 to 1) is appended at the end.
    """
    n = len(y_true)
    frames: list[pl.DataFrame] = []
    for i, (cls_val, cls_label) in enumerate(zip(class_values, class_labels)):
        y_bin = (y_true == cls_val).astype(int)
        order = np.argsort(-y_score_matrix[:, i])
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
                    "class": [cls_label] * len(xs),
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
    return pl.concat(frames, how="vertical")


# ---------------------------------------------------------------------------
# Lift
# ---------------------------------------------------------------------------


def lift_frame(
    y_true: np.ndarray,
    y_score_matrix: np.ndarray,
    class_values: list,
    class_labels: list[str],
) -> pl.DataFrame:
    """Lift curve per class with a 2-row baseline line at lift=1.0.

    Parameters
    ----------
    y_true
        Raw ground-truth labels (caller's dtype).
    y_score_matrix
        ``(n_samples, n_classes)`` per-class scores. Column ``i`` aligns with
        ``class_values[i]`` / ``class_labels[i]``.
    class_values
        Per-class comparison values in ``y_true``'s dtype.
    class_labels
        Per-class string labels for the ``class`` discriminator column.

    Returns
    -------
    pl.DataFrame
        Columns ``percent_population``, ``lift``, ``class``. Classes whose
        base rate is zero are silently omitted. A 2-row ``class='baseline'``
        line at ``lift=1.0`` is appended at the end.
    """
    n = len(y_true)
    frames: list[pl.DataFrame] = []
    for i, (cls_val, cls_label) in enumerate(zip(class_values, class_labels)):
        y_bin = (y_true == cls_val).astype(int)
        base_rate = float(y_bin.mean()) if n else 0.0
        if base_rate == 0.0:
            continue
        order = np.argsort(-y_score_matrix[:, i])
        cum_pos = np.cumsum(y_bin[order])
        denom = np.arange(1, n + 1)
        cum_rate = cum_pos / denom
        lift = cum_rate / base_rate
        pct_pop = denom / n
        frames.append(
            pl.DataFrame(
                {
                    "percent_population": pct_pop.astype(float),
                    "lift": lift.astype(float),
                    "class": [cls_label] * n,
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
    return pl.concat(frames, how="vertical")


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def threshold_sweep_frame(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> pl.DataFrame:
    """Precision/recall/F1/queue-rate threshold sweep via ``prf_at_thresholds``.

    ``y_true_bin`` is the binarized (0/1) positive-class indicator; the caller
    owns positive-class resolution. The kernel returns metrics in threshold
    order; the threshold column is prepended to match the output schema.
    """
    thresholds = thresholds.astype(np.float64)
    rb = prf_at_thresholds(
        _f64(y_true_bin),
        _f64(y_score),
        pa.array(thresholds, type=pa.float64()),
    )
    metrics = cast("pl.DataFrame", pl.from_arrow(rb))
    return metrics.with_columns(pl.Series("threshold", thresholds)).select(
        ["threshold", "precision", "recall", "f1", "queue_rate"]
    )
