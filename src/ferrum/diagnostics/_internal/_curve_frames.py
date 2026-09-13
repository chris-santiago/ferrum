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

``classification_report_frame`` is the one non-kernel builder here: its
callers already reduce their inputs to a per-class metrics dict (sklearn's
``output_dict`` on the model path, confusion-count ratios on the precomputed
path) before calling it, so it owns only the shared long-form row shaping,
not a Rust kernel call.

This module also owns *target hygiene* (#203): the one contract every
binary/multiclass diagnostic applies to a target that disagrees with the
model — nulls refuse, a resolution nothing matches refuses, and rows outside
an otherwise-matching class set absorb as negative and say so. See the
"Target hygiene" section below; the guards are called by both source
adapters, so neither path can answer the question differently.

No scikit-learn import lives here — the kernels are sklearn-free, and
``classification_report_frame`` never imports it either.

Python-facing DataFrame schemas (the chart-builder contract, unchanged):

- ROC: ``fpr, tpr, threshold, class, auc``
- PR: ``precision, recall, threshold, class, ap``
- Calibration: ``mean_predicted, fraction_positive, count, brier``
- Confusion: ``actual, predicted, value, value_fmt``
- Classification report: ``class, metric, value, value_fmt``
- Discrimination threshold: ``threshold, precision, recall, f1, queue_rate``
- Cumulative gain: ``percent_population, gain, class``
- Lift: ``percent_population, lift, class``
"""

from __future__ import annotations

import math
import warnings
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


def brier_score(y_prob: np.ndarray, y_true_bin: np.ndarray) -> float:
    """Per-sample Brier score: ``mean((y_prob - y_true_bin) ** 2)``.

    Lives here, beside the kernels, because it is curve arithmetic rather
    than a display concern: :func:`calibration_frame` broadcasts it onto
    its rows the way the Rust kernels broadcast ``auc``/``ap``.
    ``ferrum._metric_labels`` imports it from here for its metric-kind
    table, so there is exactly one Brier formula and the dependency runs
    display -> computation rather than the other way round.
    """
    p = np.asarray(y_prob, dtype=np.float64)
    obs = np.asarray(y_true_bin, dtype=np.float64)
    return float(np.mean((p - obs) ** 2))


def positive_indicator(y_true: np.ndarray, class_value: object) -> np.ndarray:
    """0/1 indicator column for ``y_true == class_value``.

    Every Rust kernel in this module counts positives by *summing* its
    ``y_true`` argument, so it requires an indicator and cannot be handed a
    raw target. That makes binarization mandatory on every path, not a
    convenience: a ``{-1, 1}`` target summed as an indicator yields negative
    "true positive" counts, a ``{1, 2}`` target double-counts, and a string
    target cannot be cast to float at all. This function is the one
    expression of that binarization, shared by :func:`one_hot` (multiclass
    one-vs-rest), :func:`gain_frame`/:func:`lift_frame`, and -- via
    :func:`binary_indicator` -- every binary curve builder.

    Values outside the class set (including nulls, which arrive as NaN or
    ``None`` and compare unequal to everything) land on 0, the same slot as
    a genuine negative. That is correct for the one-vs-rest callers, where
    "not this class" is exactly the question. It is *not* safe for a binary
    curve, where it would let a corrupted target masquerade as a clean one
    -- which is why binary callers go through :func:`binary_indicator`
    instead of calling this directly.
    """
    return (np.asarray(y_true) == class_value).astype(int)


def _observed_values(y_true: np.ndarray, limit: int = 8) -> list:
    """Distinct values of ``y_true`` for an error message, capped and sorted.

    Sorted by ``repr`` rather than by value: a target this is called on is
    by definition suspect, so it may hold mixed or unorderable types.
    """
    distinct = {repr(v): v for v in np.asarray(y_true).tolist()}
    ordered = [distinct[key] for key in sorted(distinct)]
    return ordered[:limit]


# ---------------------------------------------------------------------------
# Target hygiene (#203)
# ---------------------------------------------------------------------------
#
# One contract for a target that disagrees with the model, applied by every
# binary and multiclass diagnostic on both source paths:
#
#   - a null-bearing target refuses  (:func:`require_no_null_target`)
#   - a resolution no row matches refuses  (:func:`binary_indicator` for a
#     binary pair's positive class, :func:`check_resolved_target` for the
#     multiclass reach)
#   - rows outside an otherwise-matching class set absorb as negative and
#     say so  (:func:`check_resolved_target`), except in the confusion
#     family, where the kernel drops them instead and the warning says so
#     (:func:`check_encodable_target`)
#
# Before this, the answer depended on the target's dtype and on which
# resolution tier the model happened to land in: a string target with nulls
# crashed with an unowned ``TypeError: '<' not supported`` from whichever of
# four ``np.unique``/``sorted`` sites it reached first, a numeric one silently
# gained a NaN class that could shift the resolution tier, stray values
# absorbed silently on tier 1, and a multiclass tier-3 resolution matching
# nothing returned all-NaN curves with no diagnosis at all.


def _null_mask(y_true: np.ndarray) -> np.ndarray:
    """Boolean mask of the null slots in a target array.

    A polars target reaches numpy in one of three shapes and its nulls look
    different in each: an ``Int64``/``Float64`` column with nulls arrives as
    float64 carrying ``NaN``; ``Date``/``Datetime``/``Duration`` arrive as
    ``datetime64``/``timedelta64`` carrying ``NaT``; and ``Utf8``/
    ``Categorical``/``Enum``/``Boolean`` arrive as an object array carrying
    ``None``. All three say the same thing -- this row's class was not
    recorded -- so all three count. Any other dtype (plain integer, plain
    unicode, plain bool) cannot carry a null in numpy and masks empty,
    which is the common case and costs one dtype check.

    One null spelling is *not* detected here, deliberately: a pandas
    nullable target (``pd.Series([...], dtype="string")``) reaches numpy as
    an object array whose null is ``pd.NA``, which is neither ``None`` nor
    a float ``NaN``. Such a target is refused earlier and less precisely,
    at ferrum's polars ingestion seam (measured 2026-09-13:
    ``TypeError: unexpected value while building Series of type String;
    found value of type Object: <NA>``), so it never reaches a diagnostic
    -- it is an ingestion-layer gap, not a silent accept.
    """
    arr = np.asarray(y_true)
    if arr.dtype.kind in "fc":
        return np.isnan(arr)
    if arr.dtype.kind in "Mm":
        return np.isnat(arr)
    if arr.dtype.kind == "O":
        return np.array(
            [v is None or (isinstance(v, float) and math.isnan(v)) for v in arr.tolist()],
            dtype=bool,
        )
    return np.zeros(len(arr), dtype=bool)


def require_no_null_target(y_true: np.ndarray, *, diagnostic: str) -> None:
    """Refuse a target that carries null rows.

    A null is not a class observation. Binarizing it lands it on 0 (nulls
    compare unequal to everything), which counts it as a negative the data
    never recorded -- fabrication rather than degradation, so this refuses
    where the stray-value case merely warns (see
    :func:`check_resolved_target`). It is also what scikit-learn does with
    a NaN-bearing ``y``.

    This guard runs *before* class resolution, which makes it the owner of
    the four sites a null used to reach: :func:`binary_class_values`'
    ``np.unique``, both cardinality guards in
    ``sources/_classification._resolve_class_values``, and
    ``ModelSource.confusion_matrix``'s no-``classes_`` ``sorted(set(...))``.
    On a string-backed target each of those raised
    ``TypeError: '<' not supported between instances of 'str' and 'NoneType'``
    naming neither the target nor the diagnostic; on a numeric one none of
    them raised at all -- the target silently gained a distinct ``NaN``
    class, which could push a ``classes_``-less model from resolution tier 2
    to tier 3 and change the answer from a number to a refusal.

    ``diagnostic`` names the public entry the caller actually invoked --
    the source method on the model path (``roc_curve``), the figure
    function on the precomputed path (``roc_chart``) -- following
    :func:`binary_indicator`'s ``diagnostic=`` idiom.
    """
    mask = _null_mask(y_true)
    n_null = int(mask.sum())
    if not n_null:
        return
    raise ValueError(
        f"{diagnostic}: y has {n_null} null row(s) out of {len(mask)}, and a "
        f"null is not a class observation -- counting those rows as negatives "
        f"would measure every metric against data the target never recorded. "
        f"y holds {_observed_values(y_true)!r}. Drop or impute the null rows "
        f"before calling {diagnostic}."
    )


def _unmatched_mask(y_true: np.ndarray, class_values: list) -> np.ndarray:
    """Boolean mask of the rows equalling none of ``class_values``.

    Membership is tested with :func:`positive_indicator`, the same
    comparison the binarization and the confusion encoding perform, so what
    the guards below report and what the kernels see cannot disagree.
    """
    matched = np.zeros(len(y_true), dtype=bool)
    for value in class_values:
        matched |= positive_indicator(y_true, value).astype(bool)
    return ~matched


def check_resolved_target(
    y_true: np.ndarray,
    class_values: list,
    *,
    diagnostic: str,
    stacklevel: int,
) -> None:
    """Diagnose a resolved class set against the target it will binarize.

    Runs once per diagnostic call, after the caller has resolved its
    per-class comparison values and before any frame is built. Membership is
    tested with :func:`positive_indicator`, the same comparison the
    binarization itself uses, so what this reports and what the kernels see
    cannot disagree.

    Two outcomes:

    **A binary pair whose positive class matches no row.** This says
    nothing at all: :func:`binary_indicator` is about to refuse in the
    positive class's own voice, and warning first would assert an absorb
    the refused call never performs (measured 2026-09-13, ``classes_ =
    [0, 2]`` against a ``{0, 1}`` target: "96 row(s) ... were counted as
    negative" immediately followed by the refusal, so nothing was counted
    as anything). The deferral covers the total mismatch and the partial
    one alike, since the positive class is the only class a binary curve
    binarizes against.

    **No class value matches any row, with three or more classes.** Every
    one-vs-rest curve would be computed against an entirely-negative
    phantom target. There was no voice for this case, and the call
    returned all-NaN curves in silence (measured 2026-09-13 on the
    multiclass fixture: three NaN curves where sklearn's one-vs-rest AUCs
    over the same probabilities were 0.872 / 0.942 / 0.930). That reach
    refuses here.

    **Some rows fall outside an otherwise-matching class set.** Those rows
    absorb as negative, which is scikit-learn's own ``pos_label`` semantics
    and the right arithmetic -- "not this class" is exactly the one-vs-rest
    question -- so the answer is kept and a ``UserWarning`` names the
    diagnostic, how many rows absorbed, and which values they held. The
    silence was the defect, not the absorb: on a 200-row binary fixture,
    replacing 10 targets with a stray third value moved the AUC from 0.9843
    to 0.9395 with nothing on the chart to say so.

    The confusion-matrix family does not absorb its strays -- it drops
    them -- so it has its own guard, :func:`check_encodable_target`.
    """
    y_true = np.asarray(y_true)
    if not len(y_true):
        return

    unmatched = _unmatched_mask(y_true, class_values)
    n_absorbed = int(unmatched.sum())

    if len(class_values) < 3:
        # The binary pair's positive class is its last value (ascending class
        # order, matching predict_proba's columns). If no row holds it,
        # binary_indicator refuses moments later and this must stay silent.
        if not class_values or not positive_indicator(y_true, class_values[-1]).any():
            return
    elif n_absorbed == len(y_true):
        raise ValueError(
            f"{diagnostic}: no row of y equals any of the resolved class "
            f"values {class_values!r}, so every one-vs-rest curve would be "
            f"computed against an entirely-negative target and would be "
            f"meaningless. y holds {_observed_values(y_true)!r}. Check that y "
            f"and the model's classes_ describe the same labels."
        )
    if not n_absorbed:
        return

    strays = _observed_values(np.asarray(y_true[unmatched], dtype=object))
    warnings.warn(
        f"{diagnostic}: {n_absorbed} row(s) of y hold values outside the "
        f"resolved class set {class_values!r} and were counted as negative; "
        f"the out-of-domain values are {strays!r}.",
        UserWarning,
        stacklevel=stacklevel,
    )


def check_encodable_target(
    y_true: np.ndarray,
    encode_values: list,
    *,
    diagnostic: str,
    stacklevel: int,
) -> None:
    """Diagnose a confusion encoding against the target it will encode.

    The confusion family's stray cell, and it is not the curve family's.
    :func:`confusion_frame` encodes every value to its index in
    ``encode_values`` and maps everything else to ``-1``, which the kernel
    **drops**: an out-of-domain row is counted in no cell at all, rather
    than absorbed into a negative the way a curve's binarization absorbs it
    (see :func:`check_resolved_target`). The matrix silently describes
    fewer rows than the caller passed -- measured 2026-09-13 on the shared
    fixtures, with ten rows set to a stray value: cell counts summing to
    190 of 200 on the binary fixture and 290 of 300 on the multiclass one,
    with no diagnostic of any kind. ``class_prediction_error_chart``, which
    renders the same frame, lost the same rows.

    Dropping rows is a degradation the caller has to hear about, so this
    warns rather than refusing, and the message says *dropped* -- the word
    has to match what the kernel does, or the diagnosis is worse than
    silence.

    Only a caller whose ``encode_values`` can exclude a row needs this: it
    is the model path's ``classes_`` branch, where the encoding comes from
    the model rather than from ``y``. The no-``classes_`` branch and the
    precomputed twin both build their labels from the union of ``y_true``
    and ``y_pred``, so nothing can fall outside.
    """
    y_true = np.asarray(y_true)
    if not len(y_true):
        return

    unmatched = _unmatched_mask(y_true, encode_values)
    n_dropped = int(unmatched.sum())
    if not n_dropped:
        return

    strays = _observed_values(np.asarray(y_true[unmatched], dtype=object))
    warnings.warn(
        f"{diagnostic}: {n_dropped} row(s) of y out of {len(y_true)} hold "
        f"values outside the class set {encode_values!r} the matrix is "
        f"encoded against, and were dropped from it entirely -- they are "
        f"counted in no cell, not absorbed into one; the out-of-domain "
        f"values are {strays!r}.",
        UserWarning,
        stacklevel=stacklevel,
    )


def binary_indicator(y_true: np.ndarray, positive_value: object, *, diagnostic: str) -> np.ndarray:
    """:func:`positive_indicator`, refusing a target that matches nothing.

    The binary curve builders binarize against a single positive class. If
    *no* row matches it, every kernel downstream sees an all-negative
    target and returns something plausible and meaningless -- a Brier of
    ``mean(p**2)``, an AP of 0.0, a flat reliability curve -- with no
    indication that the target and the class value never met. That is the
    exact trade :func:`ferrum._metric_labels._broadcast_metric` refuses in
    the other direction, and it is how a mis-resolved positive class stays
    invisible (2026-09-10 close review, cycle 3).

    So a total mismatch refuses by name instead: it is never a meaningful
    request, and the two things a caller needs in order to fix it are the
    class value that was searched for and the values ``y`` actually holds.
    A null-bearing target -- an all-null one included -- is a mismatch too
    (nulls compare unequal to everything), but it is diagnosed earlier and
    more precisely by :func:`require_no_null_target`, which every
    diagnostic runs before resolving its classes (#203); this guard is what
    catches it if a caller reaches here without that check.

    An empty target is *not* an error -- there is nothing to mismatch --
    and a target whose rows are all positive is left alone too: the kernels
    report that honestly as an undefined (NaN) metric.
    """
    indicator = positive_indicator(y_true, positive_value)
    if len(indicator) and not indicator.any():
        raise ValueError(
            f"{diagnostic}: no row of y equals the positive class "
            f"{positive_value!r}, so the target is entirely negative and "
            f"every metric computed from it would be meaningless. y holds "
            f"{_observed_values(y_true)!r}. Check that y and the model's "
            f"classes_ describe the same labels."
        )
    return indicator


def binary_class_values(y_true: np.ndarray) -> tuple[list, list[str]]:
    """``(class_values, class_labels)`` read off a binary target itself.

    The fallback both source adapters use when the model offers no
    ``classes_`` to align against: the two classes are ``y_true``'s own
    distinct values, ascending, with the larger taken as positive. That
    matches what every score matrix in this subsystem assumes (column
    ``i`` scores the ``i``-th class in ascending class order, which is how
    scikit-learn orders ``predict_proba``) and it matches
    ``discrimination_threshold``'s long-standing ``np.unique(y)[-1]`` rule.

    Values come back in ``y_true``'s own dtype -- they are comparison
    values, used as ``y_true == class_values[1]`` -- and labels are their
    string forms, used only for a ``class`` discriminator column. Deriving
    them from ``y`` rather than from a display string is the whole point:
    a proba column's name may be a user-supplied alias or a synthetic
    ``class_{i}``, neither of which is a value ``y`` contains.

    A degenerate target with fewer than two distinct values has no second
    class, so both slots describe the one class present (values *and*
    labels agree, so the record never says two different things); an empty
    target falls back to ``0``/``1``.
    """
    unique = np.unique(np.asarray(y_true)).tolist()
    if len(unique) >= 2:
        return [unique[0], unique[-1]], [str(unique[0]), str(unique[-1])]
    if len(unique) == 1:
        return [unique[0], unique[0]], [str(unique[0]), str(unique[0])]
    return [0, 1], ["0", "1"]


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
        out[:, j] = positive_indicator(y_true, cls)
    return out


def _f64(arr: np.ndarray) -> pa.Array:
    """Build a float64 Arrow array from a numpy array."""
    return pa.array(np.asarray(arr).astype(np.float64), type=pa.float64())


def _descending_score_order(scores: np.ndarray) -> np.ndarray:
    """Row order for a "best-scored first" sweep, ties in original row order.

    Shared by the cumulative-gain and lift frames, which both walk the
    population from the highest-scoring sample down. The sort is stable, so
    samples sharing a score (routine for tree ensembles and any model with
    coarse probability output) keep their input order instead of whichever
    permutation numpy's quicksort happens to produce -- the curve is then a
    function of the data alone, not of the sort's internal state.
    """
    return np.argsort(-scores, kind="stable")


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
            binary_indicator(y_true, class_values[1], diagnostic="roc_curve"),
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
        # Stable so tied recall values (routine at the curve extremes) keep
        # their threshold order and the interpolated curve is the same frame
        # on every run.
        order = np.argsort(r_i, kind="stable")
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
        return _pr_one(
            binary_indicator(y_true, class_values[1], diagnostic="pr_curve"),
            y_score_matrix[:, 1],
            class_labels[1],
        )

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
    positive_value: object,
    n_bins: int,
    strategy: str,
) -> pl.DataFrame:
    """Reliability-diagram bins via ``calibration_kernel``, plus ``brier``.

    Parameters
    ----------
    y_true
        Raw ground-truth labels (caller's dtype), binarized here against
        ``positive_value`` -- the kernel averages ``y_true`` per bin to get
        ``fraction_positive``, so it needs an indicator, not a raw target.
    y_prob
        Positive-class probability per sample.
    positive_value
        The value in ``y_true``'s domain that means "positive". The caller
        owns resolving it (``model.classes_[1]`` on the model path, the
        larger of the two observed labels on the precomputed path), the
        same contract :func:`roc_frame`'s ``class_values`` carries.
    n_bins, strategy
        Forwarded to ``calibration_kernel``. Empty bins are dropped.

    Returns
    -------
    pl.DataFrame
        ``mean_predicted, fraction_positive, count, brier``. ``brier`` is
        the **per-sample** Brier score of ``(y_prob, indicator)`` broadcast
        onto every bin row, the same way ``auc``/``ap`` broadcast onto ROC
        and PR rows. It is computed here, at frame assembly, because this
        is the last place the per-sample arrays exist: once the frame is
        binned, a Brier recomputed from ``(mean_predicted,
        fraction_positive)`` is a bin-level statistic that swings with
        ``n_bins=``/``strategy=`` and is not the Brier score. Carrying it
        on the frame is what lets every display surface -- the chart title,
        the figure builder's overlay, and the composable
        ``chart + BrierLabel()`` -- read one number from one source.
    """
    y_bin = binary_indicator(y_true, positive_value, diagnostic="calibration_curve")
    rb = calibration_kernel(
        _f64(y_bin),
        _f64(y_prob),
        n_bins,
        strategy,
    )
    bins = cast("pl.DataFrame", pl.from_arrow(rb))
    # An empty target has no Brier score; NaN says so, where the mean of an
    # empty array would say the same thing plus a numpy RuntimeWarning.
    brier = brier_score(y_prob, y_bin) if len(y_bin) else float("nan")
    return bins.with_columns(pl.lit(brier).alias("brier"))


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
    kernel drops -- those rows are counted in no cell. A caller whose
    ``labels`` can exclude a row of ``y_true`` runs
    :func:`check_encodable_target` first so the drop is warned about rather
    than silent (#203).
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
# Classification report
# ---------------------------------------------------------------------------


def classification_report_frame(per_class_metrics: dict[str, dict[str, float]]) -> pl.DataFrame:
    """Long-form per-class precision/recall/F1: one row per (class, metric).

    ``per_class_metrics`` maps each class label (string) to a dict holding
    ``"precision"``, ``"recall"``, and ``"f1-score"`` -- the shape both
    callers already produce before handing off here: the sklearn-backed
    model path (``plots/classification.py``'s ``_classification_report_chart``,
    filtered from ``sklearn.metrics.classification_report(output_dict=True)``)
    and the confusion-count-derived precomputed path
    (``_PrecomputedSource.classification_report``, no sklearn). This function
    owns only the shared shaping -- turning a metrics dict into the long-form
    rows ``classification_report_chart`` renders -- not the metric math
    itself, which differs by caller (sklearn vs. confusion-count ratios) and
    stays in each caller.
    """
    rows: list[dict] = []
    for cls_label, metrics in per_class_metrics.items():
        for m_name in ("precision", "recall", "f1-score"):
            val = float(metrics[m_name])
            rows.append(
                {
                    "class": str(cls_label),
                    "metric": m_name,
                    "value": val,
                    "value_fmt": f"{val:.2f}",
                }
            )
    return pl.DataFrame(rows)


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

    Notes
    -----
    A binary class pair binarizes its positive class through
    :func:`binary_indicator`, so a target matching no positive row refuses
    by name here exactly as it does on the ROC, PR, calibration and
    threshold builders (#203). ``binary_indicator``'s docstring has always
    claimed "every binary curve builder"; gain and lift were the two that
    were not, and a binary ``cumulative_gain`` over an all-negative target
    returned a curve reaching gain 1.0 with nothing to say it was drawn
    against a target the positive class never met. Multiclass one-vs-rest
    is untouched: "no row is class k" is a legitimate answer there.
    """
    n = len(y_true)
    binary_pair = len(class_values) == 2
    frames: list[pl.DataFrame] = []
    for i, (cls_val, cls_label) in enumerate(zip(class_values, class_labels)):
        y_bin = (
            binary_indicator(y_true, cls_val, diagnostic="cumulative_gain")
            if binary_pair and i == 1
            else positive_indicator(y_true, cls_val)
        )
        order = _descending_score_order(y_score_matrix[:, i])
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

    Notes
    -----
    Same as :func:`gain_frame`: a binary pair's positive class goes through
    :func:`binary_indicator`, so a total mismatch refuses by name instead of
    being dropped by the zero-base-rate ``continue`` below (#203). That
    ``continue`` stays for multiclass one-vs-rest, which it was written for
    and which this function documents.
    """
    n = len(y_true)
    binary_pair = len(class_values) == 2
    frames: list[pl.DataFrame] = []
    for i, (cls_val, cls_label) in enumerate(zip(class_values, class_labels)):
        y_bin = (
            binary_indicator(y_true, cls_val, diagnostic="lift_curve")
            if binary_pair and i == 1
            else positive_indicator(y_true, cls_val)
        )
        base_rate = float(y_bin.mean()) if n else 0.0
        if base_rate == 0.0:
            continue
        order = _descending_score_order(y_score_matrix[:, i])
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
    y_true: np.ndarray,
    y_score: np.ndarray,
    positive_value: object,
    thresholds: np.ndarray,
) -> pl.DataFrame:
    """Precision/recall/F1/queue-rate sweep over a **whole target**.

    ``y_true`` is the raw target and ``positive_value`` the class that means
    "positive", the same ``(raw target, resolved comparison value)`` contract
    :func:`roc_frame` and :func:`calibration_frame` take -- binarization
    happens here, through :func:`binary_indicator`, not in the caller.

    It used to be the caller's job, which is structurally the same shape as
    the pre-fix ``roc_frame``: a builder that accepts a resolved class value
    and trusts someone upstream to have applied it. Three call sites each
    wrote the same ``(y == positive).astype(int)`` line, and none of them got
    the mismatch refusal.

    This entry is for a whole target, where an all-negative result means the
    caller's ``y`` and the model's classes never met, and refusing is right.
    A caller holding an already-resolved *slice* of a validated target -- the
    cross-validation fold loop is the only one -- wants
    :func:`threshold_sweep_from_indicator` instead, because a degenerate
    slice of a good target is not a bad request (2026-09-11 close review,
    cycle 4).
    """
    y_true_bin = binary_indicator(y_true, positive_value, diagnostic="discrimination_threshold")
    return threshold_sweep_from_indicator(y_true_bin, y_score, thresholds)


def threshold_sweep_from_indicator(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> pl.DataFrame:
    """Sweep an already-binarized target, with no mismatch refusal.

    The sibling of :func:`threshold_sweep_frame` for callers that have
    already resolved and validated the positive class at *request* level and
    are now feeding it rows -- i.e. a CV fold. An all-negative fold is a
    property of the split, not of the request: the kernel reports
    ``precision = recall = f1 = 0`` at every threshold with a real
    ``queue_rate``, which is the honest answer for a slice containing
    nothing to find, and which is exactly what the fold average has always
    incorporated.

    Keeping the refusal out of this entry is the whole point: a guard that
    sees an array cannot tell a user's broken target from an internal slice
    of a good one, so the guard belongs to the entry that sees the request.
    The kernel returns metrics in threshold order; the threshold column is
    prepended to match the output schema.
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
