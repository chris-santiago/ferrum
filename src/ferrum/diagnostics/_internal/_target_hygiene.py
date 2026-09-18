"""Target hygiene and class-axis vocabulary for classification diagnostics (#203, #224a).

The one contract every binary and multiclass diagnostic applies to a target
that disagrees with the model, and the one derivation of the class axis a
label-domain diagnostic puts its values on. Both source adapters --
``_PrecomputedSource`` (array inputs) and ``ClassificationCurvesMixin``
(model inputs) -- call these directly, so neither path can answer the same
malformed input differently.

Split out of ``_curve_frames`` (2026-09-17, whole-wave design review findings
6 and 14) because these are not frame assembly: the guards are called by the
*adapters*, before any builder runs, and the vocabulary derivation touches no
kernel and no frame. They lived there because both adapters needed to share
them and ``_curve_frames`` was where the sharing already happened; at 1398
lines that module was carrying three separable responsibilities and the
adapters imported it for the guards more often than for the frames.

The dependency runs one way: ``_curve_frames`` imports :func:`binary_indicator`
(its builders binarize through it) and :func:`positive_indicator` (``one_hot``
is built from it) from here, and nothing here imports ``_curve_frames``.
:func:`positive_indicator` lives on this side for that reason -- it is the
primitive the membership guards and the binarization share, so putting it with
the builders would make the two modules import each other.

No scikit-learn and no Rust kernel are involved: every refusal and warning
here is decided from the caller's own arrays.
"""

from __future__ import annotations

import datetime
import math
import numbers
import warnings
from typing import Literal

import numpy as np


def positive_indicator(y_true: np.ndarray, class_value: object) -> np.ndarray:
    """0/1 indicator column for ``y_true == class_value``.

    Every Rust kernel in ``_curve_frames`` counts positives by *summing*
    its ``y_true`` argument, so it requires an indicator and cannot be
    handed a raw target. That makes binarization mandatory on every path, not a
    convenience: a ``{-1, 1}`` target summed as an indicator yields negative
    "true positive" counts, a ``{1, 2}`` target double-counts, and a string
    target cannot be cast to float at all. This function is the one
    expression of that binarization, shared by ``_curve_frames``'
    ``one_hot`` (multiclass one-vs-rest) and its ``gain_frame``/
    ``lift_frame``, and -- via :func:`binary_indicator` -- every binary
    curve builder.

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
#   - a null-bearing hard-label prediction column refuses
#     (:func:`require_no_null_predictions`)
#   - a translated class vocabulary sharing nothing with the target refuses
#     (:func:`check_translated_classes`)
#   - rows outside an otherwise-matching class set absorb as negative and
#     say so  (:func:`check_resolved_target`), except in the confusion
#     family, where the kernel drops them instead and the warning says so
#     (:func:`check_encodable_target`)
#
# "Every diagnostic on both paths" is meant literally, and was not when the
# guards shipped: the confusion family sat outside the null-prediction and
# total-mismatch cells until 2026-09-17, so ``confusion_matrix_chart``
# fabricated a ``'None'`` class and rendered an all-zero matrix where
# ``classification_report_chart`` refused on the identical arrays.
#
# The label *rule* was the last piece, closed the same day. Both confusion
# paths now derive their class axis from :func:`report_labels`, the same
# partition the report uses, so all four label-domain surfaces answer one
# input one way. The rule it replaced on the precomputed path,
# ``sorted(set(...), key=str)``, was carried for one more cycle as an
# ordering preference and was not one: stringifying before comparing made it
# *accept* inputs it could not draw. Measured on ``y_true=[0, 1, 0, 1]``
# against ``y_pred=["0", "1", "0", "1"]``, it derived four labels that
# stringify to two and produced a frame with duplicate ``(actual,
# predicted)`` keys, which the chart rendered as superimposed text at one
# pixel position. Refusing is the answer the report always gave.
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


#: Who produced the ``y_pred`` a label-domain guard is refusing. The two
#: label-domain source paths hand the same guards two different arrays:
#: ``_PrecomputedSource`` passes the caller's own ``y_pred=``, while the
#: model path passes ``model.predict(X)``'s output. Only the corrective
#: clause of a refusal depends on which — the diagnosis, the counts and the
#: observed values are identical — so this is threaded no further than the
#: message (cycle-2 review finding H: a refusal that tells a caller to
#: "drop the null rows" from an array the model produced names a lever they
#: do not hold).
PredictionsFrom = Literal["caller", "model"]

#: Closing clause for :func:`require_no_null_predictions`, per owner.
_NULL_PREDICTION_CORRECTION: dict[str, str] = {
    # Byte-identical to the sentence this message always closed with: only
    # the model arm is new, so no pinned precomputed-path message moves.
    "caller": "Drop or impute the null rows before calling {diagnostic}.",
    "model": (
        "These predictions are the model's own predict() output, not an array "
        "you passed, so the fix is upstream of it: drop the rows the model "
        "predicts a null for, or use an estimator that predicts a class for "
        "every row, before calling {diagnostic}."
    ),
}

#: Closing clause for :func:`report_labels`' mix-of-families refusal, per owner.
_MIXED_LABEL_CORRECTION: dict[str, str] = {
    "caller": ("Use one consistent value type for y_true and y_pred before calling {diagnostic}."),
    "model": (
        "y_pred is the model's own predict() output, so the disagreement is "
        "between the model's labels and y's: fit the model on labels of y's "
        "type, or pass a y in the model's, before calling {diagnostic}."
    ),
}

#: Closing clause for :func:`_unkeyable_label_error`'s cell 4, per owner.
_UNKEYABLE_LABEL_CORRECTION: dict[str, str] = {
    "caller": (
        "Use label values that compare with < and can be hashed (numbers, "
        "strings, dates) before calling {diagnostic}."
    ),
    "model": (
        "y_pred is the model's own predict() output, so at least one side of "
        "this is the model's: fit it on label values that compare with < and "
        "can be hashed (numbers, strings, dates), and pass a y of the same "
        "kind, before calling {diagnostic}."
    ),
}


def _refuse_null_rows(
    values: np.ndarray, *, name: str, consequence: str, correction: str, diagnostic: str
) -> None:
    """Raise if ``values`` carries null rows, in the shared null vocabulary.

    One message skeleton for every array a diagnostic refuses nulls in:
    which array, how many rows of how many, *why* a null cannot be counted
    there, the observed values, and the correction. Two clauses vary, and
    both have to. ``consequence`` differs by array: what a null does to a
    target (:func:`require_no_null_target`) and what it does to a prediction
    column (:func:`require_no_null_predictions`) are different fabrications,
    and a message that blurred them would send the caller after the wrong
    array. ``correction`` differs by *who owns the array*: "drop or impute
    the null rows" is actionable for a caller who passed ``y_pred=``, and
    names a lever the caller does not hold when ``y_pred`` is
    ``model.predict``'s own output (cycle-2 review finding H).
    """
    mask = _null_mask(values)
    n_null = int(mask.sum())
    if not n_null:
        return
    raise ValueError(
        f"{diagnostic}: {name} has {n_null} null row(s) out of {len(mask)}, and "
        f"{consequence} {name} holds {_observed_values(values)!r}. {correction}"
    )


def _unkeyable_label_error(
    *,
    family: str,
    reason: str,
    correction: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    diagnostic: str,
) -> ValueError:
    """Build :func:`report_labels`' cell-4 refusal; ``reason`` and ``correction`` vary.

    Cell 4's two ways into "unusable as a class label" -- does not order,
    orders but cannot be hashed -- differ only in the sentence naming which
    of the two failed and why; the stem and the observed-values pair are
    identical between them. This is the same shape
    :func:`_refuse_null_rows`'s ``consequence=`` parameter exists for,
    applied to cell 4 rather than the null cell, and ``correction`` varies
    for the same reason it does there: the lever a caller holds depends on
    whether ``y_pred`` is theirs or ``model.predict``'s.
    """
    return ValueError(
        f"{diagnostic}: y_true and y_pred hold {family} values that cannot "
        f"be used as class labels. {reason} y_true holds "
        f"{_observed_values(y_true)!r} and y_pred holds "
        f"{_observed_values(y_pred)!r}. {correction}"
    )


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
    _refuse_null_rows(
        y_true,
        name="y",
        consequence=(
            "a null is not a class observation -- counting those rows as "
            "negatives would measure every metric against data the target "
            "never recorded."
        ),
        # `y` is the caller's on every path -- the model never produces it --
        # so this correction has no second form (contrast
        # :func:`require_no_null_predictions`).
        correction=f"Drop or impute the null rows before calling {diagnostic}.",
        diagnostic=diagnostic,
    )


def require_no_null_predictions(
    y_pred: np.ndarray, *, diagnostic: str, predictions_from: PredictionsFrom = "caller"
) -> None:
    """Refuse a hard-label prediction column that carries null rows.

    :func:`require_no_null_target`'s sibling for the other array a
    label-domain diagnostic reads. A null prediction is not a predicted
    class, and every rule that builds a class axis from the union of
    ``y_true`` and ``y_pred`` turns it into a class of its own: measured
    2026-09-13 on ``classification_report_chart``, an object-dtype
    ``y_pred = ["a", None, "a", "b"]`` rendered a ``'None'`` row at
    0/0/0 and dropped class ``'b'``'s recall to 0.5, and the float twin
    ``y_pred = [0.0, nan, 0.0, 1.0]`` rendered a ``'nan'`` row the same
    way. Both are fabrication rather than degradation -- the chart names a
    class nothing ever predicted -- so this refuses, as
    :func:`require_no_null_target` does, rather than warning.

    The confusion family calls it too, on both source paths (2026-09-17).
    It did not when this guard shipped with :func:`report_labels` (#224a,
    2026-09-13) -- widening it was a second figure's behavior change, so it
    was disclosed rather than made -- and the result was that
    ``classification_report_chart`` and ``confusion_matrix_chart`` answered
    the identical arrays with a refusal and a fabricated ``'None'`` class
    respectively. Measured 2026-09-13 on the precomputed path with
    ``y_true=["a","b","a","b"], y_pred=["a",None,"a","b"]``: the confusion
    matrix's ``predicted`` uniques were ``['None', 'a', 'b']``. Two figures
    over one input is exactly the dtype/tier-dependent divergence #203 set
    out to end, so the hole is closed and the move disclosed as a
    ``**Breaking:**`` changelog bullet instead.

    ``predictions_from`` names who produced ``y_pred`` and changes only the
    corrective clause (see :data:`PredictionsFrom`): the caller can drop
    null rows from an array they passed, but not from ``model.predict``'s
    output.
    """
    _refuse_null_rows(
        y_pred,
        name="y_pred",
        consequence=(
            "a null is not a predicted class -- it would become a class of "
            "its own on the class axis, and every row holding it would be "
            "scored against that phantom class."
        ),
        correction=_NULL_PREDICTION_CORRECTION[predictions_from].format(diagnostic=diagnostic),
        diagnostic=diagnostic,
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


def check_translated_classes(
    y_true: np.ndarray,
    class_values: list,
    classes_: object,
    *,
    diagnostic: str,
) -> None:
    """Refuse a class vocabulary that, translated into ``y``'s domain, shares no value with it.

    Named for its subject like its two siblings (``check_resolved_target``,
    ``check_encodable_target``): what it checks is the *class vocabulary*
    ``classes_`` becomes once coerced into ``y``'s dtype. ``classes_``
    itself is a third positional argument used only to quote the untranslated
    values back in the message -- the check never reads it.

    Both label-domain families call it (the report family since #206 cycle 5,
    the confusion family since 2026-09-17).
    :func:`sources._classification.report_predictions_in_y_domain` and
    ``ModelSource.confusion_matrix`` translate ``model.predict``'s output
    into ``y``'s domain through the positional ``classes_`` <->
    ``encode_values`` correspondence -- sound when ``classes_`` and ``y``
    describe the *same* labels in two dtypes (an int-coded ``classes_``
    against a readable ``Utf8``/``Categorical``/``Enum`` ``y`` is the
    ordinary shape). It is not sound when ``classes_`` and ``y`` describe
    *different* labels: a model fit with ``classes_ = [0, 1, 2]`` against
    ``y = ["c0", "c1", "c2"]`` translates every prediction to
    ``"0"``/``"1"``/``"2"`` -- values the correspondence invented, not values
    ``y`` ever held. Each family then failed its own way, and neither said
    why: every cell of the report read 0.0 (measured 2026-09-13: an 18-row,
    6-class report, every metric 0.0, no warning), while the confusion
    matrix encoded every row out of vocabulary and rendered all-zero cells
    behind a warning that 100% of the rows had been dropped. Before #206's
    translation, this exact shape raised scikit-learn's own ``ValueError:
    Mix of label input types (string and number)`` -- loud, but at least
    stopping the caller. This is the total-mismatch shape #203's standing
    policy already refuses elsewhere (:func:`check_resolved_target`'s
    zero-match case, :func:`binary_indicator`'s positive-class case), so it
    refuses here too rather than trading a loud error for a silent,
    all-zero one.

    The subject is the class *vocabulary* -- ``class_values`` is
    ``classes_`` coerced into ``y``'s dtype -- not the predictions the
    correspondence produced. Predictions are a *sample* of that vocabulary,
    so testing them answers a different question than the message asks:
    a correctly-labelled, same-dtype model evaluated on the slice it gets
    entirely wrong (``y`` holds only ``0``, the model predicts only ``1``)
    shares no predicted value with ``y`` while its ``classes_`` and ``y``
    describe the same labels -- a legitimate report request ("how does the
    model do on the rows it misses?"), which a prediction-based test refused
    while telling the caller something measurably false (cycle-5 review,
    finding 17, 2026-09-13). Membership is tested through
    :func:`_unmatched_mask`, the same primitive
    :func:`check_resolved_target`'s zero-match cell uses, so this guard and
    the precedent whose closing sentence it borrows also share a mechanism.

    A *partial* overlap -- part of the translated vocabulary lands in
    ``y``'s domain, part doesn't -- is not this shape: it is the ordinary
    out-of-vocabulary case scikit-learn's own ``classification_report``
    already tolerates (an unmatched predicted class reports 0.0 for that
    row's contribution, same as it always has), so this only fires when the
    overlap is empty and ``y`` is non-empty.
    """
    y_true = np.asarray(y_true)
    if not len(y_true) or not len(class_values):
        return
    if not _unmatched_mask(y_true, list(class_values)).all():
        return
    raise ValueError(
        f"{diagnostic}: the model's classes_ {np.asarray(classes_).tolist()!r}, "
        f"translated into y's domain, match none of y's observed values, so "
        f"every cell would be computed against a class axis that shares "
        f"nothing with y. y holds {_observed_values(y_true)!r}. "
        f"Check that y and the model's classes_ describe the same labels."
    )


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


# ---------------------------------------------------------------------------
# Class-axis vocabulary (#224a)
# ---------------------------------------------------------------------------
#
# What the class axis of a label-domain diagnostic is allowed to hold, and in
# what order. Derived from the target and the predictions themselves, never
# from a kernel or a frame -- which is why it sits beside the guards rather
# than beside the builders.


def _value_family(value: object) -> str:
    """Name the ordering family ``value`` belongs to.

    Two values can be ordered against each other only when they are the
    same kind of thing, so "same kind" is what :func:`report_labels`
    partitions on. The families are the ones Python's ``<`` compares
    *within* and refuses *across*: every real number is one family (an
    ``int``, a ``float``, a ``bool`` and a ``Decimal`` all compare, which
    is why an int/float or bool/int union must not be called mixed), each
    of ``str`` and ``bytes`` is its own, and ``datetime`` /
    ``date`` / ``time`` / ``timedelta`` are four rather than one
    (``datetime`` refuses to compare against a bare ``date``). ``complex``
    is excluded from the number family deliberately: it is a number that
    does not order.

    Anything else is named by its own type, so two different exotic types
    read as mixed while one of them alone reads as a single family whose
    orderability :func:`report_labels` then finds out by trying.
    """
    if isinstance(value, numbers.Number) and not isinstance(value, complex):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (bytes, bytearray)):
        return "bytes"
    if isinstance(value, datetime.datetime):
        return "datetime"
    if isinstance(value, datetime.date):
        return "date"
    if isinstance(value, datetime.time):
        return "time"
    if isinstance(value, datetime.timedelta):
        return "timedelta"
    return type(value).__name__


#: The families :func:`_value_family` names whose ``<`` is a **total** order
#: over the values it admits. Membership decides only *how*
#: :func:`report_labels` reaches its answer, never what the answer is: a
#: total order sorts hash-equal values adjacently, so deduplicating before
#: sorting and sorting before deduplicating produce the same list — the
#: same values, in the same order, with the same representative of an equal
#: pair. Verified across 31 inputs at cycle-2 review, including every axis
#: that could separate the two orders.
#:
#: What membership does **not** promise is which of the two refusal
#: *reasons* a cell-4 value is reported under. The builtin members admit
#: only one failure each — a ``bytearray`` orders but cannot hash, a
#: naive/aware ``datetime`` pair hashes but cannot order — so for them the
#: order of discovery is immaterial. A user type that registers with
#: ``numbers.Number`` (family ``"number"``), sets ``__hash__ = None`` *and*
#: defines no ``<`` fails both, and this path reports it as "order, but
#: cannot be hashed" where the sort-first path said "do not order"
#: (probe-confirmed, cycle-2 review finding G). Both sentences are true of
#: that value and both refusals are the same typed error from the same
#: builder, so nothing is misdiagnosed; the reason clause is simply the
#: first failure encountered. An exotic type reads as its own type name, is
#: not in here, and keeps the sort-then-dedupe path whose 4a-before-4b
#: priority it is entitled to.
_TOTALLY_ORDERED_FAMILIES = frozenset(
    {"number", "string", "bytes", "datetime", "date", "time", "timedelta"}
)

#: numpy dtype kinds whose ``.tolist()`` values all belong to one
#: :func:`_value_family`, so the family set can be read off the dtype instead
#: of from a per-value scan of a million rows. Deliberately partial:
#: ``"c"`` (complex) is a number that does not order and must reach
#: :func:`_value_family`'s own exclusion, and ``"M"``/``"m"`` (datetime64 /
#: timedelta64) are excluded because their ``.tolist()`` type depends on the
#: unit -- a ``datetime64[D]`` yields ``datetime.date`` and a
#: ``timedelta64[Y]`` yields ``int``, two different families from one kind.
#: ``"O"`` is excluded by construction: an object array is exactly the case
#: where only the values can say.
_DTYPE_KIND_FAMILY: dict[str, str] = {
    "b": "number",
    "i": "number",
    "u": "number",
    "f": "number",
    "U": "string",
    "S": "bytes",
}


def _families_of(values: np.ndarray, listed: list) -> set[str]:
    """Name the :func:`_value_family` set of ``values``, off the dtype when it can be.

    ``listed`` is ``values.tolist()`` (the caller already has it). For a
    non-object numpy dtype in :data:`_DTYPE_KIND_FAMILY`, every element is
    the same kind of Python object, so the dtype answers for all of them and
    the scan is skipped -- measured 2026-09-17 at 1e6 rows: the per-value
    scan of both arrays cost 318ms of ``report_labels``' 435ms, against 88ms
    for the sort the review asked about and 26ms for the dedupe. Object
    arrays (and the kinds deliberately left out of the map) still scan,
    which is the only way to answer for them and exactly what this function
    replaced.
    """
    family = _DTYPE_KIND_FAMILY.get(values.dtype.kind)
    if family is not None:
        return {family} if len(listed) else set()
    return {_value_family(v) for v in listed}


def report_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    diagnostic: str,
    predictions_from: PredictionsFrom = "caller",
) -> list[object]:
    """Build the report's class axis: both arrays' distinct values, ascending.

    The one owner of ``classification_report``'s class-axis derivation
    (#224a). It returns the *raw* values in scikit-learn ``unique_labels``
    order -- the union of both arrays' distinct values in their natural
    ascending order -- and the caller stringifies them, so the ``str()``
    that addresses the confusion frame's already-stringified
    ``actual``/``predicted`` columns stays visible at the call site.

    **Total by partition, not by patch.** Totality here is over the
    *values* the two arrays hold; their *shape* is the caller's
    precondition, enforced upstream and unchanged by this function (a
    non-1-D ``y_true`` or ``y_pred`` is refused by ``_require_1d_array``,
    and a scalar ``y_true`` never builds a ``_PrecomputedSource`` at all).
    Over values, every input lands in exactly one of four cells, and each
    cell is a named contract:

    1. *A null in either array* refuses, naming which one, through
       :func:`require_no_null_target` / :func:`require_no_null_predictions`
       and therefore through :func:`_null_mask`, so every dtype family's
       null spelling counts (object ``None``, float ``NaN``, temporal
       ``NaT``) rather than whichever one a repro happened to use.
    2. *Values from more than one ordering family* refuses as a mix of
       label types, the same shape scikit-learn's own ``unique_labels``
       refuses (see :func:`_value_family` for what counts as one family:
       an int/float or bool/int union is **not** mixed).
    3. *One family, orderable and hashable* is the answer: the distinct
       values, ascending, keeping ``y_true``'s representative of an equal
       pair (``0`` over ``0.0``, ``False`` over ``0``) -- the same one
       ``set(y_true) | set(y_pred)`` kept. Two routes reach it, and they
       agree by construction (see :data:`_TOTALLY_ORDERED_FAMILIES`): a
       totally-ordered family deduplicates first and sorts the survivors,
       while an exotic one sorts the full concatenation and then
       deduplicates by hash equality (``dict.fromkeys``, not adjacency --
       a partially-ordered family such as ``frozenset`` does not sort its
       equal values next to each other, so adjacency alone under-dedupes).
       The split is cost, not semantics: at 1e6 rows the full-concatenation
       sort costs 88ms against 26ms for the dedupe, and reading the family
       set off the dtype rather than scanning every value (see
       :func:`_families_of`) saves a further 318ms of the 435ms this
       function took before (measured 2026-09-17, whole-wave design review
       finding 7 -- the review's hypothesis was "seconds, not
       milliseconds", the measurement was 0.44s).
    4. *One family, unkeyable* refuses by name rather than mixed, since
       telling that caller to "use one consistent value type" would name a
       defect their input does not have. Two ways into this cell, both
       named: values that do not order (``complex``, ``dict``, a custom
       type with no ``<``), and values that order but cannot be hashed
       (``list``, ``set``, ``bytearray``) -- the confusion matrix this
       report is built from keys each class by its label value, so an
       orderable-but-unhashable label is exactly as unusable as an
       unorderable one, just discovered one statement later.

    The partition is the point. This function's predecessors each fixed
    the shape in front of them and left the next one open: the released
    ``sorted(..., key=str)`` rule stringified first, so it was total but
    ordered ``"10"`` before ``"2"``; #224a's first cycle sorted raw values
    with a bare ``sorted()`` and leaked ``TypeError: '<' not supported
    between instances of 'str' and 'int'``; its second wrapped that sort in
    ``except TypeError`` and so told a caller whose real defect was a null
    ``y_pred`` to go fix a type mix (measured 2026-09-13:
    ``y_true=["a","b","a","b"]``, ``y_pred=["a",None,"a","b"]``), while the
    float-``NaN`` twin never raised at all and rendered a ``'nan'`` class;
    its third partitioned nulls/mix/order but claimed a ``set``-free sort
    made hashability unnecessary, while the very next statement
    (``self.confusion_matrix(...)``) still hashed every label into a dict
    -- measured: ``list``/``set``/``bytearray`` labels answered here and
    then leaked ``TypeError: unhashable type`` one statement later, and a
    ``frozenset`` pair answered with a duplicate (adjacency-only dedupe
    under a partial order). Cell 4 now names both ways a value can be
    unusable as a class label, and cell 3's dedupe no longer assumes
    adjacency implies equality.

    Its two sibling union-derived label sites answered differently when this
    shipped; both have since been brought in (2026-09-17), so the class axis
    of every label-domain diagnostic on both source paths is this one
    derivation.

    ``ModelSource.confusion_matrix``'s no-``classes_`` branch had the
    totality hole this partition exists to close: a ``classes_``-less model
    predicting numbers against a string ``y`` raised the same unowned
    ``TypeError: '<' not supported`` from its raw ``sorted(set(...))``, and a
    dict-valued prediction raised ``TypeError: unhashable type`` one
    statement later. T1's disposition for that branch -- that its labels are
    the union of ``y_true``/``y_pred``, so no row can fall outside them and
    it needs no stray-value guard -- still holds.

    ``_PrecomputedSource.confusion_matrix``'s ``sorted(..., key=str)`` was
    kept one cycle longer, described as a class *order* the report's change
    was scoped not to touch. It was not only an order. Stringifying before
    comparing makes ``str`` totally order everything, which is why that rule
    accepted every input -- including ones the chart cannot draw: measured on
    ``y_true=[0, 1, 0, 1]`` against ``y_pred=["0", "1", "0", "1"]``, it
    derived four labels (``[0, '0', 1, '1']``) that stringify to two, so the
    frame carried duplicate ``(actual, predicted)`` keys and the chart drew
    them on top of each other. The ordering half moved with it: numeric
    labels now sort by value rather than by digit count, which is a rendered
    axis-order change disclosed as a ``**Breaking:**`` bullet, and string
    labels -- every golden in the repo -- sort identically under both rules.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Cell 1: a null in either array.
    require_no_null_target(y_true, diagnostic=diagnostic)
    require_no_null_predictions(y_pred, diagnostic=diagnostic, predictions_from=predictions_from)

    true_values = y_true.tolist()
    pred_values = y_pred.tolist()

    # Cell 2: values from more than one ordering family.
    families = _families_of(y_true, true_values) | _families_of(y_pred, pred_values)
    if len(families) > 1:
        raise ValueError(
            f"{diagnostic}: y_true and y_pred hold a mix of label types that "
            f"cannot be ordered together, the same shape scikit-learn's own "
            f"unique_labels refuses. y_true holds {_observed_values(y_true)!r} "
            f"and y_pred holds {_observed_values(y_pred)!r}. "
            + _MIXED_LABEL_CORRECTION[predictions_from].format(diagnostic=diagnostic)
        )

    if families <= _TOTALLY_ORDERED_FAMILIES:
        # Cells 4b then 4a, in that order, for the common case: dedupe the
        # concatenation by hash first and sort only the survivors. Under a
        # total order that is the same answer as sorting first (see
        # :data:`_TOTALLY_ORDERED_FAMILIES`) -- same representative, since
        # `dict.fromkeys` keeps the first occurrence and y_true's values
        # come first, and same order, since the survivors are distinct --
        # for a fraction of the work: 1e6 rows of three classes sort in
        # 88ms and dedupe in 26ms (measured 2026-09-17), and sorting after
        # the dedupe sorts three values instead of two million.
        try:
            uniques = list(dict.fromkeys(true_values + pred_values))
        except TypeError as exc:
            raise _unkeyable_label_error(
                family=next(iter(families)),
                reason=(
                    "They order, but cannot be hashed, and the confusion "
                    "matrix this report is built from keys each class by its "
                    "label value."
                ),
                correction=_UNKEYABLE_LABEL_CORRECTION[predictions_from].format(
                    diagnostic=diagnostic
                ),
                y_true=y_true,
                y_pred=y_pred,
                diagnostic=diagnostic,
            ) from exc
        try:
            return sorted(uniques)
        except (TypeError, ValueError) as exc:
            raise _unkeyable_label_error(
                family=next(iter(families)),
                reason=(
                    "They do not order against each other, so there is no class axis to put them on."
                ),
                correction=_UNKEYABLE_LABEL_CORRECTION[predictions_from].format(
                    diagnostic=diagnostic
                ),
                y_true=y_true,
                y_pred=y_pred,
                diagnostic=diagnostic,
            ) from exc

    # Cell 4a: values that do not order. Only the comparison operators of
    # the caller's own values can raise here, and both spellings of "these
    # do not order" are caught: TypeError from a type with no `<` (dict,
    # complex), ValueError from one whose `<` returns something that is
    # not a bool (an ndarray used as a label value).
    try:
        ordered = sorted(true_values + pred_values)
    except (TypeError, ValueError) as exc:
        # Exactly one family here: two or more already refused above, and
        # zero means both arrays were empty, which cannot raise.
        family = next(iter(families))
        raise _unkeyable_label_error(
            family=family,
            reason=(
                "They do not order against each other, so there is no class axis to put them on."
            ),
            correction=_UNKEYABLE_LABEL_CORRECTION[predictions_from].format(diagnostic=diagnostic),
            y_true=y_true,
            y_pred=y_pred,
            diagnostic=diagnostic,
        ) from exc

    # Cell 4b: values that order but cannot be hashed. The only consumer
    # of this class axis (the confusion matrix this report is built from)
    # keys each cell by its label value, so an orderable-but-unhashable
    # label is unusable in exactly the same way an unorderable one is --
    # just discovered one statement later if not caught here.
    try:
        labels = list(dict.fromkeys(ordered))
    except TypeError as exc:
        family = next(iter(families))
        raise _unkeyable_label_error(
            family=family,
            reason=(
                "They order, but cannot be hashed, and the confusion "
                "matrix this report is built from keys each class by its "
                "label value."
            ),
            correction=_UNKEYABLE_LABEL_CORRECTION[predictions_from].format(diagnostic=diagnostic),
            y_true=y_true,
            y_pred=y_pred,
            diagnostic=diagnostic,
        ) from exc

    # Cell 3: one family, orderable and hashable -- the answer.
    return labels
