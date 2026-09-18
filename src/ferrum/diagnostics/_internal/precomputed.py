"""Internal precomputed source adapter for diagnostic figure functions.

``_PrecomputedSource`` is never exported — not in ``ferrum.__init__`` or
``ferrum-spec.md §3.1``.  It adapts ``(y_true, y_pred)`` arrays to the same
method protocol ``ModelSource`` exposes, so the existing chart builders in
``ferrum.plots.classification`` and ``ferrum.plots.regression`` work without
modification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pyarrow as pa

from ferrum._core import studentized_residual_no_x

from . import _curve_frames, _target_hygiene


def _require_class_count_match(y_pred: np.ndarray, classes: list, *, diagnostic: str) -> None:
    """Refuse a 2-D ``y_pred`` whose column count doesn't match the class count (#208).

    roc/pr/gain/lift all read column ``i`` of a 2-D ``y_pred`` as class
    ``classes[i]``'s score. A width mismatch either raises deep inside
    ``_curve_frames`` (``IndexError``) or silently reads the wrong columns
    for a subset of classes -- both defect classes close to one typed
    refusal naming the two numbers.
    """
    n_cols = y_pred.shape[1]
    if n_cols != len(classes):
        raise ValueError(
            f"{diagnostic}: y_pred has {n_cols} score column(s) but y_true has "
            f"{len(classes)} class(es); a 2-D y_pred's column count must match "
            "the number of classes."
        )


def _require_not_multiclass_for_1d(classes: list, *, diagnostic: str) -> None:
    """Refuse a 1-D ``y_pred`` against a genuinely multiclass ``y_true`` (#208).

    A 1-D ``y_pred`` is a single positive-class score, so it cannot rank
    more than two classes. roc/pr used to silently map a 3+-class target
    to ``[unique[0], unique[-1]]`` (dropping every class in between) and
    gain/lift raised an untyped ``IndexError`` -- the family answered one
    question three ways. One typed refusal now, shared by all four.

    A *degenerate* single-class target (``len(classes) <= 1``) is not
    multiclass -- it is the "one class present, both slots describe it"
    case :func:`ferrum.diagnostics._internal._target_hygiene.binary_class_values`
    already handles gracefully (AUC/AP come back NaN, the curve still
    renders), so it is not refused here.
    """
    if len(classes) > 2:
        raise ValueError(
            f"{diagnostic} with a 1-D y_pred requires binary y_true; got "
            f"{len(classes)} unique classes."
        )


def _require_1d_array(array: np.ndarray, *, diagnostic: str, kind: str) -> None:
    """Refuse a non-1-D array where ``diagnostic`` requires 1-D ``kind`` (#208).

    Generic over which array it is called on -- ``kind`` names both what the
    array holds and, where more than one array could be malformed at the
    same call site, which one (e.g. ``"hard class labels (y_true)"``).
    """
    if array.ndim != 1:
        raise ValueError(
            f"{diagnostic} requires 1-D {kind}; got an array with shape {array.shape}."
        )


def _gain_lift_score_matrix(y_pred: np.ndarray, classes: list, *, diagnostic: str) -> np.ndarray:
    """Build a ``(n_samples, n_classes)`` score matrix for gain/lift assembly.

    For a 2-D ``y_pred`` (multiclass probability matrix), the columns are
    already per-class and returned as-is -- after refusing a column count
    that doesn't match ``len(classes)`` (#208).

    For a 1-D ``y_pred`` (binary positive-class scores), the matrix is built
    as ``column_stack([1 - y_pred, y_pred])`` so the negative class (column 0)
    is ranked by ``1 - p`` and the positive class (column 1) by ``p``.  This
    matches the precomputed 1-D binary roc/pr path
    (``_PrecomputedSource.roc_curve`` / ``.pr_curve``) and the model-backed
    path (where ``probabilities()`` already emits genuine per-class columns),
    so the whole precomputed-1D-binary family ranks each class by its own
    score.  ``gain_frame``/``lift_frame`` call ``argsort(-score_matrix[:, i])``
    per class against this matrix. Refuses multiclass ``y_true`` against a
    1-D ``y_pred`` (#208) -- a single positive-class score cannot rank more
    than two classes.
    """
    if y_pred.ndim == 2:
        _require_class_count_match(y_pred, classes, diagnostic=diagnostic)
        return y_pred
    _require_not_multiclass_for_1d(classes, diagnostic=diagnostic)
    # Binary: 1-D y_pred is the positive-class score. Build a per-class matrix
    # so the negative class ranks by 1 - p, mirroring the roc/pr binary path.
    return np.column_stack([1.0 - y_pred, y_pred])


class _PrecomputedSource:
    """Lightweight adapter that wraps raw (y_true, y_pred) arrays.

    Satisfies the subset of ``ModelSource``'s method protocol required by the
    nine in-scope diagnostic chart builders.  Delegates computation to Rust
    kernels in ``ferrum._core``; no scikit-learn is required.

    Every classification method here runs the shared target-hygiene guard
    (``_target_hygiene.require_no_null_target``, #203) before reading class
    values off ``y_true``, naming the *figure function* the caller invoked
    the way ``_require_1d_array`` already does. ``classification_report``
    and ``confusion_matrix`` run it inside ``_target_hygiene.report_labels``
    rather than calling it directly, because they need the same guard on
    ``y_pred`` too (#224a; confusion joined in the close remediation);
    the message and the diagnostic name are the sibling methods'. Without
    the guard a null among string labels raised an unowned ``TypeError``
    from ``np.unique`` /
    ``sorted`` here just as it did on the model path, and the two numeric
    paths disagreed: ``roc_chart`` refused for the wrong reason ("3 unique
    classes", counting ``NaN`` as one), while ``confusion_matrix_chart``
    and ``classification_report_chart`` quietly rendered ``None`` as a
    class of its own because their ``sorted(..., key=str)`` survived it.

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
        diagnostic: str | None = None,
    ) -> pl.DataFrame:
        """ROC curve(s).  Mirrors ``ModelSource.roc_curve`` column schema.

        ``diagnostic`` names the public entry a refusal or warning from this
        call carries, and defaults to the figure function that owns this
        input path -- the same keyword ``ModelSource``'s twin takes, so a
        figure function can pass its own name to either source kind without
        branching (whole-wave design review finding 4, 2026-09-17).
        """
        entry = diagnostic or "roc_chart"
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        _target_hygiene.require_no_null_target(y_true, diagnostic=entry)
        classes = list(np.unique(y_true))

        if y_pred.ndim == 1:
            # Binary: y_pred is 1-D scores for the positive class. Reshape to a
            # 2-column score matrix so the shared builder takes the binary path.
            _require_not_multiclass_for_1d(classes, diagnostic=entry)
            class_values, class_labels = _target_hygiene.binary_class_values(y_true)
            score_matrix = np.column_stack([1.0 - y_pred, y_pred])
            return _curve_frames.roc_frame(
                y_true,
                score_matrix,
                class_values,
                class_labels,
                average=None,
                drop_intermediate=drop_intermediate,
                diagnostic=entry,
            )

        # Multiclass: y_pred is (n_samples, n_classes); columns map to sorted
        # unique classes (ascending, matching the per-class score column order).
        _require_class_count_match(y_pred, classes, diagnostic=entry)
        labels = [str(cls) for cls in classes]
        return _curve_frames.roc_frame(
            y_true,
            y_pred,
            classes,
            labels,
            average=average,
            drop_intermediate=drop_intermediate,
            diagnostic=entry,
        )

    def pr_curve(
        self, *, average: str | None = None, diagnostic: str | None = None
    ) -> pl.DataFrame:
        """Precision-recall curve(s).  Mirrors ``ModelSource.pr_curve`` schema.

        ``diagnostic`` names the public entry a refusal or warning from this
        call carries, and defaults to the figure function that owns this
        input path -- the same keyword ``ModelSource``'s twin takes, so a
        figure function can pass its own name to either source kind without
        branching (whole-wave design review finding 4, 2026-09-17).
        """
        entry = diagnostic or "pr_chart"
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        _target_hygiene.require_no_null_target(y_true, diagnostic=entry)
        classes = list(np.unique(y_true))

        if y_pred.ndim == 1:
            _require_not_multiclass_for_1d(classes, diagnostic=entry)
            class_values, class_labels = _target_hygiene.binary_class_values(y_true)
            score_matrix = np.column_stack([1.0 - y_pred, y_pred])
            return _curve_frames.pr_frame(
                y_true,
                score_matrix,
                class_values,
                class_labels,
                average=None,
                diagnostic=entry,
            )

        _require_class_count_match(y_pred, classes, diagnostic=entry)
        labels = [str(cls) for cls in classes]
        return _curve_frames.pr_frame(
            y_true,
            y_pred,
            classes,
            labels,
            average=average,
            diagnostic=entry,
        )

    def calibration_curve(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
        diagnostic: str | None = None,
    ) -> pl.DataFrame:
        """Reliability diagram bins.  Mirrors ``ModelSource.calibration_curve`` schema.

        Includes the broadcast ``brier`` column the shared builder computes
        from the per-sample ``(y_pred, binarized y_true)`` pair -- the one
        number every Brier display surface reads.

        ``diagnostic`` names the public entry a refusal or warning from this
        call carries, and defaults to the figure function that owns this
        input path -- the same keyword ``ModelSource``'s twin takes, so a
        figure function can pass its own name to either source kind without
        branching (whole-wave design review finding 4, 2026-09-17).
        """
        entry = diagnostic or "calibration_chart"
        y_pred = self._y_pred_np
        _require_1d_array(y_pred, diagnostic=entry, kind="positive-class probabilities")
        _target_hygiene.require_no_null_target(self._y_true_np, diagnostic=entry)
        class_values, _ = _target_hygiene.binary_class_values(self._y_true_np)
        # The only precomputed diagnostic that can absorb a stray: roc/pr/gain/
        # lift read their class set off `y_true` itself, so every row is in it
        # by construction, while `binary_class_values` reduces a 3+-valued
        # target to its extremes and counts everything between as negative.
        _target_hygiene.check_resolved_target(
            self._y_true_np, class_values, diagnostic=entry, stacklevel=3
        )
        return _curve_frames.calibration_frame(
            self._y_true_np,
            y_pred,  # 1-D probabilities for positive class
            class_values[1],
            n_bins,
            strategy,
            diagnostic=entry,
        )

    def cumulative_gain(self, *, diagnostic: str | None = None) -> pl.DataFrame:
        """Cumulative-gain curve.  Mirrors ``ModelSource.cumulative_gain`` schema.

        ``diagnostic`` names the public entry a refusal or warning from this
        call carries, and defaults to the figure function that owns this
        input path -- the same keyword ``ModelSource``'s twin takes, so a
        figure function can pass its own name to either source kind without
        branching (whole-wave design review finding 4, 2026-09-17).
        """
        entry = diagnostic or "gain_chart"
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        _target_hygiene.require_no_null_target(y_true, diagnostic=entry)
        classes = list(np.unique(y_true))
        labels = [str(cls) for cls in classes]
        score_matrix = _gain_lift_score_matrix(y_pred, classes, diagnostic=entry)
        return _curve_frames.gain_frame(y_true, score_matrix, classes, labels, diagnostic=entry)

    def lift_curve(self, *, diagnostic: str | None = None) -> pl.DataFrame:
        """Lift curve.  Mirrors ``ModelSource.lift_curve`` schema.

        ``diagnostic`` names the public entry a refusal or warning from this
        call carries, and defaults to the figure function that owns this
        input path -- the same keyword ``ModelSource``'s twin takes, so a
        figure function can pass its own name to either source kind without
        branching (whole-wave design review finding 4, 2026-09-17).
        """
        entry = diagnostic or "lift_chart"
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        _target_hygiene.require_no_null_target(y_true, diagnostic=entry)
        classes = list(np.unique(y_true))
        labels = [str(cls) for cls in classes]
        score_matrix = _gain_lift_score_matrix(y_pred, classes, diagnostic=entry)
        return _curve_frames.lift_frame(y_true, score_matrix, classes, labels, diagnostic=entry)

    def confusion_matrix(
        self, *, normalize: str | None = None, diagnostic: str | None = None
    ) -> pl.DataFrame:
        """Confusion matrix in long form.  Mirrors ``ModelSource.confusion_matrix`` schema.

        ``y_pred`` must be 1-D hard class labels.

        The class axis is :func:`_target_hygiene.report_labels` -- the one
        partition-total derivation the whole label-domain family shares
        (#224a; this path joined it 2026-09-17). Every value either lands on
        the axis or is refused by name: a null in either array, a mix of
        ordering families, an orderable-and-hashable union (the answer), or
        a union that cannot key a class.

        This replaced the method's own ``sorted(set(...), key=str)`` rule,
        which stringified before comparing and so answered *everything* --
        including inputs it could not draw. Measured 2026-09-13/17 on
        ``y_true=[0, 1, 0, 1]`` with ``y_pred=["0", "1", "0", "1"]``: it
        derived four labels ``[0, '0', 1, '1']`` that stringify to two, and
        the frame carried duplicate ``(actual, predicted)`` keys whose true
        diagonal mass was split across visually identical cells, which
        ``confusion_matrix_chart`` rendered as four superimposed ``<text>``
        nodes at one pixel position. That is not an ordering preference, so
        it is not kept as one; it refuses now, as
        ``classification_report_chart`` always did on the identical arrays.
        Numeric labels also order by value rather than by digit count
        (``['1', '2', '10']``, not ``['1', '10', '2']``), which aligns this
        path with the model path and with the report -- a rendered axis-order
        move, disclosed as a ``**Breaking:**`` changelog bullet.

        ``diagnostic`` names the public entry a refusal or warning from this
        call carries, and defaults to the figure function that owns this
        input path -- the same keyword ``ModelSource``'s twin takes, so a
        figure function can pass its own name to either source kind without
        branching (whole-wave design review finding 4, 2026-09-17).
        """
        entry = diagnostic or "confusion_matrix_chart"
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        # Shape is this method's own precondition, refused before
        # `report_labels` is asked to make sense of the values (the same
        # order `classification_report` uses).
        _require_1d_array(y_pred, diagnostic=entry, kind="hard class labels")
        labels = _target_hygiene.report_labels(y_true, y_pred, diagnostic=entry)
        return _curve_frames.confusion_frame(y_true, y_pred, labels, normalize=normalize)

    def classification_report(self) -> pl.DataFrame:
        """Per-class precision/recall/F1 in long form, from confusion counts.

        No scikit-learn required (F-L11-05, 2026-09-10): precision, recall,
        and F1 are computed directly as ratios of the confusion matrix's
        per-class counts -- ``precision_c = TP_c / predicted_total_c``,
        ``recall_c = TP_c / actual_total_c``, and ``f1_c`` their harmonic
        mean (0 when either denominator is 0, matching sklearn's
        ``classification_report(zero_division=0)`` convention) -- so the
        precomputed path never imports sklearn. ``y_pred`` must be 1-D
        hard class labels, same as ``confusion_matrix``.

        Schema (``class, metric, value, value_fmt``) matches the model
        path's sklearn-backed report, so ``classification_report_chart``
        renders either source kind uniformly.

        Class order (#224a, 2026-09-13): the class axis is ordered by
        sklearn's ``unique_labels`` convention -- the union of ``y_true``
        and ``y_pred``'s own distinct values, in their natural ascending
        order -- computed from the *raw* arrays rather than from
        ``confusion_matrix``'s already-stringified ``actual``/``predicted``
        columns. That distinction matters for a numeric target: stringifying
        first and then sorting (the ``sorted(..., key=str)`` rule, retired
        here and — in the close remediation — on ``confusion_matrix`` too) sorts
        lexicographically, so it orders by digit count before value --
        ``"10"`` lands between ``"1"`` and ``"2"`` -- while sorting the raw
        numeric values first sorts ascending. That is exactly what the
        model path's sklearn-backed ``classification_report`` does, since
        it never stringifies before resolving ``unique_labels``. This
        retires the documented divergence between the two paths; both order
        the class axis identically now. Both paths already agreed on the
        row *values* (see the model-vs-precomputed parity test). The two
        orders differ for any numeric class set whose members have unequal
        digit counts, two classes included -- measured 2026-09-13 on
        ``{2, 10}``: ``['10', '2']`` before, ``['2', '10']`` now.

        The label derivation is :func:`_target_hygiene.report_labels`, which
        owns the whole partition of what ``y_true``/``y_pred`` can hold: a
        null in either array, a mix of ordering families, an orderable and
        hashable homogeneous union (the answer), and an unkeyable one
        (unorderable or orderable-but-unhashable) each get a named
        outcome, so no input shape reaches this method's arithmetic
        without one.
        """
        # Both arrays' shape is this method's own precondition -- unlike
        # report_labels' value partition below, a non-1-D array is not a
        # value report_labels is asked to make sense of, so it is refused
        # here, by name, before report_labels ever sees it.
        _require_1d_array(
            self._y_true_np,
            diagnostic="classification_report_chart",
            kind="hard class labels (y_true)",
        )
        _require_1d_array(
            self._y_pred_np,
            diagnostic="classification_report_chart",
            kind="hard class labels (y_pred)",
        )
        # Ahead of the confusion delegation below, so a refusal names this
        # figure rather than the matrix it is built from -- and so an
        # unkeyable label value (report_labels' cell 4b: orderable but
        # unhashable, e.g. a list or a set) is refused by name rather than
        # crashing inside the matrix's dict-keyed label lookup. report_labels
        # runs the null guard the sibling methods call directly, on both
        # arrays rather than on y_true alone.
        raw_labels = _target_hygiene.report_labels(
            self._y_true_np, self._y_pred_np, diagnostic="classification_report_chart"
        )
        # str() is the stringifier on both sides of this correspondence:
        # the confusion frame's actual/predicted columns are built the same
        # way, so these labels address its cells.
        labels = [str(v) for v in raw_labels]

        cm = self.confusion_matrix(normalize=None)
        actual_col = cm["actual"].to_list()
        predicted_col = cm["predicted"].to_list()
        value_col = cm["value"].to_list()

        predicted_totals: dict[str, float] = {}
        actual_totals: dict[str, float] = {}
        true_positives: dict[str, float] = {}
        for actual, predicted, value in zip(actual_col, predicted_col, value_col):
            value = float(value)
            actual_totals[actual] = actual_totals.get(actual, 0.0) + value
            predicted_totals[predicted] = predicted_totals.get(predicted, 0.0) + value
            if actual == predicted:
                true_positives[actual] = true_positives.get(actual, 0.0) + value

        per_class_metrics: dict[str, dict[str, float]] = {}
        for label in labels:
            tp = true_positives.get(label, 0.0)
            pred_total = predicted_totals.get(label, 0.0)
            actual_total = actual_totals.get(label, 0.0)
            precision = tp / pred_total if pred_total > 0 else 0.0
            recall = tp / actual_total if actual_total > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class_metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
            }
        return _curve_frames.classification_report_frame(per_class_metrics)

    def predictions(self, *, diagnostic: str = "residuals_chart") -> pl.DataFrame:
        """Return y_true, y_pred, residual, studentized_residual, cooks_distance, leverage.

        ``y_pred`` must be 1-D fitted values.  ``studentized_residual`` is
        computed without the design matrix (``ferrum._core.studentized_residual_no_x``),
        matching the non-linear estimator path in ``ModelSource``.
        ``cooks_distance`` and ``leverage`` are NaN (no design matrix available).

        ``diagnostic`` (#208; spelled ``chart_name`` until 2026-09-17, when
        the two names for this one concept were unified -- cycle-2 review
        finding C): this adapter method is consumed by TWO public figure
        functions -- ``residuals_chart`` (``plots/regression.py``, the
        default) and ``prediction_error_chart`` (which passes its own name
        explicitly, mirroring ``_gain_lift_score_matrix``'s own
        ``diagnostic=`` parametrization for the same reason: one adapter
        method, multiple public callers, and the #208 message idiom
        requires naming the caller actually invoked, not just any
        consumer). ``predictions()`` is also the shared
        ``DiagnosticSource`` protocol method ``ModelSource`` implements
        with no such parameter -- keeping it optional here means
        ``source.predictions()`` still dispatches polymorphically for
        callers that don't need to override the name.
        """
        _require_1d_array(self._y_pred_np, diagnostic=diagnostic, kind="fitted values")
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
        diagnostic: str | None = None,
    ) -> pl.DataFrame:
        """Threshold sweep.  Mirrors ``ModelSource.discrimination_threshold`` schema.

        ``y_pred`` must be 1-D soft scores for the positive class.
        ``cv`` is not supported on the precomputed path (no model to re-fit).

        ``diagnostic`` names the public entry a refusal or warning from this
        call carries, and defaults to the figure function that owns this
        input path -- the same keyword ``ModelSource``'s twin takes, so a
        figure function can pass its own name to either source kind without
        branching (whole-wave design review finding 4, 2026-09-17).
        """
        entry = diagnostic or "discrimination_threshold_chart"
        if cv is not None:
            raise ValueError(
                "discrimination_threshold_chart: cv= is not supported with "
                "precomputed y_true/y_pred inputs — cross-validation requires "
                "a fitted model.  Pass cv=None or use the model-backed path."
            )

        y_true = self._y_true_np
        y_pred = self._y_pred_np  # 1-D positive-class scores
        _require_1d_array(y_pred, diagnostic=entry, kind="positive-class scores")
        _target_hygiene.require_no_null_target(y_true, diagnostic=entry)

        unique = np.unique(y_true)
        if len(unique) != 2:
            # Pre-existing, strict binary-only rule (unchanged by #208):
            # unlike roc/pr/gain/lift's 1-D path, a degenerate single-class
            # target was never accepted here, so this stays `!= 2` rather
            # than sharing `_require_not_multiclass_for_1d`'s `> 2`.
            raise ValueError(
                "discrimination_threshold_chart with precomputed inputs requires "
                f"binary y_true; got {len(unique)} unique classes."
            )
        positive_class = unique[-1]
        thresholds_np = np.linspace(0.0, 1.0, n_thresholds)
        return _curve_frames.threshold_sweep_frame(
            y_true,
            y_pred,
            positive_class,
            thresholds_np,
            diagnostic=entry,
        )


if TYPE_CHECKING:
    # Type-only conformance: ``_PrecomputedSource`` hand-matches the subset of
    # the ``DiagnosticSource`` contract its nine in-scope chart builders need.
    # Annotating it here lets pyright flag any future signature/return drift
    # from ``ModelSource`` at the seam where chart builders consume them.
    from ..sources._protocols import DiagnosticSource

    def _assert_precomputed_diagnostic(src: "_PrecomputedSource") -> DiagnosticSource:
        return src
