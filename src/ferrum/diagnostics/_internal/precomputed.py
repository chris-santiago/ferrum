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

from . import _curve_frames


def _gain_lift_score_matrix(y_pred: np.ndarray, classes: list) -> np.ndarray:
    """Build a ``(n_samples, n_classes)`` score matrix for gain/lift assembly.

    For a 2-D ``y_pred`` (multiclass probability matrix), the columns are
    already per-class and returned as-is.

    For a 1-D ``y_pred`` (binary positive-class scores), the matrix is built
    as ``column_stack([1 - y_pred, y_pred])`` so the negative class (column 0)
    is ranked by ``1 - p`` and the positive class (column 1) by ``p``.  This
    matches the precomputed 1-D binary roc/pr path
    (``_PrecomputedSource.roc_curve`` / ``.pr_curve``) and the model-backed
    path (where ``probabilities()`` already emits genuine per-class columns),
    so the whole precomputed-1D-binary family ranks each class by its own
    score.  ``gain_frame``/``lift_frame`` call ``argsort(-score_matrix[:, i])``
    per class against this matrix.
    """
    if y_pred.ndim == 2:
        return y_pred
    # Binary: 1-D y_pred is the positive-class score. Build a per-class matrix
    # so the negative class ranks by 1 - p, mirroring the roc/pr binary path.
    return np.column_stack([1.0 - y_pred, y_pred])


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
            # Binary: y_pred is 1-D scores for the positive class. Reshape to a
            # 2-column score matrix so the shared builder takes the binary path.
            class_values, class_labels = _curve_frames.binary_class_values(y_true)
            score_matrix = np.column_stack([1.0 - y_pred, y_pred])
            return _curve_frames.roc_frame(
                y_true,
                score_matrix,
                class_values,
                class_labels,
                average=None,
                drop_intermediate=drop_intermediate,
            )

        # Multiclass: y_pred is (n_samples, n_classes); columns map to sorted
        # unique classes (ascending, matching the per-class score column order).
        classes = list(np.unique(y_true))
        labels = [str(cls) for cls in classes]
        return _curve_frames.roc_frame(
            y_true,
            y_pred,
            classes,
            labels,
            average=average,
            drop_intermediate=drop_intermediate,
        )

    def pr_curve(self, *, average: str | None = None) -> pl.DataFrame:
        """Precision-recall curve(s).  Mirrors ``ModelSource.pr_curve`` schema."""
        y_true = self._y_true_np
        y_pred = self._y_pred_np

        if y_pred.ndim == 1:
            class_values, class_labels = _curve_frames.binary_class_values(y_true)
            score_matrix = np.column_stack([1.0 - y_pred, y_pred])
            return _curve_frames.pr_frame(
                y_true,
                score_matrix,
                class_values,
                class_labels,
                average=None,
            )

        classes = list(np.unique(y_true))
        labels = [str(cls) for cls in classes]
        return _curve_frames.pr_frame(
            y_true,
            y_pred,
            classes,
            labels,
            average=average,
        )

    def calibration_curve(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> pl.DataFrame:
        """Reliability diagram bins.  Mirrors ``ModelSource.calibration_curve`` schema.

        Includes the broadcast ``brier`` column the shared builder computes
        from the per-sample ``(y_pred, binarized y_true)`` pair -- the one
        number every Brier display surface reads.
        """
        class_values, _ = _curve_frames.binary_class_values(self._y_true_np)
        return _curve_frames.calibration_frame(
            self._y_true_np,
            self._y_pred_np,  # 1-D probabilities for positive class
            class_values[1],
            n_bins,
            strategy,
        )

    def cumulative_gain(self) -> pl.DataFrame:
        """Cumulative-gain curve.  Mirrors ``ModelSource.cumulative_gain`` schema."""
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        classes = list(np.unique(y_true))
        labels = [str(cls) for cls in classes]
        score_matrix = _gain_lift_score_matrix(y_pred, classes)
        return _curve_frames.gain_frame(y_true, score_matrix, classes, labels)

    def lift_curve(self) -> pl.DataFrame:
        """Lift curve.  Mirrors ``ModelSource.lift_curve`` schema."""
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        classes = list(np.unique(y_true))
        labels = [str(cls) for cls in classes]
        score_matrix = _gain_lift_score_matrix(y_pred, classes)
        return _curve_frames.lift_frame(y_true, score_matrix, classes, labels)

    def confusion_matrix(self, *, normalize: str | None = None) -> pl.DataFrame:
        """Confusion matrix in long form.  Mirrors ``ModelSource.confusion_matrix`` schema.

        ``y_pred`` must be 1-D hard class labels.
        """
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)
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

        Class *order* is a documented divergence, not a bug: labels are
        ``sorted(..., key=str)`` here (the ``confusion_matrix`` precedent
        this method reuses, same as ``ModelSource.confusion_matrix``'s
        own ``sorted(...)`` fallback), whereas the model path inherits
        sklearn's numeric ``unique_labels`` order. For fewer than 10
        numeric classes the two agree; at 10+ they diverge (string order
        interleaves ``"10"``, ``"11"`` before ``"2"``). Both paths agree
        on the row *values* (see the model-vs-precomputed parity test),
        just not the class-axis render order.
        """
        cm = self.confusion_matrix(normalize=None)
        actual_col = cm["actual"].to_list()
        predicted_col = cm["predicted"].to_list()
        value_col = cm["value"].to_list()
        labels = sorted(set(actual_col) | set(predicted_col), key=str)

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
        thresholds_np = np.linspace(0.0, 1.0, n_thresholds)
        return _curve_frames.threshold_sweep_frame(y_true, y_pred, positive_class, thresholds_np)


if TYPE_CHECKING:
    # Type-only conformance: ``_PrecomputedSource`` hand-matches the subset of
    # the ``DiagnosticSource`` contract its nine in-scope chart builders need.
    # Annotating it here lets pyright flag any future signature/return drift
    # from ``ModelSource`` at the seam where chart builders consume them.
    from ..sources._protocols import DiagnosticSource

    def _assert_precomputed_diagnostic(src: "_PrecomputedSource") -> DiagnosticSource:
        return src
