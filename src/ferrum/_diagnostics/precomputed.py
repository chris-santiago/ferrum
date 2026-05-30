"""Internal precomputed source adapter for diagnostic figure functions.

``_PrecomputedSource`` is never exported — not in ``ferrum.__init__`` or
``ferrum-spec.md §3.1``.  It adapts ``(y_true, y_pred)`` arrays to the same
method protocol ``ModelSource`` exposes, so the existing chart builders in
``ferrum.plots.classification`` and ``ferrum.plots.regression`` work without
modification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa

from ferrum._core import studentized_residual_no_x

from . import _curve_frames


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
            pos_class = str(np.unique(y_true)[-1]) if len(np.unique(y_true)) >= 2 else "1"
            score_matrix = np.column_stack([1.0 - y_pred, y_pred])
            return _curve_frames.roc_frame(
                y_true,
                score_matrix,
                ["0", pos_class],
                ["0", pos_class],
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
            pos_class = str(np.unique(y_true)[-1]) if len(np.unique(y_true)) >= 2 else "1"
            score_matrix = np.column_stack([1.0 - y_pred, y_pred])
            return _curve_frames.pr_frame(
                y_true,
                score_matrix,
                ["0", pos_class],
                ["0", pos_class],
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
        """Reliability diagram bins.  Mirrors ``ModelSource.calibration_curve`` schema."""
        return _curve_frames.calibration_frame(
            self._y_true_np,
            self._y_pred_np,  # 1-D probabilities for positive class
            n_bins,
            strategy,
        )

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
        return _curve_frames.confusion_frame(y_true, y_pred, labels, normalize=normalize)

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
        return _curve_frames.threshold_sweep_frame(y_true_bin, y_pred, thresholds_np)
