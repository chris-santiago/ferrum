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


class _PrecomputedSource:
    """Lightweight adapter that wraps raw (y_true, y_pred) arrays.

    Satisfies the subset of ``ModelSource``'s method protocol required by the
    nine in-scope diagnostic chart builders.  Delegates computation directly
    to ``sklearn.metrics.*``; no bespoke math lives here.

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
        from sklearn.metrics import roc_curve as _sk_roc, roc_auc_score

        y_true = self._y_true_np
        y_pred = self._y_pred_np
        rows: list[dict] = []

        if y_pred.ndim == 1:
            # Binary: y_pred is 1-D scores for the positive class.
            pos_class = str(np.unique(y_true)[-1]) if len(np.unique(y_true)) >= 2 else "1"
            fpr, tpr, thr = _sk_roc(y_true, y_pred, drop_intermediate=drop_intermediate)
            try:
                auc = float(roc_auc_score(y_true, y_pred))
            except ValueError:
                auc = float("nan")
            for f, t, h in zip(fpr, tpr, thr):
                rows.append(
                    {
                        "fpr": float(f),
                        "tpr": float(t),
                        "threshold": float(h),
                        "class": pos_class,
                        "auc": auc,
                    }
                )
        else:
            # Multiclass: y_pred is (n_samples, n_classes); columns map to
            # sorted unique classes (sklearn convention).
            classes = list(np.unique(y_true))
            for i, cls in enumerate(classes):
                y_bin = (y_true == cls).astype(int)
                y_score = y_pred[:, i]
                fpr, tpr, thr = _sk_roc(y_bin, y_score, drop_intermediate=drop_intermediate)
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
                rows.extend(_avg_roc_rows(y_true, y_pred, classes, average, drop_intermediate))

        return pl.DataFrame(rows)

    def pr_curve(self, *, average: str | None = None) -> pl.DataFrame:
        """Precision-recall curve(s).  Mirrors ``ModelSource.pr_curve`` schema."""
        y_true = self._y_true_np
        y_pred = self._y_pred_np

        if y_pred.ndim == 1:
            rows = _pr_rows_binary_np(y_true, y_pred)
        elif average in ("micro", "macro", "weighted"):
            classes = list(np.unique(y_true))
            rows = _avg_pr_rows(y_true, y_pred, classes, average)
        else:
            classes = list(np.unique(y_true))
            rows = _pr_rows_per_class_np(y_true, y_pred, classes)

        return pl.DataFrame(rows)

    def calibration_curve(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> pl.DataFrame:
        """Reliability diagram bins.  Mirrors ``ModelSource.calibration_curve`` schema."""
        from sklearn.calibration import calibration_curve as _ccurve

        y_true = self._y_true_np
        y_pred = self._y_pred_np  # 1-D probabilities for positive class

        frac_pos, mean_pred = _ccurve(y_true, y_pred, n_bins=n_bins, strategy=strategy)

        if strategy == "uniform":
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        elif strategy == "quantile":
            edges = np.quantile(y_pred, np.linspace(0.0, 1.0, n_bins + 1))
        else:
            raise ValueError(
                f"calibration_curve(strategy={strategy!r}) not supported; "
                "use 'uniform' or 'quantile'."
            )
        bin_idx = np.clip(np.digitize(y_pred, edges[1:-1]), 0, n_bins - 1)
        counts_all = np.bincount(bin_idx, minlength=n_bins)
        centers = edges[:-1] + np.diff(edges) / 2.0
        used_bins = np.array([int(np.argmin(np.abs(centers - mp))) for mp in mean_pred], dtype=int)
        counts = counts_all[used_bins] if used_bins.size else np.empty(0, dtype=int)

        return pl.DataFrame(
            {
                "mean_predicted": [float(x) for x in mean_pred],
                "fraction_positive": [float(x) for x in frac_pos],
                "count": [int(x) for x in counts],
            }
        )

    def cumulative_gain(self) -> pl.DataFrame:
        """Cumulative-gain curve.  Mirrors ``ModelSource.cumulative_gain`` schema."""
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        n = len(y_true)
        rows: list[dict] = []

        if y_pred.ndim == 1:
            # Binary: treat as scores for the single positive class.
            classes = list(np.unique(y_true))
            for i, cls in enumerate(classes):
                y_bin = (y_true == cls).astype(int)
                scores = y_pred
                order = np.argsort(-scores)
                cum_pos = np.cumsum(y_bin[order])
                total_pos = max(int(cum_pos[-1]), 1) if n else 1
                pct_pop = np.arange(1, n + 1) / max(n, 1)
                gain = cum_pos / total_pos
                xs = np.concatenate([[0.0], pct_pop])
                ys = np.concatenate([[0.0], gain])
                for pp, g in zip(xs, ys):
                    rows.append(
                        {"percent_population": float(pp), "gain": float(g), "class": str(cls)}
                    )
        else:
            classes = list(np.unique(y_true))
            for i, cls in enumerate(classes):
                y_bin = (y_true == cls).astype(int)
                order = np.argsort(-y_pred[:, i])
                cum_pos = np.cumsum(y_bin[order])
                total_pos = max(int(cum_pos[-1]), 1) if n else 1
                pct_pop = np.arange(1, n + 1) / max(n, 1)
                gain = cum_pos / total_pos
                xs = np.concatenate([[0.0], pct_pop])
                ys = np.concatenate([[0.0], gain])
                for pp, g in zip(xs, ys):
                    rows.append(
                        {"percent_population": float(pp), "gain": float(g), "class": str(cls)}
                    )

        rows.append({"percent_population": 0.0, "gain": 0.0, "class": "baseline"})
        rows.append({"percent_population": 1.0, "gain": 1.0, "class": "baseline"})
        return pl.DataFrame(rows)

    def lift_curve(self) -> pl.DataFrame:
        """Lift curve.  Mirrors ``ModelSource.lift_curve`` schema."""
        y_true = self._y_true_np
        y_pred = self._y_pred_np
        n = len(y_true)
        rows: list[dict] = []

        if y_pred.ndim == 1:
            classes = list(np.unique(y_true))
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
                for pp, lv in zip(pct_pop, lift):
                    rows.append(
                        {"percent_population": float(pp), "lift": float(lv), "class": str(cls)}
                    )
        else:
            classes = list(np.unique(y_true))
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
                for pp, lv in zip(pct_pop, lift):
                    rows.append(
                        {"percent_population": float(pp), "lift": float(lv), "class": str(cls)}
                    )

        rows.append({"percent_population": 0.0, "lift": 1.0, "class": "baseline"})
        rows.append({"percent_population": 1.0, "lift": 1.0, "class": "baseline"})
        return pl.DataFrame(rows)

    def confusion_matrix(self, *, normalize: str | None = None) -> pl.DataFrame:
        """Confusion matrix in long form.  Mirrors ``ModelSource.confusion_matrix`` schema.

        ``y_pred`` must be 1-D hard class labels.
        """
        from sklearn.metrics import confusion_matrix as _cm

        y_true = self._y_true_np
        y_pred = self._y_pred_np
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)

        cm = _cm(y_true, y_pred, labels=labels, normalize=normalize)
        rows: list[dict] = []
        for i, a in enumerate(labels):
            for j, p in enumerate(labels):
                val = float(cm[i, j])
                fmt = f"{val:.2f}" if normalize is not None else f"{int(val)}"
                rows.append({"actual": str(a), "predicted": str(p), "value": val, "value_fmt": fmt})
        return pl.DataFrame(rows)

    def predictions(self) -> pl.DataFrame:
        """Return y_true, y_pred, residual, studentized_residual, cooks_distance, leverage.

        ``y_pred`` must be 1-D fitted values.  ``studentized_residual`` is
        computed without the design matrix (``ferrum._core.studentized_residual_no_x``),
        matching the non-linear estimator path in ``ModelSource``.
        ``cooks_distance`` and ``leverage`` are NaN (no design matrix available).
        """
        import pyarrow as pa
        from ferrum._core import studentized_residual_no_x

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
        from sklearn.metrics import precision_recall_fscore_support

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
        thresholds = np.linspace(0.0, 1.0, n_thresholds)

        rows: list[dict] = []
        for t in thresholds:
            y_hard = (y_pred >= t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                y_true_bin, y_hard, average="binary", zero_division=0
            )
            queue_rate = float((y_pred >= t).mean()) if y_pred.size else 0.0
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


# ---------------------------------------------------------------------------
# Module-private helpers (not part of any public API)
# ---------------------------------------------------------------------------


def _pr_rows_binary_np(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    from sklearn.metrics import precision_recall_curve, average_precision_score

    pos_class = str(np.unique(y_true)[-1]) if len(np.unique(y_true)) >= 2 else "1"
    p, r, thr = precision_recall_curve(y_true, y_pred)
    try:
        ap = float(average_precision_score(y_true, y_pred))
    except ValueError:
        ap = float("nan")
    thresholds_padded = np.concatenate([thr, [float("nan")]])
    return [
        {
            "precision": float(pi),
            "recall": float(ri),
            "threshold": float(ti),
            "class": pos_class,
            "ap": ap,
        }
        for pi, ri, ti in zip(p, r, thresholds_padded)
    ]


def _pr_rows_per_class_np(y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> list[dict]:
    from sklearn.metrics import precision_recall_curve, average_precision_score

    rows: list[dict] = []
    for i, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        y_score = y_pred[:, i]
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


def _avg_pr_rows(y_true: np.ndarray, y_pred: np.ndarray, classes: list, average: str) -> list[dict]:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=classes)
    if average == "micro":
        p, r, thr = precision_recall_curve(y_bin.ravel(), y_pred.ravel())
        ap = float(average_precision_score(y_bin, y_pred, average="micro"))
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
    grid = np.linspace(0.0, 1.0, 100)
    precisions = []
    for i in range(y_bin.shape[1]):
        p_i, r_i, _ = precision_recall_curve(y_bin[:, i], y_pred[:, i])
        order = np.argsort(r_i)
        precisions.append(np.interp(grid, r_i[order], p_i[order]))
    if average == "macro":
        weights = np.ones(len(classes)) / len(classes)
    else:
        total = max(int(y_bin.sum()), 1)
        weights = y_bin.sum(axis=0) / total
    precision_avg = (np.array(precisions).T * weights).sum(axis=1)
    ap = float(average_precision_score(y_bin, y_pred, average=average))
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


def _avg_roc_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    average: str,
    drop_intermediate: bool,
) -> list[dict]:
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=classes)
    if average == "micro":
        fpr, tpr, thr = roc_curve(
            y_bin.ravel(), y_pred.ravel(), drop_intermediate=drop_intermediate
        )
        auc = float(roc_auc_score(y_bin, y_pred, average="micro"))
        return [
            {"fpr": float(f), "tpr": float(t), "threshold": float(h), "class": "micro", "auc": auc}
            for f, t, h in zip(fpr, tpr, thr)
        ]
    grid = np.linspace(0.0, 1.0, 100)
    tprs = []
    for i in range(y_bin.shape[1]):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_pred[:, i])
        tprs.append(np.interp(grid, fpr_i, tpr_i))
    if average == "macro":
        weights = np.ones(len(classes)) / len(classes)
    else:
        total = max(int(y_bin.sum()), 1)
        weights = y_bin.sum(axis=0) / total
    tpr_avg = (np.array(tprs).T * weights).sum(axis=1)
    auc = float(roc_auc_score(y_bin, y_pred, average=average))
    return [
        {"fpr": float(f), "tpr": float(t), "threshold": float("nan"), "class": average, "auc": auc}
        for f, t in zip(grid, tpr_avg)
    ]
