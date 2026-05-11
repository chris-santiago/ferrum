"""ModelSource adapter — wraps a fitted estimator + data, exposes derived data.

Phase 10a: constructor, protocol detection, cache, .predictions(),
.probabilities(). Other methods land in 10b-10g; `ComparedModelSource`
in 10h.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from .deps import require_sklearn
from .stats import studentized_residual


_PROTOCOL_ATTRS: tuple[str, ...] = (
    "predict", "predict_proba", "decision_function", "transform",
    "fit_transform", "fit_predict", "score",
    "feature_importances_", "coef_", "explained_variance_ratio_",
    "cluster_centers_", "labels_", "classes_",
)


def _coerce_X_y(X: Any, y: Any) -> tuple[pl.DataFrame, pl.Series | None]:
    """Coerce X to polars.DataFrame and y to polars.Series (or None)."""
    if isinstance(X, pl.DataFrame):
        X_df = X
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        X_df = pl.from_numpy(X, schema=[f"f{i}" for i in range(X.shape[1])])
    else:
        # Route through ferrum's existing input-normalization to a pyarrow Table,
        # then convert to polars.
        from ferrum._coerce import to_arrow_table
        X_df = pl.from_arrow(to_arrow_table(X))
        if not isinstance(X_df, pl.DataFrame):
            raise TypeError(f"Could not coerce X to a polars DataFrame; got {type(X_df).__name__}")

    y_ser: pl.Series | None = None
    if y is not None:
        if isinstance(y, pl.Series):
            y_ser = y
        elif isinstance(y, np.ndarray):
            y_ser = pl.Series("y", y)
        elif isinstance(y, pl.DataFrame):
            if y.width != 1:
                raise ValueError(f"y DataFrame must have exactly 1 column; got {y.width}")
            y_ser = y.to_series()
        else:
            y_ser = pl.Series("y", list(y))
    return X_df, y_ser


class ModelSource:
    """Wraps a fitted estimator + dataset; exposes derived data as DataFrames.

    Constructor is sklearn-free: pure attribute introspection.
    Methods that need sklearn / shap / umap lazy-import on call.
    """

    def __init__(
        self,
        model: Any,
        X: Any,
        y: Any = None,
        *,
        feature_names: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        sample_weight: Any = None,
        random_state: int | None = None,
    ):
        self._model = model
        self._X, self._y = _coerce_X_y(X, y)
        self._feature_names: list[str] = (
            list(feature_names) if feature_names is not None
            else list(self._X.columns)
        )
        self._class_names: list[str] | None = (
            list(class_names) if class_names is not None else None
        )
        self._sample_weight = sample_weight
        self._random_state = random_state

        self._capabilities = frozenset(
            attr for attr in _PROTOCOL_ATTRS if hasattr(self._model, attr)
        )
        self._cache: dict[tuple, pl.DataFrame] = {}

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    def _require_capability(self, attr: str, method_name: str) -> None:
        if attr not in self._capabilities:
            raise AttributeError(
                f"ModelSource.{method_name}() requires the wrapped model to "
                f"implement '{attr}'. Got {type(self._model).__name__!r} which "
                f"does not."
            )

    def _cache_key(self, method: str, **kwargs) -> tuple:
        return (method, tuple(sorted(kwargs.items())))

    # --- 10a: predictions, probabilities ---------------------------------

    def predictions(self) -> pl.DataFrame:
        """Return y_true, y_pred, residual, studentized_residual."""
        key = self._cache_key("predictions")
        if key in self._cache:
            return self._cache[key]

        self._require_capability("predict", "predictions")
        X_np = self._X.to_numpy()
        y_pred = np.asarray(self._model.predict(X_np), dtype=np.float64)
        y_true = (
            np.asarray(self._y.to_numpy(), dtype=np.float64)
            if self._y is not None
            else np.full_like(y_pred, np.nan)
        )
        residual = y_true - y_pred

        # Studentized residual: linear-estimator path if model exposes coef_.
        if "coef_" in self._capabilities and self._y is not None:
            X_with_intercept = np.column_stack([np.ones(len(X_np)), X_np])
            stud = studentized_residual(y_true, y_pred, X_with_intercept)
        else:
            stud = studentized_residual(y_true, y_pred, X=None)

        df = pl.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": residual,
            "studentized_residual": stud,
        })
        self._cache[key] = df
        return df

    def probabilities(self) -> pl.DataFrame:
        """Return y_true + one column per class with predicted probability."""
        key = self._cache_key("probabilities")
        if key in self._cache:
            return self._cache[key]

        require_sklearn("probabilities")
        X_np = self._X.to_numpy()

        if "predict_proba" in self._capabilities:
            proba = np.asarray(self._model.predict_proba(X_np), dtype=np.float64)
        elif "decision_function" in self._capabilities:
            scores = np.asarray(self._model.decision_function(X_np), dtype=np.float64)
            if scores.ndim == 1:
                # Binary classifier — apply sigmoid.
                p1 = 1.0 / (1.0 + np.exp(-scores))
                proba = np.column_stack([1.0 - p1, p1])
            else:
                # Multiclass — softmax.
                exp = np.exp(scores - scores.max(axis=1, keepdims=True))
                proba = exp / exp.sum(axis=1, keepdims=True)
        else:
            raise AttributeError(
                "ModelSource.probabilities() requires the wrapped model to "
                "implement 'predict_proba' or 'decision_function'. Got "
                f"{type(self._model).__name__!r} which implements neither."
            )

        # Determine class labels.
        if self._class_names is not None:
            classes = list(self._class_names)
        elif hasattr(self._model, "classes_"):
            classes = [str(c) for c in self._model.classes_]
        else:
            classes = [f"class_{i}" for i in range(proba.shape[1])]

        data: dict[str, Any] = {}
        if self._y is not None:
            data["y_true"] = self._y.to_numpy()
        for i, c in enumerate(classes):
            data[f"proba_{c}"] = proba[:, i]
        df = pl.DataFrame(data)
        self._cache[key] = df
        return df

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
            "roc_curve", average=average, drop_intermediate=drop_intermediate,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("roc_curve")
        from sklearn.metrics import roc_curve as _sk_roc_curve, roc_auc_score

        if self._y is None:
            raise ValueError("ModelSource.roc_curve() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        y_true = np.asarray(self._y.to_numpy())
        classes = [c[len("proba_"):] for c in proba_cols]
        n_classes = len(classes)

        rows: list[dict] = []
        # Binary: a single curve is the only meaningful output. ``average`` is
        # accepted for API symmetry with the multiclass path but treated as a
        # no-op (there is only one class to average over).
        if n_classes == 2:
            y_score = proba_df[proba_cols[1]].to_numpy()
            fpr, tpr, thr = _sk_roc_curve(
                y_true, y_score, drop_intermediate=drop_intermediate,
            )
            auc = float(roc_auc_score(y_true, y_score))
            for f, t, h in zip(fpr, tpr, thr):
                rows.append({
                    "fpr": float(f), "tpr": float(t),
                    "threshold": float(h), "class": classes[1], "auc": auc,
                })
        else:
            for i, cls in enumerate(classes):
                y_bin = (
                    y_true == _coerce_class_label(cls, y_true.dtype)
                ).astype(int)
                y_score = proba_df[proba_cols[i]].to_numpy()
                fpr, tpr, thr = _sk_roc_curve(
                    y_bin, y_score, drop_intermediate=drop_intermediate,
                )
                try:
                    auc = float(roc_auc_score(y_bin, y_score))
                except ValueError:
                    auc = float("nan")
                for f, t, h in zip(fpr, tpr, thr):
                    rows.append({
                        "fpr": float(f), "tpr": float(t),
                        "threshold": float(h), "class": str(cls), "auc": auc,
                    })

            if average in ("micro", "macro", "weighted"):
                rows.extend(_compute_avg_roc(
                    y_true,
                    proba_df[proba_cols].to_numpy(),
                    classes, average, drop_intermediate,
                ))

        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def pr_curve(self, *, average: str | None = None) -> pl.DataFrame:
        """Precision-recall curve(s). One row per (class, threshold).

        For binary classifiers, returns a single curve on the positive
        (second) class. For multiclass, returns one-vs-rest curves per
        class. ``threshold`` is NaN at the final (recall=0) point per
        sklearn's convention.
        """
        if average is not None:
            # The spec exposes `average` for API symmetry with roc_curve but
            # multiclass averaged PR variants aren't part of Phase 10b.
            raise NotImplementedError(
                "ModelSource.pr_curve(average=...) lands in a later phase; "
                "use average=None for per-class curves."
            )
        key = self._cache_key("pr_curve", average=average)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("pr_curve")
        from sklearn.metrics import precision_recall_curve, average_precision_score

        if self._y is None:
            raise ValueError("ModelSource.pr_curve() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        y_true = np.asarray(self._y.to_numpy())
        classes = [c[len("proba_"):] for c in proba_cols]
        n_classes = len(classes)

        rows: list[dict] = []
        if n_classes == 2:
            y_score = proba_df[proba_cols[1]].to_numpy()
            p, r, thr = precision_recall_curve(y_true, y_score)
            ap = float(average_precision_score(y_true, y_score))
            thresholds_padded = np.concatenate([thr, [float("nan")]])
            for pi, ri, ti in zip(p, r, thresholds_padded):
                rows.append({
                    "precision": float(pi), "recall": float(ri),
                    "threshold": float(ti), "class": classes[1], "ap": ap,
                })
        else:
            for i, cls in enumerate(classes):
                y_bin = (
                    y_true == _coerce_class_label(cls, y_true.dtype)
                ).astype(int)
                y_score = proba_df[proba_cols[i]].to_numpy()
                p, r, thr = precision_recall_curve(y_bin, y_score)
                try:
                    ap = float(average_precision_score(y_bin, y_score))
                except ValueError:
                    ap = float("nan")
                thresholds_padded = np.concatenate([thr, [float("nan")]])
                for pi, ri, ti in zip(p, r, thresholds_padded):
                    rows.append({
                        "precision": float(pi), "recall": float(ri),
                        "threshold": float(ti), "class": str(cls), "ap": ap,
                    })

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
            "calibration_curve", n_bins=n_bins, strategy=strategy,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("calibration_curve")
        from sklearn.calibration import calibration_curve as _ccurve

        if self._y is None:
            raise ValueError(
                "ModelSource.calibration_curve() requires y to be provided."
            )
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if len(proba_cols) != 2:
            raise ValueError(
                "calibration_curve() is binary-classifier only; "
                f"got {len(proba_cols)} classes."
            )
        y_true = np.asarray(self._y.to_numpy())
        y_score = proba_df[proba_cols[1]].to_numpy()

        frac_pos, mean_pred = _ccurve(
            y_true, y_score, n_bins=n_bins, strategy=strategy,
        )

        if strategy == "uniform":
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        elif strategy == "quantile":
            edges = np.quantile(y_score, np.linspace(0.0, 1.0, n_bins + 1))
        else:
            raise ValueError(
                f"calibration_curve(strategy={strategy!r}) not supported; "
                "use 'uniform' or 'quantile'."
            )
        bin_idx = np.clip(np.digitize(y_score, edges[1:-1]), 0, n_bins - 1)
        counts_all = np.bincount(bin_idx, minlength=n_bins)
        centers = edges[:-1] + np.diff(edges) / 2.0
        used_bins = np.array([
            int(np.argmin(np.abs(centers - mp))) for mp in mean_pred
        ], dtype=int)
        counts = counts_all[used_bins] if used_bins.size else np.empty(0, dtype=int)

        df = pl.DataFrame({
            "mean_predicted": [float(x) for x in mean_pred],
            "fraction_positive": [float(x) for x in frac_pos],
            "count": [int(x) for x in counts],
        })
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
            raise ValueError(
                "ModelSource.cumulative_gain() requires y to be provided."
            )
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        y_true = np.asarray(self._y.to_numpy())
        classes = [c[len("proba_"):] for c in proba_cols]
        n = len(y_true)

        rows: list[dict] = []
        for i, cls in enumerate(classes):
            y_bin = (
                y_true == _coerce_class_label(cls, y_true.dtype)
            ).astype(int)
            order = np.argsort(-proba_df[proba_cols[i]].to_numpy())
            cum_pos = np.cumsum(y_bin[order])
            total_pos = max(int(cum_pos[-1]), 1) if n else 1
            pct_pop = np.arange(1, n + 1) / max(n, 1)
            gain = cum_pos / total_pos
            xs = np.concatenate([[0.0], pct_pop])
            ys = np.concatenate([[0.0], gain])
            for pp, g in zip(xs, ys):
                rows.append({
                    "percent_population": float(pp),
                    "gain": float(g),
                    "class": str(cls),
                })

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
            raise ValueError(
                "ModelSource.lift_curve() requires y to be provided."
            )
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        y_true = np.asarray(self._y.to_numpy())
        classes = [c[len("proba_"):] for c in proba_cols]
        n = len(y_true)

        rows: list[dict] = []
        for i, cls in enumerate(classes):
            y_bin = (
                y_true == _coerce_class_label(cls, y_true.dtype)
            ).astype(int)
            base_rate = float(y_bin.mean()) if n else 0.0
            if base_rate == 0.0:
                continue
            order = np.argsort(-proba_df[proba_cols[i]].to_numpy())
            cum_pos = np.cumsum(y_bin[order])
            denom = np.arange(1, n + 1)
            cum_rate = cum_pos / denom
            lift = cum_rate / base_rate
            pct_pop = denom / n
            for pp, lv in zip(pct_pop, lift):
                rows.append({
                    "percent_population": float(pp),
                    "lift": float(lv),
                    "class": str(cls),
                })

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
            "discrimination_threshold", n_thresholds=n_thresholds, cv=cv,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("discrimination_threshold")

        if self._y is None:
            raise ValueError(
                "ModelSource.discrimination_threshold() requires y to be provided."
            )
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if len(proba_cols) != 2:
            raise ValueError(
                "discrimination_threshold() is binary-classifier only; "
                f"got {len(proba_cols)} classes."
            )
        y_true = np.asarray(self._y.to_numpy())
        positive_class = _coerce_class_label(
            proba_cols[1][len("proba_"):], y_true.dtype,
        )
        thresholds = np.linspace(0.0, 1.0, n_thresholds)

        if cv is None:
            y_score = proba_df[proba_cols[1]].to_numpy()
            df = self._sweep_thresholds(
                y_true, y_score, thresholds, positive_class,
            )
        else:
            from sklearn.base import clone
            from sklearn.model_selection import KFold
            X_np = self._X.to_numpy()
            splitter = (
                cv if hasattr(cv, "split")
                else KFold(
                    n_splits=int(cv), shuffle=True,
                    random_state=self._random_state or 0,
                )
            )
            fold_dfs: list[pl.DataFrame] = []
            for tr, te in splitter.split(X_np):
                m = clone(self._model).fit(X_np[tr], y_true[tr])
                if hasattr(m, "predict_proba"):
                    s = np.asarray(m.predict_proba(X_np[te]), dtype=np.float64)[:, 1]
                elif hasattr(m, "decision_function"):
                    raw = np.asarray(
                        m.decision_function(X_np[te]), dtype=np.float64,
                    )
                    s = 1.0 / (1.0 + np.exp(-raw))
                else:
                    raise AttributeError(
                        "discrimination_threshold(cv=...) requires the wrapped "
                        "model to implement 'predict_proba' or 'decision_function'."
                    )
                fold_dfs.append(self._sweep_thresholds(
                    y_true[te], s, thresholds, positive_class,
                ))
            df = (
                pl.concat(fold_dfs, how="vertical")
                .group_by("threshold")
                .agg([
                    pl.col("precision").mean(),
                    pl.col("recall").mean(),
                    pl.col("f1").mean(),
                    pl.col("queue_rate").mean(),
                ])
                .sort("threshold")
            )

        self._cache[key] = df
        return df

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
            raise ValueError(
                "ModelSource.confusion_matrix() requires y to be provided."
            )
        y_true = np.asarray(self._y.to_numpy())
        X_np = self._X.to_numpy()
        y_pred = np.asarray(self._model.predict(X_np))

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
                rows.append({
                    "actual": str(a),
                    "predicted": str(p),
                    "value": val,
                    "value_fmt": fmt,
                })
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    # --- 10d: feature importance ----------------------------------------

    def importances(
        self,
        *,
        method: str = "builtin",
        n_repeats: int = 30,
        scoring: Any = None,
        random_state: int | None = None,
    ) -> pl.DataFrame:
        """Feature importance per feature, sorted by descending |importance|.

        ``method="builtin"`` reads the wrapped model's ``feature_importances_``
        (tree-based estimators) or ``coef_`` (linear estimators, averaged
        absolute value across classes for multi-output linears). ``std`` is
        zero in this path — sklearn's built-in attribute exposes no
        per-feature variance.

        ``method="permutation"`` calls sklearn's
        ``permutation_importance`` with ``n_repeats``/``scoring`` and
        populates ``std`` with the per-feature standard deviation across
        repeats.
        """
        rs = random_state if random_state is not None else self._random_state
        key = self._cache_key(
            "importances",
            kind=method,
            n_repeats=n_repeats,
            scoring=str(scoring) if scoring is not None else None,
            random_state=rs,
        )
        if key in self._cache:
            return self._cache[key]

        if method == "builtin":
            if "feature_importances_" in self._capabilities:
                imp = np.asarray(
                    self._model.feature_importances_, dtype=np.float64,
                )
            elif "coef_" in self._capabilities:
                coef = np.asarray(self._model.coef_, dtype=np.float64)
                imp = (
                    np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
                )
            else:
                raise AttributeError(
                    "ModelSource.importances(method='builtin') requires the "
                    "wrapped model to expose 'feature_importances_' or "
                    f"'coef_'. Got {type(self._model).__name__!r} which "
                    "exposes neither."
                )
            std = np.zeros_like(imp)
        elif method == "permutation":
            require_sklearn("importances(permutation)")
            from sklearn.inspection import permutation_importance

            if self._y is None:
                raise ValueError(
                    "ModelSource.importances(method='permutation') requires "
                    "y to be provided."
                )
            X_np = self._X.to_numpy()
            y_np = np.asarray(self._y.to_numpy())
            result = permutation_importance(
                self._model, X_np, y_np,
                n_repeats=n_repeats,
                scoring=scoring,
                random_state=rs if rs is not None else 0,
            )
            imp = np.asarray(result.importances_mean, dtype=np.float64)
            std = np.asarray(result.importances_std, dtype=np.float64)
        else:
            raise ValueError(
                f"ModelSource.importances(method={method!r}) — expected "
                "'builtin' or 'permutation'."
            )

        order = np.argsort(-np.abs(imp))
        rows = [
            {
                "feature": str(self._feature_names[i]),
                "importance": float(imp[i]),
                "std": float(std[i]),
                "rank": int(r),
            }
            for r, i in enumerate(order, start=1)
        ]
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
                y_true_bin, y_pred, average="binary", zero_division=0,
            )
            queue_rate = float((y_score >= t).mean()) if y_score.size else 0.0
            rows.append({
                "threshold": float(t),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "queue_rate": queue_rate,
            })
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


def _compute_avg_roc(y_true, y_score_matrix, classes, average, drop_intermediate):
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import label_binarize

    coerced_classes = [_coerce_class_label(c, y_true.dtype) for c in classes]
    y_bin = label_binarize(y_true, classes=coerced_classes)
    if average == "micro":
        fpr, tpr, thr = roc_curve(
            y_bin.ravel(), y_score_matrix.ravel(),
            drop_intermediate=drop_intermediate,
        )
        auc = float(roc_auc_score(y_bin, y_score_matrix, average="micro"))
        return [
            {"fpr": float(f), "tpr": float(t), "threshold": float(h),
             "class": "micro", "auc": auc}
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
        {"fpr": float(f), "tpr": float(t), "threshold": float("nan"),
         "class": average, "auc": auc}
        for f, t in zip(grid, tpr_avg)
    ]
