"""ModelSource adapter — wraps a fitted estimator + data, exposes derived data.

Phase 10a: constructor, protocol detection, cache, .predictions(),
.probabilities(). Other methods land in 10b-10g; `ComparedModelSource`
in 10h.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from .deps import require_shap, require_sklearn, require_umap
from .stats import cooks_distance, studentized_residual


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

    @classmethod
    def compare(
        cls,
        models: dict[str, Any],
        X: Any,
        y: Any = None,
        **kwargs: Any,
    ) -> "ComparedModelSource":
        """Build a ``ComparedModelSource`` over one ``ModelSource`` per model.

        Each value in ``models`` is wrapped in its own ``ModelSource`` with the
        shared ``X`` and ``y``. The returned ``ComparedModelSource`` proxies
        every derived-data method through all wrapped sources and stamps the
        model name as a ``model`` column on the concatenated output, so
        downstream chart builders can route ``color="model"``.
        """
        sources = {
            name: cls(model, X, y, **kwargs) for name, model in models.items()
        }
        return ComparedModelSource(sources)

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

        # Studentized residual + Cook's distance: leverage-aware variants
        # require the design matrix. Linear-estimator path uses `coef_` as
        # the gate; non-linear estimators fall back to the no-X studentized
        # residual and report NaN for Cook's D (it's undefined without a
        # hat matrix).
        if "coef_" in self._capabilities and self._y is not None:
            X_with_intercept = np.column_stack([np.ones(len(X_np)), X_np])
            stud = studentized_residual(y_true, y_pred, X_with_intercept)
            cooks = cooks_distance(y_true, y_pred, X_with_intercept)
        else:
            stud = studentized_residual(y_true, y_pred, X=None)
            cooks = np.full_like(y_pred, np.nan)

        df = pl.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": residual,
            "studentized_residual": stud,
            "cooks_distance": cooks,
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

    def shap_values(
        self,
        *,
        background: Any = None,
        max_evals: int = 500,
    ) -> pl.DataFrame:
        """Long-form SHAP values per (sample, feature).

        Returns a DataFrame with ``sample_id``, ``feature``, ``shap_value``,
        ``feature_value``, ``feature_value_normalized``. The explainer is
        auto-picked by model capability:

        - ``coef_``: ``shap.LinearExplainer`` (deterministic, fast).
        - ``feature_importances_``: ``shap.TreeExplainer`` (deterministic
          for tree ensembles).
        - otherwise: ``shap.KernelExplainer`` (model-agnostic; uses the
          first ``min(50, N)`` rows of X as the background unless an
          explicit ``background`` array is passed).

        For multi-class classifiers the underlying ``shap_values`` returns
        a list of arrays — Phase 10d uses class 1 for binary and class 0
        otherwise. Multi-class SHAP overlay is a Phase 10h follow-up.
        """
        key = self._cache_key(
            "shap_values",
            background=str(background)[:64],
            max_evals=max_evals,
        )
        if key in self._cache:
            return self._cache[key]
        shap = require_shap("shap_values")

        X_np = self._X.to_numpy()
        if "coef_" in self._capabilities:
            explainer = shap.LinearExplainer(self._model, X_np)
        elif "feature_importances_" in self._capabilities:
            explainer = shap.TreeExplainer(self._model)
        else:
            bg = (
                background
                if background is not None
                else X_np[: min(50, len(X_np))]
            )
            explainer = shap.KernelExplainer(self._model.predict, bg)

        sv = explainer.shap_values(X_np)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]
        sv = np.asarray(sv, dtype=np.float64)
        # Some shap versions return a 3D array (n_samples, n_features, n_classes)
        # for multi-class TreeExplainer; collapse to the positive-class slice.
        if sv.ndim == 3:
            sv = sv[..., 1] if sv.shape[2] == 2 else sv[..., 0]

        f_mean = X_np.mean(axis=0)
        f_std_raw = X_np.std(axis=0)
        f_std = np.where(f_std_raw > 0, f_std_raw, 1.0)
        f_norm = (X_np - f_mean) / f_std

        n_samples = X_np.shape[0]
        n_features = len(self._feature_names)
        rows: list[dict] = [
            {
                "sample_id": int(s),
                "feature": str(self._feature_names[f]),
                "shap_value": float(sv[s, f]),
                "feature_value": float(X_np[s, f]),
                "feature_value_normalized": float(f_norm[s, f]),
            }
            for s in range(n_samples)
            for f in range(n_features)
        ]
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def partial_dependence(
        self,
        features: list[str | int],
        *,
        grid_resolution: int = 100,
        kind: str = "average",
    ) -> pl.DataFrame:
        """Partial dependence per feature.

        ``kind="average"`` (default) returns the marginal PD curve per
        feature with ``sample_id = -1`` (one row per grid point per
        feature).

        ``kind="individual"`` returns per-sample ICE curves: one row per
        ``(feature, sample_id, grid_point)`` triple with ``sample_id`` in
        ``[0, n_samples)``. Chart builders pair this with the ``detail``
        encoding channel on ``sample_id`` to render one polyline per
        sample.

        ``kind="both"`` returns the union of the two: ICE rows plus
        average rows (``sample_id = -1``), so a downstream chart can
        overlay both layers on the same DataFrame.
        """
        if kind not in ("average", "individual", "both"):
            raise ValueError(
                f"ModelSource.partial_dependence(kind={kind!r}) — expected "
                "'average', 'individual', or 'both'."
            )
        key = self._cache_key(
            "partial_dependence",
            features=tuple(features),
            grid_resolution=grid_resolution,
            kind=kind,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("partial_dependence")
        from sklearn.inspection import partial_dependence as _sk_pd

        feature_idxs = [
            self._feature_names.index(f) if isinstance(f, str) else f
            for f in features
        ]
        X_np = self._X.to_numpy()
        rows: list[dict] = []
        for f_idx in feature_idxs:
            fname = self._feature_names[f_idx]
            # sklearn returns "individual" of shape (n_outputs, n_samples,
            # n_grid) when kind in ("individual", "both"); we use a single
            # output (binary class index 1 / regression scalar) by indexing
            # [0]. For "average" the shape is (n_outputs, n_grid).
            r = _sk_pd(
                self._model,
                X_np,
                features=[f_idx],
                grid_resolution=grid_resolution,
                kind=kind,
            )
            grid = r["grid_values"][0]
            if kind in ("average", "both"):
                avg = np.asarray(r["average"])[0]
                for v, p in zip(grid, avg):
                    rows.append({
                        "feature": str(fname),
                        "feature_value": float(v),
                        "pd_value": float(p),
                        "sample_id": -1,
                    })
            if kind in ("individual", "both"):
                individual = np.asarray(r["individual"])[0]
                n_samples, n_grid = individual.shape
                for s in range(n_samples):
                    for v, p in zip(grid, individual[s]):
                        rows.append({
                            "feature": str(fname),
                            "feature_value": float(v),
                            "pd_value": float(p),
                            "sample_id": int(s),
                        })
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    # --- 10e: model selection / CV curves --------------------------------

    def learning_curve(
        self,
        *,
        cv: int = 5,
        scoring: Any = None,
        train_sizes: Any = None,
    ) -> pl.DataFrame:
        """Learning curve: score per (train_size, fold, split).

        Returns long-form rows — one per (train_size, fold, split). Each
        row carries the per-fold ``score`` plus the per-(train_size, split)
        aggregates ``mean_score``, ``std_score``, ``lower``, ``upper`` (95%
        CI on the mean). Chart builders dedupe by (train_size, split) to
        render a ribbon + line; the per-fold rows enable per-fold strip
        overlays if a future caller wants them.
        """
        if self._y is None:
            raise ValueError(
                "ModelSource.learning_curve() requires y to be provided."
            )
        key = self._cache_key(
            "learning_curve",
            cv=cv,
            scoring=str(scoring) if scoring is not None else None,
            train_sizes=str(train_sizes) if train_sizes is not None else None,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("learning_curve")
        from sklearn.model_selection import learning_curve as _lc

        X_np = self._X.to_numpy()
        y_np = np.asarray(self._y.to_numpy())
        sizes = train_sizes if train_sizes is not None else np.linspace(0.1, 1.0, 5)
        ts, tr_scores, te_scores = _lc(
            self._model, X_np, y_np,
            train_sizes=sizes, cv=cv, scoring=scoring,
            random_state=self._random_state if self._random_state is not None else 0,
            shuffle=True,
        )
        rows: list[dict] = []
        for i, t in enumerate(ts):
            for split_name, arr in (("train", tr_scores[i]), ("test", te_scores[i])):
                mean = float(arr.mean())
                std = float(arr.std())
                n = len(arr)
                ci = 1.96 * std / np.sqrt(n) if n > 0 else 0.0
                for s in arr:
                    rows.append({
                        "train_size": int(t),
                        "split": split_name,
                        "score": float(s),
                        "mean_score": mean,
                        "std_score": std,
                        "lower": mean - ci,
                        "upper": mean + ci,
                    })
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def validation_curve(
        self,
        param: str,
        values: Any,
        *,
        cv: int = 5,
        scoring: Any = None,
    ) -> pl.DataFrame:
        """Validation curve: score per (param_value, fold, split).

        Same shape as ``learning_curve`` but parameterized by an
        estimator hyperparameter sweep. ``param`` is the kwarg name on
        the wrapped estimator (e.g. ``"alpha"`` for ``Ridge``).
        """
        if self._y is None:
            raise ValueError(
                "ModelSource.validation_curve() requires y to be provided."
            )
        vals = np.asarray(list(values), dtype=np.float64)
        key = self._cache_key(
            "validation_curve",
            param=param,
            values=tuple(float(v) for v in vals),
            cv=cv,
            scoring=str(scoring) if scoring is not None else None,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("validation_curve")
        from sklearn.model_selection import validation_curve as _vc

        X_np = self._X.to_numpy()
        y_np = np.asarray(self._y.to_numpy())
        tr, te = _vc(
            self._model, X_np, y_np,
            param_name=param, param_range=vals,
            cv=cv, scoring=scoring,
        )
        rows: list[dict] = []
        for i, v in enumerate(vals):
            for split_name, arr in (("train", tr[i]), ("test", te[i])):
                mean = float(arr.mean())
                std = float(arr.std())
                n = len(arr)
                ci = 1.96 * std / np.sqrt(n) if n > 0 else 0.0
                for s in arr:
                    rows.append({
                        "param_value": float(v),
                        "split": split_name,
                        "score": float(s),
                        "mean_score": mean,
                        "std_score": std,
                        "lower": mean - ci,
                        "upper": mean + ci,
                    })
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def cv_scores(
        self,
        *,
        cv: int = 5,
        scoring: Any = None,
    ) -> pl.DataFrame:
        """Per-fold cross-validation scores.

        Returns one row per (fold, split) — train and test scores for each
        cross-validation fold. Chart builders use this for boxplot / bar /
        strip distributions across folds.
        """
        if self._y is None:
            raise ValueError(
                "ModelSource.cv_scores() requires y to be provided."
            )
        key = self._cache_key(
            "cv_scores",
            cv=cv,
            scoring=str(scoring) if scoring is not None else None,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("cv_scores")
        from sklearn.model_selection import cross_validate

        X_np = self._X.to_numpy()
        y_np = np.asarray(self._y.to_numpy())
        result = cross_validate(
            self._model, X_np, y_np,
            cv=cv, scoring=scoring, return_train_score=True,
        )
        rows: list[dict] = []
        for fold, s in enumerate(result["train_score"]):
            rows.append({"fold": int(fold), "split": "train", "score": float(s)})
        for fold, s in enumerate(result["test_score"]):
            rows.append({"fold": int(fold), "split": "test", "score": float(s)})
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def alpha_selection(
        self,
        alphas: Any,
        *,
        cv: int = 5,
        scoring: Any = None,
    ) -> pl.DataFrame:
        """Regularization-strength sweep for linear models.

        Returns one row per (alpha, fold) — the per-fold test score on the
        held-out split — plus per-alpha ``mean_score`` / ``std_score``
        aggregates. Chart builders dedupe by alpha to render a single
        line, and use ``argmax(mean_score)`` to mark the best alpha.
        """
        if self._y is None:
            raise ValueError(
                "ModelSource.alpha_selection() requires y to be provided."
            )
        vals = np.asarray(list(alphas), dtype=np.float64)
        key = self._cache_key(
            "alpha_selection",
            alphas=tuple(float(v) for v in vals),
            cv=cv,
            scoring=str(scoring) if scoring is not None else None,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("alpha_selection")
        from sklearn.model_selection import validation_curve as _vc

        X_np = self._X.to_numpy()
        y_np = np.asarray(self._y.to_numpy())
        _, te = _vc(
            self._model, X_np, y_np,
            param_name="alpha", param_range=vals,
            cv=cv, scoring=scoring,
        )
        rows: list[dict] = []
        for i, a in enumerate(vals):
            mean = float(te[i].mean())
            std = float(te[i].std())
            for fold_idx, s in enumerate(te[i]):
                rows.append({
                    "alpha": float(a),
                    "fold": int(fold_idx),
                    "score": float(s),
                    "mean_score": mean,
                    "std_score": std,
                })
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    # --- 10f: clustering / manifold --------------------------------------

    def silhouette(self, *, k: int | None = None) -> pl.DataFrame:
        """Per-sample silhouette values, sorted within cluster descending.

        Returns one row per sample with columns ``sample_id`` (original X
        index), ``y_position`` (sequential 0..n-1 stack order — used by
        ``mark_silhouette`` to render bars in a tightly-packed Rousseeuw
        layout), ``cluster``, and ``silhouette_value``.

        ``k`` is informational; if provided, the result is filtered to
        clusters in ``range(k)``.
        """
        key = self._cache_key("silhouette", k=k)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("silhouette")
        from sklearn.metrics import silhouette_samples

        X_np = self._X.to_numpy()
        if "labels_" in self._capabilities:
            labels = np.asarray(self._model.labels_)
        elif "predict" in self._capabilities:
            labels = np.asarray(self._model.predict(X_np))
        else:
            raise AttributeError(
                "ModelSource.silhouette() requires the wrapped model to "
                "expose 'labels_' or 'predict'."
            )
        sv = silhouette_samples(X_np, labels)
        clusters = sorted(set(int(c) for c in labels.tolist()))
        if k is not None:
            clusters = [c for c in clusters if c < int(k)]
        rows: list[dict] = []
        y_pos = 0
        for c in clusters:
            mask = labels == c
            idxs = np.where(mask)[0]
            vals = sv[mask]
            order = np.argsort(-vals)
            for i, val in zip(idxs[order], vals[order]):
                rows.append({
                    "sample_id": int(i),
                    "y_position": int(y_pos),
                    # cluster is rendered as a categorical color channel;
                    # serialize as Utf8 to match every other 10b/10c
                    # categorical diagnostic schema (ROC class, confusion
                    # actual/predicted) and avoid the renderer's
                    # continuous-scale fallback for low-cardinality ints.
                    "cluster": str(int(c)),
                    "silhouette_value": float(val),
                })
                y_pos += 1
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def pca_variance(self, *, n_components: int | None = None) -> pl.DataFrame:
        """Explained-variance ratio per principal component plus the
        cumulative running sum. Requires the wrapped model to expose
        ``explained_variance_ratio_`` (e.g. ``sklearn.decomposition.PCA``,
        ``TruncatedSVD``).
        """
        key = self._cache_key("pca_variance", n_components=n_components)
        if key in self._cache:
            return self._cache[key]
        if "explained_variance_ratio_" not in self._capabilities:
            raise AttributeError(
                "ModelSource.pca_variance() requires the wrapped model to "
                "expose 'explained_variance_ratio_' (e.g. sklearn PCA, "
                "TruncatedSVD)."
            )
        evr = np.asarray(self._model.explained_variance_ratio_, dtype=np.float64)
        if n_components is not None:
            evr = evr[: int(n_components)]
        cum = np.cumsum(evr)
        df = pl.DataFrame({
            "component": list(range(1, len(evr) + 1)),
            "explained_variance_ratio": [float(x) for x in evr],
            "cumulative_variance_ratio": [float(x) for x in cum],
        })
        self._cache[key] = df
        return df

    def embeddings(
        self,
        *,
        method: str = "umap",
        n_components: int = 2,
        **method_kwargs: Any,
    ) -> pl.DataFrame:
        """Low-dimensional embedding of X via UMAP / t-SNE / PCA.

        Returns ``dim_0`` … ``dim_{n_components-1}`` plus a ``label`` column
        (``y`` when provided, else zeros — used to color the scatter).
        ``random_state`` is taken from the source's ``random_state``.
        """
        key = self._cache_key(
            "embeddings",
            embed_method=method,
            n_components=n_components,
            kwargs=tuple(sorted(method_kwargs.items())),
        )
        if key in self._cache:
            return self._cache[key]
        X_np = self._X.to_numpy()
        seed = self._random_state if self._random_state is not None else 0
        if method == "umap":
            umap = require_umap("embeddings")
            reducer = umap.UMAP(
                n_components=n_components, random_state=seed, **method_kwargs,
            )
            emb = reducer.fit_transform(X_np)
        elif method == "tsne":
            require_sklearn("embeddings(tsne)")
            from sklearn.manifold import TSNE

            emb = TSNE(
                n_components=n_components, random_state=seed, **method_kwargs,
            ).fit_transform(X_np)
        elif method == "pca":
            require_sklearn("embeddings(pca)")
            from sklearn.decomposition import PCA

            emb = PCA(
                n_components=n_components, random_state=seed, **method_kwargs,
            ).fit_transform(X_np)
        else:
            raise ValueError(
                f"ModelSource.embeddings(method={method!r}) — expected "
                "'umap', 'tsne', or 'pca'."
            )
        emb = np.asarray(emb, dtype=np.float64)
        if self._y is not None:
            label_arr = np.asarray(self._y.to_numpy())
        else:
            label_arr = np.zeros(emb.shape[0])
        data: dict[str, Any] = {
            f"dim_{i}": [float(v) for v in emb[:, i]]
            for i in range(emb.shape[1])
        }
        data["label"] = label_arr.tolist()
        df = pl.DataFrame(data)
        self._cache[key] = df
        return df

    def intercluster_distance(
        self,
        k: int,
        *,
        method: str = "mds",
    ) -> pl.DataFrame:
        """2D embedding of cluster centers + cluster size.

        Returns one row per cluster with ``cluster`` (Int64), ``x`` / ``y``
        (Float64, the 2D embedded coordinate), and ``size`` (Int64, sample
        count). Requires the wrapped model to expose ``cluster_centers_``.
        """
        key = self._cache_key(
            "intercluster_distance", k=int(k), embed_method=method,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("intercluster_distance")
        if "cluster_centers_" not in self._capabilities:
            raise AttributeError(
                "ModelSource.intercluster_distance() requires the wrapped "
                "model to expose 'cluster_centers_'."
            )
        centers = np.asarray(self._model.cluster_centers_, dtype=np.float64)
        if centers.shape[0] < int(k):
            k = centers.shape[0]
        seed = self._random_state if self._random_state is not None else 0
        if method == "mds":
            from sklearn.manifold import MDS

            xy = MDS(
                n_components=2, random_state=seed, normalized_stress="auto",
            ).fit_transform(centers[:k])
        elif method == "tsne":
            from sklearn.manifold import TSNE

            xy = TSNE(
                n_components=2, random_state=seed,
                perplexity=max(1, min(5, int(k) - 1)),
            ).fit_transform(centers[:k])
        else:
            raise ValueError(
                f"ModelSource.intercluster_distance(method={method!r}) — "
                "expected 'mds' or 'tsne'."
            )
        if "labels_" in self._capabilities:
            labels = np.asarray(self._model.labels_)
            sizes = np.bincount(labels.astype(int), minlength=int(k))[: int(k)]
        else:
            sizes = np.ones(int(k), dtype=int)
        df = pl.DataFrame({
            # cluster routes through a categorical color scale (one
            # color per cluster id); serialize as Utf8 — same Int64 →
            # continuous-scale gotcha as silhouette.cluster.
            "cluster": [str(i) for i in range(int(k))],
            "x": [float(v) for v in xy[:, 0]],
            "y": [float(v) for v in xy[:, 1]],
            "size": [int(s) for s in sizes],
        })
        self._cache[key] = df
        return df

    # ---- Phase 10g: feature ranking ----

    def rank1d(self, *, algorithm: str = "shapiro") -> pl.DataFrame:
        """Univariate feature ranking.

        ``algorithm`` in ``{"shapiro", "variance", "covariance"}``. The
        Shapiro-Wilk and variance algorithms operate on X alone;
        ``"covariance"`` ranks features by absolute sample covariance with
        ``y`` and therefore requires ``y`` to be present.

        Output schema (``SCHEMA_RANK1D``): ``feature: Utf8``,
        ``score: Float64``, ``rank: Int64``. Rows are pre-sorted by descending
        score so ``rank=1`` is always the top feature.
        """
        from .stats import (
            covariance_rank,
            rank1d_compute,
        )

        key = self._cache_key("rank1d", algorithm=algorithm)
        if key in self._cache:
            return self._cache[key]
        if algorithm == "covariance":
            if self._y is None:
                raise ValueError(
                    "ModelSource.rank1d(algorithm='covariance') requires y."
                )
            X_np = np.asarray(self._X.to_numpy(), dtype=np.float64)
            y_np = np.asarray(self._y.to_numpy(), dtype=np.float64)
            scores = covariance_rank(X_np, y_np)
            order = np.argsort(-scores, kind="mergesort")
            df = pl.DataFrame({
                "feature": [str(self._feature_names[int(i)]) for i in order],
                "score": [float(scores[int(i)]) for i in order],
                "rank": list(range(1, len(order) + 1)),
            })
        else:
            df = rank1d_compute(self._X, algorithm=algorithm)
        self._cache[key] = df
        return df

    def rank2d(self, *, algorithm: str = "pearson") -> pl.DataFrame:
        """Pairwise feature ranking — long-form correlation matrix.

        ``algorithm`` in ``{"pearson", "spearman", "kendall", "covariance"}``.
        ``"kendall"`` routes through ``ferrum._core.kendall_tau_b``
        (Knight's O(n log n)).

        Output schema (``SCHEMA_RANK2D``): ``feature_x: Utf8``,
        ``feature_y: Utf8``, ``correlation: Float64`` — one row per
        ordered pair of features, p × p rows total.
        """
        from .stats import rank2d_compute

        key = self._cache_key("rank2d", algorithm=algorithm)
        if key in self._cache:
            return self._cache[key]
        df = rank2d_compute(self._X, algorithm=algorithm)
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


# ---- Phase 10h: ComparedModelSource ----


# Tuple of method names that `ComparedModelSource.__getattr__` proxies to
# the underlying ``ModelSource`` instances. Mirrors every Phase 10
# derived-data method on ``ModelSource``. When adding a new diagnostic
# method, append it here so the multi-model dispatch picks it up.
_COMPARED_METHODS: frozenset[str] = frozenset({
    "predictions",
    "probabilities",
    "roc_curve",
    "pr_curve",
    "calibration_curve",
    "cumulative_gain",
    "lift_curve",
    "discrimination_threshold",
    "confusion_matrix",
    "importances",
    "shap_values",
    "partial_dependence",
    "learning_curve",
    "validation_curve",
    "cv_scores",
    "alpha_selection",
    "silhouette",
    "pca_variance",
    "embeddings",
    "intercluster_distance",
    "rank1d",
    "rank2d",
})


class ComparedModelSource:
    """Multi-model wrapper exposing the same surface as ``ModelSource``.

    Every derived-data method is proxied through each underlying
    ``ModelSource`` and the per-model outputs are concatenated with a
    ``model: Utf8`` column stamped on each frame, so downstream chart
    builders can route ``color="model"`` to render one curve per model.

    ``_X`` and ``_y`` resolve to the first source's data (every wrapped
    source shares ``X`` / ``y`` by construction in ``ModelSource.compare``,
    so any one will do); accessing ``_model`` raises since there is no
    single estimator. ``model_names`` reports the configured ordering.
    """

    __slots__ = ("_sources",)

    def __init__(self, sources: dict[str, ModelSource]):
        if not sources:
            raise ValueError("ComparedModelSource requires at least one source.")
        self._sources = dict(sources)

    @property
    def model_names(self) -> list[str]:
        return list(self._sources.keys())

    def _dispatch(self, method: str, *args: Any, **kwargs: Any) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for name, src in self._sources.items():
            df = getattr(src, method)(*args, **kwargs)
            frames.append(df.with_columns(pl.lit(name).alias("model")))
        return pl.concat(frames, how="vertical_relaxed")

    def __getattr__(self, name: str) -> Any:
        # __slots__ makes attribute access strict; the runtime falls through
        # to __getattr__ only for unknown names. We route a frozen list of
        # ModelSource methods through `_dispatch`, expose `_X`/`_y` from the
        # first wrapped source (chart builders sometimes need them), and
        # explicitly forbid `_model` access since the answer would be
        # nonsensical for a multi-model wrapper.
        if name in _COMPARED_METHODS:
            method = name
            return lambda *args, **kwargs: self._dispatch(method, *args, **kwargs)
        if name in ("_X", "_y", "_feature_names", "_class_names"):
            return getattr(next(iter(self._sources.values())), name)
        if name == "_model":
            raise AttributeError(
                "ComparedModelSource has no single _model. Iterate "
                "ComparedModelSource._sources.values() to access each "
                "wrapped ModelSource's model."
            )
        raise AttributeError(
            f"ComparedModelSource has no attribute {name!r}. "
            f"Methods routed through .compare: {sorted(_COMPARED_METHODS)}"
        )

    def __repr__(self) -> str:
        return f"ComparedModelSource({list(self._sources.keys())!r})"
