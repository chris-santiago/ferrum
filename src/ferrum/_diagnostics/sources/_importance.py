"""Phase 10d — feature importance (permutation / native, SHAP, partial dependence)."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..deps import require_shap, require_sklearn


class FeatureImportanceMixin:
    """Phase 10d — feature importance (permutation / native, SHAP, partial dependence)."""

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
            scoring=scoring,
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

