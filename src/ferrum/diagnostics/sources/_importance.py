"""Phase 10d — feature importance (permutation / native, SHAP, partial dependence)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ferrum._validate import validate_choice
from .._internal.deps import require_shap, require_sklearn

if TYPE_CHECKING:
    from ._protocols import _SourceState as _MixinBase
else:
    _MixinBase = object


class FeatureImportanceMixin(_MixinBase):
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

        Parameters
        ----------
        method : {"builtin", "permutation"}, default "builtin"
            ``"builtin"`` reads the wrapped model's ``feature_importances_``
            (tree-based estimators) or ``coef_`` (linear estimators, averaged
            absolute value across classes for multi-output linears); ``std``
            is zero in this path since the built-in attribute exposes no
            per-feature variance. ``"permutation"`` calls sklearn's
            ``permutation_importance`` and populates ``std`` with the
            per-feature standard deviation across repeats.
        n_repeats : int, default 30
            Number of permutation repeats (``method="permutation"`` only).
        scoring : str or callable, optional
            Scorer passed to ``permutation_importance``
            (``method="permutation"`` only). ``None`` uses the estimator's
            default score.
        random_state : int, optional
            Seed for the permutation shuffles. Falls back to the source's
            ``random_state`` when omitted.
        """
        validate_choice("ModelSource.importances", "method", method, ("builtin", "permutation"))
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
            imp, std = self._importances_builtin()
        else:
            # method == "permutation" (validated above)
            imp, std = self._importances_permutation(
                n_repeats=n_repeats,
                scoring=scoring,
                random_state=rs,
            )

        if len(self._feature_names) < len(imp):
            # The discriminator is name AVAILABILITY, not X-presence: X is
            # just one of two ways names arrive (`X.columns` or an
            # explicit `feature_names=` kwarg, `_base.py`), and either one
            # can independently be absent or merely too short. Truth
            # table (imp has 5 entries):
            #   _feature_names == []                              -> nothing to label with at all      -> "requires feature names" (regardless of X)
            #   _feature_names == ["f0","f1","f2"] (from X)        -> names exist but too few            -> mismatch message
            #   _feature_names == ["a","b"] (from feature_names=)  -> names exist but too few            -> mismatch message
            # Branching on `self._X is None` instead (a prior version of
            # this guard) got the third row wrong: an X-less source with
            # an explicit-but-short `feature_names=` has names -- telling
            # it to pass `feature_names=[...]` is advice it already took.
            if not self._feature_names:
                # Nothing to label with at all: no X, no explicit
                # feature_names=. This is the only quadrant where the
                # "requires feature names" message is accurate.
                raise ValueError(
                    "ModelSource.importances() requires feature names: "
                    "construct with ModelSource(model, X) or "
                    "ModelSource(model, feature_names=[...])."
                )
            # Names exist (from X's columns or an explicit feature_names=)
            # but there are fewer of them than the model's importance
            # vector -- a genuine count mismatch (e.g. X is missing
            # columns the model was fit on, or feature_names= is too
            # short), not a missing-names problem. Do not advise
            # supplying X or feature_names=: whichever the caller used to
            # arrive at `self._feature_names`, they already did.
            raise ValueError(
                f"ModelSource.importances() got {len(imp)} importance "
                f"value(s) from the model but only "
                f"{len(self._feature_names)} feature name(s) (from X's "
                f"columns or feature_names=). The model was likely fit on "
                f"more features than this ModelSource was given -- pass "
                f"the full feature matrix via X= or the full name list "
                f"via feature_names=."
            )
        # kind="stable" so tied |importance| values keep their original
        # feature order across runs -- the default "quicksort" is not
        # stable and reorders ties nondeterministically for larger
        # feature counts (observed divergence at n>=~20 tied values).
        order = np.argsort(-np.abs(imp), kind="stable")
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

    def _importances_builtin(self) -> tuple[np.ndarray, np.ndarray]:
        """Read importance from the wrapped model's `feature_importances_` or
        `coef_` attribute. ``std`` is zero — sklearn's built-in attribute
        exposes no per-feature variance.
        """
        if "feature_importances_" in self._capabilities:
            imp = np.asarray(
                self._model.feature_importances_,
                dtype=np.float64,
            )
        elif "coef_" in self._capabilities:
            coef = np.asarray(self._model.coef_, dtype=np.float64)
            imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            raise AttributeError(
                "ModelSource.importances(method='builtin') requires the "
                "wrapped model to expose 'feature_importances_' or "
                f"'coef_'. Got {type(self._model).__name__!r} which "
                "exposes neither."
            )
        return imp, np.zeros_like(imp)

    def _importances_permutation(
        self,
        *,
        n_repeats: int,
        scoring: Any,
        random_state: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute permutation importance with per-feature std across repeats."""
        require_sklearn("importances(permutation)")
        from sklearn.inspection import permutation_importance

        if self._y is None:
            raise ValueError(
                "ModelSource.importances(method='permutation') requires y to be provided."
            )
        X = _sklearn_X(
            self._model,
            self._require_X("importances"),
            x_names_synthetic=self._x_names_synthetic,
            diagnostic="importances",
        )
        result = permutation_importance(
            self._model,
            X,
            self._y,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=random_state if random_state is not None else 0,
        )
        return (
            np.asarray(result.importances_mean, dtype=np.float64),
            np.asarray(result.importances_std, dtype=np.float64),
        )

    def shap_values(
        self,
        *,
        background: Any = None,
        max_evals: int = 500,
    ) -> pl.DataFrame:
        """Long-form SHAP values per (sample, feature, class).

        Returns a DataFrame with ``sample_id``, ``feature``, ``shap_value``,
        ``feature_value``, ``feature_value_normalized``, ``class_label``.

        - Regression: ``class_label`` is the constant ``"target"`` on every row.
        - Binary classifiers: ``class_label`` is the positive-class name on
          every row; SHAP values are for the positive class.
        - Multi-class classifiers: one row per (sample, feature, class);
          ``class_label`` carries the class name. The result has
          ``n_samples * n_features * n_classes`` rows total.

        Explainer is auto-picked by model capability:

        - ``coef_``: ``shap.LinearExplainer`` (deterministic, fast).
        - ``feature_importances_``: ``shap.TreeExplainer`` (deterministic
          for tree ensembles).
        - otherwise: ``shap.KernelExplainer`` (model-agnostic).

        Parameters
        ----------
        background : array-like, optional
            Background dataset for the model-agnostic
            ``shap.KernelExplainer`` path. When ``None``, the first
            ``min(50, N)`` rows of X are used. Ignored by the deterministic
            ``LinearExplainer`` / ``TreeExplainer`` paths.
        max_evals : int, default 500
            Reserved for future use (no-op today). Participates only in the
            result cache key; not yet forwarded to the SHAP explainer.
        """
        key = self._cache_key(
            "shap_values",
            background=str(background)[:64],
            max_evals=max_evals,
        )
        if key in self._cache:
            return self._cache[key]
        shap = require_shap("shap_values")

        # SHAP explainers do not reliably accept polars DataFrames (as of shap 0.45).
        X_np = self._require_X("shap_values").to_numpy()
        if "coef_" in self._capabilities:
            explainer = shap.LinearExplainer(self._model, X_np)
        elif "feature_importances_" in self._capabilities:
            explainer = shap.TreeExplainer(self._model)
        else:
            bg = background if background is not None else X_np[: min(50, len(X_np))]
            explainer = shap.KernelExplainer(self._model.predict, bg)

        sv_raw = explainer.shap_values(X_np)
        sv_by_class = _split_shap_by_class(sv_raw, self._model)
        class_labels = _shap_class_labels(self._model, len(sv_by_class))

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
                "shap_value": float(sv_by_class[c][s, f]),
                "feature_value": float(X_np[s, f]),
                "feature_value_normalized": float(f_norm[s, f]),
                "class_label": class_labels[c],
            }
            for c in range(len(sv_by_class))
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

        Parameters
        ----------
        features : list of str or int
            Feature names or column indices to compute partial dependence
            for. One set of rows is emitted per feature.
        grid_resolution : int, default 100
            Number of grid points sampled across each feature's range.
        kind : {"average", "individual", "both"}, default "average"
            ``"average"`` returns the marginal PD curve per feature with
            ``sample_id = -1`` (one row per grid point per feature).
            ``"individual"`` returns per-sample ICE curves: one row per
            ``(feature, sample_id, grid_point)`` triple with ``sample_id``
            in ``[0, n_samples)``. ``"both"`` returns the union (ICE rows
            plus average rows), so a downstream chart can overlay both
            layers on the same DataFrame. Chart builders pair the ICE rows
            with the ``detail`` encoding channel on ``sample_id`` to render
            one polyline per sample.
        """
        validate_choice(
            "ModelSource.partial_dependence", "kind", kind, ("average", "individual", "both")
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

        # Guard before resolving string feature names: on an X-less source
        # `self._feature_names` is `[]` (it only ever populates from
        # `X.columns` or an explicit `feature_names=` kwarg), so
        # `.index(f)` for a string `f` would raise an untyped
        # `ValueError: 'f' is not in list` before the typed X refusal ever
        # runs. Integer-indexed `features` don't touch `_feature_names`
        # here, so they'd otherwise reach the typed message while string
        # ones didn't -- the guard must come first regardless of `features`'
        # shape.
        # numpy required as the fallback: sklearn.inspection.partial_dependence
        # uses numpy integer column indexing internally, which polars does not
        # support. `_sklearn_X` upgrades this to a named pandas frame when the
        # model carries `feature_names_in_`, so sklearn gets column names
        # instead of raising a feature-name UserWarning once per grid-point
        # evaluation.
        X_sklearn = _sklearn_X(
            self._model,
            self._require_X("partial_dependence"),
            x_names_synthetic=self._x_names_synthetic,
            diagnostic="partial_dependence",
        )
        feature_idxs = [self._feature_names.index(f) if isinstance(f, str) else f for f in features]
        rows: list[dict] = []
        for f_idx in feature_idxs:
            fname = str(self._feature_names[f_idx])
            rows.extend(
                self._pd_rows_for_feature(
                    f_idx,
                    fname,
                    X_sklearn,
                    grid_resolution=grid_resolution,
                    kind=kind,
                )
            )
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def _pd_rows_for_feature(
        self,
        f_idx: int,
        fname: str,
        X_sklearn: Any,
        *,
        grid_resolution: int,
        kind: str,
    ) -> list[dict]:
        """Build partial-dependence rows for a single feature.

        ``X_sklearn`` is whatever `_sklearn_X` produced: a ``pandas.DataFrame``
        carrying either the model's fitted column names (synthetic-ndarray
        case) or ``X``'s own real column names (named-frame case), or a
        plain ``numpy.ndarray`` fallback. All are valid ``X`` inputs to
        ``sklearn.inspection.partial_dependence``, which integer-indexes
        ``features=[f_idx]`` positionally either way.

        sklearn returns ``individual`` of shape ``(n_outputs, n_samples,
        n_grid)`` when ``kind in ("individual", "both")``; this method
        uses a single output (binary class index 1 / regression scalar)
        by indexing ``[0]``. For ``"average"`` the shape is
        ``(n_outputs, n_grid)``.
        """
        from sklearn.inspection import partial_dependence as _sk_pd

        r = _sk_pd(
            self._model,
            X_sklearn,
            features=[f_idx],
            grid_resolution=grid_resolution,
            kind=kind,
        )
        grid = r["grid_values"][0]
        rows: list[dict] = []
        if kind in ("average", "both"):
            avg = np.asarray(r["average"])[0]
            rows.extend(
                {"feature": fname, "feature_value": float(v), "pd_value": float(p), "sample_id": -1}
                for v, p in zip(grid, avg)
            )
        if kind in ("individual", "both"):
            individual = np.asarray(r["individual"])[0]
            n_samples, _ = individual.shape
            for s in range(n_samples):
                rows.extend(
                    {
                        "feature": fname,
                        "feature_value": float(v),
                        "pd_value": float(p),
                        "sample_id": int(s),
                    }
                    for v, p in zip(grid, individual[s])
                )
        return rows


def _sklearn_X(model: Any, X: pl.DataFrame, *, x_names_synthetic: bool, diagnostic: str) -> Any:
    """Coerce a feature frame for a call into sklearn, avoiding feature-name
    warnings when the model was fit on named columns -- without silently
    mislabeling ``X`` or masking a genuine shape mismatch.

    sklearn records the column names an estimator was fit with on
    ``feature_names_in_``; handing that estimator a positionally-indexed
    array back at predict/inspect time raises a ``UserWarning`` per
    evaluation (for ``partial_dependence`` that's one warning per grid
    point -- 100 for the default ``grid_resolution=100``; polars frames
    trigger the same warning as plain arrays, since sklearn's transform-time
    name check is pandas-specific). Three cases, checked in order:

    1. No ``feature_names_in_`` on the model -- return a plain numpy array
       (today's behavior; there's nothing to reconcile).
    2. ``feature_names_in_`` present but ``X.width`` disagrees with it --
       raise a typed :class:`ValueError` naming ``diagnostic`` *before*
       building any pandas frame. (Without this check, building
       ``pd.DataFrame(X.to_numpy(), columns=list(names))`` on a width
       mismatch raises pandas' own ``Shape of passed values ... indices
       imply ...``, which names neither the model nor ferrum.)
    3. ``feature_names_in_`` present and widths agree -- branch on
       ``x_names_synthetic``, a fact *captured at coercion time* by
       ``sources/_base.py``'s ``_coerce_X_y`` (structural: a bare
       ``numpy.ndarray`` or a ``pandas.DataFrame`` with a default
       ``RangeIndex`` column index), not guessed here from name strings.
       An earlier version of this helper guessed provenance from whether
       ``X.columns`` matched ferrum's own ``f0, f1, ...`` schema, which
       misfired in both directions: real columns that happened to be
       literally named ``f0, f1, f2`` were misread as invented (silently
       producing a sign-flipped wrong answer with zero warning), and
       invented columns under any OTHER naming convention -- e.g.
       ``pd.DataFrame(array)``'s ``'0', '1', '2'`` or
       ``pl.DataFrame(array)``'s ``column_0, column_1, ...`` -- were
       misread as real (a hard error on input that used to compute
       correctly). Provenance is knowable exactly one frame up, so this
       is a fact lookup, not a guess:

       - ``x_names_synthetic=True`` -- ``X`` carries no real name
         information to defer to (a bare-array ``X`` never offered any),
         so stamp the model's fitted names onto ``X`` positionally:
         positional alignment with the fit order is the only contract
         those shapes ever offered. This is the case the whole rebuild
         exists for.
       - ``x_names_synthetic=False`` -- ``X`` carries real,
         caller-supplied names, so hand sklearn ``X``'s *own* names
         (``pd.DataFrame(X.to_numpy(), columns=list(X.columns))``) and let
         sklearn perform its own name/order validation: a match silences
         the warning exactly like the stamp branch; a genuine mismatch
         (renamed or reordered columns) raises sklearn's own precise
         ``ValueError('The feature names should match those that were
         passed during fit...')`` instead of silently producing a
         positionally-wrong (and warning-free) result. NEVER silently
         falls back to positional trust for a named frame.

    Disclosed residual (documented in `_coerce_X_y`'s docstring too):
    ``pl.DataFrame(array)`` -- a bare-array polars DataFrame built by the
    *caller* with no ``schema=`` given -- auto-generates ``column_0,
    column_1, ...`` labels that carry no structural marker distinguishing
    them from a user's real, deliberately-chosen names; polars does not
    expose anywhere on a ``pl.DataFrame`` instance whether its column
    labels were auto-assigned. That input is therefore treated as
    genuinely named (``x_names_synthetic=False``), so
    ``ModelSource(model_fit_with_real_names, pl.DataFrame(array), y)``
    now raises sklearn's name-mismatch error where a prior version of
    this helper silently computed a positionally-correct result with a
    warning. This is a deliberate, disclosed behavior change: a loud,
    actionable error naming the mismatched columns is strictly better
    than a silent guess that happens to be right. The two most common
    ways to hand over a nameless feature matrix -- a bare
    ``numpy.ndarray``, and ``pd.DataFrame(array)`` -- are unaffected;
    both are detected structurally, not by name pattern.

    Falls back to a plain numpy array when pandas isn't importable (no
    ``feature_names_in_`` means no warning today either way).

    Both of `_importance.py`'s sklearn-inspection call sites
    (``_importances_permutation``, ``partial_dependence``) route through
    this so they agree on one conversion rule instead of one passing a
    polars frame straight through (which sklearn rejects for integer
    column indexing, and still warns on feature names) and the other
    passing a bare numpy array.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return X.to_numpy()

    if X.width != len(names):
        raise ValueError(
            f"ModelSource.{diagnostic}() got X with {X.width} column(s) but "
            f"the model's feature_names_in_ records {len(names)} name(s) -- "
            f"the model was fit on a different number of features than this "
            f"ModelSource's X. Pass the same feature matrix (or feature "
            f"count) the model was trained on via X=."
        )

    try:
        import pandas as pd
    except ImportError:
        return X.to_numpy()

    if x_names_synthetic:
        return pd.DataFrame(X.to_numpy(), columns=list(names))
    return pd.DataFrame(X.to_numpy(), columns=list(X.columns))


def _split_shap_by_class(sv_raw: Any, model: Any) -> list[np.ndarray]:
    """Normalize SHAP output into a list of ``(n_samples, n_features)`` arrays,
    one per class.

    Upstream ``shap`` returns one of three shapes:

    - ``np.ndarray`` of shape ``(n_samples, n_features)`` — single class
      (regression, or a collapsed-to-positive binary).
    - ``list[np.ndarray]`` of K arrays each ``(n_samples, n_features)`` —
      one per class (older multi-class TreeExplainer / KernelExplainer).
    - ``np.ndarray`` of shape ``(n_samples, n_features, K)`` — newer
      multi-class TreeExplainer.

    For binary classifiers the per-class outputs are equal-and-opposite
    in sign (SHAP sums to zero across classes), so binary models route
    through the positive-class slice as a single-element list for schema
    uniformity.
    """
    is_binary = hasattr(model, "classes_") and len(model.classes_) == 2
    if isinstance(sv_raw, list):
        if is_binary:
            return [np.asarray(sv_raw[1], dtype=np.float64)]
        return [np.asarray(c, dtype=np.float64) for c in sv_raw]
    arr = np.asarray(sv_raw, dtype=np.float64)
    if arr.ndim == 3:
        if is_binary:
            return [arr[..., 1]]
        return [arr[..., c] for c in range(arr.shape[2])]
    return [arr]


def _shap_class_labels(model: Any, n_arrays: int) -> list[str]:
    """Return one class-label string per SHAP-values array.

    Regression / non-classifier models without a ``classes_`` attribute
    return ``["target"]``. Binary classifiers collapse to a single
    positive-class label. Multi-class classifiers return one label per
    class in model-class order.
    """
    if not hasattr(model, "classes_"):
        return ["target"] * n_arrays
    classes = list(model.classes_)
    if n_arrays == 1:
        # Binary classifier collapsed to positive class.
        return [str(classes[-1])]
    return [str(c) for c in classes[:n_arrays]]
