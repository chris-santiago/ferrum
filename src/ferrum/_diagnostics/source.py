"""ModelSource adapter — wraps a fitted estimator + data, exposes derived data.

Phase 10a: constructor, protocol detection, cache, .predictions(),
.probabilities(). Other methods land in 10b-10g; `ComparedModelSource`
in 10h.

The implementation is split across :mod:`ferrum._diagnostics.sources`:

- :class:`sources._base.BaseSource` — constructor, capability detection,
  cache key infrastructure.
- :class:`sources._predictions.PredictionsMixin` — 10a methods.
- :class:`sources._classification.ClassificationCurvesMixin` — 10b methods.
- :class:`sources._importance.FeatureImportanceMixin` — 10d methods.
- :class:`sources._selection.ModelSelectionMixin` — 10e methods.
- :class:`sources._clustering.ClusteringMixin` — 10f methods.
- :class:`sources._ranking.RankingMixin` — 10g methods.

Keeping ``ModelSource`` at this module path preserves
``from ferrum._diagnostics.source import ModelSource`` for every
external caller (visualizers, figure-level builders, tests).
"""
from __future__ import annotations

from typing import Any

import polars as pl

from .sources._base import BaseSource
from .sources._classification import ClassificationCurvesMixin
from .sources._clustering import ClusteringMixin
from .sources._importance import FeatureImportanceMixin
from .sources._predictions import PredictionsMixin
from .sources._ranking import RankingMixin
from .sources._selection import ModelSelectionMixin


class ModelSource(
    PredictionsMixin,
    ClassificationCurvesMixin,
    FeatureImportanceMixin,
    ModelSelectionMixin,
    ClusteringMixin,
    RankingMixin,
    BaseSource,
):
    """Wrap a fitted estimator + dataset and expose model-diagnostic
    derived data as polars DataFrames.

    Constructing a ``ModelSource`` is sklearn-free — only attribute
    introspection runs at ``__init__`` time. Derived-data methods that
    need sklearn / shap / umap lazy-import on call, so ``import ferrum``
    never pulls those packages into the user's process unless they
    actually compute a diagnostic that requires them.

    Each derived-data method returns a long-form polars DataFrame
    whose schema is documented in ``ferrum._diagnostics.schemas`` —
    chart builders and Visualizers consume the same frames.

    Parameters
    ----------
    model : Any
        A fitted estimator. Must expose at least ``predict``; some
        methods require additional protocol attributes
        (``predict_proba``, ``coef_``, ``feature_importances_``,
        ``cluster_centers_``, ``explained_variance_ratio_``, …) and
        raise ``AttributeError`` with the missing attribute name when
        called against an incompatible model.
    X : polars.DataFrame | pandas.DataFrame | pyarrow.Table | numpy.ndarray
        Feature matrix. Coerced internally to a polars DataFrame; any
        ``narwhals``-compatible input also works.
    y : array-like, optional
        Target. Required by methods that depend on ground truth (every
        method except ``probabilities`` and the unsupervised
        ``silhouette`` / ``pca_variance`` / ``embeddings`` /
        ``intercluster_distance`` / ``rank1d(algorithm != "covariance")``
        / ``rank2d`` family).
    feature_names : sequence of str, optional
        Column labels. Defaults to ``X.columns`` when ``X`` is a
        DataFrame, or ``["f0", "f1", ...]`` otherwise.
    class_names : sequence of str, optional
        Per-class display labels for classification diagnostics.
        Defaults to ``model.classes_`` when available, else the unique
        values of ``y``.
    sample_weight : array-like, optional
        Per-row weights forwarded to sklearn scorers that accept them.
    random_state : int, optional
        Seed propagated to every derived-data method whose underlying
        compute consumes randomness (importances permutation, SHAP
        background sampling, UMAP / t-SNE / MDS embeddings,
        cross-validation curves, partial-dependence sampling).
        Deterministic methods ignore the value.

    Examples
    --------
    >>> import ferrum as fm
    >>> source = fm.ModelSource(model, X, y, random_state=0)
    >>> fm.roc_chart(source)              # use directly with a figure function
    >>> source.predictions()              # access derived data as a DataFrame
    >>> source.confusion_matrix(normalize="true")
    """

    # __init__, feature_names, capabilities, _require_capability, and
    # _cache_key are inherited from BaseSource — see sources/_base.py.

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

        Parameters
        ----------
        models : dict[str, Any]
            Mapping from display name to fitted estimator. Each estimator is
            wrapped in its own ``ModelSource`` constructed with the shared
            ``X``, ``y``, and any additional ``kwargs`` (e.g.
            ``random_state``, ``feature_names``, ``class_names``).
        X : array-like
            Feature matrix shared by all models. Accepted types match
            ``ModelSource.__init__``.
        y : array-like, optional
            Target shared by all models. Required by most derived-data
            methods (same constraints as ``ModelSource``).
        **kwargs : Any
            Keyword arguments forwarded verbatim to each ``ModelSource``
            constructor (e.g. ``random_state``, ``feature_names``,
            ``class_names``, ``sample_weight``).

        Returns
        -------
        ComparedModelSource
            Multi-model wrapper whose derived-data methods return long-form
            DataFrames with an extra ``model: Utf8`` column.

        Examples
        --------
        >>> import ferrum as fm
        >>> from sklearn.linear_model import Ridge, Lasso
        >>> cms = fm.ModelSource.compare(
        ...     {"ridge": Ridge().fit(X, y), "lasso": Lasso().fit(X, y)},
        ...     X, y, random_state=0,
        ... )
        >>> fm.roc_chart(cms)          # overlay both ROC curves
        >>> cms.model_names
        ['ridge', 'lasso']
        """
        _ACCEPTED_COMPARE_KWARGS: frozenset[str] = frozenset(
            {"random_state", "feature_names", "class_names", "sample_weight"}
        )
        unknown = set(kwargs) - _ACCEPTED_COMPARE_KWARGS
        if unknown:
            raise TypeError(
                f"ModelSource.compare() received unexpected keyword argument(s): "
                f"{sorted(unknown)}. "
                f"Accepted kwargs: {sorted(_ACCEPTED_COMPARE_KWARGS)}"
            )
        sources = {
            name: cls(model, X, y, **kwargs) for name, model in models.items()
        }
        return ComparedModelSource(sources)

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

    ``_X``, ``_y``, ``_feature_names``, and ``_class_names`` resolve to
    the first source's values (every wrapped source shares ``X`` / ``y``
    by construction in ``ModelSource.compare``, so any one will do);
    accessing ``_model`` raises since there is no single estimator.
    ``model_names`` reports the configured ordering.

    Parameters
    ----------
    sources : dict[str, ModelSource]
        Mapping from model name (used for the ``model`` column) to the
        underlying ``ModelSource``. Must contain at least one entry —
        passing an empty dict raises ``ValueError``.

    Examples
    --------
    >>> import ferrum as fm
    >>> cms = fm.ModelSource.compare({"ridge": ridge, "lasso": lasso}, X, y)
    >>> fm.roc_chart(cms)                  # overlay both curves
    >>> cms.model_names
    ['ridge', 'lasso']
    >>> cms.roc_curve()                    # long-form frame with `model` column
    """

    __slots__ = ("_sources",)

    def __init__(self, sources: dict[str, ModelSource]):
        if not sources:
            raise ValueError("ComparedModelSource requires at least one source.")
        self._sources = dict(sources)

    @property
    def model_names(self) -> list[str]:
        """Ordered list of model display names.

        Returns the keys of the ``sources`` dict supplied at construction time,
        in insertion order. Each name corresponds to the value written into the
        ``model`` column on every derived-data DataFrame.

        Returns
        -------
        list[str]
            Model names in the order they were registered.
        """
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
        if name in ("_X", "_y", "_feature_names", "_class_names", "_capabilities"):
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
