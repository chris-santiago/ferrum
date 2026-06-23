"""ModelSource adapter — wraps a fitted estimator + data, exposes derived data.

This module is a thin re-export of the implementation that lives in
:mod:`ferrum.diagnostics.sources`.  ``ModelSource`` composes one
mixin per phase-10 domain (predictions, classification curves,
feature importance, model selection, clustering, ranking) over the
shared :class:`sources._base.BaseSource` infrastructure;
:class:`ComparedModelSource` (Phase 10h) lives in
:mod:`sources._compared`.

Keeping ``ModelSource`` and ``ComparedModelSource`` importable from
this module path preserves every existing
``from ferrum.diagnostics.source import ...`` site without
churn — the per-domain reorganization is purely internal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .sources._base import BaseSource
from .sources._classification import ClassificationCurvesMixin
from .sources._clustering import ClusteringMixin
from .sources._compared import ComparedModelSource
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
    whose schema is documented in ``ferrum.diagnostics._internal.schemas`` —
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
        sources = {name: cls(model, X, y, **kwargs) for name, model in models.items()}
        return ComparedModelSource(sources)


if TYPE_CHECKING:
    # Type-only conformance: ``ModelSource`` is one of the concrete adapters
    # the chart builders consume, so it must satisfy the ``DiagnosticSource``
    # method/return contract. pyright flags any future signature/return drift
    # here (and the matching assertion for ``_PrecomputedSource`` lives in
    # ``_internal/precomputed.py``).
    #
    # ``ComparedModelSource`` is deliberately *not* asserted statically: it
    # dispatches the same methods dynamically via ``__getattr__``, so it
    # conforms only at runtime (``isinstance(cms, DiagnosticSource)`` is True
    # — ``DiagnosticSource`` is ``runtime_checkable``). A static assertion
    # would be a false positive, since pyright cannot see the proxied methods.
    from .sources._protocols import DiagnosticSource

    def _assert_model_source_diagnostic(src: "ModelSource") -> DiagnosticSource:
        return src


__all__ = ["ModelSource", "ComparedModelSource"]
