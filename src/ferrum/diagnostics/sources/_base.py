"""``BaseSource`` — constructor, protocol detection, cache, validation.

Holds every piece of state that the per-domain mixin classes need
(``_X``, ``_y``, ``_model``, ``_capabilities``, ``_cache``,
``_feature_names``, ``_class_names``, ``_sample_weight``,
``_random_state``) plus the small helpers every mixin reaches for —
``_require_capability`` (raises ``AttributeError`` when the wrapped
estimator is missing a needed protocol method) and ``_cache_key``
(builds a hashable cache key from method name + kwargs).

Mixins are method-only and assume these attributes exist at
``self``.  No ``super().__init__()`` chains: ``BaseSource`` sits at
the bottom of the MRO so it always runs first when ``ModelSource``
is instantiated.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl
import pyarrow as pa


_PROTOCOL_ATTRS: tuple[str, ...] = (
    "predict",
    "predict_proba",
    "decision_function",
    "transform",
    "fit_transform",
    "fit_predict",
    "score",
    "feature_importances_",
    "coef_",
    "explained_variance_ratio_",
    "cluster_centers_",
    "labels_",
    "classes_",
)


def _coerce_X_y(X: Any, y: Any) -> tuple["pl.DataFrame | None", "pl.Series | None"]:
    """Coerce X to polars.DataFrame (or None) and y to polars.Series (or None).

    ``X=None`` passes straight through — ``ModelSource`` is constructible
    without a feature matrix; diagnostics that need one read it via
    ``BaseSource._require_X`` rather than ``self._X`` directly, so an
    X-less source refuses with a typed domain message instead of this
    function ever seeing ``None`` and delegating to ``to_polars(None)``
    (which raises an unrelated ``Chart(data=None)`` error).

    ``X``'s 2D-ndarray arm uses the sklearn-style ``f0, f1, ...`` feature
    naming (distinct from ``ferrum._coerce.to_polars``'s ``col_0, col_1,
    ...`` chart-data convention), so it stays local; every other input
    shape (``pl.DataFrame`` passthrough, narwhals-compatible frames, dict,
    list[dict]) delegates to the canonical ``to_polars`` coercion.
    """
    if X is None:
        X_df: "pl.DataFrame | None" = None
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        X_df = pl.from_numpy(X, schema=[f"f{i}" for i in range(X.shape[1])])
    else:
        from ferrum._coerce import to_polars

        X_df = to_polars(X)
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


class BaseSource:
    """Shared state + helpers for every ``ModelSource`` domain mixin.

    Not part of the public API — users import :class:`ferrum.ModelSource`
    which composes :class:`BaseSource` with the seven per-domain mixins.
    The split exists because the original monolithic implementation
    grew past 1500 lines along clear domain seams; keeping the seams
    in code matches the seams in the docs.
    """

    def __init__(
        self,
        model: Any,
        X: Any = None,
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
            list(feature_names)
            if feature_names is not None
            else (list(self._X.columns) if self._X is not None else [])
        )
        self._class_names: list[str] | None = list(class_names) if class_names is not None else None
        self._sample_weight = sample_weight
        self._random_state = random_state

        self._capabilities = frozenset(
            attr for attr in _PROTOCOL_ATTRS if hasattr(self._model, attr)
        )
        self._cache: dict[tuple, pl.DataFrame] = {}

    @property
    def X(self) -> "pl.DataFrame | None":
        """Feature matrix coerced to a polars DataFrame, or ``None``.

        Returns the value supplied to ``__init__`` (after coercion), or
        ``None`` when the source was constructed without one — a
        ``ModelSource`` is constructible with just a model, and X-free
        diagnostics (``pca_variance`` off ``explained_variance_ratio_``,
        ``intercluster_distance``) work without it. Diagnostics that
        need the feature matrix call ``_require_X`` internally, which
        raises a typed error naming the fix rather than letting a
        ``None`` leak into unrelated code downstream. Use this property
        for read-only access from chart builders and external callers —
        ``source._X`` is an internal alias preserved for back-compat.
        """
        return self._X

    @property
    def y(self) -> "pl.Series | None":
        """Target series, or ``None`` when no ``y`` was supplied.

        Returns the polars Series the constructor coerced from the
        ``y`` argument.  ``None`` means unsupervised — methods that
        need ground truth raise on call.
        """
        return self._y

    @property
    def model(self) -> Any:
        """The wrapped fitted estimator.

        Returns the model object supplied at construction time unchanged.
        Chart builders use it for occasional native introspection (e.g.
        ``model.classes_``, ``model.n_clusters``); prefer the public
        derived-data methods when one exists.
        """
        return self._model

    @property
    def feature_names(self) -> list[str]:
        """Column labels for the feature matrix.

        Returns the names supplied at construction time, or the DataFrame
        column names when ``X`` was a DataFrame, or ``["f0", "f1", ...]``
        for unlabeled array inputs.

        Returns
        -------
        list[str]
            Feature names in the same order as the columns of ``X``.
        """
        return list(self._feature_names)

    @property
    def capabilities(self) -> frozenset[str]:
        """Protocol attributes present on the wrapped estimator.

        A frozen subset of ``_PROTOCOL_ATTRS`` (``"predict"``,
        ``"predict_proba"``, ``"coef_"``, ``"feature_importances_"``, …)
        detected at construction time via ``hasattr``. Derived-data methods
        gate on this set to pick the appropriate code path and raise
        ``AttributeError`` with a clear message when a required attribute
        is absent.

        Returns
        -------
        frozenset[str]
            Attribute names that are present on the wrapped model.
        """
        return self._capabilities

    def _x_record_batch(self, diagnostic: str) -> pa.RecordBatch:
        """Coerce ``self._X`` to a ``pa.RecordBatch`` for Rust kernel calls.

        All derived-data methods that pass the feature matrix to Rust kernels
        should call this instead of inlining the ``from_pydict`` expression.
        Routes through ``_require_X(diagnostic)`` itself, so an X-less
        source refuses with the typed domain message rather than crashing
        on a ``None`` subscript; the conversion is a direct column-by-column
        Arrow export with no copy beyond what polars already holds.
        """
        X = self._require_X(diagnostic)
        return pa.RecordBatch.from_pydict({c: X[c].to_arrow() for c in X.columns})

    def _require_capability(self, attr: str, method_name: str) -> None:
        if attr not in self._capabilities:
            raise AttributeError(
                f"ModelSource.{method_name}() requires the wrapped model to "
                f"implement '{attr}'. Got {type(self._model).__name__!r} which "
                f"does not."
            )

    def _require_X(self, diagnostic: str) -> pl.DataFrame:
        """Return ``self._X``, raising a typed domain error when absent.

        Every diagnostic that reads the feature matrix calls this instead
        of touching ``self._X`` directly, so an X-less ``ModelSource``
        refuses with a message naming the diagnostic and the fix, instead
        of ``self._X`` being ``None`` and leaking an unrelated error (e.g.
        ``Chart(data=None) requires per-layer data...``) from deep inside
        a chart builder or Rust kernel call.
        """
        if self._X is None:
            raise ValueError(
                f"ModelSource.{diagnostic}() requires the training data: "
                f"ModelSource(model, X)."
            )
        return self._X

    def _cache_key(self, method: str, **kwargs) -> tuple:
        """Build a hashable cache key from method name + kwargs.

        Hashable kwarg values (the common case — ints, strs, None,
        tuples) pass through unchanged so a string ``"f1"`` keys
        identically across calls.  Unhashable values (lists, ndarrays,
        cv splitter objects, …) fall back to ``repr()`` so we don't
        raise ``TypeError`` at dict-lookup time; this trades cache
        stability across calls (repr of mutable objects can change)
        for crash-freedom on legitimate inputs that the public
        signatures advertise as ``Any``.  Callers that need stable
        cache identity across calls (e.g. for an ndarray of
        ``train_sizes``) should normalize to a tuple before calling.
        """

        def _hashable(v):
            try:
                hash(v)
            except TypeError:
                return repr(v)
            return v

        return (method, tuple(sorted((k, _hashable(v)) for k, v in kwargs.items())))
