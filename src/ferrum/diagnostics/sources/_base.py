"""``BaseSource`` — constructor, protocol detection, cache, validation.

Holds every piece of state that the per-domain mixin classes need
(``_X``, ``_y``, ``_model``, ``_capabilities``, ``_cache``,
``_feature_names``, ``_class_names``, ``_sample_weight``,
``_random_state``, ``_x_names_synthetic``) plus the small helpers every
mixin reaches for — ``_require_capability`` (raises ``AttributeError`` when the wrapped
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


class _Uncacheable:
    """Marks one ``_cache_key`` kwarg value as unhashable-after-normalization.

    ``BaseSource._cache_key`` embeds a fresh instance per call for a kwarg
    that stays unhashable after ``_normalize`` (a dict, a cv splitter with
    no usable ``__hash__``), so the returned key can never equal any other
    key, past or future — the lookup always misses and the compute always
    runs fresh. ``_SourceCache`` recognizes this marker on *every* mutation
    path so the *storage* half of the contract holds too: per spec §6, an
    unnormalizable value is not cached, full stop, not cached-under-a-
    unique-key. Without this, the ~25 call sites' unconditional
    ``self._cache[key] = df`` would still leak one throwaway, never-reused
    entry into ``self._cache`` per call — unbounded growth for a caller
    that repeatedly passes the same reused unhashable object (e.g. looping
    ``discrimination_threshold(cv=my_splitter)``).
    """

    __slots__ = ()


def _key_holds_uncacheable(key: Any) -> bool:
    """Whether ``key`` carries an :class:`_Uncacheable` marker; refuses a malformed key.

    The one reading of a cache key's shape, shared by every
    :class:`_SourceCache` mutation path so they cannot disagree about which
    keys are storable. ``BaseSource._cache_key`` builds exactly one shape —
    ``(method_name, ((kwarg_name, value), ...))`` — and anything else is a
    caller mistake, not a cache miss: it is refused by name here rather than
    leaking a bare ``ValueError: too many values to unpack`` from inside the
    container, which named neither the container nor the key.
    """
    if not (isinstance(key, tuple) and len(key) == 2):
        raise TypeError(
            f"_SourceCache: a cache key must be the 2-tuple "
            f"(method_name, kwarg_pairs) that BaseSource._cache_key builds; "
            f"got {key!r} ({type(key).__name__}). Build the key with "
            f"self._cache_key(method, **kwargs) rather than by hand."
        )
    _method, pairs = key
    try:
        return any(isinstance(value, _Uncacheable) for _name, value in pairs)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"_SourceCache: a cache key's second element must be a tuple of "
            f"(kwarg_name, value) pairs, as BaseSource._cache_key builds; got "
            f"{pairs!r}. Build the key with self._cache_key(method, **kwargs) "
            f"rather than by hand."
        ) from exc


class _SourceCache(dict[tuple, "pl.DataFrame"]):
    """``self._cache``: a dict that refuses to store an uncacheable key.

    Every ``ModelSource`` derived-data method follows the same shape —
    ``if key in self._cache: return self._cache[key]`` on entry,
    ``self._cache[key] = df`` after computing — repeated identically at
    every call site across the mixin family. Rather than touch all ~25
    sites (most outside this fix's footprint) to add a conditional store,
    the store decision lives here once: a key built with an
    ``_Uncacheable`` marker (see above) never gets persisted, so those
    call sites' unconditional assignment becomes a correct no-op for
    exactly the values the contract says must not be cached.

    The refusal covers **every** way a dict can gain an entry, not just
    ``[] =``: ``update``, ``setdefault``, ``|=`` and the **constructor**
    each reach the store through their own C-level path (measured
    2026-09-17: overriding ``__setitem__`` alone left all four storing the
    entry this contract says must never be stored), so each is overridden
    to route through ``__setitem__``. ``fromkeys`` inherits the refusal
    through ``__setitem__`` already. A container whose guarantee held on
    one of five paths is weaker than its docstring, and the ~25 call sites
    read it as a plain dict.
    """

    def __init__(self, other: Any = (), /, **kwargs: Any) -> None:
        """``dict.__init__``, routed through :meth:`update` per entry.

        The one construction site today is ``_SourceCache()`` with no
        arguments, so this guards the next one rather than a live defect --
        which is the point: ``dict.__init__``'s C path does not call
        ``__setitem__``, so a populated construction would have stored what
        every other path refuses (measured 2026-09-17).
        """
        super().__init__()
        self.update(other, **kwargs)

    def __setitem__(self, key: tuple, value: "pl.DataFrame") -> None:
        if _key_holds_uncacheable(key):
            return
        super().__setitem__(key, value)

    def setdefault(self, key: tuple, default: Any = None) -> Any:
        """``dict.setdefault``, minus the store for an uncacheable key.

        Still returns what ``dict.setdefault`` would (the existing value, or
        ``default``), so a caller's control flow is unchanged — only the
        persistence is refused.
        """
        if key in self:
            return self[key]
        self[key] = default
        return default

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        """``dict.update``, routed through :meth:`__setitem__` per entry."""
        for key, value in dict(other, **kwargs).items():
            self[key] = value

    def __ior__(self, other: Any) -> "_SourceCache":
        """``|=``, routed through :meth:`update` (the C-level path is not)."""
        self.update(other)
        return self


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


def _coerce_X_y(X: Any, y: Any) -> tuple["pl.DataFrame | None", "pl.Series | None", bool]:
    """Coerce X to polars.DataFrame (or None) and y to polars.Series (or None);
    also report whether X's column names were INVENTED during coercion.

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

    Returns
    -------
    tuple[pl.DataFrame | None, pl.Series | None, bool]
        Coerced X, coerced y, and ``x_names_synthetic`` — ``True`` when
        X's column names were invented by this coercion rather than
        supplied by the caller: a bare 2D ``numpy.ndarray`` (this
        function's own ``f0, f1, ...`` naming), or a ``pandas.DataFrame``
        whose column index is a default ``RangeIndex`` (``pd.DataFrame(array)``
        with no ``columns=`` given — a *structural* check on the index
        type, not a guess from the label strings, so it can't be
        confused with a caller who deliberately named a column ``"f0"``).
        Every other shape — a pandas DataFrame with any other column
        index, any ``polars.DataFrame``, dict, list[dict], or other
        narwhals-compatible frame — reports ``False`` (genuinely named).
        That includes nameless frames whose auto-generated labels carry
        no structural marker: ``pl.DataFrame(array)``'s ``column_0,
        column_1, ...``, and a ``pd.DataFrame(array)`` whose default
        labels stopped being a ``RangeIndex`` (selecting a list of its
        columns, ``df[[0, 2]]``, yields a plain ``Index``). See
        `sources/_importance.py`'s `_sklearn_X` docstring for the
        disclosed consequence.
        Consumed by `_sklearn_X` to decide whether it's safe to stamp a
        model's ``feature_names_in_`` onto X positionally, or whether
        X's own names must be honored and validated by sklearn.
    """
    x_names_synthetic = False
    if X is None:
        X_df: "pl.DataFrame | None" = None
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        X_df = pl.from_numpy(X, schema=[f"f{i}" for i in range(X.shape[1])])
        x_names_synthetic = True
    else:
        from ferrum._coerce import to_polars

        # Structural check on the ORIGINAL object (before to_polars strips
        # the type information): a pandas DataFrame's default RangeIndex
        # column labeling means no real names were ever supplied. Duck-typed
        # by module/class name rather than `isinstance(X, pd.DataFrame)` so
        # constructing a ModelSource never imports pandas as a side effect.
        if type(X).__module__.startswith("pandas.") and type(X).__name__ == "DataFrame":
            x_names_synthetic = type(X.columns).__name__ == "RangeIndex"

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
    return X_df, y_ser, x_names_synthetic


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
        self._X, self._y, self._x_names_synthetic = _coerce_X_y(X, y)
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
        # Annotated as the subclass, not as ``dict``: this container refuses
        # stores that a plain dict would accept, and a reader at any of the
        # ~25 ``self._cache[key] = df`` sites has to see that from the type.
        self._cache: _SourceCache = _SourceCache()

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
    def class_names(self) -> list[str] | None:
        """Per-class display labels, or ``None`` when unresolvable.

        Resolves the ``class_names=`` supplied at construction time, else
        the wrapped model's ``classes_`` attribute (stringified). Returns
        ``None`` when neither is available -- unlike ``probabilities()``,
        this property cannot complete a positional ``class_{i}`` fallback
        on its own, since that requires a definite class count (the width
        of a computed prediction matrix), which is not knowable without
        running inference.

        These are **display** labels, matching the constructor's
        ``class_names`` parameter docs -- not guaranteed to be values
        that appear in ``y``. When ``class_names=`` is an alias (e.g.
        ``["setosa", "versicolor"]`` against an int-coded ``y``), this
        property returns the alias, not the underlying comparison value.
        Code that needs to binarize against ``y`` (as the classification
        curve methods on this class do) must go through the model's own
        ``classes_`` coerced into ``y``'s domain, not this property.

        Returns
        -------
        list[str] | None
            Per-class display labels in resolution order, or ``None``
            when the wrapped model exposes neither ``class_names=`` nor
            ``classes_``.
        """
        if self._class_names is not None:
            return list(self._class_names)
        if hasattr(self._model, "classes_"):
            return [str(c) for c in self._model.classes_]
        return None

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
                f"ModelSource.{diagnostic}() requires the training data: ModelSource(model, X)."
            )
        return self._X

    def _cache_key(self, method: str, **kwargs) -> tuple:
        """Build a hashable cache key from method name + kwargs.

        Array-likes are normalized centrally, content-preserving, before
        any hashability check: an ``ndarray`` (of any dimension) converts
        via ``.tolist()``, and any ``list``/``tuple`` (including one found
        nested inside an array-like) converts to a nested tuple of the
        same scalars — no rounding, no truncation. This is why a string
        ``"f1"`` still keys identically across calls (the common case,
        already hashable, untouched), and why an ndarray of ``train_sizes``
        now keys on its *values* rather than on whatever ``repr()``
        happened to print. Equal content produces an equal key only when
        the values' numeric kind also matches: every ``int`` and ``float``
        leaf (before or after ``ndarray``/``list``/``tuple`` normalization)
        is tagged with its own type, so ``train_sizes=[1]`` (sklearn:
        absolute sample count) and ``train_sizes=[1.0]`` (sklearn: fraction
        of the training set) key differently even though ``1 == 1.0`` —
        without the tag, Python's cross-type numeric equality/hash
        (``hash(1) == hash(1.0)``) would collide two semantically
        different ``learning_curve`` requests onto the same cached frame,
        the same failure class as the three repros below via a fourth,
        dtype-driven vector.

        A value that is **still** unhashable after normalization (a dict,
        a cv splitter object, any custom class without ``__hash__``) is
        never cached at all: this call gets a cache key built from a
        fresh ``_Uncacheable`` marker that cannot equal any other key,
        past or future, so the lookup always misses, the compute always
        runs fresh, and — via ``self._cache``'s ``_SourceCache.__setitem__``
        override — the result is never stored either (spec §6: an
        unnormalizable value is not cached, not cached-under-a-unique-key;
        a stored-but-unreachable entry would just leak memory for the life
        of the ``ModelSource``). The prior design fell back to ``repr(v)``
        for these, which is identity-derived for any object that doesn't
        override ``__repr__``: two unrelated temporaries of the same class
        can land at the same freed memory address and produce the *same*
        key, silently returning one call's cached frame for a completely
        different request (measured: ``discrimination_threshold(cv=Splitter(3))``
        followed by ``discrimination_threshold(cv=Splitter(7))``). A
        truncating fallback (``shap_values``' former
        ``str(background)[:64]``, or numpy's rounding/``...``-truncating
        ``repr()``) has the same failure mode without even needing address
        reuse: two different arrays can print identically. Recomputing on
        every call for these inputs is strictly safer than either.
        """

        def _normalize(v):
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, (list, tuple)):
                return tuple(_normalize(x) for x in v)
            # Tag int/float leaves with their own type: `1 == 1.0` and
            # `hash(1) == hash(1.0)`, so an untagged tuple of ints and an
            # untagged tuple of equal-valued floats would otherwise compare
            # (and hash) equal — silently colliding two array-likes whose
            # dtype carries real meaning to the caller (e.g. sklearn's
            # `learning_curve(train_sizes=)`: int means absolute count,
            # float means fraction). `bool` is left untagged (a `bool`
            # kwarg carries no such absolute/fractional split).
            if isinstance(v, int) and not isinstance(v, bool):
                return ("int", v)
            if isinstance(v, float):
                return ("float", v)
            return v

        def _hashable(v):
            normalized = _normalize(v)
            try:
                hash(normalized)
            except TypeError:
                return _Uncacheable()
            return normalized

        return (method, tuple(sorted((k, _hashable(v)) for k, v in kwargs.items())))
