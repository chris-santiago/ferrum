from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum
from tests.fixtures import load_fixture, load_dataset


class _DuckModel:
    """Minimal duck-typed model: predict only."""

    def predict(self, X):
        return np.zeros(len(X))


def test_constructor_accepts_polars_dataframe():
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    source = ferrum.ModelSource(_DuckModel(), df, y=[0, 1])
    assert source.feature_names == ["a", "b"]


def test_constructor_accepts_numpy_array():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    source = ferrum.ModelSource(_DuckModel(), X, y=[0, 1])
    assert source.feature_names == ["f0", "f1"]


def test_class_names_none_when_unresolvable():
    """No class_names= and no model.classes_ -> None (F-L11-08 / G4)."""
    df = pl.DataFrame({"a": [1.0, 2.0]})
    source = ferrum.ModelSource(_DuckModel(), df, y=[0, 1])
    assert source.class_names is None


def test_class_names_from_constructor_kwarg():
    df = pl.DataFrame({"a": [1.0, 2.0]})
    source = ferrum.ModelSource(_DuckModel(), df, y=[0, 1], class_names=["setosa", "versicolor"])
    assert source.class_names == ["setosa", "versicolor"]


def test_class_names_from_model_classes_attribute():
    class _ClassifierDuck(_DuckModel):
        classes_ = np.array([0, 1, 2])

    df = pl.DataFrame({"a": [1.0, 2.0]})
    source = ferrum.ModelSource(_ClassifierDuck(), df, y=[0, 1])
    assert source.class_names == ["0", "1", "2"]


def test_class_names_constructor_kwarg_wins_over_model_classes():
    class _ClassifierDuck(_DuckModel):
        classes_ = np.array([0, 1])

    df = pl.DataFrame({"a": [1.0, 2.0]})
    source = ferrum.ModelSource(
        _ClassifierDuck(), df, y=[0, 1], class_names=["negative", "positive"]
    )
    assert source.class_names == ["negative", "positive"]


def test_class_names_proxied_on_compared_model_source():
    """ComparedModelSource resolves class_names from its first wrapped
    source, matching the other read-only BaseSource properties."""
    df = pl.DataFrame({"a": [1.0, 2.0]})
    cms = ferrum.ModelSource.compare(
        {"m": _DuckModel()}, df, [0, 1], class_names=["negative", "positive"]
    )
    assert cms.class_names == ["negative", "positive"]


def test_capability_detection():
    df = pl.DataFrame({"a": [1.0]})
    source = ferrum.ModelSource(_DuckModel(), df)
    assert "predict" in source.capabilities
    assert "predict_proba" not in source.capabilities


def test_predictions_requires_predict():
    class NoPredict:
        pass

    df = pl.DataFrame({"a": [1.0]})
    source = ferrum.ModelSource(NoPredict(), df, y=[0.0])
    with pytest.raises(AttributeError, match="predict"):
        source.predictions()


def test_predictions_against_ridge_fixture():
    """Studentized residuals + Cook's distance computed for linear estimator."""
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    y = df["y"]

    source = ferrum.ModelSource(model, X, y)
    pred = source.predictions()
    assert pred.columns == [
        "y_true",
        "y_pred",
        "residual",
        "studentized_residual",
        "cooks_distance",
        "leverage",
    ]
    assert pred.shape == (df.height, 6)
    # Leverage for a linear estimator is in (0, 1) and sums to p (the
    # column count of X-with-intercept) by the hat-matrix trace identity.
    leverage = pred["leverage"].to_numpy()
    assert np.all(leverage >= 0.0)
    assert np.all(leverage < 1.0)
    # 5 features + intercept = 6, so sum(h_ii) == 6 up to numerical error.
    np.testing.assert_allclose(leverage.sum(), 6.0, rtol=1e-6)
    np.testing.assert_allclose(
        pred["residual"].to_numpy(),
        pred["y_true"].to_numpy() - pred["y_pred"].to_numpy(),
        rtol=1e-12,
    )
    cooks = pred["cooks_distance"].to_numpy()
    # Cook's D must be non-negative and finite for a linear estimator.
    assert (cooks >= 0.0).all()
    assert np.isfinite(cooks).all()


def test_probabilities_against_binary_logistic_fixture():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]

    source = ferrum.ModelSource(model, X, y)
    proba = source.probabilities()
    proba_cols = [c for c in proba.columns if c.startswith("proba_")]
    assert len(proba_cols) == 2
    sums = proba.select(proba_cols).to_numpy().sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-10)


def test_probabilities_caching():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    p1 = source.probabilities()
    p2 = source.probabilities()
    assert p1 is p2  # cache returns same object


def test_cache_key_accepts_unhashable_kwargs():
    # Before P3.2, _cache_key(method, train_sizes=[1, 2, 3]) crashed at
    # dict-lookup time because the kwarg value was unhashable and ended
    # up inside a tuple used as a dict key. #224(b), 2026-09-13: a list or
    # ndarray now normalizes into a nested tuple of its own values FIRST
    # (content-preserving, no rounding/truncation), so equal content
    # produces an equal, cacheable key instead of falling back to
    # identity-derived repr(). A value that is still unhashable after
    # normalization gets a fresh, never-repeating key, so it can never
    # collide with a past or future call — the lookup just always misses
    # (the call recomputes) instead of raising TypeError or caching under
    # the wrong slot.
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    # A list (unhashable) normalizes to a tuple — must not raise, and
    # equal content must produce an equal key.
    k1 = source._cache_key("dummy", train_sizes=[0.1, 0.5, 1.0])
    k2 = source._cache_key("dummy", train_sizes=[0.1, 0.5, 1.0])
    assert k1 == k2  # same input → same key
    # An ndarray (unhashable) normalizes the same way as an equal-content list.
    arr = np.array([0.1, 0.5, 1.0])
    k3 = source._cache_key("dummy", train_sizes=arr)
    assert k3 == k1  # same content, different container type → same key
    # Mixed hashable + unhashable-but-normalizable kwargs.
    k4 = source._cache_key("dummy", cv=5, train_sizes=[0.1, 0.5])
    assert hash(k4) == hash(k4)
    # A value that stays unhashable even after normalization (a plain dict
    # isn't an ndarray/list/tuple, so it is not normalized) never repeats:
    # every call gets its own key.
    k5 = source._cache_key("dummy", opts={"a": 1})
    k6 = source._cache_key("dummy", opts={"a": 1})
    assert k5 != k6
    # An unhashable kwarg value is not cached, period: a repeated call with
    # the SAME opts object also never repeats a key (correctness over
    # caching means every such call is a fresh, uncached miss).
    same_opts = {"a": 1}
    k7 = source._cache_key("dummy", opts=same_opts)
    k8 = source._cache_key("dummy", opts=same_opts)
    assert k7 != k8
    # Equal-valued int and float array-likes must NOT collapse to the same
    # key: sklearn's own learning_curve(train_sizes=) reads an int array as
    # an absolute sample count and a float array as a fraction — a real
    # semantic difference the key must preserve even though `1 == 1.0`.
    k9 = source._cache_key("dummy", train_sizes=[1])
    k10 = source._cache_key("dummy", train_sizes=[1.0])
    assert k9 != k10
    k11 = source._cache_key("dummy", train_sizes=np.array([1, 2, 3]))
    k12 = source._cache_key("dummy", train_sizes=np.array([1.0, 2.0, 3.0]))
    assert k11 != k12


def test_cache_key_uncacheable_value_never_grows_the_cache():
    """RED-proof for #224(b)/N5's storage half: an unhashable-after-
    normalization value must not just miss the cache lookup, it must not
    be STORED under its one-off key either.

    Pre-fix, ``_cache_key`` gave every such call a fresh, never-repeating
    ``object()`` sentinel embedded in the key, but every one of the ~25
    call sites unconditionally does ``self._cache[key] = df`` after
    computing — so a caller who repeatedly passes the same reused
    unhashable value (a real cross-validation exploration pattern: looping
    ``discrimination_threshold(cv=my_splitter)``) would grow
    ``self._cache`` by one throwaway, never-reused entry per call,
    unbounded, for the life of the ``ModelSource``. ``self._cache`` is now
    a ``_SourceCache`` that refuses to store any key built from an
    ``_Uncacheable`` marker, so the loop below leaves the cache exactly as
    empty as it started.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])

    class _EqOnlySplitter:
        def __init__(self, n_splits):
            self.n_splits = n_splits

        def __eq__(self, other):
            return isinstance(other, _EqOnlySplitter) and other.n_splits == self.n_splits

        def split(self, X, y=None):
            from sklearn.model_selection import KFold

            return KFold(n_splits=self.n_splits, shuffle=True, random_state=0).split(X)

    reused_splitter = _EqOnlySplitter(3)
    source.discrimination_threshold(cv=reused_splitter)
    # One legitimate entry: `discrimination_threshold` internally calls the
    # separately-cached `probabilities()`, a real, hashable-keyed hit
    # shared across every iteration below. The uncacheable `cv=` key
    # itself must never add a second entry, no matter how many times the
    # SAME reused splitter is passed.
    size_after_one_call = len(source._cache)
    for _ in range(5):
        source.discrimination_threshold(cv=reused_splitter)
    assert len(source._cache) == size_after_one_call


@pytest.mark.parametrize(
    ("store", "name"),
    [
        (lambda cache, key, value: cache.__setitem__(key, value), "[]="),
        (lambda cache, key, value: cache.update({key: value}), "update(mapping)"),
        (lambda cache, key, value: cache.update([(key, value)]), "update(pairs)"),
        (lambda cache, key, value: cache.setdefault(key, value), "setdefault"),
        (lambda cache, key, value: cache.__ior__({key: value}), "|="),
    ],
)
def test_the_uncacheable_refusal_covers_every_mutation_path(store, name):
    """The store refusal is a property of the container, not of one operator.

    ``_SourceCache`` overrode only ``__setitem__`` when it shipped, so
    ``update``, ``setdefault`` and ``|=`` each reached the C-level store
    directly and persisted exactly the entry the contract says must never
    be persisted (measured 2026-09-17, whole-wave design review finding 1).
    No call site uses those three today -- which is the point: nothing
    would have caught the first one that did, and the ~25 sites read
    ``self._cache`` as a plain dict.
    """
    from ferrum.diagnostics.sources._base import _SourceCache, _Uncacheable

    cache = _SourceCache()
    cacheable = ("method", (("cv", 5),))
    uncacheable = ("method", (("cv", _Uncacheable()),))

    store(cache, cacheable, "kept")
    store(cache, uncacheable, "refused")

    assert list(cache) == [cacheable], f"{name} stored an uncacheable key"


def test_the_uncacheable_refusal_covers_the_constructor_too():
    """``dict.__init__``'s C path does not call ``__setitem__`` either.

    The one construction site today is ``_SourceCache()`` with no
    arguments, so this guards the next one -- which is the same shape the
    original finding was about, one level down: a container whose
    invariant holds on four of five paths (cycle-2 design review finding
    F, probe-confirmed storing before the ``__init__`` override).
    """
    from ferrum.diagnostics.sources._base import _SourceCache, _Uncacheable

    cacheable = ("method", (("cv", 5),))
    uncacheable = ("method", (("cv", _Uncacheable()),))

    assert list(_SourceCache({cacheable: "kept", uncacheable: "refused"})) == [cacheable]
    assert list(_SourceCache([(uncacheable, "refused")])) == []
    assert dict(_SourceCache()) == {}
    # `fromkeys` builds through `__setitem__` and so inherits the refusal.
    assert list(_SourceCache.fromkeys([cacheable, uncacheable], "v")) == [cacheable]


def test_setdefault_still_answers_what_dict_setdefault_answers():
    """Refusing the *store* must not change what the call returns.

    ``setdefault`` is a read-and-maybe-write in one call; silently
    returning ``None`` for a refused key would break a caller's control
    flow rather than just its caching.
    """
    from ferrum.diagnostics.sources._base import _SourceCache, _Uncacheable

    cache = _SourceCache()
    cacheable = ("method", (("cv", 5),))
    uncacheable = ("method", (("cv", _Uncacheable()),))

    assert cache.setdefault(cacheable, "first") == "first"
    assert cache.setdefault(cacheable, "second") == "first"  # existing value wins
    assert cache.setdefault(uncacheable, "fresh") == "fresh"  # returned, not stored
    assert uncacheable not in cache


@pytest.mark.parametrize(
    "malformed",
    ["roc_curve", ("roc_curve",), ("roc_curve", (), "extra"), ("roc_curve", 5)],
)
def test_a_malformed_cache_key_refuses_by_name(malformed):
    """A key that is not ``_cache_key``'s shape names the container.

    It used to leak ``ValueError: too many values to unpack (expected 2)``
    from inside ``__setitem__``'s bare unpack, which named neither the
    container, the key, nor the fix.
    """
    from ferrum.diagnostics.sources._base import _SourceCache

    cache = _SourceCache()
    with pytest.raises(TypeError, match=r"_SourceCache: a cache key"):
        cache[malformed] = "df"


def test_a_real_cache_key_is_never_malformed():
    """The refusal above must not be reachable from ``_cache_key`` itself.

    Pins the producer against the consumer: whatever shape ``_cache_key``
    builds, the container has to accept it, uncacheable marker or not.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])

    source._cache[source._cache_key("dummy")] = "df"
    source._cache[source._cache_key("dummy", cv=5, train_sizes=[0.1, 0.5])] = "df"
    source._cache[source._cache_key("dummy", opts={"a": 1})] = "df"
    assert len(source._cache) == 2  # the opts= key is the refused one


# --- 10b: classification curves --------------------------------------


def test_roc_curve_binary_schema():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    roc = source.roc_curve()
    assert set(roc.columns) == {"fpr", "tpr", "threshold", "class", "auc"}
    assert roc.height >= 2
    aucs = roc["auc"].unique().to_list()
    assert len(aucs) == 1  # binary → one AUC
    assert 0.0 <= aucs[0] <= 1.0


def test_roc_curve_multiclass_macro_average():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    roc = source.roc_curve(average="macro")
    classes_seen = set(roc["class"].unique().to_list())
    assert "macro" in classes_seen
    assert len(classes_seen - {"macro"}) >= 3


def test_pr_curve_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    pr = source.pr_curve()
    assert set(pr.columns) == {"precision", "recall", "threshold", "class", "ap"}
    assert pr.height >= 2
    aps = pr["ap"].unique().to_list()
    assert len(aps) == 1
    assert 0.0 <= aps[0] <= 1.0


def test_pr_curve_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    pr = source.pr_curve()
    classes_seen = set(pr["class"].unique().to_list())
    assert len(classes_seen) >= 3


def test_pr_curve_multiclass_macro_average():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    pr = source.pr_curve(average="macro")
    # average=<single-curve mode>: ONLY the summary class appears.
    classes_seen = set(pr["class"].unique().to_list())
    assert classes_seen == {"macro"}
    ap_values = pr["ap"].unique().to_list()
    assert len(ap_values) == 1
    assert 0.0 <= ap_values[0] <= 1.0


def test_pr_curve_multiclass_micro_average():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    pr = source.pr_curve(average="micro")
    classes_seen = set(pr["class"].unique().to_list())
    assert classes_seen == {"micro"}


def test_pr_curve_bad_average_raises():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    with pytest.raises(ValueError, match=r"ModelSource.pr_curve: average must be one of"):
        source.pr_curve(average="not-a-real-mode")


def test_roc_curve_caching():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    assert source.roc_curve() is source.roc_curve()


def test_calibration_curve_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    cal = source.calibration_curve(n_bins=5)
    # `brier` joined the schema at the batch G close: the per-sample Brier
    # score broadcast onto every bin row, the same way `auc`/`ap` broadcast
    # onto ROC/PR rows, so every Brier display surface reads one number
    # from one source instead of re-deriving it from the binned curve.
    assert set(cal.columns) == {"mean_predicted", "fraction_positive", "count", "brier"}
    assert cal.height <= 5
    assert cal.height > 0
    assert (cal["mean_predicted"] >= 0).all()
    assert (cal["mean_predicted"] <= 1).all()
    assert (cal["count"] >= 0).all()


def test_calibration_curve_rejects_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    with pytest.raises(ValueError, match="binary-classifier only"):
        source.calibration_curve()


def test_cumulative_gain_includes_baseline():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    gain = source.cumulative_gain()
    assert set(gain.columns) == {"percent_population", "gain", "class"}
    classes_seen = set(gain["class"].unique().to_list())
    assert "baseline" in classes_seen
    # Per-class curves start at (0, 0) and end at (1, 1).
    assert gain.filter(pl.col("class") != "baseline").height > 0


def test_lift_curve_baseline_at_one():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    lift = source.lift_curve()
    assert set(lift.columns) == {"percent_population", "lift", "class"}
    baseline = lift.filter(pl.col("class") == "baseline")
    assert baseline.height == 2
    assert (baseline["lift"] == 1.0).all()


def test_discrimination_threshold_schema_and_grid():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    dt = source.discrimination_threshold(n_thresholds=20)
    assert set(dt.columns) == {
        "threshold",
        "precision",
        "recall",
        "f1",
        "queue_rate",
    }
    assert dt.height == 20
    near_zero = dt.filter(pl.col("threshold") < 0.05)["queue_rate"].to_list()
    assert all(qr > 0.95 for qr in near_zero)
    # queue_rate at threshold=1.0 is small.
    near_one = dt.filter(pl.col("threshold") >= 0.99)["queue_rate"].to_list()
    assert all(qr < 0.05 for qr in near_one)


def test_discrimination_threshold_multiclass_rejects():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    with pytest.raises(ValueError, match="binary-classifier only"):
        source.discrimination_threshold()


def test_discrimination_threshold_cv_averaging():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"], random_state=0)
    dt = source.discrimination_threshold(n_thresholds=10, cv=3)
    assert dt.height == 10
    assert (dt["precision"] >= 0).all() and (dt["precision"] <= 1).all()
    assert (dt["recall"] >= 0).all() and (dt["recall"] <= 1).all()
    # Sort order is ascending by threshold.
    thresholds = dt["threshold"].to_list()
    assert thresholds == sorted(thresholds)


def test_discrimination_threshold_cv_splitter_identity_never_confused():
    """RED-proof for #224(b) repro (a): two DIFFERENT custom splitters must
    never share a cached frame.

    A splitter class that defines ``__eq__`` (disabling the default
    identity ``__hash__``) without overriding ``__repr__`` is unhashable.
    Pre-fix, ``_cache_key`` fell back to ``repr(v)`` for it — CPython's
    default ``object.__repr__`` embeds the memory address, not the
    object's state. Two short-lived temporaries of the same class
    reliably land at the same freed address (measured: 10/10 trials), so
    ``cv=Splitter(3)`` and ``cv=Splitter(7)`` built the SAME cache key and
    the second call silently returned the first's frame. The fix gives
    any still-unhashable-after-normalization value a fresh,
    never-repeating key, so this is now always a genuine cache miss
    regardless of address reuse.
    """

    class _EqOnlySplitter:
        def __init__(self, n_splits):
            self.n_splits = n_splits

        def __eq__(self, other):
            return isinstance(other, _EqOnlySplitter) and other.n_splits == self.n_splits

        def split(self, X, y=None):
            from sklearn.model_selection import KFold

            return KFold(n_splits=self.n_splits, shuffle=True, random_state=0).split(X)

    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"], random_state=0)

    d3 = source.discrimination_threshold(cv=_EqOnlySplitter(3))
    d7 = source.discrimination_threshold(cv=_EqOnlySplitter(7))
    assert d3 is not d7
    assert not d3.equals(d7)


# --- Phase 10e: model-selection methods ---------------------------------


def test_learning_curve_schema_and_splits():
    from sklearn.linear_model import Ridge

    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)
    lc = source.learning_curve(cv=3)
    assert set(lc.columns) == {
        "train_size",
        "split",
        "score",
        "mean_score",
        "std_score",
        "lower",
        "upper",
    }
    assert set(lc["split"].unique().to_list()) == {"train", "test"}
    # 5 default train sizes × 2 splits × 3 folds = 30 rows.
    assert lc.height == 30
    # Per-(train_size, split) mean_score is constant across folds.
    agg = lc.group_by(["train_size", "split"]).agg(
        pl.col("mean_score").n_unique().alias("n_unique"),
    )
    assert (agg["n_unique"] == 1).all()


def test_learning_curve_train_sizes_content_not_repr_keyed():
    """RED-proof for #224(b) repro (b): two ``train_sizes`` arrays that
    numpy's ``repr()`` prints IDENTICALLY (its default ``precision=8``
    rounds away a sub-epsilon difference) must not share a cache slot.
    Pre-fix, an ndarray fell back to ``repr()`` for its cache key, so
    ``learning_curve(train_sizes=[0.2,0.4,0.6,0.8,1.0])`` and the same
    array with the last element nudged by ``1e-12`` built the SAME key
    and the second call silently returned the first's frame.
    """
    from sklearn.linear_model import Ridge

    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)

    ts1 = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    ts2 = np.array([0.2, 0.4, 0.6, 0.8, 1.0 - 1e-12])
    assert repr(ts1) == repr(ts2)  # the collision precondition

    lc1 = source.learning_curve(cv=3, train_sizes=ts1)
    lc2 = source.learning_curve(cv=3, train_sizes=ts2)
    assert lc1 is not lc2


def test_learning_curve_train_sizes_int_vs_float_not_dtype_collapsed():
    """RED-proof for #224(b)/N5 repro (d): an int ``train_sizes`` array and
    an equal-valued float one must not share a cache slot.

    sklearn's ``learning_curve`` reads an int-dtype ``train_sizes`` as
    absolute sample counts and a float-dtype one as fractions of the
    training set — ``train_sizes=[1]`` (one absolute sample) and
    ``train_sizes=[1.0]`` (100% of the training set) are different
    requests. Pre-fix, ``_normalize``'s ``ndarray.tolist()`` step
    preserved the int/float Python types, but the nested tuples
    ``(1,)`` and ``(1.0,)`` still compare and hash equal
    (``1 == 1.0``), so the two calls built the SAME cache key and the
    second silently returned the first's frame despite requesting a
    completely different train size (measured: absolute train_size=1
    vs. the float request's actual train_size=133 on this 200-row
    dataset).
    """
    from sklearn.linear_model import Ridge

    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)

    ts_int = np.array([1])
    ts_float = np.array([1.0])
    assert ts_int.tolist() == [1] and ts_float.tolist() == [1.0]
    assert ts_int.tolist() == ts_float.tolist()  # 1 == 1.0: the collision precondition

    lc_int = source.learning_curve(cv=3, train_sizes=ts_int)
    lc_float = source.learning_curve(cv=3, train_sizes=ts_float)
    assert lc_int is not lc_float
    assert not lc_int.equals(lc_float)
    assert lc_int["train_size"].unique().to_list() == [1]
    assert lc_float["train_size"].unique().to_list() != [1]


def test_validation_curve_schema_and_values():
    from sklearn.linear_model import Ridge

    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)
    vc = source.validation_curve("alpha", [0.1, 1.0, 10.0], cv=3)
    assert set(vc.columns) == {
        "param_value",
        "split",
        "score",
        "mean_score",
        "std_score",
        "lower",
        "upper",
    }
    assert set(vc["param_value"].unique().to_list()) == {0.1, 1.0, 10.0}
    # 3 alphas × 2 splits × 3 folds = 18 rows.
    assert vc.height == 18


def test_cv_scores_schema_and_count():
    from sklearn.linear_model import Ridge

    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)
    cvs = source.cv_scores(cv=3)
    assert set(cvs.columns) == {"fold", "split", "score"}
    assert set(cvs["split"].unique().to_list()) == {"train", "test"}
    # 3 folds × 2 splits = 6 rows.
    assert cvs.height == 6


def test_alpha_selection_schema_and_alphas():
    from sklearn.linear_model import Ridge

    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)
    al = source.alpha_selection([0.1, 1.0, 10.0], cv=3)
    assert set(al.columns) == {
        "alpha",
        "fold",
        "score",
        "mean_score",
        "std_score",
    }
    assert set(al["alpha"].unique().to_list()) == {0.1, 1.0, 10.0}
    # 3 alphas × 3 folds = 9 rows.
    assert al.height == 9


# ---------------------------------------------------------------------------
# Issue-4 regression: ModelSource.compare unknown kwargs validation
# ---------------------------------------------------------------------------


def test_compare_unknown_kwarg_raises_type_error():
    """compare() must reject unknown kwargs with an informative TypeError."""
    df = pl.DataFrame({"a": [1.0, 2.0]})
    source = ferrum.ModelSource(_DuckModel(), df, y=[0, 1])
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ferrum.ModelSource.compare({"m": _DuckModel()}, df, [0, 1], unknown_param=42)


def test_compare_unknown_kwarg_message_lists_accepted():
    """TypeError message must name accepted kwargs."""
    df = pl.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(TypeError, match="random_state"):
        ferrum.ModelSource.compare({"m": _DuckModel()}, df, [0, 1], bad_kwarg="x")


def test_compare_accepted_kwargs_do_not_raise():
    """All documented kwargs must pass through without error."""
    df = pl.DataFrame({"a": [1.0, 2.0]})
    # No assertion needed beyond not raising.
    ferrum.ModelSource.compare(
        {"m": _DuckModel()},
        df,
        [0, 1],
        random_state=0,
        feature_names=["a"],
    )
