"""F-L12-01 (batch G, G6) — ``ModelSource`` constructs without ``X``, and
F-L12-02 (batch G, G7) — ``intercluster_distance(method="tsne")`` perplexity
clamp.

F-L12-01: X-free diagnostics (``pca_scree_chart`` / ``intercluster_distance_chart``,
both computed from model attributes alone) render end to end without ``X``.
X-needing diagnostics (``silhouette_chart``, ``manifold_chart``, and every
other ``self._X`` consumer) refuse with a typed domain message naming the
diagnostic and the fix (``ModelSource(model, X)``) — the leaked
``Chart(data=None) requires per-layer data...`` message from
``ferrum._coerce.to_polars`` must never surface from source construction.

F-L12-02: ``intercluster_distance(method="tsne")`` must work for every
``k >= 4`` as documented (see the module's own F-L12-02 section near the
bottom of this file for the perplexity-clamp tests, and
``ModelSource.silhouette()``'s stable-sort tie-ordering pin).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture

# The leaked message this fix must make unreachable from source construction.
_LEAKED_CHART_MESSAGE = "Chart(data=None) requires per-layer data"


def _assert_typed_x_refusal(exc: Exception, diagnostic: str) -> None:
    """Shared assertion: the refusal names the diagnostic and the fix, and
    never leaks ``to_polars(None)``'s unrelated Chart-layer error."""
    text = str(exc)
    assert diagnostic in text
    assert "ModelSource(model, X)" in text
    assert _LEAKED_CHART_MESSAGE not in text


# --- Construction -------------------------------------------------------


def test_modelsource_constructs_without_x():
    model = load_fixture("kmeans_3cluster")
    source = ferrum.ModelSource(model)
    assert source.X is None
    assert source.model is model


def test_modelsource_positional_x_y_unchanged():
    """Backward compat: ``ModelSource(model, X, y)`` positional calls unchanged."""
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    y = df["y"]
    source = ferrum.ModelSource(model, X, y)
    assert source.X is not None
    assert source.X.equals(X)
    assert source.y is not None


# --- X-free diagnostics: pca_scree_chart / intercluster_distance_chart --


def test_pca_scree_chart_renders_without_x():
    model = load_fixture("pca_4comp")
    chart = ferrum.pca_scree_chart(model, threshold=0.90)
    svg = chart.to_svg()
    assert "<svg" in svg


def test_pca_variance_source_method_works_without_x():
    model = load_fixture("pca_4comp")
    source = ferrum.ModelSource(model)
    result = source.pca_variance()
    assert result.height == 4
    np.testing.assert_allclose(
        result["explained_variance_ratio"].to_numpy(),
        model.explained_variance_ratio_,
    )


def test_intercluster_distance_chart_renders_without_x():
    model = load_fixture("kmeans_3cluster")
    chart = ferrum.intercluster_distance_chart(model, method="mds")
    svg = chart.to_svg()
    assert "<svg" in svg


def test_intercluster_distance_source_method_works_without_x():
    model = load_fixture("kmeans_3cluster")
    source = ferrum.ModelSource(model)
    result = source.intercluster_distance(3, method="mds")
    assert result.height == 3


# --- X-needing diagnostics: typed refusal, not the leaked Chart error ---


def test_silhouette_chart_without_x_raises_typed_message():
    model = load_fixture("kmeans_3cluster")
    with pytest.raises(ValueError) as excinfo:
        ferrum.silhouette_chart(model)
    _assert_typed_x_refusal(excinfo.value, "silhouette")


def test_manifold_chart_without_x_raises_typed_message():
    model = load_fixture("kmeans_3cluster")
    with pytest.raises(ValueError) as excinfo:
        ferrum.manifold_chart(model)
    _assert_typed_x_refusal(excinfo.value, "embeddings")


def test_source_silhouette_without_x_raises_typed_message():
    model = load_fixture("kmeans_3cluster")
    source = ferrum.ModelSource(model)
    with pytest.raises(ValueError) as excinfo:
        source.silhouette()
    _assert_typed_x_refusal(excinfo.value, "silhouette")


def test_source_embeddings_without_x_raises_typed_message():
    model = load_fixture("kmeans_3cluster")
    source = ferrum.ModelSource(model)
    with pytest.raises(ValueError) as excinfo:
        source.embeddings()
    _assert_typed_x_refusal(excinfo.value, "embeddings")


def test_source_predictions_without_x_raises_typed_message():
    model = load_fixture("regression_ridge")
    source = ferrum.ModelSource(model, y=[0.0, 1.0])
    with pytest.raises(ValueError) as excinfo:
        source.predictions()
    _assert_typed_x_refusal(excinfo.value, "predictions")


def test_source_probabilities_without_x_raises_typed_message():
    model = load_fixture("regression_ridge")
    source = ferrum.ModelSource(model)
    with pytest.raises(ValueError) as excinfo:
        source.probabilities()
    _assert_typed_x_refusal(excinfo.value, "probabilities")


def test_source_rank1d_without_x_raises_typed_message():
    class _DuckModel:
        def predict(self, X):
            return np.zeros(len(X))

    source = ferrum.ModelSource(_DuckModel())
    with pytest.raises(ValueError) as excinfo:
        source.rank1d()
    _assert_typed_x_refusal(excinfo.value, "rank1d")


def test_source_learning_curve_without_x_raises_typed_message():
    model = load_fixture("regression_ridge")
    source = ferrum.ModelSource(model, y=[0.0, 1.0, 2.0])
    with pytest.raises(ValueError) as excinfo:
        source.learning_curve()
    _assert_typed_x_refusal(excinfo.value, "learning_curve")


def test_source_shap_values_without_x_raises_typed_message():
    model = load_fixture("regression_ridge")
    source = ferrum.ModelSource(model)
    with pytest.raises(ValueError) as excinfo:
        source.shap_values()
    _assert_typed_x_refusal(excinfo.value, "shap_values")


def test_source_partial_dependence_without_x_raises_typed_message():
    """Integer features -- exercises the X guard alone, without touching
    string feature-name resolution."""
    model = load_fixture("regression_ridge")
    source = ferrum.ModelSource(model)
    with pytest.raises(ValueError) as excinfo:
        source.partial_dependence([0])
    _assert_typed_x_refusal(excinfo.value, "partial_dependence")


def test_source_partial_dependence_without_x_string_feature_raises_typed_message():
    """String features on a no-``feature_names``, X-less source: the X
    guard must run before ``self._feature_names.index(f)``, or this raises
    the untyped ``ValueError: 'f0' is not in list`` instead of the typed
    X refusal (the exact guard-ordering hole the source method must close)."""
    model = load_fixture("regression_ridge")
    source = ferrum.ModelSource(model)
    with pytest.raises(ValueError) as excinfo:
        source.partial_dependence(["f0"])
    _assert_typed_x_refusal(excinfo.value, "partial_dependence")


def test_decision_boundary_chart_x_less_source_raises_typed_message():
    """``_resolve_source`` returns an already-constructed ``ModelSource``
    verbatim and drops the positional ``X`` argument (``plots/_helpers.py``),
    so ``decision_boundary_chart(ModelSource(model), X)`` reaches the
    builder with an X-less source: this must raise the typed refusal, not
    ``AttributeError: 'NoneType' object has no attribute 'to_numpy'``."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select([c for c in df.columns if c != "y"])
    source = ferrum.ModelSource(model)
    with pytest.raises(ValueError) as excinfo:
        ferrum.decision_boundary_chart(source, X)
    text = str(excinfo.value)
    assert "decision_boundary_chart" in text
    assert "ModelSource(model, X)" in text
    assert _LEAKED_CHART_MESSAGE not in text


# ``importances(method="builtin")`` feature-names guard -- truth table.
#
# ``imp`` (the model's importance vector) always has 5 entries
# (``regression_rf`` was fit on 5 features). The guard's discriminator is
# name AVAILABILITY (``not self._feature_names``), never X-presence --
# X is just one of two ways names arrive (``X.columns`` or an explicit
# ``feature_names=`` kwarg), and either can independently be absent or
# merely too short:
#
#   quadrant | X          | feature_names=   | len(_feature_names) | message
#   ---------|------------|------------------|----------------------|--------
#   B        | None       | not given        | 0 (nothing at all)  | "requires feature names"
#   A        | 3-of-5 cols| not given        | 3 (from X.columns)  | mismatch (5 vs 3)
#   C        | None       | ["a","b"] (2)    | 2 (from kwarg)      | mismatch (5 vs 2)
#   D        | any        | full-length (5)  | 5 == len(imp)       | no guard fires -- renders
#
# Quadrant C is the one a prior version of this guard got wrong: it
# branched on ``self._X is None`` instead of name availability, so an
# X-less source with a short explicit ``feature_names=`` (quadrant C) was
# told "construct with ModelSource(model, X) or ModelSource(model,
# feature_names=[...])" -- advice it had already taken. Each test below
# pins one row; the mismatch tests (A, C) additionally assert the
# "requires feature names" text is absent, and the no-names test (B)
# asserts the mismatch-specific text is absent, so neither message can
# silently take over the other's row.


def test_importance_chart_builtin_no_names_at_all_raises_requires_feature_names_message():
    """Quadrant B: no X, no ``feature_names=`` -- nothing to label with at
    all. ``importances(method="builtin")`` is X-free by *capability*
    (``feature_importances_`` alone is enough to compute ``imp``), but
    still needs feature *names* to label rows, so this must refuse typed,
    not raise ``IndexError: list index out of range``."""
    model = load_fixture("regression_rf")
    source = ferrum.ModelSource(model)
    with pytest.raises(ValueError) as excinfo:
        source.importances()
    text = str(excinfo.value)
    assert "requires feature names" in text
    assert "ModelSource(model, X)" in text
    assert "feature_names=" in text
    # Must NOT read as the mismatch message -- there are no names to count.
    assert "importance value(s) from the model but only" not in text


def test_importance_chart_builtin_x_derived_names_too_short_raises_mismatch_message():
    """Quadrant A: the model was fit on 5 features but ``X`` here only
    carries 3 columns -- an X-ful count mismatch. The guard must not tell
    this caller to "construct with ModelSource(model, X)"; they already
    did. The message must name the mismatch (both counts) instead."""
    model = load_fixture("regression_rf")
    df = load_dataset("regression").select(["f0", "f1", "f2"])
    source = ferrum.ModelSource(model, df)
    with pytest.raises(ValueError) as excinfo:
        source.importances()
    text = str(excinfo.value)
    assert "5" in text
    assert "3" in text
    assert "requires feature names" not in text
    assert "construct with ModelSource(model, X)" not in text


def test_importance_chart_builtin_supplied_feature_names_too_short_raises_mismatch_message():
    """Quadrant C (the regression case): X is ``None``, but an explicit
    ``feature_names=`` was supplied with too few names (2 for a
    5-importance model). Names exist -- from ``feature_names=``, not
    ``X`` -- so this is the same count-mismatch fault as quadrant A, not
    a missing-names fault, even though ``X`` is absent. The guard must
    not tell this caller to pass ``feature_names=[...]``; they already
    did, and must not claim "requires feature names" -- it doesn't."""
    model = load_fixture("regression_rf")
    source = ferrum.ModelSource(model, feature_names=["a", "b"])
    with pytest.raises(ValueError) as excinfo:
        source.importances()
    text = str(excinfo.value)
    assert "5" in text
    assert "2" in text
    assert "requires feature names" not in text
    assert "construct with ModelSource(model, X)" not in text
    assert "construct with ModelSource(model, feature_names=" not in text


def test_classification_report_visualizer_x_less_fit_raises_typed_message():
    """``ClassificationReportVisualizer(model).fit(None, y)`` forwards
    ``X=None`` straight into ``ModelSource`` (``FerrumVisualizer.fit``),
    building an X-less source; ``_materialize`` must refuse typed before
    ``self._source.model.predict(self._source.X)`` hands sklearn a bare
    ``None`` and leaks "Expected 2D array, got scalar array instead:
    array=None..."."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    y = df["y"]
    viz = ferrum.ClassificationReportVisualizer(model)
    with pytest.raises(ValueError) as excinfo:
        viz.fit(None, y)
    text = str(excinfo.value)
    assert "ClassificationReportVisualizer" in text
    assert "fit(X, y)" in text
    assert "Expected 2D array" not in text


def test_importance_chart_builtin_x_less_with_feature_names_works():
    """The same X-less source with explicit ``feature_names=`` renders
    fine -- confirms the fix above is a guard, not a redesign that breaks
    the genuinely-X-free capability path."""
    model = load_fixture("regression_rf")
    source = ferrum.ModelSource(model, feature_names=["f0", "f1", "f2", "f3", "f4"])
    result = source.importances()
    assert result.height == 5
    assert set(result["feature"].to_list()) == {"f0", "f1", "f2", "f3", "f4"}


# --- X-ful calls stay byte-identical -------------------------------------


def test_pca_scree_chart_x_ful_still_works():
    """Guards against a regression where the X-optional change breaks the
    ordinary X-ful construction path."""
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart_with_x = ferrum.pca_scree_chart(model, df, threshold=0.90)
    chart_without_x = ferrum.pca_scree_chart(model, threshold=0.90)
    # Both render; the X-ful call and the X-less call produce the same
    # scree bars, since the fitted PCA's explained_variance_ratio_ is the
    # source of truth in both cases (X is unused on this capability branch).
    assert chart_with_x.to_svg() == chart_without_x.to_svg()


def test_intercluster_distance_chart_x_ful_matches_x_less():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    chart_with_x = ferrum.intercluster_distance_chart(model, df, method="mds")
    chart_without_x = ferrum.intercluster_distance_chart(model, method="mds")
    assert chart_with_x.to_svg() == chart_without_x.to_svg()


# ---------------------------------------------------------------------------
# F-L12-02 (batch G, G7) -- intercluster_distance(method="tsne") perplexity
# clamp. Unrelated to this file's F-L12-01 X-optional scope, but placed here
# per batch-G task-5 footprint: this is where the existing ModelSource-level
# intercluster_distance tests live (test_decomposition.py only exercises
# ``_core.tsne_embedding`` directly, below the clamp this bug lives in).
#
# t-SNE's kNN size for k embedded centers is k-1 (each center's k-1 peers);
# perplexity must be strictly less than that. The old clamp,
# ``min(5, k-1)``, hit that ceiling exactly for k in {4, 5, 6} and raised
# ``tsne_embedding failed: perplexity (n) must be strictly less than the
# kNN size (n)`` -- despite the docstring promising every k >= 4 works. The
# fix is ``min(5, k-2)``.
# ---------------------------------------------------------------------------


class _KCentersModel:
    """Minimal duck-typed clusterer: only ``cluster_centers_``/``labels_``,
    exactly what ``intercluster_distance(method="tsne")`` reads. No fixture
    exists for k in {4, 5, 6, 7} clusters, so this stub -- built directly
    from the real k requested -- stays on the production
    ``ModelSource.intercluster_distance`` path rather than hand-built output
    shapes."""

    def __init__(self, k: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.cluster_centers_ = rng.rand(k, 3)
        self.labels_ = np.arange(k)


@pytest.mark.parametrize("k", [4, 5, 6, 7])
def test_intercluster_distance_tsne_renders_for_every_k_ge_4(k):
    """RED pre-fix for k in {4, 5, 6}: the old ``min(5, k-1)`` clamp hit the
    kNN-size ceiling exactly and raised. k=7 already worked pre-fix (clamp
    saturates at 5 < 6); kept in the parametrization as a non-regression
    control."""
    model = _KCentersModel(k)
    source = ferrum.ModelSource(model, X=None)
    result = source.intercluster_distance(k, method="tsne")
    assert result.height == k
    assert set(result["cluster"].to_list()) == {str(i) for i in range(k)}
    # Pin the fix's outcome (a real embedding), not just its non-raising:
    # a NaN or all-identical embedding would still pass the two asserts
    # above.
    xy = result.select("x", "y").to_numpy()
    assert np.isfinite(xy).all()
    assert len(set(map(tuple, xy))) == k


def test_intercluster_distance_chart_tsne_renders_for_k4():
    """Same fix, exercised through the documented public entry point
    (``intercluster_distance_chart``, not just the source method)."""
    model = _KCentersModel(4)
    chart = ferrum.intercluster_distance_chart(model, method="tsne", k=4)
    svg = chart.to_svg()
    assert "<svg" in svg


def test_intercluster_distance_tsne_k2_and_k3_still_refuse():
    """The k >= 4 floor is untouched by the clamp fix -- still refuses
    below it, unchanged from the pre-existing k<4 pin."""
    for k in (2, 3):
        model = _KCentersModel(4)  # centers available; k requested is what's tested
        source = ferrum.ModelSource(model, X=None)
        with pytest.raises(ValueError, match="at least 4"):
            source.intercluster_distance(k, method="tsne")


# --- silhouette() tie-ordering (carried-forward G1-sweep rider) ----------
#
# ``argsort(-vals)`` in ``ClusteringMixin.silhouette()`` changed from the
# default quicksort to ``kind="stable"`` (batch G, T5). Duplicate rows in a
# real feature matrix produce exact silhouette-value ties through the real
# Rust kernel, and ``silhouette()`` emits ``y_position`` in row order, so
# tie ordering decides ``mark_silhouette``'s rendered bar order -- this is a
# user-observable behavior change, not an internal-only sort. Mirrors T1's
# precedent at ``tests/diagnostics/test_metric_correctness.py:278-293``
# (a deliberately tie-heavy frame pinning the stable ranking through the
# real frame builder), applied to the real ``ModelSource.silhouette()``
# path instead of a hand-built frame.
#
# 40 real, distinct points from the clustering fixture, tiled 12x -> 480
# rows / ~144-192 per cluster. Exact duplicates get exactly tied silhouette
# values (each duplicate's distance profile to the rest of the dataset is
# the same multiset), and at this size numpy's introsort has long left the
# small-array insertion-sort regime, so an unstable sort's tie order is
# genuinely arbitrary rather than accidentally ascending.


class _TiledPredictStub:
    """Duck-typed model with ``predict`` only (no ``labels_``), so
    ``silhouette()`` takes the ``predict(X)`` capability branch against the
    tiled ``X`` instead of the fixture model's original 200-row
    ``labels_``."""

    def __init__(self, tiled_labels: np.ndarray):
        self._tiled_labels = tiled_labels

    def predict(self, X):
        assert len(X) == len(self._tiled_labels)
        return self._tiled_labels


def _tiled_tie_frame() -> tuple["pl.DataFrame", np.ndarray]:
    df = load_dataset("clustering")
    base_labels = np.asarray(load_fixture("kmeans_3cluster").labels_)[:40]
    base_X = df.head(40).to_numpy()
    tiled_X = np.tile(base_X, (12, 1))
    tiled_labels = np.tile(base_labels, 12)
    X = pl.DataFrame({f"f{j}": tiled_X[:, j] for j in range(tiled_X.shape[1])})
    return X, tiled_labels


def _tie_groups_ascend(sil: "pl.DataFrame") -> bool:
    """True iff, within every run of equal ``silhouette_value`` (the frame
    is sorted descending by value, so ties are contiguous), ``sample_id``
    is ascending. Also asserts at least one tie group exists per cluster,
    since an assertion that never exercises a tie proves nothing."""
    for c in sorted(set(sil["cluster"].to_list())):
        sub = sil.filter(pl.col("cluster") == c)
        values = sub["silhouette_value"].to_list()
        ids = sub["sample_id"].to_list()
        saw_tie = False
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[j + 1] == values[i]:
                j += 1
            if j > i:
                saw_tie = True
                group = ids[i : j + 1]
                if group != sorted(group):
                    return False
            i = j + 1
        if not saw_tie:
            raise AssertionError(f"cluster {c!r} has no tied silhouette_value group")
    return True


def test_silhouette_tie_ordering_is_stable_ascending():
    """RED pre-fix (``kind="quicksort"``): duplicated rows produce real
    ties whose ``sample_id`` order the unstable sort scrambles. GREEN
    post-fix (``kind="stable"``): every tie group's ``sample_id`` ascends."""
    X, tiled_labels = _tiled_tie_frame()
    source = ferrum.ModelSource(_TiledPredictStub(tiled_labels), X)
    sil = source.silhouette()
    assert _tie_groups_ascend(sil)
