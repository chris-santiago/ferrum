"""Phase 10d feature-importance / SHAP / PDP visualizer tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum
from ferrum.composition import ConcatChart
from tests.fixtures import load_dataset, load_fixture
from tests._svg_extents import extents_all_equal, x_axis_extents


@pytest.fixture(scope="module")
def rf_source():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return ferrum.ModelSource(model, X, df["y"], random_state=0)


@pytest.fixture(scope="module")
def ridge_source():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return ferrum.ModelSource(model, X, df["y"], random_state=0)


# --- .importances() -----------------------------------------------------


def test_importances_builtin_rf(rf_source):
    imp = rf_source.importances(method="builtin")
    assert set(imp.columns) == {"feature", "importance", "std", "rank"}
    assert imp.height == 5
    assert imp["rank"][0] == 1
    # builtin std is zero by construction.
    assert float(imp["std"].max()) == pytest.approx(0.0)
    # Sorted by descending |importance|.
    vals = imp["importance"].to_list()
    assert all(abs(vals[i]) >= abs(vals[i + 1]) for i in range(len(vals) - 1))


def test_importances_builtin_coef_path(ridge_source):
    """Ridge exposes coef_ but not feature_importances_, exercising the
    abs(coef) branch."""
    imp = ridge_source.importances(method="builtin")
    assert imp.height == 5
    assert (imp["importance"] >= 0).all()


def test_importances_permutation_populates_std(rf_source):
    imp = rf_source.importances(
        method="permutation",
        n_repeats=5,
        random_state=0,
    )
    assert imp.height == 5
    # At least one feature should have non-zero std after a real permutation run.
    assert float(imp["std"].max()) > 0.0
    assert (imp["std"] >= 0).all()


def test_importances_invalid_method(rf_source):
    with pytest.raises(ValueError, match="ModelSource.importances.*must be one of"):
        rf_source.importances(method="not_a_method")


def test_importances_no_capability_raises():
    """A model that exposes neither feature_importances_ nor coef_ should
    raise AttributeError on builtin."""

    class _DumbModel:
        def predict(self, X):
            return np.zeros(len(X))

    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    src = ferrum.ModelSource(_DumbModel(), X, df["y"])
    with pytest.raises(AttributeError, match="feature_importances_"):
        src.importances(method="builtin")


def test_importances_caches_per_method(rf_source):
    """Builtin and permutation share a method+kwargs cache key; different
    methods must NOT collide."""
    a = rf_source.importances(method="builtin")
    b = rf_source.importances(method="permutation", n_repeats=3, random_state=0)
    # Different std columns prove the two calls produced independent frames.
    assert float(a["std"].sum()) == 0.0
    assert float(b["std"].sum()) > 0.0


# --- importance_chart figure function -----------------------------------


def test_importance_chart_figure_function():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.importance_chart(model, X, df["y"])
    svg = chart.to_svg()
    assert "<svg" in svg
    # 5 features → 5 horizontal bars + chart-border / clip rects.
    assert svg.count("<rect ") >= 5


def test_importance_chart_top_k_truncates():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.importance_chart(model, X, df["y"], top_k=2)
    # Inspect underlying data: only 2 bars worth of rows.
    assert chart._data.height == 2


def test_importance_chart_vertical_orient():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.importance_chart(model, X, df["y"], orient="vertical")
    svg = chart.to_svg()
    assert "<svg" in svg


def test_mark_importance_invalid_orient():
    df = pl.DataFrame(
        {
            "feature": ["a", "b"],
            "importance": [0.5, 0.3],
            "std": [0.0, 0.0],
            "imp_lower": [0.5, 0.3],
            "imp_upper": [0.5, 0.3],
            "rank": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="mark_importance.*orient.*must be one of"):
        ferrum.Chart(df).mark_importance(orient="diagonal").to_svg()


# --- FeatureImportancesVisualizer ---------------------------------------


def test_feature_importances_visualizer_fit_and_repr():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    viz = ferrum.FeatureImportancesVisualizer(model).fit(X, df["y"])
    assert "top_feature_importance" in repr(viz)
    chart = viz.show()
    assert "<svg" in chart.to_svg()


def test_feature_importances_visualizer_unfit_raises():
    model = load_fixture("regression_rf")
    viz = ferrum.FeatureImportancesVisualizer(model)
    with pytest.raises(RuntimeError, match="must be fit"):
        viz.show()


def test_feature_importances_visualizer_permutation():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    viz = ferrum.FeatureImportancesVisualizer(
        model,
        method="permutation",
        random_state=0,
    ).fit(X, df["y"])
    chart = viz.show()
    assert "<svg" in chart.to_svg()


# --- ModelSource.shap_values() -----------------------------------------


def _ridge_source():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return ferrum.ModelSource(model, X, df["y"], random_state=0), X, df["y"]


def test_shap_values_schema():
    source, _X, _y = _ridge_source()
    sv = source.shap_values()
    assert set(sv.columns) == {
        "sample_id",
        "feature",
        "shap_value",
        "feature_value",
        "feature_value_normalized",
        "class_label",
    }
    assert sv["feature"].n_unique() == 5
    # Regression: class_label is "target" on every row.
    assert sv["class_label"].unique().to_list() == ["target"]
    assert sv.height == sv["sample_id"].n_unique() * sv["feature"].n_unique()


def test_shap_values_multiclass_schema():
    """Multi-class classifiers emit one row per (sample, feature, class)."""
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"], random_state=0)
    sv = source.shap_values()
    assert set(sv.columns) == {
        "sample_id",
        "feature",
        "shap_value",
        "feature_value",
        "feature_value_normalized",
        "class_label",
    }
    n_features = sv["feature"].n_unique()
    n_samples = sv["sample_id"].n_unique()
    n_classes = sv["class_label"].n_unique()
    assert n_classes == 3
    assert sv.height == n_samples * n_features * n_classes


def test_shap_values_caches():
    source, _, _ = _ridge_source()
    a = source.shap_values()
    b = source.shap_values()
    # Cache returns the same DataFrame object (identity) — the second call
    # must not recompute.
    assert a is b


# --- shap_chart figure dispatcher --------------------------------------


def test_shap_chart_beeswarm():
    source, _, _ = _ridge_source()
    chart = ferrum.shap_chart(source, kind="beeswarm")
    svg = chart.to_svg()
    assert "<svg" in svg
    # 200 samples × 5 features = 1000 points.
    assert svg.count("<circle ") == 1000


def test_shap_chart_bar():
    source, _, _ = _ridge_source()
    chart = ferrum.shap_chart(source, kind="bar")
    assert "<svg" in chart.to_svg()


def test_shap_chart_waterfall():
    source, _, _ = _ridge_source()
    chart = ferrum.shap_chart(source, kind="waterfall", sample_idx=3)
    assert "<svg" in chart.to_svg()


def test_shap_chart_waterfall_requires_sample_idx():
    source, _, _ = _ridge_source()
    with pytest.raises(ValueError, match="sample_idx"):
        ferrum.shap_chart(source, kind="waterfall")


def test_shap_chart_waterfall_bad_sample_idx():
    source, _, _ = _ridge_source()
    with pytest.raises(ValueError, match="not found"):
        ferrum.shap_chart(source, kind="waterfall", sample_idx=999_999)


def _multiclass_source():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    return ferrum.ModelSource(model, X, df["y"], random_state=0)


def test_shap_beeswarm_chart_per_class_multiclass():
    """per_class=True on a multi-class classifier facets by class_label."""
    source = _multiclass_source()
    chart = ferrum.shap_beeswarm_chart(source, per_class=True)
    svg = chart.to_svg()
    assert "<svg" in svg
    # Per-class beeswarm renders one panel per class; n_circles equals
    # n_samples * n_features * n_classes.
    sv = source.shap_values()
    expected = sv.height  # already n_samples * n_features * n_classes
    assert svg.count("<circle ") == expected


def test_shap_bar_chart_per_class_multiclass():
    source = _multiclass_source()
    chart = ferrum.shap_bar_chart(source, per_class=True)
    assert "<svg" in chart.to_svg()


def test_shap_bar_chart_per_class_uses_global_feature_set():
    """RED-proof: a class-blind ``head()`` over ``(class_label, feature)``
    pairs can hand different feature sets to different class panels when
    one class's magnitudes dominate the global sort tail. ``shap_bar``
    must instead rank features globally once (matching beeswarm/waterfall)
    and project the SAME top-``max_display`` set onto every class.

    Class A's shap magnitudes are large across all 5 features; class B's
    are small everywhere except f3/f4. A class-blind top-6 (max_display=3
    * 2 classes) over the flattened (class, feature) pairs picks
    {A: f0,f1,f2,f3; B: f3,f4} -- disagreeing feature sets, and dropping
    f0/f1/f2 entirely from B's panel. The global mean(|shap_value|) per
    feature across BOTH classes ranks f0/f1/f2 highest (50.5/45.45/40.4
    vs f3/f4's 30/22.5), so the fixed selection must give every class
    panel exactly {f0, f1, f2}.
    """
    from ferrum.plots.explanation import _shap_bar_chart_from_source

    sv = pl.DataFrame(
        {
            "class_label": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "feature": ["f0", "f1", "f2", "f3", "f4", "f0", "f1", "f2", "f3", "f4"],
            "shap_value": [100.0, 90.0, 80.0, 10.0, 5.0, 1.0, 0.9, 0.8, 50.0, 40.0],
        }
    )

    class _FakeSource:
        def shap_values(self, background=None):
            return sv

    chart = _shap_bar_chart_from_source(_FakeSource(), max_display=3, per_class=True)
    df = chart._data
    classes = df["class_label"].unique(maintain_order=True).to_list()
    assert len(classes) == 2

    per_class_features = {
        cls: set(df.filter(pl.col("class_label") == cls)["feature"].to_list()) for cls in classes
    }
    feature_sets = list(per_class_features.values())
    assert feature_sets[0] == feature_sets[1], (
        f"class panels disagree on feature set: {per_class_features}"
    )
    assert feature_sets[0] == {"f0", "f1", "f2"}, (
        f"expected the globally-ranked top-3 feature set, got: {per_class_features}"
    )


def test_shap_waterfall_chart_per_class_multiclass():
    """per_class=True on a multi-class classifier renders one waterfall
    panel per class, each with its OWN cumulative walk anchored at zero --
    not a single cumsum chained across all classes' rows (GH #46).
    """
    source = _multiclass_source()
    chart = ferrum.shap_waterfall_chart(source, sample_idx=3, per_class=True)
    svg = chart.to_svg()
    assert "<svg" in svg

    df = chart._data
    classes = df["class_label"].unique(maintain_order=True).to_list()
    assert len(classes) > 1

    # Faceted by class_label -- one panel per class. `.facet(col=...)`
    # alone is wrap mode, which stores the column under `field`.
    assert chart._facet is not None
    assert chart._facet.field == "class_label"

    n_features_per_class = df.filter(pl.col("class_label") == classes[0]).height
    assert n_features_per_class > 0
    # Total bars = kept features x classes; every class keeps the same
    # globally-ranked feature set (no per-class row drop).
    assert df.height == n_features_per_class * len(classes)
    for cls in classes:
        assert df.filter(pl.col("class_label") == cls).height == n_features_per_class

    # Each class's cumulative walk is independent and zero-anchored: the
    # first bar starts at 0, and each subsequent bar picks up where the
    # previous one left off *within that class* -- this is exactly what a
    # single cumsum chained across all classes' rows would violate for
    # every class after the first.
    for cls in classes:
        panel = df.filter(pl.col("class_label") == cls)
        x0 = panel["x0"].to_list()
        x1 = panel["x1"].to_list()
        shap_values = panel["shap_value"].to_list()
        assert x0[0] == pytest.approx(0.0)
        for i in range(len(x0)):
            assert x1[i] == pytest.approx(x0[i] + shap_values[i])
            if i > 0:
                assert x0[i] == pytest.approx(x1[i - 1])

    # x domain is the union across all classes' x0/x1 extents and is shared
    # (a single explicit scale domain bound at chart level, applied
    # uniformly across every facet panel).
    domain = chart._pending_stat_mark.kwargs["x_scale_domain"]
    x_lo = min(df["x0"].min(), df["x1"].min(), 0.0)
    x_hi = max(df["x0"].max(), df["x1"].max(), 0.0)
    assert domain[0] <= x_lo + 1e-9
    assert domain[1] >= x_hi - 1e-9


def test_shap_waterfall_chart_per_class_false_multiclass_unchanged():
    """per_class=False on a multi-class source stays single-panel with a
    plain (non-partitioned) cumulative sum -- byte-identity invariant for
    the pre-existing path (spec section 7)."""
    source = _multiclass_source()
    chart = ferrum.shap_waterfall_chart(source, sample_idx=3, per_class=False)
    assert chart._facet is None
    df = chart._data
    assert df["class_label"].n_unique() == 1

    shap_values = df["shap_value"]
    expected_x1 = shap_values.cum_sum()
    expected_x0 = pl.concat([pl.Series([0.0]), expected_x1.head(expected_x1.len() - 1)])
    assert df["x1"].to_list() == pytest.approx(expected_x1.to_list())
    assert df["x0"].to_list() == pytest.approx(expected_x0.to_list())


def test_shap_waterfall_chart_per_class_true_single_class_byte_identical():
    """per_class=True on a single-class source (regression/binary) renders
    byte-identically to per_class=False -- no facet is created because
    _should_facet_by_class requires more than one class present."""
    source, _, _ = _ridge_source()
    svg_true = ferrum.shap_waterfall_chart(source, sample_idx=3, per_class=True).to_svg()
    svg_false = ferrum.shap_waterfall_chart(source, sample_idx=3, per_class=False).to_svg()
    assert svg_true == svg_false


def _binary_source():
    """An actual binary classifier (not the regression ridge fixture) --
    binary_logistic routes SHAP through the positive-class slice as a
    single-element class_label list (see _shap_class_labels), so this
    exercises the single-class byte-identity path on a genuinely
    classification-shaped source rather than a regression stand-in."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    return ferrum.ModelSource(model, X, df["y"], random_state=0)


def test_shap_waterfall_chart_per_class_true_binary_byte_identical():
    """per_class=True on a TRUE binary classifier renders byte-identically
    to per_class=False. binary_logistic carries exactly one class_label
    (the positive class), so _should_facet_by_class's >1-class gate never
    opens -- distinct coverage from the regression-ridge single-class test
    above, which never exercises a classifier's class_label collapse."""
    source = _binary_source()
    svg_true = ferrum.shap_waterfall_chart(source, sample_idx=3, per_class=True).to_svg()
    svg_false = ferrum.shap_waterfall_chart(source, sample_idx=3, per_class=False).to_svg()
    assert svg_true == svg_false


def _multiclass_compare_sources():
    """A second, structurally different multiclass classifier for compare=.

    RandomForestClassifier exposes feature_importances_ (routes through
    shap.TreeExplainer) rather than coef_ (LinearExplainer) -- deterministic
    given a fixed random_state, and a genuinely different model from the
    multiclass_logistic base so the two compare= panels are not trivially
    identical.
    """
    from sklearn.ensemble import RandomForestClassifier

    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    base = load_fixture("multiclass_logistic")
    alt = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    alt.fit(X.to_numpy(), y.to_numpy())
    return X, y, base, alt


def test_shap_beeswarm_chart_per_class_compare_shares_x_within_panel():
    """GH #46: per_class=True facets must share the shap-value x domain
    WITHIN each compare= model panel -- a future resolve= edit that flips
    facet x to independent (the pdp-x-collapse failure class from #35)
    must fail this test.
    """
    X, y, base, alt = _multiclass_compare_sources()
    result = ferrum.shap_beeswarm_chart(base, X, y, per_class=True, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2

    for panel in result.charts:
        svg = panel.to_svg()
        extents = x_axis_extents(svg)
        assert len(extents) == 3, (
            f"expected one facet per class (3) in this compare= panel, found {len(extents)}"
        )
        assert extents_all_equal(extents), (
            f"class facets do not share x within this model panel: {extents}"
        )


def test_shap_bar_chart_per_class_compare_shares_x_within_panel():
    """GH #46: same shared-x-within-panel guarantee for shap_bar_chart."""
    X, y, base, alt = _multiclass_compare_sources()
    result = ferrum.shap_bar_chart(base, X, y, per_class=True, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2

    for panel in result.charts:
        svg = panel.to_svg()
        extents = x_axis_extents(svg)
        assert len(extents) == 3, (
            f"expected one facet per class (3) in this compare= panel, found {len(extents)}"
        )
        assert extents_all_equal(extents), (
            f"class facets do not share x within this model panel: {extents}"
        )


def test_shap_chart_invalid_kind():
    source, _, _ = _ridge_source()
    with pytest.raises(ValueError, match="must be one of"):
        ferrum.shap_chart(source, kind="violinplot")


def test_mark_shap_waterfall_requires_sample_idx():
    """Calling mark_shap_waterfall without sample_idx (left at -1 sentinel)
    raises ValueError — the guard lives in Chart.mark_shap_waterfall before
    the desugar is invoked."""
    df = pl.DataFrame(
        {
            "feature": ["a"],
            "x0": [0.0],
            "x1": [1.0],
            "shap_sign": ["positive"],
        }
    )
    with pytest.raises(ValueError, match="sample_idx"):
        ferrum.Chart(df).mark_shap_waterfall().to_svg()


# --- SHAPVisualizer -----------------------------------------------------


def test_shap_visualizer_beeswarm_default():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    viz = ferrum.SHAPVisualizer(model).fit(X, df["y"])
    assert "top_abs_shap" in repr(viz)
    assert "<svg" in viz.show().to_svg()


def test_shap_visualizer_waterfall_requires_sample_idx():
    """fit() raises immediately when kind='waterfall' but sample_idx wasn't
    provided — chart building happens inside fit()."""
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    viz = ferrum.SHAPVisualizer(model, kind="waterfall")
    with pytest.raises(ValueError, match="sample_idx"):
        viz.fit(X, df["y"])


def test_shap_visualizer_invalid_kind():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    viz = ferrum.SHAPVisualizer(model, kind="not_a_kind")
    with pytest.raises(ValueError, match="SHAPVisualizer.*must be one of"):
        viz.fit(X, df["y"])


def test_shap_visualizer_order_affects_bar_and_waterfall():
    """Regression test: `order` must materially affect bar and waterfall
    output, not just beeswarm. Bar values differ between mean / max
    aggregation; waterfall feature ordering follows the global ranking
    chosen by `order` (matching beeswarm), not per-sample |shap|.
    """
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])

    # --- bar: aggregation values must change with `order` ----------------
    bar_mean = ferrum.SHAPVisualizer(model, kind="bar", order="abs_mean").fit(
        X,
        df["y"],
    )
    bar_max = ferrum.SHAPVisualizer(model, kind="bar", order="max").fit(
        X,
        df["y"],
    )
    bar_mean_vals = bar_mean.show()._data["abs_mean_shap"].to_list()
    bar_max_vals = bar_max.show()._data["abs_mean_shap"].to_list()
    assert bar_mean_vals != bar_max_vals
    # max(|shap|) >= mean(|shap|) per feature, so every entry should be
    # strictly larger under order="max" once features are sorted.
    assert all(b >= a for a, b in zip(sorted(bar_mean_vals), sorted(bar_max_vals)))
    assert any(b > a for a, b in zip(sorted(bar_mean_vals), sorted(bar_max_vals)))

    # --- waterfall: feature order must follow the global ranking, not
    # the per-sample |shap| ordering. Sample 6 is a fixture where the
    # per-sample ordering ('f2','f1','f0','f4','f3') diverges from the
    # global mean ordering ('f0','f1','f2','f4','f3').
    wf = ferrum.SHAPVisualizer(
        model,
        kind="waterfall",
        sample_idx=6,
        order="abs_mean",
    ).fit(X, df["y"])
    wf_features = wf.show()._data["feature"].to_list()
    source = ferrum.ModelSource(model, X, df["y"], random_state=0)
    sv = source.shap_values()
    global_rank = (
        sv.group_by("feature")
        .agg(pl.col("shap_value").abs().mean().alias("v"))
        .sort("v", descending=True)["feature"]
        .to_list()
    )
    assert wf_features == global_rank
    # Verify the legacy per-sample ordering would have been different,
    # so the assertion above is meaningful for this fixture.
    per_sample = (
        sv.filter(pl.col("sample_id") == 6)
        .sort(pl.col("shap_value").abs(), descending=True)["feature"]
        .to_list()
    )
    assert per_sample != global_rank


def test_shap_visualizer_waterfall_top_abs_shap_is_per_sample():
    """`_materialize` must compute `top_abs_shap` from only the displayed
    sample when kind='waterfall', not the full dataset.
    """
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"], random_state=0)
    sv = source.shap_values()
    # Per-sample max(|shap|) for sample 3 should differ from the global
    # mean(|shap|) max used in the legacy implementation.
    sample_max = float(sv.filter(pl.col("sample_id") == 3)["shap_value"].abs().max())
    global_mean_max = float(
        sv.group_by("feature").agg(pl.col("shap_value").abs().mean().alias("v"))["v"].max()
    )
    assert sample_max != pytest.approx(global_mean_max)

    viz = ferrum.SHAPVisualizer(model, kind="waterfall", sample_idx=3).fit(
        X,
        df["y"],
    )
    assert viz._metrics["top_abs_shap"] == pytest.approx(sample_max)


# --- ModelSource.partial_dependence() -----------------------------------


def _rf_xy():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return model, X, df["y"]


def test_partial_dependence_schema():
    model, X, y = _rf_xy()
    src = ferrum.ModelSource(model, X, y, random_state=0)
    pd_df = src.partial_dependence(["f0", "f1"], grid_resolution=20)
    assert set(pd_df.columns) == {
        "feature",
        "feature_value",
        "pd_value",
        "sample_id",
    }
    assert set(pd_df["feature"].unique().to_list()) == {"f0", "f1"}
    # sample_id is -1 for kind="average" (marginal curve, not per-sample).
    assert (pd_df["sample_id"] == -1).all()
    # 2 features × 20 grid points.
    assert pd_df.height == 40


def test_partial_dependence_caches():
    model, X, y = _rf_xy()
    src = ferrum.ModelSource(model, X, y)
    a = src.partial_dependence(["f0"], grid_resolution=10)
    b = src.partial_dependence(["f0"], grid_resolution=10)
    assert a is b


def test_partial_dependence_invalid_kind():
    model, X, y = _rf_xy()
    src = ferrum.ModelSource(model, X, y)
    with pytest.raises(ValueError, match="ModelSource.partial_dependence.*must be one of"):
        src.partial_dependence(["f0"], kind="not_a_kind")


def test_partial_dependence_individual_returns_per_sample_rows():
    """ICE/both emit per-sample rows; sample_id ranges [0, n_samples)."""
    model, X, y = _rf_xy()
    src = ferrum.ModelSource(model, X, y)
    ice = src.partial_dependence(["f0"], kind="individual", grid_resolution=10)
    sample_ids = set(ice["sample_id"].to_list())
    # No average row in the individual variant.
    assert -1 not in sample_ids
    assert max(sample_ids) == X.height - 1
    assert ice.height == X.height * 10

    both = src.partial_dependence(["f0"], kind="both", grid_resolution=10)
    both_ids = set(both["sample_id"].to_list())
    # Both variant carries the average rows (-1) AND the per-sample rows.
    assert -1 in both_ids
    assert max(both_ids) == X.height - 1
    assert both.height == (X.height + 1) * 10


# --- pdp_chart figure function -----------------------------------------


def test_pdp_chart_single_feature():
    model, X, y = _rf_xy()
    chart = ferrum.pdp_chart(model, X, y, features=["f0"], grid_resolution=20)
    svg = chart.to_svg()
    assert "<svg" in svg
    assert svg.count("<polyline ") == 1


def test_pdp_chart_multiple_features():
    model, X, y = _rf_xy()
    chart = ferrum.pdp_chart(
        model,
        X,
        y,
        features=["f0", "f1", "f2"],
        grid_resolution=20,
    )
    svg = chart.to_svg()
    # One polyline per feature.
    assert svg.count("<polyline ") == 3


def test_pdp_chart_requires_features():
    model, X, y = _rf_xy()
    with pytest.raises(ValueError, match="features="):
        ferrum.pdp_chart(model, X, y)


def test_pdp_chart_individual_renders_ice_per_sample():
    """kind='individual' produces one polyline per training sample
    per feature panel via mark_style.detail routing on sample_id."""
    model, X, y = _rf_xy()
    chart = ferrum.pdp_chart(
        model,
        X,
        y,
        features=["f0"],
        grid_resolution=10,
        kind="individual",
    )
    svg = chart.to_svg()
    # n_samples polylines for the single feature panel.
    assert svg.count("<polyline ") == X.height


def test_pdp_chart_both_renders_ice_plus_average():
    """kind='both' overlays one thicker average polyline per feature
    on top of the per-sample ICE bundle."""
    model, X, y = _rf_xy()
    chart = ferrum.pdp_chart(
        model,
        X,
        y,
        features=["f0", "f1"],
        grid_resolution=10,
        kind="both",
    )
    svg = chart.to_svg()
    # 2 features * (n_samples ICE + 1 average) polylines.
    assert svg.count("<polyline ") == 2 * (X.height + 1)


def test_pdp_chart_center_starts_at_zero():
    """center=True subtracts the value at min(feature_value) per
    (feature, sample_id) so every polyline starts at 0 at the left."""
    import numpy as np

    model, X, y = _rf_xy()
    from ferrum.plots.explanation import _pdp_chart_from_source

    source = ferrum.ModelSource(model, X, y)
    chart = _pdp_chart_from_source(
        source,
        features=["f0"],
        grid_resolution=10,
        kind="individual",
        center=True,
    )
    # Pull the underlying data and assert all per-sample first values are 0.
    df = chart._data
    first = df.group_by(["feature", "sample_id"]).agg(pl.col("pd_value").first().alias("first"))
    assert np.allclose(first["first"].to_numpy(), 0.0)


def test_pdp_chart_kind_invalid_raises():
    model, X, y = _rf_xy()
    with pytest.raises(ValueError, match="ModelSource.partial_dependence.*must be one of"):
        ferrum.pdp_chart(
            model,
            X,
            y,
            features=["f0"],
            kind="not_a_kind",
        )


# ---------------------------------------------------------------------------
# Issue-1 regression: _shap_order_features ValueError for unknown order
# ---------------------------------------------------------------------------


def test_shap_order_features_unknown_order_raises():
    """_shap_order_features must raise ValueError for unknown order strings."""
    from ferrum.plots.explanation import _shap_order_features

    sv = pl.DataFrame(
        {
            "feature": ["a", "a", "b", "b"],
            "shap_value": [0.1, -0.2, 0.3, 0.4],
            "feature_value": [1.0, 2.0, 3.0, 4.0],
            "sample": [0, 1, 0, 1],
        }
    )
    with pytest.raises(ValueError, match="abs_max"):
        _shap_order_features(sv, order="abs_max", max_display=5)
    with pytest.raises(ValueError, match="must be one of"):
        _shap_order_features(sv, order="unknown", max_display=5)


def test_shap_order_features_accepted_orders_work():
    """Accepted order values must not raise."""
    from ferrum.plots.explanation import _shap_order_features

    sv = pl.DataFrame(
        {
            "feature": ["a", "a", "b", "b"],
            "shap_value": [0.1, -0.2, 0.3, 0.4],
            "feature_value": [1.0, 2.0, 3.0, 4.0],
            "sample": [0, 1, 0, 1],
        }
    )
    result_mean = _shap_order_features(sv, order="abs_mean", max_display=5)
    result_max = _shap_order_features(sv, order="max", max_display=5)
    assert set(result_mean) == {"a", "b"}
    assert set(result_max) == {"a", "b"}
