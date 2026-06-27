"""T3.5 (D-COMPARE-1): ``compare=`` is accepted for signature uniformity across
the whole model-diagnostic family.

The classification family (``roc_chart``, ``pr_chart``, ``calibration_chart``,
...) genuinely *supports* ``compare=`` (overlaid curves); those tests live in
``tests/diagnostics/test_compare.py``.

Two regression charts support ``compare=`` on their scatter path by overlay,
because their marks already accept a ``color_field`` exactly like the
classification family: ``prediction_error_chart`` (default path, no band) and
``residuals_chart`` (``panels="single"``). Both detect the ``model`` column on
the resolved compare-source frame and drive a per-model colour group.

Their single-model-aggregate paths instead render **small multiples** (one panel
per model, composed as a ``ConcatChart`` with shared x/y scales) rather than
rejecting: ``prediction_error_chart`` with ``ci`` / ``reference_band`` builds one
panel per model so each band is computed from that model's residuals only (never
pooled), and ``residuals_chart`` with a multi-panel ``panels`` value renders one
full diagnostic grid per model. ``cooks_distance_chart`` likewise renders one
residuals-vs-leverage panel per model.

Six explanation charts (``importance_chart``, ``shap_beeswarm_chart``,
``shap_bar_chart``, ``shap_waterfall_chart``, ``shap_chart``, ``pdp_chart``)
render **small multiples** when ``compare=`` is passed: one panel per model,
composed as a ``ConcatChart`` with shared x/y scales.

Four model-selection charts (``learning_curve_chart``,
``validation_curve_chart``, ``cv_scores_chart``, ``alpha_selection_chart``)
also render **small multiples** when ``compare=`` is passed: one panel per
model, composed as a ``ConcatChart`` with shared x/y scales. The internal
train/test coloring is preserved per panel.

The remaining charts across clustering stay excluded and reject with
``ValueError``.

Each test also asserts the no-compare default path still works (``compare=None``
is byte-equivalent to omitting the kwarg), and the ``compare=<dict>`` rejection
tests would fail on the pre-fix signature (``TypeError`` — unexpected kwarg),
proving the parameter was added.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum
from ferrum.composition import ConcatChart
from tests.fixtures import load_dataset, load_fixture


_REG_FEATURES = ["f0", "f1", "f2", "f3", "f4"]


def _reg_setup():
    df = load_dataset("regression")
    X = df.select(_REG_FEATURES)
    y = df["y"]
    base = load_fixture("regression_ridge")
    alt = load_fixture("regression_rf")
    return X, y, base, alt


# ---------------------------------------------------------------------------
# regression.py — prediction_error_chart and residuals_chart overlay compare=
# on their scatter paths; their single-model-aggregate paths
# (ci=/reference_band=, multi-panel) and cooks_distance_chart render small
# multiples (one ConcatChart panel per model) instead.
# ---------------------------------------------------------------------------


def _model_series(chart) -> list[str]:
    """Distinct ``model`` colour-group values carried by a built chart's data."""
    data = chart._data
    assert data is not None
    assert "model" in data.columns
    return sorted(data["model"].unique().to_list())


def test_prediction_error_chart_compare_supported():
    """Default-path prediction_error_chart overlays one actual-vs-predicted
    series per model via a ``model`` colour group."""
    X, y, base, alt = _reg_setup()
    chart = ferrum.prediction_error_chart(base, X, y, compare={"alt": alt})
    assert "<svg" in chart.to_svg()
    assert _model_series(chart) == ["alt", "base"]


def test_prediction_error_chart_compare_drives_color_field():
    """The resolved spec routes colour through the ``model`` field; the
    single-model default carries no ``model`` field reference at all."""
    X, y, base, alt = _reg_setup()
    compared_json = ferrum.prediction_error_chart(base, X, y, compare={"alt": alt}).to_json()
    single_json = ferrum.prediction_error_chart(base, X, y).to_json()
    assert "model" in compared_json
    assert "model" not in single_json


def test_prediction_error_chart_compare_with_ci_renders_small_multiples():
    """The residual confidence band is a single-model aggregate, so compare=
    with ci= renders one panel per model (each band computed from that model's
    residuals only) rather than overlaying a pooled band."""
    X, y, base, alt = _reg_setup()
    result = ferrum.prediction_error_chart(base, X, y, ci=0.9, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_prediction_error_chart_compare_with_reference_band_renders_small_multiples():
    X, y, base, alt = _reg_setup()
    result = ferrum.prediction_error_chart(base, X, y, reference_band=True, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_prediction_error_chart_compare_with_ci_none_byte_identical():
    """compare=None on the ci= path is byte-identical to omitting the kwarg."""
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.prediction_error_chart(base, X, y, ci=0.9, compare=None).to_svg()
    without_kwarg = ferrum.prediction_error_chart(base, X, y, ci=0.9).to_svg()
    assert with_kwarg == without_kwarg


def test_prediction_error_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    chart = ferrum.prediction_error_chart(base, X, y, compare=None)
    assert "<svg" in chart.to_svg()
    # Single-model default: no model colour group injected.
    assert chart._data is not None
    assert "model" not in chart._data.columns


def test_residuals_chart_compare_supported_single_panel():
    """panels="single" residuals_chart overlays one residuals-vs-fitted series
    per model via a ``model`` colour group."""
    X, y, base, alt = _reg_setup()
    chart = ferrum.residuals_chart(base, X, y, panels="single", compare={"alt": alt})
    assert "<svg" in chart.to_svg()
    assert _model_series(chart) == ["alt", "base"]


def test_residuals_chart_compare_default_panels_none_supported():
    """panels=None is treated as single-panel and likewise supports compare=."""
    X, y, base, alt = _reg_setup()
    chart = ferrum.residuals_chart(base, X, y, panels=None, compare={"alt": alt})
    assert "<svg" in chart.to_svg()
    assert _model_series(chart) == ["alt", "base"]


def test_residuals_chart_compare_multipanel_renders_small_multiples():
    """The 4-panel QQ/scale-location/leverage grid is single-model, so compare=
    with a multi-panel layout renders one full diagnostic grid per model
    (nested small multiples) rather than overlaying."""
    X, y, base, alt = _reg_setup()
    result = ferrum.residuals_chart(base, X, y, panels="auto", compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_residuals_chart_compare_explicit_panel_list_renders_small_multiples():
    X, y, base, alt = _reg_setup()
    result = ferrum.residuals_chart(
        base, X, y, panels=["residuals_vs_fitted", "qq"], compare={"alt": alt}
    )
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_residuals_chart_compare_multipanel_none_byte_identical():
    """compare=None on the multi-panel path is byte-identical to omitting it."""
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.residuals_chart(base, X, y, panels="auto", compare=None).to_svg()
    without_kwarg = ferrum.residuals_chart(base, X, y, panels="auto").to_svg()
    assert with_kwarg == without_kwarg


def test_residuals_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    chart = ferrum.residuals_chart(base, X, y, panels="single", compare=None)
    assert "<svg" in chart.to_svg()


def test_cooks_distance_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one residuals-vs-leverage panel per
    model. Cook's distance / leverage need ``coef_``, so both models are linear
    (the RandomForest fixture has no hat matrix)."""
    from sklearn.linear_model import Ridge

    X, y, base, _ = _reg_setup()
    alt = Ridge(alpha=0.1).fit(X.to_numpy(), y.to_numpy())
    result = ferrum.cooks_distance_chart(base, X, y, threshold="auto", compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_cooks_distance_chart_compare_none_byte_identical():
    """compare=None on cooks_distance_chart is byte-identical to omitting it."""
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.cooks_distance_chart(base, X, y, threshold="auto", compare=None).to_svg()
    without_kwarg = ferrum.cooks_distance_chart(base, X, y, threshold="auto").to_svg()
    assert with_kwarg == without_kwarg


def test_cooks_distance_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    chart = ferrum.cooks_distance_chart(base, X, y, threshold="auto", compare=None)
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# model_selection.py — learning / validation / cv_scores / alpha_selection
# ---------------------------------------------------------------------------


def test_learning_curve_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one panel per model (base + alt)."""
    X, y, base, alt = _reg_setup()
    result = ferrum.learning_curve_chart(base, X, y, cv=3, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_learning_curve_chart_compare_none_byte_identical():
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.learning_curve_chart(base, X, y, cv=3, compare=None).to_svg()
    without_kwarg = ferrum.learning_curve_chart(base, X, y, cv=3).to_svg()
    assert with_kwarg == without_kwarg


def test_validation_curve_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one panel per model (base + alt).

    Both models must support the swept parameter; two Ridge instances satisfy
    this (RandomForestRegressor has no ``alpha`` param).
    """
    from sklearn.linear_model import Ridge

    X, y, base, _ = _reg_setup()
    alt = Ridge(alpha=0.1)
    result = ferrum.validation_curve_chart(
        base, X, y, param="alpha", values=[0.1, 1.0, 10.0], cv=3, compare={"alt": alt}
    )
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_validation_curve_chart_compare_none_byte_identical():
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.validation_curve_chart(
        base, X, y, param="alpha", values=[0.1, 1.0, 10.0], cv=3, compare=None
    ).to_svg()
    without_kwarg = ferrum.validation_curve_chart(
        base, X, y, param="alpha", values=[0.1, 1.0, 10.0], cv=3
    ).to_svg()
    assert with_kwarg == without_kwarg


def test_cv_scores_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one panel per model (base + alt)."""
    X, y, base, alt = _reg_setup()
    result = ferrum.cv_scores_chart(base, X, y, cv=3, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_cv_scores_chart_compare_none_byte_identical():
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.cv_scores_chart(base, X, y, cv=3, compare=None).to_svg()
    without_kwarg = ferrum.cv_scores_chart(base, X, y, cv=3).to_svg()
    assert with_kwarg == without_kwarg


def test_alpha_selection_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one panel per model (base + alt).

    Both models must accept an ``alpha`` constructor parameter; two Ridge
    instances satisfy this (RandomForestRegressor has no ``alpha`` param).
    """
    from sklearn.linear_model import Ridge

    X, y, base, _ = _reg_setup()
    alt = Ridge(alpha=0.1)
    result = ferrum.alpha_selection_chart(
        base, X, y, alphas=[0.1, 1.0, 10.0], cv=3, compare={"alt": alt}
    )
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_alpha_selection_chart_compare_none_byte_identical():
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.alpha_selection_chart(
        base, X, y, alphas=[0.1, 1.0, 10.0], cv=3, compare=None
    ).to_svg()
    without_kwarg = ferrum.alpha_selection_chart(base, X, y, alphas=[0.1, 1.0, 10.0], cv=3).to_svg()
    assert with_kwarg == without_kwarg


# ---------------------------------------------------------------------------
# explanation.py — importance / shap_* / pdp
# ---------------------------------------------------------------------------


def test_importance_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one panel per model (base + alt)."""
    X, y, base, alt = _reg_setup()
    result = ferrum.importance_chart(base, X, y, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_importance_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.importance_chart(base, X, y, compare=None).to_svg()
    without_kwarg = ferrum.importance_chart(base, X, y).to_svg()
    assert with_kwarg == without_kwarg


def test_shap_beeswarm_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one panel per model."""
    X, y, base, alt = _reg_setup()
    result = ferrum.shap_beeswarm_chart(base, X, y, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_shap_beeswarm_chart_compare_none_byte_identical():
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.shap_beeswarm_chart(base, X, y, compare=None).to_svg()
    without_kwarg = ferrum.shap_beeswarm_chart(base, X, y).to_svg()
    assert with_kwarg == without_kwarg


def test_shap_bar_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one panel per model."""
    X, y, base, alt = _reg_setup()
    result = ferrum.shap_bar_chart(base, X, y, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_shap_bar_chart_compare_none_byte_identical():
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.shap_bar_chart(base, X, y, compare=None).to_svg()
    without_kwarg = ferrum.shap_bar_chart(base, X, y).to_svg()
    assert with_kwarg == without_kwarg


def test_shap_waterfall_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one panel per model."""
    X, y, base, alt = _reg_setup()
    result = ferrum.shap_waterfall_chart(base, X, y, sample_idx=0, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_shap_waterfall_chart_compare_none_byte_identical():
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.shap_waterfall_chart(base, X, y, sample_idx=0, compare=None).to_svg()
    without_kwarg = ferrum.shap_waterfall_chart(base, X, y, sample_idx=0).to_svg()
    assert with_kwarg == without_kwarg


def test_shap_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one panel per model (shap_chart is deprecated)."""
    X, y, base, alt = _reg_setup()
    with pytest.warns(DeprecationWarning):
        result = ferrum.shap_chart(base, X, y, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_shap_chart_compare_none_byte_identical():
    X, y, base, _ = _reg_setup()
    with pytest.warns(DeprecationWarning):
        with_kwarg = ferrum.shap_chart(base, X, y, compare=None).to_svg()
    with pytest.warns(DeprecationWarning):
        without_kwarg = ferrum.shap_chart(base, X, y).to_svg()
    assert with_kwarg == without_kwarg


def test_pdp_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one nested panel per model."""
    X, y, base, alt = _reg_setup()
    result = ferrum.pdp_chart(base, X, y, features=["f0"], compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_pdp_chart_compare_none_byte_identical():
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.pdp_chart(base, X, y, features=["f0"], compare=None).to_svg()
    without_kwarg = ferrum.pdp_chart(base, X, y, features=["f0"]).to_svg()
    assert with_kwarg == without_kwarg


# ---------------------------------------------------------------------------
# clustering.py — pca_scree / intercluster / silhouette / manifold now render
# small multiples (independent scales); cluster_diagnostics / elbow stay
# rejected (sweep-based, no per-model ModelSource).
# ---------------------------------------------------------------------------


def _clu_setup():
    """Clustering fixtures: dataset + two KMeans with different k."""
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    X_np = df.to_numpy()
    base = load_fixture("kmeans_3cluster")
    alt = KMeans(n_clusters=4, random_state=0, n_init=10).fit(X_np)
    return df, base, alt


def test_pca_scree_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one panel per PCA model (base + alt)."""
    from sklearn.decomposition import PCA

    df = load_dataset("regression").select(_REG_FEATURES)
    base = load_fixture("pca_4comp")
    alt = PCA(n_components=3).fit(df.to_numpy())
    result = ferrum.pca_scree_chart(base, df, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_pca_scree_chart_compare_none_byte_identical():
    """compare=None is byte-identical to omitting the kwarg."""
    df = load_dataset("regression").select(_REG_FEATURES)
    model = load_fixture("pca_4comp")
    with_kwarg = ferrum.pca_scree_chart(model, df, compare=None).to_svg()
    without_kwarg = ferrum.pca_scree_chart(model, df).to_svg()
    assert with_kwarg == without_kwarg


def test_pca_scree_chart_raw_data_compare_rejected():
    """The raw-DataFrame/array path computes Rust SVD on a single matrix and
    wraps no per-model source, so compare= there must raise (not silently
    ignore the kwarg)."""
    df = load_dataset("regression").select(_REG_FEATURES)
    with pytest.raises(ValueError, match="compare= requires a fitted PCA estimator"):
        ferrum.pca_scree_chart(df, compare={"alt": df})


def test_cluster_diagnostics_compare_rejected():
    df = load_dataset("clustering")
    with pytest.raises(ValueError, match="compare= is not supported for cluster_diagnostics"):
        ferrum.cluster_diagnostics(df, ks=[2, 3, 4], compare={"alt": df})


def test_cluster_diagnostics_compare_none_default_path_works():
    df = load_dataset("clustering")
    chart = ferrum.cluster_diagnostics(df, ks=[2, 3, 4], random_state=0, compare=None)
    assert "<svg" in chart.to_svg()


def test_intercluster_distance_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one panel per clusterer (base + alt)."""
    df, base, alt = _clu_setup()
    result = ferrum.intercluster_distance_chart(base, df, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_intercluster_distance_chart_compare_none_byte_identical():
    """compare=None is byte-identical to omitting the kwarg."""
    df, base, _ = _clu_setup()
    with_kwarg = ferrum.intercluster_distance_chart(base, df, compare=None).to_svg()
    without_kwarg = ferrum.intercluster_distance_chart(base, df).to_svg()
    assert with_kwarg == without_kwarg


def test_intercluster_distance_chart_compare_heterogeneous_k():
    """Each panel must use its OWN model's k, not the base model's k.

    Comparing KMeans(3) vs KMeans(5) must yield a base panel with 3 cluster
    centers and an alt panel with 5 cluster centers.  The pre-fix code resolved
    k once from the base model and passed it to every panel, so the alt panel
    would have embedded only 3 centers.
    """
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    X_np = df.to_numpy()
    base = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X_np)
    alt = KMeans(n_clusters=5, random_state=0, n_init=10).fit(X_np)

    result = ferrum.intercluster_distance_chart(base, df, compare={"k5": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2

    base_panel, alt_panel = result.charts
    # Each panel's underlying data has one row per cluster center.
    base_data = base_panel._data
    alt_data = alt_panel._data
    assert base_data is not None, "base panel has no data"
    assert alt_data is not None, "alt panel has no data"
    assert len(base_data) == 3, f"expected 3 rows in base panel, got {len(base_data)}"
    assert len(alt_data) == 5, f"expected 5 rows in alt panel, got {len(alt_data)}"


def test_silhouette_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one silhouette panel per clusterer."""
    df, base, alt = _clu_setup()
    result = ferrum.silhouette_chart(base, df, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_silhouette_chart_compare_none_byte_identical():
    """compare=None is byte-identical to omitting the kwarg."""
    df, base, _ = _clu_setup()
    with_kwarg = ferrum.silhouette_chart(base, df, compare=None).to_svg()
    without_kwarg = ferrum.silhouette_chart(base, df).to_svg()
    assert with_kwarg == without_kwarg


def test_manifold_chart_compare_renders_small_multiples():
    """compare= returns a ConcatChart with one embedding panel per clusterer."""
    df, base, alt = _clu_setup()
    result = ferrum.manifold_chart(base, df, method="pca", compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_manifold_chart_compare_none_byte_identical():
    """compare=None is byte-identical to omitting the kwarg."""
    df, base, _ = _clu_setup()
    with_kwarg = ferrum.manifold_chart(base, df, method="pca", compare=None).to_svg()
    without_kwarg = ferrum.manifold_chart(base, df, method="pca").to_svg()
    assert with_kwarg == without_kwarg


def test_elbow_chart_compare_rejected():
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    with pytest.raises(ValueError, match="compare= is not supported for elbow_chart"):
        ferrum.elbow_chart(KMeans, df, ks=[2, 3, 4], compare={"alt": KMeans})


# ---------------------------------------------------------------------------
# Cross-cutting: the rejection is loud (never silent), and the documented
# reason is carried in the message.
# ---------------------------------------------------------------------------


def test_rejection_message_includes_reason():
    """An EXCLUDED chart carries its documented reason in the rejection.

    ``cluster_diagnostics`` is sweep-based (no per-model source), so it keeps a
    loud rejection; the message names the chart and carries a non-empty reason
    after the colon.
    """
    df = load_dataset("clustering")
    with pytest.raises(ValueError) as exc:
        ferrum.cluster_diagnostics(df, ks=[2, 3, 4], compare={"alt": df})
    msg = str(exc.value)
    prefix = "compare= is not supported for cluster_diagnostics:"
    assert msg.startswith(prefix)
    assert msg[len(prefix) :].strip()
    # Pin the refined structural reason (not the old "meaningless" wording): it
    # must explain there is no per-model source and point at the method-sweep
    # follow-up (#43), so a regression to the stale reason is caught.
    assert "no per-model" in msg
    assert "#43" in msg


def test_compare_none_is_byte_identical_to_omitting_kwarg():
    """The no-compare default path must be unchanged: passing compare=None
    explicitly produces the exact same SVG as omitting the kwarg entirely."""
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.residuals_chart(base, X, y, panels="single", compare=None).to_svg()
    without_kwarg = ferrum.residuals_chart(base, X, y, panels="single").to_svg()
    assert with_kwarg == without_kwarg


def test_numpy_inputs_compare_supported():
    """compare= on the SUPPORTED prediction_error_chart path works regardless
    of input container type."""
    df = load_dataset("regression")
    X = df.select(_REG_FEATURES).to_numpy()
    y = df["y"].to_numpy()
    base = load_fixture("regression_ridge")
    alt = load_fixture("regression_rf")
    chart = ferrum.prediction_error_chart(base, X, y, compare={"alt": alt})
    assert "<svg" in chart.to_svg()
    assert chart._data is not None
    assert set(chart._data["model"].unique().to_list()) == {"alt", "base"}
    # Unused import guard for np (kept explicit for clarity of intent).
    assert isinstance(X, np.ndarray)
