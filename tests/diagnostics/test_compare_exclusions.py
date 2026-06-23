"""T3.5 (D-COMPARE-1): ``compare=`` is accepted for signature uniformity across
the whole model-diagnostic family.

The classification family (``roc_chart``, ``pr_chart``, ``calibration_chart``,
...) genuinely *supports* ``compare=`` (overlaid curves); those tests live in
``tests/diagnostics/test_compare.py``.

Two regression charts also support ``compare=`` on their scatter path, because
their marks already accept a ``color_field`` exactly like the classification
family: ``prediction_error_chart`` (default path, no band) and
``residuals_chart`` (``panels="single"``). Both detect the ``model`` column on
the resolved compare-source frame and drive a per-model colour group. Each gates
its single-model-only aggregate path with a loud ``ValueError``:
``prediction_error_chart`` rejects ``compare=`` combined with ``ci`` /
``reference_band`` (the band is a single-model aggregate), and
``residuals_chart`` rejects ``compare=`` with a multi-panel ``panels`` value
(the 4-panel QQ/scale-location/leverage grid is single-model).

The remaining 17 charts across regression, model-selection, explanation, and
clustering stay EXCLUDED because rendering a coherent multi-model comparison
would need a second visual dimension (model-facet or grouped bars) that the
single-model builders and their marks do not provide, because a channel
collision would result, or because the diagnostic is unsupervised (no ``y``).
They reject a non-``None`` ``compare`` with a loud, documented ``ValueError`` —
never a silent drop.

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
# regression.py — SUPPORTED: prediction_error_chart (default path),
# residuals_chart (panels="single").  Each gates its single-model-only
# aggregate path with a loud ValueError.
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


def test_prediction_error_chart_compare_with_ci_rejected():
    """The residual confidence band is a single-model aggregate; combining it
    with compare= is rejected loudly."""
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="single-model aggregate"):
        ferrum.prediction_error_chart(base, X, y, ci=0.9, compare={"alt": alt})


def test_prediction_error_chart_compare_with_reference_band_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="single-model aggregate"):
        ferrum.prediction_error_chart(base, X, y, reference_band=True, compare={"alt": alt})


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


def test_residuals_chart_compare_multipanel_rejected():
    """The 4-panel QQ/scale-location/leverage grid is single-model; compare=
    with a multi-panel layout is rejected loudly."""
    X, y, base, alt = _reg_setup()
    with pytest.raises(
        ValueError, match="compare= is only supported with panels='single' for residuals_chart"
    ):
        ferrum.residuals_chart(base, X, y, panels="auto", compare={"alt": alt})


def test_residuals_chart_compare_explicit_panel_list_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(
        ValueError, match="compare= is only supported with panels='single' for residuals_chart"
    ):
        ferrum.residuals_chart(
            base, X, y, panels=["residuals_vs_fitted", "qq"], compare={"alt": alt}
        )


def test_residuals_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    chart = ferrum.residuals_chart(base, X, y, panels="single", compare=None)
    assert "<svg" in chart.to_svg()


def test_cooks_distance_chart_compare_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="compare= is not supported for cooks_distance_chart"):
        ferrum.cooks_distance_chart(base, X, y, compare={"alt": alt})


def test_cooks_distance_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    chart = ferrum.cooks_distance_chart(base, X, y, threshold="auto", compare=None)
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# model_selection.py — learning / validation / cv_scores / alpha_selection
# ---------------------------------------------------------------------------


def test_learning_curve_chart_compare_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="compare= is not supported for learning_curve_chart"):
        ferrum.learning_curve_chart(base, X, y, cv=3, compare={"alt": alt})


def test_learning_curve_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    chart = ferrum.learning_curve_chart(base, X, y, cv=3, compare=None)
    assert "<svg" in chart.to_svg()


def test_validation_curve_chart_compare_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="compare= is not supported for validation_curve_chart"):
        ferrum.validation_curve_chart(
            base, X, y, param="alpha", values=[0.1, 1.0, 10.0], cv=3, compare={"alt": alt}
        )


def test_validation_curve_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    chart = ferrum.validation_curve_chart(
        base, X, y, param="alpha", values=[0.1, 1.0, 10.0], cv=3, compare=None
    )
    assert "<svg" in chart.to_svg()


def test_cv_scores_chart_compare_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="compare= is not supported for cv_scores_chart"):
        ferrum.cv_scores_chart(base, X, y, cv=3, compare={"alt": alt})


def test_cv_scores_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    chart = ferrum.cv_scores_chart(base, X, y, cv=3, compare=None)
    assert "<svg" in chart.to_svg()


def test_alpha_selection_chart_compare_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="compare= is not supported for alpha_selection_chart"):
        ferrum.alpha_selection_chart(
            base, X, y, alphas=[0.1, 1.0, 10.0], cv=3, compare={"alt": alt}
        )


def test_alpha_selection_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    chart = ferrum.alpha_selection_chart(base, X, y, alphas=[0.1, 1.0, 10.0], cv=3, compare=None)
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# explanation.py — importance / shap_* / pdp
# ---------------------------------------------------------------------------


def test_importance_chart_compare_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="compare= is not supported for importance_chart"):
        ferrum.importance_chart(base, X, y, compare={"alt": alt})


def test_importance_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    chart = ferrum.importance_chart(base, X, y, compare=None)
    assert "<svg" in chart.to_svg()


def test_shap_beeswarm_chart_compare_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="compare= is not supported for shap_beeswarm_chart"):
        ferrum.shap_beeswarm_chart(base, X, y, compare={"alt": alt})


def test_shap_bar_chart_compare_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="compare= is not supported for shap_bar_chart"):
        ferrum.shap_bar_chart(base, X, y, compare={"alt": alt})


def test_shap_waterfall_chart_compare_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="compare= is not supported for shap_waterfall_chart"):
        ferrum.shap_waterfall_chart(base, X, y, sample_idx=0, compare={"alt": alt})


def test_shap_chart_compare_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="compare= is not supported for shap_chart"):
        ferrum.shap_chart(base, X, y, compare={"alt": alt})


def test_pdp_chart_compare_rejected():
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError, match="compare= is not supported for pdp_chart"):
        ferrum.pdp_chart(base, X, y, features=["f0"], compare={"alt": alt})


def test_pdp_chart_compare_none_default_path_works():
    X, y, base, _ = _reg_setup()
    chart = ferrum.pdp_chart(base, X, y, features=["f0"], compare=None)
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# clustering.py — pca_scree / cluster_diagnostics / intercluster / silhouette /
# manifold / elbow (all unsupervised — no y target)
# ---------------------------------------------------------------------------


def test_pca_scree_chart_compare_rejected():
    df = load_dataset("regression").select(_REG_FEATURES)
    model = load_fixture("pca_4comp")
    with pytest.raises(ValueError, match="compare= is not supported for pca_scree_chart"):
        ferrum.pca_scree_chart(model, df, compare={"alt": model})


def test_pca_scree_chart_compare_none_default_path_works():
    df = load_dataset("regression").select(_REG_FEATURES)
    model = load_fixture("pca_4comp")
    chart = ferrum.pca_scree_chart(model, df, compare=None)
    assert "<svg" in chart.to_svg()


def test_cluster_diagnostics_compare_rejected():
    df = load_dataset("clustering")
    with pytest.raises(ValueError, match="compare= is not supported for cluster_diagnostics"):
        ferrum.cluster_diagnostics(df, ks=[2, 3, 4], compare={"alt": df})


def test_cluster_diagnostics_compare_none_default_path_works():
    df = load_dataset("clustering")
    chart = ferrum.cluster_diagnostics(df, ks=[2, 3, 4], random_state=0, compare=None)
    assert "<svg" in chart.to_svg()


def test_intercluster_distance_chart_compare_rejected():
    df = load_dataset("clustering")
    model = load_fixture("kmeans_3cluster")
    with pytest.raises(
        ValueError, match="compare= is not supported for intercluster_distance_chart"
    ):
        ferrum.intercluster_distance_chart(model, df, compare={"alt": model})


def test_silhouette_chart_compare_rejected():
    df = load_dataset("clustering")
    model = load_fixture("kmeans_3cluster")
    with pytest.raises(ValueError, match="compare= is not supported for silhouette_chart"):
        ferrum.silhouette_chart(model, df, compare={"alt": model})


def test_silhouette_chart_compare_none_default_path_works():
    df = load_dataset("clustering")
    model = load_fixture("kmeans_3cluster")
    chart = ferrum.silhouette_chart(model, df, compare=None)
    assert "<svg" in chart.to_svg()


def test_manifold_chart_compare_rejected():
    df = load_dataset("clustering")
    model = load_fixture("kmeans_3cluster")
    with pytest.raises(ValueError, match="compare= is not supported for manifold_chart"):
        ferrum.manifold_chart(model, df, method="pca", compare={"alt": model})


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
    """An EXCLUDED chart carries its documented reason in the rejection."""
    X, y, base, alt = _reg_setup()
    with pytest.raises(ValueError) as exc:
        ferrum.cooks_distance_chart(base, X, y, compare={"alt": alt})
    msg = str(exc.value)
    assert "compose one chart per model" in msg


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
