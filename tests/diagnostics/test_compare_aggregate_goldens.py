"""compare= aggregate rendering — composite golden tests (GH #35, GH #42).

Four representative small-multiples goldens, one per bucket defined in the
GH #35 spec, plus two dodge-by-model goldens added for GH #42 and one
classification-curve small-multiple added for #213:

1. NESTED case — ``pdp_chart`` with two features and ``compare=``.
   The outer ``ConcatChart`` holds one PDP facet-composite per model;
   each of those is itself a ``ConcatChart`` of per-feature panels.

2. FLAT supervised case — ``cv_scores_chart`` with ``compare=``.
   A ``ConcatChart`` of one CV-scores panel per model, arranged in a
   single row with shared x/y scales.

3. UNSUPERVISED case — ``silhouette_chart`` with ``compare=``.
   A ``ConcatChart`` of one silhouette-bar panel per clusterer,
   with independent per-panel scales (cluster coordinate systems differ).

4. TITLED-COMPOSITE case (GH #45) — ``residuals_chart`` with ``compare=``.
   Each per-model panel is itself a titled 2x2 diagnostic composite; the
   figure title lowers to a per-child label (Task 5d), so the panels share
   axes position-wise instead of gating to the old non-sharing path.

5. DODGE-BY-MODEL case (GH #42) — ``importance_chart`` with ``compare=``.
   A single ``Chart`` (not a ``ConcatChart``) with one bar per
   (feature, model), dodged within each feature band, plus dodged error
   rules and dodged value labels, rendered via the vertical desugar form +
   ``CoordFlip`` for the default ``orient="horizontal"``.

6. DODGE-BY-MODEL case (GH #42) — ``shap_bar_chart`` with ``compare=``
   (``per_class=False``). A single ``Chart`` with one bar per
   (feature, model), dodged within each feature band.

7. CLASSIFICATION-CURVE case (#213) — ``discrimination_threshold_chart``
   with ``compare=``. A ``ConcatChart`` of one four-metric sweep panel per
   model, each carrying its OWN F1-optimum annotation. The one
   classification chart that composes rather than overlays, because its
   colour channel is already spent on the metric legend.

All fixtures and seeds are fixed for byte-stable reproduction.

Regenerate with ``FERRUM_REGENERATE_GOLDENS=1 pytest
tests/diagnostics/test_compare_aggregate_goldens.py`` — which rewrites each
golden through ``tests._snapshots.regen_and_verify``, rasterizing an
inspection PNG alongside it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture

_GOLDEN_ROOT = Path(__file__).parent.parent / "goldens" / "phase_10"
_REGENERATE = bool(os.environ.get("FERRUM_REGENERATE_GOLDENS"))

_REG_FEATURES = ["f0", "f1", "f2", "f3", "f4"]


def _check_golden(svg: str, name: str) -> None:
    path = _GOLDEN_ROOT / f"{name}.svg"
    if _REGENERATE or not path.exists():
        # ``regen_and_verify`` writes the SVG *and* rasterizes an inspection
        # PNG under ``tests/snapshots/``, printing both paths, so a regen can
        # never silently skip the visual check CLAUDE.md's "goldens are not
        # blessed until visually inspected" rule requires. Mirrors
        # ``tests/test_composite_shared_legend_goldens.py``'s
        # ``_check_or_update``.
        from tests._snapshots import regen_and_verify

        regen_and_verify(path, svg)
        if not _REGENERATE:
            pytest.skip(f"created new golden at {path}; rerun to verify")
        return
    expected = path.read_text()
    # Do not use ``assert svg == expected`` — pytest's assertion rewriter
    # spends minutes diffing two large SVG strings on a mismatch.
    from tests._snapshots import assert_svg_eq

    assert_svg_eq(
        svg,
        expected,
        name=name,
        regen_hint="set FERRUM_REGENERATE_GOLDENS=1 to refresh after intentional changes",
    )


# ---------------------------------------------------------------------------
# Bucket 1 — nested PDP
# ---------------------------------------------------------------------------


def _pdp_models():
    """Two pre-fitted regression models + shared feature matrix.

    Ridge is the base; RandomForest is the alt.  Both are loaded from
    pinned .skops fixtures so the golden is byte-stable.
    """
    df = load_dataset("regression")
    X = df.select(_REG_FEATURES)
    y = df["y"]
    base = load_fixture("regression_ridge")
    alt = load_fixture("regression_rf")
    return base, alt, X, y


def test_golden_compare_pdp_two_models():
    """Nested PDP golden: ConcatChart of two per-model PDP composites.

    ``pdp_chart`` with two features produces a per-feature ConcatChart
    per model; ``compare=`` wraps those into an outer ConcatChart whose
    children are the two model-level composites.
    """
    base, alt, X, y = _pdp_models()
    chart = ferrum.pdp_chart(
        base,
        X,
        y,
        features=["f0", "f1"],
        grid_resolution=10,
        compare={"alt": alt},
    )
    _check_golden(chart.to_svg(), "compare_pdp_two_models")


# ---------------------------------------------------------------------------
# Bucket 2 — flat supervised (cv_scores)
# ---------------------------------------------------------------------------


def _cv_scores_models():
    """Two pre-fitted regression models for CV-scores comparison.

    Ridge and RandomForest, both loaded from pinned fixtures.
    ``cv_scores_chart`` with shared scales is deterministic for these
    fixture models across runs within a pinned sklearn version.
    """
    df = load_dataset("regression")
    X = df.select(_REG_FEATURES)
    y = df["y"]
    base = load_fixture("regression_ridge")
    alt = load_fixture("regression_rf")
    return base, alt, X, y


def test_golden_compare_cv_scores_two_models():
    """Flat supervised golden: ConcatChart of two CV-scores panels.

    Each panel is a single ``Chart`` produced by the cv_scores builder
    for one model; shared x/y scales make the two panels directly
    comparable.
    """
    base, alt, X, y = _cv_scores_models()
    chart = ferrum.cv_scores_chart(base, X, y, cv=3, compare={"alt": alt})
    _check_golden(chart.to_svg(), "compare_cv_scores_two_models")


# ---------------------------------------------------------------------------
# Bucket 3 — unsupervised (silhouette)
# ---------------------------------------------------------------------------


def _silhouette_clusterers():
    """KMeans-3 (fixture) vs KMeans-4 (fitted deterministically).

    KMeans(4) is fitted with a fixed ``random_state`` and ``n_init``
    so the golden is byte-stable within a pinned sklearn version.
    """
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    base = load_fixture("kmeans_3cluster")
    alt = KMeans(n_clusters=4, random_state=0, n_init=10).fit(df.to_numpy())
    return base, alt, df


def test_golden_compare_silhouette_two_clusterers():
    """Unsupervised golden: ConcatChart of two silhouette panels.

    Each panel is built from one clusterer's own k; panels resolve scales
    independently (two silhouette spaces are not comparable).
    """
    base, alt, df = _silhouette_clusterers()
    chart = ferrum.silhouette_chart(base, df, compare={"k4": alt})
    _check_golden(chart.to_svg(), "compare_silhouette_two_clusterers")


# ---------------------------------------------------------------------------
# Bucket 4 — titled-composite (residuals, GH #45 headline)
# ---------------------------------------------------------------------------


def _residuals_models():
    """Two linear regressors whose fitted ranges genuinely differ.

    Base is the pinned Ridge fixture (fitted values span roughly [-8, 5]);
    alt is a heavily-regularized ``Ridge(alpha=300)`` fitted deterministically
    (closed-form, so byte-stable within a pinned sklearn version) whose fitted
    values span the narrower [-3.5, 2.5].  Both are linear, so each yields the
    full 4-panel diagnostic (including residuals-vs-leverage), making the two
    per-model grids congruent — the precondition for position-wise pairing.
    The different fitted ranges make the shared-axis union observable: alt's
    fitted axis is pulled out to base's wider range.
    """
    from sklearn.linear_model import Ridge

    df = load_dataset("regression")
    X = df.select(_REG_FEATURES)
    y = df["y"]
    base = load_fixture("regression_ridge")
    alt = Ridge(alpha=300.0).fit(X.to_numpy(), y.to_numpy())
    return base, alt, X, y


def test_golden_compare_residuals_two_models():
    """Titled-composite golden (GH #45): ConcatChart of two per-model residuals grids.

    Each model panel is a titled 2x2 residuals diagnostic composite.  The
    per-model title lowers to a per-child label (Task 5d), so the composition
    rides the one-call Rust composite path and the two congruent grids share
    axes position-wise rather than gating to the old non-sharing path.
    """
    base, alt, X, y = _residuals_models()
    chart = ferrum.residuals_chart(base, X, y, compare={"alt": alt})
    _check_golden(chart.to_svg(), "compare_residuals_two_models")


# ---------------------------------------------------------------------------
# Buckets 5 & 6 — dodge-by-model (importance_chart / shap_bar_chart, GH #42)
# ---------------------------------------------------------------------------


def _dodge_compare_models():
    """Two pre-fitted regression models for the dodge-by-model goldens.

    Ridge and RandomForest, both loaded from pinned fixtures (same fixture
    pair used throughout ``tests/diagnostics/test_compare_dodge.py``), so
    the goldens are byte-stable within a pinned sklearn version.
    """
    df = load_dataset("regression")
    X = df.select(_REG_FEATURES)
    y = df["y"]
    base = load_fixture("regression_ridge")
    alt = load_fixture("regression_rf")
    return base, alt, X, y


def test_golden_compare_importance_two_models():
    """Dodge-by-model golden: single Chart with bars dodged per feature band.

    Default ``orient="horizontal"`` renders the vertical desugar form under
    ``CoordFlip``; default ``error_bars=True``/``show_values=True`` exercise
    the dodged error-rule and dodged value-label layers alongside the bars.
    """
    base, alt, X, y = _dodge_compare_models()
    chart = ferrum.importance_chart(base, X, y, compare={"alt": alt})
    _check_golden(chart.to_svg(), "compare_importance_two_models")


def test_golden_compare_shap_bar_two_models():
    """Dodge-by-model golden: single Chart with mean-|SHAP| bars dodged per
    feature band (``per_class=False``, the default, so no competing
    class-facet dimension)."""
    base, alt, X, y = _dodge_compare_models()
    chart = ferrum.shap_bar_chart(base, X, y, compare={"alt": alt})
    _check_golden(chart.to_svg(), "compare_shap_bar_two_models")


# ---------------------------------------------------------------------------
# Bucket 7 — classification curve (discrimination_threshold, #213)
# ---------------------------------------------------------------------------


def _threshold_compare_models():
    """The pinned binary logistic fixture plus a differently-calibrated rival.

    ``GaussianNB`` is a closed-form MLE fit (no solver, no seed), so it is
    byte-stable within a pinned sklearn version, and its probability
    calibration differs enough from the logistic fixture's that the two F1
    optima land on different thresholds and different F1 values — which is
    what makes the per-panel annotations visibly distinct in this golden
    rather than two copies of one pooled string.
    """
    from sklearn.naive_bayes import GaussianNB

    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    base = load_fixture("binary_logistic")
    alt = GaussianNB().fit(X.to_numpy(), y.to_numpy())
    return base, alt, X, y


def test_golden_compare_discrimination_threshold_two_models():
    """Classification-curve golden: ConcatChart of two threshold-sweep panels.

    Each panel holds one model's four metric curves plus that model's own
    F1-optimum label; shared x/y scales put the two sweeps on one
    comparable frame. Replaces the pre-#213 single panel, in which the two
    models' sweeps stacked with no distinguishing column and rendered as
    interleaved zig-zag polylines under one pooled optimum annotation.
    """
    base, alt, X, y = _threshold_compare_models()
    chart = ferrum.discrimination_threshold_chart(
        base,
        X,
        y,
        n_thresholds=20,
        threshold_line=True,
        compare={"alt": alt},
    )
    _check_golden(chart.to_svg(), "compare_discrimination_threshold_two_models")
