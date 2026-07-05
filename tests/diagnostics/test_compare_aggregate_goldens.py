"""compare= aggregate rendering — composite golden tests (GH #35).

Four representative goldens, one per bucket defined in the spec:

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

All fixtures and seeds are fixed for byte-stable reproduction.

Regenerate with ``FERRUM_REGENERATE_GOLDENS=1 pytest
tests/diagnostics/test_compare_aggregate_goldens.py``.
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)
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
