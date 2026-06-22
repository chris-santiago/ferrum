"""Regression tests for T3.4 — first-positional-param canonicalization (D-FIRSTPARAM-1).

Covers PLOT-02/06/08/11: the model-diagnostic figure-function family now uses a
single canonical first-positional-parameter name per family, with the legacy name
accepted as a deprecated keyword alias.

Renames in scope (only where the current name diverged from the family canonical):
- ``cluster_diagnostics``: ``X`` -> canonical ``model`` (legacy ``X=`` alias).
- ``rank_chart`` / ``rank1d_chart`` / ``rank2d_chart``: ``source`` ->
  canonical ``data_or_source`` (legacy ``source=`` alias).

Not renamed (already family-canonical): ``parallel_coordinates_chart`` (``data``,
seaborn/data family) and ``decision_boundary_chart`` (``model``, model-diagnostic
family). They are asserted here so future drift is caught.

Each renamed function is checked for:
1. positional call unchanged (canonical behavior),
2. the new canonical keyword produces an identical result,
3. the old keyword alias produces an identical result,
4. supplying both canonical + alias raises a clear ``TypeError``.

The canonical-keyword assertions FAIL on the pre-rename signature: passing
``model=`` to ``cluster_diagnostics`` (formerly ``X=``-only) or ``data_or_source=``
to the rank functions (formerly ``source=``-only) raised ``TypeError`` on the old
code, so test 2 below is the rename's regression oracle.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------


@pytest.fixture
def feature_matrix() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(60, 4))


def _spec_json(chart) -> str:
    """Canonical comparison key for a returned Chart."""
    return chart.to_spec().to_json()


# ---------------------------------------------------------------------------
# Family-canonical first-param names (drift guard for the whole family)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("func_name", "expected_first_param"),
    [
        # model-diagnostic family -> "model"
        ("cluster_diagnostics", "model"),
        ("decision_boundary_chart", "model"),
        ("pca_scree_chart", "model"),
        ("intercluster_distance_chart", "model"),
        ("silhouette_chart", "model"),
        ("manifold_chart", "model"),
        ("elbow_chart", "model"),
        # rank* family -> "data_or_source"
        ("rank_chart", "data_or_source"),
        ("rank1d_chart", "data_or_source"),
        ("rank2d_chart", "data_or_source"),
        # seaborn/data family -> "data"
        ("parallel_coordinates_chart", "data"),
    ],
)
def test_first_param_is_family_canonical(func_name, expected_first_param):
    fn = getattr(fm, func_name)
    first = next(iter(inspect.signature(fn).parameters))
    assert first == expected_first_param


# ---------------------------------------------------------------------------
# cluster_diagnostics: X -> model (legacy X= alias)
# ---------------------------------------------------------------------------


def test_cluster_diagnostics_positional_canonical_and_alias_identical(feature_matrix):
    ks = range(2, 5)
    positional = _spec_json(fm.cluster_diagnostics(feature_matrix, ks=ks, scoring="elbow"))
    canonical = _spec_json(fm.cluster_diagnostics(model=feature_matrix, ks=ks, scoring="elbow"))
    alias = _spec_json(fm.cluster_diagnostics(X=feature_matrix, ks=ks, scoring="elbow"))
    assert positional == canonical
    assert positional == alias


def test_cluster_diagnostics_double_supply_raises(feature_matrix):
    with pytest.raises(TypeError, match="model"):
        fm.cluster_diagnostics(feature_matrix, X=feature_matrix, ks=range(2, 5))


def test_cluster_diagnostics_missing_first_param_raises():
    with pytest.raises(TypeError, match="model"):
        fm.cluster_diagnostics(ks=range(2, 5))


# ---------------------------------------------------------------------------
# rank2d_chart: source -> data_or_source (legacy source= alias)
# ---------------------------------------------------------------------------


def test_rank2d_positional_canonical_and_alias_identical(feature_matrix):
    positional = _spec_json(fm.rank2d_chart(feature_matrix))
    canonical = _spec_json(fm.rank2d_chart(data_or_source=feature_matrix))
    alias = _spec_json(fm.rank2d_chart(source=feature_matrix))
    assert positional == canonical
    assert positional == alias


def test_rank2d_double_supply_raises(feature_matrix):
    with pytest.raises(TypeError, match="data_or_source"):
        fm.rank2d_chart(feature_matrix, source=feature_matrix)


def test_rank2d_missing_first_param_raises():
    with pytest.raises(TypeError, match="data_or_source"):
        fm.rank2d_chart()


# ---------------------------------------------------------------------------
# rank1d_chart: source -> data_or_source (legacy source= alias)
# ---------------------------------------------------------------------------


def test_rank1d_positional_canonical_and_alias_identical(feature_matrix):
    positional = _spec_json(fm.rank1d_chart(feature_matrix))
    canonical = _spec_json(fm.rank1d_chart(data_or_source=feature_matrix))
    alias = _spec_json(fm.rank1d_chart(source=feature_matrix))
    assert positional == canonical
    assert positional == alias


def test_rank1d_double_supply_raises(feature_matrix):
    with pytest.raises(TypeError, match="data_or_source"):
        fm.rank1d_chart(feature_matrix, source=feature_matrix)


def test_rank1d_missing_first_param_raises():
    with pytest.raises(TypeError, match="data_or_source"):
        fm.rank1d_chart()


# ---------------------------------------------------------------------------
# rank_chart (deprecated dispatcher): source -> data_or_source
# ---------------------------------------------------------------------------


def test_rank_chart_positional_canonical_and_alias_identical(feature_matrix):
    with pytest.warns(DeprecationWarning):
        positional = _spec_json(fm.rank_chart(feature_matrix, rank="2d"))
    with pytest.warns(DeprecationWarning):
        canonical = _spec_json(fm.rank_chart(data_or_source=feature_matrix, rank="2d"))
    with pytest.warns(DeprecationWarning):
        alias = _spec_json(fm.rank_chart(source=feature_matrix, rank="2d"))
    assert positional == canonical
    assert positional == alias


def test_rank_chart_double_supply_raises(feature_matrix):
    with pytest.raises(TypeError, match="data_or_source"):
        fm.rank_chart(feature_matrix, source=feature_matrix, rank="2d")


def test_rank_chart_missing_first_param_raises():
    with pytest.raises(TypeError, match="data_or_source"):
        fm.rank_chart(rank="2d")
