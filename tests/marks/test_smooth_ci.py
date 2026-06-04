"""Phase 8b Task 34: mark_smooth(ci=) emits a CI-band ribbon + line layer."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import ferrum as fe


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    n = 20
    x = np.arange(n, dtype=float)
    y = x + rng.standard_normal(n) * 0.5
    return pl.DataFrame({"x": x, "y": y})


def test_smooth_with_ci_does_not_warn(df):
    """The 8a warn-once for ci= is gone in 8b."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fe.Chart(df).mark_smooth(ci=0.95).encode(x="x", y="y")._build_spec()
    smooth_ci_warns = [
        w for w in caught if "mark_smooth" in str(w.message) and "ci" in str(w.message).lower()
    ]
    assert smooth_ci_warns == [], (
        f"unexpected mark_smooth(ci=) warnings: {[str(w.message) for w in smooth_ci_warns]}"
    )


def test_smooth_with_ci_produces_2_layers(df):
    spec = fe.Chart(df).mark_smooth(ci=0.95).encode(x="x", y="y")._build_spec()
    assert spec.layers is not None, "ci= path should produce layered spec"
    assert len(spec.layers) == 2, f"expected ribbon+line layers, got {len(spec.layers)}"


def test_smooth_no_ci_produces_single_line(df):
    """No ci → no layered mode; the 8a single-line path is preserved."""
    spec = fe.Chart(df).mark_smooth().encode(x="x", y="y")._build_spec()
    assert spec.layers is None or len(spec.layers) == 0


def test_smooth_ci_ribbon_layer_uses_ci_columns(df):
    spec = fe.Chart(df).mark_smooth(ci=0.95).encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert "ci_lower" in json_str, "ribbon layer should bind y to ci_lower"
    assert "ci_upper" in json_str, "ribbon layer should bind y2 to ci_upper"


def test_smooth_method_lm_with_ci(df):
    spec = fe.Chart(df).mark_smooth(method="lm", ci=0.9).encode(x="x", y="y")._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 2


def test_smooth_ci_renders(df):
    """End-to-end: ci= path renders SVG without raising."""
    svg = fe.Chart(df).mark_smooth(ci=0.95).encode(x="x", y="y").to_svg()
    assert "<svg" in svg
