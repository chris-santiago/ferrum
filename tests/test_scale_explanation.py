"""Scale tests for SHAP, ICE, and PDP figure functions.

Validates that ferrum handles explainability plots at real-world sample sizes
without crashing, producing degenerate output, or exceeding time budgets.

Scale tiers:
- SHAP beeswarm/bar: 5k and 10k samples × 20 features
- SHAP waterfall: 10k samples (single-sample plot, but SHAP values computed over full X)
- ICE (individual): 2k samples × 50 grid points × 3 features
- PDP+ICE (both): 5k samples × 100 grid points × 5 features

Tests are marked with @pytest.mark.slow so they are skipped by default:
    uv run pytest              # skips scale tests
    uv run pytest -m slow      # run only scale tests
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

import ferrum as fm

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regression(n_samples: int, n_features: int = 20, *, seed: int = 42):
    """Build a regression dataset and fit a RandomForestRegressor."""
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    coefs = rng.normal(0, 1, n_features)
    y = X @ coefs + rng.normal(0, 0.1, n_samples)

    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=seed)
    model.fit(X, y)

    feature_names = [f"f{i}" for i in range(n_features)]
    X_df = pl.DataFrame({name: X[:, i].tolist() for i, name in enumerate(feature_names)})
    return model, X_df, y


def _make_classification(n_samples: int, n_features: int = 20, *, seed: int = 42):
    """Build a binary classification dataset and fit a RandomForestClassifier."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    coefs = rng.normal(0, 1, n_features)
    y = (X @ coefs > 0).astype(int)

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=seed)
    model.fit(X, y)

    feature_names = [f"f{i}" for i in range(n_features)]
    X_df = pl.DataFrame({name: X[:, i].tolist() for i, name in enumerate(feature_names)})
    return model, X_df, y


# ---------------------------------------------------------------------------
# SHAP Beeswarm at scale
# ---------------------------------------------------------------------------


class TestShapBeeswarmScale:
    """SHAP beeswarm produces n_samples × n_features circles.
    5k samples × 20 features = 100k circles; 10k × 20 = 200k circles."""

    def test_5k_samples_renders(self):
        model, X, y = _make_regression(5_000)
        chart = fm.shap_beeswarm_chart(model, X, y, random_state=0)
        svg = chart.show_svg()
        assert "<svg" in svg
        circles = svg.count("<circle ")
        assert circles >= 5_000, f"Expected ≥5k circles; got {circles}"

    def test_5k_samples_under_10_seconds(self):
        model, X, y = _make_regression(5_000)
        t0 = time.monotonic()
        fm.shap_beeswarm_chart(model, X, y, random_state=0).show_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 10.0, f"5k beeswarm took {elapsed:.2f}s (limit 10s)"

    def test_10k_samples_renders(self):
        model, X, y = _make_regression(10_000)
        chart = fm.shap_beeswarm_chart(model, X, y, random_state=0)
        svg = chart.show_svg()
        assert "<svg" in svg
        circles = svg.count("<circle ")
        assert circles >= 10_000, f"Expected ≥10k circles; got {circles}"

    def test_10k_samples_under_20_seconds(self):
        model, X, y = _make_regression(10_000)
        t0 = time.monotonic()
        fm.shap_beeswarm_chart(model, X, y, random_state=0).show_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 20.0, f"10k beeswarm took {elapsed:.2f}s (limit 20s)"

    def test_5k_max_display_limits_features(self):
        """max_display=5 with 20 features: only 5k × 5 = 25k circles."""
        model, X, y = _make_regression(5_000)
        svg = fm.shap_beeswarm_chart(model, X, y, max_display=5, random_state=0).show_svg()
        circles = svg.count("<circle ")
        assert 25_000 <= circles <= 26_000, (
            f"Expected ~25k circles for max_display=5; got {circles}"
        )

    def test_5k_classifier_renders(self):
        """Binary classification SHAP at scale."""
        model, X, y = _make_classification(5_000)
        svg = fm.shap_beeswarm_chart(model, X, y, random_state=0).show_svg()
        assert "<svg" in svg
        circles = svg.count("<circle ")
        assert circles >= 5_000


# ---------------------------------------------------------------------------
# SHAP Bar at scale
# ---------------------------------------------------------------------------


class TestShapBarScale:
    """SHAP bar aggregates to one bar per feature — output is always compact.
    The cost is in SHAP computation, not rendering."""

    def test_5k_samples_renders(self):
        model, X, y = _make_regression(5_000)
        svg = fm.shap_bar_chart(model, X, y, random_state=0).show_svg()
        assert "<svg" in svg
        rects = svg.count("<rect ")
        assert rects >= 5, f"Expected ≥5 bars; got {rects}"

    def test_10k_samples_under_15_seconds(self):
        model, X, y = _make_regression(10_000)
        t0 = time.monotonic()
        fm.shap_bar_chart(model, X, y, random_state=0).show_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 15.0, f"10k bar chart took {elapsed:.2f}s (limit 15s)"

    def test_10k_svg_compact(self):
        """Bar chart SVG should be compact regardless of sample count."""
        model, X, y = _make_regression(10_000)
        svg = fm.shap_bar_chart(model, X, y, random_state=0).show_svg()
        size_kb = len(svg) / 1024
        assert size_kb < 1000, f"Bar chart SVG is {size_kb:.0f}KB (limit 1MB)"


# ---------------------------------------------------------------------------
# SHAP Waterfall at scale
# ---------------------------------------------------------------------------


class TestShapWaterfallScale:
    """Waterfall explains a single sample but computes SHAP over the full X.
    10k samples × 20 features for SHAP computation; output is one waterfall."""

    def test_10k_samples_renders(self):
        model, X, y = _make_regression(10_000)
        svg = fm.shap_waterfall_chart(model, X, y, sample_idx=0, random_state=0).show_svg()
        assert "<svg" in svg

    def test_10k_under_15_seconds(self):
        model, X, y = _make_regression(10_000)
        t0 = time.monotonic()
        fm.shap_waterfall_chart(model, X, y, sample_idx=0, random_state=0).show_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 15.0, f"10k waterfall took {elapsed:.2f}s (limit 15s)"

    def test_10k_svg_compact(self):
        """Waterfall is a fixed-size chart; SVG should be compact."""
        model, X, y = _make_regression(10_000)
        svg = fm.shap_waterfall_chart(model, X, y, sample_idx=0, random_state=0).show_svg()
        size_kb = len(svg) / 1024
        assert size_kb < 1000, f"Waterfall SVG is {size_kb:.0f}KB (limit 1MB)"


# ---------------------------------------------------------------------------
# ICE (Individual Conditional Expectation) at scale
# ---------------------------------------------------------------------------


class TestICEScale:
    """ICE plots render one polyline per sample per feature panel.
    2k samples × 3 features = 6k polylines; 5k × 5 features = 25k polylines."""

    def test_2k_individual_3_features_renders(self):
        model, X, y = _make_regression(2_000, n_features=10)
        chart = fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2"],
            grid_resolution=50,
            kind="individual",
            random_state=0,
        )
        svg = chart.show_svg()
        assert "<svg" in svg
        polylines = svg.count("<polyline ")
        expected = 2_000 * 3
        assert polylines == expected, (
            f"Expected {expected} polylines (2k × 3 features); got {polylines}"
        )

    def test_2k_individual_under_10_seconds(self):
        model, X, y = _make_regression(2_000, n_features=10)
        t0 = time.monotonic()
        fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2"],
            grid_resolution=50,
            kind="individual",
            random_state=0,
        ).show_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 10.0, f"2k ICE (3 features) took {elapsed:.2f}s (limit 10s)"

    def test_5k_individual_5_features_renders(self):
        """5k × 5 = 25k polylines — tests rendering pipeline at high element count."""
        model, X, y = _make_regression(5_000, n_features=10)
        chart = fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2", "f3", "f4"],
            grid_resolution=50,
            kind="individual",
            random_state=0,
        )
        svg = chart.show_svg()
        assert "<svg" in svg
        polylines = svg.count("<polyline ")
        expected = 5_000 * 5
        assert polylines == expected, (
            f"Expected {expected} polylines (5k × 5 features); got {polylines}"
        )

    def test_5k_individual_under_30_seconds(self):
        model, X, y = _make_regression(5_000, n_features=10)
        t0 = time.monotonic()
        fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2", "f3", "f4"],
            grid_resolution=50,
            kind="individual",
            random_state=0,
        ).show_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 30.0, f"5k ICE (5 features) took {elapsed:.2f}s (limit 30s)"

    def test_5k_individual_svg_size(self):
        """25k polylines at 50 grid points — SVG should stay manageable."""
        model, X, y = _make_regression(5_000, n_features=10)
        svg = fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2", "f3", "f4"],
            grid_resolution=50,
            kind="individual",
            random_state=0,
        ).show_svg()
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 50, f"5k ICE SVG is {size_mb:.1f}MB (limit 50MB)"


# ---------------------------------------------------------------------------
# PDP (average only) at scale
# ---------------------------------------------------------------------------


class TestPDPScale:
    """PDP average produces one polyline per feature — compact output regardless
    of sample count.  Cost is in sklearn's partial_dependence computation."""

    def test_5k_average_5_features_renders(self):
        model, X, y = _make_regression(5_000, n_features=10)
        chart = fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2", "f3", "f4"],
            grid_resolution=100,
            kind="average",
            random_state=0,
        )
        svg = chart.show_svg()
        assert "<svg" in svg
        polylines = svg.count("<polyline ")
        assert polylines == 5, f"Expected 5 polylines (1 per feature); got {polylines}"

    def test_10k_average_10_features_under_15_seconds(self):
        model, X, y = _make_regression(10_000, n_features=10)
        t0 = time.monotonic()
        fm.pdp_chart(
            model,
            X,
            y,
            features=[f"f{i}" for i in range(10)],
            grid_resolution=100,
            kind="average",
            random_state=0,
        ).show_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 15.0, f"10k PDP (10 features, grid=100) took {elapsed:.2f}s (limit 15s)"

    def test_10k_average_svg_compact(self):
        """Average PDP is always compact regardless of sample count."""
        model, X, y = _make_regression(10_000, n_features=10)
        svg = fm.pdp_chart(
            model,
            X,
            y,
            features=[f"f{i}" for i in range(10)],
            grid_resolution=100,
            kind="average",
            random_state=0,
        ).show_svg()
        size_kb = len(svg) / 1024
        assert size_kb < 1000, f"PDP average SVG is {size_kb:.0f}KB (limit 1MB)"


# ---------------------------------------------------------------------------
# PDP + ICE (kind="both") at scale
# ---------------------------------------------------------------------------


class TestPDPBothScale:
    """kind='both' overlays per-sample ICE + 1 average line per feature.
    2k samples × 3 features = 6k+3 polylines; 5k × 5 = 25k+5 polylines."""

    def test_2k_both_3_features_renders(self):
        model, X, y = _make_regression(2_000, n_features=10)
        chart = fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2"],
            grid_resolution=50,
            kind="both",
            random_state=0,
        )
        svg = chart.show_svg()
        assert "<svg" in svg
        polylines = svg.count("<polyline ")
        expected = 3 * (2_000 + 1)
        assert polylines == expected, (
            f"Expected {expected} polylines (3 × (2k + 1 avg)); got {polylines}"
        )

    def test_2k_both_under_10_seconds(self):
        model, X, y = _make_regression(2_000, n_features=10)
        t0 = time.monotonic()
        fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2"],
            grid_resolution=50,
            kind="both",
            random_state=0,
        ).show_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 10.0, f"2k PDP both (3 feat) took {elapsed:.2f}s (limit 10s)"

    def test_5k_both_5_features_renders(self):
        model, X, y = _make_regression(5_000, n_features=10)
        chart = fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2", "f3", "f4"],
            grid_resolution=50,
            kind="both",
            random_state=0,
        )
        svg = chart.show_svg()
        assert "<svg" in svg
        polylines = svg.count("<polyline ")
        expected = 5 * (5_000 + 1)
        assert polylines == expected, (
            f"Expected {expected} polylines (5 × (5k + 1 avg)); got {polylines}"
        )

    def test_5k_both_centered_renders(self):
        """center=True path at scale."""
        model, X, y = _make_regression(5_000, n_features=10)
        chart = fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2"],
            grid_resolution=50,
            kind="both",
            center=True,
            random_state=0,
        )
        svg = chart.show_svg()
        assert "<svg" in svg
        polylines = svg.count("<polyline ")
        expected = 3 * (5_000 + 1)
        assert polylines == expected


# ---------------------------------------------------------------------------
# High grid resolution tests
# ---------------------------------------------------------------------------


class TestHighGridResolution:
    """High grid_resolution increases the number of points per polyline.
    2k samples × 200 grid points × 3 features = 6k polylines, each 200 pts."""

    def test_grid_200_renders(self):
        model, X, y = _make_regression(2_000, n_features=10)
        chart = fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2"],
            grid_resolution=200,
            kind="individual",
            random_state=0,
        )
        svg = chart.show_svg()
        assert "<svg" in svg
        polylines = svg.count("<polyline ")
        assert polylines == 2_000 * 3

    def test_grid_200_under_15_seconds(self):
        model, X, y = _make_regression(2_000, n_features=10)
        t0 = time.monotonic()
        fm.pdp_chart(
            model,
            X,
            y,
            features=["f0", "f1", "f2"],
            grid_resolution=200,
            kind="individual",
            random_state=0,
        ).show_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 15.0, f"2k ICE grid=200 (3 feat) took {elapsed:.2f}s (limit 15s)"


# ---------------------------------------------------------------------------
# Render-only scale tests (bypass sklearn/SHAP computation)
#
# These synthesize DataFrames identical to what ModelSource produces, then
# drive the chart rendering directly. This isolates the rendering pipeline's
# ability to handle element counts matching real-world explainability data.
# ---------------------------------------------------------------------------


class TestBeeswarmRenderScale:
    """Render pre-synthesized SHAP data at high sample counts.
    Real-world: 50k training rows × 20 features = 1M circles."""

    @staticmethod
    def _synth_shap(n_samples: int, n_features: int = 20) -> pl.DataFrame:
        rng = np.random.default_rng(42)
        features = [f"f{i}" for i in range(n_features)]
        n_rows = n_samples * n_features
        return pl.DataFrame(
            {
                "shap_value": rng.normal(0, 0.5, n_rows).tolist(),
                "feature": [features[i % n_features] for i in range(n_rows)],
                "feature_value": rng.uniform(0, 1, n_rows).tolist(),
                "feature_value_normalized": rng.normal(0, 1, n_rows).tolist(),
                "class_label": ["class_0"] * n_rows,
                "sample_id": [i // n_features for i in range(n_rows)],
            }
        )

    def test_50k_beeswarm_renders(self):
        """50k samples × 20 features = 1M circles via mark_shap_beeswarm."""
        df = self._synth_shap(50_000, 20)
        chart = (
            fm.Chart(df)
            .mark_shap_beeswarm(max_display=20)
            .encode(x="shap_value:Q", y="feature:N")
            .properties(width=600, height=400)
        )
        t0 = time.monotonic()
        svg = chart.show_svg()
        elapsed = time.monotonic() - t0
        assert "<svg" in svg
        assert elapsed < 10.0, f"50k beeswarm render took {elapsed:.2f}s (limit 10s)"

    def test_100k_beeswarm_renders(self):
        """100k samples × 10 features = 1M circles."""
        df = self._synth_shap(100_000, 10)
        chart = (
            fm.Chart(df)
            .mark_shap_beeswarm(max_display=10)
            .encode(x="shap_value:Q", y="feature:N")
            .properties(width=600, height=400)
        )
        t0 = time.monotonic()
        svg = chart.show_svg()
        elapsed = time.monotonic() - t0
        assert "<svg" in svg
        assert elapsed < 15.0, f"100k beeswarm render took {elapsed:.2f}s (limit 15s)"

    def test_50k_beeswarm_svg_size(self):
        """1M circles should produce manageable SVG (auto-raster may apply)."""
        df = self._synth_shap(50_000, 20)
        chart = (
            fm.Chart(df)
            .mark_shap_beeswarm(max_display=20)
            .encode(x="shap_value:Q", y="feature:N")
            .properties(width=600, height=400)
        )
        svg = chart.show_svg()
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 100, f"50k beeswarm SVG is {size_mb:.1f}MB (limit 100MB)"


class TestICERenderScale:
    """Render pre-synthesized ICE data at counts matching production datasets.
    Real-world: 10k samples × 50 grid points × 5 features = 50k polylines."""

    @staticmethod
    def _synth_ice(n_samples: int, n_features: int = 5, grid_resolution: int = 50):
        rng = np.random.default_rng(42)
        features = [f"f{i}" for i in range(n_features)]
        feature_col = []
        feature_value_col = []
        pd_value_col = []
        sample_id_col = []

        for fname in features:
            grid = np.linspace(-3, 3, grid_resolution)
            for sid in range(n_samples):
                base = rng.normal(0, 1)
                slope = rng.normal(0, 0.5)
                vals = base + slope * grid + rng.normal(0, 0.1, grid_resolution)
                feature_col.extend([fname] * grid_resolution)
                feature_value_col.extend(grid.tolist())
                pd_value_col.extend(vals.tolist())
                sample_id_col.extend([sid] * grid_resolution)

        df = pl.DataFrame(
            {
                "feature": feature_col,
                "feature_value": feature_value_col,
                "pd_value": pd_value_col,
                "sample_id": sample_id_col,
            }
        ).with_columns(
            pl.col("sample_id").cast(pl.Utf8).alias("_sample_id_str"),
        )
        return df, features

    def test_10k_ice_5_features_renders(self):
        """10k samples × 5 features = 50k polylines."""
        df, features = self._synth_ice(10_000, n_features=5, grid_resolution=50)
        chart = (
            fm.Chart(df.filter(pl.col("feature") == features[0]))
            .mark_line(opacity=0.2)
            .encode(x="feature_value:Q", y="pd_value:Q", detail="_sample_id_str:N")
            .properties(width=400, height=300)
        )
        t0 = time.monotonic()
        svg = chart.show_svg()
        elapsed = time.monotonic() - t0
        assert "<svg" in svg
        polylines = svg.count("<polyline ")
        assert polylines == 10_000, f"Expected 10k polylines; got {polylines}"
        assert elapsed < 10.0, f"10k ICE single panel took {elapsed:.2f}s (limit 10s)"

    def test_50k_ice_renders(self):
        """50k polylines across 5 feature panels (10k samples × 5 features)."""
        df, _ = self._synth_ice(10_000, n_features=5, grid_resolution=50)
        chart = (
            fm.Chart(df)
            .mark_line(opacity=0.2)
            .encode(x="feature_value:Q", y="pd_value:Q", detail="_sample_id_str:N")
            .facet(col="feature")
            .properties(width=300, height=200)
        )
        t0 = time.monotonic()
        svg = chart.show_svg()
        elapsed = time.monotonic() - t0
        assert "<svg" in svg
        assert elapsed < 15.0, f"50k ICE (faceted) took {elapsed:.2f}s (limit 15s)"

    def test_50k_ice_svg_size(self):
        """50k polylines × 50 grid points — SVG will be large but bounded."""
        df, _ = self._synth_ice(10_000, n_features=5, grid_resolution=50)
        svg = (
            fm.Chart(df.filter(pl.col("feature") == "f0"))
            .mark_line(opacity=0.2)
            .encode(x="feature_value:Q", y="pd_value:Q", detail="_sample_id_str:N")
            .properties(width=400, height=300)
            .show_svg()
        )
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 100, f"10k ICE panel SVG is {size_mb:.1f}MB (limit 100MB)"
