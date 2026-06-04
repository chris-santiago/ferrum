"""10M-row stress tests — validates ferrum's "10M+ rows" scale claim.

These tests push beyond the 1M-row tests in test_scale_rendering.py to
confirm the scale ceiling advertised in the README is real.

All tests are marked ``slow`` and are skipped by default:
    uv run pytest              # skips slow tests
    uv run pytest -m slow      # runs all slow tests including these
    uv run pytest tests/test_scale_10m.py -m slow -v

Baseline results (M4 MacBook Pro, 2026-05-19, 14.39s total):
    Scatter 10M (auto-raster)  0.69s
    Histogram 10M              0.58s
    Hexbin 10M                 1.07s
    Line 10M                   3.82s
    Area 10M                   4.14s
    Memory check               0.68s
"""

from __future__ import annotations

import time
import tracemalloc
import warnings

import numpy as np
import polars as pl
import pytest

import ferrum as fm

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 10_000_000


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _df_quant(n: int) -> pl.DataFrame:
    rng = _rng()
    return pl.DataFrame(
        {
            "x": rng.normal(0, 1, n).tolist(),
            "y": rng.normal(0, 1, n).tolist(),
        }
    )


def _df_timeseries(n: int) -> pl.DataFrame:
    rng = _rng()
    return pl.DataFrame(
        {
            "t": list(range(n)),
            "y": np.cumsum(rng.normal(0, 1, n)).tolist(),
        }
    )


def _count_elements(svg: str, tag: str) -> int:
    return svg.count(f"<{tag}")


# ---------------------------------------------------------------------------
# Scatter (mark_point) with auto-raster — the primary 10M use case
# ---------------------------------------------------------------------------


class TestScatter10M:
    def test_renders_without_crash(self):
        """10M scatter completes, SVG is valid, finishes in < 60s."""
        df = _df_quant(N)
        t0 = time.monotonic()
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300)
            .to_svg()
        )
        elapsed = time.monotonic() - t0
        assert "<svg" in svg
        assert elapsed < 60.0, f"10M scatter took {elapsed:.2f}s (limit 60s)"

    def test_auto_raster_compresses_svg(self):
        """Auto-raster fires at 10M rows: SVG output stays under 5MB."""
        df = _df_quant(N)
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300)
            .to_svg()
        )
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 5, f"10M scatter SVG is {size_mb:.1f}MB; expected <5MB (auto-raster)"

    def test_auto_raster_warning_emitted(self):
        """A UserWarning matching 'Auto-raster' is emitted for 10M scatter."""
        df = _df_quant(N)
        chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=400, height=300)
        with pytest.warns(UserWarning, match="Auto-raster"):
            chart.to_svg()


# ---------------------------------------------------------------------------
# Histogram (binning mark)
# ---------------------------------------------------------------------------


class TestHistogram10M:
    def test_renders_and_bins(self):
        """10M-row histogram renders in < 30s and produces >= 5 bins."""
        df = _df_quant(N)
        t0 = time.monotonic()
        svg = fm.displot(df, x="x").properties(width=400, height=300).to_svg()
        elapsed = time.monotonic() - t0
        assert "<svg" in svg
        assert elapsed < 30.0, f"10M histogram took {elapsed:.2f}s (limit 30s)"
        rects = _count_elements(svg, "rect")
        assert rects >= 5, f"histogram should have >=5 bins; got {rects}"

    def test_svg_size(self):
        """Histogram output stays compact regardless of input size (binning aggregates)."""
        df = _df_quant(N)
        svg = fm.displot(df, x="x").properties(width=400, height=300).to_svg()
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 2, f"10M histogram SVG is {size_mb:.1f}MB; expected <2MB"


# ---------------------------------------------------------------------------
# Hexbin (spatial binning)
# ---------------------------------------------------------------------------


class TestHexbin10M:
    def test_renders(self):
        """10M-row hexbin renders in < 30s and produces hexagons."""
        df = _df_quant(N)
        t0 = time.monotonic()
        svg = (
            fm.Chart(df)
            .mark_hex(aggregate="count")
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300)
            .to_svg()
        )
        elapsed = time.monotonic() - t0
        assert "<svg" in svg
        assert elapsed < 30.0, f"10M hexbin took {elapsed:.2f}s (limit 30s)"

    def test_svg_size(self):
        """Hexbin output stays compact regardless of input size (spatial binning aggregates)."""
        df = _df_quant(N)
        svg = (
            fm.Chart(df)
            .mark_hex(aggregate="count")
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300)
            .to_svg()
        )
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 2, f"10M hexbin SVG is {size_mb:.1f}MB; expected <2MB"


# ---------------------------------------------------------------------------
# Line (polyline path serialization)
# ---------------------------------------------------------------------------


class TestLine10M:
    def test_renders(self):
        """10M-point timeseries line renders in < 60s."""
        df = _df_timeseries(N)
        t0 = time.monotonic()
        svg = (
            fm.Chart(df)
            .mark_line()
            .encode(x="t:Q", y="y:Q")
            .properties(width=400, height=300)
            .to_svg()
        )
        elapsed = time.monotonic() - t0
        assert "<svg" in svg
        assert elapsed < 60.0, f"10M line took {elapsed:.2f}s (limit 60s)"


# ---------------------------------------------------------------------------
# Area
# ---------------------------------------------------------------------------


class TestArea10M:
    def test_renders(self):
        """10M-point timeseries area renders in < 60s."""
        df = _df_timeseries(N)
        t0 = time.monotonic()
        svg = (
            fm.Chart(df)
            .mark_area()
            .encode(x="t:Q", y="y:Q")
            .properties(width=400, height=300)
            .to_svg()
        )
        elapsed = time.monotonic() - t0
        assert "<svg" in svg
        assert elapsed < 60.0, f"10M area took {elapsed:.2f}s (limit 60s)"


# ---------------------------------------------------------------------------
# Memory sanity
# ---------------------------------------------------------------------------


class TestMemory10M:
    def test_scatter_peak_memory_under_8gb(self):
        """Peak memory during a 10M scatter render stays under 8GB.

        This catches catastrophic allocations (e.g., materialising every
        SVG element before serialising) while remaining generous enough
        to allow for normal Arrow + raster buffers.
        """
        df = _df_quant(N)
        chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=400, height=300)
        tracemalloc.start()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            chart.to_svg()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_gb = peak / (1024**3)
        assert peak_gb < 8, f"Peak memory during 10M scatter is {peak_gb:.2f}GB (limit 8GB)"
