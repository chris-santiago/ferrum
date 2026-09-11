"""Scale rendering tests — verify ferrum handles datasets beyond toy size.

Ferrum positions itself as handling scale better than Altair's 5000-row
default limit. These tests verify that mark types render correctly at
10k, 50k, and 1M rows without crashing, producing degenerate output,
or taking unreasonably long.

Tests are marked with @pytest.mark.slow so they are skipped by default:
    uv run pytest              # skips scale tests (addopts in pyproject.toml)
    uv run pytest -m slow      # run only scale tests
    uv run pytest -m ""        # run everything including scale tests
"""

from __future__ import annotations

import os
import re
import tempfile
import time
import warnings

import numpy as np
import polars as pl
import pytest

import ferrum as fm

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng():
    return np.random.default_rng(42)


def _df_quant(n: int) -> pl.DataFrame:
    rng = _rng()
    return pl.DataFrame(
        {
            "x": rng.normal(0, 1, n).tolist(),
            "y": rng.normal(0, 1, n).tolist(),
        }
    )


def _df_quant_valued(n: int) -> pl.DataFrame:
    rng = _rng()
    return pl.DataFrame(
        {
            "x": rng.normal(0, 1, n).tolist(),
            "y": rng.normal(0, 1, n).tolist(),
            "val": rng.uniform(0, 1, n).tolist(),
        }
    )


def _df_grouped(n: int, n_groups: int = 5) -> pl.DataFrame:
    rng = _rng()
    return pl.DataFrame(
        {
            "x": rng.normal(0, 1, n).tolist(),
            "y": rng.normal(0, 1, n).tolist(),
            "group": [f"g{i % n_groups}" for i in range(n)],
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


def _render_svg(chart: fm.Chart) -> str:
    return chart.properties(width=400, height=300).to_svg()


def _count_elements(svg: str, tag: str) -> int:
    return svg.count(f"<{tag}")


def _distinct_fills(svg: str) -> set[str]:
    return set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg))


# ---------------------------------------------------------------------------
# Scatter (mark_point) — 1 SVG circle per row
# ---------------------------------------------------------------------------


class TestScatterScale:
    def test_10k_renders(self):
        svg = _render_svg(fm.Chart(_df_quant(10_000)).mark_point().encode(x="x:Q", y="y:Q"))
        assert _count_elements(svg, "circle") == 10_000

    def test_50k_renders(self):
        svg = _render_svg(fm.Chart(_df_quant(50_000)).mark_point().encode(x="x:Q", y="y:Q"))
        assert _count_elements(svg, "circle") == 50_000

    def test_50k_under_1_second(self):
        df = _df_quant(50_000)
        chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=400, height=300)
        t0 = time.monotonic()
        chart.to_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"50k scatter took {elapsed:.2f}s (limit 1.0s)"

    def test_10k_grouped_color(self):
        df = _df_grouped(10_000, n_groups=5)
        svg = _render_svg(fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="group:N"))
        circles = _count_elements(svg, "circle")
        # 10k data circles + legend swatches (one per group)
        assert circles >= 10_000, f"expected ≥10k circles; got {circles}"
        assert len(_distinct_fills(svg)) >= 5


# ---------------------------------------------------------------------------
# Line (mark_line) — polyline, not per-row elements
# ---------------------------------------------------------------------------


class TestLineScale:
    def test_10k_renders(self):
        svg = _render_svg(fm.Chart(_df_timeseries(10_000)).mark_line().encode(x="t:Q", y="y:Q"))
        assert "<svg" in svg
        assert "polyline" in svg.lower() or "path" in svg.lower() or "d=" in svg

    def test_50k_renders(self):
        svg = _render_svg(fm.Chart(_df_timeseries(50_000)).mark_line().encode(x="t:Q", y="y:Q"))
        assert "<svg" in svg

    def test_50k_under_1_second(self):
        df = _df_timeseries(50_000)
        chart = fm.Chart(df).mark_line().encode(x="t:Q", y="y:Q").properties(width=400, height=300)
        t0 = time.monotonic()
        chart.to_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"50k line took {elapsed:.2f}s (limit 1.0s)"

    def test_10k_multi_group(self):
        df = _df_grouped(10_000, n_groups=10)
        svg = _render_svg(fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q", color="group:N"))
        assert "<svg" in svg
        assert len(_distinct_fills(svg)) >= 5


# ---------------------------------------------------------------------------
# Bar (mark_bar) — many categories
# ---------------------------------------------------------------------------


class TestBarScale:
    def test_500_categories(self):
        rng = _rng()
        df = pl.DataFrame(
            {
                "cat": [f"c{i}" for i in range(500)],
                "val": rng.uniform(0, 100, 500).tolist(),
            }
        )
        svg = _render_svg(fm.Chart(df).mark_bar().encode(x="cat:N", y="val:Q"))
        rects = _count_elements(svg, "rect")
        assert rects >= 500, f"expected ≥500 rects; got {rects}"

    def test_grouped_bar_200_categories_5_groups(self):
        rng = _rng()
        n = 1000
        df = pl.DataFrame(
            {
                "cat": [f"c{i % 200}" for i in range(n)],
                "group": [f"g{i % 5}" for i in range(n)],
                "val": rng.uniform(0, 100, n).tolist(),
            }
        )
        svg = _render_svg(fm.Chart(df).mark_bar().encode(x="cat:N", y="val:Q", color="group:N"))
        assert "<svg" in svg
        assert len(_distinct_fills(svg)) >= 5


# ---------------------------------------------------------------------------
# Histogram (displot) — binning large data
# ---------------------------------------------------------------------------


class TestHistogramScale:
    def test_10k(self):
        svg = _render_svg(fm.displot(_df_quant(10_000), x="x"))
        assert "<svg" in svg
        rects = _count_elements(svg, "rect")
        assert rects >= 5, f"histogram should have ≥5 bins; got {rects}"

    def test_50k(self):
        svg = _render_svg(fm.displot(_df_quant(50_000), x="x"))
        assert "<svg" in svg

    def test_50k_under_1_second(self):
        df = _df_quant(50_000)
        t0 = time.monotonic()
        fm.displot(df, x="x").properties(width=400, height=300).to_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"50k histogram took {elapsed:.2f}s (limit 1.0s)"


# ---------------------------------------------------------------------------
# Hexbin (mark_hex) — spatial binning
# ---------------------------------------------------------------------------


class TestHexbinScale:
    def test_10k(self):
        svg = _render_svg(
            fm.Chart(_df_quant(10_000)).mark_hex(aggregate="count").encode(x="x:Q", y="y:Q")
        )
        assert "<svg" in svg
        paths = svg.count('d="M')
        assert paths >= 10, f"hexbin should have ≥10 hexagons; got {paths}"

    def test_50k(self):
        svg = _render_svg(
            fm.Chart(_df_quant(50_000)).mark_hex(aggregate="count").encode(x="x:Q", y="y:Q")
        )
        assert "<svg" in svg

    def test_50k_under_1_second(self):
        df = _df_quant(50_000)
        chart = (
            fm.Chart(df)
            .mark_hex(aggregate="count")
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300)
        )
        t0 = time.monotonic()
        chart.to_svg()
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"50k hexbin took {elapsed:.2f}s (limit 1.0s)"


# ---------------------------------------------------------------------------
# Area (mark_area)
# ---------------------------------------------------------------------------


class TestAreaScale:
    def test_10k(self):
        svg = _render_svg(fm.Chart(_df_timeseries(10_000)).mark_area().encode(x="t:Q", y="y:Q"))
        assert "<svg" in svg

    def test_50k(self):
        svg = _render_svg(fm.Chart(_df_timeseries(50_000)).mark_area().encode(x="t:Q", y="y:Q"))
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# SVG size sanity — don't produce absurdly large output
# ---------------------------------------------------------------------------


class TestSvgSizeBounds:
    """SVG output should stay under reasonable size limits.
    A 50k scatter at ~70 bytes/circle ≈ 3.5MB is fine.
    Anything over 20MB suggests a rendering bug (duplicate elements, etc.)."""

    def test_50k_scatter_under_10mb(self):
        svg = _render_svg(fm.Chart(_df_quant(50_000)).mark_point().encode(x="x:Q", y="y:Q"))
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 10, f"50k scatter SVG is {size_mb:.1f}MB (limit 10MB)"

    def test_50k_line_under_5mb(self):
        svg = _render_svg(fm.Chart(_df_timeseries(50_000)).mark_line().encode(x="t:Q", y="y:Q"))
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 5, f"50k line SVG is {size_mb:.1f}MB (limit 5MB)"

    def test_50k_hexbin_under_5mb(self):
        svg = _render_svg(
            fm.Chart(_df_quant(50_000)).mark_hex(aggregate="count").encode(x="x:Q", y="y:Q")
        )
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 5, f"50k hexbin SVG is {size_mb:.1f}MB (limit 5MB)"


# ---------------------------------------------------------------------------
# 1M rows — proves ferrum handles real-world dataset sizes
# ---------------------------------------------------------------------------


class TestMillionRows:
    """1M-row tests for the mark types that scale well (binning marks produce
    compact SVG regardless of input size) and for per-element marks to verify
    they don't crash."""

    def test_scatter_1m_renders(self):
        """1M scatter: auto-raster substitutes mark_raster, producing compact
        SVG output. Must not crash or take unreasonably long."""
        df = _df_quant(1_000_000)
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
        assert elapsed < 5.0, f"1M scatter took {elapsed:.2f}s (limit 5.0s)"

    def test_scatter_1m_svg_size(self):
        """1M scatter SVG should be compact (<2MB) thanks to auto-raster
        substitution. Previously ~57MB (one circle per row)."""
        svg = (
            fm.Chart(_df_quant(1_000_000))
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300)
            .to_svg()
        )
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 2, f"1M scatter SVG is {size_mb:.1f}MB (limit 2MB with auto-raster)"

    def test_line_1m_renders(self):
        df = _df_timeseries(1_000_000)
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
        assert elapsed < 5.0, f"1M line took {elapsed:.2f}s (limit 5.0s)"

    def test_histogram_1m_renders_fast(self):
        """Histogram bins 1M rows into ~20 bars — should be near-instant."""
        df = _df_quant(1_000_000)
        t0 = time.monotonic()
        svg = fm.displot(df, x="x").properties(width=400, height=300).to_svg()
        elapsed = time.monotonic() - t0
        assert "<svg" in svg
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 2, f"1M histogram SVG should be small; got {size_mb:.1f}MB"
        assert elapsed < 2.0, f"1M histogram took {elapsed:.2f}s (limit 2.0s)"

    def test_hexbin_1m_renders_fast(self):
        """Hexbin spatially bins 1M points — output is compact regardless of input."""
        df = _df_quant(1_000_000)
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
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 2, f"1M hexbin SVG should be small; got {size_mb:.1f}MB"
        assert elapsed < 2.0, f"1M hexbin took {elapsed:.2f}s (limit 2.0s)"

    def test_area_1m_renders(self):
        df = _df_timeseries(1_000_000)
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
        assert elapsed < 5.0, f"1M area took {elapsed:.2f}s (limit 5.0s)"


# ---------------------------------------------------------------------------
# Auto-raster policy tests
# ---------------------------------------------------------------------------


class TestAutoRaster:
    """Test the auto-raster policy that substitutes mark_raster when the mark
    count exceeds ``raster_threshold`` (default 500k)."""

    def test_auto_raster_fires_on_1m_scatter(self):
        """1M mark_point with x:Q, y:Q triggers auto-raster: SVG < 2 MB and
        a UserWarning matching 'Auto-raster' is emitted."""
        df = _df_quant(1_000_000)
        chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=400, height=300)
        with pytest.warns(UserWarning, match="Auto-raster"):
            svg = chart.to_svg()
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 2, f"auto-raster SVG is {size_mb:.1f}MB; expected <2MB"

    def test_auto_raster_disabled_with_none_threshold(self):
        """raster_threshold=None disables auto-raster: full per-element SVG
        (>10MB) is produced with no warning."""
        df = _df_quant(1_000_000)
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(
                width=400,
                height=300,
                render_config=fm.RenderConfig(raster_threshold=None),
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        auto_raster_warnings = [w for w in caught if "Auto-raster" in str(w.message)]
        assert not auto_raster_warnings, "Expected no auto-raster warning"
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb > 10, f"Expected full SVG >10MB; got {size_mb:.1f}MB"

    def test_auto_raster_error_mode(self):
        """raster_behavior='error' raises ValueError instead of substituting."""
        df = _df_quant(1_000_000)
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(render_config=fm.RenderConfig(raster_behavior="error"))
        )
        with pytest.raises(ValueError, match="threshold"):
            chart.to_svg()

    def test_auto_raster_silent_mode(self):
        """raster_behavior='silent' substitutes without warning."""
        df = _df_quant(1_000_000)
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(
                width=400,
                height=300,
                render_config=fm.RenderConfig(raster_behavior="silent"),
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        auto_raster_warnings = [w for w in caught if "Auto-raster" in str(w.message)]
        assert not auto_raster_warnings, "silent mode should emit no warning"
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb < 2, f"Expected compact SVG <2MB; got {size_mb:.1f}MB"

    def test_auto_raster_skips_categorical_color_encoded(self):
        """Charts with categorical/ordinal/temporal color encoding do NOT
        get auto-raster; a specific guidance warning names the exclusion,
        the reason, and the manual workaround (F-L14-04)."""
        df = _df_grouped(1_000_000, n_groups=5)
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color="group:N")
            .properties(width=400, height=300)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        # Should NOT have auto-raster substitution warning
        auto_raster_warnings = [w for w in caught if "Auto-raster" in str(w.message)]
        assert not auto_raster_warnings, "auto-raster should not fire with categorical color"
        # Should have a specific guidance warning naming the exclusion, its
        # reason, and the manual workaround -- not the generic large-chart
        # warning used for other ineligibility gates.
        guidance_warnings = [w for w in caught if "mark_raster" in str(w.message)]
        assert guidance_warnings, "Expected guidance warning for large chart with color"
        msg = str(guidance_warnings[0].message)
        assert "categorical" in msg, f"warning should name the categorical exclusion: {msg!r}"
        assert "discard" in msg, f"warning should state the reason (discards info): {msg!r}"
        assert "mark_raster(aggregate=" in msg, f"warning should show the manual workaround: {msg!r}"
        # SVG should be large (no substitution)
        size_mb = len(svg) / (1024 * 1024)
        assert size_mb > 10, f"Expected full SVG >10MB; got {size_mb:.1f}MB"

    def test_auto_raster_substitutes_quantitative_color(self):
        """Charts with a quantitative color encoding DO get auto-raster:
        the substitution uses aggregate='mean', field=<color field>, so the
        color information survives rasterisation instead of being discarded
        (F-L14-04)."""
        df = _df_quant_valued(2_000)
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color="val:Q")
            .properties(
                width=400,
                height=300,
                render_config=fm.RenderConfig(raster_threshold=0, raster_behavior="silent"),
            )
        )
        svg = chart.to_svg()
        assert "<image" in svg, "quantitative color should trigger raster substitution"

    def test_auto_raster_quantitative_color_uses_mean_of_color_field(self):
        """The quantitative-color substitution's aggregate is genuinely
        driven by the color field -- proven by comparing against an
        explicit mark_raster(aggregate='mean', field='val') built by hand."""
        df = _df_quant_valued(2_000)

        auto_svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color="val:Q")
            .properties(
                width=400,
                height=300,
                render_config=fm.RenderConfig(raster_threshold=0, raster_behavior="silent"),
            )
            .to_svg()
        )
        manual_svg = (
            fm.Chart(df)
            .mark_raster(aggregate="mean", field="val", cmap="viridis")
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300)
            .to_svg()
        )
        assert auto_svg == manual_svg, (
            "auto-raster quantitative-color substitution should match a hand-built "
            "mark_raster(aggregate='mean', field=<color field>)"
        )

    def test_auto_raster_quantitative_color_scheme_overrides_render_config_scheme(self):
        """The color channel's own declared `scheme=` -- not
        `RenderConfig.raster_scheme` -- is what reaches the substituted
        `mark_raster`'s `cmap`, when the two disagree.

        Uses `fm.Color(field, type="Q", scheme=...)` (not the bare-string
        shorthand, which never populates a `scheme` kwarg at all) so this
        actually exercises the `color_enc._kwargs.get("scheme")` extraction
        path in `_apply_auto_raster`. `RenderConfig.raster_scheme` is set to
        a deliberately different colormap ("magma") than the color channel's
        declared scheme ("plasma"): if the substitution read
        `cfg.raster_scheme` instead of (or in addition to) the color
        channel's own scheme, this chart would render magma-colored pixels
        and diverge byte-for-byte from a hand-built
        `mark_raster(aggregate="mean", field="val", cmap="plasma")` chart.
        """
        df = _df_quant_valued(2_000)

        auto_svg = (
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x:Q",
                y="y:Q",
                color=fm.Color("val", type="Q", scheme="plasma"),
            )
            .properties(
                width=400,
                height=300,
                render_config=fm.RenderConfig(
                    raster_threshold=0, raster_behavior="silent", raster_scheme="magma"
                ),
            )
            .to_svg()
        )
        manual_svg = (
            fm.Chart(df)
            .mark_raster(aggregate="mean", field="val", cmap="plasma")
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300)
            .to_svg()
        )
        assert auto_svg == manual_svg, (
            "the color channel's own declared scheme= must reach the substituted "
            "mark_raster's cmap, overriding RenderConfig.raster_scheme"
        )
        # Also prove the two schemes actually render differently -- otherwise
        # a broken extraction that fell through to cfg.raster_scheme="magma"
        # could accidentally byte-match by coincidence if magma and plasma
        # happened to render identically (they don't, but assert it directly).
        magma_svg = (
            fm.Chart(df)
            .mark_raster(aggregate="mean", field="val", cmap="magma")
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300)
            .to_svg()
        )
        assert auto_svg != magma_svg, (
            "sanity check: plasma and magma must render different pixel payloads, "
            "or this test cannot discriminate between the two scheme sources"
        )

    def test_auto_raster_explicit_raster_aggregate_count_matches_unset(self):
        """Documented edge case: an explicit `raster_aggregate="count"` (the
        default value) with no `raster_field` is indistinguishable from
        never having set either field at all -- `RenderConfig` is a frozen
        dataclass with no "was this explicitly passed" sentinel. Both fall
        through to the color-driven `mean`/<color field> substitution for a
        quantitative color chart, rather than "explicit count, no
        substitution." This pins the documented outcome (see
        `_apply_auto_raster`'s docstring and `RenderConfig.raster_aggregate`'s
        docstring) so the disclosure cannot silently rot if the
        divergence-detection expression changes.
        """
        df = _df_quant_valued(2_000)

        explicit_count_svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color="val:Q")
            .properties(
                width=400,
                height=300,
                render_config=fm.RenderConfig(
                    raster_threshold=0, raster_behavior="silent", raster_aggregate="count"
                ),
            )
            .to_svg()
        )
        unset_svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color="val:Q")
            .properties(
                width=400,
                height=300,
                render_config=fm.RenderConfig(raster_threshold=0, raster_behavior="silent"),
            )
            .to_svg()
        )
        assert explicit_count_svg == unset_svg, (
            "documented edge case: explicit raster_aggregate='count' (matching the "
            "default) must be indistinguishable from never setting it -- both fall "
            "through to the mean/<color field> substitution"
        )
        # And both must actually be the color-driven mean substitution, not a
        # bare count substitution that happens to render identically.
        manual_mean_svg = (
            fm.Chart(df)
            .mark_raster(aggregate="mean", field="val", cmap="viridis")
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300)
            .to_svg()
        )
        assert unset_svg == manual_mean_svg, (
            "the unset/explicit-count-default case must resolve to the mean/<color "
            "field> substitution, per the documented divergence rule"
        )

    def test_auto_raster_explicit_config_wins_over_quantitative_color(self):
        """An explicit RenderConfig(raster_aggregate=..., raster_field=...)
        wins wholesale over the color-driven substitution: the substituted
        raster ignores the color field entirely and uses the user's
        aggregate/field, matching a colorless chart with the same config
        byte-for-byte."""
        df = _df_quant_valued(2_000)
        cfg = fm.RenderConfig(
            raster_threshold=0,
            raster_behavior="silent",
            raster_aggregate="sum",
            raster_field="val",
        )

        with_color = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color="val:Q")
            .properties(width=400, height=300, render_config=cfg)
            .to_svg()
        )
        without_color = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300, render_config=cfg)
            .to_svg()
        )
        assert with_color == without_color, (
            "explicit raster_aggregate/raster_field must win wholesale over the "
            "color-driven substitution -- the color field must not leak in"
        )

    def test_auto_raster_disabled_threshold_none_with_quantitative_color(self):
        """raster_threshold=None still disables auto-raster for quantitative
        color charts -- the substitution mechanism doesn't fire at all when
        auto-raster itself is off."""
        df = _df_quant_valued(2_000)
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color="val:Q")
            .properties(
                width=400,
                height=300,
                render_config=fm.RenderConfig(raster_threshold=None),
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        auto_raster_warnings = [w for w in caught if "Auto-raster" in str(w.message)]
        assert not auto_raster_warnings
        assert "<image" not in svg, "disabled auto-raster must not substitute"

    def test_auto_raster_below_threshold_no_change(self):
        """Charts below the threshold are unaffected."""
        df = _df_quant(10_000)
        chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=400, height=300)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        auto_raster_warnings = [w for w in caught if "Auto-raster" in str(w.message)]
        assert not auto_raster_warnings
        assert _count_elements(svg, "circle") == 10_000

    def test_auto_raster_skips_line_marks(self):
        """Line marks are not eligible for auto-raster."""
        df = _df_timeseries(1_000_000)
        chart = fm.Chart(df).mark_line().encode(x="t:Q", y="y:Q").properties(width=400, height=300)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        auto_raster_warnings = [w for w in caught if "Auto-raster" in str(w.message)]
        assert not auto_raster_warnings
        assert "<svg" in svg

    def test_auto_raster_skips_hex_marks(self):
        """Hex marks are composite (aggregate) and skip auto-raster."""
        df = _df_quant(1_000_000)
        chart = (
            fm.Chart(df)
            .mark_hex(aggregate="count")
            .encode(x="x:Q", y="y:Q")
            .properties(width=400, height=300)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        auto_raster_warnings = [w for w in caught if "Auto-raster" in str(w.message)]
        assert not auto_raster_warnings
        assert "<svg" in svg

    def test_auto_raster_save_png(self):
        """Auto-raster applies to save() as well (via _render_inputs)."""
        df = _df_quant(1_000_000)
        chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=400, height=300)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            with pytest.warns(UserWarning, match="Auto-raster"):
                chart.save(path)
            size_kb = os.path.getsize(path) / 1024
            assert size_kb < 500, f"PNG should be compact; got {size_kb:.0f}KB"
        finally:
            os.unlink(path)
