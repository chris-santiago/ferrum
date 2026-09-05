"""Directional regression tests for M2 (batch D task 5): `swarm(size=/spacing=)`
and `raster(resolution="screen")` respond to the chart's actual pixel
dimensions, closing F-L03-05/F-L03-06.

Before this fix, the one production transform-execution site
(`render/prepare/mod.rs::run_transforms_with_facet`) always constructed
`TransformContext::default()`, so `panel_pixel_size` was `None` in every real
render: swarm's placement math fell back to `radius_data = 1.0` regardless of
`size=`/`spacing=`, and `raster(resolution="screen")` always used a fixed
256x256 grid regardless of the chart's `width`/`height`. The fix threads a
pre-layout ESTIMATE (viewport minus theme padding, divided by the facet
grid's column/row counts when faceting) into `TransformContext` at that one
site (spec §4.2 of `2026-09-05-batch-d-transform-engine-design.md`).

These tests are directional, not exact-pixel: the estimate deliberately
excludes axis/legend/facet-strip gutters (unknowable pre-layout), so it is
close to but not identical to the chart's true rendered plot area. They pin
the observable, user-facing contract: bigger chart -> bigger raster grid;
bigger `size=` -> wider swarm spread; an explicit numeric `resolution=`
ignores the viewport entirely; a facet divides the per-cell estimate by the
facet grid's own dimensions.
"""

from __future__ import annotations

import base64
import re
import struct

import numpy as np
import polars as pl
import pytest

import ferrum as fm


def _raster_png_dims(svg: str) -> tuple[int, int]:
    """Decode the embedded raster `<image>`'s PNG IHDR chunk to get its real
    pixel dimensions -- the same mechanism the F-L03-06 finding's repro used
    (`struct.unpack(">II", png[16:24])`)."""
    match = re.search(r'href="data:image/png;base64,([^"]+)"', svg)
    assert match is not None, "expected an embedded base64 PNG <image> in the raster SVG"
    png = base64.b64decode(match.group(1))
    return struct.unpack(">II", png[16:24])


def _swarm_cx_spread(svg: str) -> float:
    """Width of the beeswarm's point cloud: max(cx) - min(cx) across every
    rendered `<circle>`. This is the quantity `size=`/`spacing=` actually
    control (how far apart points are placed along the cross-axis), not the
    drawn marker radius (a separate, unrelated `r=` attribute)."""
    cx = [float(v) for v in re.findall(r'<circle[^>]*cx="([0-9.eE+-]+)"', svg)]
    assert cx, "expected at least one rendered swarm circle"
    return max(cx) - min(cx)


def _scatter_df(n: int = 5000, seed: int = 8) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.normal(0.0, 1.0, n), "y": rng.normal(0.0, 1.0, n)})


def _swarm_df(n: int = 60, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"cat": ["a"] * n, "val": rng.normal(0.0, 1.0, n)})


class TestRasterScreenResolutionRespondsToViewport:
    """F-L03-06's literal repro: `resolution="screen"` (the default) must
    scale with the chart's actual `width`/`height`, not a fixed 256x256."""

    def test_larger_viewport_yields_larger_screen_grid(self):
        df = _scatter_df()
        small = fm.Chart(df, width=300, height=200).mark_raster().encode(x="x", y="y").to_svg()
        large = fm.Chart(df, width=1200, height=900).mark_raster().encode(x="x", y="y").to_svg()

        sw, sh = _raster_png_dims(small)
        lw, lh = _raster_png_dims(large)

        assert (sw, sh) != (256, 256), "must not be pinned to the old hardcoded fallback"
        assert (lw, lh) != (256, 256), "must not be pinned to the old hardcoded fallback"
        assert lw > sw and lh > sh, (
            f"larger viewport must yield a larger screen grid: small={(sw, sh)} large={(lw, lh)}"
        )

    def test_explicit_numeric_resolution_is_unaffected_by_viewport(self):
        df = _scatter_df()
        small = (
            fm.Chart(df, width=300, height=200)
            .mark_raster(resolution=64)
            .encode(x="x", y="y")
            .to_svg()
        )
        large = (
            fm.Chart(df, width=1200, height=900)
            .mark_raster(resolution=64)
            .encode(x="x", y="y")
            .to_svg()
        )
        assert _raster_png_dims(small) == (64, 64)
        assert _raster_png_dims(large) == (64, 64)

    def test_faceted_raster_divides_the_per_cell_estimate_by_the_grid(self):
        """A one-column, two-row facet grid must give each panel roughly HALF
        the unfaceted chart's height estimate, with width unchanged."""
        rng = np.random.default_rng(9)
        n = 400
        df = pl.DataFrame(
            {
                "x": rng.normal(0.0, 1.0, n),
                "y": rng.normal(0.0, 1.0, n),
                "cat": np.tile(["a", "b"], n // 2),
            }
        )
        flat_svg = fm.Chart(df, width=600, height=400).mark_raster().encode(x="x", y="y").to_svg()
        flat_w, flat_h = _raster_png_dims(flat_svg)

        facet_svg = (
            fm.Chart(df, width=600, height=400)
            .mark_raster()
            .encode(x="x", y="y")
            .facet(col="cat", ncols=1)
            .to_svg()
        )
        cell_dims = [
            struct.unpack(">II", base64.b64decode(m)[16:24])
            for m in re.findall(r'href="data:image/png;base64,([^"]+)"', facet_svg)
        ]
        assert len(cell_dims) == 2, "expected one raster image per facet panel"
        for cell_w, cell_h in cell_dims:
            assert cell_w == flat_w, "one column: width estimate must be unchanged"
            assert abs(cell_h - flat_h / 2.0) <= 1, (
                f"two rows: height estimate must be roughly half the unfaceted "
                f"estimate, got flat_h={flat_h} cell_h={cell_h}"
            )

    def test_grid_mode_facet_divides_by_explicit_row_and_col_counts(self):
        """Grid mode (`.facet(row=..., col=...)`) with distinct row/col
        cardinality (2 rows x 3 cols): each cell's estimate must divide width
        by ncols and height by nrows on the RIGHT axis. `render/prepare/mod.rs`'s
        `FacetMode::Grid { nrows, ncols } => (*ncols, *nrows)` tuple order is a
        classic swap site with no other coverage anywhere in this batch --
        `ncols != nrows` here so a swapped tuple fails this test."""
        rng = np.random.default_rng(11)
        n = 600
        df = pl.DataFrame(
            {
                "x": rng.normal(0.0, 1.0, n),
                "y": rng.normal(0.0, 1.0, n),
                "row_cat": np.tile(["r0", "r1"], n // 2),
                "col_cat": np.tile(["c0", "c1", "c2"], n // 3),
            }
        )
        flat_svg = fm.Chart(df, width=600, height=400).mark_raster().encode(x="x", y="y").to_svg()
        flat_w, flat_h = _raster_png_dims(flat_svg)

        facet_svg = (
            fm.Chart(df, width=600, height=400)
            .mark_raster()
            .encode(x="x", y="y")
            .facet(row="row_cat", col="col_cat")
            .to_svg()
        )
        cell_dims = [
            struct.unpack(">II", base64.b64decode(m)[16:24])
            for m in re.findall(r'href="data:image/png;base64,([^"]+)"', facet_svg)
        ]
        assert len(cell_dims) == 6, (
            "expected one raster image per (row, col) cell (2 rows x 3 cols)"
        )
        for cell_w, cell_h in cell_dims:
            assert abs(cell_w - flat_w / 3.0) <= 1, (
                f"3 columns: width estimate must be divided by ncols=3, "
                f"got flat_w={flat_w} cell_w={cell_w}"
            )
            assert abs(cell_h - flat_h / 2.0) <= 1, (
                f"2 rows: height estimate must be divided by nrows=2, "
                f"got flat_h={flat_h} cell_h={cell_h}"
            )

    def test_wrap_mode_facet_derives_nrows_via_ceiling_division(self):
        """3 panels wrapped at `ncols=2` must derive `nrows = ceil(3/2) = 2`,
        not a naive panel-count `nrows = 3`. `n_panels.div_ceil(ncols)`
        (`render/prepare/mod.rs`) is indistinguishable from a count-based
        `nrows = n_panels` whenever `ncols == 1` -- every other facet test in
        this module uses `ncols=1`, so this non-degenerate shape is the only
        coverage of the div_ceil formula in the batch."""
        rng = np.random.default_rng(13)
        n = 600
        df = pl.DataFrame(
            {
                "x": rng.normal(0.0, 1.0, n),
                "y": rng.normal(0.0, 1.0, n),
                "cat": np.tile(["a", "b", "c"], n // 3),
            }
        )
        flat_svg = fm.Chart(df, width=600, height=400).mark_raster().encode(x="x", y="y").to_svg()
        flat_w, flat_h = _raster_png_dims(flat_svg)

        facet_svg = (
            fm.Chart(df, width=600, height=400)
            .mark_raster()
            .encode(x="x", y="y")
            .facet(col="cat", ncols=2)
            .to_svg()
        )
        cell_dims = [
            struct.unpack(">II", base64.b64decode(m)[16:24])
            for m in re.findall(r'href="data:image/png;base64,([^"]+)"', facet_svg)
        ]
        assert len(cell_dims) == 3, "expected one raster image per panel (3 panels)"
        for cell_w, cell_h in cell_dims:
            assert abs(cell_w - flat_w / 2.0) <= 1, (
                f"ncols=2: width estimate must be divided by 2, got flat_w={flat_w} cell_w={cell_w}"
            )
            assert abs(cell_h - flat_h / 2.0) <= 1, (
                f"3 panels at ncols=2 must derive nrows=ceil(3/2)=2, not a "
                f"count-based nrows=3, got flat_h={flat_h} cell_h={cell_h}"
            )


class TestSwarmSizeAndSpacingRespondToPanelSize:
    """F-L03-05's literal repro: `size=`/`spacing=` must actually change the
    beeswarm's rendered placement, not render byte-identically."""

    def test_size_two_vs_twenty_render_differently(self):
        df = _swarm_df()
        svg2 = fm.Chart(df).mark_swarm(size=2).encode(x="cat", y="val").to_svg()
        svg20 = fm.Chart(df).mark_swarm(size=20).encode(x="cat", y="val").to_svg()
        assert svg2 != svg20, "size=2 and size=20 must not render byte-identically"

    def test_swarm_spread_grows_monotonically_with_size(self):
        # Sizes chosen to keep the whole cloud inside the default chart's
        # 608px-wide panel (measured spreads: 14/36/56/80px for 1/2/3/4) --
        # a sequence that overflows the panel (e.g. size=20 -> ~714px) would
        # pin off-panel geometry as the contract, which a future clamp-to-band
        # fix would then read as a regression.
        df = _swarm_df()
        spreads = []
        for size in (1, 2, 3, 4):
            svg = fm.Chart(df).mark_swarm(size=size).encode(x="cat", y="val").to_svg()
            spreads.append(_swarm_cx_spread(svg))
        assert spreads == sorted(spreads), (
            f"swarm spread must grow monotonically with size: {spreads}"
        )
        assert spreads[0] < spreads[-1], "size=1 and size=4 must produce visibly different spreads"

    def test_spacing_also_affects_render(self):
        df = _swarm_df()
        narrow = fm.Chart(df).mark_swarm(size=4, spacing=0.5).encode(x="cat", y="val").to_svg()
        wide = fm.Chart(df).mark_swarm(size=4, spacing=3.0).encode(x="cat", y="val").to_svg()
        assert narrow != wide, "spacing=0.5 and spacing=3.0 must not render byte-identically"
        assert _swarm_cx_spread(wide) > _swarm_cx_spread(narrow)


class TestRasterOnDegenerateViewportRaisesTypedError:
    """Round-2 required fix (rust-quality Fix 3): a viewport whose theme
    padding meets or exceeds it must still raise layout's typed, catchable
    `ValueError` -- not an uncatchable-by-type `PanicException`. Before the
    `layout::estimate_panel_pixel_size` zero-guard, the M2 estimate could
    hand `resolution="screen"`'s transform arm a zero-sized grid, which
    panicked inside the binning loop *before* layout's own refusal ever ran."""

    def test_raster_on_tiny_viewport_raises_typed_layout_error(self):
        """`width=30, height=30` with the theme's default 16px padding: the
        true inner extent is negative on both axes."""
        df = _scatter_df()
        with pytest.raises(ValueError, match="padding left=16 exceeds viewport dimension 30"):
            fm.Chart(df, width=30, height=30).mark_raster().encode(x="x", y="y").to_svg()

    def test_raster_with_oversized_theme_padding_raises_typed_layout_error(self):
        """A normal 600x400 viewport with `Theme(padding=320)`: the bug is not
        merely about tiny charts -- an oversized padding on an ordinary chart
        hits the identical degenerate-estimate path."""
        df = _scatter_df()
        with pytest.raises(ValueError, match="padding left=320 exceeds viewport dimension 600"):
            (
                fm.Chart(df, width=600, height=400)
                .mark_raster()
                .encode(x="x", y="y")
                .theme(fm.Theme(padding=320))
                .to_svg()
            )
