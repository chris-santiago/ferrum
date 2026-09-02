"""Regression tests for RenderConfig.raster_aggregate honesty (F-L07-10).

Before this fix, ``RenderConfig(raster_aggregate=...)`` accepted any string
at construction time and deferred all validation to the moment auto-raster
actually substituted ``mark_raster`` deep inside ``_render.py`` -- so
``"max"``/``"min"``/``"median"`` (never valid raster aggregates) surfaced a
confusing Rust ``ValueError`` only when a chart happened to cross the
row-count threshold, and ``"sum"``/``"mean"`` (valid, but requiring a value
column) had no way to supply that column at all, since ``RenderConfig`` had
no ``raster_field``.

This module pins the fixed contract:

  1. An invalid ``raster_aggregate`` fails at ``RenderConfig(...)``
     construction, naming the accepted set -- never lazily at render time.
  2. ``raster_aggregate="sum"``/``"mean"`` without ``raster_field`` fails at
     construction too.
  3. ``raster_field`` threads end to end: ``sum``/``mean`` actually render
     through the real auto-raster substitution path.
  4. Charts that never touch these fields are unaffected (byte-identity).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum.render_config import RenderConfig


def _valued_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    """Bivariate DataFrame with a value column to aggregate."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n).tolist(),
            "y": rng.normal(0.0, 1.0, n).tolist(),
            "weight": rng.uniform(1.0, 10.0, n).tolist(),
        }
    )


def _clustered_valued_df(
    rows_per_location: tuple[int, ...] = (1, 3, 7, 15, 30), seed: int = 42
) -> pl.DataFrame:
    """DataFrame with repeated (x, y) locations, each holding a different
    *number* of rows, all drawn from the same weight distribution.

    Each of ``len(rows_per_location)`` distinct coordinates is repeated the
    given number of times, guaranteeing every repeat lands in the *same*
    raster pixel bin regardless of rendered resolution. Varying the row
    count per location (rather than holding it fixed) is essential: if
    every bin held the same count, ``sum == count * mean`` would be a
    constant multiple of ``mean`` for every bin, and the ``Raster``
    transform's min-max normalization to the colormap would erase that
    constant factor -- making sum and mean render identically even though
    the aggregate math genuinely differs. With unequal counts, ``sum``
    tracks the row count per bin while ``mean`` stays roughly flat
    (weights are drawn from the same range regardless of location), so the
    two aggregates' *relative* pixel values -- and thus their rendered
    colors -- diverge.
    """
    rng = np.random.default_rng(seed)
    xs: list[float] = []
    ys: list[float] = []
    weights: list[float] = []
    for i, count in enumerate(rows_per_location):
        xs.extend([float(i)] * count)
        ys.extend([float(i)] * count)
        weights.extend(rng.uniform(1.0, 10.0, count).tolist())
    return pl.DataFrame({"x": xs, "y": ys, "weight": weights})


class TestRasterAggregateConstructionValidation:
    """Invalid raster_aggregate/raster_field combos fail at construction."""

    @pytest.mark.parametrize("bad_aggregate", ["max", "min", "median", "avg", "total"])
    def test_unknown_aggregate_refused_at_construction(self, bad_aggregate):
        with pytest.raises(ValueError) as exc_info:
            RenderConfig(raster_aggregate=bad_aggregate)
        msg = str(exc_info.value)
        # Names the accepted set, not just the bad value.
        assert "raster_aggregate" in msg
        assert "count" in msg
        assert "density" in msg
        assert "mean" in msg
        assert "sum" in msg
        assert "any" in msg

    @pytest.mark.parametrize("needs_field", ["sum", "mean"])
    def test_sum_and_mean_without_field_refused_at_construction(self, needs_field):
        with pytest.raises(ValueError) as exc_info:
            RenderConfig(raster_aggregate=needs_field)
        msg = str(exc_info.value)
        assert "raster_field" in msg
        assert needs_field in msg

    @pytest.mark.parametrize("no_field_needed", ["count", "density", "any"])
    def test_no_field_aggregates_construct_without_field(self, no_field_needed):
        # Should not raise.
        cfg = RenderConfig(raster_aggregate=no_field_needed)
        assert cfg.raster_aggregate == no_field_needed
        assert cfg.raster_field is None

    @pytest.mark.parametrize("needs_field", ["sum", "mean"])
    def test_sum_and_mean_construct_with_field(self, needs_field):
        cfg = RenderConfig(raster_aggregate=needs_field, raster_field="weight")
        assert cfg.raster_aggregate == needs_field
        assert cfg.raster_field == "weight"

    def test_default_aggregate_is_count_and_needs_no_field(self):
        cfg = RenderConfig()
        assert cfg.raster_aggregate == "count"
        assert cfg.raster_field is None


class TestRasterFieldEndToEnd:
    """raster_field threads through the auto-raster substitution so sum/mean render."""

    @pytest.mark.parametrize("aggregate", ["sum", "mean"])
    def test_sum_and_mean_render_with_field(self, aggregate):
        df = _valued_df()
        cfg = RenderConfig(
            raster_threshold=0,
            raster_behavior="silent",
            raster_aggregate=aggregate,
            raster_field="weight",
        )
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(render_config=cfg)
            .to_svg(raster=True)
        )
        assert "<image" in svg, (
            f"expected auto-raster substitution to render an <image> element for "
            f"aggregate={aggregate!r} with raster_field set"
        )

    def test_sum_and_mean_produce_different_pixel_payloads(self):
        """sum vs. mean must render differently, not just "with a field".

        On a fixture with at most one row per pixel bin, sum(weight) ==
        mean(weight) for every bin, so no test could distinguish the two
        aggregates' math even if the field forward were broken in a way
        that swapped sum and mean. ``_clustered_valued_df`` repeats each
        (x, y) location a different number of times with weights drawn
        from the same distribution, so sum tracks the per-bin row count
        while mean stays roughly flat -- their normalized pixel values
        diverge, and this test pins that the aggregate *identity*, not
        just the field's presence, reaches the ``Raster`` transform.
        """
        df = _clustered_valued_df()

        def _svg(aggregate):
            cfg = RenderConfig(
                raster_threshold=0,
                raster_behavior="silent",
                raster_aggregate=aggregate,
                raster_field="weight",
            )
            return (
                fm.Chart(df)
                .mark_point()
                .encode(x="x:Q", y="y:Q")
                .properties(render_config=cfg)
                .to_svg(raster=True)
            )

        sum_svg = _svg("sum")
        mean_svg = _svg("mean")
        assert "<image" in sum_svg
        assert "<image" in mean_svg
        assert sum_svg != mean_svg, (
            "sum(weight) and mean(weight) must produce different rendered "
            "pixel payloads on a fixture with multiple rows per bin"
        )

    @pytest.mark.parametrize("aggregate", ["count", "density", "any"])
    def test_no_field_aggregates_still_render(self, aggregate):
        df = _valued_df()
        cfg = RenderConfig(raster_threshold=0, raster_behavior="silent", raster_aggregate=aggregate)
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(render_config=cfg)
            .to_svg(raster=True)
        )
        assert "<image" in svg

    @pytest.mark.parametrize("aggregate", ["count", "density", "any"])
    def test_bogus_raster_field_is_genuinely_ignored_for_no_field_aggregates(self, aggregate):
        """raster_field is documented as "ignored" for count/density/any --
        prove that literally, not just "unread by the aggregate math".

        Before the fix, _apply_auto_raster forwarded raster_field to the
        substitution unconditionally, so even a nonexistent column name was
        resolved by the Rust Raster transform regardless of aggregate --
        raising deep inside render() ("stat_raster: column 'nope_missing'
        not found") instead of never mattering. That is the exact
        deferred-to-render failure shape RenderConfig.__post_init__ exists
        to remove (F-L07-10/spec sec 4.8), newly reachable through this
        task's own raster_field plumbing. A column that does not exist in
        the DataFrame at all must not raise when the aggregate never reads
        it.
        """
        df = _valued_df()
        cfg = RenderConfig(
            raster_threshold=0,
            raster_behavior="silent",
            raster_aggregate=aggregate,
            raster_field="nope_missing_column",
        )
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(render_config=cfg)
            .to_svg(raster=True)
        )
        assert "<image" in svg

    @pytest.mark.parametrize("aggregate", ["count", "density", "any"])
    def test_bogus_raster_field_byte_identical_to_no_field(self, aggregate):
        """A nonexistent raster_field renders byte-identically to no field at
        all for aggregates that don't consume it -- confirming the value is
        dropped before the substitution, not merely tolerated by luck.
        """
        df = _valued_df()

        def _svg(field):
            cfg = RenderConfig(
                raster_threshold=0,
                raster_behavior="silent",
                raster_aggregate=aggregate,
                raster_field=field,
            )
            return (
                fm.Chart(df)
                .mark_point()
                .encode(x="x:Q", y="y:Q")
                .properties(render_config=cfg)
                .to_svg(raster=True)
            )

        assert _svg("nope_missing_column") == _svg(None)

    def test_sum_differs_from_count(self):
        """The aggregated value actually changes the raster output, proving
        raster_field is read by the render path rather than silently dropped.
        """
        df = _valued_df()

        def _svg(**cfg_kwargs):
            cfg = RenderConfig(raster_threshold=0, raster_behavior="silent", **cfg_kwargs)
            return (
                fm.Chart(df)
                .mark_point()
                .encode(x="x:Q", y="y:Q")
                .properties(render_config=cfg)
                .to_svg(raster=True)
            )

        count_svg = _svg(raster_aggregate="count")
        sum_svg = _svg(raster_aggregate="sum", raster_field="weight")
        assert count_svg != sum_svg, (
            "raster_aggregate='sum' with raster_field set must produce different "
            "pixel output than the default count aggregate"
        )


class TestRasterAggregateByteIdentity:
    """Charts that never touch raster_aggregate/raster_field are unaffected."""

    def test_default_render_config_byte_identical_to_no_render_config(self):
        df = pl.DataFrame({"x": list(range(10)), "y": [float(i) for i in range(10)]})
        no_cfg = fm.Chart(df).mark_point().encode(x="x", y="y").to_svg()
        explicit_default = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .properties(render_config=RenderConfig())
            .to_svg()
        )
        assert no_cfg == explicit_default
