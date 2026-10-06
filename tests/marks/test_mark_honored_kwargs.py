"""Per-mark honored-kwarg vocabularies (#243) and the mark_raster eager
validation + Float64-claim probe (#136).

Before #243, ``MarkBase``/``validate_user_mark_kwargs`` validated every
mark's ``**kwargs`` against one flat, shared allowlist: any name accepted
for ANY mark was silently accepted for EVERY mark, whether or not that mark
actually honored it (e.g. ``mark_bar(arrow=True)`` was accepted and then
silently dropped -- ``arrow`` only means anything on ``mark_segment``). This
settles #154 (``band_size`` on ``mark_bar``) the same way.

``ferrum.marks.base.honored_kwargs`` is each mark's vocabulary, read off
Rust's per-mark style table (``ferrum._core.mark_style_fields``) plus the
kwargs a mark method consumes itself (``METHOD_ONLY_KWARGS``);
``mark_honors()`` is the one shared predicate ``MarkBase.__init__`` and
``validate_user_mark_kwargs`` consult.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import mark_kwargs_fields, mark_style_fields
from ferrum._layer import drawn_style_fields
from ferrum.marks.base import (
    _INTERNAL_STYLE_DOORS,
    METHOD_ONLY_KWARGS,
    honored_kwargs,
    layer_style_fields,
    mark_honors,
)
from ferrum.marks._mark_kwargs import validate_user_mark_kwargs

# ---------------------------------------------------------------------------
# RED: mark_bar(arrow=True) -- accepted-and-silently-dropped before, typed
# refusal after.
# ---------------------------------------------------------------------------


class TestBarArrowRefusal:
    def test_mark_bar_arrow_raises_naming_mark_and_honored_set(self):
        df = pl.DataFrame({"cat": ["a", "b"], "val": [1, 2]})
        with pytest.raises(ValueError) as exc_info:
            fm.Chart(df).mark_bar(arrow=True).encode(x="cat", y="val")
        message = str(exc_info.value)
        assert "mark_bar" in message
        assert "arrow" in message
        # The honored set is named in the message (decision: "naming the
        # mark and its honored set").
        assert "zero" in message or "orient" in message

    def test_mark_segment_arrow_still_works(self):
        """Positive control: arrow genuinely is honored by mark_segment."""
        df = pl.DataFrame({"x1": [0.0], "y1": [0.0], "x2": [1.0], "y2": [1.0]})
        chart = (
            fm.Chart(df)
            .mark_segment(arrow=True, arrow_size=8)
            .encode(x="x1", y="y1", x2="x2", y2="y2")
        )
        assert chart._mark_kwargs.get("arrow") is True


# ---------------------------------------------------------------------------
# #154 settlement: band_size on mark_bar refuses; mark_tick/mark_rect still honor it.
# ---------------------------------------------------------------------------


class TestBandSizeSettlesFinding154:
    def test_mark_bar_band_size_raises(self):
        df = pl.DataFrame({"cat": ["a", "b"], "val": [1, 2]})
        with pytest.raises(ValueError, match="band_size"):
            fm.Chart(df).mark_bar(band_size=0.9).encode(x="cat", y="val")

    def test_mark_tick_band_size_still_works(self):
        df = pl.DataFrame({"cat": ["a", "b"], "val": [1, 2]})
        chart = fm.Chart(df).mark_tick(band_size=0.9).encode(x="cat", y="val")
        assert chart._mark_kwargs.get("band_size") == 0.9

    def test_mark_rect_band_size_still_works(self):
        df = pl.DataFrame(
            {"x1": [0], "x2": [1], "y1": [0], "y2": [1]},
        )
        chart = fm.Chart(df).mark_rect(band_size=0.5).encode(x="x1", x2="x2", y="y1", y2="y2")
        assert chart._mark_kwargs.get("band_size") == 0.5


# ---------------------------------------------------------------------------
# #116: a constant size= refuses at construction on every primitive mark
# whose renderer never reads it; only point/line (and point's circle/square
# aliases) keep size=.
# ---------------------------------------------------------------------------

#: The marks a primitive ``mark_<name>()`` method sets on a flat chart, read
#: off ``Chart``'s own methods (``mark_ribbon`` is a composite method).
_PRIMITIVE_METHOD_MARKS = sorted(
    mark
    for mark in (name[len("mark_") :] for name in dir(fm.Chart) if name.startswith("mark_"))
    if drawn_style_fields(mark) is not None
    and getattr(fm.Chart(None), f"mark_{mark}")()._pending_stat_mark is None
)

_SIZE_REFUSING_PRIMITIVES = sorted(
    mark for mark in _PRIMITIVE_METHOD_MARKS if "size" not in honored_kwargs(mark)
)

#: Literal expected refusal text for every refusing primitive, pointer and
#: no-pointer forms alike -- hand-written, not derived from the Rust size
#: rule (``ferrum._core.check_mark_size``), so a dropped/swapped/mis-pointed
#: recourse or a drifted message template fails this test instead of
#: passing vacuously against its own production oracle.
_SIZE_REFUSAL_MESSAGES: dict[str, str] = {
    "arc": "mark_arc(size=...) has no effect on mark_arc; received 20.",
    "area": "mark_area(size=...) has no effect on mark_area; received 20.",
    "bar": "mark_bar(size=...) has no effect on mark_bar; received 20.",
    "geoshape": "mark_geoshape(size=...) has no effect on mark_geoshape; received 20.",
    "image": "mark_image(size=...) has no effect on mark_image; received 20.",
    "label": "mark_label(size=...) has no effect on mark_label; received 20. Set font_size= instead.",
    "rect": "mark_rect(size=...) has no effect on mark_rect; received 20. Set band_size= instead.",
    "rule": "mark_rule(size=...) has no effect on mark_rule; received 20. Set stroke_width= instead.",
    "segment": (
        "mark_segment(size=...) has no effect on mark_segment; received 20. "
        "Set stroke_width= instead."
    ),
    "text": "mark_text(size=...) has no effect on mark_text; received 20. Set font_size= instead.",
    "tick": "mark_tick(size=...) has no effect on mark_tick; received 20. Set band_size= instead.",
}


class TestConstantSizeRefusedOnMarksThatDoNotDrawIt:
    def test_population_is_every_primitive_but_point_and_line(self):
        """Vacuity control: the refusing population is derived from Rust's
        table, not hand-listed, and must land on exactly the 13 primitive
        methods minus point/line -- neither empty nor everything."""
        assert _SIZE_REFUSING_PRIMITIVES == sorted(set(_PRIMITIVE_METHOD_MARKS) - {"point", "line"})
        assert len(_SIZE_REFUSING_PRIMITIVES) == 11
        # The literal-pin table below must track this registry-derived
        # population exactly, so the two cannot drift apart.
        assert set(_SIZE_REFUSAL_MESSAGES) == set(_SIZE_REFUSING_PRIMITIVES)

    @pytest.mark.parametrize("mark_name", _SIZE_REFUSING_PRIMITIVES)
    def test_constant_size_raises_naming_mark_and_pointer(self, mark_name):
        with pytest.raises(ValueError) as exc_info:
            getattr(fm.Chart(None), f"mark_{mark_name}")(size=20)
        assert str(exc_info.value) == _SIZE_REFUSAL_MESSAGES[mark_name]

    @pytest.mark.parametrize("ctor", ["mark_point", "mark_circle", "mark_square", "mark_line"])
    def test_size_still_accepted_by_point_line_and_point_aliases(self, ctor):
        """Positive control: point, line, and point's circle/square aliases
        (which forward to mark_point) still accept a constant size= -- only
        the 11 sibling primitives above refuse it."""
        chart = getattr(fm.Chart(None), ctor)(size=20)
        assert chart._mark_kwargs.get("size") == 20

    def test_composite_mark_size_refuses_where_no_emitted_mark_draws_it(self):
        """A composite mark's catch-all ``size=`` refuses, in the primitives'
        voice and naming the composite, once its desugar shows none of the
        marks it emits draws ``size`` (density emits one area)."""
        df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        chart = fm.Chart(df).mark_density(size=0).encode(x="v:Q")
        with pytest.raises(ValueError) as exc_info:
            chart.to_svg()
        assert str(exc_info.value) == (
            "mark_density(size=...) has no effect on mark_density; received 0."
        )

    def test_composite_mark_size_meets_the_rule_of_the_mark_that_draws_it(self):
        """A composite that emits a point holds its catch-all ``size=`` to the
        point's own rule, voiced as the composite, and spreads it onto the
        point alone: the rule it also emits never draws a size."""
        df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with pytest.raises(ValueError) as exc_info:
            fm.Chart(df).mark_qq(size=0).encode(x="v:Q").to_json()
        assert str(exc_info.value) == "mark_qq(size=...) must be a positive number; received 0"
        resolved = fm.Chart(df).mark_qq(size=30).encode(x="v:Q")._resolve_pending()
        assert [(ly.mark, (ly.mark_kwargs or {}).get("size")) for ly in resolved._layers] == [
            ("point", 30),
            ("rule", None),
        ]


# ---------------------------------------------------------------------------
# mark_honors()/honored_kwargs direct-unit coverage.
# ---------------------------------------------------------------------------


class TestMarkHonors:
    @pytest.mark.parametrize("mark", _PRIMITIVE_METHOD_MARKS)
    def test_a_mark_method_honors_its_table_row_and_its_own_kwargs(self, mark):
        drawn = set(mark_style_fields(mark))
        assert honored_kwargs(mark) == (drawn - set(_INTERNAL_STYLE_DOORS)) | set(
            METHOD_ONLY_KWARGS.get(mark, ())
        )
        assert layer_style_fields(mark) == drawn - set(_INTERNAL_STYLE_DOORS)

    def test_point_honors_shape_and_filled_only_extra(self):
        assert mark_honors("point", "shape")
        assert mark_honors("point", "filled")
        assert not mark_honors("point", "band_size")
        assert not mark_honors("point", "arrow")
        assert not mark_honors("point", "font_size")

    def test_a_composite_honors_every_public_field(self):
        """A composite spreads its style over the marks it emits, so it takes
        every field some mark draws and is held to its emitted marks once
        its desugar has run; the internal fields it never takes."""
        assert honored_kwargs("boxplot") == set(mark_kwargs_fields()) - set(_INTERNAL_STYLE_DOORS)
        assert mark_honors("boxplot", "shape")
        assert not mark_honors("boxplot", "detail")

    def test_forwarded_to_transform_kwargs_are_not_honored_by_any_mark(self):
        # method/ci/bandwidth/... were never real MarkKwargsSpec fields --
        # they are named parameters of specific mark_*() methods
        # (mark_smooth, mark_density, mark_histogram), always intercepted by
        # signature-based partitioning before reaching MarkBase. They must
        # not appear in any mark's honored set, composite marks included.
        for stale in (
            "method",
            "ci",
            "bandwidth",
            "degree",
            "n",
            "kernel",
            "extent",
            "cumulative",
            "bin_count",
            "bin_width",
            "density",
            "right",
            "multiple",
        ):
            for mark in [*_PRIMITIVE_METHOD_MARKS, "boxplot"]:
                assert not mark_honors(mark, stale), f"{mark} must not honor {stale!r}"


class TestValidateUserMarkKwargs:
    def test_honored_kwarg_passes(self):
        assert validate_user_mark_kwargs("boxplot", {"opacity": 0.5}) == {"opacity": 0.5}

    def test_not_honored_kwarg_raises_naming_mark_and_honored_set(self):
        with pytest.raises(ValueError) as exc_info:
            validate_user_mark_kwargs("boxplot", {"detail": "g"})
        message = str(exc_info.value)
        assert message.startswith(
            "mark_boxplot: keyword argument(s) ['detail'] not honored by mark_boxplot. Honored: "
        )
        assert "opacity" in message  # a member of the honored set is named

    def test_empty_dict_short_circuits(self):
        assert validate_user_mark_kwargs("anything", {}) == {}


# ---------------------------------------------------------------------------
# Chart.override(mark_<leaf>=...): the shared-vocabulary consumer.
# ---------------------------------------------------------------------------


class TestOverrideMarkPrefix:
    def test_stale_transform_forwarded_leaf_now_refuses(self):
        """.override(mark_bandwidth=...) used to resolve then get silently
        dropped by the Rust wire (MarkKwargsSpec has no such field); it now
        refuses at the Python override boundary instead."""
        from ferrum.exceptions import FerrumOverrideError

        df = pl.DataFrame({"cat": ["a", "b"], "val": [1, 2]})
        chart = fm.Chart(df).mark_bar().encode(x="cat", y="val").override(mark_bandwidth=0.5)
        with pytest.raises(FerrumOverrideError):
            chart.to_svg()

    def test_override_is_held_to_the_charts_own_mark(self):
        """The override registry resolves ``mark_shape`` against every
        mark's union, but where the style is applied the chart's own mark
        is known, and a leaf it does not honor refuses as ``mark_bar(shape=)``
        would, instead of reaching a wire field the bar never reads."""
        from ferrum.exceptions import FerrumOverrideError

        df = pl.DataFrame({"cat": ["a", "b"], "val": [1, 2]})
        chart = fm.Chart(df).mark_bar().encode(x="cat", y="val").override(mark_shape="diamond")
        with pytest.raises(FerrumOverrideError) as exc_info:
            chart.to_svg()
        assert str(exc_info.value).startswith(
            "Chart.override: keyword argument 'shape' is not honored by mark_bar. Honored: "
        )
        assert isinstance(exc_info.value, ValueError)

    def test_override_of_a_leaf_the_mark_honors_still_applies(self):
        """Control: the same leaf on a point chart draws a diamond."""
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        assert base.override(mark_shape="diamond").to_svg() != base.to_svg()


# ---------------------------------------------------------------------------
# #136: mark_raster() eager aggregate/field validation.
# ---------------------------------------------------------------------------


class TestMarkRasterEagerValidation:
    def test_bad_aggregate_raises_at_call_time(self):
        """RED (pre-fix): an invalid aggregate was accepted at mark_raster()
        call time and only failed once Rust's Raster transform ran at
        render time. Now it raises immediately, before .encode() is even
        called."""
        with pytest.raises(ValueError, match="aggregate"):
            fm.Chart(None).mark_raster(aggregate="median")

    def test_missing_field_for_mean_raises_at_call_time(self):
        """RED (pre-fix): aggregate='mean' with no field= was accepted at
        call time; desugar_raster only raised once .encode(x=,y=) resolved
        the pending mark. Now it raises immediately."""
        with pytest.raises(ValueError, match="field"):
            fm.Chart(None).mark_raster(aggregate="mean")

    def test_missing_field_for_sum_raises_at_call_time(self):
        with pytest.raises(ValueError, match="field"):
            fm.Chart(None).mark_raster(aggregate="sum")

    def test_valid_combination_does_not_raise_at_call_time(self):
        # No encode() call yet -- this must not raise, since the combination
        # itself is valid; only x/y-encoding resolution is deferred.
        fm.Chart(None).mark_raster(aggregate="mean", field="v")

    def test_count_aggregate_needs_no_field(self):
        fm.Chart(None).mark_raster(aggregate="count")

    def test_accepts_every_declared_method_with_or_without_field(self):
        """Behavioral pin (#137/#176): mark_raster's eager check reads live
        from ``transform_registry("raster")`` -- not a hand-copied literal
        that could silently outlive a new Rust raster aggregate (the #137
        latent bug: a stale Python literal stayed green while a sixth Rust
        aggregate would have been refused by the stale set). Every declared
        method is exercised with and without field=; a bogus spelling is
        refused using the registry's own accepted set, not a hardcoded
        string, so this test cannot pass by comparing one literal to
        another."""
        from ferrum._core import transform_registry

        entry = transform_registry("raster")
        for method in entry["methods"]:
            if method in entry["methods_needing_field"]:
                with pytest.raises(ValueError, match="requires field"):
                    fm.Chart(None).mark_raster(aggregate=method)
                fm.Chart(None).mark_raster(aggregate=method, field="v")
            else:
                fm.Chart(None).mark_raster(aggregate=method)
                fm.Chart(None).mark_raster(aggregate=method, field="v")

        bogus = "geomean"
        assert bogus not in entry["methods"]
        with pytest.raises(ValueError) as exc:
            fm.Chart(None).mark_raster(aggregate=bogus)
        assert str(exc.value) == (
            f"mark_raster: aggregate must be one of {sorted(entry['methods'])}; got {bogus!r}"
        )


class TestMarkRasterFloat64Probe:
    """#136's probe: is the render_config.py docstring's Float64-only claim
    still true after batch D's numeric coercion? It is not -- integer
    columns are coerced automatically. Pinned here as a real render (not a
    hand-built fixture) so the probe's finding stays true under the actual
    Raster transform, not a mirror of it."""

    def test_integer_field_column_renders_without_error(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 1.5],
                "y": [1.0, 2.0, 3.0, 1.2],
                "v": [1, 2, 3, 4],  # Int64, not Float64
            }
        )
        svg = fm.Chart(df).mark_raster(aggregate="mean", field="v").encode(x="x", y="y").to_svg()
        assert "<svg" in svg

    def test_non_numeric_field_column_now_raises_at_call_time(self):
        """Close-remediation (#136 residual): when the Chart already carries
        data, a non-numeric field= is answerable at mark_raster() call time,
        not deferred to the Rust `stat_raster` transform at render. This
        used to raise only at `.to_svg()`; it now raises before `.encode()`
        is even called."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 1.5],
                "y": [1.0, 2.0, 3.0, 1.2],
                "v": ["a", "b", "c", "d"],
            }
        )
        with pytest.raises(ValueError, match="numeric"):
            fm.Chart(df).mark_raster(aggregate="mean", field="v")

    def test_absent_field_column_now_raises_at_call_time(self):
        """Sibling to the non-numeric case: a field= naming a column the
        data doesn't have is answerable eagerly too (#136 residual)."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 1.5],
                "y": [1.0, 2.0, 3.0, 1.2],
            }
        )
        with pytest.raises(ValueError, match="not a column"):
            fm.Chart(df).mark_raster(aggregate="mean", field="nosuch")

    def test_deferred_data_chart_skips_field_check_gracefully(self):
        """A Chart with no data yet (Chart(None)) has nothing to check
        field= against -- the eager check must not raise for a chart whose
        data isn't attached yet, matching
        `test_valid_combination_does_not_raise_at_call_time`."""
        fm.Chart(None).mark_raster(aggregate="mean", field="v")

    def test_inert_field_on_count_aggregate_is_not_checked(self):
        """field= is genuinely inert for aggregate="count"/"density"/"any"
        (dropped before the substitution, per the docstring) -- the eager
        check must not fire for it even when the named column doesn't
        exist or isn't numeric."""
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "v": ["a", "b"]})
        fm.Chart(df).mark_raster(aggregate="count", field="v")
        fm.Chart(df).mark_raster(aggregate="count", field="nosuch")


class TestMarkRasterLazyFrameSchemaProbe:
    """Close-remediation cycle 2 (design-review S3): the #136 eager
    field= check must answer from a LazyFrame's SCHEMA, never by
    materializing it -- a chart-declaration method (mark_raster()) must
    stay as free on lazy data as mark_point() is, and must never let an
    untyped, non-ferrum exception (e.g. a bare FileNotFoundError from a
    missing scan path) escape from a public builder.
    """

    def _big_lazyframe_with_slow_udf(self, n: int = 50_000) -> "pl.LazyFrame":
        # A python-level map_elements is slow enough (a fraction of a
        # millisecond per row) that if collect_schema() actually executed
        # the query plan, this test would take seconds instead of
        # milliseconds -- proving "no materialization" by elapsed time,
        # not by mocking internals.
        import time

        def _slow(x):
            time.sleep(0.0001)
            return x

        df = pl.DataFrame(
            {
                "x": list(range(n)),
                "y": list(range(n)),
                "v": ["a"] * n,
                "n": [1] * n,
            }
        )
        return df.lazy().with_columns(pl.col("n").map_elements(_slow, return_dtype=pl.Int64))

    def test_lazyframe_eager_check_does_not_materialize(self):
        import time

        lf = self._big_lazyframe_with_slow_udf()
        t0 = time.monotonic()
        fm.Chart(lf).mark_raster(aggregate="mean", field="n")
        elapsed = time.monotonic() - t0
        # The slow UDF alone would take >1s over 50k rows if executed;
        # collect_schema() must resolve the schema without touching it.
        assert elapsed < 0.5, f"mark_raster() took {elapsed:.3f}s -- looks materialized"

    def test_lazyframe_non_numeric_field_still_refuses_eagerly(self):
        """The eager check must still WORK for a lazy source whose schema
        resolves cleanly -- collect_schema() is 0-cost, not a reason to
        skip validation altogether."""
        lf = self._big_lazyframe_with_slow_udf()
        with pytest.raises(ValueError, match="numeric"):
            fm.Chart(lf).mark_raster(aggregate="mean", field="v")

    def test_lazyframe_missing_field_still_refuses_eagerly(self):
        lf = self._big_lazyframe_with_slow_udf()
        with pytest.raises(ValueError, match="not a column"):
            fm.Chart(lf).mark_raster(aggregate="mean", field="nosuch")

    def test_lazyframe_over_missing_file_does_not_raise_at_declaration(self):
        """A LazyFrame whose schema resolution itself fails (e.g. a scan
        over a nonexistent path) must skip the eager check gracefully --
        mark_raster() must construct fine, exactly like mark_point() does,
        deferring to render time as before this eager check existed."""
        missing_lf = pl.scan_parquet("/tmp/does-not-exist-xyz-close-remediation.parquet")
        # Must not raise FileNotFoundError (or any exception) here.
        chart = fm.Chart(missing_lf).mark_raster(aggregate="mean", field="v")
        assert chart is not None

    def test_eager_dataframe_field_checks_unaffected(self):
        """Sibling guard: the schema-probe refactor must not change
        behavior for the already-materialized (non-lazy) case."""
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "v": ["a", "b"], "n": [1, 2]})
        with pytest.raises(ValueError, match="numeric"):
            fm.Chart(df).mark_raster(aggregate="mean", field="v")
        with pytest.raises(ValueError, match="not a column"):
            fm.Chart(df).mark_raster(aggregate="mean", field="nosuch")
        fm.Chart(df).mark_raster(aggregate="mean", field="n")
