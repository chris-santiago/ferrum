"""Per-mark honored-kwarg vocabularies (#243) and the mark_raster eager
validation + Float64-claim probe (#136).

Before this batch, ``MarkBase``/``validate_user_mark_kwargs`` validated
every mark's ``**kwargs`` against one flat, shared allowlist
(``_VALID_MARK_KWARGS``): any name accepted for ANY mark was silently
accepted for EVERY mark, whether or not that mark actually honored it (e.g.
``mark_bar(arrow=True)`` was accepted and then silently dropped -- ``arrow``
only means anything on ``mark_segment``). This settles #154 (``band_size``
on ``mark_bar``) the same way.

``MARK_HONORED_KWARGS`` (``ferrum.marks.base``) replaces the flat allowlist
with a per-mark vocabulary for the 13 primitive marks, mirrored from
``MarkKwargsSpec``'s own per-field section-header comments in
``crates/ferrum-core/src/spec/mark_style.rs``; ``mark_honors()`` is the one
shared predicate all three consumers (``MarkBase.__init__``,
``validate_user_mark_kwargs``, and -- at the union level --
``_override_apply``'s ``mark_`` prefix registry) now consult.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest

import ferrum as fm
from ferrum.marks.base import (
    MARK_HONORED_KWARGS,
    _PRIMITIVE_MARK_EXTRAS,
    _UNIVERSAL_MARK_KWARGS,
    mark_honors,
)
from ferrum.marks._mark_kwargs import validate_user_mark_kwargs

_MARK_STYLE_RS = (
    Path(__file__).resolve().parent.parent.parent
    / "crates"
    / "ferrum-core"
    / "src"
    / "spec"
    / "mark_style.rs"
)


# ---------------------------------------------------------------------------
# RED: mark_bar(arrow=True) -- accepted-and-silently-dropped before, typed
# refusal after.
# ---------------------------------------------------------------------------


class TestBarArrowRefusal:
    def test_mark_bar_arrow_raises_naming_mark_and_honored_set(self):
        df = pl.DataFrame({"cat": ["a", "b"], "val": [1, 2]})
        with pytest.raises(TypeError) as exc_info:
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
        with pytest.raises(TypeError, match="band_size"):
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

_SIZE_REFUSING_PRIMITIVES = sorted(
    mark for mark in _PRIMITIVE_MARK_EXTRAS if "size" not in MARK_HONORED_KWARGS[mark]
)

#: Literal expected refusal text for every refusing primitive, pointer and
#: no-pointer forms alike -- hand-written, not derived from
#: ``_inert_size_error``, so a dropped/swapped/mis-pointed
#: ``_SIZE_REFUSAL_POINTER`` entry or a drifted message template fails this
#: test instead of passing vacuously against its own production oracle.
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
        """Vacuity control: the refusing population is derived from the
        registry, not hand-listed, and must land on exactly the 13
        primitives minus point/line -- neither empty nor everything."""
        assert _SIZE_REFUSING_PRIMITIVES == sorted(set(_PRIMITIVE_MARK_EXTRAS) - {"point", "line"})
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

    def test_composite_mark_own_size_is_unaffected(self):
        """A composite mark's own ``size=`` (forwarded through its
        ``**mark_kwargs`` catch-all, validated against
        ``_COMPOSITE_FALLBACK_KWARGS`` rather than a primitive's narrower
        vocabulary) is not one of the 11 refusing primitives above and
        keeps its own, composite-specific meaning -- including a
        non-positive value such as ``0``, which is not a point/line
        magnitude and is not positivity-checked."""
        df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        chart = fm.Chart(df).mark_density(size=0).encode(x="v:Q")
        # Discriminating: 0 survives as the literal value on the resolved
        # area layer's mark_kwargs. If density's size= were routed through
        # the primitive's positivity check instead of its own composite
        # vocabulary, resolving the pending mark would raise "must be a
        # positive number" for exactly this value, not return it unchanged.
        assert chart._resolve_pending()._mark_kwargs == {"size": 0}
        chart.to_svg()  # the full render pipeline accepts the inert field too


# ---------------------------------------------------------------------------
# mark_honors()/MARK_HONORED_KWARGS direct-unit coverage.
# ---------------------------------------------------------------------------


class TestMarkHonors:
    def test_universal_kwargs_honored_by_every_primitive(self):
        for mark in MARK_HONORED_KWARGS:
            for kwarg in _UNIVERSAL_MARK_KWARGS:
                assert mark_honors(mark, kwarg), f"{mark} should honor universal {kwarg!r}"

    def test_point_honors_shape_and_filled_only_extra(self):
        assert mark_honors("point", "shape")
        assert mark_honors("point", "filled")
        assert not mark_honors("point", "band_size")
        assert not mark_honors("point", "arrow")

    def test_unrecognized_mark_falls_back_to_universal(self):
        # Composite/diagnostic mark names have no explicit entry; they fall
        # back to the universal set.
        assert "boxplot" not in MARK_HONORED_KWARGS
        assert mark_honors("boxplot", "opacity")
        assert not mark_honors("boxplot", "shape")

    def test_forwarded_to_transform_kwargs_are_not_honored_by_any_mark(self):
        # method/ci/bandwidth/... were never real MarkKwargsSpec fields --
        # they are named parameters of specific mark_*() methods
        # (mark_smooth, mark_density, mark_histogram), always intercepted by
        # signature-based partitioning before reaching MarkBase. They must
        # not appear in any mark's honored set nor the universal fallback.
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
            assert stale not in _UNIVERSAL_MARK_KWARGS
            for mark in MARK_HONORED_KWARGS:
                assert not mark_honors(mark, stale), f"{mark} must not honor {stale!r}"


class TestValidateUserMarkKwargs:
    def test_honored_kwarg_passes(self):
        assert validate_user_mark_kwargs("boxplot", {"opacity": 0.5}) == {"opacity": 0.5}

    def test_not_honored_kwarg_raises_naming_mark_and_honored_set(self):
        with pytest.raises(TypeError) as exc_info:
            validate_user_mark_kwargs("boxplot", {"shape": "diamond"})
        message = str(exc_info.value)
        assert "mark_boxplot" in message
        assert "shape" in message
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

    def test_documented_gap_override_is_not_yet_per_mark(self):
        """Tracked gap (see _override_apply.py's module docstring): the
        override registry validates against the UNION of every mark's
        honored kwargs, not the chart's own mark, since resolve() has no
        per-chart context. A bar chart can still .override(mark_shape=...)
        even though mark_bar() itself would refuse shape=. This test pins
        the gap so a future tightening is a deliberate, visible change."""
        df = pl.DataFrame({"cat": ["a", "b"], "val": [1, 2]})
        chart = fm.Chart(df).mark_bar().encode(x="cat", y="val").override(mark_shape="diamond")
        # Resolves (does not raise FerrumOverrideError); Rust silently
        # ignores the field for a bar mark, matching this gap's tracked
        # description.
        chart.to_svg()


# ---------------------------------------------------------------------------
# Drift pin: MARK_HONORED_KWARGS's per-primitive extras must match
# MarkKwargsSpec's own section-header comments in mark_style.rs.
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"([a-z_]+(?:\s*/\s*[a-z_]+)*)\s*\(([^()]*)\)")
_PUB_FIELD_RE = re.compile(r"^\s*pub\s+([a-z_0-9]+):\s*Option<")


def _parse_rust_field_mark_map() -> dict[str, frozenset[str]]:
    """Parse ``mark_style.rs``'s struct body for section-header comments of
    the form ``// ── <label>: <field>[ / <field>] (<mark>[, <mark>...]) ──``
    and pair each named field with the primitive mark names mentioned in its
    header (filtering out non-mark descriptive words like "constant").

    Only the struct body (before ``#[cfg(test)]``) is scanned -- the test
    module below it repeats some header text as section dividers with no
    adjoining ``pub`` field, which would otherwise pollute the map with
    stale duplicate entries (harmlessly, since they're identical, but the
    struct body is the single source of truth).
    """
    text = _MARK_STYLE_RS.read_text()
    struct_text = text.split("#[cfg(test)]")[0]
    primitive_marks = frozenset(_PRIMITIVE_MARK_EXTRAS)

    pending_fields: set[str] = set()
    pending_marks: frozenset[str] = frozenset()
    result: dict[str, set[str]] = {}
    for line in struct_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            header_match = _HEADER_RE.search(stripped)
            if header_match:
                field_names = {n.strip() for n in header_match.group(1).split("/")}
                marks = {m.strip() for m in re.split(r"[,/]", header_match.group(2))}
                marks &= primitive_marks
                if marks:
                    pending_fields = field_names
                    pending_marks = frozenset(marks)
            continue
        pub_match = _PUB_FIELD_RE.match(line)
        if pub_match and pub_match.group(1) in pending_fields:
            result.setdefault(pub_match.group(1), set()).update(pending_marks)
    return {k: frozenset(v) for k, v in result.items()}


class TestDriftPinAgainstRustSectionHeaders:
    def test_parser_finds_known_pairs(self):
        """Sanity: guard against the regex silently matching nothing, which
        would make the drift-pin assertions below vacuously true."""
        rust_map = _parse_rust_field_mark_map()
        assert rust_map, "drift-pin parser found no section headers in mark_style.rs"
        assert rust_map["interpolate"] >= {"line", "area"}
        assert rust_map["band_size"] == {"tick", "rect"}
        assert rust_map["arrow"] == {"segment"}
        assert rust_map["arrow_size"] == {"segment"}
        assert rust_map["borders"] == {"area"}
        assert rust_map["leader_line"] == {"label"}
        assert rust_map["zero"] == {"bar"}

    def test_every_rust_documented_pairing_is_granted_in_python(self):
        rust_map = _parse_rust_field_mark_map()
        for field, marks in rust_map.items():
            for mark in marks:
                assert field in _PRIMITIVE_MARK_EXTRAS[mark], (
                    f"mark_style.rs documents {field!r} as owned by mark_{mark}, "
                    f"but MARK_HONORED_KWARGS[{mark!r}] does not include "
                    f"{field!r}. Update ferrum.marks.base._PRIMITIVE_MARK_EXTRAS."
                )

    def test_python_does_not_over_grant_rust_documented_fields(self):
        rust_map = _parse_rust_field_mark_map()
        for mark, extras in _PRIMITIVE_MARK_EXTRAS.items():
            for field in extras:
                if field not in rust_map:
                    # Python-only fields (e.g. "orient") have no Rust
                    # MarkKwargsSpec entry at all; nothing to cross-check.
                    continue
                assert mark in rust_map[field], (
                    f"MARK_HONORED_KWARGS[{mark!r}] grants {field!r}, but "
                    f"mark_style.rs's section header for {field!r} only names "
                    f"{sorted(rust_map[field])} -- drift between Python and Rust."
                )


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
