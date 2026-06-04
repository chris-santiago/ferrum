"""Tests for D6 sub-tasks 5b + 5c: reactive-parameter reference sites and
params-section serialization.

Covers:
- 5b: ``_scale_to_dict`` domainParam rewrite for Parameter scale domains
- 5b: ``transform_filter(Parameter)`` pass-through predicate + param marker
- 5c: ``Chart._params`` slot, ``add_params``, merge, params-section serialize
- Byte-stability: a param-free chart's spec JSON is unchanged
- Static domain substitution: param's initial value is used as axis domain
"""

from __future__ import annotations

import json
import re

import polars as pl

import ferrum as fm
from ferrum.encoding._scale import _scale_to_dict


# ── 5b: _scale_to_dict domainParam ───────────────────────────────────────────


class TestScaleDomainParam:
    def test_parameter_domain_becomes_domain_param(self):
        d = _scale_to_dict({"domain": fm.param("d", value=[0, 100])})
        assert d == {"type": "linear", "domainParam": "d"}
        assert "domain" not in d

    def test_parameter_domain_keeps_explicit_type(self):
        d = _scale_to_dict({"type": "log", "domain": fm.param("d", value=[1, 100])})
        assert d == {"type": "log", "domainParam": "d"}
        assert "domain" not in d

    def test_non_param_domain_unchanged(self):
        d = _scale_to_dict({"domain": [0, 1]})
        assert d == {"type": "linear", "domain": [0, 1]}
        assert "domainParam" not in d

    def test_does_not_mutate_caller_dict(self):
        p = fm.param("d", value=[0, 100])
        original = {"domain": p}
        _scale_to_dict(original)
        assert original == {"domain": p}


# ── 5b: transform_filter(param) ──────────────────────────────────────────────


class TestTransformFilterParam:
    def test_param_predicate_emits_marker(self):
        brush = fm.selection_interval(name="brush")
        t = fm.transform_filter(brush)
        assert t == {"type": "filter", "predicate": "true", "param": "brush"}

    def test_variable_param_predicate_emits_marker(self):
        p = fm.param("k", value=3)
        t = fm.transform_filter(p)
        assert t == {"type": "filter", "predicate": "true", "param": "k"}

    def test_string_predicate_unchanged(self):
        t = fm.transform_filter("datum.x > 5")
        assert t == {"type": "filter", "predicate": "datum.x > 5"}

    def test_dict_predicate_unchanged(self):
        t = fm.transform_filter({"age": {">=": 18}})
        assert t["type"] == "filter"
        assert "param" not in t
        assert "datum.age" in t["predicate"]


# ── 5c: params-section serialization ─────────────────────────────────────────


def _df():
    return pl.DataFrame({"t": [0, 50, 100], "a": [1, 2, 3], "b": [4, 5, 6]})


def _params_of(spec) -> list:
    d = json.loads(spec.to_json())
    raw = d.get("params")
    if raw is None:
        return []
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


class TestParamsSerialization:
    def test_domain_param_chart_emits_params(self):
        chart = (
            fm.Chart(_df())
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )
        spec = chart.to_spec()
        params = _params_of(spec)
        names = [p["name"] for p in params]
        assert "d" in names
        d_param = next(p for p in params if p["name"] == "d")
        assert d_param["kind"] == "variable"
        assert d_param["value"] == [0, 100]

    def test_domain_param_carried_on_scale(self):
        chart = (
            fm.Chart(_df())
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )
        d = json.loads(chart.to_spec().to_json())
        x_scale = d["encoding"]["x"]["scale"]
        assert x_scale.get("domainParam") == "d"
        assert "domain" not in x_scale

    def test_domain_param_chart_renders(self):
        chart = (
            fm.Chart(_df())
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )
        svg = chart.to_svg()
        assert svg.lstrip().startswith("<")
        assert "<svg" in svg

    def test_add_selection_with_filter_param(self):
        brush = fm.selection_interval(name="brush")
        chart = (
            fm.Chart(_df())
            .mark_point()
            .encode(x=fm.X("a"), y=fm.Y("b"))
            .transform(fm.transform_filter(brush))
            .add_selection(brush)
        )
        spec = chart.to_spec()
        params = _params_of(spec)
        brush_param = next(p for p in params if p["name"] == "brush")
        assert brush_param["kind"] == "interval"
        d = json.loads(spec.to_json())
        filt = next(t for t in d["transforms"] if t.get("type") == "filter")
        assert filt["param"] == "brush"

    def test_filter_param_chart_renders(self):
        brush = fm.selection_interval(name="brush")
        chart = (
            fm.Chart(_df())
            .mark_point()
            .encode(x=fm.X("a"), y=fm.Y("b"))
            .transform(fm.transform_filter(brush))
            .add_selection(brush)
        )
        svg = chart.to_svg()
        assert "<svg" in svg

    def test_add_params(self):
        chart = (
            fm.Chart(_df())
            .mark_point()
            .encode(x=fm.X("a"), y=fm.Y("b"))
            .add_params(fm.param("k", value=3))
        )
        params = _params_of(chart.to_spec())
        names = [p["name"] for p in params]
        assert "k" in names

    def test_dedup_selection_and_domain_ref(self):
        # A selection used both as a registered selection AND referenced in a
        # scale domain must appear exactly once in params.
        brush = fm.selection_interval(name="brush")
        chart = (
            fm.Chart(_df())
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": brush}), y=fm.Y("a"))
            .add_selection(brush)
        )
        params = _params_of(chart.to_spec())
        names = [p["name"] for p in params]
        assert names.count("brush") == 1


class TestStaticDomainSubstitution:
    """Behavioral proof that the static renderer uses a param's initial value
    as the concrete axis domain (D6 static-determinism).

    The assertion uses the cx (x-position) of point-mark circle elements to
    measure how wide the x scale is.  With a forced domain of [0, 100], data
    points at t=10/20/30 should cluster near the left of the plot, producing
    small cx values.  With an auto-inferred domain of [10, 30], those same
    points span the full plot width, so the rightmost point reaches near the
    right edge.  We assert max(cx_A) << max(cx_B), which is robust and does
    not depend on any particular tick-label string.
    """

    def _cx_positions(self, svg: str) -> list[float]:
        # Extract cx attribute values from circle/point elements.
        return [float(m) for m in re.findall(r'cx="([0-9.]+)"', svg)]

    def test_param_domain_constrains_axis_range(self):
        # Data spans t=[10, 30].  A forced domain of [0, 100] should place
        # the points in roughly the leftmost 30% of the plot area.  An
        # auto-inferred domain spans [10, 30], so the rightmost point (t=30)
        # reaches the right edge (large cx).
        df = pl.DataFrame({"t": [10, 20, 30], "a": [1, 2, 3]})

        # Chart A: X domain forced to [0, 100] via param's initial value.
        chart_a = (
            fm.Chart(df)
            .mark_point()
            .encode(
                x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}),
                y=fm.Y("a"),
            )
        )
        svg_a = chart_a.to_svg()

        # Chart B: X domain auto-inferred from data (~[10, 30]).
        chart_b = fm.Chart(df).mark_point().encode(x=fm.X("t"), y=fm.Y("a"))
        svg_b = chart_b.to_svg()

        cx_a = self._cx_positions(svg_a)
        cx_b = self._cx_positions(svg_b)
        assert cx_a, "SVG A must contain circle elements"
        assert cx_b, "SVG B must contain circle elements"

        # The two SVGs must differ (sanity check that the param actually
        # changed something).
        assert svg_a != svg_b

        # With domain [0, 100] the rightmost point (t=30) occupies at most
        # ~30% of the plot width, so its cx is well below the right edge.
        # With domain [10, 30] that same point sits at the right edge.
        # In practice, max(cx_A) ≈ 226 and max(cx_B) ≈ 616 (at 640 px wide).
        # We require a factor-of-2 gap to stay robust against layout changes.
        assert max(cx_a) < max(cx_b) / 2, (
            f"Domain param [0,100] should constrain the x-axis so that the "
            f"rightmost point (t=30) sits well left of the right edge. "
            f"max cx with param domain: {max(cx_a):.1f}; "
            f"max cx with auto domain: {max(cx_b):.1f}"
        )


class TestByteStability:
    def test_param_free_chart_has_no_params_key(self):
        chart = fm.Chart(_df()).mark_point().encode(x="a", y="b")
        js = chart.to_spec().to_json()
        assert "params" not in js

    def test_param_free_chart_byte_stable(self):
        df = _df()
        a = fm.Chart(df).mark_point().encode(x="a", y="b").to_spec().to_json()
        b = fm.Chart(df).mark_point().encode(x="a", y="b").to_spec().to_json()
        assert a == b
