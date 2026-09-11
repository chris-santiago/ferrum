"""F-L13-01: Parameter.ref() is consumed at its three documented sites.

``Parameter.ref()`` (``{"param": name}``) shipped 2026-06-01 but was never
wired to any of the three sites its docstring promises -- passing it to a
scale domain raised a Rust deserialization error, to a conditional value
raised ``TypeError``, and to ``transform_filter`` silently mis-serialized
into the garbage predicate ``datum.param == "name"`` instead of a real
param-filter. This module pins the fix: one decoder (``_decode_param_ref``)
consumed at all three sites (``ferrum.encoding._scale._scale_to_dict``,
``ferrum.transforms.transform_filter``, and
``ferrum.selection._resolve_encoding_value``).

RED-proven against ``git stash push -- src/ferrum/encoding/_scale.py
src/ferrum/transforms.py src/ferrum/parameter.py src/ferrum/selection.py
src/ferrum/chart.py`` -- every test in this module fails pre-fix.
"""

from __future__ import annotations

import pytest

import ferrum as fm
import polars as pl
from ferrum.parameter import _decode_param_ref
from ferrum.selection import ConditionalSpec, _resolve_param_ref

DF = pl.DataFrame({"x": [1, 2, 3], "y": [3, 1, 2], "cat": ["a", "b", "a"]})


# ── _decode_param_ref: the shared shape test ─────────────────────────────────


class TestDecodeParamRef:
    def test_decodes_ref_dict(self):
        q = fm.param("qmax", value=10)
        assert _decode_param_ref(q.ref()) == "qmax"

    def test_decodes_exact_shape_literally(self):
        assert _decode_param_ref({"param": "qmax"}) == "qmax"

    def test_rejects_extra_keys(self):
        # Not exactly {"param": name} -- falls through, not a ref.
        assert _decode_param_ref({"param": "qmax", "other": 1}) is None

    def test_rejects_non_string_name(self):
        assert _decode_param_ref({"param": 5}) is None

    def test_rejects_non_dict(self):
        assert _decode_param_ref("qmax") is None
        assert _decode_param_ref(None) is None
        assert _decode_param_ref(5) is None

    def test_rejects_unrelated_dict(self):
        assert _decode_param_ref({"field": "x"}) is None


# ── Site 1: scale domain ──────────────────────────────────────────────────────


class TestScaleDomainRef:
    def test_ref_domain_does_not_raise(self):
        q = fm.param("qmax", value=10)
        # Pre-fix: Rust ValueError ("invalid type: map, expected a sequence").
        chart = fm.Chart(DF).mark_point().encode(x=fm.X("x", scale={"domain": q.ref()}), y="y")
        chart.to_dict()

    def test_ref_domain_equivalent_to_raw_parameter(self):
        # The raw-Parameter path auto-registers `q` (its scale-domain scan
        # sees the actual object); `.ref()` carries only the name, so it
        # needs the same explicit `add_params(q)` a user would give any
        # by-name reference. With that, the two forms emit byte-identical
        # specs.
        q = fm.param("qmax", value=10)
        ref_chart = (
            fm.Chart(DF)
            .mark_point()
            .encode(x=fm.X("x", scale={"domain": q.ref()}), y="y")
            .add_params(q)
        )
        obj_chart = fm.Chart(DF).mark_point().encode(x=fm.X("x", scale={"domain": q}), y="y")
        assert ref_chart.to_json() == obj_chart.to_json()

    def test_ref_domain_emits_domain_param_by_name(self):
        q = fm.param("qmax", value=10)
        chart = fm.Chart(DF).mark_point().encode(x=fm.X("x", scale={"domain": q.ref()}), y="y")
        spec = chart.to_dict()
        assert spec["encoding"]["x"]["scale"]["domainParam"] == "qmax"
        assert "domain" not in spec["encoding"]["x"]["scale"]

    def test_ref_domain_with_registered_param_reaches_params_array(self):
        # add_params(q) is how a ref-by-name domain gets its static value
        # onto the wire (the ref dict alone carries only the name).
        q = fm.param("qmax", value=10)
        chart = (
            fm.Chart(DF)
            .mark_point()
            .encode(x=fm.X("x", scale={"domain": q.ref()}), y="y")
            .add_params(q)
        )
        spec = chart.to_dict()
        assert any(p["name"] == "qmax" and p["value"] == 10 for p in spec["params"])


# ── Site 2: transform_filter ──────────────────────────────────────────────────


class TestTransformFilterRef:
    def test_ref_does_not_mangle_into_datum_param_predicate(self):
        q = fm.param("qmax", value=10)
        result = fm.transform_filter(q.ref())
        # The bug: {"param": "qmax"} misread as a field-equality constraint
        # dict, compiling to `datum.param == "qmax"`.
        assert "datum.param" not in str(result.get("predicate", ""))

    def test_ref_equivalent_to_object_form(self):
        q = fm.param("qmax", value=10)
        assert fm.transform_filter(q.ref()) == fm.transform_filter(q)

    def test_ref_produces_param_filter_shape(self):
        q = fm.param("qmax", value=10)
        result = fm.transform_filter(q.ref())
        assert result == {"type": "filter", "predicate": "true", "param": "qmax"}

    def test_object_form_unchanged_byte_identical(self):
        # Byte-stability: the raw-Parameter path must not move.
        q = fm.param("qmax", value=10)
        assert fm.transform_filter(q) == {
            "type": "filter",
            "predicate": "true",
            "param": "qmax",
        }

    def test_constraint_dict_still_works(self):
        # The dict-constraint branch (unrelated to .ref()) must be unaffected.
        result = fm.transform_filter({"cat": "a"})
        assert result == {"type": "filter", "predicate": 'datum.cat == "a"'}


# ── Site 3: conditional value ─────────────────────────────────────────────────


class TestConditionalValueRef:
    def test_registered_param_ref_resolves_to_static_value(self):
        q = fm.param("qmax", value=10)
        sel = fm.selection_point()
        chart = (
            fm.Chart(DF)
            .mark_point()
            .encode(x="x", y="y")
            .conditional(sel.when(fm.Color("cat")).otherwise(q.ref()))
            .add_params(q)
        )
        spec = chart.to_dict()
        (cond,) = spec["conditionals"]
        assert cond["if_not"] == {"kind": "opacity", "value": 10.0}

    def test_unregistered_ref_raises_value_error_naming_the_parameter(self):
        ghost = fm.param("ghost", value=1)
        sel = fm.selection_point()
        chart = (
            fm.Chart(DF)
            .mark_point()
            .encode(x="x", y="y")
            .conditional(sel.when(fm.Color("cat")).otherwise(ghost.ref()))
        )
        with pytest.raises(ValueError, match="ghost"):
            chart.to_dict()

    def test_registered_non_variable_parameter_ref_raises(self):
        # A ref naming a registered Selection (not a VariableParameter) has
        # no static value to resolve as a literal.
        driver = fm.selection_point()
        other = fm.selection_point()
        chart = (
            fm.Chart(DF)
            .mark_point()
            .encode(x="x", y="y")
            .conditional(driver.when(fm.Color("cat")).otherwise(other.ref()))
            .add_selection(other)
        )
        with pytest.raises(ValueError, match=other.name):
            chart.to_dict()

    def test_standalone_to_spec_dict_ref_raises_no_registry(self):
        # ConditionalSpec.to_spec_dict() called directly (no chart, no
        # registry) cannot validate registration -- and must not fabricate
        # a value either. Every EncodingValue numeric variant *replaces*
        # the targeted channel on the instance (ferrum-wasm's
        # apply_value_common does `*view.opacity = *o`, not a no-op check),
        # so any placeholder value would silently change marks whose real
        # value differs from it. The only safe behavior is to refuse.
        sel = fm.selection_point()
        cond = sel.when(sel.ref()).otherwise(fm.value("#ccc"))
        assert isinstance(cond, ConditionalSpec)
        with pytest.raises(ValueError, match="cannot be resolved without"):
            cond.to_spec_dict()

    def test_resolve_param_ref_with_no_registry_raises_not_fabricates(self):
        # No registry to validate against -- must refuse, never invent a
        # wire value (the exact silent-mangling failure mode this fix
        # eliminates for transform_filter, now also closed here: a
        # fabricated {"kind": "opacity", "value": 1.0} would overwrite any
        # mark's real opacity, not leave it alone).
        with pytest.raises(ValueError, match="anything"):
            _resolve_param_ref("anything", channel="color", params=None)

    def test_value_none_param_ref_raises(self):
        empty = fm.param("empty")  # value defaults to None
        sel = fm.selection_point()
        chart = (
            fm.Chart(DF)
            .mark_point()
            .encode(x="x", y="y")
            .conditional(sel.when(fm.Color("cat")).otherwise(empty.ref()))
            .add_params(empty)
        )
        with pytest.raises(ValueError, match="empty"):
            chart.to_dict()
