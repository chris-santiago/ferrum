"""Tests for D6 sub-task 5a: reactive-parameter Python API surface.

Covers:
- fm.param() / VariableParameter / Parameter base class
- Selection as a Parameter subtype
- selection_point(bind=...) and to_param_spec_dict()
- fm.when(...).then(...).otherwise(...) module-level builder
- Back-compat: existing sel.when(...).otherwise(...) API unchanged
- Back-compat: Selection.to_spec_dict() shape unchanged
"""

from __future__ import annotations

import ferrum as fm
from ferrum.parameter import Parameter, VariableParameter
from ferrum.selection import ConditionalSpec, Selection


# ── VariableParameter / fm.param ─────────────────────────────────────────────


class TestVariableParameter:
    def test_param_returns_variable_parameter(self):
        p = fm.param("t", value=50)
        assert isinstance(p, VariableParameter)

    def test_param_is_parameter_subtype(self):
        p = fm.param("t", value=50)
        assert isinstance(p, Parameter)

    def test_ref_returns_param_dict(self):
        p = fm.param("t", value=50)
        assert p.ref() == {"param": "t"}

    def test_to_param_spec_dict_no_bind(self):
        p = fm.param("t", value=50)
        spec = p.to_param_spec_dict()
        assert spec == {"name": "t", "kind": "variable", "value": 50}
        # "bind" key must be absent (not emitted as null)
        assert "bind" not in spec

    def test_to_param_spec_dict_with_bind_legend(self):
        p = fm.param("t", value=50, bind="legend")
        spec = p.to_param_spec_dict()
        assert spec == {"name": "t", "kind": "variable", "value": 50, "bind": "legend"}

    def test_to_param_spec_dict_with_bind_dict(self):
        bind_dict = {"input": "range", "min": 0, "max": 100}
        p = fm.param("thresh", value=30, bind=bind_dict)
        spec = p.to_param_spec_dict()
        assert spec["bind"] == bind_dict

    def test_value_none_is_emitted(self):
        # value=None is a valid initial value; key must be present
        p = fm.param("s")
        spec = p.to_param_spec_dict()
        assert "value" in spec
        assert spec["value"] is None

    def test_name_stored(self):
        p = fm.param("alpha", value=0.5)
        assert p.name == "alpha"

    def test_frozen_dataclass(self):
        p = fm.param("t", value=50)
        import pytest

        with pytest.raises((AttributeError, TypeError)):
            p.name = "other"  # type: ignore[misc]


# ── Selection as a Parameter subtype ─────────────────────────────────────────


class TestSelectionAsParameter:
    def test_selection_point_is_parameter(self):
        sel = fm.selection_point()
        assert isinstance(sel, Parameter)

    def test_selection_interval_is_parameter(self):
        brush = fm.selection_interval(name="brush", encodings=["x"])
        assert isinstance(brush, Parameter)

    def test_selection_ref_uses_name(self):
        sel = fm.selection_point(name="my_sel")
        assert sel.ref() == {"param": "my_sel"}

    def test_selection_interval_to_param_spec_dict(self):
        brush = fm.selection_interval(name="brush", encodings=["x"])
        spec = brush.to_param_spec_dict()
        assert spec["name"] == "brush"
        assert spec["kind"] == "interval"
        assert "select" in spec
        # select carries the existing params (translate, zoom, resolve, encodings)
        assert isinstance(spec["select"], dict)
        assert "encodings" in spec["select"]
        assert spec["select"]["encodings"] == ["x"]
        # no bind key when not set
        assert "bind" not in spec

    def test_selection_point_to_param_spec_dict_no_bind(self):
        sel = fm.selection_point(name="sel1")
        spec = sel.to_param_spec_dict()
        assert spec["name"] == "sel1"
        assert spec["kind"] == "point"
        assert "select" in spec
        assert "bind" not in spec

    def test_selection_point_bind_legend(self):
        sel = fm.selection_point(bind="legend")
        spec = sel.to_param_spec_dict()
        assert spec["bind"] == "legend"

    def test_selection_point_bind_legend_in_fm_namespace(self):
        sel = fm.selection_point(bind="legend")
        assert isinstance(sel, Parameter)
        spec = sel.to_param_spec_dict()
        assert "bind" in spec
        assert spec["bind"] == "legend"


# ── fm.when() module-level builder ───────────────────────────────────────────


class TestWhenBuilder:
    def test_when_then_otherwise_returns_conditional_spec(self):
        sel = fm.selection_point(name="sel1")
        result = fm.when(sel).then(1).otherwise(0.2)
        assert isinstance(result, ConditionalSpec)

    def test_when_conditional_spec_selection_name(self):
        sel = fm.selection_point(name="sel1")
        result = fm.when(sel).then(fm.value(1)).otherwise(fm.value(0.2))
        assert result.selection_name == "sel1"

    def test_when_wraps_plain_numbers_via_value(self):
        sel = fm.selection_point(name="sel2")
        result = fm.when(sel).then(1).otherwise(0.2)
        # if_selected and if_not must be _LiteralValue instances (value()-wrapped)
        from ferrum.selection import _LiteralValue

        assert isinstance(result.if_selected, _LiteralValue)
        assert isinstance(result.if_not, _LiteralValue)
        assert result.if_selected.val == 1
        assert result.if_not.val == 0.2

    def test_when_to_spec_dict_has_expected_shape(self):
        sel = fm.selection_point(name="sel3")
        cond = fm.when(sel).then(1).otherwise(0.2)
        spec = cond.to_spec_dict()
        assert spec["selection_name"] == "sel3"
        # if_selected and if_not are serialized as opacity-tagged dicts
        assert spec["if_selected"] == {"kind": "opacity", "value": 1.0}
        assert spec["if_not"] == {"kind": "opacity", "value": 0.2}
        assert "channel" in spec

    def test_when_with_already_wrapped_values(self):
        sel = fm.selection_point(name="sel4")
        cond = fm.when(sel).then(fm.value("#ff0000")).otherwise(fm.value("#cccccc"))
        spec = cond.to_spec_dict()
        assert spec["if_selected"]["kind"] == "color"
        assert spec["if_not"]["kind"] == "color"

    def test_when_rejects_variable_parameter(self):
        """fm.when() must raise TypeError when given a VariableParameter."""
        import pytest

        vp = fm.param("k", value=3)
        with pytest.raises(TypeError, match="fm.when"):
            fm.when(vp)


# ── Back-compat: sel.when().otherwise() unchanged ────────────────────────────


class TestBackCompat:
    def test_sel_when_method_returns_conditional_spec(self):
        sel = fm.selection_point()
        cond = sel.when(fm.Color("category")).otherwise(fm.value("#cccccc"))
        assert isinstance(cond, ConditionalSpec)

    def test_sel_when_method_selection_name_correct(self):
        sel = fm.selection_point(name="orig_sel")
        cond = sel.when(fm.Color("category")).otherwise(fm.value("#cccccc"))
        assert cond.selection_name == "orig_sel"

    def test_selection_to_spec_dict_unchanged(self):
        """Selection.to_spec_dict() shape must not regress (Phase 11c wire)."""
        sel = fm.selection_point(name="bc_sel")
        spec = sel.to_spec_dict()
        # existing wire: {"type": kind, "name": name, ...params}
        assert spec["type"] == "point"
        assert spec["name"] == "bc_sel"
        # spot-check a known param key
        assert "resolve" in spec

    def test_selection_interval_to_spec_dict_unchanged(self):
        brush = fm.selection_interval(name="bc_brush", encodings=["x"])
        spec = brush.to_spec_dict()
        assert spec["type"] == "interval"
        assert spec["name"] == "bc_brush"
        assert spec["encodings"] == ["x"]

    def test_selection_without_bind_to_spec_dict_unchanged(self):
        """A param-free Selection round-trips its to_spec_dict() unchanged."""
        sel = fm.selection_point(name="roundtrip")
        original = sel.to_spec_dict()
        # Must have the keys that exist today; no extra keys injected
        assert "type" in original
        assert "name" in original
        # No bind-related keys in the legacy to_spec_dict
        assert "bind" not in original

    def test_selection_with_bind_to_spec_dict_still_unchanged(self):
        """bind field on Selection must NOT leak into to_spec_dict() (only to_param_spec_dict)."""
        sel = fm.selection_point(name="bnd", bind="legend")
        spec = sel.to_spec_dict()
        # to_spec_dict is the Rust SelectionSpec wire — bind does not belong there
        assert "bind" not in spec


# ── Public API accessibility ──────────────────────────────────────────────────


class TestPublicAPI:
    def test_param_in_fm_namespace(self):
        assert hasattr(fm, "param")
        assert callable(fm.param)

    def test_when_in_fm_namespace(self):
        assert hasattr(fm, "when")
        assert callable(fm.when)

    def test_parameter_in_fm_namespace(self):
        assert hasattr(fm, "Parameter")

    def test_variable_parameter_in_fm_namespace(self):
        assert hasattr(fm, "VariableParameter")

    def test_all_entries(self):
        for name in ("param", "when", "Parameter", "VariableParameter"):
            assert name in fm.__all__, f"{name!r} missing from fm.__all__"
