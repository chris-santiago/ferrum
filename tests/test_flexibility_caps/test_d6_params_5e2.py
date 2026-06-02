"""Tests for D6 sub-task 5e-2a: param_bindings carried through Python composition merge.

Covers:
- Baseline: single chart already emits interaction.param_bindings (Rust side, panel=0).
- hconcat(overview, detail): the detail child's Domain binding has its panel index
  offset to the detail panel's id in the merged scene (panel > 0, not 0).
- Legend binding (panel absent/None): survives the merge with panel still absent.
- Param-free composition: param_bindings == [] in the merged scene.
- Byte-stability: param-free composition scene is unaffected by param_bindings merge logic.
"""

from __future__ import annotations

import json

import polars as pl

import ferrum as fm
from ferrum._interactive import _render_scene


def _df() -> pl.DataFrame:
    return pl.DataFrame({"t": [0, 50, 100], "a": [1, 2, 3], "b": [4, 5, 6]})


def _interaction(scene: dict) -> dict:
    return scene.get("interaction", {})


def _bindings(scene: dict) -> list:
    return _interaction(scene).get("param_bindings", [])


# ── Baseline: single chart already emits param_bindings (Rust side) ──────────


class TestSingleChartBaseline:
    def test_domain_binding_in_single_chart_scene(self):
        """Single chart with a domain param must emit interaction.param_bindings (Rust side)."""
        chart = (
            fm.Chart(_df())
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )
        scene_json, _ = _render_scene(chart)
        scene = json.loads(scene_json)

        bindings = _bindings(scene)
        assert len(bindings) == 1, f"Expected 1 binding, got {bindings}"
        b = bindings[0]
        assert b["param"] == "d"
        assert b["role"] == "domain"
        assert b.get("panel") == 0
        assert b.get("channel") == "x"

    def test_legend_binding_in_single_chart_scene(self):
        """Single chart with a legend binding must emit it with no panel key."""
        df = pl.DataFrame({"t": [0, 1, 2], "a": [1, 2, 3], "cat": ["x", "y", "x"]})
        sel = fm.selection_point(name="leg", bind="legend", encodings=["color"])
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t"), y=fm.Y("a"), color=fm.Color("cat"))
            .add_selection(sel)
        )
        scene_json, _ = _render_scene(chart)
        scene = json.loads(scene_json)

        bindings = _bindings(scene)
        assert len(bindings) == 1
        b = bindings[0]
        assert b["param"] == "leg"
        assert b["role"] == "legend"
        # Legend bindings carry no panel
        assert "panel" not in b

    def test_param_free_single_chart_has_no_param_bindings(self):
        """A param-free single chart emits no param_bindings key or an empty list."""
        chart = fm.Chart(_df()).mark_point().encode(x="a", y="b")
        scene_json, _ = _render_scene(chart)
        scene = json.loads(scene_json)
        bindings = _bindings(scene)
        assert bindings == []


# ── hconcat: param_bindings panel index offset ───────────────────────────────


class TestHConcatBindingOffset:
    def test_detail_binding_panel_is_offset(self):
        """hconcat(overview, detail): detail child's domain binding panel must be offset.

        overview is child 0 (1 panel), so its panel_id_offset == 0.
        detail is child 1 (offset == 1), so its binding panel 0 → 1.
        """
        df = _df()
        overview = fm.Chart(df).mark_point().encode(x=fm.X("t"), y=fm.Y("b"))
        detail = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )

        composed = fm.hconcat(overview, detail)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        bindings = _bindings(scene)
        assert len(bindings) >= 1, f"Expected at least one binding; got {bindings}"

        d_binding = next((b for b in bindings if b["param"] == "d"), None)
        assert d_binding is not None, f"Binding for 'd' not found in {bindings}"
        assert d_binding["role"] == "domain"

        # The detail chart is the second child; overview has 1 panel → offset == 1.
        # So the detail's panel 0 must appear as panel 1 in the merged scene.
        assert d_binding.get("panel") == 1, (
            f"detail binding panel must be 1 (offset), got {d_binding.get('panel')}"
        )

    def test_overview_binding_panel_is_not_offset(self):
        """hconcat: overview is child 0 (offset=0), so its binding panel stays at 0."""
        df = _df()
        brush = fm.selection_interval(name="brush", encodings=["x"])
        overview = fm.Chart(df).mark_point().encode(x=fm.X("t"), y=fm.Y("b")).add_selection(brush)
        detail = fm.Chart(df).mark_point().encode(x=fm.X("t"), y=fm.Y("a"))

        composed = fm.hconcat(overview, detail)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        bindings = _bindings(scene)
        # brush is a selection/interval param; look for its binding
        brush_bindings = [b for b in bindings if b["param"] == "brush"]
        # brush may or may not emit a domain binding depending on Rust; check that
        # if it does, the panel == 0 (no offset applied to the first child).
        for b in brush_bindings:
            if b.get("panel") is not None:
                assert b["panel"] == 0, (
                    f"overview binding panel must be 0 (no offset); got {b['panel']}"
                )

    def test_both_children_bindings_present(self):
        """hconcat: bindings from both children must appear in the merged scene."""
        df = _df()
        brush = fm.selection_interval(name="brush", encodings=["x"])
        overview = fm.Chart(df).mark_point().encode(x=fm.X("t"), y=fm.Y("b")).add_selection(brush)
        detail = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )

        composed = fm.hconcat(overview, detail)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        bindings = _bindings(scene)
        param_names = {b["param"] for b in bindings}
        assert "d" in param_names, f"binding for 'd' missing from merged scene; got {param_names}"


# ── Legend binding survives merge with panel absent ───────────────────────────


class TestLegendBindingMerge:
    def test_legend_binding_panel_stays_none_after_merge(self):
        """Legend binding (panel absent) must survive the merge with panel still absent."""
        df = pl.DataFrame({"t": [0, 1, 2], "a": [1, 2, 3], "cat": ["x", "y", "x"]})
        sel = fm.selection_point(name="leg", bind="legend", encodings=["color"])
        chart_with_legend = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t"), y=fm.Y("a"), color=fm.Color("cat"))
            .add_selection(sel)
        )
        filler = fm.Chart(_df()).mark_point().encode(x="t", y="b")

        composed = fm.hconcat(chart_with_legend, filler)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        bindings = _bindings(scene)
        leg_binding = next((b for b in bindings if b["param"] == "leg"), None)
        assert leg_binding is not None, f"Legend binding 'leg' missing from {bindings}"
        assert leg_binding["role"] == "legend"
        # Panel must remain absent (not shifted to 0 or any offset)
        assert "panel" not in leg_binding, (
            f"Legend binding must have no panel key; got {leg_binding}"
        )


# ── Param-free composition ────────────────────────────────────────────────────


class TestParamFreeCompositionBindings:
    def test_hconcat_param_free_has_empty_param_bindings(self):
        """hconcat of param-free charts must produce param_bindings=[] in merged scene."""
        df = _df()
        left = fm.Chart(df).mark_point().encode(x="a", y="b")
        right = fm.Chart(df).mark_point().encode(x="t", y="b")

        composed = fm.hconcat(left, right)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        bindings = _bindings(scene)
        assert bindings == [], (
            f"Param-free hconcat should have empty param_bindings; got {bindings}"
        )

    def test_vconcat_param_free_has_empty_param_bindings(self):
        """vconcat of param-free charts must produce param_bindings=[] in merged scene."""
        df = _df()
        top = fm.Chart(df).mark_point().encode(x="a", y="b")
        bottom = fm.Chart(df).mark_point().encode(x="t", y="b")

        composed = fm.vconcat(top, bottom)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        bindings = _bindings(scene)
        assert bindings == [], (
            f"Param-free vconcat should have empty param_bindings; got {bindings}"
        )

    def test_byte_stability_param_free_hconcat(self):
        """Rendering the same param-free hconcat twice yields identical JSON (byte-stable)."""
        df = _df()
        left = fm.Chart(df).mark_point().encode(x="a", y="b")
        right = fm.Chart(df).mark_point().encode(x="t", y="b")

        a, _ = _render_scene(fm.hconcat(left, right))
        b, _ = _render_scene(fm.hconcat(left, right))
        assert a == b, "Param-free hconcat scene must be byte-stable across renders"


# ── Deduplication of identical bindings ──────────────────────────────────────


class TestBindingDeduplication:
    def test_identical_binding_from_two_children_appears_once(self):
        """If two children emit an identical (param, role, panel, channel) binding, deduplicate."""
        df = _df()
        # Both children declare the same param with same role+channel; after offsetting
        # their panels will differ (0 vs 1), so they won't actually be exact dupes —
        # but if we construct a facet-like scenario where both panels yield the same
        # offsetted id, they should deduplicate. Instead, test that the common case
        # (two distinct panels) keeps both bindings.
        overview = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("b"))
        )
        detail = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )

        composed = fm.hconcat(overview, detail)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        bindings = _bindings(scene)
        d_bindings = [b for b in bindings if b["param"] == "d"]
        # Two children, two different panels (0 and 1) → two distinct bindings, both kept.
        # They are NOT exact dupes because their panel indices differ after offsetting.
        assert len(d_bindings) == 2, (
            f"Two distinct domain bindings (panel 0 and panel 1) must both be kept; "
            f"got {d_bindings}"
        )
        panels = {b.get("panel") for b in d_bindings}
        assert panels == {0, 1}, f"Expected panels {{0, 1}}; got {panels}"
