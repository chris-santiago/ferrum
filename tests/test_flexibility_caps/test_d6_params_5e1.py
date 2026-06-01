"""Tests for D6 sub-task 5e-1b: reactive params carried through Python composition merge.

Covers:
- Baseline: single chart already emits interaction.params via Rust (5e-1a).
- hconcat/vconcat: merged scene's interaction.params includes params from child charts.
- Dedup: two children with the same param name produce exactly one entry (first-seen wins).
- Param-free composition: merged interaction.params is [] (no spurious params).
- HTML extraction: _interaction_config_from_dict includes params for a param composition.
- Byte-stability: existing param-free composition paths are unaffected.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum._html import _interaction_config_from_dict
from ferrum._interactive import _render_scene


def _df() -> pl.DataFrame:
    return pl.DataFrame({"t": [0, 50, 100], "a": [1, 2, 3], "b": [4, 5, 6]})


def _interaction(scene: dict) -> dict:
    return scene.get("interaction", {})


def _params(scene: dict) -> list:
    return _interaction(scene).get("params", [])


def _param_names(scene: dict) -> list[str]:
    return [p["name"] for p in _params(scene)]


# ── Baseline: single chart already works (5e-1a) ─────────────────────────────


class TestSingleChartBaseline:
    def test_domain_param_in_single_chart_scene(self):
        """Single chart with a domain param must emit interaction.params (Rust side)."""
        chart = (
            fm.Chart(_df())
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )
        scene_json, _ = _render_scene(chart)
        scene = json.loads(scene_json)
        assert "params" in _interaction(scene), (
            "interaction.params must be present for a single param chart"
        )
        names = _param_names(scene)
        assert "d" in names

    def test_selection_param_in_single_chart_scene(self):
        """A selection registered on a single chart appears in interaction.params."""
        brush = fm.selection_interval(name="brush", encodings=["x"])
        chart = fm.Chart(_df()).mark_point().encode(x=fm.X("t"), y=fm.Y("a")).add_selection(brush)
        scene_json, _ = _render_scene(chart)
        scene = json.loads(scene_json)
        names = _param_names(scene)
        assert "brush" in names

    def test_param_free_single_chart_has_empty_params_or_no_key(self):
        """A single param-free chart emits params=[] or omits the key entirely."""
        chart = fm.Chart(_df()).mark_point().encode(x="a", y="b")
        scene_json, _ = _render_scene(chart)
        scene = json.loads(scene_json)
        # Rust may emit [] or omit the key; either is acceptable for a param-free chart.
        params = _interaction(scene).get("params", [])
        assert params == []


# ── hconcat: params from children flow into merged scene ─────────────────────


class TestHConcatParamsMerge:
    def test_child_param_appears_in_hconcat_scene(self):
        """hconcat: a param in the detail child must appear in the merged scene."""
        df = _df()
        detail = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )
        overview = fm.Chart(df).mark_point().encode(x=fm.X("t"), y=fm.Y("b"))

        composed = fm.hconcat(overview, detail)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        assert "params" in _interaction(scene), "merged interaction must have a params key"
        names = _param_names(scene)
        assert "d" in names, f"param 'd' must be in merged params; got {names}"

    def test_selection_in_overview_appears_in_hconcat_scene(self):
        """hconcat: a selection in the overview child flows into merged params."""
        df = _df()
        brush = fm.selection_interval(name="brush", encodings=["x"])
        overview = fm.Chart(df).mark_point().encode(x=fm.X("t"), y=fm.Y("b")).add_selection(brush)
        detail = fm.Chart(df).mark_point().encode(x=fm.X("t"), y=fm.Y("a"))

        composed = fm.hconcat(overview, detail)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        names = _param_names(scene)
        assert "brush" in names, f"selection 'brush' must appear in merged params; got {names}"

    def test_params_from_both_children_merged(self):
        """hconcat: params from two different children are both present in merged scene."""
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

        names = _param_names(scene)
        assert "brush" in names, f"'brush' missing from merged params; got {names}"
        assert "d" in names, f"'d' missing from merged params; got {names}"

    def test_param_value_preserved_in_merged_scene(self):
        """The merged param entry preserves its value and kind from the child."""
        df = _df()
        detail = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )
        overview = fm.Chart(df).mark_point().encode(x=fm.X("t"), y=fm.Y("b"))

        composed = fm.hconcat(overview, detail)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        params = _params(scene)
        d_entry = next((p for p in params if p["name"] == "d"), None)
        assert d_entry is not None
        assert d_entry["kind"] == "variable"
        assert d_entry["value"] == [0, 100]


# ── vconcat: params from children flow into merged scene ─────────────────────


class TestVConcatParamsMerge:
    def test_child_param_appears_in_vconcat_scene(self):
        """vconcat: a param in a child must appear in the merged scene."""
        df = _df()
        top = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )
        bottom = fm.Chart(df).mark_point().encode(x=fm.X("t"), y=fm.Y("b"))

        composed = fm.vconcat(top, bottom)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        assert "params" in _interaction(scene)
        names = _param_names(scene)
        assert "d" in names, f"param 'd' must be in vconcat merged params; got {names}"


# ── Deduplication ─────────────────────────────────────────────────────────────


class TestParamDeduplication:
    def test_same_name_in_two_children_appears_once(self):
        """If two children declare a param with the same name, it appears exactly once."""
        df = _df()
        left = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("shared", value=[0, 50])}), y=fm.Y("a"))
        )
        right = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("shared", value=[0, 50])}), y=fm.Y("b"))
        )

        composed = fm.hconcat(left, right)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        names = _param_names(scene)
        assert names.count("shared") == 1, (
            f"Param 'shared' must appear exactly once after dedup; got names={names}"
        )

    def test_first_seen_wins_on_dedup(self):
        """First-seen param wins when two children disagree on value for the same name."""
        df = _df()
        # left declares shared param with value=[0, 50]
        left = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("shared", value=[0, 50])}), y=fm.Y("a"))
        )
        # right declares same name with value=[0, 999]
        right = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("shared", value=[0, 999])}), y=fm.Y("b"))
        )

        composed = fm.hconcat(left, right)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        params = _params(scene)
        shared = next(p for p in params if p["name"] == "shared")
        assert shared["value"] == [0, 50], (
            f"First-seen value [0, 50] must win over second child's [0, 999]; got {shared['value']}"
        )


# ── Param-free composition: no spurious params ───────────────────────────────


class TestParamFreeComposition:
    def test_hconcat_param_free_has_empty_params(self):
        """hconcat of param-free charts must produce params=[] in the merged scene."""
        df = _df()
        left = fm.Chart(df).mark_point().encode(x="a", y="b")
        right = fm.Chart(df).mark_point().encode(x="t", y="b")

        composed = fm.hconcat(left, right)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        params = _interaction(scene).get("params", [])
        assert params == [], f"Param-free hconcat should have empty params; got {params}"

    def test_vconcat_param_free_has_empty_params(self):
        """vconcat of param-free charts must produce params=[] in the merged scene."""
        df = _df()
        top = fm.Chart(df).mark_point().encode(x="a", y="b")
        bottom = fm.Chart(df).mark_point().encode(x="t", y="b")

        composed = fm.vconcat(top, bottom)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        params = _interaction(scene).get("params", [])
        assert params == [], f"Param-free vconcat should have empty params; got {params}"

    def test_hconcat_scene_byte_stability(self):
        """Rendering the same param-free hconcat twice yields identical JSON."""
        df = _df()
        left = fm.Chart(df).mark_point().encode(x="a", y="b")
        right = fm.Chart(df).mark_point().encode(x="t", y="b")

        a, _ = _render_scene(fm.hconcat(left, right))
        b, _ = _render_scene(fm.hconcat(left, right))
        assert a == b, "Param-free hconcat scene must be byte-stable across renders"


# ── HTML-level extraction includes params ────────────────────────────────────


class TestHTMLInteractionExtraction:
    def test_interaction_config_includes_params_for_param_composition(self):
        """_interaction_config_from_dict passes params through to the HTML layer."""
        df = _df()
        detail = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": fm.param("d", value=[0, 100])}), y=fm.Y("a"))
        )
        overview = fm.Chart(df).mark_point().encode(x=fm.X("t"), y=fm.Y("b"))

        composed = fm.hconcat(overview, detail)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        config = json.loads(_interaction_config_from_dict(scene))
        assert "params" in config, "HTML interaction config must include params key"
        names = [p["name"] for p in config["params"]]
        assert "d" in names, f"param 'd' must appear in HTML interaction config; got {names}"

    def test_interaction_config_param_free_composition_has_empty_params(self):
        """HTML interaction config for a param-free composition has params=[]."""
        df = _df()
        left = fm.Chart(df).mark_point().encode(x="a", y="b")
        right = fm.Chart(df).mark_point().encode(x="t", y="b")

        composed = fm.hconcat(left, right)
        scene_json, _ = _render_scene(composed)
        scene = json.loads(scene_json)

        config = json.loads(_interaction_config_from_dict(scene))
        params = config.get("params", [])
        assert params == [], f"Param-free HTML config must have empty params; got {params}"
