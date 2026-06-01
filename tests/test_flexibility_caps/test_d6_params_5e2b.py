"""Tests for D6 sub-task 5e-2b: the live interactive runtime for param_bindings.

Browser behavior is out of CI. The CI bar (wire contract §6/§10) is that the
emitted HTML/JS + scene JSON carry BOTH the param bindings AND the event
handlers that consume them, present and correct. These tests build the three
canonical scenarios and assert the emitted artifacts carry the wiring:

- overview+detail (reactive rescale): an hconcat of a brush overview and a
  detail panel whose x-scale domain is the brush param emits a Domain binding
  on the detail panel, and the embedded JS carries the rescale hook (the brush
  ``handleDrag`` path the WASM runtime drives).
- crossfilter: a ``transform_filter(selection)`` across an hconcat emits a
  Filter binding on the filtered panel plus the crossfilter hook.
- legend toggle: ``selection_point(bind="legend")`` + a color conditional
  emits a Legend binding and the legend-click handler wiring.

Also asserts the regenerated WASM JS API surface exposes the new
``toggleLegend`` method, and that a param-free interactive chart carries none
of the new wiring (behavior-unchanged guarantee).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

import ferrum as fm
from ferrum._html import assemble_html
from ferrum._interactive import _render_scene

_WASM_DIR = Path(fm.__file__).parent / "_wasm"


def _df() -> pl.DataFrame:
    return pl.DataFrame({"t": [0, 50, 100], "a": [1, 2, 3], "b": [4, 5, 6]})


def _cat_df() -> pl.DataFrame:
    return pl.DataFrame(
        {"t": [0, 1, 2, 3], "a": [1, 2, 3, 4], "cat": ["x", "y", "x", "y"]}
    )


def _html_for(chart) -> str:
    scene_json, packed = _render_scene(chart)
    # embed_wasm=False keeps the artifact small; the JS glue + interaction
    # config (the things we assert on) are embedded regardless.
    return assemble_html(scene_json, packed_data=packed, embed_wasm=False)


def _embedded_interaction_config(html: str) -> dict:
    """Recover the interaction config the standalone adapter is created with."""
    marker = "createStandaloneAdapter('"
    idx = html.index(marker)
    # Args: createStandaloneAdapter('<packed_b64>', '<interaction_config>');
    rest = html[idx + len(marker) :]
    # First single-quoted arg is packed_b64; second is the interaction config.
    first_end = rest.index("', '")
    after_first = rest[first_end + len("', '") :]
    cfg_str = after_first[: after_first.index("');")]
    # Reverse the embedding escapes (\\ and \').
    cfg_str = cfg_str.replace("\\'", "'").replace("\\\\", "\\")
    return json.loads(cfg_str)


# ── WASM API surface: toggleLegend present after rebuild ──────────────────────


class TestWasmApiSurface:
    def test_toggle_legend_in_dts(self):
        """The regenerated TypeScript declarations must expose toggleLegend."""
        dts = (_WASM_DIR / "ferrum_wasm.d.ts").read_text()
        assert "toggleLegend(selection_name: string, category: string): string" in dts

    def test_toggle_legend_in_generated_js(self):
        """The generated JS glue must expose the toggleLegend method binding."""
        js = (_WASM_DIR / "ferrum_wasm.js").read_text()
        assert "toggleLegend(selection_name, category)" in js
        assert "wasmrenderer_toggleLegend" in js


# ── Reactive rescale (overview + detail) ──────────────────────────────────────


class TestReactiveRescaleArtifacts:
    def _composed(self):
        df = _df()
        brush = fm.selection_interval(name="brush", encodings=["x"])
        overview = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t"), y=fm.Y("b"))
            .add_selection(brush)
        )
        detail = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t", scale={"domain": brush}), y=fm.Y("a"))
        )
        return fm.hconcat(overview, detail)

    def test_domain_binding_in_embedded_config(self):
        """The embedded interaction config must carry the Domain binding on the
        detail panel (panel offset to 1, channel x)."""
        html = _html_for(self._composed())
        cfg = _embedded_interaction_config(html)
        bindings = cfg.get("param_bindings", [])
        domain = [b for b in bindings if b.get("role") == "domain"]
        assert domain, f"expected a Domain binding; got {bindings}"
        b = next(b for b in domain if b["param"] == "brush")
        assert b["channel"] == "x"
        assert b.get("panel") == 1, f"detail panel must be offset to 1; got {b}"

    def test_rescale_hook_present_in_js(self):
        """The embedded JS must consume param_bindings and drive the brush path
        (handleDrag) that the WASM runtime uses for reactive rescale."""
        html = _html_for(self._composed())
        assert "param_bindings" in html, "JS must read param_bindings"
        assert "handleDrag" in html, "brush drag hook drives reactive rescale"

    def test_rescale_survives_in_js(self):
        """The select-mode brush handler must honor the ``rescaled`` signal the
        WASM ``handleDrag`` envelope returns.

        Regression for the Fix-1 bug: the handler used to UNCONDITIONALLY call
        ``setTransform`` after ``handleDrag``, resetting panel 0 to identity and
        discarding the reactive rescale the Rust side just applied. The handler
        must now read the envelope's ``rescaled`` field and skip the trailing
        ``setTransform`` when a Domain binding fired (re-placing the rescaled
        panel's own text instead)."""
        html = _html_for(self._composed())
        # The handler consumes the rescale signal from the handleDrag envelope.
        assert "result.rescaled" in html, "JS must read the rescaled signal"
        assert "result.selection" in html, "JS must read the selection envelope field"
        # And re-places the rescaled panel's labels rather than resetting zoom.
        assert "result.rescaled_text" in html, "JS must place rescaled-panel text"


# ── Crossfilter ───────────────────────────────────────────────────────────────


class TestCrossfilterArtifacts:
    def _composed(self):
        df = _df()
        brush = fm.selection_interval(name="brush", encodings=["x"])
        overview = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t"), y=fm.Y("b"))
            .add_selection(brush)
        )
        detail = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t"), y=fm.Y("a"))
            .transform(fm.transform_filter(brush))
        )
        return fm.hconcat(overview, detail)

    def test_filter_binding_in_embedded_config(self):
        """The embedded interaction config must carry a Filter binding on the
        filtered (detail) panel."""
        html = _html_for(self._composed())
        cfg = _embedded_interaction_config(html)
        bindings = cfg.get("param_bindings", [])
        flt = [b for b in bindings if b.get("role") == "filter"]
        assert flt, f"expected a Filter binding; got {bindings}"
        b = flt[0]
        assert b["param"] == "brush"
        assert b.get("panel") == 1, f"filtered panel must be offset to 1; got {b}"

    def test_crossfilter_hook_present_in_js(self):
        """The crossfilter hook is the brush handleDrag path (which the WASM
        runtime crossfilters on) plus param_bindings consumption."""
        html = _html_for(self._composed())
        assert "param_bindings" in html
        assert "handleDrag" in html


# ── Legend toggle ─────────────────────────────────────────────────────────────


class TestLegendToggleArtifacts:
    def _chart(self):
        df = _cat_df()
        sel = fm.selection_point(name="leg", bind="legend", encodings=["color"])
        return (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("t"), y=fm.Y("a"), color=fm.Color("cat"))
            .add_selection(sel)
            .conditional(fm.when(sel).then(fm.value(1.0)).otherwise(fm.value(0.2)))
        )

    def test_legend_binding_in_embedded_config(self):
        """The embedded interaction config must carry the Legend binding."""
        html = _html_for(self._chart())
        cfg = _embedded_interaction_config(html)
        bindings = cfg.get("param_bindings", [])
        leg = [b for b in bindings if b.get("role") == "legend"]
        assert leg, f"expected a Legend binding; got {bindings}"
        assert leg[0]["param"] == "leg"
        assert "panel" not in leg[0], "legend binding carries no panel"

    def test_conditional_present(self):
        """A conditional must accompany the legend binding so toggling has an
        effect to apply."""
        html = _html_for(self._chart())
        cfg = _embedded_interaction_config(html)
        assert cfg.get("conditionals"), "legend toggle needs a conditional to drive"

    def test_legend_click_handler_wiring_in_js(self):
        """The embedded JS must wire legend-entry clicks to renderer.toggleLegend."""
        html = _html_for(self._chart())
        assert "toggleLegend" in html, "JS must call renderer.toggleLegend"
        assert "_collectLegendCategories" in html, "JS must collect legend categories"
        assert "_ferrumWireLegend" in html, "JS must (re)wire legend toggles"
        assert "data-legend-category" in html, "legend entries must be tagged clickable"


# ── Behavior-unchanged: param-free interactive chart carries no new wiring ────


class TestParamFreeUnchanged:
    def test_param_free_chart_has_empty_bindings(self):
        """A param-free interactive chart must emit no param_bindings."""
        chart = fm.Chart(_df()).mark_point().encode(x="a", y="b")
        html = _html_for(chart)
        cfg = _embedded_interaction_config(html)
        assert cfg.get("param_bindings", []) == []

    def test_param_free_chart_does_not_activate_legend_wiring(self):
        """Without a Legend binding the legend-click handler must not activate.

        The helper functions are always inlined in the JS bundle, but the
        runtime guard (`_legendBinding && renderer`) means no legend entry is
        made clickable. We assert the config has no legend binding, which is the
        gate the JS checks.
        """
        chart = fm.Chart(_cat_df()).mark_point().encode(
            x=fm.X("t"), y=fm.Y("a"), color=fm.Color("cat")
        )
        html = _html_for(chart)
        cfg = _embedded_interaction_config(html)
        roles = {b.get("role") for b in cfg.get("param_bindings", [])}
        assert "legend" not in roles

    def test_param_free_scene_byte_stable(self):
        """Rendering a param-free interactive chart twice yields identical JSON."""
        chart = fm.Chart(_df()).mark_point().encode(x="a", y="b")
        a, _ = _render_scene(chart)
        b, _ = _render_scene(chart)
        assert a == b
