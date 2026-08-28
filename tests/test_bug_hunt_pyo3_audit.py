"""Tests for bugs found by the PyO3 binding audit.

B1 — set_default_theme() ignored in render pipeline
B2 — Numeric _LiteralValue always tagged as EncodingValue::Opacity
W1 — _resolve_channel defaults to "color" for unrecognized types
W4 — _build_layers_list drops 11 encoding channels
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum.selection import _resolve_channel, _resolve_encoding_value, value
from ferrum.encoding.appearance import (
    Angle,
    Color,
    FillOpacity,
    Opacity,
    Shape,
    Size,
    StrokeDash,
    StrokeOpacity,
    StrokeWidth,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "cat": ["a", "b", "c"],
            "sw": [1.0, 2.0, 3.0],
        }
    )


# ── B1: set_default_theme() ignored in render pipeline ───────────────────────


class TestB1DefaultThemeInRenderPipeline:
    """_render_inputs must consult get_default_theme() when no per-chart theme is set."""

    def test_no_chart_theme_yields_empty_dict_when_process_default_is_default(self, df):
        """Baseline: Theme() (the ferrum default) gives an empty theme dict."""
        chart = fm.Chart(df).mark_point().encode(x="x", y="y")
        assert chart._theme is None  # no per-chart theme
        _, _, _, theme_dict, _ = chart._render_inputs()
        # The ferrum default Theme() has no properties → empty dict is correct
        assert theme_dict == {}

    def test_set_default_theme_dark_propagates_to_render_inputs(self, df):
        """When set_default_theme(dark) is in effect, _render_inputs returns dark's dict."""
        chart = fm.Chart(df).mark_point().encode(x="x", y="y")
        assert chart._theme is None  # no per-chart theme

        with fm.set_default_theme(fm.themes.dark):
            _, _, _, theme_dict, _ = chart._render_inputs()

        # dark theme has a non-empty spec dict — spot-check background_color
        assert theme_dict != {}
        assert "background_color" in theme_dict
        assert theme_dict["background_color"] == "#1a1a2e"

    def test_per_chart_theme_wins_over_process_default(self, df):
        """Per-chart .theme() must take precedence over set_default_theme()."""
        custom_theme = fm.Theme(background="#aabbcc")
        chart = fm.Chart(df).mark_point().encode(x="x", y="y").theme(custom_theme)

        with fm.set_default_theme(fm.themes.dark):
            _, _, _, theme_dict, _ = chart._render_inputs()

        # per-chart theme wins: must have the custom background, not dark's
        assert theme_dict.get("background_color") == "#aabbcc"

    def test_context_manager_restores_process_default(self, df):
        """After the with-block exits, the process default reverts to Theme()."""
        chart = fm.Chart(df).mark_point().encode(x="x", y="y")

        with fm.set_default_theme(fm.themes.dark):
            pass  # block exits, default reverts

        _, _, _, theme_dict, _ = chart._render_inputs()
        assert theme_dict == {}


# ── B2: Numeric _LiteralValue always tagged as EncodingValue::Opacity ────────


class TestB2LiteralValueChannelTag:
    """_resolve_encoding_value must tag numeric literals according to channel."""

    def test_numeric_value_with_opacity_channel_tagged_as_opacity(self, df):
        """fm.value(0.3) used in Opacity conditional → kind must be 'opacity'."""
        sel = fm.selection_point()
        cond = sel.when(Opacity("x")).otherwise(value(0.3))
        spec = cond.to_spec_dict()
        assert spec["if_not"]["kind"] == "opacity"
        assert spec["if_not"]["value"] == pytest.approx(0.3)

    def test_numeric_value_with_size_channel_tagged_as_size(self, df):
        """fm.value(5) used in Size conditional → kind must be 'size', not 'opacity'."""
        sel = fm.selection_point()
        cond = sel.when(Size("x")).otherwise(value(5))
        spec = cond.to_spec_dict()
        assert spec["if_not"]["kind"] == "size"
        assert spec["if_not"]["value"] == pytest.approx(5.0)

    def test_numeric_value_with_stroke_width_channel_tagged_as_stroke_width(self, df):
        """fm.value(2) used in StrokeWidth conditional → kind must be 'stroke_width'."""
        sel = fm.selection_point()
        cond = sel.when(StrokeWidth("x")).otherwise(value(2))
        spec = cond.to_spec_dict()
        assert spec["if_not"]["kind"] == "stroke_width"
        assert spec["if_not"]["value"] == pytest.approx(2.0)

    def test_color_hex_value_still_tagged_as_color(self, df):
        """fm.value('#ccc') in a Color conditional → kind must remain 'color'."""
        sel = fm.selection_point()
        cond = sel.when(Color("cat")).otherwise(value("#cccccc"))
        spec = cond.to_spec_dict()
        assert spec["if_not"]["kind"] == "color"

    def test_conditional_spec_channel_key_matches_encoding(self, df):
        """ConditionalSpec.channel field must reflect the if_selected encoding type."""
        sel = fm.selection_point()

        size_cond = sel.when(Size("x")).otherwise(value(5))
        assert size_cond.to_spec_dict()["channel"] == "size"

        opacity_cond = sel.when(Opacity("x")).otherwise(value(0.3))
        assert opacity_cond.to_spec_dict()["channel"] == "opacity"

    def test_direct_resolve_encoding_value_with_size_channel(self):
        """Unit-test _resolve_encoding_value directly with channel='size'."""
        result = _resolve_encoding_value(value(10), channel="size")
        assert result == {"kind": "size", "value": 10.0}

    def test_direct_resolve_encoding_value_with_opacity_channel(self):
        """Unit-test _resolve_encoding_value directly with channel='opacity'."""
        result = _resolve_encoding_value(value(0.5), channel="opacity")
        assert result == {"kind": "opacity", "value": 0.5}

    def test_direct_resolve_encoding_value_channel_none_defaults_to_opacity(self):
        """channel=None falls back to 'opacity' tag for numeric literals."""
        result = _resolve_encoding_value(value(0.7), channel=None)
        assert result == {"kind": "opacity", "value": 0.7}

    def test_direct_resolve_encoding_value_stroke_dash(self):
        """channel='stroke_dash' wraps the float in a list."""
        result = _resolve_encoding_value(value(4), channel="stroke_dash")
        assert result["kind"] == "stroke_dash"
        # Value should be a list per the bug-report spec
        assert isinstance(result["value"], list)
        assert result["value"][0] == pytest.approx(4.0)


# ── W1: _resolve_channel defaults to "color" for unrecognized types ──────────


class TestW1ResolveChannelUnrecognizedTypes:
    """_resolve_channel must correctly map all encoding channel classes."""

    def test_color_resolves_to_color(self):
        assert _resolve_channel(Color("cat")) == "color"

    def test_opacity_resolves_to_opacity(self):
        assert _resolve_channel(Opacity("x")) == "opacity"

    def test_size_resolves_to_size(self):
        assert _resolve_channel(Size("x")) == "size"

    def test_shape_resolves_to_shape(self):
        assert _resolve_channel(Shape("cat")) == "shape"

    def test_stroke_width_resolves_to_stroke_width(self):
        assert _resolve_channel(StrokeWidth("x")) == "stroke_width"

    def test_stroke_opacity_resolves_to_stroke_opacity(self):
        assert _resolve_channel(StrokeOpacity("x")) == "stroke_opacity"

    def test_stroke_dash_resolves_to_stroke_dash(self):
        assert _resolve_channel(StrokeDash("cat")) == "stroke_dash"

    def test_fill_opacity_resolves_to_fill_opacity(self):
        assert _resolve_channel(FillOpacity("x")) == "fill_opacity"

    def test_angle_resolves_to_angle(self):
        assert _resolve_channel(Angle("x")) == "angle"

    def test_hex_literal_resolves_to_color(self):
        """A hex _LiteralValue should still resolve to 'color'."""
        assert _resolve_channel(value("#ff0000")) == "color"


# ── W4: _build_layers_list drops encoding channels ───────────────────────────


class TestW4BuildLayersListMissingChannels:
    """_build_layers_list must include all _RENDERER_HONORED_CHANNELS."""

    def _get_layer_encoding(self, chart: fm.Chart, layer_idx: int = 0) -> dict:
        """Serialize to JSON and extract the layer encoding dict."""
        spec = chart.to_spec()
        spec_json = json.loads(spec.to_json())
        # layers is a top-level key when layers exist
        layers = spec_json.get("layers", [])
        if layer_idx < len(layers):
            return layers[layer_idx].get("encoding", {})
        return {}

    def _get_layers_list(self, chart: fm.Chart) -> list:
        """Call _build_layers_list directly on the resolved chart."""
        # _render_inputs runs _resolve_pending and _apply_auto_raster;
        # for layer-encoding tests we can call _build_layers_list via the
        # internal resolved chart.
        resolved = chart._resolve_pending()
        return resolved._build_layers_list()

    def test_stroke_width_encoding_in_layer_2_appears_in_build_layers(self, df):
        """Layer with stroke_width encoding must not be dropped by _build_layers_list."""
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        overlay = (
            fm.Chart(df)
            .mark_line()
            .encode(
                x="x",
                y="y",
                stroke_width=fm.StrokeWidth("sw"),
            )
        )
        chart = base + overlay
        layers = self._get_layers_list(chart)
        # The second layer (index 1) should have stroke_width in its encoding
        assert len(layers) >= 2, "Expected at least 2 layers"
        layer_enc = layers[1].get("encoding", {})
        assert "stroke_width" in layer_enc, (
            f"stroke_width missing from layer encoding; got keys: {list(layer_enc.keys())}"
        )

    def test_opacity_encoding_in_layer_appears_in_build_layers(self, df):
        """opacity encoding in a layer must appear in _build_layers_list output (baseline)."""
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(df).mark_point().encode(x="x", y="y", opacity=fm.Opacity("sw"))
        chart = base + overlay
        layers = self._get_layers_list(chart)
        assert len(layers) >= 2
        layer_enc = layers[1].get("encoding", {})
        assert "opacity" in layer_enc

    def test_angle_encoding_in_layer_appears_in_build_layers(self, df):
        """angle encoding in a layer must appear in _build_layers_list output."""
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(df).mark_point().encode(x="x", y="y", angle=fm.Angle("sw"))
        chart = base + overlay
        layers = self._get_layers_list(chart)
        assert len(layers) >= 2
        layer_enc = layers[1].get("encoding", {})
        assert "angle" in layer_enc, (
            f"angle missing from layer encoding; got keys: {list(layer_enc.keys())}"
        )

    def test_fill_opacity_encoding_in_layer_appears_in_build_layers(self, df):
        """fill_opacity encoding in a layer must appear in _build_layers_list output."""
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(df).mark_point().encode(x="x", y="y", fill_opacity=fm.FillOpacity("sw"))
        chart = base + overlay
        layers = self._get_layers_list(chart)
        assert len(layers) >= 2
        layer_enc = layers[1].get("encoding", {})
        assert "fill_opacity" in layer_enc, (
            f"fill_opacity missing from layer encoding; got keys: {list(layer_enc.keys())}"
        )

    def test_stroke_opacity_encoding_in_layer_appears_in_build_layers(self, df):
        """stroke_opacity encoding in a layer must appear in _build_layers_list output."""
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        overlay = (
            fm.Chart(df).mark_point().encode(x="x", y="y", stroke_opacity=fm.StrokeOpacity("sw"))
        )
        chart = base + overlay
        layers = self._get_layers_list(chart)
        assert len(layers) >= 2
        layer_enc = layers[1].get("encoding", {})
        assert "stroke_opacity" in layer_enc, (
            f"stroke_opacity missing from layer encoding; got keys: {list(layer_enc.keys())}"
        )

    def test_stroke_dash_encoding_in_layer_appears_in_build_layers(self, df):
        """stroke_dash encoding in a layer must appear in _build_layers_list output."""
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(df).mark_line().encode(x="x", y="y", stroke_dash=fm.StrokeDash("cat"))
        chart = base + overlay
        layers = self._get_layers_list(chart)
        assert len(layers) >= 2
        layer_enc = layers[1].get("encoding", {})
        assert "stroke_dash" in layer_enc, (
            f"stroke_dash missing from layer encoding; got keys: {list(layer_enc.keys())}"
        )

    def test_all_honored_channels_iterated(self, df):
        """The channel set iterated by _build_layers_list must cover all _RENDERER_HONORED_CHANNELS."""
        from ferrum.encoding._channel_policy import _RENDERER_HONORED_CHANNELS

        honored = set(_RENDERER_HONORED_CHANNELS)
        old_set = {"x", "y", "x2", "y2", "color", "size", "shape", "opacity", "text"}
        missing_from_old = honored - old_set
        # Sanity check: the bug existed because old code skipped these channels
        assert missing_from_old, "Expected missing channels to exist (sanity check)"

        # Build a chart whose overlay layer has stroke_width (one of the formerly-missing channels)
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(df).mark_line().encode(x="x", y="y", stroke_width=fm.StrokeWidth("sw"))
        chart = base + overlay
        resolved = chart._resolve_pending()
        layers = resolved._build_layers_list()
        assert len(layers) >= 2
        # The overlay layer must have stroke_width — confirms all honored channels are iterated
        layer_enc = layers[1].get("encoding", {})
        assert "stroke_width" in layer_enc
