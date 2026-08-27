"""Regression tests for design-review finding P1 (S5): silently-discarded
encoding channels.

Spec: .claude/output/specs/2026-08-27-findings-remediation-design.md §4-P1, §6
Decision: .claude/output/decisions/2026-08-27-design-review-findings-decision.md (P1)

``encode()`` is a total function over ``ferrum.encoding._channel_class_map()``:
every channel falls into exactly one of five disjoint buckets declared in
``ferrum.chart`` --

- ``_RENDERER_HONORED_CHANNELS`` -- becomes its own ``EncodingSpec``.
- ``_ALIAS_CHANNELS`` -- redirects to another channel or to mark-style kwargs
  (``fill``/``stroke`` -> ``color``; ``detail`` -> ``mark_style.detail``).
- ``_WARN_CHANNELS`` -- accepted, ``warn_once``, absent from the resulting
  spec and rendered output -- never reaches an ``EncodingSpec`` or a Rust
  ``Encoding`` field (``x_error``, ``y_error``, ``x_error2``, ``y_error2``,
  ``tooltip_field``).
- ``_POLAR_CHANNELS`` -- remapped to x/y under ``CoordPolar``; ``warn_once``
  without it.
- ``_FACET_CHANNELS`` -- routed through ``resolved._facet``, never through
  the encoding-warn path.

This file covers the bucket-partition invariant (§6) and the newly-fixed
behaviors: ``key`` promoted to honored (both chart-level and layered paths),
``detail``'s mark-conditional warn (chart-level and per-layer), the
``x_error*``/``tooltip_field`` warn-and-drop, and the polar without-CoordPolar
warn -- on both the chart-level and layered paths (the layered
``_build_layers_list`` safety net was added 2026-08-27 to mirror the
chart-level one exactly, per spec-review finding).
"""

from __future__ import annotations

import json
import re
import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum._warn import reset_warnings


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [3.0, 4.0, 5.0],
            "g": ["a", "b", "c"],
        }
    )


def _user_warnings(caught, *, contains: str) -> list:
    return [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and contains in str(w.message).lower()
    ]


def _circle_fills(svg: str) -> list[str]:
    """Extract the ``fill`` attribute of every ``<circle>`` element in *svg*.

    Used to assert on the ACTUALLY RENDERED per-category color, not the
    wire-level ``EncodingSpec`` dict -- a channel can round-trip correctly
    through ``_build_layers_list``'s serialization-time alias while still
    never rendering distinct colors, if an earlier merge-time seam (e.g.
    ``composition._promote_layer_color``) reads the raw, un-aliased
    encoding and never builds a chart-level color scale for it. See
    ``TestLayeredFillStrokeAlias``.
    """
    fills = []
    for circle in re.findall(r"<circle[^>]*>", svg):
        m = re.search(r'fill="([^"]+)"', circle)
        if m:
            fills.append(m.group(1))
    return fills


# ---------------------------------------------------------------------------
# §6 -- bucket partition invariant
# ---------------------------------------------------------------------------


class TestBucketPartition:
    """ALIAS ∪ POLAR ∪ FACET ∪ RENDERER_HONORED ∪ WARN == keys(_channel_class_map())."""

    def _buckets(self):
        from ferrum.chart import (
            _ALIAS_CHANNELS,
            _FACET_CHANNELS,
            _POLAR_CHANNELS,
            _RENDERER_HONORED_CHANNELS,
            _WARN_CHANNELS,
        )

        return {
            "RENDERER_HONORED": frozenset(_RENDERER_HONORED_CHANNELS),
            "ALIAS": frozenset(_ALIAS_CHANNELS),
            "WARN": frozenset(_WARN_CHANNELS),
            "POLAR": frozenset(_POLAR_CHANNELS),
            "FACET": frozenset(_FACET_CHANNELS),
        }

    def test_union_covers_every_channel(self):
        from ferrum.encoding import _channel_class_map

        buckets = self._buckets()
        union = frozenset().union(*buckets.values())
        all_channels = frozenset(_channel_class_map())
        missing = all_channels - union
        extra = union - all_channels
        assert not missing, f"Channels present in _channel_class_map() but in no bucket: {missing}"
        assert not extra, f"Bucket entries that are not real channels: {extra}"

    def test_buckets_are_pairwise_disjoint(self):
        buckets = self._buckets()
        names = list(buckets)
        for i, name_a in enumerate(names):
            for name_b in names[i + 1 :]:
                overlap = buckets[name_a] & buckets[name_b]
                assert not overlap, f"{name_a} and {name_b} overlap on {overlap}"

    def test_partition_size_equals_union_size(self):
        """A structural cross-check: disjointness + coverage implies sum(sizes) == |union|."""
        buckets = self._buckets()
        total = sum(len(b) for b in buckets.values())
        union = frozenset().union(*buckets.values())
        assert total == len(union)

    def test_key_is_renderer_honored_not_alias(self):
        from ferrum.chart import _ALIAS_CHANNELS, _RENDERER_HONORED_CHANNELS

        assert "key" in _RENDERER_HONORED_CHANNELS
        assert "key" not in _ALIAS_CHANNELS

    def test_x_error_family_is_warn_not_alias(self):
        from ferrum.chart import _ALIAS_CHANNELS, _WARN_CHANNELS

        for ch in ("x_error", "y_error", "x_error2", "y_error2"):
            assert ch in _WARN_CHANNELS, f"{ch} must be in the WARN bucket"
            assert ch not in _ALIAS_CHANNELS

    def test_tooltip_field_is_warn(self):
        from ferrum.chart import _WARN_CHANNELS

        assert "tooltip_field" in _WARN_CHANNELS


# ---------------------------------------------------------------------------
# key: promoted from silent-drop to renderer-honored (chart-level + layered)
# ---------------------------------------------------------------------------


class TestKeyChannelHonored:
    """``key`` is honored in the narrow, literal sense the RENDERER_HONORED
    bucket promises: it gets its own ``EncodingSpec`` and reaches the scene
    graph (``MarkBatch.keys``) as mark identity, for interactive/animated
    runtimes -- on both the chart-level and layered paths, in both static
    and interactive scene JSON. It is NOT visually rendered by anything
    today: no test in this class asserts an SVG-visible effect, because
    there isn't one (quality-review finding, cycle 2 -- an earlier version
    of this batch's docs claimed `key` "renders end-to-end"; corrected to
    "carried into the scene graph ... no renderer consumes it visually
    yet" in CLAUDE.md, chart.py, ferrum-spec.md, and the archaeology doc).
    ``test_key_no_visual_effect_on_static_svg`` pins that absence directly.
    """

    def test_key_round_trips_to_dict(self):
        chart = fm.Chart(_df()).mark_point().encode(x="x", y="y", key="g")
        d = chart.to_dict()
        assert d["encoding"].get("key") == {"field": "g"}

    def test_key_reaches_scene_mark_batch_keys(self):
        """key= must reach the SceneGraph's MarkBatch.keys, not just the
        declaration spec -- this is a scene-graph-presence claim, not a
        visual-rendering claim (see class docstring).

        ``ferrum._scene._render_scene`` returns the actual SceneGraph JSON
        (distinct from ``to_dict()``'s spec view); Rust's ``extract_keys``
        (scene_build.rs) populates ``MarkBatch.keys`` from the ``key``
        encoding. Before promotion, ``key`` never reached ``ChartSpec`` at
        all, so this field was always absent.
        """
        from ferrum._scene import _render_scene

        chart = fm.Chart(_df()).mark_point().encode(x="x", y="y", key="g")
        scene_json, _packed = _render_scene(chart)
        parsed = json.loads(scene_json)
        marks = parsed["panels"][0]["marks"]
        assert marks[0].get("keys") == ["a", "b", "c"]

    def test_key_absent_when_not_encoded(self):
        from ferrum._scene import _render_scene

        chart = fm.Chart(_df()).mark_point().encode(x="x", y="y")
        scene_json, _packed = _render_scene(chart)
        parsed = json.loads(scene_json)
        marks = parsed["panels"][0]["marks"]
        assert marks[0].get("keys") is None

    def test_key_appears_in_layered_encoding(self):
        """Layered path: _build_layers_list iterates _RENDERER_HONORED_CHANNELS
        generically, so promoting `key` fixes both paths at once (mirrors the
        existing test_bug_hunt_pyo3_audit.py:208 W4 coverage pattern)."""
        base = fm.Chart(_df()).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(_df()).mark_line().encode(x="x", y="y", key="g")
        chart = base + overlay
        resolved = chart._resolve_pending()
        layers = resolved._build_layers_list()
        assert len(layers) >= 2
        layer_enc = layers[1].get("encoding", {})
        assert layer_enc.get("key") == {"field": "g"}

    def test_key_no_warning_emitted(self):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = fm.Chart(_df()).mark_point().encode(x="x", y="y", key="g").to_svg()
        assert "<svg" in svg
        assert not _user_warnings(caught, contains="'key'")

    def test_key_no_visual_effect_on_static_svg(self):
        """Pins the honest contract directly: no renderer consumes
        MarkBatch.keys today, so static SVG output is byte-identical with
        and without key= (quality-review finding, cycle 2 -- this is the
        exact check the reviewer used to disprove the "renders end-to-end"
        doc claim; encoded here as a regression pin rather than left as a
        one-off verification, so a future consumer's landing is what makes
        this test start failing -- the correct signal to update the docs
        back to a rendering claim)."""
        with_key = fm.Chart(_df()).mark_point().encode(x="x", y="y", key="g").to_svg()
        without_key = fm.Chart(_df()).mark_point().encode(x="x", y="y").to_svg()
        assert with_key == without_key


# ---------------------------------------------------------------------------
# detail: mark-conditional warn (chart-level + per-layer)
# ---------------------------------------------------------------------------


class TestDetailMarkConditionalWarn:
    def test_detail_on_point_warns_once(self):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = fm.Chart(_df()).mark_point().encode(x="x", y="y", detail="g").to_svg()
        assert "<svg" in svg
        detail_warnings = _user_warnings(caught, contains="detail")
        assert len(detail_warnings) == 1
        assert "mark_point" in str(detail_warnings[0].message)

    def test_detail_on_line_no_warning(self):
        """Pin: line consumes mark_style.group.detail (render/draw.rs)."""
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = fm.Chart(_df()).mark_line().encode(x="x", y="y", detail="g").to_svg()
        assert "<svg" in svg
        assert not _user_warnings(caught, contains="detail")

    def test_detail_on_area_no_warning(self):
        """Pin: area consumes mark_style.group.detail (render/marks/area.rs)."""
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = fm.Chart(_df()).mark_area().encode(x="x", y="y", detail="g").to_svg()
        assert "<svg" in svg
        assert not _user_warnings(caught, contains="detail")

    def test_detail_warn_fires_only_once_per_context(self):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fm.Chart(_df()).mark_point().encode(x="x", y="y", detail="g").to_svg()
            fm.Chart(_df()).mark_bar().encode(x="x", y="y", detail="g").to_svg()
        assert len(_user_warnings(caught, contains="detail")) == 1

    def test_detail_still_aliases_into_mark_style_even_when_warned(self):
        """The channel is still routed to mark_style.detail; the warning only
        documents that no builder currently reads it for this mark."""
        resolved = fm.Chart(_df()).mark_point().encode(x="x", y="y", detail="g")._resolve_pending()
        spec_dict = json.loads(resolved.to_spec().to_json())
        assert spec_dict.get("mark_style", {}).get("detail") == "g"


class TestPerLayerDetailAlias:
    """Before this fix, a layer's own `detail` never reached apply_channel_aliases
    at all (only the chart-level pass ran), so it was silently dropped --
    no color/mark_style routing, no warning."""

    def test_per_layer_detail_on_line_routes_to_mark_style_no_warning(self):
        reset_warnings()
        base = fm.Chart(_df()).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(_df()).mark_line().encode(x="x", y="y", detail="g")
        chart = base + overlay
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = chart._resolve_pending()
            layers = resolved._build_layers_list()
        assert not _user_warnings(caught, contains="detail")
        assert layers[1].get("mark_style", {}).get("detail") == "g"

    def test_per_layer_detail_on_point_warns_and_still_routes(self):
        reset_warnings()
        base = fm.Chart(_df()).mark_line().encode(x="x", y="y")
        overlay = fm.Chart(_df()).mark_point().encode(x="x", y="y", detail="g")
        chart = base + overlay
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = chart._resolve_pending()
            layers = resolved._build_layers_list()
        detail_warnings = _user_warnings(caught, contains="detail")
        assert len(detail_warnings) == 1
        assert "mark_point" in str(detail_warnings[0].message)
        assert layers[1].get("mark_style", {}).get("detail") == "g"

    def test_layer_without_detail_encoding_untouched(self):
        base = fm.Chart(_df()).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(_df()).mark_line().encode(x="x", y="y")
        chart = base + overlay
        resolved = chart._resolve_pending()
        layers = resolved._build_layers_list()
        assert "mark_style" not in layers[1] or "detail" not in layers[1].get("mark_style", {})


# ---------------------------------------------------------------------------
# x_error / y_error / x_error2 / y_error2: warn-and-drop
# ---------------------------------------------------------------------------


class TestErrorExtentChannelsWarnAndDrop:
    @pytest.mark.parametrize("channel", ["x_error", "y_error", "x_error2", "y_error2"])
    def test_warns_once_and_absent_from_output(self, channel):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = fm.Chart(_df()).mark_point().encode(x="x", y="y", **{channel: "x"})
            d = chart.to_dict()
        warns = _user_warnings(caught, contains=f"'{channel}'")
        assert len(warns) == 1, f"expected exactly one warning for {channel}; got {caught}"
        assert d["encoding"].get(channel) is None, (
            f"{channel} must be absent from to_dict()['encoding']; got {d['encoding']}"
        )

    def test_warns_exactly_once_across_multiple_renders(self):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fm.Chart(_df()).mark_point().encode(x="x", y="y", x_error="x").to_svg()
            fm.Chart(_df()).mark_point().encode(x="x", y="y", x_error="x").to_svg()
        assert len(_user_warnings(caught, contains="'x_error'")) == 1


# ---------------------------------------------------------------------------
# tooltip_field: deliberately bucketed as WARN (was an accidental safety-net hit)
# ---------------------------------------------------------------------------


class TestTooltipFieldTopLevelWarns:
    def test_top_level_tooltip_field_warns_once_and_absent(self):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = fm.Chart(_df()).mark_point().encode(x="x", y="y", tooltip_field="g")
            d = chart.to_dict()
        warns = _user_warnings(caught, contains="'tooltip_field'")
        assert len(warns) == 1
        assert d["encoding"].get("tooltip_field") is None

    def test_tooltip_field_inside_tooltip_multi_field_unaffected(self):
        """TooltipField's documented use -- inside Tooltip(*fields) -- is a
        distinct code path (multi-field tooltip serialization) and must not
        warn."""
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = (
                fm.Chart(_df())
                .mark_point()
                .encode(x="x", y="y", tooltip=fm.Tooltip("x", fm.TooltipField("g", title="G")))
            )
            d = chart.to_dict()
        assert not _user_warnings(caught, contains="tooltip_field")
        assert d["encoding"].get("tooltip_fields") is not None


# ---------------------------------------------------------------------------
# theta/radius/theta2/radius2: warn without CoordPolar, unchanged with it
# ---------------------------------------------------------------------------


class TestPolarChannelsWarnWithoutCoordPolar:
    @pytest.mark.parametrize("channel", ["theta", "radius"])
    def test_warns_once_and_absent_without_coord_polar(self, channel):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = fm.Chart(_df()).mark_point().encode(x="x", y="y", **{channel: "x"})
            d = chart.to_dict()
        warns = _user_warnings(caught, contains=f"'{channel}'")
        assert len(warns) == 1
        assert d["encoding"].get(channel) is None

    def test_theta_with_coord_polar_remaps_without_warning(self):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = (
                fm.Chart(_df())
                .mark_point()
                .encode(theta=fm.Theta("y"))
                .coord(fm.CoordPolar(theta="x"))
            )
            d = chart.to_dict()
        assert not _user_warnings(caught, contains="'theta'")
        assert d["encoding"].get("theta") is None
        assert d["encoding"].get("x") == {"field": "y"}

    def test_radius_with_coord_polar_remaps_without_warning(self):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = (
                fm.Chart(_df())
                .mark_point()
                .encode(theta=fm.Theta("y"), radius=fm.Radius("x"))
                .coord(fm.CoordPolar(theta="x"))
            )
            d = chart.to_dict()
        assert not _user_warnings(caught, contains="'radius'")
        assert d["encoding"].get("radius") is None
        assert d["encoding"].get("y") == {"field": "x"}


# ---------------------------------------------------------------------------
# Layered-path safety net parity (spec-review finding, 2026-08-27): the
# per-layer drop loop in `_build_layers_list` must apply the identical
# bucket dispositions as the chart-level `_build_encoding_specs` -- fill/
# stroke get the alias treatment (including the conflict warning), and
# WARN/un-remapped-POLAR channels warn once and are absent, per layer.
# ---------------------------------------------------------------------------


def _layer_encoding(chart: fm.Chart, layer_idx: int = 1) -> dict:
    resolved = chart._resolve_pending()
    layers = resolved._build_layers_list()
    assert len(layers) > layer_idx
    return layers[layer_idx].get("encoding", {})


class TestLayeredFillStrokeAlias:
    """A layer's own fill/stroke must render distinct per-category colors +
    a legend, identically to the equivalent color= control.

    This is a render-level check, not a wire-dict check: whether a layer's
    color is ever promoted to a chart-level color scale is decided at
    MERGE time by ``composition._promote_layer_color`` -- which runs before
    ``_build_layers_list``'s serialization-time alias -- so a correct
    ``layer_enc["color"]`` wire entry does not by itself prove the chart
    renders in color. `_promote_layer_color` reading the raw, un-aliased
    ``layer.encoding.get("color")`` was exactly the bug the wire-only
    version of this test missed: `fill=`/`stroke=` on an overlay layer
    rendered byte-identically to no color channel at all (single theme-
    default fill, no legend) even though the wire dict looked correct.
    """

    def test_per_layer_fill_renders_distinct_colors_with_legend(self):
        reset_warnings()
        df = _df()
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        control_svg = (base + fm.Chart(df).mark_point().encode(x="x", y="y", color="g")).to_svg()
        fill_svg = (
            base + fm.Chart(df).mark_point().encode(x="x", y="y", fill=fm.Fill("g"))
        ).to_svg()

        control_fills = sorted(set(_circle_fills(control_svg)))
        fill_fills = sorted(set(_circle_fills(fill_svg)))
        assert len(control_fills) >= 3, (
            f"color= control must render >=3 distinct circle fills; got {control_fills}"
        )
        assert fill_fills == control_fills, (
            f"fill= must render the identical per-category fill set as the color= "
            f"control; got {fill_fills}, expected {control_fills}"
        )
        for label in ("a", "b", "c"):
            assert f">{label}<" in fill_svg, (
                f"expected legend label {label!r} in the fill= SVG (legend must "
                "render, not just the wire dict)"
            )

    def test_per_layer_stroke_renders_distinct_colors_with_legend(self):
        reset_warnings()
        df = _df()
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        control_svg = (base + fm.Chart(df).mark_point().encode(x="x", y="y", color="g")).to_svg()
        stroke_svg = (
            base + fm.Chart(df).mark_point().encode(x="x", y="y", stroke=fm.Stroke("g"))
        ).to_svg()

        control_fills = sorted(set(_circle_fills(control_svg)))
        stroke_fills = sorted(set(_circle_fills(stroke_svg)))
        assert stroke_fills == control_fills, (
            f"stroke= must render the identical per-category fill set as the color= "
            f"control; got {stroke_fills}, expected {control_fills}"
        )
        for label in ("a", "b", "c"):
            assert f">{label}<" in stroke_svg, (
                f"expected legend label {label!r} in the stroke= SVG (legend must "
                "render, not just the wire dict)"
            )

    def test_per_layer_stroke_dropped_by_color_warns_once(self):
        """Mirrors the chart-level stroke-dropped-by-color conflict warning."""
        reset_warnings()
        base = fm.Chart(_df()).mark_point().encode(x="x", y="y")
        overlay = (
            fm.Chart(_df()).mark_point().encode(x="x", y="y", color="g", stroke=fm.Stroke("g"))
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layer_enc = _layer_encoding(base + overlay)
        stroke_warnings = _user_warnings(caught, contains="stroke=...")
        assert len(stroke_warnings) == 1
        assert layer_enc.get("color") == {"field": "g"}
        assert "stroke" not in layer_enc


class TestLayeredWarnBucketParity:
    @pytest.mark.parametrize("channel", ["x_error", "y_error", "x_error2", "y_error2"])
    def test_per_layer_error_extent_warns_once_and_absent(self, channel):
        reset_warnings()
        base = fm.Chart(_df()).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(_df()).mark_point().encode(x="x", y="y", **{channel: "x"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layer_enc = _layer_encoding(base + overlay)
        warns = _user_warnings(caught, contains=f"'{channel}'")
        assert len(warns) == 1, f"expected exactly one warning for {channel}; got {caught}"
        assert channel not in layer_enc

    def test_per_layer_tooltip_field_warns_once_and_absent(self):
        reset_warnings()
        base = fm.Chart(_df()).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(_df()).mark_point().encode(x="x", y="y", tooltip_field="g")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layer_enc = _layer_encoding(base + overlay)
        warns = _user_warnings(caught, contains="'tooltip_field'")
        assert len(warns) == 1
        assert "tooltip_field" not in layer_enc


class TestLayeredPolarBucketParity:
    @pytest.mark.parametrize("channel", ["theta", "radius", "theta2", "radius2"])
    def test_per_layer_polar_channel_warns_once_and_absent(self, channel):
        """The layered path has no per-layer CoordPolar remap, so a layer's
        own theta/radius never renders -- warn rather than silently drop it,
        matching the chart-level without-CoordPolar disposition."""
        reset_warnings()
        base = fm.Chart(_df()).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(_df()).mark_point().encode(x="x", y="y", **{channel: "x"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layer_enc = _layer_encoding(base + overlay)
        warns = _user_warnings(caught, contains=f"'{channel}'")
        assert len(warns) == 1
        assert channel not in layer_enc

    def test_per_layer_polar_channel_warns_even_with_coord_polar_set(self):
        """ferrum-spec.md §3.2's Polar bullet (2026-08-27 note): only the
        CHART-LEVEL theta/radius participates in the CoordPolar remap; a
        LAYER's own polar channel warns and is never rendered regardless of
        whether the chart's coord is CoordPolar, because there is no
        per-layer remap mechanism at all (quality-review finding --
        the note previously read as though CoordPolar alone determined the
        disposition, contradicting this actual behavior)."""
        reset_warnings()
        df = _df()
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        overlay = fm.Chart(df).mark_point().encode(x="x", y="y", theta="x")
        chart = (base + overlay).coord(fm.CoordPolar(theta="x"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        warns = _user_warnings(caught, contains="'theta'")
        assert len(warns) == 1, f"expected exactly one 'theta' warning; got {caught}"
        assert "<svg" in svg


class TestLayerStringEncodings:
    """The public ``fm.Layer(encoding={...})`` API legitimately accepts plain
    strings as encoding values (its own docstring example is
    ``encoding={"x": "x", "y": "y"}``) -- a shape distinct from the
    ``ChannelBase`` instances ``Chart.encode()`` always produces on the
    ``Chart + Chart`` path every other test in this file exercises. The
    alias authority (``apply_channel_aliases``/``alias_detail_channel``) and
    the bucket-partition safety net (``_warn_unbucketed_channels``) must
    handle both shapes identically (quality-review finding: a string
    ``fill``/``stroke`` on a ``Layer`` used to raise ``AttributeError`` from
    the ``warn_drop`` conflict branch -- a chart that rendered before this
    task started routing layer encodings through the alias authority now
    crashed; a string ``detail`` was silently dropped with zero routing and
    zero warning, the P1 defect class itself, on the one path where string
    encodings are the documented normal form).
    """

    def _layered_chart(self, layer_encoding: dict, *, layer_mark: str = "point") -> fm.Chart:
        df = _df()
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        return base.layer(fm.Layer(mark=layer_mark, encoding=layer_encoding))

    def test_string_color_and_stroke_conflict_renders_and_warns_once(self):
        """Regression for the AttributeError crash: `fm.Layer(encoding={
        "color": "g", "stroke": "g"})` used to raise instead of rendering."""
        reset_warnings()
        chart = self._layered_chart({"x": "x", "y": "y", "color": "g", "stroke": "g"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        assert "<svg" in svg
        stroke_warnings = _user_warnings(caught, contains="stroke=...")
        assert len(stroke_warnings) == 1, f"expected exactly one warning; got {caught}"

    def test_string_fill_alone_renders_without_crash_or_warning(self):
        reset_warnings()
        chart = self._layered_chart({"x": "x", "y": "y", "fill": "g"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        assert "<svg" in svg
        assert not _user_warnings(caught, contains="fill")

    def test_string_detail_on_non_consuming_mark_warns_and_routes(self):
        """String-valued detail must reach mark_style.detail AND warn on a
        mark whose Rust builder does not read it -- both were silently
        dropped pre-fix (getattr(detail_ch, "field", None) returns None for
        a string)."""
        reset_warnings()
        chart = self._layered_chart({"x": "x", "y": "y", "detail": "g"}, layer_mark="point")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = chart._resolve_pending()
            layers = resolved._build_layers_list()
        detail_warnings = _user_warnings(caught, contains="detail")
        assert len(detail_warnings) == 1
        assert "mark_point" in str(detail_warnings[0].message)
        assert layers[-1].get("mark_style", {}).get("detail") == "g"

    def test_string_detail_on_consuming_mark_routes_without_warning(self):
        reset_warnings()
        chart = self._layered_chart({"x": "x", "y": "y", "detail": "g"}, layer_mark="line")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = chart._resolve_pending()
            layers = resolved._build_layers_list()
        assert not _user_warnings(caught, contains="detail")
        assert layers[-1].get("mark_style", {}).get("detail") == "g"

    def test_string_x_error_warns_once_and_absent(self):
        """String-valued WARN-bucket channels must also be caught by the
        unified safety net (`_warn_unbucketed_channels`), not just
        ChannelBase-valued ones."""
        reset_warnings()
        chart = self._layered_chart({"x": "x", "y": "y", "x_error": "x"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = chart._resolve_pending()
            layers = resolved._build_layers_list()
        warns = _user_warnings(caught, contains="'x_error'")
        assert len(warns) == 1
        assert "x_error" not in layers[-1].get("encoding", {})
