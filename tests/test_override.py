"""Tests for Chart.override(), Chart.configure_*(), Chart.configure(),
and Chart.__add__ dispatch for Configure / Annotate / structural types.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
import polars as pl

import ferrum as fm
from ferrum.configure import (
    AxisConfig,
    ColorConfig,
    Configure,
    GridConfig,
    LegendConfig,
    PaddingConfig,
    TitleConfig,
)
from ferrum.annotation.container import Annotate
from ferrum.annotation.primitives import AnnotationText
from ferrum.exceptions import FerrumOverrideError
from ferrum.marks.base import _VALID_MARK_KWARGS
from ferrum.structural import BreakAxis, Inset, SecondaryY
from ferrum import _override_apply as oa


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_chart():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# Chart.override()
# ---------------------------------------------------------------------------


class TestOverride:
    def test_stores_kwarg(self, base_chart):
        c = base_chart.override(x_axis_label_angle=-45)
        assert c._overrides == {"x_axis_label_angle": -45}

    def test_multiple_calls_merge(self, base_chart):
        c = base_chart.override(x_axis_label_angle=-45).override(width=600)
        assert c._overrides["x_axis_label_angle"] == -45
        assert c._overrides["width"] == 600

    def test_later_call_wins_on_conflict(self, base_chart):
        c = base_chart.override(x_axis_label_angle=-45).override(x_axis_label_angle=0)
        assert c._overrides["x_axis_label_angle"] == 0

    def test_does_not_mutate_original(self, base_chart):
        _ = base_chart.override(x_axis_label_angle=-45)
        assert base_chart._overrides == {}

    def test_returns_new_chart(self, base_chart):
        c = base_chart.override(x_axis_label_angle=-45)
        assert c is not base_chart

    def test_empty_override_is_noop(self, base_chart):
        c = base_chart.override()
        assert c._overrides == {}
        assert c is not base_chart


# ---------------------------------------------------------------------------
# Chart.configure_axis()
# ---------------------------------------------------------------------------


class TestConfigureAxis:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_axis(label_angle=-45)
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert isinstance(cfg, Configure)
        assert cfg.axis is not None
        assert cfg.axis.label_angle == -45

    def test_does_not_mutate_original(self, base_chart):
        _ = base_chart.configure_axis(label_angle=-45)
        assert base_chart._configure == []

    def test_returns_new_chart(self, base_chart):
        c = base_chart.configure_axis(label_angle=-45)
        assert c is not base_chart

    def test_multiple_calls_accumulate(self, base_chart):
        c = base_chart.configure_axis(label_angle=-45).configure_axis(grid=True)
        assert len(c._configure) == 2


# ---------------------------------------------------------------------------
# Chart.configure_legend()
# ---------------------------------------------------------------------------


class TestConfigureLegend:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_legend(orient="bottom", columns=3)
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.legend is not None
        assert cfg.legend.orient == "bottom"
        assert cfg.legend.columns == 3

    def test_does_not_mutate_original(self, base_chart):
        _ = base_chart.configure_legend(orient="top")
        assert base_chart._configure == []


# ---------------------------------------------------------------------------
# Chart.configure_title()
# ---------------------------------------------------------------------------


class TestConfigureTitle:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_title(font_size=18, anchor="start")
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.title is not None
        assert cfg.title.font_size == 18
        assert cfg.title.anchor == "start"


# ---------------------------------------------------------------------------
# Chart.configure_grid()
# ---------------------------------------------------------------------------


class TestConfigureGrid:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_grid(color="#eee", width=0.5)
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.grid is not None
        assert cfg.grid.color == "#eee"
        assert cfg.grid.width == 0.5


# ---------------------------------------------------------------------------
# Chart.configure_padding()
# ---------------------------------------------------------------------------


class TestConfigurePadding:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_padding(top=20, bottom=10)
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.padding is not None
        assert cfg.padding.top == 20
        assert cfg.padding.bottom == 10


# ---------------------------------------------------------------------------
# Chart.configure_color()
# ---------------------------------------------------------------------------


class TestConfigureColor:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_color(scheme="tableau10")
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.color is not None
        assert cfg.color.scheme == "tableau10"


# ---------------------------------------------------------------------------
# Chart.configure() — typed config objects
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_accepts_typed_objects(self, base_chart):
        c = base_chart.configure(
            axis=AxisConfig(label_angle=-45),
            legend=LegendConfig(orient="bottom"),
        )
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.axis.label_angle == -45
        assert cfg.legend.orient == "bottom"

    def test_does_not_mutate_original(self, base_chart):
        _ = base_chart.configure(axis=AxisConfig(label_angle=-45))
        assert base_chart._configure == []

    def test_returns_new_chart(self, base_chart):
        c = base_chart.configure(axis=AxisConfig())
        assert c is not base_chart

    def test_empty_configure_is_noop_payload(self, base_chart):
        # configure() with no args still appends one Configure() (all-None)
        c = base_chart.configure()
        assert len(c._configure) == 1
        assert c._configure[0] == Configure()

    def test_axis_x_y_y2_routed(self, base_chart):
        c = base_chart.configure(
            axis_x=AxisConfig(label_angle=0),
            axis_y=AxisConfig(label_angle=-30),
        )
        cfg = c._configure[0]
        assert cfg.axis_x is not None
        assert cfg.axis_y is not None
        assert cfg.axis is None


# ---------------------------------------------------------------------------
# Chart.__add__ dispatch
# ---------------------------------------------------------------------------


class TestAddDispatch:
    def test_configure_dispatch(self, base_chart):
        cfg = Configure(axis=AxisConfig(label_angle=-45))
        c = base_chart + cfg
        assert len(c._configure) == 1
        assert c._configure[0] is cfg

    def test_configure_dispatch_does_not_mutate_original(self, base_chart):
        cfg = Configure(axis=AxisConfig(label_angle=-45))
        _ = base_chart + cfg
        assert base_chart._configure == []

    def test_annotate_container_dispatch(self, base_chart):
        import ferrum.annotation as ann

        container = Annotate([ann.text(1.0, 2.0, "label")])
        c = base_chart + container
        assert len(c._annotations) == 1
        assert c._annotations[0] is container

    def test_annotate_dispatch_does_not_mutate_original(self, base_chart):
        import ferrum.annotation as ann

        container = Annotate([ann.text(1.0, 2.0, "label")])
        _ = base_chart + container
        assert base_chart._annotations == []

    def test_bare_annotation_primitive_wraps_in_annotate(self, base_chart):
        import ferrum.annotation as ann

        primitive = ann.text(1.0, 2.0, "hi")
        c = base_chart + primitive
        assert len(c._annotations) == 1
        assert isinstance(c._annotations[0], Annotate)
        assert len(c._annotations[0].items) == 1
        assert isinstance(c._annotations[0].items[0], AnnotationText)

    def test_bare_primitive_does_not_mutate_original(self, base_chart):
        import ferrum.annotation as ann

        _ = base_chart + ann.text(1.0, 2.0, "hi")
        assert base_chart._annotations == []

    def test_secondary_y_dispatch(self, base_chart):
        """GH #52: SecondaryY desugars to an appended independent-y layer,
        not a ``_structural`` entry (that silo is reserved for BreakAxis/Inset)."""
        sy = SecondaryY(field="x")
        c = base_chart + sy
        assert c._structural == []
        assert c._layers is not None
        assert len(c._layers) == 2
        secondary_layer = c._layers[-1]
        assert secondary_layer.independent_y is True
        assert secondary_layer.encoding["y"].field == "x"

    def test_break_axis_dispatch(self, base_chart):
        ba = BreakAxis(axis="y", gap=(50, 100))
        c = base_chart + ba
        assert len(c._structural) == 1
        assert c._structural[0] is ba

    def test_inset_dispatch(self, base_chart):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        inset_chart = fm.Chart(df).mark_point().encode(x="x", y="y")
        inset = Inset(chart=inset_chart, bounds=(0.5, 0.5, 1.0, 1.0))
        c = base_chart + inset
        assert len(c._structural) == 1
        assert c._structural[0] is inset

    def test_structural_does_not_mutate_original(self, base_chart):
        ba = BreakAxis(axis="y", gap=(50, 100))
        _ = base_chart + ba
        assert base_chart._structural == []

    def test_secondary_y_does_not_mutate_original(self, base_chart):
        sy = SecondaryY(field="x")
        _ = base_chart + sy
        assert base_chart._layers is None

    def test_chart_plus_chart_still_works(self, base_chart):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [7, 8, 9]})
        other = fm.Chart(df).mark_line().encode(x="x", y="y")
        layered = base_chart + other
        assert isinstance(layered, fm.Chart)
        assert layered._layers is not None
        assert len(layered._layers) == 2

    def test_unsupported_type_returns_not_implemented(self, base_chart):
        result = base_chart.__add__("not_a_chart")
        assert result is NotImplemented

    def test_multiple_structural_accumulate(self, base_chart):
        ba1 = BreakAxis(axis="y", gap=(10, 20))
        ba2 = BreakAxis(axis="x", gap=(5, 15))
        c = base_chart + ba1 + ba2
        assert len(c._structural) == 2

    def test_multiple_configure_layers_via_add(self, base_chart):
        c1 = Configure(axis=AxisConfig(label_angle=-45))
        c2 = Configure(legend=LegendConfig(orient="top"))
        c = base_chart + c1 + c2
        assert len(c._configure) == 2


# ---------------------------------------------------------------------------
# _override_apply — registry, resolution, validation, payload (pure logic)
# ---------------------------------------------------------------------------


class TestOverrideRegistryParity:
    """The generated leaf sets must equal the live schemas they derive from."""

    @pytest.mark.parametrize(
        ("prefix", "config_cls"),
        [
            ("x_axis_", AxisConfig),
            ("y_axis_", AxisConfig),
            ("axis_", AxisConfig),
            ("legend_", LegendConfig),
            ("title_", TitleConfig),
            ("grid_", GridConfig),
            ("padding_", PaddingConfig),
            ("color_", ColorConfig),
        ],
    )
    def test_chart_config_leaves_match_dataclass_fields(self, prefix, config_cls):
        rule = next(r for r in oa._PREFIX_RULES if r.prefix == prefix)
        actual_fields = frozenset(f.name for f in fields(config_cls))
        expected = actual_fields
        if config_cls is AxisConfig:
            # ``x``/``y`` are deprecated no-op AxisConfig fields that
            # ``AxisConfig.to_dict()`` never emits and the wire schema does
            # not accept; the override registry excludes them too (see
            # ``_build_chart_config_rules``) so the deprecated spelling
            # refuses at the override boundary, not as an opaque wire error.
            # Verify that x/y are actually present in the dataclass fields,
            # so deleting them fails this test deliberately.
            assert {"x", "y"} <= actual_fields
            expected = expected - {"x", "y"}
        assert rule.valid_leaves == expected

    def test_mark_leaves_match_valid_mark_kwargs(self):
        rule = next(r for r in oa._PREFIX_RULES if r.prefix == "mark_")
        assert rule.valid_leaves == _VALID_MARK_KWARGS

    def test_scale_leaves_include_common_options_and_exclude_derived(self):
        rule = next(r for r in oa._PREFIX_RULES if r.prefix == "x_scale_")
        # Settable scale-spec leaves are present.
        assert {"domain", "range", "type", "scheme", "padding"} <= rule.valid_leaves
        # Derived getters are not settable leaves.
        assert "num_bins" not in rule.valid_leaves

    def test_coord_leaves_match_coord_dataclass_fields(self):
        rule = next(r for r in oa._PREFIX_RULES if r.prefix == "coord_")
        # Union of CoordCartesian / CoordFixed / CoordPolar / CoordGeo fields.
        assert {"clip", "expand", "xlim", "ylim", "ratio", "theta", "projection"} <= (
            rule.valid_leaves
        )


class TestOverrideResolve:
    def test_x_axis_longest_prefix_wins(self):
        r = oa.resolve("x_axis_label_angle")
        assert r is not None
        assert r.target is oa.Target.CHART_CONFIG
        assert r.location.target_key == "axis_x"
        assert r.location.leaf == "label_angle"

    def test_y_axis_tick_count(self):
        r = oa.resolve("y_axis_tick_count")
        assert r is not None
        assert r.location.target_key == "axis_y"
        assert r.location.leaf == "tick_count"

    def test_axis_grid(self):
        r = oa.resolve("axis_grid")
        assert r is not None
        assert r.location.target_key == "axis"
        assert r.location.leaf == "grid"

    def test_legend_orient(self):
        r = oa.resolve("legend_orient")
        assert r is not None
        assert r.location.target_key == "legend"
        assert r.location.leaf == "orient"

    def test_color_scheme(self):
        r = oa.resolve("color_scheme")
        assert r is not None
        assert r.location.target_key == "color"
        assert r.location.leaf == "scheme"

    def test_x_scale_domain(self):
        r = oa.resolve("x_scale_domain")
        assert r is not None
        assert r.target is oa.Target.ENCODING_SCALE
        assert r.location.target_key == "x"
        assert r.location.leaf == "domain"

    def test_mark_corner_radius(self):
        r = oa.resolve("mark_corner_radius")
        assert r is not None
        assert r.target is oa.Target.MARK
        assert r.location.target_key is None
        assert r.location.leaf == "corner_radius"

    def test_coord_clip(self):
        r = oa.resolve("coord_clip")
        assert r is not None
        assert r.target is oa.Target.COORD
        assert r.location.target_key == "coord"
        assert r.location.leaf == "clip"

    def test_width_height_properties(self):
        for key in ("width", "height"):
            r = oa.resolve(key)
            assert r is not None
            assert r.target is oa.Target.PROPERTIES
            assert r.location.leaf == key

    def test_x_axis_not_missplit_as_axis(self):
        # ``x_axis_grid_color`` is a real AxisConfig leaf reached via the
        # longest ``x_axis_`` prefix, not via ``axis_`` with leaf ``grid_color``.
        r = oa.resolve("x_axis_grid_color")
        assert r is not None
        assert r.location.target_key == "axis_x"
        assert r.location.leaf == "grid_color"

    def test_unknown_prefix_resolves_none(self):
        assert oa.resolve("bogus_thing") is None

    def test_known_prefix_bad_leaf_resolves_none(self):
        assert oa.resolve("mark_notathing") is None
        assert oa.resolve("x_axis_lable_angle") is None

    def test_deprecated_axis_xy_leaves_resolve_none(self):
        # RED-proof: ``AxisConfig.x``/``.y`` are dataclass fields that
        # ``AxisConfig.to_dict()`` never emits and the wire schema does not
        # accept (see ``_build_chart_config_rules``'s exclusion comment). If
        # the exclusion regresses (``x``/``y`` re-added to ``axis_leaves``),
        # these resolve to a leaf again and this assertion flips.
        assert oa.resolve("axis_x") is None
        assert oa.resolve("axis_y") is None

    def test_valid_axis_leaf_still_resolves(self):
        # Control: a real AxisConfig leaf under the general ``axis_`` prefix
        # is unaffected by the ``x``/``y`` exclusion.
        r = oa.resolve("axis_grid")
        assert r is not None
        assert r.location.target_key == "axis"
        assert r.location.leaf == "grid"

    def test_typed_equivalent_present_for_config_and_properties(self):
        assert oa.resolve("x_axis_label_angle").typed_equivalent == (
            ".configure_axis(label_angle=...)"
        )
        assert oa.resolve("width").typed_equivalent == ".properties(width=...)"

    def test_typed_equivalent_absent_for_escape_hatch_paths(self):
        assert oa.resolve("x_scale_domain").typed_equivalent is None
        assert oa.resolve("mark_corner_radius").typed_equivalent is None
        assert oa.resolve("coord_clip").typed_equivalent is None


class TestOverrideValidate:
    def test_unknown_prefix_raises(self):
        with pytest.raises(FerrumOverrideError, match="bogus_thing"):
            oa.validate({"bogus_thing": 1})

    def test_known_prefix_bad_leaf_raises(self):
        with pytest.raises(FerrumOverrideError, match="mark_notathing"):
            oa.validate({"mark_notathing": 1})

    def test_typo_suggests_closest_path(self):
        with pytest.raises(FerrumOverrideError) as excinfo:
            oa.validate({"x_axis_lable_angle": -45})
        assert "x_axis_label_angle" in str(excinfo.value)

    def test_per_channel_axis_grid_color_resolves_to_chart_config(self):
        # ``x_axis_grid_color`` targets the typed AxisConfig leaf ``grid_color``
        # (chart-config), NOT the opaque per-channel AxisSpec.extra map.  Valid.
        oa.validate({"x_axis_grid_color": "#eee"})

    def test_per_channel_only_key_without_config_equivalent_raises(self):
        # ``x_legend_glyph_dx`` is a genuinely per-channel-only key with no
        # LegendConfig equivalent; the per-channel axis/legend maps are excluded
        # in v1, so it must raise.
        with pytest.raises(FerrumOverrideError, match="x_legend_glyph_dx"):
            oa.validate({"x_legend_glyph_dx": 1})

    def test_valid_dict_does_not_raise(self):
        oa.validate(
            {"x_axis_label_angle": -45, "width": 900, "mark_size": 50, "color_scheme": "viridis"}
        )

    @pytest.mark.parametrize(
        "deprecated_path",
        ["axis_x", "axis_y", "x_axis_x", "x_axis_y", "y_axis_x", "y_axis_y"],
    )
    def test_deprecated_axis_xy_leaf_raises_naming_the_kwarg(self, deprecated_path):
        # All six deprecated axis-key spellings (3 prefixes × 2 deprecated leaves)
        # must refuse with a typed, kwarg-naming FerrumOverrideError that names
        # both the offending path AND points at the documented replacement
        # ``Chart.axis(x=False)`` or ``Chart.axis(y=False)``.
        with pytest.raises(FerrumOverrideError, match=rf"{deprecated_path}.*Chart\.axis"):
            oa.validate({deprecated_path: True})


class TestOverrideBuildPayload:
    def test_routes_each_target_into_its_piece(self):
        payload = oa.build_payload(
            {
                "x_axis_label_angle": -45,
                "width": 900,
                "mark_size": 50,
                "color_scheme": "viridis",
            }
        )
        assert payload.chart_config == {
            "axis_x": {"label_angle": -45},
            "color": {"scheme": "viridis"},
        }
        assert payload.properties == {"width": 900}
        assert payload.mark_style == {"size": 50}
        # Unused pieces are empty.
        assert payload.encoding == {}
        assert payload.coord == {}

    def test_encoding_scale_piece_shape(self):
        payload = oa.build_payload({"x_scale_domain": [0, 10]})
        assert payload.encoding == {"x": {"scale": {"domain": [0, 10]}}}
        assert payload.chart_config == {}

    def test_coord_piece_shape(self):
        payload = oa.build_payload({"coord_clip": False})
        assert payload.coord == {"clip": False}

    def test_deprecations_include_typed_equivalent_paths(self):
        payload = oa.build_payload(
            {
                "x_axis_label_angle": -45,
                "width": 900,
                "mark_size": 50,
                "color_scheme": "viridis",
            }
        )
        deprecated_paths = {p for p, _ in payload.deprecations}
        # config + properties paths have typed equivalents.
        assert "x_axis_label_angle" in deprecated_paths
        assert "width" in deprecated_paths
        assert "color_scheme" in deprecated_paths
        # mark style has no typed equivalent.
        assert "mark_size" not in deprecated_paths
        # The descriptor is carried alongside the path.
        assert ("width", ".properties(width=...)") in payload.deprecations

    def test_build_payload_validates_before_building(self):
        with pytest.raises(FerrumOverrideError, match="bogus_thing"):
            oa.build_payload({"x_axis_label_angle": -45, "bogus_thing": 1})

    @pytest.mark.parametrize(
        "deprecated_path",
        ["axis_x", "axis_y", "x_axis_x", "x_axis_y", "y_axis_x", "y_axis_y"],
    )
    def test_deprecated_axis_xy_leaf_raises_at_build_payload(self, deprecated_path):
        with pytest.raises(FerrumOverrideError, match=rf"{deprecated_path}.*Chart\.axis"):
            oa.build_payload({deprecated_path: True})

    def test_valid_axis_leaf_still_builds(self):
        # Control: a real AxisConfig leaf under the general ``axis_`` prefix
        # still builds a chart-config payload, unaffected by the ``x``/``y``
        # exclusion.
        payload = oa.build_payload({"axis_grid": False})
        assert payload.chart_config == {"axis": {"grid": False}}


# ---------------------------------------------------------------------------
# Render-level wiring (Chart.override takes effect in the rendered SVG)
#
# These are the load-bearing tests that close bug B4: they assert observable
# changes in to_svg() output, never the contents of _overrides.
# ---------------------------------------------------------------------------


import warnings
import xml.etree.ElementTree as ET

_SVG_NS = "{http://www.w3.org/2000/svg}"


def _svg_root(chart):
    """Render *chart* to SVG and return the parsed root element."""
    return ET.fromstring(chart.to_svg())


def _text_elements(root):
    """Return every ``<text>`` element in the SVG tree."""
    return list(root.iter(_SVG_NS + "text"))


@pytest.fixture()
def color_chart():
    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "g": ["a", "b", "c", "d"]})
    return fm.Chart(df).mark_point().encode(x="x", y="y", color="g")


@pytest.fixture()
def bar_chart():
    df = pl.DataFrame({"cat": ["a", "b", "c"], "y": [4, 5, 6]})
    return fm.Chart(df).mark_bar().encode(x="cat", y="y")


class TestOverrideRenderChartConfig:
    def test_x_axis_label_angle_rotates_end_anchored_labels(self, bar_chart):
        root = _svg_root(bar_chart.override(x_axis_label_angle=-45))
        rotated = [
            t for t in _text_elements(root) if (t.get("transform") or "").find("rotate(-45") != -1
        ]
        assert rotated, "expected at least one x-tick label rotated by -45"
        assert all(t.get("text-anchor") == "end" for t in rotated)

    def test_baseline_has_no_rotation(self, bar_chart):
        root = _svg_root(bar_chart)
        assert not any("rotate(-45" in (t.get("transform") or "") for t in _text_elements(root))


class TestOverrideRenderProperties:
    def test_width_override_sets_viewbox_width(self, base_chart):
        root = _svg_root(base_chart.override(width=900))
        assert root.get("viewBox").split()[2] == "900"
        assert root.get("width") == "900"

    def test_height_override_sets_viewbox_height(self, base_chart):
        root = _svg_root(base_chart.override(height=700))
        assert root.get("viewBox").split()[3] == "700"
        assert root.get("height") == "700"

    def test_property_override_beats_properties_call(self, base_chart):
        # spec §8 D5: override width beats .properties(width=...).
        chart = base_chart.properties(width=300).override(width=900)
        root = _svg_root(chart)
        assert root.get("viewBox").split()[2] == "900"


class TestOverrideRenderLegend:
    def test_legend_orient_changes_render(self, color_chart):
        baseline = color_chart.to_svg()
        relocated = color_chart.override(legend_orient="bottom").to_svg()
        assert relocated != baseline


class TestOverrideRenderColor:
    def test_color_scheme_changes_mark_fills(self, color_chart):
        baseline = color_chart.to_svg()
        recolored = color_chart.override(color_scheme="viridis").to_svg()
        assert recolored != baseline

    def test_color_scheme_fill_values_differ(self, color_chart):
        def fills(svg):
            root = ET.fromstring(svg)
            out = set()
            for el in root.iter():
                fill = el.get("fill")
                if fill and fill.startswith("#"):
                    out.add(fill.lower())
            return out

        baseline = fills(color_chart.to_svg())
        recolored = fills(color_chart.override(color_scheme="viridis").to_svg())
        assert baseline != recolored


class TestOverrideRenderEncodingScale:
    def test_x_scale_domain_changes_axis_extent(self, base_chart):
        def x_tick_labels(svg):
            root = ET.fromstring(svg)
            return [t.text for t in _text_elements(root) if t.text]

        baseline = x_tick_labels(base_chart.to_svg())
        overridden = x_tick_labels(base_chart.override(x_scale_domain=[0, 100]).to_svg())
        assert overridden != baseline
        # The override domain end (100) appears among the rendered tick labels.
        assert "100" in overridden
        assert "100" not in baseline


class TestOverrideRenderMarkStyle:
    def test_mark_corner_radius_rounds_bars(self, bar_chart):
        root = _svg_root(bar_chart.override(mark_corner_radius=6))
        rects = list(root.iter(_SVG_NS + "rect"))
        assert any(r.get("rx") == "6" for r in rects)

    def test_baseline_bars_have_no_corner_radius(self, bar_chart):
        root = _svg_root(bar_chart)
        rects = list(root.iter(_SVG_NS + "rect"))
        assert not any(r.get("rx") for r in rects)


class TestOverrideRenderCascade:
    def test_override_beats_configure_axis_configure_first(self, bar_chart):
        chart = bar_chart.configure_axis(label_angle=0).override(x_axis_label_angle=-45)
        root = _svg_root(chart)
        assert any("rotate(-45" in (t.get("transform") or "") for t in _text_elements(root))

    def test_override_beats_configure_axis_override_first(self, bar_chart):
        # Reverse construction order: override still wins (top of cascade, not
        # last-call-wins between configure and override).
        chart = bar_chart.override(x_axis_label_angle=-45).configure_axis(label_angle=0)
        root = _svg_root(chart)
        assert any("rotate(-45" in (t.get("transform") or "") for t in _text_elements(root))


class TestOverrideRenderDeprecation:
    def test_typed_equivalent_path_warns(self, base_chart):
        with pytest.warns(DeprecationWarning, match=r"\.configure_axis\(label_angle"):
            base_chart.override(x_axis_label_angle=-45).to_svg()

    def test_property_path_warns(self, base_chart):
        with pytest.warns(DeprecationWarning, match=r"\.properties\(width"):
            base_chart.override(width=900).to_svg()

    def test_genuine_gap_path_does_not_warn(self, base_chart):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # x_scale_domain has no typed configure_* equivalent: applies silently.
            base_chart.override(x_scale_domain=[0, 1]).to_svg()


class TestOverrideRenderErrors:
    def test_unknown_path_raises_at_render(self, base_chart):
        with pytest.raises(FerrumOverrideError, match="bogus_path"):
            base_chart.override(bogus_path=1).to_svg()

    def test_typo_raises_with_suggestion_at_render(self, base_chart):
        with pytest.raises(FerrumOverrideError, match="x_axis_label_angle"):
            base_chart.override(x_axis_lable_angle=-45).to_svg()

    def test_known_prefix_bad_leaf_raises_at_render(self, base_chart):
        with pytest.raises(FerrumOverrideError, match="mark_notathing"):
            base_chart.override(mark_notathing=1).to_svg()

    @pytest.mark.parametrize(
        "deprecated_path",
        ["axis_x", "axis_y", "x_axis_x", "x_axis_y", "y_axis_x", "y_axis_y"],
    )
    def test_deprecated_axis_xy_leaf_raises_typed_at_render_not_wire_error(
        self, base_chart, deprecated_path
    ):
        # The seam this pins: before the fix, deprecated axis spellings resolved
        # and reached the Rust wire gate, which hard-fails with an opaque
        # ``ValueError`` naming an unrecognized wire key. They must now refuse at
        # the override boundary with a typed, kwarg-naming ``FerrumOverrideError``
        # that names both the offending path and the documented replacement
        # Chart.axis(x/y=False).
        with pytest.raises(FerrumOverrideError, match=rf"{deprecated_path}.*Chart\.axis"):
            base_chart.override(**{deprecated_path: True}).to_svg()


class TestOverrideRenderMultiMark:
    def test_mark_override_on_layered_chart_raises(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        layered = fm.Chart(df).mark_line().encode(x="x", y="y") + fm.Chart(df).mark_point().encode(
            x="x", y="y"
        )
        with pytest.raises(FerrumOverrideError, match="multi-mark"):
            layered.override(mark_size=50).to_svg()

    def test_single_mark_override_applies(self, bar_chart):
        # The normal case: a single-mark chart accepts a mark_* override.
        root = _svg_root(bar_chart.override(mark_corner_radius=6))
        assert any(r.get("rx") == "6" for r in root.iter(_SVG_NS + "rect"))


class TestOverrideRenderNoOpStability:
    def test_no_override_render_unaffected_by_machinery(self, bar_chart):
        # Sanity that the applicator guard holds: a chart with no override
        # renders identically across calls and is unchanged by .override().
        baseline = bar_chart.to_svg()
        assert bar_chart.to_svg() == baseline
        assert bar_chart.override().to_svg() == baseline
