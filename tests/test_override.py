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
from ferrum._entry import voiced_as
from ferrum.exceptions import FerrumOverrideError
from ferrum.marks.base import _VALID_MARK_KWARGS
from ferrum.structural import BreakAxis, Inset, SecondaryY
from ferrum import _override_apply as oa
from ferrum._core import scale_accepted_keys


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

    def test_a_later_none_replaces_the_earlier_value_on_a_non_mark_path(self, base_chart):
        """``merge_overrides`` keeps the earlier value on a later ``None``
        only for ``mark_<leaf>`` paths; on every other path a later
        ``None`` replaces the earlier value like any other value, landing
        on the owning dataclass's own default (``AxisConfig.label_angle``),
        so the chart renders identically to one that was never overridden."""
        c = base_chart.override(x_axis_label_angle=-45).override(x_axis_label_angle=None)
        assert c._overrides["x_axis_label_angle"] is None
        assert c.to_svg() == base_chart.to_svg()

    def test_a_later_none_replaces_the_earlier_value_on_a_second_non_mark_path(self, base_chart):
        """Same rule, a different non-mark target (``ColorConfig.scheme``),
        so the pin does not rest on one dataclass's field happening to
        default to ``None``."""
        c = base_chart.override(color_scheme="viridis").override(color_scheme=None)
        assert c._overrides["color_scheme"] is None
        assert c.to_svg() == base_chart.to_svg()

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

    def test_scale_leaves_are_the_gates_wire_keys_plus_snake_aliases(self):
        # The registry is derived from the wire gate's own per-type
        # accepted-key table, not from the *Scale pyclasses' attributes.
        # Pin the derivation exactly: the union of every type's accepted keys,
        # MINUS the system-only keys, plus "type", plus one snake_case alias per
        # camelCase wire key.
        #
        # The subtraction (#247A) is why this assertion moved: a wire key the
        # gate must PARSE is not automatically a presentation option the caller
        # may SET. `systemInjected` records that ferrum's own spec build
        # authored a scale rather than the caller, so the gate has to accept it
        # (the bar zero-anchor emits it across the same wire boundary every raw
        # scale dict crosses) while the override escape hatch must not offer a
        # way to assert ferrum's authorship on a hand-written scale.
        rule = next(r for r in oa._PREFIX_RULES if r.prefix == "x_scale_")
        wire_union = set()
        for scale_type in oa._SCALE_TYPE_TAGS:
            wire_union.update(scale_accepted_keys(scale_type))
        # The exclusion must actually bite: every system-only key is a real key
        # of the gate's table, so removing one from the wire schema without
        # updating the registry fails here rather than silently subtracting
        # nothing (the same guard shape as the AxisConfig `x`/`y` deprecation
        # check above).
        assert oa._SCALE_LEAF_SYSTEM_ONLY <= wire_union
        expected = (
            (wire_union - oa._SCALE_LEAF_SYSTEM_ONLY) | {"type"} | set(oa._SCALE_LEAF_WIRE_ALIASES)
        )
        assert rule.valid_leaves == expected
        # Derived read-only getters are not leaves: `num_bins` is
        # `len(bins)` on BinOrdinalScale, `quantiles` is QuantileScale's
        # computed thresholds, and `utc` is a TimeScale constructor flag
        # whose real override spelling is `x_scale_type="utc"`. None of the
        # three names a wire key of any scale type.
        assert {"num_bins", "quantiles", "utc"}.isdisjoint(rule.valid_leaves)

    def test_every_camelcase_gate_key_has_a_snake_alias_or_is_system_only(self):
        """Guard the reverse direction of :data:`_SCALE_LEAF_WIRE_ALIASES`:
        every wire key the gate accepts that is not all-lowercase (a
        camelCase spelling) must either have a snake_case override alias or
        be excluded as :data:`_SCALE_LEAF_SYSTEM_ONLY`. Otherwise that wire
        key has no ``.override()`` spelling that reads like the rest of the
        flat snake_case surface — the same class of gap
        :data:`_SCALE_LEAF_WIRE_ALIASES`'s forward direction closes for the
        keys it does list, just unchecked for a key nobody added an alias
        for.
        """
        wire_union = set()
        for scale_type in oa._SCALE_TYPE_TAGS:
            wire_union.update(scale_accepted_keys(scale_type))

        camel_case_keys = {key for key in wire_union if key != key.lower()}
        aliased = set(oa._SCALE_LEAF_WIRE_ALIASES.values())
        # Vacuity control: the keys this test accounts for must actually be
        # camelCase gate keys, so a change that emptied `camel_case_keys`
        # (e.g. the gate stopped serving any camelCase key, or the
        # lowercase-comparison heuristic above stopped matching) fails here
        # instead of letting `unaccounted` below pass on an empty set.
        assert aliased | oa._SCALE_LEAF_SYSTEM_ONLY <= camel_case_keys
        unaccounted = camel_case_keys - aliased - oa._SCALE_LEAF_SYSTEM_ONLY
        assert not unaccounted

    def test_scale_type_tags_cover_every_tag_the_scale_classes_emit(self):
        """``_SCALE_TYPE_TAGS`` is published from Rust's ``ScaleSpec::TAGS``
        (``ferrum._core.scale_type_tags()``), not hand-listed, so its
        validity against the gate is a Rust-test-enforced fact rather than
        something this test needs to re-prove. What is still worth guarding
        on the Python side is one direction only: every tag ferrum's own
        scale classes can actually serialise must be a member of the
        published set — a Python scale class emitting a tag the publication
        forgot would silently withhold that tag's variant-specific keys from
        the whole override surface.
        """
        from ferrum._core import (
            BandScale,
            BinOrdinalScale,
            DivergingScale,
            LinearScale,
            LogScale,
            OrdinalScale,
            PointScale,
            PowScale,
            QuantileScale,
            QuantizeScale,
            SequentialScale,
            SqrtScale,
            SymlogScale,
            ThresholdScale,
            TimeScale,
        )

        instances = [
            LinearScale(),
            LogScale(),
            PowScale(),
            SqrtScale(),
            SymlogScale(),
            OrdinalScale(domain=["a"]),
            BandScale(),
            PointScale(),
            SequentialScale(),
            DivergingScale(),
            QuantizeScale(domain=[0.0, 1.0], range=["#000", "#fff"]),
            QuantileScale(domain=[0.0, 1.0], range=[0.0, 1.0]),
            ThresholdScale(domain=[0.0, 1.0], range=[0.0, 1.0, 2.0]),
            BinOrdinalScale(bins=[1.0, 2.0]),
            TimeScale(utc=False),
            TimeScale(utc=True),
        ]
        emitted_tags = {inst._to_scale_spec_dict()["type"] for inst in instances}
        assert emitted_tags <= set(oa._SCALE_TYPE_TAGS)

    def test_every_registry_leaf_emits_a_wire_key_some_scale_type_accepts(self):
        """The single assertion that makes the override-leaf / wire-key class
        of divergence impossible.

        The registry used to advertise five leaves (``padding_inner``,
        ``padding_outer``, ``domain_mid``, ``quantiles``, ``utc``) that the
        wire gate then refused for EVERY scale type, with no working spelling
        anywhere: the gate's message named the spelling the registry rejected
        and the registry's did-you-mean named the spelling the gate rejected.
        Every leaf must fold to a key at least one scale type accepts.

        Asserted on what ``build_payload`` actually EMITS (the production
        fold, snake→wire alias included), not on the alias map read back.

        Probed with ``0`` rather than ``None``: a ``None`` scale-override
        leaf is a deliberate no-op (the chart's own value stands rather
        than being erased), so it folds nothing for ANY leaf and would make
        this generic reachability probe KeyError on
        ``payload.encoding["x"]``. ``0`` is not special-cased for anything
        this registry resolves.
        """
        wire_union = set()
        for scale_type in oa._SCALE_TYPE_TAGS:
            wire_union.update(scale_accepted_keys(scale_type))

        emitted = set()
        for leaf in oa._scale_leaves():
            payload = oa.build_payload({f"x_scale_{leaf}": 0})
            emitted.update(payload.encoding["x"]["scale"])

        assert emitted - {"type"} <= wire_union

    @pytest.mark.parametrize("orphan_leaf", ["utc", "quantiles", "num_bins"])
    def test_leaf_with_no_wire_key_refuses_at_the_registry(self, orphan_leaf, base_chart):
        """A scale attribute that names no wire key of any type is not a
        registry leaf, so it refuses with the registry's own actionable error
        rather than an opaque wire refusal further downstream.
        """
        with pytest.raises(FerrumOverrideError, match="Unknown override path"):
            base_chart.override(**{f"x_scale_{orphan_leaf}": True}).to_svg()

    def test_coord_leaves_match_coord_dataclass_fields(self):
        rule = next(r for r in oa._PREFIX_RULES if r.prefix == "coord_")
        # Union of CoordCartesian / CoordFixed / CoordPolar / CoordGeo fields.
        assert {"clip", "expand", "xlim", "ylim", "ratio", "theta", "projection"} <= (
            rule.valid_leaves
        )

    def test_padding_numeric_leaves_are_all_padding_leaves_minus_auto(self):
        # oa._PADDING_NUMERIC_LEAVES (the spec §4.7 pixel-contract guard's
        # leaf set) is derived from PaddingConfig's own fields, not
        # hand-enumerated, so a future PaddingConfig field is validated
        # automatically instead of silently passing through the override
        # boundary unchecked. Pin both halves of that derivation directly.
        all_padding_leaves = frozenset(f.name for f in fields(PaddingConfig))
        assert oa._PADDING_NUMERIC_LEAVES == all_padding_leaves - {"auto"}
        assert "auto" not in oa._PADDING_NUMERIC_LEAVES


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

    @pytest.mark.parametrize("value", [False, True])
    def test_padding_auto_bool_leaf_is_not_routed_through_pixel_validator(self, value):
        # ``padding_auto`` is a registered, valid PaddingConfig leaf (bool),
        # not a pixel value. The spec §4.7 numeric/non-negative validator
        # must not fire on it in either direction.
        oa.validate({"padding_auto": value})


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

    def test_encoding_paths_records_the_path_the_caller_wrote(self):
        """#157's provenance field: the WIRE key maps back to the exact
        flat override path, snake_case or wire-spelled alike.
        """
        payload = oa.build_payload({"x_scale_padding_inner": 0.3})
        assert payload.encoding_paths == {"x": {"paddingInner": "x_scale_padding_inner"}}

        payload = oa.build_payload({"x_scale_paddingInner": 0.3})
        assert payload.encoding_paths == {"x": {"paddingInner": "x_scale_paddingInner"}}

    def test_encoding_paths_empty_when_no_scale_override(self):
        payload = oa.build_payload({"x_axis_label_angle": -45})
        assert payload.encoding_paths == {}

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

    def test_padding_auto_override_renders(self, base_chart):
        # Regression pin: ``.override(padding_auto=False)`` is a registered,
        # valid PaddingConfig leaf (tests/test_override.py::TestOverrideRegistryParity)
        # and must reach render, not be refused by the pixel-value validator
        # that only applies to top/right/bottom/left.
        root = _svg_root(base_chart.override(padding_auto=False))
        assert root.tag.endswith("svg")


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


# One renderable override per registry scale leaf: the channel to override,
# the field bound to it, the scale type that accepts the leaf's wire key (None
# for `type` itself, whose value IS the type), and a legal value.
#
# The keys are pinned equal to `_scale_leaves()` by the sweep below, so a new
# registry leaf cannot land without a render case, and a leaf that stops being
# reachable fails the sweep instead of quietly losing coverage.
_SCALE_LEAF_RENDER_CASES: dict[str, tuple[str, str, str | None, object]] = {
    "type": ("x", "num", None, "linear"),
    "domain": ("x", "num", "linear", [0.0, 5.0]),
    "range": ("x", "num", "linear", [0.0, 200.0]),
    "clamp": ("x", "num", "linear", True),
    "padding": ("x", "num", "linear", 5.0),
    "reverse": ("x", "num", "linear", True),
    "nice": ("x", "num", "linear", True),
    "zero": ("x", "num", "linear", True),
    "domainParam": ("x", "num", "linear", "p"),
    "domain_param": ("x", "num", "linear", "p"),
    "base": ("x", "num", "log", 2.0),
    "constant": ("x", "num", "symlog", 2.0),
    "exponent": ("x", "num", "pow", 2.0),
    "align": ("x", "cat", "band", 0.0),
    "paddingInner": ("x", "cat", "band", 0.3),
    "padding_inner": ("x", "cat", "band", 0.3),
    "paddingOuter": ("x", "cat", "band", 0.2),
    "padding_outer": ("x", "cat", "band", 0.2),
    "scheme": ("color", "val", "sequential", "viridis"),
    "stops": ("color", "val", "sequential", [[0.0, "#ff0000"], [1.0, "#00ff00"]]),
    "domainMid": ("color", "val", "diverging", 15.0),
    "domain_mid": ("color", "val", "diverging", 15.0),
    "bins": ("color", "val", "bin-ordinal", [0.0, 10.0, 20.0, 30.0]),
    "bin": ("color", "val", "bin-ordinal", {"bin_count": 3}),
}


class TestOverrideRenderEncodingScale:
    def test_render_cases_cover_every_registry_leaf(self):
        assert set(_SCALE_LEAF_RENDER_CASES) == set(oa._scale_leaves())

    @pytest.mark.parametrize("leaf", sorted(_SCALE_LEAF_RENDER_CASES))
    def test_every_scale_leaf_renders_through_override(self, leaf):
        """Every advertised ``<channel>_scale_<leaf>`` override path actually
        reaches the wire and renders.

        The render block used to prove exactly ONE scale leaf (``x_scale_domain``,
        the single leaf whose Python spelling coincides with its wire spelling),
        which is why a registry derived from Python attribute names could drift
        from the per-type wire gate unnoticed: five advertised leaves resolved
        here and were then hard-refused at the wire with no working spelling
        anywhere. This sweep is the RED proof for that fix and the guard against
        the next schema divergence.
        """
        channel, field, scale_type, value = _SCALE_LEAF_RENDER_CASES[leaf]
        df = pl.DataFrame(
            {"cat": ["a", "b", "c"], "num": [1.0, 2.0, 3.0], "val": [10.0, 20.0, 15.0]}
        )
        encoding = {"x": "cat", "y": "val", channel: field}
        overrides: dict[str, object] = {f"{channel}_scale_{leaf}": value}
        if scale_type is not None:
            overrides[f"{channel}_scale_type"] = scale_type
        chart = fm.Chart(df).mark_point().encode(**encoding).override(**overrides)
        assert "<svg" in chart.to_svg()

    def test_snake_and_wire_spellings_of_one_leaf_render_identically(self):
        """Both accepted spellings fold to the same wire key, so they must
        produce byte-identical SVG — the "accept both, emit the wire one" half
        of the registry fix, proven at the render boundary rather than by
        reading the alias map back.
        """
        df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
        base = fm.Chart(df).mark_bar().encode(x="cat", y="val")
        snake = base.override(x_scale_type="band", x_scale_padding_inner=0.3).to_svg()
        wire = base.override(x_scale_type="band", x_scale_paddingInner=0.3).to_svg()
        unpadded = base.override(x_scale_type="band", x_scale_paddingInner=0.0).to_svg()
        assert snake == wire
        assert snake != unpadded, "the override must actually change geometry, or this is vacuous"

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


class TestOverrideScaleValueRefusalNamesThePath:
    """#157: a malformed override SCALE VALUE refuses naming the call and
    the exact flat override path, for any leaf -- not only band padding.
    """

    def test_unencodable_domain_value_refuses_as_valueerror_naming_the_path(self, base_chart):
        """A value ``json.dumps`` cannot encode must raise a named
        ``ValueError`` (``FerrumOverrideError``), never a raw ``TypeError``.
        """
        with pytest.raises(
            FerrumOverrideError,
            match=(
                r"^Chart\.override\(y_scale_domain=\.\.\.\) "
                r"cannot be encoded as a scale value: "
                r"Object of type object is not JSON serializable$"
            ),
        ):
            base_chart.override(y_scale_domain=object()).to_svg()

    def test_non_padding_leaf_wrong_type_refuses_naming_the_path(self, base_chart):
        """The fix is not band-padding-specific: any wire leaf's malformed
        value is named, e.g. ``reverse`` (a boolean field) given a string.
        """
        with pytest.raises(
            FerrumOverrideError,
            match=(
                r"^Chart\.override\(y_scale_reverse=\.\.\.\) "
                r'invalid type: string "a", expected a boolean$'
            ),
        ):
            base_chart.override(y_scale_reverse="a").to_svg()

    def test_malformed_temporal_domain_element_refuses_naming_the_path(self):
        """A malformed temporal domain element reaches its OWN conversion
        failure (``_convert_temporal_domain_elements``, run by
        ``_scale_to_dict``) before the merged scale dict is ever built --
        that failure must still be named by the override path, not left to
        surface as the unnamed ``TimeScale`` conversion error.
        """
        tmp = pl.DataFrame({"t": ["2020-01-01", "2021-01-01"], "v": [1.0, 2.0]}).with_columns(
            pl.col("t").str.to_datetime()
        )
        chart = (
            fm.Chart(tmp)
            .mark_point()
            .encode(x=fm.X("t", scale=fm.TimeScale(domain=[0.0, 100.0])), y="v")
        )
        with pytest.raises(
            FerrumOverrideError,
            match=(
                r"^Chart\.override\(x_scale_domain=\.\.\.\) "
                r"Cannot parse TimeScale domain value 'garbage' as an ISO-8601 date or "
                r"datetime\. Use 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS\[\.ffffff\]\[.HH:MM\]'\.$"
            ),
        ):
            chart.override(x_scale_domain=["garbage", "2020-01-10"]).to_svg()

    def test_malformed_scale_type_alone_refuses_naming_x_scale_type(self, base_chart):
        with pytest.raises(
            FerrumOverrideError,
            match=r"^Chart\.override\(x_scale_type=\.\.\.\) unknown variant `bogus`",
        ):
            base_chart.override(x_scale_type="bogus").to_svg()

    def test_malformed_scale_type_with_another_scale_override_still_names_x_scale_type(
        self, base_chart
    ):
        """A second scale override in the same call (``x_scale_padding``)
        must not steer the refusal onto the wrong leaf -- the type's own
        refusal is probed before any other key, so it keeps naming
        ``x_scale_type`` rather than ``x_scale_padding``."""
        with pytest.raises(
            FerrumOverrideError,
            match=r"^Chart\.override\(x_scale_type=\.\.\.\) unknown variant `bogus`",
        ):
            base_chart.override(x_scale_type="bogus", x_scale_padding=0.2).to_svg()

    def test_non_string_scale_type_refuses_naming_x_scale_type(self, base_chart):
        with pytest.raises(
            FerrumOverrideError,
            match=r"^Chart\.override\(x_scale_type=\.\.\.\) invalid type: integer",
        ):
            base_chart.override(x_scale_type=3).to_svg()

    @pytest.mark.parametrize(
        "domain",
        [[True, False], [object(), object()]],
        ids=["bools", "objects"],
    )
    def test_time_scale_domain_of_bools_or_objects_refuses_naming_x_scale_domain(self, domain):
        dates = pl.DataFrame({"t": ["2020-01-01", "2021-01-01"], "v": [1.0, 2.0]}).with_columns(
            pl.col("t").str.to_datetime()
        )
        chart = fm.Chart(dates).mark_point().encode(x=fm.X("t", scale=fm.TimeScale()), y="v")
        with pytest.raises(FerrumOverrideError, match=r"^Chart\.override\(x_scale_domain=\.\.\.\)"):
            chart.override(x_scale_domain=domain).to_svg()


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


# ---------------------------------------------------------------------------
# A None scale-override leaf is no override, for EVERY <channel>_scale_<leaf>,
# not only the three band-padding ones (#157). One probe chart per WIRE leaf
# in oa._scale_leaves() (aliases fold to the same wire key and behave
# identically, so padding_inner/paddingInner, padding_outer/paddingOuter,
# domain_mid/domainMid and domain_param/domainParam share one row).
# ---------------------------------------------------------------------------

import json

_df4 = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10.0, 20.0, 30.0, 40.0]})


def _scale_dict(chart: "fm.Chart", channel: str) -> dict:
    """Return *chart*'s resolved wire scale dict for *channel*."""
    return json.loads(chart.to_spec().to_json())["encoding"][channel]["scale"]


#: wire_leaf -> (probe chart with the leaf's own value already set, the
#: channel the leaf lives on, a different valid value for that leaf under
#: the same scale type).
_SCALE_LEAF_NONE_NOOP_PROBES: dict[str, tuple["fm.Chart", str, object]] = {
    "type": (
        fm.Chart(_df4)
        .mark_point()
        .encode(x="cat", y=fm.Y("val", scale=fm.LogScale(domain=[1.0, 100.0]))),
        "y",
        "sqrt",
    ),
    "domain": (
        fm.Chart(_df4)
        .mark_bar()
        .encode(x=fm.X("cat", scale=fm.BandScale(domain=["d", "c", "b", "a"])), y="val"),
        "x",
        ["a", "b", "c", "d"],
    ),
    "range": (
        fm.Chart(_df4)
        .mark_point()
        .encode(x="cat", y=fm.Y("val", scale=fm.LinearScale(range=[0.0, 200.0]))),
        "y",
        [10.0, 300.0],
    ),
    "padding": (
        fm.Chart(_df4).mark_bar().encode(x=fm.X("cat", scale=fm.BandScale(padding=0.2)), y="val"),
        "x",
        0.1,
    ),
    "paddingInner": (
        fm.Chart(_df4)
        .mark_bar()
        .encode(x=fm.X("cat", scale=fm.BandScale(padding_inner=0.2)), y="val"),
        "x",
        0.1,
    ),
    "paddingOuter": (
        fm.Chart(_df4)
        .mark_bar()
        .encode(x=fm.X("cat", scale=fm.BandScale(padding_outer=0.2)), y="val"),
        "x",
        0.1,
    ),
    "align": (
        fm.Chart(_df4).mark_bar().encode(x=fm.X("cat", scale=fm.BandScale(align=0.2)), y="val"),
        "x",
        0.1,
    ),
    "reverse": (
        fm.Chart(_df4)
        .mark_point()
        .encode(x=fm.X("cat", scale=fm.PointScale(reverse=True)), y="val"),
        "x",
        False,
    ),
    "nice": (
        fm.Chart(_df4).mark_point().encode(x="cat", y=fm.Y("val", scale=fm.LinearScale(nice=True))),
        "y",
        False,
    ),
    "zero": (
        fm.Chart(_df4)
        .mark_point()
        .encode(
            x="cat", y=fm.Y("val", scale={"type": "linear", "domain": [0.0, 10.0], "zero": False})
        ),
        "y",
        True,
    ),
    "clamp": (
        fm.Chart(_df4)
        .mark_point()
        .encode(x="cat", y=fm.Y("val", scale=fm.LinearScale(clamp=True))),
        "y",
        False,
    ),
    "base": (
        fm.Chart(_df4)
        .mark_point()
        .encode(x="cat", y=fm.Y("val", scale=fm.LogScale(base=2.0, domain=[1.0, 100.0]))),
        "y",
        10.0,
    ),
    "constant": (
        fm.Chart(_df4)
        .mark_point()
        .encode(x="cat", y=fm.Y("val", scale=fm.SymlogScale(constant=1.0, domain=[0.0, 100.0]))),
        "y",
        2.0,
    ),
    "exponent": (
        fm.Chart(_df4)
        .mark_point()
        .encode(x="cat", y=fm.Y("val", scale=fm.PowScale(exponent=2.0, domain=[0.0, 100.0]))),
        "y",
        3.0,
    ),
    "scheme": (
        fm.Chart(_df4)
        .mark_point()
        .encode(
            x="cat", y="val", color=fm.Color("val", scale=fm.SequentialScale(scheme="viridis"))
        ),
        "color",
        "plasma",
    ),
    "stops": (
        fm.Chart(_df4)
        .mark_point()
        .encode(
            x="cat",
            y="val",
            color=fm.Color(
                "val", scale={"type": "sequential", "stops": [[0.0, "#ff0000"], [1.0, "#00ff00"]]}
            ),
        ),
        "color",
        [[0.0, "#000000"], [1.0, "#ffffff"]],
    ),
    "domainMid": (
        fm.Chart(_df4)
        .mark_point()
        .encode(x="cat", y="val", color=fm.Color("val", scale=fm.DivergingScale(domain_mid=5.0))),
        "color",
        10.0,
    ),
    "domainParam": (
        fm.Chart(_df4)
        .mark_point()
        .encode(x="cat", y=fm.Y("val", scale={"type": "linear", "domainParam": "p1"})),
        "y",
        "p2",
    ),
    "bins": (
        fm.Chart(_df4)
        .mark_point()
        .encode(
            x="cat",
            y="val",
            color=fm.Color("val", scale=fm.BinOrdinalScale(bins=[0.0, 10.0, 20.0, 30.0])),
        ),
        "color",
        [0.0, 5.0, 15.0, 25.0, 35.0],
    ),
    "bin": (
        fm.Chart(_df4)
        .mark_point()
        .encode(
            x="cat",
            y="val",
            color=fm.Color("val", scale={"type": "bin-ordinal", "bin": {"bin_count": 3}}),
        ),
        "color",
        {"bin_count": 5},
    ),
}

#: Every snake_case alias folds to a wire key already covered above, so the
#: probe table's wire-key coverage is what must equal every WIRE key
#: oa._scale_leaves() advertises (the aliases are a spelling of those same
#: wire keys, not additional ones).
_SCALE_LEAF_WIRE_KEYS = {oa._SCALE_LEAF_WIRE_ALIASES.get(leaf, leaf) for leaf in oa._scale_leaves()}


class TestOverrideScaleNoneIsNoOverride:
    """A None <channel>_scale_<leaf> override is no override, for every leaf
    the registry advertises -- not only band/point padding.
    """

    def test_probe_table_covers_every_wire_leaf(self):
        assert set(_SCALE_LEAF_NONE_NOOP_PROBES) == _SCALE_LEAF_WIRE_KEYS

    @pytest.mark.parametrize("leaf", sorted(_SCALE_LEAF_NONE_NOOP_PROBES))
    def test_none_override_leaves_the_chart_value_standing(self, leaf):
        """Checked at both the render and the wire level: a few leaves
        (``domainParam``, ``align`` with no padding set, ``clamp``,
        ``nice``, ``zero``) move no pixel either way, so the render alone
        cannot tell "the chart's own value survived" from "both sides lost
        it the same way". The resolved scale dict can: it must still carry
        the chart's own value for this wire key, unchanged by the override.
        """
        chart, channel, _alt_value = _SCALE_LEAF_NONE_NOOP_PROBES[leaf]
        baseline_svg = chart.to_svg()
        baseline_scale = _scale_dict(chart, channel)
        overridden = chart.override(**{f"{channel}_scale_{leaf}": None})
        assert overridden.to_svg() == baseline_svg
        assert _scale_dict(overridden, channel) == baseline_scale

    @pytest.mark.parametrize("leaf", sorted(_SCALE_LEAF_NONE_NOOP_PROBES))
    def test_a_later_none_override_reverts_to_the_chart_value(self, leaf):
        """``.override(leaf=v).override(leaf=None)``: the later ``None``
        replaces ``v`` in the STORED override dict -- asserted directly on
        ``chart._overrides`` below, since :func:`merge_overrides` only
        special-cases mark-style paths and must store the later ``None``
        like any other value, not silently keep ``v``. The payload then
        drops the stored ``None``, so the chart's own value stands at both
        the render and the wire level, not ``v`` (the same two-level check
        as the single-override test above, for the same reason: several
        leaves move no pixel for either ``v`` or the chart's own value).
        """
        chart, channel, alt_value = _SCALE_LEAF_NONE_NOOP_PROBES[leaf]
        baseline_svg = chart.to_svg()
        baseline_scale = _scale_dict(chart, channel)
        path = f"{channel}_scale_{leaf}"
        overridden = chart.override(**{path: alt_value}).override(**{path: None})
        assert overridden._overrides[path] is None
        assert overridden.to_svg() == baseline_svg
        assert _scale_dict(overridden, channel) == baseline_scale


# ---------------------------------------------------------------------------
# An `.override` refusal from the padding (auto or a numeric side), scale,
# mark or viewport (`width`/`height`) leaf families, and a `coord_*` leaf
# the chart's coord does not take, is the one documented FerrumOverrideError
# class and re-voices through the same mechanism. Config-section leaves
# (`axis_*`/`x_axis_*`/`y_axis_*`, `legend_*`, `title_*`, `grid_*`,
# `color_*`) are NOT covered by this class: they refuse as a plain
# `ValueError`, from either the config dataclass's own validator (e.g.
# `LegendConfig.orient`/`GridConfig.color`) or the Rust `chart_config`
# serde decode (e.g. `axis_label_angle`/`title_font_size`), outside
# FerrumOverrideError's contract (a known, logged gap). A malformed value
# for a `coord_*` leaf the coord does take (`coord_clip="yes"`) refuses
# the same way, from the Rust `coord` decode.
# ---------------------------------------------------------------------------


class TestOverrideRefusalIsFerrumOverrideErrorAcrossLeafFamilies:
    """Covers exactly the padding, scale, mark, viewport and coord leaf
    families (the parametrized cases below plus `test_coord_leaf_refusal_...`).
    Config-section leaves are a separate, currently-ValueError path; see
    the module comment above this class."""

    @pytest.mark.parametrize(
        ("overrides", "text_prefix"),
        [
            (
                {"padding_auto": "yes"},
                "Chart.override(padding_auto=...) must be True or False",
            ),
            ({"padding_top": -1}, "Chart.override(padding_top=...) must be non-negative"),
            ({"x_scale_type": "bogus"}, "Chart.override(x_scale_type=...) unknown variant"),
            (
                {"mark_size": "big"},
                "Chart.override(size=...) must be a real number",
            ),
            (
                {"width": float("nan")},
                "Chart.override(width=...) must be a finite numeric pixel value",
            ),
        ],
        ids=["padding-auto", "padding-side", "scale-leaf", "mark-leaf", "viewport"],
    )
    def test_isinstance_ferrum_override_error(self, base_chart, overrides, text_prefix):
        """Each family case must reach ITS OWN refusal path, not a shared
        fallback (e.g. the unknown-path refusal) -- the assertion on
        *text_prefix* is what proves that; isinstance alone cannot
        distinguish which door actually refused."""
        with pytest.raises(FerrumOverrideError) as excinfo:
            base_chart.override(**overrides).to_svg()
        assert isinstance(excinfo.value, FerrumOverrideError)
        assert str(excinfo.value).startswith(text_prefix)

    def test_coord_leaf_refusal_is_ferrum_override_error(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .coord(fm.CoordPolar(theta="x"))
            .override(coord_ratio=1)
        )
        with pytest.raises(FerrumOverrideError) as excinfo:
            chart.to_svg()
        assert isinstance(excinfo.value, FerrumOverrideError)
        assert str(excinfo.value).startswith("Chart.override: coord_* override")


class TestOverridePaddingRefusalIsReVoiceable:
    """A figure that renders a padding override on the caller's behalf
    re-voices the refusal as itself, exactly as every other leaf family's
    override refusal does."""

    def test_a_figure_re_voices_the_padding_auto_refusal(self, base_chart):
        chart = base_chart.override(padding_auto="yes")
        with pytest.raises(FerrumOverrideError) as excinfo:
            with voiced_as("roc_chart"):
                chart.to_spec()
        assert (
            str(excinfo.value)
            == "roc_chart(padding_auto=...) must be True or False; received 'yes'"
        )

    def test_a_figure_re_voices_the_padding_side_refusal(self, base_chart):
        chart = base_chart.override(padding_top=-1)
        with pytest.raises(FerrumOverrideError) as excinfo:
            with voiced_as("roc_chart"):
                chart.to_spec()
        assert str(excinfo.value) == "roc_chart(padding_top=...) must be non-negative; received -1"
