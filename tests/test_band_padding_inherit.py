"""#157: an unset band padding inherits the auto-inferred categorical layout.

Covers the user-facing Python surface across every declaration door (class,
dict, override) for a `BandScale`/`PointScale`-switched-to-band with no
padding argument: it must render exactly like the auto-inferred axis, not
Vega-Lite's hidden 0.1 default. An explicit `padding`/`padding_inner`/
`padding_outer` keeps d3's exact band math (the "d3 control" cases below),
and malformed values are still refused. Rust (`crates/ferrum-core/src/scale/
discrete.rs`) owns the one validated `BandPadding` value -- unset state,
bounds, and resolution -- for the class, dict, override, resolver, facade and
getters alike; nothing here resolves or validates padding on the Python side.

Reference measurements below are 4 categories, `mark_bar`, default 640x400
viewport, panel width 561.636px.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import polars as pl
import pytest

import ferrum as fm
import ferrum._core as core

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_DF4 = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [1.0, 2.0, 3.0, 4.0]})

# The auto-inferred axis's bar width for 4 categories on the default
# 640x400 viewport (panel width 561.636px): 0.8 * 561.636 / 4.
_AUTO_BAR_WIDTH = 112.327

# d3-control: BandScale(padding=0.1)'s bar width under the same setup,
# unaffected by #157 -- an explicit `padding` keeps d3's exact band math.
_PADDED_0_1_BAR_WIDTH = 98.629


def _bar_widths(svg: str) -> list[float]:
    """Every distinct <rect> width in *svg*, rounded to 3dp and sorted."""
    root = ET.fromstring(svg)
    rects = [e for e in root.iter() if e.tag.endswith("rect")]
    return sorted({round(float(r.get("width")), 3) for r in rects if r.get("width")})


def _bar_chart(x_channel) -> "fm.Chart":
    return fm.Chart(_DF4).mark_bar().encode(x=x_channel, y="val")


def _x_scale_dict(chart: "fm.Chart") -> dict:
    return json.loads(chart.to_spec().to_json())["encoding"]["x"]["scale"]


# ---------------------------------------------------------------------------
# 1. Every entry door, unpadded, renders the auto-inferred bar width (#157)
# ---------------------------------------------------------------------------


class TestEveryDoorInheritsTheAutoAxis:
    """`BandScale()`, `{"type": "band"}` and `.override(x_scale_type="band")`
    with no padding argument render bars identical to the auto-inferred axis
    on every declaration door.
    """

    def test_auto_axis_baseline(self):
        auto = _bar_chart("cat")
        assert _bar_widths(auto.to_svg()) == [_AUTO_BAR_WIDTH, 561.636, 640.0]

    def test_class_unpadded(self):
        chart = _bar_chart(fm.X("cat", scale=fm.BandScale()))
        assert _bar_widths(chart.to_svg()) == _bar_widths(_bar_chart("cat").to_svg())
        assert _AUTO_BAR_WIDTH in _bar_widths(chart.to_svg())

    def test_class_unpadded_with_explicit_range(self):
        chart = _bar_chart(fm.X("cat", scale=fm.BandScale(range=[0.0, 561.636])))
        assert _AUTO_BAR_WIDTH in _bar_widths(chart.to_svg())

    def test_dict_unpadded(self):
        chart = _bar_chart(fm.X("cat", scale={"type": "band"}))
        assert _AUTO_BAR_WIDTH in _bar_widths(chart.to_svg())

    def test_override_type_switch_on_auto_axis(self):
        chart = _bar_chart("cat").override(x_scale_type="band")
        assert _AUTO_BAR_WIDTH in _bar_widths(chart.to_svg())

    def test_override_type_switch_on_point_scale(self):
        """`.override(x_scale_type="band")` on a `PointScale()` x inherits
        too: `PointScale()`'s wire omits its unset `padding`, so the
        type-switch merge carries nothing for the band side to inherit
        from, and the result is the unpadded auto-inferred layout.
        """
        chart = _bar_chart(fm.X("cat", scale=fm.PointScale())).override(x_scale_type="band")
        assert _AUTO_BAR_WIDTH in _bar_widths(chart.to_svg())

    def test_all_doors_render_byte_identical_to_auto(self):
        """Every door above renders the SAME svg as the auto-inferred axis,
        not merely the same bar width -- the strongest form of "equal to
        auto" (spec §9).

        The explicit-range door (`BandScale(range=[0, 561.636])`) is
        excluded here: an explicit `range=` decouples the scale's pixel
        bounds from the chart's own margin-derived plot rect, a genuinely
        different code path, so its gridline/margin geometry legitimately
        differs even though its bar WIDTH matches (pinned separately in
        `test_class_unpadded_with_explicit_range`).
        """
        auto_svg = _bar_chart("cat").to_svg()
        doors = [
            _bar_chart(fm.X("cat", scale=fm.BandScale())),
            _bar_chart(fm.X("cat", scale={"type": "band"})),
            _bar_chart("cat").override(x_scale_type="band"),
            _bar_chart(fm.X("cat", scale=fm.PointScale())).override(x_scale_type="band"),
        ]
        for chart in doors:
            assert chart.to_svg() == auto_svg


def test_override_type_switch_on_point_scale_with_explicit_padding_carries_it():
    """`.override(x_scale_type="band")` on `PointScale(padding=0.5)` carries
    the user-set 0.5 across the switch: the switch is not flattened to the
    unpadded inherit just because it changes kind.

    A point's end padding carries by MEANING, not by key name -- it
    becomes the band's `paddingOuter` (the d3-equivalent outer inset), not
    `padding` (both-sides shorthand), so `paddingInner` stays unset. The
    measured bar width (89.862px) is `BandScale(padding_outer=0.5)`'s own
    width; a merge that instead carried the value under the shared key
    name `padding` (both-sides shorthand) would read `BandScale(padding=
    0.5)`'s narrower 49.923px.
    """
    unpadded = _bar_chart(fm.X("cat", scale=fm.PointScale())).override(x_scale_type="band")
    carried = _bar_chart(fm.X("cat", scale=fm.PointScale(padding=0.5))).override(
        x_scale_type="band"
    )
    assert _x_scale_dict(carried)["paddingOuter"] == 0.5
    assert "padding" not in _x_scale_dict(carried)
    assert "paddingInner" not in _x_scale_dict(carried)
    # The carried paddingOuter=0.5 narrows the band from the unpadded
    # 112.327px to 89.862px -- not merely "some other width".
    assert _bar_widths(carried.to_svg())[0] == pytest.approx(89.862, abs=1e-3)
    assert _bar_widths(unpadded.to_svg())[0] == pytest.approx(_AUTO_BAR_WIDTH, abs=1e-3)


# ---------------------------------------------------------------------------
# 2. Explicit padding keeps d3's exact band math (controls, unaffected)
# ---------------------------------------------------------------------------


def test_explicit_padding_0_1_is_a_d3_control():
    """`BandScale(padding=0.1)` is untouched by #157: still 98.629px."""
    chart = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=0.1)))
    assert _PADDED_0_1_BAR_WIDTH in _bar_widths(chart.to_svg())


def test_explicit_padding_inner_only_leaves_outer_at_zero_not_point_one():
    """`BandScale(padding_inner=0.2)` alone leaves outer padding at 0, not
    Vega-Lite's 0.1: an unset side falls back to `padding`, then to 0 --
    d3's own behavior, not "0.1 unless inner is set". The measured width
    (94.591px) is recorded here for the changelog.
    """
    chart = _bar_chart(fm.X("cat", scale=fm.BandScale(padding_inner=0.2)))
    widths = _bar_widths(chart.to_svg())
    assert widths[0] == pytest.approx(94.591, abs=1e-3)
    s = fm.BandScale(padding_inner=0.2)
    assert s.padding_outer is None  # neither outer nor padding was set
    assert s.padding is None


# ---------------------------------------------------------------------------
# 3. Every band consumer follows the resolved layout (bar/box/heatmap/dodge)
# ---------------------------------------------------------------------------


class TestEveryConsumerFollowsTheUnpaddedLayout:
    """Boxplot, heatmap cell and dodge render identically on an unpadded
    named `BandScale()` and the auto-inferred axis (spec §9).
    """

    def test_boxplot_box(self):
        df = pl.DataFrame(
            {
                "cat": ["a", "b", "c", "d"] * 10,
                "val": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 1.5, 2.5, 3.5] * 4,
            }
        )
        auto = fm.Chart(df).mark_boxplot().encode(x="cat", y="val")
        named = fm.Chart(df).mark_boxplot().encode(x=fm.X("cat", scale=fm.BandScale()), y="val")
        assert named.to_svg() == auto.to_svg()

    def test_heatmap_cell(self):
        df = pl.DataFrame(
            {"row": ["a", "a", "b", "b"], "col": ["x", "y", "x", "y"], "val": [1.0, 2.0, 3.0, 4.0]}
        )
        auto = fm.Chart(df).mark_rect().encode(x="row", y="col", color="val")
        named = (
            fm.Chart(df)
            .mark_rect()
            .encode(
                x=fm.X("row", scale=fm.BandScale()),
                y=fm.Y("col", scale=fm.BandScale()),
                color="val",
            )
        )
        assert named.to_svg() == auto.to_svg()

    def test_dodge(self):
        df = pl.DataFrame(
            {
                "cat": ["a", "a", "b", "b", "c", "c", "d", "d"],
                "grp": ["g1", "g2"] * 4,
                "val": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
            }
        )
        auto = fm.Chart(df).mark_bar(position=fm.Dodge()).encode(x="cat", y="val", color="grp")
        named = (
            fm.Chart(df)
            .mark_bar(position=fm.Dodge())
            .encode(x=fm.X("cat", scale=fm.BandScale()), y="val", color="grp")
        )
        assert named.to_svg() == auto.to_svg()


# ---------------------------------------------------------------------------
# 4. Compute facade agrees with rendering
# ---------------------------------------------------------------------------


def test_facade_bandwidth_inherits_unpadded():
    s = fm.BandScale(domain=list("abcd"), range=[0.0, 400.0])
    assert s.bandwidth() == 100.0


# ---------------------------------------------------------------------------
# 5. Getters, equality, repr (spec §6/§12)
# ---------------------------------------------------------------------------


class TestGettersEqualityRepr:
    def test_unset_getters_return_none(self):
        s = fm.BandScale()
        assert s.padding is None
        assert s.padding_inner is None
        assert s.padding_outer is None

    def test_padding_getter_returns_the_user_value(self):
        assert fm.BandScale(padding=0.3).padding == 0.3

    def test_side_getter_falls_back_to_padding(self):
        s = fm.BandScale(padding=0.3)
        assert s.padding_inner == 0.3
        assert s.padding_outer == 0.3

    def test_side_getter_overrides_padding(self):
        s = fm.BandScale(padding=0.3, padding_inner=0.2)
        assert s.padding_inner == 0.2
        assert s.padding_outer == 0.3

    def test_unset_equals_explicit_none(self):
        assert fm.BandScale() == fm.BandScale(padding=None)
        assert fm.BandScale() == fm.BandScale(padding_inner=None, padding_outer=None)

    def test_unset_is_not_equal_to_explicit_zero(self):
        """`None` (inherits) and an explicit `0.0` resolve to the same
        rendered geometry but are distinct values (spec §12).
        """
        assert fm.BandScale() != fm.BandScale(padding=0.0)
        unset_svg = _bar_chart(fm.X("cat", scale=fm.BandScale())).to_svg()
        zero_svg = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=0.0))).to_svg()
        assert unset_svg == zero_svg, "the two render identically despite being unequal objects"

    def test_repr_prints_args_as_set(self):
        assert repr(fm.BandScale()) == (
            "BandScale(domain=[], padding=None, padding_inner=None, padding_outer=None, align=0.5)"
        )
        assert repr(fm.BandScale(padding=0.0)) == (
            "BandScale(domain=[], padding=0, padding_inner=None, padding_outer=None, align=0.5)"
        )

    def test_point_scale_unset_vs_explicit_default_disclosed(self):
        """`PointScale()` and `PointScale(padding=0.5)` resolve to the same
        rendered geometry (the renderer owns the 0.5 default for an unset
        padding) but are NOT equal, and `repr()` shows the raw unresolved
        value -- disclosed in the `_core.pyi` stub docstring.

        The `padding` getter returns `None` when unset, like every sibling
        scale's getter (`BandPadding`'s getters included) -- it does not
        resolve to the renderer's 0.5 default.
        """
        assert fm.PointScale() != fm.PointScale(padding=0.5)
        assert "padding=None" in repr(fm.PointScale())
        assert "padding=0.5" in repr(fm.PointScale(padding=0.5))
        assert fm.PointScale().padding is None
        assert fm.PointScale(padding=0.0).padding == 0.0


# ---------------------------------------------------------------------------
# 6. Wire: omits unset keys; a dict with explicit padding round-trips
# ---------------------------------------------------------------------------


class TestWire:
    def test_unset_omits_every_padding_key(self):
        chart = _bar_chart(fm.X("cat", scale=fm.BandScale()))
        scale = _x_scale_dict(chart)
        assert "padding" not in scale
        assert "paddingInner" not in scale
        assert "paddingOuter" not in scale

    def test_explicit_padding_dict_round_trips(self):
        chart = _bar_chart(fm.X("cat", scale={"type": "band", "paddingInner": 0.3}))
        scale = _x_scale_dict(chart)
        assert scale["paddingInner"] == 0.3
        assert "paddingOuter" not in scale
        assert "padding" not in scale

    def test_point_scale_unset_omits_padding_key(self):
        chart = fm.Chart(_DF4).mark_point().encode(x=fm.X("cat", scale=fm.PointScale()), y="val")
        assert "padding" not in _x_scale_dict(chart)

    def test_point_scale_explicit_padding_on_wire(self):
        chart = (
            fm.Chart(_DF4)
            .mark_point()
            .encode(x=fm.X("cat", scale=fm.PointScale(padding=0.5)), y="val")
        )
        assert _x_scale_dict(chart)["padding"] == 0.5


def test_explicit_none_padding_is_unset():
    """`BandScale(padding=None)` is indistinguishable from `BandScale()` on
    every surface: equality, wire, and rendering.
    """
    assert fm.BandScale(padding=None) == fm.BandScale()
    a = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=None)))
    b = _bar_chart(fm.X("cat", scale=fm.BandScale()))
    assert a.to_svg() == b.to_svg()
    assert _x_scale_dict(a) == _x_scale_dict(b)


# ---------------------------------------------------------------------------
# 7. Override door: a None scale-padding override is no override
# ---------------------------------------------------------------------------


class TestOverrideNoneIsNoOverride:
    """`.override(x_scale_padding=None)` (and inner/outer) over the chart's
    own scale padding must not erase it: the chart's value stands, as every
    non-mark override path already behaves.
    """

    def test_padding_none_override_leaves_the_chart_value_standing(self):
        base = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=0.3)))
        overridden = base.override(x_scale_padding=None)
        assert overridden.to_svg() == base.to_svg()
        assert _x_scale_dict(overridden)["padding"] == 0.3

    def test_padding_inner_none_override_leaves_the_chart_value_standing(self):
        base = _bar_chart(fm.X("cat", scale=fm.BandScale(padding_inner=0.2, padding_outer=0.1)))
        overridden = base.override(x_scale_padding_inner=None)
        assert overridden.to_svg() == base.to_svg()

    def test_padding_outer_none_override_leaves_the_chart_value_standing(self):
        base = _bar_chart(fm.X("cat", scale=fm.BandScale(padding_inner=0.2, padding_outer=0.1)))
        overridden = base.override(x_scale_padding_outer=None)
        assert overridden.to_svg() == base.to_svg()

    def test_padding_override_with_a_real_value_still_applies(self):
        """The None no-op must not swallow a REAL override value: control
        for the fix above.
        """
        base = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=0.3)))
        overridden = base.override(x_scale_padding=0.2)
        assert overridden.to_svg() != base.to_svg()
        assert _x_scale_dict(overridden)["padding"] == 0.2


class TestOverrideMalformedPaddingRefuses:
    """`.override(x_scale_padding=...)` with an out-of-bounds or wrong-type
    value refuses, naming the override call and the exact path the caller
    wrote -- not the channel or the scale kind, which is what the
    render-time dict-door voice names instead.
    """

    def test_out_of_bounds_padding_1_5_refuses(self):
        base = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=0.3)))
        with pytest.raises(
            ValueError,
            match=r"^Chart\.override\(x_scale_padding=\.\.\.\) must be a number in \[0, 1\]; received 1\.5$",
        ):
            base.override(x_scale_padding=1.5).to_svg()

    def test_out_of_bounds_padding_negative_refuses(self):
        base = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=0.3)))
        with pytest.raises(
            ValueError,
            match=r"^Chart\.override\(x_scale_padding=\.\.\.\) must be a number in \[0, 1\]; received -0\.1$",
        ):
            base.override(x_scale_padding=-0.1).to_svg()

    def test_wrong_type_padding_refuses(self):
        base = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=0.3)))
        with pytest.raises(
            ValueError,
            match=r'^Chart\.override\(x_scale_padding=\.\.\.\) invalid type: string "a", expected f64$',
        ):
            base.override(x_scale_padding="a").to_svg()

    def test_wrong_type_padding_bool_refuses(self):
        base = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=0.3)))
        with pytest.raises(
            ValueError,
            match=r"^Chart\.override\(x_scale_padding=\.\.\.\) invalid type: boolean `true`, expected f64$",
        ):
            base.override(x_scale_padding=True).to_svg()

    def test_out_of_bounds_padding_inner_refuses(self):
        base = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=0.3)))
        with pytest.raises(
            ValueError,
            match=(
                r"^Chart\.override\(x_scale_padding_inner=\.\.\.\) "
                r"must be a number in \[0, 1\]; received 1\.5$"
            ),
        ):
            base.override(x_scale_padding_inner=1.5).to_svg()

    def test_out_of_bounds_padding_outer_refuses(self):
        base = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=0.3)))
        with pytest.raises(
            ValueError,
            match=(
                r"^Chart\.override\(x_scale_padding_outer=\.\.\.\) "
                r"must be a finite number >= 0; received -1$"
            ),
        ):
            base.override(x_scale_padding_outer=-1).to_svg()

    def test_out_of_bounds_padding_through_type_switch_on_point_scale(self):
        """The same named refusal reaches through the override type-switch
        door too: `.override(x_scale_type="band", x_scale_padding=1.5)` on
        a `PointScale()` chart names `x_scale_padding`, the override's own
        leaf, not `x_scale_type`.
        """
        base = _bar_chart(fm.X("cat", scale=fm.PointScale()))
        with pytest.raises(
            ValueError,
            match=(
                r"^Chart\.override\(x_scale_padding=\.\.\.\) "
                r"must be a number in \[0, 1\]; received 1\.5$"
            ),
        ):
            base.override(x_scale_type="band", x_scale_padding=1.5).to_svg()

    def test_masked_padding_still_refuses_via_override(self):
        """A malformed `padding` masked by explicit inner/outer is still
        refused, through the override door too: every overridden value is
        checked individually, not just the ones the resolver reads.
        """
        base = _bar_chart(fm.X("cat", scale=fm.BandScale(padding=0.3)))
        with pytest.raises(
            ValueError,
            match=r"^Chart\.override\(x_scale_padding=\.\.\.\) must be a number in \[0, 1\]; received 1\.5$",
        ):
            base.override(
                x_scale_padding=1.5, x_scale_padding_inner=0.2, x_scale_padding_outer=0.2
            ).to_svg()


# ---------------------------------------------------------------------------
# 8. Class refusals, exactly
# ---------------------------------------------------------------------------


class TestClassRefusalsExactText:
    def test_band_padding_out_of_bounds(self):
        with pytest.raises(
            ValueError,
            match=r"^BandScale\(padding=\.\.\.\) must be a number in \[0, 1\]; received 1\.5$",
        ):
            core.BandScale(padding=1.5)

    def test_band_padding_inner_out_of_bounds(self):
        with pytest.raises(
            ValueError,
            match=r"^BandScale\(padding_inner=\.\.\.\) must be a number in \[0, 1\]; received 1\.5$",
        ):
            core.BandScale(padding_inner=1.5)

    def test_band_padding_outer_negative(self):
        with pytest.raises(
            ValueError,
            match=r"^BandScale\(padding_outer=\.\.\.\) must be a finite number >= 0; received -1$",
        ):
            core.BandScale(padding_outer=-1.0)

    def test_band_align_out_of_bounds(self):
        with pytest.raises(
            ValueError,
            match=r"^BandScale\(align=\.\.\.\) must be a number in \[0, 1\]; received 1\.5$",
        ):
            core.BandScale(align=1.5)

    def test_point_padding_negative(self):
        with pytest.raises(
            ValueError,
            match=r"^PointScale\(padding=\.\.\.\) must be a finite number >= 0; received -1$",
        ):
            core.PointScale(padding=-1.0)

    def test_point_align_out_of_bounds(self):
        with pytest.raises(
            ValueError,
            match=r"^PointScale\(align=\.\.\.\) must be a number in \[0, 1\]; received 1\.5$",
        ):
            core.PointScale(align=1.5)

    def test_masked_padding_still_refuses(self):
        """A malformed `padding` masked by explicit inner/outer is still
        refused: every set value is checked, not just the ones the resolver
        reads.
        """
        with pytest.raises(
            ValueError,
            match=r"^BandScale\(padding=\.\.\.\) must be a number in \[0, 1\]; received 1\.5$",
        ):
            core.BandScale(padding=1.5, padding_inner=0.2, padding_outer=0.1)


# ---------------------------------------------------------------------------
# 9. Dict malformed: typed refusal at render, exactly
# ---------------------------------------------------------------------------


class TestDictMalformedRefusesAtRender:
    def test_padding_inner_out_of_bounds(self):
        chart = _bar_chart(fm.X("cat", scale={"type": "band", "paddingInner": 1.5}))
        with pytest.raises(
            ValueError,
            match=r"^band scale on x: paddingInner must be a number in \[0, 1\]; received 1\.5$",
        ):
            chart.to_svg()

    def test_padding_outer_negative(self):
        chart = _bar_chart(fm.X("cat", scale={"type": "band", "paddingOuter": -1.0}))
        with pytest.raises(
            ValueError,
            match=r"^band scale on x: paddingOuter must be a finite number >= 0; received -1$",
        ):
            chart.to_svg()

    def test_override_type_switch_malformed_padding_refuses_in_the_override_voice(self):
        """A malformed padding leaf reached through the override type-switch
        door refuses in the OVERRIDE's own voice, naming the path the
        caller wrote -- not the dict door's unnamed render-time voice,
        because the value's own origin is the override, not the chart.
        """
        chart = _bar_chart("cat").override(x_scale_type="band", x_scale_paddingInner=1.5)
        with pytest.raises(
            ValueError,
            match=(
                r"^Chart\.override\(x_scale_paddingInner=\.\.\.\) "
                r"must be a number in \[0, 1\]; received 1\.5$"
            ),
        ):
            chart.to_svg()

    def test_chart_own_malformed_dict_padding_under_unrelated_override_keeps_dict_voice(self):
        """The chart's OWN malformed dict padding is not attributed to an
        unrelated override on the same channel: it keeps the dict door's
        exact render-time text, because the malformed VALUE's origin is the
        chart's own `scale=`, not the override.
        """
        chart = _bar_chart(fm.X("cat", scale={"type": "band", "paddingInner": 1.5})).override(
            x_scale_align=0.3
        )
        with pytest.raises(
            ValueError,
            match=r"^band scale on x: paddingInner must be a number in \[0, 1\]; received 1\.5$",
        ):
            chart.to_svg()


# ---------------------------------------------------------------------------
# 10. PointScale / OrdinalScale charts: byte-identical
# ---------------------------------------------------------------------------


class TestPointAndOrdinalUnaffected:
    """#157 touches band padding only; `PointScale` and `OrdinalScale`
    charts render exactly as before.
    """

    def test_point_scale_unset_renders_like_its_resolved_default(self):
        unset = fm.Chart(_DF4).mark_point().encode(x=fm.X("cat", scale=fm.PointScale()), y="val")
        explicit = (
            fm.Chart(_DF4)
            .mark_point()
            .encode(x=fm.X("cat", scale=fm.PointScale(padding=0.5)), y="val")
        )
        assert unset.to_svg() == explicit.to_svg()

    def test_ordinal_scale_dict_renders_like_the_auto_axis(self):
        auto = _bar_chart("cat")
        ordinal = _bar_chart(fm.X("cat", scale={"type": "ordinal"}))
        assert ordinal.to_svg() == auto.to_svg()
