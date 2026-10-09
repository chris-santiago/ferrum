"""Tests for #135: `padding.auto` has exactly one owner.

The owner is the Rust built-in default (`ThemePadding::default().padding_auto`).
Python's `PaddingConfig.auto` and `configure_padding(auto=...)` are
`bool | None = None`, so `auto` reaches the wire only when the caller sets
it -- an unmentioned `auto` always inherits the built-in default, and a
later configure layer that only sets sides can never silently reset an
earlier `auto=True` to off.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum._chrome import merge_configure_layers
from ferrum._core import ChartSpec, compute_layout
from ferrum.configure import (
    AxisConfig,
    ColorConfig,
    Configure,
    GridAxisConfig,
    GridConfig,
    LegendConfig,
    PaddingConfig,
    TitleConfig,
)
from tests._snapshots import decode_png_rgba, rasterize_svg


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _auto_sensitive_chart():
    """A 180x200 scatter over x in 0..1e6 ticked only at its ends, so its
    two currency labels stay flat and the last one ("$1,000,000") overhangs
    the right canvas edge with auto off.

    The auto-sensitive fixture for the None-is-unset tests below:
    `padding.auto` changes a render only when a drawn label or title really
    overhangs (a chart whose labels already fit renders the same either
    way, so it could not tell `None` from `False`).
    `test_the_auto_sensitive_fixture_really_is_auto_sensitive` pins that.
    """
    df = pl.DataFrame({"x": [0.0, 1_000_000.0], "y": [0.0, 1.0]})
    return (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .configure(axis_x=AxisConfig(label_format="currency", tick_values=[0, 1_000_000]))
        .properties(width=180, height=200)
    )


def test_the_auto_sensitive_fixture_really_is_auto_sensitive():
    chart = _auto_sensitive_chart()
    svg_on = chart.configure_padding(auto=True).to_svg()
    svg_off = chart.configure_padding(auto=False).to_svg()
    assert svg_on != svg_off


def _widen_svg_canvas(svg: str, viewport_width: float, viewport_height: float, extra: float) -> str:
    """Return *svg* with its outer `<svg>` `width`/`viewBox` widened by
    *extra* px, leaving every other element -- including any full-canvas
    background `<rect>` -- at the ORIGINAL dimensions.

    The newly added margin therefore stays unpainted (transparent) rather
    than cream-colored, so content that overhangs the original canvas
    paints into that margin instead of being cropped there by the
    rasterizer.
    """
    old = f'width="{viewport_width:g}" height="{viewport_height:g}" viewBox="0 0 {viewport_width:g} {viewport_height:g}"'
    new = (
        f'width="{viewport_width + extra:g}" height="{viewport_height:g}" '
        f'viewBox="0 0 {viewport_width + extra:g} {viewport_height:g}"'
    )
    assert old in svg, f"expected the exact <svg> root dimensions {old!r} in: {svg[:200]!r}"
    return svg.replace(old, new, 1)


def _rightmost_ink_column(svg: str) -> int:
    """The rightmost pixel column, in *svg*'s own rasterized canvas, whose
    color differs from the top-left corner's background AND is not fully
    transparent. Returns ``-1`` if the whole canvas is background."""
    rows, width, _height = decode_png_rgba(rasterize_svg(svg))
    background = rows[0][0]
    rightmost = -1
    for row in rows:
        for x in range(width - 1, rightmost, -1):
            if row[x] != background and row[x][3] != 0:
                rightmost = x
                break
    return rightmost


def _last_x_tick_overhang_margin(svg: str, viewport_width: float, viewport_height: float) -> float:
    """Right-edge margin (px) of *svg*'s rendered content, measured from
    the REAL rendered geometry, not an approximated text metric.

    `to_svg()` lays text out with `FontdueMetrics` (`render/font.rs`, real
    Inter glyph advances) -- not `HeuristicMetrics` (`layout/binding.rs`'s
    `chars * font_size * 0.6` estimate), which belongs only to the
    `compute_layout` PyO3 binding and does not match this render path's
    geometry. This widens the canvas (`_widen_svg_canvas`) before
    rasterizing with `resvg_py`, so a label that overhangs the ORIGINAL
    canvas paints into the newly added, otherwise-transparent margin
    instead of being cropped there -- which is what lets the result go
    negative (a raster bounded at the original canvas size cannot show a
    label sitting PAST its edge, only AT it).
    """
    widened = _widen_svg_canvas(svg, viewport_width, viewport_height, extra=viewport_width)
    rightmost = _rightmost_ink_column(widened)
    return viewport_width - 1 - rightmost


def _assert_overhang_then_rescue(
    svg_auto_false: str, svg_default: str, viewport_width: float, viewport_height: float
) -> tuple[float, float]:
    """Assert the #135 rescue property in order: a real overhang with auto
    off, THEN the label back inside the canvas by default. No tolerance
    window on either side. Returns ``(margin_false, margin_default)`` for
    the caller to pin exact values.
    """
    margin_false = _last_x_tick_overhang_margin(svg_auto_false, viewport_width, viewport_height)
    assert margin_false < 0, (
        f"auto=False must genuinely overhang the canvas (margin < 0); got "
        f"{margin_false}px -- this fixture is not a real clipping repro"
    )
    margin_default = _last_x_tick_overhang_margin(svg_default, viewport_width, viewport_height)
    assert margin_default >= 0, (
        f"the default (built-in auto) render must keep the label inside "
        f"the canvas (margin >= 0); got {margin_default}px"
    )
    return margin_false, margin_default


# ---------------------------------------------------------------------------
# Case: no padding config at all, continuous x ending in a long label that
# genuinely overhangs the canvas at auto=False -> the built-in default
# rescues it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label_format, expected_margin_false, expected_margin_default, auto_false_hash",
    [
        ("integer", -1, 4, "5e4712f904b6a6d9"),
        ("currency", -4, 5, "98ffb74f4ae488f2"),
    ],
)
def test_integer_and_currency_label_format_on_0_to_1e6_is_rescued(
    label_format, expected_margin_false, expected_margin_default, auto_false_hash
):
    """#135 headline case: a quantitative 0..1e6 x axis whose last tick
    label genuinely overhangs the 180px canvas at `auto=False` (measured
    margin below, from the real rendered geometry), and sits back inside
    it by default -- whether that default comes from never calling
    `configure_padding` at all, or from calling it with no `auto=` --
    which matches an explicit `auto=True`, not `auto=False`."""
    width, height = 180, 200
    df = pl.DataFrame({"x": [0.0, 1_000_000.0], "y": [0.0, 1.0]})
    base = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .configure(axis_x=AxisConfig(label_format=label_format, tick_values=[0, 1_000_000]))
        .properties(width=width, height=height)
    )
    svg_no_config = base.to_svg()
    # The default comes from a `configure_padding()` call that never
    # mentions `auto`, which is the Python-side regression #135 fixes: the
    # old `bool = False` default forced every such bare call to the
    # auto=False behavior, silently resetting the built-in default.
    svg_bare_call = base.configure_padding().to_svg()
    svg_auto_true = base.configure_padding(auto=True).to_svg()
    svg_auto_false = base.configure_padding(auto=False).to_svg()

    # Overhang first, then within -- no tolerance window on either side.
    margin_false, margin_default = _assert_overhang_then_rescue(
        svg_auto_false, svg_bare_call, width, height
    )
    # Measured (not merely asserted sign): auto=False overhangs by a real
    # amount, and the default sits a real amount inside the canvas.
    assert margin_false == expected_margin_false
    assert margin_default == expected_margin_default

    # The default render equals the explicit auto=True render, both ways
    # ("no config at all" and "configure_padding() with no auto=") -- and
    # differs from auto=False.
    assert svg_bare_call == svg_auto_true
    assert svg_no_config == svg_auto_true
    assert svg_no_config != svg_auto_false

    # Byte-stability pin for the auto=False render: a hash computed on the
    # currently-installed extension, not a claim of equality with any
    # earlier build of that extension -- that comparison would need the
    # earlier build itself, which is not available here. An explicit
    # `auto=False` call always sends `auto=False` on the wire regardless
    # of what Python's own default for an unset `auto` is, so the hash is
    # stable across a change to that default on this extension build.
    assert _sha16(svg_auto_false) == auto_false_hash


# ---------------------------------------------------------------------------
# Cases: a later configure layer that only sets sides, or that explicitly
# turns auto off, composes correctly with an earlier `auto=True` layer --
# the no-reset guarantee #135 exists for.
# ---------------------------------------------------------------------------


class TestConfigureLayerCascadeNeverSilentlyResetsAuto:
    @staticmethod
    def _repro_chart():
        df = pl.DataFrame({"x": [0.0, 999999999999999.0], "y": [0.0, 1.0]})
        return (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .configure_axis(label_format="integer", tick_values=[0, 999999999999999])
        )

    def test_auto_true_then_a_later_call_setting_only_a_side_keeps_auto_on(self):
        """`configure_padding(auto=True).configure_padding(left=5)` must
        render the SAME as the single combined call
        `configure_padding(auto=True, left=5)` -- the second call, which
        never mentions `auto`, must not reset it."""
        base = self._repro_chart().properties(width=400, height=200)
        chained = base.configure_padding(auto=True).configure_padding(left=5).to_svg()
        combined = base.configure_padding(auto=True, left=5).to_svg()
        reset_off = base.configure_padding(auto=True).configure_padding(left=5, auto=False).to_svg()
        assert chained == combined
        assert chained != reset_off

    def test_a_later_explicit_auto_false_overrides_an_earlier_auto_true(self):
        base = self._repro_chart().properties(width=400, height=200)
        chained_then_off = base.configure_padding(auto=True).configure_padding(auto=False).to_svg()
        plain_off = base.configure_padding(auto=False).to_svg()
        assert chained_then_off == plain_off


# ---------------------------------------------------------------------------
# Case: a composite root's `auto=True` cascades into a leaf that sets its
# own sides, without the leaf's own padding layer resetting it.
# ---------------------------------------------------------------------------


def test_composite_root_auto_true_cascades_into_a_leaf_that_sets_its_own_sides():
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    leaf = fm.Chart(df).mark_point().encode(x="x", y="y").configure_padding(left=5)
    other = fm.Chart(df).mark_point().encode(x="x", y="y")
    root = (leaf | other).configure_padding(auto=True)

    injected_leaf = root._inject_parent_config(root.charts[0])
    assert injected_leaf._resolve_chart_config()["padding"] == {"auto": True, "left": 5.0}


# ---------------------------------------------------------------------------
# Case: `PaddingConfig()`'s own shape -- `auto` unset by default, omitted
# from `to_dict()` -- and `auto=None` on both the class and
# `configure_padding` doors.
# ---------------------------------------------------------------------------


class TestPaddingConfigAutoShape:
    def test_default_construction_auto_is_none_and_omitted(self):
        cfg = PaddingConfig()
        assert cfg.auto is None
        assert "auto" not in cfg.to_dict()

    def test_explicit_auto_none_is_the_same_as_unset(self):
        assert PaddingConfig(auto=None).to_dict() == PaddingConfig().to_dict() == {}

    def test_configure_padding_auto_none_is_unset(self):
        """`configure_padding(auto=None)` must render the SAME as never
        calling `configure_padding` at all. Pinned on the auto-sensitive
        fixture, where `auto` actually changes the render -- a 2-point
        scatter would pass this even if `None` were silently coerced to
        `False` on this door."""
        base = _auto_sensitive_chart()
        svg_none = base.configure_padding(auto=None).to_svg()
        svg_no_call = base.to_svg()
        assert svg_none == svg_no_call

    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_bool_and_numpy_bool_are_accepted(self, value):
        cfg = PaddingConfig(auto=value)
        assert cfg.auto == bool(value)
        assert type(cfg.auto) is bool

    @pytest.mark.parametrize("bad", ["yes", 1, 0])
    def test_everything_else_is_refused_never_by_truthiness(self, bad):
        """`"yes"`, `1`, and `0` are each truthy/falsy stand-ins for `True`/
        `False` -- refusing them (rather than reading their truthiness)
        proves `auto` is coerced through a real bool check, not `bool(value)`."""
        with pytest.raises(ValueError, match=r"^padding\.auto must be True or False; received"):
            PaddingConfig(auto=bad)

    @pytest.mark.parametrize("bad", ["yes", 1, 0])
    def test_configure_padding_refuses_the_same_way(self, bad):
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y")
        with pytest.raises(ValueError, match=r"^padding\.auto must be True or False; received"):
            chart.configure_padding(auto=bad)

    @pytest.mark.parametrize("bad", ["yes", 1, 0])
    def test_override_refuses_in_the_override_doors_own_voice(self, bad):
        """`.override(padding_auto=...)` names the override path, not
        `PaddingConfig` or `configure_padding` -- the caller never wrote
        either of those. The override is validated at render time."""
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y").override(padding_auto=bad)
        with pytest.raises(
            ValueError, match=r"^override 'padding_auto' must be True or False; received"
        ):
            chart.to_svg()


# ---------------------------------------------------------------------------
# Case: `.override(padding_auto=...)` -- `None` is no override (the chart's
# own value, or the built-in default, stands); `False` turns it off.
# ---------------------------------------------------------------------------


class TestOverridePaddingAuto:
    @staticmethod
    def _chart():
        """The auto-sensitive fixture, where `auto` actually changes the
        render -- a chart `auto` has no visible effect on would pass every
        test below even if `None` were silently turned into `False` on
        either door."""
        return _auto_sensitive_chart()

    def test_override_none_is_unset_and_matches_no_config(self):
        chart = self._chart()
        with pytest.warns(DeprecationWarning, match="padding_auto"):
            svg_override_none = chart.override(padding_auto=None).to_svg()
        svg_no_config = chart.to_svg()
        assert svg_override_none == svg_no_config

    def test_override_none_does_not_erase_a_chart_own_auto_true(self):
        """A `None` override leaves an earlier value in place: a chart that
        already set `auto=True` via `configure_padding` keeps it."""
        chart = self._chart().configure_padding(auto=True)
        with pytest.warns(DeprecationWarning, match="padding_auto"):
            svg_override_none = chart.override(padding_auto=None).to_svg()
        assert svg_override_none == chart.to_svg()

    def test_override_false_turns_it_off(self):
        chart = self._chart()
        with pytest.warns(DeprecationWarning, match="padding_auto"):
            svg_override_false = chart.override(padding_auto=False).to_svg()
        svg_configure_false = chart.configure_padding(auto=False).to_svg()
        assert svg_override_false == svg_configure_false


# ---------------------------------------------------------------------------
# Case: every `ferrum.configure` dataclass's empty `to_dict()` is `{}` --
# the structural invariant that makes deleting the
# `_chart_config_wire_fragment` baseline-subtraction workaround safe (#135).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        AxisConfig,
        LegendConfig,
        TitleConfig,
        GridAxisConfig,
        GridConfig,
        PaddingConfig,
        ColorConfig,
        Configure,
    ],
)
def test_every_configure_dataclass_empty_to_dict_is_empty(cls):
    assert cls().to_dict() == {}


# ---------------------------------------------------------------------------
# Case: the configure-layer merge used by rendering (`_RenderMixin._resolve_chart_config`)
# and by chrome (`composition.py`'s `_composite_chrome_kwargs`) is one
# function (`ferrum._chrome.merge_configure_layers`), giving the same result
# on both paths.
# ---------------------------------------------------------------------------


def test_render_and_chrome_merge_paths_share_one_function_and_agree():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .configure_padding(auto=True)
        .configure_padding(left=5)
    )
    # The render path's own merge.
    render_merged = chart._resolve_chart_config()
    # The chrome path's merge, called directly on the SAME layer list.
    chrome_merged = merge_configure_layers(chart._configure)
    assert render_merged["padding"] == chrome_merged["padding"] == {"auto": True, "left": 5.0}


# ---------------------------------------------------------------------------
# Case: `fm.compute_layout` always builds `ThemeInputs::default()`
# internally and has no `auto=` argument of its own, so the only way its
# output can show `padding.auto`'s effect is the Rust built-in default
# itself.
# ---------------------------------------------------------------------------


def test_compute_layout_inherits_the_built_in_auto_default():
    """A y-axis title just past its viewport's half-extent -- feasible, but
    needing a genuine recenter -- shifts the panel's plot area under
    `fm.compute_layout`'s own (uncustomizable) default theme: a titled and
    an untitled call to `compute_layout` return different plot areas for
    this chart.

    `compute_layout` has no `auto=` argument, so this alone shows only that
    auto's own recentering mechanism is active with whatever theme this
    binding builds -- not, by itself, that the built-in default is
    specifically `True`. The negative control below checks the same thing
    through the one API that CAN toggle `auto` (the full `Chart` path), on
    the title fixture already pinned in
    `tests/test_padding_validation.py::TestPaddingAutoAxisTitleRecentering`:
    `auto=True` and `auto=False` render differently there.
    """
    spec = ChartSpec(mark="point", x="a", y="b")
    baseline = compute_layout(
        spec, viewport=(300.0, 150.0), x_tick_labels=["0", "1", "2"], y_tick_labels=["0", "5"]
    )["panels"][0]["plot_area"]
    corrected = compute_layout(
        spec,
        viewport=(300.0, 150.0),
        x_tick_labels=["0", "1", "2"],
        y_tick_labels=["0", "5"],
        y_title="X" * 18,
    )["panels"][0]["plot_area"]
    assert baseline == {"x": 28.6, "y": 16.0, "w": 255.4, "h": 98.8}
    assert corrected["y"] != baseline["y"], (
        f"a y title in the recentering window should shift the plot area "
        f"away from the untitled baseline ({corrected} vs {baseline})"
    )

    # Negative control, through the full Chart API (which CAN toggle
    # `auto`): the same title-recentering mechanism, on the fixture already
    # pinned in
    # tests/test_padding_validation.py::TestPaddingAutoAxisTitleRecentering
    # to discriminate auto=True from auto=False via real font metrics.
    control_df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]})
    control_title = "A moderately long horizontal axis title here"
    control_chart = (
        fm.Chart(control_df)
        .mark_point()
        .encode(x=fm.X("x", title=control_title), y="y")
        .properties(width=270, height=200)
    )
    svg_auto_true = control_chart.configure_padding(auto=True).to_svg()
    svg_auto_false = control_chart.configure_padding(auto=False).to_svg()
    assert svg_auto_true != svg_auto_false, (
        "auto=True and auto=False must render differently for this "
        "title-recentering fixture through the full Chart API"
    )


# ---------------------------------------------------------------------------
# Case: auto-rotated, culled x labels that already fit the canvas leave the
# chart exactly as auto=False draws it -- padding.auto reserves only the
# margin the DRAWN labels need, not the flat half-width of a label the
# collision cascade rotated or culled.
# ---------------------------------------------------------------------------


_X_TICK_LABEL = re.compile(
    r'<text [^>]*font-size="11" text-anchor="end" transform="rotate\((-?[\d.]+) [^"]*"[^>]*>([^<]*)</text>'
)


_TEXT = re.compile(r"<text ([^>]*)>([^<]*)</text>")


def _drawn_x_tick_labels(svg: str) -> list[tuple[str, str, str]]:
    """The x tick labels *svg* draws, in order, as ``(text, font-size,
    rotation)``. An x label is centred when flat, or carries a rotate
    transform; the y tick labels are end-anchored with no transform."""
    drawn = []
    for attrs, text in _TEXT.findall(svg):
        if not text.startswith("$"):
            continue
        rotate = re.search(r'transform="rotate\((-?[\d.]+) ', attrs)
        if 'text-anchor="middle"' not in attrs and rotate is None:
            continue
        size = re.search(r'font-size="([\d.]+)"', attrs)
        drawn.append((text, size.group(1) if size else "", rotate.group(1) if rotate else "0"))
    return drawn


class TestRotatedLabelsThatFitAreUntouched:
    WIDTH, HEIGHT = 180, 200

    @classmethod
    def _chart(cls):
        df = pl.DataFrame({"x": [0.0, 5e5, 1e6], "y": [0.0, 1.0, 2.0]})
        return (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .properties(width=cls.WIDTH, height=cls.HEIGHT)
            .configure_axis(label_format="currency")
        )

    def test_default_is_byte_identical_to_auto_false(self):
        chart = self._chart()
        svg_default = chart.to_svg()
        svg_auto_false = chart.configure_padding(auto=False).to_svg()
        # Fixture assumptions: the cascade rotated the x labels to -90 and
        # culled to every third tick (the long "$1,000,000" is not drawn),
        # and the drawn labels fit the canvas with auto off.
        assert _X_TICK_LABEL.findall(svg_auto_false) == [
            ("-90", "$0"),
            ("-90", "$300,000"),
            ("-90", "$600,000"),
            ("-90", "$900,000"),
        ]
        assert _last_x_tick_overhang_margin(svg_auto_false, self.WIDTH, self.HEIGHT) >= 0
        assert _sha16(svg_default) == _sha16(svg_auto_false)

    @pytest.mark.parametrize(
        "width",
        [
            165,  # 160-169: -90, cull stride flips
            200,  # 195-209
            310,  # 305-319
            590,  # 585-599: -90 / -60
            697,  # 690-704: -60 / -45
        ],
    )
    def test_cascade_width_bands_keep_auto_false(self, width):
        """One real-font width per cascade band where the old flat
        half-width extra flipped the label count or angle: the drawn labels
        fit with auto off, so the default is byte-identical to
        `auto=False`."""
        chart = self._chart().properties(width=width)
        svg_auto_false = chart.configure_padding(auto=False).to_svg()
        assert _last_x_tick_overhang_margin(svg_auto_false, width, self.HEIGHT) >= 0
        assert _sha16(chart.to_svg()) == _sha16(svg_auto_false)

    def test_flat_band_default_ticks_draw_the_same_labels_as_auto_false(self):
        """The 725-739 cascade band on this class's chart itself (x at 0,
        5e5 and 1e6, default ticks, currency): at 730px the cascade draws every label flat
        at a shrunk font size. The last label's ADVANCE reaches a fraction
        of a pixel past the canvas (its ink stays inside), so the default
        rescues it and differs from `auto=False`, but it must draw exactly
        the same labels, at the same angles and font sizes."""
        width = 730
        chart = self._chart().properties(width=width)
        svg_default = chart.to_svg()
        svg_auto_false = chart.configure_padding(auto=False).to_svg()
        drawn_off = _drawn_x_tick_labels(svg_auto_false)
        assert len(drawn_off) > 10, drawn_off
        assert svg_default != svg_auto_false
        assert _drawn_x_tick_labels(svg_default) == drawn_off

    def test_flat_band_label_whose_ink_overhangs_is_rescued(self):
        """The 725-739 flat band, on a chart whose flat last label's INK
        (not just its advance) overhangs with auto off: the ends-only
        currency chart at 730px. The default must end it inside the canvas,
        so removing the rescue, or granting too small an extra, fails."""
        width = 730
        chart = (
            self._chart()
            .properties(width=width)
            .configure(axis_x=AxisConfig(tick_values=[0, 1_000_000]))
        )
        svg_default = chart.to_svg()
        svg_auto_false = chart.configure_padding(auto=False).to_svg()
        assert _X_TICK_LABEL.findall(svg_auto_false) == []  # flat labels
        margins = _assert_overhang_then_rescue(svg_auto_false, svg_default, width, self.HEIGHT)
        # Pinned exactly: the last tick moves 1:1 with the right extra, so a
        # half-sized extra would land near +2, not +5.
        assert margins == (-1, 5)

    def test_a_flat_label_that_overhangs_is_still_rescued(self):
        """Vacuity control: the same chart ticked only at its ends keeps
        its two labels flat, the flat "$1,000,000" does overhang with auto
        off, and the default moves it back inside, so the identity above is
        not an inert `padding.auto`."""
        chart = self._chart().configure(axis_x=AxisConfig(tick_values=[0, 1_000_000]))
        svg_default = chart.to_svg()
        svg_auto_false = chart.configure_padding(auto=False).to_svg()
        _assert_overhang_then_rescue(svg_auto_false, svg_default, self.WIDTH, self.HEIGHT)
        assert svg_default != svg_auto_false


# ---------------------------------------------------------------------------
# Case: padding.auto never blanks a cramped chart that auto=False draws. An
# extra whose plot collapses (rotated labels' band eating the height) does
# not count as a rescue, so auto adds nothing there.
# ---------------------------------------------------------------------------


_CLIP_RECT = re.compile(
    r'<clipPath id="[^"]*"><rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)"/>'
)


def test_a_cramped_chart_is_never_blanked_by_the_default():
    """320x150, a long numeric first label (`,.12f` of -1234.56789) at a
    tick placed by `values=`: with auto off the plot is 241x12 and draws
    both points. The extra that would contain the label rotates it into a
    band taller than the chart, collapsing the plot. The default must not
    accept that: it draws both points in auto=False's plot rect."""
    lo = -1234.56789
    df = pl.DataFrame({"x": [lo, 0.0], "y": [0.0, 1.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x:Q", axis=fm.Axis(label_format=",.12f", values=[lo, 0.0])), y="y:Q")
        .properties(width=320, height=150)
    )
    svg_auto_false = chart.configure_padding(auto=False).to_svg()
    svg_default = chart.to_svg()
    assert svg_auto_false.count("<circle") == 2
    assert svg_default.count("<circle") == 2
    assert _CLIP_RECT.findall(svg_default)[:1] == _CLIP_RECT.findall(svg_auto_false)[:1]
