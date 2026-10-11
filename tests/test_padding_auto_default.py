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
from pathlib import Path

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
from tests._snapshots import _load_resvg, decode_png_rgba, rasterize_svg


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
        ("integer", -1, 4, "ffaa50727bf3dfd3"),
        ("currency", -4, 5, "545e9b2676ed1df4"),
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

    def test_auto_false_then_a_later_call_setting_only_a_side_stays_off(self):
        """`configure_padding(auto=False).configure_padding(left=5)` must
        render the SAME as the single combined call
        `configure_padding(auto=False, left=5)` -- the second call, which
        never mentions `auto`, must not turn it back on. The symmetric
        case to ``test_auto_true_then_a_later_call_setting_only_a_side_keeps_auto_on``
        above, with the starting value flipped."""
        base = self._repro_chart().properties(width=400, height=200)
        chained = base.configure_padding(auto=False).configure_padding(left=5).to_svg()
        combined = base.configure_padding(auto=False, left=5).to_svg()
        turned_on = base.configure_padding(auto=False).configure_padding(left=5, auto=True).to_svg()
        assert chained == combined
        assert chained != turned_on


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
# Case: a composite root's `auto=False` cascades into a leaf whose own
# padding layer leaves `auto` unset (`None`) -- the leaf must not silently
# fall back to the built-in default (on); the parent's `False` stands.
# ---------------------------------------------------------------------------


def test_composite_root_auto_false_cascades_into_a_leaf_with_auto_unset():
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    leaf = fm.Chart(df).mark_point().encode(x="x", y="y").configure_padding(left=5)
    other = fm.Chart(df).mark_point().encode(x="x", y="y")
    root = (leaf | other).configure_padding(auto=False)

    injected_leaf = root._inject_parent_config(root.charts[0])
    assert injected_leaf._resolve_chart_config()["padding"] == {"auto": False, "left": 5.0}


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
    def test_override_refuses_in_the_chart_override_voice(self, bad):
        """`.override(padding_auto=...)` names `Chart.override`, the public
        call, not `PaddingConfig` or `configure_padding` -- the caller
        never wrote either of those. The override is validated at render
        time, and the raised type is `FerrumOverrideError` (re-voiceable by
        a figure that wraps `.override` on the caller's behalf), not a
        plain `ValueError`."""
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y").override(padding_auto=bad)
        with pytest.raises(
            fm.FerrumOverrideError,
            match=rf"^Chart\.override\(padding_auto=\.\.\.\) must be True or False; received {bad!r}$",
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

    @pytest.mark.parametrize("value", [np.bool_(True), np.bool_(False)])
    def test_override_numpy_bool_is_byte_equal_to_the_plain_bool_configure_padding(self, value):
        """`numpy.bool_` is accepted on the override door too (`_as_bool`'s
        recognition rule, shared by `coerce_bool_value`), and renders
        byte-identically to the plain-`bool` `configure_padding(auto=...)`
        call on the auto-sensitive fixture, where `auto` actually moves
        pixels."""
        chart = self._chart()
        with pytest.warns(DeprecationWarning, match="padding_auto"):
            svg_override_numpy_bool = chart.override(padding_auto=value).to_svg()
        svg_configure_plain_bool = chart.configure_padding(auto=bool(value)).to_svg()
        assert svg_override_numpy_bool == svg_configure_plain_bool


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
    control_df = pl.DataFrame({"x": [0.0, 10.0], "y": [0.0, 1.0]})
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
        # The rescue extra, pinned exactly rather than left to the `!=` above.
        clip_off = _CLIP_RECT.findall(svg_auto_false)
        clip_default = _CLIP_RECT.findall(svg_default)
        assert clip_off == [("59.727", "16", "654.273", "124.961")]
        assert clip_default == [("59.727", "16", "653.876", "124.961")]

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
# Case: padding.auto never spends more than half of the auto=False plot, in
# either dimension, AND never ships a pass whose x axis loses more of its
# own generated ticks (a label culled or elided) than auto=False loses on
# that axis. On a continuous x axis an even stride thinning -- no label
# elided, the drawn labels exactly every s-th tick, at least 3 of them -- is
# not a loss. A rescue that passes both is applied in full. One that fails
# either is declined entirely, so the default renders exactly like
# auto=False -- not a smaller plot that still clips, or one that drops its
# labels to fit.
# ---------------------------------------------------------------------------


_CLIP_RECT = re.compile(
    r'<clipPath id="[^"]*"><rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)"/>'
)

_FLAT_NUMERIC_X_TICK_LABEL = re.compile(
    r'<text (?![^>]*transform=)[^>]*text-anchor="middle"[^>]*>(-?[\d,]+\.\d+)</text>'
)


def test_a_rescue_within_half_the_plot_is_applied():
    """192x200, a `,.8f`-formatted last label (1234.56789): auto=False draws
    both flat labels ("0.00000000" and "1,234.56789000", already at the
    shrunk 9.02pt font) in a 113.056x124.961 plot, but the last one
    overhangs the canvas by 8px of real rendered ink. Narrowing the plot to
    98.975x124.961 -- still 87.5% of the auto=False width, well above half
    in both dimensions -- pulls it back onto the canvas (6px to spare)
    without dropping either label, so the default applies the rescue in
    full: same two flat labels, same font, narrower plot, no overhanging
    ink."""
    lo = -float("1.23456789e3")
    df = pl.DataFrame({"x": [0.0, -lo], "y": [0.0, 1.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x:Q", axis=fm.Axis(label_format=",.8f", values=[0.0, -lo])), y="y:Q")
        .properties(width=192, height=200)
    )
    svg_auto_false = chart.configure_padding(auto=False).to_svg()
    svg_default = chart.to_svg()
    assert svg_auto_false.count("<circle") == 2
    assert svg_default.count("<circle") == 2
    drawn_off = _FLAT_NUMERIC_X_TICK_LABEL.findall(svg_auto_false)
    drawn_default = _FLAT_NUMERIC_X_TICK_LABEL.findall(svg_default)
    assert drawn_off == ["0.00000000", "1,234.56789000"]
    assert drawn_default == drawn_off
    clip_off = _CLIP_RECT.findall(svg_auto_false)
    clip_default = _CLIP_RECT.findall(svg_default)
    assert clip_off == [("62.944", "16", "113.056", "124.961")]
    assert clip_default == [("62.944", "16", "98.975", "124.961")]
    off_w, off_h = (float(v) for v in clip_off[0][2:4])
    default_w, default_h = (float(v) for v in clip_default[0][2:4])
    assert default_w >= off_w / 2
    assert default_h >= off_h / 2
    assert svg_default != svg_auto_false
    assert _last_x_tick_overhang_margin(svg_auto_false, 192, 200) == -8
    assert _last_x_tick_overhang_margin(svg_default, 192, 200) == 6


def _flat_x_tick_labels(svg: str) -> list[tuple[str, str]]:
    """Every flat (centred, unrotated) numeric x tick label *svg* draws, as
    ``(text, font-size)``. The y tick labels are end-anchored and the axis
    titles are not numeric, so neither matches."""
    labels = []
    for attrs, text in _TEXT.findall(svg):
        if (
            'text-anchor="middle"' not in attrs
            or "transform=" in attrs
            or not re.fullmatch(r"-?[\d,]+(\.\d+)?", text)
        ):
            continue
        size = re.search(r'font-size="([\d.]+)"', attrs)
        assert size is not None, f"x tick label {text!r} carries no font-size"
        labels.append((text, size.group(1)))
    return labels


_INTER = Path(__file__).resolve().parents[1] / "crates/ferrum-core/assets/fonts/Inter-Regular.ttf"


def _inter_ink_columns(svg: str, viewport_width: float, viewport_height: float) -> tuple[int, int]:
    """The leftmost and rightmost columns of *svg*'s ink, in canvas pixels
    (negative or past the width when it overhangs), rasterized with the
    bundled Inter the layout measures with, so a title or label's real ink
    is what is measured, not a host fallback font's. The canvas is widened
    by its own width on both sides so overhanging ink is painted, not
    cropped; the background is excluded."""
    w, h = viewport_width, viewport_height
    root = f'width="{w:g}" height="{h:g}" viewBox="0 0 {w:g} {h:g}"'
    assert root in svg
    widened = svg.replace(
        root, f'width="{3 * w:g}" height="{h:g}" viewBox="{-w:g} 0 {3 * w:g} {h:g}"', 1
    )
    png = _load_resvg().svg_to_bytes(
        svg_string=widened, font_files=[str(_INTER)], skip_system_fonts=True, dpi=96
    )
    rows, _width, _height = decode_png_rgba(bytes(png))
    background = rows[0][int(w) + 1]
    columns = [x for row in rows for x, px in enumerate(row) if px[3] != 0 and px != background]
    return min(columns) - int(w), max(columns) - int(w)


def test_a_title_rescue_that_thins_the_x_labels_evenly_ships():
    """270x200, x=[0,1,2,3] with the title "A moderately long horizontal
    axis title here": auto=False draws 8 rotated -90 x tick labels (every
    second of the 16 ticks 0, 0.2, ..., 3) in a 191.056x110.056 plot, and the
    title's ink overhangs the canvas by a measured 8px. The rescue lays the
    same 16 ticks flat and labels every third (0, 0.6, ..., 3): an even
    stride thinning of a continuous axis, 6 labels, none elided, so it is
    not a loss. It ships: a 151.999x124.961 plot (79.6% of the auto=False
    width) with the title's overhang pulled back to +11px."""
    control_df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]})
    chart = (
        fm.Chart(control_df)
        .mark_point()
        .encode(x=fm.X("x", title="A moderately long horizontal axis title here"), y="y")
        .properties(width=270, height=200)
    )
    svg_auto_false = chart.configure_padding(auto=False).to_svg()
    svg_default = chart.to_svg()
    assert svg_auto_false.count("<circle") == 4
    assert svg_default.count("<circle") == 4
    assert _X_TICK_LABEL.findall(svg_auto_false) == [
        ("-90", text) for text in ("0", "0.4", "0.8", "1.2", "1.6", "2", "2.4", "2.8")
    ]
    assert _X_TICK_LABEL.findall(svg_default) == []
    assert _flat_x_tick_labels(svg_default) == [
        (text, "9.02") for text in ("0", "0.6", "1.2", "1.8", "2.4", "3")
    ]
    assert "\u2026" not in svg_default
    clip_off = _CLIP_RECT.findall(svg_auto_false)
    clip_default = _CLIP_RECT.findall(svg_default)
    assert clip_off == [("62.944", "16", "191.056", "110.056")]
    assert clip_default == [("62.944", "16", "151.999", "124.961")]
    assert float(clip_default[0][2]) >= float(clip_off[0][2]) / 2
    assert _last_x_tick_overhang_margin(svg_auto_false, 270, 200) == -8
    assert _last_x_tick_overhang_margin(svg_default, 270, 200) == 11


def test_a_rescue_that_would_cull_every_x_label_is_declined():
    """124x150, x=[-1234567.89, 0] ticked at both values with `,.4f`:
    auto=False draws one x label, "-1,234,567.8900" at -45 degrees, in a
    45.056x9.756 plot (the second tick's "0.0000" is already culled), its
    Inter ink reaching x=2 at the canvas's left edge. A rescue here would
    cull that label too, losing more of its own ticks than auto=False, so
    the default ships byte-identical to auto=False.

    Liveness: padding.auto does act on this chart's long first label. The
    same chart at 176x150 keeps both labels with auto=False and is rescued
    by default, its left inset growing from 62.944 to 66.038 with the same
    two labels drawn; only the cull at 124x150 stops the rescue."""
    lo = -1.23456789e6
    df = pl.DataFrame({"x": [lo, 0.0], "y": [0.0, 1.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x:Q", axis=fm.Axis(label_format=",.4f", values=[lo, 0.0])), y="y:Q")
        .properties(width=124, height=150)
    )
    svg_auto_false = chart.configure_padding(auto=False).to_svg()
    svg_default = chart.to_svg()
    assert re.findall(
        r'transform="rotate\((-?[\d.]+) [^"]*"[^>]*>(-1,234,567\.8900)</text>', svg_auto_false
    ) == [("-45", "-1,234,567.8900")]
    assert ">0.0000<" not in svg_auto_false
    assert _CLIP_RECT.findall(svg_auto_false) == [("62.944", "16", "45.056", "9.756")]
    assert _inter_ink_columns(svg_auto_false, 124, 150)[0] == 2
    assert svg_default == svg_auto_false

    wider = chart.properties(width=176)
    wider_off = wider.configure_padding(auto=False).to_svg()
    wider_default = wider.to_svg()
    assert _CLIP_RECT.findall(wider_off) == [("62.944", "16", "97.056", "9.756")]
    assert _CLIP_RECT.findall(wider_default) == [("66.038", "16", "93.962", "9.756")]
    for svg in (wider_off, wider_default):
        assert ">-1,234,567.8900<" in svg
        assert ">0.0000<" in svg


def test_a_rescue_that_thins_the_x_labels_to_two_is_declined():
    """176x100, x=[0, 1234567.89] ticked at both values with `,.4f`: auto=False
    draws both flat labels ("0.0000" and "1,234,567.8900") in a 97.056x24.961
    plot, and the last one overhangs the canvas by a measured 7px. A rescue
    would keep fewer of its own ticks than auto=False, and an even thinning
    down to 2 labels is still a loss (an even thinning needs at least 3), so
    the default ships byte-identical to auto=False, label overhang and all."""
    hi = 1.23456789e6
    df = pl.DataFrame({"x": [0.0, hi], "y": [0.0, 1.0], "g": ["a", "b"]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x:Q", axis=fm.Axis(label_format=",.4f", values=[0.0, hi])), y="y:Q")
        .properties(width=176, height=100)
    )
    svg_auto_false = chart.configure_padding(auto=False).to_svg()
    svg_default = chart.to_svg()
    assert _flat_x_tick_labels(svg_auto_false) == [
        ("0.0000", "9.02"),
        ("1,234,567.8900", "9.02"),
    ]
    assert _CLIP_RECT.findall(svg_auto_false) == [("62.944", "16", "97.056", "24.961")]
    assert _last_x_tick_overhang_margin(svg_auto_false, 176, 100) == -7
    assert svg_default == svg_auto_false


def test_a_rescue_that_would_exceed_half_the_plot_is_declined():
    """128x200, a `,.8f`-formatted last label (1234.56789) under
    `label_overlap="true"`: auto=False genuinely overhangs the canvas, by
    17px of real rendered ink (not an estimate) past its right edge -- the
    liveness control that proves there is a rescue to decline. The would-be
    rescue opens a 23.513x124.961 plot, 47.9% of the auto=False width
    (49.056) -- under half, so auto adds nothing: the default renders
    byte-identically to auto=False rather than a smaller plot that still
    overhangs. The overhanging label ("1,234.56789000") is drawn, unchanged,
    in both renders."""
    lo = -float("1.23456789e3")
    df = pl.DataFrame({"x": [0.0, -lo], "y": [0.0, 1.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X(
                "x:Q",
                axis=fm.Axis(label_format=",.8f", values=[0.0, -lo], label_overlap="true"),
            ),
            y="y:Q",
        )
        .properties(width=128, height=200)
    )
    svg_auto_false = chart.configure_padding(auto=False).to_svg()
    svg_default = chart.to_svg()
    assert svg_auto_false.count("<circle") == 2
    assert svg_default.count("<circle") == 2
    assert _last_x_tick_overhang_margin(svg_auto_false, 128, 200) == -17
    assert "1,234.56789000" in svg_auto_false
    clip_off = _CLIP_RECT.findall(svg_auto_false)
    assert clip_off == [("62.944", "16", "49.056", "124.961")]
    assert _CLIP_RECT.findall(svg_default) == clip_off
    assert svg_default == svg_auto_false


# ---------------------------------------------------------------------------
# Case: the 220px x-title rescue boundary, swept by real character count
# and pinned exactly at four lengths.
#
# The layout facts (clip rect, the title's own `<text x=...>`, the drawn x
# tick set) are read straight off the SVG markup. The title's ink is
# measured by rasterizing with the bundled Inter the layout itself measures
# with (`_inter_ink_columns`), never a host fallback font, whose margins
# would move with the host.
# ---------------------------------------------------------------------------


_PLAIN_X_TICK_TEXT = re.compile(r"^-?\d+(\.\d+)?$")


def _plain_x_tick_labels(svg: str) -> list[tuple[str, str, str]]:
    """Drawn x tick labels on this section's plain-numeric (non-"$",
    non-currency) fixture, as ``(text, font-size, rotation)``. Excludes the
    y axis's own tick labels, which this fixture's y domain (``0..1e9``)
    always draws at a fixed ``x="60.054"``, outside the x axis's own ticks
    regardless of the x title."""
    drawn = []
    for attrs, text in _TEXT.findall(svg):
        if not _PLAIN_X_TICK_TEXT.match(text) or 'x="60.054"' in attrs:
            continue
        rotate = re.search(r'transform="rotate\((-?[\d.]+) ', attrs)
        size = re.search(r'font-size="([\d.]+)"', attrs)
        drawn.append((text, size.group(1) if size else "", rotate.group(1) if rotate else "0"))
    return drawn


_FLAT_SHRUNK_X_TICKS = [
    ("0", "9.02", "0"),
    ("0.2", "9.02", "0"),
    ("0.4", "9.02", "0"),
    ("0.6", "9.02", "0"),
    ("0.8", "9.02", "0"),
    ("1", "9.02", "0"),
]
_FLAT_EVERY_THIRD_X_TICKS = [
    ("0", "11", "0"),
    ("0.3", "11", "0"),
    ("0.6", "11", "0"),
    ("0.9", "11", "0"),
]
_ROTATED_FULL_SIZE_X_TICKS = [
    ("0", "11", "-90"),
    ("0.2", "11", "-90"),
    ("0.4", "11", "-90"),
    ("0.6", "11", "-90"),
    ("0.8", "11", "-90"),
    ("1", "11", "-90"),
]


@pytest.mark.parametrize(
    "chars, clip_on, drawn_default, title_x, ink_right_off",
    [
        (25, ("66.054", "16", "125.397", "110.056"), _ROTATED_FULL_SIZE_X_TICKS, "128.752", 225),
        (26, ("66.054", "16", "118.098", "124.961"), _FLAT_EVERY_THIRD_X_TICKS, "125.103", 228),
        (27, ("66.054", "16", "110.798", "124.961"), _FLAT_EVERY_THIRD_X_TICKS, "121.453", 232),
        (28, ("66.054", "16", "103.498", "124.961"), _FLAT_EVERY_THIRD_X_TICKS, "117.803", 236),
    ],
)
def test_220px_x_title_boundary_is_pinned_at_real_fonts(
    chars, clip_on, drawn_default, title_x, ink_right_off
):
    """220x200, `y` over a 0..1e9 domain (so the y label band stays wide
    and fixed regardless of the x title) and an x title of *chars*
    repeated "a" characters, over `x=[0.0, 1.0]`: `auto=False` always draws
    the six x ticks ("0"/"0.2"/.../"1") flat and shrunk to the cascade's
    9.02pt font (`_FLAT_SHRUNK_X_TICKS`), whatever the title's length.

    - 25 chars: `padding.auto` genuinely rescues the title. The plot
      narrows (clip width 137.946 -> 125.397) and the recentred title's own
      `<text x=...>` moves from 135.027 to 128.752; the x axis keeps the
      SAME six labels, but now rotated -90 at the full, unshrunk 11pt font
      (`_ROTATED_FULL_SIZE_X_TICKS`) rather than flat-and-shrunk.
    - 26/27/28 chars: `padding.auto` rescues the title too. The plot
      narrows further (clip width 137.946 -> 118.098 / 110.798 / 103.498,
      each above half the auto=False width), the title recentres (135.027
      -> 125.103 / 121.453 / 117.803, its Inter ink ending at x=218, inside
      the 220px canvas, where auto=False's ends at 228 / 232 / 236), and the
      x axis lays its 11 ticks (0, 0.1, ..., 1) flat at the full 11pt font
      labelling every third: 0, 0.3, 0.6, 0.9
      (`_FLAT_EVERY_THIRD_X_TICKS`). That is an even stride thinning of a
      continuous axis with at least 3 labels and none elided, which is not
      a loss, so the rescue ships.
    """
    title = "a" * chars
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1000000000.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", title=title), y="y")
        .properties(width=220, height=200)
    )
    svg_auto_false = chart.configure_padding(auto=False).to_svg()
    svg_default = chart.to_svg()
    assert _plain_x_tick_labels(svg_auto_false) == _FLAT_SHRUNK_X_TICKS
    assert _plain_x_tick_labels(svg_default) == drawn_default
    assert _CLIP_RECT.findall(svg_default) == [clip_on]
    title_match = re.search(
        rf'<text x="(-?[\d.]+)"[^>]*text-anchor="middle"[^>]*>{title}</text>', svg_default
    )
    assert title_match is not None
    assert title_match.group(1) == title_x
    assert svg_default != svg_auto_false
    clip_off = _CLIP_RECT.findall(svg_auto_false)
    assert clip_off == [("66.054", "16", "137.946", "124.961")]
    assert float(clip_on[2]) >= float(clip_off[0][2]) / 2
    # The title's real Inter ink: past the canvas with auto off, inside it by default.
    assert _inter_ink_columns(svg_auto_false, 220, 200)[1] == ink_right_off
    assert _inter_ink_columns(svg_default, 220, 200)[1] == 218
