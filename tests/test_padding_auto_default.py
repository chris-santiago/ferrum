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
import warnings
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
            150,  # rotated -90, 3 labels
            165,  # rotated -90, 4 labels
            200,  # rotated -90, 6 labels
            250,  # thinned flat, 3 labels
            310,  # thinned flat, 4 labels
        ],
    )
    def test_cascade_width_bands_keep_auto_false(self, width):
        """One real-font width per cascade band (rotated at full font, or
        thinned to an even stride and drawn flat): the drawn labels fit with
        auto off, so the default is byte-identical to `auto=False`."""
        chart = self._chart().properties(width=width)
        svg_auto_false = chart.configure_padding(auto=False).to_svg()
        assert _last_x_tick_overhang_margin(svg_auto_false, width, self.HEIGHT) >= 0
        assert _sha16(chart.to_svg()) == _sha16(svg_auto_false)

    def test_flat_band_default_ticks_draw_the_same_labels_as_auto_false(self):
        """This class's chart itself (x at 0, 5e5 and 1e6, default ticks,
        currency) at 740px: the cascade draws every one of its 11 ticks
        flat at the full 11pt font. The last label's ink reaches a fraction
        of a pixel past the canvas, so the default rescues it and differs
        from `auto=False`, but it must draw exactly the same labels, at the
        same angles and font sizes -- the rescue narrows the plot rather
        than thinning the tick set."""
        width = 740
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
        assert clip_off == [("59.727", "16", "664.273", "124.961")]
        assert clip_default == [("59.727", "16", "658.52", "124.961")]

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
    """192x200, a `,.8f`-formatted last label (1234.56789): both adjacent
    labels have room at the full 11pt font, so auto=False draws both flat
    labels ("0.00000000" and "1,234.56789000") at full size in a
    113.056x124.961 plot, but the last one overhangs the canvas by 14px of
    real rendered ink. Narrowing the plot to 90.881x124.961 -- still 80.4% of
    the auto=False width, well above half in both dimensions -- pulls it
    back onto the canvas (13px to spare) without dropping either label, so
    the default applies the rescue in full: same two flat labels, same font,
    narrower plot, no overhanging ink."""
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
    assert clip_default == [("62.944", "16", "90.881", "124.961")]
    off_w, off_h = (float(v) for v in clip_off[0][2:4])
    default_w, default_h = (float(v) for v in clip_default[0][2:4])
    assert default_w >= off_w / 2
    assert default_h >= off_h / 2
    assert svg_default != svg_auto_false
    assert _last_x_tick_overhang_margin(svg_auto_false, 192, 200) == -14
    assert _last_x_tick_overhang_margin(svg_default, 192, 200) == 13


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


def test_the_cascade_thins_the_x_labels_on_its_own_while_the_rescue_only_narrows_for_the_title():
    """270x200, x=[0,1,2,3] with the title "A moderately long horizontal
    axis title here": the x-axis collision cascade thins its 16 ticks
    (0, 0.2, ..., 3) to an even stride of 3 and draws them flat at the full
    11pt font on its own, so `auto=False` and the default agree on the same
    6 labels (0, 0.6, ..., 3) in that regard. What still differs is the
    TITLE: with auto off it keeps the 191.056x124.961 plot and the title's
    ink overhangs the canvas by a measured 8px; the rescue narrows the plot
    to 151.999x124.961 (79.6% of the auto=False width) and pulls the
    title's overhang back to +11px."""
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
    assert _X_TICK_LABEL.findall(svg_auto_false) == []
    assert _X_TICK_LABEL.findall(svg_default) == []
    thinned_x_labels = [(text, "11") for text in ("0", "0.6", "1.2", "1.8", "2.4", "3")]
    assert _flat_x_tick_labels(svg_auto_false) == thinned_x_labels
    assert _flat_x_tick_labels(svg_default) == thinned_x_labels
    assert "\u2026" not in svg_default
    clip_off = _CLIP_RECT.findall(svg_auto_false)
    clip_default = _CLIP_RECT.findall(svg_default)
    assert clip_off == [("62.944", "16", "191.056", "124.961")]
    assert clip_default == [("62.944", "16", "151.999", "124.961")]
    assert float(clip_default[0][2]) >= float(clip_off[0][2]) / 2
    assert _last_x_tick_overhang_margin(svg_auto_false, 270, 200) == -8
    assert _last_x_tick_overhang_margin(svg_default, 270, 200) == 11


def test_a_rescue_that_would_cull_every_x_label_is_declined():
    """144x200, x=[-1e12, -5e11, -1e3] ticked at all three values with
    `,.4f`: auto=False culls two of the three ticks already, drawing one
    rotated label ("-1,000,000,000,000.0000" at -60 degrees) in a
    65.056x3.955 plot, overhanging the canvas on the left by a measured
    13.5px. The default ships byte-identical to auto=False and warns
    naming the left side and the 13.5px overhang; auto=False itself never
    warns.

    With the left side set explicitly to the warned amount
    (`configure_padding(left=16+13.5+1)`), the plot is 50.556px wide and
    140.271px tall -- both at or above half of auto=False's 65.056x3.955
    -- yet the rescue draws no x tick label at all: the one label
    auto=False still manages is exactly what the rescue would have to
    drop, which the no-more-lost-x-labels rule refuses."""
    lo, mid, hi = -1e12, -5e11, -1e3
    df = pl.DataFrame({"x": [lo, mid, hi], "y": [0.0, 1.0, 2.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x:Q", axis=fm.Axis(label_format=",.4f", values=[lo, mid, hi])), y="y:Q")
        .properties(width=144, height=200)
    )
    svg_auto_false = chart.configure_padding(auto=False).to_svg()
    assert _X_TICK_LABEL.findall(svg_auto_false) == [("-60", "-1,000,000,000,000.0000")]
    assert _flat_x_tick_labels(svg_auto_false) == []
    assert _CLIP_RECT.findall(svg_auto_false) == [("62.944", "16", "65.056", "3.955")]
    with warnings.catch_warnings(record=True) as off_warnings:
        warnings.simplefilter("always")
        assert chart.configure_padding(auto=False).to_svg() == svg_auto_false
    assert off_warnings == []
    with pytest.warns(UserWarning, match=r"x tick labels overflow the canvas on the left by 13\.5px"):
        svg_default = chart.to_svg()
    assert svg_default == svg_auto_false

    off_w, off_h = 65.056, 3.955
    twin = chart.configure_padding(left=16 + 13.5 + 1)
    twin_off = twin.configure_padding(auto=False).to_svg()
    twin_w, twin_h = (float(v) for v in _CLIP_RECT.findall(twin_off)[0][2:4])
    assert twin_w >= off_w / 2 and twin_h >= off_h / 2
    assert _X_TICK_LABEL.findall(twin_off) == []
    assert _flat_x_tick_labels(twin_off) == []
    assert twin_off == twin.to_svg()


def test_a_rescue_that_thins_the_x_labels_to_two_is_declined():
    """252x150, x=[0, 49999999.9375, 99999999.875] ticked at all three
    values with `,.3f`: auto=False draws all three flat at the shrunk
    9.02pt font in the same 173.056x74.961 plot, the last one overhanging
    the canvas on the right by a measured 10.3px. The default ships
    byte-identical to auto=False and warns naming the right side and the
    10.3px overhang; auto=False itself never warns.

    With the right side set explicitly to the warned amount
    (`configure_padding(right=16+10.3+1)`), the plot is 161.756px wide and
    74.961px tall -- both above half of auto=False's -- yet the rescue
    keeps only the two end ticks ("0.000" and "99,999,999.875"), dropping
    the middle one: an even thinning needs at least three survivors, so
    thinning to two is still a loss the own-ticks rule refuses, which is
    why the default declines rather than applying this narrower plot."""
    xs = [0.0, 49999999.9375, 99999999.875]
    df = pl.DataFrame({"x": xs, "y": [0.0, 1.0, 2.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x:Q", axis=fm.Axis(label_format=",.3f", values=xs)), y="y:Q")
        .properties(width=252, height=150)
    )
    svg_auto_false = chart.configure_padding(auto=False).to_svg()
    assert _flat_x_tick_labels(svg_auto_false) == [
        ("0.000", "9.02"),
        ("49,999,999.938", "9.02"),
        ("99,999,999.875", "9.02"),
    ]
    assert _CLIP_RECT.findall(svg_auto_false) == [("62.944", "16", "173.056", "74.961")]
    assert _last_x_tick_overhang_margin(svg_auto_false, 252, 150) == -4
    with warnings.catch_warnings(record=True) as off_warnings:
        warnings.simplefilter("always")
        assert chart.configure_padding(auto=False).to_svg() == svg_auto_false
    assert off_warnings == []
    with pytest.warns(UserWarning, match=r"x tick labels overflow the canvas on the right by 10\.3px"):
        svg_default = chart.to_svg()
    assert svg_default == svg_auto_false

    off_w, off_h = 173.056, 74.961
    twin = chart.configure_padding(right=16 + 10.3 + 1)
    twin_off = twin.configure_padding(auto=False).to_svg()
    twin_w, twin_h = (float(v) for v in _CLIP_RECT.findall(twin_off)[0][2:4])
    assert twin_w >= off_w / 2 and twin_h >= off_h / 2
    assert _flat_x_tick_labels(twin_off) == [("0.000", "11"), ("99,999,999.875", "11")]
    assert twin_off == twin.to_svg()


def test_a_rescue_that_would_exceed_half_the_plot_is_declined():
    """128x200, a `,.8f`-formatted last label (1234.56789) under
    `label_overlap="true"`: auto=False genuinely overhangs the canvas, by
    17px of real rendered ink (not an estimate) past its right edge -- the
    liveness control that proves there is a rescue to decline. The would-be
    rescue opens a 23.513x124.961 plot, 47.9% of the auto=False width
    (49.056) -- under half, so auto adds nothing: the default renders
    byte-identically to auto=False rather than a smaller plot that still
    overhangs. The overhanging label ("1,234.56789000") is drawn, unchanged,
    in both renders. The default warns naming the right side and the
    24.3px overhang the real rendered ink measures; auto=False never
    warns."""
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
    with warnings.catch_warnings(record=True) as off_warnings:
        warnings.simplefilter("always")
        assert chart.configure_padding(auto=False).to_svg() == svg_auto_false
    assert off_warnings == []
    with pytest.warns(UserWarning, match=r"x tick labels overflow the canvas on the right by 24\.3px"):
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


_FLAT_EVERY_THIRD_X_TICKS = [
    ("0", "11", "0"),
    ("0.3", "11", "0"),
    ("0.6", "11", "0"),
    ("0.9", "11", "0"),
]


@pytest.mark.parametrize(
    "chars, clip_on, title_x, ink_right_off",
    [
        (25, ("66.054", "16", "125.397", "124.961"), "128.752", 225),
        (26, ("66.054", "16", "118.098", "124.961"), "125.103", 228),
        (27, ("66.054", "16", "110.798", "124.961"), "121.453", 232),
        (28, ("66.054", "16", "103.498", "124.961"), "117.803", 236),
    ],
)
def test_220px_x_title_boundary_is_pinned_at_real_fonts(chars, clip_on, title_x, ink_right_off):
    """220x200, `y` over a 0..1e9 domain (so the y label band stays wide
    and fixed regardless of the x title) and an x title of *chars*
    repeated "a" characters, over `x=[0.0, 1.0]`: the 11 ticks (0, 0.1,
    ..., 1) thin to an even stride of 3, labelling every third
    (0, 0.3, 0.6, 0.9) flat at the full 11pt font (`_FLAT_EVERY_THIRD_X_TICKS`)
    -- a thinning the x-axis cascade applies on its own, so `auto=False` and
    the default agree on the tick set at every title length.

    What the title's length still moves is the rescue of the TITLE itself:
    `padding.auto` narrows the plot (clip width 137.946 -> 125.397 / 118.098
    / 110.798 / 103.498, each above half the auto=False width) and recentres
    the title's own `<text x=...>` (135.027 -> 128.752 / 125.103 / 121.453 /
    117.803), pulling its Inter ink back inside the 220px canvas (ending at
    x=218) where `auto=False`'s overhangs it (ending at 225 / 228 / 232 /
    236). The x tick set itself, and its 124.961 band height (``clip_on``'s
    fourth value), is unaffected either way.
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
    assert _plain_x_tick_labels(svg_auto_false) == _FLAT_EVERY_THIRD_X_TICKS
    assert _plain_x_tick_labels(svg_default) == _FLAT_EVERY_THIRD_X_TICKS
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


# ---------------------------------------------------------------------------
# A declined rescue that still clips content warns: `padding.auto` either
# contains the labels or says why it didn't. An explicit side, or
# `auto=False`, is the user's own clip and stays silent (their choice).
# ---------------------------------------------------------------------------


def test_a_declined_rescue_warns_with_the_side_amount_and_remedy():
    """264x150, x=[0, 1e6] formatted `currency_cents`: the default declines
    to rescue the clipping right-edge label ("$1,000,000.00") and warns
    naming the content kind, the side, the overflow in pixels, the cause
    and a remedy. `auto=False` renders the identical bytes but raises no
    warning -- the decline is the user's own clip once they turn
    `padding.auto` off."""
    df = pl.DataFrame({"x": [0.0, 1_000_000.0], "y": [0.0, 1.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x:Q", axis=fm.Axis(label_format="currency_cents")), y="y:Q")
        .properties(width=264, height=150)
    )
    with pytest.warns(UserWarning, match="x tick labels overflow the canvas on the right"):
        svg_default = chart.to_svg()
    with warnings.catch_warnings(record=True) as off_warnings:
        warnings.simplefilter("always")
        svg_auto_false = chart.configure_padding(auto=False).to_svg()
    assert off_warnings == []
    assert svg_default == svg_auto_false
    with warnings.catch_warnings(record=True) as default_warnings:
        warnings.simplefilter("always")
        chart.to_svg()
    message = str(default_warnings[0].message)
    assert "x tick labels overflow the canvas on the right" in message
    assert "padding.auto declined the margin" in message
    assert "more than half the plot or drop x tick labels" in message
    assert "widen the chart" in message
    assert "label_angle or label_overlap" in message
    assert "configure_padding(right=...)" in message


def test_a_declined_rescue_is_silent_under_an_explicit_padding_side():
    """The same 264x150 currency_cents chart, with the right side set
    explicitly: that side is the user's own clip, not `padding.auto`'s, so
    no warning fires even though the same label still overflows."""
    df = pl.DataFrame({"x": [0.0, 1_000_000.0], "y": [0.0, 1.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x:Q", axis=fm.Axis(label_format="currency_cents")), y="y:Q")
        .properties(width=264, height=150)
        .configure_padding(right=20)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.to_svg()
    assert caught == []


def test_a_declined_rescue_is_silent_under_auto_false():
    """The same chart with `auto=False`: declining a rescue is only
    disclosed when `padding.auto` owns the decision; a user who turned it
    off entirely gets no warning for their own clip."""
    df = pl.DataFrame({"x": [0.0, 1_000_000.0], "y": [0.0, 1.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x:Q", axis=fm.Axis(label_format="currency_cents")), y="y:Q")
        .properties(width=264, height=150)
        .configure_padding(auto=False)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.to_svg()
    assert caught == []
