"""Pixel matrix pinning scope invariants for interactive conditionals and selections.

Extends the per-layer conditional-reach rule ``ferrum-spec.md`` documents
("a conditional reaches the layers its chart's field encodings reach") to
every real conditional channel, and pins an analogous rule for selection
input:

- **Declared scope.** A conditional of any channel restyles only the marks
  of the layer or composite child that declared it. A chart-level or
  host-chart conditional (one no layer owns) still reaches every attached
  layer, unchanged.
- **Owner-panel input.** A click or brush updates a selection only from
  input on the panels of the chart that declared it. In linked views where
  more than one chart declares the SAME selection, input on any one of
  those charts' own panels drives it -- the union of their panels, not
  just the first declarer's.

Population for the declared-scope matrix is every real conditional channel
(``_CONDITIONAL_CHANNELS``), derived rather than hand-listed. Every channel
but ``angle`` uses a point mark. ``angle`` is the one exception: rotating a
circular point mark's own shape (``apply_value_common``,
``crates/ferrum-wasm/src/conditional.rs``) produces no pixel delta, so
``angle``'s cells use a rect mark instead, whose asymmetric bounding box
makes a rotation visible at one corner (``_angle_corner_probe``, mirroring
``tests/test_conditional_pixels.py``'s own ``_check_angle_rect``, which this
file imports and calls directly for every ``angle`` assertion that expects a
mark to rotate). Per-channel rendering correctness (every value form, every
mark) is ``tests/test_conditional_pixels.py``'s job, not this file's; this
file only asks *which marks* react.

Every mark declaring a conditional starts at its own plain style while the
selection it watches is empty, and only partitions into the conditional's
two branches once that selection holds at least one member. ``if_not`` is
therefore chosen to differ from the mark's own plain style for every
channel below (unlike ``test_conditional_pixels.py``'s ``_BASELINE``
convention, which keeps ``if_not`` equal to the resting look): only a
visible plain-style-to-``if_not`` transition can show that a mark's
conditional evaluated at all once a selection leaves empty. ``angle`` is
the one exception to this literal-pair convention: ``_check_angle_rect``'s
own oracle fixes the unrotated state at 0 degrees, so ``angle``'s pair sets
``if_selected`` to the mark's own resting angle (0) and ``if_not`` to the
rotated literal (40) instead. The clicked mark's own appearance goes
unchecked (unneeded, since every probe below targets a sibling mark), and
every mark this file actually probes still shows the same plain-to-
``if_not`` transition the rest of this paragraph describes.

``color``, ``opacity`` and ``fill_opacity`` repaint a mark's own fill, so a
mark's own center is enough to see their leak. The remaining channels need
their own probe geometry (``_probe_offsets``): ``size`` moves the
fill/background boundary radially; ``stroke_width`` and ``stroke_opacity``
act on the stroke ring, which sits at a fixed radius regardless of which
literal is active; ``stroke_dash`` alternates stroke/background around
that ring at an unknown phase, so many angles are sampled and the largest
delta is taken; ``angle`` samples one corner point whose fill membership
flips between the resting and rotated literal.

``opacity`` alone also needs a victim-style workaround the other two fill
channels do not: a mark left at ``mark_point()``'s own implicit default
(``opacity=1.0``, the only one of the three with a constant-style kwarg at
all) shows zero pixel movement even when its batch receives a leaked
opacity write, while the identical fixture with an explicit near-opaque
``opacity=`` style shows a large, discriminating shift.
``fill_opacity``/``stroke_opacity`` have no such static-style path
(``mark_point(fill_opacity=...)`` is not a honored kwarg) and a leaked
conditional on either is visible against their plain default. The victim
fixtures below give `opacity` that explicit style only where it is needed:
the two asymmetric (no-own-conditional) cases.

A brush (interval selection) drag is a no-op in this renderer below 5
points per series, on a single, non-composite chart as much as a composite
one -- unrelated to the scope invariants above. Every interval-selection
fixture below therefore uses 5+ points per series, matching
``tests/test_conditional_pixels.py``'s own unpacked fixture size.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import polars as pl
import pytest

import ferrum as fm
from ferrum import value
from ferrum._core import render_interactive
from ferrum.encoding._channel_policy import _CONDITIONAL_CHANNELS

from tests import _pixel_capture as pc
from tests._conditional_oracle import rgb_distance3
from tests.test_conditional_pixels import Fixture as _AngleProbeFixture
from tests.test_conditional_pixels import _check_angle_rect

pytestmark = pytest.mark.wasm_pixels

_BUNDLE_PATH = "src/ferrum/_wasm/ferrum_wasm_bg.wasm"

BASE_A = "#2563eb"
BASE_B = "#f59e0b"
BASE_STROKE = "#111827"
RED = "#dc2626"
ORANGE = "#ea580c"

# Every real conditional channel joins the matrix population, derived from
# the vocabulary rather than hand-picked. ``angle`` observes its own reach
# on a rect mark instead of a point -- see the module docstring and
# `_angle_reach_fixture`.
_CHANNELS: tuple[str, ...] = tuple(sorted(_CONDITIONAL_CHANNELS))

# `opacity`'s own leaked write is invisible on a mark left at the implicit
# default (`mark_point()`'s own `opacity=1.0`, see the module docstring's
# opacity-pipeline paragraph) -- the asymmetric cases give their victim an
# explicit, non-default `opacity=` style for exactly this channel.
# `fill_opacity`/`stroke_opacity` have no equivalent static-style path at
# all (`mark_point(fill_opacity=...)` is not a honored kwarg, and a leaked
# conditional on either channel IS visible against the plain default), so
# they need no victim override.
_STATIC_STYLE_ALPHA_CHANNELS = frozenset({"opacity"})
_VICTIM_ALPHA_STYLE = 0.85

# (if_selected, if_not) literal pair per channel, each chosen so if_not
# differs from the plain style `_style()` below sets (see module docstring).
# `stroke_dash`'s pair uses a thicker stroke and a bigger pattern than every
# other channel -- see `_resting_stroke_width`. `angle`'s pair is the one
# exception to the if_not-differs-from-resting rule -- see the module
# docstring.
_LITERALS: dict[str, tuple[Any, Any]] = {
    "color": (RED, ORANGE),
    "opacity": (1.0, 0.3),
    "fill_opacity": (1.0, 0.3),
    "stroke_opacity": (1.0, 0.1),
    "stroke_width": (3.0, 0.5),
    "stroke_dash": ("solid", [6.0, 3.0]),
    "size": (452.0, 3000.0),
    "angle": (0.0, 40.0),
}


def test_population_is_derived_and_non_empty() -> None:
    """The reach-matrix channel population must come from the real conditional vocabulary."""
    assert _CHANNELS
    assert set(_LITERALS) == set(_CHANNELS)


# `stroke_dash`'s own resting stroke width (see `_resting_stroke_width`);
# every other channel uses this one.
_BASE_STROKE_W = 3.0
# `stroke_dash` needs a thicker stroke for its dash period to resolve at
# this mark's size (a 3px-period dash at a 3px stroke width renders solid;
# a 3px-period dash at a 6px width alternates cleanly), mirroring
# `tests/test_conditional_pixels.py`'s own `DASH_STROKE_W` convention.
_DASH_STROKE_W = 6.0


def _resting_stroke_width(channel: str) -> float:
    return _DASH_STROKE_W if channel == "stroke_dash" else _BASE_STROKE_W


def _style(fill: str, channel: str = "color", **overrides: Any) -> dict[str, Any]:
    style: dict[str, Any] = {
        "fill": fill,
        "stroke": BASE_STROKE,
        "stroke_width": _resting_stroke_width(channel),
        "size": 452.0,
    }
    style.update(overrides)
    return style


# `_style()`'s fixed base geometry (`size=452.0`):
_OUTER_RADIUS = math.sqrt(452.0 / math.pi)  # ~12.0
_EDGE_R = 18.0  # outside the shrunk size literal's radius, inside the enlarged one's fill edge


def _ring_radius(channel: str) -> float:
    """The radius, inside the resting stroke ring, that `_probe_offsets` samples at."""
    return _OUTER_RADIUS - _resting_stroke_width(channel) / 2.0


def _probe_offsets(channel: str) -> list[tuple[float, float]]:
    """SVG-space ``(dx, dy)`` offsets from a mark's own center that detect *channel*'s reaction.

    ``color``/``opacity``/``fill_opacity`` repaint the whole fill, so the
    center itself is enough. ``size`` moves the fill/background boundary
    radially -- ``_EDGE_R`` sits between the shrunk and enlarged literal's
    fill edges (see ``_LITERALS``). ``stroke_width``/``stroke_opacity`` act
    on the stroke ring -- ``_ring_radius(channel)`` sits inside the resting
    ring regardless of which literal is active. ``stroke_dash`` alternates
    stroke/background around the ring at an unknown phase, so many
    evenly-spaced angles are sampled and the caller takes the largest
    delta: a genuinely dashed ring must differ from a solid one at some
    angle, and this file only asks "did it react", not "does it look
    right" (that precision lives in ``tests/test_conditional_pixels.py``).
    """
    if channel == "size":
        return [(_EDGE_R, 0.0)]
    r = _ring_radius(channel)
    if channel in ("stroke_width", "stroke_opacity"):
        return [(r, 0.0)]
    if channel == "stroke_dash":
        n = 64
        return [
            (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)
        ]
    return [(0.0, 0.0)]


def _max_delta(before: list[tuple[int, ...]], after: list[tuple[int, ...]]) -> float:
    return max(rgb_distance3(b, a) for b, a in zip(before, after))


# ── Declared-scope reach matrix ──────────────────────────────────────────────

ReachCase = Literal["pipe_asymmetric", "plus_asymmetric", "linked_symmetric", "chart_level"]
_REACH_CASES: tuple[ReachCase, ...] = (
    "pipe_asymmetric",
    "plus_asymmetric",
    "linked_symmetric",
    "chart_level",
)
# Whether the non-declaring chart/layer is expected to restyle once the
# declaring side's selection fires. False rows must show no restyle at all
# (the asymmetric-scope rule); True rows must show one -- a cell that must
# differ from the False rows, proving the oracle can tell "restyled" from
# "untouched" at all.
_EXPECT_B_CHANGES: dict[ReachCase, bool] = {
    "pipe_asymmetric": False,
    "plus_asymmetric": False,
    "linked_symmetric": True,
    "chart_level": True,
}

_DF_A = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 1.0, 5.0, 3.0]})
# A disjoint x/y range for the single-panel (`+`, chart-level) cases, so the
# two layers' marks never land on the same pixel.
_DF_B_DISJOINT = pl.DataFrame({"x": [10.0, 11.0, 12.0, 13.0, 14.0], "y": [5.0, 7.0, 4.0, 8.0, 6.0]})

# `angle`'s own rect-mark data (point marks can't show a rotation -- see the
# module docstring), mirroring `_DF_A`/`_DF_B_DISJOINT`'s shape for the
# point-mark cells above. Unlike a point mark's style-driven `size`, a
# rect's own extent comes from its x/x2/y/y2 data mapped through its
# chart's scale, so the disjoint variant shifts only `y` (not `x`): the two
# layers' marks still never land on the same pixel, and the shared domain
# the single-panel (`+`, chart-level) cases build from both layers' data
# stays close enough to `_DF_A_RECT`'s own that every rect keeps the real
# proportions `_angle_corner_probe`'s inset needs.
_DF_A_RECT = pl.DataFrame(
    {
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "x2": [0.8, 1.8, 2.8, 3.8, 4.8],
        "y": [0.0] * 5,
        "y2": [1.0] * 5,
    }
)
_DF_B_RECT_DISJOINT = pl.DataFrame(
    {
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "x2": [0.8, 1.8, 2.8, 3.8, 4.8],
        "y": [2.0] * 5,
        "y2": [3.0] * 5,
    }
)


def _rect_style(fill: str, **overrides: Any) -> dict[str, Any]:
    """Fixed style for `angle`'s rect-mark fixtures.

    No `size` kwarg (a rect's extent comes from its own x/x2/y/y2 data, not
    a style literal). The stroke is thin, leaving `_angle_corner_probe` and
    `_check_angle_rect`'s own corner probe enough margin to clear both the
    stroke ring and the anti-aliased edge, mirroring
    `tests/test_conditional_pixels.py`'s own `angle` stroke choice.
    """
    style: dict[str, Any] = {"fill": fill, "stroke": BASE_STROKE, "stroke_width": 0.5}
    style.update(overrides)
    return style


def _angle_corner_probe(target: tuple[float, float, float, float]) -> tuple[float, float]:
    """The corner point whose fill membership flips between `angle`'s two literals.

    Mirrors `tests/test_conditional_pixels.py`'s own `_check_angle_rect`
    corner-inset formula: a fixed absolute inset keeps the probe clear of
    the anti-aliased edge blur.
    """
    x, y, w, h = target
    inset = min(2.0, w * 0.15, h * 0.15)
    return (x + w - inset, y + inset)


def _angle_fixture(
    svg: str, svg_w: float, svg_h: float, target: tuple[float, float, float, float], idx: int
) -> _AngleProbeFixture:
    """Wrap one rect target as the `Fixture` `_check_angle_rect` expects.

    Only the fields `_check_angle_rect` reads (`svg`, `svg_w`, `svg_h`,
    `target`, `target_idx`) carry real values; the rest are unused by that
    oracle.
    """
    return _AngleProbeFixture(
        svg=svg,
        svg_w=svg_w,
        svg_h=svg_h,
        html_path=Path(),
        target=target,
        target_idx=idx,
        target_category="",
        neighbor_center=(0.0, 0.0),
        viewport={},
    )


def _angle_reach_fixture(case: ReachCase) -> Any:
    """Build the (case, "angle") reach-matrix cell on rect marks.

    Mirrors `_reach_fixture`'s own four cases, swapping `mark_point` for
    `mark_rect` and the point data for `_DF_A_RECT`/`_DF_B_RECT_DISJOINT`.
    """
    if_selected, if_not = _LITERALS["angle"]
    df_b = _DF_A_RECT if case in ("pipe_asymmetric", "linked_symmetric") else _DF_B_RECT_DISJOINT

    sel = fm.selection_point(name=f"sel_{case}_angle", clear="false")
    cond_a = sel.when(value(if_selected)).otherwise(value(if_not))

    def _rect_chart(df: Any, fill: str) -> Any:
        return fm.Chart(df).mark_rect(**_rect_style(fill)).encode(x="x", x2="x2", y="y", y2="y2")

    if case == "pipe_asymmetric":
        chart_a = (
            fm.Chart(_DF_A_RECT)
            .mark_rect(**_rect_style(BASE_A))
            .encode(x="x", x2="x2", y="y", y2="y2", angle=cond_a)
            .add_selection(sel)
        )
        return chart_a | _rect_chart(df_b, BASE_B)

    if case == "plus_asymmetric":
        chart_a = (
            fm.Chart(_DF_A_RECT)
            .mark_rect(**_rect_style(BASE_A))
            .encode(x="x", x2="x2", y="y", y2="y2", angle=cond_a)
            .add_selection(sel)
        )
        return chart_a + _rect_chart(df_b, BASE_B)

    if case == "linked_symmetric":
        # Both charts declare the SAME selection and their own conditional
        # on it, matching `_reach_fixture`'s own `linked_symmetric` shape.
        cond_b = sel.when(value(if_selected)).otherwise(value(if_not))
        chart_a = (
            fm.Chart(_DF_A_RECT)
            .mark_rect(**_rect_style(BASE_A))
            .encode(x="x", x2="x2", y="y", y2="y2", angle=cond_a)
            .add_selection(sel)
        )
        chart_b = (
            fm.Chart(df_b)
            .mark_rect(**_rect_style(BASE_B))
            .encode(x="x", x2="x2", y="y", y2="y2", angle=cond_b)
            .add_selection(sel)
        )
        return chart_a | chart_b

    # chart_level: the conditional is declared on the already-merged
    # `(a + b)` chart itself, so neither layer owns it, matching
    # `_reach_fixture`'s own `chart_level` shape.
    chart_a = (
        fm.Chart(_DF_A_RECT).mark_rect(**_rect_style(BASE_A)).encode(x="x", x2="x2", y="y", y2="y2")
    )
    chart_b = fm.Chart(df_b).mark_rect(**_rect_style(BASE_B)).encode(x="x", x2="x2", y="y", y2="y2")
    return (chart_a + chart_b).encode(angle=cond_a).add_selection(sel)


def _reach_fixture(case: ReachCase, channel: str) -> Any:
    """Build the chart for one (case, channel) reach-matrix cell.

    Returns the composed ``Chart``. For every channel but `angle`
    (dispatched to `_angle_reach_fixture` instead -- see the module
    docstring), its static SVG lists the declaring chart's 5 marks first,
    then the non-declaring one's 5 marks, by construction -- `circles[1]` is
    always the declaring chart's own click target, `circles[5]` is always the
    non-declaring chart's own probe mark. `_angle_reach_fixture` builds its
    rect marks in the same order.
    """
    if channel == "angle":
        return _angle_reach_fixture(case)
    if_selected, if_not = _LITERALS[channel]
    df_b = _DF_A if case in ("pipe_asymmetric", "linked_symmetric") else _DF_B_DISJOINT

    style_a = _style(BASE_A, channel)
    # The two asymmetric cases give their non-declaring victim an explicit,
    # non-default `opacity=` style (see `_STATIC_STYLE_ALPHA_CHANNELS`
    # above) so a leak onto it is observable at all.
    victim_overrides: dict[str, Any] = {}
    if channel in _STATIC_STYLE_ALPHA_CHANNELS and case in ("pipe_asymmetric", "plus_asymmetric"):
        victim_overrides["opacity"] = _VICTIM_ALPHA_STYLE
    style_b = _style(BASE_B, channel, **victim_overrides)

    sel = fm.selection_point(name=f"sel_{case}_{channel}", clear="false")
    cond_a = sel.when(value(if_selected)).otherwise(value(if_not))

    def _chart_b(df: Any) -> Any:
        return fm.Chart(df).mark_point(**style_b).encode(x="x", y="y")

    if case == "pipe_asymmetric":
        chart_a = (
            fm.Chart(_DF_A)
            .mark_point(**style_a)
            .encode(x="x", y="y", **{channel: cond_a})
            .add_selection(sel)
        )
        return chart_a | _chart_b(df_b)

    if case == "plus_asymmetric":
        chart_a = (
            fm.Chart(_DF_A)
            .mark_point(**style_a)
            .encode(x="x", y="y", **{channel: cond_a})
            .add_selection(sel)
        )
        return chart_a + _chart_b(df_b)

    if case == "linked_symmetric":
        # Both charts declare the SAME selection and their own conditional
        # on it -- the actual "linked views" shape: input from either
        # chart's own panel must drive the one shared selection. The SAME
        # literal pair on both charts, rather than an asymmetric one: this
        # cell's own correctness must not depend on which chart's own batch
        # a conditional happens to apply to first, so equal literals make
        # the final value order-independent, which is what "both charts
        # declare their own conditional on one shared selection and both
        # restyle" actually needs to prove.
        cond_b = sel.when(value(if_selected)).otherwise(value(if_not))
        chart_a = (
            fm.Chart(_DF_A)
            .mark_point(**style_a)
            .encode(x="x", y="y", **{channel: cond_a})
            .add_selection(sel)
        )
        chart_b = (
            fm.Chart(df_b)
            .mark_point(**style_b)
            .encode(x="x", y="y", **{channel: cond_b})
            .add_selection(sel)
        )
        return chart_a | chart_b

    # chart_level: the conditional is declared on the already-merged
    # `(a + b)` chart itself, so neither layer owns it -- Chart.to_spec's
    # `_conditional_layer_scopes` stamps it "every layer" (see
    # src/ferrum/composition.py): a chart-level conditional reaches every
    # attached layer. `Chart.conditional()` is sugar for `.encode(color=...)`
    # specifically (it always targets color), so a non-color channel must go
    # through `.encode()` directly to reach the merged chart at all.
    chart_a = fm.Chart(_DF_A).mark_point(**style_a).encode(x="x", y="y")
    chart_b = fm.Chart(df_b).mark_point(**style_b).encode(x="x", y="y")
    return (chart_a + chart_b).encode(**{channel: cond_a}).add_selection(sel)


def _assert_angle_declared_scope_reach(
    page: Any, svg: str, svg_w: float, svg_h: float, rect: pc.CanvasRect, case: ReachCase
) -> None:
    """The `angle` cell of `test_declared_scope_reach`'s own matrix, on rect marks.

    `_check_angle_rect` asserts one fixed transition (unrotated before,
    rotated after), so it is reused directly for the declaring chart's own
    unclicked sibling (the vacuity proof) and for cells where the
    non-declaring chart/layer is expected to rotate too. A cell where it is
    expected to stay put needs the inverse assertion instead, sampled at the
    same corner probe.
    """
    rects = pc.parse_rects(svg)
    ax, ay, aw, ah = rects[1]
    a_click = (ax + aw / 2, ay + ah / 2)

    before = pc.screenshot(page)
    pc.click_at(page, pc.svg_to_client(rect, svg_w, svg_h, a_click))
    after = pc.screenshot(page)

    # Vacuity control: the declaring chart's own OTHER (unclicked) rect must
    # rotate too, or the click never registered and every assertion below is
    # meaningless.
    fx_vac = _angle_fixture(svg, svg_w, svg_h, rects[0], idx=0)
    _check_angle_rect(page, fx_vac, rect, before, after)

    if _EXPECT_B_CHANGES[case]:
        fx_b = _angle_fixture(svg, svg_w, svg_h, rects[5], idx=5)
        _check_angle_rect(page, fx_b, rect, before, after)
    else:
        dev_b = pc.svg_to_device(rect, svg_w, svg_h, _angle_corner_probe(rects[5]))
        before_b = pc.sample_pixels(page, before, [dev_b])[0]
        after_b = pc.sample_pixels(page, after, [dev_b])[0]
        b_delta = rgb_distance3(before_b, after_b)
        assert b_delta < 10, (
            f"{case}/angle: a conditional declared only on the other chart/layer leaked "
            f"onto this one's marks (corner probe delta {b_delta}) -- expected no change "
            f"(committed bundle: {_BUNDLE_PATH})"
        )


@pytest.mark.parametrize("channel", _CHANNELS)
@pytest.mark.parametrize("case", _REACH_CASES)
def test_declared_scope_reach(case: ReachCase, channel: str, browser: Any, tmp_path: Path) -> None:
    chart = _reach_fixture(case, channel)
    svg = chart.to_svg()
    svg_w, svg_h = pc.parse_svg_size(svg)

    html_path = tmp_path / "chart.html"
    chart.interactive().save(str(html_path))
    page = browser.new_page(
        viewport={"width": svg_w, "height": svg_h}, device_scale_factor=pc.DEVICE_SCALE_FACTOR
    )
    try:
        pc.load_interactive(page, html_path)
        rect = pc.canvas_rect(page)

        if channel == "angle":
            _assert_angle_declared_scope_reach(page, svg, svg_w, svg_h, rect, case)
            return

        circles = pc.parse_circles(svg)
        a_click = circles[1][:2]
        offsets = _probe_offsets(channel)
        # An UNCLICKED point of the declaring chart, not the clicked one: its
        # conditional starts at the mark's own plain style (selection empty) and
        # moves to `if_not` once the click makes the selection non-empty -- the
        # only reliable "did the click register at all" probe (see module
        # docstring).
        a_vac_cx, a_vac_cy = circles[0][:2]
        vacuity_points = [(a_vac_cx + dx, a_vac_cy + dy) for dx, dy in offsets]
        b_cx, b_cy = circles[5][:2]
        b_points = [(b_cx + dx, b_cy + dy) for dx, dy in offsets]

        dev_vacuity = [pc.svg_to_device(rect, svg_w, svg_h, p) for p in vacuity_points]
        dev_b = [pc.svg_to_device(rect, svg_w, svg_h, p) for p in b_points]

        before = pc.screenshot(page)
        before_vacuity = pc.sample_pixels(page, before, dev_vacuity)
        before_b = pc.sample_pixels(page, before, dev_b)

        client_pt = pc.svg_to_client(rect, svg_w, svg_h, a_click)
        pc.click_at(page, client_pt)
        after = pc.screenshot(page)
        after_vacuity = pc.sample_pixels(page, after, dev_vacuity)
        after_b = pc.sample_pixels(page, after, dev_b)

        # Vacuity control: the click must register on the chart that
        # declared the selection, or every assertion below is meaningless.
        vacuity_delta = _max_delta(before_vacuity, after_vacuity)
        assert vacuity_delta > 10, (
            f"{case}/{channel}: clicking the declaring chart's own target must visibly "
            f"change its OTHER marks too (max probe delta {vacuity_delta}) -- the click "
            f"never registered (committed bundle: {_BUNDLE_PATH})"
        )

        b_delta = _max_delta(before_b, after_b)
        changed = b_delta > 10
        if _EXPECT_B_CHANGES[case]:
            assert changed, (
                f"{case}/{channel}: expected the non-declaring chart/layer to restyle too "
                f"(max probe delta {b_delta}) (committed bundle: {_BUNDLE_PATH})"
            )
        else:
            assert not changed, (
                f"{case}/{channel}: a conditional declared only on the other chart/layer "
                f"leaked onto this one's marks (max probe delta {b_delta}) -- "
                f"expected no change (committed bundle: {_BUNDLE_PATH})"
            )
    finally:
        page.close()


def test_vacuity_control_declaring_charts_own_click_always_registers(
    browser: Any, tmp_path: Path
) -> None:
    """Standalone vacuity control for the reach matrix above.

    Every row of ``test_declared_scope_reach`` depends on the declaring
    chart's own click visibly changing its own target -- this is that same
    check, pulled out and run once on its own so a future regression that
    makes clicking inert at all cannot hide behind the scope assertions.
    """
    chart = _reach_fixture("pipe_asymmetric", "color")
    svg = chart.to_svg()
    svg_w, svg_h = pc.parse_svg_size(svg)
    a_click = pc.parse_circles(svg)[1][:2]

    html_path = tmp_path / "chart.html"
    chart.interactive().save(str(html_path))
    page = browser.new_page(
        viewport={"width": svg_w, "height": svg_h}, device_scale_factor=pc.DEVICE_SCALE_FACTOR
    )
    try:
        pc.load_interactive(page, html_path)
        rect = pc.canvas_rect(page)
        dev_a = pc.svg_to_device(rect, svg_w, svg_h, a_click)
        before = pc.sample_pixels(page, pc.screenshot(page), [dev_a])[0]
        pc.click_at(page, pc.svg_to_client(rect, svg_w, svg_h, a_click))
        after = pc.sample_pixels(page, pc.screenshot(page), [dev_a])[0]
        assert rgb_distance3(before, after) > 10, (
            f"clicking a chart's own declared target must change its own pixel "
            f"(before {before}, after {after}) (committed bundle: {_BUNDLE_PATH})"
        )
    finally:
        page.close()


@pytest.mark.parametrize("channel", _CHANNELS)
def test_linked_views_second_panel_also_drives_shared_selection(
    channel: str, browser: Any, tmp_path: Path
) -> None:
    """The "union of owner panels" case: either linked panel drives the shared selection.

    Mirrors ``test_declared_scope_reach``'s ``linked_symmetric`` cell, which
    proves the FIRST chart's click reaches the shared selection -- this
    proves the reverse direction, clicking the SECOND chart's own target.
    A fix that stamps ownership to only the first declaring panel passes
    that cell's own assertion but fails this one.
    """
    chart = _reach_fixture("linked_symmetric", channel)
    svg = chart.to_svg()
    svg_w, svg_h = pc.parse_svg_size(svg)

    html_path = tmp_path / "chart.html"
    chart.interactive().save(str(html_path))
    page = browser.new_page(
        viewport={"width": svg_w, "height": svg_h}, device_scale_factor=pc.DEVICE_SCALE_FACTOR
    )
    try:
        pc.load_interactive(page, html_path)
        rect = pc.canvas_rect(page)

        if channel == "angle":
            rects = pc.parse_rects(svg)
            bx, by, bw, bh = rects[6]  # chart b's own second rect (seam-safe offset convention)
            b_click = (bx + bw / 2, by + bh / 2)
            before = pc.screenshot(page)
            pc.click_at(page, pc.svg_to_client(rect, svg_w, svg_h, b_click))
            after = pc.screenshot(page)
            fx_a = _angle_fixture(svg, svg_w, svg_h, rects[0], idx=0)  # chart a's own first rect
            _check_angle_rect(page, fx_a, rect, before, after)
            return

        circles = pc.parse_circles(svg)
        b_click = circles[6][:2]  # chart b's own second point (seam-safe offset convention)
        offsets = _probe_offsets(channel)
        a_cx, a_cy = circles[0][:2]  # chart a's own first, unclicked point
        a_points = [(a_cx + dx, a_cy + dy) for dx, dy in offsets]
        dev_a = [pc.svg_to_device(rect, svg_w, svg_h, p) for p in a_points]

        before = pc.screenshot(page)
        before_a = pc.sample_pixels(page, before, dev_a)
        pc.click_at(page, pc.svg_to_client(rect, svg_w, svg_h, b_click))
        after = pc.screenshot(page)
        after_a = pc.sample_pixels(page, after, dev_a)

        delta = _max_delta(before_a, after_a)
        assert delta > 10, (
            f"{channel}: clicking chart b's own target in a linked pair must also drive "
            f"chart a's shared selection (max probe delta {delta}) (committed bundle: "
            f"{_BUNDLE_PATH})"
        )
    finally:
        page.close()


# ── Owner-panel input scope ──────────────────────────────────────────────────


def test_click_on_other_chart_does_not_drive_this_charts_point_selection(
    browser: Any, tmp_path: Path
) -> None:
    """A click on `b`'s panel must not change `a`'s own, separately-named selection.

    `b` declares a plain point selection with no conditional of its own, so
    there is nothing for a declared-scope leak to paint onto `a`'s marks --
    any residual change on `a`'s own probe after clicking `b`'s target can
    only come from an input-scope bug (clicking `b`'s panel incorrectly
    updating `a`'s own selection). `a` keeps its own conditional, tied only
    to its own selection, so a genuine input-scope leak stays visible.
    `b` is the FIRST (left) chart so its click target is `circles[1]`, the
    same point every other test in this file clicks. After the isolation
    check, clicking `a`'s own target (`circles[6]`, a different point of
    `a` than the probe) must still move `a`'s probe -- without this
    positive control, a Rust fix that excludes ALL input whenever a scene
    holds more than one selection of a kind would pass the isolation
    assertion for the wrong reason.
    """
    sel_a = fm.selection_point(name="sel_a_click", clear="false")
    sel_b = fm.selection_point(name="sel_b_click", clear="false")
    cond_a = sel_a.when(value(RED)).otherwise(value(ORANGE))
    chart_b = fm.Chart(_DF_A).mark_point(**_style(BASE_B)).encode(x="x", y="y").add_selection(sel_b)
    chart_a = (
        fm.Chart(_DF_A)
        .mark_point(**_style(BASE_A))
        .encode(x="x", y="y", color=cond_a)
        .add_selection(sel_a)
    )
    combo = chart_b | chart_a
    svg = combo.to_svg()
    svg_w, svg_h = pc.parse_svg_size(svg)
    circles = pc.parse_circles(svg)
    b_click = circles[1][:2]
    a_probe = circles[5][:2]
    a_click = circles[6][:2]

    html_path = tmp_path / "chart.html"
    combo.interactive().save(str(html_path))
    page = browser.new_page(
        viewport={"width": svg_w, "height": svg_h}, device_scale_factor=pc.DEVICE_SCALE_FACTOR
    )
    try:
        pc.load_interactive(page, html_path)
        rect = pc.canvas_rect(page)
        dev_a = pc.svg_to_device(rect, svg_w, svg_h, a_probe)

        before_a = pc.sample_pixels(page, pc.screenshot(page), [dev_a])[0]
        pc.click_at(page, pc.svg_to_client(rect, svg_w, svg_h, b_click))
        after_a = pc.sample_pixels(page, pc.screenshot(page), [dev_a])[0]

        assert rgb_distance3(before_a, after_a) < 10, (
            f"a click on b's own panel changed a's pixel (before {before_a}, after "
            f"{after_a}) -- a's own, separately-named point selection must stay Empty "
            f"when the click never happened on a's own panel (committed bundle: "
            f"{_BUNDLE_PATH})"
        )

        # Positive control: a click on a's OWN target, in this same
        # two-selection scene, must still drive a's own selection.
        pc.click_at(page, pc.svg_to_client(rect, svg_w, svg_h, a_click))
        after_a_click = pc.sample_pixels(page, pc.screenshot(page), [dev_a])[0]
        assert rgb_distance3(after_a, after_a_click) > 10, (
            f"clicking a's own target in this same scene left a's pixel unchanged "
            f"(after-b-click {after_a}, after-a-click {after_a_click}) -- the isolation "
            f"check above cannot be trusted if a's own selection never responds to its "
            f"own input either (committed bundle: {_BUNDLE_PATH})"
        )
    finally:
        page.close()


def test_brush_on_other_chart_does_not_drive_this_charts_interval_selection(
    browser: Any, tmp_path: Path
) -> None:
    """A brush drag on `b`'s panel must not change `a`'s own, separately-named brush.

    `b` declares a plain interval selection with no conditional of its own --
    isolates this from a declared-scope content leak the same way the click
    test above does, so a residual change on `a`'s probe can only come from
    an input-scope bug (dragging over `b`'s panel incorrectly updating `a`'s
    own interval selection). After the isolation check, brushing `a`'s own
    target in this same two-selection scene must still move `a`'s probe --
    without this positive control, a Rust fix that excludes ALL drag input
    whenever a scene holds more than one interval selection would pass the
    isolation assertion for the wrong reason.
    """
    brush_a = fm.selection_interval(name="brush_a_drag")
    brush_b = fm.selection_interval(name="brush_b_drag")
    cond_a = brush_a.when(value(RED)).otherwise(value(ORANGE))
    chart_b = (
        fm.Chart(_DF_A).mark_point(**_style(BASE_B)).encode(x="x", y="y").add_selection(brush_b)
    )
    chart_a = (
        fm.Chart(_DF_A)
        .mark_point(**_style(BASE_A))
        .encode(x="x", y="y", color=cond_a)
        .add_selection(brush_a)
    )
    combo = chart_b | chart_a
    svg = combo.to_svg()
    svg_w, svg_h = pc.parse_svg_size(svg)
    circles = pc.parse_circles(svg)
    a_probe = circles[5][:2]
    # `circles[2]`, b's own middle point: clear of both the scene's outer
    # edge and the HConcat seam against panel a, so a padded drag rect
    # around it cannot resolve to the wrong panel on this exact two-chart
    # layout.
    b_target = circles[2][:2]
    # `circles[7]`, a's own middle point (the same offset-within-chart
    # convention as `b_target` above, shifted by a's own 5-point block):
    # clear of both the scene's outer edge and the HConcat seam.
    a_target = circles[7][:2]

    html_path = tmp_path / "chart.html"
    combo.interactive().save(str(html_path))
    page = browser.new_page(
        viewport={"width": svg_w, "height": svg_h}, device_scale_factor=pc.DEVICE_SCALE_FACTOR
    )
    try:
        pc.load_interactive(page, html_path)
        rect = pc.canvas_rect(page)
        dev_a = pc.svg_to_device(rect, svg_w, svg_h, a_probe)
        before_a = pc.sample_pixels(page, pc.screenshot(page), [dev_a])[0]

        pad = 40.0
        bx, by = b_target
        start = pc.svg_to_client(rect, svg_w, svg_h, (bx - pad, by - pad))
        end = pc.svg_to_client(rect, svg_w, svg_h, (bx + pad, by + pad))
        pc.brush_drag(page, start, end)

        after_a = pc.sample_pixels(page, pc.screenshot(page), [dev_a])[0]
        assert rgb_distance3(before_a, after_a) < 10, (
            f"a brush dragged entirely over b's panel changed a's pixel (before {before_a}, "
            f"after {after_a}) -- a's own, separately-named interval selection must stay "
            f"Empty when the drag never touched a's own panel (committed bundle: "
            f"{_BUNDLE_PATH})"
        )

        # Positive control: a brush dragged over a's OWN target, in this
        # same scene, must still drive a's own selection.
        ax, ay = a_target
        a_start = pc.svg_to_client(rect, svg_w, svg_h, (ax - pad, ay - pad))
        a_end = pc.svg_to_client(rect, svg_w, svg_h, (ax + pad, ay + pad))
        pc.brush_drag(page, a_start, a_end)
        after_a_drag = pc.sample_pixels(page, pc.screenshot(page), [dev_a])[0]
        assert rgb_distance3(after_a, after_a_drag) > 10, (
            f"brushing a's own target in this same scene left a's pixel unchanged "
            f"(after-b-drag {after_a}, after-a-drag {after_a_drag}) -- the isolation "
            f"check above cannot be trusted if a's own selection never responds to its "
            f"own input either (committed bundle: {_BUNDLE_PATH})"
        )
    finally:
        page.close()


def test_vacuity_control_declaring_charts_own_brush_always_registers(
    browser: Any, tmp_path: Path
) -> None:
    """Standalone vacuity control for the owner-panel brush-input test above.

    Proves a brush drag registers at all, on this exact two-chart layout, on
    a chart's own declared selection -- pulled out on its own so a
    regression that makes dragging inert cannot hide behind the isolation
    test above, which deliberately gives the dragged-on chart no conditional
    of its own.
    """
    brush = fm.selection_interval(name="brush_vacuity")
    cond = brush.when(value(RED)).otherwise(value(ORANGE))
    chart_a = (
        fm.Chart(_DF_A)
        .mark_point(**_style(BASE_A))
        .encode(x="x", y="y", color=cond)
        .add_selection(brush)
    )
    chart_b = fm.Chart(_DF_A).mark_point(**_style(BASE_B)).encode(x="x", y="y")
    combo = chart_a | chart_b
    svg = combo.to_svg()
    svg_w, svg_h = pc.parse_svg_size(svg)
    target = pc.parse_circles(svg)[2][:2]  # chart a's own middle point

    html_path = tmp_path / "chart.html"
    combo.interactive().save(str(html_path))
    page = browser.new_page(
        viewport={"width": svg_w, "height": svg_h}, device_scale_factor=pc.DEVICE_SCALE_FACTOR
    )
    try:
        pc.load_interactive(page, html_path)
        rect = pc.canvas_rect(page)
        dev = pc.svg_to_device(rect, svg_w, svg_h, target)
        before = pc.sample_pixels(page, pc.screenshot(page), [dev])[0]

        pad = 40.0
        tx, ty = target
        start = pc.svg_to_client(rect, svg_w, svg_h, (tx - pad, ty - pad))
        end = pc.svg_to_client(rect, svg_w, svg_h, (tx + pad, ty + pad))
        pc.brush_drag(page, start, end)

        after = pc.sample_pixels(page, pc.screenshot(page), [dev])[0]
        assert rgb_distance3(before, after) > 10, (
            f"dragging a brush over a chart's own target must change its own pixel "
            f"(before {before}, after {after}) -- the drag never registered "
            f"(committed bundle: {_BUNDLE_PATH})"
        )
    finally:
        page.close()


# ── Layered brush: every layer of a nested LayerChart is an owner ───────────


@pytest.mark.parametrize("layered", [False, True], ids=["plain_chart", "layer_0_of_layerchart"])
def test_brush_drives_a_layer_0_selection_inside_a_nested_layerchart(
    layered: bool, browser: Any, tmp_path: Path
) -> None:
    """A brush over a LayerChart drives a selection its bottom layer declares.

    ``LayerChart(points, line) | other`` draws one panel per layer on one
    shared rect, and the brush lands on whichever layer is on top (the
    line). The LayerChart is the chart that declares the selection, so every
    one of its layers' panels owns it. ``plain_chart`` is the positive
    control: the same points, brushed the same way, as a plain chart beside
    ``other``.
    """
    brush = fm.selection_interval(name="layered_brush")
    cond = brush.when(value(RED)).otherwise(value(ORANGE))
    points = (
        fm.Chart(_DF_A)
        .mark_point(**_style(BASE_A))
        .encode(x="x", y="y", color=cond)
        .add_selection(brush)
    )
    line = fm.Chart(_DF_A).mark_line().encode(x="x", y="y")
    other = fm.Chart(_DF_A).mark_point(**_style(BASE_B)).encode(x="x", y="y")
    combo = (fm.LayerChart(points, line) if layered else points) | other
    svg = combo.to_svg()
    svg_w, svg_h = pc.parse_svg_size(svg)
    circles = pc.parse_circles(svg)
    assert len(circles) == 10, circles
    target = circles[2][:2]  # the points layer's own middle point

    html_path = tmp_path / "chart.html"
    combo.interactive().save(str(html_path))
    page = browser.new_page(
        viewport={"width": svg_w, "height": svg_h}, device_scale_factor=pc.DEVICE_SCALE_FACTOR
    )
    try:
        pc.load_interactive(page, html_path)
        rect = pc.canvas_rect(page)
        dev = pc.svg_to_device(rect, svg_w, svg_h, target)
        before = pc.sample_pixels(page, pc.screenshot(page), [dev])[0]

        pad = 40.0
        tx, ty = target
        start = pc.svg_to_client(rect, svg_w, svg_h, (tx - pad, ty - pad))
        end = pc.svg_to_client(rect, svg_w, svg_h, (tx + pad, ty + pad))
        pc.brush_drag(page, start, end)

        after = pc.sample_pixels(page, pc.screenshot(page), [dev])[0]
        assert rgb_distance3(before, after) > 10, (
            f"a brush over the points' own target left their pixel unchanged (before "
            f"{before}, after {after}, layered={layered}) -- the selection the points "
            f"declare must take input from every panel of the chart that declares it "
            f"(committed bundle: {_BUNDLE_PATH})"
        )
    finally:
        page.close()


# ── Faceted brush: every facet panel is an owner ────────────────────────────


@pytest.mark.parametrize("facet_index", [0, 1])
def test_faceted_brush_fires_from_every_facet_panel(
    facet_index: int, browser: Any, tmp_path: Path
) -> None:
    """Every facet panel of one faceted chart is an owner of its one selection.

    Parametrized over both facet panels: a regression that hardcodes
    ownership to one specific panel (first or last) fails on whichever
    panel it does not cover.
    """
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 1.0, 5.0, 3.0]
    df = pl.DataFrame({"x": xs + xs, "y": ys + ys, "g": ["p0"] * 5 + ["p1"] * 5})
    sel = fm.selection_interval(name="facet_brush")
    cond = sel.when(value(RED)).otherwise(value(ORANGE))
    chart = (
        fm.Chart(df)
        .mark_point(**_style(BASE_A))
        .encode(x="x", y="y", color=cond)
        .add_selection(sel)
        .facet(col="g")
    )
    svg = chart.to_svg()
    svg_w, svg_h = pc.parse_svg_size(svg)
    circles = pc.parse_circles(svg)
    assert len(circles) == 10, circles
    # Each facet's own second point: clear of both the scene's outer edge
    # and the seam between the two facet panels.
    target = circles[facet_index * 5 + 1][:2]

    html_path = tmp_path / "chart.html"
    chart.interactive().save(str(html_path))
    page = browser.new_page(
        viewport={"width": svg_w, "height": svg_h}, device_scale_factor=pc.DEVICE_SCALE_FACTOR
    )
    try:
        pc.load_interactive(page, html_path)
        rect = pc.canvas_rect(page)
        dev = pc.svg_to_device(rect, svg_w, svg_h, target)
        before = pc.sample_pixels(page, pc.screenshot(page), [dev])[0]

        pad = 40.0
        tx, ty = target
        start = pc.svg_to_client(rect, svg_w, svg_h, (tx - pad, ty - pad))
        end = pc.svg_to_client(rect, svg_w, svg_h, (tx + pad, ty + pad))
        pc.brush_drag(page, start, end)

        after = pc.sample_pixels(page, pc.screenshot(page), [dev])[0]
        assert rgb_distance3(before, after) > 10, (
            f"a brush dragged over facet panel {facet_index}'s own point must still drive "
            f"the one shared selection (before {before}, after {after}) (committed bundle: "
            f"{_BUNDLE_PATH})"
        )
    finally:
        page.close()


# ── Single-chart byte-identity ───────────────────────────────────────────────


def _render(chart: fm.Chart) -> tuple[str, bytes]:
    """Render a chart to (scene_json_str, packed_bytes) -- mirrors
    ``tests/test_html_export_regression.py``'s own helper of the same name."""
    spec, data, viewport, theme, _ = chart._render_inputs()
    return render_interactive(spec, data, viewport=viewport, theme=theme)


def test_single_chart_interactive_scene_selection_spec_is_byte_identical() -> None:
    """A single-panel interactive scene's own selection spec must not move.

    A scene with exactly one panel has exactly one possible owner for each
    of its own selections, so adding per-selection panel ownership to a
    multi-panel scene must leave a single-panel scene's own selection spec
    untouched.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 1.0], "g": ["a", "b", "c"]})
    sel = fm.selection_point(name="single_sel", clear="false")
    cond = sel.when(value(RED)).otherwise(value(BASE_A))
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=cond)
        .add_selection(sel)
        .properties(width=300, height=200)
    )
    json_str, _ = _render(chart)
    scene = json.loads(json_str)

    expected = [
        {
            "type": "point",
            "name": "single_sel",
            "fields": None,
            "encodings": None,
            "nearest": False,
            "toggle": "shift_key",
            "on": "click",
            "clear": "click",
            "resolve": "global",
        }
    ]
    assert scene["selections"] == expected, (
        f"a single-panel interactive scene's selection spec moved -- expected {expected}, "
        f"got {scene['selections']}"
    )
