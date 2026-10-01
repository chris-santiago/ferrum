"""`#247`A: a scale ferrum injected is not a scale the caller wrote.

`mark_bar` gets a `{"type": "linear", "zero": True}` scale stamped onto its
quantitative value channel by `_spec_build.py`, so that a bar chart always
starts at zero. Rust's two "is there an explicit scale here" gates could only
ask ``enc.scale.is_some()``, which answers "yes" for ferrum's own injection —
so a zero-anchored bar was locked out of every cross-mark domain union in both
directions: it could neither contribute to one
(`composite::leaf_positional_domain` excluded it) nor receive one
(`positional::build_axis_scale`'s bypass returned before either was read).

The visible result was not a slightly-wrong axis. It was two layers drawn
against two different scales under one axis: `fm.layer(bar(1..3),
point(40..80))` rendered ticks ``0..3`` with the point layer's marks placed on
its own private ``40..80`` scale, and swapping the argument order changed only
which layer's domain the ticks described, never where either layer's marks
landed.

**Both spellings are covered here, because they lower differently.**
`fm.layer(bar, point)` becomes a composite overlay tree of single-mark leaves,
and `bar + point` becomes one flat `Chart` with two layers whose chart-level
encoding is the LHS bar's — so the injected scale sits in a different place and
the union it must yield to is a different mechanism (a resolved shared domain
vs. `domain::numeric_domain_union`'s cross-layer union). Fixing one and not the
other is what `#247`'s own text ("layer a `mark_bar` chart and a `mark_point`
chart built from two different DataFrames") forbids: `+` is how a reader would
most naturally write that sentence. The `+` form's pre-fix asymmetry is the
tell that it was the same defect — `bar + point` rendered ``0..3`` while
`point + bar` rendered the correct ``0..80``, because only the first puts an
injected scale on the chart-level encoding.

So the assertions here are not "the axis reaches 80". They are: **the marks of
BOTH layers sit at the pixels this chart's own rendered tick ladder maps their
data values to.** That property is false pre-fix in every argument order, in
both spellings, and under `CoordFlip`, and it cannot be satisfied by a union
that forgets the anchor or by an anchor that forgets the union.

`mark_bar(zero=False)` is the discriminating control throughout: it suppresses
the injection, already unioned correctly before this fix, and must be
untouched by it. A test that omitted it would pass for a change that simply
stopped excluding explicit-scale leaves from sharing — which would also break
every user-written `scale=`.
"""

from __future__ import annotations

import json
import re
import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import ChartSpec as _ChartSpec

from _shared_axis_refusal import composite_wrapped_refusal, refusal
from _svg_extents import axis_tick_labels, pixel_for, value_axis_ladder, value_axis_ladders
from _svg_rects import data_bar_rects, mark_layer, svg_elements

#: Pixel coordinates are emitted rounded to 3 decimals, so a tolerance below
#: ~1e-3 would be asserting on the formatter rather than on the geometry.
EPS = 1e-3

BAR_VALUES = [1.0, 2.0, 3.0]
POINT_VALUES = [40.0, 60.0, 80.0]

BAR_FRAME = pl.DataFrame({"cat": ["a", "b", "c"], "v": BAR_VALUES})
#: A SECOND DataFrame, per the issue's own repro: the two leaves share a
#: categorical x but carry their own rows, which is what makes this a composite
#: overlay tree rather than one merged batch.
POINT_FRAME = pl.DataFrame({"cat": ["a", "b", "c"], "v": POINT_VALUES})


def bar_chart(*, flip: bool = False, zero: bool = True, scale: dict | None = None):
    y = fm.Y("v", scale=scale) if scale is not None else "v"
    chart = fm.Chart(BAR_FRAME).mark_bar(zero=zero).encode(x="cat", y=y)
    return chart.coord(fm.CoordFlip()) if flip else chart


def point_chart(*, flip: bool = False, scale: dict | None = None):
    y = fm.Y("v", scale=scale) if scale is not None else "v"
    chart = fm.Chart(POINT_FRAME).mark_point().encode(x="cat", y=y)
    return chart.coord(fm.CoordFlip()) if flip else chart


def point_marks(svg: str, *, axis: str) -> list[float]:
    """Each point mark's coordinate ON the value axis, ascending."""
    key = "cx" if axis == "x" else "cy"
    return sorted(float(a[key]) for a in svg_elements(svg, "circle", lambda a: key in a))


def bar_value_ends(svg: str, *, axis: str) -> list[float]:
    """Each bar's value-end edge -- the edge that moves with the datum.

    A bar grows from its baseline toward its value, so the moving edge is the
    rect's ``y`` (top) on a vertical bar and ``x + width`` (right) on a
    `CoordFlip`'d one. The baseline edge is deliberately NOT read: it clamps to
    the panel edge, which says nothing about the scale.
    """
    rects = data_bar_rects(svg)
    if axis == "x":
        return sorted(float(a["x"]) + float(a["width"]) for a in rects)
    return sorted(float(a["y"]) for a in rects)


def assert_marks_match_the_ladder(svg: str, *, axis: str, bar_values, point_values) -> None:
    """Both layers' marks sit where this chart's own ticks say their values are.

    The whole defect in one assertion: pre-fix the ticks described one layer's
    domain while the other layer's marks were placed on a different one, so the
    two sides of every comparison below came from different scales.

    Counts are checked before positions because the pre-fix failure had two
    shapes and they deserve distinct messages: on the composite path all three
    marks were drawn (on the wrong scale), while on the merged-flat path the
    out-of-domain ones were dropped entirely -- `to_pixel_f64` refuses a value
    outside the resolved domain, so a clipped layer renders FEWER marks, not
    misplaced ones.
    """
    ladder = value_axis_ladder(svg, axis=axis)
    got_points = point_marks(svg, axis=axis)
    assert len(got_points) == len(point_values), (
        f"expected {len(point_values)} point marks, got {len(got_points)}: the point "
        f"layer's values fell outside the resolved domain and were dropped "
        f"({axis} ladder {ladder})"
    )
    got_bars = bar_value_ends(svg, axis=axis)
    assert len(got_bars) == len(bar_values), (
        f"expected {len(bar_values)} bars, got {len(got_bars)} ({axis} ladder {ladder})"
    )
    expected_points = sorted(pixel_for(ladder, v) for v in point_values)
    for got, want in zip(got_points, expected_points, strict=True):
        assert got == pytest.approx(want, abs=EPS), (
            f"point mark at {got} but the {axis} ladder {ladder} places its value at {want}"
        )
    expected_bars = sorted(pixel_for(ladder, v) for v in bar_values)
    for got, want in zip(got_bars, expected_bars, strict=True):
        assert got == pytest.approx(want, abs=EPS), (
            f"bar value edge at {got} but the {axis} ladder {ladder} places its value at {want}"
        )


#: The warning's own text up to its placement clause. Filtering on it (rather
#: than on "no warnings at all") keeps the negative controls discriminating when
#: a render also warns for an unrelated reason.
_EXCLUDED_LAYER_WARNING = re.compile(
    r"^layer \d+: its \w+ scale differs from the one layer \d+ establishes the \w+ axis with"
)

#: Each spelling's placement clause. `+` places the excluded layer's marks on
#: the shared axis (so they may fall outside it); `fm.layer` draws them on the
#: layer's own scale, which no axis labels.
_EXCLUDED_LAYER_PLACEMENT = {
    "plus": "its marks are placed on that axis and may fall outside it",
    "layer": "its marks draw on the layer's own scale, not the labelled axis",
}


def excluded_layer_warning(spelling: str, *, layer: int, establishing: int, channel: str) -> str:
    """The exact warning a layer left out of the establishing scale emits."""
    return (
        f"layer {layer}: its {channel} scale differs from the one layer {establishing} "
        f"establishes the {channel} axis with, so it stays out of that axis; "
        f"{_EXCLUDED_LAYER_PLACEMENT[spelling]}"
    )


def render(chart) -> tuple[str, list[str]]:
    """`chart.to_svg()` plus every excluded-layer warning the render emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    excluded = [str(w.message) for w in caught if _EXCLUDED_LAYER_WARNING.match(str(w.message))]
    return svg, excluded


def ladder_span(svg: str, *, axis: str) -> tuple[float, float]:
    ladder = value_axis_ladder(svg, axis=axis)
    return (min(ladder), max(ladder))


# ── the two spellings ───────────────────────────────────────────────────────

#: The two ways a reader writes "overlay these two charts". They lower
#: differently (composite overlay tree vs. one flat `Chart` with layers), reach
#: the provenance gate through different arms, and take their zero anchor from
#: different mechanisms -- so every union assertion below runs against both.
SPELLINGS = [
    pytest.param("layer", id="fm.layer"),
    pytest.param("plus", id="chart-plus"),
]
ORDERS = [
    pytest.param(("bar", "point"), id="bar-first"),
    pytest.param(("point", "bar"), id="point-first"),
]


def overlay(order, *, spelling: str, point_scale: dict | None = None, **bar_kwargs):
    """`bar` and `point` overlaid in `order`, written the `spelling` way.

    `point_scale` is the point leaf's own `y` scale (`None` by default, i.e.
    no scale declared at all) -- the sibling-join matrix below is the only
    caller that varies it.
    """
    built = [
        bar_chart(**bar_kwargs) if name == "bar" else point_chart(scale=point_scale)
        for name in order
    ]
    if spelling == "layer":
        return fm.layer(*built)
    return built[0] + built[1]


def flipped_overlay(*, spelling: str):
    """The bar-first overlay with its value axis flipped onto x.

    The flip goes on each leaf for `fm.layer` (a composite tree honors per-leaf
    coord) and on the combined chart for `+` (one flat `Chart`, one coord) --
    two spellings of the same picture, which is the point: the injected scale is
    authored on ``y`` and must be recognized on ``x`` either way.
    """
    if spelling == "layer":
        return fm.layer(bar_chart(flip=True), point_chart(flip=True))
    return (bar_chart() + point_chart()).coord(fm.CoordFlip())


# ── the union itself ────────────────────────────────────────────────────────


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("order", ORDERS)
def test_zero_anchored_bar_overlay_spans_the_union_in_either_order(order, spelling):
    """The shared y axis covers zero AND the point layer's 80.

    Both ends are asserted because the pre-fix failure landed on a different end
    per case: `fm.layer` bar-first and `bar + point` both rendered ``0..3``
    (zero, no 80), `fm.layer` point-first rendered ``40..80`` (80, no zero), and
    `point + bar` was already correct. Asserting only "the axis reaches 80" would
    have passed one broken case, and asserting only "it reaches zero" would have
    passed two.
    """
    svg = overlay(order, spelling=spelling).to_svg()
    assert ladder_span(svg, axis="y") == (0.0, 80.0)


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("order", ORDERS)
def test_both_layers_marks_land_on_the_shared_axis(order, spelling):
    svg = overlay(order, spelling=spelling).to_svg()
    assert_marks_match_the_ladder(svg, axis="y", bar_values=BAR_VALUES, point_values=POINT_VALUES)


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_argument_order_does_not_change_the_rendered_scale(spelling):
    """Overlay order picks paint order, not the domain.

    This is the asymmetry that identified `+` as the same defect rather than a
    neighbour of it: pre-fix the two orders resolved different ladders in both
    spellings, because only the bar-first order puts an injected scale where the
    axis reads it. Stated as "both orders resolve the same ladder", not as "the
    two SVGs differ" -- and paired with the absolute span above, since two
    renders agreeing on a WRONG ladder would satisfy this on its own.
    """
    first = overlay(("bar", "point"), spelling=spelling).to_svg()
    second = overlay(("point", "bar"), spelling=spelling).to_svg()
    assert value_axis_ladder(first, axis="y") == value_axis_ladder(second, axis="y")


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_flipped_overlay_shares_its_value_axis_on_x(spelling):
    """`CoordFlip` moves the value axis to x; the union has to follow it.

    The injected scale is authored on ``y`` and reaches the resolve pass on
    ``x`` (the rendering encoding is post-flip), so a fix that matched on the
    authored channel would leave this case broken -- measured pre-fix: x ticks
    ``0..3`` with the point layer on its own 40..80 x scale.
    """
    svg = flipped_overlay(spelling=spelling).to_svg()
    assert ladder_span(svg, axis="x") == (0.0, 80.0)
    assert_marks_match_the_ladder(svg, axis="x", bar_values=BAR_VALUES, point_values=POINT_VALUES)


# ── the control: mark_bar(zero=False) is untouched ──────────────────────────


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_zero_false_bar_overlay_keeps_its_unanchored_union(spelling):
    """The control. `zero=False` suppresses the injection, so this overlay already
    unioned correctly in both spellings; the fix must not move it, and must not
    smuggle a zero anchor into it.

    The span is the data's own ``1..80``, so a ``0`` lower bound here would mean
    the anchor had stopped being conditional on the caller's request — which is
    the failure mode of "just always anchor a bar at zero", the shortcut this
    whole provenance mechanism exists to avoid.
    """
    svg = overlay(("bar", "point"), spelling=spelling, zero=False).to_svg()
    lo, hi = ladder_span(svg, axis="y")
    assert hi == 80.0
    assert lo > 0.0, "an opted-out bar must not anchor the shared axis at zero"
    assert_marks_match_the_ladder(svg, axis="y", bar_values=BAR_VALUES, point_values=POINT_VALUES)


#: The exact rendered y span of `test_user_written_scale_on_a_bar_still_wins_over_the_union`'s
#: `reverse=True` overlay, keyed by spelling. The two routes genuinely
#: diverge: the bar's zero anchor is merged into the chart-level scale copy
#: only. `fm.layer`'s composite route resolves each leaf's axis from that
#: chart-level copy, so it reaches zero; `+`'s flat route resolves the
#: shared axis from the union of the layers' own copies, which do not carry
#: the anchor, so its span starts at the lowest tick the data union and
#: `reverse=True`'s own padding produce instead of zero. This is a known
#: divergence, pinned here until the injected anchor carries provenance on
#: both routes, not something this test should paper over.
_REVERSED_BAR_SPAN = {"layer": (0.0, 80.0), "plus": (10.0, 80.0)}


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_user_written_scale_on_a_bar_still_wins_over_the_union(spelling):
    """A caller's own `scale=` on the bar layer keeps its own field, even
    though the point sibling -- which declares no `scale=` at all on `y` --
    now joins the same axis and contributes its own data to the domain.

    `reverse=True` is the discriminating field: it flips which end of the
    rendered pixel range the low value maps to. It is checked against a
    no-scale control rather than asserted in isolation, because a change
    that dropped the caller's scale entirely and fell back to the
    auto-resolved one would satisfy every other assertion in this file and
    still pass here undetected. The exact span differs by spelling -- see
    `_REVERSED_BAR_SPAN`'s comment for why the two routes are not expected
    to agree here.
    """
    with_scale = overlay(("bar", "point"), spelling=spelling, scale={"reverse": True}).to_svg()
    control = overlay(("bar", "point"), spelling=spelling).to_svg()
    with_ladder = value_axis_ladder(with_scale, axis="y")
    control_ladder = value_axis_ladder(control, axis="y")
    common = sorted(set(with_ladder) & set(control_ladder))
    assert len(common) >= 2, "the two ladders share no ticks to compare direction with"
    lo, hi = common[0], common[-1]
    assert (with_ladder[hi] > with_ladder[lo]) != (control_ladder[hi] > control_ladder[lo]), (
        "reverse=True must flip which end of the pixel range the low value maps to"
    )
    assert_marks_match_the_ladder(
        with_scale, axis="y", bar_values=BAR_VALUES, point_values=POINT_VALUES
    )
    assert ladder_span(with_scale, axis="y") == _REVERSED_BAR_SPAN[spelling]


# ── a domain-less caller scale and a scale-less sibling now join ───────────

#: The bar's own domain-less scale (states no `domain`, so its extent comes
#: from the union rather than a stated range). `zero` is explicit here
#: (rather than relying on the bar's own zero-anchor injection) so the same
#: scale, byte for byte, can also be given to the point sibling for the
#: "equal-scale sibling" cell below -- injection only ever stamps the BAR's
#: own copy, which would make an "equal" sibling scale impossible to
#: construct.
_SIBLING_JOIN_SHARED_SCALE = {"padding": 0.2, "zero": True}
#: A scale that differs from `_SIBLING_JOIN_SHARED_SCALE` in one field -- the
#: different-scale cell's whole point is that this must NOT match.
_SIBLING_JOIN_DIFFERENT_SCALE = {"padding": 0.6, "zero": True}

_SIBLING_JOIN_CASES = [
    pytest.param(None, True, id="scale-less-sibling"),
    pytest.param(_SIBLING_JOIN_SHARED_SCALE, True, id="equal-scale-sibling"),
    pytest.param(_SIBLING_JOIN_DIFFERENT_SCALE, False, id="different-scale-sibling"),
]


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("sibling_scale, joins", _SIBLING_JOIN_CASES)
def test_domainless_caller_scale_sibling_join_matrix(spelling, sibling_scale, joins):
    """A domain-less caller scale on one layer and a sibling layer that
    declares no scale at all on the same channel now join -- the sibling's
    data spans the axis together with the caller's, on both layering routes
    (`fm.layer` and `+`). A sibling carrying an equal copy of the caller's
    scale already joined before this case existed; this matrix re-proves it
    alongside the new case rather than trusting it stayed true. A sibling
    carrying a different scale is the vacuity control -- without it, a
    change that joined every sibling regardless of scale identity would
    pass this test undetected.
    """
    svg, excluded = render(
        overlay(
            ("bar", "point"),
            spelling=spelling,
            point_scale=sibling_scale,
            scale=_SIBLING_JOIN_SHARED_SCALE,
        )
    )
    if joins:
        assert_marks_match_the_ladder(
            svg, axis="y", bar_values=BAR_VALUES, point_values=POINT_VALUES
        )
        assert excluded == [], "a sibling that joins the axis must not warn"
    else:
        span = ladder_span(svg, axis="y")
        assert span[1] == 3.0, "a sibling with a different scale must stay excluded from the union"
        assert excluded == [excluded_layer_warning(spelling, layer=1, establishing=0, channel="y")]


# ── the join does not depend on where the caller-scale layer sits ─────────

#: Three point layers, no bar -- the bar zero-anchor mechanics this file
#: pins separately do not enter into this section at all.
_ORDER_CALLER_VALUES = [1.0, 2.0, 3.0]
_ORDER_SIBLING1_VALUES = [10.0, 20.0, 30.0]
_ORDER_SIBLING2_VALUES = [50.0, 60.0, 70.0]

_ORDER_CALLER_FRAME = pl.DataFrame({"cat": ["a", "b", "c"], "v": _ORDER_CALLER_VALUES})
_ORDER_SIBLING1_FRAME = pl.DataFrame({"cat": ["a", "b", "c"], "v": _ORDER_SIBLING1_VALUES})
_ORDER_SIBLING2_FRAME = pl.DataFrame({"cat": ["a", "b", "c"], "v": _ORDER_SIBLING2_VALUES})

_ORDER_CALLER_SCALE = {"padding": 0.2}
#: A scale that differs from `_ORDER_CALLER_SCALE` -- the different-scale
#: control's whole point is that this must NOT match.
_ORDER_DIFFERENT_SCALE = {"padding": 0.6}


def _order_point_chart(frame, *, scale: dict | None = None):
    y = fm.Y("v", scale=scale) if scale is not None else "v"
    return fm.Chart(frame).mark_point().encode(x="cat", y=y)


def _three_layer_overlay(
    position: str,
    *,
    spelling: str,
    sibling1_scale: dict | None = None,
    sibling2_scale: dict | None = None,
):
    """The caller-scale layer plus two siblings, with the caller-scale layer
    at `position` ("first", "middle" or "last") among the three."""
    caller = _order_point_chart(_ORDER_CALLER_FRAME, scale=_ORDER_CALLER_SCALE)
    sibling1 = _order_point_chart(_ORDER_SIBLING1_FRAME, scale=sibling1_scale)
    sibling2 = _order_point_chart(_ORDER_SIBLING2_FRAME, scale=sibling2_scale)
    charts = {
        "first": [caller, sibling1, sibling2],
        "middle": [sibling1, caller, sibling2],
        "last": [sibling1, sibling2, caller],
    }[position]
    if spelling == "layer":
        return fm.layer(*charts)
    a, b, c = charts
    return a + b + c


def assert_all_point_marks_match_the_ladder(svg: str, *, axis: str, values) -> None:
    """Every point mark, across every layer, sits where this chart's own
    ladder says its value is -- the multi-layer sibling of
    `assert_marks_match_the_ladder` above, for a fixture built entirely from
    point layers.
    """
    ladder = value_axis_ladder(svg, axis=axis)
    got = point_marks(svg, axis=axis)
    assert len(got) == len(values), (
        f"expected {len(values)} point marks, got {len(got)} ({axis} ladder {ladder})"
    )
    expected = sorted(pixel_for(ladder, v) for v in values)
    for got_px, want_px in zip(got, expected, strict=True):
        assert got_px == pytest.approx(want_px, abs=EPS), (
            f"point mark at {got_px} but the {axis} ladder {ladder} places its value at {want_px}"
        )


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_three_layer_join_is_order_independent(position, spelling):
    """The caller-scale layer's position among three overlaid point layers
    must not change whether the two scale-less siblings join its axis.

    Every mark across all three layers must sit where this chart's own
    ladder says its value is, regardless of which layer carries the caller's
    scale. On `fm.layer`, that ladder must also equal the `+` route's ladder
    for the identical order -- proving the two routes agree rather than each
    merely failing to contradict its own (possibly wrong) ladder.
    """
    svg, excluded = render(_three_layer_overlay(position, spelling=spelling))
    assert excluded == [], "scale-less siblings join the axis, so none may warn"
    all_values = _ORDER_CALLER_VALUES + _ORDER_SIBLING1_VALUES + _ORDER_SIBLING2_VALUES
    assert_all_point_marks_match_the_ladder(svg, axis="y", values=all_values)
    if spelling == "layer":
        plus_svg = _three_layer_overlay(position, spelling="plus").to_svg()
        assert value_axis_ladder(svg, axis="y") == value_axis_ladder(plus_svg, axis="y"), (
            "fm.layer must resolve the same axis + resolves for the identical layer order"
        )


#: The control's three layers keep the join test's frames but give the
#: `_ORDER_SIBLING2_*` layer a scale of its own, so two layers carry
#: caller-written scales that differ. The establishing scale is the FIRST
#: caller-written scale in layer order; the scale-less layer joins it and the
#: other scaled layer stays out. Per position of the padding-0.2 layer:
#:
#: - first  ``[0.2, none, 0.6]``: 0.2 establishes.
#: - middle ``[none, 0.2, 0.6]``: 0.2 is still the first scaled layer.
#: - last   ``[none, 0.6, 0.2]``: 0.6 establishes, so the padding-0.2 layer
#:   is the one left out.
#:
#: Values are ``(inside the axis, outside it, (excluded layer, establishing
#: layer))``; the indices are layer positions in that order.
_CONTROL_EXPECTED = {
    "first": (_ORDER_CALLER_VALUES + _ORDER_SIBLING1_VALUES, _ORDER_SIBLING2_VALUES, (2, 0)),
    "middle": (_ORDER_CALLER_VALUES + _ORDER_SIBLING1_VALUES, _ORDER_SIBLING2_VALUES, (2, 1)),
    "last": (_ORDER_SIBLING1_VALUES + _ORDER_SIBLING2_VALUES, _ORDER_CALLER_VALUES, (2, 1)),
}

#: How many of the excluded layer's marks each spelling draws. The two routes
#: diverge: `fm.layer`'s composite route draws the excluded layer on its own
#: scale, off the shared ladder, while `+`'s flat route places its marks on
#: the shared axis, where these fall outside it and are dropped. Each route
#: says which in its excluded-layer warning (`excluded_layer_warning`), so
#: the disposition is never silent.
_EXCLUDED_LAYER_MARKS = {"layer": 3, "plus": 0}


def assert_excluded_layer_disposition(
    svg: str, excluded: list[str], *, spelling: str, inside, outside, layer: int, establishing: int
) -> None:
    """Exactly the `inside` values sit on the shared y ladder, none of the
    `outside` values do, the excluded layer contributes exactly the
    `_EXCLUDED_LAYER_MARKS` count for this spelling, and the render emitted
    exactly one excluded-layer warning, naming `layer` and `establishing`.
    """
    assert excluded == [
        excluded_layer_warning(spelling, layer=layer, establishing=establishing, channel="y")
    ], f"{spelling}: expected one excluded-layer warning, got {excluded}"
    assert values_marked_on_the_ladder(svg, axis="y", values=inside) == inside, (
        f"{spelling}: the layers on the axis {inside} must all sit at their ladder pixels"
    )
    assert values_marked_on_the_ladder(svg, axis="y", values=outside) == [], (
        f"{spelling}: the excluded layer's values {outside} must not sit on the axis"
    )
    got = len(point_marks(svg, axis="y"))
    want = len(inside) + _EXCLUDED_LAYER_MARKS[spelling]
    assert got == want, f"{spelling}: expected {want} point marks, got {got}"


def values_marked_on_the_ladder(svg: str, *, axis: str, values) -> list[float]:
    """The subset of `values` that has a point mark exactly where this chart's
    own ladder places it -- i.e. the values drawn ON the shared axis.
    """
    ladder = value_axis_ladder(svg, axis=axis)
    marks = point_marks(svg, axis=axis)
    return [
        value
        for value in values
        if any(got == pytest.approx(pixel_for(ladder, value), abs=EPS) for got in marks)
    ]


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_three_layer_different_scale_sibling_stays_out_regardless_of_position(position, spelling):
    """The vacuity control for the order-independence matrix above: with two
    layers carrying different caller-written scales, the first one in layer
    order owns the axis, the scale-less layer joins it, and the other scaled
    layer stays out -- identically on `fm.layer` and `+`.
    """
    svg, excluded = render(
        _three_layer_overlay(position, spelling=spelling, sibling2_scale=_ORDER_DIFFERENT_SCALE)
    )
    inside, outside, (layer, establishing) = _CONTROL_EXPECTED[position]
    assert_excluded_layer_disposition(
        svg,
        excluded,
        spelling=spelling,
        inside=inside,
        outside=outside,
        layer=layer,
        establishing=establishing,
    )
    lo, hi = ladder_span(svg, axis="y")
    assert all(v > hi for v in outside) or all(v < lo for v in outside), (
        f"the other scaled layer's values {outside} must lie outside the axis span {(lo, hi)}"
    )
    if spelling == "layer":
        plus_svg, _ = render(
            _three_layer_overlay(position, spelling="plus", sibling2_scale=_ORDER_DIFFERENT_SCALE)
        )
        assert value_axis_ladder(svg, axis="y") == value_axis_ladder(plus_svg, axis="y"), (
            "fm.layer must resolve the same axis + resolves for the identical layer order"
        )


def _both_spellings(*charts):
    """The same layers as `(fm.layer(...), a + b + ...)`."""
    plus = charts[0]
    for chart in charts[1:]:
        plus = plus + chart
    return fm.layer(*charts), plus


def test_categorical_first_scale_owns_the_axis_and_a_later_continuous_scale_renders():
    """A first-layer `BandScale` establishes x and owns it outright; a later
    layer's domain-less continuous x scale does not establish the axis, so the
    chart renders on both spellings with the band's categories as its x ticks.
    """
    band = (
        fm.Chart(_ORDER_CALLER_FRAME)
        .mark_point()
        .encode(x=fm.X("cat", scale=fm.BandScale()), y="v")
    )
    numeric_x = pl.DataFrame({"n": [1.0, 2.0, 3.0], "v": _ORDER_SIBLING1_VALUES})
    linear = fm.Chart(numeric_x).mark_point().encode(x=fm.X("n", scale=fm.LinearScale()), y="v")
    (layered, layered_excluded), (plus, plus_excluded) = (
        render(chart) for chart in _both_spellings(band, linear)
    )
    assert axis_tick_labels(plus, axis="x") == ["a", "b", "c"]
    assert axis_tick_labels(layered, axis="x") == axis_tick_labels(plus, axis="x")
    # The linear layer is the excluded one; see `_EXCLUDED_LAYER_MARKS`.
    for spelling, svg, excluded in (
        ("layer", layered, layered_excluded),
        ("plus", plus, plus_excluded),
    ):
        got = len(point_marks(svg, axis="y"))
        want = len(_ORDER_CALLER_VALUES) + _EXCLUDED_LAYER_MARKS[spelling]
        assert got == want, f"{spelling}: expected {want} point marks, got {got}"
        assert excluded == [
            excluded_layer_warning(spelling, layer=1, establishing=0, channel="x")
        ], f"{spelling}: expected one excluded-layer warning, got {excluded}"


# ── layer 0's categorical column decides the axis ────────────────────────────

_LEAD_NUMERIC_FRAME = pl.DataFrame({"p": [40.0, 60.0, 80.0], "v": [1.0, 2.0, 3.0]})
_LATER_SCALED_FRAME = pl.DataFrame({"n": [1.0, 2.0, 3.0], "v": [4.0, 5.0, 6.0]})
_FLIP_FRAME = pl.DataFrame({"cat": ["a", "b", "c"], "v": [1.0, 5.0, 3.0], "w": [2.0, 3.0, 4.0]})


def _point(frame, **encoding):
    return fm.Chart(frame).mark_point().encode(**encoding)


def _utf8_lead_and_later_linear_x():
    return (
        _point(pl.DataFrame({"cat": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]}), x="cat", y="v"),
        _point(_LATER_SCALED_FRAME, x=fm.X("n", scale=fm.LinearScale()), y="v"),
    )


def _utf8_lead_and_later_flipped_linear_y():
    return (
        _point(_FLIP_FRAME, x="cat", y="v"),
        _point(_FLIP_FRAME, x="w", y=fm.Y("v", scale=fm.LinearScale(nice=False))).coord(
            fm.CoordFlip()
        ),
    )


def _later_unnice_linear_x():
    return _point(_LATER_SCALED_FRAME, x=fm.X("n", scale=fm.LinearScale(nice=False)), y="v")


def _nominal_lead_and_later_linear_x():
    return (_point(_LEAD_NUMERIC_FRAME, x="p:N", y="v"), _later_unnice_linear_x())


def _half_steps_down_from(top: int) -> list[str]:
    return [f"{v / 2:g}" for v in range(2 * top, 1, -1)]


#: Each pair is `(layer 0, layer 1)`, with the x ticks layer 0's categorical
#: column puts on the panel and the shared y ladder. Three categories each:
#: `axis_tick_labels` drops tick groups below its minimum size, so two would
#: read as no ticks at all.
_CATEGORICAL_LEAD_CASES = [
    pytest.param(
        _utf8_lead_and_later_linear_x, ["a", "b", "c"], _half_steps_down_from(6), id="utf8-lead"
    ),
    pytest.param(
        _utf8_lead_and_later_flipped_linear_y,
        ["a", "b", "c"],
        _half_steps_down_from(5),
        id="utf8-lead-coordflip",
    ),
    pytest.param(
        _nominal_lead_and_later_linear_x,
        ["40", "60", "80"],
        _half_steps_down_from(6),
        id="nominal-typed-lead",
    ),
]


@pytest.mark.parametrize(("build", "x_ticks", "y_ticks"), _CATEGORICAL_LEAD_CASES)
def test_a_categorical_layer_0_keeps_its_axis_beside_a_later_continuous_caller_scale(
    build, x_ticks, y_ticks
):
    """When layer 0 binds a categorical column on x (a string column, or a
    numeric one typed `:N`), the panel x is layer 0's categorical axis. A later
    layer's continuous caller scale on x, including one `CoordFlip` moves onto
    x, does not take it: that layer is left out with one excluded-layer
    warning and the chart renders on both spellings.
    """
    (layered, layered_excluded), (plus, plus_excluded) = (
        render(chart) for chart in _both_spellings(*build())
    )
    for svg in (plus, layered):
        assert axis_tick_labels(svg, axis="x") == x_ticks
        assert axis_tick_labels(svg, axis="y") == y_ticks
    # The later layer is the excluded one; see `_EXCLUDED_LAYER_MARKS`.
    for spelling, svg, excluded in (
        ("layer", layered, layered_excluded),
        ("plus", plus, plus_excluded),
    ):
        got = len(point_marks(svg, axis="y"))
        want = 3 + _EXCLUDED_LAYER_MARKS[spelling]
        assert got == want, f"{spelling}: expected {want} point marks, got {got}"
        assert excluded == [
            excluded_layer_warning(spelling, layer=1, establishing=0, channel="x")
        ], f"{spelling}: expected one excluded-layer warning, got {excluded}"


def test_a_numeric_scale_less_layer_0_still_joins_a_later_caller_scale():
    """The control for the categorical-lead cases: an untyped numeric layer 0
    carries no categorical axis, so the later caller scale establishes x and
    layer 0 joins it. The ladder spans both layers, every mark sits on it, and
    no layer is left out.
    """
    lead = _point(_LEAD_NUMERIC_FRAME, x="p", y="v")
    (layered, layered_excluded), (plus, plus_excluded) = (
        render(chart) for chart in _both_spellings(lead, _later_unnice_linear_x())
    )
    all_x = [40.0, 60.0, 80.0, 1.0, 2.0, 3.0]
    for spelling, svg, excluded in (
        ("layer", layered, layered_excluded),
        ("plus", plus, plus_excluded),
    ):
        assert axis_tick_labels(svg, axis="x") == ["10", "20", "30", "40", "50", "60", "70", "80"]
        assert excluded == [], f"{spelling}: expected no excluded-layer warning, got {excluded}"
        assert len(point_marks(svg, axis="x")) == len(all_x)
        assert values_marked_on_the_ladder(svg, axis="x", values=all_x) == all_x, (
            f"{spelling}: both layers' x values must sit at their x-ladder pixels"
        )


def test_stated_domain_first_scale_owns_the_axis_and_a_scale_less_layer_adopts_it():
    """A first-layer scale stating its `domain` owns y outright: a later
    domain-less caller scale does not re-open it, and a scale-less layer
    adopts that axis -- its marks resolve on the stated domain, even though
    no domain union is taken because the domain is stated.
    """
    stated = _order_point_chart(_ORDER_CALLER_FRAME, scale={"domain": [0.0, 40.0]})
    domainless = _order_point_chart(_ORDER_SIBLING2_FRAME, scale=_ORDER_CALLER_SCALE)
    scale_less = _order_point_chart(_ORDER_SIBLING1_FRAME)
    (layered, layered_excluded), (plus, plus_excluded) = (
        render(chart) for chart in _both_spellings(stated, domainless, scale_less)
    )
    assert ladder_span(plus, axis="y") == (0.0, 40.0)
    # The domain-less caller-scale layer is the excluded one; see `_EXCLUDED_LAYER_MARKS`.
    for spelling, svg, excluded in (
        ("plus", plus, plus_excluded),
        ("layer", layered, layered_excluded),
    ):
        assert_excluded_layer_disposition(
            svg,
            excluded,
            spelling=spelling,
            inside=_ORDER_CALLER_VALUES + _ORDER_SIBLING1_VALUES,
            outside=_ORDER_SIBLING2_VALUES,
            layer=1,
            establishing=0,
        )
    assert value_axis_ladder(layered, axis="y") == value_axis_ladder(plus, axis="y")


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_stated_domain_scale_refuses_a_string_layer_on_the_shared_axis(spelling):
    """A scale stating its `domain` still establishes a continuous shared y,
    so a later layer whose y column is a string refuses in the shared-axis
    voice on both spellings, just as it does under a domain-less caller scale.
    """
    stated = _order_point_chart(_ORDER_CALLER_FRAME, scale={"domain": [0, 10]})
    strings = fm.Chart(pl.DataFrame({"cat": ["a", "b", "c"], "s": ["p", "q", "r"]}))
    strings = strings.mark_point().encode(x="cat", y="s")
    layered, plus = _both_spellings(stated, strings)
    chart, expected = {
        "layer": (layered, composite_wrapped_refusal(1, "s", "Utf8")),
        "plus": (plus, refusal(1, "s", "Utf8")),
    }[spelling]
    with pytest.raises(ValueError) as exc:
        chart.to_svg()
    assert str(exc.value) == expected


def test_scale_less_sibling_joins_a_domainless_caller_scale_on_x():
    """The join is per channel, not a y special case: a domain-less caller
    scale on a numeric x and a scale-less x sibling share one x axis, with
    identical ladders on `fm.layer` and `+`.
    """

    def x_point_chart(frame, *, scale: dict | None = None):
        x = fm.X("v", scale=scale) if scale is not None else "v"
        return fm.Chart(frame).mark_point().encode(x=x, y="cat")

    caller = x_point_chart(_ORDER_CALLER_FRAME, scale=_ORDER_CALLER_SCALE)
    scale_less = x_point_chart(_ORDER_SIBLING1_FRAME)
    layered, plus = (chart.to_svg() for chart in _both_spellings(caller, scale_less))
    all_values = _ORDER_CALLER_VALUES + _ORDER_SIBLING1_VALUES
    for name, svg in (("+", plus), ("fm.layer", layered)):
        assert values_marked_on_the_ladder(svg, axis="x", values=all_values) == all_values, (
            f"{name}: both layers' x values must sit at their own x-ladder pixels"
        )
    assert value_axis_ladder(layered, axis="x") == value_axis_ladder(plus, axis="x")


# ── the independent-y layered bar keeps its own primary axis ────────────────


def test_independent_y_layered_bar_primary_axis_stays_the_bars_own_extent():
    """An `independent_y` sibling does NOT widen the primary axis, even though the
    injected scale now yields to the cross-layer union.

    This is not a neighbouring case — it is the mechanism three committed goldens
    depend on. `tests/goldens/secondary_y_axis/{two_layer_dual_axis,
    three_layer_stacked,temporal_numeric_dual_axis}.svg` are all
    ``LayerChart(bar, line, resolve={"y": "independent"})``, whose merged spec
    carries chart-level ``mark: "bar"`` + an injected scale + layers, so the
    widened gate's layered arm fires for them. Their bytes did not move because
    `numeric_domain_union` skips an `independent_y` layer on the y channel, so the
    union reduces to the bar layer's own field, zero-anchored — bit-for-bit what
    the bypass produced.

    Pinned here so that reasoning is a test rather than a coincidence: if the
    `independent_y` skip ever stopped applying, this fails with the sibling's
    ``80`` in the primary span, and those three goldens break at the same moment
    with no explanation of why.
    """
    chart = fm.layer(bar_chart(), point_chart(), resolve={"y": "independent"})
    ladders = value_axis_ladders(chart.to_svg(), axis="y")
    assert len(ladders) == 2, "an independent-y layer draws its own second y axis"
    primary = ladders[0][1]  # leftmost axis: layer 0's, i.e. the bar's
    secondary = ladders[1][1]
    assert (min(primary), max(primary)) == (0.0, 3.0), "the bar's own zero-anchored extent"
    assert max(secondary) == 80.0, "the sibling's extent lives on its own right-hand axis"


# ── faceted layered bar: every panel agrees with its own drawn axis ─────────

#: Two facet panels whose extents are far apart, so a panel drawn against its OWN
#: domain and one drawn against the shared cross-panel domain land in visibly
#: different places. Panel ``A``'s values are the bar-sized ones and ``B``'s the
#: point-sized ones, the same 1..3 / 40..80 split the overlay fixtures use.
FACET_FRAME = pl.DataFrame(
    {
        "cat": ["a", "b", "c", "a", "b", "c"],
        "panel": ["A", "A", "A", "B", "B", "B"],
        "v": BAR_VALUES + POINT_VALUES,
    }
)
#: Each panel's own data, left-to-right in the rendered column facet.
FACET_PANEL_VALUES = [BAR_VALUES, POINT_VALUES]


def faceted_layered_bar(*, zero: bool = True):
    """A `mark_bar` + `mark_line` overlay, column-faceted — a LAYERED bar spec
    whose panels each resolve the same chart-level y encoding."""
    frame = FACET_FRAME
    bar = fm.Chart(frame).mark_bar(zero=zero).encode(x="cat", y="v")
    line = fm.Chart(frame).mark_line().encode(x="cat", y="v")
    return (bar + line).facet(col="panel")


def bars_read_off_each_panels_own_ladder(svg: str) -> list[list[float]]:
    """Per panel, the data values this chart's own axis says its bars represent.

    Bars are assigned to a panel by position: a panel's bars are the ones to the
    right of its y axis and left of the next panel's. That is the same landmark a
    reader uses, and it needs no clip-group bookkeeping — which matters because
    `mark_layer` deliberately concatenates the clip groups, so the SVG's own
    grouping cannot separate panels here.
    """
    ladders = value_axis_ladders(svg, axis="y")
    assert len(ladders) == len(FACET_PANEL_VALUES), (
        f"expected one y ladder per panel; got {len(ladders)}"
    )
    rects = sorted((float(a["x"]), float(a["y"])) for a in data_bar_rects(mark_layer(svg)))
    out: list[list[float]] = []
    for i, (position, ladder) in enumerate(ladders):
        next_position = ladders[i + 1][0] if i + 1 < len(ladders) else float("inf")
        mine = [(x, top) for x, top in rects if position <= x < next_position]
        lo, hi = min(ladder), max(ladder)
        slope = (ladder[hi] - ladder[lo]) / (hi - lo)
        out.append([lo + (top - ladder[lo]) / slope for _, top in mine])
    return out


def test_faceted_layered_bar_panels_agree_with_the_axis_drawn_beside_them():
    """Every panel of a faceted layered bar draws its bars on the domain its own
    axis describes.

    **This is a behavior change this task introduced**, found by the spec review
    and measured both ways. A faceted layered bar reaches the auto path through
    the `spec.layers` arm, so its panels now resolve against the shared
    cross-panel union instead of each panel's own bypassed extent. Before:
    panel A's data ``[1, 2, 3]`` read back as ``[26.667, 53.333, 80.0]`` off the
    shared ``0..80`` ladder drawn right beside it. After: ``[1, 2, 3]``.

    The assertion is per panel and reads the marks BACK through the ladder,
    because everything cheaper is inert here:

    * the ladder spans ``0..80`` in both states (the facet axis was always
      shared), so no assertion about the span discriminates;
    * panel B agrees in both states, because its own extent happens to equal the
      shared one — checking "a panel agrees" without checking *which* panel would
      have passed before the fix.

    Panel A is the whole content of this test. `zero=False` is the control: it
    suppresses the injection, so both panels always agreed, and it must still
    agree now.
    """
    for panel, (got, want) in enumerate(
        zip(
            bars_read_off_each_panels_own_ladder(faceted_layered_bar().to_svg()),
            FACET_PANEL_VALUES,
            strict=True,
        )
    ):
        assert got == pytest.approx(want, abs=1e-2), (
            f"panel {panel}: its bars represent {want} but the axis drawn beside "
            f"them reads {[round(v, 3) for v in got]}"
        )


def test_faceted_layered_bar_keeps_its_zero_anchor_across_panels():
    """The shared facet axis still reaches zero — the union carries the bar's
    anchor across panels rather than dropping it on the way to the auto path.

    Paired with the agreement test above: agreement alone would also hold for a
    union that lost the anchor (every panel would agree on a ``1..80`` axis), and
    an anchor alone was already true before the fix. The defect needed both.
    """
    svg = faceted_layered_bar().to_svg()
    for position, ladder in value_axis_ladders(svg, axis="y"):
        assert (min(ladder), max(ladder)) == (0.0, 80.0), f"ladder at {position}: {ladder}"


def test_faceted_layered_bar_with_zero_false_is_unaffected():
    """The control: `zero=False` suppresses the injection, so these panels took
    the auto path before this task and still do. Their agreement is not evidence
    for the fix — it is evidence the fix did not disturb the path that already
    worked, and that no zero anchor was smuggled into an opted-out bar.
    """
    svg = faceted_layered_bar(zero=False).to_svg()
    for position, ladder in value_axis_ladders(svg, axis="y"):
        assert min(ladder) > 0.0, f"ladder at {position} must not anchor at zero: {ladder}"
    for panel, (got, want) in enumerate(
        zip(bars_read_off_each_panels_own_ladder(svg), FACET_PANEL_VALUES, strict=True)
    ):
        assert got == pytest.approx(want, abs=1e-2), f"panel {panel}: reads {got}, is {want}"


# ── the single-chart render is untouched by construction ────────────────────


def test_single_bar_chart_axis_is_unchanged():
    """A standalone bar has neither a shared domain nor layers to yield to, so it
    still resolves through the explicit-scale path: the axis is its own data
    anchored at zero. This is what keeps every single-bar golden byte-identical.
    """
    svg = bar_chart().to_svg()
    assert ladder_span(svg, axis="y") == (0.0, 3.0)
    ladder = value_axis_ladder(svg, axis="y")
    for got, want in zip(
        bar_value_ends(svg, axis="y"),
        sorted(pixel_for(ladder, v) for v in BAR_VALUES),
        strict=True,
    ):
        assert got == pytest.approx(want, abs=EPS)


# ── the one shape the widened gate would treat differently ──────────────────


def layered_specs(chart) -> list[dict]:
    """Every layered spec dict this chart lowers to, composite leaves included."""
    specs: list[dict] = []
    seen: list[int] = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict) or id(node) in seen:
            return
        seen.append(id(node))
        if node.get("layers"):
            specs.append(node)
        for value in node.values():
            walk(value)

    lowered = chart if hasattr(chart, "to_spec") else chart._build_merged()
    walk(json.loads(lowered.to_spec().to_json()))
    return specs


#: Every public way to reach a spec that carries BOTH a chart-level mark and
#: layers. Each entry is a construction; the assertion below is about the shape
#: it lowers to, not about its pixels.
#:
#: **Extend this list when you add or change a layered lowering.** The tripwire
#: below can only fire for a lowering that is listed here, so a NEW public way to
#: produce a layered spec is outside its reach until it is added — and the shape
#: it guards against (chart-level ``mark: "bar"`` + an injected scale + no bar
#: layer) is one a new lowering could introduce without anyone noticing, since
#: its only symptom is an axis quietly losing its zero anchor. Known upcoming
#: work that touches this: any catplot/displot layer restructuring changes who
#: builds a `Layer`.
#:
#: Per-layer coord (`#172`(1), wave 5 T13) landed and is the reason for the last
#: two entries: the merge now stamps the RHS member's own coordinate system onto
#: the layers it contributes, which is a new layered-lowering SHAPE (a layer
#: carrying a `coord` its siblings do not) reachable from the ordinary `+` and
#: `fm.layer` spellings. Nine of the eleven entries below reach the assertion
#: today (the two point-first entries skip it by design — their chart-level mark
#: is not a bar), so the list is non-vacuous as written.
LAYERED_LOWERINGS = {
    "bar + point": lambda: bar_chart() + point_chart(),
    "point + bar": lambda: point_chart() + bar_chart(),
    "bar + point + bar": lambda: bar_chart() + point_chart() + bar_chart(),
    "(bar + point) flipped": lambda: (bar_chart() + point_chart()).coord(fm.CoordFlip()),
    "bar + hline": lambda: bar_chart() + fm.annotate_hline(2.0),
    "fm.layer merged": lambda: fm.layer(bar_chart(), point_chart()),
    "fm.layer merged reversed": lambda: fm.layer(point_chart(), bar_chart()),
    "fm.layer independent-y": lambda: fm.layer(
        bar_chart(), point_chart(), resolve={"y": "independent"}
    ),
    "fm.layer of a layered bar": lambda: fm.layer(bar_chart() + point_chart(), point_chart()),
    # `#172`(1): the member's own coord rides its layers. Both spellings, since
    # a channel-exchanging coord (`CoordFlip`) and a projecting one
    # (`CoordPolar`) reach different consumers of `Layer.coord`.
    "bar + flipped point member": lambda: bar_chart() + point_chart(flip=True),
    "fm.layer bar + polar point member": lambda: fm.layer(
        bar_chart(), point_chart().coord(fm.CoordPolar(theta="x"))
    ),
}


@pytest.mark.parametrize("name", list(LAYERED_LOWERINGS))
def test_no_public_lowering_produces_a_bar_marked_spec_without_a_bar_layer(name):
    """A layered spec that says ``mark: "bar"`` always has a bar among its layers.

    This is the guard for the one shape where the widened yield behaves
    differently from the old bypass: a layered spec carrying chart-level
    ``mark: "bar"`` and an injected scale but NO bar layer resolves its axis from
    the cross-layer union with no zero anchor (pinned Rust-side by
    ``layered_spec_with_no_bar_layer_takes_the_union_without_an_anchor``). That is
    the right answer there, because `prepare::build_layers` draws layers from
    ``spec.layers`` alone — a layered spec's ``spec.mark`` draws nothing, so there
    is no bar baseline to keep in view.

    No construction in `LAYERED_LOWERINGS` reaches it: every lowering that keeps a
    chart-level bar mark alongside layers derives layer 0 from that same bar
    chart. This test is the tripwire for a future lowering that stops doing so —
    it would then be relying on behavior chosen for an unreachable case, and this
    fails rather than the axis quietly losing its anchor.
    """
    for spec in layered_specs(LAYERED_LOWERINGS[name]()):
        if spec.get("mark") != "bar":
            continue
        y_scale = ((spec.get("encoding") or {}).get("y") or {}).get("scale") or {}
        if not y_scale.get("systemInjected"):
            continue
        marks = [layer.get("mark") for layer in spec["layers"]]
        assert "bar" in marks, (
            f"{name}: a spec with chart-level mark='bar' and an injected y scale "
            f"lowered to layers {marks} with no bar among them — the zero anchor "
            "has nothing to come from on the auto path"
        )


# ── the wire contract ───────────────────────────────────────────────────────


def y_scale_dict(chart) -> dict | None:
    return json.loads(chart.to_spec().to_json())["encoding"]["y"].get("scale")


def test_injected_bar_scale_carries_the_provenance_flag():
    assert y_scale_dict(bar_chart()) == {
        "type": "linear",
        "clamp": False,
        "nice": False,
        "zero": True,
        "systemInjected": True,
    }


def test_zero_false_bar_emits_no_scale_at_all():
    """Nothing to mark: the injection is the only producer of the flag, and
    `zero=False` suppresses the injection rather than emitting an unflagged one.
    """
    assert y_scale_dict(bar_chart(zero=False)) is None


def test_scale_the_caller_wrote_is_not_flagged_as_injected():
    """The injection added ``zero`` to a scale the caller authored, so the scale
    as a whole is still theirs — the flag is withheld, which is what keeps
    `test_user_written_scale_on_a_bar_layer_still_wins_over_sharing` true.
    """
    scale = y_scale_dict(bar_chart(scale={"padding": 0.2}))
    assert scale["zero"] is True
    assert "systemInjected" not in scale


def test_hand_written_system_injected_scale_is_accepted_and_yields():
    """The flag is user-reachable by hand, deliberately, and does exactly what it
    says — no more.

    Nothing can stop a caller writing ``scale={"type": "linear", "zero": True,
    "systemInjected": True}``: the Python producer crosses the same wire boundary
    every user dict crosses, so the key gate has to accept it on ``linear``. This
    pins both halves of the disclosure — it is ACCEPTED (not refused), and what it
    buys is only that the scale stops short-circuiting the cross-mark union, i.e.
    the same render omitting the scale would have produced. There is no behavior
    here a caller could not already reach.
    """
    hand_written = (
        fm.Chart(BAR_FRAME)
        .mark_bar()
        .encode(
            x="cat",
            y=fm.Y("v", scale={"type": "linear", "zero": True, "systemInjected": True}),
        )
    )
    svg = fm.layer(hand_written, point_chart()).to_svg()
    assert ladder_span(svg, axis="y") == (0.0, 80.0)
    assert_marks_match_the_ladder(svg, axis="y", bar_values=BAR_VALUES, point_values=POINT_VALUES)


def test_pre_provenance_spec_payload_round_trips_byte_identically():
    """A payload captured from the running system BEFORE the field existed.

    The new wire key is additive and skipped when unset, so the bytes of an
    older spec must survive `from_json` → `to_json` untouched. Proven by
    round-tripping the captured payload rather than by reading the serde
    attributes, which would pass even if the skip predicate were wrong.
    """
    captured = (
        '{"data":{"kind":"named","name":"default"},"mark":"bar",'
        '"encoding":{"x":{"field":"cat"},"y":{"field":"v","scale":'
        '{"type":"linear","clamp":false,"nice":false,"zero":true}}}}'
    )
    spec = _ChartSpec.from_json(captured)
    assert spec.to_json() == captured
