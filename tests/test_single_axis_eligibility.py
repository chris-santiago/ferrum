"""Single-axis scale resolution is decided by what a spec actually draws.

Rust's scale resolver grants a "single-axis" exemption to charts whose marks
can render from one positional channel alone (a rug's ticks, a rule's span),
synthesizing a dummy ``0…1`` unit scale for the absent axis. It used to decide
that from the ROOT mark slot alone, which a layered spec does not fill
honestly in either direction:

* Python's layer promotion moves the primary mark into the layer list and
  leaves the slot empty; the wire encoder then substitutes a placeholder
  ``"point"``. A chart whose only marks are rug ticks arrived claiming to be a
  point and was refused the exemption.
* ``tick_chart + line_chart`` keeps ``tick`` in the slot while drawing a line
  that needs a real y-domain. It was GRANTED the exemption and rendered the
  line squashed into the bottom sliver of the panel under a fabricated ``0…1``
  axis (measured pre-fix: curve y ∈ [368.3, 416.9] of a 480px canvas, y labels
  ``1, 0.9, … 0``).

No issue was filed for the second case; it was found during wave-4 research
and adjudicated into scope alongside #201. This module pins the predicate
itself — the eligible mark SET is unchanged (tick/rule, and arc under a polar
coord), only which marks the question is asked of, and whether a layer's own
positional binding reaches scale resolution.

The rule the resolver settled on: **every mark that draws must be eligible.**
Two kinds of layer do not vote, each because it makes no demand on the axis in
question — an invisible ``_CarrierLayer`` (``opacity``/``size`` 0, which exists
only so an annotation-only composition has a layer at all) draws nothing, and
an ``independent_y`` layer takes its ``y`` from its own slot. Requiring *every*
drawing mark rather than any is load-bearing: an ``any`` rule let one eligible
mark exempt a mark beside it, so ``mark_point(x=) + mark_tick(x=)`` rendered a
fabricated ladder with the point layer's whole dataset silently dropped.

The ``#201`` composition half (displot's rug+kde root encoding) is proven
separately, in its own module, so neither half can be credited to the other.
"""

from __future__ import annotations

import itertools
import re

import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import axis_tick_labels

# The rug values and the curve's density column. The densities are all well
# below 1 so a fabricated unit scale is unmistakable in the rendered ticks: a
# real y-axis tops out near 0.13, a fabricated one at 1.
_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]
_DENSITIES = [0.02, 0.08, 0.13, 0.07, 0.01]

#: The synthesized dummy scale's rendered y ticks, top-to-bottom.
_UNIT_SCALE_LADDER = ["1", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "0"]


def _is_unit_scale(labels: list[str]) -> bool:
    """Whether a rendered tick ladder is the synthesized ``0…1`` unit scale,
    read in either axis's own direction (y descends, x ascends)."""
    return sorted(labels) == sorted(_UNIT_SCALE_LADDER)


def _polyline_y_extent(svg: str) -> tuple[float, float]:
    """``(min, max)`` of every ``<polyline>`` vertex's screen y."""
    ys = [
        float(pair.split(",")[1])
        for points in re.findall(r'<polyline[^>]*points="([^"]+)"', svg)
        for pair in points.split()
    ]
    assert ys, "expected at least one rendered polyline"
    return min(ys), max(ys)


def _rug_chart() -> fm.Chart:
    return fm.Chart(pl.DataFrame({"v": _VALUES})).mark_tick().encode(x="v")


def _curve_chart() -> fm.Chart:
    data = pl.DataFrame({"v": _VALUES, "d": _DENSITIES})
    return fm.Chart(data).mark_line().encode(x="v", y="d")


def test_tick_plus_line_layering_resolves_the_lines_own_y_scale() -> None:
    """A rug layered under a curve scales the curve by its own density column.

    The discriminating facts are the two the fabricated unit scale got wrong:
    the y ticks read in density units (topping out near 0.13, never the ``0…1``
    ladder), and the curve occupies the panel's real vertical extent instead of
    a sliver at its foot.
    """
    svg = (_rug_chart() + _curve_chart()).to_svg()

    y_labels = axis_tick_labels(svg, axis="y")
    assert not _is_unit_scale(y_labels), (
        "the y axis is still the fabricated unit scale; the line layer's "
        f"density column never reached scale resolution. labels={y_labels}"
    )
    values = [float(label) for label in y_labels]
    assert max(values) == pytest.approx(max(_DENSITIES), abs=0.01), (
        f"y ticks must span the density column's range, got {y_labels}"
    )

    lo, hi = _polyline_y_extent(svg)
    # A 480px canvas: pre-fix the curve spanned ~48px (y ∈ [368.3, 416.9]) at
    # the panel's foot. A real density scale spans nearly the whole plot area.
    assert hi - lo > 300.0, (
        f"the curve is squashed: it spans only {hi - lo:.1f}px of vertical extent"
    )
    assert lo < 100.0, f"the curve never reaches the top of the panel: y_min={lo:.1f}"


def test_rug_only_layering_keeps_the_single_axis_exemption() -> None:
    """Two rugs layered bind no y anywhere, so the dummy unit scale still applies.

    The eligible SET did not change: a spec that genuinely draws only ticks is
    exactly as eligible as before, and the x axis still unions both layers'
    data.
    """
    left = _rug_chart()
    right = fm.Chart(pl.DataFrame({"v": [1.5, 2.5, 3.5]})).mark_tick().encode(x="v")

    svg = (left + right).to_svg()

    assert axis_tick_labels(svg, axis="y") == _UNIT_SCALE_LADDER
    x_values = [float(label) for label in axis_tick_labels(svg, axis="x")]
    assert (min(x_values), max(x_values)) == (min(_VALUES), max(_VALUES))


def test_annotation_only_composition_keeps_its_single_axis_scale() -> None:
    """An invisible carrier layer does not vote, so it cannot refuse this chart.

    ``annotate_hline() + annotate_vline()`` carries its visual content in
    annotation primitives and puts a deliberately invisible ``point`` CARRIER
    layer in the layer list purely so the spec has a layer at all. A rule that
    judged every layer would read that point, find nothing binding x, and refuse
    the chart. The carrier paints nothing (``opacity`` 0), so it makes no demand
    on either axis and the y-only exemption stands.
    """
    combined = fm.annotate_hline(y=3.0, stroke="red") + fm.annotate_vline(x=2.0, stroke="blue")

    svg = combined.to_svg()

    y_values = [float(label) for label in axis_tick_labels(svg, axis="y")]
    assert min(y_values) <= 3.0 <= max(y_values), (
        f"the hline's y=3.0 must sit inside the resolved y domain, got {y_values}"
    )
    # Nothing binds x, so x keeps the synthesized unit scale.
    assert _is_unit_scale(axis_tick_labels(svg, axis="x"))


def test_layered_points_without_any_y_binding_still_refuse() -> None:
    """No eligible mark and no y binding anywhere: the refusal stands.

    Point is not a single-axis mark, and no layer here binds y, so there is
    neither an eligible mark to exempt the spec nor a binding to resolve a real
    axis from. This is the refusal the pre-fix code gave and must keep giving:
    reaching past layer 0 for a binding must not turn "nothing to draw a y axis
    from" into a silently fabricated one.
    """
    data = pl.DataFrame({"v": _VALUES})
    left = fm.Chart(data).mark_point().encode(x="v")
    right = fm.Chart(data).mark_point().encode(x="v")

    with pytest.raises(ValueError, match=r"encoding 'y' expected EncodingSpec"):
        (left + right).to_svg()


def _circle_count(svg: str) -> int:
    return len(re.findall(r"<circle", svg))


@pytest.mark.parametrize("mark", ["point", "bar"])
def test_one_eligible_mark_does_not_exempt_the_mark_beside_it(mark: str) -> None:
    """A mark that needs both axes refuses; it is not dropped onto a fabricated one.

    Under an ``any``-over-marks rule the rug's tick exempted the whole spec, and
    this rendered a ``0…1`` y ladder with ZERO ``<circle>`` elements — the point
    layer's entire dataset silently gone. A silent drop is worse than the typed
    refusal it replaced, so the refusal is the contract.
    """
    data = pl.DataFrame({"v": _VALUES})
    primary = getattr(fm.Chart(data), f"mark_{mark}")().encode(x="v")

    with pytest.raises(ValueError, match=r"encoding 'y' expected EncodingSpec"):
        (primary + _rug_chart()).to_svg()


def test_the_refused_shape_would_otherwise_have_dropped_its_marks() -> None:
    """The same spec with a real ``y`` renders its points — proving the refusal
    above is about the missing axis, not about layering a point under a rug."""
    data = pl.DataFrame({"v": _VALUES, "w": _DENSITIES})
    points = fm.Chart(data).mark_point().encode(x="v", y="w")

    svg = (points + _rug_chart()).to_svg()

    assert _circle_count(svg) == len(_VALUES)
    assert not _is_unit_scale(axis_tick_labels(svg, axis="y"))


def test_independent_y_layer_does_not_claim_the_primary_axis() -> None:
    """A secondary-axis layer's ``y`` stays in its own slot.

    ``LayerChart(rug, curve, resolve={"y": "independent"})`` draws its curve
    against a right-hand axis. Letting that layer supply the PRIMARY binding
    emitted two identical density axes — a left axis duplicating the right one.
    That is a different wrong, not an improvement, so the primary keeps the
    synthesized scale and the real domain lives on the secondary axis only.

    (That the primary is a fabricated ladder at all is a separate, pre-existing
    defect on this shape; this test pins only that the change did not alter it.)
    """
    chart = fm.LayerChart(_rug_chart(), _curve_chart(), resolve={"y": "independent"})

    svg = chart.to_svg()

    primary_y = axis_tick_labels(svg, axis="y")
    assert _is_unit_scale(primary_y), (
        "the primary y axis must keep the synthesized unit scale rather than "
        f"duplicate the independent layer's density domain; got {primary_y}"
    )
    # The density domain is present exactly once, on the secondary axis.
    density_labels = [t for t in re.findall(r"<text[^>]*>([^<]*)</text>", svg) if t == "0.13"]
    assert len(density_labels) == 1, (
        f"the density domain must appear on one axis only, found {len(density_labels)}"
    )


def test_rule_plus_tick_resolves_the_ticks_own_x_scale() -> None:
    """The fallback's x arm, symmetric with the tick+line y case above.

    A rule spanning ``y`` layered with a rug binding ``x``: pre-fix the root
    ``rule`` took the y-only arm and x was the fabricated unit ladder.
    """
    data = pl.DataFrame({"w": _DENSITIES})
    rule = fm.Chart(data).mark_rule().encode(y="w")

    svg = (rule + _rug_chart()).to_svg()

    x_labels = axis_tick_labels(svg, axis="x")
    assert not _is_unit_scale(x_labels), (
        f"the rug layer's x column must drive the x axis, got {x_labels}"
    )
    x_values = [float(label) for label in x_labels]
    assert (min(x_values), max(x_values)) == (min(_VALUES), max(_VALUES))


#: The point/line pair below shares ONE frame deliberately. Composing two
#: different frames renames the RHS columns and routes them through an Identity
#: transform, so an inherited channel naming an LHS column is unreachable from
#: the RHS layer's batch -- measured: the inheriting order draws 0 circles
#: across two frames and 5 across one, on this build and before this change
#: alike. That is a separate, pre-existing cross-frame inheritance gap; pinning
#: the ordering rule on it would be pinning the wrong mechanism.
_PAIR_FRAME = pl.DataFrame({"v": _VALUES, "d": _DENSITIES})


def _point_x() -> fm.Chart:
    return fm.Chart(_PAIR_FRAME).mark_point().encode(x="v")


def _line_xy() -> fm.Chart:
    return fm.Chart(_PAIR_FRAME).mark_line().encode(x="v", y="d")


def test_a_drawing_layer_that_cannot_reach_the_fallbacks_channel_is_refused() -> None:
    """A resolved AXIS is not a plottable FIELD.

    ``mark_point(x=) + mark_line(x=,y=)``: the point is layer 0, so the merged
    chart encoding carries ``x`` alone. The binding fallback resolves the y axis
    from the line layer, but no ``y`` field ever reaches the point — its builder
    reads a column that is not there and returns an empty batch. Measured before
    this guard: rendered with ``<circle>`` count 0 and no warning, where the
    pre-task engine had raised. The refusal names the layer, its mark and the
    channel so the caller can act on it.
    """
    with pytest.raises(ValueError, match=r"layer 0: mark_point .* no 'y' encoding") as excinfo:
        (_point_x() + _line_xy()).to_svg()

    assert "nothing to plot" in str(excinfo.value)


def test_the_same_pair_in_the_inheriting_order_still_draws_its_points() -> None:
    """The working case the rule must not break, and the discriminator for it.

    With the line as layer 0, its ``y`` is merged INTO the chart encoding, so the
    point inherits a real column and draws all five marks. Only inheritance from
    the chart encoding counts as "available" — never the fallback — which is
    exactly what separates this ordering from the refused one above. These two
    tests cannot both pass under a rule that counts the fallback.
    """
    svg = (_line_xy() + _point_x()).to_svg()

    assert _circle_count(svg) == len(_VALUES)
    assert not _is_unit_scale(axis_tick_labels(svg, axis="y"))


def test_the_x_direction_twin_refuses_the_same_way() -> None:
    """The fallback is symmetric, so its guard is too."""
    point_y = fm.Chart(_PAIR_FRAME).mark_point().encode(y="d")

    with pytest.raises(ValueError, match=r"layer 0: mark_point .* no 'x' encoding"):
        (point_y + _line_xy()).to_svg()


def test_an_eligible_layer_beside_the_fallbacks_channel_still_renders() -> None:
    """A rug tick needs only one channel, so a fallback-sourced y cannot strand
    it — this is the acceptance case itself, and it must not start refusing."""
    svg = (_rug_chart() + _curve_chart()).to_svg()

    assert not _is_unit_scale(axis_tick_labels(svg, axis="y"))


# ---------------------------------------------------------------------------
# Mechanism defense: the whole two-layer shape space, not a list of examples
# ---------------------------------------------------------------------------

#: `point`/`line` are the "needs both axes" representatives; `tick`/`rule` are
#: the single-axis-eligible ones. All four are observed, as (element pattern,
#: elements one layer contributes).
#:
#: Observing `tick`/`rule` takes more than adding their element to a presence
#: test, because neither element is theirs alone. A tick and a rule both render
#: as `<line>`, which is also what every axis tick and gridline renders as (a
#: bare `mark_tick` chart carries 5 mark lines among 46 total), and every chart
#: carries a panel `<rect>` whether or not anything drew. A presence test on
#: either element therefore matches chrome and passes for a DROPPED layer --
#: the same artifact-for-property substitution this sweep exists to rule out.
#:
#: Two changes close it together. Look only inside the marks group, which
#: carries no chrome (measured: zero non-mark elements inside it). And compare
#: COUNTS rather than presence, so a pair whose marks share an element type --
#: `tick` + `rule`, or two ticks -- still observes a drop: 10 lines when both
#: draw, 5 when one is dropped. (Both figures are `len(_PAIR_FRAME)` per layer,
#: so they track the fixture rather than restating a literal that can drift
#: from it.)
_SWEEP_MARKS = ("point", "line", "tick", "rule")
_SWEEP_CHANNELS = (("x",), ("y",), ("x", "y"))
_SWEEP_SHAPE = {
    "point": (r"<circle\b", len(_PAIR_FRAME)),
    "line": (r"<polyline\b", 1),
    "tick": (r"<line\b", len(_PAIR_FRAME)),
    "rule": (r"<line\b", len(_PAIR_FRAME)),
}

#: The marks group. Axes, gridlines and the panel background sit outside it.
_MARKS_GROUP = re.compile(r'<g clip-path="url\(#ferrum-clip-\d+\)">(.*?)</g>', re.DOTALL)


def _sweep_chart(mark: str, channels: tuple[str, ...]) -> fm.Chart:
    field = {"x": "v", "y": "d"}
    return getattr(fm.Chart(_PAIR_FRAME), f"mark_{mark}")().encode(
        **{channel: field[channel] for channel in channels}
    )


def _marks_body(svg: str) -> str:
    """The marks group's contents, or `""` when the render drew no marks at all."""
    match = _MARKS_GROUP.search(svg)
    return match.group(1) if match else ""


def test_no_two_layer_composition_renders_a_drawing_layer_empty() -> None:
    """Exhaustive: every ordered pair of (mark, bound channels) either refuses or
    draws every mark it claims.

    Two earlier cycles of this task each moved a shape that the test corpus did
    not contain, because a corpus measures COVERAGE, not reachability. This
    closes the space instead: all 144 ordered pairs over four marks and three
    channel subsets, asserting the wave's constraint directly — a typed refusal
    must not silently become an empty render.

    Measured across the three builds: pre-task 12 silent drops, the intermediate
    fix 24, this one 0. The class is not merely absent from the shapes anyone
    happened to write down; it is absent from the space.

    All four marks are observed, by element COUNT inside the marks group -- see
    `_SWEEP_SHAPE` for why presence alone cannot see a dropped tick or rule.
    """
    drops = []
    for (mark_a, channels_a), (mark_b, channels_b) in itertools.product(
        itertools.product(_SWEEP_MARKS, _SWEEP_CHANNELS), repeat=2
    ):
        composed = _sweep_chart(mark_a, channels_a) + _sweep_chart(mark_b, channels_b)
        try:
            svg = composed.to_svg()
        except ValueError:
            continue  # a refusal is the other acceptable outcome

        # Both layers draw, so each element type carries both layers' share.
        expected: dict[str, int] = {}
        for mark in (mark_a, mark_b):
            pattern, per_layer = _SWEEP_SHAPE[mark]
            expected[pattern] = expected.get(pattern, 0) + per_layer

        body = _marks_body(svg)
        for pattern, want in sorted(expected.items()):
            got = len(re.findall(pattern, body))
            if got != want:
                drops.append(
                    f"{mark_a}{list(channels_a)} + {mark_b}{list(channels_b)}: "
                    f"expected {want} {pattern!r} marks, drew {got}"
                )

    assert drops == [], "compositions that rendered while silently dropping a layer:\n" + "\n".join(
        drops
    )
