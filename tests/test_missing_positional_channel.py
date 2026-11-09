"""A mark that cannot reach a positional channel it needs is refused in its
own voice, naming the channel(s) and a remedy that runs, on every door that
reaches the drawing chokepoint: a primitive called directly, a primitive
replacing a pending statistic (the same replacement rule a figure's post-hoc
mark call already follows), a composite leaf -- an ``hconcat`` leaf and a
static ``LayerChart``/``fm.layer(...)`` member alike -- a facet, and an
interactive export.

What a mark needs depends on the mark: most need both positional channels;
``mark_tick``/``mark_rule`` need only one; ``mark_arc``/``mark_text`` under a
polar coord need only ``theta`` (named ``'theta'``/``'radius'``, never
``'x'``/``'y'``, under that coord). A channel nothing in the whole spec
binds, and a channel only a sibling layer binds, are two clauses of the one
refusal, spoken together when a layer has both. The layered-spec half of the
sibling-bound clause -- a ``+``/``.layer()`` member whose channel a sibling
binds -- is also pinned in ``tests/test_single_axis_eligibility.py``, where
the single-axis-eligibility mechanism it shares is the module's own subject.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import mark_layer, plot_rect, svg_elements

_DF = pl.DataFrame({"v": [1.0, 2.0, 3.0], "w": [4.0, 5.0, 6.0], "grp": ["a", "a", "b"]})

#: A second frame carrying plain ``x``/``y`` columns, for the layered and
#: polar cases below, which need two independently-bindable axes.
_XY_DF = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})


def _missing_channel_message(mark: str, missing: str, other: str) -> str:
    """The one-channel-missing text: names the missing channel and offers
    the single-axis remedy against the other (bound) channel."""
    return (
        f"mark_{mark} needs both positional channels, but nothing binds '{missing}'. "
        f"Encode '{missing}' on the chart to draw it, or draw from '{other}' alone "
        "with mark_tick() or mark_rule()."
    )


def _both_unbound_message(mark: str, a: str, b: str) -> str:
    """Neither of two needed channels bound: both are named together, with
    the remedy that encodes both rather than offering a single-axis
    escape."""
    return (
        f"mark_{mark} needs both positional channels, but nothing binds '{a}' or '{b}'. "
        f"Encode '{a}' and '{b}' on the chart to draw them, or draw from one axis alone "
        "with mark_tick() or mark_rule()."
    )


#: Neither axis bound: both channels are named, and the remedy drops the
#: single-axis offer since there is no bound axis left to draw from alone.
_BOTH_UNBOUND_MESSAGE = _both_unbound_message("point", "x", "y")


def _single_axis_unbound_message(mark: str, a: str, b: str) -> str:
    """The one-channel-needed text (``mark_tick``/``mark_rule``) with
    neither channel bound: no single-axis remedy to offer, since there is
    no bound axis left to draw from."""
    return f"mark_{mark} needs one positional channel, but nothing binds '{a}' or '{b}'. Encode '{a}' or '{b}' on the chart to draw it."


def _theta_unbound_message(mark: str) -> str:
    """The theta-needed text (``mark_arc``/``mark_text`` under a polar
    coord) with nothing bound: only 'theta' is ever named, never
    'radius'."""
    return f"mark_{mark} needs a 'theta' channel, but nothing binds 'theta'. Encode 'theta' on the chart to draw it."


def _sibling_bound_message(mark: str, need_clause: str, channel: str, *, layer: int = 0) -> str:
    """A layer that cannot reach ``channel`` while a sibling layer binds it:
    the validator's sibling-bound clause, one unreached channel."""
    return (
        f"layer {layer}: mark_{mark} {need_clause}, but no '{channel}' encoding reaches it "
        f"— a sibling layer binds '{channel}', which resolves the axis but leaves "
        f"mark_{mark} with nothing to plot. Encode '{channel}' on this layer, or on the "
        "chart, to draw it."
    )


def _double_sibling_bound_message(
    mark: str, need_clause: str, remedy_join: str, *, layer: int = 0
) -> str:
    """A layer that cannot reach EITHER positional channel while a sibling
    layer binds both: the validator's sibling-bound clause, two unreached
    channels."""
    return (
        f"layer {layer}: mark_{mark} {need_clause}, but no 'x' or 'y' encoding reaches it "
        f"— a sibling layer binds 'x' and 'y', which resolves the axes but leaves "
        f"mark_{mark} with nothing to plot. Encode 'x' {remedy_join} 'y' on this layer, or "
        "on the chart, to draw it."
    )


def _unbound_point() -> fm.Chart:
    return fm.Chart(_DF).mark_point().encode(x="v")


def _bound_point() -> fm.Chart:
    return fm.Chart(_DF).mark_point().encode(x="v", y="w")


#: Named two-axis member charts for the layered/sibling cases below: ``Px``
#: binds x only, ``Py`` binds y only, ``P`` binds neither, ``Lxy`` is a line
#: bound to both, ``Ty`` is a tick bound to y only, ``T`` is a tick bound to
#: neither, ``Tx`` is a tick bound to x only.
_MEMBER_BUILDERS = {
    "Px": lambda: fm.Chart(_XY_DF).mark_point().encode(x="x"),
    "Py": lambda: fm.Chart(_XY_DF).mark_point().encode(y="y"),
    "P": lambda: fm.Chart(_XY_DF).mark_point(),
    "Lxy": lambda: fm.Chart(_XY_DF).mark_line().encode(x="x", y="y"),
    "Ty": lambda: fm.Chart(_XY_DF).mark_tick().encode(y="y"),
    "T": lambda: fm.Chart(_XY_DF).mark_tick(),
    "Tx": lambda: fm.Chart(_XY_DF).mark_tick().encode(x="x"),
}


def _member(spec: str) -> fm.Chart:
    return _MEMBER_BUILDERS[spec]()


@pytest.mark.parametrize("mark", ["line", "area", "bar", "text", "rect"])
def test_a_mark_with_one_axis_bound_is_refused_in_its_own_voice(mark: str) -> None:
    """Every mark that draws both axes, bound on x only, is refused by its
    own ``mark_<mark>`` name, not a generic or a differently-voiced one."""
    chart = getattr(fm.Chart(_DF), f"mark_{mark}")().encode(x="v")

    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()

    assert str(excinfo.value) == _missing_channel_message(mark, "y", "x")


def test_a_mark_with_only_y_bound_names_the_missing_x_and_its_own_remedy() -> None:
    """The symmetric direction: a mark bound on y alone names 'x' as missing
    and offers 'y' as the single-axis remedy, not the other way around."""
    chart = fm.Chart(_DF).mark_point().encode(y="v")

    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()

    assert str(excinfo.value) == _missing_channel_message("point", "x", "y")


def test_a_mark_with_no_positional_channel_bound_names_both_unbound_channels() -> None:
    """Neither axis bound: both channels are named together, with the remedy
    that encodes both rather than offering a single-axis escape."""
    chart = fm.Chart(_DF).mark_point()

    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()

    assert str(excinfo.value) == _BOTH_UNBOUND_MESSAGE


@pytest.mark.parametrize(
    "composite, primitive, expected_mark",
    [
        pytest.param(
            lambda c: c.mark_histogram(),
            lambda c: c.mark_point(),
            "point",
            id="histogram-then-point",
        ),
        pytest.param(
            lambda c: c.mark_density(), lambda c: c.mark_point(), "point", id="density-then-point"
        ),
        pytest.param(lambda c: c.mark_qq(), lambda c: c.mark_point(), "point", id="qq-then-point"),
        pytest.param(
            lambda c: c.mark_histogram(),
            lambda c: c.mark_line(stroke_width=4),
            "line",
            id="histogram-then-line",
        ),
    ],
)
def test_a_primitive_mark_after_a_composite_without_y_refuses_in_its_own_voice(
    composite, primitive, expected_mark
) -> None:
    """A primitive mark call replaces a pending statistic (``mark_histogram``,
    ``mark_density``, ``mark_qq``), the same replacement rule a figure's
    post-hoc mark call already follows. The replaced chart draws only what
    the call itself bound, so a missing y refuses in the REPLACING call's own
    voice, never the statistic's."""
    pending = composite(fm.Chart(_DF).encode(x="v"))
    chained = primitive(pending)

    with pytest.raises(ValueError) as excinfo:
        chained.to_svg()

    assert str(excinfo.value) == _missing_channel_message(expected_mark, "y", "x")


def test_an_hconcat_leaf_with_one_axis_bound_is_refused_in_the_leaf_envelope() -> None:
    """A composite leaf that is its own flat chart speaks in its own voice,
    wrapped in the composite render's leaf envelope; it carries no layer
    prefix, since the leaf itself has no layers of its own."""
    with pytest.raises(ValueError) as excinfo:
        fm.hconcat(_unbound_point(), _bound_point()).to_svg()

    assert str(excinfo.value) == (
        "failed to render composite leaf leaf #0: " + _missing_channel_message("point", "y", "x")
    )


@pytest.mark.parametrize("members", ["Px-Lxy", "P-Lxy", "Py-Lxy", "Px-Ty"])
def test_a_layerchart_member_whose_channel_a_sibling_binds_is_refused_as_plus_refuses(
    members: str,
) -> None:
    """A static ``LayerChart`` member whose drawn channel only a sibling
    member binds is refused with the exact text ``+`` gives for the same
    pair, wrapped in the composite leaf envelope -- never the false
    "nothing binds" text a leaf validated alone (against only its own
    bindings) would give, even though the sibling plainly binds it. Every
    door that renders through the merged path -- ``+``'s own
    ``.interactive()``, ``LayerChart``'s own ``.interactive()``, and
    ``LayerChart(..., resolve={"y": "independent"})`` -- gives that exact
    body with no leaf envelope; only the static (shared-y) ``LayerChart``/
    ``fm.layer`` route wraps it."""
    a_id, b_id = members.split("-")

    with pytest.raises(ValueError) as plus_err:
        (_member(a_id) + _member(b_id)).to_svg()
    body = str(plus_err.value)

    with pytest.raises(ValueError) as plus_interactive_err:
        (_member(a_id) + _member(b_id)).interactive().to_html()
    assert str(plus_interactive_err.value) == body

    with pytest.raises(ValueError) as layerchart_interactive_err:
        fm.LayerChart(_member(a_id), _member(b_id)).interactive().to_html()
    assert str(layerchart_interactive_err.value) == body

    with pytest.raises(ValueError) as independent_y_err:
        fm.LayerChart(_member(a_id), _member(b_id), resolve={"y": "independent"}).to_svg()
    assert str(independent_y_err.value) == body

    with pytest.raises(ValueError) as static_err:
        fm.LayerChart(_member(a_id), _member(b_id)).to_svg()
    assert str(static_err.value) == "failed to render composite leaf leaf #0: " + body


def test_a_layerchart_member_whose_channel_no_member_binds_names_it_unbound() -> None:
    """When no member binds a channel at all, the static route's answer is
    the plain "nothing binds" text `+` gives for the same pair -- the
    sibling-bound clause only applies to a channel some member DOES
    bind."""
    with pytest.raises(ValueError) as plus_err:
        (_member("Px") + _member("Px")).to_svg()
    body = str(plus_err.value)

    with pytest.raises(ValueError) as static_err:
        fm.LayerChart(_member("Px"), _member("Px")).to_svg()
    assert str(static_err.value) == "failed to render composite leaf leaf #0: " + body


#: Expected static-route refusal body for each later-member case below, as
#: ``layer 1:`` (member position 1) -- built from the same sibling-bound
#: clauses the earlier-member cases use, since it is the same validator
#: partition, just at a different member position. ``None`` marks the one
#: control row that renders instead of refusing.
_LATER_MEMBER_MESSAGES = {
    "Lxy-Px": _sibling_bound_message("point", "needs both positional channels", "y", layer=1),
    "Lxy-P": _double_sibling_bound_message(
        "point", "needs both positional channels", "and", layer=1
    ),
    "Lxy-T": _double_sibling_bound_message("tick", "needs one positional channel", "or", layer=1),
    "Tx-Lxy": None,
}


@pytest.mark.parametrize("members", sorted(_LATER_MEMBER_MESSAGES))
def test_a_later_layerchart_member_reads_only_its_own_bindings(members: str) -> None:
    """A static ``LayerChart`` member reads only its own bindings, not an
    earlier member's: unlike ``+``, which lets a later operand inherit an
    earlier one's binding (``lxy() + px()`` for this exact pair renders),
    ``LayerChart(Lxy, Px/P/T)`` refuses the later member with the
    sibling-bound text, at that member's own leaf/layer index, on the one
    ordering where ``+`` and ``.interactive()`` render instead.
    ``LayerChart(Tx, Lxy)`` renders on its own terms (the tick reaches its
    own x), since neither member needs the other's binding."""
    a_id, b_id = members.split("-")
    expected = _LATER_MEMBER_MESSAGES[members]

    if expected is None:
        svg = fm.LayerChart(_member(a_id), _member(b_id)).to_svg()
        ticks = svg_elements(mark_layer(svg), "line", lambda a: True)
        assert len(ticks) == len(_XY_DF)
        return

    with pytest.raises(ValueError) as excinfo:
        fm.LayerChart(_member(a_id), _member(b_id)).to_svg()
    assert str(excinfo.value) == "failed to render composite leaf leaf #1: " + expected


def test_a_faceted_chart_with_one_axis_bound_is_refused() -> None:
    """Faceting does not change which channel the refusal names: every panel
    shares the same unbound binding, so the refusal is the chart's own."""
    chart = fm.Chart(_DF).mark_point().encode(x="v").facet(col="grp")

    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()

    assert str(excinfo.value) == _missing_channel_message("point", "y", "x")


def test_an_interactive_chart_with_one_axis_bound_is_refused() -> None:
    """The interactive export resolves the same scales the static render
    does, so it is refused the same way, not left to fail raw in the
    browser-facing bundle. ``.interactive()`` itself renders the scene
    eagerly, so the refusal is raised there, before ``.to_html()`` runs."""
    chart = fm.Chart(_DF).mark_point().encode(x="v")

    with pytest.raises(ValueError) as excinfo:
        chart.interactive().to_html()

    assert str(excinfo.value) == _missing_channel_message("point", "y", "x")


@pytest.mark.parametrize("channel", ["x", "y"])
def test_a_one_axis_tick_draws_every_row_exactly_as_the_rug_figure_does(channel: str) -> None:
    """``mark_tick`` needs only one channel, and the chart it draws from
    that binding alone is byte-identical to the rug ``displot(kind="rug")``
    already draws from the same binding -- not merely "some svg came
    back," which a blank or misplaced render would satisfy too."""
    chart_svg = fm.Chart(_DF).mark_tick().encode(**{channel: "v"}).to_svg()
    rug_svg = fm.displot(_DF, kind="rug", **{channel: "v"}).to_svg()

    assert chart_svg == rug_svg
    ticks = svg_elements(mark_layer(chart_svg), "line", lambda a: True)
    assert len(ticks) == len(_DF)


@pytest.mark.parametrize("channel", ["x", "y"])
def test_a_one_axis_rule_draws_a_full_span_per_row(channel: str) -> None:
    """``mark_rule`` needs only one channel; the rule it draws from that
    channel alone spans the panel's full extent on the OTHER axis, one per
    row -- a blank frame, a squashed rule, or a dropped row fails this
    census the same way a byte mismatch would for the tick."""
    svg = fm.Chart(_DF).mark_rule().encode(**{channel: "v"}).to_svg()
    px, py, pw, ph = plot_rect(svg)
    rules = svg_elements(mark_layer(svg), "line", lambda a: True)

    assert len(rules) == len(_DF)
    if channel == "x":
        positions = {float(r["x1"]) for r in rules}
        assert len(positions) == len(_DF), "each row's rule must sit at a distinct x"
        for r in rules:
            assert float(r["x1"]) == float(r["x2"])
            assert float(r["y1"]) == pytest.approx(py)
            assert float(r["y2"]) == pytest.approx(py + ph)
    else:
        positions = {float(r["y1"]) for r in rules}
        assert len(positions) == len(_DF), "each row's rule must sit at a distinct y"
        for r in rules:
            assert float(r["y1"]) == float(r["y2"])
            assert float(r["x1"]) == pytest.approx(px)
            assert float(r["x2"]) == pytest.approx(px + pw)


@pytest.mark.parametrize("mark", ["tick", "rule"])
def test_a_single_axis_mark_with_nothing_bound_names_the_one_channel_it_needs(mark: str) -> None:
    """A single-axis mark with neither channel bound needs just one of
    them, not both: the refusal says so, and its remedy encodes either
    channel, never the call already made (a tick or rule has no other
    mark to switch to) -- the same text whether rendered statically,
    through ``.interactive()``, or under ``CoordFlip()`` (naming stays
    'x'/'y' here, since nothing is bound for the flip to un-flip)."""
    expected = _single_axis_unbound_message(mark, "x", "y")

    with pytest.raises(ValueError) as excinfo:
        getattr(fm.Chart(_DF), f"mark_{mark}")().to_svg()
    assert str(excinfo.value) == expected

    with pytest.raises(ValueError) as excinfo:
        getattr(fm.Chart(_DF), f"mark_{mark}")().interactive().to_html()
    assert str(excinfo.value) == expected

    with pytest.raises(ValueError) as excinfo:
        getattr(fm.Chart(_DF), f"mark_{mark}")().coord(fm.CoordFlip()).to_svg()
    assert str(excinfo.value) == expected


_POLAR_DF = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


@pytest.mark.parametrize(
    "mark,encoding",
    [
        pytest.param("arc", {"radius": "x"}, id="arc-radius"),
        pytest.param("arc", {"color": "x"}, id="arc-color"),
        pytest.param("text", {"radius": "x", "text": "x"}, id="text-radius"),
    ],
)
@pytest.mark.parametrize("theta_axis", ["x", "y"], ids=["theta-x", "theta-y"])
def test_a_polar_mark_without_theta_is_refused_naming_theta(mark, encoding, theta_axis) -> None:
    """Under a polar coord, ``mark_arc`` (which paints only there) and
    ``mark_text`` (which also paints under a Cartesian coord, but needs
    ``theta`` instead of 'x'/'y' here) both need a ``theta`` channel, never
    a Cartesian 'x'/'y' pair: with ``theta`` unbound, the refusal names
    'theta', not 'x', and its remedy never offers ``mark_tick()``/
    ``mark_rule()`` -- neither can draw a pie. Unaffected by which
    Cartesian axis ``CoordPolar`` maps to the angle, since neither mark
    binds ``x``/``y`` here at all."""
    chart = (
        getattr(fm.Chart(_POLAR_DF), f"mark_{mark}")()
        .encode(**encoding)
        .coord(fm.CoordPolar(theta=theta_axis))
    )

    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()

    assert str(excinfo.value) == _theta_unbound_message(mark)


def test_a_polar_point_names_the_polar_channel_it_lacks() -> None:
    """``mark_point`` needs BOTH polar channels (unlike ``mark_arc``/
    ``mark_text``, which need only ``theta``): bound on ``theta`` alone, it
    names ``'radius'`` missing with the theta-alone remedy; with nothing
    bound, it names both, ``'theta'`` first; a polar ``mark_tick`` with
    nothing bound needs only one of the two."""
    theta_only = fm.Chart(_POLAR_DF).mark_point().encode(theta="x").coord(fm.CoordPolar())
    with pytest.raises(ValueError) as excinfo:
        theta_only.to_svg()
    assert str(excinfo.value) == _missing_channel_message("point", "radius", "theta")

    nothing_bound = fm.Chart(_POLAR_DF).mark_point().coord(fm.CoordPolar())
    with pytest.raises(ValueError) as excinfo:
        nothing_bound.to_svg()
    assert str(excinfo.value) == _both_unbound_message("point", "theta", "radius")

    polar_tick = fm.Chart(_POLAR_DF).mark_tick().coord(fm.CoordPolar())
    with pytest.raises(ValueError) as excinfo:
        polar_tick.to_svg()
    assert str(excinfo.value) == _single_axis_unbound_message("tick", "theta", "radius")


def test_a_missing_channel_is_refused_regardless_of_row_count() -> None:
    """The refusal is about bindings, not rows: an empty or single-row frame
    refuses exactly the same way a full one does."""
    empty = pl.DataFrame({"v": pl.Series([], dtype=pl.Float64)})
    one_row = pl.DataFrame({"v": [1.0]})

    for frame in (empty, one_row):
        with pytest.raises(ValueError) as excinfo:
            fm.Chart(frame).mark_point().encode(x="v").to_svg()
        assert str(excinfo.value) == _missing_channel_message("point", "y", "x")


def test_an_explicitly_none_channel_is_treated_as_unset() -> None:
    """``encode(y=None)`` leaves y exactly as unset as never writing it."""
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_DF).mark_point().encode(x="v", y=None).to_svg()

    assert str(excinfo.value) == _missing_channel_message("point", "y", "x")


def test_an_explicitly_none_x_still_names_both_channels() -> None:
    """``encode(x=None)`` with nothing else bound names both channels, the
    same as never calling ``.encode()`` at all."""
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_DF).mark_point().encode(x=None).to_svg()

    assert str(excinfo.value) == _BOTH_UNBOUND_MESSAGE


def test_a_non_string_non_channel_x_value_keeps_its_own_type_error() -> None:
    """``encode()``'s own type check runs before any of this reaches Rust,
    and stays the existing ``TypeError`` naming the value's type -- an ``int``
    is never read as "unset"."""
    with pytest.raises(TypeError) as excinfo:
        fm.Chart(_DF).mark_point().encode(x=5)

    assert str(excinfo.value) == (
        "encode(x=...) expects str, X instance, or Repeat placeholder; got int"
    )


def test_a_layer_missing_both_channels_a_sibling_binds_names_both() -> None:
    """A layer missing BOTH positional channels, beside a sibling that
    binds both, names both together in one sentence -- not just the first
    one a narrower validator would have found and stopped at."""
    with pytest.raises(ValueError) as excinfo:
        (_member("P") + _member("Lxy")).to_svg()

    assert str(excinfo.value) == _double_sibling_bound_message(
        "point", "needs both positional channels", "and"
    )


def test_a_layer_missing_an_unbound_and_a_sibling_bound_channel_names_both() -> None:
    """A layer missing one channel nothing binds at all, and one channel
    only a sibling binds, names both in the same sentence, each clause for
    its own reason -- the mixed case the validator's two-clause partition
    exists for."""
    with pytest.raises(ValueError) as excinfo:
        (_member("P") + _member("Ty")).to_svg()

    assert str(excinfo.value) == (
        "layer 0: mark_point needs both positional channels, but nothing binds 'x', "
        "and no 'y' encoding reaches it — a sibling layer binds 'y', which resolves "
        "the axis but leaves mark_point with nothing to plot. Encode 'x' and 'y' on this "
        "layer, or on the chart, to draw it."
    )


def test_a_layered_single_axis_mark_reaching_no_channel_is_refused() -> None:
    """A single-axis mark (``mark_tick``) that reaches neither channel,
    beside a sibling that binds both, is refused rather than silently
    drawing nothing: the validator asks every painting layer's own reach,
    not merely whether a tick could render from one axis in the abstract,
    so a tick layer shaped like this never renders zero ticks with no
    warning."""
    with pytest.raises(ValueError) as excinfo:
        (_member("T") + _member("Lxy")).to_svg()

    assert str(excinfo.value) == _double_sibling_bound_message(
        "tick", "needs one positional channel", "or"
    )


def _polar_arc_radius_only() -> fm.Chart:
    return fm.Chart(_POLAR_DF).mark_arc().encode(radius="x").coord(fm.CoordPolar())


def _polar_text_theta_only() -> fm.Chart:
    return fm.Chart(_POLAR_DF).mark_text().encode(theta="x", text="x").coord(fm.CoordPolar())


def test_a_polar_layer_without_theta_names_the_sibling_theta() -> None:
    """A polar ``mark_arc`` layer that cannot reach ``theta``, beside a
    sibling that binds it, names 'theta' -- not 'x' -- and the static
    ``LayerChart`` route is refused with the exact same text, wrapped in
    the leaf envelope, that the `+` route gives for the same pair."""
    with pytest.raises(ValueError) as plus_err:
        (_polar_arc_radius_only() + _polar_text_theta_only()).to_svg()
    body = str(plus_err.value)
    assert body == _sibling_bound_message("arc", "needs a 'theta' channel", "theta")

    with pytest.raises(ValueError) as static_err:
        fm.LayerChart(_polar_arc_radius_only(), _polar_text_theta_only()).to_svg()
    assert str(static_err.value) == "failed to render composite leaf leaf #0: " + body


def test_a_flipped_member_that_inherits_a_channel_from_its_layer_renders_like_plus() -> None:
    """A member with a chart-level ``CoordFlip`` whose own layer supplies
    the channel the chart binds on the other axis draws the same marks, on
    every composite door, that ``+`` draws for the same pair: the static
    route reads the member's own authored bindings, not a chart encoding an
    earlier pass already flipped. ``+`` returns a flat ``Chart`` while
    ``fm.layer``/``LayerChart`` render through the composite overlay tree,
    so their SVG bytes differ (the composite leaf's own clip/def chrome);
    what matches is the drawn marks themselves, read from ``mark_layer``."""
    member = (
        fm.Chart(_DF)
        .mark_tick()
        .encode(x="v")
        .layer(fm.Layer(mark="line", encoding={"y": "w"}))
        .coord(fm.CoordFlip())
    )
    sibling = fm.Chart(_DF).mark_point().encode(x="v", y="w")

    def marks(svg: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str]], list[str]]:
        layer = mark_layer(svg)
        circles = [(c["cx"], c["cy"]) for c in svg_elements(layer, "circle", lambda a: True)]
        lines = [
            (l["x1"], l["y1"], l["x2"], l["y2"])
            for l in svg_elements(layer, "line", lambda a: True)
        ]
        polys = [p["points"] for p in svg_elements(layer, "polyline", lambda a: "points" in a)]
        return circles, lines, polys

    plus_circles, plus_lines, plus_polys = marks((member + sibling).to_svg())
    assert plus_circles and plus_lines and plus_polys, "the control render must draw something"

    layer_svg = fm.layer(member, sibling).to_svg()
    lc_svg = fm.LayerChart(member, sibling).to_svg()
    assert layer_svg == lc_svg

    for svg in (layer_svg, lc_svg):
        circles, lines, polys = marks(svg)
        assert circles == plus_circles
        assert lines == plus_lines
        assert polys == plus_polys


def test_a_color_type_mismatch_keeps_its_own_refusal() -> None:
    """Color's own ``EncodingTypeMismatch`` is a different refusal
    entirely, untouched by the positional-channel refusal above: a
    declared quantitative color on a string column still raises its own
    typed, pinned text, never folded into it."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "g": ["a", "a", "b"]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color=fm.Color("g", type="quantitative"))

    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()

    assert str(excinfo.value) == (
        "encoding 'color' expected numeric column for quantitative/temporal type, got Utf8"
    )
