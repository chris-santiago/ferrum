"""`#172`(1) -- a layer carries its own coordinate system.

A ``LayerChart`` member declares a coordinate system on its own ``Chart``
(``fm.Chart(df).mark_arc()...coord(fm.CoordPolar())``). The static shared-y
route keeps every member a separate composite leaf, so it has always honored
that declaration, except that a member whose own coord exchanges its channels
and lands a column on an axis that cannot hold it refuses there too, in the
same voice as the merged-flat route. The **merged-flat** route -- one single-panel ``Chart``
assembled from ``spec.layers`` -- dropped it: a secondary ``CoordFlip``
rendered unflipped (silently, geometry identical to a chart with no flip at
all), and a secondary ``CoordPolar`` refused with a message prescribing the
very ``.coord(...)`` call the caller had already written.

Three public routes reach the merged-flat path and each is a row here,
because they are ONE code path (see `CLAUDE.md`'s composite-rendering
section): ``chart + chart`` static, ``LayerChart(resolve={"y":
"independent"})`` static, and ``LayerChart(...).interactive()``. A fix that
moved one and not the others would mean a caller was patched rather than the
path.

Each of those three rows exists twice: once with both positional axes
CONTINUOUS (the original matrix) and once with an ORDINAL primary axis (the
wave-5 close review's F1 -- a band ladder is built by a different mechanism
than a numeric extent, and only the numeric one had been taught to read each
layer's own effective coord, so all three routes silently dropped the member's
marks on a categorical axis while the continuous rows passed).

Every geometric assertion here is derived from the render itself rather than
from a hand-computed pixel: the control layer draws the same data at two
known domain endpoints, which pins the data->pixel map exactly (the scale is
linear and both endpoints are drawn), and the flipped layer's expected
vertices are then computed through that map. A broken (unflipped) render puts
those vertices on the opposite side of the panel, so the assertion cannot
pass for the wrong reason.
"""

from __future__ import annotations

import datetime as dt
import json
import warnings

import polars as pl
import pytest

import ferrum as fm
from tests._shared_axis_refusal import composite_envelope, coord_exchange_refusal
from tests._svg_rects import mark_layer, plot_rect, polylines, svg_elements

# Control layer: one point at each end of a 0..100 square. Both endpoints are
# drawn, so the two circles pin the linear data->pixel map on both axes.
CONTROL = pl.DataFrame({"x": [0.0, 100.0], "y": [0.0, 100.0]})
# Secondary layer, deliberately off-diagonal and confined to one corner:
# unflipped it draws at data x 0..25 / y 75..100 (upper LEFT), flipped it
# draws at data x 75..100 / y 0..25 (lower RIGHT). No tolerance-sized
# difference to argue about.
SECONDARY = pl.DataFrame({"bx": [0.0, 25.0], "by": [75.0, 100.0]})


def _control_chart():
    return fm.Chart(CONTROL).mark_point().encode(x="x", y="y")


def _secondary_line(coord=None):
    chart = fm.Chart(SECONDARY).mark_line().encode(x="bx", y="by")
    return chart if coord is None else chart.coord(coord)


def _circles(svg: str) -> list[tuple[float, float]]:
    """Return every drawn point's centre, in document order.

    Scoped to `mark_layer` (the plot-area clip groups) so a legend's own
    circular swatches -- which a `color=`-bearing member contributes, at fixed
    positions in the legend band -- cannot be mistaken for marks.
    """
    return [
        (float(a["cx"]), float(a["cy"]))
        for a in svg_elements(mark_layer(svg), "circle", lambda a: "cx" in a and "cy" in a)
    ]


def _data_to_pixel(control_points: list[tuple[float, float]]):
    """Return `(x_of, y_of)`: the exact data->pixel map the control layer drew.

    The control data is `(0, 0)` and `(100, 100)` on a linear scale, so two
    drawn points determine the whole map -- no assumption about padding,
    nicing, or the plot rect enters the expected values below.
    """
    (x0, y0), (x1, y1) = sorted(control_points)
    return (
        lambda v: x0 + (v / 100.0) * (x1 - x0),
        lambda v: y0 + (v / 100.0) * (y1 - y0),
    )


def _assert_transposed(line_points, control_points):
    """Assert the secondary line drew its data with x and y exchanged.

    Both axes resolve the same union domain (each spans `0..100`: the control
    layer's x with the secondary's y, and the control's y with the secondary's
    x), so "flipped" has an exact expected pixel for every vertex.
    """
    x_of, y_of = _data_to_pixel(control_points)
    expected = [(x_of(75.0), y_of(0.0)), (x_of(100.0), y_of(25.0))]
    assert len(line_points) == len(expected), (
        f"expected {len(expected)} line vertices, got {line_points}"
    )
    for got, want in zip(sorted(line_points), sorted(expected)):
        assert got[0] == pytest.approx(want[0], abs=0.5), (
            f"flipped layer's x should read off the 'by' field ({want[0]:.3f}); "
            f"got {got[0]:.3f} -- an unflipped render puts it at {x_of(0.0):.3f}/"
            f"{x_of(25.0):.3f}"
        )
        assert got[1] == pytest.approx(want[1], abs=0.5), (
            f"flipped layer's y should read off the 'bx' field ({want[1]:.3f}); got {got[1]:.3f}"
        )


# ---------------------------------------------------------------------------
# CoordFlip on a secondary layer -- the silent half of the defect.
# ---------------------------------------------------------------------------


def test_flat_merge_honors_a_secondary_layers_flip():
    """`chart + chart.coord(CoordFlip())` renders the second layer flipped."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (_control_chart() + _secondary_line(fm.CoordFlip())).to_svg()

    lines = polylines(svg)
    assert len(lines) == 1, f"expected one line polyline, got {len(lines)}"
    _assert_transposed(lines[0][1], _circles(svg))


def test_interactive_layer_honors_a_secondary_layers_flip():
    """The interactive merged-flat scene places the flipped layer flipped."""
    layered = fm.layer(_control_chart(), _secondary_line(fm.CoordFlip()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene_json, _packed = layered._render_interactive()
    panels = json.loads(scene_json)["panels"]
    assert len(panels) == 1, "LayerChart interactive is always one scene panel"
    by_kind = {batch["kind"]: batch["nodes"] for batch in panels[0]["marks"]}
    control = [(n["cx"], n["cy"]) for n in by_kind["point"]]
    assert len(by_kind["line"]) == 1, "one polyline node for the line layer"
    _assert_transposed(
        [tuple(p) for p in by_kind["line"][0]["points"]],
        control,
    )


def test_independent_y_static_honors_a_secondary_layers_flip():
    """`resolve={"y": "independent"}` static -- the same merged-flat path.

    The flipped layer's y comes from its OWN slot here, so only its x is
    comparable with the control layer's map; that is the coordinate the flip
    moves from the left quarter of the panel to the right, and it is asserted
    exactly. Its own y slot spans the `bx` field (0..25) over the full plot
    height, which is the second half of the same statement: a slot resolved
    from the post-flip channel, not the authored one.
    """
    layered = fm.layer(
        _control_chart(), _secondary_line(fm.CoordFlip()), resolve={"y": "independent"}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = layered.to_svg()

    x_of, _y_of = _data_to_pixel(_circles(svg))
    lines = polylines(svg)
    assert len(lines) == 1, f"expected one line polyline, got {len(lines)}"
    xs = sorted(p[0] for p in lines[0][1])
    assert xs[0] == pytest.approx(x_of(75.0), abs=0.5), (
        f"flipped layer's first x should read off 'by' (={x_of(75.0):.3f}); got {xs[0]:.3f}"
    )
    assert xs[-1] == pytest.approx(x_of(100.0), abs=0.5), (
        f"flipped layer's last x should read off 'by' (={x_of(100.0):.3f}); got {xs[-1]:.3f}"
    )
    _px, py, _pw, ph = plot_rect(svg)
    ys = [p[1] for p in lines[0][1]]
    assert max(ys) - min(ys) > 0.9 * ph, (
        f"the flipped layer's own y slot spans its 'bx' domain over the full plot "
        f"height ({ph:.3f}); got {max(ys) - min(ys):.3f}"
    )


# ---------------------------------------------------------------------------
# The same three routes on an ORDINAL positional axis (wave-5 close review F1).
#
# Every row above resolves both axes CONTINUOUSLY, which is exactly why a green
# suite missed this: the flipped member's categories reach a band ladder by a
# different mechanism than its values reach a numeric extent
# (`scale_resolve::domain::positional_category_union`, not `numeric_domain_union`),
# and only the numeric one had been taught to read each layer's OWN effective
# coord. Measured on the ordinal shape before the fix, on all three routes: the
# member drew ZERO nodes, with no warning and no error, while the x ladder
# carried three spurious bands enumerated from its VALUE column
# (`['a','b','c','40','50','60']`).
#
# The ladder itself is asserted in `test_layer_ordinal_domain_union.py`; these
# rows assert what the routes owe -- that the member draws at all.
# ---------------------------------------------------------------------------

#: Categorical primary axis, and a secondary member on its own frame so its
#: columns are renamed by the merge and its contribution is observable.
ORDINAL_CONTROL = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
#: Authored `x=`value / `y=`category, so post-flip its CATEGORIES occupy the
#: rendered x axis -- the half of the shape a shared band ladder can serve.
ORDINAL_SECONDARY = pl.DataFrame({"cat2": ["x", "y", "z"], "val2": [40.0, 50.0, 60.0]})


def _ordinal_control():
    return fm.Chart(ORDINAL_CONTROL).mark_bar().encode(x=fm.X("cat", type_="N"), y="val")


def _ordinal_secondary_flipped():
    return (
        fm.Chart(ORDINAL_SECONDARY)
        .mark_point()
        .encode(x="val2", y=fm.Y("cat2", type_="N"))
        .coord(fm.CoordFlip())
    )


def test_flat_merge_honors_a_secondary_layers_flip_on_an_ordinal_axis():
    """`chart + chart.coord(CoordFlip())` with a categorical primary axis."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (_ordinal_control() + _ordinal_secondary_flipped()).to_svg()

    assert len(_circles(svg)) == 3, (
        "the flipped member draws one mark per row; zero is the silent drop this row exists for"
    )


def test_interactive_layer_honors_a_secondary_layers_flip_on_an_ordinal_axis():
    """The interactive merged-flat scene -- the route the drop was reported on."""
    layered = fm.layer(_ordinal_control(), _ordinal_secondary_flipped())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene_json, _packed = layered._render_interactive()
    panels = json.loads(scene_json)["panels"]
    assert len(panels) == 1, "LayerChart interactive is always one scene panel"
    by_kind = {batch["kind"]: batch["nodes"] for batch in panels[0]["marks"]}
    assert len(by_kind["bar"]) == 3, "premise: the primary layer draws its three bars"
    assert len(by_kind["point"]) == 3, (
        "the flipped member must emit three nodes; pre-fix it emitted zero, silently"
    )


def test_independent_y_static_honors_a_secondary_layers_flip_on_an_ordinal_axis():
    """`resolve={"y": "independent"}` static -- the third route, same path."""
    layered = fm.layer(
        _ordinal_control(), _ordinal_secondary_flipped(), resolve={"y": "independent"}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = layered.to_svg()

    assert len(_circles(svg)) == 3


def test_the_continuous_row_is_the_control_that_always_passed():
    """Stated as its own row so the ordinal rows above cannot be read as new coverage
    of something the continuous ones already covered.

    Same composition shape with both axes quantitative: this is what the three
    rows at the top of this module assert geometrically, and it passed before the
    fix. Its presence here is the negative control for the claim that the ordinal
    path, specifically, was the gap.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (_control_chart() + _secondary_line(fm.CoordFlip())).to_svg()
    assert len(polylines(svg)) == 1


# ---------------------------------------------------------------------------
# CoordPolar on a secondary layer -- the refusing half.
# ---------------------------------------------------------------------------

WEDGES = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0], "second": [3.0, 2.0, 1.0]})


def _bar_chart():
    return fm.Chart(WEDGES).mark_bar().encode(x="cat", y="val")


def _arc_chart():
    return (
        fm.Chart(WEDGES).mark_arc().encode(theta="val", color="cat").coord(fm.CoordPolar(theta="x"))
    )


def _wedge_paths(svg: str) -> list[str]:
    """Return the `d` of every closed wedge path (`... L <centre> Z`)."""
    return [
        a["d"]
        for a in svg_elements(svg, "path", lambda a: a.get("d", "").endswith("Z") and "A" in a["d"])
    ]


def test_flat_merge_draws_a_polar_layers_wedges_at_the_panel_centre():
    """A `CoordPolar` layer projects its own marks; the panel keeps its axes.

    Pre-fix this raised `mark_arc requires a Polar coord` -- naming the fix
    the caller had already applied to that very layer.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (_bar_chart() + _arc_chart()).to_svg()

    x, y, w, h = plot_rect(svg)
    centre = (x + w / 2.0, y + h / 2.0)
    wedges = _wedge_paths(svg)
    assert len(wedges) == 3, f"expected one wedge per category, got {len(wedges)}: {wedges}"
    for d in wedges:
        tail = d.rsplit("L", 1)[1].replace("Z", "").split()
        got = (float(tail[0]), float(tail[1]))
        assert got[0] == pytest.approx(centre[0], abs=0.5), (
            f"wedge must close on the panel centre x ({centre[0]:.3f}); got {got[0]:.3f}"
        )
        assert got[1] == pytest.approx(centre[1], abs=0.5), (
            f"wedge must close on the panel centre y ({centre[1]:.3f}); got {got[1]:.3f}"
        )
    # The Cartesian chrome is the chart's own (layer 0's), unchanged: a polar
    # LAYER projects its marks, it does not suppress the panel's axes.
    assert 'transform="rotate(-90' in svg or "val" in svg, "the bar layer's own axes still draw"


def test_interactive_layer_draws_a_polar_layers_wedges():
    """The interactive merged-flat scene carries the polar layer's arc batch."""
    layered = fm.layer(_bar_chart(), _arc_chart())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene_json, _packed = layered._render_interactive()
    panels = json.loads(scene_json)["panels"]
    assert len(panels) == 1
    kinds = [batch["kind"] for batch in panels[0]["marks"]]
    assert "arc" in kinds, f"expected an arc mark batch, got {kinds}"
    arc_nodes = next(b["nodes"] for b in panels[0]["marks"] if b["kind"] == "arc")
    assert len(arc_nodes) == 3, f"one wedge per category, got {len(arc_nodes)}"


def test_independent_y_static_draws_a_polar_layers_wedges():
    """`resolve={"y": "independent"}` static reaches the same merged path."""
    layered = fm.layer(_bar_chart(), _arc_chart(), resolve={"y": "independent"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = layered.to_svg()
    assert len(_wedge_paths(svg)) == 3, "the polar layer draws its wedges here too"


def test_two_polar_members_both_draw_their_wedges():
    """The INHERITED case: both members declare the same polar coord.

    The merge stamps nothing here -- the two coords agree, so the chart-level
    one serves both layers, and only the second member would have carried a
    per-layer coord if they had differed. That makes this the case a
    layer's-own-coord-only remap would miss: the second member's `theta=`
    channel is claimed by the per-layer polar remap, the first's by the same
    remap reading the coord it inherits. Before this fix the layered path had
    no per-layer remap at all, so BOTH members' angular channel was warned
    about and dropped, and the layered chart drew no wedges whatever coord it
    carried.
    """
    second = fm.Chart(WEDGES).mark_arc().encode(theta="second").coord(fm.CoordPolar(theta="x"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (_arc_chart() + second).to_svg()
    assert len(_wedge_paths(svg)) == 6, (
        f"three wedges per polar member, got {len(_wedge_paths(svg))}"
    )


def test_a_cartesian_secondary_layer_under_a_polar_chart_stays_cartesian():
    """The inverse direction: layer 0 polar, layer 1 plain.

    The secondary member declares no coord of its own, so before this fix it
    inherited the chart-level `CoordPolar` and its marks were run through the
    polar projection -- a chart the caller never wrote. "Inherit" and
    "Cartesian" are the same thing at chart level and not here, which is why
    the merge stamps an explicit `CoordCartesian()` on such a layer.

    Discriminated against the projection's own measured output for this data:
    `mark_point().encode(x="val", y="val").coord(CoordPolar())` places the
    three rows at (320.33, 236.0), (320.0, 352.0), (301.8, 20.8) -- clustered
    around the panel centre, x neither monotone nor spanning. A Cartesian
    placement of x=y=1..3 is the panel diagonal, corner to corner.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (_arc_chart() + fm.Chart(WEDGES).mark_point().encode(x="val", y="val")).to_svg()

    circles = _circles(svg)
    assert len(circles) == 3, f"expected three points, got {circles}"
    _px, _py, w, h = plot_rect(svg)
    xs = [c[0] for c in circles]
    ys = [c[1] for c in circles]
    assert xs == sorted(xs), f"Cartesian x must rise with the data; got {xs}"
    assert ys == sorted(ys, reverse=True), f"Cartesian y must fall with the data; got {ys}"
    assert xs[-1] - xs[0] > 0.9 * w, (
        f"a Cartesian x=1..3 spans the panel (>{0.9 * w:.3f}); got {xs[-1] - xs[0]:.3f} "
        f"-- the polar projection spans ~18px around the centre"
    )
    assert ys[0] - ys[-1] > 0.9 * h, (
        f"a Cartesian y=1..3 spans the panel (>{0.9 * h:.3f}); got {ys[0] - ys[-1]:.3f}"
    )


# ---------------------------------------------------------------------------
# What cannot be placed refuses, loudly, naming the coord.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "coord,name",
    [
        (fm.CoordFixed(ratio=1.0), "CoordFixed"),
        (fm.CoordCartesian(xlim=(0.0, 10.0)), "CoordCartesian"),
    ],
)
def test_panel_scoped_secondary_layer_coord_refuses_naming_it(coord, name):
    """A coord declaring a PANEL property cannot ride one layer of a panel.

    An aspect ratio and an axis domain belong to the panel every layer
    shares, so there is no honest way to apply them to one layer of it. The
    merged-flat routes refuse and the message names the coord it could not
    place -- they do not render a chart with the declaration dropped, which is
    what all three did for every per-layer coord before this fix.

    Asserted on all three merged-flat routes, because they are one path.
    `fm.layer(...).to_svg()` is deliberately NOT here: the static shared-y
    route keeps every member a separate composite leaf with its own scales and
    its own coord, so nothing is merged and nothing is dropped -- that route's
    behavior is unchanged by `#172`(1).
    """
    flat = _control_chart() + _secondary_line(coord)
    layered = fm.layer(_control_chart(), _secondary_line(coord))
    independent = fm.layer(_control_chart(), _secondary_line(coord), resolve={"y": "independent"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors = []
        for call in (flat.to_svg, layered._render_interactive, independent.to_svg):
            with pytest.raises(ValueError) as err:
                call()
            errors.append(err)
    for err in errors:
        message = str(err.value)
        assert name in message, f"the refusal must name the coord it dropped; got {message!r}"
        assert "layer 1" in message, f"the refusal must name the layer; got {message!r}"


# ---------------------------------------------------------------------------
# The second way a per-layer coord cannot be placed: the exchange itself is
# unserviceable, because the panel's axes cannot draw the columns it moves onto
# them (wave-5 close remediation). `fm.CoordFlip` declares no panel property, so
# it passes the gate above and used to reach the mark builders, whose swallowing
# column reads dropped the member with no error and no warning -- while the same
# caller mistake spelled `fm.CoordFixed` refused loudly two tests up. One
# question, two answers; now one.
# ---------------------------------------------------------------------------

#: One frame, one categorical x, one quantitative y. Both members bind the SAME
#: columns, which is what makes the exchange unserviceable: the flipped member's
#: y becomes the Utf8 column while the panel's y axis stays quantitative.
SHARED_FRAME = pl.DataFrame({"cat": ["a", "b", "c"], "v": [3.0, 1.0, 2.0]})


def _shared_bar():
    return fm.Chart(SHARED_FRAME).mark_bar().encode(x=fm.X("cat", type_="N"), y="v")


def _shared_point_flipped():
    return (
        fm.Chart(SHARED_FRAME)
        .mark_point()
        .encode(x=fm.X("cat", type_="N"), y="v")
        .coord(fm.CoordFlip())
    )


@pytest.mark.parametrize("route", ["flat", "interactive"])
def test_an_unserviceable_layer_flip_refuses_in_the_panel_propertys_words(route):
    """The refusal a caller cannot distinguish from the panel-property one.

    Asserted on message CONTENT, not merely that something raised: the property
    being fixed is that the two gates speak with one voice, so the shared frame
    ("layer 1: <coord>", "cannot be placed on one of them", the closing
    instruction) is asserted verbatim alongside the diagnosable part (which axis,
    which column, which dtype).

    Static `fm.layer(...).to_svg()` refuses with the same text in the composite
    envelope, and independent y renders it; both are pinned below.
    """
    layered = fm.layer(_shared_bar(), _shared_point_flipped())
    call = (
        (_shared_bar() + _shared_point_flipped()).to_svg
        if route == "flat"
        else (layered._render_interactive)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as err:
            call()

    message = str(err.value)
    # The frame shared verbatim with `test_panel_scoped_secondary_layer_coord_refuses_naming_it`.
    assert message.startswith("layer 1: fm.CoordFlip"), message
    assert "cannot be placed on one of them" in message, message
    assert "Move it to the first member (whose coord IS the panel's), or drop it." in message, (
        message
    )
    # The diagnosable part: which axis could not take which column.
    assert "y axis is continuous" in message, message
    assert "'cat'" in message, message
    assert "Utf8" in message, message


def test_a_flipped_member_renders_only_where_it_gets_its_own_y_and_refuses_on_a_shared_panel():
    """A flipped member's category lands on y; only its own y slot can draw it.

    `resolve={"y": "independent"}` gives the flipped member its own y slot,
    which is categorical, so all three of its marks draw. Static
    `fm.layer(...).to_svg()` shares one panel y axis, continuous from the bar,
    so it refuses with the same coord-exchange text as `+` and `.interactive()`,
    in the composite route's envelope.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        independent = fm.layer(
            _shared_bar(), _shared_point_flipped(), resolve={"y": "independent"}
        ).to_svg()
        with pytest.raises(ValueError) as err:
            fm.layer(_shared_bar(), _shared_point_flipped()).to_svg()

    assert len(_circles(independent)) == 3, "the flipped member must draw its three marks"
    assert str(err.value) == composite_envelope(
        "y", "cat", coord_exchange_refusal(1, "fm.CoordFlip", "cat", "Utf8")
    )


_EXCHANGE_FRAME = pl.DataFrame({"s": ["a", "b"], "v": [1.0, 2.0]})


def _exchange_lead():
    return fm.Chart(_EXCHANGE_FRAME).mark_point().encode(x="s", y="v")


def _exchange_flipped():
    return _exchange_lead().coord(fm.CoordFlip())


def _plus_svg(lead, member):
    return (lead + member).to_svg()


def _layer_svg(lead, member):
    return fm.layer(lead, member).to_svg()


def _layer_interactive(lead, member):
    return fm.layer(lead, member)._render_interactive()


# (render, wrapped): only static `fm.layer` goes through the composite route's envelope.
_EVERY_OUTPUT = pytest.mark.parametrize(
    ("render", "wrapped"),
    [(_plus_svg, False), (_layer_svg, True), (_layer_interactive, False)],
)


def _refusal_text(render, lead, member) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as err:
            render(lead, member)
    return str(err.value)


@_EVERY_OUTPUT
def test_a_flipped_string_column_on_a_continuous_y_refuses_on_every_output(render, wrapped):
    """Layer 1's flip lands its Utf8 `s` on layer 0's continuous y on all three outputs."""
    core = coord_exchange_refusal(1, "fm.CoordFlip", "s", "Utf8")
    text = _refusal_text(render, _exchange_lead(), _exchange_flipped())
    assert text == (composite_envelope("y", "s", core) if wrapped else core)


_MIRROR_FRAME = pl.DataFrame(
    {
        "s": ["a", "b"],
        "v": [1.0, 2.0],
        "t": [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)],
    }
)


def _mirror_point():
    return fm.Chart(_MIRROR_FRAME).mark_point()


@_EVERY_OUTPUT
@pytest.mark.parametrize(
    ("lead_xy", "member_xy", "field", "dtype", "axis_kind"),
    [
        (("s", "v"), ("s", "v"), "s", "Utf8", "continuous"),
        (("v", "s"), ("t", "v"), "t", "Timestamp(Millisecond, None)", "categorical"),
    ],
    ids=["continuous", "categorical"],
)
def test_a_coordless_layer_beside_a_flipped_first_layer_refuses_on_every_output(
    render, wrapped, lead_xy, member_xy, field, dtype, axis_kind
):
    """The mirror: layer 0 flips, layer 1 declares no coord, so it inherits
    `fm.CoordCartesian` and sits in the opposite space from the panel's axes.
    Its x column lands on an x axis that cannot draw it, on either axis kind."""
    lead = _mirror_point().encode(x=lead_xy[0], y=lead_xy[1]).coord(fm.CoordFlip())
    member = _mirror_point().encode(x=member_xy[0], y=member_xy[1])
    core = coord_exchange_refusal(1, "fm.CoordCartesian", field, dtype, "x", axis_kind)
    text = _refusal_text(render, lead, member)
    assert text == (composite_envelope("x", field, core) if wrapped else core)


@pytest.mark.parametrize("render", [_plus_svg, _layer_svg, _layer_interactive])
def test_a_coordless_layer_whose_columns_suit_the_flipped_axes_renders(render):
    """The servable control for the mirror: the member's `v` lands on the panel's
    continuous x and `s` on its categorical y, so every output draws both layers."""
    lead = _mirror_point().encode(x="s", y="v").coord(fm.CoordFlip())
    member = _mirror_point().encode(x="v", y="s")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = render(lead, member)
    assert [str(w.message) for w in caught] == []
    if render is _layer_interactive:
        scene_json, _packed = out
        marks = json.loads(scene_json)["panels"][0]["marks"]
        total_nodes = sum(len(b["nodes"]) for b in marks)
        assert total_nodes == 4, (
            f"both layers must draw their points; got {total_nodes} nodes "
            f"across {len(marks)} mark batches"
        )
    else:
        assert len(_circles(out)) == 4


def test_a_numeric_category_column_makes_the_same_exchange_placeable():
    """The discriminating control, and the measurement that found the mechanism.

    Identical composition over an INTEGER category column: the panel's continuous
    y can read it, so the exchange is placeable and the chart renders. This is why
    the refusal names a column and an axis rather than a coord and an axis kind --
    it is the column that decides.
    """
    frame = pl.DataFrame({"k": [1, 2, 3], "v": [3.0, 1.0, 2.0]})
    bar = fm.Chart(frame).mark_bar().encode(x=fm.X("k", type_="O"), y="v")
    flipped = (
        fm.Chart(frame).mark_point().encode(x=fm.X("k", type_="O"), y="v").coord(fm.CoordFlip())
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (bar + flipped).to_svg()
    assert len(_circles(svg)) == 3


def test_a_plain_cartesian_secondary_layer_coord_renders():
    """Negative control: only PANEL-scoped declarations refuse.

    `CoordCartesian()` with no limits declares nothing the panel owns, so it
    is honored as the no-op it is -- proving the refusal above discriminates
    on the declared property, not on the mere presence of a per-layer coord.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (_control_chart() + _secondary_line(fm.CoordCartesian())).to_svg()
    lines = polylines(svg)
    assert len(lines) == 1, f"the layer still draws; got {len(lines)} polylines"


def test_no_refusal_message_prescribes_a_coord_the_caller_already_set():
    """The retired hint. A refusal may name the requirement, never the fix.

    `mark_arc` with no polar coord anywhere still refuses -- that is a real
    malformed chart -- but the message no longer prescribes
    `.coord(fm.CoordPolar(...))`, which was exactly what the layered caller
    had already written on the layer whose coord was being dropped.
    """
    chart = fm.Chart(WEDGES).mark_arc().encode(x="val", color="cat")
    with pytest.raises(ValueError) as err:
        chart.to_svg()
    message = str(err.value)
    assert "Polar" in message, f"the requirement is still named; got {message!r}"
    assert ".coord(" not in message, (
        f"a refusal must not prescribe a .coord() call the caller may already have "
        f"applied; got {message!r}"
    )


# ---------------------------------------------------------------------------
# Wire contract: additive, skip-if-absent.
# ---------------------------------------------------------------------------

#: A two-layer merged spec's wire JSON, captured from the build BEFORE
#: `Layer.coord` existed (`7e8514df`). Round-tripping this exact payload is
#: what proves the new field is additive: a spec that declares no per-layer
#: coord serializes to the bytes it always did.
PRE_CHANGE_LAYERED_PAYLOAD = (
    '{"data":{"kind":"named","name":"default"},"mark":"point",'
    '"encoding":{"x":{"field":"x"},"y":{"field":"y"}},'
    '"layers":[{"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y"}}},'
    '{"mark":"line","encoding":{"x":{"field":"bx"},"y":{"field":"by"}}}]}'
)


def test_a_spec_with_no_per_layer_coord_round_trips_byte_identically():
    from ferrum._core import ChartSpec

    reparsed = ChartSpec.from_json(PRE_CHANGE_LAYERED_PAYLOAD)
    assert reparsed.to_json() == PRE_CHANGE_LAYERED_PAYLOAD


def test_the_current_build_still_emits_that_payload_for_the_same_charts():
    """The captured bytes above are this build's bytes, not a stale snapshot."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = fm.layer(_control_chart(), _secondary_line())._build_merged()
    assert merged.to_spec().to_json() == PRE_CHANGE_LAYERED_PAYLOAD


def test_a_declared_layer_coord_rides_the_wire():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = fm.layer(_control_chart(), _secondary_line(fm.CoordFlip()))._build_merged()
    spec_json = json.loads(merged.to_spec().to_json())
    layers = spec_json["layers"]
    assert "coord" not in layers[0], (
        "the primary layer's coord IS the chart's; it must not be duplicated per layer"
    )
    assert layers[1]["coord"] == {"kind": "flip"}, (
        f"the secondary layer's own coord must reach the wire; got {layers[1]}"
    )
    from ferrum._core import ChartSpec

    payload = merged.to_spec().to_json()
    assert ChartSpec.from_json(payload).to_json() == payload


def test_the_merge_warning_no_longer_claims_the_coord_is_ignored():
    """A differing coord alone is no longer a conflict -- it is carried.

    The `+` merge still warns about the RHS's theme and facet, which it
    genuinely does discard; a warning naming `coord` would now be false.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = _control_chart() + _secondary_line(fm.CoordFlip())
    messages = [str(w.message) for w in caught]
    assert not [m for m in messages if "coord" in m], (
        f"a carried coord must not be reported as ignored; got {messages}"
    )


def test_the_merge_still_warns_about_a_discarded_theme():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = _control_chart() + _secondary_line().theme(fm.Theme(background_color="#000000"))
    messages = [str(w.message) for w in caught]
    assert [m for m in messages if "theme" in m], (
        f"the RHS theme IS still discarded and must still warn; got {messages}"
    )
