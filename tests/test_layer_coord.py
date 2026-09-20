"""`#172`(1) -- a layer carries its own coordinate system.

A ``LayerChart`` member declares a coordinate system on its own ``Chart``
(``fm.Chart(df).mark_arc()...coord(fm.CoordPolar())``). The static shared-y
route keeps every member a separate composite leaf, so it has always honored
that declaration. The **merged-flat** route -- one single-panel ``Chart``
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

Every geometric assertion here is derived from the render itself rather than
from a hand-computed pixel: the control layer draws the same data at two
known domain endpoints, which pins the data->pixel map exactly (the scale is
linear and both endpoints are drawn), and the flipped layer's expected
vertices are then computed through that map. A broken (unflipped) render puts
those vertices on the opposite side of the panel, so the assertion cannot
pass for the wrong reason.
"""

from __future__ import annotations

import json
import warnings

import polars as pl
import pytest

import ferrum as fm
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
