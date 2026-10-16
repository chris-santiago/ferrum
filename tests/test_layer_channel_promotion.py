"""A layer's own auxiliary channel binding draws that layer and no other (spec §4.6 V1, V10).

Rust resolves each auxiliary (non-positional) scale -- color, size, shape,
opacity, fill_opacity, stroke_opacity, stroke_dash -- from whichever layer
draws the channel: layer 0 if it draws the channel, else the first layer
that does, with a domain over every layer that draws it. A binding written
on one ``+`` operand (or one ``fm.LayerChart`` member) is that operand's
own: it varies that layer's marks, and the other layer stays at the mark
default, in either layer order and on every route (spec §4.6 V1). ``sized_line +
point`` and ``point + sized_line`` draw the same marks per layer, because
neither binding reaches the other layer. Each test names the alternative
it rejects: *promotion* (the binding is copied to the chart level, so the
unbound partner inherits it).

The channel population is Rust's auxiliary-scale channel list,
``ferrum._core.auxiliary_scale_channels()``, so a new auxiliary channel
joins this matrix without a hand edit.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import auxiliary_scale_channels
from ferrum.encoding import _channel_class_map

_DF = pl.DataFrame(
    {
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 1.0, 0.0, 1.0],
        "v": [1.0, 2.0, 3.0, 4.0],
        "w": [4.0, 1.0, 3.0, 2.0],
        "g": ["a", "a", "b", "b"],
        "h": ["p", "q", "q", "q"],
    }
)

_CHANNELS = tuple(auxiliary_scale_channels())


def _fields(channel: str) -> tuple[str, str]:
    """Two distinct fields the channel can bind: nominal for a sorted channel, else numeric."""
    if "sort" in _channel_class_map()[channel]._honored_kwargs:
        return ("g", "h")
    return ("v", "w")


def _bound(channel: str, mark: str = "point", field: str | None = None) -> fm.Chart:
    field = field or _fields(channel)[0]
    return getattr(fm.Chart(_DF), f"mark_{mark}")().encode(x="x", y="y", **{channel: field})


def _point() -> fm.Chart:
    return fm.Chart(_DF).mark_point().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# Rendered-value extraction
# ---------------------------------------------------------------------------


def _scene_values(node: dict, channel: str) -> list:
    """The rendered values one interactive scene node carries for *channel*."""
    style = node.get("style") or {}
    if channel == "size":
        if "widths" in node:
            return [round(w, 6) for w in node["widths"]]
        if "r" in node:
            return [round(node["r"], 6)]
        if "width" in node:
            return [round(node["width"], 6)]
        return [style.get("width")]
    if channel == "shape":
        return [node["type"]]
    if channel == "color":
        paint = style.get("fill") or style.get("color")
        return [tuple(sorted(paint.items())) if paint else None]
    if channel == "stroke_dash":
        dash = style.get("stroke_dash") or style.get("dash")
        return [tuple(dash) if dash else None]
    return [style.get(channel)]


def _scene_batches(scene_json: str, channel: str) -> list[tuple]:
    """Per mark batch (one per layer), the flattened rendered values for *channel*."""
    out = []
    for panel in json.loads(scene_json)["panels"]:
        for batch in panel["marks"]:
            values: list = []
            for node in batch["nodes"]:
                values.extend(_scene_values(node, channel))
            out.append(tuple(values))
    return out


_SVG_ATTR = {
    "color": "fill",
    "opacity": "fill",
    "fill_opacity": "fill-opacity",
    "stroke_opacity": "stroke-opacity",
    "stroke_dash": "stroke-dasharray",
}


def _svg_value(tag: str, attrib: dict, channel: str):
    if channel == "shape":
        return tag
    if channel == "size":
        if tag == "circle":
            return attrib["r"]
        if tag == "rect":
            return attrib["width"]
        return attrib.get("stroke-width")
    return attrib.get(_SVG_ATTR[channel])


def _svg_plot_elements(svg: str) -> list[tuple[str, dict]]:
    """``(tag, attrib)`` for every drawable inside a plot-area clip group, legend excluded."""
    found: list[tuple[str, dict]] = []

    def walk(elem: ET.Element, clipped: bool) -> None:
        tag = elem.tag.rsplit("}", 1)[-1]
        clipped = clipped or elem.attrib.get("clip-path", "").startswith("url(#ferrum-clip")
        if clipped and tag in ("circle", "rect", "path", "polygon", "polyline"):
            found.append((tag, dict(elem.attrib)))
        for child in elem:
            walk(child, clipped)

    walk(ET.fromstring(svg), False)
    return found


def _svg_values(svg: str, channel: str) -> tuple[list, list]:
    """(values over every plot drawable, values over the point drawables only)."""
    elements = _svg_plot_elements(svg)
    every = [_svg_value(tag, a, channel) for tag, a in elements]
    points = [_svg_value(tag, a, channel) for tag, a in elements if tag != "polyline"]
    return every, points


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _plus(a, b):
    return a + b


def _fm_layer(a, b):
    return fm.layer(a, b)


def _independent_y(a, b):
    return fm.LayerChart(a, b, resolve={"y": "independent"})


_ROUTES = {"plus": _plus, "fm_layer": _fm_layer, "independent_y": _independent_y}
_KINDS = ("static", "interactive")
_ORDERS = ("bound_first", "bound_later")


def _compose(route: str, channel: str, order: str, bound_mark: str = "point"):
    bound = _bound(channel, bound_mark)
    pair = (bound, _point()) if order == "bound_first" else (_point(), bound)
    return _ROUTES[route](*pair)


# ---------------------------------------------------------------------------
# V1: a layer's own binding varies its own marks only, on every route
# ---------------------------------------------------------------------------


def _partner_and_bound(batches: list, order: str) -> tuple:
    """Split two per-layer value lists into (bound layer's, unbound partner's)."""
    bound, partner = batches if order == "bound_first" else batches[::-1]
    return bound, partner


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("channel", _CHANNELS)
def test_a_layer_bound_channel_varies_only_its_own_marks(
    channel: str, route: str, kind: str, order: str
) -> None:
    """V1: the binding layer's marks vary by the field; the partner stays at one value.

    Rejects promotion: the unbound partner inheriting the binding would
    vary too.
    """
    chart = _compose(route, channel, order)
    if kind == "interactive":
        batches = _scene_batches(chart.interactive().scene_json, channel)
        assert len(batches) == 2, batches
    else:
        _, points = _svg_values(chart.to_svg(), channel)
        assert len(points) == 8, points
        batches = [points[:4], points[4:]]
    bound, partner = _partner_and_bound(batches, order)
    assert len(set(bound)) > 1, f"the binding layer drew one {channel} value: {batches}"
    assert len(set(partner)) == 1, f"the unbound layer varied with {channel}: {batches}"


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("channel", _CHANNELS)
def test_layer_order_does_not_change_the_rendered_values(
    channel: str, route: str, kind: str
) -> None:
    """V1: ``bound + point`` and ``point + bound`` render the same per-layer values up to layer order."""
    first = _compose(route, channel, "bound_first")
    later = _compose(route, channel, "bound_later")
    if kind == "interactive":
        a = _scene_batches(first.interactive().scene_json, channel)
        b = _scene_batches(later.interactive().scene_json, channel)
        assert sorted(a, key=repr) == sorted(b, key=repr)
    else:
        assert sorted(map(str, _svg_values(first.to_svg(), channel)[0])) == sorted(
            map(str, _svg_values(later.to_svg(), channel)[0])
        )


@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("channel", _CHANNELS)
def test_the_chart_level_encoding_never_carries_an_operands_binding(
    channel: str, route: str
) -> None:
    """V1 wire seam: the merged chart-level encoding holds no binding only one member wrote."""
    for order in _ORDERS:
        chart = _compose(route, channel, order)
        merged = chart._build_merged() if isinstance(chart, fm.LayerChart) else chart
        wire = json.loads(merged.to_spec().to_json())
        assert channel not in wire["encoding"], wire["encoding"]


# ---------------------------------------------------------------------------
# Line consumers: a tapered line and a dashed line in either position
# ---------------------------------------------------------------------------

_LINE_CHANNELS = [c for c in _CHANNELS if c in ("size", "stroke_dash")]


@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("channel", _LINE_CHANNELS)
def test_a_later_bound_line_renders_like_a_leading_one(channel: str, route: str) -> None:
    """V1: ``point + bound_line`` equals ``bound_line + point`` up to layer order, and the line varies."""
    first = _scene_batches(
        _compose(route, channel, "bound_first", "line").interactive().scene_json, channel
    )
    later = _scene_batches(
        _compose(route, channel, "bound_later", "line").interactive().scene_json, channel
    )
    assert sorted(first, key=repr) == sorted(later, key=repr)
    bound, partner = later[1], later[0]
    assert len(set(bound)) > 1, later
    assert len(set(partner)) == 1, later


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_point_plus_sized_line_tapers_the_line_and_leaves_the_dots_at_the_default(
    route: str,
) -> None:
    """V1: the line tapers as it does alone; the dots keep the default radius.

    Rejects promotion: the dots would carry the line's size scale.
    """
    scene = json.loads(_compose(route, "size", "bound_later", "line").interactive().scene_json)
    batches = {b["kind"]: b for p in scene["panels"] for b in p["marks"]}
    (line_node,) = batches["line"]["nodes"]
    alone = json.loads(_bound("size", "line").interactive().scene_json)
    (alone_node,) = [n for p in alone["panels"] for b in p["marks"] for n in b["nodes"]]
    assert line_node["widths"] == pytest.approx(alone_node["widths"])
    assert line_node["widths"] == sorted(line_node["widths"])
    assert line_node["widths"][0] < line_node["widths"][-1]
    default = json.loads(_point().interactive().scene_json)
    default_radii = {n["r"] for p in default["panels"] for b in p["marks"] for n in b["nodes"]}
    assert {n["r"] for n in batches["point"]["nodes"]} == default_radii
    # The default size range is 4..36 px^2 over v in 1..4, read as a diameter.
    assert line_node["widths"][0] == pytest.approx(2 * math.sqrt(4 / math.pi))
    assert line_node["widths"][-1] == pytest.approx(2 * math.sqrt(36 / math.pi))


# ---------------------------------------------------------------------------
# Two binders: each keeps its own field; the first decides the scale (spec §4.6 V10)
# ---------------------------------------------------------------------------

# The channels Rust draws a legend for today; the opacity family's legends
# arrive with their own task.
_LEGEND_CHANNELS = [c for c in _CHANNELS if c in ("color", "size", "shape", "stroke_dash")]


def _legend_titles(svg: str) -> list[str]:
    root = ET.fromstring(svg)
    return [
        "".join(e.itertext())
        for e in root.iter()
        if e.tag.rsplit("}", 1)[-1] == "text"
        and e.attrib.get("font-weight") == "600"
        and e.attrib.get("text-anchor") == "start"
        and "transform" not in e.attrib
    ]


@pytest.mark.parametrize("channel", _CHANNELS)
def test_two_layers_binding_different_fields_each_keep_their_own(channel: str) -> None:
    """V1: each layer draws its own field; no layer's field reaches the chart level."""
    f1, f2 = _fields(channel)
    for first, second in ((f1, f2), (f2, f1)):
        merged = _point() + _bound(channel, field=first) + _bound(channel, field=second)
        assert channel not in merged._encoding
        assert channel not in merged._layers[0].encoding
        assert [l.encoding[channel].field for l in merged._layers[1:]] == [first, second]


@pytest.mark.parametrize("channel", _LEGEND_CHANNELS)
def test_the_first_layer_drawing_a_channel_titles_its_legend(channel: str) -> None:
    """V10: one legend per channel, titled by the first layer that draws it."""
    f1, f2 = _fields(channel)
    for first, second in ((f1, f2), (f2, f1)):
        svg = (_point() + _bound(channel, field=first) + _bound(channel, field=second)).to_svg()
        assert _legend_titles(svg) == [first]


# A constant per channel that ``mark_point`` accepts as a kwarg; the coverage
# test below proves every channel left out has no such kwarg.
_CONSTANT_KWARG = {
    "color": "#ff0000",
    "size": 100,
    "shape": "square",
    "opacity": 0.5,
    "stroke_dash": "dashed",
}


@pytest.mark.parametrize("channel", sorted(_CONSTANT_KWARG))
def test_a_layer_zero_constant_keeps_its_constant_beside_a_later_field(channel: str) -> None:
    """V1: a mark-kwarg constant is that layer's style; the later field draws only its own layer.

    Rendered, the two orders draw the same per-layer values.
    """
    constant = fm.Chart(_DF).mark_point(**{channel: _CONSTANT_KWARG[channel]}).encode(x="x", y="y")
    later = constant + _bound(channel)
    first = _bound(channel) + constant
    assert channel not in later._encoding
    a = _scene_batches(later.interactive().scene_json, channel)
    b = _scene_batches(first.interactive().scene_json, channel)
    assert sorted(a, key=repr) == sorted(b, key=repr)
    assert len(set(a[0])) == 1, a


@pytest.mark.parametrize("channel", [c for c in _CHANNELS if c not in _CONSTANT_KWARG])
def test_a_channel_without_a_constant_cell_has_no_mark_kwarg(channel: str) -> None:
    with pytest.raises(ValueError, match=f"mark_point: unknown keyword argument '{channel}'"):
        fm.Chart(_DF).mark_point(**{channel: 0.5})


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

# `point + line` with no auxiliary binding, as the tree before the one-parent
# model rendered it: SVG bytes and the interactive scene JSON.
_UNBOUND_SVG_DIGEST = "a7c51bb8e0acfdf660de97f8cffd0132d67b455d91c75fdc90ee80032687c49a"
_UNBOUND_SCENE_DIGEST = "2212bfb4a1fc00406a2d8b655892ed120c58a107568dfbcebcc0786964eccb4f"


def test_an_unbound_layered_chart_renders_its_pinned_bytes() -> None:
    import hashlib

    chart = _point() + fm.Chart(_DF).mark_line().encode(x="x", y="y")
    assert hashlib.sha256(chart.to_svg().encode()).hexdigest() == _UNBOUND_SVG_DIGEST
    scene = chart.interactive().scene_json
    assert hashlib.sha256(scene.encode()).hexdigest() == _UNBOUND_SCENE_DIGEST
