"""A later layer's auxiliary channel binding promotes to the chart level like ``color``.

Rust builds every auxiliary (non-positional) scale from the chart-level
encoding only, so ``Chart + Chart`` promotes the first layer-level binding of
each auxiliary channel to chart level. Without that, ``point + sized_line``
drew the line untapered and the dots at the default radius, while
``sized_line + point`` tapered the line and sized the dots: the result
depended on layer order.

The channel population is derived from the encoding vocabulary
(:func:`ferrum.encoding._auxiliary_scale_channels`) and pinned against the
Rust ``AuxiliaryScales`` struct, so a new auxiliary channel joins this matrix
without a hand edit.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl
import pytest

import ferrum as fm
import ferrum.chart as chart_module
from ferrum.encoding import _auxiliary_scale_channels, _channel_class_map

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

_CHANNELS = _auxiliary_scale_channels()

_AUX_SCALES_RS = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "ferrum-core"
    / "src"
    / "render"
    / "scale_resolve"
    / "mod.rs"
)


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
# Population pin: the derived channel set is Rust's auxiliary-scale set
# ---------------------------------------------------------------------------


def test_derived_channels_are_exactly_rusts_auxiliary_scales() -> None:
    """The promotion set equals the fields of Rust's ``AuxiliaryScales``, in either direction.

    Every auxiliary scale Rust builds reads the chart-level encoding only, so
    a channel missing from the derived set would stay layer-order dependent,
    and an extra one would promote a channel no scale reads.
    """
    src = _AUX_SCALES_RS.read_text()
    body = re.search(r"struct AuxiliaryScales \{(.*?)\n\}", src, re.DOTALL)
    assert body, "could not locate struct AuxiliaryScales -- has it moved?"
    rust_fields = re.findall(r"^\s*(\w+):", body.group(1), re.MULTILINE)
    assert len(rust_fields) >= 7, f"suspicious AuxiliaryScales parse: {rust_fields}"
    assert set(_CHANNELS) == set(rust_fields)
    assert len(_CHANNELS) == len(set(_CHANNELS))
    assert _CHANNELS[0] == "color"


# ---------------------------------------------------------------------------
# The matrix: channel x route x kind x order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("channel", _CHANNELS)
def test_a_layer_bound_channel_varies_the_rendered_marks(
    channel: str, route: str, kind: str, order: str
) -> None:
    """Every layer's marks vary by the bound field, whichever position the binder holds.

    The unbound partner inherits the promoted channel, exactly as an unbound
    layer inherits a promoted ``color``. The one exception is static
    ``fm.layer``: it renders through the composite overlay tree, where each
    layer is its own leaf with its own encoding, so only the binding leaf
    varies (``color`` behaves the same there).
    """
    chart = _compose(route, channel, order)
    if kind == "interactive":
        batches = _scene_batches(chart.interactive().scene_json, channel)
        assert len(batches) == 2, batches
        for i, values in enumerate(batches):
            assert len(set(values)) > 1, f"layer {i} drew one {channel} value: {values}"
        return
    _, points = _svg_values(chart.to_svg(), channel)
    assert len(points) == 8, points
    layers = [points[:4], points[4:]]
    if route == "fm_layer":
        bound, partner = layers if order == "bound_first" else layers[::-1]
        assert len(set(bound)) > 1 and len(set(partner)) == 1, layers
    else:
        assert all(len(set(values)) > 1 for values in layers), (
            f"a layer drew one {channel} value: {layers}"
        )


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("channel", _CHANNELS)
def test_layer_order_does_not_change_the_rendered_values(
    channel: str, route: str, kind: str
) -> None:
    """``bound + point`` and ``point + bound`` render the same per-layer values up to layer order."""
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


# ---------------------------------------------------------------------------
# Line consumers: a tapered line and a dashed line in either position
# ---------------------------------------------------------------------------

_LINE_CHANNELS = [c for c in _CHANNELS if c in ("size", "stroke_dash")]


@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("channel", _LINE_CHANNELS)
def test_a_later_bound_line_renders_like_a_leading_one(channel: str, route: str) -> None:
    """``point + bound_line`` equals ``bound_line + point`` up to layer order, and varies."""
    first = _scene_batches(
        _compose(route, channel, "bound_first", "line").interactive().scene_json, channel
    )
    later = _scene_batches(
        _compose(route, channel, "bound_later", "line").interactive().scene_json, channel
    )
    assert sorted(first, key=repr) == sorted(later, key=repr)
    assert all(len(set(values)) > 1 for values in later), later


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_point_plus_sized_line_tapers_the_line_and_sizes_the_dots(route: str) -> None:
    """The line's widths are the dot diameters at the same values; the dots carry those radii."""
    scene = json.loads(_compose(route, "size", "bound_later", "line").interactive().scene_json)
    batches = {b["kind"]: b for p in scene["panels"] for b in p["marks"]}
    (line_node,) = batches["line"]["nodes"]
    radii = [n["r"] for n in batches["point"]["nodes"]]
    widths = line_node["widths"]
    assert widths == pytest.approx([2 * r for r in radii])
    assert widths == sorted(widths) and widths[0] < widths[-1]
    # The default size range is 4..36 px^2 over v in 1..4.
    assert radii[0] == pytest.approx(math.sqrt(4 / math.pi))
    assert radii[-1] == pytest.approx(math.sqrt(36 / math.pi))


# ---------------------------------------------------------------------------
# Conflicts: mirror color's rule per channel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", _CHANNELS)
def test_two_layers_binding_different_fields_promote_the_first_in_layer_order(channel: str) -> None:
    """The first layer-level binding wins the chart level; the later field is not promoted."""
    f1, f2 = _fields(channel)
    for first, second in ((f1, f2), (f2, f1)):
        merged = _point() + _bound(channel, field=first) + _bound(channel, field=second)
        assert merged._encoding[channel].field == first
        assert [l.encoding[channel].field for l in merged._layers[1:]] == [first, second]


@pytest.mark.parametrize("channel", _CHANNELS)
def test_a_leading_chart_level_binding_is_never_overwritten(channel: str) -> None:
    """A chart-level binding on the left-hand chart keeps the chart level."""
    f1, f2 = _fields(channel)
    merged = _bound(channel, field=f1) + _bound(channel, field=f2)
    assert merged._encoding[channel].field == f1


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
def test_a_layer_zero_constant_does_not_block_promotion(channel: str) -> None:
    """A mark-kwarg constant on layer 0 is mark style, not an encoding, so the later field promotes.

    Rendered, each layer resolves its constant against the inherited field
    the same way whichever chart leads.
    """
    constant = fm.Chart(_DF).mark_point(**{channel: _CONSTANT_KWARG[channel]}).encode(x="x", y="y")
    later = constant + _bound(channel)
    first = _bound(channel) + constant
    assert later._encoding[channel].field == _fields(channel)[0]
    a = _scene_batches(later.interactive().scene_json, channel)
    b = _scene_batches(first.interactive().scene_json, channel)
    assert sorted(a, key=repr) == sorted(b, key=repr)


@pytest.mark.parametrize("channel", [c for c in _CHANNELS if c not in _CONSTANT_KWARG])
def test_a_channel_without_a_constant_cell_has_no_mark_kwarg(channel: str) -> None:
    with pytest.raises(TypeError, match=f"mark_point: unknown keyword argument '{channel}'"):
        fm.Chart(_DF).mark_point(**{channel: 0.5})


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def _without_promotion(monkeypatch) -> None:
    monkeypatch.setattr(chart_module, "_promote_layer_channels", lambda new: None)


def test_an_unbound_layered_chart_is_byte_identical_with_and_without_promotion(monkeypatch) -> None:
    def render():
        chart = _point() + fm.Chart(_DF).mark_line().encode(x="x", y="y")
        return chart.to_svg(), chart.interactive().scene_json

    on = render()
    _without_promotion(monkeypatch)
    assert render() == on


@pytest.mark.parametrize("channel", _CHANNELS)
def test_vacuity_control_a_later_binding_without_promotion_renders_differently(
    channel: str, monkeypatch
) -> None:
    """Switching the promotion off makes ``point + bound`` collapse to one value per layer."""
    on = _scene_batches(_compose("plus", channel, "bound_later").interactive().scene_json, channel)
    _without_promotion(monkeypatch)
    off = _scene_batches(_compose("plus", channel, "bound_later").interactive().scene_json, channel)
    assert off != on
    assert all(len(set(values)) == 1 for values in off), off
