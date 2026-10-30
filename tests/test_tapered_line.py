"""`#116`(c)/(e): `mark_line` size/stroke_width width conflict and the taper.

A line layer that binds `size` to draw a tapered dot-diameter width and
`stroke_width` to draw a pixel width at the same time cannot honor both, so
it refuses with the exact spec §6.3 text -- for every pairing of {field,
constant} x {field, constant}, on every rendering door (`to_svg`,
`.interactive()`, `to_png`, `save`), and on a layered chart with the
`layer {i}: ` prefix naming the offending layer. Every layered route names
the offending layer, including nested composition: `chart + chart` and
`fm.layer(...).interactive()` read the bare `layer {i}: ` text; `fm.layer(...)
.to_svg()`, and any overlay nested inside a grid composition such as
`fm.hconcat(ok, fm.layer(ok, bad))`, read the same `layer {i}: ` text inside
the crate's `failed to render composite leaf leaf #{index}: ` envelope --
the leaf index is a figure-wide position, the layer index is the member's
position within its own overlay group, and the two agree only for a
top-level `fm.layer`. A line that binds only `size` draws a width per
vertex, `2*sqrt(size_scale(v)/pi)` -- the same area-to-diameter mapping the
point mark uses, so a line and a point sharing a size encoding carry the
same per-row magnitude (point radius = half the line width). A constant
`size=` kwarg draws the same formula as a constant width, with no per-vertex
`widths`.

A layer's own binding never reaches a sibling (see
`tests/test_layer_channel_promotion.py`), so a line layer that binds only
its own `stroke_width` (field or constant) beside a `size`-bound layer
draws at its own `stroke_width`, untapered, on `+` and `fm.layer(...)`, in
either layer order, static and interactive alike, while the other layer's
`size` sizes only its own marks. A line that binds neither draws at the
line's default width. The refusal still fires, unaffected, when one line
layer binds both channels itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import re
import tempfile
from functools import reduce
from itertools import combinations
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from ferrum import ChartSpec
from ferrum._core import (
    interpolate_accepted_values,
    render_interactive,
    render_svg,
    stroke_dash_palette,
)
from tests._snapshots import decode_png_rgba, rasterize_svg

_DF = pl.DataFrame(
    {
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 1.0, 0.0, 1.0],
        "v": [1.0, 2.0, 3.0, 4.0],
        "w": [1.0, 1.0, 1.0, 1.0],
    }
)

# ---------------------------------------------------------------------------
# The four refusal cells: {field, constant} x {field, constant} for
# size/stroke_width. Each value is (mark kwargs, encoding kwargs, the exact
# spec §6.3 text), captured from the real exception raised by this build, not
# built from a Python-side guess at Rust's `f64::Display` (a field renders
# `'name'`, a whole-number constant drops its decimal point: `size=10.0`
# prints `size=10`).
# ---------------------------------------------------------------------------

_CELLS: dict[str, tuple[dict, dict, str]] = {
    "field_field": (
        {},
        {"size": "v", "stroke_width": "w"},
        "mark_line: size and stroke_width both set the line's width (size='v', "
        "stroke_width='w'); bind only one. size= sets a dot-diameter width that can "
        "taper along the line; stroke_width= sets a width in pixels.",
    ),
    "field_const": (
        {"stroke_width": 2},
        {"size": "v"},
        "mark_line: size and stroke_width both set the line's width (size='v', "
        "stroke_width=2); bind only one. size= sets a dot-diameter width that can "
        "taper along the line; stroke_width= sets a width in pixels.",
    ),
    "const_field": (
        {"size": 10},
        {"stroke_width": "w"},
        "mark_line: size and stroke_width both set the line's width (size=10, "
        "stroke_width='w'); bind only one. size= sets a dot-diameter width that can "
        "taper along the line; stroke_width= sets a width in pixels.",
    ),
    "const_const": (
        {"size": 10, "stroke_width": 2.5},
        {},
        "mark_line: size and stroke_width both set the line's width (size=10, "
        "stroke_width=2.5); bind only one. size= sets a dot-diameter width that can "
        "taper along the line; stroke_width= sets a width in pixels.",
    ),
}

_ROUTES = ("to_svg", "interactive", "to_png", "save")


def _chart(mark_kwargs: dict, encoding_kwargs: dict) -> fm.Chart:
    return fm.Chart(_DF).mark_line(**mark_kwargs).encode(x="x", y="y", **encoding_kwargs)


def _render(chart: fm.Chart, route: str) -> None:
    if route == "to_svg":
        chart.to_svg()
    elif route == "interactive":
        chart.interactive()
    elif route == "to_png":
        chart.to_png()
    else:
        with tempfile.NamedTemporaryFile(suffix=".svg") as f:
            chart.save(f.name)


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("cell", sorted(_CELLS))
def test_the_four_refusal_cells_raise_with_the_exact_spec_text(cell: str, route: str) -> None:
    mark_kwargs, encoding_kwargs, text = _CELLS[cell]
    with pytest.raises(ValueError) as exc:
        _render(_chart(mark_kwargs, encoding_kwargs), route)
    assert str(exc.value) == text


# ---------------------------------------------------------------------------
# Vacuity control: binding only one of the two channels must not raise, on
# the same route -- guards against a matrix whose render helper swallows the
# call or whose assertion is satisfied by any `ValueError`.
# ---------------------------------------------------------------------------


def test_binding_only_one_channel_does_not_raise() -> None:
    _chart({}, {"size": "v"}).to_svg()
    _chart({"stroke_width": 2}, {}).to_svg()


# ---------------------------------------------------------------------------
# Layered charts. Route rule (mirrors `_ABSENT_DECLARED_MESSAGES` in
# `test_layer_shared_axis_refusal.py`): `chart + chart` and
# `fm.layer(...).interactive()` both render through the merged single-panel
# layer path (`spec.layers` populated on one chart), so the refusal carries
# the bare `layer {i}: ` prefix; `fm.layer(...).to_svg()`, and any overlay
# nested inside a grid composition, render each member as its own standalone
# composite leaf (no `spec.layers` on a leaf), wrapped in the crate's general
# `failed to render composite leaf leaf #{index}: ` envelope -- but the
# `layer {i}: ` prefix still names the offending member's position within its
# own overlay group, so every layered route names the offending layer. The
# leaf index is a figure-wide pre-order position and the layer index is the
# member's position within its own overlay group; the two agree only for a
# top-level `fm.layer`, so a nested overlay (e.g. `fm.hconcat(ok,
# fm.layer(ok, bad))`) shows a leaf index larger than its layer index.
# ---------------------------------------------------------------------------

_CONFLICT_CORE = (
    "mark_line: size and stroke_width both set the line's width (size='v', "
    "stroke_width=2); bind only one. size= sets a dot-diameter width that can taper "
    "along the line; stroke_width= sets a width in pixels."
)


def _base() -> fm.Chart:
    return fm.Chart(_DF).mark_point().encode(x="x", y="y")


def _conflicted_line() -> fm.Chart:
    return fm.Chart(_DF).mark_line(stroke_width=2).encode(x="x", y="y", size="v")


def test_plus_route_to_svg_reads_the_bare_layer_prefixed_text() -> None:
    with pytest.raises(ValueError) as exc:
        (_base() + _conflicted_line()).to_svg()
    assert str(exc.value) == f"layer 1: {_CONFLICT_CORE}"


def test_plus_route_interactive_reads_the_bare_layer_prefixed_text() -> None:
    with pytest.raises(ValueError) as exc:
        (_base() + _conflicted_line()).interactive()
    assert str(exc.value) == f"layer 1: {_CONFLICT_CORE}"


def test_fm_layer_interactive_reads_the_bare_layer_prefixed_text() -> None:
    with pytest.raises(ValueError) as exc:
        fm.layer(_base(), _conflicted_line()).interactive()
    assert str(exc.value) == f"layer 1: {_CONFLICT_CORE}"


def test_fm_layer_to_svg_reads_the_composite_leaf_wrapped_text() -> None:
    with pytest.raises(ValueError) as exc:
        fm.layer(_base(), _conflicted_line()).to_svg()
    assert str(exc.value) == f"failed to render composite leaf leaf #1: layer 1: {_CONFLICT_CORE}"


def test_fm_layer_to_svg_names_the_leader_as_layer_zero() -> None:
    """The offending member can be the overlay group's leader (layer 0), not
    just a non-primary member (layer 1 above): the leaf index and the layer
    index both start counting from the same member here."""
    with pytest.raises(ValueError) as exc:
        fm.layer(_conflicted_line(), _base()).to_svg()
    assert str(exc.value) == f"failed to render composite leaf leaf #0: layer 0: {_CONFLICT_CORE}"


def test_nested_overlay_to_svg_names_its_layer_not_its_leaf() -> None:
    """A top-level `fm.layer`'s leaf index equals its layer index; nested
    inside a grid composition it does not -- the leaf index is a figure-wide
    pre-order position (3rd leaf overall, `leaf #2`) while the layer index is
    still the member's position within its own overlay group (2nd member,
    `layer 1`)."""
    with pytest.raises(ValueError) as exc:
        fm.hconcat(_base(), fm.layer(_base(), _conflicted_line())).to_svg()
    assert str(exc.value) == f"failed to render composite leaf leaf #2: layer 1: {_CONFLICT_CORE}"


# ---------------------------------------------------------------------------
# Scene-level checks: the constant twin, the encoded-outranks-constant rule,
# per-vertex taper widths, and the unsized control.
# ---------------------------------------------------------------------------


# `_line_nodes` and `_point_radii` read the real parsed `scene_json` from
# `.interactive()`.
def _line_nodes(chart: fm.Chart) -> list[dict]:
    scene = json.loads(chart.interactive().scene_json)
    nodes = []
    for panel in scene["panels"]:
        for batch in panel["marks"]:
            if batch["kind"] == "line":
                nodes.extend(batch["nodes"])
    return nodes


def _point_radii(chart: fm.Chart) -> list[float]:
    scene = json.loads(chart.interactive().scene_json)
    radii = []
    for panel in scene["panels"]:
        for batch in panel["marks"]:
            if batch["kind"] == "point":
                radii.extend(node["r"] for node in batch["nodes"])
    return radii


def test_constant_size_kwarg_draws_the_dot_diameter_width_with_no_widths_key() -> None:
    nodes = _line_nodes(fm.Chart(_DF).mark_line(size=10).encode(x="x", y="y"))
    assert len(nodes) == 1
    assert nodes[0]["style"]["width"] == pytest.approx(2 * math.sqrt(10 / math.pi))
    assert "widths" not in nodes[0]


def test_an_encoded_size_outranks_the_constant_size_kwarg() -> None:
    nodes = _line_nodes(fm.Chart(_DF).mark_line(size=10).encode(x="x", y="y", size="v"))
    assert len(nodes) == 1
    assert "widths" in nodes[0]
    # The constant survives only as the fallback `style.width` a reader that
    # ignores `widths` would use (spec §6.2); the encoded size wins the
    # per-vertex widths.
    assert nodes[0]["style"]["width"] == pytest.approx(2 * math.sqrt(10 / math.pi))


def test_sized_line_widths_equal_twice_the_point_mark_radius_at_the_same_value() -> None:
    """The size scale is mark-agnostic (§6.2): a point and a line sharing the
    same size encoding and domain map the same row to the same area, so a
    line's per-vertex width is `2 * r` of a point drawn with the identical
    encoding -- a real cross-mark invariant, not a reimplementation of the
    scale's own domain/range math in Python."""
    line_nodes = _line_nodes(fm.Chart(_DF).mark_line().encode(x="x", y="y", size="v"))
    radii = _point_radii(fm.Chart(_DF).mark_point().encode(x="x", y="y", size="v"))
    assert len(line_nodes) == 1
    widths = line_nodes[0]["widths"]
    assert len(widths) == len(line_nodes[0]["points"]) == len(radii) == len(_DF)
    for w, r in zip(widths, radii):
        assert w == pytest.approx(2 * r)
    assert all(a < b for a, b in zip(widths, widths[1:]))  # monotone with v


def test_an_unsized_line_has_no_widths_key() -> None:
    nodes = _line_nodes(fm.Chart(_DF).mark_line().encode(x="x", y="y"))
    assert len(nodes) == 1
    assert "widths" not in nodes[0]


# ---------------------------------------------------------------------------
# A line layer's own `stroke_width` draws beside a sibling's `size`, which
# never reaches it. `point(size="v") + line(stroke_width=3)` sizes the dots
# and draws the line at a flat, untapered 3px width, in either layer order,
# on both `+` and `fm.layer(...)`, whether the line's own `stroke_width` is
# a constant kwarg or its own field encoding.
#
# A field-bound `stroke_width` maps through the stroke-width scale, never
# draws its raw values as pixels: a constant column (`w` here) has a
# degenerate domain and draws at the midpoint of the default 1-4 px band,
# 2.5 px.
# ---------------------------------------------------------------------------

_CONSTANT_FIELD_WIDTH = 2.5


def _sized_point() -> fm.Chart:
    return fm.Chart(_DF).mark_point().encode(x="x", y="y", size="v")


def _own_stroke_width_line(width: float) -> fm.Chart:
    return fm.Chart(_DF).mark_line(stroke_width=width).encode(x="x", y="y")


def _own_stroke_width_line_field() -> fm.Chart:
    return fm.Chart(_DF).mark_line().encode(x="x", y="y", stroke_width="w")


def _svg_polyline_stroke_widths(svg: str) -> list[str]:
    return re.findall(r'<polyline[^>]*stroke-width="([^"]+)"', svg)


_MUST_DRAW_CELLS: dict[str, tuple] = {
    "plus_point_first_constant": (lambda: _sized_point() + _own_stroke_width_line(3), "3"),
    "plus_line_first_constant": (lambda: _own_stroke_width_line(3) + _sized_point(), "3"),
    "plus_point_first_field": (lambda: _sized_point() + _own_stroke_width_line_field(), "2.5"),
    "plus_line_first_field": (lambda: _own_stroke_width_line_field() + _sized_point(), "2.5"),
    "fm_layer_point_first": (
        lambda: fm.layer(_sized_point(), _own_stroke_width_line(3)),
        "3",
    ),
    "fm_layer_line_first": (
        lambda: fm.layer(_own_stroke_width_line(3), _sized_point()),
        "3",
    ),
}


@pytest.mark.parametrize("cell", sorted(_MUST_DRAW_CELLS))
def test_own_stroke_width_draws_beside_a_siblings_size(cell: str) -> None:
    build, expected_width = _MUST_DRAW_CELLS[cell]
    chart = build()

    # Static: one untapered polyline at the line's own stroke width.
    assert _svg_polyline_stroke_widths(chart.to_svg()) == [expected_width]

    # Interactive: no per-vertex `widths`, and the fallback width is the
    # line's own, not derived from the sibling's size.
    nodes = _line_nodes(chart)
    assert len(nodes) == 1
    assert "widths" not in nodes[0]
    assert nodes[0]["style"]["width"] == pytest.approx(float(expected_width))

    # The dots are sized by their own `size`.
    radii = _point_radii(chart)
    assert len(radii) == 4
    assert radii == sorted(radii) and radii[0] < radii[-1]


def test_a_line_binding_neither_channel_draws_untapered_at_the_default_width() -> None:
    """Neither `size` nor `stroke_width` reaches the third layer from its
    siblings, so it draws untapered at the line's default width, as it does
    alone; the first line keeps its own `stroke_width`."""
    first_line = fm.Chart(_DF).mark_line().encode(x="x", y="y", stroke_width="w")
    sized_point = fm.Chart(_DF).mark_point().encode(x="x", y="y", size="v")
    bare_line = fm.Chart(_DF).mark_line().encode(x="x", y="y")

    chart = first_line + sized_point + bare_line
    nodes = _line_nodes(chart)
    assert len(nodes) == 2
    for node in nodes:
        assert "widths" not in node
    # 'w' is constant on the first line, so it draws the scale's midpoint;
    # the bare line keeps the default.
    alone = _line_nodes(bare_line)[0]["style"]["width"]
    assert [node["style"]["width"] for node in nodes] == pytest.approx(
        [_CONSTANT_FIELD_WIDTH, alone]
    )

    radii = _point_radii(chart)
    assert len(radii) == 4
    assert radii == sorted(radii) and radii[0] < radii[-1]

    assert _svg_polyline_stroke_widths(chart.to_svg()) == [
        "2.5",
        *_svg_polyline_stroke_widths(bare_line.to_svg()),
    ]


def test_the_conflict_refusal_is_unaffected_by_a_siblings_size() -> None:
    """A line layer that binds both channels itself still refuses beside a
    sibling's own `size` binding -- precedence applies only to a channel the
    line *inherits*, never to a channel it binds itself."""
    conflicted = fm.Chart(_DF).mark_line(stroke_width=2).encode(x="x", y="y", size="v")
    text = f"layer 1: {_CONFLICT_CORE}"

    with pytest.raises(ValueError) as exc:
        (_sized_point() + conflicted).to_svg()
    assert str(exc.value) == text

    with pytest.raises(ValueError) as exc:
        fm.layer(_sized_point(), conflicted).to_svg()
    assert str(exc.value) == f"failed to render composite leaf leaf #1: {text}"

    with pytest.raises(ValueError) as exc:
        (_sized_point() + conflicted).interactive()
    assert str(exc.value) == text


# ---------------------------------------------------------------------------
# A line layer's own constant `size=` or `stroke_width=` kwarg draws beside
# a sibling's width field, which never reaches it: the kwarg-bound layer
# draws its own constant width, untapered, in either layer order, static
# and interactive alike.
# ---------------------------------------------------------------------------

_OWN_KWARG_DF = pl.DataFrame(
    {
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 3.0, 1.0, 2.0],
        "v": [1.0, 2.0, 3.0, 4.0],
        "w": [2.0, 2.0, 2.0, 2.0],
    }
)


def _own_kwarg_sized_line() -> fm.Chart:
    return fm.Chart(_OWN_KWARG_DF).mark_line(size=5).encode(x="x", y="y")


def _sized_point_beside() -> fm.Chart:
    return fm.Chart(_OWN_KWARG_DF).mark_point().encode(x="x", y="y", size="v")


@pytest.mark.parametrize(
    "chart_factory",
    [
        lambda: _sized_point_beside() + _own_kwarg_sized_line(),
        lambda: _own_kwarg_sized_line() + _sized_point_beside(),
    ],
    ids=["point_first", "line_first"],
)
def test_own_size_kwarg_draws_beside_a_siblings_size_field(chart_factory, recwarn) -> None:
    """A line's own `size=` kwarg draws its constant width beside a sibling
    point layer's `size` field, in either layer order."""
    control_width = 2 * math.sqrt(5 / math.pi)
    chart = chart_factory()

    nodes = _line_nodes(chart)
    assert len(nodes) == 1
    assert "widths" not in nodes[0]
    assert nodes[0]["style"]["width"] == pytest.approx(control_width)

    svg_widths = _svg_polyline_stroke_widths(chart.to_svg())
    assert len(svg_widths) == 1
    assert float(svg_widths[0]) == pytest.approx(control_width, abs=5e-4)

    # The dots are sized by their own `size`.
    radii = _point_radii(chart)
    assert len(radii) == 4
    assert radii == sorted(radii) and radii[0] < radii[-1]

    user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert not user_warnings, [str(w.message) for w in user_warnings]


def test_own_stroke_width_field_then_own_constant_draw_each_their_own_width(recwarn) -> None:
    """The field line (layer 0) binds `stroke_width='w'` and the constant
    line (layer 1) its own `stroke_width=3`; each draws its own width, in
    layer order, neither binding reaching the other line. The constant
    field column draws the stroke-width scale's midpoint."""
    field_line = fm.Chart(_OWN_KWARG_DF).mark_line().encode(x="x", y="y", stroke_width="w")
    const_line = fm.Chart(_OWN_KWARG_DF).mark_line(stroke_width=3).encode(x="x", y="y")

    chart = field_line + const_line
    assert _svg_polyline_stroke_widths(chart.to_svg()) == ["2.5", "3"]
    nodes = _line_nodes(chart)
    assert [n["style"]["width"] for n in nodes] == [_CONSTANT_FIELD_WIDTH, 3.0]
    assert all("widths" not in n for n in nodes)

    user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert not user_warnings, [str(w.message) for w in user_warnings]


def test_own_constant_stroke_width_then_own_field_is_order_independent(recwarn) -> None:
    """Order-independence pin: the `stroke_width=` kwarg line first, the
    field line second, draws each line's own width, as the other order does."""
    const_line = fm.Chart(_OWN_KWARG_DF).mark_line(stroke_width=3).encode(x="x", y="y")
    field_line = fm.Chart(_OWN_KWARG_DF).mark_line().encode(x="x", y="y", stroke_width="w")

    chart = const_line + field_line
    assert _svg_polyline_stroke_widths(chart.to_svg()) == ["3", "2.5"]
    nodes = _line_nodes(chart)
    assert [n["style"]["width"] for n in nodes] == [3.0, _CONSTANT_FIELD_WIDTH]
    assert all("widths" not in n for n in nodes)

    user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert not user_warnings, [str(w.message) for w in user_warnings]


# ---------------------------------------------------------------------------
# A constant `size=` of 0 or below refuses with a typed `ValueError` on
# both `mark_line` and `mark_point`, at the shared numeric kwarg check in
# `MarkBase.__init__` (`src/ferrum/marks/base.py`) that already refuses
# NaN -- in that check's own voice.
# ---------------------------------------------------------------------------

_NON_POSITIVE_SIZE_CELLS: dict[str, tuple[str, int, str]] = {
    "line_zero": ("mark_line", 0, "mark_line(size=...) must be a positive number; received 0"),
    "line_negative": (
        "mark_line",
        -4,
        "mark_line(size=...) must be a positive number; received -4",
    ),
    "point_zero": (
        "mark_point",
        0,
        "mark_point(size=...) must be a positive number; received 0",
    ),
    "point_negative": (
        "mark_point",
        -4,
        "mark_point(size=...) must be a positive number; received -4",
    ),
}


def _sized_chart(mark: str, size: int) -> fm.Chart:
    return getattr(fm.Chart(_DF), mark)(size=size).encode(x="x", y="y")


@pytest.mark.parametrize("cell", sorted(_NON_POSITIVE_SIZE_CELLS))
def test_non_positive_constant_size_refuses_with_the_exact_message(cell: str) -> None:
    """The refusal fires in `MarkBase.__init__`, at mark construction --
    before `to_svg()`/`.interactive()` ever runs -- so there is one
    construction path to cover, not a per-route one."""
    mark, size, text = _NON_POSITIVE_SIZE_CELLS[cell]
    with pytest.raises(ValueError) as exc:
        _sized_chart(mark, size)
    assert str(exc.value) == text


@pytest.mark.parametrize("mark", ("mark_line", "mark_point"))
def test_positive_constant_size_still_renders(mark: str) -> None:
    _sized_chart(mark, 5).to_svg()
    _sized_chart(mark, 5).interactive()


# ---------------------------------------------------------------------------
# `#116` static SVG outline walker (spec §6.5). Every assertion below reads
# the real `to_svg()`/`to_png()`/`save()` output of a real `fm.Chart`, never
# a hand-built scene. PNG rasterization and decoding go through
# `tests._snapshots`'s shared `rasterize_svg`/`decode_png_rgba`; only the
# outline `<path>` grammar is parsed here directly, and that grammar is:
# ``M{x} {y} L{x} {y} ... Z`` per ring, numbers 3-decimal `fmt_f` form, rings
# concatenated with no separator other than the next ``M``.
# ---------------------------------------------------------------------------


def _outline_elements(svg: str) -> list[str]:
    """Every outline `<path ... stroke="none"/>` element in *svg*, in order."""
    return [m for m in re.findall(r"<path[^>]*/>", svg) if ' stroke="none"' in m]


def _outline_vertices(el: str) -> list[tuple[float, float]]:
    """Every vertex of every ring in outline element *el*, flattened.

    The ``M``/``L``/``Z`` letters never match the numeric regex, so scanning
    the whole ``d`` string (ring boundaries included) yields the same
    vertices as parsing ring-by-ring.
    """
    d = re.search(r' d="([^"]*)"', el).group(1)
    return [(float(a), float(b)) for a, b in re.findall(r"(-?[\d.]+) (-?[\d.]+)", d)]


def _outline_rings(el: str) -> list[list[tuple[float, float]]]:
    """Every ring of outline element *el* as its own vertex list."""
    d = re.search(r' d="([^"]*)"', el).group(1)
    rings = []
    for piece in d.split("M")[1:]:
        piece = piece.strip()
        if piece.endswith("Z"):
            piece = piece[:-1].strip()
        rings.append([(float(a), float(b)) for a, b in re.findall(r"(-?[\d.]+) (-?[\d.]+)", piece)])
    return rings


def _ring_area(ring: list[tuple[float, float]]) -> float:
    """The shoelace area of one ring, in absolute SVG-coordinate units."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        total += x0 * y1 - x1 * y0
    return abs(total) / 2


def _outline_area(el: str) -> float:
    """The real painted area of outline element *el*: the sum of its rings'
    absolute areas (Rust's own `painted_area`, mirrored for the Python
    door, spec §6.5's dash contract)."""
    return sum(_ring_area(ring) for ring in _outline_rings(el))


def _unit_normal(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """The unit normal of segment a->b, per spec §6.5: `n = (-dy, dx)/|b-a|`."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    return (-dy / length, dx / length)


def _unit_tangent(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    return (dx / length, dy / length)


def _has_vertex_near(
    vertices: list[tuple[float, float]], target: tuple[float, float], tol: float = 1e-2
) -> bool:
    return any(math.hypot(vx - target[0], vy - target[1]) < tol for vx, vy in vertices)


def _close(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-2) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) < tol


def _line_nodes_of(chart: fm.Chart) -> list[dict]:
    """Every line node in the real scene JSON, in panel and batch order."""
    scene = json.loads(chart.interactive().scene_json)
    return [
        n for p in scene["panels"] for b in p["marks"] if b["kind"] == "line" for n in b["nodes"]
    ]


def _single_line_node(chart: fm.Chart) -> tuple[list[tuple[float, float]], list[float]]:
    """The one line node's `(points, widths)`, read from the real scene JSON."""
    found = _line_nodes_of(chart)
    assert len(found) == 1, f"expected one line node, got {len(found)}"
    node = found[0]
    return [tuple(p) for p in node["points"]], node["widths"]


# ---------------------------------------------------------------------------
# Matrix: interpolate family (derived from `interpolate_accepted_values()`) x
# dash state {no dash, constant dash, encoded dash} x opacity {1, 0.5} x
# color grouping {off, on}. Rust's own `svg_walk.rs` matrix covers {no dash,
# constant dash} exhaustively against hand-built scenes; the "encoded dash"
# column is covered here instead, driven through the real Python door end
# to end.
# ---------------------------------------------------------------------------

_MATRIX_X = [0.0, 1.0, 2.0, 3.0]
_MATRIX_Y = [0.0, 1.0, 0.0, 1.0]
_MATRIX_V = [1.0, 2.0, 3.0, 4.0]


def _matrix_chart(
    interpolate: str, dash_state: str, opacity: float | None, grouped: bool, *, sized: bool
) -> fm.Chart:
    groups = ["a", "b"] if grouped else ["a"]
    df = pl.DataFrame(
        {
            "x": _MATRIX_X * len(groups),
            "y": _MATRIX_Y * len(groups),
            "v": _MATRIX_V * len(groups),
            "g": [g for g in groups for _ in _MATRIX_X],
        }
    )
    mark_kwargs: dict = {"interpolate": interpolate}
    if opacity is not None:
        mark_kwargs["opacity"] = opacity
    if dash_state == "constant":
        mark_kwargs["stroke_dash"] = [6.0, 3.0]
    encoding_kwargs: dict = {}
    if sized:
        encoding_kwargs["size"] = "v"
    if grouped:
        encoding_kwargs["color"] = "g"
    if dash_state == "encoded":
        encoding_kwargs["stroke_dash"] = "g"
    chart = fm.Chart(df).mark_line(**mark_kwargs).encode(x="x", y="y", **encoding_kwargs)
    return chart.properties(width=240, height=160)


def test_sized_line_outlines_under_every_interpolate_dash_state_opacity_and_grouping() -> None:
    interpolate_families = interpolate_accepted_values()
    dash_states = ("none", "constant", "encoded")
    opacities: tuple[float | None, ...] = (None, 0.5)
    groupings = (False, True)
    # The real dash palette (never a Python-side mirror of it): a categorical
    # `stroke_dash` field assigns domain position j the palette's own pattern
    # j, position 0 always solid. Series 0 (group "a", first-appearance
    # order) is therefore always solid under "encoded"; series 1 (group "b")
    # is the one that carries a real dash.
    palette = stroke_dash_palette()
    cells = 0
    for interpolate in interpolate_families:
        for dash_state in dash_states:
            for opacity in opacities:
                for grouped in groupings:
                    cell = f"{interpolate}/{dash_state}/opacity={opacity}/grouped={grouped}"
                    sized_svg = _matrix_chart(
                        interpolate, dash_state, opacity, grouped, sized=True
                    ).to_svg()
                    control_svg = _matrix_chart(
                        interpolate, dash_state, opacity, grouped, sized=False
                    ).to_svg()

                    outlines = _outline_elements(sized_svg)
                    series = 2 if grouped else 1
                    assert len(outlines) == series, f"{cell}: one outline per series"
                    assert "<polyline" not in sized_svg, f"{cell}: no stroked polyline"
                    for el in outlines:
                        assert "stroke-dasharray" not in el, f"{cell}: {el}"
                        assert "stroke-width" not in el, f"{cell}: {el}"
                        assert ("rgba(" in el) == (opacity is not None), (
                            f"{cell}: opacity on fill: {el}"
                        )

                    # Vacuity control: the unsized twin outlines nothing and
                    # differs from the sized render.
                    assert _outline_elements(control_svg) == [], f"{cell}: unsized twin outlines"
                    assert sized_svg != control_svg, cell

                    # Dashing actually happened: each dashed series paints
                    # the pattern's on-fraction of its undashed twin's area,
                    # within 0.02 (mirrors the Rust matrix's area-ratio
                    # check, `sized_lines_outline_under_every_interpolate_
                    # dash_opacity_and_grouping` in `svg_walk/tests.rs`).
                    # Ring counts don't discriminate on curves: a dash gap
                    # can drop or add a whole flattened segment either way.
                    if dash_state != "none":
                        undashed_svg = _matrix_chart(
                            interpolate, "none", opacity, grouped, sized=True
                        ).to_svg()
                        undashed_outlines = _outline_elements(undashed_svg)
                        assert len(undashed_outlines) == series, f"{cell}: undashed twin series"
                        for i, (dashed_el, undashed_el) in enumerate(
                            zip(outlines, undashed_outlines)
                        ):
                            ratio = _outline_area(dashed_el) / _outline_area(undashed_el)
                            if dash_state == "constant":
                                pattern = [6.0, 3.0]
                            else:  # encoded: series i's own assigned pattern
                                pattern = palette[i]
                            expected = sum(pattern[0::2]) / sum(pattern) if pattern else 1.0
                            assert abs(ratio - expected) <= 0.02, (
                                f"{cell} series {i}: dashed paints {ratio:.3f} of the "
                                f"undashed area, want {expected:.3f}"
                            )
                    cells += 1
    assert cells == len(interpolate_families) * len(dash_states) * len(opacities) * len(groupings)


# ---------------------------------------------------------------------------
# Per-vertex half-thickness: for the default (linear) interpolate family, the
# outline carries, at every vertex and for every adjacent segment, a corner
# at `p_i +/- (w_i/2) * n` (spec §6.5), where `w_i` is the real per-vertex
# width read from the scene (`.interactive()`'s `widths`), not a value this
# test derives independently.
# ---------------------------------------------------------------------------


def test_static_outline_half_thickness_equals_half_the_scene_width_at_every_vertex() -> None:
    df = pl.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 2.0, 0.5, 3.0], "v": [1.0, 2.0, 3.0, 4.0]}
    )
    chart = (
        fm.Chart(df).mark_line().encode(x="x", y="y", size="v").properties(width=320, height=220)
    )
    points, widths = _single_line_node(chart)

    outlines = _outline_elements(chart.to_svg())
    assert len(outlines) == 1
    vertices = _outline_vertices(outlines[0])

    for i, (p, w) in enumerate(zip(points, widths)):
        half = w / 2
        neighbor_idxs = [j for j in (i - 1, i + 1) if 0 <= j < len(points)]
        assert neighbor_idxs, f"vertex {i} has no adjacent segment"
        for j in neighbor_idxs:
            other = points[j]
            nx, ny = _unit_normal(p, other) if j == i + 1 else _unit_normal(other, p)
            for sign in (1, -1):
                target = (p[0] + sign * nx * half, p[1] + sign * ny * half)
                assert _has_vertex_near(vertices, target), (
                    f"vertex {i}: missing corner at {target} (w={w})"
                )


# ---------------------------------------------------------------------------
# Caps: butt (and the unset default) end flush at the vertex, square extends
# by `w/2` along the outward tangent, round's end vertices lie on the
# radius-`w/2` circle around the endpoint (spec §6.5).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", (None, "butt", "round", "square"))
def test_caps_follow_the_stroke_cap(cap: str | None) -> None:
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "v": [10.0, 10.0]})
    kwargs = {} if cap is None else {"stroke_cap": cap}
    chart = (
        fm.Chart(df)
        .mark_line(**kwargs)
        .encode(x="x", y="y", size="v")
        .properties(width=300, height=200)
    )
    points, widths = _single_line_node(chart)
    p0, p1 = points[0], points[1]
    half = widths[0] / 2
    tx, ty = _unit_tangent(p0, p1)
    nx, ny = _unit_normal(p0, p1)

    vertices = _outline_vertices(_outline_elements(chart.to_svg())[0])
    # Outward tangential projection of every vertex relative to p0: positive
    # means the cap reaches back past the vertex (square/round); ~0 means
    # flush (butt/unset).
    projections = [-((vx - p0[0]) * tx + (vy - p0[1]) * ty) for vx, vy in vertices]
    reach = max(projections)

    if cap in (None, "butt"):
        assert reach <= half * 0.1, f"cap={cap}: reaches {reach}, expected flush"
        for sign in (1, -1):
            assert _has_vertex_near(vertices, (p0[0] + sign * nx * half, p0[1] + sign * ny * half))
    elif cap == "square":
        assert math.isclose(reach, half, rel_tol=0.05), (
            f"cap=square: reach {reach}, expected {half}"
        )
    else:  # round
        assert reach > half * 0.3, f"cap=round: reach {reach}, expected a bulge past flush"
        near_p0 = [
            (vx, vy) for vx, vy in vertices if math.hypot(vx - p0[0], vy - p0[1]) < half * 1.2
        ]
        assert len(near_p0) >= 3, "round cap should add arc vertices beyond the two corners"
        for vx, vy in near_p0:
            dist = math.hypot(vx - p0[0], vy - p0[1])
            assert math.isclose(dist, half, rel_tol=0.1), (
                f"cap=round: vertex at radius {dist}, expected {half}"
            )


# ---------------------------------------------------------------------------
# Joins: with no join set, a tapered line uses bevel (not SVG's untapered
# miter default) -- an outer join ring of exactly the vertex plus its two
# segment corners, nothing farther. Round adds several arc vertices, all at
# radius `w/2` from the vertex. Miter adds exactly one vertex strictly past
# `w/2` (spec §6.5).
# ---------------------------------------------------------------------------


def _join_ring_other_distances(chart: fm.Chart, p1: tuple[float, float]) -> list[float]:
    outlines = _outline_elements(chart.to_svg())
    assert len(outlines) == 1
    for ring in _outline_rings(outlines[0]):
        if any(_close(v, p1) for v in ring):
            others = [v for v in ring if not _close(v, p1)]
            return [math.hypot(vx - p1[0], vy - p1[1]) for vx, vy in others]
    raise AssertionError(f"no ring contains the joint vertex {p1}")


@pytest.mark.parametrize("join", (None, "bevel", "round", "miter"))
def test_joins_follow_the_stroke_join(join: str | None) -> None:
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 1.0], "v": [20.0, 20.0, 20.0]})
    kwargs = {} if join is None else {"stroke_join": join}
    chart = (
        fm.Chart(df)
        .mark_line(**kwargs)
        .encode(x="x", y="y", size="v")
        .properties(width=300, height=200)
    )
    points, widths = _single_line_node(chart)
    p1 = points[1]
    half = widths[1] / 2

    distances = _join_ring_other_distances(chart, p1)

    if join in (None, "bevel"):
        assert len(distances) == 2, (
            f"join={join}: expected exactly the two corners, got {distances}"
        )
        for d in distances:
            assert math.isclose(d, half, rel_tol=0.05), (
                f"join={join}: corner at {d}, expected {half}"
            )
    elif join == "round":
        assert len(distances) > 2, "join=round: expected arc vertices beyond the two corners"
        for d in distances:
            assert math.isclose(d, half, rel_tol=0.1), (
                f"join=round: vertex at {d}, expected radius {half}"
            )
    else:  # miter
        assert len(distances) == 3, (
            f"join=miter: expected the two corners plus one tip, got {distances}"
        )
        tip = max(distances)
        assert tip > half * 1.05, f"join=miter: tip at {tip}, expected strictly past {half}"


def test_no_join_set_renders_identically_to_bevel() -> None:
    """With no `stroke_join=`, a tapered line uses bevel -- not SVG's
    untapered-line miter default (spec §6.5, disclosed in `ferrum-spec.md`
    and the changelog). Compared by outline geometry, not full SVG bytes:
    an explicit `stroke_join="bevel"` wraps the batch in an inert
    `<g stroke-linejoin="bevel">` that an outline never reads (the join is
    baked into the fill geometry), so the two SVGs differ by that wrapper
    even though the painted shape is identical."""
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 1.0], "v": [20.0, 20.0, 20.0]})
    default_svg = (
        fm.Chart(df)
        .mark_line()
        .encode(x="x", y="y", size="v")
        .properties(width=300, height=200)
        .to_svg()
    )
    bevel_svg = (
        fm.Chart(df)
        .mark_line(stroke_join="bevel")
        .encode(x="x", y="y", size="v")
        .properties(width=300, height=200)
        .to_svg()
    )
    default_rings = _outline_rings(_outline_elements(default_svg)[0])
    bevel_rings = _outline_rings(_outline_elements(bevel_svg)[0])
    assert default_rings == bevel_rings


# ---------------------------------------------------------------------------
# Dashed outlines: multiple on-pieces separated by real gaps along the
# stroke, with widths that vary (interpolated), not one width reused for
# every piece. The data is a horizontal line so each on-piece's width is
# simply its own y-extent -- a real property of the real rendered geometry,
# not a hand-built expectation.
# ---------------------------------------------------------------------------


def test_dashed_outlines_alternate_along_arc_length_with_interpolated_widths() -> None:
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    chart = (
        fm.Chart(df)
        .mark_line(stroke_dash=[6.0, 3.0])
        .encode(x="x", y="y", size="v")
        .properties(width=400, height=120)
    )
    outlines = _outline_elements(chart.to_svg())
    assert len(outlines) == 1
    rings = _outline_rings(outlines[0])
    assert len(rings) > 1, "a dashed tapered line must fragment into multiple rings"

    spans = sorted(
        (
            (
                min(x for x, _ in r),
                max(x for x, _ in r),
                max(y for _, y in r) - min(y for _, y in r),
            )
            for r in rings
        ),
        key=lambda s: s[0],
    )

    # Merge rings that abut (a width-interpolation cut at a data vertex, not
    # a real dash gap) into on-pieces, tracking each piece's max y-extent
    # (its width) and x-range.
    pieces: list[list[float]] = []  # [xmin, xmax, width]
    for xmin, xmax, width in spans:
        if pieces and xmin - pieces[-1][1] < 0.5:
            pieces[-1][1] = max(pieces[-1][1], xmax)
            pieces[-1][2] = max(pieces[-1][2], width)
        else:
            pieces.append([xmin, xmax, width])

    assert len(pieces) > 1, "the dash pattern must produce more than one on-piece"

    # Real gaps exist between consecutive on-pieces (the off-dash length).
    gaps = [pieces[i + 1][0] - pieces[i][1] for i in range(len(pieces) - 1)]
    assert all(g > 1.0 for g in gaps), f"expected real off-dash gaps, got {gaps}"

    # Widths are bounded by the real scene widths and are not all equal --
    # interpolation along arc length, not one reused width.
    _, widths = _single_line_node(chart)
    piece_widths = [p[2] for p in pieces]
    assert min(piece_widths) >= min(widths) - 0.5
    assert max(piece_widths) <= max(widths) + 0.5
    assert max(piece_widths) - min(piece_widths) > 1.0, piece_widths
    # Monotone overall: later on-pieces (farther along the line) are no
    # thinner than earlier ones, matching the monotone size encoding.
    assert piece_widths[0] < piece_widths[-1]


# ---------------------------------------------------------------------------
# Hairpin at opacity 0.5 paints its self-overlap once. Rasterized through the
# real `resvg` path (`tests/_snapshots.py`'s `rasterize_svg`), no covered
# pixel in the overlap column may be darker than the midpoint between the
# real single- and double-coverage blends of the stroke onto this chart's
# own background -- that rules out a doubled fill -- and the real
# single-coverage blend must actually appear. A plain "every covered pixel
# shares one color" check false-fails on real anti-aliased edge pixels,
# which blend lighter than single coverage, not darker.
# ---------------------------------------------------------------------------


def _blend(
    bg: tuple[int, int, int], fg: tuple[int, int, int], alpha: float
) -> tuple[int, int, int]:
    return tuple(round(bg[i] * (1 - alpha) + fg[i] * alpha) for i in range(3))


def test_hairpin_at_half_opacity_paints_its_overlap_once() -> None:
    df = pl.DataFrame({"x": [0.0, 10.0, 0.0], "y": [0.0, 0.0, 0.2], "v": [30.0, 30.0, 30.0]})
    chart = (
        fm.Chart(df)
        .mark_line(stroke="#ff0000", opacity=0.5)
        .encode(x="x", y=fm.Y("y", scale=fm.LinearScale(domain=[-50.0, 50.0])), size="v")
        .properties(width=400, height=300)
        .configure_grid(x=False, y=False)
    )
    svg = chart.to_svg()
    outlines = _outline_elements(svg)
    assert len(outlines) == 1, "the hairpin must paint through one outline"
    vertices = _outline_vertices(outlines[0])
    xmin = min(v[0] for v in vertices)
    xmax = max(v[0] for v in vertices)
    ymin = min(v[1] for v in vertices)
    ymax = max(v[1] for v in vertices)

    png = rasterize_svg(svg)
    rows, w, h = decode_png_rgba(png)
    background = rows[0][0][:3]
    col = int((xmin + xmax) / 2)
    column = [rows[y][col] for y in range(max(0, int(ymin) - 2), min(h, int(ymax) + 3))]
    covered = [p for p in column if p[:3] != background]
    assert covered, "expected the hairpin to paint the sampled column"

    single = _blend(background, (255, 0, 0), 0.5)
    double = _blend(single, (255, 0, 0), 0.5)
    midpoint_green = (single[1] + double[1]) / 2

    # Anti-aliased edge pixels blend *toward the background* (lighter, a
    # higher green channel than single coverage) -- that is not a defect.
    # A real double-paint pulls a pixel *darker* than single coverage,
    # toward the double-coverage blend, which is the one thing this test
    # must never see.
    greens = [p[1] for p in covered]
    assert min(greens) > midpoint_green, (
        f"a pixel in the overlap column is darker than single coverage "
        f"(green {min(greens)} vs the single/double midpoint {midpoint_green}): {sorted(set(covered))}"
    )
    assert any(abs(g - single[1]) <= 2 for g in greens), (
        f"the single-coverage blend {single} never appears in the column: {sorted(set(covered))}"
    )


# ---------------------------------------------------------------------------
# PNG: sample the actual pixel thickness of the rendered red stroke at three
# points along the line, and check it against the real per-vertex widths the
# scene carries -- not merely that `to_png()` returns bytes.
# ---------------------------------------------------------------------------


def _is_red(pixel: tuple[int, int, int, int]) -> bool:
    return pixel[0] > 200 and pixel[1] < 200


def test_to_png_pixel_thickness_matches_the_tapered_widths() -> None:
    df = pl.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 1.0, 1.0, 1.0], "v": [1.0, 2.0, 3.0, 4.0]}
    )
    chart = (
        fm.Chart(df)
        .mark_line(stroke="#ff0000")
        .encode(x="x", y="y", size="v")
        .properties(width=400, height=200)
    )
    points, widths = _single_line_node(chart)
    outlines = _outline_elements(chart.to_svg())
    assert len(outlines) == 1

    xs = [p[0] for p in points]
    xmin, xmax = min(xs), max(xs)

    png = chart.to_png(scale=1.0)
    rows, w, h = decode_png_rgba(png)
    sx = w / 400

    def red_thickness(frac: float) -> int:
        col = int((xmin + (xmax - xmin) * frac) * sx)
        return sum(1 for y in range(h) if _is_red(rows[y][col]))

    measured = [red_thickness(f) for f in (0.02, 0.5, 0.98)]
    expected = [round(widths[0]), round((widths[1] + widths[2]) / 2), round(widths[-1])]

    # Anti-aliasing at the outline's edges counts a fringe pixel or two as
    # "red" on each side, so allow a couple of pixels of slack -- the real
    # signal this proves is that thickness tracks the real per-vertex
    # widths, not a constant, untapered stroke.
    assert measured == sorted(measured), f"thickness should grow along the line: {measured}"
    for got, want in zip(measured, expected):
        assert abs(got - want) <= 2, f"measured {measured}, expected approximately {expected}"


# ---------------------------------------------------------------------------
# Vacuity control: an unsized line, dashed or curved under a round cap/join,
# never takes the tapered-outline path, since that path only fires when
# `widths` is present on the scene node. The SHA256s below are this build's
# bytes for each chart -- a snapshot pin, not a claim of provenance against
# any earlier build; a layout change moves them. The vacuity property itself
# (no per-vertex widths, no filled outline, drawn as a plain stroked line)
# is asserted structurally in the same tests, so a walker that took the
# outline path for an unsized line fails on that structural check even once
# the hash has been re-pinned to match.
# ---------------------------------------------------------------------------


def _vacuity_dashed_chart() -> fm.Chart:
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 0.0, 1.0]})
    return (
        fm.Chart(df)
        .mark_line(stroke_dash=[6.0, 3.0])
        .encode(x="x", y="y")
        .properties(width=300, height=200)
    )


def _vacuity_curved_chart() -> fm.Chart:
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 2.0, 0.5, 3.0]})
    return (
        fm.Chart(df)
        .mark_line(interpolate="monotone", stroke_cap="round", stroke_join="round")
        .encode(x="x", y="y")
        .properties(width=300, height=200)
    )


_VACUITY_DASHED_SHA256 = "e86df1d9a52d2cccb5a46989302418d37881c7557e4dfcc0ee73834ff4508688"
_VACUITY_CURVED_ROUND_SHA256 = "49e3c23086cdc080a7bec6a2dc081a61a27e5d991429dbed73f8744b1d2862a6"


def test_unsized_dashed_line_draws_a_stroked_dash_array_with_no_tapered_outline() -> None:
    chart = _vacuity_dashed_chart()
    nodes = _line_nodes_of(chart)
    assert len(nodes) == 1
    assert "widths" not in nodes[0]

    svg = chart.to_svg()
    assert _outline_elements(svg) == []
    strokes = re.findall(r'<polyline[^>]*stroke-dasharray="6,3"[^>]*/>', svg)
    assert len(strokes) == 1
    assert hashlib.sha256(svg.encode()).hexdigest() == _VACUITY_DASHED_SHA256


def test_unsized_curved_round_cap_join_line_draws_a_stroked_path_with_no_tapered_outline() -> None:
    chart = _vacuity_curved_chart()
    nodes = _line_nodes_of(chart)
    assert len(nodes) == 1
    assert "widths" not in nodes[0]

    svg = chart.to_svg()
    assert _outline_elements(svg) == []
    strokes = re.findall(r'<path[^>]*fill="none"[^>]*/>', svg)
    assert len(strokes) == 1
    assert '<g stroke-linecap="round" stroke-linejoin="round">' in svg
    assert hashlib.sha256(svg.encode()).hexdigest() == _VACUITY_CURVED_ROUND_SHA256


# ---------------------------------------------------------------------------
# Break axis: a sized line cut by a primary-y gap keeps tapering. Each piece
# carries the unbroken line's width at every kept vertex and, at each cut,
# the width linear in the cut's fraction of the source edge (spec §6.4); the
# static outline is `w/2` thick there. The expected widths come from the
# unbroken chart's scene and the data fraction of the gap edge along the
# crossing edge, not from the broken scene under test.
# ---------------------------------------------------------------------------

_BREAK_Y = [0.0, 100.0, 500.0, 900.0]
_BREAK_GAP = (150.0, 850.0)


def _break_line_chart(*, broken: bool) -> fm.Chart:
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": _BREAK_Y, "v": [1.0, 2.0, 3.0, 4.0]})
    chart = (
        fm.Chart(df).mark_line().encode(x="x", y="y", size="v").properties(width=320, height=220)
    )
    return chart + fm.BreakAxis(axis="y", gap=_BREAK_GAP) if broken else chart


def test_a_sized_line_cut_by_a_break_tapers_with_interpolated_widths_at_each_cut() -> None:
    _, source = _single_line_node(_break_line_chart(broken=False))
    lo, hi = _BREAK_GAP
    y = _BREAK_Y
    # Vertex 2 (y=500) sits in the gap: edge 1 leaves at `lo`, edge 2 re-enters at `hi`.
    t_exit = (lo - y[1]) / (y[2] - y[1])
    t_entry = (hi - y[2]) / (y[3] - y[2])
    exit_w = source[1] + (source[2] - source[1]) * t_exit
    entry_w = source[2] + (source[3] - source[2]) * t_entry
    expected = [[source[0], source[1], exit_w], [entry_w, source[3]]]

    chart = _break_line_chart(broken=True)
    nodes = _line_nodes_of(chart)
    assert [len(n["points"]) for n in nodes] == [3, 2]
    for node, want in zip(nodes, expected):
        assert node["widths"] == pytest.approx(want, abs=1e-6)

    outlines = _outline_elements(chart.to_svg())
    assert len(outlines) == 2, "one tapered outline per piece"
    for node, outline, want in zip(nodes, outlines, expected):
        points = [tuple(p) for p in node["points"]]
        vertices = _outline_vertices(outline)
        for i, (p, w) in enumerate(zip(points, want)):
            j = i + 1 if i + 1 < len(points) else i - 1
            nx, ny = _unit_normal(p, points[j]) if j > i else _unit_normal(points[j], p)
            for sign in (1, -1):
                target = (p[0] + sign * nx * w / 2, p[1] + sign * ny * w / 2)
                assert _has_vertex_near(vertices, target), (
                    f"vertex {i}: no corner at {target} (w={w})"
                )
    # The cut widths are neither endpoint's: the taper runs through the cut.
    assert source[1] < exit_w < source[2] and source[2] < entry_w < source[3]


# ---------------------------------------------------------------------------
# Legend swatches: a size legend draws line swatches exactly when every layer
# binding the size field is a line, across composite leaves for a legend
# they share, and circles otherwise. A line swatch's stroke width is the
# entry's dot diameter, `2 * r` of the circle the same entry draws on a
# point-only chart. The population is every non-empty set of size consumers
# drawn from {line, point}, on every route that combines them: one flat
# chart (`+`), the static overlay composite (`fm.layer(...).to_svg()`, one
# shared legend read over every member), a figure-level legend band shared
# across `hconcat` leaves, and the interactive scene.
# ---------------------------------------------------------------------------

_SWATCH_MARKS = ("line", "point")
_CONSUMER_SETS = [
    combo for n in range(1, len(_SWATCH_MARKS) + 1) for combo in combinations(_SWATCH_MARKS, n)
]
_SHARED_SIZE = fm.Resolve(scale={"size": "shared"}, legend={"size": "shared"})


def _consumer(mark: str) -> fm.Chart:
    return getattr(fm.Chart(_DF), f"mark_{mark}")().encode(x="x", y="y", size="v")


def _combine(route: str, marks: tuple[str, ...]) -> Any:
    charts = [_consumer(m) for m in marks]
    if route in ("plus", "interactive"):
        return reduce(operator.add, charts)
    if route == "layer":
        return fm.layer(*charts)
    return fm.hconcat(*charts, resolve=_SHARED_SIZE)


_MARK_COLOR = "#2563eb"


def _strip_clip_groups(svg: str) -> str:
    """Remove every `<g clip-path=...>...</g>` block, honoring nested `<g>`s.

    A clip group can itself contain nested `<g>` elements (e.g. per-mark
    grouping inside a clipped panel). A non-greedy `<g clip-path=...>.*?</g>`
    regex stops at the *first* `</g>` it finds, which is the nested group's
    close tag, not the clip group's own -- leaking the clip group's
    remaining marks into whatever this function is meant to exclude them
    from. This scans forward counting open/close `<g>` tags to find each
    clip group's true matching close instead.
    """
    out = []
    i = 0
    while True:
        m = re.search(r'<g\s+clip-path="[^"]*">', svg[i:])
        if not m:
            out.append(svg[i:])
            break
        start = i + m.start()
        out.append(svg[i:start])
        depth = 0
        end = len(svg)
        for tm in re.finditer(r"<g\b[^>]*>|</g>", svg[start:]):
            if tm.group().startswith("</g"):
                depth -= 1
                if depth == 0:
                    end = start + tm.end()
                    break
            else:
                depth += 1
        i = end
    return "".join(out)


def _svg_legend_swatches(svg: str) -> list[tuple[str, float]]:
    """`(kind, size)` per legend swatch of a static SVG, in document order.

    Marks render inside clip groups and axis chrome in gray, so what stays
    outside every clip group in the mark color is the legend: a circle swatch
    reports its radius, a line swatch its stroke width.
    """
    body = re.sub(r"<defs>.*?</defs>", "", svg, flags=re.S)
    body = _strip_clip_groups(body)
    swatches = []
    for el in re.findall(r"<(?:circle|line)\b[^>]*>", body):
        if el.startswith("<circle"):
            swatches.append(("circle", float(re.search(r' r="([^"]+)"', el).group(1))))
        elif f'stroke="{_MARK_COLOR}"' in el:
            swatches.append(("line", float(re.search(r' stroke-width="([^"]+)"', el).group(1))))
    return swatches


def test_strip_clip_groups_handles_nested_g() -> None:
    """A mark circle nested inside a clip group's own `<g>` must not leak out.

    A non-greedy `<g clip-path=...>.*?</g>` regex would stop at the nested
    `</g>` and leave `mark-circle` (and the clip group's own close tag) in
    the body; the balanced scanner must strip the whole outer group.
    """
    svg = (
        '<g clip-path="url(#a)"><g class="marks">'
        '<circle id="mark-circle" r="1"/></g></g>'
        '<circle id="legend-circle" r="2"/>'
    )
    stripped = _strip_clip_groups(svg)
    assert "mark-circle" not in stripped
    assert "legend-circle" in stripped


def _scene_legend_swatches(chart: Any) -> list[tuple[str, float]]:
    scene = json.loads(chart.interactive().scene_json)
    swatches = []
    for node in scene["legend"]:
        if node["type"] == "circle":
            swatches.append(("circle", node["r"]))
        elif node["type"] == "line":
            swatches.append(("line", node["style"]["width"]))
    return swatches


def _legend_swatches(route: str, chart: Any) -> list[tuple[str, float]]:
    if route == "interactive":
        return _scene_legend_swatches(chart)
    return _svg_legend_swatches(chart.to_svg())


@pytest.mark.parametrize("route", ["plus", "layer", "hconcat_shared", "interactive"])
@pytest.mark.parametrize("marks", _CONSUMER_SETS, ids="+".join)
def test_size_legend_swatch_follows_every_size_consumer(route: str, marks: tuple[str, ...]) -> None:
    radii = [r for _, r in _scene_legend_swatches(_consumer("point"))]
    swatches = _legend_swatches(route, _combine(route, marks))
    # Every route, including the static overlay composite (`fm.layer`),
    # resolves the size legend once across every member and draws exactly
    # `len(radii)` entries, read over every size consumer.
    assert len(swatches) == len(radii), swatches
    lines_only = set(marks) == {"line"}
    for i, (kind, size) in enumerate(swatches):
        r = radii[i % len(radii)]
        if lines_only:
            assert kind == "line", swatches
            assert size == pytest.approx(2 * r, abs=2e-3)
        else:
            assert kind == "circle", swatches
            assert size == pytest.approx(r, abs=2e-3)


def test_line_only_legend_differs_from_point_only() -> None:
    """Vacuity control: the matrix's two single-consumer cells must disagree."""
    line = _svg_legend_swatches(_consumer("line").to_svg())
    point = _svg_legend_swatches(_consumer("point").to_svg())
    assert {k for k, _ in line} == {"line"}
    assert {k for k, _ in point} == {"circle"}
    line_sizes = [s for _, s in line]
    point_sizes = [s for _, s in point]
    assert line_sizes != point_sizes, "line widths must differ from point radii, not just kinds"
    assert line_sizes == pytest.approx([2 * r for r in point_sizes], abs=2e-3)


@pytest.mark.parametrize("route", ["plus", "layer", "interactive"])
def test_a_rule_beside_a_sized_line_keeps_the_line_swatches(route: str) -> None:
    """A rule beside a sized line binds no size (the line's binding is its
    own), so the line is the legend's only consumer."""
    rule = fm.Chart(_DF).mark_rule().encode(y="y")
    chart = _consumer("line") + rule if route != "layer" else fm.layer(_consumer("line"), rule)
    radii = [r for _, r in _scene_legend_swatches(_consumer("point"))]
    swatches = _legend_swatches(route, chart)
    assert [k for k, _ in swatches] == ["line"] * len(radii)
    assert [w for _, w in swatches] == pytest.approx([2 * r for r in radii], abs=2e-3)


# ---------------------------------------------------------------------------
# Python doors. `relplot(kind="line", size=)` draws the same taper as
# `mark_line().encode(size=)`; a hand-built wire spec reaches the width
# refusal through the same line build as a Python chart; a conditional
# `size` draws the base appearance statically, as it does on a point.
# ---------------------------------------------------------------------------


def _endpoint_half_thickness(
    vertices: list[tuple[float, float]], end: tuple[float, float]
) -> float:
    """Half the outline's thickness at a butt-capped end: its corners are the
    outline vertices nearest the centerline's endpoint."""
    return min(math.dist(v, end) for v in vertices)


def test_relplot_line_size_tapers_its_outline_from_the_low_to_the_high_value() -> None:
    df = pl.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 0.5, 1.5], "v": [1.0, 2.0, 3.0, 4.0]}
    )
    chart = fm.relplot(df, x="x", y="y", size="v", kind="line", height=220, aspect=1.5)
    points, widths = _single_line_node(chart)

    outlines = _outline_elements(chart.to_svg())
    assert len(outlines) == 1
    vertices = _outline_vertices(outlines[0])
    low = _endpoint_half_thickness(vertices, points[0])
    high = _endpoint_half_thickness(vertices, points[-1])
    assert low == pytest.approx(widths[0] / 2, abs=1e-2)
    assert high == pytest.approx(widths[-1] / 2, abs=1e-2)
    assert high > low + 1.0


_WIRE_CONFLICT = {
    "data": {"kind": "named", "name": "default"},
    "mark": "line",
    "encoding": {"x": {"field": "x"}, "y": {"field": "y"}, "size": {"field": "v"}},
    "mark_style": {"stroke_width": 2.0},
}


@pytest.mark.parametrize("output_kind", ("static", "interactive"))
def test_a_hand_built_wire_spec_reaches_the_width_refusal(output_kind: str) -> None:
    renderer = render_svg if output_kind == "static" else render_interactive
    with pytest.raises(ValueError) as exc:
        renderer(
            ChartSpec.from_json(json.dumps(_WIRE_CONFLICT)), _DF.to_arrow(), viewport=(400.0, 300.0)
        )
    assert str(exc.value) == _CONFLICT_CORE

    # Control: the same wire without `stroke_width` renders.
    unconflicted = {k: v for k, v in _WIRE_CONFLICT.items() if k != "mark_style"}
    renderer(ChartSpec.from_json(json.dumps(unconflicted)), _DF.to_arrow(), viewport=(400.0, 300.0))


def _conditional_size_chart(mark: str) -> fm.Chart:
    sel = fm.selection_point(name="s", clear="false")
    size = fm.when(sel).then(100).otherwise(20)
    return getattr(fm.Chart(_DF), mark)().encode(x="x", y="y", size=size).add_selection(sel)


def test_a_conditional_size_draws_a_line_at_its_base_width_as_it_does_a_point(recwarn) -> None:
    line = _line_nodes(_conditional_size_chart("mark_line"))
    base_line = _line_nodes(fm.Chart(_DF).mark_line().encode(x="x", y="y"))
    assert len(line) == 1
    assert "widths" not in line[0]
    assert line[0]["style"]["width"] == base_line[0]["style"]["width"] == 1.5
    svg = _conditional_size_chart("mark_line").to_svg()
    assert re.findall(r'<polyline[^>]*\bstroke-width="([\d.]+)"', svg) == ["1.5"]
    assert not _outline_elements(svg)

    radii = _point_radii(_conditional_size_chart("mark_point"))
    assert radii == _point_radii(fm.Chart(_DF).mark_point().encode(x="x", y="y"))
    assert not recwarn.list, [str(w.message) for w in recwarn.list]


# ---------------------------------------------------------------------------
# A size conditional restyles, and checks positivity on, every line in the
# layers that declared it. One declared on the chart as a whole (a layered
# chart, `mark_line(point=True)`, the host of `Chart.layer()` or `SecondaryY`)
# belongs to every layer, lines included, as a field `size` does; one declared
# on an operand of `chart + chart` or on a `Layer`'s own encoding stays on
# those layers, so an overlay line beside a point layer's `size=fm.when(...)`
# keeps its own width. Points and rects keep it chart-wide.
# ---------------------------------------------------------------------------

_COND_SEL = fm.selection_point(name="s", clear="false")

_COND_CONFLICT = (
    "mark_line: size and stroke_width both set the line's width (size=fm.when(...), "
    "stroke_width=3); bind only one. size= sets a dot-diameter width that can taper "
    "along the line; stroke_width= sets a width in pixels."
)
_COND_BRANCH_ZERO = (
    "mark_line: size=fm.when(...) on selection 's' has a branch of size 0, which gives "
    "no positive line width. Every width along a line must be positive; give every "
    "branch a size above 0."
)


def _cond_size(otherwise: float) -> Any:
    return fm.when(_COND_SEL).then(100).otherwise(otherwise)


def _cond_point(otherwise: float = 0) -> fm.Chart:
    return fm.Chart(_DF).mark_point().encode(x="x", y="y", size=_cond_size(otherwise))


def _plain_line() -> fm.Chart:
    return fm.Chart(_DF).mark_line().encode(x="x", y="y")


def _cond_line(otherwise: float = 20, **mark_kwargs: Any) -> fm.Chart:
    return fm.Chart(_DF).mark_line(**mark_kwargs).encode(x="x", y="y", size=_cond_size(otherwise))


def _scene_line_batches(chart: Any) -> list[dict]:
    scene = json.loads(chart.interactive().scene_json)
    return [b for p in scene["panels"] for b in p["marks"] if b["kind"] == "line"]


def _conditional_stamps(chart: fm.Chart) -> list[list[int] | None]:
    wire = json.loads(chart.to_spec().to_json())
    return [c.get("layers") for c in wire["conditionals"]]


_SIBLING_ROUTES = {
    "plus_point_first": lambda: (_cond_point() + _plain_line()).add_selection(_COND_SEL),
    "plus_line_first": lambda: (_plain_line() + _cond_point()).add_selection(_COND_SEL),
    "fm_layer_point_first": lambda: fm.layer(_cond_point(), _plain_line()),
    "fm_layer_line_first": lambda: fm.layer(_plain_line(), _cond_point()),
    "independent_y": lambda: fm.LayerChart(
        _cond_point(), _plain_line(), resolve={"y": "independent"}
    ),
}


@pytest.mark.parametrize("route", sorted(_SIBLING_ROUTES))
def test_a_sibling_layers_conditional_size_neither_restyles_nor_refuses_a_line(
    route: str,
) -> None:
    chart = _SIBLING_ROUTES[route]()
    svg = chart.to_svg()
    assert re.findall(r'<polyline[^>]*\bstroke-width="([\d.]+)"', svg) == ["1.5"]
    batches = _scene_line_batches(chart)
    assert len(batches) == 1
    assert "vertex_rows" not in batches[0]
    assert [n["style"]["width"] for n in batches[0]["nodes"]] == [1.5]
    scene = json.loads(chart.interactive().scene_json)
    assert [c["channel"] for c in scene["interaction"]["conditionals"]] == ["size"]


def _cond_line_with_points(otherwise: float = 20, **mark_kwargs: Any) -> fm.Chart:
    return (
        fm.Chart(_DF)
        .mark_line(point=True, **mark_kwargs)
        .encode(x="x", y="y", size=_cond_size(otherwise))
    )


def _two_lines_chart_level(then: float = 100) -> fm.Chart:
    size = fm.when(_COND_SEL).then(then).otherwise(20)
    return (_plain_line() + _plain_line()).encode(size=size).add_selection(_COND_SEL)


def test_a_chart_level_conditional_size_restyles_every_line_of_a_layered_chart() -> None:
    batches = _scene_line_batches(_two_lines_chart_level())
    assert [b.get("vertex_rows") for b in batches] == [[[0, 1, 2, 3]], [[0, 1, 2, 3]]]


@pytest.mark.parametrize("output", ["to_svg", "interactive"])
def test_a_chart_level_conditional_size_with_a_zero_branch_refuses_on_layered_lines(
    output: str,
) -> None:
    with pytest.raises(ValueError) as exc:
        _render(_two_lines_chart_level(then=0), output)
    assert str(exc.value) == f"layer 0: {_COND_BRANCH_ZERO}"


def test_a_conditional_size_on_mark_line_point_true_restyles_the_line() -> None:
    batches = _scene_line_batches(_cond_line_with_points().add_selection(_COND_SEL))
    assert [b.get("vertex_rows") for b in batches] == [[[0, 1, 2, 3]]]


@pytest.mark.parametrize("output", ["to_svg", "interactive"])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"otherwise": 0}, _COND_BRANCH_ZERO),
        ({"stroke_width": 3}, _COND_CONFLICT),
    ],
    ids=["otherwise_zero", "stroke_width"],
)
def test_a_conditional_size_on_mark_line_point_true_refuses_like_a_plain_line(
    kwargs: dict, message: str, output: str
) -> None:
    with pytest.raises(ValueError) as exc:
        _render(_cond_line_with_points(**kwargs).add_selection(_COND_SEL), output)
    assert str(exc.value) == f"layer 0: {message}"


@pytest.mark.parametrize(
    ("build", "stamps"),
    [
        (lambda: _cond_point() + _plain_line(), [[0]]),
        (lambda: _plain_line() + _cond_point(), [[1]]),
        (lambda: _cond_line() + _plain_line() + _cond_point(), None),
        (lambda: _plain_line() + _cond_line(), [[1]]),
        (
            lambda: _base().layer(
                fm.Layer(
                    mark="line",
                    encoding={"x": fm.X("x"), "y": fm.Y("y"), "size": _cond_size(20)},
                )
            ),
            [[1]],
        ),
        (lambda: (_base() + _plain_line() + _base()).encode(size=_cond_size(0)), [[0, 1, 2]]),
        (
            lambda: (
                fm.Chart(_DF)
                .encode(x="x", y="y", size=_cond_size(20))
                .layer(fm.Layer(mark="line"), fm.Layer(mark="point"))
            ),
            [[0, 1, 2]],
        ),
        (lambda: _cond_line_with_points(), [[0, 1]]),
        # SecondaryY attaches a layer to its host, which shares its conditional
        # with that layer the way it shares a field `size`.
        (lambda: _cond_line() + fm.SecondaryY("v", mark="point"), [[0, 1]]),
        (lambda: _cond_line(), [None]),
    ],
    ids=[
        "point_plus_line",
        "line_plus_point",
        "two_size_conditionals_collide",
        "line_owns_it_on_the_right",
        "layer_encoding",
        "chart_level_on_layered_takes_every_layer",
        "markless_chart_level_takes_every_attached_layer",
        "line_point_composite_takes_both_layers",
        "secondary_y",
        "unlayered_whole_chart",
    ],
)
def test_every_layering_route_stamps_the_layers_that_declared_the_conditional(
    build: Any, stamps: list | None
) -> None:
    if stamps is None:
        # Control: scope does not lift the one-conditional-per-wire-channel rule.
        with pytest.raises(ValueError, match="both declare a conditional on the 'size'"):
            build()
        return
    assert _conditional_stamps(build()) == stamps


@pytest.mark.parametrize("layered", [False, True], ids=["unlayered", "layered"])
def test_a_line_encoding_its_own_conditional_size_carries_vertex_rows(layered: bool) -> None:
    chart = (_base() + _cond_line() if layered else _cond_line()).add_selection(_COND_SEL)
    batches = _scene_line_batches(chart)
    assert [b.get("vertex_rows") for b in batches] == [[[0, 1, 2, 3]]]
    assert re.findall(r'<polyline[^>]*\bstroke-width="([\d.]+)"', chart.to_svg()) == ["1.5"]


@pytest.mark.parametrize("output", ["to_svg", "interactive"])
@pytest.mark.parametrize("layered", [False, True], ids=["unlayered", "layered"])
def test_a_non_positive_branch_on_a_lines_own_conditional_size_refuses(
    layered: bool, output: str
) -> None:
    line = _cond_line(otherwise=0)
    chart = (_base() + line) if layered else line
    with pytest.raises(ValueError) as exc:
        _render(chart.add_selection(_COND_SEL), output)
    assert str(exc.value) == (f"layer 1: {_COND_BRANCH_ZERO}" if layered else _COND_BRANCH_ZERO)


@pytest.mark.parametrize("output", ["to_svg", "interactive"])
@pytest.mark.parametrize("layered", [False, True], ids=["unlayered", "layered"])
def test_a_conditional_size_plus_stroke_width_on_one_line_refuses(
    layered: bool, output: str
) -> None:
    line = _cond_line(stroke_width=3)
    chart = (_base() + line) if layered else line
    with pytest.raises(ValueError) as exc:
        _render(chart.add_selection(_COND_SEL), output)
    assert str(exc.value) == (f"layer 1: {_COND_CONFLICT}" if layered else _COND_CONFLICT)
