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

Since a later layer's `size` promotes to the chart level
(`tests/test_layer_channel_promotion.py`), a line layer that binds only
its own `stroke_width` (field or constant) but sits beside a `size`-bound
layer elsewhere draws at its own `stroke_width`, untapered -- its own
binding outranks a `size` it only inherits, on `+` and `fm.layer(...)`, in
either layer order, static and interactive alike. A line that binds
neither itself and inherits both channels keeps the inherited
`stroke_width` and draws untapered too. The refusal still fires,
unaffected, when one line layer binds both channels itself.
"""

from __future__ import annotations

import json
import math
import re
import tempfile

import polars as pl
import pytest

import ferrum as fm

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
# A line layer's own `stroke_width` outranks a `size` it only inherits
# through auxiliary-channel promotion. `point(size="v") + line(stroke_width=3)`
# sizes the dots and draws the line at a flat, untapered 3px width, in
# either layer order, on both `+` and `fm.layer(...)`, whether the line's
# own `stroke_width` is a constant kwarg or its own field encoding.
# ---------------------------------------------------------------------------


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
    "plus_point_first_field": (lambda: _sized_point() + _own_stroke_width_line_field(), "1"),
    "plus_line_first_field": (lambda: _own_stroke_width_line_field() + _sized_point(), "1"),
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
def test_own_stroke_width_outranks_an_inherited_size(cell: str) -> None:
    build, expected_width = _MUST_DRAW_CELLS[cell]
    chart = build()

    # Static: one untapered polyline at the line's own stroke width.
    assert _svg_polyline_stroke_widths(chart.to_svg()) == [expected_width]

    # Interactive: no per-vertex `widths`, and the fallback width is the
    # line's own, not derived from the inherited size.
    nodes = _line_nodes(chart)
    assert len(nodes) == 1
    assert "widths" not in nodes[0]
    assert nodes[0]["style"]["width"] == pytest.approx(float(expected_width))

    # The dots are still sized by the inherited `size` -- promotion still
    # reaches the point layer; only the line's own width wins on the line.
    radii = _point_radii(chart)
    assert len(radii) == 4
    assert radii == sorted(radii) and radii[0] < radii[-1]


def test_a_line_inheriting_both_channels_draws_untapered_at_the_inherited_stroke_width() -> None:
    """Neither `size` nor `stroke_width` is bound on the third layer itself --
    both are inherited -- so it keeps the inherited `stroke_width` and draws
    untapered, the width it drew before `size` could taper a line, rather
    than refusing or tapering."""
    first_line = fm.Chart(_DF).mark_line().encode(x="x", y="y", stroke_width="w")
    sized_point = fm.Chart(_DF).mark_point().encode(x="x", y="y", size="v")
    bare_line = fm.Chart(_DF).mark_line().encode(x="x", y="y")

    chart = first_line + sized_point + bare_line
    nodes = _line_nodes(chart)
    assert len(nodes) == 2
    for node in nodes:
        assert "widths" not in node
        assert node["style"]["width"] == pytest.approx(1.0)  # 'w' is constant 1.0

    radii = _point_radii(chart)
    assert len(radii) == 4
    assert radii == sorted(radii) and radii[0] < radii[-1]

    assert _svg_polyline_stroke_widths(chart.to_svg()) == ["1", "1"]


def test_the_conflict_refusal_is_unaffected_by_promotion() -> None:
    """A line layer that binds both channels itself still refuses, even
    though the chart also carries an active `size` binding that promotion
    would otherwise reach -- precedence applies only to a channel the line
    *inherits*, never to a channel it binds itself."""
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
# A line layer's own constant `size=` or `stroke_width=` kwarg outranks a
# width field it only inherits -- `size` through promotion from another
# layer's own binding, `stroke_width` through the leading chart's own
# binding, since `stroke_width` itself is never promoted -- the
# kwarg-bound layer draws its own constant width, untapered, in either
# layer order, static and interactive alike.
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


def _inherited_size_only_point() -> fm.Chart:
    return fm.Chart(_OWN_KWARG_DF).mark_point().encode(x="x", y="y", size="v")


@pytest.mark.parametrize(
    "chart_factory",
    [
        lambda: _inherited_size_only_point() + _own_kwarg_sized_line(),
        lambda: _own_kwarg_sized_line() + _inherited_size_only_point(),
    ],
    ids=["point_first", "line_first"],
)
def test_own_size_kwarg_outranks_an_inherited_size_field(chart_factory, recwarn) -> None:
    """A line's own `size=` kwarg wins over a `size` field it only inherits
    through promotion from a sibling point layer, in either layer order."""
    control_width = 2 * math.sqrt(5 / math.pi)
    chart = chart_factory()

    nodes = _line_nodes(chart)
    assert len(nodes) == 1
    assert "widths" not in nodes[0]
    assert nodes[0]["style"]["width"] == pytest.approx(control_width)

    svg_widths = _svg_polyline_stroke_widths(chart.to_svg())
    assert len(svg_widths) == 1
    assert float(svg_widths[0]) == pytest.approx(control_width, abs=5e-4)

    # The dots are still sized by the inherited `size` -- only the line's own
    # kwarg wins on the line itself.
    radii = _point_radii(chart)
    assert len(radii) == 4
    assert radii == sorted(radii) and radii[0] < radii[-1]

    user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert not user_warnings, [str(w.message) for w in user_warnings]


def test_own_stroke_width_field_then_own_constant_draw_each_their_own_width(recwarn) -> None:
    """The field line (layer 0) binds `stroke_width='w'`,
    which becomes the leading chart's own binding -- the chart-level
    `stroke_width` every later layer inherits unless it binds its own. The
    constant line (layer 1) does bind its own, `stroke_width=3`, which
    outranks that inherited field, so it draws its own width instead of the
    inherited one. Each line ends up drawing its own width, in layer
    order, but only because the second line's own kwarg won the
    precedence check -- not because there was nothing to inherit."""
    field_line = fm.Chart(_OWN_KWARG_DF).mark_line().encode(x="x", y="y", stroke_width="w")
    const_line = fm.Chart(_OWN_KWARG_DF).mark_line(stroke_width=3).encode(x="x", y="y")

    chart = field_line + const_line
    assert _svg_polyline_stroke_widths(chart.to_svg()) == ["2", "3"]
    nodes = _line_nodes(chart)
    assert [n["style"]["width"] for n in nodes] == [2.0, 3.0]
    assert all("widths" not in n for n in nodes)

    user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert not user_warnings, [str(w.message) for w in user_warnings]


def test_own_constant_stroke_width_then_own_field_is_order_independent(recwarn) -> None:
    """Order-independence pin, not a precedence pin: this cell (the
    `stroke_width=` kwarg line first, the field line second) passes both
    before and after the fix, because `stroke_width` never promotes -- there
    is no inheritance for either layer's own binding to outrank."""
    const_line = fm.Chart(_OWN_KWARG_DF).mark_line(stroke_width=3).encode(x="x", y="y")
    field_line = fm.Chart(_OWN_KWARG_DF).mark_line().encode(x="x", y="y", stroke_width="w")

    chart = const_line + field_line
    assert _svg_polyline_stroke_widths(chart.to_svg()) == ["3", "2"]
    nodes = _line_nodes(chart)
    assert [n["style"]["width"] for n in nodes] == [3.0, 2.0]
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
