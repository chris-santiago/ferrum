"""`#120`(9) + N1: a layer column that cannot serve the shared continuous axis is refused.

Precondition to a continuous shared-axis union: a present layer column's dtype
is numeric or temporal. A violation raises a typed `RenderError` (Display in
the `LayerCoordUnplaceable` voice): `layer {i}: column '{field}' ({dtype})
cannot be drawn on the shared {channel} axis, which is continuous — a layered
chart (fm.layer(...) / chart + chart) merges its members onto one panel. Cast
the column, or give the layer its own scale with resolve={"{channel}":
"independent"}.` An absent field contributes nothing and is unaffected; every
other union error still propagates.

One predicate is reached from both the static composite-overlay route
(`fm.layer(...)`, each member its own leaf, checked where the Overlay group's
shared positional scale is reconciled) and the merged-flat route (`+`,
`.interactive()`). **The establishing member (layer 0 / the group's first
leaf) decides whether the shared axis is continuous**: `fm.layer(numeric,
categorical)` refuses naming layer 1, while `fm.layer(categorical, numeric)`
takes the ordinal union instead (the later numeric column bands onto the
ordinal ladder, unaffected by this predicate). A member whose own `CoordFlip`
differs from the panel's is judged by the coord-exchange gate instead of this
one.

The two routes differ in exactly one respect: on the composite route the text
reaches the caller inside the crate's general composite envelope —
`composite scale resolution failed: failed to resolve shared {channel} domain
for field '{field}': ` followed by the text above, the shape every leaf
render error takes on that route (including a `fm.layer(...)` nested inside
another composite, e.g. `fm.hconcat`) — while the merged-flat route (`+`,
`.interactive()`) raises the bare text directly.
"""

from __future__ import annotations

import datetime
import hashlib
import itertools
import warnings

import polars as pl
import pytest

import ferrum as fm

from _shared_axis_refusal import composite_wrapped_refusal, refusal
from _svg_extents import assert_bars_start_on_the_zero_tick

LEFT = pl.DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})

_DATES = [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2), datetime.date(2020, 1, 3)]
_DATETIMES = [
    datetime.datetime(2020, 1, 1),
    datetime.datetime(2020, 1, 2),
    datetime.datetime(2020, 1, 3),
]

# ---------------------------------------------------------------------------
# The generated population: route x layer-column dtype, under the shared axis.
# `(right-layer frame, y field or None, dtype Display text or None, refuses?)`.
# `dtype` is filled in only for the one case that needs it in an assertion
# (`Utf8`); the render-only cells never read it.
#
# `absent` binds no `y` field on the right layer at all: `layer_fields[0]` for
# this channel is `None`, so the cross-layer union's per-field loop `continue`s
# before ever looking the field up -- the same "this layer contributes
# nothing" outcome the union's own internal `Ok(false)` answers for a
# DECLARED field found in no batch. It is NOT the same construction as
# `test_a_declared_but_wholly_absent_field_refuses_on_every_route` below,
# which declares `y="v2"` against a frame that lacks it and hits an earlier,
# unrelated per-layer field-EXISTENCE check (a different code path, a
# different message) before the union ever runs.
# ---------------------------------------------------------------------------
_DTYPE_CASES: dict[str, tuple[pl.DataFrame, str | None, str | None, bool]] = {
    "Utf8": (pl.DataFrame({"x2": [1, 2, 3], "s": ["a", "b", "c"]}), "s", "Utf8", True),
    "Int64": (pl.DataFrame({"x2": [1, 2, 3], "i": [10, 20, 30]}), "i", "Int64", False),
    "Float64": (pl.DataFrame({"x2": [1, 2, 3], "f": [10.0, 20.0, 30.0]}), "f", "Float64", False),
    "Date32": (pl.DataFrame({"x2": [1, 2, 3], "d": _DATES}), "d", "Date32", False),
    "Datetime": (pl.DataFrame({"x2": [1, 2, 3], "dt": _DATETIMES}), "dt", None, False),
    "absent": (pl.DataFrame({"x2": [1, 2, 3]}), None, None, False),
}

ROUTES = ("fm.layer", "+", "interactive")


def _right_chart(df: pl.DataFrame, field: str | None) -> fm.Chart:
    if field is None:
        # `mark_line` requires both positional channels even as a composite
        # leaf; `mark_rule` draws from one, which is what "binds no y field
        # at all" needs to construct.
        return fm.Chart(df).mark_rule().encode(x="x2")
    return fm.Chart(df).mark_line().encode(x="x2", y=field)


def _render_pair(route: str, left: fm.Chart, right: fm.Chart) -> None:
    if route == "fm.layer":
        fm.layer(left, right).to_svg()
    elif route == "+":
        (left + right).to_svg()
    else:
        fm.layer(left, right).interactive()


def _render(route: str, dtype_label: str) -> None:
    df, field, _dtype, _refuses = _DTYPE_CASES[dtype_label]
    left = fm.Chart(LEFT).mark_point().encode(x="x", y="y")
    right = _right_chart(df, field)
    _render_pair(route, left, right)


def _expected_refusal(route: str, layer_index: int, field: str, dtype: str) -> str:
    if route == "fm.layer":
        return composite_wrapped_refusal(layer_index, field, dtype)
    return refusal(layer_index, field, dtype)


@pytest.mark.parametrize(
    "route,dtype_label",
    list(itertools.product(ROUTES, sorted(_DTYPE_CASES))),
)
def test_shared_axis_matrix(route: str, dtype_label: str) -> None:
    _df, field, dtype, refuses = _DTYPE_CASES[dtype_label]
    if refuses:
        with pytest.raises(ValueError) as exc:
            _render(route, dtype_label)
        assert str(exc.value) == _expected_refusal(route, 1, field, dtype)
    else:
        _render(route, dtype_label)  # must not raise


def test_nested_overlay_inside_hconcat_refuses_with_the_same_wrapped_text() -> None:
    """`fm.layer(a, b)` nested as an `hconcat` cell: the same composite-route
    envelope applies, with no extra wrapping layer from the enclosing grid."""
    left = fm.Chart(LEFT).mark_point().encode(x="x", y="y")
    df, field, dtype, _refuses = _DTYPE_CASES["Utf8"]
    right = _right_chart(df, field)
    third = (
        fm.Chart(pl.DataFrame({"c": [1, 2, 3], "d": [1.0, 2.0, 3.0]}))
        .mark_point()
        .encode(x="c", y="d")
    )
    with pytest.raises(ValueError) as exc:
        fm.hconcat(fm.layer(left, right), third).to_svg()
    assert str(exc.value) == composite_wrapped_refusal(1, field, dtype)


# ---------------------------------------------------------------------------
# The establishing member: layer 0 (the group's first leaf / the chart-level
# encoding's own field) decides whether the shared axis is continuous or
# ordinal, on every route. A numeric layer 0 makes the axis continuous, so a
# `Utf8` layer 1 refuses; a `Utf8` layer 0 makes it ordinal instead, so a
# numeric layer 1 bands onto the ordinal ladder and renders (F2).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
def test_a_numeric_layer_0_makes_the_axis_continuous_and_a_later_utf8_layer_refuses(
    route: str,
) -> None:
    numeric = fm.Chart(LEFT).mark_point().encode(x="x", y="y")
    df, field, dtype, _refuses = _DTYPE_CASES["Utf8"]
    utf8 = _right_chart(df, field)
    with pytest.raises(ValueError) as exc:
        _render_pair(route, numeric, utf8)
    assert str(exc.value) == _expected_refusal(route, 1, field, dtype)


@pytest.mark.parametrize("route", ROUTES)
def test_a_utf8_layer_0_makes_the_axis_ordinal_and_a_later_numeric_layer_renders(
    route: str,
) -> None:
    df, field, _dtype, _refuses = _DTYPE_CASES["Utf8"]
    utf8 = _right_chart(df, field)
    numeric = fm.Chart(LEFT).mark_point().encode(x="x", y="y")
    _render_pair(route, utf8, numeric)  # must not raise


# ---------------------------------------------------------------------------
# The "field absent" column, exercised as DECLARED rather than sidestepped:
# the right layer's own `y` names a column present in no frame at all. Every
# route refuses this -- each layer's own positional field is validated
# against its own leaf's data before the cross-layer union ever runs, so this
# is a real, pre-existing, unrelated-to-N1 "unknown column" refusal, not the
# union's own `Ok(false)` "found nowhere, contributes nothing" outcome (see
# the `_DTYPE_CASES` comment above for where that outcome actually lives).
# ---------------------------------------------------------------------------

_ABSENT_DECLARED_MESSAGES = {
    "fm.layer": "failed to render composite leaf leaf #1: unknown column 'v2' referenced by an encoding",
    "+": "layer 1: encoding 'y' references unknown column 'v2' (not present in that layer's data)",
    "interactive": "layer 1: encoding 'y' references unknown column 'v2' (not present in that layer's data)",
}


def _render_absent_declared(route: str) -> None:
    left = fm.Chart(LEFT).mark_point().encode(x="x", y="y")
    right = fm.Chart(pl.DataFrame({"x2": [1, 2, 3]})).mark_line().encode(x="x2", y="v2")
    _render_pair(route, left, right)


@pytest.mark.parametrize("route", ROUTES)
def test_a_declared_but_wholly_absent_field_refuses_on_every_route(route: str) -> None:
    with pytest.raises(ValueError) as exc:
        _render_absent_declared(route)
    assert str(exc.value) == _ABSENT_DECLARED_MESSAGES[route]


# ---------------------------------------------------------------------------
# Wire-level pin of the exact `LayerAxisColumnUnplaceable` text, and its
# render-control: a hand-built `ChartSpec.from_json` carrying a `layers` array
# directly, rendered through `render_svg`/`render_interactive` (bypassing
# `LayerChart` lowering entirely), mirroring `test_layer_transforms_refusal.py`'s
# wire-door pair.
#
# The control uses field `missing`, which is absent from the frame and from
# every transform output. It ALSO refuses ("unknown column ... not present"),
# not the union's own internal `Ok(false)` -- that specific branch answers
# "found in no batch" only for a field a NAMED transform routes to a
# DIFFERENT layer's `data_source` and is unreachable from a spec with no
# named transforms at all (every construction tried here hits the earlier,
# unrelated per-layer field-existence check first); reaching it would need a
# composite-mark-shaped `data_source`/transform-output fixture, out of scope
# for this cell.
# ---------------------------------------------------------------------------

_WIRE_DF = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "s": ["p", "q"]})

_WIRE_UTF8_SPEC = (
    '{"data": {"kind": "named", "name": "default"}, "mark": "point", '
    '"encoding": {"x": {"field": "b"}, "y": {"field": "a"}}, '
    '"layers": [{"mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}}, '
    '{"mark": "line", "encoding": {"y": {"field": "s"}}}]}'
)
_WIRE_ABSENT_SPEC = (
    '{"data": {"kind": "named", "name": "default"}, "mark": "point", '
    '"encoding": {"x": {"field": "b"}, "y": {"field": "a"}}, '
    '"layers": [{"mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}}, '
    '{"mark": "line", "encoding": {"y": {"field": "missing"}}}]}'
)


def _render_wire(spec_json: str, output_kind: str) -> None:
    from ferrum import ChartSpec
    from ferrum._core import render_interactive, render_svg

    spec = ChartSpec.from_json(spec_json)
    renderer = render_svg if output_kind == "static" else render_interactive
    renderer(spec, _WIRE_DF.to_arrow(), viewport=(400.0, 300.0))


@pytest.mark.parametrize("output_kind", ["static", "interactive"])
def test_wire_layer_utf8_column_refuses_with_the_exact_text(output_kind: str) -> None:
    with pytest.raises(ValueError) as exc:
        _render_wire(_WIRE_UTF8_SPEC, output_kind)
    assert str(exc.value) == refusal(1, "s", "Utf8")


@pytest.mark.parametrize("output_kind", ["static", "interactive"])
def test_wire_layer_absent_field_refuses_with_the_unknown_column_text(output_kind: str) -> None:
    """Vacuity-adjacent control: swapping `s` for a field absent everywhere
    changes the refusal text, proving the Utf8 pin above is not satisfied by
    any `ValueError` at all."""
    with pytest.raises(ValueError) as exc:
        _render_wire(_WIRE_ABSENT_SPEC, output_kind)
    assert str(exc.value) == (
        "layer 1: encoding 'y' references unknown column 'missing' (not present in that "
        "layer's data)"
    )


# ---------------------------------------------------------------------------
# `resolve={"y": "independent"}`: the union is never reached, on any route,
# for any dtype -- an independent-y layer resolves its own y slot and never
# competes for the shared one, so the establishing-member rule above does not
# apply to it either. Generated over the same route x dtype product as the
# shared-axis matrix.
#
# The `+` route reaches independent-y through `SecondaryY`, the only
# mechanism that spelling has; `SecondaryY` reads its field from the base
# chart's own table (no data merge), so `BASE_ALL` carries every dtype's
# column on one frame, unlike the shared-axis cases' separate `x2`-keyed
# right frame. `SecondaryY`'s own field-existence check (unrelated to this
# predicate) refuses the `absent` cell on this route alone -- every other
# cell, on every route, renders.
# ---------------------------------------------------------------------------

BASE_ALL = pl.DataFrame(
    {
        "x": [1, 2, 3],
        "y": [1.0, 2.0, 3.0],
        "s": ["a", "b", "c"],
        "i": [10, 20, 30],
        "f": [10.0, 20.0, 30.0],
        "d": _DATES,
        "dt": _DATETIMES,
    }
)

_INDEPENDENT_PLUS_FIELD_BY_DTYPE = {
    "Utf8": "s",
    "Int64": "i",
    "Float64": "f",
    "Date32": "d",
    "Datetime": "dt",
    "absent": "missing",
}

_INDEPENDENT_PLUS_ABSENT_REFUSAL = (
    "SecondaryY: field 'missing' is not a column of the base chart's data (SecondaryY draws "
    f"from the same table as the base chart); available columns: {BASE_ALL.columns!r}"
)


def _independent_render(route: str, dtype_label: str) -> None:
    if route == "+":
        field = _INDEPENDENT_PLUS_FIELD_BY_DTYPE[dtype_label]
        chart = fm.Chart(BASE_ALL).mark_point().encode(x="x", y="y")
        (chart + fm.SecondaryY(field=field, mark="line")).to_svg()
        return
    df, field, _dtype, _refuses = _DTYPE_CASES[dtype_label]
    left = fm.Chart(LEFT).mark_point().encode(x="x", y="y")
    right = _right_chart(df, field)
    chart = fm.layer(left, right, resolve={"y": "independent"})
    if route == "fm.layer":
        chart.to_svg()
    else:
        chart.interactive()


@pytest.mark.parametrize(
    "route,dtype_label",
    list(itertools.product(ROUTES, sorted(_DTYPE_CASES))),
)
def test_independent_y_matrix(route: str, dtype_label: str) -> None:
    if route == "+" and dtype_label == "absent":
        with pytest.raises(ValueError) as exc:
            _independent_render(route, dtype_label)
        assert str(exc.value) == _INDEPENDENT_PLUS_ABSENT_REFUSAL
    else:
        _independent_render(route, dtype_label)  # must not raise


# ---------------------------------------------------------------------------
# Vacuity control: changing ONLY the dtype flips the outcome. Guards against a
# harness whose `_render` helper silently swallows the render call, or whose
# refusal assertion is satisfied by any `ValueError`.
# ---------------------------------------------------------------------------


def test_the_matrix_discriminates_refusal_from_render() -> None:
    with pytest.raises(ValueError):
        _render("fm.layer", "Utf8")
    _render("fm.layer", "Int64")  # the same route, one dtype swapped, must not raise


# ---------------------------------------------------------------------------
# Must-render controls (`test_layer_ordinal_domain_union.py`): a continuous-
# axis union is a separate rule from the ordinal union's own, already-
# correct "skip a layer column the ladder cannot enumerate" rule one
# function over, and must not touch it.
#
# Each hash below pins one route's rendered bytes. For
# "categorical_bar_temporal_line", the `fm.layer` route renders with
# `padding.auto`'s default: the band the x axis reserves is sized by the
# same adjacent-gap measure the axis itself draws with, not a more
# conservative estimate, so the merged panel's height reflects the band the
# three flat category labels actually need. The `+` and `interactive`
# routes for the same fixture, and
# both routes for "integer_category_ordinal_union", render unaffected by
# that reservation. These hashes guard the ordinal-union rule against the
# continuous-axis predicate; they are unrelated to it, and must not move on
# their own. Each digest encodes two changes: the ordinal-union rule leaving
# the route untouched, and the bar layer's bars grown from the 0 tick, which
# `test_ordinal_union_control_bars_start_on_the_zero_tick` asserts directly
# for the two static routes.
# ---------------------------------------------------------------------------

_ORDINAL_UNION_CONTROL_HASHES = {
    "categorical_bar_temporal_line": {
        "fm.layer": "9f112fc7ec2f70f0",
        "+": "1a8ac40fe0c24787",
        "interactive": "374f5ba1bc1ce13f",
    },
    "integer_category_ordinal_union": {
        "fm.layer": "b8cc832a7a444d3e",
        "+": "c1439d743c0b264a",
        # Each operand's marks index its own rows.
        "interactive": "ebdf704a9eaf0c41",
    },
}


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _control_hash(route: str, a: fm.Chart, b: fm.Chart) -> str:
    if route == "fm.layer":
        return _sha16(fm.layer(a, b).to_svg())
    if route == "+":
        return _sha16((a + b).to_svg())
    return _sha16(fm.layer(a, b).interactive().scene_json)


def _ordinal_union_control_charts() -> dict[str, tuple[fm.Chart, fm.Chart]]:
    """The two ordinal-union control fixtures below, as ``(bar, other)`` pairs."""
    cat = pl.DataFrame({"cat": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    stamps = pl.DataFrame(
        {
            "when": [datetime.datetime(2020, month, 1) for month in (1, 2, 3)],
            "v2": [4.0, 5.0, 6.0],
        }
    )
    numeric = pl.DataFrame({"n": [10, 20], "v2": [4.0, 5.0]})
    bar = fm.Chart(cat).mark_bar().encode(x=fm.X("cat", type_="N"), y="v")
    return {
        "categorical_bar_temporal_line": (
            bar,
            fm.Chart(stamps).mark_line().encode(x=fm.X("when", type_="T"), y="v2"),
        ),
        "integer_category_ordinal_union": (
            bar,
            fm.Chart(numeric).mark_point().encode(x="n", y="v2"),
        ),
    }


@pytest.mark.parametrize("fixture", sorted(_ORDINAL_UNION_CONTROL_HASHES))
@pytest.mark.parametrize("route", ["fm.layer", "+"])
def test_ordinal_union_control_bars_start_on_the_zero_tick(fixture: str, route: str) -> None:
    """The geometry the control digests pin: the bar layer grows from 0."""
    bar, other = _ordinal_union_control_charts()[fixture]
    svg = fm.layer(bar, other).to_svg() if route == "fm.layer" else (bar + other).to_svg()
    assert_bars_start_on_the_zero_tick(svg)


@pytest.mark.parametrize("route", ROUTES)
def test_categorical_x_bar_plus_temporal_x_line_is_byte_identical(route: str) -> None:
    """A layer column the ordinal ladder cannot enumerate (`Timestamp`) beside a
    categorical-x bar layer: skipped, not refused -- unrelated to the
    continuous-axis predicate under test, and must not move."""
    bar, line = _ordinal_union_control_charts()["categorical_bar_temporal_line"]
    assert (
        _control_hash(route, bar, line)
        == _ORDINAL_UNION_CONTROL_HASHES["categorical_bar_temporal_line"][route]
    )


@pytest.mark.parametrize("route", ROUTES)
def test_integer_category_columns_under_the_ordinal_union_is_byte_identical(route: str) -> None:
    """A quantitative layer column beside a categorical-x bar layer: bands and
    draws on the ordinal ladder (F2). Also unrelated to the continuous-axis
    predicate under test, and must not move."""
    bar, point = _ordinal_union_control_charts()["integer_category_ordinal_union"]
    assert (
        _control_hash(route, bar, point)
        == _ORDINAL_UNION_CONTROL_HASHES["integer_category_ordinal_union"][route]
    )


# ---------------------------------------------------------------------------
# A temporal date layer beside a TEXT category axis keeps every outcome it had.
#
# A date layer beside a text category axis is not a date category axis: the
# date stays a time value (a later layer's undeclared, or ``:T``, date column is
# never read as categories there), so the whole table below is the v0.25.0
# outcome on every mark, declaration, door and output: a recorded digest of the
# svg (or the scene) with its ordered warnings, or the refusal's full text.
# Those include v0.25.0's own route split (a rule refuses on ``+``,
# ``Chart.layer`` and both interactive overlays and renders on the static
# overlay) and its silent empty draws. Measured on the v0.25.0 build.
# ---------------------------------------------------------------------------

_BESIDE_STAMPS = pl.DataFrame(
    {
        "when": [datetime.datetime(2020, month, 1) for month in (1, 2, 3)],
        "v2": [4.0, 5.0, 6.0],
        "v3": [5.0, 6.0, 7.0],
        "lab": ["p", "q", "r"],
    }
)
_BESIDE_CAT = pl.DataFrame({"cat": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})

# mark -> the encoding its date layer writes beside the text axis (``{x}`` is the x binding)
_BESIDE_MARKS: dict[str, dict[str, object]] = {
    "line": {"y": "v2"},
    "point": {"y": "v2"},
    "bar": {"y": "v2"},
    "area": {"y": "v2"},
    "tick": {},
    "text": {"y": "v2", "text": "lab"},
    "rect": {"y": "v2", "y2": "v3"},
    "rule": {},
}
_BESIDE_X = {"undeclared": lambda: "when", "T": lambda: fm.X("when", type_="T")}
_BESIDE_DOORS = ["+", "Chart.layer", "fm.layer", "LayerChart"]
_BESIDE_OUTPUTS = ["static", "interactive"]


def _beside_chart(mark: str, declared: str, door: str):
    bar = fm.Chart(_BESIDE_CAT).mark_bar().encode(x=fm.X("cat", type_="N"), y="v")
    encoding = {"x": _BESIDE_X[declared](), **_BESIDE_MARKS[mark]}
    later = getattr(fm.Chart(_BESIDE_STAMPS), f"mark_{mark}")().encode(**encoding)
    if door == "+":
        return bar + later
    if door == "Chart.layer":
        return bar.layer(fm.Layer(mark=mark, data=_BESIDE_STAMPS, encoding=encoding))
    if door == "fm.layer":
        return fm.layer(bar, later)
    return fm.LayerChart(bar, later)


def _beside_outcome(mark: str, declared: str, door: str, output: str) -> tuple:
    """``("refused", full text)`` or ``("drawn", digest of the output, ordered warnings)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            chart = _beside_chart(mark, declared, door)
            out = chart.to_svg() if output == "static" else chart.interactive().scene_json
        except ValueError as refusal_:
            return ("refused", str(refusal_))
    return ("drawn", _sha16(out), tuple(str(w.message) for w in caught))


_BESIDE_AT_V025: dict[tuple[str, str, str, str], tuple] = {
    ("area", "T", "+", "static"): (
        "drawn",
        "1a8ac40fe0c24787",
        (
            "x scale: 3 row(s) of 'when' fall outside the scale's explicit "
            "domain, so they are not drawn; widen the domain, or pass "
            "clamp=True to draw them at the axis edge",
        ),
    ),
    ("area", "T", "+", "interactive"): (
        "drawn",
        "54765f470858906e",
        (
            "x scale: 3 row(s) of 'when' fall outside the scale's "
            "explicit domain, so they are not drawn; widen the domain, "
            "or pass clamp=True to draw them at the axis edge",
        ),
    ),
    ("area", "T", "Chart.layer", "static"): (
        "drawn",
        "1a8ac40fe0c24787",
        (
            "x scale: 3 row(s) of 'when' fall outside the scale's "
            "explicit domain, so they are not drawn; widen the "
            "domain, or pass clamp=True to draw them at the axis "
            "edge",
        ),
    ),
    ("area", "T", "Chart.layer", "interactive"): (
        "drawn",
        "54765f470858906e",
        (
            "x scale: 3 row(s) of 'when' fall outside the "
            "scale's explicit domain, so they are not drawn; "
            "widen the domain, or pass clamp=True to draw them "
            "at the axis edge",
        ),
    ),
    ("area", "T", "fm.layer", "static"): ("drawn", "19211423d0f5b8c4", ()),
    ("area", "T", "fm.layer", "interactive"): (
        "drawn",
        "54765f470858906e",
        (
            "x scale: 3 row(s) of 'when' fall outside the scale's "
            "explicit domain, so they are not drawn; widen the "
            "domain, or pass clamp=True to draw them at the axis "
            "edge",
        ),
    ),
    ("area", "T", "LayerChart", "static"): ("drawn", "19211423d0f5b8c4", ()),
    ("area", "T", "LayerChart", "interactive"): (
        "drawn",
        "54765f470858906e",
        (
            "x scale: 3 row(s) of 'when' fall outside the "
            "scale's explicit domain, so they are not drawn; "
            "widen the domain, or pass clamp=True to draw them "
            "at the axis edge",
        ),
    ),
    ("area", "undeclared", "+", "static"): (
        "drawn",
        "1a8ac40fe0c24787",
        (
            "x scale: 3 row(s) of 'when' fall outside the scale's "
            "explicit domain, so they are not drawn; widen the "
            "domain, or pass clamp=True to draw them at the axis "
            "edge",
        ),
    ),
    ("area", "undeclared", "+", "interactive"): (
        "drawn",
        "54765f470858906e",
        (
            "x scale: 3 row(s) of 'when' fall outside the "
            "scale's explicit domain, so they are not drawn; "
            "widen the domain, or pass clamp=True to draw them "
            "at the axis edge",
        ),
    ),
    ("area", "undeclared", "Chart.layer", "static"): (
        "drawn",
        "1a8ac40fe0c24787",
        (
            "x scale: 3 row(s) of 'when' fall outside the "
            "scale's explicit domain, so they are not "
            "drawn; widen the domain, or pass clamp=True "
            "to draw them at the axis edge",
        ),
    ),
    ("area", "undeclared", "Chart.layer", "interactive"): (
        "drawn",
        "54765f470858906e",
        (
            "x scale: 3 row(s) of 'when' fall outside "
            "the scale's explicit domain, so they are "
            "not drawn; widen the domain, or pass "
            "clamp=True to draw them at the axis "
            "edge",
        ),
    ),
    ("area", "undeclared", "fm.layer", "static"): ("drawn", "19211423d0f5b8c4", ()),
    ("area", "undeclared", "fm.layer", "interactive"): (
        "drawn",
        "54765f470858906e",
        (
            "x scale: 3 row(s) of 'when' fall outside "
            "the scale's explicit domain, so they are "
            "not drawn; widen the domain, or pass "
            "clamp=True to draw them at the axis edge",
        ),
    ),
    ("area", "undeclared", "LayerChart", "static"): ("drawn", "19211423d0f5b8c4", ()),
    ("area", "undeclared", "LayerChart", "interactive"): (
        "drawn",
        "54765f470858906e",
        (
            "x scale: 3 row(s) of 'when' fall outside "
            "the scale's explicit domain, so they are "
            "not drawn; widen the domain, or pass "
            "clamp=True to draw them at the axis "
            "edge",
        ),
    ),
    ("bar", "T", "+", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("bar", "T", "+", "interactive"): ("drawn", "a1eba7e11319a68e", ()),
    ("bar", "T", "Chart.layer", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("bar", "T", "Chart.layer", "interactive"): ("drawn", "a1eba7e11319a68e", ()),
    ("bar", "T", "fm.layer", "static"): ("drawn", "1c19c5df157d891b", ()),
    ("bar", "T", "fm.layer", "interactive"): ("drawn", "a1eba7e11319a68e", ()),
    ("bar", "T", "LayerChart", "static"): ("drawn", "1c19c5df157d891b", ()),
    ("bar", "T", "LayerChart", "interactive"): ("drawn", "a1eba7e11319a68e", ()),
    ("bar", "undeclared", "+", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("bar", "undeclared", "+", "interactive"): ("drawn", "a1eba7e11319a68e", ()),
    ("bar", "undeclared", "Chart.layer", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("bar", "undeclared", "Chart.layer", "interactive"): ("drawn", "a1eba7e11319a68e", ()),
    ("bar", "undeclared", "fm.layer", "static"): ("drawn", "1c19c5df157d891b", ()),
    ("bar", "undeclared", "fm.layer", "interactive"): ("drawn", "a1eba7e11319a68e", ()),
    ("bar", "undeclared", "LayerChart", "static"): ("drawn", "1c19c5df157d891b", ()),
    ("bar", "undeclared", "LayerChart", "interactive"): ("drawn", "a1eba7e11319a68e", ()),
    ("line", "T", "+", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("line", "T", "+", "interactive"): ("drawn", "374f5ba1bc1ce13f", ()),
    ("line", "T", "Chart.layer", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("line", "T", "Chart.layer", "interactive"): ("drawn", "374f5ba1bc1ce13f", ()),
    ("line", "T", "fm.layer", "static"): ("drawn", "9f112fc7ec2f70f0", ()),
    ("line", "T", "fm.layer", "interactive"): ("drawn", "374f5ba1bc1ce13f", ()),
    ("line", "T", "LayerChart", "static"): ("drawn", "9f112fc7ec2f70f0", ()),
    ("line", "T", "LayerChart", "interactive"): ("drawn", "374f5ba1bc1ce13f", ()),
    ("line", "undeclared", "+", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("line", "undeclared", "+", "interactive"): ("drawn", "374f5ba1bc1ce13f", ()),
    ("line", "undeclared", "Chart.layer", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("line", "undeclared", "Chart.layer", "interactive"): ("drawn", "374f5ba1bc1ce13f", ()),
    ("line", "undeclared", "fm.layer", "static"): ("drawn", "9f112fc7ec2f70f0", ()),
    ("line", "undeclared", "fm.layer", "interactive"): ("drawn", "374f5ba1bc1ce13f", ()),
    ("line", "undeclared", "LayerChart", "static"): ("drawn", "9f112fc7ec2f70f0", ()),
    ("line", "undeclared", "LayerChart", "interactive"): ("drawn", "374f5ba1bc1ce13f", ()),
    ("point", "T", "+", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("point", "T", "+", "interactive"): ("drawn", "dea5975c7876446a", ()),
    ("point", "T", "Chart.layer", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("point", "T", "Chart.layer", "interactive"): ("drawn", "dea5975c7876446a", ()),
    ("point", "T", "fm.layer", "static"): ("drawn", "e32fb45e04ea221f", ()),
    ("point", "T", "fm.layer", "interactive"): ("drawn", "dea5975c7876446a", ()),
    ("point", "T", "LayerChart", "static"): ("drawn", "e32fb45e04ea221f", ()),
    ("point", "T", "LayerChart", "interactive"): ("drawn", "dea5975c7876446a", ()),
    ("point", "undeclared", "+", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("point", "undeclared", "+", "interactive"): ("drawn", "dea5975c7876446a", ()),
    ("point", "undeclared", "Chart.layer", "static"): ("drawn", "1a8ac40fe0c24787", ()),
    ("point", "undeclared", "Chart.layer", "interactive"): ("drawn", "dea5975c7876446a", ()),
    ("point", "undeclared", "fm.layer", "static"): ("drawn", "e32fb45e04ea221f", ()),
    ("point", "undeclared", "fm.layer", "interactive"): ("drawn", "dea5975c7876446a", ()),
    ("point", "undeclared", "LayerChart", "static"): ("drawn", "e32fb45e04ea221f", ()),
    ("point", "undeclared", "LayerChart", "interactive"): ("drawn", "dea5975c7876446a", ()),
    ("rect", "T", "+", "static"): ("drawn", "ebfa7c2d9d1d407a", ()),
    ("rect", "T", "+", "interactive"): ("drawn", "9aa5654c99b2fb9f", ()),
    ("rect", "T", "Chart.layer", "static"): ("drawn", "ebfa7c2d9d1d407a", ()),
    ("rect", "T", "Chart.layer", "interactive"): ("drawn", "9aa5654c99b2fb9f", ()),
    ("rect", "T", "fm.layer", "static"): (
        "drawn",
        "9dc80c4da0d9303c",
        (
            "rect layer 0 drew nothing: mark_rect() needs an ordinal x "
            "or y axis to size its band-relative width, but the "
            "resolved scale is continuous",
        ),
    ),
    ("rect", "T", "fm.layer", "interactive"): ("drawn", "9aa5654c99b2fb9f", ()),
    ("rect", "T", "LayerChart", "static"): (
        "drawn",
        "9dc80c4da0d9303c",
        (
            "rect layer 0 drew nothing: mark_rect() needs an ordinal "
            "x or y axis to size its band-relative width, but the "
            "resolved scale is continuous",
        ),
    ),
    ("rect", "T", "LayerChart", "interactive"): ("drawn", "9aa5654c99b2fb9f", ()),
    ("rect", "undeclared", "+", "static"): ("drawn", "ebfa7c2d9d1d407a", ()),
    ("rect", "undeclared", "+", "interactive"): ("drawn", "9aa5654c99b2fb9f", ()),
    ("rect", "undeclared", "Chart.layer", "static"): ("drawn", "ebfa7c2d9d1d407a", ()),
    ("rect", "undeclared", "Chart.layer", "interactive"): ("drawn", "9aa5654c99b2fb9f", ()),
    ("rect", "undeclared", "fm.layer", "static"): (
        "drawn",
        "9dc80c4da0d9303c",
        (
            "rect layer 0 drew nothing: mark_rect() needs an "
            "ordinal x or y axis to size its band-relative "
            "width, but the resolved scale is continuous",
        ),
    ),
    ("rect", "undeclared", "fm.layer", "interactive"): ("drawn", "9aa5654c99b2fb9f", ()),
    ("rect", "undeclared", "LayerChart", "static"): (
        "drawn",
        "9dc80c4da0d9303c",
        (
            "rect layer 0 drew nothing: mark_rect() needs "
            "an ordinal x or y axis to size its "
            "band-relative width, but the resolved scale is "
            "continuous",
        ),
    ),
    ("rect", "undeclared", "LayerChart", "interactive"): ("drawn", "9aa5654c99b2fb9f", ()),
    ("rule", "T", "+", "static"): (
        "refused",
        "layer 1: mark_rule: column 'when' has unsupported dtype: "
        "Timestamp(Millisecond, None) (cannot convert to ordinal category "
        "string)",
    ),
    ("rule", "T", "+", "interactive"): (
        "refused",
        "layer 1: mark_rule: column 'when' has unsupported dtype: "
        "Timestamp(Millisecond, None) (cannot convert to ordinal "
        "category string)",
    ),
    ("rule", "T", "Chart.layer", "static"): (
        "refused",
        "layer 1: mark_rule: column 'when' has unsupported "
        "dtype: Timestamp(Millisecond, None) (cannot convert to "
        "ordinal category string)",
    ),
    ("rule", "T", "Chart.layer", "interactive"): (
        "refused",
        "layer 1: mark_rule: column 'when' has unsupported "
        "dtype: Timestamp(Millisecond, None) (cannot "
        "convert to ordinal category string)",
    ),
    ("rule", "T", "fm.layer", "static"): ("drawn", "82bd7514e404d178", ()),
    ("rule", "T", "fm.layer", "interactive"): (
        "refused",
        "layer 1: mark_rule: column 'when' has unsupported "
        "dtype: Timestamp(Millisecond, None) (cannot convert "
        "to ordinal category string)",
    ),
    ("rule", "T", "LayerChart", "static"): ("drawn", "82bd7514e404d178", ()),
    ("rule", "T", "LayerChart", "interactive"): (
        "refused",
        "layer 1: mark_rule: column 'when' has unsupported "
        "dtype: Timestamp(Millisecond, None) (cannot convert "
        "to ordinal category string)",
    ),
    ("rule", "undeclared", "+", "static"): (
        "refused",
        "layer 1: mark_rule: column 'when' has unsupported dtype: "
        "Timestamp(Millisecond, None) (cannot convert to ordinal "
        "category string)",
    ),
    ("rule", "undeclared", "+", "interactive"): (
        "refused",
        "layer 1: mark_rule: column 'when' has unsupported "
        "dtype: Timestamp(Millisecond, None) (cannot convert "
        "to ordinal category string)",
    ),
    ("rule", "undeclared", "Chart.layer", "static"): (
        "refused",
        "layer 1: mark_rule: column 'when' has "
        "unsupported dtype: Timestamp(Millisecond, "
        "None) (cannot convert to ordinal category "
        "string)",
    ),
    ("rule", "undeclared", "Chart.layer", "interactive"): (
        "refused",
        "layer 1: mark_rule: column 'when' has "
        "unsupported dtype: Timestamp(Millisecond, "
        "None) (cannot convert to ordinal category "
        "string)",
    ),
    ("rule", "undeclared", "fm.layer", "static"): ("drawn", "82bd7514e404d178", ()),
    ("rule", "undeclared", "fm.layer", "interactive"): (
        "refused",
        "layer 1: mark_rule: column 'when' has "
        "unsupported dtype: Timestamp(Millisecond, "
        "None) (cannot convert to ordinal category "
        "string)",
    ),
    ("rule", "undeclared", "LayerChart", "static"): ("drawn", "82bd7514e404d178", ()),
    ("rule", "undeclared", "LayerChart", "interactive"): (
        "refused",
        "layer 1: mark_rule: column 'when' has "
        "unsupported dtype: Timestamp(Millisecond, "
        "None) (cannot convert to ordinal category "
        "string)",
    ),
    ("text", "T", "+", "static"): ("drawn", "0d8eb0b7e86cd5f5", ()),
    ("text", "T", "+", "interactive"): ("drawn", "190d8613aaede355", ()),
    ("text", "T", "Chart.layer", "static"): ("drawn", "0d8eb0b7e86cd5f5", ()),
    ("text", "T", "Chart.layer", "interactive"): ("drawn", "190d8613aaede355", ()),
    ("text", "T", "fm.layer", "static"): ("drawn", "ae61410c6236d75f", ()),
    ("text", "T", "fm.layer", "interactive"): ("drawn", "190d8613aaede355", ()),
    ("text", "T", "LayerChart", "static"): ("drawn", "ae61410c6236d75f", ()),
    ("text", "T", "LayerChart", "interactive"): ("drawn", "190d8613aaede355", ()),
    ("text", "undeclared", "+", "static"): ("drawn", "0d8eb0b7e86cd5f5", ()),
    ("text", "undeclared", "+", "interactive"): ("drawn", "190d8613aaede355", ()),
    ("text", "undeclared", "Chart.layer", "static"): ("drawn", "0d8eb0b7e86cd5f5", ()),
    ("text", "undeclared", "Chart.layer", "interactive"): ("drawn", "190d8613aaede355", ()),
    ("text", "undeclared", "fm.layer", "static"): ("drawn", "ae61410c6236d75f", ()),
    ("text", "undeclared", "fm.layer", "interactive"): ("drawn", "190d8613aaede355", ()),
    ("text", "undeclared", "LayerChart", "static"): ("drawn", "ae61410c6236d75f", ()),
    ("text", "undeclared", "LayerChart", "interactive"): ("drawn", "190d8613aaede355", ()),
    ("tick", "T", "+", "static"): ("drawn", "230f8a2223a2e55e", ()),
    ("tick", "T", "+", "interactive"): ("drawn", "03d3cec4d4872def", ()),
    ("tick", "T", "Chart.layer", "static"): ("drawn", "230f8a2223a2e55e", ()),
    ("tick", "T", "Chart.layer", "interactive"): ("drawn", "03d3cec4d4872def", ()),
    ("tick", "T", "fm.layer", "static"): ("drawn", "f148000c6632e781", ()),
    ("tick", "T", "fm.layer", "interactive"): ("drawn", "03d3cec4d4872def", ()),
    ("tick", "T", "LayerChart", "static"): ("drawn", "f148000c6632e781", ()),
    ("tick", "T", "LayerChart", "interactive"): ("drawn", "03d3cec4d4872def", ()),
    ("tick", "undeclared", "+", "static"): ("drawn", "230f8a2223a2e55e", ()),
    ("tick", "undeclared", "+", "interactive"): ("drawn", "03d3cec4d4872def", ()),
    ("tick", "undeclared", "Chart.layer", "static"): ("drawn", "230f8a2223a2e55e", ()),
    ("tick", "undeclared", "Chart.layer", "interactive"): ("drawn", "03d3cec4d4872def", ()),
    ("tick", "undeclared", "fm.layer", "static"): ("drawn", "f148000c6632e781", ()),
    ("tick", "undeclared", "fm.layer", "interactive"): ("drawn", "03d3cec4d4872def", ()),
    ("tick", "undeclared", "LayerChart", "static"): ("drawn", "f148000c6632e781", ()),
    ("tick", "undeclared", "LayerChart", "interactive"): ("drawn", "03d3cec4d4872def", ()),
}


@pytest.mark.parametrize("output", _BESIDE_OUTPUTS)
@pytest.mark.parametrize("door", _BESIDE_DOORS)
@pytest.mark.parametrize("declared", sorted(_BESIDE_X))
@pytest.mark.parametrize("mark", sorted(_BESIDE_MARKS))
def test_a_temporal_date_layer_beside_a_text_category_axis_matches_head(
    mark: str, declared: str, door: str, output: str
) -> None:
    assert (
        _beside_outcome(mark, declared, door, output)
        == _BESIDE_AT_V025[(mark, declared, door, output)]
    )
