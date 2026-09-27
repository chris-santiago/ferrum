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

import polars as pl
import pytest

import ferrum as fm

from _shared_axis_refusal import composite_wrapped_refusal, refusal

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
# Wave-5 must-render controls (`test_layer_ordinal_domain_union.py`), pinned
# byte-identical on every route: the fix narrows a CONTINUOUS-axis union, and
# must not touch the ordinal union's own, already-correct, "skip a layer
# column the ladder cannot enumerate" rule one function over.
# ---------------------------------------------------------------------------

_WAVE5_CONTROL_HASHES = {
    "categorical_bar_temporal_line": {
        "fm.layer": "2446a242cabf896b",
        "+": "dddad057de646ff7",
        "interactive": "3248f9cb42c3bc79",
    },
    "integer_category_ordinal_union": {
        "fm.layer": "cb7dd96040391486",
        "+": "82230fef7158dad8",
        "interactive": "bf46421a6e9e7426",
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


@pytest.mark.parametrize("route", ROUTES)
def test_categorical_x_bar_plus_temporal_x_line_is_byte_identical(route: str) -> None:
    """A layer column the ordinal ladder cannot enumerate (`Timestamp`) beside a
    categorical-x bar layer: skipped, not refused -- unrelated to the
    continuous-axis predicate under test, and must not move."""
    cat = pl.DataFrame({"cat": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    stamps = pl.DataFrame(
        {
            "when": [
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2020, 2, 1),
                datetime.datetime(2020, 3, 1),
            ],
            "v2": [4.0, 5.0, 6.0],
        }
    )
    bar = fm.Chart(cat).mark_bar().encode(x=fm.X("cat", type_="N"), y="v")
    line = fm.Chart(stamps).mark_line().encode(x=fm.X("when", type_="T"), y="v2")
    assert (
        _control_hash(route, bar, line)
        == _WAVE5_CONTROL_HASHES["categorical_bar_temporal_line"][route]
    )


@pytest.mark.parametrize("route", ROUTES)
def test_integer_category_columns_under_the_ordinal_union_is_byte_identical(route: str) -> None:
    """A quantitative layer column beside a categorical-x bar layer: bands and
    draws on the ordinal ladder (F2). Also unrelated to the continuous-axis
    predicate under test, and must not move."""
    cat = pl.DataFrame({"cat": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    numeric = pl.DataFrame({"n": [10, 20], "v2": [4.0, 5.0]})
    bar = fm.Chart(cat).mark_bar().encode(x=fm.X("cat", type_="N"), y="v")
    point = fm.Chart(numeric).mark_point().encode(x="n", y="v2")
    assert (
        _control_hash(route, bar, point)
        == _WAVE5_CONTROL_HASHES["integer_category_ordinal_union"][route]
    )
