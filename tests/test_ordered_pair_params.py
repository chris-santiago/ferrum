"""Ordered (lo, hi) pairs are read one way on every pair parameter.

A list, tuple, ``range`` or 1-D numpy array of two real numbers is read in order. A set,
bytes, a mapping, a generator or any other kind is refused before it is read, so no door
reads a collection in hash order. A statistic range refuses a reversed or non-finite
pair; a coordinate limit and a reference-line endpoint pair keep a reversed pair and
refuse only a non-finite end.

Every door is a row of one table, so each boundary is pinned on every door: the Python
readers (``transform_density``, the coordinate limits, ``jointplot``) and the compiled
classes (``Kde``, ``Bin``, ``Bin2D``, ``Kde2D``, ``Robust``, ``Smooth``,
``ReferenceLine``) with the mark doors that forward to them. Every refusal is compared
whole, with ``==``.
"""

from __future__ import annotations

import dataclasses
import decimal
import fractions
import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

import ferrum as fm

_SET_SUFFIX = "; a set has no order, so the ends would change from run to run"
_BYTES_SUFFIX = "; iterating it yields integers"
_NAN = float("nan")
_INF = float("inf")
_HUGE = 10**400
_DEC = decimal.Decimal
_FRAC = fractions.Fraction


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "v": [float(i) / 4 for i in range(41)],
            "w": [float((i * 7) % 11) for i in range(41)],
        }
    )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --- the texts ---------------------------------------------------------------------------


def _shape(voice: str, param: str, value: object, suffix: str = "") -> str:
    return (
        f"{voice}: {param} must be a (lo, hi) pair of numbers as a list or tuple, "
        f"got {type(value).__name__} {value!r}{suffix}"
    )


def _range(voice: str, param: str, shown: str) -> str:
    return f"{voice}: {param} must be (lo, hi) with lo < hi and both finite; got {shown}"


def _limit(voice: str, param: str, shown: str) -> str:
    return f"{voice}: {param} must be a (lo, hi) pair of finite numbers; got {shown}"


def _quad_shape(value: object, suffix: str = "") -> str:
    return (
        "Kde2D: extent must be two (lo, hi) pairs of numbers, (xmin, xmax, ymin, ymax), "
        f"as a list or tuple, got {type(value).__name__} {value!r}{suffix}"
    )


def _quad_range(axis: str, shown: str) -> str:
    return f"Kde2D: extent {axis} pair must be (lo, hi) with lo < hi and both finite; got {shown}"


# --- the refused kinds: a factory per case, since a generator is read once ----------------

# (id, factory, suffix the shape text adds)
_REFUSED_KINDS: list[tuple[str, Callable[[], Any], str]] = [
    ("set", lambda: {9, 1}, _SET_SUFFIX),
    ("frozenset", lambda: frozenset({9, 1}), _SET_SUFFIX),
    ("bytes", lambda: b"\x01\x04", _BYTES_SUFFIX),
    ("bytearray", lambda: bytearray(b"\x01\x04"), _BYTES_SUFFIX),
    ("memoryview", lambda: memoryview(b"\x01\x04"), _BYTES_SUFFIX),
    ("str", lambda: "19", ""),
    ("dict", lambda: {1: 2}, ""),
    ("dict_keys", lambda: {1: 2}.keys(), ""),
    ("dict_values", lambda: {1: 2}.values(), ""),
    ("dict_items", lambda: {1: 2}.items(), ""),
    ("generator", lambda: (i for i in (1, 9)), ""),
    ("iterator", lambda: iter([1, 9]), ""),
    ("one_end", lambda: (1,), ""),
    ("three_ends", lambda: (1, 2, 3), ""),
    ("list_of_three", lambda: [1, 2, 3], ""),
    ("text_ends", lambda: ("a", "b"), ""),
    ("none_end", lambda: (None, 9), ""),
    ("bool_end", lambda: (True, 9), ""),
    ("numpy_bool_end", lambda: (np.bool_(True), 9), ""),
    ("timedelta_ends", lambda: (np.timedelta64(1), np.timedelta64(9)), ""),
    ("array_2d", lambda: np.array([[1, 9]]), ""),
    ("array_text", lambda: np.array(["a", "b"]), ""),
    ("huge_range", lambda: range(10**30), ""),
]

# (id, value, how the refused pair reads in `got (lo, hi)`), for a statistic range.
_REFUSED_RANGES: list[tuple[str, Any, str]] = [
    ("reversed", (9, 1), "(9, 1)"),
    ("equal", (1, 1), "(1, 1)"),
    ("nan_hi", (1, _NAN), "(1, nan)"),
    ("nan_lo", (_NAN, 1), "(nan, 1)"),
    ("neg_inf", (-_INF, 1), "(-inf, 1)"),
    ("pos_inf", (1, _INF), "(1, inf)"),
    ("fraction_reversed", (0.5, 0.25), "(0.5, 0.25)"),
    ("exponent_reversed", (1e-07, 1e-08), "(1e-07, 1e-08)"),
    ("beyond_float", (1, _HUGE), f"(1, {_HUGE})"),
    ("array_reversed", np.array([9, 1]), "(9, 1)"),
    ("whole_floats_reversed", [9.0, 1.0], "(9, 1)"),
    ("decimal_reversed", (_DEC(9), _DEC(1)), "(9, 1)"),
    ("fraction_type_reversed", (_FRAC(9), _FRAC(1)), "(9, 1)"),
    ("float32_reversed", np.array([9, 1], dtype=np.float32), "(9, 1)"),
]

# (id, value, shown) for a limit: only a non-finite end is refused.
_REFUSED_LIMITS: list[tuple[str, Any, str]] = [
    ("nan_hi", (0, _NAN), "(0, nan)"),
    ("nan_lo", (_NAN, 5), "(nan, 5)"),
    ("neg_inf", (-_INF, 5), "(-inf, 5)"),
    ("pos_inf", (0, _INF), "(0, inf)"),
    ("beyond_float", (1, _HUGE), f"(1, {_HUGE})"),
    ("decimal_infinity", (_DEC(0), _DEC("Infinity")), "(0, inf)"),
]

# Every spelling of the pair (1, 9) a door accepts.
_ACCEPTED: list[tuple[str, Callable[[], Any]]] = [
    ("list", lambda: [1, 9]),
    ("ndarray_int", lambda: np.array([1, 9])),
    ("ndarray_float", lambda: np.array([1.0, 9.0])),
    ("ndarray_float32", lambda: np.array([1, 9], dtype=np.float32)),
    ("ndarray_object", lambda: np.array([1, 9], dtype=object)),
    ("range", lambda: range(1, 10, 8)),
    ("float_list", lambda: [1.0, 9.0]),
    ("numpy_scalars", lambda: (np.int64(1), np.float64(9))),
    ("float32_scalars", lambda: (np.float32(1), np.float32(9))),
    ("decimals", lambda: (_DEC(1), _DEC(9))),
    ("fractions", lambda: (_FRAC(1), _FRAC(9))),
]


# --- the doors -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Door:
    """One pair parameter: how to call it, how the refusal speaks, what it checks."""

    name: str
    call: Callable[[Any], Any]
    voice: str
    param: str
    flavour: str  # "range" (lo < hi, finite) or "limit" (finite only)

    def __repr__(self) -> str:
        return self.name


def _joint(**limits: Any) -> Any:
    return fm.jointplot(_frame(), x="v", y="w", **limits).to_svg()


def _point() -> Any:
    return fm.Chart(_frame()).mark_point().encode(x="v", y="w")


def _density_spec(value: Any) -> Any:
    return fm.transform_density("v", extent=value)


def _mark_svg(mark: str, **kwargs: Any) -> str:
    chart = fm.Chart(_frame()).encode(x="v", y="w") if mark == "smooth" else fm.Chart(_frame())
    if mark != "smooth":
        chart = chart.encode(x="v")
    return getattr(chart, f"mark_{mark}")(**kwargs).to_svg()


_DOORS: list[Door] = [
    Door("transform_density", _density_spec, "transform_density", "extent", "range"),
    Door("Kde", lambda v: fm.Kde("v", extent=v), "Kde", "extent", "range"),
    Door("Bin", lambda v: fm.Bin("v", extent=v), "Bin", "extent", "range"),
    Door("Bin2D.x", lambda v: fm.Bin2D("v", "w", extent_x=v), "Bin2D", "extent_x", "range"),
    Door("Bin2D.y", lambda v: fm.Bin2D("v", "w", extent_y=v), "Bin2D", "extent_y", "range"),
    Door("Robust", lambda v: fm.Robust("v", "w", x_range=v), "Robust", "x_range", "range"),
    Door("Smooth", lambda v: fm.Smooth("v", "w", x_range=v), "Smooth", "x_range", "range"),
    Door("mark_density", lambda v: _mark_svg("density", extent=v), "Kde", "extent", "range"),
    Door("mark_histogram", lambda v: _mark_svg("histogram", extent=v), "Bin", "extent", "range"),
    Door("mark_smooth", lambda v: _mark_svg("smooth", x_range=v), "Smooth", "x_range", "range"),
    Door(
        "CoordCartesian.xlim",
        lambda v: fm.CoordCartesian(xlim=v),
        "CoordCartesian",
        "xlim",
        "limit",
    ),
    Door(
        "CoordCartesian.ylim",
        lambda v: fm.CoordCartesian(ylim=v),
        "CoordCartesian",
        "ylim",
        "limit",
    ),
    Door("CoordFixed.xlim", lambda v: fm.CoordFixed(xlim=v), "CoordFixed", "xlim", "limit"),
    Door("CoordFixed.ylim", lambda v: fm.CoordFixed(ylim=v), "CoordFixed", "ylim", "limit"),
    Door(
        "override.coord_xlim",
        lambda v: _point().coord(fm.CoordCartesian()).override(coord_xlim=v).to_svg(),
        "CoordCartesian",
        "xlim",
        "limit",
    ),
    Door(
        "override.coord_ylim",
        lambda v: _point().coord(fm.CoordCartesian()).override(coord_ylim=v).to_svg(),
        "CoordCartesian",
        "ylim",
        "limit",
    ),
    Door("jointplot.xlim", lambda v: _joint(xlim=v), "jointplot", "xlim", "limit"),
    Door("jointplot.ylim", lambda v: _joint(ylim=v), "jointplot", "ylim", "limit"),
    Door(
        "ReferenceLine.x",
        lambda v: fm.ReferenceLine("v", "w", x=v, y=(0, 1)),
        "ReferenceLine",
        "x",
        "limit",
    ),
    Door(
        "ReferenceLine.y",
        lambda v: fm.ReferenceLine("v", "w", x=(0, 1), y=v),
        "ReferenceLine",
        "y",
        "limit",
    ),
]
_RANGE_DOORS = [d for d in _DOORS if d.flavour == "range"]
_LIMIT_DOORS = [d for d in _DOORS if d.flavour == "limit"]


def _suffix_for(value: Any) -> str:
    if isinstance(value, (set, frozenset)):
        return _SET_SUFFIX
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _BYTES_SUFFIX
    return ""


def _refusal_of(door: Door, value: Any) -> str:
    with pytest.raises(ValueError) as caught:
        door.call(value)
    return str(caught.value)


# --- accepted kinds equal the tuple on every door -----------------------------------------


@pytest.mark.parametrize("door", _DOORS, ids=repr)
@pytest.mark.parametrize(("kind", "make"), _ACCEPTED, ids=[k for k, _ in _ACCEPTED])
def test_ordered_kinds_equal_the_tuple_on_every_door(door, kind, make):
    assert door.call(make()) == door.call((1, 9))


# --- refused kinds, on every door, in that door's voice -----------------------------------


@pytest.mark.parametrize("door", _DOORS, ids=repr)
@pytest.mark.parametrize(
    ("kind", "make", "suffix"), _REFUSED_KINDS, ids=[k for k, _, _ in _REFUSED_KINDS]
)
def test_unordered_and_malformed_kinds_are_refused_on_every_door(door, kind, make, suffix):
    value = make()
    assert _refusal_of(door, value) == _shape(door.voice, door.param, value, suffix)


@pytest.mark.parametrize("door", _DOORS, ids=repr)
def test_a_generator_is_refused_without_being_read(door):
    pair = iter([1, 9])
    with pytest.raises(ValueError):
        door.call(pair)
    assert list(pair) == [1, 9]


# --- statistic ranges: reversed, equal and non-finite are refused --------------------------


@pytest.mark.parametrize("door", _RANGE_DOORS, ids=repr)
@pytest.mark.parametrize(
    ("kind", "value", "shown"), _REFUSED_RANGES, ids=[k for k, _, _ in _REFUSED_RANGES]
)
def test_a_range_refuses_reversed_equal_and_non_finite_pairs(door, kind, value, shown):
    assert _refusal_of(door, value) == _range(door.voice, door.param, shown)


def test_kde_reversed_extent_reads_as_the_family_sentence():
    with pytest.raises(ValueError) as caught:
        fm.Kde("v", extent=(9, 1))
    assert (
        str(caught.value) == "Kde: extent must be (lo, hi) with lo < hi and both finite; got (9, 1)"
    )


def test_unset_ranges_are_unchanged():
    assert "extent" not in fm.transform_density("v")
    assert fm.Kde("v") == fm.Kde("v", extent=None)
    assert fm.Bin("v") == fm.Bin("v", extent=None)
    assert fm.Robust("v", "w") == fm.Robust("v", "w", x_range=None)
    assert fm.Smooth("v", "w") == fm.Smooth("v", "w", x_range=None)


def test_density_extent_spec_holds_builtin_numbers():
    spec = fm.transform_density("v", extent=np.array([1, 9]))
    assert spec["extent"] == [1, 9]
    assert all(type(end) is int for end in spec["extent"])


def test_density_extent_string_is_refused_at_construction_not_char_split():
    with pytest.raises(ValueError) as caught:
        fm.transform_density("x", extent="ab")
    assert str(caught.value) == (
        "transform_density: extent must be a (lo, hi) pair of numbers as a list or tuple, "
        "got str 'ab'"
    )


def test_density_extent_renders_the_same_for_every_ordered_kind():
    def svg(extent: Any) -> str:
        spec = fm.transform_density("v", extent=extent)
        return (
            fm.Chart(_frame())
            .transform(spec)
            .mark_line()
            .encode(x="value:Q", y="density:Q")
            .to_svg()
        )

    expected = svg((1, 9))
    for extent in ([1, 9], np.array([1, 9]), range(1, 10, 8)):
        assert svg(extent) == expected


# --- limits: finite ends only, a reversed pair kept ---------------------------------------


@pytest.mark.parametrize("door", _LIMIT_DOORS, ids=repr)
@pytest.mark.parametrize(
    ("kind", "value", "shown"), _REFUSED_LIMITS, ids=[k for k, _, _ in _REFUSED_LIMITS]
)
def test_a_limit_refuses_a_non_finite_end(door, kind, value, shown):
    assert _refusal_of(door, value) == _limit(door.voice, door.param, shown)


@pytest.mark.parametrize("door", _LIMIT_DOORS, ids=repr)
def test_a_reversed_limit_pair_is_accepted_and_equals_its_list(door):
    assert door.call([5, 0]) == door.call((5, 0))


def test_coordinate_limits_store_builtin_tuples():
    for coord in (fm.CoordCartesian, fm.CoordFixed):
        for param in ("xlim", "ylim"):
            made = coord(**{param: np.array([0, 5])})
            stored = getattr(made, param)
            assert stored == (0, 5)
            assert [type(end) for end in stored] == [int, int]
            key = "x_domain" if param == "xlim" else "y_domain"
            assert made.to_spec_dict()[key] == [0, 5]
        assert getattr(coord(), "xlim") is None
        assert "x_domain" not in coord().to_spec_dict()


def test_a_replaced_limit_is_checked_again():
    base = fm.CoordCartesian(xlim=(0, 5))
    with pytest.raises(ValueError) as caught:
        dataclasses.replace(base, ylim={1, 2})
    assert str(caught.value) == _shape("CoordCartesian", "ylim", {1, 2}, _SET_SUFFIX)


def test_chart_limit_methods_refuse_a_non_finite_end():
    chart = _point()
    with pytest.raises(ValueError) as caught:
        chart.xlim(0, _NAN)
    assert str(caught.value) == _limit("CoordCartesian", "xlim", "(0, nan)")
    with pytest.raises(ValueError) as caught:
        chart.ylim(_NAN, 1)
    assert str(caught.value) == _limit("CoordCartesian", "ylim", "(nan, 1)")


# The drawn output of a reversed limit, one digest per door and output kind. They were
# measured on the last release's Python sources for the coordinate classes, jointplot
# and the validators (the limit handed through as given), running on the installed
# extension; the readers here must draw the same bytes.
_REVERSED_LIMIT_DIGESTS = {
    "CoordCartesian.xlim": "f6a04f22f32caac4",
    "Chart.xlim": "f6a04f22f32caac4",
    "CoordFixed.xlim": "01394c71a8a2ba3c",
    "CoordFixed.ylim": "4b439b932a5aba9a",
    "Chart.ylim": "e23995783079dbd6",
    "jointplot.xlim": "6f0ac136a1764197",
    "jointplot.ylim": "c43ec1bb9b98ebd3",
    "interactive.xlim": "121b06352bac156c",
}


def test_reversed_limits_draw_the_axis_reversed_on_every_door():
    chart = _point()
    drawn = {
        "CoordCartesian.xlim": chart.coord(fm.CoordCartesian(xlim=(5, 0))).to_svg(),
        "Chart.xlim": chart.xlim(5, 0).to_svg(),
        "CoordFixed.xlim": chart.coord(fm.CoordFixed(xlim=(5, 0))).to_svg(),
        "CoordFixed.ylim": chart.coord(fm.CoordFixed(ylim=(8, 2))).to_svg(),
        "Chart.ylim": chart.ylim(8, 2).to_svg(),
        "jointplot.xlim": _joint(xlim=(5, 0)),
        "jointplot.ylim": _joint(ylim=(8, 2)),
        "interactive.xlim": str(
            chart.coord(fm.CoordCartesian(xlim=(5, 0))).interactive().scene_json
        ),
    }
    assert {name: _digest(svg) for name, svg in drawn.items()} == _REVERSED_LIMIT_DIGESTS
    # The reversed pair is not the forward pair read in sorted order.
    assert drawn["CoordCartesian.xlim"] != chart.coord(fm.CoordCartesian(xlim=(0, 5))).to_svg()
    assert drawn["jointplot.xlim"] != _joint(xlim=(0, 5))
    assert drawn["jointplot.ylim"] != _joint(ylim=(2, 8))


def test_reversed_limits_survive_the_override_door_and_a_list():
    chart = _point().coord(fm.CoordCartesian())
    expected = _point().coord(fm.CoordCartesian(xlim=(5, 0))).to_svg()
    assert chart.override(coord_xlim=[5, 0]).to_svg() == expected
    assert _joint(xlim=[5, 0]) == _joint(xlim=(5, 0))
    assert fm.CoordCartesian(xlim=(5, 0)).to_spec_dict()["x_domain"] == [5, 0]


@pytest.mark.parametrize("kind", ["scatter", "kde", "hist"])
def test_jointplot_limit_is_refused_on_every_center_kind(kind):
    with pytest.raises(ValueError) as caught:
        fm.jointplot(_frame(), x="v", y="w", kind=kind, xlim={5, 0})
    assert str(caught.value) == _shape("jointplot", "xlim", {5, 0}, _SET_SUFFIX)


# --- ReferenceLine endpoints ---------------------------------------------------------------


def _reference_svg(x: Any, y: Any) -> str:
    return (
        fm.Chart(_frame())
        ._with_statistics(fm.ReferenceLine("v", "w", x=x, y=y))
        .mark_line()
        .encode(x="v", y="w")
        .to_svg()
    )


def test_reference_line_reversed_endpoints_draw_the_line_from_the_first_endpoint():
    forward = _reference_svg((0, 1), (0, 1))
    reversed_ = _reference_svg((1, 0), (0, 1))
    assert _reference_svg([1, 0], [0, 1]) == reversed_
    # The line runs from (x=1, y=0), bottom right, to (x=0, y=1), top left; the forward
    # pair runs from bottom left to top right.
    assert 'polyline points="616,412.961 70.944,24"' in reversed_
    assert 'polyline points="70.944,412.961 616,24"' in forward


def test_reference_line_endpoints_keep_the_given_order_in_the_object():
    assert "x=(1.0, 0.0)" in repr(fm.ReferenceLine("v", "w", x=(1, 0), y=(0, 1)))


# --- Kde2D: four values read as two pairs ---------------------------------------------------


def _kde2d(extent: Any) -> Any:
    return fm.Kde2D("v", "w", extent=extent)


@pytest.mark.parametrize(
    "make",
    [
        lambda: [0, 1, 2, 3],
        lambda: np.array([0, 1, 2, 3]),
        lambda: np.array([0, 1, 2, 3], dtype=np.float32),
        lambda: range(4),
        lambda: (np.int64(0), np.float64(1), _DEC(2), _FRAC(3)),
    ],
    ids=["list", "ndarray", "float32", "range", "mixed_numbers"],
)
def test_kde2d_extent_four_values_equal_the_tuple(make):
    assert _kde2d(make()) == _kde2d((0, 1, 2, 3))


def test_kde2d_unset_extent_is_unchanged():
    assert _kde2d(None) == fm.Kde2D("v", "w")


@pytest.mark.parametrize(
    ("kind", "make", "suffix"),
    [
        ("set", lambda: {0, 1, 2, 3}, _SET_SUFFIX),
        ("frozenset", lambda: frozenset({0, 1, 2, 3}), _SET_SUFFIX),
        ("bytes", lambda: b"\x00\x01\x02\x03", _BYTES_SUFFIX),
        ("bytearray", lambda: bytearray(b"\x00\x01\x02\x03"), _BYTES_SUFFIX),
        ("memoryview", lambda: memoryview(b"\x00\x01\x02\x03"), _BYTES_SUFFIX),
        ("str", lambda: "abcd", ""),
        ("dict", lambda: {1: 2}, ""),
        ("dict_keys", lambda: {0: 1, 2: 3}.keys(), ""),
        ("generator", lambda: (i for i in range(4)), ""),
        ("iterator", lambda: iter([0, 1, 2, 3]), ""),
        ("two_values", lambda: (1, 2), ""),
        ("five_values", lambda: [0, 1, 2, 3, 4], ""),
        ("text_end", lambda: (0, 1, 2, "a"), ""),
        ("none_end", lambda: (0, 1, 2, None), ""),
        ("bool_end", lambda: (0, 1, 2, True), ""),
        ("array_2d", lambda: np.array([[0, 1], [2, 3]]), ""),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_kde2d_extent_refuses_unordered_and_malformed_kinds(kind, make, suffix):
    value = make()
    with pytest.raises(ValueError) as caught:
        _kde2d(value)
    assert str(caught.value) == _quad_shape(value, suffix)


@pytest.mark.parametrize(
    ("extent", "axis", "shown"),
    [
        ((5, 1, 0, 1), "x", "(5, 1)"),
        ((0, 1, 3, 2), "y", "(3, 2)"),
        ((1, 1, 0, 1), "x", "(1, 1)"),
        ((0, 1, 2, 2), "y", "(2, 2)"),
        ((0, 1, 2, _NAN), "y", "(2, nan)"),
        ((_NAN, 1, 0, 1), "x", "(nan, 1)"),
        ((0, _INF, 0, 1), "x", "(0, inf)"),
        ((0.5, 0.25, 0, 1), "x", "(0.5, 0.25)"),
        ((0, 1, 1e-07, 1e-08), "y", "(1e-07, 1e-08)"),
        ((0, 1, 1, _HUGE * 10), "y", f"(1, {_HUGE * 10})"),
    ],
)
def test_kde2d_extent_refuses_a_reversed_or_non_finite_pair_naming_it(extent, axis, shown):
    with pytest.raises(ValueError) as caught:
        _kde2d(extent)
    assert str(caught.value) == _quad_range(axis, shown)


# --- unchanged precedents ------------------------------------------------------------------


def test_window_frame_and_impute_keyvals_keep_their_readers():
    assert fm.transform_window("sum", "w", frame=[None, 0])["frame"] == [None, 0]
    with pytest.raises(ValueError) as caught:
        fm.transform_window("sum", "w", frame={0, 1})
    assert str(caught.value) == (
        "transform_window: frame must be a (preceding, following) pair of row offsets, "
        "each an integer or None; got {0, 1}"
    )
    assert fm.transform_impute("w", key="v", keyvals=[0, 1])["keyvals"] == [0, 1]
    with pytest.raises(ValueError) as caught:
        fm.transform_impute("w", key="v", keyvals={0, 1})
    assert str(caught.value) == (
        "transform_impute: keyvals must be an ordered sequence of key values, got set {0, 1}; "
        "a set has no order, so the output would change from run to run; "
        "pass a list, e.g. sorted(...)"
    )


def test_a_huge_range_is_refused_by_the_window_frame_reader_too():
    with pytest.raises(ValueError) as caught:
        fm.transform_window("sum", "w", frame=range(10**30))
    assert str(caught.value) == (
        "transform_window: frame must be a (preceding, following) pair of row offsets, "
        "each an integer or None; got range(0, 1000000000000000000000000000000)"
    )
