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
import json
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


_DENSITY_DATA_EXTENTS = [
    ("reversed", [9, 1], "(9, 1)"),
    ("equal", [3, 3], "(3, 3)"),
]
_DENSITY_DATA_STAT = (
    "transform failed: ValueError: data_density: extent must be (lo, hi) with lo < hi and both "
    "finite; got "
)
_COMPOSITE_LEAF = "failed to render composite leaf leaf #0: "


def _density_data_spec(extent: Any, *, groupby: bool = False) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": "density_data", "field": "v", "extent": extent}
    if groupby:
        spec["groupby"] = ["g"]
    return spec


def _density_data_chart(
    extent: Any, frame: pl.DataFrame | None = None, *, groupby: bool = False
) -> fm.Chart:
    frame = _frame() if frame is None else frame
    spec = _density_data_spec(extent, groupby=groupby)
    return fm.Chart(frame).transform(spec).mark_line().encode(x="value:Q", y="density:Q")


def _render_refusal(chart: Any, output: str) -> str:
    render = chart.to_svg if output == "svg" else lambda: chart.interactive().scene_json
    with pytest.raises(ValueError) as caught:
        render()
    return str(caught.value)


def _grouped_frame() -> pl.DataFrame:
    return _frame().with_columns(pl.Series("g", ["a", "b"] * 20 + ["a"]))


# (id, frame) for every column shape the extent guard sits in front of
_DENSITY_DATA_FRAMES: list[tuple[str, Callable[[], pl.DataFrame]]] = [
    ("float64", _frame),
    ("zero_rows", lambda: _frame().clear()),
    ("all_null", lambda: _frame().with_columns(pl.lit(None, dtype=pl.Float64).alias("v"))),
    ("int32", lambda: _frame().with_columns((pl.col("v") * 4).cast(pl.Int32))),
    ("float32", lambda: _frame().with_columns(pl.col("v").cast(pl.Float32))),
]


@pytest.mark.parametrize("output", ["svg", "interactive"])
@pytest.mark.parametrize(
    ("kind", "extent", "shown"), _DENSITY_DATA_EXTENTS, ids=[k for k, _, _ in _DENSITY_DATA_EXTENTS]
)
def test_density_data_dict_refuses_a_reversed_or_equal_extent(output, kind, extent, shown):
    assert _render_refusal(_density_data_chart(extent), output) == _DENSITY_DATA_STAT + shown


@pytest.mark.parametrize("output", ["svg", "interactive"])
@pytest.mark.parametrize(
    ("kind", "extent", "shown"), _DENSITY_DATA_EXTENTS, ids=[k for k, _, _ in _DENSITY_DATA_EXTENTS]
)
@pytest.mark.parametrize("frame_id", [i for i, _ in _DENSITY_DATA_FRAMES])
def test_density_data_extent_refusal_does_not_depend_on_the_column(
    output, kind, extent, shown, frame_id
):
    frame = dict(_DENSITY_DATA_FRAMES)[frame_id]()
    assert _render_refusal(_density_data_chart(extent, frame), output) == _DENSITY_DATA_STAT + shown


@pytest.mark.parametrize("output", ["svg", "interactive"])
def test_density_data_extent_refusal_holds_for_a_groupby_frame(output):
    chart = _density_data_chart([9, 1], _grouped_frame(), groupby=True)
    assert _render_refusal(chart, output) == _DENSITY_DATA_STAT + "(9, 1)"


def _plus() -> Any:
    return _density_data_chart([9, 1]) + _density_data_chart([9, 1])


def _layer_chart() -> Any:
    return fm.LayerChart(_density_data_chart([9, 1]), _density_data_chart([9, 1]))


def _hconcat() -> Any:
    return fm.hconcat(_density_data_chart([9, 1]), _density_data_chart([9, 1]))


# (id, factory, prefix of the static render, prefix of the interactive render): a
# composite leaf renders through the composite entry, which names the failing leaf
_DENSITY_DATA_DOORS: list[tuple[str, Callable[[], Any], str, str]] = [
    ("plus", _plus, "", ""),
    ("layer_chart", _layer_chart, _COMPOSITE_LEAF, ""),
    ("hconcat", _hconcat, _COMPOSITE_LEAF, _COMPOSITE_LEAF),
]


@pytest.mark.parametrize("output", ["svg", "interactive"])
@pytest.mark.parametrize(
    ("door", "make", "svg_prefix", "interactive_prefix"),
    _DENSITY_DATA_DOORS,
    ids=[d[0] for d in _DENSITY_DATA_DOORS],
)
def test_density_data_extent_refusal_holds_on_every_door(
    output, door, make, svg_prefix, interactive_prefix
):
    prefix = svg_prefix if output == "svg" else interactive_prefix
    assert _render_refusal(make(), output) == prefix + _DENSITY_DATA_STAT + "(9, 1)"


@pytest.mark.parametrize("output", ["svg", "interactive"])
@pytest.mark.parametrize("extent", [[_NAN, 9], [1, _INF]], ids=["nan", "inf"])
def test_density_data_dict_refuses_a_non_finite_extent(output, extent):
    """A non-finite end never reaches the transform: the spec JSON the Rust side
    reads cannot carry it, so the refusal is the JSON reader's. The reader names
    the column of the first non-finite token in the payload Python sends, so the
    expected text is built from that payload."""
    payload = json.dumps([_density_data_spec(extent)])
    token = "NaN" if extent[0] != extent[0] else "Infinity"
    expected = f"transforms_json: expected value at line 1 column {payload.index(token) + 1}"
    assert _render_refusal(_density_data_chart(extent), output) == expected


def _wire_step(kind: str, **fields: Any) -> dict[str, Any]:
    """The wire dict of a default ``kind`` transform (the dict the engine reads), with *fields*
    set over it, so the step is as complete as the constructor writes it."""
    from ferrum._layer_transforms import _transforms_to_json_list_named

    builders = {
        "kde": lambda: fm.Kde("v"),
        "violin": lambda: fm.Violin("v"),
        "bin": lambda: fm.Bin("v"),
        "bin_2d": lambda: fm.Bin2D("v", "w"),
        "kde2_d": lambda: fm.Kde2D("v", "w"),
        "robust": lambda: fm.Robust("v", "w"),
        "smooth": lambda: fm.Smooth("v", "w"),
    }
    return {**_transforms_to_json_list_named([builders[kind]()])[0], **fields}


def _wire_range(voice: str, param: str, shown: str) -> str:
    return (
        f"transform failed: ValueError: {voice}: {param} must be (lo, hi) with lo < hi and "
        f"both finite; got {shown}"
    )


# (id, wire step, the refusal): every wire door reads a malformed range in the sentence the
# constructors give, in the statistic's own voice, with each end spelled as Python spells it.
_WIRE_RANGES: list[tuple[str, dict[str, Any], str]] = [
    ("kde-extent", _wire_step("kde", extent=[9, 1]), _wire_range("stat_kde", "extent", "(9, 1)")),
    (
        "kde-count_scale_extent",
        _wire_step("kde", count_scale_extent=[10, 0]),
        _wire_range("stat_kde", "count_scale_extent", "(10, 0)"),
    ),
    (
        "violin-extent",
        _wire_step("violin", extent=[9, 1]),
        _wire_range("stat_violin", "extent", "(9, 1)"),
    ),
    ("bin-extent", _wire_step("bin", extent=[9, 1]), _wire_range("stat_bin", "extent", "(9, 1)")),
    ("bin-equal", _wire_step("bin", extent=[3, 3]), _wire_range("stat_bin", "extent", "(3, 3)")),
    (
        "bin-exponent",
        _wire_step("bin", extent=[1e-7, 1e-8]),
        _wire_range("stat_bin", "extent", "(1e-07, 1e-08)"),
    ),
    (
        "bin_2d-x",
        _wire_step("bin_2d", extent_x=[9, 1]),
        _wire_range("Bin2D", "extent_x", "(9, 1)"),
    ),
    (
        "kde2_d-x",
        _wire_step("kde2_d", extent=[9, 1, -3, 3]),
        _wire_range("stat_kde_2d", "extent x pair", "(9, 1)"),
    ),
    (
        "kde2_d-y",
        _wire_step("kde2_d", extent=[0, 1, 3, -3]),
        _wire_range("stat_kde_2d", "extent y pair", "(3, -3)"),
    ),
    (
        "robust-x_range",
        _wire_step("robust", x_range=[9, 1]),
        _wire_range("Robust", "x_range", "(9, 1)"),
    ),
    (
        "robust-equal",
        _wire_step("robust", x_range=[3, 3]),
        _wire_range("Robust", "x_range", "(3, 3)"),
    ),
    (
        "smooth-x_range",
        _wire_step("smooth", x_range=[9, 1]),
        _wire_range("stat_smooth", "x_range", "(9, 1)"),
    ),
    (
        "smooth-equal",
        _wire_step("smooth", x_range=[3, 3]),
        _wire_range("stat_smooth", "x_range", "(3, 3)"),
    ),
]


@pytest.mark.parametrize("output", ["svg", "interactive"])
@pytest.mark.parametrize(
    ("step", "refusal"), [(s, r) for _, s, r in _WIRE_RANGES], ids=[i for i, _, _ in _WIRE_RANGES]
)
def test_a_wire_range_is_refused_in_the_family_sentence(output, step, refusal):
    chart = fm.Chart(_frame()).transform(step).mark_line().encode(x="v:Q", y="w:Q")
    assert _render_refusal(chart, output) == refusal


def test_density_data_dict_keeps_an_ordered_extent():
    assert "<polyline" in _density_data_chart([1, 9]).to_svg()


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
    "interactive.xlim": "500d83f02b1b4361",
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


# --- scale constructors: domain= and range= read a pair the same way -----------------------

#: The kinds a scale's pair parameter refuses unread (the shape refusal of every other pair door).
_SCALE_REFUSED_KINDS = [
    entry
    for entry in _REFUSED_KINDS
    if entry[0]
    in {
        "set",
        "frozenset",
        "bytes",
        "bytearray",
        "memoryview",
        "str",
        "dict",
        "generator",
        "iterator",
    }
]

#: (voice, parameter, constructor taking the pair). Every continuous scale reads a
#: ``(lo, hi)`` pair on each of these; a log scale's pair is positive.
_SCALE_PAIRS: list[tuple[str, str, Callable[[Any], Any]]] = [
    (cls, param, lambda v, cls=cls, param=param: getattr(fm, cls)(**{param: v}))
    for cls in ("LinearScale", "LogScale", "PowScale", "SqrtScale", "SymlogScale", "TimeScale")
    for param in ("domain", "range")
] + [
    ("DivergingScale", "domain", lambda v: fm.DivergingScale(domain=v)),
    ("SequentialScale", "domain", lambda v: fm.SequentialScale(domain=v)),
    ("BandScale", "range", lambda v: fm.BandScale(domain=["a", "b"], range=v)),
    ("PointScale", "range", lambda v: fm.PointScale(domain=["a", "b"], range=v)),
]
_SCALE_IDS = [f"{voice}.{param}" for voice, param, _ in _SCALE_PAIRS]


@pytest.mark.parametrize(("voice", "param", "build"), _SCALE_PAIRS, ids=_SCALE_IDS)
@pytest.mark.parametrize(
    ("kind", "make", "suffix"),
    _SCALE_REFUSED_KINDS,
    ids=[k for k, _, _ in _SCALE_REFUSED_KINDS],
)
def test_a_scale_pair_refuses_unordered_and_byte_kinds_in_the_shape_words(
    voice, param, build, kind, make, suffix
):
    value = make()
    with pytest.raises(ValueError) as caught:
        build(value)
    expected = _shape(voice, param, value, suffix)
    if voice == "DivergingScale":
        # A diverging domain is a (lo, hi) pair or a (lo, mid, hi) triple.
        expected = expected.replace(
            "must be a (lo, hi) pair of numbers",
            "must be a (lo, hi) pair or (lo, mid, hi) triple of numbers",
        )
    assert str(caught.value) == expected


@pytest.mark.parametrize("domain", [[0, 10], (0, 10), [0, 4, 10], (0, 4, 10)])
def test_a_diverging_domain_reads_a_list_like_a_tuple_of_two_or_three_values(domain):
    scale = fm.DivergingScale(domain=domain)
    assert scale.domain == ([0.0, 5.0, 10.0] if len(domain) == 2 else [0.0, 4.0, 10.0])


def _scale_state(scale: Any) -> tuple[str, Any, Any]:
    """A scale's repr with its domain and range (a band or point repr omits its range)."""
    return repr(scale), getattr(scale, "domain", None), getattr(scale, "range", None)


@pytest.mark.parametrize(("voice", "param", "build"), _SCALE_PAIRS, ids=_SCALE_IDS)
def test_a_scale_pair_reads_a_list_like_a_tuple_and_keeps_a_reversed_order(voice, param, build):
    assert _scale_state(build([1, 9])) == _scale_state(build((1, 9)))
    assert _scale_state(build((9, 1))) == _scale_state(build([9, 1]))
    assert _scale_state(build((9, 1))) != _scale_state(build((1, 9)))


@pytest.mark.parametrize(("voice", "param", "build"), _SCALE_PAIRS, ids=_SCALE_IDS)
def test_a_scale_pair_generator_is_refused_without_being_read(voice, param, build):
    pair = iter([1, 9])
    with pytest.raises(ValueError):
        build(pair)
    assert list(pair) == [1, 9]


# --- BreakAxis.gap: a list of two bounds is the pair a tuple of two bounds is ---------------


def test_break_axis_gap_list_pair_equals_the_tuple_pair():
    assert fm.BreakAxis(axis="y", gap=[400, 900]) == fm.BreakAxis(axis="y", gap=(400, 900))
    assert fm.BreakAxis(axis="y", gap=[400, 900]).gap == (400.0, 900.0)


def test_break_axis_gap_list_pair_renders_the_tuple_pair_chart():
    frame = pl.DataFrame({"a": ["u", "v", "w"], "b": [1.0, 2.0, 1000.0]})

    def render(gap: Any) -> str:
        chart = fm.Chart(frame).mark_bar().encode(x="a", y="b") + fm.BreakAxis(axis="y", gap=gap)
        return chart.to_svg()

    assert render([400, 900]) == render((400, 900))


def test_break_axis_gap_list_of_pairs_and_temporal_list_pair():
    assert fm.BreakAxis(axis="y", gap=[(1, 5), (6, 9)]).gap == [(1.0, 5.0), (6.0, 9.0)]
    assert (
        fm.BreakAxis(axis="x", gap=["2020-01-01", "2020-02-01"]).gap
        == fm.BreakAxis(axis="x", gap=("2020-01-01", "2020-02-01")).gap
    )


@pytest.mark.parametrize(
    "gap",
    [[10, 20, 30], [10], [[10, 20, 30], [40, 50]]],
    ids=["three_bounds", "one_bound", "pair_of_three"],
)
def test_break_axis_gap_malformed_lists_are_still_refused(gap):
    with pytest.raises(ValueError):
        fm.BreakAxis(axis="y", gap=gap)
