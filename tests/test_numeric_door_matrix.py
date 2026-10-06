"""Numeric door x input kind matrix: every numeric door answers "is this a number" alike.

The door population is derived, not hand-listed: every hit of
``isinstance(... numbers.Real | (int, float) | int)``, ``math.isfinite`` and
``float(`` after ``is_real_number(`` in ``src/ferrum`` was classified, plus
``catplot(ci=)``'s ``float()`` probe and the numeric fields of the mark-style
funnel (``MarkBase``, behind every ``mark_*`` style kwarg), and the hits that
decide whether a user-facing numeric argument is a number are the doors
below, each exercised through its public entry. Real-number doors store what
``ferrum._validate.coerce_finite_real`` returns; integer doors store what
``coerce_positive_int`` (or, for a mark's unsigned-integer style field,
``coerce_non_negative_int``) returns. So every door accepts ``int``, ``float``,
numpy scalars and (for real-number doors) ``Decimal``, and refuses ``bool``,
``str``, ``list``, ``nan``, ``inf``, an integer a float cannot hold
(``10**400``) and a signaling NaN with a ``ValueError`` naming the public door
and the value received. ``None`` is each door's own documented unset sentinel
where it has one (heatmap bounds, ``AxisConfig.domain_min``, the t-SNE
default, ``ncols``/``columns``, mark style kwargs, ``transform_impute(value=)``),
and a refusal where the argument is required. The expression and key-universe
literals of ``transform_filter`` and ``transform_impute(keyvals=)`` also take
strings, booleans and ``None`` as literals of their own type, so only the
numeric kinds are cells there.

Every accepted cell must produce the same output as the builtin spelling of
the same value, and each door carries a control value whose output must
differ (the argument is not inert), so an "accepted" cell cannot pass
vacuously.
"""

from __future__ import annotations

import decimal
import functools
import hashlib
import json
import numbers
import operator
import warnings
from typing import Any, Callable, NamedTuple

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum._core import mark_integer_kwargs, mark_numeric_kwargs
from ferrum._layer import drawn_style_fields
from ferrum.marks.base import (
    _INTEGER_MARK_KWARGS,
    _REAL_MARK_KWARGS,
    _mark_kwarg_error,
    MarkBase,
    honored_kwargs,
)
from ferrum.plots.distribution import _CATPLOT_AGGREGATING_KINDS, _CATPLOT_VALID_KINDS

_NON_FINITE = {"nan", "inf", "huge", "sNaN"}


def _kinds(v: int) -> dict[str, Any]:
    """The fourteen input kinds, each spelling the value *v* where it can."""
    return {
        "int": v,
        "float": float(v),
        "np_int": np.int64(v),
        "np_float": np.float64(v),
        "np_float32": np.float32(v),
        "Decimal": decimal.Decimal(v),
        "bool": True,
        "str": str(v),  # a numeric string: no door may launder it into a number
        "list": [v],
        "None": None,
        "nan": float("nan"),
        "inf": float("inf"),
        "huge": 10**400,
        "sNaN": decimal.Decimal("sNaN"),
    }


KINDS = list(_kinds(1))

# --- doors, each through its public entry --------------------------------

_HEATMAP_DF = pl.DataFrame({"a": [-2.0, 0.0, 10.0], "b": [1.0, -1.0, 5.0]})
_POINTS = pl.DataFrame(
    {"x": [0.0, 5.0, 10.0, 15.0], "y": [1.0, 2.0, 3.0, 4.0], "g": ["p", "q", "r", "s"]}
)
_CAT = pl.DataFrame({"day": ["a", "a", "a", "b", "b", "b"], "tip": [1.0, 3.0, 2.0, 5.0, 4.0, 9.0]})
_TSNE_N = 20


def _heatmap(kwarg: str) -> Callable[[Any], str]:
    def call(v: Any) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fm.heatmap(_HEATMAP_DF, annot=False, **{kwarg: v}).to_svg()

    return call


def _configure_domain_min(v: Any) -> str:
    chart = fm.Chart(_POINTS).mark_point().encode(x="x", y="y")
    return chart.configure(axis_y=fm.AxisConfig(domain_min=v)).to_svg()


@functools.cache
def _tsne_source():
    rng = np.random.RandomState(0)
    x = pl.DataFrame({f"f{i}": rng.randn(_TSNE_N) for i in range(3)})
    return fm.ModelSource(None, x, random_state=0)


def _tsne_perplexity(v: Any) -> str:
    kwargs = {} if v is None else {"perplexity": v}
    emb = _tsne_source().embeddings(method="tsne", **kwargs)
    return emb.select("dim_0", "dim_1").to_numpy().tobytes().hex()


def _break_axis_gap(v: Any) -> str:
    return repr(fm.BreakAxis(axis="y", gap=(v, 5)).gap)


def _facet_ncols(v: Any) -> str:
    return fm.Chart(_POINTS).mark_point().encode(x="x", y="y").facet("g", ncols=v).to_svg()


def _facet_columns(v: Any) -> str:
    chart = fm.Chart(_POINTS).mark_point()
    return chart.encode(x="x", y="y", facet=fm.Facet("g", columns=v)).to_svg()


def _catplot(kwarg: str) -> Callable[[Any], str]:
    def call(v: Any) -> str:
        return fm.catplot(_CAT, x="day", y="tip", kind="point", seed=0, **{kwarg: v}).to_svg()

    return call


def _mark_json(build: Callable[["fm.Chart", Any], "fm.Chart"], df: pl.DataFrame):
    def call(v: Any) -> str:
        return build(fm.Chart(df), v).to_json()

    return call


_SELECTION = fm.selection_point(name="door_matrix")


def _conditional_opacity(v: Any) -> str:
    cond = _SELECTION.when(fm.value(1.0)).otherwise(fm.value(v))
    chart = fm.Chart(_POINTS).mark_point().encode(x="x", y="y", opacity=cond)
    return json.dumps(json.loads(chart.to_json())["conditionals"])


def _transform_filter(v: Any) -> str:
    return json.dumps(fm.transform_filter({"x": {">": v}}))


def _impute_keyval(v: Any) -> str:
    return json.dumps(fm.transform_impute("y", key="x", keyvals=[v]))


def _impute_value(v: Any) -> str:
    return json.dumps(fm.transform_impute("y", value=v))


# --- expected refusals ------------------------------------------------------


def _real_refusal(label: str) -> Callable[[str, Any], str]:
    """``coerce_finite_real``'s two refusals, in the door's voice."""

    def refusal(kind: str, v: Any) -> str:
        if kind in _NON_FINITE:
            return f"{label} must be a finite number a float can hold; received {v!r}"
        return (
            f"{label} must be a real number (int, float, numpy scalar, or Decimal), "
            f"not {type(v).__name__}; received {v!r}"
        )

    return refusal


def _int_refusal(label: str, expects: str = "a positive int") -> Callable[[str, Any], str]:
    """The ``coerce_*_int`` refusals, in the door's voice."""

    def refusal(kind: str, v: Any) -> str:
        if kind == "huge":
            return (
                f"{label} must be at most 2**53, the largest integer a float holds exactly; "
                f"received {v!r}"
            )
        return f"{label} must be {expects}, not {type(v).__name__}; received {v!r}"

    return refusal


def _break_axis_refusal(kind: str, v: Any) -> str:
    # A string is parsed as a date and a list is read as a list of pairs,
    # before any numeric check: this is what the public path produces.
    if kind == "str":
        return (
            f"BreakAxis.gap[0]: cannot parse {v!r} as an ISO-8601 date or datetime "
            "(use 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS')"
        )
    if kind == "list":
        return "BreakAxis.gap[0] must be a (start, end) pair of length 2; got [1]"
    return _real_refusal("BreakAxis.gap[0]: start")(kind, v)


def _stroke_dash_refusal(kind: str, v: Any) -> str:
    return (
        f"mark_line: stroke_dash={[v, 2]!r} is not a valid stroke_dash value. Expected a "
        "numeric list (e.g. [4.0, 2.0]), a comma-separated numeric string "
        '(e.g. "4,2"), or a named linetype (one of '
        "['dashdot', 'dashed', 'dotted', 'longdash', 'solid'])."
    )


def _conditional_refusal(kind: str, v: Any) -> str:
    return f"fm.value({v!r}) for channel='opacity' needs a number, got {type(v).__name__}: {v!r}"


def _ci_refusal(kind: str, v: Any) -> str:
    return (
        f"catplot(ci={v!r}) is not supported -- ci is a PERCENT and must be a number in "
        "the open interval (0, 100), exclusive at both ends (a 0 % interval has no width "
        "and a 100 % bootstrap interval is not estimable from a finite resample). Pass "
        "ci=95 (the default) or any percent strictly between 0 and 100."
    )


def _n_boot_refusal(kind: str, v: Any) -> str:
    return (
        f"catplot(n_boot={v!r}) is not supported -- n_boot is the bootstrap resample count "
        "and must be an integer >= 1 and at most 2**53. Pass n_boot=1000 (the default) or "
        "any positive count."
    )


class Door(NamedTuple):
    call: Callable[[Any], str]
    sample: int  # the value every kind spells
    # The kind whose output every accepted kind must equal, or ``None`` when
    # the door keeps an integer an integer on the wire: each accepted kind
    # then equals its own builtin spelling (``int`` or ``float``).
    reference: str | None
    control: Any  # a different valid value whose output must differ
    refusal: Callable[[str, Any], str]
    accepted: frozenset[str]
    skip: frozenset[str] = frozenset()  # kinds that are a literal of another type here


_REAL = frozenset({"int", "float", "np_int", "np_float", "np_float32", "Decimal", "None"})
_LITERAL_OF_ANOTHER_TYPE = frozenset({"bool", "str", "list", "None"})
_INTEGER = frozenset({"int", "np_int", "None"})
_TSNE_LABEL = "ModelSource.embeddings(method='tsne', perplexity=...)"
_TEXT = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "t": ["abcdefghijkl", "mnopqrstuvwx"]})
_MARK_TEXT_LIMIT = "mark_text(limit=...)"


def _mark_text_limit(v: Any) -> "fm.Chart":
    return fm.Chart(_TEXT).mark_text(limit=v).encode(x="x", y="y", text="t")


DOORS: dict[str, Door] = {
    "heatmap_vmin": Door(
        _heatmap("vmin"), 1, "float", None, _real_refusal("heatmap(vmin=...)"), _REAL
    ),
    "heatmap_vmax": Door(
        _heatmap("vmax"), 1, "float", None, _real_refusal("heatmap(vmax=...)"), _REAL
    ),
    "heatmap_center": Door(
        _heatmap("center"), 1, "float", None, _real_refusal("heatmap(center=...)"), _REAL
    ),
    "configure_domain_min": Door(
        _configure_domain_min, 2, "float", None, _real_refusal("AxisConfig(domain_min=...)"), _REAL
    ),
    "tsne_perplexity": Door(_tsne_perplexity, 5, "float", None, _real_refusal(_TSNE_LABEL), _REAL),
    "break_axis_gap": Door(_break_axis_gap, 1, "float", 2, _break_axis_refusal, _REAL - {"None"}),
    "catplot_ci": Door(_catplot("ci"), 90, "float", 50, _ci_refusal, _REAL - {"None"}),
    "mark_point_size": Door(
        _mark_json(lambda c, v: c.mark_point(size=v).encode(x="x", y="y"), _POINTS),
        3,
        None,
        100,
        _real_refusal("mark_point(size=...)"),
        _REAL,
    ),
    "mark_point_alpha": Door(
        _mark_json(lambda c, v: c.mark_point(alpha=v).encode(x="x", y="y"), _POINTS),
        1,
        None,
        0.5,
        _real_refusal("mark_point(alpha=...)"),
        _REAL,
    ),
    "mark_line_stroke_width": Door(
        _mark_json(lambda c, v: c.mark_line(stroke_width=v).encode(x="x", y="y"), _POINTS),
        2,
        None,
        5,
        _real_refusal("mark_line(stroke_width=...)"),
        _REAL,
    ),
    "mark_boxplot_opacity": Door(
        _mark_json(lambda c, v: c.mark_boxplot(opacity=v).encode(x="day", y="tip"), _CAT),
        1,
        None,
        0.5,
        _real_refusal("mark_boxplot(opacity=...)"),
        _REAL,
    ),
    "stroke_dash_length": Door(
        _mark_json(lambda c, v: c.mark_line(stroke_dash=[v, 2]).encode(x="x", y="y"), _POINTS),
        3,
        "float",
        5,
        _stroke_dash_refusal,
        _REAL - {"None"},
    ),
    "conditional_opacity": Door(
        _conditional_opacity, 1, "float", 0.25, _conditional_refusal, _REAL - {"None"}
    ),
    "transform_filter_literal": Door(
        _transform_filter,
        5,
        None,
        10,
        _real_refusal("transform_filter: value for field 'x'"),
        _REAL,
        _LITERAL_OF_ANOTHER_TYPE,
    ),
    "transform_impute_keyval": Door(
        _impute_keyval,
        5,
        None,
        10,
        _real_refusal("transform_impute: keyvals value for key 'x'"),
        _REAL,
        _LITERAL_OF_ANOTHER_TYPE,
    ),
    "transform_impute_value": Door(
        _impute_value, 5, None, 10, _real_refusal("transform_impute(value=...)"), _REAL
    ),
    "facet_ncols": Door(
        _facet_ncols, 2, "int", None, _int_refusal("Chart.facet(ncols=...)"), _INTEGER
    ),
    "facet_columns": Door(
        _facet_columns, 2, "int", None, _int_refusal("Facet(columns=...)"), _INTEGER
    ),
    "catplot_n_boot": Door(
        _catplot("n_boot"), 50, "int", 1000, _n_boot_refusal, _INTEGER - {"None"}
    ),
    "mark_text_limit": Door(
        lambda v: _mark_text_limit(v).to_json(),
        3,
        "int",
        10,
        _int_refusal(_MARK_TEXT_LIMIT, "a non-negative int"),
        _INTEGER,
    ),
}

CELLS = [(door, kind) for door, d in DOORS.items() for kind in KINDS if kind not in d.skip]


def _builtin(value: Any) -> int | float:
    """The builtin spelling of an accepted *value*: ``int`` for an integer, else ``float``."""
    return operator.index(value) if isinstance(value, numbers.Integral) else float(value)


@functools.cache
def _reference(door: str) -> str:
    d = DOORS[door]
    return d.call(_kinds(d.sample)[d.reference or "float"])


@pytest.mark.parametrize(("door", "kind"), CELLS, ids=[f"{d}-{k}" for d, k in CELLS])
def test_numeric_door_cell(door: str, kind: str) -> None:
    d = DOORS[door]
    value = _kinds(d.sample)[kind]
    if kind in d.accepted:
        out = d.call(value)
        if kind == "None":
            return
        expected = _reference(door) if d.reference else d.call(_builtin(value))
        assert out == expected
        return
    with pytest.raises(ValueError) as exc_info:
        d.call(value)
    assert str(exc_info.value) == d.refusal(kind, value)


@pytest.mark.parametrize("door", list(DOORS))
def test_numeric_door_is_not_inert(door: str) -> None:
    """Control: the door's value reaches the output, so an accepted cell's
    equality with the builtin spelling is not vacuous."""
    d = DOORS[door]
    assert d.call(d.control) != _reference(door)


def test_accepted_heatmap_bound_renders_the_bytes_it_did_before_the_predicate() -> None:
    """The builtin-float heatmap cell every accepted kind is compared against
    is pinned to its bytes: routing the bounds through the predicate moved no
    SVG byte for an input that was already accepted."""
    digest = hashlib.sha256(_reference("heatmap_vmin").encode()).hexdigest()
    assert digest == PINNED_HEATMAP_VMIN_SHA256


PINNED_HEATMAP_VMIN_SHA256 = "748f6d0c0dbb64bbff55fe9dc87aebe66cc008d121d8d294078e014f439ed46a"


# --- integer-door ceiling -----------------------------------------------------

_MAX_EXACT_INT = 2**53

# Construction only: the ceiling is the wire's exactness bound, not a
# renderable layout or resample count, so these cells never render.
_INTEGER_BUILDERS: dict[str, tuple[Callable[[Any], Any], Callable[[str, Any], str]]] = {
    "facet_ncols": (
        lambda v: fm.Chart(_POINTS).mark_point().encode(x="x", y="y").facet("g", ncols=v),
        DOORS["facet_ncols"].refusal,
    ),
    "facet_columns": (lambda v: fm.Facet("g", columns=v), DOORS["facet_columns"].refusal),
    "catplot_n_boot": (
        lambda v: fm.catplot(_CAT, x="day", y="tip", kind="point", seed=0, n_boot=v),
        DOORS["catplot_n_boot"].refusal,
    ),
    "mark_text_limit": (_mark_text_limit, DOORS["mark_text_limit"].refusal),
}


@pytest.mark.parametrize("door", list(_INTEGER_BUILDERS))
def test_integer_door_accepts_the_ceiling_and_refuses_one_past_it(door: str) -> None:
    build, refusal = _INTEGER_BUILDERS[door]
    build(_MAX_EXACT_INT)
    with pytest.raises(ValueError) as exc_info:
        build(_MAX_EXACT_INT + 1)
    assert str(exc_info.value) == refusal("huge", _MAX_EXACT_INT + 1)


# --- the unsigned-integer mark door: fractions and negatives ----------------------


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (2.5, f"{_MARK_TEXT_LIMIT} must be a non-negative int, not float; received 2.5"),
        (-1, f"{_MARK_TEXT_LIMIT} must be >= 0; received -1"),
        (np.int64(-1), f"{_MARK_TEXT_LIMIT} must be >= 0; received {np.int64(-1)!r}"),
    ],
    ids=["fraction", "negative", "np_negative"],
)
def test_mark_text_limit_refuses_a_fraction_or_negative_at_construction(
    value: Any, message: str
) -> None:
    """``limit`` is an unsigned integer on the wire: a fraction or a negative
    refuses when the mark is declared, naming ``mark_text(limit=...)``, rather
    than reaching the renderer."""
    with pytest.raises(ValueError) as exc_info:
        fm.Chart(_TEXT).mark_text(limit=value)
    assert str(exc_info.value) == message


def test_mark_text_limit_accepts_zero() -> None:
    """``0`` is an unsigned integer, so the non-negative door accepts it; the
    wire carries it as the same builtin ``0`` a plain ``0`` gives."""
    assert _mark_text_limit(np.int64(0)).to_json() == _mark_text_limit(0).to_json()
    assert json.loads(_mark_text_limit(0).to_json())["mark_style"]["limit"] == 0


# --- catplot ci=/n_boot= on the kinds that draw no interval ----------------------

_NON_AGGREGATING = sorted(_CATPLOT_VALID_KINDS - _CATPLOT_AGGREGATING_KINDS)
_ESTIMATING = ", ".join(repr(k) for k in sorted(_CATPLOT_AGGREGATING_KINDS))
# argument -> (default, a valid non-default value)
_INTERVAL_ARGS = {"ci": (95, 90), "n_boot": (1000, 50)}
_INTERVAL_VALUES = ["default", "valid", "str", "list"]
INTERVAL_CELLS = [
    (kind, arg, value)
    for kind in _NON_AGGREGATING
    for arg in _INTERVAL_ARGS
    for value in _INTERVAL_VALUES
]


def _catplot_kind(kind: str, **kwargs: Any) -> str:
    y = {} if kind == "count" else {"y": "tip"}
    return fm.catplot(_CAT, x="day", kind=kind, seed=0, **y, **kwargs).to_svg()


@pytest.mark.parametrize(
    ("kind", "arg", "value_kind"),
    INTERVAL_CELLS,
    ids=[f"{k}-{a}-{v}" for k, a, v in INTERVAL_CELLS],
)
def test_interval_arg_on_non_aggregating_kind(kind: str, arg: str, value_kind: str) -> None:
    """An explicit ``ci=``/``n_boot=`` on a kind that draws no interval
    refuses rather than being silently ignored; the default stays silent and
    renders byte-identical to omitting the argument."""
    default, valid = _INTERVAL_ARGS[arg]
    value = {"default": default, "valid": valid, "str": str(default), "list": [default]}[value_kind]
    if value_kind == "default":
        assert _catplot_kind(kind, **{arg: value}) == _catplot_kind(kind)
        return
    with pytest.raises(ValueError) as exc_info:
        _catplot_kind(kind, **{arg: value})
    assert str(exc_info.value) == (
        f"catplot({arg}={value!r}) is not supported with kind={kind!r} -- that kind draws no "
        f"confidence interval, so {arg}= would be ignored. Drop {arg}=, or use one of the "
        f"estimating kinds ({_ESTIMATING})."
    )


# --- the mark-style funnel's numeric population ---------------------------------


def test_every_published_numeric_mark_kwarg_is_coerced_at_the_mark_door() -> None:
    """Each name ``mark_numeric_kwargs()`` publishes that some mark honors is
    coerced at that mark's door by the contract of its wire type: ``np.int64``
    is stored as a builtin ``int`` and a non-number refuses in the door's
    voice; a fraction is kept by a real-number field and refused by an
    unsigned-integer one (the ``mark_integer_kwargs()`` subset). Every
    published name is drawn by some mark, so each has a door."""
    published = mark_numeric_kwargs()
    integer = mark_integer_kwargs()
    assert published, "the binding must publish the numeric fields"
    assert integer and set(integer) < set(published)
    assert _INTEGER_MARK_KWARGS == frozenset(integer)
    assert _REAL_MARK_KWARGS == frozenset(published) - frozenset(integer)
    method_marks = sorted(
        name[len("mark_") :]
        for name in dir(fm.Chart)
        if name.startswith("mark_") and drawn_style_fields(name[len("mark_") :]) is not None
    )
    honoring = {name: [m for m in method_marks if name in honored_kwargs(m)] for name in published}
    assert all(honoring.values()), honoring
    for name, marks in honoring.items():
        mark = marks[0]
        stored = MarkBase(mark, **{name: np.int64(2)}).kwargs[name]
        assert stored == 2 and type(stored) is int, (name, stored)
        label = f"mark_{mark}({name}=...)"
        if name in integer:
            expected = _int_refusal(label, "a non-negative int")
            with pytest.raises(ValueError) as exc_info:
                MarkBase(mark, **{name: 2.5})
            assert str(exc_info.value) == expected("fraction", 2.5)
        else:
            expected = _real_refusal(label)
            assert MarkBase(mark, **{name: 2.5}).kwargs[name] == 2.5
        with pytest.raises(ValueError) as exc_info:
            MarkBase(mark, **{name: "2"})
        assert str(exc_info.value) == expected("str", "2")


def test_an_unhonored_kwarg_is_refused_as_unhonored_before_its_value_is_judged() -> None:
    """A kwarg the mark does not honor is refused as not honored whatever its
    value, not as a malformed value the mark would never use."""
    assert "limit" not in honored_kwargs("point")
    with pytest.raises(ValueError) as by_value:
        MarkBase("point", limit="x")
    with pytest.raises(ValueError) as by_name:
        MarkBase("point", limit=2)
    assert str(by_value.value) == str(by_name.value)
    assert str(by_value.value) == _mark_kwarg_error(
        "point", "limit", honored_kwargs("point"), entry="mark_point"
    )


# --- stored and wire types --------------------------------------------------------


def test_integer_doors_store_a_builtin_int() -> None:
    """An integer door stores the builtin ``int`` its predicate returns, not
    the numpy scalar it was given, and the wire carries that integer."""
    by_method = (
        fm.Chart(_POINTS).mark_point().encode(x="x", y="y").facet(field="g", ncols=np.int64(2))
    )
    by_channel = (
        fm.Chart(_POINTS)
        .mark_point()
        .encode(x="x", y="y", facet=fm.Facet("g", columns=np.int64(2)))
    )
    grid = fm.Chart(_POINTS).mark_point().encode(x="x", y="y").facet(row="g", nrows=np.int64(2))
    for chart in (by_method, by_channel):
        assert type(chart._facet.ncols) is int
        assert json.loads(chart.to_json())["facet"]["mode"]["ncols"] == 2
    assert type(grid._facet.nrows) is int

    boot = fm.catplot(_CAT, x="day", y="tip", kind="point", seed=0, n_boot=np.int64(50))
    (summary,) = [t for t in json.loads(boot.to_json())["transforms"] if t["type"] == "summary"]
    assert summary["n_boot"] == 50


def test_a_real_door_keeps_an_integer_an_integer_on_the_wire() -> None:
    """``coerce_finite_real`` returns an integer as ``int``: an ``AxisConfig``
    bound given ``np.int64(2)`` serializes as ``2``, exactly as ``2`` does,
    not as ``2.0``; a float stays a float."""
    as_int = json.dumps(fm.AxisConfig(domain_min=np.int64(2)).to_dict())
    assert as_int == json.dumps(fm.AxisConfig(domain_min=2).to_dict())
    assert type(json.loads(as_int)["domain_min"]) is int
    as_float = json.loads(json.dumps(fm.AxisConfig(domain_min=np.float64(2)).to_dict()))
    assert type(as_float["domain_min"]) is float
