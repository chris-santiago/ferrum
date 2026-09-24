"""Numeric door x input kind matrix: every numeric door answers "is this a number" alike.

The door population is derived, not hand-listed: every hit of
``isinstance(... numbers.Real | (int, float) | int)`` and ``math.isfinite`` in
``src/ferrum`` was classified, plus ``catplot(ci=)``'s ``float()`` probe, and
the hits that decide whether a user-facing numeric argument is a number are
the doors below, each exercised through its public entry. Real-number doors
store what ``ferrum._validate.coerce_finite_real`` returns; integer doors ask
``validate_positive_int``. So every door accepts ``int``, ``float``, numpy
scalars and (for real-number doors) ``Decimal``, and refuses ``bool``,
``str``, ``list``, ``nan``, ``inf``, an integer a float cannot hold
(``10**400``) and a signaling NaN with a ``ValueError`` naming the public door
and the value received. ``None`` is each door's own documented unset sentinel
where it has one (heatmap bounds, ``AxisConfig.domain_min``, the t-SNE
default, ``ncols``/``columns``), and a refusal where the argument is required.

Every accepted cell must produce the same output as the builtin spelling of
the same value, and each door carries a control value whose output must
differ (the argument is not inert), so an "accepted" cell cannot pass
vacuously.
"""

from __future__ import annotations

import decimal
import functools
import hashlib
import warnings
from typing import Any, Callable, NamedTuple

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum.plots.distribution import _CATPLOT_AGGREGATING_KINDS, _CATPLOT_VALID_KINDS

_NON_FINITE = {"nan", "inf", "huge", "sNaN"}


def _kinds(v: int) -> dict[str, Any]:
    """The thirteen input kinds, each spelling the value *v* where it can."""
    return {
        "int": v,
        "float": float(v),
        "np_int": np.int64(v),
        "np_float": np.float64(v),
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


def _int_refusal(label: str) -> Callable[[str, Any], str]:
    """``validate_positive_int``'s refusals, in the door's voice."""

    def refusal(kind: str, v: Any) -> str:
        if kind == "huge":
            return (
                f"{label} must be at most 2**53, the largest integer a float holds exactly; "
                f"received {v!r}"
            )
        return f"{label} must be a positive int, not {type(v).__name__}; received {v!r}"

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
    reference: str  # the kind whose output every accepted kind must equal
    control: Any  # a different valid value whose output must differ
    refusal: Callable[[str, Any], str]
    accepted: frozenset[str]


_REAL = frozenset({"int", "float", "np_int", "np_float", "Decimal", "None"})
_INTEGER = frozenset({"int", "np_int", "None"})
_TSNE_LABEL = "ModelSource.embeddings(method='tsne', perplexity=...)"

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
    "facet_ncols": Door(
        _facet_ncols, 2, "int", None, _int_refusal("Chart.facet(ncols=...)"), _INTEGER
    ),
    "facet_columns": Door(
        _facet_columns, 2, "int", None, _int_refusal("Facet(columns=...)"), _INTEGER
    ),
    "catplot_n_boot": Door(
        _catplot("n_boot"), 50, "int", 1000, _n_boot_refusal, _INTEGER - {"None"}
    ),
}

CELLS = [(door, kind) for door in DOORS for kind in KINDS]


@functools.cache
def _reference(door: str) -> str:
    d = DOORS[door]
    return d.call(_kinds(d.sample)[d.reference])


@pytest.mark.parametrize(("door", "kind"), CELLS, ids=[f"{d}-{k}" for d, k in CELLS])
def test_numeric_door_cell(door: str, kind: str) -> None:
    d = DOORS[door]
    value = _kinds(d.sample)[kind]
    if kind in d.accepted:
        out = d.call(value)
        if kind != "None":
            assert out == _reference(door)
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


PINNED_HEATMAP_VMIN_SHA256 = "fc16742a97c0770bfb5bcbce6100824825cf31c896247dde3de4f6a163428d87"


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
}


@pytest.mark.parametrize("door", list(_INTEGER_BUILDERS))
def test_integer_door_accepts_the_ceiling_and_refuses_one_past_it(door: str) -> None:
    build, refusal = _INTEGER_BUILDERS[door]
    build(_MAX_EXACT_INT)
    with pytest.raises(ValueError) as exc_info:
        build(_MAX_EXACT_INT + 1)
    assert str(exc_info.value) == refusal("huge", _MAX_EXACT_INT + 1)


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
