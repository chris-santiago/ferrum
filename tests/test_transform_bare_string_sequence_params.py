"""Batch-D T12: every sequence-of-strings parameter on a ``transform_*``
wrapper refuses a bare string instead of char-splitting it.

`list("cat")` is `['c', 'a', 't']` — three one-character column names that
are perfectly well-typed on the wire, so a bare string passed to a parameter
that wants a sequence of column names is a *silent* corruption, not an error.
Round 1 of this task fixed nine `groupby=` wrappers and missed
`transform_pivot`; round 2 fixed `transform_pivot` and `keyvals=` and missed
`transform_fold(fields=)`, `transform_flatten(fields=)`,
`transform_window(sort=)`, `transform_stack(sort=)`; the round-3 sweep below
additionally found `transform_flatten(as_=)`, which reaches
``_validate_as_seq`` whose per-element checks accept every character of
``as_="tag"`` as a valid one-letter output column. Three rounds of
hand-enumerated "family-wide" claims, three rounds of missed members.

So this file does not enumerate the family. It **derives** it:
``_sequence_of_strings_params()`` resolves every ``transform_*`` function's
annotations and selects the parameters whose declared type is a sequence
whose elements are ``str`` (or ``Any``) — precisely the parameters where a
char-split produces type-valid garbage. Every derived parameter must refuse a
bare string and must still accept a real sequence. A parameter added
tomorrow with that annotation joins the parametrize automatically; if its
wrapper has no entry in ``_MINIMAL_CALLS`` the coverage test fails loudly
(an ``AssertionError``, not a skip) rather than under-covering silently.

Numeric-element sequence parameters (``transform_density(extent=)``,
``transform_window(frame=)``) are outside the derived set by the same rule
that defines it, and ``test_numeric_sequence_params_are_refused_downstream``
pins *why* that exclusion is safe rather than asserting it in prose: their
char-split reaches a typed wire boundary that refuses it loudly.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Sequence
from typing import Any

import polars as pl
import pytest

import ferrum as fm
import ferrum.transforms as _transforms_mod
from ferrum.parameter import Parameter

#: The value the tests pass for a derived parameter's "real sequence" case.
#: Two elements so it satisfies both variable-length parameters and the
#: fixed-arity ``as_`` pairs.
_VALID_SAMPLE = ["zzz_col_a", "zzz_col_b"]


def _is_sequence_of_strings(annotation: Any) -> bool:
    """True if *annotation* declares a sequence whose elements are ``str``.

    ``X | None`` is unwrapped first. ``Sequence[Any]``
    (``transform_impute(keyvals=)``) counts: its elements are unconstrained,
    so one-character strings pass just as silently. Sequences of numbers
    (``tuple[float, float]``) do not — see the module docstring.
    """
    origin = typing.get_origin(annotation)
    if origin is typing.Union or type(annotation) is type(int | None):
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        return any(_is_sequence_of_strings(a) for a in members)
    if origin is None or not (
        origin is tuple
        or origin is list
        or isinstance(origin, type)
        and issubclass(origin, Sequence)
    ):
        return False
    args = [a for a in typing.get_args(annotation) if a is not Ellipsis]
    return bool(args) and all(a is str or a is Any for a in args)


def _sequence_of_strings_params() -> list[tuple[str, str]]:
    """Every ``(transform_* function, parameter)`` in :mod:`ferrum.transforms`
    whose declared type is a sequence of strings, derived from the module's
    own annotations rather than enumerated by hand."""
    found: list[tuple[str, str]] = []
    for name, fn in inspect.getmembers(_transforms_mod, inspect.isfunction):
        if not name.startswith("transform_"):
            continue
        hints = typing.get_type_hints(
            fn, globalns=vars(_transforms_mod), localns={"Parameter": Parameter}
        )
        for param in inspect.signature(fn).parameters:
            if _is_sequence_of_strings(hints.get(param)):
                found.append((name, param))
    return sorted(found)


#: wrapper name -> (positional args, keyword args) for a minimal otherwise-valid
#: call.  Positional only where the wrapper takes ``*args`` (the aggregate/window
#: op dicts); everything else goes in the keyword half so the derived parameter
#: under test can override its entry here, several of them being the wrapper's
#: own leading positional parameter (``fields``, ``groupby``).
_MINIMAL_CALLS: dict[str, tuple[tuple, dict]] = {
    "transform_aggregate": (({"field": "x", "fn": "mean", "as": "m"},), {}),
    "transform_density": ((), {"field": "x"}),
    "transform_flatten": ((), {"fields": ["tags"]}),
    "transform_fold": ((), {"fields": ["col_a", "col_b"]}),
    "transform_impute": ((), {"field": "sales"}),
    "transform_join_aggregate": (({"field": "x", "fn": "mean", "as": "m"},), {}),
    "transform_loess": ((), {"x": "x", "y": "y"}),
    "transform_pivot": ((), {"field": "category", "value": "amount"}),
    "transform_regression": ((), {"x": "x", "y": "y"}),
    "transform_stack": ((), {"field": "sales", "groupby": ["g"]}),
    "transform_top_k": ((), {"n": 5, "field": "revenue"}),
    "transform_window": (({"op": "row_number", "as": "r"},), {}),
}

_DERIVED_PARAMS = _sequence_of_strings_params()
_DERIVED_WRAPPERS = sorted({name for name, _param in _DERIVED_PARAMS})


def _call_with(name: str, param: str, value):
    if name not in _MINIMAL_CALLS:
        raise AssertionError(
            f"ferrum.transforms.{name} declares a sequence-of-strings parameter "
            f"({param}) but has no entry in _MINIMAL_CALLS "
            "(tests/test_transform_bare_string_sequence_params.py) -- add one so "
            "this derived test can exercise it. This is the guard against "
            "silently missing a new wrapper the way transform_pivot was missed "
            "in round 1 and transform_fold in round 2."
        )
    args, kwargs = _MINIMAL_CALLS[name]
    return getattr(_transforms_mod, name)(*args, **{**kwargs, param: value})


# ---------------------------------------------------------------------------
# The derivation itself is load-bearing, so it is checked before it is used
# ---------------------------------------------------------------------------


def test_derivation_finds_the_known_members_of_the_family():
    """Guards the *classifier*, not the family: if `_is_sequence_of_strings`
    ever stopped recognising these annotation shapes the parametrized tests
    below would silently shrink to nothing and still pass."""
    assert ("transform_stack", "groupby") in _DERIVED_PARAMS  # Sequence[str]
    assert ("transform_window", "sort") in _DERIVED_PARAMS  # Sequence[str] | None
    assert ("transform_impute", "keyvals") in _DERIVED_PARAMS  # Sequence[Any] | None
    assert ("transform_fold", "as_") in _DERIVED_PARAMS  # tuple[str, str]
    assert ("transform_flatten", "as_") in _DERIVED_PARAMS  # Sequence[str] | None
    # Not a sequence: a direction keyword that legitimately takes a string.
    assert ("transform_top_k", "sort") not in _DERIVED_PARAMS
    # Not a sequence of *strings*: numeric extents, see the downstream pin.
    assert ("transform_density", "extent") not in _DERIVED_PARAMS
    assert ("transform_window", "frame") not in _DERIVED_PARAMS


def test_every_derived_wrapper_is_covered_by_the_minimal_call_table():
    """Fails loudly (rather than silently under-parametrizing) if a wrapper
    with a sequence-of-strings parameter has no minimal call."""
    missing = sorted(set(_DERIVED_WRAPPERS) - set(_MINIMAL_CALLS))
    assert not missing, (
        f"discovered wrapper(s) with a sequence-of-strings parameter and no "
        f"_MINIMAL_CALLS entry: {missing}"
    )


def test_every_groupby_wrapper_is_in_the_derived_set():
    """The round-1/round-2 `groupby=` family is a subset of the derived set,
    so this file covers it without a second hand-maintained list."""
    declared_groupby = sorted(
        name
        for name, fn in inspect.getmembers(_transforms_mod, inspect.isfunction)
        if name.startswith("transform_") and "groupby" in inspect.signature(fn).parameters
    )
    derived_groupby = sorted(name for name, param in _DERIVED_PARAMS if param == "groupby")
    assert derived_groupby == declared_groupby


# ---------------------------------------------------------------------------
# The family behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, param", _DERIVED_PARAMS, ids=lambda v: v)
def test_bare_string_refused_across_the_family(name, param):
    """A bare string refuses at construction time, not by silently
    char-splitting into per-letter columns."""
    with pytest.raises(ValueError, match="bare string"):
        _call_with(name, param, "cat")


@pytest.mark.parametrize("name, param", _DERIVED_PARAMS, ids=lambda v: v)
def test_real_sequence_still_reaches_the_wire(name, param):
    """The guard rejects only bare strings: a real sequence still lands on the
    wire under the parameter's own key, unchanged."""
    spec = _call_with(name, param, list(_VALID_SAMPLE))
    assert spec[param] == _VALID_SAMPLE


@pytest.mark.parametrize(
    "transform, expected_type",
    [
        (lambda: fm.transform_density("x", extent="ab"), "f64"),
        (lambda: fm.transform_window({"op": "row_number", "as": "r"}, frame="ab"), "i64"),
    ],
)
def test_numeric_sequence_params_are_refused_downstream(transform, expected_type):
    """Why numeric-element sequence parameters are outside the derived set,
    pinned rather than asserted: their char-split does reach the wire, but the
    typed deserialization boundary refuses the resulting strings loudly, so
    there is no silent-corruption mode to guard against at the wrapper."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match=f'invalid type: string "a", expected {expected_type}'):
        fm.Chart(df).transform(transform()).mark_point().encode(x="x", y="x").to_svg()


# ---------------------------------------------------------------------------
# Site-specific RED proofs (the exact repros the review cycles reported)
# ---------------------------------------------------------------------------


def test_pivot_no_longer_char_splits_a_bare_string_groupby():
    """Cycle-1 spec-reviewer repro through the public entry point."""
    with pytest.raises(ValueError, match="transform_pivot.*groupby.*bare string"):
        fm.transform_pivot("cat", "amt", groupby="date")
    assert fm.transform_pivot("cat", "amt", groupby=["date"])["groupby"] == ["date"]


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: fm.transform_fold("cat"), "transform_fold: fields"),
        (lambda: fm.transform_flatten("cat"), "transform_flatten: fields"),
        (lambda: fm.transform_flatten(["tags"], as_="tag"), "transform_flatten: as_"),
        (
            lambda: fm.transform_window({"op": "row_number", "as": "r"}, sort="date"),
            "transform_window: sort",
        ),
        (
            lambda: fm.transform_stack("sales", groupby=["g"], sort="date"),
            "transform_stack: sort",
        ),
    ],
)
def test_cycle_2_char_split_repros_now_refuse(call, match):
    """The five live char-splits measured on the round-2 tree, each through
    the public ``fm.`` entry point: `transform_fold("cat")` returned
    `{'fields': ['c', 'a', 't']}` and its siblings likewise."""
    with pytest.raises(ValueError, match=match):
        call()


def test_reject_helper_is_the_single_shared_mechanism():
    """Every refusal above comes from one predicate, not a copy per site."""
    from ferrum.transforms import _reject_bare_string, _reject_bare_string_groupby

    with pytest.raises(ValueError, match="transform_regression.*groupby.*bare string"):
        _reject_bare_string_groupby("cat", where="transform_regression")
    with pytest.raises(ValueError, match="transform_fold: fields.*bare string"):
        _reject_bare_string("cat", where="transform_fold", param="fields", noun="column names")

    # None and real sequences are not rejected -- the helper only guards
    # against the bare-string mistake, leaving None-handling and
    # materialization to each call site.
    _reject_bare_string_groupby(None, where="transform_regression")
    _reject_bare_string_groupby(["cat"], where="transform_regression")
    _reject_bare_string_groupby(("cat", "dog"), where="transform_regression")
