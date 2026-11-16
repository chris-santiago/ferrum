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

Numeric-element sequence parameters (``transform_density(extent=)``) are outside
the derived set by the same rule that defines it: each has its own typed pair
reader that refuses a string at construction
(``tests/test_ordered_pair_params.py`` pins ``extent="ab"``).
``transform_window(frame=)`` has its own typed check at construction too.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
import typing
from collections import OrderedDict, defaultdict
from collections.abc import Sequence, Set
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

import ferrum as fm
import ferrum.transforms as _transforms_mod
from ferrum import _core
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
    # Not a sequence of *strings*: numeric extents, which have their own pair reader.
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


#: The noun each family parameter's refusals call its elements.
_NOUNS = {
    "groupby": "column names",
    "fields": "column names",
    "sort": "column names",
    "as_": "output-column names",
    "keyvals": "key values",
}


class _SetSubclass(Set):
    """A third-party ``collections.abc.Set`` that is none of the builtin sets."""

    def __init__(self, *items: str) -> None:
        self._items = items

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


def _bytes_text(door: str, param: str, noun: str, value: object) -> str:
    return (
        f"{door}: {param} must be a sequence of {noun} (list or tuple), got "
        f"{type(value).__name__} {value!r}; iterating it yields integers"
    )


def _set_text(door: str, param: str, noun: str, value: object) -> str:
    return (
        f"{door}: {param} must be an ordered sequence of {noun}, got "
        f"{type(value).__name__} {value!r}; a set has no order, so the output would "
        "change from run to run; pass a list, e.g. sorted(...)"
    )


def _bare_string_text(door: str, param: str, noun: str, value: str) -> str:
    return (
        f"{door}: {param} must be a sequence of {noun} (list or tuple), got a bare string "
        f"{value!r}; iterating a string yields individual characters"
    )


def _mapping_text(door: str, param: str, noun: str, value: object) -> str:
    if param == "keyvals":
        return (
            f"{door}: keyvals must be a sequence of {noun} or a "
            f"{{'start', 'stop', 'step'}} dict, got {type(value).__name__} {value!r}; "
            "pass a list, or dict(...) for the sequence form"
        )
    return (
        f"{door}: {param} must be a sequence of {noun}, got "
        f"{type(value).__name__} {value!r}; pass a list of its keys, e.g. list(...)"
    )


#: (id, value, the message builder that refuses it)
_REFUSED = [
    ("set", {"zzz_col_a", "zzz_col_b"}, _set_text),
    ("frozenset", frozenset({"zzz_col_a", "zzz_col_b"}), _set_text),
    ("abc_set", _SetSubclass("zzz_col_a", "zzz_col_b"), _set_text),
    ("bytes", b"ab", _bytes_text),
    ("bytearray", bytearray(b"ab"), _bytes_text),
    ("memoryview", memoryview(b"ab"), _bytes_text),
    ("mappingproxy", MappingProxyType({"zzz_col_a": 1, "zzz_col_b": 2}), _mapping_text),
]


#: Mappings with a dict form on no parameter but ``keyvals``: refused everywhere else, so
#: nothing reads a dict's keys.
_REFUSED_WITHOUT_DICT_FORM = [
    ("dict", {"zzz_col_a": 1, "zzz_col_b": 2}, _mapping_text),
    ("empty_dict", {}, _mapping_text),
    ("ordered_dict", OrderedDict(zzz_col_a=1, zzz_col_b=2), _mapping_text),
    ("defaultdict", defaultdict(int, {"zzz_col_a": 1, "zzz_col_b": 2}), _mapping_text),
]


@pytest.mark.parametrize("name, param", _DERIVED_PARAMS, ids=lambda v: v)
@pytest.mark.parametrize("value, builder", [v[1:] for v in _REFUSED], ids=[v[0] for v in _REFUSED])
def test_an_unordered_or_byte_collection_is_refused_across_the_family(name, param, value, builder):
    """A set, a non-dict mapping and a byte string refuse at
    construction in the door's own words, never read in hash order or as integers."""
    with pytest.raises(ValueError) as excinfo:
        _call_with(name, param, value)
    assert str(excinfo.value) == builder(name, param, _NOUNS[param], value)


_NO_DICT_FORM_PARAMS = [(n, p) for n, p in _DERIVED_PARAMS if p != "keyvals"]


@pytest.mark.parametrize("name, param", _NO_DICT_FORM_PARAMS, ids=lambda v: v)
@pytest.mark.parametrize(
    "value, builder",
    [v[1:] for v in _REFUSED_WITHOUT_DICT_FORM],
    ids=[v[0] for v in _REFUSED_WITHOUT_DICT_FORM],
)
def test_a_dict_is_refused_where_the_parameter_has_no_dict_form(name, param, value, builder):
    """A dict, an OrderedDict, a defaultdict and an empty dict refuse in the mapping words
    on ``groupby``, ``fields``, ``sort`` and ``as_``, never read as their keys."""
    with pytest.raises(ValueError) as excinfo:
        _call_with(name, param, value)
    assert str(excinfo.value) == builder(name, param, _NOUNS[param], value)


@pytest.mark.parametrize(
    "form", [{"start": 0, "stop": 3}, OrderedDict(start=0, stop=3)], ids=["dict", "OrderedDict"]
)
def test_a_dict_keyvals_stays_the_sequence_form(form):
    assert fm.transform_impute("v", key="t", keyvals=form)["keyvals"] == {"start": 0, "stop": 3}
    assert fm.Y("v", impute={"keyvals": form})._kwargs["impute"]["keyvals"] == {
        "start": 0,
        "stop": 3,
    }


def test_a_dict_keyvals_with_an_unknown_key_keeps_its_own_refusal():
    with pytest.raises(ValueError) as excinfo:
        fm.transform_impute("v", key="t", keyvals={1.0: 0})
    assert str(excinfo.value) == (
        "transform_impute: keyvals={...} has unknown key 1.0; expected start|stop|step"
    )
    with pytest.raises(ValueError) as excinfo:
        fm.Y("v", impute={"keyvals": {1.0: 0}})
    assert str(excinfo.value) == (
        "Y(impute=...): keyvals={...} has unknown key 1.0; expected start|stop|step"
    )


def test_a_mapping_proxy_keyvals_is_refused_on_both_impute_doors():
    value = MappingProxyType({"start": 0, "stop": 3})
    expected = _mapping_text("transform_impute", "keyvals", "key values", value)
    with pytest.raises(ValueError) as excinfo:
        fm.transform_impute("v", key="t", keyvals=value)
    assert str(excinfo.value) == expected
    with pytest.raises(ValueError) as excinfo:
        fm.Y("v", impute={"keyvals": value})
    assert str(excinfo.value) == _mapping_text("Y(impute=...)", "keyvals", "key values", value)


@pytest.mark.parametrize("name, param", _DERIVED_PARAMS, ids=lambda v: v)
@pytest.mark.parametrize(
    "make",
    [
        lambda items: tuple(items),
        lambda items: (i for i in items),
        lambda items: map(str, items),
        lambda items: np.array(items),
        lambda items: pl.Series(items),
        lambda items: pd.Series(items),
    ],
    ids=["tuple", "generator", "map", "numpy", "polars", "pandas"],
)
def test_every_ordered_sequence_kind_is_still_read_once_and_unchanged(name, param, make):
    spec = _call_with(name, param, make(list(_VALID_SAMPLE)))
    assert spec[param] == _VALID_SAMPLE


@pytest.mark.parametrize("name, param", _DERIVED_PARAMS, ids=lambda v: v)
def test_the_remedy_a_mapping_refusal_names_is_accepted(name, param):
    """Following the named remedy (``list(...)`` of the keys; ``dict(...)`` for
    ``keyvals``' sequence form) is read as written."""
    proxy = MappingProxyType({"zzz_col_a": 1, "zzz_col_b": 2})
    assert _call_with(name, param, list(proxy))[param] == _VALID_SAMPLE
    assert _call_with(name, param, list({"zzz_col_a": 1, "zzz_col_b": 2}))[param] == _VALID_SAMPLE
    if param == "keyvals":
        form = dict(MappingProxyType({"start": 0, "stop": 3}))
        assert _call_with(name, param, form)[param] == {"start": 0, "stop": 3}


def _outcome(call):
    """The value a call returns, or the type of the error it raises."""
    try:
        return ("ok", call())
    except Exception as exc:  # the comparison is between two outcomes, whatever they are
        return ("raised", type(exc))


_VIEWS = {
    "keys": lambda d: d.keys(),
    "values": lambda d: d.values(),
    "items": lambda d: d.items(),
}


@pytest.mark.parametrize("name, param", _DERIVED_PARAMS, ids=lambda v: v)
@pytest.mark.parametrize("view", sorted(_VIEWS))
def test_a_dict_view_is_read_in_order_like_a_list(name, param, view):
    """``keys()``, ``values()`` and ``items()`` are ordered, so each is read exactly as
    ``list(view)`` is, on every sequence parameter: same result, or the same refusal."""
    source = {"zzz_col_b": "y", "zzz_col_a": "x"}
    as_view = _outcome(lambda: _call_with(name, param, _VIEWS[view](source)))
    as_list = _outcome(lambda: _call_with(name, param, list(_VIEWS[view](source))))
    assert as_view == as_list
    assert not (as_view[0] == "raised" and "has no order" in str(as_view[1]))


def test_a_dict_view_reaches_the_wire_in_insertion_order():
    assert _call_with("transform_fold", "fields", {"zzz_col_b": 1, "zzz_col_a": 2}.keys())[
        "fields"
    ] == ["zzz_col_b", "zzz_col_a"]
    assert _call_with("transform_fold", "fields", {"zzz_col_b": 1, "zzz_col_a": 2}.values())[
        "fields"
    ] == [1, 2]
    assert _call_with("transform_stack", "groupby", {"g": 1, "h": 2}.keys())["groupby"] == [
        "g",
        "h",
    ]


def test_impute_snapshot_reads_a_dict_view_once_into_a_list():
    """A view is copied into a list by the snapshot, so a later change to the dict it came
    from cannot move the channel's key order."""
    from ferrum.transforms import _impute_snapshot

    source = {2: 10, 0: 11}
    snapshot = _impute_snapshot({"keyvals": source.keys()})
    source[9] = 1
    assert snapshot["keyvals"] == [2, 0]


def test_a_set_keyvals_is_still_refused_on_both_impute_doors():
    with pytest.raises(ValueError, match="has no order"):
        fm.transform_impute("v", key="t", keyvals={1, 2})
    with pytest.raises(ValueError, match="has no order"):
        fm.Y("v", impute={"keyvals": frozenset({1, 2})})


def test_a_range_keyvals_is_read_once_and_unchanged():
    assert fm.transform_impute("v", key="t", keyvals=range(3))["keyvals"] == [0, 1, 2]


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


@pytest.mark.parametrize(
    "call, door, param, noun",
    [
        (
            lambda v: fm.transform_aggregate({"field": "x", "fn": "mean", "as": "m"}, groupby=v),
            "transform_aggregate",
            "groupby",
            "column names",
        ),
        (
            lambda v: fm.transform_stack("s", groupby=v),
            "transform_stack",
            "groupby",
            "column names",
        ),
        (
            lambda v: fm.transform_impute("v", key="t", groupby=v),
            "transform_impute",
            "groupby",
            "column names",
        ),
        (
            lambda v: fm.transform_window({"op": "row_number", "as": "r"}, sort=v),
            "transform_window",
            "sort",
            "column names",
        ),
        (
            lambda v: fm.transform_stack("s", groupby=["g"], sort=v),
            "transform_stack",
            "sort",
            "column names",
        ),
        (lambda v: fm.transform_fold(v), "transform_fold", "fields", "column names"),
        (lambda v: fm.transform_flatten(v), "transform_flatten", "fields", "column names"),
        (
            lambda v: fm.transform_flatten(["t"], as_=v),
            "transform_flatten",
            "as_",
            "output-column names",
        ),
    ],
)
def test_the_public_entry_point_refuses_a_dict_in_the_mapping_words(call, door, param, noun):
    value = {"g": 1}
    with pytest.raises(ValueError) as excinfo:
        call(value)
    assert str(excinfo.value) == _mapping_text(door, param, noun, value)


def test_a_dict_groupby_remedy_equals_the_one_column_list():
    assert fm.transform_stack("s", groupby=list({"g": 1}))["groupby"] == ["g"]
    assert fm.transform_stack("s", groupby=list({"g": 1})) == fm.transform_stack("s", groupby=["g"])


def test_reject_helper_is_the_single_shared_mechanism():
    """Every refusal above comes from one predicate, not a copy per site."""
    from ferrum.transforms import _reject_bare_string, _reject_bare_string_groupby

    with pytest.raises(ValueError) as excinfo:
        _reject_bare_string_groupby("cat", where="transform_regression")
    assert str(excinfo.value) == _bare_string_text(
        "transform_regression", "groupby", "column names", "cat"
    )
    with pytest.raises(ValueError) as excinfo:
        _reject_bare_string("cat", where="transform_fold", param="fields", noun="column names")
    assert str(excinfo.value) == _bare_string_text(
        "transform_fold", "fields", "column names", "cat"
    )

    # None and real sequences are not rejected -- the helper only guards
    # against the bare-string mistake, leaving None-handling and
    # materialization to each call site.
    _reject_bare_string_groupby(None, where="transform_regression")
    _reject_bare_string_groupby(["cat"], where="transform_regression")
    _reject_bare_string_groupby(("cat", "dog"), where="transform_regression")
    with pytest.raises(ValueError) as excinfo:
        _reject_bare_string_groupby({"g": 1}, where="transform_regression")
    assert str(excinfo.value) == _mapping_text(
        "transform_regression", "groupby", "column names", {"g": 1}
    )


# ---------------------------------------------------------------------------
# The written-step constructors speak in the same voice
# ---------------------------------------------------------------------------

#: Every constructor that takes ``groupby=``, with the arguments its minimal call needs.
_CONSTRUCTORS: dict[str, tuple[tuple, dict]] = {
    "Aggregate": (([_core.AggregateOp("x", "mean", "m")],), {}),
    "Bin": (("x",), {}),
    "BoxStats": (("x",), {}),
    "ErrorExtent": (("x",), {}),
    "Glm": (("x", "y"), {}),
    "Kde": (("x",), {}),
    "Kde2D": (("x", "y"), {}),
    "LetterValue": (("x",), {}),
    "Logistic": (("x", "y"), {}),
    "Outliers": (("x",), {}),
    "QQ": (("x",), {}),
    "Robust": (("x", "y"), {}),
    "Smooth": (("x", "y"), {}),
    "Summary": (("x",), {}),
    "TopK": ((3, "x"), {}),
    "Violin": (("x",), {}),
}


def _construct(name: str, groupby: object):
    args, kwargs = _CONSTRUCTORS[name]
    return getattr(_core, name)(*args, **kwargs, groupby=groupby)


def test_every_constructor_that_takes_groupby_is_in_the_table():
    """A constructor added later with a ``groupby=`` parameter fails here until it is covered."""
    declared = sorted(
        name
        for name in dir(_core)
        if name[:1].isupper()
        and isinstance(getattr(_core, name), type)
        and "groupby" in (getattr(getattr(_core, name), "__text_signature__", None) or "")
    )
    assert set(declared) <= set(_CONSTRUCTORS)


@pytest.mark.parametrize("name", sorted(_CONSTRUCTORS))
@pytest.mark.parametrize(
    "value, builder",
    [
        v[1:]
        for v in _REFUSED
        if v[0]
        in {"set", "frozenset", "abc_set", "bytes", "bytearray", "memoryview", "mappingproxy"}
    ],
    ids=["set", "frozenset", "abc_set", "bytes", "bytearray", "memoryview", "mappingproxy"],
)
def test_a_constructor_refuses_an_unordered_or_byte_groupby_in_the_transform_words(
    name, value, builder
):
    with pytest.raises(ValueError) as excinfo:
        _construct(name, value)
    assert str(excinfo.value) == builder(name, "groupby", "column names", value)


@pytest.mark.parametrize("name", sorted(_CONSTRUCTORS))
@pytest.mark.parametrize(
    "value, builder",
    [v[1:] for v in _REFUSED_WITHOUT_DICT_FORM],
    ids=[v[0] for v in _REFUSED_WITHOUT_DICT_FORM],
)
def test_a_constructor_refuses_a_dict_groupby_in_the_transform_words(name, value, builder):
    """The PyO3 constructors share the wrappers' refusal: a dict is no more a list of columns here."""
    with pytest.raises(ValueError) as excinfo:
        _construct(name, value)
    assert str(excinfo.value) == builder(name, "groupby", "column names", value)


@pytest.mark.parametrize("name", sorted(_CONSTRUCTORS))
def test_a_constructor_refuses_a_bare_string_groupby_in_the_transform_words(name):
    with pytest.raises(ValueError) as excinfo:
        _construct(name, "cat")
    assert str(excinfo.value) == (
        f"{name}: groupby must be a sequence of column names (list or tuple), "
        "got a bare string 'cat'; iterating a string yields individual characters"
    )


@pytest.mark.parametrize("name", sorted(_CONSTRUCTORS))
@pytest.mark.parametrize("make", [list, tuple], ids=["list", "tuple"])
def test_a_constructor_still_takes_an_ordered_groupby(name, make):
    _construct(name, make(["g", "h"]))


@pytest.mark.parametrize("name", sorted(_CONSTRUCTORS))
def test_a_constructor_takes_no_groupby_by_default(name):
    args, kwargs = _CONSTRUCTORS[name]
    getattr(_core, name)(*args, **kwargs)


@pytest.mark.parametrize("name", sorted(_CONSTRUCTORS))
def test_a_constructor_reads_a_dict_view_groupby_like_a_list(name):
    """A ``keys()`` view reaches the constructor as ``list(d)`` does, in insertion order."""
    source = {"h": 1, "g": 2}
    assert repr(_construct(name, source.keys())) == repr(_construct(name, list(source)))
    assert repr(_construct(name, source.keys())) != repr(_construct(name, ["g", "h"]))


@pytest.mark.parametrize(
    "mark, kwargs, encoding",
    [
        ("mark_density", {}, {"x": "x"}),
        ("mark_histogram", {}, {"x": "x"}),
        ("mark_smooth", {"method": "lm"}, {"x": "x", "y": "y"}),
        ("mark_contour", {}, {"x": "x", "y": "y"}),
    ],
)
def test_a_mark_groupby_keeps_one_column_as_a_string_and_refuses_a_set(mark, kwargs, encoding):
    """A mark's own ``groupby=`` documents ``str or list``; a bare string is one column."""
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 2.5, 4.0, 3.0, 6.0],
            "g": list("aaabbb"),
        }
    )
    base = fm.Chart(df).encode(**encoding, color="g:N")
    getattr(base, mark)(groupby="g", **kwargs).to_svg()
    with pytest.raises(ValueError) as excinfo:
        getattr(base, mark)(groupby={"g"}, **kwargs).to_svg()
    assert str(excinfo.value) == _set_text(mark, "groupby", "column names", {"g"})


#: Each mark door that reads ``groupby=``: (mark, the call's kwargs, the chart's encoding).
#: ``mark_smooth`` reaches ``groupby`` on two paths, with and without a confidence band.
_MARK_DOORS = {
    "mark_density": ("mark_density", {}, {"x": "x"}),
    "mark_histogram": ("mark_histogram", {}, {"x": "x"}),
    "mark_smooth_band": ("mark_smooth", {"method": "lm", "ci": 0.95}, {"x": "x", "y": "y"}),
    "mark_smooth_line": ("mark_smooth", {"method": "lm", "ci": None}, {"x": "x", "y": "y"}),
    "mark_contour": ("mark_contour", {}, {"x": "x", "y": "y"}),
}

_MAPPING_ROWS = _REFUSED_WITHOUT_DICT_FORM + [row for row in _REFUSED if row[0] == "mappingproxy"]


@pytest.mark.parametrize("door", sorted(_MARK_DOORS))
@pytest.mark.parametrize(
    "value", [row[1] for row in _MAPPING_ROWS], ids=[row[0] for row in _MAPPING_ROWS]
)
def test_a_mark_groupby_refuses_a_mapping_in_the_mapping_words(door, value):
    """A mark reads ``groupby=`` through the same guard as the transforms: a mapping,
    a dict included, is refused in the mark's own name instead of read as its keys."""
    mark, kwargs, encoding = _MARK_DOORS[door]
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 2.5, 4.0, 3.0, 6.0],
            "g": list("aaabbb"),
        }
    )
    chart = getattr(fm.Chart(df).encode(**encoding, color="g:N"), mark)(groupby=value, **kwargs)
    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()
    assert str(excinfo.value) == _mapping_text(mark, "groupby", "column names", value)


# ---------------------------------------------------------------------------
# The other constructors that take a list of column names
# ---------------------------------------------------------------------------

#: (class, parameter) -> (noun in the refusal, builder of an otherwise valid instance).
_NAME_LIST_DOORS: dict[tuple[str, str], tuple[str, Any]] = {
    ("Flatten", "fields"): ("column names", lambda v: _core.Flatten(v)),
    ("Flatten", "as_"): ("output-column names", lambda v: _core.Flatten(["t"], as_=v)),
    ("Unpivot", "id_vars"): ("column names", lambda v: _core.Unpivot(id_vars=v)),
    ("Unpivot", "value_vars"): ("column names", lambda v: _core.Unpivot(value_vars=v)),
}

#: Constructor parameters annotated as a list of strings that hold values, not column names.
_VALUE_LISTS = {
    ("OrdinalScale", "domain"),
    ("BandScale", "domain"),
    ("PointScale", "domain"),
    ("QuantizeScale", "range"),
}

_STR_LIST_ANNOTATION = re.compile(r"^(Optional\[)?(List|list|Sequence)\[str\]\]?( \| None)?$")


def _stub_string_list_parameters() -> set[tuple[str, str]]:
    """Every ``(class, parameter)`` that ``_core.pyi`` annotates as a list of strings."""
    stub = Path(_core.__file__).with_name("_core.pyi")
    found: set[tuple[str, str]] = set()
    for node in ast.parse(stub.read_text()).body:
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name == "__init__":
                for arg in member.args.args + member.args.kwonlyargs:
                    if arg.annotation and _STR_LIST_ANNOTATION.match(ast.unparse(arg.annotation)):
                        found.add((node.name, arg.arg))
    return found


def test_every_constructor_name_list_is_covered():
    """A constructor parameter added later that takes a list of strings fails here until it
    is a ``groupby`` of ``_CONSTRUCTORS``, a door of ``_NAME_LIST_DOORS`` or a value list."""
    stubbed = _stub_string_list_parameters()
    assert ("Flatten", "fields") in stubbed and ("Unpivot", "id_vars") in stubbed
    assert ("Bin", "groupby") in stubbed
    uncovered = sorted(
        (cls, param)
        for cls, param in stubbed
        if not (param == "groupby" and cls in _CONSTRUCTORS)
        and (cls, param) not in _NAME_LIST_DOORS
        and (cls, param) not in _VALUE_LISTS
    )
    assert not uncovered


@pytest.mark.parametrize("door", sorted(_NAME_LIST_DOORS), ids=lambda d: ".".join(d))
@pytest.mark.parametrize(
    "value, builder",
    [row[1:] for row in _MAPPING_ROWS + [r for r in _REFUSED if r[0] != "mappingproxy"]],
    ids=[row[0] for row in _MAPPING_ROWS + [r for r in _REFUSED if r[0] != "mappingproxy"]],
)
def test_a_name_list_constructor_refuses_a_mapping_set_or_bytes_in_the_transform_words(
    door, value, builder
):
    cls, param = door
    noun, build = _NAME_LIST_DOORS[door]
    with pytest.raises(ValueError) as excinfo:
        build(value)
    assert str(excinfo.value) == builder(cls, param, noun, value)


@pytest.mark.parametrize("door", sorted(_NAME_LIST_DOORS), ids=lambda d: ".".join(d))
def test_a_name_list_constructor_refuses_a_bare_string_in_the_transform_words(door):
    cls, param = door
    noun, build = _NAME_LIST_DOORS[door]
    with pytest.raises(ValueError) as excinfo:
        build("cat")
    assert str(excinfo.value) == _bare_string_text(cls, param, noun, "cat")


@pytest.mark.parametrize("door", sorted(_NAME_LIST_DOORS), ids=lambda d: ".".join(d))
@pytest.mark.parametrize(
    "make",
    [
        tuple,
        lambda items: (i for i in items),
        lambda items: {i: 0 for i in items}.keys(),
    ],
    ids=["tuple", "generator", "keys_view"],
)
def test_a_name_list_constructor_reads_ordered_values_like_a_list(door, make):
    _cls, _param = door
    _noun, build = _NAME_LIST_DOORS[door]
    items = ["zzz_col_b", "zzz_col_a"]
    assert repr(build(make(items))) == repr(build(list(items)))
    assert repr(build(make(items))) != repr(build(sorted(items)))


def test_unpivot_takes_no_id_vars_by_default():
    assert repr(_core.Unpivot()) == repr(_core.Unpivot(id_vars=[]))
    assert repr(_core.Unpivot(id_vars=None)) == repr(_core.Unpivot(id_vars=[]))
    assert "value_vars=None" in repr(_core.Unpivot())
