"""Wave 1 T4 (#179 part (c)): the dict-emitting `transform_pivot`/
`transform_aggregate`/`transform_density` wrappers refuse a repeated
`groupby` name at construction time, matching `_reject_bare_string_groupby`'s
site.

These three build a raw wire dict with no Rust pyclass constructor to
refuse at (unlike ``Aggregate``, whose ``#[new]`` already calls the Rust-side
``refuse_repeated_groupby_names`` — see
``test_chart_spec.py::test_aggregate_construct_rejects_duplicate_groupby``).
Before this guard, ``fm.transform_aggregate(op, groupby=["g", "g"])`` built
a spec with two identically-named output columns and only refused once it
reached Rust's apply-time partitioning (``group_partition_multi``) at render
— reachable, but deferred past the construction-time boundary that already
has every fact it needs.
"""

from __future__ import annotations

import pytest

import ferrum as fm
from ferrum.transforms import _reject_repeated_groupby_names


@pytest.mark.parametrize(
    "call",
    [
        lambda: fm.transform_aggregate({"field": "x", "fn": "mean", "as": "m"}, groupby=["g", "g"]),
        lambda: fm.transform_pivot("category", "amount", groupby=["g", "g"]),
        lambda: fm.transform_density("weight", groupby=["g", "g"]),
    ],
    ids=["transform_aggregate", "transform_pivot", "transform_density"],
)
def test_repeated_groupby_name_refuses(call):
    with pytest.raises(ValueError, match="repeated"):
        call()


@pytest.mark.parametrize(
    "call, expected",
    [
        (
            lambda: fm.transform_aggregate(
                {"field": "x", "fn": "mean", "as": "m"}, groupby=["a", "b"]
            ),
            ["a", "b"],
        ),
        (
            lambda: fm.transform_pivot("category", "amount", groupby=["a", "b"]),
            ["a", "b"],
        ),
        (
            lambda: fm.transform_density("weight", groupby=["a", "b"]),
            ["a", "b"],
        ),
    ],
    ids=["transform_aggregate", "transform_pivot", "transform_density"],
)
def test_distinct_groupby_still_reaches_the_wire(call, expected):
    """The guard rejects only repeats -- distinct names still land on the
    wire under ``groupby``, unchanged."""
    spec = call()
    assert spec["groupby"] == expected


def test_reject_repeated_groupby_names_is_the_single_shared_mechanism():
    """Every refusal above comes from one predicate, not a copy per site."""
    with pytest.raises(ValueError, match=r"transform_pivot.*groupby column 'g' is repeated"):
        _reject_repeated_groupby_names(["g", "g"], where="transform_pivot")

    # A single name, or all-distinct names, are not rejected.
    _reject_repeated_groupby_names([], where="transform_pivot")
    _reject_repeated_groupby_names(["g"], where="transform_pivot")
    _reject_repeated_groupby_names(["a", "b", "c"], where="transform_pivot")


def test_three_repeats_reports_the_first_offender():
    with pytest.raises(ValueError, match=r"groupby column 'b' is repeated"):
        _reject_repeated_groupby_names(["a", "b", "b", "a"], where="transform_pivot")
