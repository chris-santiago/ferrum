"""Tests that Chart._clone covers every slot in Chart.__slots__.

The goal is to ensure that adding a new slot to Chart.__slots__ without
updating _clone does not cause silent data loss during fluent chaining.
The slot-driven loop makes this impossible to forget.
"""

import polars as pl
import pytest

import ferrum as fm
from ferrum.chart import Chart


@pytest.fixture()
def base_chart():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "cat": ["a", "b", "c"]})
    return (
        fm.Chart(df, width=400, height=300, title="Test")
        .mark_point()
        .encode(x="x", y="y", color="cat")
    )


def test_every_slot_present_on_clone(base_chart):
    """Every slot declared in Chart.__slots__ must be set on the clone."""
    cloned = base_chart._clone()
    for slot in Chart.__slots__:
        assert hasattr(cloned, slot), f"slot {slot!r} missing on clone"


def test_mutable_containers_are_distinct(base_chart):
    """List and dict slots must be shallow-copied, not aliased."""
    cloned = base_chart._clone()
    for slot in Chart.__slots__:
        original_val = getattr(base_chart, slot)
        cloned_val = getattr(cloned, slot)
        if isinstance(original_val, (list, dict)):
            assert original_val is not cloned_val, (
                f"slot {slot!r} is the same object on original and clone (should be a shallow copy)"
            )


def test_mutating_clone_does_not_affect_original(base_chart):
    """Mutating a list slot on the clone must not affect the original."""
    cloned = base_chart._clone()
    # _encoding is a dict — add a spurious key on the clone
    cloned._encoding["__test__"] = "sentinel"
    assert "__test__" not in base_chart._encoding

    # _transforms is a list — append on the clone
    original_transform_count = len(base_chart._transforms)
    cloned._transforms.append(object())
    assert len(base_chart._transforms) == original_transform_count


def test_immutable_slots_are_same_reference(base_chart):
    """Immutable-typed slots (str, int, None) should be the same object."""
    cloned = base_chart._clone()
    immutable_types = (str, int, float, bool, type(None))
    for slot in Chart.__slots__:
        original_val = getattr(base_chart, slot)
        cloned_val = getattr(cloned, slot)
        if isinstance(original_val, immutable_types):
            assert original_val is cloned_val, (
                f"slot {slot!r}: expected same object for immutable value, "
                f"got {original_val!r} vs {cloned_val!r}"
            )


def test_clone_slot_count_matches_slots_definition():
    """The number of slots on a fresh Chart must equal len(Chart.__slots__)."""
    df = pl.DataFrame({"x": [1], "y": [2]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    cloned = chart._clone()
    # Every slot must be reachable — no AttributeError expected
    filled = [slot for slot in Chart.__slots__ if hasattr(cloned, slot)]
    assert len(filled) == len(Chart.__slots__), (
        f"Expected {len(Chart.__slots__)} slots, got {len(filled)} on clone"
    )
