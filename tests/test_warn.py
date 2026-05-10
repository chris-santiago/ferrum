import warnings

import pytest

from ferrum._warn import warn_once, reset_warnings


def test_first_call_emits_warning():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis", "X(axis=...) is deferred")
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "axis" in str(w[0].message)


def test_repeated_calls_only_warn_once():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis", "first")
        warn_once("X", "axis", "second")
        warn_once("X", "axis", "third")
    assert len(w) == 1
    assert "first" in str(w[0].message)


def test_distinct_keys_each_warn():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis")
        warn_once("X", "legend")
        warn_once("Y", "axis")
    assert len(w) == 3


def test_default_message_when_none_provided():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis")
    assert "X(axis=...)" in str(w[0].message)
    assert "Phase" in str(w[0].message)


def test_reset_warnings_allows_re_warning():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis")
        reset_warnings()
        warn_once("X", "axis")
    assert len(w) == 2
