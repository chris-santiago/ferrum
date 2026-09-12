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


def test_repeated_calls_with_identical_message_only_warn_once():
    """Same (channel, kwarg) AND same message: dedup as before."""
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis", "first")
        warn_once("X", "axis", "first")
        warn_once("X", "axis", "first")
    assert len(w) == 1
    assert "first" in str(w[0].message)


def test_repeated_calls_with_different_messages_each_warn():
    """Same (channel, kwarg) but a DIFFERENT message each time: #240 keystone.

    warn_once's identity used to be (channel, kwarg) only, so a second
    caller with a different fact about the same pair was silently dropped
    -- the first message permanently won the slot. Message is now part of
    the key, so each distinct message reaches the user once.
    """
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("X", "axis", "first")
        warn_once("X", "axis", "second")
        warn_once("X", "axis", "third")
    assert len(w) == 3
    assert [str(wi.message) for wi in w] == ["first", "second", "third"]


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
