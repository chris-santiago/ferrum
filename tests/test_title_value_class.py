"""Schwabish SB1 — Title value class basic behavior."""
from __future__ import annotations

import pytest

from ferrum.title import Title


def test_title_text_only():
    t = Title("Sales")
    assert t.text == "Sales"
    assert t.subtitle is None
    assert t.anchor == "start"


def test_title_with_subtitle():
    t = Title("Sales", subtitle="2024 Q3")
    assert t.text == "Sales"
    assert t.subtitle == "2024 Q3"


def test_title_is_immutable():
    t = Title("Sales")
    with pytest.raises((AttributeError, TypeError)):
        t.text = "Changed"  # type: ignore[misc]


def test_title_repr_round_trip():
    t = Title("Sales", subtitle="2024 Q3", font_weight="bold")
    rendered = repr(t)
    assert "Sales" in rendered
    assert "2024 Q3" in rendered


def test_to_spec_dict_text_only():
    t = Title("foo")
    assert t.to_spec_dict() == {"text": "foo"}


def test_to_spec_dict_with_subtitle_and_styling():
    t = Title("foo", subtitle="bar", anchor="middle", font_weight="bold")
    d = t.to_spec_dict()
    assert d["text"] == "foo"
    assert d["subtitle"] == "bar"
    assert d["anchor"] == "middle"
    assert d["font_weight"] == "bold"
