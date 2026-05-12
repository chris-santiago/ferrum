"""Schwabish SB1 — Chart accepts Title | str; ChartSpec receives dict form."""
from __future__ import annotations

import polars as pl
import pytest

from ferrum import Chart, Title


@pytest.fixture
def df():
    return pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})


def test_chart_accepts_title_string(df):
    c = Chart(df, title="My chart").encode(x="x", y="y").mark_point()
    assert c._title is not None
    assert c._title.text == "My chart"
    assert c._title.subtitle is None


def test_chart_accepts_title_class(df):
    c = Chart(df, title=Title("My chart", subtitle="2024")).encode(x="x", y="y").mark_point()
    assert c._title.text == "My chart"
    assert c._title.subtitle == "2024"


def test_properties_accepts_title_class(df):
    c = Chart(df).encode(x="x", y="y").mark_point().properties(title=Title("foo", subtitle="bar"))
    assert c._title.subtitle == "bar"


def test_properties_accepts_title_string(df):
    c = Chart(df).encode(x="x", y="y").mark_point().properties(title="simple")
    assert c._title.text == "simple"
    assert c._title.subtitle is None


def test_chartspec_title_round_trips_as_dict(df):
    import json
    c = Chart(df, title=Title("foo", subtitle="bar")).encode(x="x", y="y").mark_point()
    spec = c.to_spec()
    payload = json.loads(spec.to_json())
    assert payload["title"]["text"] == "foo"
    assert payload["title"]["subtitle"] == "bar"


def test_string_title_round_trips_with_text_field(df):
    import json
    c = Chart(df, title="just a string").encode(x="x", y="y").mark_point()
    spec = c.to_spec()
    payload = json.loads(spec.to_json())
    assert payload["title"] == {"text": "just a string"}
