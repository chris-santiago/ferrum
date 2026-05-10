"""Tests for Chart composition operators: +, |, &."""
import warnings

import polars as pl
import pytest

from ferrum import Chart


@pytest.fixture
def df():
    return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_layer_same_data_produces_layered_chart(df):
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df).mark_line().encode(x="a", y="b")
    layered = c1 + c2
    # Same data → wrapped layer Chart, not HConcat
    assert layered._layers is not None
    assert len(layered._layers) == 2


def test_layer_different_data_falls_through_to_hconcat(df):
    df2 = pl.DataFrame({"a": [10], "b": [20]})
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df2).mark_line().encode(x="a", y="b")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = c1 + c2
    # Falls through to HConcat
    from ferrum.composition import HConcatChart
    assert isinstance(result, HConcatChart)
    assert any("differing data" in str(wi.message) for wi in w)


def test_hconcat_two_charts(df):
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df).mark_line().encode(x="a", y="b")
    result = c1 | c2
    from ferrum.composition import HConcatChart
    assert isinstance(result, HConcatChart)
    assert len(result.charts) == 2


def test_vconcat_two_charts(df):
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df).mark_line().encode(x="a", y="b")
    result = c1 & c2
    from ferrum.composition import VConcatChart
    assert isinstance(result, VConcatChart)


def test_operator_precedence_and_tighter_than_or(df):
    a = Chart(df).mark_point().encode(x="a", y="b")
    b = Chart(df).mark_line().encode(x="a", y="b")
    c = Chart(df).mark_bar().encode(x="a", y="b")
    # a | b & c should parse as a | (b & c), not (a | b) & c
    result = a | b & c
    from ferrum.composition import HConcatChart, VConcatChart
    assert isinstance(result, HConcatChart)
    # Inner (b & c) is the second item
    assert isinstance(result.charts[1], VConcatChart)


def test_explicit_parens_overrides_precedence(df):
    a = Chart(df).mark_point().encode(x="a", y="b")
    b = Chart(df).mark_line().encode(x="a", y="b")
    c = Chart(df).mark_bar().encode(x="a", y="b")
    result = (a | b) & c
    from ferrum.composition import VConcatChart
    assert isinstance(result, VConcatChart)


def test_layer_marks_are_recorded(df):
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df).mark_line().encode(x="a", y="b")
    layered = c1 + c2
    assert layered._layers[0]["mark"] == "point"
    assert layered._layers[1]["mark"] == "line"


def test_layer_same_df_object_by_identity(df):
    """Two charts referencing the same df object layer without warning."""
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df).mark_line().encode(x="a", y="b")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = c1 + c2
    differing_warns = [x for x in w if "differing data" in str(x.message)]
    assert not differing_warns, "Should not warn for same df object"
    assert result._layers is not None


def test_hconcat_chaining_produces_flat_list(df):
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df).mark_line().encode(x="a", y="b")
    c3 = Chart(df).mark_bar().encode(x="a", y="b")
    # (c1 | c2) | c3 → HConcatChart([HConcatChart([c1, c2]), c3])
    result = (c1 | c2) | c3
    from ferrum.composition import HConcatChart
    assert isinstance(result, HConcatChart)
    # Outer has 2 items: the inner HConcat and c3
    assert len(result.charts) == 2


def test_add_returns_notimplemented_for_non_chart(df):
    c = Chart(df).mark_point().encode(x="a", y="b")
    result = c.__add__(42)
    assert result is NotImplemented


# ---------------------------------------------------------------------------
# Test 2: compositor render integration (Phase 8a final review)
# ---------------------------------------------------------------------------

def test_hconcat_show_svg_produces_composed_output():
    """End-to-end: (c1 | c2).show_svg() actually composes through the Rust compositor."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df).mark_line().encode(x="a", y="b")
    svg = (c1 | c2).show_svg()
    assert svg.startswith("<svg") or svg.startswith("<?xml")
    # Composed output should contain at least 2 <g transform="translate(...)"> wrappers
    assert svg.count('transform="translate(') >= 2


def test_vconcat_show_svg_produces_composed_output():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    c1 = Chart(df).mark_point().encode(x="a", y="b")
    c2 = Chart(df).mark_bar().encode(x="a", y="b")
    svg = (c1 & c2).show_svg()
    assert svg.startswith("<svg") or svg.startswith("<?xml")
    assert svg.count('transform="translate(') >= 2
