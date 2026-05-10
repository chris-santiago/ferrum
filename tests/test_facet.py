"""Tests for Chart.facet() fluent method."""
import polars as pl

from ferrum import Chart


def test_facet_with_col_only():
    df = pl.DataFrame({"a": [1, 2, 3], "species": ["s1", "s2", "s1"]})
    c = Chart(df).mark_point().encode(x="a", y="a").facet(col="species")
    assert c._facet is not None
    assert c._facet["field"] == "species"


def test_facet_with_row_and_col_grid():
    df = pl.DataFrame({"a": [1], "year": ["2024"], "species": ["s1"]})
    c = Chart(df).mark_point().encode(x="a", y="a").facet(row="year", col="species")
    assert c._facet is not None
    # grid mode produces a different shape than wrap; assert mode-distinguishing field
    assert c._facet.get("mode_kind") == "grid"


def test_facet_with_explicit_field_is_wrap():
    df = pl.DataFrame({"a": [1, 2], "grp": ["g1", "g2"]})
    c = Chart(df).mark_point().encode(x="a", y="a").facet(field="grp", ncols=2)
    assert c._facet["mode_kind"] == "wrap"
    assert c._facet["ncols"] == 2


def test_facet_with_row_only_is_wrap():
    df = pl.DataFrame({"a": [1], "year": ["2024"]})
    c = Chart(df).mark_point().encode(x="a", y="a").facet(row="year")
    assert c._facet["mode_kind"] == "wrap"
    assert c._facet["field"] == "year"


def test_facet_no_args_raises():
    import pytest
    df = pl.DataFrame({"a": [1]})
    chart = Chart(df).mark_point().encode(x="a", y="a")
    with pytest.raises(ValueError, match="facet"):
        chart.facet()


def test_facet_to_spec_round_trips_wrap():
    """_build_facet_dict() produces JSON Rust can deserialize."""
    import json
    df = pl.DataFrame({"a": [1, 2, 3], "species": ["s1", "s2", "s1"]})
    c = Chart(df).mark_point().encode(x="a", y="a").facet(col="species", ncols=2)
    spec = c.to_spec()
    j = json.loads(spec.to_json())
    facet = j["facet"]
    assert facet["field"] == "species"
    assert facet["mode"]["kind"] == "wrap"
    assert facet["mode"]["ncols"] == 2
