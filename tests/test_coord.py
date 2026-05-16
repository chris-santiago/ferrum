"""Tests for Chart.coord() fluent method and coord.py classes."""
import json

import polars as pl
import pytest

from ferrum import Chart, CoordCartesian, CoordFixed, CoordFlip, CoordGeo, CoordPolar


def test_coord_flip_stores_object():
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_bar().encode(x="a", y="b").coord(CoordFlip())
    assert isinstance(c._coord, CoordFlip)


def test_coord_other_kinds_are_now_functional():
    # Phase 11d: CoordPolar, CoordGeo, CoordFixed, CoordCartesian no longer raise.
    df = pl.DataFrame({"a": [1]})
    chart = Chart(df).mark_point().encode(x="a", y="a")
    for coord in (CoordPolar(), CoordGeo(), CoordFixed(), CoordCartesian()):
        chart.coord(coord)  # must not raise


def test_coord_flip_raises_for_unknown_coord():
    df = pl.DataFrame({"a": [1], "b": [2]})
    chart = Chart(df).mark_point().encode(x="a", y="b")
    with pytest.raises(TypeError, match="CoordFlip"):
        chart.coord("flip")  # string not accepted; only coord objects


def test_coord_flip_passes_to_spec():
    """CoordFlip() wires through to_spec() as {"kind": "flip"}."""
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    c = Chart(df).mark_bar().encode(x="a", y="b").coord(CoordFlip())
    spec = c.to_spec()
    j = json.loads(spec.to_json())
    coord = j.get("coord")
    assert coord is not None
    assert coord.get("kind") == "flip"


def test_coord_cartesian_xlim_passes_to_spec():
    """CoordCartesian(xlim=(0,10)) wires x_domain into the spec."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    c = Chart(df).mark_point().encode(x="a", y="b").coord(CoordCartesian(xlim=(0, 10)))
    j = json.loads(c.to_spec().to_json())
    coord = j.get("coord")
    assert coord is not None
    assert coord["kind"] == "cartesian"
    assert coord["x_domain"] == [0.0, 10.0]


def test_coord_fixed_passes_to_spec():
    """CoordFixed(ratio=1.0) wires ratio into the spec."""
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    c = Chart(df).mark_point().encode(x="a", y="b").coord(CoordFixed(ratio=1.0))
    j = json.loads(c.to_spec().to_json())
    assert j["coord"]["kind"] == "fixed"
    assert j["coord"]["ratio"] == 1.0


def test_coord_geo_spec_dict():
    """CoordGeo.to_spec_dict() emits the correct kind/projection keys."""
    geo = CoordGeo(projection="equal_earth")
    d = geo.to_spec_dict()
    assert d["kind"] == "geo"
    assert d["projection"] == "equal_earth"

    # Round-trip through ChartSpec directly (mark_geoshape wired in Task 11d5).
    from ferrum import ChartSpec
    spec = ChartSpec(mark="point", x="a", y="b", coord=d)
    j = json.loads(spec.to_json())
    assert j["coord"]["kind"] == "geo"
    assert j["coord"]["projection"] == "equal_earth"
