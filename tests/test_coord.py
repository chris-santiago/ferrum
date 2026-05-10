"""Tests for Chart.coord() fluent method and coord.py classes."""
import pytest
import polars as pl

from ferrum import Chart, CoordFlip


def test_coord_flip_sets_chartspec_coord():
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_bar().encode(x="a", y="b").coord(CoordFlip())
    assert c._coord == "flip"


def test_coord_other_kinds_raise_notimplemented():
    from ferrum.coord import CoordPolar, CoordGeo, CoordFixed, CoordCartesian
    df = pl.DataFrame({"a": [1]})
    chart = Chart(df).mark_point().encode(x="a", y="a")
    for cls in (CoordPolar, CoordGeo, CoordFixed, CoordCartesian):
        with pytest.raises(NotImplementedError, match="Phase 9"):
            cls()  # constructors raise immediately


def test_coord_flip_raises_for_unknown_coord():
    df = pl.DataFrame({"a": [1], "b": [2]})
    chart = Chart(df).mark_point().encode(x="a", y="b")
    with pytest.raises(TypeError, match="CoordFlip"):
        chart.coord("flip")  # string not accepted; only CoordFlip instance


def test_coord_flip_passes_to_spec():
    """coord='flip' makes it through to_spec() and round-trips."""
    import json
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    c = Chart(df).mark_bar().encode(x="a", y="b").coord(CoordFlip())
    spec = c.to_spec()
    j = json.loads(spec.to_json())
    # Rust serializes CoordKind::Flip as {"kind": "flip"}
    coord = j.get("coord")
    assert coord is not None
    assert coord.get("kind") == "flip"
