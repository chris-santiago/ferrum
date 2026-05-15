"""Phase 11d smoke tests — coordinate systems and new marks.

These tests verify the pipeline runs without error and produces valid SVG.
No golden file comparison — visual inspection handled separately.
"""

import json

import polars as pl
import pytest

import ferrum as fm


@pytest.fixture
def scatter_df():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 1.0, 3.0, 5.0]})


@pytest.fixture
def pie_df():
    return pl.DataFrame({"category": ["A", "B", "C", "D"], "value": [30.0, 20.0, 15.0, 35.0]})


@pytest.fixture
def label_df():
    return pl.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [2.0, 4.0, 3.0],
        "label": ["first", "second", "third"],
    })


# ── CoordCartesian ──────────────────────────────────────────────────────────

def test_coord_cartesian_xlim_renders(scatter_df):
    svg = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").coord(
        fm.CoordCartesian(xlim=(0.0, 6.0))
    ).show_svg()
    assert "<svg" in svg
    # Verify coord spec round-trips
    spec = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").coord(
        fm.CoordCartesian(xlim=(0.0, 6.0))
    ).to_spec()
    j = json.loads(spec.to_json())
    assert j["coord"]["x_domain"] == [0.0, 6.0]


def test_coord_cartesian_ylim_renders(scatter_df):
    svg = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").coord(
        fm.CoordCartesian(ylim=(0.0, 10.0))
    ).show_svg()
    assert "<svg" in svg


def test_coord_cartesian_expand_false(scatter_df):
    svg = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").coord(
        fm.CoordCartesian(expand=False)
    ).show_svg()
    assert "<svg" in svg


def test_coord_cartesian_clip_false(scatter_df):
    svg = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").coord(
        fm.CoordCartesian(clip=False)
    ).show_svg()
    assert "<svg" in svg


# ── CoordFixed ─────────────────────────────────────────────────────────────

def test_coord_fixed_ratio_one_renders(scatter_df):
    svg = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").coord(
        fm.CoordFixed(ratio=1.0)
    ).show_svg()
    assert "<svg" in svg


def test_coord_fixed_spec_round_trip(scatter_df):
    spec = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").coord(
        fm.CoordFixed(ratio=2.0)
    ).to_spec()
    j = json.loads(spec.to_json())
    assert j["coord"]["kind"] == "fixed"
    assert j["coord"]["ratio"] == 2.0


# ── CoordPolar + mark_arc ──────────────────────────────────────────────────

def test_mark_arc_pie_renders(pie_df):
    svg = fm.Chart(pie_df).mark_arc().encode(
        x="value", color="category"
    ).coord(fm.CoordPolar(theta="x")).show_svg()
    assert "<svg" in svg


def test_mark_arc_donut_renders(pie_df):
    """Donut: inner_radius set via CoordPolar inner_radius (spec field)."""
    spec = fm.Chart(pie_df).mark_arc().encode(
        x="value", color="category"
    ).coord(fm.CoordPolar(theta="x")).to_spec()
    j = json.loads(spec.to_json())
    assert j["coord"]["kind"] == "polar"


def test_mark_arc_spec_contains_polar_coord(pie_df):
    spec = fm.Chart(pie_df).mark_arc().encode(
        x="value"
    ).coord(fm.CoordPolar(theta="x")).to_spec()
    j = json.loads(spec.to_json())
    assert j["mark"] == "arc"
    assert j["coord"]["theta"] == "x"


# ── mark_label ────────────────────────────────────────────────────────────

def test_mark_label_renders(label_df):
    svg = fm.Chart(label_df).mark_label().encode(
        x="x", y="y", text="label"
    ).show_svg()
    assert "<svg" in svg


def test_mark_label_spec_round_trip(label_df):
    spec = fm.Chart(label_df).mark_label(dy=-12.0).encode(
        x="x", y="y", text="label"
    ).to_spec()
    j = json.loads(spec.to_json())
    assert j["mark"] == "label"


# ── CoordGeo + mark_geoshape ───────────────────────────────────────────────

MINIMAL_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "A"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "B"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[20.0, 0.0], [30.0, 0.0], [30.0, 10.0], [20.0, 10.0], [20.0, 0.0]]]
            }
        },
    ]
}


def test_geojson_coerce_adds_geometry_column():
    """GeoJSON FeatureCollection must produce a __geometry__ column."""
    from ferrum._coerce import to_arrow_table
    tbl = to_arrow_table(MINIMAL_GEOJSON)
    assert "__geometry__" in tbl.schema.names
    assert "name" in tbl.schema.names
    assert tbl.num_rows == 2


def test_mark_geoshape_with_geo_coord_renders():
    svg = fm.Chart(MINIMAL_GEOJSON).mark_geoshape().encode(
        color="name"
    ).coord(fm.CoordGeo(projection="equirectangular")).show_svg()
    assert "<svg" in svg


def test_mark_geoshape_spec_round_trip():
    spec = fm.Chart(MINIMAL_GEOJSON).mark_geoshape().encode(
        color="name"
    ).coord(fm.CoordGeo(projection="mercator")).to_spec()
    j = json.loads(spec.to_json())
    assert j["mark"] == "geoshape"
    assert j["coord"]["projection"] == "mercator"


# ── mark_image coord-awareness ────────────────────────────────────────────

def test_mark_image_callable_without_error():
    """mark_image must be callable (no longer raises NotImplementedError)."""
    df = pl.DataFrame({"x": [1.0], "y": [1.0], "url": ["data:image/png;base64,iVBORw0KGgo="]})
    chart = fm.Chart(df).mark_image()
    assert chart is not None


# ── String back-compat ─────────────────────────────────────────────────────

def test_string_coord_flip_still_works(scatter_df):
    """coord='flip' string back-compat must survive."""
    from ferrum import ChartSpec
    spec = ChartSpec(mark="point", x="x", y="y", coord="flip")
    j = json.loads(spec.to_json())
    assert j["coord"]["kind"] == "flip"


def test_string_coord_cartesian_still_works(scatter_df):
    from ferrum import ChartSpec
    spec = ChartSpec(mark="point", x="x", y="y", coord="cartesian")
    j = json.loads(spec.to_json())
    assert j["coord"]["kind"] == "cartesian"
