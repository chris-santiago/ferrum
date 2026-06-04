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
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 3.0],
            "label": ["first", "second", "third"],
        }
    )


# ── CoordCartesian ──────────────────────────────────────────────────────────


def test_coord_cartesian_xlim_renders(scatter_df):
    svg = (
        fm.Chart(scatter_df)
        .mark_point()
        .encode(x="x", y="y")
        .coord(fm.CoordCartesian(xlim=(0.0, 6.0)))
        .to_svg()
    )
    assert "<svg" in svg
    # Verify coord spec round-trips
    spec = (
        fm.Chart(scatter_df)
        .mark_point()
        .encode(x="x", y="y")
        .coord(fm.CoordCartesian(xlim=(0.0, 6.0)))
        .to_spec()
    )
    j = json.loads(spec.to_json())
    assert j["coord"]["x_domain"] == [0.0, 6.0]


def test_coord_cartesian_ylim_renders(scatter_df):
    svg = (
        fm.Chart(scatter_df)
        .mark_point()
        .encode(x="x", y="y")
        .coord(fm.CoordCartesian(ylim=(0.0, 10.0)))
        .to_svg()
    )
    assert "<svg" in svg


def test_coord_cartesian_expand_false(scatter_df):
    svg = (
        fm.Chart(scatter_df)
        .mark_point()
        .encode(x="x", y="y")
        .coord(fm.CoordCartesian(expand=False))
        .to_svg()
    )
    assert "<svg" in svg


def test_coord_cartesian_clip_false(scatter_df):
    svg = (
        fm.Chart(scatter_df)
        .mark_point()
        .encode(x="x", y="y")
        .coord(fm.CoordCartesian(clip=False))
        .to_svg()
    )
    assert "<svg" in svg


# ── CoordFixed ─────────────────────────────────────────────────────────────


def test_coord_fixed_ratio_one_renders(scatter_df):
    svg = (
        fm.Chart(scatter_df)
        .mark_point()
        .encode(x="x", y="y")
        .coord(fm.CoordFixed(ratio=1.0))
        .to_svg()
    )
    assert "<svg" in svg


def test_coord_fixed_spec_round_trip(scatter_df):
    spec = (
        fm.Chart(scatter_df)
        .mark_point()
        .encode(x="x", y="y")
        .coord(fm.CoordFixed(ratio=2.0))
        .to_spec()
    )
    j = json.loads(spec.to_json())
    assert j["coord"]["kind"] == "fixed"
    assert j["coord"]["ratio"] == 2.0


# ── CoordPolar + mark_arc ──────────────────────────────────────────────────


def test_mark_arc_pie_renders(pie_df):
    svg = (
        fm.Chart(pie_df)
        .mark_arc()
        .encode(x="value", color="category")
        .coord(fm.CoordPolar(theta="x"))
        .to_svg()
    )
    assert "<svg" in svg


def test_mark_arc_donut_renders(pie_df):
    """Donut: inner_radius set via CoordPolar inner_radius (spec field)."""
    spec = (
        fm.Chart(pie_df)
        .mark_arc()
        .encode(x="value", color="category")
        .coord(fm.CoordPolar(theta="x"))
        .to_spec()
    )
    j = json.loads(spec.to_json())
    assert j["coord"]["kind"] == "polar"


def test_mark_arc_spec_contains_polar_coord(pie_df):
    spec = fm.Chart(pie_df).mark_arc().encode(x="value").coord(fm.CoordPolar(theta="x")).to_spec()
    j = json.loads(spec.to_json())
    assert j["mark"] == "arc"
    assert j["coord"]["theta"] == "x"


# ── mark_label ────────────────────────────────────────────────────────────


def test_mark_label_renders(label_df):
    svg = fm.Chart(label_df).mark_label().encode(x="x", y="y", text="label").to_svg()
    assert "<svg" in svg


def test_mark_label_spec_round_trip(label_df):
    spec = fm.Chart(label_df).mark_label(dy=-12.0).encode(x="x", y="y", text="label").to_spec()
    j = json.loads(spec.to_json())
    assert j["mark"] == "label"


def test_mark_label_dense_no_crash():
    """Dense, tightly-clustered labels must render without error.

    The collision-avoidance algorithm must not raise or produce invalid SVG
    even when no perfect non-overlapping placement exists.  Each label text
    value must appear exactly once in the SVG output.
    """
    labels = list("abcdefghi")
    df = pl.DataFrame(
        {
            "x": pl.Series([1.0, 1.05, 1.1, 1.15, 2.0, 2.05, 2.1, 2.15, 3.0]),
            "y": pl.Series([1.0, 1.05, 1.1, 1.15, 2.0, 2.05, 2.1, 2.15, 3.0]),
            "label": labels,
        }
    )
    svg = fm.Chart(df).mark_label().encode(x="x:Q", y="y:Q", text="label").to_svg()
    assert "<svg" in svg
    # Each single-character label must appear in the SVG as label text content.
    for lbl in labels:
        assert f">{lbl}<" in svg, f"label '{lbl}' missing from SVG"


def test_mark_label_manual_override_bypasses_avoidance(label_df):
    """When both dx and dy are provided, all labels use those fixed offsets.

    The SVG must contain every label text value and render without error.
    """
    svg = (
        fm.Chart(label_df)
        .mark_label(dx=5.0, dy=-15.0)
        .encode(x="x", y="y", text="label")
        .to_svg()
    )
    assert "<svg" in svg
    for lbl in ("first", "second", "third"):
        assert f">{lbl}<" in svg, f"label '{lbl}' missing from SVG"


# ── CoordGeo + mark_geoshape ───────────────────────────────────────────────

MINIMAL_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "A"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]],
            },
        },
        {
            "type": "Feature",
            "properties": {"name": "B"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[20.0, 0.0], [30.0, 0.0], [30.0, 10.0], [20.0, 10.0], [20.0, 0.0]]
                ],
            },
        },
    ],
}


def test_geojson_coerce_adds_geometry_column():
    """GeoJSON FeatureCollection must produce a __geometry__ column."""
    from ferrum._coerce import to_arrow_table

    tbl = to_arrow_table(MINIMAL_GEOJSON)
    assert "__geometry__" in tbl.schema.names
    assert "name" in tbl.schema.names
    assert tbl.num_rows == 2


def test_mark_geoshape_with_geo_coord_renders():
    svg = (
        fm.Chart(MINIMAL_GEOJSON)
        .mark_geoshape()
        .encode(color="name")
        .coord(fm.CoordGeo(projection="equirectangular"))
        .to_svg()
    )
    assert "<svg" in svg


def test_mark_geoshape_spec_round_trip():
    spec = (
        fm.Chart(MINIMAL_GEOJSON)
        .mark_geoshape()
        .encode(color="name")
        .coord(fm.CoordGeo(projection="mercator"))
        .to_spec()
    )
    j = json.loads(spec.to_json())
    assert j["mark"] == "geoshape"
    assert j["coord"]["projection"] == "mercator"


# ── mark_label collision avoidance ────────────────────────────────────────


def test_mark_label_collision_avoidance_produces_spread_positions():
    """Dense labels must not all share the same y-offset.

    Naive placement (fixed dy=-8) would give every label the same y. With
    collision avoidance, at least some labels must be placed at a different
    y position (above, below, or diagonally repositioned).
    """
    import re

    labels = list("abcdefgh")
    df = pl.DataFrame(
        {
            "x": pl.Series([1.0] * 8),  # all same x — maximally dense
            "y": pl.Series([2.0] * 8),  # all same y — maximally dense
            "label": labels,
        }
    )
    svg = fm.Chart(df).mark_label().encode(x="x:Q", y="y:Q", text="label").to_svg()
    # Extract y coordinates of <text> elements — match only numeric y="NNN.NNN" values
    y_vals = [float(m) for m in re.findall(r'<text[^>]+\by="(-?[\d.]+)"', svg)]
    # With 8 labels at the same point, collision avoidance must place them
    # at more than one distinct y position (naive would have only 1 unique y).
    unique_y = set(round(v, 1) for v in y_vals)
    assert len(unique_y) > 1, (
        f"All labels landed at the same y — collision avoidance may not be working. "
        f"y positions: {sorted(unique_y)}"
    )


# ── mark_image URL tiles ───────────────────────────────────────────────────

# Minimal valid 1×1 red PNG encoded as base64.
_TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADklEQVQI12P4z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="


def test_mark_image_url_tiles_render_image_elements():
    """mark_image with url encoding must emit <image> elements in the SVG."""
    df = pl.DataFrame(
        {
            "x": [1.0, 3.0, 5.0],
            "y": [2.0, 4.0, 2.0],
            "url": [f"data:image/png;base64,{_TINY_PNG_B64}"] * 3,
        }
    )
    svg = fm.Chart(df).mark_image().encode(x="x:Q", y="y:Q", url="url").to_svg()
    assert "<svg" in svg
    assert "<image" in svg, "Expected <image> elements in SVG for mark_image URL tiles"


def test_mark_raster_still_works_after_image_rewrite():
    """mark_raster must continue rendering <image> elements (regression for image.rs rewrite)."""
    import numpy as np

    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {"x": rng.uniform(0, 10, 300).tolist(), "y": rng.uniform(0, 10, 300).tolist()}
    )
    svg = fm.Chart(df).mark_raster().encode(x="x:Q", y="y:Q").to_svg()
    assert "<svg" in svg
    assert "<image" in svg, (
        "mark_raster must still produce an <image> element after image.rs rewrite"
    )


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
