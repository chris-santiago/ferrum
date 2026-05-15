"""Tests for Chart.facet() fluent method."""
import polars as pl

from ferrum import Chart


def test_facet_with_col_only():
    df = pl.DataFrame({"a": [1, 2, 3], "species": ["s1", "s2", "s1"]})
    c = Chart(df).mark_point().encode(x="a", y="a").facet(col="species")
    assert c._facet is not None
    assert c._facet.field == "species"


def test_facet_with_row_and_col_grid():
    df = pl.DataFrame({"a": [1], "year": ["2024"], "species": ["s1"]})
    c = Chart(df).mark_point().encode(x="a", y="a").facet(row="year", col="species")
    assert c._facet is not None
    # grid mode produces a different shape than wrap; assert mode-distinguishing field
    assert c._facet.mode_kind == "grid"


def test_facet_with_explicit_field_is_wrap():
    df = pl.DataFrame({"a": [1, 2], "grp": ["g1", "g2"]})
    c = Chart(df).mark_point().encode(x="a", y="a").facet(field="grp", ncols=2)
    assert c._facet.mode_kind == "wrap"
    assert c._facet.ncols == 2


def test_facet_with_row_only_is_wrap():
    df = pl.DataFrame({"a": [1], "year": ["2024"]})
    c = Chart(df).mark_point().encode(x="a", y="a").facet(row="year")
    assert c._facet.mode_kind == "wrap"
    assert c._facet.field == "year"


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


def test_encode_facet_col_sets_facet():
    """encode(facet_col=...) must set _facet, not silently drop the channel."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "species": ["a", "b", "a"]})
    c = Chart(df).mark_point().encode(x="x", y="y", facet_col="species:N")
    assert c._facet is not None
    assert c._facet.mode_kind == "wrap"
    assert c._facet.field == "species"


def test_encode_facet_col_produces_panels():
    """encode(facet_col=...) must produce multiple panels in the SVG."""
    import pyarrow as pa
    from ferrum._core import render_svg

    tbl = pa.table({
        "x": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
        "y": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
        "species": pa.array(["A", "A", "B", "B", "C", "C"], type=pa.string()),
    })
    df = pl.from_arrow(tbl)
    chart = Chart(df).mark_point().encode(x="x", y="y", facet_col="species:N")
    spec = chart.to_spec()
    svg = render_svg(spec, tbl, viewport=(600.0, 400.0), theme={})
    # 3 panels → ≥6 clipPath elements (2 per panel)
    assert svg.count("clipPath") >= 6, f"Expected ≥6 clipPaths for 3 panels, got {svg.count('clipPath')}"


def test_encode_facet_row_sets_facet():
    """encode(facet_row=...) must set _facet."""
    df = pl.DataFrame({"x": [1], "y": [2], "grp": ["a"]})
    c = Chart(df).mark_point().encode(x="x", y="y", facet_row="grp:N")
    assert c._facet is not None
    assert c._facet.mode_kind == "wrap"
    assert c._facet.field == "grp"


def test_encode_facet_bare_sets_facet():
    """encode(facet=...) must set _facet."""
    df = pl.DataFrame({"x": [1], "y": [2], "grp": ["a"]})
    c = Chart(df).mark_point().encode(x="x", y="y", facet="grp:N")
    assert c._facet is not None
    assert c._facet.mode_kind == "wrap"
    assert c._facet.field == "grp"


def test_encode_facet_col_and_row_grid_mode():
    """encode(facet_col=..., facet_row=...) must produce grid mode."""
    df = pl.DataFrame({"x": [1], "y": [2], "col": ["c1"], "row": ["r1"]})
    c = Chart(df).mark_point().encode(x="x", y="y", facet_col="col:N", facet_row="row:N")
    assert c._facet is not None
    assert c._facet.mode_kind == "grid"
    assert c._facet.col == "col"
    assert c._facet.row == "row"


def test_encode_facet_preserves_ncols_from_prior_facet_call():
    """encode(facet_col=...) applied after .facet(ncols=N) must preserve ncols."""
    df = pl.DataFrame({"x": [1], "y": [2], "grp": ["a"]})
    # This is an unusual pattern but should not lose ncols.
    c = (
        Chart(df).mark_point().encode(x="x", y="y")
        .facet(col="grp", ncols=3)
        .encode(facet_col="grp:N")
    )
    assert c._facet is not None
    assert c._facet.ncols == 3


def test_faceted_svg_contains_strip_titles():
    """Phase 7 strip-title implementation emits text elements for each facet.

    Note: The Rust facet renderer requires Utf8 (not LargeUtf8) string columns.
    Polars serializes strings as LargeUtf8 in Arrow; we use a pyarrow table with
    explicit pa.string() (Utf8) as the data source to satisfy Phase 7's limitation.
    """
    import pyarrow as pa
    from ferrum._core import render_svg

    tbl = pa.table({
        "a": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
        "b": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
        "species": pa.array(["A", "A", "B", "B", "C", "C"], type=pa.string()),
    })
    # Build spec via Chart but pass the pyarrow table directly to render_svg
    df = pl.from_arrow(tbl)
    chart = Chart(df).mark_point().encode(x="a", y="b").facet(col="species")
    spec = chart.to_spec()
    svg = render_svg(spec, tbl, viewport=(600.0, 400.0), theme={})
    # Phase 7 strip-title implementation emits text elements for 3 facets
    text_count = svg.count("<text")
    # Strip titles + axis labels: at least 3 strip title text elements
    assert text_count >= 3, f"Expected ≥3 <text elements, got {text_count}"
