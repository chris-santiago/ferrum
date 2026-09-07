"""Tests for Chart.facet() fluent method."""

import polars as pl

import ferrum as fm
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

    tbl = pa.table(
        {
            "x": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
            "y": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
            "species": pa.array(["A", "A", "B", "B", "C", "C"], type=pa.string()),
        }
    )
    df = pl.from_arrow(tbl)
    chart = Chart(df).mark_point().encode(x="x", y="y", facet_col="species:N")
    spec = chart.to_spec()
    svg = render_svg(spec, tbl, viewport=(600.0, 400.0), theme={})
    # 3 panels → ≥6 clipPath elements (2 per panel)
    assert svg.count("clipPath") >= 6, (
        f"Expected ≥6 clipPaths for 3 panels, got {svg.count('clipPath')}"
    )


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
        Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .facet(col="grp", ncols=3)
        .encode(facet_col="grp:N")
    )
    assert c._facet is not None
    assert c._facet.ncols == 3


def test_facet_col_no_ncols_infers_side_by_side() -> None:
    """facet(col=X) with no explicit ncols must emit ncols == n_distinct(X).

    KG-6 / issue #24: col-only facets without an explicit ncols default to a
    single horizontal row of panels (side-by-side), matching Altair and seaborn.
    The serialized facet dict's ``mode.ncols`` must equal the number of distinct
    values in the column field.

    Variants:
    - col= only (no ncols): ncols == n_distinct → side-by-side
    - row= only (no ncols): ncols == 1 → vertical stack
    - field= (no ncols): ncols == 1 → vertical stack
    - col= with explicit ncols=2: ncols == 2 regardless of n_distinct
    """
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1, 2, 3, 4, 5, 6],
            "g": ["a", "b", "c", "a", "b", "c"],  # 3 distinct values
        }
    )

    # col= only → ncols must equal n_distinct(g) = 3
    chart_col = Chart(df).mark_point().encode(x="x", y="y").facet(col="g")
    facet_dict_col = chart_col._build_facet_dict()
    assert facet_dict_col["mode"]["ncols"] == 3, (
        f"facet(col='g') with n_distinct=3 must emit ncols=3 (side-by-side), "
        f"got ncols={facet_dict_col['mode']['ncols']}"
    )

    # row= only → ncols must be 1 (vertical stack)
    chart_row = Chart(df).mark_point().encode(x="x", y="y").facet(row="g")
    facet_dict_row = chart_row._build_facet_dict()
    assert facet_dict_row["mode"]["ncols"] == 1, (
        f"facet(row='g') must emit ncols=1 (vertical stack), "
        f"got ncols={facet_dict_row['mode']['ncols']}"
    )

    # field= → ncols must be 1 (vertical stack)
    chart_field = Chart(df).mark_point().encode(x="x", y="y").facet(field="g")
    facet_dict_field = chart_field._build_facet_dict()
    assert facet_dict_field["mode"]["ncols"] == 1, (
        f"facet(field='g') must emit ncols=1 (vertical stack), "
        f"got ncols={facet_dict_field['mode']['ncols']}"
    )

    # col= with explicit ncols=2 → ncols must be 2 (explicit wins)
    chart_explicit = Chart(df).mark_point().encode(x="x", y="y").facet(col="g", ncols=2)
    facet_dict_explicit = chart_explicit._build_facet_dict()
    assert facet_dict_explicit["mode"]["ncols"] == 2, (
        f"facet(col='g', ncols=2) must emit ncols=2 (explicit overrides default), "
        f"got ncols={facet_dict_explicit['mode']['ncols']}"
    )


def test_faceted_svg_contains_strip_titles():
    """Phase 7 strip-title implementation emits text elements for each facet.

    Note: The Rust facet renderer requires Utf8 (not LargeUtf8) string columns.
    Polars serializes strings as LargeUtf8 in Arrow; we use a pyarrow table with
    explicit pa.string() (Utf8) as the data source to satisfy Phase 7's limitation.

    Extended for F-L01-04: a facet ``title=`` must reach this strip's text
    (``"{title}: {value}"``), and leaving ``title`` unset must render
    byte-identically to the pre-fix output (no silent behavior change).
    """
    import pyarrow as pa
    from ferrum._core import render_svg

    tbl = pa.table(
        {
            "a": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
            "b": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
            "species": pa.array(["A", "A", "B", "B", "C", "C"], type=pa.string()),
        }
    )
    # Build spec via Chart but pass the pyarrow table directly to render_svg
    df = pl.from_arrow(tbl)
    chart = Chart(df).mark_point().encode(x="a", y="b").facet(col="species")
    spec = chart.to_spec()
    svg = render_svg(spec, tbl, viewport=(600.0, 400.0), theme={})
    # Phase 7 strip-title implementation emits text elements for 3 facets
    text_count = svg.count("<text")
    # Strip titles + axis labels: at least 3 strip title text elements
    assert text_count >= 3, f"Expected ≥3 <text elements, got {text_count}"

    # F-L01-04: title= unset -> byte-identical to the untouched call above.
    chart_no_title = Chart(df).mark_point().encode(x="a", y="b").facet(col="species", title=None)
    svg_no_title = render_svg(chart_no_title.to_spec(), tbl, viewport=(600.0, 400.0), theme={})
    assert svg_no_title == svg, "facet(title=None) must be byte-identical to omitting title"

    # F-L01-04: title= set -> the strip renders "{title}: {value}".
    chart_titled = Chart(df).mark_point().encode(x="a", y="b").facet(col="species", title="Group")
    svg_titled = render_svg(chart_titled.to_spec(), tbl, viewport=(600.0, 400.0), theme={})
    assert "Group: A" in svg_titled
    assert "Group: A" not in svg


def test_facet_channel_title_reaches_strip():
    """F-L01-04 audit repro (findings/L01.md): Facet(field, title=...) must
    reach the rendered strip, not be silently dropped.

    Pinned against the exact repro in the finding text (fields x/y/cat,
    values a/b/c, title="My Facet"): the titled chart's SVG must contain
    "My Facet: a", the untitled chart's SVG must be byte-identical to a
    chart built without ever touching ``title=``, and the JSON facet dict
    must carry the title (previously: absent whether or not title was set).
    """
    import json

    df = pl.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "y": [4, 3, 2, 5, 1, 6], "cat": ["a", "b", "a", "b", "c", "c"]}
    )

    titled = (
        Chart(df).mark_point().encode(x="x:Q", y="y:Q", facet=fm.Facet("cat:N", title="My Facet"))
    )
    untitled = Chart(df).mark_point().encode(x="x:Q", y="y:Q", facet=fm.Facet("cat:N"))
    untitled_shorthand = Chart(df).mark_point().encode(x="x:Q", y="y:Q", facet="cat:N")

    assert "My Facet: a" in titled.to_svg()
    assert "My Facet" not in untitled.to_svg()
    assert untitled.to_svg() == untitled_shorthand.to_svg(), (
        "no-title facets must stay byte-identical across equivalent constructions"
    )

    titled_dict = json.loads(titled.to_json())["facet"]
    untitled_dict = json.loads(untitled.to_json())["facet"]
    assert titled_dict.get("title") == "My Facet"
    assert "title" not in untitled_dict


def test_facet_grid_row_and_col_titles_are_independent():
    """F-L01-04: in grid mode (both facet_row and facet_col set), the column
    strip's title and the row strip's title route to independent FacetSpec
    keys (``title`` / ``row_title``) and neither leaks into the other."""
    import json

    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [4, 3, 2, 1],
            "row": ["r1", "r1", "r2", "r2"],
            "col": ["c1", "c2", "c1", "c2"],
        }
    )
    chart = (
        Chart(df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            facet_col=fm.FacetCol("col:N", title="Column Title"),
            facet_row=fm.FacetRow("row:N", title="Row Title"),
        )
    )
    facet_dict = json.loads(chart.to_json())["facet"]
    assert facet_dict["title"] == "Column Title"
    assert facet_dict["row_title"] == "Row Title"

    # Grid is the one mode where row_title is actually live -- both strips
    # must independently reach the rendered SVG, not just the wire dict.
    svg = chart.to_svg()
    assert "Column Title: c1" in svg
    assert "Row Title: r1" in svg


def test_facet_row_only_title_routes_to_wrap_strip():
    """CRITICAL routing rule (T8 review flag): FacetRow used ALONE lowers to
    WRAP mode, where the column/wrap strip renders and reads ``title`` — NOT
    ``row_title`` (which only has effect in grid mode). A FacetRow(title=...)
    used alone must therefore route to ``title`` so it actually reaches the
    strip, rather than silently landing in an inert ``row_title``."""
    import json

    df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "grp": ["r1", "r2"]})
    chart = (
        Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", facet_row=fm.FacetRow("grp:N", title="Row Only"))
    )
    facet_dict = json.loads(chart.to_json())["facet"]
    assert facet_dict["mode"]["kind"] == "wrap"
    assert facet_dict.get("title") == "Row Only", (
        "wrap-row facet's title must land in the wire field the wrap strip "
        "actually reads, not silently drop into an inert row_title"
    )
    assert "row_title" not in facet_dict

    # And it must actually reach the rendered strip, not just the wire dict.
    svg = chart.to_svg()
    assert "Row Only: r1" in svg


def test_chart_facet_method_title_kwarg():
    """F-L01-04: Chart.facet(title=...) sets the primary/wrap dimension's
    title -- parity with the encode(facet=...) channel path."""
    import json

    df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "grp": ["g1", "g2"]})

    wrap_chart = Chart(df).mark_point().encode(x="x", y="y").facet(field="grp", title="Group")
    assert wrap_chart._facet.title == "Group"
    assert json.loads(wrap_chart.to_json())["facet"]["title"] == "Group"

    row_only_chart = Chart(df).mark_point().encode(x="x", y="y").facet(row="grp", title="Group")
    assert row_only_chart._facet.mode_kind == "wrap"
    assert json.loads(row_only_chart.to_json())["facet"]["title"] == "Group"

    no_title_chart = Chart(df).mark_point().encode(x="x", y="y").facet(field="grp")
    assert "title" not in json.loads(no_title_chart.to_json())["facet"]
