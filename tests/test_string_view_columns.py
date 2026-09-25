"""Render-level pins: Utf8View / LargeUtf8 string columns render byte-identical to Utf8.

`normalize_string_views` (crates/ferrum-core/src/render/prepare/mod.rs) casts
Arrow `Utf8View` and `LargeUtf8` string columns to `Utf8` before scale
resolution, so a categorical string column arriving in either of Arrow's
newer string encodings must render exactly like the same values carried as
plain `Utf8` (#167(1)). These are render-level pins (SVG bytes, interactive
scene JSON) proving that contract end to end, not just at the Rust-fixture
level.
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa
import pytest

import ferrum as fm

CAT = ["alpha", "beta", "alpha", "gamma", "beta", "alpha"]
X = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
Y = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]

_HAS_STRING_VIEW = hasattr(pa, "string_view")


def _table(cat_type: "pa.DataType") -> pa.Table:
    return pa.table(
        {
            "cat": pa.array(CAT, type=cat_type),
            "x": pa.array(X, type=pa.float64()),
            "y": pa.array(Y, type=pa.float64()),
        }
    )


UTF8 = _table(pa.string())
LARGE_UTF8 = _table(pa.large_string())

_skip_no_string_view = pytest.mark.skipif(
    not _HAS_STRING_VIEW, reason="installed pyarrow has no pa.string_view()"
)
VIEW_TABLES = [
    ("large_utf8", LARGE_UTF8),
    pytest.param(
        "utf8_view",
        _table(pa.string_view()) if _HAS_STRING_VIEW else None,
        marks=_skip_no_string_view,
    ),
]


def test_schema_sanity():
    assert pa.types.is_string(UTF8.schema.field("cat").type)
    assert pa.types.is_large_string(LARGE_UTF8.schema.field("cat").type)
    if _HAS_STRING_VIEW:
        assert pa.types.is_string_view(_table(pa.string_view()).schema.field("cat").type)


@pytest.mark.parametrize("label,tbl", VIEW_TABLES)
def test_bar_mark_svg_byte_identical_across_string_encodings(label, tbl):
    base = fm.Chart(UTF8).mark_bar().encode(x="cat:N", y="mean(x):Q").to_svg()
    svg = fm.Chart(tbl).mark_bar().encode(x="cat:N", y="mean(x):Q").to_svg()
    assert svg == base, f"{label} bar-mark SVG diverged from Utf8 baseline"


@pytest.mark.parametrize("label,tbl", VIEW_TABLES)
def test_color_legend_svg_byte_identical_across_string_encodings(label, tbl):
    base = fm.Chart(UTF8).mark_point().encode(x="x:Q", y="y:Q", color="cat:N").to_svg()
    svg = fm.Chart(tbl).mark_point().encode(x="x:Q", y="y:Q", color="cat:N").to_svg()
    assert svg == base, f"{label} color-legend SVG diverged from Utf8 baseline"


@pytest.mark.parametrize("label,tbl", VIEW_TABLES)
def test_bar_mark_interactive_scene_json_byte_identical_across_string_encodings(label, tbl):
    base = fm.Chart(UTF8).mark_bar().encode(x="cat:N", y="mean(x):Q").interactive().scene_json
    scene = fm.Chart(tbl).mark_bar().encode(x="cat:N", y="mean(x):Q").interactive().scene_json
    assert scene == base, f"{label} bar-mark scene_json diverged from Utf8 baseline"


@pytest.mark.parametrize("label,tbl", VIEW_TABLES)
def test_color_legend_interactive_scene_json_byte_identical_across_string_encodings(label, tbl):
    base = (
        fm.Chart(UTF8).mark_point().encode(x="x:Q", y="y:Q", color="cat:N").interactive().scene_json
    )
    scene = (
        fm.Chart(tbl).mark_point().encode(x="x:Q", y="y:Q", color="cat:N").interactive().scene_json
    )
    assert scene == base, f"{label} color-legend scene_json diverged from Utf8 baseline"


@pytest.mark.skipif(not _HAS_STRING_VIEW, reason="installed pyarrow has no pa.string_view()")
def test_polars_exported_string_view_table_renders_byte_identical():
    """polars ``to_arrow(compat_level=CompatLevel.newest())`` exports Utf8View;
    feed that table straight into Chart to reproduce a real polars source."""
    df = pl.DataFrame({"cat": CAT, "x": X, "y": Y})
    view_tbl = df.to_arrow(compat_level=pl.CompatLevel.newest())
    assert pa.types.is_string_view(view_tbl.schema.field("cat").type)

    base = fm.Chart(UTF8).mark_bar().encode(x="cat:N", y="mean(x):Q").to_svg()
    svg = fm.Chart(view_tbl).mark_bar().encode(x="cat:N", y="mean(x):Q").to_svg()
    assert svg == base


def test_transform_materialized_large_utf8_output_renders_byte_identical():
    """``transform_flatten`` on a polars ``LargeList(LargeUtf8)`` column
    materializes a ``LargeUtf8`` output column (F-L05-05) that never passed
    through the raw-input normalization pass — it must still render
    identically to a chart built directly from the equivalent, already
    flattened, plain-``Utf8`` table."""
    df = pl.DataFrame(
        {
            "person": ["Alice", "Bob"],
            "tags": [["python", "rust"], ["go", "python", "java"]],
        }
    )
    flattened_chart = (
        fm.Chart(df)
        .transform(fm.transform_flatten(["tags"]))
        .mark_bar()
        .encode(x="tags:N", y="count():Q")
    )
    flattened_svg = flattened_chart.to_svg()

    manual_flat = pa.table(
        {
            "person": ["Alice", "Alice", "Bob", "Bob", "Bob"],
            "tags": pa.array(["python", "rust", "go", "python", "java"], type=pa.string()),
        }
    )
    manual_svg = fm.Chart(manual_flat).mark_bar().encode(x="tags:N", y="count():Q").to_svg()

    assert flattened_svg == manual_svg


def test_changing_one_string_value_changes_the_svg():
    """Discriminating control: the byte-equality assertions above are not
    vacuously true because the column got dropped or blanked — a single
    changed category value must change the rendered SVG."""
    base = fm.Chart(UTF8).mark_bar().encode(x="cat:N", y="mean(x):Q").to_svg()

    changed_cat = list(CAT)
    changed_cat[0] = "delta"
    changed_tbl = pa.table(
        {
            "cat": pa.array(changed_cat, type=pa.string()),
            "x": pa.array(X, type=pa.float64()),
            "y": pa.array(Y, type=pa.float64()),
        }
    )
    changed_svg = fm.Chart(changed_tbl).mark_bar().encode(x="cat:N", y="mean(x):Q").to_svg()

    assert changed_svg != base
