"""Regression tests for defect D8 — ``mark_line().encode(detail=)`` must split
lines by group without a color legend.

Spec (§4): ``mark_line().encode(x=, y=, detail="g")`` renders one connected
polyline per distinct value of ``g``, introducing no color legend.  With per-axis
layout this enables hand-built parallel coordinates; with a single line per group
it enables multi-series lines that are not color-encoded.

## What works today (baseline GREEN)

The Rust line batcher in ``crates/ferrum-core/src/render/marks/line.rs`` already
groups by ``mark_style.detail`` when the detail column has **Utf8 dtype** (string).
The Python ``_apply_channel_aliases`` function routes ``encode(detail=...)`` into
``mk["detail"]`` which becomes ``mark_style.detail`` in the Rust spec.  So these
tests pass today and protect against regression:

- ``detail="g:N"`` with a string column produces N separate polylines.
- No color legend is emitted; group labels do not appear as SVG text.
- Interleaved rows (groups mixed in row order) still produce separate polylines.
- Parallel-coordinates layout (ordinal x + detail) produces one polyline per
  sample.
- ``color`` + ``detail`` combined produces one polyline per (color, detail) pair
  and a color legend only for the color channel.

## What is broken today (TDD RED)

The detail column grouping in ``line.rs`` uses ``col_as_str`` to read the detail
field from the Arrow batch.  ``col_as_str`` (``render/arrow_cast.rs`` line 121)
accepts only Utf8/StringArray and returns ``Err(UnsupportedDtype)`` for any other
dtype.  When the call fails, ``detail_values`` is ``None`` and the grouping falls
through to the "otherwise" branch, which produces a single polyline over all rows.

Affected dtypes: ``Int64``, ``Int32``, ``UInt64``, ``Bool`` — any non-Utf8 column.

Root cause: ``crates/ferrum-core/src/render/marks/line.rs``, the ``detail_values``
binding (around the ``col_as_str(ctx.batch, f)`` call, ~line 118 in context).
Fix: replace ``col_as_str`` with ``col_as_ordinal_category_str`` for the detail
field so that integer and boolean group-IDs are stringified the same way the
ordinal domain builder handles them.

The parallel-coordinates ``desugar_parallel_coordinates`` always uses a
``sample_id`` column that is cast to Utf8 (``_parallel_coords_chart_from_dataframe``
line ~758), so the parallel-coordinates helper is not affected.  The bug surfaces
only when a user passes a non-string detail column directly.

## Test status summary

Tests prefixed ``test_detail_string_*`` are GREEN today (baseline).
Tests prefixed ``test_detail_integer_*`` and ``test_detail_bool_*`` are RED today.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import Detail


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def _polyline_count(svg: str) -> int:
    """Return the number of ``<polyline>`` elements in the SVG."""
    return len(re.findall(r"<polyline", svg))


def _all_text_nodes(svg: str) -> list[str]:
    """Return the text content of every ``<text>`` element in the SVG."""
    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


def _has_legend_label(svg: str, label: str) -> bool:
    """Return True if ``label`` appears as an SVG text node (legend entry)."""
    return label in _all_text_nodes(svg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def three_group_str_df() -> pl.DataFrame:
    """Three-group line dataset with a string detail column.

    Rows are interleaved in group order (a, b, c, a, b, c, …) rather than
    contiguous, so a single tangled polyline is visibly distinct from three
    separate group polylines.  Groups are widely separated on the y axis so
    each polyline traces a different horizontal band.
    """
    xs = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    ys = [0.0, 100.0, 200.0, 1.0, 101.0, 201.0, 2.0, 102.0, 202.0]
    gs = ["a", "b", "c", "a", "b", "c", "a", "b", "c"]
    return pl.DataFrame({"x": xs, "y": ys, "g": gs})


@pytest.fixture()
def three_group_int_df() -> pl.DataFrame:
    """Three-group line dataset with an Int64 detail column.

    Same layout as ``three_group_str_df`` but ``g`` is stored as ``Int64``
    rather than ``Utf8``.  This is the dtype that triggers the bug.

    x values use 10/20/30 (not 0/1/2) so the axis tick labels "10", "20",
    "30" do not collide with the group integer labels "1", "2", "3" that the
    ``_has_legend_label`` helper searches for in all SVG text nodes.
    """
    xs = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0]
    ys = [0.0, 100.0, 200.0, 1.0, 101.0, 201.0, 2.0, 102.0, 202.0]
    gs = [1, 2, 3, 1, 2, 3, 1, 2, 3]  # Int64
    return pl.DataFrame({"x": xs, "y": ys, "g": gs})


@pytest.fixture()
def two_group_bool_df() -> pl.DataFrame:
    """Two-group line dataset with a Boolean detail column."""
    return pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            "flag": [True, True, True, False, False, False],
        }
    )


@pytest.fixture()
def parallel_coords_df() -> pl.DataFrame:
    """Long-form parallel coordinates dataset.

    Five samples each measured on three features.  Rows are in feature order
    within each sample, so the expected layout is 5 polylines connecting the
    three feature positions.
    """
    features = ["f1", "f2", "f3"]
    samples = [f"sample_{i}" for i in range(5)]
    rows = [
        {"feature": f, "value": float(i * 3 + j), "sample_id": s}
        for i, s in enumerate(samples)
        for j, f in enumerate(features)
    ]
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# D8-A (GREEN) — string detail column splits line into one polyline per group
# ---------------------------------------------------------------------------


def test_detail_string_splits_into_n_polylines(
    three_group_str_df: pl.DataFrame,
) -> None:
    """``encode(detail="g:N")`` with a Utf8 column and 3 groups must render 3
    separate polylines, not 1 tangled polyline spanning all rows.

    The rows are interleaved (a, b, c, a, b, c, …) so a single merged polyline
    would visibly zigzag across all three y bands.  With detail splitting each
    group draws its own monotone line.

    This test is GREEN today — baseline for regression.
    """
    svg = fm.Chart(three_group_str_df).mark_line().encode(x="x:Q", y="y:Q", detail="g:N").to_svg()
    n = _polyline_count(svg)
    assert n == 3, (
        "encode(detail='g:N') with 3 string groups must produce 3 polylines, "
        f"got {n}.  If this regresses from GREEN to RED, check the "
        "_apply_channel_aliases detail→mark_style wiring and the line.rs "
        "col_as_str branch."
    )


def test_detail_string_via_detail_class(
    three_group_str_df: pl.DataFrame,
) -> None:
    """``encode(detail=Detail('g'))`` is equivalent to ``encode(detail='g:N')``.

    Both spellings must produce the same number of polylines.  This covers the
    ``Detail(field)`` constructor path through ``ChannelBase.__init__``.

    This test is GREEN today — baseline for regression.
    """
    svg_shorthand = (
        fm.Chart(three_group_str_df).mark_line().encode(x="x:Q", y="y:Q", detail="g:N").to_svg()
    )
    svg_class = (
        fm.Chart(three_group_str_df)
        .mark_line()
        .encode(x="x:Q", y="y:Q", detail=Detail("g"))
        .to_svg()
    )
    n_shorthand = _polyline_count(svg_shorthand)
    n_class = _polyline_count(svg_class)
    assert n_shorthand == n_class == 3, (
        "Both detail='g:N' and detail=Detail('g') must produce 3 polylines.  "
        f"Shorthand: {n_shorthand}, Detail() class: {n_class}."
    )


def test_detail_string_no_color_legend(
    three_group_str_df: pl.DataFrame,
) -> None:
    """``encode(detail='g:N')`` must emit NO color legend.

    The detail channel groups lines without a visual variable.  Group labels
    must not appear as ``<text>`` nodes in the SVG (which is how legend entries
    are rendered).

    This test is GREEN today — baseline for regression.
    """
    svg = fm.Chart(three_group_str_df).mark_line().encode(x="x:Q", y="y:Q", detail="g:N").to_svg()
    for label in ("a", "b", "c"):
        assert not _has_legend_label(svg, label), (
            f"Group label '{label}' appeared as an SVG text node.  "
            "detail encoding must not produce a color legend.  "
            "Check whether the detail channel is being routed through the "
            "color scale or emitting legend entries."
        )


def test_detail_string_color_legend_only_for_color_channel(
    three_group_str_df: pl.DataFrame,
) -> None:
    """When both ``color`` and ``detail`` are set, the legend must contain only
    the color channel's group labels, not the detail channel's labels.

    Dataset: ``g`` drives color (2 values: "a"/"b"); a separate column
    ``sample_id`` drives detail (4 values: "s1"–"s4").  The legend must show
    "a" and "b" but not "s1"–"s4".

    This test is GREEN today — baseline for regression.
    """
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0] * 4,
            "y": list(range(12)),
            "class_": ["A"] * 6 + ["B"] * 6,
            "sample_id": ["s1", "s1", "s1", "s2", "s2", "s2", "s3", "s3", "s3", "s4", "s4", "s4"],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_line()
        .encode(x="x:Q", y="y:Q", color="class_:N", detail="sample_id:N")
        .to_svg()
    )
    # Color legend entries must appear.
    assert _has_legend_label(svg, "A"), (
        "Color legend entry 'A' missing from SVG.  color encoding must produce "
        "a legend even when detail is also set."
    )
    assert _has_legend_label(svg, "B"), (
        "Color legend entry 'B' missing from SVG.  color encoding must produce "
        "a legend even when detail is also set."
    )
    # Detail labels must NOT appear as legend entries.
    for label in ("s1", "s2", "s3", "s4"):
        assert not _has_legend_label(svg, label), (
            f"Detail group '{label}' appeared as an SVG text node.  "
            "detail encoding must not add entries to any legend."
        )
    # Should produce 4 polylines (one per sample_id).
    n = _polyline_count(svg)
    assert n == 4, f"color+detail with 4 samples must produce 4 polylines, got {n}."


def test_detail_string_parallel_coordinates_style(
    parallel_coords_df: pl.DataFrame,
) -> None:
    """Parallel-coordinates layout: ordinal x (feature names) + detail (sample_id).

    5 samples × 3 features: each sample becomes one polyline spanning the three
    feature positions on the ordinal x axis.  This is the "hand-built parallel
    coordinates" use case described in the spec.

    This test is GREEN today — baseline for regression.
    """
    svg = (
        fm.Chart(parallel_coords_df)
        .mark_line()
        .encode(x="feature:N", y="value:Q", detail="sample_id:N")
        .to_svg()
    )
    n = _polyline_count(svg)
    assert n == 5, (
        "Parallel-coordinates layout with 5 samples must produce 5 polylines, "
        f"got {n}.  Each sample should trace its own line across the feature "
        "axis positions."
    )
    # Sample IDs must not appear as legend labels.
    for i in range(5):
        assert not _has_legend_label(svg, f"sample_{i}"), (
            f"Sample 'sample_{i}' appeared as an SVG text node.  "
            "detail encoding must not produce a legend."
        )


# ---------------------------------------------------------------------------
# D8-B (RED) — integer detail column silently falls back to one polyline
# ---------------------------------------------------------------------------


def test_detail_integer_splits_into_n_polylines(
    three_group_int_df: pl.DataFrame,
) -> None:
    """``encode(detail='g:N')`` with an Int64 column must render 3 separate
    polylines, one per distinct integer group value.

    Root cause: ``crates/ferrum-core/src/render/marks/line.rs`` reads the
    detail column via ``col_as_str``, which accepts only Utf8/StringArray and
    returns ``Err(UnsupportedDtype)`` for Int64.  When the call fails,
    ``detail_values`` is ``None`` and the grouping falls through to the
    "otherwise" branch, producing a single polyline connecting all rows in
    batch order.

    The fix is to replace ``col_as_str`` with ``col_as_ordinal_category_str``
    for the detail field (the latter stringifies integers using the same
    formatting as the domain builder, so group membership is consistent).

    TODAY: ``_polyline_count(svg) == 1`` (tangled).
    AFTER FIX: ``_polyline_count(svg) == 3``.
    """
    svg = fm.Chart(three_group_int_df).mark_line().encode(x="x:Q", y="y:Q", detail="g:N").to_svg()
    n = _polyline_count(svg)
    assert n == 3, (
        f"encode(detail='g:N') with an Int64 column and 3 groups must produce "
        f"3 polylines, got {n}.  "
        "Root cause: col_as_str in line.rs rejects Int64 columns; "
        "replace with col_as_ordinal_category_str.  "
        "File: crates/ferrum-core/src/render/marks/line.rs (detail_values binding)."
    )


def test_detail_integer_no_color_legend(
    three_group_int_df: pl.DataFrame,
) -> None:
    """Int64 detail grouping must not produce a color legend, just as string
    detail does not.

    This test fails today because the detail field is ignored (col_as_str fails
    silently), but the assertion still captures the no-legend requirement so the
    test can be used to verify both the polyline count AND the legend absence
    once the fix lands.

    TODAY: only 1 polyline is produced (detail is a no-op for Int64).
    AFTER FIX: 3 polylines, no legend labels for group values 1/2/3.
    """
    svg = fm.Chart(three_group_int_df).mark_line().encode(x="x:Q", y="y:Q", detail="g:N").to_svg()
    n = _polyline_count(svg)
    assert n == 3, (
        f"Int64 detail column must split into 3 polylines, got {n}.  "
        "col_as_str rejects Int64; use col_as_ordinal_category_str."
    )
    # After the fix, group integer labels must not appear as legend text.
    for label in ("1", "2", "3"):
        assert not _has_legend_label(svg, label), (
            f"Integer group label '{label}' appeared as an SVG text node.  "
            "detail encoding must never produce a color legend."
        )


def test_detail_integer_interleaved_rows_still_groups_correctly(
    three_group_int_df: pl.DataFrame,
) -> None:
    """Interleaved row order must not affect Int64 detail grouping.

    The fixture rows alternate between groups (1, 2, 3, 1, 2, 3, …).  A correct
    implementation collects all rows belonging to each group value into a single
    polyline, regardless of their position in the batch.

    TODAY: 1 polyline (groups not split; all rows connected in batch order).
    AFTER FIX: 3 separate polylines, each monotone on x.

    This is the "visibly distinct from a tangled polyline" check: the tangled
    single polyline contains 9 points that zigzag across all three y bands,
    whereas 3 correct polylines each contain 3 collinear points.
    """
    svg_with_detail = (
        fm.Chart(three_group_int_df).mark_line().encode(x="x:Q", y="y:Q", detail="g:N").to_svg()
    )
    svg_no_detail = fm.Chart(three_group_int_df).mark_line().encode(x="x:Q", y="y:Q").to_svg()
    n_detail = _polyline_count(svg_with_detail)
    n_no_detail = _polyline_count(svg_no_detail)
    # Without detail: exactly 1 tangled polyline.
    assert n_no_detail == 1, f"Baseline without detail must produce 1 polyline, got {n_no_detail}."
    # With detail: 3 separate polylines.
    assert n_detail == 3, (
        f"encode(detail='g:N') with interleaved Int64 rows must produce 3 "
        f"polylines (one per group), got {n_detail}.  Today this equals the "
        f"no-detail baseline ({n_no_detail}), confirming the detail field is "
        "silently discarded for Int64 columns."
    )


# ---------------------------------------------------------------------------
# D8-C (RED) — boolean detail column silently falls back to one polyline
# ---------------------------------------------------------------------------


def test_detail_bool_splits_into_two_polylines(
    two_group_bool_df: pl.DataFrame,
) -> None:
    """``encode(detail='flag:N')`` with a Boolean column must render 2 separate
    polylines (one for True, one for False).

    The same root cause as the Int64 case: ``col_as_str`` in ``line.rs`` only
    handles Utf8/StringArray.  Boolean columns are rejected with
    ``Err(UnsupportedDtype)`` and the detail field is silently treated as absent.

    TODAY: 1 polyline (two groups merged).
    AFTER FIX: 2 polylines (True group, False group).
    """
    svg = fm.Chart(two_group_bool_df).mark_line().encode(x="x:Q", y="y:Q", detail="flag:N").to_svg()
    n = _polyline_count(svg)
    assert n == 2, (
        f"encode(detail='flag:N') with a Boolean column must produce 2 polylines, "
        f"got {n}.  col_as_str in line.rs rejects Boolean dtype; use "
        "col_as_ordinal_category_str which stringifies booleans as 'true'/'false'."
    )
