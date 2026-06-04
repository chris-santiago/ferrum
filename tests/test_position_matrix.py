"""Verify that position adjustments (Stack, Dodge, Jitter) produce different SVG output.

Each test renders a chart without a position adjustment, then with one, and asserts
the two SVGs differ -- proving that the position logic actually modified mark coordinates.
"""

from __future__ import annotations

import pytest
import polars as pl

import ferrum as fm
from ferrum.position import Stack, Dodge, Jitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def grouped_df() -> pl.DataFrame:
    """DataFrame with categorical x, quantitative y, and a grouping column."""
    return pl.DataFrame(
        {
            "cat": ["a", "a", "b", "b", "c", "c"],
            "val": [3.0, 5.0, 2.0, 4.0, 6.0, 1.0],
            "grp": ["x", "y", "x", "y", "x", "y"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


# ---------------------------------------------------------------------------
# Parametrized test: Stack
# ---------------------------------------------------------------------------

_STACK_CASES = [
    pytest.param("mark_bar", {"x": "cat:N", "y": "val:Q", "color": "grp:N"}, id="stack-bar"),
    pytest.param("mark_area", {"x": "cat:N", "y": "val:Q", "color": "grp:N"}, id="stack-area"),
]


@pytest.mark.parametrize("mark_method,encoding", _STACK_CASES)
def test_stack_changes_svg(grouped_df: pl.DataFrame, mark_method: str, encoding: dict) -> None:
    """Applying Stack() to an eligible mark must alter the rendered SVG."""
    base = fm.Chart(grouped_df)
    mark_fn = getattr(base, mark_method)

    svg_without = mark_fn().encode(**encoding).to_svg()
    svg_with = mark_fn(position=Stack()).encode(**encoding).to_svg()

    assert svg_without != svg_with, (
        f"{mark_method} + Stack() did not change SVG output; "
        "position adjustment may not be wired through to the renderer."
    )


# ---------------------------------------------------------------------------
# Parametrized test: Dodge
# ---------------------------------------------------------------------------

_DODGE_CASES = [
    pytest.param("mark_bar", {"x": "cat:N", "y": "val:Q", "color": "grp:N"}, id="dodge-bar"),
    pytest.param("mark_point", {"x": "cat:N", "y": "val:Q", "color": "grp:N"}, id="dodge-point"),
]


@pytest.mark.parametrize("mark_method,encoding", _DODGE_CASES)
def test_dodge_changes_svg(grouped_df: pl.DataFrame, mark_method: str, encoding: dict) -> None:
    """Applying Dodge() to an eligible mark must alter the rendered SVG."""
    base = fm.Chart(grouped_df)
    mark_fn = getattr(base, mark_method)

    svg_without = mark_fn().encode(**encoding).to_svg()
    svg_with = mark_fn(position=Dodge()).encode(**encoding).to_svg()

    assert svg_without != svg_with, (
        f"{mark_method} + Dodge() did not change SVG output; "
        "position adjustment may not be wired through to the renderer."
    )


# ---------------------------------------------------------------------------
# Parametrized test: Jitter
# ---------------------------------------------------------------------------

_JITTER_CASES = [
    pytest.param("mark_point", {"x": "num:Q", "y": "val:Q"}, id="jitter-point"),
    pytest.param("mark_tick", {"x": "num:Q", "y": "val:Q"}, id="jitter-tick"),
]


@pytest.mark.parametrize("mark_method,encoding", _JITTER_CASES)
def test_jitter_changes_svg(grouped_df: pl.DataFrame, mark_method: str, encoding: dict) -> None:
    """Applying Jitter() to an eligible mark must alter the rendered SVG (no grouping needed)."""
    base = fm.Chart(grouped_df)
    mark_fn = getattr(base, mark_method)

    svg_without = mark_fn().encode(**encoding).to_svg()
    svg_with = mark_fn(position=Jitter(seed=42)).encode(**encoding).to_svg()

    assert svg_without != svg_with, (
        f"{mark_method} + Jitter() did not change SVG output; "
        "position adjustment may not be wired through to the renderer."
    )


# ---------------------------------------------------------------------------
# Stack offset variants
# ---------------------------------------------------------------------------

_STACK_OFFSETS = [
    pytest.param("zero", id="offset-zero"),
    pytest.param("normalize", id="offset-normalize"),
    pytest.param("center", id="offset-center"),
]


@pytest.mark.parametrize("offset", _STACK_OFFSETS)
def test_stack_offset_differs_from_no_position(grouped_df: pl.DataFrame, offset: str) -> None:
    """Each Stack offset variant must produce SVG different from unpositioned bars."""
    svg_without = (
        fm.Chart(grouped_df).mark_bar().encode(x="cat:N", y="val:Q", color="grp:N").to_svg()
    )
    svg_with = (
        fm.Chart(grouped_df)
        .mark_bar(position=Stack(offset=offset))
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .to_svg()
    )

    assert svg_without != svg_with, (
        f"Stack(offset={offset!r}) did not change SVG output vs. no position."
    )


@pytest.mark.parametrize(
    "offset_a,offset_b",
    [
        pytest.param("zero", "normalize", id="zero-vs-normalize"),
        pytest.param("zero", "center", id="zero-vs-center"),
        pytest.param("normalize", "center", id="normalize-vs-center"),
    ],
)
def test_stack_offsets_differ_from_each_other(
    grouped_df: pl.DataFrame, offset_a: str, offset_b: str
) -> None:
    """Different Stack offsets must produce different SVGs from each other."""
    svg_a = (
        fm.Chart(grouped_df)
        .mark_bar(position=Stack(offset=offset_a))
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .to_svg()
    )
    svg_b = (
        fm.Chart(grouped_df)
        .mark_bar(position=Stack(offset=offset_b))
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .to_svg()
    )

    assert svg_a != svg_b, (
        f"Stack(offset={offset_a!r}) and Stack(offset={offset_b!r}) "
        "produced identical SVG — offsets may not be wired."
    )


# ---------------------------------------------------------------------------
# Jitter axis variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param("x", id="jitter-axis-x"),
        pytest.param("y", id="jitter-axis-y"),
        pytest.param("both", id="jitter-axis-both"),
    ],
)
def test_jitter_axis_changes_svg(grouped_df: pl.DataFrame, axis: str) -> None:
    """Each Jitter axis setting must alter SVG vs. unpositioned marks."""
    svg_without = fm.Chart(grouped_df).mark_point().encode(x="num:Q", y="val:Q").to_svg()
    svg_with = (
        fm.Chart(grouped_df)
        .mark_point(position=Jitter(axis=axis, seed=42))
        .encode(x="num:Q", y="val:Q")
        .to_svg()
    )

    assert svg_without != svg_with, f"Jitter(axis={axis!r}) did not change SVG output."


# ---------------------------------------------------------------------------
# Dodge padding variant
# ---------------------------------------------------------------------------


def test_dodge_padding_changes_svg(grouped_df: pl.DataFrame) -> None:
    """Different Dodge padding values must produce different SVGs."""
    svg_narrow = (
        fm.Chart(grouped_df)
        .mark_bar(position=Dodge(padding=0.0))
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .to_svg()
    )
    svg_wide = (
        fm.Chart(grouped_df)
        .mark_bar(position=Dodge(padding=0.5))
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .to_svg()
    )

    assert svg_narrow != svg_wide, (
        "Dodge(padding=0.0) and Dodge(padding=0.5) produced identical SVG; "
        "padding may not be wired through to the renderer."
    )
