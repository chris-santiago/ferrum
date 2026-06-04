"""Verify interaction between explicit scale domain overrides (CoordCartesian) and position adjustments.

Tests that:
1. Explicit domain overrides don't crash when combined with position adjustments
2. Position adjustments still produce different SVG with constrained domains
3. Clipped domains (narrower than data extent) don't cause rendering crashes
"""

from __future__ import annotations

import pytest
import polars as pl

import ferrum as fm
from ferrum.position import Stack, Dodge, Jitter
from ferrum.coord import CoordCartesian


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


@pytest.fixture()
def boxplot_df() -> pl.DataFrame:
    """DataFrame with enough data points per group for boxplots (5+ per group)."""
    return pl.DataFrame(
        {
            "cat": ["a"] * 8 + ["b"] * 8 + ["c"] * 8,
            "val": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                15.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                16.0,
                0.5,
                1.5,
                2.5,
                3.5,
                4.5,
                5.5,
                6.5,
                14.0,
            ],
            "grp": (["x", "y"] * 4) * 3,
        }
    )


@pytest.fixture()
def numeric_df() -> pl.DataFrame:
    """DataFrame with numeric columns for histogram tests."""
    return pl.DataFrame(
        {
            "val": [
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
                5.0,
                5.5,
                6.0,
                6.5,
                7.0,
                7.5,
                8.0,
                8.5,
                9.0,
                9.5,
                10.0,
                10.5,
            ],
            "grp": ["x", "y"] * 10,
        }
    )


# ---------------------------------------------------------------------------
# Group 1: Stack + Domain Override (no crash, produces SVG)
# ---------------------------------------------------------------------------

_STACK_DOMAIN_CASES = [
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        CoordCartesian(ylim=(0, 20)),
        id="stack-bar-ylim-above-max",
    ),
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        CoordCartesian(ylim=(0, 5)),
        id="stack-bar-ylim-clips-data",
    ),
    pytest.param(
        "mark_area",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        CoordCartesian(ylim=(0, 15)),
        id="stack-area-ylim-constrained",
    ),
]


@pytest.mark.parametrize("mark_method,encoding,coord", _STACK_DOMAIN_CASES)
def test_stack_with_domain_override_no_crash(
    grouped_df: pl.DataFrame, mark_method: str, encoding: dict, coord: CoordCartesian
) -> None:
    """Stack + explicit domain override must not crash and must produce valid SVG."""
    base = fm.Chart(grouped_df)
    mark_fn = getattr(base, mark_method)
    svg = mark_fn(position=Stack()).encode(**encoding).coord(coord).to_svg()
    assert "<svg" in svg, f"{mark_method} + Stack + {coord} did not produce valid SVG"


def test_stack_histogram_with_ylim(numeric_df: pl.DataFrame) -> None:
    """mark_histogram + Stack + ylim must not crash."""
    svg = (
        fm.Chart(numeric_df)
        .mark_histogram(position=Stack())
        .encode(x="val:Q", color="grp:N")
        .coord(CoordCartesian(ylim=(0, 10)))
        .to_svg()
    )
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# Group 2: Dodge + Domain Override (no crash)
# ---------------------------------------------------------------------------

_DODGE_DOMAIN_CASES = [
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        CoordCartesian(ylim=(0, 10)),
        id="dodge-bar-ylim",
    ),
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        CoordCartesian(ylim=(0, 3)),
        id="dodge-bar-ylim-clips",
    ),
    pytest.param(
        "mark_point",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        CoordCartesian(ylim=(0, 8)),
        id="dodge-point-ylim",
    ),
]


@pytest.mark.parametrize("mark_method,encoding,coord", _DODGE_DOMAIN_CASES)
def test_dodge_with_domain_override_no_crash(
    grouped_df: pl.DataFrame, mark_method: str, encoding: dict, coord: CoordCartesian
) -> None:
    """Dodge + explicit domain override must not crash and must produce valid SVG."""
    base = fm.Chart(grouped_df)
    mark_fn = getattr(base, mark_method)
    svg = mark_fn(position=Dodge()).encode(**encoding).coord(coord).to_svg()
    assert "<svg" in svg, f"{mark_method} + Dodge + {coord} did not produce valid SVG"


def test_dodge_boxplot_with_ylim(boxplot_df: pl.DataFrame) -> None:
    """mark_boxplot + Dodge + ylim must not crash."""
    svg = (
        fm.Chart(boxplot_df)
        .mark_boxplot(position=Dodge())
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(CoordCartesian(ylim=(0, 12)))
        .to_svg()
    )
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# Group 3: Jitter + Domain Override (no crash)
# ---------------------------------------------------------------------------

_JITTER_DOMAIN_CASES = [
    pytest.param(
        "mark_point",
        {"x": "num:Q", "y": "val:Q"},
        CoordCartesian(ylim=(0, 8)),
        Jitter(seed=42),
        id="jitter-point-ylim",
    ),
    pytest.param(
        "mark_point",
        {"x": "num:Q", "y": "val:Q"},
        CoordCartesian(ylim=(2, 5)),
        Jitter(axis="x", seed=42),
        id="jitter-point-axis-x-ylim",
    ),
    pytest.param(
        "mark_tick",
        {"x": "num:Q", "y": "val:Q"},
        CoordCartesian(ylim=(0, 8)),
        Jitter(axis="x", seed=42),
        id="jitter-tick-axis-x-ylim",
    ),
    pytest.param(
        "mark_point",
        {"x": "num:Q", "y": "val:Q"},
        CoordCartesian(ylim=(0, 10)),
        Jitter(axis="y", seed=42),
        id="jitter-point-axis-y-ylim",
    ),
    pytest.param(
        "mark_point",
        {"x": "num:Q", "y": "val:Q"},
        CoordCartesian(ylim=(0, 10)),
        Jitter(axis="both", seed=42),
        id="jitter-point-axis-both-ylim",
    ),
]


@pytest.mark.parametrize("mark_method,encoding,coord,jitter", _JITTER_DOMAIN_CASES)
def test_jitter_with_domain_override_no_crash(
    grouped_df: pl.DataFrame,
    mark_method: str,
    encoding: dict,
    coord: CoordCartesian,
    jitter: Jitter,
) -> None:
    """Jitter + explicit domain override must not crash and must produce valid SVG."""
    base = fm.Chart(grouped_df)
    mark_fn = getattr(base, mark_method)
    svg = mark_fn(position=jitter).encode(**encoding).coord(coord).to_svg()
    assert "<svg" in svg, f"{mark_method} + {jitter} + {coord} did not produce valid SVG"


# ---------------------------------------------------------------------------
# Group 4: Position changes SVG even with domain override
# ---------------------------------------------------------------------------


def test_stack_changes_svg_with_ylim(grouped_df: pl.DataFrame) -> None:
    """mark_bar with Stack vs no-position should produce different SVG even with ylim."""
    coord = CoordCartesian(ylim=(0, 15))
    svg_without = (
        fm.Chart(grouped_df)
        .mark_bar()
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(coord)
        .to_svg()
    )
    svg_with = (
        fm.Chart(grouped_df)
        .mark_bar(position=Stack())
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(coord)
        .to_svg()
    )
    assert svg_without != svg_with, "Stack() did not change SVG output when ylim is set"


def test_dodge_changes_svg_with_ylim(grouped_df: pl.DataFrame) -> None:
    """mark_bar with Dodge vs no-position should produce different SVG even with ylim."""
    coord = CoordCartesian(ylim=(0, 15))
    svg_without = (
        fm.Chart(grouped_df)
        .mark_bar()
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(coord)
        .to_svg()
    )
    svg_with = (
        fm.Chart(grouped_df)
        .mark_bar(position=Dodge())
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(coord)
        .to_svg()
    )
    assert svg_without != svg_with, "Dodge() did not change SVG output when ylim is set"


def test_jitter_changes_svg_with_ylim(grouped_df: pl.DataFrame) -> None:
    """mark_point with Jitter vs no-position should produce different SVG even with ylim."""
    coord = CoordCartesian(ylim=(0, 8))
    svg_without = (
        fm.Chart(grouped_df).mark_point().encode(x="num:Q", y="val:Q").coord(coord).to_svg()
    )
    svg_with = (
        fm.Chart(grouped_df)
        .mark_point(position=Jitter(seed=42))
        .encode(x="num:Q", y="val:Q")
        .coord(coord)
        .to_svg()
    )
    assert svg_without != svg_with, "Jitter() did not change SVG output when ylim is set"


# ---------------------------------------------------------------------------
# Group 5: Domain wider than data extent + position
# ---------------------------------------------------------------------------

_WIDE_DOMAIN_CASES = [
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        Stack(),
        CoordCartesian(ylim=(0, 1000)),
        id="stack-bar-wide-domain",
    ),
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        Dodge(),
        CoordCartesian(ylim=(0, 1000)),
        id="dodge-bar-wide-domain",
    ),
    pytest.param(
        "mark_point",
        {"x": "num:Q", "y": "val:Q"},
        Jitter(seed=42),
        CoordCartesian(ylim=(0, 1000)),
        id="jitter-point-wide-domain",
    ),
]


@pytest.mark.parametrize("mark_method,encoding,position,coord", _WIDE_DOMAIN_CASES)
def test_wide_domain_with_position_no_crash(
    grouped_df: pl.DataFrame, mark_method: str, encoding: dict, position, coord: CoordCartesian
) -> None:
    """Very wide domain (much larger than data) + position must not crash."""
    base = fm.Chart(grouped_df)
    mark_fn = getattr(base, mark_method)
    svg = mark_fn(position=position).encode(**encoding).coord(coord).to_svg()
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# Group 6: Edge cases
# ---------------------------------------------------------------------------


def test_negative_ylim_with_stacked_positive_data(grouped_df: pl.DataFrame) -> None:
    """Negative ylim with positive stacked data must not crash."""
    svg = (
        fm.Chart(grouped_df)
        .mark_bar(position=Stack())
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(CoordCartesian(ylim=(-5, 5)))
        .to_svg()
    )
    assert "<svg" in svg


def test_very_wide_domain_with_stack(grouped_df: pl.DataFrame) -> None:
    """ylim extending far beyond data with Stack must not crash."""
    svg = (
        fm.Chart(grouped_df)
        .mark_bar(position=Stack())
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(CoordCartesian(ylim=(0, 10000)))
        .to_svg()
    )
    assert "<svg" in svg


def test_both_xlim_and_ylim_with_stack(grouped_df: pl.DataFrame) -> None:
    """Both xlim and ylim set simultaneously with Stack must not crash."""
    svg = (
        fm.Chart(grouped_df)
        .mark_bar(position=Stack())
        .encode(x="num:Q", y="val:Q", color="grp:N")
        .coord(CoordCartesian(xlim=(0, 10), ylim=(0, 15)))
        .to_svg()
    )
    assert "<svg" in svg


def test_both_xlim_and_ylim_with_dodge(grouped_df: pl.DataFrame) -> None:
    """Both xlim and ylim set simultaneously with Dodge must not crash."""
    svg = (
        fm.Chart(grouped_df)
        .mark_bar(position=Dodge())
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(CoordCartesian(xlim=(0, 10), ylim=(0, 8)))
        .to_svg()
    )
    assert "<svg" in svg


def test_zero_width_ylim_with_position(grouped_df: pl.DataFrame) -> None:
    """Very narrow ylim domain with Stack must not crash (near-zero visible area)."""
    svg = (
        fm.Chart(grouped_df)
        .mark_bar(position=Stack())
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(CoordCartesian(ylim=(3, 3.5)))
        .to_svg()
    )
    assert "<svg" in svg


def test_inverted_ylim_with_stack(grouped_df: pl.DataFrame) -> None:
    """Inverted ylim (max < min) with Stack must not crash (or raise cleanly)."""
    try:
        svg = (
            fm.Chart(grouped_df)
            .mark_bar(position=Stack())
            .encode(x="cat:N", y="val:Q", color="grp:N")
            .coord(CoordCartesian(ylim=(10, 0)))
            .to_svg()
        )
        # If it doesn't crash, it should still produce valid SVG
        assert "<svg" in svg
    except (ValueError, RuntimeError):
        # Acceptable: the library rejects inverted domains
        pass


def test_stack_normalize_with_ylim(grouped_df: pl.DataFrame) -> None:
    """Stack(offset='normalize') + ylim must not crash."""
    svg = (
        fm.Chart(grouped_df)
        .mark_bar(position=Stack(offset="normalize"))
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(CoordCartesian(ylim=(0, 1.5)))
        .to_svg()
    )
    assert "<svg" in svg


def test_stack_center_with_ylim(grouped_df: pl.DataFrame) -> None:
    """Stack(offset='center') + ylim must not crash."""
    svg = (
        fm.Chart(grouped_df)
        .mark_bar(position=Stack(offset="center"))
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(CoordCartesian(ylim=(-10, 10)))
        .to_svg()
    )
    assert "<svg" in svg


def test_dodge_with_expand_false(grouped_df: pl.DataFrame) -> None:
    """Dodge + CoordCartesian(expand=False) must not crash."""
    svg = (
        fm.Chart(grouped_df)
        .mark_bar(position=Dodge())
        .encode(x="cat:N", y="val:Q", color="grp:N")
        .coord(CoordCartesian(ylim=(0, 8), expand=False))
        .to_svg()
    )
    assert "<svg" in svg


def test_jitter_with_clip_false(grouped_df: pl.DataFrame) -> None:
    """Jitter + CoordCartesian(clip=False) must not crash."""
    svg = (
        fm.Chart(grouped_df)
        .mark_point(position=Jitter(seed=42))
        .encode(x="num:Q", y="val:Q")
        .coord(CoordCartesian(ylim=(2, 5), clip=False))
        .to_svg()
    )
    assert "<svg" in svg
