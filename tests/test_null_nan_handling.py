"""Test that all primitive marks gracefully handle null and NaN values.

Marks should skip rows with null/NaN in position or channel columns without
crashing, producing broken SVG, or panicking.  The key assertion in every test
is NO CRASH — plus a structural check that the result is a well-formed SVG
document (starts with ``<svg`` and ends with ``</svg>``).
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import ferrum as fm

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

df_with_nulls = pl.DataFrame(
    {
        "x": [1.0, None, 3.0, None, 5.0],
        "y": [2.0, 4.0, None, 5.0, None],
        "cat": ["a", "b", None, "b", "a"],
        "x2": [2.0, None, 4.0, None, 6.0],
        "y2": [3.0, 5.0, None, 6.0, None],
        "cat2": ["x", None, "x", "y", None],
    }
)

df_with_nans = pl.DataFrame(
    {
        "x": [1.0, float("nan"), 3.0, float("nan"), 5.0],
        "y": [2.0, 4.0, float("nan"), 5.0, float("nan")],
        "x2": [2.0, float("nan"), 4.0, float("nan"), 6.0],
        "y2": [3.0, 5.0, float("nan"), 6.0, float("nan")],
    }
)

df_all_null = pl.DataFrame(
    {
        "x": [None, None, None, None, None],
        "y": [None, None, None, None, None],
    }
).cast({"x": pl.Float64, "y": pl.Float64})

# ---------------------------------------------------------------------------
# Mark configurations — maps mark name to encoding kwargs and which datasets
# it is compatible with.
# ---------------------------------------------------------------------------

MARK_CONFIGS = {
    "mark_point": {"x": "x:Q", "y": "y:Q"},
    "mark_line": {"x": "x:Q", "y": "y:Q"},
    "mark_bar": {"x": "cat:N", "y": "y:Q"},
    "mark_area": {"x": "x:Q", "y": "y:Q"},
    "mark_rule": {"x": "x:Q", "y": "y:Q"},
    "mark_text": {"x": "x:Q", "y": "y:Q"},
    "mark_tick": {"x": "x:Q", "y": "y:Q"},
    "mark_rect": {"x": "cat:N", "y": "cat2:N"},
    "mark_segment": {"x": "x:Q", "y": "y:Q", "x2": "x2:Q", "y2": "y2:Q"},
}

# Marks that use categorical x (need df_with_nulls which has "cat"/"cat2")
_CAT_MARKS = {"mark_bar", "mark_rect"}

# Marks that need x2/y2 columns
_SEGMENT_MARKS = {"mark_segment"}

ALL_MARKS = list(MARK_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_chart(mark_name: str, df: pl.DataFrame) -> object:
    """Build and return a Chart with the given mark and encoding configuration."""
    chart = fm.Chart(df)
    mark_method = getattr(chart, mark_name)
    chart = mark_method()
    enc_kwargs = MARK_CONFIGS[mark_name]
    chart = chart.encode(**enc_kwargs)
    return chart


def _assert_valid_svg(svg: str, context: str) -> None:
    """Assert the SVG output is well-formed (non-empty, proper start/end tags)."""
    assert isinstance(svg, str), f"[{context}] expected str, got {type(svg)}"
    assert "<svg" in svg, f"[{context}] expected SVG output to contain '<svg'"
    assert svg.rstrip().endswith("</svg>"), f"[{context}] expected SVG output to end with '</svg>'"


# ---------------------------------------------------------------------------
# Tests: Null in x column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mark_name", ALL_MARKS, ids=ALL_MARKS)
def test_null_x_no_crash(mark_name: str) -> None:
    """Marks render without crashing when x column contains nulls."""
    # All marks can use df_with_nulls (it has both cat and numeric columns)
    chart = _build_chart(mark_name, df_with_nulls)
    svg = chart.show_svg()
    _assert_valid_svg(svg, f"null_x/{mark_name}")


# ---------------------------------------------------------------------------
# Tests: Null in y column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mark_name", ALL_MARKS, ids=ALL_MARKS)
def test_null_y_no_crash(mark_name: str) -> None:
    """Marks render without crashing when y column contains nulls."""
    chart = _build_chart(mark_name, df_with_nulls)
    svg = chart.show_svg()
    _assert_valid_svg(svg, f"null_y/{mark_name}")


# ---------------------------------------------------------------------------
# Tests: NaN in x column
# ---------------------------------------------------------------------------

# NaN tests exclude categorical marks (bar, rect) because their x encoding
# uses nominal columns; NaN is only meaningful for float columns.
_NAN_MARKS = [m for m in ALL_MARKS if m not in _CAT_MARKS]


@pytest.mark.parametrize("mark_name", _NAN_MARKS, ids=_NAN_MARKS)
def test_nan_x_no_crash(mark_name: str) -> None:
    """Marks render without crashing when x column contains NaN."""
    chart = _build_chart(mark_name, df_with_nans)
    svg = chart.show_svg()
    _assert_valid_svg(svg, f"nan_x/{mark_name}")


# ---------------------------------------------------------------------------
# Tests: NaN in y column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mark_name", _NAN_MARKS, ids=_NAN_MARKS)
def test_nan_y_no_crash(mark_name: str) -> None:
    """Marks render without crashing when y column contains NaN."""
    chart = _build_chart(mark_name, df_with_nans)
    svg = chart.show_svg()
    _assert_valid_svg(svg, f"nan_y/{mark_name}")


# ---------------------------------------------------------------------------
# Tests: All nulls (entire column is null)
# ---------------------------------------------------------------------------

# For all-null data, only marks that can work with plain Q×Q encodings are
# tested.  Categorical marks and segment need columns that don't exist in
# df_all_null.
_ALL_NULL_MARKS = [m for m in ALL_MARKS if m not in _CAT_MARKS and m not in _SEGMENT_MARKS]


@pytest.mark.parametrize("mark_name", _ALL_NULL_MARKS, ids=_ALL_NULL_MARKS)
def test_all_null_no_crash(mark_name: str) -> None:
    """Marks render without crashing when all values are null (empty chart)."""
    enc_kwargs = {"x": "x:Q", "y": "y:Q"}
    chart = fm.Chart(df_all_null)
    mark_method = getattr(chart, mark_name)
    chart = mark_method()
    chart = chart.encode(**enc_kwargs)
    svg = chart.show_svg()
    _assert_valid_svg(svg, f"all_null/{mark_name}")


# ---------------------------------------------------------------------------
# Tests: Null in channel columns (opacity, color)
# ---------------------------------------------------------------------------

# Use df_with_nulls which has "cat" (with a null) as color and "y" (with
# nulls) as an opacity channel.
_CHANNEL_MARKS = [
    ("mark_point", {"x": "x:Q", "y": "y:Q", "color": "cat:N"}),
    ("mark_point", {"x": "x:Q", "y": "y:Q", "opacity": "y:Q"}),
    ("mark_line", {"x": "x:Q", "y": "y:Q", "color": "cat:N"}),
    ("mark_bar", {"x": "cat:N", "y": "y:Q", "color": "cat:N"}),
    ("mark_area", {"x": "x:Q", "y": "y:Q", "color": "cat:N"}),
    ("mark_tick", {"x": "x:Q", "y": "y:Q", "color": "cat:N"}),
    ("mark_rule", {"x": "x:Q", "y": "y:Q", "color": "cat:N"}),
    ("mark_text", {"x": "x:Q", "y": "y:Q", "color": "cat:N"}),
    ("mark_rect", {"x": "cat:N", "y": "cat2:N", "color": "y:Q"}),
]

_CHANNEL_IDS = [f"{mark}/{list(enc.keys())[-1]}_null" for mark, enc in _CHANNEL_MARKS]


@pytest.mark.parametrize("mark_name,enc_kwargs", _CHANNEL_MARKS, ids=_CHANNEL_IDS)
def test_null_in_channel_no_crash(mark_name: str, enc_kwargs: dict) -> None:
    """Marks render without crashing when a non-position channel has nulls."""
    chart = fm.Chart(df_with_nulls)
    mark_method = getattr(chart, mark_name)
    chart = mark_method()
    chart = chart.encode(**enc_kwargs)
    svg = chart.show_svg()
    _assert_valid_svg(svg, f"null_channel/{mark_name}")
