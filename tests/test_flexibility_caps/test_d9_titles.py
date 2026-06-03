"""Regression tests for defect D9 — transform-derived internal column names leak
into axis/legend titles.

The bug: when a composite mark (``mark_contour``, ``mark_hex``) desugars into
layers that bind internal transform-output columns (``contour_x``, ``contour_y``,
``hex_x``, ``hex_y``) as encoding fields, those internal names surface as the
rendered axis title text — overwriting the user-declared source field name.

Example: ``Chart(df).mark_contour().encode(x='temp', y='depth')`` renders with
x-axis title ``"contour_x"`` instead of ``"temp"``.

Root cause (Python/Rust):
  ``src/ferrum/marks/heavy_stat.py`` — ``desugar_contour()`` line ~112 and
  ``desugar_hex()`` line ~635 set ``encoding={"x": "<internal_col>", ...}``
  without a ``title=`` override on the encoding spec.

  ``crates/ferrum-core/src/render/prepare.rs`` lines 433-448 resolves axis titles
  in order: spec-level encoding title → layer-0 encoding title → field name.
  Because neither spec-level nor layer-level titles are set for contour/hex
  layers, the fallback ``e.field.clone()`` returns the internal column name.

  The fix is to set ``title=<source_field>`` on the layer ``x``/``y`` encoding
  for composite marks (analogous to what ``desugar_errorband`` already does with
  ``Y("lower", title=y_field)``).

All tests in this file are RED today and will become GREEN once the fix lands.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import X, Y


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def _axis_title_texts(svg: str) -> list[str]:
    """Return all text strings that appear as axis titles in the SVG.

    Ferrum renders axis titles as plain ``<text>`` elements.  This helper
    extracts every ``<text>`` node's content and returns the subset that
    is plausibly a title: non-numeric, non-empty, and not a tick label.
    Numeric strings (tick labels like "0.5", "-3") are excluded so the
    returned list contains only candidate axis/legend title strings.
    """
    raw = re.findall(r"<text[^>]*>([^<]+)</text>", svg)
    titles = []
    for t in raw:
        t = t.strip()
        if not t:
            continue
        # Exclude pure numeric strings (tick labels).
        try:
            float(t)
            continue
        except ValueError:
            pass
        titles.append(t)
    return titles


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bivariate_df() -> pl.DataFrame:
    """100-row bivariate dataset with distinctly named fields.

    Field names ``temp`` and ``depth`` are chosen so any leakage of internal
    column names (``contour_x``, ``contour_y``, ``hex_x``, ``hex_y``) is
    immediately visible: if a title says ``contour_x`` it cannot be confused
    with ``temp``.
    """
    import random

    rng = random.Random(42)
    n = 100
    return pl.DataFrame(
        {
            "temp": [rng.gauss(20.0, 5.0) for _ in range(n)],
            "depth": [rng.gauss(100.0, 30.0) for _ in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# D9-A — mark_contour: source field names must appear as axis titles
# ---------------------------------------------------------------------------


def test_contour_x_axis_title_is_source_field(bivariate_df: pl.DataFrame) -> None:
    """mark_contour().encode(x='temp') must render x-axis title 'temp', not 'contour_x'.

    Today ``desugar_contour`` sets ``encoding={"x": "contour_x", ...}`` on the
    polygon layer without a ``title=`` override.  Because neither the spec-level
    nor the layer-level encoding title is set, ``prepare.rs`` falls through to
    ``e.field.clone()`` and emits ``"contour_x"`` as the axis title text.

    Root cause: ``src/ferrum/marks/heavy_stat.py`` ``desugar_contour()``
    ~line 112 — no ``title=x_field`` on the x encoding spec.
    Fix: use ``X("contour_x", title=x_field)`` (or equivalent) in the layer
    encoding, as ``desugar_errorband`` does for ``Y("lower", title=y_field)``.
    """
    svg = fm.Chart(bivariate_df).mark_contour(fill=True).encode(x="temp", y="depth").show_svg()
    titles = _axis_title_texts(svg)

    assert "contour_x" not in titles, (
        "mark_contour is leaking the internal column name 'contour_x' as the "
        "x-axis title.  The title should be the source field 'temp'.  "
        f"All title-like text found in SVG: {titles}"
    )
    assert "temp" in titles, (
        "Expected x-axis title 'temp' (the source field) but it was not found.  "
        f"All title-like text found in SVG: {titles}"
    )


def test_contour_y_axis_title_is_source_field(bivariate_df: pl.DataFrame) -> None:
    """mark_contour().encode(y='depth') must render y-axis title 'depth', not 'contour_y'.

    Today ``desugar_contour`` sets ``encoding={"y": "contour_y", ...}`` on the
    polygon layer without a ``title=`` override, causing the internal column
    name ``"contour_y"`` to appear as the y-axis title.

    Root cause: same as x — ``desugar_contour()`` ~line 112 in
    ``src/ferrum/marks/heavy_stat.py``.
    """
    svg = fm.Chart(bivariate_df).mark_contour(fill=True).encode(x="temp", y="depth").show_svg()
    titles = _axis_title_texts(svg)

    assert "contour_y" not in titles, (
        "mark_contour is leaking the internal column name 'contour_y' as the "
        "y-axis title.  The title should be the source field 'depth'.  "
        f"All title-like text found in SVG: {titles}"
    )
    assert "depth" in titles, (
        "Expected y-axis title 'depth' (the source field) but it was not found.  "
        f"All title-like text found in SVG: {titles}"
    )


def test_contour_isoline_x_axis_title_is_source_field(bivariate_df: pl.DataFrame) -> None:
    """mark_contour(fill=False) isoline mode must not leak 'contour_x' as x-axis title.

    The isoline desugaring path (``fill=False``) sets
    ``encoding={"x": "contour_x", "y": "contour_y", ...}`` on the segment
    layer — the same internal leak as isoband mode.
    """
    svg = fm.Chart(bivariate_df).mark_contour(fill=False).encode(x="temp", y="depth").show_svg()
    titles = _axis_title_texts(svg)

    assert "contour_x" not in titles, (
        "mark_contour(fill=False) is leaking 'contour_x' as x-axis title. "
        f"All title-like text: {titles}"
    )
    assert "temp" in titles, (
        f"Expected x-axis title 'temp' but it was absent. All title-like text: {titles}"
    )


# ---------------------------------------------------------------------------
# D9-B — mark_hex: source field names must appear as axis titles
# ---------------------------------------------------------------------------


def test_hex_x_axis_title_is_source_field(bivariate_df: pl.DataFrame) -> None:
    """mark_hex().encode(x='temp') must render x-axis title 'temp', not 'hex_x'.

    Today ``desugar_hex`` sets ``encoding={"x": "hex_x", "y": "hex_y", ...}``
    on the polygon layer without a ``title=`` override, causing ``"hex_x"``
    to appear as the x-axis title.

    Root cause: ``src/ferrum/marks/heavy_stat.py`` ``desugar_hex()``
    ~line 635 — no ``title=x_field`` on the x encoding spec.
    Fix: use ``X("hex_x", title=x_field)`` in the layer encoding.
    """
    svg = fm.Chart(bivariate_df).mark_hex(bin_size=5.0).encode(x="temp", y="depth").show_svg()
    titles = _axis_title_texts(svg)

    assert "hex_x" not in titles, (
        "mark_hex is leaking the internal column name 'hex_x' as the "
        "x-axis title.  The title should be the source field 'temp'.  "
        f"All title-like text found in SVG: {titles}"
    )
    assert "temp" in titles, (
        "Expected x-axis title 'temp' (the source field) but it was not found.  "
        f"All title-like text found in SVG: {titles}"
    )


def test_hex_y_axis_title_is_source_field(bivariate_df: pl.DataFrame) -> None:
    """mark_hex().encode(y='depth') must render y-axis title 'depth', not 'hex_y'.

    Today ``desugar_hex`` sets ``encoding={"y": "hex_y", ...}`` on the polygon
    layer without a ``title=`` override, causing ``"hex_y"`` to appear as the
    y-axis title.

    Root cause: same as x — ``desugar_hex()`` ~line 635 in
    ``src/ferrum/marks/heavy_stat.py``.
    """
    svg = fm.Chart(bivariate_df).mark_hex(bin_size=5.0).encode(x="temp", y="depth").show_svg()
    titles = _axis_title_texts(svg)

    assert "hex_y" not in titles, (
        "mark_hex is leaking the internal column name 'hex_y' as the "
        "y-axis title.  The title should be the source field 'depth'.  "
        f"All title-like text found in SVG: {titles}"
    )
    assert "depth" in titles, (
        "Expected y-axis title 'depth' (the source field) but it was not found.  "
        f"All title-like text found in SVG: {titles}"
    )


# ---------------------------------------------------------------------------
# D9-C — Explicit title= must always win (guard tests)
# ---------------------------------------------------------------------------


def test_contour_explicit_axis_title_overrides_source_field(
    bivariate_df: pl.DataFrame,
) -> None:
    """An explicit title= on the encoding channel must appear as the axis title,
    regardless of what the transform's internal column name is.

    This guard confirms that once the D9 fix is in place, explicit user titles
    still take precedence over both the source-field default and any internal
    column names.

    Uses ``encode(x=X('temp', title='Temperature (°C)'))`` — the explicit title
    should be the rendered x-axis title.
    """
    svg = (
        fm.Chart(bivariate_df)
        .mark_contour(fill=True)
        .encode(x=X("temp", title="Temperature (°C)"), y="depth")
        .show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "contour_x" not in titles, (
        "mark_contour is leaking 'contour_x' even when an explicit title is set. "
        f"All title-like text: {titles}"
    )
    assert "Temperature (°C)" in titles, (
        "Expected explicit title 'Temperature (°C)' to appear as the x-axis title "
        f"but it was not found.  All title-like text: {titles}"
    )


def test_hex_explicit_axis_title_overrides_source_field(
    bivariate_df: pl.DataFrame,
) -> None:
    """An explicit title= on the encoding channel must appear as the axis title
    for mark_hex, overriding both the source-field default and 'hex_x'/'hex_y'.

    Uses ``encode(y=Y('depth', title='Ocean Depth (m)'))`` — the explicit title
    should be the rendered y-axis title.
    """
    svg = (
        fm.Chart(bivariate_df)
        .mark_hex(bin_size=5.0)
        .encode(x="temp", y=Y("depth", title="Ocean Depth (m)"))
        .show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "hex_y" not in titles, (
        "mark_hex is leaking 'hex_y' even when an explicit title is set. "
        f"All title-like text: {titles}"
    )
    assert "Ocean Depth (m)" in titles, (
        "Expected explicit title 'Ocean Depth (m)' to appear as the y-axis title "
        f"but it was not found.  All title-like text: {titles}"
    )
